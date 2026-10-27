"""Strategie P : reentrainement declenche par une derive confirmee, avec validation.

C'est la contribution du memoire. Les six etapes du cadrage (§6) se repartissent
entre les trois methodes du contrat :

1. **Surveillance** (`on_window`) : le moniteur compare la distribution recente
   des 152 features a celle de l'entrainement du modele en service, et suit le
   taux d'alertes. Voir `src/drift/`.
2. **Confirmation** (`on_window`) : le signal doit persister K fenetres
   consecutives. C'est ce qui distingue une derive d'une anomalie.
3. **Reentrainement** (`propose`) : challenger entraine sur les W dernieres
   fenetres, hors fenetre reservee V, sans exclusion par defaut.
4. **Validation** (`validate`) : un **garde-fou**, pas un concours.
5. **Promotion** (le harnais applique le verdict).
6. **Refroidissement** (`on_window`) : aucun reentrainement pendant C fenetres
   apres une promotion.

Deux points ou l'implementation refuse une facilite :

- **La validation est un garde-fou.** Sans etiquettes, aucun critere ne classe
  fiablement deux detecteurs — c'est mesure, et c'est la conclusion de Ma et al.
  (2023). On ne cherche donc pas a savoir si le challenger est *meilleur*, mais
  seulement s'il est *manifestement inutilisable* : historique trop court, taux
  d'alertes degenere, seuil degenere. Un challenger qui passe est promu, meme
  sans preuve qu'il soit meilleur. Pretendre choisir le meilleur des deux sans
  etiquettes serait une tricherie silencieuse.
- **La confirmation acquise pendant le refroidissement n'est pas perdue.** Elle
  est mise en attente et declenche des la fin du refroidissement. L'oublier
  ferait dependre le comportement de la strategie de l'instant exact ou la
  derive tombe par rapport a la derniere promotion, ce qui est un artefact
  d'implementation et non une politique.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.drift.monitor import DriftMonitor
from src.strategies.base import (
    Decision,
    TrainingPlan,
    ValidationStats,
    Verdict,
    WindowContext,
)

EXCLUSION_METHODS = ("none", "top_percentile", "champion_alerts")


class ProposedStrategy:
    """Reentrainement declenche par une derive confirmee, avec garde-fou de validation.

    `validation_enabled = False` donne l'ablation A1 : meme declenchement, meme
    choix de donnees, mais promotion inconditionnelle. C'est le seul ecart entre
    P et A1, ce qui rend l'effet de la validation isolable.
    """

    def __init__(
        self,
        drift_config: dict[str, Any],
        train_windows: int,
        reserved_windows: int,
        cooldown_windows: int,
        validation_enabled: bool = True,
        min_train_windows: int = 0,
        min_history: int = 300,
        alert_rate_bounds: tuple[float, float] = (0.001, 0.30),
        exclusion_method: str = "none",
        exclusion_percentile: float = 99.0,
        name: str = "proposed",
    ) -> None:
        if train_windows < 2:
            raise ValueError("train_windows doit valoir au moins 2.")
        if reserved_windows < 0 or reserved_windows >= train_windows:
            raise ValueError(
                f"reserved_windows ({reserved_windows}) doit etre dans [0, train_windows)"
                f" ; train_windows vaut {train_windows}."
            )
        if exclusion_method not in EXCLUSION_METHODS:
            raise ValueError(
                f"Methode d'exclusion inconnue : {exclusion_method!r}. "
                f"Attendu l'une de {EXCLUSION_METHODS}."
            )
        if validation_enabled and reserved_windows == 0:
            raise ValueError(
                "La validation exige une fenetre reservee : reserved_windows doit etre > 0 "
                "quand validation_enabled est vrai."
            )
        if min_train_windows > train_windows - reserved_windows:
            raise ValueError(
                f"min_train_windows ({min_train_windows}) depasse ce que W - V peut fournir "
                f"({train_windows - reserved_windows}) : aucun reentrainement ne serait "
                "jamais declenche."
            )

        self.name = name
        self.monitor = DriftMonitor(drift_config)
        self.train_windows = int(train_windows)
        self.reserved_windows = int(reserved_windows)
        self.cooldown_windows = int(cooldown_windows)
        self.validation_enabled = bool(validation_enabled)
        self.min_history = int(min_history)
        self.alert_rate_min, self.alert_rate_max = (
            float(alert_rate_bounds[0]),
            float(alert_rate_bounds[1]),
        )
        self.exclusion_method = exclusion_method
        self.exclusion_percentile = float(exclusion_percentile)

        # Plancher sur la taille du bloc d'entrainement du challenger. Il est
        # **structurel** et non optionnel : il s'applique de la meme facon a P
        # et a A1, de sorte que l'ablation n'isole que la validation. Le
        # maximum avec l'exigence du moniteur garantit qu'aucune configuration
        # ne peut produire un modele que le moniteur ne saurait plus surveiller.
        self.min_train_windows = max(
            int(min_train_windows), self.monitor.required_training_windows()
        )

        # Etat interne. `_alerts` et `_scores` ne servent qu'aux variantes
        # d'exclusion de l'ablation ; ils ne contiennent que ce que le modele en
        # service a produit, jamais une etiquette.
        self._fitted = False
        self._last_promotion: int | None = None
        self._pending: dict[str, Any] | None = None
        self._alerts: list[bool] = []
        self._scores: list[float] = []

    # ------------------------------------------------------------------ etape 1 et 2

    def on_window(self, context: WindowContext) -> Decision:
        """Surveille, confirme, et declenche hors refroidissement."""
        self._alerts.append(context.alert)
        self._scores.append(context.score)

        state = self.monitor.update(context.features, context.alert)
        events: list[str] = []
        # Un signal est journalise sur son **front montant** seulement : le
        # journaliser a chaque fenetre produirait des dizaines de milliers de
        # lignes et rendrait `n_signals` illisible.
        if state.streak == 1:
            events.append("drift_signal")
        if state.confirmed:
            events.append("drift_confirmed")
            self._pending = {
                "confirmed_at": context.position,
                "share": round(state.share, 4),
                "streak": state.streak,
            }

        if self._pending is None:
            return Decision(retrain=False, events=tuple(events), details=state.as_details())

        cooling = self._cooling_at(context.position)
        if cooling is not None:
            # Confirmation mise en attente : elle declenchera a la fin du
            # refroidissement, elle n'est pas perdue.
            return Decision(
                retrain=False,
                reason="refroidissement",
                events=tuple(events),
                details={**state.as_details(), "cooldown_remaining": cooling},
            )

        available = self.available_train_windows(context.position + 1)
        if available < self.min_train_windows:
            # La confirmation reste en attente, exactement comme pendant le
            # refroidissement : la derive est durable, elle declenchera des que
            # l'historique permet d'entrainer un modele surveillable. Sur les
            # scenarios S1 a S4, t_d est tire dans [40 %, 60 %] du flux : une
            # vraie derive n'est jamais concernee par ce report. Ne le sont que
            # les confirmations tres precoces, qui sont par construction des
            # fausses confirmations.
            return Decision(
                retrain=False,
                reason="historique_insuffisant",
                events=tuple(events),
                details={
                    **state.as_details(),
                    "n_train_available": available,
                    "min_train_windows": self.min_train_windows,
                },
            )

        pending = self._pending
        self._pending = None
        return Decision(
            retrain=True,
            reason="derive_confirmee",
            events=tuple(events),
            details={**state.as_details(), **pending},
        )

    def available_train_windows(self, n_history: int) -> int:
        """Taille du bloc d'entrainement qu'un reentrainement obtiendrait maintenant.

        Meme arithmetique que `propose`, sans construire les tableaux : c'est ce
        qui permet de refuser de declencher avant d'avoir paye le cout d'un
        entrainement voue au rejet.
        """
        block = min(int(n_history), self.train_windows)
        if self.reserved_windows == 0:
            return block
        return max(0, block - self.reserved_windows)

    def _cooling_at(self, position: int) -> int | None:
        """Fenetres de refroidissement restantes, ou None si le refroidissement est fini."""
        if self._last_promotion is None or self.cooldown_windows <= 0:
            return None
        elapsed = position - self._last_promotion
        if elapsed >= self.cooldown_windows:
            return None
        return int(self.cooldown_windows - elapsed)

    # ------------------------------------------------------------------ etape 3

    def propose(self, history: np.ndarray, position: int) -> TrainingPlan:
        """Les W dernieres fenetres, dont les V plus recentes sont reservees.

        La fenetre reservee est la plus **recente** : c'est celle qui decrit le
        regime dans lequel le challenger devra servir. La reserver coute W - V
        fenetres d'entrainement, et c'est le prix de la validation.
        """
        start = max(0, len(history) - self.train_windows)
        block = history[start:]
        if self.reserved_windows == 0:
            train, reserved = block, history[:0]
            split = len(block)
        else:
            split = max(1, len(block) - self.reserved_windows)
            train, reserved = block[:split], block[split:]

        kept, n_excluded = self._apply_exclusion(train, start)
        return TrainingPlan(
            train_X=kept,
            reserved_X=reserved,
            details={
                "train_start": int(start),
                "train_split": int(start + split),
                "train_stop": int(len(history)),
                "n_train_proposed": int(len(train)),
                "n_train_kept": int(len(kept)),
                "n_excluded": int(n_excluded),
                "exclusion_method": self.exclusion_method,
            },
        )

    def _apply_exclusion(self, train: np.ndarray, offset: int) -> tuple[np.ndarray, int]:
        """Exclusion des fenetres suspectes. `none` par defaut, et c'est un resultat.

        D13 : exclure les alertes du champion fait s'effondrer le challenger sous
        saturation, parce que sous derive le champion alerte sur presque tout —
        il ne reste alors plus de donnees representatives du nouveau regime.
        Les deux autres variantes restent implementees pour l'ablation.
        """
        if self.exclusion_method == "none" or len(train) == 0:
            return train, 0

        if self.exclusion_method == "champion_alerts":
            flags = np.asarray(self._alerts[offset : offset + len(train)], dtype=bool)
        else:
            scores = np.asarray(self._scores[offset : offset + len(train)], dtype=np.float64)
            if scores.size == 0:
                return train, 0
            cut = float(np.percentile(scores, self.exclusion_percentile))
            flags = scores > cut

        if flags.size != len(train):  # pragma: no cover - garde-fou de coherence
            raise RuntimeError(
                f"{flags.size} indicateurs pour {len(train)} fenetres d'entrainement."
            )
        kept = train[~flags]
        # Un filtre qui ne laisse rien n'est pas un filtre : on rend alors le
        # bloc complet, et le garde-fou de validation fera son travail.
        if len(kept) < 2:
            return train, 0
        return kept, int(flags.sum())

    # ------------------------------------------------------------------ etape 4

    def validate(self, stats: ValidationStats) -> Verdict:
        """Garde-fou : rejette l'inutilisable, ne cherche pas le meilleur.

        Trois motifs de rejet, tous calculables **sans etiquette** :

        - historique trop court : on ne sait pas encore ce qu'est le normal ;
        - taux d'alertes du challenger hors bornes sur la fenetre reservee : un
          modele qui n'alerte jamais, ou qui alerte sur tout, est inutilisable
          quelle que soit la verite terrain ;
        - seuil degenere : un seuil non fini n'est pas un seuil.
        """
        if not self.validation_enabled:
            return Verdict(promote=True, reason="validation_desactivee_a1")

        details = stats.as_details()

        if stats.n_history < self.min_history:
            return Verdict(
                promote=False,
                reason="historique_trop_court",
                details={**details, "min_history": self.min_history},
            )

        if stats.n_reserved_windows == 0:
            return Verdict(
                promote=False,
                reason="fenetre_reservee_vide",
                details=details,
            )

        if not np.isfinite(stats.challenger_threshold):
            return Verdict(promote=False, reason="seuil_degenere", details=details)

        rate = stats.challenger_alert_rate
        if not np.isfinite(rate) or rate < self.alert_rate_min or rate > self.alert_rate_max:
            return Verdict(
                promote=False,
                reason="taux_alertes_hors_bornes",
                details={
                    **details,
                    "alert_rate_min": self.alert_rate_min,
                    "alert_rate_max": self.alert_rate_max,
                },
            )

        return Verdict(promote=True, reason="garde_fou_passe", details=details)

    # ------------------------------------------------------------------ etapes 5 et 6

    def on_fit(self, training_X: np.ndarray, position: int) -> None:
        """Recale le moniteur sur l'entrainement du nouveau modele en service.

        Le refroidissement ne demarre qu'apres une **promotion** : le compter
        depuis l'entrainement initial interdirait tout reentrainement pendant
        les C premieres fenetres du flux, ce qui n'est pas ce que dit le §6.
        """
        self.monitor.fit_reference(training_X)
        self._pending = None
        if self._fitted:
            self._last_promotion = int(position)
        self._fitted = True
