"""Boucle de detection en ligne (jalon M5, phase B).

C'est le mode « en ligne » du §2 du cadrage : Prometheus est interroge a
intervalle regulier, chaque fenetre est transformee en features, scoree par le
champion charge depuis le registre, puis soumise a la strategie, qui peut
declencher un reentrainement, une validation et une promotion.

Ce que ce module **ne** fait pas, et c'est voulu :

- il ne reimplemente ni les features, ni le modele, ni les detecteurs de derive,
  ni les strategies. Il appelle `make_windows`, `IsolationForestDetector` et
  `run_retrain_cycle`, exactement comme `src/evaluation/replay.py`. Une boucle
  qui recalculerait ses features autrement rendrait la demonstration du
  chapitre 5 incomparable aux chiffres du chapitre 6, sans qu'aucun test ne le
  signale ;
- il ne decide rien lui-meme. Le seuil vient du modele, le declenchement de la
  strategie, la mise en service du registre ;
- il n'envoie aucune notification et n'expose aucune metrique. Il **ecrit le
  journal d'evenements** (§9), et c'est ce journal que l'exporter (phase C) et
  Telegram (phase D) consomment. Une source unique, pas des messages ad hoc.

Quatre decisions de ce module demandent une justification, parce qu'elles ne
sont pas evidentes et qu'aucune n'est rattrapable apres coup :

1. **Le prechargement ne score pas.** Au demarrage, la boucle lit les dernieres
   heures pour remplir son historique : sans cela, la validation du §6 refuserait
   tout reentrainement pendant plusieurs heures (« historique < 300 fenetres »),
   et un redemarrage rendrait le service inerte juste quand il sert. Mais ces
   fenetres passees ne sont **ni scorees, ni alertees, ni donnees au moniteur de
   derive** : alerter sur un evenement vieux d'une heure serait une fausse alerte
   temps reel, et le §7 compte les fausses alertes.
2. **La reference du moniteur vient du registre, pas des donnees recentes.** Elle
   est l'artefact `reference` de la version en service. La recalculer sur les
   fenetres du moment reviendrait a comparer le present a lui-meme : une derive
   survenue pendant un arret du service serait invisible, definitivement.
3. **Un trou n'est pas comble au bord.** Une fenetre dont un point manque est
   interpolee si le trou est court et entoure de points connus — la meme regle
   que le chargeur de sessions, avec la meme borne de configuration. Sinon la
   fenetre est **sautee** et journalisee : scorer une valeur inventee produirait
   une alerte fausse, ou masquerait une vraie. Un point manque coute donc une
   fenetre, pas `window_size`.
4. **Le registre fait foi.** Si l'enregistrement ou le deplacement de l'alias
   echoue, le challenger est jete et le champion reste en service. Promouvoir en
   memoire un modele que le registre ignore ferait mentir les captures du
   chapitre 5 sur ce qui tourne reellement.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from datasets.serveur.settings import ReglagesServeur
from src.config import resolve_path
from src.features.signature import SignatureFeatures, signature_depuis_config
from src.features.windowing import make_windows
from src.models.isolation_forest import IsolationForestDetector, build_detector
from src.serving.prometheus import ClientPrometheus, ErreurPrometheus, grille
from src.strategies.base import Strategy, WindowContext
from src.strategies.retraining import run_retrain_cycle
from src.tracking.registry import ErreurRegistre, RegistreModeles

# Vocabulaire du §9, complete par les evenements propres au service. Les sept
# premiers sont ceux que le harnais de rejeu ecrit deja : Telegram et Grafana
# lisent le meme vocabulaire dans les deux modes.
EVENEMENTS = (
    "alert",
    "drift_signal",
    "drift_confirmed",
    "retrain_start",
    "retrain_end",
    "promoted",
    "rejected",
    # Propres au mode en ligne : ce qui n'existe pas en rejeu, ou le flux est
    # complet et le modele toujours disponible.
    "service_start",
    "service_stop",
    "window_skipped",
    "collect_gap",
    "registry_error",
)


class ErreurBoucle(RuntimeError):
    """La boucle ne peut pas demarrer, ou ne peut pas continuer honnetement."""


def _maintenant() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ReglagesBoucle:
    """Parametres d'exploitation de la boucle. Aucun parametre de detection ici."""

    source: str
    strategie: str
    retard_s: int
    rattrapage_max_points: int
    prefill_fenetres: int
    historique_max_fenetres: int
    dossier_journal: Path

    @classmethod
    def depuis_config(cls, config: dict[str, Any], racine: Path | None = None) -> ReglagesBoucle:
        boucle = dict(config.get("boucle", {}) or {})
        if not boucle:
            raise ErreurBoucle(
                "configs/serving.yaml n'a pas de section `boucle` : le service ne sait "
                "ni a quel rythme lire, ni quelle profondeur d'historique garder."
            )
        journal = dict(config.get("journal", {}) or {})
        dossier = Path(journal.get("dossier", "results/online"))
        if not dossier.is_absolute():
            dossier = (racine / dossier) if racine is not None else resolve_path(dossier)

        reglages = cls(
            source=str(config.get("source", "datasets/serveur.yaml")),
            strategie=str(config.get("strategie", "strategies/proposed.yaml")),
            retard_s=int(boucle.get("retard_s", 15)),
            rattrapage_max_points=int(boucle.get("rattrapage_max_points", 60)),
            prefill_fenetres=int(boucle.get("prefill_fenetres", 400)),
            historique_max_fenetres=int(boucle.get("historique_max_fenetres", 20000)),
            dossier_journal=dossier,
        )
        if reglages.retard_s < 0:
            raise ErreurBoucle("`retard_s` ne peut pas etre negatif.")
        if reglages.rattrapage_max_points < 1:
            raise ErreurBoucle("`rattrapage_max_points` doit valoir au moins 1 point.")
        if reglages.prefill_fenetres < 0:
            raise ErreurBoucle("`prefill_fenetres` ne peut pas etre negatif.")
        if reglages.historique_max_fenetres < 1:
            raise ErreurBoucle("`historique_max_fenetres` doit valoir au moins 1 fenetre.")
        return reglages


@dataclass
class JournalEvenements:
    """Journal JSON lignes du §9 : `{ts, run_id, strategy, event, details}`.

    Chaque ligne est ecrite **et vidée** immediatement : le journal est lu en
    continu par l'exporter et par les notifications, et un tampon de 8 ko
    retarderait une alerte de plusieurs minutes.
    """

    chemin: Path
    run_id: str
    strategie: str
    _fichier: Any = field(default=None, repr=False)
    entrees: list[dict[str, Any]] = field(default_factory=list, repr=False)
    garder_en_memoire: bool = False

    def ouvrir(self) -> JournalEvenements:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self._fichier = self.chemin.open("a", encoding="utf-8")
        return self

    def ecrire(self, event: str, **details: Any) -> dict[str, Any]:
        if event not in EVENEMENTS:
            raise ErreurBoucle(
                f"Evenement « {event} » hors du vocabulaire du §9 : {EVENEMENTS}. "
                "Grafana et Telegram filtrent sur ces noms ; en inventer un le rendrait "
                "invisible dans les deux."
            )
        entree = {
            "ts": _maintenant(),
            "run_id": self.run_id,
            "strategy": self.strategie,
            "event": event,
            "details": details,
        }
        if self._fichier is not None:
            self._fichier.write(json.dumps(entree, ensure_ascii=False, default=float) + "\n")
            self._fichier.flush()
        if self.garder_en_memoire:
            self.entrees.append(entree)
        return entree

    def fermer(self) -> None:
        if self._fichier is not None:
            self._fichier.close()
            self._fichier = None

    def __enter__(self) -> JournalEvenements:
        return self.ouvrir()

    def __exit__(self, *_: object) -> None:
        self.fermer()


@dataclass
class MesuresFenetre:
    """Temps de traitement d'une fenetre, en millisecondes (§9, indicateurs I7/T1).

    C'est **la** mesure que le rejeu ne peut pas donner : il score par blocs et
    n'obtient qu'un temps amorti. Ici chaque fenetre est chronometree de la
    lecture du tampon a la decision de la strategie, reentrainement exclu.
    """

    durees_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))

    def ajouter(self, ms: float) -> None:
        self.durees_ms.append(float(ms))

    @property
    def n(self) -> int:
        return len(self.durees_ms)

    @property
    def moyenne(self) -> float:
        return float(np.mean(self.durees_ms)) if self.durees_ms else 0.0

    @property
    def p95(self) -> float:
        return (
            float(np.percentile(np.fromiter(self.durees_ms, dtype=np.float64), 95))
            if self.durees_ms
            else 0.0
        )


@dataclass(frozen=True)
class ResultatFenetre:
    """Ce qu'une fenetre traitee a produit. Rendu pour les tests et l'exporter."""

    fin_ts: int
    position: int
    score: float
    seuil: float
    alerte: bool
    reentrainement: bool
    promue: bool
    interpolee: bool
    ms: float


class BoucleEnLigne:
    """Service de detection : lit, score, surveille, reentraine, promeut.

    Les dependances sont injectees (client Prometheus, registre, horloge,
    attente) parce que c'est la seule facon de tester une boucle temps reel sans
    serveur et sans attendre : les tests avancent l'horloge eux-memes.
    """

    def __init__(
        self,
        *,
        client: ClientPrometheus,
        registre: RegistreModeles,
        reglages_serveur: ReglagesServeur,
        reglages_boucle: ReglagesBoucle,
        strategy: Strategy,
        features_config: dict[str, Any],
        model_config: dict[str, Any],
        seed: int = 42,
        run_id: str | None = None,
        promotion: bool = True,
        horloge: Callable[[], float] = time.time,
        dormir: Callable[[float], None] = time.sleep,
        journal: JournalEvenements | None = None,
    ) -> None:
        self.client = client
        self.registre = registre
        self.serveur = reglages_serveur
        self.reglages = reglages_boucle
        self.strategy = strategy
        self.features_config = features_config
        self.model_config = model_config
        self.seed = seed
        self.promotion = promotion
        self.horloge = horloge
        self.dormir = dormir

        self.run_id = run_id or f"online_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        self.journal = journal or JournalEvenements(
            chemin=self.reglages.dossier_journal / self.run_id / "events.jsonl",
            run_id=self.run_id,
            strategie=getattr(strategy, "name", "proposed"),
        )

        self.window_size = int(features_config["window_size"])
        self.stride = int(features_config["stride"])
        self.pas = int(reglages_serveur.pas_s)
        self.signature: SignatureFeatures = signature_depuis_config(
            reglages_serveur.noms, features_config, self.pas
        )

        self.champion: IsolationForestDetector | None = None
        self.version: Any = None
        self.mesures = MesuresFenetre()
        self.n_fenetres_vues = 0
        self.n_sautees = 0
        self.n_alertes = 0
        self.n_reentrainements = 0
        self.n_promotions = 0
        self.n_rejets = 0

        # Tampon de points bruts (grille contigue) et historique de features.
        self._instants = np.empty(0, dtype=np.int64)
        self._points = np.empty((0, len(reglages_serveur.noms)), dtype=np.float64)
        self._derniere_fin_ts: int | None = None
        self._historique: deque[np.ndarray] = deque(maxlen=reglages_boucle.historique_max_fenetres)

    # ------------------------------------------------------------- demarrage

    def demarrer(self) -> None:
        """Charge le champion et sa reference, puis preremplit l'historique.

        Echoue plutot que de demarrer degrade : sans champion il n'y a rien a
        scorer, et sans reference le moniteur de derive serait aveugle a toute
        derive anterieure au demarrage.
        """
        self.journal.ouvrir()
        try:
            self.registre.verifier()
            self.champion, self.version = self.registre.charger_champion(self.signature)
            reference = self.registre.charger_reference(self.version)
        except ErreurRegistre as erreur:
            self.journal.ecrire("registry_error", phase="demarrage", message=str(erreur))
            raise ErreurBoucle(f"Demarrage impossible : {erreur}") from erreur

        if reference.shape[1] != self.signature.n_features:
            raise ErreurBoucle(
                f"La reference de la version {self.version.version} a "
                f"{reference.shape[1]} colonnes pour une signature de "
                f"{self.signature.n_features} : incoherence interne du registre."
            )

        self.strategy.on_fit(reference, 0)
        self.journal.ecrire(
            "service_start",
            version=self.version.version,
            threshold=float(self.champion.threshold),
            n_features=self.signature.n_features,
            features_empreinte=self.signature.empreinte,
            n_reference_windows=int(len(reference)),
            pas_s=self.pas,
            promotion=self.promotion,
        )
        self._precharger()

    def _precharger(self) -> None:
        """Remplit le tampon et l'historique avec le passe, sans rien scorer."""
        points_voulus = self.window_size + max(0, self.reglages.prefill_fenetres - 1) * self.stride
        fin = self.horloge() - self.reglages.retard_s
        debut = fin - (points_voulus - 1) * self.pas
        instants, matrice = self._lire(debut, fin)
        if len(instants) == 0:
            raise ErreurBoucle(
                "Prometheus n'a aucun point sur la plage de prechargement : la pile de "
                "collecte vient-elle de demarrer ? Attendre quelques minutes, ou "
                "verifier les cibles avec `mdrift session-check`."
            )

        self._instants = instants
        self._points = matrice
        self._ancrer()
        # Les fenetres du passe alimentent l'historique de reentrainement, et
        # rien d'autre : pas de score, pas d'alerte, pas de mise a jour du
        # moniteur de derive (voir le point 1 de l'en-tete).
        prechargees = 0
        for fin_ts, bloc in self._fenetres_disponibles():
            # Le curseur avance meme quand la fenetre est insaisissable : sinon
            # elle serait reproposee au premier tick, donc traitee en direct
            # alors qu'elle appartient au passe.
            self._derniere_fin_ts = fin_ts
            features = self._features(bloc)
            if features is None:
                continue
            self._historique.append(features)
            prechargees += 1
        self.n_fenetres_vues = prechargees

    # ---------------------------------------------------------------- lecture

    def _lire(self, debut: float, fin: float) -> tuple[np.ndarray, np.ndarray]:
        """Lit toutes les metriques sur une plage et les empile en colonnes."""
        try:
            instants = grille(debut, fin, self.pas)
        except ValueError:
            return np.empty(0, dtype=np.int64), np.empty((0, len(self.serveur.noms)))

        colonnes: list[np.ndarray] = []
        for metrique in self.serveur.metriques:
            serie = self.client.interroger_plage(
                metrique.expression(self.serveur.rate_interval), debut, fin, self.pas
            )
            if not np.array_equal(serie.instants, instants):  # pragma: no cover - meme grille
                raise ErreurBoucle(
                    f"La grille de « {metrique.nom} » ne correspond pas a celle demandee."
                )
            colonnes.append(serie.valeurs)
        return instants, np.column_stack(colonnes)

    def _ancrer(self) -> None:
        """Place le curseur de fenetrage pour que la premiere fenetre pleine sorte."""
        premier = int(self._instants[0])
        self._derniere_fin_ts = premier + (self.window_size - 1 - self.stride) * self.pas

    def _fenetres_disponibles(self) -> list[tuple[int, np.ndarray]]:
        """Fenetres completes pas encore emises, dans l'ordre du temps."""
        sorties: list[tuple[int, np.ndarray]] = []
        if len(self._instants) == 0 or self._derniere_fin_ts is None:
            return sorties
        premier = int(self._instants[0])
        dernier = int(self._instants[-1])
        candidat = self._derniere_fin_ts + self.stride * self.pas
        while candidat <= dernier:
            debut_offset = (candidat - premier) // self.pas - (self.window_size - 1)
            if debut_offset < 0:
                candidat += self.stride * self.pas
                continue
            fin_offset = debut_offset + self.window_size
            sorties.append((candidat, self._points[debut_offset:fin_offset]))
            candidat += self.stride * self.pas
        return sorties

    # ---------------------------------------------------------------- features

    def _combler(self, bloc: np.ndarray) -> np.ndarray | None:
        """Interpole les trous courts et interieurs. Rend `None` si c'est impossible.

        Meme regle que le chargeur de sessions (`max_trou_consecutif`), et meme
        refus d'extrapoler : un point manquant au bord de la fenetre — le plus
        recent, celui qui declencherait l'alerte — n'a pas de voisin droit, donc
        la fenetre n'est pas scoree.
        """
        if np.isfinite(bloc).all():
            return bloc

        comble = bloc.astype(np.float64, copy=True)
        comble[~np.isfinite(comble)] = np.nan
        indices = np.arange(len(comble), dtype=np.float64)
        for colonne in range(comble.shape[1]):
            valeurs = comble[:, colonne]
            manquants = np.isnan(valeurs)
            if not manquants.any():
                continue
            connus = ~manquants
            if not connus.any():
                return None
            # Bord : aucun voisin d'un cote, donc aucune interpolation possible.
            premier, dernier = (
                int(np.argmax(connus)),
                int(len(connus) - 1 - np.argmax(connus[::-1])),
            )
            if manquants[:premier].any() or manquants[dernier + 1 :].any():
                return None
            # Plus long trou consecutif, comme dans le chargeur : c'est sa
            # longueur qui decide, pas le nombre total de trous.
            longueur = 0
            maximum = 0
            for absent in manquants:
                longueur = longueur + 1 if absent else 0
                maximum = max(maximum, longueur)
            if maximum > self.serveur.max_trou_consecutif:
                return None
            valeurs[manquants] = np.interp(indices[manquants], indices[connus], valeurs[connus])
        return comble

    def _features(self, bloc: np.ndarray) -> np.ndarray | None:
        """Rend la ligne de features d'une fenetre, ou `None` si elle est insaisissable."""
        comble = self._combler(bloc)
        if comble is None:
            return None
        fenetres = make_windows(comble, self.features_config, metric_names=list(self.serveur.noms))
        if len(fenetres.X) != 1:  # pragma: no cover - une fenetre par bloc, par construction
            raise ErreurBoucle(
                f"{len(fenetres.X)} fenetres produites pour un bloc de {len(comble)} points."
            )
        return np.asarray(fenetres.X[0], dtype=np.float64)

    # ------------------------------------------------------------------- tick

    def tick(self) -> list[ResultatFenetre]:
        """Lit les nouveaux points et traite les fenetres completes qui en decoulent."""
        if self.champion is None:
            raise ErreurBoucle("`demarrer()` doit etre appele avant `tick()`.")

        fin = self.horloge() - self.reglages.retard_s
        attendu = int(self._instants[-1]) + self.pas if len(self._instants) else fin
        if fin < attendu:
            return []

        manquants = int((fin - attendu) // self.pas) + 1
        if manquants > self.reglages.rattrapage_max_points:
            # Trop de retard pour que ce soit encore du temps reel : la boucle
            # repart des points recents et le dit. Rejouer une demi-journee
            # fenetre par fenetre produirait des alertes sur des evenements
            # passes depuis longtemps, que le §7 compterait en fausses alertes.
            self.journal.ecrire(
                "collect_gap",
                points_manques=manquants,
                borne=self.reglages.rattrapage_max_points,
                reprise_ts=int(fin - (self.reglages.rattrapage_max_points - 1) * self.pas),
            )
            attendu = int(fin - (self.reglages.rattrapage_max_points - 1) * self.pas)
            self._instants = np.empty(0, dtype=np.int64)
            self._points = np.empty((0, len(self.serveur.noms)), dtype=np.float64)
            self._derniere_fin_ts = None

        try:
            instants, matrice = self._lire(attendu, fin)
        except ErreurPrometheus as erreur:
            # Prometheus indisponible : la boucle ne s'arrete pas, elle reessaie
            # au tick suivant. Un service de detection qui meurt avec sa source
            # ne survivrait pas a un redemarrage de la pile de collecte.
            self.journal.ecrire("collect_gap", message=str(erreur))
            return []
        if len(instants) == 0:
            return []

        if len(self._instants) == 0:
            self._instants, self._points = instants, matrice
            self._ancrer()
        else:
            self._instants = np.concatenate([self._instants, instants])
            self._points = np.vstack([self._points, matrice])
        self._elaguer()

        resultats: list[ResultatFenetre] = []
        for fin_ts, bloc in self._fenetres_disponibles():
            resultat = self._traiter(fin_ts, bloc)
            self._derniere_fin_ts = fin_ts
            if resultat is not None:
                resultats.append(resultat)
        return resultats

    def _elaguer(self) -> None:
        """Borne le tampon de points bruts : une demonstration dure des heures."""
        garde = self.window_size + self.reglages.rattrapage_max_points * self.stride
        if len(self._instants) > garde:
            self._instants = self._instants[-garde:]
            self._points = self._points[-garde:]

    def _traiter(self, fin_ts: int, bloc: np.ndarray) -> ResultatFenetre | None:
        """Score une fenetre, consulte la strategie, et reentraine si elle le demande."""
        assert self.champion is not None
        depart = time.perf_counter()
        features = self._features(bloc)
        if features is None:
            self.n_sautees += 1
            self.journal.ecrire(
                "window_skipped",
                fin_ts=fin_ts,
                raison="points_manquants",
                borne_trou=self.serveur.max_trou_consecutif,
            )
            return None

        interpolee = not np.isfinite(bloc).all()
        score = float(self.champion.score(features.reshape(1, -1))[0])
        seuil = float(self.champion.threshold)
        alerte = score > seuil
        position = self.n_fenetres_vues

        decision = self.strategy.on_window(
            WindowContext(
                position=position,
                end_index=fin_ts,
                features=features,
                score=score,
                alert=alerte,
                threshold=seuil,
            )
        )
        ms = 1000.0 * (time.perf_counter() - depart)

        self._historique.append(features)
        self.n_fenetres_vues += 1
        self.mesures.ajouter(ms)

        if alerte:
            self.n_alertes += 1
            self.journal.ecrire(
                "alert",
                fin_ts=fin_ts,
                position=position,
                score=round(score, 6),
                threshold=round(seuil, 6),
                interpolee=interpolee,
            )
        for event in decision.events:
            self.journal.ecrire(event, fin_ts=fin_ts, position=position, **decision.details)

        promue = False
        if decision.retrain:
            promue = self._reentrainer(fin_ts, position, decision.reason)

        return ResultatFenetre(
            fin_ts=fin_ts,
            position=position,
            score=score,
            seuil=seuil,
            alerte=alerte,
            reentrainement=decision.retrain,
            promue=promue,
            interpolee=interpolee,
            ms=ms,
        )

    # ---------------------------------------------------------- reentrainement

    def _reentrainer(self, fin_ts: int, position: int, raison: str) -> bool:
        """Deroule le cycle partage, puis promeut ou rejette. Rend vrai si promu."""
        assert self.champion is not None
        historique = np.array(self._historique, dtype=np.float64)
        self.journal.ecrire(
            "retrain_start",
            fin_ts=fin_ts,
            position=position,
            raison=raison,
            n_history=self.n_fenetres_vues,
            n_buffer=len(historique),
        )
        self.n_reentrainements += 1

        cycle = run_retrain_cycle(
            self.strategy,
            history_X=historique,
            # La strategie raisonne en indices de son historique en memoire, qui
            # est borne : la position relative au tampon est celle qu'elle peut
            # interpreter, tandis que `n_history` garde le compte absolu pour le
            # garde-fou « historique trop court ».
            position=len(historique) - 1,
            n_history=self.n_fenetres_vues,
            champion=self.champion,
            build=lambda: build_detector(self.model_config, self.features_config, self.seed),
        )
        self.journal.ecrire(
            "retrain_end",
            fin_ts=fin_ts,
            position=position,
            seconds=round(cycle.challenger.fit_seconds, 3),
            threshold=round(float(cycle.challenger.threshold), 6),
            **cycle.details(),
        )

        if not cycle.promote:
            self.n_rejets += 1
            self.journal.ecrire(
                "rejected", fin_ts=fin_ts, position=position, raison=cycle.verdict.reason
            )
            return False

        if not self.promotion:
            # Mode a blanc : la validation a dit oui, mais rien n'est mis en
            # service. Sert a verifier une configuration avant une demonstration
            # sans changer le champion que montreront les captures.
            self.journal.ecrire(
                "rejected", fin_ts=fin_ts, position=position, raison="promotion_desactivee"
            )
            self.n_rejets += 1
            return False

        try:
            version = self.registre.enregistrer(
                cycle.challenger,
                signature=self.signature,
                raison=raison or "drift_retrain",
                parametres={"seed": self.seed, "source": "serveur", "run_id": self.run_id},
                metriques={
                    "challenger_alert_rate": cycle.stats.challenger_alert_rate,
                    "champion_alert_rate": cycle.stats.champion_alert_rate,
                },
                tags={"run_id": self.run_id, "position": str(position)},
                reference_X=cycle.plan.train_X,
            )
            version = self.registre.promouvoir(version)
        except ErreurRegistre as erreur:
            # Le registre fait foi (point 4 de l'en-tete) : le challenger est
            # jete, le champion reste en service, et l'echec est journalise.
            self.journal.ecrire("registry_error", phase="promotion", message=str(erreur))
            self.n_rejets += 1
            return False

        self.champion = cycle.challenger
        self.version = version
        self.strategy.on_fit(cycle.plan.train_X, position)
        self.n_promotions += 1
        self.journal.ecrire(
            "promoted",
            fin_ts=fin_ts,
            position=position,
            version=version.version,
            threshold=round(float(cycle.challenger.threshold), 6),
            n_train_windows=int(len(cycle.plan.train_X)),
        )
        return True

    # ------------------------------------------------------------- execution

    def executer(self, duree_s: float | None = None, max_ticks: int | None = None) -> None:
        """Boucle jusqu'a l'echeance, en dormant jusqu'au prochain instant de grille."""
        debut = self.horloge()
        ticks = 0
        try:
            while True:
                if duree_s is not None and self.horloge() - debut >= duree_s:
                    break
                if max_ticks is not None and ticks >= max_ticks:
                    break
                self.tick()
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                self.dormir(self._attente())
        finally:
            self.arreter()

    def _attente(self) -> float:
        """Secondes a attendre avant que le prochain point de grille soit publie."""
        maintenant = self.horloge()
        if len(self._instants) == 0:
            return float(self.pas)
        prochain = int(self._instants[-1]) + self.pas + self.reglages.retard_s
        return max(1.0, prochain - maintenant)

    def arreter(self) -> None:
        """Journalise la fin du service avec les compteurs de la session."""
        self.journal.ecrire(
            "service_stop",
            n_windows=self.n_fenetres_vues,
            n_skipped=self.n_sautees,
            n_alerts=self.n_alertes,
            n_retrains=self.n_reentrainements,
            n_promoted=self.n_promotions,
            n_rejected=self.n_rejets,
            window_ms_mean=round(self.mesures.moyenne, 3),
            window_ms_p95=round(self.mesures.p95, 3),
        )
        self.journal.fermer()
