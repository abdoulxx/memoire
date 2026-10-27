"""Regle d'agregation des tests de derive sur les 152 features (D18).

Le signal de derive est la **part des features utilisables dont la statistique
depasse un seuil**, comparee a une part declenchante. Ni une alarme des qu'une
feature derive — avec 152 tests par fenetre il y en a presque toujours une — ni
une correction de Bonferroni sur les p-valeurs, qui a ete mesuree et ne tient
pas : les p-valeurs n'ont aucune specificite a 20 000 points, et les 152 tests
ne sont pas independants (rang effectif 120 a 131 sur 152).

La fenetre recente est maintenue par un **histogramme glissant** : a chaque
fenetre, l'intervalle de la fenetre entrante est incremente et celui de la
fenetre sortante decremente. Le cout par fenetre est alors O(F x B) et non
O(F x R log N), ce qui est la condition pour que le rejeu de la campagne M3
tienne et pour que la boucle en ligne respecte l'intervalle de collecte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.drift.ks import ks_from_proportions
from src.drift.psi import psi_from_proportions
from src.drift.reference import DriftReference

STATISTICS = ("ks", "psi")


@dataclass(frozen=True)
class FeatureDriftSignal:
    """Etat du signal de derive par features, a une fenetre donnee."""

    ready: bool
    share: float
    n_features_drifting: int
    n_features_usable: int
    statistic_median: float
    statistic_max: float

    def as_details(self) -> dict[str, float | int | bool]:
        return {
            "ready": self.ready,
            "share": round(self.share, 4),
            "n_features_drifting": self.n_features_drifting,
            "n_features_usable": self.n_features_usable,
            "statistic_median": round(self.statistic_median, 4),
            "statistic_max": round(self.statistic_max, 4),
        }


class FeatureDriftTracker:
    """Histogramme glissant de la fenetre recente, et part de features en derive.

    `recent_windows` est la taille R de la fenetre recente comparee a la
    reference. Tant que moins de R fenetres ont ete vues depuis le dernier
    recalage, le signal est declare non pret (`ready = False`) et ne peut pas
    confirmer de derive : comparer une reference de 28 000 fenetres a 12
    fenetres recentes produirait un ecart enorme et sans signification.
    """

    def __init__(
        self,
        reference: DriftReference,
        recent_windows: int,
        statistic: str = "ks",
        feature_threshold: float | np.ndarray = 0.5,
        psi_epsilon: float = 1e-6,
    ) -> None:
        if statistic not in STATISTICS:
            raise ValueError(
                f"Statistique inconnue : {statistic!r}. Attendu l'une de {STATISTICS}."
            )
        if recent_windows < 2:
            raise ValueError("recent_windows doit valoir au moins 2.")

        self.reference = reference
        self.recent_windows = int(recent_windows)
        self.statistic = statistic
        self.psi_epsilon = float(psi_epsilon)

        # Le seuil est soit un scalaire (seuil absolu, conserve pour l'ablation),
        # soit un vecteur par feature issu de l'hypothese nulle empirique
        # (src/drift/null_model.py). Le second est le defaut, le premier ne
        # separant pas les machines stables des machines derivantes.
        threshold = np.asarray(feature_threshold, dtype=np.float64)
        if threshold.ndim == 0:
            threshold = np.full(reference.n_features, float(threshold))
        elif threshold.shape != (reference.n_features,):
            raise ValueError(
                f"feature_threshold doit etre scalaire ou de forme ({reference.n_features},), "
                f"recu {threshold.shape}."
            )
        self.feature_threshold = threshold

        self._counts = np.zeros((reference.n_features, reference.n_bins), dtype=np.float64)
        self._buffer = np.zeros((self.recent_windows, reference.n_features), dtype=np.int64)
        self._seen = 0
        self._rows = np.arange(reference.n_features)
        self._last_values: np.ndarray | None = None

    @property
    def n_seen(self) -> int:
        return self._seen

    @property
    def last_values(self) -> np.ndarray | None:
        """Statistique par feature a la derniere fenetre, ou None si pas encore prete.

        Expose la valeur **deja calculee** par `update`, sans la recalculer. Elle
        ne sert pas au rejeu, qui n'a besoin que de la part agregee, mais au
        calibrage : balayer les seuils sur un vecteur de 152 statistiques par
        fenetre est gratuit, alors que rejouer le flux pour chaque point de
        grille couterait des heures.
        """
        return self._last_values

    def update(self, features: np.ndarray) -> FeatureDriftSignal:
        """Ajoute une fenetre et renvoie le signal courant."""
        slot = self._seen % self.recent_windows
        if self._seen >= self.recent_windows:
            self._counts[self._rows, self._buffer[slot]] -= 1.0

        indices = self.reference.bin_indices(np.asarray(features, dtype=np.float64))
        self._buffer[slot] = indices
        self._counts[self._rows, indices] += 1.0
        self._seen += 1

        n_in_window = min(self._seen, self.recent_windows)
        if self._seen < self.recent_windows:
            return FeatureDriftSignal(
                ready=False,
                share=0.0,
                n_features_drifting=0,
                n_features_usable=self.reference.n_usable,
                statistic_median=0.0,
                statistic_max=0.0,
            )

        proportions = self._counts / float(n_in_window)
        if self.statistic == "ks":
            values = ks_from_proportions(self.reference.proportions, proportions)
        else:
            values = psi_from_proportions(
                self.reference.proportions, proportions, epsilon=self.psi_epsilon
            )

        self._last_values = values
        usable = values[self.reference.usable]
        drifting = usable > self.feature_threshold[self.reference.usable]
        return FeatureDriftSignal(
            ready=True,
            share=float(drifting.mean()) if usable.size else 0.0,
            n_features_drifting=int(drifting.sum()),
            n_features_usable=int(usable.size),
            statistic_median=float(np.median(usable)) if usable.size else 0.0,
            statistic_max=float(usable.max()) if usable.size else 0.0,
        )
