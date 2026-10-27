"""Distribution de reference : ce que le modele en service a appris.

La surveillance de derive compare la distribution **recente** des features a
celle sur laquelle le modele courant a ete entraine (cadrage, §6.1). Cette
reference est donc recalculee a chaque promotion, et a elle seule.

Choix de conception : la reference est discretisee une fois pour toutes en
`n_bins` intervalles de quantiles, par feature. Tout le reste (PSI, KS, part de
features en derive) se calcule ensuite a partir de **comptages par intervalle**.

Pourquoi : le cout. Un KS exact entre une fenetre recente de R points et une
reference de N points coute O(R log N) par feature, soit environ 1,1 million
d'operations par fenetre pour 152 features — multiplie par 28 000 fenetres, par
4 scenarios, par 5 graines et par 12 machines, le rejeu ne tient plus. Sur la
discretisation, le cout par fenetre tombe a O(F x B) et devient negligeable.
C'est aussi la seule forme compatible avec la contrainte temps reel du §2
(indicateurs I7 / T1 : traiter une fenetre en moins d'un intervalle de
collecte). L'ecart entre le KS discretise et le KS exact est mesure et borne
par `tests/test_drift.py`.

Les features **constantes** sur la reference sont exclues (D18) : un test de
distribution sur une constante n'a pas de sens, et le SMD en compte 5 a 8 par
machine parmi les metriques brutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftReference:
    """Discretisation de la distribution d'entrainement du modele en service.

    `edges` a la forme (n_features, n_bins + 1) ; les bornes extremes sont
    infinies pour qu'une valeur hors de la plage d'entrainement tombe dans le
    premier ou le dernier intervalle au lieu d'etre perdue. C'est voulu : une
    valeur jamais vue a l'entrainement est precisement un signe de derive.

    `proportions` a la forme (n_features, n_bins) et somme a 1 par feature.
    `usable` est le masque des features non constantes sur la reference.
    """

    edges: np.ndarray
    proportions: np.ndarray
    usable: np.ndarray
    n_reference: int

    @property
    def n_features(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.edges.shape[1] - 1)

    @property
    def n_usable(self) -> int:
        return int(self.usable.sum())

    def bin_indices(self, values: np.ndarray) -> np.ndarray:
        """Intervalle de chaque feature, pour une fenetre (1-D) ou un bloc (2-D).

        Chaque feature a ses propres bornes, ce qui interdit un `searchsorted`
        global. La comparaison aux bornes interieures est vectorisee a la place :
        pour B intervalles, B - 1 comparaisons par feature, sans boucle Python.
        """
        interior = self.edges[:, 1:-1]
        if values.ndim == 1:
            return (values[:, None] >= interior).sum(axis=1).astype(np.int64)
        return (values[:, :, None] >= interior[None, :, :]).sum(axis=2).astype(np.int64)


def build_reference(
    training_features: np.ndarray,
    n_bins: int = 20,
    constant_tolerance: float = 0.0,
) -> DriftReference:
    """Construit la reference a partir des features d'entrainement du modele courant.

    Les intervalles sont des **quantiles** de la reference et non une grille
    reguliere : les features du SMD sont tres asymetriques (24 a 31 % des
    valeurs valent exactement 0, OBS03/D15), et une grille reguliere placerait
    la quasi-totalite de la masse dans un seul intervalle, ce qui rendrait les
    deux statistiques aveugles.

    Les quantiles dupliques (feature a faible support) sont dedoublonnes puis
    completes, de sorte que `edges` reste de forme fixe et que les intervalles
    vides soient traites par le plancher de probabilite du PSI plutot que par
    une forme variable.
    """
    if training_features.ndim != 2:
        raise ValueError("training_features doit etre de forme (n_fenetres, n_features).")
    if n_bins < 2:
        raise ValueError("n_bins doit valoir au moins 2.")
    n_rows, n_features = training_features.shape
    if n_rows < n_bins:
        raise ValueError(
            f"Reference trop courte : {n_rows} fenetres pour {n_bins} intervalles demandes."
        )

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw = np.quantile(training_features, quantiles, axis=0).T  # (n_features, n_bins + 1)

    edges = np.empty_like(raw)
    for feature in range(n_features):
        edges[feature] = _monotone_edges(raw[feature])

    # Une feature est constante si l'etendue de la reference est nulle (ou sous
    # la tolerance). Le test de distribution serait alors sans objet.
    spread = training_features.max(axis=0) - training_features.min(axis=0)
    usable = spread > constant_tolerance

    bounded = edges.copy()
    bounded[:, 0] = -np.inf
    bounded[:, -1] = np.inf

    indices = (training_features[:, :, None] >= bounded[None, :, 1:-1]).sum(axis=2)
    counts = _counts_from_indices(indices, n_features, n_bins)
    proportions = counts / float(n_rows)

    return DriftReference(
        edges=bounded,
        proportions=proportions,
        usable=usable,
        n_reference=int(n_rows),
    )


def _monotone_edges(values: np.ndarray) -> np.ndarray:
    """Rend une suite de bornes strictement croissante en ecartant les doublons.

    Une feature dont plus de 1 / n_bins des valeurs sont identiques produit des
    quantiles egaux. Les bornes sont alors ecartees d'un increment infinitesimal
    pour que les intervalles restent bien definis ; les intervalles ainsi vides
    recoivent une proportion nulle, geree par le plancher du PSI.
    """
    out = values.astype(np.float64, copy=True)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = np.nextafter(out[i - 1], np.inf)
    return out


def _counts_from_indices(indices: np.ndarray, n_features: int, n_bins: int) -> np.ndarray:
    """Comptage par (feature, intervalle) a partir des indices d'intervalle."""
    flat = indices + np.arange(n_features)[None, :] * n_bins
    counts = np.bincount(flat.ravel(), minlength=n_features * n_bins)
    return counts.reshape(n_features, n_bins).astype(np.float64)
