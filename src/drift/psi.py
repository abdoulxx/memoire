"""Population Stability Index, code a la main (cadrage, §3).

PSI = somme sur les intervalles de (a - e) * ln(a / e), ou `e` est la proportion
attendue (reference, le modele en service) et `a` la proportion observee
(fenetre recente). Nul si les deux distributions coincident, croissant avec
l'ecart, et non borne.

Lecture usuelle dans la litterature du credit scoring, d'ou la mesure vient :
moins de 0,1 pas de changement, 0,1 a 0,25 changement moderee, au-dela de 0,25
changement important. Ces reperes ne sont pas repris comme seuils : le seuil par
feature et la part declenchante sont calibres sur le perimetre de mise au point
(D17, D18). [A VERIFIER] la source primaire de ces reperes n'est pas identifiee
dans la litterature academique ; ne pas les citer au memoire sans source.

Un intervalle vide d'un cote donne un terme infini. Le plancher `epsilon`
l'evite : c'est la convention habituelle, et elle est explicite ici plutot que
cachee dans une bibliotheque.
"""

from __future__ import annotations

import numpy as np


def psi_from_proportions(
    reference: np.ndarray,
    current: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """PSI par feature, a partir des proportions par intervalle.

    `reference` et `current` ont la forme (n_features, n_bins). Le resultat a la
    forme (n_features,).
    """
    if reference.shape != current.shape:
        raise ValueError(
            f"Formes incompatibles : reference {reference.shape}, courant {current.shape}."
        )
    expected = np.clip(reference, epsilon, None)
    actual = np.clip(current, epsilon, None)
    return ((actual - expected) * np.log(actual / expected)).sum(axis=1)


def psi(
    reference_sample: np.ndarray,
    current_sample: np.ndarray,
    n_bins: int = 20,
    epsilon: float = 1e-6,
) -> float:
    """PSI entre deux echantillons 1-D, bins de quantiles de la reference.

    Forme directe, utilisee par les tests et les analyses hors ligne. Le rejeu
    passe par `DriftReference`, qui ne discretise la reference qu'une fois.
    """
    reference_sample = np.asarray(reference_sample, dtype=np.float64).ravel()
    current_sample = np.asarray(current_sample, dtype=np.float64).ravel()

    edges = np.quantile(reference_sample, np.linspace(0.0, 1.0, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    interior = edges[1:-1]

    ref_counts = np.bincount(
        (reference_sample[:, None] >= interior).sum(axis=1), minlength=n_bins
    ).astype(np.float64)
    cur_counts = np.bincount(
        (current_sample[:, None] >= interior).sum(axis=1), minlength=n_bins
    ).astype(np.float64)

    value = psi_from_proportions(
        (ref_counts / ref_counts.sum())[None, :],
        (cur_counts / cur_counts.sum())[None, :],
        epsilon=epsilon,
    )
    return float(value[0])
