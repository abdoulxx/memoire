"""Statistique de Kolmogorov-Smirnov a deux echantillons.

D18 a tranche : c'est la **statistique** qui est utilisee, pas la p-valeur. Sur
20 000 points, le KS rejette l'egalite des distributions pour des ecarts
infimes et sans effet sur le detecteur — 109 features sur 152 rejettent au seuil
de Bonferroni sur machine-1-2, qui est une machine stable. La statistique, elle,
discrimine : 0,0 % de features au-dela de 0,5 sur machine-1-2 contre 26,3 % sur
machine-3-1.

Deux formes coexistent, et c'est voulu :

- `ks_statistic` : la forme exacte, qui s'appuie sur `scipy.stats.ks_2samp`.
  Elle sert de reference de verite et n'est pas utilisee dans la boucle de
  rejeu, ou son cout est prohibitif (voir `reference.py`).
- `ks_from_proportions` : la forme discretisee, calculee sur les comptages par
  intervalle deja disponibles. C'est celle que le rejeu utilise.

Ecart entre les deux, **mesure** sur machine-1-1 (reference 28 470 fenetres
d'entrainement, fenetre recente de 500 fenetres, 120 features utilisables) :

    n_bins :      10        20        50       100
    erreur max    0,1468    0,0722    0,0265    0,0127
    erreur med.   0,0105    0,0069    0,0030    0,0015

L'erreur decroit bien en 1 / n_bins, mais elle **depasse** cette valeur d'un
facteur 1,3 a 1,5 : les features dont plus de 1 / n_bins des valeurs sont
identiques ont des quantiles dupliques, donc moins d'intervalles effectifs, et
le saut local de la fonction de repartition y excede 1 / n_bins. Il n'y a donc
pas de borne theorique propre a annoncer, seulement cette mesure. `n_bins = 50`
est retenu par defaut : erreur maximale 0,027 pour un cout par fenetre encore
negligeable. Un test de non-regression fige cet ordre de grandeur.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def ks_from_proportions(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Statistique de KS par feature, sur la discretisation commune.

    La statistique est le maximum de l'ecart absolu entre les deux fonctions de
    repartition. Sur une discretisation commune, ce maximum est pris sur les
    bornes d'intervalle, ou les deux repartitions sont connues exactement.

    `reference` et `current` ont la forme (n_features, n_bins) ; le resultat a la
    forme (n_features,).
    """
    if reference.shape != current.shape:
        raise ValueError(
            f"Formes incompatibles : reference {reference.shape}, courant {current.shape}."
        )
    cdf_reference = np.cumsum(reference, axis=1)
    cdf_current = np.cumsum(current, axis=1)
    return np.abs(cdf_current - cdf_reference).max(axis=1)


def ks_statistic(reference_sample: np.ndarray, current_sample: np.ndarray) -> float:
    """Statistique de KS exacte entre deux echantillons 1-D.

    Forme de reference, pour les tests et les analyses hors ligne. La p-valeur
    renvoyee par scipy est volontairement ignoree (D18).
    """
    result = ks_2samp(
        np.asarray(reference_sample, dtype=np.float64).ravel(),
        np.asarray(current_sample, dtype=np.float64).ravel(),
    )
    return float(result.statistic)
