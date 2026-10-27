"""Seuil par feature calibre sur l'entrainement lui-meme (hypothese nulle empirique).

Pourquoi ce module existe, et c'est un resultat du memoire autant qu'un detail
d'implementation.

D18 a mesure la statistique de KS entre les deux **moitiés entieres** d'une
machine (environ 20 000 points de chaque cote) et y a trouve un contraste net :
0,0 % de features au-dela de 0,5 sur machine-1-2, machine stable, contre 26,3 %
sur machine-3-1. Il etait tentant d'en tirer un seuil absolu de 0,5.

La mesure montre que cela ne marche pas. Le moniteur en service ne compare pas
deux moities : il compare la reference d'entrainement a une fenetre **recente**
de R fenetres. Avec R = 500 et un pas d'une minute, cette fenetre couvre 8,3 h,
et ses fenetres sont massivement chevauchantes donc fortement autocorrelees.
Une tranche de 8 h d'une metrique serveur ne couvre jamais l'etendue d'un
entrainement de 20 jours : la statistique de KS y est elevee **meme sans aucune
derive**. Mesure sur le perimetre de mise au point, scenario `none`, seuil
absolu de 0,5 : part mediane de features « en derive » de 0,1500 a 0,4914 sur
les machines stables et de 0,1667 a 0,4621 sur les machines instables. Les deux
populations ne se separent pas, et un seuil absolu est donc inutilisable.

La solution retenue compare ce qui est comparable. Pour chaque feature, on fait
glisser une fenetre de la meme taille R sur la moitie `train` et on mesure la
statistique atteinte **en l'absence de toute derive**. Le seuil de la feature
est un quantile eleve de cette distribution, eventuellement majore d'une marge.
Une feature est alors declaree en derive quand elle depasse ce qu'elle atteignait
de pire pendant l'entrainement — ce qui neutralise d'un coup l'autocorrelation,
la taille d'echantillon et l'etendue propre a chaque feature, sans avoir a les
modeliser.

Le cout est paye une fois par entrainement du modele en service, jamais par
fenetre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.drift.ks import ks_from_proportions
from src.drift.psi import psi_from_proportions
from src.drift.reference import DriftReference, _counts_from_indices


@dataclass(frozen=True)
class NullModel:
    """Seuils par feature, et la distribution dont ils sont tires.

    `thresholds` a la forme (n_features,). Les features inutilisables (constantes
    sur la reference) y recoivent `inf` : elles ne peuvent jamais etre declarees
    en derive, ce qui est la regle de D18.
    """

    thresholds: np.ndarray
    quantile: float
    margin: float
    n_blocks: int

    @property
    def n_features(self) -> int:
        return int(self.thresholds.shape[0])


def null_distribution(
    training_features: np.ndarray,
    reference: DriftReference,
    recent_windows: int,
    statistic: str = "ks",
    step: int | None = None,
    psi_epsilon: float = 1e-6,
) -> np.ndarray:
    """Distribution de la statistique sous l'hypothese nulle, de forme (n_blocs, F).

    C'est la mesure brute dont `build_null_model` ne retient qu'un quantile. Elle
    est exposee telle quelle parce que le calibrage balaie plusieurs quantiles et
    plusieurs marges : les derivant de la meme distribution evite de reparcourir
    l'entrainement pour chaque point de grille.
    """
    features = np.asarray(training_features, dtype=np.float64)
    n_rows, n_features = features.shape
    if recent_windows < 2:
        raise ValueError("recent_windows doit valoir au moins 2.")
    if n_rows < recent_windows:
        raise ValueError(
            f"Entrainement trop court : {n_rows} fenetres pour une fenetre recente de "
            f"{recent_windows}. Impossible de calibrer le seuil par feature."
        )
    if step is None:
        step = max(1, recent_windows // 10)

    indices = reference.bin_indices(features)
    starts = range(0, n_rows - recent_windows + 1, int(step))

    values: list[np.ndarray] = []
    for start in starts:
        block = indices[start : start + recent_windows]
        counts = _counts_from_indices(block, n_features, reference.n_bins)
        proportions = counts / float(recent_windows)
        if statistic == "ks":
            values.append(ks_from_proportions(reference.proportions, proportions))
        else:
            values.append(
                psi_from_proportions(reference.proportions, proportions, epsilon=psi_epsilon)
            )

    if not values:  # pragma: no cover - exclu par la verification de longueur
        raise RuntimeError("Aucun bloc : fenetre recente trop grande pour l'entrainement.")

    return np.vstack(values)  # (n_blocks, n_features)


def thresholds_from_distribution(
    distribution: np.ndarray,
    reference: DriftReference,
    quantile: float = 0.99,
    margin: float = 1.0,
) -> NullModel:
    """Retient un quantile de la distribution nulle comme seuil par feature."""
    thresholds = np.quantile(distribution, quantile, axis=0) * float(margin)

    # Une feature constante ne peut pas deriver : seuil infini (D18).
    thresholds = np.where(reference.usable, thresholds, np.inf)

    return NullModel(
        thresholds=thresholds,
        quantile=float(quantile),
        margin=float(margin),
        n_blocks=int(distribution.shape[0]),
    )


def build_null_model(
    training_features: np.ndarray,
    reference: DriftReference,
    recent_windows: int,
    statistic: str = "ks",
    quantile: float = 0.99,
    margin: float = 1.0,
    step: int | None = None,
    psi_epsilon: float = 1e-6,
) -> NullModel:
    """Calibre le seuil de chaque feature sur l'entrainement du modele courant.

    `step` est le pas de glissement des blocs, en fenetres. Il ne change pas la
    nature de la mesure, seulement le nombre de blocs visites : un pas de R / 10
    donne une dizaine de blocs par duree R, ce qui suffit pour un quantile a
    0,99 sur un entrainement de plusieurs dizaines de milliers de fenetres. Par
    defaut `max(1, recent_windows // 10)`.
    """
    distribution = null_distribution(
        training_features=training_features,
        reference=reference,
        recent_windows=recent_windows,
        statistic=statistic,
        step=step,
        psi_epsilon=psi_epsilon,
    )
    return thresholds_from_distribution(distribution, reference, quantile, margin)
