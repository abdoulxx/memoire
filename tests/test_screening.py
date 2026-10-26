"""Tests du criblage des machines SMD.

Tous les cas sont calcules a la main. Le dernier test encode l'erreur de mesure
qui avait fausse OBS03 (docs/decisions.md) : une moyenne glissante centree, et le
comptage de toutes les alertes au lieu des seules fausses alertes, datent le
debut de la derive trop tot.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.screening import (
    causal_rolling_mean,
    confirmed_onset,
    false_alerts,
    first_crossing,
    slice_profile,
)


def _centered_rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne centree, telle qu'elle avait ete utilisee a tort pour OBS03."""
    means = np.convolve(values.astype(np.float64), np.ones(window) / window, mode="valid")
    return means, np.arange(len(means)) + window // 2


def test_moyenne_causale_valeurs() -> None:
    values = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    means, indices = causal_rolling_mean(values, 2)
    # Moyennes des paires (0,1), (1,1), (1,0), (0,0), datees sur leur 2e element.
    assert means.tolist() == [0.5, 1.0, 0.5, 0.0]
    assert indices.tolist() == [1, 2, 3, 4]


def test_moyenne_causale_n_anticipe_pas() -> None:
    """Propriete essentielle : une valeur a l'indice i n'influence aucune moyenne
    datee avant i. C'est ce qui manquait a la mesure initiale d'OBS03."""
    window = 5
    base = np.zeros(20)
    means_base, indices = causal_rolling_mean(base, window)

    perturbee = base.copy()
    perturbee[12] = 1.0
    means_perturbee, _ = causal_rolling_mean(perturbee, window)

    changed = np.flatnonzero(means_base != means_perturbee)
    assert indices[changed].min() == 12

    # La version centree, elle, modifie des moyennes datees avant l'evenement :
    # elle anticipe exactement de window // 2 (250 fenetres, soit 4,2 h, pour OBS03).
    centered_base, centered_indices = _centered_rolling_mean(base, window)
    centered_perturbee, _ = _centered_rolling_mean(perturbee, window)
    centered_changed = np.flatnonzero(centered_base != centered_perturbee)
    assert centered_indices[centered_changed].min() == 12 - window // 2


def test_moyenne_causale_fenetre_invalide() -> None:
    with pytest.raises(ValueError):
        causal_rolling_mean(np.zeros(3), 0)
    with pytest.raises(ValueError):
        causal_rolling_mean(np.zeros(3), 4)


def test_fausses_alertes() -> None:
    alerts = np.array([1, 1, 0, 0, 1])
    y = np.array([0, 1, 0, 1, 1])
    # Seule la 1re alerte est fausse : la 2e et la 5e tombent sur des fenetres positives.
    assert false_alerts(alerts, y).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_fausses_alertes_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        false_alerts(np.array([1, 0]), np.array([0]))


def test_premier_franchissement() -> None:
    means = np.array([0.0, 0.02, 0.06, 0.01])
    indices = np.array([3, 4, 5, 6])
    assert first_crossing(means, indices, 0.05) == 5
    assert first_crossing(means, indices, 0.5) is None


def test_confirmation_exige_la_persistance() -> None:
    """Une rafale courte ne confirme rien ; une plage assez longue confirme, et
    l'instant retourne est celui ou la persistance est atteinte."""
    means = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    indices = np.arange(len(means))

    assert confirmed_onset(means, indices, 0.5, 2) == 2  # 1re rafale, longueur 2
    assert confirmed_onset(means, indices, 0.5, 3) == 6  # 2e rafale, confirmee a son 3e point
    assert confirmed_onset(means, indices, 0.5, 5) is None  # aucune plage de 5
    with pytest.raises(ValueError):
        confirmed_onset(means, indices, 0.5, 0)


def test_profil_par_tranches() -> None:
    # Intervalle de 3600 s : 1 point = 1 h, donc 4 tranches de 2 points.
    rate = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
    assert slice_profile(rate, 3600, 2.0, 10) == [50.0, 0.0, 100.0, 50.0]
    assert slice_profile(rate, 3600, 2.0, 2) == [50.0, 0.0]
    # Intervalle de 1800 s : 1 point = 30 min, donc 2 tranches de 4 points.
    assert slice_profile(rate, 1800, 2.0, 10) == [25.0, 75.0]


def test_non_regression_obs03() -> None:
    """Flux synthetique reproduisant le piege : une anomalie etiquetee tot, une
    derive tardive. La mesure retenue (causale, fausses alertes seules) date la
    derive apres son debut reel ; la mesure initiale la datait avant l'anomalie.
    """
    n, window, level, persistance = 2000, 100, 0.05, 20
    anomalie = slice(200, 260)
    debut_derive = 1200

    y = np.zeros(n, dtype=np.int64)
    y[anomalie] = 1
    alerts = np.zeros(n, dtype=np.int64)
    alerts[anomalie] = 1  # detections reussies
    alerts[debut_derive:] = 1  # degradation durable, sur fenetres negatives

    # Mesure retenue : la derive est datee apres son debut reel, jamais avant.
    means, indices = causal_rolling_mean(false_alerts(alerts, y), window)
    onset = confirmed_onset(means, indices, level, persistance)
    assert onset is not None
    assert onset >= debut_derive

    # Mesure initiale : moyenne centree sur toutes les alertes. Elle confirme une
    # "derive" pendant l'anomalie etiquetee, bien avant le debut reel.
    centered, centered_indices = _centered_rolling_mean(alerts.astype(np.float64), window)
    faux_onset = confirmed_onset(centered, centered_indices, level, persistance)
    assert faux_onset is not None
    assert faux_onset < anomalie.stop
