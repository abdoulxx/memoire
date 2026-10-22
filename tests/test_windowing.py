"""Tests du fenetrage, sur des cas calcules a la main.

Serie de reference : 5 points, 1 metrique, valeurs 0 a 4, w = 3.
Fenetres attendues : [0,1,2], [1,2,3], [2,3,4], donc end_index = [2, 3, 4].
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.windowing import make_windows

SERIE = np.arange(5, dtype=float).reshape(5, 1)
CONFIG = {
    "window_size": 3,
    "stride": 1,
    "aggregates": ["last", "mean", "std", "max"],
    "labeling": {"min_anomalous_points": 1},
}


def test_nombre_et_position_des_fenetres() -> None:
    windowed = make_windows(SERIE, CONFIG)
    assert len(windowed) == 3
    assert windowed.end_index.tolist() == [2, 3, 4]


def test_valeurs_des_agregats() -> None:
    windowed = make_windows(SERIE, CONFIG)
    # Une metrique x quatre agregats, dans l'ordre de la configuration.
    assert windowed.feature_names == ["m00_last", "m00_mean", "m00_std", "m00_max"]
    ecart_type = np.sqrt(2.0 / 3.0)  # ecart-type de population de [k, k+1, k+2]
    attendu = np.array(
        [
            [2.0, 1.0, ecart_type, 2.0],
            [3.0, 2.0, ecart_type, 3.0],
            [4.0, 3.0, ecart_type, 4.0],
        ]
    )
    np.testing.assert_allclose(windowed.X, attendu)


def test_label_de_fenetre() -> None:
    """Une fenetre est positive si elle contient au moins un point anormal."""
    label = np.array([1, 0, 0, 0, 0])
    windowed = make_windows(SERIE, CONFIG, label=label)
    assert windowed.y is not None
    assert windowed.y.tolist() == [1, 0, 0]


def test_label_seuil_configurable() -> None:
    """Avec min_anomalous_points = 2, une fenetre a un seul point anormal reste negative."""
    config = {**CONFIG, "labeling": {"min_anomalous_points": 2}}
    label = np.array([1, 1, 0, 0, 0])
    windowed = make_windows(SERIE, config, label=label)
    assert windowed.y is not None
    assert windowed.y.tolist() == [1, 0, 0]


def test_stride() -> None:
    config = {**CONFIG, "stride": 2}
    windowed = make_windows(SERIE, config)
    assert windowed.end_index.tolist() == [2, 4]
    np.testing.assert_allclose(windowed.X[:, 0], [2.0, 4.0])


def test_multivarie() -> None:
    """Les features sont groupees par agregat, et nommees d'apres les metriques."""
    serie = np.column_stack([np.arange(5.0), np.arange(5.0) * 10])
    windowed = make_windows(serie, CONFIG, metric_names=["cpu", "ram"])
    assert windowed.n_features == 2 * 4
    assert windowed.feature_names[:2] == ["cpu_last", "ram_last"]
    np.testing.assert_allclose(windowed.X[0], [2.0, 20.0, 1.0, 10.0, *windowed.X[0, 4:]])


def test_erreurs() -> None:
    with pytest.raises(ValueError, match="trop courte"):
        make_windows(SERIE, {**CONFIG, "window_size": 99})
    with pytest.raises(ValueError, match="Agregats inconnus"):
        make_windows(SERIE, {**CONFIG, "aggregates": ["median"]})
    with pytest.raises(ValueError, match="etiquettes"):
        make_windows(SERIE, CONFIG, label=np.array([0, 1]))
