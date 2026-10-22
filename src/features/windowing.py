"""Fenetrage et calcul des features.

Une fenetre couvre les `window_size` derniers points, bornes incluses, et porte
l'indice de son dernier point (`end_index`) : c'est l'instant auquel la decision
est prise en mode prequentiel. Les `window_size - 1` premiers points ne forment
pas de fenetre complete et sont donc ecartes.

Pour chaque metrique, on calcule les agregats demandes par la configuration, ce
qui donne `n_metriques x n_agregats` features par fenetre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

AGGREGATE_FUNCTIONS: dict[str, object] = {
    "last": None,  # traite a part : valeur du dernier point de la fenetre
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
}


@dataclass(frozen=True)
class WindowedData:
    """Fenetres construites sur une serie multivariee.

    `X` a la forme (n_fenetres, n_metriques * n_agregats).
    `y` vaut None quand la serie n'est pas etiquetee (moitie `train` du SMD).
    `end_index[i]` est l'indice, dans la serie d'origine, du dernier point de la fenetre i.
    """

    X: np.ndarray
    y: np.ndarray | None
    end_index: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def positive_rate(self) -> float:
        if self.y is None:
            raise ValueError("Serie non etiquetee.")
        return float(self.y.mean())


def _validate(window_size: int, stride: int, aggregates: list[str], n_points: int) -> None:
    if window_size < 1:
        raise ValueError("window_size doit valoir au moins 1.")
    if stride < 1:
        raise ValueError("stride doit valoir au moins 1.")
    if n_points < window_size:
        raise ValueError(f"Serie de {n_points} points, trop courte pour w = {window_size}.")
    unknown = [a for a in aggregates if a not in AGGREGATE_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Agregats inconnus : {unknown}. Disponibles : {sorted(AGGREGATE_FUNCTIONS)}"
        )


def make_windows(
    series: np.ndarray,
    config: dict,
    label: np.ndarray | None = None,
    metric_names: list[str] | None = None,
) -> WindowedData:
    """Construit les fenetres et leurs features a partir d'une serie (n_points, n_metriques)."""
    if series.ndim != 2:
        raise ValueError(f"Serie de forme {series.shape}, (n_points, n_metriques) attendu.")

    window_size = int(config["window_size"])
    stride = int(config["stride"])
    aggregates = list(config["aggregates"])
    min_anomalous = int(config.get("labeling", {}).get("min_anomalous_points", 1))
    n_points, n_metrics = series.shape
    _validate(window_size, stride, aggregates, n_points)

    # (n_fenetres_possibles, n_metriques, window_size)
    windows = sliding_window_view(series, window_size, axis=0)[::stride]
    end_index = np.arange(window_size - 1, n_points, stride)

    blocks: list[np.ndarray] = []
    names: list[str] = []
    labels = metric_names or [f"m{i:02d}" for i in range(n_metrics)]
    for aggregate in aggregates:
        if aggregate == "last":
            block = windows[:, :, -1]
        else:
            function = AGGREGATE_FUNCTIONS[aggregate]
            block = function(windows, axis=2)  # type: ignore[operator]
        blocks.append(block)
        names.extend(f"{name}_{aggregate}" for name in labels)

    X = np.concatenate(blocks, axis=1).astype(np.float64, copy=False)

    y: np.ndarray | None = None
    if label is not None:
        if len(label) != n_points:
            raise ValueError(f"{n_points} points pour {len(label)} etiquettes.")
        counts = sliding_window_view(label.astype(np.int64), window_size)[::stride].sum(axis=1)
        y = (counts >= min_anomalous).astype(np.int64)

    return WindowedData(X=X, y=y, end_index=end_index, feature_names=names)
