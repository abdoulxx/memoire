"""Figures produites a partir des resultats enregistres.

Les figures sont generees depuis les traces `results/runs/.../traces.npz`, jamais
en relancant un calcul : une figure du memoire correspond donc toujours a une
execution identifiee par son `run_id`.

Sortie : PNG 300 dpi dans `docs/figures/`, legendes en francais, sans nom
d'hote ni adresse (regle 4.3 du cadrage).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif : execution en ligne de commande

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.evaluation.metrics import duration_hours  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
DPI = 300


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Plages contigues ou `mask` est vrai, en indices de position."""
    padded = np.concatenate(([0], mask.astype(np.int64), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def plot_scores(
    traces_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    sampling_interval_seconds: int = 60,
    window: tuple[int, int] | None = None,
) -> Path:
    """Score d'anomalie dans le temps, avec seuil, anomalies etiquetees et alertes.

    `window` restreint la figure a une plage de positions, pour une vue zoomee.
    """
    traces = np.load(Path(traces_path))
    scores = traces["scores"]
    alerts = traces["alerts"].astype(bool)
    thresholds = traces["thresholds"]
    y_true = traces["y_true"].astype(bool)

    if window is not None:
        start, end = window
        sl = slice(start, end)
        scores, alerts, thresholds, y_true = scores[sl], alerts[sl], thresholds[sl], y_true[sl]
        offset = start
    else:
        offset = 0

    # Axe des temps en heures depuis le debut du flux.
    hours = (np.arange(len(scores)) + offset) * sampling_interval_seconds / 3600.0

    figure, axes = plt.subplots(figsize=(11, 4.2), constrained_layout=True)

    # Anomalies etiquetees : bandes de fond.
    for first, last in _spans(y_true):
        axes.axvspan(
            hours[first],
            hours[last],
            color="#d62728",
            alpha=0.16,
            linewidth=0,
            label="_nolegend_",
        )

    axes.plot(hours, scores, linewidth=0.5, color="#1f77b4", label="Score d'anomalie")
    axes.plot(
        hours,
        thresholds,
        linewidth=1.0,
        color="#2ca02c",
        linestyle="--",
        label="Seuil d'alerte",
    )
    if alerts.any():
        axes.scatter(
            hours[alerts],
            scores[alerts],
            s=2.5,
            color="#ff7f0e",
            zorder=3,
            label=f"Alertes ({int(alerts.sum())})",
        )

    # Entree de legende pour les bandes d'anomalies.
    axes.axvspan(np.nan, np.nan, color="#d62728", alpha=0.16, label="Anomalie etiquetee")

    axes.set_xlabel("Temps depuis le debut du flux (heures)")
    axes.set_ylabel("Score d'anomalie")
    axes.set_xlim(hours[0], hours[-1])
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9, ncols=2)
    if title:
        axes.set_title(title, fontsize=11)

    destination = Path(output_path) if output_path else FIGURES_DIR / "scores.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def plot_alert_rate_profile(
    traces_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    sampling_interval_seconds: int = 60,
    smoothing_windows: int = 500,
) -> Path:
    """Taux d'alertes glissant : met en evidence une derive du comportement du modele.

    Un taux d'alertes qui s'eloigne durablement de sa valeur de calibrage est le
    signal exploite par la strategie proposee (P) pour suspecter une derive.
    """
    traces = np.load(Path(traces_path))
    alerts = traces["alerts"].astype(np.float64)
    y_true = traces["y_true"].astype(bool)

    kernel = np.ones(smoothing_windows) / smoothing_windows
    smoothed = np.convolve(alerts, kernel, mode="valid")
    hours = (np.arange(len(smoothed)) + smoothing_windows // 2) * sampling_interval_seconds / 3600.0

    figure, axes = plt.subplots(figsize=(11, 3.4), constrained_layout=True)
    for first, last in _spans(y_true[: len(smoothed)]):
        axes.axvspan(hours[first], hours[last], color="#d62728", alpha=0.16, linewidth=0)

    axes.plot(hours, smoothed * 100, linewidth=1.0, color="#1f77b4")
    axes.set_xlabel("Temps depuis le debut du flux (heures)")
    axes.set_ylabel(f"Taux d'alertes\n(moyenne sur {smoothing_windows} fenetres, %)")
    axes.set_xlim(hours[0], hours[-1])
    axes.grid(alpha=0.25, linewidth=0.5)
    if title:
        axes.set_title(title, fontsize=11)

    destination = Path(output_path) if output_path else FIGURES_DIR / "taux_alertes.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def describe_traces(
    traces_path: str | Path, sampling_interval_seconds: int = 60
) -> dict[str, float]:
    """Quelques chiffres de controle sur une trace, pour accompagner la figure."""
    traces = np.load(Path(traces_path))
    alerts = traces["alerts"].astype(bool)
    y_true = traces["y_true"].astype(bool)
    hours = duration_hours(len(alerts), 1, sampling_interval_seconds)
    return {
        "n_windows": float(len(alerts)),
        "duration_hours": hours,
        "alert_rate_percent": float(alerts.mean() * 100),
        "positive_rate_percent": float(y_true.mean() * 100),
    }
