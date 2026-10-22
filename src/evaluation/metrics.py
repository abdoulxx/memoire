"""Metriques d'evaluation (memoire, §7).

Trois familles, toutes calculees sur la meme verite terrain :

1. **Qualite de detection par fenetre** : precision, rappel, F1 sur les fenetres,
   une fenetre etant positive si elle contient au moins un point anormal.
2. **Qualite de detection par evenement** : part des anomalies ayant recu au moins
   une alerte, et delai avant la premiere alerte.
3. **Fausses alertes par heure** : faux positifs de la matrice de confusion,
   ramenes a la duree reelle couverte.

Conventions, fixees dans `configs/evaluation.yaml` :
- une alerte emise a l'indice `end_index` est attribuee a l'evenement
  `[debut, fin + tolerance]` qui la contient ;
- avec `tolerance = window_size - 1`, la zone d'attribution d'un evenement
  coincide exactement avec les fenetres etiquetees positives a cause de lui, ce
  qui rend les trois familles mutuellement coherentes ;
- un denominateur nul donne `zero_division.value` (0.0 par defaut).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from datasets.smd.loader import AnomalyEvent


@dataclass
class WindowMetrics:
    """Matrice de confusion et metriques par fenetre."""

    n_windows: int
    n_positive_windows: int
    n_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass
class EventMetrics:
    """Metriques par evenement d'anomalie."""

    n_events: int
    n_events_detected: int
    event_recall: float
    detection_delay_mean_minutes: float
    detection_delay_median_minutes: float
    # Delai par evenement, en minutes ; None si l'evenement n'a pas ete detecte.
    delays_minutes: list[float | None] = field(default_factory=list)


@dataclass
class CostMetrics:
    """Cout de la strategie. Vaut zero pour un modele fige (B0)."""

    n_retrains: int = 0
    n_promoted: int = 0
    n_rejected: int = 0
    retrain_seconds: float = 0.0


def _ratio(numerator: float, denominator: float, zero_division: float) -> float:
    return zero_division if denominator == 0 else numerator / denominator


def window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> WindowMetrics:
    """Precision, rappel et F1 par fenetre."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{y_true.shape} etiquettes pour {y_pred.shape} predictions.")
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)

    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    tn = int(np.count_nonzero(~true & ~pred))

    precision = _ratio(tp, tp + fp, zero_division)
    recall = _ratio(tp, tp + fn, zero_division)
    f1 = _ratio(2 * precision * recall, precision + recall, zero_division)

    return WindowMetrics(
        n_windows=int(len(y_true)),
        n_positive_windows=int(np.count_nonzero(true)),
        n_alerts=int(np.count_nonzero(pred)),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def event_metrics(
    events: list[AnomalyEvent],
    y_pred: np.ndarray,
    end_index: np.ndarray,
    tolerance_points: int,
    sampling_interval_seconds: int,
    zero_division: float = 0.0,
) -> EventMetrics:
    """Rappel par evenement et delai de detection.

    Le delai est compte du premier point de l'anomalie a la premiere alerte qui
    lui est attribuable. La moyenne et la mediane ne portent que sur les
    evenements detectes : un evenement manque n'a pas de delai, il est compte
    dans le rappel.
    """
    if len(y_pred) != len(end_index):
        raise ValueError(f"{len(y_pred)} predictions pour {len(end_index)} indices.")

    alert_positions = end_index[y_pred.astype(bool)]
    minutes = sampling_interval_seconds / 60.0

    delays: list[float | None] = []
    for event in events:
        upper = event.end + tolerance_points
        inside = alert_positions[(alert_positions >= event.start) & (alert_positions <= upper)]
        if inside.size == 0:
            delays.append(None)
        else:
            delays.append(float(inside.min() - event.start) * minutes)

    detected = [d for d in delays if d is not None]
    return EventMetrics(
        n_events=len(events),
        n_events_detected=len(detected),
        event_recall=_ratio(len(detected), len(events), zero_division),
        detection_delay_mean_minutes=float(np.mean(detected)) if detected else zero_division,
        detection_delay_median_minutes=float(np.median(detected)) if detected else zero_division,
        delays_minutes=delays,
    )


def duration_hours(n_windows: int, stride: int, sampling_interval_seconds: int) -> float:
    """Duree couverte par une serie de fenetres, en heures."""
    return n_windows * stride * sampling_interval_seconds / 3600.0


def false_alerts_per_hour(
    false_positives: int,
    n_windows: int,
    stride: int,
    sampling_interval_seconds: int,
    zero_division: float = 0.0,
) -> float:
    """Faux positifs ramenes a la duree reelle couverte."""
    hours = duration_hours(n_windows, stride, sampling_interval_seconds)
    return _ratio(false_positives, hours, zero_division)


@dataclass
class EvaluationResult:
    """Toutes les metriques d'une execution, pretes pour `results/summary.csv`."""

    windows: WindowMetrics
    events: EventMetrics
    false_alerts_per_hour: float
    duration_hours: float
    cost: CostMetrics

    def to_row(self) -> dict[str, Any]:
        """Aplatit les metriques en une ligne de tableau (une colonne par metrique)."""
        row: dict[str, Any] = {}
        row.update(asdict(self.windows))
        events = asdict(self.events)
        events.pop("delays_minutes")
        row.update(events)
        row["false_alerts_per_hour"] = self.false_alerts_per_hour
        row["duration_hours"] = self.duration_hours
        row.update({f"cost_{key}": value for key, value in asdict(self.cost).items()})
        return row


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    events: list[AnomalyEvent],
    end_index: np.ndarray,
    window_size: int,
    stride: int,
    sampling_interval_seconds: int,
    evaluation_config: dict[str, Any],
    cost: CostMetrics | None = None,
) -> EvaluationResult:
    """Calcule toutes les metriques d'une execution."""
    zero_division = float(evaluation_config.get("zero_division", {}).get("value", 0.0))

    tolerance = evaluation_config.get("event", {}).get("tolerance_points")
    tolerance_points = window_size - 1 if tolerance is None else int(tolerance)

    windows = window_metrics(y_true, y_pred, zero_division)
    return EvaluationResult(
        windows=windows,
        events=event_metrics(
            events,
            y_pred,
            end_index,
            tolerance_points,
            sampling_interval_seconds,
            zero_division,
        ),
        false_alerts_per_hour=false_alerts_per_hour(
            windows.false_positives,
            len(y_true),
            stride,
            sampling_interval_seconds,
            zero_division,
        ),
        duration_hours=duration_hours(len(y_true), stride, sampling_interval_seconds),
        cost=cost or CostMetrics(),
    )
