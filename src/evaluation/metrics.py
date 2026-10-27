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
    """Cout de la strategie. Vaut zero pour un modele fige (B0).

    `retrain_seconds` ne compte que l'entrainement des challengers, qui est la
    grandeur demandee au §7. `validation_seconds` compte le scoring du
    challenger et du champion sur la fenetre reservee V : c'est un cout reel de
    la strategie proposee, et le compter a part permet de dire ce que la
    validation coute, separement de ce qu'elle apporte.

    Le scoring fantome (D16) n'apparait pas ici : c'est un instrument de mesure
    hors ligne, pas une piece du pipeline.
    """

    n_retrains: int = 0
    n_promoted: int = 0
    n_rejected: int = 0
    retrain_seconds: float = 0.0
    validation_seconds: float = 0.0


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
class AdaptationMetrics:
    """Famille << adaptation >> du §7 : avant / apres la derive, et recuperation.

    Toutes ces grandeurs valent None pour le scenario `none`, ou aucune derive
    n'est injectee : il n'y a alors pas d'instant de reference, et inventer un
    point de coupure arbitraire produirait des colonnes que rien ne justifie.
    """

    drift_window: int | None = None
    n_windows_before: int | None = None
    n_windows_after: int | None = None
    f1_before: float | None = None
    f1_after: float | None = None
    precision_before: float | None = None
    precision_after: float | None = None
    recall_before: float | None = None
    recall_after: float | None = None
    # Temps de recuperation : None si jamais recupere OU si le critere est
    # indefini (voir `recovery_status`).
    recovery_windows: int | None = None
    recovery_minutes: float | None = None
    f1_before_level: float | None = None
    f1_target: float | None = None
    recovery_status: str = "non_applicable"


@dataclass
class DriftMonitorMetrics:
    """Famille << specificite de la surveillance >> du §7.

    `n_confirmations` et `first_confirmation_window` sont lus dans le journal
    d'evenements : le harnais y inscrit ce que la strategie a declare, et non ce
    qu'on aimerait qu'elle ait declare.

    `false_confirmations` est la metrique de D19. Elle n'a de sens que sur le
    scenario `none` : la ou aucune derive n'est injectee, toute confirmation est
    fausse. Elle vaut None ailleurs, parce qu'une confirmation sur un flux
    reellement derive n'est pas une erreur.
    """

    n_signals: int = 0
    n_confirmations: int = 0
    first_confirmation_window: int | None = None
    detection_delay_windows: int | None = None
    detection_delay_minutes: float | None = None
    false_confirmations: int | None = None


def segment_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start: int,
    stop: int,
    zero_division: float = 0.0,
) -> float:
    """F1 par fenetre sur le segment [start, stop). Renvoie NaN si le segment est vide."""
    if stop <= start:
        return float("nan")
    return window_metrics(y_true[start:stop], y_pred[start:stop], zero_division).f1


def rolling_f1_after(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drift_window: int,
    rolling_windows: int,
    zero_division: float = 0.0,
) -> np.ndarray:
    """F1 glissant apres la derive, calcule sur des fenetres **purement** post-derive.

    A l'indice i >= drift_window, le F1 est calcule sur
    [max(drift_window, i - R + 1), i]. La fenetre glissante ne melange donc
    jamais les deux regimes : un F1 glissant classique, a cheval sur l'instant
    de derive, remonterait mecaniquement au seul fait qu'il contient encore des
    fenetres d'avant, et ferait conclure a une recuperation qui n'a pas eu lieu.

    En contrepartie, les premieres valeurs portent sur peu de fenetres et sont
    bruitees : c'est a quoi sert `recovery.min_windows` dans la configuration.
    """
    n = len(y_true)
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    # Sommes cumulees : le F1 de tout segment se lit alors en temps constant.
    tp = np.concatenate(([0], np.cumsum(true & pred)))
    fp = np.concatenate(([0], np.cumsum(~true & pred)))
    fn = np.concatenate(([0], np.cumsum(true & ~pred)))

    values = np.full(n, np.nan)
    for i in range(drift_window, n):
        start = max(drift_window, i - rolling_windows + 1)
        stop = i + 1
        seg_tp = tp[stop] - tp[start]
        seg_fp = fp[stop] - fp[start]
        seg_fn = fn[stop] - fn[start]
        precision = _ratio(seg_tp, seg_tp + seg_fp, zero_division)
        recall = _ratio(seg_tp, seg_tp + seg_fn, zero_division)
        values[i] = _ratio(2 * precision * recall, precision + recall, zero_division)
    return values


def adaptation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drift_window: int | None,
    stride: int,
    sampling_interval_seconds: int,
    recovery_config: dict[str, Any],
    zero_division: float = 0.0,
) -> AdaptationMetrics:
    """F1 avant / apres la derive, et temps de recuperation.

    Le niveau pre-derive est le F1 glissant **juste avant** t_d, calcule sur les
    R fenetres qui le precedent : c'est la definition du §7, << son niveau
    pre-derive >>.

    Piege a ne pas se tendre a soi-meme : si ce niveau est nul ou quasi nul, le
    critere << revenir a 90 % du niveau pre-derive >> est satisfait par n'importe
    quoi, zero compris. OBS09 a mesure que le F1 d'avant-derive est nul sur 6
    des 16 machines de mise au point. Dans ce cas la metrique est declaree
    indefinie (`recovery_status = "niveau_pre_derive_nul"`) et vaut None, au
    lieu d'annoncer une recuperation instantanee qui ne voudrait rien dire.
    """
    if drift_window is None:
        return AdaptationMetrics()

    n = len(y_true)
    rolling = int(recovery_config.get("rolling_windows", 500))
    ratio = float(recovery_config.get("f1_ratio", 0.90))
    min_windows = int(recovery_config.get("min_windows", 50))
    floor = float(recovery_config.get("min_pre_drift_f1", 0.05))

    before = window_metrics(y_true[:drift_window], y_pred[:drift_window], zero_division)
    after = window_metrics(y_true[drift_window:], y_pred[drift_window:], zero_division)

    metrics = AdaptationMetrics(
        drift_window=int(drift_window),
        n_windows_before=int(drift_window),
        n_windows_after=int(n - drift_window),
        f1_before=before.f1,
        f1_after=after.f1,
        precision_before=before.precision,
        precision_after=after.precision,
        recall_before=before.recall,
        recall_after=after.recall,
    )

    start = max(0, drift_window - rolling)
    if drift_window - start < min_windows:
        metrics.recovery_status = "historique_pre_derive_trop_court"
        return metrics

    level = segment_f1(y_true, y_pred, start, drift_window, zero_division)
    metrics.f1_before_level = float(level)
    if not np.isfinite(level) or level <= floor:
        metrics.recovery_status = "niveau_pre_derive_nul"
        return metrics

    target = ratio * float(level)
    metrics.f1_target = target
    values = rolling_f1_after(y_true, y_pred, drift_window, rolling, zero_division)
    # Les `min_windows` premieres fenetres post-derive portent sur trop peu de
    # points : on ne les laisse pas declarer une recuperation.
    searchable = np.arange(drift_window + min_windows - 1, n)
    reached = searchable[values[searchable] >= target] if searchable.size else np.empty(0, int)
    if reached.size == 0:
        metrics.recovery_status = "jamais_recupere"
        return metrics

    recovery = int(reached[0] - drift_window)
    metrics.recovery_windows = recovery
    metrics.recovery_minutes = recovery * stride * sampling_interval_seconds / 60.0
    metrics.recovery_status = "recupere"
    return metrics


def drift_monitor_metrics(
    event_log: list[dict[str, Any]],
    drift_window: int | None,
    stride: int,
    sampling_interval_seconds: int,
) -> DriftMonitorMetrics:
    """Compte les signaux et les confirmations de derive declares par la strategie."""
    signals = [e for e in event_log if e.get("event") == "drift_signal"]
    confirmations = [e for e in event_log if e.get("event") == "drift_confirmed"]

    metrics = DriftMonitorMetrics(
        n_signals=len(signals),
        n_confirmations=len(confirmations),
    )
    if confirmations:
        first = confirmations[0].get("details", {}).get("position")
        metrics.first_confirmation_window = None if first is None else int(first)

    if drift_window is None:
        # Scenario `none` : toute confirmation est fausse (D19).
        metrics.false_confirmations = len(confirmations)
        return metrics

    if metrics.first_confirmation_window is not None:
        delay = metrics.first_confirmation_window - int(drift_window)
        metrics.detection_delay_windows = delay
        metrics.detection_delay_minutes = delay * stride * sampling_interval_seconds / 60.0
    return metrics


@dataclass
class EvaluationResult:
    """Toutes les metriques d'une execution, pretes pour `results/summary.csv`."""

    windows: WindowMetrics
    events: EventMetrics
    false_alerts_per_hour: float
    duration_hours: float
    cost: CostMetrics
    adaptation: AdaptationMetrics = field(default_factory=AdaptationMetrics)
    drift: DriftMonitorMetrics = field(default_factory=DriftMonitorMetrics)

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
        row.update(asdict(self.adaptation))
        row.update({f"drift_{key}": value for key, value in asdict(self.drift).items()})
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
    drift_window: int | None = None,
    event_log: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    """Calcule toutes les metriques d'une execution.

    `drift_window` est l'indice de la premiere fenetre derivee, ou None pour le
    scenario `none`. `event_log` est le journal du rejeu, d'ou sont lues les
    confirmations de derive declarees par la strategie.
    """
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
        adaptation=adaptation_metrics(
            y_true,
            y_pred,
            drift_window,
            stride,
            sampling_interval_seconds,
            evaluation_config.get("recovery", {}) or {},
            zero_division,
        ),
        drift=drift_monitor_metrics(
            event_log or [],
            drift_window,
            stride,
            sampling_interval_seconds,
        ),
    )
