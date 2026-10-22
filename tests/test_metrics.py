"""Tests des metriques, sur des cas calcules a la main.

Ces tests sont la garantie que les chiffres du memoire sont justes : une erreur
de metrique est invisible a l'oeil nu sur les resultats.
"""

from __future__ import annotations

import numpy as np

from datasets.smd.loader import AnomalyEvent
from src.config import load_config
from src.evaluation.metrics import (
    CostMetrics,
    evaluate,
    event_metrics,
    false_alerts_per_hour,
    window_metrics,
)
from src.features.windowing import make_windows

EVALUATION_CONFIG = load_config("evaluation.yaml")


# --- Famille 1 : metriques par fenetre -------------------------------------


def test_matrice_de_confusion_calculee_a_la_main() -> None:
    """y_true = [0,0,1,1,0], y_pred = [0,1,1,0,0] -> TP=1, FP=1, FN=1, TN=2."""
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    metrics = window_metrics(y_true, y_pred)

    assert (metrics.true_positives, metrics.false_positives) == (1, 1)
    assert (metrics.false_negatives, metrics.true_negatives) == (1, 2)
    assert metrics.n_windows == 5
    assert metrics.n_positive_windows == 2
    assert metrics.n_alerts == 2
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_detecteur_parfait() -> None:
    y_true = np.array([0, 1, 1, 0])
    metrics = window_metrics(y_true, y_true.copy())
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    assert metrics.false_positives == 0


def test_aucune_alerte_donne_zero_et_non_une_erreur() -> None:
    """Convention zero_division : pas d'alerte -> precision, rappel et F1 nuls."""
    metrics = window_metrics(np.array([0, 1, 1]), np.zeros(3, dtype=int))
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)


def test_alertes_partout() -> None:
    """Rappel parfait, precision egale a la proportion de fenetres positives."""
    metrics = window_metrics(np.array([0, 0, 0, 1]), np.ones(4, dtype=int))
    assert metrics.recall == 1.0
    assert metrics.precision == 0.25


# --- Famille 2 : metriques par evenement -----------------------------------


def test_delai_de_detection() -> None:
    """Evenement [5,8], alerte a l'indice 7 : delai de 2 minutes."""
    events = [AnomalyEvent(5, 8)]
    end_index = np.arange(12)
    y_pred = np.zeros(12, dtype=int)
    y_pred[7] = 1

    metrics = event_metrics(
        events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=60
    )
    assert metrics.n_events_detected == 1
    assert metrics.event_recall == 1.0
    assert metrics.delays_minutes == [2.0]
    assert metrics.detection_delay_mean_minutes == 2.0


def test_premiere_alerte_retenue() -> None:
    """Avec plusieurs alertes dans l'evenement, le delai retient la premiere."""
    events = [AnomalyEvent(5, 8)]
    end_index = np.arange(12)
    y_pred = np.zeros(12, dtype=int)
    y_pred[[6, 7, 8]] = 1

    metrics = event_metrics(
        events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=60
    )
    assert metrics.delays_minutes == [1.0]


def test_alerte_hors_zone_ne_compte_pas() -> None:
    """Une alerte avant le debut ou apres fin + tolerance ne detecte pas l'evenement."""
    events = [AnomalyEvent(5, 8)]
    end_index = np.arange(20)
    avant = np.zeros(20, dtype=int)
    avant[4] = 1
    apres = np.zeros(20, dtype=int)
    apres[11] = 1  # fin 8 + tolerance 2 = 10, donc trop tard

    for y_pred in (avant, apres):
        metrics = event_metrics(
            events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=60
        )
        assert metrics.n_events_detected == 0
        assert metrics.event_recall == 0.0
        assert metrics.delays_minutes == [None]
        assert metrics.detection_delay_mean_minutes == 0.0


def test_tolerance_bornes_incluses() -> None:
    """Une alerte exactement a fin + tolerance detecte encore l'evenement."""
    events = [AnomalyEvent(5, 8)]
    end_index = np.arange(20)
    y_pred = np.zeros(20, dtype=int)
    y_pred[10] = 1

    metrics = event_metrics(
        events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=60
    )
    assert metrics.delays_minutes == [5.0]


def test_rappel_par_evenement_sur_deux_evenements() -> None:
    """Un evenement detecte sur deux : rappel de 0,5, moyenne sur le seul detecte."""
    events = [AnomalyEvent(5, 8), AnomalyEvent(20, 22)]
    end_index = np.arange(30)
    y_pred = np.zeros(30, dtype=int)
    y_pred[[7, 27]] = 1  # 27 est hors de [20, 24]

    metrics = event_metrics(
        events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=60
    )
    assert metrics.n_events == 2
    assert metrics.n_events_detected == 1
    assert metrics.event_recall == 0.5
    assert metrics.delays_minutes == [2.0, None]
    assert metrics.detection_delay_mean_minutes == 2.0


def test_delai_tient_compte_de_l_intervalle() -> None:
    """Le delai est en minutes : a 30 s par point, 2 points valent 1 minute."""
    events = [AnomalyEvent(5, 8)]
    end_index = np.arange(12)
    y_pred = np.zeros(12, dtype=int)
    y_pred[7] = 1

    metrics = event_metrics(
        events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=30
    )
    assert metrics.delays_minutes == [1.0]


def test_end_index_non_trivial() -> None:
    """Avec un pas de 2, les alertes portent les indices reels de la serie."""
    events = [AnomalyEvent(5, 8)]
    end_index = np.arange(0, 20, 2)  # 0, 2, 4, 6, 8, ...
    y_pred = np.zeros(len(end_index), dtype=int)
    y_pred[3] = 1  # end_index = 6

    metrics = event_metrics(
        events, y_pred, end_index, tolerance_points=2, sampling_interval_seconds=60
    )
    assert metrics.delays_minutes == [1.0]


# --- Famille 3 : fausses alertes par heure ---------------------------------


def test_fausses_alertes_par_heure() -> None:
    """60 fenetres d'une minute = 1 heure : 30 faux positifs font 30 par heure."""
    assert false_alerts_per_hour(30, n_windows=60, stride=1, sampling_interval_seconds=60) == 30.0
    assert false_alerts_per_hour(30, n_windows=120, stride=1, sampling_interval_seconds=60) == 15.0
    assert false_alerts_per_hour(0, n_windows=60, stride=1, sampling_interval_seconds=60) == 0.0


def test_fausses_alertes_tiennent_compte_du_pas() -> None:
    """Un pas de 2 double la duree couverte par le meme nombre de fenetres."""
    assert false_alerts_per_hour(30, n_windows=60, stride=2, sampling_interval_seconds=60) == 15.0


# --- Coherence d'ensemble --------------------------------------------------


def test_coherence_des_trois_familles() -> None:
    """Avec la tolerance par defaut (w - 1), les trois familles concordent.

    Si toutes les fenetres positives sont alertees et elles seules, alors :
    rappel par fenetre = 1, rappel par evenement = 1, zero fausse alerte.
    """
    window_size = 3
    serie = np.zeros((20, 1))
    label = np.zeros(20, dtype=int)
    label[5:9] = 1  # evenement [5, 8]

    windowed = make_windows(
        serie,
        {"window_size": window_size, "stride": 1, "aggregates": ["last"]},
        label=label,
    )
    assert windowed.y is not None

    result = evaluate(
        y_true=windowed.y,
        y_pred=windowed.y.copy(),  # detecteur parfait par fenetre
        events=[AnomalyEvent(5, 8)],
        end_index=windowed.end_index,
        window_size=window_size,
        stride=1,
        sampling_interval_seconds=60,
        evaluation_config=EVALUATION_CONFIG,
    )

    assert result.windows.recall == 1.0
    assert result.windows.f1 == 1.0
    assert result.events.event_recall == 1.0
    assert result.false_alerts_per_hour == 0.0
    # La premiere fenetre positive se termine a l'indice 5 : delai nul.
    assert result.events.delays_minutes == [0.0]


def test_ligne_de_resultats_aplatie() -> None:
    """to_row() produit une ligne de summary.csv, sans la liste des delais."""
    result = evaluate(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
        events=[AnomalyEvent(1, 2)],
        end_index=np.arange(4),
        window_size=1,
        stride=1,
        sampling_interval_seconds=60,
        evaluation_config=EVALUATION_CONFIG,
        cost=CostMetrics(n_retrains=3, n_promoted=2, n_rejected=1, retrain_seconds=4.5),
    )
    row = result.to_row()

    assert "delays_minutes" not in row
    assert row["true_positives"] == 1
    assert row["cost_n_retrains"] == 3
    assert row["cost_retrain_seconds"] == 4.5
    assert row["duration_hours"] == 4 / 60
    assert all(not isinstance(value, list | dict) for value in row.values())
