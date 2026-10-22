"""Tests du harnais de rejeu.

Le point verifie ici est la propriete dont depend toute la campagne de resultats :
**meme seed = meme resultat**. Sans elle, les moyennes sur 5 seeds du chapitre 6
ne voudraient rien dire.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.smd.loader import AnomalyEvent
from src.config import load_config
from src.evaluation.replay import (
    ReplayStream,
    append_summary,
    make_run_id,
    replay,
    summary_row,
)
from src.strategies.base import Decision, WindowContext
from src.strategies.static import StaticStrategy

FEATURES_CONFIG = load_config("features.yaml")
MODEL_CONFIG = load_config("model.yaml")
EVALUATION_CONFIG = load_config("evaluation.yaml")


def _stream(n_train: int = 600, n_test: int = 400, seed: int = 0) -> ReplayStream:
    """Flux synthetique : bruit normal, avec une anomalie franche au milieu du test."""
    rng = np.random.default_rng(seed)
    train_X = rng.normal(size=(n_train, 6))
    X = rng.normal(size=(n_test, 6))
    y = np.zeros(n_test, dtype=np.int64)
    X[200:220] += 6.0  # anomalie
    y[200:220] = 1

    return ReplayStream(
        source="test",
        machine="synthetique",
        scenario="none",
        train_X=train_X,
        X=X,
        y=y,
        end_index=np.arange(n_test),
        events=[AnomalyEvent(200, 219)],
        window_size=1,
        stride=1,
        sampling_interval_seconds=60,
    )


def _replay(stream: ReplayStream, strategy, seed: int):
    return replay(
        stream=stream,
        strategy=strategy,
        model_config=MODEL_CONFIG,
        features_config=FEATURES_CONFIG,
        evaluation_config=EVALUATION_CONFIG,
        seed=seed,
    )


def test_meme_seed_meme_resultat() -> None:
    """Propriete de reproductibilite exigee par le cadrage."""
    stream = _stream()
    a = _replay(stream, StaticStrategy(), seed=42)
    b = _replay(stream, StaticStrategy(), seed=42)

    np.testing.assert_allclose(a.scores, b.scores)
    np.testing.assert_array_equal(a.alerts, b.alerts)
    assert a.result.to_row() == b.result.to_row()
    assert a.run_id == b.run_id


def test_seed_differente_resultat_different() -> None:
    stream = _stream()
    a = _replay(stream, StaticStrategy(), seed=42)
    c = _replay(stream, StaticStrategy(), seed=7)
    assert not np.allclose(a.scores, c.scores)


def test_b0_ne_reentraine_jamais() -> None:
    """Le cout de la reference basse est nul par construction."""
    outcome = _replay(_stream(), StaticStrategy(), seed=42)
    assert outcome.result.cost.n_retrains == 0
    assert outcome.result.cost.retrain_seconds == 0.0

    evenements = [entry["event"] for entry in outcome.event_log]
    assert "retrain_start" not in evenements
    assert evenements[0] == "run_start"
    assert evenements[-1] == "run_end"


def test_anomalie_franche_est_detectee() -> None:
    """Controle de bon sens : une anomalie a 6 ecarts-types doit etre vue."""
    outcome = _replay(_stream(), StaticStrategy(), seed=42)
    assert outcome.result.events.event_recall == 1.0
    assert outcome.result.windows.recall > 0.9


def test_journal_et_traces_complets() -> None:
    stream = _stream()
    outcome = _replay(stream, StaticStrategy(), seed=42)
    assert len(outcome.scores) == len(stream)
    assert len(outcome.alerts) == len(stream)
    assert len(outcome.thresholds) == len(stream)
    assert np.all(outcome.thresholds == outcome.thresholds[0])  # B0 : seuil constant
    assert all(entry["run_id"] == outcome.run_id for entry in outcome.event_log)


def test_le_reentrainement_ne_voit_que_le_passe() -> None:
    """Une strategie factice qui reentraine une fois ne recoit que l'historique observe."""
    vues: list[int] = []

    class ReentraineUneFois:
        name = "test_retrain"

        def __init__(self) -> None:
            self.fait = False

        def on_window(self, context: WindowContext) -> Decision:
            if not self.fait and context.position == 100:
                self.fait = True
                return Decision(retrain=True, reason="test")
            return Decision(retrain=False)

        def training_set(self, history: np.ndarray, position: int) -> np.ndarray:
            vues.append(len(history))
            return history

    stream = _stream()
    outcome = _replay(stream, ReentraineUneFois(), seed=42)

    assert vues == [101]  # fenetres 0 a 100 incluses, rien au-dela
    assert outcome.result.cost.n_retrains == 1
    assert outcome.result.cost.retrain_seconds > 0
    # Le seuil change au reentrainement, et seulement a partir de la.
    assert outcome.thresholds[100] == outcome.thresholds[0]
    assert outcome.thresholds[101] != outcome.thresholds[0]


def test_run_id_stable() -> None:
    stream = _stream()
    assert make_run_id(stream, "static", 42) == "test_synthetique_none_static_seed42"


def test_summary_ne_duplique_pas_un_rejeu(tmp_path) -> None:
    """Rejouer une experience met sa ligne a jour au lieu de l'ajouter."""
    import pandas as pd

    stream = _stream()
    configs = {"features": FEATURES_CONFIG, "model": MODEL_CONFIG}
    path = tmp_path / "summary.csv"

    for _ in range(2):
        outcome = _replay(stream, StaticStrategy(), seed=42)
        append_summary(summary_row(outcome, stream, configs), path=path)

    table = pd.read_csv(path)
    assert len(table) == 1

    autre = _replay(stream, StaticStrategy(), seed=7)
    append_summary(summary_row(autre, stream, configs), path=path)
    assert len(pd.read_csv(path)) == 2


def test_b0_refuse_de_reentrainer() -> None:
    with pytest.raises(RuntimeError, match="jamais"):
        StaticStrategy().training_set(np.zeros((5, 2)), 4)
