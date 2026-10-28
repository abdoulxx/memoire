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
from src.strategies.base import (
    Decision,
    TrainingPlan,
    ValidationStats,
    Verdict,
    WindowContext,
)
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


class ReentraineUneFois:
    """Strategie factice : reentraine une seule fois, a la fenetre demandee.

    Elle sert a verifier le contrat du harnais sans dependre du moniteur de
    derive : ce qui est teste ici est le harnais, pas la strategie P.
    """

    name = "test_retrain"

    def __init__(self, at: int = 100, promote: bool = True, reserved: int = 0) -> None:
        self.at = at
        self.promote = promote
        self.reserved = reserved
        self.fait = False
        self.historiques: list[int] = []
        self.stats: list[ValidationStats] = []
        self.fits: list[int] = []

    def on_window(self, context: WindowContext) -> Decision:
        if not self.fait and context.position == self.at:
            self.fait = True
            return Decision(retrain=True, reason="test", events=("drift_confirmed",))
        return Decision(retrain=False)

    def propose(self, history: np.ndarray, position: int) -> TrainingPlan:
        self.historiques.append(len(history))
        if self.reserved:
            return TrainingPlan(
                train_X=history[: -self.reserved], reserved_X=history[-self.reserved :]
            )
        return TrainingPlan(train_X=history, reserved_X=history[:0])

    def validate(self, stats: ValidationStats) -> Verdict:
        self.stats.append(stats)
        return Verdict(promote=self.promote, reason="test")

    def on_fit(self, training_X: np.ndarray, position: int) -> None:
        self.fits.append(position)


def test_le_reentrainement_ne_voit_que_le_passe() -> None:
    """La strategie ne recoit que l'historique observe, jamais les fenetres futures."""
    strategie = ReentraineUneFois(at=100)
    outcome = _replay(_stream(), strategie, seed=42)

    assert strategie.historiques == [101]  # fenetres 0 a 100 incluses, rien au-dela
    assert outcome.result.cost.n_retrains == 1
    assert outcome.result.cost.n_promoted == 1
    assert outcome.result.cost.retrain_seconds > 0
    # Le seuil change au reentrainement, et seulement a partir de la.
    assert outcome.thresholds[100] == outcome.thresholds[0]
    assert outcome.thresholds[101] != outcome.thresholds[0]
    # `on_fit` est appele pour l'entrainement initial puis pour la promotion.
    assert strategie.fits == [0, 100]


def test_un_rejet_laisse_le_champion_en_service() -> None:
    """Un challenger rejete est jete : le cout est compte, le modele ne change pas."""
    strategie = ReentraineUneFois(at=100, promote=False, reserved=40)
    outcome = _replay(_stream(), strategie, seed=42)

    assert outcome.result.cost.n_retrains == 1
    assert outcome.result.cost.n_rejected == 1
    assert outcome.result.cost.n_promoted == 0
    assert np.all(outcome.thresholds == outcome.thresholds[0])
    # Seul le modele initial a pris du service : aucun changement apres.
    assert outcome.model_changes == [0]
    assert strategie.fits == [0]  # pas de recalage : le champion n'a pas change

    evenements = [entry["event"] for entry in outcome.event_log]
    assert "rejected" in evenements
    assert "promoted" not in evenements


def test_la_strategie_ne_voit_aucune_etiquette() -> None:
    """Invariant du §6, verifie sur les objets reellement transmis.

    Aucune etiquette ne doit traverser la frontiere harnais / strategie. Le test
    compare les champs de `WindowContext` et de `ValidationStats` au contenu de
    `stream.y` : si une etiquette etait passee, elle serait dans l'un des deux.
    """
    recus: list[WindowContext] = []

    class Espionne(ReentraineUneFois):
        def on_window(self, context: WindowContext) -> Decision:
            recus.append(context)
            return super().on_window(context)

    strategie = Espionne(at=100, reserved=40)
    stream = _stream()
    _replay(stream, strategie, seed=42)

    champs_contexte = set(vars(recus[0]))
    assert champs_contexte == {"position", "end_index", "features", "score", "alert", "threshold"}
    champs_stats = set(vars(strategie.stats[0]))
    assert not any("label" in nom or nom in {"y", "y_true"} for nom in champs_stats)

    # Les features transmises sont bien celles du flux, pas les etiquettes.
    for contexte in recus[:50]:
        np.testing.assert_allclose(contexte.features, stream.X[contexte.position])


def test_les_evenements_declares_sont_journalises() -> None:
    """La strategie declare, le harnais ecrit : le journal suit le vocabulaire du §9."""
    outcome = _replay(_stream(), ReentraineUneFois(at=100), seed=42)
    evenements = [entry["event"] for entry in outcome.event_log]
    assert "drift_confirmed" in evenements
    assert evenements.index("drift_confirmed") < evenements.index("retrain_start")
    confirme = next(e for e in outcome.event_log if e["event"] == "drift_confirmed")
    assert confirme["details"]["position"] == 100


def test_scoring_fantome_enregistre_le_modele_remplace() -> None:
    """D16 : apres une promotion, l'ancien modele est encore score sur N fenetres."""
    outcome = _replay(_stream(), ReentraineUneFois(at=100), seed=42)
    traces = outcome.shadow.as_arrays()

    assert list(traces["shadow_positions"]) == [100]
    assert list(traces["shadow_starts"]) == [101]
    # Le flux fait 400 fenetres : il en reste 299 apres la promotion, moins que
    # les 500 de la configuration, donc le bloc est tronque a ce qui existe.
    assert int(traces["shadow_lengths"][0]) == 299
    assert len(traces["shadow_scores"]) == 299
    # Le cout du scoring fantome est exclu du temps de rejeu.
    assert outcome.shadow_seconds > 0
    assert outcome.result.cost.retrain_seconds > 0


def test_run_id_stable() -> None:
    stream = _stream()
    assert make_run_id(stream, "static", 42) == "test_synthetique_none_static_seed42"


def test_summary_ne_duplique_pas_un_rejeu(tmp_path) -> None:
    """Rejouer une experience met sa ligne a jour au lieu de l'ajouter."""
    import pandas as pd

    stream = _stream()
    configs = {
        "features": FEATURES_CONFIG,
        "model": MODEL_CONFIG,
        "evaluation": EVALUATION_CONFIG,
    }
    path = tmp_path / "summary.csv"

    for _ in range(2):
        outcome = _replay(stream, StaticStrategy(), seed=42)
        append_summary(summary_row(outcome, stream, configs), path=path)

    table = pd.read_csv(path)
    assert len(table) == 1

    autre = _replay(stream, StaticStrategy(), seed=7)
    append_summary(summary_row(autre, stream, configs), path=path)
    assert len(pd.read_csv(path)) == 2


def test_summary_accepte_une_metrique_absente_sans_avertissement(tmp_path) -> None:
    """Une colonne vide ne doit ni avertir ni disparaitre du fichier.

    Un rejeu sans derive injectee n'a pas de `f1_after`. La ligne s'ajoute donc
    avec une colonne entierement vide, cas que pandas 2.2 signale par un
    `FutureWarning` sur le dtype du concat. Le test porte sur les deux choses
    qu'on exige : aucun avertissement, et la colonne toujours presente avec les
    valeurs des rejeux qui, eux, l'avaient.
    """
    import warnings

    import pandas as pd

    path = tmp_path / "summary.csv"
    append_summary(
        {
            "run_id": "avec",
            "source": "smd",
            "machine": "m",
            "scenario": "s1",
            "strategy": "p",
            "seed": 42,
            "f1": 0.4,
            "f1_after": 0.5,
        },
        path=path,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        append_summary(
            {
                "run_id": "sans",
                "source": "smd",
                "machine": "m",
                "scenario": "none",
                "strategy": "p",
                "seed": 42,
                "f1": 0.4,
                "f1_after": None,
            },
            path=path,
        )

    table = pd.read_csv(path).set_index("run_id")
    assert "f1_after" in table.columns
    assert table.loc["avec", "f1_after"] == pytest.approx(0.5)
    assert pd.isna(table.loc["sans", "f1_after"])


def test_b0_refuse_de_reentrainer() -> None:
    """B0 ne doit meme pas pouvoir repondre a la question : l'erreur est explicite."""
    with pytest.raises(RuntimeError, match="jamais"):
        StaticStrategy().propose(np.zeros((5, 2)), 4)
