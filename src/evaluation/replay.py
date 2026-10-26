"""Harnais de rejeu prequentiel.

C'est la piece centrale du dispositif experimental : le **meme** flux de fenetres
est rejoue a l'identique pour chaque strategie, de sorte qu'un ecart de
performance soit attribuable a la strategie et a rien d'autre.

Trois garanties, qui font la validite de la comparaison :

1. **Meme flux** : le flux est construit une fois (donnees + derive injectee +
   seed) et passe tel quel a chaque strategie.
2. **Prequentiel** : chaque fenetre est d'abord scoree par le modele en service,
   et seulement ensuite soumise a la strategie. Aucune information du futur ne
   remonte vers le passe.
3. **Tout est identique sauf la strategie** : memes features, meme modele, meme
   entrainement initial, meme regle de seuil.

Note d'implementation : entre deux reentrainements, le modele en service ne
change pas. Le harnais score donc par blocs plutot que fenetre par fenetre, ce
qui est **exactement equivalent** au scoring point par point et evite des
dizaines de milliers d'appels unitaires a scikit-learn. La boucle de decision,
elle, reste bien sequentielle.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from datasets.smd.loader import AnomalyEvent
from src.config import PROJECT_ROOT
from src.evaluation.metrics import CostMetrics, EvaluationResult, evaluate
from src.models.isolation_forest import IsolationForestDetector, build_detector
from src.strategies.base import Strategy, WindowContext

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"


@dataclass(frozen=True)
class ReplayStream:
    """Le flux rejoue : construit une fois, partage par toutes les strategies."""

    source: str
    machine: str
    scenario: str
    train_X: np.ndarray
    X: np.ndarray
    y: np.ndarray
    end_index: np.ndarray
    events: list[AnomalyEvent]
    window_size: int
    stride: int
    sampling_interval_seconds: int

    # Verite terrain sur la derive injectee (None pour le scenario `none`).
    # En indices de points de la serie d'origine, comme `end_index`.
    drift_index: int | None = None
    drift_end_index: int | None = None
    drift_metadata: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.X)

    def drift_window(self) -> int | None:
        """Indice de la premiere fenetre touchee par la derive.

        Les metriques raisonnent en fenetres, la derive est injectee en points :
        la conversion retient la premiere fenetre dont le dernier point atteint
        t_d. Les `window_size - 1` fenetres suivantes melangent les deux regimes,
        c'est une consequence inevitable du fenetrage et non une approximation.
        """
        if self.drift_index is None:
            return None
        return int(np.searchsorted(self.end_index, self.drift_index, side="left"))


@dataclass
class ReplayOutcome:
    """Resultat d'un rejeu : metriques, traces et journal d'evenements."""

    run_id: str
    seed: int
    strategy: str
    result: EvaluationResult
    scores: np.ndarray
    alerts: np.ndarray
    thresholds: np.ndarray
    event_log: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    wall_seconds: float = 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_run_id(stream: ReplayStream, strategy_name: str, seed: int) -> str:
    """Identifiant stable : deux executions de memes parametres portent le meme nom."""
    return f"{stream.source}_{stream.machine}_{stream.scenario}_{strategy_name}_seed{seed}"


def replay(
    stream: ReplayStream,
    strategy: Strategy,
    model_config: dict[str, Any],
    features_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    seed: int,
) -> ReplayOutcome:
    """Rejoue le flux pour une strategie et retourne ses metriques."""
    started = time.perf_counter()
    run_id = make_run_id(stream, strategy.name, seed)
    started_at = _now()
    log: list[dict[str, Any]] = []

    def record(event: str, **details: Any) -> None:
        log.append(
            {
                "ts": _now(),
                "run_id": run_id,
                "strategy": strategy.name,
                "event": event,
                "details": details,
            }
        )

    record("run_start", n_windows=len(stream), n_train_windows=len(stream.train_X))

    # Entrainement initial : identique pour toutes les strategies, a seed egale.
    detector: IsolationForestDetector = build_detector(model_config, features_config, seed)
    detector.fit(stream.train_X)
    record(
        "initial_fit",
        seconds=round(detector.fit_seconds, 3),
        threshold=detector.threshold,
        n_train_windows=detector.n_train_windows,
    )

    n = len(stream)
    scores = np.empty(n, dtype=np.float64)
    alerts = np.zeros(n, dtype=np.int64)
    thresholds = np.empty(n, dtype=np.float64)
    cost = CostMetrics()

    position = 0
    while position < n:
        # Le modele en service est constant jusqu'au prochain reentrainement :
        # on score le reste du flux en un bloc, puis on decide sequentiellement.
        block = detector.score(stream.X[position:])
        threshold = detector.threshold
        retrained_at: int | None = None

        for offset in range(len(block)):
            index = position + offset
            score = float(block[offset])
            alert = bool(score > threshold)

            scores[index] = score
            alerts[index] = int(alert)
            thresholds[index] = threshold

            decision = strategy.on_window(
                WindowContext(
                    position=index,
                    end_index=int(stream.end_index[index]),
                    features=stream.X[index],
                    score=score,
                    alert=alert,
                    threshold=threshold,
                )
            )

            if decision.retrain:
                record("retrain_start", position=index, reason=decision.reason, **decision.details)
                # Le reentrainement ne voit que l'historique deja observe.
                training_set = strategy.training_set(stream.X[: index + 1], index)
                detector = build_detector(model_config, features_config, seed)
                detector.fit(training_set)
                cost.n_retrains += 1
                cost.n_promoted += 1  # B0/B1 : tout reentrainement est mis en service
                cost.retrain_seconds += detector.fit_seconds
                record(
                    "retrain_end",
                    position=index,
                    seconds=round(detector.fit_seconds, 3),
                    threshold=detector.threshold,
                    n_train_windows=detector.n_train_windows,
                )
                retrained_at = index
                break

        position = n if retrained_at is None else retrained_at + 1

    result = evaluate(
        y_true=stream.y,
        y_pred=alerts,
        events=stream.events,
        end_index=stream.end_index,
        window_size=stream.window_size,
        stride=stream.stride,
        sampling_interval_seconds=stream.sampling_interval_seconds,
        evaluation_config=evaluation_config,
        cost=cost,
    )
    record("run_end", f1=result.windows.f1, n_alerts=result.windows.n_alerts)

    return ReplayOutcome(
        run_id=run_id,
        seed=seed,
        strategy=strategy.name,
        result=result,
        scores=scores,
        alerts=alerts,
        thresholds=thresholds,
        event_log=log,
        started_at=started_at,
        wall_seconds=time.perf_counter() - started,
    )


def run_directory(stream: ReplayStream, strategy_name: str, seed: int) -> Path:
    """`results/runs/<source>/<machine>/<scenario>/<strategie>/<seed>/`."""
    return (
        RESULTS_DIR
        / "runs"
        / stream.source
        / stream.machine
        / stream.scenario
        / strategy_name
        / f"seed{seed}"
    )


def save_run(
    outcome: ReplayOutcome,
    stream: ReplayStream,
    configs: dict[str, Any],
) -> Path:
    """Ecrit a cote des resultats tout ce qui permet de les reproduire et de les tracer."""
    directory = run_directory(stream, outcome.strategy, outcome.seed)
    directory.mkdir(parents=True, exist_ok=True)

    # La configuration complete, telle qu'elle a servi.
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": outcome.run_id,
                "seed": outcome.seed,
                "source": stream.source,
                "machine": stream.machine,
                "scenario": stream.scenario,
                "strategy": outcome.strategy,
                "started_at": outcome.started_at,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "configs": configs,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    (directory / "metrics.json").write_text(
        json.dumps(outcome.result.to_row(), indent=2), encoding="utf-8"
    )

    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        for entry in outcome.event_log:
            handle.write(json.dumps(entry) + "\n")

    # Traces pour les figures : scores, alertes, seuils, verite terrain.
    np.savez_compressed(
        directory / "traces.npz",
        scores=outcome.scores,
        alerts=outcome.alerts,
        thresholds=outcome.thresholds,
        y_true=stream.y,
        end_index=stream.end_index,
    )
    return directory


def summary_row(
    outcome: ReplayOutcome,
    stream: ReplayStream,
    configs: dict[str, Any],
) -> dict[str, Any]:
    """Une ligne de `results/summary.csv` : identification, parametres, metriques."""
    features = configs["features"]
    model = configs["model"]
    row: dict[str, Any] = {
        "run_id": outcome.run_id,
        "source": stream.source,
        "machine": stream.machine,
        "scenario": stream.scenario,
        "strategy": outcome.strategy,
        "seed": outcome.seed,
        "started_at": outcome.started_at,
        "wall_seconds": round(outcome.wall_seconds, 2),
        "window_size": features["window_size"],
        "stride": features["stride"],
        "model": model["model"],
        "threshold_percentile": model["threshold"]["percentile"],
        "n_train_windows": len(stream.train_X),
    }
    if stream.drift_metadata is not None:
        # Parametres effectifs de l'injection, pour que chaque ligne de resultat
        # porte la derive qu'elle a subie (D15 : part ecretee journalisee).
        meta = stream.drift_metadata
        row.update(
            {
                "drift_index": stream.drift_index,
                "drift_window": stream.drift_window(),
                "drift_end_index": stream.drift_end_index,
                "drift_level_shift_k": meta.get("level_shift_k"),
                "drift_scale_factor": meta.get("scale_factor"),
                "drift_n_metriques": meta.get("n_metriques_touchees"),
                "drift_part_ecretee": meta.get("part_ecretee"),
            }
        )
    row.update(outcome.result.to_row())
    return row


def append_summary(row: dict[str, Any], path: Path = SUMMARY_PATH) -> Path:
    """Ajoute la ligne a `summary.csv`, en remplacant celle du meme `run_id`.

    Rejouer une experience met donc sa ligne a jour au lieu de la dupliquer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.is_file():
        existing = pd.read_csv(path)
        existing = existing[existing["run_id"] != row["run_id"]]
        new = pd.concat([existing, new], ignore_index=True)
    new = new.sort_values(["source", "machine", "scenario", "strategy", "seed"])
    new.to_csv(path, index=False)
    return path
