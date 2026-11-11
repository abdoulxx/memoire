"""Harnais de rejeu prequentiel.

C'est la piece centrale du dispositif experimental : le **meme** flux de fenetres
est rejoue a l'identique pour chaque strategie, de sorte qu'un ecart de
performance soit attribuable a la strategie et a rien d'autre.

Quatre garanties, qui font la validite de la comparaison :

1. **Meme flux** : le flux est construit une fois (donnees + derive injectee +
   seed) et passe tel quel a chaque strategie.
2. **Prequentiel** : chaque fenetre est d'abord scoree par le modele en service,
   et seulement ensuite soumise a la strategie. Aucune information du futur ne
   remonte vers le passe.
3. **Tout est identique sauf la strategie** : memes features, meme modele, meme
   entrainement initial, meme regle de seuil.
4. **Aucune etiquette ne traverse la frontiere** : le harnais detient `stream.y`
   et ne le passe jamais a la strategie. Il ne lui transmet que des
   `WindowContext` et des `ValidationStats`, dont aucune signature ne comporte
   de champ d'etiquette. C'est ce qui rend verifiable la regle du §6 : aucune
   etiquette dans une decision de la strategie.

Cycle d'un reentrainement, du cote du harnais :

    decision = strategy.on_window(ctx)       # faut-il reentrainer ?
    plan     = strategy.propose(history, i)  # quelles donnees, quelle fenetre V
    challenger.fit(plan.train_X)             # le HARNAIS entraine
    stats    = mesures sans etiquette sur V  # le HARNAIS mesure
    verdict  = strategy.validate(stats)      # mise en service ou rejet
    si promu : champion = challenger, puis strategy.on_fit(plan.train_X, i)

Les quatre lignes du milieu vivent dans `src/strategies/retraining.py` : la
boucle en ligne du jalon M5 execute le **meme** cycle, et le factoriser est ce
qui garantit que le chapitre 5 montre bien la strategie evaluee au chapitre 6.

Note d'implementation : entre deux promotions, le modele en service ne change
pas. Le harnais score donc par blocs plutot que fenetre par fenetre, ce qui est
**exactement equivalent** au scoring point par point et evite des dizaines de
milliers d'appels unitaires a scikit-learn. Un rejet ne change pas le modele :
le bloc deja calcule reste valide et n'est pas recalcule. La boucle de decision,
elle, reste bien sequentielle.

Consequence a assumer : le temps mesure par fenetre n'est qu'un temps **amorti**
(`window_ms_amortized` dans `summary.csv`). La vraie mesure par fenetre se fait
dans la boucle en ligne (M5).

Scoring fantome (D16) : a chaque promotion, le modele remplace est encore scoré
sur les N fenetres suivantes, en un seul bloc, pour permettre le calcul **hors
ligne** des promotions degradantes. Ce cout est exclu du cout de la strategie et
du temps de rejeu : c'est un instrument de mesure, pas une piece du pipeline.
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
from src.evaluation.promotions import evaluate_promotions
from src.models.isolation_forest import IsolationForestDetector, build_detector
from src.strategies.base import Strategy, WindowContext
from src.strategies.retraining import run_retrain_cycle

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
class ShadowTrace:
    """Traces du modele remplace apres chaque promotion (D16).

    Stockage aplati : les scores de toutes les promotions sont concatenes et
    `lengths` permet de retrouver le segment de chacune. C'est le format que
    `np.savez_compressed` accepte sans ruse, et il se relit en deux lignes.

    Le calcul des promotions degradantes est fait **hors ligne** depuis ces
    traces (`src/evaluation/promotions.py`), jamais pendant le rejeu : la
    strategie ne doit pas pouvoir en dependre.
    """

    positions: list[int] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    scores: list[np.ndarray] = field(default_factory=list)

    def add(self, position: int, start: int, threshold: float, scores: np.ndarray) -> None:
        self.positions.append(int(position))
        self.starts.append(int(start))
        self.lengths.append(int(len(scores)))
        self.thresholds.append(float(threshold))
        self.scores.append(np.asarray(scores, dtype=np.float64))

    def as_arrays(self) -> dict[str, np.ndarray]:
        concatenated = np.concatenate(self.scores) if self.scores else np.empty(0, dtype=np.float64)
        return {
            "shadow_positions": np.asarray(self.positions, dtype=np.int64),
            "shadow_starts": np.asarray(self.starts, dtype=np.int64),
            "shadow_lengths": np.asarray(self.lengths, dtype=np.int64),
            "shadow_thresholds": np.asarray(self.thresholds, dtype=np.float64),
            "shadow_scores": concatenated,
        }


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
    # Indice de la fenetre a partir de laquelle chaque modele est en service :
    # 0 pour le modele initial, puis l'indice de chaque promotion.
    model_changes: list[int] = field(default_factory=list)
    shadow: ShadowTrace = field(default_factory=ShadowTrace)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    wall_seconds: float = 0.0
    shadow_seconds: float = 0.0

    @property
    def window_ms_amortized(self) -> float:
        """Temps amorti par fenetre, en millisecondes.

        Amorti et non mesure : le scoring par blocs (voir l'en-tete du module)
        interdit d'attribuer un temps a une fenetre precise. La mesure reelle
        (`window_ms_mean`, `window_ms_p95`) viendra de la boucle en ligne.
        """
        n = len(self.scores)
        return 0.0 if n == 0 else 1000.0 * self.wall_seconds / n


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

    shadow_windows = int(evaluation_config.get("promotions", {}).get("shadow_windows", 500))

    record("run_start", n_windows=len(stream), n_train_windows=len(stream.train_X))

    # Entrainement initial : identique pour toutes les strategies, a seed egale.
    detector: IsolationForestDetector = build_detector(model_config, features_config, seed)
    detector.fit(stream.train_X)
    strategy.on_fit(stream.train_X, 0)
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
    shadow = ShadowTrace()
    shadow_seconds = 0.0
    model_changes: list[int] = [0]

    position = 0
    while position < n:
        # Le modele en service est constant jusqu'a la prochaine promotion :
        # on score le reste du flux en un bloc, puis on decide sequentiellement.
        block = detector.score(stream.X[position:])
        threshold = detector.threshold
        promoted_at: int | None = None

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
            for name in decision.events:
                record(name, **{**decision.details, "position": index})

            if not decision.retrain:
                continue

            record(
                "retrain_start",
                **{**decision.details, "position": index, "reason": decision.reason},
            )

            # Le reentrainement ne voit que l'historique deja observe. Le cycle
            # lui-meme est dans `src/strategies/retraining.py`, parce que la
            # boucle en ligne (M5) doit executer exactement le meme : une
            # divergence entre les deux rendrait le chapitre 5 incomparable au
            # chapitre 6.
            cycle = run_retrain_cycle(
                strategy,
                history_X=stream.X[: index + 1],
                position=index,
                n_history=index + 1,
                champion=detector,
                build=lambda: build_detector(model_config, features_config, seed),
            )
            challenger, plan, verdict = cycle.challenger, cycle.plan, cycle.verdict
            cost.n_retrains += 1
            cost.retrain_seconds += challenger.fit_seconds
            cost.validation_seconds += cycle.validation_seconds

            record(
                "retrain_end",
                position=index,
                seconds=round(challenger.fit_seconds, 3),
                threshold=challenger.threshold,
                **cycle.details(),
            )

            if not verdict.promote:
                cost.n_rejected += 1
                record(
                    "rejected",
                    **{**verdict.details, "position": index, "reason": verdict.reason},
                )
                # Le champion reste en service : le bloc deja score est toujours
                # valide, la boucle continue sans rien recalculer.
                continue

            # Scoring fantome du modele remplace, hors cout de la strategie (D16).
            shadow_started = time.perf_counter()
            shadow_start = index + 1
            shadow_stop = min(n, shadow_start + shadow_windows)
            if shadow_stop > shadow_start:
                shadow.add(
                    position=index,
                    start=shadow_start,
                    threshold=detector.threshold,
                    scores=detector.score(stream.X[shadow_start:shadow_stop]),
                )
            shadow_seconds += time.perf_counter() - shadow_started

            detector = challenger
            cost.n_promoted += 1
            model_changes.append(index + 1)
            strategy.on_fit(plan.train_X, index)
            record(
                "promoted",
                **{**verdict.details, "position": index, "reason": verdict.reason},
            )
            promoted_at = index
            break

        position = n if promoted_at is None else promoted_at + 1

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
        drift_window=stream.drift_window(),
        event_log=log,
    )
    record("run_end", f1=result.windows.f1, n_alerts=result.windows.n_alerts)

    # Le temps du scoring fantome est retire : il ne fait pas partie du rejeu.
    wall_seconds = time.perf_counter() - started - shadow_seconds

    return ReplayOutcome(
        run_id=run_id,
        seed=seed,
        strategy=strategy.name,
        result=result,
        scores=scores,
        alerts=alerts,
        thresholds=thresholds,
        model_changes=model_changes,
        shadow=shadow,
        event_log=log,
        started_at=started_at,
        wall_seconds=wall_seconds,
        shadow_seconds=shadow_seconds,
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

    # Traces pour les figures et pour le calcul hors ligne des promotions
    # degradantes : scores, alertes, seuils, changements de modele, verite
    # terrain, et scores fantomes du modele remplace (D16).
    np.savez_compressed(
        directory / "traces.npz",
        scores=outcome.scores,
        alerts=outcome.alerts,
        thresholds=outcome.thresholds,
        y_true=stream.y,
        end_index=stream.end_index,
        model_changes=np.asarray(outcome.model_changes, dtype=np.int64),
        **outcome.shadow.as_arrays(),
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
        # Temps amorti, pas mesure : voir ReplayOutcome.window_ms_amortized.
        "window_ms_amortized": round(outcome.window_ms_amortized, 4),
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

    # Promotions degradantes : calculees ici, apres le rejeu, depuis les traces
    # fantomes (D16). Elles utilisent les etiquettes, ce qui est licite dans la
    # mesure du resultat mais ne l'aurait pas ete dans une decision de la
    # strategie : `replay` a deja rendu la main quand ce calcul a lieu.
    promotions = configs.get("evaluation", {}).get("promotions", {})
    row.update(
        evaluate_promotions(
            y_true=stream.y,
            alerts=outcome.alerts,
            model_changes=np.asarray(outcome.model_changes, dtype=np.int64),
            delta_f1=float(promotions.get("delta_f1", 0.05)),
            **outcome.shadow.as_arrays(),
        ).as_row()
    )
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
        # Une metrique absente de ce rejeu-ci (`f1_after` n'existe pas sans derive
        # injectee) donne une colonne entierement vide, dont pandas 2.2 annonce
        # que le dtype resultant du concat changera. On retire donc ces colonnes
        # de la ligne ajoutee quand `summary.csv` les porte deja : le resultat
        # ecrit est le meme, sans dependre d'un comportement annonce comme
        # changeant. Une colonne vide mais INCONNUE du fichier est conservee,
        # sinon la ligne perdrait une metrique.
        vides = [nom for nom in new.columns if nom in existing.columns and new[nom].isna().all()]
        new = pd.concat([existing, new.drop(columns=vides)], ignore_index=True)
    new = new.sort_values(["source", "machine", "scenario", "strategy", "seed"])
    new.to_csv(path, index=False)
    return path
