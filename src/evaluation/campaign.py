"""Campagne de rejeu : machines x scenarios x strategies x graines.

Deux proprietes de ce module comptent pour la validite des resultats.

**Un seul flux par (machine, scenario, graine).** Le flux est construit une fois
et rejoue tel quel par chacune des strategies. Si chaque strategie reconstruisait
le sien, une difference de F1 pourrait venir de l'injection et non de la
strategie, et la comparaison ne vaudrait rien. L'ordre des boucles n'est donc
pas un detail d'optimisation : machine, scenario, graine, puis strategies.

**Reprise.** Une campagne complete (12 machines x 4 scenarios x 4 strategies x
5 graines = 960 rejeux) ne tient pas dans une seule execution. `summary.csv`
porte un `run_id` par rejeu ; par defaut les rejeux deja presents sont sautes,
ce qui rend la campagne reprenable sans jamais dupliquer ni recalculer. Avec
`resume=False`, tout est recalcule et chaque ligne remplace la precedente.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.smd.loader import SMDMachine, load_machines
from src.config import load_config
from src.evaluation.replay import (
    SUMMARY_PATH,
    ReplayStream,
    append_summary,
    replay,
    save_run,
    summary_row,
)
from src.evaluation.streams import build_smd_stream
from src.strategies.factory import build_strategy


def make_run_id_for(
    machine: str, scenario: str, strategy: str, seed: int, source: str = "smd"
) -> str:
    """Meme formule que `make_run_id`, sans avoir a construire le flux.

    C'est ce qui permet a la reprise de decider de sauter un rejeu avant d'en
    payer le cout : construire le flux d'une machine SMD prend quelques secondes,
    et une campagne reprise en saute des centaines.
    """
    return f"{source}_{machine}_{scenario}_{strategy}_seed{seed}"


@dataclass(frozen=True)
class Job:
    """Un rejeu identifie. `run_id` est celui de `summary.csv`."""

    machine: str
    scenario: str
    strategy: str
    seed: int
    run_id: str


def _as_list(value: str | Iterable[str]) -> list[str]:
    """Accepte « a,b,c » ou une liste. Les CLI passent des chaines, les tests des listes."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value]


def _as_seeds(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def load_strategy_configs(names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Charge `configs/strategies/<nom>.yaml` pour chaque nom demande.

    Le nom de la campagne est celui du fichier, pas celui de la classe : `proposed`
    et `ablation` partagent `strategy: proposed` mais sont deux bras distincts et
    doivent produire deux `run_id` distincts.
    """
    configs: dict[str, dict[str, Any]] = {}
    for name in names:
        config = load_config(f"strategies/{name}.yaml")
        configs[name] = {**config, "name": config.get("name", name)}
    return configs


def completed_run_ids(path: Path = SUMMARY_PATH) -> set[str]:
    """`run_id` deja presents dans `summary.csv`. Vide si le fichier n'existe pas."""
    if not Path(path).is_file():
        return set()
    table = pd.read_csv(path, usecols=["run_id"])
    return set(table["run_id"].astype(str))


def plan(
    machines: Sequence[str],
    scenarios: Sequence[str],
    strategies: Sequence[str],
    seeds: Sequence[int],
    source: str = "smd",
) -> list[Job]:
    """Enumere les rejeux d'une campagne, dans l'ordre ou ils seront executes."""
    jobs: list[Job] = []
    for machine in machines:
        for scenario in scenarios:
            for seed in seeds:
                for strategy in strategies:
                    jobs.append(
                        Job(
                            machine=machine,
                            scenario=scenario,
                            strategy=strategy,
                            seed=int(seed),
                            run_id=f"{source}_{machine}_{scenario}_{strategy}_seed{seed}",
                        )
                    )
    return jobs


def run_campaign(
    dataset_config: str = "datasets/smd_dev.yaml",
    features_config: str = "features.yaml",
    model_config: str = "model.yaml",
    evaluation_config: str = "evaluation.yaml",
    drift_config: str = "drift.yaml",
    strategies: str | Iterable[str] = ("static", "periodic", "proposed"),
    scenarios: str | Iterable[str] = ("none",),
    seeds: str | Iterable[int] = (42,),
    machines: str | Iterable[str] | None = None,
    resume: bool = True,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    """Execute la campagne et retourne les lignes produites pendant cette execution.

    Les lignes sautees par la reprise ne sont pas dans le retour : il decrit ce
    qui a ete calcule ici, pas l'etat global de `summary.csv`.
    """
    configs = {
        "dataset": load_config(dataset_config),
        "features": load_config(features_config),
        "model": load_config(model_config),
        "evaluation": load_config(evaluation_config),
        "drift": load_config(drift_config),
    }
    strategy_configs = load_strategy_configs(_as_list(strategies))
    scenario_names = _as_list(scenarios)
    seed_values = _as_seeds(seeds)

    available: dict[str, SMDMachine] = {m.name: m for m in load_machines(dataset_config)}
    wanted = list(available) if machines is None else _as_list(machines)
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ValueError(
            f"Machines absentes de {dataset_config} : {', '.join(unknown)}. "
            "Le perimetre d'une campagne est celui de sa configuration de dataset (D17)."
        )

    done = completed_run_ids(summary_path) if resume else set()
    rows: list[dict[str, Any]] = []

    for name in wanted:
        machine = available[name]
        for scenario in scenario_names:
            for seed in seed_values:
                pending = {
                    key: config
                    for key, config in strategy_configs.items()
                    if make_run_id_for(name, scenario, config["name"], seed) not in done
                }
                if not pending:
                    continue
                # Un seul flux pour toutes les strategies de ce triplet.
                stream = build_smd_stream(machine, configs["features"], scenario, seed)
                for key, strategy_config in pending.items():
                    rows.append(
                        _run_one(
                            stream=stream,
                            strategy_key=key,
                            strategy_config=strategy_config,
                            configs=configs,
                            seed=seed,
                            summary_path=summary_path,
                            on_row=on_row,
                        )
                    )

    return pd.DataFrame(rows)


def _run_one(
    stream: ReplayStream,
    strategy_key: str,
    strategy_config: dict[str, Any],
    configs: dict[str, Any],
    seed: int,
    summary_path: Path,
    on_row: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    """Un rejeu : construction de la strategie, rejeu, ecriture, ligne de resume."""
    started = time.perf_counter()
    strategy = build_strategy(
        strategy_config=strategy_config,
        features_config=configs["features"],
        drift_config=configs["drift"],
        sampling_interval_seconds=stream.sampling_interval_seconds,
    )
    outcome = replay(
        stream=stream,
        strategy=strategy,
        model_config=configs["model"],
        features_config=configs["features"],
        evaluation_config=configs["evaluation"],
        seed=seed,
    )
    # `strategy.name` vient du YAML : il distingue `proposed` de `ablation`, qui
    # partagent la meme classe. Les dossiers de resultats suivent donc le bras
    # de l'experience et non l'implementation.
    run_configs = {**configs, "strategy": strategy_config}
    save_run(outcome, stream, run_configs)
    row = summary_row(outcome, stream, run_configs)
    row["strategy_config"] = strategy_key
    row["total_seconds"] = round(time.perf_counter() - started, 2)
    append_summary(row, summary_path)
    if on_row is not None:
        on_row(row)
    return row
