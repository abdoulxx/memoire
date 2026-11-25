"""Rejeu des quatre strategies sur une session enregistree (jalon M6).

Une session n'est pas une campagne : il n'y a qu'un flux, deja construit a
partir du Parquet enregistre par `mdrift session-record`, pas une grille de
machines x scenarios x graines. Ce module reutilise donc `campaign.run_one`,
le seul morceau de `campaign.py` qui ne soit pas specifique au SMD, plutot que
de reecrire la construction de strategie, le rejeu et l'ecriture du resume.

Les quatre strategies (B0, B1, P, A1) rejouent **le meme flux**, construit une
seule fois par `build_server_stream` : c'est la meme garantie qu'en campagne
SMD, pour la meme raison (§6.1.1 du memoire).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.serveur.loader import SessionServeur, charger_session
from datasets.serveur.settings import ReglagesServeur
from src.config import load_config
from src.evaluation.campaign import (
    completed_run_ids,
    load_strategy_configs,
    make_run_id_for,
    run_one,
)
from src.evaluation.replay import SUMMARY_PATH
from src.evaluation.streams import build_server_stream

SOURCE = "serveur"


def _as_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value]


def load_session_stream(
    run_id: str,
    dataset_config: str = "datasets/serveur.yaml",
    features_config: dict[str, Any] | str = "features.yaml",
) -> tuple[SessionServeur, Any]:
    """Charge la session et construit son flux, une seule fois pour tous les rejeux."""
    reglages = ReglagesServeur.depuis_config(load_config(dataset_config))
    session = charger_session(run_id, reglages)
    features = (
        features_config if isinstance(features_config, dict) else load_config(features_config)
    )
    stream = build_server_stream(session, features)
    return session, stream


def run_session_campaign(
    run_id: str,
    strategies: str | Sequence[str] = ("static", "periodic", "proposed", "ablation"),
    dataset_config: str = "datasets/serveur.yaml",
    features_config: str = "features.yaml",
    model_config: str = "model.yaml",
    evaluation_config: str = "evaluation.yaml",
    drift_config: str = "drift.yaml",
    seed: int = 42,
    resume: bool = True,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    """Rejoue B0/B1/P/A1 sur une session enregistree, dans `results/summary.csv`.

    Retourne les lignes produites par **cette** execution ; les rejeux deja
    presents (meme `run_id`) sont sautes si `resume` est vrai, comme en
    campagne SMD.
    """
    configs = {
        "features": load_config(features_config),
        "model": load_config(model_config),
        "evaluation": load_config(evaluation_config),
        "drift": load_config(drift_config),
    }
    strategy_configs = load_strategy_configs(_as_list(strategies))
    _session, stream = load_session_stream(run_id, dataset_config, configs["features"])

    done = completed_run_ids(summary_path) if resume else set()
    rows: list[dict[str, Any]] = []
    for key, strategy_config in strategy_configs.items():
        target = make_run_id_for(run_id, stream.scenario, strategy_config["name"], seed, SOURCE)
        if target in done:
            continue
        rows.append(
            run_one(
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
