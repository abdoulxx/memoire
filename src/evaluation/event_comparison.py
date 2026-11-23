"""Comparaison par evenement entre deux strategies, depuis des traces deja ecrites (D48).

Complement demande apres retours du directeur sur les resultats M3 deja rejoues
(aucun rejeu relance) : quels evenements etiquetes sont detectes par une
strategie (par defaut B0, `static`) et rates par une autre (par defaut P,
`proposed`), et inversement.

`src.evaluation.metrics.event_metrics` renvoie deja `delays_minutes` dans le
meme ordre que les evenements passes en entree : il suffit de l'appeler une
fois par strategie sur la MEME liste d'evenements (meme flux, donc memes
evenements verite-terrain) et de comparer terme a terme.

La liste d'evenements et `end_index` ne sont pas lus depuis `replay.py` /
`streams.py` (qui importent `src.models.isolation_forest`, donc `sklearn`) :
ce module reconstruit directement le flux etiquete avec les memes briques que
`build_smd_stream` (`datasets.smd.loader`, `scenarios.injection`,
`src.features.windowing`), qui n'ont aucune dependance sklearn/scipy. La
reconstruction est verifiee bit a bit (`end_index` identique) contre les
traces deja ecrites avant toute comparaison, par securite si jamais une config
differait entre temps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from datasets.smd.loader import events_from_label, load_machine
from scenarios.injection import inject, load_scenario
from src.config import load_config
from src.evaluation.metrics import event_metrics
from src.features.windowing import make_windows

NO_DRIFT = "none"


def build_events_and_end_index(
    machine_name: str,
    scenario_name: str,
    seed: int,
    dataset_config: str,
    features_config: dict[str, Any],
) -> tuple[list, np.ndarray]:
    """Reconstruit les evenements verite-terrain et `end_index`, sans rejeu.

    Meme logique que `src.evaluation.streams.build_smd_stream` (injection puis
    fenetrage sur le flux brut), rejouee sans construire de `ReplayStream` ni
    importer `src.evaluation.replay`. `scenario_name == "none"` suit la meme
    convention que `build_smd_stream` : aucune injection, flux `test` original.
    """
    machine = load_machine(machine_name, dataset_config)
    series = machine.test
    label = machine.test_label
    if scenario_name != NO_DRIFT:
        scenario = load_scenario(f"scenarios/{scenario_name}.yaml")
        injected = inject(
            train=machine.train,
            test=machine.test,
            label=machine.test_label,
            scenario=scenario,
            seed=seed,
            sampling_interval_seconds=machine.sampling_interval_seconds,
            stream_key=machine.name,
        )
        series = injected.series
        label = injected.label
    events = events_from_label(label)
    windows = make_windows(series, features_config, label=label)
    return events, windows.end_index


def collect_event_comparison(
    runs_dir: Path | str,
    dataset_config: str,
    features_config_path: str = "features.yaml",
    evaluation_config_path: str = "evaluation.yaml",
    baseline: str = "static",
    challenger: str = "proposed",
) -> list[dict[str, Any]]:
    """Compare `baseline` et `challenger` evenement par evenement, pour chaque rejeu.

    Meme arborescence que `src.evaluation.promotions.collect_promotions` :
    `<source>/<machine>/<scenario>/<strategie>/seed<n>/traces.npz`.
    """
    features_cfg = load_config(features_config_path)
    eval_cfg = load_config(evaluation_config_path)
    tolerance_points = eval_cfg["event"]["tolerance_points"]
    if tolerance_points is None:
        tolerance_points = int(features_cfg["window_size"]) - 1
    sampling_interval_seconds = int(load_config(dataset_config)["sampling_interval_seconds"])

    racine = Path(runs_dir)
    lignes: list[dict[str, Any]] = []
    for trace_baseline in sorted(racine.glob(f"*/*/*/{baseline}/seed*/traces.npz")):
        graine_str = trace_baseline.parent.name.removeprefix("seed")
        scenario = trace_baseline.parent.parent.parent.name
        machine = trace_baseline.parent.parent.parent.parent.name
        source = trace_baseline.parent.parent.parent.parent.parent.name
        trace_challenger = (
            trace_baseline.parent.parent.parent / challenger / f"seed{graine_str}" / "traces.npz"
        )
        if not trace_challenger.is_file():
            continue

        seed = int(graine_str)
        events, end_index_rebuilt = build_events_and_end_index(
            machine, scenario, seed, dataset_config, features_cfg
        )
        with np.load(trace_baseline) as data_baseline:
            alerts_baseline = data_baseline["alerts"]
            end_index_trace = data_baseline["end_index"]
        if not np.array_equal(end_index_rebuilt, end_index_trace):
            continue
        with np.load(trace_challenger) as data_challenger:
            alerts_challenger = data_challenger["alerts"]
            end_index_challenger = data_challenger["end_index"]
        if not np.array_equal(end_index_challenger, end_index_trace):
            continue

        em_baseline = event_metrics(
            events, alerts_baseline, end_index_trace, tolerance_points, sampling_interval_seconds
        )
        em_challenger = event_metrics(
            events, alerts_challenger, end_index_trace, tolerance_points, sampling_interval_seconds
        )
        for idx, event in enumerate(events):
            delay_baseline = em_baseline.delays_minutes[idx]
            delay_challenger = em_challenger.delays_minutes[idx]
            detected_baseline = delay_baseline is not None
            detected_challenger = delay_challenger is not None
            if detected_baseline == detected_challenger:
                statut = "les_deux" if detected_baseline else "aucun"
            elif detected_baseline:
                statut = f"{baseline}_seul"
            else:
                statut = f"{challenger}_seul"
            lignes.append(
                {
                    "source": source,
                    "machine": machine,
                    "scenario": scenario,
                    "seed": seed,
                    "event_index": idx,
                    "event_start": event.start,
                    "event_end": event.end,
                    "event_length_points": event.length,
                    f"detected_{baseline}": detected_baseline,
                    f"detected_{challenger}": detected_challenger,
                    f"delay_{baseline}_minutes": delay_baseline,
                    f"delay_{challenger}_minutes": delay_challenger,
                    "statut": statut,
                }
            )
    return lignes
