"""Audit et calibrage de la derive injectee (M1).

Deux experiences, qui repondent a deux questions distinctes :

- `audit` : a quoi ressemble la derive produite sur chaque machine, pour chaque
  scenario et chaque graine. Produit `results/injections.csv`, qui est la
  « description des donnees » demandee au §11 du cadrage : instant de derive,
  metriques touchees, part ecretee, duree de transition.

- `calibrate_k` : quelle amplitude k retenir. Mesure la degradation du F1 de B0
  de part et d'autre de t_d en fonction de k. Produit
  `results/calibration_k_dev.csv` et fonde D22.

Le calibrage n'est fait que sur les machines de mise au point (D17). Deux
economies rendent la mesure rapide sans rien changer a son sens : la moitie
`train` n'etant jamais transformee, les fenetres d'entrainement et le modele B0
sont calcules **une fois par machine** et reutilises pour toute la grille ; seul
le flux de test est reconstruit a chaque (k, graine).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from datasets.smd.loader import SMDMachine, load_machine
from scenarios.injection import SCENARIOS, DriftScenario, inject, load_scenario
from src.config import load_config
from src.evaluation.metrics import window_metrics
from src.evaluation.replay import RESULTS_DIR
from src.features.windowing import make_windows
from src.models.isolation_forest import build_detector

INJECTIONS_PATH = RESULTS_DIR / "injections.csv"
K_CALIBRATION_PATH = RESULTS_DIR / "calibration_k_dev.csv"


def _drift_window(end_index: np.ndarray, drift_index: int) -> int:
    """Premiere fenetre dont le dernier point atteint t_d."""
    return int(np.searchsorted(end_index, drift_index, side="left"))


def audit_machine(
    machine: SMDMachine,
    features_config: dict[str, Any],
    scenarios: list[DriftScenario],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Decrit la derive injectee sur une machine, scenario par scenario."""
    rows: list[dict[str, Any]] = []
    n_points = len(machine.test)
    interval = machine.sampling_interval_seconds

    for scenario in scenarios:
        for seed in seeds:
            flux = inject(
                train=machine.train,
                test=machine.test,
                label=machine.test_label,
                scenario=scenario,
                seed=seed,
                sampling_interval_seconds=interval,
                stream_key=machine.name,
            )
            windowed = make_windows(flux.series, features_config, label=flux.label)
            drift_window = _drift_window(windowed.end_index, flux.drift_index)

            rows.append(
                {
                    "machine": machine.name,
                    "scenario": scenario.name,
                    "profil": scenario.profil,
                    "seed": seed,
                    "n_points": n_points,
                    "n_fenetres": len(windowed),
                    "drift_index": flux.drift_index,
                    "drift_position": round(flux.drift_index / n_points, 4),
                    "drift_heures": round(flux.drift_index * interval / 3600, 2),
                    "drift_fenetre": drift_window,
                    "duree_transition_points": flux.duree_transition,
                    "duree_transition_heures": round(flux.duree_transition * interval / 3600, 2),
                    "n_metriques_touchees": len(flux.metriques),
                    "metriques_touchees": " ".join(str(m) for m in flux.metriques),
                    "part_ecretee": round(flux.part_ecretee, 6),
                    "part_temps_regime_transforme": round(float(flux.alpha.mean()), 4),
                    "level_shift_k": scenario.level_shift_k,
                    "scale_factor": scenario.scale_factor,
                    "n_fenetres_avant": drift_window,
                    "n_fenetres_apres": len(windowed) - drift_window,
                    "taux_positif_avant": round(float(windowed.y[:drift_window].mean()), 4)
                    if drift_window > 0
                    else None,
                    "taux_positif_apres": round(float(windowed.y[drift_window:].mean()), 4)
                    if drift_window < len(windowed)
                    else None,
                }
            )
    return rows


def audit(
    dataset_config: str = "datasets/smd_dev.yaml",
    features_config: str = "features.yaml",
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    destination: Any = INJECTIONS_PATH,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Decrit la derive injectee sur toutes les machines de la configuration."""
    dataset = load_config(dataset_config)
    features = load_config(features_config)
    # Le garde-fou d'ecretage est neutralise ici : la part ecretee est la
    # grandeur que l'audit mesure, elle ne doit pas interrompre la mesure. Le
    # depassement de la tolerance de production est rapporte colonne
    # `ecretage_au_dela_tolerance`, et le garde-fou reste actif en campagne.
    bases = [load_scenario(f"scenarios/{name}.yaml") for name in (scenarios or SCENARIOS)]
    chosen = [replace(base, part_ecretee_max=1.0) for base in bases]
    tolerances = {base.name: base.part_ecretee_max for base in bases}
    graines = seeds or [42]

    rows: list[dict[str, Any]] = []
    for name in dataset["machines"]:
        machine = load_machine(name, dataset_config)
        for row in audit_machine(machine, features, chosen, graines):
            tolerance = tolerances[row["scenario"]]
            row["part_ecretee_max_tolere"] = tolerance
            row["ecretage_au_dela_tolerance"] = bool(row["part_ecretee"] > tolerance)
            rows.append(row)
            if on_row is not None:
                on_row(row)

    table = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table


def calibrate_k_machine(
    machine: SMDMachine,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    scenario: DriftScenario,
    k_grid: list[float],
    seeds: list[int],
    min_segment_windows: int,
) -> list[dict[str, Any]]:
    """Mesure la degradation de B0 en fonction de k sur une machine.

    Le modele est entraine une seule fois : B0 est par definition fige, et la
    moitie `train` n'est pas transformee par l'injection.
    """
    zero_division = float(evaluation_config.get("zero_division", {}).get("value", 0.0))
    interval = machine.sampling_interval_seconds

    train = make_windows(machine.train, features_config)
    detector = build_detector(model_config, features_config, random_state=seeds[0])
    detector.fit(train.X)

    # Reference sans injection : le F1 que B0 obtient sur le flux d'origine.
    reference = make_windows(machine.test, features_config, label=machine.test_label)
    reference_f1 = window_metrics(reference.y, detector.predict(reference.X), zero_division).f1

    rows: list[dict[str, Any]] = []
    for k in k_grid:
        for seed in seeds:
            started = time.perf_counter()
            flux = inject(
                train=machine.train,
                test=machine.test,
                label=machine.test_label,
                scenario=replace(scenario, level_shift_k=k),
                seed=seed,
                sampling_interval_seconds=interval,
                stream_key=machine.name,
            )
            windowed = make_windows(flux.series, features_config, label=flux.label)
            alerts = detector.predict(windowed.X)
            cut = _drift_window(windowed.end_index, flux.drift_index)

            avant = slice(0, cut)
            apres = slice(cut, len(windowed))
            assez_avant = cut >= min_segment_windows
            assez_apres = len(windowed) - cut >= min_segment_windows

            f1_avant = (
                window_metrics(windowed.y[avant], alerts[avant], zero_division).f1
                if assez_avant
                else None
            )
            f1_apres = (
                window_metrics(windowed.y[apres], alerts[apres], zero_division).f1
                if assez_apres
                else None
            )
            global_metrics = window_metrics(windowed.y, alerts, zero_division)

            rows.append(
                {
                    "machine": machine.name,
                    "scenario": scenario.name,
                    "k": k,
                    "scale_factor": scenario.scale_factor,
                    "seed": seed,
                    "n_fenetres": len(windowed),
                    "drift_fenetre": cut,
                    "n_metriques_touchees": len(flux.metriques),
                    "part_ecretee": round(flux.part_ecretee, 6),
                    "f1_reference_sans_injection": round(reference_f1, 4),
                    "f1_avant": None if f1_avant is None else round(f1_avant, 4),
                    "f1_apres": None if f1_apres is None else round(f1_apres, 4),
                    "f1_global": round(global_metrics.f1, 4),
                    "precision_apres": None
                    if not assez_apres
                    else round(
                        window_metrics(windowed.y[apres], alerts[apres], zero_division).precision,
                        4,
                    ),
                    "rappel_apres": None
                    if not assez_apres
                    else round(
                        window_metrics(windowed.y[apres], alerts[apres], zero_division).recall, 4
                    ),
                    "ratio_f1": None
                    if not (assez_avant and assez_apres) or not f1_avant
                    else round(f1_apres / f1_avant, 4),
                    "taux_alertes_avant": None
                    if not assez_avant
                    else round(float(alerts[avant].mean()), 4),
                    "taux_alertes_apres": None
                    if not assez_apres
                    else round(float(alerts[apres].mean()), 4),
                    "secondes": round(time.perf_counter() - started, 2),
                }
            )
    return rows


def calibrate_k(
    dataset_config: str = "datasets/smd_dev.yaml",
    features_config: str = "features.yaml",
    model_config: str = "model.yaml",
    evaluation_config: str = "evaluation.yaml",
    calibration_config: str = "calibration_k.yaml",
    destination: Any = K_CALIBRATION_PATH,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Calibre k sur les machines de mise au point. Produit D22."""
    dataset = load_config(dataset_config)
    features = load_config(features_config)
    model = load_config(model_config)
    evaluation = load_config(evaluation_config)
    settings = load_config(calibration_config)

    base = load_scenario(settings["scenario"])
    # Le garde-fou d'ecretage est desactive pendant le calibrage : la part
    # ecretee est precisement la grandeur mesuree, elle ne doit pas interrompre
    # la mesure. La configuration de production garde sa limite.
    scenario = replace(base, part_ecretee_max=float(settings["part_ecretee_max"]))

    rows: list[dict[str, Any]] = []
    for name in dataset["machines"]:
        machine = load_machine(name, dataset_config)
        for row in calibrate_k_machine(
            machine=machine,
            features_config=features,
            model_config=model,
            evaluation_config=evaluation,
            scenario=scenario,
            k_grid=[float(k) for k in settings["k_grid"]],
            seeds=[int(s) for s in settings["seeds"]],
            min_segment_windows=int(settings["min_segment_windows"]),
        ):
            rows.append(row)
            if on_row is not None:
                on_row(row)

    table = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table


def k_verdict(table: pd.DataFrame, calibration_config: dict[str, Any]) -> pd.DataFrame:
    """Resume par k : perturbation obtenue et part ecretee. Lu pour decider D22.

    Le critere central est le **taux d'alertes** de B0 apres t_d, rapporte a
    celui d'avant. Le rapport des F1 est conserve a titre indicatif : sur le
    perimetre de mise au point, son denominateur est quasi nul (F1 avant = 0 sur
    4 machines, 0,0047 sur machine-2-7) et il ne classe rien. Ce que la derive
    fait au modele fige est un afflux de fausses alertes, pas une perte de
    rappel.
    """
    lecture = calibration_config.get("lecture", {})
    facteur_min = float(lecture.get("facteur_alertes_min", 3.0))
    sature = float(lecture.get("taux_alertes_sature", 0.50))
    ecretage_cible = float(lecture.get("part_ecretee_cible_max", 0.15))
    ratio_cible = float(lecture.get("ratio_f1_cible_max", 0.85))

    # Moyenne par machine d'abord, puis agregation : les graines d'une meme
    # machine ne sont pas des observations independantes (OBS06).
    par_machine = table.groupby(["machine", "k"], as_index=False).agg(
        ratio_f1=("ratio_f1", "mean"),
        f1_avant=("f1_avant", "mean"),
        f1_apres=("f1_apres", "mean"),
        precision_apres=("precision_apres", "mean"),
        rappel_apres=("rappel_apres", "mean"),
        part_ecretee=("part_ecretee", "mean"),
        taux_alertes_avant=("taux_alertes_avant", "mean"),
        taux_alertes_apres=("taux_alertes_apres", "mean"),
    )
    par_machine["facteur_alertes"] = (
        par_machine["taux_alertes_apres"] / par_machine["taux_alertes_avant"]
    )

    resume = par_machine.groupby("k", as_index=False).agg(
        n_machines=("machine", "nunique"),
        taux_alertes_avant_median=("taux_alertes_avant", "median"),
        taux_alertes_apres_median=("taux_alertes_apres", "median"),
        facteur_alertes_median=("facteur_alertes", "median"),
        precision_apres_median=("precision_apres", "median"),
        rappel_apres_median=("rappel_apres", "median"),
        f1_avant_median=("f1_avant", "median"),
        f1_apres_median=("f1_apres", "median"),
        ratio_f1_median=("ratio_f1", "median"),
        part_ecretee_mediane=("part_ecretee", "median"),
        part_ecretee_max=("part_ecretee", "max"),
    )

    def compte(masque: Callable[[pd.DataFrame], pd.Series]) -> list[int]:
        return [int(masque(par_machine.loc[par_machine["k"] == k]).sum()) for k in resume["k"]]

    resume["n_machines_perturbees"] = compte(
        lambda g: g["taux_alertes_apres"] >= facteur_min * g["taux_alertes_avant"]
    )
    resume["n_machines_saturees"] = compte(lambda g: g["taux_alertes_apres"] >= sature)
    resume["n_machines_ecretage_excessif"] = compte(lambda g: g["part_ecretee"] > ecretage_cible)
    # Indicatif : le rapport n'est defini que la ou F1 avant est non nul.
    resume["n_machines_ratio_exploitable"] = compte(lambda g: g["ratio_f1"].notna())
    resume["n_machines_degradees"] = compte(lambda g: g["ratio_f1"] <= ratio_cible)
    return resume.round(4)
