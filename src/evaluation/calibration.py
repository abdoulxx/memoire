"""Mesures de mise au point pour les etapes 3 et 4 de la strategie proposee (D13, D14).

Ce module refait, sur le **seul perimetre de mise au point** (D17), les deux mesures
qui avaient initialement ete faites sur machine-1-1, machine-2-1 et machine-3-1 :
machine-2-1 appartient au perimetre d'evaluation, les mesures etaient donc
contaminees. Il produit en outre les bornes du garde-fou de l'etape 4, que D14
laissait a calibrer.

Protocole, pour chaque machine et chaque instant de reentrainement t_r :

    |---- W fenetres candidates a l'entrainement ----|-- V validation --| t_r |-- aval --|

- Le champion est le modele B0 (`mdrift baseline`). Ses scores et alertes sont relus
  depuis `traces.npz` : c'est exactement ce dont la strategie disposerait a cet instant.
- **Plusieurs instants t_r par machine.** Le premier est celui que donnerait l'etape de
  confirmation ; les autres sont des positions relatives fixees dans la config. C'est
  necessaire : a la premiere confirmation, le champion n'a presque pas encore alerte,
  donc les regles d'exclusion sont indistinguables (elles gardent les memes fenetres).
  L'effondrement du challenger mesure en D13 apparait aux reentrainements **tardifs**,
  quand le champion sature.
- La moyenne glissante du taux d'alertes est **causale**, et compte **toutes** les
  alertes : la strategie n'a pas les etiquettes. C'est l'inverse de la mesure
  descriptive d'OBS03, qui en dispose et ne compte que les fausses.
- Les fenetres de validation sont retirees de l'entrainement du challenger.
- Les regles qui gardent exactement les memes fenetres sont fusionnees : sans cela,
  le meme modele compterait plusieurs fois dans la correlation de l'etape 4.
- Les etiquettes ne servent qu'a l'**evaluation aval** et a la calibration des bornes,
  jamais a une decision de la strategie.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from datasets.smd.loader import load_machines
from src.config import load_config
from src.evaluation.metrics import false_alerts_per_hour, window_metrics
from src.evaluation.replay import RESULTS_DIR
from src.features.windowing import make_windows
from src.models.isolation_forest import build_detector

CALIBRATION_PATH = RESULTS_DIR / "calibration_dev.csv"


@dataclass(frozen=True)
class Protocol:
    """Parametres du protocole, lus dans `configs/calibration.yaml`."""

    w_train: int
    v_valid: int
    confirm_ratio: float
    confirm_k: int
    rolling: int
    min_train_windows: int
    min_downstream_windows: int
    retrain_points: tuple[float, ...]
    include_confirmation: bool
    seeds: tuple[int, ...]
    healthy_f1_margin: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Protocol:
        return cls(
            w_train=int(config["w_train"]),
            v_valid=int(config["v_valid"]),
            confirm_ratio=float(config["confirm_ratio"]),
            confirm_k=int(config["confirm_k"]),
            rolling=int(config["rolling"]),
            min_train_windows=int(config["min_train_windows"]),
            min_downstream_windows=int(config["min_downstream_windows"]),
            retrain_points=tuple(float(p) for p in config["retrain_points"]),
            include_confirmation=bool(config["include_confirmation"]),
            seeds=tuple(int(s) for s in config["seeds"]),
            healthy_f1_margin=float(config["healthy_f1_margin"]),
        )


def _bursts(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int64), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def confirmation_point(alerts: np.ndarray, protocol: Protocol, target_rate: float) -> int | None:
    """Premiere fenetre ou le taux d'alertes glissant causal reste eleve K fenetres.

    La moyenne est **causale** : la valeur attribuee a la fenetre t n'utilise que
    les fenetres <= t. Une moyenne centree anticiperait de `rolling / 2` fenetres
    (voir OBS03 dans docs/decisions.md).
    """
    window = protocol.rolling
    if len(alerts) < window:
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(alerts, dtype=np.float64)))
    means = (cumulative[window:] - cumulative[:-window]) / window
    high = means >= protocol.confirm_ratio * target_rate
    for start, end in _bursts(high):
        if end - start + 1 >= protocol.confirm_k:
            return int(start + protocol.confirm_k - 1 + window - 1)
    return None


def retrain_instants(
    alerts: np.ndarray,
    n_windows: int,
    protocol: Protocol,
    target_rate: float,
) -> list[tuple[str, int]]:
    """Instants de reentrainement a visiter, nommes puis tries, sans doublon."""
    instants: list[tuple[str, int]] = []
    if protocol.include_confirmation:
        t_r = confirmation_point(alerts, protocol, target_rate)
        if t_r is not None:
            instants.append(("confirmation", t_r))
    for fraction in protocol.retrain_points:
        instants.append((f"t{int(round(fraction * 100))}pc", int(round(fraction * n_windows))))

    kept: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name, position in sorted(instants, key=lambda item: item[1]):
        if position in seen:
            continue
        seen.add(position)
        kept.append((name, position))
    return kept


def exclusion_rules(
    champion_alerts: np.ndarray,
    champion_scores: np.ndarray,
    lo: int,
    hi: int,
) -> dict[str, np.ndarray]:
    """Regles d'exclusion candidates, appliquees aux fenetres [lo, hi).

    Les regles qui gardent exactement les memes fenetres sont fusionnees, leur nom
    etant joint par `=`. Sans cette fusion, un meme modele serait compte plusieurs
    fois dans la correlation de l'etape 4 et la gonflerait artificiellement.
    """
    idx = np.arange(lo, hi)
    alerts = champion_alerts[lo:hi].astype(bool)
    scores = champion_scores[lo:hi]

    raw: dict[str, np.ndarray] = {
        "R0_aucune_exclusion": idx.copy(),
        "R1_exclut_toutes_alertes": idx[~alerts],
    }
    for k in (10, 30, 60):
        short = np.zeros(hi - lo, dtype=bool)
        for start, end in _bursts(alerts):
            if end - start + 1 < k:
                short[start : end + 1] = True
        raw[f"R2_exclut_rafales_<{k}"] = idx[~short]
    for pct in (1, 5, 10):
        raw[f"R3_exclut_top_{pct}pc"] = idx[scores < np.percentile(scores, 100 - pct)]

    merged: dict[bytes, tuple[list[str], np.ndarray]] = {}
    for name, kept in raw.items():
        key = kept.tobytes()
        if key in merged:
            merged[key][0].append(name)
        else:
            merged[key] = ([name], kept)
    return {"=".join(names): kept for names, kept in merged.values()}


def _trace_path(machine_name: str) -> Path:
    return RESULTS_DIR / "runs" / "smd" / machine_name / "none" / "static" / "seed42" / "traces.npz"


def measure_machine(
    machine: Any,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    protocol: Protocol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mesure une machine. Renvoie une ligne par (instant, regle), et le contexte par instant."""
    test = make_windows(machine.test, features_config, label=machine.test_label)
    if test.y is None:
        raise ValueError(f"{machine.name} : etiquettes de test absentes.")

    path = _trace_path(machine.name)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} absent : lancer `mdrift baseline --dataset-config datasets/smd_dev.yaml`."
        )
    traces = np.load(path)
    champion_alerts, champion_scores = traces["alerts"], traces["scores"]
    if len(champion_alerts) != len(test):
        raise ValueError(f"{machine.name} : traces et fenetres de longueurs differentes.")

    target_rate = 1.0 - model_config["threshold"]["percentile"] / 100.0
    hours = machine.sampling_interval_seconds / 3600
    rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []

    for instant, t_r in retrain_instants(champion_alerts, len(test), protocol, target_rate):
        context: dict[str, Any] = {
            "machine": machine.name,
            "instant": instant,
            "t_r": t_r,
            "t_r_heures": round(t_r * hours, 1),
            "raison": None,
        }
        val_lo, val_hi = t_r - protocol.v_valid, t_r
        tr_lo, tr_hi = max(0, val_lo - protocol.w_train), val_lo
        if tr_hi - tr_lo < protocol.min_train_windows:
            context["raison"] = "historique insuffisant"
            contexts.append(context)
            continue
        if len(test) - t_r < protocol.min_downstream_windows:
            context["raison"] = "aval trop court"
            contexts.append(context)
            continue

        champion_downstream = window_metrics(test.y[t_r:], champion_alerts[t_r:])
        context |= {
            "aval_n_fenetres": len(test) - t_r,
            "aval_taux_positif": round(float(test.y[t_r:].mean()), 4),
            "champion_aval_f1": round(champion_downstream.f1, 4),
            "champion_aval_taux_alertes": round(float(champion_alerts[t_r:].mean()), 4),
            "champion_val_taux_alertes": round(float(champion_alerts[val_lo:val_hi].mean()), 4),
        }
        contexts.append(context)

        instant_rows: list[dict[str, Any]] = []
        rules = exclusion_rules(champion_alerts, champion_scores, tr_lo, tr_hi)
        for name, kept in rules.items():
            if len(kept) < protocol.min_train_windows:
                continue
            val_rates: list[float] = []
            ks_stats: list[float] = []
            f1s: list[float] = []
            precisions: list[float] = []
            recalls: list[float] = []
            down_rates: list[float] = []
            fa_hours: list[float] = []
            for seed in protocol.seeds:
                detector = build_detector(model_config, features_config, seed).fit(test.X[kept])
                train_scores = detector.score(test.X[kept])
                val_scores = detector.score(test.X[val_lo:val_hi])
                val_rates.append(float((val_scores > detector.threshold).mean()))
                ks_stats.append(float(ks_2samp(train_scores, val_scores, method="asymp").statistic))
                predicted = detector.predict(test.X[t_r:])
                metrics = window_metrics(test.y[t_r:], predicted)
                f1s.append(metrics.f1)
                precisions.append(metrics.precision)
                recalls.append(metrics.recall)
                down_rates.append(float(predicted.mean()))
                fa_hours.append(
                    false_alerts_per_hour(
                        metrics.false_positives,
                        len(predicted),
                        features_config["stride"],
                        machine.sampling_interval_seconds,
                    )
                )
            instant_rows.append(
                {
                    "machine": machine.name,
                    "instant": instant,
                    "t_r": t_r,
                    "t_r_heures": round(t_r * hours, 1),
                    "regle": name,
                    "n_regles_equivalentes": name.count("=") + 1,
                    "n_gardees": int(len(kept)),
                    "part_gardee": round(len(kept) / (tr_hi - tr_lo), 4),
                    "val_taux_alertes": round(float(np.mean(val_rates)), 4),
                    "c1_ecart_au_cible": round(abs(float(np.mean(val_rates)) - target_rate), 4),
                    "c2_ks_train_val": round(float(np.mean(ks_stats)), 4),
                    "aval_f1": round(float(np.mean(f1s)), 4),
                    "aval_f1_std": round(float(np.std(f1s)), 4),
                    "aval_precision": round(float(np.mean(precisions)), 4),
                    "aval_recall": round(float(np.mean(recalls)), 4),
                    "aval_taux_alertes": round(float(np.mean(down_rates)), 4),
                    "aval_fa_par_heure": round(float(np.mean(fa_hours)), 2),
                    "champion_aval_f1": context["champion_aval_f1"],
                    "champion_val_taux_alertes": context["champion_val_taux_alertes"],
                    "aval_taux_positif": context["aval_taux_positif"],
                }
            )

        if instant_rows:
            best = max(row["aval_f1"] for row in instant_rows)
            for row in instant_rows:
                row["sain"] = bool(row["aval_f1"] >= best - protocol.healthy_f1_margin)
                row["bat_le_champion"] = bool(row["aval_f1"] > row["champion_aval_f1"])
            rows.extend(instant_rows)

    return rows, contexts


def calibrate(
    dataset_config: str = "datasets/smd_dev.yaml",
    features_config: str = "features.yaml",
    model_config: str = "model.yaml",
    calibration_config: str = "calibration.yaml",
    output_path: Path = CALIBRATION_PATH,
    on_context: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Mesure toutes les machines du perimetre et ecrit le tableau des resultats."""
    features = load_config(features_config)
    model = load_config(model_config)
    protocol = Protocol.from_config(load_config(calibration_config))

    rows: list[dict[str, Any]] = []
    for machine in load_machines(dataset_config):
        machine_rows, contexts = measure_machine(machine, features, model, protocol)
        rows.extend(machine_rows)
        if on_context is not None:
            for context in contexts:
                on_context(context)

    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _no_exclusion(table: pd.DataFrame) -> pd.Series:
    """Ligne de la regle sans exclusion, qui peut etre fusionnee avec d'autres."""
    return table["regle"].str.contains("R0_aucune_exclusion", regex=False)


def exclusion_verdict(table: pd.DataFrame) -> dict[str, Any]:
    """Conclusion de D13 : ne rien exclure coute-t-il du F1 aval ?"""
    if table.empty:
        return {}
    gaps: list[float] = []
    collapses: list[dict[str, Any]] = []
    for (machine, instant), group in table.groupby(["machine", "instant"]):
        none_rows = group.loc[_no_exclusion(group)]
        if none_rows.empty:
            continue
        none_f1 = float(none_rows["aval_f1"].iloc[0])
        gaps.append(float(group["aval_f1"].max()) - none_f1)
        drop = group.loc[group["regle"].str.contains("R1_exclut_toutes_alertes", regex=False)]
        if not drop.empty and none_f1 > 0:
            row = drop.iloc[0]
            collapses.append(
                {
                    "machine": str(machine),
                    "instant": str(instant),
                    "part_gardee": float(row["part_gardee"]),
                    "f1_R1": float(row["aval_f1"]),
                    "f1_R0": none_f1,
                    "ratio": round(float(row["aval_f1"]) / none_f1, 3),
                }
            )
    gaps_array = np.array(gaps)
    severe = [c for c in collapses if c["ratio"] < 0.5]
    return {
        "n_decisions": int(len(gaps_array)),
        "n_machines": int(table["machine"].nunique()),
        "ecart_median_au_meilleur": round(float(np.median(gaps_array)), 4),
        "ecart_max_au_meilleur": round(float(gaps_array.max()), 4),
        "n_fois_R0_optimale": int((gaps_array <= 1e-9).sum()),
        "ecart_type_inter_graines_median": round(float(table["aval_f1_std"].median()), 4),
        "ecart_type_inter_graines_max": round(float(table["aval_f1_std"].max()), 4),
        "n_R1_distincte": len(collapses),
        "n_R1_effondrement": len(severe),
        "effondrements": sorted(severe, key=lambda c: c["ratio"])[:5],
    }


def validation_verdict(table: pd.DataFrame) -> dict[str, Any]:
    """Conclusion de D14 : les criteres sans etiquettes classent-ils les challengers ?

    On attend une correlation **negative** : un critere faible devrait signaler un
    bon modele. Une correlation positive signifie que le critere choisit le pire.
    Les decisions ou moins de trois challengers sont distincts sont ecartees : un
    classement n'a pas de sens sur deux candidats.
    """
    if table.empty:
        return {}
    per_decision: list[dict[str, Any]] = []
    for (machine, instant), group in table.groupby(["machine", "instant"]):
        distinct = group.drop_duplicates(subset=["aval_f1", "c1_ecart_au_cible"])
        if len(distinct) < 3 or distinct["c1_ecart_au_cible"].nunique() < 2:
            continue
        rho1 = spearmanr(distinct["c1_ecart_au_cible"], distinct["aval_f1"]).statistic
        rho2 = spearmanr(distinct["c2_ks_train_val"], distinct["aval_f1"]).statistic
        chosen = distinct.loc[distinct["c1_ecart_au_cible"].idxmin()]
        per_decision.append(
            {
                "machine": str(machine),
                "instant": str(instant),
                "n_candidats": int(len(distinct)),
                "rho_c1": None if np.isnan(rho1) else round(float(rho1), 2),
                "rho_c2": None if np.isnan(rho2) else round(float(rho2), 2),
                "c1_choisit": str(chosen["regle"]),
                "f1_du_choix": float(chosen["aval_f1"]),
                "f1_du_meilleur": float(distinct["aval_f1"].max()),
            }
        )
    rho1_values = [d["rho_c1"] for d in per_decision if d["rho_c1"] is not None]
    rho2_values = [d["rho_c2"] for d in per_decision if d["rho_c2"] is not None]
    lost = [
        1 - d["f1_du_choix"] / d["f1_du_meilleur"] for d in per_decision if d["f1_du_meilleur"] > 0
    ]
    return {
        "par_decision": per_decision,
        "n_decisions": len(per_decision),
        "rho_c1_median": round(float(np.median(rho1_values)), 2) if rho1_values else None,
        "n_rho_c1_positif": sum(1 for r in rho1_values if r > 0),
        "rho_c2_median": round(float(np.median(rho2_values)), 2) if rho2_values else None,
        "n_rho_c2_positif": sum(1 for r in rho2_values if r > 0),
        "perte_f1_mediane_du_choix_c1": round(float(np.median(lost)), 4) if lost else None,
        "perte_f1_max_du_choix_c1": round(float(np.max(lost)), 4) if lost else None,
        "n_choix_est_le_pire": sum(
            1
            for d in per_decision
            if d["f1_du_choix"]
            == min(
                row["aval_f1"]
                for _, row in table.loc[
                    (table["machine"] == d["machine"]) & (table["instant"] == d["instant"])
                ].iterrows()
            )
        ),
    }


def retrain_verdict(table: pd.DataFrame) -> dict[str, Any]:
    """Reentrainer bat-il le champion fige, et de combien ?"""
    if table.empty:
        return {}
    none_rows = table.loc[_no_exclusion(table)]
    ratios = [
        float(row["aval_f1"]) / float(row["champion_aval_f1"])
        for _, row in none_rows.iterrows()
        if row["champion_aval_f1"] > 0
    ]
    return {
        "n_decisions": int(len(none_rows)),
        "n_bat_le_champion": int(none_rows["bat_le_champion"].sum()),
        "gain_median": round(float(np.median(ratios)), 2) if ratios else None,
        "gain_max": round(float(np.max(ratios)), 2) if ratios else None,
        "gain_min": round(float(np.min(ratios)), 2) if ratios else None,
    }


def guardrail_bounds(table: pd.DataFrame, model_config: dict[str, Any]) -> dict[str, Any]:
    """Bornes de l'etape 4, deduites des challengers sains et degrades (D14).

    `alert_rate_max` doit rejeter les challengers degrades qui **saturent** sans
    rejeter aucun challenger sain ; `alert_rate_min` est symetrique pour ceux qui
    s'eteignent. Si les deux populations se chevauchent, aucune borne ne les separe
    et il faut le dire : c'est le constat de D14, le garde-fou n'ecarte que
    l'aberrant, pas le mediocre.
    """
    if table.empty or "sain" not in table:
        return {}
    healthy = table.loc[table["sain"], "val_taux_alertes"]
    degraded = table.loc[~table["sain"], "val_taux_alertes"]
    high, low = float(healthy.max()), float(healthy.min())
    bounds: dict[str, Any] = {
        "n_sains": int(len(healthy)),
        "n_degrades": int(len(degraded)),
        "sain_taux_min": round(low, 4),
        "sain_taux_max": round(high, 4),
        "degrade_taux_min": round(float(degraded.min()), 4) if len(degraded) else None,
        "degrade_taux_max": round(float(degraded.max()), 4) if len(degraded) else None,
        "alert_rate_min_propose": round(low, 4),
        "alert_rate_max_propose": round(high, 4),
        "min_train_windows": int(model_config["isolation_forest"]["max_samples"]),
    }
    if len(degraded):
        bounds["degrades_ecartes_par_borne_haute"] = int((degraded > high).sum())
        bounds["degrades_ecartes_par_borne_basse"] = int((degraded < low).sum())
        bounds["degrades_non_ecartes"] = int(((degraded >= low) & (degraded <= high)).sum())
        bounds["separation_complete"] = bool(bounds["degrades_non_ecartes"] == 0)
    return bounds
