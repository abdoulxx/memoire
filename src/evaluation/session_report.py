"""Rapport detaille d'une session rejouee (jalon M6, complement a summary.csv).

`results/summary.csv` porte une ligne par strategie, au meme format que la
campagne SMD. Ce module ajoute ce que ce format ne porte pas et que le
memoire demande specifiquement pour la session reelle :

- le nombre de fenetres de la session (total, entrainement, evaluation) ;
- l'AP sur la marge score - seuil (PR-AUC), absente de `summary.csv` (D48) ;
- pour P et A1, l'instant REEL (horodatage de la session, pas un indice de
  fenetre) de la confirmation de la derive du 21/11 et du reentrainement s'il
  a lieu dans la session ;
- un tableau par anomalie etiquetee : detectee ou non, et avec quel delai, par
  chaque strategie, avec les etiquettes brutes (`data/labels/<run_id>.csv`)
  qui composent chaque evenement — c'est ce qui rend visible le doublon
  `pic_cpu` du 23/11.

Rien n'est recalcule depuis les donnees brutes : tout vient de
`results/runs/serveur/<run_id>/.../{traces.npz,events.jsonl}`, ecrits par
`mdrift session-replay`. Un rapport correspond donc toujours a un rejeu
identifie, jamais a un calcul refait pour l'occasion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.serveur.loader import SessionServeur, instant
from datasets.serveur.settings import ReglagesServeur
from infra.injector.labels import lire_anomalies
from src.config import load_config
from src.evaluation.margins import pr_auc_from_traces
from src.evaluation.metrics import event_metrics
from src.evaluation.replay import RESULTS_DIR, SUMMARY_PATH, ReplayStream, run_directory
from src.evaluation.session_campaign import load_session_stream

# Un sous-dossier par session, comme `results/runs/` : plusieurs sessions
# enregistrees (M6 n'en impose qu'une, mais rien ne l'interdit) ne s'ecrasent
# pas les unes les autres.
REPORT_DIR = RESULTS_DIR / "session_reports"

STRATEGIES = ("static", "periodic", "proposed", "ablation")
# Les strategies qui portent une surveillance de derive : seules elles peuvent
# avoir confirme une derive ou reentraine dessus.
STRATEGIES_AVEC_DERIVE = ("proposed", "ablation")

# Colonnes de `summary.csv` recopiees dans le rapport, pour que la table des
# strategies soit lisible sans ouvrir deux fichiers. Les noms sont ceux de
# `EvaluationResult.to_row()` (metrics.py) et de `evaluate_promotions`.
COLONNES_SUMMARY = (
    "precision",
    "recall",
    "f1",
    "precision_before",
    "recall_before",
    "f1_before",
    "precision_after",
    "recall_after",
    "f1_after",
    "n_events",
    "n_events_detected",
    "event_recall",
    "detection_delay_median_minutes",
    "false_alerts_per_hour",
    "duration_hours",
    "cost_n_retrains",
    "cost_n_promoted",
    "cost_n_rejected",
)


@dataclass(frozen=True)
class RunArtefacts:
    """Ce qui a ete ecrit par `session-replay` pour une strategie."""

    strategy: str
    directory: Path
    traces: dict[str, np.ndarray]
    event_log: list[dict[str, Any]]


def _load_artefacts(stream: ReplayStream, strategy: str, seed: int) -> RunArtefacts | None:
    directory = run_directory(stream, strategy, seed)
    chemin_traces = directory / "traces.npz"
    chemin_events = directory / "events.jsonl"
    if not chemin_traces.is_file():
        return None
    with np.load(chemin_traces) as data:
        traces = {name: data[name] for name in data.files}
    event_log: list[dict[str, Any]] = []
    if chemin_events.is_file():
        with chemin_events.open("r", encoding="utf-8") as handle:
            event_log = [json.loads(ligne) for ligne in handle if ligne.strip()]
    return RunArtefacts(strategy=strategy, directory=directory, traces=traces, event_log=event_log)


def _point_to_timestamp(session: SessionServeur, n_train_points: int, point_index: int) -> str:
    """Indice de point du flux evalue -> horodatage RFC 3339 UTC de la session."""
    from infra.injector.labels import horodatage

    return horodatage(float(session.instants[n_train_points + point_index]))


def _window_to_real_ts(
    session: SessionServeur, stream: ReplayStream, n_train_points: int, window_position: int
) -> str | None:
    if window_position is None or window_position >= len(stream.end_index):
        return None
    point_index = int(stream.end_index[window_position])
    return _point_to_timestamp(session, n_train_points, point_index)


def window_counts(stream: ReplayStream) -> dict[str, int]:
    """Nombre de fenetres de la session : total, entrainement, evaluation."""
    n_train = len(stream.train_X)
    n_eval = len(stream)
    return {
        "n_windows_train": n_train,
        "n_windows_eval": n_eval,
        "n_windows_total": n_train + n_eval,
    }


def strategy_summary(
    stream: ReplayStream,
    session: SessionServeur,
    n_train_points: int,
    seed: int,
    strategies: tuple[str, ...] = STRATEGIES,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    """Une ligne par strategie : metriques de `summary.csv` + PR-AUC + horodatages reels.

    Les colonnes `drift_confirmed_at` / `retrain_at` ne sont renseignees que
    pour les strategies qui surveillent une derive (P, A1) ; elles valent None
    pour B0 et B1, qui n'emettent jamais ces evenements.
    """
    synthese = pd.read_csv(summary_path) if Path(summary_path).is_file() else pd.DataFrame()

    lignes: list[dict[str, Any]] = []
    for strategie in strategies:
        artefacts = _load_artefacts(stream, strategie, seed)
        ligne: dict[str, Any] = {"strategy": strategie}
        if artefacts is None:
            ligne["disponible"] = False
            lignes.append(ligne)
            continue
        ligne["disponible"] = True

        if not synthese.empty:
            correspondance = synthese[
                (synthese.source == stream.source)
                & (synthese.machine == stream.machine)
                & (synthese.scenario == stream.scenario)
                & (synthese.strategy == strategie)
                & (synthese.seed == seed)
            ]
            if not correspondance.empty:
                for colonne in COLONNES_SUMMARY:
                    if colonne in correspondance.columns:
                        ligne[colonne] = correspondance.iloc[0][colonne]

        pr_auc = pr_auc_from_traces(artefacts.directory / "traces.npz")
        ligne["pr_auc_margin"] = pr_auc["pr_auc_margin"]

        confirmations = [e for e in artefacts.event_log if e.get("event") == "drift_confirmed"]
        retrains = [e for e in artefacts.event_log if e.get("event") == "retrain_start"]
        promotions = [e for e in artefacts.event_log if e.get("event") == "promoted"]

        if strategie in STRATEGIES_AVEC_DERIVE:
            premiere_confirmation = confirmations[0] if confirmations else None
            ligne["drift_confirmed_window"] = (
                premiere_confirmation["details"].get("position") if premiere_confirmation else None
            )
            ligne["drift_confirmed_at"] = (
                _window_to_real_ts(
                    session, stream, n_train_points, int(ligne["drift_confirmed_window"])
                )
                if premiere_confirmation is not None
                else None
            )
            ligne["n_retrains_in_session"] = len(retrains)
            if retrains:
                premier_reentrainement = retrains[0]
                position = int(premier_reentrainement["details"]["position"])
                ligne["retrain_window"] = position
                ligne["retrain_at"] = _window_to_real_ts(session, stream, n_train_points, position)
            else:
                ligne["retrain_window"] = None
                ligne["retrain_at"] = None
            ligne["n_promotions_in_session"] = len(promotions)
        lignes.append(ligne)

    return pd.DataFrame(lignes)


def _label_rows_for_event(
    etiquettes_brutes: list[dict[str, str]],
    event_start_point: int,
    event_end_point: int,
    n_train_points: int,
    session: SessionServeur,
) -> list[dict[str, str]]:
    """Etiquettes brutes (`data/labels/<run_id>.csv`) dont la plage touche cet evenement.

    Un evenement fusionne toute etiquette brute dont la plage chevauche ou
    touche celle d'une autre (c'est la regle de `events_from_label`, une plage
    contigue de points etiquetes 1). Une seule etiquette donne un seul evenement
    ; plusieurs etiquettes qui se chevauchent dans le temps (le doublon
    `pic_cpu` du 23/11) donnent UN SEUL evenement, et apparaissent ici comme
    plusieurs lignes brutes pour le meme `event_index`.
    """
    debut_instant = float(session.instants[n_train_points + event_start_point])
    fin_instant = float(session.instants[n_train_points + event_end_point])
    touchees = []
    for ligne in etiquettes_brutes:
        debut = instant(ligne["start_ts"])
        fin = instant(ligne["end_ts"])
        if fin >= debut_instant and debut <= fin_instant:
            touchees.append(ligne)
    return touchees


def anomaly_table(
    stream: ReplayStream,
    session: SessionServeur,
    reglages: ReglagesServeur,
    n_train_points: int,
    seed: int,
    evaluation_config: dict[str, Any] | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
) -> pd.DataFrame:
    """Une ligne par anomalie etiquetee (verite terrain) x strategie : detectee, delai.

    `event_metrics` est appele une fois par strategie sur la MEME liste
    d'evenements (le meme flux), comme dans `event_comparison.py` : les delais
    de deux strategies restent directement comparables terme a terme.
    """
    eval_cfg = evaluation_config or load_config("evaluation.yaml")
    tolerance = eval_cfg.get("event", {}).get("tolerance_points")
    tolerance_points = stream.window_size - 1 if tolerance is None else int(tolerance)

    etiquettes_brutes = lire_anomalies(reglages.dossier_etiquettes, session.run_id)

    par_strategie: dict[str, Any] = {}
    for strategie in strategies:
        artefacts = _load_artefacts(stream, strategie, seed)
        if artefacts is None:
            continue
        metrics = event_metrics(
            stream.events,
            artefacts.traces["alerts"],
            artefacts.traces["end_index"],
            tolerance_points,
            stream.sampling_interval_seconds,
        )
        par_strategie[strategie] = metrics

    lignes: list[dict[str, Any]] = []
    for idx, event in enumerate(stream.events):
        brutes = _label_rows_for_event(
            etiquettes_brutes, event.start, event.end, n_train_points, session
        )
        types = sorted({ligne.get("type", "?") for ligne in brutes}) or ["?"]
        ligne: dict[str, Any] = {
            "event_index": idx,
            "type": "+".join(types),
            "n_etiquettes_brutes": len(brutes),
            "start_at": _point_to_timestamp(session, n_train_points, event.start),
            "end_at": _point_to_timestamp(session, n_train_points, event.end),
            "length_points": event.length,
        }
        for strategie, metrics in par_strategie.items():
            delai = metrics.delays_minutes[idx]
            ligne[f"detected_{strategie}"] = delai is not None
            ligne[f"delay_{strategie}_minutes"] = delai
        lignes.append(ligne)

    return pd.DataFrame(lignes)


def build_session_report(
    run_id: str,
    dataset_config: str = "datasets/serveur.yaml",
    features_config: str = "features.yaml",
    evaluation_config: str = "evaluation.yaml",
    seed: int = 42,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, Any]:
    """Assemble les trois sorties du rapport : fenetres, strategies, anomalies.

    Suppose que `mdrift session-replay` a deja ecrit les traces des strategies
    demandees ; ce module ne rejoue rien.
    """
    reglages = ReglagesServeur.depuis_config(load_config(dataset_config))
    features = load_config(features_config)
    eval_cfg = load_config(evaluation_config)
    session, stream = load_session_stream(run_id, dataset_config, features)
    n_train_points = len(session.train)

    return {
        "windows": window_counts(stream),
        "strategies": strategy_summary(stream, session, n_train_points, seed, strategies),
        "anomalies": anomaly_table(
            stream, session, reglages, n_train_points, seed, eval_cfg, strategies
        ),
        "drift_real": {
            "n_derives": len(session.derives),
            "derives": session.derives,
            "scenario": stream.scenario,
        },
        "doublon_pic_cpu": _doublon_pic_cpu_note(reglages, session.run_id),
    }


def _doublon_pic_cpu_note(reglages: ReglagesServeur, run_id: str) -> str:
    """Explicite comment le doublon pic_cpu du 23/11 est compte (demande de Sidy).

    Les deux lignes d'etiquette se chevauchent (`14:00:02Z-14:03:03Z` et
    `14:01:53Z-14:04:54Z`) : `_etiqueter` met a 1 l'union des deux plages, et
    `events_from_label` (plage contigue de points a 1) y voit UN SEUL
    evenement, pas deux. Le tableau `anomalies` le confirme : `event_index`
    correspondant porte `n_etiquettes_brutes == 2` pour un seul `event_index`.
    """
    toutes = lire_anomalies(reglages.dossier_etiquettes, run_id)
    lignes = [ligne for ligne in toutes if ligne.get("type") == "pic_cpu"]
    chevauchantes = []
    for i, a in enumerate(lignes):
        for b in lignes[i + 1 :]:
            if instant(a["start_ts"]) <= instant(b["end_ts"]) and instant(b["start_ts"]) <= instant(
                a["end_ts"]
            ):
                chevauchantes.append((a, b))
    if not chevauchantes:
        return "aucun doublon pic_cpu detecte dans les etiquettes de cette session."
    a, b = chevauchantes[0]
    return (
        f"doublon pic_cpu : {a['start_ts']}->{a['end_ts']} et {b['start_ts']}->{b['end_ts']} "
        "se chevauchent ; etiquetage fusionne en UN evenement (union des deux plages), "
        "pas compte deux fois dans le rappel par evenement."
    )


def report_directory(run_id: str, output_dir: Path = REPORT_DIR) -> Path:
    return output_dir / run_id


def write_session_report(
    report: dict[str, Any], run_id: str, output_dir: Path = REPORT_DIR
) -> Path:
    """Ecrit les trois sorties du rapport sur disque, sous `output_dir/run_id/`.

    `strategies.csv` et `anomalies.csv` au format tabulaire (le reste du
    memoire lit du CSV) ; `report.json` porte ce qui ne tient pas dans un
    tableau (fenetres, derives reelles, note sur le doublon pic_cpu).
    """
    directory = report_directory(run_id, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report["strategies"].to_csv(directory / "strategies.csv", index=False)
    report["anomalies"].to_csv(directory / "anomalies.csv", index=False)
    resume = {
        "run_id": run_id,
        "windows": report["windows"],
        "drift_real": report["drift_real"],
        "doublon_pic_cpu": report["doublon_pic_cpu"],
    }
    (directory / "report.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return directory
