"""Calibrage des parametres de surveillance de derive (etapes 1 et 2 de la strategie P).

A ne pas confondre avec `src/evaluation/calibration.py`, qui calibre les etapes
3 et 4 (donnees du reentrainement et bornes du garde-fou) et ecrit
`results/calibration_dev.csv`. Ce module-ci ne regle que le declenchement, et
ecrit `results/calibration_m2_dev.csv`.

Quatre parametres fixent le comportement de l'etape 1 et de l'etape 2 de la
strategie P : le quantile et la marge du modele nul (seuil par feature), la part
declenchante `drifting_share`, et la persistance K. Ils ne peuvent pas etre
devines : D18 a montre qu'un seuil absolu ne separe pas les machines stables des
machines derivantes, et D19 qu'une persistance de K = 30 confirme une derive sur
13 des 15 machines stables, c'est-a-dire qu'elle ne confirme rien.

**Regle D17, non negociable** : le calibrage ne lit que les 16 machines de
`configs/datasets/smd_dev.yaml`. Les 12 machines d'evaluation ne sont jamais
ouvertes ici, et les parametres sont figes avant la premiere execution sur
elles.

Une precision sur la specificite. La metrique de fausse confirmation du §7 est
definie sur les machines **stables sans injection**. Or 12 des 15 machines
stables sont dans le perimetre d'evaluation : le calibrage ne dispose que de
trois d'entre elles (machine-1-3, machine-2-7, machine-3-10). La mesure sur les
15 machines stables est un **resultat** de M3, jamais une cible de reglage.

Pourquoi un balayage en memoire et non une campagne de rejeux. Un point de
grille ne change ni les features, ni la reference, ni la distribution nulle : il
ne change que deux comparaisons. Le flux est donc parcouru **une fois** par
(machine, scenario, graine), en conservant les 152 statistiques de chaque
fenetre, puis les centaines de points de grille sont evalues sur cette matrice.
Rejouer le flux pour chaque point couterait des jours au lieu de minutes.

Ce que le balayage mesure, et ce qu'il ne mesure pas : la surveillance seule,
avec une reference **figee** sur l'entrainement initial. Aucune promotion, donc
aucun recalage, donc au plus une confirmation par flux. C'est exactement ce
qu'il faut pour regler le declenchement, et cela ne dit rien du F1 ni du cout,
qui dependent du rejeu complet.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.smd.loader import SMDMachine, load_machines
from src.config import load_config
from src.drift.aggregate import FeatureDriftTracker
from src.drift.null_model import null_distribution
from src.drift.reference import build_reference
from src.evaluation.replay import RESULTS_DIR, ReplayStream
from src.evaluation.streams import build_smd_stream

DRIFT_CALIBRATION_PATH = RESULTS_DIR / "calibration_m2_dev.csv"

# Grille par defaut. Les bornes viennent des mesures de M0+ : un quantile sous
# 0,99 laisse passer trop de bruit d'entrainement, une part declenchante au-dela
# de 0,30 n'est jamais atteinte, et K en dessous de 60 fenetres (une heure) ne
# distingue pas une derive d'une rafale d'anomalies.
DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "quantiles": (0.99, 0.995, 0.999, 1.0),
    "margins": (1.0, 1.1, 1.25),
    "shares": (0.05, 0.10, 0.15, 0.20, 0.30),
}
DEFAULT_CONFIRMATIONS: tuple[int, ...] = (60, 120, 240, 480)


@dataclass
class MonitorTrace:
    """Un parcours du flux, conserve sous une forme qui permet de balayer la grille.

    `values` a la forme (n_fenetres, F) et contient la statistique par feature a
    chaque fenetre, `NaN` tant que la fenetre recente n'est pas pleine. C'est la
    seule chose couteuse a produire, et elle ne depend d'aucun parametre de la
    grille.
    """

    machine: str
    scenario: str
    seed: int
    values: np.ndarray
    distribution: np.ndarray
    usable: np.ndarray
    ready_from: int
    drift_window: int | None
    n_windows: int
    trace_seconds: float
    reference_seconds: float

    @property
    def n_usable(self) -> int:
        return int(self.usable.sum())


def build_monitor_trace(
    stream: ReplayStream,
    drift_config: dict[str, Any],
) -> MonitorTrace:
    """Parcourt le flux une fois avec une reference figee, et conserve les statistiques.

    La reference est celle de l'entrainement initial et n'est jamais recalee :
    le balayage regle le **declenchement**, pas la boucle complete de la
    strategie. Le recalage apres promotion est une propriete de P, mesuree par
    le rejeu (`mdrift campaign`).
    """
    statistic = str(drift_config.get("statistic", "ks"))
    n_bins = int(drift_config.get("n_bins", 50))
    recent_windows = int(drift_config.get("recent_windows", 500))
    constant_tolerance = float(drift_config.get("constant_tolerance", 0.0))
    psi_epsilon = float(drift_config.get("psi_epsilon", 1e-6))
    step = drift_config.get("feature_threshold", {}).get("step")

    started = time.perf_counter()
    reference = build_reference(
        stream.train_X, n_bins=n_bins, constant_tolerance=constant_tolerance
    )
    distribution = null_distribution(
        stream.train_X,
        reference,
        recent_windows=recent_windows,
        statistic=statistic,
        step=step,
        psi_epsilon=psi_epsilon,
    )
    reference_seconds = time.perf_counter() - started

    # Le seuil passe au tracker est sans importance : seules les statistiques
    # brutes sont conservees, et la part en derive est recalculee par le
    # balayage pour chaque seuil de la grille.
    tracker = FeatureDriftTracker(
        reference=reference,
        recent_windows=recent_windows,
        statistic=statistic,
        feature_threshold=np.inf,
        psi_epsilon=psi_epsilon,
    )

    n_windows = len(stream)
    values = np.full((n_windows, reference.n_features), np.nan, dtype=np.float64)
    started = time.perf_counter()
    for position in range(n_windows):
        signal = tracker.update(stream.X[position])
        if signal.ready:
            values[position] = tracker.last_values
    trace_seconds = time.perf_counter() - started

    return MonitorTrace(
        machine=stream.machine,
        scenario=stream.scenario,
        seed=0,
        values=values,
        distribution=distribution,
        usable=reference.usable,
        ready_from=min(recent_windows - 1, n_windows),
        drift_window=stream.drift_window(),
        n_windows=n_windows,
        trace_seconds=trace_seconds,
        reference_seconds=reference_seconds,
    )


def first_confirmation(signal: np.ndarray, k: int) -> int | None:
    """Premiere fenetre ou le signal atteint K fenetres consecutives, ou None.

    Reproduit exactement `Confirmation` : la confirmation est acquise a la
    K-ieme fenetre consecutive, et une seule interruption remet le compteur a
    zero. Le calcul est vectorise parce qu'il est repete des milliers de fois.
    """
    if k < 1:
        raise ValueError("K doit valoir au moins 1.")
    if not signal.any():
        return None

    hits = np.where(streak_lengths(signal) >= k)[0]
    return int(hits[0]) if hits.size else None


def streak_lengths(signal: np.ndarray) -> np.ndarray:
    """Longueur de la serie de True se terminant a chaque position.

    Somme cumulee remise a zero sur chaque False : `cumulative` est croissante,
    donc le maximum courant de ses valeurs aux positions False vaut sa valeur au
    dernier False rencontre. La difference est le nombre de True depuis.
    """
    cumulative = np.cumsum(signal.astype(np.int64))
    at_breaks = np.where(signal, 0, cumulative)
    return cumulative - np.maximum.accumulate(at_breaks)


def longest_streak(signal: np.ndarray) -> int:
    """Plus longue serie de signal consecutif. C'est le K minimal qui la supprime."""
    if not signal.any():
        return 0
    padded = np.concatenate(([False], signal, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    return int((stops - starts).max())


@dataclass
class GridPoint:
    """Un point de grille : seuil par feature, part declenchante, persistance."""

    quantile: float
    margin: float
    share: float
    k: int


def sweep_trace(
    trace: MonitorTrace,
    quantiles: Sequence[float] = DEFAULT_GRID["quantiles"],
    margins: Sequence[float] = DEFAULT_GRID["margins"],
    shares: Sequence[float] = DEFAULT_GRID["shares"],
    confirmations: Sequence[int] = DEFAULT_CONFIRMATIONS,
) -> list[dict[str, Any]]:
    """Evalue toute la grille sur une trace deja calculee.

    Une ligne par point de grille. Les colonnes separent ce qui releve de la
    **specificite** (confirmation avant l'instant de derive, ou sur le scenario
    `none`) de ce qui releve de la **sensibilite** (confirmation apres l'instant
    de derive, et son delai). Les melanger donnerait un chiffre flatteur et sans
    contenu : un moniteur qui confirme toujours a un delai de detection parfait.
    """
    usable = trace.usable
    values = trace.values[:, usable]
    ready = ~np.isnan(values[:, 0]) if values.shape[1] else np.zeros(len(values), bool)
    drift_window = trace.drift_window

    rows: list[dict[str, Any]] = []
    for quantile in quantiles:
        base = np.quantile(trace.distribution[:, usable], quantile, axis=0)
        for margin in margins:
            thresholds = base * float(margin)
            drifting = np.zeros(values.shape, dtype=bool)
            np.greater(values, thresholds, out=drifting, where=ready[:, None])
            share = np.where(ready, drifting.mean(axis=1), 0.0)

            for minimum in shares:
                signal = ready & (share >= minimum)
                avant = signal.copy()
                apres = signal.copy()
                if drift_window is not None:
                    avant[drift_window:] = False
                    apres[:drift_window] = False

                serie_avant = longest_streak(avant)
                serie_apres = longest_streak(apres) if drift_window is not None else 0

                for k in confirmations:
                    position = first_confirmation(signal, k)
                    confirme = position is not None
                    if drift_window is None:
                        apres_derive = False
                        delai = None
                    else:
                        apres_derive = confirme and position >= drift_window
                        delai = (position - drift_window) if apres_derive else None

                    rows.append(
                        {
                            "machine": trace.machine,
                            "scenario": trace.scenario,
                            "seed": trace.seed,
                            "quantile": quantile,
                            "margin": margin,
                            "drifting_share": minimum,
                            "confirmation_windows": k,
                            "n_windows": trace.n_windows,
                            "n_features_usable": trace.n_usable,
                            "drift_window": drift_window,
                            "confirme": confirme,
                            "fenetre_confirmation": position,
                            # Specificite : une confirmation avant t_d (ou sur
                            # `none`) est fausse, quelle que soit la suite.
                            "confirmation_avant_derive": bool(confirme and not apres_derive),
                            # Sensibilite : confirmation apres t_d, et son delai.
                            "detection_apres_derive": bool(apres_derive),
                            "delai_detection_fenetres": delai,
                            # K minimal qui aurait supprime la fausse
                            # confirmation, et K maximal qui garde la vraie.
                            "plus_longue_serie_avant": serie_avant,
                            "plus_longue_serie_apres": serie_apres,
                            "part_mediane": float(np.median(share[ready])) if ready.any() else 0.0,
                            "part_max": float(share[ready].max()) if ready.any() else 0.0,
                        }
                    )
    return rows


@dataclass
class CalibrationRun:
    """Trace de l'execution, pour le journal du memoire."""

    n_traces: int = 0
    n_rows: int = 0
    seconds: float = 0.0
    per_trace: list[dict[str, Any]] = field(default_factory=list)


def calibrate_drift(
    dataset_config: str = "datasets/smd_dev.yaml",
    features_config: str = "features.yaml",
    drift_config: str = "drift.yaml",
    scenarios: Iterable[str] = ("none", "s1_soudaine", "s2_incrementale", "s3_graduelle"),
    seeds: Iterable[int] = (42,),
    machines: Iterable[str] | None = None,
    quantiles: Sequence[float] = DEFAULT_GRID["quantiles"],
    margins: Sequence[float] = DEFAULT_GRID["margins"],
    shares: Sequence[float] = DEFAULT_GRID["shares"],
    confirmations: Sequence[int] = DEFAULT_CONFIRMATIONS,
    recent_windows: Sequence[int] | None = None,
    path: Path = DRIFT_CALIBRATION_PATH,
    on_trace: Any = None,
) -> tuple[pd.DataFrame, CalibrationRun]:
    """Balaie la grille sur le perimetre de mise au point et ecrit le CSV.

    Le scenario `none` n'est parcouru qu'une fois par machine : sans injection,
    la graine ne change rien au flux, et le repeter n'ajouterait que du temps de
    calcul et des lignes identiques.

    `recent_windows` est le seul parametre de la grille qui ne soit **pas**
    factorisable : changer R change la distribution nulle et l'histogramme
    glissant, donc la trace elle-meme. Chaque valeur demande un parcours
    complet du perimetre, d'ou sa separation des trois autres axes.
    """
    features = load_config(features_config)
    drift = load_config(drift_config)
    tailles = (
        [int(drift["recent_windows"])]
        if recent_windows is None
        else [int(valeur) for valeur in recent_windows]
    )

    available: dict[str, SMDMachine] = {m.name: m for m in load_machines(dataset_config)}
    wanted = list(available) if machines is None else list(machines)
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ValueError(
            f"Machines absentes de {dataset_config} : {', '.join(unknown)}. "
            "Le calibrage ne lit que le perimetre de mise au point (D17)."
        )

    scenario_names = list(scenarios)
    seed_values = list(seeds)
    run = CalibrationRun()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for name in wanted:
        machine = available[name]
        for scenario in scenario_names:
            # Sans injection, la graine n'a aucun effet : un seul parcours.
            graines = [seed_values[0]] if scenario == "none" else seed_values
            for seed in graines:
                stream = build_smd_stream(machine, features, scenario, seed)
                for taille in tailles:
                    trace = build_monitor_trace(stream, {**drift, "recent_windows": taille})
                    trace.seed = seed
                    produced = sweep_trace(trace, quantiles, margins, shares, confirmations)
                    for row in produced:
                        row["recent_windows"] = taille
                    rows.extend(produced)

                    run.n_traces += 1
                    run.n_rows += len(produced)
                    info = {
                        "machine": name,
                        "scenario": scenario,
                        "seed": seed,
                        "recent_windows": taille,
                        "n_windows": trace.n_windows,
                        "n_features_usable": trace.n_usable,
                        "n_blocks": int(trace.distribution.shape[0]),
                        "drift_window": trace.drift_window,
                        "reference_seconds": round(trace.reference_seconds, 2),
                        "trace_seconds": round(trace.trace_seconds, 2),
                        "n_rows": len(produced),
                    }
                    run.per_trace.append(info)
                    if on_trace is not None:
                        on_trace(info)
                    # La matrice des statistiques est le gros de l'empreinte
                    # memoire : on la libere des que le balayage est fait.
                    del trace
                del stream

    run.seconds = time.perf_counter() - started
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table, run


def summarise(table: pd.DataFrame, stable: Sequence[str]) -> pd.DataFrame:
    """Agrege la grille en un tableau de decision : specificite contre sensibilite.

    `stable` est la liste des machines **stables** du perimetre de mise au point.
    La specificite n'est comptee que sur elles : sur une machine naturellement
    derivante, une confirmation sans injection n'est pas une erreur du moniteur
    mais une derive reelle (OBS03), et la compter comme fausse ferait regler K
    sur un faux probleme. Les confirmations des machines instables sont donc
    rapportees a part, dans une colonne qui n'entre pas dans le tri.

    Deux colonnes de specificite, et la difference compte. `fausses_confirmations_stables`
    reprend la definition exacte du §7 et de D19 : confirmations sur le scenario
    `none`, soit **une** trace par machine stable, donc 3 au total ici.
    `faux_signaux_stables` elargit a toute confirmation **anterieure a t_d** sur
    une machine stable, quel que soit le scenario : avant t_d le flux injecte est
    identique au flux d'origine, une confirmation y est donc tout aussi fausse.
    Cela porte la base a 3 x 5 = **15** traces pour le meme cout de calcul, et
    c'est ce critere elargi qui sert au tri. Le premier reste affiche parce que
    c'est lui que le memoire compare au point de reference de D19.
    """
    stable = list(stable)
    cles = ["quantile", "margin", "drifting_share", "confirmation_windows"]
    if "recent_windows" in table.columns:
        cles = ["recent_windows", *cles]

    machines_stables = table["machine"].isin(stable)
    # Sur `none` il n'y a pas de t_d : toute confirmation est prematuree.
    anterieure = table["confirmation_avant_derive"] | (
        table["confirme"] & table["drift_window"].isna()
    )
    elargi = (
        table[machines_stables]
        .assign(faux=anterieure[machines_stables])
        .groupby(cles)["faux"]
        .agg(faux_signaux_stables="sum", n_traces_stables="size")
    )

    sans_injection = table[table["scenario"] == "none"]
    specificite = (
        sans_injection[sans_injection["machine"].isin(stable)]
        .groupby(cles)["confirme"]
        .agg(fausses_confirmations_stables="sum", n_stables="size")
    )
    naturelle = (
        sans_injection[~sans_injection["machine"].isin(stable)]
        .groupby(cles)["confirme"]
        .agg(confirmations_derive_naturelle="sum", n_instables="size")
    )

    injecte = table[table["scenario"] != "none"]
    sensibilite = injecte.groupby(cles).agg(
        detections=("detection_apres_derive", "sum"),
        n_injections=("detection_apres_derive", "size"),
        confirmations_precoces=("confirmation_avant_derive", "sum"),
        delai_median=("delai_detection_fenetres", "median"),
        delai_max=("delai_detection_fenetres", "max"),
    )

    resume = (
        elargi.join(specificite, how="outer")
        .join(naturelle, how="outer")
        .join(sensibilite, how="outer")
    )
    resume["taux_detection"] = resume["detections"] / resume["n_injections"]
    return resume.reset_index().sort_values(
        ["faux_signaux_stables", "taux_detection", "delai_median"],
        ascending=[True, False, True],
    )
