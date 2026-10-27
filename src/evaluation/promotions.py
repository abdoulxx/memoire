"""Promotions degradantes, calculees **hors ligne** depuis les traces (§7, D16).

Definition du cadrage : une promotion est degradante si, sur les N fenetres qui
la suivent, le modele promu obtient un F1 inferieur de plus de delta a celui du
modele qu'il a remplace.

Pourquoi hors ligne, et pourquoi cela compte. Ce calcul utilise les etiquettes.
Si la strategie y avait acces, elle choisirait ses promotions en regardant la
verite terrain, ce qu'aucun systeme en production ne peut faire : les resultats
du memoire ne vaudraient plus rien. Le harnais se contente donc d'enregistrer
les scores du modele remplace sur les N fenetres suivantes (scoring fantome, en
parallele, un bloc par promotion) et le calcul est fait ici, apres coup, a
partir de `traces.npz`. Son cout est exclu du cout de la strategie.

Cette metrique sert a tester H3 : la validation de P reduit-elle les promotions
degradantes par rapport a A1, qui promeut sans condition ?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import window_metrics


@dataclass(frozen=True)
class PromotionOutcome:
    """Comparaison d'une promotion : modele promu contre modele remplace."""

    position: int
    start: int
    n_windows: int
    f1_new: float
    f1_old: float
    delta_f1: float
    degrading: bool
    # Vrai si une autre promotion a eu lieu dans la fenetre de comparaison :
    # l'ecart ne compare alors plus deux modeles mais une sequence de modeles.
    overlapping: bool


@dataclass(frozen=True)
class PromotionReport:
    """Agregat sur toutes les promotions d'une execution."""

    n_promotions: int
    n_degrading: int
    n_overlapping: int
    delta_f1_mean: float | None
    delta_f1_median: float | None
    delta_f1_min: float | None
    outcomes: tuple[PromotionOutcome, ...] = ()

    def as_row(self) -> dict[str, Any]:
        return {
            "n_promotions_evaluees": self.n_promotions,
            "n_promotions_degradantes": self.n_degrading,
            "n_promotions_chevauchantes": self.n_overlapping,
            "delta_f1_mean": self.delta_f1_mean,
            "delta_f1_median": self.delta_f1_median,
            "delta_f1_min": self.delta_f1_min,
        }


def evaluate_promotions(
    y_true: np.ndarray,
    alerts: np.ndarray,
    model_changes: np.ndarray,
    shadow_positions: np.ndarray,
    shadow_starts: np.ndarray,
    shadow_lengths: np.ndarray,
    shadow_thresholds: np.ndarray,
    shadow_scores: np.ndarray,
    delta_f1: float,
    zero_division: float = 0.0,
) -> PromotionReport:
    """Compare, pour chaque promotion, le modele promu et le modele remplace.

    `alerts` porte ce que le modele **promu** a reellement produit : c'est le
    cote « nouveau ». Les scores fantomes portent ce que le modele remplace
    aurait produit sur les memes fenetres : c'est le cote « ancien ». La
    comparaison est donc faite sur les memes fenetres et les memes etiquettes,
    ce qui est la seule facon d'isoler l'effet du changement de modele.

    La distribution complete des ecarts est renvoyee, et pas seulement le
    comptage : le §7 demande de la rapporter, parce qu'un comptage depend
    entierement du delta choisi.
    """
    offsets = np.concatenate(([0], np.cumsum(shadow_lengths))).astype(np.int64)
    changes = np.asarray(model_changes, dtype=np.int64)

    outcomes: list[PromotionOutcome] = []
    for i in range(len(shadow_positions)):
        start = int(shadow_starts[i])
        length = int(shadow_lengths[i])
        stop = start + length
        segment = shadow_scores[offsets[i] : offsets[i + 1]]
        old_alerts = (segment > float(shadow_thresholds[i])).astype(np.int64)

        new = window_metrics(y_true[start:stop], alerts[start:stop], zero_division)
        old = window_metrics(y_true[start:stop], old_alerts, zero_division)
        delta = new.f1 - old.f1

        # Une promotion ulterieure dans [start, stop) signifie que `alerts` y
        # melange deux modeles promus : l'ecart reste mesure, mais il est
        # signale pour ne pas etre interprete comme un effet d'une seule
        # promotion.
        later = changes[(changes > start) & (changes < stop)]
        outcomes.append(
            PromotionOutcome(
                position=int(shadow_positions[i]),
                start=start,
                n_windows=length,
                f1_new=new.f1,
                f1_old=old.f1,
                delta_f1=float(delta),
                degrading=bool(delta < -float(delta_f1)),
                overlapping=bool(later.size > 0),
            )
        )

    deltas = np.array([o.delta_f1 for o in outcomes], dtype=np.float64)
    return PromotionReport(
        n_promotions=len(outcomes),
        n_degrading=int(sum(o.degrading for o in outcomes)),
        n_overlapping=int(sum(o.overlapping for o in outcomes)),
        delta_f1_mean=float(deltas.mean()) if deltas.size else None,
        delta_f1_median=float(np.median(deltas)) if deltas.size else None,
        delta_f1_min=float(deltas.min()) if deltas.size else None,
        outcomes=tuple(outcomes),
    )


def evaluate_promotions_from_traces(
    traces_path: Path | str,
    delta_f1: float,
    zero_division: float = 0.0,
) -> PromotionReport:
    """Charge `traces.npz` et calcule le rapport des promotions."""
    with np.load(Path(traces_path)) as data:
        if "shadow_positions" not in data:
            # Trace produite avant D16 : aucune promotion n'y est comparable.
            return PromotionReport(0, 0, 0, None, None, None)
        return evaluate_promotions(
            y_true=data["y_true"],
            alerts=data["alerts"],
            model_changes=data["model_changes"],
            shadow_positions=data["shadow_positions"],
            shadow_starts=data["shadow_starts"],
            shadow_lengths=data["shadow_lengths"],
            shadow_thresholds=data["shadow_thresholds"],
            shadow_scores=data["shadow_scores"],
            delta_f1=delta_f1,
            zero_division=zero_division,
        )


def collect_promotions(
    runs_dir: Path | str,
    delta_f1: float,
) -> list[dict[str, Any]]:
    """Recalcule toutes les promotions de `results/runs/`, une ligne par promotion.

    Le §7 demande de rapporter la **distribution des ecarts** et pas seulement
    leur comptage, precisement parce qu'un comptage depend entierement du delta
    choisi. Comme tout le calcul repart de `traces.npz`, changer delta ne coute
    pas un rejeu : c'est ce qui permet de montrer qu'une conclusion ne tient pas
    au seul reglage de ce seuil.

    L'arborescence lue est `<source>/<machine>/<scenario>/<strategie>/seed<N>/`,
    ce qui suffit a reconstituer le `run_id` de `summary.csv` sans le relire.
    """
    racine = Path(runs_dir)
    lignes: list[dict[str, Any]] = []
    for trace in sorted(racine.glob("*/*/*/*/seed*/traces.npz")):
        graine = trace.parent.name.removeprefix("seed")
        strategie = trace.parent.parent.name
        scenario = trace.parent.parent.parent.name
        machine = trace.parent.parent.parent.parent.name
        source = trace.parent.parent.parent.parent.parent.name
        rapport = evaluate_promotions_from_traces(trace, delta_f1=delta_f1)
        for promotion in rapport.outcomes:
            lignes.append(
                {
                    "run_id": f"{source}_{machine}_{scenario}_{strategie}_seed{graine}",
                    "source": source,
                    "machine": machine,
                    "scenario": scenario,
                    "strategy": strategie,
                    "seed": int(graine),
                    "delta_f1_seuil": delta_f1,
                    "position": promotion.position,
                    "start": promotion.start,
                    "n_windows": promotion.n_windows,
                    "f1_new": promotion.f1_new,
                    "f1_old": promotion.f1_old,
                    "delta_f1": promotion.delta_f1,
                    "degrading": promotion.degrading,
                    "overlapping": promotion.overlapping,
                }
            )
    return lignes
