"""PR-AUC sur la marge score - seuil, calculee hors ligne depuis `traces.npz` (§7, D48).

Complement demande apres retours du directeur sur les resultats M3 deja rejoues
(aucun rejeu relance). La marge (`scores - thresholds`) est comparable d'une
fenetre a l'autre et d'un reentrainement a l'autre, contrairement au score brut
dont l'echelle suit le seuil courant : c'est elle qui sert de variable continue
au calcul de l'AP, le rappel/precision/F1 par fenetre (`window_metrics`) ne
portant que sur la decision binaire deja seuillee.

`average_precision` est reimplementee en numpy pur (definition de
Davis & Goadrich, 2006 : pas d'interpolation lineaire entre deux points de la
courbe precision-rappel, optimiste par construction) plutot qu'un appel a
`sklearn.metrics.average_precision_score`, pour que ce module reste utilisable
meme dans un environnement ou `sklearn`/`scipy.sparse` ne se charge pas (D48).
Les deux definitions coincident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (aire sous la courbe precision-rappel, sans interpolation)."""
    true = y_true.astype(bool)
    n_pos = int(true.sum())
    if n_pos == 0:
        return float("nan")
    ordre = np.argsort(-score, kind="mergesort")
    trie = true[ordre]
    tp_cumule = np.cumsum(trie)
    fp_cumule = np.cumsum(~trie)
    precision = tp_cumule / (tp_cumule + fp_cumule)
    rappel = tp_cumule / n_pos
    rappel_precedent = np.concatenate(([0.0], rappel[:-1]))
    delta_rappel = rappel - rappel_precedent
    return float(np.sum(delta_rappel * precision))


def pr_auc_from_traces(traces_path: Path | str) -> dict[str, Any]:
    """Charge `traces.npz` et calcule l'AP sur la marge score - seuil."""
    with np.load(Path(traces_path)) as data:
        y_true = data["y_true"]
        marge = data["scores"] - data["thresholds"]
        return {
            "n_windows": int(len(y_true)),
            "n_positive_windows": int(y_true.sum()),
            "pr_auc_margin": average_precision(y_true, marge),
        }


def collect_pr_auc(runs_dir: Path | str) -> list[dict[str, Any]]:
    """Calcule l'AP sur la marge pour chaque rejeu present dans `results/runs/`.

    Meme arborescence et meme convention de `run_id` que
    `src.evaluation.promotions.collect_promotions`, pour rester directement
    comparable a `promotions.csv` / `summary.csv`.
    """
    racine = Path(runs_dir)
    lignes: list[dict[str, Any]] = []
    for trace in sorted(racine.glob("*/*/*/*/seed*/traces.npz")):
        graine = trace.parent.name.removeprefix("seed")
        strategie = trace.parent.parent.name
        scenario = trace.parent.parent.parent.name
        machine = trace.parent.parent.parent.parent.name
        source = trace.parent.parent.parent.parent.parent.name
        stats = pr_auc_from_traces(trace)
        lignes.append(
            {
                "run_id": f"{source}_{machine}_{scenario}_{strategie}_seed{graine}",
                "source": source,
                "machine": machine,
                "scenario": scenario,
                "strategy": strategie,
                "seed": int(graine),
                **stats,
            }
        )
    return lignes
