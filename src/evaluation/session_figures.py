"""Figure du flux d'une session rejouee : score, alertes, derive, anomalies.

`figures.plot_scores` trace un rejeu isole. `result_figures.plot_f1_glissant`
superpose les quatre strategies sur un meme flux SMD, mais en F1 glissant, pas
en score brut. Cette fonction fait l'equivalent pour une session serveur, en
score : B0, B1 et P sur le MEME flux (la session rejouee une seule fois,
§6.1.1), avec les memes couleurs que `result_figures` pour que les figures de
la session et de la campagne SMD se lisent ensemble.

Rien n'est recalcule : les scores et les alertes viennent de
`results/runs/serveur/<run_id>/.../traces.npz`, ecrites par
`mdrift session-replay`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.figures import DPI, FIGURES_DIR, _spans  # noqa: E402
from src.evaluation.replay import run_directory  # noqa: E402
from src.evaluation.result_figures import COULEURS, LIBELLES  # noqa: E402
from src.evaluation.session_campaign import load_session_stream  # noqa: E402

STRATEGIES_FIGURE = ("static", "periodic", "proposed")


def plot_session_scores(
    run_id: str,
    strategies: tuple[str, ...] = STRATEGIES_FIGURE,
    seed: int = 42,
    dataset_config: str = "datasets/serveur.yaml",
    features_config: str = "features.yaml",
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Score d'anomalie et alertes de B0, B1 et P sur une session, derive et anomalies marquees.

    Suppose que `mdrift session-replay <run_id>` a deja ecrit les traces des
    strategies demandees ; cette fonction ne rejoue rien.
    """
    _session, stream = load_session_stream(run_id, dataset_config, features_config)
    intervalle = stream.sampling_interval_seconds

    traces_par_strategie: dict[str, dict[str, np.ndarray]] = {}
    manquantes: list[str] = []
    for strategie in strategies:
        chemin = run_directory(stream, strategie, seed) / "traces.npz"
        if not chemin.is_file():
            manquantes.append(strategie)
            continue
        with np.load(chemin) as donnees:
            traces_par_strategie[strategie] = {nom: donnees[nom] for nom in donnees.files}
    if manquantes:
        raise FileNotFoundError(
            f"trace(s) absente(s) pour {', '.join(manquantes)} "
            f"(lancer `mdrift session-replay {run_id}` pour ces strategies)."
        )

    figure, axes = plt.subplots(figsize=(11, 4.6), constrained_layout=True)

    hours = np.arange(len(stream.y)) * intervalle / 3600.0
    for first, last in _spans(stream.y.astype(bool)):
        axes.axvspan(hours[first], hours[last], color="#d62728", alpha=0.16, linewidth=0)
    axes.axvspan(np.nan, np.nan, color="#d62728", alpha=0.16, label="Anomalie etiquetee")

    for strategie in strategies:
        traces = traces_par_strategie[strategie]
        couleur = COULEURS[strategie]
        axes.plot(
            hours,
            traces["scores"],
            linewidth=0.8,
            color=couleur,
            alpha=0.85,
            label=f"{LIBELLES[strategie]} — score",
        )
        alerts = traces["alerts"].astype(bool)
        if alerts.any():
            axes.scatter(
                hours[alerts],
                traces["scores"][alerts],
                s=4,
                color=couleur,
                zorder=3,
                label=f"{LIBELLES[strategie]} — alertes ({int(alerts.sum())})",
            )
        changements = traces.get("model_changes")
        if changements is not None and len(changements):
            axes.scatter(
                changements * intervalle / 3600.0,
                np.full(len(changements), -0.04),
                marker="^",
                s=20,
                color=couleur,
                clip_on=False,
                label="_nolegend_",
            )

    derive_fenetre = stream.drift_window()
    if derive_fenetre is not None:
        heure_derive = derive_fenetre * intervalle / 3600.0
        axes.axvline(
            heure_derive, color="black", linestyle="--", linewidth=1.0, label="debut de la derive"
        )

    axes.set_xlabel("Temps depuis le debut du flux evalue (heures)")
    axes.set_ylabel("Score d'anomalie")
    axes.set_xlim(hours[0], hours[-1])
    axes.set_ylim(bottom=-0.08)
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.legend(loc="upper left", fontsize=7.5, framealpha=0.9, ncols=2)
    if title:
        axes.set_title(title, fontsize=11)
    else:
        axes.set_title(
            f"Session {run_id} — B0, B1, P (triangles : promotions de modele)", fontsize=10
        )

    destination = Path(output_path) if output_path else FIGURES_DIR / f"scores_session_{run_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination
