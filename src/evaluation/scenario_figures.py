"""Figures des scenarios de derive (livrable M1).

Deux figures, qui repondent a deux questions du lecteur du memoire :

- `plot_profiles` : a quoi ressemblent les quatre types de derive de Gama et al.
  (2014) ? C'est le profil temporel a(t) seul, sans donnees : la figure
  pedagogique du chapitre 5.
- `plot_injection_grid` : qu'est-ce que l'injection fait reellement a une
  metrique du SMD ? La serie d'origine et la serie derivee superposees, pour une
  machine reelle, les quatre scenarios cote a cote.

Comme pour `figures.py` : PNG 300 dpi dans `docs/figures/`, legendes en
francais, aucun nom d'hote ni adresse (regle 4.3 du cadrage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif : execution en ligne de commande

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datasets.smd.loader import SMDMachine  # noqa: E402
from scenarios.injection import (  # noqa: E402
    DriftScenario,
    alpha_profile,
    drift_onset,
    inject,
    load_scenario,
)
from src.evaluation.figures import DPI, FIGURES_DIR  # noqa: E402

# Intitules lisibles, ceux qui figureront dans les legendes du memoire.
LIBELLES = {
    "s1_soudaine": "S1 - derive soudaine",
    "s2_incrementale": "S2 - derive incrementale",
    "s3_graduelle": "S3 - derive graduelle",
    "s4_recurrente": "S4 - derive recurrente",
}

COULEUR_ORIGINE = "#8c8c8c"
COULEUR_DERIVE = "#1f77b4"
COULEUR_TD = "#d62728"


def _heures(n: int, sampling_interval_seconds: int) -> np.ndarray:
    return np.arange(n) * sampling_interval_seconds / 3600.0


def plot_profiles(
    scenarios: list[DriftScenario],
    output_path: str | Path | None = None,
    n_points: int = 20_000,
    seed: int = 42,
    sampling_interval_seconds: int = 60,
) -> Path:
    """Les quatre profils temporels a(t), sur un flux de longueur fixe.

    Les quatre scenarios partagent la meme transformation : seul a(t) change.
    C'est tout l'interet de la figure, et la raison pour laquelle un ecart de
    performance entre S1 et S2 est attribuable au **type** de derive.
    """
    figure, axes = plt.subplots(
        len(scenarios), 1, figsize=(11, 1.9 * len(scenarios)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, scenario in zip(axes, scenarios, strict=True):
        t_d = drift_onset(n_points, scenario, seed)
        alpha, fin = alpha_profile(
            n_points, t_d, scenario, seed, sampling_interval_seconds, stream_key="figure"
        )
        hours = _heures(n_points, sampling_interval_seconds)

        ax.fill_between(hours, 0.0, alpha, color=COULEUR_DERIVE, alpha=0.30, linewidth=0)
        ax.plot(hours, alpha, linewidth=0.8, color=COULEUR_DERIVE)

        if scenario.profil == "graduel":
            # a(t) ne vaut que 0 ou 1, tire point par point : a cette echelle la
            # trace est un bloc plein. La proportion lissee montre ce que le
            # profil fait reellement, c'est-a-dire monter la probabilite du
            # regime transforme de 0 a 1 sur la duree de transition.
            fenetre = max(1, n_points // 100)
            # `valid` et non `same` : `same` complete les bords par des zeros et
            # ferait retomber la courbe en fin de flux, ce qui serait faux.
            proportion = np.convolve(alpha, np.ones(fenetre) / fenetre, mode="valid")
            centre = hours[fenetre // 2 : fenetre // 2 + len(proportion)]
            ax.plot(
                centre,
                proportion,
                linewidth=1.4,
                color="#08306b",
                label="proportion du regime transforme",
            )
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        ax.axvline(hours[t_d], color=COULEUR_TD, linestyle="--", linewidth=1.0)
        ax.text(
            hours[t_d],
            1.12,
            " t_d",
            color=COULEUR_TD,
            fontsize=8,
            va="center",
        )
        if fin > t_d:
            ax.axvspan(hours[t_d], hours[fin], color=COULEUR_TD, alpha=0.07, linewidth=0)

        ax.set_ylim(-0.08, 1.25)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylabel("a(t)")
        ax.set_title(LIBELLES.get(scenario.name, scenario.name), fontsize=10, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[-1].set_xlabel("Temps depuis le debut du flux (heures)")
    axes[-1].set_xlim(0, _heures(n_points, sampling_interval_seconds)[-1])

    destination = Path(output_path) if output_path else FIGURES_DIR / "profils_derive.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def _metrique_la_plus_visible(
    origine: np.ndarray, derivee: np.ndarray, metriques: tuple[int, ...], depuis: int
) -> int:
    """Parmi les metriques touchees, celle ou l'ecart obtenu est le plus grand.

    Le critere porte sur l'ecart **apres ecretage**, et non sur sigma_train :
    une metrique deja proche de 1 recoit un grand decalage theorique mais se
    retrouve plafonnee, et la figure ne montre alors qu'une ligne collee au
    plafond. Le critere est explicite pour que la figure ne soit pas accusee de
    montrer un cas favorable choisi a la main.
    """
    colonnes = list(metriques)
    ecarts = np.abs(derivee[depuis:, colonnes] - origine[depuis:, colonnes]).mean(axis=0)
    return int(colonnes[int(np.argmax(ecarts))])


def plot_injection_grid(
    machine: SMDMachine,
    scenarios: list[DriftScenario],
    seed: int = 42,
    metric: int | None = None,
    output_path: str | Path | None = None,
    lissage: int = 60,
) -> tuple[Path, list[dict[str, Any]]]:
    """Serie d'origine et serie derivee superposees, un panneau par scenario.

    Les series sont lissees sur `lissage` points : a 28 000 points, la trace
    brute est un bloc noir et la derive y est invisible. Le lissage ne sert qu'a
    l'affichage, jamais au calcul.
    """
    figure, axes = plt.subplots(
        len(scenarios), 1, figsize=(11, 2.1 * len(scenarios)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    noyau = np.ones(lissage) / lissage
    interval = machine.sampling_interval_seconds
    descriptions: list[dict[str, Any]] = []

    flux_par_scenario = [
        inject(
            train=machine.train,
            test=machine.test,
            label=machine.test_label,
            scenario=scenario,
            seed=seed,
            sampling_interval_seconds=interval,
            stream_key=machine.name,
        )
        for scenario in scenarios
    ]
    reference = flux_par_scenario[0]
    colonne = (
        metric
        if metric is not None
        else _metrique_la_plus_visible(
            machine.test, reference.series, reference.metriques, reference.drift_end_index
        )
    )

    for ax, scenario, flux in zip(axes, scenarios, flux_par_scenario, strict=True):
        origine = np.convolve(machine.test[:, colonne], noyau, mode="valid")
        derivee = np.convolve(flux.series[:, colonne], noyau, mode="valid")
        hours = (np.arange(len(origine)) + lissage // 2) * interval / 3600.0

        ax.plot(hours, origine, linewidth=0.9, color=COULEUR_ORIGINE, label="Serie d'origine")
        ax.plot(hours, derivee, linewidth=0.9, color=COULEUR_DERIVE, label="Serie derivee")
        ax.axvline(flux.drift_index * interval / 3600.0, color=COULEUR_TD, linestyle="--", lw=1.0)
        ax.set_ylabel(f"Metrique {colonne}\n(moyenne sur {lissage} min)")
        ax.set_title(LIBELLES.get(scenario.name, scenario.name), fontsize=10, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)

        descriptions.append(
            {
                "scenario": scenario.name,
                "metrique": colonne,
                "t_d_heures": round(flux.drift_index * interval / 3600.0, 1),
                "n_metriques_touchees": len(flux.metriques),
                "part_ecretee_percent": round(flux.part_ecretee * 100, 2),
            }
        )

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("Temps depuis le debut du flux de test (heures)")
    axes[-1].set_xlim(0, len(machine.test) * interval / 3600.0)

    destination = Path(output_path) if output_path else FIGURES_DIR / "injection_scenarios.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination, descriptions


def load_scenarios(names: list[str]) -> list[DriftScenario]:
    """Charge les scenarios depuis `configs/scenarios/<nom>.yaml`."""
    return [load_scenario(f"scenarios/{name}.yaml") for name in names]
