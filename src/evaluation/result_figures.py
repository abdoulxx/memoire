"""Figures du chapitre 6 : comparaison des strategies sur la campagne M3.

`figures.py` trace un rejeu isole (score, seuil, alertes). Ce module-ci trace les
**comparaisons** : les quatre strategies sur le meme flux, et les agregats de
`results/summary.csv`. La separation est volontaire — les figures de diagnostic
d'un rejeu et les figures de resultat n'ont ni les memes entrees ni la meme duree
de vie.

Trois regles tenues partout ici :

- **Moyenne par machine avant toute moyenne generale** (OBS06) : les 5 graines
  d'une machine sont 5 tirages de *t_d* sur le meme flux, pas 5 observations
  independantes. Une barre d'erreur calculee sur les 960 lignes brutes serait
  trop serree.
- **Memes couleurs pour une strategie dans toutes les figures**, pour que le
  lecteur n'ait pas a relire la legende a chaque page.
- **Aucune valeur ecrite en dur** : tout vient des CSV de `results/`. Une figure
  qui ne se regenere pas a l'identique n'a rien a faire dans le memoire.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.figures import DPI, FIGURES_DIR  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ordre d'affichage et libelles du memoire. L'ordre suit la lecture du chapitre 6 :
# reference basse, reference classique, strategie proposee, ablation.
STRATEGIES = ("static", "periodic", "proposed", "ablation")
LIBELLES = {
    "static": "B0 — modele fige",
    "periodic": "B1 — periodique",
    "proposed": "P — proposee",
    "ablation": "A1 — P sans validation",
}
COULEURS = {
    "static": "#7f7f7f",
    "periodic": "#1f77b4",
    "proposed": "#d62728",
    "ablation": "#ff7f0e",
}
SCENARIOS = {
    "s1_soudaine": "S1 soudaine",
    "s2_incrementale": "S2 incrementale",
    "s3_graduelle": "S3 graduelle",
    "s4_recurrente": "S4 recurrente",
}

SAMPLING_INTERVAL_SECONDS = 60


# --------------------------------------------------------------- utilitaires


def _par_machine(summary: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Moyenne des graines, machine par machine (OBS06)."""
    presentes = [nom for nom in colonnes if nom in summary.columns]
    return (
        summary.groupby(["strategy", "scenario", "machine"], as_index=False)[presentes]
        .mean()
        .reset_index(drop=True)
    )


def _moyenne_et_erreur(
    par_machine: pd.DataFrame, colonne: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moyenne sur les machines et erreur standard, par (scenario, strategie).

    L'erreur standard est calculee **sur les machines**, pas sur les rejeux : elle
    repond a « une autre machine donnerait-elle autre chose ? », qui est la
    question du chapitre 6.
    """
    groupes = par_machine.groupby(["scenario", "strategy"])[colonne]
    moyennes = groupes.mean().unstack()
    erreurs = (groupes.std(ddof=1) / np.sqrt(groupes.count())).unstack()
    return moyennes, erreurs


def _f1_glissant(y_true: np.ndarray, alerts: np.ndarray, largeur: int) -> np.ndarray:
    """F1 sur une fenetre glissante de `largeur` fenetres, par sommes cumulees.

    Renvoie NaN sur les positions ou la fenetre ne contient aucun positif et
    aucune alerte : le F1 n'y est pas defini, et le remplacer par 0 ferait croire
    a un effondrement la ou il n'y a simplement rien a detecter.
    """
    vrais = y_true.astype(np.float64)
    tires = alerts.astype(np.float64)

    def fenetre(valeurs: np.ndarray) -> np.ndarray:
        cumul = np.concatenate([[0.0], np.cumsum(valeurs)])
        fin = np.arange(largeur, len(valeurs) + 1)
        return cumul[fin] - cumul[fin - largeur]

    tp = fenetre(vrais * tires)
    positifs = fenetre(vrais)
    alertes = fenetre(tires)

    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2.0 * tp / (positifs + alertes)
    f1[(positifs + alertes) == 0] = np.nan
    return f1


def _traces(machine: str, scenario: str, strategie: str, seed: int) -> dict[str, np.ndarray]:
    chemin = (
        PROJECT_ROOT
        / "results"
        / "runs"
        / "smd"
        / machine
        / scenario
        / strategie
        / f"seed{seed}"
        / "traces.npz"
    )
    with np.load(chemin) as donnees:
        return {nom: donnees[nom] for nom in donnees.files}


# ------------------------------------------------------------ F1 glissant


def plot_f1_glissant(
    machine: str,
    scenario: str,
    seed: int = 42,
    largeur: int = 1000,
    summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """F1 glissant des quatre strategies sur un meme flux, derive marquee.

    C'est la figure demandee au §11 (« F1 glissant par strategie »). Les quatre
    courbes partent du meme modele initial : tout ecart apres l'instant de derive
    vient de la strategie de reentrainement, pas de l'initialisation.
    """
    chemin_synthese = Path(summary_path) if summary_path else PROJECT_ROOT / "results/summary.csv"
    synthese = pd.read_csv(chemin_synthese)
    lignes = synthese[
        (synthese.machine == machine) & (synthese.scenario == scenario) & (synthese.seed == seed)
    ]
    if lignes.empty:
        raise ValueError(f"aucun rejeu pour {machine} / {scenario} / graine {seed}")

    figure, axes = plt.subplots(figsize=(11, 4.4), constrained_layout=True)

    derive = int(lignes.drift_window.iloc[0])
    # `drift_end_index` compte des points, l'axe de la figure des fenetres : sans
    # cette conversion la bande de transition de S2 et S3 serait decalee.
    fin_point = lignes.drift_end_index.iloc[0]
    taille = int(lignes.window_size.iloc[0])
    pas = int(lignes.stride.iloc[0])
    derive_fin = None if pd.isna(fin_point) else (int(fin_point) - taille + 1) // pas

    for strategie in STRATEGIES:
        if strategie not in set(lignes.strategy):
            continue
        traces = _traces(machine, scenario, strategie, seed)
        f1 = _f1_glissant(traces["y_true"], traces["alerts"], largeur)
        # Le F1 d'une fenetre glissante est porte par sa fin : a la position i, la
        # courbe resume ce qui vient de se passer, jamais ce qui va arriver.
        positions = np.arange(largeur - 1, largeur - 1 + len(f1))
        heures = positions * SAMPLING_INTERVAL_SECONDS / 3600.0
        axes.plot(
            heures,
            f1,
            linewidth=1.1,
            color=COULEURS[strategie],
            label=LIBELLES[strategie],
        )

        # Promotions : une marque par changement de modele en service.
        changements = traces.get("model_changes")
        if changements is not None and len(changements):
            visibles = changements[changements >= largeur - 1]
            if len(visibles):
                axes.scatter(
                    visibles * SAMPLING_INTERVAL_SECONDS / 3600.0,
                    np.full(len(visibles), -0.02),
                    marker="^",
                    s=18,
                    color=COULEURS[strategie],
                    clip_on=False,
                    label="_nolegend_",
                )

    heure_derive = derive * SAMPLING_INTERVAL_SECONDS / 3600.0
    axes.axvline(heure_derive, color="black", linestyle="--", linewidth=1.0)
    axes.annotate(
        "debut de la derive",
        xy=(heure_derive, 1.0),
        xytext=(4, -2),
        textcoords="offset points",
        fontsize=8,
        va="top",
    )
    if derive_fin is not None and derive_fin > derive:
        axes.axvspan(
            heure_derive,
            derive_fin * SAMPLING_INTERVAL_SECONDS / 3600.0,
            color="black",
            alpha=0.07,
            linewidth=0,
            label="transition de la derive",
        )

    axes.set_xlabel("Temps depuis le debut du flux (heures)")
    axes.set_ylabel(f"F1 sur les {largeur} dernieres fenetres")
    axes.set_ylim(-0.02, 1.0)
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9, ncols=2)
    axes.set_title(
        f"{machine} — {SCENARIOS.get(scenario, scenario)} — graine {seed}"
        "  (triangles : promotions de modele)",
        fontsize=10,
    )

    destination = (
        Path(output_path)
        if output_path
        else FIGURES_DIR / f"f1_glissant_{machine}_{scenario}_seed{seed}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


# ------------------------------------------------------- cout et qualite


def plot_cout_reentrainement(
    summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Nombre de reentrainements et secondes CPU, par scenario et strategie.

    Deux panneaux plutot qu'un axe secondaire : superposer un compte et des
    secondes sur deux echelles invite a lire un croisement de courbes qui n'a
    aucun sens.
    """
    chemin = Path(summary_path) if summary_path else PROJECT_ROOT / "results/summary.csv"
    synthese = pd.read_csv(chemin)
    synthese = synthese[synthese.scenario != "none"]
    colonnes = ["cost_n_retrains", "cost_retrain_seconds", "cost_n_promoted", "cost_n_rejected"]
    par_machine = _par_machine(synthese, colonnes)

    figure, paires = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for axes, colonne, titre, unite in (
        (paires[0], "cost_n_retrains", "Reentrainements declenches", "nombre par rejeu"),
        (paires[1], "cost_retrain_seconds", "Cout de calcul du reentrainement", "secondes CPU"),
    ):
        moyennes, erreurs = _moyenne_et_erreur(par_machine, colonne)
        scenarios = [nom for nom in SCENARIOS if nom in moyennes.index]
        strategies = [nom for nom in STRATEGIES if nom in moyennes.columns]
        x = np.arange(len(scenarios))
        largeur = 0.8 / max(len(strategies), 1)
        for rang, strategie in enumerate(strategies):
            axes.bar(
                x + rang * largeur - 0.4 + largeur / 2,
                moyennes.loc[scenarios, strategie],
                largeur,
                yerr=erreurs.loc[scenarios, strategie],
                capsize=2.5,
                color=COULEURS[strategie],
                label=LIBELLES[strategie],
                error_kw={"linewidth": 0.8},
            )
        axes.set_xticks(x, [SCENARIOS[nom] for nom in scenarios], fontsize=8)
        axes.set_ylabel(unite)
        axes.set_title(titre, fontsize=10)
        axes.grid(axis="y", alpha=0.25, linewidth=0.5)
    paires[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    figure.suptitle(
        "Cout du reentrainement (moyenne sur 12 machines d'evaluation, "
        "barres : erreur standard inter-machines)",
        fontsize=10,
    )

    destination = Path(output_path) if output_path else FIGURES_DIR / "cout_reentrainement.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def plot_f1_avant_apres(
    summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """F1 avant et apres la derive, par scenario et strategie.

    Le F1 avant derive doit etre quasi identique pour les quatre strategies : avant
    la derive, aucune n'a encore de raison d'avoir reentraine. C'est le controle de
    coherence de la figure — un ecart visible a gauche signalerait un defaut du
    harnais, pas un resultat.
    """
    chemin = Path(summary_path) if summary_path else PROJECT_ROOT / "results/summary.csv"
    synthese = pd.read_csv(chemin)
    synthese = synthese[synthese.scenario != "none"]
    par_machine = _par_machine(synthese, ["f1_before", "f1_after"])

    figure, paires = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True, sharey=True)
    for axes, colonne, titre in (
        (paires[0], "f1_before", "Avant la derive"),
        (paires[1], "f1_after", "Apres la derive"),
    ):
        moyennes, erreurs = _moyenne_et_erreur(par_machine, colonne)
        scenarios = [nom for nom in SCENARIOS if nom in moyennes.index]
        strategies = [nom for nom in STRATEGIES if nom in moyennes.columns]
        x = np.arange(len(scenarios))
        largeur = 0.8 / max(len(strategies), 1)
        for rang, strategie in enumerate(strategies):
            axes.bar(
                x + rang * largeur - 0.4 + largeur / 2,
                moyennes.loc[scenarios, strategie],
                largeur,
                yerr=erreurs.loc[scenarios, strategie],
                capsize=2.5,
                color=COULEURS[strategie],
                label=LIBELLES[strategie],
                error_kw={"linewidth": 0.8},
            )
        axes.set_xticks(x, [SCENARIOS[nom] for nom in scenarios], fontsize=8)
        axes.set_title(titre, fontsize=10)
        axes.grid(axis="y", alpha=0.25, linewidth=0.5)
    paires[0].set_ylabel("F1 par fenetre")
    paires[1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    figure.suptitle(
        "Qualite de detection de part et d'autre de l'instant de derive "
        "(12 machines d'evaluation, 5 graines)",
        fontsize=10,
    )

    destination = Path(output_path) if output_path else FIGURES_DIR / "f1_avant_apres.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def plot_qualite_contre_cout(
    summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """F1 apres derive en fonction du cout de reentrainement, une marque par strategie.

    La question du memoire est un arbitrage (« mieux qu'un modele fige, a moindre
    cout qu'un reentrainement periodique »). Une figure a deux axes la pose
    directement : l'ideal est en haut a gauche.
    """
    chemin = Path(summary_path) if summary_path else PROJECT_ROOT / "results/summary.csv"
    synthese = pd.read_csv(chemin)
    synthese = synthese[synthese.scenario != "none"]
    par_machine = _par_machine(synthese, ["f1_after", "cost_retrain_seconds", "cost_n_retrains"])

    figure, axes = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for strategie in STRATEGIES:
        bloc = par_machine[par_machine.strategy == strategie]
        if bloc.empty:
            continue
        x = bloc.cost_n_retrains.mean()
        y = bloc.f1_after.mean()
        ex = bloc.cost_n_retrains.std(ddof=1) / np.sqrt(len(bloc))
        ey = bloc.f1_after.std(ddof=1) / np.sqrt(len(bloc))
        axes.errorbar(
            x,
            y,
            xerr=ex,
            yerr=ey,
            marker="o",
            markersize=8,
            capsize=3,
            linewidth=1.0,
            color=COULEURS[strategie],
            label=LIBELLES[strategie],
        )
        axes.annotate(
            LIBELLES[strategie].split(" ")[0],
            xy=(x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            color=COULEURS[strategie],
        )

    axes.set_xlabel("Reentrainements par rejeu (moyenne sur les machines)")
    axes.set_ylabel("F1 apres la derive")
    axes.set_xscale("symlog", linthresh=1.0)
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.set_title(
        "Arbitrage qualite / cout, tous scenarios confondus\n"
        "(axe des couts en echelle symlog : B1 est a un ordre de grandeur)",
        fontsize=10,
    )
    axes.legend(loc="lower right", fontsize=8, framealpha=0.9)

    destination = Path(output_path) if output_path else FIGURES_DIR / "qualite_contre_cout.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


# --------------------------------------------------- specificite du moniteur


def _machines_stables(screening_path: str | Path | None = None) -> set[str]:
    """Machines declarees stables au criblage (OBS03, critere des fausses alertes)."""
    chemin = Path(screening_path) if screening_path else PROJECT_ROOT / "results/smd_screening.csv"
    table = pd.read_csv(chemin)
    return set(table.loc[table.stable.astype(bool), "machine"])


def plot_confirmations_contre_k(
    calibration_path: str | Path | None = None,
    drift_config_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Effet de K sur la confirmation de derive, au point de fonctionnement fige.

    Figure du compromis central de l'etape de confirmation. Elle porte **trois**
    courbes, parce que deux ne suffisent pas a expliquer ce qu'on observe :
    le taux de detection monte avec K jusqu'a 960 puis redescend (D29), ce qui
    est contre-intuitif tant qu'on ne voit pas la troisieme. La confirmation
    etant latchee — une au plus par flux —, un K trop court confirme *avant*
    l'instant de derive, sur une fluctuation, et verrouille la detection de la
    vraie derive qui suit. La courbe des confirmations prematurees est donc la
    cause, pas un detail.

    Le balayage de `calibration_m2_dev.csv` croise cinq parametres. On y fixe
    les quatre autres a leurs valeurs **figees** (lues dans `configs/drift.yaml`,
    jamais recopiees ici) et on ne fait varier que K : moyenner sur la grille
    entiere melangerait des points de fonctionnement qui n'ont pas ete retenus.

    Renvoie le chemin de la figure et le tableau tracé, pour que le chapitre 6
    puisse citer les memes nombres que la figure.
    """
    from src.config import load_config

    chemin = (
        Path(calibration_path)
        if calibration_path
        else PROJECT_ROOT / "results/calibration_m2_dev.csv"
    )
    table = pd.read_csv(chemin)
    if "confirmation_windows" not in table.columns:
        raise ValueError(f"{chemin} n'a pas de colonne confirmation_windows")

    reglage = load_config(drift_config_path or "drift.yaml")
    fige = {
        "recent_windows": int(reglage["recent_windows"]),
        "quantile": float(reglage["feature_threshold"]["quantile"]),
        "margin": float(reglage["feature_threshold"]["margin"]),
        "drifting_share": float(reglage["drifting_share"]),
    }
    for colonne, valeur in fige.items():
        table = table[table[colonne] == valeur]
    if table.empty:
        raise ValueError(f"aucune ligne de {chemin} au point de fonctionnement fige {fige}")

    stables = _machines_stables()
    injectes = table[table.scenario != "none"]
    sans_injection = table[table.scenario == "none"]
    sans_injection_stables = sans_injection[sans_injection.machine.isin(stables)]

    mesures = pd.DataFrame(
        {
            "n_injections": injectes.groupby("confirmation_windows").size(),
            "detections": injectes.groupby("confirmation_windows").detection_apres_derive.sum(),
            "confirmations_prematurees": injectes.groupby(
                "confirmation_windows"
            ).confirmation_avant_derive.sum(),
            "n_stables": sans_injection_stables.groupby("confirmation_windows").size(),
            "confirmations_stables": sans_injection_stables.groupby(
                "confirmation_windows"
            ).confirme.sum(),
            "n_sans_injection": sans_injection.groupby("confirmation_windows").size(),
            "confirmations_sans_injection": sans_injection.groupby(
                "confirmation_windows"
            ).confirme.sum(),
        }
    ).sort_index()
    mesures["part_detectee"] = 100.0 * mesures.detections / mesures.n_injections
    mesures["part_prematuree"] = 100.0 * mesures.confirmations_prematurees / mesures.n_injections
    mesures["part_stables"] = 100.0 * mesures.confirmations_stables / mesures.n_stables

    figure, axes = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    axes.plot(
        mesures.index,
        mesures.part_detectee,
        marker="o",
        linewidth=1.3,
        color="#2ca02c",
        label=f"derives injectees detectees (n = {int(mesures.n_injections.iloc[0])})",
    )
    axes.plot(
        mesures.index,
        mesures.part_prematuree,
        marker="v",
        linewidth=1.3,
        color="#9467bd",
        linestyle="-.",
        label="confirmees avant l'instant de derive (cause de la non-monotonie)",
    )
    axes.plot(
        mesures.index,
        mesures.part_stables,
        marker="s",
        linewidth=1.3,
        color="#d62728",
        linestyle="--",
        label=f"machines stables confirmees a tort (n = {int(mesures.n_stables.iloc[0])})",
    )

    retenu = int(reglage["confirmation_windows"])
    if retenu in mesures.index:
        axes.axvline(retenu, color="black", linewidth=0.9, linestyle=":")
        axes.annotate(
            f"K retenu = {retenu} fenetres (16 h)",
            xy=(retenu, 100.0),
            xytext=(-6, -4),
            textcoords="offset points",
            fontsize=8,
            ha="right",
            va="top",
        )

    axes.set_xlabel("K — fenetres consecutives de signal exigees pour confirmer")
    axes.set_ylabel("Part des rejeux (%)")
    axes.set_ylim(-3, 103)
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.legend(loc="center right", fontsize=8, framealpha=0.9)
    axes.set_title(
        "Effet de K sur la confirmation de derive — 16 machines de mise au point,\n"
        "quatre autres parametres du moniteur figes a leurs valeurs retenues (D29)",
        fontsize=10,
    )

    destination = Path(output_path) if output_path else FIGURES_DIR / "confirmations_contre_k.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination, mesures
