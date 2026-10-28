"""Tableaux du chapitre 6, engendres depuis `results/` (livrable M3, phase D).

Les figures montrent, les tableaux chiffrent : le §11 demande que rien ne soit
ecrit dans le memoire sans fichier source correspondant, donc aucun des nombres
ci-dessous n'est recopie a la main. Ce module relit les memes CSV que
`result_figures.py` et ecrit un seul fichier Markdown, pret a etre transpose dans
le document.

Les trois regles de `result_figures.py` tiennent ici aussi, et une quatrieme s'y
ajoute : **toute valeur agregee est donnee avec sa dispersion**. Un F1 moyen de
0,31 sans barre d'erreur laisse croire a une precision que la campagne n'a pas —
les ecarts entre strategies sont du meme ordre que l'ecart entre deux machines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.result_figures import (
    LIBELLES,
    PROJECT_ROOT,
    SCENARIOS,
    STRATEGIES,
    _machines_stables,
    _par_machine,
)

TABLES_PATH = PROJECT_ROOT / "docs/tableaux_chapitre6.md"

# Metrique -> (libelle du memoire, nombre de decimales). L'ordre suit le §7 :
# qualite de detection, adaptation, cout, fiabilite des mises a jour.
METRIQUES = [
    ("f1", "F1 par fenetre", 3),
    ("f1_before", "F1 avant la derive", 3),
    ("f1_after", "F1 apres la derive", 3),
    ("event_recall", "Rappel par evenement", 3),
    ("false_alerts_per_hour", "Fausses alertes / h", 2),
    ("detection_delay_mean_minutes", "Delai de detection, moyen (min)", 1),
    ("detection_delay_median_minutes", "Delai de detection, median (min)", 1),
    ("recovery_minutes", "Temps de recuperation (min)", 0),
    ("drift_detection_delay_minutes", "Delai de detection de la derive (min)", 0),
    ("cost_n_retrains", "Reentrainements", 2),
    ("cost_n_promoted", "dont promus", 2),
    ("cost_n_rejected", "dont rejetes", 2),
    ("cost_retrain_seconds", "Secondes CPU de reentrainement", 2),
    ("n_promotions_degradantes", "Promotions degradantes", 3),
]


def _format(moyenne: float, erreur: float, decimales: int) -> str:
    """`moyenne ± erreur`, ou `—` si la metrique n'existe pas pour la strategie.

    Une case vide n'est pas une valeur nulle : B0 n'a pas de delai de detection
    de derive parce qu'il ne surveille rien, et l'ecrire « 0 » serait faux.
    """
    if moyenne is None or not np.isfinite(moyenne):
        return "—"
    if erreur is None or not np.isfinite(erreur):
        return f"{moyenne:.{decimales}f}"
    return f"{moyenne:.{decimales}f} ± {erreur:.{decimales}f}"


def _moyenne_sur_machines(par_machine: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Moyenne et erreur standard **sur les machines**, par strategie."""
    groupes = par_machine.groupby("strategy")[colonne]
    return pd.DataFrame(
        {
            "moyenne": groupes.mean(),
            "erreur": groupes.std(ddof=1) / np.sqrt(groupes.count()),
        }
    )


def _table_synthese(injectes: pd.DataFrame) -> list[str]:
    colonnes = [nom for nom, _, _ in METRIQUES]
    # La moyenne par machine passe d'abord par le scenario, puis on moyenne les
    # quatre scenarios : sans cela, un scenario ou une strategie ne produit aucune
    # ligne peserait moins que les autres dans la moyenne generale.
    par_machine = _par_machine(injectes, colonnes)
    par_machine = par_machine.groupby(["strategy", "machine"], as_index=False).mean(
        numeric_only=True
    )

    lignes = ["| Metrique | " + " | ".join(LIBELLES[s] for s in STRATEGIES) + " |"]
    lignes.append("|---" * (len(STRATEGIES) + 1) + "|")
    for colonne, libelle, decimales in METRIQUES:
        if colonne not in par_machine.columns:
            continue
        stats = _moyenne_sur_machines(par_machine, colonne)
        cases = []
        for strategie in STRATEGIES:
            if strategie not in stats.index:
                cases.append("—")
                continue
            cases.append(
                _format(stats.loc[strategie, "moyenne"], stats.loc[strategie, "erreur"], decimales)
            )
        lignes.append(f"| {libelle} | " + " | ".join(cases) + " |")
    return lignes


def _table_par_scenario(injectes: pd.DataFrame, colonne: str, decimales: int) -> list[str]:
    par_machine = _par_machine(injectes, [colonne])
    lignes = ["| Scenario | " + " | ".join(LIBELLES[s] for s in STRATEGIES) + " |"]
    lignes.append("|---" * (len(STRATEGIES) + 1) + "|")
    for scenario, libelle in SCENARIOS.items():
        bloc = par_machine[par_machine.scenario == scenario]
        if bloc.empty:
            continue
        stats = _moyenne_sur_machines(bloc, colonne)
        cases = [
            _format(stats.loc[s, "moyenne"], stats.loc[s, "erreur"], decimales)
            if s in stats.index
            else "—"
            for s in STRATEGIES
        ]
        lignes.append(f"| {libelle} | " + " | ".join(cases) + " |")
    return lignes


def _table_hypotheses(chemin: Path) -> list[str]:
    """Reprise de `results/hypotheses.csv`, ligne `tous` uniquement.

    Le verdict vient du fichier, jamais d'une relecture a la main : c'est le
    meme nombre qui sort du test, du tableau et du memoire.
    """
    table = pd.read_csv(chemin)
    table = table[table.scenario == "tous"]
    lignes = [
        "| Hypothese | Metrique | Attendu | Observe | paires +/- | p (Holm) | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, ligne in table.iterrows():
        p_holm = "—" if pd.isna(ligne.p_holm) else f"{ligne.p_holm:.4f}"
        paires = f"{int(ligne.n_ecarts_positifs)}/{int(ligne.n_ecarts_negatifs)}"
        lignes.append(
            f"| {ligne.hypothese} | `{ligne.metrique}` | {ligne.sens_attendu} | "
            f"{ligne.sens_observe} | {paires} | {p_holm} | {ligne.verdict} |"
        )
    return lignes


def _table_derive_naturelle(chemin: Path) -> list[str]:
    """Scenario `none` : ce que font les strategies quand il n'y a rien a faire.

    Le tableau separe machines stables et machines instables parce que la bonne
    conduite n'y est pas la meme : ne rien faire d'un cote, s'adapter de l'autre.
    """
    table = pd.read_csv(chemin)
    stables = _machines_stables()
    colonnes = [
        ("f1", "F1", 3),
        ("false_alerts_per_hour", "Fausses alertes / h", 2),
        ("cost_n_retrains", "Reentrainements", 2),
        ("drift_false_confirmations", "Confirmations de derive", 2),
        ("n_promotions_degradantes", "Promotions degradantes", 2),
    ]
    lignes = [
        "| Perimetre | Strategie | " + " | ".join(libelle for _, libelle, _ in colonnes) + " |",
        "|---" * (len(colonnes) + 2) + "|",
    ]
    perimetres = [
        ("Stables", table[table.machine.isin(stables)]),
        ("Instables", table[~table.machine.isin(stables)]),
    ]
    for nom, bloc in perimetres:
        n_machines = bloc.machine.nunique()
        for strategie in STRATEGIES:
            part = bloc[bloc.strategy == strategie]
            if part.empty:
                continue
            cases = []
            for colonne, _, decimales in colonnes:
                if colonne not in part.columns:
                    cases.append("—")
                    continue
                valeurs = part.groupby("machine")[colonne].mean()
                erreur = valeurs.std(ddof=1) / np.sqrt(len(valeurs)) if len(valeurs) > 1 else np.nan
                cases.append(_format(valeurs.mean(), erreur, decimales))
            lignes.append(
                f"| {nom} ({n_machines} machines) | {LIBELLES[strategie]} | "
                + " | ".join(cases)
                + " |"
            )
    return lignes


def ecrire_tableaux(
    summary_path: str | Path | None = None,
    hypotheses_path: str | Path | None = None,
    naturelle_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Ecrit les tableaux du chapitre 6 dans un seul fichier Markdown."""
    chemin_synthese = Path(summary_path) if summary_path else PROJECT_ROOT / "results/summary.csv"
    synthese = pd.read_csv(chemin_synthese)
    injectes = synthese[synthese.scenario != "none"]
    if injectes.empty:
        raise ValueError(f"{chemin_synthese} ne contient aucun rejeu avec derive injectee")

    n_machines = injectes.machine.nunique()
    n_graines = injectes.seed.nunique()
    blocs: list[str] = [
        "# Tableaux du chapitre 6",
        "",
        "> Fichier **engendre** par `mdrift figures-m3`. Ne pas le modifier a la main :",
        "> toute correction se fait dans `src/evaluation/result_tables.py`, puis on",
        "> relance la commande. Source : `results/summary.csv`,",
        "> `results/hypotheses.csv`, `results/summary_m3_derive_naturelle.csv`.",
        "",
        f"Campagne M3 : **{n_machines} machines d'evaluation** x 4 scenarios x "
        f"4 strategies x **{n_graines} graines** = {len(injectes)} rejeux avec derive injectee.",
        "",
        "Toutes les valeurs sont des **moyennes sur les machines ± erreur standard",
        "inter-machines**, apres moyenne des graines machine par machine (OBS06). La",
        "dispersion repond a « une autre machine donnerait-elle autre chose ? ».",
        "",
        "## Tableau 1 — Synthese par strategie, les quatre scenarios confondus",
        "",
    ]
    blocs += _table_synthese(injectes)
    blocs += [
        "",
        "## Tableau 2 — F1 apres la derive, par scenario",
        "",
    ]
    blocs += _table_par_scenario(injectes, "f1_after", 3)
    blocs += [
        "",
        "## Tableau 3 — Cout : reentrainements declenches, par scenario",
        "",
    ]
    blocs += _table_par_scenario(injectes, "cost_n_retrains", 2)

    chemin_hypotheses = (
        Path(hypotheses_path) if hypotheses_path else PROJECT_ROOT / "results/hypotheses.csv"
    )
    if chemin_hypotheses.exists():
        blocs += [
            "",
            "## Tableau 4 — Tests d'hypotheses (Wilcoxon apparie, Holm par famille)",
            "",
            "48 paires machine x scenario. `paires +/-` compte les paires ou P est "
            "au-dessus / en dessous,",
            "ex aequo exclus ; la direction est lue sur la somme des rangs signes, pas "
            "sur la mediane des ecarts.",
            "",
        ]
        blocs += _table_hypotheses(chemin_hypotheses)

    chemin_naturelle = (
        Path(naturelle_path)
        if naturelle_path
        else PROJECT_ROOT / "results/summary_m3_derive_naturelle.csv"
    )
    if chemin_naturelle.exists():
        blocs += [
            "",
            "## Tableau 5 — Derive naturelle (scenario `none`, sans injection)",
            "",
            "Les 28 machines du SMD, graine 42. Sur une machine stable, la bonne "
            "conduite est de ne rien faire.",
            "",
            "**Lecture de la colonne « confirmations »** : sur les machines stables, "
            "toute confirmation est fausse",
            "par construction — c'est la metrique de specificite du §7, et elle vaut "
            "**0** la ou K = 30 en donnait 13",
            "sur 15 (D19). Sur les machines instables elle n'est pas fausse : le "
            "criblage a montre que ces machines",
            "derivent reellement entre leur moitie train et leur moitie test (OBS03), "
            "et la confirmation y est le",
            "comportement attendu. Les deux colonnes ne se lisent donc pas dans le meme sens.",
            "",
        ]
        blocs += _table_derive_naturelle(chemin_naturelle)

    destination = Path(output_path) if output_path else TABLES_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(blocs) + "\n", encoding="utf-8")
    return destination
