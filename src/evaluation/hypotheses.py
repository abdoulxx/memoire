"""Tests statistiques des trois hypotheses du memoire, sur `results/summary.csv`.

Le cadrage (§7) demande un « test statistique entre P et B1 sur les seeds (ex.
Wilcoxon apparie) si les ecarts sont faibles ». Ce module le fait pour les trois
comparaisons, avec trois precautions qui changent les conclusions et doivent donc
etre dans le code plutot que dans un notebook :

1. **Moyenne par machine avant le test** (OBS06). Les machines d'un meme groupe
   SMD sont correlees (0,80 entre 3-1 et 3-9) et les 5 graines d'une meme machine
   ne sont pas des observations independantes : ce sont cinq tirages de *t*_d sur
   le meme flux. Traiter les 60 couples (machine, graine) comme 60 observations
   independantes gonflerait artificiellement la puissance. L'unite
   d'observation est donc la **machine**, soit n = 12 par scenario.
2. **Les paires sont formees sur le flux, pas sur la strategie**. Un rejeu de P et
   un rejeu de B1 ne sont comparables que s'ils ont vu exactement le meme flux :
   l'appariement se fait sur (machine, scenario), et un couple dont une des deux
   strategies manque est **ecarte**, pas remplace par une moyenne.
3. **Correction de Holm a l'interieur de chaque hypothese**. Tester cinq metriques
   pour H1 puis annoncer la plus favorable serait de la peche. Les metriques de
   chaque hypothese sont donc declarees d'avance, ici, et la correction porte sur
   cette famille. Les p-valeurs brutes restent dans le fichier de sortie.

Les p-valeurs sont **bilaterales**. Les hypotheses ont bien un sens attendu, ce qui
autoriserait un test unilateral plus puissant, mais le sens observe est rapporte a
part : une hypothese qui echoue dans la direction inverse doit se lire comme telle
et non disparaitre derriere une p-valeur unilaterale non significative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from src.config import PROJECT_ROOT

HYPOTHESES_PATH = PROJECT_ROOT / "results" / "hypotheses.csv"

# Metriques declarees d'avance pour chaque hypothese, avec le sens attendu.
# `sens_attendu` vaut "superieur" si la strategie proposee devrait avoir la valeur
# la plus grande, "inferieur" si elle devrait avoir la plus petite, et "aucun" si
# l'hypothese ne porte pas sur cette metrique mais qu'elle est suivie pour le
# contexte (une hypothese de cout ne doit pas se payer d'un effondrement du F1).
PLAN: dict[str, dict[str, Any]] = {
    "H1": {
        "question": "P maintient-il la performance mieux qu'un modele fige ?",
        "reference": "static",
        "metriques": {
            "f1": "superieur",
            "f1_after": "superieur",
            "event_recall": "superieur",
            "false_alerts_per_hour": "inferieur",
            "recovery_minutes": "inferieur",
        },
    },
    "H2": {
        "question": "P y parvient-il a moindre cout qu'un reentrainement periodique ?",
        "reference": "periodic",
        "metriques": {
            "cost_n_retrains": "inferieur",
            "cost_retrain_seconds": "inferieur",
            "f1": "aucun",
            "f1_after": "aucun",
            "false_alerts_per_hour": "aucun",
        },
    },
    "H3": {
        "question": "La validation reduit-elle les promotions degradantes ?",
        "reference": "ablation",
        "metriques": {
            "n_promotions_degradantes": "inferieur",
            "cost_n_promoted": "inferieur",
            "f1": "aucun",
            "f1_after": "aucun",
        },
    },
}

STRATEGIE_TESTEE = "proposed"


@dataclass(frozen=True)
class Comparaison:
    """Resultat d'une comparaison appariee, une ligne de `results/hypotheses.csv`."""

    hypothese: str
    metrique: str
    scenario: str
    strategie: str
    reference: str
    sens_attendu: str
    n_paires: int
    n_ecarts_non_nuls: int
    n_ecarts_positifs: int
    n_ecarts_negatifs: int
    mediane_strategie: float | None
    mediane_reference: float | None
    mediane_ecart: float | None
    mediane_ecart_non_nuls: float | None
    somme_rangs_signes: float | None
    statistique: float | None
    p_value: float | None
    p_holm: float | None
    sens_observe: str
    verdict: str


def _par_machine(table: pd.DataFrame) -> pd.DataFrame:
    """Moyenne sur les graines : une ligne par (strategie, scenario, machine).

    C'est l'etape que le §7 appelle « moyenner par machine sur les graines avant
    les tests ». Les colonnes non numeriques sont abandonnees ; `recovery_minutes`
    peut valoir NaN (statut `niveau_pre_derive_nul`, OBS09) et la moyenne ignore
    alors les graines concernees, ce qui est le comportement voulu : une machine
    dont le critere est indefini pour toutes ses graines sortira du test.
    """
    numeriques = table.select_dtypes("number").columns.tolist()
    garde = ["strategy", "scenario", "machine", *numeriques]
    return table[garde].groupby(["strategy", "scenario", "machine"], as_index=False).mean()


def _holm(p_values: list[float | None]) -> list[float | None]:
    """Correction de Holm-Bonferroni sur une famille de p-valeurs.

    Holm plutot que Bonferroni : meme garantie sur le risque de premiere espece
    familial, mais uniformement plus puissante. Les entrees `None` (test
    impossible) ne comptent pas dans la taille de la famille.
    """
    indices = [i for i, p in enumerate(p_values) if p is not None]
    m = len(indices)
    ajustees: list[float | None] = list(p_values)
    if m == 0:
        return ajustees
    ordre = sorted(indices, key=lambda i: p_values[i])  # type: ignore[arg-type,index]
    precedent = 0.0
    for rang, i in enumerate(ordre):
        brute = float(p_values[i])  # type: ignore[arg-type]
        valeur = min(1.0, (m - rang) * brute)
        precedent = max(precedent, valeur)  # monotonie imposee par Holm
        ajustees[i] = precedent
    return ajustees


def _verdict(sens_attendu: str, sens_observe: str, p: float | None, seuil: float) -> str:
    """Traduit p-valeur et direction en une conclusion lisible, sans surinterpretation."""
    if p is None:
        return "indecidable"
    if p >= seuil:
        return "pas de difference detectable"
    if sens_attendu == "aucun":
        return f"difference detectee ({sens_observe})"
    return "conforme" if sens_observe == sens_attendu else "contraire a l'attendu"


def _comparer(
    hypothese: str,
    metrique: str,
    sens_attendu: str,
    scenario: str,
    teste: pd.DataFrame,
    reference: pd.DataFrame,
    nom_reference: str,
) -> Comparaison:
    """Wilcoxon apparie sur une metrique, un scenario, deux strategies."""
    paires = teste.merge(reference, on="machine", suffixes=("_a", "_b"))
    a = paires[f"{metrique}_a"]
    b = paires[f"{metrique}_b"]
    valides = a.notna() & b.notna()
    a, b = a[valides], b[valides]
    ecarts = (a - b).to_numpy()
    non_nuls = int((ecarts != 0).sum())

    statistique: float | None = None
    p: float | None = None
    # Wilcoxon a besoin d'au moins quelques ecarts non nuls pour dire quoi que ce
    # soit. En dessous de 5, la p-valeur minimale atteignable depasse deja 0,05 :
    # le test ne pourrait rien conclure et laisser croire le contraire serait pire
    # que de s'abstenir.
    if non_nuls >= 5:
        resultat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        statistique = float(resultat.statistic)
        p = float(resultat.pvalue)

    mediane_ecart = float(pd.Series(ecarts).median()) if len(ecarts) else None

    # La DIRECTION ne peut pas se lire sur la mediane des ecarts. Des que plus de
    # la moitie des machines sont a egalite — le cas normal pour
    # `n_promotions_degradantes`, nulle presque partout —, cette mediane vaut 0
    # alors que Wilcoxon, qui ignore les ex aequo, detecte une direction nette.
    # On lit donc le signe de la somme des RANGS SIGNES, c'est-a-dire la
    # statistique du test elle-meme : la direction annoncee est alors celle que le
    # test a effectivement mesuree. Sans cela, une hypothese verifiee s'affichait
    # « contraire a l'attendu ».
    ecarts_non_nuls = ecarts[ecarts != 0]
    if len(ecarts_non_nuls):
        rangs = rankdata(np.abs(ecarts_non_nuls))
        somme_rangs_signes = float(np.sum(np.sign(ecarts_non_nuls) * rangs))
        mediane_ecart_non_nuls = float(np.median(ecarts_non_nuls))
    else:
        somme_rangs_signes = None
        mediane_ecart_non_nuls = None

    if somme_rangs_signes is None or somme_rangs_signes == 0:
        sens_observe = "egal"
    else:
        sens_observe = "superieur" if somme_rangs_signes > 0 else "inferieur"

    return Comparaison(
        hypothese=hypothese,
        metrique=metrique,
        scenario=scenario,
        strategie=STRATEGIE_TESTEE,
        reference=nom_reference,
        sens_attendu=sens_attendu,
        n_paires=int(len(a)),
        n_ecarts_non_nuls=non_nuls,
        n_ecarts_positifs=int((ecarts > 0).sum()),
        n_ecarts_negatifs=int((ecarts < 0).sum()),
        mediane_strategie=float(a.median()) if len(a) else None,
        mediane_reference=float(b.median()) if len(b) else None,
        mediane_ecart=mediane_ecart,
        mediane_ecart_non_nuls=mediane_ecart_non_nuls,
        somme_rangs_signes=somme_rangs_signes,
        statistique=statistique,
        p_value=p,
        p_holm=None,
        sens_observe=sens_observe,
        verdict="",
    )


def tester(
    summary_path: Path | str = "results/summary.csv",
    alpha: float = 0.05,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Applique le plan de tests declare dans `PLAN` et renvoie une table de resultats.

    Chaque scenario est teste separement — une strategie de reentrainement n'a
    aucune raison de se comporter pareil sous une derive soudaine et sous une
    derive recurrente, et moyenner les quatre masquerait exactement ce qu'on
    cherche a mesurer. La ligne `scenario = tous` ajoute l'appariement sur les
    couples (machine, scenario) reunis, qui repond a la question globale.
    """
    table = pd.read_csv(summary_path)
    table = table[table.scenario != "none"]
    if scenarios:
        table = table[table.scenario.isin(scenarios)]
    if table.empty:
        return pd.DataFrame()

    moyennes = _par_machine(table)
    lignes: list[Comparaison] = []

    for hypothese, plan in PLAN.items():
        nom_reference = str(plan["reference"])
        metriques: dict[str, str] = plan["metriques"]
        famille: list[Comparaison] = []

        for scenario in [*sorted(moyennes.scenario.unique()), "tous"]:
            if scenario == "tous":
                # Appariement sur (machine, scenario) : chaque flux reste sa propre
                # paire, on ne moyenne pas les scenarios entre eux.
                bloc = moyennes.assign(machine=moyennes.machine + "|" + moyennes.scenario)
            else:
                bloc = moyennes[moyennes.scenario == scenario]
            teste = bloc[bloc.strategy == STRATEGIE_TESTEE]
            reference = bloc[bloc.strategy == nom_reference]
            if teste.empty or reference.empty:
                continue
            for metrique, sens in metriques.items():
                if metrique not in bloc.columns:
                    continue
                famille.append(
                    _comparer(hypothese, metrique, sens, scenario, teste, reference, nom_reference)
                )

        # Holm par hypothese et par scenario : la famille est l'ensemble des
        # metriques declarees pour cette hypothese sur ce scenario.
        for scenario in {c.scenario for c in famille}:
            groupe = [c for c in famille if c.scenario == scenario]
            ajustees = _holm([c.p_value for c in groupe])
            for comparaison, p_holm in zip(groupe, ajustees, strict=True):
                lignes.append(
                    Comparaison(
                        **{
                            **asdict(comparaison),
                            "p_holm": p_holm,
                            "verdict": _verdict(
                                comparaison.sens_attendu,
                                comparaison.sens_observe,
                                p_holm,
                                alpha,
                            ),
                        }
                    )
                )

    resultats = pd.DataFrame([asdict(ligne) for ligne in lignes])
    if resultats.empty:
        return resultats
    ordre_scenario = {nom: i for i, nom in enumerate([*sorted(moyennes.scenario.unique()), "tous"])}
    resultats["_ordre"] = resultats.scenario.map(ordre_scenario)
    resultats = resultats.sort_values(["hypothese", "_ordre", "metrique"]).drop(columns="_ordre")
    return resultats.reset_index(drop=True)


def par_groupe(summary_path: Path | str = "results/summary.csv") -> pd.DataFrame:
    """Moyennes par groupe SMD, que le §4 demande de rapporter a cote des tests.

    Les machines d'un meme groupe sont correlees : une difference portee par un
    seul groupe n'est pas la meme chose qu'une difference presente dans les trois.
    Cette table permet de le voir sans refaire de test.
    """
    table = pd.read_csv(summary_path)
    table = table[table.scenario != "none"].copy()
    if table.empty:
        return pd.DataFrame()
    table["groupe"] = table.machine.str.split("-").str[1]
    colonnes = [
        "f1",
        "f1_before",
        "f1_after",
        "event_recall",
        "false_alerts_per_hour",
        "cost_n_retrains",
        "cost_retrain_seconds",
        "n_promotions_degradantes",
        "drift_detection_delay_minutes",
    ]
    colonnes = [c for c in colonnes if c in table.columns]
    par_machine = _par_machine(table.assign(machine=table.machine))
    par_machine["groupe"] = par_machine.machine.str.split("-").str[1]
    return par_machine.groupby(["strategy", "groupe"], as_index=False)[colonnes].mean().round(4)


def ecrire(resultats: pd.DataFrame, path: Path | str = HYPOTHESES_PATH) -> Path:
    """Ecrit la table des tests, en creant le dossier au besoin."""
    sortie = Path(path)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    resultats.to_csv(sortie, index=False)
    return sortie
