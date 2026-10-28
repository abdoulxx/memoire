"""Tests des tests statistiques : ce qui decide d'une conclusion du chapitre 6.

Un module qui produit des p-valeurs est un endroit ou une erreur ne se voit pas
— le resultat reste un nombre plausible. D'ou des cas calcules a la main, et deux
tests qui portent sur l'abstention plutot que sur le resultat : un test qui
conclut quand il ne peut pas conclure est le defaut le plus couteux ici.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.hypotheses import _holm, _par_machine, par_groupe
from src.evaluation.hypotheses import tester as lancer_les_tests


def _synthese(lignes: list[dict]) -> pd.DataFrame:
    """Construit une synthese minimale, avec les colonnes que `tester` exige."""
    defauts = {
        "scenario": "s1_soudaine",
        "seed": 42,
        "f1_after": 0.0,
        "event_recall": 0.0,
        "false_alerts_per_hour": 0.0,
        "recovery_minutes": 0.0,
        "cost_n_retrains": 0.0,
        "cost_retrain_seconds": 0.0,
        "cost_n_promoted": 0.0,
        "n_promotions_degradantes": 0.0,
    }
    return pd.DataFrame([{**defauts, **ligne} for ligne in lignes])


def _appliquer(table: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Passe une table en memoire a `tester`, via un CSV temporaire."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "summary.csv"
        table.to_csv(chemin, index=False)
        return lancer_les_tests(chemin, **kwargs)


# ------------------------------------------------------------------ Holm


def test_holm_sur_un_cas_calcule_a_la_main() -> None:
    """m = 3, p = 0,01 / 0,02 / 0,04 : Holm donne 0,03 / 0,04 / 0,04.

    Le troisieme est 0,04 x 1 = 0,04, mais le deuxieme vaut 0,02 x 2 = 0,04 et
    Holm impose la monotonie : le troisieme ne peut pas etre inferieur au
    deuxieme. C'est precisement la partie que l'on oublie en reimplementant la
    correction, et elle change le verdict a alpha = 0,05.
    """
    assert _holm([0.01, 0.02, 0.04]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_ne_compte_pas_les_tests_impossibles() -> None:
    """Un `None` (test non realisable) ne doit pas alourdir la correction.

    Sinon s'abstenir sur une metrique penaliserait les autres, ce qui pousserait
    a forcer un test plutot qu'a s'abstenir.
    """
    assert _holm([0.01, None, 0.02]) == pytest.approx([0.02, None, 0.02])


def test_holm_plafonne_a_un() -> None:
    assert _holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


# ------------------------------------------------- moyenne par machine


def test_la_moyenne_par_machine_precede_le_test() -> None:
    """Trois graines d'une machine donnent une seule observation (OBS06)."""
    table = _synthese(
        [
            {"machine": "machine-1-1", "strategy": "proposed", "seed": 42, "f1": 0.2},
            {"machine": "machine-1-1", "strategy": "proposed", "seed": 43, "f1": 0.4},
            {"machine": "machine-1-1", "strategy": "proposed", "seed": 44, "f1": 0.6},
        ]
    )
    moyennes = _par_machine(table)
    assert len(moyennes) == 1
    assert moyennes.f1.iloc[0] == pytest.approx(0.4)


# --------------------------------------------------------- appariement


def test_une_machine_sans_la_strategie_de_reference_est_ecartee() -> None:
    """L'appariement se fait sur le flux : pas de paire, pas d'observation.

    Remplacer la valeur manquante par une moyenne de groupe reviendrait a
    comparer P a une reference qui n'a jamais vu ce flux.
    """
    lignes = []
    for i in range(1, 8):
        lignes.append({"machine": f"machine-1-{i}", "strategy": "proposed", "f1": 0.5})
        if i != 7:  # machine-1-7 n'a pas de rejeu B0
            lignes.append({"machine": f"machine-1-{i}", "strategy": "static", "f1": 0.1})
    resultats = _appliquer(_synthese(lignes))
    h1 = resultats[(resultats.hypothese == "H1") & (resultats.metrique == "f1")]
    assert set(h1.n_paires) == {6}


# -------------------------------------------------------- verdicts


def test_un_ecart_net_et_conforme_est_declare_conforme() -> None:
    """P a un F1 superieur a B0 sur les 12 machines : verdict conforme."""
    lignes = []
    for i in range(1, 13):
        lignes.append({"machine": f"machine-1-{i}", "strategy": "proposed", "f1": 0.5})
        lignes.append({"machine": f"machine-1-{i}", "strategy": "static", "f1": 0.2})
    resultats = _appliquer(_synthese(lignes))
    ligne = resultats[
        (resultats.hypothese == "H1")
        & (resultats.metrique == "f1")
        & (resultats.scenario == "s1_soudaine")
    ].iloc[0]
    assert ligne.n_paires == 12
    assert ligne.sens_observe == "superieur"
    assert ligne.mediane_ecart == pytest.approx(0.3)
    assert ligne.p_holm < 0.05
    assert ligne.verdict == "conforme"


def test_un_ecart_net_mais_inverse_est_declare_contraire() -> None:
    """La conclusion doit nommer l'echec, pas se taire.

    Si P faisait moins bien que le modele fige, une p-valeur unilaterale non
    significative laisserait croire a une absence d'effet. Le verdict doit dire
    « contraire a l'attendu ».
    """
    lignes = []
    for i in range(1, 13):
        lignes.append({"machine": f"machine-1-{i}", "strategy": "proposed", "f1": 0.2})
        lignes.append({"machine": f"machine-1-{i}", "strategy": "static", "f1": 0.5})
    resultats = _appliquer(_synthese(lignes))
    ligne = resultats[
        (resultats.hypothese == "H1")
        & (resultats.metrique == "f1")
        & (resultats.scenario == "s1_soudaine")
    ].iloc[0]
    assert ligne.verdict == "contraire a l'attendu"


def test_la_direction_ne_se_lit_pas_sur_la_mediane_des_ecarts() -> None:
    """Majorite d'ex aequo : la mediane des ecarts vaut 0, la direction non.

    C'est le cas normal de `n_promotions_degradantes`, nulle sur la plupart des
    machines. Ici 9 machines sur 20 ou P fait mieux que A1, 11 a egalite : la
    mediane des ecarts vaut 0, mais Wilcoxon ignore les ex aequo et mesure une
    direction nette. Lire la direction sur la mediane affichait « contraire a
    l'attendu » une hypothese pourtant verifiee — c'est la pire erreur possible
    ici, puisqu'elle inverse la conclusion du memoire.

    Il faut 20 machines, pas 12 : Wilcoxon exige au moins 6 ecarts non nuls pour
    pouvoir descendre sous p = 0,05, et il faut en meme temps que les ex aequo
    restent majoritaires pour que la mediane soit nulle. Avec 12 machines les
    deux conditions sont incompatibles.
    """
    lignes = []
    for i in range(1, 21):
        degradantes_p, degradantes_a1 = (0.0, 1.0) if i <= 9 else (0.0, 0.0)
        lignes.append(
            {
                "machine": f"machine-1-{i}",
                "strategy": "proposed",
                "f1": 0.3,
                "n_promotions_degradantes": degradantes_p,
            }
        )
        lignes.append(
            {
                "machine": f"machine-1-{i}",
                "strategy": "ablation",
                "f1": 0.3,
                "n_promotions_degradantes": degradantes_a1,
            }
        )
    resultats = _appliquer(_synthese(lignes))
    ligne = resultats[
        (resultats.hypothese == "H3")
        & (resultats.metrique == "n_promotions_degradantes")
        & (resultats.scenario == "s1_soudaine")
    ].iloc[0]
    assert ligne.mediane_ecart == pytest.approx(0.0)  # la mediane est bien nulle
    assert ligne.n_ecarts_non_nuls == 9
    assert ligne.n_ecarts_negatifs == 9
    assert ligne.somme_rangs_signes < 0
    assert ligne.sens_observe == "inferieur"
    assert ligne.verdict == "conforme"


def test_le_test_s_abstient_quand_les_ecarts_non_nuls_sont_trop_rares() -> None:
    """4 ecarts non nuls sur 12 paires : aucune p-valeur, verdict indecidable.

    Avec si peu d'ecarts, la p-valeur minimale atteignable depasse 0,05. Produire
    un nombre serait pire que de s'abstenir, parce qu'il serait lu.
    """
    lignes = []
    for i in range(1, 13):
        ecart = 0.3 if i <= 4 else 0.0
        lignes.append({"machine": f"machine-1-{i}", "strategy": "proposed", "f1": 0.2 + ecart})
        lignes.append({"machine": f"machine-1-{i}", "strategy": "static", "f1": 0.2})
    resultats = _appliquer(_synthese(lignes))
    ligne = resultats[
        (resultats.hypothese == "H1")
        & (resultats.metrique == "f1")
        & (resultats.scenario == "s1_soudaine")
    ].iloc[0]
    assert ligne.n_ecarts_non_nuls == 4
    assert pd.isna(ligne.p_value)
    assert ligne.verdict == "indecidable"


def test_chaque_scenario_est_teste_separement_puis_l_ensemble() -> None:
    """Une strategie n'a aucune raison de se comporter pareil sous S1 et sous S4."""
    lignes = []
    for scenario in ("s1_soudaine", "s4_recurrente"):
        for i in range(1, 13):
            lignes.append(
                {
                    "machine": f"machine-1-{i}",
                    "scenario": scenario,
                    "strategy": "proposed",
                    "f1": 0.5,
                }
            )
            lignes.append(
                {
                    "machine": f"machine-1-{i}",
                    "scenario": scenario,
                    "strategy": "static",
                    "f1": 0.2,
                }
            )
    resultats = _appliquer(_synthese(lignes))
    scenarios = set(resultats[resultats.metrique == "f1"].scenario)
    assert scenarios == {"s1_soudaine", "s4_recurrente", "tous"}
    ensemble = resultats[(resultats.metrique == "f1") & (resultats.scenario == "tous")].iloc[0]
    assert ensemble.n_paires == 24  # 12 machines x 2 scenarios, apparies par flux


def test_le_scenario_sans_derive_est_exclu() -> None:
    """`none` n'a pas de derive injectee : il n'a rien a faire dans ces tests."""
    lignes = []
    for i in range(1, 13):
        lignes.append(
            {"machine": f"machine-1-{i}", "scenario": "none", "strategy": "proposed", "f1": 0.5}
        )
        lignes.append(
            {"machine": f"machine-1-{i}", "scenario": "none", "strategy": "static", "f1": 0.2}
        )
    assert _appliquer(_synthese(lignes)).empty


# ----------------------------------------------------------- par groupe


def test_les_moyennes_par_groupe_regroupent_bien_les_machines() -> None:
    """Le §4 demande de rapporter aussi par groupe, les groupes etant correles."""
    lignes = [
        {"machine": "machine-1-1", "strategy": "proposed", "f1": 0.2},
        {"machine": "machine-1-2", "strategy": "proposed", "f1": 0.4},
        {"machine": "machine-3-1", "strategy": "proposed", "f1": 0.9},
    ]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "summary.csv"
        _synthese(lignes).to_csv(chemin, index=False)
        groupes = par_groupe(chemin)
    assert set(groupes.groupe) == {"1", "3"}
    assert groupes[groupes.groupe == "1"].f1.iloc[0] == pytest.approx(0.3)
