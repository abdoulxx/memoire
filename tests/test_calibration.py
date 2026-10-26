"""Tests des mesures de mise au point des etapes 3 et 4 (D13, D14).

Tous les cas sont calcules a la main. Deux pieges y sont encodes, parce qu'ils ont
tous deux fausse une mesure avant d'etre corriges (voir docs/decisions.md) :

- la datation de la confirmation doit etre **causale**, une moyenne centree
  anticipe de la moitie de la fenetre (OBS03) ;
- deux regles d'exclusion qui gardent exactement les memes fenetres donnent le
  meme modele : elles doivent etre fusionnees, sinon ce modele compte plusieurs
  fois dans la correlation de l'etape 4 et la gonfle (D14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.calibration import (
    Protocol,
    _bursts,
    confirmation_point,
    exclusion_rules,
    exclusion_verdict,
    guardrail_bounds,
    retrain_instants,
    retrain_verdict,
    validation_verdict,
)

PROTOCOLE = {
    "w_train": 2000,
    "v_valid": 500,
    "confirm_ratio": 1.0,
    "confirm_k": 2,
    "rolling": 2,
    "min_train_windows": 300,
    "min_downstream_windows": 2000,
    "retrain_points": [0.2, 0.4],
    "include_confirmation": True,
    "seeds": [42],
    "healthy_f1_margin": 0.05,
}


def _protocole(**surcharges: object) -> Protocol:
    return Protocol.from_config({**PROTOCOLE, **surcharges})


def test_rafales_bornes_incluses() -> None:
    assert _bursts(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert _bursts(np.zeros(4)) == []
    assert _bursts(np.ones(3)) == [(0, 2)]


def test_confirmation_datee_de_facon_causale() -> None:
    """Moyennes sur 2 fenetres de [0,0,1,1,1,0] : 0 ; 0,5 ; 1 ; 1 ; 0,5.

    Le seuil (1,0 x 0,5) est franchi a partir de la moyenne d'indice 1, qui est
    datee de la fenetre 2. La deuxieme fenetre consecutive au-dela du seuil est
    donc la fenetre 3 : c'est la que la derive est confirmee avec K = 2.
    """
    alerts = np.array([0, 0, 1, 1, 1, 0])
    assert confirmation_point(alerts, _protocole(), target_rate=0.5) == 3


def test_confirmation_ne_devance_pas_la_premiere_alerte() -> None:
    """Piege d'OBS03 : une moyenne centree aurait date la confirmation avant la
    premiere alerte. La version causale ne peut pas confirmer avant elle."""
    for position in (10, 25, 40):
        alerts = np.zeros(60)
        alerts[position:] = 1
        t_r = confirmation_point(alerts, _protocole(rolling=8, confirm_k=3), target_rate=0.5)
        assert t_r is not None
        assert t_r >= position


def test_confirmation_exige_la_persistance() -> None:
    alerts = np.array([0, 0, 1, 1, 1, 0])
    assert confirmation_point(alerts, _protocole(confirm_k=5), target_rate=0.5) is None


def test_confirmation_impossible_sur_flux_trop_court() -> None:
    assert confirmation_point(np.ones(1), _protocole(), target_rate=0.5) is None


def test_instants_tries_et_nommes() -> None:
    alerts = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    instants = retrain_instants(alerts, 10, _protocole(), target_rate=0.5)
    assert instants == [("t20pc", 2), ("confirmation", 3), ("t40pc", 4)]


def test_instant_en_doublon_garde_la_confirmation() -> None:
    alerts = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    instants = retrain_instants(alerts, 10, _protocole(retrain_points=[0.3]), target_rate=0.5)
    assert instants == [("confirmation", 3)]


def test_instants_sans_confirmation() -> None:
    alerts = np.zeros(10)
    instants = retrain_instants(alerts, 10, _protocole(), target_rate=0.5)
    assert instants == [("t20pc", 2), ("t40pc", 4)]


def test_regles_equivalentes_fusionnees() -> None:
    """Fenetre [0, 20) : une rafale de 12 alertes (0 a 11) et une de 2 (15 et 16).

    - exclure toutes les alertes garde [12, 13, 14, 17, 18, 19] ;
    - exclure les rafales de moins de 10 ne retire que la courte, donc 18 fenetres ;
    - exclure les rafales de moins de 30 ou de 60 retire les deux : meme resultat
      qu'exclure toutes les alertes, les trois regles sont donc fusionnees ;
    - sur des scores 0 a 19, le 99e et le 95e percentile (18,81 et 18,05) retirent
      tous deux la seule fenetre 19, le 90e percentile (17,1) retire 18 et 19.
    """
    alerts = np.zeros(20, dtype=np.int64)
    alerts[0:12] = 1
    alerts[15:17] = 1
    regles = exclusion_rules(alerts, np.arange(20, dtype=float), 0, 20)

    assert list(regles) == [
        "R0_aucune_exclusion",
        "R1_exclut_toutes_alertes=R2_exclut_rafales_<30=R2_exclut_rafales_<60",
        "R2_exclut_rafales_<10",
        "R3_exclut_top_1pc=R3_exclut_top_5pc",
        "R3_exclut_top_10pc",
    ]
    assert regles["R0_aucune_exclusion"].tolist() == list(range(20))
    fusionnee = "R1_exclut_toutes_alertes=R2_exclut_rafales_<30=R2_exclut_rafales_<60"
    assert regles[fusionnee].tolist() == [12, 13, 14, 17, 18, 19]
    assert regles["R2_exclut_rafales_<10"].tolist() == [*range(15), 17, 18, 19]
    assert regles["R3_exclut_top_1pc=R3_exclut_top_5pc"].tolist() == list(range(19))
    assert regles["R3_exclut_top_10pc"].tolist() == list(range(18))


def test_regles_renvoient_des_indices_absolus() -> None:
    """Les indices servent a decouper `test.X` : ils doivent etre dans le repere du
    flux complet, pas dans celui de la fenetre [lo, hi)."""
    alerts = np.zeros(10, dtype=np.int64)
    alerts[7] = 1
    regles = exclusion_rules(alerts, np.arange(10, dtype=float), 5, 10)
    assert regles["R0_aucune_exclusion"].tolist() == [5, 6, 7, 8, 9]
    assert [kept.tolist() for nom, kept in regles.items() if "R1_" in nom] == [[5, 6, 8, 9]]


def _table_exclusion() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Decision 1 : ne rien exclure est optimal, tout exclure s'effondre.
            {"machine": "m1", "instant": "i1", "regle": "R0_aucune_exclusion", "aval_f1": 0.50},
            {
                "machine": "m1",
                "instant": "i1",
                "regle": "R1_exclut_toutes_alertes",
                "aval_f1": 0.10,
            },
            # Decision 2 : nom fusionne contenant R0, et R1 ne s'effondre pas.
            {
                "machine": "m1",
                "instant": "i2",
                "regle": "R0_aucune_exclusion=R3_exclut_top_1pc",
                "aval_f1": 0.40,
            },
            {
                "machine": "m1",
                "instant": "i2",
                "regle": "R1_exclut_toutes_alertes",
                "aval_f1": 0.44,
            },
        ]
    ).assign(part_gardee=0.5, aval_f1_std=0.01)


def test_verdict_exclusion_calcule_ecarts_et_effondrements() -> None:
    verdict = exclusion_verdict(_table_exclusion())
    assert verdict["n_decisions"] == 2
    assert verdict["n_machines"] == 1
    # Ecarts au meilleur : 0,50 - 0,50 = 0 et 0,44 - 0,40 = 0,04.
    assert verdict["ecart_median_au_meilleur"] == 0.02
    assert verdict["ecart_max_au_meilleur"] == 0.04
    assert verdict["n_fois_R0_optimale"] == 1
    assert verdict["n_R1_distincte"] == 2
    assert verdict["n_R1_effondrement"] == 1
    assert verdict["effondrements"][0]["ratio"] == 0.2


def test_verdict_exclusion_sur_table_vide() -> None:
    assert exclusion_verdict(pd.DataFrame()) == {}


def _ligne(machine: str, instant: str, regle: str, c1: float, f1: float) -> dict[str, object]:
    return {
        "machine": machine,
        "instant": instant,
        "regle": regle,
        "c1_ecart_au_cible": c1,
        "c2_ks_train_val": c1,
        "aval_f1": f1,
    }


def test_verdict_validation_mesure_le_sens_de_la_correlation() -> None:
    """Decision i1 : le critere classe parfaitement (Spearman -1), il choisit le
    meilleur. Decision i2 : il classe a l'envers (+1) et choisit le pire, 0,1 au
    lieu de 0,3, soit 66,7 % du F1 atteignable perdu."""
    table = pd.DataFrame(
        [
            _ligne("m1", "i1", "A", 0.01, 0.3),
            _ligne("m1", "i1", "B", 0.05, 0.2),
            _ligne("m1", "i1", "C", 0.10, 0.1),
            _ligne("m1", "i2", "A", 0.01, 0.1),
            _ligne("m1", "i2", "B", 0.05, 0.2),
            _ligne("m1", "i2", "C", 0.10, 0.3),
        ]
    )
    verdict = validation_verdict(table)
    assert verdict["n_decisions"] == 2
    assert [d["rho_c1"] for d in verdict["par_decision"]] == [-1.0, 1.0]
    assert verdict["rho_c1_median"] == 0.0
    assert verdict["n_rho_c1_positif"] == 1
    assert verdict["perte_f1_mediane_du_choix_c1"] == 0.3333
    assert verdict["perte_f1_max_du_choix_c1"] == 0.6667
    assert verdict["n_choix_est_le_pire"] == 1


def test_verdict_validation_ecarte_les_decisions_sans_classement() -> None:
    """Deux candidats distincts ne font pas un classement, et un critere constant
    n'en fait pas non plus : les deux decisions sont ecartees."""
    table = pd.DataFrame(
        [
            _ligne("m1", "deux_candidats", "A", 0.01, 0.3),
            _ligne("m1", "deux_candidats", "B", 0.05, 0.2),
            _ligne("m1", "critere_constant", "A", 0.02, 0.3),
            _ligne("m1", "critere_constant", "B", 0.02, 0.2),
            _ligne("m1", "critere_constant", "C", 0.02, 0.1),
        ]
    )
    assert validation_verdict(table)["n_decisions"] == 0


def test_verdict_reentrainement_compare_au_champion_fige() -> None:
    table = pd.DataFrame(
        [
            {
                "machine": "m1",
                "instant": "i1",
                "regle": "R0_aucune_exclusion",
                "aval_f1": 0.50,
                "champion_aval_f1": 0.25,
                "bat_le_champion": True,
            },
            {
                "machine": "m1",
                "instant": "i2",
                "regle": "R0_aucune_exclusion=R2_exclut_rafales_<10",
                "aval_f1": 0.10,
                "champion_aval_f1": 0.20,
                "bat_le_champion": False,
            },
            {
                "machine": "m1",
                "instant": "i1",
                "regle": "R1_exclut_toutes_alertes",
                "aval_f1": 0.05,
                "champion_aval_f1": 0.25,
                "bat_le_champion": False,
            },
        ]
    )
    verdict = retrain_verdict(table)
    # Seules les deux lignes sans exclusion comptent : rapports x2,0 et x0,5.
    assert verdict["n_decisions"] == 2
    assert verdict["n_bat_le_champion"] == 1
    assert verdict["gain_median"] == 1.25
    assert verdict["gain_min"] == 0.5
    assert verdict["gain_max"] == 2.0


def test_bornes_du_garde_fou_separent_ou_le_disent() -> None:
    """Les challengers sains alertent entre 0,8 % et 2,0 % sur la fenetre reservee.
    Ces bornes ecartent le challenger sature (75 %) et celui qui s'eteint (0,4 %),
    mais pas le degrade a 1,5 % : la separation n'est pas complete, et c'est le
    constat de D14 — le garde-fou n'ecarte que l'aberrant."""
    table = pd.DataFrame(
        {
            "sain": [True, True, False, False, False],
            "val_taux_alertes": [0.008, 0.020, 0.004, 0.015, 0.750],
        }
    )
    bornes = guardrail_bounds(table, {"isolation_forest": {"max_samples": 256}})
    assert bornes["n_sains"] == 2
    assert bornes["n_degrades"] == 3
    assert bornes["alert_rate_min_propose"] == 0.008
    assert bornes["alert_rate_max_propose"] == 0.020
    assert bornes["degrades_ecartes_par_borne_haute"] == 1
    assert bornes["degrades_ecartes_par_borne_basse"] == 1
    assert bornes["degrades_non_ecartes"] == 1
    assert bornes["separation_complete"] is False
    assert bornes["min_train_windows"] == 256


def test_bornes_du_garde_fou_sans_degrade() -> None:
    table = pd.DataFrame({"sain": [True, True], "val_taux_alertes": [0.009, 0.011]})
    bornes = guardrail_bounds(table, {"isolation_forest": {"max_samples": 256}})
    assert bornes["n_degrades"] == 0
    assert "separation_complete" not in bornes
