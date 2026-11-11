"""Signature du contrat de features (livrable M5, phase A).

La propriete que ces tests protegent n'a pas d'autre garde-fou : un modele
charge depuis le registre et score sur des colonnes qui ont change de sens ne
leve aucune erreur. Il rend des scores plausibles, calcules sur autre chose que
ce qu'il a appris, et le service alerte au hasard sans que rien ne le signale.
"""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.features.signature import SignatureFeatures, signature_depuis_config
from src.features.windowing import make_windows

METRIQUES = ["cpu", "memoire", "latence_p95"]


def _config(**surcharges: object) -> dict:
    base = {"window_size": 10, "stride": 1, "aggregates": ["last", "mean", "std", "max"]}
    base.update(surcharges)
    return base


def _signature(**surcharges: object) -> SignatureFeatures:
    base: dict = {
        "feature_names": ("a_mean", "b_mean"),
        "window_size": 10,
        "stride": 1,
        "sampling_interval_seconds": 60,
    }
    base.update(surcharges)
    return SignatureFeatures(**base)  # type: ignore[arg-type]


def test_la_signature_decrit_exactement_ce_que_make_windows_produit() -> None:
    # L'invariant central : si les deux divergent, la signature enregistree
    # decrit un contrat que le code ne respecte pas, et tous les controles
    # construits dessus deviennent decoratifs.
    config = _config()
    serie = np.arange(60 * len(METRIQUES), dtype=np.float64).reshape(60, len(METRIQUES))
    fenetres = make_windows(serie, config, metric_names=METRIQUES)

    signature = signature_depuis_config(METRIQUES, config, 60)

    assert signature.feature_names == tuple(fenetres.feature_names)
    assert signature.n_features == fenetres.n_features


def test_la_signature_suit_la_configuration_reelle_du_projet() -> None:
    config = load_config("features.yaml")
    serie = np.zeros((int(config["window_size"]) + 5, len(METRIQUES)))
    fenetres = make_windows(serie, config, metric_names=METRIQUES)
    signature = signature_depuis_config(METRIQUES, config, 60)
    assert signature.feature_names == tuple(fenetres.feature_names)


def test_l_empreinte_est_stable_et_ne_depend_pas_du_hachage_de_python() -> None:
    # `hash()` est randomise a chaque processus : une empreinte calculee avec lui
    # ne pourrait pas etre comparee a celle enregistree hier dans le registre.
    attendue = _signature().empreinte
    assert len(attendue) == 16
    assert attendue == _signature().empreinte
    assert all(c in "0123456789abcdef" for c in attendue)


def test_deux_signatures_identiques_ont_la_meme_empreinte() -> None:
    config = _config()
    assert (
        signature_depuis_config(METRIQUES, config, 60).empreinte
        == signature_depuis_config(METRIQUES, config, 60).empreinte
    )


def test_changer_l_ordre_des_agregats_change_l_empreinte() -> None:
    # Meme nombre de colonnes, meme noms a l'ensemble pres, mais la colonne j
    # n'a plus le meme sens : c'est exactement le cas dangereux.
    une = signature_depuis_config(METRIQUES, _config(aggregates=["mean", "max"]), 60)
    autre = signature_depuis_config(METRIQUES, _config(aggregates=["max", "mean"]), 60)
    assert une.n_features == autre.n_features
    assert une.empreinte != autre.empreinte


def test_changer_la_taille_de_fenetre_change_l_empreinte() -> None:
    une = signature_depuis_config(METRIQUES, _config(window_size=10), 60)
    autre = signature_depuis_config(METRIQUES, _config(window_size=20), 60)
    assert une.feature_names == autre.feature_names
    assert une.empreinte != autre.empreinte


def test_changer_le_pas_d_echantillonnage_change_l_empreinte() -> None:
    # Une fenetre de 10 points ne couvre pas la meme duree au pas de 60 s et au
    # pas de 15 s : les agregats glissants ne decrivent pas la meme chose.
    une = signature_depuis_config(METRIQUES, _config(), 60)
    autre = signature_depuis_config(METRIQUES, _config(), 15)
    assert une.empreinte != autre.empreinte


def test_l_ecart_est_vide_quand_les_deux_concordent() -> None:
    assert _signature().ecart(_signature()) == ""


def test_l_ecart_nomme_le_nombre_de_features_et_la_cause_probable() -> None:
    message = _signature(feature_names=("a_mean", "b_mean", "c_mean")).ecart(_signature())
    assert "2 features attendues par le modele" in message
    assert "3 produites" in message
    assert "metrique" in message


def test_l_ecart_nomme_le_fenetrage_quand_c_est_lui_qui_change() -> None:
    message = _signature(window_size=20).ecart(_signature())
    assert "window_size=10" in message and "window_size=20" in message


def test_l_ecart_nomme_le_pas_quand_c_est_lui_qui_change() -> None:
    message = _signature(sampling_interval_seconds=15).ecart(_signature())
    assert "60s a l'entrainement" in message
    assert "15s maintenant" in message


def test_l_ecart_nomme_les_colonnes_renommees_a_nombre_egal() -> None:
    message = _signature(feature_names=("a_mean", "z_mean")).ecart(_signature())
    assert "colonne 1" in message
    assert "b_mean" in message and "z_mean" in message


def test_les_etiquettes_enregistrees_sont_toutes_des_chaines() -> None:
    # Contrainte de MLflow : une valeur d'etiquette est une chaine. Un int passe
    # silencieusement a l'ecriture et revient en chaine a la lecture, ce qui
    # ferait echouer la comparaison la ou on l'attend le moins.
    tags = _signature().as_tags()
    assert all(isinstance(valeur, str) for valeur in tags.values())
    assert tags["features_n"] == "2"
    assert tags["features_empreinte"] == _signature().empreinte


def test_une_signature_vide_n_est_pas_une_signature() -> None:
    vide = _signature(feature_names=())
    assert vide.n_features == 0
    assert vide.empreinte != _signature().empreinte
