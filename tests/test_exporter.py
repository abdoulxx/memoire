"""Tests de l'exporter Prometheus du service de detection (M5, phase C).

Ce que ces tests protegent, et pourquoi rien d'autre ne le ferait :

- **Une metrique n'existe pas sans sa ligne de journal.** L'exporter s'abonne a
  `JournalEvenements.ecrire` au lieu de compter en parallele. Un test verifie
  que le compteur ne bouge **que** par le journal, et qu'un evenement refuse par
  le vocabulaire du §9 ne laisse aucune trace dans les metriques. Sans cela, les
  captures Grafana du chapitre 5 et le journal remis au jury pourraient se
  contredire, et rien dans le code ne le signalerait.
- **Toutes les series valent zero des le demarrage.** Dans Grafana, une serie
  absente et une serie a zero se dessinent pareil : un panneau « derives
  confirmees » vide signifierait indifferemment « aucune derive » ou « service
  tombe ». Un test rend donc les metriques avant tout evenement et exige que les
  douze etiquettes `event` et toutes les jauges soient deja la.
- **L'observation ne casse jamais le service.** Une detection en cours ne doit
  pas s'interrompre parce qu'un detail de journal a un type inattendu. Les tests
  passent des entrees deliberement malformees et exigent le silence.
- **La fraicheur se mesure sur la fenetre, pas sur le scrape.**
  `mdrift_last_window_timestamp_seconds` porte l'instant de fin de fenetre : le
  tableau de bord en fait `time() - gauge`, qui est le retard reel de la
  detection. Un test fixe un instant de fenetre connu et verifie que c'est bien
  lui qui sort.

Les metriques sont lues en re-analysant le texte d'exposition (`rendre()`), pas
en interrogeant les attributs de l'objet : c'est exactement ce que Prometheus
recevra, noms de series et suffixes compris.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from prometheus_client.parser import text_string_to_metric_families

from src.serving.evenements import EVENEMENTS, EVENEMENTS_EN_LIGNE, EVENEMENTS_REJEU
from src.serving.exporter import (
    VERSION_INCONNUE,
    ErreurExporteur,
    ExporteurDetection,
    ReglagesExporteur,
)
from src.serving.loop import ErreurBoucle, JournalEvenements, ResultatFenetre

CONFIG_MINIMALE: dict[str, Any] = {"exporter": {"actif": True, "hote": "127.0.0.1", "port": 8001}}


def _series(exporteur: ExporteurDetection) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Les series telles que Prometheus les verra, indexees par nom et etiquettes."""
    texte = exporteur.rendre().decode("utf-8")
    sorties: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for famille in text_string_to_metric_families(texte):
        for echantillon in famille.samples:
            cle = (echantillon.name, tuple(sorted(echantillon.labels.items())))
            sorties[cle] = echantillon.value
    return sorties


def _valeur(exporteur: ExporteurDetection, nom: str, **etiquettes: str) -> float:
    cle = (nom, tuple(sorted(etiquettes.items())))
    series = _series(exporteur)
    assert cle in series, (cle, sorted(n for n, _ in series))
    return series[cle]


def _fenetre(**remplacements: Any) -> ResultatFenetre:
    champs: dict[str, Any] = {
        "fin_ts": 1_700_000_000,
        "position": 12,
        "score": 0.42,
        "seuil": 0.30,
        "alerte": True,
        "reentrainement": False,
        "promue": False,
        "interpolee": False,
        "ms": 18.0,
    }
    champs.update(remplacements)
    return ResultatFenetre(**champs)


def _journal(tmp_path: Path, exporteur: ExporteurDetection) -> JournalEvenements:
    journal = JournalEvenements(
        chemin=tmp_path / "evenements.jsonl",
        run_id="essai",
        strategie="proposed",
        garder_en_memoire=True,
    ).ouvrir()
    journal.abonnes.append(exporteur.observer_evenement)
    return journal


# --------------------------------------------------------------------- reglages


def test_les_reglages_viennent_de_la_configuration() -> None:
    reglages = ReglagesExporteur.depuis_config(CONFIG_MINIMALE, env={})
    assert (reglages.actif, reglages.hote, reglages.port) == (True, "127.0.0.1", 8001)


def test_une_configuration_sans_section_exporter_est_refusee() -> None:
    with pytest.raises(ErreurExporteur, match="exporter"):
        ReglagesExporteur.depuis_config({}, env={})


def test_l_environnement_a_priorite_sur_le_fichier_suivi() -> None:
    """Le conteneur doit ecouter sur 0.0.0.0 sans modifier un fichier du depot."""
    reglages = ReglagesExporteur.depuis_config(
        CONFIG_MINIMALE,
        env={"DETECTION_EXPORTER_HOST": "0.0.0.0", "DETECTION_EXPORTER_PORT": "9123"},
    )
    assert (reglages.hote, reglages.port) == ("0.0.0.0", 9123)


@pytest.mark.parametrize("port", [0, 70000, -1])
def test_un_port_hors_bornes_est_refuse(port: int) -> None:
    with pytest.raises(ErreurExporteur, match="port"):
        ReglagesExporteur.depuis_config({"exporter": {"port": port}}, env={})


def test_une_adresse_vide_est_refusee() -> None:
    with pytest.raises(ErreurExporteur, match="hote"):
        ReglagesExporteur.depuis_config({"exporter": {"hote": "", "port": 8001}}, env={})


# ------------------------------------------------------------- etat au demarrage


def test_toutes_les_series_valent_zero_avant_le_premier_evenement() -> None:
    exporteur = ExporteurDetection(strategie="proposed", budget_ms=60_000.0)
    series = _series(exporteur)

    for event in EVENEMENTS:
        cle = ("mdrift_events_total", (("event", event),))
        assert cle in series, event
        assert series[cle] == 0.0, event

    for nom in (
        "mdrift_service_up",
        "mdrift_window_score",
        "mdrift_window_threshold",
        "mdrift_window_alert",
        "mdrift_last_window_timestamp_seconds",
        "mdrift_window_duration_ms_mean",
        "mdrift_window_duration_ms_p95",
        "mdrift_window_duration_ms_median",
        "mdrift_window_duration_ms_max",
        "mdrift_drift_share",
        "mdrift_drift_streak",
        "mdrift_drift_features",
        "mdrift_drift_alert_rate_signal",
        "mdrift_model_version",
        "mdrift_last_promotion_timestamp_seconds",
        "mdrift_windows_total",
        "mdrift_retrain_seconds_total",
    ):
        assert (nom, ()) in series, nom
        assert series[(nom, ())] == 0.0, nom

    # Le budget, lui, est connu des la construction : c'est l'intervalle de
    # collecte, et la ligne de reference des panneaux I7 / T1.
    assert series[("mdrift_window_budget_ms", ())] == 60_000.0


def test_les_etiquettes_event_sont_exactement_le_vocabulaire_du_paragraphe_9() -> None:
    exporteur = ExporteurDetection()
    etiquettes = {
        echantillon[1][0][1]
        for echantillon in _series(exporteur)
        if echantillon[0] == "mdrift_events_total"
    }
    assert etiquettes == set(EVENEMENTS)
    assert set(EVENEMENTS_REJEU) <= etiquettes
    assert set(EVENEMENTS_EN_LIGNE) <= etiquettes


def test_deux_services_du_meme_processus_ne_se_marchent_pas_dessus() -> None:
    """Chaque exporter a son registre : sinon le second lever une collision."""
    premier = ExporteurDetection(strategie="proposed")
    second = ExporteurDetection(strategie="periodic")
    premier.observer_fenetre(_fenetre(score=0.9), {})
    assert _valeur(premier, "mdrift_windows_total") == 1.0
    assert _valeur(second, "mdrift_windows_total") == 0.0


# ------------------------------------------------------- abonnement au journal


def test_aucune_metrique_sans_sa_ligne_de_journal(tmp_path: Path) -> None:
    exporteur = ExporteurDetection()
    journal = _journal(tmp_path, exporteur)

    journal.ecrire("alert", score=0.9, seuil=0.3)
    journal.ecrire("alert", score=0.8, seuil=0.3)
    journal.ecrire("drift_confirmed", streak=30)

    assert _valeur(exporteur, "mdrift_events_total", event="alert") == 2.0
    assert _valeur(exporteur, "mdrift_events_total", event="drift_confirmed") == 1.0
    lignes = (tmp_path / "evenements.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 3


def test_un_evenement_hors_vocabulaire_ne_laisse_aucune_trace(tmp_path: Path) -> None:
    """Le journal refuse d'abord, donc l'exporter ne peut pas inventer une serie."""
    exporteur = ExporteurDetection()
    journal = _journal(tmp_path, exporteur)

    with pytest.raises(ErreurBoucle):
        journal.ecrire("tout_va_bien")

    series = _series(exporteur)
    assert ("mdrift_events_total", (("event", "tout_va_bien"),)) not in series
    assert sum(v for (nom, _), v in series.items() if nom == "mdrift_events_total") == 0.0


def test_un_abonne_en_erreur_n_empeche_pas_la_ligne_de_journal(tmp_path: Path) -> None:
    exporteur = ExporteurDetection()
    journal = _journal(tmp_path, exporteur)

    def casse(_entree: dict[str, Any]) -> None:
        raise RuntimeError("notification indisponible")

    journal.abonnes.insert(0, casse)
    journal.ecrire("promoted", version=4)

    assert (tmp_path / "evenements.jsonl").read_text(encoding="utf-8").count("promoted") == 1
    assert _valeur(exporteur, "mdrift_events_total", event="promoted") == 1.0


def test_le_demarrage_et_l_arret_du_service_pilotent_la_disponibilite() -> None:
    exporteur = ExporteurDetection(strategie="proposed")
    exporteur.observer_evenement(
        {
            "run_id": "r1",
            "strategy": "proposed",
            "event": "service_start",
            "details": {"version": 7, "threshold": 0.25, "features_empreinte": "abc123"},
        }
    )
    assert _valeur(exporteur, "mdrift_service_up") == 1.0
    assert _valeur(exporteur, "mdrift_model_version") == 7.0
    assert _valeur(exporteur, "mdrift_window_threshold") == 0.25
    assert (
        _valeur(
            exporteur,
            "mdrift_build_info",
            strategy="proposed",
            run_id="r1",
            features_empreinte="abc123",
        )
        == 1.0
    )

    exporteur.observer_evenement({"event": "service_stop", "details": {}})
    assert _valeur(exporteur, "mdrift_service_up") == 0.0


def test_une_promotion_met_a_jour_la_version_et_son_instant() -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_evenement(
        {"event": "promoted", "details": {"version": 12, "threshold": 0.4, "fin_ts": 1_700_000_600}}
    )
    assert _valeur(exporteur, "mdrift_model_version") == 12.0
    assert _valeur(exporteur, "mdrift_window_threshold") == 0.4
    assert _valeur(exporteur, "mdrift_last_promotion_timestamp_seconds") == 1_700_000_600.0


def test_un_rejet_ne_change_pas_le_modele_en_service() -> None:
    """Le garde-fou du §6 : une version rejetee ne doit pas apparaitre en service."""
    exporteur = ExporteurDetection()
    exporteur.observer_evenement({"event": "promoted", "details": {"version": 3}})
    exporteur.observer_evenement({"event": "rejected", "details": {"version": 4, "raison": "taux"}})
    assert _valeur(exporteur, "mdrift_model_version") == 3.0
    assert _valeur(exporteur, "mdrift_events_total", event="rejected") == 1.0


def test_le_temps_de_reentrainement_s_accumule() -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_evenement({"event": "retrain_end", "details": {"seconds": 2.5}})
    exporteur.observer_evenement({"event": "retrain_end", "details": {"seconds": 1.5}})
    assert _valeur(exporteur, "mdrift_retrain_seconds_total") == 4.0


@pytest.mark.parametrize("details", [{}, {"seconds": None}, {"seconds": "lent"}, {"seconds": -3}])
def test_un_temps_de_reentrainement_inexploitable_est_ignore(details: dict[str, Any]) -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_evenement({"event": "retrain_end", "details": details})
    assert _valeur(exporteur, "mdrift_retrain_seconds_total") == 0.0
    # La ligne est comptee quand meme : elle a bien ete journalisee.
    assert _valeur(exporteur, "mdrift_events_total", event="retrain_end") == 1.0


@pytest.mark.parametrize("valeur", [None, "v3", float("nan"), {}])
def test_une_version_illisible_retombe_sur_la_version_inconnue(valeur: Any) -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_evenement({"event": "promoted", "details": {"version": valeur}})
    rendu = _valeur(exporteur, "mdrift_model_version")
    assert rendu == float(VERSION_INCONNUE) or rendu != rendu  # nan : int() leve, pas float()


@pytest.mark.parametrize(
    "entree",
    [
        {},
        {"event": None},
        {"event": "alert", "details": None},
        {"event": "service_start", "details": "pas un dictionnaire"},
        {"event": "promoted", "details": {"fin_ts": object()}},
    ],
)
def test_l_observation_d_une_entree_malformee_ne_leve_jamais(entree: dict[str, Any]) -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_evenement(entree)  # ne doit pas lever
    assert _valeur(exporteur, "mdrift_service_up") in (0.0, 1.0)


# -------------------------------------------------------------- fenetres scorees


def test_une_fenetre_scoree_renseigne_score_seuil_alerte_et_fraicheur() -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_fenetre(_fenetre(fin_ts=1_700_000_060, score=0.77, seuil=0.31), {})

    assert _valeur(exporteur, "mdrift_windows_total") == 1.0
    assert _valeur(exporteur, "mdrift_window_score") == 0.77
    assert _valeur(exporteur, "mdrift_window_threshold") == 0.31
    assert _valeur(exporteur, "mdrift_window_alert") == 1.0
    # L'instant de la fenetre, pas celui du scrape : c'est ce qui rend
    # `time() - gauge` lisible comme le retard reel de la detection.
    assert _valeur(exporteur, "mdrift_last_window_timestamp_seconds") == 1_700_000_060.0


def test_l_etat_de_la_surveillance_de_derive_est_expose() -> None:
    exporteur = ExporteurDetection()
    exporteur.observer_fenetre(
        _fenetre(alerte=False),
        {"share": 0.21, "streak": 7, "n_features_drifting": 32, "alert_rate_signal": True},
    )
    assert _valeur(exporteur, "mdrift_drift_share") == 0.21
    assert _valeur(exporteur, "mdrift_drift_streak") == 7.0
    assert _valeur(exporteur, "mdrift_drift_features") == 32.0
    assert _valeur(exporteur, "mdrift_drift_alert_rate_signal") == 1.0
    assert _valeur(exporteur, "mdrift_window_alert") == 0.0


def test_une_strategie_sans_surveillance_laisse_les_jauges_de_derive_a_zero() -> None:
    """B0 et B1 ne surveillent pas la derive : zero est la valeur exacte pour elles."""
    exporteur = ExporteurDetection(strategie="static")
    exporteur.observer_fenetre(_fenetre(), {})
    for nom in ("mdrift_drift_share", "mdrift_drift_streak", "mdrift_drift_features"):
        assert _valeur(exporteur, nom) == 0.0


def test_les_mesures_de_temps_sont_exposees_avec_leur_budget() -> None:
    exporteur = ExporteurDetection(budget_ms=60_000.0)
    exporteur.observer_mesures(moyenne=18.4, p95=41.2, mediane=15.0, maximum=52.7)
    assert _valeur(exporteur, "mdrift_window_duration_ms_mean") == 18.4
    assert _valeur(exporteur, "mdrift_window_duration_ms_p95") == 41.2
    assert _valeur(exporteur, "mdrift_window_duration_ms_median") == 15.0
    assert _valeur(exporteur, "mdrift_window_duration_ms_max") == 52.7
    assert _valeur(exporteur, "mdrift_window_budget_ms") == 60_000.0


def test_le_rendu_ne_contient_ni_valeur_de_feature_ni_nom_de_metrique_serveur() -> None:
    """§4.3 : rien de l'infrastructure ne doit fuir par /metrics.

    Les etiquettes sont bornees a `event`, `strategy`, `version` : une etiquette
    par feature ou par fenetre ferait exploser la cardinalite, premiere cause de
    saturation du disque de Prometheus, et exposerait la forme des donnees.
    """
    exporteur = ExporteurDetection(strategie="proposed")
    exporteur.observer_fenetre(_fenetre(), {"share": 0.4, "streak": 3})
    texte = exporteur.rendre().decode("utf-8")

    etiquettes = set()
    for famille in text_string_to_metric_families(texte):
        for echantillon in famille.samples:
            etiquettes |= set(echantillon.labels)
    assert etiquettes <= {"event", "strategy", "run_id", "features_empreinte", "version"}
    for interdit in ("node_", "container_", "cpu_usage", "/srv", "http://"):
        assert interdit not in texte, interdit
