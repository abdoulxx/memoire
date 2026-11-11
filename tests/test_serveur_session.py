"""Sessions enregistrees sur le serveur (livrable M4, phase D).

Ces tests portent sur l'alignement temporel entre les metriques et les
etiquettes, et c'est le point le plus fragile de toute la source `serveur`. Les
metriques vivent sur une grille reguliere ; les anomalies sont horodatees a la
seconde par l'injecteur, qui ne sait rien de cette grille. Un decalage d'un seul
pas fait basculer une fenetre positive en negative, et les chiffres du chapitre 6
sur donnees reelles seraient faux sans qu'aucun test ne s'en apercoive.

Les trois autres proprietes verifiees ici sont des garde-fous du meme genre : une
anomalie plus courte que le pas est signalee et non ignoree, un trou long fait
echouer le chargement au lieu d'etre comble, et une anomalie dans la part
d'entrainement est refusee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from datasets.serveur.loader import SessionServeur, charger_session, instant
from datasets.serveur.recorder import (
    COLONNE_INSTANT,
    ErreurSession,
    chemin_donnees,
    enregistrer,
    instant_depuis_texte,
    verifier_cibles,
)
from datasets.serveur.settings import MARQUE_RATE, ReglagesServeur
from infra.injector.labels import Anomalie, Derive, ecrire_anomalie, ecrire_derive
from src.config import load_config
from src.evaluation.streams import NO_DRIFT, SCENARIO_REEL, build_server_stream
from src.serving.prometheus import ClientPrometheus, SerieTemporelle

CONFIG_REELLE = "datasets/serveur.yaml"

# Base multiple du pas : les instants de la grille en sont des multiples.
T0 = 1_793_000_040
PAS = 60


def _config(**surcharges: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "serveur-test",
        "prometheus": {"url": "http://prom:9090", "step_seconds": PAS, "rate_interval": "1m"},
        "metriques": [
            {"nom": "cpu", "expr": f"rate(cpu{MARQUE_RATE})", "obligatoire": True},
            {"nom": "latence", "expr": "latence", "obligatoire": False},
        ],
        "trous": {"max_points_consecutifs": 3, "max_part_interpolee": 0.2},
        "decoupage": {"part_entrainement": 0.4, "refuser_anomalie_dans_entrainement": True},
        "etiquetage": {"bornes_incluses": True, "signaler_anomalies_sans_point": True},
    }
    for cle, valeur in surcharges.items():
        if isinstance(valeur, dict) and isinstance(base.get(cle), dict):
            base[cle] = {**base[cle], **valeur}
        else:
            base[cle] = valeur
    return base


def _reglages(dossier: Path, **surcharges: Any) -> ReglagesServeur:
    config = _config(**surcharges)
    config["chemins"] = {"sessions": "sessions", "etiquettes": "labels"}
    return ReglagesServeur.depuis_config(config, dossier)


def _ecrire_session(
    reglages: ReglagesServeur,
    run_id: str,
    n: int = 20,
    cpu: np.ndarray | None = None,
    instants: np.ndarray | None = None,
) -> np.ndarray:
    """Ecrit un Parquet de session, comme `session-record` le ferait."""
    if instants is None:
        instants = np.arange(T0, T0 + n * PAS, PAS, dtype=np.int64)
    n = len(instants)
    reglages.dossier_sessions.mkdir(parents=True, exist_ok=True)
    tableau = {
        COLONNE_INSTANT: instants,
        "cpu": np.arange(n, dtype=float) if cpu is None else cpu,
        "latence": np.full(n, 0.1),
    }
    pd.DataFrame(tableau).to_parquet(chemin_donnees(reglages, run_id), index=False)
    return instants


def _anomalie(reglages: ReglagesServeur, run_id: str, debut: float, fin: float) -> None:
    ecrire_anomalie(
        reglages.dossier_etiquettes,
        run_id,
        Anomalie(type="pic_cpu", intensite=2, debut_ts=debut, fin_ts=fin),
    )


# --------------------------------------------------------------------------- #
# Reglages
# --------------------------------------------------------------------------- #


def test_la_configuration_reelle_se_lit_et_decrit_les_metriques_du_tableau_de_bord() -> None:
    reglages = ReglagesServeur.depuis_config(load_config(CONFIG_REELLE))
    assert reglages.pas_s == 60
    # Le pas doit etre celui du SMD : le meme code de fenetrage sert aux deux
    # sources, et une fenetre de 10 points doit valoir 10 minutes des deux cotes.
    assert reglages.pas_s == load_config("datasets/smd.yaml").get("sampling_interval_seconds", 60)
    assert "cpu_hote" in reglages.noms
    assert len(reglages.noms) == len(set(reglages.noms))
    for metrique in reglages.metriques:
        assert MARQUE_RATE not in metrique.expression(reglages.rate_interval)


def test_la_fenetre_de_rate_vient_de_la_configuration_et_non_de_l_expression() -> None:
    reglages = _reglages(Path("/inexistant"), prometheus={"rate_interval": "5m"})
    assert reglages.metriques[0].expression(reglages.rate_interval) == "rate(cpu[5m])"


def test_une_part_d_entrainement_hors_bornes_est_refusee() -> None:
    with pytest.raises(ValueError, match="part_entrainement"):
        ReglagesServeur.depuis_config(_config(decoupage={"part_entrainement": 1.0}))


def test_une_metrique_declaree_deux_fois_est_refusee() -> None:
    config = _config()
    config["metriques"].append({"nom": "cpu", "expr": "autre"})
    with pytest.raises(ValueError, match="deux fois"):
        ReglagesServeur.depuis_config(config)


def test_l_url_de_l_environnement_a_la_priorite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMETHEUS_URL", "http://autre:9091")
    assert ReglagesServeur.depuis_config(_config()).url == "http://autre:9091"


# --------------------------------------------------------------------------- #
# Enregistrement
# --------------------------------------------------------------------------- #


class ClientFactice(ClientPrometheus):
    """Rend des series connues, sans reseau."""

    def __init__(self, series: dict[str, np.ndarray], etats: dict[str, str] | None = None) -> None:
        super().__init__(url="http://prom")
        self.series = series
        self.etats = etats or {"prometheus": "up", "app": "up"}
        self.expressions: list[str] = []

    def interroger_plage(
        self, expression: str, debut: float, fin: float, pas: int
    ) -> SerieTemporelle:
        from src.serving.prometheus import grille

        self.expressions.append(expression)
        instants = grille(debut, fin, pas)
        for nom, valeurs in self.series.items():
            if nom in expression:
                return SerieTemporelle(instants=instants, valeurs=valeurs.astype(float))
        return SerieTemporelle(instants=instants, valeurs=np.full(len(instants), np.nan))

    def cibles(self) -> dict[str, str]:
        return self.etats


def test_une_session_enregistree_garde_ses_trous_tels_quels(tmp_path: Path) -> None:
    # Le Parquet est la mesure, pas son interpretation : interpoler a
    # l'enregistrement rendrait impossible de savoir apres coup ce qui a
    # vraiment ete mesure.
    reglages = _reglages(tmp_path)
    cpu = np.array([1.0, np.nan, 3.0, 4.0])
    client = ClientFactice({"cpu": cpu, "latence": np.full(4, 0.2)})

    session = enregistrer(client, reglages, T0, T0 + 3 * PAS, "s1")

    assert session.n_points == 4
    assert session.parts_manquantes["cpu"] == pytest.approx(0.25)
    relu = pd.read_parquet(session.donnees)
    assert np.isnan(relu["cpu"][1])
    assert list(relu[COLONNE_INSTANT]) == [T0, T0 + PAS, T0 + 2 * PAS, T0 + 3 * PAS]


def test_les_metadonnees_ne_portent_ni_url_ni_nom_d_hote(tmp_path: Path) -> None:
    # §4.3 : ce fichier est celui qu'on recopie dans le memoire.
    import json

    reglages = _reglages(tmp_path)
    client = ClientFactice({"cpu": np.ones(4), "latence": np.ones(4)})
    session = enregistrer(client, reglages, T0, T0 + 3 * PAS, "s1")

    texte = session.metadonnees.read_text(encoding="utf-8")
    assert "prom" not in texte
    assert "9090" not in texte
    metadonnees = json.loads(texte)
    assert metadonnees["pas_s"] == PAS
    # Les expressions envoyees sont recopiees : changer une expression change
    # les valeurs, donc la session doit porter celles qui l'ont produite.
    assert metadonnees["metriques"][0]["expr"] == "rate(cpu[1m])"


def test_une_metrique_obligatoire_vide_fait_echouer_l_enregistrement(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    client = ClientFactice({"latence": np.ones(4)})  # `cpu` ne repond pas
    with pytest.raises(ErreurSession, match="cpu"):
        enregistrer(client, reglages, T0, T0 + 3 * PAS, "s1")
    assert not chemin_donnees(reglages, "s1").exists()


def test_une_cible_tombee_empeche_de_lancer_une_session(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path, prometheus={"cibles_requises": ["prometheus", "node"]})
    client = ClientFactice({}, etats={"prometheus": "up", "node": "down"})
    with pytest.raises(ErreurSession, match="node"):
        verifier_cibles(client, reglages)


@pytest.mark.parametrize(
    ("texte", "attendu"),
    [("-2h", T0 - 7200), ("-30m", T0 - 1800), ("-90s", T0 - 90), (str(T0), T0)],
)
def test_une_plage_relative_se_lit(texte: str, attendu: float) -> None:
    assert instant_depuis_texte(texte, maintenant=T0) == pytest.approx(attendu)


def test_un_instant_illisible_dit_les_formes_acceptees() -> None:
    with pytest.raises(ErreurSession, match="Formes acceptees"):
        instant_depuis_texte("hier matin")


# --------------------------------------------------------------------------- #
# Alignement temporel : la propriete centrale
# --------------------------------------------------------------------------- #


def test_les_bornes_de_l_anomalie_sont_incluses(tmp_path: Path) -> None:
    # A `end_ts`, l'injection tournait encore : le point de `end_ts` est anormal.
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    # Points 10 a 12 du flux complet, soit les instants T0+600 a T0+720.
    _anomalie(reglages, "s1", float(instants[10]), float(instants[12]))

    session = charger_session("s1", reglages)

    n_train = 8  # 40 % de 20
    attendus = np.zeros(len(instants), dtype=int)
    attendus[10:13] = 1
    assert list(session.test_label) == list(attendus[n_train:])
    assert len(session.events) == 1
    assert session.events[0].length == 3


def test_une_anomalie_qui_commence_entre_deux_points_marque_le_point_suivant(
    tmp_path: Path,
) -> None:
    # L'injecteur horodate a la seconde, la grille est a la minute. Une anomalie
    # de T0+605 a T0+700 couvre le point T0+660 et lui seul : arrondir vers le
    # bas marquerait un point ou rien ne se passait encore.
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    _anomalie(reglages, "s1", float(instants[10]) + 5, float(instants[11]) + 40)

    session = charger_session("s1", reglages)

    marques = np.flatnonzero(session.test_label) + 8
    assert list(marques) == [11]


def test_une_anomalie_plus_courte_que_le_pas_est_signalee_et_non_ignoree(
    tmp_path: Path,
) -> None:
    # Elle serait invisible dans les donnees tout en comptant comme un evenement
    # a detecter, ce qui ferait baisser le rappel par evenement de toutes les
    # strategies sans qu'aucune puisse y faire quoi que ce soit.
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    _anomalie(reglages, "s1", float(instants[10]) + 5, float(instants[10]) + 40)

    session = charger_session("s1", reglages)

    assert session.test_label.sum() == 0
    assert len(session.anomalies_sans_point) == 1
    assert session.anomalies_sans_point[0]["type"] == "pic_cpu"


def test_une_anomalie_dont_la_fin_precede_le_debut_est_refusee(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    _anomalie(reglages, "s1", float(instants[12]), float(instants[10]))
    with pytest.raises(ErreurSession, match="anterieure au debut"):
        charger_session("s1", reglages)


def test_deux_anomalies_qui_se_touchent_forment_un_seul_evenement(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    _anomalie(reglages, "s1", float(instants[10]), float(instants[11]))
    _anomalie(reglages, "s1", float(instants[12]), float(instants[13]))

    session = charger_session("s1", reglages)

    assert len(session.events) == 1
    assert session.events[0].length == 4


def test_une_grille_irreguliere_est_refusee(tmp_path: Path) -> None:
    # Un Parquet dont la grille n'est pas au pas de la configuration n'a pas ete
    # ecrit par `session-record`, ou le pas a change depuis : les etiquettes ne
    # tomberaient pas ou il faut.
    reglages = _reglages(tmp_path)
    instants = np.array([T0, T0 + PAS, T0 + 3 * PAS, T0 + 4 * PAS], dtype=np.int64)
    _ecrire_session(reglages, "s1", instants=instants)
    with pytest.raises(ErreurSession, match="pas reguliere"):
        charger_session("s1", reglages)


def test_des_instants_non_multiples_du_pas_sont_refuses(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    instants = np.arange(T0 + 7, T0 + 7 + 20 * PAS, PAS, dtype=np.int64)
    _ecrire_session(reglages, "s1", instants=instants)
    with pytest.raises(ErreurSession, match="multiples"):
        charger_session("s1", reglages)


# --------------------------------------------------------------------------- #
# Trous
# --------------------------------------------------------------------------- #


def test_un_trou_court_est_interpole_et_la_part_est_rapportee(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    cpu = np.arange(20, dtype=float)
    cpu[12] = np.nan
    _ecrire_session(reglages, "s1", n=20, cpu=cpu)

    session = charger_session("s1", reglages)

    assert session.test[12 - 8, 0] == pytest.approx(12.0)
    assert session.parts_interpolees["cpu"] == pytest.approx(1 / 20)
    assert not np.isnan(session.test).any()


def test_un_trou_long_fait_echouer_le_chargement(tmp_path: Path) -> None:
    # Interpoler vingt minutes inventerait un regime entier.
    reglages = _reglages(tmp_path)
    cpu = np.arange(20, dtype=float)
    cpu[10:16] = np.nan
    _ecrire_session(reglages, "s1", n=20, cpu=cpu)
    with pytest.raises(ErreurSession, match="6 points consecutifs"):
        charger_session("s1", reglages)


def test_une_collecte_globalement_trop_trouee_est_refusee(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path, trous={"max_part_interpolee": 0.05})
    cpu = np.arange(20, dtype=float)
    cpu[[11, 13, 15]] = np.nan  # 15 % de la session, trous de 1 point
    _ecrire_session(reglages, "s1", n=20, cpu=cpu)
    with pytest.raises(ErreurSession, match="15.0% de points manquants"):
        charger_session("s1", reglages)


def test_un_trou_en_bord_de_session_est_comble_sans_laisser_de_nan(tmp_path: Path) -> None:
    # L'interpolation lineaire ne peut pas encadrer un trou de bord : sans
    # `limit_direction="both"`, le `NaN` survivrait et traverserait la
    # normalisation.
    reglages = _reglages(tmp_path)
    cpu = np.arange(20, dtype=float)
    cpu[0] = np.nan
    cpu[-1] = np.nan
    _ecrire_session(reglages, "s1", n=20, cpu=cpu)

    session = charger_session("s1", reglages)

    assert not np.isnan(session.train).any()
    assert not np.isnan(session.test).any()


# --------------------------------------------------------------------------- #
# Decoupage et derive
# --------------------------------------------------------------------------- #


def test_une_anomalie_dans_la_part_d_entrainement_est_refusee(tmp_path: Path) -> None:
    # Un modele initial entraine sur une anomalie ne la signalera jamais, et la
    # comparaison des strategies qui en decoule ne veut plus rien dire.
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    _anomalie(reglages, "s1", float(instants[3]), float(instants[4]))
    with pytest.raises(ErreurSession, match="part d'entrainement"):
        charger_session("s1", reglages)


def test_une_derive_dans_la_part_d_entrainement_est_refusee(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    ecrire_derive(
        reglages.dossier_etiquettes,
        "s1",
        Derive(type="deploiement", description="v2", instant_ts=float(instants[2])),
    )
    with pytest.raises(ErreurSession, match="part d'entrainement"):
        charger_session("s1", reglages)


def test_la_contamination_peut_etre_autorisee_explicitement(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path, decoupage={"refuser_anomalie_dans_entrainement": False})
    instants = _ecrire_session(reglages, "s1", n=20)
    _anomalie(reglages, "s1", float(instants[3]), float(instants[4]))
    session = charger_session("s1", reglages)
    assert session.test_label.sum() == 0


def test_la_derive_est_situee_dans_le_flux_evalue_et_non_dans_la_session(
    tmp_path: Path,
) -> None:
    # `drift_index` est un indice de `test`, comme pour le SMD : le harnais de
    # rejeu ne voit que le flux evalue.
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    ecrire_derive(
        reglages.dossier_etiquettes,
        "s1",
        Derive(type="trafic", description="montee", instant_ts=float(instants[12])),
    )

    session = charger_session("s1", reglages)

    assert session.drift_index == 12 - 8


def test_une_derive_entre_deux_points_est_rattachee_au_point_suivant(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=20)
    ecrire_derive(
        reglages.dossier_etiquettes,
        "s1",
        Derive(type="deploiement", description="v2", instant_ts=float(instants[12]) + 20),
    )
    session = charger_session("s1", reglages)
    assert session.drift_index == 13 - 8


def test_une_session_absente_dit_comment_en_obtenir_une(tmp_path: Path) -> None:
    with pytest.raises(ErreurSession, match="session-record"):
        charger_session("jamais_enregistree", _reglages(tmp_path))


def test_une_session_enregistree_avec_d_autres_metriques_est_refusee(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    _ecrire_session(reglages, "s1", n=20)
    autres = ReglagesServeur.depuis_config(
        {
            **_config(),
            "metriques": [{"nom": "memoire", "expr": "memoire", "obligatoire": True}],
            "chemins": {"sessions": "sessions", "etiquettes": "labels"},
        },
        tmp_path,
    )
    with pytest.raises(ErreurSession, match="memoire"):
        charger_session("s1", autres)


def test_une_session_trop_courte_est_refusee(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    _ecrire_session(reglages, "s1", n=3)
    with pytest.raises(ErreurSession, match="trop courte"):
        charger_session("s1", reglages)


def test_un_horodatage_d_etiquette_illisible_est_rapporte() -> None:
    with pytest.raises(ErreurSession, match="RFC 3339"):
        instant("28 octobre 2026")


# --------------------------------------------------------------------------- #
# Flux de rejeu
# --------------------------------------------------------------------------- #


def _session_chargee(tmp_path: Path, avec_derive: bool) -> SessionServeur:
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=60)
    _anomalie(reglages, "s1", float(instants[40]), float(instants[44]))
    if avec_derive:
        ecrire_derive(
            reglages.dossier_etiquettes,
            "s1",
            Derive(type="deploiement", description="v2", instant_ts=float(instants[30])),
        )
    return charger_session("s1", reglages)


def test_le_flux_du_serveur_a_la_meme_forme_que_celui_du_smd(tmp_path: Path) -> None:
    session = _session_chargee(tmp_path, avec_derive=True)
    features = load_config("features.yaml")

    flux = build_server_stream(session, features)

    assert flux.source == "serveur"
    assert flux.machine == "s1"
    assert flux.scenario == SCENARIO_REEL
    # Deux metriques x quatre agregats : le fenetrage est celui du SMD.
    assert flux.X.shape[1] == 2 * len(features["aggregates"])
    assert flux.train_X.shape[1] == flux.X.shape[1]
    assert len(flux) == len(flux.y) == len(flux.end_index)
    assert flux.sampling_interval_seconds == PAS
    # Une derive reelle ne se replie pas : elle n'a pas de fin.
    assert flux.drift_end_index is None
    assert flux.drift_metadata is not None
    assert flux.drift_metadata["origine"] == "reelle"


def test_la_fenetre_de_derive_se_deduit_du_point_de_derive(tmp_path: Path) -> None:
    session = _session_chargee(tmp_path, avec_derive=True)
    flux = build_server_stream(session, load_config("features.yaml"))
    fenetre = flux.drift_window()
    assert fenetre is not None
    # La derive tombe dans la fenetre qui se termine a cet instant ou juste apres.
    assert flux.end_index[fenetre] >= session.drift_index


def test_une_session_sans_derive_porte_le_scenario_none(tmp_path: Path) -> None:
    session = _session_chargee(tmp_path, avec_derive=False)
    flux = build_server_stream(session, load_config("features.yaml"))
    assert flux.scenario == NO_DRIFT
    assert flux.drift_index is None
    assert flux.drift_metadata is None


def test_une_fenetre_est_positive_des_qu_elle_contient_un_point_anormal(
    tmp_path: Path,
) -> None:
    # Meme regle que pour le SMD (§4.4), sinon les deux sources ne sont pas
    # comparables.
    session = _session_chargee(tmp_path, avec_derive=False)
    flux = build_server_stream(session, load_config("features.yaml"))
    taille = flux.window_size
    for position, fin in enumerate(flux.end_index):
        debut = fin - taille + 1
        attendu = int(session.test_label[debut : fin + 1].max())
        assert int(flux.y[position]) == attendu
