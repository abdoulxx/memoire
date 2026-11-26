"""Tests des notifications Telegram (M5, phase D).

Ce que ces tests protegent, et pourquoi rien d'autre ne le ferait :

- **Aucune fuite vers un service tiers.** C'est le point le plus grave de ce
  module : une notification part chez Telegram, et ce qui est parti ne revient
  pas. Les tests passent des `details` contenant une URL, un chemin, un nom
  d'hote et un message d'erreur de collecte, et exigent qu'aucun de ces
  fragments n'apparaisse dans le texte redige. La regle testee est celle du
  §4.3, et elle est tenue par une liste de cles autorisees par evenement : un
  test verifie aussi que toute cle non declaree est ecartee, meme ajoutee plus
  tard dans la boucle.
- **Le regroupement du §9.** Au plus un message par minute et par type, mais le
  **premier** d'un type part immediatement : une alerte retardee d'une minute
  n'est plus une alerte en temps reel. Les tests pilotent une horloge a la main
  — un test qui attendrait reellement soixante secondes ne serait jamais lance.
- **Un echec d'envoi n'interrompt pas la detection.** Le transport leve, et le
  notificateur doit compter l'echec et continuer. Sans ce test, une coupure de
  reseau mobile arreterait un service de detection.
- **Le jeton ne se recopie nulle part.** `repr` des reglages, et le secret
  n'apparait que dans l'URL construite a l'envoi.

Aucun test n'ouvre de connexion : le transport est injecte. La seule facon de
verifier l'appel reel reste `mdrift notify-test` sur le serveur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.notify.base import Notificateur
from src.notify.telegram import (
    CLES_PUBLIABLES,
    ErreurTelegram,
    NotificateurTelegram,
    ReglagesTelegram,
    construire_notificateur,
    decrire,
)
from src.serving.evenements import EVENEMENTS
from src.serving.loop import JournalEvenements

ENV = {"TELEGRAM_BOT_TOKEN": "123:faux-jeton-de-test", "TELEGRAM_CHAT_ID": "-100123"}

CONFIG: dict[str, Any] = {
    "telegram": {
        "actif": True,
        "evenements": ["alert", "drift_confirmed", "promoted", "service_stop", "collect_gap"],
        "regroupement_s": 60,
        "silencieux": ["service_stop"],
        "timeout_s": 5,
    }
}

# `k_fenetres: 1` : un episode s'ouvre a la premiere alerte, comme avant le
# correctif par episode. Utilise par les tests qui portent sur autre chose que
# le regroupement lui-meme (robustesse, forme de l'URL, silence sonore).
CONFIG_EPISODE_K1: dict[str, Any] = {
    "telegram": {**CONFIG["telegram"], "episode": {"k_fenetres": 1}},
}


class Horloge:
    """Horloge pilotee par le test : aucun `sleep` reel n'a sa place ici."""

    def __init__(self, depart: float = 1_000.0) -> None:
        self.t = depart

    def __call__(self) -> float:
        return self.t

    def avancer(self, secondes: float) -> None:
        self.t += secondes


def _notificateur(
    config: dict[str, Any] | None = None,
) -> tuple[NotificateurTelegram, list[dict[str, Any]], Horloge]:
    envoyes: list[dict[str, Any]] = []
    horloge = Horloge()
    reglages = ReglagesTelegram.depuis_config(config or CONFIG, env=ENV)
    return (
        NotificateurTelegram(reglages, horloge=horloge, transport=envoyes.append),
        envoyes,
        horloge,
    )


def _entree(event: str, **details: Any) -> dict[str, Any]:
    return {
        "ts": "2026-11-12T10:00:00Z",
        "run_id": "r1",
        "strategy": "proposed",
        "event": event,
        "details": details,
    }


# --------------------------------------------------------------------- reglages


def test_la_configuration_livree_est_lisible_et_inactive_par_defaut() -> None:
    """Un depot clone ne doit pas appeler une API externe au premier `serve`."""
    reglages = ReglagesTelegram.depuis_config(load_config("notify.yaml"), env={})
    assert reglages.actif is False
    assert construire_notificateur(load_config("notify.yaml"), horloge=Horloge(), env={}) is None


def test_les_evenements_de_la_configuration_livree_sont_du_vocabulaire_du_paragraphe_9() -> None:
    reglages = ReglagesTelegram.depuis_config(load_config("notify.yaml"), env={})
    assert set(reglages.evenements) <= set(EVENEMENTS)
    # `drift_signal` et `window_skipped` sont volontairement exclus : ils peuvent
    # survenir a chaque fenetre et rendraient le telephone inutilisable.
    assert "drift_signal" not in reglages.evenements
    assert "window_skipped" not in reglages.evenements


def test_un_evenement_invente_est_refuse() -> None:
    with pytest.raises(ErreurTelegram, match="vocabulaire"):
        ReglagesTelegram.depuis_config(
            {"telegram": {"actif": True, "evenements": ["tout_va_bien"]}}, env=ENV
        )


def test_une_section_absente_est_refusee() -> None:
    with pytest.raises(ErreurTelegram, match="telegram"):
        ReglagesTelegram.depuis_config({}, env=ENV)


def test_les_notifications_actives_sans_secret_dans_l_environnement_sont_refusees() -> None:
    """Plutot echouer au demarrage que croire alerter pendant une demonstration."""
    with pytest.raises(ErreurTelegram, match="TELEGRAM_BOT_TOKEN"):
        ReglagesTelegram.depuis_config(CONFIG, env={})
    with pytest.raises(ErreurTelegram, match="TELEGRAM_CHAT_ID"):
        ReglagesTelegram.depuis_config(CONFIG, env={"TELEGRAM_BOT_TOKEN": "x"})


def test_une_liste_d_evenements_vide_est_refusee_quand_les_notifications_sont_actives() -> None:
    with pytest.raises(ErreurTelegram, match="vide"):
        ReglagesTelegram.depuis_config({"telegram": {"actif": True, "evenements": []}}, env=ENV)


def test_le_jeton_n_apparait_pas_dans_la_representation_des_reglages() -> None:
    reglages = ReglagesTelegram.depuis_config(CONFIG, env=ENV)
    texte = repr(reglages)
    assert "faux-jeton-de-test" not in texte
    assert "-100123" not in texte


def test_la_base_de_l_api_est_surchargeable_par_l_environnement() -> None:
    reglages = ReglagesTelegram.depuis_config(
        CONFIG, env={**ENV, "TELEGRAM_API_BASE": "http://127.0.0.1:8081"}
    )
    assert reglages.api == "http://127.0.0.1:8081"


# ---------------------------------------------------------------- non-fuite §4.3


def test_aucune_cle_non_declaree_n_entre_dans_un_message() -> None:
    """Le point le plus grave du module : ce qui part chez Telegram ne revient pas."""
    texte = decrire(
        _entree(
            "collect_gap",
            points_manques=12,
            message="HTTPConnectionPool(host='prom.exemple.interne', port=9090)",
            url="http://10.0.0.4:9090/api/v1/query_range",
            chemin="/srv/results/online/r1/events.jsonl",
        )
    )
    assert "points_manques 12" in texte
    for fragment in ("exemple.interne", "10.0.0.4", "/srv/", "http://", "HTTPConnectionPool"):
        assert fragment not in texte, fragment


def test_une_cle_ajoutee_plus_tard_dans_la_boucle_n_est_pas_publiee() -> None:
    texte = decrire(_entree("alert", score=0.9, seuil=0.3, hote="serveur-de-sidy"))
    assert "score 0.9" in texte
    assert "serveur-de-sidy" not in texte


def test_chaque_evenement_du_vocabulaire_sait_se_decrire() -> None:
    """Un evenement sans libelle ni cles publiables sortirait en nom technique."""
    for event in EVENEMENTS:
        assert event in CLES_PUBLIABLES, event
        texte = decrire(_entree(event))
        assert texte and texte != event, event


def test_un_evenement_sans_detail_exploitable_sort_quand_meme() -> None:
    assert decrire(_entree("promoted")) == "Nouveau modele en service"
    assert decrire({"event": "alert"}) == "Anomalie detectee"


# ------------------------------------------------------------------ regroupement


def test_le_premier_evenement_d_un_type_part_immediatement() -> None:
    """Une alerte isolee retardee d'une minute n'est plus une alerte en temps reel.

    `collect_gap` ici, et non `alert` : depuis le correctif par episode (D51),
    `alert` suit une regle differente, testee plus bas dans sa propre section.
    """
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("collect_gap", points_manques=3))
    assert len(envoyes) == 1
    assert "Trou de collecte" in envoyes[0]["text"]
    assert envoyes[0]["chat_id"] == "-100123"


def test_une_rafale_ne_produit_qu_un_message_par_fenetre_et_par_type() -> None:
    notificateur, envoyes, horloge = _notificateur()
    for i in range(40):
        notificateur.observer_evenement(_entree("collect_gap", points_manques=i))
    assert len(envoyes) == 1  # seul le premier est parti
    assert notificateur.en_attente == 39

    horloge.avancer(60.0)
    notificateur.rincer()
    assert len(envoyes) == 2
    # Le compte vient en premier : c'est lui qui decide d'agir, pas le detail.
    assert "39 en 60 s" in envoyes[1]["text"]
    assert "Derniere" in envoyes[1]["text"]
    assert notificateur.en_attente == 0


def test_la_fenetre_de_regroupement_est_propre_a_chaque_type() -> None:
    """Une rafale d'un type ne doit pas retarder l'annonce d'un autre."""
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("collect_gap", points_manques=1))
    notificateur.observer_evenement(_entree("collect_gap", points_manques=2))
    notificateur.observer_evenement(_entree("promoted", version=4, threshold=0.28))
    assert len(envoyes) == 2
    assert "Nouveau modele en service" in envoyes[-1]["text"]
    assert "version 4" in envoyes[-1]["text"]


def test_un_rincage_avant_la_fin_de_la_fenetre_n_envoie_rien() -> None:
    notificateur, envoyes, horloge = _notificateur()
    notificateur.observer_evenement(_entree("collect_gap", points_manques=1))
    notificateur.observer_evenement(_entree("collect_gap", points_manques=2))
    horloge.avancer(59.0)
    assert notificateur.rincer() == 0
    assert len(envoyes) == 1
    horloge.avancer(1.0)
    assert notificateur.rincer() == 1


def test_le_rincage_force_ne_laisse_rien_en_attente() -> None:
    """Le resume de session ne doit pas rester bloque par le regroupement."""
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("collect_gap", points_manques=1))
    notificateur.observer_evenement(_entree("collect_gap", points_manques=2))
    notificateur.rincer(forcer=True)
    assert notificateur.en_attente == 0
    assert len(envoyes) == 2


def test_un_evenement_hors_liste_n_est_pas_notifie() -> None:
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("drift_signal", share=0.4))
    assert envoyes == []
    assert notificateur.ignores == 1


def test_les_evenements_silencieux_partent_sans_notification_sonore() -> None:
    notificateur, envoyes, _ = _notificateur(CONFIG_EPISODE_K1)
    notificateur.observer_evenement(_entree("service_stop", n_windows=120, n_alerts=3))
    notificateur.observer_evenement(_entree("alert", score=0.9, threshold=0.3))
    silencieux = {
        message["text"].split(" —")[0]: message["disable_notification"] for message in envoyes
    }
    assert silencieux["Service de detection arrete"] is True
    assert silencieux["Episode d'anomalie ouvert (1 fenetres consecutives)"] is False


# --------------------------------------------------------------- robustesse


def test_un_echec_d_envoi_est_compte_et_n_interrompt_rien() -> None:
    def casse(_charge: dict[str, Any]) -> None:
        raise OSError("reseau indisponible")

    reglages = ReglagesTelegram.depuis_config(CONFIG_EPISODE_K1, env=ENV)
    notificateur = NotificateurTelegram(reglages, horloge=Horloge(), transport=casse)
    notificateur.observer_evenement(_entree("alert", score=0.9, threshold=0.3))
    assert notificateur.echecs == 1
    assert notificateur.envois == 0
    # Et le service continue : l'evenement suivant est traite normalement.
    notificateur.observer_evenement(_entree("promoted", version=2))
    assert notificateur.echecs == 2


@pytest.mark.parametrize(
    "entree",
    [{}, {"event": None}, {"event": "alert", "details": None}, {"event": "alert", "details": "x"}],
)
def test_une_entree_malformee_ne_leve_jamais(entree: dict[str, Any]) -> None:
    notificateur, _, _ = _notificateur()
    notificateur.observer_evenement(entree)  # ne doit pas lever


def test_le_notificateur_satisfait_l_interface_attendue_par_la_boucle() -> None:
    notificateur, _, _ = _notificateur()
    assert isinstance(notificateur, Notificateur)


# ----------------------------------------------------------- episodes d'alerte
#
# Correctif demande apres D51 : le champion en service alertait sur des dizaines
# de fenetres consecutives (faute de reentrainement), une notification par
# fenetre. Un episode regroupe une rafale d'alertes consecutives en un seul fil
# de messages : ouverture, rappels espaces, cloture.
#
# La boucle appelle `rincer()` une fois par fenetre (tick), avant de savoir si
# cette fenetre va alerter. C'est donc la encore `rincer()`, et seulement lui,
# qui peut detecter qu'une fenetre n'a pas alerte — avec un tick de retard,
# puisqu'il n'existe aucun evenement explicite de "non-alerte" dans le journal.


def _fenetre(
    notificateur: NotificateurTelegram, *, alerte: bool, fin_ts: int, score: float = 0.9
) -> None:
    """Simule un tick de la boucle : `rincer()` d'abord, puis l'alerte si elle a lieu."""
    notificateur.rincer()
    if alerte:
        notificateur.observer_evenement(_entree("alert", fin_ts=fin_ts, score=score, threshold=0.3))


def test_un_episode_ne_s_ouvre_qu_apres_k_fenetres_consecutives() -> None:
    notificateur, envoyes, _ = _notificateur()  # k_fenetres par defaut = 3
    _fenetre(notificateur, alerte=True, fin_ts=100)
    _fenetre(notificateur, alerte=True, fin_ts=160)
    assert envoyes == []  # 2 sur 3 : pas encore d'episode
    _fenetre(notificateur, alerte=True, fin_ts=220)
    assert len(envoyes) == 1
    assert "Episode d'anomalie ouvert (3 fenetres consecutives)" in envoyes[0]["text"]


def test_une_fenetre_sans_alerte_remet_le_compteur_a_zero_avant_l_ouverture() -> None:
    """Une rupture avant le seuil d'ouverture oblige a recommencer la serie."""
    notificateur, envoyes, _ = _notificateur()
    _fenetre(notificateur, alerte=True, fin_ts=100)
    _fenetre(notificateur, alerte=True, fin_ts=160)
    _fenetre(notificateur, alerte=False, fin_ts=220)  # rupture (pas encore visible)
    # La rupture du tick precedent n'est reconnue qu'au prochain `rincer()` :
    # cette 4e fenetre remet le compteur a zero avant de s'ajouter elle-meme.
    _fenetre(notificateur, alerte=True, fin_ts=280)
    _fenetre(notificateur, alerte=True, fin_ts=340)
    assert envoyes == []  # une seule fenetre consecutive depuis la rupture
    _fenetre(notificateur, alerte=True, fin_ts=400)
    assert len(envoyes) == 1  # 3 fenetres consecutives depuis la rupture


def test_un_episode_ouvert_se_cloture_une_fenetre_apres_la_derniere_alerte() -> None:
    notificateur, envoyes, _ = _notificateur()
    for fin_ts, score in ((100, 0.6), (160, 0.9), (220, 0.7)):
        _fenetre(notificateur, alerte=True, fin_ts=fin_ts, score=score)
    assert len(envoyes) == 1  # ouverture

    notificateur.rincer()  # tick sans alerte : la rupture n'est pas encore visible
    assert len(envoyes) == 1
    notificateur.rincer()  # tick suivant : la rupture du tick precedent est revelee
    assert len(envoyes) == 2
    texte = envoyes[1]["text"]
    assert "Episode d'anomalie termine" in texte
    assert "duree 2" in texte  # (220 - 100) s = 2 min
    assert "score max 0.9" in texte


def test_le_rincage_force_cloture_un_episode_ouvert() -> None:
    """Arret du service : la cloture ne doit pas attendre un tick de plus."""
    notificateur, envoyes, _ = _notificateur(CONFIG_EPISODE_K1)
    notificateur.observer_evenement(_entree("alert", fin_ts=100, score=0.5, threshold=0.3))
    assert len(envoyes) == 1
    notificateur.rincer(forcer=True)
    assert len(envoyes) == 2
    assert "Episode d'anomalie termine" in envoyes[1]["text"]


def test_les_rappels_sont_espaces_d_au_moins_le_delai_configure() -> None:
    config = {
        "telegram": {**CONFIG["telegram"], "episode": {"k_fenetres": 1, "rappel_s": 120}},
    }
    notificateur, envoyes, horloge = _notificateur(config)
    notificateur.observer_evenement(_entree("alert", fin_ts=100, score=0.5, threshold=0.3))
    assert len(envoyes) == 1  # ouverture

    horloge.avancer(60)
    notificateur.observer_evenement(_entree("alert", fin_ts=160, score=0.5, threshold=0.3))
    assert len(envoyes) == 1  # 60 s < 120 s : pas de rappel

    horloge.avancer(70)  # 130 s depuis l'ouverture
    notificateur.observer_evenement(_entree("alert", fin_ts=220, score=0.5, threshold=0.3))
    assert len(envoyes) == 2
    assert "Episode d'anomalie en cours" in envoyes[1]["text"]

    horloge.avancer(130)  # 130 s depuis le dernier rappel
    notificateur.observer_evenement(_entree("alert", fin_ts=280, score=0.5, threshold=0.3))
    assert len(envoyes) == 3


def test_le_message_d_episode_contient_les_champs_attendus_sans_cle_imprevue() -> None:
    """Seules les cles lues explicitement par `_rediger_episode` peuvent apparaitre."""
    notificateur, envoyes, _ = _notificateur(CONFIG_EPISODE_K1)
    notificateur.observer_evenement(_entree("service_start", version=7))
    notificateur.observer_evenement(_entree("drift_confirmed", streak=40, share=0.3))
    notificateur.observer_evenement(
        _entree(
            "alert",
            fin_ts=0,
            score=0.812345,
            threshold=0.5,
            top_features=[
                {"feature": "latence_p95_mean", "z": 4.1},
                {"feature": "requetes_par_s_max", "z": -3.2},
                {"feature": "disque_libre_pct_mean", "z": 0.1},
            ],
            interpolee=False,
            hote="serveur-de-sidy",  # cle non autorisee : ne doit jamais apparaitre
        )
    )
    texte = envoyes[-1]["text"]
    assert "1970-01-01T00:00:00+00:00" in texte
    assert "score 0.8123 (seuil 0.5)" in texte
    assert "latence_p95_mean 4.1" in texte
    assert "requetes_par_s_max -3.2" in texte
    assert "modele v7" in texte
    assert "derive confirmee, reentrainement en attente" in texte
    assert "serveur-de-sidy" not in texte
    assert "http" not in texte  # pas de GRAFANA_BASE_URL : pas de lien


def test_l_etat_de_derive_disparait_apres_une_promotion() -> None:
    notificateur, envoyes, _ = _notificateur(CONFIG_EPISODE_K1)
    notificateur.observer_evenement(_entree("drift_confirmed", streak=40, share=0.3))
    notificateur.observer_evenement(_entree("promoted", version=9))
    notificateur.observer_evenement(_entree("alert", fin_ts=0, score=0.6, threshold=0.3))
    texte = envoyes[-1]["text"]
    assert "modele v9" in texte
    assert "derive confirmee" not in texte


def test_le_lien_grafana_n_apparait_que_si_base_url_et_uid_sont_configures() -> None:
    config = {
        "telegram": {
            **CONFIG["telegram"],
            "episode": {"k_fenetres": 1},
            "grafana_dashboard_uid": "memoire-pipeline",
        },
    }
    # Sans GRAFANA_BASE_URL dans l'environnement : pas de lien, pas d'erreur.
    sans_url, envoyes_sans, _ = _notificateur(config)
    sans_url.observer_evenement(_entree("alert", fin_ts=3600, score=0.5, threshold=0.3))
    assert "http" not in envoyes_sans[-1]["text"]

    # L'hote utilise ici est une fausse adresse locale de test, jamais celle du
    # serveur reel (§4.3) : elle ne vient que de l'environnement du test.
    reglages = ReglagesTelegram.depuis_config(
        config, env={**ENV, "GRAFANA_BASE_URL": "http://127.0.0.1:3000"}
    )
    envoyes_avec: list[dict[str, Any]] = []
    avec_url = NotificateurTelegram(reglages, horloge=Horloge(), transport=envoyes_avec.append)
    avec_url.observer_evenement(_entree("alert", fin_ts=3600, score=0.5, threshold=0.3))
    assert (
        "http://127.0.0.1:3000/d/memoire-pipeline?from=3600000&to=3600000"
        in envoyes_avec[-1]["text"]
    )


# ------------------------------------------------------- abonnement au journal


def test_les_messages_viennent_du_journal_et_de_rien_d_autre(tmp_path: Path) -> None:
    """Une notification sans ligne de journal serait indefendable, et l'inverse aussi."""
    notificateur, envoyes, _ = _notificateur()
    journal = JournalEvenements(
        chemin=tmp_path / "events.jsonl",
        run_id="r1",
        strategie="proposed",
    ).ouvrir()
    journal.abonnes.append(notificateur.observer_evenement)

    # k_fenetres par defaut = 3 : il faut trois alertes consecutives pour que
    # l'episode s'ouvre (voir la section « episodes d'alerte » plus bas).
    for fin_ts in (100, 160, 220):
        journal.ecrire("alert", fin_ts=fin_ts, score=0.91, threshold=0.3)
    journal.ecrire("drift_confirmed", streak=30, share=0.27)
    journal.ecrire("drift_signal", share=0.27)  # journalise, non notifie

    lignes = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 5
    assert len(envoyes) == 2  # ouverture de l'episode + derive confirmee
    assert "Episode d'anomalie ouvert" in envoyes[0]["text"]
    assert "Derive confirmee" in envoyes[1]["text"]
    assert "streak 30" in envoyes[1]["text"]


def test_l_url_d_envoi_porte_le_jeton_et_la_methode_sendmessage() -> None:
    """Verifie la forme de l'appel sans l'emettre : le transport est remplace."""
    captures: list[str] = []
    reglages = ReglagesTelegram.depuis_config(CONFIG_EPISODE_K1, env=ENV)
    notificateur = NotificateurTelegram(reglages, horloge=Horloge())

    def faux_urlopen(requete: Any, timeout: float = 0.0) -> Any:  # noqa: ANN401
        captures.append(requete.full_url)
        raise OSError("pas de reseau dans les tests")

    import urllib.request

    original = urllib.request.urlopen
    urllib.request.urlopen = faux_urlopen  # type: ignore[assignment]
    try:
        notificateur.observer_evenement(_entree("alert", score=0.9, threshold=0.3))
    finally:
        urllib.request.urlopen = original  # type: ignore[assignment]

    assert captures == ["https://api.telegram.org/bot123:faux-jeton-de-test/sendMessage"]
    assert notificateur.echecs == 1
