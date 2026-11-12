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
    """Une alerte retardee d'une minute n'est plus une alerte en temps reel."""
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    assert len(envoyes) == 1
    assert "Anomalie detectee" in envoyes[0]["text"]
    assert envoyes[0]["chat_id"] == "-100123"


def test_une_rafale_ne_produit_qu_un_message_par_fenetre_et_par_type() -> None:
    notificateur, envoyes, horloge = _notificateur()
    for i in range(40):
        notificateur.observer_evenement(_entree("alert", score=0.5 + i / 100, seuil=0.3))
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
    """Une rafale d'alertes ne doit pas retarder l'annonce d'une promotion."""
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    notificateur.observer_evenement(_entree("alert", score=0.95, seuil=0.3))
    notificateur.observer_evenement(_entree("promoted", version=4, threshold=0.28))
    assert len(envoyes) == 2
    assert "Nouveau modele en service" in envoyes[-1]["text"]
    assert "version 4" in envoyes[-1]["text"]


def test_un_rincage_avant_la_fin_de_la_fenetre_n_envoie_rien() -> None:
    notificateur, envoyes, horloge = _notificateur()
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    horloge.avancer(59.0)
    assert notificateur.rincer() == 0
    assert len(envoyes) == 1
    horloge.avancer(1.0)
    assert notificateur.rincer() == 1


def test_le_rincage_force_ne_laisse_rien_en_attente() -> None:
    """Le resume de session ne doit pas rester bloque par le regroupement."""
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    notificateur.observer_evenement(_entree("alert", score=0.8, seuil=0.3))
    notificateur.rincer(forcer=True)
    assert notificateur.en_attente == 0
    assert len(envoyes) == 2


def test_un_evenement_hors_liste_n_est_pas_notifie() -> None:
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("drift_signal", share=0.4))
    assert envoyes == []
    assert notificateur.ignores == 1


def test_les_evenements_silencieux_partent_sans_notification_sonore() -> None:
    notificateur, envoyes, _ = _notificateur()
    notificateur.observer_evenement(_entree("service_stop", n_windows=120, n_alerts=3))
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    silencieux = {
        message["text"].split(" —")[0]: message["disable_notification"] for message in envoyes
    }
    assert silencieux["Service de detection arrete"] is True
    assert silencieux["Anomalie detectee"] is False


# --------------------------------------------------------------- robustesse


def test_un_echec_d_envoi_est_compte_et_n_interrompt_rien() -> None:
    def casse(_charge: dict[str, Any]) -> None:
        raise OSError("reseau indisponible")

    reglages = ReglagesTelegram.depuis_config(CONFIG, env=ENV)
    notificateur = NotificateurTelegram(reglages, horloge=Horloge(), transport=casse)
    notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
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

    journal.ecrire("alert", score=0.91, seuil=0.3)
    journal.ecrire("drift_confirmed", streak=30, share=0.27)
    journal.ecrire("drift_signal", share=0.27)  # journalise, non notifie

    lignes = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 3
    assert len(envoyes) == 2
    assert "Derive confirmee" in envoyes[1]["text"]
    assert "streak 30" in envoyes[1]["text"]


def test_l_url_d_envoi_porte_le_jeton_et_la_methode_sendmessage() -> None:
    """Verifie la forme de l'appel sans l'emettre : le transport est remplace."""
    captures: list[str] = []
    reglages = ReglagesTelegram.depuis_config(CONFIG, env=ENV)
    notificateur = NotificateurTelegram(reglages, horloge=Horloge())

    def faux_urlopen(requete: Any, timeout: float = 0.0) -> Any:  # noqa: ANN401
        captures.append(requete.full_url)
        raise OSError("pas de reseau dans les tests")

    import urllib.request

    original = urllib.request.urlopen
    urllib.request.urlopen = faux_urlopen  # type: ignore[assignment]
    try:
        notificateur.observer_evenement(_entree("alert", score=0.9, seuil=0.3))
    finally:
        urllib.request.urlopen = original  # type: ignore[assignment]

    assert captures == ["https://api.telegram.org/bot123:faux-jeton-de-test/sendMessage"]
    assert notificateur.echecs == 1
