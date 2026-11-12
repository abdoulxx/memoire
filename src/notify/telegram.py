"""Notifications Telegram construites depuis le journal d'evenements (§9).

Quatre choix structurent ce module, et aucun n'est evident :

1. **Les messages sont construits depuis le journal, pas a cote.** Le
   notificateur s'abonne a `JournalEvenements.ecrire`, comme l'exporter
   Prometheus. Une notification sans ligne de journal correspondante — ou
   l'inverse — serait indefendable : c'est le journal qui est remis avec le
   memoire, et les captures Telegram du chapitre 5 doivent pouvoir s'y retrouver
   ligne par ligne.

2. **Seules des cles autorisees entrent dans un message.** Un formateur qui
   deverserait `details` en entier finirait par pousser un chemin, un nom d'hote
   ou une URL vers un serveur tiers des qu'un evenement porterait un message
   d'erreur de collecte. Le §4.3 l'interdit sans condition, et une fuite vers
   l'API Telegram n'est pas rattrapable apres coup. Chaque type d'evenement
   declare donc la liste des cles qu'il publie ; tout le reste est ecarte, y
   compris une cle ajoutee plus tard dans la boucle.

3. **Regroupement : au plus un message par minute et par type** (§9). Le premier
   evenement d'un type part immediatement — une alerte retardee d'une minute
   n'est plus une alerte en temps reel — et les suivants sont agreges en un seul
   message de synthese. Sans cela, une saturation produirait une alerte par
   fenetre : le telephone devient inutilisable, et l'API Telegram elle-meme
   limite a une vingtaine de messages par minute et par conversation. Avec une
   dizaine de types au plus, cette regle garde une marge confortable.

4. **Aucun `parse_mode`.** Le texte part en clair. En Markdown ou en HTML, un
   caractere `_`, `*` ou `<` dans un identifiant de modele ou un message
   d'erreur fait rejeter l'appel par l'API avec un 400 : la notification serait
   perdue au moment ou elle importe le plus.

Le jeton et l'identifiant de conversation viennent **uniquement** de
l'environnement (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) : ils ne sont jamais
lus dans un fichier suivi par git, et `ReglagesTelegram` masque le jeton dans
son `repr` pour qu'une trace de pile ne le recopie pas.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.serving.evenements import EVENEMENTS

API_PAR_DEFAUT = "https://api.telegram.org"

# Ce que chaque evenement a le droit de publier. Volontairement restrictif :
# ajouter une cle ici est un geste conscient, l'oublier ne fuite rien.
CLES_PUBLIABLES: dict[str, tuple[str, ...]] = {
    "alert": ("score", "seuil"),
    "drift_signal": ("share", "streak"),
    "drift_confirmed": ("share", "streak", "n_features_drifting"),
    "retrain_start": ("n_windows",),
    "retrain_end": ("seconds", "n_windows"),
    "promoted": ("version", "threshold", "alert_rate"),
    "rejected": ("raison", "alert_rate"),
    "service_start": ("version", "threshold"),
    "service_stop": ("n_windows", "n_alerts", "n_retrains", "n_promoted", "n_rejected"),
    "window_skipped": ("raison",),
    "collect_gap": ("points_manques",),
    "registry_error": ("operation",),
}

LIBELLES: dict[str, str] = {
    "alert": "Anomalie detectee",
    "drift_signal": "Signal de derive",
    "drift_confirmed": "Derive confirmee",
    "retrain_start": "Reentrainement lance",
    "retrain_end": "Reentrainement termine",
    "promoted": "Nouveau modele en service",
    "rejected": "Challenger rejete par le garde-fou",
    "service_start": "Service de detection demarre",
    "service_stop": "Service de detection arrete",
    "window_skipped": "Fenetre sautee",
    "collect_gap": "Trou de collecte",
    "registry_error": "Erreur du registre de modeles",
}


class ErreurTelegram(RuntimeError):
    """Configuration de notification inutilisable."""


@dataclass(frozen=True)
class ReglagesTelegram:
    """Configuration des notifications, lue dans `configs/notify.yaml`.

    `jeton` et `conversation` sont exclus du `repr` : une trace de pile, un log
    de CI ou un `print` de mise au point recopierait sinon un secret dans un
    endroit qu'on ne controle pas.
    """

    actif: bool
    evenements: tuple[str, ...]
    regroupement_s: float
    timeout_s: float
    silencieux: tuple[str, ...]
    api: str = field(default=API_PAR_DEFAUT, repr=False)
    jeton: str = field(default="", repr=False)
    conversation: str = field(default="", repr=False)

    @classmethod
    def depuis_config(
        cls, config: dict[str, Any], env: dict[str, str] | None = None
    ) -> ReglagesTelegram:
        bloc = dict(config.get("telegram", {}) or {})
        if not bloc:
            raise ErreurTelegram(
                "Section `telegram:` absente de la configuration des notifications. "
                "Voir configs/notify.yaml."
            )
        environnement = os.environ if env is None else env
        actif = bool(bloc.get("actif", False))

        evenements = tuple(str(nom) for nom in bloc.get("evenements", ()) or ())
        inconnus = [nom for nom in evenements if nom not in EVENEMENTS]
        if inconnus:
            # Un nom invente ne serait jamais notifie, et rien ne le dirait : le
            # service tournerait en silence en croyant alerter.
            raise ErreurTelegram(
                f"Evenements hors du vocabulaire du §9 dans `telegram.evenements` : {inconnus}. "
                f"Vocabulaire : {list(EVENEMENTS)}."
            )
        if actif and not evenements:
            raise ErreurTelegram("`telegram.evenements` est vide : rien ne serait notifie.")

        regroupement_s = float(bloc.get("regroupement_s", 60.0))
        if regroupement_s < 0:
            raise ErreurTelegram(f"`telegram.regroupement_s` negatif : {regroupement_s}.")
        timeout_s = float(bloc.get("timeout_s", 5.0))
        if timeout_s <= 0:
            raise ErreurTelegram(f"`telegram.timeout_s` doit etre positif : {timeout_s}.")

        silencieux = tuple(str(nom) for nom in bloc.get("silencieux", ()) or ())
        jeton = environnement.get("TELEGRAM_BOT_TOKEN", "")
        conversation = environnement.get("TELEGRAM_CHAT_ID", "")
        if actif and not (jeton and conversation):
            raise ErreurTelegram(
                "TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID doivent etre definis dans "
                "l'environnement quand `telegram.actif` vaut true (§4.3 : aucun secret "
                "dans le depot). Mettre `actif: false` pour s'en passer."
            )
        return cls(
            actif=actif,
            evenements=evenements,
            regroupement_s=regroupement_s,
            timeout_s=timeout_s,
            silencieux=silencieux,
            api=str(environnement.get("TELEGRAM_API_BASE") or bloc.get("api", API_PAR_DEFAUT)),
            jeton=jeton,
            conversation=conversation,
        )


def _nombre(valeur: Any) -> str:
    if isinstance(valeur, bool):
        return "oui" if valeur else "non"
    if isinstance(valeur, float):
        return f"{valeur:.4g}"
    return str(valeur)


def decrire(entree: dict[str, Any]) -> str:
    """Une ligne de journal rendue en texte, cles autorisees uniquement.

    Fonction libre et non methode : c'est elle qui porte la regle de non-fuite
    du §4.3, et elle doit pouvoir etre testee sans jeton, sans reseau et sans
    notificateur.
    """
    event = str(entree.get("event", ""))
    libelle = LIBELLES.get(event, event)
    details = entree.get("details")
    morceaux: list[str] = []
    if isinstance(details, dict):
        for cle in CLES_PUBLIABLES.get(event, ()):
            if cle in details and details[cle] is not None:
                morceaux.append(f"{cle} {_nombre(details[cle])}")
    if not morceaux:
        return libelle
    return f"{libelle} — {', '.join(morceaux)}"


@dataclass
class _Groupe:
    """Ce qui attend d'etre notifie pour un type d'evenement."""

    entrees: list[dict[str, Any]] = field(default_factory=list)
    dernier_envoi: float | None = None


class NotificateurTelegram:
    """Abonne du journal qui pousse les evenements retenus vers Telegram.

    N'echoue jamais vers l'appelant : une detection en cours ne doit pas
    s'interrompre parce que le reseau mobile ou l'API Telegram est indisponible.
    Les echecs sont comptes dans `echecs` et affiches a l'arret du service, pour
    qu'un silence de Telegram ne passe pas pour « aucun evenement ».
    """

    def __init__(
        self,
        reglages: ReglagesTelegram,
        *,
        horloge: Callable[[], float],
        transport: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.reglages = reglages
        self.horloge = horloge
        self.transport = transport if transport is not None else self._poster
        self.groupes: dict[str, _Groupe] = {nom: _Groupe() for nom in reglages.evenements}
        self.envois = 0
        self.echecs = 0
        self.ignores = 0

    # ------------------------------------------------------------- abonnement

    def observer_evenement(self, entree: dict[str, Any]) -> None:
        """Appele par `JournalEvenements.ecrire`. Ne leve jamais."""
        try:
            event = str(entree.get("event", ""))
            groupe = self.groupes.get(event)
            if groupe is None:
                self.ignores += 1
                return
            groupe.entrees.append(entree)
            self.rincer()
        except Exception:  # noqa: BLE001 - la notification ne casse pas le service
            return

    def rincer(self, *, forcer: bool = False) -> int:
        """Envoie les groupes dont la fenetre de regroupement est ecoulee.

        Appele a chaque tick de la boucle, et non seulement a l'arrivee d'un
        evenement : sans cela, le dernier evenement d'une rafale resterait en
        attente jusqu'au prochain du meme type, c'est-a-dire peut-etre jamais.
        `forcer` sert a l'arret du service, qui ne doit rien laisser en attente.
        """
        envoyes = 0
        maintenant = self.horloge()
        for event, groupe in self.groupes.items():
            if not groupe.entrees:
                continue
            attente = (
                0.0
                if groupe.dernier_envoi is None
                else self.reglages.regroupement_s - (maintenant - groupe.dernier_envoi)
            )
            if attente > 0 and not forcer:
                continue
            texte = self._rediger(event, groupe.entrees)
            groupe.entrees = []
            groupe.dernier_envoi = maintenant
            if self._envoyer(texte, silencieux=event in self.reglages.silencieux):
                envoyes += 1
        return envoyes

    @property
    def en_attente(self) -> int:
        return sum(len(groupe.entrees) for groupe in self.groupes.values())

    # ---------------------------------------------------------------- redaction

    def _rediger(self, event: str, entrees: list[dict[str, Any]]) -> str:
        """Une entree : le detail. Plusieurs : un compte et la plus recente.

        Le compte vient en premier parce que c'est l'information qui decide
        d'agir ou non : « 42 anomalies » et « 1 anomalie » n'appellent pas la
        meme reaction, alors que le detail de la 42e n'apprend rien de plus.
        """
        derniere = entrees[-1]
        if len(entrees) == 1:
            return decrire(derniere)
        return (
            f"{LIBELLES.get(event, event)} — {len(entrees)} en "
            f"{int(self.reglages.regroupement_s)} s. Derniere : {decrire(derniere)}"
        )

    # ------------------------------------------------------------------ envoi

    def _envoyer(self, texte: str, *, silencieux: bool) -> bool:
        charge: dict[str, Any] = {
            "chat_id": self.reglages.conversation,
            "text": texte,
            "disable_notification": silencieux,
        }
        try:
            self.transport(charge)
        except Exception:  # noqa: BLE001 - voir la docstring de la classe
            self.echecs += 1
            return False
        self.envois += 1
        return True

    def _poster(self, charge: dict[str, Any]) -> None:
        """L'appel HTTPS `sendMessage`, sans dependance supplementaire.

        `urllib` de la bibliotheque standard suffit pour un POST JSON :
        `python-telegram-bot` n'est justifie que par les boutons du bonus (§3).
        """
        url = f"{self.reglages.api}/bot{self.reglages.jeton}/sendMessage"
        requete = urllib.request.Request(  # noqa: S310 - schema fixe par `api`
            url,
            data=json.dumps(charge).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(requete, timeout=self.reglages.timeout_s) as reponse:  # noqa: S310
            if reponse.status >= 400:
                raise ErreurTelegram(f"Telegram a repondu {reponse.status}.")


def construire_notificateur(
    config: dict[str, Any],
    *,
    horloge: Callable[[], float],
    env: dict[str, str] | None = None,
    transport: Callable[[dict[str, Any]], None] | None = None,
) -> NotificateurTelegram | None:
    """Le notificateur, ou `None` si les notifications sont desactivees.

    Rendre `None` plutot qu'un objet inerte : le service affiche alors
    « notifications desactivees » au demarrage, et personne ne croit a tort
    qu'un telephone va sonner pendant la demonstration.
    """
    reglages = ReglagesTelegram.depuis_config(config, env=env)
    if not reglages.actif:
        return None
    return NotificateurTelegram(reglages, horloge=horloge, transport=transport)
