"""Exporter Prometheus du service de detection (M5, phase C).

Le service de detection est, du point de vue de Prometheus, une cible comme
l'appli de test ou node_exporter : il expose `/metrics`, Prometheus le collecte
toutes les 15 s, Grafana lit la base. C'est ce qui permet de montrer sur un
**meme** tableau de bord la cause (CPU, latence, erreurs) et l'effet (score
d'anomalie, alerte, derive confirmee, promotion) — le livrable du chapitre 5.

Trois choix de cette phase valent d'etre expliques, parce qu'ils ne se devinent
pas en lisant les series :

1. **L'exporter s'abonne au journal d'evenements, il ne le double pas.** Le
   compteur `mdrift_events_total{event=...}` est incremente depuis
   `JournalEvenements.ecrire`, pas depuis la boucle. Deux chemins paralleles
   auraient fini par divergier — un evenement journalise mais non compte, ou
   l'inverse — et une capture Grafana qui contredit le journal ne se defend pas
   devant un jury. Le vocabulaire du §9 devient ainsi, litteralement, la liste
   des valeurs de l'etiquette `event`.

2. **Toutes les series sont initialisees a zero au demarrage.** Une serie qui
   n'apparait qu'au premier evenement laisse un trou dans Grafana, et un trou ne
   se distingue pas d'un zero : un panneau « derives confirmees » vide pourrait
   aussi bien dire « aucune derive » que « le service ne tourne pas ». Meme
   raison que les jauges d'injection de l'appli de test (M4, phase A).

3. **`mdrift_last_window_timestamp_seconds` porte l'instant de la fenetre, pas
   celui de la collecte.** C'est la seule facon de voir qu'un service *repond*
   mais n'avance plus : `time() - mdrift_last_window_timestamp_seconds` mesure
   la fraicheur reelle de la detection. Un `up == 1` ne dit rien de cela.

Ce que l'exporter n'expose pas, et c'est voulu : aucune valeur de feature,
aucun nom de metrique du serveur, aucun identifiant de conversation. Les
etiquettes sont bornees a `event`, `strategy` et `version` — trois ensembles
petits et connus d'avance. Une etiquette par fenetre ou par feature ferait
exploser la cardinalite de la base, premiere cause de saturation disque d'un
Prometheus (§4.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

from src.serving.evenements import EVENEMENTS

if TYPE_CHECKING:  # import en anneau : c'est la boucle qui construit l'exporter
    from src.serving.loop import ResultatFenetre

# Valeur portee par `mdrift_model_version` tant qu'aucun champion n'est charge.
# Zero et non NaN : une version de modele est un entier >= 1, donc 0 ne peut etre
# confondu avec une version reelle, et un stat Grafana l'affiche sans broncher.
VERSION_INCONNUE = 0


class ErreurExporteur(RuntimeError):
    """Configuration d'exporter inutilisable."""


@dataclass(frozen=True)
class ReglagesExporteur:
    """Configuration de l'exporter, lue dans `configs/serving.yaml`."""

    actif: bool
    hote: str
    port: int

    @classmethod
    def depuis_config(
        cls, config: dict[str, Any], env: dict[str, str] | None = None
    ) -> ReglagesExporteur:
        bloc = dict(config.get("exporter", {}) or {})
        if not bloc:
            raise ErreurExporteur(
                "Section `exporter:` absente de la configuration du mode en ligne. "
                "Voir configs/serving.yaml."
            )
        environnement = os.environ if env is None else env
        # Meme regle que MLFLOW_TRACKING_URI et PROMETHEUS_URL : l'adresse
        # d'ecoute depend du deploiement. Dans un conteneur il faut 0.0.0.0 pour
        # que Prometheus atteigne la cible ; sur un poste, 127.0.0.1 pour ne rien
        # exposer. Aucune des deux n'a sa place en dur dans un fichier suivi.
        hote = environnement.get("DETECTION_EXPORTER_HOST") or str(bloc.get("hote", "127.0.0.1"))
        port = int(environnement.get("DETECTION_EXPORTER_PORT") or bloc.get("port", 8001))
        if not 1 <= port <= 65535:
            raise ErreurExporteur(f"`exporter.port` hors bornes : {port}.")
        if not hote:
            raise ErreurExporteur("`exporter.hote` est vide.")
        return cls(actif=bool(bloc.get("actif", True)), hote=hote, port=port)


class ExporteurDetection:
    """Les series du service de detection, et ce qui les alimente.

    Construit son propre `CollectorRegistry` : les collecteurs de
    `prometheus_client` s'enregistrent sinon dans un registre global, et deux
    services construits dans le meme processus de test entreraient en collision.
    """

    def __init__(
        self,
        registre: CollectorRegistry | None = None,
        *,
        strategie: str = "",
        budget_ms: float | None = None,
    ) -> None:
        self.registre = registre if registre is not None else CollectorRegistry()
        self.strategie = strategie

        self.info = Gauge(
            "mdrift_build_info",
            "Toujours 1 ; les etiquettes portent l'information sur le service en cours.",
            ["strategy", "run_id", "features_empreinte"],
            registry=self.registre,
        )
        self.actif = Gauge(
            "mdrift_service_up",
            "1 quand le service de detection traite des fenetres, 0 apres son arret.",
            registry=self.registre,
        )
        self.evenements = Counter(
            "mdrift_events_total",
            "Evenements du journal du §9, par type. Source unique : le journal lui-meme.",
            ["event"],
            registry=self.registre,
        )
        self.fenetres = Counter(
            "mdrift_windows_total",
            "Fenetres scorees depuis le demarrage (prechargement exclu).",
            registry=self.registre,
        )
        self.score = Gauge(
            "mdrift_window_score",
            "Score d'anomalie de la derniere fenetre scoree.",
            registry=self.registre,
        )
        self.seuil = Gauge(
            "mdrift_window_threshold",
            "Seuil d'alerte du modele en service, recalcule a chaque entrainement.",
            registry=self.registre,
        )
        self.alerte = Gauge(
            "mdrift_window_alert",
            "1 si la derniere fenetre a depasse le seuil, 0 sinon.",
            registry=self.registre,
        )
        self.derniere_fenetre = Gauge(
            "mdrift_last_window_timestamp_seconds",
            "Instant de fin de la derniere fenetre traitee. Mesure la fraicheur reelle.",
            registry=self.registre,
        )
        self.duree_moyenne = Gauge(
            "mdrift_window_duration_ms_mean",
            "Temps de traitement moyen d'une fenetre, reentrainement exclu (§9).",
            registry=self.registre,
        )
        self.duree_p95 = Gauge(
            "mdrift_window_duration_ms_p95",
            "Temps de traitement d'une fenetre au 95e percentile, reentrainement exclu.",
            registry=self.registre,
        )
        self.duree_mediane = Gauge(
            "mdrift_window_duration_ms_median",
            "Temps de traitement median d'une fenetre, reentrainement exclu (§9).",
            registry=self.registre,
        )
        self.duree_max = Gauge(
            "mdrift_window_duration_ms_max",
            "Temps de traitement maximal d'une fenetre, reentrainement exclu (§9).",
            registry=self.registre,
        )
        self.budget = Gauge(
            "mdrift_window_budget_ms",
            "Intervalle de collecte en millisecondes : la borne du temps reel (I7 / T1).",
            registry=self.registre,
        )
        self.derive_part = Gauge(
            "mdrift_drift_share",
            "Part des features dont la statistique depasse le seuil de derive.",
            registry=self.registre,
        )
        self.derive_persistance = Gauge(
            "mdrift_drift_streak",
            "Fenetres consecutives ou le signal de derive est present (le K du §6).",
            registry=self.registre,
        )
        self.derive_features = Gauge(
            "mdrift_drift_features",
            "Nombre de features en derive sur la derniere fenetre.",
            registry=self.registre,
        )
        self.derive_taux_alertes = Gauge(
            "mdrift_drift_alert_rate_signal",
            "1 quand le signal secondaire sur le taux d'alertes est present (ADWIN).",
            registry=self.registre,
        )
        self.version = Gauge(
            "mdrift_model_version",
            "Version du modele en service dans le registre (0 = aucun champion charge).",
            registry=self.registre,
        )
        self.promotion_ts = Gauge(
            "mdrift_last_promotion_timestamp_seconds",
            "Instant de la fenetre de la derniere promotion, 0 si aucune.",
            registry=self.registre,
        )
        self.secondes_reentrainement = Counter(
            "mdrift_retrain_seconds_total",
            "Temps de calcul cumule passe a reentrainer (metrique de cout du §7).",
            registry=self.registre,
        )

        # Toutes les series a zero des le depart : un trou dans un panneau ne se
        # distingue pas d'un zero, et un tableau de bord vide au demarrage du
        # service ne permet pas de dire si le service tourne.
        for event in EVENEMENTS:
            self.evenements.labels(event=event)
        self.actif.set(0)
        self.version.set(VERSION_INCONNUE)
        self.promotion_ts.set(0)
        for jauge in (
            self.score,
            self.seuil,
            self.alerte,
            self.derniere_fenetre,
            self.duree_moyenne,
            self.duree_p95,
            self.derive_part,
            self.derive_persistance,
            self.derive_features,
            self.derive_taux_alertes,
        ):
            jauge.set(0)
        if budget_ms is not None:
            self.budget.set(float(budget_ms))

    # -------------------------------------------------------------- abonnement

    def observer_evenement(self, entree: dict[str, Any]) -> None:
        """Compte une ligne du journal. Appele par `JournalEvenements.ecrire`.

        Ne leve jamais : l'exporter est de l'observation, et une erreur de
        comptage ne doit pas interrompre un service de detection ni, pire,
        empecher l'ecriture de la ligne de journal qui la precede.
        """
        try:
            event = str(entree.get("event", ""))
            details = entree.get("details") or {}
            self.evenements.labels(event=event).inc()
            if event == "service_start":
                self.actif.set(1)
                self.info.labels(
                    strategy=str(entree.get("strategy", self.strategie)),
                    run_id=str(entree.get("run_id", "")),
                    features_empreinte=str(details.get("features_empreinte", "")),
                ).set(1)
                self._version(details.get("version"))
                self._nombre(self.seuil, details.get("threshold"))
            elif event == "service_stop":
                self.actif.set(0)
            elif event == "promoted":
                self._version(details.get("version"))
                self._nombre(self.seuil, details.get("threshold"))
                self._nombre(self.promotion_ts, details.get("fin_ts"))
            elif event == "retrain_end":
                secondes = details.get("seconds")
                if isinstance(secondes, int | float) and secondes > 0:
                    self.secondes_reentrainement.inc(float(secondes))
        except Exception:  # noqa: BLE001 - l'observation ne casse pas le service
            return

    def observer_fenetre(self, resultat: ResultatFenetre, details: dict[str, Any]) -> None:
        """Enregistre une fenetre scoree et l'etat de la surveillance de derive.

        `details` est ce que la strategie a renvoye pour cette fenetre. Les cles
        attendues sont celles de `DriftState.as_details` ; une strategie qui ne
        surveille pas la derive (B0, B1) n'en fournit aucune, et les jauges
        correspondantes restent a zero, ce qui est exact pour elle.
        """
        try:
            self.fenetres.inc()
            self.score.set(resultat.score)
            self.seuil.set(resultat.seuil)
            self.alerte.set(1 if resultat.alerte else 0)
            self.derniere_fenetre.set(resultat.fin_ts)
            self._nombre(self.derive_part, details.get("share"))
            self._nombre(self.derive_persistance, details.get("streak"))
            self._nombre(self.derive_features, details.get("n_features_drifting"))
            signal = details.get("alert_rate_signal")
            if signal is not None:
                self.derive_taux_alertes.set(1 if signal else 0)
        except Exception:  # noqa: BLE001 - voir `observer_evenement`
            return

    def observer_mesures(self, moyenne: float, p95: float, mediane: float, maximum: float) -> None:
        self.duree_moyenne.set(moyenne)
        self.duree_p95.set(p95)
        self.duree_mediane.set(mediane)
        self.duree_max.set(maximum)

    # ---------------------------------------------------------------- service

    def rendre(self) -> bytes:
        """Le corps de la reponse `/metrics`, au format d'exposition Prometheus."""
        return generate_latest(self.registre)

    def servir(self, reglages: ReglagesExporteur) -> Any:
        """Ouvre le serveur HTTP `/metrics` dans un thread. Rend le serveur.

        Import local : `prometheus_client.start_http_server` ouvre un socket, et
        importer ce module ne doit pas dependre de la possibilite d'en ouvrir un
        (les tests rendent les metriques sans serveur).
        """
        from prometheus_client import start_http_server

        serveur, _ = start_http_server(
            port=reglages.port, addr=reglages.hote, registry=self.registre
        )
        return serveur

    # ---------------------------------------------------------------- interne

    def _version(self, valeur: Any) -> None:
        try:
            self.version.set(int(valeur))
        except (TypeError, ValueError):
            self.version.set(VERSION_INCONNUE)

    @staticmethod
    def _nombre(jauge: Gauge, valeur: Any) -> None:
        if isinstance(valeur, bool):
            jauge.set(1 if valeur else 0)
        elif isinstance(valeur, int | float):
            jauge.set(float(valeur))
