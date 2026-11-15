"""Metriques Prometheus de l'appli de test (livrable M4, phase A).

Ce que l'appli expose doit couvrir les metriques applicatives annoncees au §4.2
du cadrage : requetes/s, latence p50/p95, erreurs 5xx, connexions. Le detail des
choix :

- **Un histogramme, pas une jauge de latence moyenne.** Le cadrage demande p50 et
  p95 ; une moyenne ne permet pas de les reconstituer, et c'est precisement la
  queue de distribution qu'une injection de latence deplace.
- **Des bornes de buckets explicites**, centrees sur la milliseconde a la dizaine
  de millisecondes. Les bornes par defaut de `prometheus_client` commencent a
  5 ms et montent a 10 s : avec une latence de base de 25 ms, le p95 tomberait
  presque toujours dans le meme bucket et l'injection serait invisible.
- **Une jauge par type d'injection** (`app_chaos_active`). Elle n'est pas une
  metrique de l'application mais de l'experience : elle permet de hachurer les
  fenetres d'injection dans Grafana et de verifier le fichier d'etiquettes contre
  ce que l'appli a reellement subi.
- **`app_build_info` et `app_base_latency_ms`.** Une derive reelle S1 est un
  deploiement ; sans ces deux series, rien dans les donnees ne dit *quand* la
  version a change, et l'instant de derive devrait etre note a la main.
- **`ProcessCollector`** (`process_cpu_seconds_total`, `process_resident_memory_bytes`).
  Depuis D46, c'est la source de `cpu_appli` et `memoire_appli` dans
  `configs/datasets/serveur.yaml` : cAdvisor ne voit aucun conteneur sur cet
  hote (Docker Engine 29, magasin d'images par containerd-snapshotter), donc ces
  deux metriques doivent venir du processus lui-meme, pas du conteneur qui le
  porte. Un seul worker uvicorn : le processus mesure est bien celui de l'appli.

Les metriques d'infrastructure qui restent hors de l'application (disque, reseau,
autres conteneurs) viennent de node_exporter et de cAdvisor (phase B) ; elles
sont marquees non obligatoires pour une session serveur depuis D46, pour la
meme raison que ci-dessus.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector

from infra.app.chaos import TYPES

# Bornes en secondes. Resolution fine jusqu'a 100 ms, puis grossiere : au-dela,
# seule compte la presence d'une queue, pas sa forme exacte.
BORNES_LATENCE = (
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.150,
    0.250,
    0.500,
    1.0,
    2.5,
    5.0,
)


class Metriques:
    """Les series exposees sur `/metrics`.

    Regroupees dans une classe et non en variables de module pour qu'un test
    puisse en construire un jeu neuf : les collecteurs de `prometheus_client`
    s'enregistrent dans un registre global et deux applications construites dans
    le meme processus de test entreraient en collision.
    """

    def __init__(self, registre: CollectorRegistry | None = None) -> None:
        self.registre = registre if registre is not None else CollectorRegistry()

        # D46 : process_cpu_seconds_total et process_resident_memory_bytes,
        # scopes au seul processus uvicorn de l'appli — c'est ce qui remplace
        # cAdvisor pour cpu_appli/memoire_appli sur cet hote. Un registre neuf
        # n'a pas de collecteur de processus par defaut, contrairement au
        # registre global de `prometheus_client` : il faut l'ajouter ici.
        ProcessCollector(registry=self.registre)

        self.requetes = Counter(
            "app_requests_total",
            "Requetes servies, par endpoint et par code de statut.",
            ["endpoint", "method", "status"],
            registry=self.registre,
        )
        self.duree = Histogram(
            "app_request_duration_seconds",
            "Duree de traitement d'une requete, injection de latence incluse.",
            ["endpoint"],
            buckets=BORNES_LATENCE,
            registry=self.registre,
        )
        self.en_vol = Gauge(
            "app_in_flight_requests",
            "Requetes en cours de traitement.",
            registry=self.registre,
        )
        self.chaos_actif = Gauge(
            "app_chaos_active",
            "1 si une injection de ce type est en cours, 0 sinon.",
            ["type"],
            registry=self.registre,
        )
        self.chaos_intensite = Gauge(
            "app_chaos_intensity",
            "Intensite de l'injection en cours : millisecondes, part de 5xx, "
            "nombre de fils d'execution ou Mio alloues, selon le type.",
            ["type"],
            registry=self.registre,
        )
        self.version = Gauge(
            "app_build_info",
            "Toujours 1 ; l'etiquette `version` porte l'information.",
            ["version"],
            registry=self.registre,
        )
        self.latence_base = Gauge(
            "app_base_latency_ms",
            "Latence de base declaree par le deploiement courant, en millisecondes.",
            registry=self.registre,
        )

        # Les series d'injection sont initialisees a zero : une serie qui
        # n'apparait qu'au moment de la premiere injection laisse un trou dans
        # les requetes Grafana, et un trou ne se distingue pas d'un zero.
        for type_ in TYPES:
            self.chaos_actif.labels(type=type_).set(0)
            self.chaos_intensite.labels(type=type_).set(0)

    def declarer_deploiement(self, version: str, latence_base_ms: float) -> None:
        self.version.labels(version=version).set(1)
        self.latence_base.set(latence_base_ms)

    def observer(self, endpoint: str, methode: str, statut: int, duree_s: float) -> None:
        self.requetes.labels(endpoint=endpoint, method=methode, status=str(statut)).inc()
        self.duree.labels(endpoint=endpoint).observe(duree_s)

    def refleter_chaos(self, intensites: dict[str, float]) -> None:
        """Met a jour les jauges d'injection pour les types donnes.

        Generique sur le type, pour que l'ajout du pic CPU et de la fuite
        memoire (D46) n'ait rien demande ici : seul `main.py` sait, pour chaque
        type, comment lire l'intensite courante auprès de `Chaos`.
        """
        for type_, valeur in intensites.items():
            self.chaos_actif.labels(type=type_).set(1 if valeur > 0 else 0)
            self.chaos_intensite.labels(type=type_).set(valeur)
