"""Verifie la coherence de la pile de collecte (livrable M4, phase B).

Ces tests ne lancent pas Docker : ils verifient ce qui casse en silence. Un
tableau de bord qui pointe vers une source de donnees inexistante s'affiche
vide, une cible absente de `prometheus.yml` ne produit aucune alerte, et un JSON
reecrit a la main dans l'interface Grafana divergerait du depot sans que rien ne
le signale — or une capture du memoire ne vaut que si le depot la reproduit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from infra.grafana.build_dashboard import construire

RACINE = Path(__file__).resolve().parents[1]
TABLEAU = RACINE / "infra/grafana/dashboards/infrastructure.json"
DATASOURCE = RACINE / "infra/grafana/provisioning/datasources/prometheus.yml"
PROVIDER = RACINE / "infra/grafana/provisioning/dashboards/dashboards.yml"
PROMETHEUS = RACINE / "infra/prometheus/prometheus.yml"
COMPOSE = RACINE / "docker-compose.yml"


def _lire(chemin: Path) -> dict:
    return yaml.safe_load(chemin.read_text(encoding="utf-8"))


def _tableau() -> dict:
    return json.loads(TABLEAU.read_text(encoding="utf-8"))


def _panneaux(tableau: dict) -> list[dict]:
    return [p for p in tableau["panels"] if p["type"] != "row"]


def test_le_json_suivi_est_celui_du_generateur() -> None:
    # Si ce test echoue, c'est que le JSON a ete modifie sans passer par le
    # generateur : relancer `python infra/grafana/build_dashboard.py`.
    assert _tableau() == construire()


def test_tous_les_panneaux_pointent_vers_la_source_provisionnee() -> None:
    uid = _lire(DATASOURCE)["datasources"][0]["uid"]
    for panneau in _panneaux(_tableau()):
        assert panneau["datasource"]["uid"] == uid, panneau["title"]
        for cible in panneau["targets"]:
            assert cible["datasource"]["uid"] == uid, panneau["title"]


def test_chaque_panneau_porte_un_titre_et_au_moins_une_requete() -> None:
    for panneau in _panneaux(_tableau()):
        assert panneau["title"].strip()
        assert panneau["targets"]
        for cible in panneau["targets"]:
            assert cible["expr"].strip()


def test_le_plan_de_controle_est_exclu_des_metriques_applicatives() -> None:
    # Les sondes de sante et les appels d'injection ne sont pas du trafic
    # mesure : les compter melangerait l'experience et son pilotage.
    for panneau in _panneaux(_tableau()):
        for cible in panneau["targets"]:
            expr = cible["expr"]
            if "app_requests_total" in expr or "app_request_duration" in expr:
                assert "/health" in expr and "/chaos" in expr, panneau["title"]


def test_grafana_ouvre_le_tableau_de_bord_du_depot() -> None:
    compose = _lire(COMPOSE)
    chemin = compose["services"]["grafana"]["environment"][
        "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"
    ]
    dossier = _lire(PROVIDER)["providers"][0]["options"]["path"]
    assert chemin == f"{dossier}/{TABLEAU.name}"


def test_les_cibles_de_collecte_couvrent_les_services_de_la_pile() -> None:
    jobs = {bloc["job_name"] for bloc in _lire(PROMETHEUS)["scrape_configs"]}
    # `detection` est declaree des M4 pour que la configuration ne change pas a
    # M5 ; elle reste `down` jusque-la, ce qui est visible et sans effet.
    assert {"prometheus", "app", "node", "cadvisor", "detection"} <= jobs


def test_la_source_de_donnees_suit_le_pas_de_collecte() -> None:
    # `timeInterval` sert de pas minimal a Grafana : plus petit que le pas de
    # collecte, il engendre des trous entre deux points.
    pas = _lire(PROMETHEUS)["global"]["scrape_interval"]
    assert _lire(DATASOURCE)["datasources"][0]["jsonData"]["timeInterval"] == pas


def test_aucun_port_n_est_publie_sur_toutes_les_interfaces() -> None:
    # §4.3 : rien ne doit etre joignable depuis l'exterieur du serveur par
    # defaut. La valeur par defaut de BIND_ADDR est dans le fichier Compose
    # lui-meme, donc un oubli de .env ne peut pas tout exposer.
    for nom, service in _lire(COMPOSE)["services"].items():
        for publication in service.get("ports", []):
            assert publication.startswith("${BIND_ADDR:-127.0.0.1}:"), f"{nom} : {publication}"


def test_les_images_sont_epinglees() -> None:
    for nom, service in _lire(COMPOSE)["services"].items():
        image = service.get("image", "")
        assert ":" in image, nom
        assert not image.endswith(":latest"), nom


def test_la_retention_prometheus_est_bornee_en_duree_et_en_taille() -> None:
    # Une borne de taille n'a pas besoin qu'on la surveille, contrairement a
    # l'espace disque (§4.3).
    commande = " ".join(_lire(COMPOSE)["services"]["prometheus"]["command"])
    assert "--storage.tsdb.retention.time=" in commande
    assert "--storage.tsdb.retention.size=" in commande


def test_le_mot_de_passe_grafana_n_a_pas_de_valeur_par_defaut() -> None:
    env = _lire(COMPOSE)["services"]["grafana"]["environment"]
    mot_de_passe = env["GF_SECURITY_ADMIN_PASSWORD"]
    assert mot_de_passe.startswith("${GRAFANA_ADMIN_PASSWORD:?")
    assert "GRAFANA_ADMIN_PASSWORD=" in (RACINE / ".env.example").read_text(encoding="utf-8")


def test_aucun_secret_ni_nom_d_hote_dans_la_pile() -> None:
    # §4.3 : les cibles sont des noms de service Docker, jamais des IP ni des
    # noms d'hote. 127.0.0.1 et localhost sont la boucle locale, pas une adresse
    # de serveur.
    adresse = re.compile(r"\b(?!127\.0\.0\.1|0\.0\.0\.0)\d{1,3}(\.\d{1,3}){3}\b")
    sensible = re.compile(r"PASSWORD|TOKEN|SECRET|API_KEY", re.IGNORECASE)
    for chemin in (COMPOSE, PROMETHEUS, DATASOURCE, PROVIDER, TABLEAU):
        texte = chemin.read_text(encoding="utf-8")
        assert not adresse.search(texte), chemin.name
        # Toute valeur sensible vient de l'environnement, jamais du fichier :
        # toute ligne qui en parle doit contenir une interpolation `${...}`.
        for numero, ligne in enumerate(texte.splitlines(), start=1):
            nue = ligne.strip()
            if nue.startswith("#") or not sensible.search(nue):
                continue
            assert "${" in nue, f"{chemin.name}:{numero} : {nue}"


def test_le_registre_sert_les_artefacts_par_http() -> None:
    # Sans `--serve-artifacts`, le serveur renvoie au client un chemin de systeme
    # de fichiers qui n'existe que dans le conteneur : `registry.charger` echoue
    # a telecharger le modele, et le service refuse de demarrer sans que rien
    # n'indique la cause. C'est le defaut le plus facile a introduire ici.
    commande = " ".join(_lire(COMPOSE)["services"]["mlflow"]["command"])
    assert "--serve-artifacts" in commande
    assert "--artifacts-destination=" in commande
    assert "--no-serve-artifacts" not in commande


def test_le_registre_a_un_backend_de_base_de_donnees() -> None:
    # L'alias `champion` releve du model registry, qui n'existe pas sur un simple
    # magasin de fichiers : un `--backend-store-uri` pointant un dossier ferait
    # echouer toute promotion.
    commande = " ".join(_lire(COMPOSE)["services"]["mlflow"]["command"])
    assert "--backend-store-uri=sqlite:" in commande


def test_la_version_de_l_image_mlflow_est_celle_du_client_python() -> None:
    # Client et serveur partagent le schema de la base : les faire diverger
    # laisserait celui des deux qui ouvre la base la migrer pour l'autre.
    image = _lire(COMPOSE)["services"]["mlflow"]["image"]
    epinglee = image.rsplit(":v", 1)[1]
    pyproject = (RACINE / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"mlflow=={epinglee}"' in pyproject, image


def test_le_registre_n_impose_aucun_emplacement_d_artefacts_au_serveur() -> None:
    # `artifact_location` ne vaut que pour un backend sqlite local (tests, mise
    # au point). Avec le serveur, c'est lui qui decide, et le renseigner cote
    # client court-circuiterait le service d'artefacts.
    registre = _lire(RACINE / "configs/serving.yaml")["registry"]
    assert not registre.get("artifact_location")
