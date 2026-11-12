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

from infra.grafana.build_dashboard import TABLEAUX, construire
from src.serving.evenements import EVENEMENTS

RACINE = Path(__file__).resolve().parents[1]
TABLEAU = RACINE / "infra/grafana/dashboards/infrastructure.json"
PIPELINE = RACINE / "infra/grafana/dashboards/pipeline.json"
DOSSIER_TABLEAUX = RACINE / "infra/grafana/dashboards"
DATASOURCE = RACINE / "infra/grafana/provisioning/datasources/prometheus.yml"
PROVIDER = RACINE / "infra/grafana/provisioning/dashboards/dashboards.yml"
PROMETHEUS = RACINE / "infra/prometheus/prometheus.yml"
COMPOSE = RACINE / "docker-compose.yml"
NOTIFY = RACINE / "configs/notify.yaml"

# Ce que PromQL apporte lui-meme : fonctions, agregateurs, mots-cles,
# etiquettes et intervalles. Tout identifiant restant dans une expression est un
# nom de metrique, et doit donc exister dans l'exporter.
PROMQL_RESERVE = {
    "rate",
    "irate",
    "increase",
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "by",
    "without",
    "clamp_min",
    "clamp_max",
    "vector",
    "time",
    "histogram_quantile",
    "or",
    "and",
    "unless",
    "le",
    "job",
    "name",
    "status",
    "event",
    "strategy",
    "version",
    "alert",
    "__rate_interval",
    "h",
    "m",
    "s",
}


def _lire(chemin: Path) -> dict:
    return yaml.safe_load(chemin.read_text(encoding="utf-8"))


def _tableau(chemin: Path = TABLEAU) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def _tous_les_tableaux() -> list[tuple[str, dict]]:
    """Les deux tableaux de bord suivis. Les tests generiques valent pour les deux."""
    return [(nom, _tableau(DOSSIER_TABLEAUX / f"{nom}.json")) for nom in TABLEAUX]


def _panneaux(tableau: dict) -> list[dict]:
    return [p for p in tableau["panels"] if p["type"] != "row"]


def _expressions(tableau: dict) -> list[tuple[str, str]]:
    return [
        (panneau["title"], cible["expr"])
        for panneau in _panneaux(tableau)
        for cible in panneau["targets"]
    ]


def _identifiants(expr: str) -> set[str]:
    """Les identifiants d'une expression PromQL, valeurs d'etiquettes exclues.

    Sans retirer les chaines entre guillemets, `event="drift_signal"` ferait
    passer `drift_signal` pour un nom de metrique : le test se plaindrait d'une
    serie absente qui n'en est pas une.
    """
    sans_valeurs = re.sub(r'"[^"]*"', "", expr)
    return set(re.findall(r"\b[a-z_][a-z0-9_]*\b", sans_valeurs)) - PROMQL_RESERVE


def test_le_json_suivi_est_celui_du_generateur() -> None:
    # Si ce test echoue, c'est que le JSON a ete modifie sans passer par le
    # generateur : relancer `python infra/grafana/build_dashboard.py`.
    assert _tableau() == construire()


def test_les_deux_tableaux_de_bord_sont_ceux_du_generateur() -> None:
    # `allowUiUpdates: false` : un panneau ajoute dans l'interface serait perdu
    # au prochain provisionnement, et une capture du memoire qui ne se reproduit
    # pas est une capture indefendable.
    for nom, constructeur in TABLEAUX.items():
        assert _tableau(DOSSIER_TABLEAUX / f"{nom}.json") == constructeur(), nom


def test_les_tableaux_de_bord_ont_des_uid_et_des_titres_distincts() -> None:
    # Deux tableaux de bord provisionnes avec le meme uid : Grafana n'en garde
    # qu'un, silencieusement, et c'est l'autre qu'on cherchera pendant la
    # soutenance.
    uids = [tableau["uid"] for _, tableau in _tous_les_tableaux()]
    titres = [tableau["title"] for _, tableau in _tous_les_tableaux()]
    assert len(set(uids)) == len(uids), uids
    assert len(set(titres)) == len(titres), titres


def test_tous_les_panneaux_pointent_vers_la_source_provisionnee() -> None:
    uid = _lire(DATASOURCE)["datasources"][0]["uid"]
    for nom, tableau in _tous_les_tableaux():
        for panneau in _panneaux(tableau):
            assert panneau["datasource"]["uid"] == uid, (nom, panneau["title"])
            for cible in panneau["targets"]:
                assert cible["datasource"]["uid"] == uid, (nom, panneau["title"])


def test_chaque_panneau_porte_un_titre_et_au_moins_une_requete() -> None:
    for nom, tableau in _tous_les_tableaux():
        for panneau in _panneaux(tableau):
            assert panneau["title"].strip(), nom
            assert panneau["targets"], (nom, panneau["title"])
            for cible in panneau["targets"]:
                assert cible["expr"].strip(), (nom, panneau["title"])


def test_chaque_panneau_porte_sa_justification() -> None:
    # Le JSON n'accepte pas de commentaires : la `description` est le seul
    # endroit ou la raison d'un panneau puisse vivre, et Grafana l'affiche au
    # survol, donc elle est lisible sur une capture.
    for nom, tableau in _tous_les_tableaux():
        for panneau in _panneaux(tableau):
            assert len(panneau["description"].strip()) > 40, (nom, panneau["title"])


def test_le_plan_de_controle_est_exclu_des_metriques_applicatives() -> None:
    # Les sondes de sante et les appels d'injection ne sont pas du trafic
    # mesure : les compter melangerait l'experience et son pilotage.
    for panneau in _panneaux(_tableau()):
        for cible in panneau["targets"]:
            expr = cible["expr"]
            if "app_requests_total" in expr or "app_request_duration" in expr:
                assert "/health" in expr and "/chaos" in expr, panneau["title"]


def test_le_tableau_de_bord_du_pipeline_n_interroge_que_l_exporter() -> None:
    # La separation des deux tableaux de bord n'a de sens que si elle tient :
    # une metrique d'infrastructure glissee ici brouillerait la lecture
    # « cause / effet » qui justifie les deux fichiers.
    for titre, expr in _expressions(_tableau(PIPELINE)):
        noms = _identifiants(expr)
        assert noms, (titre, expr)
        assert all(nom.startswith("mdrift_") for nom in noms), (titre, sorted(noms))


def test_les_series_du_tableau_de_bord_du_pipeline_existent_dans_l_exporter() -> None:
    """C'est le test qui attrape la faute la plus silencieuse de cette phase.

    Une metrique mal orthographiee dans une expression PromQL ne produit ni
    erreur ni avertissement : le panneau s'affiche, vide, et un panneau vide ne
    se distingue pas d'une absence d'evenement. On ne s'en apercevrait qu'en
    regardant une capture du chapitre 5, trop tard.
    """
    from src.serving.exporter import ExporteurDetection

    exporteur = ExporteurDetection(strategie="proposed", budget_ms=60_000.0)
    # `mdrift_build_info` porte des etiquettes : la serie n'existe qu'une fois
    # le service demarre, donc on simule la ligne de journal qui la cree.
    exporteur.observer_evenement(
        {
            "event": "service_start",
            "run_id": "test",
            "strategy": "proposed",
            "details": {"version": 1, "threshold": 0.5, "features_empreinte": "abc"},
        }
    )
    exposees = {
        ligne.split("{")[0].split(" ")[0]
        for ligne in exporteur.rendre().decode("utf-8").splitlines()
        if ligne and not ligne.startswith("#")
    }
    for titre, expr in _expressions(_tableau(PIPELINE)):
        for nom in re.findall(r"\bmdrift_[a-z0-9_]*\b", expr):
            assert nom in exposees, (titre, nom, sorted(exposees))


def test_les_etiquettes_event_du_tableau_de_bord_sont_du_vocabulaire_du_journal() -> None:
    # L'etiquette `event` prend ses valeurs dans le vocabulaire du §9, que le
    # journal refuse de depasser. Une valeur inventee dans une requete donnerait
    # un panneau vide, sans rien signaler.
    for titre, expr in _expressions(_tableau(PIPELINE)):
        for valeur in re.findall(r'event="([^"]+)"', expr):
            assert valeur in EVENEMENTS, (titre, valeur)


def test_grafana_ouvre_le_tableau_de_bord_du_depot() -> None:
    compose = _lire(COMPOSE)
    chemin = compose["services"]["grafana"]["environment"][
        "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"
    ]
    dossier = _lire(PROVIDER)["providers"][0]["options"]["path"]
    assert chemin == f"{dossier}/{TABLEAU.name}"


def test_les_cibles_de_collecte_couvrent_les_services_de_la_pile() -> None:
    jobs = {bloc["job_name"] for bloc in _lire(PROMETHEUS)["scrape_configs"]}
    # `detection` a ete declaree des M4 pour que la configuration de collecte
    # n'ait pas a changer a M5 ; le service qui la tient existe depuis.
    assert {"prometheus", "app", "node", "cadvisor", "detection"} <= jobs


def test_la_cible_de_detection_est_le_service_de_detection_au_bon_port() -> None:
    """Trois fichiers doivent s'accorder sur un seul port, sinon la cible est `down`.

    La collecte nomme `detection:8001`, le service Compose doit donc s'appeler
    `detection` et exposer 8001, et la section `exporter:` de serving.yaml doit
    ecouter sur ce port. Changer l'un des trois sans les autres ne casse rien de
    visible : la cible passe simplement `down`, et le tableau de bord du
    pipeline reste vide.
    """
    jobs = {bloc["job_name"]: bloc for bloc in _lire(PROMETHEUS)["scrape_configs"]}
    cibles = jobs["detection"]["static_configs"][0]["targets"]
    assert cibles == ["detection:8001"]
    service, port = cibles[0].split(":")

    compose = _lire(COMPOSE)["services"]
    assert service in compose, service
    assert port in [str(expose) for expose in compose[service]["expose"]]
    assert _lire(RACINE / "configs/serving.yaml")["exporter"]["port"] == int(port)


def test_l_exporter_du_service_ecoute_sur_toutes_les_interfaces_du_conteneur() -> None:
    # Dans le conteneur, c'est Prometheus — depuis un autre conteneur — qui vient
    # collecter : 127.0.0.1 rendrait la cible injoignable. Le defaut de
    # configs/serving.yaml reste 127.0.0.1, pour qu'un `mdrift serve` lance a la
    # main sur un poste n'expose rien.
    service = _lire(COMPOSE)["services"]["detection"]
    assert service["environment"]["DETECTION_EXPORTER_HOST"] == "0.0.0.0"
    assert _lire(RACINE / "configs/serving.yaml")["exporter"]["hote"] == "127.0.0.1"
    # Et rien n'est publie sur l'hote : /metrics decrit l'etat de la detection,
    # ce n'est pas une interface.
    assert "ports" not in service


def test_le_service_de_detection_joint_ses_voisins_par_leur_nom_de_service() -> None:
    # §4.3 : aucun nom d'hote ni IP. Et ces deux variables ont la priorite sur
    # les YAML de configs/, qui sont ecrits pour un poste de developpement.
    env = _lire(COMPOSE)["services"]["detection"]["environment"]
    assert env["PROMETHEUS_URL"] == "http://prometheus:9090"
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_le_journal_d_evenements_survit_au_conteneur() -> None:
    """Le journal est la source unique du §9 : il ne peut pas vivre dans la couche
    jetable du conteneur.

    Volume nomme et non monture depuis l'hote : Docker creerait la monture avec
    l'appartenance de l'hote alors que le conteneur tourne en uid 10003, et la
    premiere ligne de journal echouerait en « permission denied » — au demarrage
    d'une session de demonstration.
    """
    service = _lire(COMPOSE)["services"]["detection"]
    dossier = _lire(RACINE / "configs/serving.yaml")["journal"]["dossier"]
    montures = {m.split(":")[1]: m.split(":")[0] for m in service["volumes"]}
    assert f"/srv/{dossier}" in montures, service["volumes"]
    volume = montures[f"/srv/{dossier}"]
    assert not volume.startswith("."), volume
    assert volume in _lire(COMPOSE)["volumes"], volume


def test_le_conteneur_de_detection_est_borne() -> None:
    # §4.3 : tout conteneur de la pile est borne. Celui-ci plus large que l'appli
    # de test, parce qu'un reentrainement tue en plein milieu ne produirait ni
    # promotion, ni rejet, ni `retrain_end`.
    service = _lire(COMPOSE)["services"]["detection"]
    assert service["cpus"].startswith("${DETECTION_CPUS")
    assert service["mem_limit"].startswith("${DETECTION_MEMORY")


def test_l_image_du_service_installe_le_meme_client_mlflow_que_le_serveur() -> None:
    # Client et serveur partagent le schema de la base. L'image du service
    # installe `mlflow-skinny` — il est client, il n'heberge rien — mais la
    # version doit rester celle de l'image du registre et de pyproject.toml.
    image = _lire(COMPOSE)["services"]["mlflow"]["image"]
    epinglee = image.rsplit(":v", 1)[1]
    dockerfile = (RACINE / "infra/detection/Dockerfile").read_text(encoding="utf-8")
    assert f'"mlflow-skinny=={epinglee}"' in dockerfile, image


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
    for chemin in (COMPOSE, PROMETHEUS, DATASOURCE, PROVIDER, TABLEAU, PIPELINE, NOTIFY):
        texte = chemin.read_text(encoding="utf-8")
        assert not adresse.search(texte), chemin.name
        # Toute valeur sensible vient de l'environnement, jamais du fichier :
        # toute ligne qui en parle doit contenir une interpolation `${...}`.
        for numero, ligne in enumerate(texte.splitlines(), start=1):
            nue = ligne.strip()
            if nue.startswith("#") or not sensible.search(nue):
                continue
            assert "${" in nue, f"{chemin.name}:{numero} : {nue}"


def test_le_service_de_detection_recoit_les_secrets_telegram_par_l_environnement() -> None:
    # §4.3 : le jeton et la conversation ne sont ni dans l'image, ni dans un
    # fichier suivi. Les valeurs par defaut sont vides (`:-`) et non
    # obligatoires (`:?`) : la pile doit demarrer sans Telegram, parce que la
    # detection et les resultats du chapitre 6 n'en dependent pas.
    env = _lire(COMPOSE)["services"]["detection"]["environment"]
    assert env["TELEGRAM_BOT_TOKEN"] == "${TELEGRAM_BOT_TOKEN:-}"
    assert env["TELEGRAM_CHAT_ID"] == "${TELEGRAM_CHAT_ID:-}"
    exemple = (RACINE / ".env.example").read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN=" in exemple
    assert "TELEGRAM_CHAT_ID=" in exemple


def test_les_notifications_sont_inactives_dans_le_depot() -> None:
    # Un depot clone ne doit pas tenter d'appeler une API externe au premier
    # `mdrift serve`, et une CI encore moins.
    import yaml

    bloc = yaml.safe_load(NOTIFY.read_text(encoding="utf-8"))["telegram"]
    assert bloc["actif"] is False


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
