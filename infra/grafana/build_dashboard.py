"""Engendre les tableaux de bord Grafana suivis par git (M4 phase B, M5 phase C).

    python infra/grafana/build_dashboard.py

Deux fichiers, deux lectures :

- `infrastructure.json` : ce que l'infrastructure et l'appli de test produisent,
  avec les fenetres d'injection qui permettent de les lire. C'est la **cause**.
- `pipeline.json` : ce que le service de detection en fait — score, alertes,
  derive confirmee, reentrainements, modele en service, temps de traitement.
  C'est l'**effet**, et ce sont les captures du chapitre 5.

Les deux partagent la meme source de donnees et la meme fenetre de temps, donc
passer de l'un a l'autre compare bien le meme intervalle. Les separer plutot que
tout empiler sur une page : une capture de 25 panneaux est illisible dans un
memoire imprime, et les deux publics ne sont pas les memes — l'un sert a valider
une session, l'autre a montrer le pipeline MLOps.

Pourquoi un generateur et non un JSON ecrit a la main : un tableau de bord
Grafana ne tolere pas une virgule de trop, les requetes PromQL s'y repetent d'un
panneau a l'autre, et le JSON n'accepte pas de commentaires. Les justifications
vivent donc dans le champ `description` de chaque panneau — Grafana les affiche
au survol, ce qui les rend lisibles sur une capture du memoire — et le filtre qui
exclut le plan de controle est defini une seule fois (`SANS_CONTROLE`).

Le JSON engendre est suivi par git : c'est lui que Grafana lit au demarrage
(`infra/grafana/provisioning/dashboards/dashboards.yml`). Apres toute
modification de ce fichier, relancer la commande ci-dessus puis
`docker compose restart grafana`.
"""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}

SANS_CONTROLE = 'endpoint!~"/health|/metrics|/chaos.*"'


def cible(expr: str, legende: str | None = None, ref: str = "A", instant: bool = False) -> dict:
    c: dict = {
        "datasource": DS,
        "editorMode": "code",
        "expr": expr,
        "refId": ref,
        "range": not instant,
    }
    if instant:
        c["instant"] = True
    if legende is not None:
        c["legendFormat"] = legende
    return c


def serie(
    titre: str,
    x: int,
    y: int,
    w: int,
    h: int,
    cibles: list[dict],
    unite: str | None = None,
    description: str = "",
    mini: float | None = None,
    maxi: float | None = None,
    pile: bool = False,
) -> dict:
    custom: dict = {
        "drawStyle": "line",
        "lineWidth": 1,
        "fillOpacity": 35 if pile else 12,
        "showPoints": "never",
        "spanNulls": False,
    }
    if pile:
        custom["stacking"] = {"mode": "normal", "group": "A"}
    champs: dict = {"custom": custom}
    if unite:
        champs["unit"] = unite
    if mini is not None:
        champs["min"] = mini
    if maxi is not None:
        champs["max"] = maxi
    return {
        "type": "timeseries",
        "title": titre,
        "description": description,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": champs, "overrides": []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": cibles,
    }


def stat(
    titre: str,
    x: int,
    y: int,
    w: int,
    h: int,
    cibles: list[dict],
    unite: str | None = None,
    description: str = "",
    nom: bool = False,
) -> dict:
    champs: dict = {"color": {"mode": "fixed", "fixedColor": "text"}}
    if unite:
        champs["unit"] = unite
    return {
        "type": "stat",
        "title": titre,
        "description": description,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": champs, "overrides": []},
        "options": {
            "colorMode": "none",
            "graphMode": "none",
            "textMode": "name" if nom else "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
        "targets": cibles,
    }


def rangee(titre: str, y: int) -> dict:
    return {
        "type": "row",
        "title": titre,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def construire() -> dict:
    panneaux: list[dict] = []
    y = 0

    panneaux.append(rangee("Etat de l'experience", y))
    y += 1
    panneaux.append(
        stat(
            "Version deployee",
            0,
            y,
            4,
            4,
            [cible("app_build_info", "{{version}}", instant=True)],
            description=(
                "Une derive reelle S1 est un deploiement. Cette serie date le "
                "changement de version dans les donnees elles-memes, au lieu de le "
                "noter a la main apres coup."
            ),
            nom=True,
        )
    )
    panneaux.append(
        stat(
            "Latence de base declaree",
            4,
            y,
            4,
            4,
            [cible("app_base_latency_ms", "ms", instant=True)],
            unite="ms",
            description=(
                "Valeur annoncee par le deploiement courant, a comparer au p50 "
                "observe. L'ecart entre les deux est ce que l'injection de latence "
                "ajoute ; si elle s'en ecarte sans injection, c'est l'appli de test "
                "qui ne tient pas sa promesse et la session n'est pas exploitable."
            ),
        )
    )
    panneaux.append(
        serie(
            "Injections applicatives en cours",
            8,
            y,
            16,
            4,
            [cible("app_chaos_active", "{{type}}")],
            mini=0,
            maxi=1.2,
            description=(
                "1 pendant une injection, 0 sinon. Sert a situer les fenetres "
                "d'anomalie sur les autres panneaux, et a recouper le fichier "
                "d'etiquettes contre ce que l'appli a reellement subi."
            ),
        )
    )
    y += 4

    panneaux.append(rangee("Appli de test", y))
    y += 1
    panneaux.append(
        serie(
            "Requetes par seconde, par statut",
            0,
            y,
            12,
            7,
            [
                cible(
                    "sum by (status) (rate(app_requests_total{"
                    + SANS_CONTROLE
                    + "}[$__rate_interval]))",
                    "{{status}}",
                )
            ],
            unite="reqps",
            pile=True,
            description=(
                "Le plan de controle est exclu : il ne subit pas les injections, et "
                "le compter melangerait le trafic mesure avec les sondes de sante."
            ),
        )
    )
    panneaux.append(
        serie(
            "Latence de service : p50 et p95",
            12,
            y,
            12,
            7,
            [
                cible(
                    f"histogram_quantile(0.50, sum by (le) (rate("
                    f"app_request_duration_seconds_bucket{{{SANS_CONTROLE}}}"
                    f"[$__rate_interval])))",
                    "p50",
                    "A",
                ),
                cible(
                    f"histogram_quantile(0.95, sum by (le) (rate("
                    f"app_request_duration_seconds_bucket{{{SANS_CONTROLE}}}"
                    f"[$__rate_interval])))",
                    "p95",
                    "B",
                ),
            ],
            unite="s",
            mini=0,
            description=(
                "Les deux quantiles demandes au §4.2. Une injection de latence "
                "deplace d'abord le p95 : c'est pour cela que les bornes de "
                "l'histogramme sont resserrees sous 100 ms."
            ),
        )
    )
    y += 7
    panneaux.append(
        serie(
            "Part de reponses 5xx",
            0,
            y,
            12,
            7,
            [
                cible(
                    "(sum(rate(app_requests_total{"
                    + SANS_CONTROLE
                    + ',status=~"5.."}[$__rate_interval])) or vector(0))'
                    " / clamp_min(sum(rate(app_requests_total{"
                    + SANS_CONTROLE
                    + "}[$__rate_interval])), 0.001)",
                    "part 5xx",
                )
            ],
            unite="percentunit",
            mini=0,
            description=(
                "Deux protections, pour deux trous differents, tous deux constates "
                "sur la pile reelle. `or vector(0)` au numerateur : sans aucune "
                "reponse 5xx, la somme ne renvoie aucune serie et le panneau "
                "restait vide alors que la bonne reponse est zero. `clamp_min` au "
                "denominateur : sans trafic du tout, le rapport vaudrait 0/0. Dans "
                "les deux cas, un trou se confondrait avec un taux d'erreur nul, "
                "c'est-a-dire avec l'absence d'anomalie."
            ),
        )
    )
    panneaux.append(
        serie(
            "Requetes en cours de traitement",
            12,
            y,
            12,
            7,
            [cible("app_in_flight_requests", "en vol")],
            mini=0,
            description=(
                "Les « connexions » du §4.2. Monte sous saturation Locust et sous "
                "injection de latence, pour deux raisons differentes."
            ),
        )
    )
    y += 7

    panneaux.append(rangee("Hote", y))
    y += 1
    panneaux.append(
        serie(
            "CPU hote utilise",
            0,
            y,
            8,
            7,
            [
                cible(
                    '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[$__rate_interval])))',
                    "cpu",
                )
            ],
            unite="percent",
            mini=0,
            maxi=100,
            description=(
                "Le CPU de l'hote entier, toutes charges confondues. A lire avec le "
                "panneau « CPU par conteneur » : une injection doit se voir ici sans "
                "que la part des autres conteneurs bouge, sinon le confinement du "
                "§4.3 n'a pas tenu."
            ),
        )
    )
    panneaux.append(
        serie(
            "Memoire hote utilisee",
            8,
            y,
            8,
            7,
            [
                cible(
                    "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
                    "memoire",
                )
            ],
            unite="percent",
            mini=0,
            maxi=100,
            description=(
                "Calcule sur `MemAvailable` et non sur `MemFree` : le cache de pages "
                "est reclamable, et le compter comme occupe ferait apparaitre un "
                "serveur sain a 95 % de memoire utilisee en permanence. Une fuite "
                "injectee doit plafonner a la limite --memory de son conteneur."
            ),
        )
    )
    panneaux.append(
        serie(
            "Reseau hote",
            16,
            y,
            8,
            7,
            [
                cible(
                    "sum(rate(node_network_receive_bytes_total"
                    '{device!~"lo|veth.*|docker.*|br-.*"}[$__rate_interval]))',
                    "entrant",
                    "A",
                ),
                cible(
                    "sum(rate(node_network_transmit_bytes_total"
                    '{device!~"lo|veth.*|docker.*|br-.*"}[$__rate_interval]))',
                    "sortant",
                    "B",
                ),
            ],
            unite="Bps",
            description=(
                "Les interfaces virtuelles de Docker sont exclues : elles compteraient "
                "une deuxieme fois le trafic interne entre conteneurs."
            ),
        )
    )
    y += 7

    panneaux.append(rangee("Conteneurs", y))
    y += 1
    panneaux.append(
        serie(
            "CPU par conteneur",
            0,
            y,
            12,
            7,
            [
                cible(
                    'sum by (name) (rate(container_cpu_usage_seconds_total{name!=""}'
                    "[$__rate_interval]))",
                    "{{name}}",
                )
            ],
            unite="percentunit",
            mini=0,
            description=(
                "C'est ici que se verifie le §4.3 : une injection CPU doit rester "
                "confinee au conteneur d'injection et ne jamais depasser la limite "
                "--cpus qui lui a ete donnee."
            ),
        )
    )
    panneaux.append(
        serie(
            "Memoire par conteneur",
            12,
            y,
            12,
            7,
            [cible('sum by (name) (container_memory_working_set_bytes{name!=""})', "{{name}}")],
            unite="bytes",
            mini=0,
            description=(
                "Une fuite memoire injectee doit plafonner a la limite --memory du "
                "conteneur dedie, et non consommer la RAM du serveur."
            ),
        )
    )
    y += 7

    panneaux.append(rangee("Sante de la collecte et retention", y))
    y += 1
    panneaux.append(
        serie(
            "Cibles de collecte joignables",
            0,
            y,
            8,
            6,
            [cible("up", "{{job}}")],
            mini=0,
            maxi=1.2,
            description=(
                "Si la collecte decroche pendant une injection, la session est "
                "inexploitable : le trou ne se distingue pas d'une absence "
                "d'anomalie. A verifier avant de retenir une session. La cible "
                "`detection` est le service de detection : elle vaut 0 quand il "
                "n'est pas lance, ce qui n'empeche pas d'enregistrer une session."
            ),
        )
    )
    panneaux.append(
        serie(
            "Series actives dans Prometheus",
            8,
            y,
            8,
            6,
            [cible("prometheus_tsdb_head_series", "series")],
            mini=0,
            description=(
                "Surveille l'explosion de cardinalite, premiere cause de saturation "
                "disque d'un Prometheus."
            ),
        )
    )
    panneaux.append(
        serie(
            "Espace disque de la base et de l'hote",
            16,
            y,
            8,
            6,
            [
                cible("sum(prometheus_tsdb_storage_blocks_bytes)", "base Prometheus", "A"),
                cible(
                    'node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay|squashfs"}',
                    "libre sur /",
                    "B",
                ),
            ],
            unite="bytes",
            mini=0,
            description=(
                "Le §4.3 demande de surveiller la retention Prometheus et l'espace "
                "disque. La retention est bornee en taille autant qu'en duree "
                "(docker-compose.yml) ; ce panneau est la pour voir venir la borne."
            ),
        )
    )
    y += 6

    return {
        "uid": "memoire-infra",
        "title": "Memoire — infrastructure et appli de test",
        "description": (
            "Tableau de bord M4 : ce que l'infrastructure et l'appli de test "
            "produisent, et les fenetres d'injection qui permettent de les lire. "
            "Le score d'anomalie, les alertes, les derives confirmees et la version "
            "du modele en service sont sur le tableau de bord « pipeline de "
            "detection » (uid memoire-pipeline), qui couvre le meme intervalle de "
            "temps. Fichier provisionne depuis le depot : toute "
            "modification se fait dans infra/grafana/dashboards/, pas dans "
            "l'interface."
        ),
        "tags": ["memoire", "m4"],
        "timezone": "browser",
        "editable": False,
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": panneaux,
    }


# Fraicheur de la detection : la seule mesure qui distingue un service qui
# *repond* d'un service qui *avance*. Le filtre `> 0` est necessaire : avant la
# premiere fenetre la jauge vaut 0, et `time() - 0` afficherait l'age de l'epoque
# Unix, soit cinquante ans de retard. Avec le filtre, la serie est simplement
# absente jusqu'a la premiere fenetre, ce qui est la verite.
FRAICHEUR = "time() - (mdrift_last_window_timestamp_seconds > 0)"


def construire_pipeline() -> dict:
    """Tableau de bord du service de detection (M5) : les captures du chapitre 5."""
    panneaux: list[dict] = []
    y = 0

    panneaux.append(rangee("Etat du service", y))
    y += 1
    panneaux.append(
        stat(
            "Service de detection",
            0,
            y,
            4,
            4,
            [cible("mdrift_service_up", "actif", instant=True)],
            description=(
                "1 quand le service traite des fenetres, 0 apres un arret propre, "
                "absent quand le processus ne tourne pas. Les trois cas se "
                "distinguent, ce qui n'aurait pas ete le cas avec une serie creee "
                "au premier evenement."
            ),
        )
    )
    panneaux.append(
        stat(
            "Modele en service",
            4,
            y,
            4,
            4,
            [cible("mdrift_model_version", "version", instant=True)],
            description=(
                "Version portant l'alias `champion` dans le registre MLflow. "
                "Deplacer cet alias **est** la mise en service (§6, point 5) : "
                "ce nombre est donc la seule reponse vraie a « quel modele tourne ». "
                "0 signifie qu'aucun champion n'est charge."
            ),
        )
    )
    panneaux.append(
        stat(
            "Fraicheur de la detection",
            8,
            y,
            4,
            4,
            [cible(FRAICHEUR, "age", instant=True)],
            unite="s",
            description=(
                "Age de la derniere fenetre traitee. C'est le panneau a regarder "
                "d'abord : un service peut repondre a Prometheus tout en n'avancant "
                "plus, parce que la collecte a un trou ou que le registre ne repond "
                "pas. Au-dela de deux fois l'intervalle de collecte, quelque chose "
                "ne va pas."
            ),
        )
    )
    panneaux.append(
        stat(
            "Seuil d'alerte courant",
            12,
            y,
            4,
            4,
            [cible("mdrift_window_threshold", "seuil", instant=True)],
            description=(
                "Percentile eleve des scores d'entrainement du modele en service, "
                "recalcule a chaque reentrainement (§6). Il saute a chaque "
                "promotion : c'est normal, et c'est ce que montre le panneau des "
                "scores juste en dessous."
            ),
        )
    )
    panneaux.append(
        stat(
            "Fenetres traitees",
            16,
            y,
            4,
            4,
            [cible("mdrift_windows_total", "fenetres", instant=True)],
            description=(
                "Depuis le demarrage du service. Les fenetres du prechargement n'y "
                "sont pas : elles remplissent l'historique de reentrainement mais ne "
                "sont ni scorees ni alertees (§2), et les compter ici ferait croire "
                "a des centaines de detections instantanees au demarrage."
            ),
        )
    )
    panneaux.append(
        stat(
            "Derive : part des features",
            20,
            y,
            4,
            4,
            [cible("mdrift_drift_share", "part", instant=True)],
            unite="percentunit",
            description=(
                "Part des 152 features dont la statistique de KS depasse le seuil, "
                "sur la derniere fenetre. C'est la regle d'agregation du §6, point 1 : "
                "une part, pas un test corrige, parce qu'a 20 000 points les "
                "p-valeurs n'ont aucune specificite (D18)."
            ),
        )
    )
    y += 4

    panneaux.append(rangee("Detection", y))
    y += 1
    panneaux.append(
        serie(
            "Score d'anomalie et seuil",
            0,
            y,
            16,
            8,
            [
                cible("mdrift_window_score", "score", "A"),
                cible("mdrift_window_threshold", "seuil", "B"),
            ],
            mini=0,
            description=(
                "Le panneau central de la demonstration : une alerte est exactement "
                "le moment ou la premiere courbe passe au-dessus de la seconde. Le "
                "seuil est une marche et non une droite, parce qu'un reentrainement "
                "promu le recalcule."
            ),
        )
    )
    panneaux.append(
        serie(
            "Alertes par heure",
            16,
            y,
            8,
            8,
            [
                cible(
                    '3600 * rate(mdrift_events_total{event="alert"}[$__rate_interval])',
                    "alertes/h",
                )
            ],
            mini=0,
            description=(
                "Le taux d'alertes du §7, exprime a l'heure comme les fausses "
                "alertes par heure. Hors periode d'anomalie injectee, ce taux "
                "**est** le taux de fausses alertes : c'est la lecture directe du "
                "cout d'un detecteur en exploitation."
            ),
        )
    )
    y += 8

    panneaux.append(rangee("Surveillance de derive", y))
    y += 1
    panneaux.append(
        serie(
            "Part des features en derive et persistance",
            0,
            y,
            12,
            7,
            [
                cible("mdrift_drift_share", "part des features", "A"),
                cible("mdrift_drift_streak", "fenetres consecutives", "B"),
            ],
            mini=0,
            description=(
                "Les deux signaux de la strategie P. La part dit qu'il se passe "
                "quelque chose maintenant ; la persistance dit depuis combien de "
                "fenetres, et c'est elle qui distingue une derive durable d'une "
                "anomalie passagere (§6, point 2). Une confirmation arrive quand la "
                "seconde courbe atteint K."
            ),
        )
    )
    panneaux.append(
        serie(
            "Signaux et confirmations de derive",
            12,
            y,
            12,
            7,
            [
                cible(
                    'increase(mdrift_events_total{event="drift_signal"}[$__rate_interval])',
                    "signaux",
                    "A",
                ),
                cible(
                    'increase(mdrift_events_total{event="drift_confirmed"}[$__rate_interval])',
                    "confirmations",
                    "B",
                ),
            ],
            mini=0,
            description=(
                "Un signal est une fenetre suspecte, une confirmation est une "
                "decision. L'ecart entre les deux courbes est ce que l'etape de "
                "confirmation filtre : sans elle, chaque pic du panneau de gauche "
                "declencherait un reentrainement. Le §7 compte les confirmations "
                "sur machines stables comme des fausses confirmations."
            ),
        )
    )
    y += 7

    panneaux.append(rangee("Reentrainement, validation, mise en service", y))
    y += 1
    panneaux.append(
        serie(
            "Reentrainements, promotions, rejets",
            0,
            y,
            12,
            7,
            [
                cible('mdrift_events_total{event="retrain_start"}', "reentrainements", "A"),
                cible('mdrift_events_total{event="promoted"}', "promus", "B"),
                cible('mdrift_events_total{event="rejected"}', "rejetes", "C"),
            ],
            mini=0,
            description=(
                "Cumuls, pas des taux : ce sont les trois nombres de la famille "
                "« cout » du §7, et le rapport promus / rejetes est ce que le "
                "garde-fou de validation produit. Un rejet n'est pas un echec : "
                "c'est un modele qui n'est pas entre en service, et le champion a "
                "tenu."
            ),
        )
    )
    panneaux.append(
        serie(
            "Temps de calcul passe a reentrainer",
            12,
            y,
            6,
            7,
            [cible("mdrift_retrain_seconds_total", "cumul", "A")],
            unite="s",
            mini=0,
            description=(
                "L'autre metrique de cout du §7, celle qui permet de comparer la "
                "strategie proposee au reentrainement periodique : ce que la "
                "periodicite depense sans derive, la strategie P ne le depense pas."
            ),
        )
    )
    panneaux.append(
        serie(
            "Version du modele en service",
            18,
            y,
            6,
            7,
            [cible("mdrift_model_version", "version", "A")],
            mini=0,
            description=(
                "Chaque marche est une promotion effective dans le registre. Si "
                "cette courbe ne bouge pas alors que des promotions sont comptees a "
                "gauche, c'est que le registre a refuse : le champion reste en "
                "service, et l'evenement `registry_error` le dit."
            ),
        )
    )
    y += 7

    panneaux.append(rangee("Temps reel et sante de la boucle (I7 / T1)", y))
    y += 1
    panneaux.append(
        serie(
            "Temps de traitement d'une fenetre",
            0,
            y,
            12,
            7,
            [
                cible("mdrift_window_duration_ms_mean", "moyenne", "A"),
                cible("mdrift_window_duration_ms_p95", "p95", "B"),
                cible("mdrift_window_budget_ms", "budget (intervalle de collecte)", "C"),
            ],
            unite="ms",
            mini=0,
            description=(
                "Les indicateurs I7 / T1 du chapitre 2 : le traitement d'une fenetre "
                "doit tenir sous l'intervalle de collecte, sinon le service prend du "
                "retard sans jamais le rattraper. C'est **la** mesure que le rejeu ne "
                "peut pas donner : il score par blocs et ne produit qu'un temps "
                "amorti (§9). Le reentrainement en est exclu, il est compte a part."
            ),
        )
    )
    panneaux.append(
        serie(
            "Fenetres sautees, trous de collecte, erreurs de registre",
            12,
            y,
            12,
            7,
            [
                cible(
                    'increase(mdrift_events_total{event="window_skipped"}[$__rate_interval])',
                    "fenetres sautees",
                    "A",
                ),
                cible(
                    'increase(mdrift_events_total{event="collect_gap"}[$__rate_interval])',
                    "trous de collecte",
                    "B",
                ),
                cible(
                    'increase(mdrift_events_total{event="registry_error"}[$__rate_interval])',
                    "erreurs de registre",
                    "C",
                ),
            ],
            mini=0,
            description=(
                "Les trois facons dont la boucle rate quelque chose sans mentir : une "
                "fenetre dont un point manque au bord est sautee plutot que scoree "
                "sur une valeur inventee ; un retard trop grand n'est pas rattrape "
                "fenetre par fenetre, sous peine d'alerter sur des evenements "
                "passes ; une erreur de registre laisse le champion en service. Ce "
                "panneau rend ces renoncements visibles — une session ou ils "
                "s'accumulent n'est pas exploitable."
            ),
        )
    )
    y += 7

    return {
        "uid": "memoire-pipeline",
        "title": "Memoire — pipeline de detection",
        "description": (
            "Tableau de bord M5 : ce que le service de detection fait des metriques "
            "collectees — score, alertes, surveillance de derive, reentrainements, "
            "modele en service, temps de traitement. Toutes les series viennent de "
            "l'exporter du service, qui s'abonne au journal d'evenements du §9 : "
            "aucun compteur ici ne peut exister sans la ligne de journal "
            "correspondante. Les metriques brutes de l'infrastructure sont sur le "
            "tableau de bord « infrastructure et appli de test » (uid memoire-infra), "
            "sur le meme intervalle de temps. Fichier provisionne depuis le depot : "
            "toute modification se fait dans infra/grafana/dashboards/, pas dans "
            "l'interface."
        ),
        "tags": ["memoire", "m5"],
        "timezone": "browser",
        "editable": False,
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": panneaux,
    }


# Les deux tableaux de bord suivis, par nom de fichier. Les tests comparent le
# JSON commite a ce que ces fonctions produisent : un panneau modifie dans
# l'interface Grafana serait perdu au prochain provisionnement, et le test est la
# pour que personne ne l'apprenne par une capture ratee.
TABLEAUX = {
    "infrastructure": construire,
    "pipeline": construire_pipeline,
}


def main() -> None:
    # Chemin deduit de l'emplacement du fichier : la commande doit marcher
    # depuis n'importe quel repertoire courant.
    depot = Path(__file__).resolve().parent / "dashboards"
    depot.mkdir(parents=True, exist_ok=True)
    for nom, tableau in TABLEAUX.items():
        chemin = depot / f"{nom}.json"
        donnees = tableau()
        chemin.write_text(
            json.dumps(donnees, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        panneaux = [p for p in donnees["panels"] if p["type"] != "row"]
        print(f"ecrit {chemin} : {len(panneaux)} panneaux, {len(donnees['panels'])} entrees")
        for p in donnees["panels"]:
            if p["type"] != "row":
                for c in p["targets"]:
                    # Une accolade orpheline dans une expression PromQL passe la
                    # relecture et casse le panneau a l'affichage.
                    assert c["expr"].count("{") == c["expr"].count("}"), p["title"]
    print("accolades equilibrees dans toutes les expressions")


if __name__ == "__main__":
    main()
