"""Engendre `infra/grafana/dashboards/infrastructure.json` (livrable M4, phase B).

    python infra/grafana/build_dashboard.py

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
            description="Valeur annoncee par le deploiement courant, a comparer au p50 observe.",
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
                "`detection` reste a 0 jusqu'a M5, c'est normal."
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
            "du modele en service arrivent avec le service de detection (M5), sur un "
            "second tableau de bord. Fichier provisionne depuis le depot : toute "
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


def main() -> None:
    # Chemin deduit de l'emplacement du fichier : la commande doit marcher
    # depuis n'importe quel repertoire courant.
    depot = Path(__file__).resolve().parent / "dashboards"
    depot.mkdir(parents=True, exist_ok=True)
    chemin = depot / "infrastructure.json"
    tableau = construire()
    chemin.write_text(json.dumps(tableau, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    panneaux = [p for p in tableau["panels"] if p["type"] != "row"]
    print(f"ecrit {chemin} : {len(panneaux)} panneaux, {len(tableau['panels'])} entrees")
    for p in tableau["panels"]:
        if p["type"] != "row":
            for c in p["targets"]:
                assert c["expr"].count("{") == c["expr"].count("}"), p["title"]
    print("accolades equilibrees dans toutes les expressions")


if __name__ == "__main__":
    main()
