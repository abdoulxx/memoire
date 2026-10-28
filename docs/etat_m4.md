# Etat de M4 — fichier de reprise

But de ce fichier : permettre de reprendre le travail **sans contexte**, apres une
interruption (limite d'usage atteinte, machine redemarree, session perdue). Mis a
jour et pousse a la fin de chaque phase. Il sera supprime quand M4 sera clos — son
contenu definitif ira dans `docs/journal.md`, comme pour M3.

Fenetre de travail autonome : jusqu'au **29/10/2026 04h00**.

## Ce que M4 doit livrer

Jalon M4 du cadrage : « Infrastructure sur mon serveur : Docker Compose (appli de
test, Prometheus, node_exporter, cAdvisor, Grafana), injecteur, enregistrement des
sessions » -> livrable memoire : **description des donnees reelles** (chapitre 5 et
section 6 sur le test final).

M4 produit du **code et des configurations**, deployables sur le serveur de Sidy.
Le deploiement lui-meme et l'enregistrement des vraies sessions lui appartiennent :
ce depot doit lui donner une commande par etape, et rien qui demande un reglage a
la main.

## Regles qui ne changent pas

- **§4.3 du cadrage, non negociable.** L'injecteur ne touche **que** des conteneurs
  qu'il a lui-meme crees, nommes par un prefixe reserve et crees avec des limites
  Docker explicites. Il refuse tout conteneur qu'il n'a pas cree. `make stop-chaos`
  arrete tout. Jamais d'injection sur un conteneur de production ou de client.
- **Aucun secret, IP, nom d'hote, nom de client ou domaine** dans le depot, les
  resultats ou les captures. Tout ce qui est propre au serveur passe par `.env`
  (non suivi) et `.env.example` (suivi, vide).
- **Le meme code qu'en rejeu.** Features, modeles, detecteurs de derive et
  strategies ne sont pas reecrits pour le mode en ligne : seule la source du flux
  change. Une session enregistree doit se rejouer avec le harnais de M2.
- Aucune valeur en dur : tout parametre va dans un YAML de `configs/`.
- Push a la fin de chaque phase, avec un message qui documente ce qui a ete fait.

## Phases

| Phase | Contenu | Etat |
|---|---|---|
| A | Appli de test FastAPI : charge simulee, endpoints de panne (latence, 5xx), metriques Prometheus, version deployee | **terminee** le 28/10 a 19h20 (13 tests, image construite et sondee) |
| B | Docker Compose complet : Prometheus + scrape, node_exporter, cAdvisor, Grafana provisionne (datasource + dashboard JSON) | **terminee** le 28/10 a 20h05 (pile lancee, 5 cibles collectees, 12 tests) |
| C | Injecteur de pannes : conteneurs stress-ng dedies et limites, allowlist par prefixe, `make stop-chaos`, Locust, ecriture des labels horodates | **en cours** |
| D | Enregistrement des sessions : API Prometheus `query_range` -> Parquet, chargeur de session branche sur les features existantes | a faire |
| E | Tests, D35+ dans `decisions.md`, entree M4 de `journal.md`, suppression de ce fichier | a faire |

## Commandes de reprise

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

## Resultats de la phase A (appli de test)

`infra/app/` : `settings.py` (etat du deploiement), `chaos.py` (etat des
injections applicatives), `metrics.py` (series Prometheus), `main.py`
(application FastAPI), `Dockerfile`. 13 tests dans `tests/test_app_test.py`.

| Endpoint | Role |
|---|---|
| `GET /work?complexite=` | Trafic normal : latence de base dispersee en log-normale |
| `GET /health` | Sonde de sante, exempte d'injection |
| `GET /metrics` | Exposition Prometheus |
| `POST /chaos/latency` | Anomalie « latence » du §4.2 : +X ms pendant D secondes |
| `POST /chaos/errors` | Anomalie « rafale d'erreurs » : X % de 5xx pendant D secondes |
| `POST /chaos/reset` | Arret immediat de toutes les injections applicatives |
| `GET /chaos/state` | Injections en cours et historique, pour recouper les etiquettes |

### Trois decisions qui portent la validite des etiquettes

1. **Le plan de controle n'est jamais perturbe.** `/health`, `/metrics` et
   `/chaos/*` ne subissent ni la latence ajoutee ni les 5xx. Sans cela, injecter
   100 % de 5xx rendrait `/chaos/reset` injoignable : il faudrait tuer le
   conteneur pour revenir a la normale, et l'instant de retour au calme ne
   correspondrait plus aux etiquettes — la session serait a refaire. Un test fixe
   la propriete.
2. **Aucune injection n'est permanente** : duree obligatoire, plafonnee a 3 600 s.
   Une anomalie est un evenement court par definition du cadrage ; ce qui dure est
   une derive, et une derive s'obtient par un deploiement. Une injection oubliee
   contaminerait toutes les etiquettes suivantes.
3. **L'historique dit ce qui a ete subi, pas ce qui a ete demande** : un
   `/chaos/reset` tronque l'echeance a l'instant present. L'injecteur de la phase C
   ecrira les etiquettes, mais c'est l'appli qui sait ce qu'elle a reellement subi,
   et pouvoir comparer les deux est ce qui permet de reperer une etiquette decalee
   avant d'avoir construit un resultat dessus.

### Deux points mesures, pas supposes

- **Bornes de l'histogramme de latence** : celles de `prometheus_client` commencent
  a 5 ms et montent a 10 s. Avec une latence de base de 25 ms, le p95 tomberait
  presque toujours dans le meme bucket et une injection de latence serait
  invisible. Les bornes sont explicites et resserrees sous 100 ms.
- **Sonde de sante necessaire** : au premier essai du conteneur, les deux premieres
  requetes ont ete refusees (code 000) parce qu'uvicorn n'ecoutait pas encore. Un
  `HEALTHCHECK` a ete ajoute — `docker compose` s'en servira en phase B pour
  ordonner le demarrage. La sonde reste verte pendant une rafale de 5xx, et c'est
  voulu : un conteneur qui subit une anomalie injectee n'est pas en panne.

### Verification de l'image

`docker build -f infra/app/Dockerfile -t memoire-app:1.0.0 .` puis conteneur
lance : `/health` repond, `/work` passe de **0,10 s a 0,44 s** sous une injection
de 400 ms, `app_chaos_active{type="latence"}` vaut 1 et
`app_chaos_intensity{type="latence"}` vaut 400 pendant l'injection. Etat de sante
`healthy`. Conteneur supprime apres l'essai.

## Resultats de la phase B (pile de collecte)

`docker-compose.yml` a la racine, cinq services : `app` (image construite depuis
`infra/app/Dockerfile`), `prometheus`, `node-exporter`, `cadvisor`, `grafana`.
Configurations dans `infra/prometheus/prometheus.yml` et
`infra/grafana/provisioning/`. Tableau de bord engendre par
`infra/grafana/build_dashboard.py` vers `infra/grafana/dashboards/infrastructure.json`
(15 panneaux). 12 tests dans `tests/test_infra_stack.py`. **200 tests passent.**

### Quatre decisions de securite, toutes verifiees par un test

1. **Aucun port publie sur toutes les interfaces.** Chaque publication est liee a
   `${BIND_ADDR:-127.0.0.1}` : la valeur par defaut est dans le fichier Compose
   lui-meme, donc un `.env` oublie ne peut pas exposer Grafana sur Internet.
   L'acces au serveur se fait par tunnel SSH. Mettre `0.0.0.0` est alors un choix
   explicite et visible dans `.env`.
2. **node-exporter et cAdvisor ne publient plus rien du tout.** Prometheus les
   atteint par le reseau Compose ; publier 9100 et 8080 exposerait les metriques
   de l'hote et l'interface d'un conteneur privilegie sans aucun besoin. Constat
   qui a declenche la decision : au premier lancement, le port 9100 etait deja
   pris sur la machine.
3. **Grafana refuse de demarrer sans mot de passe** (`${GRAFANA_ADMIN_PASSWORD:?}`).
   Un defaut `admin` sur un serveur joignable serait pire qu'un demarrage rate.
4. **Les ports publies sont des variables** (`APP_PORT`, `PROM_PORT`,
   `GRAFANA_PORT`). Au premier lancement, le port 8000 etait occupe par une
   application sans rapport : le §4.3 interdit de toucher aux conteneurs des
   autres, donc c'est la pile du memoire qui se decale.

### Retention bornee, et non surveillee

Le §4.3 demande de surveiller la retention Prometheus et l'espace disque. La
configuration borne la retention **en duree et en taille**
(`--storage.tsdb.retention.time`, `--storage.tsdb.retention.size`) : une borne de
taille n'a pas besoin qu'on la regarde. Le panneau « Espace disque de la base et
de l'hote » et le panneau « Series actives » servent a voir venir la borne, pas a
la remplacer.

### Verification sur la pile reellement lancee

| Verification | Resultat |
|---|---|
| Services demarres | 5 sur 5, `app` passe `healthy` avant que Prometheus demarre (`depends_on: service_healthy`) |
| Cibles Prometheus | `prometheus`, `app`, `node`, `cadvisor` **up** ; `detection` **down** (normal jusqu'a M5) |
| Provisionnement Grafana | source `prometheus` par defaut, dossier `Memoire`, tableau `memoire-infra` charge avec 15 panneaux, `meta.provisioned = true` |
| Requetes des panneaux | les 18 requetes du tableau de bord ont ete passees a Prometheus : **15 renvoient des series**, 3 restent vides (voir ci-dessous) |
| Injection de latence a travers la pile | p95 mesure **0,459 s** sous une injection de 250 ms sur une base de 25 ms |
| Injection de 5xx a travers la pile | 120 requetes, **32 refusees (27 %)** pour une injection reglee a 30 % ; le panneau affiche **27,3 %** |

### Deux defauts trouves par cette verification, et corriges

1. **Le panneau « Part de reponses 5xx » restait vide sans aucune erreur.**
   `sum(rate(...))` sur une serie absente ne renvoie aucune serie : le panneau
   affichait un trou la ou la bonne reponse est zero — et un trou se confond avec
   l'absence d'anomalie. Corrige par `or vector(0)` au numerateur. Le `clamp_min`
   du denominateur, lui, protege du cas « aucun trafic du tout » : deux trous
   differents, deux protections.
2. **Le port 8000 etait deja pris**, et le port 9100 aussi. D'ou les points 2 et 4
   ci-dessus. Un serveur fait tourner autre chose que cette experience ; la pile
   doit s'adapter, jamais l'inverse.

### Ce qui reste a verifier sur le serveur Linux

Trois requetes restent vides sur un poste Windows, pour une seule raison
technique, et il faut le reverifier au premier deploiement :

- **CPU par conteneur** et **Memoire par conteneur** : sous Docker Desktop,
  cAdvisor n'arrive pas a identifier la couche lecture-ecriture des conteneurs
  (magasin d'images `overlayfs` et non `overlay2`), echoue a les enregistrer et
  n'expose alors aucune etiquette `name`. `cgroup: host` a ete ajoute — sans lui
  cAdvisor ne voyait meme pas les cgroups des conteneurs — mais ne suffit pas sur
  ce poste. Sur un hote Linux avec le magasin d'images par defaut, le chemin
  existe. **Ces deux panneaux portent la verification du §4.3** (une injection
  reste-t-elle dans son conteneur et sous sa limite) : ils doivent etre verifies
  avant la premiere session enregistree.
- **Espace libre sur /** : la surcouche `infra/compose/compose.windows.yml`
  retire le montage de la racine, que Docker Desktop refuse (`/` n'y est ni
  « shared » ni « slave »). Le fichier de base, celui du serveur, le conserve.

### Comment relancer

```bash
cp .env.example .env            # puis renseigner GRAFANA_ADMIN_PASSWORD
docker compose up -d            # sur le serveur Linux
docker compose ps

# sur un poste Windows, pour la mise au point seulement :
docker compose -f docker-compose.yml -f infra/compose/compose.windows.yml up -d
```
