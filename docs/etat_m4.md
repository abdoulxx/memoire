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
| C | Injecteur de pannes : conteneurs stress-ng dedies et limites, allowlist par prefixe, `make stop-chaos`, Locust, ecriture des labels horodates | **terminee** le 29/10 a 08h40 (39 tests, les 5 types injectes pour de vrai) |
| D | Enregistrement des sessions : API Prometheus `query_range` -> Parquet, chargeur de session branche sur les features existantes | **terminee** le 11/11 (289 tests, alignement temporel verifie) |
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

## Resultats de la phase C (injecteur d'anomalies)

`infra/injector/` : `docker_cli.py` (acces a Docker, plus un client factice pour
les tests), `labels.py` (ecriture et lecture des deux fichiers d'etiquettes du
§4.2), `injector.py` (l'injecteur et ses garde-fous), `app_client.py` (plan de
controle de l'appli de test, avec verification de cible), `Dockerfile` (image
`memoire-chaos`, Alpine + stress-ng epingle). `infra/locust/locustfile.py` pour la
charge. Parametres dans `configs/injection.yaml`. Quatre commandes : `chaos`,
`chaos-list`, `chaos-stop`, `drift-note`, plus `make stop-chaos` et
`make stop-chaos-force`. **39 tests** dans `tests/test_injector.py`, **239 tests
au total**.

### Trois regles qui portent le §4.3

1. **Deux marques, jamais une seule.** Un conteneur d'injection porte a la fois un
   prefixe de nom reserve (`memoire-chaos-`) **et** une etiquette Docker
   (`memoire.injection=1`), et l'injecteur exige les deux avant d'arreter quoi que
   ce soit. Le prefixe seul se laisse usurper par une erreur de frappe ;
   l'etiquette seule, n'importe qui peut la poser. Exiger les deux rend impossible
   de viser par accident un conteneur que l'injecteur n'a pas cree — la propriete
   la plus importante du §4.3, et la seule qu'on ne peut pas tester sur une vraie
   machine sans prendre le risque qu'elle ecarte. D'ou le client Docker factice.
2. **Les garde-fous agissent a la lecture de la configuration, pas au lancement.**
   Un type qui declare `moyen: conteneur` ou `moyen: charge` sans bloc `limites`
   fait echouer `Reglages.depuis_config` : il ne peut meme pas etre charge, donc
   aucun conteneur d'injection ne peut exister sans `--cpus` et `--memory`.
3. **Aucune injection sans fin.** La duree est obligatoire, plafonnee par
   `duree_max_s: 600`, et passee **aussi** a `stress-ng --timeout` et a
   `locust --run-time` : si l'injecteur meurt, le conteneur s'arrete seul.

### Les cinq types du §4.2, injectes pour de vrai

Verification sur la machine de mise au point, qui faisait tourner **14 conteneurs,
dont 9 sans aucun rapport avec le memoire** — exactement la situation pour laquelle
le §4.3 existe.

| Type | Demande | Mesure |
|---|---|---|
| `pic_cpu` | 2 travailleurs, 200 s | `cpu=98,77 %`, `mem=16,69 MiB / 256 MiB` : la limite d'un cœur tient exactement ; etiquette 201 s |
| `fuite_memoire` | 192 MiB, 30 s | `205,9 MiB / 256 MiB`, aucun OOM-kill : la limite doit bien rester superieure a l'intensite ; etiquette 32 s |
| `latence` | +250 ms, 20 s | etiquette 08:28:56 -> 08:29:16, **exactement 20 s** (pas de conteneur a demarrer) |
| `erreurs` | 30 % de 5xx, 20 s | etiquette 08:29:17 -> 08:29:37, exactement 20 s |
| `saturation` | 150 utilisateurs, 40 s | trafic de 0 a **230 req/s**, p95 de 0,10 s a **0,234 s** ; etiquette 44 s |

Les injections par conteneur portent 1 a 4 secondes de demarrage de plus que la
duree demandee, et l'etiquette porte la duree **subie** : c'est voulu, et c'est la
raison d'etre du point 3 ci-dessous.

Garde-fous verifies sur la meme machine : les deux leurres (un conteneur avec
l'etiquette seule, un autre avec le prefixe seul) sont **refuses, code de sortie 2,
et survivent** ; `chaos-list` ne liste que la vraie injection ; `chaos-stop` arrete
**1 conteneur et laisse les 13 autres tourner**.

### Quatre defauts trouves par cette verification, et corriges

Les quatre tiennent au meme endroit : **une etiquette qui ne decrit pas ce qui
s'est passe est pire que pas d'etiquette du tout**, parce qu'elle produit un
resultat faux au lieu d'un resultat manquant.

1. **Une anomalie annoncee qui n'a pas eu lieu.** Les premieres injections par
   conteneur s'arretaient a la premiere seconde (`aborting: temp-path '.' must be
   readable and writeable` : utilisateur non privilegie dans un repertoire non
   inscriptible). `docker run` renvoyait pourtant 0, `--rm` effacait le conteneur,
   et l'injecteur ecrivait une etiquette de 180 s sur des metriques parfaitement
   normales. Corrige par `WORKDIR /tmp` dans l'image, `--read-only` plus un
   `--tmpfs /tmp:size=64m,mode=1777` borne, et surtout par une **verification de
   demarrage** : si le conteneur ne tourne pas, l'injecteur leve une erreur avec
   les dernieres lignes du journal et **n'ecrit aucune etiquette**. Les fausses
   etiquettes produites pendant l'essai ont ete supprimees.
2. **Une injection interrompue etiquetait la duree demandee.** L'attente etait un
   `sleep` aveugle : un `make stop-chaos` pendant une injection de 240 s laissait
   l'injecteur attendre l'echeance, puis etiqueter 242 s alors que la charge avait
   cesse au bout de 90 s. L'attente **surveille maintenant le conteneur** (un
   `docker inspect` toutes les 2 s) et rend la main des qu'il disparait. Remesure
   apres correction : arret declenche a 08:35:51 sur une injection de 240 s
   commencee a 08:34:19, etiquette **08:34:19 -> 08:35:55**, soit 96 s — la duree
   subie, detectee en 4 s. Le meme chemin couvre l'OOM-kill et l'arret imprevu de
   `stress-ng`.
3. **L'arret d'urgence disait « remis a zero » alors qu'il avait echoue.** La
   remise a zero du plan de controle de l'appli etait tentee dans un `except`
   silencieux : `chaos-stop` affichait un succes alors que la latence injectee
   continuait de courir. Un arret d'urgence qui ment est lui-meme un probleme de
   securite. La commande renvoie desormais le resultat du plan de controle en
   clair (`remis a zero`, `inchange : <erreur>`, ou `non joignable : ...`), et un
   test fixe la propriete.
4. **L'URL par defaut de l'appli pointait vers un service tiers.** Le port publie
   est une variable depuis la phase B parce que 8000 etait deja pris sur la
   machine : `configs/injection.yaml` pointait donc vers une application sans
   rapport, et un POST `/chaos/latency` lui serait parti — ce que le §4.3
   interdit. Le client **verifie sa cible** avant la premiere injection (la
   signature de `/health` de l'appli de test) et refuse tout le reste ; un 404
   serait arrive trop tard, la requete etant deja partie. `MEMOIRE_APP_URL` est
   documentee dans `.env.example` et a la priorite sur le YAML.

### Ce qui reste a verifier sur le serveur Linux

- **Les deux panneaux par conteneur** (CPU et memoire), toujours vides sous Docker
  Desktop pour la raison expliquee en phase B. Ce sont eux qui permettent de
  verifier a l'œil qu'une injection **reste dans son conteneur et sous sa
  limite** : a verifier avant la premiere session enregistree, pas apres.
- **`make stop-chaos-force`** utilise `xargs -r`, qui est une extension GNU : vrai
  sur le serveur, non verifiable sur ce poste.
- **La duree de demarrage des conteneurs d'injection** (1 a 4 s ici) sera
  differente sur le serveur. Elle n'a pas besoin d'etre connue a l'avance puisque
  l'etiquette porte la duree subie, mais elle fixe le plancher utile d'une
  injection : une anomalie de 5 s ne serait pas mesurable.

### Comment rejouer la verification

```bash
docker build -f infra/injector/Dockerfile -t memoire-chaos:0.18.07 .
docker pull locustio/locust:2.32.4

uv run mdrift chaos pic_cpu --duree 60 --run-id essai
uv run mdrift chaos-list
make stop-chaos                      # arret d'urgence du §4.3
uv run mdrift drift-note deploiement "appli de test 1.1.0" --run-id essai
cat data/labels/essai.csv data/labels/essai_drift.csv
```

## Resultats de la phase D (enregistrement et rejeu des sessions)

Objet de la phase : transformer une plage de temps vecue par le serveur en un
objet de **la meme forme qu'une machine du SMD**, pour que le harnais de rejeu de
M2 n'ait rien a savoir de la source du flux. Une session est enregistree une fois
et rejouee pour chaque strategie : c'est la seule facon de comparer B0, B1 et P
sur donnees reelles, puisqu'on ne peut pas faire subir deux fois la meme heure a
un serveur.

| Fichier | Role |
|---|---|
| `src/serving/prometheus.py` | Le seul endroit du depot qui parle a Prometheus. Rend une expression PromQL sur une grille d'instants connue d'avance. Reutilise tel quel par la boucle en ligne de M5 |
| `configs/datasets/serveur.yaml` | 14 metriques (nom, expression, obligatoire), pas de 60 s, fenetre de `rate`, cibles requises, tolerances de trous, decoupage, regles d'etiquetage |
| `datasets/serveur/settings.py` | Objet de reglages partage par l'enregistreur et le chargeur : ils doivent s'accorder sur le pas, l'ordre des metriques et les tolerances |
| `datasets/serveur/recorder.py` | Ecrit `data/sessions/<run_id>.parquet` + son JSON de metadonnees |
| `datasets/serveur/loader.py` | Relit le Parquet, comble les trous courts, etiquette les points, decoupe train/test, situe la derive |
| `src/evaluation/streams.py` | `build_server_stream` : un `ReplayStream` de source `serveur` |
| `src/cli.py` | `session-check`, `session-record`, `session-info` |

### Commandes

```bash
uv run mdrift session-check                       # avant la session, pas apres
uv run mdrift session-record essai --debut -3h    # -> data/sessions/essai.parquet
uv run mdrift session-info essai                  # description pour le §11, point 5
```

### Les cinq decisions de la phase, et le defaut que chacune ecarte

1. **La grille est faite de multiples du pas, et construite avant la requete.**
   `query_range` evalue a `start`, `start + step`, ... : si `start` est l'heure a
   laquelle la commande a ete lancee, la grille en depend, et deux
   enregistrements de la meme plage ne tombent pas sur les memes instants. Au pas
   de la minute, quelques secondes suffisent a deplacer une fenetre d'un cran par
   rapport a une anomalie etiquetee. Les instants sont arrondis (debut vers le
   haut, fin vers le bas) et la grille n'est jamais deduite de la reponse.
2. **Un trou reste un trou dans le Parquet.** Quand une cible n'a pas repondu,
   Prometheus renvoie simplement une reponse plus courte, sans rien qui le
   signale. Les valeurs absentes sont posees sur la grille en `NaN` et le Parquet
   les garde : il est la mesure, pas son interpretation. C'est le **chargeur** qui
   interpole, sous les tolerances de `configs/datasets/serveur.yaml` (3 points
   consecutifs, 2 % de la session), et il **echoue** au-dela plutot que
   d'inventer un regime.
3. **Les bornes de l'anomalie sont incluses.** A `end_ts`, l'injection tournait
   encore. Une anomalie qui commence entre deux points de grille marque le point
   **suivant** : arrondir vers le bas marquerait un point ou rien ne se passait.
4. **Une anomalie plus courte que le pas est signalee.** Elle serait invisible
   dans les donnees tout en comptant comme un evenement a detecter, ce qui ferait
   baisser le rappel par evenement de **toutes** les strategies sans qu'aucune
   puisse y faire quoi que ce soit. `session-info` l'affiche en ATTENTION.
5. **Une anomalie ou une derive dans la part d'entrainement fait echouer le
   chargement.** Un modele initial entraine sur une anomalie ne la signalera
   jamais, et la comparaison des strategies qui en decoule ne veut plus rien
   dire. Volontairement bloquant (`refuser_anomalie_dans_entrainement: true`).

### Deux refus explicites

- **Une expression qui rend plusieurs series est refusee**, avec les etiquettes
  des series en cause. Prendre la premiere serait un choix arbitraire fait par la
  bibliotheque a la place de l'experimentateur. Zero serie, en revanche, est
  legitime (aucune 5xx sur la plage) : c'est le chargeur qui tranche, parce que
  lui seul sait si la colonne est obligatoire.
- **La fenetre des `rate(...)` est fixee dans la configuration**, pas laissee a
  `$__rate_interval`. Grafana l'adapte au zoom, ce qui est bon pour l'œil et
  inacceptable pour une mesure : la meme session reenregistree doit rendre les
  memes valeurs.

### Defaut trouve par les tests

`float("+Inf")` et `float("NaN")` **reussissent** : le code comptait sur une
exception pour ecarter les valeurs non finies que Prometheus rend en texte, et
les laissait donc passer. Un `histogram_quantile` sur un bucket vide rend `NaN`,
une part calculee sur un denominateur nul rend `+Inf` : un seul `inf` dans une
colonne traverse la normalisation et rend la session entiere inexploitable, sans
que rien ne le signale. Les valeurs non finies comptent desormais comme des
points absents (`math.isfinite`), ce qui est le meme traitement qu'un trou.

### Tests

`tests/test_prometheus.py` (12) et `tests/test_serveur_session.py` (38).
289 tests au total. Les tests d'alignement temporel sont les plus importants de
la phase : ils fixent qu'une anomalie de `T+605` a `T+700` marque le point `T+660`
et lui seul, qu'une anomalie de 35 s ne marque rien mais est signalee, et qu'une
fenetre est positive des qu'elle contient un point anormal — la meme regle que
pour le SMD, sans quoi les deux sources ne sont pas comparables.

### Ce qui reste a Sidy sur le serveur Linux

- Verifier `session-check` sur la pile reelle : c'est la commande qui dira si les
  14 expressions correspondent a quelque chose. Les deux metriques par conteneur
  (`cpu_appli`, `memoire_appli`) sont marquees **obligatoires** precisement parce
  qu'elles manquaient sous Docker Desktop ; un enregistrement qui les perdrait
  doit s'arreter net.
- Enregistrer les vraies sessions. Une session utile commence par une periode
  calme (au moins 40 % du total, sans anomalie ni derive) avant la premiere
  injection.
