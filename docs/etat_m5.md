# Etat de M5 — fichier de reprise

But de ce fichier : permettre de reprendre le travail **sans contexte**, apres une
interruption (limite d'usage atteinte, machine redemarree, session perdue). Mis a
jour et pousse a la fin de chaque phase. Il sera supprime quand M5 sera clos — son
contenu definitif ira dans `docs/journal.md`, comme pour M3 et M4.

## Ce que M5 doit livrer

Jalon M5 du cadrage : « Boucle en ligne : service de detection, registre MLflow,
exporter, tableau de bord Grafana, notifications Telegram » -> livrable memoire :
**captures du pipeline MLOps (chapitre 5)**.

Le schema du §2 (mode en ligne) est le contrat a realiser :

```
[Prometheus] --(query_range, chaque minute)--> [Service de detection]
                                                 features -> champion -> score
                                                 surveillance de derive -> strategie P
                                                 reentrainement -> validation -> promotion
   |                    |                  |                    |
[Exporter]        [MLflow registry]   [Journal JSON]        [Telegram]
   |
[Grafana]
```

## Regles qui ne changent pas

- **Le meme code qu'en rejeu.** `src/features/`, `src/models/`, `src/drift/` et
  `src/strategies/` ne sont pas reecrits : la boucle en ligne consomme les memes
  interfaces que `src/evaluation/replay.py`, et une divergence entre les deux
  invaliderait la comparaison du chapitre 6 avec le chapitre 5.
- **Aucune etiquette dans une decision de la strategie** (§6). En ligne, il n'y a
  de toute facon pas d'etiquette : c'est la situation que le harnais simulait.
- **Aucun secret, IP, nom d'hote ou domaine** dans le depot. Jeton Telegram,
  identifiant de conversation et URL MLflow par l'environnement.
- Aucune valeur en dur : tout parametre va dans un YAML de `configs/`.
- **Le journal d'evenements est la source unique.** Telegram et Grafana consomment
  les memes evenements que le harnais de rejeu ecrit (§9), pas des messages ad hoc.
- Push a la fin de chaque phase, avec un message qui documente ce qui a ete fait.

## Phases

| Phase | Contenu | Etat |
|---|---|---|
| A | Registre de modeles : wrapper MLflow (tracking + registry, alias `champion`), service `mlflow` dans Compose, enregistrement de l'entrainement initial et des promotions | **terminee** (11/11/2026) |
| B | Boucle en ligne : lecture Prometheus fenetre par fenetre, features, score, seuil, strategie, reentrainement / validation / promotion, journal JSON lignes, mesure `window_ms` | **terminee** (11/11/2026) |
| C | Exporter Prometheus du service de detection + tableau de bord Grafana du pipeline | **terminee** (12/11/2026) |
| D | Notifications Telegram : un message par minute et par type, construites depuis le journal d'evenements | **terminee** (12/11/2026) |
| E | Tests, D39+ dans `decisions.md`, entree M5 de `journal.md`, suppression de ce fichier | a faire |

## Phase A, ce qui est en place

- `src/features/signature.py` : le contrat de features (noms des colonnes +
  fenetrage + pas d'echantillonnage) condense en une empreinte sha256, enregistree
  avec chaque version de modele et **verifiee au chargement**. Rien dans un
  `IsolationForest` ne verifie ce contrat : scorer des colonnes qui ont change de
  sens ne leve aucune erreur tant que leur nombre est le meme.
- `src/tracking/registry.py` : `ReglagesRegistre` (lecture de `configs/serving.yaml`,
  priorite a `MLFLOW_TRACKING_URI`, bornes de requete), `enregistrer`,
  `promouvoir`, `champion`, `historique`, `charger`, `charger_champion`. L'artefact
  enregistre est le **detecteur entier** (standardisation + foret + seuil), le meme
  fichier joblib que relit le harnais de rejeu.
- `configs/serving.yaml` : section `registry:` seule pour l'instant. Le fichier ne
  contient **aucun** parametre de detection — ils restent dans `model.yaml`,
  `features.yaml`, `drift.yaml` et `strategies/proposed.yaml`.
- `docker-compose.yml` : service `mlflow` (image `v2.22.4`, backend sqlite,
  `--serve-artifacts`, port lie a `${BIND_ADDR}`, bornes CPU/memoire), volume
  `memoire-mlflow-data`, variables dans `.env.example`.
- `mdrift registry-init <run_id>` (entrainement initial + promotion) et
  `mdrift registry-info` (quel modele est en service, avec sa signature).
- 39 tests nouveaux (`tests/test_signature.py`, `tests/test_registry.py`, 4 dans
  `tests/test_infra_stack.py`), tournes contre un vrai registre sqlite. Total 328.

Ce qui reste a verifier sur le serveur, et que seul Sidy peut faire : `docker
compose up -d mlflow` puis `uv run mdrift registry-info` depuis l'hote — c'est le
chemin complet (serveur -> client -> telechargement d'artefact) que les tests
sqlite ne traversent pas.

## Phase B, ce qui est en place

- `src/serving/loop.py` : `BoucleEnLigne`, le service de detection du §2. Un tick
  lit les nouveaux points de Prometheus, construit les fenetres completes qui en
  decoulent, score avec le champion, consulte la strategie, et reentraine /
  valide / promeut quand elle le demande. Le temps d'horloge et le sommeil sont
  injectes : les tests avancent l'horloge eux-memes, sans aucun `sleep` reel.
- Quatre decisions de cette phase ne sont pas evidentes et aucune n'est
  rattrapable apres coup :
  1. **Le prechargement ne score pas.** Au demarrage, la boucle lit les dernieres
     heures pour remplir l'historique de reentrainement — sans quoi le garde-fou
     « historique < 300 fenetres » du §6 interdirait tout reentrainement pendant
     des heures apres chaque redemarrage. Mais ces fenetres passees ne sont ni
     scorees, ni alertees, ni donnees au moniteur de derive : alerter sur un
     evenement vieux d'une heure serait une fausse alerte en temps reel, que le
     §7 compterait comme telle.
  2. **La reference du moniteur vient du registre.** `charger_reference(version)`
     telecharge les features d'entrainement du champion, et c'est elles que
     `strategy.on_fit` recoit. Recaler le moniteur sur les fenetres du moment
     reviendrait a comparer le present a lui-meme : une derive survenue pendant
     un arret du service serait invisible, definitivement, et rien ne le
     signalerait. Le service **refuse de demarrer** sur un champion sans
     reference (tag `reference_n_windows = 0`).
  3. **Pas d'extrapolation au bord.** Un trou court et interieur est interpole,
     avec la meme borne que le chargeur de sessions (`max_points_consecutifs`) ;
     un point manquant au bord — le plus recent, celui qui declencherait
     l'alerte — n'a pas de voisin droit, donc la fenetre est **sautee** et
     journalisee. Un scrape rate coute ainsi une fenetre, pas `window_size`.
  4. **Le registre fait foi.** Si l'enregistrement ou le deplacement de l'alias
     echoue, le challenger est jete et le champion reste en service : promouvoir
     en memoire un modele que le registre ignore ferait affirmer aux captures du
     chapitre 5 quelque chose de faux.
- `JournalEvenements` : le journal JSON lignes du §9
  (`{ts, run_id, strategy, event, details}`), vide ligne par ligne pour etre lu
  en direct. Un nom d'evenement hors vocabulaire est **refuse** : Grafana et
  Telegram filtrent sur ces noms, un nom invente y serait invisible. Vocabulaire
  = les sept evenements du §9 plus cinq propres au mode en ligne
  (`service_start`, `service_stop`, `window_skipped`, `collect_gap`,
  `registry_error`).
- `MesuresFenetre` : `window_ms` moyen et p95 mesures de la lecture du tampon a
  la decision de la strategie, reentrainement exclu. C'est **la** mesure que le
  rejeu ne peut pas donner (il score par blocs, d'ou le seul
  `window_ms_amortized` de `summary.csv`) et c'est elle qui alimente les
  indicateurs I7 / T1 du chapitre 2 : traitement < intervalle de collecte.
- `src/strategies/retraining.py` : `run_retrain_cycle`, le cycle de
  reentrainement partage entre le harnais de rejeu et la boucle en ligne
  (proposition -> entrainement du challenger -> statistiques sur la fenetre
  reservee -> verdict). `src/evaluation/replay.py` y delegue desormais : c'est ce
  partage qui rend le chapitre 5 comparable au chapitre 6.
- `configs/serving.yaml` : section `boucle:` (`retard_s`, `rattrapage_max_points`,
  `prefill_fenetres`, `historique_max_fenetres`), section `journal:` et la source
  et la strategie par defaut du service. Chaque valeur porte sa justification.
- `mdrift serve` : lance le service. `--une-fois` fait un seul tick (verification
  de configuration), `--sans-promotion` reentraine et valide sans rien mettre en
  service (repetition avant demonstration), `--duree-min` borne la session.
  A lancer apres `mdrift session-check` et `mdrift registry-info`.
- 36 tests nouveaux (`tests/test_loop.py`, 5 dans `tests/test_registry.py`),
  dont celui qui compare les features de la boucle a celles que `make_windows`
  produit sur les memes points bruts : c'est le seul endroit ou une divergence
  entre le mode en ligne et le rejeu se verrait. Total 364.

## Phase C, ce qui est en place

- `src/serving/exporter.py` : `ExporteurDetection`, les series que Prometheus
  lit sur `/metrics`. Trois choix structurent ce module, et aucun n'est
  rattrapable apres les captures du chapitre 5 :
  1. **L'exporter s'abonne au journal, il ne compte pas en parallele.**
     `mdrift_events_total{event=...}` est incremente depuis
     `JournalEvenements.ecrire` via une liste d'abonnes. Deux comptages
     independants finissent toujours par diverger, et une capture Grafana qui
     contredirait le journal remis au jury ne serait defendable ni dans un sens
     ni dans l'autre. Consequence voulue : le vocabulaire du §9 **est** l'ensemble
     des valeurs de l'etiquette `event`, et un evenement refuse par le journal ne
     peut laisser aucune trace dans les metriques.
  2. **Toutes les series valent zero des la construction.** Dans Grafana, une
     serie absente et une serie a zero se dessinent pareil : un panneau
     « derives confirmees » vide signifierait indifferemment « aucune derive » ou
     « service tombe ». Meme raison que les jauges d'injection de l'appli de test
     a M4.
  3. **`mdrift_last_window_timestamp_seconds` porte l'instant de la fenetre, pas
     celui du scrape.** Le tableau de bord en fait `time() - jauge`, qui est le
     retard reel de la detection. `up == 1` ne dit rien de l'avancement de la
     boucle : un service vivant mais bloque sur un trou de collecte resterait
     vert.
  Les etiquettes sont bornees a `event`, `strategy` et `version` — jamais une
  etiquette par fenetre ou par feature : l'explosion de cardinalite est la
  premiere cause de saturation du disque de Prometheus (§4.3), et `/metrics`
  n'expose aucune valeur de feature ni aucun nom de metrique du serveur.
- `src/serving/evenements.py` : le vocabulaire d'evenements extrait dans son
  propre module. C'est un **contrat** entre le journal, l'exporter, Telegram et
  Grafana ; le laisser dans `loop.py` imposait un import en anneau
  (`loop` -> `exporter` -> `loop`). La boucle en reste un producteur, pas le
  proprietaire.
- `infra/grafana/dashboards/pipeline.json` (genere par
  `infra/grafana/build_dashboard.py`, 15 panneaux) : etat du service, detection
  (score et seuil, alertes/h), surveillance de derive (part de features, suite
  consecutive, signaux contre confirmations), reentrainement / validation / mise
  en service, temps reel et sante de la boucle (I7 / T1). Deux fichiers et non un
  seul : `infrastructure.json` montre la **cause** (ce que fait le serveur),
  `pipeline.json` l'**effet** (ce que le detecteur en conclut) ; meme source et
  meme fenetre de temps, mais une capture de 25 panneaux est illisible en
  impression et les deux publics ne sont pas les memes.
- `infra/detection/Dockerfile` + service `detection` dans `docker-compose.yml` :
  aucun port publie (Prometheus joint la cible par le reseau interne), journal
  sur un **volume nomme** et non un montage hote — un montage est cree avec la
  propriete de l'hote alors que le conteneur tourne en uid 10003, et la premiere
  ligne de journal echouerait en « permission denied ». Recuperation des traces :
  `docker compose cp detection:/srv/results/online ./results/`. L'image installe
  `mlflow-skinny==2.22.4`, **la meme version** que le serveur du service
  `mlflow` : client et serveur partagent le schema de la base.
- `configs/serving.yaml` : section `exporter:` (`actif`, `hote`, `port`), avec
  priorite a `DETECTION_EXPORTER_HOST` / `DETECTION_EXPORTER_PORT`. Dans un
  conteneur il faut ecouter sur `0.0.0.0` pour que Prometheus atteigne la cible,
  sur un poste `127.0.0.1` pour ne rien exposer : aucune des deux valeurs n'a sa
  place en dur dans un fichier suivi.
- `mdrift serve` ouvre `/metrics` **avant** le premier tick et sort en code 2 si
  le port est pris : un service qui detecte sans etre observable ne produit
  aucune capture, et l'echec doit etre immediat. `--sans-exporter` pour un essai
  local.
- 47 tests nouveaux (`tests/test_exporter.py`, 12 dans
  `tests/test_infra_stack.py`). Les plus utiles verifient ce qu'aucune relecture
  ne verrait : que chaque serie `mdrift_*` citee dans une expression PromQL du
  tableau de bord existe bien dans le rendu de l'exporter, que chaque valeur
  d'etiquette `event` appartient au vocabulaire du journal, et que
  `prometheus.yml`, Compose et `serving.yaml` s'accordent sur le port 8001.
  Total 411.

Ce qui reste a verifier sur le serveur, et que seul Sidy peut faire :
`docker compose build detection` puis `docker compose up -d detection`, et la
cible `detection` en UP dans Prometheus (Status -> Targets).

## Phase D, ce qui est en place

- `src/notify/telegram.py` : `NotificateurTelegram`, second abonne du journal
  d'evenements apres l'exporter. Quatre choix portent cette phase :
  1. **Les messages sont construits depuis le journal, pas a cote.** Il n'y a
     nulle part dans le code un « message Telegram » : il n'y a que des
     evenements journalises dont certains sont pousses. Les captures Telegram du
     chapitre 5 se retrouvent donc ligne par ligne dans le journal remis avec le
     memoire.
  2. **Seules des cles autorisees entrent dans un message** (`CLES_PUBLIABLES`,
     une liste par type d'evenement). Un formateur qui deverserait `details` en
     entier finirait par pousser un chemin, un nom d'hote ou une URL vers un
     serveur tiers des que la boucle journaliserait une erreur de collecte. Le
     §4.3 l'interdit sans condition et une fuite vers l'API Telegram n'est pas
     rattrapable : c'est le point le plus grave du module, et celui que les
     tests attaquent le plus directement.
  3. **Regroupement : au plus un message par minute et par type** (§9). Le
     premier evenement d'un type part **immediatement** — une alerte retardee
     d'une minute n'est plus une alerte en temps reel — et les suivants sont
     agreges (« 39 anomalies en 60 s. Derniere : ... »). Le compte vient avant
     le detail parce que c'est lui qui decide d'agir. Sans cette regle, une
     saturation produirait une alerte par fenetre, et l'API Telegram elle-meme
     limite a une vingtaine de messages par minute et par conversation.
  4. **Aucun `parse_mode`.** En Markdown ou en HTML, un `_`, un `*` ou un `<`
     dans un identifiant ou un message d'erreur fait rejeter l'appel avec un
     400 : la notification serait perdue au moment ou elle importe le plus.
  L'envoi est un POST JSON par `urllib` de la bibliotheque standard : aucune
  dependance nouvelle, `python-telegram-bot` n'etant justifie que par les
  boutons du bonus (§3). Un echec d'envoi est **compte**, jamais leve : une
  coupure de reseau mobile ne doit pas arreter un service de detection. Le
  decompte des echecs s'affiche a l'arret, pour qu'un silence de Telegram ne
  passe pas pour « aucun evenement ».
- `src/notify/base.py` : le `Protocol` `Notificateur` (`observer_evenement`,
  `rincer`). La boucle ne connait pas Telegram, et le test de la boucle n'a pas
  a construire un notificateur reel pour verifier le rincage.
- `src/serving/loop.py` : la boucle rince les groupes **a chaque tick, avant
  toute sortie anticipee**, et **force** le rincage a l'arret. Sans le premier,
  le dernier evenement d'une rafale attendrait le prochain du meme type, donc
  peut-etre la fin du service ; sans le second, le resume de session — le
  message qui clot une demonstration — ne partirait jamais.
- `configs/notify.yaml` : `actif: false` par defaut (un depot clone ne doit pas
  appeler une API externe au premier `serve`, une CI encore moins), la liste des
  evenements pousses, `regroupement_s: 60`, la liste `silencieux`
  (`disable_notification` : consultables dans l'historique de la conversation,
  ce qui fait de bonnes captures, sans reveiller personne) et `timeout_s: 5`.
  Trois exclusions sont justifiees dans le fichier : `drift_signal` (present a
  chaque fenetre pendant des heures sans confirmation), `retrain_start`
  (toujours suivi d'un `retrain_end` ou d'un rejet quelques secondes plus tard)
  et `window_skipped` (un scrape rate coute une fenetre et se lit dans Grafana).
- Jeton et conversation **uniquement** par `TELEGRAM_BOT_TOKEN` et
  `TELEGRAM_CHAT_ID` (§4.3), passes au conteneur `detection` avec des valeurs
  vides par defaut : la pile demarre sans Telegram. `ReglagesTelegram` masque le
  jeton dans son `repr` pour qu'une trace de pile ne le recopie pas.
- `mdrift notify-test` : envoie un evenement de verification et s'arrete. A
  lancer avant une demonstration, comme `session-check` pour la collecte et
  `registry-info` pour le registre. Il passe par le chemin reel — meme
  redaction, meme transport — pour qu'un succes signifie vraiment que la
  demonstration sonnera.
- 30 tests nouveaux (`tests/test_telegram.py`, 1 dans `tests/test_loop.py`,
  2 dans `tests/test_infra_stack.py`). Aucun n'ouvre de connexion : le transport
  est injecte et l'horloge est pilotee par le test — un test qui attendrait
  reellement soixante secondes ne serait jamais lance. Total 441.

Ce qui reste a verifier sur le serveur, et que seul Sidy peut faire : creer le
bot avec @BotFather, mettre `TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID` dans
`.env`, passer `telegram.actif` a `true`, puis `uv run mdrift notify-test`.

## Commandes de reprise

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```
