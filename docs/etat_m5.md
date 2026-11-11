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
| C | Exporter Prometheus du service de detection + tableau de bord Grafana du pipeline | a faire |
| D | Notifications Telegram : un message par minute et par type, construites depuis le journal d'evenements | a faire |
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

## Commandes de reprise

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```
