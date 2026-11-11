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
| B | Boucle en ligne : lecture Prometheus fenetre par fenetre, features, score, seuil, strategie, reentrainement / validation / promotion, journal JSON lignes, mesure `window_ms` | a faire |
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

## Commandes de reprise

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```
