# memoire-drift

Code experimental du memoire de Master 2 (IUA) : **detection d'anomalies en temps reel sur une
infrastructure applicative sous derive de concept — conception et evaluation d'une strategie de
reentrainement dans un pipeline MLOps**.

Le depot repond a une question : une strategie de reentrainement **declenchee par une derive
confirmee**, avec validation avant mise en service, maintient-elle la performance d'un detecteur
d'anomalies mieux qu'un modele fige, et a moindre cout qu'un reentrainement periodique ?

Le cadrage complet du projet est dans [CLAUDE.md](CLAUDE.md).

## Installation

Prerequis : [`uv`](https://docs.astral.sh/uv/). Python 3.11 est installe automatiquement.

```bash
uv sync
uv run mdrift --help
```

## Commandes

| Commande | Role | Jalon |
|---|---|---|
| `uv run mdrift version` | Verifie que l'environnement repond | M0 |
| `uv run mdrift smd-download` | Telecharge les machines SMD retenues | M0 |
| `uv run mdrift smd-info` | Decrit les donnees chargees : tailles, durees, anomalies | M0 |
| `uv run mdrift baseline` | Performance de reference en regime stable (B0, sans derive) | M0 |
| `uv run mdrift smd-screening` | Quantifie la derive naturelle des 28 machines (partage dev / eval) | M0+ |
| `uv run mdrift calibration` | Mesure les etapes 3 et 4 de la strategie proposee (D13, D14) | M0+ |
| `uv run mdrift inject` | Decrit la derive injectee par machine, scenario et graine | M1 |
| `uv run mdrift calibration-k` | Calibre l'amplitude *k* de la derive injectee (D22) | M1 |
| `uv run mdrift scenario-figures` | Figures des quatre scenarios de derive | M1 |
| `uv run mdrift run` | Rejoue un flux pour une strategie : un rejeu, une ligne de resultat | M2 |
| `uv run mdrift campaign` | Enchaine machines x scenarios x strategies x graines, reprenable | M2 |
| `uv run mdrift calibrate-drift` | Balaie la grille des parametres de derive sur le perimetre de mise au point | M2 |
| `uv run mdrift promotions` | Recalcule les promotions degradantes hors ligne depuis les traces | M2 |
| `uv run mdrift hypotheses` | Teste H1, H2 et H3 (Wilcoxon apparie, Holm par famille) | M3 |
| `uv run mdrift figures-m3` | Figures de comparaison et tableaux du chapitre 6 | M3 |
| `uv run mdrift figures` | Figures PNG 300 dpi a partir des traces enregistrees | M0 |
| `docker compose up -d` | Lance la pile de collecte : appli de test, Prometheus, node_exporter, cAdvisor, Grafana | M4 |
| `python infra/grafana/build_dashboard.py` | Reengendre les deux tableaux de bord Grafana — infrastructure et pipeline (ne pas les editer a la main) | M4, M5 |
| `uv run mdrift chaos <type>` | Injecte une anomalie bornee et l'horodate dans `data/labels/` | M4 |
| `uv run mdrift chaos-list` | Liste les conteneurs d'injection en cours | M4 |
| `make stop-chaos` | Arret d'urgence : tue toute injection en cours (§4.3) | M4 |
| `uv run mdrift drift-note <type>` | Horodate une derive reelle (deploiement, changement de ressources) | M4 |
| `uv run mdrift session-check` | Verifie que la pile est prete : cibles `up`, et chaque metrique rend une valeur | M4 |
| `uv run mdrift session-record <run_id> --debut -3h` | Enregistre une plage de Prometheus dans `data/sessions/<run_id>.parquet` | M4 |
| `uv run mdrift session-info <run_id>` | Decrit la session chargee : duree, fenetres, anomalies, derives, trous combles | M4 |
| `uv run mdrift registry-init <run_id>` | Entraine le modele initial sur la part calme d'une session et le promeut `champion` | M5 |
| `uv run mdrift registry-info` | Dit quel modele est en service, avec sa signature de features et l'historique des versions | M5 |
| `uv run mdrift serve` | Lance le service de detection en ligne : Prometheus -> score -> derive -> reentrainement -> promotion, avec exporter `/metrics` et notifications | M5 |
| `uv run mdrift notify-test` | Envoie un message de verification sur Telegram et s'arrete (a lancer avant une demonstration) | M5 |

Les commandes de calibrage (`smd-screening`, `calibration`, `calibration-k`, `calibrate-drift`)
n'ecrivent **jamais** dans `results/summary.csv` : elles ont leur propre fichier. Une campagne de mise
au point passe `--summary-path`, pour que la synthese du memoire ne contienne que des executions a
configuration figee.

### Donnees

Le SMD (Su et al., KDD 2019, licence MIT) est echantillonne **a la minute**, 38 metriques par
machine. `configs/datasets/smd.yaml` declare le perimetre de developpement courant, 3 machines, une
par groupe. Deux perimetres disjoints encadrent les experiences : **16 machines de mise au point**
(`smd_dev.yaml`), sur lesquelles tous les parametres sont regles, et **12 machines d'evaluation**
(`smd_eval.yaml`), sur lesquelles rien n'est execute avant que les parametres soient figes. Aucun
reglage ne regarde les etiquettes d'une machine d'evaluation. Les donnees brutes ne sont pas
versionnees ; un manifeste `data/raw/smd/manifest.json` enregistre l'empreinte SHA-256 de chaque
fichier utilise.

## Developpement

```bash
uv run ruff check .    # lint
uv run pytest          # tests
```

Sur Linux, les trois controles de l'integration continue se lancent par leurs cibles de Makefile,
qui sont exactement celles que la CI appelle :

```bash
make lint     # ruff check + ruff format --check
make test     # pytest
make build    # wheel installe dans un environnement neuf, point d'entree appele
```

### Integration continue

Deux declencheurs, un seul controle. `.gitlab-ci.yml` est le livrable demande par le cadrage
(jalon M7) et tourne tel quel sur une instance GitLab ; `.github/workflows/ci.yml` est celui qui
s'execute reellement, le depot etant heberge sur GitHub. Les deux appellent les memes cibles du
Makefile, et un test (`tests/test_ci.py`) verifie qu'ils ne divergent pas. La CI ne rejoue pas la
campagne (plusieurs heures) et ne telecharge pas le SMD : le seul test qui exige les donnees se
declare `skipped`.

## Organisation

- `configs/` — tous les parametres (aucune valeur en dur dans le code)
- `datasets/`, `scenarios/` — chargement des donnees, injection de derive
- `src/` — features, modeles, detecteurs de derive, strategies, evaluation, service en ligne
- `results/` — `summary.csv` et un dossier par execution
- `docs/` — journal des choix techniques, journal d'avancement et figures

## Resultats

Chaque execution ecrit `results/runs/<source>/<machine>/<scenario>/<strategie>/<seed>/` avec la
configuration complete, les metriques, le journal d'evenements et les traces, et met a jour sa
ligne dans `results/summary.csv`. Un `run_id` est stable : rejouer une experience met sa ligne a
jour au lieu de la dupliquer.
