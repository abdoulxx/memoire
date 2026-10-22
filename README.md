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

| Commande | Role |
|---|---|
| `uv run mdrift version` | Verifie que l'environnement repond |
| `uv run mdrift smd-download` | Telecharge les machines SMD retenues (55,5 Mo pour 3 machines) |
| `uv run mdrift smd-info` | Decrit les donnees chargees : tailles, durees, anomalies |
| `uv run mdrift baseline` | Performance de reference en regime stable (strategie B0, sans derive) |
| `uv run mdrift figures` | Figures PNG 300 dpi a partir des traces enregistrees |

(La table se remplit au fil des jalons : une sous-commande par experience.)

### Donnees

Le SMD (Su et al., KDD 2019, licence MIT) est echantillonne **a la minute**, 38 metriques par
machine. Le perimetre courant est de 3 machines, une par groupe, declarees dans
`configs/datasets/smd.yaml`. Les donnees brutes ne sont pas versionnees ; un manifeste
`data/raw/smd/manifest.json` enregistre l'empreinte SHA-256 de chaque fichier utilise.

## Developpement

```bash
uv run ruff check .    # lint
uv run pytest          # tests
```

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
