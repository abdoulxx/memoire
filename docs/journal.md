# Journal d'avancement

> Dates de debut et de fin de chaque jalon -> planification et diagramme de Gantt (memoire, 2.6).

| Jalon | Debut | Fin | Etat |
|---|---|---|---|
| M0 — Depot, SMD, features, Isolation Forest, evaluation sans derive | 2026-10-22 | 2026-10-22 | termine |

## 2026-10-22
- Initialisation du depot : arborescence, `pyproject.toml`, `uv.lock`, `ruff`, `pytest`, CLI `mdrift`.
- Python 3.11.15 installe via `uv`. Environnement verifie : scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, scipy 1.15.3.
- SMD : telechargement selectif de 3 machines (une par groupe), 55,5 Mo, manifeste SHA-256.
- Donnees verifiees : 38 metriques, intervalle d'une minute (source + verification empirique).
  | machine | train | test | duree | anomalies | evenements |
  |---|---|---|---|---|---|
  | machine-1-1 | 28 479 | 28 479 | 39,6 j | 9,46 % | 8 |
  | machine-2-1 | 23 693 | 23 694 | 32,9 j | 4,94 % | 13 |
  | machine-3-1 | 28 700 | 28 700 | 39,9 j | 1,07 % | 4 |
- Deux limites consignees (OBS01 normalisation prealable du dataset, OBS02 nombre de metriques).
- Etapes 1 et 2 de M0 terminees. 7 tests passent, lint propre.

- Features : fenetres de 10 min, pas de 1 min, 152 features par fenetre (38 metriques x 4 agregats).
- Isolation Forest (100 arbres, max_samples 256), seuil au 99e percentile des scores d'entrainement.
- Metriques implementees et testees sur cas calcules a la main : precision, rappel, F1 par fenetre,
  rappel par evenement, delai de detection, fausses alertes par heure, cout du reentrainement.
- Harnais de rejeu prequentiel : meme seed = meme resultat (verifie par test), le reentrainement
  ne voit que l'historique deja observe (verifie par test).
- **Resultat M0 — performance de reference en regime stable, strategie B0, seed 42** (6.2.1) :

  | machine | fenetres | precision | rappel | F1 | rappel par evenement | delai moyen (min) | fausses alertes/h |
  |---|---|---|---|---|---|---|---|
  | machine-1-1 | 28 470 | 0,311 | 0,413 | 0,355 | 0,63 (5/8) | 259,0 | 5,32 |
  | machine-2-1 | 23 685 | 0,447 | 0,184 | 0,261 | 0,85 (11/13) | 15,2 | 0,74 |
  | machine-3-1 | 28 691 | 0,016 | 0,619 | 0,032 | 1,00 (4/4) | 25,3 | 26,78 |

- Observation majeure consignee (OBS03) : derive naturelle entre les deux moities du SMD.
- 44 tests passent, lint propre. Duree d'un rejeu : 1 a 1,5 s par machine.

## Prochaine etape : M1
Injection de derive S1 a S4 avec seeds, et quantification de la derive naturelle du SMD.
