# Tableaux du chapitre 6

> Fichier **engendre** par `mdrift figures-m3`. Ne pas le modifier a la main :
> toute correction se fait dans `src/evaluation/result_tables.py`, puis on
> relance la commande. Source : `results/summary.csv`,
> `results/hypotheses.csv`, `results/summary_m3_derive_naturelle.csv`.

Campagne M3 : **12 machines d'evaluation** x 4 scenarios x 4 strategies x **5 graines** = 960 rejeux avec derive injectee.

Toutes les valeurs sont des **moyennes sur les machines ± erreur standard
inter-machines**, apres moyenne des graines machine par machine (OBS06). La
dispersion repond a « une autre machine donnerait-elle autre chose ? ».

## Tableau 1 — Synthese par strategie, les quatre scenarios confondus

| Metrique | B0 — modele fige | B1 — periodique | P — proposee | A1 — P sans validation |
|---|---|---|---|---|
| F1 par fenetre | 0.265 ± 0.033 | 0.252 ± 0.036 | 0.277 ± 0.033 | 0.277 ± 0.033 |
| F1 avant la derive | 0.177 ± 0.048 | 0.191 ± 0.049 | 0.177 ± 0.048 | 0.177 ± 0.048 |
| F1 apres la derive | 0.306 ± 0.035 | 0.324 ± 0.045 | 0.326 ± 0.035 | 0.326 ± 0.035 |
| Rappel par evenement | 0.773 ± 0.064 | 0.708 ± 0.065 | 0.742 ± 0.066 | 0.742 ± 0.066 |
| Fausses alertes / h | 3.21 ± 0.55 | 1.31 ± 0.19 | 2.07 ± 0.29 | 2.07 ± 0.29 |
| Delai de detection, moyen (min) | 12.6 ± 6.6 | 13.8 ± 5.0 | 13.3 ± 6.3 | 13.3 ± 6.3 |
| Delai de detection, median (min) | 2.0 ± 0.5 | 3.3 ± 0.7 | 2.5 ± 0.5 | 2.5 ± 0.5 |
| Temps de recuperation (min) | 2561 ± 458 | 2554 ± 431 | 2410 ± 351 | 2410 ± 351 |
| Delai de detection de la derive (min) | — | — | 4029 ± 330 | 4029 ± 330 |
| Reentrainements | 0.00 ± 0.00 | 16.25 ± 0.25 | 1.54 ± 0.23 | 1.54 ± 0.23 |
| dont promus | 0.00 ± 0.00 | 16.25 ± 0.25 | 1.54 ± 0.23 | 1.54 ± 0.23 |
| dont rejetes | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Secondes CPU de reentrainement | 0.00 ± 0.00 | 3.94 ± 0.08 | 0.44 ± 0.07 | 0.38 ± 0.06 |
| Promotions degradantes | 0.000 ± 0.000 | 0.517 ± 0.118 | 0.158 ± 0.065 | 0.158 ± 0.065 |

## Tableau 2 — F1 apres la derive, par scenario

| Scenario | B0 — modele fige | B1 — periodique | P — proposee | A1 — P sans validation |
|---|---|---|---|---|
| S1 soudaine | 0.301 ± 0.039 | 0.305 ± 0.047 | 0.325 ± 0.042 | 0.325 ± 0.042 |
| S2 incrementale | 0.317 ± 0.037 | 0.353 ± 0.048 | 0.339 ± 0.035 | 0.339 ± 0.035 |
| S3 graduelle | 0.293 ± 0.038 | 0.320 ± 0.043 | 0.300 ± 0.033 | 0.300 ± 0.033 |
| S4 recurrente | 0.311 ± 0.032 | 0.317 ± 0.045 | 0.338 ± 0.039 | 0.338 ± 0.039 |

## Tableau 3 — Cout : reentrainements declenches, par scenario

| Scenario | B0 — modele fige | B1 — periodique | P — proposee | A1 — P sans validation |
|---|---|---|---|---|
| S1 soudaine | 0.00 ± 0.00 | 16.25 ± 0.25 | 1.48 ± 0.18 | 1.48 ± 0.18 |
| S2 incrementale | 0.00 ± 0.00 | 16.25 ± 0.25 | 2.13 ± 0.38 | 2.13 ± 0.38 |
| S3 graduelle | 0.00 ± 0.00 | 16.25 ± 0.25 | 2.32 ± 0.35 | 2.32 ± 0.35 |
| S4 recurrente | 0.00 ± 0.00 | 16.25 ± 0.25 | 0.23 ± 0.14 | 0.23 ± 0.14 |

## Tableau 4 — Tests d'hypotheses (Wilcoxon apparie, Holm par famille)

48 paires machine x scenario. `paires +/-` compte les paires ou P est au-dessus / en dessous,
ex aequo exclus ; la direction est lue sur la somme des rangs signes, pas sur la mediane des ecarts.

| Hypothese | Metrique | Attendu | Observe | paires +/- | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| H1 | `event_recall` | superieur | inferieur | 2/17 | 0.0016 | contraire a l'attendu |
| H1 | `f1` | superieur | superieur | 22/14 | 0.8561 | pas de difference detectable |
| H1 | `f1_after` | superieur | superieur | 22/14 | 0.8561 | pas de difference detectable |
| H1 | `false_alerts_per_hour` | inferieur | inferieur | 4/32 | 0.0000 | conforme |
| H1 | `recovery_minutes` | inferieur | inferieur | 4/6 | 0.8561 | pas de difference detectable |
| H2 | `cost_n_retrains` | inferieur | inferieur | 0/48 | 0.0000 | conforme |
| H2 | `cost_retrain_seconds` | inferieur | inferieur | 0/48 | 0.0000 | conforme |
| H2 | `f1` | aucun | superieur | 26/22 | 0.2732 | pas de difference detectable |
| H2 | `f1_after` | aucun | superieur | 24/24 | 0.7643 | pas de difference detectable |
| H2 | `false_alerts_per_hour` | aucun | superieur | 42/6 | 0.0000 | difference detectee (superieur) |
| H3 | `cost_n_promoted` | inferieur | egal | 0/0 | — | indecidable |
| H3 | `f1` | aucun | egal | 0/0 | — | indecidable |
| H3 | `f1_after` | aucun | egal | 0/0 | — | indecidable |
| H3 | `n_promotions_degradantes` | inferieur | egal | 0/0 | — | indecidable |

## Tableau 5 — Derive naturelle (scenario `none`, sans injection)

Les 28 machines du SMD, graine 42. Sur une machine stable, la bonne conduite est de ne rien faire.

**Lecture de la colonne « confirmations »** : sur les machines stables, toute confirmation est fausse
par construction — c'est la metrique de specificite du §7, et elle vaut **0** la ou K = 30 en donnait 13
sur 15 (D19). Sur les machines instables elle n'est pas fausse : le criblage a montre que ces machines
derivent reellement entre leur moitie train et leur moitie test (OBS03), et la confirmation y est le
comportement attendu. Les deux colonnes ne se lisent donc pas dans le meme sens.

| Perimetre | Strategie | F1 | Fausses alertes / h | Reentrainements | Confirmations de derive | Promotions degradantes |
|---|---|---|---|---|---|---|
| Stables (15 machines) | B0 — modele fige | 0.333 ± 0.049 | 0.58 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Stables (15 machines) | B1 — periodique | 0.295 ± 0.042 | 0.87 ± 0.13 | 16.20 ± 0.20 | 0.00 ± 0.00 | 0.33 ± 0.13 |
| Stables (15 machines) | P — proposee | 0.333 ± 0.049 | 0.58 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Stables (15 machines) | A1 — P sans validation | 0.333 ± 0.049 | 0.58 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Instables (13 machines) | B0 — modele fige | 0.241 ± 0.048 | 10.28 ± 2.95 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Instables (13 machines) | B1 — periodique | 0.274 ± 0.026 | 1.65 ± 0.28 | 17.85 ± 0.42 | 0.00 ± 0.00 | 0.54 ± 0.27 |
| Instables (13 machines) | P — proposee | 0.247 ± 0.049 | 6.60 ± 1.63 | 0.92 ± 0.38 | 0.92 ± 0.38 | 0.23 ± 0.17 |
| Instables (13 machines) | A1 — P sans validation | 0.253 ± 0.048 | 5.70 ± 1.28 | 1.15 ± 0.45 | 1.15 ± 0.45 | 0.23 ± 0.17 |
