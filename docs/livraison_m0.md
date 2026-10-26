# Livraison M0 — paquet de faits pour la rédaction du mémoire

> Destinataire : le Claude qui rédige le mémoire. Document factuel, à lire avec
> `docs/decisions.md` et `docs/journal.md`.
> Jalons couverts : **M0** (socle + performance de référence B0), **M0+** (mesures
> préparatoires sur 28 machines), **M0++** (calibrage des étapes 3 et 4 de la stratégie P).
> Arrêté le **26 octobre 2026**. Aucun chiffre de ce document n'est saisi à la main :
> chaque ligne renvoie au fichier qui le produit.

---

## 1. Ce qui existe et ce qui n'existe pas

**Existe et est mesuré**
- Chargement du SMD, fenêtrage, features, Isolation Forest, seuil par percentile.
- Harnais de rejeu préquentiel, métriques du §7 du cadrage, stratégie B0 (modèle figé).
- Criblage des 28 machines : dérive naturelle, critère de stabilité, partage dev/éval.
- Calibrage des étapes 3 (exclusion) et 4 (garde-fou) de la stratégie proposée.
- 70 tests, `ruff` sans remarque.

**N'existe pas encore** — ne rien écrire au passé sur ces points
- Injection de dérive S1–S4 (M1). Aucun scénario n'a été exécuté.
- Stratégies B1 (périodique) et P (proposée) : **conçues et calibrées, pas implémentées**.
- Détecteurs de dérive PSI / KS / ADWIN en tant que modules : règle d'agrégation décidée
  (D18), code à écrire à M2. `river` et `mlflow` ne sont pas encore des dépendances.
- Tout le pipeline en ligne : Prometheus, Grafana, MLflow, Telegram, serveur personnel
  (M4 à M6). Aucune capture d'écran n'existe.
- Autoencodeur, assistant LLM, GitLab CI.

---

## 2. Environnement et paramètres

| Élément | Valeur | Source |
|---|---|---|
| Python | 3.11.15 (installé par `uv`) | `pyproject.toml`, `uv.lock` |
| numpy / pandas | 2.1.3 / 2.2.3 | `uv.lock` |
| scikit-learn / scipy | 1.6.1 / 1.15.3 | `uv.lock` |
| matplotlib | 3.10.9 | `uv.lock` |
| Fenêtre | `window_size: 10` points = 10 min, `stride: 1` | `configs/features.yaml` |
| Agrégats | `last`, `mean`, `std`, `max` → **152 features** (38 × 4) | `configs/features.yaml` |
| Normalisation | standardisation ajustée sur l'entraînement seul | `configs/features.yaml` |
| Fenêtre positive | au moins 1 point étiqueté anormal | `configs/features.yaml` |
| Modèle | Isolation Forest, `n_estimators: 100`, `max_samples: 256`, `n_jobs: -1` | `configs/model.yaml` |
| Seuil | 99ᵉ percentile des scores d'entraînement → 1 % d'alertes attendu | `configs/model.yaml` |
| Graine | 42 pour toutes les exécutions livrées | `results/summary.csv` |

`max_samples: 256` est la valeur de référence de Liu, Ting et Zhou (2008) ; c'est aussi
elle qui fonde `min_train_windows` du garde-fou (§5).

---

## 3. Données

### 3.1 Le jeu de données
Server Machine Dataset (SMD), Su et al., KDD 2019, via le dépôt `NetManAIOps/OmniAnomaly`,
licence MIT. 28 machines, 3 groupes, 38 métriques, moitié `train` sans étiquettes et moitié
`test` étiquetée point par point. Intervalle d'échantillonnage **1 minute** : il n'est pas
écrit dans le README du dépôt, il est établi par Garg et al. (IEEE TNNLS, 2022) et vérifié
empiriquement (décision D06).

### 3.2 Les 28 machines, chiffres d'ensemble
Source : `results/smd_screening.csv`, produit par `uv run mdrift smd-screening` (≈ 35 s).

| Grandeur | Valeur |
|---|---|
| Fenêtres de test, total | 708 168 |
| Durée de test, total | 11 803 h (de 394,6 à 478,9 h par machine) |
| Événements d'anomalie étiquetés | 327 |
| Fenêtres positives | médiane 3,94 % par machine (de 0,69 % à 16,80 %) |

### 3.3 La dérive naturelle, et le partage dev / éval
La moitié `test` diffère déjà de la moitié `train` sur **13 machines sur 28** : le taux de
**fausses** alertes de B0 y dépasse 3× le taux calibré de 1 % (jusqu'à **48,3×** sur
machine-1-6). Les 15 autres sont dites stables. Le critère porte sur les fausses alertes et
non sur les alertes totales : une alerte sur une fenêtre positive est une détection réussie
(voir OBS04 ; compter toutes les alertes classait machine-3-10 instable à 3,95× alors que
ses fausses alertes ne font que 0,45 %).

Conséquence, décision **D17** — deux périmètres disjoints et figés dans des configs :

| Périmètre | Fichier | Machines |
|---|---|---|
| Mise au point (16) | `configs/datasets/smd_dev.yaml` | 1-1, 1-3, 1-5, 1-6, 1-8, 2-6, 2-7, 2-8, 2-9, 3-1, 3-5, 3-6, 3-8, 3-9, 3-10, 3-11 |
| Évaluation S1–S4 (12) | `configs/datasets/smd_eval.yaml` | 1-2, 1-4, 1-7, 2-1, 2-2, 2-3, 2-4, 2-5, 3-2, 3-3, 3-4, 3-7 |

= les 13 instables + 3 stables tirées avec la graine 42, contre les 12 stables restantes.
Tout paramètre de P est réglé sur le périmètre de mise au point et **figé avant** la
première exécution sur le périmètre d'évaluation. C'est un point de méthodologie à écrire
au 6.1.

**Les machines d'un même groupe ne sont pas indépendantes** (OBS06) : corrélation moyenne
par métrique de **+0,801** entre machine-3-1 et machine-3-9, contre −0,008 entre groupes.
28 machines ne font donc pas 28 répétitions indépendantes : à M3, agréger par machine sur
les graines avant tout test statistique, et rapporter aussi par groupe.

---

## 4. Résultat principal de M0 : la référence B0

Source : `results/summary.csv` (17 lignes, une par machine, graine 42), produit par
`uv run mdrift baseline --dataset-config datasets/smd_dev.yaml`. Traces complètes dans
`results/runs/smd/<machine>/none/static/seed42/`.

| Métrique (par fenêtre) | Valeur sur les 17 machines |
|---|---|
| Fenêtres évaluées | 442 604, soit 7 377 h |
| Fenêtres positives | 18 480, soit 4,18 % |
| F1 | médiane **0,2744**, moyenne 0,3017, de **0,0297** (machine-3-9) à **0,7905** (machine-3-10) |
| Précision | médiane 0,2436 |
| Rappel | médiane 0,5784 |
| Rappel par événement | médiane 0,9667, moyenne 0,7764 (126 événements détectés sur 181) |
| Fausses alertes par heure | médiane 3,30, maximum 28,96 |
| Délai de détection | médiane 1,0 min |

Lecture à proposer au chapitre 6 : un modèle figé, calibré pour 1 % d'alertes, atteint un
**rappel par événement élevé** (il voit presque toutes les anomalies) mais une **précision
très faible** (il alerte beaucoup à côté), et l'écart entre machines est d'un facteur 27 sur
le F1. C'est la référence basse que les stratégies adaptatives doivent battre.

**Coût de calcul** : 0,59 à 1,05 s par machine pour ≈ 24 000 fenêtres, soit un temps amorti
de **0,030 ms par fenêtre** en médiane. À utiliser pour l'indicateur « temps réel » du
chapitre 2 **avec la réserve suivante** : le rejeu score par blocs (D10), ce chiffre est donc
amorti et non une mesure par fenêtre. La vraie mesure (`window_ms_mean`, `window_ms_p95`)
viendra de la boucle en ligne à M5.

**Deux réserves sur ce tableau, à ne pas masquer**
1. **Une seule graine** (42). Le cadrage vise 5 graines par scénario et par machine ; ce
   sera le cas à partir de M1. Aucun écart-type inter-graines ne peut donc être cité sur ce
   tableau. Ordre de grandeur disponible par ailleurs : l'écart-type inter-graines du F1
   mesuré sur 473 challengers vaut 0,0120 en médiane et 0,0469 au 90ᵉ centile
   (`results/calibration_dev.csv`).
2. **machine-2-1 est dans ce fichier** alors qu'elle appartient au périmètre d'évaluation :
   son B0 a été exécuté le 22/10, avant que le partage n'existe. C'est sans conséquence —
   B0 n'a aucun paramètre réglé sur elle — mais il faut le dire plutôt que de le cacher,
   et ne pas réutiliser cette ligne dans un tableau de résultats d'évaluation.

---

## 5. Ce que M0++ a tranché sur la stratégie proposée

Source : `results/calibration_dev.csv`, produit par `uv run mdrift calibration` (≈ 1 h) :
16 machines × 5 instants de réentraînement = **78 décisions**, 8 règles d'exclusion
candidates, 3 graines, soit **473 challengers** entraînés puis évalués en aval.
Protocole : W = 2 000 fenêtres candidates, V = 500 fenêtres réservées à la validation,
évaluation sur tout l'aval (≥ 2 000 fenêtres).

### 5.1 Étape 3, l'exclusion des anomalies au réentraînement (D13)
**Décision : ne rien exclure** (`retrain.exclusion.method: none`).

| Règle | Optimale sur 78 | Écart médian au meilleur | Écart moyen | Perd > 0,10 de F1 |
|---|---|---|---|---|
| Aucune exclusion | **43** | **0,0000** | **0,0396** | 9 |
| Exclut top 1 % des scores | 15 | 0,0196 | 0,0419 | 8 |
| Exclut rafales < 10 fenêtres | 21 | 0,0078 | 0,0453 | 9 |
| Exclut toutes les alertes du champion | 19 (sur 76) | 0,0356 | 0,1059 | 24 |
| Exclut top 10 % des scores | 6 | 0,0634 | 0,1074 | 32 |

- Face à sa seule concurrente sérieuse (`top_percentile` à 1 %), ne rien exclure gagne
  **52 décisions sur 78** contre 24 — Wilcoxon apparié sur les 76 différences non nulles,
  **p = 0,0040**.
- Exclure toutes les alertes du champion **s'effondre sous la moitié du F1 sur 11 décisions** :
  sur machine-3-1 à 80 % du flux, elle ne garde que 25,3 % de la fenêtre et donne F1 0,0615
  contre 0,5527, soit **×0,11**, le challenger héritant de la saturation du champion (73,7 %
  d'alertes, 42,8 fausses alertes/h).
- **Mais aucune règle ne domine partout** : l'écart au meilleur dépasse 0,10 point de F1 sur
  9 décisions et atteint **0,7324** (machine-1-6 à 80 % du flux : 0,0771 sans exclusion
  contre 0,8095 en excluant les 5 % de scores les plus élevés).

À écrire comme un **résultat négatif** : une étape intuitive du cadrage n'apporte rien en
moyenne, sa version agressive nuit gravement, et son intérêt résiduel se limite à quelques
instants que rien ne permet d'identifier sans étiquettes.

### 5.2 Étape 4, la validation avant mise en service (D14)
**Décision : un garde-fou, pas un concours.** L'étape 4 rejette un challenger aberrant au
lieu de le comparer au champion.

Les deux critères sans étiquettes envisagés au cadrage ont été mesurés sur les 71 décisions
comportant au moins trois challengers distincts :

| Critère | Spearman médian avec le F1 aval | Signe inversé | Coût du choix |
|---|---|---|---|
| Écart au taux d'alertes cible | **−0,66** | 18 / 71 | 5,1 % du F1 atteignable en médiane, **jusqu'à 100 %** |
| KS scores entraînement / validation | −0,55 | 18 / 71 | — |

Le signe attendu est négatif, il l'est : le critère va dans le bon sens **en médiane**. Mais
il se trompe de sens une fois sur quatre et **sélectionne le pire candidat de la décision
11 fois sur 71**, soit une fois sur six. Un tel critère ne peut pas arbitrer une mise en
service automatique. Appui bibliographique : Ma, Zhao, Zhang et Akoglu (2023), *ACM SIGKDD
Explorations* 25(1), doi 10.1145/3606274.3606277, qui concluent qu'aucune stratégie
d'évaluation interne ne serait utile en pratique. `[À VÉRIFIER]` la pagination 19–35 n'a pas
été confirmée sur la source primaire (ACM renvoie une erreur 403).

**Bornes du garde-fou, calibrées — et elles contredisent les valeurs du cadrage**

| Paramètre | Cadrage | Retenu | Pourquoi |
|---|---|---|---|
| `min_train_windows` | 300 | **256** | `max_samples` du modèle ; entraîner une forêt sur moins de fenêtres que son propre échantillon n'a pas de sens |
| `alert_rate_min` | 0,1 % | **désactivée** | 98 challengers sur 473 n'alertent sur **aucune** fenêtre réservée, et 63 sont sains, avec un F1 aval médian de 0,4202 contre 0,2605 pour l'ensemble |
| `alert_rate_max` | 30 % | **0,90** | à 30 %, la borne rejetterait **37 challengers sains sur 285** pour seulement 22 dégradés sur 188. Les sains vont jusqu'à 87 % d'alertes légitimement : sur machine-1-6 à 80 % du flux, 60,1 % des fenêtres aval sont réellement positives |

**Conséquence assumée : le garde-fou ne rejette que 5 candidats sur 473, soit 1,1 %.** Aucune
borne intermédiaire ne sépare les sains des dégradés. Il faut donc l'écrire ainsi : en rejeu,
l'étape 4 est un filet de sécurité contre l'aberrant, et **l'ablation A1 devrait égaler P** —
c'est une prédiction testable, pas une excuse. L'apport de l'étape 4 est à chercher en mode
en ligne (M5), où un modèle dégénéré promu reste en service jusqu'à la confirmation suivante.

### 5.3 Résultat inattendu : l'instant du réentraînement fait tout (OBS08)
Le challenger sans exclusion ne bat le champion figé que **43 fois sur 78**, rapport de F1
médian **×1,04** (quartiles 0,85 et 1,37 ; de ×0,08 à ×12,13).

| Instant de réentraînement | Bat le champion | Rapport de F1 médian |
|---|---|---|
| Confirmation du moniteur naïf (taux d'alertes, K = 30) | **3 / 14** | **×0,78** |
| 20 % du flux | 11 / 16 | ×1,07 |
| 40 % du flux | 11 / 16 | ×1,20 |
| 60 % du flux | 8 / 16 | ×1,01 |
| 80 % du flux | 10 / 16 | ×1,38 |

Par machine, l'écart est énorme : ×5,13 sur machine-3-1 et ×3,42 sur machine-3-9, mais ×0,32
sur machine-3-10 et ×0,76 sur machine-3-8.

Deux conséquences pour la rédaction :
1. Le chiffre **×7,4** mesuré le 22/10 sur machine-3-1 est un cas favorable, **pas** le gain
   du réentraînement. Il ne doit pas être présenté comme tel.
2. **Réentraîner trop tôt nuit**, et le moniteur naïf déclenche précisément trop tôt. C'est
   l'argument empirique le plus direct en faveur de l'étape 2 de la stratégie proposée : la
   valeur de P n'est pas de réentraîner, c'est de réentraîner **au bon moment**. Les
   hypothèses doivent donc être testées contre B1 (périodique) et pas seulement contre B0.

### 5.4 La règle d'agrégation de la surveillance (D18)
Le signal de dérive est la **part des 152 features dont la statistique de KS dépasse un
seuil**, pas une correction de Bonferroni sur les p-valeurs. Bonferroni a été mesuré et ne
tient pas, pour deux raisons indépendantes :
- **aucune spécificité à cette taille d'échantillon** : sur machine-1-2, machine **stable**,
  115 features sur 152 rejettent à p < 0,05 et **109 rejettent encore au seuil de Bonferroni**
  (3,29 × 10⁻⁴) ;
- **les 152 tests ne sont pas indépendants**, hypothèse dont Bonferroni a besoin :
  corrélation absolue moyenne de 0,468 à 0,592 entre features issues d'une même métrique
  contre 0,090 à 0,168 entre métriques différentes, et rang effectif de 120 à 131 sur 152.

La **statistique**, elle, discrimine : part de features au-delà de KS 0,5 égale à **0,0 %**
sur machine-1-2 (stable), 9,2 % sur machine-2-1, **26,3 %** sur machine-3-1.

### 5.5 Nouvelle métrique : les fausses confirmations de dérive (D19)
Le délai de détection de la dérive ne mesure que la sensibilité : un moniteur qui confirme
tout le temps obtient un délai excellent. Il faut donc compter, à côté, le nombre de
confirmations de dérive sur les machines **stables sans injection** (scénario `none`).
Point de référence chiffré à battre : le moniteur naïf (taux d'alertes, K = 30) confirme une
dérive sur **26 des 28 machines, dont 13 des 15 machines stables**. K = 30 fenêtres ne valant
que 30 minutes, la persistance exigée est manifestement trop faible.

---

## 6. Figures disponibles

Dans `docs/figures/`, 300 dpi, 6 fichiers — pour 3 machines (1-1, 2-1, 3-1), scénario `none`,
stratégie `static`, graine 42 :
- `scores_machine-<m>_none_static_seed42.png` : score d'anomalie dans le temps, seuil et
  fenêtres étiquetées marqués ;
- `taux_alertes_machine-<m>_none_static_seed42.png` : taux d'alertes glissant.

**Aucune figure de dérive injectée, de F1 glissant comparé entre stratégies, ni de nombre de
réentraînements n'existe** : elles dépendent de M1 et M2. Figures à produire plus tard,
déjà identifiées : rapport de F1 challenger/champion en fonction de l'instant (OBS08), part
de features en dérive dans le temps pour une machine stable et une machine dérivante (D18),
tableau des 8 règles d'exclusion (D13).

---

## 7. Commandes reproductibles

```bash
uv run mdrift smd-download --dataset-config datasets/smd_dev.yaml
uv run mdrift smd-screening                                        # results/smd_screening.csv
uv run mdrift baseline --dataset-config datasets/smd_dev.yaml      # results/summary.csv
uv run mdrift calibration                                          # results/calibration_dev.csv
uv run pytest && uv run ruff check .                               # 70 tests
```

Chaque exécution écrit son `run_id`, sa graine et la copie intégrale de ses configs à côté de
ses résultats.

---

## 8. Problèmes rencontrés → matière pour les limites (6.3.3 et 7.3)

1. **Une erreur de mesure corrigée, à citer comme exemple de méthode.** La datation de la
   dérive naturelle avait été faite avec une moyenne glissante **centrée**, qui anticipe de la
   moitié de la fenêtre (+4,1 h mesurés), et en comptant **toutes** les alertes : le premier
   dépassement tombait en fait sur une anomalie étiquetée à 20,1 h. En moyenne **causale** et
   sur les **fausses alertes seules**, la dérive durable démarre à **93,5 h**. Deux garde-fous
   ont été ajoutés : `configs/screening.yaml` refuse toute autre méthode de mesure, et
   `tests/test_screening.py` reproduit le piège en test de non-régression.
2. **Un paramètre réglé sur une machine d'évaluation, puis corrigé.** D13 et D14 avaient été
   décidés en partie sur machine-2-1. Les deux mesures ont été refaites sur le seul périmètre
   de mise au point ; les deux décisions tiennent, mais **deux affirmations fausses ont dû
   être retirées** (l'équivalence des règles d'exclusion, le sens inversé de la corrélation).
   À raconter tel quel : c'est ce que le partage dev/éval sert à éviter.
3. **Un défaut de protocole détecté avant conclusion.** Ne visiter qu'un instant de
   réentraînement par machine rend les règles d'exclusion indistinguables (le champion n'a
   presque pas encore alerté), et les modèles dupliqués gonflent les corrélations par des ex
   æquo. Le module visite donc cinq instants et fusionne les règles qui gardent exactement les
   mêmes fenêtres.
4. **Lecture des fichiers SMD** : `np.loadtxt` était trop lent pour 56 fichiers de 17 Mo.
   Remplacé par `pandas.read_csv`, contrôles inchangés : 0,8 à 1,9 s par machine (OBS07,
   résolue).
5. **Limites qui restent** : une seule graine sur le tableau B0 ; machines d'un même groupe
   corrélées, donc puissance statistique surestimée si l'on traite 28 machines comme 28
   répétitions ; le SMD étiquette les anomalies et **pas** les dérives, donc l'instant de
   dérive est connu uniquement parce que nous l'injectons ; le garde-fou de l'étape 4 ne
   rejette que 1,1 % des candidats en rejeu.

---

## 9. Ce qui vient ensuite

| Jalon | Contenu | État |
|---|---|---|
| M1 | Injection S1–S4 avec graines, sur `smd_eval.yaml` | non commencé |
| M2 | Modules de dérive, B1 et P, métriques d'adaptation | non commencé |
| M3 | Campagne complète machines × scénarios × stratégies × graines (5 graines) | non commencé |
| M4–M6 | Serveur personnel, boucle en ligne, test final | non commencé |

Les paramètres de P restant à calibrer sur le périmètre de mise au point avant M1/M2 : *k*
(amplitude de l'injection), *K* (persistance de la confirmation), *W*, *C*, le seuil par
feature et la part déclenchante de D18, et le δ des promotions dégradantes (à calibrer sur
l'écart-type inter-graines, dont l'ordre de grandeur mesuré est 0,0120 en médiane).
