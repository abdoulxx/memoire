# CLAUDE.md — Projet expérimental du mémoire

> Fichier de cadrage pour Claude Code (et pour moi). À placer à la racine du dépôt.
> Mémoire de Master 2 IUA — SAMB Abdoulaye Sidy — 2025-2026. Version 2.4 (22 octobre 2026) — décisions de Sidy après M0 : 15 machines stables (dont 12 d'évaluation), aucune exclusion, validation = garde-fou.

## 1. De quoi parle ce projet

**Thème** : Détection d'anomalies en temps réel sur une infrastructure applicative sous dérive de concept : conception et évaluation d'une stratégie de réentraînement dans un pipeline MLOps.

**Question à laquelle le code doit répondre** : une stratégie de réentraînement **déclenchée par une dérive confirmée** (avec validation avant mise en service) maintient-elle la performance d'un détecteur d'anomalies mieux qu'un modèle figé, et à moindre coût qu'un réentraînement périodique ?

Le code produit **des résultats mesurés et reproductibles** qui alimentent les chapitres 4, 5 et 6 du mémoire, puis une **démonstration en direct** sur mon serveur pour la soutenance.

### Règle n° 1 : le pipeline est un moyen, pas le sujet
Prometheus, Grafana, MLflow, Docker, Telegram et GitLab CI servent l'expérience et la démonstration. Ne pas sur-construire. Chaque brique doit répondre à un besoin et se justifier en une phrase dans le mémoire.

### Deux sources de données, deux rôles
| Source | Rôle | Pourquoi |
|---|---|---|
| **SMD** (Server Machine Dataset, public) | Développement et **résultats principaux** (chapitre 6) | Métriques serveur réelles, multivariées, anomalies étiquetées ; reproductible ; comparable à la littérature |
| **Mon serveur personnel** (applications de test) | **Test final** sur infrastructure réelle + **démonstration** du pipeline MLOps (chapitre 5, soutenance) | Dérives réelles (déploiements) et anomalies injectées, avec horodatage exact |

### Livrables
**Socle (indispensable)**
1. Le mémoire + la présentation de soutenance (côté chat, pas dans ce dépôt).
2. Ce dépôt GitLab : README, tests, CI, une commande par expérience.
3. La stratégie proposée (P) et les stratégies de référence (B0, B1) implémentées.
4. Les résultats : `results/summary.csv`, tableaux, figures.

**Démonstration (planifiée)**
5. Pipeline en ligne sur mon serveur : Prometheus → service de détection → registre MLflow.
6. Tableau de bord Grafana : métriques, score d'anomalie, alertes, dérives, version du modèle en service.
7. Notifications Telegram : anomalie, dérive confirmée, réentraînement, promotion ou rejet.
8. Vidéo de démonstration (secours pour la soutenance).

**Bonus (seulement après le jalon M3)**
9. Assistant IA avec **validation humaine** : résumé de l'alerte par un LLM, proposition d'action, boutons Telegram « valider / refuser ». Il peut au plus approuver ou refuser la promotion d'un modèle. **Il n'agit jamais sur l'infrastructure.**

### Hors périmètre (ne pas coder)
- Agent autonome qui agit sur le serveur (redémarrage, mise à l'échelle) : le détecteur a des fausses alertes, l'agent agirait à tort.
- Application mobile (Telegram couvre le mobile) ; application web sur mesure (Grafana + interface MLflow suffisent).
- Détection de tous les types d'anomalies, prédiction de pannes, infrastructure DevOps complète (Kubernetes, multi-nœuds).
- Spark / Flink. Kafka seulement si un vrai besoin de découplage apparaît (voir §3).

---

## 2. Architecture

### Mode rejeu (évaluation, chapitre 6) : la source des chiffres du mémoire
```
 SMD (train / test / test_label)            Session enregistrée sur mon serveur (Parquet + labels)
            │                                                  │
            ▼                                                  ▼
  Injection de dérive (S1–S4, seed)  ─────►  Flux labellisé (anomalies + instant de dérive)
                                                     │
                                                     ▼
                         Harnais de rejeu prequentiel (même flux pour chaque stratégie)
                         features → modèle courant → score → alerte
                         surveillance de dérive → décision → réentraînement → validation
                                                     │
                                                     ▼
                         results/runs/<source>/<scenario>/<strategie>/<seed>/  → summary.csv
```

### Mode en ligne (démonstration, chapitre 5)
```
 [Applis de test] [node_exporter] [cAdvisor]
        │ métriques
        ▼
   [Prometheus] ──(API query_range, chaque minute)──► [Service de détection]
                                                        features → champion → score
                                                        surveillance de dérive → stratégie P
                                                        réentraînement → validation → promotion
        ┌──────────────────────────────┬────────────────┼───────────────────┐
        ▼                              ▼                ▼                   ▼
 [Exporter Prometheus :        [MLflow tracking    [Journal JSON      [Telegram :
  score, alertes, dérive,       + registry]         d'événements]      alertes et événements]
  version du modèle]
        │
        ▼
    [Grafana]
```
**Le même code** (features, modèles, détecteurs de dérive, stratégies) sert aux deux modes. Seule la source du flux change.

---

## 3. Stack

| Brique | Choix | Pourquoi (à reprendre dans le mémoire) |
|---|---|---|
| Langage | Python 3.11 | Écosystème ML |
| Modèle principal | Isolation Forest (scikit-learn) | Non supervisé, rapide à réentraîner : le coût du réentraînement est mesurable |
| Modèle secondaire (option) | Autoencodeur (PyTorch ou Keras) | Vérifier que la conclusion ne dépend pas du modèle |
| Dérive | `river` (ADWIN, PageHinkley, disponibles dans river 0.26) + `scipy` (test KS) + PSI codé à la main | Méthodes classiques de la littérature |
| Suivi MLOps | MLflow (tracking + model registry, alias `champion`) | Traçabilité des versions et des réentraînements |
| Collecte (serveur) | Prometheus + node_exporter + cAdvisor | Standard de fait pour les métriques d'infrastructure |
| Visualisation | Grafana | Tableaux de bord et alertes |
| Notifications | API Bot Telegram (requête HTTPS `sendMessage`) ; `python-telegram-bot` seulement pour les boutons du bonus | Alertes sur mobile sans application dédiée |
| Charge / pannes (serveur) | Locust + endpoints de panne dans une appli de test + `stress-ng` en conteneur limité | Anomalies et dérives contrôlées |
| Transport | Lecture directe de l'API Prometheus. **Kafka optionnel** | Un seul serveur, un seul consommateur : Kafka n'est pas nécessaire au départ |
| Conteneurs / CI | Docker Compose, GitLab CI | Reproductibilité |
| LLM (bonus) | API d'un LLM, appelée avec des métriques agrégées uniquement | Résumé d'alerte lisible |

Pas de One-Class SVM : coût d'entraînement élevé, sans apport pour la question posée (une phrase au chapitre 4).
Vérifier les API au moment de l'implémentation (river, MLflow) et **épingler les versions** dans `pyproject.toml`.

---

## 4. Données

### 4.1 SMD (source principale)
- Dépôt : `NetManAIOps/OmniAnomaly` sur GitHub, dossier `ServerMachineDataset`, licence MIT.
- 28 machines (3 groupes, `machine-<groupe>-<index>`), 38 métriques, environ 5 semaines. Chaque machine est découpée en deux moitiés : `train` (sans étiquettes, supposé majoritairement normal) et `test` (avec `test_label` point par point et `interpretation_label`). Intervalle d'échantillonnage : **1 minute** (non indiqué dans le README ; source : Garg et al., IEEE TNNLS 2022, + vérification empirique, décision D06).
- **Le SMD étiquette les anomalies, pas les dérives** : la dérive est injectée par nous (§5), à un instant connu.
- Utilisation : entraînement initial sur `train` ; flux évalué = `test` avec dérive injectée.
- **Dérive naturelle (OBS03)** : la moitié test diffère déjà de la moitié train sur 15 machines sur 28 (taux d'alertes de B0 ≥ 3× le taux calibré). Décision du 22/10/2026 :
  - Le critère porte sur les **fausses alertes** (alertes sur fenêtres étiquetées négatives), pas sur les alertes totales : une alerte sur une fenêtre positive est une détection réussie et ne témoigne d'aucune dérive. Compter toutes les alertes classait `machine-3-10` instable (3,95×) alors que ses fausses alertes ne font que 0,45 %.
  - **Mise au point** — `configs/datasets/smd_dev.yaml`, **16 machines** : les 13 instables, plus 3 stables tirées avec la graine 42 (machine-1-3, machine-2-7, machine-3-10). Réglage de k, K, δ, des bornes du garde-fou et des seuils de dérive.
  - **Évaluation S1–S4** — `configs/datasets/smd_eval.yaml`, **12 machines** : machine-1-2, 1-4, 1-7, 2-1, 2-2, 2-3, 2-4, 2-5, 3-2, 3-3, 3-4, 3-7. Les paramètres sont **figés avant** la première exécution sur ce périmètre.
  - **Étude de la dérive naturelle** (scénario `none`, sans injection, 4 stratégies) : les 13 machines instables, qui sont dans le périmètre de mise au point.
  - D13 et D14 ont été décidés avant ce partage, en partie sur machine-2-1, qui est une machine d'évaluation. Mesures refaites sur le seul périmètre de mise au point (`mdrift calibration` → `results/calibration_dev.csv`).
  - Ne jamais régler un paramètre en regardant les étiquettes d'une machine d'évaluation.
  - Criblage reproductible : `mdrift smd-screening` → `results/smd_screening.csv` (cité par le mémoire). Lecture des fichiers avec `pandas.read_csv` (np.loadtxt trop lent).
  - Les machines d'un même groupe sont corrélées (3-1 / 3-9 : 0,80) : moyenner par machine sur les graines avant les tests, rapporter aussi par groupe.
- Citer : Su et al. (2019), KDD (déjà dans la bibliographie du mémoire).

### 4.2 Mon serveur (test final et démonstration)
- **Applications de test uniquement.** Les applications en production ne servent que de bruit de fond réaliste, sans jamais être perturbées.
- Métriques : CPU, RAM, disque, réseau, requêtes/s, latence p50/p95, erreurs 5xx, connexions, métriques conteneurs.
- Anomalies injectées (horodatées dans `data/labels/<run_id>.csv` : `start_ts, end_ts, type, intensité`) :
  | Type | Comment |
  |---|---|
  | Pic CPU | `stress-ng --cpu` dans un conteneur limité, 1 à 5 min |
  | Fuite mémoire | `stress-ng --vm` montant progressivement, conteneur limité |
  | Latence | endpoint de l'appli de test : +X ms |
  | Rafale d'erreurs | endpoint de l'appli de test : X % de réponses 5xx |
  | Saturation | pic de trafic Locust bref et intense |
- Dérives réelles, horodatées dans `data/labels/<run_id>_drift.csv` (`ts, type, description`) : déploiement d'une nouvelle version d'une appli de test, changement des ressources d'un conteneur, montée durable du trafic.

### 4.3 Règles de sécurité sur mon serveur (non négociables)
- Injecter les pannes **uniquement** dans des conteneurs dédiés, avec limites Docker (`--cpus`, `--memory`).
- Ne jamais injecter sur un conteneur d'application en production ou de client.
- Prévoir un arrêt d'urgence : `make stop-chaos` qui tue tous les conteneurs d'injection.
- Surveiller la rétention Prometheus et l'espace disque.
- Aucun secret, IP, nom d'hôte, nom de client ou domaine dans le dépôt, les résultats ou les captures. Jeton Telegram et clés en variables d'environnement.

### 4.4 Features
- SMD : valeurs à l'instant *t* + statistiques glissantes sur *w* points (moyenne, écart-type, max) ; *w* dans `configs/features.yaml`.
- Serveur : fenêtres de 1 minute, mêmes agrégats (p95 pour la latence).
- La normalisation est **ajustée sur les données d'entraînement uniquement** et sauvegardée avec le modèle.
- Une fenêtre est **positive** si elle contient au moins un point étiqueté anormal.

---

## 5. Scénarios de dérive

Une dérive = un **changement durable** du comportement normal ; une anomalie = un événement **court**. Toute la difficulté est de ne pas confondre les deux. Les quatre scénarios couvrent les quatre types décrits au chapitre 3 (Gama et al., 2014).

| Code | Type | Sur le SMD (injection) | Sur mon serveur (réel) |
|---|---|---|---|
| S1 | Soudaine | À l'instant *t*<sub>d</sub>, décalage de niveau (+*k*·σ<sub>train</sub>) et changement d'échelle sur un sous-ensemble de métriques | Déploiement d'une nouvelle version (latence et RAM de base différentes) |
| S2 | Incrémentale | Même transformation, appliquée progressivement sur une durée *D* (rampe) | Montée durable du trafic sur plusieurs heures |
| S3 | Graduelle | Alternance : la proportion de points issus du régime transformé passe de 0 à 1 sur *D* | Déploiement progressif (canari) : part croissante du trafic vers la nouvelle version |
| S4 | Récurrente | Alternance périodique entre régime d'origine et régime transformé | Alternance de régimes de charge (jour / nuit simulés) |

- Les anomalies étiquetées d'origine sont conservées, **avant et après** la dérive.
- Règles d'injection : métriques de M tirées parmi celles de σ_train > 0 ; décalage **positif** ; valeurs **écrêtées à [0, 1]** (saturation physique ; une valeur > 1 rendrait la dérive triviale), part écrêtée enregistrée dans `summary.csv` ; *t*<sub>d</sub> tiré uniformément dans **[40 %, 60 %]** du flux test ; *D* ≤ **25 %** du flux (S2, S3). Calibrer k sur les machines de mise au point (dégradation de B0 en fonction de k).
- Tous les paramètres (*t*<sub>d</sub>, *k*, *D*, métriques touchées, période) sont dans `configs/scenarios/*.yaml` ; la seed fixe *t*<sub>d</sub> et le choix des métriques.
- Cible : **5 seeds par scénario et par machine** ; résultats en moyenne ± écart-type.
- Sur le serveur : enregistrer les sessions une fois (Parquet + labels), puis les **rejouer** pour chaque stratégie.

---

## 6. Les stratégies comparées

Toutes partagent : même modèle, mêmes features, même entraînement initial, même règle de seuil.

| Code | Stratégie | Rôle |
|---|---|---|
| B0 | **Modèle figé** : entraîné une fois, jamais réentraîné | Référence basse |
| B1 | **Réentraînement périodique** : toutes les *T* minutes, sur une fenêtre glissante récente | Référence classique |
| P | **Stratégie proposée** : réentraînement déclenché par une dérive confirmée + validation avant promotion | Contribution du mémoire |
| A1 | *(ablation, option)* P sans l'étape de validation | Montre ce que la validation apporte |

### Détail de la stratégie proposée (P)
1. **Surveillance** : à chaque fenêtre, comparer la distribution récente des **152 features** (pas des 38 métriques brutes) à celle de l'entraînement du modèle courant (PSI / KS par feature) et suivre le taux d'alertes (ADWIN). Règle d'agrégation : **part des features dont la statistique de KS (ou le PSI) dépasse un seuil**, comparée à une part déclenchante. Pas de correction de Bonferroni : sur `machine-1-2`, machine stable, 109 des 152 features rejettent au seuil de Bonferroni — les p-valeurs n'ont aucune spécificité à 20 000 points — et les 152 tests ne sont pas indépendants (rang effectif 120 à 131). La statistique, elle, discrimine : 0,0 % de features au-delà de KS 0,5 sur machine-1-2, 9,2 % sur machine-2-1, 26,3 % sur machine-3-1 (D18). Les features constantes sur la fenêtre d'entraînement sont exclues du calcul. Le taux d'alertes seul est un signal secondaire : avec K = 30, il « confirme » une dérive sur 26 des 28 machines, dont 13 des 15 machines stables — K = 30 fenêtres ne valant que 30 minutes, la persistance exigée est trop faible (D19).
2. **Confirmation** : la dérive n'est retenue que si le signal persiste **K fenêtres consécutives** (dérive durable ≠ anomalie passagère).
3. **Réentraînement** : nouveau modèle (challenger) sur les *W* dernières minutes, hors fenêtre réservée *V*, **sans exclusion par défaut** (`exclusion.method: none`). Exclure les alertes du champion fait s'effondrer le challenger sous saturation (mesuré à M0) ; les variantes `top_percentile` et `champion_alerts` restent implémentées pour l'ablation.
4. **Validation = garde-fou, pas concours** : sans étiquettes, aucun critère ne classe fiablement deux détecteurs (mesuré ; Ma et al., 2023). Rejet si historique < 300 fenêtres, taux d'alertes sur *V* hors [0,1 % ; 30 %], ou seuil dégénéré. Bornes à calibrer sur les machines de mise au point. **Aucune étiquette** dans une décision de la stratégie.
5. **Promotion** : si le garde-fou est passé, le challenger devient `champion` dans le registre MLflow ; sinon rejet journalisé. En mode en ligne, option `human_approval: true` pour le bonus Telegram.
6. **Refroidissement** : aucun réentraînement pendant *C* minutes après une promotion.

Tous les paramètres (*T*, *K*, *W*, *C*, seuils) sont dans des YAML. **Aucune valeur en dur.** Les valeurs retenues finissent dans un tableau du chapitre 4.

**Seuil d'anomalie** : percentile élevé (ex. 99e) des scores sur les données d'entraînement, recalculé à chaque réentraînement. Même règle pour toutes les stratégies.

---

## 7. Métriques d'évaluation

| Famille | Métrique | Définition |
|---|---|---|
| Qualité de détection | Précision, rappel, F1 par fenêtre | Fenêtre positive = contient au moins un point anormal |
| | Rappel par événement | Part des anomalies avec au moins une alerte pendant leur durée (+ tolérance) |
| | Fausses alertes par heure | Alertes hors de toute anomalie étiquetée |
| | Délai de détection | Début de l'anomalie → première alerte |
| Adaptation | F1 avant / après dérive | Calculés séparément sur chaque segment |
| | Temps de récupération | Début de la dérive → retour du F1 glissant à ≥ 90 % de son niveau pré-dérive (seuil figé dans la config) |
| | Délai de détection de la dérive | Début réel de la dérive → confirmation (stratégie P) |
| Coût | Nombre de réentraînements | Total, promus, rejetés |
| | Temps de calcul cumulé | Secondes CPU passées à réentraîner |
| Fiabilité des mises à jour | Promotions dégradantes | Promotions après lesquelles le modèle promu obtient, sur les N = 500 fenêtres suivantes, un F1 inférieur de plus de δ à celui du modèle remplacé (δ calibré sur l'écart-type inter-graines à M2 ; rapporter aussi la distribution des écarts). Ancien modèle scoré **en parallèle (shadow)** en un bloc par promotion ; calcul **hors ligne** à partir des traces ; coût **exclu** du temps de calcul de la stratégie. Sert à tester H3 (P contre A1) |
| Spécificité de la surveillance | Fausses confirmations de dérive | Confirmations de dérive sur les machines stables sans injection (scénario `none`), en valeur absolue et par machine. Sans elle, l'étape de confirmation n'est pas évaluable : un moniteur qui confirme tout le temps obtient un excellent délai de détection. Référence à battre : 13 fausses confirmations sur 15 machines stables (D19) |

Évaluation **prequentielle** : chaque fenêtre est d'abord scorée par le modèle courant, puis utilisée (ou non) pour la suite. Aucune fuite d'information du futur vers le passé.
Test statistique entre P et B1 sur les seeds (ex. Wilcoxon apparié) si les écarts sont faibles.

---

## 8. Structure du dépôt

```
memoire-drift/
├── CLAUDE.md · README.md · pyproject.toml · Makefile · docker-compose.yml · .gitlab-ci.yml · .env.example
├── configs/
│   ├── datasets/{smd,serveur}.yaml
│   ├── model.yaml · features.yaml · drift.yaml
│   ├── strategies/{static,periodic,proposed,ablation}.yaml
│   ├── scenarios/{s1_soudaine,s2_incrementale,s3_graduelle,s4_recurrente}.yaml
│   └── notify.yaml
├── datasets/smd/            # téléchargement + chargement du SMD
├── scenarios/               # injection de dérive (SMD) + composition des sessions serveur
├── src/
│   ├── features/            # fenêtrage, agrégats, normalisation
│   ├── models/              # isolation_forest.py, autoencoder.py
│   ├── drift/               # psi.py, ks.py, adwin.py, confirmation (K fenêtres)
│   ├── strategies/          # static.py, periodic.py, proposed.py
│   ├── evaluation/          # replay.py, metrics.py
│   ├── serving/             # boucle en ligne, lecture Prometheus, exporter
│   ├── tracking/            # wrapper MLflow
│   └── notify/              # telegram.py
├── infra/                   # appli de test (FastAPI), Locust, injecteur, Prometheus, Grafana (dashboards en JSON)
├── assistant/               # bonus : résumé LLM + validation humaine
├── notebooks/               # analyse et figures (lecture seule des résultats)
├── tests/
├── data/                    # .gitignore : raw/, labels/, sessions/
├── results/                 # runs/<source>/<scenario>/<strategie>/<seed>/ + summary.csv
└── docs/
    ├── decisions.md         # journal des choix techniques
    ├── journal.md           # journal daté de l'avancement
    └── figures/             # graphiques et captures pour le mémoire
```

---

## 9. Conventions de code

- Type hints, `ruff`, `pytest`.
- Chaque exécution : un `run_id`, une seed fixée, la config complète sauvegardée à côté des résultats.
- Interfaces simples : `Detector.fit / score`, `DriftMonitor.update -> bool`, `Strategy.on_window(window) -> Decision`, `Notifier.send(event)`.
- Journal d'événements en JSON lignes : `{ts, run_id, strategy, event, details}` avec `event ∈ {alert, drift_signal, drift_confirmed, retrain_start, retrain_end, promoted, rejected}`. Telegram et Grafana consomment ces mêmes événements.
- Mesurer le **temps de traitement d'une fenêtre** (lecture, features, score, surveillance de dérive ; hors réentraînement). En rejeu, le scoring par blocs (décision D10) ne donne qu'un temps amorti : ajouter `window_ms_amortized = 1000 × wall_seconds / n_windows` dans `summary.csv`. La vraie mesure par fenêtre (`window_ms_mean`, `window_ms_p95`) se fait dans la boucle en ligne (M5), exposée en jauge à Prometheus. Sert aux indicateurs I7 / T1 du chapitre 2 (temps réel : traitement < intervalle de collecte).
- Notifications Telegram : regroupement (pas plus d'un message par minute et par type) pour éviter le spam.
- Tests minimum : fenêtrage, métriques (cas calculés à la main), injection de dérive (instant et amplitude corrects), stratégie P (déclenche après K fenêtres, respecte le refroidissement), rejeu (même seed = même résultat).

---

## 10. Jalons

Chaque jalon produit quelque chose d'utilisable dans le mémoire. **Les résultats (M2, M3) passent avant la démonstration.**

| Jalon | Contenu | Livrable pour le mémoire |
|---|---|---|
| M0 | Dépôt, chargement du SMD, features, Isolation Forest, évaluation sans dérive injectée | Performance de référence sans dérive injectée (6.2.1) — **terminé le 22/10/2026** |
| M1 | Injection de dérive S1–S4 avec seeds | Figures de chaque scénario |
| M2 | Détecteurs de dérive + stratégies B0, B1, P + harnais de rejeu + métriques | **Résultats principaux** (6.2.2 à 6.2.4) |
| M3 | Campagne complète : machines × scénarios × stratégies × seeds (+ ablation, + autoencodeur en option) | Tableaux finaux et graphiques |
| M4 | Infrastructure sur mon serveur : Docker Compose (appli de test, Prometheus, node_exporter, cAdvisor, Grafana), injecteur, enregistrement des sessions | Description des données réelles |
| M5 | Boucle en ligne : service de détection, registre MLflow, exporter, tableau de bord Grafana, notifications Telegram | Captures du pipeline MLOps (chapitre 5) |
| M6 | Test final : dérives réelles + anomalies injectées sur le serveur, rejeu des trois stratégies sur les sessions enregistrées | Validation sur infrastructure réelle (chapitre 6) |
| M7 | GitLab CI (lint, tests, build) + vidéo de démonstration | Capture CI, support de soutenance |
| M8 | *(bonus)* Assistant IA avec validation humaine sur Telegram | Section complémentaire ou perspectives (chapitre 7) |

---

## 11. Ce que Sidy renvoie à Claude (chat) pour le mémoire

À chaque jalon, envoyer :
1. **`docs/decisions.md`** : chaque choix technique en 3 lignes (choix, alternatives écartées, raison) → justifications du chapitre 4.
2. **`docs/journal.md`** : dates de début et de fin de chaque jalon → planification et Gantt (2.6).
3. **Résultats** : `results/summary.csv` (une ligne par source × scénario × stratégie × seed, toutes les métriques du §7) + les configs utilisées.
4. **Figures** : score d'anomalie dans le temps avec dérive et alertes marquées, F1 glissant par stratégie, nombre de réentraînements (PNG 300 dpi) ; captures Grafana, MLflow et Telegram recadrées, sans IP ni nom d'hôte.
5. **Description des données** : machines SMD retenues, durée des sessions serveur, nombre de fenêtres, nombre d'anomalies par type, dérives réelles avec leur date.
6. **Problèmes rencontrés** et contournements → limites (6.3.3 et 7.3).

Rien n'est écrit dans le mémoire sans chiffre ou fichier source correspondant.

---

## 12. Repères d'outillage pour Claude Code

Section technique (commandes, architecture du code), distincte du cadrage de recherche ci-dessus.
Le détail de chaque commande du CLI est dans `README.md` ; ce qui suit est ce qui ne se découvre
pas en lisant un seul fichier.

### Commandes courantes

```bash
uv sync                 # installe les dépendances (Python 3.11 épinglé par .python-version)
uv run mdrift --help    # CLI, point d'entrée unique de toutes les expériences

make lint                # ruff check . && ruff format --check .
make test                # pytest (tests/, voir pyproject.toml : pythonpath = ["."])
make build                # wheel installé dans un venv neuf hors dépôt, mdrift --help + import src/infra vérifiés

uv run pytest tests/test_drift.py                       # un seul fichier
uv run pytest tests/test_drift.py::test_nom_du_cas -v    # un seul test
```

Les trois cibles `lint` / `test` / `build` du Makefile sont la **seule** définition de ce qui est
vérifié : `.gitlab-ci.yml` (le livrable du jalon M7) et `.github/workflows/ci.yml` (ce qui tourne
réellement, le dépôt étant hébergé sur GitHub) appellent toutes les deux exactement ces cibles, et
`tests/test_ci.py` vérifie qu'elles ne divergent pas. Le seul test qui exige le SMD téléchargé se
déclare `skipped` en CI ; la campagne complète (`mdrift campaign`) n'y tourne jamais (plusieurs
heures).

### Convention de configuration

Aucune valeur en dur dans le code : tout paramètre vient d'un YAML de `configs/` et est recopié à
côté des résultats de chaque exécution. `src/config.py::load_config` résout un chemin relatif
d'abord depuis `configs/`, puis depuis la racine du dépôt — donc `load_config("drift.yaml")` et
`load_config("datasets/smd_dev.yaml")` fonctionnent tous les deux sans que l'appelant connaisse le
sous-dossier.

### Le même code sert au rejeu (offline) et à la boucle en ligne (démonstration)

`src/evaluation/replay.py` (harnais prequentiel, chapitre 6) et `src/serving/loop.py` (boucle en
ligne branchée sur Prometheus, chapitre 5) exécutent le **même** cycle de réentraînement :

```
decision = strategy.on_window(ctx)        # faut-il réentraîner ?
plan     = strategy.propose(history, i)    # quelles données, quelle fenêtre V réservée
challenger.fit(plan.train_X)               # le HARNAIS entraîne, jamais la stratégie
stats    = mesures sans étiquette sur V    # le HARNAIS mesure
verdict  = strategy.validate(stats)        # mise en service ou rejet
si promu : champion = challenger, puis strategy.on_fit(plan.train_X, i)
```

Ces quatre lignes centrales sont factorisées dans `src/strategies/retraining.py` précisément pour
garantir que la démonstration (chapitre 5) montre la même logique que celle évaluée (chapitre 6).
Une stratégie (`src/strategies/base.py::Strategy`) ne voit **jamais** d'étiquette : `WindowContext`
et `ValidationStats` n'ont aucun champ par lequel une étiquette pourrait transiter — c'est un
invariant structurel, pas une discipline de code à vérifier à la main. `src/strategies/factory.py`
construit une stratégie depuis son YAML et convertit les durées en minutes des configs en nombre de
fenêtres (`minutes_to_windows`), seul endroit où cette conversion existe.

### Surveillance de dérive

`src/drift/monitor.py` combine un signal principal (part des features dont la statistique KS/PSI
dépasse un seuil — `src/drift/aggregate.py`) et un signal secondaire (ADWIN sur le taux d'alertes —
`src/drift/adwin.py`), puis exige une persistance de K fenêtres consécutives
(`src/drift/confirmation.py`) avant de confirmer une dérive. Le mode de combinaison
(`features` / `features_and_alerts` / `features_or_alerts`) est un paramètre de
`configs/drift.yaml`, pas un choix figé dans le code.

### Deux périmètres SMD disjoints

`datasets/smd/loader.py` charge les machines décrites par un YAML de `configs/datasets/`.
`smd_dev.yaml` (16 machines) sert à régler tous les paramètres ; `smd_eval.yaml` (12 machines) ne
doit jamais être regardé — ni ses étiquettes ni ses résultats — avant que les paramètres soient
figés. Toute commande de calibrage (`smd-screening`, `calibration`, `calibration-k`,
`calibrate-drift`) écrit dans son propre fichier, jamais dans `results/summary.csv`, pour qu'une
exécution de mise au point ne puisse pas contaminer la synthèse citée au mémoire.

### Démonstration en ligne (serveur, jalons M4–M5)

`infra/app/` est l'application de test FastAPI ; `infra/injector/` injecte les pannes dans des
conteneurs dédiés, marqués à la fois par un label Docker (`memoire.injection=1`) et un préfixe de
nom (`memoire-chaos-`) — les deux marques sont redondantes par construction, pour que
`make stop-chaos-force` (l'arrêt d'urgence du §4.3, qui ne passe pas par Python) ne puisse jamais
cibler un conteneur de production même tapé dans l'urgence. Les tableaux de bord Grafana
(`infra/grafana/build_dashboard.py`) ne se modifient jamais à la main : un test vérifie que le JSON
commité est bien la sortie du générateur. `src/tracking/registry.py` encapsule MLflow (alias
`champion` dans le registry) ; `src/serving/loop.py` est la boucle qui lit Prometheus, score,
surveille la dérive, réentraîne, valide, promeut et notifie Telegram (`src/notify/telegram.py`,
regroupé pour éviter le spam).

### Résultats

Chaque exécution écrit dans `results/runs/<source>/<machine>/<scenario>/<stratégie>/<seed>/`
(config complète, métriques, journal d'événements JSON lignes, traces) et met à jour sa ligne dans
`results/summary.csv`. Le `run_id` est stable : rejouer la même expérience met à jour sa ligne au
lieu de la dupliquer — utile pour reprendre une `campaign` interrompue sans relire tout ce qui a
déjà tourné.
