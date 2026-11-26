# Journal d'avancement

> Dates de debut et de fin de chaque jalon -> planification et diagramme de Gantt (memoire, 2.6).

| Jalon | Debut | Fin | Etat |
|---|---|---|---|
| M0 — Depot, SMD, features, Isolation Forest, evaluation sans derive | 2026-10-22 | 2026-10-22 | termine |
| M0+ — Mesures preparatoires : derive naturelle a 28 machines, regles de P | 2026-10-22 | 2026-10-22 | termine |
| M0++ — Calibrage des etapes 3 et 4 sur le perimetre de mise au point | 2026-10-26 | 2026-10-26 | termine |
| M1 — Injection de derive S1 a S4, calibrage de l'amplitude, figures | 2026-10-26 | 2026-10-26 | termine |
| M2 — Surveillance de derive, strategies B0 / B1 / P / A1, harnais de rejeu | 2026-10-26 | 2026-10-27 | termine |
| M3 — Campagne 12 machines x S1-S4 x 5 graines, tests H1-H3, figures et tableaux | 2026-10-28 | 2026-10-28 | termine |

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

## 2026-10-22 (suite) — mesures preparatoires a M1

Mesures demandees par la relecture du jalon M0, destinees a fixer les parametres de la strategie P
et le protocole d'injection avant de coder M1. Reponses detaillees dans `docs/questions-memoire.md`.

- SMD complet telecharge : 28 machines, 486,0 Mo annonces, 464 MiB sur disque, 151 s.
  Config `configs/datasets/smd_all.yaml` ajoutee (D12).
- **Derive naturelle quantifiee sur les 28 machines** (OBS04) : rapport median 3,84x le taux calibre,
  de 0,59x a 63,81x ; **15 machines sur 28 sont stables** (taux de fausses alertes < 3x le taux
  calibre ; le critere a ete corrige, voir plus bas). Le rapport d'alertes et le KS
  par metrique brute ne concordent pas (Spearman +0,25) : a M2, PSI et KS porteront sur les 152 features.
- **Le taux d'alertes seul ne suffit pas a confirmer une derive** (OBS05) : un moniteur naif declenche
  sur 28 machines sur 28, y compris les plus stables. K = 30 fenetres est trop permissif.
- **Les machines d'un meme groupe ne sont pas independantes** (OBS06) : correlation moyenne +0,801 entre
  machine-3-1 et machine-3-9, contre -0,008 entre groupes. Consequence sur l'agregation de M3.
- **Regle d'exclusion au reentrainement tranchee par la mesure** (D13) : exclure toutes les alertes du
  champion fait s'effondrer le challenger sous saturation (F1 0,0731 contre 0,5172 a 400 h sur machine-3-1),
  et ne rien exclure egale la meilleure regle aux cinq instants testes. Defaut retenu : aucune exclusion.
  Chiffre marquant : a 400 h, reentrainer donne F1 0,5172 contre 0,0695 pour le champion fige (x7,4).
- **Critere de validation sans etiquettes refute par la mesure** (D14) : le critere envisage correle dans
  le mauvais sens avec le F1 aval (Spearman jusqu'a +0,90). L'etape 4 de P devient un garde-fou bilateral,
  et ce qu'elle apporte sera mesure par l'ablation A1.
- **Protocole d'injection fixe** (D15) : ecretage a [0, 1], t_d tire dans [40 %, 60 %] du flux,
  rampe D bornee a 25 %, metriques injectables restreintes aux non constantes (30 a 33 sur 38).
- **Evaluation fantome de l'ancien champion specifiee** (D16), compatible avec le scoring par blocs.
- Corrections documentaires : citation de D06 remplacee par Garg et al. (IEEE TNNLS 33(6), 2022), qui
  tranche aussi OBS02 en faveur de 38 metriques.
- Reserve technique relevee (OBS07) : le loader devra lire avec pandas avant la campagne M3.

## 2026-10-22 (suite 2) — corrections apres relecture du memoire

La relecture du jalon M0 a renvoye six remarques. Les six sont traitees ; l'une invalidait une de
mes corrections de la veille.

- **OBS03 remis d'aplomb (erreur de mesure de mon cote)**. La "correction" que j'avais apportee la
  veille, qui datait la degradation de machine-3-1 a 16 h, etait un artefact : moyenne glissante
  **centree** (anticipation de 4,2 h, mesuree) et comptage de **toutes** les alertes, le premier
  depassement tombant en fait sur l'anomalie etiquetee 20,1 h - 20,6 h. En moyenne causale et sur les
  **fausses alertes seules**, la derive durable demarre a **93,5 h** : la description initiale, vers
  90 h, etait donc la bonne. Deux garde-fous ajoutes : la config refuse toute autre methode de mesure,
  et un test de non-regression reproduit le piege.
- **Critere de stabilite corrige dans la foulee** : il porte desormais sur les fausses alertes et non
  sur le taux d'alertes total, ce qui fait passer le nombre de machines stables de 13 a **15**
  (machine-2-2 et machine-3-10 rejoignent le groupe ; cette derniere avait 3,95x d'alertes mais
  seulement 0,45 % de fausses alertes).
- **Partage mise au point / evaluation declare** (D17) : 16 machines pour le reglage des parametres,
  12 machines stables reservees aux resultats S1-S4. Plusieurs choix de la veille avaient ete faits en
  regardant les etiquettes de test, sans que ce partage existe.
- **Regle d'agregation des tests de derive tranchee par la mesure** (D18) : la part de features dont la
  statistique de KS depasse un seuil, et non une correction de Bonferroni sur les p-valeurs. Sur
  machine-1-2, machine stable, 109 features sur 152 rejettent au seuil de Bonferroni : les p-valeurs
  n'ont aucune specificite a 20 000 points. Les 152 tests ne sont de toute facon pas independants
  (rang effectif 120 a 131).
- **Nouvelle metrique de specificite** (D19) : les fausses confirmations de derive, comptees sur les
  machines stables sans injection. Point de reference a battre : le moniteur naif en declenche sur
  13 des 15 machines stables.
- **Criblage transforme en commande** (D20) : `mdrift smd-screening` ecrit `results/smd_screening.csv`,
  28 lignes, environ 35 s. Le memoire citera ce fichier et non une note de travail.
- **Reserve OBS07 levee** : le loader lit avec pandas, controles et tests inchanges.
- Appui bibliographique ajoute au garde-fou de validation : Ma, Zhao, Zhang et Akoglu (2023),
  ACM SIGKDD Explorations 25(1), doi 10.1145/3606274.3606277.

Etat du code : 53 tests passent, `ruff` sans remarque.

## 2026-10-26
Journee consacree a une seule chose : refaire proprement les deux mesures qui fondaient D13 et D14.
Elles avaient ete faites sur machine-1-1, machine-2-1 et machine-3-1, or machine-2-1 appartient au
perimetre d'evaluation declare la veille (D17). Plutot que de declarer la limite au 6.3.3, les
mesures ont ete refaites sur le seul perimetre de mise au point.

- **Nouvelle commande `mdrift calibration`** (`src/evaluation/calibration.py`,
  `configs/calibration.yaml`) : 16 machines, 78 instants de reentrainement, 8 regles d'exclusion,
  3 graines, soit **473 challengers** entraines et evalues en aval. Resultat dans
  `results/calibration_dev.csv`. Environ une heure de calcul.
- **Deux defauts de protocole corriges avant d'en tirer la moindre conclusion.** (1) Ne visiter qu'un
  instant de reentrainement par machine rend les regles d'exclusion indistinguables : a la premiere
  confirmation, le champion n'a presque pas encore alerte, les regles gardent les memes fenetres et
  produisent le meme modele plusieurs fois, ce qui gonfle les correlations par des ex aequo. Le
  module visite donc cinq instants et **fusionne les regles qui gardent exactement les memes
  fenetres**. (2) La datation de la confirmation utilise une moyenne glissante **causale**, et compte
  **toutes** les alertes, a l'inverse de la mesure descriptive d'OBS03 qui ne compte que les fausses :
  la strategie n'a pas les etiquettes. Les deux choix sont opposes a dessein et documentes comme tels.
- **D13 confirmee, justification refaite.** Ne rien exclure est optimale 43 fois sur 78, ecart median
  au meilleur 0,0000, et bat sa seule concurrente serieuse (`top_percentile` a 1 %) 52 fois contre 24,
  Wilcoxon apparie p = 0,0040. Exclure toutes les alertes du champion s'effondre sous la moitie du F1
  sur 11 decisions. **Une affirmation du 22/10 etait fausse** : l'equivalence des regles (ecart
  maximal 0,011) etait un artefact de la mesure sur une seule machine ; sur 16 machines l'ecart au
  meilleur atteint 0,7324. Aucune regle ne domine partout, `none` est le meilleur defaut et non un
  choix gratuit.
- **D14 confirmee, justification inversee.** Le critere sans etiquettes ne correle pas dans le mauvais
  sens, comme ecrit le 22/10 : Spearman median **-0,66**, donc le bon sens. Mais le signe s'inverse
  sur 18 decisions sur 71, choisir par ce critere coute jusqu'a 100 % du F1 atteignable et selectionne
  le pire candidat une fois sur six. La conclusion pratique ne bouge pas, sa formulation si : le
  critere n'est pas inverse, il est trop instable pour classer.
- **Bornes du garde-fou calibrees, et elles contredisent le cadrage.** `alert_rate_min` est
  **desactivee** : 98 challengers sur 473 n'alertent sur aucune fenetre reservee et 63 sont sains,
  avec un meilleur F1 aval que la moyenne. `alert_rate_max` passe de 30 % a **0,90** : a 30 % elle
  rejetterait 37 challengers sains sur 285 pour 22 degrades seulement. `min_train_windows: 256`,
  deduite de `max_samples` du modele. Consequence assumee : le garde-fou ne rejette que 5 candidats
  sur 473. En rejeu, A1 devrait donc egaler P ; c'est une prediction testable, a ecrire comme telle.
- **OBS08, resultat inattendu et utile** : reentrainer a un instant arbitraire ne bat le champion fige
  que 43 fois sur 78, rapport de F1 median x1,04. A l'instant que donnerait le moniteur naif, 3 fois
  sur 14 seulement, rapport median x0,78 : **reentrainer trop tot nuit**. Le x7,4 mesure le 22/10 sur
  machine-3-1 est un cas favorable et ne doit pas etre presente comme le gain du reentrainement.
  C'est l'argument empirique le plus direct en faveur de l'etape de confirmation.
- **`CLAUDE.md` aligne sur les decisions** (version 2.4) : il portait encore 13 machines stables et
  proposait machine-2-1 pour la mise au point. La ou le cadrage et `docs/decisions.md` divergent,
  ce sont D17 a D20 qui font foi.
- **Tests** : `tests/test_calibration.py`, 17 cas calcules a la main, dont deux encodent les pieges
  ci-dessus (datation causale, fusion des regles equivalentes).

Etat du code : **70 tests passent**, `ruff` sans remarque.

## 2026-10-26 (suite) — M1 : injection de derive S1 a S4

Jalon M1 realise dans la journee, apres M0++. Objectif : produire les quatre scenarios de derive du
paragraphe 5 du cadrage, calibrer leur amplitude et montrer ce qu'ils font aux donnees.

- **Une transformation, quatre profils.** `configs/scenarios/_transformation.yaml` porte l'amplitude
  (decalage k, changement d'echelle, ecretage, plage de tirage de t_d), partagee par les quatre
  scenarios ; `s1_soudaine`, `s2_incrementale`, `s3_graduelle` et `s4_recurrente` ne contiennent que
  leur **profil temporel** a(t) et designent ce fichier. C'est la condition pour qu'un ecart de
  performance entre deux scenarios soit attribuable au type de derive et a rien d'autre.
- **Injection sur les 38 metriques brutes, avant fenetrage** (`scenarios/injection.py`), jamais sur
  les 152 features. La moitie `train` n'est jamais transformee : mu_train et sigma_train restent le
  comportement normal appris par le modele. Les etiquettes d'anomalie ne sont pas touchees.
- **Defaut de conception trouve en lisant la premiere sortie** (D21) : avec la graine seule, toutes
  les machines derivaient a la **meme position relative** du flux (0,418 a la graine 42). Les cinq
  graines du cadrage n'exploraient donc que cinq positions au total. La graine inclut desormais la
  machine, via `zlib.crc32` ; 47 positions distinctes sur 192 flux, de 0,4015 a 0,5942. L'appariement
  entre strategies est preserve et deux tests l'encodent.
- **Amplitude calibree** (D22) : `mdrift calibration-k`, 16 machines de mise au point, grille 0,5 a 8,
  trois graines, 288 rejeux en 132 s -> `results/calibration_k_dev.csv`. **k = 2** retenu : 13 machines
  sur 16 perturbees d'un facteur 3 au moins, 5 saturees, une seule au-dela de la cible d'ecretage.
  k = 3 ne gagne qu'une machine perturbee pour deux saturees de plus. Parametre **fige avant** toute
  execution sur les 12 machines d'evaluation.
- **OBS09, resultat qui a oblige a changer de critere** : sous derive injectee, le modele fige ne perd
  pas le rappel, il le **gagne** (0,93 a k = 2) et perd la precision (0,17). La degradation est un
  afflux de fausses alertes. Le rapport F1 apres / F1 avant, prevu comme critere de calibrage, est
  inexploitable ici : F1 avant nul sur 6 machines sur 16, et un rapport de 63,8 sur machine-2-7 dont
  le F1 avant vaut 0,0070. Critere remplace par le taux d'alertes, qui est aussi celui d'OBS03.
- **Audit des 192 flux** (`mdrift inject` -> `results/injections.csv`, 16 machines x 4 scenarios x 3
  graines) : part ecretee mediane 6,1 %, maximum 25,98 %, aucun depassement du garde-fou de 35 %.
- **Deux figures** (`mdrift scenario-figures`, PNG 300 dpi) : `docs/figures/profils_derive.png`, les
  quatre profils a(t) seuls, pour le chapitre 5 ; `docs/figures/injection_scenarios.png`, la serie
  d'origine et la serie derivee superposees sur une metrique reelle de machine-1-1, les quatre
  scenarios cote a cote. La metrique affichee est choisie sur l'ecart **obtenu apres ecretage** et non
  sur sigma_train, pour ne pas montrer une metrique plafonnee.
- **Tests** : `tests/test_injection.py`, **26 cas** (instant et amplitude corrects, train intacte,
  etiquettes intactes, profils a(t), ecretage et garde-fou, reproductibilite, decorrelation des
  machines).

- **Trou documentaire comble** (D23) : le choix du SMD lui-meme n'etait justifie nulle part. Six
  criteres tires de la question du memoire, et les alternatives ecartees avec leur motif : SWaT
  (capteurs d'un procede physique, acces sur demande nominative), SMAP et MSL (multivaries en
  apparence seulement, un seul canal continu par entite), Yahoo S5 (univarie, anomalies en partie
  synthetiques), traces de mon seul serveur (anomalies choisies par moi, une seule entite, non
  reproductible par le jury). Trois marqueurs [A VERIFIER] laisses sur les citations de seconde main.

Etat du code : **96 tests passent**, `ruff check` et `ruff format --check` sans remarque sur 40 fichiers.

## 2026-10-26 / 2026-10-27 — M2 : surveillance de derive, trois strategies, harnais de rejeu
Le jalon le plus lourd du depot : c'est lui qui produit les resultats principaux du chapitre 6. Deux
journees, et deux surprises qui ont chacune change une decision — un seuil de derive qui ne marchait
pas du tout comme D18 le laissait croire, et une persistance K dont l'effet est l'inverse de
l'intuition.

- **Module de derive** (`src/drift/`, 9 fichiers, 1 064 lignes) : statistiques par feature (`ks.py`,
  `psi.py`, sur une discretisation commune de la reference), reference d'entrainement
  (`reference.py`), modele nul empirique (`null_model.py`), agregation en part de features en derive
  (`aggregate.py`), persistance sur K fenetres (`confirmation.py`), signal secondaire ADWIN sur le
  taux d'alertes (`adwin.py`) et le moniteur qui combine tout (`monitor.py`). La surveillance porte
  sur les **152 features** et non sur les 38 metriques brutes, comme le demande le §6.
- **Trois strategies et une ablation** (`src/strategies/`, 747 lignes) : B0 figee, B1 periodique, P
  proposee, A1 = P sans validation. A1 n'a **aucun code propre** : c'est la meme classe avec
  `validation_enabled = False`, sans quoi l'ecart P - A1 melangerait l'effet de la validation a celui
  d'une seconde implementation.
- **Contrat champion / challenger** (D26) : `on_window` -> `propose` -> le harnais entraine et score
  -> `validate` -> `on_fit`. La strategie decide et choisit ses donnees, le harnais entraine, score,
  mesure et journalise. L'interdiction du §6 (« aucune etiquette dans une decision de la strategie »)
  devient verifiable par la signature des types et non par relecture ; un test l'affirme directement.
- **Le seuil de derive absolu de D18 ne marchait pas** (D25), et c'etait le defaut le plus grave des
  deux journees. D18 avait mesure un contraste reel — 0,0 % de features au-dela de KS 0,5 sur une
  machine stable contre 26,3 % sur une machine derivante — mais **entre les deux moities entieres**
  d'une machine. Le moniteur en service compare tout autre chose : la reference d'entrainement contre
  une fenetre recente de 500 fenetres, soit 8,3 h fortement autocorrelees, qui ne couvrent jamais
  l'etendue d'un entrainement de 20 jours. Mesure sur le perimetre de mise au point, scenario `none`,
  seuil absolu de 0,5 : part mediane de features « en derive » de 0,1500 a 0,4914 sur les machines
  **stables** et de 0,1667 a 0,4621 sur les **instables**. Les deux populations se recouvrent
  entierement — aucune part declenchante ne pouvait les separer. Remplace par un **modele nul
  empirique** : le seuil de chaque feature est un quantile eleve de la statistique atteinte en faisant
  glisser un bloc de meme taille sur son propre entrainement. Contraste retabli, facteur median par
  machine de **6,4**. Sur machine-1-3, les seuils vont de 0,0045 a 0,9452 (mediane 0,6829) sur 464
  blocs : un scalaire ne pouvait pas convenir.
- **Calibrage par balayage, pas par rejeu** (D28) : un point de grille ne change ni les features, ni
  la reference, ni la distribution nulle, seulement deux comparaisons. Le flux est donc parcouru une
  fois par trace et des centaines de points sont evalues sur la matrice conservee en memoire.
  Equivalence verifiee et non supposee : les fonctions vectorisees sont testees contre la classe
  `Confirmation` reellement utilisee en rejeu sur **150 cas aleatoires**, avec egalite exacte de la
  fenetre de confirmation. Cout : **536 points de grille sur 320 traces** en trois balayages, la ou
  un balayage par rejeu aurait demande plusieurs jours.
- **Parametres de la surveillance figes** (D29, `results/calibration_m2_dev.csv`, 42 880 lignes) :
  R = 500, quantile 0,999, marge 1,0, part declenchante 0,20, **K = 960** (16 h). C'est le point le
  plus sensible parmi les 398 (sur 536) qui ne produisent **aucun** faux signal sur les machines
  stables : **51 detections sur 64 injections (79,7 %)**, delai median de confirmation 2 431 fenetres.
  Par scenario : S1 14/16, S3 15/16, S2 13/16, **S4 9/16** — la derive recurrente est le cas faible,
  et c'est coherent, un regime qui alterne plus vite que K ne presente jamais K fenetres consecutives
  de signal.
- **Le resultat contre-intuitif du jalon** : le taux de detection **monte** avec K jusqu'a 960 avant
  de descendre (36/64 a K = 240, 51/64 a 960, 45/64 a 2 400). L'explication est structurelle : la
  reference est figee tant qu'aucun reentrainement n'a eu lieu et la confirmation est latchee, donc
  il y a au plus **une** confirmation par flux. Un K trop court confirme avant *t_d*, sur une
  fluctuation, et **verrouille** la detection de la vraie derive — 28 des 64 injections sont perdues
  ainsi a K = 240, contre 4 a K = 960. Le cas est visible a l'etat pur sur machine-1-6 : confirmation
  a la fenetre 5 483 alors que *t_d* vaut 9 905, sur les quatre scenarios, et les quatre injections
  manquees. C'est une figure du 6.2.3, pas une note de reglage.
- **Plancher structurel d'entrainement** (D27) : un challenger n'est entraine que si la fenetre
  disponible permet de calculer son propre modele nul. Sans cela, le modele promu etait parfois un
  modele que le moniteur ne savait plus surveiller — la surveillance devenait aveugle pour tout le
  reste du flux, ce qui est pire que de ne pas promouvoir. Le plancher s'applique a l'identique a P
  et a A1, sinon l'ecart P - A1 ne mesurerait plus la validation.
- **Harnais de rejeu et campagne** (`src/evaluation/replay.py`, `campaign.py`, `promotions.py`) :
  evaluation prequentielle, un seul flux par (machine, scenario, graine) rejoue tel quel par chaque
  strategie, `summary.csv` reprenable par `run_id`. Les promotions degradantes sont calculees **hors
  ligne** depuis les traces, avec scoring fantome du modele remplace : le calcul utilise les
  etiquettes, il ne peut donc pas vivre dans la strategie, et son cout est exclu du cout mesure.
- **Campagne de mise au point** (`results/summary_m2_dev.csv`, 192 rejeux = 16 machines x S1 x 3
  graines x 4 strategies, 1,0 h de calcul cumule). Elle sert a calibrer **delta** et a verifier que le
  harnais tient la distance, pas a repondre aux hypotheses. Vue d'ensemble, moyenne par machine
  d'abord (OBS06) : B0 F1 0,236 et aucun reentrainement ; B1 0,302 pour **17,5** reentrainements ;
  A1 0,279 pour 2,48 ; P 0,271 pour 1,90 reentrainement, dont 1,50 promu et 0,40 rejete. **P n'est pas
  devant B1 en F1 sur ce perimetre**, pour un dixieme du cout — c'est consigne des maintenant (D30)
  pour qu'on ne puisse pas, a M3, presenter comme une decouverte un resultat deja visible ici.
- **delta mesure et non postule** (D30) : le §7 demande de le calibrer sur l'ecart-type inter-graines
  du F1. Mesure : mediane 0,0301, q75 0,0450, **maximum 0,0997**. Retenu **delta = 0,10**, juste
  au-dessus de ce maximum, pour qu'aucune degradation declaree ne soit imputable au bruit de graine.
  La commande `mdrift promotions` sort une ligne par promotion (`results/promotions_m2_dev.csv`,
  **1 031 promotions**) : le §7 demande la **distribution** des ecarts, pas seulement leur comptage,
  et repartir des traces permet de balayer huit valeurs de delta sans relancer un rejeu. L'ordre
  P < A1 tient pour les huit. Nuance qui va au memoire telle quelle : P promeut 72 fois contre 119
  pour A1, donc sa protection vient surtout de ce qu'il **promeut moins** (9,7 % contre 12,6 % de
  promotions degradantes, un ecart bien plus faible que l'ecart en valeur absolue).
- **Parametres des strategies figes** (D30) : W = 10 080 min, V = 1 440, C = 1 440, T = 1 440,
  `min_train_minutes` = 5 450. Decouverte en figeant : **C est structurellement inerte**, parce qu'une
  promotion remet le moniteur a zero et qu'il lui faut R + K = 1 460 fenetres pour reconfirmer, soit
  plus que C. Le parametre est conserve mais **declare inerte** plutot que laisse passer pour actif,
  et un test fixe l'invariant.

Etat du code : **157 tests passent**, `ruff check` et `ruff format --check` sans remarque sur 56
fichiers. Parametres de derive et de strategies figes ; rien n'a encore ete execute sur les 12
machines d'evaluation.

## 2026-10-28 — M3 : campagne complete sur les 12 machines d'evaluation
Debut 28/10 a 16h55, fin 28/10 a 19h30. Cinq phases, une a la fois : la campagne mesure des
secondes CPU qui sont des metriques du §7, deux executions simultanees gonfleraient les deux
mesures. Push a la fin de chaque phase.

- **Phase A — campagne principale** (`results/summary.csv`, 18h13) : **960 rejeux** = 12 machines
  d'evaluation x 4 scenarios (S1-S4) x 4 strategies x 5 graines, 24 104 fenetres par flux en
  moyenne, **1,25 h de calcul cumule**. Configuration **figee avant la premiere execution**
  (D29, D30) ; aucun parametre n'a bouge pendant la campagne.
- **Phase B — derive naturelle** (`results/summary_m3_derive_naturelle.csv`, 18h28) : scenario
  `none`, sans injection, **les 28 machines** du SMD x 4 strategies, graine 42, 112 rejeux, 0,17 h.
  Le perimetre a ete **etendu** des 13 machines instables de mise au point aux 28 machines, parce que
  12 des 15 machines stables sont des machines d'evaluation : s'y limiter aurait laisse 3 machines et
  rendu la metrique de specificite du §7 inevaluable. D17 n'est pas viole — une confirmation de
  derive ne lit aucune etiquette, et aucun parametre n'a bouge ni ne bougera apres cette mesure
  (D33).
- **Phase C — tests statistiques** (`results/hypotheses.csv`, 18h40) : Wilcoxon apparie bilateral
  sur les 48 paires machine x scenario, **moyenne des graines machine par machine d'abord** (OBS06),
  Holm **par famille** d'hypothese, abstention declaree sous 5 ecarts non nuls (D31).
- **Phase D — figures et tableaux** (19h05) : `mdrift figures-m3` produit les **6 figures a 300 dpi**
  de `docs/figures/` et `docs/tableaux_chapitre6.md` (5 tableaux, 13 metriques du §7). Tout est relu
  depuis `results/` ; aucun nombre n'est recopie a la main (D34).
- **Phase E — documentation** (19h30) : D31 a D34 dans `docs/decisions.md`, cette entree,
  suppression du fichier de reprise `docs/etat_m3.md` dont le contenu definitif est ici.

### Les trois verdicts, tels qu'ils doivent etre ecrits au chapitre 6

- **H2 est verifiee, et c'est le resultat central** : P reentraine moins que B1 **48 fois sur 48**,
  facteur **12** en nombre (1,38 contre 16,25 par rejeu) comme en secondes CPU (0,33 contre 3,94),
  p < 0,0001 apres Holm, pour un F1 **indiscernable** (0,265 contre 0,252 ; p Holm 0,88). Revers a
  ecrire : B1 produit **deux fois moins** de fausses alertes (1,31 contre 2,64 par heure, 42 paires
  sur 48). Reentrainer toutes les *T* minutes recalibre le seuil en continu, ce qui resserre
  mecaniquement le taux d'alertes. **P domine B1 sur le cout, pas sur toutes les dimensions de la
  qualite.**
- **H1 n'est pas verifiee au sens large** : un seul critere sur cinq tient — fausses alertes
  **-18 %** contre le modele fige (2,64 contre 3,21 par heure, p Holm 0,0027) — et un autre part
  **a l'envers** : rappel par evenement **plus faible** que B0 (0,748 contre 0,773 ; 2 paires
  gagnantes contre 16 ; p Holm 0,0027). F1, F1 apres derive et temps de recuperation ne montrent
  aucune difference detectable. Le reentrainement declenche par derive **echange du rappel contre de
  la precision** plutot qu'il n'ameliore la detection. Ce resultat ne doit pas etre embelli : c'est
  la limite principale a porter au 6.3.3 et au 7.3.
- **H3 est verifiee sur la fiabilite, avec un cout mesure** : P n'a **jamais** plus de promotions
  degradantes que A1 (0 cas sur les 6 paires ou les deux different, p Holm 0,0235) et promeut un
  quart de fois en moins (1,15 contre 1,54 ; p Holm 0,0001, 23 paires sur 23). Mais P perd **1,2
  point de F1** (p Holm 0,0124) et **1,6 point de F1 apres derive** (p Holm 0,0047). La validation
  n'est pas gratuite sous derive injectee — alors que sur les 15 machines stables de la phase B, le
  meme garde-fou transforme 3 promotions degradantes en **0 sans aucun cout en F1**. L'arbitrage
  depend du regime, et **les deux lectures doivent figurer**.

### Specificite de la surveillance : le chiffre que la phase B apporte

**0 fausse confirmation sur les 15 machines stables**, contre 13 sur 15 a K = 30 (D19), **sans que
le moniteur devienne muet** : il confirme sur **7 des 13** machines dont le criblage avait montre
qu'elles derivent reellement entre leur moitie train et leur moitie test (OBS03). C'est ce couple de
chiffres qui rend l'etape de confirmation evaluable : un moniteur qui confirmerait tout le temps
aurait un excellent delai de detection et ne vaudrait rien.

Sur ces machines stables, P, A1 et B0 sont **identiques au chiffre pres** (F1 0,333 ; 0,58 fausse
alerte/h) parce que P n'y reentraine jamais, tandis que B1 y reentraine **16,2 fois par machine**,
y **perd** du F1 (0,295) et y alerte **davantage** a tort (0,87/h). Sur une infrastructure qui ne
derive pas, le cout de la discipline periodique est **negatif, pas neutre** — l'argument le plus
direct en faveur du declenchement par derive, et il ne demande aucune etiquette.

### Deux defauts corriges, tous deux dans la facon de rendre compte

- **La direction d'un test de Wilcoxon ne se lit pas sur la mediane des ecarts.** Des que plus de la
  moitie des paires sont a egalite — le cas normal pour `n_promotions_degradantes`, nulle sur 42
  paires sur 48 —, la mediane vaut 0 alors que le test, qui ignore les ex aequo, mesure une direction
  nette. H3 s'affichait `contraire a l'attendu` alors que P fait **mieux** que A1 : le tableau
  annoncait l'inverse du test, sur l'hypothese qui porte la contribution du memoire. La direction est
  desormais lue sur le signe de la **somme des rangs signes** et le compte `+/-` est publie (D31).
- **La courbe de la strategie proposee etait invisible** sur la figure centrale du chapitre 6 : sur un
  meme flux, P trace la meme courbe que A1 partout ou la validation n'a rien rejete, et colle a B0
  partout ou elle n'a pas reentraine. Les traits sont empiles par largeur decroissante et une note
  donne la convention (D34).

Etat du code : **175 tests passent**, `ruff check` et `ruff format --check` sans remarque sur 61
fichiers. Les resultats de M3 sont complets ; `results/summary.csv`, `results/hypotheses.csv`,
`results/summary_m3_derive_naturelle.csv`, `docs/tableaux_chapitre6.md` et les 6 figures de
`docs/figures/` sont les sources du chapitre 6.

## 2026-10-28 / 2026-11-11 — M4 : infrastructure du serveur, injecteur de pannes, enregistrement des sessions
Debut 28/10 a 18h40, fin 11/11. Cinq phases, push a la fin de chacune. M4 produit du **code et des
configurations** : le deploiement sur le serveur et l'enregistrement des vraies sessions appartiennent
a Sidy, et le depot doit lui donner **une commande par etape**, rien qui demande un reglage a la main.
Les regles du §4.3 encadrent tout le jalon.

- **Phase A — appli de test FastAPI** (28/10, 19h20, 13 tests) : charge simulee, endpoints de panne
  (`/chaos/latency`, `/chaos/errors`), metriques Prometheus, version deployee exposee en etiquette.
  L'appli **s'auto-perturbe** et son plan de controle est exempte de toute injection (D35) ; image
  construite et sondee : `/work` passe de 0,10 s a 0,44 s sous une injection de 400 ms. Deux points
  mesures ont change le code : les bornes de l'histogramme de latence, resserrees sous 100 ms, sans
  quoi une injection de latence serait invisible dans les donnees ; et un `HEALTHCHECK`, ajoute apres
  deux requetes refusees au demarrage du conteneur.
- **Phase B — pile Docker Compose** (28/10, 20h05, 12 tests) : Prometheus avec sa configuration de
  collecte, node_exporter, cAdvisor, Grafana provisionne (datasource + tableau de bord engendre,
  15 panneaux). **5 services sur 5** demarres, **4 cibles `up`** (la cible `detection` reste `down`
  jusqu'a M5), **15 des 18 requetes** de panneaux rendant des series. Quatre regles de publication
  (D36) : rien sur `0.0.0.0` par defaut, aucun port pour node_exporter et cAdvisor, pas de Grafana
  sans mot de passe, ports publies variables — parce que **8000 et 9100 etaient deja pris** par des
  services sans rapport, et que le §4.3 interdit d'y toucher. Retention Prometheus bornee en duree
  **et** en taille.
- **Phase C — injecteur de pannes** (29/10, 08h40, 39 tests) : conteneurs `stress-ng` dedies et
  limites, **deux marques exigees** (prefixe de nom **et** etiquette Docker), `make stop-chaos`,
  Locust, ecriture des labels horodates. Les cinq types du §4.2 ont ete **reellement injectes** sur
  une machine faisant tourner 14 conteneurs dont 9 etrangers au memoire : `pic_cpu` 98,77 % de CPU et
  16,69 MiB sur 256 MiB ; `fuite_memoire` 205,9 MiB sur 256 MiB sans OOM-kill ; `latence` +250 ms ;
  `erreurs` 30 % de 5xx ; `saturation` 230 requetes/s et p95 de 0,10 s a 0,234 s. Deux conteneurs
  leurres, chacun portant une seule des deux marques, ont ete **refuses** et ont survecu (D37).
- **Phase D — enregistrement et rejeu des sessions** (11/11, 50 tests ajoutes) : lecture de
  `query_range` vers Parquet, chargeur de session, et `build_server_stream` qui rend un flux que le
  **harnais de M2 traite sans modification**. Quatre regles d'alignement temporel sont posees
  explicitement et tenues par des tests : grille d'instants multiples du pas, bornes d'anomalie
  incluses, anomalie plus courte que le pas signalee, part d'entrainement refusee si contaminee (D38).
  Trois commandes : `session-check`, `session-record`, `session-info`.
- **Phase E — documentation** (11/11) : D35 a D38 dans `docs/decisions.md`, cette entree, suppression
  du fichier de reprise `docs/etat_m4.md` dont le contenu definitif est ici.

### Ce que M4 apporte au memoire
La **description des donnees reelles** du chapitre 5 : l'architecture de la pile (5.1), l'appli de
test et ses endpoints (5.2), le tableau des cinq types d'anomalies avec leurs mesures reelles et les
regles de securite du §4.3 (5.3). Les captures Grafana illustreront le chapitre 5 une fois la pile
deployee. La source `serveur` est branchee sur le harnais de M2 : une session enregistree se rejoue
comme une machine du SMD, ce qui est la condition du test final de M6.

### Le defaut le plus grave du jalon, et pourquoi il ne pouvait etre trouve qu'en lancant
L'injecteur **attendait l'echeance a l'aveugle** avant d'ecrire l'etiquette. Un `make stop-chaos` a
90 s sur une injection de 240 s produisait une etiquette de **240 s** : une anomalie etiquetee la ou
il ne s'etait plus rien passe pendant deux minutes. Une etiquette qui decrit mal ce qui a eu lieu est
**pire** qu'une etiquette absente : elle produit un faux resultat au lieu d'un resultat manquant.
L'attente surveille desormais le conteneur et l'etiquette porte la duree **reellement subie**
(mesure : arret detecte en 4 s, etiquette de 96 s pour une demande de 240 s). Aucun test ne l'avait
vu, parce que tous passaient par un client Docker factice ou le conteneur ne meurt jamais tout seul.
Trois autres defauts du meme genre ont ete corriges : OOM-kill silencieux sur une fuite memoire trop
intense, `chaos-stop` qui annoncait un succes alors que la remise a zero avait echoue dans un `except`
muet, et une URL par defaut qui pointait vers un service tiers. A l'inverse, le defaut de la phase D
— `float("+Inf")` qui reussit, donc une valeur non finie recopiee dans les donnees — a ete trouve
**par un test**, qui attendait 1,0 et a obtenu 0,5.

### Trois choses que seul Sidy peut faire, et dans cet ordre
1. Deployer la pile sur le serveur Linux et **verifier les deux panneaux par conteneur** (CPU et
   memoire), vides sous Docker Desktop parce que cAdvisor n'y expose pas l'etiquette `name`. Ce sont
   eux qui montrent qu'une injection reste dans son conteneur et sous sa limite : a verifier **avant**
   la premiere session, pas apres.
2. Verifier `make stop-chaos-force` sur le serveur (`xargs -r` est une extension GNU) et lancer
   `uv run mdrift session-check`, qui refuse d'enregistrer si une metrique obligatoire est muette.
3. Enregistrer les sessions avec `session-record`, en commencant par une **periode calme d'au moins
   40 %** du flux : le chargeur refuse une session dont la part d'entrainement contient une anomalie
   ou une derive.

Etat du code : **289 tests passent** (175 a la fin de M3), `ruff check` et `ruff format --check` sans
remarque sur 85 fichiers.

## 2026-11-11 / 2026-11-12 — M5 : boucle en ligne, registre de modeles, exporter, tableaux de bord, notifications
Debut 11/11, fin 12/11. Cinq phases, push a la fin de chacune. M5 realise le **mode en ligne** du §2
avec **le meme code** que le mode rejeu : `src/features/`, `src/models/`, `src/drift/` et
`src/strategies/` ne sont pas reecrits, et le cycle de reentrainement est desormais un module partage
(`src/strategies/retraining.py`) auquel le harnais de rejeu delegue. Une divergence entre les deux
modes rendrait le chapitre 5 incomparable au chapitre 6, qui est tout l'interet d'avoir les deux.

- **Phase A — registre de modeles** (11/11, 39 tests) : wrapper MLflow (tracking + registry, alias
  `champion`), service `mlflow` dans Compose, et le **contrat de features** condense en une empreinte
  sha256 enregistree avec chaque version et **verifiee au chargement** (D39). L'artefact est le
  detecteur entier — standardisation, foret, seuil — dans le meme joblib que relit le rejeu. Si le
  registre refuse une promotion, le challenger est jete et le champion reste en service : le registre
  fait foi. Commandes `registry-init` et `registry-info` ; tests contre un vrai registre sqlite.
- **Phase B — boucle de detection en ligne** (11/11, 36 tests) : un tick lit les nouveaux points de
  Prometheus, construit les fenetres completes, score avec le champion, consulte la strategie, et
  reentraine / valide / promeut quand elle le demande. Quatre regles non evidentes et non rattrapables
  (D40) : le **prechargement ne score pas**, la **reference du moniteur vient du registre**, un point
  manquant **au bord** fait sauter la fenetre au lieu d'etre extrapole, et le registre fait foi. Le
  journal JSON lignes du §9 **refuse** tout nom d'evenement hors vocabulaire. `window_ms_mean` et
  `window_ms_p95` sont enfin mesures par fenetre, ce que le rejeu par blocs ne peut pas donner : ce
  sont eux qui repondent aux indicateurs I7 / T1 du chapitre 2.
- **Phase C — exporter Prometheus et tableau de bord du pipeline** (12/11, 47 tests) : `/metrics`
  ouvert **avant** le premier tick (un service qui detecte sans etre observable ne produit aucune
  capture), conteneur `detection` sans aucun port publie, journal sur volume nomme. Trois invariants
  (D41) : l'exporter **s'abonne au journal** au lieu de compter en parallele, **toutes les series
  valent zero** des la construction, et la fraicheur exposee est **l'instant de la fenetre** et non
  celui du scrape. Second tableau de bord `pipeline.json` (15 panneaux) : `infrastructure.json` montre
  la cause, `pipeline.json` l'effet. Les tests les plus utiles comparent le tableau de bord a
  l'exporter — chaque serie `mdrift_*` citee en PromQL doit exister dans le rendu — et verifient que
  `prometheus.yml`, Compose et `serving.yaml` s'accordent sur le port.
- **Phase D — notifications Telegram** (12/11, 30 tests) : second abonne du journal. Les messages sont
  **construits depuis le journal**, donc les captures Telegram du chapitre 5 se retrouvent ligne par
  ligne dans le journal remis au jury ; seules des **cles declarees** entrent dans un message ;
  **au plus un message par minute et par type** (§9), le premier partant immediatement ; aucun
  `parse_mode` ; un echec d'envoi est compte, jamais leve (D42). Jeton et conversation par
  l'environnement uniquement, `actif: false` dans le depot, `mdrift notify-test` pour verifier le
  chemin reel avant une demonstration. Aucun test n'ouvre de connexion : transport injecte, horloge
  pilotee par le test.
- **Phase E — documentation** (12/11) : D39 a D42 dans `docs/decisions.md`, cette entree, suppression
  du fichier de reprise `docs/etat_m5.md` dont le contenu definitif est ici.

### Ce que M5 apporte au memoire
Les **captures du pipeline MLOps du chapitre 5** et la matiere du 5.1 / 5.2 : l'architecture du
service de detection, le registre et l'alias `champion`, les metriques exposees, les deux tableaux de
bord et les notifications. M5 apporte aussi un chiffre que le chapitre 6 ne pouvait pas produire —
`window_ms_p95`, le temps de traitement d'une **vraie** fenetre, reentrainement exclu — qui est la
mesure attendue par l'indicateur I7 du 2.7 : traitement < intervalle de collecte.

### Le point le plus consequent du jalon
La reference du moniteur de derive est **telechargee depuis le registre**, pas recalee sur les
premieres fenetres du service. Recaler reviendrait a comparer le present a lui-meme : une derive
survenue pendant un arret du service — mise a jour, redemarrage de la machine, c'est-a-dire
precisement les moments ou le comportement normal change — serait **definitivement invisible**, sans
que rien ne le signale. La surveillance tournerait, les panneaux resteraient verts, la strategie
proposee ne se declencherait jamais, et le chapitre 5 montrerait un pipeline qui ne fait rien tout en
ayant l'air de fonctionner. Le service **refuse donc de demarrer** sur un champion sans reference.
Le meme raisonnement vaut pour le contrat de features (D39) : des colonnes qui ont change de sens ne
font lever aucune erreur a un `IsolationForest`, qui continue a produire des scores, des alertes et
des captures, tous faux.

### Trois choses que seul Sidy peut faire, et dans cet ordre
1. `docker compose up -d mlflow` puis `uv run mdrift registry-info` depuis l'hote : c'est le chemin
   complet (serveur -> client -> telechargement d'artefact) que les tests sqlite ne traversent pas.
2. `docker compose build detection` puis `docker compose up -d detection`, et verifier la cible
   `detection` en **UP** dans Prometheus (Status -> Targets). C'est la seule etape qui prouve que
   l'exporter est joignable depuis le reseau interne.
3. Creer le bot avec @BotFather, mettre `TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID` dans `.env`, passer
   `telegram.actif` a `true`, puis `uv run mdrift notify-test`. A refaire juste avant la soutenance :
   un jeton revoque ne se voit qu'a l'envoi.

Etat du code : **441 tests passent** (289 a la fin de M4), `ruff check` et `ruff format --check` sans
remarque sur 98 fichiers.

## 2026-11-13 — Correction d'un defaut de la campagne M3 : les bornes du garde-fou (D43)
La relecture du memoire a trouve ce que ni les tests ni les resultats n'avaient signale : la campagne
M3 a tourne avec les bornes du garde-fou **du cadrage** (`300 / 0,001 / 0,30`) et non avec celles que
D14 avait calibrees le 26/10 sur les 473 challengers du perimetre de mise au point (`256 / 0,0 /
0,90`). D30 a fige les premieres **en citant D14**, qui recommandait justement les secondes et
ecrivait que celles du cadrage « font plus de mal que de bien ».

Conduite de la correction, dans cet ordre :
1. **Verification independante** du retour avant d'y toucher : 54 challengers rejetes sur les 12
   machines d'evaluation (36 sous la borne basse, 18 au-dessus de 0,30, taux maximal observe 0,7069),
   19 de plus sur la mise au point. Aucun rejet ne vient du critere d'historique, `min_train_minutes`
   le dominant : l'effet entier vient des bornes du taux d'alertes.
2. **Sauvegarde** des resultats du garde-fou strict sous `results/*_p_strict.csv` — ils deviennent une
   analyse de sensibilite, ils ne sont pas jetes.
3. **Correction** des deux configurations et des defauts du code, chaque valeur portant desormais sa
   justification chiffree a cote d'elle.
4. **Rejeu de P seule** sur les 240 flux d'evaluation (B0, B1 et A1 ne dependent pas de ces bornes)
   puis sur les 28 machines du scenario `none`, et regeneration de `hypotheses`, `promotions`,
   `figures-m3` et des tableaux du chapitre 6.

Resultat : **0 rejet sur 240 rejeux**, exactement ce que D14 predisait. P devient identique a A1 sur
les scenarios injectes, a 0,06 seconde CPU pres — le cout de scorer la fenetre reservee. Le F1 de P
passe de 0,265 a **0,277**, les fausses alertes de 2,64 a **2,07/h**. H1 et H2 sont confirmees plus
nettement qu'avant ; **H3 devient indecidable**, et c'est le resultat : avec des bornes calibrees sur
donnees reelles, le garde-fou ne se declenche **jamais** sous derive injectee.

Le seul endroit ou il agit encore est la derive naturelle, et il y nuit : sur `machine-3-5`, un
challenger a 92,57 % d'alertes est rejete alors que le champion qu'il devait remplacer alerte
lui-meme sur 90,62 % des fenetres, et le F1 de P tombe a 0,072 contre 0,156 pour A1. Une borne
**relative au champion** aurait laisse passer ce challenger ; le reglage est gele, la piste va aux
perspectives du chapitre 7 avec le chiffre a l'appui.

Deux defauts secondaires corriges au passage : `mdrift promotions` melangeait les machines de mise au
point et d'evaluation (option `--dataset-config` ajoutee), et D32 contenait deux erreurs de redaction
signalees par la relecture (un resultat des machines instables attribue aux stables, une moyenne
appelee mediane). D30, D32 et D33 portent desormais la trace du defaut a l'endroit exact ou il a ete
commis : un journal de decisions qui se corrige en silence ne vaut rien.

Ce que l'incident coute et ce qu'il apprend : une valeur calibree qui n'est tenue par aucun test n'est
pas figee, elle est seulement ecrite. La contradiction se tenait entre deux decisions et deux
fichiers de configuration, chacun coherent pris seul — exactement ce qu'une relecture ne voit pas. Le
test `test_les_bornes_figees_sont_celles_calibrees_et_non_celles_du_cadrage` ancre maintenant les
trois valeurs dans les configurations **et** dans l'objet construit. A declarer comme limite
methodologique au 6.3.3 et au 7.3.

Etat du code : **442 tests passent**, `ruff check` et `ruff format --check` sans remarque sur
98 fichiers.

## 2026-11-13 — M7, partie CI : trois controles, deux declencheurs (D44)
M7 est pris maintenant, hors de l'ordre des jalons, parce que c'est la seule phase qui n'a pas besoin
du serveur : M6 attend Sidy. Il en reste la video de demonstration, qui attend elle aussi le serveur.

La CI verifie **trois** choses, et ce sont trois cibles du Makefile pour que la CI et le poste de
developpement ne puissent pas verifier deux choses differentes :
- `make lint` — `ruff check` et `ruff format --check` ;
- `make test` — les 454 tests ;
- `make build` — le wheel construit, installe dans un environnement **neuf hors du depot**, et le
  point d'entree `mdrift` appele depuis le dossier d'installation. Construire le wheel ne prouve
  rien : ce qui casse est son **contenu**. `pyproject.toml` declare `packages = ["src", "infra"]`
  parce que le CLI importe `infra.injector` ; un paquet qui n'embarquerait que `src` s'installerait
  sans erreur et echouerait a la premiere commande d'injection. Verifie : 50 fichiers sous `src/`,
  23 sous `infra/`, `mdrift --help` repond, `import src.cli, infra.injector` passe.

**Le depot est sur GitHub, pas sur GitLab.** Le cadrage demande GitLab CI et le memoire le decrit
ainsi ; un `.gitlab-ci.yml` ne declenche rien sur GitHub, donc le jalon ne produirait aucune capture
de pipeline, qui est precisement son livrable. Les deux fichiers existent : `.gitlab-ci.yml` est le
livrable demande et tourne tel quel sur une instance GitLab, `.github/workflows/ci.yml` est celui qui
s'execute. Ils appellent les memes cibles, et `tests/test_ci.py` (12 tests) tient l'invariant : memes
cibles, meme version de `uv` epinglee des deux cotes, installation depuis `uv.lock` avec `--frozen`,
aucun `campaign` ni `smd-download`, aucun `docker push`, et YAML valide des deux cotes — un
`.gitlab-ci.yml` casse ne se verrait autrement qu'au prochain push sur GitLab, que personne ne fera
avant la soutenance. C'est la lecon de D43 appliquee le jour meme : deux fichiers qui decrivent le
meme controle divergent en silence.

Ce que la CI **ne** fait pas, et qui doit etre dit au memoire aussi bien que ce qu'elle fait : elle ne
rejoue pas la campagne M3 (plusieurs heures) et ne telecharge pas le SMD (§4 : les donnees ne sont pas
dans le depot). Le seul test qui exige les donnees se declare `skipped` au lieu d'echouer — une CI
rouge pour une donnee absente devient une CI qu'on cesse de lire. La reproductibilite des chiffres
vient de `uv.lock` et des configurations sauvegardees a cote de chaque execution, pas de la CI.

### Ce que Sidy doit faire pour clore M7
1. Pousser sur `main` et verifier le pipeline **vert** dans l'onglet Actions de GitHub, puis en faire
   une capture recadree (sans nom de compte si possible) : c'est le livrable du jalon.
2. Si le memoire doit montrer une interface **GitLab**, creer un miroir du depot sur gitlab.com et y
   pousser une fois : `.gitlab-ci.yml` est pret et n'a besoin d'aucun secret. A dire tel quel dans le
   memoire, miroir ou pas.
3. La video de demonstration, apres M5 et M6 : pile en ligne, tableaux de bord, une anomalie injectee,
   une derive confirmee, une promotion, une notification Telegram.

Etat du code : **454 tests passent** (442 avant la CI), `ruff check` et `ruff format --check` sans
remarque sur 99 fichiers.

## Prochaine etape : M6
Test final sur infrastructure reelle, a faire avec Sidy sur le serveur : enregistrer des sessions
portant des **derives reelles** (deploiement d'une nouvelle version d'une appli de test, changement
des ressources d'un conteneur, montee durable du trafic) et des **anomalies injectees** horodatees,
puis rejouer B0, B1, P et A1 sur ces sessions avec le harnais de M2 et les parametres **figes** depuis
M3. Livrable memoire : la validation sur infrastructure reelle du chapitre 6, a comparer aux resultats
SMD. Rappel du §4.3 : injections uniquement dans les conteneurs dedies et limites, jamais sur une
application en production, et aucune IP, nom d'hote ou domaine dans les resultats ni les captures.

## 2026-11-15 / 2026-11-22 (en cours) — M6 : bilan factuel de la session d'enregistrement a ce jour

Journal factuel, sans interpretation au-dela de ce que les fichiers permettent de dire. Le
run d'enregistrement de session est `m6_session_01` ; T0 reel : **2026-11-15T02:14:32Z** ; fin
prevue, inchangee : **2026-11-25T02:14:32Z** (environ 9 jours 24 h). Frontiere 40 % / 60 %
train / evaluation : **2026-11-19T19:11:49Z**.

**Anomalies injectees jusqu'ici** (`data/labels/m6_session_01.csv`) :

| Type | Debut (UTC) | Fin (UTC) | Intensite |
|---|---|---|---|
| pic_cpu | 2026-11-20T23:40:12Z | 2026-11-20T23:43:12Z | 2 |
| fuite_memoire | 2026-11-21T06:26:05Z | 2026-11-21T06:30:05Z | 192 |
| latence | 2026-11-22T19:57:49Z | 2026-11-22T19:59:49Z | 200 ms |

Restent a injecter avant le 24/11 au soir : `erreurs` (programmee ce soir, 21:57Z) et `saturation`
(demain matin).

**Derive reelle confirmee** (`data/labels/m6_session_01_drift.csv`) : **2026-11-21T12:11:56Z**,
deploiement de `memoire-app` 1.0.0 → 1.1.0 (`APP_BASE_LATENCY_MS` 25 → 45, `APP_LATENCY_SPREAD`
0,35 → 0,5). Une seule derive reelle a ce jour.

**Ce que le service en ligne (`memoire-detection`, pile de demonstration du chapitre 5) a fait
depuis le demarrage, avec les heures** :
- **2026-11-16T06:36:20Z** : premier demarrage du service, apres le prefill initial.
- **2026-11-16T06:36:20Z → 2026-11-22T21:15:02Z (six jours)** : **rien**. Defaut dans le
  comblement des trous de la boucle en ligne (`_combler`, D47) : les deux colonnes facultatives
  `reseau_entrant`/`reseau_sortant`, toujours vides sur cet hote (D46), faisaient refuser **chaque**
  fenetre au lieu d'etre mises a zero comme le prevoyait deja D46 x D38 pour le chargeur de
  sessions. Consequence : aucune fenetre scoree, aucune alerte, aucun signal ni confirmation de
  derive, aucun reentrainement, aucune promotion ni rejet pendant toute cette periode — y compris
  pendant les trois anomalies injectees ci-dessus et la derive reelle du 21/11. Le service n'a
  donc techniquement rien vu de ce qui s'est passe sur cette fenetre de six jours ; les trois
  anomalies et la derive restent neanmoins mesurables **hors ligne**, dans les metriques
  Prometheus elles-memes, pour le rejeu M6 du chapitre 6, qui ne depend pas de ce service.
- **2026-11-22T21:15:02Z** : correction deployee (D47) et service redemarre ; **21:15:20Z**,
  premiere fenetre reellement scoree de toute la session (position 399, score 0,699 contre seuil
  0,555 — alerte). Un second redemarrage a 21:20:05Z (ajout mediane/max du temps de traitement,
  sans rapport avec D47) a repris le scoring normalement (`online_20261122T212005Z`).
- **21:15Z → 06:21Z le 23/11 (toute la nuit)** : chaque fenetre alerte sans interruption (score
  entre 0,67 et 0,70 contre un seuil de 0,555, position 940 a 06:21Z). L'injection `erreurs`
  programmee la veille a bien tourne, sans intervention manuelle : **2026-11-22T21:57:08Z →
  21:59:08Z**, intensite 0,2, etiquetee dans `data/labels/m6_session_01.csv`.
- **2026-11-23T05:39:20Z** : premier `drift_signal` journalise (position 898, 500e fenetre
  reellement scoree depuis 21:20:05Z) : **100 % des 24 features utilisables** au-dela du seuil KS
  (`statistic_median: 0,9541`), `streak: 1` — pas encore de persistance. Aucun `drift_confirmed`,
  `retrain_start/end`, `promoted` ni `rejected` n'a ete journalise a ce jour.
  `configs/drift.yaml` exige 960 fenetres consecutives (`confirmation_windows`, ~16 h) : si le
  streak se maintient sans interruption, une confirmation n'arriverait pas avant environ
  **2026-11-23T21:39Z** — apres l'echeance du 23/11 14:00Z. A ne pas presenter comme acquis avant
  verification continue.

**Consequence pour les captures du 23/11 au matin** : le tableau de bord Grafana « pipeline »
(score, alertes, derive — alimente par ce service) est vide par construction avant
2026-11-22T21:15Z ; une capture doit couvrir une plage posterieure a cette heure, pas la derive du
21/11 ni les deux premieres anomalies. Le tableau « infrastructure » (metriques brutes Prometheus)
n'a pas cette limite et montre normalement toute la session depuis le 15/11.

## 2026-11-26 — Premier rejeu reel de `m6_session_01`, et diagnostic du service en ligne (D50, D51)

**Rejeu** : `mdrift session-replay m6_session_01` (B0/B1/P/A1), `session-report` et `session-figure`
executes sur le serveur, deploiement prealable par `git pull --ff-only` (commit `7d9f891`, aucun
conflit avec la modification locale non commitee `configs/notify.yaml:actif=true`). Resultats et
figure dans `results/session_reports/m6_session_01/` et
`docs/figures/scores_session_m6_session_01.png` — detail chiffre en D50.

**Diagnostic demande par Sidy sur le service en ligne, a partir de l'archive de l'ancien run**
`online_20261122T212005Z` (volume `memoire-detection-data`, extraite sans toucher au conteneur
actif, archivee dans `data/archives/online/`) **et du run courant** `online_20261125T062356Z` :
- **Alertes par heure, avant/apres la confirmation de derive du 23/11 21:38:20Z** : ~60/h des deux
  cotes (1463 alertes sur 24,4 h avant, 1966 sur 32,75 h apres) — **aucune difference**, parce que
  la confirmation n'a jamais declenche de reentrainement effectif (voir D51). Score constant entre
  0,62 et 0,73 (mediane 0,674) contre un seuil fixe a 0,555387 du debut a la fin des deux runs :
  essentiellement chaque fenetre alerte, confirmant l'hypothese de Sidy — le champion en service
  (version 1) a ete entraine avant la derive reelle du 21/11 et n'a jamais ete reentraine depuis.
- **Injections apres la fin de session (2026-11-25T02:14:32Z)** : aucune. La derniere anomalie
  etiquetee (`data/labels/m6_session_01.csv`) se termine le 24/11 a 19:03:08Z ; `mdrift chaos-list`
  et `docker ps` confirment zero conteneur d'injection residuel au 26/11.
- **Le plancher de reentrainement a ete repousse par un redemarrage non programme** du conteneur
  `memoire-detection`, survenu le 25/11 a 06:23:59Z (cause non identifiee depuis ce poste). Le
  compteur `n_train_available` (jamais persiste) est reparti du prechargement habituel (~400
  fenetres) au lieu de continuer depuis ou il en etait ; la derive a ete re-confirmee le 26/11 a
  06:42:20Z, de nouveau refusee par le garde-fou (`n_train_available: 418` contre
  `min_train_windows: 5450`). Plancher desormais attendu vers le **28/11 en fin d'apres-midi UTC**
  si le service n'est plus redemarre d'ici la, et non le 27/11 09:30 initialement attendu par Sidy.
  Decision : ne pas redemarrer `memoire-detection` avant cette echeance ; le correctif du
  notificateur (notifications par episode plutot que par fenetre) est code et teste mais pas
  deploye, pour la meme raison.
