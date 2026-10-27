# Journal d'avancement

> Dates de debut et de fin de chaque jalon -> planification et diagramme de Gantt (memoire, 2.6).

| Jalon | Debut | Fin | Etat |
|---|---|---|---|
| M0 — Depot, SMD, features, Isolation Forest, evaluation sans derive | 2026-10-22 | 2026-10-22 | termine |
| M0+ — Mesures preparatoires : derive naturelle a 28 machines, regles de P | 2026-10-22 | 2026-10-22 | termine |
| M0++ — Calibrage des etapes 3 et 4 sur le perimetre de mise au point | 2026-10-26 | 2026-10-26 | termine |
| M1 — Injection de derive S1 a S4, calibrage de l'amplitude, figures | 2026-10-26 | 2026-10-26 | termine |

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

## Prochaine etape : M3
Campagne complete sur le perimetre d'evaluation : 12 machines x 4 scenarios x 5 graines x 4
strategies, a configuration **figee** (D29, D30). Puis les tableaux et figures du chapitre 6, le test
de Wilcoxon apparie sur les graines pour H1 et H2, et le comptage des promotions degradantes pour H3.
L'etude de la derive naturelle (scenario `none`, les 13 machines instables de mise au point, quatre
strategies) est disponible et reste a lancer : c'est elle qui donne les fausses confirmations de
derive du §7, en complement des 3 machines stables sur lesquelles le calibrage a pu s'appuyer.
