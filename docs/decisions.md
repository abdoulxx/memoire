# Journal des choix techniques

> Un choix = 3 lignes : le choix, les alternatives ecartees, la raison.
> Alimente les justifications du chapitre 4 du memoire (CLAUDE.md §11).

## 2026-10-22 — D01 : gestionnaire d'environnement `uv` + Python 3.11
- **Choix** : `uv` avec `.python-version` = 3.11 (3.11.15 installee par uv), dependances declarees dans `pyproject.toml` et figees dans `uv.lock`.
- **Alternatives ecartees** : `venv` + `pip` + `requirements.txt` (pas de resolution deterministe) ; conda (lourd, non installe) ; Python 3.13/3.14 deja presents sur la machine.
- **Raison** : reproductibilite exigee par le livrable 2 ; 3.11 est la version ciblee par la CI et celle pour laquelle les roues de `river`, `mlflow` et `scikit-learn` sont les plus stables.

## 2026-10-22 — D02 : point d'entree CLI Python au lieu de `make`
- **Choix** : un CLI `mdrift` (Typer), une sous-commande par experience ; le `Makefile` n'est qu'une enveloppe pour la CI Linux.
- **Alternatives ecartees** : `make` seul (absent sous Windows, poste de developpement principal) ; scripts `.ps1` (non portables vers la CI).
- **Raison** : « une commande par experience » doit fonctionner a l'identique sur le poste de developpement et dans la CI.

## 2026-10-22 — D03 : paquet nomme `src`
- **Choix** : les modules sont importes via `src.<module>` ; `pythonpath = ["."]` pour pytest, `packages = ["src"]` pour le build.
- **Alternatives ecartees** : layout `src/memoire_drift/` (plus conventionnel) ; modules a la racine.
- **Raison** : conserver a l'identique l'arborescence decrite dans le cadrage du projet, qui est reprise telle quelle dans le memoire (chapitre 5).

## 2026-10-22 — D04 : dependances installees au fur et a mesure des jalons
- **Choix** : M0 n'installe que numpy, pandas, scikit-learn, scipy, pyyaml, typer, matplotlib, pyarrow. `river` et `mlflow` arrivent a M2/M5.
- **Alternatives ecartees** : installer toute la pile des le depart.
- **Raison** : regle n°1 du cadrage (« le pipeline est un moyen, pas le sujet ») ; chaque dependance doit se justifier au moment ou elle sert, et l'environnement reste rapide a reconstruire.

## 2026-10-22 — D05 : telechargement selectif du SMD plutot que clone du depot
- **Choix** : telechargement fichier par fichier depuis `raw.githubusercontent.com`, limite aux machines listees dans `configs/datasets/smd.yaml` (3 machines = 55,5 Mo), avec manifeste `manifest.json` (taille + SHA-256 de chaque fichier).
- **Alternatives ecartees** : `git clone` du depot OmniAnomaly (environ 500 Mo pour 28 machines, dont du code TensorFlow 1.x inutile) ; versionnement des donnees dans le depot (interdit par le .gitignore et inutile).
- **Raison** : le perimetre de M0-M2 est de 3 machines ; le manifeste rend le jeu de donnees utilise identifiable dans le memoire et verifiable a la reexecution.

## 2026-10-22 — D06 : intervalle d'echantillonnage du SMD fixe a 60 secondes
- **Choix** : `sampling_interval_seconds: 60`. Toutes les metriques temporelles (fausses alertes par heure, delai de detection, temps de recuperation) en decoulent.
- **Alternatives ecartees** : aucune — la valeur n'etait pas documentee dans le README du depot et devait etre etablie.
- **Raison** : la litterature decrivant le SMD indique une collecte a la minute sur 5 semaines ; verification empirique concordante : machine-1-1 compte 28 479 + 28 479 points, soit 39,6 jours (5,7 semaines) a un point par minute. Sources : Kim et al., *Towards a Rigorous Evaluation of Time-Series Anomaly Detection*, AAAI 2022 ; Su et al., KDD 2019.

## 2026-10-22 — D07 : un evenement d'anomalie = une plage contigue de points etiquetes 1
- **Choix** : les evenements sont derives de `test_label` par decoupage en plages contigues (`events_from_label`). `interpretation_label` n'est pas utilise pour definir les evenements, mais conserve pour savoir quelles dimensions sont impliquees.
- **Alternatives ecartees** : utiliser directement les plages de `interpretation_label` comme definition des evenements.
- **Raison** : la definition doit decouler des etiquettes qui servent au calcul du rappel par fenetre, pour que les deux familles de metriques reposent sur la meme verite terrain. Controle de coherence effectue : le nombre de plages contigues egale exactement le nombre de lignes de `interpretation_label` sur les trois machines (8, 13 et 4).

## 2026-10-22 — OBS01 : le SMD est deja normalise a la source (limite a declarer)
- **Observation** : les valeurs de `train` et `test` sont deja ramenees dans l'intervalle [0, 1] par les auteurs du dataset, vraisemblablement par une mise a l'echelle min-max calculee sur l'ensemble de la serie.
- **Consequence** : la regle « normalisation ajustee sur les donnees d'entrainement uniquement » s'applique a notre propre standardisation des features derivees (moyennes, ecarts-types, maxima glissants), mais pas a la mise a l'echelle d'origine, qui a deja vu la moitie de test.
- **A declarer** : limite methodologique a mentionner au 6.3.3 du memoire (fuite d'information residuelle, inherente au dataset public et subie par tous les travaux qui l'utilisent). Elle n'avantage aucune des strategies comparees, puisqu'elles partagent exactement les memes donnees d'entree.

## 2026-10-22 — OBS02 : divergence de la litterature sur le nombre de metriques du SMD
- **Observation** : certaines publications decrivent le SMD avec 33 capteurs, d'autres avec 38.
- **Fait verifie localement** : les fichiers telecharges comportent **38 colonnes** (verification automatisee dans `tests/test_smd_loader.py`).
- **A declarer** : annoncer 38 metriques dans le memoire, en precisant qu'il s'agit d'un comptage direct sur les fichiers du depot officiel.

## 2026-10-22 — D08 : taille de fenetre w = 10 minutes, pas de 1 minute
- **Choix** : `window_size: 10`, `stride: 1`, agregats `last`, `mean`, `std`, `max` par metrique, soit 152 features par fenetre.
- **Alternatives ecartees** : w = 30 min (rappel par fenetre plus facile a obtenir, donc chiffres moins comparables a la litterature) ; w = 5 min (statistiques glissantes trop bruitees) ; fenetres disjointes (delai de detection mesurable seulement au multiple de w, et ecart avec le mode en ligne).
- **Raison** : mesure de l'effet de dilatation du label, w = 10 ne gonfle le taux de fenetres positives que de 1,03x a 1,12x par rapport au taux de points anormaux (9,46 % -> 9,72 % ; 4,94 % -> 5,43 % ; 1,07 % -> 1,20 %). Le pas de 1 minute rend le mode rejeu strictement coherent avec la boucle en ligne, qui interrogera Prometheus chaque minute.

## 2026-10-22 — D09 : niveau `machine` ajoute dans l'arborescence des resultats
- **Choix** : `results/runs/<source>/<machine>/<scenario>/<strategie>/<seed>/`.
- **Alternatives ecartees** : l'arborescence du cadrage, sans niveau machine.
- **Raison** : la campagne porte sur plusieurs machines SMD ; sans ce niveau, les executions de machines differentes se recouvriraient. Le `run_id` et `summary.csv` portent egalement la machine.

## 2026-10-22 — D10 : scoring par blocs entre deux reentrainements
- **Choix** : le harnais score en un appel toutes les fenetres separant deux reentrainements, puis deroule la boucle de decision sequentiellement.
- **Alternatives ecartees** : un appel a `score()` par fenetre.
- **Raison** : entre deux reentrainements le modele en service est constant, le resultat est donc **exactement identique** au scoring point par point, pour un temps de calcul reduit de plusieurs ordres de grandeur (1,5 s par machine au lieu de plusieurs minutes). La contrainte prequentielle est preservee : la decision reste sequentielle et le reentrainement ne recoit que l'historique deja observe (verifie par test).

## 2026-10-22 — D11 : fausses alertes definies comme les faux positifs par fenetre
- **Choix** : une fausse alerte est une alerte emise sur une fenetre etiquetee negative ; la tolerance d'attribution ne sert qu'au rappel par evenement et au delai de detection.
- **Alternatives ecartees** : definir une fausse alerte comme une alerte non attribuable a un evenement (deux verites terrain concurrentes dans le memoire).
- **Raison** : avec `tolerance = w - 1`, la zone d'attribution d'un evenement coincide exactement avec les fenetres positives qu'il engendre ; les trois familles de metriques reposent alors sur une seule et meme verite terrain. Propriete verifiee par test (`test_coherence_des_trois_familles`).

## 2026-10-22 — OBS03 : derive naturelle entre les deux moities du SMD (resultat majeur)
- **Observation** : le seuil est calibre pour 1 % d'alertes sur les donnees d'entrainement. Sur la moitie `test`, le modele fige produit 12,88 % d'alertes (machine-1-1), 2,24 % (machine-2-1) et **45,38 %** (machine-3-1), soit 12,9x, 2,2x et 45,4x le taux de calibrage. La mediane des scores monte systematiquement entre les deux moities (ex. 0,4404 -> 0,5402 sur machine-3-1).
- **Interpretation** : les deux moities du SMD ne suivent pas la meme distribution. Une derive de concept **naturelle** existe donc dans le dataset, sans aucune injection de notre part. La figure `taux_alertes_machine-3-1_none_static_seed42.png` en montre la forme : taux d'alertes proche de zero pendant 90 heures, puis montee progressive (derive incrementale) avec une oscillation periodique (derive recurrente), jusqu'a saturation a 100 % d'alertes apres 320 heures — le modele fige devient alors totalement inutilisable.
- **Portee pour le memoire** : c'est une justification empirique de la problematique, obtenue sur donnees reelles publiques et non sur un cas construit. A exploiter au 6.2.1 et en introduction.
- **Consequence methodologique** : les scenarios S1 a S4 (M1) s'ajouteront a cette derive naturelle. Il faudra le dire explicitement et, a M2, quantifier la derive naturelle avec les memes detecteurs (PSI, KS, ADWIN) que ceux de la strategie proposee.
