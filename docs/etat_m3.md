# Etat de la campagne M3 — fichier de reprise

But de ce fichier : permettre de reprendre le travail **sans contexte**, apres une
interruption (limite d'usage atteinte, machine redemarree, session perdue). Il est
mis a jour a la fin de chaque phase et pousse avec elle. Il sera supprime quand M3
sera clos — son contenu definitif va dans `docs/journal.md`.

Fenetre de travail autonome accordee : du **28/10/2026 16h55** au **29/10/2026 04h00**
(GST). Reinitialisations de limite annoncees : 18h10, puis 23h10, puis 04h10.

## Regles qui ne changent pas

- `configs/drift.yaml` et `configs/strategies/*.yaml` sont **figes** (D29, D30).
  Aucun parametre ne bouge pendant M3, quel que soit le resultat. Un resultat qui
  deplait se commente, il ne se corrige pas.
- **Une phase a la fois, jamais deux en parallele.** La campagne mesure des
  secondes CPU qui sont des metriques du §7 : deux executions simultanees
  gonfleraient les deux mesures.
- Push a la fin de chaque phase, avec un message qui documente ce qui a ete mesure.

## Phases

| Phase | Contenu | Etat |
|---|---|---|
| A | Campagne principale S1-S4 : 12 machines d'evaluation x 4 scenarios x 4 strategies x 5 graines = **960 rejeux** | **terminee** le 28/10 a 18h13 (960/960) |
| B | Derive naturelle : scenario `none`, 13 machines instables de mise au point, 4 strategies -> fausses confirmations du §7, perimetre etendu aux 28 machines | **terminee** le 28/10 a 18h28 (112 rejeux) |
| C | Tests statistiques H1 / H2 / H3 (Wilcoxon apparie, moyenne par machine d'abord — OBS06) | **terminee** le 28/10 a 18h40 (H2 verifiee, H1 non, H3 avec un cout) |
| D | Figures 300 dpi et tableaux du chapitre 6 | **terminee** le 28/10 a 19h05 (6 figures + 5 tableaux) |
| E | D31+ dans `decisions.md`, entree M3 de `journal.md`, suppression de ce fichier | **en cours** |

## Resultats de la phase A (lecture brute, avant les tests statistiques)

960 rejeux, 12 machines d'evaluation, 4 scenarios, 5 graines. Moyennes par machine
d'abord (OBS06), puis moyenne sur les 12 machines, tous scenarios confondus :

| Metrique | B0 (fige) | B1 (periodique) | P (proposee) | A1 (sans validation) |
|---|---|---|---|---|
| F1 par fenetre | 0,265 | 0,252 | **0,265** | 0,277 |
| F1 apres derive | 0,306 | 0,324 | 0,309 | 0,326 |
| Rappel par evenement | 0,773 | 0,708 | 0,748 | 0,742 |
| Fausses alertes / h | 3,212 | 1,309 | 2,641 | 2,066 |
| Reentrainements | 0 | 16,25 | **1,375** | 1,542 |
| Secondes CPU de reentrainement | 0 | 3,941 | **0,333** | 0,375 |
| Promotions degradantes | 0 | 0,517 | **0,121** | 0,158 |

Trois choses a ne pas enjoliver au chapitre 6 :

1. **P egale B0 sur le F1 global** (0,265 contre 0,265) et ne le depasse que de peu
   apres la derive (0,309 contre 0,306). L'ecart attendu pour H1 est donc faible ; les
   tests de la phase C diront s'il est detectable. Sur le perimetre de mise au point,
   P etait *derriere* B1 (0,271 contre 0,302) ; sur le perimetre d'evaluation il passe
   devant (0,265 contre 0,252). Les parametres etaient figes avant (D29, D30) : cette
   inversion est un resultat, pas un reglage.
2. **Le gain de P sur B1 est un gain de cout, pas de qualite** : 12 fois moins de
   reentrainements et 12 fois moins de secondes CPU, pour un F1 comparable. C'est
   exactement la formulation de la question du memoire (« a moindre cout »), mais il
   faut le dire ainsi et non pretendre a une meilleure detection.
3. **Les delais sont enormes** : delai de confirmation de derive de 4091 minutes en
   moyenne et temps de recuperation de 2413 minutes. K = 960 fenetres valent 16 heures :
   le moniteur est deliberement lent, par construction (D19 : K = 30 confirmait une
   derive sur 13 des 15 machines stables). A commenter comme un arbitrage
   specificite / reactivite, chiffres a l'appui, pas comme un defaut cache.

Le detail par scenario est dans `results/summary.csv` ; les promotions degradantes
sont recalculables hors ligne a n'importe quel delta avec `mdrift promotions`.

## Resultats de la phase B (derive naturelle, sans injection)

112 rejeux, scenario `none`, les **28 machines** du SMD, 4 strategies, graine 42
(`results/summary_m3_derive_naturelle.csv`). Le perimetre a ete etendu aux 12
machines d'evaluation apres coup, et il faut dire pourquoi : la metrique du §7
« fausses confirmations de derive » se compare a une reference de **15 machines
stables** (D19), or 3 seulement sont du cote mise au point, les 12 autres etant
precisement les machines d'evaluation. Mesurer dessus ne viole pas D17 : une
confirmation de derive ne regarde aucune etiquette, et **aucun parametre n'a bouge
ni ne bougera** apres cette mesure.

### Fausses confirmations de derive, strategie P

| Perimetre | Machines | Confirmations | Machines concernees |
|---|---|---|---|
| Machines **stables** | 15 | **0** | aucune |
| Machines **instables** | 13 | 9 | 7 (1-6, 1-8, 3-1, 3-5, 3-8, 3-9, 3-11) |

Reference a battre : **13 fausses confirmations sur 15 machines stables** avec
K = 30 (D19). Avec K = 960, le compte tombe a **0 sur 15**, sans rendre le moniteur
muet pour autant : il confirme sur 7 des 13 machines dont le criblage avait montre
qu'elles derivent reellement entre leur moitie train et leur moitie test (OBS03).
C'est le resultat qui rend l'etape de confirmation evaluable : un moniteur qui
confirmerait tout le temps aurait un excellent delai de detection et ne vaudrait
rien.

### Ce que les 15 machines stables montrent de la validation

Sur une machine stable, la bonne conduite est de **ne rien faire**. C'est mesurable :

| Strategie | F1 | Fausses alertes / h | Promotions (28 machines) | Rejets | s CPU | Promotions degradantes |
|---|---|---|---|---|---|---|
| B0 fige | 0,290 | 5,08 | 0 | 0 | 0 | 0 |
| B1 periodique | 0,285 | 1,23 | 475 | 0 | 137,7 | 12 |
| P proposee | 0,299 | 3,39 | 6 | 3 | 2,1 | **0** |
| A1 sans validation | 0,296 | 2,96 | 15 | 0 | 3,4 | 3 |

Restreint aux 15 machines stables, P, A1 et B0 ont **exactement** le meme F1 (0,333)
et les memes fausses alertes (0,579/h) : P n'y reentraine jamais, donc n'y change
rien. B1 y reentraine 17 fois par machine et y perd du F1 (0,295) tout en alertant
davantage a tort (0,871/h). Le cout de la discipline periodique sur une
infrastructure qui ne derive pas est donc negatif, pas neutre.

Sur les 13 machines instables, P ameliore le F1 de B0 (0,259 contre 0,241) et
reduit ses fausses alertes (6,64/h contre 10,28/h).

**H3 sur ce perimetre** : la validation a transforme 3 promotions degradantes (A1)
en **0** (P), en promouvant 6 fois au lieu de 15 et en rejetant 3 fois. C'est la
lecture la plus nette obtenue jusqu'ici du garde-fou de l'etape 4 — a mettre en
regard de la lecture plus nuancee de M2 (D30), ou P reduisait le *compte* des
promotions degradantes bien plus que leur *taux*.

### Defaut corrige

`mdrift campaign --scenarios none` echouait **apres** avoir ecrit ses 64 resultats :
le resume final agregeait `f1_after`, qui n'existe pas sans derive injectee
(`KeyError`). Les donnees n'ont jamais ete perdues, mais un code de sortie 1 sur
une campagne reussie est exactement ce qui fait relancer un calcul de 10 minutes
pour rien. Le resume n'agrege plus que les colonnes presentes.

## Commandes de reprise

Phase A — reprend la ou elle s'est arretee, les `run_id` deja dans
`results/summary.csv` sont sautes :

```bash
uv run mdrift campaign --dataset-config datasets/smd_eval.yaml \
  --strategies static,periodic,proposed,ablation \
  --scenarios s1_soudaine,s2_incrementale,s3_graduelle,s4_recurrente \
  --seeds 42,43,44,45,46
```

Avancement : `python -c "import pandas as pd; t=pd.read_csv('results/summary.csv');
print((t.scenario!='none').sum(), '/ 960')"`.

Phase B :

```bash
uv run mdrift campaign --dataset-config datasets/smd_dev.yaml \
  --strategies static,periodic,proposed,ablation --scenarios none --seeds 42 \
  --summary-path results/summary_m3_derive_naturelle.csv
```

(perimetre de mise au point assume : les fausses confirmations se comptent sur les
machines **instables**, qui sont toutes du cote mise au point — D29, limite (i).)

## Resultats de la phase C (tests statistiques H1 / H2 / H3)

`mdrift hypotheses` -> `results/hypotheses.csv`. Wilcoxon apparie bilateral,
**moyenne par machine d'abord** (OBS06 : les 5 graines d'une machine sont 5 tirages
de *t*<sub>d</sub> sur le meme flux, pas 5 observations independantes), correction de
Holm **a l'interieur de chaque famille** d'hypothese, abstention declaree
(`indecidable`) sous 5 ecarts non nuls. 12 paires machine x scenario par scenario,
**48 sur la ligne `tous`** — c'est cette ligne qui est rapportee ici.

Les tableaux donnent **mediane et moyenne**, parce que les deux ne disent pas
toujours la meme chose : sur `f1_after` contre B1, la mediane met P devant
(0,312 contre 0,298) et la moyenne le met derriere (0,309 contre 0,324). Le test,
lui, porte sur les differences appariees et conclut a l'absence de difference
detectable — c'est la seule lecture qui vaut. Citer une seule des deux statistiques
dans le memoire laisserait croire a un ecart qui n'est pas significatif.
La colonne `+/-` compte les paires ou P est au-dessus / en dessous, ex aequo exclus.

### H1 — P contre B0 fige

| Metrique | med. P | med. B0 | moy. P | moy. B0 | +/- | p Holm | Verdict |
|---|---|---|---|---|---|---|---|
| Fausses alertes / h | 2,277 | 2,968 | 2,641 | 3,212 | 7/29 | **0,0027** | conforme |
| F1 | 0,258 | 0,268 | 0,265 | 0,265 | 20/16 | 1,0000 | pas de difference detectable |
| F1 apres derive | 0,312 | 0,325 | 0,309 | 0,306 | 20/16 | 1,0000 | pas de difference detectable |
| Rappel par evenement | 0,800 | 0,800 | 0,748 | 0,774 | 2/16 | **0,0027** | **contraire a l'attendu** |
| Temps de recuperation (min) | 2592,5 | 2592,5 | 2413,0 | 2561,2 | 4/5 | 1,0000 | pas de difference detectable |

Un seul des cinq criteres de H1 est verifie, et un autre l'est **a l'envers** :
P detecte **moins** d'evenements que le modele fige (rappel par evenement moyen
0,748 contre 0,774, soit 2,5 points perdus ; 30 des 48 paires sont a egalite, mais
sur les 18 autres P perd 16 fois). Le gain est ailleurs : **18 % de fausses alertes
en moins** en moyenne (2,641 contre 3,212 par heure), significatif apres Holm.

Autrement dit, sur le SMD avec derive injectee, le reentrainement declenche par
derive confirmee **echange du rappel contre de la precision** plutot qu'il
n'ameliore la detection. **H1 doit etre rapportee comme non verifiee au sens
large**, et ce resultat ne doit pas etre embelli : c'est la limite principale a
porter au chapitre 6 (et au 7.3).

### H2 — P contre B1 periodique

| Metrique | med. P | med. B1 | moy. P | moy. B1 | +/- | p Holm | Verdict |
|---|---|---|---|---|---|---|---|
| Nombre de reentrainements | 1,40 | 16,00 | 1,38 | 16,25 | 0/48 | **0,0000** | conforme |
| Secondes CPU de reentrainement | 0,330 | 3,832 | 0,333 | 3,941 | 0/48 | **0,0000** | conforme |
| F1 | 0,258 | 0,223 | 0,265 | 0,252 | 24/24 | 0,8843 | pas de difference detectable |
| F1 apres derive | 0,312 | 0,298 | 0,309 | 0,324 | 23/25 | 0,8843 | pas de difference detectable |
| Fausses alertes / h | 2,277 | 1,300 | 2,641 | 1,309 | 42/6 | **0,0000** | difference detectee (superieur) |

**H2 est verifiee sur les deux criteres de cout, sans aucune exception** : sur les
48 paires, P reentraine moins que B1 **48 fois sur 48**, soit un facteur **12** en
nombre de reentrainements comme en secondes CPU, pour un F1 statistiquement
indiscernable. C'est le resultat central du memoire, et c'est le seul qui soit a
la fois significatif, unanime et conforme a ce qui etait attendu.

Le revers doit etre ecrit : B1 produit **deux fois moins** de fausses alertes que
P (1,309 contre 2,641 par heure en moyenne, 42 paires sur 48, tres significatif).
Reentrainer toutes les *T* minutes recalibre le seuil en continu sur des donnees
recentes, ce qui resserre mecaniquement le taux d'alertes — au prix de 12 fois plus
de calcul. Conclusion honnete : **P domine B1 sur le cout, pas sur toutes les
dimensions de la qualite.**

### H3 — P contre A1 (meme strategie, sans l'etape de validation)

| Metrique | med. P | med. A1 | moy. P | moy. A1 | +/- | p Holm | Verdict |
|---|---|---|---|---|---|---|---|
| Promotions degradantes | 0,000 | 0,000 | 0,121 | 0,158 | 0/6 | **0,0235** | conforme |
| Promotions | 1,000 | 1,500 | 1,150 | 1,542 | 0/23 | **0,0001** | conforme |
| F1 | 0,258 | 0,260 | 0,265 | 0,277 | 6/17 | **0,0124** | difference detectee (inferieur) |
| F1 apres derive | 0,312 | 0,313 | 0,309 | 0,326 | 6/17 | **0,0047** | difference detectee (inferieur) |

Le garde-fou fait ce qu'on lui demande : sur les 6 paires ou les deux strategies
different sur les promotions degradantes, **P en a moins dans les 6 cas** et
jamais plus, en rejetant un quart des promotions (1,15 contre 1,54 par rejeu en
moyenne). Les comptes sont faibles en valeur absolue (0,12 promotion degradante
par rejeu) : 42 des 48 paires sont a egalite a zero, et c'est precisement pour cela
que la direction ne pouvait pas etre lue sur une mediane (voir le defaut corrige
ci-dessous).

Et il le paie : P perd **1,2 point de F1** et **1,6 point de F1 apres derive**
contre A1 (significatif apres Holm, 17 paires sur 23 non nulles). C'est le
compromis a enoncer tel quel : **la validation n'est pas gratuite**, elle echange
un peu de performance moyenne contre la garantie de ne pas mettre en service un
modele pire que le precedent.

A mettre en regard de la phase B : sur les 15 machines stables, le meme garde-fou
transformait 3 promotions degradantes en 0 **sans aucun cout en F1** (P = A1 = B0
au chiffre pres, puisque P n'y reentraine jamais). L'arbitrage depend donc du
regime — cout nul quand il n'y a rien a faire, cout reel sous derive injectee — et
**les deux lectures doivent figurer** au chapitre 6.

### Defaut corrige — la direction d'un test de Wilcoxon

La premiere execution lisait le sens de l'ecart sur la **mediane des differences
appariees**. Des que plus de la moitie des machines sont a egalite — le cas normal
pour `n_promotions_degradantes`, nulle sur 42 paires sur 48 —, cette mediane vaut 0
alors que Wilcoxon, qui **ignore les ex aequo**, mesure une direction nette. H3
s'affichait donc `contraire a l'attendu` (p Holm 0,0235) alors que P a **moins** de
promotions degradantes que A1 : le tableau annoncait l'inverse de ce que le test
avait mesure, sur l'hypothese qui porte la contribution du memoire.

La direction est maintenant lue sur le signe de la **somme des rangs signes**,
c'est-a-dire la statistique du test elle-meme, et le tableau affiche en clair le
compte `+/-` des ecarts non nuls pour qu'elle soit verifiable a l'oeil. Un test de
non-regression fixe ce comportement (`tests/test_hypotheses.py`) : 20 machines dont
9 seulement a ecart non nul — il faut au moins 6 ecarts non nuls pour que Wilcoxon
puisse descendre sous p = 0,05, et en meme temps une majorite d'ex aequo pour que
la mediane soit nulle ; avec 12 machines les deux conditions sont incompatibles.

## Resultats de la phase D (figures et tableaux du chapitre 6)

Une commande, `mdrift figures-m3`, produit **six figures a 300 dpi** dans
`docs/figures/` et **un fichier de tableaux** `docs/tableaux_chapitre6.md`. Tout
est relu depuis `results/` : aucune figure n'est produite par un calcul refait
pour l'occasion, aucun nombre n'est recopie a la main. Le fichier de tableaux
porte en tete l'avertissement qu'il est engendre et ne doit pas etre edite.

| Figure | Ce qu'elle montre |
|---|---|
| `f1_glissant_machine-2-1_s1_soudaine_seed42.png` | Les 4 strategies sur un meme flux, instant de derive et promotions marques |
| `f1_glissant_machine-2-2_s4_recurrente_seed42.png` | Idem sur le scenario recurrent |
| `cout_reentrainement.png` | Reentrainements et secondes CPU, par scenario |
| `f1_avant_apres.png` | F1 de part et d'autre de l'instant de derive |
| `qualite_contre_cout.png` | F1 apres derive contre cout, axe des couts en symlog |
| `confirmations_contre_k.png` | Effet de K sur la confirmation, au point de fonctionnement fige |

La machine du F1 glissant n'est pas choisie a la main : `--machine mediane`
retient celle dont le F1 apres derive de P est **le plus proche de la mediane**
des machines. C'est reproductible, et surtout ce n'est pas la plus flatteuse.

### Trois defauts corriges dans les figures

**La courbe de la strategie proposee etait invisible.** Sur un meme flux, P et A1
tracent la meme courbe partout ou la validation n'a rien rejete, et P colle a B0
partout ou elle n'a pas reentraine. Avec quatre traits identiques, la courbe rouge
de P disparaissait entierement sous l'orange de A1 : la figure centrale du
chapitre 6 ne montrait pas la contribution du memoire. Les traits sont desormais
empiles par largeur decroissante (B0 en bande large et pale, P au milieu, A1 en
pointilles au-dessus), si bien que deux courbes confondues restent lisibles — un
trait fin dans une bande large, ou des tirets sur un trait plein. Une note sous
l'axe l'explique au lecteur.

**Les legendes masquaient les barres les plus hautes** de `cout_reentrainement`
et de `f1_avant_apres` — dont la barre B1 de S1, que le chapitre commente. Les
legendes sont passees sous les panneaux.

**Une colonne mal nommee.** Le tableau de la derive naturelle intitulait
« fausses confirmations » la meme colonne pour les machines stables et pour les
machines instables. Sur une machine instable la confirmation n'est pas fausse :
le criblage a montre que ces machines derivent reellement entre leur moitie train
et leur moitie test (OBS03), et confirmer y est le comportement attendu. La
colonne s'appelle maintenant « confirmations de derive » et une note dit que les
deux lignes ne se lisent pas dans le meme sens.

### Ce que la courbe de K montre, et qui n'etait pas evident

| K | Derives injectees detectees | Confirmees avant l'instant de derive | Machines stables confirmees a tort |
|---|---|---|---|
| 240 | 56,2 % | 28 / 64 | 3 / 3 |
| 600 | 70,3 % | 16 / 64 | 1 / 3 |
| **960 (retenu)** | **79,7 %** | **4 / 64** | **0 / 3** |
| 1440 | 73,4 % | 4 / 64 | 0 / 3 |
| 2400 | 70,3 % | 4 / 64 | 0 / 3 |

Le taux de detection **monte puis redescend** avec K, ce qui est contre-intuitif
tant qu'on ne voit pas la troisieme colonne. La confirmation etant latchee — une
au plus par flux —, un K trop court confirme *avant* l'instant de derive, sur une
simple fluctuation, et verrouille la detection de la vraie derive qui suit : 28
injections perdues ainsi a K = 240, contre 4 a K = 960. La non-monotonie est
**structurelle, pas statistique**, et la figure porte les trois courbes pour que
le chapitre puisse le dire sans arriere-pensee.

Reserve a ecrire : la colonne « machines stables » de cette figure ne porte que
**3 machines**, les seules stables du perimetre de mise au point, donc un point y
vaut 33 points de pourcentage. C'est la mesure de la phase B, sur **15** machines
stables, qui fait foi pour la metrique de specificite du §7 ; cette courbe-ci ne
sert qu'a justifier le choix de K.

### Tableaux produits

1. Synthese par strategie, quatre scenarios confondus, **13 metriques du §7**.
2. F1 apres la derive, par scenario.
3. Reentrainements declenches, par scenario.
4. Tests d'hypotheses, reprise directe de `results/hypotheses.csv`.
5. Derive naturelle, machines stables contre machines instables.

Toutes les valeurs sont des **moyennes sur les machines ± erreur standard
inter-machines**, apres moyenne des graines machine par machine (OBS06). Une
metrique qui n'existe pas pour une strategie s'affiche `—` et jamais `0` : B0 n'a
pas de delai de detection de derive parce qu'il ne surveille rien, et ecrire « 0 »
ferait croire a une detection instantanee, soit l'inverse exact de ce qu'il fait.
Un test fige ce comportement.

## Dette technique relevee pendant M3

- `src/evaluation/replay.py` — `FutureWarning` de pandas sur
  `pd.concat([existing, new])` quand une colonne est entierement vide.
  **Corrige** a la phase B, avec un test de non-regression qui transforme ce
  `FutureWarning` en erreur (`tests/test_replay.py`).
- `mdrift campaign --scenarios none` — `KeyError` sur `f1_after` dans le resume
  final, apres ecriture des resultats. **Corrige** a la phase B.
