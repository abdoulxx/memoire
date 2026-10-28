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
| C | Tests statistiques H1 / H2 / H3 (Wilcoxon apparie, moyenne par machine d'abord — OBS06) | **en cours** |
| D | Figures 300 dpi et tableaux du chapitre 6 | a faire |
| E | D31+ dans `decisions.md`, entree M3 de `journal.md`, suppression de ce fichier | a faire |

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

## Dette technique relevee pendant M3

- `src/evaluation/replay.py` — `FutureWarning` de pandas sur
  `pd.concat([existing, new])` quand une colonne est entierement vide.
  **Corrige** a la phase B, avec un test de non-regression qui transforme ce
  `FutureWarning` en erreur (`tests/test_replay.py`).
- `mdrift campaign --scenarios none` — `KeyError` sur `f1_after` dans le resume
  final, apres ecriture des resultats. **Corrige** a la phase B.
