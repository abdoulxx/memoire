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
| B | Derive naturelle : scenario `none`, 13 machines instables de mise au point, 4 strategies -> fausses confirmations du §7 | **en cours** (lancee 18h15) |
| C | Tests statistiques H1 / H2 / H3 (Wilcoxon apparie, moyenne par machine d'abord — OBS06) | a faire |
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

- `src/evaluation/replay.py:536` — `FutureWarning` de pandas sur
  `pd.concat([existing, new])` quand une colonne est entierement vide. A corriger
  **apres** la campagne : modifier le module pendant qu'il tourne ne changerait rien
  au processus en cours, mais autant ne pas melanger.
