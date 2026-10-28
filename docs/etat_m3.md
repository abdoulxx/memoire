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
| A | Campagne principale S1-S4 : 12 machines d'evaluation x 4 scenarios x 4 strategies x 5 graines = **960 rejeux** | **en cours** (lancee 16h55) |
| B | Derive naturelle : scenario `none`, 13 machines instables de mise au point, 4 strategies -> fausses confirmations du §7 | a faire |
| C | Tests statistiques H1 / H2 / H3 (Wilcoxon apparie, moyenne par machine d'abord — OBS06) | a faire |
| D | Figures 300 dpi et tableaux du chapitre 6 | a faire |
| E | D31+ dans `decisions.md`, entree M3 de `journal.md`, suppression de ce fichier | a faire |

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
