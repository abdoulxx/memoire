# Etat de M4 — fichier de reprise

But de ce fichier : permettre de reprendre le travail **sans contexte**, apres une
interruption (limite d'usage atteinte, machine redemarree, session perdue). Mis a
jour et pousse a la fin de chaque phase. Il sera supprime quand M4 sera clos — son
contenu definitif ira dans `docs/journal.md`, comme pour M3.

Fenetre de travail autonome : jusqu'au **29/10/2026 04h00**.

## Ce que M4 doit livrer

Jalon M4 du cadrage : « Infrastructure sur mon serveur : Docker Compose (appli de
test, Prometheus, node_exporter, cAdvisor, Grafana), injecteur, enregistrement des
sessions » -> livrable memoire : **description des donnees reelles** (chapitre 5 et
section 6 sur le test final).

M4 produit du **code et des configurations**, deployables sur le serveur de Sidy.
Le deploiement lui-meme et l'enregistrement des vraies sessions lui appartiennent :
ce depot doit lui donner une commande par etape, et rien qui demande un reglage a
la main.

## Regles qui ne changent pas

- **§4.3 du cadrage, non negociable.** L'injecteur ne touche **que** des conteneurs
  qu'il a lui-meme crees, nommes par un prefixe reserve et crees avec des limites
  Docker explicites. Il refuse tout conteneur qu'il n'a pas cree. `make stop-chaos`
  arrete tout. Jamais d'injection sur un conteneur de production ou de client.
- **Aucun secret, IP, nom d'hote, nom de client ou domaine** dans le depot, les
  resultats ou les captures. Tout ce qui est propre au serveur passe par `.env`
  (non suivi) et `.env.example` (suivi, vide).
- **Le meme code qu'en rejeu.** Features, modeles, detecteurs de derive et
  strategies ne sont pas reecrits pour le mode en ligne : seule la source du flux
  change. Une session enregistree doit se rejouer avec le harnais de M2.
- Aucune valeur en dur : tout parametre va dans un YAML de `configs/`.
- Push a la fin de chaque phase, avec un message qui documente ce qui a ete fait.

## Phases

| Phase | Contenu | Etat |
|---|---|---|
| A | Appli de test FastAPI : charge simulee, endpoints de panne (latence, 5xx), metriques Prometheus, version deployee | **terminee** le 28/10 a 19h20 (13 tests, image construite et sondee) |
| B | Docker Compose complet : Prometheus + scrape, node_exporter, cAdvisor, Grafana provisionne (datasource + dashboard JSON) | **en cours** |
| C | Injecteur de pannes : conteneurs stress-ng dedies et limites, allowlist par prefixe, `make stop-chaos`, Locust, ecriture des labels horodates | a faire |
| D | Enregistrement des sessions : API Prometheus `query_range` -> Parquet, chargeur de session branche sur les features existantes | a faire |
| E | Tests, D35+ dans `decisions.md`, entree M4 de `journal.md`, suppression de ce fichier | a faire |

## Commandes de reprise

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

## Resultats de la phase A (appli de test)

`infra/app/` : `settings.py` (etat du deploiement), `chaos.py` (etat des
injections applicatives), `metrics.py` (series Prometheus), `main.py`
(application FastAPI), `Dockerfile`. 13 tests dans `tests/test_app_test.py`.

| Endpoint | Role |
|---|---|
| `GET /work?complexite=` | Trafic normal : latence de base dispersee en log-normale |
| `GET /health` | Sonde de sante, exempte d'injection |
| `GET /metrics` | Exposition Prometheus |
| `POST /chaos/latency` | Anomalie « latence » du §4.2 : +X ms pendant D secondes |
| `POST /chaos/errors` | Anomalie « rafale d'erreurs » : X % de 5xx pendant D secondes |
| `POST /chaos/reset` | Arret immediat de toutes les injections applicatives |
| `GET /chaos/state` | Injections en cours et historique, pour recouper les etiquettes |

### Trois decisions qui portent la validite des etiquettes

1. **Le plan de controle n'est jamais perturbe.** `/health`, `/metrics` et
   `/chaos/*` ne subissent ni la latence ajoutee ni les 5xx. Sans cela, injecter
   100 % de 5xx rendrait `/chaos/reset` injoignable : il faudrait tuer le
   conteneur pour revenir a la normale, et l'instant de retour au calme ne
   correspondrait plus aux etiquettes — la session serait a refaire. Un test fixe
   la propriete.
2. **Aucune injection n'est permanente** : duree obligatoire, plafonnee a 3 600 s.
   Une anomalie est un evenement court par definition du cadrage ; ce qui dure est
   une derive, et une derive s'obtient par un deploiement. Une injection oubliee
   contaminerait toutes les etiquettes suivantes.
3. **L'historique dit ce qui a ete subi, pas ce qui a ete demande** : un
   `/chaos/reset` tronque l'echeance a l'instant present. L'injecteur de la phase C
   ecrira les etiquettes, mais c'est l'appli qui sait ce qu'elle a reellement subi,
   et pouvoir comparer les deux est ce qui permet de reperer une etiquette decalee
   avant d'avoir construit un resultat dessus.

### Deux points mesures, pas supposes

- **Bornes de l'histogramme de latence** : celles de `prometheus_client` commencent
  a 5 ms et montent a 10 s. Avec une latence de base de 25 ms, le p95 tomberait
  presque toujours dans le meme bucket et une injection de latence serait
  invisible. Les bornes sont explicites et resserrees sous 100 ms.
- **Sonde de sante necessaire** : au premier essai du conteneur, les deux premieres
  requetes ont ete refusees (code 000) parce qu'uvicorn n'ecoutait pas encore. Un
  `HEALTHCHECK` a ete ajoute — `docker compose` s'en servira en phase B pour
  ordonner le demarrage. La sonde reste verte pendant une rafale de 5xx, et c'est
  voulu : un conteneur qui subit une anomalie injectee n'est pas en panne.

### Verification de l'image

`docker build -f infra/app/Dockerfile -t memoire-app:1.0.0 .` puis conteneur
lance : `/health` repond, `/work` passe de **0,10 s a 0,44 s** sous une injection
de 400 ms, `app_chaos_active{type="latence"}` vaut 1 et
`app_chaos_intensity{type="latence"}` vaut 400 pendant l'injection. Etat de sante
`healthy`. Conteneur supprime apres l'essai.
