# Enveloppe fine autour du CLI, pour la CI GitLab (Linux).
# Sur Windows, utiliser directement : uv run mdrift <commande>
.PHONY: install lint test build smd-download dashboard up down ps logs chaos-list stop-chaos stop-chaos-force

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

# Le troisieme controle exige par le jalon M7, apres le lint et les tests.
# Construire le wheel ne suffit pas : ce qui casse en pratique, c'est le
# **contenu** du paquet. `pyproject.toml` declare `packages = ["src", "infra"]`
# parce que le CLI importe `infra.injector` ; un paquet qui n'embarquerait que
# `src` s'installerait sans erreur et echouerait a la premiere commande `chaos*`.
# Le wheel est donc installe dans un environnement **neuf**, hors du depot, et on
# verifie que le point d'entree `mdrift` repond et que les deux paquets
# s'importent. Teste depuis le dossier d'installation pour qu'aucun import ne
# puisse passer par les sources du depot.
build:
	rm -rf dist build-verif
	uv build
	uv venv --python 3.11 build-verif
	VIRTUAL_ENV=build-verif uv pip install --quiet dist/*.whl
	cd build-verif && ./bin/mdrift --help > /dev/null
	cd build-verif && ./bin/python -c "import src.cli, infra.injector; print('paquet importable')"

smd-download:
	uv run mdrift smd-download

# --- Pile de collecte du jalon M4 (docker-compose.yml) ---
# Prerequis : cp .env.example .env puis renseigner GRAFANA_ADMIN_PASSWORD.
# Grafana refuse de demarrer sans, et c'est voulu.

# Reengendre le JSON du tableau de bord depuis son generateur, puis recharge
# Grafana. Ne jamais modifier le JSON a la main : un test verifie qu'il est bien
# la sortie du generateur.
dashboard:
	python infra/grafana/build_dashboard.py
	docker compose restart grafana

up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=50

# --- Injection d'anomalies du jalon M4 (§4.2) et arret d'urgence (§4.3) ---

chaos-list:
	uv run mdrift chaos-list

# L'arret d'urgence exige par le §4.3 du cadrage. Il arrete tous les conteneurs
# d'injection et remet a zero le plan de controle de l'appli de test (la latence
# et les erreurs injectees ne vivent pas dans un conteneur).
stop-chaos:
	uv run mdrift chaos-stop

# Secours : le meme arret sans passer par Python, au cas ou l'environnement
# serait casse au mauvais moment. Les DEUX marques du §4.3 sont exigees ici
# aussi — l'etiquette Docker et le prefixe de nom — pour qu'une commande tapee
# en urgence ne puisse pas viser un conteneur de production.
stop-chaos-force:
	docker ps -q --filter label=memoire.injection=1 --filter name=memoire-chaos- | xargs -r docker kill
