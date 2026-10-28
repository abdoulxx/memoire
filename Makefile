# Enveloppe fine autour du CLI, pour la CI GitLab (Linux).
# Sur Windows, utiliser directement : uv run mdrift <commande>
.PHONY: install lint test smd-download dashboard up down ps logs

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

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
