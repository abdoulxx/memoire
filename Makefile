# Enveloppe fine autour du CLI, pour la CI GitLab (Linux).
# Sur Windows, utiliser directement : uv run mdrift <commande>
.PHONY: install lint test smd-download

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

smd-download:
	uv run mdrift smd-download
