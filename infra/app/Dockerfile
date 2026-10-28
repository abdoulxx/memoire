# Image de l'appli de test (livrable M4, phase A).
# Contexte de construction : la racine du depot, pour que `infra.app` soit un
# paquet importable tel quel — le meme chemin d'import qu'en test.
#
#   docker build -f infra/app/Dockerfile -t memoire-app:1.0.0 .
#
# Une derive reelle S1 se produit en reconstruisant cette image avec un
# APP_VERSION et un APP_BASE_LATENCY_MS differents, puis en relancant le service.
FROM python:3.11-slim

# Pas de cache pip ni de .pyc : l'image est jetable et l'espace disque du serveur
# est une contrainte du §4.3.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Versions epinglees, et identiques a celles du groupe `serveur` de
# pyproject.toml : ce qui tourne sur le serveur doit etre ce qui est teste en CI.
RUN pip install --no-cache-dir \
      "fastapi>=0.115,<0.120" \
      "uvicorn>=0.32,<0.36" \
      "prometheus-client>=0.21,<0.24"

WORKDIR /srv
COPY infra/__init__.py /srv/infra/__init__.py
COPY infra/app/ /srv/infra/app/

# Utilisateur non privilegie : ce conteneur recoit du trafic et subit des
# injections, il n'a aucune raison d'etre root.
RUN useradd --create-home --uid 10001 appli && chown -R appli:appli /srv
USER appli

EXPOSE 8000

# Sonde de sante : `docker compose` s'en sert pour ordonner le demarrage, et le
# smoke test de la phase A a montre que c'etait necessaire — les premieres
# requetes arrivaient avant qu'uvicorn ecoute. `/health` est exempt d'injection,
# donc la sonde reste verte pendant une rafale de 5xx : c'est voulu, un conteneur
# qui subit une anomalie injectee n'est pas un conteneur en panne. L'image slim
# n'a ni curl ni wget, d'ou l'appel par Python.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# Les trois variables qui decrivent le deploiement courant (voir settings.py).
ENV APP_VERSION=1.0.0 \
    APP_BASE_LATENCY_MS=25 \
    APP_LATENCY_SPREAD=0.35 \
    APP_CHAOS_ENABLED=1

# Un seul worker : la metrique `app_in_flight_requests` et l'etat des injections
# vivent en memoire du processus. Avec plusieurs workers, `/chaos/reset` ne
# toucherait qu'un worker sur n et l'etiquette ne correspondrait plus a ce que le
# service a subi. La charge visee (quelques centaines de requetes/s via Locust)
# ne justifie pas de passer a un etat partage.
CMD ["uvicorn", "infra.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
