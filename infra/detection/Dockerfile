# Image du service de detection en ligne (livrable M5, phase C).
# Contexte de construction : la racine du depot.
#
#   docker build -f infra/detection/Dockerfile -t memoire-detection:0.1.0 .
#
# C'est le conteneur qui tient la cible Prometheus `detection:8001`, declaree
# des M4 dans infra/prometheus/prometheus.yml. Tant qu'il ne tourne pas, cette
# cible est `down`, ce qui est visible dans Grafana et sans effet sur le reste.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Le code est copie, pas installe en paquet : `datasets/` n'est pas dans les
# `packages` du wheel (pyproject.toml), et c'est de la que viennent les reglages
# de la source serveur. Un PYTHONPATH donne le meme chemin d'import qu'en test,
# ce qui est exactement ce qu'on veut verifier.
ENV PYTHONPATH=/srv

# Versions identiques a celles de pyproject.toml : ce qui tourne sur le serveur
# doit etre ce qui est teste en CI. Deux ecarts, tous deux voulus :
#
#  - `mlflow-skinny` et non `mlflow`. Le service est **client** du registre : il
#    n'en heberge pas. Tout ce que src/tracking/registry.py appelle passe par
#    `MlflowClient` (log_artifact, download_artifacts, alias), que skinny
#    fournit. Le paquet complet tirerait en plus un serveur web, Alembic, un
#    client Docker et une cinquantaine de dependances, sur un serveur dont le
#    §4.3 demande de surveiller l'espace disque. La version est la meme que
#    celle de l'image `mlflow` du docker-compose (2.22.4) : client et serveur
#    partagent le schema de la base, les faire diverger n'est pas une option.
#  - ni matplotlib ni pyarrow : le service ne trace aucune figure et n'ecrit
#    aucun Parquet. Ce sont les notebooks et l'enregistreur de sessions qui en
#    ont besoin, et ils ne tournent pas ici.
RUN pip install --no-cache-dir \
      "numpy>=1.26,<2.2" \
      "pandas>=2.2,<2.3" \
      "scikit-learn>=1.5,<1.7" \
      "scipy>=1.14,<1.16" \
      "pyyaml>=6.0,<7.0" \
      "typer>=0.15,<0.17" \
      "joblib>=1.4,<2.0" \
      "river>=0.21,<0.23" \
      "httpx>=0.27,<0.29" \
      "prometheus-client>=0.21,<0.24" \
      "mlflow-skinny==2.22.4" \
      "sqlalchemy>=2.0,<2.1"

WORKDIR /srv

# Seul le necessaire a la boucle : le code, les configs, et rien de data/ ni de
# results/ (exclus par .dockerignore, et le §4.3 interdit qu'une trace du
# serveur entre dans une image).
COPY src/ /srv/src/
COPY datasets/ /srv/datasets/
COPY configs/ /srv/configs/
COPY infra/__init__.py /srv/infra/__init__.py

# Le journal d'evenements est ecrit ici ; le dossier est monte depuis l'hote par
# le docker-compose, pour qu'il survive au conteneur — c'est la source que
# lisent Telegram et l'analyse hors ligne.
RUN mkdir -p /srv/results/online \
    && useradd --create-home --uid 10003 detection \
    && chown -R detection:detection /srv
USER detection

EXPOSE 8001

# Dans le conteneur, l'exporter doit ecouter sur toutes les interfaces : c'est
# Prometheus, depuis un autre conteneur, qui vient le collecter. La publication
# du port sur l'hote, elle, reste liee a ${BIND_ADDR} comme le reste de la pile.
ENV DETECTION_EXPORTER_HOST=0.0.0.0 \
    DETECTION_EXPORTER_PORT=8001

# La sonde interroge l'exporter, pas un endpoint de sante dedie : si /metrics
# repond, le processus vit et la cible Prometheus est joignable. Un `up == 1`
# ne dit pas pour autant que la detection avance — c'est
# `mdrift_last_window_timestamp_seconds` qui le dit, et c'est voulu.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/metrics', timeout=4).status == 200 else 1)"]

# Pas de duree : le service tourne jusqu'a l'arret du conteneur. La strategie,
# la source et les bornes viennent de configs/serving.yaml ; les surcharger se
# fait par `command:` dans le docker-compose, pas en reconstruisant l'image.
CMD ["python", "-m", "src.cli", "serve"]
