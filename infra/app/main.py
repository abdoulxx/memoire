"""Appli de test FastAPI : la cible des injections (livrable M4, phase A).

Role dans l'experience : produire un flux de metriques applicatives realiste, que
Prometheus collecte a la minute, et pouvoir subir a la demande les deux anomalies
applicatives du §4.2 (latence ajoutee, rafale de 5xx) ainsi que la derive reelle
S1 (deploiement d'une version a latence de base differente).

Deux choix de conception portent toute la surete de ce service :

1. **Le plan de controle n'est jamais perturbe.** `/health`, `/metrics` et
   `/chaos/*` ne subissent ni la latence ajoutee ni les 5xx. Sans cela, injecter
   90 % de 5xx rendrait `/chaos/reset` injoignable : il faudrait tuer le
   conteneur pour arreter l'injection, et la session serait perdue.
2. **Aucune injection n'est permanente.** La duree est obligatoire et plafonnee
   (`infra/app/chaos.py`). Une anomalie est un evenement court, par definition du
   cadrage ; ce qui dure est une derive, et une derive se produit par deploiement.

Cette image ne doit jamais etre deployee a cote d'une application de production :
c'est une appli de test dediee, et c'est la seule garantie qui compte (§4.3).
"""

from __future__ import annotations

import asyncio
import math
import random
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from infra.app.chaos import Chaos
from infra.app.metrics import Metriques
from infra.app.settings import Reglages

# Endpoints exempts d'injection : le plan de controle de l'experience.
EXEMPTS = ("/health", "/metrics", "/chaos")


class DemandeLatence(BaseModel):
    supplement_ms: float = Field(..., ge=0, description="Millisecondes ajoutees a chaque reponse.")
    duree_s: float = Field(..., gt=0, description="Duree de l'injection, en secondes.")


class DemandeErreurs(BaseModel):
    taux: float = Field(..., ge=0, le=1, description="Part des reponses transformees en 5xx.")
    duree_s: float = Field(..., gt=0, description="Duree de l'injection, en secondes.")


def _est_exempt(chemin: str) -> bool:
    return any(chemin == exempt or chemin.startswith(exempt + "/") for exempt in EXEMPTS)


def creer_application(
    reglages: Reglages | None = None,
    chaos: Chaos | None = None,
    metriques: Metriques | None = None,
    alea: random.Random | None = None,
) -> FastAPI:
    """Construit l'application. Tout est injectable pour que les tests soient rapides.

    En particulier l'horloge du `Chaos` et le generateur aleatoire : une rafale de
    5xx a 30 % ne se verifie pas en tirant au hasard, et une injection de 90
    secondes ne se teste pas en attendant 90 secondes.
    """
    reglages = reglages or Reglages.depuis_environnement()
    chaos = chaos or Chaos()
    metriques = metriques or Metriques()
    alea = alea or random.Random()

    application = FastAPI(
        title="Appli de test — memoire derive",
        version=reglages.version,
        docs_url=None,
        redoc_url=None,
    )
    application.state.reglages = reglages
    application.state.chaos = chaos
    application.state.metriques = metriques
    application.state.alea = alea
    metriques.declarer_deploiement(reglages.version, reglages.latence_base_ms)

    @application.middleware("http")
    async def mesurer_et_perturber(request: Request, appeler_suivant):  # noqa: ANN001
        chemin = request.url.path
        if _est_exempt(chemin):
            return await appeler_suivant(request)

        debut = time.perf_counter()
        metriques.en_vol.inc()
        try:
            supplement_ms = chaos.latence_ajoutee_ms()
            taux = chaos.taux_erreurs()
            metriques.refleter_chaos(supplement_ms, taux)

            if supplement_ms > 0:
                await asyncio.sleep(supplement_ms / 1000.0)

            if taux > 0 and alea.random() < taux:
                reponse = Response(
                    content='{"detail":"erreur injectee"}',
                    status_code=500,
                    media_type="application/json",
                )
            else:
                reponse = await appeler_suivant(request)
        finally:
            metriques.en_vol.dec()

        duree = time.perf_counter() - debut
        # L'etiquette est le chemin de la route et non l'URL complete : une
        # etiquette par valeur de parametre ferait exploser le nombre de series
        # cote Prometheus, et la retention du serveur est une contrainte du §4.3.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", chemin)
        metriques.observer(endpoint, request.method, reponse.status_code, duree)
        return reponse

    @application.get("/health")
    async def sante() -> dict[str, object]:
        return {
            "statut": "ok",
            "version": reglages.version,
            "latence_base_ms": reglages.latence_base_ms,
            "chaos_autorise": reglages.chaos_autorise,
        }

    @application.get("/metrics")
    async def exposer_metriques() -> Response:
        metriques.refleter_chaos(chaos.latence_ajoutee_ms(), chaos.taux_erreurs())
        return Response(generate_latest(metriques.registre), media_type=CONTENT_TYPE_LATEST)

    @application.get("/work")
    async def travailler(complexite: float = 1.0) -> dict[str, float]:
        """Travail simule : une latence de base dispersee, proportionnelle a `complexite`.

        La dispersion est log-normale et non uniforme, parce que c'est la forme
        qu'ont les latences de service reelles : une queue a droite, jamais de
        valeur negative. C'est aussi ce qui rend le p95 informatif.
        """
        complexite = min(max(complexite, 0.1), 10.0)
        facteur = math.exp(alea.gauss(0.0, reglages.dispersion))
        attente_s = (reglages.latence_base_ms * complexite * facteur) / 1000.0
        await asyncio.sleep(attente_s)
        return {"complexite": complexite, "service_ms": attente_s * 1000.0}

    @application.post("/chaos/latency")
    async def injecter_latence(demande: DemandeLatence) -> dict[str, object]:
        _verifier_autorisation(reglages)
        try:
            injection = chaos.injecter_latence(demande.supplement_ms, demande.duree_s)
        except ValueError as erreur:
            raise HTTPException(status_code=400, detail=str(erreur)) from erreur
        return _vue(injection)

    @application.post("/chaos/errors")
    async def injecter_erreurs(demande: DemandeErreurs) -> dict[str, object]:
        _verifier_autorisation(reglages)
        try:
            injection = chaos.injecter_erreurs(demande.taux, demande.duree_s)
        except ValueError as erreur:
            raise HTTPException(status_code=400, detail=str(erreur)) from erreur
        return _vue(injection)

    @application.post("/chaos/reset")
    async def arreter_injections() -> dict[str, int]:
        arretees = chaos.reinitialiser()
        metriques.refleter_chaos(0.0, 0.0)
        return {"arretees": arretees}

    @application.get("/chaos/state")
    async def etat_injections() -> dict[str, object]:
        return chaos.etat()

    return application


def _verifier_autorisation(reglages: Reglages) -> None:
    if not reglages.chaos_autorise:
        raise HTTPException(
            status_code=403,
            detail="Injections desactivees sur ce deploiement (APP_CHAOS_ENABLED).",
        )


def _vue(injection) -> dict[str, object]:  # noqa: ANN001
    return {
        "type": injection.type,
        "intensite": injection.intensite,
        "debut_ts": injection.debut_ts,
        "fin_ts": injection.fin_ts,
    }


# Point d'entree pour uvicorn : `uvicorn infra.app.main:app`.
app = creer_application()
