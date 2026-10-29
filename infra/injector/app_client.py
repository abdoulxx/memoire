"""Acces au plan de controle de l'appli de test (livrable M4, phase C).

Deux moyens d'injection sur cinq (`latence`, `erreurs`) ne passent pas par
Docker mais par les endpoints `/chaos/*` de l'appli de test : une latence
supplementaire et une part de reponses 5xx sont des anomalies applicatives, et
les produire depuis l'exterieur demanderait un proxy que rien ne justifie.

Ce client verifie sa cible avant la premiere injection, et c'est la raison
principale de son existence. Le port publie sur l'hote est une variable depuis
la phase B — sur la machine de mise au point, 8000 etait deja pris par une
application sans rapport — donc une URL perimee pointe vers un service tiers.
Une requete envoyee a un service tiers est exactement ce que le §4.3 interdit,
et un 404 arrive trop tard : la requete est deja partie.

Le client est volontairement minuscule et isole dans son module : l'injecteur ne
depend que du protocole `ClientAppli`, ce qui permet de tester tous ses
garde-fous sans aucune appli qui tourne.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

# La signature de `/health` de l'appli de test (infra/app/main.py). Un service
# tiers qui repondrait a cette URL n'a aucune raison de renvoyer ces trois cles.
SIGNATURE = ("statut", "version", "chaos_autorise")


class CibleInattendue(RuntimeError):
    """Ce qui repond a l'URL n'est pas l'appli de test du memoire."""


@dataclass
class ClientAppliHttp:
    """Le plan de controle de l'appli de test, joint en HTTP."""

    url: str
    delai_s: float = 5.0
    _verifiee: bool = field(default=False, repr=False)

    @property
    def base(self) -> str:
        return self.url.rstrip("/")

    def sante(self) -> dict[str, Any]:
        reponse = httpx.get(f"{self.base}/health", timeout=self.delai_s)
        reponse.raise_for_status()
        return dict(reponse.json())

    def verifier_cible(self) -> dict[str, Any]:
        """Refuse d'injecter si l'URL ne repond pas comme l'appli de test."""
        try:
            sante = self.sante()
        except httpx.HTTPError as erreur:
            raise CibleInattendue(
                f"{self.base} n'a pas repondu comme l'appli de test : l'appli est-elle "
                "lancee, et MEMOIRE_APP_URL pointe-t-elle sur son port publie ? "
                f"({erreur})"
            ) from erreur
        manquantes = [cle for cle in SIGNATURE if cle not in sante]
        if manquantes:
            raise CibleInattendue(
                f"{self.base} repond, mais ce n'est pas l'appli de test du memoire "
                f"(cles manquantes : {', '.join(manquantes)}). Aucune injection "
                "n'est envoyee a un service tiers (§4.3)."
            )
        if not sante.get("chaos_autorise"):
            raise CibleInattendue(
                f"{self.base} est bien l'appli de test, mais son plan de controle "
                "est desactive (APP_CHAOS_ENABLED=0)."
            )
        self._verifiee = True
        return sante

    def _poster(self, chemin: str, charge: dict[str, float] | None) -> dict[str, Any]:
        if not self._verifiee:
            self.verifier_cible()
        reponse = httpx.post(f"{self.base}{chemin}", json=charge or {}, timeout=self.delai_s)
        reponse.raise_for_status()
        return dict(reponse.json())

    def injecter(self, chemin: str, charge: dict[str, float]) -> dict[str, Any]:
        return self._poster(chemin, charge)

    def reinitialiser(self) -> dict[str, Any]:
        return self._poster("/chaos/reset", None)

    def etat(self) -> dict[str, Any]:
        reponse = httpx.get(f"{self.base}/chaos/state", timeout=self.delai_s)
        reponse.raise_for_status()
        return dict(reponse.json())

    def version(self) -> str:
        """La version declaree par l'appli, pour horodater un deploiement (§4.2, S1)."""
        return str(self.sante().get("version", "inconnue"))
