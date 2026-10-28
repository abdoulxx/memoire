"""Reglages de l'appli de test, lus dans l'environnement (livrable M4, phase A).

Pourquoi l'environnement et non un YAML de `configs/`, contrairement a la regle
du projet : ces trois valeurs **sont** le deploiement. Changer `APP_VERSION` et
`APP_BASE_LATENCY_MS` puis relancer le conteneur, c'est exactement la derive
reelle S1 du cadrage (« deploiement d'une nouvelle version : latence et RAM de
base differentes »). Les mettre dans un fichier suivi par git forcerait a
committer pour injecter une derive, et la trace de l'experience se melangerait a
l'historique du code. Les parametres de l'**experience**, eux, restent dans
`configs/` : ceux-ci ne decrivent que l'etat courant du service.

Aucune de ces valeurs n'est un secret, et aucune ne contient de nom d'hote : le
§4.3 du cadrage est respecte meme si le fichier `.env` fuitait.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Valeurs par defaut d'un service « sain », avant toute injection. Elles n'ont
# pas de sens physique particulier : seules leurs *variations* sont mesurees.
VERSION_DEFAUT = "1.0.0"
LATENCE_BASE_MS_DEFAUT = 25.0
DISPERSION_DEFAUT = 0.35


def _flottant(nom: str, defaut: float) -> float:
    """Lit un flottant dans l'environnement, en retombant sur le defaut si illisible.

    Un service qui refuse de demarrer parce qu'une variable est mal ecrite ferait
    perdre la session en cours sur le serveur ; mieux vaut demarrer avec le defaut
    et le rendre visible dans `/metrics` et `/chaos/state`.
    """
    brut = os.environ.get(nom)
    if brut is None or brut.strip() == "":
        return defaut
    try:
        return float(brut)
    except ValueError:
        return defaut


@dataclass(frozen=True)
class Reglages:
    """Etat declare du service, expose tel quel dans les metriques."""

    version: str
    latence_base_ms: float
    dispersion: float
    chaos_autorise: bool

    @classmethod
    def depuis_environnement(cls) -> Reglages:
        version = os.environ.get("APP_VERSION", "").strip() or VERSION_DEFAUT
        latence = max(0.0, _flottant("APP_BASE_LATENCY_MS", LATENCE_BASE_MS_DEFAUT))
        dispersion = max(0.0, _flottant("APP_LATENCY_SPREAD", DISPERSION_DEFAUT))
        # L'autorisation est explicite et par defaut **vraie** dans cette appli,
        # parce que c'est une appli de test dediee : son seul role est de subir
        # des pannes. Le drapeau existe pour pouvoir la deployer une fois sans
        # endpoints de panne (mesure du bruit de fond), pas comme garde-fou de
        # securite — le garde-fou, c'est de ne jamais deployer cette image a cote
        # d'une application de production (§4.3).
        chaos = os.environ.get("APP_CHAOS_ENABLED", "1").strip().lower()
        return cls(
            version=version,
            latence_base_ms=latence,
            dispersion=dispersion,
            chaos_autorise=chaos not in {"0", "false", "no", "non"},
        )
