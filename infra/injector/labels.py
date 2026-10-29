"""Ecriture des etiquettes d'une session (livrable M4, phase C).

Deux fichiers, comme le demande le §4.2 du cadrage :

  data/labels/<run_id>.csv        start_ts, end_ts, type, intensite
  data/labels/<run_id>_drift.csv  ts, type, description

Les instants sont ecrits en **RFC 3339 / UTC** (`2026-10-28T20:15:00Z`) et non en
secondes depuis l'epoque, pour deux raisons. La premiere est que l'API de
Prometheus accepte ce format tel quel : la fenetre d'une anomalie peut etre
recopiee dans un `query_range` sans conversion, donc sans risque de decalage
d'une heure entre l'etiquette et les donnees. La seconde est qu'une annexe de
memoire ou une verification a l'œil se lit, alors qu'un entier de dix chiffres ne
se lit pas.

Une etiquette est ecrite **apres** l'injection, avec l'instant de fin reellement
atteint. Une injection interrompue produit donc une etiquette plus courte que
prevu, et non une etiquette fausse : c'est la meme regle qu'au plan de controle
de l'appli de test, ou l'historique dit ce qui a ete subi et non ce qui a ete
demande.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

COLONNES_ANOMALIES = ("start_ts", "end_ts", "type", "intensite")
COLONNES_DERIVES = ("ts", "type", "description")


def horodatage(instant: float) -> str:
    """Secondes depuis l'epoque -> RFC 3339 UTC a la seconde."""
    return datetime.fromtimestamp(instant, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Anomalie:
    """Une fenetre d'anomalie telle qu'elle a ete subie."""

    type: str
    intensite: float
    debut_ts: float
    fin_ts: float

    @property
    def duree_s(self) -> float:
        return self.fin_ts - self.debut_ts

    def ligne(self) -> dict[str, str]:
        return {
            "start_ts": horodatage(self.debut_ts),
            "end_ts": horodatage(self.fin_ts),
            "type": self.type,
            "intensite": f"{self.intensite:g}",
        }


@dataclass(frozen=True)
class Derive:
    """Un changement durable du comportement normal, a un instant connu."""

    type: str
    description: str
    instant_ts: float

    def ligne(self) -> dict[str, str]:
        return {
            "ts": horodatage(self.instant_ts),
            "type": self.type,
            "description": self.description,
        }


def chemin_anomalies(dossier: Path, run_id: str) -> Path:
    return dossier / f"{run_id}.csv"


def chemin_derives(dossier: Path, run_id: str) -> Path:
    return dossier / f"{run_id}_drift.csv"


def _ajouter(chemin: Path, colonnes: tuple[str, ...], ligne: dict[str, str]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    nouveau = not chemin.exists()
    # newline="" : sans cela, le module csv ecrit une ligne vide entre chaque
    # enregistrement sous Windows, et le fichier devient illisible par pandas.
    with chemin.open("a", encoding="utf-8", newline="") as flux:
        auteur = csv.DictWriter(flux, fieldnames=list(colonnes))
        if nouveau:
            auteur.writeheader()
        auteur.writerow(ligne)


def ecrire_anomalie(dossier: Path, run_id: str, anomalie: Anomalie) -> Path:
    chemin = chemin_anomalies(dossier, run_id)
    _ajouter(chemin, COLONNES_ANOMALIES, anomalie.ligne())
    return chemin


def ecrire_derive(dossier: Path, run_id: str, derive: Derive) -> Path:
    chemin = chemin_derives(dossier, run_id)
    _ajouter(chemin, COLONNES_DERIVES, derive.ligne())
    return chemin


def lire_anomalies(dossier: Path, run_id: str) -> list[dict[str, str]]:
    chemin = chemin_anomalies(dossier, run_id)
    if not chemin.exists():
        return []
    with chemin.open("r", encoding="utf-8", newline="") as flux:
        return list(csv.DictReader(flux))


def lire_derives(dossier: Path, run_id: str) -> list[dict[str, str]]:
    chemin = chemin_derives(dossier, run_id)
    if not chemin.exists():
        return []
    with chemin.open("r", encoding="utf-8", newline="") as flux:
        return list(csv.DictReader(flux))
