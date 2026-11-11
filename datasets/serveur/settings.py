"""Reglages de la source `serveur` (livrable M4, phase D).

Un seul objet lit `configs/datasets/serveur.yaml`, et il est partage par
l'enregistreur et le chargeur de session. C'est volontaire : les deux doivent
s'accorder sur le pas de la grille, sur l'ordre des metriques et sur les
tolerances de trous, sinon une session enregistree ne se recharge pas comme elle
a ete ecrite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

# Les expressions du YAML portent `[RATE]` la ou PromQL attend une fenetre de
# `rate(...)`. La valeur vient de la configuration et non de l'expression : une
# fenetre differente change les valeurs mesurees, et il faut qu'elle soit
# ecrite une seule fois, a cote du pas de la grille.
MARQUE_RATE = "[RATE]"


@dataclass(frozen=True)
class Metrique:
    """Une colonne de la session."""

    nom: str
    expr: str
    obligatoire: bool

    def expression(self, rate_interval: str) -> str:
        return self.expr.replace(MARQUE_RATE, f"[{rate_interval}]")


@dataclass(frozen=True)
class ReglagesServeur:
    """Tout ce que la source `serveur` lit dans sa configuration."""

    url: str
    pas_s: int
    rate_interval: str
    cibles_requises: tuple[str, ...]
    metriques: tuple[Metrique, ...]
    max_trou_consecutif: int
    max_part_interpolee: float
    part_entrainement: float
    refuser_anomalie_dans_entrainement: bool
    bornes_incluses: bool
    signaler_anomalies_sans_point: bool
    dossier_sessions: Path
    dossier_etiquettes: Path

    @property
    def noms(self) -> list[str]:
        return [m.nom for m in self.metriques]

    @classmethod
    def depuis_config(cls, config: dict[str, Any], racine: Path = PROJECT_ROOT) -> ReglagesServeur:
        prom = dict(config["prometheus"])
        metriques = tuple(
            Metrique(
                nom=str(m["nom"]),
                expr=str(m["expr"]),
                obligatoire=bool(m.get("obligatoire", False)),
            )
            for m in config["metriques"]
        )
        if not metriques:
            raise ValueError("Aucune metrique declaree : une session sans colonne n'existe pas.")
        doublons = sorted({m.nom for m in metriques if [x.nom for x in metriques].count(m.nom) > 1})
        if doublons:
            raise ValueError(
                f"Metriques declarees deux fois : {doublons}. Les noms deviennent des "
                "colonnes, donc ils doivent etre uniques."
            )

        trous = dict(config.get("trous", {}))
        decoupage = dict(config.get("decoupage", {}))
        etiquetage = dict(config.get("etiquetage", {}))
        chemins = dict(config.get("chemins", {}))

        part = float(decoupage.get("part_entrainement", 0.4))
        if not 0.0 < part < 1.0:
            raise ValueError(
                f"part_entrainement vaut {part} : elle doit etre strictement entre 0 et 1, "
                "sinon il ne reste rien a entrainer ou rien a evaluer."
            )

        return cls(
            # La variable d'environnement a la priorite : l'URL depend du port
            # publie, qui depend de la machine (phase B).
            url=os.environ.get("PROMETHEUS_URL") or str(prom["url"]),
            pas_s=int(prom.get("step_seconds", 60)),
            rate_interval=str(prom.get("rate_interval", "1m")),
            cibles_requises=tuple(str(c) for c in prom.get("cibles_requises", [])),
            metriques=metriques,
            max_trou_consecutif=int(trous.get("max_points_consecutifs", 3)),
            max_part_interpolee=float(trous.get("max_part_interpolee", 0.02)),
            part_entrainement=part,
            refuser_anomalie_dans_entrainement=bool(
                decoupage.get("refuser_anomalie_dans_entrainement", True)
            ),
            bornes_incluses=bool(etiquetage.get("bornes_incluses", True)),
            signaler_anomalies_sans_point=bool(
                etiquetage.get("signaler_anomalies_sans_point", True)
            ),
            dossier_sessions=racine / str(chemins.get("sessions", "data/sessions")),
            dossier_etiquettes=racine / str(chemins.get("etiquettes", "data/labels")),
        )
