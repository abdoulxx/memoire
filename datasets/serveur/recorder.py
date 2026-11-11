"""Enregistrement d'une session du serveur (livrable M4, phase D).

Une session est enregistree **une fois** puis rejouee autant de fois qu'il y a
de strategies a comparer (§5 du cadrage). C'est la seule facon de comparer B0,
B1 et P sur des donnees reelles : on ne peut pas faire subir deux fois la meme
heure a un serveur, et deux heures differentes ne sont pas comparables.

Ce que ce module ecrit, et ce qu'il n'ecrit pas :

- Un Parquet, une colonne par metrique, plus la colonne `ts` des instants. Les
  valeurs sont celles que Prometheus avait, **trous compris** : un trou reste un
  `NaN` dans le fichier. Interpoler ici serait melanger la mesure et son
  interpretation, et rendrait impossible de savoir apres coup ce qui a vraiment
  ete mesure. C'est le chargeur qui interpole, sous les tolerances de la
  configuration.
- Un JSON de metadonnees a cote : plage, pas, expressions, part de points
  manquants par metrique, etat des cibles. **Aucune URL, aucun nom d'hote** n'y
  figure (§4.3), parce que ce fichier est celui qu'on recopie dans le memoire.

Les etiquettes ne sont pas ecrites ici : elles le sont par l'injecteur de la
phase C, au moment ou l'anomalie a lieu. Un enregistrement est une lecture.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.serveur.settings import ReglagesServeur
from infra.injector.labels import horodatage
from src.serving.prometheus import ClientPrometheus, grille

COLONNE_INSTANT = "ts"

# `-2h`, `-45m`, `-90s` : une plage donnee relativement a maintenant, qui est la
# facon naturelle de dire « la session vient de se terminer ».
MOTIF_RELATIF = re.compile(r"^-(\d+)([smhd])$")
SECONDES = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ErreurSession(RuntimeError):
    """La session ne peut pas etre enregistree, ou ne vaut pas la peine de l'etre."""


@dataclass(frozen=True)
class SessionEnregistree:
    """Ce qu'a produit un enregistrement."""

    run_id: str
    donnees: Path
    metadonnees: Path
    instants: np.ndarray
    parts_manquantes: dict[str, float]

    @property
    def n_points(self) -> int:
        return len(self.instants)


def instant_depuis_texte(texte: str, maintenant: float | None = None) -> float:
    """Convertit `2026-11-11T08:00:00Z`, `-2h` ou un entier en secondes Unix."""
    reference = time.time() if maintenant is None else maintenant
    texte = texte.strip()

    relatif = MOTIF_RELATIF.match(texte)
    if relatif:
        quantite, unite = relatif.groups()
        return reference - int(quantite) * SECONDES[unite]

    if re.fullmatch(r"\d+", texte):
        return float(texte)

    try:
        # `Z` est accepte par la norme mais pas par `fromisoformat` avant 3.11 ;
        # il est remplace explicitement pour que la meme chaine serve ici et
        # dans les fichiers d'etiquettes.
        return datetime.fromisoformat(texte.replace("Z", "+00:00")).timestamp()
    except ValueError as erreur:
        raise ErreurSession(
            f"Instant illisible : « {texte} ». Formes acceptees : "
            "2026-11-11T08:00:00Z, -2h, -90m, ou des secondes Unix."
        ) from erreur


def chemin_donnees(reglages: ReglagesServeur, run_id: str) -> Path:
    return reglages.dossier_sessions / f"{run_id}.parquet"


def chemin_metadonnees(reglages: ReglagesServeur, run_id: str) -> Path:
    return reglages.dossier_sessions / f"{run_id}.json"


def verifier_cibles(client: ClientPrometheus, reglages: ReglagesServeur) -> dict[str, str]:
    """Verifie que les cibles de collecte requises repondent.

    A appeler **avant** de lancer une session, pas apres : une cible tombee
    pendant l'enregistrement se voit dans les trous, mais une cible qui n'a
    jamais tourne coute la session entiere.
    """
    etats = client.cibles()
    manquantes = [c for c in reglages.cibles_requises if etats.get(c) != "up"]
    if manquantes:
        raise ErreurSession(
            f"Cibles de collecte indisponibles : {manquantes} (etat releve : {etats}). "
            "Lancer `docker compose up -d` et attendre que les cibles passent `up` "
            "avant d'enregistrer une session."
        )
    return etats


def enregistrer(
    client: ClientPrometheus,
    reglages: ReglagesServeur,
    debut: float,
    fin: float,
    run_id: str,
) -> SessionEnregistree:
    """Lit la plage sur Prometheus et ecrit le Parquet et ses metadonnees."""
    if fin <= debut:
        raise ErreurSession("La fin de la session doit etre posterieure a son debut.")

    instants = grille(debut, fin, reglages.pas_s)
    tableau: dict[str, np.ndarray] = {COLONNE_INSTANT: instants}
    parts: dict[str, float] = {}
    vides: list[str] = []

    for metrique in reglages.metriques:
        serie = client.interroger_plage(
            metrique.expression(reglages.rate_interval), debut, fin, reglages.pas_s
        )
        if not np.array_equal(serie.instants, instants):  # pragma: no cover - meme grille
            raise ErreurSession(
                f"La grille de « {metrique.nom} » ne correspond pas a celle de la session."
            )
        tableau[metrique.nom] = serie.valeurs
        parts[metrique.nom] = serie.part_manquante
        if serie.part_manquante >= 1.0:
            vides.append(f"{metrique.nom} (obligatoire)" if metrique.obligatoire else metrique.nom)

    obligatoires_vides = [
        m.nom for m in reglages.metriques if m.obligatoire and parts[m.nom] >= 1.0
    ]
    if obligatoires_vides:
        raise ErreurSession(
            f"Metriques obligatoires sans aucun point : {obligatoires_vides}. "
            "La session n'est pas enregistree. Causes deja rencontrees : cAdvisor "
            "qui n'expose pas l'etiquette `name` (voir la phase B), exportateur "
            "absent, ou plage anterieure au lancement de la pile."
        )

    reglages.dossier_sessions.mkdir(parents=True, exist_ok=True)
    donnees = pd.DataFrame(tableau)
    chemin = chemin_donnees(reglages, run_id)
    donnees.to_parquet(chemin, index=False)

    metadonnees: dict[str, Any] = {
        "run_id": run_id,
        "source": "serveur",
        "debut_ts": horodatage(float(instants[0])),
        "fin_ts": horodatage(float(instants[-1])),
        "n_points": int(len(instants)),
        "pas_s": reglages.pas_s,
        "rate_interval": reglages.rate_interval,
        "duree_heures": round(len(instants) * reglages.pas_s / 3600.0, 3),
        "enregistre_le": horodatage(time.time()),
        # Les expressions sont recopiees telles qu'elles ont ete envoyees :
        # changer une expression change les valeurs, donc une session doit
        # porter celles qui l'ont produite et non celles du YAML du jour.
        "metriques": [
            {
                "nom": m.nom,
                "expr": m.expression(reglages.rate_interval),
                "obligatoire": m.obligatoire,
                "part_manquante": round(parts[m.nom], 6),
            }
            for m in reglages.metriques
        ],
        "metriques_vides": vides,
    }
    chemin_meta = chemin_metadonnees(reglages, run_id)
    chemin_meta.write_text(json.dumps(metadonnees, indent=2, ensure_ascii=False), encoding="utf-8")

    return SessionEnregistree(
        run_id=run_id,
        donnees=chemin,
        metadonnees=chemin_meta,
        instants=instants,
        parts_manquantes=parts,
    )
