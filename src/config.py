"""Chargement des fichiers de configuration YAML.

Regle du projet : aucune valeur en dur dans le code, tout parametre vient d'un YAML
de `configs/` et est recopie a cote des resultats de chaque execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Charge un YAML. Un chemin relatif est resolu depuis `configs/` puis la racine."""
    candidate = Path(path)
    if not candidate.is_absolute():
        for base in (CONFIGS_DIR, PROJECT_ROOT):
            if (base / candidate).is_file():
                candidate = base / candidate
                break
    if not candidate.is_file():
        raise FileNotFoundError(f"Configuration introuvable : {path}")
    with candidate.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"La configuration {candidate} doit etre un mapping YAML.")
    return content


def resolve_path(path: str | Path) -> Path:
    """Resout un chemin de la configuration relativement a la racine du depot."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
