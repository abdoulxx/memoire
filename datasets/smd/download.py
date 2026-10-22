"""Telechargement selectif du Server Machine Dataset (SMD).

Le dataset complet pese environ 500 Mo pour 28 machines. On ne recupere que les
machines listees dans `configs/datasets/smd.yaml`, et on enregistre un manifeste
(taille + empreinte SHA-256) pour que le jeu de donnees utilise soit identifiable
dans le memoire et verifiable a la reexecution.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_path

MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            temporary.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Telechargement impossible ({error.code}) : {url}") from error
    temporary.replace(destination)


def download_smd(
    config_path: str | Path = "datasets/smd.yaml",
    force: bool = False,
) -> dict[str, Any]:
    """Telecharge les machines retenues et retourne le manifeste.

    Les fichiers deja presents ne sont pas retelecharges, sauf si `force`.
    """
    config = load_config(config_path)
    base_url = config["source"]["base_url"].rstrip("/")
    raw_dir = resolve_path(config["raw_dir"])
    machines: list[str] = config["machines"]
    parts: dict[str, str] = config["parts"]

    manifest: dict[str, Any] = {
        "source": config["source"],
        "machines": machines,
        "files": {},
    }

    for part_name, part_dir in parts.items():
        for machine in machines:
            relative = f"{part_dir}/{machine}.txt"
            destination = raw_dir / relative
            if force or not destination.is_file():
                _download_file(f"{base_url}/{relative}", destination)
            manifest["files"][relative] = {
                "part": part_name,
                "machine": machine,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }

    licence = raw_dir / "LICENSE"
    if force or not licence.is_file():
        _download_file(f"{base_url}/LICENSE", licence)

    manifest_path = raw_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
