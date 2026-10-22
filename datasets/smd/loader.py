"""Chargement du Server Machine Dataset (SMD).

Format du dataset (verifie localement le 2026-10-22) :
- `train/<machine>.txt` et `test/<machine>.txt` : CSV sans en-tete, 38 colonnes de
  reels deja ramenes dans [0, 1] par les auteurs du dataset ;
- `test_label/<machine>.txt` : une valeur par ligne, 0 ou 1, alignee sur `test` ;
- `interpretation_label/<machine>.txt` : une ligne par anomalie, au format
  `debut-fin:dim1,dim2,...` (indices de points et de dimensions commencant a 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import load_config, resolve_path


@dataclass(frozen=True)
class AnomalyEvent:
    """Une anomalie etiquetee, en indices de points dans la serie `test` (bornes incluses)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class SMDMachine:
    """Une machine du SMD : moitie d'entrainement, moitie de test, etiquettes."""

    name: str
    train: np.ndarray
    test: np.ndarray
    test_label: np.ndarray
    events: list[AnomalyEvent]
    interpretation: list[tuple[int, int, list[int]]]
    sampling_interval_seconds: int

    @property
    def n_features(self) -> int:
        return self.train.shape[1]

    @property
    def anomaly_rate(self) -> float:
        """Part des points de `test` etiquetes anormaux."""
        return float(self.test_label.mean())

    def duration_hours(self, n_points: int) -> float:
        return n_points * self.sampling_interval_seconds / 3600.0


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)


def _read_labels(path: Path) -> np.ndarray:
    labels = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=1)
    unexpected = set(np.unique(labels)) - {0, 1}
    if unexpected:
        raise ValueError(f"Etiquettes non binaires dans {path} : {sorted(unexpected)}")
    return labels


def _parse_interpretation(path: Path) -> list[tuple[int, int, list[int]]]:
    """Lit `interpretation_label` et convertit en indices commencant a 0."""
    entries: list[tuple[int, int, list[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        span, _, dims = line.partition(":")
        start_text, _, end_text = span.partition("-")
        dimensions = [int(d) - 1 for d in dims.split(",") if d.strip()]
        entries.append((int(start_text) - 1, int(end_text) - 1, dimensions))
    return entries


def events_from_label(label: np.ndarray) -> list[AnomalyEvent]:
    """Decoupe un vecteur d'etiquettes en evenements : plages contigues de 1.

    C'est la definition retenue pour le rappel par evenement (memoire, §7) :
    une anomalie est une plage continue de points anormaux.
    """
    events: list[AnomalyEvent] = []
    padded = np.concatenate(([0], label.astype(np.int64), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start, end in zip(starts, ends, strict=True):
        events.append(AnomalyEvent(int(start), int(end)))
    return events


def load_machine(
    machine: str,
    config_path: str | Path = "datasets/smd.yaml",
) -> SMDMachine:
    """Charge une machine et verifie la coherence des fichiers."""
    config = load_config(config_path)
    raw_dir = resolve_path(config["raw_dir"])
    parts = config["parts"]

    paths = {part: raw_dir / parts[part] / f"{machine}.txt" for part in parts}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Fichiers SMD absents, lancer `uv run mdrift smd-download` :\n  " + "\n  ".join(missing)
        )

    train = _read_matrix(paths["train"])
    test = _read_matrix(paths["test"])
    test_label = _read_labels(paths["test_label"])
    interpretation = _parse_interpretation(paths["interpretation_label"])

    expected = int(config["expected_n_features"])
    if train.shape[1] != expected or test.shape[1] != expected:
        raise ValueError(
            f"{machine} : {train.shape[1]} colonnes en train et {test.shape[1]} en test, "
            f"{expected} attendues."
        )
    if len(test) != len(test_label):
        raise ValueError(
            f"{machine} : {len(test)} points de test pour {len(test_label)} etiquettes."
        )

    return SMDMachine(
        name=machine,
        train=train,
        test=test,
        test_label=test_label,
        events=events_from_label(test_label),
        interpretation=interpretation,
        sampling_interval_seconds=int(config["sampling_interval_seconds"]),
    )


def load_machines(
    config_path: str | Path = "datasets/smd.yaml",
) -> list[SMDMachine]:
    """Charge toutes les machines retenues dans la configuration."""
    config = load_config(config_path)
    return [load_machine(name, config_path) for name in config["machines"]]
