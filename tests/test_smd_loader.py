"""Tests du chargement du SMD.

Les cas de `events_from_label` et du parsing sont calcules a la main : ils ne
dependent pas du telechargement. Les tests d'integration sont ignores si les
donnees brutes ne sont pas presentes.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.smd.loader import (
    _parse_interpretation,
    events_from_label,
    load_machines,
)
from src.config import load_config, resolve_path


def test_events_simple() -> None:
    label = np.array([0, 1, 1, 0, 0, 1, 0])
    events = events_from_label(label)
    assert [(e.start, e.end, e.length) for e in events] == [(1, 2, 2), (5, 5, 1)]


def test_events_bords() -> None:
    """Une anomalie qui commence au premier point ou finit au dernier est bien fermee."""
    assert [(e.start, e.end) for e in events_from_label(np.array([1, 1, 0]))] == [(0, 1)]
    assert [(e.start, e.end) for e in events_from_label(np.array([0, 1, 1]))] == [(1, 2)]
    assert [(e.start, e.end) for e in events_from_label(np.array([1, 1, 1]))] == [(0, 2)]


def test_events_vides() -> None:
    assert events_from_label(np.zeros(5, dtype=int)) == []


def test_parse_interpretation(tmp_path) -> None:
    """Le format `debut-fin:dims` est en base 1, converti en base 0."""
    path = tmp_path / "interpretation.txt"
    path.write_text("15849-16368:1,9,10\n16963-17517:2,3\n", encoding="utf-8")
    assert _parse_interpretation(path) == [
        (15848, 16367, [0, 8, 9]),
        (16962, 17516, [1, 2]),
    ]


def _donnees_absentes() -> bool:
    config = load_config("datasets/smd.yaml")
    raw_dir = resolve_path(config["raw_dir"])
    return not (raw_dir / config["parts"]["train"] / f"{config['machines'][0]}.txt").is_file()


@pytest.mark.skipif(_donnees_absentes(), reason="SMD non telecharge")
def test_machines_coherentes() -> None:
    """Sur les donnees reelles : dimensions, alignement des etiquettes, evenements."""
    for machine in load_machines():
        assert machine.n_features == 38
        assert len(machine.test) == len(machine.test_label)
        assert machine.train.shape[1] == machine.test.shape[1]
        assert 0.0 < machine.anomaly_rate < 0.5
        # Chaque plage d'interpretation correspond a un evenement contigu d'etiquettes.
        assert len(machine.events) == len(machine.interpretation)
