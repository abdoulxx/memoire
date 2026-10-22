"""Interface commune des detecteurs.

Convention de score, valable pour tous les detecteurs du projet :
**un score eleve signifie plus anormal**. scikit-learn utilise la convention
inverse pour l'Isolation Forest, l'implementation prend donc l'oppose de
`score_samples`. Cette convention unique evite de reinterpreter les signes dans
les detecteurs de derive, les strategies et les metriques.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    """Contrat minimal attendu par le harnais de rejeu et par la boucle en ligne."""

    threshold: float

    def fit(self, X: np.ndarray) -> Detector:
        """Entraine le detecteur et fixe son seuil d'alerte."""
        ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """Score d'anomalie, un par ligne. Score eleve = plus anormal."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alerte binaire : 1 si le score depasse le seuil."""
        ...
