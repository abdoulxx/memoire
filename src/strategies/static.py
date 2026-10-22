"""Strategie B0 : modele fige.

Le modele est entraine une fois sur les donnees initiales, puis jamais
reentraine. C'est la **reference basse** du memoire : elle mesure ce que l'on
perd a ne rien faire quand le comportement normal de l'infrastructure change.

Son cout de reentrainement est nul par construction, ce qui en fait aussi le
point de comparaison pour la famille de metriques « cout ».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.strategies.base import Decision, WindowContext


@dataclass
class StaticStrategy:
    """Ne reentraine jamais."""

    name: str = "static"

    def on_window(self, context: WindowContext) -> Decision:
        return Decision(retrain=False)

    def training_set(self, history: np.ndarray, position: int) -> np.ndarray:
        raise RuntimeError("La strategie B0 ne reentraine jamais : training_set est inatteignable.")
