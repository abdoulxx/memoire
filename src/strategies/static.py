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

from src.strategies.base import Decision, TrainingPlan, ValidationStats, Verdict, WindowContext


@dataclass
class StaticStrategy:
    """Ne reentraine jamais."""

    name: str = "static"

    def on_window(self, context: WindowContext) -> Decision:
        return Decision(retrain=False)

    def propose(self, history: np.ndarray, position: int) -> TrainingPlan:
        raise RuntimeError("La strategie B0 ne reentraine jamais : propose est inatteignable.")

    def validate(self, stats: ValidationStats) -> Verdict:
        raise RuntimeError("La strategie B0 ne reentraine jamais : validate est inatteignable.")

    def on_fit(self, training_X: np.ndarray, position: int) -> None:
        """Rien a recaler : B0 n'a ni moniteur de derive ni etat interne."""
        return None
