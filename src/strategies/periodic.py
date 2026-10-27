"""Strategie B1 : reentrainement periodique.

Toutes les T minutes, le modele est reentraine sur une fenetre glissante des W
dernieres minutes, et le resultat est mis en service **sans condition**. C'est
la reference classique du memoire : la pratique courante quand on sait qu'un
modele se perime, mais qu'on ne surveille pas la derive.

Elle est volontairement aveugle, et c'est tout son interet comme point de
comparaison. Deux consequences a mesurer, et non a supposer :

- elle paye le cout de T reentrainements meme quand rien ne change ;
- elle peut mettre en service un modele entraine sur une periode anormale,
  puisqu'elle ne valide rien.

B1 ne reserve donc aucune fenetre V (`reserved_X` est vide) et son verdict est
toujours la promotion. C'est une propriete de la strategie, pas un raccourci
d'implementation : A1 (P sans validation) existe pour isoler l'effet de la
validation a politique de declenchement egale, et B1 pour isoler l'effet de la
politique de declenchement.
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import (
    Decision,
    TrainingPlan,
    ValidationStats,
    Verdict,
    WindowContext,
)


class PeriodicStrategy:
    """Reentraine toutes les `period_windows` fenetres, sur les `train_windows` dernieres."""

    name = "periodic"

    def __init__(self, period_windows: int, train_windows: int) -> None:
        if period_windows < 1:
            raise ValueError("period_windows doit valoir au moins 1.")
        if train_windows < 1:
            raise ValueError("train_windows doit valoir au moins 1.")
        self.period_windows = int(period_windows)
        self.train_windows = int(train_windows)
        self._last_fit: int = 0

    def on_window(self, context: WindowContext) -> Decision:
        """Declenche quand T fenetres se sont ecoulees depuis le dernier entrainement.

        Le compteur part de l'entrainement initial : la premiere echeance tombe
        T fenetres apres le debut du flux, et non a la fenetre 0.
        """
        elapsed = context.position - self._last_fit
        if elapsed < self.period_windows:
            return Decision(retrain=False)
        return Decision(
            retrain=True,
            reason="periode_ecoulee",
            details={"elapsed_windows": elapsed, "period_windows": self.period_windows},
        )

    def propose(self, history: np.ndarray, position: int) -> TrainingPlan:
        """Fenetre glissante des W dernieres fenetres observees. Aucune fenetre reservee."""
        start = max(0, len(history) - self.train_windows)
        train = history[start:]
        return TrainingPlan(
            train_X=train,
            reserved_X=history[:0],
            details={"train_start": int(start), "train_stop": int(len(history))},
        )

    def validate(self, stats: ValidationStats) -> Verdict:
        """Promotion inconditionnelle : B1 ne valide pas, par definition."""
        return Verdict(promote=True, reason="b1_promotion_inconditionnelle")

    def on_fit(self, training_X: np.ndarray, position: int) -> None:
        """Remet le compteur de periode a zero."""
        self._last_fit = int(position)
