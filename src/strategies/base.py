"""Interface commune des strategies de reentrainement.

Toutes les strategies comparees dans le memoire partagent le meme modele, les
memes features, le meme entrainement initial et la meme regle de seuil. La
**seule** chose qui les distingue est la reponse a la question : *faut-il
reentrainer maintenant, et avec quelles donnees ?*

C'est cette contrainte qui rend la comparaison interpretable : un ecart de
performance entre deux strategies est attribuable a leur politique de
reentrainement, et a rien d'autre.

Strategies prevues :
- `static` (B0)   : jamais de reentrainement — reference basse ;
- `periodic` (B1) : toutes les T minutes — reference classique ;
- `proposed` (P)  : declenche par une derive confirmee, avec validation ;
- `ablation` (A1) : P sans l'etape de validation.

Seule `static` est implementee a M0 ; les autres arrivent a M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Decision:
    """Ce que la strategie decide a l'issue d'une fenetre.

    `retrain` declenche un reentrainement ; `reason` est journalisee pour que
    chaque reentrainement du memoire soit explicable.
    """

    retrain: bool = False
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowContext:
    """Ce que la strategie sait au moment de decider, et rien de plus.

    Contrainte prequentielle : la fenetre a **deja** ete scoree par le modele en
    service avant que la strategie ne soit consultee, et la strategie ne voit
    jamais les fenetres futures ni les etiquettes de verite terrain.
    """

    position: int
    end_index: int
    features: np.ndarray
    score: float
    alert: bool
    threshold: float


@runtime_checkable
class Strategy(Protocol):
    """Contrat attendu par le harnais de rejeu et par la boucle en ligne."""

    name: str

    def on_window(self, context: WindowContext) -> Decision:
        """Decide s'il faut reentrainer, au vu de la seule fenetre courante et de l'etat interne."""
        ...

    def training_set(self, history: np.ndarray, position: int) -> np.ndarray:
        """Donnees sur lesquelles reentrainer, choisies dans l'historique deja observe."""
        ...
