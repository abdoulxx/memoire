"""Interface commune des strategies de reentrainement.

Toutes les strategies comparees dans le memoire partagent le meme modele, les
memes features, le meme entrainement initial et la meme regle de seuil. La
**seule** chose qui les distingue est la reponse a la question : *faut-il
reentrainer maintenant, avec quelles donnees, et le resultat merite-t-il d'etre
mis en service ?*

C'est cette contrainte qui rend la comparaison interpretable : un ecart de
performance entre deux strategies est attribuable a leur politique de
reentrainement, et a rien d'autre.

Strategies comparees :
- `static` (B0)   : jamais de reentrainement — reference basse ;
- `periodic` (B1) : toutes les T minutes — reference classique ;
- `proposed` (P)  : declenche par une derive confirmee, avec validation ;
- `ablation` (A1) : P sans l'etape de validation (meme code, validation
  desactivee par configuration).

Le cycle d'un reentrainement se joue en quatre temps, et le decoupage n'est pas
cosmetique : il place chaque responsabilite du cote qui peut l'assumer sans
tricher.

    on_window(ctx)              -> Decision(retrain)
    propose(history, position)  -> TrainingPlan(train_X, reserved_X)
       [le harnais entraine le challenger et le score sur reserved_X]
    validate(stats)             -> Verdict(promote, reason)
    on_fit(training_X)          # le modele en service a change

La strategie decide et choisit ses donnees ; le **harnais** entraine, score et
mesure. La strategie ne voit donc du challenger que des statistiques sans
etiquette, ce qui est la regle du §6 : « aucune etiquette dans une decision de
la strategie ». Cet invariant est verifie par construction, parce que la
signature de `WindowContext` et celle de `ValidationStats` ne contiennent aucun
champ d'ou une etiquette pourrait venir.

`on_fit` est appele apres l'entrainement initial **et** apres chaque promotion.
C'est la que la strategie proposee recale la reference de son moniteur de
derive : une reference heritee du modele precedent decrirait un comportement
normal qui n'est plus celui du modele en service.
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
    # Evenements que la strategie demande au harnais de journaliser a cette
    # fenetre, pris dans le vocabulaire du §9 : `drift_signal`,
    # `drift_confirmed`. La strategie declare, le harnais ecrit : le journal
    # reste la propriete d'un seul composant, et les metriques de specificite
    # de la surveillance se calculent depuis ce qui a reellement ete declare.
    events: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class TrainingPlan:
    """Decoupage des donnees de reentrainement propose par la strategie.

    `train_X` sert a entrainer le challenger ; `reserved_X` est la fenetre V
    reservee, sur laquelle le harnais le scorera pour alimenter la validation.
    Les deux sont **disjointes** : valider un modele sur ses propres donnees
    d'entrainement ne mesurerait rien.

    `reserved_X` peut etre vide pour une strategie qui ne valide pas (B1). La
    validation en tirera le seul verdict honnete dans ce cas : elle ne peut rien
    conclure, et la strategie doit alors avoir dit explicitement qu'elle promeut
    sans condition.
    """

    train_X: np.ndarray
    reserved_X: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.train_X.ndim != 2:
            raise ValueError(f"train_X doit etre 2D, recu {self.train_X.shape}.")
        if self.reserved_X.ndim != 2:
            raise ValueError(f"reserved_X doit etre 2D, recu {self.reserved_X.shape}.")
        if len(self.train_X) == 0:
            raise ValueError("train_X vide : rien a entrainer.")


@dataclass(frozen=True)
class ValidationStats:
    """Ce que le harnais mesure du challenger, et que la strategie peut lire.

    Aucune etiquette, par construction. Les grandeurs disponibles sont celles
    qu'un exploitant aurait reellement en production :

    - `n_history` : nombre de fenetres observees depuis le debut du flux. C'est
      le garde-fou « historique trop court » du §6 : en dessous d'un seuil, on
      ne sait pas encore ce qu'est le comportement normal.
    - `challenger_alert_rate` / `champion_alert_rate` : taux d'alertes des deux
      modeles sur la **meme** fenetre reservee V. Un taux degenere (presque 0 ou
      tres eleve) signe un modele inutilisable, sans avoir besoin d'etiquettes.
    - `challenger_threshold` : un seuil non fini, ou egal au minimum des scores,
      est degenere.
    - les tailles, pour journalisation et pour les bornes.
    """

    position: int
    n_history: int
    n_train_windows: int
    n_reserved_windows: int
    challenger_alert_rate: float
    champion_alert_rate: float
    challenger_threshold: float
    champion_threshold: float
    challenger_fit_seconds: float

    def as_details(self) -> dict[str, Any]:
        return {
            "n_history": self.n_history,
            "n_train_windows": self.n_train_windows,
            "n_reserved_windows": self.n_reserved_windows,
            "challenger_alert_rate": round(self.challenger_alert_rate, 4),
            "champion_alert_rate": round(self.champion_alert_rate, 4),
            "challenger_threshold": self.challenger_threshold,
        }


@dataclass(frozen=True)
class Verdict:
    """Decision de mise en service du challenger.

    `promote` faux signifie que le challenger est rejete : il est jete, le
    champion reste en service, et le rejet est journalise. Le reentrainement
    reste compte dans le cout — il a bien eu lieu.
    """

    promote: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """Contrat attendu par le harnais de rejeu et par la boucle en ligne."""

    name: str

    def on_window(self, context: WindowContext) -> Decision:
        """Decide s'il faut reentrainer, au vu de la seule fenetre courante et de l'etat interne."""
        ...

    def propose(self, history: np.ndarray, position: int) -> TrainingPlan:
        """Choisit les donnees d'entrainement et la fenetre reservee, dans l'historique observe."""
        ...

    def validate(self, stats: ValidationStats) -> Verdict:
        """Dit si le challenger peut etre mis en service, sans voir aucune etiquette."""
        ...

    def on_fit(self, training_X: np.ndarray, position: int) -> None:
        """Appele apres l'entrainement initial et apres chaque promotion."""
        ...
