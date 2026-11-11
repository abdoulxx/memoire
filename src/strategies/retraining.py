"""Cycle de reentrainement, partage par le rejeu et la boucle en ligne.

Ce module existe pour une raison precise : le chapitre 6 compare des strategies
en rejeu, le chapitre 5 montre la meme strategie en ligne, et les deux ne valent
que si c'est **le meme** cycle qui tourne. Si la boucle en ligne reecrivait ce
qui suit, une divergence — un `reserved_X` mesure autrement, une statistique de
validation calculee sur le challenger plutot que sur la fenetre reservee —
rendrait la demonstration et les resultats incomparables, sans qu'aucun test ne
le signale.

Le cycle est celui du §6 du cadrage, et il est **le meme** pour P, A1 et B1 :

    plan     = strategy.propose(history, position)   # quelles donnees, quelle fenetre V
    challenger.fit(plan.train_X)                     # l'appelant entraine
    stats    = mesures sans etiquette sur V          # l'appelant mesure
    verdict  = strategy.validate(stats)              # mise en service ou rejet

Deux choses restent a l'appelant, parce qu'elles n'ont pas le meme sens dans les
deux modes : la **promotion** (remplacement en memoire en rejeu, deplacement de
l'alias `champion` dans le registre en ligne) et le **scoring fantome** (D16),
qui n'existe qu'en rejeu puisqu'il sert a calculer hors ligne une metrique du §7.

La regle du §6 — aucune etiquette dans une decision de la strategie — est tenue
ici par construction : ce module ne recoit aucune etiquette, et `ValidationStats`
n'a pas de champ ou en mettre.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.isolation_forest import IsolationForestDetector
from src.strategies.base import Strategy, TrainingPlan, ValidationStats, Verdict


def alert_rate(scores: np.ndarray, threshold: float) -> float:
    """Taux d'alertes d'un modele sur un bloc de fenetres. Aucune etiquette requise."""
    if len(scores) == 0:
        return float("nan")
    return float(np.count_nonzero(scores > threshold) / len(scores))


@dataclass(frozen=True)
class RetrainOutcome:
    """Ce qu'un cycle de reentrainement produit, avant toute promotion."""

    challenger: IsolationForestDetector
    plan: TrainingPlan
    stats: ValidationStats
    verdict: Verdict
    validation_seconds: float

    @property
    def promote(self) -> bool:
        return self.verdict.promote

    def details(self) -> dict[str, Any]:
        """Details du plan puis de la validation, dans cet ordre.

        `stats` est la mesure faite par l'appelant et doit primer sur ce que la
        strategie a declare dans `plan.details`.
        """
        return {**self.plan.details, **self.stats.as_details()}


def run_retrain_cycle(
    strategy: Strategy,
    *,
    history_X: np.ndarray,
    position: int,
    n_history: int,
    champion: IsolationForestDetector,
    build: Callable[[], IsolationForestDetector],
) -> RetrainOutcome:
    """Entraine un challenger, le mesure sans etiquette, et recueille le verdict.

    `history_X` ne contient que des fenetres **deja observees** : c'est la
    garantie prequentielle, et c'est a l'appelant de ne pas lui passer le futur.
    `n_history` est le nombre de fenetres vues depuis le debut du flux ; il peut
    depasser `len(history_X)` en ligne, ou l'historique en memoire est borne,
    alors qu'en rejeu les deux coincident. Le garde-fou de validation s'appuie
    sur `n_history` (« a-t-on vu assez de fenetres pour savoir ce qu'est le
    normal ? »), question a laquelle la taille du tampon ne repond pas.
    """
    plan = strategy.propose(history_X, position)
    challenger = build()
    challenger.fit(plan.train_X)

    # Les deux modeles sont mesures sur la **meme** fenetre reservee, la plus
    # recente : c'est celle qui decrit le regime dans lequel le challenger devra
    # servir. Le temps de ces scorings est compte a part du reentrainement, pour
    # que le cout rapporte au §7 distingue entrainement et validation.
    started = time.perf_counter()
    vide = np.empty(0, dtype=np.float64)
    challenger_reserved = challenger.score(plan.reserved_X) if len(plan.reserved_X) else vide
    champion_reserved = champion.score(plan.reserved_X) if len(plan.reserved_X) else vide
    validation_seconds = time.perf_counter() - started

    stats = ValidationStats(
        position=position,
        n_history=int(n_history),
        n_train_windows=len(plan.train_X),
        n_reserved_windows=len(plan.reserved_X),
        challenger_alert_rate=alert_rate(challenger_reserved, challenger.threshold),
        champion_alert_rate=alert_rate(champion_reserved, champion.threshold),
        challenger_threshold=float(challenger.threshold),
        champion_threshold=float(champion.threshold),
        challenger_fit_seconds=float(challenger.fit_seconds),
    )
    return RetrainOutcome(
        challenger=challenger,
        plan=plan,
        stats=stats,
        verdict=strategy.validate(stats),
        validation_seconds=validation_seconds,
    )
