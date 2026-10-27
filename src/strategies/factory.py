"""Construction d'une strategie depuis sa configuration.

Un seul endroit convertit les **minutes** des fichiers de configuration en
**fenetres** du harnais. Les configurations parlent en minutes parce que c'est
l'unite dans laquelle un exploitant raisonne et celle du memoire ; le code
raisonne en fenetres parce que c'est l'unite du flux. Dupliquer cette
conversion serait le genre d'erreur qui ne se voit pas dans les resultats.

Avec un pas d'echantillonnage d'une minute et un stride de 1 (SMD), une fenetre
vaut une minute et la conversion est l'identite. Elle ne le restera pas sur les
sessions du serveur : la conversion est donc explicite et testee.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.strategies.base import Strategy
from src.strategies.periodic import PeriodicStrategy
from src.strategies.proposed import ProposedStrategy
from src.strategies.static import StaticStrategy

STRATEGIES = ("static", "periodic", "proposed")


def minutes_to_windows(minutes: float, stride: int, sampling_interval_seconds: int) -> int:
    """Convertit une duree en nombre de fenetres, arrondi au plus proche, minimum 1.

    Une fenetre avance de `stride` points et un point vaut
    `sampling_interval_seconds` secondes : une fenetre couvre donc
    `stride * sampling_interval_seconds / 60` minutes.
    """
    per_window = stride * sampling_interval_seconds / 60.0
    if per_window <= 0:
        raise ValueError(
            f"Duree de fenetre nulle ou negative : stride={stride}, "
            f"sampling_interval_seconds={sampling_interval_seconds}."
        )
    return max(1, int(round(float(minutes) / per_window)))


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Fusion recursive, les surcharges gagnant. La base n'est jamais modifiee."""
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_strategy(
    strategy_config: dict[str, Any],
    features_config: dict[str, Any],
    drift_config: dict[str, Any],
    sampling_interval_seconds: int,
) -> Strategy:
    """Instancie la strategie decrite par `strategy_config`."""
    kind = str(strategy_config.get("strategy", "static"))
    if kind not in STRATEGIES:
        raise ValueError(f"Strategie inconnue : {kind!r}. Attendu l'une de {STRATEGIES}.")

    stride = int(features_config["stride"])

    def windows(minutes: float) -> int:
        return minutes_to_windows(minutes, stride, sampling_interval_seconds)

    if kind == "static":
        return StaticStrategy()

    if kind == "periodic":
        return PeriodicStrategy(
            period_windows=windows(strategy_config["period_minutes"]),
            train_windows=windows(strategy_config["train_minutes"]),
        )

    validation = dict(strategy_config.get("validation", {}) or {})
    exclusion = dict(strategy_config.get("exclusion", {}) or {})
    return ProposedStrategy(
        drift_config=_merge(drift_config, strategy_config.get("drift_overrides", {})),
        train_windows=windows(strategy_config["train_minutes"]),
        reserved_windows=windows(strategy_config["reserved_minutes"])
        if strategy_config.get("reserved_minutes")
        else 0,
        cooldown_windows=windows(strategy_config["cooldown_minutes"])
        if strategy_config.get("cooldown_minutes")
        else 0,
        validation_enabled=bool(validation.get("enabled", True)),
        min_train_windows=windows(strategy_config.get("min_train_minutes", 0)),
        min_history=int(validation.get("min_history", 300)),
        alert_rate_bounds=(
            float(validation.get("alert_rate_min", 0.001)),
            float(validation.get("alert_rate_max", 0.30)),
        ),
        exclusion_method=str(exclusion.get("method", "none")),
        exclusion_percentile=float(exclusion.get("percentile", 99.0)),
        name=str(strategy_config.get("name", "proposed")),
    )
