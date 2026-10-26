"""Scenarios de derive : injection sur le SMD, composition des sessions serveur.

API publique de l'injection (M1) :
    load_scenario("scenarios/s1_soudaine.yaml") -> DriftScenario
    inject(train, test, label, scenario, seed)  -> InjectedStream
"""

from scenarios.injection import (
    DriftScenario,
    InjectedStream,
    alpha_profile,
    drift_onset,
    inject,
    injectable_metrics,
    load_scenario,
    select_metrics,
)

__all__ = [
    "DriftScenario",
    "InjectedStream",
    "alpha_profile",
    "drift_onset",
    "inject",
    "injectable_metrics",
    "load_scenario",
    "select_metrics",
]
