"""Scenario de charge pour l'anomalie de saturation (livrable M4, phase C).

Monte dans un conteneur Locust limite (`--cpus`, `--memory`) par l'injecteur, en
lecture seule. Il ne frappe que `/work` de l'appli de test : ni `/health`, ni
`/metrics`, ni le plan de controle `/chaos/*`, qui sont exclus des metriques du
tableau de bord (filtre `SANS_CONTROLE`) et dont le trafic ne representerait
aucune charge utile.

La saturation est une anomalie de type « pic de trafic bref et intense » (§4.2) :
le but n'est pas de mesurer la capacite de l'appli mais de produire une periode
ou la latence et l'usage CPU sortent du regime normal, a un instant connu.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class Visiteur(HttpUser):
    """Un client qui demande du travail a l'appli de test."""

    # Une attente courte et variable : des arrivees regulieres au millieme de
    # seconde pres produiraient un trafic qui ne ressemble a rien de reel, et
    # une latence sans dispersion.
    wait_time = between(0.1, 0.5)

    @task(4)
    def travail_ordinaire(self) -> None:
        self.client.get("/work", name="/work")

    @task(1)
    def travail_lourd(self) -> None:
        """Une part minoritaire de requetes couteuses, qui etire la queue du p95."""
        self.client.get(
            f"/work?complexite={random.uniform(3.0, 8.0):.2f}",  # noqa: S311
            name="/work (lourd)",
        )
