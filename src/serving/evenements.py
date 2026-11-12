"""Vocabulaire des evenements du mode en ligne.

Ce vocabulaire est un **contrat** entre quatre consommateurs : le journal JSON
lignes (§9) qui refuse d'ecrire un nom hors liste, l'exporter Prometheus qui en
fait les valeurs de l'etiquette `event`, les notifications Telegram qui filtrent
dessus, et les panneaux Grafana qui les interrogent par nom. Il vit donc dans son
propre module, et non dans la boucle : la boucle est l'un des producteurs, pas la
proprietaire du contrat, et un import depuis la boucle rendrait l'exporter
dependant d'elle pour une simple liste de chaines.
"""

from __future__ import annotations

# Les sept premiers sont ceux du §9, que le harnais de rejeu ecrit deja :
# Telegram et Grafana lisent le meme vocabulaire dans les deux modes.
EVENEMENTS_REJEU = (
    "alert",
    "drift_signal",
    "drift_confirmed",
    "retrain_start",
    "retrain_end",
    "promoted",
    "rejected",
)

# Propres au mode en ligne : ce qui n'existe pas en rejeu, ou le flux est complet
# et le modele toujours disponible.
EVENEMENTS_EN_LIGNE = (
    "service_start",
    "service_stop",
    "window_skipped",
    "collect_gap",
    "registry_error",
)

EVENEMENTS = EVENEMENTS_REJEU + EVENEMENTS_EN_LIGNE
