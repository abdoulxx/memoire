"""L'interface de notification attendue par le service de detection.

Un `Protocol` et non une classe de base : la boucle n'a pas besoin de connaitre
Telegram, et le test de la boucle ne doit pas avoir a construire un
notificateur reel pour verifier qu'elle rince ses groupes a chaque tick. C'est
l'interface `Notifier.send(event)` du §9, dans la forme qu'a prise le mode en
ligne — un abonne du journal d'evenements, plus un rincage periodique impose par
la regle de regroupement du §9.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Notificateur(Protocol):
    """Abonne du journal qui pousse certains evenements vers l'exterieur."""

    def observer_evenement(self, entree: dict[str, Any]) -> None:
        """Prend connaissance d'une ligne de journal. Ne doit jamais lever."""

    def rincer(self, *, forcer: bool = False) -> int:
        """Envoie ce qui attend et dont la fenetre de regroupement est ecoulee."""
