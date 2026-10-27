"""Etape 2 de la strategie proposee : la confirmation sur K fenetres.

Une derive est un changement **durable** du comportement normal ; une anomalie
est un evenement **court**. Toute la difficulte du sujet est de ne pas
confondre les deux, et c'est cette etape qui s'en charge : le signal doit
persister K fenetres **consecutives** pour etre retenu.

Une seule interruption remet le compteur a zero. C'est volontairement strict :
D19 a montre qu'un signal de taux d'alertes avec K = 30 confirme une derive sur
13 des 15 machines stables, c'est-a-dire qu'il ne confirme rien du tout. K est
calibre sur le perimetre de mise au point, avec pour cible zero fausse
confirmation sur le scenario `none`.
"""

from __future__ import annotations


class Confirmation:
    """Compteur de persistance : K signaux consecutifs valent confirmation.

    Le compteur ne se reclenche pas tant qu'il n'a pas ete remis a zero par
    `reset()`, appele apres une promotion : sans cela, une derive confirmee
    continuerait a se confirmer a chaque fenetre suivante et declencherait un
    reentrainement en rafale.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("K doit valoir au moins 1.")
        self.k = int(k)
        self._streak = 0
        self._latched = False

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def confirmed(self) -> bool:
        return self._latched

    def update(self, signal: bool) -> bool:
        """Renvoie True a la fenetre **exacte** ou la confirmation est acquise.

        Renvoie False aux fenetres suivantes meme si le signal persiste : la
        confirmation est un evenement, pas un etat. C'est ce qui permet de
        dater le delai de detection de la derive sans ambiguite.
        """
        if not signal:
            self._streak = 0
            return False

        self._streak += 1
        if self._streak >= self.k and not self._latched:
            self._latched = True
            return True
        return False

    def reset(self) -> None:
        """Remet le compteur et le verrou a zero, apres une promotion."""
        self._streak = 0
        self._latched = False
