"""Signal secondaire : ADWIN sur le taux d'alertes (river).

ADWIN (Bifet et Gavalda, 2007) maintient une fenetre adaptative sur un flux de
valeurs et signale un changement quand deux sous-fenetres different assez. On
l'alimente ici avec l'indicatrice d'alerte du detecteur en service, fenetre par
fenetre : c'est le signal du cadrage, « suivre le taux d'alertes ».

Il est **secondaire**, et pour une raison mesuree. D19 : un moniteur fonde sur
le seul taux d'alertes, avec K = 30, confirme une derive sur 26 des 28 machines,
dont 13 des 15 machines stables. OBS09 confirme l'autre moitie du tableau : le
taux d'alertes est tres **sensible** a la derive injectee (facteur median 7,0 a
k = 2), mais sans specificite. Il est donc utilise en complement de la part de
features en derive (D18), jamais seul.

Note d'API, verifiee sur river 0.22.0 : seul `drift_detected` est fiable.
L'attribut `n_detections` reste a 0 alors que `drift_detected` passe bien a True
(verifie sur un flux passant de 5 % a 30 % d'alertes : detection 47 pas apres le
changement). Le compteur est donc tenu ici et non lu depuis river.
"""

from __future__ import annotations

from typing import Any

from river.drift import ADWIN


class AlertRateMonitor:
    """ADWIN alimente par l'indicatrice d'alerte, avec compteur local.

    `delta` est le niveau de confiance d'ADWIN : plus il est petit, plus le
    detecteur est conservateur. `grace_period` evite de signaler sur les toutes
    premieres fenetres, ou la fenetre adaptative est encore vide.
    """

    def __init__(self, delta: float = 0.002, grace_period: int = 10, clock: int = 32) -> None:
        self._params: dict[str, Any] = {
            "delta": float(delta),
            "grace_period": int(grace_period),
            "clock": int(clock),
        }
        self._detector = ADWIN(**self._params)
        self.n_detections = 0
        self._latched = False

    def update(self, alert: bool) -> bool:
        """Ajoute une fenetre ; renvoie True si ADWIN signale un changement a cette fenetre."""
        self._detector.update(int(bool(alert)))
        if self._detector.drift_detected:
            self.n_detections += 1
            self._latched = True
            return True
        return False

    @property
    def latched(self) -> bool:
        """Vrai si ADWIN a signale au moins un changement depuis le dernier recalage.

        C'est cette forme **durable** que le moniteur combine au signal par
        features, et non le signal instantane. ADWIN signale a une fenetre
        isolee : exiger cette detection sur K fenetres consecutives ne se
        produirait jamais, et la regle `features_and_alerts` ne confirmerait
        aucune derive. L'etat latche dit « le taux d'alertes a change depuis que
        ce modele est en service », ce qui est bien la question posee.
        """
        return self._latched

    def reset(self) -> None:
        """Repart d'un detecteur neuf, apres une promotion.

        Le taux d'alertes du nouveau champion n'a pas de raison de prolonger
        celui de l'ancien : conserver la fenetre adaptative ferait signaler un
        changement qui n'est que l'effet du remplacement de modele.
        """
        self._detector = ADWIN(**self._params)
        self._latched = False

    @property
    def width(self) -> int:
        """Taille courante de la fenetre adaptative, journalisee pour diagnostic."""
        return int(self._detector.width)
