"""Etat des pannes injectees dans l'appli de test (livrable M4, phase A).

Deux des cinq types d'anomalies du cadrage se jouent **dans** l'application, parce
qu'ils n'ont pas de sens au niveau du conteneur : la latence ajoutee et la rafale
d'erreurs 5xx. Les trois autres (pic CPU, fuite memoire, saturation) sont injectes
de l'exterieur, par l'injecteur de la phase C.

Deux proprietes comptent ici, et toutes deux pour la meme raison — l'etiquetage.
Le §4.2 demande des anomalies horodatees dans `data/labels/<run_id>.csv` avec
`start_ts, end_ts, type, intensite` :

1. **Toute injection a une fin.** Une injection sans duree qu'on oublie d'arreter
   contamine le reste de la session, et une session dont les etiquettes sont
   fausses est une session perdue — il faut la rejouer entiere.
2. **L'etat est lisible de l'exterieur** (`/chaos/state`, et une jauge dans
   `/metrics`). L'injecteur ecrit les etiquettes, mais c'est l'appli qui sait ce
   qu'elle a reellement subi : pouvoir comparer les deux est ce qui permet de
   detecter une etiquette decalee avant d'avoir construit tout un resultat dessus.

L'horloge est injectable : une panne qui doit durer 90 secondes ne peut pas se
tester en attendant 90 secondes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

# Types d'injection geres dans l'application. Les noms sont ceux du tableau du
# §4.2 du cadrage et se retrouvent tels quels dans le fichier d'etiquettes.
TYPE_LATENCE = "latence"
TYPE_ERREURS = "erreurs_5xx"
TYPES = (TYPE_LATENCE, TYPE_ERREURS)

# Duree maximale d'une injection, en secondes. Une rafale de 5xx de plusieurs
# heures n'est pas une anomalie au sens du cadrage (« un evenement court ») : ce
# serait une derive, et une derive s'obtient par un deploiement, pas par une
# panne. Le plafond protege aussi du doigt qui glisse sur un zero.
DUREE_MAX_S = 3600.0


@dataclass
class Injection:
    """Une injection en cours ou passee, dans la forme attendue par les etiquettes."""

    type: str
    intensite: float
    debut_ts: float
    fin_ts: float

    def est_active(self, maintenant: float) -> bool:
        return self.debut_ts <= maintenant < self.fin_ts


@dataclass
class Chaos:
    """Les injections applicatives en cours, avec leur echeance."""

    horloge: Callable[[], float] = time.time
    actives: dict[str, Injection] = field(default_factory=dict)
    historique: list[Injection] = field(default_factory=list)

    def _injecter(self, type_: str, intensite: float, duree_s: float) -> Injection:
        if type_ not in TYPES:
            raise ValueError(f"Type d'injection inconnu : {type_}")
        if intensite < 0:
            raise ValueError("L'intensite d'une injection ne peut pas etre negative.")
        if duree_s <= 0:
            raise ValueError("Une injection doit avoir une duree strictement positive.")
        if duree_s > DUREE_MAX_S:
            raise ValueError(
                f"Duree de {duree_s:.0f} s refusee : le plafond est {DUREE_MAX_S:.0f} s. "
                "Une anomalie est un evenement court ; une derive s'obtient par un deploiement."
            )
        maintenant = self.horloge()
        injection = Injection(
            type=type_,
            intensite=float(intensite),
            debut_ts=maintenant,
            fin_ts=maintenant + float(duree_s),
        )
        # Une injection du meme type remplace la precedente : deux latences
        # ajoutees simultanement rendraient l'intensite de l'etiquette fausse.
        self.actives[type_] = injection
        self.historique.append(injection)
        return injection

    def injecter_latence(self, supplement_ms: float, duree_s: float) -> Injection:
        return self._injecter(TYPE_LATENCE, supplement_ms, duree_s)

    def injecter_erreurs(self, taux: float, duree_s: float) -> Injection:
        if not 0.0 <= taux <= 1.0:
            raise ValueError("Le taux d'erreurs est une part, entre 0 et 1.")
        return self._injecter(TYPE_ERREURS, taux, duree_s)

    def reinitialiser(self) -> int:
        """Arrete toutes les injections applicatives. Renvoie le nombre arrete.

        L'arret tronque l'echeance a l'instant present dans l'historique, pour que
        l'etiquette corresponde a ce qui a reellement ete subi et non a ce qui
        avait ete demande.
        """
        maintenant = self.horloge()
        arretees = 0
        for injection in self.actives.values():
            if injection.est_active(maintenant):
                injection.fin_ts = maintenant
                arretees += 1
        self.actives.clear()
        return arretees

    def _active(self, type_: str) -> Injection | None:
        injection = self.actives.get(type_)
        if injection is None:
            return None
        if not injection.est_active(self.horloge()):
            del self.actives[type_]
            return None
        return injection

    def latence_ajoutee_ms(self) -> float:
        injection = self._active(TYPE_LATENCE)
        return injection.intensite if injection else 0.0

    def taux_erreurs(self) -> float:
        injection = self._active(TYPE_ERREURS)
        return injection.intensite if injection else 0.0

    def etat(self) -> dict[str, object]:
        """Vue lisible de l'exterieur, et verifiable contre le fichier d'etiquettes."""
        maintenant = self.horloge()
        return {
            "maintenant_ts": maintenant,
            "actives": [
                {
                    "type": injection.type,
                    "intensite": injection.intensite,
                    "debut_ts": injection.debut_ts,
                    "fin_ts": injection.fin_ts,
                    "restant_s": max(0.0, injection.fin_ts - maintenant),
                }
                for type_ in TYPES
                if (injection := self._active(type_)) is not None
            ],
            "historique": [
                {
                    "type": injection.type,
                    "intensite": injection.intensite,
                    "debut_ts": injection.debut_ts,
                    "fin_ts": injection.fin_ts,
                }
                for injection in self.historique
            ],
        }
