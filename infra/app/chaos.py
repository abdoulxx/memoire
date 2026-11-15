"""Etat des pannes injectees dans l'appli de test (livrable M4, phase A ; D46).

Quatre des cinq types d'anomalies du cadrage se jouent **dans** l'application :
la latence ajoutee, la rafale d'erreurs 5xx, et depuis D46, le pic CPU et la
fuite memoire. Les deux dernieres etaient a l'origine confiees a `stress-ng`
dans un conteneur dedie (injecteur de la phase C), mais cAdvisor ne parvient pas
a identifier les conteneurs sur le magasin d'images par containerd-snapshotter
de Docker Engine 29 (constat du 15/11/2026 sur `sidy-dev`) : `cpu_appli` et
`memoire_appli` resteraient vides quelle que soit l'origine du conteneur. Les
porter dans l'application permet de les mesurer via le collecteur de processus
de `prometheus_client` (`infra/app/metrics.py`), independant de Docker. Seule la
saturation (Locust) reste injectee de l'exterieur : une charge HTTP n'a pas de
sens au niveau du processus qui la subit.

Deux proprietes comptent ici, et toutes deux pour la meme raison — l'etiquetage.
Le §4.2 demande des anomalies horodatees dans `data/labels/<run_id>.csv` avec
`start_ts, end_ts, type, intensite` :

1. **Toute injection a une fin.** Une injection sans duree qu'on oublie d'arreter
   contamine le reste de la session, et une session dont les etiquettes sont
   fausses est une session perdue — il faut la rejouer entiere. Pour le pic CPU
   et la fuite memoire, la fin doit aussi etre **reelle** : le travail de fond
   qu'ils lancent surveille en continu si l'injection qui l'a demande est
   toujours active, pour qu'un arret d'urgence (`/chaos/reset`) l'arrete
   vraiment, et pas seulement dans l'etiquette.
2. **L'etat est lisible de l'exterieur** (`/chaos/state`, et une jauge dans
   `/metrics`). L'injecteur ecrit les etiquettes, mais c'est l'appli qui sait ce
   qu'elle a reellement subi : pouvoir comparer les deux est ce qui permet de
   detecter une etiquette decalee avant d'avoir construit tout un resultat dessus.

L'horloge est injectable : une panne qui doit durer 90 secondes ne peut pas se
tester en attendant 90 secondes. Le travail de fond du CPU et de la memoire,
lui, consomme de vraies ressources et doit donc se caler sur l'horloge reelle
quelle que soit l'horloge de test utilisee par ailleurs ; c'est pourquoi il est
injectable separement (`demarrer_cpu`, `demarrer_memoire`), et les tests le
remplacent par une fonction qui ne fait rien.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

# Types d'injection geres dans l'application. Les noms sont ceux du tableau du
# §4.2 du cadrage et se retrouvent tels quels dans le fichier d'etiquettes.
TYPE_LATENCE = "latence"
TYPE_ERREURS = "erreurs_5xx"
TYPE_CPU = "pic_cpu"
TYPE_MEMOIRE = "fuite_memoire"
TYPES = (TYPE_LATENCE, TYPE_ERREURS, TYPE_CPU, TYPE_MEMOIRE)

# Duree maximale d'une injection, en secondes. Une rafale de 5xx de plusieurs
# heures n'est pas une anomalie au sens du cadrage (« un evenement court ») : ce
# serait une derive, et une derive s'obtient par un deploiement, pas par une
# panne. Le plafond protege aussi du doigt qui glisse sur un zero. L'injecteur
# impose par ailleurs un plafond plus bas (`configs/injection.yaml`,
# `securite.duree_max_s`) : celui-ci est le garde-fou de l'application seule.
DUREE_MAX_S = 3600.0

# Plafond d'une fuite memoire, en Mio. Nettement sous la limite de 512 Mio du
# conteneur de l'appli (`APP_MEMORY`, docker-compose.yml) : l'anomalie doit
# rester mesurable sans jamais risquer que le noyau tue le conteneur par manque
# de memoire, ce qui tronquerait l'etiquette et pourrait interrompre la session.
PLAFOND_MEMOIRE_MO = 200.0


def _boucle_cpu(fin_absolue_ts: float, actif: Callable[[], bool]) -> None:
    """Charge un cœur a 100 % jusqu'a l'echeance ou a l'arret anticipe.

    Boucle delibrement vide : c'est le travail lui-meme qui doit coûter du CPU,
    pas une attente. `actif()` est revérifié en continu pour qu'un arret
    d'urgence coupe reellement le travail, et pas seulement l'etiquette.
    """
    while time.time() < fin_absolue_ts and actif():
        pass


def _demarrer_cpu_reel(threads: float, duree_s: float, actif: Callable[[], bool]) -> None:
    fin_absolue = time.time() + duree_s
    for _ in range(max(1, int(threads))):
        thread = threading.Thread(target=_boucle_cpu, args=(fin_absolue, actif), daemon=True)
        thread.start()


def _rampe_memoire(mo: float, duree_s: float, actif: Callable[[], bool]) -> None:
    """Alloue progressivement jusqu'a `mo` Mio, puis libere a la fin.

    Une rampe et non un bloc d'un coup : une fuite memoire reelle monte
    progressivement, et une rampe rend aussi l'anomalie visible plus longtemps
    dans `process_resident_memory_bytes` avant d'atteindre son plateau.
    """
    paliers = 10
    octets_par_palier = max(1, int(mo * 1024 * 1024 / paliers))
    fin_absolue = time.time() + duree_s
    pas = max(0.05, duree_s / paliers)
    blocs: list[bytearray] = []
    for _ in range(paliers):
        if not actif() or time.time() >= fin_absolue:
            break
        blocs.append(bytearray(octets_par_palier))
        time.sleep(min(pas, max(0.0, fin_absolue - time.time())))
    while actif() and time.time() < fin_absolue:
        time.sleep(min(1.0, fin_absolue - time.time()))
    blocs.clear()


def _demarrer_memoire_reelle(mo: float, duree_s: float, actif: Callable[[], bool]) -> None:
    thread = threading.Thread(target=_rampe_memoire, args=(mo, duree_s, actif), daemon=True)
    thread.start()


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
    # Travail de fond reel du pic CPU et de la fuite memoire. Injectable pour
    # que les tests n'aient pas a bruler un vrai cœur ou a allouer de la vraie
    # memoire : ils y substituent une fonction qui ne fait rien.
    demarrer_cpu: Callable[[float, float, Callable[[], bool]], None] = _demarrer_cpu_reel
    demarrer_memoire: Callable[[float, float, Callable[[], bool]], None] = _demarrer_memoire_reelle

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

    def injecter_cpu(self, threads: float, duree_s: float) -> Injection:
        if threads < 1:
            raise ValueError("Une injection de CPU demande au moins un fil d'execution.")
        injection = self._injecter(TYPE_CPU, threads, duree_s)
        self.demarrer_cpu(threads, duree_s, lambda: self.actives.get(TYPE_CPU) is injection)
        return injection

    def injecter_memoire(self, mo: float, duree_s: float) -> Injection:
        if mo > PLAFOND_MEMOIRE_MO:
            raise ValueError(
                f"Allocation de {mo:g} Mio refusee : le plafond est "
                f"{PLAFOND_MEMOIRE_MO:g} Mio, nettement sous la limite memoire du "
                "conteneur (APP_MEMORY), pour ne jamais risquer un arret par "
                "manque de memoire."
            )
        injection = self._injecter(TYPE_MEMOIRE, mo, duree_s)
        self.demarrer_memoire(mo, duree_s, lambda: self.actives.get(TYPE_MEMOIRE) is injection)
        return injection

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

    def intensite_active(self, type_: str) -> float:
        """Intensite de l'injection en cours pour ce type, 0 si aucune.

        Generique sur les quatre types : c'est ce qui permet a `/metrics` de
        refleter le pic CPU et la fuite memoire sans cas particulier, exactement
        comme la latence et les erreurs.
        """
        injection = self._active(type_)
        return injection.intensite if injection else 0.0

    def latence_ajoutee_ms(self) -> float:
        return self.intensite_active(TYPE_LATENCE)

    def taux_erreurs(self) -> float:
        return self.intensite_active(TYPE_ERREURS)

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
