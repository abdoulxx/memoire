"""Injecteur d'anomalies sur le serveur (livrable M4, phase C).

Ce qui est en jeu ici n'est pas la difficulte technique — lancer `stress-ng`
dans un conteneur est simple — mais le §4.3 du cadrage : les applications de
production tournent sur la meme machine, et une erreur de cible ne se repare
pas. Trois regles, chacune couverte par un test :

1. **L'injecteur ne cree que des conteneurs marques deux fois** : un prefixe de
   nom reserve ET une etiquette Docker. Il n'arrete que des conteneurs qui
   portent les deux marques, et refuse tous les autres, meme pour un arret.
2. **Aucun conteneur sans limites.** Les limites CPU et memoire viennent de
   `configs/injection.yaml`, et un type d'injection qui cree un conteneur sans
   limites est refuse a la lecture de la configuration, pas au lancement.
3. **Aucune injection sans fin.** La duree est obligatoire, plafonnee par la
   configuration, et passee **aussi** a `stress-ng --timeout` : si l'injecteur
   est tue, le conteneur s'arrete quand meme de lui-meme.

L'injection est synchrone : l'injecteur attend la fin, puis ecrit l'etiquette
avec l'instant de fin reellement atteint. Une interruption produit une etiquette
plus courte, jamais une etiquette fausse.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from infra.injector.docker_cli import ClientDocker, ErreurDocker
from infra.injector.labels import Anomalie, Derive, ecrire_anomalie, ecrire_derive

MOYEN_CONTENEUR = "conteneur"
MOYEN_APPLI = "appli"
MOYEN_CHARGE = "charge"
MOYENS = (MOYEN_CONTENEUR, MOYEN_APPLI, MOYEN_CHARGE)

# Les types qui creent un conteneur doivent declarer des limites.
MOYENS_CONTENEUR = (MOYEN_CONTENEUR, MOYEN_CHARGE)

# Endpoint applicatif et cle du corps de requete, pour chaque type dont le
# `moyen` est `appli`. D46 : pic_cpu et fuite_memoire s'y sont ajoutes, parce
# que cAdvisor ne voit aucun conteneur sur l'hote de demonstration (Docker
# Engine 29, magasin d'images par containerd-snapshotter) et que le pic CPU et
# la fuite memoire restent donc invisibles s'ils sont injectes dans un
# conteneur a part. `configs/injection.yaml` garde `moyen: conteneur` comme
# option pour un usage ponctuel, hors session enregistree (stress-ng reste
# dans `_injecter_par_conteneur` / `_commande_stress`, inchange).
ENDPOINTS_APPLI: dict[str, tuple[str, str]] = {
    "latence": ("/chaos/latency", "supplement_ms"),
    "erreurs": ("/chaos/errors", "taux"),
    "pic_cpu": ("/chaos/cpu", "threads"),
    "fuite_memoire": ("/chaos/memory", "mo"),
}


class RefusDeSecurite(RuntimeError):
    """L'injecteur a refuse d'agir. Jamais rattrapee en silence : c'est le garde-fou."""


@dataclass(frozen=True)
class ResultatArret:
    """Ce que l'arret d'urgence a reellement fait.

    Le plan de controle de l'appli de test est un second moyen d'injection, qui
    peut echouer independamment des conteneurs : l'appli peut etre eteinte, ou
    l'URL configuree peut pointer ailleurs. Un arret d'urgence qui annoncerait
    une remise a zero qui n'a pas eu lieu serait un probleme de surete a lui
    seul — d'ou ce resultat explicite plutot qu'un simple booleen.
    """

    conteneurs: list[str]
    plan_de_controle: str


class ClientAppli(Protocol):
    """Le plan de controle de l'appli de test, vu par l'injecteur."""

    def injecter(self, chemin: str, charge: dict[str, float]) -> dict[str, Any]: ...

    def reinitialiser(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Limites:
    """Limites Docker d'un conteneur d'injection. Obligatoires (§4.3)."""

    cpus: float
    memoire: str

    def arguments(self) -> list[str]:
        # `--memory-swap` egal a `--memory` : sans cela le noyau laisse le
        # conteneur deborder sur l'echange, et la limite memoire ne limite plus.
        return [
            "--cpus",
            f"{self.cpus:g}",
            "--memory",
            self.memoire,
            "--memory-swap",
            self.memoire,
        ]


@dataclass(frozen=True)
class TypeInjection:
    """Un type d'anomalie du §4.2, tel que la configuration le decrit."""

    nom: str
    moyen: str
    intensite_defaut: float
    duree_defaut_s: float
    limites: Limites | None = None
    locust: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reglages:
    """Contenu verifie de `configs/injection.yaml`."""

    prefixe: str
    etiquette: str
    duree_max_s: float
    reseau: str
    image_stress: str
    image_charge: str
    url_appli: str
    dossier_etiquettes: Path
    types: dict[str, TypeInjection]

    @classmethod
    def depuis_config(cls, config: dict[str, Any], racine: Path) -> Reglages:
        securite = config["securite"]
        types: dict[str, TypeInjection] = {}
        for nom, brut in config["types"].items():
            moyen = brut["moyen"]
            if moyen not in MOYENS:
                raise ValueError(f"Type `{nom}` : moyen inconnu `{moyen}`.")
            limites = None
            if "limites" in brut:
                limites = Limites(
                    cpus=float(brut["limites"]["cpus"]),
                    memoire=str(brut["limites"]["memoire"]),
                )
            # Garde-fou a la lecture, et non au lancement : un type qui cree un
            # conteneur sans limites ne doit meme pas pouvoir etre charge.
            if moyen in MOYENS_CONTENEUR and limites is None:
                raise ValueError(
                    f"Type `{nom}` : un type qui cree un conteneur doit declarer "
                    "`limites.cpus` et `limites.memoire` (CLAUDE.md §4.3)."
                )
            types[nom] = TypeInjection(
                nom=nom,
                moyen=moyen,
                intensite_defaut=float(brut["intensite_defaut"]),
                duree_defaut_s=float(brut["duree_defaut_s"]),
                limites=limites,
                locust=dict(brut.get("locust") or {}),
            )
        return cls(
            prefixe=str(securite["prefixe"]),
            etiquette=str(securite["etiquette"]),
            duree_max_s=float(securite["duree_max_s"]),
            reseau=str(securite["reseau"]),
            image_stress=str(config["image_stress"]),
            image_charge=str(config["image_charge"]),
            url_appli=os.environ.get("MEMOIRE_APP_URL") or str(config["appli"]["url"]),
            dossier_etiquettes=racine / config["etiquettes"]["dossier"],
            types=types,
        )


@dataclass
class Injecteur:
    """Cree, surveille et arrete les injections d'une session."""

    reglages: Reglages
    docker: ClientDocker
    appli: ClientAppli | None = None
    horloge: Callable[[], float] = time.time
    attente: Callable[[float], None] = time.sleep
    racine: Path = Path(".")
    # Pas de surveillance du conteneur pendant l'injection. Un `docker inspect`
    # toutes les deux secondes coute une vingtaine de millisecondes et permet de
    # ne pas etiqueter une anomalie plus longue que celle qui a eu lieu.
    pas_surveillance_s: float = 2.0

    # ----- garde-fous -------------------------------------------------------

    def _verifier_duree(self, duree_s: float) -> None:
        if duree_s <= 0:
            raise RefusDeSecurite(
                "Une injection sans duree est refusee : une anomalie est un "
                "evenement court, ce qui dure est une derive (§5)."
            )
        if duree_s > self.reglages.duree_max_s:
            raise RefusDeSecurite(
                f"Duree {duree_s:g} s au-dela du plafond "
                f"{self.reglages.duree_max_s:g} s de configs/injection.yaml."
            )

    def _type(self, nom: str) -> TypeInjection:
        if nom not in self.reglages.types:
            connus = ", ".join(sorted(self.reglages.types))
            raise RefusDeSecurite(f"Type d'injection inconnu `{nom}`. Connus : {connus}.")
        return self.reglages.types[nom]

    def _nom_conteneur(self, type_: str) -> str:
        return f"{self.reglages.prefixe}-{type_}-{int(self.horloge())}"

    def _nous_appartient(self, nom: str, etiquettes: dict[str, str]) -> bool:
        """Les DEUX marques, jamais une seule.

        Le prefixe seul s'usurpe par une faute de frappe ; l'etiquette seule a
        pu etre posee par quelqu'un d'autre. Exiger les deux rend impossible de
        viser par accident un conteneur que l'injecteur n'a pas cree.
        """
        return nom.startswith(f"{self.reglages.prefixe}-") and (
            etiquettes.get(self.reglages.etiquette) == "1"
        )

    def _inspecter(self, nom: str) -> dict[str, Any]:
        donnees = json.loads(self.docker.executer(["inspect", nom]))
        if not donnees:
            raise RefusDeSecurite(f"Conteneur `{nom}` introuvable.")
        return dict(donnees[0])

    def _etiquettes_de(self, nom: str) -> dict[str, str]:
        return self._inspecter(nom).get("Config", {}).get("Labels") or {}

    def _verifier_demarrage(self, nom: str) -> None:
        """Un conteneur qui s'arrete aussitot ne produit aucune anomalie.

        Sans cette verification, `docker run` reussit, l'injecteur attend la
        duree demandee puis ecrit une etiquette : le fichier annonce une anomalie
        qui n'a jamais eu lieu, et c'est la pire chose qui puisse arriver a un
        jeu de donnees etiquete. Le cas s'est produit a la verification de la
        phase C — `stress-ng` s'interrompait faute de `temp-path` inscriptible.
        """
        try:
            etat = self._inspecter(nom).get("State", {})
        except ErreurDocker as erreur:  # deja disparu : `--rm` a fait son office
            raise ErreurDocker(
                f"Le conteneur `{nom}` a disparu avant d'avoir pu etre verifie, "
                "donc l'injection n'a pas eu lieu. Aucune etiquette n'est ecrite."
            ) from erreur
        if etat.get("Running"):
            return
        journal = ""
        with contextlib.suppress(ErreurDocker):
            journal = self.docker.executer(["logs", "--tail", "5", nom]).strip()
        raise ErreurDocker(
            f"Le conteneur `{nom}` s'est arrete aussitot (code "
            f"{etat.get('ExitCode')}), donc l'injection n'a pas eu lieu et aucune "
            f"etiquette n'est ecrite. Journal : {journal or '(vide)'}"
        )

    def _arguments_communs(self, nom: str, type_: str, limites: Limites) -> list[str]:
        return [
            "run",
            "--rm",
            "--detach",
            "--name",
            nom,
            "--label",
            f"{self.reglages.etiquette}=1",
            "--label",
            f"memoire.type={type_}",
            *limites.arguments(),
        ]

    # ----- inventaire -------------------------------------------------------

    def lister(self) -> list[str]:
        """Les conteneurs d'injection en cours, et eux seuls."""
        sortie = self.docker.executer(
            [
                "ps",
                "--filter",
                f"label={self.reglages.etiquette}=1",
                "--format",
                "{{.Names}}",
            ]
        )
        noms = [ligne.strip() for ligne in sortie.splitlines() if ligne.strip()]
        # Deuxieme marque verifiee ici : le filtre Docker ne porte que sur
        # l'etiquette, et une etiquette peut etre posee par n'importe qui.
        return sorted(nom for nom in noms if nom.startswith(f"{self.reglages.prefixe}-"))

    def arreter(self, nom: str) -> None:
        """Arrete un conteneur d'injection. Refuse tout le reste."""
        if not self._nous_appartient(nom, self._etiquettes_de(nom)):
            raise RefusDeSecurite(
                f"Refus d'arreter `{nom}` : ce conteneur ne porte pas les deux "
                f"marques de l'injecteur (prefixe `{self.reglages.prefixe}-` et "
                f"etiquette `{self.reglages.etiquette}=1`). L'injecteur ne touche "
                "que ce qu'il a cree."
            )
        self.docker.executer(["stop", "--time", "2", nom])

    def arreter_tout(self) -> ResultatArret:
        """L'arret d'urgence du §4.3 (`make stop-chaos`).

        Les conteneurs d'abord, puis le plan de controle : un arret d'urgence
        qui laisserait une rafale de 5xx en cours ne serait pas un arret
        d'urgence. L'echec du second n'empeche pas le premier, et il est
        rapporte au lieu d'etre avale.
        """
        arretes: list[str] = []
        for nom in self.lister():
            self.arreter(nom)
            arretes.append(nom)
        return ResultatArret(conteneurs=arretes, plan_de_controle=self._reinitialiser())

    # ----- injections -------------------------------------------------------

    def injecter(
        self,
        type_: str,
        intensite: float | None = None,
        duree_s: float | None = None,
        run_id: str | None = None,
    ) -> Anomalie:
        """Injecte une anomalie, attend sa fin, et renvoie ce qui a ete subi."""
        reglage = self._type(type_)
        intensite = reglage.intensite_defaut if intensite is None else float(intensite)
        duree_s = reglage.duree_defaut_s if duree_s is None else float(duree_s)
        self._verifier_duree(duree_s)
        if intensite <= 0:
            raise RefusDeSecurite("Une intensite nulle ou negative n'est pas une anomalie.")

        if reglage.moyen == MOYEN_APPLI:
            anomalie = self._injecter_par_appli(reglage, intensite, duree_s)
        elif reglage.moyen == MOYEN_CHARGE:
            anomalie = self._injecter_par_charge(reglage, intensite, duree_s)
        else:
            anomalie = self._injecter_par_conteneur(reglage, intensite, duree_s)

        if run_id:
            ecrire_anomalie(self.reglages.dossier_etiquettes, run_id, anomalie)
        return anomalie

    def _subir_conteneur(self, nom: str, duree_s: float) -> float:
        """Attend la fin de l'injection en surveillant le conteneur.

        Attendre l'horloge seule suffirait si rien ne pouvait interrompre une
        injection. Mais `make stop-chaos` peut la tuer, le noyau peut tuer un
        travailleur memoire, et `stress-ng` peut echouer en cours de route : dans
        ces trois cas l'anomalie s'arrete avant l'echeance, et une etiquette qui
        porterait la duree demandee serait fausse. Mesure a la verification de la
        phase C, ou un arret d'urgence a laisse l'injecteur attendre encore deux
        minutes.
        """
        pas = max(self.pas_surveillance_s, 0.1)
        restant = duree_s
        try:
            while restant > 0:
                tranche = min(pas, restant)
                self.attente(tranche)
                restant -= tranche
                if not self._existe(nom):
                    return self.horloge()
        except KeyboardInterrupt:
            fin = self.horloge()
            self._arreter_silencieux(nom)
            return fin
        self._arreter_silencieux(nom)
        return self.horloge()

    def _subir(self, duree_s: float, arret: Callable[[], None]) -> float:
        """Attend la duree demandee puis arrete, et renvoie l'instant de fin reel.

        Une interruption au clavier n'est pas rattrapee pour etre ignoree : elle
        sert a mesurer la duree reellement subie, qui est ce que l'etiquette doit
        contenir, puis l'injection est arretee comme dans le cas normal.
        """
        try:
            self.attente(duree_s)
        except KeyboardInterrupt:
            fin = self.horloge()
            arret()
            return fin
        arret()
        return self.horloge()

    def _injecter_par_conteneur(
        self, reglage: TypeInjection, intensite: float, duree_s: float
    ) -> Anomalie:
        assert reglage.limites is not None  # garanti par Reglages.depuis_config
        nom = self._nom_conteneur(reglage.nom)
        arguments = [
            *self._arguments_communs(nom, reglage.nom, reglage.limites),
            # Aucun montage, aucun privilege, aucun partage de namespace, et pas
            # de reseau : un conteneur de `stress-ng` n'a aucune raison de voir
            # l'hote ni de joindre quoi que ce soit.
            "--network",
            "none",
            # Systeme de fichiers en lecture seule, avec un seul espace
            # inscriptible, borne. stress-ng exige un `temp-path` accessible en
            # ecriture, et un tmpfs de 64 Mio le lui donne sans qu'une injection
            # puisse remplir le disque du serveur (§4.3).
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m,mode=1777",
            self.reglages.image_stress,
            *self._commande_stress(reglage.nom, intensite, duree_s),
        ]
        debut = self.horloge()
        self.docker.executer(arguments)
        self._verifier_demarrage(nom)
        fin = self._subir_conteneur(nom, duree_s)
        return Anomalie(type=reglage.nom, intensite=intensite, debut_ts=debut, fin_ts=fin)

    def _commande_stress(self, type_: str, intensite: float, duree_s: float) -> list[str]:
        # `--timeout` est la seconde securite : si l'injecteur meurt, stress-ng
        # s'arrete tout seul. La premiere est l'arret explicite a la fin.
        duree = f"{int(duree_s)}s"
        if type_ == "pic_cpu":
            return [
                "stress-ng",
                "--cpu",
                str(int(intensite)),
                "--cpu-method",
                "all",
                "--timeout",
                duree,
            ]
        if type_ == "fuite_memoire":
            # `--vm-keep` garde les pages allouees : une fuite monte et ne
            # redescend pas, contrairement a un pic d'allocation.
            return [
                "stress-ng",
                "--vm",
                "1",
                "--vm-bytes",
                f"{int(intensite)}m",
                "--vm-keep",
                "--timeout",
                duree,
            ]
        raise RefusDeSecurite(
            f"Type `{type_}` declare `moyen: conteneur` mais aucune commande "
            "stress-ng ne lui correspond."
        )

    def _injecter_par_appli(
        self, reglage: TypeInjection, intensite: float, duree_s: float
    ) -> Anomalie:
        if self.appli is None:
            raise RefusDeSecurite(
                f"Le type `{reglage.nom}` passe par le plan de controle de l'appli "
                "de test, qui n'est pas joignable."
            )
        if reglage.nom not in ENDPOINTS_APPLI:
            raise RefusDeSecurite(
                f"Type `{reglage.nom}` declare `moyen: appli` mais aucun endpoint "
                "applicatif ne lui correspond (ENDPOINTS_APPLI)."
            )
        chemin, cle = ENDPOINTS_APPLI[reglage.nom]
        debut = self.horloge()
        self.appli.injecter(chemin, {cle: intensite, "duree_s": duree_s})
        fin = self._subir(duree_s, self._reinitialiser_silencieux)
        return Anomalie(type=reglage.nom, intensite=intensite, debut_ts=debut, fin_ts=fin)

    def _injecter_par_charge(
        self, reglage: TypeInjection, intensite: float, duree_s: float
    ) -> Anomalie:
        assert reglage.limites is not None
        nom = self._nom_conteneur(reglage.nom)
        dossier = (self.racine / "infra/locust").resolve()
        arguments = [
            *self._arguments_communs(nom, reglage.nom, reglage.limites),
            # Le seul conteneur d'injection qui a besoin du reseau de la pile :
            # il doit joindre l'appli de test par son nom de service.
            "--network",
            self.reglages.reseau,
            # Le seul montage de tout l'injecteur, en lecture seule, et il ne
            # contient qu'un scenario de charge.
            "--volume",
            f"{dossier.as_posix()}:/mnt/locust:ro",
            self.reglages.image_charge,
            "-f",
            "/mnt/locust/locustfile.py",
            "--headless",
            "--host",
            str(reglage.locust.get("cible", "http://app:8000")),
            "--users",
            str(int(intensite)),
            "--spawn-rate",
            str(int(reglage.locust.get("arrivee_par_s", 10))),
            # Locust s'arrete de lui-meme, comme `stress-ng --timeout`.
            "--run-time",
            f"{int(duree_s)}s",
            "--only-summary",
        ]
        debut = self.horloge()
        self.docker.executer(arguments)
        self._verifier_demarrage(nom)
        fin = self._subir_conteneur(nom, duree_s)
        return Anomalie(type=reglage.nom, intensite=intensite, debut_ts=debut, fin_ts=fin)

    def _arreter_silencieux(self, nom: str) -> None:
        # Dans le cas normal le conteneur s'est deja arrete tout seul
        # (`stress-ng --timeout`, `locust --run-time`) : son absence n'est pas
        # une erreur. On n'avale que cela, et le refus de securite est reemis.
        try:
            self.arreter(nom)
        except ErreurDocker:
            pass
        except RefusDeSecurite:
            if self._existe(nom):
                raise

    def _existe(self, nom: str) -> bool:
        try:
            self._inspecter(nom)
        except (ErreurDocker, RefusDeSecurite):
            return False
        return True

    def _reinitialiser(self) -> str:
        """Remet le plan de controle a zero, et dit ce qui s'est passe."""
        if self.appli is None:
            return "non joignable : aucune appli de test configuree"
        try:
            self.appli.reinitialiser()
        except Exception as erreur:  # noqa: BLE001 - l'appli peut etre eteinte
            # Une appli eteinte n'a plus d'injection en cours, donc ce n'est pas
            # grave ; mais le dire est le seul moyen de distinguer ce cas d'une
            # URL qui pointe ailleurs, qui est grave.
            return f"inchange : {erreur}"
        return "remis a zero"

    def _reinitialiser_silencieux(self) -> None:
        self._reinitialiser()

    # ----- derives ----------------------------------------------------------

    def enregistrer_derive(self, type_: str, description: str, run_id: str | None = None) -> Derive:
        """Horodate une derive reelle (§4.2 : deploiement, ressources, trafic).

        L'injecteur ne provoque pas la derive — c'est Sidy qui relance l'appli de
        test avec une autre version ou change ses limites. Il en fixe l'instant,
        qui est ce dont l'evaluation a besoin et la seule chose qu'on ne peut pas
        retrouver apres coup.
        """
        derive = Derive(type=type_, description=description, instant_ts=self.horloge())
        if run_id:
            ecrire_derive(self.reglages.dossier_etiquettes, run_id, derive)
        return derive
