"""Acces a Docker par le client en ligne de commande (livrable M4, phase C).

Pourquoi la ligne de commande et non `docker-py` : l'injecteur a besoin de six
verbes (`run`, `ps`, `inspect`, `stop`, `rm`, `logs`) et le client est deja
installe partout ou la pile tourne. Une dependance Python de plus pour cela
serait de la sur-construction, et le §3 du cadrage demande que chaque brique se
justifie en une phrase.

L'interet secondaire est le test : tout passe par un objet qui expose une seule
methode, donc les garde-fous de securite se testent sans Docker et sans jamais
creer de conteneur.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Protocol


class ErreurDocker(RuntimeError):
    """Docker a refuse la commande, ou n'est pas joignable."""


class ClientDocker(Protocol):
    """Ce dont l'injecteur a besoin de Docker, et rien de plus."""

    def executer(self, arguments: list[str], attendre: bool = True) -> str:
        """Lance `docker <arguments>` et renvoie la sortie standard."""


@dataclass
class DockerCli:
    """Client reel : un `subprocess` autour du binaire `docker`."""

    binaire: str = "docker"
    delai_s: float = 30.0

    def __post_init__(self) -> None:
        if shutil.which(self.binaire) is None:
            raise ErreurDocker(
                f"`{self.binaire}` est introuvable. L'injecteur tourne sur l'hote, "
                "pas dans un conteneur."
            )

    def executer(self, arguments: list[str], attendre: bool = True) -> str:
        # `attendre=False` sert aux injections qui durent : le conteneur tourne,
        # l'injecteur garde la main pour pouvoir l'arreter proprement.
        if not attendre:
            subprocess.Popen(  # noqa: S603
                [self.binaire, *arguments],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ""
        acheve = subprocess.run(  # noqa: S603
            [self.binaire, *arguments],
            capture_output=True,
            text=True,
            timeout=self.delai_s,
            check=False,
        )
        if acheve.returncode != 0:
            raise ErreurDocker(
                f"docker {' '.join(arguments)} a echoue ({acheve.returncode}) : "
                f"{acheve.stderr.strip() or acheve.stdout.strip()}"
            )
        return acheve.stdout


@dataclass
class DockerEnregistre:
    """Client factice pour les tests : enregistre les appels, ne lance rien.

    Il tient aussi un inventaire de conteneurs pretendus presents, ce qui permet
    de verifier que l'injecteur refuse bien de toucher un conteneur qui n'est pas
    le sien — la propriete la plus importante du §4.3, et celle qu'on ne peut pas
    tester sur une vraie machine sans prendre le risque qu'on cherche a ecarter.
    """

    inventaire: dict[str, dict[str, str]] = field(default_factory=dict)
    appels: list[list[str]] = field(default_factory=list)
    # Conteneurs qui demarrent puis s'arretent aussitot. C'est arrive pour de
    # vrai a la verification de la phase C (stress-ng sans `temp-path`
    # inscriptible), et l'injecteur doit le detecter plutot que d'etiqueter une
    # anomalie qui n'a pas eu lieu.
    mort_nes: set[str] = field(default_factory=set)
    journaux: dict[str, str] = field(default_factory=dict)

    def executer(self, arguments: list[str], attendre: bool = True) -> str:
        self.appels.append(list(arguments))
        if arguments[0] == "ps":
            return self._ps(arguments)
        if arguments[0] == "inspect":
            return self._inspect(arguments)
        if arguments[0] == "run":
            nom = arguments[arguments.index("--name") + 1]
            etiquettes = {}
            for i, a in enumerate(arguments):
                if a == "--label" and "=" in arguments[i + 1]:
                    cle, valeur = arguments[i + 1].split("=", 1)
                    etiquettes[cle] = valeur
            self.inventaire[nom] = etiquettes
            return nom
        if arguments[0] == "logs":
            return self.journaux.get(arguments[-1], "")
        if arguments[0] in {"stop", "rm", "kill"}:
            for nom in arguments[1:]:
                self.inventaire.pop(nom, None)
            return ""
        return ""

    def _ps(self, arguments: list[str]) -> str:
        exigees = {}
        for i, a in enumerate(arguments):
            if a == "--filter" and arguments[i + 1].startswith("label="):
                corps = arguments[i + 1].removeprefix("label=")
                cle, _, valeur = corps.partition("=")
                exigees[cle] = valeur
        noms = [
            nom
            for nom, etiquettes in self.inventaire.items()
            if all(etiquettes.get(c) == v for c, v in exigees.items())
        ]
        return "\n".join(sorted(noms))

    def _inspect(self, arguments: list[str]) -> str:
        nom = arguments[-1]
        if nom not in self.inventaire:
            raise ErreurDocker(f"No such object: {nom}")
        return json.dumps(
            [
                {
                    "Name": f"/{nom}",
                    "Config": {"Labels": self.inventaire[nom]},
                    "State": {"Running": nom not in self.mort_nes, "ExitCode": 1},
                }
            ]
        )
