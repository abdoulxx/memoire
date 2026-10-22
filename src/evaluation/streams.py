"""Construction des flux rejoues.

Le flux est l'objet partage par toutes les strategies. Il est construit une
seule fois par (source, machine, scenario, seed), puis rejoue tel quel : c'est
ce qui garantit que les strategies sont comparees sur des donnees identiques.

A M0, le seul scenario est `none` : le flux de test est pris tel quel, sans
derive injectee. Les scenarios S1 a S4 s'inserent ici a M1, entre le chargement
et la construction des fenetres.
"""

from __future__ import annotations

from typing import Any

from datasets.smd.loader import SMDMachine
from src.evaluation.replay import ReplayStream
from src.features.windowing import make_windows

NO_DRIFT = "none"


def build_smd_stream(
    machine: SMDMachine,
    features_config: dict[str, Any],
    scenario: str = NO_DRIFT,
) -> ReplayStream:
    """Construit le flux d'une machine SMD.

    La moitie `train` sert a l'entrainement initial ; la moitie `test`, etiquetee,
    constitue le flux evalue. Les features sont calculees avec la meme
    configuration pour les deux, condition pour que le modele voie en service des
    features de meme nature qu'a l'entrainement.
    """
    if scenario != NO_DRIFT:
        raise NotImplementedError(
            f"Scenario '{scenario}' non implemente : l'injection de derive arrive a M1."
        )

    train = make_windows(machine.train, features_config)
    test = make_windows(machine.test, features_config, label=machine.test_label)
    if test.y is None:  # pragma: no cover - make_windows garantit y avec un label
        raise RuntimeError("Le flux de test doit etre etiquete.")

    return ReplayStream(
        source="smd",
        machine=machine.name,
        scenario=scenario,
        train_X=train.X,
        X=test.X,
        y=test.y,
        end_index=test.end_index,
        events=machine.events,
        window_size=int(features_config["window_size"]),
        stride=int(features_config["stride"]),
        sampling_interval_seconds=machine.sampling_interval_seconds,
    )
