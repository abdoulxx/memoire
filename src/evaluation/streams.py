"""Construction des flux rejoues.

Le flux est l'objet partage par toutes les strategies. Il est construit une
seule fois par (source, machine, scenario, seed), puis rejoue tel quel : c'est
ce qui garantit que les strategies sont comparees sur des donnees identiques.

Ordre des operations, qui n'est pas negociable : la derive est injectee dans les
**metriques brutes**, puis les fenetres et les features sont calculees sur le
flux deja derive. L'inverse (injecter dans les 152 features) ne propagerait pas
la derive dans les agregats glissants et produirait un flux qu'aucun serveur ne
peut produire.

La moitie `train` n'est jamais transformee : le modele initial est entraine sur
le comportement normal d'avant la derive, c'est tout le sujet du memoire.
"""

from __future__ import annotations

from typing import Any

from datasets.serveur.loader import SessionServeur
from datasets.smd.loader import SMDMachine, events_from_label
from scenarios.injection import DriftScenario, inject, load_scenario
from src.evaluation.replay import ReplayStream
from src.features.windowing import make_windows

NO_DRIFT = "none"

# Les sessions du serveur ne portent pas S1-S4 : la derive y est reelle. Le nom
# est explicite dans `results/summary.csv` pour que les deux sources ne se
# melangent pas dans les tableaux du chapitre 6.
SCENARIO_REEL = "reel"


def resolve_scenario(scenario: str | DriftScenario) -> DriftScenario | None:
    """Convertit un nom de scenario en objet. `none` donne None (aucune injection)."""
    if isinstance(scenario, DriftScenario):
        return scenario
    if scenario == NO_DRIFT:
        return None
    return load_scenario(f"scenarios/{scenario}.yaml")


def build_smd_stream(
    machine: SMDMachine,
    features_config: dict[str, Any],
    scenario: str | DriftScenario = NO_DRIFT,
    seed: int = 42,
) -> ReplayStream:
    """Construit le flux d'une machine SMD, avec ou sans derive injectee.

    La moitie `train` sert a l'entrainement initial ; la moitie `test`, etiquetee,
    constitue le flux evalue. Les features sont calculees avec la meme
    configuration pour les deux, condition pour que le modele voie en service des
    features de meme nature qu'a l'entrainement.
    """
    drift = resolve_scenario(scenario)
    scenario_name = NO_DRIFT if drift is None else drift.name

    series = machine.test
    label = machine.test_label
    events = machine.events
    drift_index: int | None = None
    drift_end_index: int | None = None
    drift_metadata: dict[str, Any] | None = None

    if drift is not None:
        injected = inject(
            train=machine.train,
            test=machine.test,
            label=machine.test_label,
            scenario=drift,
            seed=seed,
            sampling_interval_seconds=machine.sampling_interval_seconds,
            stream_key=machine.name,
        )
        series = injected.series
        label = injected.label
        # Les etiquettes etant inchangees, les evenements sont les memes ; ils
        # sont recalcules plutot que repris pour que l'invariant soit verifie
        # par construction et non suppose.
        events = events_from_label(label)
        drift_index = injected.drift_index
        drift_end_index = injected.drift_end_index
        drift_metadata = {
            **injected.metadata(),
            "profil": drift.profil,
            "level_shift_k": drift.level_shift_k,
            "scale_factor": drift.scale_factor,
        }

    train = make_windows(machine.train, features_config)
    test = make_windows(series, features_config, label=label)
    if test.y is None:  # pragma: no cover - make_windows garantit y avec un label
        raise RuntimeError("Le flux de test doit etre etiquete.")

    return ReplayStream(
        source="smd",
        machine=machine.name,
        scenario=scenario_name,
        train_X=train.X,
        X=test.X,
        y=test.y,
        end_index=test.end_index,
        events=events,
        window_size=int(features_config["window_size"]),
        stride=int(features_config["stride"]),
        sampling_interval_seconds=machine.sampling_interval_seconds,
        drift_index=drift_index,
        drift_end_index=drift_end_index,
        drift_metadata=drift_metadata,
    )


def build_server_stream(
    session: SessionServeur,
    features_config: dict[str, Any],
) -> ReplayStream:
    """Construit le flux d'une session enregistree sur le serveur.

    Aucun scenario n'est resolu ici, et c'est le point de la source `serveur` :
    la derive n'est pas injectee, elle a **eu lieu** (deploiement, changement de
    ressources, montee de trafic) et son instant est lu dans
    `data/labels/<run_id>_drift.csv`. Le harnais de rejeu recoit donc le meme
    objet que pour le SMD, avec une verite terrain de derive qui vient du terrain
    et non d'une transformation.

    Le nom du scenario vaut `reel` quand une derive est horodatee et `none`
    sinon : `results/summary.csv` garde ainsi une colonne `scenario` lisible, ou
    les lignes du serveur ne se confondent pas avec S1-S4.
    """
    train = make_windows(session.train, features_config)
    test = make_windows(session.test, features_config, label=session.test_label)
    if test.y is None:  # pragma: no cover - make_windows garantit y avec un label
        raise RuntimeError("Le flux de test doit etre etiquete.")

    drift_index = session.drift_index
    drift_metadata: dict[str, Any] | None = None
    if drift_index is not None:
        drift_metadata = {
            "origine": "reelle",
            "n_derives": len(session.derives),
            "derives": session.derives,
        }

    return ReplayStream(
        source="serveur",
        # La session tient la place de la machine : une session est a la source
        # `serveur` ce qu'une machine est au SMD, et l'arborescence de
        # `results/runs/` n'a pas a changer de forme.
        machine=session.run_id,
        scenario=SCENARIO_REEL if drift_index is not None else NO_DRIFT,
        train_X=train.X,
        X=test.X,
        y=test.y,
        end_index=test.end_index,
        events=session.events,
        window_size=int(features_config["window_size"]),
        stride=int(features_config["stride"]),
        sampling_interval_seconds=session.sampling_interval_seconds,
        drift_index=drift_index,
        # Une derive reelle n'a pas de fin : un deploiement ne se replie pas.
        # `drift_end_index` reste None, ce que le harnais interprete deja comme
        # « le regime derive court jusqu'a la fin du flux ».
        drift_end_index=None,
        drift_metadata=drift_metadata,
    )
