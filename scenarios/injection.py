"""Injection de derive de concept dans un flux de metriques reelles.

Pourquoi injecter plutot que chercher une derive reelle : aucun jeu de donnees
public n'etiquette la derive de concept sur une infrastructure applicative. Les
jeux de reference etiquettent les *anomalies*, pas l'instant ou le comportement
normal change. Or trois metriques du cadrage (§7) exigent de connaitre cet
instant : le delai de detection de la derive, le F1 avant / apres, et le temps
de recuperation. L'injection parametree est donc la seule facon d'obtenir une
verite terrain sur la derive. La validite externe est assuree ailleurs : par les
13 machines du SMD qui derivent naturellement (OBS03, scenario `none`) et par
les derives reelles horodatees du serveur personnel (M6).

Les quatre scenarios partagent **une seule** transformation et ne different que
par leur profil temporel a(t) dans [0, 1] :

    x'(t) = clip( x(t) + a(t) * k * sigma_train
                       + a(t) * (echelle - 1) * (x(t) - mu_train) , bas, haut )

    marche    (S1, soudaine)     : 0 avant t_d, 1 ensuite
    rampe     (S2, incrementale) : montee lineaire de 0 a 1 sur D, puis 1
    graduel   (S3, graduelle)    : 0 ou 1 tire point par point, P(1) montant sur D
    recurrent (S4, recurrente)   : creneaux periodiques entre 0 et 1

Deux invariants, verifies par les tests :

1. **L'injection porte sur les 38 metriques brutes**, avant le fenetrage et le
   calcul des features. Injecter dans les 152 features serait incoherent : la
   derive ne se propagerait pas dans les agregats glissants et le flux ne
   ressemblerait plus a ce qu'un serveur produit.
2. **Les etiquettes d'anomalie ne sont jamais modifiees**, ni avant ni apres
   t_d. Une derive change le comportement normal, pas la verite terrain des
   anomalies.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config

PROFILS = ("marche", "rampe", "graduel", "recurrent")

# Les quatre scenarios du cadrage (§5), dans l'ordre S1 a S4. Le nom est celui
# du fichier `configs/scenarios/<nom>.yaml`.
SCENARIOS = ("s1_soudaine", "s2_incrementale", "s3_graduelle", "s4_recurrente")

# Ordre de consommation du generateur aleatoire, fixe une fois pour toutes :
# l'instant de derive, puis les metriques touchees, puis les tirages point par
# point de S3. Cet ordre fait partie du contrat de reproductibilite (meme seed =
# meme flux), il ne doit pas changer sans invalider les resultats deja produits.
_FLUX_ALEATOIRE = ("onset", "metriques", "profil")


@dataclass(frozen=True)
class DriftScenario:
    """Un scenario de derive : le profil temporel et la transformation partagee."""

    name: str
    profil: str
    duree_fraction: float
    periode_minutes: int
    rapport_cyclique: float
    level_shift_k: float
    scale_factor: float
    metric_fraction: float
    exclure_constantes: bool
    clip_bas: float
    clip_haut: float
    part_ecretee_max: float
    onset_min: float
    onset_max: float

    @classmethod
    def from_config(
        cls,
        scenario_config: dict[str, Any],
        transformation_config: dict[str, Any] | None = None,
    ) -> DriftScenario:
        """Construit un scenario depuis son YAML et celui de la transformation.

        Le scenario designe sa transformation par la cle `transformation`, ce qui
        garantit que les quatre scenarios partagent la meme amplitude.
        """
        if transformation_config is None:
            reference = scenario_config.get("transformation")
            if reference is None:
                raise ValueError(
                    f"Le scenario '{scenario_config.get('name')}' ne designe aucune "
                    "transformation (cle `transformation`)."
                )
            transformation_config = load_config(reference)

        profil = scenario_config["profil"]
        kind = str(profil["type"])
        if kind not in PROFILS:
            raise ValueError(f"Profil '{kind}' inconnu. Disponibles : {list(PROFILS)}")

        metriques = transformation_config["metriques"]
        ecretage = transformation_config["ecretage"]
        onset = transformation_config["onset"]

        scenario = cls(
            name=str(scenario_config["name"]),
            profil=kind,
            duree_fraction=float(profil.get("duree_fraction", 0.0)),
            periode_minutes=int(profil.get("periode_minutes", 0)),
            rapport_cyclique=float(profil.get("rapport_cyclique", 0.5)),
            level_shift_k=float(transformation_config["level_shift_k"]),
            scale_factor=float(transformation_config["scale_factor"]),
            metric_fraction=float(metriques["fraction"]),
            exclure_constantes=bool(metriques["exclure_constantes"]),
            clip_bas=float(ecretage["bas"]),
            clip_haut=float(ecretage["haut"]),
            part_ecretee_max=float(ecretage["part_max_toleree"]),
            onset_min=float(onset["position_min"]),
            onset_max=float(onset["position_max"]),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        """Verifie les bornes que le cadrage impose (D15)."""
        if not 0.0 <= self.onset_min <= self.onset_max <= 1.0:
            raise ValueError(f"Plage de t_d invalide : [{self.onset_min}, {self.onset_max}].")
        if self.profil in ("rampe", "graduel") and not 0.0 < self.duree_fraction <= 0.25:
            raise ValueError(
                f"{self.name} : duree_fraction = {self.duree_fraction}, attendu dans "
                "]0, 0.25]. D15 borne la transition a 25 % du flux pour qu'il reste un "
                "segment post-derive stabilise."
            )
        if self.profil == "recurrent":
            if self.periode_minutes < 2:
                raise ValueError(f"{self.name} : periode_minutes = {self.periode_minutes}.")
            if not 0.0 < self.rapport_cyclique < 1.0:
                raise ValueError(
                    f"{self.name} : rapport_cyclique = {self.rapport_cyclique}, attendu "
                    "dans ]0, 1[ (0 ou 1 supprimerait l'alternance)."
                )
        if not 0.0 < self.metric_fraction <= 1.0:
            raise ValueError(f"{self.name} : metriques.fraction = {self.metric_fraction}.")
        if self.level_shift_k < 0.0:
            raise ValueError(
                f"{self.name} : level_shift_k = {self.level_shift_k}. Le decalage doit etre "
                "positif (D15 : 24 a 31 % des valeurs valent exactement 0, un decalage "
                "negatif serait absorbe par le plancher)."
            )
        if self.clip_bas >= self.clip_haut:
            raise ValueError(f"{self.name} : ecretage [{self.clip_bas}, {self.clip_haut}].")


@dataclass(frozen=True)
class InjectedStream:
    """Un flux de test apres injection, avec sa verite terrain sur la derive."""

    scenario: str
    seed: int
    series: np.ndarray
    label: np.ndarray
    alpha: np.ndarray
    drift_index: int
    drift_end_index: int
    metriques: tuple[int, ...]
    part_ecretee: float
    n_cellules_touchees: int

    def __len__(self) -> int:
        return len(self.series)

    @property
    def duree_transition(self) -> int:
        """Nombre de points entre le debut de la derive et le regime etabli."""
        return self.drift_end_index - self.drift_index

    def metadata(self) -> dict[str, Any]:
        """Resume journalisable, recopie a cote des resultats de l'execution."""
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "n_points": len(self.series),
            "drift_index": self.drift_index,
            "drift_end_index": self.drift_end_index,
            "duree_transition_points": self.duree_transition,
            "n_metriques_touchees": len(self.metriques),
            "metriques_touchees": list(self.metriques),
            "part_ecretee": round(self.part_ecretee, 6),
            "n_cellules_touchees": self.n_cellules_touchees,
        }


def load_scenario(path: str | Path) -> DriftScenario:
    """Charge un scenario depuis `configs/scenarios/<nom>.yaml`."""
    return DriftScenario.from_config(load_config(path))


def _graine_flux(stream_key: str) -> int:
    """Entier stable derive du nom du flux (la machine, en pratique).

    `zlib.crc32` est utilise plutot que `hash()`, qui est randomise d'un
    processus Python a l'autre : la reproductibilite ne survivrait pas a un
    redemarrage.
    """
    return int(zlib.crc32(stream_key.encode("utf-8")))


def _rng(seed: int, flux: str, stream_key: str = "") -> np.random.Generator:
    """Un generateur par usage et par flux, derive de la seed de l'execution.

    Des generateurs independants evitent qu'un changement de longueur de flux
    (donc du nombre de tirages consommes par le profil) ne deplace l'instant de
    derive ou le choix des metriques.

    `stream_key` entre dans la graine pour que l'instant de derive et le choix
    des metriques varient **d'une machine a l'autre**. Sans lui, toutes les
    machines derivaient a la meme position relative pour une graine donnee
    (0,418 du flux pour la graine 42) : les 5 graines n'exploraient que 5
    positions, et les machines d'une meme graine partageaient exactement la
    meme configuration de derive. L'appariement entre strategies est preserve :
    le flux reste une fonction de (machine, scenario, graine).
    """
    return np.random.default_rng([seed, _FLUX_ALEATOIRE.index(flux), _graine_flux(stream_key)])


def drift_onset(n_points: int, scenario: DriftScenario, seed: int, stream_key: str = "") -> int:
    """Tire t_d uniformement dans la plage de la configuration (D15)."""
    bas = int(np.floor(scenario.onset_min * n_points))
    haut = int(np.ceil(scenario.onset_max * n_points))
    if haut <= bas:
        return bas
    return int(_rng(seed, "onset", stream_key).integers(bas, haut))


def injectable_metrics(train: np.ndarray, scenario: DriftScenario) -> np.ndarray:
    """Indices des metriques injectables : non constantes sur `train` si demande."""
    if not scenario.exclure_constantes:
        return np.arange(train.shape[1])
    sigma = train.std(axis=0)
    return np.flatnonzero(sigma > 0.0)


def select_metrics(
    train: np.ndarray, scenario: DriftScenario, seed: int, stream_key: str = ""
) -> np.ndarray:
    """Tire le sous-ensemble de metriques touchees, par ordre croissant d'indice."""
    candidates = injectable_metrics(train, scenario)
    if len(candidates) == 0:
        raise ValueError("Aucune metrique injectable : toutes sont constantes sur la moitie train.")
    n = max(1, int(round(scenario.metric_fraction * len(candidates))))
    chosen = _rng(seed, "metriques", stream_key).choice(candidates, size=n, replace=False)
    return np.sort(chosen)


def alpha_profile(
    n_points: int,
    drift_index: int,
    scenario: DriftScenario,
    seed: int,
    sampling_interval_seconds: int = 60,
    stream_key: str = "",
) -> tuple[np.ndarray, int]:
    """Construit le profil temporel a(t) et l'indice de fin de transition.

    Retourne `(alpha, drift_end_index)`. `drift_end_index` est l'instant a partir
    duquel le nouveau regime est etabli : t_d pour une marche, la fin de la rampe
    pour S2 et S3, et la fin du flux pour S4 dont le regime alterne jusqu'au bout.
    """
    alpha = np.zeros(n_points, dtype=np.float64)
    apres = np.arange(drift_index, n_points)
    if len(apres) == 0:
        return alpha, drift_index

    if scenario.profil == "marche":
        alpha[drift_index:] = 1.0
        return alpha, drift_index

    if scenario.profil in ("rampe", "graduel"):
        duree = max(1, int(round(scenario.duree_fraction * n_points)))
        montee = np.clip((apres - drift_index + 1) / duree, 0.0, 1.0)
        if scenario.profil == "rampe":
            alpha[drift_index:] = montee
        else:
            tirages = _rng(seed, "profil", stream_key).random(len(apres))
            alpha[drift_index:] = (tirages < montee).astype(np.float64)
        return alpha, min(n_points - 1, drift_index + duree)

    # recurrent : creneaux periodiques, le regime d'origine revient.
    periode = max(2, int(round(scenario.periode_minutes * 60 / sampling_interval_seconds)))
    haut = max(1, int(round(scenario.rapport_cyclique * periode)))
    position = (apres - drift_index) % periode
    alpha[drift_index:] = (position < haut).astype(np.float64)
    return alpha, n_points - 1


def inject(
    train: np.ndarray,
    test: np.ndarray,
    label: np.ndarray,
    scenario: DriftScenario,
    seed: int,
    sampling_interval_seconds: int = 60,
    stream_key: str = "",
) -> InjectedStream:
    """Injecte la derive dans la moitie `test` d'une machine.

    `train` ne sert qu'a estimer mu et sigma du comportement normal : il n'est pas
    transforme, puisque le modele initial est entraine avant la derive.

    `stream_key` identifie le flux (le nom de la machine) et decorrele les
    tirages d'une machine a l'autre : voir `_rng`.
    """
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{train.shape[1]} metriques en train, {test.shape[1]} en test.")
    if len(test) != len(label):
        raise ValueError(f"{len(test)} points de test pour {len(label)} etiquettes.")

    n_points = len(test)
    drift_index = drift_onset(n_points, scenario, seed, stream_key)
    metriques = select_metrics(train, scenario, seed, stream_key)
    alpha, drift_end_index = alpha_profile(
        n_points, drift_index, scenario, seed, sampling_interval_seconds, stream_key
    )

    mu = train[:, metriques].mean(axis=0)
    sigma = train[:, metriques].std(axis=0)

    bloc = test[:, metriques]
    poids = alpha[:, None]
    transforme = (
        bloc
        + poids * scenario.level_shift_k * sigma
        + poids * (scenario.scale_factor - 1.0) * (bloc - mu)
    )
    ecrete = np.clip(transforme, scenario.clip_bas, scenario.clip_haut)

    # Part ecretee mesuree sur les seules cellules effectivement touchees
    # (metrique tiree et a(t) > 0) : c'est le denominateur qui a un sens, le
    # reste du flux est inchange par construction.
    touchees = poids > 0.0
    n_touchees = int(np.broadcast_to(touchees, bloc.shape).sum())
    if n_touchees == 0:
        part_ecretee = 0.0
    else:
        modifiees = (ecrete != transforme) & touchees
        part_ecretee = float(modifiees.sum() / n_touchees)

    if part_ecretee > scenario.part_ecretee_max:
        raise ValueError(
            f"{scenario.name} : {part_ecretee:.1%} des valeurs touchees sont ecretees, "
            f"au-dela de la part toleree ({scenario.part_ecretee_max:.0%}). La derive "
            "sature les metriques et deviendrait triviale a detecter : baisser "
            "level_shift_k ou scale_factor."
        )

    series = test.copy()
    series[:, metriques] = ecrete

    return InjectedStream(
        scenario=scenario.name,
        seed=seed,
        series=series,
        label=label.copy(),  # jamais modifie, copie pour couper tout partage
        alpha=alpha,
        drift_index=drift_index,
        drift_end_index=drift_end_index,
        metriques=tuple(int(m) for m in metriques),
        part_ecretee=part_ecretee,
        n_cellules_touchees=n_touchees,
    )
