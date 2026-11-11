"""Chargement d'une session enregistree (livrable M4, phase D).

Ce module transforme un Parquet et deux fichiers d'etiquettes en un objet de la
meme forme qu'une machine du SMD, pour que le harnais de rejeu de M2 n'ait rien
a savoir de la source du flux.

Tout l'enjeu est l'**alignement temporel**, et il est plus fragile qu'il n'y
parait. Les metriques vivent sur une grille d'instants reguliere ; les anomalies
sont horodatees a la seconde par l'injecteur, qui ne sait rien de cette grille.
Un decalage d'un seul pas fait basculer une fenetre positive en negative, et
tous les chiffres du chapitre 6 sur donnees reelles seraient faux sans qu'aucun
test ne s'en apercoive. D'ou quatre decisions explicites :

1. **Bornes incluses.** Un point est anormal si son instant tombe dans
   `[start_ts, end_ts]` : a `end_ts`, l'injection tournait encore.
2. **Une anomalie sans aucun point de grille est signalee.** Une anomalie plus
   courte que le pas peut passer entre deux instants : elle serait alors
   invisible dans les donnees tout en comptant comme un evenement a detecter,
   ce qui ferait baisser le rappel par evenement de **toutes** les strategies
   sans qu'aucune puisse y faire quoi que ce soit.
3. **Les trous courts sont interpoles, les longs font echouer le chargement.**
   Une valeur inventee dans une fenetre etiquetee anormale est un faux resultat,
   mais refuser toute session ayant un trou d'une minute rendrait la source
   inutilisable. Les tolerances sont dans la configuration, la part reellement
   interpolee est rapportee.
4. **La part d'entrainement doit etre propre.** Si une anomalie etiquetee ou une
   derive tombe dedans, le chargement echoue : un modele initial entraine sur
   une anomalie ne la signalera jamais, et la comparaison des strategies qui en
   decoule ne veut plus rien dire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from datasets.serveur.recorder import (
    COLONNE_INSTANT,
    ErreurSession,
    chemin_donnees,
    chemin_metadonnees,
)
from datasets.serveur.settings import ReglagesServeur
from datasets.smd.loader import AnomalyEvent, events_from_label
from infra.injector.labels import horodatage, lire_anomalies, lire_derives


def instant(texte: str) -> float:
    """Convertit un horodatage RFC 3339 UTC d'un fichier d'etiquettes en secondes."""
    try:
        return datetime.fromisoformat(texte.strip().replace("Z", "+00:00")).timestamp()
    except ValueError as erreur:
        raise ErreurSession(
            f"Horodatage d'etiquette illisible : « {texte} ». "
            "Les fichiers de `data/labels/` portent des instants RFC 3339 UTC."
        ) from erreur


@dataclass(frozen=True)
class SessionServeur:
    """Une session enregistree, prete a etre fenetree.

    Expose la meme chose qu'une `SMDMachine` : une moitie d'entrainement non
    etiquetee, un flux evalue etiquete point par point, et ses evenements.
    """

    run_id: str
    instants: np.ndarray
    train: np.ndarray
    test: np.ndarray
    test_label: np.ndarray
    events: list[AnomalyEvent]
    metric_names: list[str]
    sampling_interval_seconds: int

    # Verite terrain sur la derive reelle, en indices de points de `test`.
    drift_index: int | None
    derives: list[dict[str, str]]

    # Ce que le chargement a du corriger ou signaler : recopie dans le memoire
    # au titre des limites (§11, point 6).
    parts_interpolees: dict[str, float]
    anomalies_sans_point: list[dict[str, str]]
    metadonnees: dict[str, object]

    @property
    def n_features(self) -> int:
        return self.test.shape[1]

    @property
    def anomaly_rate(self) -> float:
        return float(self.test_label.mean())

    def duration_hours(self, n_points: int) -> float:
        return n_points * self.sampling_interval_seconds / 3600.0


def _verifier_grille(instants: np.ndarray, pas: int, run_id: str) -> None:
    """La grille doit etre reguliere et faite de multiples du pas."""
    if len(instants) < 2:
        raise ErreurSession(f"Session « {run_id} » : {len(instants)} point(s), c'est trop peu.")
    ecarts = np.unique(np.diff(instants))
    if len(ecarts) != 1 or int(ecarts[0]) != pas:
        raise ErreurSession(
            f"Session « {run_id} » : la grille n'est pas reguliere au pas de {pas} s "
            f"(ecarts observes : {sorted(int(e) for e in ecarts[:5])}). Elle n'a pas ete "
            "ecrite par `mdrift session-record`, ou le pas de la configuration a change "
            "depuis l'enregistrement."
        )
    reste = np.unique(instants % pas)
    if len(reste) != 1 or int(reste[0]) != 0:
        raise ErreurSession(
            f"Session « {run_id} » : les instants ne sont pas des multiples de {pas} s. "
            "Deux enregistrements de la meme plage ne tomberaient pas sur les memes "
            "instants, donc les etiquettes ne seraient pas reproductibles."
        )


def _combler(
    colonnes: pd.DataFrame, reglages: ReglagesServeur, run_id: str
) -> tuple[np.ndarray, dict[str, float]]:
    """Interpole les trous courts, refuse les longs, et dit ce qui a ete comble."""
    parts: dict[str, float] = {}
    comblees: dict[str, np.ndarray] = {}

    for nom in reglages.noms:
        serie = colonnes[nom].astype(float)
        manquants = serie.isna()
        parts[nom] = float(manquants.mean())

        if manquants.any():
            # Longueur du plus long trou consecutif : c'est elle qui decide, pas
            # le nombre total de trous. Trois trous d'une minute sont anodins,
            # un trou de vingt minutes efface un regime entier.
            groupes = (manquants != manquants.shift()).cumsum()[manquants]
            plus_long = int(groupes.value_counts().max()) if len(groupes) else 0
            if plus_long > reglages.max_trou_consecutif:
                raise ErreurSession(
                    f"Session « {run_id} », metrique « {nom} » : trou de {plus_long} "
                    f"points consecutifs, la limite est {reglages.max_trou_consecutif}. "
                    "Interpoler aussi long inventerait un regime ; la session doit etre "
                    "refaite, ou la plage reduite a une partie sans trou."
                )
            if parts[nom] > reglages.max_part_interpolee:
                raise ErreurSession(
                    f"Session « {run_id} », metrique « {nom} » : {parts[nom]:.1%} de points "
                    f"manquants, la limite est {reglages.max_part_interpolee:.1%}. La "
                    "collecte a ete trop irreguliere pour que la session serve de mesure."
                )
            # `limit_direction="both"` traite aussi les trous de bord, que
            # l'interpolation lineaire ne peut pas encadrer ; la limite par trou
            # a deja ete verifiee juste au-dessus.
            serie = serie.interpolate(
                method="linear", limit=reglages.max_trou_consecutif, limit_direction="both"
            )
            if serie.isna().any():  # pragma: no cover - couvert par les deux gardes
                raise ErreurSession(
                    f"Session « {run_id} », metrique « {nom} » : des points manquants "
                    "subsistent apres interpolation."
                )
        comblees[nom] = serie.to_numpy(dtype=np.float64)

    series = np.column_stack([comblees[nom] for nom in reglages.noms])
    return series, parts


def _etiqueter(
    instants: np.ndarray, reglages: ReglagesServeur, run_id: str
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Marque les points couverts par une anomalie du fichier d'etiquettes."""
    anomalies = lire_anomalies(reglages.dossier_etiquettes, run_id)
    label = np.zeros(len(instants), dtype=np.int64)
    sans_point: list[dict[str, str]] = []

    for ligne in anomalies:
        debut = instant(ligne["start_ts"])
        fin = instant(ligne["end_ts"])
        if fin < debut:
            raise ErreurSession(
                f"Anomalie « {ligne.get('type', '?')} » de la session « {run_id} » : "
                f"fin ({ligne['end_ts']}) anterieure au debut ({ligne['start_ts']})."
            )
        if reglages.bornes_incluses:
            dedans = (instants >= debut) & (instants <= fin)
        else:
            dedans = (instants >= debut) & (instants < fin)
        if not dedans.any():
            sans_point.append(dict(ligne))
            continue
        label[dedans] = 1

    return label, sans_point


def _index_derive(instants: np.ndarray, derives: list[dict[str, str]], decalage: int) -> int | None:
    """Indice du premier point touche par la premiere derive, dans `test`.

    La derive est un changement durable : son instant est converti en indice du
    premier point de la grille qui l'atteint, exactement comme pour le SMD ou la
    verite terrain est aussi un indice de point.
    """
    if not derives:
        return None
    premiere = min(instant(d["ts"]) for d in derives)
    indice = int(np.searchsorted(instants, premiere, side="left"))
    if indice >= len(instants):
        return None
    return max(indice - decalage, 0)


def charger_session(run_id: str, reglages: ReglagesServeur) -> SessionServeur:
    """Charge une session enregistree et ses etiquettes."""
    chemin = chemin_donnees(reglages, run_id)
    if not chemin.exists():
        raise ErreurSession(
            f"Session « {run_id} » introuvable ({chemin}). Les sessions sont enregistrees "
            "par `mdrift session-record` et ne sont pas suivies par git : elles doivent "
            "etre enregistrees sur la machine ou elles sont rejouees."
        )

    colonnes = pd.read_parquet(chemin)
    attendues = [COLONNE_INSTANT, *reglages.noms]
    absentes = [c for c in attendues if c not in colonnes.columns]
    if absentes:
        raise ErreurSession(
            f"Session « {run_id} » : colonnes absentes {absentes}. La session a ete "
            "enregistree avec une autre liste de metriques que celle de "
            "`configs/datasets/serveur.yaml` ; le JSON de metadonnees porte celle qui "
            "l'a produite."
        )

    instants = colonnes[COLONNE_INSTANT].to_numpy(dtype=np.int64)
    _verifier_grille(instants, reglages.pas_s, run_id)
    series, parts = _combler(colonnes, reglages, run_id)
    label, sans_point = _etiqueter(instants, reglages, run_id)
    derives = lire_derives(reglages.dossier_etiquettes, run_id)

    n_train = int(round(reglages.part_entrainement * len(instants)))
    if n_train < 2 or len(instants) - n_train < 2:
        raise ErreurSession(
            f"Session « {run_id} » : {len(instants)} points, trop courte pour etre "
            f"coupee en {reglages.part_entrainement:.0%} d'entrainement."
        )

    if reglages.refuser_anomalie_dans_entrainement:
        _refuser_contamination(instants, label, derives, n_train, run_id)

    test_label = label[n_train:]
    metadonnees: dict[str, object] = {}
    chemin_meta = chemin_metadonnees(reglages, run_id)
    if chemin_meta.exists():
        metadonnees = json.loads(chemin_meta.read_text(encoding="utf-8"))

    return SessionServeur(
        run_id=run_id,
        instants=instants,
        train=series[:n_train],
        test=series[n_train:],
        test_label=test_label,
        events=events_from_label(test_label),
        metric_names=list(reglages.noms),
        sampling_interval_seconds=reglages.pas_s,
        drift_index=_index_derive(instants, derives, n_train),
        derives=derives,
        parts_interpolees=parts,
        anomalies_sans_point=sans_point,
        metadonnees=metadonnees,
    )


def _refuser_contamination(
    instants: np.ndarray,
    label: np.ndarray,
    derives: list[dict[str, str]],
    n_train: int,
    run_id: str,
) -> None:
    """La part d'entrainement doit ne contenir ni anomalie etiquetee ni derive."""
    contaminees = int(label[:n_train].sum())
    if contaminees:
        fin_train = horodatage(float(instants[n_train - 1]))
        raise ErreurSession(
            f"Session « {run_id} » : {contaminees} point(s) etiquete(s) anormaux dans la "
            f"part d'entrainement, qui s'arrete a {fin_train}. Le modele initial doit "
            "voir du comportement normal uniquement : laisser la session commencer par "
            "une periode calme, ou reduire `part_entrainement`."
        )
    limite = float(instants[n_train - 1])
    tot = [d for d in derives if instant(d["ts"]) <= limite]
    if tot:
        raise ErreurSession(
            f"Session « {run_id} » : derive(s) {[d.get('type') for d in tot]} dans la part "
            "d'entrainement. Le modele initial serait entraine apres la derive qu'il est "
            "cense subir, et la question du memoire n'aurait plus d'objet."
        )
