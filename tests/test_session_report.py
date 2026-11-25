"""Rejeu et rapport d'une session serveur enregistree (jalon M6).

`mdrift session-replay` (`session_campaign.py`) et `mdrift session-report`
(`session_report.py`) ne recalculent rien de special : ils assemblent des
briques deja testees ailleurs (`replay`, `event_metrics`, `pr_auc_from_traces`).
Ce qui est propre a ces deux modules, et donc ce qui est teste ici, c'est :

- que le flux d'une session se construit et se rejoue comme celui du SMD
  (memes `run_id`, meme reprise) ;
- que la position de fenetre loguee par le harnais se retraduit dans
  l'horodatage REEL de la session (et non un indice), pour P et A1 ;
- que deux etiquettes brutes qui se chevauchent (le doublon `pic_cpu` du
  23/11) deviennent UN SEUL evenement, pas deux, dans le tableau par anomalie.

`proposed` et `ablation` ne sont pas rejouees ici : leur fenetre de confirmation
(960 fenetres) et la fenetre recente du moniteur (500) rendent tout jeu de
test assez grand pour les declencher bien trop lent pour des tests unitaires.
L'extraction d'horodatage pour ces deux bras est donc verifiee sur des artefacts
(`traces.npz` / `events.jsonl`) ecrits a la main, au meme format que ceux que
`session-replay` aurait produits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from datasets.serveur.recorder import COLONNE_INSTANT, chemin_donnees
from datasets.serveur.settings import MARQUE_RATE, ReglagesServeur
from infra.injector.labels import Anomalie, Derive, ecrire_anomalie, ecrire_derive, horodatage
from src.evaluation import replay as replay_module
from src.evaluation.campaign import make_run_id_for
from src.evaluation.session_campaign import load_session_stream, run_session_campaign
from src.evaluation.session_report import (
    anomaly_table,
    build_session_report,
    strategy_summary,
    window_counts,
    write_session_report,
)
from src.evaluation.streams import NO_DRIFT

T0 = 1_793_000_040
PAS = 60


def _config(tmp_path: Path, **surcharges: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "serveur-test",
        "prometheus": {"url": "http://prom:9090", "step_seconds": PAS, "rate_interval": "1m"},
        "metriques": [
            {"nom": "cpu", "expr": f"rate(cpu{MARQUE_RATE})", "obligatoire": True},
            {"nom": "latence", "expr": "latence", "obligatoire": False},
        ],
        "trous": {"max_points_consecutifs": 3, "max_part_interpolee": 0.2},
        "decoupage": {"part_entrainement": 0.4, "refuser_anomalie_dans_entrainement": True},
        "etiquetage": {"bornes_incluses": True, "signaler_anomalies_sans_point": True},
        # Chemins absolus : `ReglagesServeur.depuis_config` les joint a sa
        # racine par defaut (la racine du depot), et un chemin absolu a droite
        # d'une jointure `Path` l'emporte toujours sur la gauche.
        "chemins": {
            "sessions": str(tmp_path / "sessions"),
            "etiquettes": str(tmp_path / "labels"),
        },
    }
    for cle, valeur in surcharges.items():
        if isinstance(valeur, dict) and isinstance(base.get(cle), dict):
            base[cle] = {**base[cle], **valeur}
        else:
            base[cle] = valeur
    return base


def _dataset_config_path(tmp_path: Path, **surcharges: Any) -> str:
    """`load_session_stream`/`run_session_campaign` lisent un chemin, pas un dict."""
    chemin = tmp_path / "dataset.yaml"
    chemin.write_text(yaml.safe_dump(_config(tmp_path, **surcharges)), encoding="utf-8")
    return str(chemin)


def _reglages(tmp_path: Path, **surcharges: Any) -> ReglagesServeur:
    return ReglagesServeur.depuis_config(_config(tmp_path, **surcharges))


def _ecrire_session(
    reglages: ReglagesServeur,
    run_id: str,
    n: int = 200,
    cpu: np.ndarray | None = None,
    latence: np.ndarray | None = None,
) -> np.ndarray:
    instants = np.arange(T0, T0 + n * PAS, PAS, dtype=np.int64)
    reglages.dossier_sessions.mkdir(parents=True, exist_ok=True)
    tableau = {
        COLONNE_INSTANT: instants,
        "cpu": np.arange(n, dtype=float) if cpu is None else cpu,
        "latence": np.full(n, 0.1) if latence is None else latence,
    }
    pd.DataFrame(tableau).to_parquet(chemin_donnees(reglages, run_id), index=False)
    return instants


def _anomalie(reglages: ReglagesServeur, run_id: str, debut: float, fin: float) -> None:
    ecrire_anomalie(
        reglages.dossier_etiquettes,
        run_id,
        Anomalie(type="pic_cpu", intensite=2, debut_ts=debut, fin_ts=fin),
    )


# --------------------------------------------------------------------------- #
# session_campaign.py
# --------------------------------------------------------------------------- #


def test_load_session_stream_construit_le_meme_flux_que_build_server_stream(
    tmp_path: Path,
) -> None:
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=200)
    _anomalie(reglages, "s1", float(instants[150]), float(instants[152]))

    session, stream = load_session_stream("s1", dataset_config=_dataset_config_path(tmp_path))

    assert session.run_id == "s1"
    assert stream.source == "serveur"
    assert stream.machine == "s1"
    assert stream.scenario == NO_DRIFT  # pas de derive ecrite ici
    assert len(stream) == len(stream.y) == len(stream.end_index)


def test_run_session_campaign_rejoue_chaque_strategie_demandee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `run_one` ecrit toujours sous le `RESULTS_DIR` global du module replay,
    # pas sous `summary_path` : sans ce monkeypatch, le test ecrirait dans les
    # vrais `results/runs/` du depot.
    monkeypatch.setattr(replay_module, "RESULTS_DIR", tmp_path / "results")
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=200)
    _anomalie(reglages, "s1", float(instants[150]), float(instants[152]))
    summary_path = tmp_path / "results" / "summary.csv"

    table = run_session_campaign(
        "s1",
        strategies=["static", "periodic"],
        dataset_config=_dataset_config_path(tmp_path),
        summary_path=summary_path,
        seed=42,
    )

    assert sorted(table["strategy"]) == ["periodic", "static"]
    assert summary_path.is_file()
    synthese = pd.read_csv(summary_path)
    attendus = {
        make_run_id_for("s1", NO_DRIFT, "static", 42, "serveur"),
        make_run_id_for("s1", NO_DRIFT, "periodic", 42, "serveur"),
    }
    assert set(synthese["run_id"]) == attendus
    for directory in (
        replay_module.run_directory(
            load_session_stream("s1", dataset_config=_dataset_config_path(tmp_path))[1],
            strategie,
            42,
        )
        for strategie in ("static", "periodic")
    ):
        assert (directory / "traces.npz").is_file()
        assert (directory / "events.jsonl").is_file()


def test_run_session_campaign_resume_saute_les_rejeux_deja_presents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_module, "RESULTS_DIR", tmp_path / "results")
    _ecrire_session(_reglages(tmp_path), "s1", n=200)
    dataset_config = _dataset_config_path(tmp_path)
    summary_path = tmp_path / "results" / "summary.csv"

    run_session_campaign(
        "s1",
        strategies=["static"],
        dataset_config=dataset_config,
        summary_path=summary_path,
        seed=42,
    )
    second = run_session_campaign(
        "s1",
        strategies=["static"],
        dataset_config=dataset_config,
        summary_path=summary_path,
        seed=42,
    )

    assert second.empty
    assert len(pd.read_csv(summary_path)) == 1


# --------------------------------------------------------------------------- #
# session_report.py : fenetres, doublon pic_cpu
# --------------------------------------------------------------------------- #


def test_window_counts_correspond_a_la_longueur_du_flux(tmp_path: Path) -> None:
    reglages = _reglages(tmp_path)
    _ecrire_session(reglages, "s1", n=200)
    _, stream = load_session_stream("s1", dataset_config=_dataset_config_path(tmp_path))

    compte = window_counts(stream)

    assert compte["n_windows_train"] == len(stream.train_X)
    assert compte["n_windows_eval"] == len(stream)
    assert compte["n_windows_total"] == len(stream.train_X) + len(stream)


def test_deux_etiquettes_pic_cpu_qui_se_chevauchent_forment_un_seul_evenement(
    tmp_path: Path,
) -> None:
    # Le doublon pic_cpu du 23/11 (M6) : deux lignes d'etiquette chevauchantes.
    # `events_from_label` (deja teste dans test_serveur_session.py) les fusionne
    # en un seul evenement ; ce test verifie que le rapport le dit explicitement.
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=200)
    _anomalie(reglages, "s1", float(instants[150]), float(instants[153]))
    _anomalie(reglages, "s1", float(instants[152]), float(instants[155]))

    rapport = build_session_report(
        "s1", dataset_config=_dataset_config_path(tmp_path), strategies=()
    )

    assert "UN" in rapport["doublon_pic_cpu"]
    _, stream = load_session_stream("s1", dataset_config=_dataset_config_path(tmp_path))
    assert len(stream.events) == 1

    table = anomaly_table(stream, *_session_et_reglages(tmp_path, "s1"), seed=42)
    assert table.loc[0, "n_etiquettes_brutes"] == 2


def _session_et_reglages(tmp_path: Path, run_id: str):
    reglages = _reglages(tmp_path)
    session, _ = load_session_stream(run_id, dataset_config=_dataset_config_path(tmp_path))
    n_train_points = len(session.train)
    return session, reglages, n_train_points


# --------------------------------------------------------------------------- #
# session_report.py : horodatages reels (P, A1)
# --------------------------------------------------------------------------- #


def test_strategy_summary_traduit_la_position_de_fenetre_en_horodatage_reel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_module, "RESULTS_DIR", tmp_path / "results")
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=200)
    ecrire_derive(
        reglages.dossier_etiquettes,
        "s1",
        Derive(type="deploiement", description="v2", instant_ts=float(instants[150])),
    )
    dataset_config = _dataset_config_path(tmp_path)
    session, stream = load_session_stream("s1", dataset_config=dataset_config)
    n_train_points = len(session.train)

    position_confirmation = 5
    position_reentrainement = 7
    point_index_confirmation = int(stream.end_index[position_confirmation])
    point_index_reentrainement = int(stream.end_index[position_reentrainement])
    horodatage_confirmation = horodatage(
        float(session.instants[n_train_points + point_index_confirmation])
    )
    horodatage_reentrainement = horodatage(
        float(session.instants[n_train_points + point_index_reentrainement])
    )

    # Artefacts ecrits a la main, au format de `save_run` (traces.npz +
    # events.jsonl), sans rejouer `proposed` : K = 960 fenetres de confirmation
    # rendrait ce test beaucoup trop lent pour ce qu'il verifie.
    directory = replay_module.run_directory(stream, "proposed", 42)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "traces.npz",
        scores=np.zeros(len(stream)),
        alerts=np.zeros(len(stream), dtype=bool),
        thresholds=np.zeros(len(stream)),
        y_true=stream.y,
        end_index=stream.end_index,
    )
    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"event": "drift_confirmed", "details": {"position": position_confirmation}})
            + "\n"
        )
        handle.write(
            json.dumps({"event": "retrain_start", "details": {"position": position_reentrainement}})
            + "\n"
        )

    resultat = strategy_summary(
        stream,
        session,
        n_train_points,
        seed=42,
        strategies=("proposed",),
        summary_path=tmp_path / "results" / "summary.csv",
    )

    ligne = resultat.iloc[0]
    assert ligne["drift_confirmed_at"] == horodatage_confirmation
    assert ligne["retrain_at"] == horodatage_reentrainement
    assert ligne["n_retrains_in_session"] == 1


def test_strategy_summary_ne_renseigne_pas_les_horodatages_pour_b0_et_b1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B0 et B1 ne surveillent aucune derive : ils n'ont ni confirmation ni
    # reentrainement declenche par une derive, et le rapport ne doit pas en
    # inventer pour eux.
    monkeypatch.setattr(replay_module, "RESULTS_DIR", tmp_path / "results")
    reglages = _reglages(tmp_path)
    _ecrire_session(reglages, "s1", n=200)
    dataset_config = _dataset_config_path(tmp_path)
    summary_path = tmp_path / "results" / "summary.csv"
    run_session_campaign(
        "s1",
        strategies=["static"],
        dataset_config=dataset_config,
        summary_path=summary_path,
        seed=42,
    )
    session, stream = load_session_stream("s1", dataset_config=dataset_config)

    resultat = strategy_summary(
        stream,
        session,
        len(session.train),
        seed=42,
        strategies=("static",),
        summary_path=summary_path,
    )

    assert "drift_confirmed_at" not in resultat.columns


# --------------------------------------------------------------------------- #
# Rapport complet : ecriture sur disque
# --------------------------------------------------------------------------- #


def test_write_session_report_ecrit_les_trois_fichiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_module, "RESULTS_DIR", tmp_path / "results")
    reglages = _reglages(tmp_path)
    instants = _ecrire_session(reglages, "s1", n=200)
    _anomalie(reglages, "s1", float(instants[150]), float(instants[152]))
    dataset_config = _dataset_config_path(tmp_path)
    summary_path = tmp_path / "results" / "summary.csv"
    run_session_campaign(
        "s1",
        strategies=["static"],
        dataset_config=dataset_config,
        summary_path=summary_path,
        seed=42,
    )

    rapport = build_session_report(
        "s1",
        dataset_config=dataset_config,
        strategies=("static",),
    )
    sortie = write_session_report(rapport, "s1", output_dir=tmp_path / "reports")

    assert (sortie / "strategies.csv").is_file()
    assert (sortie / "anomalies.csv").is_file()
    assert (sortie / "report.json").is_file()
    contenu = json.loads((sortie / "report.json").read_text(encoding="utf-8"))
    assert contenu["run_id"] == "s1"
    assert contenu["windows"]["n_windows_total"] == (
        contenu["windows"]["n_windows_train"] + contenu["windows"]["n_windows_eval"]
    )
    strategies_ecrites = pd.read_csv(sortie / "strategies.csv")
    assert list(strategies_ecrites["strategy"]) == ["static"]
    assert strategies_ecrites.loc[0, "disponible"]
    assert not pd.isna(strategies_ecrites.loc[0, "pr_auc_margin"])
