"""Criblage des machines SMD : quantification de la derive naturelle du dataset.

Repond a une question posee par la relecture du jalon M0 : le tableau des 28
machines doit exister comme fichier de resultats citable, et non seulement dans
une note. La commande `mdrift smd-screening` ecrit `results/smd_screening.csv`.

Pour chaque machine, on entraine le modele fige (B0) sur la moitie `train`, on
l'applique a la moitie `test`, et on mesure :
- le rapport entre le taux d'alertes observe et le taux calibre par le seuil ;
- la forme temporelle de la degradation ;
- la divergence de distribution par metrique brute (Kolmogorov-Smirnov).

Deux precautions de mesure, apprises a nos depens (docs/decisions.md, OBS03) :

1. **Moyenne glissante causale.** Une moyenne centree sur `w` fenetres attribue a
   l'instant t une information issue de t + w / 2. Avec w = 500 fenetres de 60 s,
   elle anticipe de 4,2 h et datait le debut de la derive 4 h trop tot.
2. **Fausses alertes seules.** Une alerte sur une fenetre etiquetee positive est
   une detection reussie, pas un signe de derive. En comptant toutes les alertes,
   le premier depassement mesure sur machine-3-1 tombait exactement sur une
   anomalie etiquetee (20,1 h - 20,6 h) et non sur la derive (vers 90 h).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from datasets.smd.loader import SMDMachine, load_machine
from src.config import load_config, resolve_path
from src.evaluation.replay import RESULTS_DIR
from src.features.windowing import make_windows
from src.models.isolation_forest import build_detector

SCREENING_PATH = RESULTS_DIR / "smd_screening.csv"


def causal_rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne des `window` dernieres valeurs, bornes incluses.

    Retourne `(moyennes, indices)` ou `indices[i]` est l'indice de la **derniere**
    valeur entrant dans `moyennes[i]` : aucune information posterieure a cet
    indice n'intervient dans la moyenne. C'est la seule position compatible avec
    l'evaluation prequentielle du memoire.
    """
    if window < 1:
        raise ValueError("window doit valoir au moins 1.")
    if len(values) < window:
        raise ValueError(f"{len(values)} valeurs, trop peu pour une moyenne sur {window}.")

    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    means = (cumulative[window:] - cumulative[:-window]) / window
    indices = np.arange(window - 1, len(values))
    return means, indices


def false_alerts(alerts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Alertes emises sur des fenetres etiquetees negatives (0 ou 1 par fenetre).

    Definition de `configs/evaluation.yaml` : identique aux faux positifs de la
    matrice de confusion par fenetre, pour n'avoir qu'une seule verite terrain.
    """
    if len(alerts) != len(y):
        raise ValueError(f"{len(alerts)} alertes pour {len(y)} etiquettes.")
    return (alerts.astype(bool) & ~y.astype(bool)).astype(np.float64)


def first_crossing(means: np.ndarray, indices: np.ndarray, level: float) -> int | None:
    """Indice du premier instant ou la moyenne atteint `level`, sinon None."""
    above = np.flatnonzero(means >= level)
    return int(indices[above[0]]) if len(above) else None


def confirmed_onset(
    means: np.ndarray,
    indices: np.ndarray,
    level: float,
    consecutive: int,
) -> int | None:
    """Premier instant ou la moyenne depasse `level` sans interruption pendant
    `consecutive` fenetres. Retourne l'indice de **confirmation**, c'est-a-dire
    la fin de cette plage : c'est l'instant ou un moniteur en ligne conclurait.

    La persistance exigee est la meme que celle de l'etape 2 de la strategie
    proposee : elle est ce qui separe une derive durable d'une anomalie passagere.
    """
    if consecutive < 1:
        raise ValueError("consecutive doit valoir au moins 1.")
    high = means >= level
    padded = np.concatenate(([0], high.astype(np.int64), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if end - start + 1 >= consecutive:
            return int(indices[start + consecutive - 1])
    return None


def slice_profile(
    rate: np.ndarray,
    sampling_interval_seconds: int,
    slice_hours: float,
    max_slices: int,
) -> list[float]:
    """Taux moyen par tranches successives de `slice_hours`, en pourcentage.

    Decrit la forme de la degradation sans lissage, donc sans decalage possible.
    """
    per_slice = int(round(slice_hours * 3600 / sampling_interval_seconds))
    n_slices = min(max_slices, len(rate) // per_slice)
    return [
        round(float(rate[index * per_slice : (index + 1) * per_slice].mean()) * 100, 1)
        for index in range(n_slices)
    ]


def screen_machine(
    machine: SMDMachine,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    screening_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Crible une machine : une ligne du tableau de resultats."""
    started = time.perf_counter()

    train = make_windows(machine.train, features_config)
    test = make_windows(machine.test, features_config, label=machine.test_label)
    if test.y is None:
        raise ValueError(f"{machine.name} : moitie test sans etiquettes.")

    detector = build_detector(model_config, features_config, seed).fit(train.X)
    alerts = detector.predict(test.X)

    calibrated_rate = 1.0 - float(model_config["threshold"]["percentile"]) / 100.0
    alert_rate = float(alerts.mean())
    ratio = alert_rate / calibrated_rate

    rolling = screening_config["rolling"]
    if rolling["position"] != "causal":
        raise NotImplementedError(
            f"Position de moyenne glissante non admise : {rolling['position']} "
            "(voir docs/decisions.md, OBS03)."
        )
    if rolling["counted_alerts"] != "false_only":
        raise NotImplementedError(
            f"Alertes comptees non admises : {rolling['counted_alerts']} "
            "(voir docs/decisions.md, OBS03)."
        )

    false_alert_mask = false_alerts(alerts, test.y)
    false_alert_rate = float(false_alert_mask.mean())
    false_alert_ratio = false_alert_rate / calibrated_rate
    window = int(rolling["windows"])
    means, indices = causal_rolling_mean(false_alert_mask, window)

    onset_config = screening_config["drift_onset"]
    level = float(onset_config["ratio_vs_calibrated"]) * calibrated_rate
    onset = confirmed_onset(means, indices, level, int(onset_config["consecutive_windows"]))
    crossing = first_crossing(means, indices, level)

    interval = machine.sampling_interval_seconds
    hours = lambda index: round(index * interval / 3600, 1)  # noqa: E731

    shift = screening_config["distribution_shift"]
    ks = np.array(
        [
            ks_2samp(
                machine.train[:, column],
                machine.test[:, column],
                method=shift["ks_method"],
            ).statistic
            for column in range(machine.n_features)
        ]
    )

    profile = screening_config["profile"]
    stability = screening_config["stability"]
    basis = stability.get("basis", "false_alerts")
    if basis == "false_alerts":
        stability_ratio = false_alert_ratio
    elif basis == "all_alerts":
        stability_ratio = ratio
    else:
        raise NotImplementedError(f"Base du critere de stabilite inconnue : {basis}")

    return {
        "machine": machine.name,
        "n_points_train": int(len(machine.train)),
        "n_points_test": int(len(machine.test)),
        "n_fenetres_test": int(len(test)),
        "duree_test_heures": hours(len(test)),
        "taux_anomalies_points": round(machine.anomaly_rate, 4),
        "taux_fenetres_positives": round(float(test.y.mean()), 4),
        "n_evenements": len(machine.events),
        "taux_alertes_test": round(alert_rate, 4),
        "taux_calibre": calibrated_rate,
        "rapport_vs_calibre": round(ratio, 2),
        "taux_fausses_alertes": round(false_alert_rate, 4),
        "rapport_fausses_alertes": round(false_alert_ratio, 2),
        "critere_stabilite": basis,
        "stable": bool(stability_ratio < float(stability["max_ratio_vs_calibrated"])),
        "derive_franchissement_heures": None if crossing is None else hours(crossing),
        "derive_confirmee_heures": None if onset is None else hours(onset),
        "ks_max": round(float(ks.max()), 3),
        "ks_median": round(float(np.median(ks)), 3),
        "n_metriques_ks_eleve": int((ks > float(shift["ks_statistic_threshold"])).sum()),
        "profil_fausses_alertes_pct": " ".join(
            f"{value:.1f}"
            for value in slice_profile(
                false_alert_mask,
                interval,
                float(profile["slice_hours"]),
                int(profile["max_slices"]),
            )
        ),
        "seed": seed,
        "secondes": round(time.perf_counter() - started, 1),
    }


def screen(
    dataset_config: str | Path = "datasets/smd_all.yaml",
    features_config: str | Path = "features.yaml",
    model_config: str | Path = "model.yaml",
    screening_config: str | Path = "screening.yaml",
    seed: int = 42,
    output_path: Path = SCREENING_PATH,
    on_row: Any = None,
) -> pd.DataFrame:
    """Crible toutes les machines de la configuration et ecrit le CSV de resultats."""
    dataset = load_config(dataset_config)
    features = load_config(features_config)
    model = load_config(model_config)
    screening = load_config(screening_config)

    rows: list[dict[str, Any]] = []
    for name in dataset["machines"]:
        machine = load_machine(name, dataset_config)
        row = screen_machine(machine, features, model, screening, seed)
        rows.append(row)
        if on_row is not None:
            on_row(row)

    table = pd.DataFrame(rows)
    destination = resolve_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table


def screening_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Chiffres de synthese du criblage, repris dans docs/decisions.md et au 6.2.1."""
    confirmed = table["derive_confirmee_heures"].notna()
    return {
        "n_machines": int(len(table)),
        "n_stables": int(table["stable"].sum()),
        "rapport_median": round(float(table["rapport_vs_calibre"].median()), 2),
        "rapport_min": round(float(table["rapport_vs_calibre"].min()), 2),
        "rapport_max": round(float(table["rapport_vs_calibre"].max()), 2),
        "n_derive_confirmee": int(confirmed.sum()),
        "n_derive_confirmee_parmi_stables": int((confirmed & table["stable"]).sum()),
        "confirmation_mediane_heures": (
            None
            if not confirmed.any()
            else round(float(table.loc[confirmed, "derive_confirmee_heures"].median()), 1)
        ),
        "ks_metriques_median": float(table["n_metriques_ks_eleve"].median()),
    }
