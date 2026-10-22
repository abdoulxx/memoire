"""Detecteur par Isolation Forest.

Modele principal du memoire : non supervise, et surtout **rapide a reentrainer**,
ce qui rend le cout du reentrainement mesurable (c'est l'une des metriques
comparees entre les strategies).

Le detecteur encapsule trois elements qui doivent rester solidaires et etre
sauvegardes ensemble :
1. la standardisation, ajustee sur les donnees d'entrainement uniquement ;
2. la foret ;
3. le seuil d'alerte, recalcule a chaque entrainement comme un percentile eleve
   des scores obtenus sur ces memes donnees d'entrainement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class IsolationForestDetector:
    """Isolation Forest avec standardisation et seuil par percentile.

    `random_state` vient de la seed de l'execution : deux executions de meme
    seed produisent le meme modele.
    """

    params: dict[str, Any]
    threshold_percentile: float = 99.0
    random_state: int | None = None
    scaling: str = "standard"

    threshold: float = field(default=float("nan"), init=False)
    fit_seconds: float = field(default=0.0, init=False)
    n_train_windows: int = field(default=0, init=False)
    _forest: IsolationForest | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._forest is not None

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        """Entraine la foret et fixe le seuil. Mesure le temps de calcul."""
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"Matrice de features attendue, recu {X.shape}.")

        started = time.perf_counter()

        if self.scaling == "standard":
            self._scaler = StandardScaler().fit(X)
        elif self.scaling == "none":
            self._scaler = None
        else:
            raise ValueError(f"Methode de standardisation inconnue : {self.scaling}")

        params = {k: v for k, v in self.params.items() if k != "random_state"}
        self._forest = IsolationForest(random_state=self.random_state, **params)
        self._forest.fit(self._transform(X))

        # Seuil : percentile eleve des scores d'entrainement. Recalcule a chaque
        # entrainement, donc aussi a chaque reentrainement d'une strategie.
        self.threshold = float(np.percentile(self.score(X), self.threshold_percentile))
        self.fit_seconds = time.perf_counter() - started
        self.n_train_windows = int(len(X))
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X if self._scaler is None else self._scaler.transform(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Score d'anomalie. Score eleve = plus anormal (oppose de scikit-learn)."""
        if self._forest is None:
            raise RuntimeError("Detecteur non entraine : appeler fit() d'abord.")
        return -self._forest.score_samples(self._transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alerte binaire : 1 si le score depasse strictement le seuil."""
        return (self.score(X) > self.threshold).astype(np.int64)

    def save(self, path: str | Path) -> Path:
        """Sauvegarde standardisation, foret et seuil dans un seul fichier."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @staticmethod
    def load(path: str | Path) -> IsolationForestDetector:
        detector = joblib.load(Path(path))
        if not isinstance(detector, IsolationForestDetector):
            raise TypeError(f"{path} ne contient pas un IsolationForestDetector.")
        return detector


def build_detector(
    model_config: dict[str, Any],
    features_config: dict[str, Any],
    random_state: int | None = None,
) -> IsolationForestDetector:
    """Construit le detecteur decrit par `configs/model.yaml`."""
    name = model_config.get("model", "isolation_forest")
    if name != "isolation_forest":
        raise NotImplementedError(f"Modele non implemente : {name}")

    threshold = model_config["threshold"]
    if threshold["method"] != "train_percentile":
        raise NotImplementedError(f"Regle de seuil non implementee : {threshold['method']}")

    return IsolationForestDetector(
        params=dict(model_config["isolation_forest"]),
        threshold_percentile=float(threshold["percentile"]),
        random_state=random_state,
        scaling=features_config.get("scaling", {}).get("method", "standard"),
    )
