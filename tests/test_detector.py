"""Tests du detecteur Isolation Forest.

On verifie le contrat, pas la qualite de detection (mesuree par les metriques) :
convention de signe, seuil par percentile, reproductibilite par la seed,
sauvegarde/rechargement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.models.isolation_forest import IsolationForestDetector, build_detector

MODEL_CONFIG = load_config("model.yaml")
FEATURES_CONFIG = load_config("features.yaml")


def _donnees(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 5))


def test_contrat_fit_score_predict() -> None:
    X = _donnees()
    detector = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=42).fit(X)
    scores = detector.score(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(detector.threshold)
    assert detector.predict(X).shape == (len(X),)
    assert detector.n_train_windows == len(X)
    assert detector.fit_seconds > 0


def test_seuil_est_le_percentile_des_scores_d_entrainement() -> None:
    """Le seuil doit laisser passer environ (100 - p) % des fenetres d'entrainement."""
    X = _donnees(n=1000)
    detector = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=42).fit(X)
    taux_alertes = detector.predict(X).mean()
    attendu = 1.0 - MODEL_CONFIG["threshold"]["percentile"] / 100.0
    assert taux_alertes == pytest.approx(attendu, abs=0.005)


def test_score_eleve_signifie_plus_anormal() -> None:
    """Convention du projet : un point aberrant doit obtenir un score plus eleve."""
    X = _donnees()
    detector = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=42).fit(X)
    aberrant = np.full((1, X.shape[1]), 25.0)
    assert detector.score(aberrant)[0] > detector.score(X).max()
    assert detector.predict(aberrant)[0] == 1


def test_reproductibilite_par_la_seed() -> None:
    X = _donnees()
    a = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=7).fit(X)
    b = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=7).fit(X)
    c = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=8).fit(X)
    np.testing.assert_allclose(a.score(X), b.score(X))
    assert a.threshold == b.threshold
    assert not np.allclose(a.score(X), c.score(X))


def test_sauvegarde_et_rechargement(tmp_path) -> None:
    """Standardisation, foret et seuil doivent rester solidaires."""
    X = _donnees()
    detector = build_detector(MODEL_CONFIG, FEATURES_CONFIG, random_state=42).fit(X)
    chemin = detector.save(tmp_path / "detecteur.joblib")
    recharge = IsolationForestDetector.load(chemin)
    assert recharge.threshold == detector.threshold
    np.testing.assert_allclose(recharge.score(X), detector.score(X))


def test_erreurs() -> None:
    detector = build_detector(MODEL_CONFIG, FEATURES_CONFIG)
    with pytest.raises(RuntimeError, match="non entraine"):
        detector.score(_donnees())
    with pytest.raises(ValueError):
        detector.fit(np.empty((0, 3)))
