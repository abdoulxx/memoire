"""Le moniteur de derive : etapes 1 et 2 de la strategie proposee.

Il combine le signal principal — part des features dont la statistique depasse
un seuil (D18) — et le signal secondaire — ADWIN sur le taux d'alertes (D19,
OBS09) — puis exige une persistance de K fenetres consecutives avant de
confirmer.

Regle de combinaison, parametree par `configs/drift.yaml` :

- `features` : seul le signal par features compte. C'est le defaut, parce que
  c'est le seul des deux dont la specificite a ete etablie.
- `features_and_alerts` : les deux doivent etre actifs. Plus specifique, au prix
  d'un delai de detection plus long.
- `features_or_alerts` : l'un des deux suffit. Plus sensible, et c'est la
  variante dont D19 laisse craindre qu'elle confirme a tort.

Les trois sont implementees pour que le choix soit **mesure** sur le perimetre
de mise au point et non postule. Aucune ne lit d'etiquette : le moniteur ne voit
que les features, le score et l'indicatrice d'alerte du modele en service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.drift.adwin import AlertRateMonitor
from src.drift.aggregate import FeatureDriftTracker
from src.drift.confirmation import Confirmation
from src.drift.null_model import NullModel, build_null_model
from src.drift.reference import DriftReference, build_reference

COMBINATIONS = ("features", "features_and_alerts", "features_or_alerts")


@dataclass(frozen=True)
class DriftState:
    """Ce que le moniteur rapporte a chaque fenetre."""

    signal: bool
    confirmed: bool
    streak: int
    share: float
    n_features_drifting: int
    n_features_usable: int
    statistic_median: float
    statistic_max: float
    alert_rate_signal: bool
    alert_rate_fired: bool
    ready: bool

    def as_details(self) -> dict[str, Any]:
        return {
            "share": round(self.share, 4),
            "streak": self.streak,
            "n_features_drifting": self.n_features_drifting,
            "n_features_usable": self.n_features_usable,
            "statistic_median": round(self.statistic_median, 4),
            "statistic_max": round(self.statistic_max, 4),
            "alert_rate_signal": self.alert_rate_signal,
        }


class DriftMonitor:
    """Surveillance de derive sur un flux de fenetres, recalee a chaque promotion.

    Cycle de vie : `fit_reference(training_features)` apres chaque entrainement
    du modele en service, puis `update(features, alert)` a chaque fenetre. La
    remise a zero est faite par `fit_reference`, qui est le seul moment ou la
    reference change.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        statistic = str(config.get("statistic", "ks"))
        combination = str(config.get("combination", "features"))
        if combination not in COMBINATIONS:
            raise ValueError(
                f"Combinaison inconnue : {combination!r}. Attendu l'une de {COMBINATIONS}."
            )

        self.statistic = statistic
        self.combination = combination
        self.n_bins = int(config.get("n_bins", 50))
        self.recent_windows = int(config.get("recent_windows", 500))
        self.drifting_share = float(config.get("drifting_share", 0.20))
        self.k = int(config.get("confirmation_windows", 120))
        self.psi_epsilon = float(config.get("psi_epsilon", 1e-6))
        self.constant_tolerance = float(config.get("constant_tolerance", 0.0))

        # Seuil par feature. `empirical_null` est le defaut : le seuil de chaque
        # feature est un quantile de ce qu'elle atteint sur l'entrainement meme,
        # sans derive. `absolute` conserve un seuil fixe, pour l'ablation et pour
        # montrer au memoire pourquoi il ne suffit pas.
        threshold_config = dict(config.get("feature_threshold", {}) or {})
        self.threshold_method = str(threshold_config.get("method", "empirical_null"))
        if self.threshold_method not in ("empirical_null", "absolute"):
            raise ValueError(
                f"Methode de seuil inconnue : {self.threshold_method!r}. "
                "Attendu 'empirical_null' ou 'absolute'."
            )
        self.threshold_quantile = float(threshold_config.get("quantile", 0.99))
        self.threshold_margin = float(threshold_config.get("margin", 1.0))
        self.threshold_step = threshold_config.get("step")
        self.threshold_absolute = float(threshold_config.get("absolute", 0.5))

        alert_config = dict(config.get("alert_rate", {}) or {})
        self.alert_rate_enabled = bool(alert_config.get("enabled", True))
        self._alert_params = {
            "delta": float(alert_config.get("delta", 0.002)),
            "grace_period": int(alert_config.get("grace_period", 10)),
            "clock": int(alert_config.get("clock", 32)),
        }

        self._reference: DriftReference | None = None
        self._null: NullModel | None = None
        self._tracker: FeatureDriftTracker | None = None
        self._confirmation = Confirmation(self.k)
        self._alerts = AlertRateMonitor(**self._alert_params) if self.alert_rate_enabled else None

    @property
    def reference(self) -> DriftReference | None:
        return self._reference

    def required_training_windows(self) -> int:
        """Nombre minimal de fenetres d'entrainement que ce moniteur sait exploiter.

        Le modele nul glisse un bloc de R fenetres sur l'entrainement pour
        calibrer le seuil de chaque feature : en dessous de R fenetres, il n'y a
        pas un seul bloc et le seuil n'existe pas. C'est une contrainte
        structurelle du moniteur, pas un reglage : une strategie qui reentraine
        sur moins que cela produit un modele qu'elle ne pourra plus surveiller.

        La valeur renvoyee est un plancher dur. Le nombre de blocs reellement
        souhaitable pour qu'un quantile a 0,99 ait un sens est bien plus eleve,
        et c'est un parametre de calibrage (`min_train_minutes`), pas une borne
        de validite.
        """
        if self.threshold_method == "empirical_null":
            return int(self.recent_windows)
        # Methode `absolute` : seule la reference est construite, et elle a
        # besoin d'assez de lignes pour que les quantiles de binning soient
        # distincts.
        return int(self.n_bins)

    @property
    def null_model(self) -> NullModel | None:
        return self._null

    def fit_reference(self, training_features: np.ndarray) -> None:
        """Recale la reference sur les donnees d'entrainement du modele en service.

        Le seuil par feature est recalibre ici aussi, et pour la meme raison : il
        decrit ce que chaque feature atteint **sur cet entrainement precis** en
        l'absence de derive. Un seuil herite d'un modele precedent n'aurait pas
        de sens apres une promotion.
        """
        features = np.asarray(training_features, dtype=np.float64)
        self._reference = build_reference(
            features,
            n_bins=self.n_bins,
            constant_tolerance=self.constant_tolerance,
        )
        if self.threshold_method == "empirical_null":
            self._null = build_null_model(
                features,
                self._reference,
                recent_windows=self.recent_windows,
                statistic=self.statistic,
                quantile=self.threshold_quantile,
                margin=self.threshold_margin,
                step=self.threshold_step,
                psi_epsilon=self.psi_epsilon,
            )
            threshold: float | np.ndarray = self._null.thresholds
        else:
            self._null = None
            threshold = self.threshold_absolute

        self._tracker = FeatureDriftTracker(
            reference=self._reference,
            recent_windows=self.recent_windows,
            statistic=self.statistic,
            feature_threshold=threshold,
            psi_epsilon=self.psi_epsilon,
        )
        self._confirmation.reset()
        if self._alerts is not None:
            self._alerts.reset()

    def update(self, features: np.ndarray, alert: bool) -> DriftState:
        """Ajoute une fenetre et renvoie l'etat de la surveillance.

        `confirmed` est vrai a la fenetre **exacte** ou la persistance de K est
        atteinte, et a elle seule : c'est ce qui rend le delai de detection de la
        derive datable sans ambiguite.
        """
        if self._tracker is None:
            raise RuntimeError("fit_reference doit etre appele avant update.")

        feature_signal = self._tracker.update(features)
        alert_fired = False
        alert_signal = False
        if self._alerts is not None:
            alert_fired = self._alerts.update(alert)
            # Etat latche, et non signal instantane : voir AlertRateMonitor.latched.
            alert_signal = self._alerts.latched

        features_active = feature_signal.ready and feature_signal.share >= self.drifting_share
        if self.combination == "features":
            signal = features_active
        elif self.combination == "features_and_alerts":
            signal = features_active and alert_signal
        else:
            signal = features_active or (alert_signal and feature_signal.ready)

        confirmed = self._confirmation.update(signal)
        return DriftState(
            signal=signal,
            confirmed=confirmed,
            streak=self._confirmation.streak,
            share=feature_signal.share,
            n_features_drifting=feature_signal.n_features_drifting,
            n_features_usable=feature_signal.n_features_usable,
            statistic_median=feature_signal.statistic_median,
            statistic_max=feature_signal.statistic_max,
            alert_rate_signal=alert_signal,
            alert_rate_fired=alert_fired,
            ready=feature_signal.ready,
        )
