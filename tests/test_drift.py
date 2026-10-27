"""Tests de la surveillance de derive (etapes 1 et 2 de la strategie P).

Les deux statistiques sont verifiees contre des valeurs **calculees a la main**
et contre les implementations de reference (scipy pour KS) : un detecteur de
derive qui se trompe d'un facteur sur sa statistique rendrait tout le chapitre 6
faux sans qu'aucune execution ne plante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift.aggregate import FeatureDriftTracker
from src.drift.confirmation import Confirmation
from src.drift.ks import ks_from_proportions, ks_statistic
from src.drift.monitor import DriftMonitor
from src.drift.null_model import build_null_model
from src.drift.psi import psi, psi_from_proportions
from src.drift.reference import build_reference
from src.evaluation.drift_calibration import (
    build_monitor_trace,
    first_confirmation,
    longest_streak,
    streak_lengths,
    summarise,
    sweep_trace,
)
from src.evaluation.replay import ReplayStream

# ----------------------------------------------------------------------- KS


def test_ks_sur_proportions_calcul_a_la_main() -> None:
    """Deux repartitions connues : la statistique est l'ecart maximal des CDF.

    Reference : [0,5 ; 0,5 ; 0] -> CDF [0,5 ; 1,0 ; 1,0]
    Courant   : [0 ; 0,5 ; 0,5] -> CDF [0,0 ; 0,5 ; 1,0]
    Ecarts absolus : [0,5 ; 0,5 ; 0,0] -> maximum 0,5.
    """
    reference = np.array([[0.5, 0.5, 0.0]])
    current = np.array([[0.0, 0.5, 0.5]])
    assert ks_from_proportions(reference, current) == pytest.approx([0.5])


def test_ks_sur_proportions_est_nul_pour_deux_distributions_egales() -> None:
    reference = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert ks_from_proportions(reference, reference.copy()) == pytest.approx([0.0, 0.0])


def test_ks_sur_proportions_vaut_un_pour_deux_supports_disjoints() -> None:
    reference = np.array([[1.0, 0.0]])
    current = np.array([[0.0, 1.0]])
    assert ks_from_proportions(reference, current) == pytest.approx([1.0])


def test_ks_discretise_approche_le_ks_exact() -> None:
    """La discretisation est une approximation, et il faut savoir de combien.

    Le rejeu calcule KS sur une discretisation en intervalles de quantiles, et
    non sur les echantillons : c'est ce qui rend le cout O(F x B) par fenetre.
    Ce test borne l'erreur ainsi introduite sur un decalage franc.
    """
    rng = np.random.default_rng(0)
    reference_sample = rng.normal(size=(4000, 1))
    current_sample = rng.normal(loc=1.0, size=(800, 1))

    exact = ks_statistic(reference_sample, current_sample)

    reference = build_reference(reference_sample, n_bins=50)
    indices = reference.bin_indices(current_sample)
    counts = np.zeros((1, 50), dtype=np.float64)
    for row in indices:
        counts[0, row[0]] += 1.0
    approx = float(ks_from_proportions(reference.proportions, counts / len(indices))[0])

    assert approx == pytest.approx(exact, abs=0.02)


# ---------------------------------------------------------------------- PSI


def test_psi_sur_proportions_calcul_a_la_main() -> None:
    """PSI = somme (a - e) ln(a / e), sur un cas calculable de tete.

    e = [0,5 ; 0,5], a = [0,25 ; 0,75] :
      (0,25 - 0,5) ln(0,5)  = -0,25 x (-0,693147) = 0,173287
      (0,75 - 0,5) ln(1,5)  =  0,25 x   0,405465 = 0,101366
      total = 0,274653
    """
    reference = np.array([[0.5, 0.5]])
    current = np.array([[0.25, 0.75]])
    attendu = 0.25 * np.log(1.5) - 0.25 * np.log(0.5)
    assert psi_from_proportions(reference, current) == pytest.approx([attendu])
    assert float(attendu) == pytest.approx(0.274653, abs=1e-6)


def test_psi_est_nul_pour_deux_distributions_egales() -> None:
    reference = np.array([[0.2, 0.3, 0.5]])
    assert psi_from_proportions(reference, reference.copy()) == pytest.approx([0.0])


def test_psi_est_symetrique() -> None:
    """La forme retenue est symetrique : PSI(e, a) = PSI(a, e)."""
    a = np.array([[0.1, 0.4, 0.5]])
    b = np.array([[0.3, 0.3, 0.4]])
    assert psi_from_proportions(a, b) == pytest.approx(psi_from_proportions(b, a))


def test_psi_sur_echantillons_croit_avec_le_decalage() -> None:
    rng = np.random.default_rng(1)
    reference_sample = rng.normal(size=5000)
    valeurs = [
        psi(reference_sample, rng.normal(loc=decalage, size=1000), n_bins=20)
        for decalage in (0.0, 0.5, 1.0, 2.0)
    ]
    assert valeurs == sorted(valeurs)
    assert valeurs[0] < 0.1  # pas de derive : PSI proche de zero


# ------------------------------------------------------------- confirmation


def test_la_confirmation_exige_k_fenetres_consecutives() -> None:
    """Etape 2 du §6 : K fenetres consecutives, ni plus tot, ni sur un signal interrompu."""
    confirmation = Confirmation(k=3)
    assert confirmation.update(True) is False
    assert confirmation.update(True) is False
    assert confirmation.update(True) is True  # la troisieme, et pas avant
    assert confirmation.confirmed is True


def test_un_signal_interrompu_ne_confirme_pas() -> None:
    """C'est la propriete qui distingue une derive d'une anomalie passagere."""
    confirmation = Confirmation(k=3)
    for signal in (True, True, False, True, True):
        assert confirmation.update(signal) is False
    assert confirmation.streak == 2
    assert confirmation.confirmed is False


def test_la_confirmation_est_un_evenement_pas_un_etat() -> None:
    """Elle ne se reclenche pas : le delai de detection de la derive est datable."""
    confirmation = Confirmation(k=2)
    confirmation.update(True)
    assert confirmation.update(True) is True
    assert confirmation.update(True) is False
    assert confirmation.update(True) is False
    assert confirmation.confirmed is True

    confirmation.reset()
    assert confirmation.confirmed is False
    assert confirmation.update(True) is False
    assert confirmation.update(True) is True


def test_k_inferieur_a_un_est_refuse() -> None:
    with pytest.raises(ValueError, match="au moins 1"):
        Confirmation(k=0)


# ------------------------------------------------------------------ tracker


def _tracker(seuil: float = 0.5, recent: int = 50, n_bins: int = 10) -> tuple:
    rng = np.random.default_rng(2)
    reference_sample = rng.normal(size=(2000, 3))
    reference = build_reference(reference_sample, n_bins=n_bins)
    return (
        FeatureDriftTracker(
            reference=reference,
            recent_windows=recent,
            statistic="ks",
            feature_threshold=seuil,
        ),
        rng,
    )


def test_le_signal_n_est_pas_pret_avant_r_fenetres() -> None:
    """Comparer une reference de 2 000 lignes a 12 fenetres n'aurait aucun sens."""
    tracker, rng = _tracker(recent=50)
    for _ in range(49):
        signal = tracker.update(rng.normal(size=3))
        assert signal.ready is False
        assert signal.share == 0.0
    assert tracker.update(rng.normal(size=3)).ready is True


def test_la_part_de_features_en_derive_monte_sous_decalage() -> None:
    tracker, rng = _tracker(seuil=0.3, recent=50)
    for _ in range(50):
        tracker.update(rng.normal(size=3))
    sans_derive = tracker.update(rng.normal(size=3)).share

    for _ in range(60):
        signal = tracker.update(rng.normal(loc=3.0, size=3))
    assert sans_derive < 0.5
    assert signal.share == 1.0  # les trois features depassent le seuil


def test_une_feature_constante_ne_derive_jamais() -> None:
    """Regle de D18 : une feature constante sur la reference est exclue du calcul."""
    reference_sample = np.concatenate(
        [np.random.default_rng(3).normal(size=(500, 1)), np.ones((500, 1))], axis=1
    )
    reference = build_reference(reference_sample, n_bins=10)
    assert reference.n_usable == 1

    tracker = FeatureDriftTracker(
        reference=reference, recent_windows=20, statistic="ks", feature_threshold=0.1
    )
    for _ in range(40):
        signal = tracker.update(np.array([5.0, 42.0]))
    assert signal.n_features_usable == 1
    assert signal.n_features_drifting == 1  # la constante n'est pas comptee
    assert signal.share == 1.0


def test_seuil_par_feature_de_mauvaise_forme_est_refuse() -> None:
    tracker, _ = _tracker()
    with pytest.raises(ValueError, match="scalaire ou de forme"):
        FeatureDriftTracker(
            reference=tracker.reference,
            recent_windows=10,
            feature_threshold=np.zeros(99),
        )


# --------------------------------------------------------------- modele nul


def test_le_modele_nul_calibre_un_seuil_par_feature() -> None:
    """Le seuil est celui qu'atteint l'entrainement lui-meme, sans aucune derive.

    C'est tout l'interet de la methode : un seuil absolu commun ignore
    l'autocorrelation et l'etendue propres a chaque feature, et declare en
    derive des features qui n'ont fait que bouger normalement.
    """
    rng = np.random.default_rng(4)
    # Feature 0 : bruit blanc. Feature 1 : marche aleatoire, tres autocorrelee,
    # donc des blocs de 100 lignes y sont naturellement tres differents.
    bruit = rng.normal(size=(3000, 1))
    marche = np.cumsum(rng.normal(size=(3000, 1)), axis=0)
    features = np.concatenate([bruit, marche], axis=1)

    reference = build_reference(features, n_bins=20)
    nul = build_null_model(features, reference, recent_windows=100, quantile=0.99, step=50)

    assert nul.n_features == 2
    assert nul.n_blocks > 50
    # La feature autocorrelee recoit un seuil nettement plus eleve : c'est la
    # propriete que le seuil absolu n'avait pas.
    assert nul.thresholds[1] > nul.thresholds[0]
    assert np.all(np.isfinite(nul.thresholds))


def test_le_modele_nul_refuse_un_entrainement_trop_court() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(80, 2))
    reference = build_reference(features, n_bins=10)
    with pytest.raises(ValueError, match="trop court"):
        build_null_model(features, reference, recent_windows=100)


def test_une_feature_constante_recoit_un_seuil_infini() -> None:
    rng = np.random.default_rng(6)
    features = np.concatenate([rng.normal(size=(600, 1)), np.zeros((600, 1))], axis=1)
    reference = build_reference(features, n_bins=10)
    nul = build_null_model(features, reference, recent_windows=100, step=50)
    assert np.isfinite(nul.thresholds[0])
    assert np.isinf(nul.thresholds[1])


# ------------------------------------------------------------------ moniteur


def _drift_config(**surcharges) -> dict:
    config = {
        "statistic": "ks",
        "n_bins": 10,
        "recent_windows": 50,
        "drifting_share": 0.50,
        "confirmation_windows": 5,
        "constant_tolerance": 0.0,
        "feature_threshold": {"method": "absolute", "absolute": 0.3},
        "alert_rate": {"enabled": False},
        "combination": "features",
    }
    config.update(surcharges)
    return config


def test_le_moniteur_confirme_une_derive_franche_et_pas_un_flux_stable() -> None:
    """Propriete minimale : derive franche confirmee, flux stable non confirme."""
    rng = np.random.default_rng(7)
    entrainement = rng.normal(size=(1000, 4))

    stable = DriftMonitor(_drift_config())
    stable.fit_reference(entrainement)
    confirmations = sum(stable.update(rng.normal(size=4), False).confirmed for _ in range(400))
    assert confirmations == 0

    derive = DriftMonitor(_drift_config())
    derive.fit_reference(entrainement)
    for _ in range(60):
        derive.update(rng.normal(size=4), False)
    etats = [derive.update(rng.normal(loc=4.0, size=4), False) for _ in range(200)]
    assert any(etat.confirmed for etat in etats)
    # Une seule confirmation : c'est un evenement, pas un etat.
    assert sum(etat.confirmed for etat in etats) == 1


def test_le_moniteur_exige_le_nombre_de_fenetres_du_modele_nul() -> None:
    """Contrainte structurelle utilisee par la strategie pour ne pas se tirer dessus."""
    moniteur = DriftMonitor(_drift_config(feature_threshold={"method": "empirical_null"}))
    assert moniteur.required_training_windows() == 50

    absolu = DriftMonitor(_drift_config())
    assert absolu.required_training_windows() == 10


# --------------------------------------------------- balayage de calibrage


def test_le_balayage_reproduit_exactement_la_confirmation() -> None:
    """Le calibrage vectorise `Confirmation` : les deux doivent coincider.

    C'est le test qui autorise a calibrer K sans rejouer le flux. S'ils
    divergeaient, les parametres choisis ne seraient pas ceux appliques par la
    strategie, et le chapitre 4 annoncerait un reglage que le code n'applique
    pas.
    """
    rng = np.random.default_rng(8)
    for _ in range(30):
        signal = rng.random(400) < rng.uniform(0.1, 0.9)
        for k in (1, 2, 5, 30, 120):
            attendu = None
            confirmation = Confirmation(k)
            for position, value in enumerate(signal):
                if confirmation.update(bool(value)):
                    attendu = position
                    break
            assert first_confirmation(signal, k) == attendu, (k, signal.sum())


def test_les_longueurs_de_serie_sont_exactes() -> None:
    signal = np.array([False, True, True, True, False, True, False, True, True])
    np.testing.assert_array_equal(streak_lengths(signal), [0, 1, 2, 3, 0, 1, 0, 1, 2])
    assert longest_streak(signal) == 3
    assert longest_streak(np.zeros(10, dtype=bool)) == 0
    assert longest_streak(np.ones(10, dtype=bool)) == 10


def test_la_trace_de_calibrage_ne_depend_pas_du_seuil_passe_au_tracker() -> None:
    """Les statistiques brutes sont conservees, pas une decision deja prise."""
    rng = np.random.default_rng(9)
    stream = _stream_synthetique(rng)
    config = {
        "statistic": "ks",
        "n_bins": 10,
        "recent_windows": 50,
        "constant_tolerance": 0.0,
        "feature_threshold": {"step": 25},
    }
    trace = build_monitor_trace(stream, config)

    assert trace.values.shape == (len(stream), 3)
    assert np.all(np.isnan(trace.values[:49]))  # pas pret avant R fenetres
    assert np.all(np.isfinite(trace.values[49:]))
    assert trace.distribution.shape[1] == 3
    assert trace.drift_window is None

    rows = sweep_trace(trace, quantiles=(0.99,), margins=(1.0,), shares=(0.1,), confirmations=(5,))
    assert len(rows) == 1
    assert set(rows[0]) >= {"confirme", "fenetre_confirmation", "plus_longue_serie_avant"}


def _stream_synthetique(rng) -> ReplayStream:
    """Flux minimal : la seconde moitie du test derive franchement."""
    train = rng.normal(size=(600, 3))
    test = rng.normal(size=(300, 3))
    test[150:] += 5.0
    return ReplayStream(
        source="test",
        machine="synthetique",
        scenario="none",
        train_X=train,
        X=test,
        y=np.zeros(300, dtype=np.int64),
        end_index=np.arange(300),
        events=[],
        window_size=1,
        stride=1,
        sampling_interval_seconds=60,
    )


# ------------------------------------------------- tableau de decision (summarise)


def _ligne(
    machine: str,
    scenario: str,
    confirme: bool,
    avant: bool,
    apres: bool,
    drift_window: float | None,
    recent_windows: int = 500,
) -> dict:
    """Une ligne de grille, telle que `sweep_trace` l'emet."""
    return {
        "recent_windows": recent_windows,
        "machine": machine,
        "scenario": scenario,
        "seed": 42,
        "quantile": 0.999,
        "margin": 1.0,
        "drifting_share": 0.20,
        "confirmation_windows": 960,
        "confirme": confirme,
        "confirmation_avant_derive": avant,
        "detection_apres_derive": apres,
        "delai_detection_fenetres": 100.0 if apres else np.nan,
        "drift_window": drift_window,
    }


def test_le_critere_elargi_compte_les_confirmations_precoces_des_machines_stables() -> None:
    """Une confirmation avant t_d sur une machine stable est fausse, meme sous injection.

    Avant t_d, le flux injecte est identique au flux d'origine : la definition
    du §7, restreinte au scenario `none`, laisse donc passer des faux que le
    critere elargi attrape. Les deux colonnes doivent differer sur ce cas.
    """
    table = pd.DataFrame(
        [
            # Machine stable, `none` propre : invisible pour les deux criteres.
            _ligne("machine-1-3", "none", False, False, False, None),
            # Machine stable, injection, confirmation AVANT t_d : fausse.
            _ligne("machine-1-3", "s1_soudaine", True, True, False, 500.0),
            # Machine stable, injection, confirmation APRES t_d : une detection.
            _ligne("machine-2-7", "s1_soudaine", True, False, True, 500.0),
            _ligne("machine-2-7", "none", False, False, False, None),
        ]
    )
    resume = summarise(table, ["machine-1-3", "machine-2-7"])
    assert len(resume) == 1
    ligne = resume.iloc[0]
    assert ligne["fausses_confirmations_stables"] == 0  # definition du §7
    assert ligne["faux_signaux_stables"] == 1  # critere elargi
    assert ligne["n_traces_stables"] == 4
    assert ligne["detections"] == 1


def test_une_confirmation_sans_injection_est_toujours_fausse() -> None:
    """Sur `none` il n'y a pas de t_d : `confirmation_avant_derive` est faux par construction.

    Le critere elargi doit malgre tout la compter, sinon il serait moins severe
    que celui du §7 sur le scenario qui le definit.
    """
    table = pd.DataFrame(
        [
            _ligne("machine-1-3", "none", True, False, False, None),
            _ligne("machine-1-3", "s1_soudaine", False, False, False, 500.0),
        ]
    )
    resume = summarise(table, ["machine-1-3"])
    assert resume.iloc[0]["fausses_confirmations_stables"] == 1
    assert resume.iloc[0]["faux_signaux_stables"] == 1


def test_la_taille_de_fenetre_recente_est_un_axe_du_tableau() -> None:
    """Deux valeurs de R ne doivent pas etre agregees ensemble.

    R change la trace elle-meme (distribution nulle et histogramme glissant) :
    melanger deux valeurs dans la meme ligne de decision n'aurait aucun sens.
    """
    table = pd.DataFrame(
        [
            _ligne("machine-1-3", "none", True, False, False, None, recent_windows=120),
            _ligne("machine-1-3", "none", False, False, False, None, recent_windows=500),
        ]
    )
    resume = summarise(table, ["machine-1-3"])
    assert len(resume) == 2
    par_r = resume.set_index("recent_windows")["faux_signaux_stables"].to_dict()
    assert par_r == {120: 1, 500: 0}


def test_une_machine_instable_ne_compte_pas_dans_la_specificite() -> None:
    """Sur une machine naturellement derivante, une confirmation n'est pas une erreur.

    C'est la regle d'OBS03 et de D19 : la compter comme fausse ferait regler K
    sur un faux probleme. Elle est rapportee a part.
    """
    table = pd.DataFrame(
        [
            _ligne("machine-3-1", "none", True, False, False, None),
            _ligne("machine-1-3", "none", False, False, False, None),
        ]
    )
    resume = summarise(table, ["machine-1-3"])
    ligne = resume.iloc[0]
    assert ligne["faux_signaux_stables"] == 0
    assert ligne["confirmations_derive_naturelle"] == 1
    assert ligne["n_traces_stables"] == 1
