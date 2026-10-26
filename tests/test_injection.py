"""Tests de l'injection de derive (M1).

Tous les cas sont calcules a la main sur des series synthetiques de mu et sigma
connus, de sorte que l'amplitude attendue soit verifiable exactement.

Deux tests encodent des invariants dont depend toute l'interpretation du
chapitre 6, et pas seulement la correction du code :
- `test_les_etiquettes_ne_sont_jamais_modifiees` : une derive change le
  comportement normal, pas la verite terrain des anomalies ;
- `test_meme_seed_meme_flux` : la reproductibilite du flux rejoue, condition
  pour que deux strategies soient comparees sur des donnees identiques.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios.injection import (
    DriftScenario,
    alpha_profile,
    drift_onset,
    inject,
    injectable_metrics,
    load_scenario,
    select_metrics,
)

TRANSFORMATION = {
    "level_shift_k": 2.0,
    "scale_factor": 1.0,  # neutre : isole le decalage de niveau dans les tests d'amplitude
    "metriques": {"fraction": 0.5, "exclure_constantes": True},
    # Part ecretee toleree a 1,0 par defaut : les series synthetiques de ces tests
    # sont uniformes et saturent vite, ce qui n'a rien a voir avec le reglage reel.
    # Le test du garde-fou repose la borne de la configuration du depot.
    "ecretage": {"bas": 0.0, "haut": 1.0, "part_max_toleree": 1.0},
    "onset": {"position_min": 0.40, "position_max": 0.60},
}

PROFILS = {
    "marche": {"type": "marche"},
    "rampe": {"type": "rampe", "duree_fraction": 0.25},
    "graduel": {"type": "graduel", "duree_fraction": 0.25},
    "recurrent": {"type": "recurrent", "periode_minutes": 10, "rapport_cyclique": 0.5},
}


def _scenario(kind: str = "marche", **surcharges: dict | float) -> DriftScenario:
    """Construit un scenario de test, surcharge cle par cle.

    `kind` nomme le profil temporel ; la cle `profil` surcharge ses parametres.
    """
    transformation: dict = {
        **TRANSFORMATION,
        **{k: v for k, v in surcharges.items() if k in ("level_shift_k", "scale_factor")},
    }
    for cle in ("metriques", "ecretage", "onset"):
        if cle in surcharges:
            transformation[cle] = {**TRANSFORMATION[cle], **surcharges[cle]}  # type: ignore[dict-item]
    profil = {**PROFILS[kind], **surcharges.get("profil", {})}  # type: ignore[dict-item]
    return DriftScenario.from_config({"name": f"test_{kind}", "profil": profil}, transformation)


# --- 1. L'instant de derive -------------------------------------------------


def test_t_d_tombe_dans_la_plage_de_la_configuration() -> None:
    """D15 : t_d uniforme dans [40 %, 60 %] du flux."""
    scenario = _scenario()
    for seed in range(50):
        t_d = drift_onset(1000, scenario, seed)
        assert 400 <= t_d < 600


def test_t_d_depend_de_la_seed() -> None:
    instants = {drift_onset(1000, _scenario(), seed) for seed in range(20)}
    assert len(instants) > 1


def test_profil_marche_vaut_zero_avant_et_un_apres() -> None:
    alpha, fin = alpha_profile(10, 4, _scenario("marche"), seed=42)
    assert alpha.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert fin == 4  # une marche n'a pas de transition


# --- 2. L'amplitude ---------------------------------------------------------


def test_amplitude_du_decalage_vaut_exactement_k_sigma() -> None:
    """Sur une metrique de sigma connu, le decalage mesure doit valoir k * sigma."""
    # train : alternance 0,2 / 0,4 -> mu = 0,3 et sigma = 0,1 exactement.
    train = np.tile(np.array([[0.2], [0.4]]), (50, 1))
    assert train.mean() == pytest.approx(0.3)
    assert train.std() == pytest.approx(0.1)

    test = np.full((100, 1), 0.3)
    label = np.zeros(100, dtype=np.int64)
    scenario = _scenario("marche", level_shift_k=2.0, metriques={"fraction": 1.0})
    flux = inject(train, test, label, scenario, seed=42)

    avant = flux.series[: flux.drift_index, 0]
    apres = flux.series[flux.drift_index :, 0]
    assert avant == pytest.approx(0.3)
    # 0,3 + 2 * 0,1 = 0,5
    assert apres == pytest.approx(0.5)


def test_le_changement_d_echelle_elargit_l_ecart_a_la_moyenne() -> None:
    """Avec k = 0 et echelle = 2, l'ecart a mu_train doit doubler."""
    train = np.tile(np.array([[0.2], [0.4]]), (50, 1))  # mu = 0,3
    test = np.tile(np.array([[0.2], [0.4]]), (50, 1))
    label = np.zeros(100, dtype=np.int64)
    scenario = _scenario("marche", level_shift_k=0.0, scale_factor=2.0, metriques={"fraction": 1.0})
    flux = inject(train, test, label, scenario, seed=42)

    apres = flux.series[flux.drift_index :, 0]
    # 0,3 + 2 * (0,2 - 0,3) = 0,1  et  0,3 + 2 * (0,4 - 0,3) = 0,5
    assert set(np.round(apres, 6).tolist()) == {0.1, 0.5}


def test_l_ecretage_borne_les_valeurs_et_la_part_est_journalisee() -> None:
    """Un k enorme sature la metrique : les valeurs sont bornees et la part comptee."""
    train = np.tile(np.array([[0.4], [0.6]]), (50, 1))  # mu = 0,5 ; sigma = 0,1
    test = np.full((100, 1), 0.9)
    label = np.zeros(100, dtype=np.int64)
    scenario = _scenario(
        "marche",
        level_shift_k=5.0,  # 0,9 + 0,5 = 1,4 -> ecrete a 1,0
        metriques={"fraction": 1.0},
    )
    flux = inject(train, test, label, scenario, seed=42)

    assert flux.series.max() == pytest.approx(1.0)
    # Toutes les cellules touchees sont ecretees, et elles seules.
    assert flux.part_ecretee == pytest.approx(1.0)
    assert flux.n_cellules_touchees == 100 - flux.drift_index


def test_ecretage_excessif_refuse_le_flux() -> None:
    """Garde-fou de calibrage : au-dela de la part toleree, l'injection echoue
    plutot que de produire une derive triviale a detecter."""
    train = np.tile(np.array([[0.4], [0.6]]), (50, 1))
    test = np.full((100, 1), 0.9)
    label = np.zeros(100, dtype=np.int64)
    scenario = _scenario(
        "marche",
        level_shift_k=5.0,
        metriques={"fraction": 1.0},
        ecretage={"part_max_toleree": 0.35},  # la borne de configs/scenarios
    )
    with pytest.raises(ValueError, match="ecretees"):
        inject(train, test, label, scenario, seed=42)


# --- 3. Reproductibilite ----------------------------------------------------


def test_meme_seed_meme_flux() -> None:
    """Condition de validite de la comparaison entre strategies (cadrage, §9)."""
    rng = np.random.default_rng(0)
    train = rng.random((200, 6))
    test = rng.random((300, 6))
    label = np.zeros(300, dtype=np.int64)
    scenario = _scenario("graduel")

    a = inject(train, test, label, scenario, seed=7)
    b = inject(train, test, label, scenario, seed=7)
    assert np.array_equal(a.series, b.series)
    assert a.drift_index == b.drift_index
    assert a.metriques == b.metriques
    assert np.array_equal(a.alpha, b.alpha)


def test_seed_differente_change_instant_et_metriques() -> None:
    rng = np.random.default_rng(0)
    train = rng.random((200, 10))
    test = rng.random((300, 10))
    label = np.zeros(300, dtype=np.int64)
    scenario = _scenario("marche")

    flux = [inject(train, test, label, scenario, seed=s) for s in range(10)]
    assert len({f.drift_index for f in flux}) > 1
    assert len({f.metriques for f in flux}) > 1


def test_deux_flux_differents_derivent_a_des_instants_differents() -> None:
    """Deux machines a graine egale ne doivent pas deriver au meme endroit.

    Sans cle de flux, toutes les machines derivaient a 41,8 % de leur flux pour
    la graine 42 : les graines n'exploraient qu'une poignee de positions et les
    machines d'une meme graine partageaient la configuration de derive.
    """
    scenario = _scenario()
    instants = {drift_onset(1000, scenario, 42, f"machine-1-{i}") for i in range(1, 9)}
    assert len(instants) > 1


def test_meme_cle_de_flux_meme_resultat() -> None:
    """La cle decorrele les machines sans casser la reproductibilite."""
    scenario = _scenario()
    assert drift_onset(1000, scenario, 42, "machine-2-1") == drift_onset(
        1000, scenario, 42, "machine-2-1"
    )


# --- 4. Les etiquettes sont intouchables ------------------------------------


def test_les_etiquettes_ne_sont_jamais_modifiees() -> None:
    """Invariant du cadrage §5 : les anomalies etiquetees d'origine sont
    conservees, avant et apres la derive."""
    rng = np.random.default_rng(1)
    train = rng.random((200, 5))
    test = rng.random((400, 5))
    label = (rng.random(400) < 0.1).astype(np.int64)

    for profil in ("marche", "rampe", "graduel", "recurrent"):
        flux = inject(train, test, label, _scenario(profil), seed=3)
        assert np.array_equal(flux.label, label)
        assert flux.label is not label  # copie, aucun partage de memoire


# --- 5. Les metriques constantes ne sont jamais touchees --------------------


def test_les_metriques_constantes_sont_exclues() -> None:
    train = np.zeros((100, 4))
    train[:, 1] = np.tile([0.2, 0.4], 50)
    train[:, 3] = np.tile([0.1, 0.9], 50)
    scenario = _scenario("marche", metriques={"fraction": 1.0})

    assert injectable_metrics(train, scenario).tolist() == [1, 3]
    assert select_metrics(train, scenario, seed=42).tolist() == [1, 3]


def test_les_metriques_constantes_restent_inchangees_dans_le_flux() -> None:
    train = np.zeros((100, 4))
    train[:, 1] = np.tile([0.2, 0.4], 50)
    train[:, 3] = np.tile([0.1, 0.9], 50)
    test = np.full((100, 4), 0.5)
    label = np.zeros(100, dtype=np.int64)
    scenario = _scenario("marche", metriques={"fraction": 1.0})

    flux = inject(train, test, label, scenario, seed=42)
    assert flux.series[:, 0] == pytest.approx(0.5)
    assert flux.series[:, 2] == pytest.approx(0.5)
    assert not np.allclose(flux.series[flux.drift_index :, 1], 0.5)


def test_part_de_metriques_touchees_respectee() -> None:
    rng = np.random.default_rng(2)
    train = rng.random((100, 20))  # 20 metriques injectables
    scenario = _scenario("marche", metriques={"fraction": 0.30})
    assert len(select_metrics(train, scenario, seed=42)) == 6


# --- 6. Les quatre profils temporels ----------------------------------------


def test_profil_rampe_monte_lineairement_puis_reste_a_un() -> None:
    # n = 100, duree_fraction 0,25 -> transition sur 25 points.
    alpha, fin = alpha_profile(100, 40, _scenario("rampe"), seed=42)
    assert alpha[:40].tolist() == [0.0] * 40
    assert alpha[40] == pytest.approx(1 / 25)
    assert alpha[64] == pytest.approx(1.0)
    assert alpha[65:] == pytest.approx(1.0)
    assert np.all(np.diff(alpha) >= 0.0)  # monotone : S2 ne revient jamais
    assert fin == 65


def test_profil_graduel_alterne_entre_les_deux_regimes() -> None:
    """S3 : alpha ne prend que 0 ou 1, et la part de 1 croit sur la transition."""
    alpha, fin = alpha_profile(1000, 400, _scenario("graduel"), seed=42)
    assert set(np.unique(alpha).tolist()) == {0.0, 1.0}
    assert alpha[:400].tolist() == [0.0] * 400
    # Transition sur 250 points, decoupee en deux moities.
    premiere = alpha[400:525].mean()
    seconde = alpha[525:650].mean()
    assert premiere < seconde
    assert alpha[650:].mean() == pytest.approx(1.0)
    assert fin == 650


def test_profil_graduel_differe_du_profil_rampe() -> None:
    """La distinction S2 / S3 de Gama et al. : meme duree, dynamique differente."""
    rampe, _ = alpha_profile(1000, 400, _scenario("rampe"), seed=42)
    graduel, _ = alpha_profile(1000, 400, _scenario("graduel"), seed=42)
    assert not np.array_equal(rampe, graduel)
    assert len(np.unique(rampe)) > 2
    assert len(np.unique(graduel)) == 2


def test_profil_recurrent_respecte_la_periode() -> None:
    # periode 10 min a 60 s par point -> 10 points, rapport 0,5 -> 5 hauts.
    alpha, fin = alpha_profile(
        40, 10, _scenario("recurrent"), seed=42, sampling_interval_seconds=60
    )
    assert alpha[:10].tolist() == [0.0] * 10
    assert alpha[10:20].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert alpha[20:30].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert fin == 39  # le regime alterne jusqu'au bout du flux


def test_profil_recurrent_tient_compte_de_l_intervalle_d_echantillonnage() -> None:
    """Une periode est exprimee en minutes, pas en points : a 30 s par point,
    10 minutes font 20 points."""
    alpha, _ = alpha_profile(60, 0, _scenario("recurrent"), seed=42, sampling_interval_seconds=30)
    assert alpha[:20].tolist() == [1.0] * 10 + [0.0] * 10


def test_un_regime_revient_en_recurrent_et_jamais_en_incremental() -> None:
    """Propriete qui distingue S4 des trois autres : apres t_d, le regime
    d'origine reapparait."""
    for profil in ("marche", "rampe"):
        alpha, _ = alpha_profile(200, 80, _scenario(profil), seed=42)
        assert np.all(alpha[81:] > 0.0)
    alpha, _ = alpha_profile(200, 80, _scenario("recurrent"), seed=42)
    assert np.any(alpha[81:] == 0.0)


# --- 7. Validation des configurations ---------------------------------------


def test_les_quatre_configs_du_depot_se_chargent() -> None:
    noms = ("s1_soudaine", "s2_incrementale", "s3_graduelle", "s4_recurrente")
    scenarios = [load_scenario(f"scenarios/{nom}.yaml") for nom in noms]
    assert [s.profil for s in scenarios] == ["marche", "rampe", "graduel", "recurrent"]
    # Invariant : les quatre scenarios partagent la meme transformation, sinon un
    # ecart entre scenarios ne serait plus attribuable au type de derive.
    assert len({(s.level_shift_k, s.scale_factor, s.metric_fraction) for s in scenarios}) == 1


def test_duree_de_transition_bornee_a_25_pourcent() -> None:
    with pytest.raises(ValueError, match="duree_fraction"):
        _scenario("rampe", profil={"duree_fraction": 0.40})


def test_decalage_negatif_refuse() -> None:
    with pytest.raises(ValueError, match="positif"):
        _scenario("marche", level_shift_k=-1.0)


def test_profil_inconnu_refuse() -> None:
    with pytest.raises(ValueError, match="inconnu"):
        DriftScenario.from_config({"name": "x", "profil": {"type": "sinusoidal"}}, TRANSFORMATION)


def test_metadonnees_journalisables() -> None:
    rng = np.random.default_rng(3)
    train = rng.random((100, 8))
    test = rng.random((200, 8))
    label = np.zeros(200, dtype=np.int64)
    flux = inject(train, test, label, _scenario("rampe"), seed=11)

    meta = flux.metadata()
    assert meta["scenario"] == "test_rampe"
    assert meta["seed"] == 11
    assert meta["n_points"] == 200
    assert 80 <= meta["drift_index"] < 120
    assert meta["n_metriques_touchees"] == 4
    assert meta["duree_transition_points"] == 50
