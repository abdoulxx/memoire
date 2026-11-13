"""Tests des strategies B1, P et A1, et de la conversion minutes -> fenetres.

Les proprietes verifiees ici sont celles qui font la difference entre les trois
bras de comparaison du chapitre 6. Si P declenchait sans attendre K fenetres, ou
ignorait son refroidissement, elle ne serait plus la strategie decrite au §6 et
les resultats ne mesureraient pas ce que le memoire affirme mesurer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.strategies.base import ValidationStats, WindowContext
from src.strategies.factory import build_strategy, minutes_to_windows
from src.strategies.periodic import PeriodicStrategy
from src.strategies.proposed import ProposedStrategy

# --------------------------------------------------------- conversion d'unites


def test_une_fenetre_vaut_une_minute_sur_le_smd() -> None:
    """Stride 1 et pas d'echantillonnage de 60 s : la conversion est l'identite."""
    assert minutes_to_windows(1440, stride=1, sampling_interval_seconds=60) == 1440


def test_la_conversion_tient_compte_du_stride() -> None:
    """Sur les sessions du serveur, une fenetre ne vaudra plus une minute."""
    assert minutes_to_windows(1440, stride=5, sampling_interval_seconds=60) == 288
    assert minutes_to_windows(60, stride=1, sampling_interval_seconds=30) == 120


def test_la_conversion_ne_renvoie_jamais_zero() -> None:
    """Arrondir une duree courte a zero fenetre ferait reentrainer a chaque fenetre."""
    assert minutes_to_windows(0.1, stride=10, sampling_interval_seconds=60) == 1


def test_une_fenetre_de_duree_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="nulle ou negative"):
        minutes_to_windows(10, stride=0, sampling_interval_seconds=60)


# ----------------------------------------------------------------------- B1


def _context(position: int, alert: bool = False, n_features: int = 4) -> WindowContext:
    return WindowContext(
        position=position,
        end_index=position,
        features=np.zeros(n_features),
        score=0.0,
        alert=alert,
        threshold=1.0,
    )


def test_b1_declenche_toutes_les_t_fenetres() -> None:
    """Et la premiere echeance tombe a T, pas a la fenetre 0."""
    strategie = PeriodicStrategy(period_windows=10, train_windows=50)
    declenchements = []
    for position in range(45):
        decision = strategie.on_window(_context(position))
        if decision.retrain:
            declenchements.append(position)
            strategie.on_fit(np.zeros((10, 4)), position)
    assert declenchements == [10, 20, 30, 40]


def test_b1_ne_reserve_aucune_fenetre() -> None:
    """Propriete de B1 : elle ne valide rien, donc elle n'a rien a reserver."""
    strategie = PeriodicStrategy(period_windows=10, train_windows=20)
    plan = strategie.propose(np.arange(100).reshape(50, 2), position=49)
    assert len(plan.train_X) == 20
    assert len(plan.reserved_X) == 0
    assert strategie.validate(_stats()).promote is True


# -------------------------------------------------------------- P : reglages


def _drift_config(**surcharges) -> dict:
    config = {
        "statistic": "ks",
        "n_bins": 10,
        "recent_windows": 20,
        "drifting_share": 0.50,
        "confirmation_windows": 5,
        "constant_tolerance": 0.0,
        "feature_threshold": {"method": "absolute", "absolute": 0.3},
        "alert_rate": {"enabled": False},
        "combination": "features",
    }
    config.update(surcharges)
    return config


def _proposed(**surcharges) -> ProposedStrategy:
    kwargs = {
        "drift_config": _drift_config(),
        "train_windows": 200,
        "reserved_windows": 40,
        "cooldown_windows": 50,
        "min_train_windows": 0,
        "min_history": 0,
    }
    kwargs.update(surcharges)
    return ProposedStrategy(**kwargs)


def _stats(**surcharges) -> ValidationStats:
    champs = {
        "position": 500,
        "n_history": 501,
        "n_train_windows": 160,
        "n_reserved_windows": 40,
        "challenger_alert_rate": 0.02,
        "champion_alert_rate": 0.35,
        "challenger_threshold": -0.4,
        "champion_threshold": -0.5,
        "challenger_fit_seconds": 0.1,
    }
    champs.update(surcharges)
    return ValidationStats(**champs)


def test_une_validation_sans_fenetre_reservee_est_refusee() -> None:
    """Valider sans fenetre reservee serait valider sur rien : la config est invalide."""
    with pytest.raises(ValueError, match="exige une fenetre reservee"):
        _proposed(reserved_windows=0)


def test_un_plancher_plus_grand_que_w_moins_v_est_refuse() -> None:
    """Sinon la strategie ne reentrainerait jamais, sans que rien ne l'indique."""
    with pytest.raises(ValueError, match="depasse ce que W - V peut fournir"):
        _proposed(train_windows=200, reserved_windows=40, min_train_windows=161)


def test_le_plancher_du_moniteur_est_applique_meme_si_la_config_se_tait() -> None:
    """Contrainte structurelle : le code l'impose quoi que dise le fichier."""
    strategie = _proposed(
        drift_config=_drift_config(
            recent_windows=100, feature_threshold={"method": "empirical_null"}
        ),
        min_train_windows=0,
    )
    assert strategie.min_train_windows == 100


# ------------------------------------------------- P : declenchement et attente


def _derive(strategie: ProposedStrategy, n: int, start: int = 0, loc: float = 6.0) -> list:
    """Fait defiler n fenetres franchement derivees et renvoie les decisions."""
    rng = np.random.default_rng(11)
    return [
        strategie.on_window(_context(start + offset, n_features=4))
        if loc is None
        else strategie.on_window(
            WindowContext(
                position=start + offset,
                end_index=start + offset,
                features=rng.normal(loc=loc, size=4),
                score=0.0,
                alert=False,
                threshold=1.0,
            )
        )
        for offset in range(n)
    ]


def _entraine(strategie: ProposedStrategy, seed: int = 10) -> None:
    rng = np.random.default_rng(seed)
    strategie.on_fit(rng.normal(size=(400, 4)), 0)


def test_p_ne_declenche_pas_avant_k_fenetres_de_signal() -> None:
    """Etape 2 du §6 : le declenchement suit la confirmation, qui suit K fenetres."""
    strategie = _proposed(cooldown_windows=0)
    _entraine(strategie)

    decisions = _derive(strategie, n=60, start=0)
    declenchements = [i for i, d in enumerate(decisions) if d.retrain]
    assert declenchements, "une derive a 6 ecarts-types doit finir par declencher"

    premier = declenchements[0]
    # Le signal doit avoir ete present pendant K fenetres avant le declenchement :
    # la fenetre recente du moniteur (R = 20) plus K = 5.
    assert premier >= 20 + 5 - 1
    # Un seul declenchement : la confirmation est consommee.
    assert len(declenchements) == 1


def test_p_respecte_le_refroidissement() -> None:
    """Etape 6 : aucun reentrainement pendant C fenetres apres une promotion."""
    strategie = _proposed(cooldown_windows=50)
    _entraine(strategie)

    decisions = _derive(strategie, n=60, start=0)
    premier = next(i for i, d in enumerate(decisions) if d.retrain)

    # Le harnais promeut : `on_fit` est rappele, ce qui demarre le refroidissement.
    strategie.on_fit(np.random.default_rng(12).normal(loc=6.0, size=(400, 4)), premier)

    # Les fenetres premier+1 a premier+49 sont dans le refroidissement.
    pendant = _derive(strategie, n=49, start=premier + 1, loc=-6.0)
    refroidis = [d for d in pendant if d.reason == "refroidissement"]
    assert not any(d.retrain for d in pendant)
    assert refroidis, "une confirmation pendant le refroidissement doit etre mise en attente"

    # A la fenetre premier+50, C fenetres se sont ecoulees : l'attente se solde.
    apres = strategie.on_window(_context(premier + 50))
    assert apres.retrain is True
    assert apres.reason == "derive_confirmee"


def test_une_confirmation_pendant_le_refroidissement_n_est_pas_perdue() -> None:
    """Sinon le comportement dependrait de l'instant de la derive, pas de la derive."""
    strategie = _proposed(cooldown_windows=10)
    _entraine(strategie)
    # Etat force : promotion a la fenetre 100, confirmation en attente.
    strategie._last_promotion = 100
    strategie._pending = {"confirmed_at": 101}

    assert strategie.on_window(_context(105)).reason == "refroidissement"
    assert strategie.on_window(_context(110)).retrain is True


def test_une_confirmation_trop_precoce_est_reportee_et_non_perdue() -> None:
    """Le plancher d'entrainement differe le declenchement, il ne l'annule pas."""
    strategie = _proposed(
        train_windows=200, reserved_windows=40, cooldown_windows=0, min_train_windows=120
    )
    _entraine(strategie)
    strategie._pending = {"confirmed_at": 50}

    tot = strategie.on_window(_context(50))
    assert tot.retrain is False
    assert tot.reason == "historique_insuffisant"
    assert tot.details["min_train_windows"] == 120

    # Plus tard, l'historique suffit : W - V = 200 - 40 = 160 >= 120.
    assert strategie.on_window(_context(400)).retrain is True


def test_le_bloc_d_entrainement_reserve_les_fenetres_les_plus_recentes() -> None:
    """Etape 3 : V est la fenetre la plus recente, celle du regime a servir."""
    strategie = _proposed(train_windows=100, reserved_windows=20)
    history = np.arange(600, dtype=np.float64).reshape(300, 2)
    plan = strategie.propose(history, position=299)

    assert len(plan.train_X) == 80
    assert len(plan.reserved_X) == 20
    np.testing.assert_array_equal(plan.train_X, history[200:280])
    np.testing.assert_array_equal(plan.reserved_X, history[280:])
    assert plan.details["n_excluded"] == 0  # `none` par defaut (D13)


# ------------------------------------------------------- P : garde-fou (etape 4)


def test_le_garde_fou_promeut_un_challenger_ordinaire() -> None:
    """Un challenger qui n'est pas manifestement inutilisable est promu.

    C'est le point delicat du §6 : la validation ne cherche pas le meilleur des
    deux modeles, faute d'etiquettes. Elle laisse donc passer un challenger sans
    preuve de superiorite.
    """
    verdict = _proposed(min_history=300).validate(_stats())
    assert verdict.promote is True
    assert verdict.reason == "garde_fou_passe"


def test_le_garde_fou_rejette_un_historique_trop_court() -> None:
    verdict = _proposed(min_history=300).validate(_stats(n_history=120))
    assert verdict.promote is False
    assert verdict.reason == "historique_trop_court"


def test_le_garde_fou_rejette_une_fenetre_reservee_vide() -> None:
    verdict = _proposed(min_history=0).validate(_stats(n_reserved_windows=0))
    assert verdict.promote is False
    assert verdict.reason == "fenetre_reservee_vide"


def test_le_garde_fou_rejette_un_seuil_degenere() -> None:
    verdict = _proposed(min_history=0).validate(_stats(challenger_threshold=float("nan")))
    assert verdict.promote is False
    assert verdict.reason == "seuil_degenere"


@pytest.mark.parametrize("taux", [0.0, 0.0005, 0.45, 1.0, float("nan")])
def test_le_garde_fou_rejette_un_taux_d_alertes_hors_bornes(taux: float) -> None:
    """Un modele qui n'alerte jamais ou qui alerte sur tout est inutilisable."""
    strategie = _proposed(min_history=0, alert_rate_bounds=(0.001, 0.30))
    verdict = strategie.validate(_stats(challenger_alert_rate=taux))
    assert verdict.promote is False
    assert verdict.reason == "taux_alertes_hors_bornes"


def test_a1_promeut_sans_condition() -> None:
    """L'ablation : meme declenchement, meme donnees, promotion inconditionnelle.

    C'est le seul ecart entre P et A1, ce qui rend l'effet de la validation
    isolable (hypothese H3).
    """
    a1 = _proposed(validation_enabled=False, reserved_windows=0, min_history=300)
    verdict = a1.validate(_stats(n_history=1, n_reserved_windows=0, challenger_alert_rate=1.0))
    assert verdict.promote is True
    assert verdict.reason == "validation_desactivee_a1"


# --------------------------------------------------------------------- factory


FEATURES_CONFIG = {"stride": 1}


def _build(config: dict):
    return build_strategy(
        strategy_config=config,
        features_config=FEATURES_CONFIG,
        drift_config=_drift_config(),
        sampling_interval_seconds=60,
    )


def test_la_factory_convertit_les_minutes_en_fenetres() -> None:
    strategie = _build(
        {
            "strategy": "proposed",
            "name": "P",
            "train_minutes": 1440,
            "reserved_minutes": 120,
            "cooldown_minutes": 360,
            "min_train_minutes": 500,
        }
    )
    assert isinstance(strategie, ProposedStrategy)
    assert (strategie.train_windows, strategie.reserved_windows) == (1440, 120)
    assert strategie.cooldown_windows == 360
    assert strategie.min_train_windows == 500
    assert strategie.name == "P"


def test_la_factory_applique_les_surcharges_de_derive() -> None:
    """Une strategie peut surcharger un seuil sans modifier `configs/drift.yaml`."""
    strategie = _build(
        {
            "strategy": "proposed",
            "train_minutes": 1440,
            "reserved_minutes": 120,
            "cooldown_minutes": 0,
            "drift_overrides": {"confirmation_windows": 99},
        }
    )
    assert strategie.monitor.k == 99
    # La configuration de base n'a pas ete modifiee en place.
    assert _drift_config()["confirmation_windows"] == 5


def test_la_factory_refuse_une_strategie_inconnue() -> None:
    with pytest.raises(ValueError, match="Strategie inconnue"):
        _build({"strategy": "magique"})


# ------------------------------------------- parite des configurations figees


def test_les_deux_bras_de_l_ablation_ne_divergent_que_sur_la_validation() -> None:
    """A1 doit etre P moins l'etape 4, et rien d'autre (D30).

    C'est la condition de validite du test de H3. Si `train_minutes`,
    `reserved_minutes`, `cooldown_minutes` ou `min_train_minutes` divergeaient
    entre les deux fichiers, l'ecart P - A1 melangerait l'effet de la validation
    a celui d'un autre parametre, et la conclusion du 6.2.4 ne tiendrait pas.
    Une relecture ne suffit pas a garantir cette parite dans le temps : les deux
    fichiers sont longs et commentes separement, d'ou ce test.
    """
    propose = load_config("strategies/proposed.yaml")
    ablation = load_config("strategies/ablation.yaml")

    divergentes = {
        cle for cle in set(propose) | set(ablation) if propose.get(cle) != ablation.get(cle)
    }
    assert divergentes == {"name", "validation"}

    # Et dans le bloc `validation`, seul l'interrupteur change : les bornes du
    # garde-fou restent identiques, meme si A1 ne les applique pas.
    validation_propose = dict(propose["validation"])
    validation_ablation = dict(ablation["validation"])
    assert validation_propose.pop("enabled") is True
    assert validation_ablation.pop("enabled") is False
    assert validation_propose == validation_ablation


def test_le_refroidissement_ne_peut_pas_mordre_avant_le_moniteur() -> None:
    """Consequence du gel de D29, verifiee plutot que supposee.

    `fit_reference` reconstruit le tracker et remet la persistance a zero : apres
    une promotion il faut R fenetres pour que le signal redevienne pret, puis K
    fenetres consecutives pour confirmer. Si C devenait superieur a R + K, le
    refroidissement deviendrait la contrainte active et le nombre de
    reentrainements de P changerait sans qu'aucune decision ne l'ait voulu.
    Le test fixe l'ordre de grandeur en place, pas la valeur exacte.
    """
    drift = load_config("drift.yaml")
    propose = load_config("strategies/proposed.yaml")
    plancher_moniteur = int(drift["recent_windows"]) + int(drift["confirmation_windows"])
    assert propose["cooldown_minutes"] <= plancher_moniteur


def test_les_bornes_figees_sont_celles_calibrees_et_non_celles_du_cadrage() -> None:
    """Le test qui manquait, et dont l'absence a coute une campagne (D43).

    D14 a calibre les bornes du garde-fou le 26/10/2026 sur les 473 challengers
    du perimetre de mise au point et retenu 256 / 0,0 / 0,90. D30 les a pourtant
    figees a 300 / 0,001 / 0,30 — les valeurs du **cadrage**, celles que D14
    declare justement faire « plus de mal que de bien » — tout en citant D14.
    L'erreur a traverse toute la campagne M3 : 54 challengers rejetes sur les 12
    machines d'evaluation, dont aucun n'aurait du l'etre, et un verdict de H3
    mesurant l'effet de bornes que la mise au point avait ecartees.

    Une relecture n'avait pas suffi : la contradiction etait entre deux fichiers
    de documentation et deux fichiers de configuration, et chacun pris seul
    paraissait coherent. Ce test ancre les valeurs calibrees dans le code, ou
    elles ne peuvent plus deriver en silence. Si une nouvelle mesure les change,
    il faut changer ce test **et** ecrire la decision correspondante : c'est
    exactement le garde-fou qui a manque.
    """
    calibrees_d14 = {"min_history": 256, "alert_rate_min": 0.0, "alert_rate_max": 0.90}

    for nom in ("proposed", "ablation"):
        validation = load_config(f"strategies/{nom}.yaml")["validation"]
        for cle, attendu in calibrees_d14.items():
            assert float(validation[cle]) == attendu, (nom, cle, validation[cle])

    # Les valeurs du cadrage ne doivent plus apparaitre, ni dans les
    # configurations, ni comme defaut du code : un fichier incomplet retomberait
    # sinon sur les bornes ecartees sans que rien ne le signale.
    strategie = build_strategy(
        load_config("strategies/proposed.yaml"),
        load_config("features.yaml"),
        load_config("drift.yaml"),
        60,
    )
    assert strategie.min_history == 256
    assert (strategie.alert_rate_min, strategie.alert_rate_max) == (0.0, 0.90)
