"""Registre de modeles (livrable M5, phase A).

Ces tests tournent contre un **vrai** registre MLflow adosse a sqlite, pas
contre un double. Un faux client ne dirait rien de ce qui compte ici : que
l'alias se deplace vraiment, qu'un modele relu produit les memes scores que
l'original, et que les etiquettes survivent a l'aller-retour. Le prix est une
base de donnees par module de test (migrations alembic a la premiere ouverture) ;
elle est creee une fois et partagee, chaque test travaillant sous un nom de
modele distinct.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.signature import SignatureFeatures
from src.models.isolation_forest import IsolationForestDetector
from src.tracking.registry import ErreurRegistre, RegistreModeles, ReglagesRegistre

N_FEATURES = 8


@pytest.fixture(scope="module")
def backend(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Une base sqlite et un dossier d'artefacts, partages par le module."""
    dossier = tmp_path_factory.mktemp("registre")
    base = f"sqlite:///{(dossier / 'mlflow.db').as_posix()}"
    artefacts = (dossier / "artefacts").as_uri()
    return base, artefacts


@pytest.fixture
def registre(backend: tuple[str, str], request: pytest.FixtureRequest) -> RegistreModeles:
    base, artefacts = backend
    # Un nom de modele par test : les versions d'un test ne doivent pas
    # apparaitre dans l'historique d'un autre.
    nom = f"detecteur-{request.node.name[:40]}"
    return RegistreModeles(
        ReglagesRegistre(
            tracking_uri=base,
            experiment="tests-registre",
            model_name=nom,
            artifact_location=artefacts,
        )
    )


def _signature(n_features: int = N_FEATURES, **surcharges: int) -> SignatureFeatures:
    base: dict = {"window_size": 10, "stride": 1, "sampling_interval_seconds": 60}
    base.update(surcharges)
    return SignatureFeatures(
        feature_names=tuple(f"m{i:02d}_mean" for i in range(n_features)), **base
    )


def _detecteur(seed: int = 1, n_windows: int = 200) -> tuple[IsolationForestDetector, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_windows, N_FEATURES))
    detecteur = IsolationForestDetector(
        params={"n_estimators": 20}, threshold_percentile=99.0, random_state=seed
    )
    detecteur.fit(X)
    return detecteur, X


# ----------------------------------------------------------------- reglages


def test_les_reglages_se_lisent_dans_la_configuration_du_projet() -> None:
    from src.config import load_config

    reglages = ReglagesRegistre.depuis_config(load_config("serving.yaml"), env={})
    assert reglages.alias == "champion"
    assert reglages.timeout_s > 0
    assert reglages.model_name


def test_la_variable_d_environnement_a_la_priorite_sur_le_fichier() -> None:
    # L'URL depend du deploiement : elle n'a rien a faire dans un fichier suivi
    # par git, et la variable doit pouvoir la remplacer sans toucher au depot.
    config = {"registry": {"tracking_uri": "http://du-fichier:5000"}}
    reglages = ReglagesRegistre.depuis_config(
        config, env={"MLFLOW_TRACKING_URI": "http://de-l-environnement:5000"}
    )
    assert reglages.tracking_uri == "http://de-l-environnement:5000"


def test_sans_uri_le_message_dit_ou_la_renseigner() -> None:
    with pytest.raises(ErreurRegistre, match="MLFLOW_TRACKING_URI"):
        ReglagesRegistre.depuis_config({"registry": {"experiment": "x"}}, env={})


def test_une_section_registry_absente_est_refusee() -> None:
    with pytest.raises(ErreurRegistre, match="serving.yaml"):
        ReglagesRegistre.depuis_config({}, env={})


@pytest.mark.parametrize(
    "bloc", [{"tracking_uri": "x", "timeout_s": 0}, {"tracking_uri": "x", "max_retries": -1}]
)
def test_des_bornes_absurdes_sont_refusees(bloc: dict) -> None:
    # Un timeout nul ou negatif ne borne rien : il faut le dire a la lecture de
    # la configuration, pas decouvrir la boucle bloquee en production.
    with pytest.raises(ErreurRegistre):
        ReglagesRegistre.depuis_config({"registry": bloc}, env={})


# ------------------------------------------------------------ registre neuf


def test_un_registre_neuf_repond_sans_champion_ni_historique(registre: RegistreModeles) -> None:
    # Distinction essentielle : « rien n'a encore ete promu » n'est pas une
    # panne. Si les deux cas se confondaient, le service refuserait de demarrer
    # la premiere fois, ou demarrerait sans modele le jour d'une panne.
    registre.verifier()
    assert registre.champion() is None
    assert registre.historique() == []


def test_charger_le_champion_sans_champion_dit_quoi_lancer(registre: RegistreModeles) -> None:
    with pytest.raises(ErreurRegistre, match="registry-init"):
        registre.charger_champion(_signature())


def test_un_registre_injoignable_est_une_erreur_explicite(monkeypatch: pytest.MonkeyPatch) -> None:
    # `setdefault` dans le client : on efface d'abord les bornes posees par un
    # test precedent, sinon ce test attendrait le timeout de l'autre.
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    injoignable = RegistreModeles(
        ReglagesRegistre(
            tracking_uri="http://127.0.0.1:1",
            experiment="tests-registre",
            model_name="detecteur-injoignable",
            timeout_s=2,
            max_retries=0,
        )
    )
    with pytest.raises(ErreurRegistre, match="MLFLOW_TRACKING_URI"):
        injoignable.verifier()
    # Et surtout : le champion ne vaut pas None quand le registre est muet.
    with pytest.raises(ErreurRegistre):
        injoignable.champion()


# --------------------------------------------------- enregistrer et promouvoir


def test_enregistrer_ne_met_pas_en_service(registre: RegistreModeles) -> None:
    # L'enregistrement precede la validation, et la plupart des challengers sont
    # rejetes : une version enregistree ne doit jamais etre servie du seul fait
    # d'exister.
    detecteur, _ = _detecteur()
    version = registre.enregistrer(detecteur, signature=_signature(), raison="challenger")
    assert version.version == 1
    assert not version.est_champion
    assert registre.champion() is None


def test_promouvoir_deplace_l_alias_et_c_est_la_mise_en_service(
    registre: RegistreModeles,
) -> None:
    premier, _ = _detecteur(seed=1)
    second, _ = _detecteur(seed=2)
    v1 = registre.enregistrer(premier, signature=_signature(), raison="initial_fit")
    registre.promouvoir(v1)
    assert registre.champion() is not None
    assert registre.champion().version == 1  # type: ignore[union-attr]

    v2 = registre.enregistrer(second, signature=_signature(), raison="drift_retrain")
    assert registre.champion().version == 1  # type: ignore[union-attr]
    promue = registre.promouvoir(v2)
    assert promue.est_champion
    assert registre.champion().version == 2  # type: ignore[union-attr]


def test_un_modele_non_entraine_n_est_pas_enregistrable(registre: RegistreModeles) -> None:
    vide = IsolationForestDetector(params={"n_estimators": 10}, threshold_percentile=99.0)
    with pytest.raises(ErreurRegistre, match="non entraine"):
        registre.enregistrer(vide, signature=_signature(), raison="initial_fit")


def test_la_raison_et_les_etiquettes_survivent_a_l_aller_retour(
    registre: RegistreModeles,
) -> None:
    detecteur, _ = _detecteur()
    registre.enregistrer(
        detecteur,
        signature=_signature(),
        raison="drift_retrain",
        parametres={"seed": 7},
        metriques={"n_train_points": 1234.0},
        tags={"session": "run-2026-11-11", "source": "serveur"},
    )
    relue = registre.historique()[0]
    assert relue.raison == "drift_retrain"
    assert relue.tags is not None
    assert relue.tags["session"] == "run-2026-11-11"
    assert relue.n_train_windows == detecteur.n_train_windows


def test_l_historique_est_ordonne_et_marque_le_champion(registre: RegistreModeles) -> None:
    for i in range(3):
        detecteur, _ = _detecteur(seed=i + 1)
        version = registre.enregistrer(detecteur, signature=_signature(), raison=f"r{i}")
        if i == 1:
            registre.promouvoir(version)
    versions = registre.historique()
    assert [v.version for v in versions] == [3, 2, 1]
    assert [v.est_champion for v in versions] == [False, True, False]


# ------------------------------------------------------------------ charger


def test_le_modele_recharge_donne_exactement_les_memes_scores(
    registre: RegistreModeles,
) -> None:
    # La vraie garantie derriere la decision d'enregistrer le detecteur entier
    # (standardisation + foret + seuil) plutot qu'un flavour sklearn : si le
    # seuil ou le scaler ne faisaient pas l'aller-retour, les scores relus
    # resteraient plausibles tout en etant faux.
    detecteur, X = _detecteur()
    version = registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")
    registre.promouvoir(version)

    recharge, info = registre.charger_champion(_signature())

    assert info.version == version.version
    assert info.est_champion
    assert recharge.threshold == pytest.approx(detecteur.threshold)
    assert recharge.n_train_windows == detecteur.n_train_windows
    np.testing.assert_allclose(recharge.score(X), detecteur.score(X))
    np.testing.assert_array_equal(
        recharge.score(X) >= recharge.threshold, detecteur.score(X) >= detecteur.threshold
    )


def test_une_signature_de_largeur_differente_est_refusee(registre: RegistreModeles) -> None:
    detecteur, _ = _detecteur()
    registre.promouvoir(
        registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")
    )
    with pytest.raises(ErreurRegistre, match="features"):
        registre.charger_champion(_signature(n_features=N_FEATURES + 1))


def test_un_fenetrage_different_est_refuse(registre: RegistreModeles) -> None:
    # Meme nombre de colonnes : seule la signature enregistree peut detecter ce
    # changement, et c'est exactement le cas qu'un controle de forme laisserait
    # passer.
    detecteur, _ = _detecteur()
    registre.promouvoir(
        registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")
    )
    with pytest.raises(ErreurRegistre, match="Fenetrage different"):
        registre.charger_champion(_signature(window_size=30))


def test_un_pas_d_echantillonnage_different_est_refuse(registre: RegistreModeles) -> None:
    detecteur, _ = _detecteur()
    registre.promouvoir(
        registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")
    )
    with pytest.raises(ErreurRegistre, match="Pas d'echantillonnage"):
        registre.charger_champion(_signature(sampling_interval_seconds=15))


def test_une_version_sans_signature_ne_peut_pas_etre_mise_en_service(
    registre: RegistreModeles,
) -> None:
    # Cas d'un modele enregistre a la main ou par un autre outil : on ne sait
    # pas ce qu'il attend, donc on ne le sert pas.
    detecteur, _ = _detecteur()
    version = registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")
    for etiquette in ("features_empreinte", "features_n"):
        registre.client.delete_model_version_tag(
            registre.reglages.model_name, str(version.version), etiquette
        )
    with pytest.raises(ErreurRegistre, match="aucune signature"):
        registre.charger(version.version, _signature())


def test_charger_sans_signature_attendue_ne_verifie_rien(registre: RegistreModeles) -> None:
    # Utile pour l'analyse hors ligne d'une ancienne version (metrique
    # « promotions degradantes » du §7) : on relit un modele pour le scorer sur
    # des traces, pas pour le mettre en service.
    detecteur, X = _detecteur()
    version = registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")
    relu, _ = registre.charger(version.version)
    np.testing.assert_allclose(relu.score(X), detecteur.score(X))


def test_charger_une_version_inexistante_echoue_clairement(registre: RegistreModeles) -> None:
    with pytest.raises(ErreurRegistre, match="version 99"):
        registre.charger(99)


# --------------------------------------------------------------------- reference
# La matrice d'entrainement est la reference du moniteur de derive. Elle n'a
# aucun effet sur les scores, donc rien ne la signale si elle manque : le
# service recalerait son moniteur sur les donnees du moment et comparerait le
# present a lui-meme. Ces tests sont le seul endroit ou cette absence se voit.


def test_la_reference_fait_l_aller_retour_sans_alteration(registre: RegistreModeles) -> None:
    detecteur, X = _detecteur()
    version = registre.enregistrer(
        detecteur, signature=_signature(), raison="initial_fit", reference_X=X
    )

    assert version.a_une_reference
    assert version.reference_n_windows == len(X)
    relue = registre.charger_reference(version)
    assert relue.shape == X.shape
    np.testing.assert_allclose(relue, X)


def test_une_version_sans_reference_le_dit_et_refuse_de_la_rendre(
    registre: RegistreModeles,
) -> None:
    detecteur, _ = _detecteur()
    version = registre.enregistrer(detecteur, signature=_signature(), raison="initial_fit")

    assert not version.a_une_reference
    assert version.reference_n_windows == 0
    with pytest.raises(ErreurRegistre, match="moniteur de derive"):
        registre.charger_reference(version)


def test_la_reference_se_lit_aussi_par_numero_de_version(registre: RegistreModeles) -> None:
    detecteur, X = _detecteur()
    version = registre.enregistrer(
        detecteur, signature=_signature(), raison="initial_fit", reference_X=X
    )

    np.testing.assert_allclose(registre.charger_reference(int(version.version)), X)


def test_la_reference_d_une_version_inexistante_est_une_erreur_explicite(
    registre: RegistreModeles,
) -> None:
    with pytest.raises(ErreurRegistre, match="version 99"):
        registre.charger_reference(99)


def test_chaque_version_porte_sa_propre_reference(registre: RegistreModeles) -> None:
    """Deux promotions successives n'entrainent pas sur les memes donnees.

    Si les references se melangeaient, le moniteur du champion promu
    comparerait les fenetres recentes a l'entrainement du modele precedent :
    une derive deja absorbee par le reentrainement resterait signalee.
    """
    premier, X1 = _detecteur(seed=1, n_windows=200)
    second, X2 = _detecteur(seed=2, n_windows=150)
    v1 = registre.enregistrer(premier, signature=_signature(), raison="initial_fit", reference_X=X1)
    v2 = registre.enregistrer(
        second, signature=_signature(), raison="drift_retrain", reference_X=X2
    )

    np.testing.assert_allclose(registre.charger_reference(v1), X1)
    np.testing.assert_allclose(registre.charger_reference(v2), X2)
