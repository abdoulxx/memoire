"""Tests de la boucle de detection en ligne (M5, phase B).

Ce que ces tests protegent, et pourquoi aucun autre controle ne le ferait :

- **Les features en ligne sont celles du rejeu.** C'est l'invariant central du
  chapitre 5 : si la boucle agregeait ses fenetres autrement que `make_windows`,
  le modele scorerait des colonnes qui n'ont plus le meme sens et ne leverait
  rien. Un test compare donc ligne par ligne les features de la boucle a celles
  que le harnais produirait sur les memes points bruts.
- **Le prechargement ne score pas.** Une boucle qui scorerait son historique de
  demarrage emettrait des alertes sur des evenements vieux de plusieurs heures.
  Elles seraient indiscernables des vraies dans le journal, et le §7 les
  compterait en fausses alertes.
- **Le registre fait foi.** Si la promotion echoue, le champion en memoire ne
  doit pas changer : sinon le service tournerait sur un modele que le registre
  ignore, et les captures du chapitre 5 affirmeraient quelque chose de faux.

La boucle est testee avec un faux Prometheus et une horloge que les tests
avancent eux-memes — mais avec un **vrai** registre sqlite : un registre
factice ne dirait rien de ce qui compte ici.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from datasets.serveur.settings import ReglagesServeur
from src.config import load_config
from src.features.signature import signature_depuis_config
from src.features.windowing import make_windows
from src.models.isolation_forest import build_detector
from src.serving.loop import (
    EVENEMENTS,
    BoucleEnLigne,
    ErreurBoucle,
    JournalEvenements,
    ReglagesBoucle,
)
from src.serving.prometheus import ErreurPrometheus, SerieTemporelle, grille
from src.strategies.base import Decision, TrainingPlan, ValidationStats, Verdict, WindowContext
from src.tracking.registry import ErreurRegistre, RegistreModeles, ReglagesRegistre

PAS = 60
METRIQUES = ("cpu", "memoire", "latence")
FENETRE = 5
N_FEATURES = len(METRIQUES) * 4  # last, mean, std, max


# --------------------------------------------------------------------- doubles


@dataclass
class Horloge:
    """Horloge que le test avance lui-meme : aucun `sleep` reel dans la suite."""

    t: float

    def __call__(self) -> float:
        return self.t

    def avancer(self, secondes: float) -> None:
        self.t += secondes


@dataclass
class PrometheusSynthetique:
    """Faux Prometheus : une sinusoide par metrique, avec trous et pannes au choix."""

    trous: set[int] = field(default_factory=set)
    pannes: int = 0
    decalage: float = 0.0
    bascule_ts: int = 0
    requetes: int = 0

    def valeur(self, nom: str, instant: int) -> float:
        base = {"cpu": 0.30, "memoire": 0.50, "latence": 0.20}[nom]
        phase = {"cpu": 0.0, "memoire": 1.0, "latence": 2.0}[nom]
        valeur = base + 0.02 * np.sin((instant / PAS + phase) / 7.0)
        if self.bascule_ts and instant >= self.bascule_ts:
            valeur += self.decalage
        return float(valeur)

    def interroger_plage(
        self, expression: str, debut: float, fin: float, pas: int
    ) -> SerieTemporelle:
        self.requetes += 1
        if self.pannes > 0:
            self.pannes -= 1
            raise ErreurPrometheus("Prometheus injoignable (panne simulee).")
        nom = expression  # les expressions du test sont les noms des metriques
        instants = grille(debut, fin, pas)
        valeurs = np.array(
            [np.nan if int(t) in self.trous else self.valeur(nom, int(t)) for t in instants],
            dtype=np.float64,
        )
        return SerieTemporelle(instants=instants, valeurs=valeurs)


@dataclass
class StrategieEspionne:
    """Strategie minimale, pilotee par le test. Enregistre ce qu'elle a vu."""

    name: str = "espionne"
    reentrainer_a: set[int] = field(default_factory=set)
    promouvoir: bool = True
    evenements: tuple[str, ...] = ()
    train_windows: int = 80
    reserved_windows: int = 10
    fenetres_vues: list[WindowContext] = field(default_factory=list)
    references: list[np.ndarray] = field(default_factory=list)

    def on_window(self, context: WindowContext) -> Decision:
        self.fenetres_vues.append(context)
        if context.position in self.reentrainer_a:
            return Decision(retrain=True, reason="derive_confirmee", events=self.evenements)
        return Decision(events=self.evenements)

    def propose(self, history: np.ndarray, position: int) -> TrainingPlan:
        reserve = history[-self.reserved_windows :]
        entrainement = history[
            -(self.train_windows + self.reserved_windows) : -self.reserved_windows
        ]
        return TrainingPlan(train_X=entrainement, reserved_X=reserve, details={"source": "test"})

    def validate(self, stats: ValidationStats) -> Verdict:
        return Verdict(promote=self.promouvoir, reason="test")

    def on_fit(self, training_X: np.ndarray, position: int) -> None:
        self.references.append(np.asarray(training_X))


# -------------------------------------------------------------------- fixtures


@pytest.fixture
def reglages_serveur(tmp_path: Path) -> ReglagesServeur:
    return ReglagesServeur.depuis_config(
        {
            "prometheus": {"url": "http://exemple.invalide", "step_seconds": PAS},
            # Les expressions sont les noms : le faux Prometheus les reconnait
            # ainsi, et la boucle ne fait que les transmettre.
            "metriques": [{"nom": nom, "expr": nom, "obligatoire": True} for nom in METRIQUES],
            "trous": {"max_points_consecutifs": 2},
        },
        tmp_path,
    )


@pytest.fixture
def features_config() -> dict[str, Any]:
    return {
        "window_size": FENETRE,
        "stride": 1,
        "aggregates": ["last", "mean", "std", "max"],
        "scaling": {"method": "standard"},
    }


@pytest.fixture
def model_config() -> dict[str, Any]:
    return {
        "model": "isolation_forest",
        "isolation_forest": {"n_estimators": 20, "max_samples": 64, "n_jobs": 1},
        "threshold": {"method": "train_percentile", "percentile": 99.0},
    }


@pytest.fixture
def reglages_boucle(tmp_path: Path) -> ReglagesBoucle:
    return ReglagesBoucle(
        source="datasets/serveur.yaml",
        strategie="strategies/proposed.yaml",
        retard_s=10,
        rattrapage_max_points=30,
        prefill_fenetres=120,
        historique_max_fenetres=500,
        dossier_journal=tmp_path / "journaux",
    )


@pytest.fixture
def registre(tmp_path: Path, request: pytest.FixtureRequest) -> RegistreModeles:
    base = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return RegistreModeles(
        ReglagesRegistre(
            tracking_uri=base,
            experiment="tests-boucle",
            model_name=f"detecteur-{request.node.name[:40]}",
            artifact_location=(tmp_path / "artefacts").as_uri(),
        )
    )


def _matrice(instants: np.ndarray, client: PrometheusSynthetique) -> np.ndarray:
    """Les memes points bruts que ceux que la boucle lira, en colonnes."""
    return np.column_stack(
        [np.array([client.valeur(nom, int(t)) for t in instants]) for nom in METRIQUES]
    )


def _champion(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    client: PrometheusSynthetique,
    *,
    avec_reference: bool = True,
    n_points: int = 600,
    fin_ts: int = 1_700_000_000,
) -> np.ndarray:
    """Entraine et promeut un champion initial. Rend sa matrice de reference."""
    instants = np.arange(fin_ts - (n_points - 1) * PAS, fin_ts + PAS, PAS, dtype=np.int64)
    fenetres = make_windows(
        _matrice(instants, client), features_config, metric_names=list(METRIQUES)
    )
    detecteur = build_detector(model_config, features_config, 7)
    detecteur.fit(fenetres.X)
    signature = signature_depuis_config(reglages_serveur.noms, features_config, PAS)
    version = registre.enregistrer(
        detecteur,
        signature=signature,
        raison="initial_fit",
        reference_X=fenetres.X if avec_reference else None,
    )
    registre.promouvoir(version)
    return np.asarray(fenetres.X)


def _lignes(journal: Path) -> list[dict[str, Any]]:
    return [json.loads(ligne) for ligne in journal.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------- reglages


def test_les_reglages_de_boucle_se_lisent_dans_la_configuration_du_projet() -> None:
    reglages = ReglagesBoucle.depuis_config(load_config("serving.yaml"))

    assert reglages.retard_s > 0
    assert reglages.rattrapage_max_points >= 1
    # Le prechargement doit depasser le garde-fou « historique trop court » du
    # §6, sinon un redemarrage interdit tout reentrainement pendant des heures.
    minimum = int(
        load_config("strategies/proposed.yaml").get("validation", {}).get("min_history", 300)
    )
    assert reglages.prefill_fenetres >= minimum
    assert reglages.historique_max_fenetres > reglages.prefill_fenetres
    assert reglages.dossier_journal.name == "online"


def test_une_section_boucle_absente_est_refusee() -> None:
    with pytest.raises(ErreurBoucle, match="section `boucle`"):
        ReglagesBoucle.depuis_config({"registry": {}})


@pytest.mark.parametrize(
    "bloc",
    [
        {"retard_s": -1},
        {"rattrapage_max_points": 0},
        {"prefill_fenetres": -5},
        {"historique_max_fenetres": 0},
    ],
)
def test_des_bornes_absurdes_sont_refusees(bloc: dict[str, int]) -> None:
    config = {"boucle": {"retard_s": 10, **bloc}}
    with pytest.raises(ErreurBoucle):
        ReglagesBoucle.depuis_config(config)


# -------------------------------------------------------------------- journal


def test_le_journal_ecrit_une_ligne_par_evenement_au_schema_du_paragraphe_9(
    tmp_path: Path,
) -> None:
    chemin = tmp_path / "events.jsonl"
    with JournalEvenements(chemin=chemin, run_id="run-1", strategie="proposed") as journal:
        journal.ecrire("alert", score=1.5)
        journal.ecrire("promoted", version="3")

    lignes = _lignes(chemin)
    assert [ligne["event"] for ligne in lignes] == ["alert", "promoted"]
    for ligne in lignes:
        assert set(ligne) == {"ts", "run_id", "strategy", "event", "details"}
        assert ligne["run_id"] == "run-1"
        assert ligne["strategy"] == "proposed"
    assert lignes[0]["details"] == {"score": 1.5}


def test_le_journal_refuse_un_evenement_hors_vocabulaire(tmp_path: Path) -> None:
    """Un nom invente serait invisible dans Grafana comme dans Telegram."""
    with (
        JournalEvenements(chemin=tmp_path / "e.jsonl", run_id="r", strategie="p") as journal,
        pytest.raises(ErreurBoucle, match="vocabulaire"),
    ):
        journal.ecrire("tout_va_bien")


def test_le_vocabulaire_contient_les_sept_evenements_du_cadrage() -> None:
    attendus = {
        "alert",
        "drift_signal",
        "drift_confirmed",
        "retrain_start",
        "retrain_end",
        "promoted",
        "rejected",
    }
    assert attendus <= set(EVENEMENTS)


# ------------------------------------------------------------------ demarrage


def _service(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    client: PrometheusSynthetique,
    horloge: Horloge,
    strategy: Any,
    **surcharges: Any,
) -> BoucleEnLigne:
    return BoucleEnLigne(
        client=client,
        registre=registre,
        reglages_serveur=reglages_serveur,
        reglages_boucle=reglages_boucle,
        strategy=strategy,
        features_config=features_config,
        model_config=model_config,
        seed=7,
        run_id="test",
        horloge=horloge,
        dormir=lambda _: None,
        **surcharges,
    )


Service = tuple[BoucleEnLigne, PrometheusSynthetique, Horloge, StrategieEspionne, np.ndarray]


@pytest.fixture
def service(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
) -> Service:
    """Un service pret a demarrer, avec son champion deja promu dans le registre."""
    client = PrometheusSynthetique()
    reference = _champion(
        registre, reglages_serveur, features_config, model_config, client, fin_ts=1_700_000_000
    )
    horloge = Horloge(t=1_700_040_000 + reglages_boucle.retard_s)
    strategy = StrategieEspionne()
    boucle = _service(
        registre,
        reglages_serveur,
        reglages_boucle,
        features_config,
        model_config,
        client,
        horloge,
        strategy,
    )
    return boucle, client, horloge, strategy, reference


def test_le_demarrage_charge_le_champion_et_sa_reference_depuis_le_registre(
    service: Service,
) -> None:
    """La reference du moniteur vient du registre, pas des fenetres du moment.

    Si elle etait recalculee sur les donnees recentes, le moniteur comparerait
    le present a lui-meme et une derive survenue pendant un arret du service
    serait invisible, definitivement.
    """
    boucle, _, _, strategy, reference = service
    boucle.demarrer()

    assert boucle.champion is not None
    assert boucle.version.version == 1
    assert len(strategy.references) == 1
    np.testing.assert_allclose(strategy.references[0], reference)
    boucle.arreter()


def test_un_champion_sans_reference_fait_refuser_le_demarrage(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    client = PrometheusSynthetique()
    _champion(
        registre,
        reglages_serveur,
        features_config,
        model_config,
        client,
        avec_reference=False,
    )
    boucle = _service(
        registre,
        reglages_serveur,
        reglages_boucle,
        features_config,
        model_config,
        client,
        Horloge(t=1_700_040_010),
        StrategieEspionne(),
    )

    with pytest.raises(ErreurBoucle, match="Demarrage impossible"):
        boucle.demarrer()
    assert any(ligne["event"] == "registry_error" for ligne in _lignes(boucle.journal.chemin))


def test_sans_champion_le_demarrage_dit_quoi_lancer(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    boucle = _service(
        registre,
        reglages_serveur,
        reglages_boucle,
        features_config,
        model_config,
        PrometheusSynthetique(),
        Horloge(t=1_700_040_010),
        StrategieEspionne(),
    )
    with pytest.raises(ErreurBoucle, match="registry-init"):
        boucle.demarrer()


def test_le_prechargement_remplit_l_historique_sans_scorer_aucune_fenetre(
    service: Service,
) -> None:
    """Scorer le passe au demarrage produirait des alertes sur des evenements finis."""
    boucle, _, _, strategy, _ = service
    boucle.demarrer()

    assert boucle.n_fenetres_vues == boucle.reglages.prefill_fenetres
    assert len(boucle._historique) == boucle.reglages.prefill_fenetres
    # Rien n'a ete score : la strategie n'a vu aucune fenetre et le journal ne
    # contient que le demarrage.
    assert strategy.fenetres_vues == []
    assert boucle.mesures.n == 0
    assert [ligne["event"] for ligne in _lignes(boucle.journal.chemin)] == ["service_start"]
    boucle.arreter()


def test_le_demarrage_journalise_l_empreinte_des_features_en_service(service: Service) -> None:
    """La capture du chapitre 5 doit pouvoir dire quelles colonnes tournaient."""
    boucle, _, _, _, _ = service
    boucle.demarrer()

    details = _lignes(boucle.journal.chemin)[0]["details"]
    assert details["features_empreinte"] == boucle.signature.empreinte
    assert details["n_features"] == N_FEATURES
    assert details["version"] == 1
    boucle.arreter()


# -------------------------------------------------------------------- features


def test_les_features_en_ligne_sont_exactement_celles_du_rejeu(service: Service) -> None:
    """L'invariant central du chapitre 5.

    Rien dans un `IsolationForest` ne verifie que les colonnes qu'on lui donne
    ont le sens de celles sur lesquelles il a appris : il rend un score, sans
    broncher. Ce test est le seul endroit ou une divergence entre les features
    de la boucle et celles du harnais se verrait.
    """
    boucle, client, horloge, _, _ = service
    boucle.demarrer()

    horloge.avancer(3 * PAS)
    resultats = boucle.tick()
    assert len(resultats) == 3

    # Les memes points bruts, passes au harnais : les features des trois
    # dernieres fenetres doivent coincider au bit pres.
    instants = np.arange(
        resultats[0].fin_ts - (FENETRE - 1) * PAS, resultats[-1].fin_ts + PAS, PAS, dtype=np.int64
    )
    attendues = make_windows(
        _matrice(instants, client), boucle.features_config, metric_names=list(METRIQUES)
    ).X
    obtenues = np.array(list(boucle._historique)[-3:])

    assert attendues.shape == obtenues.shape == (3, N_FEATURES)
    np.testing.assert_array_equal(obtenues, attendues)
    boucle.arreter()


def test_une_fenetre_dont_le_dernier_point_manque_est_sautee_et_journalisee(
    service: Service,
) -> None:
    """Pas d'extrapolation au bord : le point le plus recent est celui qui alerte."""
    boucle, client, horloge, _, _ = service
    boucle.demarrer()

    absent = int(boucle._instants[-1]) + PAS
    client.trous = {absent}
    horloge.avancer(PAS)
    resultats = boucle.tick()

    assert resultats == []
    assert boucle.n_sautees == 1
    sautees = [
        ligne for ligne in _lignes(boucle.journal.chemin) if ligne["event"] == "window_skipped"
    ]
    assert len(sautees) == 1
    assert sautees[0]["details"]["fin_ts"] == absent
    assert sautees[0]["details"]["raison"] == "points_manquants"
    boucle.arreter()


def test_un_point_manquant_ne_coute_qu_une_seule_fenetre(service: Service) -> None:
    """Un trou court et interieur est interpole, comme dans le chargeur de sessions.

    Refuser toute fenetre contenant le point manquant couterait `window_size`
    fenetres, soit cinq minutes d'aveuglement pour un scrape rate.
    """
    boucle, client, horloge, _, _ = service
    boucle.demarrer()

    client.trous = {int(boucle._instants[-1]) + PAS}
    horloge.avancer(PAS)
    assert boucle.tick() == []  # la fenetre qui finit sur le trou est sautee

    horloge.avancer(3 * PAS)
    suivantes = boucle.tick()

    assert len(suivantes) == 3
    assert all(r.interpolee for r in suivantes)
    assert boucle.n_sautees == 1
    boucle.arreter()


def test_un_trou_trop_long_fait_sauter_les_fenetres_concernees(service: Service) -> None:
    """La borne est celle du chargeur : interpoler plus long inventerait un regime."""
    boucle, client, horloge, _, _ = service
    boucle.demarrer()

    depart = int(boucle._instants[-1]) + PAS
    longueur = boucle.serveur.max_trou_consecutif + 1
    client.trous = {depart + i * PAS for i in range(longueur)}
    horloge.avancer((longueur + 2) * PAS)
    resultats = boucle.tick()

    # Les fenetres qui recouvrent le trou long sont toutes sautees ; celles qui
    # n'en voient plus qu'une part assez courte repassent.
    assert boucle.n_sautees >= longueur
    assert all(np.isfinite(r.score) for r in resultats)
    boucle.arreter()


# ------------------------------------------------------------------- collecte


def test_un_retard_superieur_a_la_borne_ne_rattrape_pas(service: Service) -> None:
    """Rejouer des heures fenetre par fenetre produirait des alertes perimees."""
    boucle, _, horloge, _, _ = service
    boucle.demarrer()
    avant = len(boucle._historique)

    horloge.avancer((boucle.reglages.rattrapage_max_points + 200) * PAS)
    resultats = boucle.tick()

    trous = [ligne for ligne in _lignes(boucle.journal.chemin) if ligne["event"] == "collect_gap"]
    assert len(trous) == 1
    assert trous[0]["details"]["points_manques"] > boucle.reglages.rattrapage_max_points
    assert len(resultats) <= boucle.reglages.rattrapage_max_points
    # L'historique de reentrainement est conserve : ce sont de vraies donnees
    # passees, et les jeter interdirait tout reentrainement pendant des heures.
    assert len(boucle._historique) > avant
    boucle.arreter()


def test_prometheus_indisponible_ne_tue_pas_le_service(service: Service) -> None:
    """Un detecteur qui meurt avec sa source ne survit pas a un redemarrage de la pile."""
    boucle, client, horloge, _, _ = service
    boucle.demarrer()

    client.pannes = 1
    horloge.avancer(PAS)
    assert boucle.tick() == []
    assert any(ligne["event"] == "collect_gap" for ligne in _lignes(boucle.journal.chemin))

    horloge.avancer(PAS)
    assert len(boucle.tick()) >= 1
    boucle.arreter()


def test_un_tick_avant_le_prochain_instant_de_grille_ne_lit_rien(service: Service) -> None:
    boucle, client, _, _, _ = service
    boucle.demarrer()
    requetes = client.requetes

    assert boucle.tick() == []
    assert client.requetes == requetes  # aucune requete inutile a Prometheus
    boucle.arreter()


def test_l_attente_vise_le_prochain_instant_de_grille_publie(service: Service) -> None:
    boucle, _, horloge, _, _ = service
    boucle.demarrer()

    attente = boucle._attente()
    cible = int(boucle._instants[-1]) + PAS + boucle.reglages.retard_s

    assert attente == pytest.approx(cible - horloge.t)
    assert 0 < attente <= PAS + boucle.reglages.retard_s
    boucle.arreter()


# -------------------------------------------------------------------- scoring


def test_chaque_fenetre_est_scoree_et_chronometree(service: Service) -> None:
    """`window_ms` est la mesure que le rejeu ne peut pas donner (§9)."""
    boucle, _, horloge, strategy, _ = service
    boucle.demarrer()
    depart = boucle.n_fenetres_vues

    horloge.avancer(5 * PAS)
    resultats = boucle.tick()

    assert len(resultats) == 5
    assert boucle.mesures.n == 5
    assert boucle.mesures.moyenne > 0
    assert boucle.mesures.p95 > 0
    assert [c.position for c in strategy.fenetres_vues] == list(range(depart, depart + 5))
    assert all(c.threshold == boucle.champion.threshold for c in strategy.fenetres_vues)
    boucle.arreter()


def test_une_alerte_est_journalisee_avec_son_score_et_son_seuil(service: Service) -> None:
    """Ce que ce test verifie est la trace, pas la sensibilite du detecteur.

    La sensibilite du modele se mesure au chapitre 6, sur des donnees reelles et
    etiquetees ; elle ne se teste pas sur une sinusoide de synthese. Ici le seuil
    est abaisse volontairement pour que la fenetre alerte a coup sur, et seule la
    ligne du journal est controlee : c'est elle que l'exporter et Telegram liront.
    """
    boucle, _, horloge, _, _ = service
    boucle.demarrer()
    # Les scores d'un Isolation Forest sont positifs : un seuil nul fait alerter
    # chaque fenetre, sans toucher au modele lui-meme.
    boucle.champion.threshold = 0.0

    horloge.avancer(2 * PAS)
    resultats = boucle.tick()

    assert all(r.alerte for r in resultats)
    alertes = [ligne for ligne in _lignes(boucle.journal.chemin) if ligne["event"] == "alert"]
    assert len(alertes) == len(resultats) == 2
    assert alertes[0]["details"]["score"] > alertes[0]["details"]["threshold"]
    assert alertes[0]["details"]["fin_ts"] == resultats[0].fin_ts
    assert alertes[0]["details"]["interpolee"] is False
    assert boucle.n_alertes == len(alertes)
    boucle.arreter()


def test_les_evenements_de_derive_demandes_par_la_strategie_sont_journalises(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    """La strategie declare, la boucle ecrit : le journal reste d'un seul tenant."""
    client = PrometheusSynthetique()
    _champion(registre, reglages_serveur, features_config, model_config, client)
    horloge = Horloge(t=1_700_040_000 + reglages_boucle.retard_s)
    boucle = _service(
        registre,
        reglages_serveur,
        reglages_boucle,
        features_config,
        model_config,
        client,
        horloge,
        StrategieEspionne(evenements=("drift_signal",)),
    )
    boucle.demarrer()
    horloge.avancer(2 * PAS)
    boucle.tick()

    signaux = [
        ligne for ligne in _lignes(boucle.journal.chemin) if ligne["event"] == "drift_signal"
    ]
    assert len(signaux) == 2
    boucle.arreter()


# -------------------------------------------------------------- reentrainement


def test_un_challenger_valide_est_enregistre_promu_et_mis_en_service(
    service: Service, registre: RegistreModeles
) -> None:
    boucle, _, horloge, strategy, _ = service
    boucle.demarrer()
    strategy.reentrainer_a = {boucle.n_fenetres_vues + 1}
    ancien = boucle.champion

    horloge.avancer(3 * PAS)
    resultats = boucle.tick()

    assert [r.promue for r in resultats] == [False, True, False]
    assert boucle.n_reentrainements == 1
    assert boucle.n_promotions == 1
    assert boucle.champion is not ancien
    # Le registre porte la nouvelle version, et c'est elle qui est championne.
    courant = registre.champion()
    assert courant is not None
    assert courant.version == 2
    assert boucle.version.version == 2
    # La nouvelle version emporte sa propre reference, et la strategie a recale
    # son moniteur dessus.
    assert courant.a_une_reference
    np.testing.assert_allclose(registre.charger_reference(courant), strategy.references[-1])

    evenements = [ligne["event"] for ligne in _lignes(boucle.journal.chemin)]
    assert evenements.count("retrain_start") == 1
    assert evenements.count("retrain_end") == 1
    assert evenements.count("promoted") == 1
    boucle.arreter()


def test_un_challenger_rejete_laisse_le_champion_en_place(
    service: Service, registre: RegistreModeles
) -> None:
    """Le reentrainement reste compte dans le cout : il a bien eu lieu."""
    boucle, _, horloge, strategy, _ = service
    boucle.demarrer()
    strategy.reentrainer_a = {boucle.n_fenetres_vues + 1}
    strategy.promouvoir = False
    ancien = boucle.champion

    horloge.avancer(3 * PAS)
    resultats = boucle.tick()

    assert not any(r.promue for r in resultats)
    assert boucle.n_reentrainements == 1
    assert boucle.n_rejets == 1
    assert boucle.champion is ancien
    assert registre.champion().version == 1
    evenements = [ligne["event"] for ligne in _lignes(boucle.journal.chemin)]
    assert evenements.count("rejected") == 1
    assert "promoted" not in evenements
    boucle.arreter()


def test_sans_promotion_la_validation_a_lieu_mais_rien_n_est_mis_en_service(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    """Repetition avant demonstration : la configuration se verifie sans rien changer."""
    client = PrometheusSynthetique()
    _champion(registre, reglages_serveur, features_config, model_config, client)
    horloge = Horloge(t=1_700_040_000 + reglages_boucle.retard_s)
    strategy = StrategieEspionne()
    boucle = _service(
        registre,
        reglages_serveur,
        reglages_boucle,
        features_config,
        model_config,
        client,
        horloge,
        strategy,
        promotion=False,
    )
    boucle.demarrer()
    strategy.reentrainer_a = {boucle.n_fenetres_vues + 1}
    ancien = boucle.champion
    horloge.avancer(3 * PAS)
    boucle.tick()

    assert boucle.n_reentrainements == 1
    assert boucle.n_promotions == 0
    assert boucle.champion is ancien
    assert registre.champion().version == 1
    rejets = [ligne for ligne in _lignes(boucle.journal.chemin) if ligne["event"] == "rejected"]
    assert rejets[0]["details"]["raison"] == "promotion_desactivee"
    boucle.arreter()


def test_un_registre_en_echec_a_la_promotion_garde_le_champion_en_service(
    service: Service,
) -> None:
    """Le registre fait foi : promouvoir en memoire ferait mentir les captures."""
    boucle, _, horloge, strategy, _ = service
    boucle.demarrer()
    strategy.reentrainer_a = {boucle.n_fenetres_vues + 1}
    ancien = boucle.champion
    version = boucle.version

    def tombe(*_: Any, **__: Any) -> None:
        raise ErreurRegistre("Registre injoignable (panne simulee).")

    boucle.registre.enregistrer = tombe  # type: ignore[method-assign]
    horloge.avancer(3 * PAS)
    boucle.tick()

    assert boucle.champion is ancien
    assert boucle.version is version
    assert boucle.n_promotions == 0
    assert boucle.n_rejets == 1
    erreurs = [
        ligne for ligne in _lignes(boucle.journal.chemin) if ligne["event"] == "registry_error"
    ]
    assert erreurs[0]["details"]["phase"] == "promotion"
    boucle.arreter()


# ---------------------------------------------------------------------- arret


def test_un_tick_avant_le_demarrage_est_refuse(service: Service) -> None:
    boucle, _, _, _, _ = service
    with pytest.raises(ErreurBoucle, match="demarrer"):
        boucle.tick()


def test_executer_rend_la_main_et_journalise_les_compteurs(service: Service) -> None:
    boucle, _, horloge, _, _ = service
    boucle.demarrer()
    horloge.avancer(2 * PAS)
    boucle.executer(max_ticks=1)

    derniere = _lignes(boucle.journal.chemin)[-1]
    assert derniere["event"] == "service_stop"
    assert derniere["details"]["n_windows"] == boucle.n_fenetres_vues
    assert derniere["details"]["window_ms_mean"] > 0
    assert derniere["details"]["n_alerts"] == boucle.n_alertes


# -------------------------------------------------------------- notifications


class FauxNotificateur:
    """Enregistre ce que la boucle lui demande, sans rien envoyer."""

    def __init__(self) -> None:
        self.entrees: list[dict[str, Any]] = []
        self.rincages = 0
        self.forces = 0

    def observer_evenement(self, entree: dict[str, Any]) -> None:
        self.entrees.append(entree)

    def rincer(self, *, forcer: bool = False) -> int:
        self.rincages += 1
        if forcer:
            self.forces += 1
        return 0


def test_la_boucle_abonne_le_notificateur_au_journal_et_rince_ses_groupes(
    registre: RegistreModeles,
    reglages_serveur: ReglagesServeur,
    reglages_boucle: ReglagesBoucle,
    features_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    """Le rincage periodique n'est pas un detail d'implementation.

    La regle du §9 (au plus un message par minute et par type) met des
    evenements en attente. Sans rincage a chaque tick, le dernier evenement
    d'une rafale resterait en attente jusqu'au prochain du meme type — donc
    peut-etre jusqu'a la fin du service — et l'arret sans rincage force
    perdrait le resume de session, qui est le message qui clot une
    demonstration.
    """
    client = PrometheusSynthetique()
    _champion(
        registre, reglages_serveur, features_config, model_config, client, fin_ts=1_700_000_000
    )
    horloge = Horloge(t=1_700_040_000 + reglages_boucle.retard_s)
    faux = FauxNotificateur()
    boucle = _service(
        registre,
        reglages_serveur,
        reglages_boucle,
        features_config,
        model_config,
        client,
        horloge,
        StrategieEspionne(),
        notificateur=faux,
    )

    boucle.demarrer()
    assert [entree["event"] for entree in faux.entrees] == ["service_start"]
    assert faux.rincages == 0  # aucun tick encore

    horloge.avancer(2 * PAS)
    boucle.executer(max_ticks=1)

    assert faux.rincages >= 1
    assert faux.forces == 1
    assert faux.entrees[-1]["event"] == "service_stop"
    # Le notificateur ne voit que des lignes de journal, jamais autre chose.
    assert all(entree["event"] in EVENEMENTS for entree in faux.entrees)
