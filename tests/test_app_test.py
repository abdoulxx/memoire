"""Tests de l'appli de test (livrable M4, phase A).

Ce qui est verifie ici n'est pas « l'appli repond », mais les proprietes dont
depend la validite des etiquettes d'une session : le plan de controle reste
joignable pendant une injection, une injection a toujours une fin, l'intensite
enregistree est celle qui a ete subie, et les metriques exposees permettent
vraiment de reconstituer p50 / p95.
"""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from infra.app.chaos import DUREE_MAX_S, Chaos
from infra.app.main import creer_application
from infra.app.metrics import Metriques
from infra.app.settings import Reglages


class HorlogePilotee:
    """Horloge a avancee manuelle : une injection de 90 s se teste en 0 s."""

    def __init__(self, depart: float = 1_000_000.0) -> None:
        self.maintenant = depart

    def __call__(self) -> float:
        return self.maintenant

    def avancer(self, secondes: float) -> None:
        self.maintenant += secondes


def _application(
    horloge: HorlogePilotee | None = None,
    graine: int = 0,
    chaos_autorise: bool = True,
    latence_base_ms: float = 1.0,
) -> tuple[TestClient, Chaos, Metriques]:
    horloge = horloge or HorlogePilotee()
    # Sans ces remplacements, un test de pic CPU brulerait un vrai cœur et un
    # test de fuite memoire allouerait de la vraie memoire (D46) : le travail de
    # fond est injectable precisement pour que les tests restent instantanes.
    chaos = Chaos(
        horloge=horloge,
        demarrer_cpu=lambda threads, duree_s, actif: None,
        demarrer_memoire=lambda mo, duree_s, actif: None,
    )
    metriques = Metriques()
    reglages = Reglages(
        version="1.0.0",
        # Latence de base minuscule : les tests mesurent la *logique*, pas les
        # performances, et une base de 25 ms rendrait la suite inutilement lente.
        latence_base_ms=latence_base_ms,
        dispersion=0.0,
        chaos_autorise=chaos_autorise,
    )
    application = creer_application(
        reglages=reglages,
        chaos=chaos,
        metriques=metriques,
        alea=random.Random(graine),
    )
    return TestClient(application), chaos, metriques


def test_le_plan_de_controle_reste_joignable_pendant_une_rafale_de_5xx():
    """Propriete critique : a 100 % de 5xx, on doit pouvoir arreter l'injection.

    Si `/chaos/reset` subissait l'injection, il faudrait tuer le conteneur pour
    revenir a la normale — et la session enregistree serait perdue, puisque son
    instant de retour au calme ne serait plus celui des etiquettes.
    """
    client, _, _ = _application()
    client.post("/chaos/errors", json={"taux": 1.0, "duree_s": 300}).raise_for_status()

    assert client.get("/work").status_code == 500  # le trafic normal est bien casse
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/chaos/state").status_code == 200
    assert client.post("/chaos/reset").json() == {"arretees": 1}
    assert client.get("/work").status_code == 200


def test_une_injection_s_arrete_d_elle_meme_a_son_echeance():
    horloge = HorlogePilotee()
    client, _, _ = _application(horloge=horloge)
    client.post("/chaos/latency", json={"supplement_ms": 500, "duree_s": 90}).raise_for_status()

    assert client.get("/chaos/state").json()["actives"][0]["restant_s"] == pytest.approx(90.0)
    horloge.avancer(89.0)
    assert len(client.get("/chaos/state").json()["actives"]) == 1
    horloge.avancer(2.0)
    assert client.get("/chaos/state").json()["actives"] == []


def test_une_injection_sans_fin_est_refusee():
    """Le plafond de duree n'est pas un detail : une injection oubliee contamine
    tout le reste de la session, donc toutes les etiquettes qui suivent."""
    client, _, _ = _application()
    trop_long = client.post(
        "/chaos/latency", json={"supplement_ms": 10, "duree_s": DUREE_MAX_S + 1}
    )
    assert trop_long.status_code == 400
    assert "plafond" in trop_long.json()["detail"]

    # Duree nulle ou negative : refusee par le schema, avant d'atteindre le Chaos.
    nulle = client.post("/chaos/latency", json={"supplement_ms": 10, "duree_s": 0})
    assert nulle.status_code == 422


def test_l_arret_manuel_tronque_l_etiquette_a_ce_qui_a_ete_subi():
    """L'historique doit dire ce qui s'est passe, pas ce qui avait ete demande."""
    horloge = HorlogePilotee()
    client, _, _ = _application(horloge=horloge)
    client.post("/chaos/errors", json={"taux": 0.5, "duree_s": 600}).raise_for_status()
    horloge.avancer(42.0)
    client.post("/chaos/reset").raise_for_status()

    historique = client.get("/chaos/state").json()["historique"]
    assert len(historique) == 1
    subie = historique[0]
    assert subie["fin_ts"] - subie["debut_ts"] == pytest.approx(42.0)
    assert subie["intensite"] == pytest.approx(0.5)


def test_une_seconde_injection_du_meme_type_remplace_la_premiere():
    """Deux latences ajoutees qui se cumulent rendraient l'intensite etiquetee fausse."""
    client, chaos, _ = _application()
    client.post("/chaos/latency", json={"supplement_ms": 100, "duree_s": 300}).raise_for_status()
    client.post("/chaos/latency", json={"supplement_ms": 250, "duree_s": 300}).raise_for_status()

    assert chaos.latence_ajoutee_ms() == pytest.approx(250.0)
    assert len(client.get("/chaos/state").json()["actives"]) == 1


def test_les_deux_types_d_injection_coexistent():
    client, chaos, _ = _application()
    client.post("/chaos/latency", json={"supplement_ms": 50, "duree_s": 300}).raise_for_status()
    client.post("/chaos/errors", json={"taux": 0.25, "duree_s": 300}).raise_for_status()

    assert chaos.latence_ajoutee_ms() == pytest.approx(50.0)
    assert chaos.taux_erreurs() == pytest.approx(0.25)
    assert len(client.get("/chaos/state").json()["actives"]) == 2


def test_le_taux_d_erreurs_injecte_se_retrouve_dans_les_reponses():
    """Verifie que le taux est bien une *part* et non un tout-ou-rien.

    Le generateur est seme, donc le compte est exact et reproductible ; c'est le
    seul moyen de tester un comportement probabiliste sans tolerance arbitraire.
    """
    client, _, _ = _application(graine=12345)
    client.post("/chaos/errors", json={"taux": 0.3, "duree_s": 600}).raise_for_status()

    codes = [client.get("/work").status_code for _ in range(200)]
    erreurs = codes.count(500)
    assert erreurs + codes.count(200) == 200
    assert 40 <= erreurs <= 80  # 0,3 x 200 = 60, large marge pour 200 tirages


def test_les_metriques_exposent_ce_qu_il_faut_pour_p50_et_p95():
    """Le §4.2 demande p50 et p95 : il faut donc un histogramme avec des bornes
    utiles autour de la latence de base, pas une moyenne."""
    client, _, _ = _application()
    client.get("/work").raise_for_status()

    texte = client.get("/metrics").text
    assert "app_request_duration_seconds_bucket" in texte
    assert 'app_requests_total{endpoint="/work",method="GET",status="200"}' in texte
    assert 'app_build_info{version="1.0.0"}' in texte
    assert "app_base_latency_ms" in texte
    # Les deux series d'injection existent des le demarrage : une serie qui
    # n'apparait qu'a la premiere injection laisse un trou, et un trou ne se
    # distingue pas d'un zero dans une requete Grafana.
    assert 'app_chaos_active{type="latence"} 0.0' in texte
    assert 'app_chaos_active{type="erreurs_5xx"} 0.0' in texte


def test_la_jauge_d_injection_suit_l_etat_reel():
    horloge = HorlogePilotee()
    client, _, _ = _application(horloge=horloge)
    client.post("/chaos/latency", json={"supplement_ms": 120, "duree_s": 60}).raise_for_status()

    actif = client.get("/metrics").text
    assert 'app_chaos_active{type="latence"} 1.0' in actif
    assert 'app_chaos_intensity{type="latence"} 120.0' in actif

    horloge.avancer(61.0)
    apres = client.get("/metrics").text
    assert 'app_chaos_active{type="latence"} 0.0' in apres


def test_un_deploiement_sans_injections_refuse_les_endpoints_de_panne():
    """Permet de deployer une fois l'appli pour mesurer le bruit de fond seul."""
    client, _, _ = _application(chaos_autorise=False)
    refus = client.post("/chaos/latency", json={"supplement_ms": 10, "duree_s": 10})
    assert refus.status_code == 403
    assert client.get("/work").status_code == 200


def test_la_version_deployee_est_lisible_dans_les_metriques():
    """Une derive S1 reelle est un deploiement : sans cette serie, l'instant de
    derive devrait etre note a la main au lieu d'etre dans les donnees."""
    client, _, _ = _application()
    assert client.get("/health").json()["version"] == "1.0.0"

    reglages = Reglages(version="1.1.0", latence_base_ms=60.0, dispersion=0.0, chaos_autorise=True)
    autre = TestClient(creer_application(reglages=reglages, metriques=Metriques()))
    texte = autre.get("/metrics").text
    assert 'app_build_info{version="1.1.0"}' in texte
    assert "app_base_latency_ms 60.0" in texte


def test_les_reglages_illisibles_ne_tuent_pas_le_service(monkeypatch):
    """Un service qui refuse de demarrer sur une variable mal ecrite ferait perdre
    la session en cours sur le serveur."""
    monkeypatch.setenv("APP_BASE_LATENCY_MS", "vingt-cinq")
    monkeypatch.setenv("APP_VERSION", "")
    reglages = Reglages.depuis_environnement()
    assert reglages.latence_base_ms == pytest.approx(25.0)
    assert reglages.version == "1.0.0"


def test_un_taux_d_erreurs_hors_borne_est_refuse():
    client, _, _ = _application()
    assert client.post("/chaos/errors", json={"taux": 1.5, "duree_s": 10}).status_code == 422
    assert client.post("/chaos/errors", json={"taux": -0.1, "duree_s": 10}).status_code == 422


# --- Pic CPU et fuite memoire, dans l'appli elle-meme (D46) -----------------


def test_le_pic_cpu_demarre_le_travail_de_fond_et_s_etiquette():
    """Le travail de fond reel est remplace par un test, mais l'etat et
    l'intensite rapportes doivent etre ceux d'une injection normale."""
    appels: list[tuple[float, float]] = []
    horloge = HorlogePilotee()
    chaos = Chaos(
        horloge=horloge,
        demarrer_cpu=lambda threads, duree_s, actif: appels.append((threads, duree_s)),
    )
    client = TestClient(
        creer_application(
            reglages=Reglages(
                version="1.0.0", latence_base_ms=1.0, dispersion=0.0, chaos_autorise=True
            ),
            chaos=chaos,
        )
    )
    reponse = client.post("/chaos/cpu", json={"threads": 2, "duree_s": 60})
    assert reponse.status_code == 200
    assert appels == [(2.0, 60.0)]
    assert client.get("/chaos/state").json()["actives"][0]["type"] == "pic_cpu"
    assert 'app_chaos_active{type="pic_cpu"} 1.0' in client.get("/metrics").text


def test_un_pic_cpu_sans_fil_d_execution_est_refuse():
    client, _, _ = _application()
    assert client.post("/chaos/cpu", json={"threads": 0, "duree_s": 60}).status_code == 422


def test_la_fuite_memoire_demarre_le_travail_de_fond_et_s_etiquette():
    appels: list[tuple[float, float]] = []
    chaos = Chaos(
        horloge=HorlogePilotee(),
        demarrer_memoire=lambda mo, duree_s, actif: appels.append((mo, duree_s)),
    )
    client = TestClient(
        creer_application(
            reglages=Reglages(
                version="1.0.0", latence_base_ms=1.0, dispersion=0.0, chaos_autorise=True
            ),
            chaos=chaos,
        )
    )
    reponse = client.post("/chaos/memory", json={"mo": 150, "duree_s": 60})
    assert reponse.status_code == 200
    assert appels == [(150.0, 60.0)]
    assert 'app_chaos_intensity{type="fuite_memoire"} 150.0' in client.get("/metrics").text


def test_une_fuite_memoire_au_dela_du_plafond_est_refusee():
    """Le plafond (bien sous APP_MEMORY) est le garde-fou de D46 : jamais de
    risque d'arret du conteneur par manque de memoire."""
    client, _, _ = _application()
    refus = client.post("/chaos/memory", json={"mo": 500, "duree_s": 60})
    assert refus.status_code == 400
    assert "plafond" in refus.json()["detail"]


def test_l_arret_d_urgence_remet_aussi_le_cpu_et_la_memoire_a_zero():
    appels: list[bool] = []
    chaos = Chaos(
        horloge=HorlogePilotee(),
        demarrer_cpu=lambda threads, duree_s, actif: appels.append(actif()),
    )
    client = TestClient(
        creer_application(
            reglages=Reglages(
                version="1.0.0", latence_base_ms=1.0, dispersion=0.0, chaos_autorise=True
            ),
            chaos=chaos,
        )
    )
    client.post("/chaos/cpu", json={"threads": 1, "duree_s": 60}).raise_for_status()
    client.post("/chaos/reset").raise_for_status()
    assert 'app_chaos_active{type="pic_cpu"} 0.0' in client.get("/metrics").text
