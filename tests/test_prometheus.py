"""Lecture de l'API Prometheus (livrable M4, phase D).

Ces tests portent sur les trois proprietes du module : la grille est faite de
multiples du pas, un trou reste un trou, et une expression qui rend plusieurs
series est refusee. Aucun des trois ne se verifie en regardant un tableau de
bord : un decalage d'un pas ou un trou bouche en silence produit des chiffres
plausibles et faux, ce qui est le pire cas possible pour le chapitre 6.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import pytest

from src.serving.prometheus import (
    POINTS_MAX_PAR_REQUETE,
    ClientPrometheus,
    ErreurPrometheus,
    SerieTemporelle,
    grille,
)

# Base des instants, choisie multiple de 60 : les instants de la grille sont des
# multiples du pas, donc un test qui part d'une base quelconque teste surtout
# l'arrondi et non la propriete visee.
T0 = 1_793_000_040
PAS = 60


class PrometheusFactice:
    """Repond a la place de Prometheus, et retient ce qu'on lui a demande."""

    def __init__(self, reponses: list[dict[str, Any]] | None = None) -> None:
        self.reponses = reponses or []
        self.appels: list[dict[str, Any]] = []

    def installer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.serving.prometheus.httpx.get", self.get)

    def get(self, url: str, params: dict[str, Any], timeout: float) -> httpx.Response:
        self.appels.append({"url": url, **params})
        corps = self.reponses.pop(0) if self.reponses else {"status": "success", "data": {}}
        return httpx.Response(200, json=corps)


def _matrice(points: list[tuple[int, str]], etiquettes: dict[str, str] | None = None) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [{"metric": etiquettes or {}, "values": [list(p) for p in points]}],
        },
    }


def test_la_grille_ne_contient_que_des_multiples_du_pas() -> None:
    # Un debut a 10h00m37s doit donner une grille a 10h01, 10h02... : sinon la
    # grille depend de l'heure a laquelle la commande a ete lancee, et deux
    # enregistrements de la meme session ne tombent pas sur les memes instants.
    instants = grille(T0 - 3, T0 + 4 * PAS, PAS)
    assert list(instants) == [T0, T0 + PAS, T0 + 2 * PAS, T0 + 3 * PAS, T0 + 4 * PAS]
    assert all(int(t) % PAS == 0 for t in instants)


def test_la_grille_reste_contenue_dans_la_plage_demandee() -> None:
    instants = grille(T0 - 19, T0 + 2 * PAS + 11, PAS)
    assert int(instants[0]) >= T0 - 19
    assert int(instants[-1]) <= T0 + 2 * PAS + 11


def test_une_plage_sans_multiple_du_pas_est_refusee() -> None:
    with pytest.raises(ValueError, match="Plage trop courte"):
        # Entre T0 + 1 et T0 + 59, aucun instant n'est multiple du pas.
        grille(T0 + 1, T0 + PAS - 1, PAS)


def test_un_instant_absent_devient_nan_et_non_la_valeur_precedente(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prometheus ne signale pas les points manquants : il renvoie simplement une
    # reponse plus courte. Boucher le trou par la valeur precedente inventerait
    # une mesure, et une mesure inventee dans une fenetre etiquetee anormale est
    # un faux resultat.
    faux = PrometheusFactice([_matrice([(T0, "1.0"), (T0 + 2 * PAS, "3.0")])])
    faux.installer(monkeypatch)
    serie = ClientPrometheus("http://prom").interroger_plage("up", T0, T0 + 2 * PAS, PAS)
    assert len(serie) == 3
    assert serie.valeurs[0] == 1.0
    assert np.isnan(serie.valeurs[1])
    assert serie.valeurs[2] == 3.0
    assert serie.part_manquante == pytest.approx(1 / 3)


def test_un_point_illisible_vaut_un_point_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # `histogram_quantile` sur un bucket vide rend "NaN" en texte.
    faux = PrometheusFactice([_matrice([(T0, "NaN"), (T0 + PAS, "+Inf")])])
    faux.installer(monkeypatch)
    serie = ClientPrometheus("http://prom").interroger_plage("q", T0, T0 + PAS, PAS)
    assert serie.part_manquante == 1.0


def test_une_expression_rendant_plusieurs_series_est_refusee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deux = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {"cpu": "0"}, "values": [[T0, "1"]]},
                {"metric": {"cpu": "1"}, "values": [[T0, "2"]]},
            ]
        },
    }
    faux = PrometheusFactice([deux])
    faux.installer(monkeypatch)
    with pytest.raises(ErreurPrometheus, match="rend 2 series"):
        ClientPrometheus("http://prom").interroger_plage(
            "node_cpu_seconds_total", T0, T0 + PAS, PAS
        )


def test_zero_serie_n_est_pas_une_erreur_mais_une_colonne_vide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Aucune 5xx sur la plage : legitime. C'est au chargeur de session de
    # trancher, parce que lui seul sait si la colonne est indispensable.
    faux = PrometheusFactice([{"status": "success", "data": {"result": []}}])
    faux.installer(monkeypatch)
    serie = ClientPrometheus("http://prom").interroger_plage("part_5xx", T0, T0 + PAS, PAS)
    assert serie.part_manquante == 1.0
    assert len(serie) == 2


def test_une_longue_plage_est_decoupee_sur_des_multiples_du_pas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prometheus refuse au-dela de 11 000 points. Le decoupage doit tomber sur
    # des multiples du pas, sinon les morceaux ne se recollent pas et il manque
    # un point a chaque jointure.
    pas = PAS
    debut = T0
    n = POINTS_MAX_PAR_REQUETE + 500
    fin = debut + (n - 1) * pas
    faux = PrometheusFactice([_matrice([]), _matrice([])])
    faux.installer(monkeypatch)

    serie = ClientPrometheus("http://prom").interroger_plage("up", debut, fin, pas)

    assert len(serie) == n
    assert len(faux.appels) == 2
    premier, second = faux.appels
    assert int(premier["start"]) == debut
    assert int(premier["end"]) == debut + (POINTS_MAX_PAR_REQUETE - 1) * pas
    assert int(second["start"]) == debut + POINTS_MAX_PAR_REQUETE * pas
    assert int(second["start"]) - int(premier["end"]) == pas
    assert all(int(appel["start"]) % pas == 0 for appel in faux.appels)


def test_une_erreur_promql_est_rapportee_avec_le_message_de_prometheus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get(url: str, params: dict[str, Any], timeout: float) -> httpx.Response:
        return httpx.Response(
            400,
            json={"status": "error", "errorType": "bad_data", "error": "parse error at char 3"},
        )

    monkeypatch.setattr("src.serving.prometheus.httpx.get", get)
    with pytest.raises(ErreurPrometheus, match="parse error at char 3"):
        ClientPrometheus("http://prom").interroger_plage("up{", T0, T0 + PAS, PAS)


def test_prometheus_injoignable_dit_quoi_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str, params: dict[str, Any], timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refuse")

    monkeypatch.setattr("src.serving.prometheus.httpx.get", get)
    with pytest.raises(ErreurPrometheus, match="PROMETHEUS_URL"):
        ClientPrometheus("http://prom:9090/").cibles()


def test_la_pire_sante_l_emporte_quand_un_job_a_plusieurs_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cibles = {
        "status": "success",
        "data": {
            "activeTargets": [
                {"labels": {"job": "node"}, "health": "up"},
                {"labels": {"job": "node"}, "health": "down"},
                {"labels": {"job": "app"}, "health": "up"},
            ]
        },
    }
    faux = PrometheusFactice([cibles])
    faux.installer(monkeypatch)
    etats = ClientPrometheus("http://prom").cibles()
    assert etats == {"node": "down", "app": "up"}


def test_une_serie_vide_est_declaree_entierement_manquante() -> None:
    vide = SerieTemporelle(instants=np.array([], dtype=np.int64), valeurs=np.array([]))
    assert vide.part_manquante == 1.0
    assert len(vide) == 0
