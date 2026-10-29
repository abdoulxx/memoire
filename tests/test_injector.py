"""Garde-fous de l'injecteur d'anomalies (livrable M4, phase C).

Ces tests ne verifient pas que l'injection fonctionne — cela se verifie sur la
machine, en regardant Grafana. Ils verifient qu'elle ne peut pas deraper : le
§4.3 du cadrage interdit de toucher un conteneur de production, et c'est
precisement la propriete qu'on ne peut pas tester sur une vraie machine sans
prendre le risque qu'on cherche a ecarter. D'ou le client Docker factice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from infra.injector.docker_cli import DockerEnregistre, ErreurDocker
from infra.injector.injector import Injecteur, Limites, RefusDeSecurite, Reglages
from infra.injector.labels import horodatage, lire_anomalies, lire_derives
from src.config import PROJECT_ROOT, load_config

CONFIG_REELLE = "injection.yaml"


def _config() -> dict[str, Any]:
    return load_config(CONFIG_REELLE)


def _reglages(dossier: Path | None = None) -> Reglages:
    reglages = Reglages.depuis_config(_config(), PROJECT_ROOT)
    if dossier is not None:
        reglages = Reglages(**{**reglages.__dict__, "dossier_etiquettes": dossier})
    return reglages


class ApplieFactice:
    """Le plan de controle de l'appli de test, sans appli."""

    def __init__(self) -> None:
        self.appels: list[tuple[str, dict[str, float]]] = []
        self.remises_a_zero = 0

    def injecter(self, chemin: str, charge: dict[str, float]) -> dict[str, Any]:
        self.appels.append((chemin, charge))
        return {"actif": True}

    def reinitialiser(self) -> dict[str, Any]:
        self.remises_a_zero += 1
        return {"arretees": 1}


def _injecteur(
    docker: DockerEnregistre | None = None,
    appli: ApplieFactice | None = None,
    dossier: Path | None = None,
) -> Injecteur:
    return Injecteur(
        reglages=_reglages(dossier),
        docker=docker or DockerEnregistre(),
        appli=appli,
        horloge=lambda: 1_793_000_000.0,
        attente=lambda _: None,
        racine=PROJECT_ROOT,
    )


def _lancement(docker: DockerEnregistre) -> list[str]:
    return next(appel for appel in docker.appels if appel[0] == "run")


# --- La configuration reelle doit decrire les cinq types du §4.2 -------------


def test_la_configuration_decrit_les_cinq_types_du_cadrage() -> None:
    assert set(_reglages().types) == {
        "pic_cpu",
        "fuite_memoire",
        "latence",
        "erreurs",
        "saturation",
    }


def test_tout_type_qui_cree_un_conteneur_declare_ses_limites() -> None:
    for type_ in _reglages().types.values():
        if type_.moyen in {"conteneur", "charge"}:
            assert type_.limites is not None, type_.nom
            assert type_.limites.cpus > 0, type_.nom
            assert type_.limites.memoire, type_.nom


def test_un_type_sans_limites_est_refuse_a_la_lecture() -> None:
    """Le garde-fou est a la lecture de la configuration, pas au lancement."""
    config = _config()
    del config["types"]["pic_cpu"]["limites"]
    with pytest.raises(ValueError, match="limites"):
        Reglages.depuis_config(config, PROJECT_ROOT)


def test_la_limite_memoire_ferme_aussi_l_echange() -> None:
    # Sans `--memory-swap`, le conteneur deborde sur l'echange et la limite
    # memoire ne limite plus rien.
    arguments = Limites(cpus=0.5, memoire="256m").arguments()
    assert arguments.count("256m") == 2
    assert "--memory-swap" in arguments


# --- Aucune injection sans fin ----------------------------------------------


def test_une_injection_sans_duree_est_refusee() -> None:
    with pytest.raises(RefusDeSecurite, match="duree"):
        _injecteur().injecter("pic_cpu", duree_s=0)


def test_une_duree_au_dela_du_plafond_est_refusee() -> None:
    injecteur = _injecteur()
    with pytest.raises(RefusDeSecurite, match="plafond"):
        injecteur.injecter("pic_cpu", duree_s=injecteur.reglages.duree_max_s + 1)


def test_le_conteneur_porte_son_propre_delai_d_expiration() -> None:
    """Si l'injecteur meurt, `stress-ng --timeout` arrete le conteneur."""
    docker = DockerEnregistre()
    _injecteur(docker).injecter("pic_cpu", duree_s=120)
    lancement = _lancement(docker)
    assert lancement[lancement.index("--timeout") + 1] == "120s"


def test_la_charge_porte_aussi_son_propre_delai() -> None:
    docker = DockerEnregistre()
    _injecteur(docker).injecter("saturation", duree_s=90)
    lancement = _lancement(docker)
    assert lancement[lancement.index("--run-time") + 1] == "90s"


def test_une_intensite_nulle_est_refusee() -> None:
    with pytest.raises(RefusDeSecurite, match="intensite"):
        _injecteur().injecter("pic_cpu", intensite=0)


def test_un_type_inconnu_est_refuse() -> None:
    with pytest.raises(RefusDeSecurite, match="inconnu"):
        _injecteur().injecter("rm_rf_slash")


# --- Aucun conteneur sans limites, aucun privilege --------------------------


def test_tout_conteneur_lance_porte_ses_limites() -> None:
    for type_ in ("pic_cpu", "fuite_memoire", "saturation"):
        docker = DockerEnregistre()
        _injecteur(docker).injecter(type_, duree_s=30)
        lancement = _lancement(docker)
        assert "--cpus" in lancement, type_
        assert "--memory" in lancement, type_
        assert "--memory-swap" in lancement, type_


def test_aucun_conteneur_d_injection_n_obtient_de_privilege() -> None:
    """Un conteneur de `stress-ng` n'a aucune raison de voir l'hote."""
    interdits = {
        "--privileged",
        "--pid",
        "--pid=host",
        "--cgroupns",
        "--ipc",
        "--cap-add",
        "--device",
        "--userns",
    }
    for type_ in ("pic_cpu", "fuite_memoire", "saturation"):
        docker = DockerEnregistre()
        _injecteur(docker).injecter(type_, duree_s=30)
        assert not interdits & set(_lancement(docker)), type_


def test_le_stress_tourne_sans_reseau_et_sans_montage() -> None:
    for type_ in ("pic_cpu", "fuite_memoire"):
        docker = DockerEnregistre()
        _injecteur(docker).injecter(type_, duree_s=30)
        lancement = _lancement(docker)
        assert lancement[lancement.index("--network") + 1] == "none", type_
        assert "--volume" not in lancement, type_


def test_le_seul_montage_de_l_injecteur_est_le_scenario_de_charge() -> None:
    docker = DockerEnregistre()
    _injecteur(docker).injecter("saturation", duree_s=30)
    lancement = _lancement(docker)
    montages = [lancement[i + 1] for i, a in enumerate(lancement) if a == "--volume"]
    assert len(montages) == 1
    assert montages[0].endswith("infra/locust:/mnt/locust:ro")


# --- Les deux marques, jamais une seule -------------------------------------


def test_tout_conteneur_cree_porte_les_deux_marques() -> None:
    docker = DockerEnregistre()
    injecteur = _injecteur(docker)
    injecteur.injecter("pic_cpu", duree_s=30)
    lancement = _lancement(docker)
    nom = lancement[lancement.index("--name") + 1]
    assert nom.startswith(f"{injecteur.reglages.prefixe}-")
    assert f"{injecteur.reglages.etiquette}=1" in lancement


def test_un_conteneur_sans_le_prefixe_n_est_pas_arrete() -> None:
    """Le cas qui compte : l'etiquette seule ne suffit pas."""
    injecteur = _injecteur()
    injecteur.docker.inventaire["appli-de-production"] = {injecteur.reglages.etiquette: "1"}
    with pytest.raises(RefusDeSecurite, match="deux"):
        injecteur.arreter("appli-de-production")
    assert "appli-de-production" in injecteur.docker.inventaire


def test_un_conteneur_sans_l_etiquette_n_est_pas_arrete() -> None:
    """Et le prefixe seul non plus : une faute de frappe l'usurpe."""
    injecteur = _injecteur()
    nom = f"{injecteur.reglages.prefixe}-pic_cpu-1"
    injecteur.docker.inventaire[nom] = {"autre": "1"}
    with pytest.raises(RefusDeSecurite, match="deux"):
        injecteur.arreter(nom)
    assert nom in injecteur.docker.inventaire


def test_l_arret_d_urgence_laisse_les_autres_conteneurs_tranquilles() -> None:
    injecteur = _injecteur(appli=ApplieFactice())
    etiquette = injecteur.reglages.etiquette
    a_nous = f"{injecteur.reglages.prefixe}-pic_cpu-1793000000"
    injecteur.docker.inventaire.update(
        {
            a_nous: {etiquette: "1"},
            # Un conteneur de la pile du memoire, et un conteneur d'application :
            # ni l'un ni l'autre ne doit bouger.
            "memoire-app": {"memoire.role": "appli_de_test"},
            "appli-de-production": {},
            # Le piege : l'etiquette posee par quelqu'un d'autre.
            "autre-chose": {etiquette: "1"},
        }
    )
    resultat = injecteur.arreter_tout()
    assert resultat.conteneurs == [a_nous]
    assert set(injecteur.docker.inventaire) == {
        "memoire-app",
        "appli-de-production",
        "autre-chose",
    }


def test_l_inventaire_exige_les_deux_marques() -> None:
    injecteur = _injecteur()
    etiquette = injecteur.reglages.etiquette
    prefixe = injecteur.reglages.prefixe
    injecteur.docker.inventaire.update(
        {
            f"{prefixe}-latence-2": {etiquette: "1"},
            "autre-chose": {etiquette: "1"},
            f"{prefixe}-latence-3": {},
        }
    )
    assert injecteur.lister() == [f"{prefixe}-latence-2"]


def test_un_conteneur_introuvable_n_est_pas_arrete() -> None:
    with pytest.raises(ErreurDocker):
        _injecteur().arreter("fantome")


def test_l_arret_d_urgence_remet_le_plan_de_controle_a_zero() -> None:
    # Sinon une rafale de 5xx en cours survivrait a `make stop-chaos`, et ce ne
    # serait pas un arret d'urgence.
    appli = ApplieFactice()
    resultat = _injecteur(appli=appli).arreter_tout()
    assert appli.remises_a_zero == 1
    assert resultat.plan_de_controle == "remis a zero"


# --- Injections par le plan de controle de l'appli --------------------------


def test_la_latence_passe_par_le_plan_de_controle() -> None:
    appli = ApplieFactice()
    docker = DockerEnregistre()
    _injecteur(docker, appli).injecter("latence", intensite=200, duree_s=60)
    assert appli.appels[0] == ("/chaos/latency", {"supplement_ms": 200.0, "duree_s": 60.0})
    # Et surtout : aucun conteneur cree pour cela.
    assert not [appel for appel in docker.appels if appel[0] == "run"]


def test_les_erreurs_passent_par_le_plan_de_controle() -> None:
    appli = ApplieFactice()
    _injecteur(appli=appli).injecter("erreurs", intensite=0.2, duree_s=60)
    assert appli.appels[0] == ("/chaos/errors", {"taux": 0.2, "duree_s": 60.0})


def test_une_injection_applicative_est_remise_a_zero_a_la_fin() -> None:
    appli = ApplieFactice()
    _injecteur(appli=appli).injecter("latence", duree_s=60)
    assert appli.remises_a_zero == 1


def test_sans_appli_joignable_l_injection_applicative_est_refusee() -> None:
    with pytest.raises(RefusDeSecurite, match="joignable"):
        _injecteur().injecter("latence", duree_s=60)


# --- Etiquettes : ce qui a ete subi, pas ce qui a ete demande ---------------


def test_l_anomalie_est_etiquetee_avec_ce_qui_a_ete_subi(tmp_path: Path) -> None:
    injecteur = _injecteur(dossier=tmp_path)
    anomalie = injecteur.injecter("pic_cpu", intensite=2, duree_s=180, run_id="s1")
    lignes = lire_anomalies(tmp_path, "s1")
    assert len(lignes) == 1
    assert lignes[0]["type"] == "pic_cpu"
    assert float(lignes[0]["intensite"]) == 2.0
    # Les fichiers d'etiquettes portent des horodatages RFC 3339 UTC, que
    # l'API de Prometheus accepte tels quels dans un `query_range`.
    assert lignes[0]["start_ts"] == horodatage(anomalie.debut_ts)


def test_sans_run_id_rien_n_est_etiquete(tmp_path: Path) -> None:
    _injecteur(dossier=tmp_path).injecter("pic_cpu", duree_s=30)
    assert not list(tmp_path.glob("*.csv"))


def test_une_interruption_etiquette_la_duree_atteinte(tmp_path: Path) -> None:
    """Ctrl+C doit produire une etiquette plus courte, jamais une etiquette fausse."""
    instants = iter([1_793_000_000.0, 1_793_000_000.0, 1_793_000_042.0])

    def interrompre(_: float) -> None:
        raise KeyboardInterrupt

    injecteur = Injecteur(
        reglages=_reglages(tmp_path),
        docker=DockerEnregistre(),
        horloge=lambda: next(instants),
        attente=interrompre,
        racine=PROJECT_ROOT,
    )
    anomalie = injecteur.injecter("pic_cpu", duree_s=600, run_id="s2")
    assert anomalie.duree_s == pytest.approx(42.0)
    assert lire_anomalies(tmp_path, "s2")[0]["end_ts"] == horodatage(anomalie.fin_ts)


class DockerQuiTue(DockerEnregistre):
    """Docker ou le conteneur disparait pendant l'injection.

    C'est ce qui arrive avec `make stop-chaos`, avec un OOM-kill, ou si
    `stress-ng` echoue en cours de route.
    """

    def executer(self, arguments: list[str], attendre: bool = True) -> str:
        sortie = super().executer(arguments, attendre)
        if arguments[0] == "inspect":
            # Le premier `inspect` est la verification de demarrage ; le conteneur
            # est tue juste apres, donc le premier sondage de surveillance ne le
            # trouve plus.
            self.inventaire.pop(arguments[-1], None)
        return sortie


def test_un_conteneur_tue_en_cours_route_etiquette_la_duree_subie(tmp_path: Path) -> None:
    """Une injection interrompue vaut l'anomalie subie, pas l'anomalie demandee.

    Mesure a la verification de la phase C : un `chaos-stop` pendant une injection
    de 240 s laissait l'injecteur attendre l'echeance puis etiqueter 242 s, alors
    que la charge s'etait arretee au bout de 90 s. L'etiquette aurait annonce deux
    minutes d'anomalie sur des metriques redevenues normales.
    """
    instants = iter([1_793_000_000.0, 1_793_000_000.0, 1_793_000_030.0])
    docker = DockerQuiTue()
    injecteur = Injecteur(
        reglages=_reglages(tmp_path),
        docker=docker,
        horloge=lambda: next(instants),
        attente=lambda _: None,
        racine=PROJECT_ROOT,
    )
    anomalie = injecteur.injecter("pic_cpu", duree_s=600, run_id="s4")
    assert anomalie.duree_s == pytest.approx(30.0)
    assert lire_anomalies(tmp_path, "s4")[0]["end_ts"] == horodatage(anomalie.fin_ts)


def test_la_surveillance_sonde_le_conteneur_sans_attendre_l_echeance(tmp_path: Path) -> None:
    """La surveillance doit sonder plusieurs fois, sinon elle ne detecte rien."""
    docker = DockerEnregistre()
    injecteur = Injecteur(
        reglages=_reglages(tmp_path),
        docker=docker,
        horloge=lambda: 1_793_000_000.0,
        attente=lambda _: None,
        racine=PROJECT_ROOT,
        pas_surveillance_s=2.0,
    )
    injecteur.injecter("pic_cpu", duree_s=20)
    sondages = [a for a in docker.appels if a[0] == "inspect"]
    # Une verification de demarrage, 20 / 2 = 10 sondages, puis la verification
    # des deux marques faite par l'arret final.
    assert len(sondages) == 12


def test_une_derive_reelle_est_horodatee_dans_son_propre_fichier(tmp_path: Path) -> None:
    injecteur = _injecteur(dossier=tmp_path)
    derive = injecteur.enregistrer_derive("deploiement", "appli de test 1.1.0", "s3")
    assert lire_derives(tmp_path, "s3") == [
        {
            "ts": horodatage(derive.instant_ts),
            "type": "deploiement",
            "description": "appli de test 1.1.0",
        }
    ]
    # Anomalies et derives ne vivent pas dans le meme fichier (§4.2) : une
    # derive n'est pas une anomalie et ne doit jamais etre comptee comme telle.
    assert not (tmp_path / "s3.csv").exists()


# --- Verification de cible du plan de controle ------------------------------
# Le port publie sur l'hote est une variable depuis la phase B : une URL perimee
# pointe vers un service tiers, et une requete envoyee a un service tiers est ce
# que le §4.3 interdit.


def _client(reponse: dict[str, Any] | None, erreur: bool = False) -> Any:
    from infra.injector.app_client import ClientAppliHttp

    client = ClientAppliHttp(url="http://127.0.0.1:65000")

    def sante() -> dict[str, Any]:
        if erreur:
            raise httpx.ConnectError("refuse")
        assert reponse is not None
        return reponse

    client.sante = sante  # type: ignore[method-assign]
    return client


def test_une_url_muette_est_refusee() -> None:
    from infra.injector.app_client import CibleInattendue

    with pytest.raises(CibleInattendue, match="pas repondu comme"):
        _client(None, erreur=True).verifier_cible()


def test_un_service_tiers_sur_le_port_est_refuse() -> None:
    """Le cas reel : 8000 etait deja pris par une application sans rapport."""
    from infra.injector.app_client import CibleInattendue

    with pytest.raises(CibleInattendue, match="pas l'appli de test"):
        _client({"status": "ok", "service": "autre-chose"}).verifier_cible()


def test_un_plan_de_controle_desactive_est_refuse() -> None:
    from infra.injector.app_client import CibleInattendue

    sante = {"statut": "ok", "version": "1.0.0", "chaos_autorise": False}
    with pytest.raises(CibleInattendue, match="desactive"):
        _client(sante).verifier_cible()


def test_l_appli_de_test_est_acceptee() -> None:
    sante = {"statut": "ok", "version": "1.0.0", "chaos_autorise": True}
    assert _client(sante).verifier_cible() == sante


# --- Un conteneur mort-ne ne doit produire aucune etiquette -----------------
# Le defaut reel de la phase C : `docker run` reussissait, stress-ng
# s'interrompait a la premiere seconde faute de `temp-path` inscriptible, et
# l'injecteur ecrivait quand meme une etiquette. Une anomalie annoncee qui n'a
# pas eu lieu est la pire chose qui puisse arriver a un jeu de donnees etiquete.


class DockerQuiEchoue(DockerEnregistre):
    """Docker cree le conteneur, mais le processus s'arrete a la premiere seconde."""

    def executer(self, arguments: list[str], attendre: bool = True) -> str:
        sortie = super().executer(arguments, attendre)
        if arguments[0] == "run":
            nom = arguments[arguments.index("--name") + 1]
            self.mort_nes.add(nom)
            self.journaux[nom] = "aborting: temp-path '.' must be readable and writeable"
        return sortie


def test_un_conteneur_qui_s_arrete_aussitot_est_signale(tmp_path: Path) -> None:
    injecteur = _injecteur(DockerQuiEchoue(), dossier=tmp_path)
    with pytest.raises(ErreurDocker, match="arrete aussitot"):
        injecteur.injecter("pic_cpu", duree_s=30, run_id="s4")
    # Le point du test : aucune etiquette pour une anomalie qui n'a pas eu lieu.
    assert not list(tmp_path.glob("*.csv"))


def test_le_journal_du_conteneur_mort_ne_est_rapporte(tmp_path: Path) -> None:
    """Sans le journal, un echec d'injection est indiagnostiquable."""
    injecteur = _injecteur(DockerQuiEchoue(), dossier=tmp_path)
    with pytest.raises(ErreurDocker, match="temp-path"):
        injecteur.injecter("fuite_memoire", duree_s=30)


def test_le_systeme_de_fichiers_du_stress_est_en_lecture_seule() -> None:
    """Avec un seul espace inscriptible, borne : stress-ng en a besoin."""
    docker = DockerEnregistre()
    _injecteur(docker).injecter("pic_cpu", duree_s=30)
    lancement = _lancement(docker)
    assert "--read-only" in lancement
    assert lancement[lancement.index("--tmpfs") + 1] == "/tmp:size=64m,mode=1777"


def test_un_plan_de_controle_injoignable_est_dit_et_non_avale() -> None:
    """Un arret d'urgence qui annonce une remise a zero qui n'a pas eu lieu est
    un probleme de surete a lui seul. C'est arrive a la verification de la phase
    C : l'URL par defaut pointait vers un service tiers, la verification de
    cible a bloque la requete, et le message annoncait quand meme un succes.
    """

    class ApplieEteinte(ApplieFactice):
        def reinitialiser(self) -> dict[str, Any]:
            raise httpx.ConnectError("connection refused")

    resultat = _injecteur(appli=ApplieEteinte()).arreter_tout()
    assert resultat.plan_de_controle.startswith("inchange")
    # Et les conteneurs ont bien ete arretes malgre cela.
    assert resultat.conteneurs == []
