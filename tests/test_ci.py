"""Les deux fichiers d'integration continue doivent verifier la meme chose.

Le depot porte deux declencheurs de CI pour une raison expliquee en D44 : le
cadrage demande GitLab CI, l'hebergement est GitHub. Deux fichiers qui decrivent
le meme controle finissent par diverger en silence — c'est exactement le defaut
qui a coute la campagne M3 (D43), ou une valeur calibree dans une decision et
une autre valeur figee dans une configuration paraissaient coherentes prises
separement. Ces tests tiennent l'invariant : le Makefile est la seule definition
de ce qui est verifie, et les deux CI appellent ses cibles.
"""

from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[1]
GITLAB = RACINE / ".gitlab-ci.yml"
GITHUB = RACINE / ".github/workflows/ci.yml"
MAKEFILE = RACINE / "Makefile"

CIBLES = ("lint", "test", "build")


def _texte(chemin: Path) -> str:
    return chemin.read_text(encoding="utf-8")


def test_les_deux_fichiers_de_ci_existent() -> None:
    assert GITLAB.is_file(), "le livrable demande par le cadrage (M7)"
    assert GITHUB.is_file(), "celui qui tourne reellement et produit la capture"


@pytest.mark.parametrize("cible", CIBLES)
def test_le_makefile_declare_les_trois_cibles_de_la_ci(cible: str) -> None:
    makefile = _texte(MAKEFILE)
    assert f"\n{cible}:\n" in makefile, cible
    # `.PHONY` sans la cible et un fichier du meme nom suffirait a rendre la
    # cible muette, make la considerant a jour.
    ligne_phony = next(ligne for ligne in makefile.splitlines() if ligne.startswith(".PHONY:"))
    assert cible in ligne_phony.split(), (cible, ligne_phony)


@pytest.mark.parametrize("cible", CIBLES)
def test_les_deux_ci_appellent_les_cibles_du_makefile(cible: str) -> None:
    """Chaque CI doit passer par `make`, pas recopier les commandes.

    GitLab appelle `ruff` et `pytest` directement pour que l'etage qui echoue se
    lise dans l'interface ; la verification porte donc sur la **commande** pour
    `build` et sur la presence du controle pour les deux autres.
    """
    assert f"make {cible}" in _texte(GITHUB), cible
    gitlab = _texte(GITLAB)
    if cible == "build":
        assert "make build" in gitlab
    elif cible == "lint":
        assert "ruff check ." in gitlab and "ruff format --check ." in gitlab
    else:
        assert "pytest" in gitlab


def test_les_deux_ci_epinglent_la_meme_version_de_uv() -> None:
    """Deux CI qui resolvent les dependances differemment ne prouvent rien.

    La version de `uv` est epinglee des deux cotes, et les deux doivent etre la
    meme : c'est elle qui lit `uv.lock`.
    """
    import re

    versions = set()
    for chemin in (GITLAB, GITHUB):
        trouvees = re.findall(r'uv==?([0-9]+\.[0-9]+\.[0-9]+)"?', _texte(chemin))
        trouvees += re.findall(r'version:\s*"([0-9]+\.[0-9]+\.[0-9]+)"', _texte(chemin))
        assert trouvees, f"aucune version de uv epinglee dans {chemin.name}"
        versions.update(trouvees)
    assert len(versions) == 1, versions


def test_les_dependances_sont_installees_depuis_le_verrou() -> None:
    """`uv sync` sans `--frozen` resoudrait a nouveau, et la CI testerait autre
    chose que ce qui a produit les resultats des chapitres 6 (D24)."""
    for chemin in (GITLAB, GITHUB):
        assert "uv sync --frozen" in _texte(chemin), chemin.name


def test_la_ci_ne_rejoue_pas_la_campagne_et_ne_telecharge_pas_le_smd() -> None:
    """Garde-fou de duree et de perimetre.

    Une CI qui lancerait `campaign` ou `smd-download` prendrait des heures et
    ferait entrer les donnees dans un agent, ce que le §4 du cadrage exclut. Si
    cela devient souhaitable un jour, ce sera une decision, pas une ligne
    ajoutee au detour d'un commit.
    """
    for chemin in (GITLAB, GITHUB):
        texte = _texte(chemin)
        for interdit in ("mdrift campaign", "smd-download", "mdrift figures"):
            assert interdit not in texte, (chemin.name, interdit)


def test_aucun_secret_ni_registre_dans_les_fichiers_de_ci() -> None:
    """§4.3 : aucun jeton, aucune adresse, et rien qui pousse vers un registre."""
    for chemin in (GITLAB, GITHUB):
        texte = _texte(chemin)
        for interdit in ("docker push", "TELEGRAM_BOT_TOKEN", "CI_REGISTRY_PASSWORD"):
            assert interdit not in texte, (chemin.name, interdit)


def test_les_deux_fichiers_de_ci_sont_du_yaml_valide() -> None:
    """Un `.gitlab-ci.yml` casse ne se voit qu'au prochain push sur GitLab, que
    personne ne fera avant la soutenance : il est verifie ici."""
    for chemin in (GITLAB, GITHUB):
        contenu = yaml.safe_load(_texte(chemin))
        assert isinstance(contenu, dict), chemin.name

    gitlab = yaml.safe_load(_texte(GITLAB))
    etages = gitlab["stages"]
    jobs = {
        nom: valeur
        for nom, valeur in gitlab.items()
        if isinstance(valeur, dict) and "stage" in valeur
    }
    assert jobs, "aucun job declare"
    for nom, job in jobs.items():
        assert job["stage"] in etages, (nom, job["stage"])

    github = yaml.safe_load(_texte(GITHUB))
    etapes = github["jobs"]["controles"]["steps"]
    assert any("make lint" in str(etape.get("run", "")) for etape in etapes)
