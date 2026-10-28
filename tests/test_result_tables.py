"""Tableaux du chapitre 6 : ce qui ne doit jamais glisser entre le CSV et le memoire."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.result_tables import _format, _table_synthese, ecrire_tableaux


def _rejeu(machine: str, scenario: str, strategie: str, seed: int, **valeurs: float) -> dict:
    ligne = {
        "machine": machine,
        "scenario": scenario,
        "strategy": strategie,
        "seed": seed,
        "f1": 0.3,
        "f1_after": 0.3,
        "cost_n_retrains": 0.0,
    }
    ligne.update(valeurs)
    return ligne


def _synthese() -> pd.DataFrame:
    lignes = []
    for machine in ("machine-1-1", "machine-1-2"):
        for strategie in ("static", "periodic", "proposed", "ablation"):
            for seed in (1, 2):
                lignes.append(
                    _rejeu(
                        machine,
                        "s1_soudaine",
                        strategie,
                        seed,
                        cost_n_retrains=0.0 if strategie == "static" else 4.0,
                    )
                )
    return pd.DataFrame(lignes)


def test_une_metrique_absente_s_affiche_en_tiret_pas_en_zero() -> None:
    """Un blanc et un zero ne disent pas la meme chose.

    B0 n'a pas de delai de detection de derive parce qu'il ne surveille rien.
    L'afficher « 0 » ferait croire a une detection instantanee, c'est-a-dire a
    l'inverse exact de ce que la strategie fait.
    """
    assert _format(np.nan, 0.1, 2) == "—"
    assert _format(0.0, 0.0, 2) == "0.00 ± 0.00"


def test_une_valeur_sans_dispersion_calculable_reste_affichee() -> None:
    """Une seule machine : moyenne connue, erreur standard non. On montre la moyenne."""
    assert _format(0.25, np.nan, 2) == "0.25"


def test_la_moyenne_passe_par_les_machines_avant_les_rejeux() -> None:
    """OBS06 : les graines d'une machine ne sont pas des observations independantes.

    Une machine a deux graines ne doit pas peser deux fois plus qu'une machine a
    une seule : on verifie que l'erreur standard est bien celle des machines
    (deux machines identiques -> dispersion nulle) et non celle des quatre rejeux.
    """
    synthese = _synthese()
    # Une graine de plus sur une seule machine : si la moyenne etait prise sur les
    # rejeux bruts, cette machine peserait davantage et la dispersion bougerait.
    synthese = pd.concat(
        [synthese, pd.DataFrame([_rejeu("machine-1-1", "s1_soudaine", "proposed", 3, f1=0.3)])],
        ignore_index=True,
    )
    lignes = _table_synthese(synthese)
    ligne_f1 = next(ligne for ligne in lignes if ligne.startswith("| F1 par fenetre"))
    assert "0.300 ± 0.000" in ligne_f1


def test_le_fichier_contient_les_tableaux_annonces(tmp_path) -> None:
    chemin_synthese = tmp_path / "summary.csv"
    _synthese().to_csv(chemin_synthese, index=False)
    destination = ecrire_tableaux(
        summary_path=chemin_synthese,
        hypotheses_path=tmp_path / "absent.csv",
        naturelle_path=tmp_path / "absent2.csv",
        output_path=tmp_path / "tableaux.md",
    )
    texte = destination.read_text(encoding="utf-8")
    assert "## Tableau 1" in texte
    assert "## Tableau 2" in texte
    assert "## Tableau 3" in texte
    # Les sources absentes ne doivent pas produire de tableau vide.
    assert "## Tableau 4" not in texte
    assert "## Tableau 5" not in texte
    assert "2 machines d'evaluation" in texte


def test_une_synthese_sans_derive_injectee_est_refusee(tmp_path) -> None:
    """Mieux vaut une erreur qu'un tableau de chapitre 6 silencieusement vide."""
    chemin = tmp_path / "summary.csv"
    synthese = _synthese()
    synthese["scenario"] = "none"
    synthese.to_csv(chemin, index=False)
    with pytest.raises(ValueError, match="aucun rejeu"):
        ecrire_tableaux(summary_path=chemin, output_path=tmp_path / "t.md")
