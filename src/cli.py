"""Point d'entree unique des experiences.

Remplace le Makefile sur Windows : une sous-commande = une experience
reproductible (CLAUDE.md, livrable 2 « une commande par experience »).
"""

from __future__ import annotations

import typer

from src.config import PROJECT_ROOT as PROJECT_ROOT_DISPLAY

app = typer.Typer(
    add_completion=False,
    help="Experiences du memoire : detection d'anomalies sous derive de concept.",
)


@app.callback()
def main() -> None:
    """Groupe de commandes (force typer a exposer les sous-commandes)."""


@app.command()
def version() -> None:
    """Affiche la version du paquet et verifie que l'environnement repond."""
    from importlib.metadata import version as pkg_version

    typer.echo(f"memoire-drift {pkg_version('memoire-drift')}")


@app.command("smd-download")
def smd_download(
    config: str = typer.Option("datasets/smd.yaml", help="Configuration du dataset."),
    force: bool = typer.Option(False, "--force", help="Retelecharge meme si present."),
) -> None:
    """Telecharge les machines SMD retenues dans la configuration."""
    from datasets.smd.download import download_smd

    manifest = download_smd(config, force=force)
    total = sum(entry["bytes"] for entry in manifest["files"].values())
    typer.echo(
        f"{len(manifest['files'])} fichiers pour {len(manifest['machines'])} machines "
        f"({total / 1e6:.1f} Mo) dans data/raw/smd."
    )


@app.command("smd-info")
def smd_info(
    config: str = typer.Option("datasets/smd.yaml", help="Configuration du dataset."),
) -> None:
    """Decrit les machines SMD chargees : tailles, durees, anomalies.

    Produit la description des donnees demandee au §11 du cadrage.
    """
    from datasets.smd.loader import load_machines

    machines = load_machines(config)
    typer.echo(
        f"{'machine':<14}{'train':>8}{'test':>8}{'features':>10}"
        f"{'duree (j)':>12}{'anomalies':>12}{'evenements':>12}"
    )
    for machine in machines:
        days = machine.duration_hours(len(machine.train) + len(machine.test)) / 24
        typer.echo(
            f"{machine.name:<14}{len(machine.train):>8}{len(machine.test):>8}"
            f"{machine.n_features:>10}{days:>12.1f}"
            f"{machine.anomaly_rate * 100:>11.2f}%{len(machine.events):>12}"
        )


@app.command("smd-screening")
def smd_screening(
    dataset_config: str = typer.Option(
        "datasets/smd_all.yaml", help="Configuration du dataset (28 machines par defaut)."
    ),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    model_config: str = typer.Option("model.yaml", help="Configuration du modele."),
    screening_config: str = typer.Option("screening.yaml", help="Parametres du criblage."),
    seed: int = typer.Option(42, help="Seed de l'execution."),
) -> None:
    """Quantifie la derive naturelle du SMD machine par machine.

    Ecrit `results/smd_screening.csv` : une ligne par machine, avec le rapport
    entre taux d'alertes observe et taux calibre, l'instant de confirmation de la
    derive et le profil des fausses alertes. Fichier cite au 6.2.1 du memoire
    (OBS03, OBS04) et base du partage machines de mise au point / d'evaluation.
    """
    from src.evaluation.screening import SCREENING_PATH, screen, screening_summary

    typer.echo(
        f"{'machine':<14}{'rapport':>9}{'alertes':>9}{'fausses':>9}"
        f"{'derive (h)':>12}{'KS eleve':>10}{'stable':>8}{'duree (s)':>11}"
    )

    def show(row: dict) -> None:
        onset = row["derive_confirmee_heures"]
        typer.echo(
            f"{row['machine']:<14}{row['rapport_vs_calibre']:>8.2f}x"
            f"{row['taux_alertes_test'] * 100:>8.2f}%"
            f"{row['taux_fausses_alertes'] * 100:>8.2f}%"
            f"{'aucune' if onset is None else f'{onset:.1f}':>12}"
            f"{row['n_metriques_ks_eleve']:>10}"
            f"{'oui' if row['stable'] else 'non':>8}{row['secondes']:>11.1f}"
        )

    table = screen(
        dataset_config=dataset_config,
        features_config=features_config,
        model_config=model_config,
        screening_config=screening_config,
        seed=seed,
        on_row=show,
    )
    summary = screening_summary(table)

    typer.echo(
        f"\n{summary['n_stables']} machines stables sur {summary['n_machines']} "
        f"(rapport mediane {summary['rapport_median']}x, "
        f"de {summary['rapport_min']}x a {summary['rapport_max']}x)."
    )
    typer.echo(
        f"Derive confirmee sur {summary['n_derive_confirmee']} machines, "
        f"dont {summary['n_derive_confirmee_parmi_stables']} parmi les stables."
    )
    typer.echo(f"-> {SCREENING_PATH.relative_to(PROJECT_ROOT_DISPLAY)}")


@app.command("calibration")
def calibration(
    dataset_config: str = typer.Option(
        "datasets/smd_dev.yaml", help="Perimetre de mise au point (D17)."
    ),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    model_config: str = typer.Option("model.yaml", help="Configuration du modele."),
    calibration_config: str = typer.Option("calibration.yaml", help="Parametres du protocole."),
    detail: bool = typer.Option(False, help="Afficher le detail par decision."),
) -> None:
    """Mesure les etapes 3 et 4 de la strategie proposee sur les machines de mise au point.

    Repond a trois questions, sur le seul perimetre de mise au point : reentrainer
    bat-il le champion fige, faut-il exclure les anomalies au reentrainement (D13),
    et un critere sans etiquettes peut-il classer deux challengers (D14) ? Produit
    en outre les bornes du garde-fou de l'etape 4. Ecrit `results/calibration_dev.csv`.

    Prerequis : `mdrift baseline --dataset-config datasets/smd_dev.yaml`.
    """
    from src.config import load_config
    from src.evaluation.calibration import (
        CALIBRATION_PATH,
        calibrate,
        exclusion_verdict,
        guardrail_bounds,
        retrain_verdict,
        validation_verdict,
    )

    typer.echo(
        f"{'machine':<14}{'instant':>14}{'t_r (h)':>10}{'aval':>8}"
        f"{'positifs':>10}{'F1 champion':>13}"
    )

    def show(context: dict) -> None:
        if context.get("raison"):
            typer.echo(
                f"{context['machine']:<14}{context['instant']:>14}"
                f"{context['t_r_heures']:>10.1f}   {context['raison']}"
            )
            return
        typer.echo(
            f"{context['machine']:<14}{context['instant']:>14}{context['t_r_heures']:>10.1f}"
            f"{context['aval_n_fenetres']:>8}"
            f"{context['aval_taux_positif'] * 100:>9.1f}%"
            f"{context['champion_aval_f1']:>13.4f}"
        )

    table = calibrate(
        dataset_config=dataset_config,
        features_config=features_config,
        model_config=model_config,
        calibration_config=calibration_config,
        on_context=show,
    )
    if table.empty:
        typer.echo("Aucune decision exploitable.")
        raise typer.Exit(code=1)

    retrain = retrain_verdict(table)
    typer.echo(
        f"\nReentrainement — sans exclusion, le challenger bat le champion fige sur "
        f"{retrain['n_bat_le_champion']}/{retrain['n_decisions']} decisions ; "
        f"rapport de F1 median x{retrain['gain_median']:.2f} "
        f"(de x{retrain['gain_min']:.2f} a x{retrain['gain_max']:.2f})."
    )

    exclusion = exclusion_verdict(table)
    typer.echo(
        f"\nD13 — exclusion : sur {exclusion['n_decisions']} decisions "
        f"({exclusion['n_machines']} machines), ne rien exclure est optimal "
        f"{exclusion['n_fois_R0_optimale']} fois ; ecart median au meilleur "
        f"{exclusion['ecart_median_au_meilleur']:.4f} point de F1, maximum "
        f"{exclusion['ecart_max_au_meilleur']:.4f}."
    )
    typer.echo(
        f"           ecart-type inter-graines : median "
        f"{exclusion['ecart_type_inter_graines_median']:.4f}, maximum "
        f"{exclusion['ecart_type_inter_graines_max']:.4f}."
    )
    typer.echo(
        f"           exclure toutes les alertes est distincte sur "
        f"{exclusion['n_R1_distincte']} decisions et s'effondre "
        f"(moins de la moitie du F1) sur {exclusion['n_R1_effondrement']} :"
    )
    for case in exclusion["effondrements"]:
        typer.echo(
            f"             {case['machine']:<14}{case['instant']:>12}  garde "
            f"{case['part_gardee'] * 100:5.1f} %  F1 {case['f1_R1']:.4f} "
            f"contre {case['f1_R0']:.4f}  (x{case['ratio']:.2f})"
        )

    validation = validation_verdict(table)
    typer.echo(
        f"\nD14 — validation : sur {validation['n_decisions']} decisions a trois "
        f"challengers distincts ou plus, Spearman median critere/F1 aval "
        f"c1 {validation['rho_c1_median']:+.2f} ({validation['n_rho_c1_positif']} au signe "
        f"positif), c2 {validation['rho_c2_median']:+.2f} "
        f"({validation['n_rho_c2_positif']}). On attend du negatif."
    )
    typer.echo(
        f"           choisir par c1 coute "
        f"{validation['perte_f1_mediane_du_choix_c1'] * 100:.1f} % du F1 atteignable en "
        f"median, jusqu'a {validation['perte_f1_max_du_choix_c1'] * 100:.1f} % ; "
        f"c1 choisit le pire candidat {validation['n_choix_est_le_pire']} fois."
    )
    if detail:
        for case in validation["par_decision"]:
            typer.echo(
                f"             {case['machine']:<14}{case['instant']:>12}"
                f"{case['n_candidats']:>4} cand."
                f"  rho_c1 {case['rho_c1']:+.2f}  rho_c2 {case['rho_c2']:+.2f}"
                f"  F1 choisi {case['f1_du_choix']:.4f} / "
                f"meilleur {case['f1_du_meilleur']:.4f}"
            )

    bounds = guardrail_bounds(table, load_config(model_config))
    typer.echo(
        f"\nGarde-fou — {bounds['n_sains']} challengers sains "
        f"(taux d'alertes sur la fenetre reservee de {bounds['sain_taux_min'] * 100:.2f} % "
        f"a {bounds['sain_taux_max'] * 100:.2f} %), {bounds['n_degrades']} degrades "
        f"(de {bounds['degrade_taux_min'] * 100:.2f} % a "
        f"{bounds['degrade_taux_max'] * 100:.2f} %)."
    )
    if bounds.get("n_degrades"):
        typer.echo(
            f"           bornes proposees : alert_rate_min "
            f"{bounds['alert_rate_min_propose'] * 100:.2f} %, alert_rate_max "
            f"{bounds['alert_rate_max_propose'] * 100:.2f} %, min_train_windows "
            f"{bounds['min_train_windows']}."
        )
        typer.echo(
            f"           elles ecartent {bounds['degrades_ecartes_par_borne_haute']} degrades "
            f"par le haut et {bounds['degrades_ecartes_par_borne_basse']} par le bas ; "
            f"{bounds['degrades_non_ecartes']} degrades passent. "
            f"Separation complete : {'oui' if bounds['separation_complete'] else 'non'}."
        )
    typer.echo(f"-> {CALIBRATION_PATH.relative_to(PROJECT_ROOT_DISPLAY)}")


@app.command("baseline")
def baseline(
    machine: str = typer.Option("all", help="Nom d'une machine SMD, ou 'all'."),
    seed: int = typer.Option(42, help="Seed de l'execution."),
    dataset_config: str = typer.Option("datasets/smd.yaml", help="Configuration du dataset."),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    model_config: str = typer.Option("model.yaml", help="Configuration du modele."),
    evaluation_config: str = typer.Option("evaluation.yaml", help="Configuration des metriques."),
) -> None:
    """Performance de reference en regime stable : strategie B0, sans derive.

    Resultat utilise au 6.2.1 du memoire. Ecrit un dossier par execution sous
    `results/runs/` et une ligne par execution dans `results/summary.csv`.
    """
    from datasets.smd.loader import load_machine, load_machines
    from src.config import load_config
    from src.evaluation.replay import append_summary, replay, save_run, summary_row
    from src.evaluation.streams import build_smd_stream
    from src.strategies.static import StaticStrategy

    configs = {
        "dataset": load_config(dataset_config),
        "features": load_config(features_config),
        "model": load_config(model_config),
        "evaluation": load_config(evaluation_config),
    }

    machines = (
        load_machines(dataset_config)
        if machine == "all"
        else [load_machine(machine, dataset_config)]
    )

    typer.echo(
        f"{'machine':<14}{'fenetres':>10}{'precision':>11}{'rappel':>9}{'F1':>8}"
        f"{'rappel ev.':>12}{'delai (min)':>13}{'FA/h':>8}{'duree (s)':>11}"
    )
    for item in machines:
        stream = build_smd_stream(item, configs["features"])
        outcome = replay(
            stream=stream,
            strategy=StaticStrategy(),
            model_config=configs["model"],
            features_config=configs["features"],
            evaluation_config=configs["evaluation"],
            seed=seed,
        )
        save_run(outcome, stream, configs)
        append_summary(summary_row(outcome, stream, configs))

        windows = outcome.result.windows
        events = outcome.result.events
        typer.echo(
            f"{item.name:<14}{windows.n_windows:>10}{windows.precision:>11.3f}"
            f"{windows.recall:>9.3f}{windows.f1:>8.3f}"
            f"{events.event_recall:>11.2f} {events.detection_delay_mean_minutes:>12.1f}"
            f"{outcome.result.false_alerts_per_hour:>8.2f}{outcome.wall_seconds:>11.1f}"
        )

    typer.echo("\nResultats dans results/summary.csv et results/runs/.")


@app.command("figures")
def figures(
    machine: str = typer.Option("all", help="Nom d'une machine SMD, ou 'all'."),
    strategy: str = typer.Option("static", help="Strategie dont on trace les resultats."),
    scenario: str = typer.Option("none", help="Scenario de derive."),
    seed: int = typer.Option(42, help="Seed de l'execution a tracer."),
    dataset_config: str = typer.Option("datasets/smd.yaml", help="Configuration du dataset."),
) -> None:
    """Produit les figures PNG 300 dpi a partir des traces enregistrees.

    Lit `results/runs/.../traces.npz` : une figure correspond toujours a une
    execution identifiee, jamais a un calcul refait pour l'occasion.
    """
    from src.config import load_config
    from src.evaluation.figures import (
        FIGURES_DIR,
        describe_traces,
        plot_alert_rate_profile,
        plot_scores,
    )
    from src.evaluation.replay import RESULTS_DIR

    configuration = load_config(dataset_config)
    noms = configuration["machines"] if machine == "all" else [machine]
    intervalle = int(configuration["sampling_interval_seconds"])

    for nom in noms:
        traces = (
            RESULTS_DIR / "runs" / "smd" / nom / scenario / strategy / f"seed{seed}" / "traces.npz"
        )
        if not traces.is_file():
            typer.echo(f"{nom} : trace absente, lancer d'abord `mdrift baseline`.")
            raise typer.Exit(code=1)

        chiffres = describe_traces(traces, intervalle)
        base = f"{nom}_{scenario}_{strategy}_seed{seed}"
        titre = f"{nom} — strategie {strategy}, scenario {scenario} (seed {seed})"

        scores = plot_scores(
            traces,
            FIGURES_DIR / f"scores_{base}.png",
            title=titre,
            sampling_interval_seconds=intervalle,
        )
        taux = plot_alert_rate_profile(
            traces,
            FIGURES_DIR / f"taux_alertes_{base}.png",
            title=f"Taux d'alertes glissant — {titre}",
            sampling_interval_seconds=intervalle,
        )
        typer.echo(
            f"{nom} : {chiffres['alert_rate_percent']:.2f}% d'alertes pour "
            f"{chiffres['positive_rate_percent']:.2f}% de fenetres positives "
            f"sur {chiffres['duration_hours']:.0f} h"
        )
        typer.echo(f"  -> {scores.relative_to(PROJECT_ROOT_DISPLAY)}")
        typer.echo(f"  -> {taux.relative_to(PROJECT_ROOT_DISPLAY)}")


if __name__ == "__main__":
    app()
