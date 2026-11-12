"""Point d'entree unique des experiences.

Remplace le Makefile sur Windows : une sous-commande = une experience
reproductible (CLAUDE.md, livrable 2 « une commande par experience »).
"""

from __future__ import annotations

import typer

from src.config import PROJECT_ROOT as PROJECT_ROOT_DISPLAY
from src.config import load_config

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


@app.command("inject")
def inject_command(
    dataset_config: str = typer.Option(
        "datasets/smd_dev.yaml", help="Configuration du dataset (mise au point par defaut)."
    ),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    scenario: str = typer.Option("all", help="Nom d'un scenario, ou 'all' pour S1 a S4."),
    seeds: str = typer.Option("42", help="Seeds separees par des virgules."),
) -> None:
    """Decrit la derive injectee sur chaque machine, scenario et graine.

    Ecrit `results/injections.csv` : instant de derive, metriques touchees, duree
    de transition, part ecretee. C'est la description des donnees derivees
    demandee au §11 du cadrage, et la verification que l'injection se comporte
    comme prevu sur les donnees reelles avant toute campagne.
    """
    from scenarios.injection import SCENARIOS
    from src.evaluation.injection import INJECTIONS_PATH, audit

    choisis = list(SCENARIOS) if scenario == "all" else [scenario]
    graines = [int(s) for s in seeds.split(",") if s.strip()]

    typer.echo(
        f"{'machine':<14}{'scenario':<18}{'graine':>7}{'t_d (h)':>9}{'position':>10}"
        f"{'transition (h)':>16}{'metriques':>11}{'ecrete':>9}"
    )

    def show(row: dict) -> None:
        typer.echo(
            f"{row['machine']:<14}{row['scenario']:<18}{row['seed']:>7}"
            f"{row['drift_heures']:>9.1f}{row['drift_position']:>10.3f}"
            f"{row['duree_transition_heures']:>16.1f}{row['n_metriques_touchees']:>11}"
            f"{row['part_ecretee'] * 100:>8.2f}%"
            f"{'  !' if row['ecretage_au_dela_tolerance'] else ''}"
        )

    table = audit(
        dataset_config=dataset_config,
        features_config=features_config,
        scenarios=choisis,
        seeds=graines,
        on_row=show,
    )
    typer.echo(
        f"\n{len(table)} injections sur {table['machine'].nunique()} machines, "
        f"{table['scenario'].nunique()} scenarios, {table['seed'].nunique()} graines."
    )
    typer.echo(
        f"Part ecretee : mediane {table['part_ecretee'].median() * 100:.2f} %, "
        f"max {table['part_ecretee'].max() * 100:.2f} %."
    )
    au_dela = int(table["ecretage_au_dela_tolerance"].sum())
    if au_dela:
        typer.echo(
            f"{au_dela} injections sur {len(table)} depassent la tolerance d'ecretage "
            "(marquees !) : le garde-fou les refuserait en campagne. "
            "C'est ce que le calibrage de k doit corriger."
        )
    typer.echo(f"-> {INJECTIONS_PATH.relative_to(PROJECT_ROOT_DISPLAY)}")


@app.command("calibration-k")
def calibration_k(
    dataset_config: str = typer.Option(
        "datasets/smd_dev.yaml", help="Perimetre de mise au point. NE PAS pointer smd_eval.yaml."
    ),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    model_config: str = typer.Option("model.yaml", help="Configuration du modele."),
    evaluation_config: str = typer.Option("evaluation.yaml", help="Configuration des metriques."),
    calibration_config: str = typer.Option(
        "calibration_k.yaml", help="Grille de k et criteres de lecture."
    ),
) -> None:
    """Calibre l'amplitude k de la derive injectee (etape 5 de M1, decision D22).

    Mesure la degradation du F1 de B0 de part et d'autre de t_d pour chaque k de
    la grille, sur les machines de mise au point uniquement (D17). Ecrit
    `results/calibration_k_dev.csv` et affiche le tableau de lecture.
    """
    from src.config import load_config
    from src.evaluation.injection import K_CALIBRATION_PATH, calibrate_k, k_verdict

    typer.echo(
        f"{'machine':<14}{'k':>6}{'graine':>8}{'F1 avant':>10}{'F1 apres':>10}"
        f"{'ratio':>8}{'alertes apres':>15}{'ecrete':>9}"
    )

    def show(row: dict) -> None:
        ratio = row["ratio_f1"]
        typer.echo(
            f"{row['machine']:<14}{row['k']:>6.1f}{row['seed']:>8}"
            f"{row['f1_avant'] or 0:>10.3f}{row['f1_apres'] or 0:>10.3f}"
            f"{'-' if ratio is None else f'{ratio:.2f}':>8}"
            f"{(row['taux_alertes_apres'] or 0) * 100:>14.1f}%"
            f"{row['part_ecretee'] * 100:>8.2f}%"
        )

    table = calibrate_k(
        dataset_config=dataset_config,
        features_config=features_config,
        model_config=model_config,
        evaluation_config=evaluation_config,
        calibration_config=calibration_config,
        on_row=show,
    )
    resume = k_verdict(table, load_config(calibration_config))

    typer.echo("\nLecture par k (moyenne par machine d'abord, voir OBS06) :")
    typer.echo(resume.to_string(index=False))
    typer.echo(f"\n-> {K_CALIBRATION_PATH.relative_to(PROJECT_ROOT_DISPLAY)}")


@app.command("scenario-figures")
def scenario_figures(
    machine: str = typer.Option(
        "machine-1-1", help="Machine de mise au point servant d'illustration."
    ),
    dataset_config: str = typer.Option("datasets/smd_dev.yaml", help="Configuration du dataset."),
    seed: int = typer.Option(42, help="Graine de l'injection illustree."),
    metric: int | None = typer.Option(
        None, help="Indice de metrique a tracer ; par defaut la plus visible."
    ),
) -> None:
    """Produit les figures des quatre scenarios de derive (livrable M1).

    Deux PNG dans `docs/figures/` : les profils temporels a(t) seuls, et l'effet
    de l'injection sur une metrique reelle du SMD.
    """
    from datasets.smd.loader import load_machine
    from scenarios.injection import SCENARIOS
    from src.evaluation.scenario_figures import (
        load_scenarios,
        plot_injection_grid,
        plot_profiles,
    )

    scenarios = load_scenarios(list(SCENARIOS))

    profils = plot_profiles(scenarios)
    typer.echo(f"-> {profils.relative_to(PROJECT_ROOT_DISPLAY)}")

    smd_machine = load_machine(machine, dataset_config)
    grille, descriptions = plot_injection_grid(
        machine=smd_machine, scenarios=scenarios, seed=seed, metric=metric
    )
    typer.echo(f"-> {grille.relative_to(PROJECT_ROOT_DISPLAY)}")

    typer.echo("")
    typer.echo(f"{machine}, graine {seed} :")
    for ligne in descriptions:
        typer.echo(
            f"  {ligne['scenario']:<16} metrique {ligne['metrique']:>2}  "
            f"t_d = {ligne['t_d_heures']:>6.1f} h  "
            f"{ligne['n_metriques_touchees']} metriques touchees  "
            f"ecrete {ligne['part_ecretee_percent']:.2f} %"
        )


def _drift_positions(
    machine: str, scenario: str, strategy: str, seed: int
) -> tuple[int | None, int | None]:
    """Instant et fin de transition de la derive injectee, lus dans `summary.csv`.

    Renvoie `(None, None)` quand le rejeu n'est pas dans la synthese ou quand le
    scenario n'injecte rien : une figure sans marque vaut mieux qu'une marque
    inventee.
    """
    import pandas as pd

    from src.evaluation.replay import SUMMARY_PATH

    if not SUMMARY_PATH.is_file():
        return None, None
    table = pd.read_csv(SUMMARY_PATH)
    lignes = table[
        (table.machine == machine)
        & (table.scenario == scenario)
        & (table.strategy == strategy)
        & (table.seed == seed)
    ]
    if lignes.empty or "drift_window" not in lignes.columns:
        return None, None
    debut = lignes.drift_window.iloc[0]
    if pd.isna(debut):
        return None, None
    # `drift_end_index` est un indice de POINT, `drift_window` un indice de
    # FENETRE : il faut convertir, sinon la bande de transition de S2 et S3 est
    # decalee de la taille de la fenetre. La relation est celle du fenetrage,
    # verifiee a l'identique sur les 960 rejeux de la campagne.
    fin_point = lignes.drift_end_index.iloc[0] if "drift_end_index" in lignes.columns else None
    if fin_point is None or pd.isna(fin_point):
        return int(debut), None
    taille = int(lignes.window_size.iloc[0])
    pas = int(lignes.stride.iloc[0])
    return int(debut), int((int(fin_point) - taille + 1) // pas)


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

        # L'instant de derive n'est pas dans les traces : il appartient au
        # scenario, donc a la ligne de synthese du rejeu. On le relit la plutot
        # que de le recalculer, pour que la figure et le tableau du memoire ne
        # puissent pas se contredire.
        derive, derive_fin = _drift_positions(nom, scenario, strategy, seed)

        scores = plot_scores(
            traces,
            FIGURES_DIR / f"scores_{base}.png",
            title=titre,
            sampling_interval_seconds=intervalle,
            drift_index=derive,
            drift_end_index=derive_fin,
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


_RUN_HEADER = (
    f"{'machine':<14}{'scenario':<16}{'strategie':<11}{'gr.':>5}"
    f"{'F1':>7}{'F1 av.':>8}{'F1 ap.':>8}{'recup.':>9}"
    f"{'reent.':>8}{'prom.':>7}{'rej.':>6}{'conf.':>7}{'delai':>8}{'duree':>8}"
)


def _show_run(row: dict) -> None:
    """Une ligne par rejeu. Les None s'affichent `-` : ils ont un sens (§7)."""

    def num(key: str, fmt: str = ".3f") -> str:
        value = row.get(key)
        return "-" if value is None or value != value else format(value, fmt)

    typer.echo(
        f"{row['machine']:<14}{row['scenario']:<16}{row['strategy']:<11}{row['seed']:>5}"
        f"{num('f1'):>7}{num('f1_before'):>8}{num('f1_after'):>8}"
        f"{num('recovery_minutes', '.0f'):>9}"
        f"{row['cost_n_retrains']:>8}{row['cost_n_promoted']:>7}{row['cost_n_rejected']:>6}"
        f"{row['drift_n_confirmations']:>7}{num('drift_detection_delay_minutes', '.0f'):>8}"
        f"{row['total_seconds']:>8.1f}"
    )


def _show_campaign_summary(table) -> None:
    """Resume d'une campagne : une ligne par strategie, moyenne sur les machines.

    La moyenne est prise **par machine d'abord** (OBS06) : les machines d'un meme
    groupe SMD sont correlees, et moyenner directement sur les rejeux donnerait
    plus de poids aux groupes qui comptent le plus de machines.
    """
    if table.empty:
        typer.echo("Aucun rejeu execute (tout etait deja dans summary.csv).")
        return
    par_machine = table.groupby(["strategy", "machine"], as_index=False).mean(numeric_only=True)
    # Les colonnes dependent du scenario : un rejeu sans derive injectee n'a ni
    # `f1_after` ni delai de detection de derive. On n'agrege donc que ce qui est
    # present, au lieu de supposer la campagne toujours identique — c'est ce qui
    # faisait echouer `--scenarios none` apres les 64 rejeux, une fois les
    # resultats deja ecrits.
    voulues = {
        "f1": "f1",
        "f1_apres": "f1_after",
        "reentrainements": "cost_n_retrains",
        "promus": "cost_n_promoted",
        "rejetes": "cost_n_rejected",
        "secondes_reentrainement": "cost_retrain_seconds",
        "confirmations": "drift_n_confirmations",
        "promotions_degradantes": "n_promotions_degradantes",
    }
    agregats = {"machines": ("machine", "nunique")}
    agregats.update(
        {
            libelle: (colonne, "mean")
            for libelle, colonne in voulues.items()
            if colonne in par_machine.columns
        }
    )
    resume = par_machine.groupby("strategy", as_index=False).agg(**agregats)
    typer.echo("\nMoyenne par strategie (moyenne par machine d'abord, OBS06) :")
    typer.echo(resume.round(3).to_string(index=False))


@app.command("figures-m3")
def figures_m3(
    summary_path: str = typer.Option("results/summary.csv", help="Synthese de la campagne."),
    machine: str = typer.Option(
        "mediane",
        help="Machine du F1 glissant. `mediane` = celle dont le F1 apres derive de P "
        "est le plus proche de la mediane des machines (choix reproductible, "
        "et non la plus flatteuse).",
    ),
    scenarios: str = typer.Option(
        "s1_soudaine,s4_recurrente",
        help="Scenarios pour lesquels tracer le F1 glissant.",
    ),
    seed: int = typer.Option(42, help="Graine du F1 glissant."),
    largeur: int = typer.Option(1000, help="Largeur de la fenetre glissante, en fenetres."),
) -> None:
    """Produit les figures de resultats du chapitre 6 (livrable M3, phase D).

    Quatre figures d'agregat — cout du reentrainement, F1 de part et d'autre de la
    derive, arbitrage qualite / cout, effet de K sur la confirmation — plus un F1
    glissant par scenario demande. Tout est relu depuis `results/` : aucune figure
    n'est produite par un calcul refait pour l'occasion, et aucune valeur n'est
    recopiee a la main.
    """
    from pathlib import Path

    import pandas as pd

    from src.evaluation.result_figures import (
        plot_confirmations_contre_k,
        plot_cout_reentrainement,
        plot_f1_avant_apres,
        plot_f1_glissant,
        plot_qualite_contre_cout,
    )
    from src.evaluation.result_tables import ecrire_tableaux

    synthese = pd.read_csv(summary_path)
    injectes = synthese[synthese.scenario != "none"]
    if injectes.empty:
        typer.echo("Aucun rejeu avec derive injectee dans la synthese.")
        raise typer.Exit(code=1)

    sorties: list[Path] = []
    for tracer in (plot_cout_reentrainement, plot_f1_avant_apres, plot_qualite_contre_cout):
        sorties.append(tracer(summary_path=summary_path))

    figure_k, mesures_k = plot_confirmations_contre_k()
    sorties.append(figure_k)
    typer.echo("Effet de K (point de fonctionnement fige, machines de mise au point) :")
    typer.echo(
        mesures_k[
            [
                "n_injections",
                "detections",
                "confirmations_prematurees",
                "confirmations_stables",
                "part_detectee",
            ]
        ].to_string()
    )

    for scenario in [nom.strip() for nom in scenarios.split(",") if nom.strip()]:
        bloc = injectes[(injectes.scenario == scenario) & (injectes.strategy == "proposed")]
        if bloc.empty:
            typer.echo(f"{scenario} : aucun rejeu de la strategie proposee, scenario ignore.")
            continue
        if machine == "mediane":
            par_machine = bloc.groupby("machine").f1_after.mean()
            choisie = (par_machine - par_machine.median()).abs().idxmin()
        else:
            choisie = machine
        sorties.append(
            plot_f1_glissant(
                choisie,
                scenario,
                seed=seed,
                largeur=largeur,
                summary_path=summary_path,
            )
        )
        typer.echo(f"{scenario} : F1 glissant trace sur {choisie} (graine {seed}).")

    sorties.append(ecrire_tableaux(summary_path=summary_path))

    typer.echo("")
    for chemin in sorties:
        typer.echo(f"-> {chemin.relative_to(PROJECT_ROOT_DISPLAY)}")


@app.command("run")
def run_command(
    machine: str = typer.Option(..., help="Nom d'une machine SMD."),
    strategy: str = typer.Option(
        "proposed", help="Fichier de configs/strategies/ : static, periodic, proposed, ablation."
    ),
    scenario: str = typer.Option("none", help="Scenario de derive, ou 'none'."),
    seed: int = typer.Option(42, help="Graine de l'execution."),
    dataset_config: str = typer.Option(
        "datasets/smd_dev.yaml", help="Perimetre. NE PAS pointer smd_eval.yaml avant le gel."
    ),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    model_config: str = typer.Option("model.yaml", help="Configuration du modele."),
    evaluation_config: str = typer.Option("evaluation.yaml", help="Configuration des metriques."),
    drift_config: str = typer.Option("drift.yaml", help="Configuration du moniteur de derive."),
) -> None:
    """Rejoue un flux pour une strategie : un rejeu, une ligne de resultat.

    C'est la commande de travail : un triplet (machine, scenario, graine) et une
    strategie. Elle recalcule toujours et remplace la ligne correspondante de
    `results/summary.csv` ; pour enchainer des centaines de rejeux reprenables,
    utiliser `mdrift campaign`.
    """
    from src.evaluation.campaign import run_campaign

    typer.echo(_RUN_HEADER)
    run_campaign(
        dataset_config=dataset_config,
        features_config=features_config,
        model_config=model_config,
        evaluation_config=evaluation_config,
        drift_config=drift_config,
        strategies=[strategy],
        scenarios=[scenario],
        seeds=[seed],
        machines=[machine],
        resume=False,
        on_row=_show_run,
    )
    typer.echo("\n-> results/summary.csv et results/runs/.")


@app.command("campaign")
def campaign_command(
    strategies: str = typer.Option(
        "static,periodic,proposed",
        help="Fichiers de configs/strategies/, separes par des virgules.",
    ),
    scenarios: str = typer.Option("none", help="Scenarios, separes par des virgules."),
    seeds: str = typer.Option("42", help="Graines, separees par des virgules."),
    machine: str = typer.Option("all", help="Nom d'une machine, ou 'all' pour tout le perimetre."),
    dataset_config: str = typer.Option(
        "datasets/smd_dev.yaml", help="Perimetre. NE PAS pointer smd_eval.yaml avant le gel."
    ),
    features_config: str = typer.Option("features.yaml", help="Configuration des features."),
    model_config: str = typer.Option("model.yaml", help="Configuration du modele."),
    evaluation_config: str = typer.Option("evaluation.yaml", help="Configuration des metriques."),
    drift_config: str = typer.Option("drift.yaml", help="Configuration du moniteur de derive."),
    force: bool = typer.Option(
        False, "--force", help="Recalcule meme les rejeux deja dans summary.csv."
    ),
    summary_path: str = typer.Option(
        "",
        help="Fichier de synthese. Vide = results/summary.csv. A renseigner pour une "
        "campagne de mise au point, qui n'a pas sa place dans la synthese du memoire.",
    ),
) -> None:
    """Enchaine machines x scenarios x strategies x graines, de facon reprenable.

    Chaque rejeu est ecrit des qu'il est fini : interrompre la campagne ne perd
    rien, et la relancer reprend la ou elle s'etait arretee (les `run_id` deja
    presents dans `results/summary.csv` sont sautes). Toutes les strategies d'un
    meme triplet (machine, scenario, graine) rejouent le **meme** flux.
    """
    from pathlib import Path

    from src.evaluation.campaign import completed_run_ids, plan, run_campaign
    from src.evaluation.replay import SUMMARY_PATH

    sortie = SUMMARY_PATH if not summary_path else Path(summary_path)
    noms = (
        load_config(dataset_config)["machines"]
        if machine == "all"
        else [m.strip() for m in machine.split(",") if m.strip()]
    )
    strategy_names = [s.strip() for s in strategies.split(",") if s.strip()]
    scenario_names = [s.strip() for s in scenarios.split(",") if s.strip()]
    seed_values = [int(s) for s in seeds.split(",") if s.strip()]

    jobs = plan(noms, scenario_names, strategy_names, seed_values)
    deja = set() if force else completed_run_ids(sortie)
    restants = [job for job in jobs if job.run_id not in deja]
    typer.echo(
        f"{len(jobs)} rejeux au programme ({len(noms)} machines x {len(scenario_names)} "
        f"scenarios x {len(strategy_names)} strategies x {len(seed_values)} graines), "
        f"{len(jobs) - len(restants)} deja faits, {len(restants)} a executer."
    )
    if not restants:
        raise typer.Exit(code=0)
    typer.echo("")
    typer.echo(_RUN_HEADER)

    table = run_campaign(
        dataset_config=dataset_config,
        features_config=features_config,
        model_config=model_config,
        evaluation_config=evaluation_config,
        drift_config=drift_config,
        strategies=strategy_names,
        scenarios=scenario_names,
        seeds=seed_values,
        machines=noms,
        resume=not force,
        on_row=_show_run,
        summary_path=sortie,
    )
    _show_campaign_summary(table)
    typer.echo(f"\n-> {sortie} et results/runs/.")


@app.command("calibrate-drift")
def calibrate_drift(
    dataset_config: str = typer.Option(
        "datasets/smd_dev.yaml",
        help="Perimetre. NE PAS pointer smd_eval.yaml : le calibrage lit les etiquettes (D17).",
    ),
    features_config: str = typer.Option("features.yaml"),
    drift_config: str = typer.Option("drift.yaml"),
    scenarios: str = typer.Option(
        "none,s1_soudaine,s2_incrementale,s3_graduelle,s4_recurrente",
        help="Scenarios balayes. `none` mesure la specificite, les autres la sensibilite.",
    ),
    seeds: str = typer.Option("42", help="Graines. Sans effet sur `none`."),
    machine: str = typer.Option("all", help="Une machine, plusieurs separees par une virgule."),
    quantiles: str = typer.Option("0.99,0.999,1.0", help="Quantiles du modele nul balayes."),
    margins: str = typer.Option("1.0,1.25,1.5,2.0", help="Marges sur le seuil par feature."),
    shares: str = typer.Option("0.10,0.15,0.20,0.30,0.40", help="Parts declenchantes balayees."),
    confirmations: str = typer.Option("240,360,480,600,720,960", help="Valeurs de K, en fenetres."),
    recent_windows: str = typer.Option(
        "",
        help="Valeurs de R balayees. Vide = celle de la config. Chaque valeur coute un "
        "parcours complet du perimetre : R change la trace, pas seulement une comparaison.",
    ),
    stable: str = typer.Option(
        "machine-1-3,machine-2-7,machine-3-10",
        help="Machines stables du perimetre : la specificite n'est comptee que sur elles (OBS03).",
    ),
    path: str = typer.Option(
        "",
        help="Fichier de sortie. Vide = results/calibration_m2_dev.csv. A renseigner pour "
        "un balayage cible, sinon il ecrase la grille large.",
    ),
) -> None:
    """Balaie la grille des parametres de derive sur le perimetre de mise au point.

    Produit `results/calibration_m2_dev.csv` (une ligne par machine x scenario x
    graine x point de grille) et affiche le tableau de decision : fausses
    confirmations sur les machines stables sans injection, contre taux de
    detection et delai sur les scenarios injectes.

    Le flux de chaque (machine, scenario, graine) est parcouru **une fois** ; les
    points de grille sont ensuite evalues sur les statistiques conservees.
    """
    from pathlib import Path

    from src.evaluation.drift_calibration import DRIFT_CALIBRATION_PATH, summarise
    from src.evaluation.drift_calibration import calibrate_drift as run_calibration

    noms = None if machine == "all" else [nom.strip() for nom in machine.split(",")]
    sortie = DRIFT_CALIBRATION_PATH if not path else Path(path)

    def floats(texte: str) -> tuple[float, ...]:
        return tuple(float(valeur) for valeur in texte.split(","))

    def ints(texte: str) -> tuple[int, ...]:
        return tuple(int(valeur) for valeur in texte.split(","))

    table, run = run_calibration(
        dataset_config=dataset_config,
        features_config=features_config,
        drift_config=drift_config,
        scenarios=[nom.strip() for nom in scenarios.split(",")],
        seeds=ints(seeds),
        machines=noms,
        quantiles=floats(quantiles),
        margins=floats(margins),
        shares=floats(shares),
        confirmations=ints(confirmations),
        recent_windows=None if not recent_windows else ints(recent_windows),
        path=sortie,
        on_trace=lambda info: typer.echo(
            f"  {info['machine']:<14} {info['scenario']:<16} seed {info['seed']:<3} "
            f"R {info['recent_windows']:<5} {info['n_windows']} fenetres, "
            f"{info['n_features_usable']} features, {info['n_blocks']} blocs nuls, "
            f"t_d {info['drift_window']}, {info['trace_seconds']}s"
        ),
    )
    typer.echo(
        f"\n{run.n_traces} parcours, {run.n_rows} points de grille, {run.seconds / 60:.1f} min."
    )

    noms_stables = [nom.strip() for nom in stable.split(",")]
    resume = summarise(table, noms_stables)
    typer.echo("\nTableau de decision (trie : zero fausse confirmation d'abord).")
    typer.echo(f"Machines stables du perimetre : {', '.join(noms_stables)}.")
    colonnes = [
        "recent_windows",
        "quantile",
        "margin",
        "drifting_share",
        "confirmation_windows",
        "faux_signaux_stables",
        "n_traces_stables",
        "fausses_confirmations_stables",
        "confirmations_derive_naturelle",
        "detections",
        "n_injections",
        "delai_median",
    ]
    colonnes = [nom for nom in colonnes if nom in resume.columns]
    typer.echo(resume[colonnes].head(25).to_string(index=False))
    typer.echo(f"\n-> {sortie}")


@app.command("promotions")
def promotions_command(
    runs_dir: str = typer.Option("results/runs", help="Racine des traces de rejeu."),
    delta: str = typer.Option(
        "",
        help="Valeurs de delta balayees, separees par des virgules. Vide = celle de "
        "configs/evaluation.yaml. Plusieurs valeurs montrent la sensibilite du comptage.",
    ),
    evaluation_config: str = typer.Option("evaluation.yaml"),
    scenarios: str = typer.Option("", help="Filtre. Vide = tous les scenarios presents."),
    path: str = typer.Option("", help="Fichier de sortie. Vide = results/promotions.csv."),
) -> None:
    """Recalcule les promotions degradantes hors ligne, depuis `traces.npz`.

    Le calcul utilise les etiquettes : il ne pouvait donc pas vivre dans la
    strategie, et son cout est exclu du cout de reentrainement (D16). Comme tout
    repart des traces, changer delta ne coute pas un rejeu — ce qui est le seul
    moyen honnete de montrer qu'une conclusion ne tient pas au reglage de ce
    seuil. Une ligne par promotion, pour que la **distribution** des ecarts
    demandee au §7 soit dans le fichier et pas seulement son comptage.
    """
    from pathlib import Path

    import pandas as pd

    from src.evaluation.promotions import collect_promotions
    from src.evaluation.replay import RESULTS_DIR

    seuils = (
        [float(load_config(evaluation_config).get("promotions", {}).get("delta_f1", 0.05))]
        if not delta
        else [float(valeur) for valeur in delta.split(",") if valeur.strip()]
    )
    sortie = RESULTS_DIR / "promotions.csv" if not path else Path(path)
    filtres = [nom.strip() for nom in scenarios.split(",") if nom.strip()]

    lignes = []
    for seuil in seuils:
        lignes.extend(collect_promotions(runs_dir, delta_f1=seuil))
    table = pd.DataFrame(lignes)
    if table.empty:
        typer.echo("Aucune promotion trouvee. Les rejeux ont-ils ete executes ?")
        raise typer.Exit(code=1)
    if filtres:
        table = table[table["scenario"].isin(filtres)]
    sortie.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(sortie, index=False)

    reference = table[table.delta_f1_seuil == seuils[0]]
    typer.echo(
        f"{len(reference)} promotions sur {reference.run_id.nunique()} rejeux, "
        f"{reference.machine.nunique()} machines."
    )
    typer.echo("\nEcarts F1(promu) - F1(remplace) sur les N fenetres suivantes :")
    typer.echo(
        reference.groupby("strategy")["delta_f1"]
        .describe(percentiles=[0.05, 0.5, 0.95])
        .round(4)
        .to_string()
    )
    typer.echo("\nPromotions degradantes, par strategie et par delta :")
    compte = (
        table.groupby(["delta_f1_seuil", "strategy"])
        .agg(promotions=("degrading", "size"), degradantes=("degrading", "sum"))
        .reset_index()
    )
    compte["part"] = (compte.degradantes / compte.promotions).round(4)
    typer.echo(compte.to_string(index=False))
    typer.echo(f"\n-> {sortie}")


@app.command("hypotheses")
def hypotheses_command(
    summary_path: str = typer.Option("results/summary.csv", help="Synthese a analyser."),
    alpha: float = typer.Option(0.05, help="Risque de premiere espece, apres correction."),
    scenarios: str = typer.Option("", help="Filtre. Vide = tous les scenarios injectes."),
    path: str = typer.Option("", help="Sortie. Vide = results/hypotheses.csv."),
    groupes: bool = typer.Option(True, help="Afficher aussi les moyennes par groupe SMD."),
) -> None:
    """Teste les trois hypotheses du memoire (Wilcoxon apparie par machine).

    Le plan de tests est **declare dans le code** (`src/evaluation/hypotheses.py`,
    dictionnaire `PLAN`) et non choisi apres avoir vu les resultats : metriques de
    chaque hypothese, sens attendu, correction de Holm sur la famille. L'unite
    d'observation est la machine, moyennee sur ses graines (OBS06), et chaque
    scenario est teste separement avant la ligne d'ensemble.
    """
    from pathlib import Path

    import pandas as pd

    from src.evaluation.hypotheses import HYPOTHESES_PATH, PLAN, ecrire, par_groupe, tester

    filtres = [nom.strip() for nom in scenarios.split(",") if nom.strip()]
    resultats = tester(summary_path, alpha=alpha, scenarios=filtres or None)
    if resultats.empty:
        typer.echo("Aucun rejeu avec derive injectee dans la synthese.")
        raise typer.Exit(code=1)

    for hypothese, plan in PLAN.items():
        bloc = resultats[resultats.hypothese == hypothese]
        if bloc.empty:
            continue
        par_scenario = bloc[bloc.scenario != "tous"]
        machines = int(par_scenario.n_paires.max()) if not par_scenario.empty else 0
        typer.echo("")
        typer.echo(f"=== {hypothese} — {plan['question']}")
        typer.echo(
            f"    P contre {plan['reference']} — {machines} machines par scenario, "
            f"{int(bloc.n_paires.max())} paires sur la ligne `tous`"
        )
        # `med. P` et `med. ref` sont des medianes de niveaux, `med. ecart` la
        # mediane des ecarts apparies : les trois ne se deduisent pas l'une de
        # l'autre et peuvent meme ne pas avoir le meme signe. `+/-` compte les
        # machines ou P depasse la reference et celles ou il est en dessous ; c'est
        # ce comptage, et non la mediane, qui porte la direction du test.
        typer.echo(
            f"{'scenario':<16}{'metrique':<28}{'med. P':>10}{'med. ref':>10}"
            f"{'med. ecart':>12}{'+/-':>9}{'p':>9}{'p Holm':>9}  verdict"
        )
        for ligne in bloc.itertuples():
            p_brut = "-" if pd.isna(ligne.p_value) else f"{ligne.p_value:.4f}"
            p_holm = "-" if pd.isna(ligne.p_holm) else f"{ligne.p_holm:.4f}"
            signes = f"{ligne.n_ecarts_positifs}/{ligne.n_ecarts_negatifs}"
            typer.echo(
                f"{ligne.scenario:<16}{ligne.metrique:<28}"
                f"{ligne.mediane_strategie:>10.3f}{ligne.mediane_reference:>10.3f}"
                f"{ligne.mediane_ecart:>12.4f}{signes:>9}{p_brut:>9}{p_holm:>9}  {ligne.verdict}"
            )

    if groupes:
        table_groupes = par_groupe(summary_path)
        if not table_groupes.empty:
            typer.echo("")
            typer.echo("=== Moyennes par groupe SMD (les machines d'un groupe sont correlees)")
            typer.echo(table_groupes.to_string(index=False))

    sortie = HYPOTHESES_PATH if not path else Path(path)
    ecrire(resultats, sortie)
    typer.echo("")
    typer.echo(f"-> {sortie}")


# --- Injection d'anomalies sur le serveur (livrable M4, phase C) -------------
# Les garde-fous du §4.3 vivent dans `infra/injector/injector.py` et non ici :
# le CLI ne doit etre qu'une facon de les appeler, jamais une facon de les
# contourner. Imports tardifs, comme partout dans ce fichier, pour que
# `mdrift --help` ne charge ni Docker ni httpx.


def _injecteur(config: str, avec_appli: bool = True):
    """Construit l'injecteur depuis `configs/injection.yaml`."""
    from infra.injector.app_client import ClientAppliHttp
    from infra.injector.docker_cli import DockerCli
    from infra.injector.injector import Injecteur, Reglages

    reglages = Reglages.depuis_config(load_config(config), PROJECT_ROOT_DISPLAY)
    appli = ClientAppliHttp(reglages.url_appli) if avec_appli else None
    return Injecteur(
        reglages=reglages,
        docker=DockerCli(),
        appli=appli,
        racine=PROJECT_ROOT_DISPLAY,
    )


@app.command("chaos")
def chaos_command(
    type_: str = typer.Argument(
        ..., metavar="TYPE", help="pic_cpu, fuite_memoire, latence, erreurs, saturation."
    ),
    intensite: float = typer.Option(
        None, "--intensite", help="Defaut : celui du type dans la configuration."
    ),
    duree: float = typer.Option(None, "--duree", help="Duree en secondes. Defaut : celle du type."),
    run_id: str = typer.Option(
        None, "--run-id", help="Si fourni, l'anomalie est etiquetee dans data/labels/."
    ),
    config: str = typer.Option("injection.yaml", help="Configuration des injections."),
) -> None:
    """Injecte une anomalie sur l'appli de test, attend sa fin, et l'etiquette.

    La commande est bloquante : elle rend la main quand l'injection est finie,
    et c'est a ce moment que l'etiquette est ecrite, avec la duree reellement
    subie. Ctrl+C arrete l'injection et etiquette la duree atteinte (§4.2).
    """
    from infra.injector.injector import RefusDeSecurite
    from infra.injector.labels import horodatage

    injecteur = _injecteur(config)
    try:
        anomalie = injecteur.injecter(type_, intensite, duree, run_id)
    except RefusDeSecurite as refus:
        typer.echo(f"Refus : {refus}", err=True)
        raise typer.Exit(code=2) from refus
    # Les memes horodatages que le fichier d'etiquettes, et non les flottants :
    # c'est cette forme que l'API de Prometheus accepte telle quelle.
    typer.echo(
        f"{anomalie.type} intensite={anomalie.intensite:g} "
        f"duree={anomalie.duree_s:.0f}s "
        f"{horodatage(anomalie.debut_ts)} -> {horodatage(anomalie.fin_ts)}"
    )
    if run_id:
        typer.echo(f"-> data/labels/{run_id}.csv")
    else:
        typer.echo("Aucun --run-id : l'anomalie n'a pas ete etiquetee.")


@app.command("chaos-list")
def chaos_list(
    config: str = typer.Option("injection.yaml", help="Configuration des injections."),
) -> None:
    """Liste les conteneurs d'injection en cours."""
    injecteur = _injecteur(config, avec_appli=False)
    noms = injecteur.lister()
    if not noms:
        typer.echo("Aucune injection par conteneur en cours.")
        return
    for nom in noms:
        typer.echo(nom)


@app.command("chaos-stop")
def chaos_stop(
    nom: str = typer.Option(None, "--nom", help="Un seul conteneur. Par defaut, tous."),
    config: str = typer.Option("injection.yaml", help="Configuration des injections."),
) -> None:
    """Arret d'urgence : arrete les injections en cours (§4.3, `make stop-chaos`).

    Sans `--nom`, arrete tous les conteneurs d'injection et remet le plan de
    controle de l'appli de test a zero. L'injecteur refuse tout conteneur qui ne
    porte pas ses deux marques, y compris pour un arret.
    """
    from infra.injector.injector import RefusDeSecurite

    injecteur = _injecteur(config)
    try:
        if nom:
            injecteur.arreter(nom)
            typer.echo(f"Arrete : {nom}")
            return
        resultat = injecteur.arreter_tout()
    except RefusDeSecurite as refus:
        typer.echo(f"Refus : {refus}", err=True)
        raise typer.Exit(code=2) from refus
    for arrete in resultat.conteneurs:
        typer.echo(f"Arrete : {arrete}")
    typer.echo(f"{len(resultat.conteneurs)} conteneur(s) arrete(s).")
    typer.echo(f"Plan de controle de l'appli de test : {resultat.plan_de_controle}")


@app.command("drift-note")
def drift_note(
    type_: str = typer.Argument(..., metavar="TYPE", help="deploiement, ressources, trafic."),
    description: str = typer.Argument(..., help="Ce qui a change, en une phrase."),
    run_id: str = typer.Option(..., "--run-id", help="Session concernee."),
    config: str = typer.Option("injection.yaml", help="Configuration des injections."),
) -> None:
    """Horodate une derive reelle sur le serveur (§4.2).

    A lancer juste apres le changement : un deploiement d'une nouvelle version
    de l'appli de test, un changement de limites d'un conteneur, une montee
    durable du trafic. L'instant de la derive est la seule chose qu'on ne peut
    pas retrouver apres coup, et c'est ce dont l'evaluation a besoin.
    """
    injecteur = _injecteur(config, avec_appli=False)
    from infra.injector.labels import horodatage

    derive = injecteur.enregistrer_derive(type_, description, run_id)
    typer.echo(f"{horodatage(derive.instant_ts)} {derive.type} : {derive.description}")
    typer.echo(f"-> data/labels/{run_id}_drift.csv")


def _reglages_serveur(config: str):
    """Construit les reglages de la source `serveur` depuis sa configuration."""
    from datasets.serveur.settings import ReglagesServeur

    return ReglagesServeur.depuis_config(load_config(config), PROJECT_ROOT_DISPLAY)


@app.command("session-check")
def session_check(
    config: str = typer.Option("datasets/serveur.yaml", help="Configuration de la source."),
) -> None:
    """Verifie que la pile de collecte est prete a enregistrer une session.

    A lancer **avant** une session, pas apres : une cible qui n'a jamais tourne
    coute la session entiere, et cela ne se voit qu'au moment de l'enregistrer.
    La commande verifie aussi, metrique par metrique, que l'expression rend bien
    une valeur maintenant : une expression juste mais qui ne correspond a aucune
    serie est le defaut le plus couteux, parce qu'il ne se voit pas.
    """
    import time

    import numpy as np

    from datasets.serveur.recorder import ErreurSession, verifier_cibles
    from src.serving.prometheus import ClientPrometheus, ErreurPrometheus

    reglages = _reglages_serveur(config)
    client = ClientPrometheus(reglages.url)
    try:
        etats = verifier_cibles(client, reglages)
    except (ErreurSession, ErreurPrometheus) as erreur:
        typer.echo(f"Non pret : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    for cible in sorted(etats):
        typer.echo(f"  cible {cible:<12} {etats[cible]}")

    # Une plage de deux pas suffit : le but est de savoir si l'expression rend
    # quelque chose, pas de mesurer quoi que ce soit.
    fin = time.time()
    debut = fin - 2 * reglages.pas_s
    indisponibles: list[str] = []
    for metrique in reglages.metriques:
        try:
            serie = client.interroger_plage(
                metrique.expression(reglages.rate_interval), debut, fin, reglages.pas_s
            )
        except ErreurPrometheus as erreur:
            typer.echo(f"  {metrique.nom:<18} EXPRESSION REFUSEE : {erreur}", err=True)
            indisponibles.append(metrique.nom)
            continue
        if serie.part_manquante >= 1.0:
            typer.echo(
                f"  {metrique.nom:<18} "
                + ("VIDE (obligatoire)" if metrique.obligatoire else "vide (optionnelle)")
            )
            if metrique.obligatoire:
                indisponibles.append(metrique.nom)
        else:
            derniere = serie.valeurs[~np.isnan(serie.valeurs)][-1]
            typer.echo(f"  {metrique.nom:<18} {derniere:.4g}")

    if indisponibles:
        typer.echo(
            f"{len(indisponibles)} metrique(s) obligatoire(s) indisponible(s) : "
            f"{indisponibles}. Enregistrer maintenant produirait une session inutilisable.",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo("Pret a enregistrer une session.")


@app.command("session-record")
def session_record(
    run_id: str = typer.Argument(..., help="Identifiant de la session (nom des fichiers)."),
    debut: str = typer.Option(..., "--debut", help="2026-11-11T08:00:00Z, -3h, ou secondes Unix."),
    fin: str = typer.Option("-0s", "--fin", help="Idem. Defaut : maintenant."),
    config: str = typer.Option("datasets/serveur.yaml", help="Configuration de la source."),
    sans_controle: bool = typer.Option(
        False,
        "--sans-controle",
        help="N'exige pas que les cibles soient `up` (plage deja passee).",
    ),
) -> None:
    """Enregistre une plage de Prometheus dans data/sessions/<run_id>.parquet.

    La session est enregistree une fois et rejouee pour chaque strategie : c'est
    la seule facon de comparer B0, B1 et P sur des donnees reelles, puisqu'on ne
    peut pas faire subir deux fois la meme heure a un serveur.
    """
    from datasets.serveur.recorder import (
        ErreurSession,
        enregistrer,
        instant_depuis_texte,
        verifier_cibles,
    )
    from src.serving.prometheus import ClientPrometheus, ErreurPrometheus

    reglages = _reglages_serveur(config)
    client = ClientPrometheus(reglages.url)
    try:
        if not sans_controle:
            verifier_cibles(client, reglages)
        session = enregistrer(
            client,
            reglages,
            debut=instant_depuis_texte(debut),
            fin=instant_depuis_texte(fin),
            run_id=run_id,
        )
    except (ErreurSession, ErreurPrometheus) as erreur:
        typer.echo(f"Session non enregistree : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    typer.echo(
        f"{session.n_points} points au pas de {reglages.pas_s}s "
        f"({session.n_points * reglages.pas_s / 3600:.2f} h) -> {session.donnees.name}"
    )
    troues = {nom: part for nom, part in session.parts_manquantes.items() if part > 0}
    for nom in sorted(troues, key=lambda n: -troues[n]):
        typer.echo(f"  {nom:<18} {troues[nom]:.1%} de points manquants")
    if not troues:
        typer.echo("  aucun trou de collecte.")
    typer.echo(f"Metadonnees : {session.metadonnees.name}")


@app.command("session-info")
def session_info(
    run_id: str = typer.Argument(..., help="Session a decrire."),
    config: str = typer.Option("datasets/serveur.yaml", help="Configuration de la source."),
    features: str = typer.Option("features.yaml", help="Configuration des features."),
) -> None:
    """Charge une session et decrit le flux qu'elle produit (§11, point 5).

    Rend la description des donnees reelles que le memoire demande : duree,
    nombre de fenetres, anomalies etiquetees, derives horodatees, et ce que le
    chargement a du corriger ou signaler.
    """
    from datasets.serveur.loader import charger_session
    from datasets.serveur.recorder import ErreurSession
    from src.evaluation.streams import build_server_stream

    reglages = _reglages_serveur(config)
    try:
        session = charger_session(run_id, reglages)
    except ErreurSession as erreur:
        typer.echo(f"Session illisible : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    flux = build_server_stream(session, load_config(features))
    n_points = len(session.train) + len(session.test)
    typer.echo(f"Session {run_id} : {n_points} points, {session.n_features} metriques")
    typer.echo(
        f"  duree           {session.duration_hours(n_points):.2f} h "
        f"au pas de {session.sampling_interval_seconds}s"
    )
    typer.echo(f"  entrainement    {len(session.train)} points (ni anomalie, ni derive)")
    typer.echo(
        f"  flux evalue     {len(session.test)} points, {len(flux)} fenetres, "
        f"{session.anomaly_rate:.2%} de points anormaux"
    )
    typer.echo(f"  evenements      {len(session.events)} anomalie(s) etiquetee(s)")
    for evenement in session.events:
        typer.echo(f"    points {evenement.start}-{evenement.end} ({evenement.length} point(s))")
    if flux.drift_index is None:
        typer.echo("  derive          aucune (scenario `none`)")
    else:
        typer.echo(
            f"  derive          point {flux.drift_index} du flux evalue "
            f"(fenetre {flux.drift_window()}), scenario `{flux.scenario}`"
        )
        for derive in session.derives:
            typer.echo(f"    {derive['ts']} {derive['type']} : {derive['description']}")

    interpolees = {nom: part for nom, part in session.parts_interpolees.items() if part > 0}
    for nom in sorted(interpolees, key=lambda n: -interpolees[n]):
        typer.echo(f"  interpole       {nom:<18} {interpolees[nom]:.2%}")
    if session.anomalies_sans_point:
        typer.echo(
            f"  ATTENTION       {len(session.anomalies_sans_point)} anomalie(s) plus courte(s) "
            "que le pas : aucun point de grille ne tombe dedans, donc aucune strategie "
            "ne peut les detecter."
        )
        for ligne in session.anomalies_sans_point:
            typer.echo(f"    {ligne['start_ts']} -> {ligne['end_ts']} {ligne.get('type', '?')}")


def _registre(serving: str):
    """Construit l'acces au registre depuis `configs/serving.yaml`."""
    from src.tracking.registry import RegistreModeles, ReglagesRegistre

    return RegistreModeles(ReglagesRegistre.depuis_config(load_config(serving)))


@app.command("registry-init")
def registry_init(
    run_id: str = typer.Argument(..., help="Session enregistree servant a l'entrainement initial."),
    config: str = typer.Option("datasets/serveur.yaml", help="Configuration de la source."),
    features: str = typer.Option("features.yaml", help="Configuration des features."),
    model: str = typer.Option("model.yaml", help="Configuration du modele."),
    serving: str = typer.Option("serving.yaml", help="Configuration du mode en ligne."),
    seed: int = typer.Option(42, help="Seed du modele initial."),
    sans_promotion: bool = typer.Option(
        False,
        "--sans-promotion",
        help="Enregistre la version sans lui donner l'alias champion.",
    ),
) -> None:
    """Entraine le modele initial sur la part calme d'une session et le promeut champion.

    Le service en ligne refuse de demarrer sans champion : il faut bien que le
    premier modele vienne de quelque part, et il doit venir de la **meme** part
    d'entrainement que celle qu'utiliserait un rejeu de cette session. C'est ce
    qui rend comparables la demonstration du chapitre 5 et les chiffres du
    chapitre 6 : meme session, meme part calme, meme code d'entrainement.
    """
    from datasets.serveur.loader import charger_session
    from datasets.serveur.recorder import ErreurSession
    from src.evaluation.streams import build_server_stream
    from src.features.signature import signature_depuis_config
    from src.models.isolation_forest import build_detector
    from src.tracking.registry import ErreurRegistre

    features_config = load_config(features)
    reglages = _reglages_serveur(config)
    try:
        session = charger_session(run_id, reglages)
    except ErreurSession as erreur:
        typer.echo(f"Session illisible : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    flux = build_server_stream(session, features_config)
    signature = signature_depuis_config(
        session.metric_names, features_config, session.sampling_interval_seconds
    )
    if signature.n_features != flux.train_X.shape[1]:
        # Garde-fou de coherence interne : si la signature deduite de la
        # configuration ne decrit pas la matrice que `make_windows` vient de
        # produire, c'est le calcul de signature qui est faux, et l'enregistrer
        # rendrait tous les controles ulterieurs inutiles.
        typer.echo(
            f"Incoherence interne : signature de {signature.n_features} features pour une "
            f"matrice de {flux.train_X.shape[1]} colonnes.",
            err=True,
        )
        raise typer.Exit(code=2)

    detecteur = build_detector(load_config(model), features_config, seed)
    detecteur.fit(flux.train_X)
    typer.echo(
        f"Modele initial entraine sur {detecteur.n_train_windows} fenetres "
        f"({len(session.train)} points calmes de {run_id}), "
        f"seuil {detecteur.threshold:.4f}, {detecteur.fit_seconds:.2f}s"
    )

    registre = _registre(serving)
    try:
        version = registre.enregistrer(
            detecteur,
            signature=signature,
            raison="initial_fit",
            parametres={"seed": seed, "source": "serveur", "session": run_id},
            metriques={"n_train_points": float(len(session.train))},
            tags={"session": run_id, "source": "serveur"},
            # La matrice d'entrainement part avec le modele : c'est la reference
            # du moniteur de derive. Un service qui redemarre doit comparer les
            # fenetres recentes a ce sur quoi le champion a appris, pas aux
            # donnees du moment, sinon une derive survenue pendant l'arret ne
            # sera jamais vue.
            reference_X=flux.train_X,
        )
        typer.echo(f"Enregistre : {version.resume()}")
        if not sans_promotion:
            version = registre.promouvoir(version)
            typer.echo(f"Promu : alias `{registre.reglages.alias}` -> version {version.version}")
    except ErreurRegistre as erreur:
        typer.echo(f"Registre : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur


@app.command("registry-info")
def registry_info(
    serving: str = typer.Option("serving.yaml", help="Configuration du mode en ligne."),
    limite: int = typer.Option(10, help="Nombre de versions a lister."),
) -> None:
    """Dit quel modele est en service et liste les versions enregistrees.

    C'est la commande a lancer avant une demonstration : un service qui tourne
    sur un champion different de celui qu'on croit rend les captures du
    chapitre 5 inexploitables.
    """
    from src.tracking.registry import ErreurRegistre

    registre = _registre(serving)
    typer.echo(f"Registre {registre.reglages.tracking_uri}, modele {registre.reglages.model_name}")
    try:
        courant = registre.champion()
        versions = registre.historique(limite)
    except ErreurRegistre as erreur:
        typer.echo(f"Registre : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    if courant is None:
        typer.echo(
            f"  aucun modele ne porte l'alias `{registre.reglages.alias}` : le service "
            "refusera de demarrer. Lancer `mdrift registry-init <run_id>`."
        )
    else:
        typer.echo(f"  en service      {courant.resume()}")
        if courant.signature is not None:
            typer.echo(
                f"  features        {courant.signature.n_features} colonnes, "
                f"empreinte {courant.signature.empreinte}, "
                f"fenetre {courant.signature.window_size} x pas "
                f"{courant.signature.sampling_interval_seconds}s"
            )
        if courant.a_une_reference:
            typer.echo(f"  reference       {courant.reference_n_windows} fenetres d'entrainement")
        else:
            typer.echo(
                "  reference       ABSENTE : le service refusera de demarrer, le moniteur "
                "de derive n'aurait rien a quoi comparer les fenetres recentes."
            )

    typer.echo(f"  versions        {len(versions)}")
    for version in versions:
        marque = "*" if courant is not None and version.version == courant.version else " "
        typer.echo(
            f"   {marque} v{version.version:<4} {version.cree_le}  {version.raison:<12} "
            f"seuil={version.threshold:.4f} n_train={version.n_train_windows}"
        )


@app.command("serve")
def serve(
    config: str = typer.Option("datasets/serveur.yaml", help="Configuration de la source."),
    features: str = typer.Option("features.yaml", help="Configuration des features."),
    model: str = typer.Option("model.yaml", help="Configuration du modele."),
    drift: str = typer.Option("drift.yaml", help="Configuration des detecteurs de derive."),
    strategie: str = typer.Option("", help="Strategie appliquee (defaut : celle de serving.yaml)."),
    serving: str = typer.Option("serving.yaml", help="Configuration du mode en ligne."),
    seed: int = typer.Option(42, help="Seed des challengers reentraines."),
    duree_min: float = typer.Option(
        0.0, help="Duree de service en minutes (0 = jusqu'a interruption)."
    ),
    une_fois: bool = typer.Option(
        False, "--une-fois", help="Un seul tick, puis arret : verification de configuration."
    ),
    sans_promotion: bool = typer.Option(
        False,
        "--sans-promotion",
        help="Reentraine et valide, mais ne met rien en service : repetition avant demonstration.",
    ),
    run_id: str = typer.Option("", help="Identifiant du run (defaut : horodate)."),
    sans_exporter: bool = typer.Option(
        False,
        "--sans-exporter",
        help="N'ouvre pas le port /metrics : utile quand un autre service l'occupe deja.",
    ),
) -> None:
    """Lance le service de detection en ligne sur la pile de collecte.

    C'est le mode en ligne du §2 : Prometheus -> features -> champion -> score,
    surveillance de derive, reentrainement, validation, promotion. Le modele en
    service vient du registre et y retourne ; le journal d'evenements ecrit ici
    est ce que liront l'exporter et les notifications.

    A lancer apres `mdrift session-check` (la collecte repond) et
    `mdrift registry-info` (un champion existe et porte la bonne signature) :
    ces deux commandes disent en une seconde ce que le service mettrait une
    minute a decouvrir.
    """
    from src.serving.exporter import ErreurExporteur, ExporteurDetection, ReglagesExporteur
    from src.serving.loop import BoucleEnLigne, ErreurBoucle, ReglagesBoucle
    from src.serving.prometheus import ClientPrometheus, ErreurPrometheus
    from src.strategies.factory import build_strategy
    from src.tracking.registry import ErreurRegistre

    serving_config = load_config(serving)
    try:
        reglages_boucle = ReglagesBoucle.depuis_config(serving_config, PROJECT_ROOT_DISPLAY)
    except ErreurBoucle as erreur:
        typer.echo(f"Configuration : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    # La source et la strategie viennent de serving.yaml, sauf si la ligne de
    # commande les surcharge : une demonstration doit pouvoir montrer B0 ou B1
    # sans modifier un fichier suivi par git.
    reglages = _reglages_serveur(config if config else reglages_boucle.source)
    features_config = load_config(features)
    strategy = build_strategy(
        load_config(strategie or reglages_boucle.strategie),
        features_config,
        load_config(drift),
        reglages.pas_s,
    )

    try:
        reglages_exporteur = ReglagesExporteur.depuis_config(serving_config)
    except ErreurExporteur as erreur:
        typer.echo(f"Configuration : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    exporteur = ExporteurDetection(
        strategie=strategy.name, budget_ms=float(reglages.pas_s) * 1000.0
    )

    boucle = BoucleEnLigne(
        client=ClientPrometheus(reglages.url),
        registre=_registre(serving),
        reglages_serveur=reglages,
        reglages_boucle=reglages_boucle,
        strategy=strategy,
        features_config=features_config,
        model_config=load_config(model),
        seed=seed,
        run_id=run_id or None,
        promotion=not sans_promotion,
        exporteur=exporteur,
    )

    typer.echo(f"Service {boucle.run_id}, strategie {strategy.name}, pas {reglages.pas_s}s")
    # L'exporter est ouvert **avant** le demarrage : si le champion manque, la
    # cible Prometheus repond quand meme, avec `mdrift_service_up` a 0, et le
    # tableau de bord montre un service qui n'a pas demarre au lieu d'un trou.
    if reglages_exporteur.actif and not sans_exporter:
        try:
            exporteur.servir(reglages_exporteur)
        except OSError as erreur:
            typer.echo(f"Exporter : port indisponible ({erreur}).", err=True)
            raise typer.Exit(code=2) from erreur
        typer.echo(
            f"  exporter        http://{reglages_exporteur.hote}:{reglages_exporteur.port}/metrics"
        )
    else:
        typer.echo("  exporter        desactive")
    try:
        boucle.demarrer()
    except (ErreurBoucle, ErreurRegistre, ErreurPrometheus) as erreur:
        typer.echo(f"Demarrage impossible : {erreur}", err=True)
        raise typer.Exit(code=2) from erreur

    typer.echo(
        f"  champion        version {boucle.version.version}, "
        f"seuil {boucle.champion.threshold:.4f}, "
        f"empreinte features {boucle.signature.empreinte}"
    )
    typer.echo(f"  historique      {boucle.n_fenetres_vues} fenetres prechargees (non scorees)")
    typer.echo(f"  journal         {boucle.journal.chemin}")
    if sans_promotion:
        typer.echo("  promotion       DESACTIVEE : les challengers valides ne seront pas promus")

    try:
        boucle.executer(
            duree_s=duree_min * 60.0 if duree_min > 0 else None,
            max_ticks=1 if une_fois else None,
        )
    except KeyboardInterrupt:  # pragma: no cover - interruption manuelle
        typer.echo("\nInterrompu.")

    typer.echo(
        f"Arret : {boucle.n_fenetres_vues} fenetres, {boucle.n_alertes} alertes, "
        f"{boucle.n_sautees} sautees, {boucle.n_reentrainements} reentrainements "
        f"({boucle.n_promotions} promus, {boucle.n_rejets} rejetes)"
    )
    if boucle.mesures.n:
        typer.echo(
            f"  window_ms       moyenne {boucle.mesures.moyenne:.1f}, "
            f"p95 {boucle.mesures.p95:.1f} (budget : {reglages.pas_s * 1000} ms)"
        )


if __name__ == "__main__":
    app()
