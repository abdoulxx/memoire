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
