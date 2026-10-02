from rich.panel import Panel
from rich.table import Table


def banner(console, title: str, text: str):
    console.print(Panel(text, title=f"TORCH ARENA / {title}", border_style="magenta"), markup=False)


def race_table(console, result: dict):
    def fmt(value, spec):
        return format(value, spec) if value is not None else "unavailable"

    banner(
        console,
        "Model Race",
        "Train. Race. Break. Resume.\nSequential real training · " + result["note"],
    )
    table = Table(title="Measured contestants", show_lines=True)
    for row in result["rows"]:
        console.print(
            f"{row['model']} · Dataset: {row['dataset']} · Seed: {row['seed']} · "
            f"Epochs: {row['epochs']}",
            markup=False,
        )
    for heading in ("Model / Run", "Accuracy (%)", "Val loss", "Time (s)", "Parameters", "Epochs"):
        table.add_column(heading)
    for row in result["rows"]:
        table.add_row(
            row["model"] + "\n" + row["run_id"],
            fmt(row["accuracy_fraction"], ".2%"),
            fmt(row["val_loss"], ".6f"),
            fmt(row["seconds"], ".3f"),
            str(row["parameters"]),
            str(row["epochs"]),
        )
    console.print(table)
    for label, winners in result["categories"].items():
        console.print(label + ": " + ", ".join(winners), markup=False)


def runs_table(console, repository, graveyard: bool = False):
    from torcharena.report import recoverability

    table = Table(title="Failure Graveyard" if graveyard else "Experiment history")
    for heading in ("Run / Name", "Status", "Cause", "Epoch / Step", "Recoverable"):
        table.add_column(heading)
    for row in repository.list_runs(graveyard):
        recoverable, _ = recoverability(repository, row["run_id"])
        table.add_row(
            row["run_id"] + "\n" + row["name"],
            row["status"],
            row["failure_type"] or ("user_interrupt" if row["status"] == "INTERRUPTED" else "-"),
            f"{row['last_epoch']} / {row['global_step']}",
            "YES" if recoverable else "NO",
        )
    console.print(table)
