import json
import os
import shutil
import sqlite3
import sys
from functools import wraps
from pathlib import Path
from typing import Annotated

import torch
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from torcharena.comparison import compare
from torcharena.comparison import race as run_race
from torcharena.config import load_config
from torcharena.report import inspect_data
from torcharena.report import report as make_report
from torcharena.safety import TrainingError, preflight
from torcharena.service import resume as resume_run
from torcharena.service import train as train_run
from torcharena.showcase import showcase as run_showcase
from torcharena.storage import Repository
from torcharena.ui import banner, race_table, runs_table

app = typer.Typer(no_args_is_help=True, help="TorchArena — Train. Race. Break. Resume.")
console = Console(markup=False, highlight=False)


def home() -> Path:
    return Path(os.environ.get("TORCHARENA_HOME", ".torcharena"))


def handled(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (
            ValueError,
            OSError,
            sqlite3.Error,
            yaml.YAMLError,
            ValidationError,
            TrainingError,
        ) as error:
            console.print(f"FAILED: {error}", style="bold red")
            for note in getattr(error, "__notes__", []):
                console.print(note)
            raise typer.Exit(1) from error

    return wrapper


@app.command()
@handled
def doctor():
    """Read-only runtime inspection; does not create the DB or artifacts."""
    target = home().resolve()
    parent = target
    while not parent.exists():
        parent = parent.parent
    writable = os.access(parent, os.W_OK)
    python_ok = sys.version_info >= (3, 11)
    db = target / "torcharena.db"
    db_status = "NOT CREATED (lazy initialization)"
    if db.exists():
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()[0]
        db_status = "PASS" if result == "ok" else "FAILED: " + result
    table = Table(title="TorchArena Doctor")
    table.add_column("Check")
    table.add_column("Result")
    for label, result in [
        ("Python", ("PASS" if python_ok else "FAIL") + " " + sys.version.split()[0]),
        ("PyTorch", str(torch.__version__)),
        ("TorchArena DB", db_status),
        ("Device", "CUDA" if torch.cuda.is_available() else "CPU"),
        ("CUDA", "available" if torch.cuda.is_available() else "unavailable"),
        ("Writable artifacts", "PASS (permission check)" if writable else "FAIL"),
        ("Git", "detected" if shutil.which("git") else "not detected"),
        (
            "Docker",
            "client detected (engine unchecked)" if shutil.which("docker") else "not detected",
        ),
    ]:
        table.add_row(label, result)
    console.print(table)
    if not python_ok or not writable or db_status.startswith("FAILED"):
        raise typer.Exit(1)
    console.print("READY · CPU is supported")


@app.command()
@handled
def validate(config: Path):
    """Exercise a temporary model, batch, backward pass and optimizer step."""
    checks = preflight(load_config(config))
    console.print(json.dumps(checks, indent=2))
    console.print("READY TO TRAIN")


@app.command()
@handled
def train(config: Path):
    repository = Repository(home())
    run_id = train_run(repository, load_config(config))
    record = repository.get(run_id)
    banner(
        console,
        "Training",
        f"{record['status']}: {run_id}\n" + json.dumps(record["summary"], indent=2),
    )
    if record["status"] == "INTERRUPTED":
        console.print(f"Resume: torcharena resume {run_id}")
        raise typer.Exit(130)


@app.command()
@handled
def resume(run_id: str):
    status = resume_run(Repository(home()), run_id)
    console.print(f"{status}: {run_id}")
    if status == "INTERRUPTED":
        console.print(f"Resume: torcharena resume {run_id}")
        raise typer.Exit(130)


@app.command()
@handled
def runs():
    runs_table(console, Repository(home()))


@app.command()
@handled
def inspect(run_id: str):
    console.print(json.dumps(inspect_data(Repository(home()), run_id), indent=2))


@app.command("compare")
@handled
def compare_runs(run_ids: Annotated[list[str], typer.Argument()]):
    """Compare persisted runs; incompatible datasets have no winner categories."""
    repository = Repository(home())
    race_table(console, compare([repository.get(run_id) for run_id in run_ids]))


@app.command()
@handled
def race(configs: Annotated[list[Path], typer.Argument()]):
    result = run_race(Repository(home()), [load_config(path) for path in configs])
    race_table(console, result)


@app.command()
@handled
def graveyard():
    runs_table(console, Repository(home()), graveyard=True)


@app.command()
@handled
def report(run_id: str):
    console.print(str(make_report(Repository(home()), run_id)))


@app.command()
@handled
def showcase(scenario: str):
    """Run race, crash-resume or graveyard; export genuine terminal and HTML evidence."""
    output = Path(os.environ.get("TORCHARENA_SHOWCASE_DIR", "artifacts/showcase"))
    run_showcase(scenario, home(), output)
    console.print(f"Artifacts: {output / scenario}")
