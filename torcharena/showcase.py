"""Demonstrations execute real service paths; no prewritten metrics or transcripts."""

import gc
import json
from pathlib import Path

from rich.console import Console

from torcharena.checkpoint import load_checkpoint
from torcharena.comparison import race
from torcharena.config import RunConfig
from torcharena.report import inspect_data, report
from torcharena.service import resume, train
from torcharena.storage import Repository
from torcharena.ui import banner, race_table, runs_table


def interrupted_hook(trainer, phase):
    if phase == "after_step" and trainer.progress["global_step"] == 3:
        raise KeyboardInterrupt


def nonfinite_hook(trainer, phase):
    if phase == "before_step" and trainer.progress["global_step"] == 3:
        # The next actual forward/loss operation hits TrainingGuard; no simulated DB failure.
        trainer.train_data[0].fill_(float("nan"))


nonfinite_hook.injection = True


def demo_config(model: str = "tiny_cnn") -> RunConfig:
    return RunConfig.model_validate(
        {
            "experiment": {"name": f"showcase-{model}", "seed": 42},
            "model": {"name": model, "num_classes": 2},
        }
    )


def showcase(scenario: str, home: Path, output: Path) -> dict:
    if scenario not in {"race", "crash-resume", "graveyard"}:
        raise ValueError("Showcase must be race, crash-resume or graveyard")
    output.mkdir(parents=True, exist_ok=True)
    repository = Repository(home)
    console = Console(record=True, width=110, markup=False, highlight=False)
    result = {}
    if scenario == "race":
        result = race(repository, [demo_config(), demo_config("tiny_resnet")])
        race_table(console, result)
    elif scenario == "crash-resume":
        config = demo_config()
        baseline_id = train(repository, config)
        run_id = train(repository, config, demo_hook=interrupted_hook)
        interrupted = repository.get(run_id)
        if interrupted["status"] != "INTERRUPTED":
            raise RuntimeError("Demo did not produce an interrupted run")
        banner(
            console,
            "Crash → Resume",
            f"INTERRUPTED: {run_id}\nSafe step: "
            f"{interrupted['global_step']}\nCheckpoint persisted; original trainer released.",
        )
        gc.collect()
        status = resume(repository, run_id)  # Builds a new Trainer/model/optimizer.
        baseline = load_checkpoint(
            repository.run_dir(baseline_id) / "checkpoints/last.pt", config.config_hash, baseline_id
        )
        recovered = load_checkpoint(
            repository.run_dir(run_id) / "checkpoints/last.pt", config.config_hash, run_id
        )
        max_error = max(
            (baseline["model"][key] - recovered["model"][key]).abs().max().item()
            for key in baseline["model"]
        )
        equivalent = (
            max_error <= 1e-7
            and baseline["progress"]["history"] == recovered["progress"]["history"]
            and baseline["progress"]["global_step"] == recovered["progress"]["global_step"]
        )
        if not equivalent or status != "COMPLETED":
            raise RuntimeError("Resume equivalence failed")
        result = {
            "run_id": run_id,
            "baseline_id": baseline_id,
            "status": status,
            "safe_step": interrupted["global_step"],
            "max_parameter_error": max_error,
            "tolerance": 1e-7,
            "equivalent": equivalent,
            "optimizer_restored": bool(recovered["optimizer"]["state"]),
        }
        console.print("RESTORED: model / optimizer / scheduler / scaler / RNG / batch cursor")
        console.print(f"RESUMED → {status} · step {recovered['progress']['global_step']}")
        console.print(f"Baseline equivalence: PASS · max parameter error {max_error:g} · atol 1e-7")
    else:
        try:
            train(repository, demo_config(), demo_hook=nonfinite_hook)
        except Exception as error:
            from torcharena.safety import TrainingError

            if not isinstance(error, TrainingError) or error.condition != "non_finite_loss":
                raise
        failed = repository.list_runs(graveyard=True)[-1]
        if failed["status"] != "FAILED" or failed["failure_type"] != "non_finite_loss":
            raise RuntimeError("Demo failure was not persisted")
        result = inspect_data(repository, failed["run_id"])
        banner(console, "Failure Graveyard", "Explicit deterministic demo-only NaN injection")
        runs_table(console, repository, graveyard=True)
        console.print(f"Detected: {failed['failure_type']} · {failed['failure_message']}")
        console.print("Last healthy checkpoint preserved; failure traceback persisted.")
    run_ids = [row["run_id"] for row in result.get("rows", [])]
    if "run_id" in result:
        run_ids.append(result["run_id"])
    for run_id in run_ids:
        report_path = report(repository, run_id)
        # Keep generated report previews shareable; runtime binaries/DB remain ignored.
        (output / f"{scenario}-{run_id}.html").write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    console.save_text(str(output / f"{scenario}.txt"), clear=False)
    console.save_html(str(output / f"{scenario}.html"), clear=False)
    (output / f"{scenario}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
