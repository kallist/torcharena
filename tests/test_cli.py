import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = shutil.which("torcharena") or str(
    Path(sys.executable).parent / ("torcharena.exe" if os.name == "nt" else "torcharena")
)


def run_cli(tmp_path, *args, success=True):
    env = {
        **os.environ,
        "TORCHARENA_HOME": str(tmp_path / "home"),
        "TORCHARENA_SHOWCASE_DIR": str(tmp_path / "exports"),
        "NO_COLOR": "1",
        "PYTHONUTF8": "1",
        "COLUMNS": "160",
    }
    result = subprocess.run(
        [CLI, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_installed_cli_doctor_validate_train_inspect_report_compare(tmp_path):
    assert Path(CLI).exists(), "Install editable package before running tests"
    assert "READY" in run_cli(tmp_path, "doctor")
    assert not (tmp_path / "home").exists()  # doctor is read-only
    assert "READY TO TRAIN" in run_cli(tmp_path, "validate", "examples/tiny_cnn.yaml")
    output = run_cli(tmp_path, "train", "examples/tiny_cnn.yaml")
    run_id = re.search(r"run-[0-9a-f]{32}", output).group()
    assert "COMPLETED" in output
    assert run_id in run_cli(tmp_path, "runs")
    assert "config_hash" in run_cli(tmp_path, "inspect", run_id)
    report = run_cli(tmp_path, "report", run_id)
    assert "report.html" in report
    assert (tmp_path / "home/runs" / run_id / "report.html").is_file()
    assert "Highest accuracy" in run_cli(tmp_path, "compare", run_id, run_id)
    assert "FAILED" in run_cli(tmp_path, "resume", run_id, success=False)


def test_installed_cli_race_persists_real_runs(tmp_path):
    output = run_cli(tmp_path, "race", "examples/tiny_cnn.yaml", "examples/tiny_resnet.yaml")
    assert "Highest accuracy" in output
    assert "tiny_cnn" in output and "tiny_resnet" in output
    import sqlite3

    with sqlite3.connect(tmp_path / "home/torcharena.db") as db:
        assert db.execute("SELECT COUNT(*) FROM races").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM runs WHERE status='COMPLETED'").fetchone()[0] == 2


@pytest.mark.parametrize("scenario", ["race", "crash-resume", "graveyard"])
def test_installed_cli_showcases(tmp_path, scenario):
    output = run_cli(tmp_path, "showcase", scenario)
    assert (tmp_path / "exports" / f"{scenario}.txt").is_file()
    assert (tmp_path / "exports" / f"{scenario}.html").is_file()
    result = json.loads((tmp_path / "exports" / f"{scenario}.json").read_text(encoding="utf-8"))
    if scenario == "crash-resume":
        assert result["equivalent"] and result["optimizer_restored"]
        assert result["safe_step"] == 3
        assert "RESTORED" in output
    if scenario == "graveyard":
        assert result["status"] == "FAILED"
        assert "non_finite_loss" in run_cli(tmp_path, "graveyard")
        assert "demo injection" in run_cli(tmp_path, "inspect", result["run_id"])


@pytest.mark.parametrize(
    "args",
    [
        ("train", "missing.yaml"),
        ("inspect", "../outside"),
        ("report", "run-" + "f" * 32),
        ("showcase", "unknown"),
    ],
)
def test_installed_cli_actionable_errors(tmp_path, args):
    assert "FAILED" in run_cli(tmp_path, *args, success=False)


def test_hard_process_exit_then_fresh_cli_resume(tmp_path):
    from torcharena.checkpoint import load_checkpoint
    from torcharena.config import RunConfig
    from torcharena.storage import Repository

    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/hard_exit_worker.py"), str(tmp_path / "home")],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 17, result.stderr
    repository = Repository(tmp_path / "home")
    record = repository.list_runs()[0]
    assert record["status"] == "RUNNING"
    state = load_checkpoint(
        repository.run_dir(record["run_id"]) / "checkpoints/last.pt", RunConfig().config_hash
    )
    assert state["progress"]["global_step"] == 2
    assert "COMPLETED" in run_cli(tmp_path, "resume", record["run_id"])
    assert repository.get(record["run_id"])["global_step"] == 16
