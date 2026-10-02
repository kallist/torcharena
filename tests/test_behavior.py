import gc
import random
import signal

import pytest
import torch

from torcharena.checkpoint import load_checkpoint
from torcharena.config import RunConfig
from torcharena.models import make_model
from torcharena.runtime import seed_all
from torcharena.safety import TrainingError, preflight
from torcharena.service import resume, train
from torcharena.showcase import interrupted_hook, nonfinite_hook
from torcharena.trainer import Trainer


def test_python_numpy_torch_rng_roundtrip():
    import numpy as np

    from torcharena.runtime import capture_rng, restore_rng

    seed_all(42, True)
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(4))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_parameters_update_and_model_learns(tmp_path, config):
    trainer = Trainer(config, "test", tmp_path)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    initial = trainer.evaluate()
    assert trainer.fit() == "COMPLETED"
    final = trainer.evaluate()
    assert any(not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())
    assert final["val_loss"] < initial["val_loss"] * 0.5
    assert final["val_accuracy"] >= 0.9
    assert trainer.progress["global_step"] == 16


def test_checkpoint_roundtrip_with_new_objects(tmp_path, config):
    original = Trainer(config, "test", tmp_path)
    assert original.fit(demo_hook=interrupted_hook) == "INTERRUPTED"
    snapshot = original.snapshot()
    del original
    gc.collect()
    fresh = Trainer(config, "test", tmp_path)
    fresh.restore()
    current = fresh.snapshot()
    for key in ("model", "optimizer", "scheduler", "scaler", "progress"):
        assert_nested_equal(snapshot[key], current[key])
    assert bool(fresh.optimizer.state_dict()["state"])
    assert torch.equal(snapshot["rng"]["torch"], current["rng"]["torch"])
    assert snapshot["rng"]["python"] == current["rng"]["python"]


@pytest.mark.parametrize("step", [1, 3, 4, 5, 15, 16])
def test_uninterrupted_vs_fresh_trainer_mid_epoch_resume(tmp_path, config, step):
    baseline = Trainer(config, "baseline", tmp_path / "baseline")
    baseline.fit()
    expected = baseline.snapshot()

    def stop(trainer, phase):
        if phase == "after_step" and trainer.progress["global_step"] == step:
            raise KeyboardInterrupt

    partial = Trainer(config, "resumed", tmp_path / "resumed")
    assert partial.fit(demo_hook=stop) == "INTERRUPTED"
    assert partial.progress["global_step"] == step
    del partial
    gc.collect()
    fresh = Trainer(config, "resumed", tmp_path / "resumed")
    fresh.restore()
    assert fresh.progress["global_step"] == step
    assert fresh.optimizer.state_dict()["state"]
    assert fresh.fit(resume=True) == "COMPLETED"
    actual = fresh.snapshot()
    assert actual["progress"]["global_step"] == expected["progress"]["global_step"] == 16
    assert actual["progress"]["epoch"] == expected["progress"]["epoch"] == 4
    assert actual["progress"]["history"] == expected["progress"]["history"]
    for key in ("model", "optimizer", "scheduler", "scaler"):
        assert_nested_equal(expected[key], actual[key])


def test_preflight_preserves_training_model_rng_and_runtime(config):
    seed_all(42, True)
    model = make_model(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    assert preflight(config)["Optimizer step"] == "PASS"
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    assert_nested_equal(before, model.state_dict())
    after_model = make_model(config)
    torch.set_rng_state(torch_rng)
    control_model = make_model(config)
    assert_nested_equal(after_model.state_dict(), control_model.state_dict())


def test_interrupted_persisted_run_resumes(repository, config):
    run_id = train(repository, config, demo_hook=interrupted_hook)
    assert repository.get(run_id)["status"] == "INTERRUPTED"
    assert (repository.run_dir(run_id) / "checkpoints/last.pt").is_file()
    assert resume(repository, run_id) == "COMPLETED"
    assert repository.get(run_id)["global_step"] == 16
    assert [event["target"] for event in repository.history(run_id)][-3:] == [
        "RESUMING",
        "RUNNING",
        "COMPLETED",
    ]
    with pytest.raises(ValueError, match="COMPLETED"):
        resume(repository, run_id)


def test_nonfinite_failure_preserves_healthy_checkpoint(repository, config):
    with pytest.raises(TrainingError, match="non_finite_loss"):
        train(repository, config, demo_hook=nonfinite_hook)
    failed = repository.list_runs(graveyard=True)[0]
    assert failed["status"] == "FAILED"
    assert failed["failure_type"] == "non_finite_loss"
    failure = repository.failures(failed["run_id"])[0]
    assert failure["confirmed_cause"] == "demo injection"
    assert "traceback" in failure
    path = repository.run_dir(failed["run_id"]) / "checkpoints/last.pt"
    state = load_checkpoint(path, config.config_hash)
    assert state["progress"]["global_step"] == 2  # failure at step 3, last durable step 2
    assert all(torch.isfinite(p).all() for p in state["model"].values())


def test_sigint_handler_defers_until_safe_step(repository, config):
    previous = signal.getsignal(signal.SIGINT)

    def interrupt_during_update(trainer, phase):
        if phase == "before_step" and trainer.progress["global_step"] == 1:
            signal.raise_signal(signal.SIGINT)

    run_id = train(repository, config, demo_hook=interrupt_during_update)
    assert signal.getsignal(signal.SIGINT) == previous
    assert repository.get(run_id)["status"] == "INTERRUPTED"
    assert resume(repository, run_id) == "COMPLETED"


def test_empty_batch_preflight_records_failure(repository, config, monkeypatch):
    import torcharena.safety as module

    monkeypatch.setattr(
        module,
        "make_dataset",
        lambda *_: (torch.empty(0, 1, 8, 8), torch.empty(0, dtype=torch.long)),
    )
    with pytest.raises(TrainingError, match="empty"):
        train(repository, config)
    failed = repository.list_runs(graveyard=True)[0]
    assert failed["status"] == "FAILED"
    assert not (repository.run_dir(failed["run_id"]) / "checkpoints/last.pt").exists()


def test_abandoned_running_run_and_metric_reconciliation(repository, config):
    run_id = train(repository, config, demo_hook=interrupted_hook)
    with repository.connection() as db:
        db.execute("UPDATE runs SET status='RUNNING' WHERE run_id=?", (run_id,))
    repository.metric(run_id, {"epoch": 99, "fake": "ahead of authoritative checkpoint"})
    assert resume(repository, run_id) == "COMPLETED"
    assert [m["epoch"] for m in repository.metrics(run_id)] == [1, 2, 3, 4]


def test_changed_config_rejects_resume_before_status_mutation(repository, config):
    import json

    run_id = train(repository, config, demo_hook=interrupted_hook)
    altered = config.model_dump()
    altered["training"]["learning_rate"] = 0.1
    with repository.connection() as db:
        db.execute("UPDATE runs SET config_json=? WHERE run_id=?", (json.dumps(altered), run_id))
    with pytest.raises(ValueError, match="hash mismatch"):
        resume(repository, run_id)
    assert repository.get(run_id)["status"] == "INTERRUPTED"
    from torcharena.report import recoverability

    assert not recoverability(repository, run_id)[0]


def test_disk_failure_still_records_failed_status(repository, config, monkeypatch):
    from pathlib import Path

    original_write = Path.write_text

    def fail_export(path, *args, **kwargs):
        if path.name == "failure.json":
            raise OSError("demo full disk during failure artifact export")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_export)
    with pytest.raises(TrainingError, match="non_finite_loss") as captured:
        train(repository, config, demo_hook=nonfinite_hook)
    record = repository.list_runs(graveyard=True)[0]
    assert record["status"] == "FAILED"
    assert repository.failures(record["run_id"])
    assert any("artifact export also failed" in note for note in captured.value.__notes__)


def test_race_cancellation_does_not_launch_next_contestant(repository, config, monkeypatch):
    import torcharena.comparison as module

    monkeypatch.setattr(
        module, "train", lambda repo, cfg: train(repo, cfg, demo_hook=interrupted_hook)
    )
    with pytest.raises(TrainingError, match="Race stopped"):
        module.race(repository, [config, config])
    assert len(repository.list_runs()) == 1
    assert repository.list_runs()[0]["status"] == "INTERRUPTED"
    with repository.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM races").fetchone()[0] == 1


def test_early_stopping_persisted_and_restored(tmp_path):
    config = RunConfig.model_validate(
        {
            "training": {"epochs": 10, "learning_rate": 0.02},
            "early_stopping": {
                "enabled": True,
                "metric": "val_accuracy",
                "mode": "max",
                "patience": 1,
            },
        }
    )
    trainer = Trainer(config, "test", tmp_path)
    trainer.fit()
    assert trainer.progress["early_stopped"]
    assert trainer.progress["epoch"] < 10
    fresh = Trainer(config, "test", tmp_path)
    fresh.restore()
    assert fresh.progress["bad_epochs"] == 1
    assert fresh.progress["best_epoch"] == trainer.progress["best_epoch"]


def test_corrupt_model_state_resume_is_actionable_and_keeps_status(repository, config):
    from torcharena.checkpoint import atomic_save

    run_id = train(repository, config, demo_hook=interrupted_hook)
    path = repository.run_dir(run_id) / "checkpoints/last.pt"
    state = load_checkpoint(path, config.config_hash)
    key = next(iter(state["model"]))
    state["model"][key] = torch.zeros(1)  # Valid serialization, wrong architecture tensor shape.
    atomic_save(state, path)
    with pytest.raises(ValueError, match="Incompatible/corrupt"):
        resume(repository, run_id)
    assert repository.get(run_id)["status"] == "INTERRUPTED"
