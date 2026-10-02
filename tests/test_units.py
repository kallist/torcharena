import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from pydantic import ValidationError

from torcharena.checkpoint import atomic_save, load_checkpoint
from torcharena.comparison import compare, race
from torcharena.config import RunConfig, load_config
from torcharena.data import DATASETS, epoch_order, make_dataset
from torcharena.locking import run_lock
from torcharena.models import MODELS, make_model
from torcharena.report import report
from torcharena.runtime import environment
from torcharena.safety import TrainingError, TrainingGuard
from torcharena.storage import checked_id
from torcharena.trainer import Trainer


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_learning_rate(value):
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"training": {"learning_rate": value}})


@pytest.mark.parametrize(
    "payload",
    [
        {"dataset": {"train_samples": 0}},
        {"dataset": {"val_samples": 0}},
        {"training": {"batch_size": 0}},
        {"model": {"name": "os.system"}},
        {"dataset": {"name": "external"}},
        {"training": {"inject_failure": True}},
        {"model": {"num_classes": 3}},
        {"training": {"epochs": True}},
    ],
)
def test_closed_config_rejects_invalid_inputs(payload):
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_hash_normalizes_yaml_order_and_defaults(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  num_classes: 2\n  name: tiny_cnn\n", encoding="utf-8")
    assert load_config(path).config_hash == config.config_hash
    assert (
        RunConfig.model_validate(json.loads(config.canonical_json())).config_hash
        == config.config_hash
    )


@pytest.mark.parametrize(
    "content", ["!!python/object/apply:os.system ['bad']", "[]", "", "model: {}\nmodel: {}"]
)
def test_unsafe_or_ambiguous_yaml_rejected(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(Exception):
        load_config(path)


def test_registries_and_learnable_dataset(config):
    assert set(MODELS) == {"tiny_cnn", "tiny_resnet"}
    assert set(DATASETS) == {"synthetic_image_classification"}
    for name in MODELS:
        c = config.model_copy(update={"model": config.model.model_copy(update={"name": name})})
        assert make_model(c)(torch.zeros(3, 1, 8, 8)).shape == (3, 2)
    train_x, train_y = make_dataset(config, "train")
    assert torch.equal(train_x, make_dataset(config, "train")[0])
    assert train_y.dtype == torch.long
    assert not torch.equal(train_x[:32], make_dataset(config, "val")[0])
    assert torch.equal(epoch_order(config, 1), epoch_order(config, 1))
    assert not torch.equal(epoch_order(config, 1), epoch_order(config, 2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_guard_loss(value):
    with pytest.raises(TrainingError, match="non_finite_loss"):
        TrainingGuard.finite(torch.tensor(value), "non_finite_loss")
    TrainingGuard.finite(torch.tensor(0.0), "loss")


def test_guard_gradients_and_parameters(config):
    model = make_model(config)
    with pytest.raises(TrainingError, match="No trainable"):
        TrainingGuard.gradients(model)
    model(torch.zeros(1, 1, 8, 8)).sum().backward()
    TrainingGuard.gradients(model)
    next(model.parameters()).grad.fill_(float("inf"))
    with pytest.raises(TrainingError, match="non_finite_gradient"):
        TrainingGuard.gradients(model)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(TrainingError, match="non_finite_parameter"):
        TrainingGuard.parameters(model)


def test_atomic_write_failure_preserves_previous(tmp_path, monkeypatch):
    import torcharena.checkpoint as module

    path = tmp_path / "last.pt"
    atomic_save({"healthy": 123}, path)
    before = path.read_bytes()

    def fail(state, stream):
        stream.write(b"partial corrupt snapshot")
        raise OSError("disk full")

    monkeypatch.setattr(module.torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_save({"healthy": 456}, path)
    assert path.read_bytes() == before
    assert torch.load(path, weights_only=True) == {"healthy": 123}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_replace_failure_preserves_previous(tmp_path, monkeypatch):
    import torcharena.checkpoint as module

    path = tmp_path / "last.pt"
    atomic_save({"healthy": True}, path)
    previous = path.read_bytes()

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        atomic_save({"healthy": False}, path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_schema_hash_and_identity_fail_closed(tmp_path, config):
    trainer = Trainer(config, "test-run", tmp_path)
    state = trainer.snapshot()
    path = tmp_path / "last.pt"
    atomic_save(state, path)
    assert load_checkpoint(path, config.config_hash)["schema_version"] == 1
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, "different")
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, config.config_hash, "other")
    state["schema_version"] = 999
    atomic_save(state, path)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path, config.config_hash)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        load_checkpoint(path, config.config_hash)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path, config.config_hash)


def test_state_machine_persistence_and_atomic_failure(repository, config):
    run_id = repository.create(config, environment(torch.device("cpu")))
    repository.transition(run_id, "VALIDATING")
    repository.transition(run_id, "RUNNING")
    with pytest.raises(ValueError, match="Invalid run transition"):
        repository.transition(run_id, "RESUMING")
    failure = {"condition": "non_finite_loss", "message": "NaN", "technical": "details"}
    repository.transition(run_id, "FAILED", failure=failure)
    assert repository.get(run_id)["status"] == "FAILED"
    assert repository.failures(run_id) == [failure]
    repository.transition(run_id, "RESUMING")
    repository.transition(run_id, "RUNNING")
    repository.transition(run_id, "COMPLETED")
    with pytest.raises(ValueError):
        repository.transition(run_id, "RUNNING")
    assert repository.get(run_id)["status"] == "COMPLETED"
    assert len(repository.history(run_id)) == 7


def test_transaction_rolls_back_status_when_failure_insert_fails(repository, config):
    run_id = repository.create(config, {})
    with repository.connection() as db:
        db.execute(
            "CREATE TRIGGER reject_failure BEFORE INSERT ON failures "
            "BEGIN SELECT RAISE(ABORT, 'test failure insert'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        repository.transition(run_id, "FAILED", failure={"condition": "error", "message": "bad"})
    assert repository.get(run_id)["status"] == "CREATED"
    assert repository.failures(run_id) == []


def test_concurrent_status_claim_is_single_winner(repository, config):
    run_id = repository.create(config, {})

    def claim():
        try:
            repository.transition(run_id, "VALIDATING")
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == [False, True]


def test_read_contention_and_unique_ids(repository, config):
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _: repository.create(config, {}), range(6)))
        statuses = list(pool.map(lambda i: repository.get(i)["status"], ids))
    assert len(set(ids)) == 6
    assert statuses == ["CREATED"] * 6


def test_exclusive_lock_rejects_second_owner(tmp_path):
    with run_lock(tmp_path / "owner.lock"):
        with pytest.raises(ValueError, match="live trainer"):
            with run_lock(tmp_path / "owner.lock"):
                pytest.fail("lock allowed second owner")
    with run_lock(tmp_path / "owner.lock"):
        pass


@pytest.mark.parametrize("run_id", ["../../outside", "run-1", "/etc/passwd", "run-" + "f" * 33])
def test_paths_reject_untrusted_ids(run_id):
    with pytest.raises(ValueError):
        checked_id(run_id)


def test_comparison_ties_and_mismatch(repository, config):
    ids = [repository.create(config, {}) for _ in range(2)]
    for run_id in ids:
        repository.transition(run_id, "VALIDATING")
        repository.transition(run_id, "RUNNING")
        repository.transition(
            run_id,
            "COMPLETED",
            summary={
                "final_metrics": {"val_accuracy": 0.75, "val_loss": 0.5},
                "parameters": 12,
                "duration_seconds": 1,
                "last_epoch": 4,
            },
        )
    records = [repository.get(run_id) for run_id in ids]
    result = compare(records)
    assert result["categories"]["Highest accuracy"] == ids
    mismatch = copy.deepcopy(records)
    mismatch[1]["config"]["experiment"]["seed"] += 1
    assert compare(mismatch)["categories"] == {}
    assert not compare(mismatch)["comparable"]
    other = RunConfig.model_validate(mismatch[1]["config"])
    with pytest.raises(ValueError, match="NOT COMPARABLE"):
        race(repository, [config, other])
    assert len(repository.list_runs()) == 2  # Rejected before any training.


def test_report_escapes_user_content(repository):
    config = RunConfig.model_validate({"experiment": {"name": '<script>alert("bad")</script>'}})
    run_id = repository.create(config, {})
    content = report(repository, run_id).read_text(encoding="utf-8")
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "http://" not in content and "https://" not in content
