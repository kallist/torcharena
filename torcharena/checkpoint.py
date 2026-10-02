"""Only load trusted locally-created snapshots; torch.load is not a sandbox."""

import os
import tempfile
from pathlib import Path

import torch

SCHEMA_VERSION = 1
REQUIRED = {
    "schema_version",
    "run_id",
    "config_hash",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "progress",
    "rng",
    "device_type",
    "runtime",
}


def atomic_save(state: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: Path, config_hash: str, run_id: str | None = None) -> dict:
    if not path.is_file():
        raise ValueError("Checkpoint missing; no recoverable snapshot exists")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Checkpoint corrupt/unreadable: {type(error).__name__}") from error
    if not isinstance(state, dict) or not REQUIRED.issubset(state):
        raise ValueError("Checkpoint missing required full-state fields")
    if state["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Incompatible checkpoint schema")
    if state["config_hash"] != config_hash:
        raise ValueError("Config hash mismatch; V0.1 rejects all config changes on resume")
    if run_id is not None and state["run_id"] != run_id:
        raise ValueError("Checkpoint run identity mismatch")
    mappings = ("model", "optimizer", "scheduler", "scaler", "progress", "rng", "runtime")
    if not all(isinstance(state[key], dict) for key in mappings):
        raise ValueError("Checkpoint full-state fields must be mappings")
    if not {"state", "param_groups"}.issubset(state["optimizer"]):
        raise ValueError("Checkpoint optimizer state is incomplete")
    progress_keys = {
        "epoch",
        "batch_cursor",
        "global_step",
        "loss_sum",
        "correct",
        "samples",
        "best_metric",
        "best_epoch",
        "bad_epochs",
        "history",
        "duration_seconds",
        "early_stopped",
    }
    if not progress_keys.issubset(state["progress"]):
        raise ValueError("Checkpoint progress state is incomplete")
    if not {"python", "torch", "numpy", "cuda"}.issubset(state["rng"]):
        raise ValueError("Checkpoint RNG state is incomplete")
    if "torch" not in state["runtime"]:
        raise ValueError("Checkpoint runtime metadata is incomplete")
    if not state["model"] or not all(
        isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in state["model"].values()
    ):
        raise ValueError("Checkpoint contains invalid model tensors")
    return state
