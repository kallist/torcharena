"""Tensor training engine. No SQLite or Rich imports; storage is an optional adapter."""

import contextlib
import copy
import json
import signal
import threading
import time
import traceback
from pathlib import Path

import torch

from torcharena.checkpoint import SCHEMA_VERSION, atomic_save, load_checkpoint
from torcharena.config import RunConfig
from torcharena.data import epoch_order, make_dataset
from torcharena.models import make_model
from torcharena.runtime import capture_rng, device_for, restore_rng, seed_all
from torcharena.safety import TrainingError, TrainingGuard, preflight


class Trainer:
    def __init__(self, config: RunConfig, run_id: str, directory: Path, repository=None):
        self.config, self.run_id, self.directory = config, run_id, directory.resolve()
        self.repository = repository
        seed_all(config.experiment.seed, config.training.deterministic)
        self.device = device_for(config)
        self.model = make_model(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=config.training.amp)
        self.train_data = make_dataset(config, "train")
        self.val_data = make_dataset(config, "val")
        self.progress = {
            "epoch": 0,
            "batch_cursor": 0,
            "global_step": 0,
            "loss_sum": 0.0,
            "correct": 0,
            "samples": 0,
            "best_metric": None,
            "best_epoch": None,
            "bad_epochs": 0,
            "history": [],
            "duration_seconds": 0.0,
            "early_stopped": False,
        }
        self.interrupt_requested = False
        self.safe = True
        self.phase = "CREATED"
        self._clock = None
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self):
        path = self.directory / "checkpoints" / "last.pt"
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError("Checkpoint path escapes the run directory")
        return path

    def snapshot(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_hash": self.config.config_hash,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "progress": copy.deepcopy(self.progress),
            "rng": capture_rng(),
            "device_type": self.device.type,
            "runtime": {"torch": str(torch.__version__)},
        }

    def save(self, best: bool = False):
        if not self.safe:
            raise RuntimeError("Refusing to checkpoint an incomplete optimizer/epoch unit")
        self._account_time()
        state = self.snapshot()
        # Retained safe snapshots plus an atomically replaced authoritative last.pt.
        step = self.progress["global_step"]
        retained = self.checkpoint_path.parent / f"step-{step:012d}.pt"
        atomic_save(state, retained)
        atomic_save(state, self.checkpoint_path)
        if best and self.config.checkpoint.save_best:
            atomic_save(state, self.checkpoint_path.parent / "best.pt")
        snapshots = sorted(self.checkpoint_path.parent.glob("step-*.pt"))
        for old in snapshots[: -self.config.checkpoint.keep_last]:
            old.unlink()
        if self.repository:
            relative = self.checkpoint_path.relative_to(self.repository.home).as_posix()
            self.repository.progress(self.run_id, self.progress["epoch"], step, relative)

    def restore(self):
        state = load_checkpoint(self.checkpoint_path, self.config.config_hash, self.run_id)
        if state["device_type"] != self.device.type:
            raise ValueError("Resume device type changed; restore on the original device type")
        if state["runtime"]["torch"] != str(torch.__version__):
            raise ValueError("Resume requires the same PyTorch version")
        try:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.progress = state["progress"]
            self._validate_progress()
            TrainingGuard.parameters(self.model)
            restore_rng(state["rng"])
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Incompatible/corrupt checkpoint state: {error}") from error
        self.safe = True
        if self.repository:
            self.repository.reconcile(self.run_id, self.progress["history"])
            self.repository.progress(
                self.run_id,
                self.progress["epoch"],
                self.progress["global_step"],
                self.checkpoint_path.relative_to(self.repository.home).as_posix(),
            )

    def _validate_progress(self):
        p = self.progress
        n = len(self.train_data[0])
        batches = (n + self.config.training.batch_size - 1) // self.config.training.batch_size
        if not (
            0 <= p["epoch"] <= self.config.training.epochs
            and 0 <= p["batch_cursor"] <= batches
            and p["global_step"] == p["epoch"] * batches + p["batch_cursor"]
            and p["samples"] == min(p["batch_cursor"] * self.config.training.batch_size, n)
            and len(p["history"]) == p["epoch"]
        ):
            raise ValueError("Checkpoint resume cursor is inconsistent")

    @contextlib.contextmanager
    def deferred_interrupt(self):
        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "interrupt_requested", True))
        try:
            yield
        finally:
            if previous is not None:
                signal.signal(signal.SIGINT, previous)

    def evaluate(self) -> dict:
        self.model.eval()
        images, targets = self.val_data
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(images), self.config.training.batch_size):
                x = images[start : start + self.config.training.batch_size].to(self.device)
                y = targets[start : start + self.config.training.batch_size].to(self.device)
                with torch.autocast(self.device.type, enabled=self.config.training.amp):
                    output = self.model(x)
                    loss = torch.nn.functional.cross_entropy(output, y)
                TrainingGuard.finite(loss, "non_finite_validation_loss")
                total_loss += loss.item() * len(y)
                correct += (output.argmax(1) == y).sum().item()
        return {"val_loss": total_loss / len(images), "val_accuracy": correct / len(images)}

    def _finish_epoch(self):
        self.safe = False
        p = self.progress
        metric = {
            "epoch": p["epoch"] + 1,
            "global_step": p["global_step"],
            "train_loss": p["loss_sum"] / p["samples"],
            "train_accuracy": p["correct"] / p["samples"],
            **self.evaluate(),
        }
        early = self.config.early_stopping
        value = metric[early.metric]
        improved = p["best_metric"] is None or (
            value < p["best_metric"] if early.mode == "min" else value > p["best_metric"]
        )
        if improved:
            p["best_metric"], p["best_epoch"], p["bad_epochs"] = value, metric["epoch"], 0
        else:
            p["bad_epochs"] += 1
        p["history"].append(metric)
        p["epoch"] += 1
        p["batch_cursor"], p["loss_sum"], p["correct"], p["samples"] = 0, 0.0, 0, 0
        p["early_stopped"] = early.enabled and p["bad_epochs"] >= early.patience
        self.scheduler.step()
        self.safe = True
        self.save(best=improved)
        if self.repository:
            self.repository.metric(self.run_id, metric)

    def fit(self, *, resume: bool = False, demo_hook=None) -> str:
        """demo_hook is a Python-only test/showcase hook, never exposed via YAML or train CLI."""
        self._clock = time.perf_counter()
        try:
            with self.deferred_interrupt():
                if not resume:
                    self.phase = "VALIDATING"
                    if self.repository:
                        self.repository.transition(self.run_id, "VALIDATING")
                    preflight(self.config)
                    self.phase = "RUNNING"
                    if self.repository:
                        self.repository.transition(self.run_id, "RUNNING")
                    self.save()
                else:
                    self.phase = "RUNNING"
                    if self.repository:
                        self.repository.transition(self.run_id, "RUNNING", "checkpoint restored")
                while (
                    self.progress["epoch"] < self.config.training.epochs
                    and not self.progress["early_stopped"]
                ):
                    p = self.progress
                    self.model.train()
                    order = epoch_order(self.config, p["epoch"])
                    batches = order.split(self.config.training.batch_size)
                    if not batches:
                        raise TrainingError("empty_dataloader", "No training batches")
                    for cursor in range(p["batch_cursor"], len(batches)):
                        if self.interrupt_requested:
                            raise KeyboardInterrupt
                        if demo_hook:
                            demo_hook(self, "before_step")
                        ids = batches[cursor]
                        images, targets = self.train_data
                        x, y = images[ids].to(self.device), targets[ids].to(self.device)
                        self.safe = False
                        self.optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(self.device.type, enabled=self.config.training.amp):
                            output = self.model(x)
                            loss = torch.nn.functional.cross_entropy(output, y)
                        TrainingGuard.finite(loss, "non_finite_loss")
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        TrainingGuard.gradients(self.model)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        TrainingGuard.parameters(self.model)
                        p["global_step"] += 1
                        p["batch_cursor"] = cursor + 1
                        p["loss_sum"] += loss.item() * len(y)
                        p["correct"] += (output.argmax(1) == y).sum().item()
                        p["samples"] += len(y)
                        self.safe = True
                        if p["global_step"] % self.config.checkpoint.every_n_steps == 0:
                            self.save()
                        if demo_hook:
                            demo_hook(self, "after_step")
                        if self.interrupt_requested:
                            raise KeyboardInterrupt
                    self._finish_epoch()
                self.save()
                self._exports()
                if self.repository:
                    self.repository.transition(self.run_id, "COMPLETED", summary=self.summary())
                return "COMPLETED"
        except KeyboardInterrupt:
            # Unexpected KeyboardInterrupt inside mutation retains last.pt; SIGINT is deferred.
            if self.safe:
                self.save()
            if self.repository:
                self.repository.transition(
                    self.run_id, "INTERRUPTED", "user_interrupt", summary=self.summary()
                )
            self._exports()
            return "INTERRUPTED"
        except Exception as error:
            condition = getattr(error, "condition", None)
            if condition is None:
                condition = (
                    "cuda_oom"
                    if isinstance(error, torch.cuda.OutOfMemoryError)
                    else "preflight_failure"
                    if self.phase == "VALIDATING"
                    else type(error).__name__
                )
            failure = {
                "condition": condition,
                "message": str(error),
                "time": environment_time(),
                "epoch": self.progress["epoch"],
                "step": self.progress["global_step"],
                "traceback": traceback.format_exc().replace(str(Path.cwd()), "<workspace>"),
                "detected_condition": condition,
                "confirmed_cause": "demo injection"
                if getattr(demo_hook, "injection", False)
                else None,
                "checkpoint_available": self.checkpoint_path.is_file(),
            }
            self._account_time()
            if self.repository:
                try:
                    self.repository.transition(
                        self.run_id, "FAILED", str(error), failure=failure, summary=self.summary()
                    )
                except Exception as persistence_error:
                    error.add_note(f"Failure DB persistence also failed: {persistence_error}")
            try:
                (self.directory / "failure.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                self._exports()
            except OSError as artifact_error:
                error.add_note(f"Failure artifact export also failed: {artifact_error}")
            raise
        finally:
            self._clock = None

    def _account_time(self):
        if self._clock is not None:
            current = time.perf_counter()
            self.progress["duration_seconds"] += current - self._clock
            self._clock = current

    def _exports(self):
        # SQLite is the canonical metric store. This human-readable export can be regenerated.
        history = (
            self.repository.metrics(self.run_id) if self.repository else self.progress["history"]
        )
        (self.directory / "metrics.jsonl").write_text(
            "".join(json.dumps(metric) + "\n" for metric in history), encoding="utf-8"
        )

    def summary(self) -> dict:
        return {
            "best_metric_name": self.config.early_stopping.metric,
            "best_metric": self.progress["best_metric"],
            "best_epoch": self.progress["best_epoch"],
            "last_epoch": self.progress["epoch"],
            "global_step": self.progress["global_step"],
            "duration_seconds": self.progress["duration_seconds"],
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "final_metrics": self.progress["history"][-1] if self.progress["history"] else {},
        }


def environment_time():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
