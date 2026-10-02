"""Composition of the tensor engine with local persistence and ownership locks."""

import json
import traceback

import torch
import yaml

from torcharena.config import RunConfig
from torcharena.locking import run_lock
from torcharena.runtime import environment
from torcharena.storage import Repository, confined, now
from torcharena.trainer import Trainer


def train(repository: Repository, config: RunConfig, *, demo_hook=None) -> str:
    # Record configuration/device failures too, before creating the tensor engine.
    env = environment(torch.device("cpu"))
    run_id = repository.create(config, env)
    directory = repository.run_dir(run_id)
    with run_lock(confined(repository.home, directory / "owner.lock")):
        try:
            (directory / "config.yaml").write_text(
                yaml.safe_dump(config.model_dump()), encoding="utf-8"
            )
            trainer = Trainer(config, run_id, directory, repository)
            env = environment(trainer.device)
            with repository.connection() as db:
                db.execute(
                    "UPDATE runs SET environment_json=? WHERE run_id=?", (json.dumps(env), run_id)
                )
            (directory / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        except Exception as error:
            failure = {
                "condition": "preflight_failure",
                "message": str(error),
                "time": now(),
                "traceback": traceback.format_exc().replace(
                    str(repository.home.parent), "<workspace>"
                ),
            }
            repository.transition(run_id, "FAILED", failure=failure)
            try:
                (directory / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
            except OSError as artifact_error:
                error.add_note(f"Startup failure export also failed: {artifact_error}")
            raise
        trainer.fit(demo_hook=demo_hook)
    return run_id


def resume(repository: Repository, run_id: str) -> str:
    directory = repository.run_dir(run_id)
    with run_lock(confined(repository.home, directory / "owner.lock")):
        record = repository.get(run_id)
        if record["status"] not in {"INTERRUPTED", "FAILED", "RUNNING", "RESUMING"}:
            raise ValueError(f"Cannot resume run in status {record['status']}")
        config = RunConfig.model_validate(record["config"])
        trainer = Trainer(config, run_id, directory, repository)
        trainer.restore()  # Validate before changing status. Bad snapshots fail closed.
        if record["status"] in {"RUNNING", "RESUMING"}:
            # Lock acquisition proves no live owner; a crashed resume is first marked failed.
            target = "INTERRUPTED" if record["status"] == "RUNNING" else "FAILED"
            repository.transition(
                run_id, target, "abandoned process recovered under exclusive lock"
            )
        repository.transition(run_id, "RESUMING", "loading local trusted last.pt")
        return trainer.fit(resume=True)
