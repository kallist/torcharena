"""Local SQLite transactions and guarded state transitions."""

import contextlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRANSITIONS = {
    "CREATED": {"VALIDATING", "FAILED"},
    "VALIDATING": {"RUNNING", "FAILED", "INTERRUPTED"},
    "RUNNING": {"COMPLETED", "INTERRUPTED", "FAILED"},
    "INTERRUPTED": {"RESUMING"},
    "FAILED": {"RESUMING"},
    "RESUMING": {"RUNNING", "FAILED"},
    "COMPLETED": set(),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_id(run_id: str) -> str:
    if not re.fullmatch(r"run-[0-9a-f]{32}", run_id):
        raise ValueError("Invalid run ID; use the full ID printed by torcharena runs")
    return run_id


def confined(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes TorchArena home")
    return path


class Repository:
    def __init__(self, home: Path):
        self.home = home.resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.db = self.home / "torcharena.db"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, ended_at TEXT,
                    config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
                    environment_json TEXT NOT NULL, summary_json TEXT NOT NULL DEFAULT '{}',
                    last_epoch INTEGER NOT NULL DEFAULT 0, global_step INTEGER NOT NULL DEFAULT 0,
                    checkpoint_path TEXT, failure_type TEXT, failure_message TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL REFERENCES runs(run_id), epoch INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(run_id, epoch)
                );
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    source TEXT, target TEXT NOT NULL, at TEXT NOT NULL, reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS races (
                    race_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.db, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def run_dir(self, run_id: str) -> Path:
        return confined(self.home, self.home / "runs" / checked_id(run_id))

    def create(self, config, env: dict) -> str:
        run_id = "run-" + uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO runs(run_id,name,status,created_at,updated_at,"
                "config_json,config_hash,environment_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    config.experiment.name,
                    "CREATED",
                    now(),
                    now(),
                    config.canonical_json(),
                    config.config_hash,
                    json.dumps(env),
                ),
            )
            db.execute(
                "INSERT INTO transitions(run_id,target,at,reason) VALUES(?,?,?,?)",
                (run_id, "CREATED", now(), "new experiment"),
            )
        return run_id

    def get(self, run_id: str) -> dict:
        checked_id(run_id)
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        result = dict(row)
        for key in ("config_json", "environment_json", "summary_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def transition(
        self,
        run_id: str,
        target: str,
        reason: str = "",
        failure: dict | None = None,
        summary: dict | None = None,
    ):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("Run not found")
            source = row[0]
            if target not in TRANSITIONS[source]:
                raise ValueError(f"Invalid run transition: {source} -> {target}")
            db.execute(
                "UPDATE runs SET status=?,updated_at=?,ended_at=? WHERE run_id=?",
                (
                    target,
                    now(),
                    now() if target in {"COMPLETED", "FAILED", "INTERRUPTED"} else None,
                    run_id,
                ),
            )
            db.execute(
                "INSERT INTO transitions(run_id,source,target,at,reason) VALUES(?,?,?,?,?)",
                (run_id, source, target, now(), reason),
            )
            if failure is not None:
                db.execute(
                    "INSERT INTO failures(run_id,payload) VALUES(?,?)",
                    (run_id, json.dumps(failure)),
                )
                db.execute(
                    "UPDATE runs SET failure_type=?,failure_message=? WHERE run_id=?",
                    (failure["condition"], failure["message"], run_id),
                )
            if summary is not None:
                db.execute(
                    "UPDATE runs SET summary_json=? WHERE run_id=?", (json.dumps(summary), run_id)
                )

    def progress(self, run_id: str, epoch: int, step: int, checkpoint_path: str):
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET last_epoch=?,global_step=?,checkpoint_path=?,"
                "updated_at=? WHERE run_id=?",
                (epoch, step, checkpoint_path, now(), run_id),
            )

    def metric(self, run_id: str, payload: dict):
        with self.connection() as db:
            db.execute(
                "INSERT INTO metrics VALUES(?,?,?) ON CONFLICT(run_id,epoch) "
                "DO UPDATE SET payload=excluded.payload",
                (run_id, payload["epoch"], json.dumps(payload)),
            )

    def reconcile(self, run_id: str, history: list[dict]):
        # The snapshot is authoritative after a crash between file and DB commits.
        with self.connection() as db:
            db.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
            db.executemany(
                "INSERT INTO metrics VALUES(?,?,?)",
                [(run_id, p["epoch"], json.dumps(p)) for p in history],
            )

    def metrics(self, run_id: str) -> list[dict]:
        with self.connection() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM metrics WHERE run_id=? ORDER BY epoch", (run_id,)
                )
            ]

    def failures(self, run_id: str) -> list[dict]:
        with self.connection() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM failures WHERE run_id=? ORDER BY id", (run_id,)
                )
            ]

    def history(self, run_id: str) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT source,target,at,reason FROM transitions WHERE run_id=? ORDER BY id",
                    (run_id,),
                )
            ]

    def list_runs(self, graveyard: bool = False) -> list[dict]:
        with self.connection() as db:
            query = "SELECT run_id FROM runs"
            if graveyard:
                query += " WHERE status IN ('FAILED','INTERRUPTED')"
            ids = [row[0] for row in db.execute(query + " ORDER BY created_at")]
        return [self.get(run_id) for run_id in ids]

    def save_race(self, payload: dict) -> str:
        race_id = "race-" + uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO races VALUES(?,?,?)", (race_id, now(), json.dumps(payload)))
        return race_id
