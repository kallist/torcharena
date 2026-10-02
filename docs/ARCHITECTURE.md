# Architecture and decisions

`cli.py` handles command arguments, bounded error presentation and Rich output.
`service.py` owns the SQLite/trainer composition and per-run OS lock. `trainer.py`
contains tensor work with an optional storage adapter and no Rich/SQLite imports;
it can train independently in behavioral tests.

`config.py` uses frozen, strict Pydantic models and a safe YAML loader that rejects
duplicate keys. `data.py` and `models.py` use explicit registries. No config code
execution, dynamic imports or shell commands are available. The only built-in
dataset is a learnable 8x8 synthetic classification task; both CNNs use genuine
trainable convolutions and distinct architectures.

`safety.py` performs isolated temporary-model preflight and checks finite loss,
gradients and post-update parameters. `runtime.py` seeds RNGs and captures metadata.
`checkpoint.py` serializes full state; `locking.py` prevents concurrent ownership.
`storage.py` holds parameterized SQLite operations and explicit transitions.
`comparison.py` checks comparability and persists race results. `report.py` escapes
user content and generates HTML/SVG without external scripts/CDNs. `showcase.py`
injects controlled demo faults via Python-only hooks, never YAML fields.

## ADR: config-first

Closed schemas make experiments inspectable and resume compatibility unambiguous.
V0.1 rejects every config change on resume. No generic plugin system or arbitrary
custom DataLoader support is exposed. Public extensibility can be designed later.

## ADR: local SQLite

Runs, metrics, failures, transition history and races are sufficient for a local
single-trainer tool. Foreign keys, 5-second busy timeout, short transactions and
`BEGIN IMMEDIATE` transition checks cover ordinary accidental contention. WAL is
not required for this small workload. Failure + FAILED status are one transaction.
Filesystem snapshots and SQLite cannot share a transaction; see RECOVERY.md.

## ADR: sequential Race

Resource contention would confound the small demo and require supervision. Execute
models in order with identical dataset/split/seed/batch size/epoch budget/device/AMP/
determinism/early-stopping policy. Learning rate is an explicit model configuration,
not a hidden tuning result. Compare final validation accuracy/loss, measured fit time
and parameter count separately. Ties remain ties. Timing is descriptive, includes
preflight/checkpoint I/O and is not a benchmark (order, warmup and host load matter).
No arbitrary overall champion or serialized-size claim is generated.

## ADR: atomic full-state checkpoints

Training continuation needs optimizer moments, scheduler/scaler, RNG and cursor as
well as weights. A temporary write + flush + fsync + replace protects last.pt from
partial serialization. Keep a bounded number of step snapshots and optional best.pt.
Recovery accepts only the authoritative last.pt. See RECOVERY.md for limitations.

## ADR: deterministic synthetic CI

No network dataset downloads, GPU, account or API key is needed for test execution.
CPU PyTorch installation still requires a package index. The synthetic data is
deliberately learnable: tests verify parameters move, loss falls and accuracy rises.
This proves training/recovery behavior, not real-world model quality.

## Status and ownership

CREATED → VALIDATING → RUNNING → COMPLETED / INTERRUPTED / FAILED.
FAILED or INTERRUPTED → RESUMING → RUNNING. COMPLETED is terminal.
Same logical run, with every transition recording an explicit attempt history.
Local nonblocking OS locks reject simultaneous train/resume ownership; locks are
released by process exit. After lock acquisition, a RUNNING record may represent
an abandoned process and is explicitly transitioned through INTERRUPTED. An abandoned
RESUMING record goes through FAILED. This is a local filesystem guarantee, not a
distributed lock and not support for network filesystems.

The terminal is the sole product UI; HTML is a static exported report. There is no
web service, authentication, remote artifact backend or distributed/DDP training.
These are intentionally outside V0.1.
