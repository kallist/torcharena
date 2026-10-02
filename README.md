# TorchArena

Train. Race. Break. Resume.

A lightweight, reproducible PyTorch training workbench where models can compete,
experiments survive interruptions, and failed runs leave useful evidence behind.

**Two models race. A run breaks and resumes. A deliberately failed run stays inspectable.**

[Model Race](artifacts/showcase/race.html) ·
[Crash → Resume](artifacts/showcase/crash-resume.html) ·
[Failure Graveyard](artifacts/showcase/graveyard.html)

Static HTML previews can be downloaded/opened locally; GitHub does not render HTML
as a live page. The terminal transcripts below are generated from real executions.
This is a small local workbench, not a distributed ML platform.

## Why TorchArena exists

Weights alone do not tell you how to continue an interrupted experiment. TorchArena
keeps optimizer/scheduler/scaler/RNG state, a deterministic batch cursor and an
inspectable SQLite history. Preflight catches broken batches before training;
guarded failures preserve the last healthy checkpoint. The playful shell wraps
ordinary, explainable PyTorch engineering.

## Installation

Python 3.11 or 3.12 is recommended for this V0.1; local evidence currently covers
Windows Python 3.12.14 / PyTorch 2.8.0+cpu. No GPU or dataset download is needed.

```bash
python -m venv .venv
# POSIX
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

The CPU index deliberately installs CPU PyTorch even on a host with an NVIDIA GPU.
CUDA/AMP paths exist but have **NOT TESTED** status in this delivery. NumPy is a dev
dependency for RNG tests; it is optional for the tensor-only runtime. A minimal
installation without NumPy may emit PyTorch's optional NumPy initialization warning.

## 30-second quick start after installation

```bash
torcharena doctor
torcharena validate examples/tiny_cnn.yaml
torcharena showcase race
torcharena showcase crash-resume
torcharena showcase graveyard
```

Each demo trains tiny 8x8 synthetic images with real Adam updates. Final numbers are
measured, not predefined. Showcase execution creates recorded TXT, HTML and JSON
under `artifacts/showcase/`. Runtime state normally lives in `.torcharena/`.

## Model Race

```bash
torcharena race examples/tiny_cnn.yaml examples/tiny_resnet.yaml
```

TinyCNN includes dropout; TinyResidualCNN uses a residual convolution. The Race
executes contestants sequentially with the same dataset/split/seed/batch size/epoch
budget/device. It refuses incompatible configurations. `compare` can inspect old
incompatible runs but marks them **NOT COMPARABLE** and produces no winner categories.

Measured CPU values from the checked-in `race.json` (seed 42, 64 train / 32 validation samples):

| Model | Validation accuracy | Validation loss | Fit time (s) | Parameters | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny_cnn | 100.00% | 0.000079 | 0.381 | 554 | 4 |
| tiny_resnet | 100.00% | 0.000003 | 0.398 | 702 | 4 |

Both contestants tie for highest accuracy. Categories come from measured values.

[Complete generated terminal transcript](artifacts/showcase/race.txt) ·
[Measured JSON](artifacts/showcase/race.json)

This easy synthetic task is a training/recovery fixture, not an accuracy benchmark.
Timing includes preflight/checkpoint I/O and varies with host load and contestant
order. Ties remain ties. There is no arbitrary overall winner.

## Crash → Resume

```bash
torcharena showcase crash-resume
# For your own run: Ctrl+C, then use its printed ID
torcharena resume <run-id>
```

The demonstration interrupts real training at a safe step, releases the original
Trainer/model/optimizer, constructs new objects, loads the persisted full state and
finishes. It also trains an uninterrupted baseline and checks metrics/parameters.

```text
RESTORED: model / optimizer / scheduler / scaler / RNG / batch cursor
RESUMED → COMPLETED · step 16
Baseline equivalence: PASS · max parameter error 0 · atol 1e-7
```

[Complete generated transcript](artifacts/showcase/crash-resume.txt) ·
[Recovery JSON](artifacts/showcase/crash-resume.json) ·
[Recovery contract](docs/RECOVERY.md)

SIGINT is deferred to a safe update/epoch boundary. A hard process termination
cannot be caught; recovery uses the last already-written snapshot and repeats
uncheckpointed work. Exact mid-epoch continuation is supported only for the owned
deterministic dataset. No arbitrary DataLoader or cross-hardware equivalence claim.

## Failure Graveyard

```bash
torcharena showcase graveyard
torcharena graveyard
torcharena inspect <run-id>
```

This is an **explicit deterministic fault-injection demo**, not a naturally occurring
model bug. A Python-only showcase hook puts NaN into the next real training batch;
TrainingGuard detects the resulting loss, records FAILED plus technical details,
and leaves the previous healthy checkpoint intact. Normal `train` accepts no fault
injection fields or flags.

Persisted run `run-0f55a5ffea9b482fa45aad993e074553`: **FAILED**, last durable step **2**, recoverable **YES**.

```text
Detected: non_finite_loss · Detected non_finite_loss; inspect data and learning rate
Last healthy checkpoint preserved; failure traceback persisted.
```

[Complete generated transcript](artifacts/showcase/graveyard.txt) ·
[Failure inspection JSON](artifacts/showcase/graveyard.json)

## Training preflight

`validate` and `train` share the same temporary-model check: config, both datasets,
batch/output shapes, class compatibility, loss finiteness, backward gradients,
optimizer viability, device and requested AMP. Preflight restores caller RNG and
determinism settings; it does not modify the training model.

The YAML schema is closed. Unknown model/dataset names, empty sample counts,
invalid learning rates, class mismatch, duplicate keys and unexpected fields fail
with actionable errors. Only two explicit model factories and one dataset factory
are registered in V0.1.

## CLI and reports

```bash
torcharena train examples/tiny_cnn.yaml
torcharena runs
torcharena inspect <run-id>
torcharena compare <run-a> <run-b>
torcharena report <run-id>
```

Use the full printed run ID. `report` creates a self-contained HTML battle report
inside the run directory: measured metrics/loss curve, model/dataset, config hash,
timing/parameters, environment/seed/Git metadata, recovery details and attempt history.
There is no web server or browser frontend. Rich tables remain understandable with
`NO_COLOR=1`; color is never the only status signal. `doctor` is read-only and does
not create the registry.

## Architecture

```text
Typer CLI + Rich presentation
          ↓
Config / explicit registries → isolated preflight
          ↓
Service + local ownership lock
          ↓
PyTorch Trainer ↔ full-state recovery
          ↓
SQLite runs / metrics / failures / attempt history / races
          ↓
Sequential comparison + static HTML evidence
```

[Architecture and ADRs](docs/ARCHITECTURE.md) · [Recovery design](docs/RECOVERY.md) ·
[Repository audit](docs/REPOSITORY_AUDIT.md) · [Separate review](docs/REVIEW.md)

## Experiment registry and reproducibility

Default: `.torcharena/torcharena.db` and `.torcharena/runs/<run-id>/`. Each run stores
its canonical config/seed/hash, Python/PyTorch/platform/device/CUDA/Git metadata,
status/progress/metrics and failure details. JSON/YAML/JSONL are inspectable exports;
checkpoints remain files, not database blobs. Set `TORCHARENA_HOME` to select another
local store and `TORCHARENA_SHOWCASE_DIR` to choose a demo export directory.

Resume uses the same logical run with explicit transition history. COMPLETED is
terminal. Per-run OS locks reject concurrent resume; SQLite transactions and a
5-second busy timeout cover ordinary local contention. After a crash between file
and database commits, the checkpoint is authoritative and metrics are reconciled.
This does not make the tool a distributed or power-loss-proof system.

Deterministic mode seeds Python, NumPy when installed, torch CPU and available CUDA,
and requests deterministic kernels. Some kernels/environments may reject this mode;
GPU determinism can cost performance. Resume rejects every config change and requires
the same device type and PyTorch version. [Detailed guarantees](docs/RECOVERY.md).

## Testing

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
```

Behavioral tests prove parameter updates, actual learning, fresh-object full-state
roundtrip, CPU uninterrupted/resumed equality at six interruption points, SIGINT,
hard process exit, healthy-checkpoint survival, failure persistence and preflight
isolation. Subprocess tests invoke the installed console script. Tiny CPU datasets
need no downloads or external provider calls. [Executed evidence](docs/VALIDATION.md)
separates local testing from unobserved Hosted CI.

## Docker and CI

```bash
docker build -t torcharena:local .
docker run --rm torcharena:local torcharena doctor
docker run --rm torcharena:local torcharena train /app/examples/tiny_cnn.yaml
```

The CPU image runs as an unprivileged user. Docker configuration is IMPLEMENTED;
local execution is NOT TESTED because the engine was unavailable. GitHub Actions
defines Python 3.11/3.12 CPU lint/tests/build/showcase jobs and a Docker smoke job.
Hosted CI is BLOCKED until a remote is configured; it has not passed.

## Security and limitations

- Safe YAML, explicit registries, parameterized SQL and validated managed paths.
- No API keys, user accounts, provider calls or arbitrary config code execution.
- Only load locally-created or otherwise trusted TorchArena checkpoints. Full-state
  loading uses ordinary PyTorch deserialization; it is **not sandboxed**. Listing/
  inspection may deserialize local snapshots to check recoverability.
- File flush/fsync/replace protects against failed temporary writes. Directory fsync,
  hardware/storage failure tolerance and network filesystem locks are not promised.
- Static type checking, CUDA/AMP execution, Python 3.11/Linux execution and Docker/
  Hosted runs are not established by the local Windows evidence. One StepLR resume
  boundary emits a documented ordering warning despite state-equivalence PASS.

## Roadmap

V0.1 prioritizes owned synthetic training, recovery and inspectable evidence.
Future work could evaluate a real dataset and define explicit sampler contracts.
Replay, CIFAR-10, remote artifacts, DDP, cloud services and web UI are not implemented.

## Reference / inspiration / license

Concepts were studied from [torchkeras](https://github.com/lyhue1991/torchkeras), an
Apache-2.0 Keras-style training template. Training loops, metrics, early stopping,
GPU, AMP, checkpointing and logging integrations are existing/common capabilities.
TorchArena's independent design centers on local experiment history, guarded failures
and versioned full-state continuation with behavioral evidence. No reference source,
assets or notebooks were copied. [Reference audit](docs/REFERENCE_AUDIT.md).

TorchArena original source: [MIT](LICENSE). No superiority, MLflow replacement,
production readiness or universal bitwise reproducibility claim is made.
