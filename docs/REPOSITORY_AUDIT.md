# Initial repository audit

Audit date: 2026-10-03. The supplied workspace was an empty directory, not a Git
repository. No existing source, AGENTS.md, README, manifests, tests, CI, Docker
configuration, branch, HEAD, default branch or remote existed. No unrelated work
needed relocation. Initialize a local main baseline, then work on
feat/torcharena-v0.1. Do not invent a remote or create a hosted repository.

System Python launcher: Python 3.10. Bundled executable: Python 3.12.14, selected
for a workspace-local virtual environment and pip. No uv executable was found.
PyTorch was not initially installed in that runtime. NVIDIA driver detects an RTX
3060; that does not establish PyTorch CUDA support. CPU-only validation is planned.
Docker client 29.5.3 is installed; Docker Desktop Linux engine cannot be reached.
Git 2.40.1 and GitHub CLI 2.97.0 are installed; gh authentication succeeds.
Without a supplied remote, push, Draft PR and Hosted CI are blocked.

## Implementation plan

- Config: Pydantic strict models, safe YAML, explicit model/dataset registries.
- Engine: PyTorch/Adam, weighted classification metrics, deterministic epoch order.
- Safety: isolated preflight, finite loss/gradient/parameter guards, deferred SIGINT.
- Recovery: versioned atomic full-state snapshots, batch cursor and partial epoch
  totals, RNG/optimizer/scheduler/scaler restoration, exclusive local run lock.
- Storage: SQLite transactions/CAS transitions, metrics, failures, attempt history,
  persisted sequential race results. Checkpoints are authoritative on recovery.
- Presentation: Typer/Rich CLI and self-contained escaped HTML, real showcase exports.
- Tests: real CPU learning, new-object recovery equivalence, write-failure regression,
  database contention/state tests and installed CLI subprocess tests.
- Risks: interruption during optimizer mutation; filesystem/database commit boundary;
  checkpoint trust; concurrent resumes. Test and document each boundary.
- Non-goals: web backend/UI, distributed training, remote datasets, cloud storage,
  optional CIFAR-10, replay and external DataLoader support.
