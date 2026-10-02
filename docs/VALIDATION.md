# Executed validation

Date: 2026-10-03 (Asia/Shanghai). Local Windows CPU evidence only.
Python 3.12.14, torch 2.8.0+cpu, NumPy 2.5.3, pytest 8.4.2, Ruff 0.16.10.
Base: `1dcbbcaafc8117044fdd9aca5cf722afb00dd8cb`.
Training source commit: `90a3e5fe802148f08af4f2bca4ead82008fb3ebf`.
Final Race metadata/path-test commit: `e166b49b0a23455b808116fb7d12dd01395cc490`.
Final tested training/recovery source: `930fadc1f11dab4d07e487d638f9f169c63aee8c`.
Final showcase metadata records the latter commit and dirty=true while documentation
and generated evidence were pending. This is retained honestly in exported reports.

## Installation and quality

Commands below were run in PowerShell; activate .venv to use short command names.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dev]' --no-build-isolation --index-url https://pypi.org/simple --disable-pip-version-check --no-input
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
git diff --check
.\.venv\Scripts\python.exe -m build --no-isolation
```

Editable/dev install: PASS. Dependencies: no broken requirements. Ruff check and
format check: PASS. Build: real sdist and wheel produced. No static type checker is
configured: NOT IMPLEMENTED / NOT TESTED, not a hidden PASS.

## Tests and exact results

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_acceptance_tmp --tb=short
# 66 passed, 1 warning in 100.33s before the final recovery-badge review
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_delivery_tmp --tb=short
# 67 passed, 1 warning in 106.55s on the final training/recovery source
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py::test_installed_cli_race_persists_real_runs tests/test_cli.py::test_installed_cli_showcases -q --basetemp=.pytest_race_metadata_tmp --tb=short
# 4 passed in 31.55s after final Race display/assertion changes
.\.venv\Scripts\python.exe -m pytest tests/test_units.py::test_artifact_confinement_rejects_external_path -q --basetemp=.pytest_path_tmp --tb=short
# 1 passed in 2.51s after adding the final path regression
```

Final coverage accounting: 37 unit cases, 20 training behavior cases and 10 installed
CLI cases, all in the final single 67-case full-suite PASS. The 4 CLI rechecks repeat
existing cases, not 4 extra tests. Final recovery badge hardening also received two
corrupt/config regressions plus three installed showcase rechecks (5 PASS in 40.12s).
The full suite includes those final assertions.

Flagship proof: six interruption points (steps 1, 3, 4, 5, 15, 16) discard original
objects and reconstruct a fresh Trainer. Final metrics, model tensors, optimizer and
scheduler match exactly on the deterministic CPU fixture, including dropout RNG.
Roundtrip also checks nonempty optimizer state and RNG. Separate subprocess fixture
calls os._exit(17): DB remains RUNNING at the last snapshot (step 2), and a new CLI
process recovers under an exclusive OS lock and finishes at step 16.

The one warning is StepLR's private optimizer-call marker after first-epoch-final-
batch resume. No assertion is skipped/removed and no private marker is modified.
Tensor/optimizer/scheduler equality passes. Details: RECOVERY.md.

Initial failures were not hidden: sandboxed pytest failed during temporary-directory
creation (WinError 5), before training fixtures could execute. Authorized execution
resolved that environment boundary. A first product run found a report parent-directory
bug (51 PASS / 1 FAIL), which was fixed and regression-tested before the full PASS.
Other review fixes and regression evidence are in REVIEW.md.

## Final product demonstrations

```powershell
$env:PYTHONUTF8='1'
$env:TORCHARENA_HOME='artifacts/showcase/runtime/final'
.\.venv\Scripts\torcharena.exe showcase race
.\.venv\Scripts\torcharena.exe showcase crash-resume
.\.venv\Scripts\torcharena.exe showcase graveyard
```

All three exit successfully, execute real training and generate recorded TXT/HTML/JSON.
Runtime DB/checkpoints are ignored; only deliberate shareable artifacts are committed.

- Race: `run-4141941f8dce46ed9c78e906a66699c8`, `run-06656865c41f4eec8a9e72c86a991d5c`; seed 42,
  same 64/32 split, 4 epochs, CPU. Both accuracy fractions are 1.0 on the intentionally
  easy fixture. Val losses 0.00007874906077631749 and 0.000003263348048676562.
  Measured fit times 0.380950s and 0.397924s;
  not a controlled performance benchmark. Parameters: 554 and 702.
- Recovery: step 3 interruption → new objects → step 16 COMPLETED. Baseline metric
  history matches; max parameter error 0 against an explicit 1e-7 tolerance.
- Graveyard: `run-0f55a5ffea9b482fa45aad993e074553`, FAILED / non_finite_loss. Failed attempt step 3;
  last durable healthy step 2. Failure row/traceback/config exist, recoverability YES.
  Confirmed cause is explicit demo injection, not automated root-cause speculation.

Report generation is tested both by the installed CLI and the showcase helpers.
Reports have no CDN/scripts and escape untrusted names and failure content.

## Environment-limited gates

- Dockerfile/.dockerignore: IMPLEMENTED. `docker version` detects client 29.5.3
  but cannot reach Docker Desktop Linux engine. Image build/run: NOT TESTED / BLOCKED.
- GitHub Actions: IMPLEMENTED (.github/workflows/ci.yml), CPU Python 3.11/3.12 plus
  Docker smoke. Hosted run/URL: none. Status: BLOCKED; no remote is configured.
- GitHub authentication was verified. Push and Draft PR cannot target an unspecified
  repository. Prepared PR text: DRAFT_PR.md. No remote, tag, merge or release was created.
- CUDA/AMP and CUDA RNG restore: NOT TESTED (installed torch is CPU-only).
- Python 3.11, Linux and cross-hardware behavior: NOT TESTED locally.
- Generic external DataLoaders, CIFAR-10, replay, serialized-size comparison, provider
  integrations, cloud/DDP/web UI and static typing gates: NOT IMPLEMENTED.

## Acceptance boundary

All mandatory product and local verification gates possible in this environment
are satisfied. Local showcase/interview readiness does not assert Hosted CI PASS,
Docker execution, production reliability or universal bitwise reproducibility.
