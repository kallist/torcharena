# feat: add TorchArena v0.1 reproducible training workbench

Delivery blocker: no Git remote is configured. This is a prepared Draft PR body,
not an opened PR. Keep the eventual PR Draft; do not merge, tag or release.

## Summary

Adds a lightweight local config-first PyTorch workbench with strict safe YAML,
two tiny models, isolated preflight, finite guards, SQLite history and atomic
full-state checkpoint recovery. Resume reconstructs fresh objects and continues
owned deterministic batches from a saved cursor.

## Product demo

- Real sequential Model Race with dataset/seed/budget comparability and objective categories.
- Crash → Resume at step 3, new objects, persisted restore, final step 16.
- Explicit demo-only NaN failure, FAILED record, queryable graveyard and healthy snapshot.
- Rich terminal exports and self-contained HTML battle reports.

## Architecture

CLI/presentation is separate from the tensor engine; service composition adds local
OS ownership and transactional SQLite state. Checkpoint replacement is atomic per
file. Snapshots are authoritative when filesystem and DB commits diverge. All
config changes are rejected on resume; no distributed or cloud architecture.

## Reference-project differences

Studied torchkeras's StepRunner/EpochRunner/KerasModel and Apache-2.0 license.
Training loops, metrics, early stopping, GPU/AMP and checkpointing are common/existing
capabilities. TorchArena focuses on inspectable failure/recovery history and
behavior-tested full-state continuation. Direct reference source/asset reuse: none.
Original source is MIT. See docs/REFERENCE_AUDIT.md.

## Testing

See docs/VALIDATION.md for exact commands. Final full local CPU suite: 67 PASS, one
documented StepLR warning. Final Race/three-showcase CLI rechecks: 4 PASS; recovery
badge regressions/showcase rechecks: 5 PASS. Ruff and dependency checks PASS. Static type checking is not
configured. Docker execution and Hosted CI have not passed or been observed.

## Showcase evidence

artifacts/showcase contains genuine TXT/HTML/JSON for all three commands. Recovery
matches the baseline with maximum parameter error 0 at a documented 1e-7 tolerance.
No fabricated metric/output, overall winner or real-world accuracy claim.

## Known limitations

Local Windows Python 3.12 CPU evidence only. CUDA/AMP, Python 3.11/Linux and Docker
execution need separate evidence. No generic external DataLoader recovery, DDP,
remote artifacts, web frontend, universal bitwise reproducibility or production claim.
Docker engine was unavailable; no remote exists for push/Hosted CI.

## Security

Closed schema and registries, parameterized SQL, managed path confinement, escaped
HTML and no provider keys/calls. Full-state torch.load is trusted-only, not sandboxed.
Atomic file replacement does not guarantee power-loss or distributed durability.
