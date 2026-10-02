# Separate engineering review

Reviewed 2026-10-03 after implementation and initial training/CLI validation.
This is a separate skeptical self-review pass by the implementing agent, not an
external human approval or a second-agent review. No independent human approval
is claimed.

## Findings and fixes

| Severity | Finding | Resolution / evidence |
| --- | --- | --- |
| IMPORTANT | Report assumed a run directory already existed | Create the report parent; report escaping test covers a CREATED record |
| IMPORTANT | Recovery badge could say YES for changed config/runtime or wrong tensor shape | Revalidate config/hash/device/torch and model shape/dtype under isolated RNG; changed-config and corrupt-state regressions |
| IMPORTANT | Writing failure JSON before DB could leave RUNNING on a full disk | Commit failure/status first; retain original exception and export error note; disk-failure regression |
| IMPORTANT | Sequential Race would continue to another contestant after interruption | Persist interrupted race and stop; regression executes real interrupted training and checks only one run exists |
| IMPORTANT | Structurally readable checkpoint with wrong model tensor shape raised raw RuntimeError | Fail closed with actionable corrupt/incompatible error before status change; regression |
| IMPORTANT | Artifact paths could follow a checkpoints-directory symlink outside the managed run | Validate IDs and resolved managed paths; external checkpoints path rejected |
| NIT | Race terminal table omitted dataset/seed despite JSON retaining them | Add visible contestant metadata and installed CLI assertions |
| NIT | Hard termination lost timing accumulated before intermediate snapshots | Accumulate elapsed segment time at each safe save; document remaining write overhead limitation |

## Reviewed boundaries

- Real Adam update, train/eval mode, cross-entropy and sample-weighted epoch metrics.
- Dropout makes RNG restoration part of the exact CPU continuation proof.
- Fresh model/optimizer/scheduler/scaler objects; no in-memory object reuse in resume tests.
- Batch cursor includes end-of-epoch interruption; validation/scheduler advances once.
- Healthy snapshot is preserved when finite guards reject loss/gradient/parameters.
- File fsync/replace, failed write/replace tests, bounded retention and ignored stale temps.
- OS ownership lock, transactional CAS status changes, failure insert rollback and read contention.
- Checkpoint-authoritative SQLite reconciliation avoids ghost/duplicate epoch metrics.
- YAML safe loader, strict closed schema, no config imports/eval/shell or fault-injection fields.
- HTML escapes names/config/failures; full state deserialization is explicitly trusted-only.
- Race fairness checks precede training; accuracy ties remain ties; no overall champion.
- Generated outputs contain actual training measurements and recorded failures, not static output.
- Reference architecture/license inspection and no copied reference source/assets.
- CPU workflows require no secret/GPU/dataset download; Hosted execution remains unobserved.

## Remaining limits

No remaining known correctness blocker in the tested CPU built-in paths. StepLR may
emit an ordering warning on one first-epoch resume boundary because optimizer private
call markers are not persisted; tensor/optimizer/scheduler state equality passes.
CUDA/AMP, Python 3.11/Linux, Docker execution and Hosted CI are not locally established.
File/directory power-loss durability, distributed locks, untrusted checkpoints,
custom DataLoader recovery and cross-hardware bitwise equality are not promised.
No static type checker is configured; strict config validation, type hints and Ruff
are not claimed as mypy verification. See VALIDATION.md for exact executed evidence.
