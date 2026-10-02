# Recovery contract

## Schema 1

Each trusted full-state checkpoint contains run identity, canonical config SHA256,
schema version, device type and PyTorch version; model, Adam optimizer, StepLR
scheduler and GradScaler state; Python/NumPy (if installed)/torch CPU/CUDA RNG;
completed epoch, next batch cursor, global step, partial weighted epoch loss/correct/
sample totals, metric history, best metric/epoch, early-stop counter and stop flag.

The CPU scaler is disabled and its state is empty by design. CUDA AMP uses a real
GradScaler. Scheduler advances after each completed epoch. Best means the configured
early-stopping metric/direction, even when stopping is disabled. last.pt is for
continuation; best.pt is retained for inspection, not a CLI resume target.

## Atomic writes and retention

Create a unique temporary file in the destination directory; serialize, flush,
fsync the file, close it, and use os.replace. A failed serialization/replacement
does not destroy the old target. The current writer removes its own temp file in
finally. Hard termination may leave ignored temporary files; they are never loaded
or automatically deleted across owners. Bounded step snapshots are retained after
last.pt replacement. This is not a guarantee against power loss/filesystem failure:
directory fsync is not implemented, and checkpoint sets are not one multi-file commit.

## Safe units and interruption

SIGINT/KeyboardInterrupt from the terminal is deferred by a temporary main-thread
handler until after a finite optimizer update and committed cursor, or after epoch
validation/scheduler/metric updates. The flag triggers the same KeyboardInterrupt
recovery branch used by the demo. Signal handlers are restored on exit. A second
SIGINT remains deferred; SIGKILL/process termination is not catchable.

An unexpected KeyboardInterrupt raised inside an unsafe mutation preserves the last
durable snapshot rather than serializing half an update. A healthy interruption at
a safe boundary writes last.pt and marks INTERRUPTED. Resume output includes a real
run ID. Failure guards mark FAILED and preserve the previous known-good snapshot.
Non-finite parameters after an update are not saved.

For hard termination, the OS releases ownership. Resume acquires the exclusive lock,
loads the latest previously written last.pt and records abandoned-process recovery.
Uncheckpointed work is repeated. A process dying before the first snapshot cannot
resume. CREATED/VALIDATING abandoned records cannot resume without valid state.

## Mid-epoch semantics

The built-in dataset is generated with split-specific isolated generators. Each
epoch has a deterministic permutation based on seed + epoch, independent of global
RNG. Cursor indexes the **next** batch; partially completed epoch totals are saved.
On resume a fresh model/optimizer/scheduler/scaler is created, all states loaded and
RNG restored **after** object construction, then consumed batches are skipped. No
worker-prefetch or random augmentation state exists in V0.1.

If interrupted after the final batch but before epoch finalization, cursor equals
batch count; resume validates the epoch exactly once and advances the scheduler.
Cursor/global-step/history consistency is checked. The configured early-stop state
is preserved so recovery cannot accidentally train extra epochs.

## Filesystem / database boundary

last.pt is authoritative. A crash may happen between its replacement and SQLite
progress/metric writes; resume rebuilds SQLite metric history from the checkpoint
and sets progress to the saved cursor. Metric uniqueness is (run_id, epoch).
Failure/status updates are atomic SQLite transactions. Metrics JSONL is a derived
human-readable export; it may lag after hard termination and is rebuilt on completion.
Run directories created before startup failure may remain for diagnosis. Disk/database
errors exit as errors; neither SQLite nor files are claimed to survive arbitrary
storage loss. Nonterminal DB status alone does not prove a process is alive.
Failure status is committed before its optional JSON export. If the export also
fails, the original exception gains a visible diagnostic note; a full disk need
not prevent an available database from recording FAILED. Duration is accumulated
at each safe snapshot and excludes downtime; work after a hard-crash checkpoint
and the last snapshot's own write overhead are not recoverable timing evidence.

## Compatibility and trust

All V0.1 config fields are resume-critical; no override flags exist. The recorded
config is revalidated and its hash compared to last.pt. Device type and exact torch
version must match. Checkpoint schema/run identity must match. Changed environment
or Python versions remain an inspection concern; no cross-hardware equivalence claim.

Only load TorchArena checkpoints created locally or obtained from trusted sources.
Full state uses `torch.load(weights_only=False)` including RNG metadata; deserialization
is not sandboxed. IDs and managed artifact paths are validated, but file access is
not a security boundary against a malicious user who can modify the local store.
Report/run listing may inspect trusted checkpoints to determine recoverability.

## Evidence boundary

Behavioral tests reconstruct fresh objects and compare metrics, model tensors,
optimizer and scheduler at interruption points including epoch boundaries. CPU demo
uses parameter tolerance 1e-7; tests use exact equality for this environment. This
does not promise universal bitwise reproducibility. Deterministic kernels can cost
performance; unsupported CUDA deterministic operations can fail. See
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).
GPU/AMP, cross-platform execution, real-world datasets and arbitrary external
DataLoaders require separate evidence.

PyTorch StepLR can emit an ordering warning when resuming immediately after the
last batch of the very first epoch, before that epoch's scheduler step: the new
optimizer's private call marker is not serialized. The restored optimizer moments
and scheduler learning rates are tested for exact equivalence; TorchArena does not
mutate private PyTorch flags or suppress this warning.
