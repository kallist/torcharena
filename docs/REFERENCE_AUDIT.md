# Reference audit

Reference: [lyhue1991/torchkeras](https://github.com/lyhue1991/torchkeras), master,
inspected 2026-10-03 at commit `c7afdb987572491f077da20c8c6e057317267cb7`.
License: [Apache-2.0](https://github.com/lyhue1991/torchkeras/blob/c7afdb987572491f077da20c8c6e057317267cb7/LICENSE).
Sources studied: README, `torchkeras/kerasmodel.py` and LICENSE. This is a bounded
architecture inspection, not an exhaustive novelty search or performance comparison.

torchkeras offers a Keras-style PyTorch training template. Its abstraction layers
are StepRunner, EpochRunner and KerasModel: batch work, epoch aggregation, then
fit/evaluate lifecycle. Its README documents metrics, early stopping, GPU/DDP,
fp16/bf16, notebook visualization, TensorBoard and W&B callbacks. The source
contains checkpoint save/load behavior, optimizer/scheduler integration and
Accelerator-based execution. TorchArena does not claim these common capabilities
as inventions.

In the inspected `kerasmodel.py`, `KerasModel.save_ckpt` saves the network state;
`load_ckpt` restores network weights. TorchArena's local continuation snapshots
include optimizer, scheduler, scaler, RNG and cursor. This is a concrete difference
in these inspected implementations, not a claim about every upstream extension.

## Borrowed ideas and independent design

Ideas studied: small understandable training loops, measured validation metrics,
early stopping and readable feedback. TorchArena is independently written around
a closed config registry, local SQLite experiment state, explicit failure records,
atomic versioned full-state checkpoints and behavior-tested synthetic mid-epoch
recovery. Its CLI demonstrations run real training and persist inspectable history.
This product focus is deliberately different from a Keras-style general training
wrapper. The Race is sequential comparison, not resource competition.

TorchArena's own work combines preflight isolation, recorded failure graveyard,
new-object recovery equivalence and generated terminal/HTML evidence. These are
project-specific improvements, not claims that no other tool has similar features.
TorchArena does not claim to be better than torchkeras, replace MLflow/W&B or invent
training loops, checkpointing, RNG restoration, SQLite, residual networks or AMP.

## License handling

Direct source reuse: **none**. No reference code, notebooks, images or assets were
copied into this repository. No torchkeras runtime dependency is used. TorchArena's
original source is MIT licensed. Reference attribution is retained here and in the
README. If source reuse is introduced later, identify exact portions and preserve
the applicable Apache-2.0 license/notice obligations before distribution.
