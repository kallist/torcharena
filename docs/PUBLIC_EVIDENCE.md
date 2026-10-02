# Public claim and evidence map

V0.1 main revision: `b24a976617d6ee6a5c4cfc8e7a84b7e9139e34c7`.
Verified 2026-10-03. This supplements the historical local-delivery records without
relabeling them as Hosted CI results.

| Public claim | Inspectable evidence | Boundary |
| --- | --- | --- |
| Config-driven training, preflight and finite guards | [Architecture](ARCHITECTURE.md), `torcharena/config.py`, `torcharena/safety.py`, `tests/test_behavior.py` | Two owned models and one synthetic dataset; no arbitrary config execution |
| Full-state checkpoint/resume | [Recovery contract](RECOVERY.md), `torcharena/checkpoint.py`, `torcharena/trainer.py` | Trusted checkpoints, same config/device type/PyTorch version |
| Six interruption positions | `tests/test_behavior.py` recovery-equivalence parametrization | Owned deterministic CPU data; not arbitrary DataLoaders or cross-hardware proof |
| 67 tests on Python 3.11 and 3.12 | [Merged main run 37044363967](https://github.com/kallist/torcharena/actions/runs/37044363967), `cpu (3.11)` / `cpu (3.12)` | Each CPU job runs the suite; one documented StepLR warning |
| Docker smoke PASS | Same run, `docker`: build, doctor, tiny CNN training | Docker does not run the 67-test suite; no CUDA container validation |
| Showcase recovery parameter error 0 | [Transcript](../artifacts/showcase/crash-resume.txt), [JSON](../artifacts/showcase/crash-resume.json) | This recorded CPU run, `atol=1e-7`; not a universal claim |
| Model Race uses real updates | [Transcript](../artifacts/showcase/race.txt), [JSON](../artifacts/showcase/race.json) | Sequential, comparable tiny training; no accuracy/performance benchmark claim |
| Failure stays inspectable | [Transcript](../artifacts/showcase/graveyard.txt), [JSON](../artifacts/showcase/graveyard.json) | Explicit demo-only NaN; last healthy snapshot retained |
| Visuals preserve real output | [Asset provenance](assets/provenance.json), [generation guide](assets/README.md) | Transcript replay, editorial pacing, not a timed recording |
| MIT, with torchkeras inspiration | [MIT](../LICENSE), [reference audit](REFERENCE_AUDIT.md) | Common training concepts are not claimed as inventions; direct source reuse recorded as none |

## Recruiting boundary

These sources prove repository behavior, validation and delivery, not a candidate's
unaided authorship or workplace impact. The copy in RESUME_COPY.md is phrased around
the delivered project. Do not replace it with “独立手写全部源码”, “生产级”, team-lead
claims, business outcomes or performance improvements without separate evidence.
When asked how it was built, describe AI-assisted implementation and the actual
decisions, checks and review you can explain yourself.
