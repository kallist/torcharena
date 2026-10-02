# Portfolio presentation validation

Date: 2026-10-03. Branch: `chore/torcharena-portfolio-polish`.
Base: `b24a976617d6ee6a5c4cfc8e7a84b7e9139e34c7`.

## Scope review

Training source, tests, configuration schemas, checkpoint behavior, SQLite schema,
package dependencies, Dockerfile, workflow and original showcase evidence are
unchanged. Changes are README/docs, a scoped optional asset-environment ignore,
the standalone Pillow renderer, SVG/PNG/GIF and the source manifest.

## Executed local checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp=.pytest_portfolio_tmp
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m build --no-isolation
git diff --check
```

- pytest: **67 passed, 1 warning in 98.10s**, Windows Python 3.12 CPU. The retained
  warning is the previously documented StepLR resume boundary; assertions were not changed.
- Ruff: PASS; format check: PASS, 36 Python files checked.
- Build: PASS, sdist and wheel. Asset generation is separate from runtime packaging.
- Renderer: executed successfully using the available Pillow asset runtime; no
  Pillow dependency was added to TorchArena. Source text must match the base commit.
- Link validation: 77 Markdown links checked across README and five new guides;
  local files/anchors resolved, all 10 distinct external URLs returned HTTP 200.
- Image validation: four PNGs fully decoded; SVG parsed as XML; GIF decoded across
  all 18 frames at 1120 × 640, total 21,900 ms, 287,964 bytes.
- Transcript provenance: canonical UTF-8/LF SHA-256 values match all three sources;
  CRLF checkout conversion does not invalidate provenance.
- GIF fidelity: decoded final frames for all three scenes exactly match the
  corresponding renderer's palette images. Pacing is editorial, not execution time.

## Privacy and claims review

All transcript inputs passed the renderer's private-path/email/credential checks.
The social graphic, three complete scenes and decoded final GIF frames were viewed.
Shared frame generation uses only prefixes of those reviewed source lines, plus
fixed public captions. No terminal chrome, private paths, emails, account tokens,
hostname or machine-information output is included. A source-text scan matched only
the generator's intentional detection patterns, not an actual credential.

No new speed, accuracy, coverage, adoption or production claims. NaN remains labeled
demo-only. CUDA/AMP stays NOT TESTED; generic external DataLoader resume and DDP stay
NOT IMPLEMENTED. Resume error 0 refers to the documented deterministic CPU run.
Recruiting copy explicitly distinguishes AI assistance and project capability from
unaided authorship/personal mastery. BOSS copy is 137 characters.

## Hosted evidence and account settings

The merged V0.1 [main run](https://github.com/kallist/torcharena/actions/runs/37044363967)
was inspected separately: all jobs passed, 67 tests per Python version, Docker smoke.
Current presentation-branch checks must be read from its Draft PR after pushing;
the baseline run is not evidence for an untested future HEAD.

Description and twelve Topics were applied and verified through GitHub API/CLI.
Website remains blank. Repository-pin and social-preview upload are not exposed by
available tools; [manual steps](GITHUB_PROFILE.md) are delivered. No merge, tag,
release, profile reordering or new portfolio deployment is part of this task.
