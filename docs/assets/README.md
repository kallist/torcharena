# Showcase visual sources

The PNG screenshots and 21.9-second GIF replay the real, committed terminal
transcripts in `artifacts/showcase/`. They do not invent output, represent a fresh
run or preserve real execution timing. Full transcripts, measured JSON and original
HTML provenance stay unchanged. `provenance.json` records source SHA-256 hashes,
the source revision, image dimensions, duration and byte sizes. Transcript hashes
use UTF-8 with LF line endings, so Windows checkout conversion does not change them.

`social-preview.svg` is the editable vector source for the 1280 × 640 PNG. Its lane
and checkpoint motif is schematic, with no performance numbers. The PNG is below
GitHub's 1 MB social-preview limit. The GIF is below the preferred 8 MB budget.

## Regenerate

Use Python 3.11+ with Pillow installed in an optional asset environment; Pillow is
not a TorchArena runtime dependency. From the repository root:

```bash
python -m venv .asset-venv
# Activate .asset-venv using your shell's normal activation command.
python -m pip install Pillow
python scripts/generate_showcase_assets.py
```

The renderer uses installed Segoe UI/Consolas, DejaVu or Liberation TrueType fonts.
For another font, pass `--sans-font <font.ttf> --mono-font <monospace.ttf>`.
Different fonts/Pillow versions can change pixels and byte sizes; transcript text
and values remain identical. The committed exports were made with Segoe UI/Consolas.
No font binaries, raw video, terminal window chrome or machine metadata are committed.
Git is required to check that the transcript text matches the recorded revision.
The default source revision is the merged V0.1 commit. To use new genuine transcripts,
commit them first and pass `--source-ref <full-commit-sha>`; uncommitted replacements
are rejected rather than attributed to an older run.

The generator refuses likely private paths, emails or credential patterns in source
transcripts and refuses to crop overflowing output. Inspect every resulting scene
before committing. The same source lines drive the screenshots and all GIF frames.

## Recording a longer video

See [the Chinese demo script](../DEMO_SCRIPT_ZH.md). For a live recording, use a clean
terminal, activate the development environment, and run the three documented CLI
commands. The graveyard command deliberately injects NaN; label this on screen.
Use isolated `TORCHARENA_HOME` and `TORCHARENA_SHOWCASE_DIR` directories outside the
tracked evidence. Do not overwrite the original showcase transcripts for cosmetic
reasons. Screen recording or FFmpeg is optional; no raw video is required for README.
