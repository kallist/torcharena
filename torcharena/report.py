"""Self-contained HTML, no CDN or script execution from experiment content."""

import html
import json

import torch

from torcharena.checkpoint import load_checkpoint
from torcharena.config import RunConfig
from torcharena.models import make_model
from torcharena.runtime import device_for, isolated_rng
from torcharena.storage import confined


def recoverability(repository, run_id: str) -> tuple[bool, str]:
    record = repository.get(run_id)
    path = repository.run_dir(run_id) / "checkpoints" / "last.pt"
    try:
        confined(repository.home, path)
        config = RunConfig.model_validate(record["config"])
        if config.config_hash != record["config_hash"]:
            raise ValueError("Stored config/hash mismatch")
        state = load_checkpoint(path, config.config_hash, run_id)
        if state["device_type"] != device_for(config).type:
            raise ValueError("Checkpoint device type is unavailable or changed")
        if state["runtime"]["torch"] != str(torch.__version__):
            raise ValueError("Checkpoint PyTorch version mismatch")
        with isolated_rng():
            expected = make_model(config).state_dict()
        if expected.keys() != state["model"].keys() or any(
            expected[key].shape != state["model"][key].shape
            or expected[key].dtype != state["model"][key].dtype
            for key in expected
        ):
            raise ValueError("Checkpoint model shape/dtype mismatch")
    except (ValueError, OSError) as error:
        return False, str(error)
    return record["status"] in {"FAILED", "INTERRUPTED", "RUNNING", "RESUMING"}, (
        path.relative_to(repository.home).as_posix()
    )


def inspect_data(repository, run_id: str) -> dict:
    return {
        **repository.get(run_id),
        "metrics": repository.metrics(run_id),
        "failures": repository.failures(run_id),
        "transitions": repository.history(run_id),
        "recovery": recoverability(repository, run_id),
    }


def report(repository, run_id: str):
    record = inspect_data(repository, run_id)

    def escape(value):
        return html.escape(str(value))

    config = record["config"]
    metrics = record["metrics"]
    rows = "".join(
        f"<tr><td>{m['epoch']}</td><td>{m['train_loss']:.6f}</td>"
        f"<td>{m['val_loss']:.6f}</td><td>{m['val_accuracy']:.2%}</td></tr>"
        for m in metrics
    )
    # SVG is generated exclusively from measured numeric metrics.
    curve = ""
    if metrics:
        peak = max(m["val_loss"] for m in metrics) or 1.0
        points = " ".join(
            f"{30 + i * 540 / max(1, len(metrics) - 1):.1f},{180 - m['val_loss'] / peak * 140:.1f}"
            for i, m in enumerate(metrics)
        )
        curve = (
            f'<svg viewBox="0 0 600 220" role="img" aria-label="Validation loss by epoch">'
            f'<polyline points="{points}" fill="none" stroke="#a78bfa" stroke-width="3"/>'
            '<text x="30" y="210" fill="#cbd5e1">Validation loss · epoch →</text></svg>'
        )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TorchArena — {escape(record["name"])}</title><style>
body{{background:#0f172a;color:#e2e8f0;font:16px system-ui;margin:0;padding:4vw}}
main{{max-width:1000px;margin:auto}}h1{{font-size:3rem}}h2{{color:#c4b5fd}}
.tag{{color:#fbbf24}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1e293b;
padding:20px;border-radius:12px}}table{{width:100%;border-collapse:collapse}}
td,th{{text-align:left;padding:12px;border-bottom:1px solid #475569}}svg{{max-width:700px}}
</style><main><p class="tag">TRAIN. RACE. BREAK. RESUME.</p><h1>TorchArena</h1>
<h2>{escape(record["name"])} · {escape(record["status"])}</h2>
<p>{escape(config["model"]["name"])} / {escape(config["dataset"]["name"])} / seed
{config["experiment"]["seed"]}</p><p>Run: {escape(run_id)}</p>
<h2>Battle report</h2><pre>{escape(json.dumps(record["summary"], indent=2))}</pre>
<h2>Metric history</h2>{curve}<table><tr><th>Epoch</th><th>Train loss</th><th>Val loss</th>
<th>Val accuracy (%)</th></tr>{rows}</table>
<h2>Recovery</h2><pre>{escape(json.dumps(record["recovery"], indent=2))}</pre>
<h2>Reproducibility</h2><pre>{escape(json.dumps(record["environment"], indent=2))}</pre>
<p>Config SHA256: {escape(record["config_hash"])}</p>
<h2>Config</h2><pre>{escape(json.dumps(config, indent=2))}</pre>
<h2>Failures / detected conditions</h2><pre>{escape(json.dumps(record["failures"], indent=2))}</pre>
<h2>Attempt history</h2><pre>{escape(json.dumps(record["transitions"], indent=2))}</pre>
<p>Local trusted checkpoints only. Timing is descriptive; this is not a controlled benchmark.</p>
</main></html>"""
    path = confined(repository.home, repository.run_dir(run_id) / "report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
