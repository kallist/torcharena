"""Render presentation assets from committed CLI transcripts; requires Pillow only."""

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#0c111b"
PANEL = "#141e2b"
BORDER = "#33465b"
WHITE = "#edf2f7"
MUTED = "#a8bbce"
ORANGE = "#ffb36b"
CYAN = "#7bddeb"
SCENES = (
    ("race", "Model Race", "model-race", ORANGE),
    ("crash-resume", "Crash → Resume", "crash-resume", CYAN),
    ("graveyard", "Failure Graveyard", "failure-graveyard", ORANGE),
)
SOURCE_COMMIT = "b24a976617d6ee6a5c4cfc8e7a84b7e9139e34c7"


def font(size, mono=False, bold=False, override=None):
    if override:
        return ImageFont.truetype(override, size)
    candidates = (
        ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf")
        if mono
        else (
            ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
            if bold
            else ("segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
        )
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Install a TrueType font or pass --mono-font and --sans-font")


class Canvas:
    """Keep the social graphic's editable SVG and PNG on the same geometry."""

    def __init__(self, size):
        self.image = Image.new("RGB", size, BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" '
            f'height="{size[1]}" viewBox="0 0 {size[0]} {size[1]}">',
            "<title>TorchArena — Train. Race. Break. Resume.</title>",
            "<desc>PyTorch training workbench: two model lanes, a full-state checkpoint, "
            "and three CLI showcases. Schematic; no benchmark numbers.</desc>",
            f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        ]

    def rect(self, box, fill=PANEL, outline=BORDER, radius=12):
        self.draw.rounded_rectangle(box, radius, fill, outline, width=1)
        x1, y1, x2, y2 = box
        self.svg.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" '
            f'rx="{radius}" fill="{fill}" stroke="{outline}"/>'
        )

    def line(self, points, color=BORDER, width=2):
        self.draw.line(points, fill=color, width=width)
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>'
        )

    def text(self, xy, text, size, color=WHITE, mono=False, bold=False, override=None):
        face = font(size, mono, bold, override)
        self.draw.text(xy, text, font=face, fill=color, anchor="lt")
        family = "Consolas, monospace" if mono else "Segoe UI, sans-serif"
        self.svg.append(
            f'<text x="{xy[0]}" y="{xy[1]}" dominant-baseline="text-before-edge" '
            f'font-family="{family}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" fill="{color}">{html.escape(text)}</text>'
        )


def social_preview(out, sans):
    canvas = Canvas((1280, 640))
    canvas.text((80, 48), "PYTORCH / TRAINING WORKBENCH", 18, CYAN, override=sans)
    canvas.text((76, 92), "TORCHARENA", 80, bold=True, override=sans)
    canvas.text((80, 195), "Train. Race. Break. Resume.", 32, ORANGE, override=sans)
    canvas.text(
        (80, 246), "Compare models. Continue training. Inspect failures.", 21, MUTED, override=sans
    )
    canvas.rect((80, 306, 770, 448))
    canvas.text((104, 323), "MODEL LANES / SEQUENTIAL TRAINING", 14, MUTED, mono=True)
    for y, label, color in ((365, "tiny_cnn", ORANGE), (409, "tiny_resnet", CYAN)):
        canvas.text((104, y - 7), label, 18, color, mono=True)
        canvas.line([(268, y), (732, y)], color, 3)
        for x in (320, 445, 570, 700):
            canvas.rect((x - 5, y - 5, x + 5, y + 5), color, color, 2)
    canvas.rect((804, 306, 1200, 448))
    canvas.text((828, 324), "FULL-STATE CHECKPOINT", 17, CYAN, mono=True)
    canvas.text((828, 366), "interrupt → restore → continue", 17, WHITE, override=sans)
    canvas.text((828, 403), "State + RNG + batch cursor", 17, MUTED, override=sans)
    for x, number, title in (
        (80, "01", "Model Race"),
        (460, "02", "Crash Recovery"),
        (840, "03", "Failure Graveyard"),
    ):
        canvas.rect((x, 484, x + 360, 556))
        canvas.text((x + 18, 507), number, 18, CYAN, mono=True)
        canvas.text((x + 62, 502), title, 22, bold=True, override=sans)
    canvas.text((80, 592), "CLI + PyTorch + SQLite", 16, MUTED, override=sans)
    canvas.text((968, 592), "kallist/torcharena", 16, MUTED, mono=True)
    canvas.image.save(out / "social-preview.png", optimize=True)
    (out / "social-preview.svg").write_text(
        "\n".join(canvas.svg + ["</svg>"]) + "\n", encoding="utf-8"
    )


def transcript(path):
    content = path.read_text(encoding="utf-8")
    unsafe = (
        r"[A-Za-z]:[/\\]|/Users/|/home/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
        r"|github_pat_|gh[pousr]_|sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}"
    )
    if re.search(unsafe, content):
        raise ValueError(f"Privacy scan failed: {path.name}; recapture before rendering")
    return content.splitlines()


def terminal_frame(lines, visible, index, title, command, accent, mono, sans, height=640):
    image = Image.new("RGB", (1120, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 12, 1108, height - 12), radius=14, fill=PANEL, outline=BORDER)
    draw.text((32, 28), "TORCHARENA", font=font(20, bold=True, override=sans), fill=WHITE)
    draw.text((795, 34), "RECORDED CLI OUTPUT", font=font(14, mono=True, override=mono), fill=MUTED)
    draw.line((32, 70, 1088, 70), fill=BORDER)
    draw.text((32, 87), f"0{index + 1} / 03   {title}", font=font(21, override=sans), fill=accent)
    draw.text((32, 121), "$ " + command, font=font(17, mono=True, override=mono), fill=CYAN)
    size = 16
    while size > 10:
        face = font(size, mono=True, override=mono)
        if max(face.getlength(line) for line in lines) <= 1056:
            break
        size -= 1
    if max(face.getlength(line) for line in lines) > 1056:
        raise ValueError("Transcript exceeds canvas width; refusing to crop output")
    line_height = 21
    if 174 + len(lines) * line_height > height - 30:
        raise ValueError("Transcript exceeds canvas height; refusing to crop output")
    for row, line in enumerate(lines[:visible]):
        draw.text((32, 174 + row * line_height), line, font=face, fill=WHITE)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--mono-font")
    parser.add_argument("--sans-font")
    parser.add_argument("--source-ref", default=SOURCE_COMMIT)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_ref):
        parser.error("--source-ref must be a full commit SHA")
    args.output.mkdir(parents=True, exist_ok=True)
    social_preview(args.output, args.sans_font)
    frames, durations, sources = [], [], []
    for index, (stem, title, export, accent) in enumerate(SCENES):
        path = ROOT / "artifacts" / "showcase" / f"{stem}.txt"
        lines = transcript(path)
        text = path.read_text(encoding="utf-8")
        recorded = (
            subprocess.check_output(
                ["git", "show", f"{args.source_ref}:{path.relative_to(ROOT).as_posix()}"],
                cwd=ROOT,
            )
            .decode("utf-8")
            .replace("\r\n", "\n")
        )
        if text != recorded:
            raise ValueError(
                "Transcript differs from source revision; commit it and set --source-ref"
            )
        command = f"torcharena showcase {stem}"
        full = terminal_frame(
            lines,
            len(lines),
            index,
            title,
            command,
            accent,
            args.mono_font,
            args.sans_font,
            height=max(400, 174 + len(lines) * 21 + 42),
        )
        full.save(args.output / f"{export}.png", optimize=True)
        # Six progressive views per scene; elapsed GIF time is editorial, not runtime.
        for fraction, duration in (
            (0, 400),
            (0.2, 700),
            (0.4, 900),
            (0.6, 1000),
            (0.8, 1000),
            (1, 3300),
        ):
            frame = terminal_frame(
                lines,
                int(len(lines) * fraction),
                index,
                title,
                command,
                accent,
                args.mono_font,
                args.sans_font,
            )
            frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
            durations.append(duration)
        sources.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "lines": len(lines),
            }
        )
    gif = args.output / "torcharena-demo.gif"
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    if gif.stat().st_size >= 8_000_000:
        raise ValueError("GIF exceeds preferred 8 MB budget")
    manifest = {
        "source_commit": args.source_ref,
        "hash_format": "SHA-256 of UTF-8 text with LF line endings",
        "source_note": (
            "Committed real-output transcripts; original runs and provenance remain "
            "in artifacts/showcase JSON/HTML. This is a replay, not a timed screen "
            "recording or a new training run."
        ),
        "transcripts": sources,
        "gif_duration_ms": sum(durations),
        "gif_frames": len(frames),
        "assets": [
            {"path": p.name, "bytes": p.stat().st_size, "size": list(Image.open(p).size)}
            for p in sorted(args.output.iterdir())
            if p.suffix in (".png", ".gif")
        ],
    }
    (args.output / "provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
