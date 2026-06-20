#!/usr/bin/env python3
"""Classify candidate thumbnails using a turquoise/teal color signature.

The classifier is intentionally simple:
  1. Learn the target hue/saturation/value signature from known spawn images.
  2. Measure how much each candidate thumbnail contains that signature.
  3. Compare the turquoise fraction against a background baseline learned from
     the candidate pool, then label the result.

Outputs:
  - data/candidates_v2/color_classification.html

Usage:
  python scripts/color_classify.py
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "candidates_v2" / "manifest.json"
OUTPUT_PATH = REPO_ROOT / "data" / "candidates_v2" / "color_classification.html"

DEFAULT_REFERENCE_IMAGES = [
    REPO_ROOT / "data" / "candidates_v2" / "nanaimo_2024-03-18_score1.03_49.135_-123.677_20240318.png",
    REPO_ROOT / "data" / "candidates_v2" / "nootka-sound_2024-03-16_score0.99_49.585_-126.609_20240316.png",
]


@dataclass(frozen=True)
class Metrics:
    turquoise_frac: float
    turquoise_share: float
    color_frac: float
    white_frac: float
    dark_frac: float
    turquoise_pixels: int
    total_pixels: int


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB→HSV conversion.

    Input: uint8 or float RGB array shaped (H, W, 3) in [0, 255].
    Output: float HSV array shaped (H, W, 3) where H is degrees [0, 360),
    S/V are [0, 1].
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    nonzero = delta > 1e-8

    rc = ((g - b) / (delta + 1e-8)) % 6.0
    gc = ((b - r) / (delta + 1e-8)) + 2.0
    bc = ((r - g) / (delta + 1e-8)) + 4.0

    rmax = (maxc == r) & nonzero
    gmax = (maxc == g) & nonzero
    bmax = (maxc == b) & nonzero

    h = np.where(rmax, 60.0 * rc, h)
    h = np.where(gmax, 60.0 * gc, h)
    h = np.where(bmax, 60.0 * bc, h)
    h = np.mod(h, 360.0)

    s = np.where(maxc <= 1e-8, 0.0, delta / (maxc + 1e-8))
    v = maxc
    return np.stack([h, s, v], axis=-1)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resolve_image_path(path: Path) -> Path:
    """Resolve a reference image path, falling back to a best-match glob.

    The user-provided reference paths are approximate. If the exact file does
    not exist, look for sibling files with the same region/date prefix and pick
    the highest-score match.
    """
    if path.exists():
        return path

    prefix = path.stem.split("_score", 1)[0]
    matches = sorted(path.parent.glob(f"{prefix}_score*.png"))
    if not matches:
        raise FileNotFoundError(path)

    def score_key(p: Path) -> float:
        stem = p.stem
        try:
            score_text = stem.split("_score", 1)[1].split("_", 1)[0]
            return float(score_text)
        except Exception:
            return -1.0

    return max(matches, key=score_key)


def resize_for_analysis(img: Image.Image, max_side: int = 256) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return copy


def to_data_uri(img: Image.Image, max_side: int = 160) -> str:
    thumb = resize_for_analysis(img, max_side=max_side)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def compute_metrics(img: Image.Image) -> Metrics:
    sample = resize_for_analysis(img, max_side=256)
    rgb = np.asarray(sample)
    hsv = rgb_to_hsv_np(rgb)

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    color_mask = (s >= 0.14) & (v >= 0.12)
    turquoise_mask = (h >= 140.0) & (h <= 205.0) & color_mask
    white_mask = (s <= 0.20) & (v >= 0.80)
    dark_mask = v <= 0.18

    total = int(h.size)
    color = int(color_mask.sum())
    turquoise = int(turquoise_mask.sum())
    white = int(white_mask.sum())
    dark = int(dark_mask.sum())

    return Metrics(
        turquoise_frac=turquoise / total,
        turquoise_share=turquoise / color if color else 0.0,
        color_frac=color / total,
        white_frac=white / total,
        dark_frac=dark / total,
        turquoise_pixels=turquoise,
        total_pixels=total,
    )


def classify(metrics: Metrics, baseline: float, ref_floor: float) -> tuple[str, float]:
    delta = metrics.turquoise_frac - baseline
    if metrics.turquoise_frac >= ref_floor and delta >= 0.30:
        return "SPAWN", delta
    if metrics.white_frac >= max(0.22, metrics.turquoise_frac * 1.25):
        return "WAVES", delta
    return "OTHER", delta


def build_html(
    reference_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    baseline: float,
    ref_floor: float,
) -> str:
    counts = {"SPAWN": 0, "WAVES": 0, "OTHER": 0}
    for row in candidate_rows:
        counts[row["label"]] += 1

    ref_cards = []
    for row in reference_rows:
        ref_cards.append(
            f"""
            <div class="ref-card">
              <img src="{row['thumb_uri']}" alt="reference">
                <div class="meta"><strong>{html.escape(str(row['name']))}</strong></div>
              <div class="meta">turquoise {row['metrics'].turquoise_frac:.3f} · share {row['metrics'].turquoise_share:.3f}</div>
            </div>
            """
        )

    candidate_cards = []
    for row in candidate_rows:
        badge_class = {
            "SPAWN": "spawn",
            "WAVES": "waves",
            "OTHER": "other",
        }[row["label"]]
        candidate_cards.append(
            f"""
            <div class="card {badge_class}">
              <img src="{row['thumb_uri']}" alt="candidate">
              <div class="body">
                <div class="title">{html.escape(str(row['region']))}</div>
                <div class="sub">{html.escape(str(row['date']))} · score {row['score']:.4f}</div>
                <div class="sub">turquoise {row['metrics'].turquoise_frac:.3f} · share {row['metrics'].turquoise_share:.3f} · color {row['metrics'].color_frac:.3f} · white {row['metrics'].white_frac:.3f} · dark {row['metrics'].dark_frac:.3f}</div>
                <div class="badge">{row['label']}</div>
                <div class="sub">delta {row['delta']:+.3f}</div>
              </div>
            </div>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Color Classification</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #101214; color: #e8e8e8; }}
    .top {{ position: sticky; top: 0; background: #16191d; border-bottom: 1px solid #2a2f36; padding: 14px 18px; z-index: 10; }}
    .top h1 {{ margin: 0 0 6px 0; font-size: 18px; }}
    .top .sub {{ color: #9ba3af; font-size: 13px; line-height: 1.4; }}
    .summary {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 14px 18px 0; }}
    .stat {{ background: #16191d; border: 1px solid #2a2f36; border-radius: 10px; padding: 10px 12px; min-width: 120px; }}
    .stat .n {{ font-size: 20px; font-weight: 700; }}
    .stat .l {{ color: #9ba3af; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
    .section {{ padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }}
    .ref-card, .card {{ background: #16191d; border: 1px solid #2a2f36; border-radius: 12px; overflow: hidden; }}
    .ref-card img, .card img {{ width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: cover; background: #0c0e10; }}
    .ref-card .meta, .card .body {{ padding: 8px 10px; }}
    .card .body {{ line-height: 1.35; }}
    .title {{ font-weight: 700; font-size: 13px; }}
    .sub {{ color: #9ba3af; font-size: 11px; margin-top: 2px; word-break: break-word; }}
    .badge {{ display: inline-block; margin-top: 6px; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .spawn {{ box-shadow: inset 0 0 0 1px rgba(46, 204, 113, .55); }}
    .spawn .badge {{ background: #173321; color: #7ff0a0; }}
    .waves {{ box-shadow: inset 0 0 0 1px rgba(255, 193, 7, .55); }}
    .waves .badge {{ background: #332a11; color: #ffd766; }}
    .other {{ box-shadow: inset 0 0 0 1px rgba(148, 163, 184, .35); }}
    .other .badge {{ background: #23272d; color: #c8d0da; }}
    h2 {{ margin: 0 0 10px 0; font-size: 15px; }}
    .muted {{ color: #9ba3af; font-size: 12px; }}
    code {{ background: #0c0e10; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="top">
    <h1>Color-based spawn classification</h1>
    <div class="sub">
      Baseline turquoise fraction: <code>{baseline:.3f}</code> · reference floor: <code>{ref_floor:.3f}</code> · spawn threshold: <code>delta ≥ 0.300</code>
    </div>
  </div>

  <div class="summary">
    <div class="stat"><div class="n">{len(candidate_rows)}</div><div class="l">candidates</div></div>
    <div class="stat"><div class="n">{counts['SPAWN']}</div><div class="l">spawn</div></div>
    <div class="stat"><div class="n">{counts['WAVES']}</div><div class="l">waves</div></div>
    <div class="stat"><div class="n">{counts['OTHER']}</div><div class="l">other</div></div>
  </div>

  <div class="section">
    <h2>Confirmed spawn references</h2>
    <div class="grid">{''.join(ref_cards)}</div>
  </div>

  <div class="section">
    <h2>Classified candidates</h2>
    <div class="muted">Sorted by strongest turquoise delta first.</div>
    <div class="grid">{''.join(candidate_cards)}</div>
  </div>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--references",
        type=Path,
        nargs="*",
        default=DEFAULT_REFERENCE_IMAGES,
        help="Confirmed spawn reference images",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())

    references: list[dict[str, object]] = []
    ref_fracs: list[float] = []
    for path in args.references:
        resolved = resolve_image_path(path)
        img = load_rgb(resolved)
        metrics = compute_metrics(img)
        ref_fracs.append(metrics.turquoise_frac)
        references.append({
            "name": resolved.name,
            "thumb_uri": to_data_uri(img),
            "metrics": metrics,
        })

    ref_floor = min(ref_fracs) * 0.60 if ref_fracs else 0.0

    scored_rows: list[dict[str, object]] = []
    raw_metrics: list[Metrics] = []
    for row in manifest:
        img_path = args.manifest.parent / row["thumbnail_path"]
        img = load_rgb(img_path)
        metrics = compute_metrics(img)
        raw_metrics.append(metrics)
        scored_rows.append({
            "region": row["region"],
            "date": row["date"],
            "score": float(row["score"]),
            "thumb_uri": to_data_uri(img),
            "metrics": metrics,
            "row": row,
        })

    # Robust off-season proxy from the lower turquoise tail of the candidate pool.
    sorted_fracs = sorted(m.turquoise_frac for m in raw_metrics)
    bottom_n = max(10, len(sorted_fracs) // 5)
    baseline = float(np.median(sorted_fracs[:bottom_n])) if sorted_fracs else 0.0

    classified: list[dict[str, object]] = []
    for item in scored_rows:
        label, delta = classify(item["metrics"], baseline=baseline, ref_floor=ref_floor)
        item["label"] = label
        item["delta"] = delta
        classified.append(item)

    classified.sort(key=lambda x: (x["label"] != "SPAWN", -x["delta"], -x["metrics"].turquoise_frac))

    html_text = build_html(references, classified, baseline, ref_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")

    counts = {"SPAWN": 0, "WAVES": 0, "OTHER": 0}
    for item in classified:
        counts[item["label"]] += 1

    print(f"Wrote {args.output}")
    print(f"Candidates: {len(classified)}")
    print(f"Baseline turquoise fraction: {baseline:.4f}")
    print(f"Reference floor: {ref_floor:.4f}")
    print(f"Spawn: {counts['SPAWN']}  Waves: {counts['WAVES']}  Other: {counts['OTHER']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
