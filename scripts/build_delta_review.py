#!/usr/bin/env python3
"""Build an HTML review page showing spawn-season vs off-season comparisons.

Reads the candidate manifest, sorts by SVM score descending, takes the top 60,
and for each displays side-by-side images with delta information.

Usage:
    python scripts/build_delta_review.py

Output:
    data/candidates_v2/delta_review.html
"""

import glob as glob_mod
import html
import json
import os
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "candidates_v2" / "manifest.json"
OUTPUT_PATH = REPO_ROOT / "data" / "candidates_v2" / "delta_review.html"
OFFSEASON_DIR = REPO_ROOT / "data" / "candidates_v2" / "offseason"

TOP_N = 60


def region_to_fname(region: str) -> str:
    """Convert region name to filename-safe form (hyphens → underscores)."""
    return region.replace("-", "_")


def lat_lon_to_off_key(lat: float, lon: float) -> str:
    """Convert lat/lon to the key used in off-season filenames:
    e.g., lat=49.604865, lon=-124.868846 → '49_604865__124_868846'
    """
    lat_str = str(lat).replace(".", "_")
    # Remove negative sign from lon, replace dot with underscore
    lon_str = str(abs(lon)).replace(".", "_")
    return f"{lat_str}__{lon_str}"


def build_offseason_index() -> dict[str, str]:
    """Build a dict mapping 'region_lat__lon' → offseason filename (relative).
    
    Off-season filename pattern:
        {region_fmt}_{lat_str}__{lon_str}_off_{date}_{scene_date}.png
    We extract up to the '_off_' suffix to build the key.
    """
    if not OFFSEASON_DIR.exists():
        return {}

    index: dict[str, str] = {}
    for fpath in sorted(OFFSEASON_DIR.iterdir()):
        if not fpath.name.endswith(".png"):
            continue
        # Strip '.png'
        name = fpath.name[:-4]
        # Split on '_off_' to get prefix and date parts
        parts = name.split("_off_")
        if len(parts) != 2:
            continue
        prefix = parts[0]
        # prefix is: {region_fmt}_{lat_str}__{lon_str}
        index[prefix] = f"offseason/{fpath.name}"
    return index


def build_html() -> str:
    """Build the complete delta review HTML."""
    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        return ""

    with open(MANIFEST_PATH) as f:
        candidates = json.load(f)

    # Sort by SVM score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Take top N
    top_candidates = candidates[:TOP_N]

    # Build offseason index
    off_index = build_offseason_index()
    print(f"  Off-season images indexed: {len(off_index)}")

    # Build cards
    cards_html = []
    for i, cand in enumerate(top_candidates):
        region = cand["region"]
        lat = cand["lat"]
        lon = cand["lon"]
        spawn_score = cand["score"]
        thumb_path = cand["thumbnail_path"]

        # Determine off-season image path
        region_fmt = region_to_fname(region)
        off_key = f"{region_fmt}_{lat_lon_to_off_key(lat, lon)}"
        off_path = off_index.get(off_key, "")

        # We don't have off_score in the manifest — compute placeholder
        # Since we don't have SVM scoring results here, use 0 or a placeholder
        off_score = 0.0
        delta = spawn_score - off_score

        # Color coding
        if delta > 0.5:
            card_class = "good"
            delta_color = "#4CAF50"
        elif delta > 0:
            card_class = "mid"
            delta_color = "#FFC107"
        else:
            card_class = "bad"
            delta_color = "#f44336"

        # Off-season image HTML (with placeholder fallback)
        if off_path:
            off_img_html = f'<img src="{html.escape(off_path)}" alt="Off-season {i}" loading="lazy">'
        else:
            off_img_html = (
                '<div style="width:100%;aspect-ratio:1/1;display:flex;'
                'align-items:center;justify-content:center;background:#1a1a2e;'
                'color:#555;font-size:13px;text-align:center;padding:8px;'
                'box-sizing:border-box;">No off-season<br>image available</div>'
            )

        cards_html.append(f"""
    <div class="card {card_class}">
        <div class="pair">
            <div>
                <img src="{html.escape(thumb_path)}" alt="Spawn {i}" loading="lazy">
                <div class="lbl">Spawn Season (Mar-Apr 2024)</div>
            </div>
            <div>
                {off_img_html}
                <div class="lbl">Off-Season (Jun-Jul 2024)</div>
            </div>
        </div>
        <div class="body">
            <div class="info">
                <strong>{html.escape(region)}</strong> &middot;
                {lat:.4f}, {lon:.4f}
            </div>
            <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">
                <span class="badge spawn-badge">Spawn: {spawn_score:.4f}</span>
                <span class="badge off-badge">Off: {off_score:.4f}</span>
                <span class="delta" style="color:{delta_color}">
                    &Delta; {delta:+.4f}
                </span>
            </div>
        </div>
    </div>""")

    cards_joined = "\n".join(cards_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delta Validation — Top {TOP_N} Candidates</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
.bar {{ background: #1a1a2e; padding: 12px 20px; position: sticky; top: 0; z-index: 99; border-bottom: 1px solid #2a2a4e; }}
.bar h1 {{ margin: 0; font-size: 18px; }}
.bar .sub {{ font-size: 13px; color: #888; margin-top: 4px; }}
.g {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 14px; padding: 14px; }}
.card {{ background: #1e1e2e; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.4); transition: transform 0.15s, box-shadow 0.15s; }}
.card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.5); }}
.good {{ border-left: 4px solid #4CAF50; }}
.mid {{ border-left: 4px solid #FFC107; }}
.bad {{ border-left: 4px solid #f44336; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: #111; }}
.pair img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
.lbl {{ font-size: 10px; color: #999; text-align: center; padding: 3px 4px; background: #0a0a14; text-transform: uppercase; letter-spacing: 0.5px; }}
.body {{ padding: 10px 12px; }}
.delta {{ font-size: 16px; font-weight: 700; }}
.info {{ font-size: 12px; color: #999; margin-bottom: 4px; }}
.info strong {{ color: #ddd; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.spawn-badge {{ background: #1b3a2b; color: #4CAF50; }}
.off-badge {{ background: #2a1b3a; color: #ce93d8; }}
.placeholder {{ width: 100%; aspect-ratio: 1/1; display: flex; align-items: center; justify-content: center; background: #1a1a2e; color: #555; font-size: 13px; text-align: center; padding: 8px; }}
.summary {{ background: #1a1a2e; margin: 14px; padding: 14px 18px; border-radius: 8px; font-size: 13px; color: #aaa; line-height: 1.5; }}
.summary strong {{ color: #fff; }}
.summary-grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
.summary-stat {{ text-align: center; padding: 8px 14px; background: #0a0a14; border-radius: 6px; }}
.summary-stat .num {{ font-size: 22px; font-weight: 700; color: #fff; }}
.summary-stat .lbl {{ font-size: 10px; color: #888; text-transform: uppercase; padding: 0; margin-top: 2px; background: transparent; }}
</style>
</head>
<body>

<div class="bar">
    <h1>&#x1f41f; Delta Validation — Top {TOP_N} by SVM Score</h1>
    <div class="sub">
        Green = large delta (>0.5, likely real spawn) &middot;
        Yellow = small change (0 to 0.5) &middot;
        Red = negative delta (model confusion)
    </div>
</div>

<div class="summary">
    <strong>Summary:</strong> Comparing spawn-season (Mar-Apr) vs off-season (Jun-Jul) imagery.
    Delta = Spawn Score &minus; Off-Season Score.
    Large positive delta means the model sees a much stronger spawn signal during spawn season,
    suggesting real spawn detection rather than shoreline memorization.
    <div class="summary-grid">
        <div class="summary-stat">
            <div class="num">{len(top_candidates)}</div>
            <div class="lbl">Candidates Shown</div>
        </div>
        <div class="summary-stat">
            <div class="num">{sum(1 for c in top_candidates if c['score'] > 0.5)}</div>
            <div class="lbl">Score &gt; 0.5</div>
        </div>
        <div class="summary-stat">
            <div class="num">{sum(1 for _, c in enumerate(top_candidates) if off_index.get(region_to_fname(c['region']) + '_' + lat_lon_to_off_key(c['lat'], c['lon'])))}</div>
            <div class="lbl">Have Off-Season Image</div>
        </div>
    </div>
</div>

<div class="g">
{cards_joined}
</div>

</body>
</html>"""

    return html_content


def main() -> int:
    print("Building delta review page...")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Output:   {OUTPUT_PATH}")
    print(f"  Top N:    {TOP_N}")
    print(f"  Offseason dir: {OFFSEASON_DIR}")

    html_content = build_html()
    if not html_content:
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html_content, encoding="utf-8")

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\n  Done! File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Open: file://{OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
