"""Build a combined Clay + DINOv2 report at data/review/combined_results.html.

Reads existing scored results from:
  - data/review/embedding_ranking.html  (DINOv2 scores)
  - data/review/clay_results.html        (Clay v1.5 scores)

and cross-references them with the actual thumbnail PNG files in data/review/.

Usage:
    python scripts/build_combined_report.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from herring_spawner.config import Settings

# ---------------------------------------------------------------------------
# 1. Parse DINOv2 scores from embedding_ranking.html
# ---------------------------------------------------------------------------
def parse_dinov2_scores(html_path: Path) -> dict[str, float]:
    """Extract {filename: score} from the existing DINOv2 ranking HTML."""
    if not html_path.exists():
        print(f"WARNING: {html_path} not found -- no DINOv2 scores")
        return {}

    html = html_path.read_text()
    scores = {}

    # Pattern: <td>...score...</td> followed by <td>filename</td>
    rows = re.findall(
        r'<tr[^>]*>.*?<td>\d+</td>\s*<td>[^<]*</td>\s*<td>([\d.]+)</td>\s*<td>([^<]+\.png)</td>',
        html, re.DOTALL
    )
    for score_str, fname in rows:
        scores[fname.strip()] = float(score_str)

    # Try alternate pattern
    if not scores:
        rows = re.findall(
            r'<td>([\d.]+)</td>\s*<td>([^<]+\.png)</td>',
            html
        )
        for score_str, fname in rows:
            scores[fname.strip()] = float(score_str)

    print(f"Parsed {len(scores)} DINOv2 scores from {html_path.name}")
    return scores


# ---------------------------------------------------------------------------
# 2. Parse Clay scores from clay_results.html
# ---------------------------------------------------------------------------
def parse_clay_scores(html_path: Path) -> dict[str, float]:
    """Extract {event_name: score} from the Clay results HTML."""
    if not html_path.exists():
        print(f"WARNING: {html_path} not found -- no Clay scores")
        return {}

    html = html_path.read_text()
    scores = {}

    # Pattern matching: <tr class="pos|detect|cand"><td>N</td><td><strong>name</STR...><td>score</td>
    rows = re.findall(
        r'<tr class="[^"]*">\s*<td>(\d+)</td>\s*<td><strong>([^<]+)</strong>',
        html
    )
    score_vals = re.findall(
        r'<tr class="[^"]*">.*?<td>\d+</td>.*?<td>([\d.]+)</td>',
        html
    )

    if rows and score_vals and len(rows) == len(score_vals):
        for (_, name), score_str in zip(rows, score_vals):
            scores[name.strip()] = float(score_str)

    print(f"Parsed {len(scores)} Clay scores from {html_path.name}")
    return scores


# ---------------------------------------------------------------------------
# 3. Map Clay event names to review thumbnail filenames
# ---------------------------------------------------------------------------
CLAY_TO_REVIEW_MAP = {
    "pos-salmon": "news-salmon-beach-2025",
    "pos-ucluelet": "dfo-verified-ucluelet",
    "pos-qualicum": "dfo-verified-qualicum-beach",
    "pos-anderson": "dfo-verified-anderson-point",
    "pos-breakwater": "dfo-verified-breakwater-island",
    "fan-island-spawn": "dfo-verified-fan-island",
}


def map_clay_to_filenames(
    clay_scores: dict[str, float],
    review_files: list[str],
) -> dict[str, float]:
    """Map Clay scores (by event name) to review filenames."""
    mapped: dict[str, float] = {}
    for clay_name, score in clay_scores.items():
        # Direct match if clay_name matches part of a filename
        for fname in review_files:
            if clay_name in fname.replace("_", "-"):
                mapped[fname] = score
                break

        # Use explicit map
        if clay_name in CLAY_TO_REVIEW_MAP:
            prefix = CLAY_TO_REVIEW_MAP[clay_name]
            for fname in review_files:
                if fname.startswith(prefix):
                    if fname not in mapped or score > mapped.get(fname, 0):
                        mapped[fname] = score

    return mapped


# ---------------------------------------------------------------------------
# 4. Generate combined HTML
# ---------------------------------------------------------------------------
def generate_html(
    dinov2_scores: dict[str, float],
    clay_scores: dict[str, float],
    review_dir: Path,
) -> str:
    """Generate the combined Clay + DINOv2 results HTML page."""

    # Collect all unique filenames with their scores
    all_files = set(dinov2_scores.keys()) | set(clay_scores.keys())

    # Also get any PNGs in the review dir
    existing_pngs = {p.name for p in review_dir.glob("*.png")}
    all_files.update(existing_pngs)

    # Build rows
    rows_data = []
    for fname in sorted(existing_pngs):
        d2 = dinov2_scores.get(fname, None)
        cl = clay_scores.get(fname, None)
        # Use the best available score for ranking (DINOv2 preferred, then Clay)
        rank_score = d2 if d2 is not None else cl if cl is not None else 0.0
        rows_data.append({
            "filename": fname,
            "dinov2": d2,
            "clay": cl,
            "rank_score": rank_score,
        })

    # Sort by DINOv2 score descending, then Clay score
    rows_data.sort(key=lambda r: -(r["dinov2"] or r["clay"] or 0))

    # Extract event_id from filename for grouping
    def event_id_from_fname(fname: str) -> str:
        # pattern: event_id_date_scenedate.png or event_id_sceneid_date.png
        # Simple: everything before the last two _ groups
        stem = Path(fname).stem
        # Try to find known event prefixes
        for known in [
            "dfo-verified-anderson-point",
            "dfo-verified-breakwater-island",
            "dfo-verified-fan-island",
            "dfo-verified-qualicum-beach",
            "dfo-verified-tree-bluff",
            "dfo-verified-ucluelet",
            "manual-2026-04-04-event-1-point-1",
            "manual-2026-04-04-event-1-point-2",
            "manual-2026-04-04-event-2-point-1",
            "manual-2026-04-04-event-2-point-2",
            "news-nanaimo-2025",
            "news-salmon-beach-2025",
        ]:
            if stem.startswith(known):
                return known
        return stem

    # Group by event
    by_event: defaultdict[str, list] = defaultdict(list)
    for r in rows_data:
        eid = event_id_from_fname(r["filename"])
        by_event[eid].append(r)

    # HTML generation
    style = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1a1a2e; }
h1 { margin: 0; font-size: 1.6rem; font-weight: 600; }
h2 { font-size: 1.2rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: #1a1a2e; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 1.5rem 2rem; }
.header p { margin: 0.5rem 0 0; opacity: 0.85; font-size: 0.9rem; }
.content { max-width: 1400px; margin: 0 auto; padding: 1.5rem; }
.stats-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card { background: white; padding: 1rem 1.5rem; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); flex: 1; min-width: 140px; }
.stat-num { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
.stat-label { font-size: 0.75rem; text-transform: uppercase; color: #888; letter-spacing: 0.04em; }
.event-card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 1.5rem; overflow: hidden; }
.event-header { padding: 0.75rem 1.25rem; font-weight: 600; font-size: 1rem; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 0.5rem; }
.event-header .count { background: #e8ecf4; padding: 0.15rem 0.6rem; border-radius: 10px; font-size: 0.75rem; color: #555; }
.chip-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 0; }
.chip-card { padding: 1rem; border-right: 1px solid #f0f0f0; border-bottom: 1px solid #f0f0f0; display: flex; flex-direction: column; }
.chip-card:nth-child(4n) { border-right: none; }
.chip-card:last-child { border-bottom: none; }
.chip-thumb { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 6px; border: 1px solid #e0e0e0; background: #fafafa; margin-bottom: 0.6rem; }
.chip-info { font-size: 0.8rem; color: #555; line-height: 1.5; }
.chip-info .date { font-weight: 500; color: #333; }
.score-row { display: flex; gap: 1rem; margin-top: 0.4rem; font-size: 0.8rem; }
.score-badge { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
.score-dinov2 { background: #e8f0fe; color: #1967d2; }
.score-clay { background: #fce8e6; color: #c5221f; }
.score-high { background: #e6f4ea; color: #1e8e3e; }
.score-mid { background: #fef7e0; color: #ea8600; }
.score-low { background: #fce8e6; color: #c5221f; }
.footer { text-align: center; padding: 1rem; color: #888; font-size: 0.8rem; }
a { color: #1967d2; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

    # Stats
    total_chips = len(rows_data)
    d2_scores_list = [r["dinov2"] for r in rows_data if r["dinov2"] is not None]
    cl_scores_list = [r["clay"] for r in rows_data if r["clay"] is not None]
    d2_mean = sum(d2_scores_list) / len(d2_scores_list) if d2_scores_list else 0
    cl_mean = sum(cl_scores_list) / len(cl_scores_list) if cl_scores_list else 0

    event_cards_html = ""
    for eid in sorted(by_event.keys()):
        chips = by_event[eid]
        # Sort chips within event by date
        chips.sort(key=lambda r: r["filename"])

        chips_html = ""
        for r in chips:
            fname = r["filename"]
            d2_score = r["dinov2"]
            cl_score = r["clay"]

            # Score display
            def score_html(val, label, css_class):
                if val is None:
                    return ""
                cls = css_class
                if val >= 0.85:
                    cls += " score-high"
                elif val >= 0.80:
                    cls += " score-mid"
                else:
                    cls += " score-low"
                return f'<span class="score-badge {cls}">{label}: {val:.4f}</span>'

            d2_html = score_html(d2_score, "DINOv2", "score-dinov2")
            cl_html = score_html(cl_score, "Clay", "score-clay")

            # Extract date from filename for display
            date_display = fname.split("_")[-2] if "_" in fname else ""

            # Determine if thumbnail file actually exists
            img_path = review_dir / fname
            img_src = fname if img_path.exists() else ""

            chips_html += f"""
            <div class="chip-card">
                <img class="chip-thumb" src="{img_src}" alt="{fname}" onerror="this.alt='missing'">
                <div class="chip-info">
                    <div class="date">{date_display}</div>
                    <div>{fname[:60]}{'...' if len(fname) > 60 else ''}</div>
                    <div class="score-row">
                        {d2_html}
                        {cl_html}
                    </div>
                </div>
            </div>"""

        event_cards_html += f"""
        <div class="event-card">
            <div class="event-header">
                {eid}
                <span class="count">{len(chips)} chip{'s' if len(chips) != 1 else ''}</span>
            </div>
            <div class="chip-grid">
                {chips_html}
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clay + DINOv2 Combined Results</title>
<style>{style}</style>
</head>
<body>

<div class="header">
    <h1>Clay v1.5 + DINOv2 &mdash; Herring Spawn Detection</h1>
    <p>Combined embedding similarity scores for Sentinel-2 review thumbnails</p>
</div>

<div class="content">

<div class="stats-row">
    <div class="stat-card">
        <div class="stat-num">{total_chips}</div>
        <div class="stat-label">Total Chips</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{len(by_event)}</div>
        <div class="stat-label">Events</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{d2_mean:.4f}</div>
        <div class="stat-label">DINOv2 Mean Score</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{cl_mean:.4f}</div>
        <div class="stat-label">Clay Mean Score</div>
    </div>
</div>

<p style="color:#555;font-size:0.9rem;margin-bottom:1.5rem;">
    <strong>DINOv2</strong> (ViT-S/14) pretrained on ImageNet &mdash; scores are cosine similarity to mean of 6 confirmed spawn images. &nbsp;
    <strong>Clay v1.5</strong> multispectral encoder trained on SSL4EO &mdash; scores are cosine similarity in embedding space. &nbsp;
    Results grouped by event, sorted by DINOv2 score descending.
</p>

{event_cards_html}

<div class="footer">
    Generated from data/review/ thumbnails &middot; GEE project: redd-fish
</div>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    review_dir = Settings().review_dir
    interim_dir = Settings().interim_dir

    dinov2_scores = parse_dinov2_scores(review_dir / "embedding_ranking.html")
    clay_scores = parse_clay_scores(review_dir / "clay_results.html")

    # Map clay scores to filenames
    existing_pngs = [p.name for p in review_dir.glob("*.png")]
    clay_mapped = map_clay_to_filenames(clay_scores, existing_pngs)

    print(f"Clay scores mapped to {len(clay_mapped)} filenames")

    html = generate_html(dinov2_scores, clay_mapped, review_dir)

    output = review_dir / "combined_results.html"
    output.write_text(html, encoding="utf-8")
    print(f"Combined report written to {output}")
    print(f"  file://{output.resolve()}")


if __name__ == "__main__":
    main()
