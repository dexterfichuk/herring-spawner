#!/usr/bin/env python3
"""Generate a sortable review HTML page from a candidates manifest.json.

Usage:
    python scripts/generate_review_page.py \
        --candidates-dir data/candidates_svm_8pos \
        --port 8774
"""
import argparse
import json
import sys
from pathlib import Path


def generate_review_page(candidates_dir: Path) -> str:
    """Generate HTML review page from candidate manifest and return file path."""
    manifest_path = candidates_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest.json found in {candidates_dir}")
        return ""

    entries = json.loads(manifest_path.read_text())
    if not entries:
        print("ERROR: Manifest is empty")
        return ""

    # Sort by score descending
    entries_sorted = sorted(entries, key=lambda e: e.get("score", 0), reverse=True)

    # Build cards
    cards = []
    for e in entries_sorted:
        thumb = e.get("thumbnail_path", "")
        score = e.get("score", 0)
        region = e.get("region", "?")
        lat = e.get("lat", 0)
        lon = e.get("lon", 0)
        date = e.get("date", "?")
        scene_id = e.get("scene_id", "?")
        cloud = e.get("cloud", "?")
        # Color class based on score
        score_class = "high" if score > 0.5 else ("mid" if score > 0.2 else "low")

        cards.append(f"""
    <div class="card {score_class}" data-score="{score}" data-region="{region}" data-date="{date}">
        <img src="{thumb}" alt="" loading="lazy" onclick="toggleZoom(this)">
        <div class="body">
            <div class="region">{region}</div>
            <div class="coords">{lat:.4f}, {lon:.4f}</div>
            <div class="meta">{date} &middot; {scene_id[:30]} &middot; cloud {cloud}%</div>
            <div class="score">Score: {score:.4f}</div>
        </div>
    </div>""")

    cards_html = "\n".join(cards)

    # Count top scores
    high_count = sum(1 for e in entries_sorted if e.get("score", 0) > 0.5)
    mid_count = sum(1 for e in entries_sorted if 0.2 < e.get("score", 0) <= 0.5)
    low_count = sum(1 for e in entries_sorted if e.get("score", 0) <= 0.2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Herring Spawn Candidates — SVM 8pos+126neg Review</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d0d1a; color: #eee; }}
.bar {{ background: #1a1a2e; padding: 14px 20px; position: sticky; top: 0; z-index: 99; border-bottom: 1px solid #333; }}
.bar h1 {{ margin: 0; font-size: 20px; }}
.bar .sub {{ font-size: 13px; color: #888; margin-top: 4px; }}
.g {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; padding: 12px; }}
.card {{ background: #1a1a2e; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.5); cursor: pointer; }}
.high {{ border-left: 5px solid #00E676; }}
.mid {{ border-left: 5px solid #FFD740; }}
.low {{ border-left: 5px solid #FF5252; }}
.card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; transition: transform 0.2s; }}
.card img.zoomed {{ transform: scale(2); transform-origin: top left; position: relative; z-index: 10; }}
.body {{ padding: 10px 14px; }}
.region {{ font-weight: 700; font-size: 15px; color: #fff; }}
.coords {{ font-size: 12px; color: #888; }}
.meta {{ font-size: 11px; color: #666; margin-top: 2px; }}
.score {{ margin-top: 6px; font-size: 14px; font-weight: 600; }}
.high .score {{ color: #00E676; }}
.mid .score {{ color: #FFD740; }}
.low .score {{ color: #FF5252; }}
.summary {{ background: #1a1a2e; margin: 12px; padding: 16px 20px; border-radius: 8px; font-size: 13px; }}
.summary-stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }}
.stat {{ text-align: center; padding: 8px 16px; background: #0a0a18; border-radius: 8px; min-width: 100px; }}
.stat .num {{ font-size: 24px; font-weight: 700; color: #fff; }}
.stat .lbl {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.stat.high .num {{ color: #00E676; }}
.stat.mid .num {{ color: #FFD740; }}
.stat.low .num {{ color: #FF5252; }}
.sort-bar {{ display: flex; gap: 8px; padding: 8px 12px; background: #151528; align-items: center; flex-wrap: wrap; }}
.sort-bar label {{ font-size: 12px; color: #888; }}
.sort-bar select {{ background: #1a1a2e; color: #eee; border: 1px solid #333; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
#search {{ flex: 1; max-width: 300px; background: #1a1a2e; color: #eee; border: 1px solid #333; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<div class="bar">
    <h1>Herring Spawn Candidates — SVM (8 pos + 126 neg)</h1>
    <div class="sub">{len(entries_sorted)} candidates from {len(set(e.get("region","?") for e in entries_sorted))} regions &middot; DINOv2+SVM RBF &middot; Sorted by score</div>
</div>
<div class="summary">
    <strong>Score distribution:</strong>
    <div class="summary-stats">
        <div class="stat high"><div class="num">{high_count}</div><div class="lbl">High (&gt;0.5)</div></div>
        <div class="stat mid"><div class="num">{mid_count}</div><div class="lbl">Mid (0.2-0.5)</div></div>
        <div class="stat low"><div class="num">{low_count}</div><div class="lbl">Low (&le;0.2)</div></div>
    </div>
</div>
<div class="sort-bar">
    <label for="sortSelect">Sort:</label>
    <select id="sortSelect" onchange="sortCards()">
        <option value="score-desc">Score ↓</option>
        <option value="score-asc">Score ↑</option>
        <option value="region">Region A-Z</option>
        <option value="date">Date</option>
    </select>
    <label for="filterSelect">Filter:</label>
    <select id="filterSelect" onchange="filterCards()">
        <option value="all">All</option>
        <option value="high">High (&gt;0.5)</option>
        <option value="mid">Mid (0.2-0.5)</option>
        <option value="low">Low (≤0.2)</option>
    </select>
    <input type="text" id="search" placeholder="Search region/coords..." oninput="filterCards()">
    <span style="font-size:12px;color:#888;" id="count">{len(entries_sorted)} shown</span>
</div>
<div class="g" id="grid">
{cards_html}
</div>
<script>
function toggleZoom(img) {{
    img.classList.toggle('zoomed');
}}
function sortCards() {{
    const grid = document.getElementById('grid');
    const cards = Array.from(grid.children);
    const sortVal = document.getElementById('sortSelect').value;
    cards.sort((a, b) => {{
        const sA = parseFloat(a.dataset.score);
        const sB = parseFloat(b.dataset.score);
        const rA = a.dataset.region;
        const rB = b.dataset.region;
        const dA = a.dataset.date;
        const dB = b.dataset.date;
        if (sortVal === 'score-desc') return sB - sA;
        if (sortVal === 'score-asc') return sA - sB;
        if (sortVal === 'region') return rA.localeCompare(rB);
        if (sortVal === 'date') return dA.localeCompare(dB);
    }});
    cards.forEach(c => grid.appendChild(c));
}}
function filterCards() {{
    const filter = document.getElementById('filterSelect').value;
    const search = document.getElementById('search').value.toLowerCase();
    const cards = document.querySelectorAll('.card');
    let shown = 0;
    cards.forEach(c => {{
        const score = parseFloat(c.dataset.score);
        const text = c.textContent.toLowerCase();
        const cls = score > 0.5 ? 'high' : (score > 0.2 ? 'mid' : 'low');
        const matchFilter = filter === 'all' || cls === filter;
        const matchSearch = !search || text.includes(search);
        c.style.display = (matchFilter && matchSearch) ? '' : 'none';
        if (matchFilter && matchSearch) shown++;
    }});
    document.getElementById('count').textContent = shown + ' shown';
}}
</script>
</body>
</html>"""

    review_path = candidates_dir / "review.html"
    review_path.write_text(html, encoding="utf-8")
    print(f"Review page generated: {review_path}")
    print(f"  {len(entries_sorted)} candidates")
    print(f"  High (>0.5): {high_count}")
    print(f"  Mid (0.2-0.5): {mid_count}")
    print(f"  Low (<=0.2): {low_count}")
    return str(review_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate review HTML from candidate manifest")
    parser.add_argument("--candidates-dir", default="data/candidates_svm_8pos")
    parser.add_argument("--port", type=int, default=8774, help="Port for serving (informational)")
    args = parser.parse_args(argv)

    candidates_dir = Path(args.candidates_dir).resolve()
    if not candidates_dir.exists():
        print(f"ERROR: Candidates directory not found: {candidates_dir}")
        return 1

    result = generate_review_page(candidates_dir)
    if not result:
        return 1

    print(f"\nTo serve:")
    print(f"  python -m http.server {args.port} --directory {candidates_dir.parent}")
    print(f"  http://localhost:{args.port}/candidates_svm_8pos/review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
