#!/usr/bin/env python3
"""Render before/after herring spawn sequences as a review page."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELTA_DIR = PROJECT_ROOT / "data" / "delta_pairs" / "pairs"
NEW_DIR = PROJECT_ROOT / "data" / "baseline_pairs"
OUTPUT = NEW_DIR / "review.html"

# Parse existing delta pairs from filenames
delta_pairs = {}
for p in sorted(DELTA_DIR.glob("*_pos.png")):
    name = p.name.replace("delta_", "").replace(".png", "")
    parts = name.rsplit("_", 1)
    label = parts[-1]  # pos
    rest = parts[0]
    for typ in ("baseline", "spawn"):
        if f"_{typ}_" in rest:
            idx = rest.index(f"_{typ}_")
            coords = rest[:idx].replace("_", ", ")
            date_str = rest[idx + len(f"_{typ}_"):]
            key = f"{coords}_{label}"
            if key not in delta_pairs:
                delta_pairs[key] = {}
            delta_pairs[key][typ] = {
                "path": f"../delta_pairs/pairs/{p.name}",
                "date": date_str,
            }
            break

# Parse new baselines from manifest
new_pairs = {}
with open(NEW_DIR / "manifest.json") as f:
    manifest = json.load(f)
for m in manifest:
    region = m["region"]
    new_pairs[region] = {
        "baseline": {
            "path": m["baseline_file"],
            "date": m["baseline_date"],
        },
        "spawn": {
            "path": m["spawn_file"],
            "date": m["spawn_date"],
        },
    }

# --- Build HTML rows ---
rows = []

for region, pair in sorted(new_pairs.items()):
    b = pair["baseline"]
    s = pair["spawn"]
    rows.append(f"""
    <tr class="pos">
      <td class="label pos">POS</td>
      <td><strong>{region}</strong></td>
      <td>{b['date']}</td>
      <td><img src="{b['path']}" loading="lazy"></td>
      <td class="arrow">→</td>
      <td>{s['date']}</td>
      <td><img src="{s['path']}" loading="lazy"></td>
    </tr>""")

for key, pair in sorted(delta_pairs.items()):
    if "baseline" not in pair or "spawn" not in pair:
        continue
    b = pair["baseline"]
    s = pair["spawn"]
    coords = key.rsplit("_", 1)[0]
    rows.append(f"""
    <tr class="pos">
      <td class="label pos">POS</td>
      <td><strong>SoG {coords}</strong></td>
      <td>{b['date']}</td>
      <td><img src="{b['path']}" loading="lazy"></td>
      <td class="arrow">→</td>
      <td>{s['date']}</td>
      <td><img src="{s['path']}" loading="lazy"></td>
    </tr>""")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Herring Spawn Baseline-Spawn Sequences</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ font-size: 18px; margin-bottom: 8px; }}
  .summary {{ font-size: 13px; color: #8b949e; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; padding: 8px 10px; font-size: 12px; color: #8b949e; border-bottom: 1px solid #30363d; }}
  td {{ padding: 10px 8px; border-bottom: 1px solid #21262d; vertical-align: top; font-size: 12px; }}
  td img {{ width: 224px; height: 224px; border-radius: 4px; border: 1px solid #30363d; object-fit: cover; }}
  tr:hover {{ background: #161b22; }}
  .label {{ font-weight: bold; font-size: 11px; }}
  .label.pos {{ color: #2ea043; }}
  .arrow {{ font-size: 20px; text-align: center; color: #58a6ff; }}
</style>
</head>
<body>
<h1>Herring Spawn — Baseline → Spawn (before/after pairs)</h1>
<div class="summary">
  {len(new_pairs)} new GEE downloads + {len(delta_pairs)} existing delta_pairs = {len(rows)} positive pairs
  | All golden-set positives shown
</div>
<table>
<thead><tr>
  <th></th><th>Location</th><th>Baseline Date</th><th>Baseline</th><th></th><th>Spawn Date</th><th>Spawn</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""

OUTPUT.write_text(html)
print(f"Wrote {OUTPUT} ({len(rows)} pairs)")
