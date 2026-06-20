#!/usr/bin/env python3
"""Combine yearly BC coast SVM scans into multi-year confirmation groups.

This script reads multiple candidate manifests, groups candidates by rounded
latitude/longitude, copies or symlinks the matching thumbnails into a combined
output directory, and writes a static review page that shows each location's
observations year-by-year.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from html import escape


YEAR_RE = re.compile(r"(20\d{2})")


def _infer_year(source_dir: Path, entry: dict[str, Any]) -> int:
    date_text = str(entry.get("date") or "")
    if len(date_text) >= 4 and date_text[:4].isdigit():
        return int(date_text[:4])
    match = YEAR_RE.search(source_dir.name) or YEAR_RE.search(str(source_dir))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer year for {source_dir} entry {entry!r}")


def _round_coord(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def _group_key(lat: float, lon: float, decimals: int) -> tuple[float, float, str]:
    rounded_lat = _round_coord(lat, decimals)
    rounded_lon = _round_coord(lon, decimals)
    return rounded_lat, rounded_lon, f"{rounded_lat:.{decimals}f},{rounded_lon:.{decimals}f}"


def load_candidate_rows(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_dir in inputs:
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {source_dir}")
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{manifest_path} must contain a list")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "lat" not in entry or "lon" not in entry:
                continue
            year = _infer_year(source_dir, entry)
            rows.append(
                {
                    **entry,
                    "year": year,
                    "source_dir": str(source_dir),
                    "source_manifest": str(manifest_path),
                }
            )
    return rows


def _sort_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            int(item.get("year", 0)),
            -float(item.get("score", 0.0) or 0.0),
            str(item.get("date") or ""),
            str(item.get("thumbnail_path") or ""),
        ),
    )


def group_candidates(rows: list[dict[str, Any]], decimals: int = 2) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, str], dict[str, Any]] = {}

    for row in rows:
        lat = float(row["lat"])
        lon = float(row["lon"])
        rounded_lat, rounded_lon, key = _group_key(lat, lon, decimals)
        group = groups.setdefault(
            (rounded_lat, rounded_lon, key),
            {
                "group_key": key,
                "lat": lat,
                "lon": lon,
                "rounded_lat": rounded_lat,
                "rounded_lon": rounded_lon,
                "items": [],
            },
        )
        item = {
            **row,
            "thumb_src": None,
            "thumb_href": None,
        }
        group["items"].append(item)
        group["lat"] = float(row["lat"])
        group["lon"] = float(row["lon"])

    result: list[dict[str, Any]] = []
    for group in groups.values():
        items = _sort_items(group["items"])
        years = sorted({int(item["year"]) for item in items})
        group["items"] = items
        group["years"] = years
        group["n_years"] = len(years)
        group["multi_year_confirmed"] = len(years) >= 2
        group["high_confidence"] = len(years) >= 3
        group["best_score"] = max(float(item.get("score", 0.0) or 0.0) for item in items)
        group["count"] = len(items)
        result.append(group)

    return sorted(
        result,
        key=lambda item: (
            -int(item["n_years"]),
            -float(item["best_score"]),
            item["group_key"],
        ),
    )


def _safe_copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_output_assets(groups: list[dict[str, Any]], output_dir: Path) -> None:
    for group in groups:
        for item in group["items"]:
            source_dir = Path(item["source_dir"])
            thumb_name = str(item.get("thumbnail_path") or "")
            source_path = source_dir / thumb_name
            if not source_path.exists():
                raise FileNotFoundError(f"Missing thumbnail: {source_path}")
            year = int(item["year"])
            dest_path = output_dir / "thumbs" / str(year) / source_path.name
            _safe_copy_or_link(source_path, dest_path)
            item["thumb_href"] = str(Path("thumbs") / str(year) / source_path.name)
            item["thumb_src"] = str(source_path)


def build_review_html(groups: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards: list[str] = []
    for group in groups:
        item_cols = []
        for item in group["items"]:
            item_cols.append(
                f"""
                <div class="year-card" data-year="{int(item['year'])}">
                  <div class="year-head">{int(item['year'])}</div>
                  <img src="{escape(str(item['thumb_href']))}" alt="{escape(str(item.get('thumbnail_path') or ''))}">
                  <div class="meta">score {float(item.get('score', 0.0)):.4f}</div>
                  <div class="meta">{escape(str(item.get('date') or ''))} · {escape(str(item.get('scene_id') or ''))[:42]}</div>
                </div>
                """
            )
        label_key = escape(group["group_key"])
        status = "high-confidence" if group["high_confidence"] else ("multi-year" if group["multi_year_confirmed"] else "single-year")
        count = int(group.get("count", len(group["items"])))
        best_score = float(group.get("best_score", max(float(item.get("score", 0.0) or 0.0) for item in group["items"])))
        cards.append(
            f"""
            <article class="card {status}" data-group="{label_key}">
              <div class="card-top">
                <div>
                  <div class="location">{group['rounded_lat']:.2f}, {group['rounded_lon']:.2f}</div>
                  <div class="meta">actual {group['lat']:.4f}, {group['lon']:.4f} · {group['n_years']} year(s) · {count} candidate(s)</div>
                </div>
                <div class="badges">
                  <span class="badge {'green' if group['high_confidence'] else 'gold' if group['multi_year_confirmed'] else 'gray'}">{status}</span>
                  <span class="badge blue">best {best_score:.4f}</span>
                </div>
              </div>
              <div class="items">{''.join(item_cols)}</div>
              <div class="actions">
                <button onclick="setLabel('{label_key}', 'accept')">Accept</button>
                <button onclick="setLabel('{label_key}', 'review')">Review</button>
                <button onclick="setLabel('{label_key}', 'reject')">Reject</button>
                <span class="label" id="label-{label_key}"></span>
              </div>
            </article>
            """
        )

    summary_html = f"""
      <div class="summary-grid">
        <div class="summary-box"><div class="value">{summary['group_count']}</div><div class="label">groups</div></div>
        <div class="summary-box"><div class="value">{summary['multi_year_count']}</div><div class="label">2+ years</div></div>
        <div class="summary-box"><div class="value">{summary['high_confidence_count']}</div><div class="label">3 years</div></div>
      </div>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-year confirmed candidates</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    header {{ padding: 20px 24px; background: linear-gradient(135deg, #111827, #0f172a); border-bottom: 1px solid #1f2937; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 20px 24px 40px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .summary-box {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 14px 16px; }}
    .summary-box .value {{ font-size: 28px; font-weight: 800; color: #fff; }}
    .summary-box .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }}
    .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 16px; margin-bottom: 16px; }}
    .card.multi-year {{ border-left: 6px solid #f59e0b; }}
    .card.high-confidence {{ border-left: 6px solid #22c55e; }}
    .card-top {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; align-items: start; }}
    .location {{ font-size: 18px; font-weight: 800; color: #fff; }}
    .meta {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .badge {{ border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .badge.green {{ background: rgba(34,197,94,.15); color: #86efac; }}
    .badge.gold {{ background: rgba(245,158,11,.15); color: #fcd34d; }}
    .badge.blue {{ background: rgba(59,130,246,.15); color: #93c5fd; }}
    .badge.gray {{ background: rgba(148,163,184,.15); color: #cbd5e1; }}
    .items {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }}
    .year-card {{ background: #0b1220; border: 1px solid #1f2937; border-radius: 12px; padding: 10px; }}
    .year-head {{ font-size: 14px; font-weight: 800; margin-bottom: 8px; color: #fff; }}
    .year-card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 10px; display: block; background: #020617; }}
    .actions {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 14px; }}
    button {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 8px 12px; cursor: pointer; }}
    button:hover {{ background: #334155; }}
    .label {{ margin-left: auto; font-size: 12px; color: #93c5fd; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 12px; }}
    .toolbar input {{ min-width: 280px; background: #0b1220; color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 8px 12px; }}
  </style>
</head>
<body>
  <header>
    <h1 style="margin:0;font-size:22px;">Multi-year confirmed candidates</h1>
    <div style="color:#94a3b8;margin-top:6px;">Grouped by rounded 0.01° coordinates · year comparison side-by-side · labels stored in localStorage</div>
    <div class="toolbar">
      <input id="search" placeholder="Search coords or year..." oninput="filterCards()">
      <span id="count">{len(groups)} groups</span>
    </div>
  </header>
  <main>
    {summary_html}
    <div id="cards">{''.join(cards)}</div>
  </main>
  <script>
    function setLabel(key, value) {{
      localStorage.setItem('multiyear-label:' + key, value);
      const el = document.getElementById('label-' + key);
      if (el) el.textContent = 'label: ' + value;
    }}
    function filterCards() {{
      const q = document.getElementById('search').value.toLowerCase();
      let shown = 0;
      document.querySelectorAll('.card').forEach(card => {{
        const text = card.textContent.toLowerCase();
        const ok = !q || text.includes(q);
        card.style.display = ok ? '' : 'none';
        if (ok) shown++;
      }});
      document.getElementById('count').textContent = shown + ' groups';
    }}
    document.querySelectorAll('.card').forEach(card => {{
      const key = card.dataset.group;
      const value = localStorage.getItem('multiyear-label:' + key);
      if (value) {{
        const el = document.getElementById('label-' + key);
        if (el) el.textContent = 'label: ' + value;
      }}
    }});
  </script>
</body>
</html>"""


def write_output(groups: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_assets(groups, output_dir)
    manifest = []
    for group in groups:
        manifest.append(
            {
                "group_key": group["group_key"],
                "lat": group["lat"],
                "lon": group["lon"],
                "rounded_lat": group["rounded_lat"],
                "rounded_lon": group["rounded_lon"],
                "years": group["years"],
                "n_years": group["n_years"],
                "multi_year_confirmed": group["multi_year_confirmed"],
                "high_confidence": group["high_confidence"],
                "best_score": group["best_score"],
                "count": group["count"],
                "items": [
                    {
                        **item,
                        "thumb_href": item["thumb_href"],
                    }
                    for item in group["items"]
                ],
            }
        )

    summary = {
        "group_count": len(manifest),
        "multi_year_count": sum(1 for group in manifest if group["multi_year_confirmed"]),
        "high_confidence_count": sum(1 for group in manifest if group["high_confidence"]),
        "year_counts": dict(Counter(year for group in manifest for year in group["years"])),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "review.html").write_text(build_review_html(manifest, summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Yearly candidate output directories")
    parser.add_argument("--output", type=Path, required=True, help="Combined output directory")
    parser.add_argument("--decimals", type=int, default=2, help="Coordinate rounding decimals")
    parser.add_argument("--port", type=int, default=8783, help="Suggested review port")
    args = parser.parse_args(argv)

    rows = load_candidate_rows([path.resolve() for path in args.inputs])
    if not rows:
        print("ERROR: no candidates loaded")
        return 1

    groups = group_candidates(rows, decimals=args.decimals)
    summary = write_output(groups, args.output.resolve())

    print(f"Loaded {len(rows)} candidate rows from {len(args.inputs)} input dirs")
    print(f"Combined groups: {summary['group_count']}")
    print(f"Multi-year confirmed: {summary['multi_year_count']}")
    print(f"High-confidence: {summary['high_confidence_count']}")
    print(f"Review page: {args.output.resolve() / 'review.html'}")
    print(f"Serve with: python -m http.server {args.port} --directory {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
