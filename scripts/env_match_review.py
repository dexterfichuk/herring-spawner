#!/usr/bin/env python3
"""
Environmental Matching Review App.

Serves 426 review rows — S2 thumbnails from disk, Landsat thumbnails
on-demand from Google Earth Engine (with local caching) — plus a labeling
UI with filters, sorting, keyboard navigation, and environmental info.

Usage:
    source .venv/bin/activate
    python scripts/env_match_review.py
    # Open http://localhost:8785

Keyboard:
    →  Accept + next
    ←  Reject + next
    ↓  Skip (next)
    ↑  Previous
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template_string, request, send_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENV_MATCH_DIR = DATA_DIR / "environmental_matching" / "s2_landsat"
MANIFEST = ENV_MATCH_DIR / "review_data.json"
LABELS_FILE = ENV_MATCH_DIR / "labels.json"
LANDSAT_CACHE = DATA_DIR / "landsat_thumbnails"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(MANIFEST) as f:
    ALL_ROWS: list[dict[str, Any]] = json.load(f)

# Build lookup: filename → absolute path for S2 thumbnails that exist on disk
S2_PATH_MAP: dict[str, Path] = {}
for row in ALL_ROWS:
    tp = row.get("thumbnail_path", "")
    fn = row.get("filename", "")
    if tp and fn:
        abspath = PROJECT_ROOT / tp
        if abspath.exists():
            S2_PATH_MAP[fn] = abspath

# ---------------------------------------------------------------------------
# Landsat filename parser
# ---------------------------------------------------------------------------
# Format: L{8|9}_{region}_{YYYY-MM-DD}_score{X}_cloud{X}_{lat}_{lon}_{YYYYMMDD}.npy
# Region can contain hyphens, e.g. "barkley-sound", "milbanke-sound".
_LANDSAT_PATTERN = re.compile(
    r"^L([89])_(.+)_(\d{4}-\d{2}-\d{2})_score([\d.eE+-]+)_cloud(\d+)_"
    r"([\d.-]+)_([\d.-]+)_(\d{8})\.npy$"
)

# Older-format fallback: L{8|9}_{region}_{date}_score{X}_cloud{X}_{lat}_{lon}_{acqdate}.npy
# where region has no hyphens and uses underscore separators within the prefix.
_LANDSAT_FALLBACK = re.compile(
    r"^L([89])_([A-Za-z]+(?:_[A-Za-z]+)*)_(\d{4}-\d{2}-\d{2})_score([\d.eE+-]+)_cloud(\d+)_"
    r"([\d.-]+)_([\d.-]+)_(\d{8})\.npy$"
)

LANDSAT_COLLECTIONS: dict[str, str] = {
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}


def parse_landsat_filename(filename: str) -> dict[str, Any] | None:
    """Parse Landsat embedding filename into metadata dict."""
    m = _LANDSAT_PATTERN.match(filename)
    if not m:
        m = _LANDSAT_FALLBACK.match(filename)
    if not m:
        return None
    return {
        "satellite": f"L{m.group(1)}",
        "region": m.group(2),
        "date": m.group(3),
        "score": float(m.group(4)),
        "cloud": int(m.group(5)),
        "lat": float(m.group(6)),
        "lon": float(m.group(7)),
        "acq_date": m.group(8),
    }


# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------
try:
    import ee  # type: ignore[import-untyped]

    ee.Initialize(project="redd-fish")
    _GEE_OK = True
except Exception as exc:
    print(f"[WARN] GEE not available: {exc}", file=sys.stderr)
    _GEE_OK = False


# ---------------------------------------------------------------------------
# Landsat thumbnail cache
# ---------------------------------------------------------------------------
LANDSAT_CACHE.mkdir(parents=True, exist_ok=True)


def _landsat_cache_path(filename: str) -> Path:
    """Map .npy → .png in the landsat cache directory."""
    return LANDSAT_CACHE / filename.replace(".npy", ".png")


def fetch_landsat_thumbnail(
    satellite: str, lat: float, lon: float, date_str: str
) -> bytes | None:
    """Query GEE for the best Landsat scene at (lon,lat) on *date_str*, download 256×256 RGB PNG."""
    if not _GEE_OK:
        return None
    collection_id = LANDSAT_COLLECTIONS.get(satellite)
    if not collection_id:
        return None

    try:
        point = ee.Geometry.Point(lon, lat)
        collection = ee.ImageCollection(collection_id)

        # Date range: the full day in UTC
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

        scenes = (
            collection.filterBounds(point)
            .filterDate(date_str, end)
            .sort("CLOUD_COVER")
        )

        scene_ids: list[str] = scenes.aggregate_array("system:index").getInfo()
        if not scene_ids:
            return None

        scene_id = scene_ids[0]  # lowest cloud cover
        scene_img = ee.Image(f"{collection_id}/{scene_id}")
        rgb = scene_img.select(["SR_B4", "SR_B3", "SR_B2"])
        region = ee.Geometry.Point(lon, lat).buffer(1280).bounds()

        url = rgb.getThumbURL({
            "min": 0,
            "max": 3000,
            "region": region,
            "dimensions": 256,
            "format": "png",
        })

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    except Exception as exc:
        print(f"[WARN] Landsat thumb fetch failed ({satellite} {date_str}): {exc}",
              file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Labels persistence
# ---------------------------------------------------------------------------
def _load_labels() -> dict[str, int]:
    """Load user labels. Returns {filename: 0|1}."""
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            return json.load(f)
    return {}


def _save_labels(labels: dict[str, int]) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Attach labels to app config so routes can share
app.config["user_labels"] = _load_labels()
app.config["all_rows"] = ALL_ROWS
app.config["s2_path_map"] = S2_PATH_MAP

# ---------------------------------------------------------------------------
# HTML template (embedded for self-contained script)
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Env Match Review</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f0f1a;color:#ddd;font-family:system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
/* Toolbar */
#toolbar{display:flex;gap:8px;padding:10px 16px;background:#1a1a2e;align-items:center;flex-shrink:0;flex-wrap:wrap}
#toolbar button,#toolbar select{padding:5px 12px;border:1px solid #333;border-radius:5px;background:#16213e;color:#ddd;cursor:pointer;font-size:13px}
#toolbar button:hover{background:#1f3a6e}
#toolbar button:disabled{opacity:.4;cursor:default}
#progress{font-size:13px;color:#888;white-space:nowrap}
#nav{display:flex;gap:4px;align-items:center}
#nav input{width:48px;text-align:center;padding:4px;border-radius:4px;border:1px solid #333;background:#16213e;color:#ddd;font-size:13px}
.filter-group{display:flex;gap:6px;align-items:center;margin-left:auto}
.filter-group label{font-size:12px;color:#888}
#status-badge{padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600}
#status-badge.unlabeled{background:#444;color:#aaa}
#status-badge.accepted{background:#1b5e20;color:#a5d6a7}
#status-badge.rejected{background:#b71c1c;color:#ef9a9a}
/* Main layout */
#main{flex:1;display:flex;overflow:hidden}
#image-panel{flex:3;display:flex;align-items:center;justify-content:center;background:#0a0a14;position:relative;min-width:0}
#image-panel img{max-width:100%;max-height:100%;object-fit:contain}
.spinner{border:3px solid #333;border-top:3px solid #4a90d9;border-radius:50%;width:36px;height:36px;animation:spin .8s linear infinite;position:absolute}
@keyframes spin{to{transform:rotate(360deg)}}
.placeholder-text{color:#555;font-size:14px;text-align:center;padding:20px}
/* Info panel */
#info-panel{flex:1;padding:14px;overflow-y:auto;background:#14142a;border-left:1px solid #242440;min-width:240px;max-width:360px}
#info-panel h3{font-size:14px;font-weight:600;margin-bottom:10px;color:#eee;word-break:break-all}
.metric{font-size:12px;margin:4px 0;color:#aaa}
.metric .label{color:#666}
.metric .value{color:#eef;font-weight:500}
.metric .value.pos{color:#66bb6a}
.metric .value.neg{color:#ef5350}
.section-title{font-size:11px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.5px;margin:12px 0 4px;border-bottom:1px solid #1e1e38;padding-bottom:2px}
/* Actions */
#actions{display:flex;gap:10px;padding:12px 16px;background:#1a1a2e;flex-shrink:0;justify-content:center;border-top:1px solid #242440}
#actions button{font-size:15px;padding:8px 24px;border:none;border-radius:6px;cursor:pointer;font-weight:600;transition:transform .1s}
#actions button:active{transform:scale(.96)}
#btn-reject{background:#7f1d1d;color:#fff}
#btn-reject:hover{background:#a12727}
#btn-skip{background:#333;color:#ccc}
#btn-skip:hover{background:#444}
#btn-accept{background:#1b5e20;color:#fff}
#btn-accept:hover{background:#2a7a30}
/* Export / reset */
#export-btn,#reset-btn{font-size:11px!important;padding:4px 10px!important;background:transparent!important;border:1px solid #444!important;border-radius:4px!important}
#export-btn{color:#66bb6a!important}
#reset-btn{color:#ef5350!important}
</style>
</head>
<body>

<div id="toolbar">
  <span id="progress">0 / 0</span>
  <div id="nav">
    <button id="btn-first" title="First">⏮</button>
    <button id="btn-prev" title="Previous">◀</button>
    <input id="idx-input" type="number" min="1" value="1">
    <span style="color:#666">/ <span id="total-count">0</span></span>
    <button id="btn-next" title="Next">▶</button>
    <button id="btn-last" title="Last">⏭</button>
  </div>

  <div class="filter-group">
    <label>Filter</label>
    <select id="filter-select">
      <option value="all">All</option>
      <option value="local">S2 Only</option>
      <option value="landsat_embedding">Landsat Only</option>
      <option value="pos">Positive (label=1)</option>
      <option value="neg">Negative (label=0)</option>
      <option value="unlabeled">Unlabeled</option>
      <option value="accepted">Accepted</option>
      <option value="rejected">Rejected</option>
    </select>
    <label>Sort</label>
    <select id="sort-select">
      <option value="score-desc">Score ↓</option>
      <option value="score-asc">Score ↑</option>
      <option value="date">Date</option>
      <option value="location">Location</option>
    </select>
    <button id="export-btn">Export Labels</button>
    <button id="reset-btn">Reset</button>
  </div>
</div>

<div id="main">
  <div id="image-panel">
    <div class="spinner" id="spinner" style="display:none"></div>
    <img id="main-image" alt="Thumbnail" style="display:none">
    <div class="placeholder-text" id="placeholder">No row selected</div>
  </div>
  <div id="info-panel">
    <div id="info-content"></div>
  </div>
</div>

<div id="actions">
  <button id="btn-reject">✗ Reject (←)</button>
  <button id="btn-skip">Skip (↓)</button>
  <button id="btn-accept">✓ Accept (→)</button>
</div>

<script>
// ── Globals ──────────────────────────────────────────────────────────
let allRows = [];
let userLabels = {};
let filtered = [];
let current = 0;

// ── Init ─────────────────────────────────────────────────────────────
async function init() {
  const r = await fetch('/api/data');
  const payload = await r.json();
  allRows = payload.rows;
  userLabels = payload.labels;
  document.getElementById('total-count').textContent = allRows.length;
  applySort();
  applyFilter();
}

// ── State helpers ────────────────────────────────────────────────────
function getFilteredIdx(rawIdx) {
  // rawIdx is index into allRows. Return index into filtered, or -1.
  return filtered.indexOf(allRows[rawIdx]);
}

function getLabel(fn) {
  if (fn in userLabels) return userLabels[fn]; // 0 or 1
  return -1; // unlabeled
}

function labelText(l) {
  if (l === 1) return 'Accepted';
  if (l === 0) return 'Rejected';
  return 'Unlabeled';
}

function labelClass(l) {
  if (l === 1) return 'accepted';
  if (l === 0) return 'rejected';
  return 'unlabeled';
}

// ── Sorting & Filtering ──────────────────────────────────────────────
function applySort() {
  const s = document.getElementById('sort-select').value;
  const rows = allRows;
  if (s === 'score-desc') rows.sort((a, b) => (b.score||0) - (a.score||0));
  else if (s === 'score-asc') rows.sort((a, b) => (a.score||0) - (b.score||0));
  else if (s === 'date') rows.sort((a, b) => (a.date||'').localeCompare(b.date||''));
  else if (s === 'location') rows.sort((a, b) => (a.location_key||'').localeCompare(b.location_key||''));
  applyFilter();
}

function applyFilter() {
  const f = document.getElementById('filter-select').value;
  const rows = allRows;
  if (f === 'all') filtered = [...rows];
  else if (f === 'local') filtered = rows.filter(r => r.source === 'local');
  else if (f === 'landsat_embedding') filtered = rows.filter(r => r.source === 'landsat_embedding');
  else if (f === 'pos') filtered = rows.filter(r => r.label === 1);
  else if (f === 'neg') filtered = rows.filter(r => r.label === 0);
  else if (f === 'unlabeled') filtered = rows.filter(r => getLabel(r.filename) === -1);
  else if (f === 'accepted') filtered = rows.filter(r => getLabel(r.filename) === 1);
  else if (f === 'rejected') filtered = rows.filter(r => getLabel(r.filename) === 0);
  document.getElementById('total-count').textContent = filtered.length;
  if (filtered.length === 0) { showEmpty(); return; }
  gotoIdx(0);
}

// ── Navigation ───────────────────────────────────────────────────────
function gotoIdx(idx) {
  if (filtered.length === 0) { showEmpty(); return; }
  current = Math.max(0, Math.min(idx, filtered.length - 1));
  document.getElementById('idx-input').value = current + 1;
  document.getElementById('progress').textContent = `${current + 1} / ${filtered.length}`;
  renderRow(filtered[current]);
}

function goPrev() { if (current > 0) gotoIdx(current - 1); }
function goNext() { if (current < filtered.length - 1) gotoIdx(current + 1); }
function goFirst() { gotoIdx(0); }
function goLast() { gotoIdx(filtered.length - 1); }

function showEmpty() {
  document.getElementById('progress').textContent = '0 / 0';
  document.getElementById('main-image').style.display = 'none';
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('placeholder').style.display = 'block';
  document.getElementById('placeholder').textContent = 'No rows match filter';
  document.getElementById('info-content').innerHTML = '';
  document.getElementById('status-badge') && document.getElementById('status-badge').remove();
}

// ── Render ────────────────────────────────────────────────────────────
function renderRow(row) {
  const isLandsat = row.source === 'landsat_embedding';
  const img = document.getElementById('main-image');
  const spinner = document.getElementById('spinner');
  const placeholder = document.getElementById('placeholder');

  // Show loading for Landsat (may need GEE fetch)
  if (isLandsat) {
    spinner.style.display = 'block';
    img.style.display = 'none';
    placeholder.style.display = 'none';
    img.src = '/thumb/landsat/' + encodeURIComponent(row.filename);
  } else {
    spinner.style.display = 'none';
    img.style.display = 'block';
    placeholder.style.display = 'none';
    img.src = '/thumb/local/' + encodeURIComponent(row.filename);
  }

  // Update info panel
  const lbl = getLabel(row.filename);
  const html = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <span id="status-badge" class="${labelClass(lbl)}">${labelText(lbl)}</span>
      <span style="font-size:11px;color:#666">#${current+1} of ${filtered.length}</span>
    </div>
    <div class="section-title">Scene</div>
    <div class="metric"><span class="label">Source:</span> <span class="value">${row.source === 'local' ? 'Sentinel-2' : 'Landsat'}</span></div>
    <div class="metric"><span class="label">Date:</span> <span class="value">${row.date || '—'}</span></div>
    <div class="metric"><span class="label">Region:</span> <span class="value">${row.region_key || '—'}</span></div>
    <div class="metric"><span class="label">Filename:</span> <span class="value" style="font-size:11px">${row.filename}</span></div>

    <div class="section-title">Location</div>
    <div class="metric"><span class="label">Lat:</span> <span class="value">${row.location_key ? row.location_key.replace('coords:','').split('_')[0] : '—'}</span></div>
    <div class="metric"><span class="label">Lon:</span> <span class="value">${row.location_key ? row.location_key.replace('coords:','').split('_')[1] : '—'}</span></div>

    <div class="section-title">Score</div>
    <div class="metric"><span class="label">Env. Match Score:</span> <span class="value ${row.label === 1 ? 'pos' : row.label === 0 ? 'neg' : ''}">${typeof row.score === 'number' ? row.score.toFixed(4) : row.score}</span></div>
    <div class="metric"><span class="label">Orig. Label:</span> <span class="value ${row.label === 1 ? 'pos' : row.label === 0 ? 'neg' : ''}">${row.label === 1 ? 'Positive (1)' : row.label === 0 ? 'Negative (0)' : '—'}</span></div>
    <div class="metric"><span class="label">Value Dist.:</span> <span class="value">${row.value_distance ? parseFloat(row.value_distance).toFixed(4) : '—'}</span></div>

    <div class="section-title">Environmental</div>
    <div class="metric"><span class="label">Sun Elevation:</span> <span class="value">${typeof row.sun_elevation === 'number' ? row.sun_elevation.toFixed(2) + '°' : '—'}</span></div>
    <div class="metric"><span class="label">Sun Azimuth:</span> <span class="value">${typeof row.sun_azimuth === 'number' ? row.sun_azimuth.toFixed(2) + '°' : '—'}</span></div>
    <div class="metric"><span class="label">Tide Height:</span> <span class="value">${typeof row.tide_height === 'number' ? row.tide_height.toFixed(2) + 'm' : '—'}</span></div>
    <div class="metric"><span class="label">Tide Source:</span> <span class="value">${row.tide_source || '—'}</span></div>

    <div class="section-title">Baseline</div>
    <div class="metric"><span class="label">Scope:</span> <span class="value">${row.baseline_scope || '—'}</span></div>
    <div class="metric"><span class="label">Group Size:</span> <span class="value">${row.group_size || '—'}</span></div>
    <div class="metric"><span class="label">Mean Dist.:</span> <span class="value">${row.baseline_mean_distance ? parseFloat(row.baseline_mean_distance).toFixed(4) : '—'}</span></div>
    <div class="metric"><span class="label">Std Dist.:</span> <span class="value">${row.baseline_std_distance ? parseFloat(row.baseline_std_distance).toFixed(4) : '—'}</span></div>
    <div class="metric"><span class="label">Count:</span> <span class="value">${row.baseline_count || '—'}</span></div>
  `;
  document.getElementById('info-content').innerHTML = html;
}

// Image load / error handlers
document.getElementById('main-image').addEventListener('load', function() {
  document.getElementById('spinner').style.display = 'none';
  this.style.display = 'block';
  document.getElementById('placeholder').style.display = 'none';
});
document.getElementById('main-image').addEventListener('error', function() {
  document.getElementById('spinner').style.display = 'none';
  this.style.display = 'none';
  document.getElementById('placeholder').style.display = 'block';
  document.getElementById('placeholder').textContent = 'Thumbnail not available';
});

// ── Labeling ─────────────────────────────────────────────────────────
async function setLabel(value) {
  if (filtered.length === 0) return;
  const row = filtered[current];
  userLabels[row.filename] = value;
  try {
    await fetch('/api/label', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: row.filename, label: value}),
    });
  } catch(e) { console.error('Label save failed', e); }
  // Re-render current row to show new label status
  renderRow(row);
  goNext();
}

// ── Keyboard ─────────────────────────────────────────────────────────
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT') return;  // allow typing in index box
  switch (e.key) {
    case 'ArrowRight': e.preventDefault(); setLabel(1); break;
    case 'ArrowLeft':  e.preventDefault(); setLabel(0); break;
    case 'ArrowDown':  e.preventDefault(); goNext(); break;
    case 'ArrowUp':    e.preventDefault(); goPrev(); break;
  }
});

// ── Export / Reset ────────────────────────────────────────────────────
document.getElementById('export-btn').addEventListener('click', function() {
  const blob = new Blob([JSON.stringify(userLabels, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'env_match_labels.json';
  a.click();
  URL.revokeObjectURL(a.href);
});

document.getElementById('reset-btn').addEventListener('click', async function() {
  if (!confirm('Reset all user labels? This cannot be undone.')) return;
  userLabels = {};
  await fetch('/api/reset', {method: 'POST'});
  applyFilter();
});

// ── Event bindings ───────────────────────────────────────────────────
document.getElementById('btn-first').addEventListener('click', goFirst);
document.getElementById('btn-prev').addEventListener('click', goPrev);
document.getElementById('btn-next').addEventListener('click', goNext);
document.getElementById('btn-last').addEventListener('click', goLast);
document.getElementById('btn-reject').addEventListener('click', () => setLabel(0));
document.getElementById('btn-skip').addEventListener('click', goNext);
document.getElementById('btn-accept').addEventListener('click', () => setLabel(1));
document.getElementById('filter-select').addEventListener('change', applyFilter);
document.getElementById('sort-select').addEventListener('change', applySort);
document.getElementById('idx-input').addEventListener('change', function() {
  const v = parseInt(this.value) - 1;
  if (!isNaN(v) && v >= 0 && v < filtered.length) gotoIdx(v);
});

// ── Start ─────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/data")
def api_data():
    """Return all rows with current user labels merged in."""
    labels = app.config["user_labels"]
    return jsonify({"rows": app.config["all_rows"], "labels": labels})


@app.route("/api/labels")
def api_labels():
    return jsonify(app.config["user_labels"])


@app.route("/api/label", methods=["POST"])
def api_label():
    body = request.get_json()
    fn = body.get("filename", "")
    label_val = body.get("label")
    if not fn or label_val not in (0, 1):
        return jsonify({"error": "invalid payload"}), 400

    labels = app.config["user_labels"]
    labels[fn] = label_val
    _save_labels(labels)
    app.config["user_labels"] = labels
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    app.config["user_labels"] = {}
    if LABELS_FILE.exists():
        LABELS_FILE.unlink()
    return jsonify({"ok": True})


@app.route("/thumb/local/<path:filename>")
def serve_s2_thumbnail(filename: str):
    """Serve S2 thumbnail from the positive/negative sample directories."""
    path = app.config["s2_path_map"].get(filename)
    if path and path.exists():
        return send_file(str(path), mimetype="image/png")
    # Fallback: try to find it by scanning thumbnail_path fields
    for row in app.config["all_rows"]:
        if row.get("filename") == filename and row.get("thumbnail_path"):
            fallback = PROJECT_ROOT / row["thumbnail_path"]
            if fallback.exists():
                return send_file(str(fallback), mimetype="image/png")
    return "Thumbnail not found", 404


@app.route("/thumb/landsat/<path:filename>")
def serve_landsat_thumbnail(filename: str):
    """Fetch Landsat thumbnail from GEE (or cache) on-demand."""
    cache_path = _landsat_cache_path(filename)

    # Return cached if exists
    if cache_path.exists():
        return send_file(str(cache_path), mimetype="image/png")

    # Parse filename
    meta = parse_landsat_filename(filename)
    if meta is None:
        return "Could not parse Landsat filename", 400

    # Fetch from GEE
    png_bytes = fetch_landsat_thumbnail(
        meta["satellite"], meta["lat"], meta["lon"], meta["date"]
    )
    if png_bytes is None:
        return "Landsat thumbnail not available", 404

    # Cache to disk
    cache_path.write_bytes(png_bytes)
    return send_file(str(cache_path), mimetype="image/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n_local = sum(1 for r in ALL_ROWS if r.get("source") == "local")
    n_landsat = sum(1 for r in ALL_ROWS if r.get("source") == "landsat_embedding")
    n_labels = len(app.config["user_labels"])
    print(f"  Loaded: {len(ALL_ROWS)} rows ({n_local} S2, {n_landsat} Landsat)")
    print(f"  Labels: {n_labels} existing")
    print(f"  Landsat cache: {len(list(LANDSAT_CACHE.glob('*.png')))} thumbnails")
    print(f"  GEE: {'ready' if _GEE_OK else 'NOT AVAILABLE'}")
    print(f"  URL: http://localhost:8785")
    app.run(host="0.0.0.0", port=8785, debug=False)
