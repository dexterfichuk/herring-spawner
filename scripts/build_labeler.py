#!/usr/bin/env python3
"""Build the v2 herring spawn labeling interface.

This script composes a single static HTML review tool that can be opened via
file:// or served over HTTP. It combines the DFO ingressed review set with the
candidate delta review set, persists labels to localStorage, and exports a JSON
label manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INGRESSED_MANIFEST = REPO_ROOT / "data" / "ingressed" / "manifest.json"
INGRESSED_THUMBS = REPO_ROOT / "data" / "ingressed" / "thumbnails"
CANDIDATES_MANIFEST = REPO_ROOT / "data" / "candidates_v2" / "manifest.json"
CANDIDATES_THUMBS = REPO_ROOT / "data" / "candidates_v2"
CANDIDATES_OFFSEASON = CANDIDATES_THUMBS / "offseason"
OUTPUT_HTML = REPO_ROOT / "data" / "review" / "labeler_v2.html"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def lat_lon_key(lat: float, lon: float) -> str:
    lat_str = str(lat).replace(".", "_")
    lon_str = str(abs(lon)).replace(".", "_")
    return f"{lat_str}__{lon_str}"


def build_offseason_index() -> dict[str, str]:
    if not CANDIDATES_OFFSEASON.exists():
        return {}

    index: dict[str, str] = {}
    for path in sorted(CANDIDATES_OFFSEASON.glob("*.png")):
        stem = path.stem
        prefix = stem.split("_off_")[0]
        index[prefix] = f"offseason/{path.name}"
    return index


def load_ingressed_items() -> list[dict[str, object]]:
    if not INGRESSED_MANIFEST.exists():
        return []

    payload = json.loads(INGRESSED_MANIFEST.read_text())
    items: list[dict[str, object]] = []
    for row in payload.get("thumbnails", []):
        thumb = row.get("thumbnail_path")
        items.append(
            {
                "id": row["event_id"],
                "dataset": "dfo",
                "kind": "single",
                "title": row.get("location") or row["event_id"],
                "subtitle": row.get("start_date") or "",
                "meta": {
                    "scene_date": row.get("scene_date"),
                    "cloud": row.get("cloud"),
                    "days_from_spawn": row.get("days_from_spawn"),
                    "spawn_length_m": row.get("spawn_length_m"),
                    "spawn_width_m": row.get("spawn_width_m"),
                },
                "spawn_thumb": f"../ingressed/thumbnails/{thumb}" if thumb else None,
                "off_thumb": None,
                "note": "DFO reviewed scene",
            }
        )
    return items


def load_candidate_items() -> list[dict[str, object]]:
    if not CANDIDATES_MANIFEST.exists():
        return []

    off_index = build_offseason_index()
    payload = json.loads(CANDIDATES_MANIFEST.read_text())
    items: list[dict[str, object]] = []
    for row in payload:
        region = str(row["region"])
        lat = row["lat"]
        lon = row["lon"]
        date = row.get("date") or ""
        spawn_thumb = row.get("thumbnail_path")
        off_key = f"{region.replace('-', '_')}_{lat_lon_key(lat, lon)}"
        off_thumb = off_index.get(off_key)
        items.append(
            {
                "id": f"cand:{region}:{lat}:{lon}:{date}",
                "dataset": "candidate",
                "kind": "compare",
                "title": region,
                "subtitle": date,
                "meta": {
                    "scene_id": row.get("scene_id"),
                    "cloud": row.get("cloud"),
                    "score": row.get("score"),
                    "lat": lat,
                    "lon": lon,
                },
                "spawn_thumb": f"../candidates_v2/{spawn_thumb}" if spawn_thumb else None,
                "off_thumb": f"../candidates_v2/{off_thumb}" if off_thumb else None,
                "note": "Spawn-season vs off-season",
            }
        )
    return items


def sort_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(item: dict[str, object]) -> tuple[int, float, str]:
        dataset_rank = 0 if item["dataset"] == "candidate" else 1
        meta = item.get("meta", {})
        score = float(meta.get("score") or 0.0)
        subtitle = str(item.get("subtitle") or "")
        # Candidates first by score desc, DFO by date desc.
        if item["dataset"] == "candidate":
            return (dataset_rank, -score, subtitle)
        return (dataset_rank, 0.0, subtitle)

    return sorted(items, key=key)


def build_html(items: list[dict[str, object]]) -> str:
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Herring Spawn Labeler v2</title>
  <style>
    :root {
      --bg: #0b0d12;
      --panel: #121722;
      --panel-2: #171d2a;
      --panel-3: #1d2433;
      --line: #273042;
      --text: #e7ecf5;
      --muted: #92a0b5;
      --muted-2: #6c7890;
      --accent: #63d471;
      --accent-2: #5ab7ff;
      --warn: #ffd166;
      --danger: #ff6b6b;
      --cloud: #ffb84d;
      --unknown: #9aa4b2;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #121827 0%, var(--bg) 45%);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, select { font: inherit; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(360px, 0.9fr);
      gap: 16px;
      padding: 16px;
    }
    .main, .side {
      min-width: 0;
      background: color-mix(in srgb, var(--panel) 94%, black);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .hero {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(10px);
    }
    .title-row {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .title-row h1 { margin: 0; font-size: 20px; line-height: 1.15; }
    .subtitle { color: var(--muted); font-size: 13px; margin-top: 5px; }
    .pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .pill {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--muted);
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .pill strong { color: var(--text); }
    .progress-wrap {
      margin-top: 12px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      height: 12px;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 160ms ease;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .field {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .field label { font-size: 12px; color: var(--muted-2); }
    .field select {
      background: var(--panel-3);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 10px;
      outline: none;
    }
    .btn {
      background: var(--panel-3);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    .btn:hover { transform: translateY(-1px); border-color: #3b4963; }
    .btn.primary { background: rgba(99, 212, 113, 0.12); border-color: rgba(99, 212, 113, 0.4); }
    .btn.ghost { background: transparent; }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .stat {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      min-width: 0;
    }
    .stat .num { font-size: 20px; font-weight: 700; line-height: 1; }
    .stat .lbl { color: var(--muted); font-size: 11px; margin-top: 5px; text-transform: uppercase; letter-spacing: 0.08em; }
    .stat.spawn .num { color: var(--accent); }
    .stat.nospawn .num { color: var(--danger); }
    .stat.cloudy .num { color: var(--cloud); }
    .stat.unknown .num { color: var(--unknown); }
    .stat.total .num { color: var(--text); }
    .grid-shell {
      padding: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }
    .card:hover, .card.selected { transform: translateY(-2px); border-color: #3f5277; box-shadow: 0 14px 34px rgba(0,0,0,0.3); }
    .card.selected { outline: 2px solid rgba(90, 183, 255, 0.3); }
    .card .images { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: #0b0d12; }
    .card.single .images { grid-template-columns: 1fr; }
    .img-wrap { position: relative; overflow: hidden; background: #0b0d12; min-height: 160px; }
    .img-wrap img, .img-wrap .placeholder {
      width: 100%; height: 100%; display: block; object-fit: cover; aspect-ratio: 1 / 1;
    }
    .img-wrap.zoomable img { transition: transform 180ms ease; transform-origin: center center; }
    .img-wrap.zoomable:hover img { transform: scale(1.18); }
    .placeholder {
      display: grid; place-items: center;
      color: var(--muted-2);
      background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
      text-align: center;
      padding: 12px;
      font-size: 12px;
      line-height: 1.45;
    }
    .img-label {
      position: absolute; left: 8px; top: 8px;
      background: rgba(0,0,0,0.65);
      border: 1px solid rgba(255,255,255,0.08);
      color: #f5f7fb;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      backdrop-filter: blur(8px);
    }
    .card-body { padding: 11px 12px 12px; }
    .meta-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
    .headline { font-weight: 700; font-size: 14px; margin: 0; }
    .subline { color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.4; }
    .badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
    .badge {
      display: inline-flex; align-items: center; gap: 6px;
      border-radius: 999px; padding: 5px 8px; font-size: 11px; font-weight: 600;
      background: var(--panel-3); border: 1px solid var(--line); color: var(--muted);
    }
    .badge.spawn { color: var(--accent); border-color: rgba(99,212,113,0.3); }
    .badge.nospawn { color: var(--danger); border-color: rgba(255,107,107,0.3); }
    .badge.cloudy { color: var(--cloud); border-color: rgba(255,184,77,0.3); }
    .badge.unknown { color: var(--unknown); border-color: rgba(154,164,178,0.3); }
    .badge.dataset { color: #9ec5ff; }
    .label-actions {
      display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;
    }
    .label-btn {
      padding: 7px 10px; border-radius: 10px; border: 1px solid var(--line);
      background: var(--panel-3); color: var(--text); cursor: pointer;
      font-size: 12px; font-weight: 600;
    }
    .label-btn.active { border-color: currentColor; }
    .label-btn.spawn { color: var(--accent); }
    .label-btn.nospawn { color: var(--danger); }
    .label-btn.cloudy { color: var(--cloud); }
    .label-btn.unknown { color: var(--unknown); }
    .detail {
      height: 100%; display: flex; flex-direction: column;
    }
    .detail-head {
      padding: 16px; border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
    }
    .detail-head h2 { margin: 0; font-size: 18px; }
    .detail-head .desc { color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.45; }
    .detail-body { padding: 16px; display: grid; gap: 14px; overflow: auto; }
    .detail-view {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }
    .detail-view.compare .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: #0b0d12; }
    .detail-view.single .compare-grid { display: block; }
    .compare-cell { position: relative; min-height: 220px; background: #0b0d12; }
    .compare-cell img { width: 100%; height: 100%; object-fit: cover; display: block; aspect-ratio: 1 / 1; }
    .compare-cell.zoomable img { transition: transform 180ms ease; transform-origin: center center; }
    .compare-cell.zoomable:hover img { transform: scale(1.2); }
    .compare-cell .placeholder { min-height: 220px; }
    .detail-meta { display: grid; gap: 10px; }
    .info-grid {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;
    }
    .info-box {
      background: var(--panel-2); border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px;
    }
    .info-box .k { color: var(--muted-2); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
    .info-box .v { margin-top: 5px; font-size: 13px; line-height: 1.45; }
    .hint {
      color: var(--muted); font-size: 12px; line-height: 1.6;
      background: rgba(255,255,255,0.02); border: 1px dashed var(--line);
      border-radius: 14px; padding: 12px;
    }
    .kbd {
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 1.8em; padding: 0.15em 0.45em; margin: 0 2px;
      border-radius: 6px; border: 1px solid #3b4660; background: #101523; color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
    }
    .footer-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .empty {
      padding: 30px 16px; text-align: center; color: var(--muted);
    }
    .app.focus-mode {
      min-height: 100vh;
      height: 100vh;
      overflow: hidden;
      display: block;
      padding: 0;
      gap: 0;
      background: radial-gradient(circle at top, #121827 0%, #05070b 70%);
    }
    .focus-shell {
      height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 12px;
      padding: 12px;
      overflow: hidden;
    }
    .focus-bar,
    .focus-footer {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(10, 13, 20, 0.72);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
    }
    .focus-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 12px;
      min-height: 44px;
    }
    .focus-bar .pill-row { margin: 0; }
    .focus-stage {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      align-items: stretch;
    }
    .focus-panel {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 10px;
      background: rgba(18, 23, 34, 0.82);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px;
      overflow: hidden;
      transition: opacity 160ms ease, transform 160ms ease;
    }
    .focus-panel:hover { transform: translateY(-1px); }
    .focus-panel .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .focus-media {
      min-height: 0;
      background: #05070b;
      border-radius: 14px;
      overflow: hidden;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .focus-media img {
      width: 100%;
      height: 100%;
      max-height: 68vh;
      object-fit: contain;
      display: block;
      background: #05070b;
    }
    .focus-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      display: grid;
      gap: 8px;
    }
    .focus-meta strong { color: var(--text); }
    .focus-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .focus-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .focus-hints {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .focus-hints .kbd { margin: 0; }
    .focus-hints .muted { color: var(--muted-2); }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .focus-stage { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="app" class="app"></div>

  <script>
    const DATA = __ITEMS__;
    const LS_KEY = 'herring-labeler-v2';
    const LABEL_ORDER = ['spawn', 'nospawn', 'cloudy', 'unknown'];
    const LABEL_META = {
      spawn: { name: 'Spawn', short: 'S', className: 'spawn' },
      nospawn: { name: 'No Spawn', short: 'N', className: 'nospawn' },
      cloudy: { name: 'Cloudy', short: 'C', className: 'cloudy' },
      unknown: { name: 'Unknown', short: 'U', className: 'unknown' },
    };

    const state = {
      items: DATA.map((item) => ({ ...item, label: null, updatedAt: null })),
      filter: 'all',
      mode: 'compare',
      activeId: null,
      search: '',
      focusMode: false,
    };

    let focusMode = false;

    function safeJsonParse(value, fallback) {
      try { return JSON.parse(value); } catch { return fallback; }
    }

    function storageGet() {
      try { return localStorage.getItem(LS_KEY); } catch { return null; }
    }

    function storageSet(value) {
      try { localStorage.setItem(LS_KEY, value); return true; } catch { return false; }
    }

    function storageRemove() {
      try { localStorage.removeItem(LS_KEY); } catch {}
    }

    function loadState() {
      const saved = safeJsonParse(storageGet(), null);
      if (!saved || typeof saved !== 'object') return;
      if (saved.filter) state.filter = saved.filter;
      if (saved.mode) state.mode = saved.mode;
      if (saved.activeId) state.activeId = saved.activeId;
      if (saved.search) state.search = saved.search;
      if (typeof saved.focusMode === 'boolean') {
        state.focusMode = saved.focusMode;
        focusMode = saved.focusMode;
      }
      if (saved.labels && typeof saved.labels === 'object') {
        for (const item of state.items) {
          if (Object.prototype.hasOwnProperty.call(saved.labels, item.id)) {
            item.label = saved.labels[item.id] || null;
            item.updatedAt = saved.updatedAt?.[item.id] || null;
          }
        }
      }
    }

    function persistState() {
      const labels = {};
      const updatedAt = {};
      for (const item of state.items) {
        if (item.label) labels[item.id] = item.label;
        if (item.updatedAt) updatedAt[item.id] = item.updatedAt;
      }
      const persisted = storageSet(JSON.stringify({
        filter: state.filter,
        mode: state.mode,
        activeId: state.activeId,
        search: state.search,
        focusMode,
        labels,
        updatedAt,
      }));
      document.getElementById('saveState').textContent = persisted ? 'Saved' : 'Memory only';
      window.clearTimeout(window.__saveTimer);
      window.__saveTimer = window.setTimeout(() => {
        const el = document.getElementById('saveState');
        if (el) el.textContent = 'Auto-saved';
      }, 700);
    }

    function setActiveId(id) {
      state.activeId = id;
      persistState();
      render();
      scrollActiveIntoView();
    }

    function toggleFocusMode() {
      focusMode = !focusMode;
      state.focusMode = focusMode;
      persistState();
      render();
    }

    function setLabel(id, label) {
      const item = state.items.find((row) => row.id === id);
      if (!item) return;
      item.label = label || null;
      item.updatedAt = new Date().toISOString();
      persistState();
      render();
    }

    function getVisibleItems() {
      const search = state.search.trim().toLowerCase();
      return state.items.filter((item) => {
        const label = item.label || 'unlabeled';
        if (state.filter !== 'all' && label !== state.filter) return false;
        if (search) {
          const hay = [item.id, item.title, item.subtitle, item.dataset, item.note].filter(Boolean).join(' ').toLowerCase();
          if (!hay.includes(search)) return false;
        }
        return true;
      });
    }

    function counts() {
      const out = { spawn: 0, nospawn: 0, cloudy: 0, unknown: 0, unlabeled: 0 };
      for (const item of state.items) {
        if (!item.label) out.unlabeled += 1;
        else if (Object.prototype.hasOwnProperty.call(out, item.label)) out[item.label] += 1;
      }
      return out;
    }

    function renderStats() {
      const c = counts();
      const total = state.items.length;
      const labeled = total - c.unlabeled;
      const pct = total ? (labeled / total) * 100 : 0;
      const progressText = document.getElementById('progressText');
      const progressBar = document.getElementById('progressBar');
      const countSpawn = document.getElementById('countSpawn');
      const countNoSpawn = document.getElementById('countNoSpawn');
      const countCloudy = document.getElementById('countCloudy');
      const countUnknown = document.getElementById('countUnknown');
      const countUnlabeled = document.getElementById('countUnlabeled');
      const countVisible = document.getElementById('countVisible');
      if (progressText) progressText.textContent = `${labeled} / ${total} labeled`;
      if (progressBar) progressBar.style.width = `${pct}%`;
      if (countSpawn) countSpawn.textContent = c.spawn;
      if (countNoSpawn) countNoSpawn.textContent = c.nospawn;
      if (countCloudy) countCloudy.textContent = c.cloudy;
      if (countUnknown) countUnknown.textContent = c.unknown;
      if (countUnlabeled) countUnlabeled.textContent = c.unlabeled;
      if (countVisible) countVisible.textContent = getVisibleItems().length;
    }

    function formatMeta(item) {
      const meta = item.meta || {};
      const bits = [];
      if (item.dataset === 'candidate') {
        if (typeof meta.score === 'number') bits.push(`score ${meta.score.toFixed(3)}`);
        if (typeof meta.cloud === 'number') bits.push(`cloud ${meta.cloud.toFixed(1)}%`);
      } else {
        if (meta.scene_date) bits.push(`scene ${meta.scene_date}`);
        if (typeof meta.cloud === 'number' && Number.isFinite(meta.cloud)) bits.push(`cloud ${meta.cloud.toFixed(1)}%`);
        if (typeof meta.days_from_spawn === 'number' && Number.isFinite(meta.days_from_spawn)) bits.push(`${meta.days_from_spawn}d from spawn`);
      }
      return bits.join(' · ');
    }

    function labelBadge(item) {
      if (!item.label) return '<span class="badge unknown">Unlabeled</span>';
      const meta = LABEL_META[item.label];
      return `<span class="badge ${item.label}">${meta.name}</span>`;
    }

    function renderCard(item) {
      const selected = item.id === state.activeId ? 'selected' : '';
      const itemLabel = item.label || 'unlabeled';
      const compare = item.off_thumb ? 'compare' : 'single';
      const leftSrc = item.spawn_thumb ? item.spawn_thumb : '';
      const rightSrc = item.off_thumb ? item.off_thumb : '';
      const leftHtml = leftSrc
        ? `<img src="${leftSrc}" alt="${item.title} spawn image" loading="lazy">`
        : `<div class="placeholder">No image</div>`;
      const rightHtml = rightSrc
        ? `<img src="${rightSrc}" alt="${item.title} off-season image" loading="lazy">`
        : `<div class="placeholder">No off-season image</div>`;
      const badges = [
        `<span class="badge dataset">${item.dataset === 'candidate' ? 'Candidate' : 'DFO'}</span>`,
        item.off_thumb ? `<span class="badge ${item.label || 'unknown'}">Compare</span>` : `<span class="badge unknown">Single</span>`,
        labelBadge(item),
      ].join('');
      const labelButtons = LABEL_ORDER.map((label) => {
        const active = item.label === label ? 'active' : '';
        return `<button class="label-btn ${label} ${active}" data-label="${label}" data-id="${item.id}">${LABEL_META[label].short} ${LABEL_META[label].name}</button>`;
      }).join('');

      return `
        <article class="card ${selected} ${compare}" data-id="${item.id}">
          <div class="images">
            <div class="img-wrap zoomable">
              <div class="img-label">Spawn-season</div>
              ${leftHtml}
            </div>
            <div class="img-wrap zoomable">
              <div class="img-label">Off-season</div>
              ${rightHtml}
            </div>
          </div>
          <div class="card-body">
            <div class="meta-row">
              <div>
                <div class="headline">${escapeHtml(item.title)}</div>
                <div class="subline">${escapeHtml(item.subtitle || item.id)}</div>
              </div>
            </div>
            <div class="subline">${escapeHtml(formatMeta(item) || item.note || '')}</div>
            <div class="badges">${badges}</div>
            <div class="label-actions">${labelButtons}</div>
          </div>
        </article>
      `;
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function activeItem() {
      return state.items.find((item) => item.id === state.activeId) || getVisibleItems()[0] || state.items[0] || null;
    }

    function renderDetail(visibleItems) {
      const item = activeItem();
      const detail = document.getElementById('detailContent');
      const mode = state.mode;
      if (!item || !visibleItems.length) {
        detail.innerHTML = '<div class="empty">No items available.</div>';
        return;
      }
      const spawnHtml = item.spawn_thumb
        ? `<img src="${item.spawn_thumb}" alt="${escapeHtml(item.title)} spawn-season" loading="eager">`
        : `<div class="placeholder">No spawn-season image</div>`;
      const offHtml = item.off_thumb
        ? `<img src="${item.off_thumb}" alt="${escapeHtml(item.title)} off-season" loading="eager">`
        : `<div class="placeholder">No off-season image available</div>`;
      const compareClass = mode === 'single' ? 'single' : 'compare';
      const labelButtons = LABEL_ORDER.map((label) => {
        const active = item.label === label ? 'active' : '';
        return `<button class="label-btn ${label} ${active}" data-label="${label}" data-id="${item.id}">${LABEL_META[label].short} ${LABEL_META[label].name}</button>`;
      }).join('');

      const viewerHtml = mode === 'compare'
        ? `
          <div class="compare-grid">
            <div class="compare-cell zoomable">
              <div class="img-label">Spawn-season</div>
              ${spawnHtml}
            </div>
            <div class="compare-cell zoomable">
              <div class="img-label">Off-season</div>
              ${offHtml}
            </div>
          </div>`
        : `
          <div class="compare-grid">
            <div class="compare-cell zoomable" style="grid-column:1 / -1;">
              <div class="img-label">Spawn-season</div>
              ${spawnHtml}
            </div>
          </div>`;

      detail.innerHTML = `
        <div class="detail-view ${compareClass}">
          ${viewerHtml}
        </div>
        <div class="detail-meta">
          <div class="info-grid">
            <div class="info-box"><div class="k">Item</div><div class="v">${escapeHtml(item.title)}</div></div>
            <div class="info-box"><div class="k">ID</div><div class="v" style="word-break:break-all;">${escapeHtml(item.id)}</div></div>
            <div class="info-box"><div class="k">Dataset</div><div class="v">${item.dataset === 'candidate' ? 'Candidate delta review' : 'DFO ingressed review'}</div></div>
            <div class="info-box"><div class="k">Meta</div><div class="v">${escapeHtml(formatMeta(item) || '—')}</div></div>
          </div>
          <div class="info-box">
            <div class="k">Review</div>
            <div class="v">${item.label ? `Current label: <strong>${escapeHtml(item.label)}</strong>` : 'Current label: <strong>unlabeled</strong>'}</div>
          </div>
          <div class="footer-actions">
            <button class="btn primary" id="modeToggle">${mode === 'compare' ? 'Switch to single view' : 'Switch to compare view'}</button>
            <button class="btn" id="downloadBtn">Download JSON</button>
            <button class="btn ghost" id="clearBtn">Clear saved labels</button>
          </div>
          <div class="hint">
            Shortcuts: <span class="kbd">s</span> spawn · <span class="kbd">n</span> no-spawn · <span class="kbd">c</span> cloudy · <span class="kbd">u</span> unknown · <span class="kbd">j</span> next · <span class="kbd">k</span> previous.
            Hover the imagery to zoom.
          </div>
        </div>
      `;

      document.getElementById('modeToggle').onclick = () => {
        state.mode = state.mode === 'compare' ? 'single' : 'compare';
        persistState();
        render();
      };
      document.getElementById('downloadBtn').onclick = downloadJSON;
      document.getElementById('clearBtn').onclick = clearSavedState;
      for (const button of detail.querySelectorAll('.label-btn')) {
        button.onclick = (event) => {
          event.stopPropagation();
          const id = button.getAttribute('data-id');
          const label = button.getAttribute('data-label');
          setLabel(id, label);
        };
      }
    }

    function renderGridView(visibleItems) {
      const grid = document.getElementById('grid');
      if (!visibleItems.length) {
        grid.innerHTML = '<div class="empty">No items match the current filter.</div>';
        return;
      }
      grid.innerHTML = visibleItems.map(renderCard).join('');
      for (const card of grid.querySelectorAll('.card')) {
        const id = card.getAttribute('data-id');
        card.onclick = () => setActiveId(id);
      }
      for (const button of grid.querySelectorAll('.label-btn')) {
        button.onclick = (event) => {
          event.stopPropagation();
          const id = button.getAttribute('data-id');
          const label = button.getAttribute('data-label');
          setLabel(id, label);
        };
      }
    }

    function renderFocusView() {
      const panel = document.getElementById('focusContent');
      const item = activeItem();
      if (!panel) return;
      if (!item) {
        panel.innerHTML = '<div class="empty">No items available.</div>';
        return;
      }

      const spawnHtml = item.spawn_thumb
        ? `<img src="${item.spawn_thumb}" alt="${escapeHtml(item.title)} spawn-season" loading="eager">`
        : `<div class="placeholder">No spawn-season image</div>`;
      const offHtml = item.off_thumb
        ? `<img src="${item.off_thumb}" alt="${escapeHtml(item.title)} off-season" loading="eager">`
        : `<div class="placeholder">No off-season image available</div>`;
      const labelButtons = LABEL_ORDER.map((label) => {
        const active = item.label === label ? 'active' : '';
        return `<button class="label-btn ${label} ${active}" data-label="${label}" data-id="${item.id}">${LABEL_META[label].short} ${LABEL_META[label].name}</button>`;
      }).join('');

      panel.innerHTML = `
        <div class="focus-stage">
          <div class="focus-panel">
            <div class="panel-title"><span>Spawn-season</span><span>${escapeHtml(item.title)}</span></div>
            <div class="focus-media">${spawnHtml}</div>
            <div class="focus-meta"><strong>${escapeHtml(item.subtitle || item.id)}</strong><span>${escapeHtml(formatMeta(item) || item.note || '')}</span></div>
          </div>
          <div class="focus-panel">
            <div class="panel-title"><span>Off-season</span><span>${item.label ? escapeHtml(item.label) : 'unlabeled'}</span></div>
            <div class="focus-media">${offHtml}</div>
            <div class="focus-meta">
              <span>${item.dataset === 'candidate' ? 'Candidate delta review' : 'DFO ingressed review'}</span>
              <div class="focus-actions">
                ${labelButtons}
              </div>
            </div>
          </div>
        </div>`;

      for (const button of panel.querySelectorAll('.label-btn')) {
        button.onclick = (event) => {
          event.stopPropagation();
          const id = button.getAttribute('data-id');
          const label = button.getAttribute('data-label');
          setLabel(id, label);
        };
      }
    }

    function downloadJSON() {
      const payload = {
        version: 2,
        exported_at: new Date().toISOString(),
        items: state.items.map((item) => ({
          id: item.id,
          dataset: item.dataset,
          title: item.title,
          subtitle: item.subtitle,
          label: item.label,
          updated_at: item.updatedAt,
        })),
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'herring-labels-v2.json';
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1500);
    }

    function clearSavedState() {
      if (!confirm('Clear saved labels from this browser?')) return;
      storageRemove();
      for (const item of state.items) {
        item.label = null;
        item.updatedAt = null;
      }
      state.filter = 'all';
      state.mode = 'compare';
      state.activeId = state.items[0]?.id || null;
      state.search = '';
      state.focusMode = false;
      focusMode = false;
      syncControls();
      render();
      persistState();
    }

    function syncControls() {
      const filter = document.getElementById('labelFilter');
      const search = document.getElementById('searchInput');
      const saveState = document.getElementById('saveState');
      const focusButtons = document.querySelectorAll('[data-focus-toggle]');
      if (filter) filter.value = state.filter;
      if (search) search.value = state.search;
      if (saveState) saveState.textContent = 'Auto-saved';
      for (const button of focusButtons) {
        button.textContent = focusMode ? 'Grid Mode' : 'Focus Mode';
      }
    }

    function nextItem(delta) {
      const items = getVisibleItems();
      if (!items.length) return;
      const current = items.findIndex((item) => item.id === state.activeId);
      const start = current >= 0 ? current : 0;
      let next = (start + delta + items.length) % items.length;
      state.activeId = items[next].id;
      persistState();
      render();
      scrollActiveIntoView();
    }

    function nextUnlabeledItem(delta) {
      const items = getVisibleItems();
      if (!items.length) return;
      const current = items.findIndex((item) => item.id === state.activeId);
      const start = current >= 0 ? current : (delta > 0 ? -1 : 0);
      for (let step = 1; step <= items.length; step += 1) {
        const next = (start + delta * step + items.length) % items.length;
        if (!items[next].label) {
          state.activeId = items[next].id;
          persistState();
          render();
          scrollActiveIntoView();
          return;
        }
      }
    }

    function scrollActiveIntoView() {
      requestAnimationFrame(() => {
        const card = document.querySelector(`.card[data-id="${CSS.escape(state.activeId || '')}"]`);
        if (card) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      });
    }

    function render() {
      const visibleItems = getVisibleItems();
      if (!visibleItems.some((item) => item.id === state.activeId)) {
        state.activeId = visibleItems[0]?.id || null;
      }
      focusMode = state.focusMode;
      const app = document.getElementById('app');
      if (app) {
        app.className = focusMode ? 'app focus-mode' : 'app';
        app.innerHTML = focusMode ? buildFocusShell() : buildShell();
        attachToolbarHandlers();
        syncControls();
        renderStats();
        if (focusMode) {
          renderFocusView();
        } else {
          renderGridView(visibleItems);
          renderDetail(visibleItems);
        }
      }
      persistState();
    }

    function handleKeydown(event) {
      const tag = (event.target && event.target.tagName || '').toLowerCase();
      if (['input', 'textarea', 'select'].includes(tag)) return;
      const key = event.key.toLowerCase();
      if (key === 'j') { event.preventDefault(); nextItem(1); return; }
      if (key === 'k') { event.preventDefault(); nextItem(-1); return; }
      if (['s', 'n', 'c', 'u'].includes(key)) {
        event.preventDefault();
        const item = activeItem();
        if (item) {
          setLabel(item.id, key === 's' ? 'spawn' : key === 'n' ? 'nospawn' : key === 'c' ? 'cloudy' : 'unknown');
          nextUnlabeledItem(1);
        }
      }
    }

    function attachToolbarHandlers() {
      const filter = document.getElementById('labelFilter');
      const search = document.getElementById('searchInput');
      const downloadTop = document.getElementById('downloadTop');
      const clearTop = document.getElementById('clearTop');
      const focusToggle = document.getElementById('focusToggle');
      if (filter) {
        filter.addEventListener('change', (event) => {
          state.filter = event.target.value;
          persistState();
          render();
        });
      }
      if (search) {
        search.addEventListener('input', (event) => {
          state.search = event.target.value;
          persistState();
          render();
        });
      }
      if (downloadTop) downloadTop.onclick = downloadJSON;
      if (clearTop) clearTop.onclick = clearSavedState;
      if (focusToggle) focusToggle.onclick = toggleFocusMode;
    }

    function buildShell() {
      return `
        <section class="main">
          <div class="hero">
            <div class="title-row">
              <div>
                <h1>Herring Spawn Labeler v2</h1>
                <div class="subtitle">Fast review workspace for spawn-season vs off-season imagery, with keyboard-first labeling and browser persistence.</div>
              </div>
              <div class="pill-row">
                <div class="pill"><strong id="progressText">0 / 0 labeled</strong></div>
                <div class="pill"><span id="countVisible">0</span> visible</div>
                <div class="pill"><span id="saveState">Auto-saved</span></div>
              </div>
            </div>
            <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>
            <div class="stats">
              <div class="stat spawn"><div class="num" id="countSpawn">0</div><div class="lbl">Spawn</div></div>
              <div class="stat nospawn"><div class="num" id="countNoSpawn">0</div><div class="lbl">No Spawn</div></div>
              <div class="stat cloudy"><div class="num" id="countCloudy">0</div><div class="lbl">Cloudy</div></div>
              <div class="stat unknown"><div class="num" id="countUnknown">0</div><div class="lbl">Unknown</div></div>
              <div class="stat total"><div class="num" id="countUnlabeled">0</div><div class="lbl">Unlabeled</div></div>
            </div>
            <div class="toolbar">
              <div class="field">
                <label for="labelFilter">Filter</label>
                <select id="labelFilter">
                  <option value="all">All</option>
                  <option value="unlabeled">Unlabeled</option>
                  <option value="spawn">Spawn</option>
                  <option value="nospawn">No Spawn</option>
                  <option value="cloudy">Cloudy</option>
                  <option value="unknown">Unknown</option>
                </select>
              </div>
              <div class="field" style="flex:1 1 240px;">
                <label for="searchInput">Search</label>
                <input id="searchInput" type="text" placeholder="Region, location, or id" style="flex:1;border:none;background:transparent;color:var(--text);outline:none;min-width:0;">
              </div>
              <button class="btn primary" id="downloadTop">Download JSON</button>
              <button class="btn" id="clearTop">Clear saved labels</button>
              <button class="btn ghost" id="focusToggle" data-focus-toggle>${focusMode ? 'Grid Mode' : 'Focus Mode'}</button>
            </div>
          </div>
          <div class="grid-shell">
            <div class="grid" id="grid"></div>
          </div>
        </section>
        <aside class="side">
          <div class="detail">
            <div class="detail-head">
              <h2>Selected item</h2>
              <div class="desc">Use <span class="kbd">j</span>/<span class="kbd">k</span> to move, then label with <span class="kbd">s</span>, <span class="kbd">n</span>, <span class="kbd">c</span>, or <span class="kbd">u</span>.</div>
            </div>
            <div class="detail-body" id="detailContent"></div>
          </div>
        </aside>
      `;
    }

    function buildFocusShell() {
      return `
        <section class="focus-shell">
          <div class="focus-bar">
            <div class="pill-row">
              <div class="pill"><strong id="progressText">0 / 0 labeled</strong></div>
              <div class="pill"><span id="countVisible">0</span> visible</div>
              <div class="pill"><span id="saveState">Auto-saved</span></div>
            </div>
            <button class="btn ghost" id="focusToggle" data-focus-toggle>${focusMode ? 'Grid Mode' : 'Focus Mode'}</button>
          </div>
          <div id="focusContent"></div>
          <div class="focus-footer">
            <div class="focus-hints">
              <span class="muted">Shortcuts:</span>
              <span><span class="kbd">s</span> spawn</span>
              <span><span class="kbd">n</span> no-spawn</span>
              <span><span class="kbd">c</span> cloudy</span>
              <span><span class="kbd">u</span> unknown</span>
              <span><span class="kbd">j</span>/<span class="kbd">k</span> prev/next</span>
            </div>
            <div class="muted">Auto-advances to the next unlabeled image.</div>
          </div>
        </section>
      `;
    }

    function init() {
      loadState();
      if (!state.activeId) state.activeId = state.items[0]?.id || null;
      render();
      document.addEventListener('keydown', handleKeydown);
    }

    init();
  </script>
</body>
</html>
"""

    return template.replace("__ITEMS__", json.dumps(items, indent=2))


def main() -> None:
    items = sort_items(load_candidate_items() + load_ingressed_items())
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(build_html(items), encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")
    print(f"Open file://{OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
