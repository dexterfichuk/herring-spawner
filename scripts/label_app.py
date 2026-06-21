#!/usr/bin/env python3
"""
Gradio labeling app for herring spawn candidate thumbnails.

Loads candidates from data/candidates_fresh/manifest.json, shows each true-color
thumbnail, and lets the user label them with keyboard shortcuts.

Keyboard shortcuts:
  Y  = spawn         N  = no-spawn       S  = skip
  ←  = previous      →  = next           U  = next unlabeled
  C  = clear label    G  = grid view      1  = single view
  Exp = export labels

Labels are auto-saved to labels.json on every keystroke.

Usage:
  python scripts/label_app.py
  python scripts/label_app.py --manifest path/to/manifest.json --port 7890
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh/manifest.json")
DEFAULT_IMAGE_DIR = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh")
DEFAULT_LABELS = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh/labels.json")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
MANIFEST: list[dict] = []
IMAGE_DIR: Path = DEFAULT_IMAGE_DIR
LABELS_FILE: Path = DEFAULT_LABELS
_state: dict = {"labels": {}, "idx": 0}


def load_labels() -> dict[str, str]:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text())
    return {}


def save_labels() -> None:
    LABELS_FILE.write_text(json.dumps(_state["labels"], indent=2, sort_keys=True))


def _img_fname(entry: dict) -> str:
    return entry.get("filename", "")


def _effective_label(entry: dict) -> str | None:
    fname = _img_fname(entry)
    if fname in _state["labels"]:
        return _state["labels"][fname]
    return entry.get("status") if entry.get("status") == "ok" else None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def compute_stats() -> dict:
    total = len(MANIFEST)
    n_labeled = 0
    n_spawn = 0
    n_nospawn = 0
    n_skip = 0
    n_no_scene = 0
    n_download_failed = 0

    for entry in MANIFEST:
        label = _state["labels"].get(_img_fname(entry))
        if label is not None:
            n_labeled += 1
            if label == "spawn":
                n_spawn += 1
            elif label == "no-spawn":
                n_nospawn += 1
            elif label == "skip":
                n_skip += 1
        elif entry.get("status") == "no_scene":
            n_no_scene += 1
        elif entry.get("status") == "download_failed":
            n_download_failed += 1

    return {
        "total": total,
        "labeled": n_labeled,
        "spawn": n_spawn,
        "nospawn": n_nospawn,
        "skip": n_skip,
        "remaining": total - n_labeled,
        "no_scene": n_no_scene,
        "download_failed": n_download_failed,
    }


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------
def get_current_view() -> tuple:
    """Return (image, info_html, stats_html) for current index."""
    if not MANIFEST:
        return None, "<p>No candidates.</p>", "<p>—</p>"

    idx = _state["idx"]
    entry = MANIFEST[idx]
    fname = _img_fname(entry)
    img_path = IMAGE_DIR / fname if fname else None

    # --- Info panel ---
    info_lines = [
        f"**{entry.get('region', '?')}** — {entry.get('location_name', '?')}"
    ]
    info_lines.append(f"Date: {entry.get('date', '?')}")
    info_lines.append(f"Lat: {entry.get('lat', '?'):.4f}, Lon: {entry.get('lon', '?'):.4f}")

    scene_date = entry.get("scene_date")
    if scene_date:
        info_lines.append(f"Satellite scene: {scene_date}")
    cloud = entry.get("cloud_cover")
    if cloud is not None:
        info_lines.append(f"Cloud cover: {cloud:.1f}%")
    days = entry.get("days_from_spawn")
    if days is not None:
        info_lines.append(f"Days from spawn: ±{days}")
    spawn_len = entry.get("spawn_length_m")
    spawn_wid = entry.get("spawn_width_m")
    if spawn_len:
        info_lines.append(f"Spawn extent: {spawn_len:.0f}m × {spawn_wid:.0f}m")
    if entry.get("method"):
        info_lines.append(f"Survey method: {entry['method']}")
    scene_id = entry.get("scene_id")
    if scene_id:
        info_lines.append(f"Scene: {scene_id}")

    if entry.get("status") != "ok":
        info_lines.append("")
        info_lines.append(
            f'<span style="color:#e74c3c;font-weight:bold">⚠ {entry["status"]}</span>'
        )

    info_html = "<br>".join(info_lines)

    # --- Stats panel ---
    stats = compute_stats()
    label = _effective_label(entry)
    if label == "spawn":
        status = '<span style="color:#2d6a4f;font-weight:bold">✓ SPAWN</span>'
    elif label == "no-spawn":
        status = '<span style="color:#9b2226;font-weight:bold">✗ NO SPAWN</span>'
    elif label == "skip":
        status = '<span style="color:#6b7280;font-weight:bold">— SKIPPED</span>'
    else:
        status = '<span style="color:#f59e0b;font-weight:bold">? UNLABELED</span>'

    pct = (stats["labeled"] / stats["total"] * 100) if stats["total"] else 0
    bar = (
        f'<div style="background:#e5e7eb;border-radius:8px;overflow:hidden;height:12px;margin:8px 0">'
        f'<div style="background:linear-gradient(90deg,#2d6a4f,#40916c);height:100%;width:{pct:.0f}%"></div></div>'
    )

    stats_lines = [
        f"**{idx + 1} / {stats['total']}**",
        f"Status: {status}",
        bar,
        f"Labeled:  {stats['labeled']} ({pct:.0f}%)",
        f"Spawn:    {stats['spawn']}",
        f"No spawn: {stats['nospawn']}",
        f"Skipped:  {stats['skip']}",
        f"Remaining: {stats['remaining']}",
    ]
    if stats["no_scene"] > 0:
        stats_lines.append(f"No scene: {stats['no_scene']}")
    if stats["download_failed"] > 0:
        stats_lines.append(f"DL failed: {stats['download_failed']}")

    stats_html = "<br>".join(stats_lines)

    if img_path and img_path.exists():
        return str(img_path), info_html, stats_html
    return None, info_html, stats_html


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def set_label(choice: str) -> tuple:
    if not MANIFEST:
        return get_current_view()
    entry = MANIFEST[_state["idx"]]
    fname = _img_fname(entry)
    if fname:
        _state["labels"][fname] = choice
        save_labels()
    # Advance to next unlabeled
    next_idx = _state["idx"] + 1
    while next_idx < len(MANIFEST):
        nf = _img_fname(MANIFEST[next_idx])
        if nf and nf not in _state["labels"]:
            break
        next_idx += 1
    if next_idx < len(MANIFEST):
        _state["idx"] = next_idx
    return get_current_view()


def go_next() -> tuple:
    if not MANIFEST:
        return get_current_view()
    _state["idx"] = min(_state["idx"] + 1, len(MANIFEST) - 1)
    return get_current_view()


def go_prev() -> tuple:
    _state["idx"] = max(_state["idx"] - 1, 0)
    return get_current_view()


def go_to(idx_str: str) -> tuple:
    try:
        idx = int(idx_str) - 1
        _state["idx"] = max(0, min(idx, len(MANIFEST) - 1))
    except (ValueError, TypeError):
        pass
    return get_current_view()


def jump_to_next_unlabeled() -> tuple:
    for i in range(_state["idx"] + 1, len(MANIFEST)):
        fname = _img_fname(MANIFEST[i])
        if fname and fname not in _state["labels"]:
            _state["idx"] = i
            return get_current_view()
    return get_current_view()


def jump_to_prev_unlabeled() -> tuple:
    for i in range(_state["idx"] - 1, -1, -1):
        fname = _img_fname(MANIFEST[i])
        if fname and fname not in _state["labels"]:
            _state["idx"] = i
            return get_current_view()
    return get_current_view()


def clear_label() -> tuple:
    if not MANIFEST:
        return get_current_view()
    entry = MANIFEST[_state["idx"]]
    fname = _img_fname(entry)
    if fname and fname in _state["labels"]:
        del _state["labels"][fname]
        save_labels()
    return get_current_view()


def export_labels() -> str:
    if not _state["labels"]:
        return "No labels to export."
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = LABELS_FILE.parent / f"labels_export_{now}.json"
    stats = compute_stats()
    summary = {
        "exported_at": datetime.now().isoformat(),
        "manifest": str(MANIFEST_PATH),
        "total": stats["total"],
        "labeled": stats["labeled"],
        "spawn": stats["spawn"],
        "no_spawn": stats["nospawn"],
        "skip": stats["skip"],
        "labels": _state["labels"],
    }
    export_path.write_text(json.dumps(summary, indent=2))
    return f"Exported to {export_path}"


# ---------------------------------------------------------------------------
# Grid scan view
# ---------------------------------------------------------------------------
GRID_PAGE_SIZE = 40
GRID_PAGE = 0


def grid_page_count() -> int:
    return max(1, (len(MANIFEST) + GRID_PAGE_SIZE - 1) // GRID_PAGE_SIZE)


def grid_get_page(page: int = 0) -> tuple:
    global GRID_PAGE
    GRID_PAGE = max(0, min(page, grid_page_count() - 1))
    start = GRID_PAGE * GRID_PAGE_SIZE
    end = min(start + GRID_PAGE_SIZE, len(MANIFEST))
    page_entries = MANIFEST[start:end]

    cards = []
    for i, entry in enumerate(page_entries):
        fname = _img_fname(entry)
        img_path = IMAGE_DIR / fname if fname else None
        label = _effective_label(entry)
        gi = start + i

        if label == "spawn":
            border = "4px solid #2d6a4f"
            badge = '<div style="position:absolute;top:4px;right:4px;background:#2d6a4f;color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">SPAWN</div>'
        elif label == "no-spawn":
            border = "2px solid #9b2226"
            badge = '<div style="position:absolute;top:4px;right:4px;background:#9b2226;color:white;padding:2px 8px;border-radius:4px;font-size:12px">NO</div>'
        elif entry.get("status") != "ok":
            border = "2px dashed #6b7280"
            badge = '<div style="position:absolute;top:4px;right:4px;background:#6b7280;color:white;padding:2px 8px;border-radius:4px;font-size:11px">NO DATA</div>'
        else:
            border = "2px solid #e5e7eb"
            badge = ""

        img_url = f"/file={img_path.resolve()}" if img_path and img_path.exists() else ""

        region = entry.get("region", "?")
        date_str = str(entry.get("date", ""))[:10]

        cards.append(f"""
        <div style="position:relative;display:inline-block;margin:4px;border:{border};border-radius:6px;overflow:hidden;width:170px;height:170px">
            <button onclick="gridSpawn({gi})" style="position:absolute;inset:0;border:none;background:none;cursor:pointer;padding:0;width:100%;height:100%" title="Click=spawn"
                    oncontextmenu="gridNoSpawn({gi});return false">
                <img src="{img_url}" style="width:100%;height:100%;object-fit:cover" loading="lazy">
            </button>
            {badge}
            <div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.6);color:white;padding:2px 6px;font-size:10px;display:flex;justify-content:space-between;pointer-events:none">
                <span>#{gi + 1} {region}</span>
                <span>{date_str}</span>
            </div>
        </div>""")

    gallery_html = f"""
    <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0;padding:8px">
        {''.join(cards)}
    </div>"""

    stats = compute_stats()
    page_info = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:#f9fafb;border-radius:8px;margin-bottom:8px">
        <span><b>Page {GRID_PAGE + 1} / {grid_page_count()}</b> · {start + 1}–{end} of {stats['total']}</span>
        <span style="color:#2d6a4f">✓ {stats['spawn']} spawn</span>
        <span style="color:#9b2226">✗ {stats['nospawn']} no-spawn</span>
        <span style="color:#f59e0b">{stats['remaining']} remaining</span>
    </div>"""

    return gallery_html, page_info


def grid_mark(idx_str: str, label: str) -> tuple:
    global GRID_PAGE
    try:
        idx = int(idx_str)
    except (ValueError, TypeError):
        return grid_get_page(GRID_PAGE)
    if 0 <= idx < len(MANIFEST):
        fname = _img_fname(MANIFEST[idx])
        if fname:
            _state["labels"][fname] = label
            save_labels()
    return grid_get_page(GRID_PAGE)


def grid_prev_page() -> tuple:
    return grid_get_page(GRID_PAGE - 1)


def grid_next_page() -> tuple:
    return grid_get_page(GRID_PAGE + 1)


def grid_jump_to_unlabeled() -> tuple:
    for i, entry in enumerate(MANIFEST):
        if _effective_label(entry) is None:
            return grid_get_page(i // GRID_PAGE_SIZE)
    return grid_get_page(0)


# ---------------------------------------------------------------------------
# Keyboard JS
# ---------------------------------------------------------------------------
KEYBOARD_JS = """
function() {
    window.gridSpawn = function(idx) {
        var el = document.querySelector('#grid_click_idx input, #grid_click_idx textarea');
        if (el) {
            var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            s.call(el, String(idx)); el.dispatchEvent(new Event('input', {bubbles:true}));
        }
        setTimeout(function(){ document.querySelector('#grid_click_btn')?.click(); }, 30);
    };
    window.gridNoSpawn = function(idx) {
        var el = document.querySelector('#grid_nospawn_idx input, #grid_nospawn_idx textarea');
        if (el) {
            var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            s.call(el, String(idx)); el.dispatchEvent(new Event('input', {bubbles:true}));
        }
        setTimeout(function(){ document.querySelector('#grid_nospawn_btn')?.click(); }, 30);
    };
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        switch(e.key.toLowerCase()) {
            case 'y': document.querySelector('#spawn_btn')?.click(); break;
            case 'n': document.querySelector('#nospawn_btn')?.click(); break;
            case 's': document.querySelector('#skip_btn')?.click(); break;
            case 'arrowleft': document.querySelector('#prev_btn')?.click(); break;
            case 'arrowright': document.querySelector('#next_btn')?.click(); break;
            case 'u': document.querySelector('#next_unlabeled_btn')?.click(); break;
            case 'c': document.querySelector('#clear_btn')?.click(); break;
            case 'g':
                var tabs = document.querySelectorAll('button[role="tab"]');
                if (tabs.length > 1) tabs[1].click();
                break;
            case '1':
                var tabs = document.querySelectorAll('button[role="tab"]');
                if (tabs.length > 0) tabs[0].click();
                break;
        }
    });
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Herring Spawn Labeler") as demo:
        gr.Markdown("# Herring Spawn Labeler")
        gr.Markdown(
            "**Y** = spawn · **N** = no spawn · **S** = skip · "
            "**←→** = navigate · **U** = next unlabeled · **C** = clear<br>"
            "Grid: **click** = spawn · **right-click** = no-spawn"
        )

        with gr.Tabs():
            # ---- Tab 1: Single Image Review ----
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column(scale=4):
                        image = gr.Image(type="filepath", height=620, label=None)

                    with gr.Column(scale=2):
                        stats = gr.HTML(value="Loading...")
                        info = gr.HTML(value="")

                        with gr.Row():
                            spawn_btn = gr.Button("✓ SPAWN", elem_id="spawn_btn", size="lg")
                            nospawn_btn = gr.Button("✗ NO SPAWN", elem_id="nospawn_btn", size="lg")

                        with gr.Row():
                            skip_btn = gr.Button("Skip", elem_id="skip_btn", size="lg")

                        gr.Markdown("---")

                        with gr.Row():
                            prev_btn = gr.Button("◀ Prev", elem_id="prev_btn")
                            next_btn = gr.Button("Next ▶", elem_id="next_btn")

                        with gr.Row():
                            next_unlabeled_btn = gr.Button("⏩ Next Unlabeled", elem_id="next_unlabeled_btn")
                            clear_btn = gr.Button("Clear Label", elem_id="clear_btn")

                        gr.Markdown("---")

                        jump_input = gr.Textbox(label="Go to #", placeholder="e.g. 42", max_lines=1)
                        jump_btn = gr.Button("Go", size="sm")

                        export_btn = gr.Button("📤 Export Labels", variant="secondary", size="sm")
                        export_msg = gr.Textbox(label="", show_label=False, interactive=False)

                outputs = [image, info, stats]

                spawn_btn.click(lambda: set_label("spawn"), None, outputs)
                nospawn_btn.click(lambda: set_label("no-spawn"), None, outputs)
                skip_btn.click(lambda: set_label("skip"), None, outputs)
                next_btn.click(go_next, None, outputs)
                prev_btn.click(go_prev, None, outputs)
                next_unlabeled_btn.click(jump_to_next_unlabeled, None, outputs)
                clear_btn.click(clear_label, None, outputs)
                jump_btn.click(go_to, jump_input, outputs)
                export_btn.click(export_labels, None, export_msg)

                demo.load(get_current_view, None, outputs)

            # ---- Tab 2: Grid Quick Scan ----
            with gr.TabItem("Grid Scan"):
                grid_page_info = gr.HTML(value="Loading...")
                grid_gallery = gr.HTML(value="Loading...")

                with gr.Row():
                    grid_prev_btn = gr.Button("◀ Prev Page", size="sm")
                    grid_next_btn = gr.Button("Next Page ▶", size="sm")
                    grid_unlabeled_btn = gr.Button("⏩ Jump to Unlabeled", size="sm")

                grid_click_idx = gr.Textbox(visible=False, elem_id="grid_click_idx")
                grid_click_btn = gr.Button("_spawn", visible=False, elem_id="grid_click_btn")
                grid_nospawn_idx = gr.Textbox(visible=False, elem_id="grid_nospawn_idx")
                grid_nospawn_btn = gr.Button("_nospawn", visible=False, elem_id="grid_nospawn_btn")

                grid_outputs = [grid_gallery, grid_page_info]

                grid_click_btn.click(lambda v: grid_mark(v, "spawn"), grid_click_idx, grid_outputs)
                grid_nospawn_btn.click(lambda v: grid_mark(v, "no-spawn"), grid_nospawn_idx, grid_outputs)
                grid_prev_btn.click(grid_prev_page, None, grid_outputs)
                grid_next_btn.click(grid_next_page, None, grid_outputs)
                grid_unlabeled_btn.click(grid_jump_to_unlabeled, None, grid_outputs)

                demo.load(lambda: grid_get_page(0), None, grid_outputs, show_progress=False)

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Herring spawn labeling app")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest JSON path")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Image directory")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Labels JSON path")
    parser.add_argument("--port", type=int, default=7888, help="Port (default: 7888)")
    parser.add_argument("--share", action="store_true", help="Public share link")
    args = parser.parse_args(argv)

    global MANIFEST, IMAGE_DIR, LABELS_FILE, MANIFEST_PATH
    MANIFEST_PATH = Path(args.manifest)
    IMAGE_DIR = Path(args.image_dir)
    LABELS_FILE = Path(args.labels)

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        print(f"Run scripts/fetch_candidates.py first to create the manifest.")
        return 1
    if not IMAGE_DIR.exists():
        print(f"ERROR: Image directory not found: {IMAGE_DIR}")
        return 1

    with open(MANIFEST_PATH) as f:
        raw = json.load(f)
    MANIFEST = raw if isinstance(raw, list) else []
    if not MANIFEST:
        print("ERROR: Manifest is empty.")
        return 1

    _state["labels"] = load_labels()
    _state["idx"] = 0

    # Start at first unlabeled
    for i, entry in enumerate(MANIFEST):
        fname = _img_fname(entry)
        if fname and fname not in _state["labels"]:
            _state["idx"] = i
            break

    stats = compute_stats()
    print(f"\n  Candidates: {stats['total']}")
    print(f"  Previously labeled: {stats['labeled']} ({stats['spawn']} spawn, {stats['nospawn']} no-spawn)")
    print(f"  Remaining: {stats['remaining']}")
    print(f"\n  http://localhost:{args.port}")
    print()

    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(IMAGE_DIR.resolve())],
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
        css="""
        #spawn_btn button { background: #2d6a4f !important; color: white !important; font-size: 18px !important; padding: 16px !important; }
        #nospawn_btn button { background: #9b2226 !important; color: white !important; font-size: 18px !important; padding: 16px !important; }
        #skip_btn button { background: #4b5563 !important; color: white !important; font-size: 18px !important; padding: 16px !important; }
        #next_unlabeled_btn button { background: #1e40af !important; color: white !important; }
        #clear_btn button { background: #92400e !important; color: white !important; }
        """,
        js=KEYBOARD_JS,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
