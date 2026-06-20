#!/usr/bin/env python3
"""Gradio labeling app for binary herring spawn classification.

Keyboard shortcuts: Y=spawn, N=no-spawn, S=skip, Left=back, Right=next.
Labels auto-saved to labels.json on every decision.

Usage:
    python scripts/label_gradio.py \
        --manifest data/candidates_shsi/manifest.json \
        --image-dir data/candidates_shsi \
        --labels data/candidates_shsi/labels.json

    # Then open http://localhost:7888
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Label state
# ---------------------------------------------------------------------------

MANIFEST: list[dict] = []
IMAGE_DIR: Path = Path("data/candidates_shsi")
LABELS_FILE: Path = Path("data/candidates_shsi/labels.json")
_state: dict = {"labels": {}, "idx": 0}


def load_labels() -> dict[str, str]:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text())
    return {}


def save_labels() -> None:
    LABELS_FILE.write_text(json.dumps(_state["labels"], indent=2, sort_keys=True))


def _img_fname(entry: dict) -> str:
    """Get image filename from entry, handling both manifest formats."""
    return entry.get("filename", entry.get("thumbnail_path", ""))


def _has_field(entry: dict, field: str) -> bool:
    return field in entry and entry[field] is not None


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def get_current_state() -> tuple:
    """Return (image, info_html, stats_html) for the current index."""
    if not MANIFEST:
        return None, "<p>No candidates loaded.</p>", "<p>No manifest.</p>"

    idx = _state["idx"]
    entry = MANIFEST[idx]
    fname = _img_fname(entry)
    img_path = IMAGE_DIR / fname

    label_val = _state["labels"].get(fname, entry.get("label"))

    info_lines = [f"**{entry.get('region', 'unknown')}**"]

    if _has_field(entry, "date"):
        info_lines.append(f"Date: {entry['date']}")
    if _has_field(entry, "shsi_mean"):
        info_lines.append(f"SHSI mean: {entry['shsi_mean']:.4f}")
        info_lines.append(f"SHSI max: {entry.get('shsi_max', 0):.4f}")
    if _has_field(entry, "cloud"):
        info_lines.append(f"Cloud: {entry['cloud']:.1f}%")
    if _has_field(entry, "lat") and _has_field(entry, "lon"):
        info_lines.append(f"Lat: {entry['lat']:.4f}, Lon: {entry['lon']:.4f}")
    if _has_field(entry, "scene_id"):
        info_lines.append(f"Scene: {entry['scene_id']}")
    if _has_field(entry, "source_path"):
        info_lines.append(f"Source: {entry['source_path']}")
    if _has_field(entry, "label_sources") and entry["label_sources"]:
        info_lines.append(f"Prior labels: {', '.join(entry['label_sources'])}")

    info_lines.append(f"File: {fname}")
    info_html = "<br>".join(info_lines)

    total = len(MANIFEST)
    n_labeled = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) in ("spawn", "no-spawn", "skip")
    )
    n_spawn = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) == "spawn"
    )
    n_nospawn = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) == "no-spawn"
    )
    n_skip = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) == "skip"
    )
    n_remaining = total - n_labeled

    if label_val == "spawn":
        status = '<span style="color:#2d6a4f;font-weight:bold">SPAWN</span>'
    elif label_val == "no-spawn":
        status = '<span style="color:#9b2226;font-weight:bold">NO SPAWN</span>'
    elif label_val == "skip":
        status = '<span style="color:#6b7280;font-weight:bold">SKIPPED</span>'
    else:
        status = '<span style="color:#f59e0b;font-weight:bold">UNLABELED</span>'

    stats_lines = [
        f"**{idx + 1} / {total}**",
        f"Status: {status}",
        f"",
        f"Spawn: {n_spawn}",
        f"No spawn: {n_nospawn}",
        f"Skipped: {n_skip}",
        f"Remaining: {n_remaining}",
    ]
    stats_html = "<br>".join(stats_lines)

    if img_path.exists():
        return str(img_path), info_html, stats_html
    return None, info_html + "<br><br>⚠️ Image not found", stats_html


def set_label(choice: str) -> tuple:
    """Label current image and advance."""
    if not MANIFEST:
        return get_current_state()

    entry = MANIFEST[_state["idx"]]
    fname = _img_fname(entry)
    _state["labels"][fname] = choice
    save_labels()

    # Advance to next unlabeled if available
    next_idx = _state["idx"] + 1
    while next_idx < len(MANIFEST):
        nf = _img_fname(MANIFEST[next_idx])
        if nf not in _state["labels"]:
            break
        next_idx += 1

    if next_idx < len(MANIFEST):
        _state["idx"] = next_idx

    return get_current_state()


def go_next() -> tuple:
    if not MANIFEST:
        return get_current_state()
    _state["idx"] = min(_state["idx"] + 1, len(MANIFEST) - 1)
    return get_current_state()


def go_prev() -> tuple:
    _state["idx"] = max(_state["idx"] - 1, 0)
    return get_current_state()


def go_to(idx_str: str) -> tuple:
    try:
        idx = int(idx_str) - 1
        _state["idx"] = max(0, min(idx, len(MANIFEST) - 1))
    except (ValueError, TypeError):
        pass
    return get_current_state()


def jump_to_next_unlabeled() -> tuple:
    for i in range(_state["idx"], len(MANIFEST)):
        fname = _img_fname(MANIFEST[i])
        if fname not in _state["labels"]:
            _state["idx"] = i
            return get_current_state()
    return get_current_state()


def jump_to_prev_unlabeled() -> tuple:
    for i in range(_state["idx"] - 1, -1, -1):
        fname = _img_fname(MANIFEST[i])
        if fname not in _state["labels"]:
            _state["idx"] = i
            return get_current_state()
    return get_current_state()


def export_labels() -> str:
    """Export labels as a summary JSON with timestamps."""
    if not _state["labels"]:
        return "No labels to export."

    export_path = LABELS_FILE.parent / f"labels_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    summary = {
        "exported_at": datetime.now().isoformat(),
        "manifest": str(IMAGE_DIR / "manifest.json"),
        "total_candidates": len(MANIFEST),
        "total_labeled": len(_state["labels"]),
        "spawn": sum(1 for v in _state["labels"].values() if v == "spawn"),
        "no_spawn": sum(1 for v in _state["labels"].values() if v == "no-spawn"),
        "skip": sum(1 for v in _state["labels"].values() if v == "skip"),
        "labels": _state["labels"],
    }
    export_path.write_text(json.dumps(summary, indent=2))
    return f"Exported to {export_path}"


def clear_current_label() -> tuple:
    """Remove label from current image."""
    if not MANIFEST:
        return get_current_state()
    entry = MANIFEST[_state["idx"]]
    fname = _img_fname(entry)
    if fname in _state["labels"]:
        del _state["labels"][fname]
        save_labels()
    return get_current_state()


# ===================================================================
# Grid scan view
# ===================================================================

GRID_PAGE_SIZE = 48
GRID_PAGE = 0


def _effective_label(entry: dict) -> str | None:
    """Get effective label for an entry (runtime overrides manifest)."""
    fname = _img_fname(entry)
    if fname in _state["labels"]:
        return _state["labels"][fname]
    return entry.get("label")


def grid_page_count() -> int:
    return max(1, (len(MANIFEST) + GRID_PAGE_SIZE - 1) // GRID_PAGE_SIZE)


def get_grid_page(page: int = 0) -> tuple:
    """Return (gallery_images_html, page_info_html) for the grid view."""
    global GRID_PAGE
    GRID_PAGE = max(0, min(page, grid_page_count() - 1))

    start = GRID_PAGE * GRID_PAGE_SIZE
    end = min(start + GRID_PAGE_SIZE, len(MANIFEST))
    page_entries = MANIFEST[start:end]

    cards: list[str] = []
    for i, entry in enumerate(page_entries):
        fname = _img_fname(entry)
        img_path = IMAGE_DIR / fname
        label = _effective_label(entry)
        global_idx = start + i

        if label == "spawn":
            border = "4px solid #2d6a4f"
            badge = '<div style="position:absolute;top:4px;right:4px;background:#2d6a4f;color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">SPAWN</div>'
        elif label == "no-spawn":
            border = "2px solid #9b2226"
            badge = '<div style="position:absolute;top:4px;right:4px;background:#9b2226;color:white;padding:2px 8px;border-radius:4px;font-size:12px">NO</div>'
        elif label == "skip":
            border = "2px solid #6b7280"
            badge = '<div style="position:absolute;top:4px;right:4px;background:#6b7280;color:white;padding:2px 8px;border-radius:4px;font-size:12px">SKIP</div>'
        else:
            border = "2px solid #e5e7eb"
            badge = ""

        img_url = f"/file={img_path.absolute()}" if img_path.exists() else ""

        region = entry.get("region", "?")
        date_str = entry.get("date", "")[:10] if entry.get("date") else ""

        cards.append(f"""
        <div style="position:relative;display:inline-block;margin:4px;border:{border};border-radius:6px;overflow:hidden;width:180px;height:180px">
            <button onclick="gridSpawn({global_idx})" style="position:absolute;inset:0;border:none;background:none;cursor:pointer;padding:0;width:100%;height:100%" title="Click=spawn, Right-click=no-spawn"
                    oncontextmenu="gridNoSpawn({global_idx});return false">
                <img src="{img_url}" style="width:100%;height:100%;object-fit:cover" loading="lazy">
            </button>
            {badge}
            <div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.6);color:white;padding:2px 6px;font-size:10px;display:flex;justify-content:space-between;pointer-events:none">
                <span>#{global_idx + 1} {region}</span>
                <span>{date_str}</span>
            </div>
        </div>""")

    gallery_html = f"""
    <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0;padding:8px">
        {''.join(cards)}
    </div>"""

    total = len(MANIFEST)
    n_spawn = sum(1 for m in MANIFEST if _effective_label(m) == "spawn")
    n_nospawn = sum(1 for m in MANIFEST if _effective_label(m) == "no-spawn")
    n_unlabeled = total - n_spawn - n_nospawn - sum(1 for m in MANIFEST if _effective_label(m) == "skip")

    page_info = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:#f9fafb;border-radius:8px;margin-bottom:8px">
        <span><b>Page {GRID_PAGE + 1} / {grid_page_count()}</b> · {start + 1}–{end} of {total}</span>
        <span style="color:#2d6a4f">✓ {n_spawn} spawn</span>
        <span style="color:#9b2226">✗ {n_nospawn} no-spawn</span>
        <span style="color:#f59e0b">{n_unlabeled} unlabeled</span>
    </div>"""

    return gallery_html, page_info


def grid_mark_spawn(click_data: str) -> tuple:
    """Mark the clicked image as spawn from the grid view."""
    global GRID_PAGE
    try:
        idx = int(click_data)
    except (ValueError, TypeError):
        return get_grid_page(GRID_PAGE)

    if 0 <= idx < len(MANIFEST):
        entry = MANIFEST[idx]
        fname = _img_fname(entry)
        _state["labels"][fname] = "spawn"
        save_labels()

    return get_grid_page(GRID_PAGE)


def grid_mark_nospawn(click_data: str) -> tuple:
    """Mark the clicked image as no-spawn."""
    global GRID_PAGE
    try:
        idx = int(click_data)
    except (ValueError, TypeError):
        return get_grid_page(GRID_PAGE)

    if 0 <= idx < len(MANIFEST):
        entry = MANIFEST[idx]
        fname = _img_fname(entry)
        _state["labels"][fname] = "no-spawn"
        save_labels()

    return get_grid_page(GRID_PAGE)


def grid_clear(click_data: str) -> tuple:
    """Clear label on clicked image."""
    global GRID_PAGE
    try:
        idx = int(click_data)
    except (ValueError, TypeError):
        return get_grid_page(GRID_PAGE)

    if 0 <= idx < len(MANIFEST):
        entry = MANIFEST[idx]
        fname = _img_fname(entry)
        if fname in _state["labels"]:
            del _state["labels"][fname]
            save_labels()

    return get_grid_page(GRID_PAGE)


def grid_prev_page() -> tuple:
    return get_grid_page(GRID_PAGE - 1)


def grid_next_page() -> tuple:
    return get_grid_page(GRID_PAGE + 1)


def grid_jump_to_unlabeled() -> tuple:
    """Jump to the first page containing an unlabeled image."""
    for i, entry in enumerate(MANIFEST):
        if _effective_label(entry) is None:
            return get_grid_page(i // GRID_PAGE_SIZE)
    return get_grid_page(0)


# ===================================================================
# Keyboard JS
# ===================================================================

KEYBOARD_JS = """
function() {
    window.gridSpawn = function(idx) {
        var el = document.querySelector('#grid_click_idx input, #grid_click_idx textarea');
        if (el) {
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(el, String(idx));
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        setTimeout(function() { document.querySelector('#grid_click_btn')?.click(); }, 30);
    };
    window.gridNoSpawn = function(idx) {
        var el = document.querySelector('#grid_nospawn_idx input, #grid_nospawn_idx textarea');
        if (el) {
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(el, String(idx));
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        setTimeout(function() { document.querySelector('#grid_nospawn_btn')?.click(); }, 30);
    };
    window.gridClear = function(idx) {
        var el = document.querySelector('#grid_clear_idx input, #grid_clear_idx textarea');
        if (el) {
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(el, String(idx));
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        setTimeout(function() { document.querySelector('#grid_clear_btn')?.click(); }, 30);
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
# CLI + App
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Herring Spawn Labeler") as demo:

        gr.Markdown("# 🐟 Herring Spawn Labeler")
        gr.Markdown(
            "**Y** = spawn | **N** = no spawn | **S** = skip | "
            "**←→** = navigate | **U** = next unlabeled | **C** = clear label<br>"
            "Grid: **click image** = spawn | **right-click** = no-spawn"
        )

        with gr.Tabs():
            # ---- Tab 1: Single Image Review ----
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column(scale=4):
                        image = gr.Image(
                            type="filepath",
                            height=620,
                            label=None,
                        )

                    with gr.Column(scale=2):
                        stats = gr.HTML(value="Loading...", elem_classes=["stats-panel"])
                        info = gr.HTML(value="", elem_classes=["info-panel"])

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

                        jump_input = gr.Textbox(
                            label="Go to #",
                            placeholder="e.g. 42",
                            max_lines=1,
                        )
                        jump_btn = gr.Button("Go", size="sm")

                        export_btn = gr.Button("📤 Export Labels", variant="secondary", size="sm")
                        export_msg = gr.Textbox(label="", show_label=False, interactive=False)

                # Wire up callbacks
                outputs = [image, info, stats]

                spawn_btn.click(lambda: set_label("spawn"), None, outputs)
                nospawn_btn.click(lambda: set_label("no-spawn"), None, outputs)
                skip_btn.click(lambda: set_label("skip"), None, outputs)
                next_btn.click(go_next, None, outputs)
                prev_btn.click(go_prev, None, outputs)
                next_unlabeled_btn.click(jump_to_next_unlabeled, None, outputs)
                clear_btn.click(clear_current_label, None, outputs)
                jump_btn.click(go_to, jump_input, outputs)
                export_btn.click(export_labels, None, export_msg)

                demo.load(get_current_state, None, outputs)

            # ---- Tab 2: Grid Quick Scan ----
            with gr.TabItem("Grid Scan"):
                gr.Markdown("**Click** any image to mark it as SPAWN. **Double-click** to mark NO-SPAWN.")

                grid_page_info = gr.HTML(value="Loading...")
                grid_gallery = gr.HTML(value="Loading...")

                with gr.Row():
                    grid_prev_btn = gr.Button("◀ Prev Page", size="sm")
                    grid_next_btn = gr.Button("Next Page ▶", size="sm")
                    grid_unlabeled_btn = gr.Button("⏩ Jump to Unlabeled", size="sm")

                # Hidden components for click handling
                grid_click_idx = gr.Textbox(visible=False, elem_id="grid_click_idx")
                grid_click_btn = gr.Button("_grid_spawn", visible=False, elem_id="grid_click_btn")
                grid_nospawn_idx = gr.Textbox(visible=False, elem_id="grid_nospawn_idx")
                grid_nospawn_btn = gr.Button("_grid_nospawn", visible=False, elem_id="grid_nospawn_btn")
                grid_clear_idx = gr.Textbox(visible=False, elem_id="grid_clear_idx")
                grid_clear_btn = gr.Button("_grid_clear", visible=False, elem_id="grid_clear_btn")

                grid_outputs = [grid_gallery, grid_page_info]

                grid_click_btn.click(grid_mark_spawn, grid_click_idx, grid_outputs)
                grid_nospawn_btn.click(grid_mark_nospawn, grid_nospawn_idx, grid_outputs)
                grid_clear_btn.click(grid_clear, grid_clear_idx, grid_outputs)
                grid_prev_btn.click(grid_prev_page, None, grid_outputs)
                grid_next_btn.click(grid_next_page, None, grid_outputs)
                grid_unlabeled_btn.click(grid_jump_to_unlabeled, None, grid_outputs)

                demo.load(lambda: get_grid_page(0), None, grid_outputs, show_progress=False)

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gradio labeling app for binary herring spawn classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest",
        default="data/candidates_shsi/manifest.json",
        help="Path to candidate manifest JSON",
    )
    parser.add_argument(
        "--image-dir",
        default="data/candidates_shsi",
        help="Directory containing thumbnail images",
    )
    parser.add_argument(
        "--labels",
        default="data/candidates_shsi/labels.json",
        help="Path to labels JSON file (created if missing)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7888,
        help="Port to serve on (default: 7888)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    args = parser.parse_args(argv)

    global MANIFEST, IMAGE_DIR, LABELS_FILE

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    IMAGE_DIR = repo_root / args.image_dir if not Path(args.image_dir).is_absolute() else Path(args.image_dir)
    LABELS_FILE = repo_root / args.labels if not Path(args.labels).is_absolute() else Path(args.labels)

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1
    if not IMAGE_DIR.exists():
        print(f"ERROR: Image directory not found: {IMAGE_DIR}")
        return 1

    MANIFEST = json.loads(manifest_path.read_text())
    if not isinstance(MANIFEST, list):
        print("ERROR: Manifest must be a JSON array")
        return 1
    if not MANIFEST:
        print("ERROR: Manifest is empty")
        return 1

    _state["labels"] = load_labels()
    _state["idx"] = 0

    # Jump to first unlabeled (checking both runtime and manifest labels)
    for i, entry in enumerate(MANIFEST):
        fname = _img_fname(entry)
        if fname not in _state["labels"] and entry.get("label") is None:
            _state["idx"] = i
            break

    n_labeled = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) in ("spawn", "no-spawn", "skip")
    )
    n_spawn = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) == "spawn"
    )
    n_nospawn = sum(
        1 for m in MANIFEST
        if _state["labels"].get(_img_fname(m), m.get("label")) == "no-spawn"
    )

    print(f"  Candidates: {len(MANIFEST)}")
    print(f"  Previously labeled: {n_labeled} ({n_spawn} spawn, {n_nospawn} no-spawn)")
    print(f"  Remaining: {len(MANIFEST) - n_labeled}")
    print(f"  Starting at: {_state['idx'] + 1} / {len(MANIFEST)}")
    print(f"\n  http://localhost:{args.port}")

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
        .info-panel { font-size: 15px; line-height: 1.6; }
        .stats-panel { font-size: 15px; line-height: 1.8; }
        """,
        js=KEYBOARD_JS,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

