"""Herring Spawn Detection Web App — Phase 1.

Flask web application for reviewing candidates, marking spawns,
and tracking confirmed events.
"""

import os
import sys
import json
from datetime import datetime

from flask import Flask, render_template, request, jsonify, abort, send_from_directory

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.models import init_db, import_candidates, seed_spawn_events, get_db

app = Flask(__name__)

# Path to candidates_v2 for serving thumbnails
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "data", "candidates_v2")


# ── Static file serving for candidate thumbnails ──────────────────────────

@app.route("/static/candidates/<path:filename>")
def candidates_static(filename):
    """Serve files from the candidates_v2 directory."""
    return send_from_directory(CANDIDATES_DIR, filename)


def cand_url(thumbnail_path):
    """Build a URL for a candidate thumbnail."""
    if not thumbnail_path:
        return ""
    return "/static/candidates/" + thumbnail_path


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Dashboard with map, stats, and quick actions."""
    db = get_db()

    cur = db.execute("SELECT COUNT(*) as cnt FROM candidates")
    total_candidates = cur.fetchone()["cnt"]

    cur = db.execute("SELECT COUNT(*) as cnt FROM candidates WHERE user_label='spawn'")
    total_spawn = cur.fetchone()["cnt"]

    cur = db.execute("SELECT COUNT(*) as cnt FROM candidates WHERE user_label='nospawn'")
    total_nospawn = cur.fetchone()["cnt"]

    cur = db.execute(
        "SELECT COUNT(*) as cnt FROM candidates WHERE user_label='unlabeled'"
    )
    unlabeled_count = cur.fetchone()["cnt"]

    cur = db.execute("SELECT DISTINCT region FROM candidates ORDER BY region")
    regions = [r["region"] for r in cur.fetchall()]

    # Build GeoJSON for confirmed spawn events
    features = []
    cur = db.execute("SELECT * FROM spawn_events ORDER BY region")
    for row in cur.fetchall():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {
                "region": row["region"],
                "first_detected": row["first_detected"],
                "confidence": row["confidence"],
                "notes": row["notes"],
            },
        })
    spawn_geojson = json.dumps({"type": "FeatureCollection", "features": features})

    db.close()

    return render_template(
        "dashboard.html",
        total_candidates=total_candidates,
        total_spawn=total_spawn,
        total_nospawn=total_nospawn,
        unlabeled_count=unlabeled_count,
        regions=regions,
        spawn_geojson=spawn_geojson,
    )


@app.route("/candidates")
def candidates_list():
    """List all candidates with filtering and sorting."""
    db = get_db()

    # Build query
    where_clauses = []
    params = []

    region = request.args.get("region", "")
    label = request.args.get("label", "")
    min_delta = request.args.get("min_delta", "")

    if region:
        where_clauses.append("region = ?")
        params.append(region)
    if label:
        where_clauses.append("user_label = ?")
        params.append(label)
    if min_delta:
        try:
            md = float(min_delta)
            where_clauses.append("COALESCE(delta, spawn_score) >= ?")
            params.append(md)
        except ValueError:
            pass

    # Sorting
    sort_map = {
        "delta_desc": "COALESCE(delta, spawn_score) DESC",
        "delta_asc": "COALESCE(delta, spawn_score) ASC",
        "score_desc": "spawn_score DESC",
        "score_asc": "spawn_score ASC",
    }
    sort_by = request.args.get("sort", "delta_desc")
    order_clause = sort_map.get(sort_by, sort_map["delta_desc"])

    where = ""
    if where_clauses:
        where = "WHERE " + " AND ".join(where_clauses)

    query = f"SELECT * FROM candidates {where} ORDER BY {order_clause}"
    cur = db.execute(query, params)
    candidates = [dict(row) for row in cur.fetchall()]

    db.close()

    return render_template(
        "candidates.html",
        candidates=candidates,
        regions=sorted(set(c["region"] for c in candidates)),
        candidates_static="/static/candidates/",
        request=request,
    )


@app.route("/positives")
def positives():
    """Gallery of all images labeled 'spawn'."""
    db = get_db()
    cur = db.execute(
        "SELECT * FROM candidates WHERE user_label='spawn' ORDER BY region, date DESC"
    )
    positives = [dict(row) for row in cur.fetchall()]
    db.close()

    return render_template(
        "positives.html",
        positives=positives,
        candidates_static="/static/candidates/",
    )


@app.route("/scans")
def scans_list():
    """List all scans."""
    db = get_db()
    cur = db.execute("SELECT * FROM scans ORDER BY id DESC")
    scans = [dict(row) for row in cur.fetchall()]
    db.close()

    return render_template("scans.html", scans=scans)


@app.route("/scans/<int:scan_id>")
def scan_detail(scan_id):
    """View candidates from a specific scan."""
    db = get_db()
    cur = db.execute("SELECT * FROM scans WHERE id=?", (scan_id,))
    scan = cur.fetchone()
    if not scan:
        abort(404)

    cur = db.execute(
        "SELECT * FROM candidates WHERE scan_id=? ORDER BY COALESCE(delta, spawn_score) DESC",
        (scan_id,),
    )
    candidates = [dict(row) for row in cur.fetchall()]
    db.close()

    return render_template(
        "scan_detail.html",
        scan=dict(scan),
        candidates=candidates,
        candidates_static="/static/candidates/",
    )


# ── API Routes ────────────────────────────────────────────────────────────

@app.route("/api/candidates/<int:candidate_id>/label", methods=["POST"])
def label_candidate(candidate_id):
    """Label a candidate and optionally create/update a spawn event."""
    data = request.get_json()
    if not data or "label" not in data:
        return jsonify({"status": "error", "message": "Missing 'label' field"}), 400

    label = data["label"]
    valid_labels = {"spawn", "nospawn", "cloudy", "unknown", "unlabeled"}
    if label not in valid_labels:
        return jsonify({"status": "error", "message": f"Invalid label: {label}"}), 400

    db = get_db()

    # Update candidate
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    db.execute(
        "UPDATE candidates SET user_label=?, labeled_at=? WHERE id=?",
        (label, now, candidate_id),
    )

    if label == "spawn":
        # Get candidate details
        cur = db.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,))
        candidate = cur.fetchone()
        if candidate:
            # Check if a spawn event already exists at this location
            cur = db.execute(
                "SELECT id FROM spawn_events WHERE ABS(lat-?) < 0.01 AND ABS(lon-?) < 0.01",
                (candidate["lat"], candidate["lon"]),
            )
            existing = cur.fetchone()

            if existing:
                # Update last_confirmed (just update notes)
                db.execute(
                    "UPDATE spawn_events SET notes=?, confidence='high' WHERE id=?",
                    (f"Re-confirmed from candidate #{candidate_id}", existing["id"]),
                )
            else:
                # Create new spawn event
                db.execute(
                    "INSERT INTO spawn_events (lat, lon, region, first_detected, confidence, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (candidate["lat"], candidate["lon"], candidate["region"],
                     candidate["date"] or now[:10], "high",
                     f"From candidate #{candidate_id}"),
                )

    db.commit()
    db.close()

    return jsonify({"status": "ok", "label": label})


@app.route("/api/spawn-events")
def spawn_events_api():
    """Return all confirmed spawn events as GeoJSON."""
    db = get_db()
    features = []
    cur = db.execute("SELECT * FROM spawn_events ORDER BY region")
    for row in cur.fetchall():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {
                "id": row["id"],
                "region": row["region"],
                "first_detected": row["first_detected"],
                "confidence": row["confidence"],
                "notes": row["notes"],
                "created_at": row["created_at"],
            },
        })
    db.close()

    return jsonify({"type": "FeatureCollection", "features": features})


# ── App initialization ────────────────────────────────────────────────────

def initialize():
    """Set up the database and import data on first run."""
    init_db()
    seeded = seed_spawn_events()
    if seeded:
        print(f"  Seeded {seeded} spawn events")
    imported = import_candidates()
    if imported:
        print(f"  Imported {imported} new candidates")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing Herring Spawn Detection Web App...")
    initialize()
    print()
    print("  Running on http://localhost:5050")
    print()
    app.run(host="0.0.0.0", port=5050, debug=True)
