"""Database models and initialization for the Herring Spawn Detection web app."""

import json
import glob
import os
import sqlite3
from datetime import datetime

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "webapp", "herring.db")
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "data", "candidates_v2")
OFFSEASON_DIR = os.path.join(CANDIDATES_DIR, "offseason")
MANIFEST_PATH = os.path.join(CANDIDATES_DIR, "manifest.json")

SEED_SPAWNS = [
    {"lat": 49.2349, "lon": -126.0266, "region": "Tofino", "confidence": "high",
     "notes": "Confirmed spawn near Tofino"},
    {"lat": 49.1349, "lon": -123.6766, "region": "Nanaimo", "confidence": "high",
     "notes": "Confirmed spawn near Nanaimo"},
    {"lat": 49.5849, "lon": -126.6085, "region": "Nootka Sound", "confidence": "high",
     "notes": "Confirmed spawn in Nootka Sound"},
    {"lat": 49.5849, "lon": -126.6285, "region": "Nootka Sound", "confidence": "high",
     "notes": "Confirmed spawn in Nootka Sound (second site)"},
]


def get_db():
    """Get a database connection, creating it if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS spawn_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            region TEXT NOT NULL,
            first_detected TEXT,
            confidence TEXT DEFAULT 'medium',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            ended_at TEXT,
            grid_points INTEGER DEFAULT 0,
            candidates_found INTEGER DEFAULT 0,
            status TEXT DEFAULT 'done'
        );

        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            region TEXT NOT NULL,
            spawn_score REAL,
            off_score REAL,
            delta REAL,
            thumbnail_path TEXT,
            off_thumbnail_path TEXT,
            scene_id TEXT,
            date TEXT,
            cloud REAL,
            user_label TEXT DEFAULT 'unlabeled',
            labeled_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        );

        CREATE INDEX IF NOT EXISTS idx_candidates_label ON candidates(user_label);
        CREATE INDEX IF NOT EXISTS idx_candidates_region ON candidates(region);
        CREATE INDEX IF NOT EXISTS idx_candidates_delta ON candidates(delta);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_unique
            ON candidates(lat, lon, date, thumbnail_path);
    """)
    conn.commit()
    conn.close()


def _normalize_lat_lon(lat, lon):
    """Create lat/lon strings for matching off-season filenames."""
    lat_str = f"{lat:.6f}".replace(".", "_")
    lon_str = f"{abs(lon):.6f}".replace(".", "_")
    return lat_str, lon_str


def _find_offseason_image(region, lat, lon):
    """Find matching off-season image for a given candidate location."""
    lat_str, lon_str = _normalize_lat_lon(lat, lon)
    pattern = os.path.join(OFFSEASON_DIR, f"{region}_{lat_str}__{lon_str}_off_*.png")
    matches = glob.glob(pattern)
    if matches:
        return os.path.relpath(matches[0], CANDIDATES_DIR)
    return None


def _parse_offseason_score(image_path):
    """Extract score from a spawn-season candidate thumbnail filename."""
    # Off-season images don't have a score in the filename
    # We'll leave it as NULL for now
    return None


def import_candidates():
    """Import candidates from manifest.json into the database."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 0

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    conn = get_db()
    imported = 0

    # Create a default scan for these historical candidates
    cur = conn.execute("SELECT id FROM scans ORDER BY id DESC LIMIT 1")
    scan = cur.fetchone()
    if scan:
        scan_id = scan["id"]
    else:
        conn.execute(
            "INSERT INTO scans (started_at, ended_at, grid_points, candidates_found, status) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2024-03-01", "2024-04-15", len(manifest), 0, "done"),
        )
        scan_id = cur.lastrowid or 1

    for entry in manifest:
        lat = entry["lat"]
        lon = entry["lon"]
        region = entry["region"]
        date = entry.get("date", "")
        thumbnail = entry.get("thumbnail_path", "")

        # Find matching off-season image
        off_thumbnail = _find_offseason_image(region, lat, lon) or ""

        # Check if candidate already exists
        existing = conn.execute(
            "SELECT id FROM candidates WHERE lat=? AND lon=? AND date=? AND thumbnail_path=?",
            (lat, lon, date, thumbnail),
        ).fetchone()

        if existing:
            continue

        score = entry.get("score", 0)
        scene_id = entry.get("scene_id", "")

        conn.execute(
            "INSERT INTO candidates (scan_id, lat, lon, region, spawn_score, "
            "off_score, delta, thumbnail_path, off_thumbnail_path, scene_id, date, cloud, user_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scan_id, lat, lon, region, score, None, None,
             thumbnail, off_thumbnail, scene_id, date, entry.get("cloud"), "unlabeled"),
        )
        imported += 1

    conn.commit()

    # Update scan count
    cur = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE scan_id=?", (scan_id,))
    total = cur.fetchone()["cnt"]
    conn.execute("UPDATE scans SET candidates_found=? WHERE id=?", (total, scan_id))
    conn.commit()
    conn.close()

    return imported


def seed_spawn_events():
    """Insert seed spawn events if none exist."""
    conn = get_db()
    cur = conn.execute("SELECT COUNT(*) as cnt FROM spawn_events")
    if cur.fetchone()["cnt"] > 0:
        conn.close()
        return 0

    for spawn in SEED_SPAWNS:
        conn.execute(
            "INSERT INTO spawn_events (lat, lon, region, first_detected, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (spawn["lat"], spawn["lon"], spawn["region"],
             "2024-03-16", spawn["confidence"], spawn["notes"]),
        )

    conn.commit()
    conn.close()
    return len(SEED_SPAWNS)
