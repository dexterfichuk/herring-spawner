import sqlite3

from webapp import app as webapp_app
from webapp import models


def test_find_offseason_image_matches_lat_lon(tmp_path, monkeypatch):
    off_dir = tmp_path / "off_season"
    off_dir.mkdir()
    expected = off_dir / "tofino_49.2349_-126.0266_off.png"
    expected.write_bytes(b"png")

    monkeypatch.setattr(models, "CANDIDATES_DIR", str(tmp_path))
    monkeypatch.setattr(models, "OFFSEASON_DIR", str(off_dir))

    rel = models._find_offseason_image("wrong-region", 49.23492, -126.02658)

    assert rel == "off_season/tofino_49.2349_-126.0266_off.png"


def test_delta_candidates_api_returns_unlabeled_sorted_by_delta(tmp_path, monkeypatch):
    db_path = tmp_path / "herring.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE candidates (
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
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO candidates
            (lat, lon, region, spawn_score, off_score, delta, thumbnail_path, off_thumbnail_path, user_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (49.2, -126.0, "A", 0.9, 0.1, None, "spawn-a.png", "off-a.png", "unlabeled"),
            (49.3, -126.1, "B", 0.4, 0.05, 0.2, "spawn-b.png", "off-b.png", "unlabeled"),
            (49.4, -126.2, "C", 0.8, 0.0, 0.8, "spawn-c.png", "off-c.png", "spawn"),
        ],
    )
    conn.commit()
    conn.close()

    def fake_get_db():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(webapp_app, "get_db", fake_get_db)

    client = webapp_app.app.test_client()
    response = client.get("/api/delta-candidates")

    assert response.status_code == 200
    data = response.get_json()
    assert [c["id"] for c in data["candidates"]] == [1, 2]
    assert data["candidates"][0]["delta"] == 0.8
    assert data["candidates"][0]["thumbnail_path"] == "spawn-a.png"
    assert data["candidates"][0]["off_thumbnail_path"] == "off-a.png"
