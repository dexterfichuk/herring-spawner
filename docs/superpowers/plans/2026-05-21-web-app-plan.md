# Herring Spawn Detection Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan.

**Goal:** Build a lightweight web application for continuous BC coast scanning, interactive candidate review, and spawn event tracking.

**Architecture:** Flask web app (simple Python, no heavy framework) with SQLite for persistence. Frontend is server-rendered HTML + vanilla JS for interactivity (no npm/build step). Background scans triggered via CLI or web endpoint.

**Tech Stack:** Python 3.11+, Flask, SQLite, Leaflet.js (maps), DINOv2 + SVM for detection

## File Structure

- `webapp/app.py` — Flask application with routes and API
- `webapp/templates/` — Jinja2 HTML templates
- `webapp/static/` — CSS, JS files
- `webapp/models.py` — SQLite database models
- `webapp/scanner.py` — Scan orchestrator (wraps scan_bc_coast.py)
- `data/webapp/` — Database and scan results storage

## Database Schema (SQLite)

### `spawn_events`
- id, lat, lon, region, first_detected (date), last_confirmed (date), confidence (high/medium/low), notes

### `scans`
- id, started_at, ended_at, grid_points, candidates_found, status (running/done/failed)

### `candidates`
- id, scan_id, lat, lon, region, spawn_score, off_score, delta, thumbnail_path, off_thumbnail_path, scene_id, date
- user_label (unlabeled/spawn/nospawn/cloudy), labeled_at

## MVP Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Dashboard — latest scan summary, map of confirmed spawns |
| `/scans` | GET | List all scans with stats |
| `/scans/<id>` | GET | Review candidates from a specific scan |
| `/candidates` | GET | Review all unlabeled candidates, sorted by delta |
| `/api/candidates/<id>/label` | POST | Label a candidate (spawn/nospawn/cloudy) |
| `/api/spawn-events` | GET | Get all confirmed spawn events as GeoJSON |
| `/scan/start` | POST | Trigger a new scan |
| `/positives` | GET | Gallery of all confirmed spawn images |

## UI Pages

### Dashboard (`/`)
- Map (Leaflet) showing all confirmed spawn events
- Latest scan summary stats
- Quick-action buttons

### Candidate Review (`/candidates`)
- Same delta side-by-side view as current delta_review.html
- Click buttons to label: Spawn / No Spawn / Cloudy
- Filter by label status, region, score range
- Sort by delta, score, date

### Scan Detail (`/scans/<id>`)
- Scan parameters and stats
- All candidates from that scan

### Positive Gallery (`/positives`)
- Grid of all images labeled "spawn"
- Click to enlarge, remove label, add notes

## Implementation Phases

### Phase 1: Basic Flask App + Database
- Set up Flask app with SQLite
- Import existing candidates into DB
- Serve delta review page
- Label API (mark spawn/nospawn)

### Phase 2: Map + Scan Triggering
- Leaflet.js map of confirmed spawns
- Scan trigger endpoint
- Background scan with status tracking

### Phase 3: Continuous Scanning
- Automated periodic scans
- Delta-based detection logic
- Push notifications for new high-confidence candidates

## Current Confirmed Spawns (Seed Data from User)
- Tofino: 49.2349, -126.0266
- Nanaimo: 49.1349, -123.6766
- Nootka Sound: 49.5849, -126.6085
- Nootka Sound: 49.5849, -126.6285

## Notes
- Clay question: Clay multi-spectral could help distinguish seasonal water color changes from actual spawn, since NIR/SWIR bands are less affected by turbidity/seasonality than RGB. Worth testing on the confirmed positives once we have enough labeled GeoTIFF chips.
- The key insight from the 77% FP rate is that **delta-based detection** (spawn-season vs off-season) is essential. The web app should compute and display delta automatically.
