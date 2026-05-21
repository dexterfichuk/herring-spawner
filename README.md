# Herring Spawner

Research prototype for detecting Pacific herring spawn in BC/PNW satellite imagery.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Google Earth Engine

The default Earth Engine project is `redd-fish`.

Authenticate locally before running GEE-backed scripts:

```bash
earthengine authenticate
```

## First Workflow

1. Build the event catalog from DFO, manual April 2026 points, and local tracks.
2. Search Sentinel-2 scenes around known events.
3. Export thumbnails/chip metadata for review.
4. Compute spectral features and Clay embeddings.
5. Review nearest-neighbor candidate results before making detection claims.
