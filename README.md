# Herring Spawner

Research prototype for detecting Pacific herring spawn in BC/PNW satellite imagery.

## Labeled Dataset (243 images, 136 spawn / 107 no-spawn)

`labeled_locations.csv` contains every location with a human-verified label (spawn/no-spawn).
No image files are stored in git — the CSV has all metadata needed to reproduce them.

### Reproduce the images

```bash
pip install pystac-client planetary-computer rasterio Pillow
python scripts/fetch_from_csv.py labeled_locations.csv --output-dir ./spawn_images
```

This downloads the exact same 512×512 true-color Sentinel-2 crops from Microsoft Planetary Computer.

### Label new images

```bash
# Single-image review (what was used for the 243 labels):
python -m pip install gradio
python scripts/label_app.py

# Or re-rank unlabeled candidates by visual similarity to your labels:
python scripts/rank_georsclip.py
python scripts/label_app.py
```

## Setup

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

## Smoke Workflow

Build local event and scene-search request artifacts:

```bash
python scripts/build_event_catalog.py \
  --output data/interim/events.geojson \
  --track-root /Users/dexterfichuk/Downloads/2025\ Tracks

python scripts/search_known_events.py \
  --events data/interim/events.geojson \
  --output data/interim/scene_search_requests.json
```

The generated files are ignored by git. Review them locally before running GEE-backed exports.

## Generated Dataset Storage

Image-heavy generated candidate assets live in the public Hugging Face dataset:
https://huggingface.co/datasets/dfichuk/herring-spawn-candidates

GitHub should keep code, docs, tests, labels, manifests, and lightweight model summaries. Upload regenerated candidate imagery and review assets with:

```bash
source .venv/bin/activate
python -m pip install huggingface_hub
huggingface-cli login
python scripts/upload_hf_dataset.py --repo-id dfichuk/herring-spawn-candidates
```

## Running Tests

```bash
source .venv/bin/activate
pytest -v
ruff check .
```

## Earth Engine Connectivity Check

```bash
source .venv/bin/activate
python - <<'PY'
import ee
ee.Initialize(project="redd-fish")
print(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).size().getInfo())
PY
```

Expected output is `1`. If authentication fails, run `earthengine authenticate` and retry.
