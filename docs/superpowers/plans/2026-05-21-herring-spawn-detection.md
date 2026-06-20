# Herring Spawn Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first notebook-friendly research prototype for detecting BC herring spawn from Sentinel-2 imagery using event catalogs, visual QA thumbnails, spectral features, and Clay-ready embedding boundaries.

**Architecture:** Implement a small Python package with reusable modules for datasets, imagery providers, chip metadata, spectral features, embeddings, and review outputs. Notebooks or scripts should orchestrate these modules, but core logic should be importable and covered by tests. Google Earth Engine is the first imagery provider using project ID `redd-fish`; the provider interface remains STAC-shaped for later direct API support.

**Tech Stack:** Python 3.11+, pytest, pandas, geopandas, shapely, fastkml, gpxpy, numpy, scikit-learn, earthengine-api, jinja2, pyarrow, ruff.

---

## File Structure

- Create: `pyproject.toml` for packaging, dependencies, pytest, and ruff configuration.
- Create: `.gitignore` for generated data, credentials, caches, and local artifacts.
- Create: `README.md` with setup, GEE authentication, and first prototype workflow.
- Create: `herring_spawner/__init__.py` for package metadata.
- Create: `herring_spawner/config.py` for runtime paths and GEE project defaults.
- Create: `herring_spawner/models.py` for dataclasses shared across modules.
- Create: `herring_spawner/datasets/manual.py` for April 2026 known events.
- Create: `herring_spawner/datasets/tracks.py` for KML/KMZ/GPX AOI extraction.
- Create: `herring_spawner/datasets/dfo.py` for DFO spawn-index ingestion.
- Create: `herring_spawner/imagery/base.py` for provider-neutral scene interfaces.
- Create: `herring_spawner/imagery/gee.py` for Sentinel-2 GEE search and chip-export request construction.
- Create: `herring_spawner/chips/catalog.py` for chip catalog serialization.
- Create: `herring_spawner/features/spectral.py` for interpretable spectral features.
- Create: `herring_spawner/embeddings/search.py` for local vector search over Clay embeddings.
- Create: `herring_spawner/review/static.py` for static HTML and GeoJSON review artifacts.
- Create: `scripts/build_event_catalog.py` for generating the first event/AOI catalog.
- Create: `scripts/search_known_events.py` for searching GEE scenes for catalog events.
- Create: tests matching each module under `tests/`.

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `herring_spawner/__init__.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_imports.py`:

```python
def test_package_imports():
    import herring_spawner

    assert herring_spawner.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_imports.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'herring_spawner'`.

- [ ] **Step 3: Create project package files**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "herring-spawner"
version = "0.1.0"
description = "Research prototype for detecting Pacific herring spawn from satellite imagery."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "earthengine-api>=0.1.395",
  "fastkml>=0.12",
  "geopandas>=0.14",
  "gpxpy>=1.6",
  "jinja2>=3.1",
  "numpy>=1.26",
  "pandas>=2.2",
  "pyarrow>=15.0",
  "scikit-learn>=1.4",
  "shapely>=2.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Create `.gitignore`:

```gitignore
.DS_Store
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.egg-info/

data/raw/
data/interim/
data/processed/
data/exports/
data/embeddings/
data/review/

.env
*.jsonl
*.tif
*.tiff
*.parquet
*.gpkg
.superpowers/
```

Create `README.md`:

```markdown
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
```

Create `herring_spawner/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_imports.py -v`

Expected: PASS.

- [ ] **Step 5: Run formatting/lint smoke check**

Run: `python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md herring_spawner/__init__.py tests/test_imports.py
git commit -m "chore: scaffold herring spawn prototype"
```

---

### Task 2: Shared Configuration and Models

**Files:**
- Create: `herring_spawner/config.py`
- Create: `herring_spawner/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for config and models**

Create `tests/test_models.py`:

```python
from datetime import date

from shapely.geometry import Point, mapping

from herring_spawner.config import Settings
from herring_spawner.models import Event, LabelConfidence, SourceType


def test_default_settings_use_redd_fish_project():
    settings = Settings()

    assert settings.gee_project == "redd-fish"
    assert settings.data_dir.name == "data"


def test_event_serializes_geometry_and_dates():
    event = Event(
        event_id="manual-2026-04-04-turnour-1",
        source_type=SourceType.MANUAL,
        label="known_spawn",
        label_confidence=LabelConfidence.HIGH,
        start_date=date(2026, 4, 4),
        end_date=date(2026, 4, 4),
        geometry=Point(-126.192323333333, 50.8254366666667),
        source="user-provided April 4 2026 points",
        properties={"region": "Turnour area"},
    )

    row = event.to_record()

    assert row["event_id"] == "manual-2026-04-04-turnour-1"
    assert row["start_date"] == "2026-04-04"
    assert row["end_date"] == "2026-04-04"
    assert row["geometry"] == mapping(Point(-126.192323333333, 50.8254366666667))
    assert row["label_confidence"] == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.config` or `herring_spawner.models`.

- [ ] **Step 3: Implement config and models**

Create `herring_spawner/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    gee_project: str = "redd-fish"
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    exports_dir: Path = Path("data/exports")
    review_dir: Path = Path("data/review")
```

Create `herring_spawner/models.py`:

```python
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping


class SourceType(StrEnum):
    DFO = "dfo"
    MANUAL = "manual"
    TRACK = "track"
    NEWS = "news"


class LabelConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Event:
    event_id: str
    source_type: SourceType
    label: str
    label_confidence: LabelConfidence
    start_date: date | None
    end_date: date | None
    geometry: BaseGeometry
    source: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_type": self.source_type.value,
            "label": self.label,
            "label_confidence": self.label_confidence.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "geometry": mapping(self.geometry),
            "source": self.source,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class Scene:
    scene_id: str
    provider: str
    collection: str
    acquired: date
    cloud_score: float | None
    geometry: BaseGeometry
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chip:
    chip_id: str
    event_id: str
    scene_id: str
    acquired: date
    geometry: BaseGeometry
    bands: tuple[str, ...]
    asset_path: str | None
    thumbnail_path: str | None
    properties: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/config.py herring_spawner/models.py tests/test_models.py
git commit -m "feat: add shared catalog models"
```

---

### Task 3: Manual April 2026 Event Catalog

**Files:**
- Create: `herring_spawner/datasets/__init__.py`
- Create: `herring_spawner/datasets/manual.py`
- Test: `tests/datasets/test_manual.py`

- [ ] **Step 1: Write failing tests for manual events**

Create `tests/datasets/test_manual.py`:

```python
from datetime import date

from herring_spawner.datasets.manual import load_manual_events


def test_load_manual_april_2026_events():
    events = load_manual_events()

    assert len(events) == 4
    assert {event.start_date for event in events} == {date(2026, 4, 4)}
    assert {event.label for event in events} == {"known_spawn"}
    assert events[0].geometry.x == -126.192323333333
    assert events[0].geometry.y == 50.8254366666667
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/datasets/test_manual.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'herring_spawner.datasets'`.

- [ ] **Step 3: Implement manual event loader**

Create `herring_spawner/datasets/__init__.py`:

```python
"""Dataset ingestion helpers."""
```

Create `herring_spawner/datasets/manual.py`:

```python
from datetime import date

from shapely.geometry import Point

from herring_spawner.models import Event, LabelConfidence, SourceType


APRIL_2026_POINTS = [
    ("event-1-point-1", 50.8254366666667, -126.192323333333),
    ("event-1-point-2", 50.8262033333333, -126.19123),
    ("event-2-point-1", 50.824935, -126.192928333333),
    ("event-2-point-2", 50.82505, -126.19266),
]


def load_manual_events() -> list[Event]:
    spawn_date = date(2026, 4, 4)
    return [
        Event(
            event_id=f"manual-2026-04-04-{name}",
            source_type=SourceType.MANUAL,
            label="known_spawn",
            label_confidence=LabelConfidence.HIGH,
            start_date=spawn_date,
            end_date=spawn_date,
            geometry=Point(lon, lat),
            source="user-provided April 4 2026 herring spawn points",
            properties={"original_name": name, "region_hint": "BC central coast"},
        )
        for name, lat, lon in APRIL_2026_POINTS
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/datasets/test_manual.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/datasets/__init__.py herring_spawner/datasets/manual.py tests/datasets/test_manual.py
git commit -m "feat: add manual spawn event catalog"
```

---

### Task 4: Local Track AOI Parser

**Files:**
- Create: `herring_spawner/datasets/tracks.py`
- Test: `tests/datasets/test_tracks.py`

- [ ] **Step 1: Write failing tests for KML coordinate parsing**

Create `tests/datasets/test_tracks.py`:

```python
from pathlib import Path

from herring_spawner.datasets.tracks import load_track_aois


def test_load_kml_linestring_as_candidate_aoi(tmp_path: Path):
    kml_path = tmp_path / "sample.kml"
    kml_path.write_text(
        """<?xml version="1.0"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>segment 1</name>
      <LineString>
        <coordinates>-126.1,50.1,0 -126.2,50.2,0</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
""",
        encoding="utf-8",
    )

    events = load_track_aois([kml_path], month_label="July 2025")

    assert len(events) == 1
    assert events[0].event_id == "track-july-2025-sample-0001"
    assert events[0].label == "candidate_aoi"
    assert events[0].label_confidence == "low"
    assert events[0].geometry.length > 0
    assert events[0].properties["month_label"] == "July 2025"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/datasets/test_tracks.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.datasets.tracks`.

- [ ] **Step 3: Implement KML/KMZ/GPX track loading**

Create `herring_spawner/datasets/tracks.py`:

```python
from datetime import date
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import gpxpy
from shapely.geometry import LineString

from herring_spawner.models import Event, LabelConfidence, SourceType


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def load_track_aois(paths: list[Path], month_label: str | None = None) -> list[Event]:
    events: list[Event] = []
    for path in paths:
        geometries = _load_geometries(path)
        slug = _slug(path.stem)
        month_slug = _slug(month_label or "unknown-date")
        for index, geometry in enumerate(geometries, start=1):
            events.append(
                Event(
                    event_id=f"track-{month_slug}-{slug}-{index:04d}",
                    source_type=SourceType.TRACK,
                    label="candidate_aoi",
                    label_confidence=LabelConfidence.LOW,
                    start_date=None,
                    end_date=None,
                    geometry=geometry,
                    source=str(path),
                    properties={"month_label": month_label, "file_name": path.name},
                )
            )
    return events


def _load_geometries(path: Path) -> list[LineString]:
    suffix = path.suffix.lower()
    if suffix == ".kml":
        return _load_kml(path.read_text(encoding="utf-8"))
    if suffix == ".kmz":
        with ZipFile(path) as archive:
            with archive.open("doc.kml") as file:
                return _load_kml(file.read().decode("utf-8"))
    if suffix == ".gpx":
        return _load_gpx(path.read_text(encoding="utf-8"))
    return []


def _load_kml(text: str) -> list[LineString]:
    root = ElementTree.fromstring(text)
    geometries: list[LineString] = []
    for coordinates in root.findall(".//kml:LineString/kml:coordinates", KML_NS):
        points = []
        for token in (coordinates.text or "").split():
            lon, lat, *_ = token.split(",")
            points.append((float(lon), float(lat)))
        if len(points) >= 2:
            geometries.append(LineString(points))
    return geometries


def _load_gpx(text: str) -> list[LineString]:
    parsed = gpxpy.parse(text)
    geometries: list[LineString] = []
    for track in parsed.tracks:
        for segment in track.segments:
            points = [(point.longitude, point.latitude) for point in segment.points]
            if len(points) >= 2:
                geometries.append(LineString(points))
    return geometries


def _slug(value: str) -> str:
    return "-".join(value.lower().replace("_", "-").split())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/datasets/test_tracks.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS. If ruff flags the unused `date` import, remove `from datetime import date` from `tracks.py` and rerun.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/datasets/tracks.py tests/datasets/test_tracks.py
git commit -m "feat: parse local track candidate aois"
```

---

### Task 5: DFO Spawn Index Ingestion

**Files:**
- Create: `herring_spawner/datasets/dfo.py`
- Test: `tests/datasets/test_dfo.py`

- [ ] **Step 1: Write failing tests for DFO row normalization**

Create `tests/datasets/test_dfo.py`:

```python
from datetime import date

import pandas as pd

from herring_spawner.datasets.dfo import events_from_dataframe


def test_events_from_dfo_dataframe_with_known_columns():
    frame = pd.DataFrame(
        [
            {
                "Location": "Qualicum Beach",
                "Latitude": 49.355704,
                "Longitude": -124.456910,
                "StartDate": "2024-03-13",
                "EndDate": "2024-03-15",
                "Length": 5700,
                "Width": 199,
            }
        ]
    )

    events = events_from_dataframe(frame, source="unit-test")

    assert len(events) == 1
    assert events[0].event_id == "dfo-2024-03-13-qualicum-beach"
    assert events[0].start_date == date(2024, 3, 13)
    assert events[0].end_date == date(2024, 3, 15)
    assert events[0].label == "known_spawn"
    assert events[0].properties["spawn_length_m"] == 5700
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/datasets/test_dfo.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.datasets.dfo`.

- [ ] **Step 3: Implement DFO normalization**

Create `herring_spawner/datasets/dfo.py`:

```python
from datetime import date
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

from herring_spawner.models import Event, LabelConfidence, SourceType


def load_dfo_csv(path: Path) -> list[Event]:
    return events_from_dataframe(pd.read_csv(path), source=str(path))


def events_from_dataframe(frame: pd.DataFrame, source: str) -> list[Event]:
    events: list[Event] = []
    for row in frame.to_dict(orient="records"):
        lat = _first(row, "Latitude", "LATITUDE", "lat")
        lon = _first(row, "Longitude", "LONGITUDE", "lon", "Long")
        start = _parse_date(_first(row, "StartDate", "START_DATE", "Start Date", "start_date"))
        end = _parse_date(_first(row, "EndDate", "END_DATE", "End Date", "end_date")) or start
        location = str(_first(row, "Location", "LOCATION", "LocationName", "location") or "unknown")
        if lat is None or lon is None or start is None:
            continue
        events.append(
            Event(
                event_id=f"dfo-{start.isoformat()}-{_slug(location)}",
                source_type=SourceType.DFO,
                label="known_spawn",
                label_confidence=LabelConfidence.HIGH,
                start_date=start,
                end_date=end,
                geometry=Point(float(lon), float(lat)),
                source=source,
                properties={
                    "location": location,
                    "spawn_length_m": _first(row, "Length", "LENGTH", "length"),
                    "spawn_width_m": _first(row, "Width", "WIDTH", "width"),
                },
            )
        )
    return events


def _first(row: dict, *keys: str):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def _parse_date(value) -> date | None:
    if value is None:
        return None
    return pd.to_datetime(value).date()


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in value)
    return "-".join(cleaned.split())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/datasets/test_dfo.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/datasets/dfo.py tests/datasets/test_dfo.py
git commit -m "feat: ingest dfo spawn index records"
```

---

### Task 6: Event Catalog Builder Script

**Files:**
- Create: `scripts/build_event_catalog.py`
- Test: `tests/test_build_event_catalog.py`

- [ ] **Step 1: Write failing test for catalog serialization**

Create `tests/test_build_event_catalog.py`:

```python
import json
from pathlib import Path

from scripts.build_event_catalog import write_event_catalog


def test_write_event_catalog_includes_manual_events(tmp_path: Path):
    output = tmp_path / "events.geojson"

    write_event_catalog(output=output, dfo_csv=None, track_root=None)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 4
    assert payload["features"][0]["properties"]["source_type"] == "manual"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_event_catalog.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`.

- [ ] **Step 3: Implement event catalog script**

Create `scripts/build_event_catalog.py`:

```python
import argparse
import json
from pathlib import Path

from herring_spawner.datasets.dfo import load_dfo_csv
from herring_spawner.datasets.manual import load_manual_events
from herring_spawner.datasets.tracks import load_track_aois


def write_event_catalog(output: Path, dfo_csv: Path | None, track_root: Path | None) -> None:
    events = load_manual_events()
    if dfo_csv is not None:
        events.extend(load_dfo_csv(dfo_csv))
    if track_root is not None:
        for month_dir in sorted(path for path in track_root.iterdir() if path.is_dir()):
            paths = sorted(
                path for path in month_dir.rglob("*") if path.suffix.lower() in {".kml", ".kmz", ".gpx"}
            )
            events.extend(load_track_aois(paths, month_label=month_dir.name))

    features = []
    for event in events:
        record = event.to_record()
        geometry = record.pop("geometry")
        features.append({"type": "Feature", "geometry": geometry, "properties": record})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/interim/events.geojson"))
    parser.add_argument("--dfo-csv", type=Path)
    parser.add_argument("--track-root", type=Path, default=Path("/Users/dexterfichuk/Downloads/2025 Tracks"))
    args = parser.parse_args()
    write_event_catalog(output=args.output, dfo_csv=args.dfo_csv, track_root=args.track_root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_event_catalog.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_event_catalog.py tests/test_build_event_catalog.py
git commit -m "feat: add event catalog builder"
```

---

### Task 7: Imagery Provider Interfaces and GEE Search

**Files:**
- Create: `herring_spawner/imagery/__init__.py`
- Create: `herring_spawner/imagery/base.py`
- Create: `herring_spawner/imagery/gee.py`
- Test: `tests/imagery/test_gee.py`

- [ ] **Step 1: Write failing tests for GEE provider construction**

Create `tests/imagery/test_gee.py`:

```python
from herring_spawner.imagery.gee import GeeSentinel2Provider


def test_gee_provider_uses_redd_fish_project_by_default():
    provider = GeeSentinel2Provider()

    assert provider.project == "redd-fish"
    assert provider.collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert provider.cloud_collection == "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"


def test_scene_search_request_is_provider_neutral():
    provider = GeeSentinel2Provider(project="custom-project")
    request = provider.build_search_request(
        bounds=(-126.3, 50.7, -126.1, 50.9),
        start_date="2026-03-25",
        end_date="2026-04-14",
        max_cloud=50,
    )

    assert request["provider"] == "gee"
    assert request["project"] == "custom-project"
    assert request["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert request["bounds"] == (-126.3, 50.7, -126.1, 50.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/imagery/test_gee.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.imagery`.

- [ ] **Step 3: Implement base and GEE provider**

Create `herring_spawner/imagery/__init__.py`:

```python
"""Imagery provider interfaces and implementations."""
```

Create `herring_spawner/imagery/base.py`:

```python
from dataclasses import dataclass
from typing import Protocol

from herring_spawner.models import Scene


@dataclass(frozen=True)
class SearchRequest:
    bounds: tuple[float, float, float, float]
    start_date: str
    end_date: str
    max_cloud: float


class SceneProvider(Protocol):
    def search(self, request: SearchRequest) -> list[Scene]:
        raise NotImplementedError
```

Create `herring_spawner/imagery/gee.py`:

```python
from herring_spawner.config import Settings
from herring_spawner.imagery.base import SearchRequest
from herring_spawner.models import Scene


class GeeSentinel2Provider:
    collection = "COPERNICUS/S2_SR_HARMONIZED"
    cloud_collection = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

    def __init__(self, project: str | None = None):
        self.project = project or Settings().gee_project

    def build_search_request(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud: float = 50,
    ) -> dict:
        return {
            "provider": "gee",
            "project": self.project,
            "collection": self.collection,
            "cloud_collection": self.cloud_collection,
            "bounds": bounds,
            "start_date": start_date,
            "end_date": end_date,
            "max_cloud": max_cloud,
        }

    def search(self, request: SearchRequest) -> list[Scene]:
        try:
            import ee
        except ImportError as error:
            raise RuntimeError("earthengine-api is required for GEE searches") from error

        ee.Initialize(project=self.project)
        geometry = ee.Geometry.Rectangle(request.bounds)
        collection = (
            ee.ImageCollection(self.collection)
            .filterBounds(geometry)
            .filterDate(request.start_date, request.end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", request.max_cloud))
        )
        scene_ids = collection.aggregate_array("system:index").getInfo()
        return [
            Scene(
                scene_id=scene_id,
                provider="gee",
                collection=self.collection,
                acquired=_scene_date(scene_id),
                cloud_score=None,
                geometry=geometry,
                properties={"gee_system_index": scene_id},
            )
            for scene_id in scene_ids
        ]


def _scene_date(scene_id: str):
    from datetime import datetime

    return datetime.strptime(scene_id[:8], "%Y%m%d").date()
```

- [ ] **Step 4: Run provider-construction tests**

Run: `python -m pytest tests/imagery/test_gee.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS. If ruff flags dynamic `dict`, change the return annotation to `dict[str, object]`.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/imagery/__init__.py herring_spawner/imagery/base.py herring_spawner/imagery/gee.py tests/imagery/test_gee.py
git commit -m "feat: add gee sentinel scene provider"
```

---

### Task 8: Chip Catalog Serialization

**Files:**
- Create: `herring_spawner/chips/__init__.py`
- Create: `herring_spawner/chips/catalog.py`
- Test: `tests/chips/test_catalog.py`

- [ ] **Step 1: Write failing tests for chip catalog writes**

Create `tests/chips/test_catalog.py`:

```python
from datetime import date
from pathlib import Path

import pandas as pd
from shapely.geometry import box

from herring_spawner.chips.catalog import write_chip_catalog
from herring_spawner.models import Chip


def test_write_chip_catalog_parquet(tmp_path: Path):
    chip = Chip(
        chip_id="chip-1",
        event_id="event-1",
        scene_id="scene-1",
        acquired=date(2026, 4, 4),
        geometry=box(-126.2, 50.8, -126.1, 50.9),
        bands=("blue", "green", "red", "nir"),
        asset_path="data/exports/chip-1.tif",
        thumbnail_path="data/review/chip-1.png",
        properties={"cloud_score": 0.12},
    )
    output = tmp_path / "chips.parquet"

    write_chip_catalog([chip], output)

    frame = pd.read_parquet(output)
    assert frame.loc[0, "chip_id"] == "chip-1"
    assert frame.loc[0, "bands"] == "blue,green,red,nir"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/chips/test_catalog.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.chips`.

- [ ] **Step 3: Implement chip catalog writer**

Create `herring_spawner/chips/__init__.py`:

```python
"""Chip catalog helpers."""
```

Create `herring_spawner/chips/catalog.py`:

```python
from pathlib import Path

import pandas as pd
from shapely.geometry import mapping

from herring_spawner.models import Chip


def write_chip_catalog(chips: list[Chip], output: Path) -> None:
    rows = []
    for chip in chips:
        rows.append(
            {
                "chip_id": chip.chip_id,
                "event_id": chip.event_id,
                "scene_id": chip.scene_id,
                "acquired": chip.acquired.isoformat(),
                "geometry": mapping(chip.geometry),
                "bands": ",".join(chip.bands),
                "asset_path": chip.asset_path,
                "thumbnail_path": chip.thumbnail_path,
                "properties": chip.properties,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/chips/test_catalog.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/chips/__init__.py herring_spawner/chips/catalog.py tests/chips/test_catalog.py
git commit -m "feat: add chip catalog serialization"
```

---

### Task 9: Spectral Feature Baselines

**Files:**
- Create: `herring_spawner/features/__init__.py`
- Create: `herring_spawner/features/spectral.py`
- Test: `tests/features/test_spectral.py`

- [ ] **Step 1: Write failing tests for visible-ratio features**

Create `tests/features/test_spectral.py`:

```python
import numpy as np

from herring_spawner.features.spectral import compute_visible_features


def test_compute_visible_features_from_band_arrays():
    bands = {
        "blue": np.array([[0.10, 0.20], [0.10, 0.20]]),
        "green": np.array([[0.30, 0.40], [0.30, 0.40]]),
        "red": np.array([[0.10, 0.20], [0.10, 0.20]]),
    }

    features = compute_visible_features(bands)

    assert features["mean_blue"] == 0.15
    assert features["mean_green"] == 0.35
    assert features["mean_red"] == 0.15
    assert round(features["green_blue_ratio"], 3) == 2.333
    assert round(features["green_red_ratio"], 3) == 2.333
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_spectral.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.features`.

- [ ] **Step 3: Implement spectral features**

Create `herring_spawner/features/__init__.py`:

```python
"""Interpretable spectral feature baselines."""
```

Create `herring_spawner/features/spectral.py`:

```python
import numpy as np


def compute_visible_features(bands: dict[str, np.ndarray]) -> dict[str, float]:
    blue = _finite(bands["blue"])
    green = _finite(bands["green"])
    red = _finite(bands["red"])

    mean_blue = float(np.mean(blue))
    mean_green = float(np.mean(green))
    mean_red = float(np.mean(red))
    epsilon = 1e-9

    return {
        "mean_blue": round(mean_blue, 6),
        "mean_green": round(mean_green, 6),
        "mean_red": round(mean_red, 6),
        "visible_brightness": round(float(np.mean([mean_blue, mean_green, mean_red])), 6),
        "green_blue_ratio": round(mean_green / (mean_blue + epsilon), 6),
        "green_red_ratio": round(mean_green / (mean_red + epsilon), 6),
    }


def _finite(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    return values[np.isfinite(values)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/features/test_spectral.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/features/__init__.py herring_spawner/features/spectral.py tests/features/test_spectral.py
git commit -m "feat: add spectral feature baselines"
```

---

### Task 10: Embedding Similarity Search

**Files:**
- Create: `herring_spawner/embeddings/__init__.py`
- Create: `herring_spawner/embeddings/search.py`
- Test: `tests/embeddings/test_search.py`

- [ ] **Step 1: Write failing tests for nearest-neighbor search**

Create `tests/embeddings/test_search.py`:

```python
import numpy as np

from herring_spawner.embeddings.search import search_similar


def test_search_similar_returns_cosine_ranked_neighbors():
    query = np.array([1.0, 0.0])
    embeddings = {
        "spawn-a": np.array([0.9, 0.1]),
        "water-b": np.array([0.0, 1.0]),
        "spawn-c": np.array([0.8, 0.2]),
    }

    results = search_similar(query=query, embeddings=embeddings, limit=2)

    assert [result.item_id for result in results] == ["spawn-a", "spawn-c"]
    assert results[0].score > results[1].score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/embeddings/test_search.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.embeddings`.

- [ ] **Step 3: Implement cosine similarity search**

Create `herring_spawner/embeddings/__init__.py`:

```python
"""Embedding storage and similarity search."""
```

Create `herring_spawner/embeddings/search.py`:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float


def search_similar(
    query: np.ndarray,
    embeddings: dict[str, np.ndarray],
    limit: int = 10,
) -> list[SearchResult]:
    query_norm = _normalize(query)
    results = [
        SearchResult(item_id=item_id, score=float(np.dot(query_norm, _normalize(vector))))
        for item_id, vector in embeddings.items()
    ]
    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values)
    if norm == 0:
        return values
    return values / norm
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/embeddings/test_search.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/embeddings/__init__.py herring_spawner/embeddings/search.py tests/embeddings/test_search.py
git commit -m "feat: add embedding similarity search"
```

---

### Task 11: Static Review Outputs

**Files:**
- Create: `herring_spawner/review/__init__.py`
- Create: `herring_spawner/review/static.py`
- Test: `tests/review/test_static.py`

- [ ] **Step 1: Write failing tests for static HTML output**

Create `tests/review/test_static.py`:

```python
from pathlib import Path

from herring_spawner.review.static import write_review_page


def test_write_review_page(tmp_path: Path):
    output = tmp_path / "review.html"
    rows = [
        {
            "chip_id": "chip-1",
            "event_id": "event-1",
            "acquired": "2026-04-04",
            "thumbnail_path": "chip-1.png",
            "review_label": "unknown",
        }
    ]

    write_review_page(rows, output)

    html = output.read_text(encoding="utf-8")
    assert "chip-1" in html
    assert "2026-04-04" in html
    assert "unknown" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/review/test_static.py -v`

Expected: FAIL with `ModuleNotFoundError` for `herring_spawner.review`.

- [ ] **Step 3: Implement static review page writer**

Create `herring_spawner/review/__init__.py`:

```python
"""Review artifact generation."""
```

Create `herring_spawner/review/static.py`:

```python
from pathlib import Path

from jinja2 import Template


TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Herring Spawn Review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
    img { max-width: 240px; }
  </style>
</head>
<body>
  <h1>Herring Spawn Review</h1>
  <table>
    <thead>
      <tr><th>Chip</th><th>Event</th><th>Date</th><th>Thumbnail</th><th>Review Label</th></tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td>{{ row.chip_id }}</td>
        <td>{{ row.event_id }}</td>
        <td>{{ row.acquired }}</td>
        <td><img src="{{ row.thumbnail_path }}" alt="{{ row.chip_id }} thumbnail"></td>
        <td>{{ row.review_label }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


def write_review_page(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(TEMPLATE.render(rows=rows), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/review/test_static.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add herring_spawner/review/__init__.py herring_spawner/review/static.py tests/review/test_static.py
git commit -m "feat: add static review page output"
```

---

### Task 12: Known-Event Scene Search Script

**Files:**
- Create: `scripts/search_known_events.py`
- Test: `tests/test_search_known_events.py`

- [ ] **Step 1: Write failing test for event date-window generation**

Create `tests/test_search_known_events.py`:

```python
from datetime import date

from shapely.geometry import Point

from herring_spawner.models import Event, LabelConfidence, SourceType
from scripts.search_known_events import build_search_window


def test_build_search_window_adds_ten_day_padding():
    event = Event(
        event_id="event-1",
        source_type=SourceType.MANUAL,
        label="known_spawn",
        label_confidence=LabelConfidence.HIGH,
        start_date=date(2026, 4, 4),
        end_date=date(2026, 4, 4),
        geometry=Point(-126.192, 50.825),
        source="unit-test",
    )

    assert build_search_window(event) == ("2026-03-25", "2026-04-14")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search_known_events.py -v`

Expected: FAIL with `ModuleNotFoundError` for `scripts.search_known_events`.

- [ ] **Step 3: Implement known-event scene search helper**

Create `scripts/search_known_events.py`:

```python
import argparse
import json
from datetime import timedelta
from pathlib import Path

from shapely.geometry import shape

from herring_spawner.imagery.base import SearchRequest
from herring_spawner.imagery.gee import GeeSentinel2Provider
from herring_spawner.models import Event, LabelConfidence, SourceType


def build_search_window(event: Event, padding_days: int = 10) -> tuple[str, str]:
    if event.start_date is None or event.end_date is None:
        raise ValueError(f"event {event.event_id} has no exact date window")
    start = event.start_date - timedelta(days=padding_days)
    end = event.end_date + timedelta(days=padding_days)
    return start.isoformat(), end.isoformat()


def event_from_feature(feature: dict) -> Event:
    properties = feature["properties"]
    from datetime import date

    start = date.fromisoformat(properties["start_date"]) if properties.get("start_date") else None
    end = date.fromisoformat(properties["end_date"]) if properties.get("end_date") else None
    return Event(
        event_id=properties["event_id"],
        source_type=SourceType(properties["source_type"]),
        label=properties["label"],
        label_confidence=LabelConfidence(properties["label_confidence"]),
        start_date=start,
        end_date=end,
        geometry=shape(feature["geometry"]),
        source=properties["source"],
        properties=properties.get("properties", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path("data/interim/events.geojson"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/scene_search_requests.json"))
    args = parser.parse_args()

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    provider = GeeSentinel2Provider()
    requests = []
    for feature in payload["features"]:
        event = event_from_feature(feature)
        if event.start_date is None:
            continue
        minx, miny, maxx, maxy = event.geometry.buffer(0.02).bounds
        start_date, end_date = build_search_window(event)
        requests.append(
            provider.build_search_request(
                bounds=(minx, miny, maxx, maxy),
                start_date=start_date,
                end_date=end_date,
                max_cloud=50,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(requests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_search_known_events.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests and lint**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/search_known_events.py tests/test_search_known_events.py
git commit -m "feat: add known event scene search planning"
```

---

### Task 13: End-to-End Prototype Smoke Run

**Files:**
- Modify: `README.md`
- Test: shell commands only

- [ ] **Step 1: Run full automated checks**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 2: Generate manual-only event catalog**

Run: `python scripts/build_event_catalog.py --output data/interim/events.geojson --track-root /Users/dexterfichuk/Downloads/2025\ Tracks`

Expected: creates `data/interim/events.geojson` with manual events and track candidate AOIs. The exact feature count depends on local track files.

- [ ] **Step 3: Generate GEE search request JSON without calling GEE**

Run: `python scripts/search_known_events.py --events data/interim/events.geojson --output data/interim/scene_search_requests.json`

Expected: creates `data/interim/scene_search_requests.json` containing requests for events with known dates, including the April 4, 2026 manual points.

- [ ] **Step 4: Document the smoke workflow in README**

Modify `README.md` to include:

```markdown
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
```

- [ ] **Step 5: Run checks again**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document prototype smoke workflow"
```

---

### Task 14: First GEE Connectivity Check

**Files:**
- Modify: `README.md`
- Test: manual command

- [ ] **Step 1: Verify Earth Engine authentication outside tests**

Run: `python - <<'PY'
import ee
ee.Initialize(project="redd-fish")
print(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).size().getInfo())
PY`

Expected: prints `1`. If authentication is missing, the command fails with an Earth Engine authentication or project-access error.

- [ ] **Step 2: If authentication fails, authenticate and retry**

Run: `earthengine authenticate`

Expected: browser-based auth flow completes, then rerun Step 1 and get `1`.

- [ ] **Step 3: Document the connectivity check**

Modify `README.md` to include:

```markdown
## Earth Engine Connectivity Check

```bash
python - <<'PY'
import ee
ee.Initialize(project="redd-fish")
print(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).size().getInfo())
PY
```

Expected output is `1`. If authentication fails, run `earthengine authenticate` and retry.
```

- [ ] **Step 4: Run docs-neutral checks**

Run: `python -m pytest -v && python -m ruff check .`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add gee connectivity check"
```

---

## Plan Self-Review

Spec coverage:

- Event/AOI catalog from DFO, manual points, and local tracks: Tasks 3-6.
- GEE project `redd-fish` and Sentinel-2 provider: Tasks 2 and 7.
- STAC-shaped provider boundary: Task 7.
- Chip catalog and review artifacts: Tasks 8 and 11.
- Spectral features: Task 9.
- Clay-ready embedding similarity boundary: Task 10.
- Static-review-first web path: Task 11.
- Kelp not implemented, generic boundaries retained: Tasks 2, 7, 10, 11.

Placeholder scan:

- No steps require unspecified behavior.
- No code blocks contain incomplete markers.
- Local GEE export of actual raster chips is intentionally deferred until after connectivity and event-search smoke tests because the design requires first validating image availability and cloud conditions.

Type consistency:

- `Event`, `Scene`, and `Chip` dataclasses are used consistently across modules.
- Internal band names use `blue`, `green`, `red`, and `nir` consistently.
- `GeeSentinel2Provider` uses `redd-fish` by default through `Settings`.
