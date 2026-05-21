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
