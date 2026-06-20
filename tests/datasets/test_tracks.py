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
