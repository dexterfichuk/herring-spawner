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
