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
