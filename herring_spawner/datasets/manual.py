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
