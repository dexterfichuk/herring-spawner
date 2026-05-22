from datetime import date

from shapely.geometry import Point

from herring_spawner.models import Event, LabelConfidence, SourceType

APRIL_2026_POINTS = [
    ("event-1-point-1", 50.8254366666667, -126.192323333333),
    ("event-1-point-2", 50.8262033333333, -126.19123),
    ("event-2-point-1", 50.824935, -126.192928333333),
    ("event-2-point-2", 50.82505, -126.19266),
]

DFO_2024_EVENTS = [
    ("qualicum-beach", 49.355704, -124.456910, date(2024, 3, 13), date(2024, 3, 15), "Qualicum Beach, Strait of Georgia"),
    ("fan-island", 53.905833, -130.739444, date(2024, 3, 19), date(2024, 3, 21), "Fan Island, Prince Rupert"),
    ("tree-bluff", 54.429167, -130.488889, date(2024, 3, 17), date(2024, 3, 19), "Tree Bluff, Prince Rupert"),
    ("anderson-point", 49.646389, -126.468889, date(2024, 3, 16), date(2024, 3, 17), "Anderson Point, WCVI"),
    ("breakwater-island", 49.135000, -123.683056, date(2024, 3, 19), date(2024, 3, 20), "Breakwater Island, Gabriola"),
    ("ucluelet", 48.942778, -125.546111, date(2024, 3, 16), date(2024, 3, 19), "Ucluelet Inlet, WCVI"),
]

EVENTS_WITH_NEWS_PHOTOS_2025 = [
    ("nanaimo-2025", 49.223, -123.970, date(2025, 3, 18), date(2025, 3, 20), "Neck Point Park, Nanaimo"),
    ("salmon-beach-2025", 48.92, -125.55, date(2025, 2, 11), date(2025, 2, 13), "Salmon Beach, Barkley Sound"),
]


def load_manual_events() -> list[Event]:
    events = []
    for name, lat, lon in APRIL_2026_POINTS:
        events.append(
            Event(
                event_id=f"manual-2026-04-04-{name}",
                source_type=SourceType.MANUAL,
                label="known_spawn",
                label_confidence=LabelConfidence.HIGH,
                start_date=date(2026, 4, 4),
                end_date=date(2026, 4, 4),
                geometry=Point(lon, lat),
                source="user-provided April 4 2026 herring spawn points",
                properties={"original_name": name, "region_hint": "BC central coast"},
            )
        )
    for name, lat, lon, start, end, desc in DFO_2024_EVENTS:
        events.append(
            Event(
                event_id=f"dfo-verified-{name}",
                source_type=SourceType.MANUAL,
                label="known_spawn",
                label_confidence=LabelConfidence.HIGH,
                start_date=start,
                end_date=end,
                geometry=Point(lon, lat),
                source=f"DFO spawn index: {desc}",
                properties={"location": desc},
            )
        )
    for name, lat, lon, start, end, desc in EVENTS_WITH_NEWS_PHOTOS_2025:
        events.append(
            Event(
                event_id=f"news-{name}",
                source_type=SourceType.NEWS,
                label="known_spawn",
                label_confidence=LabelConfidence.MEDIUM,
                start_date=start,
                end_date=end,
                geometry=Point(lon, lat),
                source=f"News article: {desc}",
                properties={"location": desc},
            )
        )
    return events
