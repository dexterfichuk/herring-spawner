import argparse
import json
from datetime import timedelta
from pathlib import Path

from shapely.geometry import shape

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

    start = (
        date.fromisoformat(properties["start_date"]) if properties.get("start_date") else None
    )
    end = (
        date.fromisoformat(properties["end_date"]) if properties.get("end_date") else None
    )
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
    parser.add_argument(
        "--events", type=Path, default=Path("data/interim/events.geojson")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/interim/scene_search_requests.json")
    )
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
