from datetime import date
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

from herring_spawner.models import Event, LabelConfidence, SourceType, slugify


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
                event_id=f"dfo-{start.isoformat()}-{slugify(location)}",
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



