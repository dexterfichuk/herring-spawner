from datetime import date
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

from herring_spawner.models import Event, LabelConfidence, SourceType, slugify


def load_alaska_csv(path: Path) -> list[Event]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    return events_from_dataframe(frame, source=str(path))


def events_from_dataframe(frame: pd.DataFrame, source: str) -> list[Event]:
    if frame.empty:
        return []

    events: list[Event] = []
    for row in frame.to_dict(orient="records"):
        lat = _first(row, "Latitude", "LATITUDE", "Lat", "lat")
        lon = _first(row, "Longitude", "LONGITUDE", "Lon", "lon")
        survey_date = _parse_date(
            _first(row, "SurveyDate", "SURVEY_DATE", "Date", "StartDate", "start_date")
        )
        location = str(
            _first(row, "Location", "LOCATION", "LocationName", "Area", "Site") or "unknown"
        )
        if lat is None or lon is None or survey_date is None:
            continue

        properties = {
            "location": location,
            "survey_date": survey_date.isoformat(),
        }
        spawn_mileage = _first(row, "SpawnMileage", "Spawn Mileage", "Mileage", "Miles")
        if spawn_mileage is not None:
            properties["spawn_mileage"] = spawn_mileage

        events.append(
            Event(
                event_id=f"alaska-{survey_date.isoformat()}-{slugify(location)}",
                source_type=SourceType.ALASKA,
                label="known_spawn",
                label_confidence=LabelConfidence.HIGH,
                start_date=survey_date,
                end_date=survey_date,
                geometry=Point(float(lon), float(lat)),
                source=source,
                properties=properties,
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
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()
