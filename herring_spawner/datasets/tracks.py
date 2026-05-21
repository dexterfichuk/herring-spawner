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
