from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


class SourceType(StrEnum):
    DFO = "dfo"
    ALASKA = "alaska"
    WASHINGTON = "washington"
    MANUAL = "manual"
    TRACK = "track"
    NEWS = "news"


class LabelConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Event:
    event_id: str
    source_type: SourceType
    label: str
    label_confidence: LabelConfidence
    start_date: date | None
    end_date: date | None
    geometry: BaseGeometry
    source: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_type": self.source_type.value,
            "label": self.label,
            "label_confidence": self.label_confidence.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "geometry": mapping(self.geometry),
            "source": self.source,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class Scene:
    scene_id: str
    provider: str
    collection: str
    acquired: date
    cloud_score: float | None
    geometry: BaseGeometry
    properties: dict[str, Any] = field(default_factory=dict)


def slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in value)
    return "-".join(cleaned.split())


@dataclass(frozen=True)
class Chip:
    chip_id: str
    event_id: str
    scene_id: str
    acquired: date
    geometry: BaseGeometry
    bands: tuple[str, ...]
    asset_path: str | None
    thumbnail_path: str | None
    properties: dict[str, Any] = field(default_factory=dict)
