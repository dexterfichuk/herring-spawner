from dataclasses import dataclass
from typing import Protocol

from herring_spawner.models import Scene


@dataclass(frozen=True)
class SearchRequest:
    bounds: tuple[float, float, float, float]
    start_date: str
    end_date: str
    max_cloud: float


class SceneProvider(Protocol):
    def search(self, request: SearchRequest) -> list[Scene]:
        raise NotImplementedError
