from herring_spawner.config import Settings
from herring_spawner.imagery.base import SearchRequest
from herring_spawner.models import Scene


class GeeSentinel2Provider:
    collection = "COPERNICUS/S2_SR_HARMONIZED"
    cloud_collection = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

    def __init__(self, project: str | None = None):
        self.project = project or Settings().gee_project

    def build_search_request(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud: float = 50,
    ) -> dict:
        return {
            "provider": "gee",
            "project": self.project,
            "collection": self.collection,
            "cloud_collection": self.cloud_collection,
            "bounds": bounds,
            "start_date": start_date,
            "end_date": end_date,
            "max_cloud": max_cloud,
        }

    def search(self, request: SearchRequest) -> list[Scene]:
        try:
            import ee
        except ImportError as error:
            raise RuntimeError("earthengine-api is required for GEE searches") from error

        ee.Initialize(project=self.project)
        geometry = ee.Geometry.Rectangle(request.bounds)
        collection = (
            ee.ImageCollection(self.collection)
            .filterBounds(geometry)
            .filterDate(request.start_date, request.end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", request.max_cloud))
        )
        scene_ids = collection.aggregate_array("system:index").getInfo()
        return [
            Scene(
                scene_id=scene_id,
                provider="gee",
                collection=self.collection,
                acquired=_scene_date(scene_id),
                cloud_score=None,
                geometry=geometry,
                properties={"gee_system_index": scene_id},
            )
            for scene_id in scene_ids
        ]


def _scene_date(scene_id: str):
    from datetime import datetime

    return datetime.strptime(scene_id[:8], "%Y%m%d").date()
