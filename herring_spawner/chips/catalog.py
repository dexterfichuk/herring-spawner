from pathlib import Path

import pandas as pd
from shapely.geometry import mapping

from herring_spawner.models import Chip


def write_chip_catalog(chips: list[Chip], output: Path) -> None:
    rows = []
    for chip in chips:
        rows.append({
            "chip_id": chip.chip_id,
            "event_id": chip.event_id,
            "scene_id": chip.scene_id,
            "acquired": chip.acquired.isoformat(),
            "geometry": mapping(chip.geometry),
            "bands": ",".join(chip.bands),
            "asset_path": chip.asset_path,
            "thumbnail_path": chip.thumbnail_path,
            "properties": chip.properties,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
