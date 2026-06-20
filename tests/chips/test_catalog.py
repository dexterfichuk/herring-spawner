from datetime import date
from pathlib import Path

import pandas as pd
from shapely.geometry import box

from herring_spawner.chips.catalog import write_chip_catalog
from herring_spawner.models import Chip


def test_write_chip_catalog_parquet(tmp_path: Path):
    chip = Chip(
        chip_id="chip-1",
        event_id="event-1",
        scene_id="scene-1",
        acquired=date(2026, 4, 4),
        geometry=box(-126.2, 50.8, -126.1, 50.9),
        bands=("blue", "green", "red", "nir"),
        asset_path="data/exports/chip-1.tif",
        thumbnail_path="data/review/chip-1.png",
        properties={"cloud_score": 0.12},
    )
    output = tmp_path / "chips.parquet"
    write_chip_catalog([chip], output)
    frame = pd.read_parquet(output)
    assert frame.loc[0, "chip_id"] == "chip-1"
    assert frame.loc[0, "bands"] == "blue,green,red,nir"
