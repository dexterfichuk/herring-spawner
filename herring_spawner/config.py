from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    gee_project: str = "redd-fish"
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    exports_dir: Path = Path("data/exports")
    review_dir: Path = Path("data/review")
