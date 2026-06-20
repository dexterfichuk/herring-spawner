import argparse
import json
from pathlib import Path

from herring_spawner.datasets.dfo import load_dfo_csv
from herring_spawner.datasets.manual import load_manual_events
from herring_spawner.datasets.tracks import load_track_aois


def write_event_catalog(output: Path, dfo_csv: Path | None, track_root: Path | None) -> None:
    events = load_manual_events()
    if dfo_csv is not None:
        events.extend(load_dfo_csv(dfo_csv))
    if track_root is not None:
        for month_dir in sorted(path for path in track_root.iterdir() if path.is_dir()):
            paths = sorted(
                path
                for path in month_dir.rglob("*")
                if path.suffix.lower() in {".kml", ".kmz", ".gpx"}
            )
            events.extend(load_track_aois(paths, month_label=month_dir.name))

    features = []
    for event in events:
        record = event.to_record()
        geometry = record.pop("geometry")
        features.append({"type": "Feature", "geometry": geometry, "properties": record})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/interim/events.geojson"))
    parser.add_argument("--dfo-csv", type=Path)
    parser.add_argument(
        "--track-root",
        type=Path,
        default=Path("/Users/dexterfichuk/Downloads/2025 Tracks"),
    )
    args = parser.parse_args()
    write_event_catalog(output=args.output, dfo_csv=args.dfo_csv, track_root=args.track_root)


if __name__ == "__main__":
    main()
