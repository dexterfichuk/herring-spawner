"""Launch FiftyOne UI for labeling spawn images.

Usage:
    python scripts/label_fiftyone.py
    python scripts/label_fiftyone.py --labels spawn_labels_2.json  # resume from saved labels
"""

import argparse
import json
import re
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.labels as foul
from fiftyone import ViewField as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THUMBS_DIR = PROJECT_ROOT / "data" / "review" / "thumbnails2"

FILENAME_RE = re.compile(r"^(?P<event>.+)_(?P<date>\d{4}-\d{2}-\d{2})_cld(?P<cloud>\d+)\.png$")

CLASSES = ["spawn", "nospawn", "cloudy"]


def parse_filename(filename: str) -> dict:
    match = FILENAME_RE.match(filename)
    if match:
        return match.groupdict()
    return {}


def load_existing_labels(label_path: Path) -> dict:
    if not label_path.exists():
        return {}
    with open(label_path) as f:
        records = json.load(f)
    return {r["filename"]: r["label"] for r in records}


def build_dataset(existing_labels: dict) -> fo.Dataset:
    dataset = fo.Dataset(name="herring_spawn_thumbs2", overwrite=True)

    samples = []
    for img_path in sorted(THUMBS_DIR.glob("*.png")):
        info = parse_filename(img_path.name)
        label = existing_labels.get(img_path.name, "unknown")

        sample = fo.Sample(
            filepath=str(img_path),
            tags=[label] if label != "unknown" else [],
            event=info.get("event", ""),
            date=info.get("date", ""),
            cloud_pct=int(info.get("cloud", 0)),
            ground_truth=fo.Classification(label=label),
        )
        samples.append(sample)

    dataset.add_samples(samples)

    mask = F("ground_truth.label") != "unknown"
    labeled_view = dataset.match(mask)
    if len(labeled_view) > 0:
        foul.classification_label_quality(
            labeled_view, "ground_truth.label", "quality"
        )

    dataset.classes["ground_truth"] = CLASSES + ["unknown"]
    dataset.save()
    return dataset


def export_labels(dataset: fo.Dataset, path: Path):
    labeled = dataset.match(F("ground_truth.label") != "unknown")
    results = []
    for sample in labeled:
        filename = Path(sample.filepath).name
        label = sample.ground_truth.label
        results.append({"filename": filename, "label": label})

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExported {len(results)} labels -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Label spawn images with FiftyOne")
    parser.add_argument(
        "--labels",
        default="spawn_labels_2.json",
        help="JSON file to load/save labels (default: spawn_labels_2.json)",
    )
    args = parser.parse_args()

    labels_file = PROJECT_ROOT / "data" / "review" / args.labels
    existing_labels = load_existing_labels(labels_file)

    if existing_labels:
        print(f"Loaded {len(existing_labels)} existing labels from {labels_file}")
        counts = {}
        for label in existing_labels.values():
            counts[label] = counts.get(label, 0) + 1
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v}")

    dataset = build_dataset(existing_labels)
    print(f"\nLoaded {len(dataset)} images")
    print(f"  Spawn labels loaded: {len(existing_labels)}")

    # Launch the FiftyOne App
    session = fo.launch_app(dataset, address="0.0.0.0", port=5151)

    print("\n=== FiftyOne App running at http://localhost:5151 ===")
    print("  Click on an image, then click the tag icon to set spawn/nospawn/cloudy")
    print("  Press Ctrl+C in this terminal when done labeling")

    try:
        session.wait()
    except KeyboardInterrupt:
        pass

    export_labels(dataset, labels_file)


if __name__ == "__main__":
    main()
