import json
from pathlib import Path

from scripts.scan_bc_coast_knn import build_sog_review_html, load_sog_records


def _write_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def test_load_sog_records_filters_and_sorts_by_combined_si(tmp_path: Path):
    p1 = tmp_path / "part1.geojson"
    p2 = tmp_path / "part2.geojson"

    _write_geojson(
        p1,
        [
            {
                "type": "Feature",
                "properties": {
                    "Year": 2018,
                    "Region": "SoG",
                    "LocationNa": "Alpha Bay",
                    "Longitude": -124.1,
                    "Latitude": 49.2,
                    "Start": "20180306000000",
                    "End_": "NA",
                    "CombinedSI": 5.0,
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "Year": 2024,
                    "Region": "SoG",
                    "LocationNa": "Too New",
                    "Longitude": -124.1,
                    "Latitude": 49.2,
                    "Start": "20240306000000",
                    "CombinedSI": 99.0,
                },
            },
        ],
    )
    _write_geojson(
        p2,
        [
            {
                "type": "Feature",
                "properties": {
                    "Year": 2017,
                    "Region": "SoG",
                    "LocationNa": "Bravo Inlet",
                    "Longitude": -123.8,
                    "Latitude": 49.4,
                    "End_": "20170412000000",
                    "CombinedSI": 12.5,
                },
            }
        ],
    )

    records, summary = load_sog_records([p1, p2])

    assert summary["raw_features"] == 3
    assert summary["kept_records"] == 2
    assert [row["location_name"] for row in records] == ["Bravo Inlet", "Alpha Bay"]
    assert records[0]["target_date"] == "2017-04-12"
    assert records[1]["target_date"] == "2018-03-06"
    assert records[0]["combined_si"] == 12.5


def test_build_sog_review_html_shows_knn_scores():
    html = build_sog_review_html(
        [
            {
                "location_name": "Bravo Inlet",
                "region": "SoG",
                "date": "2017-04-12",
                "thumbnail_path": "thumbnails/bravo.png",
                "score": 0.73,
                "spawn_votes": 2,
                "k": 3,
                "knn_score": 0.67,
                "combined_si": 12.5,
                "cloud": 11.0,
                "lat": 49.4,
                "lon": -123.8,
            }
        ],
        {"record_count": 1, "thumbnail_count": 1, "top_regions": {"SoG": 1}},
    )

    assert "SoG spawn candidate review" in html
    assert "KNN score" in html
    assert "Bravo Inlet" in html
