from pathlib import Path

from scripts.multi_year_confirm import build_review_html, group_candidates


def test_group_candidates_marks_two_years_and_three_years(tmp_path: Path):
    rows = [
        {
            "lat": 49.2341,
            "lon": -124.5372,
            "year": 2023,
            "date": "2023-03-01",
            "score": 0.6,
            "thumbnail_path": "a.png",
            "source_dir": "data/candidates_svm_2023",
        },
        {
            "lat": 49.2342,
            "lon": -124.5371,
            "year": 2024,
            "date": "2024-03-01",
            "score": 0.7,
            "thumbnail_path": "b.png",
            "source_dir": "data/candidates_svm_2024",
        },
        {
            "lat": 49.2343,
            "lon": -124.5370,
            "year": 2025,
            "date": "2025-03-01",
            "score": 0.8,
            "thumbnail_path": "c.png",
            "source_dir": "data/candidates_svm_2025",
        },
    ]

    groups = group_candidates(rows, decimals=2)

    assert len(groups) == 1
    group = groups[0]
    assert group["n_years"] == 3
    assert group["high_confidence"] is True
    assert group["multi_year_confirmed"] is True
    assert [item["year"] for item in group["items"]] == [2023, 2024, 2025]


def test_build_review_html_shows_year_columns():
    html = build_review_html(
        [
            {
                "group_key": "49.23,-124.54",
                "lat": 49.2341,
                "lon": -124.5372,
                "rounded_lat": 49.23,
                "rounded_lon": -124.54,
                "n_years": 2,
                "multi_year_confirmed": True,
                "high_confidence": False,
                "items": [
                    {
                        "year": 2023,
                        "score": 0.6,
                        "thumb_href": "thumbs/2023/a.png",
                        "date": "2023-03-01",
                        "scene_id": "scene-a",
                    },
                    {
                        "year": 2024,
                        "score": 0.7,
                        "thumb_href": "thumbs/2024/b.png",
                        "date": "2024-03-01",
                        "scene_id": "scene-b",
                    },
                ],
            }
        ],
        {"group_count": 1, "multi_year_count": 1, "high_confidence_count": 0},
    )

    assert "Multi-year confirmed candidates" in html
    assert "2023" in html and "2024" in html
    assert "localStorage" in html
