from pathlib import Path

from herring_spawner.review.static import write_review_page


def test_write_review_page(tmp_path: Path):
    output = tmp_path / "review.html"
    rows = [
        {
            "chip_id": "chip-1",
            "event_id": "event-1",
            "acquired": "2026-04-04",
            "thumbnail_path": "chip-1.png",
            "review_label": "unknown",
        }
    ]
    write_review_page(rows, output)
    html = output.read_text(encoding="utf-8")
    assert "chip-1" in html
    assert "2026-04-04" in html
    assert "unknown" in html
