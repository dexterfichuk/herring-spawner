from datetime import date
from pathlib import Path

import pandas as pd

from herring_spawner.datasets.alaska import (
    events_from_dataframe as alaska_events_from_dataframe,
)
from herring_spawner.datasets.alaska import load_alaska_csv
from herring_spawner.datasets.dfo import events_from_dataframe as dfo_events_from_dataframe
from herring_spawner.models import SourceType


def test_events_from_dfo_dataframe_supports_spawn_index_columns():
    frame = pd.DataFrame(
        [
            {
                "Location": "Klaskish Inlt",
                "Latitude": 50.238889,
                "Longitude": -127.777778,
                "StartDate": "1951-02-18",
                "EndDate": "1951-02-18",
                "Length": 594,
                "Width": 6,
            }
        ]
    )

    events = dfo_events_from_dataframe(frame, source="unit-test")

    assert len(events) == 1
    assert events[0].source_type == SourceType.DFO
    assert events[0].event_id == "dfo-1951-02-18-klaskish-inlt"
    assert events[0].start_date == date(1951, 2, 18)


def test_events_from_alaska_dataframe_supports_survey_columns():
    frame = pd.DataFrame(
        [
            {
                "SurveyDate": "2024-03-12",
                "Location": "Nelson Bay",
                "Latitude": 54.1234,
                "Longitude": -132.5678,
                "SpawnMileage": 3.5,
            }
        ]
    )

    events = alaska_events_from_dataframe(frame, source="unit-test")

    assert len(events) == 1
    assert events[0].source_type.name == "ALASKA"
    assert events[0].event_id == "alaska-2024-03-12-nelson-bay"
    assert events[0].start_date == date(2024, 3, 12)


def test_alaska_loader_skips_bad_request_files(tmp_path: Path):
    path = tmp_path / "alaska_herring_surveys.csv"
    path.write_text("Bad Request", encoding="utf-8")

    events = load_alaska_csv(path)

    assert events == []
