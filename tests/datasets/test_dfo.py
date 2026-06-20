from datetime import date

import pandas as pd

from herring_spawner.datasets.dfo import events_from_dataframe


def test_events_from_dfo_dataframe_with_known_columns():
    frame = pd.DataFrame(
        [
            {
                "Location": "Qualicum Beach",
                "Latitude": 49.355704,
                "Longitude": -124.456910,
                "StartDate": "2024-03-13",
                "EndDate": "2024-03-15",
                "Length": 5700,
                "Width": 199,
            }
        ]
    )

    events = events_from_dataframe(frame, source="unit-test")

    assert len(events) == 1
    assert events[0].event_id == "dfo-2024-03-13-qualicum-beach"
    assert events[0].start_date == date(2024, 3, 13)
    assert events[0].end_date == date(2024, 3, 15)
    assert events[0].label == "known_spawn"
    assert events[0].properties["spawn_length_m"] == 5700
