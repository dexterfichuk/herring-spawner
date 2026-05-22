from datetime import date

from herring_spawner.datasets.manual import load_manual_events


def test_load_manual_april_2026_events():
    events = load_manual_events()

    names = {e.event_id for e in events}
    assert "manual-2026-04-04-event-1-point-1" in names
    assert "dfo-verified-qualicum-beach" in names
    assert "news-nanaimo-2025" in names
    manual_events = [e for e in events if e.event_id.startswith("manual-2026")]
    assert len(manual_events) == 4
    assert {e.start_date for e in manual_events} == {date(2026, 4, 4)}
