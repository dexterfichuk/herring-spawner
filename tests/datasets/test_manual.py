from datetime import date

from herring_spawner.datasets.manual import load_manual_events


def test_load_manual_april_2026_events():
    events = load_manual_events()

    assert len(events) == 4
    assert {event.start_date for event in events} == {date(2026, 4, 4)}
    assert {event.label for event in events} == {"known_spawn"}
    assert events[0].geometry.x == -126.192323333333
    assert events[0].geometry.y == 50.8254366666667
