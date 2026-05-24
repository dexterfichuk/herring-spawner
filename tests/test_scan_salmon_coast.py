from pathlib import Path

from scripts.scan_salmon_coast import DEFAULT_END, DEFAULT_OUTPUT, DEFAULT_START, resolve_scan_config


def test_resolve_scan_config_uses_year_defaults_and_output_suffix():
    output_dir, start, end = resolve_scan_config(year=2025, output=None, start=None, end=None)

    assert output_dir == Path(str(DEFAULT_OUTPUT) + "_2025")
    assert start == "2025-02-01"
    assert end == "2025-05-31"


def test_resolve_scan_config_keeps_explicit_values():
    output_dir, start, end = resolve_scan_config(
        year=2025,
        output=Path("/tmp/custom-out"),
        start="2025-03-01",
        end="2025-04-15",
    )

    assert output_dir == Path("/tmp/custom-out")
    assert start == "2025-03-01"
    assert end == "2025-04-15"
