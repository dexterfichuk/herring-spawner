from scripts.build_labeler import build_html


def test_build_html_includes_focus_mode_controls_and_shortcuts():
    html = build_html([])

    assert "let focusMode = false;" in html
    assert "function toggleFocusMode()" in html
    assert "function renderFocusView()" in html
    assert "function nextUnlabeledItem(delta)" in html
    assert "Focus Mode" in html
    assert "Grid Mode" in html
    assert "nextUnlabeledItem(1);" in html
