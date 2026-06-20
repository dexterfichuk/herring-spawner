from pathlib import Path

from jinja2 import Template

TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Herring Spawn Review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
    img { max-width: 240px; }
  </style>
</head>
<body>
  <h1>Herring Spawn Review</h1>
  <table>
    <thead>
      <tr><th>Chip</th><th>Event</th><th>Date</th><th>Thumbnail</th><th>Review Label</th></tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td>{{ row.chip_id }}</td>
        <td>{{ row.event_id }}</td>
        <td>{{ row.acquired }}</td>
        <td><img src="{{ row.thumbnail_path }}" alt="{{ row.chip_id }} thumbnail"></td>
        <td>{{ row.review_label }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


def write_review_page(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(TEMPLATE.render(rows=rows), encoding="utf-8")
