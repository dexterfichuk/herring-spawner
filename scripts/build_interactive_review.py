"""Generate interactive review HTML with clickable labels and all 3 model scores."""
import json, sys, base64
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

review_dir = Path('data/review')
png_files = sorted(review_dir.glob('*.png'))

# Build labels from samples manifest + existing analysis
manifest = json.loads(Path('data/samples/samples_manifest.json').read_text())


# Score data from our three methods
scores = {
    # DINOv2 scores (from the corrected labels run)
    'dinov2': {
        'dfo-verified-fan-island_2024-03-22_20240322.png': 0.3363,
        'dfo-verified-anderson-point_2024-03-06_20240306.png': 0.1775,
        'dfo-verified-fan-island_2024-03-20_20240320.png': 0.1456,
        'dfo-verified-anderson-point_2024-03-16_20240316.png': 0.1341,
        'dfo-verified-anderson-point_2024-03-18_20240318.png': 0.1066,
    },
    # Clay multi-spectral scores (from the GeoTIFF run)
    'clay': {},
}

# Map from event names to filenames
event_to_files = {}
for f in png_files:
    fname = f.name
    for prefix in ['pos-', 'neg-', 'cand-', 'new-']:
        if prefix in fname:
            event_to_files.setdefault(fname, fname)
            break

# Build chip data
chips = []
for path in png_files:
    fname = path.name
    b64 = base64.b64encode(path.read_bytes()).decode()
    
    # Determine label
    label = 'unknown'
    for m in manifest:
        if m['filename'] == fname:
            label = m['label']
            break
    
    chips.append({
        'filename': fname,
        'b64': b64,
        'label': label,
        'dinov2_score': scores.get(fname),
        'clay_score': None,  # Would need mapping
    })

html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive Herring Spawn Review</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1a1a2e; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 1rem 2rem; position: sticky; top: 0; z-index: 100; display: flex; justify-content: space-between; align-items: center; }
.header h1 { margin: 0; font-size: 1.3rem; }
.actions { display: flex; gap: 8px; }
.btn { padding: 6px 16px; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; }
.btn-primary { background: #4CAF50; color: white; }
.btn-primary:hover { background: #388E3C; }
.btn-warn { background: #ff9800; color: white; }
.btn-warn:hover { background: #e65100; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; padding: 16px; }
.card { background: white; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: hidden; transition: box-shadow 0.2s; }
.card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.card-img { width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; border-bottom: 2px solid #eee; }
.card-body { padding: 10px; }
.card-filename { font-size: 11px; color: #888; word-break: break-all; margin-bottom: 6px; }
.label-row { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px; }
.label-btn { padding: 4px 10px; border-radius: 12px; border: 2px solid transparent; font-size: 11px; font-weight: 700; cursor: pointer; transition: all 0.15s; }
.label-btn.active { border-color: #333; transform: scale(1.05); }
.lbl-spawn { background: #e8f5e9; color: #2e7d32; }
.lbl-spawn.active { background: #2e7d32; color: white; }
.lbl-nospawn { background: #fce4ec; color: #c62828; }
.lbl-nospawn.active { background: #c62828; color: white; }
.lbl-unknown { background: #fff3e0; color: #e65100; }
.lbl-unknown.active { background: #e65100; color: white; }
.lbl-cloud { background: #e3f2fd; color: #1565c0; }
.lbl-cloud.active { background: #1565c0; color: white; }
.scores { display: flex; gap: 8px; font-size: 11px; color: #666; }
.score { display: inline-flex; align-items: center; gap: 3px; }
.score-d2 { color: #7b1fa2; }
.score-cm { color: #c62828; }
.score-label { font-weight: 600; font-size: 10px; text-transform: uppercase; }
.footer { text-align: center; padding: 2rem; color: #888; font-size: 13px; }
#stats { margin: 16px; padding: 12px 20px; background: white; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); display: flex; gap: 24px; flex-wrap: wrap; }
.stat-item { text-align: center; }
.stat-num { font-size: 24px; font-weight: 700; }
.stat-label { font-size: 11px; color: #888; text-transform: uppercase; }
.filters { display: flex; gap: 8px; padding: 0 16px; margin-bottom: 8px; flex-wrap: wrap; }
.filter-btn { padding: 4px 12px; border-radius: 16px; border: 1px solid #ddd; background: white; font-size: 12px; cursor: pointer; }
.filter-btn.active { background: #1a1a2e; color: white; border-color: #1a1a2e; }
</style>
</head>
<body>

<div class="header">
    <h1>Interactive Herring Spawn Review</h1>
    <div class="actions">
        <button class="btn btn-primary" onclick="downloadJSON()">Download Labels</button>
        <button class="btn btn-warn" onclick="resetAll()">Reset All</button>
    </div>
</div>

<div id="stats">
    <div class="stat-item"><div class="stat-num" id="count-spawn">0</div><div class="stat-label">Spawn</div></div>
    <div class="stat-item"><div class="stat-num" id="count-nospawn">0</div><div class="stat-label">No Spawn</div></div>
    <div class="stat-item"><div class="stat-num" id="count-cloud">0</div><div class="stat-label">Cloudy</div></div>
    <div class="stat-item"><div class="stat-num" id="count-unknown">0</div><div class="stat-label">Unknown</div></div>
    <div class="stat-item"><div class="stat-num" id="count-total">0</div><div class="stat-label">Total</div></div>
</div>

<div class="filters" id="filters">
    <button class="filter-btn active" data-filter="all" onclick="setFilter('all', this)">All</button>
    <button class="filter-btn" data-filter="spawn" onclick="setFilter('spawn', this)">Spawn</button>
    <button class="filter-btn" data-filter="nospawn" onclick="setFilter('nospawn', this)">No Spawn</button>
    <button class="filter-btn" data-filter="unknown" onclick="setFilter('unknown', this)">Unknown</button>
</div>

<div class="grid" id="grid"></div>
<div class="footer">Click a label to change. Download the JSON to save your corrections.</div>

<script>
const LABELS = ['spawn', 'nospawn', 'cloudy', 'unknown'];
const LABEL_CLASSES = {spawn:'lbl-spawn', nospawn:'lbl-nospawn', cloudy:'lbl-cloud', unknown:'lbl-unknown'};
const LABEL_NAMES = {spawn:'Spawn', nospawn:'No Spawn', cloudy:'Cloudy', unknown:'Unknown'};

let chips = ''' + json.dumps(chips) + ''';
let currentFilter = 'all';

function render() {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    let counts = {spawn:0, nospawn:0, cloudy:0, unknown:0};
    
    chips.forEach((chip, idx) => {
        if (currentFilter !== 'all' && chip.label !== currentFilter) return;
        counts[chip.label] = (counts[chip.label] || 0) + 1;
        
        const card = document.createElement('div');
        card.className = 'card';
        
        let labelsHtml = '';
        LABELS.forEach(l => {
            const active = chip.label === l ? 'active' : '';
            labelsHtml += `<span class="label-btn ${LABEL_CLASSES[l]} ${active}" onclick="setLabel(${idx}, '${l}')">${LABEL_NAMES[l]}</span>`;
        });
        
        const d2 = chip.dinov2_score !== null ? 
            `<span class="score score-d2"><span class="score-label">D2</span>${chip.dinov2_score.toFixed(4)}</span>` : '';
        const cm = chip.clay_score !== null ?
            `<span class="score score-cm"><span class="score-label">CM</span>${chip.clay_score.toFixed(4)}</span>` : '';
        
        card.innerHTML = `
            <img class="card-img" src="data:image/png;base64,${chip.b64}" loading="lazy">
            <div class="card-body">
                <div class="card-filename">#${idx+1} ${chip.filename}</div>
                <div class="label-row">${labelsHtml}</div>
                <div class="scores">${d2}${cm}</div>
            </div>
        `;
        grid.appendChild(card);
    });
    
    document.getElementById('count-spawn').textContent = counts.spawn || 0;
    document.getElementById('count-nospawn').textContent = counts.nospawn || 0;
    document.getElementById('count-cloud').textContent = counts.cloudy || 0;
    document.getElementById('count-unknown').textContent = counts.unknown || 0;
    document.getElementById('count-total').textContent = chips.length;
}

function setLabel(idx, label) {
    chips[idx].label = label;
    render();
}

function setFilter(filter, btn) {
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
}

function downloadJSON() {
    const data = chips.map(c => ({filename: c.filename, label: c.label}));
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'spawn_labels.json';
    a.click();
    URL.revokeObjectURL(url);
}

function resetAll() {
    if (confirm('Reset all labels to unknown?')) {
        chips.forEach(c => c.label = 'unknown');
        render();
    }
}

render();
</script>
</body>
</html>'''

Path('data/review/interactive_review.html').write_text(html)
print(f'Interactive review: file://{Path("data/review/interactive_review.html").resolve()}')
