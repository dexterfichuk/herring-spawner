"""Download GEE thumbnails for all spawn events and generate interactive review."""
import ee, requests, json, base64
from datetime import date, timedelta
from pathlib import Path

ee.Initialize(project="redd-fish")

# Parse events file
lines = Path("/Users/dexterfichuk/Projects/EwE-MCP/artifacts/herring_spawn_events_2023_2025.txt").read_text().strip().split("\n")
header = lines[0]
events = []
for line in lines[1:]:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 6:
        name = parts[0]
        sd = parts[1]
        ed = parts[2]
        lat = float(parts[3])
        lon = float(parts[4])
        source = parts[5]
        if sd and ed:
            start = date.fromisoformat(sd)
            end = date.fromisoformat(ed)
            events.append((name, lat, lon, start, end, source))

print(f"Parsed {len(events)} events")

thumb_dir = Path("data/review/thumbnails2")
thumb_dir.mkdir(parents=True, exist_ok=True)

collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
downloaded = 0

for name, lat, lon, sd, ed, source in events:
    start = (sd - timedelta(days=10)).isoformat()
    end = (ed + timedelta(days=14)).isoformat()
    slug = name.lower().replace(" ","-").replace(",","").replace("'","")[:40]
    
    scenes = (collection
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
        .sort("CLOUDY_PIXEL_PERCENTAGE"))
    ids = scenes.aggregate_array("system:index").getInfo()
    clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    
    if not ids:
        print(f"  NO SCENES: {name}")
        continue
    
    for i, (sid, cloud) in enumerate(zip(ids[:2], clouds[:2])):
        scenedate = f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}"
        fname = f"{slug}_{scenedate}_cld{cloud:.0f}.png"
        fpath = thumb_dir / fname
        if fpath.exists():
            continue
        
        img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{sid}")
        rgb = img.select(["B4","B3","B2"])
        region = ee.Geometry.Point(lon, lat).buffer(1280).bounds()
        url = rgb.getThumbURL({"min":0,"max":3000,"region":region,"dimensions":512,"format":"png"})
        resp = requests.get(url, timeout=60)
        fpath.write_bytes(resp.content)
        downloaded += 1
    
    print(f"  OK: {name} ({ids[0][:8]}, cloud={clouds[0]:.0f}%)")

print(f"\nDownloaded {downloaded} new thumbnails. Total: {len(list(thumb_dir.glob('*.png')))} in {thumb_dir}")

# Generate interactive review HTML
png_files = sorted(thumb_dir.glob("*.png"))
chips = []
for path in png_files:
    b64 = base64.b64encode(path.read_bytes()).decode()
    fname = path.name
    parts = fname.replace(".png","").split("_")
    chips.append({"filename": fname, "event": parts[0], "date": parts[1], "cloud": parts[2].replace("cld",""), "b64": b64, "label": "unknown"})

html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Herring Spawn Review 2</title>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#f0f2f5;color:#1a1a2e}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:1rem 2rem;position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.header h1{margin:0;font-size:1.3rem}
.actions{display:flex;gap:8px;flex-wrap:wrap}
.btn{padding:6px 16px;border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}
.btn-pri{background:#4CAF50;color:#fff}
.btn-pri:hover{background:#388E3C}
.btn-warn{background:#ff9800;color:#fff}
.btn-warn:hover{background:#e65100}
.stats{display:flex;gap:20px;padding:12px 20px;background:#fff;border-bottom:1px solid #e0e0e0;flex-wrap:wrap}
.stat{text-align:center}
.stat-num{font-size:22px;font-weight:700}
.stat-label{font-size:11px;color:#888;text-transform:uppercase}
.filters{padding:10px 20px;display:flex;gap:8px;flex-wrap:wrap}
.filter-btn{padding:4px 14px;border-radius:16px;border:1px solid #ccc;background:#fff;font-size:12px;cursor:pointer}
.filter-btn.active{background:#1a1a2e;color:#fff;border-color:#1a1a2e}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;padding:16px}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.08);overflow:hidden}
.card-img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}
.card-body{padding:10px}
.meta{font-size:11px;color:#888;margin-bottom:6px;word-break:break-all}
.labels{display:flex;gap:4px;flex-wrap:wrap}
.lb{padding:4px 10px;border-radius:12px;border:2px solid transparent;font-size:11px;font-weight:700;cursor:pointer;transition:all 0.12s}
.lb.active{border-color:#333;transform:scale(1.05)}
.lb-s{background:#e8f5e9;color:#2e7d32}
.lb-s.active{background:#2e7d32;color:#fff}
.lb-n{background:#fce4ec;color:#c62828}
.lb-n.active{background:#c62828;color:#fff}
.lb-c{background:#e3f2fd;color:#1565c0}
.lb-c.active{background:#1565c0;color:#fff}
.lb-u{background:#fff3e0;color:#e65100}
.lb-u.active{background:#e65100;color:#fff}
.footer{text-align:center;padding:2rem;color:#888;font-size:13px}
</style></head>
<body>
<div class="header">
<h1>Herring Spawn Review 2</h1>
<div class="actions">
<button class="btn btn-pri" onclick="downloadJSON()">Download Labels</button>
<button class="btn btn-warn" onclick="resetAll()">Reset</button>
<button class="btn" style="background:#e0e0e0;color:#333" onclick="loadJSON()">Load Labels</button>
</div></div>
<div class="stats" id="stats"></div>
<div class="filters" id="filters"></div>
<div class="grid" id="grid"></div>
<div class="footer">Click labels to classify. Download JSON when done.</div>
<script>
const LS=["spawn","nospawn","cloudy","unknown"];
const LN={spawn:"Spawn",nospawn:"No Spawn",cloudy:"Cloudy",unknown:"Unknown"};
const LC={spawn:"lb-s",nospawn:"lb-n",cloudy:"lb-c",unknown:"lb-u"};
const LCOL={spawn:"#2e7d32",nospawn:"#c62828",cloudy:"#1565c0",unknown:"#e65100"};
let chips=''' + json.dumps(chips) + ''';
let filter="all";
function render(){
const grid=document.getElementById("grid"),stats=document.getElementById("stats"),filters=document.getElementById("filters");grid.innerHTML="";
let counts={spawn:0,nospawn:0,cloudy:0,unknown:0};
let f=chips.filter(c=>filter==="all"||c.label===filter);
f.forEach((chip,i)=>{counts[chip.label]++;
const card=document.createElement("div");card.className="card";
card.innerHTML='<img class="card-img" src="data:image/png;base64,'+chip.b64+'" loading="lazy"><div class="card-body"><div class="meta">'+chip.event+" | "+chip.date+" | cloud "+chip.cloud+'%</div><div class="labels">'+LS.map(l=>'<span class="lb '+LC[l]+(chip.label===l?" active":"")+'" onclick="setLabel(\''+chip.filename+"','"+l+'\')">'+LN[l]+"</span>").join("")+"</div></div>";grid.appendChild(card)});
stats.innerHTML=Object.keys(counts).map(k=>'<div class="stat"><div class="stat-num" style="color:'+LCOL[k]+'">'+counts[k]+'</div><div class="stat-label">'+LN[k]+"</div></div>").join("")+'<div class="stat"><div class="stat-num">'+f.length+'</div><div class="stat-label">Showing</div></div>';
filters.innerHTML=["all",...LS].map(l=>'<button class="filter-btn'+(filter===l?" active":"")+'" onclick="filter=\''+l+"';render()\">"+(l==="all"?"All":LN[l])+"</button>").join("");
const lbl={};chips.forEach(c=>lbl[c.filename]=c.label);localStorage.setItem("herring_labels2",JSON.stringify(lbl))}
function setLabel(fn,l){chips.forEach(c=>{if(c.filename===fn)c.label=l});render()}
function downloadJSON(){const data=chips.map(c=>({filename:c.filename,label:c.label}));const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));a.download="spawn_labels_2.json";a.click()}
function resetAll(){if(confirm("Reset?")){chips.forEach(c=>c.label="unknown");render()}}
function loadJSON(){const i=document.createElement("input");i.type="file";i.accept=".json";i.onchange=e=>{const r=new FileReader();r.onload=ev=>{const d=JSON.parse(ev.target.result);chips.forEach(c=>{const f=d.find(x=>x.filename===c.filename);if(f)c.label=f.label});render()};r.readAsText(e.target.files[0])};i.click()}
try{const s=localStorage.getItem("herring_labels2");if(s){const lbl=JSON.parse(s);chips.forEach(c=>{if(lbl[c.filename])c.label=lbl[c.filename]})}}catch(e){}
render();
</script></body></html>'''

Path("data/review/review2.html").write_text(html_template)
print(f"\nInteractive review: file://{Path('data/review/review2.html').resolve()}")
