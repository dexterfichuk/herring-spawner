"""Full Clay v1.5 multi-spectral pipeline: download GeoTIFF chips, embed, classify."""
import json, math, sys
from datetime import date, timedelta, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import ee, numpy as np, torch, yaml
import tifffile as tiff
from box import Box
from torchvision.transforms import v2
from claymodel.module import ClayMAEModule

ee.Initialize(project="redd-fish")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_default_device(device)

# Load model - teacher now cached
print("Loading Clay v1.5...")
model = ClayMAEModule.load_from_checkpoint(
    "checkpoints/v1.5/clay-v1.5.ckpt", model_size="large",
    metadata_path="configs/metadata.yaml",
    dolls=[16, 32, 64, 128, 256, 768, 1024],
    doll_weights=[1, 1, 1, 1, 1, 1, 1],
    mask_ratio=0.0, shuffle=False,
)
model.eval().to(device)
print(f"Clay loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# All labeled events
EVENTS = [
    # Positives (user-confirmed)
    ("pos-breakwater", 49.135000, -123.683056, date(2024,3,19), "positive"),
    ("pos-anderson", 49.646389, -126.468889, date(2024,3,16), "positive"),
    ("pos-qualicum", 49.355704, -124.456910, date(2024,3,13), "positive"),
    ("pos-ucluelet", 48.942778, -125.546111, date(2024,3,16), "positive"),
    ("pos-salmon", 48.92, -125.55, date(2025,2,11), "positive"),
    ("pos-fan-island", 53.905833, -130.739444, date(2024,3,19), "positive"),
    # Negatives (user-confirmed)
    ("neg-qualicum-after", 49.355704, -124.456910, date(2024,3,13), "negative"),
    ("neg-tree-bluff-pre", 54.429167, -130.488889, date(2024,3,17), "negative"),
    ("neg-pt2-1", 50.824935, -126.192928, date(2026,4,4), "negative"),
    # Unlabeled candidates to classify
    ("cand-big-qualicum", 49.3989, -124.6091, date(2024,3,13), "candidate"),
    ("cand-bowser", 49.4333, -124.6667, date(2024,3,14), "candidate"),
    ("cand-capelazo", 49.7014, -124.8600, date(2024,3,13), "candidate"),
    ("cand-chetarpe", 49.2459, -126.0095, date(2024,3,17), "candidate"),
    ("cand-bawden", 49.2903, -126.0164, date(2024,3,13), "candidate"),
    ("cand-boca", 49.6192, -126.6261, date(2024,3,16), "candidate"),
    ("cand-abrams", 52.5350, -128.8283, date(2024,3,23), "candidate"),
    ("cand-spiller", 52.2758, -128.3617, date(2024,3,23), "candidate"),
    ("cand-alder", 52.4369, -131.3165, date(2024,4,1), "candidate"),
    ("cand-skeena", 54.4566, -130.3907, date(2024,3,17), "candidate"),
    ("cand-absalom", 53.8497, -130.6022, date(2024,3,20), "candidate"),
]

BANDS = ["B2","B3","B4","B8"]
BAND_NAMES = ["blue","green","red","nir"]
PLATFORM = "sentinel-2-l2a"
SIZE, GSD = 256, 10

metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
mean = [metadata[PLATFORM].bands.mean[b] for b in BAND_NAMES]
std = [metadata[PLATFORM].bands.std[b] for b in BAND_NAMES]
waves = [metadata[PLATFORM].bands.wavelength[b] for b in BAND_NAMES]
transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

def normalize_ts(d):
    w = d.isocalendar().week * 2 * math.pi / 52
    h = d.hour * 2 * math.pi / 24 if hasattr(d, 'hour') else 0
    return (math.sin(w), math.cos(w)), (math.sin(h), math.cos(h))

def normalize_ll(lat, lon):
    return (math.sin(lat*math.pi/180), math.cos(lat*math.pi/180)), \
           (math.sin(lon*math.pi/180), math.cos(lon*math.pi/180))

chips_dir = Path("data/chips"); chips_dir.mkdir(parents=True, exist_ok=True)
embeddings, meta_list, errors = {}, [], []

for name, lat, lon, sd, label in EVENTS:
    start = (sd - timedelta(days=7)).isoformat()
    end = (sd + timedelta(days=14)).isoformat()
    print(f"\n{label[:3]}: {name} ({lat:.2f}, {lon:.2f})", end="", flush=True)

    scenes = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
        .sort("CLOUDY_PIXEL_PERCENTAGE"))
    ids = scenes.aggregate_array("system:index").getInfo()
    if not ids:
        print(" NO SCENES"); errors.append(name); continue

    sid = ids[0]
    region = ee.Geometry.Point(lon, lat).buffer(GSD * SIZE / 2).bounds()
    chip_path = chips_dir / f"{name}_{sid[:8]}.tif"

    if not chip_path.exists():
        try:
            img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{sid}")
            url = img.select(BANDS).getDownloadURL({
                "region": region, "dimensions": [SIZE, SIZE], "format": "GEO_TIFF",
            })
            import requests
            resp = requests.get(url, timeout=120)
            chip_path.write_bytes(resp.content)
        except Exception as e:
            print(f" DL_ERR", end="")
            errors.append(name)
            chip_data = np.zeros((4, SIZE, SIZE), dtype=np.float32)
            pixel_tensor = transform(torch.from_numpy(chip_data))
    else:
        print(f" cached", end="")

    # Load chip
    if chip_path.exists():
        chip_data = tiff.imread(str(chip_path)).astype(np.float32)
        # tifffile reads as (height, width, bands), reshape to (bands, height, width)
        if chip_data.ndim == 3:
            chip_data = np.transpose(chip_data, (2, 0, 1))
        if chip_data.shape != (4, SIZE, SIZE):
            from skimage.transform import resize
            chip_data = np.stack([resize(chip_data[i], (SIZE, SIZE), preserve_range=True) for i in range(chip_data.shape[0])])
        pixel_tensor = transform(torch.from_numpy(chip_data))
        print(f" | {sid[:8]}", end="")

    # Prepare model inputs
    scenedate = date.fromisoformat(f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}")
    wn, hn = normalize_ts(dt(scenedate.year, scenedate.month, scenedate.day, 12, 0))
    ln, lo = normalize_ll(lat, lon)
    datacube = {
        "platform": PLATFORM,
        "time": torch.tensor(np.hstack([wn, hn]), dtype=torch.float32, device=device).unsqueeze(0),
        "latlon": torch.tensor(np.hstack([ln, lo]), dtype=torch.float32, device=device).unsqueeze(0),
        "pixels": pixel_tensor.unsqueeze(0).to(device),
        "gsd": torch.tensor(GSD, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    with torch.no_grad():
        unmsk_patch, _, _, _ = model.model.encoder(datacube)
    emb = unmsk_patch[:, 0, :].cpu().numpy().flatten()
    embeddings[name] = emb
    meta_list.append({"name": name, "label": label, "scene": sid[:15], "lat": lat, "lon": lon})

print(f"\n\n{'='*70}")
print("CLAY v1.5 MULTI-SPECTRAL CLASSIFICATION")
print(f"{'='*70}")

# Classification
pos_names = [m["name"] for m in meta_list if m["label"] == "positive"]
neg_names = [m["name"] for m in meta_list if m["label"] == "negative"]
pos_e = [embeddings[n] for n in pos_names]
neg_e = [embeddings[n] for n in neg_names]

mean_pos = np.mean(pos_e, axis=0) / np.linalg.norm(np.mean(pos_e, axis=0))
mean_neg = np.mean(neg_e, axis=0) / np.linalg.norm(np.mean(neg_e, axis=0))

scored = []
for m in meta_list:
    emb = embeddings[m["name"]] / np.linalg.norm(embeddings[m["name"]])
    sp = float(np.dot(mean_pos, emb))
    sn = float(np.dot(mean_neg, emb))
    scored.append((m["name"], sp - sn, sp, sn, m["label"]))

scored.sort(key=lambda x: x[1], reverse=True)

print(f"{'Name':<25} {'Score':<9} {'SimPos':<8} {'SimNeg':<8} {'Label'}")
print("-"*70)
for name, score, sp, sn, label in scored:
    print(f"{name:<25} {score:<+9.4f} {sp:<8.4f} {sn:<8.4f} {label}")

ps = [s for _,s,_,_,l in scored if l == "positive"]
ns = [s for _,s,_,_,l in scored if l == "negative"]
cs = [s for _,s,_,_,l in scored if l == "candidate"]
print(f"\nPositive mean: {np.mean(ps):+.4f}")
print(f"Negative mean: {np.mean(ns):+.4f}")
print(f"Candidate mean: {np.mean(cs):+.4f}")
print(f"Separation: {np.mean(ps)-np.mean(ns):.4f}")

# Save
np.savez("data/embeddings/clay_v1.5_multispectral.npz", **embeddings)
Path("data/embeddings/metadata.json").write_text(json.dumps(meta_list, indent=2))
print(f"\nSaved to data/embeddings/")
print(f"Errors: {len(errors)} events with download issues")
