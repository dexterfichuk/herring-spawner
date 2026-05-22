"""Clay embedding pipeline - loads only the encoder, no teacher needed."""
import json, math, sys
from datetime import date, timedelta, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import ee, numpy as np, torch, yaml
from box import Box
from torchvision.transforms import v2
from collections import OrderedDict

from claymodel.model import clay_mae_large
from claymodel.utils import posemb_sincos_2d_with_gsd

ee.Initialize(project="redd-fish")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load encoder weights from checkpoint (ignoring teacher + decoder)
ckpt = torch.load("checkpoints/v1.5/clay-v1.5.ckpt", map_location="cpu", weights_only=True)
state = ckpt["state_dict"]
encoder_state = OrderedDict((k.removeprefix("model.encoder."), v) for k, v in state.items() if k.startswith("model.encoder.") and not k.startswith("model.encoder.decoder"))
# Remove decoder keys
encoder_state = OrderedDict((k, v) for k, v in encoder_state.items() if not k.startswith("decoder."))
print(f"Loaded encoder state_dict: {len(encoder_state)} keys")

# Create encoder with correct dimensions (large = dim=1024)
metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
# Create Encoder directly (no teacher, no decoder needed)
from claymodel.model import Encoder
encoder = Encoder(
    mask_ratio=0.0, shuffle=False, patch_size=8,
    dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4,
)
encoder.load_state_dict(encoder_state, strict=True)
encoder.eval().to(device)
print(f"Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")

# Rest of pipeline
BAND_NAMES = ["blue", "green", "red", "nir"]
PLATFORM = "sentinel-2-l2a"
metadata_b = Box(yaml.safe_load(open("configs/metadata.yaml")))
mean = [metadata_b[PLATFORM].bands.mean[b] for b in BAND_NAMES]
std = [metadata_b[PLATFORM].bands.std[b] for b in BAND_NAMES]
waves = [metadata_b[PLATFORM].bands.wavelength[b] for b in BAND_NAMES]
transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

EVENTS = [
    ("pos-breakwater", 49.135000, -123.683056, date(2024,3,19), date(2024,3,20), "positive"),
    ("pos-anderson", 49.646389, -126.468889, date(2024,3,16), date(2024,3,17), "positive"),
    ("pos-qualicum", 49.355704, -124.456910, date(2024,3,13), date(2024,3,15), "positive"),
    ("pos-ucluelet", 48.942778, -125.546111, date(2024,3,16), date(2024,3,19), "positive"),
    ("pos-salmon", 48.92, -125.55, date(2025,2,11), date(2025,2,13), "positive"),
    ("new-big-qualicum", 49.3989, -124.6091, date(2024,3,13), date(2024,3,16), "candidate"),
    ("new-bowser", 49.4333, -124.6667, date(2024,3,14), date(2024,3,16), "candidate"),
    ("new-cape-lazo", 49.7014, -124.8600, date(2024,3,13), date(2024,3,15), "candidate"),
    ("new-nanaimo", 49.2294, -123.9569, date(2025,3,18), date(2025,3,20), "candidate"),
    ("new-clam-bay", 48.9850, -123.6497, date(2025,3,22), date(2025,3,22), "candidate"),
    ("new-esquimalt", 48.4417, -123.4417, date(2024,3,18), date(2024,3,19), "candidate"),
    ("new-antons", 49.4164, -126.4714, date(2024,3,15), date(2024,3,15), "candidate"),
    ("new-bawden", 49.2903, -126.0164, date(2024,3,13), date(2024,3,13), "candidate"),
    ("new-amphitrite", 48.9208, -125.5419, date(2024,3,17), date(2024,3,17), "candidate"),
    ("new-boca", 49.6192, -126.6261, date(2024,3,16), date(2024,3,16), "candidate"),
    ("new-spiller", 52.2758, -128.3617, date(2024,3,23), date(2024,3,23), "candidate"),
    ("new-abrams", 52.5350, -128.8283, date(2024,3,23), date(2024,3,23), "candidate"),
    ("new-absalom", 53.8497, -130.6022, date(2024,3,20), date(2024,3,20), "candidate"),
    ("new-skeena", 54.4566, -130.3907, date(2024,3,17), date(2024,3,17), "candidate"),
    ("new-alder-crk", 52.4369, -131.3165, date(2024,4,1), date(2024,4,1), "candidate"),
    ("new-chetarpe", 49.2459, -126.0095, date(2024,3,17), date(2024,3,17), "candidate"),
    ("fan-island-spawn", 53.905833, -130.739444, date(2024,3,19), date(2024,3,21), "positive"),
]

def normalize_ts(d):
    w = d.isocalendar().week * 2 * math.pi / 52
    return (math.sin(w), math.cos(w)), (0.0, 0.0)

def normalize_ll(lat, lon):
    lr = lat * math.pi / 180
    lo = lon * math.pi / 180
    return (math.sin(lr), math.cos(lr)), (math.sin(lo), math.cos(lo))

embeddings, metadata_list = {}, []

for name, lat, lon, sd, ed, label in EVENTS:
    start = (sd - timedelta(days=7)).isoformat()
    end = (ed + timedelta(days=14)).isoformat()
    print(f"{label[:3]}: {name} ({lat:.2f}, {lon:.2f}) ", end="")

    scenes = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
        .sort("CLOUDY_PIXEL_PERCENTAGE"))
    ids = scenes.aggregate_array("system:index").getInfo()
    if not ids:
        print("no scenes"); continue

    sid = ids[0]
    print(f"| {sid[:8]}", end="")
    try:
        img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{sid}")
        bands = ["B2","B3","B4","B8"]
        region = ee.Geometry.Point(lon, lat).buffer(1280).bounds()
        chip_data = np.zeros((4, 256, 256), dtype=np.float32)
        for i, b in enumerate(bands):
            arr = img.select(b).sampleRectangle(region=region, defaultValue=0).get(b).getInfo()
            if arr and len(arr) > 0:
                a = np.array(arr, dtype=np.float32)
                # Resize to 256x256
                from skimage.transform import resize
                chip_data[i] = resize(a, (256, 256), preserve_range=True)
    except Exception as e:
        print(f" chip_err", end="")
        chip_data = np.zeros((4, 256, 256), dtype=np.float32)
    print()

    pixel_tensor = transform(torch.from_numpy(chip_data.astype(np.float32)))
    wn, hn = normalize_ts(dt(2024, 3, 15))
    ln, lon_n = normalize_ll(lat, lon)
    time_t = torch.tensor(np.hstack([wn, hn]), dtype=torch.float32, device=device).unsqueeze(0)
    latlon_t = torch.tensor(np.hstack([ln, lon_n]), dtype=torch.float32, device=device).unsqueeze(0)
    pixels_t = pixel_tensor.unsqueeze(0).to(device)
    gsd_t = torch.tensor(10.0, device=device)
    waves_t = torch.tensor(waves, device=device)

    with torch.no_grad():
        patches, _ = encoder.to_patch_embed(pixels_t, waves_t)
        patches = encoder.add_encodings(patches, time_t, latlon_t, gsd_t)
        tokens = encoder.transformer(patches)

    emb = tokens[:, 0, :].cpu().numpy().flatten()
    embeddings[name] = emb
    metadata_list.append({"name": name, "label": label, "scene": sid[:15], "lat": lat, "lon": lon})

# Score
pos_e = [embeddings[m["name"]] for m in metadata_list if m["label"] == "positive"]
mean_p = np.mean(pos_e, axis=0)
mean_p = mean_p / np.linalg.norm(mean_p)
scored = [(m["name"], float(np.dot(mean_p, embeddings[m["name"]] / np.linalg.norm(embeddings[m["name"]]))), m["label"]) for m in metadata_list]
scored.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*65}")
print(f"CLAY v1.5 ENCODER - EMBEDDING SIMILARITY RANKING")
print(f"{'='*65}")
print(f"{'Name':<25} {'Score':<8} {'Label':<10}")
print("-"*65)
for name, sim, label in scored:
    mk = " ✓" if label == "positive" else ""
    print(f"{name:<25} {sim:<8.4f} {label:<10}{mk}")

ps = [s for _,s,l in scored if l == "positive"]
cs = [s for _,s,l in scored if l == "candidate"]
print(f"\nPositive mean: {np.mean(ps):.4f}")
print(f"Candidate mean: {np.mean(cs):.4f}")
print(f"Separation: {abs(np.mean(ps)-np.mean(cs)):.4f}")

threshold = (np.mean(ps) + np.mean(cs)) / 2
above = [(n,s) for n,s,l in scored if l == "candidate" and s >= threshold]
print(f"\nThreshold: {threshold:.4f}")
print(f"Candidates classified as spawn: {len(above)}")
for n,s in above:
    print(f"  {n:<25} {s:.4f}")

np.savez("data/embeddings/clay_embeddings.npz", **embeddings)
Path("data/embeddings/metadata.json").write_text(json.dumps(metadata_list, indent=2))
print(f"\nEmbeddings saved to data/embeddings/")
