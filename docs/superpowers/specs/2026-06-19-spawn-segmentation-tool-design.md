# Herring Spawn Segmentation Tool — Design Spec

**Date:** 2026-06-19
**Status:** Approved

## Overview

Flask web app for interactively segmenting herring spawn plumes from Sentinel-2 RGB thumbnails. Uses DINOv3 ViT-L/16 SAT-493M for feature-based coarse localization + SAM for pixel-level refinement. User clicks a spawn region, gets a pixel-level binary mask overlaid on the image.

## Stack

- **DINOv3 ViT-L/16 SAT-493M** (`timm/vit_large_patch16_dinov3.sat493m`) — satellite-pretrained feature extractor
- **SAM** (ViT-H or SAM2 Hiera-Large) — pixel-level mask refinement from coarse prompts
- **Flask** — web UI matching existing `label_*_app.py` pattern
- **timm** — model loading (already installed)
- **segment-anything** or **sam2** — new dependency for SAM

## Components

### 1. `scripts/segment_spawn.py` — Flask App (main entry)

- CLI: `--port`, `--manifest`, `--image-dir`, `--output-dir`, `--sam-checkpoint`
- Loads DINOv3 and SAM once at startup
- Routes:
  - `GET /` — image gallery/list from manifest
  - `GET /image/<id>` — single image view with click-to-segment
  - `POST /segment` — receives `{x, y, image_path, refine: bool}`, returns `{mask_b64, overlay_b64, mask_path}`
  - `POST /accept` — saves mask as accepted, updates manifest
  - `POST /reject` — deletes mask, marks as no-spawn
  - `GET /mask/<path>` — serve saved mask PNGs
- Templates: `templates/segment.html` — image view with click overlay, `templates/segment_list.html` — gallery

### 2. `scripts/dinov3_feature_extractor.py` — DINOv3 Similarity Map

```
Click (x, y) + image (PIL)
  → Resize to 224×224
  → DINOv3 ViT-L/16 → patch tokens [14×14, 1024]
  → Map click to nearest patch index
  → Cosine similarity: clicked_patch · all_patches → [14×14] heatmap
  → Resize to original image size
  → Threshold (mean + 1.5*std) → coarse binary mask
```

- Class `DINOv3Segmenter` with `get_similarity_map(image, point)` method
- Returns: coarse binary mask (numpy array, same size as input image)

### 3. SAM Refiner (integrated in segment_spawn.py)

```
Coarse mask + original image
  → Find largest connected component
  → Extract bounding box → SAM box prompt
  → Extract centroid → SAM point prompt
  → SAM.predict(box=bbox, points=[centroid]) → refined mask
  → If SAM fails, fall back to coarse mask
```

- Class `SAMRefiner` wrapping `SamPredictor` or `SAM2ImagePredictor`
- `refine(image, coarse_mask)` → refined binary mask

### 4. Overlay + Output

- Binary mask saved as PNG to `--output-dir/{image_name}_mask.png`
- Overlay: original image with semi-transparent green mask, rendered server-side
- Optional: `scripts/export_masks.py` to convert masks to GeoJSON polygons (for GIS)

## Data Flow

```
Manifest → Flask gallery → User clicks image → Single image view
  → User clicks spawn point → POST /segment
  → DINOv3 similarity map → coarse mask → SAM refinement
  → Overlay returned to browser
  → User can add more clicks to refine (POST /segment with refine=true)
  → User accepts → mask saved, manifest updated
```

## Input

- Sentinel-2 RGB thumbnails (512×512 PNG, already downloaded)
- Manifest JSON with image paths, lat/lon, date metadata
- Works with `training_manifest.json`, `data/candidates_knn/manifest.json`, etc.

## Output

- Binary mask PNGs in `--output-dir/` (default: `data/segmentation_masks/`)
- Updated manifest with mask paths
- Optional: GeoJSON polygon export for vector GIS use

## Dependencies (new)

- `segment-anything` (Meta SAM) — `pip install git+https://github.com/facebookresearch/segment-anything.git`
  - OR `sam2` — `pip install -e .` from `https://github.com/facebookresearch/sam2`
- `timm` — already installed

## Edge Cases

- **No spawn detected:** similarity heatmap flat → return "no distinct region" message
- **SAM fails:** coarse mask from DINOv3 is the fallback output
- **Multiple spawns:** user clicks each, masks unioned
- **Memory:** Models loaded once at startup. ViT-L ~1.2GB + SAM ViT-H ~2.6GB = ~4GB VRAM. CPU fallback available.
- **Image formats:** Handle both 512×512 (raw thumbnail) and 224×224 (DINOv3-processed)

## Testing

- Unit: `test_dinov3_segmenter.py` — similarity map on known positive
- Unit: `test_sam_refiner.py` — refinement from coarse mask
- Integration: Start Flask, click known spawn image, verify mask produced
- Visual: Review masks on 16 golden positives

## Non-goals

- Training a segmentation model (few-shot only)
- Automatic batch segmentation (click-driven, not auto)
- Video/timeseries segmentation
- Deployment beyond localhost
