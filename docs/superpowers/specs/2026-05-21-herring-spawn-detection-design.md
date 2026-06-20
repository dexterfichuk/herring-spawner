# Herring Spawn Detection Design

## Goal

Build a notebook-first research prototype, with reusable pipeline boundaries, to determine whether herring spawn can be detected from free satellite imagery using visual QA, simple spectral features, and Clay Foundation embeddings. The prototype will focus on herring spawn first, while keeping the data and imagery architecture generic enough to support future detectors such as kelp forests.

## Current Context

The project directory is currently empty and is not a git repository. The available local track data is under `/Users/dexterfichuk/Downloads/2025 Tracks` and includes KML/KMZ/GPX survey or shoreline track files grouped by month and source.

Initial inspection found that the 2025 KML tracks contain line geometries and coordinates but no obvious timestamp metadata. KMZ files contain `doc.kml` files. The month folder names should be treated as coarse timing labels unless more exact survey dates are found elsewhere.

The Google Earth Engine project ID is `redd-fish`.

## MVP Scope

The first milestone is not to detect every herring spawn in BC. The first milestone is to answer this question:

Can known herring spawn dates and locations retrieve clear Sentinel-2 imagery with a visible or embedding-based signal that separates spawn scenes from nearby pre-spawn, post-spawn, and no-spawn coastal scenes?

The MVP will produce:

- A local catalog of known and candidate spawn events.
- A catalog of matching Sentinel-2 scenes and quality metadata.
- RGB thumbnails for human verification.
- Exported Sentinel-2 chips suitable for feature extraction and Clay embeddings.
- Simple spectral feature baselines.
- Clay embeddings and nearest-neighbor similarity results.
- Static review outputs that can later become a web review tool.

## Inputs

Use three classes of input data:

- DFO/Open Canada Pacific Herring Spawn Index records as the main high-confidence label source. These records include dates or date ranges, coordinates, and spawn geometry attributes such as length and width.
- Manual April 4, 2026 points near `50.825, -126.192` as high-priority validation events.
- Local 2025 track files under `/Users/dexterfichuk/Downloads/2025 Tracks` as candidate survey/search AOIs. These should not be treated as confirmed spawn labels until visually or externally confirmed.

Public web/news examples may supplement the catalog, but DFO records should be the primary authoritative label source. Candidate examples from news articles should retain source URLs and confidence levels.

## Approach

Use **Approach A with B-shaped boundaries**:

- Start with a notebook-first research prototype for fast validation.
- Keep reusable modules from the beginning so successful pieces can become a production pipeline and later web tool.
- Avoid a web-first implementation until the satellite/model signal has been validated.

This gives the fastest path to learning while avoiding a throwaway prototype.

## Architecture

The prototype should be a small Python project. Notebooks should orchestrate workflows, while reusable logic should live in importable modules.

### `herring_spawner/datasets/`

Responsible for ingesting and normalizing event and AOI data.

Responsibilities:

- Load DFO spawn index records.
- Load manually provided points and dates.
- Parse KML, KMZ, and GPX track geometries.
- Normalize all inputs into event or AOI catalog rows.
- Preserve provenance, source URL/path, date confidence, and label confidence.

### `herring_spawner/imagery/`

Responsible for imagery search and loading through a provider-neutral interface.

First implementation:

- Google Earth Engine Sentinel-2 provider.
- GEE project ID: `redd-fish`.

Future implementation:

- STAC/direct API provider for sources such as Element84 Earth Search, Microsoft Planetary Computer, Copernicus Data Space, or Canadian-hosted STAC catalogs.

The interface should be shaped around concepts like:

```text
SceneProvider
  search(aoi, date_range, collections, max_cloud, query) -> list[Scene]
  load(scene, bands, resolution, bounds) -> RasterChip
  mask(scene, chip, mask_policy) -> RasterChip
  metadata(scene) -> SceneMetadata
```

Use provider-neutral band names internally:

```text
blue, green, red, nir, red_edge_1, swir1, swir2, cloud_mask, scene_class
```

### `herring_spawner/chips/`

Responsible for creating imagery chips and thumbnails.

Responsibilities:

- Build AOIs from buffered points, DFO length/width metadata, or track geometries.
- Search for scenes around each event date or date range.
- Export scene chips.
- Export RGB thumbnails.
- Record chip metadata, quality metadata, date windows, source scene IDs, and file paths.

### `herring_spawner/features/`

Responsible for interpretable non-ML baselines.

Initial features:

- Visible brightness.
- Blue, green, and red ratios.
- Green/blue and green/red contrast.
- Approximate bright-water or plume area after masking land and obvious cloud.
- Pre/post deltas for the same AOI.
- Cloud, haze, and scene-quality scores.

These features are required so Clay is not treated as a black box.

### `herring_spawner/embeddings/`

Responsible for Clay embedding generation and similarity search.

Responsibilities:

- Prepare Sentinel-2 chips and metadata for Clay Foundation.
- Generate embeddings for usable chips.
- Store embeddings locally with chip IDs and provenance.
- Run nearest-neighbor searches from known-spawn examples.
- Rank candidate chips by embedding similarity.

### `herring_spawner/review/`

Responsible for human QA outputs.

Initial outputs:

- Static HTML review pages or a lightweight local dashboard.
- Map-friendly GeoJSON for events, AOIs, chips, and detections.
- CSV summaries for manual inspection.

Review labels:

```text
likely_spawn
possible_spawn
no_spawn
cloud_blocked
bad_scene
unknown
```

## Data Flow

1. Build the event/AOI catalog from DFO records, manual April 2026 points, and 2025 tracks.
2. For each known event, search Sentinel-2 scenes in GEE over an event-specific date window.
3. Export clear-enough chips and RGB thumbnails.
4. Manually inspect thumbnails and assign review labels.
5. Compute baseline spectral features.
6. Generate Clay embeddings for usable chips.
7. Compare known-spawn chips against pre/post and no-spawn chips from the same or nearby areas.
8. Run nearest-neighbor similarity search across candidate events and track AOIs.
9. Expand to broader BC/PNW seasonal search only after known-event validation is promising.

All rows should store provenance, source confidence, label confidence, dates, geometry, and source file or URL.

## Satellite Imagery Strategy

Use Sentinel-2 L2A Surface Reflectance through Google Earth Engine for the first prototype.

Primary collection:

```text
COPERNICUS/S2_SR_HARMONIZED
```

Cloud support:

```text
GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED
```

or Sentinel-2 cloud probability where needed.

Initial Sentinel-2 bands:

```text
B2, B3, B4, B8
```

Optional bands for Clay, masking, or diagnostics:

```text
B5, B6, B7, B11, B12, SCL
```

Search windows:

- For DFO date ranges: start with `spawn_start - 10 days` through `spawn_end + 10 days`.
- For exact manual dates: use a tighter window around the known date, while still including pre/post scenes.
- For 2025 tracks: use month-level windows unless exact dates are discovered.

Cloud masking must be conservative. Bright herring milt can resemble cloud, foam, haze, or glint, so thumbnails and QA metadata must be retained for every candidate scene.

Landsat 8/9 can be added after the Sentinel-2 pipeline works. Landsat is useful for long historical context and large visible events, but 30 m resolution is likely too coarse for many shoreline spawns.

MODIS and VIIRS should not be used for direct detection because their pixels are too large. They may be useful later for broad regional cloud or turbidity context.

## Model Strategy

The first evaluation should compare spawn-date chips against nearby pre-spawn, post-spawn, and no-spawn chips from the same location. This controls for local shoreline, substrate, kelp, eelgrass, water color, and bathymetry.

Clay embeddings should be used as a ranking and similarity-search layer, not as the only signal.

Success criteria for the first model experiment:

- Known-spawn chips retrieve other known-spawn chips more often than random coastal chips.
- Spawn-date chips are distinguishable from pre/post chips at the same AOI in at least some clear-scene examples.
- False positives from cloud, glint, shallow bottom, and kelp can be identified during review.

Avoid training a heavy classifier until a visually reviewed dataset contains enough positive and negative examples.

## Search Strategy

After validation on known events:

1. Search historical imagery around DFO spawn locations first.
2. Expand to likely herring-spawn coastline in BC/PNW during expected spawn seasons.
3. Rank candidates using a combined score:
   - cloud and scene quality,
   - spectral plume signal,
   - Clay embedding similarity to known spawn chips,
   - temporal fit to regional spawn season,
   - shoreline proximity.
4. Send high-ranking candidates into review outputs before making any claim of a new unknown spawn.

Detections should always be treated as candidate evidence until manually reviewed or externally confirmed.

## Web Tool Path

The first deliverable should be static or lightweight review output, not a full web application.

Later web tool capabilities:

- Interactive map of candidate detections.
- Filters by date, confidence, source, review status, and detector type.
- Side-by-side pre-spawn, candidate-date, and post-spawn imagery.
- Human review workflow for confirming or rejecting detections.
- CSV and GeoJSON export.
- Detector registry for herring spawn first and kelp later.

The prototype should emit map-friendly artifacts from the start so the web app can consume them later without redesigning the data model.

## Future Kelp Support

Do not build kelp detection now. Keep the shared infrastructure generic enough to support it later.

Common infrastructure:

- dataset ingestion,
- imagery providers,
- chip generation,
- embeddings,
- vector search,
- review UI outputs.

Domain-specific detector configuration:

```text
herring_spawn:
  visual target: bright/turquoise transient nearshore plume
  temporal pattern: short spring event windows
  likely features: visible brightness, green/blue/red ratios, pre/post deltas

kelp_forest:
  visual target: persistent or seasonal nearshore vegetation signal
  temporal pattern: summer/fall vegetation windows
  likely features: NIR, red-edge, time-series persistence, polygon outputs
```

## Risks and Constraints

- Clouds, fog, and haze may block the short spawn window.
- Sun glint and foam can look like spawn.
- Bright milt may be removed by aggressive cloud masks.
- Nearshore pixels mix land, water, shallow bottom, kelp, eelgrass, boats, and substrate.
- Sentinel-2 L2A surface reflectance is convenient but not optimized for coastal ocean color.
- DFO point coordinates may represent spawn records, not full plume polygons.
- 2025 local tracks lack exact timestamps in the inspected files and should be treated as candidate AOIs, not confirmed spawn labels.

## Success Criteria

The MVP is successful if it produces:

- A reproducible event/AOI catalog from DFO, manual points, and local tracks.
- Sentinel-2 thumbnails and chips for known events using GEE project `redd-fish`.
- A reviewed set of usable and unusable scenes.
- Baseline spectral features for each chip.
- Clay embeddings and similarity-search results.
- Evidence that known spawn examples are visually or embedding-distinct from suitable negatives in at least some clear examples.
- Static review outputs suitable for later conversion into a web tool.

## Out of Scope for MVP

- Full production web app.
- Automated claims of unknown spawn without human review.
- Heavy supervised classifier training.
- Kelp detection implementation.
- Non-GEE imagery provider implementation, though interfaces should allow STAC later.
