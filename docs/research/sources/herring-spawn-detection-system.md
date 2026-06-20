# **Automated Spatiotemporal Detection and Monitoring of Pacific Herring Spawning along the British Columbia Coast: A Synergy of Spectral Indices, Legacy Ecosystem Datasets, and Few-Shot Foundation Models**

Pacific herring (*Clupea pallasii*) serve as a vital ecological cornerstone of the British Columbia coastal ecosystem, acting as a primary forage species that sustains salmon, marine mammals, and seabirds, while supporting long-standing cultural and economic harvesting practices of First Nations and commercial fisheries.1 The species exhibits extreme vulnerability to population fluctuations, most notably highlighted by the catastrophic stock collapses of 1967 driven by intensive overfishing.4 While partial stock recoveries occurred following subsequent fishery closures, modern spawning populations remain highly variable, prompting profound conservation concerns among coastal communities.4
Monitoring these dynamics is logistically challenging. Traditional survey methods managed by Fisheries and Oceans Canada (DFO) rely on visual observations from planes and boats to log spawning locations, supplemented by subtidal SCUBA dive surveys to map egg deposition on understory vegetation like eelgrass (*Zostera marina*) and kelp (*Macrocystis*).2 These methods are heavily constrained by high operational costs, weather dependencies, and a lack of synoptic coverage, which often leaves remote shorelines and marginal spawning windows unmonitored.1
During the spring spawning window—extending from mid-February through early May—herring congregate in shallow coastal areas, where the simultaneous release of eggs and sperm-containing milt clouds the water into a vibrant, highly reflective turquoise, green, or milky-blue plume.1 Remote sensing provides a scalable, cost-effective, and continuous means to detect these ephemeral plumes.3
However, developing an automated machine learning tool for this task is severely bottlenecked by data scarcity, with only twenty positive labeled samples initially available from Google Earth Engine (GEE).5 Resolving this requires a multi-tiered architecture that synthesizes physics-based spectral screening, historical spatial priors, and self-supervised geospatial and vision foundation models.3

## **Technical Evaluation of Satellite Imagery Providers**

Detecting highly dynamic coastal spawning plumes requires selecting imagery sources that balance spatial detail, temporal revisit frequency, spectral band configurations, and budgetary constraints.8 Because milt plumes can dissipate within hours to several days and frequently occur in narrow channels or nearshore zones, the imagery must resolve complex coastal geometries on a highly frequent basis.8

| Provider and Constellation | Spatial Resolution | Temporal Resolution (Revisit) | Spectral Band Suitability | Cost (per km2) | System Implementation Role |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **ESA Sentinel-2 (MSI)** 8 | 10 m / 20 m / 60 m 14 | 5 days 8 | High (13 bands; optimized for vegetation and red-edge) 8 | Free & Open 14 | Core data feed for continuous regional monitoring and index calculation. |
| **NASA/USGS Landsat 8-9 (OLI/OLI-2)** 1 | 15 m (Pan) / 30 m (MS) 14 | 8 days (combined) 14 | High (excellent radiometric calibration across visible/NIR) 1 | Free & Open 15 | Secondary data engine for long-term historical trends and spatial verification. |
| **ESA Sentinel-3 (OLCI)** 8 | 300 m 8 | 2–3 days 8 | Very High (21 bands optimized for ocean color) 8 | Free & Open 17 | Coarse regional screening to capture massive offshore spawning events. |
| **Planet Labs PlanetScope (SuperDove)** 14 | 3.7 m 14 | Daily 14 | Moderate (4 to 8 visible/NIR bands) 14 | Commercial subscription 14 | High-frequency validation of flagged hotspots and rapid plume tracking. |
| **Maxar WorldView-3** 14 | 0.31 m (Pan) / 1.24 m (MS) 14 | 1 day (taskable) 14 | Very High (panchromatic and deep multispectral) 14 | High (![][image1]–![][image2] archive/tasking) 15 | Targeted high-resolution auditing of contested or high-value spawning habitats. |
| **Airbus Pléiades Neo** 15 | 0.3 m 15 | 2 times daily 15 | High (standard visible and NIR bands) 15 | High (![][image1] tasking) 15 | Precision structural mapping and benthic vegetation verification. |

For continuous, wide-area monitoring of the extensive British Columbia coast, open-access public imagery represents the only economically viable foundation.5 ESA's Sentinel-2 Multispectral Instrument (MSI) provides the optimal spatial baseline of 10 meters, allowing nearshore environments to be resolved without incurring commercial data costs.8
By virtually combining Sentinel-2 with Landsat 8 and 9, the effective temporal revisit interval over the Pacific coast is compressed to a 2-to-3-day window.8 This is critical for capturing ephemeral plumes before tidal flushing or dilution obscures their signature.8
Sentinel-3 OLCI imagery, while highly sensitive to ocean color variations, is constrained by its 300-meter spatial resolution, which renders it unable to detect narrow, shoreline-hugging spawns.8 However, it remains a useful macro-scale asset for open waters.8
For high-priority localized monitoring or verification of potential new spawning grounds, commercial PlanetScope constellations offer daily revisits at 3.7-meter resolution, mitigating cloud cover issues through sheer capture frequency.14

## **Physics-Based Spectral Signatures and the Spectral Herring Spawning Index**

The optical characteristics of milt-rich spawning waters are highly distinct from adjacent clean water bodies, deep pelagic zones, and other highly reflective coastal targets such as terrigenous sediment plumes, glacial runoff, or shallow sandy benthos.10 Spawning pixels are physically defined by two key spectral conditions: a pronounced reflectance peak in the blue-green wavelengths (approximately 490 to 560 nm) and a steep negative slope transitioning from the green band to the red band.12
To automate the detection of these events across decades of satellite archives, researchers at the University of Victoria Spectral Remote Sensing Laboratory developed the Spectral Herring Spawning Index (SHSI).12 The index is designed for seamless, computationally efficient implementation within cloud-based geoprocessing platforms like Google Earth Engine.12

### **Mathematical Formulation of the SHSI**

![][image3]
In this equation, ![][image4] and ![][image5] represent the surface reflectance values of the green and red bands, respectively.12 On the Sentinel-2 MSI sensor, these correspond to Band 3 and Band 4, while on Landsat 8-9 OLI, they correspond to Band 3 and Band 4\.12 The coefficient of 2 applied to the red band mathematically amplifies the steep green-to-red spectral slope, maximizing the separability of milt plumes from other bright coastal targets.12

### **Operational Pre-Filtering Pipeline**

Before the SHSI is computed, the input imagery must pass through a multi-band masking pipeline to suppress false positives arising from terrestrial vegetation, cloud edges, sun glint, and turbid intertidal sediments 12:

1. **Near-Infrared (NIR) Water Segmentation Mask:**
   ![][image6]
   Due to the near-total absorption of near-infrared wavelengths by clean water, this threshold isolates the aquatic domain, immediately masking out land surfaces, intertidal rocks, surface vessels, and cloud contamination.12
2. **Green Band Background Suppression Mask:**
   ![][image7]
   This filter removes highly absorbing, dark, or deep oligotrophic pelagic waters from the processing queue, isolating only those pixels that exhibit sufficient optical backscattering to contain potential spawning plumes.12

### **Comparative Performance of Classification Paradigms**

Academic literature from the University of Victoria evaluates two primary classification methods utilizing satellite-derived spectral data: a Threshold-Based Approach (ThBA) using the SHSI and a Random Forest (RF) machine learning model trained on a combination of raw spectral bands and the computed SHSI.3

| Performance and Operational Metrics | Threshold-Based Approach (ThBA) | Random Forest (RF) Model |
| :---- | :---- | :---- |
| **Mean Detection Accuracy** | 78.3% 3 | 68.7% 3 |
| **Sensor Interoperability** | Highly consistent and robust across multiple sensors (Landsat 5–9, Sentinel-2).3 | Sensitive to sensor-specific radiometric variations; requires cross-calibration.3 |
| **Sensitivity to Optical Noise** | Low; effectively bypassed by physical pre-filtering constraints.3 | High; prone to misclassifying optically complex environments like sediment plumes.3 |
| **Primary System Role** | Serves as the high-confidence inventory generator for automated wide-area searches.3 | Serves as a highly inclusive, secondary candidate generator under clean conditions.3 |

The lower accuracy of the Random Forest model (68.7%) stems from its sensitivity to complex coastal environments, such as the Fraser River sediment plume, suspended glacial flour, and variable shallow benthic habitats.3 Under these optically complex conditions, the non-linear decision boundaries of the Random Forest model can fail to distinguish the specific green-to-red slope of milt from highly scattering terrigenous clays.3
The ThBA, anchored by physical thresholds, maintains a higher accuracy (78.3%) and provides superior consistency across different satellite generations, making it the preferred first-line detector.3

## **Integrating Historical Spatiotemporal Priors and Ecological Datasets**

The challenge of having only twenty positive labeled samples is significantly alleviated by integrating decades of historical spawning records and spatial baselines compiled by Canadian agencies and academic partners.5 Pacific herring display strong spawning site fidelity, concentrating their reproductive activity within highly specific coastal zones.1
An analysis of historical spawning events since 1928 reveals that although the British Columbia coastline exceeds 29,500 km, intensive spawning is restricted to approximately 18% to 19% (roughly 5,348 km) of the total shoreline.6

### **DFO Spatial Architecture and Database Ingestion**

The Department of Fisheries and Oceans manages this historical spatial data through a digitized grid consisting of approximately 7,800 reference positions spaced roughly one kilometer apart along the BC shoreline.19 These reference points are aggregated into 147 Herring Sections nested within Major and Minor Stock Assessment Regions (SARs).6

| Stock Assessment Region (SAR) | Active Herring Sections | Primary Spawning Habitat and Substrates | Core Historical Spawning Windows |
| :---- | :---- | :---- | :---- |
| **Haida Gwaii (HG)** 6 | Sections 001 to 006, 011 to 012, 021 to 025 6 | Deep inlets, rock surfaces, and subtidal kelp forest zones.1 | Late March to early May.1 |
| **Prince Rupert District (PRD)** 6 | Sections 031 to 033, 041 to 043, 051 to 053 6 | Estuarine marshes, eelgrass beds, and sheltered rocky channels.6 | Mid-March to mid-April.1 |
| **Central Coast (CC)** 6 | Sections 061 to 067, 071 to 078, 081 to 086, 091 to 093, 101 to 103 6 | Sheltered bays, intertidal eelgrass, and macro-algal understory.6 | Early March to mid-April.9 |
| **Strait of Georgia (SoG)** 6 | High-density historical areas concentrated in the northwestern region.5 | Sandy estuaries, shallow vegetated bays, and rock benches.1 | Late February to late March.5 |
| **West Coast Vancouver Island (WCVI)** 6 | Sections 211, 220, 231 to 233, 241 to 245, 251 to 253, 261 to 263, 271 to 274 6 | Protected sounds (e.g., Barkley Sound), rocky inlets, and kelp forests.1 | Mid-February to late March.1 |

These spatial sections are fully accessible via open-source tools:

* **The SpawnIndex R Package:** Programmatically executes the standardized mathematical frameworks used by DFO to compute relative spawning biomass from egg deposition surveys.7
* **The FIND (Find Pacific Herring Spawns) Shiny Application:** Connects directly to the DFO database, allowing users to query, map, and output localized annual spawning data summarized in tonnes.21
* **Coastal Resource Information Management System (CRIMS):** A legacy provincial geographic dataset mapping shoreline segments with attributes for relative spawning frequency, biological index, and historical significance.24

By leveraging these databases, the system can generate a dynamic, spatiotemporal prior mask. For instance, in mid-February, computing resources are focused strictly on the West Coast of Vancouver Island and the southern Strait of Georgia.1 As the season transitions into late April, the active search window moves north to Haida Gwaii and Prince Rupert.1 This prior masking restricts the satellite search space by more than 80%, reducing false positives in areas with zero historical likelihood of spawning activity.6

## **Few-Shot and Zero-Shot Advanced Machine Learning Architectures**

In a scenario constrained by only twenty positive labeled samples, standard deep neural networks (e.g., fully supervised U-Net or YOLO architectures) will overfit to the specific environmental features—such as atmospheric haze, tide lines, or sea state—present in those twenty scenes, resulting in poor generalizability.25
To overcome this, the system must utilize self-supervised **Geospatial Foundation Models (GFMs)** and **Vision Foundation Models (VFMs)**.11 These models are pre-trained on millions of unlabeled images using Masked Autoencoder (MAE) reconstruction or self-distillation, enabling them to learn highly robust, generalizable features that require minimal downstream data to adapt to specific targets.11

### **Evaluating Candidate Foundation Backbones**

The selection of the primary machine learning backbone must balance the multispectral, geographic intelligence of geospatial foundation models against the high-resolution, textural discrimination of vision foundation models.11

#### **CLAY Foundation Model (v1.5)**

CLAY is a foundational model of Earth pre-trained via self-supervised Masked Autoencoding (MAE) on diverse multispectral archives, including Sentinel-2.29 CLAY’s transformer architecture is uniquely designed to ingest spatial coordinates (latitude/longitude), temporal acquisition metadata (Julian date), and variable spectral bands directly.30 This allows the model to output 768-dimensional semantic embeddings that naturally incorporate seasonal and geographic context.29 This native multi-band integration makes CLAY highly effective at extracting sub-surface oceanographic signals.30

#### **Prithvi-EO-2.0**

Developed in partnership by IBM and NASA, Prithvi-EO-2.0 is a 600-million parameter multi-temporal geospatial foundation model pre-trained on 4.2 million global Harmonized Landsat Sentinel-2 (HLS) time-series samples at 30-meter resolution.11 Prithvi-EO-2.0 incorporates spatial and temporal attention mechanisms alongside coordinate embeddings to structure its representation space.11
Through its integration with the **TerraTorch** framework (built on PyTorch Lightning and TorchGeo), Prithvi-EO-2.0 is highly customizable, allowing developers to fine-tune downstream tasks like flood mapping and change detection with very small training samples.11

#### **DINOv2 and DINOv3**

Meta’s DINOv2 and the newer DINOv3 are visual-only foundation models trained via discriminative self-distillation on massive datasets of natural images.13 Unlike geospatial-specific models, DINOv2 is optimized to preserve high-resolution, patch-level details, geometric edges, and subtle visual textures.13
In few-shot object detection and segmentation tasks, DINOv2 features exhibit an exceptional ability to discriminate between highly similar, rare visual classes, outperforming vision-language alternatives.25

### **Why Visual-Only Backbones Outperform Vision-Language Models (VLMs)**

While Vision-Language Models (such as CLIP, RemoteCLIP, or GeoRSCLIP) allow for zero-shot text-prompting, academic evaluations show they perform poorly in highly specialized earth observation tasks.25 CLIP models are constrained by the granularity of their pre-training image captions, which lack the domain-specific vocabulary to differentiate milt plumes from optically similar features like river sediment or sandbanks.25
Visual-only backbones like DINOv2, which learn representations purely from image structure, capture fine-grained spatial and textural cues that are highly discriminative.25 This allows them to separate the irregular, organic, and highly textured boundaries of a dispersing milt plume from the smoother, unidirectional gradients of a river sediment plume.10

### **Specialized Few-Shot and Zero-Shot Architectures**

To detect herring spawn with only twenty positive samples, the development team can leverage several advanced few-shot learning and anomaly detection frameworks:

* **SubspaceAD (Training-Free Anomaly Detection):** Under this paradigm, patch-level features are extracted from a small set of normal, non-spawning coastal images of BC using a frozen DINOv2 backbone.13 A Principal Component Analysis (PCA) model is fit to these features to estimate the low-dimensional subspace of normal coastal variation (e.g., standard tides, benthos, and sun glint).13 At inference, anomalies (the rare spawning events) are detected via high reconstruction residuals relative to this normal subspace, producing statistically grounded, training-free anomaly scores.13
* **FoundAD:** A lightweight few-shot anomaly detector that employs a frozen foundation visual encoder paired with a nonlinear projection operator.35 The projector maps image features onto a natural image manifold, enabling the model to isolate out-of-distribution visual patterns (such as milt plumes) without requiring textual prompts or extensive training.35
* **P-ALIGN (Prototypical Alignment):** This framework integrates patch-based feature extraction with prototypical alignment and contrastive learning.37 An encoder learns a set of normal prototypes to serve as latent reference anchors.37 By aligning query features with these safe-harbor anchors, the system dampens background noise and amplifies the discriminative gap for anomalous milt signatures.37
* **GeCo2 (AAAI 2026 Few-Shot Counting/Detection):** Spawning plumes vary drastically in size, from small, localized cove events to multi-kilometer shoreline strips.1 GeCo2 addresses this scale variance by aggregating features across multiple backbone resolutions, fusing them into a generalized-scale dense query map.39 It supports visual prompting, where a user specifies the target class by drawing a few exemplar bounding boxes (derived from the 20 labeled samples) to locate similar features across unmapped territories.39
* **DiffSatSeg (Diffusion-Based Few-Shot Segmentation):** A diffusion-based framework that leverages the compositional generalization of diffusion models to segment targets from a single annotated image.36 It uses parameter-efficient fine-tuning via learnable proxy queries to retain rich structural and geometric priors, making it highly robust against visual distortions.36

## **Technical Pipeline Integration Blueprint**

To construct a robust, end-to-end monitoring system that achieves the dual goals of continuously monitoring historical beds and discovering new spawning sites, a hybrid pipeline is recommended.5 This system combines the computational efficiency of Google Earth Engine with the advanced representation power of a PyTorch deep learning backend running on local GPU infrastructure.5

                                    BC Coastline Satellite Ingestion
                                  (Sentinel-2 MSI & Landsat 8-9 OLI)
                                                  │
                                                  ▼
                                 Stage 1: GEE Masking & Pre-Filtering
                                   \- Near-Infrared Mask: NIR \< 0.02
                                   \- Green Band Mask: Green \> 0.025
                                                  │
                                                  ▼
                                 Stage 2: Physics-Based SHSI Engine
                                    \- Compute SHSI: Green \- 2\*Red
                                                  │
                            ┌─────────────────────┴─────────────────────┐
                            ▼                                           ▼
             Pathway A: Monitoring Old Spots             Pathway B: Finding New Spots
             \- Apply DFO Prior Spatial Mask              \- Scan Unmonitored Coastlines
             \- Target historic sections (e.g., WCVI)     \- Trigger high-resolution scans
                            │                                           │
                            └─────────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                Stage 3: Feature Extraction (PyTorch)
                                  \- Generate 768-D Patch Embeddings
                                  \- Models: CLAY v1.5 & DINOv2
                                                  │
                                                  ▼
                                Stage 4: Few-Shot Anomaly Inference
                                  \- SubspaceAD PCA Reconstruction Residual
                                  \- P-ALIGN Distance-Based Matching
                                                  │
                                                  ▼
                                Stage 5: Alerting & Database Logging
                                  \- Alert First Nations & DFO via API
                                  \- Output GIS Polygons; update GEE database

### **Stage 1: Google Earth Engine Ingestion and Pre-Filtering**

The pipeline continuously ingests Level-2A surface reflectance imagery from Sentinel-2 and Landsat 8-9 across the British Columbia coast.1 To optimize compute resources, the temporal search window is restricted to February 15 through May 15\.1 For each ingested scene, GEE applies the NIR water mask (![][image8]) to isolate the water body, followed by the green background mask (![][image9]) to eliminate deep, dark water pixels, outputting a cleaned, nearshore water mask.12

### **Stage 2: Coarse Physics-Based Screening (SHSI)**

The GEE engine computes the Spectral Herring Spawning Index (![][image10]) over the unmasked pixels.12 The processing then branches into two parallel pathways to address the dual objectives of the tool:

* **Pathway A: Monitoring Historical Sites (High-Precision):** The system applies a spatial-temporal prior mask derived from the DFO SpawnIndex database.6 The search is restricted to historically active Herring Sections based on the current Julian date (e.g., WCVI in late February, HG in late April).1 Bounding boxes are generated around any clusters where the SHSI exceeds a conservative local threshold of 0.015, capturing even subtle, marginal spawns.3
* **Pathway B: Finding New Spawning Spots (High-Recall):** The system scans historically unmonitored or abandoned shorelines.3 Because sediment plumes and cloud edge artifacts can cause optical confusion, the SHSI activation threshold is set to a more inclusive level of 0.010.3 All triggered areas generate coarse candidate bounding boxes for downstream validation.3

### **Stage 3: Deep Feature Extraction and Embedding Generation**

For every flagged bounding box, high-resolution spectral crops (10-meter Sentinel-2 bands) are downloaded from GEE and passed to a PyTorch-based inference pipeline running on local GPU instances.5 The crops are processed through a frozen, multi-modal foundation model (such as CLAY v1.5) and a visual foundation model (such as DINOv2).13
This step converts the raw imagery into 768-dimensional patch-level embeddings that capture fine-grained spatial textures, shoreline geometries, and spectral correlations while incorporating geographic coordinates and acquisition timestamps.29

### **Stage 4: Few-Shot Anomaly Inference and Verification**

The generated patch embeddings are passed through the few-shot classification backend:

1. **SubspaceAD Filtering:** The patch embeddings are projected onto the normal coastal subspace pre-calculated from thousands of unlabeled, clean BC coastal scenes.13 If a candidate patch exhibits a low reconstruction residual, it is dismissed as normal background (e.g., shallow sand or a standard river plume).13
2. **P-ALIGN Prototypical Matching:** If the patch exhibits a high reconstruction residual, the system calculates its cosine similarity to the "spawning prototype" generated from the twenty positive labeled samples.37

If the cosine similarity exceeds a validated threshold, the patch is flagged as an active spawning event.37 This dual-filter strategy ensures that new spawning locations are confirmed without being misidentified as sediment plumes, while maintaining high sensitivity.3

### **Stage 5: Alerting, Logging, and Database Integration**

Verified spawns are automatically converted into spatial GIS polygons.12 The pipeline logs the event into an interactive dashboard, updating the GEE database with a new positive sample.5
Automated alerts are dispatched via webhooks to local First Nations marine guardians, university researchers, and DFO officers, facilitating rapid field validation, SCUBA sampling, or targeted high-resolution commercial satellite tasking.1

## **Technical Recommendations for System Implementation**

To successfully build and deploy this hybrid remote sensing system, the development team should execute the following technical steps:

1. **Establish the Google Earth Engine Baseline:** Programmatically ingest the DFO historical GIS datasets.6 Use the SpawnIndex R package to extract the coordinates of the 7,800 reference points along the BC coastline and generate a static geodatabase of historical spawning sections.19 Map these sections as a high-probability spatial mask within GEE.6
2. **Implement the Spectral Filtering Engine in GEE:** Deploy the pre-filtering masking pipeline (NIR and Green band thresholds) and the SHSI index calculation.12 Set up an automated weekly cron job within GEE to scan the BC coast during the spawning season, outputting candidate bounding boxes to an enterprise storage bucket.1
3. **Build the Few-Shot PyTorch Inference Server:** Deploy a local GPU inference server pre-loaded with frozen weights for **DINOv2** and **CLAY v1.5**.30 Ingest the twenty positive GEE samples to construct the core "spawning prototype" embedding vector.37
4. **Train the Background SubspaceAD Model:** Pull approximately 2,000 random, cloud-free coastal imagery tiles of the BC coast captured outside of the spawning season.9 Extract their patch-level embeddings using the frozen DINOv2 backbone and fit a PCA model to establish the "normal coastal variation" subspace.13
5. **Configure the GeCo2 Bounding Box Visual Promoter:** Integrate the GeCo2 model into a user-facing annotation dashboard.39 This allows operators to draw visual rectangle-prompts over the twenty known positive samples to continuously refine the multi-scale query aggregation weights, optimizing detection accuracy for varying plume sizes.39
6. **Deploy the Automated Alerting and Validation System:** Connect the output of the PyTorch inference pipeline to a web GIS platform.5 Configure automated email and SMS alerts that broadcast validated coordinates to local First Nations fisheries managers and university teams.1 This loop ensures that newly discovered spawning sites are quickly validated on the ground, with the resulting field data fed back into the model to expand the positive training set.1

#### **Works cited**

1. Satellite Spots a Spawn \- NASA Science, accessed May 24, 2026, [https://science.nasa.gov/earth/earth-observatory/satellite-spots-a-spawn/](https://science.nasa.gov/earth/earth-observatory/satellite-spots-a-spawn/)
2. Loic Dallaire in Chek News: "UVic researchers use satellites to track turquoise herring spawning off Vancouver Island." \- GEOG, accessed May 24, 2026, [https://www.uvic.ca/socialsciences/geography/announcements/loic-dallaire-in-chek-news-uvic-researchers-use-satellites-to-track-turquoise-herring-spawning-off-vancouver-island.php](https://www.uvic.ca/socialsciences/geography/announcements/loic-dallaire-in-chek-news-uvic-researchers-use-satellites-to-track-turquoise-herring-spawning-off-vancouver-island.php)
3. Analysis of the Pacific herring population spawning areas using ..., accessed May 24, 2026, [https://dspace.library.uvic.ca/items/0ecd7837-4041-414e-9a30-18d284cbc0b6](https://dspace.library.uvic.ca/items/0ecd7837-4041-414e-9a30-18d284cbc0b6)
4. High-tech helping researchers better understand this year's colourful herring spawn, accessed May 24, 2026, [https://www.mycoastnow.com/76965/uncategorized/high-tech-helping-researchers-better-understand-this-years-colourful-herring-spawn/](https://www.mycoastnow.com/76965/uncategorized/high-tech-helping-researchers-better-understand-this-years-colourful-herring-spawn/)
5. Herring — SPECTRAL Remote Sensing Laboratory, accessed May 24, 2026, [http://uvicspectral.com/herring](http://uvicspectral.com/herring)
6. Herring stock assessments | Pacific Region | Fisheries and Oceans Canada, accessed May 24, 2026, [https://www.pac.dfo-mpo.gc.ca/science/species-especes/herring-hareng/stock-assessments-evaluations-stocks-eng.html](https://www.pac.dfo-mpo.gc.ca/science/species-especes/herring-hareng/stock-assessments-evaluations-stocks-eng.html)
7. Calculating the spawn index for Pacific herring (Clupea pallasii) in British Columbia, Canada, accessed May 24, 2026, [https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/41216787.pdf](https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/41216787.pdf)
8. Herring spawn seen from space \- EUMETSAT \- User Portal, accessed May 24, 2026, [https://user.eumetsat.int/resources/case-studies/herring-spawn-seen-from-space](https://user.eumetsat.int/resources/case-studies/herring-spawn-seen-from-space)
9. Spawning Spectacle \- NASA Science, accessed May 24, 2026, [https://science.nasa.gov/earth/earth-observatory/spawning-spectacle-154243/](https://science.nasa.gov/earth/earth-observatory/spawning-spectacle-154243/)
10. Satellite Remote Sensing of Herring ( Clupea pallasii ) Spawning Events: A Case Study in the Strait of Georgia | Request PDF \- ResearchGate, accessed May 24, 2026, [https://www.researchgate.net/publication/350301858\_Satellite\_Remote\_Sensing\_of\_Herring\_Clupea\_pallasii\_Spawning\_Events\_A\_Case\_Study\_in\_the\_Strait\_of\_Georgia](https://www.researchgate.net/publication/350301858_Satellite_Remote_Sensing_of_Herring_Clupea_pallasii_Spawning_Events_A_Case_Study_in_the_Strait_of_Georgia)
11. Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications \- arXiv, accessed May 24, 2026, [https://arxiv.org/html/2412.02732v3](https://arxiv.org/html/2412.02732v3)
12. Loïc T. Dallaire1,2, Alejandra Mora-Soto1, Jessica Qualey2,Maycira Costa1 \- ResearchGate, accessed May 24, 2026, [https://www.researchgate.net/profile/Loic-Dallaire/publication/389744460\_From\_Semen\_to\_Satellite\_Pacific\_Herring\_Spawn\_Science\_Goes\_Orbital/links/67d0ad67d7597000650805fd/From-Semen-to-Satellite-Pacific-Herring-Spawn-Science-Goes-Orbital.pdf](https://www.researchgate.net/profile/Loic-Dallaire/publication/389744460_From_Semen_to_Satellite_Pacific_Herring_Spawn_Science_Goes_Orbital/links/67d0ad67d7597000650805fd/From-Semen-to-Satellite-Pacific-Herring-Spawn-Science-Goes-Orbital.pdf)
13. Daily Papers \- Hugging Face, accessed May 24, 2026, [https://huggingface.co/papers?q=DINOv2%20backbone](https://huggingface.co/papers?q=DINOv2+backbone)
14. High resolution satellite imagery in precision agriculture \- Qaltivate, accessed May 24, 2026, [https://qaltivate.com/blog/high-resolution-satellite-imagery/](https://qaltivate.com/blog/high-resolution-satellite-imagery/)
15. Satellite Imagery 101: Your introductory guide to choosing the right data \- ThinkOnward, accessed May 24, 2026, [https://thinkonward.com/resources/content-hub/satellite-imagery-101-your-introductory-guide-to-choosing-the-right-data](https://thinkonward.com/resources/content-hub/satellite-imagery-101-your-introductory-guide-to-choosing-the-right-data)
16. Chuanmin HU | Research profile \- ResearchGate, accessed May 24, 2026, [https://www.researchgate.net/profile/Chuanmin-Hu-2](https://www.researchgate.net/profile/Chuanmin-Hu-2)
17. FAQ \- Sentinel Hub, accessed May 24, 2026, [https://www.sentinel-hub.com/faq/](https://www.sentinel-hub.com/faq/)
18. Satellite Providers, accessed May 24, 2026, [https://landscape.satsummit.io/capture/satellite-providers.html](https://landscape.satsummit.io/capture/satellite-providers.html)
19. Herring Spawning Areas of British Columbia \- Canada.ca, accessed May 24, 2026, [https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/40897461.pdf](https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/40897461.pdf)
20. Herring Roe Fishery Catch Data \- Open Government Portal \- Canada.ca, accessed May 24, 2026, [https://open.canada.ca/data/en/dataset/71c25df1-0577-43f7-b9f8-95cc321e7cbc](https://open.canada.ca/data/en/dataset/71c25df1-0577-43f7-b9f8-95cc321e7cbc)
21. grinnellm/FIND: :mag\_right: Find Pacific Herring spawns \- GitHub, accessed May 24, 2026, [https://github.com/grinnellm/FIND](https://github.com/grinnellm/FIND)
22. Pacific Herring Spawn Index Data \- Open Government Portal \- Canada.ca, accessed May 24, 2026, [https://open.canada.ca/data/en/dataset/d892511c-d851-4f85-a0ec-708bc05d2810](https://open.canada.ca/data/en/dataset/d892511c-d851-4f85-a0ec-708bc05d2810)
23. Matthew Grinnell grinnellm \- GitHub, accessed May 24, 2026, [https://github.com/grinnellm](https://github.com/grinnellm)
24. Herring Spawn \- Coastal Resource Information Management System (CRIMS) \- GEO.CA, accessed May 24, 2026, [https://app.geo.ca/en-ca/map-browser/record/4908a522-524f-461f-b954-96568ade85d4](https://app.geo.ca/en-ca/map-browser/record/4908a522-524f-461f-b954-96568ade85d4)
25. Exploring Robust Features for Few-Shot Object Detection in Satellite Imagery \- CVF Open Access, accessed May 24, 2026, [https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Bou\_Exploring\_Robust\_Features\_for\_Few-Shot\_Object\_Detection\_in\_Satellite\_Imagery\_CVPRW\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Bou_Exploring_Robust_Features_for_Few-Shot_Object_Detection_in_Satellite_Imagery_CVPRW_2024_paper.pdf)
26. Few-Shot Object Detection Based on Contrastive Class-Attention Feature Reweighting for Remote Sensing Images \- IEEE Xplore, accessed May 24, 2026, [https://ieeexplore.ieee.org/iel7/4609443/10330207/10375080.pdf](https://ieeexplore.ieee.org/iel7/4609443/10330207/10375080.pdf)
27. Daily Papers \- Hugging Face, accessed May 24, 2026, [https://huggingface.co/papers?q=few-shot%20object%20detection](https://huggingface.co/papers?q=few-shot+object+detection)
28. The Modern Pipeline for Satellite Imagery Processing for Model Training \- Kili Technology, accessed May 24, 2026, [https://kili-technology.com/blog/the-modern-pipeline-for-satellite-imagery-processing-for-model-training-c0cd4](https://kili-technology.com/blog/the-modern-pipeline-for-satellite-imagery-processing-for-model-training-c0cd4)
29. GeoAI in the Age of Foundation Models | Spring 2026 | ArcNews \- Esri, accessed May 24, 2026, [https://www.esri.com/about/newsroom/arcnews/geoai-in-the-age-of-foundation-models](https://www.esri.com/about/newsroom/arcnews/geoai-in-the-age-of-foundation-models)
30. Clay Foundation Model, accessed May 24, 2026, [https://clay-foundation.github.io/model/](https://clay-foundation.github.io/model/)
31. Exploring Robust Features for Few-Shot Object Detection in Satellite Imagery \- arXiv, accessed May 24, 2026, [https://arxiv.org/html/2403.05381v1](https://arxiv.org/html/2403.05381v1)
32. made-with-clay/Clay \- Hugging Face, accessed May 24, 2026, [https://huggingface.co/made-with-clay/Clay](https://huggingface.co/made-with-clay/Clay)
33. Knowledge Base: Geoscience Foundation Models: Clay \- RCAC, accessed May 24, 2026, [https://www.rcac.purdue.edu/knowledge/gfms/clay](https://www.rcac.purdue.edu/knowledge/gfms/clay)
34. Introduction to Earth Foundation Models \- IGARSS 2025 EarthFM tutorial, accessed May 24, 2026, [https://developmentseed.org/igarss25tutorial/tut1-intro/](https://developmentseed.org/igarss25tutorial/tut1-intro/)
35. Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors \- OpenReview, accessed May 24, 2026, [https://openreview.net/forum?id=YRrlJ8oVEH](https://openreview.net/forum?id=YRrlJ8oVEH)
36. Exploiting Diffusion Priors for Generalizable Few-Shot Satellite Image Semantic Segmentation \- MDPI, accessed May 24, 2026, [https://www.mdpi.com/2072-4292/17/22/3706](https://www.mdpi.com/2072-4292/17/22/3706)
37. Prototypical contrastive learning with patch-based spatio-temporal alignment for multivariate time series anomaly detection \- PMC, accessed May 24, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC13103383/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13103383/)
38. Self-Supervised Contrastive Learning for Few-Shot Anomaly Detection in Cloud Infrastructure \- ResearchGate, accessed May 24, 2026, [https://www.researchgate.net/publication/399508772\_Self-Supervised\_Contrastive\_Learning\_for\_Few-Shot\_Anomaly\_Detection\_in\_Cloud\_Infrastructure](https://www.researchgate.net/publication/399508772_Self-Supervised_Contrastive_Learning_for_Few-Shot_Anomaly_Detection_in_Cloud_Infrastructure)
39. GeCo2 in practice: few-shot object counting for dense, scale-varying scenes \- Reddit, accessed May 24, 2026, [https://www.reddit.com/r/computervision/comments/1sw8n94/geco2\_in\_practice\_fewshot\_object\_counting\_for/](https://www.reddit.com/r/computervision/comments/1sw8n94/geco2_in_practice_fewshot_object_counting_for/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAVCAYAAAD7NJjdAAAAtElEQVR4Xu3SPQrCQBiE4U8QwcbGVsHW3lLsrYScQTyB4AGsBC9g5QE8hK29Ip7EFOLPyG5wM4iVjZ/zwAvJJNWyZiIiIvJjuqjN4xsVHrzqoDu6oXN8nqQ/JOqox6NXzwNhW3RBLdpzendtyEPURFcLN6ioX/pDrMrDv1jY61Zs6Bsb8ODVDs2T95WFA8qSrTBDNR69WvIQ7S0c0BSN0AGdSn849+kGNNAaHdGYvomIiHzPA6YCGzrn07C1AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAVCAYAAAAElr0/AAACcklEQVR4Xu2WS+hNURTGF5KkJPIoM8XEM0ZKUgbKQOSVgYgSRkYGykQZkfyVESOJSDHwCGMlj6I8ZkpMiBR5lOf6zl7fPd9/3XPcO/TP/dXXWa+z99nn7McxG/DvsyUHRiKrXb9dm3NipDDaygCOxfVwXKdqkbPQ9T5yd12Thqc73LJS88G1I+V6cdD10fXFtTPlyASr+7jvGsXED9eSsJEkau92nRD/rJX8YokBxMaGjWkK/22d/ivPXLfFf+K6Iz6YaaXN8eFPCb+i6eHfNcTVb4rdcF0RH1yzUrMmxTMTrbt9gJh++c+uC+KDBzRQjJHSBpetfCnyWnIkDwT18DdIbG7E3kisiUfW3T5A7HTyN4kPDtDgA2H6NDXWBG5GLTYIgpdxRnywwkrdwxTP5JdCNL487GV1umIbjRlW3wB9dc1msoG1VuqGcqKBm1Zq5+VEop+B7As7r8uN6oxxPbbhA8K8zRx1XXT9dK1MuQzaRDvYWXrRz0AOhb2gTlesS34FCvULtcHd42pOCPiyvaYUaetP47vCXlSnKzpr8qVre9i8CVOhqWGlrXPw1MqX65e2tjTONbK0TldspYHkK7E1TjCVTokP2ElefJesHKoK22/jk7UP5HnY48Jv3bW+S5CNzRJ7fdi5I8awFsh+1x7xwXTXyRTrdB7g4XL7ADEe1vT1YAbXaRx3nQsbhTiZcZ3DgvDxRggWHGKdRqwsfg4ua5XU4dcFsb0SA4jhCCBHIqZw21eG+fid4IGGLzRNk85k168QT/38lvPDq/A/R+a7XohP8NuB2ntWdtBvJv9RwnkrUx1X1GNb7iKPdsCA/5k/1arOt6vuoTsAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABECAYAAAA89WlXAAAFT0lEQVR4Xu3dWahuUxwA8GWWeSqS6UFJiPBCXohkSMn04EGklIwZUoYQEnkgZcr8It5QZIoMkcg8JEqmB1wk87j+7bV8667z3Xu/e865xznn/n7179vrv75pr+9h/9t7rf2lBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATGyvHDfm2Ky0d2n6pmvNHOfkOK7JXdJsAwAwoX9yXJFjvRwP5fiw5MJGOZaUdo1vSt/XOf5q8j/kOL703Zfj9xy75Ti59N+e4/3SH+/RvuefOZ4tfcyOA9Iwzt/lOKXrAwAWkCiWDu2TaVSwVbuOyVWRv3JMrvdCGhVsVTzv1S7HzPXjXwvjdbs8ALAAxEH8pD6Z/d21Z6Ng2zSNL9he7nLMXIzrb0376JJ7oMkBAAtEHMT74iyc17WnU7Cd2eXCnV07nvdSl2PmYlx/bdpbldz3TQ4AWCA+SaPLZV/k2HHp7v9Mp2CrcWrX14r+F/vkHDk4x5c5vsrxcde32FyYhrHuC+YQ4xB9T6fFPw4AsGBtn4YFAm2R9dFSzxgVbHHmrY9xBVu4Iy39nhFbL/WMyQq2E3LcPyZiYcM9Oe5OQyESnxcLJyYRn3tL1/65aS8ma6Vh/8bNFVydxgEAFo0j0qi4aq3sGbbWJmk4ezPufScp2GbbtWn89+hz/5ctc+w7QUzqlxyv9cmi3+eZjMOzafqvBQCW47A+kYb7p/UH3pkUbNXbaep7RHuu57DFZ743Jjdf5nftmOOoCWISm+d4sE8WR6bZH4f+9wUAZsGyDrB9fmULtnoD3tbeaep7RPuVLteLy5+x4nGSiMu7KxKfuUXTvrzkolAN66dhVWXcg+6uNJwhPKj0HV4eQ6x6vTpNvcnwpWl0aXb3HAeW7fYGwnPhpjTMXWs91Wz341BzMQ47p6Ggi33co+mP3/CCNPXSc701TP/7AgCzIA6wq+I+bONWnj6Spp69ide+3uVWtfjMmNfVtm8r2+emoUgLsSAhiq0o4J7PcX0anluLtnvLY+xX2DjHT2U7bgS8dhoKnw9ybFfyyxrD2XZijjfS8M8SEVFYxj6c1TynH4fL0mgcYq5h9G9bHkMsSKjFW+xf2DMNizbCrWnu9g8AVitxgI0D7ltpmBcVt3/4I42KkQ3SMDcp5kHFcz/P8Uzpi3lp75R8RBQEh5S+x0rupNK+qrTXKe14bXxmfW0UbbVYWJW2SUOxEf8AsEMa/qkh/oGhiu+yT7NdXdRs1/YaZTvGaMM0TNY/reRif68r2/FPA9VcFTR1XPvYqfTHOES7jsM1aWqR3d4WJLTfPfav5uIsXDVX+wcAq5W4ZBfioB1nyR5PUy+TzUSc6YlFBWf0Hf+TWBF5Q9k+tu0o4hYkD5ftmHNX9QXbk2ko1GqEKFZi0UbN1eL03fIY5ktBE+NQv0uMQz0D2BpXsI3b51bfBgBYaSsqKB5No0KrdXGfyM5vtuMMW1we/LTJnV4e2393WNHnz5X4HnHJdHniTGvfjhWsrThLenbZjvmD82X/AIAFbEUFxX5peE6Nm9NQrC1Jw/y7OBNZRT4WOsRct2r/NLzuidKOOW3fpqGwicvKsT0f7nNWzwYuS3zH+K4/dvm4lP1ZGu1feC7Hm2mY21fHDQBgWuLMWRRNyxNz2lqLsfiYZBwAAOatWIBRb8NxTBr+ugsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWDT+BcAsT1XvJMj/AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAZCAYAAACl8achAAABpUlEQVR4Xu2VSyhFURSGV2EkpQwMlBmllIGBKAOZS3mmPMJUmZkbMVImZgYMpShlhCJFQnlPRVGeA4Uk/r+9rrvsc+7tDjedr77uPv9e+9x9z1ntK5KQkJCQ8BepgEuwzZ8IlQ84rON1eGvmguQLzpjrEs3yTRYUE+I2aKnVrNHLg4GbO/OyOc2DhZtrjcmC3XSnRDeXp1mLyd7hItyG0/Baa+iafpZpbb9mW3BcM7IJp+AbrIbd4tYtwEO4AXd+qrNwLm7hoMk+4ay5ZuvU65ibHjFzXFsIR2EBLNXMzpNJuByT95mxzbPCoj1xm+GYxx6fguUBDuh4FzanpyJfMi/urVypfACV4uruTJ5a1wMvdEz8+8XCIr+ffWrgEdyHq96c/yU8Nk+9jLCuww9BFzw21/79IrRLDkVgRdyrjyNuvc2qxLUPf/ilyflGSK+4Fk0Rd79fHEgOReL6mXVW8gLv4TMs14wUwUd4AodM3iBuLTfP3mdrPIlrP77JVx3zfhnh4jE/jIF9aeGJY/89g+QGNumYxyGfRnF6OmzqJHNvJ/wrvgEeUnCAUdaVHAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAZCAYAAABQDyyRAAABJ0lEQVR4XmNgGAWjYBSMAuxAFYg3AHEwugQ9wG8gToGy9wHxCyQ5moP/QDwdiS8MFWNBEqMZ6GCAWIYMjKFitmjipIBFDJjmYgUgRdfQxIjWTAAQZQZIUQAWMaI0EwAEzQhjwFTEDBXzg/JB7L9AfBfKhoHvQFzKAEm8jFAxEP0TiOcA8TkGTLMxwHUGiKIkJDGQZfOQ+CAAMwiWUD8BsSGU7Q7Et6FskDp2KBvGxwtACk4D8VEoG+QbHRQVEIBuEIj/GAmD+CDfY1OHF4AUoMc/NoBuEDofBHgYMMXR+SgghIGAAiSArm4VEJcg8W9CaZA6LiRxdH0ogKhEAgQfgPgtEH9GE69mgCS4Z0hioIILpG43EE9jgJiP046HQFyOLjgKRgG9AADxSk7EjrNaTAAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABICAYAAABLN6ksAAAED0lEQVR4Xu3dS+htUxwH8OX9KlzKBN1SJiIDGZgSwozymCmPhIEBijyia4RihAHKRPIIJSEpI7kKEZlgYCIhN6+816+9Vnf91937nH3uOYPb9fnUr/9Zv7X3Of//f/Rt7bX3SQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2A8clevNXP/m2pnrgK3Tk+7OtSvXr7mu6eaq+r7v5TqwmwMAYIYT0xCojijj48t4mc9yvdWMP017BrJ4n0PK6wh1MT5l9zQAAJf0jRG/5Hqu632Q69iu1zo6jYe635vXr+d6uRmHOGfsPACA/5UX0rCatb2fGFFX1y7q+jfn+qbrtaaCV9t7o4yPa3pT5wEAbEQEjbfLz1O7uX3B+7m+zXVMP7HA1Wn4e87u+leW/pSp4LXsfzN1HgDA2v5pXl+V9q3QcVMafp+6V2wVd6Th3DO7/mWlP2UqeEWvX62rzk3D/DP9BADAur7L9Xkzjr1dETxi8/zTpdpVpVtzPVzqxlyP5nqojJ/IdW8aznky1+NlbpVVsWpHGn6PC/qJFdyZxgPbpaU/ZVFgu6JvFjH3Rd8EAFjXoWkIGic1vWtL7+QyfrWMW+enrZcZYz4CVhUB765m/FIa7rqcK97vwr65F2owO6vrX176UxYFtj78he253umbAACbEKtjfTD5uutFYPsj1ydN74RcRzbjOP6eZhyB7fZmHOKYCDZzxWrdX7lO7ydWcFoaPve8rn996U9ZFNgO63qHl34Vjw0BANiYCBrt/rXai2eOVRHYDir9c0ovAlsElSrm2hW1CGy3NePYfxbHzH1gbeu+NB6e5opz467Q1iO5Huh6rXgMyNhnzum1/zsAgLX1K0l/53qqGYcIbCEeGhvHxub6CGxxObUaC2wf53o+DXd2vtbM7a0IjR/m+iptDYvL1H1srX78Y+nFnrwqxjc04wfTcIm1ilW7+v/rCwBgI2LzfISLbeXnn2n88mMNbKFeSozA1t61ORbY2hW2n9NwM8KmvJLrhzT/8uOzaQij8TN+11u2Tqczcn3Z9eKbEeLYeJxIhM/2obmhD2m1fmsPAgBYR9wEMGc1qA1sIYLST2nPwLZoD9u7ad5nreq6vgEAsD+JAPVR3xwxdjkzzu0D2/3NOAJbXIqsXky7A1u/nwwAgAkRoJZ9F2d8D+f3abhLtHVxroPL611pOCZW3uK4nWnYExbjuMxaxR2fj6Xh2xQAAFgiVsfstQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANiU/wAkN9hHciVrEQAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABICAYAAABLN6ksAAAEVUlEQVR4Xu3dS8huUxgH8OUWSu6Xgdtxl4lMDGQi5BIldzMxkTIQJSQKIyUhTEzIgIEwcZuiXFIid0ruIbfcE+v51t48Z513f+97Psl5nd+v/r17Pevd53v3GT3ty9qlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABL7rqa72p+rLm4m1vNIvs9VfNHzdc1W3Zzl9c8WLNLzU41Z5f2PQAAktdrnk7j12qeTeMpi+wXjdo2w/YFw/jAv6fLTUMtZ580DwCwlI7uC/9QNEm9qO3cF5Mdy/R+o8drHknjMDZlo+trbq+5q+b4VAcAWHpx1urtmldrtu7mNsbeZbrx+qgvJn3jNcq1J4fxrqnW73dtzXFpDAAw1wmlNRSf1rzXzW2qnqn5vKx+RmzKhWW68ZpVH03NR+2Qvpj0+11TWsP2QM1zNevSHADABq4s6zcTsR2X65bFQzX39MU5ri7Tjdes+mhqPmqn9MUk5u9L46uG2uj30h5CAADYwG6lNQ7bpdpUU7Ip26+0Jzb7e8emxBmuWcc479in5qN2Xl8cXFrzVlc7uGb7ND6ntH8jX0YFAFgRlxT7BmSqKVkGYwO6bz/RObPMPsZ5xz41H7Uj+2K1f5n9/d7upX3vsX4CACCahDtm1OL+sGUST29+VvN8zRbd3CxHlNmN1FRDNpqaj9q2XS3OWkZ9XIMtmsnR+6XdR5fFd7/qagDAZm7P0pqEfBnuhqGWF3o9dvi8oubE0i7nnTbUTh8+QywAe3PNYakWYpHZU9M4lueIBWN3qDk51dfqtpqH++ICphqvOIYpcZ/Z1H69vhbrtYU47pj7Ic2NT63emWoAAOWN0pqEi1ItxvkM1dh0xMKwl6X6rzU31hw+jGPV/6OG7ZNq3hm2Hx0+Q25gPk7bfWOziCdKezPAHv3ERoi/e0ka3zLURvH/EOP+983bL9ZVG/frM4r77fJv/7nmpzQGAFgRDcSLpTVjsf3b+tN/NSwhLjXmBV6jwTg0jeN7H6bE+KCaL7ra6Py03TdEU+L3vFTasiP5IYm1ipv+42+/UPNKacfUX06NhxjiKdps3n59kzamb8ji8mc0rjHXvykBAGBFNApn9MVOnCn7vrTV+7NoUtal8ayma6/Snn6c5ay0PWvfWeJSat9QAQD8b8XLxhdplN7sC4M4G5ffjRlroeUzUeMyFh+k2v1pOy+BscjvAADY7LxcFmuU4l61fFkvxP1XcTkv7iG7d6iFeN3SLzWfpNoxpe0XjVuccQvflLbvuTXf1nxZ2lk8AACSaKBitf3V3Fra+mCjaLDuTmMAAP5jB9S8O2xvVdpZtbW8txMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4l/0Jsjn5iIM+dP0AAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGYAAAAZCAYAAADDq1t2AAADHElEQVR4Xu2Y26tOQRjGX6dCkhKKXIkUbpDckEgulBQptw6FP4BSLlw47iskkijFBS5QkkOUcuOsHG4ckpTzMefj+zSz2u9+Zs23Zn22+mzzq6c987zvrL3WzFrvmm+JZDKZTCaT+VusUb1TfVQtplgVKWNPqX6pXqsWUSwT4bbqtOnfVF0w/UakjMWC9PLthb7/rD38/zGbjRL6i5soBt4ANomUsSdUR0wMHBeXM4f8Ls8u1TfVaA6UcF3ik7ubTSJl7Hffn98eljHee2q8WowUt9rzONCinFG9VQ3mQAMwQbHJLfMtsRzrD1PtMzEwTVz8CvlJ4I5b4ttnVU9MrJXoKa7OP1T1oVgKKZMbI5YT8wtOiouP5UAVGLTD9Ad6D5PQKqC+oxRcVnWnWB1ikxjzLbGcmA96iItd4kAVGyU86ATvTVFt822bM0J113t7VG2qi6aP+BZxZQbeVtUh354p9UBp+KA6xoEm4WspiPmWWE7MB5+kyRKGA6I0WFAj7T/CnYo+JteCBSmYJOHJ7Rd3YhbkjCOvEXih44W6kwNNEpvEmG+J5cT8W6qDbKaCA84t8XhhivI21PibTHuihCeHhcHdbkHOAfJSKJ6coxyoyXsJzxPAu8MmUWfsYdU68h5RP8oCCf9RURPtnrvY5u31sYINpl2UPwsvzHAJc+rST/VYXOnsRrEUyq4ZwMM1WFZTP3XsStVy0wdDVNvJi4JVxkHtJ4Mf4t4TFrv//qk659vrjT9ewpPGwqAM4XE+r/rSMfxH4Aa6pnqg6k2xKnCey0wfJZnPHZ9S4K0gv2rsDN8v0yyT1xAkY7eATwpoY8tctqV7btp9xeWOko6PasoTs1ncBXc2KG+vxJXbFLDNxrniqbuh+izh04f34H3yQNVYXgyr5N0kkvn9UsZL6uPdgrF2YWLvGHzoK5guYU5nspSNfxF8MkidJHxJZb5Kx4WZLOHxsDDIK8BdiBzcYYOMnzFclXAiy8BvkRfidiQW1Pi1vo2FQ4lC3hvVVHHvE5QXPG22nOH3zz3/N1MCPmmsYjOTyWQymUzX5DcdYAXzRbMsgwAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHgAAAAZCAYAAAD6zOotAAADv0lEQVR4Xu2YWaiNURTHlylJSoQHpSizEA8idOXBA5lncXVJJIUMGUoZiielJC9kKJmKMkSGIkWmMj8SZR4ezEOs/11737POsu853/nuUee4+1f/zrf/e6/d9337fHuvvYkikUgkEolESovmrLOs36zrrAbZ1TlJGuvbvGdVmTqwhHWQ1cOVu7H2sxbXtIikoj3Ji2/myq1duWFNi9pJGguvibue5sqvMtXVbHS+1u2sFvWEjtaoI59IvhzNDdZX44VIEnuadUyVwUmSARytvHWsbaw9rLWsRqquXjGf5OWMshUpQV+Tjbfa+flIEvvTlScqr6fzXioPgzpMlYtGZ5J/2ARbUeJMIXlJC21FAQwl6WOw8Sud38r4mqSxmMb3ZqqrqSBpc1N5a+gfDPAP1lx3fYH1QtWVC/5Fb7EVCUACg9h+xp/k/AHG19Ql9gxJm17KW8Xa5Pzd7nenqi8YdLBDlX2C0Fh55URX1nfWAVuRg/Ukz9zb+OOcP8P4mrSxWFtRj4xbs5Rk4DVot8F4idhMEqzp77whxi832rI+0N8vK8Q8kmfua3ysl/CHG1+TNvYLZU/NuUA/dpwSgaAHxsM6kaqzEgNblsesu8YP4af3gcaf6Xysn7WRJvY+65A1HaH98y9KOSYIGhvwUnVWIrQhOUQ4Zyty0JTkmfNlwiEKjT1CssZqnqprxLxVZe+F+soJbsgG+XVB78u+sY6yrpDsz565NtB59+v/pZXOu0yyNnkusbaS7AuRUEwliTtMsom/yLpa0zodXUjWXuwf04D7wfNpTjlfg+SpnfGSxq5gLTAe+tquyohZrsres33l5SFJkD4uw1SwS5UxffupBwO8SNUhFsdzyCJxOoMb1Tfhr5HVHg/4s9S19gvFT5E4AaoLoS8O5fGqjOkz9LKTxGIt9rFWI1S7zySzkKeCpA2Sx4JAEDI4DByusVXS6TrAVDHbXV+j7ITBPtA+kq8d0w2EPwu+KrR7rXwfN531yF0D218+xpDE4MCjWCDzxn3jF32Hzn9xVrDMmpQ/1g6qlj3SxHvX9Z2yq5OBQLv+Wvqw7pBkezhq09gBwVYLyYMF7TCtWXBAoRMg218+RlojksGn8Pk4QZnDcUsoXnvdSaZw/EmeKB9fOkCWiWXCE+ovkpJblOyFYv210wn4yHpDss/s4DzQgvWOdY81R/mDSGIx0FirMT0j08VUhBkCe0Jco79IEcCLXmnNAFhTNMi89alXpMx5TpmDb2yh8JW1zFRH/hdwYF7bWhyJRCKRovEHj4AhXXTGafUAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALoAAAAZCAYAAACGhjSxAAAE40lEQVR4Xu2aW6hWRRTHl1pImWFgCYmlPaSRUkEUlE+CSBQ9dUPKoyiGL6JWkOiDGhH2kEigUHnpQlAJ4YvQQxglIt7TTKUHDStvqVmHUkNr/ZlZ7PWtvfZ39ved73qcH/w5e9bMmW8ua++9ZmYTJRKJRCKRSHQtD7JWs55VtqXqur8MZi1gvcoaEW3js+xEovn8x1rOGsqaFdPvsQ7H/FtY56Nd1Mv6IOafZV1VebiGTfiIdYV1f9QXrKMUygq/x7TWRZWfaA+TKZubHRQeWF3JZQpPcgs6Jo4u3Bftu41dEAfV3OnYQA/5dq+ORHt4l7VGpf+mMDf3KFvXgIZPsEbmGSp2dNzZHp6Tfu3YBM/u1ZFoD5iHRx1bV84PGv2pNUaKHH2bsQveIBxwbMIhayC/jkTrGUb+XHi2rkAavspmONTj6C8ru/fmsHh1tJs3WRtYN9iMAc4brMeNrcz8dOR4TaWs8aIjrCG6UEQcHU/iJxwVDcIlyv/GvIoSGUV1tIMZrL9Yg1g3UWgX1hzXMxiDa9YY6fjxGkNhV8Q64yhdiDJHP8Z6xVE1J32f8vX/VFEiUK0Oy8dV9CFrI2s9ax2F38euUllepHw7kMaC7HpFwtCbbQZ14XjdStkC0ja8ntDF40nKymL/XlO2jmaDNix0bJ3Qtmo8XINqAYtS9P12mxFpxnhNYp1m7bQZ9YCQw+Mg5RtZj6PbzgvYj0XZr4zdq6PVPEV+G2D7wxo7jKdrUFluo9D3ojdiM8cL7WyIox+3hsgjlG+8OPp2Yxc8Jz1j0hqUtZ3w6iji7RqFt1UZfqB8G8ZFW4+xD3TkgaRBaKhp5njhJrI+Uhe2gcIcyueJoxf9sOekOCF9x9gElH3dsdk6Wg1+f7+x/RLtAtY1OLndy9pC4ZRYxgyS8G90LP8thZ0tLMwnRhv4h/Ua618Ki7gXKPwfTo/3sbZS8blFK/AWnjj51vQ1XjiUxDWih+9Zy6K9h8I4fcdaEW0A4/Ej6xsKD6gif6sJNOAc5XdZYF9sbDgOhh3H9x6ek8LRYZtr7POj3SJ13GgzWojth7ya4YiCzrf9QBr70Pi2B/1Yydps8sGfrIfi9TTKFufYvahWf6vwNijs2ABr88YLdWG7ElvM2LLERofXR4Qqx5QdD4+GODrusOGUbQFi8PFX32F45Z9n/co6QeFuPcP6LObjqXQy5kG4ltU2HB2x3VtUOVD4PQ3u+lOU1fEb9T++q4fnKbRP4lLok4oSAW+SqqUxDtI3ycdfsWn7dApbvIKtrxXgTaTnSwvzLZQdL8z3vSqN8AdzLn3HWwL52J7Ub/mZ1CBHT1SCV2YZx/qSQuiCD56eM3n2/5HWX4UKtpwA58FmgFBUrhMoO15w9LEqvZb8k3GEcPqr2ZmsXSqdaBCYNBtvelSbXJv3AOtnlcbTDHxO4bNlQULCl6jy8wtbXydRdrzgwPZjMN0vrP8Q7uEbK0QOAtY1WKskGgwGH/v8fbGJslc1hNAO9FJ4yiPsuivawGMUysHh9UHcEgqvcIRqAGHLBQrrJhzQIEzAdTvCuDKUGS+EsegD+oXDOwEhM8YNuzazlR2L0eMU/mcRhd/Yo/IT/QQLRx1/FnEH5T9tRqhhT5IHOmXHK9HF4CkzMl7fTf42XCIxYJhiDYlEIpFIJJrB/wKrwMNre3uyAAAAAElFTkSuQmCC>