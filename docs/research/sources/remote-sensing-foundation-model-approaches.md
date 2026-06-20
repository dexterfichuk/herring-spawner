# Remote Sensing and Foundation-Model Approaches for Detecting Pacific Herring Spawn on the BC Coast

## Executive summary

Pacific herring spawn events along British Columbia’s coast produce dramatic turquoise to milky-white water color signatures that are detectable from space, and there is now a small but directly relevant body of work that has already demonstrated satellite-based detection and mapping in the Strait of Georgia and around Vancouver Island. The most mature operational approaches for BC are based on multispectral sensors such as Landsat 4–9 and Sentinel‑2, combined with historical DFO spawn-survey datasets to train and validate spectral indices or machine-learning classifiers. With only about 20 labeled positive examples, a classical supervised CNN pipeline will be brittle, but modern remote-sensing foundation models (self-supervised pretraining on Sentinel/Landsat or RS-specific vision–language models) can provide strong representations that unlock few-shot or even zero-shot detection when combined with clever prompting, pseudo-labeling, and active learning.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11]

This report summarizes (1) the herring-specific remote-sensing literature and available BC data sources, (2) satellite providers and sensors that are realistically usable for an operational tool, and (3) concrete zero/few-shot architectures and training strategies for building a monitoring system on top of foundation embeddings such as DINOv2, CLAY, RemoteCLIP, SatMAE-style temporal transformers, and newer RS models surveyed in 2026.[^12][^13][^3][^5]

## Herring spawn remote-sensing literature

### Core proof‑of‑concept studies

A key reference is “Satellite Remote Sensing of Herring (Clupea pallasii) Spawning Events: A Case Study in the Strait of Georgia,” which shows that high-density spawning events generate distinctive reflectance in the blue and green bands, enabling detection with Landsat sensors over the Salish Sea. The authors demonstrate that the milt-induced color change persists for hours to days and can be discriminated from other turbidity events by spectral signatures, using Landsat-8 OLI and earlier Landsat missions as primary data sources. NASA’s Earth Observatory has highlighted follow-on work from the University of Victoria Spectral Remote Sensing Laboratory and the Pacific Salmon Foundation (PSF), where researchers are systematically mapping historical and contemporary spawn by combining decades of Landsat imagery with DFO survey records.[^2][^4][^7][^1]

These studies collectively establish that (1) spawn is detectable at Landsat spatial resolution for sufficiently dense events, (2) multi-decadal archives enable reconstruction of historical spawning geography, and (3) imagery from February–May is the critical seasonal window in southern BC.[^4][^1]

### UVic Spectral Remote Sensing Laboratory and PSF projects

The UVic Spectral lab’s herring project explicitly targets the Salish Sea using Landsat 4–9 and Sentinel‑2A/B from 1982 to present, with a goal of both historical mapping and near-real-time detection via Google Earth Engine (GEE). Their description emphasizes using open-access imagery and building an interactive GEE-based tool, suggesting they likely have workflows for pre-filtering scenes by date, cloud cover, and known survey locations, which could inform your data-ingestion pipeline. PSF media also notes that drones are used in tandem with satellite scenes for high-resolution mapping of spawn edges, which is relevant for generating dense labels around spawn polygons for training and validation.[^14][^2]

### Auxiliary ecological and timing literature

Timing of spawn is tightly coupled to season and latitude; Fisheries and Oceans Canada (DFO) and independent studies report that in southern BC, spawning generally occurs from mid‑February through early May, though precise timing varies interannually. A graduate study focused on Howe Sound showed that lunar cycles, specifically the waxing crescent phase, are strongly correlated with spawn timing, whereas sea-surface temperature and photoperiod were weaker predictors, which can help prioritize environmental covariates if you decide to add a temporal/forecasting component. Long-term DFO datasets and analysis from NGOs such as Pacific Wild provide biomass indices and spatial distributions of spawn that can be used both as prior information (where to expect events) and as sources of negative examples in non-spawn years.[^15][^16][^10][^1]

## Data sources for BC herring spawn detection

### Government and NGO spawn datasets

DFO has compiled over 70 years of Pacific herring spawn data along BC shorelines, with cumulative spawn tables and habitat maps that are updated annually. These data underlie published “Herring Spawning Areas of British Columbia” maps and feed into DFO’s internal spawn database, which is used to derive indices for habitat value and long-term trends. Pacific Wild and other NGOs have re-visualized DFO’s biomass and spawn data, including interactive maps showing spawn extent from 1960–2023, often exposing the data in formats that can be scraped or requested.[^8][^16][^10]

The Pacific Salmon Foundation hosts ArcGIS Hub layers related to herring, including maps of spawn density and distribution for specific years such as 1955 and 2020. These ArcGIS layers likely link back to shapefiles or feature services that can be intersected with satellite imagery footprints to generate training labels or priors on likely spawn areas.[^17]

### Satellite imagery platforms and archives

The UVic project and NASA coverage indicate that Landsat (4–9) and Sentinel‑2 are the primary workhorses for spawn detection because they combine adequate spatial resolution (30 m for Landsat, 10–20 m for Sentinel‑2) with decades of coverage and open access via GEE and other APIs. Landsat provides a continuous time series back to 1982, while Sentinel‑2 offers higher spatial resolution since 2015 with more frequent revisit (5 days at mid-latitudes), which is valuable given the multi-day but transient nature of spawn events.[^1][^2][^4]

Other missions such as MODIS or VIIRS have coarser spatial resolution that may miss smaller bays but could be useful for wide-area anomaly screening, while commercial constellations (Maxar WorldView, PlanetScope, Skysat) offer sub‑10‑m or even sub‑meter resolution at higher cost and with licensing constraints. For a system geared to open, reproducible monitoring and leveraging foundation models, the open Landsat/Sentinel stack is generally the best base layer, with occasional tasking of high-res commercial imagery for key sites and for generating precise labels.[^3][^6]

### Access platforms and APIs

Google Earth Engine is already being used by UVic and PSF to implement a herring-spawn imagery tool, and its Python API will let you reproduce many of their workflows, including filtering multi-decadal Landsat collections by spawn season, area of interest, and cloud cover. ESA’s Copernicus Open Access Hub and NASA’s Earthdata portals provide bulk access to Sentinel and Landsat scenes, respectively, and can be integrated with cloud object storage for offline foundation-model processing pipelines. ArcGIS Online/Hub layers from PSF and others can be queried via REST to retrieve spawn polygons that are spatially aligned with imagery tiles.[^2][^3][^17][^1]

## Remote-sensing foundation models relevant to herring spawn

### Overview of RS foundation models

A recent “genealogy” survey of remote-sensing foundation models documents rapid growth in self-supervised and vision–language pretraining on Sentinel, Landsat, and other RS imagery. Models such as SeCo, GASSL, SatMAE, DINO‑based variants, and GeoKR are trained with contrastive or masked-image objectives on massive multi-temporal archives and produce general-purpose embeddings that can be fine-tuned with very limited labels. A curated “Awesome Remote Sensing Foundation Models” repository lists dozens of such models with code and pretrained weights, including SatMAE (temporal multi-spectral transformer), RS-BYOL, DINO‑MM (joint SAR–optical), and others that are directly applicable to Sentinel‑2 and Landsat data.[^5][^6][^3]

These RS-specific models address the domain gap that hurts performance when using generic computer-vision backbones such as ImageNet-pretrained ResNets or DINOv2, making them strong candidates for your application.[^3][^5]

### RemoteCLIP: RS vision–language model with strong few-shot behavior

RemoteCLIP is a vision–language foundation model explicitly designed for remote sensing and trained using a CLIP-like objective on a very large RS-specific image–text corpus. To overcome the limited availability of natural captions in RS, the authors scale data via Box-to-Caption (B2C) and Mask-to-Box (M2B) conversions, turning heterogeneous detection/segmentation annotations from many datasets into synthetic captions and bounding boxes, and further incorporate UAV imagery, yielding a pretraining set over 10 times larger than the union of existing RS datasets.[^13][^18][^12]

RemoteCLIP supports a wide range of downstream tasks: zero-shot image classification, linear probing, k‑NN classification, few-shot classification, image–text retrieval, and even object counting, and it consistently outperforms baseline RS foundation models across 16 benchmark datasets. On RS captioning benchmarks RSITMD and RSICD it improves mean recall by roughly 9 percentage points over previous state of the art, and in zero-shot classification it beats generic CLIP by up to around 6 percentage points average accuracy on twelve downstream RS datasets.[^18][^12][^13]

From a few-shot perspective, RemoteCLIP’s image encoder functions as a strong, semantically aligned RS backbone: the authors show that both linear probes and k‑NN classifiers on RemoteCLIP features are competitive, and that prototype-based few-shot classification on top of its embeddings yields robust performance even when only a handful of labeled examples per class are available. Because the model is released in OpenCLIP format, it can be fine-tuned using the OpenCLIP codebase for task-specific adaptation, though the authors and maintainers caution about dataset licensing for reproducing the full pretraining pipeline.[^19][^20][^12][^13]

### RS vision–language models for zero‑shot (Mammut-based and related)

Google researchers have proposed a remote-sensing vision–language foundation model trained on two large datasets, RS-WebLI and a Google Maps-derived aerial dataset, together totaling about 20 million image–text pairs. They fine-tune the Mammut VLM architecture with contrastive and generative losses and show strong zero-shot classification and retrieval results across several RS benchmarks such as FMOW, RESISC45, and RS image-caption datasets using simple prompts of the form “an aerial image of class name.”[^21][^9][^11][^22]

Complementary work on “A Recipe for Improving Remote Sensing VLM Zero Shot Generalization” introduces attention-based pseudo-labeling and careful dataset curation to improve localization and robustness, demonstrating that RS VLMs can be pushed beyond global scene labels into more fine-grained region grounding without human-drawn boxes. Together with RemoteCLIP, these results suggest that CLIP-style VLMs adapted to RS are a powerful option when labels are scarce and text prompts can encode useful prior knowledge.[^23][^24][^22]

### Self-supervised encoders for few‑shot fine‑tuning

Self-supervised encoders like SatMAE, SeCo, GASSL, RS-BYOL, and DINO-MM provide high-quality feature embeddings for Sentinel‑2 and Landsat that can be leveraged as frozen backbones with lightweight heads for classification or segmentation. In few-shot settings, standard practice is to freeze most of the backbone and fine-tune only a small projection head or adapter layers, which mitigates overfitting while still adapting the representation to the specific task. SatMAE in particular is designed for temporal and multi-spectral data and can ingest time-series stacks of images at each pixel, which is appealing for capturing the onset and fading of spawn color signatures across days.[^6][^5][^3]

DINOv2 and CLAY (generic foundation embeddings) remain useful as baselines or for cross-modal workflows, but the RS-specific models above, including RemoteCLIP and Mammut-based RS VLMs, are more likely to capture subtle spectral differences between milt-induced turbidity and other phenomena (sediment plumes, algae, clouds) because they are pre-trained directly on multi-spectral satellite data and aerial imagery.[^12][^5][^3]

## Satellite / provider options for a BC herring system

### Comparison of key satellite options

The table below summarizes the most relevant satellites and providers for this use case.

| Sensor/provider | Spatial resolution | Spectral bands for spawn | Temporal coverage | Access/licensing | Notes |
|-----------------|--------------------|--------------------------|-------------------|------------------|-------|
| Landsat 4–9 (USGS/NASA) | 30 m multispectral | Blue–green–red, coastal/aerosol | 1982–present, 16‑day | Free open data via GEE/Earthdata | Proven spawn detection; long time series.[^1][^2][^4] |
| Sentinel‑2A/B (ESA) | 10–20 m multispectral | Visible, NIR, red-edge, coastal | 2015–present, 5‑day | Free open data via Copernicus/GEE | Higher resolution and revisit; complementary to Landsat.[^2][^3] |
| MODIS/VIIRS | 250 m–1 km | Ocean color bands | 2000–present, daily | Free open data | Coarse; good for wide-area anomaly flags, not fine-scale bays.[^3] |
| PlanetScope (Planet) | 3–5 m RGB+NIR | Visible & NIR | 2014–present, near-daily | Commercial, subscription/tasking | Great detail for small inlets; licensing costs and restrictions. |
| Maxar WorldView | 0.3–2 m multispectral | High-res RGB/NIR | 2007–present, tasking | Commercial, high cost | Excellent for label generation and validation where budgets allow. |

For an open, scalable system, Landsat and Sentinel‑2 should be the primary data sources, potentially augmented by occasional commercial imagery over priority bays to create high-quality training labels or to confirm ambiguous detections.

### Temporal and spatial design for monitoring

Spawn detection requires careful temporal sampling because events may last from several hours to several days. Landsat’s 16‑day revisit alone is insufficient for dense monitoring, so combining Landsat with Sentinel‑2’s 5‑day revisit significantly improves the chance of capturing events, especially when considering both satellites in the constellation. Seasonal filtering (e.g., February–May in southern BC) and optional lunar-phase constraints derived from ecological studies can further reduce the imagery volume while focusing on high-probability windows.[^4][^15][^1][^2][^3]

Spatially, focusing on known DFO/PSF spawn polygons and adjacent similar habitats (shallow, vegetated bays) provides a natural way to balance exploration (finding new sites) and exploitation (monitoring known ones). Historical DFO habitat maps and PSF Atlas layers can be used to seed a grid of candidate shoreline segments that are then scored for spawn-like anomalies each year.[^16][^8][^17]

## Zero-shot and few-shot modelling strategies

### Zero-shot anomaly and similarity search

A straightforward first step with 20 labeled positives is to use a strong embedding model and perform similarity search: embed your positive spawn patches and then rank candidate patches in new imagery by cosine similarity in embedding space. This can be implemented with RS foundation encoders (e.g., SatMAE, SeCo) or with RS VLMs like RemoteCLIP and Mammut-based models by pairing image embeddings with textual prompts about spawn appearance. Given the distinctive turquoise/white spectral signature, anomaly-detection methods that look for outliers in multispectral feature space within each scene (compared to local background water) provide an additional zero-shot cue, especially when combined with simple morphological constraints (contiguous patches near shore).[^11][^13][^21][^5][^12][^3][^4]

For RS VLMs, “prompt engineering” can be used to create a set of spawn-related textual descriptions (“aerial image of bright turquoise water in a sheltered bay from fish spawning”, etc.), and zero-shot classification can then assign scores to patches by computing image–text similarity and selecting those whose best-matching textual prompt corresponds to spawn.[^22][^13][^21]

### Few-shot classification with RemoteCLIP and related VLMs

RemoteCLIP is particularly attractive for few-shot classification because its visual encoder is trained to align with descriptive RS text, yielding semantically meaningful embeddings even when only a few labeled samples per class exist. In practice, one can freeze the RemoteCLIP image encoder, compute embeddings for labeled spawn and non-spawn patches, and then train a simple classifier (e.g., linear probe, prototype-based classifier, or lightweight MLP) while treating the text side as a source of additional semantic priors.[^25][^13][^12]

The authors demonstrate that few-shot classification on top of RemoteCLIP features outperforms both generic CLIP and RS self-supervised backbones across multiple datasets, and that k‑NN classification in feature space is a strong baseline when labels per class are in the single- to double-digit range. Related work on few-shot RS scene classification with CLIP and prompt learning further shows that tuning prompts (static or conditional context tokens, self-regularizing prompts) can significantly boost few-shot accuracy over naive zero-shot prompts and simple linear probes, suggesting that prompt-learning techniques could also be applied to RemoteCLIP-style models for tasks like spawn vs non-spawn discrimination.[^26][^27][^28][^13][^12]

For your herring use case, a concrete strategy would be to (1) start with RemoteCLIP or another RS VLM, (2) encode your ~20 positive spawn patches and a carefully curated set of negatives, (3) train a prototype-based or linear classifier in the embedding space, and (4) optionally explore prompt-learning approaches that adapt text prompts to better capture the visual semantics of spawn events.[^27][^26][^12]

### Few-shot fine-tuning with self-supervised backbones

With 20 positive examples and many unlabeled or negative patches, few-shot approaches that freeze most of a self-supervised encoder and train a shallow head are attractive. A recommended pipeline is:[^5][^3]

- Use a Sentinel‑2/Landsat-specific foundation encoder (SatMAE, SeCo, GASSL, or similar) pretrained on large RS archives.[^3][^5]
- Extract patch-level embeddings for water-adjacent coastal pixels, using a fixed spatial window around shoreline polygons.
- Train a linear or shallow non-linear classifier on your 20 positives plus carefully curated negatives (e.g., turbid river plumes, algal blooms, clouds over water) with aggressive data augmentation.
- Optionally fine-tune a small number of adapter parameters or last transformer blocks if validation performance supports it.

This architecture uses the foundation encoder as a nearly frozen featurizer, greatly reducing overfitting risk relative to training a deep network end-to-end on 20 labels. You can treat CLAY embeddings and DINOv2 as additional baselines, but RS-specific encoders and RS VLMs will likely perform better due to spectral and geometric priors.[^12][^5][^3]

### Pseudo-labeling and active learning

Given your very small labeled set, pseudo-labeling and active learning offer powerful ways to expand training data.

- Pseudo-labeling: Run your initial classifier or zero-shot similarity search across many historical scenes and select high-confidence spawn detections. Manually vet a subset to confirm, then add them as labeled positives or hard negatives to fine-tune the classifier.[^9][^3]
- Active learning: Each season, prioritize new scenes where the model is uncertain (e.g., mid-range probabilities) for human review. Use expert or community annotations (e.g., Indigenous monitoring groups, NGOs following a monitoring guide) to refine labels.[^29]

Over time, this creates a virtuous cycle where model-guided exploration yields new sites and better training data, improving both recall and precision.

### Temporal modelling

Herring spawn is inherently temporal, with events rising and fading over a period of days and constrained to a seasonal window. Temporal encoders such as SatMAE, or simpler recurrent or temporal-convolutional heads on top of per-date embeddings, can model trajectories of reflectance over time at each shoreline pixel. This allows the system to distinguish persistent features (e.g., sand bars, eelgrass, chronic turbidity) from transient milt events, improving robustness.[^6][^15][^1][^4][^5]

A practical approach is to define a temporal context window (e.g., ±1–2 weeks around a candidate detection) and aggregate features such as maximum blue/green deviation from baseline, event duration, and rate of onset, feeding these into a temporal classifier or threshold rule.[^4]

## System architecture and practical considerations

### Data ingestion and preprocessing

A pragmatic architecture for your tool is:

- Use GEE (or Sentinel/Landsat APIs) to fetch multi-spectral scenes for the BC coast filtered by date (Feb–May), cloud cover, and AOI (known and candidate spawn habitats).[^1][^2]
- Reproject and resample Landsat and Sentinel‑2 to a common grid, handling cloud masking and basic atmospheric correction using standard processing pipelines.[^3]
- Generate coastal water masks using shoreline vectors and shallow depth contours to limit analysis to plausible spawn zones.
- Tile the masked imagery into patches (e.g., 64×64 or 128×128 pixels) with overlaps to capture spawn patches that straddle tiles.

Embedding generation can then be run on a GPU-accelerated pipeline outside GEE (e.g., on cloud VMs) because many RS foundation models are implemented in PyTorch or JAX.[^5][^3]

### Detection, scoring, and visualization

The model layer would:

- Compute embeddings for each patch via the chosen foundation encoder.
- Apply one or more scoring heads: similarity to labeled positives, anomaly scores relative to local background, and zero-shot image–text similarity if using a VLM.[^21][^12][^3]
- Fuse scores using simple ensembles or calibrated probabilities, flagging patches above a threshold as candidate spawn.

Detected patches can be merged into polygons and visualized in a web map alongside historical DFO/PSF layers, with probability scores and temporal context to support expert review. Building an interactive viewer similar to the UVic GEE tool, but tailored to your model outputs, would support both quality control and engagement with partner organizations.[^17]

### Collaboration and leveraging existing work

Because the UVic Spectral lab and PSF are already building a herring-spawn tool on GEE using Landsat and Sentinel‑2, and PSF is actively communicating about this work, there is an opportunity to collaborate rather than start entirely from scratch. Their existing spawn masks, drone mapping campaigns, and labeling protocols could substantially increase your labeled dataset beyond the 20 Google Earth Engine positives you currently have. DFO and community-monitoring groups (e.g., Indigenous/NGO efforts documented in monitoring guides) are further potential partners for sharing ground-truth data and validating detections.[^10][^14][^29][^8][^2][^1]

## Recommendations tailored to your situation

1. **Base imagery stack**: Use Landsat 4–9 and Sentinel‑2 as the core data sources, accessed via GEE for rapid prototyping and via bulk downloads for model training, focusing on February–May for southern BC and aligning with DFO/PSF spawn datasets.[^2][^1][^4]
2. **Foundation backbone**: Start with an RS-specific self-supervised encoder such as SatMAE or SeCo instead of generic DINOv2, using it as a frozen backbone to extract patch embeddings; consider RemoteCLIP as an alternative backbone when you want strong text alignment for prompt-based zero-shot classification.[^12][^5][^3]
3. **Zero-shot / similarity baseline**: Implement an embedding-similarity search using your 20 labeled positives to find visually similar patches in historical imagery; combine this with RemoteCLIP or Mammut-based RS VLM text–image similarity scores driven by spawn-specific prompts, and manually curate a small number of additional positives and hard negatives from this search.[^21][^12][^3]
4. **Few-shot classifier**: Train a shallow classifier on RS-foundation embeddings with your curated labels, freezing most backbone layers and focusing on robust augmentations and strong negative selection; for RemoteCLIP and CLIP-like models, prototype k‑NN and prototype-based classifiers as simple but strong few-shot baselines.[^26][^27][^12]
5. **RS VLM experiments**: Experiment with a remote-sensing VLM such as RemoteCLIP or the Mammut-based model trained on RS-WebLI/Google Maps, using prompt-based zero-shot classification for spawn vs non-spawn patches and comparing performance against the self-supervised encoder approach.[^13][^9][^11][^21]
6. **Pseudo-labeling and active learning**: Each season, run the model across the BC coast and review high-confidence new detections plus uncertain cases; integrate feedback from partners (PSF, DFO, local Nations) to rapidly grow the labeled dataset.[^29][^8]
7. **Temporal modelling upgrade**: Once a baseline spatial model is working, incorporate temporal sequences using a temporal encoder (SatMAE or custom) to exploit the rise-and-fall dynamics of spawn.[^6][^5]

Taken together, these steps outline a realistic path from your current 20 labeled samples to an operational, foundation-model-powered monitoring system that can both rediscover known spawn sites and flag new ones along BC’s coast.

---

## References

1. [Spawning Spectacle - NASA Science](https://science.nasa.gov/earth/earth-observatory/spawning-spectacle-154243/) - Spectral Remote Sensing Laboratory Herring Spawn Habitat: Spatiotemporal analysis of historical spaw...

2. [Herring — SPECTRAL Remote Sensing Laboratory](http://uvicspectral.com/herring) - SPECTRAL Remote Sensing Laboratory · Herring · Herring Spawn Habitat: Spatiotemporal analysis of his...

3. [A Genealogy of Foundation Models in Remote Sensing - arXiv](https://arxiv.org/html/2504.17177v3) - Foundation models have garnered increasing attention for representation learning in remote sensing. ...

4. [Satellite Remote Sensing of Herring (Clupea pallasii) Spawning ...](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL092126) - An online search of fish spawn resulted in digital photos of herring spawning events, where high con...

5. [Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models - GitHub](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models) - A collection of papers, datasets, benchmarks, code, and pre-trained weights for Remote Sensing Found...

6. [[PDF] FOUNDATION MODELS IN REMOTE SENSING](https://elib.dlr.de/212882/1/ibm_presentation.pdf) - Introduction to Self-Supervised Learning. 2. SSL on Sentinel 2 data: a forest-monitoring use-case. 3...

7. [Satellite Remote Sensing of Herring (Clupea pallasii) Spawning ...](https://ui.adsabs.harvard.edu/abs/2021GeoRL..4892126Q/abstract) - In this proof of concept study, we show how satellite remote sensing can be used to detect and monit...

8. [[PDF] Herring Spawning Areas of British Columbia - Canada.ca](https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/40897461.pdf) - Spawn habitat maps are based on indices calculated in cumulative spawn tables. These maps are update...

9. [A Recipe for Improving Remote Sensing VLM Zero Shot ... - arXiv](https://arxiv.org/html/2503.08722v1) - In this work we introduce two novel image-caption datasets for training of remote sensing foundation...

10. [[PDF] Canadian Manuscript Report of Fisheries and Aquatic Sciences 2714](https://publications.gc.ca/collections/collection_2007/dfo-mpo/Fs97-4-2714E.pdf) - Pacific herring spawning data has been collected along the shorelines of British. Columbia (BC) for ...

11. [arXiv:2503.08722v2 [cs.CV] 17 Mar 2025](https://arxiv.org/pdf/2503.08722.pdf) - by A Barzilai · 2025 · Cited by 8 — In this work we intro- duce two novel image-caption datasets for...

12. [RemoteCLIP: A Vision Language Foundation Model for Remote ...](https://huggingface.co/papers/2306.11029) - Join the discussion on this paper page

13. [A Vision Language Foundation Model for Remote Sensing - arXiv](https://arxiv.org/abs/2306.11029) - We propose RemoteCLIP, the first vision-language foundation model for remote sensing that aims to le...

14. [Mapping historical herring habitat from space - YouTube](https://www.youtube.com/watch?v=rHDlLZj9njs) - work will help spot current spawning activity that might not be ... herring habitat has changed on B...

15. [Predicting Pacific Herring Spawn Events in the Howe Sound, British ...](https://open.library.ubc.ca/collections/researchdata/items/1.0448460) - This study investigates the role of lunar cycles, sea surface temperature, and photoperiod in determ...

16. [Herring - Gone with the Tides - Pacific Wild](https://pacificwild.org/herring-gone-with-the-tides/) - Biomass of herring spawn estimated by surface and dive surveys from 1960-2023. Source: DFO. Explore ...

17. [Pacific Salmon Foundation - ArcGIS Hub](https://hub.arcgis.com/search?tags=herring+spawn) - This map visualizes approximate density and distribution of herring spawn events for 1955 compared t...

18. [A Vision Language Foundation Model for Remote Sensing》2024 ...](https://blog.csdn.net/qq_46981910/article/details/140643774) - RemoteCLIP是首个针对遥感领域的视觉-语言基础模型，旨在学习具有丰富语义的视觉特征和与文本嵌入对齐的鲁棒特征，以实现无缝的下游应用。

19. [Fine tuning? · Issue #20 · ChenDelong1999/RemoteCLIP - GitHub](https://github.com/ChenDelong1999/RemoteCLIP/issues/20) - However, as the RemoteCLIP checkpoint is provided in OpenCLIP format, it's possible to finetune Remo...

20. [dataset · Issue #21 · ChenDelong1999/RemoteCLIP - GitHub](https://github.com/ChenDelong1999/RemoteCLIP/issues/21) - This is impressive work! I do have a small query regarding the intellectual property rights involved...

21. [A Remote Sensing Vision-Language Foundation Model for Zero ...](https://research.google/pubs/a-remote-sensing-vision-language-foundation-model-for-zero-shot-tasks/) - The first, zero-shot classification, tests the ability of the model to classify a remote sensing ima...

22. [A Recipe for Improving Remote Sensing VLM Zero Shot ... - arXiv](https://arxiv.org/html/2503.08722v2) - The RS-WebLI dataset is comprised of aerial and satellite imagery taken from the WebLI dataset (Chen...

23. [[Revue de papier] A Recipe for Improving Remote Sensing VLM Zero Shot Generalization](https://www.themoonlight.io/fr/review/a-recipe-for-improving-remote-sensing-vlm-zero-shot-generalization) - The paper outlines the development and training of enhanced Vision-Language Models (VLMs) specifical...

24. [[Literature Review] A Recipe for Improving Remote Sensing VLM ...](https://www.themoonlight.io/en/review/a-recipe-for-improving-remote-sensing-vlm-zero-shot-generalization) - The paper outlines the development and training of enhanced Vision-Language Models (VLMs) specifical...

25. [RemoteCLIP Classification Model: What is, How to Use - Roboflow](https://roboflow.com/model/remoteclip) - You can use RemoteCLIP to calculate image embeddings. These embeddings can be used for: Zero-shot im...

26. [Few-Shot Remote Sensing Image Scene Classification with CLIP ...](https://lrc.perdanauniversity.edu.my/sdi/few-shot-remote-sensing-image-scene-classification-with-clip-and-prompt-learning/) - However, their performance is often constrained by the scarcity of labeled data and the high cost of...

27. [Few-Shot Remote Sensing Image Scene Classification with CLIP ...](https://arxiv.org/html/2510.24321v1) - Our experiments span multiple benchmark remote sensing datasets, evaluating performance under few-sh...

28. [[PDF] zero shot remote sensing scene classification via contrastive vision](https://centaur.reading.ac.uk/119820/1/rs-clip.pdf) - We conducted experiments on four benchmark datasets and showed considerable performance improvement ...

29. [Herring Spawn Monitoring Guide - Project Watershed](https://projectwatershed.ca/2026/02/11/herring-spawn-monitoring-guide/) - Welcome to our herring spawn monitoring guide! Here you will find all of the information you need to...
