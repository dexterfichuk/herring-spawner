# Research: Zero-Shot Herring Spawn Detection via Foundation Models

This document evaluates creative, unsupervised methods for detecting herring spawn using pre-trained DINOv2 and Clay foundation models, without additional training or fine-tuning.

---

## 1. DINOv2 Strategies (RGB Thumbnails)

### 1.1 Attention Map Analysis
DINOv2's self-attention mechanism indicates which parts of an image are semantically important.
- **Concept:** Visualize the [CLS] token's attention to spatial patches. Spawn images should show high attention density in near-shore water regions compared to non-spawn images where attention might be more uniform or focused on land features.
- **Labeled Data Required:** None (Zero-shot).
- **Computational Cost:** Low (Extra pass to extract attention tensors from the last layer).
- **Expected Improvement:** High for localization; helps filter out images where the model is "looking" at the wrong thing (e.g., a boat or deep water).
- **Implementation:** Use `model.get_intermediate_layers(..., n=1)` and extract `attentions` from the block. Map the 14x14 attention weights back to the 224x224 image grid.

### 1.2 Patch-Level Similarity
Instead of comparing whole-image [CLS] tokens, we compare the individual patch embeddings (384-dim for ViT-S).
- **Concept:** Define a set of "prototypical spawn patches" from a single known image. For any new image, calculate the similarity of every patch to these prototypes. If a significant percentage (e.g., >15%) of patches exceed a similarity threshold, flag as spawn.
- **Labeled Data Required:** Minimal (1-2 confirmed positive images).
- **Computational Cost:** Medium (N x M patch comparisons per image).
- **Expected Improvement:** Very high. This overcomes the "77% false positive" issue of the [CLS] token by focusing on the local texture of spawn rather than the global composition of the coastal scene.
- **Implementation:** Reshape the output tensor from `[1, 197, 384]` to `[196, 384]` (removing CLS). Use `torch.cdist` or cosine similarity against reference patch vectors.

### 1.3 Augmentation Consensus (Test-Time Augmentation)
- **Concept:** Pass multiple random crops/rotations of the same image through the model. Calculate the variance of the embedding scores. Real spawn (which covers a wide area) should produce consistent high scores across all crops, while a small white object (like a boat) will only trigger a high score in one or two crops.
- **Labeled Data Required:** None.
- **Computational Cost:** High (K passes per image).
- **Expected Improvement:** Good for reducing false positives from small localized features (surf, foam, boats).
- **Implementation:** Wrapper around `scripts/run_embeddings.py` using `transforms.RandomResizedCrop`.

### 1.4 KNN Density in Embedding Space
- **Concept:** Map all candidates into the 384-dimensional DINOv2 space. For a target image, find its K nearest neighbors. Use the density of neighbors (distance-weighted) to score the image.
- **Labeled Data Required:** Uses existing unlabeled candidate pool as a distribution reference.
- **Computational Cost:** Low (Vector search via FAISS or Scikit-Learn).
- **Expected Improvement:** Moderate; identifies clusters of similar events which often correspond to specific lighting/weather conditions that cause false positives.
- **Implementation:** Use `sklearn.neighbors.NearestNeighbors` on the `.npz` files produced by `scripts/run_embeddings.py`.

---

## 2. Clay Strategies (Multi-spectral GeoTIFF)

### 2.1 Band-Specific Anomaly (Spectral Attention)
Clay processes 10+ bands. We can mask or weight specific bands.
- **Concept:** Herring spawn has a distinct spectral signature in the Green (B3) and NIR (B8) bands. We can compare the attention weights of these specific bands against the Blue/Red bands.
- **Labeled Data Required:** None.
- **Computational Cost:** Low.
- **Expected Improvement:** High for separating milkiness (green-shifted) from clouds/surf (white/flat spectrum).
- **Implementation:** Modify `scripts/run_clay_multispectral.py` to extract band-wise attention from the Clay encoder's patch-embedding layer.

### 2.2 Seasonal Position Encoding Analysis
Clay uses Sine/Cosine encodings for Date and Lat/Lon.
- **Concept:** The model has a learned expectation of what a coastal coordinate "should" look like in March vs. July. We can calculate the distance between the *predicted* embedding (based only on metadata) and the *actual* embedding (from the pixels). A large delta suggests a temporal anomaly—exactly what we want.
- **Labeled Data Required:** None.
- **Computational Cost:** Low.
- **Expected Improvement:** Moderate; very creative use of the "Foundation" knowledge.
- **Implementation:** Pass metadata (lat/lon/date) with zeroed-out pixels to get a "baseline" embedding, then compare to the real pixel embedding.

### 2.3 Reconstruction Error (Zero-Shot Anomaly Detection)
Clay is a Masked Autoencoder (MAE).
- **Concept:** MAEs are excellent anomaly detectors. If we mask 75% of a spawn image and ask Clay to reconstruct it, it will likely have a higher reconstruction error than a "normal" shoreline because the "milky water" texture is rarer in its global training set.
- **Labeled Data Required:** None.
- **Computational Cost:** Medium (Requires a decoder pass).
- **Expected Improvement:** High. This is a classic unsupervised approach.
- **Implementation:** Use `model.model.decoder` on the masked patches. Calculate Mean Squared Error (MSE) between original and reconstructed pixels.

---

## 3. Summary Recommendation

| Method | Best For | Labeled Data | Effort |
| :--- | :--- | :--- | :--- |
| **Patch-Level Similarity** | Reducing False Positives | 1-2 images | Low |
| **Attention Maps** | Localization/Review | None | Low |
| **Reconstruction Error** | True Zero-Shot Discovery | None | Medium |
| **Seasonal Delta** | Temporal Filtering | None | Medium |

**Proposed Next Step:** Implement **Patch-Level Similarity** for DINOv2 first, as it directly addresses the high false-positive rate of the current [CLS] approach with minimal code changes.
