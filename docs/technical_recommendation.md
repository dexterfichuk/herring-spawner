# Herring Spawn Detection — Technical Recommendation

## What We Tried

| Method | Result | Why |
|--------|--------|-----|
| DINOv2 single-image SVM | 88.9% train accuracy, 77% FP in practice | Learned shoreline patterns, not spawn |
| Clay embedding deltas | **0.3297 separation** (5.4× better) | Seasonal spectral change is predictive |
| HSV color analysis | Worked on obvious cases | Too many false positives from surf/sediment |
| Temporal validation | Strong signal for multi-date events | 2+ dates = real spawn, 1 date = likely noise |
| Rose visual review | Most reliable | Human comparing seasonal pairs works |

## What We Have
- **10 confirmed spawn images** — too few for ML training
- **Clay delta model** — 0.3297 separation, p=0.0013, best method so far
- **Improved detector** — delta + HSV + texture features, 97% accuracy on test data
- **46 super-positive candidates** — 10 confirmed by rose as real spawn

## Recommended Path

**Stop using single-image DINOv2.** It doesn't work for this problem — it learns shoreline appearance, not spawn events.

### Best approach: Clay delta + human review
1. Use **Clay pre-trained multi-spectral embeddings** on paired pre-spawn vs spawn-season GeoTIFF chips
2. Model the **delta** between embeddings (not raw images)
3. Require **temporal repeatability** (2+ dates above threshold)
4. Send candidates to **human review** for confirmation

### Why this works
- Clay is satellite-native (understands NIR bands)
- Delta removes shoreline bias — learns *change*, not place
- Temporal validation filters out random noise/surf/clouds
- Human review catches edge cases

### Next Steps
1. Build paired dataset: same locations, pre/post spawn windows
2. Extract Clay delta embeddings + spectral change features
3. Train small classifier on delta features
4. Add temporal consistency rule
5. Keep DINOv2 only as cheap thumbnail prefilter
