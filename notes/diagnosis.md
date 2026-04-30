# Diagnosis: Val AUC 0.96 → LB 0.558

## Root Causes (ranked by impact)

### 1. 28 species have ZERO focal training data (critical)
- 25 insect sonotypes + 3 amphibians exist ONLY in train_soundscapes labels
- The model literally cannot predict these species from focal-only training
- These species are heavily represented in soundscape labels (e.g., 517063 appears 626 times, 47158son25 appears 168 times)
- If these species appear in test, our model outputs ~0 for all of them → AUC ≈ 0.5 for each

### 2. Validation metric is misleading
- We validate on focal recordings (clean, single-species, close-mic)
- Test is multi-species soundscapes (distant, noisy, overlapping calls)
- Val AUC 0.96 measures "can the model distinguish species in clean audio" — irrelevant to test performance
- Need: validate on soundscape data with class-wise macro AUC

### 3. Domain shift: focal vs soundscape
- Focal: single species, close microphone, high SNR, variable duration
- Soundscape: multiple species, distant, low SNR, background noise, fixed 5s segments
- Model overfits to focal characteristics (clean spectrograms, single dominant call)

### 4. Soundscape data severely underutilized
- Only 1478 labeled segments from 66 files, covering 75 species
- 5x oversampling is naive — still dominated by 35k focal samples
- Unlabeled soundscapes (hundreds of files) not used at all

### 5. Metric understanding
- Competition metric: class-wise macro AUC (AUC per species column, averaged)
- Equal weight to rare species and non-birds
- Our model is weakest exactly where it matters most: rare species and non-birds

## Data Summary
- 234 classes total: 162 birds, 35 amphibians, 28 insects, 8 mammals, 1 reptile
- ~35,549 focal recordings (18k XC + 12k iNat)
- 1,478 labeled soundscape segments from 66 files, covering 75 species
- 28 species have NO focal data (25 insect sonotypes + 3 amphibians)
- Many more unlabeled soundscape files available

## Priority Actions (by expected LB impact)

1. **Pseudo-label unlabeled soundscapes** — biggest single lever historically
2. **Train on soundscape-only species** — fix the 28-species blind spot
3. **Validate on soundscape data** — stop optimizing the wrong metric
4. **Heavy background mixing** — bridge focal→soundscape domain gap
5. **Upgrade backbone** — EfficientNetV2-S or eca_nfnet_l0 for better features
6. **Ensemble** — 2-3 diverse models with rank averaging
