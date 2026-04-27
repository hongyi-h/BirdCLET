# BirdCLEF 2026 Competition Overview

## Task
Identify which species (birds, amphibians, mammals, reptiles, insects) are calling in 1-minute field recordings from the Brazilian Pantanal.
- Multi-label classification: predict probability of presence for each of 234 species per 5-second segment.
- Metric: **macro-averaged ROC-AUC** (skipping classes with no true positives in test set).
- Threshold-free: only ranking matters, not calibration.

## Constraints
- Submission: Kaggle notebook, CPU only, 90 minutes, no internet.
- External training allowed: train on own GPU → upload weights as Kaggle Dataset → notebook loads and infers.
- ~600 test soundscapes, 1 min each, 32kHz ogg → 12 non-overlapping 5s segments each → ~7200 rows.

## Data Summary

| Source | Count | Coverage | Nature |
|---|---|---|---|
| train_audio | 35,549 recordings | 206 species | Clean focal recordings (XC + iNat) |
| labeled train_soundscapes | 1,478 segments, 66 files, 9 sites | 75 species | In-domain multi-label field recordings |
| unlabeled train_soundscapes | Unknown count | Unknown | Same sites/format as test, no labels |

### Species Distribution (234 total)
- Aves: 162 (34,799 train_audio samples)
- Amphibia: 35 (451 samples)
- Insecta: 28 (199 samples)
- Mammalia: 8 (99 samples)
- Reptilia: 1 (1 sample)

### Critical: Soundscape-Only Species (28)
These species have NO focal recordings in train_audio. All training data comes from labeled soundscapes only:
- 25 insect sonotypes (47158son01-son25): 6-168 segments each
- 3 amphibians: 1491113 (158 seg), 517063 (626 seg), 25073 (24 seg)

### Species Coverage
- In train_audio ONLY: 159 species
- In soundscapes ONLY: 28 species
- In BOTH: 47 species
- No training data: 0 species

## Key Challenges (Priority Order)
1. **Domain shift**: focal recordings (clean, close, single-species) vs soundscapes (noisy, distant, overlapping)
2. **Soundscape-only species**: 28 species can only be learned from limited soundscape segments
3. **Class imbalance**: 14 species with <5 train_audio samples; non-bird taxa severely underrepresented
4. **Multi-taxa heterogeneity**: frequency ranges and call patterns differ across taxa
5. **CPU inference budget**: must process ~600 files in 90 minutes
