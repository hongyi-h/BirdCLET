# SOTA Analysis & Cross-Validation (2026-05-02)

## Competition Summary
- **Task**: 234-class multi-label species ID in 5s soundscape segments (Brazilian Pantanal)
- **Metric**: Class-wise macro AUC (AUC per species column, averaged)
- **Inference**: CPU-only, ~90min budget for ~600 x 1min soundscapes
- **Key constraint**: Equal weight to rare species → non-birds (insects, amphibians) disproportionately important

## BirdCLEF Historical Winners

### 2023 (1st: "Correct Data is All You Need")
- Data quality > model complexity
- SED ensemble: `eca_nfnet_l0`, `convnext_small_fb_in22k`, `convnextv2_tiny`
- Class-balanced sampling (freq^-0.5), focal loss, MixUp, SpecAug
- ONNX for inference

### 2024 (1st)
- Surprisingly simple: `efficientnet_b0` + `regnety_008` on 10s mel chunks
- Trained with CE/softmax, inferred with sigmoid
- Pseudo-labeled unlabeled soundscapes
- Filtered bad chunks with Google bird vocalization classifier
- OpenVINO/joblib/RAM caching
- Private LB ~0.69

### 2025 (1st: Nikita Babych) — MOST RELEVANT (same multi-taxon format)
- Multi-iterative Noisy Student (4 rounds)
- Probability power scaling (p^gamma, gamma≈1.3-2.0) to suppress noisy mid-confidence pseudo-labels
- Separate insect/amphibian handling pipeline
- Ensemble: EfficientNet/NFNet/RegNet variants
- 2nd place: `tf_efficientnetv2_s_in21k` + `eca_nfnet_l0`, scored 0.928 private

## Pretrained Bioacoustic Models
1. **Perch 2.0** (Google): Best bioacoustic embeddings, multi-taxa, TFLite-fast. ~0.71 private alone in 2025.
2. **BirdNET** (Kahl et al.): Strong bird classifier, useful for filtering/pseudo-labeling/embeddings.
3. **BirdAVES/AVES2** (Earth Species Project): Self-supervised, good for low-label tasks.
4. **AST/BEATs/AudioMAE**: Generic audio models, usually worse than domain-specific unless heavily adapted.

## Gap Analysis: Our Pipeline vs SOTA

| Aspect | Our Current | SOTA (2025 winners) | Gap |
|--------|-------------|---------------------|-----|
| Backbone | EfficientNetV2-S (single) | Ensemble of 3-5 diverse architectures | High |
| Pseudo-labeling | 1-2 rounds, hard threshold | 4 rounds iterative Noisy Student with power scaling | Medium |
| Non-bird handling | Same pipeline for all | Separate specialist for insects/amphibians | High |
| Pretrained features | ImageNet only | Perch/BirdNET embeddings as auxiliary | High |
| Data cleaning | None | Manual inspection of rare classes | Medium |
| Inference optimization | ONNX | OpenVINO FP16 + batch + precompute mels | Low |
| Architecture diversity | Single model | 3-5 diverse backbones | High |
| Input context | 5s | 10s context around 5s target | Medium |

## Highest-Impact Actions (ranked by expected LB gain)

1. **Iterative Noisy Student pseudo-labeling** (4 rounds with power scaling) — +0.05-0.10
2. **Architecture diversity ensemble** (add eca_nfnet_l0, regnety_008) — +0.03-0.05
3. **Non-bird specialist** (separate pipeline or heavy oversampling + Perch embeddings) — +0.03-0.05
4. **Perch/BirdNET as teacher/filter** (locate vocal regions, filter bad chunks) — +0.02-0.04
5. **10s context window** (predict 5s but feed 10s) — +0.01-0.02
6. **Data cleaning for rare classes** (<20 samples) — +0.01-0.03
7. **Frontend sweep** (n_fft=2048, n_mels=256, larger image) — +0.01-0.02

## Common Failure Modes to Avoid
- Validating on focal recordings (we already fixed this)
- Softmax inference in multi-label task
- Pseudo-label confirmation bias (need power scaling)
- Over-smoothing short calls (tune per taxon)
- Ignoring CPU time budget (export early, profile)
- Arbitrary 5s crops where primary species is silent
