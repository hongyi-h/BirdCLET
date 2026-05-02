# Execution Plan — BirdCLEF 2026

## Current Status (2026-05-02)
- v1 model: EfficientNet-B0, val_auc=0.9599 (focal, misleading), **LB=0.558**
- v3 training: EfficientNetV2-S on 8x MetaX C500, **stopped at epoch 4** (24h for 4 epochs — too slow)
- Root cause: single GPU, num_workers=0, per-sample .ogg decode from AFS
- **v4 pipeline implemented**: DDP, precomputed mels, 3 backbones, iterative Noisy Student, non-bird specialist

## Target: Win (LB 0.80+)

---

## Pipeline v4 — Implemented

### Phase 0: Training Infrastructure (DONE — code ready)
- `src/train.py`: DDP via torchrun, spawn multiprocessing, precomputed mel support
- `src/preprocess.py`: Pre-extract mel spectrograms to .npy
- DataLoader: spawn context, persistent_workers, prefetch_factor=4

### Phase 1: Train 3 Diverse Backbones (code ready)
- `tf_efficientnetv2_s.in21k_ft_in1k` (v2s)
- `eca_nfnet_l0` (nfnet)
- `regnety_016` (regnety)

### Phase 2: Iterative Noisy Student (code ready)
- `src/pseudo_label.py`: Ensemble teacher, power scaling (p^gamma), progressive thresholds
- 4 rounds: threshold 0.7→0.6→0.5→0.4, gamma 1.5→1.5→1.3→1.3

### Phase 3: Non-Bird Specialist (code ready)
- `src/train_specialist.py`: Separate model for 72 non-bird species
- Heavy oversampling, site-split validation

### Phase 5: Ensemble Inference (code ready)
- `inference.py`: Multi-model rank averaging, specialist blending, temporal smoothing
- `src/export_onnx.py`: Supports all backbones + specialist

---

## Execution Commands

See `notes/execution_guide.md` for full commands.

Quick start:
```bash
# Step 1: Train first backbone (8-GPU)
torchrun --standalone --nproc_per_node=8 -m src.train --backbone v2s --epochs 40 --batch_size 128 --lr 3e-4 --num_workers 4 --save_tag v4_v2s

# Step 2: Export and submit to get baseline LB
python -m src.export_onnx --checkpoint best_v4_v2s.pt --backbone v2s --output model_v2s.onnx
```

---

## Experiment Log

| Exp | Backbone | Pseudo | Val AUC (soundscape) | LB | Notes |
|-----|----------|--------|---------------------|-----|-------|
| v1  | B0       | No     | 0.9599 (focal!)     | 0.558 | Baseline, wrong val metric |
| v3  | V2-S     | No     | 0.9410 (ep3, sc)    | — | Stopped: too slow (24h/4ep) |
| v4-v2s | V2-S  | No     | TBD                 | TBD | DDP + spawn + precomputed |
| v4-nfnet | NFNet | No   | TBD                 | TBD | |
| v4-regnety | RegNetY | No | TBD              | TBD | |
| v4-r1 | Ensemble | R1    | TBD                | TBD | Noisy Student round 1 |
| v4-r2 | Ensemble | R2    | TBD                | TBD | |
| specialist | B0 | No    | TBD                 | — | Non-bird only |
| final | All    | R2+    | TBD                 | TBD | Full ensemble |

---

## Key Decisions & Rationale

| Decision | Why |
|----------|-----|
| DDP + spawn | Fix 24h/4ep bottleneck. Single GPU + fork = broken on MUSA. |
| 3 diverse backbones | 2025 winners used 3-5. Architecture diversity > fold diversity for AUC. |
| Power scaling in pseudo-labels | 2025 1st place technique. p^1.5 suppresses noisy 0.3-0.6 predictions. |
| Non-bird specialist | 28 species with zero focal data = ~12% of macro AUC metric. |
| Rank averaging | More robust than logit/prob averaging for AUC metric. |
| Precomputed mels | Eliminates per-sample .ogg decode. 20-50x speedup expected. |
