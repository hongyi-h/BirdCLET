# Execution Plan

## Current Status
- v1 model: EfficientNet-B0, val_auc=0.9599, **LB=0.558**
- Root cause: domain shift (focal→soundscape) + 28 species with zero focal data

## Target
- LB ≥ 0.70 (conservative), stretch goal ≥ 0.75

---

## Phase 1: Train baseline (EfficientNetV2-S, soundscape-aware)

```bash
python -m src.train --backbone v2s --epochs 40 --batch_size 48 --lr 2e-4 --num_workers 4
```

---

## Phase 2: Pseudo-label round 1

```bash
python -m src.pseudo_label --checkpoint best_v3.pt --backbone tf_efficientnetv2_s.in21k_ft_in1k --round 1 --threshold 0.6 --tta
```

---

## Phase 3: Retrain with pseudo-labels

```bash
python -m src.train --backbone v2s --epochs 40 --batch_size 48 --lr 1e-4 --pseudo --num_workers 4 --resume checkpoints/best_v3.pt
```

---

## Phase 4: Pseudo-label round 2 (optional)

```bash
python -m src.pseudo_label --checkpoint best_v3.pt --backbone tf_efficientnetv2_s.in21k_ft_in1k --round 2 --threshold 0.5 --tta
python -m src.train --backbone v2s --epochs 30 --batch_size 48 --lr 5e-5 --pseudo --num_workers 4 --resume checkpoints/best_v3.pt
```

---

## Phase 5: Export and submit

```bash
python -m src.export_onnx --checkpoint best_v3.pt --backbone tf_efficientnetv2_s.in21k_ft_in1k
```

Upload `checkpoints/model.onnx` to Kaggle dataset, use `inference.py` as notebook.

---

## Key Decisions & Rationale

| Decision | Why |
|----------|-----|
| EfficientNetV2-S over B0 | 5x more params, 21k pretrained = better features for rare species. Still fast enough for CPU inference via ONNX. |
| Validate on soundscape (by site) | Focal val AUC is meaningless for LB prediction. Site-split prevents data leakage. |
| Focal loss γ=2 | Rare species (1-5 samples) get upweighted automatically. Better than BCE for long-tail. |
| Background mixing 60% | Most impactful single augmentation for focal→soundscape transfer. |
| Pseudo-label threshold 0.6 | Conservative first round avoids noise. Lower to 0.5 in round 2 after model improves. |
| Temporal smoothing in inference | Adjacent 5s segments are correlated. Cheap post-processing, ~+0.01-0.02 LB. |
| CosineAnnealingWarmRestarts | Better than linear cosine for multi-phase training (initial + pseudo-label fine-tune). |

---

## Risk Assessment

- **If soundscape val AUC < 0.60 after Phase 1**: try `--lr 5e-4` with warmup.
- **If pseudo-labels are noisy (retention < 20%)**: lower threshold to 0.4.
- **If CPU inference too slow**: quantize ONNX to FP16 or switch to OpenVINO.
- **If EfficientNetV2-S too large for Kaggle**: fall back to B0 with same training strategy.

---

## Experiment Log

| Exp | Backbone | Pseudo | Val AUC (soundscape) | LB | Notes |
|-----|----------|--------|---------------------|-----|-------|
| v1  | B0       | No     | 0.9599 (focal!)     | 0.558 | Baseline, wrong val metric |
| Phase1 | V2-S | No     | TBD                 | TBD | |
| Phase3 | V2-S | R1     | TBD                 | TBD | |
| Phase4 | V2-S | R2     | TBD                 | TBD | |
