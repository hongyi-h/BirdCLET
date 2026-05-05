# Non-submission validation plan - 2026-05-05

## Decision

Yes: we should test most design changes before spending one of the five daily Kaggle submissions.

The non-submission score is a gate, not the objective. Hidden public/private LB remains the final judge, but daily submissions should only be used after a candidate passes proxy checks.

## Why

BirdCLEF 2026 scores class-wise macro ROC-AUC over 5-second soundscape rows. The only in-domain labeled proxy available outside hidden scoring is `train_soundscapes_labels.csv`.

Therefore the first-principles validation stack is:

1. produce submission-like predictions for labeled `train_soundscapes`;
2. align by `row_id`;
3. compute per-class ROC-AUC and macro average;
4. inspect taxon/class failures;
5. submit only candidates that improve or add credible orthogonal rank signal.

## Leakage rule

Any branch trained on `train_soundscapes_labels.csv` must be evaluated with file/group out-of-fold predictions.

Non-OOF predictions are still useful for IO smoke tests, runtime checks, and branch shape checks, but not for model selection.

## Implemented code

- `src/offline_score.py`
  - Scores one CSV or a rank-blend of multiple CSVs against `train_soundscapes_labels.csv`.
  - Reports macro AUC, evaluable class count, positive label count, and per-taxon AUC when taxonomy is available.
  - Can write JSON and per-class CSV reports.

- `src/perch_sed_head_experiment.py --save_features`
  - Saves `perch_meta.csv`, `perch_scores.npy`, `perch_embs.npy`, and `primary_labels.csv`.
  - Enables custom Perch-head training without recomputing embeddings every time.

- `src/train_perch_head_v2.py`
  - Trains a class-balanced linear head from cached Perch embeddings.
  - Uses filename `GroupKFold` and writes `oof_head_v2.csv`.
  - Saves full-data raw-space linear weights to `head_v2_weights.npz` for later inference integration.

Syntax check passed:

```bash
/opt/anaconda3/envs/wqb/bin/python3.12 -m py_compile \
  src/offline_score.py \
  src/perch_sed_head_experiment.py \
  src/train_perch_head_v2.py
```

No local experiment was run.

## Cloud/Kaggle non-submission commands

Run on Kaggle notebook or the cloud server with full data. On Kaggle, replace the input directory with `/kaggle/input/birdclef-2026/train_soundscapes` if needed:

```bash
python -m src.perch_sed_head_experiment \
  --input_dir "data/BirdCLEF+ 2026/train_soundscapes" \
  --output_dir outputs/val_perch_sed_head \
  --save_features \
  --batch_files 16 \
  --io_workers 4 \
  --ort_threads 4
```

Score the produced blend:

```bash
python -m src.offline_score \
  --inputs outputs/val_perch_sed_head/submission.csv \
  --out_json outputs/val_perch_sed_head/offline_score.json \
  --out_class_csv outputs/val_perch_sed_head/offline_class_auc.csv
```

Score branch-weight candidates without resubmitting:

```bash
python -m src.offline_score \
  --inputs \
    outputs/val_perch_sed_head/submission_perch_direct.csv \
    outputs/val_perch_sed_head/submission_sed.csv \
    outputs/val_perch_sed_head/submission_head.csv \
  --weights 0.50 0.30 0.20 \
  --out_json outputs/val_perch_sed_head/score_w_50_30_20.json
```

Train and score a custom Perch head v2:

```bash
python -m src.train_perch_head_v2 \
  --features_dir outputs/val_perch_sed_head \
  --output_dir outputs/perch_head_v2 \
  --use_scores \
  --n_splits 5 \
  --min_pos 2 \
  --C 0.1

python -m src.offline_score \
  --inputs outputs/perch_head_v2/oof_head_v2.csv \
  --out_json outputs/perch_head_v2/oof_score.json \
  --out_class_csv outputs/perch_head_v2/oof_class_auc.csv
```

## Submission gate

Submit only if at least one condition holds:

1. The candidate improves OOF train-soundscape macro AUC over the current 0.944-family proxy.
2. The candidate improves rare/non-bird taxon AUC without damaging Aves too much.
3. The candidate is intentionally orthogonal and will receive small blend weight, with branch correlation lower than the current branches.
4. The change is a pure inference/runtime/bug fix with no ranking-risk ambiguity.

## Next model changes to test offline

1. Custom Perch head v2 trained from saved embeddings:
   - class-balanced logistic baseline;
   - low-rank shared head;
   - taxon-aware regularization;
   - OOF validation by soundscape filename.
2. Weight sweep around current best:
   - ProtoSSM/SED/head neighborhood first;
   - do not burn Kaggle submissions on candidates that lose proxy AUC and add no diversity.
3. Longer-context branch:
   - only useful if standalone OOF soundscape score is high enough;
   - current 0.801 CNN should not receive meaningful blend weight without new evidence.

## External evidence

- Perch 2.0 supports keeping Perch as the main bioacoustic representation: https://research.google/pubs/perch-20-the-bittern-lesson-for-bioacoustics/
- BirdNET and related bioacoustic SED work support domain-specific augmentation and temporal context: https://www.sciencedirect.com/science/article/pii/S1574954121000273
- BEATs-style general audio models are possible diversity branches, but only after CPU/runtime feasibility is solved: https://proceedings.mlr.press/v202/chen23ag.html
