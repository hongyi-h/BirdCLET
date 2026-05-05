# Strategy Rethink - 2026-05-04

## Objective

Maximize leaderboard rank, not local validation AUC.

BirdCLEF+ 2026 is scored by class-wise macro ROC-AUC over 5-second windows. This makes the problem primarily a per-class ranking problem under sparse positives, not a probability calibration problem.

Current facts:

- Our confirmed public LB anchor: `model_v2s_full_melmix.onnx` scored `0.801`.
- Competition data is available under `data/BirdCLEF+ 2026/`.
- External public inputs are available under `data/`:
  - Perch ONNX.
  - Perch meta.
  - distilled SED 5-fold ONNX.
  - train-audio Perch head.
- Pretrained Perch SavedModel is available under `pretrained_models/`.
- `src.check_artifacts --check_precomputed --check_external` passes.

## First-Principles Diagnosis

The hidden test unit is a 60-second soundscape split into twelve 5-second rows. A real species call is often temporally structured: repeated, persistent, bursty, or site/time dependent. A model that scores each 5-second mel independently is throwing away information present in the input format.

Macro AUC means every class column matters. The common failure mode is not just missing frequent birds; it is poor ranking for rare, weak, non-bird, or soundscape-only classes. A single ImageNet-initialized CNN trained from competition audio is unlikely to learn a robust acoustic representation for all 234 classes, especially classes with no focal audio.

The high public notebooks solve the representation problem first:

1. Use Perch v2 as the acoustic foundation model.
2. Preserve the 12-window sequence.
3. Add public SED as a complementary local spectrogram branch.
4. Blend in rank space because ROC-AUC only needs ordering and heterogeneous model scales are incompatible.

Therefore the highest-leverage move is to make the Perch/ProtoSSM/SED/head branch our main submission path. Our CNN is useful only as a low-weight diversity branch after the high baseline is working.

## Current Strategy

### Mainline

Use `reference/bird26-wliilamsam-0943-with-train-audio-head.ipynb` as the first target because it already combines:

- ProtoSSM / Perch sequence branch.
- public distilled SED branch.
- train-audio Perch logistic head.
- rank blend: 50% ProtoSSM, 35% SED, 15% train-audio head.

This is the shortest path to a 0.94-level public score because it uses the exact public artifacts now present locally.

### Our Unique Contribution

Do not try to replace the reference branch yet. Add our model only after the reference branch is submitted.

Initial probe:

- 47.5% ProtoSSM rank.
- 32.5% SED rank.
- 12.5% train-audio Perch head rank.
- 7.5% our `model_v2s_full_melmix.onnx` rank.

Reason: our CNN has lower standalone quality but may be decorrelated from Perch/SED on some classes. A small weight can help private LB if it adds independent ordering; a large weight will likely damage the public score.

### Secondary Unique Work

If the reference+CNN blend does not improve:

1. Train our own Perch embedding head from `train_audio` and labeled soundscapes, using class-balanced logistic/linear models and rank-blend it at 5-15%.
2. Use Perch/SED predictions, not our 0.801 CNN, as the teacher for pseudo-labeling unlabeled soundscape windows.
3. Train a compact local SED/CNN branch on those pseudo labels only as a diversity branch.

## What To Stop Doing

- Do not train `nfnet`/`regnety` as the next main action.
- Do not run multi-round pseudo-labeling from the current CNN teacher.
- Do not tune blend weights on train-soundscape dry-run as if it were LB.
- Do not add more public checkpoint datasets unless they either improve public LB in a controlled probe or add clear private diversity.

## Submission Sequence

1. Submit exact reference high baseline:
   - `ProtoSSM + SED + train-audio head`.
   - Expected purpose: establish 0.94-level baseline.

2. Submit reference plus our CNN:
   - Small CNN rank weight, 5-10%.
   - Keep only if public LB improves or drops negligibly with a plausible private-diversity reason.

3. Submit a safer public-private blend:
   - If public score drops too much, reduce our CNN to 2.5-5%.
   - If public improves, try one nearby weight only.

4. Only then spend GPU time on new training.

## Decision Rules

- If exact reference public LB is below 0.93, debug path/data/model loading before inventing new modeling.
- If exact reference public LB is around 0.94, prioritize low-risk blend probes.
- If adding our CNN lowers public LB by more than 0.003, do not keep it unless we have class-level evidence that it helps hard non-public classes.
- If runtime approaches Kaggle's limit, drop our CNN branch before dropping Perch/ProtoSSM or SED.

## External Check

Web spot-check on 2026-05-04 found current public Kaggle-related material still emphasizing:

- hidden test soundscapes populated only during scoring,
- approximately 600 one-minute 32 kHz test soundscapes,
- 5-second prediction windows,
- macro ROC-AUC and CPU-only inference constraints.

These match the local competition understanding and do not change the strategy.

Sources checked:

- Kaggle BirdCLEF+ 2026 repack data page: https://www.kaggle.com/datasets/llkh0a/birdclef-2026-repack
- Public analysis report mirror: https://storage.googleapis.com/kaggle-forum-message-attachments/3420357/39213/BirdCLEF2026_Analysis_Report.html
