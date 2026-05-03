# Reference Notebook Design Review - 2026-05-04

## Decision

The design priority changes after reading the reference notebooks in `reference/`.

Our confirmed public LB anchor is `0.801` from `model_v2s_full_melmix.onnx`. The 0.94+ public notebooks are not mainly better versions of our CNN training loop. Their core advantage is a different observation layer:

- Perch v2 ONNX logits and 1536-d embeddings for every 5-second window.
- Sequence modeling over the 12 windows in each 60-second soundscape.
- Public distilled SED ONNX folds as a complementary spectrogram branch.
- Per-class rank blending, not raw probability averaging.
- Small train-audio head on Perch embeddings as a conservative extra signal.

Therefore the shortest path to a high leaderboard score is not training more CNN backbones first. It is to add a Perch/ProtoSSM/SED branch and blend our CNN only as a low-weight diversity component.

## Evidence From Reference Notebooks

Local references checked:

- `reference/birdclef-2026-0-941-onnx-perch-sequence-sed.ipynb`
- `reference/birdclef-2026-onnx-sed-repro.ipynb`
- `reference/birdclef-2026-0-943-better-blend.ipynb`
- `reference/birdclef-protossm-efficientnet-sed-all-public.ipynb`
- `reference/bird26-wliilamsam-0943-with-train-audio-head.ipynb`
- `reference/sub-080-yuriy-dualssm.ipynb`
- `reference/birdclef-2026.ipynb`

Common high-score structure:

1. Split each 60-second file into 12 x 5-second windows, matching the submission rows.
2. Run Perch v2 ONNX to get target-class logits and 1536-d embeddings.
3. Map BirdCLEF labels to Perch labels; use genus proxy logits for unmapped classes where possible.
4. Train tiny in-notebook learners on labeled train soundscapes:
   - MLP/logistic probes on Perch embeddings.
   - LightProtoSSM over the 12-window sequence with site/hour metadata.
   - ResidualSSM to correct first-pass errors.
5. Apply site/hour priors, confidence scaling, adaptive temporal smoothing, and class threshold calibration.
6. Run public `sed_fold*.onnx` distilled SED models.
7. Rank-average heterogeneous branches per class.

Representative weights:

- 0.941 branch: about 60% ProtoSSM/Perch sequence + 40% public SED.
- 0.943+ train-audio-head branch: 50% ProtoSSM + 35% SED + 15% Perch train-audio logistic head.

## First-Principles Diagnosis

The metric is class-wise macro ROC-AUC over soundscape windows. It rewards correct global ordering per class more than calibrated probabilities.

Our current model observes only a mel image from one 5-second window and learns from limited competition labels. It has no foundation bioacoustic prior and only weak temporal context. Adding nfnet/regnety improves diversity, but it does not change the observation bottleneck.

The reference systems start from Perch, a strong bioacoustic foundation model. They also reason over the whole 60-second file. That directly matches the hidden test structure: calls persist, repeat, and occur in bursts; isolated 5-second predictions are under-informed.

Rank blending is important because Perch, SED, and our CNN have incompatible score scales. Since ROC-AUC depends on ordering, rank space is the correct common space.

## Revised Priority

1. Build or adopt a Perch/ProtoSSM inference branch from the references.
2. Add public distilled SED folds as a second branch.
3. Add the train-audio Perch linear/logistic head if the required public weights or reproducible training cache are available.
4. Blend our `model_v2s_full_melmix.onnx` at low weight, initially 5-15%, only if LB improves.
5. Delay further full-data nfnet/regnety training until after the Perch+SED blend is submitted.

## Initial Blend Experiments

Submit these as separate leaderboard probes:

1. Reference reproduction only:
   - 60% ProtoSSM/Perch sequence rank
   - 40% public SED rank

2. Add our CNN as diversity:
   - 55% ProtoSSM/Perch sequence rank
   - 35% public SED rank
   - 10% `model_v2s_full_melmix.onnx` rank

3. If train-audio head is available:
   - 50% ProtoSSM/Perch sequence rank
   - 35% public SED rank
   - 10% train-audio Perch head rank
   - 5% our CNN rank

Only public LB can decide the exact weight. Local train-soundscape dry-run is a sanity check, not a trustworthy optimizer.

## Engineering Notes

- Do not silently fall back to dummy submissions. `inference.py` already refuses missing main ONNX models and constant predictions.
- If we integrate the reference pipeline into our repo, keep it separate from the current CNN inference path first. A single monolithic rewrite risks breaking the known 0.801 anchor.
- Use `python -m src.blend_submissions` for final CSV-level rank blends; it validates row ids, class columns, NaN/Inf values, and sample submission column order.
- Attach required Kaggle datasets explicitly:
  - Perch ONNX and ONNX Runtime wheel.
  - Perch metadata or Google Perch labels.
  - Public distilled SED ONNX folds.
  - Optional train-audio head weights.
- Keep all final branch outputs as CSVs with the same row/column order, then blend in a small deterministic final cell/script.

## Bottom Line

The current CNN pipeline is a valid anchor, not the main route to 0.94+. The high-rank route is Perch foundation features plus 12-window sequence modeling plus public SED plus rank blending.
