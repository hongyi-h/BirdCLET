# Train Log — v2s Precomputed — 2026-05-03

Source log: `logs/src.train.log`

## Run

Configuration from log:
- DDP: `world_size=8`, backend `nccl`.
- Backbone: `tf_efficientnetv2_s.in21k_ft_in1k`.
- Epochs: 40.
- Batch size: 128 per GPU, effective 1024.
- LR: `3e-4`.
- Data: `--precomputed`, no pseudo labels.
- Precomputed focal: 106,647 crops.
- Precomputed labeled soundscape: 1,478 segments.
- Split used by this run: train 1,358 soundscape segments, validation 120 segments from sites `S13`, `S19`.
- Train domains after oversampling: focal 106,647, soundscape 35,308.
- Train species: 230.
- Validation evaluable species upper bound: 22.

## Result

Training completed all 40 epochs.

Best validation:
- Epoch 36.
- `val_auc(soundscape)=0.7362`.
- Checkpoint saved as `checkpoints/best_v4_v2s.pt`.

Top validation epochs:
- Epoch 36: 0.7362.
- Epoch 34: 0.7201.
- Epoch 37: 0.7183.
- Epoch 39: 0.7179.
- Epoch 32: 0.7054.

Last epoch:
- Epoch 40: loss 0.0004, val AUC 0.6680.

## Interpretation

This is a successful infrastructure run: precomputed DDP training is fast enough and reaches a much more plausible soundscape CV score than the earlier broken focal validation setup.

Do not over-trust the absolute AUC:
- Validation has only 120 segments and 22 evaluable species.
- Only one bird species is evaluable in this split.
- Training species count is 230, meaning 4 taxonomy classes had no training positives in this split. This is unacceptable for final leaderboard training because macro AUC gives class-level weight to those species if they occur in hidden test.

The best checkpoint is still useful:
- Use it as the first v2s teacher/baseline.
- Export and submit a single-model LB check if fast.
- Do not treat 0.7362 as a reliable global estimate.

## Code Changes After This Run

`src/train.py` was updated after reading this log:
- Site split now protects training label coverage where possible, so a validation split should not remove an entire species from training.
- Added `--train_all_soundscapes` for final/teacher runs where all labeled soundscape positives should be used.
- Added precomputed batch-level mel mixup via `--precomputed_mixup_prob` to restore some overlap augmentation lost by fixed mel precompute.
- DDP barrier now passes the CUDA device id to avoid repeated barrier warnings.
- Resume checkpoint loading now accepts both bare checkpoint names and explicit relative/absolute paths.
- Training now saves `last_{save_tag}.pt` in addition to `best_{save_tag}.pt`; final runs with `--train_all_soundscapes` should prefer `last_*` because their validation metric is leaky.

`src/pseudo_label.py` was updated for the next stage:
- Added `--precomputed` so pseudo-labeling can use `data/precomputed/soundscape_unlabeled/manifest.csv` directly instead of decoding `.ogg` and recomputing mels.
- Empty pseudo-label outputs are now still written with the expected columns, so a high threshold fails gracefully instead of crashing in summary code.

## Next Decision

Shortest useful next step:

1. Export `best_v4_v2s.pt` and run a single-model LB smoke test.
2. Start a second v2s run with the updated split/mel mixup to see whether CV remains stable with full train label coverage.
3. For pseudo-label teacher or final submission, train with all labeled soundscape segments:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone v2s --epochs 36 --batch_size 128 --lr 3e-4 \
  --precomputed --precomputed_mixup_prob 0.3 \
  --train_all_soundscapes --num_workers 4 --save_tag v4_v2s_full
```

The `--train_all_soundscapes` validation number is leaky and should only be used for monitoring. The reason to use it is first-principles simple: final leaderboard models should not deliberately withhold the only positive examples for soundscape-only classes.
For that run, export `last_v4_v2s_full.pt` rather than `best_v4_v2s_full.pt`.

## Verification

- `py_compile` passed for `src/train.py`, `src/check_artifacts.py`, `src/pseudo_label.py`, and `src/preprocess.py`.
- Local `--help` execution could not be completed because the local `/opt/anaconda3/envs/wqb` environment does not have `torch`; server env `birdclef` is still the intended runtime.

## Server Relaunch Failure — 2026-05-03 13:21 CST

Command failed:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone v2s --epochs 40 --batch_size 128 --lr 3e-4 \
  --precomputed --precomputed_mixup_prob 0.3 \
  --num_workers 4 --save_tag v4_v2s_melmix
```

Root cause: the server executed an older `src/train.py`. Its argparse help did not include `--val_ratio`, `--train_all_soundscapes`, or `--precomputed_mixup_prob`.

Decision: do not remove `--precomputed_mixup_prob` to make the old code run. Sync the patched source first, then verify on the server with:

```bash
python -m src.train --help | grep -E 'precomputed_mixup_prob|train_all_soundscapes|val_ratio'
```

## Mel Mixup Relaunch — 2026-05-03 15:30 CST

Command completed:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone v2s --epochs 40 --batch_size 128 --lr 3e-4 \
  --precomputed --precomputed_mixup_prob 0.3 \
  --num_workers 4 --save_tag v4_v2s_melmix
```

Configuration:
- DDP: `world_size=8`, backend `nccl`.
- `--precomputed`, `--precomputed_mixup_prob 0.3`.
- Split: train 1,400 soundscape segments, validation 78 segments from sites `S03`, `S18`.
- Train species: 234.
- Validation evaluable species: 6, all Amphibia.

Result:
- Training completed all 40 epochs.
- Best validation AUC: 0.9044 at epoch 15, saved as `checkpoints/best_v4_v2s_melmix.pt`.
- Final epoch AUC: 0.7436.
- Final checkpoint saved as `checkpoints/last_v4_v2s_melmix.pt`.

Interpretation:
- Mel mixup did not break training.
- The 0.9044 CV number is not comparable to previous runs because the validation split is only 78 segments and one taxon.
- The important win is that training now covers all 234 taxonomy classes. The important remaining problem is validation representativeness.

Code change after this run:
- `split_soundscape_by_site` no longer takes the first random legal sites. For small site counts, it exhaustively searches site combinations, keeps training label coverage, targets the requested validation segment ratio, and then maximizes validation label coverage.
- On the current labels, this should avoid the degenerate `S03+S18` Amphibia-only split.

## Full Soundscape Teacher — 2026-05-03 17:18 CST

Command completed:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone v2s --epochs 36 --batch_size 128 --lr 3e-4 \
  --precomputed --precomputed_mixup_prob 0.3 \
  --train_all_soundscapes --num_workers 4 --save_tag v4_v2s_full_melmix
```

Configuration:
- DDP: `world_size=8`, backend `nccl`.
- `--train_all_soundscapes=True`, so validation is intentionally leaky and only monitors training health.
- Train domains: focal 106,647, soundscape 35,472.
- Train species: 234.
- Validation monitor: 78 segments, 6 Amphibia species.

Result:
- Training completed 36 epochs.
- Validation monitor reached 1.0000 from epoch 5 onward except epoch 35, confirming leakage/memorization rather than generalization.
- Final checkpoint saved as `checkpoints/last_v4_v2s_full_melmix.pt`.
- `checkpoints/best_v4_v2s_full_melmix.pt` also exists, but should not be used for final export because it was selected by leaky validation.

Decision:
- Use `last_v4_v2s_full_melmix.pt` as the v2s full-data teacher/checkpoint.
- Export command:

```bash
python -m src.export_onnx --checkpoint last_v4_v2s_full_melmix.pt \
  --backbone v2s --output model_v2s_full_melmix.onnx
```

Code change after this run:
- `src/train.py` now prints `last_*` as the export checkpoint when `--train_all_soundscapes` is set, instead of misleadingly suggesting `best_*`.

## ONNX Export Failure — 2026-05-03

Command failed before loading the checkpoint:

```bash
python -m src.export_onnx \
  --checkpoint last_v4_v2s_full_melmix.pt \
  --backbone v2s \
  --output model_v2s_full_melmix.onnx
```

Root cause:
- `src.model` imports `timm`.
- This server's `timm` import path triggers `torch.compiler.disable`, which imports `torch._dynamo`, then `triton`.
- The installed Triton has a MetaX backend that expects a MACA install path; `maca_home_dirs()` returned `None`, causing `TypeError`.
- Export is CPU-only and does not need Dynamo/Triton, so this is an environment/import side effect, not a model/checkpoint problem.

Code changes:
- `src/model.py` now patches `torch.compiler.disable` to an identity decorator before importing `timm`, preventing the unnecessary Dynamo/Triton import path.
- `src/export_onnx.py` now accepts both bare checkpoint names and explicit checkpoint paths.

Retry after syncing code:

```bash
python -m src.export_onnx \
  --checkpoint last_v4_v2s_full_melmix.pt \
  --backbone v2s \
  --output model_v2s_full_melmix.onnx
```

## ONNX Export Success — 2026-05-03

Export succeeded:

```bash
python -m src.export_onnx \
  --checkpoint last_v4_v2s_full_melmix.pt \
  --backbone v2s \
  --output model_v2s_full_melmix.onnx
```

Result:
- Loaded checkpoint: `checkpoints/last_v4_v2s_full_melmix.pt`.
- Exported ONNX: `checkpoints/model_v2s_full_melmix.onnx`.
- Output shape: `(1, 234)`.
- File size: 81.8 MB.
- PyTorch vs ONNX max diff: `0.000003`.
- Status: OK.

Decision:
- This ONNX is valid for a single-model leaderboard smoke submission.
- Root `inference.py` discovers all non-specialist `.onnx` files in the model dataset directory, so this filename is acceptable.
- For a clean single-model smoke test, upload only `model_v2s_full_melmix.onnx` in the attached model dataset.

## Leaderboard Smoke Score 0.500 — 2026-05-03

Observation:
- Single-model leaderboard score returned exactly 0.500.

Root-cause diagnosis:
- A score this close to exactly 0.500 is much more consistent with a constant/dummy submission than with a weak trained model.
- The active `inference.py` had a silent fallback: if no ONNX model was loaded, it wrote a full-zero `submission.csv`.
- The Kaggle model paths were also brittle. Model datasets are attached under `/kaggle/input/<dataset-slug>/`, not inside the competition data directory.

Code changes:
- `inference.py` now discovers ONNX models recursively under `/kaggle/input`.
- It prints the resolved test/taxonomy/sample/model paths.
- If no main ONNX model loads, it raises `RuntimeError` instead of writing a dummy submission.
- It checks prediction class count, finite values, and prediction variance before writing `submission.csv`.

Decision:
- Rerun the notebook with the fixed `inference.py`.
- In the notebook log, verify that it prints a non-empty `Found main ONNX models: [...]` containing `model_v2s_full_melmix.onnx`.
- Also verify non-constant prediction stats before submitting.

## Leaderboard Smoke Score 0.801 — 2026-05-04

Observation:
- Single-model leaderboard score: `0.801`.
- Model: `model_v2s_full_melmix.onnx`, exported from `last_v4_v2s_full_melmix.pt`.

Interpretation:
- The previous `0.500` result was an invalid/dummy submission path, not a trained-model quality signal.
- The inference pipeline now loads the ONNX model and produces useful rankings.
- `0.801` is the first valid public LB anchor for this project.

Decision:
- Stop spending time on inference plumbing unless logs show another loading or constant-prediction issue.
- The shortest likely improvement is model diversity: train full-data `nfnet` and `regnety` with the same precomputed mel mixup setup, export their `last_*` checkpoints, and let `inference.py` rank-average all attached ONNX models.
- Pseudo-labeling should come after at least two valid full-data teachers unless the ensemble improvement is unexpectedly small.

Next commands:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone nfnet --epochs 36 --batch_size 128 --lr 3e-4 \
  --precomputed --precomputed_mixup_prob 0.3 \
  --train_all_soundscapes --num_workers 4 --save_tag v4_nfnet_full_melmix

python -m src.export_onnx \
  --checkpoint last_v4_nfnet_full_melmix.pt \
  --backbone nfnet \
  --output model_nfnet_full_melmix.onnx

torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone regnety --epochs 36 --batch_size 128 --lr 3e-4 \
  --precomputed --precomputed_mixup_prob 0.3 \
  --train_all_soundscapes --num_workers 4 --save_tag v4_regnety_full_melmix

python -m src.export_onnx \
  --checkpoint last_v4_regnety_full_melmix.pt \
  --backbone regnety \
  --output model_regnety_full_melmix.onnx
```
