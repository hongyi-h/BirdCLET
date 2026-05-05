# Experiment - Perch/SED/Head Direct Smoke Probe - 2026-05-04

## Purpose

Start the high-score branch work with a controlled script that runs the public artifacts now available locally:

- Perch v2 ONNX direct target logits with genus proxies.
- distilled SED 5-fold ONNX.
- train-audio Perch logistic head.
- per-class rank blend.

This is not the final ProtoSSM reference reproduction. It is the first executable experiment to validate paths, ONNX IO, row ids, class ordering, runtime, and branch CSV compatibility.

## Code

- `src/perch_sed_head_experiment.py`

Default outputs:

- `outputs/perch_sed_head_direct/submission_perch_direct.csv`
- `outputs/perch_sed_head_direct/submission_sed.csv`
- `outputs/perch_sed_head_direct/submission_head.csv`
- `outputs/perch_sed_head_direct/submission.csv`

## Commands

Cloud server probe on one train soundscape:

```bash
python -m src.perch_sed_head_experiment \
  --fallback_train_count 1 \
  --output_dir outputs/perch_sed_head_direct_dryrun1
```

Cloud server probe on 20 train soundscapes:

```bash
python -m src.perch_sed_head_experiment \
  --fallback_train_count 20 \
  --output_dir outputs/perch_sed_head_direct_dryrun20
```

Kaggle hidden test run uses automatic path discovery under `/kaggle/input`; local runs use `data/BirdCLEF+ 2026/test_soundscapes` and fall back to train soundscapes when local test audio is absent.

Policy update: after 2026-05-04 user instruction, do not run these commands locally. They are cloud/Kaggle-only experiment commands unless Kaggle MCP becomes available.

## Expected Role

This branch should be useful for smoke testing and as a fallback blend component. It should not be expected to match the 0.94+ notebooks until the ProtoSSM sequence branch is ported.

## 2026-05-04 Dry-Run Note

First local run in `/opt/anaconda3/envs/birdclef/bin/python` reached Perch direct successfully, then failed in SED mel extraction because `librosa -> numba` could not create/load its cache from site-packages. Fix applied in `src/perch_sed_head_experiment.py`: set `NUMBA_CACHE_DIR=/tmp/birdclef_numba_cache` before importing `librosa`.

After the fix:

- `--fallback_train_count 1` passed.
  - Perch direct mapped 203/234 classes.
  - Genus proxies covered 3 more classes.
  - Head has 202/234 trained classes.
  - Output: 12 rows x 235 columns.
- `--fallback_train_count 5` passed.
  - Output: 60 rows x 235 columns.
  - Perch time: 7.6s for 5 files.
  - SED time: 9.1s for 5 files after session/model setup.
  - Blend std: 0.217682.
- Script was updated after the dry-runs to auto-discover Kaggle competition paths and attached Perch/SED/head files under `/kaggle/input`. The 1-file and 5-file local dry-runs still pass after that change.
