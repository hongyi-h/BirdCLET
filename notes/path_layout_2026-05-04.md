# Path Layout Update - 2026-05-04

The local dataset layout changed:

- Competition data lives in `data/BirdCLEF+ 2026/`.
- External public datasets live under `data/`.
- Pretrained SavedModels live under `pretrained_models/`.
- Generated artifacts remain under `data/precomputed/` and `data/pseudo_labels*.csv`.

Code changes:

- `src/config.py` now makes `CFG.DATA_DIR` point to `data/BirdCLEF+ 2026`.
- `CFG.PRECOMPUTED_DIR` remains `data/precomputed`.
- `CFG.PSEUDO_LABEL_PATH` is `data/pseudo_labels.csv`.
- Local `inference.py` now reads taxonomy/sample/test paths from `data/BirdCLEF+ 2026`.
- `src.check_artifacts --check_external` verifies the local Perch, SED, Perch metadata, and pretrained model inputs.

Verification:

- `python -m src.check_artifacts --check_external` passes.
- `python -m src.check_artifacts --check_precomputed --check_external` passes after regenerating `data/precomputed/soundscape_unlabeled/manifest.csv`.
- Current precomputed counts:
  - focal: 106,647 rows.
  - soundscape_labeled: 1,478 rows.
  - soundscape_unlabeled: 127,157 rows.
  - pseudo: missing, expected until pseudo-label conversion is run.

Environment overrides are available:

- `BIRDCLEF_DATA_ROOT`
- `BIRDCLEF_COMPETITION_DATA_DIR`
- `BIRDCLEF_PRECOMPUTED_DIR`
- `BIRDCLEF_PSEUDO_LABEL_DIR`
- `BIRDCLEF_PRETRAINED_MODEL_DIR`
