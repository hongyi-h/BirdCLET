# Kaggle Perch head v2 experiment - 2026-05-05

## Constraint

- No local experiments.
- Kaggle is preferred for execution.
- Submission budget is 5 per day, so this work used Kaggle notebook runs and offline proxy scoring before any competition submission.

## Training notebook

Kernel:

- `benfromhrbust/birdclef-2026-perch-head-v2-train-codex`
- Version 6 completed.
- Execution path: Kaggle CLI `kaggle kernels push -p kaggle/perch_head_v2_train`
- Output downloaded to `outputs/kaggle_perch_head_v2_train_v6/`.

Important debugging:

1. Kaggle MCP `save_notebook` did not mount `/kaggle/input` data sources correctly.
2. Kaggle CLI push with `kernel-metadata.json` did mount competition/dataset sources correctly.
3. `train_soundscapes_labels.csv` uses `HH:MM:SS` end times; row-id creation must parse timedeltas.

## Head v2 result

Head v2 was trained from labeled train soundscape Perch features:

- Rows: 739 labeled windows.
- Files: 66.
- Positive labels: 3,122.
- Features: Perch embedding 1536 + mapped Perch logits 234 = 1,770.
- Model: per-class balanced logistic regression, filename GroupKFold OOF.
- Full-data trained classes: 70.

OOF proxy score:

| Branch | Macro AUC | Amphibia | Aves | Insecta | Mammalia | Reptilia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Perch direct | 0.736659 | 0.805694 | 0.900460 | 0.500000 | 0.834931 | 0.500000 |
| Head v2 OOF | 0.885963 | 0.870060 | 0.938774 | 0.820546 | 0.989856 | 0.897436 |
| Direct 0.85 + OOF head 0.15 | 0.874089 | 0.856541 | 0.916280 | 0.820546 | 0.982134 | 0.897436 |

Interpretation:

- Head v2 is a real signal on labeled soundscapes, especially for taxa where direct Perch has weak/no mapping.
- Because it trains on only 70 classes and only 66 files, it should be added conservatively to the high-score stack.
- It is not a standalone replacement for the public train-audio head.

## Weight dataset

Created private Kaggle dataset:

- `benfromhrbust/birdclef-2026-perch-head-v2-weights-codex`
- File: `head_v2_weights.npz`
- Size: 442 KB.

## Submission notebook dry-run

Kernel:

- `benfromhrbust/birdclef-2026-0-944-head-v2-codex`
- Version 1 completed.
- Execution path: Kaggle CLI `kaggle kernels push -p kaggle/konbu_head_v2`
- Output downloaded to `outputs/kaggle_konbu_head_v2_v1/`.

Dry-run result:

- Final `submission.csv`: `(240, 235)`.
- No NaNs.
- Public stack ran successfully with competition data, Perch ONNX, SED folds, public train-audio head, and private head v2 dataset.
- Runtime log shows final blend:
  - ProtoSSM 50%
  - SED 32%
  - public train-audio head 10%
  - custom soundscape head v2 8%

Output sanity:

| File | Shape | NaNs | Min | Max | Std |
| --- | --- | ---: | ---: | ---: | ---: |
| `submission.csv` | `(240, 235)` | 0 | 0.025083 | 1.000000 | 0.218986 |
| `submission_protossm.csv` | `(240, 235)` | 0 | 0.000000 | 0.997427 | 0.163488 |
| `submission_sed.csv` | `(240, 235)` | 0 | 0.000022 | 0.974855 | 0.070373 |
| `submission_head.csv` | `(240, 235)` | 0 | 0.000001 | 0.999939 | 0.186720 |
| `submission_head_v2.csv` | `(240, 235)` | 0 | 0.000000 | 1.000000 | 0.209325 |

## Decision

This candidate is eligible for one Kaggle competition submission.

Reason:

- It passed Kaggle dry-run with correct inputs and output contract.
- Head v2 OOF proxy is materially stronger than Perch direct on labeled soundscapes.
- It is conservatively blended at 8%, so it tests additional signal without replacing the confirmed 0.944 stack.

Risk:

- OOF proxy uses train soundscape labels, not hidden public/private distribution.
- Head v2 has only 70 trained classes and may overfit site/time patterns.
- A small public LB drop is possible if hidden test differs from labeled train soundscapes.

Next action:

- Submit version 1 of `benfromhrbust/birdclef-2026-0-944-head-v2-codex` only if we are willing to spend one daily submission.
- If it beats 0.944, keep and explore weight variants.
- If it drops, reduce head v2 to 3-5% or restrict it to non-bird/soundscape-only classes.

## Competition submission

Submitted candidate:

- Ref: `52348805`
- Date: `2026-05-05T09:32:26.173Z`
- Kernel: `benfromhrbust/birdclef-2026-0-944-head-v2-codex`
- Script version: `316748375`
- Description: `Codex candidate: 0.944 public stack plus custom labeled-soundscape Perch head v2 at 8% rank weight`

Current status while recording:

- The submission appears in the Kaggle submissions list.
- `total_bytes` is still `0`.
- No `status` or `public_score` has been returned yet.
- Interpretation: Kaggle code rerun/scoring is still pending or queued.
