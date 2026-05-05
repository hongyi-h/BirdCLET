# Kaggle MCP experiment log - 2026-05-04

## Execution constraint

- User constraint: do not run local experiments.
- Allowed execution paths:
  1. Kaggle MCP experiments/submissions directly on Kaggle.
  2. Prepare code/commands for the user to run on the cloud server.
- Current priority: use Kaggle MCP first.

## Kaggle state verified through MCP

- Competition: `birdclef-2026`
- Deadline: 2026-06-03 23:59:00 UTC
- Metric: BirdCLEF ROC AUC
- Current account has entered the competition.
- Public search showed stronger public baselines than our current local CNN path:
  - `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling`: public score 0.944
  - `artemnazemtsev/birdclef-pseudo-labeling-cross-year-ensemble`: public score 0.944
  - `konbu17/bird26-wliilamsam-0943-with-train-audio-head`: public score 0.943
  - `mattiaangeli/birdclef-2026-0-943-better-blend`: public score 0.943

## External data/model sources verified through MCP

- `rishikeshjani/perch-onnx-for-birdclef-2026`
  - `perch_v2.onnx`
  - `labels.csv`
  - ONNX Runtime wheel
- `tuckerarrants/bc2026-distilled-sed-public`
  - `sed_fold0.onnx` ... `sed_fold4.onnx`
- `konbu17/bird26-train-audio-head-v1`
  - `head_weights_train_audio.npz`
- `jaejohn/perch-meta`
- Model source used by public kernels:
  - `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`

## First-principles decision

The leaderboard objective rewards final ranking, not ownership of every model component. Our current trained CNN reached a much lower public score than the public Perch + sequence SED family. Therefore the shortest evidence-generating path is:

1. Reproduce or submit the strongest public Kaggle baseline as an anchor.
2. Only then test additions that can plausibly add orthogonal signal, such as the train-audio head or a very low-weight rank blend with our own CNN.
3. Avoid spending GPU on backbone sweeps until the public high-score stack is reproduced and measured under our account.

## Next Kaggle-side experiment

Candidate A:
- Submit or rerun `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling` version 97 if Kaggle MCP allows direct kernel submission.
- Reason: public score 0.944, currently the strongest verified public baseline.

Candidate B:
- If direct submission is unavailable, run the local reference notebook `reference/bird26-wliilamsam-0943-with-train-audio-head.ipynb` on Kaggle through MCP.
- Reason: source is locally available and uses verified public datasets/models.

## Submission attempts

### A1 - direct public kernel submission

- Time: 2026-05-04
- Competition: `birdclef-2026`
- Kernel: `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling`
- Kernel version: 97
- File: `submission.csv`
- Kaggle submission ref: `52326054`
- Initial MCP result:
  - accepted submission ref
  - submission list shows no `public_score` yet
  - submission list shows no `status` yet
  - `total_bytes=0`
- Final observed result:
  - `status=COMPLETE`
  - `total_bytes=17,824,527`
  - `public_score=0.942`
- Interpretation:
  - This is a valid leaderboard anchor and replaces our previous 0.801 as the working baseline.
  - It is lower than the public kernel page's reported 0.944. The likely cause is nondeterministic submit-time training/inference in the public notebook or a difference between the displayed best version and the rerun submitted under our team.

## Current leaderboard state after A1

- Best submitted score under our team: 0.942
- Previous best self-run score: 0.801
- Delta: +0.141 public AUC
- Kaggle MCP competition search reports team rank: 300 / 2958 teams
- Decision: continue with direct Kaggle submissions of nearby public high-score variants before spending cloud GPU on our own CNN/backbone work.

### A4 - follow-up public high-score candidate submissions

Submitted after A1 completed:

| Ref | Kernel | Version | Rationale | Initial status |
| --- | --- | ---: | --- | --- |
| `52327096` | `artemnazemtsev/birdclef-pseudo-labeling-cross-year-ensemble` | 15 | public 0.944 candidate with cross-year/pseudo-labeling additions | pending, `total_bytes=0` |
| `52327102` | `mattiaangeli/birdclef-2026-0-943-better-blend` | 7 | different post-processing/blend variant around the 0.943 family | pending, `total_bytes=0` |
| `52327110` | `konbu17/bird26-wliilamsam-0943-with-train-audio-head` | 3 | adds train-audio Perch linear head as orthogonal signal | pending, `total_bytes=0` |

Do not compare these candidates until Kaggle returns `status=COMPLETE` and `public_score`.

Final observed results:

| Ref | Kernel | Version | Public score | Status | Interpretation |
| --- | --- | ---: | ---: | --- | --- |
| `52327110` | `konbu17/bird26-wliilamsam-0943-with-train-audio-head` | 3 | 0.944 | COMPLETE | Best current submission. The train-audio head adds useful signal on this public split. |
| `52327102` | `mattiaangeli/birdclef-2026-0-943-better-blend` | 7 | 0.938 | COMPLETE | Worse than A1. Do not use as selected submission. |
| `52327096` | `artemnazemtsev/birdclef-pseudo-labeling-cross-year-ensemble` | 15 | 0.928 | COMPLETE | Much worse than public page score on rerun. Treat as non-reproducible under our submission. |

Updated leaderboard state:

- Best score: 0.944
- Best ref: `52327110`
- Improvement vs A1 0.942 anchor: +0.002
- Improvement vs self-trained CNN 0.801: +0.143
- Kaggle MCP competition search reports team rank: 92 / 2971 teams

Decision:

- Select `52327110` unless a later run beats it.
- The only public follow-up worth pursuing from this cluster is not generic pseudo-label/cross-year rewrites; it is targeted orthogonal signal around the confirmed high-score stack, especially train-audio-head weight tuning, selected-output blending, or another reproducible 0.944+ public version.

Selection check:

- MCP `group=Selected` returned no selected submissions.
- Action needed before deadline: ensure `52327110` is selected in Kaggle UI if no stronger submission supersedes it.

Final observed results:

| Ref | Kernel | Public score | Delta vs A1 `0.942` | Interpretation |
| --- | --- | ---: | ---: | --- |
| `52327110` | `konbu17/bird26-wliilamsam-0943-with-train-audio-head` v3 | 0.944 | +0.002 | best current submission; train-audio head adds useful orthogonal signal |
| `52327102` | `mattiaangeli/birdclef-2026-0-943-better-blend` v7 | 0.938 | -0.004 | worse on our rerun; do not use as current baseline |
| `52327096` | `artemnazemtsev/birdclef-pseudo-labeling-cross-year-ensemble` v15 | 0.928 | -0.014 | public-page score did not reproduce; do not trust this path without source-level diagnosis |

Updated leaderboard state:

- Best submitted score under our team: 0.944
- Current best ref: `52327110`
- Rank after score update: 92 / 2971 teams
- Selected submissions check: Kaggle MCP returned no selected submissions.
- Decision:
  - Treat the 3-way ProtoSSM + SED + train-audio-head rank blend as the working baseline.
  - The train-audio head result is actionable because it improved the same family by +0.002 over A1.
  - The low-scoring pseudo/cross-year result means source lineage/page score alone is not sufficient; every public variant must be rerun and scored under our team before adoption.

### A2 - public kernel session attempt

- Target: `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling`
- MCP tool: `create_notebook_session`
- Result: failed with `Permission 'kernelSessions.create' was denied`
- Interpretation: Kaggle MCP cannot currently create an interactive rerun session for this public kernel in this environment.

### A3 - Kaggle save permission probe

- MCP tool: `save_notebook`
- Result:
  - Passing an explicit slug failed with `Invalid slug`
  - Omitting slug succeeded and Kaggle generated `/code/benfromhrbust/codex-mcp-bird-smoke`
- Interpretation:
  - `save_notebook` can create private kernels from source text.
  - The MCP interface does not accept a local file path as source; large reference notebooks must either be pasted as text or uploaded by another route.

### B1 - cloud/Kaggle CLI-ready high-score anchor

- Created: `kaggle/protossm_sed_public/`
- Source: mechanical script export of `reference/birdclef-2026-0-943-better-blend.ipynb`
- Code file: `kaggle/protossm_sed_public/kernel.py`
- Metadata: `kaggle/protossm_sed_public/kernel-metadata.json`
- Intended execution:
  - `kaggle kernels push -p kaggle/protossm_sed_public`
- This is not a local experiment. It is a prepared Kaggle-side run package for the user/cloud environment because MCP cannot ingest the local notebook file path directly.
