# Strategy after 0.944 public LB - 2026-05-05

## Execution policy

- Do not run local experiments.
- Preferred execution: Kaggle MCP submissions.
- Fallback execution: prepare code for the cloud server and wait for returned logs/results.

## Raw results

| Ref | Submission | Public score | Delta vs previous best 0.801 | Decision |
| --- | --- | ---: | ---: | --- |
| `52327110` | `konbu17/bird26-wliilamsam-0943-with-train-audio-head` v3 | 0.944 | +0.143 | current best |
| `52326054` | `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling` v97 | 0.942 | +0.141 | valid anchor |
| `52327102` | `mattiaangeli/birdclef-2026-0-943-better-blend` v7 | 0.938 | +0.137 | reject |
| `52327096` | `artemnazemtsev/birdclef-pseudo-labeling-cross-year-ensemble` v15 | 0.928 | +0.127 | reject |
| `52295051` | self-trained CNN/ONNX | 0.801 | baseline | diversity only, not mainline |

Kaggle MCP leaderboard snapshot:

- Our rank after `0.944`: about 92 / 2971 teams.
- Top 20 public scores are roughly 0.948 to 0.958.
- Gap to top 20 floor: about +0.004.
- Gap to current top: about +0.014.

## First-principles reading

The metric is macro ROC-AUC over class columns. Therefore the submission is primarily judged by per-class ordering, not probability calibration.

The current winning stack works because it solves three separate problems:

1. Perch gives a strong general bioacoustic representation.
2. ProtoSSM uses the 12-window temporal structure in each 60-second soundscape.
3. SED and train-audio head add complementary ordering signal.

The `0.944` result proves the train-audio head is not cosmetic. It improved the same public-family anchor by +0.002 over the direct sequence-modeling submission. The poor `0.938` and `0.928` reruns prove public notebook titles and page scores are insufficient; only our scored reruns count.

## What not to do next

- Do not restart backbone sweeps as the main path. A 0.801 CNN is too far below 0.944.
- Do not trust broad pseudo-label/cross-year notebooks without source-level diagnosis; the tested one collapsed to 0.928.
- Do not optimize on train-soundscape dry-runs as if they predict hidden LB.
- Do not spend many submissions on random public forks below 0.944 page score.

## Highest-leverage next experiments

### 1. Lock current best

Select `52327110` in Kaggle UI unless a later submission beats it. MCP currently reports no selected submissions.

Why: this protects the current best score while we explore.

### 2. Head-weight neighborhood sweep

Current blend:

- ProtoSSM 0.50
- SED 0.35
- train-audio head 0.15

Next candidates should only move one degree of freedom at a time:

| Experiment | ProtoSSM | SED | Head | Why |
| --- | ---: | ---: | ---: | --- |
| H20a | 0.50 | 0.30 | 0.20 | test whether head signal is underweighted |
| H20b | 0.45 | 0.35 | 0.20 | increase head without reducing SED |
| H10a | 0.55 | 0.35 | 0.10 | test whether 0.15 was slightly too high |
| S40H10 | 0.50 | 0.40 | 0.10 | test if SED should dominate head |

Decision rule:

- Keep only if public score beats 0.944.
- If score ties at 0.944, prefer the version with more diverse head/SED balance only as a possible second selected/private hedge.

### 3. Build our own stronger Perch head

The public head was trained on `train_audio` only and has 202/234 trained classes. A better head should use:

- train_audio focal embeddings;
- labeled train_soundscape window embeddings;
- class-balanced logistic/linear heads;
- per-class regularization/thresholds;
- optional taxon-specific handling for Aves, Amphibia, Insecta.

Why: this is the cleanest way to create a real team-owned signal while staying inside the confirmed 0.944 architecture.

Do this on the cloud server, not locally.

### 4. Low-weight CNN diversity only after head work

The CNN branch can be tested only at tiny rank weights:

- 2.5% or 5% CNN max.
- Remove weight from ProtoSSM first, not from SED/head, because SED/head are the confirmed complementary signals.

Why: standalone 0.801 is too weak to justify a large blend, but it may contain rare class ordering differences.

### 5. Public candidate triage

Current MCP search shows no obvious public kernel above the confirmed `0.944` stack. Public 0.943 candidates may still be useful if they contain materially different components:

- `udaysonawane/cnn-perch-multi-head-mamba-rank-blend-sed`: page score 0.943, has different architecture language.
- `udaysonawane/protossm-input-conditional`: page score 0.943, recent update.
- `ulyanovantonamaranta/birdclef-2026-better-blend-repro`: page score 0.943.

These are lower priority than head-weight and custom-head experiments because the direct public reruns already showed significant non-reproducibility.

## Submission budget priority

For the next Kaggle submission day:

1. H20a head-weight sweep.
2. H20b head-weight sweep.
3. H10a or S40H10, chosen based on the first two.
4. One public architecture-diversity candidate only if no head sweep improves.
5. One tiny-CNN blend only if we can create it without risking runtime or row alignment.

## Current decision

The main competition solution is now:

`Perch ONNX -> ProtoSSM sequence branch + public SED branch + train-audio Perch head -> rank blend`

Everything else must justify itself as an orthogonal rank signal on top of that stack.
