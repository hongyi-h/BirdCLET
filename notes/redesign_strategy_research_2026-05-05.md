# Redesigned competition strategy - 2026-05-05

## Constraint

- Goal: highest possible Kaggle leaderboard rank.
- Do not run local experiments.
- Preferred validation: Kaggle MCP submissions.
- Fallback validation: prepare cloud-server jobs and wait for returned logs/results.

## Current scoreboard evidence

| Ref | Submission | Public score | Decision |
| --- | --- | ---: | --- |
| `52327110` | `konbu17/bird26-wliilamsam-0943-with-train-audio-head` v3 | 0.944 | current best |
| `52326054` | `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling` v97 | 0.942 | valid anchor |
| `52327102` | `mattiaangeli/birdclef-2026-0-943-better-blend` v7 | 0.938 | reject |
| `52327096` | `artemnazemtsev/birdclef-pseudo-labeling-cross-year-ensemble` v15 | 0.928 | reject |
| `52295051` | our CNN/ONNX | 0.801 | not a main branch |

Kaggle MCP leaderboard snapshot:

- Our current rank: about 92 / 2971.
- Top 20 public scores: about 0.948-0.958.
- Immediate target: +0.004 to reach the current top-20 floor.

## Research and community evidence

| Source | Relevant finding | Impact on strategy |
| --- | --- | --- |
| Perch 2.0 paper, Google Research, 2025 | Perch 2.0 is a supervised multi-taxa bioacoustic foundation model with strong embeddings and off-the-shelf species scores; it reports SOTA on BirdSet and BEANS. | Keep Perch as the observation backbone. Do not replace it with an ImageNet CNN. |
| Foundation model review for bioacoustics, 2026 | Perch 2.0 is reported as strongest on BirdSet and strong for linear probing on BEANS; attentive probing can unlock transformer audio models. | Our best owned contribution should be better probing/heads over foundation embeddings, not raw spectrogram backbone sweeps. |
| BEATs, ICML 2023 | General audio SSL with acoustic tokenizers is strong on AudioSet/ESC-50. | BEATs/AudioSet-style models may be useful only as a separate diversity branch, if CPU-runtime and conversion constraints are solved. |
| BirdNET, Ecological Informatics 2021 | Domain-specific augmentation and temporal resolution matter for noisy overlapping bird sound recognition. | If training our own branch, use longer context and domain augmentations; avoid generic 5s-only image classification. |
| Kaggle discussion `683791` | Strong single models around 0.922-0.932 are often trained on longer 15-20s contexts, train_audio + labeled soundscapes, and ensembles; runtime matters. | A useful owned CNN/SED branch must be longer-context and inference-budget aware. |
| Kaggle discussion `696857` | Top teams emphasize heterogeneous ensembles; pseudo-labeling can help but often hurts unless architecture and filtering match sparse noisy labels. | Pseudo labels are a branch-specific tool, not a global next step. |

Sources:

- https://research.google/pubs/perch-20-the-bittern-lesson-for-bioacoustics/
- https://www.sciencedirect.com/science/article/pii/S1574954126001718
- https://proceedings.mlr.press/v202/chen23ag.html
- https://www.sciencedirect.com/science/article/pii/S1574954121000273
- https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- https://www.kaggle.com/competitions/birdclef-2026/discussion/696857

## First-principles redesign

The hidden test object is a 60-second soundscape split into twelve 5-second scoring rows. The metric is macro ROC-AUC per class. Therefore:

1. The model must rank each class well across all windows.
2. Probability calibration is secondary to class-wise ordering.
3. Temporal context is signal, not decoration.
4. Rare classes and non-bird taxa matter as much as frequent birds.
5. Heterogeneous rank signals are the shortest path after a strong foundation model.

The current 0.944 stack is:

`Perch ONNX logits/embeddings -> ProtoSSM sequence branch + SED branch + train-audio Perch head -> per-class rank blend`

This becomes the base system. New work must add orthogonal ranking signal to this base.

## New architecture target

### Branch A - locked public foundation stack

- Keep `konbu17` 3-way stack as the baseline:
  - ProtoSSM 0.50
  - SED 0.35
  - train-audio head 0.15
- It is the selected-submission candidate until beaten.

### Branch B - our custom Perch head v2

Train a new head over Perch embeddings, not mel images.

Inputs:

- `train_audio` focal embeddings, mean/attention pooled.
- labeled train soundscape window embeddings.
- optional site/hour/taxon metadata.

Targets:

- focal primary labels;
- soundscape multi-label windows;
- class-balanced targets with taxon-specific regularization.

Model candidates:

1. per-class logistic regression with calibrated class weights;
2. low-rank multi-head linear model sharing taxon/family structure;
3. shallow attentive pooling over 12 windows for soundscape rows;
4. optional isotonic/rank calibration per class.

Why this is first:

- The public train-audio head already improved +0.002.
- It only covers 202/234 classes; there is room to improve labels and coverage.
- It is cheap to train once Perch embeddings are cached.
- It adds owned signal without fighting the foundation backbone.

### Branch C - longer-context owned SED/CNN

Train a 15-20s context model only as a diversity branch.

Design:

- Train on train_audio + labeled soundscapes.
- Predict 5s rows using neighboring context, not isolated 5s crops.
- Use KD from the 0.944 stack, but keep a non-KD variant for diversity.
- Target standalone LB around 0.92+ before any meaningful blend weight.

Why this is second:

- Discussions suggest 15-20s context single models can reach 0.922-0.932.
- Our current 5s CNN is 0.801, so it is not worth large blend weight.

### Branch D - pseudo labels only for selected branches

Use pseudo labels only after Branch B/C have a clean validation story.

Rules:

- Teacher must be the 0.944 stack, not our 0.801 CNN.
- Use high precision thresholds and class/taxon caps.
- Train only one branch at a time with pseudo labels.
- Keep a no-pseudo sibling for ensemble diversity.

Why:

- Public discussion shows pseudo labels can improve 0.01-0.02, but also often hurt.
- Our tested pseudo/cross-year public notebook collapsed to 0.928.

## Ranked ideas

| Rank | Idea | Expected public gain | Risk | Cost | First validation |
| ---: | --- | ---: | --- | --- | --- |
| 1 | Head-weight sweep around confirmed `0.944` stack | +0.001 to +0.003 | Low | 3-4 Kaggle submissions | modify only final blend weights |
| 2 | Custom Perch head v2 using train_audio + labeled soundscapes | +0.002 to +0.006 | Medium | cloud training + 1-2 submissions | replace public head only |
| 3 | 15-20s context SED/CNN diversity branch | +0.002 to +0.008 if standalone >=0.92 | Medium/High | GPU training | low-weight rank blend |
| 4 | Public architecture-diversity candidates | +0.000 to +0.003 | Medium | Kaggle submissions | submit only candidates with materially different components |
| 5 | Pseudo-label branch | +0.000 to +0.010 | High | GPU training + filtering | one branch, high precision only |
| 6 | BEATs/BirdNET/BioCLIP external branch | unknown | High | model conversion/runtime | only if a public CPU-compatible checkpoint exists |

## Immediate experiment plan

### Kaggle submission day

1. H20a: ProtoSSM 0.50 / SED 0.30 / Head 0.20.
2. H20b: ProtoSSM 0.45 / SED 0.35 / Head 0.20.
3. H10a: ProtoSSM 0.55 / SED 0.35 / Head 0.10.
4. Submit one public architecture-diversity candidate only if still below 0.945.
5. Stop if a variant exceeds 0.946; preserve submissions for more targeted follow-up.

### Cloud-server build

1. Generate Perch embeddings for train_audio and labeled soundscapes if not already available.
2. Train custom head v2:
   - logistic baseline;
   - low-rank shared head;
   - optional 12-window attentive pooling for soundscape supervision.
3. Export head weights to a Kaggle dataset or embed them in a kernel.
4. Submit as:
   - ProtoSSM 0.50 / SED 0.30-0.35 / public head 0.05-0.10 / custom head 0.10-0.15.

## What changes from the old plan

Old plan still treated our CNN as a plausible next branch. New plan demotes it.

The owned branch should start from Perch embeddings because:

- the confirmed leaderboard gain came from a Perch train-audio head;
- literature supports Perch 2.0 as the strongest bioacoustic representation;
- the current metric rewards rank improvements more than raw probability calibration;
- custom heads are faster to iterate than full spectrogram backbones.

## Current action items

- Select `52327110` in Kaggle UI.
- Prepare a Kaggle/CLI weight-sweep package for the `konbu17` notebook.
- Prepare cloud code for custom Perch head v2 training.
- Keep all local experiment execution disabled unless the user explicitly changes policy.
