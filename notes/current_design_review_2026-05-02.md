# Current Design Review — 2026-05-02

## Problem From First Principles

BirdCLEF+ 2026 is a 234-class multi-label detection problem over 5-second windows from 1-minute Pantanal soundscapes. The scoring target is class-wise macro ROC-AUC, so each species column matters roughly equally once it has positives in hidden test. This makes the problem less about maximizing average segment accuracy and more about preserving ranking quality for rare, weak, non-bird, and soundscape-only species.

Primary sources checked:
- Kaggle competition overview/data page: https://www.kaggle.com/competitions/birdclef-2026/overview
- Local data description: `docs/data_description.md`

Local data facts:
- `taxonomy.csv`: 234 classes: Aves 162, Amphibia 35, Insecta 28, Mammalia 8, Reptilia 1.
- `train.csv`: 35,549 focal recordings, 206 species with focal data.
- `train_soundscapes_labels.csv`: 1,478 labeled segments from 66 soundscape files, 75 species.
- 28 taxonomy classes have no focal training audio and appear only through soundscape labels: 25 insect sonotypes plus `1491113`, `25073`, `517063`.
- Labeled soundscape positives are dominated by non-birds: Amphibia 4,174 label occurrences, Insecta 1,136, Aves 824, Mammalia 84, Reptilia 26.

Implication: focal-only validation is structurally misaligned. A high focal AUC can coexist with poor leaderboard because the submission is evaluated on in-domain 5-second soundscape ranking, not clean close-mic single-label clips.

## Current Pipeline

Core files:
- `src/model.py`: raw waveform -> 128-bin mel spectrogram -> timm CNN backbone -> attention pooling -> 234 logits.
- `src/train.py`: v4 unified training with focal + labeled soundscape entries, secondary labels, waveform mixup, background mix, SpecAugment, DDP, optional precomputed mel mode, focal BCE loss.
- `src/preprocess.py`: pre-extracts focal and soundscape mel arrays into `data/precomputed`.
- `src/pseudo_label.py`: ensemble teacher for iterative Noisy Student pseudo-label generation on train soundscapes, with power scaling.
- `src/train_specialist.py`: separate non-bird specialist over Amphibia/Insecta/Mammalia/Reptilia.
- `inference.py`: ONNX CPU inference, multi-model rank averaging, specialist blend for non-birds, temporal smoothing, Kaggle submission formatting.

Current strategic direction is sound:
- Use labeled soundscapes as the validation anchor.
- Oversample in-domain soundscape examples.
- Add pseudo-labeling because unlabeled soundscapes are the biggest in-domain data reservoir.
- Keep inference ONNX and batch-oriented because CPU 90 minutes is a hard constraint.
- Treat non-birds separately because macro AUC makes the 72 non-bird classes too important to leave to a bird-dominated model.

## Current Experiment State

Observed in `train.log`:
- Command shown: `python -m src.train --backbone v2s --epochs 40 --batch_size 48 --lr 2e-4 --num_workers 4`
- Actual printed loop says 30 epochs, suggesting this log may be from an older `train.py` state or stale argument handling.
- DataLoader printed `num_workers=0`, consistent with the earlier bottleneck diagnosis.
- Validation AUC on soundscape-style validation reached:
  - epoch 1: 0.8966
  - epoch 2: 0.9259
  - epoch 3: 0.9410
- Run was very slow on the remote MetaX environment due to GPU queue/context delays and likely per-sample OGG decode from AFS.

Current local workspace has only CSV metadata under `data/`; audio directories are not present locally. Full training/preprocessing cannot run in this workspace until `data/train_audio`, `data/train_soundscapes`, and test/sample files are available or paths are mounted.

## Highest-Risk Issues To Fix Before Long Runs

1. Specialist mapping mismatch:
   - `src/train_specialist.py` saves `checkpoints/specialist_mapping.pt` with `torch.save`.
   - `inference.py` expects `specialist_mapping.npy` and loads it with `np.load`.
   - Result: specialist ONNX can load but class-index mapping will not, so specialist predictions will be silently disabled or ignored.

2. DDP training loses balanced sampling:
   - In `src/train.py`, `world_size > 1` uses plain `DistributedSampler`.
   - Existing class/domain balancing in `build_balanced_sampler` is only used for single-process waveform training.
   - Result: multi-GPU runs likely under-sample rare and soundscape-only species, exactly where macro AUC is most sensitive.

3. Precomputed validation no longer preserves site split:
   - `src/train.py` precomputed mode splits `soundscape_labeled` by sorted index because file/site metadata is not retained in `.npy` paths.
   - Result: validation can become easier or arbitrary, and it no longer tests cross-site generalization.

4. Precomputed pseudo-label path is incomplete:
   - `src/pseudo_label.py` writes CSV pseudo-labels, but `src/train.py --precomputed --pseudo` expects `.npy` pairs under `data/precomputed/pseudo`.
   - Result: pseudo-labels are ignored in precomputed training unless an extra conversion step exists.

5. Soundscape-only species primary-label weighting is imperfect:
   - `build_soundscape_entries` uses the first semicolon label as `primary_label` for sampling.
   - In multi-label soundscape rows, rare co-occurring classes may not drive sampler weights.
   - Result: entries containing soundscape-only insects/amphibians can still be underrepresented relative to their metric value.

6. Rank averaging is only across the 12 windows of each file:
   - `inference.py` rank-normalizes per file because each batch is one soundscape.
   - For ROC-AUC, class ranking is global across all test rows; per-file rank normalization can erase between-file confidence differences.
   - This may help calibration robustness but should be validated against soundscape CV, not assumed.

## Immediate Decision

Do not launch more long training yet. First fix the experiment plumbing that changes what the result means:

1. Make specialist mapping save/load format consistent.
2. Preserve metadata in precomputed manifests so validation can split by site and pseudo-labels can be consumed.
3. Restore rare/domain-aware sampling in DDP or use a weighted distributed sampler.
4. Add a small local integrity check that confirms:
   - all 234 class columns align with `taxonomy.csv` sorted order,
   - specialist indices map to non-bird labels correctly,
   - precomputed train/val counts and evaluable validation species are printed before training,
   - inference can load all provided model artifacts.

Only after those are fixed should the next expensive run start. Otherwise the next leaderboard result will not isolate model quality from broken data plumbing.
