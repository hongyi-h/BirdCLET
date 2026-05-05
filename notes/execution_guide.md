# Execution Guide — BirdCLEF 2026 Pipeline

## 2026-05-04 Priority Update

Confirmed anchor:
- `model_v2s_full_melmix.onnx`: public LB `0.801`.

Local path layout:
- Competition data: `data/BirdCLEF+ 2026/`
- External public datasets: `data/`
- Pretrained SavedModels: `pretrained_models/`
- Generated mels/pseudo labels: `data/precomputed/` and `data/pseudo_labels*.csv`

After reviewing the local reference notebooks in `reference/`, the highest-leverage path is no longer "train three more versions of our CNN first." The 0.94+ public notebooks are driven by Perch v2 ONNX embeddings/logits, 12-window sequence modeling, public distilled SED folds, and per-class rank blending.

Immediate priority:

1. Reproduce a Perch/ProtoSSM + public SED reference branch.
2. Prefer the train-audio-head reference branch first: `ProtoSSM + SED + train-audio head`.
3. Submit it alone as a high baseline.
4. Rank-blend our `model_v2s_full_melmix.onnx` at low weight, initially 5-10%, and keep it only if public LB improves or drops negligibly with a plausible private-diversity reason.
5. Resume nfnet/regnety full-data training only after the Perch+SED blend is working.

See `notes/reference_notebook_design_review_2026-05-04.md` for the first-principles rationale.
See `notes/strategy_rethink_2026-05-04.md` for the current submission sequence.

### Rank-blend branch CSVs

After a reference notebook produces branch CSVs, blend them with strict row/column checks:

```bash
python -m src.blend_submissions \
    --inputs submission_protossm.csv submission_sed.csv \
    --weights 0.60 0.40 \
    --output submission.csv \
    --sample 'data/BirdCLEF+ 2026/sample_submission.csv'
```

Preferred train-audio-head baseline:

```bash
python -m src.blend_submissions \
    --inputs submission_protossm.csv submission_sed.csv submission_head.csv \
    --weights 0.50 0.35 0.15 \
    --output submission.csv \
    --sample 'data/BirdCLEF+ 2026/sample_submission.csv'
```

To test our CNN as a small diversity branch on top of the head baseline:

```bash
python -m src.blend_submissions \
    --inputs submission_protossm.csv submission_sed.csv submission_head.csv submission_cnn.csv \
    --weights 0.475 0.325 0.125 0.075 \
    --output submission.csv \
    --sample 'data/BirdCLEF+ 2026/sample_submission.csv'
```

Check that the local layout is complete:

```bash
python -m src.check_artifacts --check_external
```

Start the first controlled experiment on the cloud server, not locally:

```bash
python -m src.perch_sed_head_experiment \
    --fallback_train_count 20 \
    --output_dir outputs/perch_sed_head_direct_cloud_probe20
```

Current policy: do not run experiments locally. See `notes/experiment_execution_policy_2026-05-04.md`.

## Prerequisites

```bash
# On GPU server (8x MetaX C500)
conda activate birdclef
pip install timm>=1.0 soundfile onnxruntime onnx scikit-learn
```

---

## Phase 0: Precompute Mel Spectrograms (Optional, Recommended)

Eliminates per-sample .ogg decoding during training. ~30min one-time cost.

```bash
python -m src.preprocess --mode focal --crops 3
python -m src.preprocess --mode soundscape
python -m src.preprocess --mode soundscape_unlabeled
```

---

## Phase 1: Train 3 Diverse Backbones

### Option A: 8-GPU DDP (recommended)

```bash
# Backbone 1: EfficientNetV2-S
torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone v2s --epochs 40 --batch_size 128 --lr 3e-4 \
    --precomputed --precomputed_mixup_prob 0.3 \
    --num_workers 4 --save_tag v4_v2s

# Backbone 2: eca_nfnet_l0
torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone nfnet --epochs 40 --batch_size 128 --lr 3e-4 \
    --precomputed --precomputed_mixup_prob 0.3 \
    --num_workers 4 --save_tag v4_nfnet

# Backbone 3: RegNetY-016
torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone regnety --epochs 40 --batch_size 128 --lr 3e-4 \
    --precomputed --precomputed_mixup_prob 0.3 \
    --num_workers 4 --save_tag v4_regnety
```

### Option B: Single GPU fallback

```bash
python -m src.train --backbone v2s --epochs 40 --batch_size 48 --lr 2e-4 \
    --num_workers 4 --save_tag v4_v2s
```

### Option C: With precomputed mels

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone v2s --epochs 40 --batch_size 256 --lr 3e-4 \
    --num_workers 4 --precomputed --precomputed_mixup_prob 0.3 --save_tag v4_v2s
```

### Checkpoints after Phase 1
- `checkpoints/best_v4_v2s.pt`
- `checkpoints/best_v4_nfnet.pt`
- `checkpoints/best_v4_regnety.pt`
- `checkpoints/last_*.pt` is also saved. For runs using `--train_all_soundscapes`, prefer `last_*` because `best_*` is selected by leaky validation.

### Full-data models after valid single-model LB

Current valid anchor:
- `model_v2s_full_melmix.onnx`: public LB `0.801`.

Next full-data diversity runs:

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

### Quick LB check: export best single model and submit
```bash
python -m src.export_onnx \
    --checkpoint last_v4_v2s_full_melmix.pt \
    --backbone v2s \
    --output model_v2s_full_melmix.onnx
```

---

## Phase 2: Iterative Noisy Student (4 Rounds)

### Round 1: Ensemble teacher → pseudo-label → retrain

```bash
# Pseudo-label with 3-model ensemble
python -m src.pseudo_label \
    --checkpoints best_v4_v2s.pt best_v4_nfnet.pt best_v4_regnety.pt \
    --backbones v2s nfnet regnety \
    --round 1 --threshold 0.7 --power_gamma 1.5 --precomputed --tta

# Convert current round pseudo labels into fast training mels
python -m src.preprocess --mode pseudo \
    --pseudo_path data/pseudo_labels_r1.csv --soft_threshold 0.1

# Retrain each backbone with pseudo-labels
torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone v2s --epochs 30 --batch_size 128 --lr 1e-4 \
    --precomputed --precomputed_mixup_prob 0.3 --pseudo --train_all_soundscapes \
    --num_workers 4 --resume best_v4_v2s.pt \
    --save_tag v4_v2s_r1

torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone nfnet --epochs 30 --batch_size 128 --lr 1e-4 \
    --precomputed --precomputed_mixup_prob 0.3 --pseudo --train_all_soundscapes \
    --num_workers 4 --resume best_v4_nfnet.pt \
    --save_tag v4_nfnet_r1

torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone regnety --epochs 30 --batch_size 128 --lr 1e-4 \
    --precomputed --precomputed_mixup_prob 0.3 --pseudo --train_all_soundscapes \
    --num_workers 4 --resume best_v4_regnety.pt \
    --save_tag v4_regnety_r1
```

### Round 2

```bash
python -m src.pseudo_label \
    --checkpoints best_v4_v2s_r1.pt best_v4_nfnet_r1.pt best_v4_regnety_r1.pt \
    --backbones v2s nfnet regnety \
    --round 2 --threshold 0.6 --power_gamma 1.5 --precomputed --tta

python -m src.preprocess --mode pseudo \
    --pseudo_path data/pseudo_labels_r2.csv --soft_threshold 0.1

torchrun --standalone --nproc_per_node=8 -m src.train \
    --backbone v2s --epochs 25 --batch_size 128 --lr 5e-5 \
    --precomputed --precomputed_mixup_prob 0.3 --pseudo --train_all_soundscapes \
    --num_workers 4 --resume best_v4_v2s_r1.pt \
    --save_tag v4_v2s_r2
# (repeat for nfnet, regnety)
```

### Round 3-4 (same pattern, lower thresholds)

```bash
# Round 3: pseudo_label --precomputed --threshold 0.5 --power_gamma 1.3;
#          preprocess --mode pseudo with data/pseudo_labels_r3.csv;
#          train --precomputed --pseudo --train_all_soundscapes --lr 3e-5 --epochs 20
# Round 4: same pattern with --threshold 0.4 --power_gamma 1.3 --lr 1e-5 --epochs 15
```

---

## Phase 3: Non-Bird Specialist

```bash
python -m src.train_specialist --backbone b0 --epochs 50 --batch_size 64 --lr 3e-4

# Export
python -m src.export_onnx --checkpoint best_specialist.pt --backbone b0 \
    --num_classes 72 --output specialist.onnx

# Save mapping for inference
# (automatically saved by train_specialist.py as specialist_mapping.pt)
# Convert to numpy for Kaggle:
python -c "
import torch, numpy as np
m = torch.load('checkpoints/specialist_mapping.pt')
np.save('checkpoints/specialist_mapping.npy', m)
"
```

---

## Phase 5: Export All Models & Submit

```bash
# Export best models from final pseudo-label round
python -m src.export_onnx --checkpoint best_v4_v2s_r2.pt --backbone v2s --output model_v2s.onnx
python -m src.export_onnx --checkpoint best_v4_nfnet_r2.pt --backbone nfnet --output model_nfnet.onnx
python -m src.export_onnx --checkpoint best_v4_regnety_r2.pt --backbone regnety --output model_regnety.onnx
python -m src.export_onnx --checkpoint best_specialist.pt --backbone b0 --num_classes 72 --output specialist.onnx
```

### Upload to Kaggle

1. Create Kaggle Dataset `birdclef2026-models` containing:
   - `model_v2s_full_melmix.onnx` for the single-model smoke test.
   - Later ensemble runs can add `model_nfnet.onnx` and `model_regnety.onnx`.

2. Create Kaggle Dataset `birdclef2026-specialist` containing:
   - `specialist.onnx`
   - `specialist_mapping.npy`

3. Use `inference.py` as the Kaggle notebook code.

The notebook log must show a non-empty `Found main ONNX models: [...]`. A run that writes a dummy or constant submission is not a valid leaderboard signal.

---

## Troubleshooting

### DDP fails with "mccl backend not available"
```bash
# Check available backends
python -c "import torch.distributed as dist; print([b for b in ['mccl','nccl','gloo'] if dist.is_backend_available(b)])"

# If only gloo: edit train.py line ~100, or just use single-GPU
python -m src.train --backbone v2s --epochs 40 --batch_size 48 --lr 2e-4 --num_workers 4
```

### num_workers still 0
```bash
# Verify spawn works
python -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader, TensorDataset
import torch
ds = TensorDataset(torch.randn(100, 10))
dl = DataLoader(ds, batch_size=10, num_workers=2, multiprocessing_context=mp.get_context('spawn'))
for x in dl: break
print('OK: num_workers=2 works')
"
```

### MUSA queue block timeout
- Normal on first init (up to 5min). If persistent, try:
  ```bash
  MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun ...
  ```

### Inference too slow on Kaggle CPU
- Profile: check mel precomputation vs model inference time
- Reduce ensemble: drop weakest model
- Try OpenVINO: `pip install openvino` and convert ONNX → IR
