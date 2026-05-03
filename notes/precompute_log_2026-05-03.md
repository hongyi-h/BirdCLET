# Precompute Log — 2026-05-03

Source logs:
- `logs/src.preprocess-focal.log`
- `logs/src.preprocess-soundscape.log`
- `logs/src.preprocess-soundscape_unlabeled.log`
- `logs/src.check_artifacts.log`

## Result

Precompute completed successfully.

Artifacts verified by `python -m src.check_artifacts --check_precomputed`:
- Taxonomy: 234 classes.
- Focal metadata: 35,549 rows, 206 focal species.
- Labeled soundscapes: 1,478 rows, 66 files, 75 species.
- Taxonomy classes without focal audio: 28.
- `data/precomputed/focal`: 106,647 rows.
- `data/precomputed/soundscape_labeled`: 1,478 rows.
- `data/precomputed/pseudo`: missing, expected because pseudo-labeling has not run yet.
- Final status: `OK`.

Additional log-derived counts:
- `data/precomputed/soundscape_unlabeled`: 127,157 segments from 10,658 soundscape files.

Approximate elapsed time from logs:
- Focal precompute: 2h 26m 19s.
- Labeled soundscape precompute: 1m 43s.
- Unlabeled soundscape precompute: 1h 44m 29s.

## Decision

The precomputed data is sufficient to start Phase 1 training with `--precomputed`.

Do not wait for pseudo labels before the first backbone run. From first principles, the next uncertainty is whether the training/evaluation loop produces a credible soundscape CV checkpoint and exportable ONNX model. Pseudo-labeling depends on a trained teacher, so it comes after the initial model ensemble or at least after a strong first teacher.

## Follow-up

`src.check_artifacts` was updated locally after these logs to also validate `soundscape_unlabeled` manifest/mel files. After syncing the code to the server, rerun:

```bash
python -m src.check_artifacts --check_precomputed
```

Then start the first training run:

```bash
torchrun --standalone --nproc_per_node=8 -m src.train \
  --backbone v2s --epochs 40 --batch_size 128 --lr 3e-4 \
  --precomputed --num_workers 4 --save_tag v4_v2s
```

## Training Launch Failure

`logs/train.log` shows the first DDP launch failed before data/model loading:

- Command effectively requested `world_size=8`.
- MetaX/PyTorch reported device ids 1-7 do not exist.
- ranks 1-7 failed at `torch.cuda.set_device(local_rank)`.
- Root error: `torch.AcceleratorError: CUDA error: invalid device ordinal`.

Decision:
- This is a GPU visibility/resource mismatch, not a BirdCLEF data or model bug.
- Relaunch with `--nproc_per_node` equal to the visible device count.
- If the job is supposed to have 8 GPUs, first make all 8 visible to the process environment, for example via the cluster scheduler or `MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

Before relaunching on the server:

```bash
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("MTHREADS_VISIBLE_DEVICES=", os.environ.get("MTHREADS_VISIBLE_DEVICES"))
print("torch.cuda.device_count()=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```
