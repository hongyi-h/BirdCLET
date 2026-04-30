"""
BirdCLEF 2026 — Inference Notebook v2
Improvements over v1:
- Temporal smoothing across adjacent segments
- Support for EfficientNetV2-S backbone
- Optimized batch inference
- Optional multi-model ensemble

Upload this as a Kaggle notebook. Attach your trained ONNX model(s) as a Kaggle Dataset.
"""

import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
import librosa

# ============================================================
# Config
# ============================================================
SR = 32000
DURATION = 5
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 20
FMAX = 16000
NUM_CLASSES = 234

# Temporal smoothing: weighted average with neighbors
TEMPORAL_SMOOTH = True
SMOOTH_WEIGHTS = np.array([0.15, 0.7, 0.15])  # [prev, current, next]

# ============================================================
# Paths
# ============================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    TEST_DIR = "/kaggle/input/birdclef-2026/test_soundscapes"
    TAXONOMY_PATH = "/kaggle/input/birdclef-2026/taxonomy.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/birdclef-2026/sample_submission.csv"
    # Support multiple models for ensemble
    MODEL_PATHS = []
    model_dirs = [
        "/kaggle/input/birdclef2026-model",
        "/kaggle/input/birdclef2026-model-v2",
    ]
    for d in model_dirs:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".onnx"):
                    MODEL_PATHS.append(os.path.join(d, f))
    if not MODEL_PATHS:
        MODEL_PATHS = ["/kaggle/input/birdclef2026-model/model.onnx"]
else:
    TEST_DIR = "data/test_soundscapes"
    TAXONOMY_PATH = "data/taxonomy.csv"
    SAMPLE_SUB_PATH = "data/sample_submission.csv"
    MODEL_PATHS = ["checkpoints/model.onnx"]


# ============================================================
# Mel spectrogram
# ============================================================
def compute_mel(audio, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    return mel_db


# ============================================================
# Temporal smoothing
# ============================================================
def temporal_smooth(probs, weights=SMOOTH_WEIGHTS):
    """Apply weighted smoothing across time axis. probs: (n_segments, n_classes)"""
    if not TEMPORAL_SMOOTH or len(probs) < 2:
        return probs

    smoothed = np.zeros_like(probs)
    n = len(probs)

    for i in range(n):
        if i == 0:
            # First segment: use [current, next]
            w = np.array([weights[1] + weights[0], weights[2]])
            w /= w.sum()
            smoothed[i] = w[0] * probs[i] + w[1] * probs[i + 1]
        elif i == n - 1:
            # Last segment: use [prev, current]
            w = np.array([weights[0], weights[1] + weights[2]])
            w /= w.sum()
            smoothed[i] = w[0] * probs[i - 1] + w[1] * probs[i]
        else:
            smoothed[i] = (weights[0] * probs[i - 1] +
                           weights[1] * probs[i] +
                           weights[2] * probs[i + 1])

    return smoothed


# ============================================================
# Main inference
# ============================================================
def main():
    tax = pd.read_csv(TAXONOMY_PATH)
    species_cols = sorted(tax["primary_label"].astype(str).tolist())

    # Load ONNX models
    sessions = []
    for mpath in MODEL_PATHS:
        if not os.path.exists(mpath):
            continue
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count()
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(mpath, sess_options=opts)
        sessions.append(sess)
        print(f"Loaded model: {mpath}")

    if not sessions:
        print("No models found. Generating dummy submission.")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub[species_cols] = 0.0
        sub.to_csv("submission.csv", index=False)
        return

    input_name = sessions[0].get_inputs()[0].name
    print(f"Using {len(sessions)} model(s) for ensemble")

    # Find test files
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.ogg")))
    if not test_files:
        print("No test files found. Generating dummy submission.")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub[species_cols] = 0.0
        sub.to_csv("submission.csv", index=False)
        return

    print(f"Processing {len(test_files)} test soundscapes...")

    rows = []
    num_samples_per_seg = SR * DURATION

    for fpath in test_files:
        fname = os.path.basename(fpath)
        fname_no_ext = fname.replace(".ogg", "")

        audio, file_sr = sf.read(fpath, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)

        # Pad to 60 seconds
        total_samples = SR * 60
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)))

        # 12 x 5-second segments
        mels = []
        for i in range(12):
            start = i * num_samples_per_seg
            seg = audio[start:start + num_samples_per_seg]
            mel = compute_mel(seg)
            mels.append(mel)

        batch = np.stack(mels, axis=0)[:, np.newaxis, :, :].astype(np.float32)

        # Ensemble: average logits across models
        all_logits = []
        for sess in sessions:
            logits = sess.run(None, {input_name: batch})[0]
            all_logits.append(logits)

        avg_logits = np.mean(all_logits, axis=0)  # (12, 234)
        probs = 1.0 / (1.0 + np.exp(-avg_logits))  # sigmoid

        # Temporal smoothing
        probs = temporal_smooth(probs)

        for j in range(12):
            end_t = (j + 1) * DURATION
            row_id = f"{fname_no_ext}_{end_t}"
            row = {"row_id": row_id}
            for k, sp in enumerate(species_cols):
                row[sp] = float(probs[j, k])
            rows.append(row)

    sub = pd.DataFrame(rows)

    if os.path.exists(SAMPLE_SUB_PATH):
        sample = pd.read_csv(SAMPLE_SUB_PATH, nrows=1)
        sub = sub[sample.columns]

    sub.to_csv("submission.csv", index=False)
    print(f"Submission saved: {sub.shape[0]} rows, {sub.shape[1]} columns")


if __name__ == "__main__":
    main()
