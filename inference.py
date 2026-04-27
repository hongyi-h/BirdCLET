"""
BirdCLEF 2026 — Inference Notebook
Upload this as a Kaggle notebook. Attach your trained ONNX model as a Kaggle Dataset.

Expected Kaggle input layout:
  /kaggle/input/birdclef-2026/test_soundscapes/   <- competition data
  /kaggle/input/birdclef-2026/taxonomy.csv
  /kaggle/input/birdclef-2026/sample_submission.csv
  /kaggle/input/birdclef2026-model/model.onnx     <- your uploaded model
"""

import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
import librosa

# ============================================================
# Config — must match training config exactly
# ============================================================
SR = 32000
DURATION = 5
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 20
FMAX = 16000
NUM_CLASSES = 234

# ============================================================
# Paths — adjust the model dataset name to match your upload
# ============================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    TEST_DIR = "/kaggle/input/birdclef-2026/test_soundscapes"
    TAXONOMY_PATH = "/kaggle/input/birdclef-2026/taxonomy.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/birdclef-2026/sample_submission.csv"
    MODEL_PATH = "/kaggle/input/birdclef2026-model/model.onnx"
else:
    # Local testing
    TEST_DIR = "data/test_soundscapes"
    TAXONOMY_PATH = "data/taxonomy.csv"
    SAMPLE_SUB_PATH = "data/sample_submission.csv"
    MODEL_PATH = "checkpoints/model.onnx"


# ============================================================
# Mel spectrogram (numpy/librosa, no torch dependency needed)
# ============================================================
def compute_mel(audio, sr=SR):
    """Compute log-mel spectrogram matching the training pipeline."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    return mel_db  # (n_mels, T)


# ============================================================
# Main inference
# ============================================================
def main():
    # Load taxonomy for column order
    tax = pd.read_csv(TAXONOMY_PATH)
    species_cols = sorted(tax["primary_label"].astype(str).tolist())

    # Load ONNX model
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts)
    input_name = sess.get_inputs()[0].name

    # Check if model expects raw audio or mel spectrogram
    input_shape = sess.get_inputs()[0].shape
    print(f"Model input: name={input_name}, shape={input_shape}")

    # Find test files
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.ogg")))
    if not test_files:
        # No test files available (local dev) — generate dummy submission
        print("No test files found. Generating dummy submission from sample_submission.csv")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub[species_cols] = 0.0
        sub.to_csv("submission.csv", index=False)
        return

    print(f"Processing {len(test_files)} test soundscapes...")

    rows = []
    num_samples_per_seg = SR * DURATION  # 160000

    for fpath in test_files:
        fname = os.path.basename(fpath)
        fname_no_ext = fname.replace(".ogg", "")

        # Load full 1-minute file
        audio, file_sr = sf.read(fpath, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (should already be 32kHz)
        if file_sr != SR:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)

        # Split into 5-second segments (12 segments for 60s)
        total_samples = SR * 60
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)))

        segments = []
        end_times = []
        for i in range(12):
            start = i * num_samples_per_seg
            seg = audio[start : start + num_samples_per_seg]
            segments.append(seg)
            end_times.append((i + 1) * DURATION)

        # Batch inference
        batch = np.stack(segments, axis=0).astype(np.float32)  # (12, 160000)
        logits = sess.run(None, {input_name: batch})[0]        # (12, 234)
        probs = 1.0 / (1.0 + np.exp(-logits))                  # sigmoid

        for j, end_t in enumerate(end_times):
            row_id = f"{fname_no_ext}_{end_t}"
            row = {"row_id": row_id}
            for k, sp in enumerate(species_cols):
                row[sp] = float(probs[j, k])
            rows.append(row)

    sub = pd.DataFrame(rows)

    # Ensure column order matches sample submission
    if os.path.exists(SAMPLE_SUB_PATH):
        sample = pd.read_csv(SAMPLE_SUB_PATH, nrows=1)
        sub = sub[sample.columns]

    sub.to_csv("submission.csv", index=False)
    print(f"Submission saved: {sub.shape[0]} rows, {sub.shape[1]} columns")


if __name__ == "__main__":
    main()
