"""
BirdCLEF 2026 — Inference Notebook v4
Multi-model ensemble with rank averaging, non-bird specialist, temporal smoothing.

Upload this as a Kaggle notebook. Attach ONNX models as Kaggle Datasets.
"""
import os
import time
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

TEMPORAL_SMOOTH = True
SMOOTH_WEIGHTS = np.array([0.15, 0.7, 0.15])

# ============================================================
# Paths
# ============================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    TEST_DIR = "/kaggle/input/birdclef-2026/test_soundscapes"
    TAXONOMY_PATH = "/kaggle/input/birdclef-2026/taxonomy.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/birdclef-2026/sample_submission.csv"

    MAIN_MODEL_DIRS = [
        "/kaggle/input/birdclef2026-models",
        "/kaggle/input/birdclef2026-model",
        "/kaggle/input/birdclef2026-model-v2",
    ]
    SPECIALIST_DIR = "/kaggle/input/birdclef2026-specialist"
    SPECIALIST_MAPPING_PATH = "/kaggle/input/birdclef2026-specialist/specialist_mapping.npy"
else:
    TEST_DIR = "data/test_soundscapes"
    TAXONOMY_PATH = "data/taxonomy.csv"
    SAMPLE_SUB_PATH = "data/sample_submission.csv"
    MAIN_MODEL_DIRS = ["checkpoints"]
    SPECIALIST_DIR = "checkpoints"
    SPECIALIST_MAPPING_PATH = "checkpoints/specialist_mapping.npy"


def find_onnx_models(dirs, prefix="model"):
    """Find all .onnx files across multiple directories."""
    paths = []
    seen = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".onnx") and not f.startswith("specialist"):
                path = os.path.join(d, f)
                if path not in seen:
                    paths.append(path)
                    seen.add(path)
    return paths


def find_specialist_model(specialist_dir):
    if not os.path.isdir(specialist_dir):
        return None
    for f in sorted(os.listdir(specialist_dir)):
        if f.startswith("specialist") and f.endswith(".onnx"):
            return os.path.join(specialist_dir, f)
    return None


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
    if not TEMPORAL_SMOOTH or len(probs) < 2:
        return probs

    smoothed = np.zeros_like(probs)
    n = len(probs)

    for i in range(n):
        if i == 0:
            w = np.array([weights[1] + weights[0], weights[2]])
            w /= w.sum()
            smoothed[i] = w[0] * probs[i] + w[1] * probs[i + 1]
        elif i == n - 1:
            w = np.array([weights[0], weights[1] + weights[2]])
            w /= w.sum()
            smoothed[i] = w[0] * probs[i - 1] + w[1] * probs[i]
        else:
            smoothed[i] = (weights[0] * probs[i - 1] +
                           weights[1] * probs[i] +
                           weights[2] * probs[i + 1])
    return smoothed


# ============================================================
# Rank averaging
# ============================================================
def rank_average(prob_list):
    """Rank-average across multiple model predictions. More robust than logit/prob averaging for AUC."""
    if len(prob_list) == 1:
        return prob_list[0]

    n_samples, n_classes = prob_list[0].shape
    ranked = np.zeros((n_samples, n_classes), dtype=np.float64)

    for probs in prob_list:
        for j in range(n_classes):
            col = probs[:, j]
            order = col.argsort()
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(n_samples, dtype=np.float64)
            ranked[:, j] += ranks

    ranked /= (len(prob_list) * n_samples)
    return ranked.astype(np.float32)


def rank_transform(probs):
    n_samples, n_classes = probs.shape
    ranked = np.zeros((n_samples, n_classes), dtype=np.float32)
    for j in range(n_classes):
        col = probs[:, j]
        order = col.argsort()
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(n_samples, dtype=np.float32) / max(n_samples, 1)
        ranked[:, j] = ranks
    return ranked


def sigmoid(logits):
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def load_specialist_mapping(path):
    if not os.path.exists(path):
        return None
    mapping = np.load(path, allow_pickle=True).item()
    indices = [int(i) for i in mapping["nonbird_indices"]]
    if any(i < 0 or i >= NUM_CLASSES for i in indices):
        raise ValueError("specialist_mapping.npy contains out-of-range class indices")
    return indices


# ============================================================
# Load ONNX session
# ============================================================
def load_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() // 2)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts)


# ============================================================
# Main inference
# ============================================================
def main():
    t_start = time.time()

    tax = pd.read_csv(TAXONOMY_PATH)
    species_cols = sorted(tax["primary_label"].astype(str).tolist())

    # --- Load main models ---
    main_paths = find_onnx_models(MAIN_MODEL_DIRS)
    main_sessions = []
    for p in main_paths:
        try:
            sess = load_session(p)
            main_sessions.append(sess)
            print(f"Loaded main model: {p}")
        except Exception as e:
            print(f"Failed to load {p}: {e}")

    # --- Load specialist model ---
    specialist_sess = None
    specialist_indices = None
    spec_path = find_specialist_model(SPECIALIST_DIR)
    if spec_path:
        try:
            specialist_sess = load_session(spec_path)
            specialist_indices = load_specialist_mapping(SPECIALIST_MAPPING_PATH)
            if specialist_indices is not None:
                print(f"Loaded specialist: {spec_path} ({len(specialist_indices)} species)")
            else:
                print(f"Specialist model found but mapping missing: {SPECIALIST_MAPPING_PATH}")
                specialist_sess = None
        except Exception as e:
            print(f"Failed to load specialist: {e}")
            specialist_sess = None

    if not main_sessions:
        print("No models found. Generating dummy submission.")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub[species_cols] = 0.0
        sub.to_csv("submission.csv", index=False)
        return

    print(f"Ensemble: {len(main_sessions)} main + {'1 specialist' if specialist_sess else 'no specialist'}")

    # --- Find test files ---
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.ogg")))
    if not test_files:
        print("No test files found. Generating dummy submission.")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub[species_cols] = 0.0
        sub.to_csv("submission.csv", index=False)
        return

    print(f"Processing {len(test_files)} test soundscapes...")

    main_input_names = [sess.get_inputs()[0].name for sess in main_sessions]
    specialist_input_name = specialist_sess.get_inputs()[0].name if specialist_sess is not None else None
    num_samples_per_seg = SR * DURATION

    # --- Process files: keep predictions, not all mels, to preserve memory ---
    print("Running inference...")
    row_meta = []
    main_prob_parts = [[] for _ in main_sessions]
    specialist_prob_parts = []
    mel_seconds = 0.0
    infer_seconds = 0.0

    for fpath in test_files:
        t0 = time.time()
        fname = os.path.basename(fpath)
        audio, file_sr = sf.read(fpath, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)

        total_samples = SR * 60
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)))

        mels = []
        for i in range(12):
            start = i * num_samples_per_seg
            seg = audio[start:start + num_samples_per_seg]
            mel = compute_mel(seg)
            mels.append(mel)
            row_meta.append((fname.replace(".ogg", ""), (i + 1) * DURATION))

        batch = np.stack(mels, axis=0)[:, np.newaxis, :, :].astype(np.float32)
        mel_seconds += time.time() - t0

        t1 = time.time()
        for model_i, sess in enumerate(main_sessions):
            logits = sess.run(None, {main_input_names[model_i]: batch})[0]
            main_prob_parts[model_i].append(sigmoid(logits).astype(np.float32))

        if specialist_sess is not None and specialist_indices is not None:
            spec_logits = specialist_sess.run(None, {specialist_input_name: batch})[0]
            specialist_prob_parts.append(sigmoid(spec_logits).astype(np.float32))
        infer_seconds += time.time() - t1

    main_prob_list = [np.concatenate(parts, axis=0) for parts in main_prob_parts]
    if len(main_prob_list) > 1:
        final_probs_all = rank_average(main_prob_list)
    else:
        final_probs_all = main_prob_list[0]

    if specialist_sess is not None and specialist_indices is not None and specialist_prob_parts:
        spec_probs_all = np.concatenate(specialist_prob_parts, axis=0)
        if spec_probs_all.shape[1] < len(specialist_indices):
            print("Specialist output has fewer columns than mapping; disabling specialist blend.")
        else:
            spec_scores = rank_transform(spec_probs_all) if len(main_prob_list) > 1 else spec_probs_all
            for local_i, global_i in enumerate(specialist_indices):
                final_probs_all[:, global_i] = (
                    0.6 * spec_scores[:, local_i] + 0.4 * final_probs_all[:, global_i]
                )

    # Temporal smoothing must stay per soundscape file after global ensemble ranking.
    for offset in range(0, len(row_meta), 12):
        final_probs_all[offset:offset + 12] = temporal_smooth(final_probs_all[offset:offset + 12])

    rows = []
    for row_i, (fname_no_ext, end_t) in enumerate(row_meta):
        row = {"row_id": f"{fname_no_ext}_{end_t}"}
        for k, sp in enumerate(species_cols):
            row[sp] = float(final_probs_all[row_i, k])
        rows.append(row)

    sub = pd.DataFrame(rows)

    if os.path.exists(SAMPLE_SUB_PATH):
        sample = pd.read_csv(SAMPLE_SUB_PATH, nrows=1)
        sub = sub[sample.columns]

    sub.to_csv("submission.csv", index=False)

    t_end = time.time()
    print(f"Submission saved: {sub.shape[0]} rows, {sub.shape[1]} columns")
    print(f"Total time: {t_end - t_start:.1f}s ({(t_end - t_start)/60:.1f}min)")
    print(f"  Mel: {mel_seconds:.1f}s, Model inference: {infer_seconds:.1f}s")


if __name__ == "__main__":
    main()
