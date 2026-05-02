"""Pre-extract mel spectrograms to .npy for fast training.

Usage:
    python -m src.preprocess --mode focal --crops 3
    python -m src.preprocess --mode soundscape
    python -m src.preprocess --mode soundscape_unlabeled
"""
import os
import argparse
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import torch

import src.config as CFG
from src.model import MelSpecTransform
from src.dataset import build_label_map

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["focal", "soundscape", "soundscape_unlabeled", "pseudo"])
    parser.add_argument("--crops", type=int, default=3, help="Random crops per focal recording")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pseudo_path", default=None, help="Pseudo-label CSV path for --mode pseudo")
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--soft_threshold", type=float, default=0.2)
    return parser.parse_args()


def split_label_string(value):
    return [label.strip() for label in str(value).split(";") if label.strip()]


def site_from_filename(filename):
    parts = os.path.basename(str(filename)).split("_")
    return parts[3] if len(parts) > 3 else "unknown"


def seconds_to_hms(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time_to_seconds(value):
    parts = str(value).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def save_manifest(rows, out_dir):
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)


def progress_iter(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def extract_mel_numpy(audio_np, mel_transform):
    """Convert raw audio numpy array to mel spectrogram numpy array."""
    with torch.no_grad():
        wav = torch.from_numpy(audio_np).float().unsqueeze(0)
        mel = mel_transform(wav)  # (1, n_mels, T)
    return mel.squeeze(0).numpy().astype(np.float16)


def process_focal(crops_per_file, mel_transform):
    out_dir = os.path.join(CFG.PRECOMPUTED_DIR, "focal")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    label_map = build_label_map()
    num_samples = CFG.SR * CFG.DURATION
    total = len(train_df)
    manifest = []

    row_iter = progress_iter(train_df.iterrows(), total=total, desc="Focal audio", unit="file")
    for idx, row in row_iter:
        path = os.path.join(CFG.TRAIN_AUDIO_DIR, row["filename"])
        if not os.path.exists(path):
            continue

        try:
            audio, file_sr = sf.read(path, dtype="float32")
        except Exception:
            continue

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        label = str(row["primary_label"])
        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float16)
        if label in label_map:
            target[label_map[label]] = 1.0

        if pd.notna(row.get("secondary_labels")) and str(row["secondary_labels"]) not in ("[]", ""):
            try:
                sec_labels = eval(row["secondary_labels"]) if isinstance(row["secondary_labels"], str) else []
                for sl in sec_labels:
                    sl = str(sl)
                    if sl in label_map:
                        target[label_map[sl]] = 0.3
            except Exception:
                pass

        for crop_i in range(crops_per_file):
            if len(audio) <= num_samples:
                seg = np.pad(audio, (0, max(0, num_samples - len(audio))))
            else:
                start = np.random.randint(0, len(audio) - num_samples)
                seg = audio[start:start + num_samples]

            mel = extract_mel_numpy(seg, mel_transform)
            fname = f"focal_{idx:06d}_c{crop_i}"
            mel_name = f"{fname}_mel.npy"
            target_name = f"{fname}_target.npy"
            np.save(os.path.join(out_dir, mel_name), mel)
            np.save(os.path.join(out_dir, target_name), target)
            manifest.append({
                "stem": fname,
                "mel_path": mel_name,
                "target_path": target_name,
                "filename": row["filename"],
                "start": "",
                "primary_label": label,
                "labels": label,
                "domain": "focal",
                "site": "focal",
                "source_idx": idx,
                "crop": crop_i,
            })

        if tqdm is None and (idx + 1) % 1000 == 0:
            print(f"  Focal: {idx+1}/{total}", flush=True)

    save_manifest(manifest, out_dir)
    print(f"Focal done. Output: {out_dir}")


def process_soundscape_labeled(mel_transform):
    out_dir = os.path.join(CFG.PRECOMPUTED_DIR, "soundscape_labeled")
    os.makedirs(out_dir, exist_ok=True)

    sl_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    label_map = build_label_map()
    num_samples = CFG.SR * CFG.DURATION
    manifest = []

    row_iter = progress_iter(sl_df.iterrows(), total=len(sl_df), desc="Labeled soundscapes", unit="segment")
    for idx, row in row_iter:
        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        if not os.path.exists(path):
            continue

        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        try:
            info = sf.info(path)
            start_frame = int(offset_sec * info.samplerate)
            num_frames = int(CFG.DURATION * info.samplerate)
            audio, _ = sf.read(path, start=start_frame, stop=start_frame + num_frames, dtype="float32")
        except Exception:
            continue

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if len(audio) < num_samples:
            audio = np.pad(audio, (0, num_samples - len(audio)))
        elif len(audio) > num_samples:
            audio = audio[:num_samples]

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float16)
        labels = split_label_string(row["primary_label"])
        for label in labels:
            if label in label_map:
                target[label_map[label]] = 1.0

        mel = extract_mel_numpy(audio, mel_transform)
        fname = f"sc_{idx:06d}"
        mel_name = f"{fname}_mel.npy"
        target_name = f"{fname}_target.npy"
        np.save(os.path.join(out_dir, mel_name), mel)
        np.save(os.path.join(out_dir, target_name), target)
        manifest.append({
            "stem": fname,
            "mel_path": mel_name,
            "target_path": target_name,
            "filename": row["filename"],
            "start": row["start"],
            "primary_label": labels[0] if labels else "",
            "labels": ";".join(labels),
            "domain": "soundscape",
            "site": site_from_filename(row["filename"]),
            "source_idx": idx,
        })

        if tqdm is None and (idx + 1) % 500 == 0:
            print(f"  Soundscape labeled: {idx+1}/{len(sl_df)}", flush=True)

    save_manifest(manifest, out_dir)
    print(f"Soundscape labeled done. Output: {out_dir}")


def process_soundscape_unlabeled(mel_transform):
    """Pre-extract all 5s segments from unlabeled soundscapes for pseudo-labeling."""
    out_dir = os.path.join(CFG.PRECOMPUTED_DIR, "soundscape_unlabeled")
    os.makedirs(out_dir, exist_ok=True)

    labeled_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    labeled_keys = set()
    key_iter = progress_iter(
        labeled_df.iterrows(),
        total=len(labeled_df),
        desc="Labeled keys",
        unit="row",
        leave=False,
    )
    for _, row in key_iter:
        labeled_keys.add(f"{row['filename']}_{row['start']}")

    all_files = sorted(glob.glob(os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, "*.ogg")))
    num_samples = CFG.SR * CFG.DURATION
    seg_count = 0

    manifest = []

    file_iter = progress_iter(enumerate(all_files), total=len(all_files), desc="Unlabeled files", unit="file")
    for fi, fpath in file_iter:
        fname = os.path.basename(fpath)
        try:
            audio, _ = sf.read(fpath, dtype="float32")
        except Exception:
            continue

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        n_segs = len(audio) // num_samples
        for i in range(n_segs):
            seconds = i * CFG.DURATION
            start_str = seconds_to_hms(seconds)

            if f"{fname}_{start_str}" in labeled_keys:
                continue

            seg = audio[i * num_samples:(i + 1) * num_samples]
            if len(seg) < num_samples:
                seg = np.pad(seg, (0, num_samples - len(seg)))

            mel = extract_mel_numpy(seg, mel_transform)
            out_name = f"ul_{seg_count:07d}"
            mel_name = f"{out_name}_mel.npy"
            np.save(os.path.join(out_dir, mel_name), mel)
            manifest.append({
                "idx": seg_count,
                "stem": out_name,
                "mel_path": mel_name,
                "filename": fname,
                "start": start_str,
                "site": site_from_filename(fname),
            })
            seg_count += 1

        if hasattr(file_iter, "set_postfix"):
            file_iter.set_postfix(segments=seg_count)
        elif (fi + 1) % 20 == 0:
            print(f"  Unlabeled: {fi+1}/{len(all_files)} files, {seg_count} segments", flush=True)

    save_manifest(manifest, out_dir)
    print(f"Unlabeled done. {seg_count} segments. Output: {out_dir}")


def build_unlabeled_mel_lookup():
    manifest_path = os.path.join(CFG.PRECOMPUTED_DIR, "soundscape_unlabeled", "manifest.csv")
    if not os.path.exists(manifest_path):
        return {}

    manifest = pd.read_csv(manifest_path)
    lookup = {}
    base_dir = os.path.dirname(manifest_path)
    row_iter = progress_iter(
        manifest.iterrows(),
        total=len(manifest),
        desc="Unlabeled mel lookup",
        unit="row",
        leave=False,
    )
    for _, row in row_iter:
        mel_path = row.get("mel_path", "")
        if pd.isna(mel_path) or not str(mel_path):
            stem = row.get("stem", f"ul_{int(row['idx']):07d}")
            mel_path = f"{stem}_mel.npy"
        mel_path = str(mel_path)
        if not os.path.isabs(mel_path):
            mel_path = os.path.join(base_dir, mel_path)
        key = f"{row['filename']}_{row['start']}"
        if os.path.exists(mel_path):
            lookup[key] = mel_path
    return lookup


def load_soundscape_segment(filename, start, audio_cache):
    if filename not in audio_cache:
        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, filename)
        audio, _ = sf.read(path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio_cache[filename] = audio

    audio = audio_cache[filename]
    num_samples = CFG.SR * CFG.DURATION
    offset = parse_time_to_seconds(start) * CFG.SR
    seg = audio[offset:offset + num_samples]
    if len(seg) < num_samples:
        seg = np.pad(seg, (0, num_samples - len(seg)))
    return seg


def process_pseudo_labels(mel_transform, pseudo_path, confidence_threshold, soft_threshold):
    out_dir = os.path.join(CFG.PRECOMPUTED_DIR, "pseudo")
    os.makedirs(out_dir, exist_ok=True)

    if pseudo_path is None:
        pseudo_path = os.path.join(CFG.DATA_DIR, "pseudo_labels.csv")
    if not os.path.exists(pseudo_path):
        raise FileNotFoundError(f"Pseudo-label CSV not found: {pseudo_path}")

    pseudo_df = pd.read_csv(pseudo_path)
    label_map = build_label_map()
    species_list = sorted(label_map.keys())
    mel_lookup = build_unlabeled_mel_lookup()
    audio_cache = {}
    manifest = []
    kept = 0

    row_iter = progress_iter(pseudo_df.iterrows(), total=len(pseudo_df), desc="Pseudo labels", unit="row")
    for idx, row in row_iter:
        if "max_prob" in row and float(row["max_prob"]) < confidence_threshold:
            continue

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float16)
        labels = []
        for k, sp in enumerate(species_list):
            if sp not in row:
                continue
            prob = float(row[sp])
            if prob > soft_threshold:
                target[k] = min(prob, 1.0)
                labels.append(sp)

        if not labels:
            continue

        filename = row["filename"]
        start = row["start"]
        key = f"{filename}_{start}"

        if key in mel_lookup:
            mel = np.load(mel_lookup[key]).astype(np.float16)
        else:
            seg = load_soundscape_segment(filename, start, audio_cache)
            mel = extract_mel_numpy(seg, mel_transform)

        stem = f"pseudo_{kept:07d}"
        mel_name = f"{stem}_mel.npy"
        target_name = f"{stem}_target.npy"
        np.save(os.path.join(out_dir, mel_name), mel)
        np.save(os.path.join(out_dir, target_name), target)
        manifest.append({
            "stem": stem,
            "mel_path": mel_name,
            "target_path": target_name,
            "filename": filename,
            "start": start,
            "primary_label": labels[int(np.argmax(target[[label_map[l] for l in labels]]))],
            "labels": ";".join(labels),
            "domain": "pseudo",
            "site": site_from_filename(filename),
            "source_idx": idx,
            "max_prob": float(row["max_prob"]) if "max_prob" in row else float(target.max()),
        })
        kept += 1

        if hasattr(row_iter, "set_postfix") and kept % 100 == 0:
            row_iter.set_postfix(kept=kept)
        elif tqdm is None and kept % 1000 == 0:
            print(f"  Pseudo: {kept} segments", flush=True)

    save_manifest(manifest, out_dir)
    print(f"Pseudo done. {kept} segments. Output: {out_dir}")


def main():
    args = parse_args()
    mel_transform = MelSpecTransform()
    mel_transform.eval()

    if args.mode == "focal":
        process_focal(args.crops, mel_transform)
    elif args.mode == "soundscape":
        process_soundscape_labeled(mel_transform)
    elif args.mode == "soundscape_unlabeled":
        process_soundscape_unlabeled(mel_transform)
    elif args.mode == "pseudo":
        process_pseudo_labels(
            mel_transform,
            pseudo_path=args.pseudo_path,
            confidence_threshold=args.confidence_threshold,
            soft_threshold=args.soft_threshold,
        )


if __name__ == "__main__":
    main()
