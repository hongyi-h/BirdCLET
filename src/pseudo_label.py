"""Pseudo-labeling pipeline for unlabeled train_soundscapes.

Usage:
    python -m src.pseudo_label

Reads the trained model, predicts on all unlabeled train_soundscapes,
and saves pseudo-labels to data/pseudo_labels.csv.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import soundfile as sf

import src.config as CFG
from src.model import BirdModel, MelSpecTransform
from src.dataset import build_label_map


def generate_pseudo_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = BirdModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(CFG.CHECKPOINT_DIR, "best.pt"), map_location=device))
    model.eval()

    # Label map for column names
    label_map = build_label_map()
    species_list = sorted(label_map.keys())

    # Find all train soundscape files
    soundscape_dir = CFG.TRAIN_SOUNDSCAPES_DIR
    all_files = sorted(glob.glob(os.path.join(soundscape_dir, "*.ogg")))
    print(f"Found {len(all_files)} soundscape files")

    if not all_files:
        print("No soundscape files found. Exiting.")
        return

    # Already-labeled segments (exclude from pseudo-labeling)
    labeled_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    labeled_keys = set()
    for _, row in labeled_df.iterrows():
        key = f"{row['filename']}_{row['start']}"
        labeled_keys.add(key)
    print(f"Excluding {len(labeled_keys)} already-labeled segments")

    num_samples_per_seg = CFG.SR * CFG.DURATION
    mel_transform = MelSpecTransform().to(device)

    rows = []
    with torch.no_grad():
        for fi, fpath in enumerate(all_files):
            fname = os.path.basename(fpath)

            # Load audio
            audio, file_sr = sf.read(fpath, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Process in 5-second segments
            total_segments = len(audio) // num_samples_per_seg
            segments = []
            seg_starts = []

            for i in range(total_segments):
                start_sample = i * num_samples_per_seg
                start_time = f"00:{i*5:02d}:00" if i * 5 < 60 else f"00:00:{i*5:02d}"
                # Proper time format
                minutes = (i * 5) // 60
                seconds = (i * 5) % 60
                start_str = f"{minutes // 60:02d}:{minutes % 60:02d}:{seconds:02d}"

                key = f"{fname}_{start_str}"
                if key in labeled_keys:
                    continue

                seg = audio[start_sample:start_sample + num_samples_per_seg]
                if len(seg) < num_samples_per_seg:
                    seg = np.pad(seg, (0, num_samples_per_seg - len(seg)))

                segments.append(seg)
                seg_starts.append(start_str)

            if not segments:
                continue

            # Batch predict
            batch_size = 32
            for batch_start in range(0, len(segments), batch_size):
                batch_segs = segments[batch_start:batch_start + batch_size]
                batch_starts = seg_starts[batch_start:batch_start + batch_size]

                audio_tensor = torch.from_numpy(np.stack(batch_segs)).float().to(device)
                logits = model(audio_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()

                for j, (start_str, prob) in enumerate(zip(batch_starts, probs)):
                    max_prob = prob.max()
                    if max_prob < 0.5:
                        continue

                    # Zero out low-confidence predictions
                    prob[prob < 0.1] = 0.0

                    row = {
                        "filename": fname,
                        "start": start_str,
                        "max_prob": max_prob,
                    }
                    for k, sp in enumerate(species_list):
                        row[sp] = float(prob[k])
                    rows.append(row)

            if (fi + 1) % 10 == 0:
                print(f"  Processed {fi+1}/{len(all_files)} files, {len(rows)} pseudo-labeled segments so far",
                      flush=True)

    pseudo_df = pd.DataFrame(rows)
    out_path = os.path.join(CFG.DATA_DIR, "pseudo_labels.csv")
    pseudo_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(pseudo_df)} pseudo-labeled segments to {out_path}")
    print(f"Species distribution: {(pseudo_df[species_list].sum(axis=0) > 0).sum()} species detected")


if __name__ == "__main__":
    generate_pseudo_labels()
