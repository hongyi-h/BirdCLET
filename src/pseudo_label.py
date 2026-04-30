"""Pseudo-labeling v2: Iterative, confidence-aware, with teacher ensemble support.

Usage:
    # Round 1: use best_v3.pt as teacher
    python -m src.pseudo_label_v2 --checkpoint best_v3.pt --round 1

    # Round 2: use model retrained on round-1 pseudo-labels
    python -m src.pseudo_label_v2 --checkpoint best_v3_r1.pt --round 2 --threshold 0.5
"""
import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import soundfile as sf

import src.config as CFG
from src.model import BirdModel, MelSpecTransform
from src.dataset import build_label_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_v3.pt")
    parser.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Min max-prob to keep a segment")
    parser.add_argument("--soft_threshold", type=float, default=0.15,
                        help="Min per-species prob to include in soft label")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation (3 crops + flip)")
    return parser.parse_args()


def load_model(checkpoint, backbone, device):
    model = BirdModel(pretrained=False, model_name=backbone).to(device)

    ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}")
    return model


@torch.no_grad()
def predict_segments(model, segments, device, batch_size=64):
    """Predict on a list of numpy arrays. Returns (N, 234) probabilities."""
    all_probs = []
    for i in range(0, len(segments), batch_size):
        batch = np.stack(segments[i:i+batch_size])
        tensor = torch.from_numpy(batch).float().to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_segments_tta(model, segments, device, batch_size=64):
    """TTA: original + time-reversed + two random crops averaged."""
    probs_orig = predict_segments(model, segments, device, batch_size)

    # Time reversal
    segments_rev = [s[::-1].copy() for s in segments]
    probs_rev = predict_segments(model, segments_rev, device, batch_size)

    return (probs_orig + probs_rev) / 2.0


def generate_pseudo_labels(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Round {args.round} | threshold={args.threshold} | soft_threshold={args.soft_threshold}")

    model = load_model(args.checkpoint, args.backbone, device)

    label_map = build_label_map()
    species_list = sorted(label_map.keys())

    # Find all soundscape files
    soundscape_dir = CFG.TRAIN_SOUNDSCAPES_DIR
    all_files = sorted(glob.glob(os.path.join(soundscape_dir, "*.ogg")))
    print(f"Found {len(all_files)} soundscape files")

    if not all_files:
        print("No soundscape files found. Exiting.")
        return

    # Exclude already-labeled segments
    labeled_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    labeled_keys = set()
    for _, row in labeled_df.iterrows():
        key = f"{row['filename']}_{row['start']}"
        labeled_keys.add(key)
    print(f"Excluding {len(labeled_keys)} already-labeled segments")

    num_samples_per_seg = CFG.SR * CFG.DURATION
    rows = []
    total_segments = 0
    kept_segments = 0

    for fi, fpath in enumerate(all_files):
        fname = os.path.basename(fpath)

        try:
            audio, file_sr = sf.read(fpath, dtype="float32")
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            continue

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        n_segs = len(audio) // num_samples_per_seg
        segments = []
        seg_starts = []

        for i in range(n_segs):
            start_sample = i * num_samples_per_seg
            seconds = i * CFG.DURATION
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            start_str = f"{h:02d}:{m:02d}:{s:02d}"

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

        total_segments += len(segments)

        # Predict
        if args.tta:
            probs = predict_segments_tta(model, segments, device, args.batch_size)
        else:
            probs = predict_segments(model, segments, device, args.batch_size)

        # Filter and save
        for j, (start_str, prob) in enumerate(zip(seg_starts, probs)):
            max_prob = prob.max()
            if max_prob < args.threshold:
                continue

            # Apply soft threshold
            prob_filtered = prob.copy()
            prob_filtered[prob_filtered < args.soft_threshold] = 0.0

            kept_segments += 1
            row = {"filename": fname, "start": start_str, "max_prob": float(max_prob)}
            for k, sp in enumerate(species_list):
                row[sp] = float(prob_filtered[k])
            rows.append(row)

        if (fi + 1) % 20 == 0:
            print(f"  [{fi+1}/{len(all_files)}] total_segs={total_segments}, "
                  f"kept={kept_segments} ({100*kept_segments/max(total_segments,1):.1f}%)", flush=True)

    pseudo_df = pd.DataFrame(rows)
    out_path = os.path.join(CFG.DATA_DIR, f"pseudo_labels_r{args.round}.csv")
    pseudo_df.to_csv(out_path, index=False)

    # Also save as the default pseudo_labels.csv for train_v3
    default_path = os.path.join(CFG.DATA_DIR, "pseudo_labels.csv")
    pseudo_df.to_csv(default_path, index=False)

    print(f"\nSaved {len(pseudo_df)} pseudo-labeled segments to {out_path}")
    print(f"Retention rate: {100*kept_segments/max(total_segments,1):.1f}%")
    species_detected = (pseudo_df[species_list].sum(axis=0) > 0).sum()
    print(f"Species detected: {species_detected}/{len(species_list)}")

    # Per-taxon stats
    taxonomy = pd.read_csv(os.path.join(CFG.DATA_DIR, "taxonomy.csv"))
    tax_map = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))
    taxon_counts = {}
    for sp in species_list:
        if pseudo_df[sp].sum() > 0:
            taxon = tax_map.get(sp, "Unknown")
            taxon_counts[taxon] = taxon_counts.get(taxon, 0) + 1
    print("Per-taxon species detected:")
    for taxon, count in sorted(taxon_counts.items()):
        print(f"  {taxon}: {count}")


if __name__ == "__main__":
    args = parse_args()
    generate_pseudo_labels(args)
