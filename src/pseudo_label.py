"""Iterative Noisy Student pseudo-labeling with power scaling and ensemble teacher.

Usage:
    # Round 1: ensemble of 3 backbones as teacher
    python -m src.pseudo_label --round 1 --checkpoints best_v4_v2s.pt best_v4_nfnet.pt best_v4_regnety.pt \
        --backbones v2s nfnet regnety --threshold 0.7 --power_gamma 1.5

    # Round 2: use R1 students as teacher, lower threshold
    python -m src.pseudo_label --round 2 --checkpoints best_v4_v2s_r1.pt best_v4_nfnet_r1.pt best_v4_regnety_r1.pt \
        --backbones v2s nfnet regnety --threshold 0.6 --power_gamma 1.5

    # Round 3-4: progressively lower
    python -m src.pseudo_label --round 3 --threshold 0.5 --power_gamma 1.3 ...
    python -m src.pseudo_label --round 4 --threshold 0.4 --power_gamma 1.3 ...
"""
import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import soundfile as sf

import src.config as CFG
from src.model import BirdModel
from src.dataset import build_label_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint filenames for ensemble teacher")
    parser.add_argument("--backbones", nargs="+", required=True,
                        help="Backbone keys matching checkpoints (e.g. v2s nfnet regnety)")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Min max-prob to keep a segment")
    parser.add_argument("--power_gamma", type=float, default=1.5,
                        help="Power scaling exponent to suppress mid-confidence noise")
    parser.add_argument("--soft_threshold", type=float, default=0.1,
                        help="Min per-species prob after power scaling to include")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--precomputed", action="store_true",
                        help="Use precomputed soundscape_unlabeled mels instead of decoding .ogg files")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    return parser.parse_args()


def load_ensemble(checkpoints, backbones, device):
    """Load multiple models as an ensemble teacher."""
    models = []
    for ckpt, bb_key in zip(checkpoints, backbones):
        bb_name = CFG.BACKBONE_REGISTRY[bb_key]
        model = BirdModel(pretrained=False, model_name=bb_name).to(device)
        ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, ckpt)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        models.append(model)
        print(f"Loaded teacher: {ckpt} ({bb_name})")
    return models


def load_precomputed_unlabeled_records():
    manifest_path = os.path.join(CFG.PRECOMPUTED_DIR, "soundscape_unlabeled", "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Precomputed unlabeled manifest not found: {manifest_path}. "
            "Run: python -m src.preprocess --mode soundscape_unlabeled"
        )

    manifest = pd.read_csv(manifest_path)
    base_dir = os.path.dirname(manifest_path)
    records = []
    for _, row in manifest.iterrows():
        mel_path = row.get("mel_path", "")
        if pd.isna(mel_path) or not str(mel_path):
            stem = row.get("stem", f"ul_{int(row['idx']):07d}")
            mel_path = f"{stem}_mel.npy"
        mel_path = str(mel_path)
        if not os.path.isabs(mel_path):
            mel_path = os.path.join(base_dir, mel_path)
        if not os.path.exists(mel_path):
            continue
        records.append({
            "filename": row["filename"],
            "start": row["start"],
            "mel_path": mel_path,
        })
    return records


def reverse_segments(segments, precomputed):
    if precomputed:
        return [s[:, ::-1].copy() for s in segments]
    return [s[::-1].copy() for s in segments]


@torch.no_grad()
def predict_ensemble(models, segments, device, batch_size=64, precomputed=False):
    """Ensemble prediction: average sigmoid probabilities across models."""
    all_probs = np.zeros((len(segments), CFG.NUM_CLASSES), dtype=np.float64)

    for model in models:
        model_probs = []
        for i in range(0, len(segments), batch_size):
            batch = np.stack(segments[i:i + batch_size])
            tensor = torch.from_numpy(batch).float().to(device)
            if precomputed:
                tensor = tensor.unsqueeze(1)
                logits = model(tensor, precomputed=True)
            else:
                logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            model_probs.append(probs)
        model_probs = np.concatenate(model_probs, axis=0)
        all_probs += model_probs

    all_probs /= len(models)
    return all_probs


@torch.no_grad()
def predict_ensemble_tta(models, segments, device, batch_size=64, precomputed=False):
    """TTA: original + time-reversed, averaged."""
    probs_orig = predict_ensemble(models, segments, device, batch_size, precomputed=precomputed)
    segments_rev = reverse_segments(segments, precomputed)
    probs_rev = predict_ensemble(models, segments_rev, device, batch_size, precomputed=precomputed)
    return (probs_orig + probs_rev) / 2.0


def power_scale(probs, gamma):
    """Apply power scaling: p^gamma. Suppresses mid-confidence, preserves high-confidence."""
    return np.power(probs, gamma)


def generate_pseudo_labels(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Round {args.round} | threshold={args.threshold} | "
          f"power_gamma={args.power_gamma} | soft_threshold={args.soft_threshold}")
    print(f"Ensemble: {len(args.checkpoints)} models")
    print(f"Input: {'precomputed mels' if args.precomputed else 'raw soundscape audio'}")

    models = load_ensemble(args.checkpoints, args.backbones, device)
    label_map = build_label_map()
    species_list = sorted(label_map.keys())

    if args.precomputed:
        records = load_precomputed_unlabeled_records()
        print(f"Found {len(records)} precomputed unlabeled segments")

        if not records:
            print("No precomputed unlabeled segments found.")
            return

        rows = []
        total_segments = 0
        kept_segments = 0

        for start_idx in range(0, len(records), args.batch_size):
            batch_records = records[start_idx:start_idx + args.batch_size]
            segments = [np.load(r["mel_path"]).astype(np.float32) for r in batch_records]
            total_segments += len(segments)

            if args.tta:
                probs = predict_ensemble_tta(
                    models, segments, device, args.batch_size, precomputed=True)
            else:
                probs = predict_ensemble(
                    models, segments, device, args.batch_size, precomputed=True)

            probs_scaled = power_scale(probs, args.power_gamma)

            for record, prob_raw, prob_sc in zip(batch_records, probs, probs_scaled):
                max_prob = prob_raw.max()
                if max_prob < args.threshold:
                    continue

                prob_final = prob_sc.copy()
                prob_final[prob_final < args.soft_threshold] = 0.0

                if prob_final.max() < args.soft_threshold:
                    continue

                kept_segments += 1
                row = {
                    "filename": record["filename"],
                    "start": record["start"],
                    "max_prob": float(max_prob),
                }
                for k, sp in enumerate(species_list):
                    row[sp] = float(prob_final[k])
                rows.append(row)

            batch_no = start_idx // args.batch_size + 1
            if batch_no % 100 == 0:
                print(f"  [{total_segments}/{len(records)}] kept={kept_segments} "
                      f"({100*kept_segments/max(total_segments,1):.1f}%)", flush=True)
    else:
        # Find all soundscape files
        all_files = sorted(glob.glob(os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, "*.ogg")))
        print(f"Found {len(all_files)} soundscape files")

        if not all_files:
            print("No soundscape files found.")
            return

        # Exclude already-labeled segments
        labeled_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
        labeled_keys = set()
        for _, row in labeled_df.iterrows():
            labeled_keys.add(f"{row['filename']}_{row['start']}")
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

                if f"{fname}_{start_str}" in labeled_keys:
                    continue

                seg = audio[start_sample:start_sample + num_samples_per_seg]
                if len(seg) < num_samples_per_seg:
                    seg = np.pad(seg, (0, num_samples_per_seg - len(seg)))

                segments.append(seg)
                seg_starts.append(start_str)

            if not segments:
                continue

            total_segments += len(segments)

            # Ensemble prediction
            if args.tta:
                probs = predict_ensemble_tta(models, segments, device, args.batch_size)
            else:
                probs = predict_ensemble(models, segments, device, args.batch_size)

            # Power scaling to suppress noisy mid-confidence predictions
            probs_scaled = power_scale(probs, args.power_gamma)

            # Filter and save
            for j, (start_str, prob_raw, prob_sc) in enumerate(zip(seg_starts, probs, probs_scaled)):
                max_prob = prob_raw.max()
                if max_prob < args.threshold:
                    continue

                # Apply soft threshold on power-scaled probs
                prob_final = prob_sc.copy()
                prob_final[prob_final < args.soft_threshold] = 0.0

                if prob_final.max() < args.soft_threshold:
                    continue

                kept_segments += 1
                row = {"filename": fname, "start": start_str, "max_prob": float(max_prob)}
                for k, sp in enumerate(species_list):
                    row[sp] = float(prob_final[k])
                rows.append(row)

            if (fi + 1) % 20 == 0:
                print(f"  [{fi+1}/{len(all_files)}] total_segs={total_segments}, "
                      f"kept={kept_segments} ({100*kept_segments/max(total_segments,1):.1f}%)", flush=True)

    output_columns = ["filename", "start", "max_prob"] + species_list
    pseudo_df = pd.DataFrame(rows)
    for col in output_columns:
        if col not in pseudo_df.columns:
            pseudo_df[col] = 0.0 if col not in {"filename", "start"} else ""
    pseudo_df = pseudo_df[output_columns]
    os.makedirs(CFG.PSEUDO_LABEL_DIR, exist_ok=True)
    out_path = os.path.join(CFG.PSEUDO_LABEL_DIR, f"pseudo_labels_r{args.round}.csv")
    pseudo_df.to_csv(out_path, index=False)

    # Also save as default for training
    default_path = CFG.PSEUDO_LABEL_PATH
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
