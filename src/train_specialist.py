"""Non-bird specialist: separate model for Amphibia, Insecta, Mammalia, Reptilia.

These taxa have different acoustic signatures and 28 species have zero focal data.
Training on soundscape-only data with heavy oversampling.

Usage:
    python -m src.train_specialist --epochs 50 --batch_size 64 --lr 3e-4
"""
import os
import argparse
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import soundfile as sf

import src.config as CFG
from src.model import BirdModel, BirdBackbone, MelSpecTransform
from src.dataset import build_label_map
from src.augment import SpecAugment, gain_augment, time_shift


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="b0", choices=list(CFG.BACKBONE_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pseudo", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def get_nonbird_species():
    """Return sorted list of non-bird species and their indices in the full 234-class label space."""
    taxonomy = pd.read_csv(os.path.join(CFG.DATA_DIR, "taxonomy.csv"))
    label_map = build_label_map()
    species_list = sorted(label_map.keys())

    nonbird_species = []
    nonbird_indices = []
    for _, row in taxonomy.iterrows():
        sp = str(row["primary_label"])
        cls = row["class_name"]
        if cls in CFG.NON_BIRD_CLASSES and sp in label_map:
            nonbird_species.append(sp)
            nonbird_indices.append(label_map[sp])

    return nonbird_species, nonbird_indices


class SpecialistDataset(Dataset):
    """Dataset for non-bird specialist. Uses soundscape segments + focal non-bird recordings."""

    def __init__(self, entries, training=True):
        self.entries = entries
        self.training = training
        self.num_samples = CFG.SR * CFG.DURATION

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio = self._load_audio(entry)
        target = entry["target"].copy()

        if self.training:
            audio = gain_augment(audio, min_db=-8, max_db=8)
            if random.random() < 0.5:
                audio = time_shift(audio, max_shift_sec=1.5)
            if random.random() < 0.4:
                idx2 = random.randint(0, len(self.entries) - 1)
                audio2 = self._load_audio(self.entries[idx2])
                lam = np.random.beta(0.5, 0.5)
                audio = lam * audio + (1 - lam) * audio2
                target = np.maximum(target, self.entries[idx2]["target"])

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return torch.from_numpy(audio.astype(np.float32)), torch.from_numpy(target.astype(np.float32))

    def _load_audio(self, entry):
        path = entry["path"]
        offset = entry.get("offset_sec", 0.0)
        try:
            info = sf.info(path)
            start_frame = int(offset * info.samplerate)
            num_frames = int(CFG.DURATION * info.samplerate)
            audio, _ = sf.read(path, start=start_frame, stop=start_frame + num_frames, dtype="float32")
        except Exception:
            return np.zeros(self.num_samples, dtype=np.float32)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))
        elif len(audio) > self.num_samples:
            start = random.randint(0, len(audio) - self.num_samples) if self.training else 0
            audio = audio[start:start + self.num_samples]
        return audio


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def build_entries(nonbird_species, nonbird_indices):
    """Build training entries from soundscape labels + focal non-bird recordings."""
    label_map = build_label_map()
    n_nonbird = len(nonbird_species)
    nb_set = set(nonbird_species)
    nb_local_map = {sp: i for i, sp in enumerate(nonbird_species)}

    entries = []

    # Soundscape labeled segments containing non-bird species
    sl_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    for _, row in sl_df.iterrows():
        labels = [l.strip() for l in str(row["primary_label"]).split(";")]
        has_nonbird = any(l in nb_set for l in labels)
        if not has_nonbird:
            continue

        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        target = np.zeros(n_nonbird, dtype=np.float32)
        for l in labels:
            if l in nb_local_map:
                target[nb_local_map[l]] = 1.0

        primary = labels[0] if labels[0] in nb_set else nonbird_species[target.argmax()]
        entries.append({
            "path": path, "offset_sec": offset_sec, "target": target,
            "primary_label": primary, "domain": "soundscape"
        })

    # Focal recordings of non-bird species
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    for _, row in train_df.iterrows():
        sp = str(row["primary_label"])
        if sp not in nb_set:
            continue
        path = os.path.join(CFG.TRAIN_AUDIO_DIR, row["filename"])
        target = np.zeros(n_nonbird, dtype=np.float32)
        target[nb_local_map[sp]] = 1.0
        entries.append({
            "path": path, "offset_sec": 0.0, "target": target,
            "primary_label": sp, "domain": "focal"
        })

    return entries


def main():
    args = parse_args()
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nonbird_species, nonbird_indices = get_nonbird_species()
    n_nonbird = len(nonbird_species)
    print(f"Non-bird specialist: {n_nonbird} species")
    print(f"Device: {device}")

    entries = build_entries(nonbird_species, nonbird_indices)
    print(f"Total entries: {len(entries)}")

    # Split by soundscape site
    sites = {}
    for e in entries:
        if e["domain"] == "soundscape":
            fname = os.path.basename(e["path"])
            parts = fname.split("_")
            site = parts[3] if len(parts) > 3 else "unknown"
        else:
            site = "focal"
        sites.setdefault(site, []).append(e)

    site_list = [s for s in sorted(sites.keys()) if s != "focal"]
    random.shuffle(site_list)
    val_count = max(1, int(len(site_list) * 0.3))
    val_sites = set(site_list[:val_count])

    train_entries = sites.get("focal", [])[:]
    val_entries = []
    for site, se in sites.items():
        if site == "focal":
            continue
        if site in val_sites:
            val_entries.extend(se)
        else:
            train_entries.extend(se)

    # Heavy oversampling of soundscape entries
    sc_entries = [e for e in train_entries if e["domain"] == "soundscape"]
    focal_entries = [e for e in train_entries if e["domain"] == "focal"]
    oversample = max(1, len(focal_entries) // max(len(sc_entries), 1)) * 3
    train_entries = focal_entries + sc_entries * oversample

    print(f"Train: {len(train_entries)} (focal={len(focal_entries)}, sc={len(sc_entries)}x{oversample})")
    print(f"Val: {len(val_entries)} (sites: {val_sites})")

    train_ds = SpecialistDataset(train_entries, training=True)
    val_ds = SpecialistDataset(val_entries, training=False)

    # Balanced sampler
    label_counts = {}
    for e in train_entries:
        pl = e["primary_label"]
        label_counts[pl] = label_counts.get(pl, 0) + 1
    weights = [1.0 / (label_counts[e["primary_label"]] ** 0.5) for e in train_entries]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_entries), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model: smaller backbone, fewer output classes
    model_name = CFG.BACKBONE_REGISTRY[args.backbone]
    model = BirdModel(num_classes=n_nonbird, pretrained=True, model_name=model_name).to(device)

    if args.resume:
        ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, args.resume)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Resumed from {ckpt_path}")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name}, params: {param_count:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = FocalBCELoss(gamma=2.0)
    spec_aug = SpecAugment(freq_mask_param=24, time_mask_param=50,
                           n_freq_masks=2, n_time_masks=2).to(device)

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for i, (audio, targets) in enumerate(train_loader):
            audio, targets = audio.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                mel = model.mel_spec(audio)
            mel = spec_aug(mel)
            mel = mel.unsqueeze(1)
            logits = model(mel, precomputed=True)
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

            if i % 50 == 0:
                print(f"  [train] batch {i}/{len(train_loader)} | loss={loss.item():.4f}", flush=True)

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for audio, targets in val_loader:
                audio = audio.to(device)
                logits = model(audio)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        aucs = []
        for i in range(n_nonbird):
            if all_targets[:, i].sum() > 0 and all_targets[:, i].sum() < len(all_targets[:, i]):
                try:
                    aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
                except ValueError:
                    pass
        val_auc = np.mean(aucs) if aucs else 0.0

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs} | loss={np.mean(losses):.4f} | "
              f"val_auc={val_auc:.4f} | lr={lr:.6f} | evaluable_species={len(aucs)}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(CFG.CHECKPOINT_DIR, "best_specialist.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best specialist (AUC={best_auc:.4f})")

    # Save species mapping for inference. `inference.py` loads this with np.load.
    mapping = {"nonbird_species": nonbird_species, "nonbird_indices": nonbird_indices}
    np.save(os.path.join(CFG.CHECKPOINT_DIR, "specialist_mapping.npy"), mapping)

    print(f"\nBest specialist AUC: {best_auc:.4f}")
    print(f"Export: python -m src.export_onnx --checkpoint best_specialist.pt "
          f"--backbone {CFG.BACKBONE_REGISTRY[args.backbone]}")


if __name__ == "__main__":
    main()
