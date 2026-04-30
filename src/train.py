"""Training v3: Domain-adaptive training for BirdCLEF 2026.

Key changes from v1/v2:
1. Soundscape-first validation (grouped by site, class-wise macro AUC)
2. Soundscape-only species handled via soundscape segment training
3. Aggressive background mixing (focal onto soundscape backgrounds)
4. Pseudo-label integration with confidence filtering
5. Class-balanced sampling with taxon-aware quotas
6. Upgraded backbone option (EfficientNetV2-S)
7. Focal loss for better rare-class handling

Usage:
    python -m src.train [--backbone b0|v2s] [--pseudo] [--epochs 40]
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import soundfile as sf

import src.config as CFG
from src.model import BirdModel, MelSpecTransform, BirdBackbone
from src.dataset import build_label_map
from src.augment import SpecAugment, gain_augment, time_shift


# ============================================================
# Config
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="v2s", choices=["b0", "v2s"])
    parser.add_argument("--pseudo", action="store_true", help="Use pseudo-labels")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--soundscape_ratio", type=float, default=0.4,
                        help="Fraction of each batch from soundscape domain")
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--bg_mix_prob", type=float, default=0.6)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


BACKBONE_MAP = {
    "b0": "tf_efficientnet_b0.ns_jft_in1k",
    "v2s": "tf_efficientnetv2_s.in21k_ft_in1k",
}


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Focal Loss
# ============================================================
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ============================================================
# Dataset
# ============================================================
class UnifiedDataset(Dataset):
    """Handles focal, soundscape, and pseudo-labeled data with domain-aware augmentation."""

    def __init__(self, entries, label_map, bg_pool=None, training=True,
                 mixup_prob=0.5, bg_mix_prob=0.6):
        self.entries = entries
        self.label_map = label_map
        self.bg_pool = bg_pool or []
        self.training = training
        self.mixup_prob = mixup_prob
        self.bg_mix_prob = bg_mix_prob
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

            # Additive mixup with another sample
            if random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self.entries) - 1)
                audio2 = self._load_audio(self.entries[idx2])
                audio2 = gain_augment(audio2, min_db=-8, max_db=8)
                target2 = self.entries[idx2]["target"]
                lam = np.random.beta(0.5, 0.5)
                audio = lam * audio + (1 - lam) * audio2
                target = np.maximum(target, target2)

            # Background mixing with soundscape (domain adaptation)
            elif self.bg_pool and random.random() < self.bg_mix_prob:
                bg_entry = random.choice(self.bg_pool)
                bg_audio = self._load_audio(bg_entry)
                snr_db = np.random.uniform(-3, 15)
                amp = 10 ** (-snr_db / 20.0)
                audio = audio + amp * bg_audio
                if "target" in bg_entry:
                    target = np.maximum(target, bg_entry["target"])

        # Normalize amplitude
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return torch.from_numpy(audio.astype(np.float32)), torch.from_numpy(target.astype(np.float32))

    def _load_audio(self, entry):
        path = entry["path"]
        offset = entry.get("offset_sec", 0.0)

        try:
            info = sf.info(path)
            file_sr = info.samplerate
            start_frame = int(offset * file_sr)
            num_frames = int(CFG.DURATION * file_sr)
            audio, _ = sf.read(path, start=start_frame, stop=start_frame + num_frames, dtype="float32")
        except Exception:
            return np.zeros(self.num_samples, dtype=np.float32)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))
        elif len(audio) > self.num_samples:
            if self.training:
                start = random.randint(0, len(audio) - self.num_samples)
            else:
                start = 0
            audio = audio[start:start + self.num_samples]

        return audio


# ============================================================
# Data preparation
# ============================================================
def build_focal_entries(df, label_map):
    entries = []
    for _, row in df.iterrows():
        path = os.path.join(CFG.TRAIN_AUDIO_DIR, row["filename"])
        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        label = str(row["primary_label"])
        if label in label_map:
            target[label_map[label]] = 1.0
        # Secondary labels at reduced weight
        if pd.notna(row.get("secondary_labels")) and str(row["secondary_labels"]) not in ("[]", ""):
            try:
                sec_labels = eval(row["secondary_labels"]) if isinstance(row["secondary_labels"], str) else []
                for sl in sec_labels:
                    sl = str(sl)
                    if sl in label_map:
                        target[label_map[sl]] = 0.3
            except Exception:
                pass
        entries.append({
            "path": path, "offset_sec": 0.0, "target": target,
            "primary_label": label, "domain": "focal"
        })
    return entries


def build_soundscape_entries(df, label_map):
    entries = []
    for _, row in df.iterrows():
        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        for label in str(row["primary_label"]).split(";"):
            label = label.strip()
            if label in label_map:
                target[label_map[label]] = 1.0

        entries.append({
            "path": path, "offset_sec": offset_sec, "target": target,
            "primary_label": str(row["primary_label"]).split(";")[0], "domain": "soundscape"
        })
    return entries


def build_pseudo_entries(pseudo_df, label_map, confidence_threshold=0.6):
    """Build entries from pseudo-labels with confidence filtering."""
    species_list = sorted(label_map.keys())
    entries = []
    for _, row in pseudo_df.iterrows():
        if row.get("max_prob", 1.0) < confidence_threshold:
            continue

        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        for k, sp in enumerate(species_list):
            if sp in row and row[sp] > 0.2:
                target[k] = float(min(row[sp], 1.0))

        if target.max() < 0.3:
            continue

        primary = species_list[target.argmax()]
        entries.append({
            "path": path, "offset_sec": offset_sec, "target": target,
            "primary_label": primary, "domain": "pseudo"
        })
    return entries


def build_background_pool(soundscape_dir):
    """Build pool of random soundscape segments for background mixing."""
    import glob
    files = sorted(glob.glob(os.path.join(soundscape_dir, "*.ogg")))
    pool = []
    for f in files:
        try:
            info = sf.info(f)
            duration = info.duration
            n_segments = int(duration // CFG.DURATION)
            for i in range(n_segments):
                pool.append({
                    "path": f,
                    "offset_sec": i * CFG.DURATION,
                    "target": np.zeros(CFG.NUM_CLASSES, dtype=np.float32),
                })
        except Exception:
            continue
    return pool


# ============================================================
# Validation split: by soundscape site
# ============================================================
def split_soundscape_by_site(entries, val_ratio=0.3):
    """Split soundscape entries by recording site for proper validation."""
    sites = {}
    for e in entries:
        fname = os.path.basename(e["path"])
        # Format: BC2026_Train_XXXX_SYY_...
        parts = fname.split("_")
        site = parts[3] if len(parts) > 3 else "unknown"
        sites.setdefault(site, []).append(e)

    site_list = sorted(sites.keys())
    random.shuffle(site_list)

    val_count = max(1, int(len(site_list) * val_ratio))
    val_sites = set(site_list[:val_count])

    train_entries = []
    val_entries = []
    for site, site_entries in sites.items():
        if site in val_sites:
            val_entries.extend(site_entries)
        else:
            train_entries.extend(site_entries)

    return train_entries, val_entries, val_sites


# ============================================================
# Class-balanced sampler with taxon awareness
# ============================================================
def build_balanced_sampler(entries, num_samples_per_epoch):
    """Inverse-sqrt frequency weighting, boosted for soundscape-only species."""
    label_counts = {}
    for e in entries:
        pl = e["primary_label"]
        label_counts[pl] = label_counts.get(pl, 0) + 1

    # Identify soundscape-only species (boost their weight)
    soundscape_only = set()
    focal_species = set(e["primary_label"] for e in entries if e["domain"] == "focal")
    for e in entries:
        if e["primary_label"] not in focal_species:
            soundscape_only.add(e["primary_label"])

    weights = []
    for e in entries:
        pl = e["primary_label"]
        w = 1.0 / (label_counts[pl] ** 0.5)
        # Boost soundscape-domain entries
        if e["domain"] in ("soundscape", "pseudo"):
            w *= 3.0
        # Extra boost for soundscape-only species
        if pl in soundscape_only:
            w *= 2.0
        weights.append(w)

    return WeightedRandomSampler(weights, num_samples=num_samples_per_epoch, replacement=True)


# ============================================================
# Metrics
# ============================================================
def compute_classwise_macro_auc(targets, preds):
    """Class-wise macro AUC (competition metric)."""
    aucs = []
    per_class = {}
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets[:, i]):
            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
                aucs.append(auc)
                per_class[i] = auc
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0, per_class


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device, spec_aug, scaler=None):
    model.train()
    losses = []
    mel_transform = model.mel_spec
    total = len(loader)

    for i, (audio, targets) in enumerate(loader):
        audio, targets = audio.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.no_grad():
                mel = mel_transform(audio)
            mel = spec_aug(mel)
            mel = mel.unsqueeze(1)
            logits = model.backbone(mel)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.item())
        if i % 100 == 0:
            print(f"  [train] batch {i}/{total} | loss={loss.item():.4f}", flush=True)

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for audio, targets in loader:
        audio = audio.to(device)
        logits = model(audio)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    auc, per_class = compute_classwise_macro_auc(all_targets, all_preds)
    return auc, per_class


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: backbone={args.backbone}, epochs={args.epochs}, bs={args.batch_size}, "
          f"lr={args.lr}, pseudo={args.pseudo}")

    label_map = build_label_map()
    species_list = sorted(label_map.keys())

    # --- Load data ---
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    focal_entries = build_focal_entries(train_df, label_map)
    print(f"Focal entries: {len(focal_entries)}")

    sl_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    soundscape_entries = build_soundscape_entries(sl_df, label_map)
    print(f"Soundscape entries: {len(soundscape_entries)}")

    pseudo_entries = []
    if args.pseudo:
        pseudo_path = os.path.join(CFG.DATA_DIR, "pseudo_labels.csv")
        if os.path.exists(pseudo_path):
            pseudo_df = pd.read_csv(pseudo_path)
            pseudo_entries = build_pseudo_entries(pseudo_df, label_map)
            print(f"Pseudo-label entries: {len(pseudo_entries)}")
        else:
            print("WARNING: --pseudo specified but no pseudo_labels.csv found")

    # --- Validation: soundscape-based, split by site ---
    sc_train, sc_val, val_sites = split_soundscape_by_site(soundscape_entries, val_ratio=0.3)
    print(f"Soundscape train: {len(sc_train)}, val: {len(sc_val)} (sites: {val_sites})")

    # --- Background pool for mixing ---
    bg_pool = build_background_pool(CFG.TRAIN_SOUNDSCAPES_DIR)
    print(f"Background pool: {len(bg_pool)} segments")

    # --- Combine training data ---
    # Oversample soundscape entries to balance with focal
    sc_oversample = max(1, len(focal_entries) // (3 * max(len(sc_train), 1)))
    all_train = focal_entries + sc_train * sc_oversample + pseudo_entries
    print(f"Total training entries: {len(all_train)} "
          f"(focal={len(focal_entries)}, sc={len(sc_train)}x{sc_oversample}, pseudo={len(pseudo_entries)})")

    # --- Datasets ---
    train_ds = UnifiedDataset(all_train, label_map, bg_pool=bg_pool, training=True,
                              mixup_prob=args.mixup_prob, bg_mix_prob=args.bg_mix_prob)
    val_ds = UnifiedDataset(sc_val, label_map, training=False)

    # --- Sampler ---
    sampler = build_balanced_sampler(all_train, num_samples_per_epoch=len(all_train))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    model_name = BACKBONE_MAP[args.backbone]
    print(f"Loading model: {model_name}")
    model = BirdModel(pretrained=True, model_name=model_name).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        matched, total = 0, 0
        for k in state:
            total += 1
            if k in model.state_dict() and state[k].shape == model.state_dict()[k].shape:
                matched += 1
        model.load_state_dict(state, strict=False)
        print(f"Resumed from {args.resume} ({matched}/{total} params matched)")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {param_count:.1f}M")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = FocalBCELoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    spec_aug = SpecAugment(freq_mask_param=24, time_mask_param=50,
                           n_freq_masks=2, n_time_masks=2).to(device)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # --- Training loop ---
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    best_auc = 0.0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, spec_aug, scaler)

        print("  Validating on soundscape data...", flush=True)
        val_auc, per_class = validate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_auc(soundscape)={val_auc:.4f} | lr={lr:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(CFG.CHECKPOINT_DIR, "best_v3.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (AUC={best_auc:.4f})")

        # Log per-taxon AUC every 5 epochs
        if (epoch + 1) % 5 == 0 and per_class:
            taxonomy = pd.read_csv(os.path.join(CFG.DATA_DIR, "taxonomy.csv"))
            tax_map = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))
            taxon_aucs = {}
            for cls_idx, auc in per_class.items():
                sp = species_list[cls_idx]
                taxon = tax_map.get(sp, "Unknown")
                taxon_aucs.setdefault(taxon, []).append(auc)
            print("  Per-taxon AUC:")
            for taxon, aucs in sorted(taxon_aucs.items()):
                print(f"    {taxon}: {np.mean(aucs):.4f} ({len(aucs)} species)")

    print(f"\nBest soundscape validation AUC: {best_auc:.4f}")
    print("Done. Export with: python -m src.export_onnx --checkpoint best_v3.pt")


if __name__ == "__main__":
    main()
