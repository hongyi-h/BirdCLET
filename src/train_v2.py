"""Training script v2: augmentation + pseudo-labels + class-balanced sampling.

Usage:
    python -m src.train_v2

Improvements over v1:
- Additive mixup and background mixing augmentation
- SpecAugment on mel spectrogram
- Pseudo-label integration (if data/pseudo_labels.csv exists)
- Class-balanced sampling (inverse sqrt frequency weighting)
- Larger batch size (64) to exploit available GPU memory
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import soundfile as sf

import src.config as CFG
from src.model import BirdModel, MelSpecTransform
from src.dataset import build_label_map
from src.augment import SpecAugment, additive_mixup, background_mix, gain_augment, time_shift


# ============================================================
# Config overrides for v2
# ============================================================
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 30
MIXUP_PROB = 0.5
BG_MIX_PROB = 0.3


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_macro_auc(targets, preds):
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0


# ============================================================
# Dataset with augmentation
# ============================================================
class AugmentedDataset(Dataset):
    """Unified dataset that handles focal, soundscape, and pseudo-labeled data with augmentation."""

    def __init__(self, entries, label_map, bg_entries=None, training=True):
        """
        entries: list of dicts with keys: path, offset_sec, duration, target (np array)
        bg_entries: list of background audio entries for mixing
        """
        self.entries = entries
        self.label_map = label_map
        self.bg_entries = bg_entries or []
        self.training = training
        self.num_samples = CFG.SR * CFG.DURATION

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio = self._load_audio(entry)
        target = entry["target"].copy()

        if self.training:
            # Gain augmentation
            audio = gain_augment(audio)

            # Time shift
            if random.random() < 0.5:
                audio = time_shift(audio)

            # Additive mixup with another random sample
            if random.random() < MIXUP_PROB:
                idx2 = random.randint(0, len(self.entries) - 1)
                audio2 = self._load_audio(self.entries[idx2])
                audio2 = gain_augment(audio2)
                target2 = self.entries[idx2]["target"]
                audio, target = additive_mixup(audio, target, audio2, target2)

            # Background mixing with soundscape
            elif self.bg_entries and random.random() < BG_MIX_PROB:
                bg_entry = random.choice(self.bg_entries)
                bg_audio = self._load_audio(bg_entry)
                bg_target = bg_entry["target"]
                audio, target = background_mix(audio, target, bg_audio, bg_target)

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
            audio = np.zeros(self.num_samples, dtype=np.float32)

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
# Build entries from different data sources
# ============================================================
def build_focal_entries(df, label_map):
    """Build entries from train.csv focal recordings."""
    entries = []
    for _, row in df.iterrows():
        path = os.path.join(CFG.TRAIN_AUDIO_DIR, row["filename"])
        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        label = str(row["primary_label"])
        if label in label_map:
            target[label_map[label]] = 1.0
        # Include secondary labels with lower weight
        if pd.notna(row.get("secondary_labels")) and row["secondary_labels"] != "[]":
            try:
                sec_labels = eval(row["secondary_labels"]) if isinstance(row["secondary_labels"], str) else []
                for sl in sec_labels:
                    if sl in label_map:
                        target[label_map[sl]] = 0.5
            except Exception:
                pass
        entries.append({"path": path, "offset_sec": 0.0, "duration": CFG.DURATION, "target": target,
                        "primary_label": label})
    return entries


def build_soundscape_entries(df, label_map):
    """Build entries from labeled train_soundscapes."""
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

        entries.append({"path": path, "offset_sec": offset_sec, "duration": CFG.DURATION, "target": target,
                        "primary_label": str(row["primary_label"]).split(";")[0]})
    return entries


def build_pseudo_entries(pseudo_df, label_map):
    """Build entries from pseudo-labeled soundscapes (soft labels)."""
    species_list = sorted(label_map.keys())
    entries = []
    for _, row in pseudo_df.iterrows():
        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        for k, sp in enumerate(species_list):
            if sp in row and row[sp] > 0:
                target[k] = float(row[sp])

        primary = species_list[target.argmax()]
        entries.append({"path": path, "offset_sec": offset_sec, "duration": CFG.DURATION, "target": target,
                        "primary_label": primary})
    return entries


# ============================================================
# Class-balanced sampler
# ============================================================
def build_sampler(entries, num_samples_per_epoch):
    """Inverse sqrt frequency weighting for class balance."""
    label_counts = {}
    for e in entries:
        pl = e["primary_label"]
        label_counts[pl] = label_counts.get(pl, 0) + 1

    weights = []
    for e in entries:
        pl = e["primary_label"]
        w = 1.0 / (label_counts[pl] ** 0.5)
        weights.append(w)

    return WeightedRandomSampler(weights, num_samples=num_samples_per_epoch, replacement=True)


# ============================================================
# Training loop
# ============================================================
def train_one_epoch(model, loader, optimizer, device, spec_aug):
    model.train()
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    total = len(loader)
    mel_transform = model.mel_spec

    for i, (audio, targets) in enumerate(loader):
        audio, targets = audio.to(device), targets.to(device)
        optimizer.zero_grad()

        # Compute mel and apply SpecAugment
        with torch.no_grad():
            mel = mel_transform(audio)  # (B, n_mels, T)
        mel = spec_aug(mel)
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, T)

        logits = model.backbone(mel)
        loss = criterion(logits, targets)

        loss.backward()
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
    return compute_macro_auc(all_targets, all_preds)


# ============================================================
# Main
# ============================================================
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_map = build_label_map()

    # --- Build all data entries ---
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    train_df["primary_label"] = train_df["primary_label"].astype(str)

    # Focal entries
    focal_entries = build_focal_entries(train_df, label_map)
    print(f"Focal entries: {len(focal_entries)}")

    # Soundscape entries
    sl_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    soundscape_entries = build_soundscape_entries(sl_df, label_map)
    print(f"Soundscape entries: {len(soundscape_entries)}")

    # Pseudo-label entries (if available)
    pseudo_path = os.path.join(CFG.DATA_DIR, "pseudo_labels.csv")
    pseudo_entries = []
    if os.path.exists(pseudo_path):
        pseudo_df = pd.read_csv(pseudo_path)
        pseudo_entries = build_pseudo_entries(pseudo_df, label_map)
        print(f"Pseudo-label entries: {len(pseudo_entries)}")
    else:
        print("No pseudo-labels found. Training without them.")

    # --- Split for validation (use 10% of focal as val) ---
    random.shuffle(focal_entries)
    val_size = len(focal_entries) // 10
    val_entries = focal_entries[:val_size]
    train_focal_entries = focal_entries[val_size:]

    # Combine all training entries
    all_train_entries = train_focal_entries + soundscape_entries * 5 + pseudo_entries
    print(f"Total training entries: {len(all_train_entries)}")

    # Background entries for mixing (soundscape segments)
    bg_entries = soundscape_entries + pseudo_entries

    # --- Datasets ---
    train_ds = AugmentedDataset(all_train_entries, label_map, bg_entries=bg_entries, training=True)
    val_ds = AugmentedDataset(val_entries, label_map, training=False)

    # Class-balanced sampler
    sampler = build_sampler(all_train_entries, num_samples_per_epoch=len(all_train_entries))

    num_workers = int(os.environ.get("NUM_WORKERS", 0))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # --- Model ---
    print("Loading model...", flush=True)
    model = BirdModel(pretrained=True).to(device)

    # Optionally load v1 checkpoint as starting point
    v1_ckpt = os.path.join(CFG.CHECKPOINT_DIR, "best.pt")
    if os.path.exists(v1_ckpt):
        try:
            state = torch.load(v1_ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            print("Loaded v1 checkpoint as initialization.")
        except Exception as e:
            print(f"Could not load v1 checkpoint: {e}. Training from pretrained ImageNet.")

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    spec_aug = SpecAugment().to(device)

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    best_auc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, spec_aug)
        print("  Validating...", flush=True)
        val_auc = validate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | lr={lr:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(CFG.CHECKPOINT_DIR, "best_v2.pt"))
            print(f"  -> Saved best model (AUC={best_auc:.4f})")

    print(f"\nBest validation AUC: {best_auc:.4f}")
    print("Done. Run: python -m src.export_onnx (update to load best_v2.pt)")


if __name__ == "__main__":
    main()
