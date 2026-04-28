import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import src.config as CFG
from src.dataset import FocalAudioDataset, SoundscapeDataset, build_label_map
from src.model import BirdModel


def get_device_type():
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "musa") and torch.musa.is_available():
        torch.musa.manual_seed_all(seed)


def compute_macro_auc(targets, preds):
    """Macro-averaged ROC-AUC, skipping classes with no positives."""
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0


def train_one_epoch(model, loader, optimizer, scaler, device, amp_device):
    model.train()
    losses = []
    criterion = nn.BCEWithLogitsLoss()

    for audio, targets in loader:
        audio, targets = audio.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast(device_type=amp_device, enabled=amp_device != "cpu"):
            logits = model(audio)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, device, amp_device):
    model.eval()
    all_preds, all_targets = [], []

    for audio, targets in loader:
        audio = audio.to(device)
        with autocast(device_type=amp_device, enabled=amp_device != "cpu"):
            logits = model(audio)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    auc = compute_macro_auc(all_targets, all_preds)
    return auc


def main():
    set_seed()
    device_type = get_device_type()
    device = torch.device(device_type)
    print(f"Device: {device}")

    label_map = build_label_map()

    # --- Focal audio data ---
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    train_df["primary_label"] = train_df["primary_label"].astype(str)

    # Stratified split on primary_label.
    # Classes with only 1 sample cannot be split into train+val; keep them in
    # train only and exclude from the stratified fold to avoid sklearn warnings.
    label_counts = train_df["primary_label"].value_counts()
    rare_mask = train_df["primary_label"].map(label_counts) < 2
    rare_df = train_df[rare_mask]
    common_df = train_df[~rare_mask]

    if common_df.empty:
        raise ValueError("No class has ≥2 samples; cannot build a validation split.")

    min_common_count = int(common_df["primary_label"].value_counts().min())
    n_splits = min(5, min_common_count)
    if n_splits < 5:
        print(f"[split] Reduced n_splits to {n_splits} (min class count = {min_common_count})")
    if len(rare_df) > 0:
        print(f"[split] {len(rare_df)} singleton sample(s) kept train-only: "
              f"{sorted(rare_df['primary_label'].unique())}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = next(skf.split(common_df, common_df["primary_label"]))
    common_train_df = common_df.iloc[train_idx]
    common_val_df = common_df.iloc[val_idx]

    focal_train_df = pd.concat([common_train_df, rare_df], ignore_index=True)
    focal_train_df = focal_train_df.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
    focal_val_df = common_val_df.reset_index(drop=True)

    focal_train_ds = FocalAudioDataset(focal_train_df, label_map)
    focal_val_ds = FocalAudioDataset(focal_val_df, label_map)

    # --- Soundscape data ---
    sl_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
    soundscape_ds = SoundscapeDataset(sl_df, label_map)

    # Combine focal train + all soundscape labels for training
    # Soundscape data is small, so oversample it 5x
    train_ds = ConcatDataset([focal_train_ds] + [soundscape_ds] * 5)
    val_ds = focal_val_ds

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --- Model ---
    model = BirdModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
    scaler = GradScaler(device=device_type, enabled=device_type != "cpu")

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    best_auc = 0.0

    for epoch in range(CFG.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, device_type)
        val_auc = validate(model, val_loader, device, device_type)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{CFG.EPOCHS} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | lr={lr:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(CFG.CHECKPOINT_DIR, "best.pt"))
            print(f"  -> Saved best model (AUC={best_auc:.4f})")

    print(f"\nBest validation AUC: {best_auc:.4f}")
    print("Training complete. Run export_onnx.py to export for Kaggle submission.")


if __name__ == "__main__":
    main()
