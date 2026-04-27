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

import src.config as CFG
from src.dataset import FocalAudioDataset, SoundscapeDataset, build_label_map
from src.model import BirdModel


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    losses = []
    criterion = nn.BCEWithLogitsLoss()

    for audio, targets in loader:
        audio, targets = audio.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):
            logits = model(audio)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for audio, targets in loader:
        audio = audio.to(device)
        with autocast("cuda"):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_map = build_label_map()

    # --- Focal audio data ---
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    train_df["primary_label"] = train_df["primary_label"].astype(str)

    # Stratified split on primary_label
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = next(skf.split(train_df, train_df["primary_label"]))
    focal_train_df = train_df.iloc[train_idx]
    focal_val_df = train_df.iloc[val_idx]

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
    scaler = GradScaler()

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    best_auc = 0.0

    for epoch in range(CFG.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_auc = validate(model, val_loader, device)
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
