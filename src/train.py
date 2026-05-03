"""Training v4: DDP + precomputed mels + iterative Noisy Student.

Launch (8-GPU):
    torchrun --standalone --nproc_per_node=8 -m src.train --backbone v2s --epochs 40

Single GPU fallback:
    python -m src.train --backbone v2s --epochs 40
"""
import os
import datetime
import argparse
import random
import glob
import math
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Sampler
from sklearn.metrics import roc_auc_score
import soundfile as sf

import src.config as CFG
from src.model import BirdModel, MelSpecTransform, BirdBackbone
from src.dataset import build_label_map
from src.augment import SpecAugment, gain_augment, time_shift


# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="v2s", choices=list(CFG.BACKBONE_REGISTRY.keys()))
    parser.add_argument("--pseudo", action="store_true", help="Use pseudo-labels")
    parser.add_argument("--pseudo_round", type=int, default=0, help="Pseudo-label round (0=none)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--soundscape_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--train_all_soundscapes", action="store_true",
                        help="Include validation soundscape segments in training. Use only for final/teacher runs.")
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--precomputed_mixup_prob", type=float, default=0.3,
                        help="Batch-level mel mixup probability for --precomputed training")
    parser.add_argument("--bg_mix_prob", type=float, default=0.6)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed mel spectrograms")
    parser.add_argument("--save_tag", type=str, default="v4", help="Checkpoint name tag")
    return parser.parse_args()


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DDP helpers
# ============================================================
def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (rank, world_size, device)."""
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    visible_devices = torch.cuda.device_count()

    if visible_devices == 0:
        raise RuntimeError(
            "DDP was launched but torch.cuda.device_count() == 0. "
            "Run on a GPU node or use the single-process CPU/GPU command."
        )
    if local_world_size > visible_devices:
        raise RuntimeError(
            f"torchrun requested LOCAL_WORLD_SIZE={local_world_size}, but only "
            f"{visible_devices} CUDA device(s) are visible in this process. "
            f"Relaunch with --nproc_per_node={visible_devices}, or make more GPUs "
            "visible to the job, e.g. CUDA_VISIBLE_DEVICES=0,1,... or "
            "MTHREADS_VISIBLE_DEVICES=0,1,... on MetaX."
        )
    if local_rank >= visible_devices:
        raise RuntimeError(
            f"local_rank={local_rank} is out of range for {visible_devices} visible CUDA device(s). "
            "Check --nproc_per_node and GPU visibility."
        )

    # MetaX MUSA: try mccl, fall back to nccl, then gloo
    for backend in ["mccl", "nccl", "gloo"]:
        try:
            if dist.is_backend_available(backend):
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    timeout=datetime.timedelta(minutes=30),
                )
                break
        except Exception:
            continue
    else:
        dist.init_process_group(backend="gloo", init_method="env://",
                                timeout=datetime.timedelta(minutes=30))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"DDP initialized: world_size={world_size}, backend={dist.get_backend()}")

    return rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def ddp_barrier(device):
    if not dist.is_initialized():
        return
    try:
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


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
# Precomputed Dataset (fast: loads .npy)
# ============================================================
class PrecomputedDataset(Dataset):
    """Loads precomputed mel spectrograms from manifest records."""

    def __init__(self, records, training=True):
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        mel = np.load(record["mel_path"]).astype(np.float32)  # (n_mels, T)
        target = np.load(record["target_path"]).astype(np.float32)  # (NUM_CLASSES,)
        return torch.from_numpy(mel), torch.from_numpy(target)


# ============================================================
# Waveform Dataset (flexible: supports augmentation)
# ============================================================
class UnifiedDataset(Dataset):
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

            if random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self.entries) - 1)
                audio2 = self._load_audio(self.entries[idx2])
                audio2 = gain_augment(audio2, min_db=-8, max_db=8)
                target2 = self.entries[idx2]["target"]
                lam = np.random.beta(0.5, 0.5)
                audio = lam * audio + (1 - lam) * audio2
                target = np.maximum(target, target2)
            elif self.bg_pool and random.random() < self.bg_mix_prob:
                bg_entry = random.choice(self.bg_pool)
                bg_audio = self._load_audio(bg_entry)
                snr_db = np.random.uniform(-3, 15)
                amp = 10 ** (-snr_db / 20.0)
                audio = audio + amp * bg_audio
                if "target" in bg_entry:
                    target = np.maximum(target, bg_entry["target"])

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
def split_label_string(value):
    return [label.strip() for label in str(value).split(";") if label.strip()]


def site_from_filename(filename):
    parts = os.path.basename(str(filename)).split("_")
    return parts[3] if len(parts) > 3 else "unknown"


def entry_labels(entry):
    labels = entry.get("labels")
    if labels:
        return [str(label) for label in labels]
    return [str(entry["primary_label"])]


def build_focal_entries(df, label_map):
    entries = []
    for _, row in df.iterrows():
        path = os.path.join(CFG.TRAIN_AUDIO_DIR, row["filename"])
        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        label = str(row["primary_label"])
        labels = [label]
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
        entries.append({
            "path": path, "offset_sec": 0.0, "target": target,
            "primary_label": label, "labels": labels, "domain": "focal", "site": "focal",
        })
    return entries


def build_soundscape_entries(df, label_map):
    entries = []
    for _, row in df.iterrows():
        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        labels = split_label_string(row["primary_label"])
        for label in labels:
            if label in label_map:
                target[label_map[label]] = 1.0
        entries.append({
            "path": path, "offset_sec": offset_sec, "target": target,
            "primary_label": labels[0] if labels else "unknown",
            "labels": labels,
            "domain": "soundscape",
            "site": site_from_filename(row["filename"]),
        })
    return entries


def build_pseudo_entries(pseudo_df, label_map, confidence_threshold=0.6):
    species_list = sorted(label_map.keys())
    entries = []
    for _, row in pseudo_df.iterrows():
        if row.get("max_prob", 1.0) < confidence_threshold:
            continue
        path = os.path.join(CFG.TRAIN_SOUNDSCAPES_DIR, row["filename"])
        parts = str(row["start"]).split(":")
        offset_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        labels = []
        for k, sp in enumerate(species_list):
            if sp in row and row[sp] > 0.2:
                target[k] = float(min(row[sp], 1.0))
                labels.append(sp)
        if target.max() < 0.3:
            continue
        primary = species_list[target.argmax()]
        entries.append({
            "path": path, "offset_sec": offset_sec, "target": target,
            "primary_label": primary,
            "labels": labels or [primary],
            "domain": "pseudo",
            "site": site_from_filename(row["filename"]),
        })
    return entries


def build_background_pool(soundscape_dir):
    files = sorted(glob.glob(os.path.join(soundscape_dir, "*.ogg")))
    pool = []
    for f in files:
        try:
            info = sf.info(f)
            n_segments = int(info.duration // CFG.DURATION)
            for i in range(n_segments):
                pool.append({
                    "path": f, "offset_sec": i * CFG.DURATION,
                    "target": np.zeros(CFG.NUM_CLASSES, dtype=np.float32),
                })
        except Exception:
            continue
    return pool


def build_precomputed_paths(subset_dir):
    """Collect paired (mel, target) .npy paths from a precomputed directory."""
    mel_files = sorted(glob.glob(os.path.join(subset_dir, "*_mel.npy")))
    mel_paths, target_paths = [], []
    for mf in mel_files:
        tf = mf.replace("_mel.npy", "_target.npy")
        if os.path.exists(tf):
            mel_paths.append(mf)
            target_paths.append(tf)
    return mel_paths, target_paths


def resolve_precomputed_path(subset_dir, value):
    path = str(value)
    if os.path.isabs(path):
        return path
    return os.path.join(subset_dir, path)


def labels_from_target_path(target_path, species_list, threshold=0.2):
    target = np.load(target_path)
    return [species_list[i] for i, value in enumerate(target) if value > threshold]


def build_precomputed_records(subset_dir, domain, label_map):
    """Collect precomputed mel/target pairs with domain, labels, and site metadata."""
    species_list = sorted(label_map.keys())
    manifest_path = os.path.join(subset_dir, "manifest.csv")
    records = []

    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        for _, row in manifest.iterrows():
            mel_value = row.get("mel_path", "")
            target_value = row.get("target_path", "")
            if pd.isna(mel_value) or pd.isna(target_value) or not str(mel_value) or not str(target_value):
                stem = str(row.get("stem", row.get("name", "")))
                mel_value = f"{stem}_mel.npy"
                target_value = f"{stem}_target.npy"

            mel_path = resolve_precomputed_path(subset_dir, mel_value)
            target_path = resolve_precomputed_path(subset_dir, target_value)
            if not os.path.exists(mel_path) or not os.path.exists(target_path):
                continue

            labels_value = row.get("labels", row.get("primary_label", ""))
            if pd.isna(labels_value):
                labels_value = ""
            labels = split_label_string(labels_value)
            if not labels:
                labels = labels_from_target_path(target_path, species_list)
            primary = str(row.get("primary_label", labels[0] if labels else "unknown"))
            if primary == "nan":
                primary = labels[0] if labels else "unknown"
            if ";" in primary:
                primary = split_label_string(primary)[0]
            filename = row.get("filename", "")
            if pd.isna(filename):
                filename = ""
            site = row.get("site", site_from_filename(filename) if filename else domain)
            if pd.isna(site):
                site = site_from_filename(filename) if filename else domain
            record_domain = row.get("domain", domain)
            if pd.isna(record_domain):
                record_domain = domain

            records.append({
                "mel_path": mel_path,
                "target_path": target_path,
                "primary_label": primary,
                "labels": labels or [primary],
                "domain": str(record_domain),
                "site": str(site),
            })
        return records

    mel_paths, target_paths = build_precomputed_paths(subset_dir)
    if not mel_paths:
        return records

    train_df = None
    soundscape_df = None
    if domain == "focal":
        train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
        train_df["primary_label"] = train_df["primary_label"].astype(str)
    elif domain == "soundscape":
        soundscape_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))

    for mel_path, target_path in zip(mel_paths, target_paths):
        stem = os.path.basename(mel_path).replace("_mel.npy", "")
        labels = []
        primary = "unknown"
        site = domain

        try:
            if domain == "focal" and train_df is not None:
                source_idx = int(stem.split("_")[1])
                row = train_df.iloc[source_idx]
                primary = str(row["primary_label"])
                labels = [primary]
                site = "focal"
            elif domain == "soundscape" and soundscape_df is not None:
                source_idx = int(stem.split("_")[1])
                row = soundscape_df.iloc[source_idx]
                labels = split_label_string(row["primary_label"])
                primary = labels[0] if labels else "unknown"
                site = site_from_filename(row["filename"])
            else:
                labels = labels_from_target_path(target_path, species_list)
                primary = labels[0] if labels else "unknown"
        except Exception:
            labels = labels_from_target_path(target_path, species_list)
            primary = labels[0] if labels else "unknown"

        records.append({
            "mel_path": mel_path,
            "target_path": target_path,
            "primary_label": primary,
            "labels": labels or [primary],
            "domain": domain,
            "site": site,
        })

    return records


# ============================================================
# Validation split by site
# ============================================================
def count_entry_labels(entries):
    counts = {}
    for entry in entries:
        for label in entry_labels(entry):
            counts[label] = counts.get(label, 0) + 1
    return counts


def split_soundscape_by_site(entries, val_ratio=0.3, protect_train_labels=True):
    sites = {}
    for e in entries:
        site = e.get("site")
        if not site:
            fname = os.path.basename(e.get("path", ""))
            site = site_from_filename(fname)
        sites.setdefault(site, []).append(e)

    site_list = sorted(sites.keys())
    target_val_count = max(1, int(len(site_list) * val_ratio))
    target_val_segments = max(1, int(len(entries) * val_ratio))

    def site_label_counts(site_names):
        counts = {}
        for site_name in site_names:
            for label, count in count_entry_labels(sites[site_name]).items():
                counts[label] = counts.get(label, 0) + count
        return counts

    def valid_val_sites(site_names, total_counts):
        if not protect_train_labels:
            return True
        val_counts = site_label_counts(site_names)
        return all(
            total_counts.get(label, 0) - val_counts.get(label, 0) > 0
            for label in total_counts
        )

    if len(site_list) <= 20:
        total_counts = count_entry_labels(entries)
        best = None
        for r in range(1, len(site_list)):
            for combo in itertools.combinations(site_list, r):
                if not valid_val_sites(combo, total_counts):
                    continue
                val_segment_count = sum(len(sites[site]) for site in combo)
                val_label_count = len(site_label_counts(combo))
                score = (
                    -abs(val_segment_count - target_val_segments),
                    val_label_count,
                    -abs(len(combo) - target_val_count),
                    -len(combo),
                )
                if best is None or score > best[0]:
                    best = (score, set(combo))
        if best is not None:
            val_sites = best[1]
        else:
            val_sites = set()
    else:
        val_sites = set()

    if not val_sites and protect_train_labels:
        random.shuffle(site_list)
        remaining_counts = count_entry_labels(entries)
        val_sites = set()
        for site in site_list:
            if len(val_sites) >= target_val_count:
                break
            site_counts = count_entry_labels(sites[site])
            would_remove_label = any(
                remaining_counts.get(label, 0) - count <= 0
                for label, count in site_counts.items()
            )
            if would_remove_label:
                continue
            val_sites.add(site)
            for label, count in site_counts.items():
                remaining_counts[label] -= count
        if not val_sites:
            val_sites = set(site_list[:target_val_count])
    elif not val_sites:
        random.shuffle(site_list)
        val_sites = set(site_list[:target_val_count])

    train_entries, val_entries = [], []
    for site, se in sites.items():
        (val_entries if site in val_sites else train_entries).extend(se)
    return train_entries, val_entries, val_sites


# ============================================================
# Balanced sampler
# ============================================================
def build_sample_weights(entries):
    label_counts = {}
    for e in entries:
        for label in entry_labels(e):
            label_counts[label] = label_counts.get(label, 0) + 1

    focal_species = set()
    for e in entries:
        if e["domain"] == "focal":
            focal_species.update(entry_labels(e))

    weights = []
    for e in entries:
        labels = entry_labels(e)
        w = max(1.0 / (label_counts[label] ** 0.5) for label in labels if label in label_counts)
        if e["domain"] in ("soundscape", "pseudo"):
            w *= 3.0
        if any(label not in focal_species for label in labels):
            w *= 2.0
        weights.append(w)

    return weights


def build_balanced_sampler(entries, num_samples_per_epoch):
    weights = build_sample_weights(entries)
    return WeightedRandomSampler(weights, num_samples=num_samples_per_epoch, replacement=True)


class WeightedDistributedSampler(Sampler):
    """Replacement sampler that preserves class/domain weights under DDP."""

    def __init__(self, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, seed=CFG.SEED):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.num_samples = int(math.ceil(num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=generator
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# ============================================================
# Metrics
# ============================================================
def compute_classwise_macro_auc(targets, preds):
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


def summarize_entries(entries, label_map):
    domain_counts = {}
    label_set = set()
    for entry in entries:
        domain = entry.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        label_set.update(label for label in entry_labels(entry) if label in label_map)
    return domain_counts, label_set


def apply_precomputed_mixup(mel, targets, mixup_prob):
    if mixup_prob <= 0 or mel.size(0) < 2 or random.random() >= mixup_prob:
        return mel, targets
    perm = torch.randperm(mel.size(0), device=mel.device)
    lam = float(np.random.beta(0.5, 0.5))
    mel = lam * mel + (1.0 - lam) * mel[perm]
    targets = torch.maximum(targets, targets[perm])
    return mel, targets


# ============================================================
# Training loop
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device, spec_aug, scaler=None,
                    mel_transform=None, precomputed=False, precomputed_mixup_prob=0.0):
    model.train()
    losses = []
    total = len(loader)

    for i, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if precomputed:
                mel = data
                mel, targets = apply_precomputed_mixup(mel, targets, precomputed_mixup_prob)
            else:
                with torch.no_grad():
                    mel = mel_transform(data)

            mel = spec_aug(mel)
            mel = mel.unsqueeze(1)  # (B, 1, n_mels, T)

            # Forward through DDP wrapper to preserve gradient sync
            logits = model(mel, precomputed=True)
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
        if i % 100 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"  [train] batch {i}/{total} | loss={loss.item():.4f}", flush=True)

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, device, precomputed=False):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    mel_transform = raw_model.mel_spec
    all_preds, all_targets = [], []

    for data, targets in loader:
        data = data.to(device)
        if precomputed:
            mel = data.unsqueeze(1)
            logits = raw_model(mel, precomputed=True)
        else:
            logits = raw_model(data)
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

    rank, world_size, device = setup_ddp()

    if is_main(rank):
        print(f"Device: {device} | World size: {world_size}")
        print(f"Config: backbone={args.backbone}, epochs={args.epochs}, "
              f"bs={args.batch_size}x{world_size}={args.batch_size*world_size}, "
              f"lr={args.lr}, pseudo={args.pseudo}, precomputed={args.precomputed}, "
              f"val_ratio={args.val_ratio}, train_all_soundscapes={args.train_all_soundscapes}")

    label_map = build_label_map()
    species_list = sorted(label_map.keys())
    model_name = CFG.BACKBONE_REGISTRY[args.backbone]

    # --- Build datasets ---
    if args.precomputed:
        focal_records = build_precomputed_records(
            os.path.join(CFG.PRECOMPUTED_DIR, "focal"), "focal", label_map)
        soundscape_records = build_precomputed_records(
            os.path.join(CFG.PRECOMPUTED_DIR, "soundscape_labeled"), "soundscape", label_map)

        if is_main(rank):
            print(f"Precomputed focal: {len(focal_records)}, soundscape: {len(soundscape_records)}")

        sc_train, sc_val, val_sites = split_soundscape_by_site(soundscape_records, val_ratio=args.val_ratio)
        train_sc_records = soundscape_records if args.train_all_soundscapes else sc_train

        # Oversample soundscape
        sc_oversample = max(1, len(focal_records) // (3 * max(len(train_sc_records), 1)))
        all_train = focal_records + train_sc_records * sc_oversample

        if args.pseudo:
            pseudo_dir = os.path.join(CFG.PRECOMPUTED_DIR, "pseudo")
            if os.path.isdir(pseudo_dir):
                pseudo_records = build_precomputed_records(pseudo_dir, "pseudo", label_map)
                all_train += pseudo_records
                if is_main(rank):
                    print(f"Pseudo precomputed: {len(pseudo_records)}")
            elif is_main(rank):
                print("WARNING: --pseudo set but data/precomputed/pseudo is missing. "
                      "Run: python -m src.preprocess --mode pseudo")

        train_ds = PrecomputedDataset(all_train, training=True)
        val_ds = PrecomputedDataset(sc_val, training=False)

        if is_main(rank):
            train_domains, train_labels = summarize_entries(all_train, label_map)
            _, val_labels = summarize_entries(sc_val, label_map)
            print(f"SC train: {len(sc_train)}, SC val: {len(sc_val)} (sites: {val_sites})")
            if args.train_all_soundscapes:
                print("WARNING: --train_all_soundscapes is set; validation is leaky and for monitoring only.")
            print(f"Total train: {len(train_ds)}, val: {len(val_ds)}")
            print(f"Train domains: {train_domains}")
            print(f"Train species: {len(train_labels)}, val evaluable species upper bound: {len(val_labels)}")
            missing_train = set(label_map) - train_labels
            if missing_train:
                print(f"WARNING: {len(missing_train)} taxonomy species have no training positives in this split.")

    else:
        # Waveform-based pipeline (original)
        train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
        train_df["primary_label"] = train_df["primary_label"].astype(str)
        focal_entries = build_focal_entries(train_df, label_map)

        sl_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv"))
        soundscape_entries = build_soundscape_entries(sl_df, label_map)

        pseudo_entries = []
        if args.pseudo:
            pseudo_name = f"pseudo_labels_r{args.pseudo_round}.csv" if args.pseudo_round > 0 else "pseudo_labels.csv"
            pseudo_path = os.path.join(CFG.DATA_DIR, pseudo_name)
            if not os.path.exists(pseudo_path) and args.pseudo_round > 0:
                pseudo_path = os.path.join(CFG.DATA_DIR, "pseudo_labels.csv")
            if os.path.exists(pseudo_path):
                pseudo_df = pd.read_csv(pseudo_path)
                pseudo_entries = build_pseudo_entries(pseudo_df, label_map)
                if is_main(rank):
                    print(f"Pseudo-label entries: {len(pseudo_entries)} ({pseudo_path})")
            elif is_main(rank):
                print(f"WARNING: --pseudo set but pseudo label CSV is missing: {pseudo_path}")

        sc_train, sc_val, val_sites = split_soundscape_by_site(soundscape_entries, val_ratio=args.val_ratio)
        if is_main(rank):
            print(f"Focal: {len(focal_entries)}, SC train: {len(sc_train)}, "
                  f"SC val: {len(sc_val)} (sites: {val_sites})")

        bg_pool = build_background_pool(CFG.TRAIN_SOUNDSCAPES_DIR)

        train_sc_entries = soundscape_entries if args.train_all_soundscapes else sc_train
        sc_oversample = max(1, len(focal_entries) // (3 * max(len(train_sc_entries), 1)))
        all_train = focal_entries + train_sc_entries * sc_oversample + pseudo_entries

        train_ds = UnifiedDataset(all_train, label_map, bg_pool=bg_pool, training=True,
                                  mixup_prob=args.mixup_prob, bg_mix_prob=args.bg_mix_prob)
        val_ds = UnifiedDataset(sc_val, label_map, training=False)

        if is_main(rank):
            train_domains, train_labels = summarize_entries(all_train, label_map)
            _, val_labels = summarize_entries(sc_val, label_map)
            print(f"Total train: {len(train_ds)}, val: {len(val_ds)}")
            if args.train_all_soundscapes:
                print("WARNING: --train_all_soundscapes is set; validation is leaky and for monitoring only.")
            print(f"Train domains: {train_domains}")
            print(f"Train species: {len(train_labels)}, val evaluable species upper bound: {len(val_labels)}")
            missing_train = set(label_map) - train_labels
            if missing_train:
                print(f"WARNING: {len(missing_train)} taxonomy species have no training positives in this split.")

    # --- DataLoaders ---
    mp_ctx = mp.get_context("spawn")

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Check audio/precomputed paths on the server.")
    if len(val_ds) == 0:
        raise RuntimeError("Soundscape validation dataset is empty. Check train_soundscapes_labels/precomputed manifests.")

    train_weights = build_sample_weights(all_train)
    if world_size > 1:
        train_sampler = WeightedDistributedSampler(
            train_weights, num_samples=len(all_train),
            num_replicas=world_size, rank=rank, seed=CFG.SEED,
        )
    else:
        train_sampler = WeightedRandomSampler(
            train_weights, num_samples=len(all_train), replacement=True
        )
    val_sampler = None

    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_common.update({
            "multiprocessing_context": mp_ctx,
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=False, drop_last=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        **loader_common,
    )

    # --- Model ---
    if is_main(rank):
        print(f"Loading model: {model_name}")
    model = BirdModel(pretrained=True, model_name=model_name).to(device)

    if args.resume:
        if os.path.isabs(args.resume) or os.path.exists(args.resume):
            ckpt_path = args.resume
        else:
            ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, args.resume)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        if is_main(rank):
            print(f"Resumed from {ckpt_path}")

    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main(rank):
        print(f"Model params: {param_count:.1f}M")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = FocalBCELoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    spec_aug = SpecAugment(freq_mask_param=24, time_mask_param=50,
                           n_freq_masks=2, n_time_masks=2).to(device)

    raw_model = model.module if hasattr(model, "module") else model
    mel_transform = raw_model.mel_spec

    scaler = None  # Disabled for MUSA compatibility

    # --- Training loop ---
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    best_auc = 0.0

    for epoch in range(args.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        if is_main(rank):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, spec_aug,
            scaler=scaler, mel_transform=mel_transform, precomputed=args.precomputed,
            precomputed_mixup_prob=args.precomputed_mixup_prob if args.precomputed else 0.0)

        # Rank 0 validates on the full soundscape validation set. Other ranks wait
        # before the next epoch so DDP does not advance with a different schedule.
        if is_main(rank):
            print("  Validating on soundscape data...", flush=True)
            val_auc, per_class = validate(model, val_loader, device, precomputed=args.precomputed)

            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} | "
                  f"val_auc(soundscape)={val_auc:.4f} | lr={lr:.6f}")

            if val_auc > best_auc:
                best_auc = val_auc
                save_path = os.path.join(CFG.CHECKPOINT_DIR, f"best_{args.save_tag}.pt")
                state_dict = raw_model.state_dict()
                torch.save(state_dict, save_path)
                print(f"  -> Saved best model (AUC={best_auc:.4f})")

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
        ddp_barrier(device)
        scheduler.step()

    if is_main(rank):
        final_path = os.path.join(CFG.CHECKPOINT_DIR, f"last_{args.save_tag}.pt")
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), final_path)
        print(f"Saved final model: {final_path}")
        print(f"\nBest soundscape validation AUC: {best_auc:.4f}")
        export_ckpt = f"last_{args.save_tag}.pt" if args.train_all_soundscapes else f"best_{args.save_tag}.pt"
        if args.train_all_soundscapes:
            print("Validation was leaky; export the final checkpoint, not the best-validation checkpoint.")
        print(f"Export with: python -m src.export_onnx --checkpoint {export_ckpt} "
              f"--backbone {args.backbone}")

    cleanup_ddp()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
