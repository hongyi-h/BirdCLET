import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
import src.config as CFG


def load_audio(path, sr=CFG.SR, duration=CFG.DURATION):
    """Load audio file, random-crop to duration. Returns (num_samples,) float32."""
    num_samples = int(sr * duration)
    audio, file_sr = sf.read(path, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))
    elif len(audio) > num_samples:
        start = np.random.randint(0, len(audio) - num_samples)
        audio = audio[start : start + num_samples]

    return audio


def build_label_map():
    """Build species -> index mapping from taxonomy.csv (sorted order)."""
    tax = pd.read_csv(os.path.join(CFG.DATA_DIR, "taxonomy.csv"))
    labels = sorted(tax["primary_label"].astype(str).tolist())
    return {label: i for i, label in enumerate(labels)}


class FocalAudioDataset(Dataset):
    """Dataset for train_audio focal recordings (single-label per file)."""

    def __init__(self, df, label_map, audio_dir=CFG.TRAIN_AUDIO_DIR):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])
        audio = load_audio(path)

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        label = str(row["primary_label"])
        if label in self.label_map:
            target[self.label_map[label]] = 1.0

        return torch.from_numpy(audio), torch.from_numpy(target)


class SoundscapeDataset(Dataset):
    """Dataset for labeled train_soundscapes (multi-label per 5s segment)."""

    def __init__(self, df, label_map, audio_dir=CFG.TRAIN_SOUNDSCAPES_DIR):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])

        start_sec = self._parse_time(row["start"])
        info = sf.info(path)
        file_sr = info.samplerate
        start_frame = int(start_sec * file_sr)
        num_frames = int(CFG.DURATION * file_sr)

        audio, _ = sf.read(path, start=start_frame, stop=start_frame + num_frames, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        num_samples = int(CFG.SR * CFG.DURATION)
        if len(audio) < num_samples:
            audio = np.pad(audio, (0, num_samples - len(audio)))
        elif len(audio) > num_samples:
            audio = audio[:num_samples]

        target = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        for label in str(row["primary_label"]).split(";"):
            label = label.strip()
            if label in self.label_map:
                target[self.label_map[label]] = 1.0

        return torch.from_numpy(audio), torch.from_numpy(target)

    @staticmethod
    def _parse_time(t):
        """Parse HH:MM:SS to seconds."""
        parts = str(t).split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
