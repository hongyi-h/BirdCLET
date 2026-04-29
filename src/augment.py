import numpy as np
import torch
import torch.nn as nn
import src.config as CFG


class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking on mel spectrogram."""

    def __init__(self, freq_mask_param=20, time_mask_param=40, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        # x: (B, n_mels, T)
        if not self.training:
            return x
        cloned = x.clone()
        _, n_mels, n_time = cloned.shape

        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            cloned[:, f0:f0 + f, :] = 0.0

        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            cloned[:, :, t0:t0 + t] = 0.0

        return cloned


def additive_mixup(audio1, target1, audio2, target2):
    """Additive mixup: sum waveforms, union labels. Simulates overlapping calls."""
    amp = 10 ** np.random.uniform(-0.5, 0.5)
    mixed = audio1 + amp * audio2
    target = np.maximum(target1, target2)
    return mixed, target


def background_mix(focal_audio, focal_target, bg_audio, bg_target, snr_range=(-5, 10)):
    """Mix focal recording onto soundscape background at random SNR."""
    snr_db = np.random.uniform(*snr_range)
    amp = 10 ** (-snr_db / 20.0)
    mixed = focal_audio + amp * bg_audio
    target = np.maximum(focal_target, bg_target)
    return mixed, target


def gain_augment(audio, min_db=-6, max_db=6):
    """Random gain augmentation."""
    gain_db = np.random.uniform(min_db, max_db)
    return audio * (10 ** (gain_db / 20.0))


def time_shift(audio, max_shift_sec=1.0, sr=CFG.SR):
    """Random circular time shift."""
    max_shift = int(max_shift_sec * sr)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)
