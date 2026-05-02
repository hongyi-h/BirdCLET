import torch
import torch.nn as nn
import timm
import torchaudio.transforms as T
import src.config as CFG


class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=CFG.SR,
            n_fft=CFG.N_FFT,
            hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS,
            f_min=CFG.FMIN,
            f_max=CFG.FMAX,
            power=2.0,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        # x: (B, num_samples)
        x = self.mel(x)  # (B, n_mels, T)
        x = self.db(x)   # (B, n_mels, T)
        return x


class BirdBackbone(nn.Module):
    """Mel spectrogram (B, 1, n_mels, T) -> 234 logits. ONNX-exportable."""

    def __init__(self, num_classes=CFG.NUM_CLASSES, pretrained=True, model_name=None):
        super().__init__()
        self.backbone = timm.create_model(
            model_name or CFG.MODEL_NAME,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )
        self.feat_dim = self.backbone.num_features
        self.att_proj = nn.Linear(self.feat_dim, 1)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T) mel spectrogram
        x = self.backbone(x)  # (B, C, H, W)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, T', C)

        att = torch.softmax(self.att_proj(x), dim=1)  # (B, T', 1)
        x = (x * att).sum(dim=1)  # (B, C)

        return self.classifier(x)  # (B, num_classes)


class BirdModel(nn.Module):
    """Full training model: raw waveform -> mel -> backbone -> logits."""

    def __init__(self, num_classes=CFG.NUM_CLASSES, pretrained=True, model_name=None):
        super().__init__()
        self.mel_spec = MelSpecTransform()
        self.backbone = BirdBackbone(num_classes=num_classes, pretrained=pretrained, model_name=model_name)

    def forward(self, x, precomputed=False):
        if precomputed:
            # x: (B, 1, n_mels, T) mel spectrogram already computed
            return self.backbone(x)
        # x: (B, num_samples) raw waveform
        x = self.mel_spec(x)  # (B, n_mels, T)
        x = x.unsqueeze(1)    # (B, 1, n_mels, T)
        return self.backbone(x)
