"""Export trained model's backbone to ONNX for CPU inference on Kaggle.

The ONNX model takes mel spectrogram input (B, 1, n_mels, T), not raw audio.
Inference notebook computes mel via librosa before feeding into ONNX.
"""
import torch
import src.config as CFG
from src.model import BirdModel
import os


def export(checkpoint=None, backbone_name=None):
    device = torch.device("cpu")

    # Load full model (mel_spec + backbone)
    full_model = BirdModel(pretrained=False, model_name=backbone_name).to(device)

    # Checkpoint priority: arg > best_v3 > best_v2 > best
    if checkpoint:
        ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, checkpoint)
    else:
        for name in ["best_v3.pt", "best_v2.pt", "best.pt"]:
            ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, name)
            if os.path.exists(ckpt_path):
                break

    full_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    full_model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Extract just the backbone (takes mel input, ONNX-exportable)
    backbone = full_model.backbone
    backbone.eval()

    # Dummy mel spectrogram input: (1, 1, n_mels, T)
    n_time_frames = CFG.SR * CFG.DURATION // CFG.HOP_LENGTH + 1
    dummy = torch.randn(1, 1, CFG.N_MELS, n_time_frames, device=device)

    os.makedirs(os.path.dirname(CFG.ONNX_PATH) or ".", exist_ok=True)

    export_kwargs = dict(
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    # Use legacy TorchScript exporter if available (avoids dynamo numeric drift)
    try:
        torch.onnx.export(backbone, dummy, CFG.ONNX_PATH, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(backbone, dummy, CFG.ONNX_PATH, **export_kwargs)

    print(f"Exported ONNX model to {CFG.ONNX_PATH}")

    # Verify
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(CFG.ONNX_PATH)
    pt_out = backbone(dummy).detach().numpy()
    onnx_out = sess.run(None, {"mel": dummy.numpy()})[0]

    diff = np.abs(pt_out - onnx_out).max()
    print(f"ONNX output shape: {onnx_out.shape}")
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")
    print("Verification passed." if diff < 0.01 else f"WARNING: large diff ({diff:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None,
                        help="e.g. tf_efficientnetv2_s.in21k_ft_in1k")
    args = parser.parse_args()
    export(checkpoint=args.checkpoint, backbone_name=args.backbone)
