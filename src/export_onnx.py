"""Export trained model's backbone to ONNX for CPU inference on Kaggle.

Usage:
    # Main models
    python -m src.export_onnx --checkpoint best_v4_v2s.pt --backbone v2s --output model_v2s.onnx
    python -m src.export_onnx --checkpoint best_v4_nfnet.pt --backbone nfnet --output model_nfnet.onnx
    python -m src.export_onnx --checkpoint best_v4_regnety.pt --backbone regnety --output model_regnety.onnx

    # Specialist model
    python -m src.export_onnx --checkpoint best_specialist.pt --backbone b0 --num_classes 72 --output specialist.onnx
"""
import os
import argparse
import torch
import numpy as np
import src.config as CFG
from src.model import BirdModel


def export(checkpoint, backbone_key, output_name, num_classes=CFG.NUM_CLASSES):
    device = torch.device("cpu")

    backbone_name = CFG.BACKBONE_REGISTRY.get(backbone_key, backbone_key)

    full_model = BirdModel(
        pretrained=False, model_name=backbone_name, num_classes=num_classes
    ).to(device)

    ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    full_model.load_state_dict(state, strict=False)
    full_model.eval()
    print(f"Loaded: {ckpt_path} (backbone={backbone_name}, classes={num_classes})")

    backbone = full_model.backbone
    backbone.eval()

    n_time_frames = CFG.SR * CFG.DURATION // CFG.HOP_LENGTH + 1
    dummy = torch.randn(1, 1, CFG.N_MELS, n_time_frames, device=device)

    out_path = os.path.join(CFG.CHECKPOINT_DIR, output_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    export_kwargs = dict(
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    try:
        torch.onnx.export(backbone, dummy, out_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(backbone, dummy, out_path, **export_kwargs)

    print(f"Exported: {out_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(out_path)
    pt_out = backbone(dummy).detach().numpy()
    onnx_out = sess.run(None, {"mel": dummy.numpy()})[0]

    diff = np.abs(pt_out - onnx_out).max()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Shape: {onnx_out.shape}, Size: {size_mb:.1f}MB, Max diff: {diff:.6f}")
    print("OK" if diff < 0.01 else f"WARNING: large diff ({diff:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True, help="Key from BACKBONE_REGISTRY or full timm name")
    parser.add_argument("--output", default="model.onnx", help="Output filename")
    parser.add_argument("--num_classes", type=int, default=CFG.NUM_CLASSES)
    args = parser.parse_args()
    export(args.checkpoint, args.backbone, args.output, args.num_classes)
