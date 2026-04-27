"""Export trained PyTorch model to ONNX for CPU inference on Kaggle."""
import torch
import src.config as CFG
from src.model import BirdModel
import os


def export():
    device = torch.device("cpu")
    model = BirdModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(CFG.CHECKPOINT_DIR, "best.pt"), map_location=device))
    model.eval()

    dummy = torch.randn(1, CFG.SR * CFG.DURATION, device=device)

    os.makedirs(os.path.dirname(CFG.ONNX_PATH) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        CFG.ONNX_PATH,
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={"audio": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {CFG.ONNX_PATH}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(CFG.ONNX_PATH)
    out = sess.run(None, {"audio": dummy.numpy()})
    print(f"ONNX output shape: {out[0].shape}")
    print("Verification passed.")


if __name__ == "__main__":
    export()
