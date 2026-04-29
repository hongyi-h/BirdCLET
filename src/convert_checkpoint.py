"""Convert old-format checkpoint (flat BirdModel) to new format (BirdModel.backbone)."""
import torch
import os
import src.config as CFG


def convert():
    old_path = os.path.join(CFG.CHECKPOINT_DIR, "best.pt")
    new_path = os.path.join(CFG.CHECKPOINT_DIR, "best.pt")

    state = torch.load(old_path, map_location="cpu")

    # Check if already in new format
    if any(k.startswith("backbone.backbone.") for k in state.keys()):
        print("Checkpoint already in new format. Nothing to do.")
        return

    new_state = {}
    for k, v in state.items():
        if k.startswith("mel_spec."):
            new_state[k] = v
        elif k.startswith("backbone."):
            # old: backbone.X -> new: backbone.backbone.X
            new_state[f"backbone.{k}"] = v
        elif k.startswith("att_proj.") or k.startswith("classifier."):
            # old: att_proj.X -> new: backbone.att_proj.X
            new_state[f"backbone.{k}"] = v
        else:
            new_state[k] = v

    torch.save(new_state, new_path)
    print(f"Converted checkpoint saved to {new_path}")
    print(f"  Old keys: {len(state)}, New keys: {len(new_state)}")

    # Verify it loads
    from src.model import BirdModel
    model = BirdModel(pretrained=False)
    model.load_state_dict(new_state)
    print("Verification: loads successfully into new BirdModel.")


if __name__ == "__main__":
    convert()
