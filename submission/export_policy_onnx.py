import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import os

from models.policy_model import PolicyModel

MODEL_PATH = "models/checkpoints/policy.pt"
ONNX_PATH = "submission/policy.onnx"


def export():

    if not os.path.exists(MODEL_PATH):
        raise ValueError("❌ policy.pt not found. Train first.")

    model = PolicyModel(seq_len=5, num_ops=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    os.makedirs("submission", exist_ok=True)

    dummy = torch.randn(1, 30, 30)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["grid"],
        output_names=["logits"],
        opset_version=11
    )

    print(f"✅ ONNX exported → {ONNX_PATH}")


if __name__ == "__main__":
    export()