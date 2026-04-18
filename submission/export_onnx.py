import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import os

from models.ranker import ProgramRanker

CHECKPOINT_PATH = "models/checkpoints/ranker.pt"
ONNX_PATH = "submission/ranker.onnx"


def load_model():

    model = ProgramRanker(input_dim=12)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def export():

    os.makedirs("submission", exist_ok=True)

    model = load_model()

    dummy_input = torch.randn(1, 12)  # 🔥 must match feature size

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["features"],
        output_names=["score"],
        opset_version=11
    )

    print(f"✅ ONNX exported: {ONNX_PATH}")


if __name__ == "__main__":
    export()