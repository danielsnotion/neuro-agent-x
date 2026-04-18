import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import onnxruntime as ort

from models.ranker import ProgramRanker

CHECKPOINT_PATH = "models/checkpoints/ranker.pt"
ONNX_PATH = "submission/ranker.onnx"


def load_pytorch_model():
    model = ProgramRanker(input_dim=12)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def run_test():

    # 🔹 load models
    torch_model = load_pytorch_model()
    ort_session = ort.InferenceSession(ONNX_PATH)

    # 🔹 test multiple inputs
    for i in range(5):

        # random input
        x = np.random.randn(1, 12).astype(np.float32)

        # PyTorch output
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(x)).numpy()

        # ONNX output
        ort_out = ort_session.run(None, {"features": x})[0]

        # 🔹 compare
        diff = np.abs(torch_out - ort_out)
        max_diff = diff.max()

        print(f"\nTest {i}")
        print("PyTorch:", torch_out)
        print("ONNX   :", ort_out)
        print("Max diff:", max_diff)

        # 🔥 assertion
        if max_diff > 1e-5:
            print("❌ Mismatch detected!")
        else:
            print("✅ Match")


if __name__ == "__main__":
    run_test()