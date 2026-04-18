import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import numpy as np

from submission.onnx_model import ARCModel


def apply_op(grid, op_id):

    if op_id == 0:
        return np.rot90(grid)
    elif op_id == 1:
        return np.rot90(grid, 2)
    elif op_id == 2:
        return np.flip(grid, axis=1)
    elif op_id == 3:
        return np.flip(grid, axis=0)
    else:
        return grid


def run_inference(grid):

    model = ARCModel()
    model.load_state_dict(torch.load("models/checkpoints/ranker.pt"))

    x = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    logits = model(x)
    op_id = torch.argmax(logits).item()

    return apply_op(grid, op_id)