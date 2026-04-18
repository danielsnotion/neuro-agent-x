import torch
import numpy as np
import os

from models.ranker import ProgramRanker
from engine.features import extract_features, extract_object_features

MODEL = None


def load_model():

    global MODEL

    if MODEL is not None:
        return MODEL

    model = ProgramRanker(input_dim=12)

    path = "models/checkpoints/ranker.pt"

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            print("✅ Loaded trained ranker model")

        except Exception as e:
            print("⚠️ Model load failed (architecture changed). Using fresh model.")
            print(f"Reason: {e}")
    else:
        print("⚠️ No trained model found")

    model.eval()
    MODEL = model

    return MODEL


# 🔥 PROGRAM FEATURES
def encode_program(program):

    return [
        len(program),
        sum("rotate" in str(p) for p in program),
        sum("flip" in str(p) for p in program),
        sum("color_map" in str(p) for p in program),
        sum("extract" in str(p) for p in program),
        len(set(str(p) for p in program)),  # diversity
    ]


# 🔥 GRID FEATURES
def encode_grid(grid):

    f = extract_features(grid)
    obj = extract_object_features(grid)

    return [
        f["height"],
        f["width"],
        f["num_colors"],
        f["is_square"],
        f["color_variance"],
        obj["non_zero_ratio"],
    ]


def predict_score(program, grid=None):

    model = load_model()

    prog_vec = encode_program(program)

    if grid is not None:
        grid_vec = encode_grid(grid)
    else:
        grid_vec = [0] * 6  # fallback

    len(prog_vec + grid_vec) == 12
    x = np.array(prog_vec + grid_vec, dtype=np.float32)

    x = torch.tensor(x).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    return score