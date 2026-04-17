import torch
import numpy as np

from models.ranker import ProgramRanker

model = ProgramRanker()


def encode_program(program):
    # simple encoding: length + counts
    return np.array([
        len(program),
        sum("rotate" in str(p) for p in program),
        sum("flip" in str(p) for p in program),
        sum("color_map" in str(p) for p in program),
    ], dtype=np.float32)


def predict_score(program):
    x = encode_program(program)
    x = torch.tensor(x).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    return score