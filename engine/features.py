import numpy as np


def extract_features(grid):
    return {
        "shape": grid.shape,
        "num_colors": len(np.unique(grid)),
        "is_square": grid.shape[0] == grid.shape[1],
    }


def extract_task_features(examples):
    feats = []

    for inp, tgt in examples:
        f = extract_features(inp)
        f["target_shape"] = tgt.shape
        feats.append(f)

    return feats