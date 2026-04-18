import numpy as np


def extract_features(grid):

    unique_colors = np.unique(grid)

    return {
        "height": grid.shape[0],
        "width": grid.shape[1],
        "num_colors": len(unique_colors),
        "is_square": int(grid.shape[0] == grid.shape[1]),
        "color_variance": np.var(grid),
    }


def extract_object_features(grid):
    # simple proxy (can upgrade later)
    return {
        "non_zero_ratio": np.mean(grid != 0),
    }