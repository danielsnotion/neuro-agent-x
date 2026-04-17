import numpy as np


def detect_symmetry(grid):
    if np.array_equal(grid, np.rot90(grid)):
        return "rotational"
    if np.array_equal(grid, np.flip(grid, axis=1)):
        return "horizontal"
    return "none"


def detect_color_pattern(grid):
    unique = np.unique(grid)

    if len(unique) == 1:
        return "single_color"
    elif len(unique) <= 3:
        return "few_colors"
    else:
        return "many_colors"