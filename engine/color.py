import numpy as np
from itertools import permutations


def get_color_map(input_grid, target_grid):
    """
    Try to infer color mapping between input and target
    """

    input_colors = list(np.unique(input_grid))
    target_colors = list(np.unique(target_grid))

    if len(input_colors) > len(target_colors):
        return []

    mappings = []

    for perm in permutations(target_colors, len(input_colors)):
        mapping = dict(zip(input_colors, perm))
        mappings.append(mapping)

    return mappings


def apply_color_map(grid, mapping):
    new_grid = grid.copy()

    for src, dst in mapping.items():
        new_grid[grid == src] = dst

    return new_grid