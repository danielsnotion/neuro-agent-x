import numpy as np
from engine.color import apply_color_map
from engine.objects import find_objects, crop_object



def identity(grid):
    return grid.copy()


def rotate_90(grid):
    return np.rot90(grid, k=1)


def rotate_180(grid):
    return np.rot90(grid, k=2)


def flip_horizontal(grid):
    return np.flip(grid, axis=1)


def flip_vertical(grid):
    return np.flip(grid, axis=0)


def transpose(grid):
    return np.transpose(grid)

def color_map_wrapper(mapping):
    def transform(grid):
        return apply_color_map(grid, mapping)
    return transform




def extract_largest_object(grid):
    objects = find_objects(grid)

    if not objects:
        return grid

    largest = max(objects, key=lambda o: len(o["coords"]))
    return crop_object(grid, largest)


def extract_smallest_object(grid):
    objects = find_objects(grid)

    if not objects:
        return grid

    smallest = min(objects, key=lambda o: len(o["coords"]))
    return crop_object(grid, smallest)


# registry (important for later search)
TRANSFORMS = {
    "identity": identity,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "flip_h": flip_horizontal,
    "flip_v": flip_vertical,
    "transpose": transpose,
     "extract_largest": extract_largest_object,
    "extract_smallest": extract_smallest_object,
}

