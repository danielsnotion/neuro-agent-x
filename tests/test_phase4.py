import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from engine.objects import find_objects, crop_object


def run_test():

    grid = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,2,2]
    ])

    objects = find_objects(grid)

    print("Objects found:", len(objects))

    for i, obj in enumerate(objects):
        cropped = crop_object(grid, obj)
        print(f"\nObject {i} (color {obj['color']}):")
        print(cropped)


if __name__ == "__main__":
    run_test()