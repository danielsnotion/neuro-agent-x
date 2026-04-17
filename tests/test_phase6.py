import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from engine.rules import apply_rule


def run_test():

    grid1 = np.array([
        [1, 2],
        [3, 4]
    ])

    grid2 = np.array([
        [1, 1],
        [1, 1]
    ])

    rule = {
        "condition": "multi_color",
        "true_program": ["rotate_180"],
        "false_program": ["identity"]
    }

    print("Grid 1:")
    print(apply_rule(grid1, rule))

    print("\nGrid 2:")
    print(apply_rule(grid2, rule))


if __name__ == "__main__":
    run_test()