import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from engine.search import beam_search


def run_test():

    input_grid = np.array([
        [1, 2],
        [3, 4]
    ])

    target_grid = np.array([
        [2, 4],
        [1, 3]
    ])

    program, score = beam_search(input_grid, target_grid)

    print("Program:", program)
    print("Score:", score)


if __name__ == "__main__":
    run_test()