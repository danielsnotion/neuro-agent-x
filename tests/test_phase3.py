import sys
from pathlib import Path
import numpy as np
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.search import beam_search


def run_test():

    input_grid = np.array([
        [1, 1],
        [2, 2]
    ])

    # target = color swapped
    target_grid = np.array([
        [2, 2],
        [1, 1]
    ])

    program, score = beam_search(
        input_grid,
        target_grid,
        beam_width=10,
        max_depth=3
    )

    print("\nBest Program:", program)
    print("Score:", score)


if __name__ == "__main__":
    run_test()