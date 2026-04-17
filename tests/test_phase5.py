import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from engine.search_multi import beam_search_multi


def run_test():

    examples = [
        (
            np.array([[1, 1], [2, 2]]),
            np.array([[2, 2], [1, 1]])
        ),
        (
            np.array([[3, 3], [4, 4]]),
            np.array([[4, 4], [3, 3]])
        )
    ]

    program, score = beam_search_multi(
        examples,
        beam_width=5,
        max_depth=3
    )

    print("\nBest Program:", program)
    print("Score:", score)


if __name__ == "__main__":
    run_test()