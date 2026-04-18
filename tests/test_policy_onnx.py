import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from submission.inference_policy import run


def test():

    grid = np.array([
        [1, 2],
        [3, 4]
    ])

    output, seq = run(grid)

    print("Input:\n", grid)
    print("Predicted sequence:", seq)
    print("Output:\n", output)


if __name__ == "__main__":
    test()