import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from engine.search import beam_search


def run_test():

    # Task 1
    input1 = np.array([[1, 2], [3, 4]])
    target1 = np.array([[2, 4], [1, 3]])

    beam_search(input1, target1)

    # Task 2 (similar)
    input2 = np.array([[5, 6], [7, 8]])
    target2 = np.array([[6, 8], [5, 7]])

    program, score = beam_search(input2, target2)

    print("\nReused Program:", program)
    print("Score:", score)


if __name__ == "__main__":
    run_test()