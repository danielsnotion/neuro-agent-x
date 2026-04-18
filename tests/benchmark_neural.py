import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import time

from engine.search import beam_search


def run_benchmark():

    tasks = [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[2, 4], [1, 3]])
        ),
        (
            np.array([[1, 1], [2, 2]]),
            np.array([[2, 2], [1, 1]])
        ),
        (
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        )
    ]

    print("\n===== BASELINE (NO NEURAL) =====")
    start = time.time()

    for i, (inp, tgt) in enumerate(tasks):
        prog, score = beam_search(inp, tgt, use_neural=True, beam_width=20, max_depth=5)
        print(f"Task {i}: Score={score}, Program={prog}")

    baseline_time = time.time() - start

    print(f"\nBaseline Time: {baseline_time:.4f}s")

    print("\n===== NEURAL GUIDED =====")
    start = time.time()

    for i, (inp, tgt) in enumerate(tasks):
        prog, score = beam_search(inp, tgt, use_neural=True)
        print(f"Task {i}: Score={score}, Program={prog}")

    neural_time = time.time() - start

    print(f"\nNeural Time: {neural_time:.4f}s")

    print("\n===== RESULT =====")

    print(f"Speedup: {baseline_time / neural_time:.2f}x")


if __name__ == "__main__":
    run_benchmark()