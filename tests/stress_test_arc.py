import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time

from engine.search import beam_search


def run_stress_test():

    tasks = []

    # 🔥 Task 1: rotation + color map
    tasks.append((
        np.array([
            [1, 0, 1],
            [0, 2, 0],
            [1, 0, 1]
        ]),
        np.array([
            [2, 1, 2],
            [1, 0, 1],
            [2, 1, 2]
        ])
    ))

    # 🔥 Task 2: object extraction + transform
    tasks.append((
        np.array([
            [0,0,0,0],
            [0,3,3,0],
            [0,3,3,0],
            [0,0,0,0]
        ]),
        np.array([
            [3,3],
            [3,3]
        ])
    ))

    # 🔥 Task 3: multi-step transformation
    tasks.append((
        np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]),
        np.array([
            [3,6,9],
            [2,5,8],
            [1,4,7]
        ])  # rotate_90 + flip
    ))

    # 🔥 Task 4: pattern inversion
    tasks.append((
        np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ]),
        np.array([
            [1,0,1],
            [0,0,0],
            [1,0,1]
        ])
    ))

    # 🔥 Task 5: harder (requires deeper reasoning)
    tasks.append((
        np.array([
            [1,1,0,0],
            [1,1,0,0],
            [0,0,2,2],
            [0,0,2,2]
        ]),
        np.array([
            [2,2,1,1],
            [2,2,1,1],
            [1,1,2,2],
            [1,1,2,2]
        ])
    ))

    print("\n===== STRESS TEST: BASELINE =====")

    start = time.time()

    for i, (inp, tgt) in enumerate(tasks):
        prog, score = beam_search(
            inp, tgt,
            beam_width=20,
            max_depth=5,
            use_neural=False
        )
        print(f"[Baseline] Task {i}: Score={score}, Program={prog}")

    baseline_time = time.time() - start

    print(f"\nBaseline Time: {baseline_time:.4f}s")

    print("\n===== STRESS TEST: NEURAL =====")

    start = time.time()

    for i, (inp, tgt) in enumerate(tasks):
        prog, score = beam_search(
            inp, tgt,
            beam_width=20,
            max_depth=5,
            use_neural=True
        )
        print(f"[Neural] Task {i}: Score={score}, Program={prog}")

    neural_time = time.time() - start

    print(f"\nNeural Time: {neural_time:.4f}s")

    print("\n===== RESULT =====")

    print(f"Speedup: {baseline_time / neural_time:.2f}x")


if __name__ == "__main__":
    run_stress_test()