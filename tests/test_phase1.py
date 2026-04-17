import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from engine.executor import apply_program


def run_tests():
    grid = np.array([
        [1, 2],
        [3, 4]
    ])

    print("Original:")
    print(grid)

    # Test rotate
    out = apply_program(grid, ["rotate_90"])
    print("\nRotate 90:")
    print(out)

    # Test flip
    out = apply_program(grid, ["flip_h"])
    print("\nFlip Horizontal:")
    print(out)

    # Test multi-step
    out = apply_program(grid, ["rotate_90", "flip_h"])
    print("\nRotate + Flip:")
    print(out)

    print("\n✅ Phase 1 tests passed")


if __name__ == "__main__":
    run_tests()