# training/build_value_dataset.py

import numpy as np
import json
import os
import random

SAVE_PATH = "models/value_dataset.json"


def random_transform(grid):

    ops = []

    if random.random() < 0.5:
        grid = np.rot90(grid)
        ops.append("rotate")

    if random.random() < 0.5:
        grid = np.flip(grid, axis=1)
        ops.append("flip")

    if random.random() < 0.3:
        grid = grid.T
        ops.append("transpose")

    return grid, ops


def run():

    data = []

    for i in range(2000):

        input_grid = np.random.randint(0, 3, (3, 3))

        # ✅ correct output
        correct_output, _ = random_transform(input_grid.copy())

        # ❌ wrong but realistic output
        wrong_output = correct_output.copy()

        # apply small mistake
        if random.random() < 0.5:
            wrong_output = np.rot90(wrong_output)
        else:
            wrong_output = np.flip(wrong_output, axis=1)

        # second wrong sample
        wrong_output2 = np.rot90(input_grid)  # partial transform

        data.append({
            "input": input_grid.tolist(),
            "output": wrong_output2.tolist(),
            "label": 0
        })

        data.append({
            "input": input_grid.tolist(),
            "output": correct_output.tolist(),
            "label": 1
        })

        data.append({
            "input": input_grid.tolist(),
            "output": wrong_output.tolist(),
            "label": 0
        })

    os.makedirs("models", exist_ok=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(data, f)

    print("✅ Value dataset created:", len(data))


if __name__ == "__main__":
    run()