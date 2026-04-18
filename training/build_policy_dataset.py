import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import os
import random
from engine.search import beam_search

SAVE_PATH = "models/policy_dataset.json"

OP_MAP = {
    "rotate_90": 0,
    "rotate_180": 1,
    "flip": 2,
    "color_map": 3,
    "extract": 4,
    "transpose": 5,
    "STOP": 6
}

NUM_OPS = 7
PAD_TOKEN = -1
STOP_TOKEN = 6
MAX_SEQ_LEN = 5


def encode_program(program):

    seq = []

    for step in program:
        step_str = str(step)

        matched = False
        for k in OP_MAP:
            if k in step_str:
                seq.append(OP_MAP[k])
                matched = True
                break

        if not matched:
            seq.append(0)

    # 🔥 ADD STOP TOKEN
    if len(seq) < MAX_SEQ_LEN:
        seq.append(STOP_TOKEN)

    # trim
    seq = seq[:MAX_SEQ_LEN]

    # pad
    seq += [PAD_TOKEN] * (MAX_SEQ_LEN - len(seq))

    return seq


def run():

    print("🚀 Building dataset...")

    data = []

    for i in range(500):

        input_grid = np.random.randint(0, 3, (3, 3))

        target_grid = input_grid.copy()

        # ✅ INSERT HERE
        program = []
        num_steps = random.choice([1, 2, 3])

        for _ in range(num_steps):
            op = random.choice(["rotate_90", "flip", "transpose"])

            if op == "rotate_90":
                target_grid = np.rot90(target_grid)
            elif op == "flip":
                target_grid = np.flip(target_grid, axis=1)
            elif op == "transpose":
                target_grid = target_grid.T

            program.append(op)

        # keep rest same
        if np.array_equal(input_grid, target_grid):
            continue

        seq = encode_program(program)

        if seq[0] == STOP_TOKEN:
            continue

        data.append({
            "grid": input_grid.tolist(),
            "sequence": seq
        })

        print(f"Sample {i}: Program={program}, Seq={seq}")

    stop_count = sum(1 for d in data if d["sequence"][0] == STOP_TOKEN)
    print("STOP at step 0:", stop_count / len(data))
    # save dataset
    os.makedirs("models", exist_ok=True)

    with open("models/policy_dataset.json", "w") as f:
        json.dump(data, f)

    print("✅ Dataset saved")
    print("Total samples:", len(data))


if __name__ == "__main__":
    run()