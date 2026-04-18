import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# training/train_policy.py

import torch
import json
import numpy as np
import os

from models.policy_model import PolicyModel

DATA_PATH = "models/policy_dataset.json"
STOP_TOKEN = 6

# 🔥 MUST match dataset builder
OP_MAP = {
    "rotate_90": 0,
    "rotate_180": 1,
    "flip": 2,
    "color_map": 3,
    "extract": 4,
    "transpose": 5
}

PAD_TOKEN = -1


def load_data():

    if not os.path.exists(DATA_PATH):
        raise ValueError("❌ Dataset file not found")

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print("Dataset size (raw):", len(data))

    X, y = [], []

    for item in data:
        grid = np.array(item["grid"])
        seq = item["sequence"]

        # 🔥 FIX 1: skip invalid samples
        if all(v == PAD_TOKEN for v in seq):
            continue

        padded = np.zeros((30, 30))
        padded[:grid.shape[0], :grid.shape[1]] = grid

        X.append(padded)
        y.append(seq)

    print("Dataset size (filtered):", len(X))

    if len(X) == 0:
        raise ValueError("❌ No valid training samples")

    return np.array(X), np.array(y)


def train():

    X, y = load_data()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    model = PolicyModel(seq_len=5, num_ops=7)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    print("🚀 Training started...")

    for epoch in range(30):

        logits = model(X)  # (batch, seq_len, num_ops)

        loss = 0
        valid_steps = 0

        for i in range(logits.shape[1]):

            mask = y[:, i] != PAD_TOKEN
            if mask.sum() == 0:
                continue

            step_logits = logits[mask, i, :]
            step_targets = y[mask, i]

            step_loss = loss_fn(step_logits, step_targets)

            # 🔥 CRITICAL: give higher weight to early steps
            if i == 0:
                step_loss = step_loss * 3.0   # <<<<< KEY FIX
            elif i == 1:
                step_loss = step_loss * 2.0

            if i == 0:
                stop_mask = step_targets == STOP_TOKEN
                if stop_mask.sum() > 0:
                    step_loss += 1.0   # stronger penalty
            loss += step_loss
            valid_steps += 1

        loss = loss / valid_steps

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # 🔥 save model
    os.makedirs("models/checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/checkpoints/policy.pt"
    )

    print("✅ Policy model saved!")


if __name__ == "__main__":
    train()