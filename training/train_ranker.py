import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# training/train_ranker.py

import torch
import os
import numpy as np

from models.dataset import get_dataset
from models.ranker import ProgramRanker

from engine.neural_scorer import encode_program, encode_grid


def train():

    data = get_dataset()

    if len(data) == 0:
        raise ValueError("❌ Dataset is empty. Run search first to collect samples.")

    X = []
    y = []

    for item in data:
        program = item["program"]
        score = item["score"]
        features = item["features"]  # stored from search

        # 🔥 FIX: features is already extracted from grid
        # so we directly convert it to vector

        # program features
        prog_vec = encode_program(program)

        # grid features (reconstruct minimal vector)
        grid_vec = [
            features.get("height", 0),
            features.get("width", 0),
            features.get("num_colors", 0),
            features.get("is_square", 0),
            features.get("color_variance", 0),
            features.get("non_zero_ratio", 0),
        ]

        full_vec = prog_vec + grid_vec

        X.append(full_vec)
        y.append(score)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    print(f"✅ Training samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")

    model = ProgramRanker(input_dim=X.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(100):

        model.train()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    print(f"✅ Final Loss: {best_loss:.4f}")

    # 🔥 SAVE MODEL
    os.makedirs("models/checkpoints", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
        },
        "models/checkpoints/ranker.pt"
    )

    print("✅ Model saved at models/checkpoints/ranker.pt")


if __name__ == "__main__":
    train()