import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import json
import numpy as np
import os

from models.value_model import ValueModel

DATA_PATH = "models/value_dataset.json"


def load_data():

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    X, y = [], []

    for item in data:

        inp = np.array(item["input"])
        out = np.array(item["output"])

        # combine input + output
        combined = np.zeros((30, 30))
        combined[:inp.shape[0], :inp.shape[1]] = inp
        combined[15:15+out.shape[0], :out.shape[1]] = out

        X.append(combined)
        y.append(item["label"])

    return np.array(X), np.array(y)


def train():

    X, y = load_data()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = ValueModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(20):

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    os.makedirs("models/checkpoints", exist_ok=True)

    torch.save(model.state_dict(), "models/checkpoints/value.pt")

    print("✅ Value model saved!")


if __name__ == "__main__":
    train()