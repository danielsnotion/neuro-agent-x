import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn as nn


class ARCModel(nn.Module):

    def __init__(self):
        super().__init__()

        # simple policy network
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # predict operation logits
        self.head = nn.Linear(64, 10)  # 10 ops

    def forward(self, x):
        x = self.encoder(x)
        logits = self.head(x)
        return logits