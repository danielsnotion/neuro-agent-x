import torch.nn as nn


class PolicyModel(nn.Module):

    def __init__(self, seq_len=5, num_ops=7):
        super().__init__()

        self.seq_len = seq_len
        self.num_ops = num_ops

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.head = nn.Linear(64, seq_len * num_ops)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x.view(-1, self.seq_len, self.num_ops)