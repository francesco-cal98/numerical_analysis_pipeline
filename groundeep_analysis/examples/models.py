"""Toy models used for generating demonstration assets."""

from __future__ import annotations

import torch.nn as nn


class ToyMLP(nn.Module):
    """Simple MLP that maps flattened images to an embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
