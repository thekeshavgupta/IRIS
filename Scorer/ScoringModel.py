import torch
import torch.nn as nn

class ScoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1280, 2500),
            nn.ReLU(),
            nn.Linear(2500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Implement the full model here, for now assume it outputs a float value.
    def forward(self, x):
        return self.model(x)
    