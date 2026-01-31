import torch
import torch.nn as nn

class ScoringModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2050, 2500),
            nn.BatchNorm1d(2500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2500, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for better convergence"""
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)
    