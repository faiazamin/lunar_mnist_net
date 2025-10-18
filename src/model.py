import torch
import torch.nn as nn

class L2Norm(nn.Module):
    """Normalizes each sample to unit L2 length."""
    def forward(self, x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-6)

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            L2Norm(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)