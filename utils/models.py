"""Hide and reveal networks for deep steg hiding"""

import torch
import torch.nn as nn


class HideNetwork(nn.Module):
    """Hides a secret in a cover to create a container."""
    def __init__(self):
        super(HideNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output 1 channel for the container
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)  # Concatenate cover and secret along the channel dimension
        container = self.conv(x)
        return container

class RevealNetwork(nn.Module):
    """Reveals the hidden secret from a container"""
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output 1 channel for the revealed secret
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, container):
        revealed_secret = self.conv(container)
        return revealed_secret