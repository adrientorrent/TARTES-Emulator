#!/usr/bin/env python3

import torch
import torch.nn as nn


class TartesEmulator(nn.Module):
    """General description"""

    def __init__(self):
        super().__init__()

        # --- SNOWPACK CONV LAYERS ---

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256,
                               kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512,
                               kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)

        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()

        self.snowpack_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # --- SUN DENSE LAYERS ---

        self.sun_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # --- FINAL DENSE LAYERS ---

        self.fc1 = nn.Linear(64 + 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _forward_snowpack(self, x_snowpack: torch.Tensor) -> torch.Tensor:
        """Snowpack forward function"""

        x_snowpack = self.conv1(x_snowpack)
        x_snowpack = self.bn1(x_snowpack)

        x_snowpack = self.conv2(x_snowpack)
        x_snowpack = self.bn2(x_snowpack)

        x_snowpack = self.conv3(x_snowpack)
        x_snowpack = self.bn3(x_snowpack)

        x_snowpack = self.conv4(x_snowpack)
        x_snowpack = self.bn4(x_snowpack)

        x_snowpack = self.conv5(x_snowpack)
        x_snowpack = self.bn5(x_snowpack)

        x_snowpack = self.gmp(x_snowpack)
        x_snowpack = self.flatten(x_snowpack)

        x_snowpack = self.snowpack_fc(x_snowpack)

        return x_snowpack

    def _forward_sun(self, x_sun: torch.Tensor) -> torch.Tensor:
        """Sun forward function"""

        x_sun = self.sun_fc(x_sun)

        return x_sun

    def forward(self, x_snowpack: torch.Tensor, x_sun: torch.Tensor):
        """Forward function"""

        x_snowpack = self._forward_snowpack(x_snowpack)
        x_sun = self._forward_sun(x_sun)

        x = torch.cat((x_snowpack, x_sun), dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)
        x = self.relu(x)

        x = self.fc6(x)
        x = self.sigmoid(x)

        return x
