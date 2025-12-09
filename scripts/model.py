#!/usr/bin/env python3

import torch
import torch.nn as nn


class CnnTartesEmulator(nn.Module):
    """CNN model"""

    def __init__(self):
        super().__init__()

        # --- SNOWPACK CONV LAYERS ---

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32,
                               kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64,
                               kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool1d(kernel_size=5)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)

        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=3, padding=1)

        self.gmpool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()

        self.snowpack_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )

        # --- SUN DENSE LAYERS ---

        self.sun_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU()
        )

        # --- FINAL DENSE LAYERS ---

        self.fc1 = nn.Linear(32 + 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def _forward_snowpack(self, x_snowpack: torch.Tensor) -> torch.Tensor:
        """Snowpack forward function"""

        x_snowpack = self.conv1(x_snowpack)
        x_snowpack = self.conv2(x_snowpack)

        x_snowpack = self.pool1(x_snowpack)

        x_snowpack = self.conv3(x_snowpack)
        x_snowpack = self.conv4(x_snowpack)

        x_snowpack = self.pool2(x_snowpack)

        x_snowpack = self.conv5(x_snowpack)
        x_snowpack = self.conv6(x_snowpack)

        x_snowpack = self.gmpool(x_snowpack)
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
        x = self.elu(x)

        x = self.fc2(x)
        x = self.elu(x)
        x = self.dropout1(x)

        x = self.fc3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        x = self.fc4(x)
        x = self.elu(x)
        x = self.dropout3(x)

        x = self.fc5(x)
        x = self.elu(x)

        x = self.fc6(x)
        x = self.sigmoid(x)

        return x


class MlpTartesEmulator(nn.Module):
    """Model with only dense layers"""

    def __init__(self):
        super().__init__()

        # --- DENSE LAYERS ---

        self.net = nn.Sequential(
            nn.Linear(300 + 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """Forward function"""
        return self.net(x)
