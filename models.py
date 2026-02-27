#!/usr/bin/env python3

import torch
import torch.nn as nn

"""
Size computation after a convolution1d or a pooling1d :
H_out = ceil[ (H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride ] + 1
"""


class CnnTartesEmulator(nn.Module):
    """Current Model"""

    def __init__(self):
        super().__init__()

        # conv block
        self.conv_block = nn.Sequential(

            # size: 6 * 50 * 1 -> 32 * 50 * 1
            nn.Conv1d(in_channels=6, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            # size: 32 * 50 * 1 -> 64 * 50 * 1
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            # size: 64 * 50 * 1 -> 64 * 50 * 1
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            # size: 64 * 50 * 1 -> 64 * 25 * 1
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, stride=2),
            nn.ELU(),

            # size: 64 * 25 * 1 -> 64 * 13 * 1
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, stride=2),
            nn.ELU(),

            # size: 64 * 13 * 1 -> 64 * 7 * 1
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, stride=2),
            nn.ELU()
        )

        # flatten
        self.flatten = nn.Flatten()

        # fully-connected block
        self.fc_block = nn.Sequential(
            # layer 1
            nn.Linear(64*7 + 4, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            # layer 2
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            # layer 3
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.2),
            # layer 4
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.2)
        )

        # out layer
        self.out_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_snowpack: torch.Tensor, x_sun: torch.Tensor):
        """Custom forward function"""
        x_snowpack = self.conv_block(x_snowpack)
        x_snowpack = self.flatten(x_snowpack)
        x = torch.cat((x_snowpack, x_sun), dim=1)
        x = self.fc_block(x)
        x = self.out_layer(x)
        return x


class CnnTartesEmulatorLinear(CnnTartesEmulator):
    """Current model with a linear output"""

    def __init__(self):
        super().__init__()

        # out layer
        self.out_layer = nn.Sequential(nn.Linear(128, 1))


class CnnTartesEmulatorMax(CnnTartesEmulator):
    """Model with a global max pooling at the end of the conv block"""

    def __init__(self):
        super().__init__()

        # conv block
        self.conv_block = nn.Sequential(

            # size: 6 * 50 * 1 -> 32 * 50 * 1
            nn.Conv1d(in_channels=6, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ELU(),

            # size: 32 * 50 * 1 -> 64 * 50 * 1
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ELU(),

            # size: 64 * 50 * 1 -> 128 * 50 * 1
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ELU(),

            # size: 128 * 50 * 1 -> 256 * 50 * 1
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ELU(),

            # size: 256 * 50 * 1 -> 256 * 50 * 1
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ELU(),

            # size: 256 * 50 * 1 -> 256 * 50 * 1
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ELU(),

            # size: 256 * 50 * 1 -> 256 * 1
            nn.AdaptiveMaxPool1d(1)
        )

        # fully-connected block
        self.fc_block = nn.Sequential(
            # layer 1
            nn.Linear(256 + 4, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            # layer 2
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            # layer 3
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.2),
            # layer 4
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.2)
        )


class CnnTartesEmulatorInitial(nn.Module):
    """CNN initial model, the one described in the internship report"""

    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=256,
                               kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256,
                               kernel_size=3, padding=1)

        # Pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.gmpool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()

        # Dense layers
        self.fc1 = nn.Linear(256 + 3, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 512)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(512, 512)
        self.dropout5 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(512, 512)
        self.dropout6 = nn.Dropout(0.2)
        self.fc7 = nn.Linear(512, 512)
        self.dropout7 = nn.Dropout(0.2)
        self.fc8 = nn.Linear(512, 512)
        self.dropout8 = nn.Dropout(0.2)
        self.fc9 = nn.Linear(512, 512)
        self.dropout9 = nn.Dropout(0.2)
        self.fc10 = nn.Linear(512, 512)
        self.dropout10 = nn.Dropout(0.2)
        self.fc11 = nn.Linear(512, 256)
        self.dropout11 = nn.Dropout(0.2)
        self.fc12 = nn.Linear(256, 128)
        self.dropout12 = nn.Dropout(0.2)
        self.fc13 = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _forward_snowpack(self, x_snowpack: torch.Tensor) -> torch.Tensor:
        """Snowpack forward function"""

        x_snowpack = self.conv1(x_snowpack)
        x_snowpack = self.conv2(x_snowpack)

        x_snowpack = self.pool1(x_snowpack)

        x_snowpack = self.conv3(x_snowpack)
        x_snowpack = self.conv4(x_snowpack)
        x_snowpack = self.conv5(x_snowpack)

        x_snowpack = self.pool2(x_snowpack)

        x_snowpack = self.conv6(x_snowpack)
        x_snowpack = self.conv7(x_snowpack)

        x_snowpack = self.gmpool(x_snowpack)
        x_snowpack = self.flatten(x_snowpack)

        return x_snowpack

    def forward(self, x_snowpack: torch.Tensor, x_sun: torch.Tensor):
        """Custom forward function"""

        x_snowpack = self._forward_snowpack(x_snowpack)

        x = torch.cat((x_snowpack, x_sun), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout7(x)

        x = self.fc8(x)
        x = self.relu(x)
        x = self.dropout8(x)

        x = self.fc9(x)
        x = self.relu(x)
        x = self.dropout9(x)

        x = self.fc10(x)
        x = self.relu(x)
        x = self.dropout10(x)

        x = self.fc11(x)
        x = self.relu(x)
        x = self.dropout11(x)

        x = self.fc12(x)
        x = self.relu(x)
        x = self.dropout12(x)

        x = self.fc13(x)
        x = self.sigmoid(x)

        return x


class MlpTartesEmulator(nn.Module):
    """Model with only dense layers"""

    def __init__(self):
        super().__init__()

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
        """Custom forward function"""
        return self.net(x)
