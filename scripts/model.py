import torch
import torch.nn as nn

# Première architecture simple pour les premiers essais

class TartesEmulator(nn.Module):

    def __init__(self):
        
        super().__init__()

        # conv layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()

        # dense layers
        self.fc1 = nn.Linear(35, 64)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def _forward_conv(self, x_snowpack):
        # conv layers forward
        x_snowpack = self.conv1(x_snowpack)
        x_snowpack = self.conv2(x_snowpack)
        x_snowpack = self.global_max_pool(x_snowpack)
        x_snowpack = self.flatten(x_snowpack)
        return x_snowpack

    def forward(self, x_snowpack, x_sun):
        # forward
        x_snowpack = self._forward_conv(x_snowpack)
        x = torch.cat((x_snowpack, x_sun), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
