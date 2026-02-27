#!/usr/bin/env python3

import torch

# print("cuda version:", torch.version.cuda)

"""Have a look on your GPUs"""

print("GPUs available:")
for i in range(torch.cuda.device_count()):
    print("-", i, torch.cuda.get_device_name(i))

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Current device index: {DEVICE.index}")
print(f"Your GPU ({DEVICE}) is a {torch.cuda.get_device_name(DEVICE.index)}")
