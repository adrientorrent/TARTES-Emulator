#!/usr/bin/env python3

import torch

print(torch.version.cuda)

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE} is a {torch.cuda.get_device_name(DEVICE.index)}")
