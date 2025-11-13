#!/usr/bin/env python3

import torch

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
