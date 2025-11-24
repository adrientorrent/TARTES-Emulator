#!/usr/bin/env python3

import time
import torch
from scripts.utils.data_selection import get_parquet_paths
from dataset import TartesDataset
from torch.utils.data import DataLoader
from model import TartesEmulator

t0 = time.time()

# Load data
file = [[get_parquet_paths()[0]]]
dataset = TartesDataset(file)
dataloader = DataLoader(dataset=dataset, batch_size=1)

# Model architecture
tartes_model = TartesEmulator()

# Load model params
params_path = "/home/torrenta/TARTES-Emulator/data/model/tartes-model.pt"
tartes_model.load_state_dict(torch.load(params_path, weights_only=True))

# Evaluation mode
tartes_model.eval()

# Device
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"The following experiments will be launched on {DEVICE}")
tartes_model = tartes_model.to(DEVICE)

# Inference
for X_snowpack, X_sun, y in dataloader:
    X_snowpack, X_sun = X_snowpack.to(DEVICE), X_sun.to(DEVICE)
    y_hat = tartes_model(X_snowpack, X_sun)
    print(f"TARTES ALBEDO: {y[0][0]}")
    print(f"MODEL PREDICTION: {y_hat[0][0]}")
    break

t1 = time.time()
print(f"=> Run time: {(t1 - t0):.2f} secondes")
