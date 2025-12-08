#!/usr/bin/env python3

import time
import logging

import torch
from torch.utils.data import DataLoader

from utils.data_selection import train_test_split
from dataset import TartesDataset
from model import TartesEmulator


"""General description"""


t0 = time.time()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load data
_, file, _ = train_test_split(train=1, test=1, seed=2025)
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
logger.info(f"The following experiments will be launched on {DEVICE}")
tartes_model = tartes_model.to(DEVICE)

# Inference
i = 0
for X_snowpack, X_sun, y in dataloader:
    X_snowpack, X_sun = X_snowpack.to(DEVICE), X_sun.to(DEVICE)
    y_hat = tartes_model(X_snowpack, X_sun)
    print(f"TARTES ALBEDO: {y[0][0]}")
    print(f"MODEL PREDICTION: {y_hat[0][0]}")
    print("\n")
    if i < 5:
        i += 1
    else:
        break

t1 = time.time()
logger.info(f"Run time: {(t1 - t0):.2f} secondes")
