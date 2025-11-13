#!/usr/bin/env python3

import time
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchinfo import summary
from utils.get_parquet_files import split_train_test_val
from dataset import TartesDataset
from model import TartesEmulator
from training import train

debug = True

t0 = time.time()

# Datasets
train_files, test_files, val_files = split_train_test_val(0.7, 0.2, 0.1)
train_files, test_files = [train_files[0]], [test_files[0]]
train_dataset = TartesDataset(parquet_files_groupby_year=train_files)
test_dataset = TartesDataset(parquet_files_groupby_year=test_files)
val_dataset = TartesDataset(parquet_files_groupby_year=val_files)

# Dataloaders
batch_size = 64
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

# 1st batch
for X_snowpack, X_sun, y in train_dataloader:
    print("--- BATCH INFO | SHAPES ---")
    print(f"X_snowpack: {X_snowpack.shape}")
    print(f"X_sun: {X_sun.shape}")
    print(f"y: {y.shape}")
    break

# Model
tartes_model = TartesEmulator()
summary(
    tartes_model,
    input_size=[(1, 6, 50, 1), (1, 3)],
    col_names=["kernel_size", "input_size", "output_size", "num_params"]
)

# Loss, optimizer and metric
loss_fn = MSELoss()
optimizer = Adam(tartes_model.parameters())
metric = MeanSquaredError()

# Device
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"The following experiments will be launched on {DEVICE}")
tartes_model = tartes_model.to(DEVICE)
metric = metric.to(DEVICE)

# Forward pass
for X_snowpack, X_sun, y in train_dataloader:
    X_snowpack, X_sun = X_snowpack.to(DEVICE), X_sun.to(DEVICE)
    y_hat = tartes_model(X_snowpack, X_sun)
    print("--- FORWARD PASS | SHAPES ---")
    print(f"TARTES ALBEDO: {y.shape}")
    print(f"MODEL PREDICTION: {y_hat.shape}")
    print("--- FORWARD PASS | INFERENCE ---")
    print(f"TARTES ALBEDO: {y[0][0]}")
    print(f"MODEL PREDICTION: {y_hat[0][0]}")
    break

# Training
epochs = 1
for ep in range(epochs):
    model = train(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=tartes_model,
        loss_function=loss_fn,
        optimizer=optimizer,
        metric=metric,
        device=DEVICE,
        epoch=ep,
        debug=debug
    )

# Save model parameters (save full model ?)
if not debug:
    PATH = "/home/torrenta/TARTES-Emulator/data/model/tartes-model.pt"
    torch.save(tartes_model.state_dict(), PATH)

t1 = time.time()
print(f"=> Run time: {(t1 - t0):.2f} secondes")
