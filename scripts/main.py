#!/usr/bin/env python3

import sys
import time
import logging

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchinfo import summary

from utils.data_selection import train_test_split
from normalization.mean_and_std import trigger_mean_and_std
from dataset import TartesDataset
from model import TartesEmulator
from training import train


"""General description"""


def main():
    """Description"""

    t0 = time.time()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Files selection
    logger.info("Data selection")
    train_files, test_files, _ = train_test_split(train=1, test=1, seed=2025)

    # Normalization
    logger.info("Computing normalization stats")
    trigger_mean_and_std(files_paths=train_files)

    # Batch and buffer
    BATCH_SIZE = 128
    BUFFER_SIZE = 50

    # Datasets
    logger.info("Creating datasets")
    train_dataset = TartesDataset(files_paths=train_files,
                                  batch_size=BATCH_SIZE,
                                  buffer_size=BUFFER_SIZE)
    test_dataset = TartesDataset(files_paths=test_files,
                                 batch_size=BATCH_SIZE,
                                 buffer_size=BUFFER_SIZE)

    # Dataloaders
    logger.info("Creating dataloaders")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                  num_workers=0, drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                 num_workers=0, drop_last=False)

    # 1st batch shapes
    for X_snowpack, X_sun, y in train_dataloader:
        logger.info("--- BATCH INFOS | SHAPES ---")
        logger.info(f"X_snowpack: {X_snowpack.shape}")
        logger.info(f"X_sun: {X_sun.shape}")
        logger.info(f"y: {y.shape}")
        break

    # Model
    logger.info("Creating model")
    tartes_model = TartesEmulator()
    summary(
        tartes_model,
        input_size=[(1, 6, 50, 1), (1, 3)],
        col_names=["kernel_size", "input_size", "output_size", "num_params"]
    )

    # Loss, optimizer and metric
    logger.info("Creating loss, optimizer and metric")
    loss_fn = MSELoss()
    optimizer = Adam(tartes_model.parameters())
    metric = MeanSquaredError()

    # Device
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"The following experiments will be launched on {DEVICE}")
    tartes_model = tartes_model.to(DEVICE)
    metric = metric.to(DEVICE)

    # Forward pass
    for X_snowpack, X_sun, y in train_dataloader:
        X_snowpack, X_sun = X_snowpack.to(DEVICE), X_sun.to(DEVICE)
        y_hat = tartes_model(X_snowpack, X_sun)
        logger.info("--- FORWARD PASS | SHAPES ---")
        logger.info(f"TARTES ALBEDO: {y.shape}")
        logger.info(f"MODEL PREDICTION: {y_hat.shape}")
        logger.info("--- FORWARD PASS | INFERENCE ---")
        logger.info(f"TARTES ALBEDO: {y[0][0]}")
        logger.info(f"MODEL PREDICTION: {y_hat[0][0]}")
        break

    # Training
    epochs = 1
    for ep in range(epochs):
        tartes_model = train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=tartes_model,
            loss_function=loss_fn,
            optimizer=optimizer,
            metric=metric,
            device=DEVICE,
            epoch=ep,
            logger=logger
        )

    # Save model parameters (save full model ?)
    if logger.level == logging.INFO:
        PATH = "/home/torrenta/TARTES-Emulator/data/model/tartes-model.pt"
        torch.save(tartes_model.state_dict(), PATH)

    t1 = time.time()
    logger.info(f"Run time: {(t1 - t0):.2f} seconds")

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
