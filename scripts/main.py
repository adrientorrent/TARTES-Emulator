#!/usr/bin/env python3

import sys
import time
import logging

import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchinfo import summary

from utils.data_selection import train_test_split
from normalization.mean_and_std import trigger_mean_and_std
from dataset import TartesDataset
from mlp import MLPTartesEmulator
from training import train, evaluate


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
    logger.debug("Data selection")
    train_files, test_files, _ = train_test_split(train=2, test=1, seed=2025)

    print("=" * 140)
    print("Train dataset:")
    train_years = []
    for file_path in train_files:
        geo = "Alpes" if "Alpes" in file_path else "Pyrénées"
        y1 = file_path.split("_")[1]
        y2 = file_path.split("_")[2][:4]
        txt = f" - {geo} {y1} {y2}"
        if txt not in train_years:
            print(txt)
            train_years.append(txt)
    print("Test dataset:")
    test_years = []
    for file_path in test_files:
        geo = "Alpes" if "Alpes" in file_path else "Pyrénées"
        y1 = file_path.split("_")[1]
        y2 = file_path.split("_")[2][:4]
        txt = f" - {geo} {y1} {y2}"
        if txt not in test_years:
            print(txt)
            test_years.append(txt)

    # Normalization
    logger.debug("Computing normalization stats")
    trigger_mean_and_std(files_paths=train_files, logger=logger)

    # Datasets
    logger.debug("Creating datasets")
    train_dataset = TartesDataset(files_paths=train_files)
    test_dataset = TartesDataset(files_paths=test_files)

    # Dataloaders
    logger.debug("Creating dataloaders")

    BATCH_SIZE = 256
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 32
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    DROP_LAST = False

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=DROP_LAST
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=DROP_LAST
    )

    # 1st batch shapes
    if logger.getEffectiveLevel() == logging.DEBUG:

        for X_snowpack, X_sun, y in train_dataloader:

            logger.debug("--- BATCH INFOS | SHAPES ---")
            logger.debug(f"X_snowpack: {X_snowpack.shape}")
            logger.debug(f"X_sun: {X_sun.shape}")
            logger.debug(f"y: {y.shape}")

            break

    # Model
    logger.debug("Build model")

    tartes_model = MLPTartesEmulator()

    summary(
        tartes_model,
        input_size=[(1, 6, 50), (1, 3)],
        col_names=["kernel_size", "input_size", "output_size", "num_params"]
    )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Drop last: {DROP_LAST}")
    print("=" * 140)

    # Loss, metric, optimizer and scheduler
    logger.debug("Learning setup")

    mse_loss_fn = MSELoss()
    print("Loss: MSELoss")

    mse_metric_fn = MeanSquaredError()
    print("Metric: MeanSquaredError")

    adam_optimizer = Adam(tartes_model.parameters(),
                          lr=1e-3, weight_decay=1e-4)
    print("Opimizer: Adam(lr=1e-3, weight_decay=1e-4)")

    explr_scheduler = lr_scheduler.ExponentialLR(adam_optimizer, gamma=0.98)
    print("Scheduler: ExponentialLR(gamma=0.98)")

    print("=" * 140)

    # Device
    # GPUs on sxbigdata1 : 0, 1 are NVIDIA A30 and 2 is a Tesla V100-PCIE-16GB
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"The following experiments will be launched on {DEVICE}")
    logger.debug(
        f"GPU {DEVICE.index} is a {torch.cuda.get_device_name(DEVICE.index)}"
    )
    print("-" * 140)

    tartes_model = tartes_model.to(DEVICE)
    mse_metric_fn = mse_metric_fn.to(DEVICE)

    # Forward pass
    if logger.getEffectiveLevel() == logging.DEBUG:

        for X_snowpack, X_sun, y in train_dataloader:

            X_snowpack, X_sun = X_snowpack.to(DEVICE), X_sun.to(DEVICE)
            y_hat = tartes_model(X_snowpack, X_sun)

            logger.debug("--- FORWARD PASS | SHAPES ---")
            logger.debug(f"TARTES ALBEDO: {y.shape}")
            logger.debug(f"MODEL PREDICTION: {y_hat.shape}")

            logger.debug("--- FORWARD PASS | INFERENCE ---")
            logger.debug(f"TARTES ALBEDO: {y[0][0]}")
            logger.debug(f"MODEL PREDICTION: {y_hat[0][0]}")

            break

    # Training loop
    epochs = 100
    logger.debug(f"Training on {epochs} epochs")
    for ep in range(epochs):

        print(f"EPOCH {ep + 1}/{epochs}")

        start_time = time.time()

        # Training
        tartes_model = train(
            train_dataloader=train_dataloader,
            model=tartes_model,
            loss_function=mse_loss_fn,
            optimizer=adam_optimizer,
            scheduler=explr_scheduler,
            device=DEVICE
        )

        # Evaluation based on testing data
        test_loss, test_metric = evaluate(
            dataloader=test_dataloader,
            model=tartes_model,
            loss_function=mse_loss_fn,
            metric_function=mse_metric_fn,
            device=DEVICE,
            desc="Evaluation on testing data"
        )

        # Results
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time : {elapsed_time:.2f} s")
        print(f"[TEST] Loss: {test_loss}")
        print(f"[TEST] Metric: {test_metric}")

        print("-" * 140)

    # Save model parameters (save full model ?)
    if logger.getEffectiveLevel() == logging.INFO:
        PATH = "/home/torrenta/TARTES-Emulator/data/model/tartes-model.pt"
        torch.save(tartes_model.state_dict(), PATH)
        print(f"Model save in {PATH}")

    t1 = time.time()
    print(f"Run time: {(t1 - t0):.2f} seconds")

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
