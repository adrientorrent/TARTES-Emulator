#!/usr/bin/env python3

import sys
import time
import logging
import csv

import torch
from torch.nn import L1Loss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchinfo import summary

from utils.data_selection import train_test_split, print_selection
from normalization.mean_and_std import trigger_mean_and_std
from normalization.normalize import CustomNorm2
from dataset import CnnTartesIterableDataset
from model import CnnTartesEmulator
from training import train_cnn, evaluate_cnn


def main():

    t0 = time.time()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Files selection
    logger.debug("Data selection")
    train_files, test_files, _ = train_test_split(train=10, test=2, seed=2025)

    print("=" * 115)
    print("Train dataset:")
    print_selection(files=train_files)
    print("Test dataset:")
    print_selection(files=test_files)

    # Normalization
    logger.debug("Computing normalization stats")
    data_dir = "/home/torrenta/TARTES-Emulator/data"
    mean_and_std_path = f"{data_dir}/normalization/mean_and_std_1022025.parquet"
    trigger_mean_and_std(
        files_paths=train_files,
        out_path=mean_and_std_path,
        logger=logger
    )
    custom_norm = CustomNorm2(mean_and_std_path)

    # Datasets
    logger.debug("Creating datasets")
    train_dataset = CnnTartesIterableDataset(files=train_files,
                                             norm=custom_norm)
    test_dataset = CnnTartesIterableDataset(files=test_files,
                                            norm=custom_norm)

    # Dataloaders
    logger.debug("Creating dataloaders")

    BATCH_SIZE = 256
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 64
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

    print("=" * 115)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Drop last: {DROP_LAST}")

    # 1st batch shapes
    if logger.getEffectiveLevel() == logging.DEBUG:
        for X_snow, X_sun, y in train_dataloader:
            print(f"BATCH SHAPES | X_snow: {X_snow.shape}, "
                  f"x_sun: {X_sun.shape}, y: {y.shape}")
            for i in range(1):
                print(X_snow[i])
                print(X_sun[i])
                print(y[i])
            break

    # Model
    logger.debug("Build model")
    cnn_model = CnnTartesEmulator()
    summary(
        cnn_model,
        input_size=[(1, 6, 50), (1, 3)],
        col_names=["kernel_size", "input_size", "output_size", "num_params"]
    )

    # Loss and metric
    logger.debug("Loss and metric")
    mae_loss_fn = L1Loss()
    print("Loss: L1Loss")
    mse_metric_fn = MeanSquaredError()
    print("Metric: MeanSquaredError")
    learning_rate = 1e-4
    adam_optimizer = Adam(cnn_model.parameters(), lr=learning_rate)
    print("Opimizer: Adam")
    print(f"Learning rate: {learning_rate}")
    gamma = 0.95
    explr_scheduler = lr_scheduler.ExponentialLR(adam_optimizer, gamma=gamma)
    print("Scheduler: ExponentialLR")
    print(f"Gamma: {gamma}")
    print("=" * 115)

    # Device
    # GPUs on sxbigdata1 : 0, 1 are NVIDIA A30 and 2 is a Tesla V100-PCIE-16GB
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(DEVICE)
    mse_metric_fn = mse_metric_fn.to(DEVICE)
    print(f"The following experiments will be launched on {DEVICE}")
    logger.debug(
        f"GPU {DEVICE.index} is a {torch.cuda.get_device_name(DEVICE.index)}"
    )
    print("-" * 115)

    # Forward pass
    if logger.getEffectiveLevel() == logging.DEBUG:
        for X_snow, X_sun, y in train_dataloader:
            X_snow, X_sun = X_snow.to(DEVICE), X_sun.to(DEVICE)
            y_hat = cnn_model(X_snow, X_sun)
            print(f"OUTPUT SHAPES | y: {y.shape}, y_hat: {y_hat.shape}")
            print(f"FORWARD PASS | "
                  f"Tartes: {y[0][0]}, Model prediction: {y_hat[0][0]}")
            break

    # Head of csv file
    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["epoch", "loss", "metric"])

    # Training loop
    epochs = 100
    logger.debug(f"Training on {epochs} epochs")
    for ep in range(epochs):

        print(f"EPOCH {ep + 1}/{epochs}")

        start_time = time.time()

        # Training
        cnn_model = train_cnn(
            train_dataloader=train_dataloader,
            model=cnn_model,
            loss_function=mae_loss_fn,
            optimizer=adam_optimizer,
            scheduler=explr_scheduler,
            device=DEVICE
        )

        # Evaluation based on testing data
        test_loss, test_metric = evaluate_cnn(
            dataloader=test_dataloader,
            model=cnn_model,
            loss_function=mae_loss_fn,
            metric_function=mse_metric_fn,
            device=DEVICE,
            desc="Evaluation"
        )

        # Results
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time : {elapsed_time:.2f} s")
        print(f"Loss: {test_loss}")
        print(f"Metric: {test_metric}")
        with open("results.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([ep + 1, test_loss, test_metric])

        print("-" * 115)

    # Save model parameters (save full model ?)
    if logger.getEffectiveLevel() == logging.INFO:
        PATH = "/home/torrenta/TARTES-Emulator/data/model/big-tartes-model.pt"
        torch.save(cnn_model.state_dict(), PATH)
        print(f"Model save in {PATH}")

    t1 = time.time()
    print(f"Run time: {(t1 - t0):.2f} seconds")

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
