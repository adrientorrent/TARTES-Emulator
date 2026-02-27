#!/usr/bin/env python3

import sys
import time
import logging
import csv

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.nn import MSELoss
from torchmetrics import MeanAbsoluteError
from torch.optim import Adam, lr_scheduler

from utils.data_selection import train_test_split, print_selection
from utils.mean_and_std import trigger_mean_and_std
from dataset import CnnCustomDataset
from models import CnnTartesEmulator
from training import train_loop, evaluate_loop
from utils.norm import CustomNorm


"""Same as main.py, with LR scheduler: CyclicLR"""


def main():

    t0 = time.time()

    data_dir = "/home/torrenta/new-TARTES-Emulator/data"

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # resume training ?
    RESUME_TRAINING = True
    model_path = f"{data_dir}/model_cyclicLR.pt"
    checkpoint = {}
    if RESUME_TRAINING:
        checkpoint = torch.load(model_path, weights_only=True)

    # ======================================================================= #
    print("=" * 140)
    logger.debug("Data selection")
    train_files, test_files, _ = train_test_split(
        train={("Alpes", "1979_1980"), ("Pyrenees", "1980_1981")},
        test={("Alpes", "1981_1982")}
    )
    print("Train dataset:")
    print_selection(files=train_files)
    print("Test dataset:")
    print_selection(files=test_files)

    # ======================================================================= #
    logger.debug("Computing normalization stats")
    mean_and_std_path = f"{data_dir}/mean_and_std.parquet"
    trigger_mean_and_std(files_paths=train_files, out_path=mean_and_std_path)
    logger.debug("Set norm")
    norm = CustomNorm(mean_and_std_path)

    # ======================================================================= #
    logger.debug("Creating datasets")
    bigdata_dir = "/bigdata/BIGDATA/torrenta/Temp"
    train_cache = f"{bigdata_dir}/train_dataset_cache.pt"
    train_dataset = CnnCustomDataset(train_files, norm, train_cache)
    test_cache = f"{bigdata_dir}/test_dataset_cache.pt"
    test_dataset = CnnCustomDataset(test_files, norm, test_cache)

    # ======================================================================= #
    print("=" * 140)
    logger.debug("Creating dataloaders")
    BATCH_SIZE = 256
    print(f"Batch size: {BATCH_SIZE}")
    DROP_LAST = False
    print(f"Drop last: {DROP_LAST}")
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 16
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    print(f"Workers: {NUM_WORKERS}")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=DROP_LAST,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=DROP_LAST,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    if logger.getEffectiveLevel() == logging.DEBUG:
        logger.debug("First batch loading")
        for X_snow, X_sun, y in train_dataloader:
            print(f"SHAPES : {X_snow.shape}, {X_sun.shape}, {y.shape}")
            for i in range(1):
                print(X_snow[i])
                print(X_sun[i])
                print(y[i])
            break

    # ======================================================================= #
    logger.debug("Build model")
    model = CnnTartesEmulator()
    summary(
        model,
        input_size=[(1, 6, 50), (1, 4)],
        col_names=["kernel_size", "input_size", "output_size", "num_params"]
    )
    if RESUME_TRAINING:
        model.load_state_dict(checkpoint["model_state"])

    # ======================================================================= #
    logger.debug("GPU device")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ======================================================================= #
    logger.debug("Loss and metric")
    loss_fn = MSELoss()
    print("Loss: MSELoss")
    metric_fn = MeanAbsoluteError()
    metric_fn = metric_fn.to(device)
    print("Metric: MeanAbsoluteError")
    learning_rate = 1e-4
    # weight_decay nécessaire ?
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    print("Opimizer: Adam")
    print(f"Start learning rate: {learning_rate}")
    learning_rate_scheduler = lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=1e-6, max_lr=1e-4,
        step_size_up=20, step_size_down=20,
        mode="triangular2"
    )
    print("LRScheduler: CyclicLR(base_lr=1e-6, max_lr=1e-4, "
          "step_size_up=20, step_size_down=20, mode='triangular2')")
    if RESUME_TRAINING:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        learning_rate_scheduler.load_state_dict(checkpoint["scheduler_state"])

    print("=" * 140)
    print(f"The following experiments will be launched on {device}")
    print("-" * 140)

    # ======================================================================= #
    if logger.getEffectiveLevel() == logging.DEBUG:
        logger.debug("Forward pass")
        for X_snow, X_sun, y in train_dataloader:
            X_snow, X_sun = X_snow.to(device), X_sun.to(device)
            y_hat = model(X_snow, X_sun)
            print(f"y: {y.shape}, y_hat: {y_hat.shape}")
            print(f"Target: {y[0][0]}, Prediction: {y_hat[0][0]}")
            break
        print("-" * 140)

    # ======================================================================= #
    logger.debug("CSV file")
    csv_file_path = f"{data_dir}/results_cyclicLR.csv"
    if not RESUME_TRAINING:
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([
                "epoch", "learning_rate",
                "train_loss", "train_metric",
                "test_loss", "test_metric"
            ])

    # ======================================================================= #
    epochs = 200
    logger.debug(f"Training on {epochs} epochs")

    start_epoch = 0
    if RESUME_TRAINING:
        start_epoch = checkpoint["epoch"] + 1

    for ep in range(start_epoch, start_epoch + epochs):

        print(f"EPOCH {ep + 1}/{start_epoch + epochs}")
        start_time = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        model = train_loop(
            train_dataloader=train_dataloader,
            model=model,
            loss_function=loss_fn,
            optimizer=optimizer,
            device=device
        )

        train_loss, train_metric = evaluate_loop(
            dataloader=train_dataloader,
            model=model,
            loss_function=loss_fn,
            metric_function=metric_fn,
            device=device
        )

        test_loss, test_metric = evaluate_loop(
            dataloader=test_dataloader,
            model=model,
            loss_function=loss_fn,
            metric_function=metric_fn,
            device=device
        )

        learning_rate_scheduler.step()

        elapsed_time = time.time() - start_time
        print(f"Elapsed Time : {elapsed_time:.2f} s")
        print(f"Train loss: {train_loss}")
        print(f"Train metric: {train_metric}")
        print(f"Test loss: {test_loss}")
        print(f"Test metric: {test_metric}")

        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([
                ep + 1, current_lr,
                train_loss, train_metric,
                test_loss, test_metric
            ])

        # early stopping ?
        if logger.getEffectiveLevel() == logging.INFO:
            new_checkpoint = {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": learning_rate_scheduler.state_dict()
            }
            torch.save(new_checkpoint, model_path)
            print(f"Model save in {model_path}")

        print("-" * 140)

    t1 = time.time()
    print(f"Run time: {(t1 - t0):.2f} seconds")

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
