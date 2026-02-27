#!/usr/bin/env python3

import sys
import time
import logging

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.nn import L1Loss
from torchmetrics import MeanSquaredError

from utils.data_selection import train_test_split, print_selection
from dataset import CnnCustomDataset
from models import CnnTartesEmulator
from training import evaluate_loop
from utils.norm import CustomNorm

"""
Full script to evaluate a model
-> same structure as main.py without training part
"""


def main():

    t0 = time.time()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
    logger.debug("Set norm")
    data_dir = "/home/torrenta/new-TARTES-Emulator/data"
    mean_and_std_path = f"{data_dir}/mean_and_std.parquet"
    norm = CustomNorm(mean_and_std_path)

    # ======================================================================= #
    logger.debug("Creating datasets")
    bigdata_dir = "/bigdata/BIGDATA/torrenta/Temp"
    # train_cache = f"{bigdata_dir}/train_dataset_cache.pt"
    # train_dataset = CnnCustomDataset(train_files, norm, train_cache)
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

    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     drop_last=DROP_LAST,
    #     prefetch_factor=PREFETCH_FACTOR,
    #     pin_memory=PIN_MEMORY,
    #     persistent_workers=PERSISTENT_WORKERS
    # )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=DROP_LAST,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # ======================================================================= #
    logger.debug("Build model")
    model = CnnTartesEmulator()
    model_path = f"{data_dir}/model.pt"
    model_state = torch.load(model_path, weights_only=True)["model_state"]
    model.load_state_dict(model_state)
    summary(
        model,
        input_size=[(1, 6, 50), (1, 4)],
        col_names=["kernel_size", "input_size", "output_size", "num_params"]
    )

    # ======================================================================= #
    logger.debug("Loss and metric")
    loss_fn = L1Loss()
    print("Loss: L1Loss (MAE)")
    metric_fn = MeanSquaredError()
    print("Metric: MeanSquaredError")
    print("=" * 140)

    # ======================================================================= #
    logger.debug("GPU device")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)
    metric_fn = metric_fn.to(device)
    print(f"The following experiments will be launched on {device}")
    print("-" * 140)

    # ======================================================================= #
    logger.debug("Evaluation on train/test datasets")

    # train_loss, train_metric = evaluate_loop(
    #     dataloader=train_dataloader,
    #     model=model,
    #     loss_function=loss_fn,
    #     metric_function=metric_fn,
    #     device=device
    # )

    test_loss, test_metric = evaluate_loop(
        dataloader=test_dataloader,
        model=model,
        loss_function=loss_fn,
        metric_function=metric_fn,
        device=device
    )

    # print(f"Loss on train: {train_loss}")
    # print(f"Metric on train: {train_metric}")
    print(f"Loss on test: {test_loss}")
    print(f"Metric on test: {test_metric}")
    print("-" * 140)

    t1 = time.time()
    print(f"Run time: {(t1 - t0):.2f} seconds")

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
