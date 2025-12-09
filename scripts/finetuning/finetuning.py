#!/usr/bin/env python3

import sys
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchmetrics import MeanSquaredError
from torch.optim import Adam, lr_scheduler

import optuna

from utils.data_selection import train_test_split
from normalization.mean_and_std import trigger_mean_and_std
from normalization.normalize import CustomNorm2
from dataset import MlpTartesIterableDataset
from training import train, evaluate


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


train_files, _, _ = train_test_split(train=1, test=1, seed=1209)
data_dir = "/home/torrenta/TARTES-Emulator/data"
mean_and_std_path = f"{data_dir}/normalization/mean_and_std_111209.parquet"
trigger_mean_and_std(files_paths=train_files, out_path=mean_and_std_path,
                     logger=logger)
custom_norm = CustomNorm2(mean_and_std_path)
train_dataset = MlpTartesIterableDataset(files=train_files, norm=custom_norm)


class OptunaTartesEmulator(nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        fc_layers = []
        prev_size = 300 + 3
        for size in layer_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2))
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.Sigmoid())
        self.fc_net = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor):
        """Forward function"""
        return self.fc_net(x)


def objective(trial: optuna.trial.Trial):
    """"Do 1 Trial"""

    # output
    metric = 0.

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        num_workers=16,
        prefetch_factor=64,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False
    )

    n_layers = trial.suggest_int("n_layers", 5, 15)
    layer_sizes = []
    for i in range(n_layers):
        size = trial.suggest_categorical(
            f"layer_{i+1}_size",
            [10, 50, 100, 300, 500, 1000]
        )
        layer_sizes.append(size)
    model = OptunaTartesEmulator(layer_sizes=layer_sizes)

    mse_loss_fn = MSELoss()
    mse_metric_fn = MeanSquaredError()
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    adam_optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    explr_scheduler = lr_scheduler.ExponentialLR(adam_optimizer, gamma=0.98)
    scaler = torch.GradScaler()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mse_metric_fn = mse_metric_fn.to(device)

    epochs = 3
    for _ in range(epochs):
        model = train(
            train_dataloader=train_dataloader,
            model=model,
            loss_function=mse_loss_fn,
            optimizer=adam_optimizer,
            scheduler=explr_scheduler,
            scaler=scaler,
            device=device
        )
        _, metric = evaluate(
            dataloader=train_dataloader,
            model=model,
            loss_function=mse_loss_fn,
            metric_function=mse_metric_fn,
            device=device,
            desc="Evaluation"
        )

    return metric


if __name__ == "__main__":

    t0 = time.time()

    study_path = "/home/torrenta/TARTES-Emulator/scripts/finetuning/study.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="TARTES",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)
    print(study.best_params, study.best_value)

    t1 = time.time()
    logger.info(f"Run time: {(t1 - t0):.2f} seconds")

    sys.exit(0)
