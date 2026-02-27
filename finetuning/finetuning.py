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
from utils.mean_and_std import trigger_mean_and_std
from utils.norm import CustomNormMlp
from utils.iterable_dataset import MlpTartesIterableDataset
from training import train_loop, evaluate_loop


"""
Bayesian search of the best MLP, with Optuna tools

The script is old and may not work with the current version of the code.
-> it's still showing how to use Optuna.
"""


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

train_files, _, _ = train_test_split(train=1, test=1, seed=1209)
data_dir = "/home/torrenta/TARTES-Emulator/data"
mean_and_std_path = f"{data_dir}/normalization/mean_and_std_111209.parquet"
trigger_mean_and_std(files_paths=train_files, out_path=mean_and_std_path)
custom_norm = CustomNormMlp(mean_and_std_path)
train_dataset = MlpTartesIterableDataset(files=train_files, norm=custom_norm)


class OptunaTartesEmulator(nn.Module):

    def __init__(self, layer_sizes, dropout):
        super().__init__()
        fc_layers = []
        prev_size = 300 + 3
        for size in layer_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
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
        num_workers=4,
        prefetch_factor=64,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False
    )

    hidden_layers = trial.suggest_int("hidden_layers", 2, 15)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 1000, log=True)
    shrinkage = trial.suggest_float("shrinkage", 0.5, 1.0)
    layer_sizes = []
    prev_size = hidden_dim
    for i in range(hidden_layers):
        layer_sizes.append(prev_size)
        prev_size = int(shrinkage * prev_size)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    model = OptunaTartesEmulator(layer_sizes=layer_sizes, dropout=dropout)

    mse_loss_fn = MSELoss()
    mse_metric_fn = MeanSquaredError()
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    adam_optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    explr_scheduler = lr_scheduler.ExponentialLR(adam_optimizer, gamma=1.)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mse_metric_fn = mse_metric_fn.to(device)

    epochs = 5
    for _ in range(epochs):
        model = train_loop(
            train_dataloader=train_dataloader,
            model=model,
            loss_function=mse_loss_fn,
            optimizer=adam_optimizer,
            device=device
        )
        _, metric = evaluate_loop(
            dataloader=train_dataloader,
            model=model,
            loss_function=mse_loss_fn,
            metric_function=mse_metric_fn,
            device=device,
        )
        explr_scheduler.step()

    return metric


if __name__ == "__main__":

    t0 = time.time()

    study_path = "/home/torrenta/TARTES-Emulator/scripts/finetuning/study1.db"
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
