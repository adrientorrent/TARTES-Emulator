#!/usr/bin/env python3

import sys

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm


"""General description"""


def train(
    train_dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device
) -> Module:
    '''
    Model training over 1 epoch

    * Don't forget to move inputs model to the device
    '''

    # turn on training mode
    model.train()

    # loop on batches
    running_loss, count = 0.0, 0
    disable_tqdm = not sys.stdout.isatty()
    desc = "Training"
    progress_bar = tqdm(train_dataloader, desc=desc, disable=disable_tqdm)
    for X_snowpack, X_sun, y in progress_bar:
        # move tensors to the device
        X_snowpack, X_sun = X_snowpack.to(device), X_sun.to(device)
        y = y.to(device)
        # predict
        y_hat = model(X_snowpack, X_sun)
        # loss
        loss = loss_function(y_hat, y)
        # gradients
        optimizer.zero_grad()
        loss.backward()
        # model weights update
        optimizer.step()
        # running loss
        batch_size = X_snowpack.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size
        # progress bar update
        progress_bar.set_postfix({"lr": f"{scheduler.get_last_lr()[0]}",
                                  "loss": f"{running_loss / count}"})

    # Update Learning rate
    scheduler.step()

    return model


def evaluate(
    dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    metric_function: Metric,
    device: torch.device,
    desc: str
) -> tuple[float, float]:
    '''
    Model evaluation

    * Don't forget to move inputs model and metric function to the device
    '''

    # turn off training mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():

        # init
        running_loss, count = 0.0, 0
        metric_function.reset()

        # loop on batches
        disable_tqdm = not sys.stdout.isatty()
        progress_bar = tqdm(dataloader, desc=desc, disable=disable_tqdm)
        for X_snowpack, X_sun, y in progress_bar:
            # move tensors to the device
            X_snowpack, X_sun = X_snowpack.to(device), X_sun.to(device)
            y = y.to(device)
            # predict
            y_hat = model(X_snowpack, X_sun)
            # loss
            loss = loss_function(y_hat, y)
            # running loss
            batch_size = X_snowpack.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size
            # update metric
            metric_function.update(preds=y_hat, target=y)

        # mean loss
        mean_loss = running_loss / count
        # compute metric
        metric = metric_function.compute().item()

    return mean_loss, metric
