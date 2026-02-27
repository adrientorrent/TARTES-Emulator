#!/usr/bin/env python3

import sys

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm


"""
Training and evaluation loops

The models take 2 inputs, so : y_hat = model(X1, X2)
-> If you want to train the mlp model :
   you need to change the code to have only 1 input : y_hat = model(X)
"""


def train_loop(
    train_dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    optimizer: Optimizer,
    device: torch.device
) -> Module:
    '''
    CNN Model training over 1 epoch

    * Don't forget to move the model to the device before
    '''

    # turn on training mode
    model.train()

    # init running loss
    running_loss, count = 0.0, 0

    # loop on batches
    disable = not sys.stdout.isatty()
    desc = "Training"
    progress_bar = tqdm(train_dataloader, desc=desc, disable=disable, ncols=140)
    for X_snow, X_sun, y in progress_bar:
        # move tensors to the device
        X_snow, X_sun = X_snow.to(device), X_sun.to(device)
        y = y.to(device)
        # predict
        y_hat = model(X_snow, X_sun)
        # loss
        loss = loss_function(y_hat, y)
        # gradients update
        optimizer.zero_grad()
        loss.backward()
        # weights update
        optimizer.step()
        # running loss
        batch_size = X_snow.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size
        # progress bar update
        progress_bar.set_postfix({"loss": f"{running_loss / count}"})

    return model


def evaluate_loop(
    dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    metric_function: Metric,
    device: torch.device
) -> tuple[float, float]:
    '''
    CNN Model evaluation

    * Don't forget to the model and the metric function to the device before
    '''

    # turn off training mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():

        # init running loss
        running_loss, count = 0.0, 0
        metric_function.reset()

        # loop on batches
        disable = not sys.stdout.isatty()
        desc = "Evaluation"
        progress_bar = tqdm(dataloader, desc=desc, disable=disable, ncols=140)
        for X_snow, X_sun, y in progress_bar:
            # move tensors to the device
            X_snow, X_sun = X_snow.to(device), X_sun.to(device)
            y = y.to(device)
            # predict
            y_hat = model(X_snow, X_sun)
            # loss
            loss = loss_function(y_hat, y)
            # running loss
            batch_size = X_snow.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size
            # update metric
            metric_function.update(preds=y_hat, target=y)

        mean_loss = running_loss / count
        metric = metric_function.compute().item()

    return mean_loss, metric
