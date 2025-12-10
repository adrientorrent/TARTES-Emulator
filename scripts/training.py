#!/usr/bin/env python3

import sys

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm


"""Training and evaluation loops"""


def train(
    train_dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.GradScaler,
    device: torch.device
) -> Module:
    '''
    Model training over 1 epoch

    * Don't forget to move the model to the device
    '''

    # turn on training mode
    model.train()

    # loop on batches
    running_loss, count = 0.0, 0
    disable_tqdm = not sys.stdout.isatty()
    desc = "Training"
    progress_bar = tqdm(train_dataloader, desc=desc, disable=disable_tqdm)
    for X, y in progress_bar:
        # move tensors to the device
        X = X.to(device)
        y = y.to(device)
        # set gradients to 0
        optimizer.zero_grad()
        # scaler optimization
        with torch.autocast(device_type="cuda"):
            # predict
            y_hat = model(X)
            # loss
            loss = loss_function(y_hat, y)
        # gradients update
        scaler.scale(loss).backward()
        # weights update
        scaler.step(optimizer)
        scaler.update()
        # running loss
        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size
        # progress bar update
        progress_bar.set_postfix({"lr": f"{scheduler.get_last_lr()[0]}",
                                  "loss": f"{running_loss / count}"})

    # Update Learning rate
    scheduler.step()

    return model


def train_cnn(
    train_dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.GradScaler,
    device: torch.device
) -> Module:
    '''
    CNN Model training over 1 epoch

    * Don't forget to move the model to the device
    '''

    # turn on training mode
    model.train()

    # loop on batches
    running_loss, count = 0.0, 0
    disable_tqdm = not sys.stdout.isatty()
    desc = "Training"
    progress_bar = tqdm(train_dataloader, desc=desc, disable=disable_tqdm)
    for X_snow, X_sun, y in progress_bar:
        # move tensors to the device
        X_snow, X_sun = X_snow.to(device), X_sun.to(device)
        y = y.to(device)
        # set gradients to 0
        optimizer.zero_grad()
        # scaler optimization
        with torch.autocast(device_type="cuda"):
            # predict
            y_hat = model(X_snow, X_sun)
            # loss
            loss = loss_function(y_hat, y)
        # gradients update
        scaler.scale(loss).backward()
        # weights update
        scaler.step(optimizer)
        scaler.update()
        # running loss
        batch_size = X_snow.size(0)
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

    * Don't forget to the model and the metric function to the device
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
        for X, y in progress_bar:
            # move tensors to the device
            X = X.to(device)
            y = y.to(device)
            # predict
            y_hat = model(X)
            # loss
            loss = loss_function(y_hat, y)
            # running loss
            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size
            # update metric
            metric_function.update(preds=y_hat, target=y)

        # mean loss
        mean_loss = running_loss / count
        # compute metric
        metric = metric_function.compute().item()

    return mean_loss, metric


def evaluate_cnn(
    dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    metric_function: Metric,
    device: torch.device,
    desc: str
) -> tuple[float, float]:
    '''
    CNN Model evaluation

    * Don't forget to the model and the metric function to the device
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

        # mean loss
        mean_loss = running_loss / count
        # compute metric
        metric = metric_function.compute().item()

    return mean_loss, metric
