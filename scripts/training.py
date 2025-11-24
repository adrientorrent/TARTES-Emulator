#!/usr/bin/env python3

import time
import logging

import torch
from torch.nn import Module, MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from tqdm import tqdm


"""General description"""


def evaluate(
    dataloader: DataLoader,
    model: Module,
    loss_function: MSELoss,
    metric: MeanSquaredError,
    device: torch.device,
    desc: str,
    logger: logging.Logger
) -> tuple[float, float]:
    '''
    Model evaluation

    * Don't forget to move inputs model and metric to the device
    ** Model must be in evaluation mode
    '''

    # init
    running_loss, count = 0.0, 0
    metric.reset()

    # loop on batches
    for X_snowpack, X_sun, y in tqdm(dataloader, desc=desc):
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
        metric.update(preds=y_hat, target=y)

        if logger.level == logging.DEBUG:
            break

    # mean loss
    mean_loss = running_loss / count
    # compute metric
    mse = metric.compute().item()

    return mean_loss, mse


def train(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: Module,
    loss_function: MSELoss,
    optimizer: Optimizer,
    metric: MeanSquaredError,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> Module:
    '''
    Model training over 1 epoch

    * Don't forget to move inputs model and metric to the device
    '''

    start_time = time.time()

    # --- LEARNING ---

    # turn on training mode
    model.train()

    # loop on batches
    running_loss, count = 0.0, 0
    progress_bar = tqdm(train_dataloader, desc="Training")
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
        progress_bar.set_postfix({"mse": f"{(running_loss / count):.4f}"})

        if logger.level == logging.DEBUG:
            break

    # --- EVALUATION ---

    # turn off training mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        # Evaluation based on training data
        train_loss, train_metric = evaluate(
            dataloader=train_dataloader,
            model=model,
            loss_function=loss_function,
            metric=metric,
            device=device,
            desc="Evaluation on training data",
            logger=logger
        )
        # Evaluation based on testing data
        test_loss, test_metric = evaluate(
            dataloader=test_dataloader,
            model=model,
            loss_function=loss_function,
            metric=metric,
            device=device,
            desc="Evaluation on testing data",
            logger=logger
        )

    # --- RESULTS ---

    elapsed_time = time.time() - start_time
    print(f"--- Epoch {epoch} | Elapsed Time : {elapsed_time:.2f} secondes ---")
    print(f"Train Loss : {train_loss:.5f}")
    print(f"Train Metric : {train_metric:.5f}")
    print(f"Test Loss : {test_loss:.5f}")
    print(f"Test Metric : {test_metric:.5f}")

    return model
