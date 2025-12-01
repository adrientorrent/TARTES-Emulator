#!/usr/bin/env python3

import sys
import time
import logging

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from tqdm import tqdm


"""General description"""


def evaluate(
    dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    metric_function: MeanSquaredError,
    device: torch.device,
    desc: str
) -> tuple[float, float]:
    '''
    Model evaluation

    * Don't forget to move inputs model and metric function to the device
    ** Model must be in evaluation mode
    '''

    # init
    running_loss, count = 0.0, 0
    metric_function.reset()

    # loop on batches
    disable_tqdm = not sys.stdout.isatty()
    for X_snowpack, X_sun, y in tqdm(dataloader, desc=desc,
                                     disable=disable_tqdm):
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


def train(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: Module,
    loss_function: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    metric_function: MeanSquaredError,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> Module:
    '''
    Model training over 1 epoch

    * Don't forget to move inputs model and metric function to the device
    '''

    start_time = time.time()

    # --- LEARNING ---
    logger.info("Learning")

    # turn on training mode
    model.train()

    # loop on batches
    running_loss, count = 0.0, 0
    disable_tqdm = not sys.stdout.isatty()
    progress_bar = tqdm(train_dataloader, desc="Training",
                        disable=disable_tqdm)
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
        progress_bar.set_postfix({"loss": f"{running_loss / count}",
                                  "lr": f"{scheduler.get_last_lr()[0]}"})

    # Update Learning rate
    scheduler.step()

    # --- EVALUATION ---
    logger.info("Evaluation")

    # turn off training mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        # Evaluation based on training data
        # train_loss, train_metric = evaluate(
        #     dataloader=train_dataloader,
        #     model=model,
        #     loss_function=loss_function,
        #     metric_function=metric_function,
        #     device=device,
        #     desc="Evaluation on training data"
        # )
        # Evaluation based on testing data
        test_loss, test_metric = evaluate(
            dataloader=test_dataloader,
            model=model,
            loss_function=loss_function,
            metric_function=metric_function,
            device=device,
            desc="Evaluation on testing data"
        )

    # --- RESULTS ---

    elapsed_time = time.time() - start_time
    print(f"--- Epoch {epoch} | Elapsed Time : {elapsed_time:.2f} s ---")
    # print(f"Train Loss: {train_loss}")
    # print(f"Train Metric: {train_metric}")
    print(f"Test Loss: {test_loss}")
    print(f"Test Metric: {test_metric}")

    return model
