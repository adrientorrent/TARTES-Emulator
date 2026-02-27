#!/usr/bin/env python3

import sys
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np

from utils.dataframe_dataset import CustomDataFrameDataset
from utils.norm import CustomNorm
from models import CnnTartesEmulator

import torch
from torch.utils.data import DataLoader


plt.style.use("/home/torrenta/new-TARTES-Emulator/plot/_rcparams.mplstyle")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # ======================================================================= #
    logger.debug("Files selection")
    parquet_dir = "/bigdata/BIGDATA/torrenta/Alpes/Alpes_1979_1980"
    files = []
    for i in [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]:
        files.append(f"{parquet_dir}/Alpes_1979_1980_{i}.parquet")

    # ======================================================================= #
    logger.debug("Build DataFrame")
    df = pd.concat(
        (pd.read_parquet(
            f,
            engine="pyarrow",
            filters=[
                ("massif_number", "==", 3),
                ("ZS", "==", 2700),
                ("slope", "==", 0),
                ("aspect", "==", -1),
                ("time", ">=", "1980-02-01"),
                ("time", "<", "1980-02-08")
            ]
        ) for f in files),
        ignore_index=True
    )

    # ======================================================================= #
    logger.debug("Set norm")
    data_dir = "/home/torrenta/new-TARTES-Emulator/data"
    mean_and_std_path = f"{data_dir}/mean_and_std.parquet"
    norm = CustomNorm(mean_and_std_path)

    # ======================================================================= #
    logger.debug("Creating dataset and dataloader")
    dataset = CustomDataFrameDataset(df, norm)
    dataloader = DataLoader(dataset=dataset, batch_size=512, drop_last=False)

    # ======================================================================= #
    logger.debug("Build model")
    model = CnnTartesEmulator()
    model_path = f"{data_dir}/model.pt"
    model_state = torch.load(model_path, weights_only=True)["model_state"]
    model.load_state_dict(model_state)

    # ======================================================================= #
    logger.debug("GPU device")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ======================================================================= #
    logger.debug("Predictions")
    predictions = []
    disable = not sys.stdout.isatty()
    progress_bar = tqdm(dataloader, desc="Predict", disable=disable)
    for X_snow, X_sun, y in progress_bar:
        X_snow, X_sun = X_snow.to(device), X_sun.to(device)
        y_hat = model(X_snow, X_sun).detach().cpu()
        for pred in y_hat:
            predictions.append(pred[0])

    # ======================================================================= #
    logger.debug("Gathering plot data")
    time = list(pd.to_datetime(df["time"]))
    albedo = list(df["albedo"])

    # ======================================================================= #
    logger.debug("Compute RMSE")
    rmse = np.sqrt(np.mean((np.array(albedo) - np.array(predictions))**2))
    # print(rmse)

    # ======================================================================= #
    logger.debug("Add NaN values when it's night")
    plot_time = [time[0]]
    plot_alb = [albedo[0]]
    plot_pred = [predictions[0]]
    for i in range(1, len(time)):
        while True:
            if time[i] - plot_time[-1] > np.timedelta64(1, "h"):
                plot_time.append(plot_time[-1] + np.timedelta64(1, "h"))
                plot_alb.append(np.nan)
                plot_pred.append(np.nan)
            else:
                plot_time.append(time[i])
                plot_alb.append(albedo[i])
                plot_pred.append(predictions[i])
                break

    # ======================================================================= #
    logger.debug("Plot")

    fig, ax = plt.subplots()
    ax.plot(np.array(plot_time), plot_alb, color="red")
    ax.plot(np.array(plot_time), plot_pred, color="blue", dashes=[2, 2])
    ax.set_xlabel("Date")
    ax.set_ylabel("Broadband albedo")
    ax.legend(["TARTES", "Emulator"])
    ax.scatter(np.array(plot_time), plot_alb, color="red")
    ax.scatter(np.array(plot_time), plot_pred, color="blue")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.show()

    sys.exit(0)
