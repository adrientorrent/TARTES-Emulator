#!/usr/bin/env python3

import matplotlib.pyplot as plt

import pandas as pd


plt.style.use("/home/torrenta/new-TARTES-Emulator/plot/_rcparams.mplstyle")


if __name__ == "__main__":

    fig, ax = plt.subplots()

    dir = "/home/torrenta/new-TARTES-Emulator/data"
    df = pd.read_csv(dir+"/results.csv", delimiter=",")

    # ax.plot(df["epoch"], df["train_loss"], color="red")
    # ax.plot(df["epoch"], df["test_loss"], color="blue")
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("MSE")

    ax.plot(df["epoch"], df["train_metric"], color="red")
    ax.plot(df["epoch"], df["test_metric"], color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")

    ax_bis = ax.twinx()
    ax_bis.plot(df["epoch"], df["learning_rate"], color="green")
    ax_bis.set_ylabel("Learning Rate")

    ax.legend(["train", "test"])
    plt.show()
