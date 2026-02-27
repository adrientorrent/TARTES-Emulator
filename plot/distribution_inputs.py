#!/usr/bin/env python3

import sys
import logging

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


plt.style.use("/home/torrenta/new-TARTES-Emulator/plot/_rcparams.mplstyle")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.debug("Files selection")
    parquet_dir = "/bigdata/BIGDATA/torrenta/Alpes/Alpes_1979_1980"
    files = []
    for i in [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]:
        files.append(f"{parquet_dir}/Alpes_1979_1980_{i}.parquet")

    logger.debug("Build DataFrame")
    cols = ["conc_dust1"]
    df = pd.concat(
        (pd.read_parquet(f, engine="pyarrow", columns=cols) for f in files),
        ignore_index=True
    )

    logger.debug("Gathering plot data")
    dust = list(df["conc_dust1"])
    logbins = list(np.logspace(-21, 0, 200))

    logger.debug("Plot")
    fig, ax = plt.subplots()
    ax.hist(dust, bins=logbins)
    ax.set_xscale("log")
    plt.show()

    sys.exit(0)
