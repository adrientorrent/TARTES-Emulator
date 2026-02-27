#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from utils.norm import CustomNorm

"""
Same as the main dataset.py file, with Pandas DataFrame input
-> usefull if I need only specific situations, and not the all years
-> Exemple of use : to plot the year and day profiles of albedo
"""


class CustomDataFrameDataset(Dataset):
    """
    Custom pytorch iterable dataset with pandas DataFrame

    set num_workers = 0
    """

    meta_cols = ["time", "ZS", "aspect", "slope", "massif_number"]
    snowpack_cols = [
        f"{k}{i + 1}"
        for k in ["snow_layer", "dz", "ssa", "density",
                  "conc_soot", "conc_dust"]
        for i in range(50)
    ]
    sun_cols = ["direct_sw", "diffuse_sw", "cos_sza"]
    target_cols = ["albedo"]

    def __init__(self, df: pd.DataFrame, norm: CustomNorm):
        """
        :df: DataFrame
        :norm: normalization operator
        """
        super().__init__()

        self.df = df
        self.norm = norm

        self.X_snow, self.X_sun, self.y = self._load_data()

    def _load_data(self):

        # load data
        snowpack = self.df[self.snowpack_cols].astype(np.float64).values
        sun = self.df[self.sun_cols].astype(np.float64).values
        target = self.df[self.target_cols].astype(np.float64).values

        # num of samples
        n_samples = len(snowpack)

        # new data
        thickness = np.zeros((n_samples, 1))

        # for each sample
        for i in range(n_samples):

            # compute thickness
            total_thickness = np.nansum(snowpack[i][50:100])
            thickness[i] = min(1., total_thickness)

            # normalization
            snowpack[i] = self.norm.normalize_snowpack(snowpack[i])
            sun[i] = self.norm.normalize_sun(sun[i])
            # target[i] = self.norm.normalize_target(target[i])

        # add new data
        sun = np.hstack((sun, thickness))

        return (
            torch.from_numpy(snowpack.astype(np.float32)),
            torch.from_numpy(sun.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X_snow[idx].reshape((6, 50)), self.X_sun[idx], self.y[idx]
