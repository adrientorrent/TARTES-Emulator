#!/usr/bin/env python3

import os
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
import pyarrow.parquet as pq

from utils.norm import CustomNorm


"""
This file define the current dataset use.

Go to utils/iterable_dataset.py for :
1. The same dataset with the IterableDataset class.
   -> useful with big datasets, too big to be load in one time in RAM
2. The dataset class for mlp models.
"""


class CnnCustomDataset(Dataset):
    """
    Custom pytorch dataset for CNN model
    with preprocessing cache (.pt)

    set num_workers = 0 when building cache
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

    def __init__(self, files: list[str], norm: CustomNorm, cache_path: str,
                 force_rebuild: bool = False):
        """
        :files: parquet files
        :norm: normalization operator
        :cache_path: path to .pt cache file
        :force_rebuild: ignore cache and rebuild
        """
        super().__init__()

        self.cache_path = cache_path
        if os.path.exists(cache_path) and not force_rebuild:
            self._load_cache()
        else:
            self.files = files
            self.norm = norm
            self._build_dataset()
            self._save_cache()

    def _save_cache(self):
        torch.save({
            "X_snow": self.X_snow,
            "X_sun": self.X_sun,
            "y": self.y
        }, self.cache_path)

    def _load_cache(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        data = torch.load(self.cache_path, map_location="cpu")
        self.X_snow = data["X_snow"]
        self.X_sun = data["X_sun"]
        self.y = data["y"]

    def _build_dataset(self):
        X_snow, X_sun, y = [], [], []
        for fp in self.files:
            pf = pq.ParquetFile(fp)
            for rg in tqdm(range(pf.num_row_groups), desc=os.path.basename(fp)):
                snowpack, sun, target = self._load_row_group(pf, rg)
                for i in range(len(snowpack)):
                    X_snow.append(snowpack[i].reshape(6, 50))
                    X_sun.append(sun[i])
                    y.append(target[i])
        self.X_snow = torch.stack(X_snow)
        self.X_sun = torch.stack(X_sun)
        self.y = torch.stack(y)

    def _load_row_group(self, pf, rg):

        # load data
        snowpack = pf.read_row_group(
            rg, columns=self.snowpack_cols
        ).to_pandas().astype(np.float64).values
        sun = pf.read_row_group(
            rg, columns=self.sun_cols
        ).to_pandas().astype(np.float64).values
        target = pf.read_row_group(
            rg, columns=self.target_cols
        ).to_pandas().astype(np.float64).values

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
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_snow[idx], self.X_sun[idx], self.y[idx]
