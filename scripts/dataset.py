#!/usr/bin/env python3

import pandas as pd
import numpy as np

from normalization.normalize import Norm

from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from threading import Thread
from queue import Queue


class TartesDataset(IterableDataset):
    """General description"""

    sampletype = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def __init__(self, files_paths: list[str], batch_size: int,
                 buffer_size: int):
        """
        Description

        buffer_size: Number of batchs to preload
        """
        super().__init__()
        self.files_paths = files_paths
        self.norm = Norm()
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Description

        --- currently ---
        x = [dz1, ..., conc_dust50, direct_sw, diffuse_sw, cos_sza, albedo]
        """
        x[:, 0:50] = self.norm.normalize_dz(x[:, 0:50])
        x[:, 50:100] = self.norm.normalize_ssa(x[:, 50:100])
        x[:, 100:150] = self.norm.normalize_density(x[:, 100:150])
        x[:, 150:200] = self.norm.normalize_conc_soot(x[:, 150:200])
        x[:, 200:250] = self.norm.normalize_conc_dust(x[:, 200:250])
        x[:, 250] = self.norm.normalize_direct_sw(x[:, 250])
        x[:, 251] = self.norm.normalize_diffuse_sw(x[:, 251])
        return x

    @staticmethod
    def add_snow_layers(x: np.ndarray) -> np.ndarray:
        """Add a boolean value indicating the presence of snow to each layer"""
        return np.hstack((~np.isnan(x[:, 0:50]), x))

    @staticmethod
    def fill_nan(x: np.ndarray) -> np.ndarray:
        """Fill nan values with -1"""
        return np.nan_to_num(x, nan=-1)

    @staticmethod
    def build_snowpack_tensor(row: np.ndarray) -> torch.Tensor:
        """Description"""
        return torch.from_numpy(np.reshape(row[0:300], (6, 50, 1)))

    @staticmethod
    def build_sun_tensor(row: np.ndarray) -> torch.Tensor:
        """Description"""
        return torch.from_numpy(row[300:303])

    @staticmethod
    def build_albedo_tensor(row: np.ndarray) -> torch.Tensor:
        """Description"""
        return torch.from_numpy(row[303:304])

    def loader(self, queue: Queue):
        """Description"""

        for file_path in self.files_paths:

            # --- read dataframe ---
            df = pd.read_parquet(file_path)

            # --- processing data before looping through rows ---
            # suffle
            df = df.sample(axis='index', frac=1).reset_index(drop=True)
            # get numpy array (remove meta data)
            tab = df.values[:, 5:].astype(np.float64)
            # normalize
            tab = self.normalize(tab)
            # add snow layers
            tab = self.add_snow_layers(tab)
            # fill nan values
            tab = self.fill_nan(tab)
            # GPUs accept only 32 bit
            tab = tab.astype(np.float32)

            # --- loop on each row ---
            for row in tab:
                queue.put(row)

            # --- clean memory ---
            del df
            del tab

        queue.put(None)

    def __iter__(self) -> Iterator[sampletype]:
        """Description"""

        queue = Queue(maxsize=(self.batch_size * self.buffer_size))
        loader_process = Thread(target=self.loader, args=(queue, ))
        loader_process.daemon = True
        loader_process.start()

        data_available = True
        while data_available:

            row = queue.get()

            if row is None:
                data_available = False
                break

            snowpack_tensor = self.build_snowpack_tensor(row)
            sun_tensor = self.build_sun_tensor(row)
            albedo_tensor = self.build_albedo_tensor(row)
            yield snowpack_tensor, sun_tensor, albedo_tensor

        loader_process.join()
