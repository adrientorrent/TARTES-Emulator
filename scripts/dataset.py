#!/usr/bin/env python3

import numpy as np
import polars as pl

from normalization.normalize import Norm

from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


class TartesDataset(IterableDataset):
    """General description"""

    sampletype = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def __init__(self, files_paths: list[str]):
        """Description"""
        super().__init__()
        self.files_paths = files_paths
        self.norm = Norm()

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
        return torch.from_numpy(np.reshape(row[0:300], (6, 50)))

    @staticmethod
    def build_sun_tensor(row: np.ndarray) -> torch.Tensor:
        """Description"""
        return torch.from_numpy(row[300:303])

    @staticmethod
    def build_albedo_tensor(row: np.ndarray) -> torch.Tensor:
        """Description"""
        return torch.from_numpy(row[303:304])

    def loader(self, file_path: str) -> np.ndarray:
        """Description"""

        # read dataframe
        df = pl.read_parquet(file_path)
        # shuffle
        df = df.sample(fraction=1.0, shuffle=True)
        # convert to numpy array
        arr = df[:, 5:].to_numpy().astype(np.float64)
        # normalize
        arr = self.normalize(arr)
        # add snow layers
        arr = self.add_snow_layers(arr)
        # fill nan values
        arr = self.fill_nan(arr)
        # GPUs accept only 32 bit
        arr = arr.astype(np.float32)

        return arr

    def __iter__(self) -> Iterator[sampletype]:
        """Description"""

        worker_info = get_worker_info()
        worker_files_path: list[str]

        if worker_info is None:
            worker_files_path = self.files_paths
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(np.ceil(len(self.files_paths) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files_paths))
            worker_files_path = self.files_paths[start:end]

        for file_path in worker_files_path:
            arr = self.loader(file_path)
            for row in arr:
                snowpack_tensor = self.build_snowpack_tensor(row)
                sun_tensor = self.build_sun_tensor(row)
                albedo_tensor = self.build_albedo_tensor(row)
                yield snowpack_tensor, sun_tensor, albedo_tensor
