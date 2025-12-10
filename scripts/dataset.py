#!/usr/bin/env python3

from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

import numpy as np
import pyarrow.parquet as pq

from normalization.normalize import CustomNorm2


class CnnTartesIterableDataset(IterableDataset):
    """Custom pytorch iterable dataset for CNN model"""

    sampletype = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def __init__(self, files: list[str], norm: CustomNorm2) -> None:
        """
        :files: list of parquet files paths
        :norm: custom normalization object
        """

        super().__init__()

        self.files = files
        self.norm = norm

        self.meta_cols = ["time", "ZS", "aspect", "slope", "massif_number"]
        self.snowpack_cols = [
             f"{k}{i + 1}"
             for k in ["snow_layer", "dz", "ssa", "density",
                       "conc_soot", "conc_dust"]
             for i in range(50)
        ]
        self.sun_cols = ["direct_sw", "diffuse_sw", "cos_sza"]
        self.target_cols = ["albedo"]

        self.all_files_row_groups: list[tuple[str, int]] = []
        for fp in files:
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                self.all_files_row_groups.append((fp, rg))

    @staticmethod
    def build_snowpack_tensor(row: np.ndarray) -> torch.Tensor:
        """snowpack data numpy array to torch tensor"""
        return torch.from_numpy(np.reshape(row[0:300], (6, 50)))

    @staticmethod
    def build_sun_tensor(row: np.ndarray) -> torch.Tensor:
        """sun data numpy array to torch tensor"""
        return torch.from_numpy(row[300:303])

    @staticmethod
    def build_albedo_tensor(row: np.ndarray) -> torch.Tensor:
        """Albedo numpy array to torch tensor"""
        return torch.from_numpy(row[303:304])

    def loader(self, file_path: str, row_group: int) -> sampletype:
        """Files loader method"""

        pf = pq.ParquetFile(file_path)

        snowpack_table = pf.read_row_group(row_group,
                                           columns=self.snowpack_cols)
        sun_table = pf.read_row_group(row_group, columns=self.sun_cols)
        target_table = pf.read_row_group(row_group, columns=self.target_cols)

        snowpack_arr = snowpack_table.to_pandas().astype(np.float64).values
        sun_arr = sun_table.to_pandas().astype(np.float64).values
        target_arr = target_table.to_pandas().astype(np.float64).values

        for i in range(len(snowpack_arr)):
            nan_mask = np.isnan(snowpack_arr[i])
            if nan_mask.any():
                snowpack_arr[i][nan_mask] = self.norm.snowpack_means[nan_mask]
            snowpack_arr[i] = self.norm.normalize_snowpack(snowpack_arr[i])
            sun_arr[i] = self.norm.normalize_sun(sun_arr[i])
            # target_arr[i] = self.norm.normalize_target(target_arr[i])

        snowpack_tensor = torch.from_numpy(snowpack_arr.astype(np.float32))
        sun_tensor = torch.from_numpy(sun_arr.astype(np.float32))
        target_tensor = torch.from_numpy(target_arr.astype(np.float32))

        return snowpack_tensor, sun_tensor, target_tensor

    def __iter__(self) -> Iterator[sampletype]:
        """Custom __iter__ method"""

        worker_info = get_worker_info()
        worker_files_row_groups: list[tuple[str, int]]

        if worker_info is None:
            worker_files_row_groups = self.all_files_row_groups
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(
                np.ceil(len(self.all_files_row_groups) / num_workers)
            )
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.all_files_row_groups))
            worker_files_row_groups = self.all_files_row_groups[start:end]

        for k in range(len(worker_files_row_groups)):
            fp, rg = worker_files_row_groups[k]
            tab_snowpack, tab_sun, tab_target = self.loader(
                file_path=fp, row_group=rg
            )
            for row in range(len(tab_snowpack)):
                out1 = tab_snowpack[row].reshape((6, 50))
                out2 = tab_sun[row]
                out3 = tab_target[row]
                yield out1, out2, out3


class MlpTartesIterableDataset(IterableDataset):
    """Custom pytorch iterable dataset for MLP model"""

    sampletype = tuple[torch.Tensor, torch.Tensor]

    def __init__(self, files: list[str], norm: CustomNorm2) -> None:
        """
        :files: list of parquet files paths
        :norm: custom normalization object
        """

        super().__init__()

        self.files = files
        self.norm = norm

        self.meta_cols = ["time", "ZS", "aspect", "slope", "massif_number"]
        self.input_cols = [
             f"{k}{i + 1}"
             for k in ["snow_layer", "dz", "ssa", "density",
                       "conc_soot", "conc_dust"]
             for i in range(50)
        ] + ["direct_sw", "diffuse_sw", "cos_sza"]
        self.target_cols = ["albedo"]

        self.all_files_row_groups: list[tuple[str, int]] = []
        for fp in files:
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                self.all_files_row_groups.append((fp, rg))

    def loader(self, file_path: str, row_group: int) -> sampletype:
        """Files loader method"""

        pf = pq.ParquetFile(file_path)

        inputs_table = pf.read_row_group(row_group, columns=self.input_cols)
        target_table = pf.read_row_group(row_group, columns=self.target_cols)

        inputs_arr = inputs_table.to_pandas().astype(np.float64).values
        target_arr = target_table.to_pandas().astype(np.float64).values

        for i in range(len(inputs_arr)):
            nan_mask = np.isnan(inputs_arr[i])
            if nan_mask.any():
                inputs_arr[i][nan_mask] = self.norm.input_means[nan_mask]
            inputs_arr[i] = self.norm.normalize_inputs(inputs_arr[i])
            # target_arr[i] = self.norm.normalize_target(target_arr[i])

        inputs_tensor = torch.from_numpy(inputs_arr.astype(np.float32))
        target_tensor = torch.from_numpy(target_arr.astype(np.float32))

        return inputs_tensor, target_tensor

    def __iter__(self) -> Iterator[sampletype]:
        """Custom __iter__ method"""

        worker_info = get_worker_info()
        worker_files_row_groups: list[tuple[str, int]]

        if worker_info is None:
            worker_files_row_groups = self.all_files_row_groups
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(
                np.ceil(len(self.all_files_row_groups) / num_workers)
            )
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.all_files_row_groups))
            worker_files_row_groups = self.all_files_row_groups[start:end]

        for k in range(len(worker_files_row_groups)):
            fp, rg = worker_files_row_groups[k]
            tab_inputs, tab_target = self.loader(file_path=fp, row_group=rg)
            for row in range(len(tab_inputs)):
                yield tab_inputs[row], tab_target[row]
