#!/usr/bin/env python3

import torch
from torch.utils.data import IterableDataset, get_worker_info

import numpy as np
import pyarrow.parquet as pq

from utils.norm import CustomNorm, CustomNormMlp


"""
Same as the main dataset.py file, with IterableDataset class
-> useful with big datasets, too big to be load in one time in RAM

And MlpTartesIterableDataset is for MLP models.
"""


class CnnCustomIterableDataset(IterableDataset):
    """
    Custom pytorch iterable dataset for CNN model

    set num_workers > 0 to be efficient
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

    def __init__(self, files: list[str], norm: CustomNorm):
        """
        :files: parquet files
        :norm: normalization operator
        """
        super().__init__()

        self.files = files
        self.norm = norm

        self.all_files_row_groups: list[tuple[str, int]] = []
        for fp in files:
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                self.all_files_row_groups.append((fp, rg))

    def _load_row_group(self, fp: str, rg: int):

        # load data
        pf = pq.ParquetFile(fp)
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

    def __iter__(self):
        """Custom __iter__ method"""

        # worker info
        worker_info = get_worker_info()
        worker_files_row_groups: list[tuple[str, int]]

        # set worker dataset
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

        # iterate on each sample
        for k in range(len(worker_files_row_groups)):
            fp, rg = worker_files_row_groups[k]
            snowpack, sun, target = self._load_row_group(fp, rg)
            for row in range(len(snowpack)):
                yield snowpack[row].reshape((7, 50)), sun[row], target[row]


class MlpTartesIterableDataset(IterableDataset):
    """Custom pytorch iterable dataset for MLP model"""

    sampletype = tuple[torch.Tensor, torch.Tensor]

    def __init__(self, files: list[str], norm: CustomNormMlp) -> None:
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

    def __iter__(self):
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
