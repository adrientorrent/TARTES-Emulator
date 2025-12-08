#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, get_worker_info

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

from normalization.new_normalize import CustomNorm


class NewTartesDataset(Dataset):
    """Custom pytorch dataset"""

    def __init__(self, files: list[str], norm: CustomNorm) -> None:
        """
        :files: list of parquet files paths
        :norm: custom normalization object
        """

        super().__init__()

        self.parquet_files = files
        self.norm = norm

        self.meta_cols = ["time", "ZS", "aspect", "slope", "massif_number"]
        self.input_cols = [
             f"{k}{i + 1}"
             for k in ["snow_layer", "dz", "ssa", "density",
                       "conc_soot", "conc_dust"]
             for i in range(50)
        ] + ["direct_sw", "diffuse_sw", "cos_sza"]
        self.target_cols = ["albedo"]

        self.file_row_counts = []
        self.file_row_group_counts = []
        self.file_row_group_offsets = []

        total = 0
        for fp in files:
            pf = pq.ParquetFile(fp)
            num_rows = pf.metadata.num_rows
            self.file_row_counts.append(num_rows)

            rg_counts = []
            offsets = []
            cum = 0
            for rg in range(pf.num_row_groups):
                rg_meta = pf.metadata.row_group(rg)
                rg_num_rows = rg_meta.num_rows
                rg_counts.append(rg_num_rows)
                offsets.append(cum)
                cum += rg_num_rows
            self.file_row_group_counts.append(rg_counts)
            self.file_row_group_offsets.append(offsets)

            total += num_rows

        self.cumsum_file_rows = np.cumsum([0] + self.file_row_counts)
        self.total_len = int(total)

        self._worker_cache: dict[int, dict] = {}

    def __len__(self):
        return self.total_len

    def _global_idx_to_file_row(self, idx: int) -> tuple[int, int]:
        """Returns the file index and the row index from the global index"""
        file_idx = int(
            np.searchsorted(self.cumsum_file_rows, idx, side="right") - 1
        )
        row_in_file = int(idx - self.cumsum_file_rows[file_idx])
        return file_idx, row_in_file

    def _row_in_file_to_row_group(
            self, file_idx: int, row_in_file: int
    ) -> tuple[int, int]:
        """
        Returns the group index and the offset within from
        the file index and the row index
        """
        offsets = self.file_row_group_offsets[file_idx]
        rg_idx = int(np.searchsorted(offsets, row_in_file, side="right") - 1)
        if rg_idx < 0:
            rg_idx = 0
        offset_within = row_in_file - offsets[rg_idx]
        return rg_idx, offset_within

    def _load_row_group(self, file_idx: int, rg_idx: int):
        """Load/Update worker cache"""

        worker = get_worker_info()
        wid = 0 if worker is None else worker.id

        fp = self.parquet_files[file_idx]
        pqf = pq.ParquetFile(fp)
        cols = self.input_cols + self.target_cols

        table = pqf.read_row_group(rg_idx, columns=cols)
        self._worker_cache[wid] = {
            "file_idx": file_idx,
            "rg_idx": rg_idx,
            "table": table
        }

    def __getitem__(self, idx: int):
        """Custom __getitem__ method"""

        file_idx, row_in_file = self._global_idx_to_file_row(idx)
        rg_idx, offset_within = self._row_in_file_to_row_group(
            file_idx, row_in_file
        )

        worker = get_worker_info()
        wid = 0 if worker is None else worker.id

        cache = self._worker_cache.get(wid)
        if cache is None \
           or cache.get("file_idx") != file_idx \
           or cache.get("rg_idx") != rg_idx:
            self._load_row_group(file_idx, rg_idx)
            cache = self._worker_cache[wid]

        table: pa.Table = cache["table"]

        x_arr = np.empty((len(self.input_cols),))
        for i, icol in enumerate(self.input_cols):
            col = table.column(icol)
            x_arr[i] = col[offset_within].as_py()

        y_arr = np.empty((len(self.target_cols),))
        for i, icol in enumerate(self.target_cols):
            col = table.column(icol)
            y_arr[i] = col[offset_within].as_py()

        nan_mask = np.isnan(x_arr)
        if nan_mask.any():
            x_arr[nan_mask] = self.norm.input_means[nan_mask]

        x_norm = self.norm.normalize_inputs(x_arr)
        y_norm = self.norm.normalize_target(y_arr)

        x_tensor = torch.from_numpy(x_norm.astype(np.float32))
        y_tensor = torch.from_numpy(y_norm.astype(np.float32))

        return x_tensor, y_tensor
