#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
from normalization.normalize import Normalizer


class TartesDataset(IterableDataset):

    def __init__(self, parquet_files_groupby_year):
        super().__init__()
        self.files = parquet_files_groupby_year
        self.normalizer = Normalizer()

    def get_snowpack(self, raw_sample):
        # build and normalize snowpack image
        snowpack = np.reshape(raw_sample[0:250], (5, 50, 1))
        snowpack = np.nan_to_num(snowpack, nan=-1)
        snow_layers = ~np.isnan([[[raw_sample[i]] for i in range(50)]])
        snowpack = np.vstack((snowpack, snow_layers))
        snowpack = self.normalizer.normalize_snowpack(snowpack)
        snowpack_tensor = torch.from_numpy(snowpack).float()
        return snowpack_tensor

    def get_sun(self, raw_sample):
        # build and normalize sun data
        sun = np.array(raw_sample[250:253])
        sun[0] = self.normalizer.normalize_direct_sw(sun[0])
        sun[1] = self.normalizer.normalize_diffuse_sw(sun[1])
        sun_tensor = torch.from_numpy(sun).float()
        return sun_tensor

    def get_albedo(self, raw_sample):
        # get albedo label
        return torch.tensor([raw_sample[-1]]).float()

    def __iter__(self):
        for year in self.files:
            for file in year:
                # read dataframe
                df = pd.read_parquet(file)
                # and shuffle df
                df = df.sample(axis='index', frac=1).reset_index(drop=True)
                for idx in range(len(df)):
                    # get row
                    row = df.iloc[idx].values
                    # snowpack
                    snowpack_tensor = self.get_snowpack(row)
                    # sun
                    sun_tensor = self.get_sun(row)
                    # albedo broadband
                    albedo_tensor = self.get_albedo(row)
                    # yield sample
                    yield snowpack_tensor, sun_tensor, albedo_tensor
                # clean ram
                del df
