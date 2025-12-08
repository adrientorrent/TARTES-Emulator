#!/usr/bin/env python3

import pandas as pd
import numpy as np


class CustomNorm:
    """
    Normalization class
    to normalize : x_new = (x - mean) / std
    """

    def __init__(self, stats_parquet_file_path):

        df = pd.read_parquet(stats_parquet_file_path)

        self.input_means = np.empty((303,))
        self.input_means[0:50] = 1.
        self.input_means[50:100] = df["dz"]["mean"]
        self.input_means[100:150] = df["ssa"]["mean"]
        self.input_means[150:200] = df["density"]["mean"]
        self.input_means[200:250] = df["conc_soot"]["mean"]
        self.input_means[250:300] = df["conc_dust"]["mean"]
        self.input_means[300] = df["direct_sw"]["mean"]
        self.input_means[301] = df["diffuse_sw"]["mean"]
        self.input_means[302] = df["cos_sza"]["mean"]

        self.input_stds = np.empty((303,))
        self.input_stds[0:50] = 1.
        self.input_stds[50:100] = df["dz"]["std"]
        self.input_stds[100:150] = df["ssa"]["std"]
        self.input_stds[150:200] = df["density"]["std"]
        self.input_stds[200:250] = df["conc_soot"]["std"]
        self.input_stds[250:300] = df["conc_dust"]["std"]
        self.input_stds[300] = df["direct_sw"]["std"]
        self.input_stds[301] = df["diffuse_sw"]["std"]
        self.input_stds[302] = df["cos_sza"]["std"]

        self.target_mean = df["albedo"]["mean"]
        self.target_std = df["albedo"]["std"]

    def normalize_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_means) / self.input_stds

    def normalize_target(self, x: np.ndarray) -> np.ndarray:
        return (x - self.target_mean) / self.target_std
