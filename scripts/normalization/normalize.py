#!/usr/bin/env python3

import pandas as pd
import numpy as np


class CustomNorm1:
    """to normalize : x_new = (x - mean) / std"""

    DATA_DIR = "/home/torrenta/TARTES-Emulator/data"
    STATS_PATH = DATA_DIR + "/normalization/mean_and_std.parquet"

    def __init__(self):
        self.stats = pd.read_parquet(self.STATS_PATH)

    def normalize_dz(self, x_dz: np.ndarray) -> np.ndarray:
        mean = self.stats.dz["mean"]
        std = self.stats.dz["std"]
        return (x_dz - mean) / std

    def normalize_ssa(self, x_ssa: np.ndarray) -> np.ndarray:
        mean = self.stats.ssa["mean"]
        std = self.stats.ssa["std"]
        return (x_ssa - mean) / std

    def normalize_density(self, x_density: np.ndarray) -> np.ndarray:
        mean = self.stats.density["mean"]
        std = self.stats.density["std"]
        return (x_density - mean) / std

    def normalize_conc_soot(self, x_conc_soot: np.ndarray) -> np.ndarray:
        mean = self.stats.conc_soot["mean"]
        std = self.stats.conc_soot["std"]
        return (x_conc_soot - mean) / std

    def normalize_conc_dust(self, x_conc_dust: np.ndarray) -> np.ndarray:
        mean = self.stats.conc_dust["mean"]
        std = self.stats.conc_dust["std"]
        return (x_conc_dust - mean) / std

    def normalize_direct_sw(self, x_direct_sw: np.ndarray) -> np.ndarray:
        mean = self.stats.direct_sw["mean"]
        std = self.stats.direct_sw["std"]
        return (x_direct_sw - mean) / std

    def normalize_diffuse_sw(self, x_diffuse_sw: np.ndarray) -> np.ndarray:
        mean = self.stats.diffuse_sw["mean"]
        std = self.stats.diffuse_sw["std"]
        return (x_diffuse_sw - mean) / std


class CustomNorm2:
    """to normalize : x_new = (x - mean) / std"""

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
