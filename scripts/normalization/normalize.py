#!/usr/bin/env python3

import pandas as pd
import numpy as np


class Norm:
    """
    Description
    to normalize : x_new = (x - mean) / std
    """

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
