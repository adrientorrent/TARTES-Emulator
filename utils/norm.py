#!/usr/bin/env python3

import numpy as np
import pandas as pd


"""
Tools for normalization

to normalize : x_new = (x - mean) / std

One question remains:
do I need to normalize the albedo? It's already between 0 and 1.
-> If yes, need to denormalize the output (see denormalize methods)
"""


class CustomNorm:
    """for CNN models"""

    def __init__(self, stats_path) -> None:
        """
        :stats_path: parquet stats file
        """
        self.stats = pd.read_parquet(stats_path)

        self._build_snowpack_stats()
        self._build_sun_stats()
        self._build_target_stats()

    def _build_snowpack_stats(self):
        self.snowpack_means = np.empty((300,), dtype=np.float64)
        self.snowpack_stds = np.empty((300,), dtype=np.float64)

        self.snowpack_means[0:50] = 0.
        self.snowpack_stds[0:50] = 1.

        self.snowpack_means[50:100] = self.stats["dz"]["mean"]
        self.snowpack_stds[50:100] = self.stats["dz"]["std"]

        self.snowpack_means[100:150] = self.stats["ssa"]["mean"]
        self.snowpack_stds[100:150] = self.stats["ssa"]["std"]

        self.snowpack_means[150:200] = self.stats["density"]["mean"]
        self.snowpack_stds[150:200] = self.stats["density"]["std"]

        self.snowpack_means[200:250] = self.stats["conc_soot"]["mean"]
        self.snowpack_stds[200:250] = self.stats["conc_soot"]["std"]

        self.snowpack_means[250:300] = self.stats["conc_dust"]["mean"]
        self.snowpack_stds[250:300] = self.stats["conc_dust"]["std"]

    def _build_sun_stats(self):
        self.sun_means = np.array([
            self.stats["direct_sw"]["mean"],
            self.stats["diffuse_sw"]["mean"],
            0.
        ], dtype=np.float64)
        self.sun_stds = np.array([
            self.stats["direct_sw"]["std"],
            self.stats["diffuse_sw"]["std"],
            1.
        ], dtype=np.float64)

    def _build_target_stats(self):
        self.target_mean = float(self.stats["albedo"]["mean"])
        self.target_std = float(self.stats["albedo"]["std"])
        self.target_std2 = self.target_std ** 2

    def normalize_snowpack(self, snowpack: np.ndarray) -> np.ndarray:
        """snowpack shape: (,300)"""
        nan_mask = np.isnan(snowpack)
        if nan_mask.any():
            snowpack[nan_mask] = self.snowpack_means[nan_mask]
        snowpack -= self.snowpack_means
        snowpack /= self.snowpack_stds
        return snowpack

    def normalize_sun(self, sun: np.ndarray) -> np.ndarray:
        """sun shape: (,3)"""
        sun -= self.sun_means
        sun /= self.sun_stds
        return sun

    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        """target shape: (,N)"""
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target: np.ndarray) -> np.ndarray:
        """target shape: (,N)"""
        return target * self.target_std + self.target_mean

    def denormalize_mse(self, mse: float) -> float:
        """
        DEMO

        mse = sum[(y - y_hat) ** 2]
        and
        y_norm = (y - mean) / std
        y_hat_norm = (y_hat - mean) / std

        so
        mse_norm = sum[(y_norm - y_hat_norm) ** 2]
                 = sum[(((y - mean) / std) - ((y_hat - mean) / std)) ** 2]
                 = sum[((y - y_hat - mean + mean) / std) ** 2]
                 = sum[(y - y_hat) ** 2] / (std ** 2)
                 = mse / (std ** 2)

        finally
        mse = mse_norm * (std **2)
        """
        return mse * self.target_std2


class CustomNormMlp:
    """for MLP models"""

    def __init__(self, stats_parquet_file_path):

        df = pd.read_parquet(stats_parquet_file_path)

        self.input_means = np.empty((303,))
        self.input_means[0:50] = 0.
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

        self.snowpack_means = self.input_means[0:300]
        self.sun_means = self.input_means[300:303]
        self.snowpack_stds = self.input_stds[0:300]
        self.sun_stds = self.input_stds[300:303]

    def normalize_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_means) / self.input_stds

    def normalize_target(self, x: np.ndarray) -> np.ndarray:
        return (x - self.target_mean) / self.target_std

    def normalize_snowpack(self, x: np.ndarray) -> np.ndarray:
        return (x - self.snowpack_means) / self.snowpack_stds

    def normalize_sun(self, x: np.ndarray) -> np.ndarray:
        return (x - self.sun_means) / self.sun_stds
