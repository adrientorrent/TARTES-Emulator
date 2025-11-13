# to normalize : x_new = (x - mean) / std
# use standard scaler from scikit-learn

import pandas as pd
import numpy as np

class Normalizer:

    df_stats_path = "/home/torrenta/TARTES-Emulator/data/normalization/mean_and_std.parquet"

    def __init__(self):
        self.stats = pd.read_parquet(self.df_stats_path)

    def _normalize_dz(self, x_dz):
        return (x_dz - self.stats.dz["mean"]) / self.stats.dz["std"]

    def _normalize_ssa(self, x_ssa):
        return (x_ssa - self.stats.ssa["mean"]) / self.stats.ssa["std"]

    def _normalize_density(self, x_density):
        return (x_density - self.stats.density["mean"]) / self.stats.density["std"]

    def _normalize_conc_soot(self, x_conc_soot):
        return (x_conc_soot - self.stats.conc_soot["mean"]) / self.stats.conc_soot["std"]

    def _normalize_conc_dust(self, x_conc_dust):
        return (x_conc_dust - self.stats.conc_dust["mean"]) / self.stats.conc_dust["std"]
    
    def normalize_direct_sw(self, x_direct_sw):
        return (x_direct_sw - self.stats.direct_sw["mean"]) / self.stats.direct_sw["std"]

    def normalize_diffuse_sw(self, x_diffuse_sw):
        return (x_diffuse_sw - self.stats.diffuse_sw["mean"]) / self.stats.diffuse_sw["std"]

    def normalize_snowpack(self, x_snowpack):
        # x_snowpack = [[dz], [ssa], [density], [conc_soot], [conc_dust], [snow_layers]]
        x_new = np.zeros(x_snowpack.shape)
        x_new[0][0] = self._normalize_dz(x_snowpack[0][0])
        x_new[1][0] = self._normalize_ssa(x_snowpack[1][0])
        x_new[2][0] = self._normalize_density(x_snowpack[2][0])
        x_new[3][0] = self._normalize_conc_soot(x_snowpack[3][0])
        x_new[4][0] = self._normalize_conc_dust(x_snowpack[4][0])
        x_new[5][0] = x_snowpack[5][0]
        return x_new
    