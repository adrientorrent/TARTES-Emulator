#!/usr/bin/env python3

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from normalization.normalize import CustomNorm2
from model import CnnTartesEmulator


# pour plot joli (merci Léon)
plt.rcParams.update({
    'figure.constrained_layout.use': True,
    'font.size': 10,
    'grid.color': (0.5, 0.5, 0.5, 0.3),
    'axes.edgecolor': 'black',
    'xtick.color':    'black',
    'ytick.color':    'black',
    'axes.labelcolor': 'black',
    'axes.spines.right': False,
    'axes.spines.top':  False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.pad': 2,
    'xtick.minor.pad': 2,
    'ytick.major.pad': 2,
    'ytick.minor.pad': 2,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 6,
    })


class CustomDataset(torch.utils.data.IterableDataset):
    """Custom pytorch iterable dataset for CNN model"""

    def __init__(self, dataframes: list[pd.DataFrame], norm: CustomNorm2):

        super().__init__()

        self.dataframes = dataframes
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

    def __iter__(self):
        """Custom __iter__ method"""

        for df in self.dataframes:
            snowpack_arr = df[self.snowpack_cols].to_numpy(dtype=np.float64)
            sun_arr = df[self.sun_cols].to_numpy(dtype=np.float64)
            target_arr = df[self.target_cols].to_numpy(dtype=np.float64)

            for i in range(len(snowpack_arr)):
                mask = np.isnan(snowpack_arr[i])
                if mask.any():
                    snowpack_arr[i][mask] = self.norm.snowpack_means[mask]
                snowpack_arr[i] = self.norm.normalize_snowpack(snowpack_arr[i])
                sun_arr[i] = self.norm.normalize_sun(sun_arr[i])

            snowpack_tensor = torch.from_numpy(snowpack_arr.astype(np.float32))
            sun_tensor = torch.from_numpy(sun_arr.astype(np.float32))
            target_tensor = torch.from_numpy(target_arr.astype(np.float32))

            for row in range(len(snowpack_tensor)):
                out1 = snowpack_tensor[row].reshape((6, 50))
                out2 = sun_tensor[row]
                out3 = target_tensor[row]
                yield out1, out2, out3


# parquets files
DIR = "/bigdata/BIGDATA/torrenta/Alpes/Alpes_1979_1980"
fp_year = []
for i in [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]:
    fp_year.append(f"{DIR}/Alpes_1979_1980_{i}.parquet")

# dataframes
dataframes = []
tartes_time = []
tartes_albedo = []
for file in fp_year:
    df = pd.read_parquet(file)
    df = df.loc[df['massif_number'] == 3]
    df = df.loc[df['ZS'] == 2700]
    df = df.loc[df['slope'] == 40]
    df = df.loc[df['aspect'] == 45]
    # df = df.loc[df['time'].str.contains("12:00:00")]
    dataframes.append(df)
    tartes_time += list(pd.to_datetime(df["time"]))
    tartes_albedo += list(df["albedo"])

# predictions
data_dir = "/home/torrenta/TARTES-Emulator/data"
mean_and_std_path = f"{data_dir}/normalization/mean_and_std_212026.parquet"
custom_norm = CustomNorm2(mean_and_std_path)
dataset = CustomDataset(dataframes, custom_norm)
dataloader = DataLoader(dataset=dataset, batch_size=512, drop_last=False)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = CnnTartesEmulator()
params_path = "/home/torrenta/TARTES-Emulator/data/model/cnn-tartes-model.pt"
model.load_state_dict(torch.load(params_path, weights_only=True))
model = model.to(DEVICE)
model.eval()
batch_it = 0
predictions = []
for X_snow, X_sun, y in dataloader:
    X_snow, X_sun = X_snow.to(DEVICE), X_sun.to(DEVICE)
    y_hat = model(X_snow, X_sun)
    y_hat = y_hat.detach().cpu()
    for alb in y_hat:
        predictions.append(alb[0])
    batch_it += 1

tartes_time = tartes_time[900:950]
tartes_albedo = tartes_albedo[900:950]
predictions = predictions[900:950]

# plot
fig, ax = plt.subplots()
fig.suptitle("Day cycle")
ax.plot(tartes_time, tartes_albedo, color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Tartes Albedo")
ax2 = ax.twinx()
ax2.plot(tartes_time, predictions, color="blue", dashes=[2, 2])
ax2.set_ylabel("Prediction")
plt.show()
