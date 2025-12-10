#!/usr/bin/env python3

import matplotlib.pyplot as plt

import pandas as pd
import xarray as xr

import torch
from torch.utils.data import DataLoader

from utils.data_selection import train_test_split
from normalization.normalize import CustomNorm2
from dataset import MlpTartesIterableDataset
from model import MlpTartesEmulator


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

# parquets files
DIR1 = "/bigdata/BIGDATA/torrenta/Alpes/Alpes_1979_1980"
fp_year = []
for i in range(12):
    fp_year.append(f"{DIR1}/Alpes_1979_1980_{i+1}.parquet")

# xarray file
DIR2 = "/cnrm/cen/users/NO_SAVE/torrenta/Alpes/PRO"
fp = f"{DIR2}/PRO_1979080106_1980080106.nc"

# data
final_tartes_albedo = []
for file in fp_year:
    df = pd.read_parquet(file)
    final_tartes_albedo += list(df["albedo"])

ds = xr.open_dataset(fp)
ds = ds.isel(Number_of_Patches=0)
all_tartes_albedo = ds["ASN_VEG"]
all_tartes_albedo = all_tartes_albedo.stack(tp=("time", "Number_of_points"))

# model prediction
files, _, _ = train_test_split(train={("Alpes", "1979_1980")},
                               test={("Alpes", "1980_1981")}, seed=2025)
data_dir = "/home/torrenta/TARTES-Emulator/data"
mean_and_std_path = f"{data_dir}/normalization/mean_and_std_212025.parquet"
custom_norm = CustomNorm2(mean_and_std_path)
dataset = MlpTartesIterableDataset(files=files, norm=custom_norm)
dataloader = DataLoader(dataset=dataset, batch_size=512, num_workers=16,
                        pin_memory=True, prefetch_factor=64,
                        persistent_workers=True, drop_last=False)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
mlp_model = MlpTartesEmulator()
params_path = "/home/torrenta/TARTES-Emulator/data/model/mlp-tartes-model.pt"
mlp_model.load_state_dict(torch.load(params_path, weights_only=True))
mlp_model = mlp_model.to(DEVICE)
mlp_model.eval()
model_predictions = []
i = 0
for X, y in dataloader:
    X = X.to(DEVICE)
    y_hat = mlp_model(X)
    y_hat = y_hat.detach().cpu()
    for alb in y_hat:
        model_predictions.append(alb[0])
    print(i)
    i += 1

# plot
fig, axs = plt.subplots(3)
fig.suptitle("Distribution of broadband albedo "
             "between August 1979 and July 1980 in the Alps")
axs[0].hist(all_tartes_albedo, bins=200, range=(0, 0.99))
axs[0].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[0].set_title("from the crocus output (masking albedo=1)")
axs[1].hist(final_tartes_albedo, bins=200, range=(0, 0.99))
axs[1].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[1].set_title("from my preprocessed files")
axs[2].hist(model_predictions, bins=200, range=(0, 0.99))
axs[2].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[2].set_title("model predictions")
plt.show()
