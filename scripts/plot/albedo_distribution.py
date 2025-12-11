#!/usr/bin/env python3

import matplotlib.pyplot as plt

import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils.data_selection import train_test_split
from normalization.normalize import CustomNorm2
from dataset import CnnTartesIterableDataset
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

# parquets files
DIR = "/bigdata/BIGDATA/torrenta/Alpes/Alpes_1979_1980"
fp_year = []
for i in range(12):
    fp_year.append(f"{DIR}/Alpes_1979_1980_{i+1}.parquet")
final_tartes_albedo = []
for file in fp_year:
    df = pd.read_parquet(file)
    final_tartes_albedo += list(df["albedo"])

# model prediction
files, _, _ = train_test_split(train={("Alpes", "1979_1980")},
                               test={("Alpes", "1980_1981")}, seed=2025)
data_dir = "/home/torrenta/TARTES-Emulator/data"
mean_and_std_path = f"{data_dir}/normalization/mean_and_std_212025.parquet"
custom_norm = CustomNorm2(mean_and_std_path)
dataset = CnnTartesIterableDataset(files=files, norm=custom_norm)
dataloader = DataLoader(dataset=dataset, batch_size=512, num_workers=16,
                        pin_memory=True, prefetch_factor=64,
                        persistent_workers=True, drop_last=False)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = CnnTartesEmulator()
params_path = "/home/torrenta/TARTES-Emulator/data/model/cnn-tartes-model.pt"
model.load_state_dict(torch.load(params_path, weights_only=True))
model = model.to(DEVICE)
model.eval()
model_predictions = []
i = 0
for X_snow, X_sun, y in dataloader:
    X_snow, X_sun = X_snow.to(DEVICE), X_sun.to(DEVICE)
    y_hat = model(X_snow, X_sun)
    y_hat = y_hat.detach().cpu()
    for alb in y_hat:
        model_predictions.append(alb[0])
    print(i)
    i += 1

# plot
fig, axs = plt.subplots(2)
fig.suptitle("Distribution of broadband albedo "
             "between August 1979 and July 1980 in the Alps")
axs[0].hist(final_tartes_albedo, bins=200, range=(0, 0.99))
axs[0].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[0].set_title("targets")
axs[1].hist(model_predictions, bins=200, range=(0, 0.99))
axs[1].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[1].set_title("model predictions")
plt.show()
