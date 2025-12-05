#!/usr/bin/env python3

import matplotlib.pyplot as plt

import pandas as pd
import xarray as xr


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

# plot
fig, axs = plt.subplots(2)
fig.suptitle("Distribution of broadband albedo "
             "between August 1979 and July 1980 in the Alps")
axs[0].hist(all_tartes_albedo, bins=200, range=(0, 0.99))
axs[0].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[0].set_title("from the crocus output (masking albedo=1)")
axs[1].hist(final_tartes_albedo, bins=200, range=(0, 0.99))
axs[1].set(xlim=(0, 1), xticks=[i/100 for i in range(0, 100, 5)])
axs[1].set_title("from my preprocessed files")
plt.show()
