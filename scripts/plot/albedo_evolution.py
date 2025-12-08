#!/usr/bin/env python3

import matplotlib.pyplot as plt

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

# xarray file
DIR = "/cnrm/cen/users/NO_SAVE/torrenta/Alpes/PRO"
fp = f"{DIR}/PRO_1979080106_1980080106.nc"

# data
ds = xr.open_dataset(fp)
ds = ds.isel(Number_of_Patches=0)
ds = ds.isel(Number_of_points=500)
midi = ds["time.hour"].values == 12
ds = ds.sel(time=midi)
time = ds["time"]
albedo = ds["ASN_VEG"]
swe = ds["WSN_T_ISBA"]
# mean_albedo = albedo.mean(dim="Number_of_points")
# mean_swe = swe.mean(dim="Number_of_points")

fig, ax = plt.subplots()
fig.suptitle("Evolution of broadband albedo at midday"
             "\n between August 1979 and July 1980"
             "\n (at 2700m, in the Mont Blanc massif)")
ax.plot(time, albedo, color="red")
ax.set_xlabel("Time")
ax.set_ylabel("Broadband albedo")
ax2 = ax.twinx()
ax2.plot(time, swe, color="blue")
ax2.set_ylabel("Snow Water Equivalent")
plt.show()
