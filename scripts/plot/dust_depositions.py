#!/usr/bin/env python3

import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
# import matplotlib.colors as colors


"""
General description

author: Léon ROUSSEL
"""

# paramètre pour plot joli
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


if __name__ == "__main__":

    # filename of the forcing, à changer
    dir = "/cnrm/cen/users/NO_SAVE/torrenta/Test/FORCING"
    fn_forcing = f"{dir}/FORCING_1979080106_1980080106.nc"

    # ncdump -h <filename> pour avoir un résumé rapide du fichier

    # utilisation de xarray, select first point of the simulation
    ds_forcing = xr.open_dataset(fn_forcing).sel(Number_of_points=0)

    # selection temporelle, change the dates
    d1, d2 = pd.to_datetime("1979-09-01"), pd.to_datetime("1980-07-31")
    assert d1 > ds_forcing.time[0] and d2 < ds_forcing.time[-1]
    ds_forcing = ds_forcing.sel(time=slice(d1, d2))

    # 1: BC, black carbon
    # 2: dust, mineral dust
    # wet: humid deposition (in rain or snowfall)
    # dry: dry deposition
    # each in g/m2/s, *3600 to have per hour.
    ds_forcing["BC"] = (ds_forcing["IMPDRY1"] + ds_forcing["IMPWET1"]) * 3600
    ds_forcing["dust"] = (ds_forcing["IMPDRY2"] + ds_forcing["IMPWET2"]) * 3600

    print("ds:", ds_forcing)

    # resample from hourly data to monthly data
    dust = ds_forcing["dust"].resample(time='MS').sum()
    # resample from hourly data to monthly data
    bc = ds_forcing["BC"].resample(time='MS').sum()

    fig, axs = plt.subplots(2, 1, figsize=(12/2.54, 12/2.54))
    width = (dust.time[1] - dust.time[0]) * 0.8
    axs[0].bar(dust.time, dust.values, color="orange", width=width)
    axs[1].bar(bc.time, bc.values, color="grey", width=width)

    # Style and legend
    axs[0].set_ylabel("Dust deposition (g)")
    maxi = 8
    axs[0].set_ylim(0, maxi)
    axs[1].set_ylabel("BC deposition (g)")
    axs[1].set_ylim(0, maxi / 300)
    for ax in axs:
        ax.grid(True)
        ax.set_xlabel("Time")

    plt.show()
