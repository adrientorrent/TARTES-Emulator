#!/usr/bin/env python3

import sys

import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""Author: L. ROUSSEL"""

plt.style.use("/home/torrenta/new-TARTES-Emulator/plot/_rcparams.mplstyle")


if __name__ == "__main__":

    # filename of the forcing
    dir = "/cnrm/cen/users/NO_SAVE/torrenta/Alpes/FORCING"
    fn_forcing = f"{dir}/FORCING_1979080106_1980080106.nc"

    # select the first point
    ds_forcing = xr.open_dataset(fn_forcing).sel(Number_of_points=0)

    # select the dates
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

    # resample from hourly data to monthly data
    dust = ds_forcing["dust"].resample(time='MS').sum()
    bc = ds_forcing["BC"].resample(time='MS').sum()

    # plot
    fig, ax = plt.subplots()

    width = pd.Timedelta(days=10)
    ax.bar(dust.time - width/2, dust.values, color="orange", width=width)
    ax.set_ylabel("Dust deposition (g)")
    ax.set_ylim(0, 3)
    ax.grid(True)
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis='x', rotation=45)

    ax_bis = ax.twinx()
    ax_bis.bar(bc.time + width/2, bc.values, color="grey", width=width)
    ax_bis.set_ylabel("BC deposition (g)")
    ax_bis.set_ylim(0, 0.02)

    # fig.legend(["Dust", "BC"])

    plt.show()

    sys.exit(0)
