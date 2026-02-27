#!/usr/bin/env python3

import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import pandas as pd
import numpy as np


DZ_COLS = [f"dz{i + 1}" for i in range(50)]
SSA_COLS = [f"ssa{i + 1}" for i in range(50)]


def draw_snowpack(dz, param, param_label, date, geo):

    norm = mcolors.Normalize(vmin=min(param), vmax=max(param))
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots()

    z = 0

    for i in range(len(dz)):
        height = dz[i]
        value = param[i]
        color = cmap(norm(value))
        rect = patches.Rectangle(xy=(0, z), width=1, height=height,
                                 facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        z += height

    ax.set_xlim(0, 1)
    ax.set_ylim(0, z)
    ax.set_xticks([])
    ax.set_ylabel("Thickness (m)")
    ax.set_title(f"{date} \n {geo}")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=param_label)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parquet_dir = "/bigdata/BIGDATA/torrenta/Alpes/Alpes_1979_1980"
    files = []
    for i in [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]:
        files.append(f"{parquet_dir}/Alpes_1979_1980_{i}.parquet")

    df = pd.concat(
        (pd.read_parquet(
            f,
            engine="pyarrow",
            filters=[
                ("massif_number", "==", 3),
                ("ZS", "==", 2700),
                ("slope", "==", 0),
                ("aspect", "==", -1),
                ("ssa1", "<=", 20),
                ("albedo", ">=", 0.90)
            ]
        ) for f in files),
        ignore_index=True
    )

    point = df.iloc[0]
    time_label = f"Date: {point["time"][0:10]}"
    massif_label = f"massif: {point["massif_number"]}"
    alt_label = f"altitude: {int(point["ZS"])}m"
    slope_label = f"slope: {int(point["slope"])}°"
    aspect_label = f"aspect: {int(point["aspect"])}"
    geo_label = f"{massif_label}, {alt_label}, {slope_label}, {aspect_label}"

    dz = point[DZ_COLS].to_numpy(dtype=np.float64)
    dz = dz[~np.isnan(dz)]
    dz = dz[::-1]

    ssa = point[SSA_COLS].to_numpy(dtype=np.float64)
    ssa = ssa[~np.isnan(ssa)]
    ssa = ssa[::-1]

    draw_snowpack(dz, ssa, "SSA (m2/kg)", time_label, geo_label)

    sys.exit(0)
