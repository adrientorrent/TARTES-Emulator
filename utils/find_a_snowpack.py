#!/usr/bin/env python3

import sys

import pandas as pd

"""
Useful if you're looking for a specific snowpack in the parquet files.

Exemple here, I was looking for :
- surface ssa <= 20
- broadband albedo >= 0.90
"""


META_COLS = ["time", "ZS", "aspect", "slope", "massif_number"]

DZ_COLS = [f"dz{i + 1}" for i in range(50)]
SSA_COLS = [f"ssa{i + 1}" for i in range(50)]
DENSITY_COLS = [f"density{i + 1}" for i in range(50)]
SOOT_COLS = [f"conc_soot{i + 1}" for i in range(50)]
DUST_COLS = [f"conc_dust{i + 1}" for i in range(50)]

SUN_COLS = ["direct_sw", "diffuse_sw", "cos_sza"]

ALBEDO_COL = ["albedo"]


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

    print(df[DZ_COLS])

    sys.exit(0)
