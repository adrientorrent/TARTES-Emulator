import time
import sys
sys.path.append("../utils")
from utils.get_parquet_files import get_parquet_paths
import pandas as pd
import numpy as np

# It may take more than 1H
t0 = time.time()

# --- paths ---
out_dir = "/home/torrenta/TARTES-Emulator/data/normalization"
out_path = out_dir+"/mean_and_std.parquet"

# --- list all parquet files ---
all_files = get_parquet_paths()
n_files = len(all_files)

# --- initializing variables ---
snowpack = ["dz", "ssa", "density", "conc_soot", "conc_dust"]
shortwaves = ["direct_sw", "diffuse_sw"]
var_dict = {}
for x in snowpack + shortwaves:
    var_dict[x] = {"count":0, "sum": 0, "mean": 0, "sum2": 0, "std": 0}

# --- computing mean ---
print("--- START COMPUTING MEAN ---")
n = 0
for file in all_files:
    df = pd.read_parquet(file)
    # snowpack
    for x in snowpack:
        for i in range(1, 51):
            var_dict[x]["sum"] += df[x+str(i)].sum()
            var_dict[x]["count"] += df.shape[0] - df[x+str(i)].isna().sum()
    # shortwaves
    for x in shortwaves:
        var_dict[x]["sum"] += df[x].sum()
        var_dict[x]["count"] += df.shape[0]
    # clean ram
    del df
    n += 1
    print(f"MEAN {n}/{n_files}")
# mean = sum / count
for x in snowpack + shortwaves:
    var_dict[x]["mean"] = var_dict[x]["sum"] / var_dict[x]["count"]

# --- computing standard deviation ---
print("--- START COMPUTING STD ---")
n = 0
for file in all_files:
    df = pd.read_parquet(file)
    # snowpack
    for x in snowpack:
        for i in range(1, 51):
            var_dict[x]["sum2"] += df[x+str(i)].pow(2).sum()
    # shortwaves
    for x in shortwaves:
        var_dict[x]["sum2"] += df[x].pow(2).sum()
    # clean ram
    del df
    n += 1
    print(f"STD {n}/{n_files}")
# std = sqrt((sum2 / count) - mean**2)
for x in snowpack + shortwaves:
    var_dict[x]["std"] = np.sqrt((var_dict[x]["sum2"] / var_dict[x]["count"]) - var_dict[x]["mean"]**2)

# --- save mean and std ---
print(f"Save mean and std in {out_path}")
df_out = pd.DataFrame(np.array([[var_dict[x]["mean"] for x in snowpack + shortwaves], \
                                [var_dict[x]["std"] for x in snowpack + shortwaves]]))
df_out.columns = snowpack + shortwaves
df_out.index = ["mean", "std"]
df_out.to_parquet(out_path)

t1 = time.time()
print("Elapsed time: %.2f " %(t1 - t0) + "s")
