#!/usr/bin/env python3

import time
import pandas as pd
from utils.get_parquet_files import get_parquet_paths

t0 = time.time()

all_files = get_parquet_paths()
n_files = len(all_files)

n = 0
samples = 0
for file in all_files:
    # read file
    df = pd.read_parquet(file)
    # increment samples
    samples += len(df)
    # clean ram
    del df
    n += 1
    print(f"{n}/{n_files}")

print(f"Number of samples: {samples}")

t1 = time.time()
print(f"Elapsed time: {(t1 - t0):.2f} s")
