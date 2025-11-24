#!/usr/bin/env python3

import time
import logging

import pandas as pd

from scripts.utils.data_selection import all_paths

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

t0 = time.time()

logger.info("Building full dataset paths")
all_files_paths = all_paths()
n_files = len(all_files_paths)

n = 0
samples = 0
logger.info("Reading files one by one")
for file_path in all_files_paths:
    # read file
    df = pd.read_parquet(file_path)
    # increment samples
    samples += len(df)
    # clean ram
    del df
    n += 1
    logger.info(f"{n}/{n_files}")

print(f"Number of samples: {samples}")

t1 = time.time()
logger.info(f"Elapsed time: {(t1 - t0):.2f} s")
