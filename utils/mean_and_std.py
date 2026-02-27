#!/usr/bin/env python3

from genericpath import exists

from multiprocessing import Pool

import pandas as pd
import numpy as np
import polars as pl
import time


"""
Tools to compute mean and standard deviation
"""


def _sub_mean_and_std(paths: list[str]):
    """sub function to parallelize"""

    # build output
    snowpack = ["dz", "ssa", "density", "conc_soot", "conc_dust"]
    sun = ["direct_sw", "diffuse_sw", "cos_sza", "albedo"]
    sub_var_dict = {}
    for x in snowpack + sun:
        sub_var_dict[x] = {"count": 0, "sum": 0, "sum2": 0}

    # read 12 files on by one
    for path in paths:

        # read file
        polars_df = pl.read_parquet(path)

        # snowpack
        for x in snowpack:
            for i in range(50):

                xi = x+str(i + 1)

                polars_sum = polars_df.select(pl.sum(xi))
                sub_var_dict[x]["sum"] += polars_sum.item()

                polars_sum2 = polars_df.select((pl.col(xi)**2).sum())
                sub_var_dict[x]["sum2"] += polars_sum2.item()

                polars_count = polars_df.select(pl.count(xi))
                sub_var_dict[x]["count"] += polars_count.item()

        # shortwaves
        for x in sun:

            polars_sum = polars_df.select(pl.sum(x))
            sub_var_dict[x]["sum"] += polars_sum.item()

            polars_sum2 = polars_df.select((pl.col(x)**2).sum())
            sub_var_dict[x]["sum2"] += polars_sum2.item()

            polars_count = polars_df.select(pl.count(x))
            sub_var_dict[x]["count"] += polars_count.item()

        # clean ram
        del polars_df

    return sub_var_dict


def mean_and_std(
        files_paths: list[str],
        out_path: str
) -> None:
    """Main function, call _sub_mean_and_std()"""

    # initialize variables
    snowpack = ["dz", "ssa", "density", "conc_soot", "conc_dust"]
    sun = ["direct_sw", "diffuse_sw", "cos_sza", "albedo"]
    var_dict = {}
    for x in snowpack + sun:
        var_dict[x] = {"count": 0, "sum": 0, "mean": 0, "sum2": 0, "std": 0}

    # parallel processing
    pool = Pool(processes=10, maxtasksperchild=1)
    assert (len(files_paths) % 12) == 0, "Missing files"
    paths12 = [files_paths[i:i+12] for i in range(0, len(files_paths), 12)]
    sub_var_dicts: list[dict]
    with pool:
        sub_var_dicts = pool.map(_sub_mean_and_std, paths12)
    pool.close()
    for sub_var_dict in sub_var_dicts:
        for x in var_dict:
            var_dict[x]["sum"] += sub_var_dict[x]["sum"]
            var_dict[x]["sum2"] += sub_var_dict[x]["sum2"]
            var_dict[x]["count"] += sub_var_dict[x]["count"]

    # mean = sum / count
    for x in var_dict:
        var_dict[x]["mean"] = var_dict[x]["sum"] / var_dict[x]["count"]

    # std = sqrt((sum2 / count) - mean**2)
    for x in var_dict:
        var_dict[x]["std"] = np.sqrt(
            var_dict[x]["sum2"] / var_dict[x]["count"] - var_dict[x]["mean"]**2
        )

    # build dataframe
    df_out = pd.DataFrame(np.array([
        [var_dict[x]["mean"] for x in snowpack + sun],
        [var_dict[x]["std"] for x in snowpack + sun]
    ]))
    df_out.columns = snowpack + sun
    df_out.index = pd.Index(["mean", "std"])

    # add meta data
    df_out.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    df_out.attrs["files"] = files_paths

    # save into a parquet file
    df_out.to_parquet(out_path)
    return


def trigger_mean_and_std(
        files_paths: list[str],
        out_path: str
) -> None:
    """
    If we need to compute again ?
    Check if it has already been done, else call mean_and_std().
    """

    if not exists(out_path):
        mean_and_std(files_paths=files_paths, out_path=out_path)
        return

    # get previous files paths
    df = pd.read_parquet(out_path)
    prev_files_paths = df.attrs["files"]
    del df

    # trigger mean_and_std function
    if set(files_paths) != set(prev_files_paths):
        mean_and_std(files_paths=files_paths, out_path=out_path)

    return
