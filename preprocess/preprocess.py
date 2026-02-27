#!/usr/bin/env python3

import os
import time
import logging
from genericpath import exists
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import xarray as xr
import pandas as pd

from snowtools.utils.sun import sun


"""
Preprocessing data

Read the FORCING and PRO netcdf files, and create new files with only needed
(parquet files are binary tabular files, working well with Pandas)

Please check all your file paths are the good ones
"""


# --- preprocess main function ---

def preprocess_function(forcing_path: str, pro_path: str,
                        out_path: str, month: int):

    # xarray datasets
    logger.debug("Opening FORCING and PRO datasets")
    FORCING = xr.open_dataset(forcing_path)
    PRO = xr.open_dataset(pro_path)

    # drop useless data
    logger.debug("Droping useless data")
    # forcing file
    useful_Fvar = ["ZS", "aspect", "slope", "massif_number", "LAT", "LON",
                   "DIR_SWdown", "SCA_SWdown"]
    useless_Fvar = sorted(set(FORCING.data_vars).difference(set(useful_Fvar)))
    FORCING = FORCING.drop_vars(useless_Fvar)
    # pro file
    useful_Pvar = ["WSN_T_ISBA", "SNOWDZ", "SNOWSSA", "RSN_VEG", "SNOWIMP1",
                   "SNOWIMP2", "ASN_VEG"]
    useless_Pvar = sorted(set(PRO.data_vars).difference(set(useful_Pvar)))
    PRO = PRO.drop_vars(useless_Pvar)

    # time sync
    logger.debug("Syncing time of FORCING and PRO")
    sync = np.intersect1d(FORCING["time"].values, PRO["time"].values)
    FORCING = FORCING.sel(time=sync)

    # month selection
    logger.debug("Month filtering")
    month_selection = FORCING["time.month"].values == month
    FORCING = FORCING.sel(time=month_selection)
    PRO = PRO.sel(time=month_selection)

    # remove tile dimension
    logger.debug("Removing tile dimension")
    PRO = PRO.isel(Number_of_Patches=0)

    # compute sun angle
    logger.debug("Computing sun angle")
    TAB_COS_SZA = sun().coszenith(
        tab_time_date=pd.to_datetime(FORCING["time"].values),
        lat=FORCING["LAT"].values,
        lon=FORCING["LON"].values,
        slope=FORCING["slope"].values,
        aspect=FORCING["aspect"].values
    )
    PRO["COS_SZA"] = (('time', 'Number_of_points'), TAB_COS_SZA)

    # reshape to stack time and Number_of_points dimensions
    # (include data loading)
    logger.debug("Stacking time and space dimensions (include data loading)")
    FORCING = FORCING.stack(tp=('time', 'Number_of_points'))
    PRO = PRO.stack(tp=('time', 'Number_of_points'))

    # select only days with snow on ground, and "day" time
    logger.debug("Removing days with no snow on ground + night time")
    # 1. remove days with no snow on ground
    snow_days = PRO["WSN_T_ISBA"].values > 0
    # 2. remove "night" time
    dir_sw = FORCING["DIR_SWdown"].values > 0
    dif_sw = FORCING["SCA_SWdown"].values > 0
    daytime = dir_sw + dif_sw
    # update datasets
    updated_time = snow_days & daytime
    FORCING = FORCING.sel(tp=updated_time)
    PRO = PRO.sel(tp=updated_time)

    # build new dataset
    logger.debug("Building new dataset")
    samples = {}

    # id data: (time + point) dimension only
    logger.debug("Gathering id data")
    samples["time"] = FORCING["time"].values
    samples["ZS"] = FORCING["ZS"].values
    samples["aspect"] = FORCING["aspect"].values
    samples["slope"] = FORCING["slope"].values
    samples["massif_number"] = FORCING["massif_number"].values

    # format id data
    logger.debug("Formatting id data")
    samples["time"] = samples["time"].astype(str)
    samples["massif_number"] = samples["massif_number"].astype(int)

    # snow data: (time + point) + snow layers dimensions
    logger.debug("Gathering snow data")
    for i in range(50):
        DS_TEMP = PRO.sel(snow_layer=i)
        samples[f"snow_layer{i+1}"] = ~np.isnan(DS_TEMP["SNOWDZ"].values)
        samples[f"dz{i+1}"] = DS_TEMP["SNOWDZ"].values
        samples[f"ssa{i+1}"] = DS_TEMP["SNOWSSA"].values
        samples[f"density{i+1}"] = DS_TEMP["RSN_VEG"].values
        samples[f"conc_soot{i+1}"] = DS_TEMP["SNOWIMP1"].values
        samples[f"conc_dust{i+1}"] = DS_TEMP["SNOWIMP2"].values
        del DS_TEMP

    # sun data: (time + point) dimension only
    logger.debug("Gathering sun data")
    samples["direct_sw"] = FORCING["DIR_SWdown"].values
    samples["diffuse_sw"] = FORCING["SCA_SWdown"].values
    samples["cos_sza"] = PRO["COS_SZA"].values

    # broadband albedo data: (time + point) dimension only
    logger.debug("Gathering albedo data")
    samples["albedo"] = PRO["ASN_VEG"].values

    # build pandas dataframe
    logger.debug("Building pandas dataframe")
    df = pd.DataFrame(samples)
    # sort columns
    snow_var = ["snow_layer", "dz", "ssa", "density", "conc_soot", "conc_dust"]
    snow_col = [f"{k}{i + 1}" for k in snow_var for i in range(50)]
    sun_col = ["direct_sw", "diffuse_sw", "cos_sza"]
    albedo_col = ["albedo"]
    id_col = ["time", "ZS", "aspect", "slope", "massif_number"]
    col = id_col + snow_col + sun_col + albedo_col
    df = df[col]

    # add metadata
    logger.debug("Adding metadata")
    df.attrs["description"] = "description"
    df.attrs["ids"] = {
        "time": "time",
        "ZS": "altitude",
        "aspect": "slope aspect",
        "slope": "slope angle",
        "massif_number": "massif number"
    }
    df.attrs["variables"] = {
        "snow_layer": "snow layer existence (bool)",
        "dz": "snow layer thickness (m)",
        "ssa": "snow layer specific surface area (m2 kg-1)",
        "density": "snow density (kg/m3)",
        "conc_soot": "concentration of soot (g/g)",
        "conc_dust": "concentration of dust (g/g)",
        "direct_sw": "surface incident direct shortwave radiation (W/m2)",
        "diffuse_sw": "surface incident diffuse shortwave radiation (W/m2)",
        "cos_sza": "cosinus of the solar zenith angle",
        "albedo": "snow albedo"
    }
    df.attrs["sources"] = {
        "FORCING": forcing_path,
        "PRO": pro_path
    }
    df.attrs["xpid"] = {
        "snowtools_command": PRO.snowtools_command,
        "forcingid": PRO.forcingid,
        "prep_xpid": PRO.prep_xpid,
        "id": PRO.id
    }
    df.attrs["time_coverage"] = {
        "start": str(np.min(PRO.time.values)),
        "end": str(np.max(PRO.time.values)),
        "resolution": PRO.time_coverage_resolution
    }

    # write parquet
    logger.debug("Writing parquet")
    df.to_parquet(out_path, engine="pyarrow")

    # clean memory
    del FORCING
    del PRO

    logger.info(f"{out_path} available")
    return


# --- multiprocessing utils ---

@dataclass
class PreprocessInput:
    forcing_path: str
    pro_path: str
    out_path: str
    month: int


def do_preprocess_function(input: PreprocessInput):

    # input checks
    logger.debug("Checking inputs presence")
    assert exists(input.forcing_path), f"{input.forcing_path} doesn't exist"
    assert exists(input.pro_path), f"{input.pro_path} doesn't exist"
    assert (input.month - 1) in range(12), "month must be in range of 1 and 12"

    # launch preprocess function
    preprocess_function(
        forcing_path=input.forcing_path,
        pro_path=input.pro_path,
        out_path=input.out_path,
        month=input.month
    )

    return


# --- PREPROCESSING ---

if __name__ == "__main__":

    NOSAVE_DIR = "/cnrm/cen/users/NO_SAVE/torrenta"
    BIGDATA_DIR = "/bigdata/BIGDATA/torrenta"
    CURRENT_DIR = "/home/torrenta/TARTES-Emulator/scripts/preprocessing"

    t0 = time.time()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- build multiprocessing queue ---

    logger.info("Building queue")
    queue = []

    # for each geo
    GEOS = ["Alpes", "Pyrenees"]
    for geo in GEOS:

        # forcing and pro files location
        forcing_dir = f"{NOSAVE_DIR}/{geo}/FORCING"
        pro_dir = f"{NOSAVE_DIR}/{geo}/PRO"

        # for each year
        for forcing_file in os.listdir(forcing_dir):

            # set input paths
            forcing_path = f"{forcing_dir}/{forcing_file}"
            pro_file = "PRO" + forcing_file[7:]
            pro_path = f"{pro_dir}/{pro_file}"

            # set output dir
            year = forcing_file[8:12] + "_" + forcing_file[19:23]
            output_dir = f"{BIGDATA_DIR}/{geo}/{geo}_{year}"

            # delete old data
            if exists(output_dir):
                os.system(f"rm -f {output_dir}/*")
            else:
                os.mkdir(output_dir)

            # for each month
            for month in range(12):

                # set output path
                output_path = f"{output_dir}/{geo}_{year}_{month + 1}.parquet"

                # add to the queue
                queue.append(PreprocessInput(
                    forcing_path=forcing_path,
                    pro_path=pro_path,
                    out_path=output_path,
                    month=(month + 1)
                ))

    logger.debug("Checking that queue is not empty")
    if len(queue) == 0:
        raise Exception(f"No files found in {NOSAVE_DIR}")
    logger.debug(f"Queue size: {len(queue)}")

    # --- multiprocessing ---

    logger.info("Launching pool")
    pool = Pool(processes=12, maxtasksperchild=1)
    with pool:
        pool.map(do_preprocess_function, queue)

    t1 = time.time()
    logger.info(f"Run time: {(t1 - t0):.2f} s")
