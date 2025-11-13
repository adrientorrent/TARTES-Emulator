#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import logging

import numpy as np
import xarray as xr
import pandas as pd

from snowtools.utils.sun import sun

logging.basicConfig(level=logging.INFO)

t0 = time.time()

# args
parser = argparse.ArgumentParser(description="Save training samples to a parquet file")
parser.add_argument("-f", "--forcing", type=str, required=True, help="Path of the forcing file")
parser.add_argument("-p", "--pro", type=str, required=True, help="Path of the pro file")
parser.add_argument("-m", "--month", type=int, help="Month number to save")
parser.add_argument("-o", "--output", type=str, required=True, help="Path of the output file")
args = parser.parse_args()

# paths
forcing_path = args.forcing
pro_path = args.pro
output_path = args.output

# xarray datasets
logging.info('Open datasets')
FORCING = xr.open_dataset(forcing_path)
PRO = xr.open_dataset(pro_path)

# drop useless data
logging.info('Drop useless data')
useful_Fvar = ["LAT", "LON", "slope", "aspect", "DIR_SWdown", "SCA_SWdown"]
useful_Pvar = ["WSN_T_ISBA", "SNOWDZ", "SNOWSSA", "RSN_VEG", "SNOWIMP1", "SNOWIMP2", "ASN_VEG"]
useless_Fvar = sorted(set(FORCING.data_vars).difference(set(useful_Fvar)))
useless_Pvar = sorted(set(PRO.data_vars).difference(set(useful_Pvar)))
FORCING = FORCING.drop_vars(useless_Fvar)
PRO = PRO.drop_vars(useless_Pvar)

# time sync
logging.info('Time sync')
FORCING = FORCING.sel(time=np.intersect1d(FORCING["time"].values, PRO["time"].values))

# remove tile dimension
PRO = PRO.sel(Number_of_Patches=0)

# Month filtering
logging.info('Month filtering')
if args.month is not None:
    assert args.month in range(1, 13)
    months = FORCING["time"].values.astype('datetime64[M]').astype(int) % 12 + 1 == args.month
    FORCING = FORCING.sel(time=months)
    PRO = PRO.sel(time=months)


# load data (7GB)
logging.info('Load data')
# FORCING.load()
# PRO.load()

# Compute sun angle
logging.info('Sun angle')
TAB_cosSZA = sun().coszenith(
    tab_time_date=pd.to_datetime(FORCING["time"].values),
    lat=FORCING["LAT"].values,
    lon=FORCING["LON"].values,
    slope=FORCING["slope"].values,
    aspect=FORCING["aspect"].values
)
PRO['coszenith'] = (('time', 'Number_of_points'), TAB_cosSZA)

# Reshaping to have only one dimension that gather time and space
# and get unidimensional tables for most of variables (except profiles that are still 2D)
logging.info('Stack (include data loading)')
FORCING = FORCING.stack(tp = ('time', 'Number_of_points'))
PRO = PRO.stack(tp = ('time', 'Number_of_points'))


# Filtering
logging.info('Filtering')
#   Remove days with no snow on ground
snow_days = PRO["WSN_T_ISBA"].values > 0
#   removing "night" time
daytime = (FORCING["DIR_SWdown"].values > 0) + (FORCING["SCA_SWdown"].values > 0)
# TODO: Chat does  + mean ?
# update datasets
updated_time = snow_days & daytime
FORCING = FORCING.sel(tp=updated_time)
PRO = PRO.sel(tp=updated_time)


# Data gathering
data = {}

# (time + point) dimension only
logging.info('Data1')
data['direct_sw'] = FORCING['DIR_SWdown'].values
data['diffuse_sw'] = FORCING['SCA_SWdown'].values
data['sza'] = PRO['coszenith'].values
data['albedo'] = PRO["ASN_VEG"].values

# (time + point) + layers dimensions
logging.info('Data2')
keys_profiles = {
    'dz': 'SNOWDZ',
    'ssa': 'SNOWSSA',
    'density': 'RSN_VEG',
    'conc_soot': 'SNOWIMP1',
    'conc_dust': 'SNOWIMP2'}
for i in range(50):
    PL = PRO.sel(snow_layer=i)
    for k1, k2 in keys_profiles.items():
        data[f'{k1}{i + 1}'] = PL[k2].values

logging.info('Dataframe')
df = pd.DataFrame(data)
logging.info('Writting output')
df.to_parquet(output_path)

t1 = time.time()
logging.info(f"Temps d'exécution : {t1 - t0:.2f} secondes")
