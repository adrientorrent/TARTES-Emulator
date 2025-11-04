import time
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from snowtools.utils.sun import sun

t0 = time.time()

# args
parser = argparse.ArgumentParser(description="Save training samples to a parquet file")
parser.add_argument("-f", "--forcing", type=str, metavar="", required=True, help="Path of the forcing file")
parser.add_argument("-p", "--pro", type=str, metavar="", required=True, help="Path of the pro file")
parser.add_argument("-m", "--month", type=int, metavar="", required=False, help="Month number to save")
parser.add_argument("-o", "--output", type=str, metavar="", required=True, help="Path of the output file")
args = parser.parse_args()

# paths
forcing_path = args.forcing
pro_path = args.pro
output_path = args.output

# xarray datasets
FORCING = xr.open_dataset(forcing_path)
PRO = xr.open_dataset(pro_path)

# drop useless data
useful_Fvar = ["LAT", "LON", "slope", "aspect", "DIR_SWdown", "SCA_SWdown"]
useful_Pvar = ["WSN_T_ISBA", "SNOWDZ", "SNOWSSA", "RSN_VEG", "SNOWIMP1", "SNOWIMP2", "ASN_VEG"]
useless_Fvar = sorted(set(FORCING.data_vars).difference(set(useful_Fvar)))
useless_Pvar = sorted(set(PRO.data_vars).difference(set(useful_Pvar)))
FORCING = FORCING.drop_vars(useless_Fvar)
PRO = PRO.drop_vars(useless_Pvar)

# time sync
FORCING = FORCING.sel(time=np.intersect1d(FORCING["time"].values, PRO["time"].values))
if args.month is not None:
    assert args.month in range(1, 13)
    months = FORCING["time"].values.astype('datetime64[M]').astype(int) % 12 + 1 == args.month
    FORCING = FORCING.sel(time=months)
    PRO = PRO.sel(time=months)

# load data (7GB)
FORCING.load()
PRO.load()

samples = []
# for each point
for point in FORCING.Number_of_points.values:

    FP = FORCING.sel(Number_of_points=point)
    PP = PRO.sel(Number_of_points=point)

    # removing days without snow
    snow_days = PP["WSN_T_ISBA"].values > 0
    
    # removing night time
    daytime = (FP["DIR_SWdown"].values > 0) + (FP["SCA_SWdown"].values > 0)

    # update datasets
    updated_time = snow_days & daytime
    n_time = sum(updated_time)
    if n_time == 0: continue # if it doesn't snow all year (possible at low altitudes)
    FP = FP.sel(time=updated_time)
    PP = PP.sel(time=updated_time)

    # sza computing
    TAB_cosSZA = sun().coszenith(
        tab_time_date=pd.to_datetime(FP["time"].values), 
        lat=FP["LAT"].values, 
        lon=FP["LON"].values, 
        slope=FP["slope"].values, 
        aspect=FP["aspect"].values
    )

    # for each time
    for t_id in range(n_time):
        
        FPT = FP.isel(time=t_id)
        PPT = PP.isel(time=t_id)

        # sample reading
        dz = PPT["SNOWDZ"].values
        ssa = PPT["SNOWSSA"].values
        density = PPT["RSN_VEG"].values
        conc_soot = PPT["SNOWIMP1"].values
        conc_dust = PPT["SNOWIMP2"].values
        direct_sw = np.array([[FPT["DIR_SWdown"].values]])
        diffuse_sw = np.array([[FPT["SCA_SWdown"].values]])
        cos_sza = np.array([TAB_cosSZA[t_id]])
        albedo = np.array([PPT["ASN_VEG"].values])

        # sample writing
        x = np.hstack((dz, ssa, density, conc_soot, conc_dust, \
                       direct_sw, diffuse_sw, cos_sza, albedo))[0]
        samples.append(x)

        # clean ram
        del FPT
        del PPT

    # clean ram
    del FP
    del PP

# clean ram
del FORCING
del PRO

# pandas dataframe
if len(samples) > 0:
    df = pd.DataFrame(np.array(samples))
    df.columns = [k+str(i) for k in ["dz", "ssa", "density", "conc_soot", "conc_dust"] \
                  for i in range(1, 51)] + ["direct_sw", "diffuse_sw", "sza", "albedo"]
    df.to_parquet(output_path)
else:
    print(f"[{args.month}]", "Empty")

t1 = time.time()
print(f"[{args.month}]", "Temps d'exécution : %.2f " %(t1 - t0) + "secondes")
