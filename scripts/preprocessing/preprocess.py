#!/usr/bin/env python3

import os
from time import sleep, gmtime, strftime
from genericpath import exists

# --- PATHS ---
nosave_dir = "/cnrm/cen/users/NO_SAVE/torrenta"
bigdata_dir = "/bigdata/BIGDATA/torrenta"
current_dir = "/home/torrenta/TARTES-Emulator/scripts/preprocessing"
script = current_dir+"/write_parquet.py"

# --- FOR EACH GEO ---
geos = ["Alpes", "Pyrenees"]
for geo in geos:

    print(f"----- GEO : {geo} -----")

    # forcing and pro folders
    forcing_dir = nosave_dir+"/"+geo+"/FORCING"
    pro_dir = nosave_dir+"/"+geo+"/PRO"

    # monitoring progression
    k = 0
    n_year = len(os.listdir(forcing_dir))

    # --- FOR EACH YEAR ---
    for forcing in os.listdir(forcing_dir):

        k += 1

        # print time
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        # print year
        year = forcing[8:12]+"-"+forcing[19:23]
        print(f"YEAR : {year}")

        # paths
        forcing_path = forcing_dir+"/"+forcing
        pro = "PRO"+forcing[7:]
        pro_path = pro_dir+"/"+pro
        output_dir = bigdata_dir+"/"+geo+"/"+geo[0]+year

        # check if the year has already been processed
        if exists(output_dir):
            print(f"({k}/{n_year}) Année {year} déjà traitée")
        else:
            # create directory
            os.mkdir(output_dir)
            # processing each month independently
            for month in range(1, 13):

                # set path of the output parquet file
                output_file = output_dir+"/"+geo[0]+year+"-"+str(month)+".parquet"

                # run script (it will finish in 20 minutes)
                options = f"-f {forcing_path} -p {pro_path} -m {month} -o {output_file}"
                command = f"python3 {script} {options} &"
                os.system(command)

            # wait 30 minutes
            sleep(1800)
            print(f"({k}/{n_year}) Année {year} traitée")

print("Preprocessing terminé.")
