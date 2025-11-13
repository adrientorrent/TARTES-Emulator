#!/usr/bin/env python3

import os
from math import floor
from random import shuffle

# TODO
# Améliorations : pouvoir indiquer un nombre d'années à récupérer

bigdata_dir = "/bigdata/BIGDATA/torrenta"
geo = ["Alpes", "Pyrenees"]


def _build_path(dir: str, filename: str) -> str:
    return dir + '/' + filename


def get_parquet_paths():
    # collect all parquet files paths
    paths_list = []
    for g in geo:
        geo_dir = _build_path(bigdata_dir, g)
        years = os.listdir(geo_dir)
        for y in years:
            year_dir = _build_path(geo_dir, y)
            parquet_files = os.listdir(year_dir)
            for file in parquet_files:
                file_path = _build_path(year_dir, file)
                paths_list.append(file_path)
    return paths_list


def get_parquet_paths_groupby_year():
    # collect parquet files paths group by year
    years_list = []
    for g in geo:
        geo_dir = _build_path(bigdata_dir, g)
        years = os.listdir(geo_dir)
        for y in years:
            paths_list = []
            year_dir = _build_path(geo_dir, y)
            parquet_files = os.listdir(year_dir)
            for file in parquet_files:
                file_path = _build_path(year_dir, file)
                paths_list.append(file_path)
            years_list.append(paths_list)
    return years_list


def split_train_test_val(train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    # split parquet files - group by year - into 3 datasets: train, test and val

    assert train_ratio + test_ratio + val_ratio <= 1.0

    # get all parquet files group by year
    files_groupby_year = get_parquet_paths_groupby_year()
    # and shuffle list
    shuffle(files_groupby_year)

    # split full dataset
    n_total = len(files_groupby_year)
    n_train = floor(n_total * train_ratio)
    n_test = floor(n_total * test_ratio)
    # and build new datasets
    train_files = files_groupby_year[:n_train]
    test_files = files_groupby_year[n_train:n_train+n_test]
    val_files = files_groupby_year[n_train+n_test:]
    return train_files, test_files, val_files
