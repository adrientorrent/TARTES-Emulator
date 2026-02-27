#!/usr/bin/env python3

import os
from genericpath import exists

import random
from math import floor


"""Tools for gathering the paths of parquet files"""


# --- global variables ---

BIGDATA_DIR = "/bigdata/BIGDATA/torrenta"
GEOS = ["Alpes", "Pyrenees"]


# --- utils ---

def _build_path(dir: str, filename: str) -> str:
    return dir + '/' + filename


# --- typing ---

type pathslist = list[str]


# --- functions ---

def all_paths() -> pathslist:
    """Gather all parquet files paths"""
    out: pathslist = []
    for geo in GEOS:
        geo_dir = _build_path(BIGDATA_DIR, geo)
        years_list = os.listdir(geo_dir)
        for year in years_list:
            year_dir = _build_path(geo_dir, year)
            parquet_files_list = os.listdir(year_dir)
            for parquet_file in parquet_files_list:
                file_path = _build_path(year_dir, parquet_file)
                out.append(file_path)
    return out


def all_paths_groupby_year() -> list[pathslist]:
    """Gather all parquet files paths group by year"""
    out: list = []
    for geo in GEOS:
        geo_dir = _build_path(BIGDATA_DIR, geo)
        years_list = os.listdir(geo_dir)
        for year in years_list:
            sublist_out: pathslist = []
            year_dir = _build_path(geo_dir, year)
            parquet_files_list = os.listdir(year_dir)
            for parquet_file in parquet_files_list:
                file_path = _build_path(year_dir, parquet_file)
                sublist_out.append(file_path)
            out.append(sublist_out)
    return out


def all_paths_groupby_geo_year() -> list[list[pathslist]]:
    """Gather all parquet files paths, group by geo then by year"""
    out: list = []
    for geo in GEOS:
        sublist1_out: list = []
        geo_dir = _build_path(BIGDATA_DIR, geo)
        years_list = os.listdir(geo_dir)
        for year in years_list:
            sublist2_out: pathslist = []
            year_dir = _build_path(geo_dir, year)
            parquet_files_list = os.listdir(year_dir)
            for parquet_file in parquet_files_list:
                file_path = _build_path(year_dir, parquet_file)
                sublist2_out.append(file_path)
            sublist1_out.append(sublist2_out)
        out.append(sublist1_out)
    return out


def one_year_only(geo: str, year: str) -> pathslist:
    """
    Gather all parquet files paths for 1 year only
    12 months => 12 files paths
    """
    geo_dir = _build_path(BIGDATA_DIR, geo)
    year_dir = _build_path(geo_dir, f"{geo}_{year}")
    assert geo in GEOS, f"{geo} geo doesn't exist"
    assert len(year) == 9, f"{year}: wrong year format, use startyear_endyear"
    assert exists(year_dir), f"{year_dir} doesn't exist"
    out: pathslist = []
    parquet_files_list = os.listdir(year_dir)
    for parquet_file in parquet_files_list:
        file_path = _build_path(year_dir, parquet_file)
        out.append(file_path)
    return out


def train_test_split_1(
        train: set[tuple[str, str]],
        test: set[tuple[str, str]],
        val: set[tuple[str, str]] | None = None
) -> tuple[pathslist, pathslist, pathslist]:
    """
    Gather all files paths and split them into 3 datasets train, test, val

    Option 1: specify exactly which years of which geos you want
    -> formats: sets of tuples like ("geo", "year")
    -> ex: {("Alpes", "1979_1980")}
    """

    # init outputs
    out_train: pathslist = []
    out_test: pathslist = []
    out_val: pathslist = []

    # case where val is not specified
    if val is None:
        val = set()

    # ensure that the sets have any samples in common
    assert len(train.intersection(test)) == 0, \
        "Sets train and test have common samples"
    assert len(train.intersection(val)) == 0, \
        "Sets train and val have common samples"
    assert len(test.intersection(val)) == 0, \
        "Sets test and val have common samples"

    # fill datasets
    for t in train:
        out_train += one_year_only(geo=t[0], year=t[1])
    for t in test:
        out_test += one_year_only(geo=t[0], year=t[1])
    for t in val:
        out_val += one_year_only(geo=t[0], year=t[1])

    return out_train, out_test, out_val


def train_test_split_2(
        train: int,
        test: int,
        val: int | None = None
) -> tuple[pathslist, pathslist, pathslist]:
    """
    Gather all files paths and split them into 3 datasets train, test, val

    Option 2: specify the number of years you want
    -> formats: ints
    """

    # init outputs
    out_train: pathslist = []
    out_test: pathslist = []
    out_val: pathslist = []

    # case where val is not specified
    if val is None:
        val = 0

    # ensure that number of files requested <= total number of files available
    files_groupby_year = all_paths_groupby_year()
    assert train + test + val <= len(files_groupby_year), \
        "Number of files requested > total number of files available"

    # shuffle full dataset
    random.shuffle(files_groupby_year)

    # fill datasets
    for i in range(train):
        out_train += files_groupby_year[i]
    for i in range(test):
        out_test += files_groupby_year[train + i]
    for i in range(val):
        out_val += files_groupby_year[train + test + i]

    return out_train, out_test, out_val


def train_test_split_3(
        train: float,
        test: float,
        val: float | None = None
) -> tuple[pathslist, pathslist, pathslist]:
    """
    Gather all files paths and split them into 3 datasets train, test, val

    Option 3: specify the ratio of the full dataset you want
    -> formats: floats <= 1. (train + test + val must be <= 1. ofc)
    """

    # init outputs
    out_train: pathslist = []
    out_test: pathslist = []
    out_val: pathslist = []

    # case where val is not specified
    if val is None:
        val = 0.

    # ensure that the total ratio requested <= 1
    assert train + test + val <= 1., "Total ratio requested > 1"

    # convert ratio to files count
    files_groupby_year = all_paths_groupby_year()
    n_train = floor(train * len(files_groupby_year))
    n_test = floor(test * len(files_groupby_year))
    n_val = len(files_groupby_year) - n_train - n_test

    # shuffle full dataset
    random.shuffle(files_groupby_year)

    # fill datasets
    for i in range(n_train):
        out_train += files_groupby_year[i]
    for i in range(n_test):
        out_test += files_groupby_year[n_train + i]
    for i in range(n_val):
        out_val += files_groupby_year[n_train + n_test + i]

    return out_train, out_test, out_val


def train_test_split(
        train: set[tuple[str, str]] | int | float,
        test: set[tuple[str, str]] | int | float,
        val: set[tuple[str, str]] | int | float | None = None,
        seed: int | None = None
) -> tuple[pathslist, pathslist, pathslist]:
    """
    Gather all files paths and split them into 3 datasets train, test, val

    Option 1: specify exactly which years of which geos you want
    -> formats: sets of tuples like ("geo", "year")
    -> ex: {("Alpes", "1979_1980")}

    Option 2: specify the number of years you want
    -> formats: ints

    Option 3: specify the ratio of the full dataset you want
    -> formats: floats <= 1. (train + test + val must be <= 1. ofc)
    """

    # set a random seed to retrieve the same files
    if seed is not None:
        random.seed(seed)

    # option 1, 2 or 3
    match train, test, val:

        case set(), set(), (set() | None):
            return train_test_split_1(train=train, test=test, val=val)

        case int(), int(), (int() | None):
            return train_test_split_2(train=train, test=test, val=val)

        case float(), float(), (float() | None):
            return train_test_split_3(train=train, test=test, val=val)

        case _, _, _:
            raise Exception("Args train, test, val must be of the same type")


def print_selection(files: pathslist) -> None:
    already_printed = []
    for file_path in files:
        geo = "Alpes" if "Alpes" in file_path else "Pyrénées"
        y1 = file_path.split("_")[1]
        y2 = file_path.split("_")[2][:4]
        txt = f" - {geo} {y1} {y2}"
        if txt not in already_printed:
            print(txt)
            already_printed.append(txt)
