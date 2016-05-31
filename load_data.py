import h5py
from glob import glob
import numpy as np


def read_angle_measurements(path):
    with h5py.File(path, "r") as hf:
        res = hf['0']['configuration']['measured'][:]

    return res


def load_data(dir):
    """
    Loads all the data from the *.h5 files in the specified directory
    :param dir: the directory to load from
    :return: X, a list of measured sequences
    """

    path = dir + '/*.h5'
    files = glob(path)
    return [read_angle_measurements(path) for path in files]


