import logging

import numpy as np
import random

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)


def get_logger(name):
    return logging.getLogger(name)


def generate_x_y(data, predict_forward=1, shuffle=True):
    if shuffle:
        idxs = range(data.shape[0])
        random.shuffle(idxs)
        data = data[idxs]
    return data[:, :-predict_forward, :], data[:, predict_forward:, :]


def add_samples_until_divisible(x, batch_size):
    num_samples = x.shape[0]
    sample_shape = x.shape[1:]
    num_missing = batch_size * (num_samples // batch_size + 1) - num_samples
    missing_shape = tuple([num_missing] + list(sample_shape))
    return np.vstack([x, np.zeros(shape=missing_shape)])


def subsample(sequence, step):
    """
    :param sequence: A sequence to be sub-sampled. The original sampling period must be at least 2*step.
    :param step: The sub-sampling period.
    :return: The sub-sampled result.
    """
    result = []
    prev = sequence[0, 0]
    for i, current_timestamp in enumerate(sequence[:, 0]):
        if current_timestamp >= (prev + step):
            result.append(sequence[i, :])
            prev = current_timestamp
    return np.asarray(result, dtype='float32')
