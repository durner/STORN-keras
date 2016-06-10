import logging

import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)


def get_logger(name):
    return logging.getLogger(name)


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
