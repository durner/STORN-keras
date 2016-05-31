import logging

import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)


def get_logger(name):
    return logging.getLogger(name)


def subsample(sequence, step):
    result = []
    max_error = 0

    prev = sequence[0, 0]
    last_diff = np.inf
    for i, current_timestamp in enumerate(sequence[:, 0]):
        current_diff = abs((prev + step) - current_timestamp)
        if current_diff > last_diff:
            result.append(sequence[i - 1, :])
            prev = sequence[i - 1, 0]
            if last_diff > max_error:
                max_error = last_diff
            last_diff = np.inf
        else:
            last_diff = current_diff
    return np.asarray(result, dtype='float32'), max_error
