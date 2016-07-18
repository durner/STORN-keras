import logging

import numpy as np
import random

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)


def get_logger(name):
    return logging.getLogger(name)


logger = get_logger(__name__)


def generate_shifted(data, predict_forward=1):
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


def shuffle_together(*arrays):
    new_indexes = range(arrays[0].shape[0])
    random.shuffle(new_indexes)
    return [array[new_indexes] for array in arrays]


def print_eval(predicted, ground_truth):
    tp, fp, tn, fn = 0.000000001, 0.000000001, 0.000000001, 0.000000001
    total = predicted.shape[0]
    corrects = 0.
    for p, gt in zip(predicted, ground_truth):
        if p == gt:
            corrects += 1.
            if gt:
                tp += 1.
            else:
                tn += 1.
        else:
            if gt:
                fn += 1.
            else:
                fp += 1.
    P = (tp / tp + fp)
    R = (tp / tp + fn)
    logger.debug("Total: %s. Positives: %s. Negatives: %s" % (total, predicted.sum(), total - predicted.sum()))
    logger.debug("TP: %s. FP: %s. TN: %s. FN: %s." % (tp, fp, tn, fn))
    logger.debug("Accuracy: %s" % (corrects / float(total)))
    logger.debug("Precision: %s" % P)
    logger.debug("Recall (Sensitivity): %s" % R)
    logger.debug("TN Rate (Specificity): %s" % (tn / tn + fp))
    logger.debug("F1: %s" % ((2. * P * R) / (P + R)))
