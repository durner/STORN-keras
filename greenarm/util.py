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
    P = (tp / (tp + fp))
    R = (tp / (tp + fn))
    logger.info("Total: %s. Positives: %s. Negatives: %s" % (total, ground_truth.sum(), total - ground_truth.sum()))
    logger.info("Predicted: Positives: %s. Negatives: %s" % (predicted.sum(), total - predicted.sum()))
    logger.info("TP: %s. FP: %s. TN: %s. FN: %s." % (tp, fp, tn, fn))
    logger.info("Accuracy: %s" % (corrects / float(total)))
    logger.info("Precision: %s" % P)
    logger.info("Recall (Sensitivity): %s" % R)
    logger.info("TN Rate (Specificity): %s" % (tn / (tn + fp)))
    logger.info("F1: %s" % ((2. * P * R) / (P + R)))


def pad_sequences_3d(sequences, maxlen, return_paddings=False, skip_first_n_dims=0):
    data_dimensionality = sequences[0].shape[-1] - skip_first_n_dims
    data = np.zeros(shape=(len(sequences), maxlen, data_dimensionality), dtype="float32")

    paddings = []
    for sample_index, sample in enumerate(sequences):
        if maxlen >= sample.shape[0]:
            data[sample_index] = np.vstack(
                (np.zeros((maxlen - sample.shape[0], data_dimensionality)), sample[:, skip_first_n_dims:])
            )
            paddings.append(maxlen - sample.shape[0])
        else:
            data[sample_index] = sample[:maxlen, skip_first_n_dims:]
            paddings.append(0)

    if return_paddings:
        return data, paddings

    return data
