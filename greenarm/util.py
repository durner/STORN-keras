import logging
import numpy as np
import random
from sklearn.metrics import roc_curve, auc

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.INFO)


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

    logger.info("Total: %s. Positives: %s. Negatives: %s" % (total, ground_truth.sum(), total - ground_truth.sum()))
    logger.info("Predicted: Positives: %s. Negatives: %s" % (predicted.sum(), total - predicted.sum()))
    logger.info("Accuracy: %s" % (corrects / float(total)))

    print_eval_from_counts(tp, fp, tn, fn)


def print_eval_from_counts(tp, fp, tn, fn):
    P = (tp / (tp + fp))
    R = (tp / (tp + fn))

    logger.info("TP: %s. FP: %s. TN: %s. FN: %s." % (tp, fp, tn, fn))

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


dimension_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
cols = ["yellow", "orange", "purple", "black", "green", "blue", "cyan"]


def plot_model_output(plot, ground_truth, title="Model output", prediction=None, alpha_gt=1,
                      flip_color=False):
    plot.set_title(title, fontsize=30)
    plot.set_ylim([-4, 4])

    # plot the ground truths and predictions
    for i in range(ground_truth.shape[-1]):
        if flip_color:
            col1 = "grey"
            col2 = cols[i]
        else:
            col1 = cols[i]
            col2 = "grey"

        plot.plot(ground_truth[:, i],
                  label=dimension_names[i], linewidth=4.0, alpha=alpha_gt, color=col1)
        if prediction is not None:
            plot.plot(prediction[:, i], color=col2, label='prediction', linewidth=3.0)

    plot.legend(loc="lower left", prop={'size': 30})


def plot_model_error(plot, error, label="Model loss"):
    plot.plot(error, label=label, color="red", linewidth=3.0)
    plot.legend(loc="lower left", prop={'size': 30})


def plot_full(plot, error, ground_truth, prediction, original_anomal, detected_anomal,
              threshold, title="Full plot", alpha_gt=1, loss_label="Model loss"):
    # plot anomalies
    for anomaly in original_anomal:
        plot.axvline(anomaly, color='m', linewidth=4.0)
        plot.axvspan(anomaly, anomaly + 40, facecolor='grey', alpha=0.2)
    for anomaly in detected_anomal:
        plot.axvline(anomaly, color='g', linewidth=4.0)

    # plot the STORN output first
    plot_model_output(plot, ground_truth, title=title, alpha_gt=alpha_gt)

    # plot the error
    plot_model_error(plot, error, label=loss_label)

    # plot anomaly threshold
    plot.axhline(y=threshold, color='blue', ls='dashed')


def plot_ROC_curve(plot, target, pred_scores, name):
    """
    Plots an ROC curve given a vector of binary targets and predicted confidence scores.
    :param target: vector of 0, 1 target labels
    :param pred_scores: the predicted probability scores, or confidence scores
    """
    fp_rate, tp_rate, _ = roc_curve(target, pred_scores)
    roc_auc = auc(fp_rate, tp_rate)

    plot.plot(fp_rate, tp_rate, label='%s, (AUC = %0.4f)' % (name, roc_auc))
    plot.plot([0, 1], [0, 1], 'k--')
    plot.xlim([0.0, 1.0])
    plot.ylim([0.0, 1.05])
    plot.xlabel('FP Rate')
    plot.ylabel('TP Rate')
    plot.title('ROC curves')
    plot.legend(loc="lower right")
