import keras.backend as K


def biased_binary_crossentropy(positive_weight, y_true, y_pred):
    """
    Biased binary crossentropy loss to improve sensitvity
    :param positive_weight: the push of negativley classified positves mistakes
    :param y_true: true y label
    :param y_pred: predicated y label
    :return: the keras loss tensor
    """
    return K.mean(- (positive_weight * (y_true * K.log(y_pred)) + (1.0 - y_true) * K.log(1.0 - y_pred)), axis=-1)