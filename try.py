import numpy as np
import theano.tensor as tensor
import theano
import numpy
from keras.layers import Input, Masking, GRU, TimeDistributed, Dense, Dropout, Lambda, Merge, merge
from keras.models import Model
from keras import backend as K

from greenarm.load_data import load_data
from greenarm.util import subsample, generate_x_y, get_logger
from greenarm.models.sampling.sampling import sample_gauss
from greenarm.models.loss.variational import keras_variational

ln = get_logger(__name__)

def get_data():

    normal1 = load_data('data/normal1')
    normal2 = load_data('data/normal2')

    normal = normal1 + normal2

    anormal = load_data('data/anomal')

    def preprocess_dataset(dataset, maxlen_=None):
        # sample frequency of 15 Hz
        period = 1. / 15.0
        subsampled = [subsample(s, period) for s in dataset]

        if maxlen_ is None:
            maxlen = max([s.shape[0] for s in subsampled])
        else:
            maxlen = maxlen_

        data = np.zeros(shape=(len(subsampled), maxlen, 7), dtype="float32")

        # zero padding needs to be done manually here, apparently
        for sample_index, sample in enumerate(subsampled):
            if maxlen >= sample.shape[0]:
                data[sample_index] = np.vstack((np.zeros((maxlen - sample.shape[0], 7)), sample[:, 1:]))
            else:
                data[sample_index] = sample[:maxlen, 1:]

        x, y = generate_x_y(data, predict_forward=1)

        if maxlen_ is None:
            return (x, y), maxlen
        else:
            return x, y

    (x, y), maxlen = preprocess_dataset(normal)
    a_x, a_y = preprocess_dataset(anormal, maxlen)

    train_x, test_x = x[0: int(x.shape[0] * 0.8)], x[int(x.shape[0] * 0.8):]
    train_y, test_y = y[0: int(y.shape[0] * 0.8)], y[int(y.shape[0] * 0.8):]

    ln.debug("Shape of train_x {}".format(train_x.shape))
    ln.debug("Shape of train_y {}".format(train_y.shape))

    return maxlen, train_x, train_y

"""
TODO: recognition model
Input() of X
LSTM() <--------- use this as one output
Lambda() sampling from the sigma and mu (1)
Input() the previous X (2)
Based on (1) and (2) LSTM <----------- second output
"""

if __name__ == '__main__':
    def do_sample(statistics):
        # split in half
        dim = statistics.shape[-1] / 2
        mu = statistics[:, :dim]
        sigma = statistics[:, dim:]

        # sample with this mean and variance
        return sample_gauss(mu, sigma)


    # inin = Input(shape=(6,))
    # extracted = Lambda(do_sample)(inin)
    # model = Model(input=inin, output=extracted)
    # model.compile(optimizer='rmsprop', loss='mean_squared_error')
    # test = model.predict(np.ones((6, 6)))
    # ln.info("{}".format(test))


    # consider this as well
    """
    https: // github.com / fchollet / keras / blob / master / examples / variational_autoencoder.py
    """
    maxlen = 672
    # maxlen, train_x, train_y = get_data()

    input_x_t = Input(shape=(maxlen, 7))
    embed1 = TimeDistributed(Dense(32, activation="tanh"))(input_x_t)
    embed1 = Dropout(0.3)(embed1)
    rnn_recogn = GRU(
        128, return_sequences=True, dropout_W=0.2, dropout_U=0.2
    )(embed1)
    rnn_recogn_stats = TimeDistributed(Dense(14, activation="relu"))(rnn_recogn)

    # sample z from the distribution in X
    sample_z = TimeDistributed(Lambda(do_sample, output_shape=(maxlen, 14)))(rnn_recogn_stats)
    input_x_tm1 = Input(shape=(maxlen, 7))

    print("{} {}".format(sample_z.shape[-1], input_x_tm1.shape[-1]))
    input_gen = K.concatenate([input_x_tm1, sample_z], axis=-1)
    embed2 = TimeDistributed(Dense(32, activation="relu"), input_shape=(maxlen, 14))(input_gen)
    embed2 = Dropout(0.3)(embed2)
    rnn_gen = GRU(
        128, return_sequences=True, dropout_W=0.2, dropout_U=0.2
    )(embed2)
    rnn_gen_stats = TimeDistributed(Dense(14, activation="relu"))(rnn_gen)

    output = K.concatenate([rnn_gen_stats, rnn_recogn_stats], axis=-1)
    model = Model(input=[input_x_t, input_x_tm1], output=output)
    model.compile(optimizer='rmsprop', loss=keras_variational)
