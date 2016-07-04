"""
Implementation of the STORN model from the paper.
"""
import theano.tensor as tensor
import theano
import numpy
import time
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import merge
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Dropout, GRU, LSTM, Lambda
from greenarm.models.loss.variational import keras_variational
from greenarm.models.sampling.sampling import sample_gauss
from greenarm.util import add_samples_until_divisible, get_logger

logger = get_logger(__name__)


# enum for different phases
class Phases:
    def __init__(self):
        pass

    predict = 1
    train = 2


class STORNModel:
    def __init__(self):
        self.train_model = None
        self.predict_model = None
        self.z_prior_model = None
        self.z_recognition_model = None
        self._weights_updated = False

    def _build(self, phase, joint_shape, seq_shape=None, batch_size=None):
        self.z_recognition_model = STORNRecognitionModel()
        self.z_recognition_model.build(joint_shape, phase=phase, seq_shape=seq_shape, batch_size=batch_size)

        if phase == Phases.train:
            x_tm1 = Input(shape=(seq_shape, joint_shape), name="storn_input_train", dtype="float32")
            z_t = self.z_recognition_model.train_z_t
            x_t = self.z_recognition_model.train_input
            z_post_stats = self.z_recognition_model.train_recogn_stats
        else:
            x_tm1 = Input(batch_shape=(batch_size, 1, joint_shape), name="storn_input_predict", dtype="float32")
            z_t = self.z_recognition_model.predict_z_t
            x_t = self.z_recognition_model.predict_input
            z_post_stats = self.z_recognition_model.predict_recogn_stats

        z_tm1 = Lambda(self.shift_z, output_shape=self.shift_z_output_shape)(z_t)
        self.z_prior_model = STORNStandardPriorModel(x_tm1=x_tm1, z_tm1=z_tm1)
        self.z_prior_model.build(joint_shape, phase=phase, seq_shape=seq_shape, batch_size=batch_size)

        if phase == Phases.train:
            z_prior_stats = self.z_prior_model.train_prior_stats
        else:
            z_prior_stats = self.z_prior_model.predict_prior_stats

        gen_input = merge(inputs=[x_tm1, z_t], mode='concat')
        embed1 = TimeDistributed(Dense(50, activation="relu"))(gen_input)
        embed2 = TimeDistributed(Dense(50, activation="relu"))(embed1)
        embed3 = TimeDistributed(Dense(50, activation="relu"))(embed2)
        embed4 = TimeDistributed(Dense(50, activation="relu"))(embed3)
        embed5 = TimeDistributed(Dense(50, activation="relu"))(embed4)
        embed6 = TimeDistributed(Dense(50, activation="relu"))(embed5)
        # embed4 = Dropout(0.3)(embed4)

        rnn_gen = GRU(128, return_sequences=True, stateful=(phase == Phases.predict), consume_less='gpu')(
            embed6)

        gen_map1 = TimeDistributed(Dense(50, activation="relu"))(rnn_gen)
        gen_map2 = TimeDistributed(Dense(50, activation="relu"))(gen_map1)
        gen_map3 = TimeDistributed(Dense(50, activation="relu"))(gen_map2)
        gen_map4 = TimeDistributed(Dense(50, activation="relu"))(gen_map3)
        gen_map5 = TimeDistributed(Dense(50, activation="relu"))(gen_map4)
        gen_map6 = TimeDistributed(Dense(50, activation="relu"))(gen_map5)

        gen_mu = TimeDistributed(Dense(joint_shape, activation="linear"))(gen_map6)
        gen_sigma = TimeDistributed(Dense(joint_shape, activation="softplus"))(gen_map6)

        output = merge([gen_mu, gen_sigma, z_post_stats, z_prior_stats], mode='concat')
        model = Model(input=[x_t, x_tm1], output=output)
        model.compile(optimizer='rmsprop', loss=keras_variational)

        return model

    def build(self, joint_shape, seq_shape=None, batch_size=None):
        self.train_model = self._build(Phases.train, joint_shape, seq_shape=seq_shape)
        self.predict_model = self._build(Phases.predict, joint_shape, batch_size=batch_size)

    def load_predict_weights(self):
        # self.train_model.save_weights("storn_weights.h5", overwrite=True)
        self.predict_model.load_weights("best_storn_weights.h5")
        self._weights_updated = False

    def reset_predict_model(self):
        self.predict_model.reset_states()

    def fit(self, inputs, target, max_epochs=2, validation_split=0.1):
        seq_len = inputs[0].shape[1]
        self.train_model = self._build(Phases.train, 7, seq_shape=seq_len)

        split_idx = int((1. - validation_split) * inputs[0].shape[0])

        train_input, valid_input = [list(t) for t in zip(*[(X[:split_idx], X[split_idx:]) for X in inputs])]
        train_target, valid_target = target[:split_idx], target[split_idx:]

        checkpoint = ModelCheckpoint("best_storn_weights.h5", monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
        try:
            # A workaround so that kears does not complain
            padded_target = numpy.concatenate((train_target, numpy.zeros((train_target.shape[0],
                                                                          train_target.shape[1],
                                                                          5 * train_target.shape[2]))), axis=-1)
            padded_valid_target = numpy.concatenate((valid_target, numpy.zeros((valid_target.shape[0],
                                                                                valid_target.shape[1],
                                                                                5 * valid_target.shape[2]))), axis=-1)
            self.train_model.fit(
                train_input, padded_target,
                validation_data=(valid_input, [padded_valid_target]),
                callbacks=[checkpoint, early_stop],
                nb_epoch=max_epochs
            )

        except KeyboardInterrupt:
            logger.debug("Training interrupted! Restoring best weights and saving..")

        self.train_model.load_weights("best_storn_weights.h5")
        self._weights_updated = True
        self.save()

    def predict_one_step(self, inputs):
        original_num_samples = inputs[0].shape[0]
        _batch_size = 32
        inputs = [add_samples_until_divisible(input_x, _batch_size) for input_x in inputs]

        if self.predict_model is None:
            self.predict_model = self._build(Phases.predict, joint_shape=7, batch_size=_batch_size)
            self.load_predict_weights()

        return self.predict_model.predict(inputs, batch_size=_batch_size)[:original_num_samples, :, :]

    def evaluate(self, inputs, ground_truth):
        """
        :param inputs: a list of inputs for the model. In this case, it's a
                       one element list.
        :param ground_truth: the expected value to compare to
        :return: plotting artifacts: input, prediction, and error matrices
        """
        pred = self.predict_one_step(inputs)[:, :, :7]
        return pred, (ground_truth - pred) ** 2

    def reset_predict_model_states(self):
        self.predict_model.reset_states()

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/STORN_%s.model" % int(time.time())

        self.train_model.save_weights(prefix + ".weights.h5", overwrite=True)
        return prefix

    @staticmethod
    def shift_z(rec_z):
        return K.concatenate((K.random_normal(shape=(rec_z.shape[0], 1, rec_z.shape[2])),
                              rec_z[:, :-1, :]), axis=1)

    @staticmethod
    def shift_z_output_shape(input_shape):
        return input_shape


class STORNRecognitionModel:
    def __init__(self):
        self.train_recogn_stats = None
        self.train_input = None
        self.train_z_t = None
        self.predict_recogn_stats = None
        self.predict_input = None
        self.predict_z_t = None

    def _build(self, phase, joint_shape, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            x_t = Input(shape=(seq_shape, joint_shape), name="stornREC_input_train", dtype="float32")
        else:
            x_t = Input(batch_shape=(batch_size, 1, joint_shape), name="stornREC_input_predict", dtype="float32")

        embed1 = TimeDistributed(Dense(50, activation="relu"))(x_t)
        embed2 = TimeDistributed(Dense(50, activation="relu"))(embed1)
        embed3 = TimeDistributed(Dense(50, activation="relu"))(embed2)
        embed4 = TimeDistributed(Dense(50, activation="relu"))(embed3)
        embed5 = TimeDistributed(Dense(50, activation="relu"))(embed4)
        embed6 = TimeDistributed(Dense(50, activation="relu"))(embed5)
        # embed4 = Dropout(0.3)(embed4)
        rnn_recogn = GRU(128, return_sequences=True, stateful=(phase == Phases.predict), consume_less='gpu')(
            embed6)
        recogn_map1 = TimeDistributed(Dense(50, activation="relu"))(rnn_recogn)
        recogn_map2 = TimeDistributed(Dense(50, activation="relu"))(recogn_map1)
        recogn_map3 = TimeDistributed(Dense(50, activation="relu"))(recogn_map2)
        recogn_map4 = TimeDistributed(Dense(50, activation="relu"))(recogn_map3)
        recogn_map5 = TimeDistributed(Dense(50, activation="relu"))(recogn_map4)
        recogn_map6 = TimeDistributed(Dense(50, activation="relu"))(recogn_map5)
        rnn_recogn_mu = TimeDistributed(Dense(joint_shape, activation='linear'))(recogn_map6)
        rnn_recogn_sigma = TimeDistributed(Dense(joint_shape, activation="softplus"))(recogn_map6)

        # sample z|
        rnn_recogn_stats = merge([rnn_recogn_mu, rnn_recogn_sigma], mode='concat')

        # sample z from the distribution in X
        z_t = TimeDistributed(Lambda(self.do_sample,
                                     output_shape=self.sample_output_shape,
                                     arguments={'batch_size': (None if (phase == Phases.train) else batch_size),
                                                'dim_size': joint_shape}))(rnn_recogn_stats)

        return rnn_recogn_stats, x_t, z_t

    def build(self, joint_shape, phase=Phases.train, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            self.train_recogn_stats, self.train_input, self.train_z_t = self._build(Phases.train, joint_shape,
                                                                                    seq_shape=seq_shape)
        else:
            self.predict_recogn_stats, self.predict_input, self.predict_z_t = self._build(Phases.predict, joint_shape,
                                                                                          batch_size=batch_size)

    @staticmethod
    def do_sample(statistics, batch_size, dim_size):
        # split in half
        mu = statistics[:, :dim_size]
        sigma = statistics[:, dim_size:]

        if batch_size is None:
            batch_size = mu.shape[0]

        # sample with this mean and variance
        return sample_gauss(mu, sigma, batch_size, dim_size)

    @staticmethod
    def sample_output_shape(input_shape):
        shape = list(input_shape)
        shape[-1] /= 2
        return tuple(shape)


class STORNStandardPriorModel:
    def __init__(self, x_tm1, z_tm1):
        self.n_hidden_recurrent = 128
        self.x_tm1 = x_tm1
        self.z_tm1 = z_tm1
        self.train_prior_stats = None
        self.predict_prior_stats = None

    def _build(self, phase, latent_dim, seq_shape=None, batch_size=None):

        prior_input = merge([self.x_tm1, self.z_tm1], mode="concat")
        rnn_prior = GRU(self.n_hidden_recurrent,
                        return_sequences=True,
                        stateful=(phase == Phases.predict),
                        consume_less='gpu')(
            prior_input)
        rnn_rec_mu = TimeDistributed(Dense(latent_dim, activation='linear'))(rnn_prior)
        rnn_rec_sigma = TimeDistributed(Dense(latent_dim, activation="softplus"))(rnn_prior)

        # if phase == Phases.train:
        #     input_layer = Input(shape=(seq_shape, 2 * latent_dim), name="storn_prior_input_train", dtype="float32")
        # else:
        #     input_layer = Input(batch_shape=(batch_size, 1, 2 * latent_dim), name="storn_prior_input_predict",
        #                         dtype="float32")

        return merge([rnn_rec_mu, rnn_rec_sigma], mode="concat")

    def build(self, latent_dim, phase=Phases.train, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            self.train_prior_stats = self._build(Phases.train, latent_dim, seq_shape=seq_shape)
        else:
            self.predict_prior_stats = self._build(Phases.predict, latent_dim, batch_size=batch_size)

    @staticmethod
    def standard_input(number_of_series, seq_len, joint):
        sigma = numpy.ones(
            (number_of_series, seq_len, joint),
            dtype="float32"
        )
        my = numpy.zeros(
            (number_of_series, seq_len, joint),
            dtype="float32"
        )
        return numpy.concatenate([my, sigma], axis=-1)
