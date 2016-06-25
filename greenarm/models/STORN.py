"""
Implementation of the STORN model from the paper.
"""
import theano.tensor as tensor
import theano
import numpy
import time
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
        self.train_input = None
        self.predict_model = None
        self.predict_input = None
        self.storn_trending_prior = None
        self.storn_rec = None
        self._weights_updated = False

    def _build(self, phase, joint_shape, seq_shape=None, batch_size=None):
        self.storn_trending_prior = STORNStandardPriorModel()
        self.storn_trending_prior.build(joint_shape, phase=phase, seq_shape=seq_shape, batch_size=batch_size)

        self.storn_rec = STORNRecognitionModel()
        self.storn_rec.build(joint_shape, phase=phase,  seq_shape=seq_shape, batch_size=batch_size)

        if phase == Phases.train:
            input_layer = Input(shape=(seq_shape, joint_shape), name="storn_input_train", dtype="float32")
            rec_z = self.storn_rec.train_z
            rec_input = self.storn_rec.train_input
            rec = self.storn_rec.train_rnn_recogn_stats
            prior = self.storn_trending_prior.train_trending
            prior_input = self.storn_trending_prior.train_input
        else:
            input_layer = Input(batch_shape=(batch_size, 1, joint_shape), name="storn_input_predict", dtype="float32")
            rec_z = self.storn_rec.predict_z
            rec_input = self.storn_rec.predict_input
            rec = self.storn_rec.predict_rnn_recogn_stats
            prior = self.storn_trending_prior.predict_trending
            prior_input = self.storn_trending_prior.predict_input

        gen_input = merge(inputs=[input_layer, rec_z], mode='concat')
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

        rnn_gen_mu = TimeDistributed(Dense(joint_shape, activation="linear"))(gen_map6)
        rnn_gen_sigma = TimeDistributed(Dense(joint_shape, activation="softplus"))(gen_map6)

        output = merge([rnn_gen_mu, rnn_gen_sigma, rec, prior], mode='concat')
        model = Model(input=[rec_input, input_layer, prior_input], output=output)
        model.compile(optimizer='rmsprop', loss=keras_variational)

        return model, input_layer

    def build(self, joint_shape, seq_shape=None, batch_size=None):
        self.train_model, self.train_input = self._build(Phases.train, joint_shape, seq_shape=seq_shape)
        self.predict_model, self.predict_input = self._build(Phases.predict, joint_shape, batch_size=batch_size)

    def load_predict_weights(self):
        # self.train_model.save_weights("storn_weights.h5", overwrite=True)
        self.predict_model.load_weights("best_storn_weights.h5")
        self._weights_updated = False

    def reset_predict_model(self):
        self.predict_model.reset_states()

    def fit(self, inputs, target, max_epochs=2, validation_split=0.1):
        list_in = inputs[:]
        list_in.append(STORNStandardPriorModel.standard_input(inputs[0].shape[0], inputs[0].shape[1],
                                                              inputs[0].shape[2]))

        seq_len = inputs[0].shape[1]
        self.train_model, self.train_input = self._build(Phases.train, 7, seq_shape=seq_len)

        split_idx = int((1. - validation_split) * inputs[0].shape[0])

        train_input, valid_input = [list(t) for t in zip(*[(X[:split_idx], X[split_idx:]) for X in list_in])]
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
        list_in = inputs[:]
        list_in.append(STORNStandardPriorModel.standard_input(inputs[0].shape[0], inputs[0].shape[1],
                                                              inputs[0].shape[2]))

        original_num_samples = inputs[0].shape[0]
        _batch_size = 32
        inputs = [add_samples_until_divisible(input_x, _batch_size) for input_x in list_in]

        if self.predict_model is None:
            self.predict_model, self.predict_input = self._build(Phases.predict, joint_shape=7, batch_size=_batch_size)
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


class STORNRecognitionModel:
    def __init__(self):
        self.train_rnn_recogn_stats = None
        self.train_input = None
        self.train_z = None
        self.predict_rnn_recogn_stats = None
        self.predict_input = None
        self.predict_z = None

    def _build(self, phase, joint_shape, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            input_layer = Input(shape=(seq_shape, joint_shape), name="stornREC_input_train", dtype="float32")
        else:
            input_layer = Input(batch_shape=(batch_size, 1, joint_shape), name="stornREC_input_predict", dtype="float32")

        embed1 = TimeDistributed(Dense(50, activation="relu"))(input_layer)
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
        sample_z = TimeDistributed(Lambda(self.do_sample,
                                          output_shape=self.sample_output_shape,
                                          arguments={'batch_size': (None if (phase == Phases.train) else batch_size),
                                                     'dim_size': joint_shape}))(rnn_recogn_stats)

        return rnn_recogn_stats, input_layer, sample_z

    def build(self, joint_shape, phase=Phases.train, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            self.train_rnn_recogn_stats, self.train_input, self.train_z = self._build(Phases.train, joint_shape,
                                                                                      seq_shape=seq_shape)
        else:
            self.predict_rnn_recogn_stats, self.predict_input, self.predict_z = self._build(Phases.predict, joint_shape,
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
    def __init__(self):
        self.train_trending = None
        self.train_input = None
        self.predict_trending = None
        self.predict_input = None

    def _build(self, phase, joint_shape, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            input_layer = Input(shape=(seq_shape, 2 * joint_shape), name="storn_prior_input_train", dtype="float32")
        else:
            input_layer = Input(batch_shape=(batch_size, 1, 2 * joint_shape), name="storn_prior_input_predict", dtype="float32")

        return input_layer, input_layer

    def build(self, joint_shape, phase=Phases.train, seq_shape=None, batch_size=None):
        if phase == Phases.train:
            self.train_trending, self.train_input = self._build(Phases.train, joint_shape, seq_shape=seq_shape)
        else:
            self.predict_trending, self.predict_input = self._build(Phases.predict, joint_shape, batch_size=batch_size)

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
