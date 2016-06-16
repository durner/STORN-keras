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
from keras.layers import Input, TimeDistributed, Dense, Dropout, GRU, Lambda
from greenarm.models.loss.variational import keras_variational
from greenarm.models.sampling.sampling import sample_gauss
from greenarm.util import add_samples_until_divisible, get_logger

logger = get_logger(__name__)


class STORNModel:
    def __init__(self):
        self.train_model = None
        self.train_input = None
        self.predict_model = None
        self.predict_input = None
        self.storn_rec = None
        self._weights_updated = False

    def _build(self, phase, recognition_model, joint_shape, seq_shape=None, batch_size=None):
        if phase == "train":
            input_layer = Input(shape=(seq_shape, joint_shape))
            rec_z = recognition_model.train_z
            rec_input = recognition_model.train_input
            rec = recognition_model.train_rnn_recogn_stats
        else:
            input_layer = Input(shape=(1, joint_shape))
            rec_z = recognition_model.predict_z
            rec_input = recognition_model.predict_input
            rec = recognition_model.predict_rnn_recogn_stats

        gen_input = merge(inputs=[input_layer, rec_z], mode='concat')
        embed2 = TimeDistributed(Dense(32, activation="relu"), input_shape=(seq_shape, 2*joint_shape))(gen_input)
        embed2 = Dropout(0.3)(embed2)
        rnn_gen = GRU(128, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(embed2)

        rnn_gen_mu = TimeDistributed(Dense(joint_shape, activation="linear"))(rnn_gen)
        rnn_gen_sigma = TimeDistributed(Dense(joint_shape, activation="softplus"))(rnn_gen)

        output = merge([rnn_gen_mu, rnn_gen_sigma, rec], mode='concat')
        model = Model(input=[rec_input, input_layer], output=output)
        model.compile(optimizer='rmsprop', loss=keras_variational)

        return model, input_layer

    def build(self, recognition_model, joint_shape, seq_shape=None, batch_size=None):
        self.train_model, self.train_input = self._build("train", recognition_model, joint_shape, seq_shape=seq_shape)
        self.predict_model, self.predict_input = self._build("predict", recognition_model, joint_shape, batch_size=batch_size)

    def load_predict_weights(self):
        self.train_model.save_weights("storn_weights.h5", overwrite=True)
        self.predict_model.load_weights("storn_weights.h5")
        self._weights_updated = False

    def reset_predict_model(self):
        self.predict_model.reset_states()

    def fit(self, X, y, max_epochs=2, validation_split=0.1):
        seq_len = X.shape[1]
        self.storn_rec = STORNRecognitionModel()
        self.storn_rec.build(7, seq_shape=seq_len)
        self.train_model, self.train_input = self._build("train", self.storn_rec, 7, seq_shape=seq_len)

        split_idx = int((1. - validation_split) * X.shape[0])
        X, X_val = X[:split_idx], X[split_idx:]
        y, y_val = y[:split_idx], y[split_idx:]

        checkpoint = ModelCheckpoint("best_storn_weights.h5", monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        try:
            target = numpy.concatenate((y, numpy.zeros((y.shape[0], y.shape[1], 3 * y.shape[2]))), axis=-1)
            target_val = numpy.concatenate((y_val, numpy.zeros((y_val.shape[0], y_val.shape[1], 3 * y_val.shape[2]))), axis=-1)

            self.train_model.fit(
                [y, X], [target], validation_data=([y_val, X_val], [target_val]), callbacks=[checkpoint, early_stop], nb_epoch=max_epochs
            )

        except KeyboardInterrupt:
            logger.debug("Trianing interrupted! Restoring best weights and saving..")

        self.train_model.load_weights("best_storn_weights.h5")
        self._weights_updated = True
        self.save()

    # todo @durner: thuink about predict method with the right input arguments for our network
    def predict_one_step(self, x):
        original_num_samples = x.shape[0]
        _batch_size = 1
        x = add_samples_until_divisible(x, _batch_size)

        if self.predict_model is None:
            self.predict_model, self.predict_input = self._build("predict", self.storn_rec, 7, batch_size=_batch_size)

        if self._weights_updated:
            self.load_predict_weights()

        return self.predict_model.predict(x, batch_size=_batch_size)[:original_num_samples, :, :]

    def reset_predict_model_states(self):
        self.predict_model.reset_states()

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/STORN_%s.model" % int(time.time())

        logger.debug("Saving model to %s" % prefix)

        with open(prefix + ".json", "w") as of:
            of.write(self.train_model.to_json())

        self.train_model.save_weights(prefix + ".weights.h5")
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
        if phase == "train":
            input_layer = Input(shape=(seq_shape, joint_shape))
        else:
            input_layer = Input(shape=(1, joint_shape))

        embed1 = TimeDistributed(Dense(32, activation="tanh"))(input_layer)
        embed1 = Dropout(0.3)(embed1)
        rnn_recogn = GRU(128, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(embed1)
        rnn_recogn_mu = TimeDistributed(Dense(joint_shape, activation='linear'))(rnn_recogn)
        rnn_recogn_sigma = TimeDistributed(Dense(joint_shape, activation="softplus"))(rnn_recogn)

        # sample z|x
        rnn_recogn_stats = merge([rnn_recogn_mu, rnn_recogn_sigma], mode='concat')

        # sample z from the distribution in X
        sample_z = TimeDistributed(Lambda(self.do_sample, output_shape=(joint_shape,)))(rnn_recogn_stats)

        return rnn_recogn_stats, input_layer, sample_z

    def build(self, joint_shape, seq_shape=None, batch_size=None):
        self.train_rnn_recogn_stats, self. train_input, self.train_z = self._build("train", joint_shape, seq_shape=seq_shape)
        self.predict_rnn_recogn_stats, self.predict_input, self.predict_z = self._build("predict", joint_shape, batch_size=batch_size)

    @staticmethod
    def do_sample(statistics):
        # split in half
        dim = statistics.shape[-1] / 2
        mu = statistics[:, :dim]
        sigma = statistics[:, dim:]

        # sample with this mean and variance
        return sample_gauss(mu, sigma)


class STORNPriorModel:
    def __init__(self):
        pass

    def predict(self, z, t, k):
        sigma = numpy.ones(
                (t, k),
                dtype=tensor.dscalar()
        )
        my = numpy.zeros(
                (t, k),
                dtype=tensor.dscalar()
        )
        sigma = theano.shared(value=sigma, name='sigma')
        my = theano.shared(value=my, name='my')

        return my, sigma
