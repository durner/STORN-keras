import time

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import TimeDistributed, Dense, Input, GRU, Masking, Dropout
from keras.models import Model
from keras.regularizers import l1

from greenarm.util import get_logger, add_samples_until_divisible

logger = get_logger(__name__)


class TimeSeriesPredictor(object):
    def __init__(self):
        self.train_model = None
        self.predict_model = None
        self.num_hidden_recurrent = 128
        self.num_hidden_dense = 128
        self.embed_size = 32
        self._weights_updated = False

    def _build_model(self, maxlen=None, batch_size=None, phase="train"):
        if phase == "train":
            assert maxlen is not None
            input_layer = Input(shape=(maxlen, 7))
        else:
            assert batch_size is not None
            input_layer = Input(batch_shape=(batch_size, 1, 7))

        masked = Masking()(input_layer)

        embed = TimeDistributed(Dense(self.embed_size, activation="tanh"))(masked)
        embed = Dropout(0.3)(embed)

        recurrent = GRU(
            self.num_hidden_recurrent, return_sequences=True, stateful=phase == "predict", dropout_W=0.2, dropout_U=0.2
        )(embed)

        dense1 = TimeDistributed(Dense(self.num_hidden_dense, activation="tanh"))(recurrent)
        dense1 = Dropout(0.3)(dense1)

        output = TimeDistributed(Dense(7))(dense1)

        model = Model(input=input_layer, output=output)
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        return model

    def build_train_model(self, maxlen):
        self.train_model = self._build_model(maxlen=maxlen, phase="train")

    def build_predict_model(self, batch_size):
        self.predict_model = self._build_model(batch_size=batch_size, phase="predict")

    def load_predict_weights(self):
        self.train_model.save_weights("tmp_weights.h5", overwrite=True)
        self.predict_model.load_weights("tmp_weights.h5")
        self._weights_updated = False

    def reset_predict_model(self):
        self.predict_model.reset_states()

    def fit(self, X, y, max_epochs=20, validation_split=0.1):
        seq_len = X.shape[1]
        self.build_train_model(seq_len)
        split_idx = int((1. - validation_split) * X.shape[0])
        X, X_val = X[:split_idx], X[split_idx:]
        y, y_val = y[:split_idx], y[split_idx:]

        checkpoint = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        try:
            self.train_model.fit(
                X, y, nb_epoch=max_epochs, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stop]
            )

        except KeyboardInterrupt:
            logger.debug("Trianing interrupted! Restoring best weights and saving..")

        self.train_model.load_weights("best_weights.h5")
        self._weights_updated = True
        self.save()

    def predict_one_step(self, x):
        original_num_samples = x.shape[0]
        _batch_size = 32
        x = add_samples_until_divisible(x, _batch_size)

        if self.predict_model is None:
            self.build_predict_model(_batch_size)

        if self._weights_updated:
            self.load_predict_weights()

        return self.predict_model.predict(x, batch_size=_batch_size)[:original_num_samples, :, :]

    def evaluate(self, inputs):
        """
        :param inputs: a list of inputs for the model. In this case, it's a
                       one element list.
        :return: plotting artifacts: input, prediction, and error matrices
        """
        x = inputs[-1]
        pred = self.predict_one_step(x)
        return pred, (x-pred)**2

    def reset_predict_model_states(self):
        self.predict_model.reset_states()

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/TimeSeriesPredictor_%s.model" % int(time.time())

        logger.debug("Saving model to %s" % prefix)

        with open(prefix + ".json", "w") as of:
            of.write(self.train_model.to_json())

        self.train_model.save_weights(prefix + ".weights.h5")
        return prefix
