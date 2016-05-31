from keras.models import Model
from keras.layers import TimeDistributed, Dense, Input, GRU, Masking
from keras.regularizers import l1
import time
import numpy as np
from util import get_logger
logger = get_logger(__name__)


class TimeSeriesPredictor(object):
    def __init__(self, predict_batch_size=64):
        self.model = None
        self.predict_model = None
        self.num_hidden_recurrent = 100
        self.num_hidden_dense = 100
        self._weights_updated = False

    def build_model(self, maxlen):
        input_layer = Input(shape=(maxlen, 7))
        masked = Masking()(input_layer)
        recurrent = GRU(self.num_hidden_recurrent, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(masked)
        dense1 = TimeDistributed(Dense(self.num_hidden_dense, activation="tanh", W_regularizer=l1(0.01)))(recurrent)
        output = TimeDistributed(Dense(7))(dense1)

        self.model = Model(input=input_layer, output=output)
        self.model.compile(optimizer='rmsprop', loss='mean_squared_error')

    def build_predict_model(self):
        # TODO fix this
        test_input_layer = Input(shape=(1, 7))
        test_recurrent = GRU(
            self.num_hidden_recurrent, stateful=True, batch_input_shape=(1, 1, 7),
            return_sequences=False, dropout_W=0.2, dropout_U=0.2
        )(test_input_layer)
        test_dense1 = Dense(self.num_hidden_dense, activation="tanh", W_regularizer=l1(0.01))(test_recurrent)
        test_output = Dense(7)(test_dense1)

        self.predict_model = Model(input=test_input_layer, output=test_output)
        self.predict_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    def load_predict_weights(self):
        self.model.save_weights("tmp_weights.h5")
        self.predict_model.load_weights("tmp_weights.h5")
        self._weights_updated = False

    def fit(self, X, y, max_epochs=20, validation_split=0.1):
        seq_len = X.shape[1]
        self.build_model(seq_len)
        split_idx = int((1. - validation_split) * X.shape[0])
        X, X_val = X[:split_idx], X[split_idx:]
        y, y_val = y[:split_idx], y[split_idx:]
        try:
            for epoch_no in range(max_epochs):
                self.model.fit(X, y, nb_epoch=1, validation_data=(X_val, y_val))

        except KeyboardInterrupt:
            logger.debug("Interrupted after %s epochs! Saving.." % epoch_no)

        self._weights_updated = True
        self.save()

    def predict_one_step(self, x):
        if self._weights_updated:
            self.load_predict_weights()

        return self.predict_model.predict(np.asarray(x))

    def reset_predict_model_states(self):
        self.predict_model.reset_states()

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/TimeSeriesPredictor_%s.model" % int(time.time())

        logger.debug("Saving model to %s" % prefix)

        with open(prefix + ".json", "w") as of:
            of.write(self.model.to_json())

        self.model.save_weights(prefix + ".weights.h5")
        return prefix
