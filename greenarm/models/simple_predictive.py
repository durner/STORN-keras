import time

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, RemoteMonitor
from keras.layers import TimeDistributed, Dense, Input, GRU, Masking, Dropout
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from greenarm.models.grid_search.keras_grid import ModelSelector
from greenarm.util import get_logger, add_samples_until_divisible

logger = get_logger(__name__)

RecurrentLayer = GRU


class TimeSeriesPredictor(object):
    def __init__(self, n_deep_dense=5, n_deep_dense_input=3, n_deep_recurrent=4, num_hidden_recurrent=128,
                 num_hidden_dense=32, dropout=0, activation="sigmoid", monitor=False):
        self.n_deep_dense = n_deep_dense
        self.n_deep_dense_input = n_deep_dense_input
        self.n_deep_recurrent = n_deep_recurrent
        self.dropout = dropout
        self.activation = activation
        self.num_hidden_recurrent = num_hidden_recurrent
        self.num_hidden_dense = num_hidden_dense

        self.train_model = None
        self.predict_model = None
        self._weights_updated = False

        # Misc
        self.monitor = monitor

    def get_params(self, deep=True):
        return {
            "n_deep_dense": self.n_deep_dense,
            "n_deep_dense_input": self.n_deep_dense_input,
            "n_deep_recurrent": self.n_deep_recurrent,
            "dropout": self.dropout,
            "activation": self.activation,
            "num_hidden_recurrent": self.num_hidden_recurrent,
            "num_hidden_dense": self.num_hidden_dense
        }

    def set_params(self, **params):
        for param_name, param in params.items():
            setattr(self, param_name, param)

        return self

    def _build_model(self, maxlen=None, batch_size=None, phase="train"):
        if phase == "train":
            assert maxlen is not None
            input_layer = Input(shape=(maxlen, 7))
        else:
            assert batch_size is not None
            input_layer = Input(batch_shape=(batch_size, 1, 7))

        masked = Masking()(input_layer)

        x_in = masked
        for i in range(self.n_deep_dense_input):
            x_in = TimeDistributed(Dense(self.num_hidden_dense, activation=self.activation))(x_in)
            if self.dropout != 0.0:
                x_in = Dropout(self.dropout)(x_in)
        deep = x_in
        for i in range(self.n_deep_recurrent):
            deep = RecurrentLayer(
                self.num_hidden_recurrent,
                return_sequences=True,
                stateful=phase == "predict",
                dropout_W=self.dropout,
                dropout_U=self.dropout,
                init="glorot_normal"
            )(deep)

        for i in range(self.n_deep_dense):
            deep = TimeDistributed(Dense(self.num_hidden_dense, activation=self.activation))(deep)
            if self.dropout != 0.0:
                deep = Dropout(self.dropout)(deep)

        output = TimeDistributed(Dense(7))(deep)

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
        early_stop = EarlyStopping(monitor='val_loss', patience=150, verbose=1)

        callbacks = [checkpoint, early_stop]
        if self.monitor:
            monitor = RemoteMonitor(root='http://localhost:9000')
            callbacks = callbacks + [monitor]

        try:
            self.train_model.fit(
                X, y, nb_epoch=max_epochs, validation_data=(X_val, y_val), callbacks=callbacks
            )

        except KeyboardInterrupt:
            logger.info("Training interrupted! Restoring best weights and saving..")

        self.train_model.load_weights("best_weights.h5")
        self._weights_updated = True
        self.save()

    def predict(self, X):
        return self.train_model.predict(X)

    def predict_one_step(self, x):
        original_num_samples = x.shape[0]
        _batch_size = 32
        x = add_samples_until_divisible(x, _batch_size)

        if self.predict_model is None:
            self.build_predict_model(_batch_size)

        if self._weights_updated:
            self.load_predict_weights()

        return self.predict_model.predict(x, batch_size=_batch_size)[:original_num_samples, :, :]

    def evaluate_online(self, inputs, ground_truth):
        """
        :param inputs: a list of inputs for the model. In this case, it's a
                       one element list.
        :param ground_truth: the expected value to compare to
        :return: plotting artifacts: prediction, and error matrices
        """
        x = inputs[-1]
        pred = self.predict_one_step(x)
        error = (ground_truth - pred) ** 2
        return pred, np.mean(error, axis=-1)

    def evaluate_offline(self, inputs, ground_truth):
        x = inputs[-1]
        pred = self.train_model.predict(x)
        error = (ground_truth - pred) ** 2
        return pred, np.mean(error, axis=-1)

    def reset_predict_model_states(self):
        self.predict_model.reset_states()

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/TimeSeriesPredictor_%s.model" % int(time.time())

        logger.info("Saving model to %s" % prefix)

        with open(prefix + ".json", "w") as of:
            of.write(self.train_model.to_json())

        self.train_model.save_weights(prefix + ".weights.h5")
        return prefix


def run_tsp_grid_search(inputs, target):
    def model_build_fn(n_deep=3, activation='relu', dropout=0):
        tsp = TimeSeriesPredictor(n_deep=n_deep, activation=activation, dropout=dropout)
        return tsp._build_model(maxlen=inputs.shape[1],
                                phase='train')

    selector = ModelSelector(KerasRegressor(build_fn=model_build_fn))
    param_grid = {
        'n_deep': [0, 5, 10],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'dropout': [0, 0.2]
    }
    selector.score_hyper_params(inputs, target, param_grid=param_grid)
