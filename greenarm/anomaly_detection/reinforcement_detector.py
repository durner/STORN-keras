from qlearning4k import Agent
import numpy as np

from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Dropout, Masking, SimpleRNN, Flatten

from greenarm.anomaly_detection.game.anomaly_detect_game import AnomalyDetection
from greenarm.util import get_logger
import time

logger = get_logger(__name__)

# Basic idea: run the data through the STORN model, and get back
#   - error (KL especially)
#   - prediction
# Combine this with the original data, and feed an RNN a concat-merge of these

RecurrentLayer = SimpleRNN


class ReinforcementLearningAnomalyDetector(object):
    def __init__(
            self, recurrent=False, n_deep_dense_input=0, num_hidden_dense=64, n_deep_recurrent=0,
            num_hidden_recurrent=30, n_deep_dense=0, activation="tanh", dropout=0.3
    ):
        self.recurrent = recurrent
        self.n_deep_dense_input = n_deep_dense_input
        self.num_hidden_dense = num_hidden_dense
        self.num_hidden_recurrent = num_hidden_recurrent
        self.n_deep_recurrent = n_deep_recurrent
        self.n_deep_dense = n_deep_dense
        self.activation = activation
        self.dropout = dropout

        self.model = None
        self.agent = None

    def build_model(self, input_dims, maxlen=None):
        if self.recurrent:
            input_layer = Input(shape=(1, maxlen, input_dims))
            masked = Masking()(input_layer)
            x_in = masked

            for i in range(self.n_deep_dense_input):
                x_in = TimeDistributed(Dense(self.num_hidden_dense, activation=self.activation))()
                if self.dropout != 0.0:
                    x_in = Dropout(self.dropout)(x_in)

            deep = x_in

            for i in range(self.n_deep_recurrent):
                is_last_recurrent = i == (self.n_deep_recurrent - 1)
                deep = RecurrentLayer(
                    self.num_hidden_recurrent,
                    return_sequences=not is_last_recurrent,
                    stateful=False,
                    dropout_W=self.dropout,
                    dropout_U=self.dropout,
                    init="glorot_normal"
                )(deep)

            for i in range(self.n_deep_dense):
                deep = Dense(self.num_hidden_dense, activation=self.activation)(deep)
                if self.dropout != 0.0:
                    deep = Dropout(self.dropout)(deep)

            output = Dense(2)(deep)
        else:
            input_layer = Input(shape=(1, input_dims))
            flat = Flatten()(input_layer)

            x_in = flat
            for i in range(self.n_deep_dense):
                x_in = Dense(self.num_hidden_dense, activation=self.activation)(x_in)

            output = Dense(2)(x_in)

        model = Model(input=[input_layer], output=output)
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def train(self, x, y, validation_split=0.1, max_epochs=10000):
        seq_len = x.shape[1]
        data_dim = x.shape[2]
        if self.model is None:
            self.model = self.build_model(
                data_dim, maxlen=seq_len
            )

        split_idx = int((1. - validation_split) * x.shape[0])

        x, x_val = x[:split_idx], x[split_idx:]

        y, y_val = y[:split_idx], y[split_idx:]

        anomaly_game = AnomalyDetection(x, y, valid_window=(0, 33), sequence_like_states=self.recurrent)
        self.agent = Agent(self.model, memory_size=-1, nb_frames=None)  # TODO figure out memory vs nb_frames

        try:
            logger.debug("Beginning anomaly detector training..")
            self.agent.train(anomaly_game, nb_epoch=max_epochs)

        except KeyboardInterrupt:
            logger.debug("Training interrupted! Restoring best weights and saving..")

        val_acc, val_detected_anomalies = self.evaluate(x_val, y_val, return_detections=True)
        logger.debug("Final val accuracy: %s" % val_acc)
        logger.debug("Detections: %s" % val_detected_anomalies[:10])

        self.save()

    def evaluate(self, x, y, return_detections=False):
        val_game = AnomalyDetection(x, y, valid_window=(-5, 10), sequence_like_states=self.recurrent)
        self.agent.play(val_game, nb_epoch=x.shape[0], visualize=False)

        accuracy = float(val_game.num_spotted_correctly) / sum([len(y_row) for y_row in y])
        if return_detections:
            return accuracy, val_game.detected_anomalies
        return accuracy

    def predict_coarse(self, x):
        game = AnomalyDetection(
            x, [np.array([])] * x.shape[0], valid_window=(-5, 10), sequence_like_states=self.recurrent
        )
        self.agent.play(game, nb_epoch=x.shape[0], visualize=False)
        res = np.zeros(shape=(x.shape[0],))
        for i in range(x.shape[0]):
            if game.detected_anomalies[i]:
                res[i] = 1.

        return res

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/RNNAnomalyDetector_%s.model" % int(time.time())

        logger.debug("Saving model to %s" % prefix)

        with open(prefix + ".json", "w") as of:
            of.write(self.model.to_json())

        self.model.save_weights(prefix + ".weights.h5")
        return prefix
