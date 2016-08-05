import time
from keras.callbacks import ModelCheckpoint, EarlyStopping, RemoteMonitor
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Dropout, Masking, GRU
from greenarm.util import get_logger

logger = get_logger(__name__)
RecurrentLayer = GRU


class RNNAnomalyDetector(object):
    """
    The RNN Anomaly Detector is trained on a
    1 dimensional array of the loss value of
    the STORN model.
    """

    def __init__(self, n_deep=1, num_hidden_dense=4, num_hidden_recurrent=4,
                 activation="relu", dropout=0.0):

        # Network configuration
        self.n_deep = n_deep
        self.n_hidden_dense = num_hidden_dense
        self.n_hidden_recurrent = num_hidden_recurrent
        self.activation = activation
        self.dropout = dropout

        # Object state
        self.model = None

        # Misc
        self.monitor = True

    def build_model(self, seq_len=None):
        loss_input = Input(shape=(seq_len, 33))
        masked_input = Masking()(loss_input)

        # deep feature extraction for the loss
        deep = masked_input
        for i in range(self.n_deep):
            deep = TimeDistributed(Dense(self.n_hidden_dense, activation=self.activation))(deep)
            if self.dropout != 0.0:
                deep = Dropout(self.dropout)(deep)

        # RNN node to process the loss time-series
        rnn = RecurrentLayer(self.n_hidden_recurrent, return_sequences=False, stateful=False)(deep)

        # deep feature extraction for the RNN output
        output = rnn
        for i in range(self.n_deep):
            output = Dense(self.n_hidden_dense, activation=self.activation)(output)
            if self.dropout != 0.0:
                output = Dropout(self.dropout)(output)

        # The RNN output is in the end of a sequence, and
        # corresponds to the prediction "there was an anomaly"
        # in the whole time-series or not
        output = Dense(1, activation="sigmoid")(output)

        model = Model(input=[loss_input], output=output)
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
        return model

    def train(self, X, y, validation_split=0.1, max_epochs=100):
        n_samples = X.shape[0]
        seq_len = X.shape[1]

        if self.model is None:
            self.model = self.build_model(seq_len=seq_len)

        split_idx = int((1. - validation_split) * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        checkpoint = ModelCheckpoint("best_anomaly_weights.h5", monitor='val_acc', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_acc', patience=150, verbose=1)

        callbacks = [checkpoint, early_stop]
        if self.monitor:
            monitor = RemoteMonitor(root='http://localhost:9000')
            callbacks = callbacks + [monitor]
        try:
            logger.info("Beginning anomaly detector training..")
            self.model.fit(
                [X_train], y_train,
                nb_epoch=max_epochs, validation_data=([X_val], y_val),
                callbacks=callbacks
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted! Restoring best weights and saving..")

        self.model.load_weights("best_anomaly_weights.h5")
        self.save()
        
    def score(self, X):
        return self.model.predict([X])

    def predict(self, X):
        return self.score(X) > 0.5

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/RNNAnomalyDetector_%s.model" % int(time.time())

        logger.info("Saving model to %s" % prefix)
