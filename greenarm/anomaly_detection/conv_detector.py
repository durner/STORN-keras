from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution1D, MaxPooling1D, Flatten
from scipy.ndimage import gaussian_filter
from greenarm.models.loss.binary_crossentropy import biased_binary_crossentropy
from greenarm.util import get_logger
import time
import numpy

logger = get_logger(__name__)


class CovNetAnomalyDetector(object):
    """
    The Conv Net Anomaly Detector is trained on a 1 dimensional array of the loss value of the STORN model.
    Using a deep convolutional network to find from the input loss the corresponding anomalies.
    """

    def __init__(self, bias=1.2, optimizer='rmsprop', monitor='val_acc'):
        # Object state
        self.model = None
        # bias positives that were predicted negative errors more to increase recall
        self.bias = bias
        # optimizer that shall be used
        self.optimizer = optimizer
        # monitor that shall be used
        self.monitor = monitor

    def build_model(self, seq_len=None):
        model = Sequential()
        model.add(Convolution1D(64, 4, border_mode='same', input_shape=(seq_len, 1)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=1))
        model.add(Activation("sigmoid"))
        model.compile(optimizer=self.optimizer, loss=self.biased_binary_crossentropy_wrapper, metrics=['acc'])
        return model

    def biased_binary_crossentropy_wrapper(self, y_true, y_pred):
        return biased_binary_crossentropy(self.bias, y_true, y_pred)

    def train(self, X, y, validation_split=0.1, max_epochs=500):
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        X = numpy.reshape(X, (n_samples, seq_len, 1))
        y = numpy.reshape(y, (n_samples, 1))
        X = numpy.apply_along_axis(lambda x: gaussian_filter(x, sigma=1.), axis=-1, arr=X)

        if self.model is None:
            self.model = self.build_model(seq_len=seq_len)

        split_idx = int((1. - validation_split) * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        checkpoint = ModelCheckpoint("best_anomaly_cnn_weights.h5", monitor=self.monitor, save_best_only=True,
                                     verbose=1)
        early_stop = EarlyStopping(monitor=self.monitor, patience=100, verbose=1)
        try:
            logger.info("Beginning anomaly detector training..")
            self.model.fit(
                [X_train], y_train,
                nb_epoch=max_epochs, validation_data=([X_val], y_val),
                callbacks=[checkpoint, early_stop]
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted! Restoring best weights and saving..")

        self.model.load_weights("best_anomaly_cnn_weights.h5")
        self.save()

    def score(self, X):
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        X = numpy.reshape(X, (n_samples, seq_len, 1))
        X = numpy.apply_along_axis(lambda x: gaussian_filter(x, sigma=1.), axis=-1, arr=X)
        return self.model.predict([X])

    def predict(self, X):
        return self.score(X) > 0.5

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/ConvAnomalyDetector_%s.model" % int(time.time())

        logger.info("Saving model to %s" % prefix)

        with open(prefix + ".json", "w") as of:
            of.write(self.model.to_json())

        self.model.save_weights(prefix + ".weights.h5")
        return prefix
