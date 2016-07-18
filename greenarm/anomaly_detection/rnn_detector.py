
# Basic idea: run the data through the STORN model, and get back
#   - error (KL especially)
#   - prediction
# Combine this with the original data, and feed an RNN a concat-merge of these


class RNNAnomalyDetector(object):
    def __init__(
            self, n_deep_dense_input=1, num_hidden_dense=128, n_deep_recurrent=2, num_hidden_recurrent=128,
            n_deep_dense=2, activation="tanh", dropout=0.3
    ):
        self.n_deep_dense_input = n_deep_dense_input
        self.num_hidden_dense = num_hidden_dense
        self.num_hidden_recurrent = num_hidden_recurrent
        self.n_deep_recurrent = n_deep_recurrent
        self.n_deep_dense = n_deep_dense
        self.activation = activation
        self.dropout = dropout

        self.model = None

    def build_model(self, time_series_input_dims, predictor_output_dims, maxlen=None):
        ts_input_layer = Input(shape=(maxlen, time_series_input_dims))
        ts_masked = Masking()(ts_input_layer)
        pred_input_layer = Input(shape=(maxlen, predictor_output_dims))
        pred_masked = Masking()(pred_input_layer)

        input_layer = merge([ts_masked, pred_masked], mode="concat")

        x_in = input_layer

        for i in range(self.n_deep_dense_input):
            x_in = TimeDistributed(Dense(self.num_hidden_dense, activation=self.activation))(x_in)
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

        output = Dense(1, activation="sigmoid")(deep)

        model = Model(input=[ts_input_layer, pred_input_layer], output=output)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    def train(self, x_sequence_data, x_predictor_output, Y, validation_split=0.1, max_epochs=1000):
        seq_len = x_sequence_data.shape[1]
        if self.model is None:
            self.model = self.build_model(
                x_sequence_data.shape[2], x_predictor_output.shape[2], maxlen=seq_len
            )

        split_idx = int((1. - validation_split) * x_sequence_data.shape[0])
        x_sequence_data, x_sequence_data_val = x_sequence_data[:split_idx], x_sequence_data[split_idx:]
        x_predictor_output, x_predictor_output_val = x_predictor_output[:split_idx], x_predictor_output[split_idx:]

        Y, Y_val = Y[:split_idx], Y[split_idx:]

        checkpoint = ModelCheckpoint("best_anomaly_weights.h5", monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=150, verbose=1)
        try:
            logger.debug("Beginning anomaly detector training..")
            self.model.fit(
                [x_sequence_data, x_predictor_output], Y,
                nb_epoch=max_epochs, validation_data=([x_sequence_data_val, x_predictor_output_val], Y_val),
                callbacks=[checkpoint, early_stop]
            )

        except KeyboardInterrupt:
            logger.debug("Trianing interrupted! Restoring best weights and saving..")

        self.model.load_weights("best_anomaly_weights.h5")
        self._weights_updated = True
        self.save()

    def predict(self, x_sequence_data, x_predictor_output):
        return self.model.predict([x_sequence_data, x_predictor_output]) > 0.5

    def save(self, prefix=None):
        if prefix is None:
            prefix = "saved_models/RNNAnomalyDetector_%s.model" % int(time.time())

        logger.debug("Saving model to %s" % prefix)
