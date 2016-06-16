"""
Implementation of the STORN model from the paper.
"""
import theano.tensor as tensor
import theano
import numpy
from keras.engine import merge
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Dropout, GRU, Lambda
from greenarm.models.loss.variational import keras_variational
from greenarm.models.sampling.sampling import sample_gauss


class STORNRecognitionModel:
    def __init__(self):
        self.train_rnn_recogn_stats = 0
        self.train_input = 0
        self.train_z = 0
        self.predict_rnn_recogn_stats = 0
        self.preduct_input = 0
        self.predict_z = 0

    def _build(self, phase, seq_shape, joint_shape):
        if phase == "train":
            input_layer = Input(shape=(seq_shape, joint_shape))
        else:
            input_layer = Input(batch_shape=(1, 1, joint_shape))

        embed1 = TimeDistributed(Dense(32, activation="tanh"))(input_layer)
        embed1 = Dropout(0.3)(embed1)
        rnn_recogn = GRU(128, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(embed1)
        rnn_recogn_stats = TimeDistributed(Dense(14, activation="relu"))(rnn_recogn)

        # sample z from the distribution in X
        sample_z = TimeDistributed(Lambda(self.do_sample, output_shape=(joint_shape,)))(rnn_recogn_stats)

        return rnn_recogn_stats, input_layer, sample_z

    def build(self, seq_shape, joint_shape):
        self.train_rnn_recogn_stats, self. train_input, self.train_z = self._build("train", seq_shape, joint_shape)
        self.predict_rnn_recogn_stats, self.preduct_input, self.predict_z = self._build("predict", seq_shape, joint_shape)

    def predict(self):
        pass

    @staticmethod
    def do_sample(statistics):
        # split in half
        dim = statistics.shape[-1] / 2
        mu = statistics[:, :dim]
        sigma = statistics[:, dim:]

        # sample with this mean and variance
        return sample_gauss(mu, sigma)


class STORNGeneratingModel:
    def __init__(self):
        self.train_model = 0
        self.train_input = 0
        self.predict_model = 0
        self.preduct_input = 0

    def _build(self, phase, recognition_model, seq_shape, joint_shape):
        if phase == "train":
            input_layer = Input(shape=(seq_shape, joint_shape))
            rec_z = recognition_model.train_z
            rec_input = recognition_model.train_input
            rec = recognition_model.train_rnn_recogn_stats
        else:
            input_layer = Input(batch_shape=(1, 1, joint_shape))
            rec_z = recognition_model.predict_z
            rec_input = recognition_model.predict_input
            rec = recognition_model.predict_rnn_recogn_stats

        gen_input = merge(inputs=[input_layer, rec_z], mode='concat')
        embed2 = TimeDistributed(Dense(32, activation="relu"), input_shape=(seq_shape, 2*joint_shape))(gen_input)
        embed2 = Dropout(0.3)(embed2)
        rnn_gen = GRU(128, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(embed2)
        rnn_gen_stats = TimeDistributed(Dense(14, activation="relu"))(rnn_gen)

        output = merge([rnn_gen_stats, rec], mode='concat')
        model = Model(input=[rec_input, input_layer], output=output)
        model.compile(optimizer='rmsprop', loss=keras_variational)

        return model, input_layer

    def build(self, recognition_model, seq_shape, joint_shape):
        self.train_model, self.train_input = self._build("train", recognition_model, seq_shape, joint_shape)
        self.predict_model, self.preduct_input = self._build("predict", recognition_model, seq_shape, joint_shape)

    def predict(self):
        pass


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
