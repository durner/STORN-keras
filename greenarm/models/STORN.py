"""
Implementation of the STORN model from the paper.
"""
import theano.tensor as tensor
import theano
import numpy
from keras.layers import Input, Masking


class STORNRecognitionModel:
    def __init__(self):
        pass

    def fit(self, X, train_seq):

        input_layer_train = Input(shape=(train_seq, 7))
        input_layer = Input(batch_shape=(1, 1, 7))

        masked = Masking()(input_layer_train)

    def predict(self):
        pass


class STORNGeneratingModel:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class STORNPriorModel:
    def __init__(self):
        pass

    def fit(self):
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
