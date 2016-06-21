import keras.backend as K


def sample_gauss(mu, sig, batch_size, dim_size):
    epsilon = K.random_normal(shape=(batch_size,
                                     dim_size),
                              mean=0., std=1.,
                              dtype="float32")
    sample = mu + sig * epsilon
    return sample
