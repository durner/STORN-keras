import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams

seed_rng = np.random.RandomState(np.random.randint(1024))
max_int = np.iinfo(np.int32).max
seed = seed_rng.randint(max_int)
theano_rng = MRG_RandomStreams(seed)


def sample_gauss(mu, sig):
    # put the sample as the mid dimension
    epsilon = theano_rng.normal(size=(mu.shape[0],
                                      mu.shape[1]),
                                avg=0., std=1.,
                                dtype=mu.dtype)
    sample = mu + sig * epsilon
    return sample
