"""
TODO: define the log likelihood upper bound for variational bayes in here
"""
import numpy as np
import keras.backend as K


def divergence(mu1, sigma1, mu2=0, sigma2=1):
    """
    Computes the KL divergence of p || q, where p is gaussian
    with (mu1, sigma1) and q is gaussian with (mu2, sigma2).
    The last dimension is expected to be the dimensionality of a single
    data point x_t in time.

    :param mu1: a tensor for the first gaussian's mean
    :param sigma1: a tensor for the first gaussian's std. deviation
    :param mu2: a tensor for the second gaussian's mean
    :param sigma2: a tensor for the second gaussian's std. deviation
    :return: scalar for the KL divergence
    """

    """
    We sum over the components k (e.g. 7), which are the last dimension
    of our data points because the gaussians are isotropic.

    Then we use mean() over the rest of the dimensions (sequences
    in the data batch, elements in the sequence) to make every point equally
    important.
    """
    term = K.sum(K.log(sigma2 / sigma1) +
                 ((K.square(sigma1) + K.square(mu1 - mu2)) / (2 * K.square(sigma2))) - 0.5, axis=-1)
    return term


def gauss(x, mu, sigma):
    """
    Computes the negative log likelihood for a set of X values with
    respect to a isotropic gaussian defined through mu and sigma.
    The last dimension is expected to be the dimensionality of a single
    data point x_t in time.

    :param x: a tensor for the variables
    :param mu: the mean of the gaussian distribution
    :param sigma: the std. deviation of the gaussian distribution
    :return:
    """

    """
    We sum over the components k (e.g. 7), which are the last dimension
    of our data points because the gaussians are isotropic.

    Then we use mean() over the rest of the dimensions (sequences
    in the data batch, elements in the sequence) to make every point equally
    important.
    """
    nll = 0.5 * K.sum(K.square(x - mu) / K.square(sigma) + 2 * K.log(sigma) +
                      K.log(2 * np.pi), axis=-1)
    return nll


def keras_divergence(x, output_statistics):
    x_dim = 7
    # the output has 2*x_dim, the mu and sigma of x|z
    # and then 4*latent_dim, the mu, sigma of z|x, and mu, sigma of z (prior)
    latent_dim = (x.shape[-1] - x_dim * 2) / 4
    return K.mean(divergence(output_statistics[:, :, 2 * x_dim:2 * x_dim + latent_dim],
                             output_statistics[:, :, 2 * x_dim + latent_dim:2 * x_dim + 2 * latent_dim],
                             output_statistics[:, :, 2 * x_dim + 2 * latent_dim: 2 * x_dim + 3 * latent_dim],
                             output_statistics[:, :, 2 * x_dim + 3 * latent_dim:],
                             ))


def keras_gauss(x, output_statistics):
    x_dim = 7
    # the output has 2*x_dim, the mu and sigma of x|z
    # and then 4*latent_dim, the mu, sigma of z|x, and mu, sigma of z (prior)
    x = x[:, :, :x_dim]
    return K.mean(gauss(x, output_statistics[:, :, :x_dim], output_statistics[:, :, x_dim:2 * x_dim]))


def keras_variational(x, output_statistics):
    """
    A wrapper around the variational upper bound loss for keras.

    :param x: the x we want to compute the NLL for
    :param output_statistics: the statistics of the distributions for the
            generating and recognition model. First third is the generating model's
            mu and sigma, second third is the recognition model's mu and sigma.
            The last third represents the mu and sigma of the trending prior.
    :return: the keras loss tensor
    """

    x_dim = 7
    # the output has 2*x_dim, the mu and sigma of x|z
    # and then 4*latent_dim, the mu, sigma of z|x, and mu, sigma of z (prior)
    latent_dim = (x.shape[-1] - x_dim * 2) / 4
    x_stripped = x[:, :, :x_dim]
    expect_term = gauss(x_stripped, output_statistics[:, :, :x_dim], output_statistics[:, :, x_dim:2 * x_dim])
    kl_term = divergence(output_statistics[:, :, 2 * x_dim:2 * x_dim + latent_dim],
                         output_statistics[:, :, 2 * x_dim + latent_dim:2 * x_dim + 2 * latent_dim],
                         output_statistics[:, :, 2 * x_dim + 2 * latent_dim: 2 * x_dim + 3 * latent_dim],
                         output_statistics[:, :, 2 * x_dim + 3 * latent_dim:],
                         )
    return kl_term + expect_term


def mean_sigma(x, output_statistics):
    sigma = output_statistics[:, :, 7:14]
    return K.mean(sigma)


def mu_minus_x(x, output_statistics):
    x_stripped = x[:, :, :7]
    mu = output_statistics[:, :, :7]
    return K.mean(K.sum(K.square(x_stripped - mu), axis=-1))


def gauss_mixture():
    pass
