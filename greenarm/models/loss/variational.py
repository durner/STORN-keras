"""
TODO: define the log likelihood upper bound for variational bayes in here
"""
import numpy as np
import theano.tensor as T


def divergence(mu1, sigma1, mu2, sigma2):
    """
    Computes the KL divergence of p || q, where p is gaussian
    with (mu1, sigma1) and q is gaussian with (mu2, sigma2).
    The last dimension is expected to be the dimensionality of a single
    data point x_t in time.

    :param mu1: a tensor for the first gaussian's mean
    :param sigma1: a tensor for the first gaussian's std. deviation
    :param mu2: a tensor for the second gaussian's mean
    :param sigma2: a tensor for the second gaussian's std. deviation
    :return: tensor with the KL divergence for each component k
    """
    term = T.sum(T.log(sigma2) - T.log(sigma1) +
                 0.5 * (sigma1 ** 2 + (mu1 - mu2) ** 2) /
                 (sigma2 ** 2) - 1, axis=-1)
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
    nll = 0.5 * T.sum(T.sqr(x - mu) / sigma ** 2 + 2 * T.log(sigma) +
                      T.log(2 * np.pi), axis=-1)
    return nll


def gauss_mixture():
    pass
