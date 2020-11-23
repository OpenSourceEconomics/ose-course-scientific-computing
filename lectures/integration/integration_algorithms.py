import numpy as np


def quadrature_trapezoid(f, a, b, n):

    xvals = np.linspace(a, b, n + 1)
    fvals = np.tile(np.nan, n + 1)
    h = xvals[1] - xvals[0]

    weights = np.tile(h, n + 1)
    weights[0] = weights[-1] = 0.5 * h

    for i, xval in enumerate(xvals):
        fvals[i] = f(xval)

    return np.sum(weights * fvals)


def quadrature_simpson(f, a, b, n):

    if n % 2 == 0:
        raise Warning("n must be an odd integer. Increasing by 1")
        n += 1

    xvals = np.linspace(a, b, n)
    fvals = np.tile(np.nan, n)

    h = xvals[1] - xvals[0]

    weights = np.tile(np.nan, n)
    weights[0::2] = 2 * h / 3
    weights[1::2] = 4 * h / 3
    weights[0] = weights[-1] = h / 3

    for i, xval in enumerate(xvals):
        fvals[i] = f(xval)

    return np.sum(weights * fvals)


def quadrature_gauss_legendre(f, a, b, n):
    xvals, weights = np.polynomial.legendre.leggauss(n)

    fvals = np.tile(np.nan, n)
    for i, xval in enumerate(xvals):
        xval_trans = (b - a) * (xval + 1.0) / 2.0 + a
        fvals[i] = ((b - a) / 2.0) * f(xval_trans)

    return np.sum(weights * fvals)


def monte_carlo_naive_unidimensional(f, a=0, b=1, n=10, seed=123):
    np.random.seed(seed)
    xvals = np.random.uniform(size=n)
    fvals = np.tile(np.nan, n)
    weights = np.tile(1 / n, n)

    scale = b - a
    for i, xval in enumerate(xvals):
        fvals[i] = f(a + xval * (b - a))

    return scale * np.sum(weights * fvals)
