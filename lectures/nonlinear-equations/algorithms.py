import numpy as np
from numpy.linalg import norm


def bisect(f, a, b, tol=1.5e-8):

    s = np.sign(f(a))

    x = (a + b) / 2
    d = (b - a) / 2

    while d > tol:
        d = d / 2
        if s == np.sign(f(x)):
            x = x + d
        else:
            x = x - d

    return x


def fixpoint(f, x0, tol=10e-5):
    """ Fixed point algorithm """
    e = 1
    while e > tol:
        x = f(x0)  # fixed point equation
        e = norm(x0 - x)  # error at the current step
        x0 = x
    return x


def newton_method(f, df, x0, tol=1.5e-8):

    """inspired by https://www.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/"""

    xn = x0.copy()

    while True:
        fxn = f(xn)
        if np.abs(fxn) < tol:
            return xn
        else:
            dfxn = df(xn)
            xn = xn - fxn / dfxn
