import numpy as np


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
