import numpy as np


def golden_search(f, a, b, tol=1e-5):
    alpha1 = (3 - np.sqrt(5)) / 2
    alpha2 = (np.sqrt(5) - 1) / 2
    x1 = a + alpha1 * (b - a)
    f1 = f(x1)
    x2 = a + alpha2 * (b - a)
    f2 = f(x2)
    d = alpha1 * alpha2 * (b - a)

    while d > tol:
        d = d * alpha2
        if f2 < f1:
            x2 = x1
            x1 = x1 - d
            f2 = f1
            f1 = f(x1)
        else:
            x1 = x2
            x2 = x2 + d
            f1 = f2
            f2 = f(x2)

        if f2 > f1:
            x = x2
        else:
            x = x1

    return x
