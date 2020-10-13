import numpy as np


def get_cournot_problem(c, eta, q, jac=True):
    e = -1 / eta
    s = q.sum()
    fval = s ** e + e * s ** (e - 1) * q - c * q
    fjac = (
        e * s ** (e - 1) * np.ones([2, 2])
        + e * s ** (e - 1) * np.identity(2)
        + (e - 1) * e * s ** (e - 2) * np.outer(q, [1, 1])
        - np.diag(c)
    )

    if jac:
        ret = [fval, fjac]
    else:
        ret = fval

    return ret