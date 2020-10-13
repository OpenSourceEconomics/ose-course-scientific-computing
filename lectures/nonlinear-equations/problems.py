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
        ret = fval[0], fjac
    else:
        ret = fval

    return ret


def get_mcp_problem(z):
    x, y = z
    f = [1 + x * y - 2 * x ** 3 - x, 2 * x ** 2 - y]
    J = [[y - 6 * x ** 2 - 1, x], [4 * x, -1]]
    return np.array(f), np.array(J)
