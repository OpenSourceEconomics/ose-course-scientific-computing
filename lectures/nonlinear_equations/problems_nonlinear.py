"""Problems for nonlinear equations lecture."""
import numpy as np


def get_cournot_problem(c, eta, q, jac=True):
    """Create Cournot problem."""
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
    """Create MCP problem."""
    x, y = z
    f = [1 + x * y - 2 * x ** 3 - x, 2 * x ** 2 - y]
    j = [[y - 6 * x ** 2 - 1, x], [4 * x, -1]]

    return np.array(f), np.array(j)


def get_spacial_market(x):
    """Create special market."""
    a = np.array
    as_ = a([9, 3, 18])
    bs = a([1, 2, 1])
    ad = a([42, 54, 51])
    bd = a([2, 3, 1])
    c = a([[0, 3, 9], [3, 0, 3], [6, 3, 0.0]])

    quantities = x.reshape((3, 3))
    ps = as_ + bs * quantities.sum(0)
    pd = ad - bd * quantities.sum(1)
    ps, pd = np.meshgrid(ps, pd)
    fval = (pd - ps - c).flatten()

    return fval, None


def get_fischer_problem(x):
    """Create Fischer problem."""
    return np.array(1.01 - (x - 1.0) ** 2), None
