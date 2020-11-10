"""Problems for nonlinear equations lecture."""
import numpy as np
from scipy import optimize


def function_iteration_test_function(x):
    return np.sqrt(x)


def bisection_test_function(x):
    return x ** 3 - 2


def newton_pathological_example_fjac(x, f):
    return [optimize.approx_fprime(x[0], f, 1e-6)]


def newton_pathological_example_fval(x):
    return np.cbrt(x) * np.exp(-(x ** 2))


def newton_pathological_example(x):
    fval = newton_pathological_example_fval(x)
    fjac = newton_pathological_example_fjac(x, newton_pathological_example_fval)
    return fval, fjac


def get_cournot_problem(alpha, beta, q):
    qsum = q.sum()
    P = qsum ** (-alpha)
    P1 = -alpha * qsum ** (-alpha - 1)
    return P + (P1 - beta) * q


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
