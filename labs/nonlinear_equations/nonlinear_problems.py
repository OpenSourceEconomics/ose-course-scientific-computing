"""Problems for nonlinear equations lab."""
import numpy as np
from scipy import optimize


def function_iteration_test_function(x):
    """Get test function for function iteration."""
    return np.sqrt(x)


def bisection_test_function(x):
    """Get test function for bisection."""
    return x ** 3 - 2


def newton_pathological_example_fjac(x, f):
    """Get Newton Pathological example jacobian."""
    return [optimize.approx_fprime(x[0], f, 1e-6)]


def newton_pathological_example_fval(x):
    """Get Newton Pathological example function value."""
    return np.cbrt(x) * np.exp(-(x ** 2))


def newton_pathological_example(x):
    """Get Newton Pathological example."""
    fval = newton_pathological_example_fval(x)
    fjac = newton_pathological_example_fjac(x, newton_pathological_example_fval)
    return fval, fjac


def get_cournot_problem(alpha, beta, q):
    """Get cournot problem."""
    qsum = q.sum()
    P = qsum ** (-alpha)
    P1 = -alpha * qsum ** (-alpha - 1)
    return P + (P1 - beta) * q


def get_spacial_market(x):
    """Create special market example."""
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
