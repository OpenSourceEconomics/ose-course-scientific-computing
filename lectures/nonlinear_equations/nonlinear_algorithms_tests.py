"""Tests for nonlinear equations lecture."""
from functools import partial

import numpy as np
from nonlinear_algorithms import bisect
from nonlinear_algorithms import fixpoint
from nonlinear_algorithms import mcp_fischer
from nonlinear_algorithms import mcp_minmax
from nonlinear_algorithms import newton_method
from nonlinear_problems import get_cournot_problem
from nonlinear_problems import get_fischer_problem
from nonlinear_problems import get_mcp_problem
from scipy.optimize import bisect as sp_bisect


def test_1():
    """Bisection method is working."""

    def example(x):
        return x ** 3 - 2

    y = bisect(example, 1, 2)[0]

    np.testing.assert_almost_equal(y, 1.259921)
    np.testing.assert_almost_equal(sp_bisect(example, 1, 2), y)


def test_2():
    """Fixpoint method is working."""

    def example(x):
        return np.sqrt(x)

    y = fixpoint(example, 2)[0]
    np.testing.assert_almost_equal(y, 1.0, decimal=3)


def test_3():
    """Newton method is working."""
    c, e = np.array([0.6, 0.8]), 1.6
    cournot_p = partial(get_cournot_problem, c, e)

    y = newton_method(cournot_p, np.array([0.2, 0.2]))
    np.testing.assert_almost_equal(y, [0.8395676, 0.68879643])


def test_4():
    """MCP routine is working."""
    x0 = np.array([0.5, 0.5])
    a, b = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = mcp_minmax(get_mcp_problem, x0, a, b)["x"]
    np.testing.assert_almost_equal(y, [0.7937005, 1.0])


def test_5():
    """Smoothing example is working properly."""
    a = np.zeros(1)
    b = np.full(1, np.inf)
    x0 = np.zeros(1)

    x_sol = mcp_minmax(get_fischer_problem, x0, a, b)["x"]
    np.testing.assert_almost_equal(x_sol, 0.0035564)

    x_sol = mcp_fischer(get_fischer_problem, x0, a, b)["x"]
    np.testing.assert_almost_equal(x_sol, 2.0049876)
