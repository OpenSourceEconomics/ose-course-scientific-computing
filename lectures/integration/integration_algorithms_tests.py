"""Tests for integration lab."""
from functools import partial

import chaospy as cp
import numpy as np
from labs.integration.integration_algorithms import monte_carlo_naive_one
from labs.integration.integration_algorithms import monte_carlo_naive_two_dimensions
from labs.integration.integration_algorithms import monte_carlo_quasi_two_dimensions
from labs.integration.integration_algorithms import quadrature_gauss_legendre_one
from labs.integration.integration_algorithms import quadrature_gauss_legendre_two
from labs.integration.integration_algorithms import quadrature_newton_simpson_one
from labs.integration.integration_algorithms import quadrature_newton_trapezoid_one
from scipy.stats import uniform


def test_1():
    """Return test 1."""
    approaches = []
    approaches += [quadrature_gauss_legendre_one, quadrature_newton_simpson_one]
    approaches += [monte_carlo_naive_one, quadrature_newton_trapezoid_one]
    for approach in approaches:
        np.testing.assert_almost_equal(approach(uniform.pdf, 0, 1, 11), 1.0)


def test_2():
    """Return test 2."""
    p_gc_legendre_two = partial(quadrature_gauss_legendre_two, a=0, b=1)

    approaches = []
    approaches += [monte_carlo_naive_two_dimensions, monte_carlo_quasi_two_dimensions]
    approaches += [p_gc_legendre_two]
    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1))
    for approach in approaches:
        np.testing.assert_almost_equal(approach(distribution.pdf), 1.0)
