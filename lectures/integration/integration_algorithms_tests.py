import numpy as np
from integration_algorithms import monte_carlo_naive
from integration_algorithms import quadrature_gauss_legendre
from integration_algorithms import quadrature_simpson
from integration_algorithms import quadrature_trapezoid
from scipy.stats import uniform


def test_1():
    approaches = list()
    approaches += [quadrature_gauss_legendre, quadrature_simpson, quadrature_trapezoid]
    approaches += [monte_carlo_naive]
    for approach in approaches:
        np.testing.assert_almost_equal(approach(uniform.pdf, 0, 1, 11), 1.0)
