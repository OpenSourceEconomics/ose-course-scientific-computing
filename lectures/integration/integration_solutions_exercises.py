"""Solutions for integration lecture."""
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from integration_algorithms import monte_carlo_naive_two_dimensions as mc_naive
from integration_algorithms import monte_carlo_quasi_two_dimensions as mc_quasi
from integration_algorithms import quadrature_gauss_legendre_one
from integration_algorithms import quadrature_gauss_legendre_two as gc_legendre_two
from integration_algorithms import quadrature_newton_simpson_one
from integration_algorithms import quadrature_newton_trapezoid_one
from integration_problems import problem_kinked
from integration_problems import problem_smooth
from temfpy.integration import discontinuous


def test_exercise_1():
    """Get solution for exercise 1."""
    index = product(["Smooth", "Kinked"], [5, 11, 21, 31])
    index = pd.MultiIndex.from_tuples(index, names=("Function", "Nodes"))

    df_errors = pd.DataFrame(columns=["Trapezoid", "Simpson", "Gauss", "Truth"], index=index)

    df_errors.loc[("Smooth", slice(None)), "Truth"] = np.exp(1) - np.exp(-1)
    df_errors.loc[("Kinked", slice(None)), "Truth"] = 4 / 3

    for label, test_function in [("Smooth", problem_smooth), ("Kinked", problem_kinked)]:
        p_trapezoid = partial(quadrature_newton_trapezoid_one, test_function, -1, 1)
        p_simpson = partial(quadrature_newton_simpson_one, test_function, -1, 1)
        p_gauss = partial(quadrature_gauss_legendre_one, test_function, -1, 1)
        for nodes in df_errors.index.get_level_values("Nodes"):
            index = (label, nodes)
            df_errors.loc[index, "Trapezoid"] = np.abs(p_trapezoid(nodes))
            df_errors.loc[index, "Simpson"] = np.abs(p_simpson(nodes))
            df_errors.loc[index, "Gauss"] = np.abs(p_gauss(nodes))


def test_exercise_2():
    """Get solution for exercise 2."""
    index = pd.Index(np.linspace(100, 10000, dtype=int), name="Nodes")

    df_results = pd.DataFrame(columns=["Naive", "Sobol", "Halton", "Gauss", "Truth"], index=index)

    # Determining the true value of the double integral is straightforward as we can tackle each
    # dimension separately and than just multiply them.
    integrand = 1 / 5 * np.exp(5 * 0.5) - 1 / 5 * np.exp(5 * 0)
    df_results["Truth"] = integrand * integrand

    discontinuous = partial(discontinuous, u=(0.5, 0.5), a=(5, 5))

    mc_quasi_halton = partial(mc_quasi, discontinuous, 0, 1, rule="halton")
    mc_quasi_sobol = partial(mc_quasi, discontinuous, 0, 1, rule="sobol")
    gc_legendre = partial(gc_legendre_two, discontinuous, 0, 1)

    for nodes in df_results.index.get_level_values("Nodes"):
        df_results.loc[nodes, "Naive"] = mc_naive(discontinuous, 0, 1, nodes)
        df_results.loc[nodes, "Halton"] = mc_quasi_halton(nodes)
        df_results.loc[nodes, "Sobol"] = mc_quasi_sobol(nodes)
        df_results.loc[nodes, "Gauss"] = gc_legendre(nodes)

    df_results.plot()
