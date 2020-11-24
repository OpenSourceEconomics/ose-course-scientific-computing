from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from integration_algorithms import monte_carlo_naive_two_unit_cube as mc_naive
from integration_algorithms import monte_carlo_quasi_two_unit_cube as mc_quasi
from integration_algorithms import quadrature_gauss_legendre_one
from integration_algorithms import quadrature_gauss_legendre_two as gc_legendre_two
from integration_algorithms import quadrature_newton_simpson_one
from integration_algorithms import quadrature_newton_trapezoid_one
from integration_problems import problem_genz_discontinuous
from integration_problems import problem_kinked
from integration_problems import problem_smooth


def test_exercise_1():
    index = product(["Smooth", "Kinked"], [5, 11, 21, 31])
    index = pd.MultiIndex.from_tuples(index, names=("Function", "Nodes"))

    df_errors = pd.DataFrame(columns=["Trapezoid", "Simpson", "Gauss", "Truth"], index=index)

    df_errors.loc[("Smooth", slice(None)), "Truth"] = 2.3504023872876028
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
    index = pd.Index([100, 1000, 10000], name="Nodes")

    df_results = pd.DataFrame(columns=["Naive", "Sobol", "Halton", "Gauss", "Truth"], index=index)
    df_results["Truth"] = 5.001926847246786

    mc_quasi_halton = partial(mc_quasi, problem_genz_discontinuous, rule="halton")
    mc_quasi_sobol = partial(mc_quasi, problem_genz_discontinuous, rule="sobol")
    gc_legendre = partial(gc_legendre_two, problem_genz_discontinuous, 0, 1)

    for nodes in df_results.index.get_level_values("Nodes"):
        df_results.loc[nodes, "Naive"] = mc_naive(problem_genz_discontinuous, nodes)
        df_results.loc[nodes, "Halton"] = mc_quasi_halton(nodes)
        df_results.loc[nodes, "Sobol"] = mc_quasi_sobol(nodes)
        df_results.loc[nodes, "Gauss"] = gc_legendre(nodes)
