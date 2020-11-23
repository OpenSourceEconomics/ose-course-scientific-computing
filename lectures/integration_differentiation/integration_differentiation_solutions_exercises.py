from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from integration_differentiation_algorithms import quadrature_gauss_legendre
from integration_differentiation_algorithms import quadrature_simpson
from integration_differentiation_algorithms import quadrature_trapezoid
from integration_differentiation_problems import problem_kinked
from integration_differentiation_problems import problem_smooth


def test_exercise_1():

    index = pd.MultiIndex.from_tuples(
        product(["Smooth", "Kinked"], [5, 11, 21, 31]), names=("Function", "Nodes")
    )

    df_errors = pd.DataFrame(columns=["Trapezoid", "Simpson", "Gauss", "Truth"], index=index)

    df_errors.loc[("Smooth", slice(None)), "Truth"] = 2.3504023872876028
    df_errors.loc[("Kinked", slice(None)), "Truth"] = 4 / 3

    for label, test_function in [
        ("Smooth", problem_smooth),
        ("Kinked", problem_kinked),
    ]:
        p_trapezoid = partial(quadrature_trapezoid, test_function, -1, 1)
        p_simpson = partial(quadrature_simpson, test_function, -1, 1)
        p_gauss = partial(quadrature_gauss_legendre, test_function, -1, 1)
        for nodes in df_errors.index.get_level_values("Nodes"):
            index = (label, nodes)
            df_errors.loc[index, "Trapezoid"] = np.abs(p_trapezoid(nodes))
            df_errors.loc[index, "Simpson"] = np.abs(p_simpson(nodes))
            df_errors.loc[index, "Gauss"] = np.abs(p_gauss(nodes))
