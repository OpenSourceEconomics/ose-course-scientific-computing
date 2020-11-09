import numpy as np
import scipy as sp


def golden_search_problem(x):
    return x * np.cos(x ** 2)


def get_nelder_mead_problem(x):
    # TODO: in progress of integrated in temfpy
    return sp.optimize.rosen(x)
