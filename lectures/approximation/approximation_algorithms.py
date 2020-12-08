from numpy.polynomial import Chebyshev as C
from numpy.polynomial import Polynomial as P
from scipy.interpolate import interp1d
import numpy as np

from approximation_auxiliary import get_chebyshev_nodes
from approximation_auxiliary import get_uniform_nodes


def get_interpolator_runge_baseline(func):
    xnodes = np.linspace(-1, 1, 5)
    poly = P.fit(xnodes, func(xnodes), 5)
    return poly


def get_interpolator_monomial_uniform(func, degree, a=-1, b=1):
    xnodes = np.linspace(a, b, degree)
    poly = P.fit(xnodes, func(xnodes), degree)
    return poly


def get_interpolator_monomial_flexible_nodes(func, degree, nodes="uniform", a=-1, b=1):

    if nodes == "uniform":
        get_nodes = get_uniform_nodes
    elif nodes == "chebychev":
        get_nodes = get_chebyshev_nodes

    xnodes = get_nodes(degree, a, b)
    poly = P.fit(xnodes, func(xnodes), degree)

    return poly


def get_interpolator_flexible_basis_flexible_nodes(
    func, degree, basis="monomial", nodes="uniform", a=-1, b=1
):

    if nodes == "uniform":
        get_nodes = get_uniform_nodes
    elif nodes == "chebychev":
        get_nodes = get_chebyshev_nodes

    if basis == "monomial":
        fit = P.fit
    elif basis == "chebychev":
        fit = C.fit

    xnodes = get_nodes(degree, a, b)
    poly = fit(xnodes, func(xnodes), degree)

    return poly


def get_interpolator(name, degree, func):
    args = (degree, -1, 1)
    if name in ["linear", "cubic"]:
        xnodes = get_uniform_nodes(*args)
        interp = interp1d(xnodes, func(xnodes), name)
    elif name in ["chebychev"]:
        xnodes = get_chebyshev_nodes(*args)
        interp = P.fit(xnodes, func(xnodes), degree)

    return interp
