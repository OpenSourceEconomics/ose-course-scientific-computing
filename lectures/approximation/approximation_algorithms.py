"""This module contains the algorithms for the approximation lab."""
import numpy as np
from labs.approximation.approximation_auxiliary import get_chebyshev_nodes
from labs.approximation.approximation_auxiliary import get_uniform_nodes
from scipy.interpolate import interp1d


def get_interpolator_runge_baseline(func):
    """Return interpolator runge function (baseline)."""
    xnodes = np.linspace(-1, 1, 5)
    poly = np.polynomial.Polynomial.fit(xnodes, func(xnodes), 5)
    return poly


def get_interpolator_monomial_uniform(func, degree, a=-1, b=1):
    """Return interpolator monomial function (uniform)."""
    xnodes = np.linspace(a, b, degree)
    poly = np.polynomial.Polynomial.fit(xnodes, func(xnodes), degree)
    return poly


def get_interpolator_monomial_flexible_nodes(func, degree, nodes="uniform", a=-1, b=1):
    """Return monomial function (flexible nodes)."""
    if nodes == "uniform":
        get_nodes = get_uniform_nodes
    elif nodes == "chebychev":
        get_nodes = get_chebyshev_nodes

    xnodes = get_nodes(degree, a, b)
    poly = np.polynomial.Polynomial.fit(xnodes, func(xnodes), degree)

    return poly


def get_interpolator_flexible_basis_flexible_nodes(
    func, degree, basis="monomial", nodes="uniform", a=-1, b=1
):
    """Return interpolator (flexible basis, flexible nodes)."""
    if nodes == "uniform":
        get_nodes = get_uniform_nodes
    elif nodes == "chebychev":
        get_nodes = get_chebyshev_nodes

    if basis == "monomial":
        fit = np.polynomial.Polynomial.fit
    elif basis == "chebychev":
        fit = np.polynomial.Chebyshev.fit

    xnodes = get_nodes(degree, a, b)
    poly = fit(xnodes, func(xnodes), degree)

    return poly


def get_interpolator(name, degree, func):
    """Return interpolator."""
    args = (degree, -1, 1)
    if name in ["linear", "cubic"]:
        xnodes = get_uniform_nodes(*args)
        interp = interp1d(xnodes, func(xnodes), name)
    elif name in ["chebychev"]:
        xnodes = get_chebyshev_nodes(*args)
        interp = np.polynomial.Polynomial.fit(xnodes, func(xnodes), degree)

    return interp
