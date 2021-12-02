"""Plotting functions for approximation lab."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from approximation_algorithms import get_interpolator_monomial_flexible_nodes
from approximation_auxiliary import get_chebyshev_nodes
from approximation_auxiliary import get_uniform_nodes
from approximation_auxiliary import spline_basis
from approximation_problems import problem_reciprocal_exponential
from approximation_problems import problem_two_dimensions
from scipy.interpolate import interp1d
from temfpy.interpolation import runge


def plot_runge():
    """Plot runge function."""
    fig, ax = plt.subplots()
    xvals = np.linspace(-1, 1, 1000)
    yvals = runge(xvals)
    ax.plot(xvals, yvals, label="Function")


def plot_runge_multiple():
    """Plot multiple runge functions."""
    a, b = -1, 1
    xvals = np.linspace(a, b, 1000)
    yvals = runge(xvals)

    fig, ax = plt.subplots()

    ax.plot(xvals, yvals, label="Function")

    for degree in [5, 9]:
        xnodes = np.linspace(a, b, degree)
        poly = np.polynomial.Polynomial.fit(xnodes, runge(xnodes), degree)
        yfit = poly(xvals)
        ax.plot(xvals, yfit, label=" 9th-order")

    ax.legend()


def plot_basis_functions(name="monomial"):
    """Plot basis functions."""
    x = np.linspace(0, 1, 100)

    for i in range(6):
        fig, ax = plt.subplots()

        if name == "chebychev":
            yvals = np.polynomial.Chebyshev.basis(i)(x)
        elif name == "monomial":
            yvals = x ** i

        elif name == "linear":
            a, b = 0, 1
            yvals = np.tile(np.nan, 100)
            h = (b - a) / 5

            for j, element in enumerate(x):
                yvals[j] = spline_basis(element, i + 1, h, 0)

        ax.plot(x, yvals, lw=2, label="$T_%d$" % i)


def plot_reciprocal_exponential(a=-5, b=5):
    """Plot reciprocal exponential function."""
    fig, ax = plt.subplots()

    yvals = np.linspace(a, b)
    ax.plot(yvals, problem_reciprocal_exponential(yvals))


def plot_approximation_nodes(num_nodes, nodes="uniform"):
    """Plot approximation nodes."""
    if nodes == "uniform":
        get_nodes = get_uniform_nodes
    elif nodes == "chebychev":
        get_nodes = get_chebyshev_nodes

    fig, ax = plt.subplots()
    for i, n in enumerate(num_nodes):
        ax.scatter(get_nodes(n), [i] * n, label=f"{n} nodes")
    ax.legend(ncol=5)
    ax.set_yticks([])
    ax.set_ylim(-1, 4)


def plot_runge_different_nodes():
    """Plot runge function at different nodes."""
    get_interpolator = partial(get_interpolator_monomial_flexible_nodes, runge, 11)
    interp_unif = get_interpolator(nodes="uniform")
    interp_cheby = get_interpolator(nodes="chebychev")

    xvalues = np.linspace(-1, 1, 10000)

    fig, ax = plt.subplots()
    ax.plot(xvalues, runge(xvalues), label="True")
    ax.plot(xvalues, interp_unif(xvalues), label="Uniform")
    ax.plot(xvalues, interp_cheby(xvalues), label="Chebychev")
    ax.legend()
    ax.set_title("Runge function")

    fig, ax = plt.subplots()

    ax.plot(
        xvalues,
        interp_unif(xvalues) - runge(xvalues),
        label="Uniform",
    )
    ax.plot(
        xvalues,
        interp_cheby(xvalues) - runge(xvalues),
        label="Chebychev",
    )
    ax.legend()
    ax.set_title("Approximation error")


def plot_two_dimensional_grid(nodes):
    """Plot two-dimensional grid."""
    if nodes == "chebychev":
        x = get_chebyshev_nodes(10)
        y = get_chebyshev_nodes(10)
    elif nodes == "uniform":
        x = get_uniform_nodes(10)
        y = get_uniform_nodes(10)

    X, Y = np.meshgrid(x, y)  # grid of point

    fig, ax = plt.subplots()
    for i in range(len(x)):
        ax.plot(X[i, :], Y[i, :], marker=".", color="k", linestyle="none")


def plot_two_dimensional_problem():
    """Plot two-dimensional problem."""
    x_fit = get_uniform_nodes(50)
    y_fit = get_uniform_nodes(50)
    X_fit, Y_fit = np.meshgrid(x_fit, y_fit)
    Z_fit = problem_two_dimensions(X_fit, Y_fit)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X_fit, Y_fit, Z_fit)
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_zlabel("f(x, y)")


def plot_runge_function_cubic():
    """Plot cubic runge function."""
    xvalues = get_uniform_nodes(10000, -1, 1)

    for degree in [5, 10, 15]:
        x_fit = get_uniform_nodes(degree, -1, 1)

        interp = interp1d(x_fit, runge(x_fit), kind="cubic")
        yfit = interp(xvalues)

        fig, ax = plt.subplots()
        ax.plot(xvalues, runge(xvalues), label="True")
        ax.plot(xvalues, yfit, label="Approximation")
        ax.legend()
        ax.set_title(f"Degree {degree}")
