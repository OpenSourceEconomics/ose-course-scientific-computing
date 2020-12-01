from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from approximation_algorithms import get_interpolator_monomial_flexible_nodes
from approximation_auxiliary import get_chebyshev_nodes
from approximation_auxiliary import get_uniform_nodes
from approximation_problems import problem_reciprocal_exponential
from approximation_problems import problem_runge
from numpy.polynomial import Chebyshev as T
from numpy.polynomial import Polynomial as P


def plot_problem_runge():
    fig, ax = plt.subplots()
    xvals = np.linspace(-1, 1, 1000)
    yvals = problem_runge(xvals)
    ax.plot(xvals, yvals, label="Function")


def plot_runge_multiple():
    a, b = -1, 1
    xvals = np.linspace(a, b, 1000)
    yvals = problem_runge(xvals)

    fig, ax = plt.subplots()

    ax.plot(xvals, yvals, label="Function")

    for degree in [5, 9]:
        xnodes = np.linspace(a, b, degree)
        poly = P.fit(xnodes, problem_runge(xnodes), degree)
        yfit = poly(xvals)
        ax.plot(xvals, yfit, label=" 9th-order")

    ax.legend()


def plot_basis_functions(name="monomial"):

    x = np.linspace(-1, 1, 100)

    for i in range(6):
        fig, ax = plt.subplots()

        if name == "chebychev":
            yvals = T.basis(i)(x)
        elif name == "monomial":
            yvals = x ** i
        ax.plot(x, yvals, lw=2, label="$T_%d$" % i)


def plot_reciprocal_exponential(a=-5, b=5):
    fig, ax = plt.subplots()

    yvals = np.linspace(a, b)
    ax.plot(yvals, problem_reciprocal_exponential(yvals))


def plot_approximation_nodes(num_nodes, nodes="uniform"):

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
    get_interpolator = partial(get_interpolator_monomial_flexible_nodes, problem_runge, 11)
    interp_unif = get_interpolator(nodes="uniform")
    interp_cheby = get_interpolator(nodes="chebychev")

    xvalues = np.linspace(-1, 1, 10000)

    fig, ax = plt.subplots()
    ax.plot(xvalues, problem_runge(xvalues), label="True")
    ax.plot(xvalues, interp_unif(xvalues), label="Uniform")
    ax.plot(xvalues, interp_cheby(xvalues), label="Chebychev")
    ax.legend()
    ax.set_title("Runge function")

    fig, ax = plt.subplots()

    ax.plot(
        xvalues, interp_unif(xvalues) - problem_runge(xvalues), label="Uniform",
    )
    ax.plot(
        xvalues, interp_cheby(xvalues) - problem_runge(xvalues), label="Chebychev",
    )
    ax.legend()
    ax.set_title("Approximation error")
