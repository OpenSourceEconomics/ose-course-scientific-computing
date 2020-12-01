import matplotlib.pyplot as plt
import numpy as np
from approximation_auxiliary import get_chebyshev_nodes
from approximation_auxiliary import get_uniform_nodes
from approximation_problems import problem_reciprocal_exponential
from approximation_problems import problem_runge
from numpy.polynomial import Chebyshev as T


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
        c = np.polyfit(xnodes, problem_runge(xnodes), degree)
        yfit = np.polyval(c, xvals)
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
        plt.plot(x, yvals, lw=2, label="$T_%d$" % i)


def plot_reciprocal_exponential(a=-5, b=5):
    fig, ax = plt.subplots()

    yvals = np.linspace(a, b)
    ax.plot(yvals, problem_reciprocal_exponential(yvals))


def plot_approximation_nodes(num_nodes, strategy="uniform"):

    if strategy == "uniform":
        get_nodes = get_uniform_nodes
    elif strategy == "chebychev":
        get_nodes = get_chebyshev_nodes

    fig, ax = plt.subplots()
    for i, n in enumerate(num_nodes):
        ax.scatter(get_nodes(n), [i] * n, label=f"{n} nodes")
    ax.legend(ncol=5)
    ax.set_yticks([])
    ax.set_ylim(-1, 4)
