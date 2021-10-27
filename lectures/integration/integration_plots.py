"""Plotting functions for integration lecture."""
from functools import partial

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lectures.integration.integration_algorithms import monte_carlo_naive_one
from lectures.integration.integration_algorithms import quadrature_gauss_legendre_one
from lectures.integration.integration_algorithms import quadrature_newton_trapezoid_one
from lectures.integration.integration_problems import problem_kinked
from lectures.integration.integration_problems import problem_smooth


def plot_gauss_legendre_weights(deg):
    """Plot Gauss-Legendre weights."""
    xevals, weights = np.polynomial.legendre.leggauss(deg)

    fig, ax = plt.subplots()

    ax.bar(xevals, weights, width=0.02)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Node")
    ax.set_xlim([-1, 1])
    plt.show()


def plot_benchmarking_exercise():
    """Plot benchmarking exercise."""
    xvals = np.linspace(-1, 1, 10000)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(xvals, problem_smooth(xvals), label=r"$e^-x$")
    ax1.legend()

    ax2.plot(xvals, problem_kinked(xvals), label=r"$\sqrt{|x|}$")
    ax2.legend()


def plot_naive_monte_carlo(num_nodes):
    """Plot naive Monte Carlo example."""
    fig, ax = plt.subplots(figsize=(4, 4))
    x, y = np.hsplit(np.random.uniform(size=num_nodes * 2).reshape(num_nodes, 2), 2)
    ax.scatter(x, y)
    ax.get_yticklabels()[0].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)


def plot_quasi_monte_carlo(num_points):
    """Plot Quasi-Monte Carlo example."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1))
    samples = distribution.sample(num_points, rule="halton")
    x, y = np.hsplit(samples.T, 2)

    ax1.get_yticklabels()[0].set_visible(False)
    ax1.scatter(x, y)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    ax1.set_title("Halton")

    samples = distribution.sample(num_points, rule="sobol")
    x, y = np.hsplit(samples.T, 2)
    ax2.get_yticklabels()[0].set_visible(False)
    ax2.scatter(x, y)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 1)

    ax2.set_title("Sobol")


def plot_naive_monte_carlo_error(max_nodes):
    """Plot naive Monte Carlo error."""
    index = pd.Index(np.linspace(5, max_nodes, dtype=int), name="Nodes")
    df_results = pd.DataFrame(columns=["Trapezoid", "Gauss", "Naive", "Truth"], index=index)

    p_trapezoid = partial(quadrature_newton_trapezoid_one, problem_smooth, -1, 1)
    p_gauss = partial(quadrature_gauss_legendre_one, problem_smooth, -1, 1)
    p_naive = partial(monte_carlo_naive_one, problem_smooth, -1, 1)

    df_results.loc[:, "Truth"] = np.exp(1) - np.exp(-1)
    for nodes in df_results.index.get_level_values("Nodes"):
        df_results.loc[nodes, "Trapezoid"] = np.abs(p_trapezoid(nodes))
        df_results.loc[nodes, "Gauss"] = np.abs(p_gauss(nodes))
        df_results.loc[nodes, "Naive"] = np.abs(p_naive(nodes))

    fig, ax = plt.subplots()
    for column in df_results.columns:
        ax.plot(df_results.index.get_level_values("Nodes"), df_results[column], label=column)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Error")
    ax.legend()


def plot_naive_monte_carlo_randomness():
    """Plot naive Monte Carlo randomness."""
    grid = range(10)
    yvals = []
    for seed in grid:
        rslt = monte_carlo_naive_one(problem_smooth, a=-1, b=1, n=50, seed=seed)
        yvals += [np.abs(rslt - 2.3504023872876028)]

    fig, ax = plt.subplots()
    ax.scatter(grid, yvals)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Error")


def plot_starting_illustration():
    """
    Plot example of continuous real-valued function over bounded interval.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    x_range = np.array([0, 1])
    a_b = np.array([0.25, 0.75])
    n = 401

    z = np.linspace(*a_b, n)
    x = np.linspace(*x_range, n)

    def f(x):
        return 25 - np.cos(np.pi * x) * (2 * np.pi * x - np.pi + 0.5) ** 2

    fig, ax = plt.subplots(figsize=[10, 6])
    ax.fill_between(z, 0, f(z), alpha=0.35, color="gold")
    ax.hlines(0, *x_range, "k", linewidth=2)
    ax.vlines(a_b, 0, f(a_b), color="tab:red", linestyle="--", linewidth=2)
    ax.plot(x, f(x), linewidth=3)
    ax.set(xlim=x_range, xticks=a_b, ylim=[-5, f(x).max() + 2], yticks=[0])

    ax.set_yticklabels(["0"], size=15)
    ax.set_xticklabels(["$a$", "$b$"], size=15)

    ax.annotate(
        r"$f(x)$",
        [x_range[1] - 0.1, f(x_range[1]) - 5],
        fontsize=16,
        color="black",
        va="top",
    )
    ax.annotate(r"$A = \int_a^bf(x)dx$", [a_b.mean(), 10], fontsize=18, ha="center")


def get_trapezoid_quadrature_nodes_and_weights(n, a, b):
    """
    Compute univariate trapezoid rule quadrature nodes and weights.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    if n < 1:
        raise ValueError("n must be at least one")

    nodes = np.linspace(a, b, n)
    dx = nodes[1] - nodes[0]

    weights = dx * np.ones(n)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    return nodes, weights


def plot_trapezoid_rule_illustration():
    """
    Plot illustrations of trapezoid rule.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    xmin, xmax = -1, 1
    f = np.poly1d([2.0, -1.0, 0.5, 5.0])
    x = np.linspace(xmin, xmax, 1001)

    for n in 2, 4, 8:

        xi, wi = get_trapezoid_quadrature_nodes_and_weights(n + 1, xmin, xmax)

        fig, ax = plt.subplots(figsize=[10, 6])
        ax.fill_between(xi, f(xi), alpha=0.35, color="gold")
        ax.plot(x, f(x), linewidth=3, label=r"$f(x)$")
        ax.plot(xi, f(xi), color="Tab:red", linestyle="--", label=f"$\\tilde{{f}}_{n+1}(x)$")
        ax.vlines(xi, 0, f(xi), color="Tab:red", linestyle=":")
        ax.axhline(0, color="k", linewidth=2)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        xtl = [f"$x_{i}$" for i in range(n + 1)]
        xtl[0] += "=a"
        xtl[n] += "=b"
        ax.set(
            xlim=[xmin - 0.1, xmax + 0.1],
            xticks=xi,
            xticklabels=xtl,
            yticks=[0],
            yticklabels=["0"],
        )
        ax.legend(fontsize=16)


def get_simpsons_quadrature_nodes_and_weights(n, a, b):
    """
    Compute univariate Simpson quadrature nodes and weights.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    nodes = np.linspace(a, b, n)
    dx = nodes[1] - nodes[0]
    weights = np.tile([2.0, 4.0], int((n + 1) / 2))
    weights = weights[:n]
    weights[0] = weights[-1] = 1
    weights = (dx / 3.0) * weights

    return nodes, weights


def plot_simpsons_rule_illustration():
    """
    Plot illustrations of Simpson's rule.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    xmin, xmax = -1, 1
    f = np.poly1d([2.0, -1.0, 0.5, 5.0])
    x = np.linspace(xmin, xmax, 1001)

    def fitquad(xi):
        newcoef = np.polyfit(xi, f(xi), 2)
        return np.poly1d(newcoef)

    for n in [2, 4, 8]:
        xi, wi = get_simpsons_quadrature_nodes_and_weights(n + 1, xmin, xmax)

        fig, ax = plt.subplots(figsize=[10, 6])
        ax.plot(x, f(x), linewidth=3)

        for k in range(n // 2):
            xii = xi[(2 * k) : (2 * k + 3)]
            xiii = np.linspace(xii[0], xii[2], 125)
            p = fitquad(xii)
            ax.fill_between(xiii, p(xiii), alpha=0.35, color="gold")
            ax.plot(xiii, p(xiii), color="tab:red", linestyle="--")

        plt.vlines(xi, 0, f(xi), "k", linestyle=":")
        plt.hlines(0, xmin - 0.1, xmax + 0.1, "k", linewidth=2)
        plt.xlim(xmin - 0.1, xmax + 0.1)
        xtl = ["$x_{%d}$" % i for i in range(n + 1)]
        xtl[0] += "=a"
        xtl[n] += "=b"
        plt.xticks(xi, xtl, fontsize=14)
        plt.yticks([0], ["0"], fontsize=14)
        plt.legend([r"$f(x)$", f"$\\tilde{{f}}_{n+1}(x)$"], fontsize=16)
