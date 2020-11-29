from functools import partial

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from integration_algorithms import monte_carlo_naive_one
from integration_algorithms import quadrature_gauss_legendre_one
from integration_algorithms import quadrature_newton_trapezoid_one
from integration_problems import problem_kinked
from integration_problems import problem_smooth


def plot_gauss_legendre_weights(deg):
    xevals, weights = np.polynomial.legendre.leggauss(deg)

    fig, ax = plt.subplots()

    ax.bar(xevals, weights, width=0.02)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Node")
    ax.set_xlim([-1, 1])
    plt.show()


def plot_benchmarking_exercise():
    xvals = np.linspace(-1, 1, 10000)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(xvals, problem_smooth(xvals), label=r"$e^-x$")
    ax1.legend()

    ax2.plot(xvals, problem_kinked(xvals), label=r"$\sqrt{|x|}$")
    ax2.legend()


def plot_naive_monte_carlo(num_nodes):
    fig, ax = plt.subplots(figsize=(4, 4))
    x, y = np.hsplit(np.random.uniform(size=num_nodes * 2).reshape(num_nodes, 2), 2)
    ax.scatter(x, y)
    ax.get_yticklabels()[0].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)


def plot_quasi_monte_carlo(num_points):
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

    grid = range(10)
    yvals = list()
    for seed in grid:
        rslt = monte_carlo_naive_one(problem_smooth, a=-1, b=1, n=50, seed=seed)
        yvals += [np.abs(rslt - 2.3504023872876028)]

    fig, ax = plt.subplots()
    ax.scatter(grid, yvals)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Error")
