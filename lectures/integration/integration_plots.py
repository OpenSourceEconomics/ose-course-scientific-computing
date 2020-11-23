import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
from integration_algorithms import monte_carlo_naive_unidimensional
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


def plot_naive_monte_carlo(num_points):
    x, y = np.hsplit(np.random.uniform(size=num_points * 2).reshape(num_points, 2), 2)
    plt.scatter(x, y)


def plot_quasi_monte_carlo(num_points):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1))
    samples = distribution.sample(num_points, rule="halton")
    x, y = np.hsplit(samples.T, 2)
    ax1.scatter(x, y)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    ax1.set_title("Halton")

    samples = distribution.sample(num_points, rule="sobol")
    x, y = np.hsplit(samples.T, 2)
    ax2.scatter(x, y)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 1)

    ax2.set_title("Sobol")


def plot_naive_monte_carlo_error(max_points):

    grid = np.linspace(5, max_points, dtype=int)
    yvals = list()
    for nodes in grid:
        rslt = monte_carlo_naive_unidimensional(problem_smooth, a=-1, b=1, n=nodes, seed=123)
        yvals += [np.abs(rslt - 2.3504023872876028)]

    fig, ax = plt.subplots()
    ax.plot(grid, yvals)


def plot_naive_monte_carlo_randomness():

    grid = range(10)
    yvals = list()
    for seed in grid:
        rslt = monte_carlo_naive_unidimensional(problem_smooth, a=-1, b=1, n=50, seed=seed)
        yvals += [np.abs(rslt - 2.3504023872876028)]

    fig, ax = plt.subplots()
    ax.scatter(grid, yvals)
