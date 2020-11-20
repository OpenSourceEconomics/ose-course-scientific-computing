"""Plots for nonlinear equations lecture."""
import matplotlib.pyplot as plt
import numpy as np


def plot_bisection_test_function(f):
    """Plot bisect example."""
    fig, ax = plt.subplots()
    grid = np.linspace(1, 2)

    values = []
    for value in grid:
        values.append(f(value))
    ax.plot(grid, values)

    ax.axes.axhline(0, color="grey")
    ax.set_ylabel("f(x)")
    ax.set_xlabel("x")


def plot_fixpoint_example(f):
    """Plot fixpoint example."""
    fig, ax = plt.subplots()
    grid = np.linspace(0, 2)
    vf = np.vectorize(f)
    ax.plot(grid, vf(grid))
    ax.axline([0, 0], [1, 1])


def plot_newton_pathological_example(f):
    """Plot fixpoint example."""
    fig, ax = plt.subplots()
    grid = np.linspace(-2, 2)
    values = []
    for value in grid:
        values.append(f(np.array([value]))[0])
    ax.plot(grid, values)
