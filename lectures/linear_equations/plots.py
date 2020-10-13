"""Plotting functions for lecture on linear equations."""
import matplotlib.pyplot as plt
import numpy as np


def plot_iterative_convergence(conv_gs, conv_gj):
    """Plot iterative convergence."""
    fig, ax = plt.subplots()
    ax.plot(conv_gs, label="Gauss-Seidel")
    ax.plot(conv_gj, label="Gauss-Jacobi")
    ax.legend()


def plot_ill_problem_2(cond, err, grid):
    """Plot ill problem."""
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(grid, cond, label="Condition")
    ax2.plot(grid, err, label="Error")
    ax1.legend()
    ax2.legend()


def plot_operation_count():
    """Plot operation count."""
    dim = np.arange(10) + 1
    fig, ax = plt.subplots()
    ax.plot(dim, dim / 3 + dim ** 2, label="LU")
    ax.plot(dim, dim ** 3 + dim ** 2, label="Inverse")
    ax.legend()
