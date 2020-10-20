"""Plotting functions for lecture on linear equations."""
import matplotlib.pyplot as plt
import numpy as np


def plot_iterative_convergence(conv_gs, conv_gj):
    """Plot iterative convergence."""
    fig, ax = plt.subplots()
    ax.plot(conv_gs, label="Gauss-Seidel")
    ax.plot(conv_gj, label="Gauss-Jacobi")
    ax.legend()


def plot_operation_count():
    """Plot operation count."""
    dim = np.arange(10) + 1
    fig, ax = plt.subplots()
    ax.plot(dim, dim / 3 + dim ** 2, label="LU")
    ax.plot(dim, dim ** 3 + dim ** 2, label="Inverse")
    ax.set_xlabel("Dimension")
    ax.legend()
