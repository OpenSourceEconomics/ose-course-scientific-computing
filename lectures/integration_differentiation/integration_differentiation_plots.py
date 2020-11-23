import matplotlib.pyplot as plt
import numpy as np
from integration_differentiation_problems import problem_kinked
from integration_differentiation_problems import problem_smooth


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
