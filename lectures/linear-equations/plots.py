import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_iterative_convergence(conv_gs, conv_gj):
    
    fig, ax = plt.subplots()

    ax.plot(conv_gs, label="Gauss-Seidel")
    ax.plot(conv_gj, label="Gauss-Jacobi")

    ax.legend()


def plot_ill_problem_2(cond, err, grid):
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(grid, cond, label="Condition")
    ax2.plot(grid, err, label="Error")

    ax1.legend()
    ax2.legend()
    


def plot_operation_count():
    df = pd.DataFrame(columns=["LU", "Inverse", "Dimension"])


    df["Dimension"] = range(10)
    df["LU"] = df["Dimension"] / 3 + df["Dimension"] ** 2
    df["Inverse"] = df["Dimension"] ** 3 + df["Dimension"] ** 2
    
    fig, ax = plt.subplots()

    ax.plot(df["Dimension"], df["LU"], label="LU")
    ax.plot(df["Dimension"], df["Inverse"], label="Inverse")

    ax.legend()