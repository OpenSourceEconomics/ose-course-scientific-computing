"""Plotting functions for integration lecture."""
from functools import partial

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import poly1d,polyfit, linspace, array
from integration_algorithms import monte_carlo_naive_one
from integration_algorithms import quadrature_gauss_legendre_one
from integration_algorithms import quadrature_newton_trapezoid_one
from integration_problems import problem_kinked
from integration_problems import problem_smooth


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
 
#Integration plots from the lecture
#bounded interval function

from numpy import cos, pi, linspace, array
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 25 - cos(pi*x)*(2*pi*x - pi + 0.5)**2

x_range = array([0, 1])
a_b = array([0.25, 0.75])
n = 401

z = linspace(*a_b, n)
x = linspace(*x_range, n)

fig, ax = plt.subplots(figsize=[8,4])
ax.fill_between(z, 0, f(z), alpha=0.35, color='lightgreen')
ax.hlines(0, *x_range, 'k', linewidth=2)
ax.vlines(a_b, 0, f(a_b), color='tab:red',linestyle='--',linewidth=2)
ax.plot(x,f(x), linewidth=3)
ax.set(xlim=x_range, xticks=a_b,
       ylim=[-5, f(x).max()+2], yticks=[0])
       
ax.set_yticklabels(['0'], size=15)
ax.set_xticklabels(['$a$', '$b$'], size=15)
ax.set_facecolor("whitesmoke")

ax.annotate(r'$f(x)$', [x_range[1] - 0.1, f(x_range[1])-5], fontsize=13, color='black', va='top')
ax.annotate(r'$A = \int_a^bf(x)dx$', [a_b.mean(), 10] ,fontsize=18, ha='center');

# trapezoid rule

from numpy import poly1d, linspace
import matplotlib.pyplot as plt
!pip install compecon
from compecon import qnwtrap, demo

n = 1001
xmin, xmax = -1, 1
xwid = xmax-xmin
x = linspace(xmin, xmax, n)

f = poly1d([2.0, -1.0, 0.5, 5.0])


def plot_trap(n):
    xi, wi = qnwtrap(n+1, xmin, xmax)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(xi, f(xi), alpha=0.35, color='lightgreen')
    ax.plot(x, f(x), linewidth=3, label=r'$f(x)$')
    ax.plot(xi, f(xi), color='Tab:red', linestyle='--', label=f'$\\tilde{{f}}_{n+1}(x)$')
    ax.vlines(xi, 0, f(xi),color='Tab:red', linestyle=':')
    ax.axhline(0,color='k',linewidth=2)
    ax.set_facecolor("whitesmoke")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)



    xtl = [f'$x_{i}$' for i in range(n+1)]
    xtl[0] += '=a'
    xtl[n] += '=b'
    ax.set(
        xlim=[xmin-0.1, xmax+0.1],
        xticks=xi, 
        xticklabels=xtl,
        yticks=[0],
        yticklabels=['0'])
    ax.legend()
    return fig

figs = [plot_trap(n) for n in [2, 4, 8]]


#Simpsons rule

from numpy import poly1d,polyfit, linspace, array
import matplotlib.pyplot as plt
!pip install compecon
from compecon import qnwsimp, demo


n = 1001
xmin, xmax = -1, 1
xwid = xmax-xmin
x = linspace(xmin, xmax, n)

f = poly1d([2.0, -1.0, 0.5, 5.0])

def fitquad(xi):
    newcoef = polyfit(xi, f(xi), 2)
    return poly1d(newcoef)

def plot_simp(n):
    xi, wi = qnwsimp(n+1, xmin, xmax)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, f(x), linewidth=3)
    ax.set_facecolor("whitesmoke")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    

    for k in range(n//2):
        xii = xi[(2*k):(2*k+3)]
        xiii = linspace(xii[0], xii[2], 125)
        p = fitquad(xii)
        ax.fill_between(xiii, p(xiii), alpha=0.35, color='lightgreen')    
        ax.plot(xiii, p(xiii),color='Tab:red', linestyle='--')
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_facecolor("whitesmoke")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    plt.vlines(xi, 0, f(xi),'k', linestyle=':')
    plt.hlines(0,xmin-0.1, xmax+0.1,'k',linewidth=2)
    plt.xlim(xmin-0.1, xmax+0.1)
    xtl = ['$x_{%d}$' % i for i in range(n+1)]
    xtl[0] += '=a'
    xtl[n] += '=b'
    plt.xticks(xi, xtl)
    plt.yticks([0],['0'])
    plt.legend([r'$f(x)$', f'$\\tilde{{f}}_{n+1}(x)$'])
    return fig

def plot_simp(n):
    xi, wi = qnwsimp(n+1, xmin, xmax)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, f(x), linewidth=3)
    ax.set_facecolor("whitesmoke")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    for k in range(n//2):
        xii = xi[(2*k):(2*k+3)]
        xiii = linspace(xii[0], xii[2], 125)
        p = fitquad(xii)
        ax.fill_between(xiii, p(xiii), alpha=0.35, color='lightgreen')    
        ax.plot(xiii, p(xiii), color='Tab:red', linestyle='--')
    
    ax.vlines(xi, 0, f(xi), color='Tab:red', linestyle=':')
    ax.axhline(0,color='k',linewidth=2)
    
    xtl = [f'$x_{i}$' for i in range(n+1)]
    xtl[0] += '=a'
    xtl[n] += '=b'
    
    ax.set(xlim=[xmin-0.1, xmax+0.1], xticks=xi, xticklabels=xtl,
           yticks=[0], yticklabels=['0'])
    
    plt.legend([r'$f(x)$', f'$\\tilde{{f}}_{n+1}(x)$'])
    return fig

figs = [plot_simp(n) for n in [2, 4, 8]]



def integration_plot(): 
    f = poly1d([2.0, -1.0, 0.5, 5.0])
    a = -1; b = 1; N = 2

    # x and y values for the trapezoid rule
    x = np.linspace(a,b,N+1)
    y = f(x)

    # X and Y values for plotting y=f(x)
    X = np.linspace(a,b,100)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.fill_between(x, f(x), alpha=0.35, color='lightblue')
    ax.plot(x, f(x), linewidth=3, label=r'$f(x)$', color='lightblue')
    ax.plot(x, f(x), color='tab:red', linestyle='--')
    ax.set_facecolor("whitesmoke")
    plt.xticks(fontsize=14)
    plt.yticks(y, " ")
    ax.axes.yaxis.set_visible(False)
    plt.plot(X,Y)

    for i in range(N):
        xs = [x[i],x[i],x[i+1],x[i+1]]
        ys = [0,f(x[i]),f(x[i+1]),0]
        plt.fill(xs,ys,'lightblue',edgecolor='r', linestyle='--', alpha=0.2)
        plt.xticks(np.arange(-1, 2, 1.0))
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        my_xticks = ['X0=a'.translate(SUB),'X1'.translate(SUB),'X2=b'.translate(SUB)]
        plt.xticks(x, my_xticks)
        plt.legend(['f(x)', r'$\tilde{f}3(x)$'.translate(SUB)], fontsize=12)

    plt.title('Trapezoid Rule, N = {}'.format(N))
integration_plot()




def integration_plot_two(): 
    f = poly1d([2.0, -1.0, 0.5, 5.0])
    a = -1; b = 1; N = 4

    # x and y values for the trapezoid rule
    x = np.linspace(a,b,N+1)
    y = f(x)

    # X and Y values for plotting y=f(x)
    X = np.linspace(a,b,100)
    Y = f(X)

    fig, ax = plt.subplots(figsize=[8,4])
    ax.fill_between(x, f(x), alpha=0.35, color='lightblue')
    ax.plot(x, f(x), linewidth=3, label=r'$f(x)$', color='lightblue')
    ax.plot(x, f(x), color='tab:red', linestyle='--')
    ax.set_facecolor("whitesmoke")
    plt.xticks(fontsize=14)
    plt.yticks(y, " ")
    ax.axes.yaxis.set_visible(False)
    plt.plot(X,Y)

    for i in range(N):
        xs = [x[i],x[i],x[i+1],x[i+1]]
        ys = [0,f(x[i]),f(x[i+1]),0]
        plt.fill(xs,ys,'lightblue',edgecolor='r', linestyle='--', alpha=0.2)
        plt.xticks(np.arange(-1, 2, 1.0))
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        my_xticks = ['X0=a'.translate(SUB),'X1'.translate(SUB), 'X2'.translate(SUB), 'X3'.translate(SUB), 'X4=b'.translate(SUB)]
        plt.xticks(x, my_xticks)
        plt.legend(['f(x)', r'$\tilde{f}3(x)$'.translate(SUB)], fontsize=12)

    plt.title('Trapezoid Rule, N = {}'.format(N))
integration_plot_two()





def integration_plot_three():
    f = poly1d([2.0, -1.0, 0.5, 5.0])
    a = -1; b = 1; N = 8

    # x and y values for the trapezoid rule
    x = np.linspace(a,b,N+1)
    y = f(x)

    # X and Y values for plotting y=f(x)
    X = np.linspace(a,b,100)
    Y = f(X)
    fig, ax = plt.subplots(figsize=[8,4])
    ax.fill_between(x, f(x), alpha=0.35, color='lightblue')
    ax.plot(x, f(x), linewidth=3, label=r'$f(x)$', color='lightblue')
    ax.plot(x, f(x), color='tab:red', linestyle='--')
    ax.set_facecolor("whitesmoke")
    plt.xticks(fontsize=14)
    plt.yticks(y, " ")
    ax.axes.yaxis.set_visible(False)
    plt.plot(X,Y)


    for i in range(N):
        xs = [x[i],x[i],x[i+1],x[i+1]]
        ys = [0,f(x[i]),f(x[i+1]),0]
        plt.fill(xs,ys,'lightblue',edgecolor='red', linestyle='--', alpha=0.2)
        plt.xticks(np.arange(-1, 2, 1.0))
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        my_xticks = ['X0=a'.translate(SUB),'X1'.translate(SUB), 'X2'.translate(SUB), 'X3'.translate(SUB), 'X4'.translate(SUB), 
                     'X5'.translate(SUB), 'X6'.translate(SUB), 'X7'.translate(SUB), 'X8=b'.translate(SUB)]
        plt.xticks(x, my_xticks)
        plt.legend(['f(x)', r'$\tilde{f}9(x)$'.translate(SUB)], fontsize=12)


    plt.title('Trapezoid Rule, N = {}'.format(N))
integration_plot_three()
