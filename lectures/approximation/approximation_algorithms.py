import numpy as np


def get_order_five_interpolation(func):

    xvalues = np.linspace(-1, 1, 1000)
    xnodes = np.linspace(-1, 1, 5)
    poly = np.polyfit(xnodes, func(xnodes), 5)
    yfit = np.polyval(poly, xvalues)

    return yfit, xvalues
