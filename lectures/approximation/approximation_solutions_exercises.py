from itertools import product

import numpy as np
import pandas as pd
from approximation_auxiliary import compute_interpolation_error
from approximation_auxiliary import get_uniform_nodes
from approximation_problems import runge


def test_exercise_1():

    index = product([10, 20, 30, 40, 50], np.linspace(-5, 5, 1000))

    index = pd.MultiIndex.from_tuples(index, names=("Degree", "Point"))
    df = pd.DataFrame(columns=["Value", "Approximation"], index=index)

    df["Value"] = runge(df.index.get_level_values("Point"))

    for degree in [10, 20, 30, 40, 50]:

        xnodes = get_uniform_nodes(degree, -5, 5)
        poly = np.polyfit(xnodes, runge(xnodes), degree)

        xvalues = df.index.get_level_values("Point").unique()
        yvalues = np.polyval(poly, xvalues)

        df.loc[(degree, slice(None)), "Approximation"] = yvalues

        df["Error"] = df["Value"] - df["Approximation"]

    df.groupby("Degree").apply(compute_interpolation_error).plot()
