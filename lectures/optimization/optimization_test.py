import numpy as np

from optimization_problems import get_golden_problem
from optimization_algorithms import golden_search


def test_golden():
    a, b = 0, 3
    y = golden_search(get_golden_problem, a, b)
    np.testing.assert_almost_equal(y, 0.8082516731363114)
