import numpy as np
from optimization_algorithms import golden_search
from optimization_problems import golden_search_problem


def test_golden():
    a, b = 0, 3
    y = golden_search(golden_search_problem, a, b)
    np.testing.assert_almost_equal(y, 0.8082516731363114)
