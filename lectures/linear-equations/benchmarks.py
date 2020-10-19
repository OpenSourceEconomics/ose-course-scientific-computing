import time

import numpy as np


def benchmarking_alterantives():
    def tic():
        return time.time()

    def toc(t):
        return time.time() - t

    print(
        "{:^5} {:^5}   {:^11} {:^11} \n{}".format(
            "m", "n", "np.solve(A,b)", "dot(inv(A), b)", "-" * 40
        )
    )

    for m in [1, 100]:
        for n in [50, 500]:
            A = np.random.rand(n, n)
            b = np.random.rand(n, 1)

            tt = tic()
            for j in range(m):
                np.linalg.solve(A, b)

            f1 = 100 * toc(tt)

            tt = tic()
            Ainv = np.linalg.inv(A)
            for j in range(m):
                np.dot(Ainv, b)

            f2 = 100 * toc(tt)
            print(" {:3}   {:3} {:11.2f} {:11.2f}".format(m, n, f1, f2))
