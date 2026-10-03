"""A sum-of-squares lower bound: the global minimum of p(x) = x^4 - 3 x^2 + x.

    max t  s.t.  p(x) - t = z(x)' Q z(x),  Q >= 0,  z(x) = (1, x, x^2)

For univariate polynomials the bound is exact.
"""
import cvxpy as cp
import numpy as np

import brisk

p = {0: 0.0, 1: 1.0, 2: -3.0, 3: 0.0, 4: 1.0}   # coefficients of x^k
Q = cp.Variable((3, 3), symmetric=True)
t = cp.Variable()
cons = [Q >> 0]
for k in range(5):                              # match the coefficient of x^k
    cons.append(sum(Q[i, k - i] for i in range(3) if 0 <= k - i < 3) == p[k] - (t if k == 0 else 0))
prob = cp.Problem(cp.Maximize(t), cons)
prob.solve(solver=brisk.BRISK())

xs = np.linspace(-3, 3, 600001)
print("SOS bound      ", prob.value)
print("grid minimum   ", min(np.polyval([1, 0, -3, 1, 0], xs)))
print("form used      ", prob.solver_stats.extra_stats["form"])   # kernel: Q is BRISK's X
