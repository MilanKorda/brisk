"""BRISK from CVXPY: the max-cut relaxation of a small random graph.

    python examples/quickstart.py
"""
import cvxpy as cp
import numpy as np

import brisk

rng = np.random.default_rng(0)
n = 20
W = np.triu(rng.random((n, n)) < 0.3, 1).astype(float)
W = W + W.T                                     # adjacency matrix
L = np.diag(W.sum(1)) - W                       # Laplacian

X = cp.Variable((n, n), symmetric=True)
prob = cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [X >> 0, cp.diag(X) == 1])
prob.solve(solver=brisk.BRISK(), log=True, threads=2)   # log=True: BRISK's log (verbose=True: CVXPY's too); any ./brisk option as a keyword

print("status       ", prob.status)
print("upper bound  ", f"{prob.value:.10e}")
print("BRISK        ", prob.solver_stats.extra_stats["status"],
      "in", prob.solver_stats.num_iters, "iterations,", f"{prob.solver_stats.solve_time:.2f} s")
print("DIMACS error ", f"{max(abs(e) for e in prob.solver_stats.extra_stats['dimacs']):.1e}")
print("dual of diag(X) == 1 (first 5):", np.round(prob.constraints[1].dual_value[:5], 6).tolist())

# a cut from the solution (Goemans-Williamson rounding)
w, V = np.linalg.eigh(X.value)
F = V * np.sqrt(np.maximum(w, 0))
x = np.sign(F @ rng.standard_normal(n))
print("a cut        ", (x @ L @ x) / 4)
