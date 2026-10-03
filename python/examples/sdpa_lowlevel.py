"""BRISK without CVXPY: an SDPA file, and the same problem given as numbers.

SDPA primal:  min c'x  s.t.  x1 F1 + x2 F2 + x3 F3 - F0 >= 0  (the SDPA manual's example 1)
"""
import os
import tempfile

import numpy as np

import brisk

# the entries of the SDPA file: matrix (0 = F0), block, i, j, value (1-based, upper triangle)
data = dict(blocksizes=[2], c=[48.0, -8.0, 20.0],
            mat=[0, 0, 1, 1, 2, 3, 3], blk=[1] * 7,
            i=[1, 2, 1, 1, 2, 1, 2], j=[1, 2, 1, 2, 2, 2, 2],
            v=[-11.0, 23.0, 10.0, 4.0, -8.0, -8.0, -2.0])

r = brisk.solve_sdpa(**data, options={"acc": "high"}, verbose=False)
print(r)                                  # status, objectives, DIMACS errors, time
print("x =", r.x, " c'x =", np.dot(data["c"], r.x))
print("dual matrix X =\n", r.X[0])        # the SDPA dual variable; r.Z[0] = F(x)

f = os.path.join(tempfile.mkdtemp(), "example1.dat-s")
brisk.write_sdpa(f, **data)               # the same problem as a file
r2 = brisk.solve_file(f, verbose=True)    # as ./brisk example1.dat-s
