#!/usr/bin/env python3
"""Generator of the reproducibility test's SDPs (tools/repro_suite.sh; ISSUES 708): SeDuMi .mat problems shaped like a
PolyOpt cut SDP (a user's report of 27 October 2026), 20 blocks (one of order 60, eight of order 2, nine of order 9,
one of order 17) and 504 LP variables, m = 396 constraints of a few entries each, with a degenerate optimum (the dual
slack of low rank in the large block). Random data from a seed; two seeds give two problems of the same dimensions
with other constraint lists.  usage: gen_repro_cutlike.py SEED OUT.mat"""
import sys
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

seed, out = int(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(seed)
nl, bs, m = 504, [60] + [2] * 8 + [9] * 9 + [17], 396
nsd = sum(n * n for n in bs)
N = nl + nsd
off = np.cumsum([nl] + [n * n for n in bs])[:-1]          # first column of each block in vec form
rows, cols, vals = [], [], []
# each constraint belongs to one block and to that block's segment of the LP variables (the Schur complement is then
# nearly block diagonal and factors sparse, as the cut SDP's does); the block of order 60 has 136 constraints, the one
# of order 17 has 60, the others share the rest; 6 constraints couple several blocks
nb = len(bs)
grp = np.concatenate([np.zeros(136, int), np.full(60, nb - 1)] + [np.full(1, 1 + i % (nb - 2)) for i in range(m - 196 - 6)]
                     + [np.full(6, -1)])
seg = np.array_split(np.arange(nl), nb)
def sdp_entry(i, k):
    n = bs[k]
    a, b = rng.integers(0, n, size=2)
    v = rng.uniform(-1.0, 1.0)
    rows.append(i); cols.append(off[k] + a + n * b); vals.append(v)
    if a != b: rows.append(i); cols.append(off[k] + b + n * a); vals.append(v)
for i in range(m):
    g = grp[i]
    ks = list(rng.choice(nb, size=4, replace=False)) if g < 0 else [g]
    for k in ks:
        for j in rng.choice(seg[k], size=rng.integers(1, 3), replace=False):
            rows.append(i); cols.append(j); vals.append(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.0))
        for _ in range(rng.integers(1, 3)): sdp_entry(i, k)
A = sp.csc_matrix((vals, (rows, cols)), shape=(m, N))
A.sum_duplicates()
# a strictly feasible primal point: x > 0 on the LP part, identity blocks; b = A x0
x0 = np.zeros(N); x0[:nl] = rng.uniform(0.5, 1.5, nl)
for k, n in enumerate(bs): x0[off[k]:off[k] + n * n] = np.eye(n).ravel(order='F')
b = A @ x0
# a dual point: c = A'y0 + z0 with z0 >= 0 on the LP part and PSD blocks, of low rank in the large block (degenerate)
y0 = rng.standard_normal(m)
z0 = np.zeros(N); z0[:nl] = rng.uniform(0.0, 1.0, nl) * (rng.random(nl) < 0.6)
for k, n in enumerate(bs):
    r = n // 3 if k == 0 else max(1, n - 1)
    G = rng.standard_normal((n, r))
    z0[off[k]:off[k] + n * n] = (G @ G.T / r).ravel(order='F')
c = A.T @ y0 + z0
K = {'f': 0.0, 'l': float(nl), 'q': np.zeros((1, 0)), 's': np.array(bs, dtype=float).reshape(1, -1)}
sio.savemat(out, {'A': A, 'b': b.reshape(-1, 1), 'c': c.reshape(-1, 1), 'K': K}, do_compression=True)
print(f"{out}: m {m}, {len(bs)} semidefinite blocks (largest {max(bs)}), {nl} LP variables, {A.nnz} nonzeros, seed {seed}")
