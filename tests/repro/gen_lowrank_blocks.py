#!/usr/bin/env python3
"""Two semidefinite blocks of order n, each constraint in one block (SDPA sparse format, written to argv[1]).
Block 1: dense constraints of full rank (the low-rank check runs on the threads and rejects the block); block 2: dense
constraints of rank two, whose supports are larger than 64 rows (their factors come from random vectors). The next
block's random vectors must not depend on which thread checked which constraint of the block before (ISSUES 736).
max -tr X (SDPA: F0 = -I) subject to <A_i, X> = <A_i, I>: X = I is feasible, y = 0 with Z = I is dual feasible."""
import sys, random
n, m1, m2 = 80, 40, 40
rnd = random.Random(20261101)
cons = []
for i in range(m1):
    M = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a, n):
            M[a][b] = M[b][a] = rnd.uniform(-1, 1)
    cons.append((1, M))
for i in range(m2):
    u = [rnd.uniform(-1, 1) for _ in range(n)]; v = [rnd.uniform(-1, 1) for _ in range(n)]
    M = [[u[a] * u[b] - 0.5 * v[a] * v[b] for b in range(n)] for a in range(n)]
    cons.append((2, M))
with open(sys.argv[1], "w") as f:
    f.write(f"{m1 + m2}\n2\n{n} {n}\n")
    f.write(" ".join("%.17g" % sum(M[a][a] for a in range(n)) for _, M in cons) + "\n")
    for blk in (1, 2):
        for a in range(n):
            f.write(f"0 {blk} {a + 1} {a + 1} -1\n")
    for i, (blk, M) in enumerate(cons):
        for a in range(n):
            for b in range(a, n):
                f.write(f"{i + 1} {blk} {a + 1} {b + 1} {M[a][b]:.17g}\n")
