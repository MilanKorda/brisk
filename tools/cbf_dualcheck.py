"""cbf_dualcheck.py FILE.cbf[.gz] X Y Z (12 November 2026, the SOCP developer): checks the point the command line writes for a CBF
file (-x, -y, -z, one value per line) against CBF's duality (linear and second-order cones only):
  primal  min c'x + c0  s.t.  A x + b in K_con,  x in K_var
  dual    max -b'y + c0 s.t.  c - A'y = z,  z in K_var*,  y in K_con*
(a maximized file: z and y in the negated dual cones). Prints the relative residual of z = c - A'y, the largest violation of each
cone condition relative to the point's size, and both objectives; exit code 1 when a measure is above 1e-6 (relative). `python3 -I tools/cbf_dualcheck.py ...`"""
import sys, gzip, numpy as np, scipy.sparse as sp


def read_cbf(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        L = [ln.strip() for ln in f]
    L = [l for l in L if l and not l.startswith("#")]
    i = 0; d = {"sense": "MIN", "c0": 0.0}
    nv = nc = 0; vd = []; cd = []; c = None; A = ([], [], []); b = None
    while i < len(L):
        k = L[i]; i += 1
        if k == "OBJSENSE": d["sense"] = L[i]; i += 1
        elif k in ("VAR", "CON"):
            n, g = map(int, L[i].split()); i += 1; dom = []
            for _ in range(g):
                t, m = L[i].split(); i += 1; dom.append((t, int(m)))
            if k == "VAR": nv, vd = n, dom
            else: nc, cd = n, dom
        elif k == "OBJACOORD":
            m = int(L[i]); i += 1; c = np.zeros(nv)
            for _ in range(m):
                j, v = L[i].split(); i += 1; c[int(j)] += float(v)
        elif k == "OBJBCOORD": d["c0"] = float(L[i]); i += 1
        elif k == "ACOORD":
            m = int(L[i]); i += 1
            for _ in range(m):
                r, j, v = L[i].split(); i += 1; A[0].append(int(r)); A[1].append(int(j)); A[2].append(float(v))
        elif k == "BCOORD":
            m = int(L[i]); i += 1; b = np.zeros(nc)
            for _ in range(m):
                r, v = L[i].split(); i += 1; b[int(r)] += float(v)
        elif k in ("VER", "POWCONES", "POW*CONES", "INT", "PSDVAR", "PSDCON"):
            if k in ("PSDVAR", "PSDCON", "POWCONES", "POW*CONES"): raise SystemExit("not checked: " + k)
            if k == "INT": m = int(L[i]); i += 1 + m
            else: i += 1
        else: raise SystemExit("unknown section " + k)
    if c is None: c = np.zeros(nv)
    if b is None: b = np.zeros(nc)
    Am = sp.csr_matrix((A[2], (A[0], A[1])), shape=(nc, nv))
    return d, nv, nc, vd, cd, c, Am, b


def viol_dual(v, dom, sgn):
    """largest violation of v (times sgn) in the dual cones of the domain list (self-dual cones; F's dual is {0}, L='s is free)"""
    w = sgn * v; p = 0; worst = 0.0
    for t, m in dom:
        s = w[p:p + m]; p += m
        if t == "F": worst = max(worst, np.abs(s).max())
        elif t == "L+": worst = max(worst, (-s).max())
        elif t == "L-": worst = max(worst, s.max())
        elif t == "L=": pass
        elif t == "Q": worst = max(worst, np.linalg.norm(s[1:]) - s[0])
        elif t == "QR": worst = max(worst, max(-s[0], -s[1], s[2:] @ s[2:] - 2 * s[0] * s[1]) if m > 2 else max(-s[0], -s[1]))
        else: raise SystemExit("domain " + t)
    return max(worst, 0.0)


def readv(p):
    return np.array([float(l.split()[-1]) for l in open(p) if l.strip()])


if __name__ == "__main__":
    d, nv, nc, vd, cd, c, A, b = read_cbf(sys.argv[1])
    x, y, z = readv(sys.argv[2]), readv(sys.argv[3]), readv(sys.argv[4])
    print("variables %d (x %d, z %d), constraints %d (y %d), %s" % (nv, x.size, z.size, nc, y.size, d["sense"]))
    sg = -1.0 if d["sense"] == "MAX" else 1.0
    r = c - A.T @ y - z
    print("z = c - A'y: residual %.2e (relative to 1 + |c| + |A'y|: %.2e)" % (np.abs(r).max(), np.abs(r).max() / (1 + np.abs(c).max() + np.abs(A.T @ y).max())))
    print("z in the dual cones of the variables (sign %+d): violation %.2e (|z| %.2e)" % (sg, viol_dual(z, vd, sg), np.abs(z).max()))
    print("y in the dual cones of the constraints (sign %+d): violation %.2e (|y| %.2e)" % (sg, viol_dual(y, cd, sg), np.abs(y).max()))
    print("primal: c'x + c0 = %.10e, A x + b in K_con: violation %.2e" % (c @ x + d["c0"], viol_dual(A @ x + b, cd, 1.0)))
    print("dual:  -b'y + c0 = %.10e" % (-b @ y + d["c0"]))
    # the exit code (the release check, RELEASE_CHECKS R20): every measure relative to the point's size below 1e-6
    sc = 1 + np.abs(c).max() + np.abs(A.T @ y).max(); po = c @ x + d["c0"]; do = -b @ y + d["c0"]
    bad = [np.abs(r).max() / sc, viol_dual(z, vd, sg) / (1 + np.abs(z).max()), viol_dual(y, cd, sg) / (1 + np.abs(y).max()),
           abs(po - do) / (1 + abs(po))]
    if x.size != nv or y.size != nc or z.size != nv or max(bad) > 1e-6: print("cbf_dualcheck: FAILED", bad); sys.exit(1)
    print("cbf_dualcheck: passed")
