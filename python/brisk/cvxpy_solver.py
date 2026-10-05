"""CVXPY interface:  prob.solve(solver=brisk.BRISK(), acc="high", timelimit=60)

CVXPY hands a conic solver  min c'x  s.t.  s = b - A x in K,  K = {0}^z x R_+^l x SOC x PSD
(PSD in lower-triangle svec form, no sqrt(2) scaling). Here every nonzero cone is written
through CONE ENTRIES, the entries of BRISK's blocks:

    entry = T s + H w_aux          (T: rows -> entries, H: auxiliary variables -> entries)

  * one LP block: the Nonnegatives rows (and small second-order cones, below);
  * one SDP block per PSD cone: entry (p, q), p >= q, is row k of the svec;
  * a second-order cone (t, u) of size k:
      k = 1: t >= 0 (LP);  k = 2: t + u >= 0, t - u >= 0 (LP);  k = 3: [[t + u1, u2], [u2, t - u1]];
      k >= 4, "arrow": [[t, u'], [u, t I]] (one k x k block);
      k >= 4, "tree":  t >= ||u|| split in halves with auxiliary bounds s_L >= ||u_L||,
             s_R >= ||u_R|| and t >= ||(s_L, s_R)||, recursively: k - 2 blocks of order 2 and
             k - 3 auxiliary variables (exact; O(k) entries instead of k^2).
    soc="auto" (default) takes the arrow for k <= soc_arrow_max (default 8), the tree above.

With w = (x, w_aux) and entry = g - G w (g = T b, G = [T A, -H]) the problem goes to BRISK in
one of two SDPA forms (form="auto" (default), "image" or "kernel"), as in the Julia wrapper:

  IMAGE  - the SDPA primal  min c'w  s.t.  sum_i w_i F_i - F0 >= 0,  F_i = -G[:, i], F0 = -g:
           the cone entries are BRISK's Z; every Zeros row is a pair of LP entries (+s_r, -s_r),
           which BRISK's presolve eliminates. Rows of BRISK's Schur complement: the variables.
  KERNEL - BRISK's (P)  min <C,X>  s.t.  <A_i,X> = b_i:  X is the cone entries, one row
           X_e + G_e w = g_e per entry and A_r x = b_r per Zeros row. An entry that is exactly
           one variable (g_e = 0, one coefficient: what CVXPY makes for X >> 0 on a symmetric
           variable, or x >= 0) CLAIMS that variable: w_j = -X_e / G_ej is substituted and the
           row dropped. The other variables are free, split pairs in the LP block.
  auto   - counts the rows left after BRISK eliminates the split pairs (one row each):
           image  n_w - n_zero,  kernel  n_zero + n_entries - n_w, and takes the form with fewer
           split pairs unless the other has under half its rows (dualize.c's rule).

Duals (CVXPY: c + A'z = 0, z in K*): z = T'(d o D) with D = BRISK's X (image) or Z (kernel) at
the entries and d = 2 on off-diagonal entries (the adjoint), except in PSD cones, where CVXPY's
dual is the lower triangle of the dual matrix itself (d = 1). Zeros rows: X(+) - X(-) (image),
-y (kernel). Infeasible problems return the Farkas certificate (A'z = 0, b'z = -1, z in K*):
from BRISK's X ray (image) or its y ray and Z - C (kernel). Statuses: in the image form BRISK's
(P) is CVXPY's dual, in the kernel form its primal.

Options (solver_opts): any command-line option of ./brisk as a keyword, e.g. acc="high",
timelimit=60, threads=2, tol=1e-9, fomssn=0; options=[...] for raw strings; form, soc,
soc_arrow_max as above; conesolver=0 sends a problem without semidefinite cones to the
semidefinite solver (by default it goes to BRISK's second-order cone solver); write_sdpa="f.dat-s" writes the SDPA file handed to BRISK;
inaccurate_tol (default 1e-4): iteration/time limits, interrupts and numerical difficulties
give OPTIMAL_INACCURATE when the largest DIMACS error is at most this, else USER_LIMIT /
SOLVER_ERROR.
"""
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.constraints import SOC, SvecPSD
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.psd_utils import TriangleKind

from . import _core


class _Seg:
    """A group of cone entries: local entry index e, position (i, j), and the terms
    entry_e = sum T_v s_{T_r} + sum H_v w_aux{H_a} as triplets on local entries."""
    __slots__ = ("n", "psd", "i", "j", "Te", "Tr", "Tv", "He", "Ha", "Hv")

    def __init__(self, n, psd, i, j, Te, Tr, Tv, He=(), Ha=(), Hv=()):
        self.n, self.psd = n, psd                  # n = 0: LP entries (i = j = local index + 1)
        self.i, self.j = np.asarray(i, np.int64), np.asarray(j, np.int64)
        self.Te, self.Tr, self.Tv = np.asarray(Te, np.int64), np.asarray(Tr, np.int64), np.asarray(Tv, float)
        self.He, self.Ha, self.Hv = np.asarray(He, np.int64), np.asarray(Ha, np.int64), np.asarray(Hv, float)


def _seg_of_terms(n, psd, ent):
    """A segment from a list of (i, j, terms), terms = [(kind, index, coef)]."""
    i, j, Te, Tr, Tv, He, Ha, Hv = [], [], [], [], [], [], [], []
    for e, (ii, jj, terms) in enumerate(ent):
        i.append(ii); j.append(jj)
        for kind, idx, c in terms:
            if kind == "r":
                Te.append(e); Tr.append(idx); Tv.append(c)
            else:
                He.append(e); Ha.append(idx); Hv.append(c)
    return _Seg(n, psd, i, j, Te, Tr, Tv, He, Ha, Hv)


class _Entries:
    """The cone entries of a CVXPY cone program (see the module docstring)."""

    def __init__(self, dims, nrow, soc="arrow", soc_arrow_max=None):
        if soc not in ("arrow", "tree"):
            raise ValueError('soc must be "auto", "arrow" or "tree"')
        if getattr(dims, "exp", 0) or getattr(dims, "p3d", []) or getattr(dims, "pnd", []):
            raise ValueError("BRISK supports Zero, NonNeg, SOC and PSD cones only")
        self.nz, self.nl = int(dims.zero), int(dims.nonneg)
        self.nrow = nrow
        self.lp = []           # LP segments (n = 0)
        self.sdp = []          # SDP segments, one per block
        self._lpterms = []     # small LP pieces of second-order cones: lists of terms
        self.naux = 0
        row = self.nz
        if self.nl:
            r = np.arange(self.nl)
            self.lp.append(_Seg(0, False, r + 1, r + 1, r, row + r, np.ones(self.nl)))
        row += self.nl
        for k in (int(k) for k in dims.soc):
            if k >= 4 and (soc == "arrow" and (soc_arrow_max is None or k <= soc_arrow_max)):
                p = np.arange(1, k + 1)
                q = np.arange(2, k + 1)
                zi, zj = np.triu_indices(k - 1, 1)       # the zero entries X_pq, 2 <= p < q
                i = np.concatenate((p, np.ones(k - 1, int), zi + 2))
                j = np.concatenate((p, q, zj + 2))
                Te = np.arange(2 * k - 1)
                Tr = np.concatenate((np.full(k, row), row + q - 1))
                self.sdp.append(_Seg(k, False, i, j, Te, Tr, np.ones(2 * k - 1)))
            else:
                self._norm_le([("r", row, 1.0)], [[("r", row + q, 1.0)] for q in range(1, k)])
            row += k
        if self._lpterms:
            self.lp.append(_seg_of_terms(0, False, [(e + 1, e + 1, tt) for e, tt in enumerate(self._lpterms)]))
        for n in (int(n) for n in dims.psd):
            cc, rr = np.triu_indices(n)          # column-major lower triangle = row-major upper
            e = np.arange(cc.size)
            self.sdp.append(_Seg(n, True, cc + 1, rr + 1, e, row + e, np.ones(cc.size)))
            row += cc.size
        if row != nrow:
            raise ValueError(f"cone dimensions ({row}) do not match A ({nrow} rows)")

    def _norm_le(self, t, items):
        """t >= ||items|| (terms lists) with LP entries and blocks of order 2."""
        neg = lambda e: [(k, i, -c) for (k, i, c) in e]     # noqa: E731
        if not items:
            self._lpterms.append(t)
        elif len(items) == 1:
            self._lpterms.append(t + neg(items[0]))
            self._lpterms.append(t + items[0])
        elif len(items) == 2:
            a, b = items
            self.sdp.append(_seg_of_terms(2, False, [(1, 1, t + a), (1, 2, b), (2, 2, t + neg(a))]))
        else:
            h = len(items) // 2
            bounds = []
            for part in (items[:h], items[h:]):
                if len(part) == 1:
                    bounds.append(part[0])
                else:
                    a = [("a", self.naux, 1.0)]
                    self.naux += 1
                    self._norm_le(a, part)
                    bounds.append(a)
            self._norm_le(t, bounds)

    def build(self, with_zero_pairs):
        """Global entry arrays (the LP entries first, then the SDP blocks in order): blk (0 =
        the LP block, b = SDP block b), i, j (1-based), off, d (dual weights), sdp_sizes,
        nlp_entries; returns the sparse maps T (entries x rows) and H (entries x aux)."""
        segs = []
        if with_zero_pairs and self.nz:
            r = np.arange(self.nz)
            e = np.arange(2 * self.nz)
            segs.append(_Seg(0, False, e + 1, e + 1, e, np.repeat(r, 2), np.tile([1.0, -1.0], self.nz)))
        segs += self.lp
        nlp = 0
        blk, ei, ej, d = [], [], [], []
        Tr, Tc, Tv, Hr, Hc, Hv = [], [], [], [], [], []
        off = 0
        for sg in segs:                              # LP: renumber positions consecutively
            m = sg.i.size
            blk.append(np.zeros(m, np.intc)); ei.append(nlp + sg.i); ej.append(nlp + sg.j)
            d.append(np.ones(m))
            Tr.append(off + sg.Te); Tc.append(sg.Tr); Tv.append(sg.Tv)
            Hr.append(off + sg.He); Hc.append(sg.Ha); Hv.append(sg.Hv)
            nlp += m
            off += m
        sizes = []
        for b, sg in enumerate(self.sdp):
            m = sg.i.size
            sizes.append(sg.n)
            blk.append(np.full(m, b + 1, np.intc)); ei.append(sg.i); ej.append(sg.j)
            d.append(np.where((sg.i != sg.j) & (not sg.psd), 2.0, 1.0))
            Tr.append(off + sg.Te); Tc.append(sg.Tr); Tv.append(sg.Tv)
            Hr.append(off + sg.He); Hc.append(sg.Ha); Hv.append(sg.Hv)
            off += m
        cat = lambda L, t: np.concatenate(L).astype(t) if L else np.zeros(0, t)   # noqa: E731
        self.blk, self.ei, self.ej = cat(blk, np.intc), cat(ei, np.intc), cat(ej, np.intc)
        self.d = cat(d, float)
        self.off = self.ei != self.ej
        self.ne = off
        self.nlp_entries = nlp
        self.sdp_sizes = sizes
        T = sp.csr_matrix((cat(Tv, float), (cat(Tr, np.int64), cat(Tc, np.int64))), shape=(off, self.nrow))
        H = sp.csr_matrix((cat(Hv, float), (cat(Hr, np.int64), cat(Hc, np.int64))), shape=(off, self.naux))
        return T, H

    def gather(self, blocks, has_lp):
        """The entry values of BRISK blocks (X or Z; the LP block first when has_lp)."""
        vals = np.empty(self.ne)
        lp = self.blk == 0
        k0 = 1 if has_lp else 0
        if lp.any():
            vals[lp] = blocks[0][self.ei[lp] - 1]
        for b in range(len(self.sdp_sizes)):
            m = self.blk == b + 1
            B = blocks[k0 + b]
            vals[m] = 0.5 * (B[self.ei[m] - 1, self.ej[m] - 1] + B[self.ej[m] - 1, self.ei[m] - 1])
        return vals


def _sdpa_entries(E, Mx, Mf, lpblk, npair0, forms):
    """SDPA entries (mat, blk, i, j, v): row t of Mx (x entries) and of Mf (x free variables,
    split pairs at LP positions npair0 + 2f + 1, + 2) is SDPA matrix t. forms=True: the rows
    are linear forms on the entry values (<A, X> counts an off-diagonal entry twice, so its
    matrix value is half the coefficient); False: the rows are the matrix values."""
    Mx = sp.coo_matrix(Mx)
    Mx.sum_duplicates()
    keep = Mx.data != 0
    rr, ee, vv = Mx.row[keep], Mx.col[keep], Mx.data[keep]
    if forms:
        vv = np.where(E.off[ee], 0.5 * vv, vv)
    mat = [rr]
    blk = [np.where(E.blk[ee] == 0, 1, E.blk[ee] + (1 if lpblk else 0))]
    ii, jj, v = [E.ei[ee]], [E.ej[ee]], [vv]
    if Mf is not None:
        Mf = sp.coo_matrix(Mf)
        Mf.sum_duplicates()
        keep = Mf.data != 0
        fr, fc, fv = Mf.row[keep], Mf.col[keep], Mf.data[keep]
        p = npair0 + 2 * fc + 1
        mat += [fr, fr]; blk += [np.ones(fr.size, int)] * 2
        ii += [p, p + 1]; jj += [p, p + 1]; v += [fv, -fv]
    cat = np.concatenate
    return cat(mat), cat(blk), cat(ii), cat(jj), cat(v)


def cone_program_to_sdpa(A, b, c, dims, form="auto", soc="auto", soc_arrow_max=None):
    """The SDPA data of  min c'x  s.t.  b - A x in K  (CVXPY's cone dims). Returns
    (blocksizes, sdpa_c, mat, blk, i, j, v, info); info maps the solution back.
    soc="auto": arrow blocks in the image form, trees in the kernel form."""
    if form not in ("auto", "image", "kernel"):
        raise ValueError('form must be "auto", "image" or "kernel"')
    if soc not in ("auto", "arrow", "tree"):
        raise ValueError('soc must be "auto", "arrow" or "tree"')
    A = sp.csr_matrix(A)
    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    nrow, nvar = A.shape
    nz = int(dims.zero)
    Az, bz = A[:nz], b[:nz]

    def prep(kind):
        E = _Entries(dims, nrow, ("arrow" if kind == "image" else "tree") if soc == "auto" else soc,
                     soc_arrow_max)
        T, H = E.build(with_zero_pairs=(kind == "image"))
        G = sp.hstack([T @ A, -H]).tocsr()
        G.eliminate_zeros()
        return dict(E=E, T=T, G=G, g=T @ b, nw=nvar + E.naux)

    def kernel_counts(K):
        G, g, nw, E = K["G"], K["g"], K["nw"], K["E"]
        cnt = np.diff(G.indptr)
        cand = np.nonzero((cnt == 1) & (g == 0))[0]           # entries that are one variable
        js = G.indices[G.indptr[cand]]
        _, first = np.unique(js, return_index=True)           # each variable claimed once
        K["ce"], K["cj"] = cand[first], js[first]
        K["ca"] = G.data[G.indptr[K["ce"]]]
        used = np.zeros(nw, bool)
        used[G.indices] = True
        used[Az.indices] = True
        used[:nvar] |= c != 0            # in no constraint but in the objective: unbounded or 0
        claimed = np.zeros(nw, bool)
        claimed[K["cj"]] = True
        K["freej"] = np.nonzero(used & ~claimed)[0]
        K["m"] = nz + E.ne - K["ce"].size                     # rows of (P)
        K["pairs"] = K["freej"].size

    if form == "auto":
        I, K = prep("image"), prep("kernel")
        kernel_counts(K)
        usedI = np.zeros(I["nw"], bool)
        usedI[I["G"].indices] = True
        effI, effK = int(usedI.sum()) - nz, K["m"] - K["pairs"]
        form = ("image" if 2 * effI < effK else "kernel") if K["pairs"] <= nz else \
               ("kernel" if 2 * effK < effI else "image")
        if K["m"] < 1:
            form = "image"
        S = I if form == "image" else K
    else:
        S = prep(form)
        if form == "kernel":
            kernel_counts(S)
    E, T, G, g, nw = S["E"], S["T"], S["G"], S["g"], S["nw"]
    naux = E.naux
    cw = np.concatenate((c, np.zeros(naux)))
    info = dict(nvar=nvar, naux=naux, nz=nz, form=form, E=E, T=T)

    if form == "image":
        lpblk = E.nlp_entries > 0
        bs = ([-E.nlp_entries] if lpblk else []) + E.sdp_sizes
        # F_i = -G[:, i] (i = 1..nw), F0 = -g: the rows are matrix values on the entries
        Mx = sp.vstack([sp.csr_matrix(-g[None, :]), -G.T.tocsr()])
        mat, blk, ii, jj, v = _sdpa_entries(E, Mx, None, lpblk, 0, forms=False)
        info.update(lpblk=lpblk)
        return (np.array(bs, np.intc), cw, mat, blk, ii, jj, v, info)

    # KERNEL: w = P X + Q w_free (claimed variables from their entries)
    ce, cj, ca, freej = S["ce"], S["cj"], S["ca"], S["freej"]
    nf = freej.size
    P = sp.csr_matrix((-1.0 / ca, (cj, ce)), shape=(nw, E.ne))
    Q = sp.csr_matrix((np.ones(nf), (freej, np.arange(nf))), shape=(nw, nf))
    unc = np.ones(E.ne, bool)
    unc[ce] = False
    U = np.nonzero(unc)[0]
    GU = G[U]
    Ie = sp.csr_matrix((np.ones(U.size), (np.arange(U.size), U)), shape=(U.size, E.ne))
    AzW = sp.hstack([Az, sp.csr_matrix((nz, naux))]).tocsr()       # Zeros rows on w
    MX = sp.vstack([Ie + GU @ P, AzW @ P]).tocsr()
    MF = sp.vstack([GU @ Q, AzW @ Q]).tocsr()
    rhs = np.concatenate((g[U], bz))
    CX = sp.csr_matrix(cw @ P)
    CF = sp.csr_matrix(cw @ Q)
    npair0 = E.nlp_entries
    nlp = npair0 + 2 * nf
    lpblk = nlp > 0
    bs = ([-nlp] if lpblk else []) + E.sdp_sizes
    # SDPA: c = rhs, F_t = A_t (t = 1..m), F0 = -C; the rows are linear forms on X
    mat, blk, ii, jj, v = _sdpa_entries(E, sp.vstack([-CX, MX]), sp.vstack([-CF, MF]), lpblk, npair0,
                                        forms=True)
    info.update(lpblk=lpblk, P=P, Q=Q, npair0=npair0, nf=nf, nrow_sdpa=rhs.size)
    return (np.array(bs, np.intc), rhs, mat, blk, ii, jj, v, info)


def _x_of(res, info):
    if info["form"] == "image":
        return -res.y[:info["nvar"]]
    E = info["E"]
    Xv = E.gather(res.X, info["lpblk"])
    wf = np.zeros(info["nf"])
    if info["nf"]:
        lp = res.X[0]
        p0 = info["npair0"]
        wf = lp[p0:p0 + 2 * info["nf"]:2] - lp[p0 + 1:p0 + 2 * info["nf"]:2]
    w = info["P"] @ Xv + info["Q"] @ wf
    return w[:info["nvar"]]


def _z_of(D, yzero, info, euclid=False):
    """CVXPY's z (all rows) from the entry values D (the cone dual) and the Zeros-row duals;
    euclid: the plain adjoint (weight 2 on every off-diagonal entry) instead of CVXPY's
    convention for PSD cones."""
    E = info["E"]
    z = info["T"].T @ ((np.where(E.off, 2.0, 1.0) if euclid else E.d) * D)
    if info["form"] == "kernel":
        z[:info["nz"]] = yzero
    return z


def _duals(res, info, ray=False, euclid=False):
    E = info["E"]
    if info["form"] == "image":
        if res.X is None:
            return None
        return _z_of(E.gather(res.X, info["lpblk"]), None, info, euclid)
    # kernel: the cone dual is Z (a ray: Z - C = -sum y_i A_i), the Zeros rows are -y
    if res.Z is None or res.y is None:
        return None
    # the ray: -sum_i y_i A_i (= Z - C), from the SDPA data
    Z = info["ray_blocks"](res.y) if ray else res.Z
    D = E.gather(Z, info["lpblk"])
    yz = -res.y[info["nrow_sdpa"] - info["nz"]:info["nrow_sdpa"]]
    return _z_of(D, yz, info, euclid)


def _ray_blocks_factory(bs, mat, blk, ii, jj, v):
    """-sum_i y_i A_i per block (the kernel form's infeasibility ray)."""
    sel = mat > 0
    mat, blk, ii, jj, v = mat[sel], blk[sel], ii[sel], jj[sel], v[sel]

    def f(y):
        coef = -y[mat - 1] * v
        out = []
        for k, n in enumerate(bs):
            m = blk == k + 1
            if n < 0:
                B = np.zeros(-n)
                np.add.at(B, ii[m] - 1, coef[m])
            else:
                B = np.zeros((n, n))
                np.add.at(B, (ii[m] - 1, jj[m] - 1), coef[m])
                o = m & (ii != jj)
                np.add.at(B, (jj[o] - 1, ii[o] - 1), coef[o])
            out.append(B)
        return out
    return f


class BRISK(ConicSolver):
    """BRISK (SDP solver) for CVXPY:  prob.solve(solver=brisk.BRISK(), acc="high")."""

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, SvecPSD]
    PSD_TRIANGLE_KIND = TriangleKind.LOWER
    PSD_SQRT2_SCALING = False
    # BRISK status -> CVXPY, per form (BRISK's (P) is CVXPY's dual in the image form)
    STATUS_IMAGE = {0: s.OPTIMAL, 5: s.OPTIMAL_INACCURATE, 1: s.UNBOUNDED, 2: s.INFEASIBLE,
                    3: s.USER_LIMIT, 6: s.USER_LIMIT, 4: s.SOLVER_ERROR}
    STATUS_KERNEL = {**STATUS_IMAGE, 1: s.INFEASIBLE, 2: s.UNBOUNDED}

    def name(self):
        return "BRISK"

    def import_solver(self):
        _core.load_library()

    def supports_quad_obj(self):
        return False

    def cite(self, data):
        return ("@misc{brisk, title = {{BRISK}: an {SDP} solver}, note = {version "
                + _core.version() + "}}")

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        opts = dict(solver_opts or {})
        raw = opts.pop("options", None)
        wfile = opts.pop("write_sdpa", None)
        itol = opts.pop("inaccurate_tol", 1e-4)
        form = opts.pop("form", "auto")
        soc = opts.pop("soc", "auto")
        amax = opts.pop("soc_arrow_max", None)
        bound = opts.pop("bound", None)
        do_cert = bool(opts.pop("certify", False))
        opts.pop("use_quad_obj", None)
        dims = data[ConicSolver.DIMS]
        cone = opts.pop("conesolver", 1)
        if (int(cone) != 0 and not list(dims.psd) and not bound and form == "auto" and soc == "auto"
                and not wfile and not any(k in opts for k in ("prec", "fom", "mfipm", "lralm"))):
            # no semidefinite cone: BRISK's second-order cone solver, on the SeDuMi form whose
            # dual is CVXPY's problem (A_s = A', b_s = -c, c_s = b; y = x, x_s = CVXPY's dual)
            A = sp.csc_matrix(data[s.A]).T
            K = dict(f=int(dims.zero), l=int(dims.nonneg), q=[int(k) for k in dims.soc])
            try:
                res = _core.solve_sedumi(A, -np.asarray(data[s.C], float).ravel(), np.asarray(data[s.B], float).ravel(), K,
                                         options=_core.options_to_argv(opts) + _core.options_to_argv(raw), verbose=verbose)
            except RuntimeError as e:
                raise SolverError(str(e)) from e
            return {"cone": res, "itol": itol, "c": np.asarray(data[s.C], float).ravel()}
        *sd, info = cone_program_to_sdpa(data[s.A], data[s.B], data[s.C], data[ConicSolver.DIMS],
                                         form=form, soc=soc, soc_arrow_max=amax)
        if info["form"] == "kernel":
            info["ray_blocks"] = _ray_blocks_factory(*sd[:1], *sd[2:])
        if wfile:
            _core.write_sdpa(wfile, *sd)
        argv = _core.options_to_argv(opts) + _core.options_to_argv(raw)
        side = None
        if bound:
            # "primal"/"sos": a feasible point of the model (the side of its variables, the
            # SOS Gram matrices); "dual": a feasible dual. In the image form CVXPY's primal
            # is BRISK's (D), in the kernel form BRISK's (P).
            b0 = str(bound).lower()
            if b0 not in ("primal", "dual", "sos", "p", "d"):
                raise ValueError('bound must be "primal", "dual" or "sos"')
            prim = b0 in ("primal", "sos", "p")
            side = ("d" if prim else "p") if info["form"] == "image" else ("p" if prim else "d")
            argv += ["-bound", side] + (["-certify"] if do_cert else [])
        info["bound_side"] = side
        try:
            res = _core.solve_sdpa(*sd, options=argv, verbose=verbose)
        except RuntimeError as e:
            raise SolverError(str(e)) from e
        return {"res": res, "info": info, "c": np.asarray(data[s.C], float).ravel(), "itol": itol,
                "b": np.asarray(data[s.B], float).ravel()}

    def invert(self, solution, inverse_data):
        if "cone" in solution:
            return self._invert_cone(solution, inverse_data)
        res, info = solution["res"], solution["info"]
        attr = {s.SOLVE_TIME: res.time, s.NUM_ITERS: res.iterations,
                s.EXTRA_STATS: {"status": res.status_str, "dimacs": res.dimacs,
                                "exit_code": res.exit_code, "form": info["form"],
                                "blocksizes": res.blocksizes, "interrupted": res.interrupted}}
        if res.bound:
            # the bound on the conic problem  min c'x + offset  that CVXPY solves (for a
            # Maximize model CVXPY minimizes -f: the kind and the sign then flip)
            v = res.bound["value"]
            img = info["form"] == "image"
            conic = (-v if img else v) + inverse_data[s.OFFSET]
            kind = res.bound["kind"]
            if img:
                kind = "lower" if kind == "upper" else "upper"
            rig = res.bound.get("rigorous")
            if rig is not None and np.isfinite(rig):
                rig = (-rig if img else rig) + inverse_data[s.OFFSET]
            attr[s.EXTRA_STATS]["bound"] = dict(value=conic, kind=kind, brisk_side=res.bound["side"],
                                                resid=res.bound["resid"], lammin=res.bound["lammin"],
                                                valid=res.bound["valid"], certified=res.bound["certified"],
                                                rigorous=rig)
        attr[s.EXTRA_STATS]["n_resolves"] = res.n_resolves
        attr[s.EXTRA_STATS]["cause"] = res.cause
        smap = self.STATUS_IMAGE if info["form"] == "image" else self.STATUS_KERNEL
        status = smap.get(res.status, s.SOLVER_ERROR)
        if status in (s.USER_LIMIT, s.SOLVER_ERROR) and res.y is not None \
                and np.max(np.abs(res.dimacs)) <= solution["itol"]:
            status = s.OPTIMAL_INACCURATE
        nz = inverse_data[ConicSolver.DIMS].zero

        def dual_dict(z):
            if z is None:
                return {}
            d = utilities.get_dual_values(z[:nz], utilities.extract_dual_value,
                                          inverse_data[self.EQ_CONSTR])
            d.update(utilities.get_dual_values(z[nz:], utilities.extract_dual_value,
                                               inverse_data[self.NEQ_CONSTR]))
            return d
        if status in s.SOLUTION_PRESENT and res.y is not None and \
                (info["form"] == "image" or res.X is not None):
            x = _x_of(res, info)
            opt_val = float(solution["c"] @ x) + inverse_data[s.OFFSET]
            return Solution(status, opt_val, {inverse_data[self.VAR_ID]: x},
                            dual_dict(_duals(res, info)), attr)
        if status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            z = self._certificate(res, info, solution["b"])
            return failure_solution(status, attr, dual_dict(z))
        return failure_solution(status, attr)

    def _invert_cone(self, solution, inverse_data):
        """The result of the second-order cone solver (the SeDuMi form's dual is CVXPY's problem)."""
        res = solution["cone"]
        attr = {s.SOLVE_TIME: res.time, s.NUM_ITERS: res.iterations,
                s.EXTRA_STATS: {"status": res.status_str, "dimacs": res.dimacs, "exit_code": res.exit_code,
                                "form": "cone", "interrupted": res.interrupted}}
        status = self.STATUS_IMAGE.get(res.status, s.SOLVER_ERROR)
        if status in (s.USER_LIMIT, s.SOLVER_ERROR) and res.y is not None \
                and np.max(np.abs(res.dimacs)) <= solution["itol"]:
            status = s.OPTIMAL_INACCURATE
        nz = inverse_data[ConicSolver.DIMS].zero

        def dual_dict(z):
            if z is None:
                return {}
            d = utilities.get_dual_values(z[:nz], utilities.extract_dual_value, inverse_data[self.EQ_CONSTR])
            d.update(utilities.get_dual_values(z[nz:], utilities.extract_dual_value, inverse_data[self.NEQ_CONSTR]))
            return d
        if status in s.SOLUTION_PRESENT and res.y is not None:
            opt_val = float(solution["c"] @ res.y) + inverse_data[s.OFFSET]
            return Solution(status, opt_val, {inverse_data[self.VAR_ID]: res.y}, dual_dict(res.x), attr)
        if status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            return failure_solution(status, attr, dual_dict(res.x))     # A'z = 0, z in K*, b'z = -1
        return failure_solution(status, attr)

    @staticmethod
    def _certificate(res, info, b):
        """Farkas certificate A'z = 0, z in K*, scaled to b'z = -1 (None if not available)."""
        try:
            z = _duals(res, info, ray=True)
            ze = _duals(res, info, ray=True, euclid=True)
        except Exception:
            return None
        if z is None:
            return None
        bz = float(b @ ze)
        if not np.isfinite(bz) or bz >= 0:
            return None
        return z / -bz
