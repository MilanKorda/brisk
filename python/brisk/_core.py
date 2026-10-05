"""ctypes binding of libbrisk's flat C interface (capi.c).

Conventions are those of the command line (brisk.h): the problem is the numbers of an SDPA
sparse file,

    SDPA primal   min c'x   s.t.   F(x) = sum_i x_i F_i - F_0  >= 0,

given as block sizes (LP blocks negative) and the file's entries (mat, blk, i, j, v), 1-based
as in the file (mat 0 = F_0). BRISK solves (P) min <C,X> s.t. <A_i,X> = b_i, X >= 0 and its
dual (D) max b'y s.t. C - sum y_i A_i = Z >= 0 with C = -F_0, A_i = F_i, b = c; the SDPA
variable is x = -y, the SDPA slack F(x) is Z, and X is the SDPA dual matrix.
"""
import ctypes
import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

STATUS = {0: "OPTIMAL", 1: "PRIMAL INFEASIBLE", 2: "DUAL INFEASIBLE", 3: "ITERATION LIMIT",
          4: "NUMERICAL DIFFICULTIES", 5: "REDUCED ACCURACY", 6: "TIME LIMIT", -1: "NOT SOLVED"}

_lib = None
_lib_path = None
_lock = threading.Lock()          # libbrisk is not reentrant: one solve at a time per process
_PRINT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int)


def _default_path():
    p = os.environ.get("BRISK_LIBRARY", "")
    if p:
        return p
    ext = "dylib" if sys.platform == "darwin" else "so"
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "libbrisk." + ext),                      # copied into the package
                 os.path.join(here, "..", "..", "libbrisk." + ext)):         # BRISK source tree
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return os.path.normpath(os.path.join(here, "..", "..", "libbrisk." + ext))


def load_library(path=None):
    """Load libbrisk (done automatically on first use). The search order is `path`, the
    environment variable BRISK_LIBRARY, a copy inside the package, the BRISK source tree
    (python/brisk/../../libbrisk.so). On Linux the library is opened with RTLD_DEEPBIND so
    that its BLAS/LAPACK calls go to the (LP64) BLAS it was linked with, not to one already
    loaded by numpy/scipy."""
    global _lib, _lib_path
    if _lib is not None and (path is None or os.path.abspath(path) == _lib_path):
        return _lib
    path = os.path.abspath(path or _default_path())
    if not os.path.exists(path):
        raise OSError(f"libbrisk not found at {path}: build it with `make libbrisk` in the BRISK "
                      "source directory, or set the environment variable BRISK_LIBRARY to its path")
    mode = os.RTLD_LOCAL if hasattr(os, "RTLD_LOCAL") else 0
    if sys.platform.startswith("linux") and hasattr(os, "RTLD_DEEPBIND"):
        mode |= os.RTLD_DEEPBIND
    L = ctypes.CDLL(path, mode=mode)
    P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    pd = ctypes.POINTER(ctypes.c_double)
    L.brisk_version.restype = ctypes.c_char_p
    L.brisk_set_output.argtypes = [I, _PRINT]
    L.brisk_set_output.restype = None
    L.brisk_result_new.restype = P
    L.brisk_result_delete.argtypes = [P]
    L.brisk_result_delete.restype = None
    for name, rt in (("status", I), ("exit_code", I), ("iterations", I), ("m", I), ("nblk", I),
                     ("have_x", I), ("pobj", D), ("dobj", D), ("time", D)):
        f = getattr(L, "brisk_result_" + name)
        f.argtypes, f.restype = [P], rt
    L.brisk_result_status_str.argtypes, L.brisk_result_status_str.restype = [P], ctypes.c_char_p
    L.brisk_result_blocksize.argtypes, L.brisk_result_blocksize.restype = [P, I], I
    L.brisk_result_dimacs.argtypes, L.brisk_result_dimacs.restype = [P, pd], None
    L.brisk_result_y.argtypes, L.brisk_result_y.restype = [P], pd
    L.brisk_result_X.argtypes, L.brisk_result_X.restype = [P, I], pd
    L.brisk_result_Z.argtypes, L.brisk_result_Z.restype = [P, I], pd
    pi = ctypes.POINTER(ctypes.c_int)
    ppc = ctypes.POINTER(ctypes.c_char_p)
    L.brisk_solve_data.argtypes = [I, I, pi, pd, ctypes.c_int64, pi, pi, pi, pi, pd, I, ppc, P]
    L.brisk_solve_data.restype = I
    L.brisk_solve_file.argtypes = [ctypes.c_char_p, I, ppc, P]
    L.brisk_solve_file.restype = I
    L.brisk_result_bound.argtypes, L.brisk_result_bound.restype = [P, pd], I
    L.brisk_result_resolves.argtypes, L.brisk_result_resolves.restype = [P, pd], I
    L.brisk_result_cause.argtypes, L.brisk_result_cause.restype = [P], ctypes.c_char_p
    if hasattr(L, "brisk_hp_text"):          # high precision: all the digits as text
        L.brisk_hp_digits.argtypes, L.brisk_hp_digits.restype = [], I
        L.brisk_hp_nblk.argtypes, L.brisk_hp_nblk.restype = [], I
        L.brisk_hp_blocksize.argtypes, L.brisk_hp_blocksize.restype = [I], I
        L.brisk_hp_count.argtypes, L.brisk_hp_count.restype = [I, I], ctypes.c_longlong
        L.brisk_hp_text.argtypes, L.brisk_hp_text.restype = [I, I, ctypes.c_char_p, ctypes.c_longlong], ctypes.c_longlong
    if hasattr(L, "brisk_solve_sedumi"):     # problems in SeDuMi format
        L.brisk_solve_sedumi.argtypes = [I, I, pi, pi, pd, pd, pd, I, I, I, pi, I, pi, I, pi, I, ppc, P]
        L.brisk_solve_sedumi.restype = I
        L.brisk_result_sn.argtypes, L.brisk_result_sn.restype = [P], I
        L.brisk_result_sx.argtypes, L.brisk_result_sx.restype = [P], pd
        L.brisk_result_sz.argtypes, L.brisk_result_sz.restype = [P], pd
    L.brisk_interrupt.argtypes, L.brisk_interrupt.restype = [], None
    L.brisk_interrupted.argtypes, L.brisk_interrupted.restype = [], I
    _lib, _lib_path = L, path
    return L


def library_path():
    load_library()
    return _lib_path


def version():
    """The version of the loaded libbrisk."""
    return load_library().brisk_version().decode()


@dataclass
class Result:
    """The solution of the problem as given. y: BRISK's y (the SDPA x is -y); X, Z: one array
    per block, n x n (SDP) or n (LP); dimacs: DIMACS errors 1..6 on the data as given."""
    status: int
    status_str: str
    exit_code: int
    iterations: int
    pobj: float                      # <C,X>
    dobj: float                      # b'y  (the SDPA value c'x is -dobj)
    time: float
    dimacs: np.ndarray
    blocksizes: List[int] = field(default_factory=list)
    y: Optional[np.ndarray] = None
    X: Optional[List[np.ndarray]] = None
    Z: Optional[List[np.ndarray]] = None
    interrupted: bool = False        # stopped by Ctrl-C / brisk.interrupt() (status 6)
    bound: Optional[dict] = None     # -bound p|d: side, value, resid, lammin, valid, certified
    n_resolves: int = 0              # re-solves run (method retry, fallbacks) and their time
    t_resolves: float = 0.0
    cause: str = ""                  # likely cause when the tolerance is missed

    @property
    def x(self):
        """The SDPA primal variable x = -y."""
        return None if self.y is None else -self.y

    def __repr__(self):
        return (f"brisk.Result({self.status_str}, pobj={self.pobj:.10g}, dobj={self.dobj:.10g}, "
                f"max DIMACS {np.max(np.abs(self.dimacs)):.1e}, {self.iterations} it, {self.time:.2f} s)")


def options_to_argv(options):
    """Command-line options from a dict {"acc": "high", "timelimit": 60, "chordal": 0} (a key
    may carry its dash; an underscore stands for a dash, "certify_y" -> "-certify-y"; True
    gives the bare flag, None/False drops it; 0/1 on a flag also works), a list of strings
    (["-acc", "high"]) or a string ("-acc high"). Every option of the command line can be
    given; OPTIONS.md in the BRISK directory lists them."""
    if options is None:
        return []
    if isinstance(options, str):
        return options.split()
    if isinstance(options, dict):
        av = []
        for k, v in options.items():
            if v is None or v is False:
                continue
            k = str(k)
            av.append(k if k.startswith("-") else "-" + k.replace("_", "-"))
            if v is not True:
                av.append(repr(v) if isinstance(v, float) else str(v))
        return av
    return [str(o) for o in options]


class _Worker:
    """One long-lived thread that runs the solves started from the main thread, so that the
    main thread stays free to receive Ctrl-C (a ctypes call cannot be interrupted) and the
    OpenMP thread pool is created once."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="brisk-solver", daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            job = self.jobs.get()
            job()


_worker = None
_interruptible = True


def set_interruptible(flag):
    """True (default): solves started from the main thread run in a worker thread and Ctrl-C
    stops them cleanly (the result so far is returned with status TIME LIMIT and
    `interrupted` set; a second Ctrl-C abandons the solve and raises KeyboardInterrupt, the
    solver then finishes in the background). False: solves run in the calling thread and
    cannot be interrupted."""
    global _interruptible
    _interruptible = bool(flag)


def interrupt():
    """Stop the running solve (from any thread or a signal handler) as if its time limit
    were reached now."""
    load_library().brisk_interrupt()


def _run(call, options, verbose, collect=None):
    global _worker
    L = load_library()
    av = options_to_argv(options)
    arr = (ctypes.c_char_p * max(len(av), 1))(*[a.encode() for a in av])

    def cb(s, is_err):
        (sys.stderr if is_err else sys.stdout).write(s.decode(errors="replace"))
        return 0
    cbp = _PRINT(cb)
    r = L.brisk_result_new()
    if not r:
        raise MemoryError("brisk_result_new failed")
    state = {}
    done = threading.Event()
    guard = threading.Lock()

    def cleanup():
        L.brisk_set_output(0, _PRINT(0))
        L.brisk_result_delete(r)
        _lock.release()

    def job():
        try:
            L.brisk_set_output(2 if verbose else 1, cbp if verbose else _PRINT(0))
            state["rc"] = call(L, len(av), arr, r)
            state["interrupted"] = bool(L.brisk_interrupted())
        except BaseException as e:          # noqa: B036 (ctypes errors, re-raised by the caller)
            state["exc"] = e
        finally:
            with guard:
                done.set()
                if state.get("abandoned"):
                    cleanup()

    _lock.acquire()
    try:
        if _interruptible and threading.current_thread() is threading.main_thread():
            if _worker is None:
                _worker = _Worker()
            _worker.jobs.put(job)
            stopping = False
            while True:
                try:
                    while not done.wait(0.05):
                        pass
                    break
                except KeyboardInterrupt:
                    if not stopping:
                        stopping = True
                        L.brisk_interrupt()
                        sys.stderr.write("\nbrisk: interrupted - stopping at the next iteration "
                                         "(Ctrl-C again to abandon the solve)\n")
                        continue
                    with guard:
                        if not done.is_set():
                            state["abandoned"] = True     # the worker cleans up
                            raise
                    break
        else:
            job()
    except BaseException:
        if not state.get("abandoned"):
            cleanup()
        raise
    try:
        sys.stdout.flush()
        if "exc" in state:
            raise state["exc"]
        res = (collect or _collect)(L, r, state["rc"])
        res.interrupted = state.get("interrupted", False)
        if res.interrupted:
            res.status_str = "INTERRUPTED"
        return res
    finally:
        cleanup()


def _collect(L, r, rc):
    err = (ctypes.c_double * 6)()
    L.brisk_result_dimacs(r, err)
    nblk = L.brisk_result_nblk(r)
    bs = [L.brisk_result_blocksize(r, k) for k in range(nblk)]
    m = L.brisk_result_m(r)
    status = L.brisk_result_status(r)
    res = Result(status=status, status_str=L.brisk_result_status_str(r).decode(), exit_code=rc,
                 iterations=L.brisk_result_iterations(r), pobj=L.brisk_result_pobj(r),
                 dobj=L.brisk_result_dobj(r), time=L.brisk_result_time(r),
                 dimacs=np.array(err[:]), blocksizes=bs)
    bo = (ctypes.c_double * 6)()
    side = L.brisk_result_bound(r, bo)
    if side:
        res.bound = dict(side="p" if side == 1 else "d", kind="upper" if side == 1 else "lower",
                         value=bo[0], resid=bo[1], lammin=bo[2], valid=int(bo[3]), certified=bool(bo[4]),
                         rigorous=bo[5])
    tr = ctypes.c_double(0)
    res.n_resolves = L.brisk_result_resolves(r, ctypes.byref(tr))
    res.t_resolves = tr.value
    res.cause = (L.brisk_result_cause(r) or b"").decode()
    py = L.brisk_result_y(r)
    if py and m > 0:
        res.y = np.ctypeslib.as_array(py, shape=(m,)).copy()

    def blocks(get):
        out = []
        for k, n in enumerate(bs):
            p = get(r, k)
            if not p:
                return None
            if n < 0:
                out.append(np.ctypeslib.as_array(p, shape=(-n,)).copy())
            else:
                out.append(np.ctypeslib.as_array(p, shape=(n * n,)).reshape((n, n), order="F").copy())
        return out
    if L.brisk_result_have_x(r):
        res.X = blocks(L.brisk_result_X)
    res.Z = blocks(L.brisk_result_Z)
    if rc < 0 or rc in (1, 2) or (status == -1 and rc not in (0, 20)):
        raise RuntimeError(f"BRISK did not solve the problem (exit code {rc}: "
                           + {1: "bad options", 2: "invalid problem data"}.get(rc, "aborted") + ")")
    return res


@dataclass
class ConeResult:
    """The solution of a problem in SeDuMi format: x (primal), y, z = c - A'y (dual slack), in
    the order of K. pobj = c'x, dobj = b'y. For an infeasible problem y (status 1, b'y = 1) or
    x (status 2, c'x = -1) is the certificate."""
    status: int
    status_str: str
    exit_code: int
    iterations: int
    pobj: float
    dobj: float
    time: float
    dimacs: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    interrupted: bool = False

    def __repr__(self):
        return (f"brisk.ConeResult({self.status_str}, pobj={self.pobj:.10g}, dobj={self.dobj:.10g}, "
                f"max error={np.max(np.abs(self.dimacs)):.1e}, {self.iterations} iterations, {self.time:.2f}s)")


def _collect_sedumi(L, r, rc):
    err = (ctypes.c_double * 6)()
    L.brisk_result_dimacs(r, err)
    status = L.brisk_result_status(r)
    if rc < 0 or rc in (1, 2) or (status == -1 and rc not in (0, 20)):
        raise RuntimeError(f"BRISK did not solve the problem (exit code {rc}: "
                           + {1: "bad options", 2: "invalid problem data"}.get(rc, "aborted") + ")")
    res = ConeResult(status=status, status_str=L.brisk_result_status_str(r).decode(), exit_code=rc,
                     iterations=L.brisk_result_iterations(r), pobj=L.brisk_result_pobj(r),
                     dobj=L.brisk_result_dobj(r), time=L.brisk_result_time(r), dimacs=np.array(err[:]))
    m, n = L.brisk_result_m(r), L.brisk_result_sn(r)
    for name, ptr, k in (("x", L.brisk_result_sx(r), n), ("y", L.brisk_result_y(r), m), ("z", L.brisk_result_sz(r), n)):
        if ptr and k > 0:
            setattr(res, name, np.ctypeslib.as_array(ptr, shape=(k,)).copy())
    return res


def _is_mat(path):
    with open(path, "rb") as f:
        return f.read(6) == b"MATLAB"


def solve_file(path, options=None, verbose=True):
    """Solve a problem file as the command line (./brisk path options...): an SDPA file
    (.dat-s; returns a Result) or a MAT-file with a problem in SeDuMi format (A or At, b, c,
    K; returns a ConeResult)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such problem file: {path}")   # was "exit code 2: invalid problem data"
    p = os.fsencode(path)
    mat = str(path).endswith(".mat") and _is_mat(path)
    return _run(lambda L, n, a, r: L.brisk_solve_file(p, n, a, r), options, verbose,
                _collect_sedumi if mat else None)


def solve_sedumi(A, b, c, K, options=None, verbose=True):
    """Solve a problem in SeDuMi format:

        min c'x  s.t.  A x = b,  x in K        max b'y  s.t.  c - A'y = z in K*

    A: m x n (scipy sparse or dense; n x m is accepted when unambiguous). K: a dict (or an
    object with these attributes) with f (free variables), l (nonnegative variables), q (list:
    second-order cones x0 >= |x(1:)|), r (list: rotated cones 2 x0 x1 >= |x(2:)|^2), s (list:
    semidefinite blocks, each as its d*d entries by columns), in this order. Without
    semidefinite blocks the problem is solved by BRISK's second-order cone solver, with them by
    the semidefinite solver. Returns a ConeResult."""
    import scipy.sparse as sp
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
    c = np.ascontiguousarray(np.asarray(c, dtype=np.float64).ravel())
    A = sp.csc_matrix(A)
    if A.shape != (b.size, c.size) and A.shape == (c.size, b.size):
        A = sp.csc_matrix(A.T)
    if A.shape != (b.size, c.size):
        raise ValueError(f"A is {A.shape[0]} x {A.shape[1]}, b has {b.size} entries, c {c.size}")
    A.sum_duplicates()
    A.sort_indices()
    get = (lambda k: K.get(k)) if isinstance(K, dict) else (lambda k: getattr(K, k, None))

    def lst(k):
        v = get(k)
        v = [] if v is None else np.atleast_1d(np.asarray(v)).ravel()
        return np.ascontiguousarray([int(d) for d in v if int(d) > 0], dtype=np.intc)

    def cnt(k):
        v = get(k)
        return 0 if v is None or np.size(v) == 0 else int(np.sum(v))
    nf, nl, q, rr, ss = cnt("f"), cnt("l"), lst("q"), lst("r"), lst("s")
    if nf + nl + int(q.sum()) + int(rr.sum()) + int((ss.astype(np.int64) ** 2).sum()) != c.size:
        raise ValueError("K does not match the number of columns of A")
    Ap = np.ascontiguousarray(A.indptr, dtype=np.intc)
    Ai = np.ascontiguousarray(A.indices, dtype=np.intc)
    Ax = np.ascontiguousarray(A.data, dtype=np.float64)
    pi = ctypes.POINTER(ctypes.c_int)
    pd = ctypes.POINTER(ctypes.c_double)
    P = lambda a, t: a.ctypes.data_as(t)   # noqa: E731
    return _run(lambda L, n, a, r: L.brisk_solve_sedumi(
        int(b.size), int(c.size), P(Ap, pi), P(Ai, pi), P(Ax, pd), P(b, pd), P(c, pd), nf, nl,
        int(q.size), P(q, pi), int(rr.size), P(rr, pi), int(ss.size), P(ss, pi), n, a, r),
        options, verbose, _collect_sedumi)


def solve_sdpa(blocksizes, c, mat, blk, i, j, v, options=None, verbose=True):
    """Solve the SDPA problem given by its numbers (see the module docstring): blocksizes
    (LP blocks negative), c (m), and the entries mat (0..m), blk (1..nblk), i, j (1-based
    within the block; i > j is swapped), v."""
    bs = np.ascontiguousarray(blocksizes, dtype=np.intc)
    c = np.ascontiguousarray(c, dtype=np.float64)
    arrs = [np.ascontiguousarray(a, dtype=np.intc) for a in (mat, blk, i, j)]
    v = np.ascontiguousarray(v, dtype=np.float64)
    nnz = v.size
    if any(a.size != nnz for a in arrs):
        raise ValueError("mat, blk, i, j and v must have the same length")
    pi = ctypes.POINTER(ctypes.c_int)
    pd = ctypes.POINTER(ctypes.c_double)
    P = lambda a, t: a.ctypes.data_as(t)   # noqa: E731  (arrays stay referenced for the solve)
    return _run(lambda L, n, a, r: L.brisk_solve_data(
        int(c.size), int(bs.size), P(bs, pi), P(c, pd), nnz, P(arrs[0], pi), P(arrs[1], pi),
        P(arrs[2], pi), P(arrs[3], pi), P(v, pd), n, a, r), options, verbose)


def certify(problem, side, X=None, y=None, verbose=False, prec=None):
    """Rigorous check of a certificate (in libbrisk: -certify-x / -certify-y): side "p"
    with X (a list of blocks, n x n or LP vectors) proves the upper bound <C,X> on the optimal
    value of BRISK's (P), side "d" with y the lower bound b'y. problem: an SDPA file name or a
    dict of solve_sdpa's arguments (blocksizes, c, mat, blk, i, j, v; data in memory are exact).
    prec ("dd", "qd" or a number of digits): the check runs in that precision (plus guard
    digits); X or y may then hold decimal.Decimal, mpmath.mpf or strings (as hp_solution
    returns them), and `bound` is a decimal.Decimal with all the digits.
    Returns dict(certified, bound, reason)."""
    import os
    import tempfile
    side = str(side).lower()[:1]
    if side not in ("p", "d"):
        raise ValueError('side must be "p" or "d"')

    def txt(v):
        return repr(float(v)) if prec is None or isinstance(v, (float, np.floating)) else str(v)
    fd, tmp = tempfile.mkstemp(suffix=".cert")
    try:
        with os.fdopen(fd, "w") as f:
            if side == "p":
                if X is None:
                    raise ValueError("side p needs X")
                for k, B in enumerate(X):
                    B = np.asarray(B, dtype=float if prec is None else object)
                    if B.ndim == 1:
                        for i in range(B.shape[0]):
                            if B[i] != 0:
                                f.write(f"{k + 1} {i + 1} {i + 1} {txt(B[i])}\n")
                    else:
                        iu, ju = np.triu_indices(B.shape[0])
                        for i, j in zip(iu, ju):
                            if B[i, j] != 0:
                                f.write(f"{k + 1} {i + 1} {j + 1} {txt(B[i, j])}\n")
            else:
                if y is None:
                    raise ValueError("side d needs y")
                for v in np.asarray(y, dtype=float if prec is None else object):
                    f.write(f"{txt(v)}\n")
        opt = ["-certify-x" if side == "p" else "-certify-y", tmp]
        if prec is not None:
            opt += ["-prec", str(prec)]
        if isinstance(problem, dict):
            r = solve_sdpa(**problem, options=opt, verbose=verbose)
        else:
            r = solve_file(str(problem), options=opt, verbose=verbose)
    finally:
        os.remove(tmp)
    b = r.bound or {}
    bound = b.get("rigorous")
    if prec is not None and b.get("certified"):
        h = hp_solution()
        if h is not None and h.get("bound") is not None:
            bound = h["bound"]
    return dict(certified=bool(b.get("certified")), bound=bound, reason=r.cause)


def write_sdpa(path, blocksizes, c, mat, blk, i, j, v):
    """Write the SDPA problem to a sparse SDPA file (for the command line or other solvers)."""
    with open(path, "w") as f:
        f.write(f"{len(c)}\n{len(blocksizes)}\n{' '.join(str(int(b)) for b in blocksizes)}\n")
        f.write(" ".join(repr(float(x)) for x in c) + "\n")
        for q in range(len(v)):
            ii, jj = int(i[q]), int(j[q])
            if ii > jj:
                ii, jj = jj, ii
            f.write(f"{int(mat[q])} {int(blk[q])} {ii} {jj} {float(v[q])!r}\n")


def hp_solution(convert=None):
    """All the digits of the last high-precision solve (option prec = "dd", "qd" or a number of
    digits). The Result of that solve holds the solution rounded to doubles; this returns a
    dict with digits, blocksizes, pobj (<C,X>), dobj (b'y), y, X and Z (one array per block,
    n x n or n), each value converted from its decimal text by `convert` (default
    decimal.Decimal; e.g. mpmath.mpf, fractions.Fraction or str). With the option
    bound = "p" or "d" and a certificate, `bound` is the rigorous bound on the optimal value
    of (P) (an upper bound for p, a lower bound for d; rounded outward) and the certificate
    is the X or the y returned, with all the digits of its check; otherwise `bound` is None.
    None when the last solve was not a high-precision one."""
    L = load_library()
    if not hasattr(L, "brisk_hp_text") or L.brisk_hp_digits() <= 0:
        return None
    if convert is None:
        import decimal
        convert = decimal.Decimal

    def get(what, block=0):
        n = -L.brisk_hp_text(what, block, None, 0)
        if n <= 0:
            return []
        buf = ctypes.create_string_buffer(n)
        L.brisk_hp_text(what, block, buf, n)
        return [convert(t) for t in buf.value.decode().split()]
    bs = [L.brisk_hp_blocksize(k) for k in range(L.brisk_hp_nblk())]
    first = lambda v: v[0] if v else None
    out = dict(digits=L.brisk_hp_digits(), blocksizes=bs, pobj=first(get(0)), dobj=first(get(1)),
               y=np.array(get(2), dtype=object), X=[], Z=[], bound=first(get(5)))
    for k, n in enumerate(bs):
        for what, key in ((3, "X"), (4, "Z")):
            v = np.array(get(what, k), dtype=object)
            out[key].append(v if n < 0 else v.reshape((n, n)))
    return out
