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
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

STATUS = {0: "OPTIMAL", 1: "PRIMAL INFEASIBLE", 2: "DUAL INFEASIBLE", 3: "ITERATION LIMIT",
          4: "NUMERICAL DIFFICULTIES", 5: "REDUCED ACCURACY", 6: "TIME LIMIT", -1: "NOT SOLVED"}

_lib = None
_lib_path = None
_lock = threading.Lock()          # libbrisk is not reentrant: one solve at a time per process
_PRINT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
_NOPRINT = _PRINT(0)      # (28 October 2026: one null callback for every quiet solve; a new one cost about 1 us a call)


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
    global _caller_ok
    _caller_ok = None                       # (the version is read again for a library loaded anew)
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
    if hasattr(L, "brisk_result_bound_of"):  # per side (-bound p|d|pd) with the certificates
        L.brisk_result_bound_of.argtypes, L.brisk_result_bound_of.restype = [P, I, pd], I
        L.brisk_result_bound_src.argtypes, L.brisk_result_bound_src.restype = [P, I], ctypes.c_char_p
        L.brisk_result_bound_X.argtypes, L.brisk_result_bound_X.restype = [P, I], pd
        L.brisk_result_bound_y.argtypes, L.brisk_result_bound_y.restype = [P], pd
        L.brisk_result_bound_sx.argtypes, L.brisk_result_bound_sx.restype = [P], pd
    L.brisk_result_resolves.argtypes, L.brisk_result_resolves.restype = [P, pd], I
    L.brisk_result_cause.argtypes, L.brisk_result_cause.restype = [P], ctypes.c_char_p
    if hasattr(L, "brisk_hp_text"):          # high precision: all the digits as text
        L.brisk_hp_digits.argtypes, L.brisk_hp_digits.restype = [], I
        L.brisk_hp_nblk.argtypes, L.brisk_hp_nblk.restype = [], I
        L.brisk_hp_blocksize.argtypes, L.brisk_hp_blocksize.restype = [I], I
        L.brisk_hp_count.argtypes, L.brisk_hp_count.restype = [I, I], ctypes.c_longlong
        L.brisk_hp_text.argtypes, L.brisk_hp_text.restype = [I, I, ctypes.c_char_p, ctypes.c_longlong], ctypes.c_longlong
        if hasattr(L, "brisk_hp_cert_digits"):
            L.brisk_hp_cert_digits.argtypes, L.brisk_hp_cert_digits.restype = [], I
    if hasattr(L, "brisk_solve_sedumi"):     # problems in SeDuMi format
        L.brisk_solve_sedumi.argtypes = [I, I, pi, pi, pd, pd, pd, I, I, I, pi, I, pi, I, pi, I, ppc, P]
        L.brisk_solve_sedumi.restype = I
        L.brisk_result_sn.argtypes, L.brisk_result_sn.restype = [P], I
        L.brisk_result_sx.argtypes, L.brisk_result_sx.restype = [P], pd
        L.brisk_result_sz.argtypes, L.brisk_result_sz.restype = [P], pd
    if hasattr(L, "brisk_solve_qp"):         # convex QPs (and QCQPs through files)
        L.brisk_solve_qp.argtypes = [I, I, pi, pi, pd, pd, D, pi, pi, pd, pd, pd, pd, pd, I, ppc, P]
        L.brisk_solve_qp.restype = I
        # (23 October 2026) the same function with the arrays as addresses (numpy's ctypes.data, checked by _qp_data):
        # the pointer casts cost 29 microseconds a call, more than the active-set solve of a QP of 16 variables
        if hasattr(L, "brisk_qp_lower"):     # (23 October 2026) H's lower triangle and the symmetry check in C
            L.brisk_qp_lower.argtypes = [I, P, P, P, P, P, P]
            L.brisk_qp_lower.restype = ctypes.c_long
        if hasattr(L, "brisk_solve_qp_lean"):      # (28 October 2026) the lean route for small QPs (_qp_lean)
            L.brisk_solve_qp_lean.argtypes = [I, I, P, P, P, P, D, P, P, P, P, P, P, P, I, ppc, P, P, P, P]
            L.brisk_solve_qp_lean.restype = I
        L.brisk_solve_qp_addr = L["brisk_solve_qp"]
        L.brisk_solve_qp_addr.argtypes = [I, I, P, P, P, P, D, P, P, P, P, P, P, P, I, ppc, P]
        L.brisk_solve_qp_addr.restype = I
    if hasattr(L, "brisk_qp_create"):
        L.brisk_qp_create.argtypes = [I, I, pi, pi, pd, pd, D, pi, pi, pd, pd, pd, pd, pd, I, ppc]
        L.brisk_qp_create.restype = ctypes.c_void_p
        for f in ("brisk_qp_set_var_bounds", "brisk_qp_set_row_bounds"):
            getattr(L, f).argtypes = [ctypes.c_void_p, I, D, D]; getattr(L, f).restype = I
        L.brisk_qp_set_cost.argtypes = [ctypes.c_void_p, P]; L.brisk_qp_set_cost.restype = I
        L.brisk_qp_resolve.argtypes = [ctypes.c_void_p, P]; L.brisk_qp_resolve.restype = I
        if hasattr(L, "brisk_qp_set_bounds"):        # (28 October 2026) every bound at once, the re-solve into arrays
            L.brisk_qp_set_bounds.argtypes = [ctypes.c_void_p, P, P, P, P]; L.brisk_qp_set_bounds.restype = I
            L.brisk_qp_resolve_lean.argtypes = [ctypes.c_void_p, P, P, P, P]; L.brisk_qp_resolve_lean.restype = I
        L.brisk_qp_destroy.argtypes = [ctypes.c_void_p]; L.brisk_qp_destroy.restype = None
    if hasattr(L, "brisk_socp_create"):      # the SOCP model (warm starts for branch and bound)
        L.brisk_socp_create.argtypes = [I, I, pi, pi, pd, pd, pd, I, I, I, pi, I, pi, I, ppc]
        L.brisk_socp_create.restype = ctypes.c_void_p
        L.brisk_socp_set_var_bounds.argtypes = [ctypes.c_void_p, I, D, D]; L.brisk_socp_set_var_bounds.restype = I
        L.brisk_socp_set_cost.argtypes = [ctypes.c_void_p, pd]; L.brisk_socp_set_cost.restype = I
        L.brisk_socp_set_cutoff.argtypes = [ctypes.c_void_p, D]; L.brisk_socp_set_cutoff.restype = None
        L.brisk_socp_resolve.argtypes = [ctypes.c_void_p, P]; L.brisk_socp_resolve.restype = I
        L.brisk_socp_destroy.argtypes = [ctypes.c_void_p]; L.brisk_socp_destroy.restype = None
        if hasattr(L, "brisk_socp_save_point"):
            L.brisk_socp_save_point.argtypes = [ctypes.c_void_p]; L.brisk_socp_save_point.restype = ctypes.c_void_p
            L.brisk_socp_restore_point.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; L.brisk_socp_restore_point.restype = ctypes.c_int
            L.brisk_socp_point_free.argtypes = [ctypes.c_void_p]; L.brisk_socp_point_free.restype = None
    L.brisk_interrupt.argtypes, L.brisk_interrupt.restype = [], None
    L.brisk_interrupted.argtypes, L.brisk_interrupted.restype = [], I
    # (24 October 2026, ISSUES 272) the interrupt flag cleared here before each solve is handed over, not by the library
    # when the solve starts: a Ctrl-C between the two was lost and the solve ran to its end
    if hasattr(L, "brisk_set_interrupt_owner"):
        L.brisk_clear_interrupt.argtypes, L.brisk_clear_interrupt.restype = [], None
        L.brisk_set_interrupt_owner.argtypes, L.brisk_set_interrupt_owner.restype = [I], None
        L.brisk_set_interrupt_owner(1)
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
    bound: Optional[dict] = None     # -bound p|d: side, value, resid, lammin, valid, certified (the first side asked for)
    bounds: Optional[dict] = None    # per side asked for ("p", "d"): kind, value, resid, lammin, valid, certified,
                                     # rigorous (a bound only when certified), src, and the certificate X (p) or y (d)
    n_resolves: int = 0              # re-solves run (method retry, fallbacks) and their time
    t_resolves: float = 0.0
    cause: str = ""                  # likely cause when the tolerance is missed

    @property
    def x(self):
        """The SDPA primal variable x = -y."""
        return None if self.y is None else -self.y

    def __repr__(self):
        return _result_text("Result", self)


def _result_text(cls, r):
    """The printed result (4 November 2026, the owner on the printed results: "last lines weird"): two short lines; the
    objectives as the log's Summary prints them (the same digits for both); none for an infeasible problem, where they
    mean nothing."""
    head = f"brisk.{cls}: {r.status_str}, {r.iterations} iterations, {r.time:.2f} s"
    if "INFEASIBLE" in r.status_str:
        return head
    return (head + f"\n  primal objective {r.pobj:.10e}, dual objective {r.dobj:.10e}, "
            f"DIMACS error {np.max(np.abs(r.dimacs)):.1e}")

def _ints(x, what):
    """Integers from x, checked before they cross into C (31 October 2026, the owner's external SDP benchmark, audit A3:
    np.intc truncated 3.9 to 3 and 1.9 to 1, and a different problem was solved): whole numbers within int32, else
    ValueError. Booleans and complex numbers are refused."""
    a = np.atleast_1d(np.asarray(x)).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=np.intc)
    if a.dtype == bool or np.iscomplexobj(a) or a.dtype == object:
        raise ValueError("%s: integers expected" % what)
    if not np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(a != np.floor(a)):
            raise ValueError("%s: integers expected (a value is not a whole number)" % what)
    lo, hi = np.iinfo(np.intc).min, np.iinfo(np.intc).max
    if np.any(a > hi) or np.any(a < lo):
        raise ValueError("%s: a value is out of range" % what)
    return np.ascontiguousarray(a, dtype=np.intc)


def _finite(x, what):
    a = np.ascontiguousarray(x, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s: a value is not finite (NaN or infinity)" % what)
    return a


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
    cannot be interrupted. A small QP (fewer than 20,000 entries, rows and variables together)
    always runs in the calling thread: it takes milliseconds."""
    global _interruptible
    _interruptible = bool(flag)


def interrupt():
    """Stop the running solve (from any thread or a signal handler) as if its time limit
    were reached now."""
    load_library().brisk_interrupt()


def _caller_args(tag):
    """The option pair that tells libbrisk which interface and command called it, so that the
    advice at the end of the log (more accuracy, a guaranteed bound) is written in that
    command's syntax. Libraries before 1.3.2 do not know it and are not given it."""
    global _caller_ok
    if _caller_ok is None:                  # (the library's version read once: the call and the match cost 5 us a solve)
        import re
        m = re.match(r"(\d+)\.(\d+)\.?(\d*)", version())    # (a suffix such as "-experimental" is allowed)
        _caller_ok = bool(m) and (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) >= (1, 3, 2)
    return ["-caller", tag] if tag and _caller_ok else []


_caller_ok = None


def _run(call, options, verbose, collect=None, caller=None, small=False):
    global _worker
    L = load_library()
    av = options_to_argv(options) + _caller_args(caller)
    arr = (ctypes.c_char_p * max(len(av), 1))(*[a.encode() for a in av])

    def cb(s, is_err):
        (sys.stderr if is_err else sys.stdout).write(s.decode(errors="replace"))
        return 0
    cbp = _PRINT(cb) if verbose else _NOPRINT
    r = L.brisk_result_new()
    if not r:
        raise MemoryError("brisk_result_new failed")
    state = {}
    done = threading.Event()
    guard = threading.Lock()
    # (31 October 2026, the owner's SDP benchmark: a fixed sleep of 50 ms between the tests of the flag added up to 50 ms
    # to the time of every solve handed to the worker thread) the worker writes one byte to a pipe when it ends and
    # the main thread waits on it with select, which a Ctrl-C interrupts without holding any lock (the reason for the
    # sleep instead of done.wait, ISSUES 444); not on Windows, where select does not take pipes
    wake = os.pipe() if os.name != "nt" else None

    def cleanup():
        # (24 October 2026) the lock released however the calls before it end: a Ctrl-C that arrived while the main
        # thread ran them left the lock held, and the next solve waited for it forever (test_ctrl_c hung once, ISSUES 272)
        try:
            L.brisk_set_output(0, _NOPRINT)
            L.brisk_result_delete(r)
        finally:
            _lock.release()

    def job():
        try:
            L.brisk_set_output(2 if verbose else 1, cbp if verbose else _NOPRINT)
            state["rc"] = call(L, len(av), arr, r)
            state["interrupted"] = bool(L.brisk_interrupted())
        except BaseException as e:          # noqa: B036 (ctypes errors, re-raised by the caller)
            state["exc"] = e
        finally:
            with guard:
                done.set()
                if wake is not None:
                    try:
                        os.write(wake[1], b"x")
                    except OSError:
                        pass
                    os.close(wake[1])
                if state.get("abandoned"):
                    cleanup()

    _lock.acquire()
    if hasattr(L, "brisk_clear_interrupt"):
        L.brisk_clear_interrupt()          # (before the hand-off: an interrupt from here on stops this solve)
    try:
        # (small problems run in the calling thread: they finish within milliseconds, and the hand-off to the worker
        # thread cost about as much as the solve: 0.3 ms of 0.7 ms on a QP of 16 variables)
        if _interruptible and not small and threading.current_thread() is threading.main_thread():
            if _worker is None:
                _worker = _Worker()
            _worker.jobs.put(job)
            stopping = False
            while True:
                try:
                    # (a sleep and a test of the flag, not done.wait: a Ctrl-C landing inside Event.wait
                    # just after it took the event's lock can leave the lock held, and the worker's
                    # done.set() then waits forever; the solve also clears the stop flag when it starts,
                    # so after a first Ctrl-C the interrupt is sent again until it returns; ISSUES 444)
                    while not done.is_set():
                        if stopping:
                            L.brisk_interrupt()
                        if wake is not None:
                            select.select([wake[0]], [], [], 0.05)
                        else:
                            time.sleep(0.05)
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
        if wake is not None:
            os.close(wake[0])       # (an abandoned worker's write then fails quietly and closes its end)
        if not state.get("abandoned"):
            cleanup()
        raise
    if wake is not None:
        os.close(wake[0])
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


def _bounds(L, r, blocks=None, n=0, m=0):
    """The bound of each side asked for (-bound p|d|pd), with its certificate: X (a list of
    blocks; x for a problem in SeDuMi format) for p, y for d. A value is a guaranteed bound
    only when "certified" is True (then "rigorous" is that bound)."""
    if not hasattr(L, "brisk_result_bound_of"):
        return None
    out = {}
    bo = (ctypes.c_double * 6)()
    for side, name in ((1, "p"), (2, "d")):
        st = L.brisk_result_bound_of(r, side, bo)
        if st == 0:
            continue
        d = dict(kind="upper" if side == 1 else "lower", have=st == 2, value=bo[0], resid=bo[1], lammin=bo[2],
                 valid=int(bo[3]), certified=bool(bo[4]), rigorous=bo[5],
                 src=(L.brisk_result_bound_src(r, side) or b"").decode())
        if side == 1:
            if blocks is not None:
                d["X"] = blocks(L.brisk_result_bound_X)
            ps = L.brisk_result_bound_sx(r)
            if ps and n > 0:
                d["x"] = np.ctypeslib.as_array(ps, shape=(n,)).copy()
        else:
            py = L.brisk_result_bound_y(r)
            if py and m > 0:
                d["y"] = np.ctypeslib.as_array(py, shape=(m,)).copy()
        out[name] = d
    return out or None


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
    res.bounds = _bounds(L, r, blocks=blocks, m=m)
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
    bounds: Optional[dict] = None    # per side asked for (-bound p|d|pd), as Result.bounds; certificate x (p) or y (d)

    def __repr__(self):
        return _result_text("ConeResult", self)


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
    res.bounds = _bounds(L, r, n=n, m=m)
    return res


class QpResult:
    """The result of solve_qp (or of solve_file on a .qps file): status (0 optimal, 1 primal infeasible, 2 dual
    infeasible, 3 the iteration limit, 4 numerical difficulties, 5 reduced accuracy, 6 the time limit, -1 not solved: the
    input refused, e.g. a QP that is not convex: status_str NOT SOLVED, the reason on stderr), status_str,
    obj (the objective value), x, y (the rows' multipliers), z (the bounds' multipliers), with H x + q - A'y - z = 0,
    y_i > 0 where the row is at l_i and < 0 at u_i, z_j > 0 at lb_j and < 0 at ub_j; errors: the relative primal, dual
    and gap errors on the problem as given; iterations, time."""
    def __init__(self, **kw):
        self.x = self.y = self.z = None
        self.__dict__.update(kw)

    def __repr__(self):
        return f"brisk.QpResult(status {self.status_str}, objective {self.obj:.10g}, {self.iterations} iterations)"


_ERR6 = (ctypes.c_double * 6)()       # (read under _lock: one solve at a time)


def _collect_qp(L, r, rc):
    err = _ERR6
    L.brisk_result_dimacs(r, err)
    status = L.brisk_result_status(r)
    if rc < 0 or (status == -1 and rc not in (0, 1)):
        raise RuntimeError(f"BRISK did not solve the problem (exit code {rc})")
    res = QpResult(status=status, status_str=L.brisk_result_status_str(r).decode(), exit_code=rc,
                   iterations=L.brisk_result_iterations(r), obj=L.brisk_result_pobj(r), time=L.brisk_result_time(r),
                   errors=np.array([err[0], err[2], err[4]]))
    m, n = L.brisk_result_m(r), L.brisk_result_sn(r)
    for name, ptr, k in (("x", L.brisk_result_sx(r), n), ("y", L.brisk_result_y(r), m), ("z", L.brisk_result_sz(r), n)):
        if ptr and k > 0:
            v = np.empty(k)                       # (a copy by memmove: as_array builds a ctypes array type, 4 us each)
            ctypes.memmove(_addr(v), ptr, 8 * k)
            setattr(res, name, v)
    if res.y is None:
        res.y = np.zeros(0)
    return res


_SMALL_NZ = 20000      # (a QP with fewer entries and rows: solved in the calling thread, see _run)


def _csc_canonical(M, shape):
    """M as (indptr, indices, data) when it is already a scipy CSC matrix of that shape in canonical form (sorted
    indices, no duplicates), else None: the numpy path of _qp_data, without the scipy conversions that cost more
    than the solve of a small QP"""
    import scipy.sparse as sp
    if not (sp.issparse(M) and M.format == "csc" and M.shape == shape and M.has_canonical_format):
        return None
    return (np.ascontiguousarray(M.indptr, dtype=np.intc), np.ascontiguousarray(M.indices, dtype=np.intc),
            np.ascontiguousarray(M.data, dtype=np.float64))


def _lower_of_canonical(Hp, Hi, Hx, n):
    """the lower triangle of a canonical CSC matrix that is symmetric, lower or upper triangular, else None (the caller
    raises); the symmetry is checked entry by entry against the transpose (the entries sorted stably by row: in CSC
    order the columns of a row come ascending, so this is the transpose by columns). (23 October 2026: one stable sort
    instead of two sorts of the two triangles, 41 -> 15 microseconds on a QP of 16 variables)"""
    cnt = np.diff(Hp)
    col = np.repeat(np.arange(n, dtype=np.intc), cnt)
    up = Hi < col
    if not up.any():
        return Hp, Hi, Hx
    o = np.argsort(Hi, kind="stable")
    rcnt = np.bincount(Hi, minlength=n)
    if not (Hi > col).any():                               # upper triangular: its transpose
        Hp2 = np.zeros(n + 1, dtype=np.intc); np.cumsum(rcnt, out=Hp2[1:])
        return Hp2, np.ascontiguousarray(col[o], dtype=np.intc), np.ascontiguousarray(Hx[o], dtype=np.float64)
    if not (np.array_equal(rcnt, cnt) and np.array_equal(col[o], Hi)):
        return None
    if np.abs(Hx[o] - Hx).max(initial=0.0) > 1e-14 * max(1.0, np.abs(Hx).max(initial=0.0)):
        return None
    keep = ~up
    Hp2 = np.zeros(n + 1, dtype=np.intc); np.cumsum(np.bincount(col[keep], minlength=n), out=Hp2[1:])
    return Hp2, np.ascontiguousarray(Hi[keep]), np.ascontiguousarray(Hx[keep])


def _qp_data(P, q, A, l, u, lb, ub, c0):
    """solve_qp's data as the C arrays (H's lower triangle and A by columns)"""
    import scipy.sparse as sp
    if q is None:
        raise ValueError("q (the linear cost) is needed; its length is the number of variables")
    q = np.ascontiguousarray(np.asarray(q, dtype=np.float64).ravel())
    n = q.size
    fastP = _csc_canonical(P, (n, n)) if P is not None else None
    fastA = _csc_canonical(A, (A.shape[0] if A is not None and hasattr(A, "shape") else 0, n)) if A is not None else None
    if fastP is not None and fastA is not None:
        L0 = load_library()
        if hasattr(L0, "brisk_qp_lower"):
            Hp0, Hi0, Hx0 = fastP; nz = int(Hp0[-1])
            Lp, Li, Lx = np.empty(n + 1, dtype=np.intc), np.empty(max(nz, 1), dtype=np.intc), np.empty(max(nz, 1))
            k = L0.brisk_qp_lower(n, _addr(Hp0), _addr(Hi0), _addr(Hx0), _addr(Lp), _addr(Li), _addr(Lx))
            HL = None if k < 0 else (Lp, Li[:k], Lx[:k])
        else:
            HL = _lower_of_canonical(*fastP, n)
        if HL is None:
            raise ValueError("P is neither symmetric nor triangular")
        Hp, Hi, Hx = HL; Ap, Ai, Ax = fastA; m = A.shape[0]
        vec = lambda v, k, d: np.ascontiguousarray(np.full(k, d) if v is None else np.asarray(v, dtype=np.float64).ravel())   # noqa: E731
        l, u, lb, ub = vec(l, m, -np.inf), vec(u, m, np.inf), vec(lb, n, -np.inf), vec(ub, n, np.inf)
        for name, v, k in (("l", l, m), ("u", u, m), ("lb", lb, n), ("ub", ub, n)):
            if v.size != k:
                raise ValueError(f"{name} has {v.size} entries, expected {k}")
        return dict(n=n, m=m, Hp=Hp, Hi=Hi, Hx=Hx, q=q, c0=float(c0), Ap=Ap, Ai=Ai, Ax=Ax, l=l, u=u, lb=lb, ub=ub)
    if P is None:
        H = sp.csc_matrix((n, n))
    else:
        P = sp.csc_matrix(P, dtype=np.float64)
        if P.shape != (n, n):
            raise ValueError(f"P is {P.shape[0]} x {P.shape[1]}, q has {n} entries")
        if abs(P - P.T).max() <= 1e-14 * max(1.0, abs(P).max()):
            H = sp.tril(P, format="csc")
        elif sp.triu(P, 1).nnz == 0:
            H = P
        elif sp.tril(P, -1).nnz == 0:
            H = sp.csc_matrix(P.T)
        else:
            raise ValueError("P is neither symmetric nor triangular")
    H = sp.csc_matrix(H); H.sum_duplicates(); H.sort_indices()
    if A is None:
        A = sp.csc_matrix((0, n))
    A = sp.csc_matrix(A, dtype=np.float64)
    if A.shape[1] != n:
        raise ValueError(f"A has {A.shape[1]} columns, q has {n} entries")
    A.sum_duplicates(); A.sort_indices()
    m = A.shape[0]
    vec = lambda v, k, d: np.ascontiguousarray(np.full(k, d) if v is None else np.asarray(v, dtype=np.float64).ravel())   # noqa: E731
    l, u, lb, ub = vec(l, m, -np.inf), vec(u, m, np.inf), vec(lb, n, -np.inf), vec(ub, n, np.inf)
    for name, v, k in (("l", l, m), ("u", u, m), ("lb", lb, n), ("ub", ub, n)):
        if v.size != k:
            raise ValueError(f"{name} has {v.size} entries, expected {k}")
    Hp, Hi, Hx = (np.ascontiguousarray(H.indptr, dtype=np.intc), np.ascontiguousarray(H.indices, dtype=np.intc),
                  np.ascontiguousarray(H.data, dtype=np.float64))
    Ap, Ai, Ax = (np.ascontiguousarray(A.indptr, dtype=np.intc), np.ascontiguousarray(A.indices, dtype=np.intc),
                  np.ascontiguousarray(A.data, dtype=np.float64))
    return dict(n=n, m=m, Hp=Hp, Hi=Hi, Hx=Hx, q=q, c0=float(c0), Ap=Ap, Ai=Ai, Ax=Ax, l=l, u=u, lb=lb, ub=ub)


def solve_qp(P=None, q=None, A=None, l=None, u=None, lb=None, ub=None, c0=0.0, options=None, verbose=True,
             _caller="python:solve_qp"):
    """Solve the convex QP

        min 1/2 x'Px + q'x + c0  s.t.  l <= A x <= u,  lb <= x <= ub

    by BRISK's QP solver: its dual active-set method for a small QP with P positive definite (the dense work at most
    4e6 flops), its interior-point method otherwise (options=["-qpas"], ["-qpipm"]: that method; other options as on the
    command line for a QP file: -tol, -maxit, -timelimit, -threads, -v). P: n x n, symmetric (both triangles), or
    one triangle of it (upper or lower); None: an LP. A: m x n (scipy sparse or dense); l, u, lb, ub: arrays, None
    or +-inf for an absent bound (an equality row has l = u). Returns a QpResult."""
    L0 = load_library()
    if not verbose and hasattr(L0, "brisk_solve_qp_lean"):
        r = _qp_lean(L0, P, q, A, l, u, lb, ub, c0, options, _caller)
        if r is not None:
            return r
    d = _qp_data(P, q, A, l, u, lb, ub, c0)
    if not hasattr(L0, "brisk_solve_qp"):
        raise RuntimeError("this libbrisk has no QP solver (brisk_solve_qp): rebuild it")
    small = d["Hp"][-1] + d["Ap"][-1] + d["n"] + d["m"] < _SMALL_NZ
    if small and not verbose and hasattr(L0, "brisk_solve_qp_addr"):
        return _qp_small(L0, d, options_to_argv(options) + _caller_args(_caller))
    return _run(lambda L, k, a, r: L.brisk_solve_qp_addr(d["n"], d["m"], *_qp_addrs(d), k, a, r), options, verbose, _collect_qp,
                caller=_caller, small=small)


_ARGV = {}                # (the options' C arrays, by their strings: built once)
_QP_STATUS = {0: "OPTIMAL", 1: "PRIMAL INFEASIBLE", 2: "DUAL INFEASIBLE (unbounded)", 3: "ITERATION LIMIT", 4: "NUMERICAL DIFFICULTIES",
              5: "SOLVED TO REDUCED ACCURACY", 6: "TIME LIMIT", -1: "INPUT ERROR"}
_F8, _I4 = np.dtype(np.float64), np.dtype(np.intc)


_CSC_TYPES = None


def _csc_parts(M, nr, nc):
    """a scipy CSC matrix's arrays as C arrays (int32 indices, float64 values), or None for anything else (the type checked
    against scipy's CSC classes, cached: the format property is Python code, read on every call)"""
    global _CSC_TYPES
    if _CSC_TYPES is None:
        import scipy.sparse as sp
        _CSC_TYPES = tuple(t for t in (getattr(sp, "csc_matrix", None), getattr(sp, "csc_array", None)) if t is not None)
    if type(M) not in _CSC_TYPES or M.shape != (nr, nc):
        return None
    p, i, x = M.indptr, M.indices, M.data
    if p.dtype != _I4: p = p.astype(np.intc)
    if i.dtype != _I4: i = i.astype(np.intc)
    if x.dtype != _F8: x = x.astype(np.float64)
    return np.ascontiguousarray(p), np.ascontiguousarray(i), np.ascontiguousarray(x)


def _vec(v, k):
    if v is None:
        return None
    v = np.asarray(v)
    if v.dtype != _F8: v = v.astype(np.float64)
    v = np.ascontiguousarray(v.ravel()) if v.ndim != 1 else np.ascontiguousarray(v)
    return v if v.size == k else False


def _qp_lean(L, P, q, A, l, u, lb, ub, c0, options, caller):
    """(28 October 2026, the SOCP developer) solve_qp for scipy CSC data without output: the arrays handed to the library as
    they are (it checks their form, takes H's lower triangle and solves the QP without the general route: brisk_solve_qp_lean),
    x, y, z and the numbers of the result written into arrays made here. None when the data or the options need the general
    route (other matrix types, wrong sizes, options outside the QP's), which then gives its own messages. On the owner's
    independent QP benchmark, a QP of ten variables: 325 us a call through the general route (cold caches: the Python
    conversions and the whole command-line program around a 20 us solve)"""
    if q is None:
        return None
    q = np.ascontiguousarray(np.asarray(q, dtype=np.float64).ravel())
    n = q.shape[0]
    if A is None:
        return None
    m = A.shape[0] if hasattr(A, "shape") and len(A.shape) == 2 else -1
    pa = _csc_parts(A, m, n)
    pp = _csc_parts(P, n, n) if P is not None else (None, None, None)
    if pa is None or pp is None:
        return None
    if (pp[2].size if pp[2] is not None else 0) + pa[2].size + n + m >= _SMALL_NZ:     # (the general route above the small size)
        return None
    vs = [_vec(l, m), _vec(u, m), _vec(lb, n), _vec(ub, n)]
    if any(v is False for v in vs):
        return None
    # (the options' C array cached by the options themselves: a dict of plain values, a list or a string, with the caller)
    try:
        okey = (tuple(options.items()) if isinstance(options, dict) else tuple(options) if isinstance(options, list) else options, caller)
        arr = _ARGV.get(okey)
    except TypeError:
        okey, arr = None, None
    if arr is None:
        av = options_to_argv(options) + _caller_args(caller)
        arr = (ctypes.c_char_p * max(len(av), 1))(*[a.encode() for a in av])
        arr._n = len(av)
        if okey is not None and len(_ARGV) < 64:
            _ARGV[okey] = arr
    m1 = max(m, 1)
    buf = np.empty(2 * n + m1 + 9)          # (one allocation: x, y, z and the result's numbers)
    a = _addr(buf)
    with _lock:
        rc = L.brisk_solve_qp_lean(n, m, 0 if pp[0] is None else _addr(pp[0]), 0 if pp[1] is None else _addr(pp[1]),
                                   0 if pp[2] is None else _addr(pp[2]), _addr(q), float(c0), _addr(pa[0]), _addr(pa[1]),
                                   _addr(pa[2]), 0 if vs[0] is None else _addr(vs[0]), 0 if vs[1] is None else _addr(vs[1]),
                                   0 if vs[2] is None else _addr(vs[2]), 0 if vs[3] is None else _addr(vs[3]), arr._n, arr,
                                   a, a + 8 * n, a + 8 * (n + m1), a + 8 * (2 * n + m1))
    if rc in (-2, -3):
        return None
    info = buf[2 * n + m1:]
    st, it, obj, _, _, _, tm, _, intr = info.tolist()
    status = int(st)
    if rc < 0 or (status == -1 and rc not in (0, 1)):
        raise RuntimeError(f"BRISK did not solve the problem (exit code {rc})")
    res = QpResult.__new__(QpResult)          # (without the keyword constructor: see QpModel.solve)
    res.__dict__ = {"status": status, "status_str": "INTERRUPTED" if intr else _QP_STATUS.get(status, "UNKNOWN"), "exit_code": rc,
                    "iterations": int(it), "obj": obj, "time": tm, "errors": info[3:6].copy(), "x": buf[:n],
                    "y": buf[n:n + m] if m > 0 else buf[:0], "z": buf[n + m1:2 * n + m1], "interrupted": bool(intr)}
    return res


def _qp_small(L, d, av):
    """(28 October 2026, the SOCP developer) a small QP without output, solved in the calling thread with nothing around the
    call that its size does not need: _run's thread hand-off, callbacks and closures cost about four times the solve of a QP
    of ten variables (the owner's independent QP benchmark: 146 us a solve against DAQP's 73 us)"""
    key = tuple(av)
    arr = _ARGV.get(key)
    if arr is None:
        arr = (ctypes.c_char_p * max(len(av), 1))(*[a.encode() for a in av])
        if len(_ARGV) < 64:
            _ARGV[key] = arr
    _lock.acquire()
    try:
        L.brisk_clear_interrupt()
        L.brisk_set_output(1, _NOPRINT)
        r = L.brisk_result_new()
        if not r:
            raise MemoryError("brisk_result_new failed")
        try:
            rc = L.brisk_solve_qp_addr(d["n"], d["m"], *_qp_addrs(d), len(av), arr, r)
            res = _collect_qp(L, r, rc)
            res.interrupted = bool(L.brisk_interrupted())
            if res.interrupted:
                res.status_str = "INTERRUPTED"
            return res
        finally:
            L.brisk_set_output(0, _NOPRINT)
            L.brisk_result_delete(r)
    finally:
        _lock.release()


def _addr(a):
    """a contiguous array's address (28 October 2026: ctypes.c_char.from_buffer, 0.4 us, against 1.4 us for .ctypes.data;
    a read-only or empty array by .ctypes.data)"""
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(a))
    except (TypeError, ValueError, BufferError):
        return a.ctypes.data


def _qp_addrs(d):
    """the arrays' addresses (the arrays are contiguous and of the C types: _qp_data makes them so, and d keeps them)"""
    return (_addr(d["Hp"]), _addr(d["Hi"]), _addr(d["Hx"]), _addr(d["q"]), d["c0"], _addr(d["Ap"]),
            _addr(d["Ai"]), _addr(d["Ax"]), _addr(d["l"]), _addr(d["u"]), _addr(d["lb"]), _addr(d["ub"]))


def _qp_ptrs(d):
    pi, pd = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    return (d["Hp"].ctypes.data_as(pi), d["Hi"].ctypes.data_as(pi), d["Hx"].ctypes.data_as(pd), d["q"].ctypes.data_as(pd), d["c0"],
            d["Ap"].ctypes.data_as(pi), d["Ai"].ctypes.data_as(pi), d["Ax"].ctypes.data_as(pd), d["l"].ctypes.data_as(pd),
            d["u"].ctypes.data_as(pd), d["lb"].ctypes.data_as(pd), d["ub"].ctypes.data_as(pd))


class QpModel:
    """A QP kept for re-solves after changes of bounds, row bounds or the cost, each solve warm-started from the
    previous active set (BRISK's dual active-set method; active_set=False: the interior-point method, every solve cold).
    The data as solve_qp's. For branch and bound on a mixed-integer QP:

        mdl = brisk.QpModel(P, q, A, l, u, lb, ub)
        r = mdl.solve()                       # cold
        mdl.set_var_bounds(j, lo=2.0)         # a branch: x_j >= 2
        r = mdl.solve()                       # from the parent's active set
        mdl.set_var_bounds(j, lo=lb[j])       # back

    options: -tol, -maxit, -timelimit (as on the command line, a list or a dict). Results are QpResult."""

    def __init__(self, P=None, q=None, A=None, l=None, u=None, lb=None, ub=None, c0=0.0, active_set=True, options=None,
                 verbose=False):
        self._d = _qp_data(P, q, A, l, u, lb, ub, c0)
        for k in ("q", "l", "u", "lb", "ub"):
            self._d[k] = self._d[k].copy()          # (the caller's arrays are not changed by the setters)
        L = load_library()
        if not hasattr(L, "brisk_qp_create"):
            raise RuntimeError("this libbrisk has no QP solver object (brisk_qp_create): rebuild it")
        av = options_to_argv(options) + ["-qpas" if active_set else "-qpipm"] + ([] if verbose else ["-q"])
        arr = (ctypes.c_char_p * len(av))(*[a.encode() for a in av])
        self._h = L.brisk_qp_create(self._d["n"], self._d["m"], *_qp_ptrs(self._d), len(av), arr)
        if not self._h:
            raise ValueError("brisk_qp_create failed (an unknown option or bad data): " + " ".join(av))
        self.verbose = verbose
        self._bdirty = 0
        d = self._d          # (the setters' arrays and sizes as attributes: they run hundreds of times a re-solve)
        self._n, self._m, self._lb, self._ub, self._l, self._u = d["n"], d["m"], d["lb"], d["ub"], d["l"], d["u"]
        self._small = int(d["Hp"][-1]) + int(d["Ap"][-1]) + d["n"] + d["m"] < _SMALL_NZ
        self._lean = hasattr(L, "brisk_qp_resolve_lean")

    @property
    def n(self):
        return self._d["n"]

    @property
    def m(self):
        return self._d["m"]

    def _check(self, rc, what):
        if rc:
            raise IndexError(what)

    # (28 October 2026, the SOCP developer) the setters change the model's own arrays and mark them; the solve hands every
    # change to the library in one call (brisk_qp_set_bounds): a receding-horizon controller changes hundreds of bounds a
    # step, and a library call for each cost 2.5 ms of a 4 ms step (a condensed MPC QP of 120 variables, the owner's
    # independent QP benchmark). The solve of a small model writes into arrays made once (brisk_qp_resolve_lean)

    def set_var_bounds(self, j, lo=None, hi=None):
        """the bounds of variable j (None: keep that side; -inf/inf: none)"""
        if not 0 <= j < self._n:
            raise IndexError(f"variable {j}")
        if lo is not None: self._lb[j] = lo
        if hi is not None: self._ub[j] = hi
        self._bdirty |= 1

    def set_row_bounds(self, i, lo=None, hi=None):
        """the bounds of row i (None: keep that side)"""
        if not 0 <= i < self._m:
            raise IndexError(f"row {i}")
        if lo is not None: self._l[i] = lo
        if hi is not None: self._u[i] = hi
        self._bdirty |= 2

    def update(self, q=None, l=None, u=None, lb=None, ub=None):
        """new vectors for the next solve (None: kept): the cost q, the row bounds l, u, the variable bounds lb, ub (-inf/inf:
        none). One call for a receding-horizon controller's step, in place of a setter call per changed row; the setters
        and update may be mixed. The structure (H, A) is the model's."""
        d = self._d
        for key, v, size, flag in (("q", q, self._n, 4), ("l", l, self._m, 2), ("u", u, self._m, 2), ("lb", lb, self._n, 1), ("ub", ub, self._n, 1)):
            if v is None:
                continue
            a = np.asarray(v, dtype=np.float64).ravel()
            if a.size != size:
                raise ValueError(f"{key} has {a.size} entries, expected {size}")
            np.copyto(d[key], a)
            self._bdirty |= flag

    def set_cost(self, q):
        q = np.array(q, dtype=np.float64).ravel()           # (a copy: the caller may change its array before the solve)
        if q.size != self.n:
            raise ValueError(f"q has {q.size} entries, expected {self.n}")
        self._d["q"] = q
        self._bdirty |= 4

    def _flush(self, L):
        f, d, h = self._bdirty, self._d, self._h
        if not f:
            return
        if f & 4:
            self._check(L.brisk_qp_set_cost(h, _addr(d["q"])), "cost")
        if f & 3:
            if hasattr(L, "brisk_qp_set_bounds"):
                self._check(L.brisk_qp_set_bounds(h, _addr(d["lb"]) if f & 1 else None, _addr(d["ub"]) if f & 1 else None,
                                                  _addr(d["l"]) if f & 2 else None, _addr(d["u"]) if f & 2 else None), "bounds")
            else:
                for j in range(d["n"]) if f & 1 else ():
                    L.brisk_qp_set_var_bounds(h, j, float(d["lb"][j]), float(d["ub"][j]))
                for i in range(d["m"]) if f & 2 else ():
                    L.brisk_qp_set_row_bounds(h, i, float(d["l"][i]), float(d["u"][i]))
        self._bdirty = 0

    def solve(self):
        """the re-solve from the previous active set (the first: cold); returns a QpResult"""
        h = self._h
        d = self._d
        L = load_library()
        small = self._small
        if not self.verbose and (small or not _interruptible) and self._lean:
            n, m = self._n, self._m
            m1 = max(m, 1)
            buf = np.empty(2 * n + m1 + 9)          # (x, y, z and the result's numbers; one library call writes them)
            a = _addr(buf)
            with _lock:
                if self._bdirty:
                    self._flush(L)
                rc = L.brisk_qp_resolve_lean(h, a, a + 8 * n, a + 8 * (n + m1), a + 8 * (2 * n + m1))
            if rc < 0:
                raise RuntimeError(f"BRISK did not solve the problem (exit code {rc})")
            info = buf[2 * n + m1:]
            st, it, obj, _, _, _, tm, _, intr = info.tolist()
            status = int(st)
            # (the result made without QpResult.__init__'s keyword dictionary: 2 us of a 10 us re-solve of a QP of 20
            # variables)
            res = QpResult.__new__(QpResult)
            res.__dict__ = {"status": status, "status_str": "INTERRUPTED" if intr else _QP_STATUS.get(status, "UNKNOWN"),
                            "exit_code": rc, "iterations": int(it), "obj": obj, "time": tm, "errors": info[3:6].copy(),
                            "x": buf[:n], "y": buf[n:n + m] if m > 0 else buf[:0], "z": buf[n + m1:2 * n + m1],
                            "interrupted": bool(intr)}
            return res
        self._flush(L)
        return _run(lambda L, k, a, r: L.brisk_qp_resolve(h, r), None, self.verbose, _collect_qp, small=small)

    def close(self):
        if getattr(self, "_h", None):
            load_library().brisk_qp_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _is_mat(path):
    with open(path, "rb") as f:
        return f.read(6) == b"MATLAB"


def solve_file(path, options=None, verbose=True, _caller="python:solve_file"):
    """Solve a problem file as the command line (./brisk path options...): an SDPA file
    (.dat-s; returns a Result) or a MAT-file with a problem in SeDuMi format (A or At, b, c,
    K; returns a ConeResult) or a QPS file (.qps: a convex QP or QCQP; returns a QpResult)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such problem file: {path}")   # was "exit code 2: invalid problem data"
    p = os.fsencode(path)
    mat = str(path).endswith(".mat") and _is_mat(path)
    qps = str(path).lower().endswith(".qps")
    return _run(lambda L, n, a, r: L.brisk_solve_file(p, n, a, r), options, verbose,
                _collect_sedumi if mat else _collect_qp if qps else None, caller=_caller)


def _cone_sizes(get):
    """(31 October 2026; the audit's A4) readers of K's fields: a cone's size must be a finite integer (3.0 is 3; 3.9, a
    negative size, NaN or Inf are refused with ValueError, where int() had made 3.9 a cone of 3 and dropped a negative one);
    sizes 0 are left out, as before"""
    def ints(k):
        v = get(k)
        a = np.atleast_1d(np.asarray([] if v is None else v, dtype=np.float64)).ravel()
        if a.size and (not np.all(np.isfinite(a)) or np.any(a != np.floor(a)) or np.any(a < 0) or np.any(a > 2**31 - 1)):
            raise ValueError(f"K.{k}: cone sizes must be nonnegative integers, got {v!r}")
        return a.astype(np.int64)

    def lst(k):
        a = ints(k)
        return np.ascontiguousarray(a[a > 0], dtype=np.intc)

    def cnt(k):
        return int(ints(k).sum())
    return lst, cnt


def solve_sedumi(A, b, c, K, options=None, verbose=True, _caller="python:solve_sedumi"):
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

    lst, cnt = _cone_sizes(get)
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
        options, verbose, _collect_sedumi, caller=_caller)


def _collect_socpm(L, r, rc):
    """the result of an SOCP model's solve: the status code is the return value (9: stopped by the cutoff)"""
    err = (ctypes.c_double * 6)()
    L.brisk_result_dimacs(r, err)
    res = ConeResult(status=L.brisk_result_status(r), status_str=L.brisk_result_status_str(r).decode(), exit_code=rc,
                     iterations=L.brisk_result_iterations(r), pobj=L.brisk_result_pobj(r),
                     dobj=L.brisk_result_dobj(r), time=L.brisk_result_time(r), dimacs=np.array(err[:]))
    m, n = L.brisk_result_m(r), L.brisk_result_sn(r)
    for name, ptr, k in (("x", L.brisk_result_sx(r), n), ("y", L.brisk_result_y(r), m), ("z", L.brisk_result_sz(r), n)):
        if ptr and k > 0:
            setattr(res, name, np.ctypeslib.as_array(ptr, shape=(k,)).copy())
    return res


class _SocpPoint:
    """a saved warm-start point of an SocpModel (SocpModel.save_point)"""

    def __init__(self, p):
        self._p = p

    def __del__(self):
        try:
            if self._p:
                load_library().brisk_socp_point_free(self._p); self._p = None
        except Exception:
            pass


class SocpModel:
    """An SOCP in SeDuMi format (min c'x, A x = b, x in K; K as solve_sedumi's, without semidefinite blocks) kept for
    re-solves after bound changes, each solve after an optimal one warm-started from it. For branch and bound on a
    mixed-integer SOCP:

        mdl = brisk.SocpModel(A, b, c, K)
        r = mdl.solve()                        # cold
        mdl.set_var_bounds(j, lo=2.0)          # a branch: x_j >= 2 (any column; a bound is a row with a slack)
        mdl.set_cutoff(incumbent)              # stop once the node cannot beat the incumbent (status 9)
        r = mdl.solve()                        # warm, from the last optimal point
        mdl.set_var_bounds(j, lo=-np.inf)      # back
        pt = mdl.save_point(); ...; mdl.restore_point(pt)   # each child from its parent's point

    options: -tol, -maxit, -timelimit (as on the command line, a list or a dict). Results are ConeResult (x, y, z of the
    problem as given; y over A's rows)."""

    def __init__(self, A, b, c, K, options=None, verbose=False):
        import scipy.sparse as sp
        b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
        c = np.ascontiguousarray(np.asarray(c, dtype=np.float64).ravel())
        A = sp.csc_matrix(A)
        if A.shape != (b.size, c.size):
            raise ValueError(f"A is {A.shape[0]} x {A.shape[1]}, b has {b.size} entries, c {c.size}")
        A.sum_duplicates(); A.sort_indices()
        get = (lambda k: K.get(k)) if isinstance(K, dict) else (lambda k: getattr(K, k, None))

        lst, cnt = _cone_sizes(get)
        if get("s") is not None and np.size(get("s")):
            raise ValueError("SocpModel: semidefinite blocks are not supported")
        nf, nl, q, rr = cnt("f"), cnt("l"), lst("q"), lst("r")
        if nf + nl + int(q.sum()) + int(rr.sum()) != c.size:
            raise ValueError("K does not match the number of columns of A")
        self._a = dict(Ap=np.ascontiguousarray(A.indptr, dtype=np.intc), Ai=np.ascontiguousarray(A.indices, dtype=np.intc),
                       Ax=np.ascontiguousarray(A.data, dtype=np.float64), b=b, c=c, q=q, r=rr)
        self._lo = np.full(c.size, -np.inf); self._hi = np.full(c.size, np.inf)
        L = load_library()
        if not hasattr(L, "brisk_socp_create"):
            raise RuntimeError("this libbrisk has no SOCP model (brisk_socp_create): rebuild it")
        av = options_to_argv(options) + ([] if not verbose else ["-v"])
        arr = (ctypes.c_char_p * max(len(av), 1))(*[x.encode() for x in av])
        pi, pd = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
        a = self._a
        self._h = L.brisk_socp_create(int(b.size), int(c.size), a["Ap"].ctypes.data_as(pi), a["Ai"].ctypes.data_as(pi),
                                      a["Ax"].ctypes.data_as(pd), b.ctypes.data_as(pd), c.ctypes.data_as(pd), nf, nl,
                                      int(q.size), q.ctypes.data_as(pi), int(rr.size), rr.ctypes.data_as(pi), len(av), arr)
        if not self._h:
            raise ValueError("brisk_socp_create failed (an unknown option or bad data): " + " ".join(av))
        self.verbose = verbose
        self._nz = int(a["Ap"][-1]) + int(b.size) + int(c.size)

    @property
    def n(self):
        return self._a["c"].size

    @property
    def m(self):
        return self._a["b"].size

    def set_var_bounds(self, j, lo=None, hi=None):
        """the bounds of column j (None: keep that side; -inf/inf: none). A rejected call (a column out of range, NaN, or
        lo > hi) raises and changes nothing"""
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)) or not 0 <= int(j) < self.n:
            raise IndexError(f"column {j}: the columns are 0 .. {self.n - 1}")
        j = int(j)
        nlo = float(self._lo[j]) if lo is None else float(lo)
        nhi = float(self._hi[j]) if hi is None else float(hi)
        if np.isnan(nlo) or np.isnan(nhi) or nlo > nhi:
            raise ValueError(f"column {j}: bounds lo = {nlo}, hi = {nhi} (NaN or lo > hi)")
        if load_library().brisk_socp_set_var_bounds(self._h, j, nlo, nhi):
            raise IndexError(f"column {j} (or lo > hi)")
        self._lo[j] = nlo; self._hi[j] = nhi          # (stored only once the model has taken them)

    def set_cost(self, c):
        c = np.ascontiguousarray(np.asarray(c, dtype=np.float64).ravel())
        if c.size != self.n:
            raise ValueError(f"c has {c.size} entries, expected {self.n}")
        self._a["c"] = c
        load_library().brisk_socp_set_cost(self._h, c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def set_cutoff(self, value=np.inf):
        """stop a solve (status 9) once an estimate of its lower bound exceeds value (a branch and bound's incumbent); inf:
        off. The estimate, b'y - |c - A'y - z| max(1, |x|) at the current iterate, counts the iterate's size for that of a
        feasible point: it is a test, not a proof, so a node stopped by it is not certified to be worse than the incumbent"""
        load_library().brisk_socp_set_cutoff(self._h, float(value))

    def save_point(self):
        """the warm-start point the model keeps (that of the last optimal solve), as an opaque object for restore_point;
        None when there is none yet. In a depth-first branch and bound: saved at a node, put back before each child"""
        p = load_library().brisk_socp_save_point(self._h)
        return _SocpPoint(p) if p else None

    def restore_point(self, point):
        """the next solve starts from this saved point, which must have been saved by this model (save_point); a point of
        another model is refused with ValueError and the model's own point is kept"""
        if point is not None:
            rc = load_library().brisk_socp_restore_point(self._h, point._p)
            if rc == 1:
                raise ValueError("restore_point: the point was saved by another SocpModel (or one of another shape)")
            if rc != 0:
                raise MemoryError("restore_point: out of memory")

    def solve(self):
        """the re-solve, warm from the last optimal point (the first: cold); returns a ConeResult"""
        h = self._h
        return _run(lambda L, k, a, r: L.brisk_socp_resolve(h, r), None, self.verbose, _collect_socpm,
                    small=self._nz < _SMALL_NZ)

    def close(self):
        if getattr(self, "_h", None):
            load_library().brisk_socp_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def solve_sdpa(blocksizes, c, mat, blk, i, j, v, options=None, verbose=True, _caller="python:solve_sdpa"):
    """Solve the SDPA problem given by its numbers (see the module docstring): blocksizes
    (LP blocks negative), c (m), and the entries mat (0..m), blk (1..nblk), i, j (1-based
    within the block; i > j is swapped), v."""
    bs = _ints(blocksizes, "blocksizes")
    c = _finite(c, "c")
    arrs = [_ints(a, nm) for a, nm in ((mat, "mat"), (blk, "blk"), (i, "i"), (j, "j"))]
    v = _finite(v, "v")
    nnz = v.size
    if any(a.size != nnz for a in arrs):
        raise ValueError("mat, blk, i, j and v must have the same length")
    pi = ctypes.POINTER(ctypes.c_int)
    pd = ctypes.POINTER(ctypes.c_double)
    P = lambda a, t: a.ctypes.data_as(t)   # noqa: E731  (arrays stay referenced for the solve)
    return _run(lambda L, n, a, r: L.brisk_solve_data(
        int(c.size), int(bs.size), P(bs, pi), P(c, pd), nnz, P(arrs[0], pi), P(arrs[1], pi),
        P(arrs[2], pi), P(arrs[3], pi), P(v, pd), n, a, r), options, verbose, caller=_caller)


def certify(problem, side, X=None, y=None, verbose=False, prec=None, prior=True):
    """Rigorous check of a certificate (in libbrisk: -certify-x / -certify-y): side "p"
    with X (a list of blocks, n x n or LP vectors) proves the upper bound <C,X> on the optimal
    value of BRISK's (P), side "d" with y the lower bound b'y. problem: an SDPA file name or a
    dict of solve_sdpa's arguments (blocksizes, c, mat, blk, i, j, v; data in memory are exact).
    prec ("dd", "qd" or a number of digits): the check runs in that precision (plus guard
    digits); X or y may then hold decimal.Decimal, mpmath.mpf or strings (as hp_solution
    returns them), and `bound` is a decimal.Decimal with all the digits.
    prior: a certificate without a margin (the other side without an interior) is certified by
    a-priori bounds of the other side found from the problem (-boundprior; the bound then holds
    for the optimal value of that other side), unless prior=False.
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
        if not prior:
            opt += ["-boundprior", "0"]
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
    decimal.Decimal; e.g. mpmath.mpf, fractions.Fraction or str). The solution is the solve's
    own point. With the option bound = "p", "d" or "pd", `bounds` maps each certified side
    ("p", "d") to its rigorous bound on the optimal value of (P) (an upper bound for p, a lower
    bound for d; rounded outward) and its certificate (X for p, y for d) with all the digits of
    its check; `bound` is that of the first side (None without one).
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
               y=np.array(get(2), dtype=object), X=[], Z=[], bound=first(get(5)), bounds={})
    for k, n in enumerate(bs):
        for what, key in ((3, "X"), (4, "Z")):
            v = np.array(get(what, k), dtype=object)
            out[key].append(v if n < 0 else v.reshape((n, n)))
    if hasattr(L, "brisk_hp_cert_digits"):
        bp, bd = first(get(6)), first(get(7))
        if bp is not None:
            Xc = []
            for k, n in enumerate(bs):
                v = np.array(get(8, k), dtype=object)
                Xc.append(v if n < 0 else v.reshape((n, n)))
            out["bounds"]["p"] = dict(bound=bp, X=Xc)
        if bd is not None:
            out["bounds"]["d"] = dict(bound=bd, y=np.array(get(9), dtype=object))
    return out
