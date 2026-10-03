"""python -m pytest python/tests   (make pytest)

Needs libbrisk (built by pip install ./python, or make libbrisk for an editable install). The
CVXPY tests need cvxpy >= 1.9 and compare with Clarabel (installed with cvxpy)."""
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import brisk

# a small SDPA problem (the SDPA manual's example 1); SDPA optimum c'x = -41.9
# F0 = [[-11,0],[0,23]], F1 = [[10,4],[4,0]], F2 = [[0,0],[0,-8]], F3 = [[0,-8],[-8,-2]]
EX1 = dict(blocksizes=[2], c=[48.0, -8.0, 20.0], mat=[0, 0, 1, 1, 2, 3, 3], blk=[1] * 7,
           i=[1, 2, 1, 1, 2, 1, 2], j=[1, 2, 1, 2, 2, 2, 2],
           v=[-11.0, 23.0, 10.0, 4.0, -8.0, -8.0, -2.0])


def maxcut_sdpa(n, seed=0):
    """min sum x_i  s.t.  diag(x) - L/4 >= 0 (the dual of the max-cut relaxation)."""
    rng = np.random.default_rng(seed)
    W = np.triu(rng.random((n, n)) < 0.1, 1).astype(float)
    W = W + W.T
    L = np.diag(W.sum(1)) - W
    iu = np.triu_indices(n)
    v0 = L[iu] / 4
    nz = v0 != 0
    mat = np.concatenate((np.zeros(nz.sum(), int), np.arange(1, n + 1)))
    i = np.concatenate((iu[0][nz] + 1, np.arange(1, n + 1)))
    j = np.concatenate((iu[1][nz] + 1, np.arange(1, n + 1)))
    v = np.concatenate((v0[nz], np.ones(n)))
    return dict(blocksizes=[n], c=np.ones(n), mat=mat, blk=np.ones(mat.size, int), i=i, j=j, v=v)


# a first-order solve that runs until stopped
LONG = {"threads": 1, "fom": 1, "tol": 1e-14, "fomtol": 1e-14, "fommaxit": 10**8, "fomssn": 0}


def test_version():
    assert brisk.version() == "1.1"
    assert os.path.exists(brisk.library_path())


def test_solve_sdpa_and_file(tmp_path):
    r = brisk.solve_sdpa(**EX1, verbose=False)
    assert r.status == 0 and r.status_str == "OPTIMAL" and not r.interrupted
    assert np.isclose(np.dot(EX1["c"], r.x), -41.9, atol=1e-6) and np.isclose(r.dobj, 41.9)
    assert r.X[0].shape == (2, 2) and np.max(np.abs(r.dimacs)) < 1e-7
    f = tmp_path / "ex1.dat-s"
    brisk.write_sdpa(f, **EX1)
    r2 = brisk.solve_file(str(f), {"acc": "high"}, verbose=False)
    assert r2.status == 0 and np.allclose(r2.y, r.y, atol=1e-6)


def test_output(capsys):
    brisk.solve_sdpa(**EX1, verbose=True)
    assert "OPTIMAL" in capsys.readouterr().out
    brisk.solve_sdpa(**EX1, verbose=False)
    assert capsys.readouterr().out == ""


def test_bad_option():
    with pytest.raises(RuntimeError):
        brisk.solve_sdpa(**EX1, options=["-nosuchoption"], verbose=False)


def test_options_to_argv():
    assert brisk.options_to_argv({"acc": "high", "-tol": 1e-9, "x": True, "y": None}) == \
        ["-acc", "high", "-tol", "1e-09", "-x"]
    assert brisk.options_to_argv("-acc high") == ["-acc", "high"]
    # an underscore stands for a dash
    assert brisk.options_to_argv({"certify_y": "y.txt", "fomstart_x": "x.txt"}) == \
        ["-certify-y", "y.txt", "-fomstart-x", "x.txt"]


def band_sdpa(n=120):
    """max-cut-type SDP on a path: a tridiagonal (chordal) pattern, X_ii = 1."""
    i, j, v, mat = [], [], [], []
    for k in range(1, n + 1):
        mat.append(0); i.append(k); j.append(k); v.append(-0.25 * (1 if k in (1, n) else 2))
        if k < n:
            mat.append(0); i.append(k); j.append(k + 1); v.append(0.25)
    for k in range(1, n + 1):
        mat.append(k); i.append(k); j.append(k); v.append(1.0)
    return dict(blocksizes=[n], c=np.ones(n), mat=mat, blk=[1] * len(mat), i=i, j=j, v=v)


@pytest.mark.parametrize("chordal, converted", [(-1, True), (0, False), (1, True)])
def test_option_chordal(capsys, chordal, converted):
    """the options reach the solver: chordal decomposition automatic / off / forced."""
    r = brisk.solve_sdpa(**band_sdpa(), options={"chordal": chordal}, verbose=True)
    out = capsys.readouterr().out
    assert r.status == 0 and ("chordal decomposition" in out) == converted
    r0 = brisk.solve_sdpa(**band_sdpa(), verbose=False)
    assert np.isclose(r.pobj, r0.pobj, rtol=1e-6)


def test_option_symalg(capsys):
    """the *-algebra symmetry reduction (a hidden symmetry under a random orthogonal
    basis): on by default, off with symalg=0, the same objective either way."""
    f = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "hidden_symmetry23.dat-s")
    r = brisk.solve_file(f, verbose=True)
    out = capsys.readouterr().out
    assert r.status == 0 and "algebra symmetry: block 1: 23 -> 5 3(x2) 4(x3)" in out
    r0 = brisk.solve_file(f, options={"symalg": 0}, verbose=True)
    out = capsys.readouterr().out
    assert r0.status == 0 and "algebra symmetry" not in out
    assert np.isclose(r.pobj, r0.pobj, rtol=1e-7)
    assert r.X[0].shape == (23, 23) and np.linalg.eigvalsh(r.X[0]).min() > -1e-8


def test_missing_file_and_returnx():
    """a missing file is a FileNotFoundError; returnx=0 on a chordal problem returns no X
    (the point is verified on the cliques), returnx=1 and the default return it."""
    with pytest.raises(FileNotFoundError):
        brisk.solve_file("no_such_file.dat-s", verbose=False)
    r0 = brisk.solve_sdpa(**band_sdpa(), options={"chordal": 1, "returnx": 0}, verbose=False)
    assert r0.status == 0 and r0.X is None and r0.y is not None
    for o in ({"chordal": 1}, {"chordal": 1, "returnx": 1}):
        r = brisk.solve_sdpa(**band_sdpa(), options=o, verbose=False)
        assert r.status == 0 and r.X[0].shape == (120, 120) and np.isclose(r.pobj, r0.pobj, atol=1e-6)
        assert np.linalg.eigvalsh(r.X[0]).min() > -1e-7


def test_option_flags_and_values():
    """Flags take True/False and 1/0; numeric options reach the solver."""
    for opts in ({"nohsd": True}, {"nohsd": 1}, {"nohsd": 0}, {"hsd": True}, {"sym": "none"},
                 {"dir": "nt"}, {"nofr": 1}, {"dualize": 0}, {"freeelim": 0}, {"threads": 1}):
        r = brisk.solve_sdpa(**EX1, options=opts, verbose=False)
        assert r.status == 0, opts
    r = brisk.solve_sdpa(**maxcut_sdpa(60), options={"maxit": 2, "nodd": True}, verbose=False)
    assert r.status in (3, 6) or r.iterations <= 2 * 3
    with pytest.raises(RuntimeError):
        brisk.solve_sdpa(**EX1, options={"chordal": "notanumber"}, verbose=False)


def test_interrupt_from_thread():
    threading.Timer(0.5, brisk.interrupt).start()
    t = time.time()
    r = brisk.solve_sdpa(**maxcut_sdpa(150), options=LONG, verbose=False)
    assert r.interrupted and r.status in (5, 6) and r.status_str == "INTERRUPTED"   # 5: the point was already within 1e-6
    assert time.time() - t < 30 and r.y is not None
    r = brisk.solve_sdpa(**EX1, verbose=False)            # the next solve is not affected
    assert r.status == 0 and not r.interrupted


def test_ctrl_c():
    """SIGINT (Ctrl-C) in a script: the solve stops and returns; a second one abandons it."""
    code = textwrap.dedent(f"""
        import os, signal, threading, sys, time
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import brisk, test_brisk as T
        threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        r = brisk.solve_sdpa(**T.maxcut_sdpa(150), options=T.LONG, verbose=False)
        print("FIRST", r.status_str, flush=True)
        class Slow:                     # keeps the solver busy in its output callback
            def write(self, s):
                time.sleep(0.05)
            def flush(self):
                pass
        sys.stdout = Slow()
        for dt in (0.3, 0.6):
            threading.Timer(dt, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        try:
            brisk.solve_sdpa(**T.EX1, verbose=True)
        except KeyboardInterrupt:
            sys.__stdout__.write("SECOND KeyboardInterrupt\\n")
        sys.stdout = sys.__stdout__
        print("THIRD", brisk.solve_sdpa(**T.EX1, verbose=False).status_str, flush=True)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "FIRST INTERRUPTED" in out.stdout, out.stdout + out.stderr
    assert "SECOND KeyboardInterrupt" in out.stdout, out.stdout + out.stderr
    assert "THIRD OPTIMAL" in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------------------------- CVXPY
cp = pytest.importorskip("cvxpy")
from brisk.cvxpy_solver import cone_program_to_sdpa  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


def p_maxcut(n=12):
    rng = _rng(1)
    W = np.triu(rng.random((n, n)) < 0.4, 1).astype(float)
    W = W + W.T
    L = np.diag(W.sum(1)) - W
    X = cp.Variable((n, n), symmetric=True)
    return cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [X >> 0, cp.diag(X) == 1])


def p_lp_soc_eq():
    rng = _rng(2)
    A = rng.standard_normal((6, 10))
    b = A @ rng.random(10)
    x = cp.Variable(10)
    return cp.Problem(cp.Minimize(cp.norm(x - 1, 2) + cp.sum(x)), [A @ x == b, x >= 0, x <= 3])


def p_lmi_offset():
    rng = _rng(3)
    M = rng.standard_normal((5, 5))
    M = M + M.T
    t = cp.Variable()
    return cp.Problem(cp.Minimize(t + 2.5), [t * np.eye(5) - M >> 0])


def p_psd_variable():
    rng = _rng(4)
    C = rng.standard_normal((6, 6))
    X = cp.Variable((6, 6), PSD=True)
    return cp.Problem(cp.Minimize(cp.trace((C + C.T) @ X)), [cp.trace(X) == 1, X[0, 1] >= 0.05])


def p_socs():
    rng = _rng(5)
    A = rng.standard_normal((30, 20))
    b = rng.standard_normal(30)
    x = cp.Variable(20)
    return cp.Problem(cp.Minimize(cp.norm(A @ x - b) + 0.5 * cp.norm(x[:3]) + cp.norm(x, 1)),
                      [cp.sum(x) == 1])


def p_mixed():
    X = cp.Variable((4, 4), symmetric=True)
    y = cp.Variable(3)
    return cp.Problem(cp.Minimize(cp.trace(X) + cp.sum(y)),
                      [X >> 0.1 * np.eye(4), X[0, 1] == y[0], cp.norm(y) <= X[2, 2], y >= -1,
                       X[3, 3] + y[1] >= 2])


PROBLEMS = [p_maxcut, p_lp_soc_eq, p_lmi_offset, p_psd_variable, p_socs, p_mixed]


@pytest.mark.parametrize("make", PROBLEMS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("form,soc", [("auto", "auto"), ("image", "arrow"), ("image", "tree"),
                                      ("kernel", "arrow"), ("kernel", "tree")])
def test_cvxpy_against_clarabel(make, form, soc):
    prob = make()
    prob.solve(solver=cp.CLARABEL)
    ref = prob.value
    duals = [np.asarray(c.dual_value, float) for c in prob.constraints]
    prob.solve(solver=brisk.BRISK(), form=form, soc=soc)
    assert prob.status == cp.OPTIMAL
    if form != "auto":
        assert prob.solver_stats.extra_stats["form"] == form
    assert np.isclose(prob.value, ref, rtol=1e-6, atol=1e-7)
    for c, d in zip(prob.constraints, duals):
        # p_psd_variable's dual is not unique in the off-diagonal part: loose tolerance
        tol = 2e-3 if make is p_psd_variable else 2e-4
        assert np.allclose(np.asarray(c.dual_value, float), d, atol=tol, rtol=tol)


def test_cvxpy_lambda_max():
    prob = p_lmi_offset()
    prob.solve(solver=brisk.BRISK(), acc="high")
    M = prob.constraints[0].args[0].value + prob.variables()[0].value * 0   # t I - M
    assert np.isclose(np.linalg.eigvalsh(M)[0], 0, atol=1e-7)


def test_cvxpy_options_and_write(tmp_path):
    prob = p_maxcut()
    f = tmp_path / "maxcut.dat-s"
    prob.solve(solver=brisk.BRISK(), write_sdpa=str(f), options=["-acc", "high"], threads=1)
    assert prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and f.exists()
    r = brisk.solve_file(str(f), verbose=False)
    assert np.isclose(abs(r.pobj), abs(prob.value), rtol=1e-7)
    with pytest.raises(cp.error.SolverError):
        prob.solve(solver=brisk.BRISK(), nosuchoption=1)


def _certificate_ok(prob, form):
    """Solve an infeasible problem and check the returned Farkas certificate on CVXPY's data."""
    prob.solve(solver=brisk.BRISK(), form=form)
    assert prob.status == cp.INFEASIBLE, prob.status
    assert all(c.dual_value is not None for c in prob.constraints)
    data, _, _ = prob.get_problem_data(solver=brisk.BRISK())
    s = brisk.BRISK()
    res = s.solve_via_data(data, False, False, {"form": form})
    z = s._certificate(res["res"], res["info"], np.asarray(data["b"], float))
    # CVXPY's convention: PSD duals are matrix entries; for the checks take the adjoint
    from brisk.cvxpy_solver import _duals
    ze = _duals(res["res"], res["info"], ray=True, euclid=True)
    ze = ze * (np.linalg.norm(z) / np.linalg.norm(_duals(res["res"], res["info"], ray=True)))
    A = data["A"]
    assert np.isclose(float(np.asarray(data["b"]) @ ze), -1, rtol=1e-6)
    assert np.max(np.abs(A.T @ ze)) < 1e-5


@pytest.mark.parametrize("form", ["image", "kernel"])
def test_cvxpy_infeasible_certificates(form):
    x = cp.Variable(2)
    _certificate_ok(cp.Problem(cp.Minimize(x[0]), [x >= 1, x[0] + x[1] <= 1]), form)
    t = cp.Variable()
    _certificate_ok(cp.Problem(cp.Minimize(t), [t * np.eye(3) - np.diag([1.0, 2, 3]) >> 0, t <= 0]), form)
    y = cp.Variable(3)
    _certificate_ok(cp.Problem(cp.Minimize(0), [cp.norm(y) <= 1, y[0] >= 2]), form)
    _certificate_ok(cp.Problem(cp.Minimize(cp.sum(y)), [y >= 0, cp.sum(y) == -1]), form)


@pytest.mark.parametrize("form", ["image", "kernel", "auto"])
def test_cvxpy_unbounded(form):
    x = cp.Variable(2)
    p = cp.Problem(cp.Minimize(x[0]), [x[1] >= 1])            # x[0] in no constraint
    p.solve(solver=brisk.BRISK(), form=form)
    assert p.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)
    p = cp.Problem(cp.Minimize(x[0] + x[1]), [x[0] >= x[1] - 1, x[1] <= 5])
    p.solve(solver=brisk.BRISK(), form=form)
    assert p.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)


def test_cvxpy_auto_form():
    def form_of(prob):
        data, _, _ = prob.get_problem_data(solver=brisk.BRISK())
        return cone_program_to_sdpa(data["A"], data["b"], data["c"], data["dims"])[-1]["form"]
    assert form_of(p_maxcut(30)) == "kernel"             # a PSD variable: its entries are X
    y = cp.Variable(10)                                  # a small LMI in few variables
    M = sum(y[k] * np.diag(np.arange(30) == k).astype(float) for k in range(10))
    assert form_of(cp.Problem(cp.Minimize(cp.sum(y)), [M + np.eye(30) >> 0])) == "image"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cvxpy_time_limit_inaccurate():
    prob = p_maxcut(60)
    prob.solve(solver=brisk.BRISK(), maxit=3)
    assert prob.status in (cp.USER_LIMIT, cp.OPTIMAL_INACCURATE)
    prob.solve(solver=brisk.BRISK(), maxit=3, inaccurate_tol=1e30)
    assert prob.status == cp.OPTIMAL_INACCURATE and prob.value is not None


# ------------------------------------------------------------------- 4.38 bound mode
def _read_sdpa_exact(path):
    """(m, bs, c, entries) with Fractions: the test's own exact reader (an oracle, independent
    of libbrisk)."""
    import re
    from fractions import Fraction
    toks = []
    for line in open(path):
        s = line.strip()
        if s and s[0] not in '"*':
            toks.extend(x for x in re.split(r"[\s,{}()]+", s) if x)
    m, nb = int(toks[0]), int(toks[1])
    bs = [int(float(x)) for x in toks[2:2 + nb]]
    c = [Fraction(x) for x in toks[2 + nb:2 + nb + m]]
    rest = toks[2 + nb + m:]
    ent = []
    for q in range(0, len(rest) - 4, 5):
        t, k, i, j, v = int(rest[q]), int(rest[q + 1]) - 1, int(rest[q + 2]) - 1, int(rest[q + 3]) - 1, Fraction(rest[q + 4])
        ent.append((t, k, min(i, j), max(i, j), v))
    return m, bs, c, ent


def test_bound_mode_low_level(tmp_path):
    """-bound p / d on the max-cut SDPA problem: valid certificates, rigorously certified in
    libbrisk (-certify, and brisk.certify on the returned arrays), bracketing the optimum."""
    data = maxcut_sdpa(30, seed=3)
    f = tmp_path / "mc.dat-s"
    brisk.write_sdpa(f, **data)
    r0 = brisk.solve_file(str(f), {"acc": "high"}, verbose=False)
    opt = r0.pobj
    rp = brisk.solve_file(str(f), {"bound": "p", "certify": True}, verbose=False)
    assert rp.bound["side"] == "p" and rp.bound["valid"] >= 1 and rp.bound["certified"]
    assert rp.bound["rigorous"] >= opt - 1e-7 * (1 + abs(opt))
    cp_ = brisk.certify(str(f), "p", X=rp.X)
    assert cp_["certified"] and abs(cp_["bound"] - rp.bound["rigorous"]) <= 1e-12 * (1 + abs(opt))
    rd = brisk.solve_file(str(f), {"bound": "d", "certify": True}, verbose=False)
    assert rd.bound["side"] == "d" and rd.bound["certified"]
    cd = brisk.certify(str(f), "d", y=rd.y)
    assert cd["certified"] and cd["bound"] <= opt + 1e-7 * (1 + abs(opt))
    assert cd["bound"] <= cp_["bound"] and abs(cp_["bound"] - cd["bound"]) < 1e-5 * (1 + abs(opt))
    # in memory (exact data) as well
    cm = brisk.certify(data, "d", y=rd.y)
    assert cm["certified"]
    # the default: no certification
    rn = brisk.solve_file(str(f), {"bound": "d"}, verbose=False)
    assert not rn.bound["certified"] and not np.isfinite(rn.bound["rigorous"])
    # a certificate that is not one
    bad = brisk.certify(str(f), "d", y=rd.y * 1.1 + 1.0)
    assert not bad["certified"] and bad["reason"]


def test_certify_against_exact_rationals(tmp_path):
    """libbrisk's certificate for the SDPA manual's example 1, checked in exact rational
    arithmetic by the test's own reader: Z(y) positive definite (LDL' in Fractions) and the
    rigorous bound below b'y."""
    from fractions import Fraction
    f = tmp_path / "ex1.dat-s"
    brisk.write_sdpa(f, **EX1)
    rd = brisk.solve_file(str(f), {"bound": "d", "certify": True}, verbose=False)
    assert rd.bound["certified"]
    m, bs, c, ent = _read_sdpa_exact(str(f))
    yq = [Fraction(float(v)) for v in rd.y]
    n = bs[0]
    Z = [[Fraction(0)] * n for _ in range(n)]
    for t, k, i, j, v in ent:
        coef = -v if t == 0 else -yq[t - 1] * v
        Z[i][j] += coef
        if i != j:
            Z[j][i] += coef
    A = [row[:] for row in Z]
    for p in range(n):
        assert A[p][p] > 0
        for i in range(p + 1, n):
            f_ = A[i][p] / A[p][p]
            for j in range(p, n):
                A[i][j] -= f_ * A[p][j]
    assert Fraction(rd.bound["rigorous"]) <= sum(bi * yi for bi, yi in zip(c, yq))


def test_bound_cvxpy():
    prob = p_maxcut(14)
    prob.solve(solver=cp.CLARABEL)
    ref = prob.value                       # Maximize: CVXPY minimizes -f
    for bound, kind in (("primal", "upper"), ("dual", "lower")):
        for form in ("image", "kernel"):
            prob.solve(solver=brisk.BRISK(), bound=bound, form=form, certify=True)
            b = prob.solver_stats.extra_stats["bound"]
            assert b["kind"] == kind and b["valid"] >= 1 and b["certified"]
            assert (b["rigorous"] >= b["value"] - 1e-9) if kind == "upper" else (b["rigorous"] <= b["value"] + 1e-9)
            # the conic problem is  min -f:  an upper bound U >= -ref, a lower bound L <= -ref
            if kind == "upper":
                assert b["value"] >= -ref - 1e-6
            else:
                assert b["value"] <= -ref + 1e-6
    with pytest.raises(ValueError):
        prob.solve(solver=brisk.BRISK(), bound="both")


def test_resolve_note_and_cause_fields():
    r = brisk.solve_sdpa(**EX1, verbose=False)
    assert r.n_resolves == 0 and r.cause == "" and r.bound is None


# ---- high precision ----------------------------------------------------------------------
@pytest.mark.parametrize("prec, digits, tol", [("dd", 31, 1e-22), ("qd", 63, 1e-45), (80, 82, 1e-58)])
def test_high_precision_ipm(prec, digits, tol):
    r = brisk.solve_sdpa(**EX1, options={"prec": prec}, verbose=False)
    assert r.status == 0 and r.status_str == "OPTIMAL"
    assert np.max(np.abs(r.dimacs)) < tol
    assert abs(r.dobj - 41.9) < 1e-12
    h = brisk.hp_solution()
    assert h["digits"] >= digits and h["blocksizes"] == [2]
    assert abs(h["pobj"] - h["dobj"]) < tol * 100 and len(h["y"]) == 3 and h["X"][0].shape == (2, 2)
    # all the digits: the dual value agrees with 41.9 far beyond double precision
    import decimal
    assert abs(h["dobj"] - decimal.Decimal("41.9")) < decimal.Decimal(tol) * 100
    assert brisk.hp_solution(str)["y"].dtype == object


@pytest.mark.parametrize("method", ["fom", "lralm"])
def test_high_precision_first_order(method):
    p = maxcut_sdpa(30, seed=1)
    r0 = brisk.solve_sdpa(**p, options={"prec": "dd"}, verbose=False)
    r = brisk.solve_sdpa(**p, options={"prec": "dd", method: 1}, verbose=False)
    assert r.status == 0 and np.max(np.abs(r.dimacs)) < 1e-17
    assert abs(r.dobj - r0.dobj) < 1e-12 * (1 + abs(r0.dobj))


def test_high_precision_cleared_by_double_solve():
    brisk.solve_sdpa(**EX1, options={"prec": "dd"}, verbose=False)
    assert brisk.hp_solution() is not None
    brisk.solve_sdpa(**EX1, verbose=False)
    assert brisk.hp_solution() is None


def _free_pair_problem():
    """EX1 with a split free variable f = x1 - x2 (an LP block of two, columns +-a, cost +-1)."""
    p = dict(blocksizes=[2, -2], c=EX1["c"], mat=list(EX1["mat"]), blk=list(EX1["blk"]),
             i=list(EX1["i"]), j=list(EX1["j"]), v=list(EX1["v"]))
    for con, a in ((1, 1.0), (3, 1.0), (0, 1.0)):
        for s, idx in ((1.0, 1), (-1.0, 2)):
            p["mat"].append(con); p["blk"].append(2); p["i"].append(idx); p["j"].append(idx); p["v"].append(s * a)
    return p


def test_high_precision_presolve_free_variable(monkeypatch):
    p = _free_pair_problem()
    r = brisk.solve_sdpa(**p, options={"prec": "qd"}, verbose=False)
    assert r.status == 0 and np.max(np.abs(r.dimacs)) < 1e-40
    assert np.min(r.X[1]) >= 0 and np.prod(r.X[1]) == 0          # the pair: one of the two is zero
    monkeypatch.setenv("BRISK_HPNOPRE", "1")                      # the same without the presolve
    r2 = brisk.solve_sdpa(**p, options={"prec": "qd"}, verbose=False)
    assert abs(r.dobj - r2.dobj) < 1e-10 * (1 + abs(r.dobj))


def test_high_precision_presolve_face():
    """min <C,X>, diag(X) = 1, <ee',X> = 0: no interior; the presolve restricts X to e-perp."""
    n = 6
    rng = np.random.default_rng(3)
    C = rng.standard_normal((n, n)); C = C + C.T
    mat, blk, ii, jj, vv = [], [], [], [], []
    for a in range(n):
        for b in range(a, n):
            mat.append(0); blk.append(1); ii.append(a + 1); jj.append(b + 1); vv.append(-C[a, b])   # F0 = -C
            mat.append(n + 1); blk.append(1); ii.append(a + 1); jj.append(b + 1); vv.append(1.0)
        mat.append(a + 1); blk.append(1); ii.append(a + 1); jj.append(a + 1); vv.append(1.0)
    c = [1.0] * n + [0.0]
    r = brisk.solve_sdpa([n], c, mat, blk, ii, jj, vv, options={"prec": "dd"}, verbose=False)
    assert r.status == 0 and np.max(np.abs(r.dimacs)) < 1e-18
    assert np.max(np.abs(r.X[0] @ np.ones(n))) < 1e-12 and np.allclose(np.diag(r.X[0]), 1.0)


def test_high_precision_extension():
    """A tolerance below what double-double reaches (1e-36): without the extension levels the
    solve ends at its precision limit, with them (the default) it continues in quad-double."""
    n = 5
    rng = np.random.default_rng(11)
    C = rng.standard_normal((n, n)); C = C + C.T
    mat, blk, ii, jj, vv = [], [], [], [], []
    for a in range(n):
        for b in range(a, n):
            mat.append(0); blk.append(1); ii.append(a + 1); jj.append(b + 1); vv.append(-C[a, b])   # F0 = -C
        mat.append(a + 1); blk.append(1); ii.append(a + 1); jj.append(a + 1); vv.append(1.0)
    c = [1.0] * n
    opts = {"prec": "dd", "hptol": 1e-36}
    r0 = brisk.solve_sdpa([n], c, mat, blk, ii, jj, vv, options=dict(opts, hpext=0), verbose=False)
    r = brisk.solve_sdpa([n], c, mat, blk, ii, jj, vv, options=opts, verbose=False)
    assert r0.status != 0 and np.max(np.abs(r0.dimacs)) < 1e-18
    assert r.status == 0 and np.max(np.abs(r.dimacs)) <= 1e-36


def test_high_precision_file_with_header_text(tmp_path):
    """The SDPA manual's layout: text after the numbers of the header lines ("3 =mDIM",
    "2 =bLOCKsTRUCT"); the high-precision reader stopped at the block sizes' text."""
    f = tmp_path / "ex1.dat-s"
    f.write_text('"Example 1 of the SDPA manual\n3 =mDIM\n1 =nBLOCK\n2 =bLOCKsTRUCT\n48 -8 20\n'
                 '0 1 1 1 -11\n0 1 2 2 23\n1 1 1 1 10\n1 1 1 2 4\n2 1 2 2 -8\n3 1 1 2 -8\n3 1 2 2 -2\n')
    r = brisk.solve_file(str(f), {"prec": "dd"}, verbose=False)
    assert r.status == 0 and abs(r.dobj - 41.9) < 1e-12 and np.max(np.abs(r.dimacs)) < 1e-20


def test_high_precision_block_split(monkeypatch):
    """One 6x6 block whose data couple only {1,3,5} and {2,4,6}: solved as two 3x3 blocks,
    returned in the original shape, the same optimum as without the split."""
    n = 6
    rng = np.random.default_rng(5)
    mat, blk, ii, jj, vv = [], [], [], [], []
    for g in (0, 1):
        idx = [g + 1, g + 3, g + 5]
        C = rng.standard_normal((3, 3)); C = C + C.T
        for a in range(3):
            for b in range(a, 3):
                mat.append(0); blk.append(1); ii.append(idx[a]); jj.append(idx[b]); vv.append(-C[a, b])
    for a in range(n):
        mat.append(a + 1); blk.append(1); ii.append(a + 1); jj.append(a + 1); vv.append(1.0)
    c = [1.0] * n
    r = brisk.solve_sdpa([n], c, mat, blk, ii, jj, vv, options={"prec": "dd"}, verbose=False)
    h = brisk.hp_solution()
    assert r.status == 0 and np.max(np.abs(r.dimacs)) < 1e-22
    assert h["blocksizes"] == [n] and r.X[0].shape == (n, n) and r.Z[0].shape == (n, n)
    assert np.allclose(np.diag(r.X[0]), 1.0) and np.all(r.X[0][0::2, 1::2] == 0)
    monkeypatch.setenv("BRISK_HPNOSPLIT", "1")
    r2 = brisk.solve_sdpa([n], c, mat, blk, ii, jj, vv, options={"prec": "dd"}, verbose=False)
    assert r2.status == 0 and abs(r.dobj - r2.dobj) < 1e-12 * (1 + abs(r.dobj))


def test_high_precision_certificates(tmp_path):
    """-prec dd with -bound d / p on the SDPA manual's example 1: rigorous bounds that bracket
    the optimum to 20 digits; the d certificate checked in exact rational arithmetic by the
    test's own reader; brisk.certify(prec=...) on the returned certificates."""
    from fractions import Fraction
    from decimal import Decimal
    f = tmp_path / "ex1.dat-s"
    brisk.write_sdpa(f, **EX1)
    rd = brisk.solve_file(str(f), {"prec": "dd", "bound": "d"}, verbose=False)
    assert rd.status == 0 and rd.bound["side"] == "d" and rd.bound["certified"]
    hd = brisk.hp_solution()
    L = hd["bound"]
    assert isinstance(L, Decimal) and hd["digits"] >= 90
    # exact oracle: Z(y) is positive definite and the bound is below b'y
    m, bs, c, ent = _read_sdpa_exact(str(f))
    yq = [Fraction(v) for v in hd["y"]]
    n = bs[0]
    Z = [[Fraction(0)] * n for _ in range(n)]
    for t, k, i, j, v in ent:
        coef = -v if t == 0 else -yq[t - 1] * v
        Z[i][j] += coef
        if i != j:
            Z[j][i] += coef
    A = [row[:] for row in Z]
    for p in range(n):
        assert A[p][p] > 0
        for i in range(p + 1, n):
            f_ = A[i][p] / A[p][p]
            for j in range(p, n):
                A[i][j] -= f_ * A[p][j]
    by = sum(bi * yi for bi, yi in zip(c, yq))
    assert Fraction(L) <= by and by - Fraction(L) < Fraction(1, 10 ** 25) * (1 + abs(by))
    cd = brisk.certify(str(f), "d", y=hd["y"], prec="dd")
    assert cd["certified"] and abs(cd["bound"] - L) <= Decimal("1e-28") * (1 + abs(L))
    rp = brisk.solve_file(str(f), {"prec": "dd", "bound": "p"}, verbose=False)
    assert rp.bound["side"] == "p" and rp.bound["certified"]
    hp_ = brisk.hp_solution()
    U = hp_["bound"]
    assert L <= U and U - L < Decimal("1e-20") * (1 + abs(L))
    assert rd.bound["rigorous"] <= float(L) and float(U) <= rp.bound["rigorous"]
    cp_ = brisk.certify(str(f), "p", X=hp_["X"], prec="dd")
    assert cp_["certified"] and cp_["bound"] >= L
    # a wrong certificate is not certified
    bad = brisk.certify(str(f), "d", y=[v * 2 + 1 for v in hd["y"]], prec="dd")
    assert not bad["certified"]
    # without a bound option there is no bound
    brisk.solve_file(str(f), {"prec": "dd"}, verbose=False)
    assert brisk.hp_solution()["bound"] is None
