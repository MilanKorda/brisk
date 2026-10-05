"""BRISK from Python: libbrisk through ctypes, and a CVXPY solver.

    import brisk
    r = brisk.solve_file("theta3.dat-s", {"acc": "high"})
    r = brisk.solve_sdpa(blocksizes, c, mat, blk, i, j, v)
    r = brisk.solve_sedumi(A, b, c, K)               # SeDuMi format: K = dict(f=, l=, q=, r=, s=)

    import cvxpy as cp
    prob.solve(solver=brisk.BRISK(), acc="high")      # needs cvxpy >= 1.9
"""
from ._core import (STATUS, ConeResult, Result, certify, hp_solution, interrupt, library_path, load_library, options_to_argv,
                    set_interruptible, solve_file, solve_sdpa, solve_sedumi, version, write_sdpa)

__version__ = "1.2.1"
__all__ = ["STATUS", "ConeResult", "Result", "certify", "hp_solution", "interrupt", "set_interruptible", "library_path", "load_library", "options_to_argv", "solve_file",
           "solve_sdpa", "solve_sedumi", "version", "write_sdpa", "BRISK"]


def __getattr__(name):              # cvxpy is optional: imported on first use of brisk.BRISK
    if name in ("BRISK", "cvxpy_solver"):
        import importlib
        cvxpy_solver = importlib.import_module(".cvxpy_solver", __name__)
        return cvxpy_solver.BRISK if name == "BRISK" else cvxpy_solver
    raise AttributeError(name)
