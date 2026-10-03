# BRISK change log

## Version 1.1 (October 2026)

- **High precision**: `-prec dd | qd | <digits>` solves in double-double (about 32 digits),
  quad-double (about 64) or a variable precision, with the solver's own arithmetic: no
  external library. From every interface (`prec` in Python, Julia, MATLAB and Octave).
  - The interior-point method starts from the double-precision solution and goes up a ladder
    of precisions; it ends quadratically (one to three iterations in the higher precisions).
    With `-fom 1` or `-lralm 1` a first-order method runs in high precision instead.
  - The numbers of an SDPA file are read exactly as written. A presolve in the working
    precision (free variables, simple faces) with an exact postsolve; blocks are split by
    their sparsity pattern; a sparse Schur complement (many small blocks) is reordered and
    factored within its envelope.
  - `-hptol` sets the tolerance. When the precision is exhausted short of it (problems
    without an interior), the solve continues in the next precision (`-hpext`). New status
    PRECISION LIMIT (exit code 10).
- **Rigorous bounds in high precision**: `-bound p|d` with `-prec` builds and checks the
  certificate in high precision and prints the bound with all the digits; `-certify-x` /
  `-certify-y` with `-prec` check a given certificate.
- **All the digits of a high-precision solution**: `-x`, `-y`, `-z` files; `brisk_hp_text` in
  the library; `brisk.hp_solution()` in Python (Decimal, or any type built from a string);
  `Brisk.hp_solution()` in Julia (BigFloat). MATLAB and Octave return doubles.
- Fixed: a memory leak in the symmetry reduction when it returned early.

## Version 1.0 (October 2026)

First public release.

- Primal-dual interior-point method for SDP with LP blocks: Mehrotra predictor-corrector, the
  homogeneous self-dual embedding and the infeasible start, HKM and NT directions, each the
  other's fallback.
- Presolve, undone on return: facial reduction, free variables, the dual form, chordal
  decomposition, exact sign and permutation symmetry reduction, block structure shared by the
  data matrices.
- Sparse and low-rank Schur complement routes; sparse Cholesky with the AMD ordering.
- Infeasibility detection; guaranteed bounds (`-bound`) with a rigorous check (`-certify`).
- Beyond the interior-point method: a first-order engine (`-fom`), a matrix-free
  interior-point method (`-mfipm`), and an experimental low-rank method (`-lralm`).
- Interfaces: command line, C library, Python (with CVXPY), Julia (with JuMP/MathOptInterface),
  MATLAB and GNU Octave. Every option can be passed from every interface (OPTIONS.md).

Known limitations: the macOS build has not been tested on a Mac; time limits are checked
between iterations; the dual bound's certificate can be slow on problems solved in the dual
form.
