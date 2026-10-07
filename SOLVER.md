# BRISK 1.3.2: the solver in detail

BRISK solves block-diagonal semidefinite programs with LP blocks:

```
(P)  min <C,X>  s.t. <A_i,X> = b_i,  X in K        (D)  max b'y  s.t.  sum y_i A_i + Z = C,  Z in K
```

It reads SDPA sparse files (`.dat-s`), so SDPLIB and Mittelmann's instances run unchanged.
It can also be called from Python/CVXPY, Julia/JuMP and MATLAB/Octave. All interfaces run the
same C code (about 28 000 lines on BLAS/LAPACK).

**Installation: see [INSTALL.md](INSTALL.md).** In short: `make && ./brisk examples/theta1.dat-s`.

## Linear and second-order cone programs, the SeDuMi format

`./brisk problem.mat` reads a problem in SeDuMi format (a MAT-file with `A` or `At`, `b`, `c`
and `K`): min c'x, A x = b, x in K, with K.f (free), K.l (nonnegative), K.q (second-order
cones), K.r (rotated second-order cones) and K.s (semidefinite blocks). The same data are
taken by `brisk.solve_sedumi` (Python), `Brisk.solve_sedumi` (Julia), `brisk_sedumi`
(MATLAB/Octave) and `brisk_solve_sedumi` (C). A problem without semidefinite blocks is solved
by a separate cone solver for linear and second-order cone programs; with semidefinite blocks
(or with `-conesolver 0`, `-prec`, `-bound`, `-certify`) the semidefinite solver takes the
second-order cones as arrow blocks. CVXPY and JuMP models with second-order cone constraints
are routed the same way.

## Linear programs (MPS) and CBF files

`./brisk problem.mps` reads a linear program in MPS format (free or fixed), presolves it and
solves it by an interior-point method for linear programs (upper bounds kept as bounds, normal
equations); infeasible and unbounded problems are handed to the cone solver, which returns the
certificate. `-x file` writes x of the problem as read. The solution is an interior one (no
crossover). `./brisk problem.cbf` (or `.cbf.gz`) reads the conic benchmark format of CBLIB for
linear and second-order cone problems; integer variables are relaxed.

## Interfaces

- **Command line:** `./brisk problem.dat-s [options]` (or `problem.mat`, SeDuMi format). `./brisk` alone lists every option.
- **Python / CVXPY** ([python/README.md](python/README.md)): `pip install "./python[cvxpy]"`.
  Then `prob.solve(solver=brisk.BRISK())` in CVXPY, or `brisk.solve_file` / `brisk.solve_sdpa`
  on SDPA data.
- **Julia / JuMP** ([julia/Brisk/README.md](julia/Brisk/README.md)): `Model(Brisk.Optimizer)`,
  a MathOptInterface optimizer, plus direct calls on SDPA files or data.
- **MATLAB / Octave** ([matlab/README.md](matlab/README.md)): a MEX interface taking SeDuMi
  data (`brisk_sedumi(A, b, c, K)`) or SDPA data (`brisk_sdpa`).
- **C:** the shared library `libbrisk` (`make libbrisk`, interface in `capi.c`).

## Running

```
./brisk problem.dat-s [-acc low|default|high] [-tol 1e-8] [-q|-v] [-timelimit s] [-threads k]
                      [-y y.txt] [-x X.txt] [-z Z.txt] ...
```

All options, and how to pass them from Python, Julia and MATLAB, are in [OPTIONS.md](OPTIONS.md).
For example, `-chordal 0` turns the chordal decomposition off, and `-sym none` the symmetry
reduction.

**Accuracy:**
- `-acc low|default|high` sets the tolerance to 1e-6 / 1e-8 / 1e-10. `-tol` sets it directly.
- The returned X, y and Z, the printed DIMACS errors and the status all refer to the problem
  as read. Presolve reductions are undone, and the errors are measured on the original data.
- The printed optimal value uses the SDPA/SDPLIB convention (max tr(F0 Y)), so it compares
  directly with SDPLIB's table. BRISK's "primal infeasible" refers to SDPA's dual.

**The summary at the end of the log (1.3):** every run closes with a block that says what
happened — the problem as read; every preprocessing step applied and what it changed (sign
and permutation symmetries, the *-algebra block diagonalisation, the dual form, free
variables eliminated, the trace bound, the moment-form conversion, the chordal
decomposition, the facial reduction); the engine and the method (interior-point with the
infeasible start or the self-dual embedding and its direction, matrix-free, first-order,
low-rank); every re-solve, method switch (e.g. the matrix-free attempt and the standard
method continuing from its iterate), double-double endgame, crossover and
bound; the time; the number of threads; the six DIMACS errors on the data as read; and the closing line "Solved to
DIMACS error e" (or "Primal/Dual infeasible" with whether the returned point is a
certificate). The `status:` line with the classification (OPTIMAL, SOLVED TO REDUCED
ACCURACY, ...) and the exit code are unchanged. `BRISK_NOSUMMARY=1` leaves the block out.
After a solve in standard precision the log ends with the ways to more accuracy:
`-acc high` (tolerance 1e-10, still in double precision arithmetic) and the high-precision
solver, `-prec dd` (about 31 digits), `-prec qd` (about 63) or `-prec <digits>`; after
`-acc high` only the latter, for a linear program `-tol`. It also says how to get a
guaranteed bound on the optimal value from a certified feasible point (`-bound d` or `p`,
and `-certify` for the rigorous check); a high-precision log ends with that part. The
options are written in the syntax of whoever called the solver: on the command line as
above, from Python as `"acc": "high"` in `brisk.solve_sdpa(..., options={...})`, from CVXPY
as `acc="high"`, from Julia as the keyword `acc = "high"`, from JuMP as
`set_attribute(model, "acc", "high")`, from MATLAB as `opts.acc = 'high'`, from C as the
option strings `"-acc", "high"`; what was already asked for is not advised again.

**Exit codes:**

| code | meaning |
|---|---|
| 0 | optimal |
| 10 | reduced accuracy (max error ≤ 1e-6); with `-prec`: PRECISION LIMIT, the tolerance was not reached |
| 11 / 12 | primal / dual infeasible |
| 13 | iteration limit |
| 14 | numerical difficulties |
| 15 | time limit |
| 1 / 2 | usage / input error |

**Threads:** `-threads k` or `OMP_NUM_THREADS` (default: all cores). With the default count,
tiny problems run on one thread, and on Linux the interior-point solve uses only the cores
that are free when it starts (`-v` shows the choice). With a fixed count on a machine where
other processes occupy cores, use fewer threads or `OMP_WAIT_POLICY=passive`.

**Ctrl-C** stops the solve and returns the current point. Pressing it twice aborts.

## What is chosen automatically

Every automatic choice below can be overridden from the command line.

- **Method.** A Mehrotra predictor-corrector path-following method in two variants: the
  homogeneous self-dual embedding and the infeasible-start method.
  - The embedding runs first when its extra dense work is small next to the Schur
    factorization (12·Σn³ ≤ m³/3) or next to factorization and assembly together (at most
    half of their estimated work, `-hsdfirstasm`), when the problem is tiny, when there are
    many split free pairs, or in the chordal moment form.
  - The infeasible start runs first only on large dense blocks with a cheap Schur complement
    (max-cut and graph-partitioning type problems).
  - Each method is the other's fallback, and a result that misses 1e-6 on the original data
    is re-solved with the other one.
  - `-hsd` / `-nohsd` force one method.
- **Direction.** HKM or NT: NT where its extra dense work is cheap next to the Schur
  factorization, HKM otherwise (`-dir`).
- **Schur complement.** Assembled by fused sparse kernels, row products, dense BLAS-3, a
  low-rank route or a dictionary route, chosen by a cost model. It is factored dense (mixed
  precision), by envelope, or by a supernodal sparse Cholesky.
- **Presolve:**
  - facial reduction;
  - sign and permutation symmetry reduction (`-sym none` turns it off);
  - block structure shared by all data matrices of an SDP block, in any orthogonal basis: each
    block of size up to 1000 is split along the matrix algebra its data generate. This covers
    a symmetry group that fixes every constraint matrix (any such group, not only
    permutations), commuting data and hidden direct sums. It does not cover symmetries that
    map the constraints onto each other, unless they are permutations (the item above).
    `-symalg 0` turns it off, `-symalg 1` checks every block;
  - elimination of split free variables;
  - the dual form, when it has at most half the rows;
  - chordal decomposition, when it lowers the cost per iteration.
- **Problems too large for the Schur complement.** The first-order engine (an ADMM/
  Douglas–Rachford splitting with Anderson acceleration, then an augmented Lagrangian with
  semismooth Newton-CG) is chosen automatically when the *dense* Schur complement would not
  fit in memory (8 m² bytes > half the memory), with at most 64 SDP blocks and one of them at
  least 200 × 200. A large block with a very sparse pattern (≤ 1% dense, e.g. AC-OPF relaxations)
  first gets the chordal conversion, and the first-order engine runs only if nothing converts.
  `-fom 1` forces it. The envelope and sparse Schur paths have no such limit.
  At a tolerance of 1e-6 or looser (`-acc low`) the engine also runs *first* on such
  few-large-dense-block problems when their Gram matrix A A' is sparse (theta-type problems)
  and the interior-point solve is estimated at 20 s or more:
  it gets 20% of that time (`-fomrace 0.2`; 0 turns it off), and the interior-point method
  follows if it has not reached the tolerance.

- **The matrix-free method first.** On problems with one to four dense blocks of order 100 or
  more that the presolve leaves unchanged and whose Schur factorization is estimated at 3.5
  times a matrix-free solve or more — at least equal when every constraint has an entry in an
  LP block — (truss topology problems, SOS relaxations with a low-rank Gram matrix), the matrix-free interior-point method (`-mfipm 1` below) runs first. The
  attempt is stopped when a solve away from the optimum needs more CG steps than a sixth of
  one factorization costs, or after a quarter of the standard solve's work; unless it ends
  OPTIMAL the standard method follows (from the attempt's iterate of merit 1e-3 if it got that far). `-mftry 0` turns this off. The attempt is also made when the dense Schur complement does
  not fit in memory: the first-order engine is then the fallback and a result at reduced
  accuracy is kept; when neither the Schur complement nor the engine's factor of A A' fits,
  the matrix-free method runs as well.
- **A double-double endgame for small problems that end short of the tolerance** (m ≤ 1,200,
  Σ n² ≤ 2e5): a few iterations with all the linear algebra in double-double arithmetic,
  within a work budget; the result is kept when it is better. `-nodd` turns it off. The log
  ends with the ways to more accuracy (`-acc high`, `-prec`) and to a guaranteed bound
  (`-bound`, `-certify`), in the syntax of the calling interface.
- **A point that missed the tolerance has its primal feasibility restored before it is
  returned** (1.3.2): X ← X^½ (I + W) X^½ with W chosen so that A(X) = b. On problems without
  an interior the primal residual rises to about 1e-6 in the last iterations while the dual
  residual and the complementarity keep falling; this correction removes the primal residual
  (to 1e-12), keeps X positive semidefinite and leaves the complementarity where it was. It
  starts from the iterate with the smallest dual residual and complementarity. `-nopolishr`
  turns it off.

## Optional methods

- **`-prec dd | qd | <digits>`:** high precision. The solve runs in double-double (about 32
  digits), quad-double (about 64) or a variable precision; the solve aims at an error of 1e-24, 1e-48,
  or 1e-77 at 100 digits (`-hptol` sets a tolerance). The data are read exactly as written in the file, and
  `-x`, `-y`, `-z` write all the digits. With `-fom 1` or `-lralm 1` a first-order method
  runs in high precision instead of the interior-point method. Blocks whose data are block diagonal
  after a permutation are split; the reductions of the double solver are applied to the data
  in the working precision (sign and permutation symmetry, with the group verified entry by
  entry; the block diagonalisation of the data; the chordal decomposition where the banded
  factorization can use it); a presolve in the working
  precision removes free variables and simple faces, and the solve starts from a double
  solve by the default method. Problems without an interior that the presolve does not
  reduce reach about half the digits of a precision; the solve then continues in the next
  precision (up to `-hpext 2` times), with the tolerance unchanged. Status PRECISION LIMIT:
  the tolerance was not reached.
- **`-mfipm 1`:** a matrix-free interior-point method. Its Newton systems are solved by
  preconditioned CG, so the Schur complement is never formed. It suits problems with a
  low-rank optimal side, e.g. SOS relaxations with a low-rank Gram matrix, or truss topology
  problems (m = 41,616 in 2.5 minutes on one core, where a Schur complement would need 14 GB). **`-mfipm 2`** is
  the hybrid: the matrix-free iteration while its CG is cheap, then the standard method from
  that iterate (a few Schur factorizations instead of thirty; the hand-off at a merit of 1e-3, `-mfhand`). The default does the same by itself when its
  matrix-free attempt gets near but does not end OPTIMAL.
- **`-lralm 1` (experimental):** a low-rank augmented Lagrangian method (X = R Rᵀ) for very large
  sparse SDPs such as AC-OPF relaxations beyond the interior-point method's reach. Its Newton
  steps use a sparse n × n preconditioner, so its memory stays O(nnz + n·rank). It reports
  b'y as a lower bound when Z = C − A'y passes a sparse Cholesky test. It converges slowly on
  mid-size problems; see `./brisk` for its options.
- **`-dual`:** a sparse dual-scaling method. It keeps Z sparse and never forms X, and is fast on
  sparse relaxations such as max-cut and box-QP.
- **`-bound p|d`:** returns the best *guaranteed* bound on one side instead of the best-balanced
  primal-dual pair.
  - `p`: the feasible X with the smallest <C,X>, an upper bound on the optimum even when the
    solve does not converge.
  - `d`: the feasible y with the largest b'y (Z ⪰ 0 verified), a lower bound.
  - For SOS programs from GloptiPoly or SeDuMi, use `-bound p` (`opts.bound = 'sos'` in
    MATLAB; `bound="sos"` in CVXPY and Julia).
  - `-certify` makes the bound rigorous: the file's decimals in intervals, directed rounding, a
    verified λmin.
  - `brisk problem.dat-s -certify-x X.txt` (or `-certify-y y.txt`) checks a given certificate,
    from BRISK or any solver, without solving.
  - With `-prec` the certificate is built and checked in high precision: the rigorous bound
    is printed with all the digits (for `-prec dd` typically within 1e-24 of the optimal
    value), `-x` / `-y` write the certificate with all the digits of its check, and
    `-certify-x` / `-certify-y` with `-prec` check a given one.

## Reproducibility

For a given build, thread count and BLAS library a run is deterministic: no decision depends on
timings or on random seeds, and the threaded loops give the same result for any schedule. Two
things outside BRISK can still make two runs differ:

- **A BLAS whose threaded rounding varies from run to run** (reported with macOS/Accelerate). The
  differences start at the level of a rounding error, and the single-precision Schur
  factorizations used far from the optimum (m ≥ 1000) amplify them to 1e-7 … 1e-5 in the early
  iterates; the result is the same to the tolerance, the iteration count can change by a few.
  `-mixed 0` (`mixed = 0`) keeps every factorization in double; one BLAS thread
  (`VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) removes the source.
- **A different thread count or BLAS**: other rounding, same remarks.

## Files

| path | contents |
|---|---|
| `INSTALL.md` | installation of the command line and all interfaces |
| `OPTIONS.md` | every solver option, explained, and how to pass it from each interface |
| `Makefile` | build (`make`, `make libbrisk`, `make capitest`, `make pytest`) |
| `*.c`, `brisk.h` | the solver |
| `capi.c`, `libbrisk.map`, `libbrisk.exp` | C interface of the shared library `libbrisk` |
| `tools/capi_test.c` | test and example of the C interface |
| `amd/` | the AMD ordering (SuiteSparse, BSD 3-clause, `amd/LICENSE_AMD.txt`) |
| `omp/` | a small OpenMP runtime for compilers without one (macOS clang) |
| `python/` | Python package: ctypes binding of `libbrisk`, CVXPY solver, examples, tests |
| `julia/Brisk/` | Julia package: MOI/JuMP optimizer, examples, tests |
| `matlab/` | MEX gateway, SeDuMi and SDPA wrappers, build script, tests |
| `examples/` | small SDPLIB problems (theta1, theta3, control1, truss1, arch0, mcp250-1); `afiro.mps` (a linear program of Netlib), `soc_small.cbf` and `soc_small.mat` (a second-order cone program in CBF and in SeDuMi format) |
