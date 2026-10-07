# BRISK change log

## Version 1.3.2 (October 2026)

- **A point that missed the tolerance has its primal feasibility restored in the square-root
  metric of X** before it is returned. On problems without an interior (many relaxations of
  polynomial optimization) the Schur complement is numerically singular in the last
  iterations: the Newton direction no longer satisfies the primal equations, the primal
  residual rises to about 1e-6 while the dual residual and the complementarity keep falling,
  and the correction applied so far, which solves with that same matrix, could not repair
  it. The new correction, X ← X^½ (I + W) X^½ with W chosen so that A(X) = b, keeps X positive
  semidefinite and the complementarity where it was, and its matrix has the square root of the
  condition number. It is applied to the iterate with the smallest dual residual and
  complementarity, which is now kept as well as the iterate with the smallest largest error.
  On a set of 56 moment-SOS relaxations: 32 → 34 solved to 1e-8, six more accurate
  (attr_henon_d10_K 2.9e-7 → 1.2e-8, roa_dint_d8_K 3.9e-8 → 3.7e-9, roa_vdp_d12_I
  7.7e-8 → 1.1e-8), none less; SDPLIB unchanged. On a user's relaxations of structural
  optimization (blocks of order 450): primal residual 2e-7 → 7e-14, largest error
  8e-8 → 6e-11 on one, and on another a re-solve by the other method that took two thirds of
  the time is no longer needed. `-nopolishr` turns it off. Where the error is in the
  complementarity itself the result stays at reduced accuracy; `-prec dd` is the way there.

- **The same correction is a stopping test in the self-dual embedding:** once the dual
  residual and the complementarity are within half the tolerance and only the primal residual
  is not, the current iterate is corrected and the run stops if the corrected point passes
  all six error measures. The iterations this replaces could not repair the primal residual.
- **The automatic choice of the first method counts the assembly of the Schur complement.**
  The embedding ran first when its extra dense work was small next to the factorization of
  the Schur complement. On problems with a few thousand constraints that have wide rows on
  blocks of order 250–800 the assembly, not the factorization, is most of an iteration; they
  started with the infeasible-start method, which fails on problems without an interior, and
  then ran the embedding as well. The embedding is now also first when its extra work is at
  most half of factorization plus assembly (`-hsdfirstasm`). On eight such relaxations (two
  threads) the total time went from 1487 to 738 s, with two more solved to 1e-8; where the
  first method had succeeded the solve is 10–30 % slower. Max-cut, graph-partitioning and the
  other problems with large dense blocks and a cheap Schur complement are not affected.
- **The advice at the end of the log is written in the caller's syntax and names the way to a
  guaranteed bound.** The last lines of a log say how to get more accuracy (`acc high`, the
  high-precision solver `prec dd`) and now also how to get a guaranteed bound on the optimal
  value from a certified feasible point (`bound`, and `certify` for the rigorous check).
  Each option is shown as it is typed where the solve was started: `-acc high` on the command
  line, `"acc": "high"` in the `options` of the Python function that was called,
  `acc="high"` in CVXPY's `prob.solve`, `acc = "high"` as a keyword in Julia,
  `set_attribute(model, "acc", "high")` in JuMP, `opts.acc = 'high'` in MATLAB, the strings
  `"-acc", "high"` in C. What was already asked for is not advised again, and a
  high-precision log ends with the lines on the bound. A second-order cone problem solved
  through an interface now ends with the summary block as well.
- **Fixes:** MATLAB/Octave `brisk_sedumi` failed on a problem with a single constraint; a
  SeDuMi or CBF file without PSD blocks solved with `-conesolver 0` printed its summary twice.

## Version 1.3.1 (October 2026)

- **The high-precision solver applies the reductions of the double-precision solver**
  (`-prec dd`, `qd`, digits): sign symmetries; permutation symmetries (the constraints of an
  orbit summed, blocks that the group exchanges reduced to one), with the group verified
  entry by entry in the working precision; the block diagonalisation of the data's matrix
  \*-algebra, with a basis computed and verified in the working precision; and the chordal
  decomposition where the banded factorization of the Schur complement can use it. The
  solution is mapped back and measured on the problem as read. One core, `-prec dd`:
  roa_brockett_d4_I 0.60 → 0.07 s, roa_acrobot_d4_I 23.9 → 5.2 s, neu2 (3,003 constraints, a
  block of 252, group of order 120) 487 → 6 s, cnhil10 more than 600 → 0.3 s, a banded
  max-cut problem with n = 1000 87 → 5.6 s. Not with `-bound`, `-fom`, `-lralm`.
- **The double-double endgame handles free variables** (a bordered system instead of two
  nonnegative columns without an interior), factors the Schur complement as a sparse matrix
  and skips linearly dependent rows. The 69 option-pricing relaxations of the POEMA database:
  38 optimal in 1.2.1 → 68, in less time (5.1 against 7.4 s in all).
- **The double-double continuation of 1.3 is removed.** It ran the high-precision solver from
  the returned point of a small problem and could double the time of a run; the endgame now
  does that work inside the solve.
- **The log ends with the ways to more accuracy** after a solve in standard precision:
  `-acc high` (tolerance 1e-10, still in double precision arithmetic) and the high-precision
  solver `-prec dd`, `-prec qd`, `-prec <digits>`.
- **Schur formation:** the choice between the two row-product paths follows the block order
  (suggested by a user and fitted on his relaxations of structural optimization): 5–20 %
  faster formation on blocks of order 550–800 whose staircase holds 67–86 % of the products.
- **Fix: the symmetry search missed every permutation symmetry of a problem whose objective
  matrix has off-diagonal entries of the same value throughout** (the theta problem of a
  vertex-transitive graph, C = J): the initial colouring of the indices depended on their
  order.
- **Fix: bound OpenMP threads all ran on one core on the command line** (`OMP_PROC_BIND`,
  `OMP_PLACES`; reported with an analysis and a patch by a user). The solver restarts itself
  once so that OpenBLAS starts with one thread, and the restarted process inherited the
  affinity mask of the initial thread, which the OpenMP runtime had already pinned. The mask
  is now restored to the runtime's places before the restart (theta6 with six bound threads:
  9.7 → 4.7 s in the report). The interfaces were not affected. `make bindtest` checks it.

## Version 1.3 (October 2026)

- **A summary at the end of every log:** the problem as read, every preprocessing step
  applied and what it changed, the engine and method, every re-solve, method switch,
  double-double endgame or continuation, crossover and bound, the time, the number of
  threads, the six DIMACS errors, and the closing line `Solved to DIMACS error e` (or the infeasibility verdict with
  whether the returned point is a certificate). The `status:` line and the exit codes are
  unchanged; `BRISK_NOSUMMARY=1` leaves the block out.
- **Double-double continuation of small problems** (removed in 1.3.1)**:** a problem with at most 1,200 constraints
  that ends short of the tolerance in double precision continues in double-double from its
  point, within twice the double time; the double result stands when that does not do
  better. The 69 option-pricing relaxations of the POEMA database: 38 optimal → 69; on SDPLIB
  hinf12, hinf13, gpp124-1, qap6, qap7 and qap8 become optimal. `-nodd` turns it off.
- **The matrix-free interior-point method is tried first automatically** (`-mftry`, default 1)
  on problems with one to four dense blocks whose Schur factorization is estimated at 3.5
  matrix-free solves or more — also when it would not fit in memory, where the first-order
  engine is the fallback. The attempt is limited in work; unless it ends optimal the standard
  method follows, from the attempt's iterate when it got near. Truss topology and vibration
  problems (POEMA): tru11 108 → 3 s, vib11 123 → 16 s, tru13 905 → 305 s, tru15 and vib15
  (25,200 constraints) and tru17 (41,616) solved where 1.2 ran out of memory; an SOS
  relaxation with 46,375 constraints 24 s instead of a time limit.
- **The matrix-free method itself** (`-mfipm 1`): the preconditioner takes the scaling matrix
  scaled or unscaled per block; truss and vibration problems 3–4× faster than in 1.2. The
  hybrid `-mfipm 2` hands off at a merit of 1e-3 (`-mfhand`).
- **A faster embedding step:** the second-order term re-evaluated at the direction that is
  taken, and a longer step after kept passes (`-hsdhoc 4`, `-hsdnb 0.3`); dense moment
  relaxations take 6–7 iterations where 1.2 took 7–10.
- **Linear programs:** a faster sparse factorization (relaxed supernodes, a left-looking
  update); Netlib 25 % faster.
- **Schur formation through precomputed positions** (contributed by a user): 17–20 % off the
  assembly where it is the cost (Bex2_1_5, neosfbr25).
- **Memory limits follow the machine:** the limit of the process's own cgroup, `ulimit -v`;
  the internal caps grow with the memory above 8 GB; `-v` prints the limit.
- **Infeasibility verdicts** say whether the returned point is a Farkas ray of the data as
  read (after a facial reduction it need not be: a weakly infeasible problem has no ray).
- **Fixes:** a double free in the symmetry reduction and a heap overflow in the free-variable
  elimination (reported with patches by a user); the algebra block diagonalisation no longer
  applies where it would make the data dense; a late dense Schur complement restarts with the
  first-order engine instead of exiting with advice; the first-order engine's single-precision
  phase no longer stalls; a segmentation fault where neither the Schur complement nor the
  Gram factor fits; the `-fomssnafter` text.

## Version 1.2.1 (October 2026)

- The log states the number of threads the solver uses: a line `Number of threads: k` after
  the solver's banner, in every interface (not in quiet mode). The cone solver and the solver
  for linear programs are sequential and report 1.

## Version 1.2 (October 2026)

- **Second-order cones.** Problems with free variables, linear variables, second-order and
  rotated second-order cones, with or without semidefinite blocks. A problem without
  semidefinite blocks goes to a cone solver of its own (homogeneous model, sparse
  factorization of the reduced KKT system), which returns certificates of infeasibility.
  - Input in SeDuMi format: `brisk problem.mat` (a MAT-file with `A`, `b`, `c`, `K`),
    `brisk_solve_sedumi` in the C library, `brisk.solve_sedumi` in Python,
    `Brisk.solve_sedumi` in Julia, `brisk_sedumi` with `K.q` and `K.r` in MATLAB and Octave.
  - Second-order and rotated second-order cone constraints in CVXPY and JuMP models.
  - CBF files (the format of CBLIB), plain or gzipped: `brisk problem.cbf`. Integer variables
    are relaxed with a message.
- **Linear programs from MPS files**: `brisk problem.mps` (free or fixed format). A presolve
  with an exact postsolve and an interior-point method for linear programs (upper bounds as
  bounds, a nested dissection ordering). Options `-lppresolve`, `-lpmethod`, `-lpcorr`. The
  solution is an interior one: there is no crossover to a basis.
- **Matrix-free interior-point method** (`-mfipm 1`): no projection of the primal direction, a
  step length without cancellation. Truss topology problems with tens of thousands of
  constraints are solved in minutes on one core (the largest ones to an accuracy of 1e-7
  rather than 1e-8). The method is not chosen automatically. `-mfipm 2` is a hybrid: the matrix-free method
  while its conjugate gradients are cheap, then the standard method from its iterate.
- **Semidefinite programs**: the Schur complement of constraint matrices with small row
  supports is assembled only where later constraints read it (dense moment relaxations
  without symmetry: 10-40 % faster); a symmetry reduction is also applied when the estimated
  work shrinks by `symmin` (few blocks, many constraints).
- **Rigorous bounds for large problems**: `-certify` beyond 6,000 constraints (sparse Gram
  matrix); `-boundanchor <file>`.
- **MATLAB on Linux**: `build_brisk_mex` links MATLAB's own BLAS and LAPACK.
- When the homogeneous embedding stops early and the solve continues with the standard
  method, the log says why.
- README: a section on the reproducibility of runs (a threaded BLAS can change the last digits
  from run to run; BRISK's own code path is deterministic).

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
