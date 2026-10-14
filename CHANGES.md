# BRISK change log

## Version 1.4.4 (October 2026)

- **Second-order cones next to semidefinite blocks are solved as cones** (`-soc`, `-socmin`). In a
  problem with semidefinite blocks a cone of dimension q used to be solved as a q x q arrow block
  (O(q³) work an iteration and q² unknowns). The self-dual embedding now treats every such cone, and
  every semidefinite block whose data are arrow matrices (a norm constraint written as a matrix
  inequality), as a second-order cone: Nesterov-Todd scaling W = β(2vvᵀ − J), the cone's term of the
  Schur complement β²(A Aᵀ + rank two) formed on the threads, the step length from a quadratic, X
  returned as the rank-two completion of the cone point. A run that ends worse than reduced accuracy
  is repeated with the cones as semidefinite blocks; `-soc 0` keeps the old treatment, `-socmin` sets
  the smallest order. Four threads, before → now (MOSEK 11 for comparison): a synthetic family with
  two cones of dimension 1024, 28.1 → 0.24 s (0.05 s); nearest correlation matrices of order 100,
  30.7 → 5.4 s (21.5 s); noisy matrix completion of order 60, 3.0 → 0.45 s (1.1 s); PhaseLift with
  n = 40, 0.68 → 0.31 s (0.37 s); a robust portfolio with 100 assets, 6.1 → 5.8 s (8.6 s). About
  half the iterations; DIMACS filter48_socp 2.0 → 1.1 s. The errors of large cones on the problem as
  read are computed in closed form (λ_min of an arrow matrix, of the rank-two completion) instead of
  by a dense factorization: two cones of 8192, 15.1 → 4.3 s at one thread.
- **More accurate on ill-posed problems.** The self-dual embedding eliminates the homogenizing
  variable τ through a scalar equation whose terms grow with X; on problems whose solutions are large
  (no strictly feasible point, or nearly so) that equation lost its digits late in the solve. Its
  column is now formed from X, Z and the dual residual alone (the substitution dy = dy' + (dτ/τ) y),
  each iteration in whichever of the two forms has the smaller estimated error (and no longer in the
  first once a step in it leaves the primal residual above twice its prediction). One thread, against
  1.4.3: SDPLIB OPTIMAL 68 → 70 of 92 (hinf1, hinf6); 56 occupation-measure relaxations OPTIMAL 32 → 35,
  numerical difficulties 8 → 7; 69 option-pricing relaxations, numerical difficulties 1 → 0; a user's 23
  structural-optimization relaxations (two threads) OPTIMAL 14 → 19, numerical difficulties 3 → 1, total
  time 2376 → 2016 s. `BRISK_HSDSHIFT=0` turns it off.
- **Facial reduction: the reduced solve is returned when the duals of the removed constraints
  cannot be rebuilt** (the dual optimum of the problem as read is not attained, or a long chain of
  reductions loses them). If the reduced solve is good in itself, its X (feasible for the problem as
  read) is returned with the optimal value in [b'y, ⟨C,X⟩] and a note that y is not dual feasible for
  the removed constraints; before, the problem was solved twice more (the reduced problem by the
  embedding, then the problem without the reduction) and the better of the answers kept. On a user's
  structural-optimization relaxation: numerical difficulties in 205 s → OPTIMAL in 4.2 s, the value
  within 1.2e-8 of the double-double one. `BRISK_FRKEEP=0`: the re-solves as before.
- **The embedding's stall rules follow its path.** In exact arithmetic its residuals fall as θR₀
  (θ the product of the 1 − α(1 − σ)); once a residual exceeds twice that, the progress of the
  current iterate no longer keeps a run going. A frame relaxation without an interior: 33 → 19
  iterations, the same answer; no status changes on SDPLIB, the occupation measures, option pricing
  and the sphere relaxations.
- **The same result at any thread count and under any load, also with several blocks of which one
  is large.** The inverse of Z of a large block was computed by one routine at one thread and by
  another inside the parallel loop over blocks, and the single-precision frame changes of the NT
  scaling were chosen by whether a parallel region was active: now the same arithmetic in both cases.
  The symmetry search's budget is counted in units of work instead of seconds, so a busy machine no
  longer stops it at another point.
- **Fixes.** The chordal conversion's density test (and three other tests of the pattern density of
  large blocks) counted an entry once for every constraint that touches it; they count the union of
  the patterns now (an AC-OPF relaxation with the line limits as cones: 4.8 → 0.74 s). A data race in
  the threaded Schur assembly (its tile settings read lazily by several threads; found with
  ThreadSanitizer). The library's copy of X did not check its allocations: under a memory cap it
  crashed after the solve; now it reports that X cannot be returned and returns y and the objectives.
- **macOS on Intel: the post-link check no longer removes good binaries.** Clang places switch jump
  tables inside the code on x86-64, and the check took those bytes, which no disassembler decodes,
  for a damaged link and deleted `brisk` and `libbrisk.dylib`. It now compares the linked file with
  the objects that went into it: a word counts as damage only if the compiler did not write it.
  The MATLAB build, which keeps no objects, is checked only on Apple silicon. Thanks to the user who
  found the cause.

## Version 1.4.3 (October 2026)

- **macOS: the same result from run to run and for any thread count.** Two properties of Apple's
  Accelerate made runs of the same problem differ in the last digits, which the interior-point
  method can amplify into other iteration counts:
  - before macOS 15, Accelerate ran threads of its own even under `-threads 1`; BRISK now sets
    `VECLIB_MAXIMUM_THREADS=1` in its process unless the variable is set (on macOS 15 and later it
    asks Accelerate for one thread, as before; `BRISK_BLASMT=1` keeps Accelerate's threads);
  - Accelerate rounds some vector operations differently depending on where the vectors lie in
    memory; BRISK now aligns its arrays to 256 bytes on macOS and keeps its work vectors in the
    same place whatever the thread count.

  Measured on an Apple M3 under macOS 14: before, 3–5 % of the runs of SDPLIB theta4 at one thread
  and 1–8 % of those of gpp124-1 at seven threads ended differently (other iteration counts, once
  with a reduced-accuracy status), and one thread gave other digits than several on theta3 and
  theta6; now every run gives the same result, at 1 to 8 threads. SDPLIB at one thread about 7 %
  faster, at four threads unchanged.
- **The same result when a problem is solved again in the same process** (Python, Julia, MATLAB,
  the C library), on every platform. Two randomized steps of the presolve, the detection of
  low-rank constraint matrices and the block diagonalization of the symmetry reduction, now start
  from their seeds at each solve; before, a second solve of the same problem could take other
  low-rank factors and end with other digits or, rarely, another status. Results from the command
  line are unchanged.
- **macOS build: the linked binaries are checked.** The linker of the command-line tools of Xcode
  15.0 corrupted one instruction of the solver (a crash with "illegal instruction" in the cone
  solver). With that linker the build now uses Apple's classic linker, and on macOS every link
  (the command-line program, the library, the MATLAB MEX file) is followed by a check that
  disassembles the result (`tools/link_check.sh`); a damaged binary is linked again with the
  classic linker, or removed with a message. `BRISK_LINKCHECK=0` turns the check off. Updating
  the command-line tools is recommended.

## Version 1.4.2 (October 2026)

- **A Burer-Monteiro solver for large sparse problems with low-rank solutions** (`-bm`). X = R Rᵀ
  with R of n × r; every operation is a product of a sparse matrix on the data's pattern with the
  factor, and nothing of size n² is formed. It runs by default, before the interior-point method,
  on two kinds of problems with one semidefinite block:
  - every constraint fixes a diagonal entry and the order is 500 or more (max-cut and the other
    relaxations over {−1,1}ⁿ): coordinate steps on the product of spheres, then a Riemannian trust
    region, until the gap to a lower bound is within the tolerance. The bound comes from the dual
    matrix Z made positive semidefinite by a shift δ that a sparse Cholesky factorization of Z + δI
    confirms (where that factorization would fill, the bound is reported as an estimate);
  - every constraint fixes one entry (matrix completion) and the interior-point method's Schur
    complement does not fit in memory: an augmented Lagrangian method with L-BFGS and an exact line
    search.

  If it misses the tolerance within half the time limit, the standard method follows; `-bm 0` turns
  it off, `-bm 1` uses it for any problem with one semidefinite block. Four threads, the default
  tolerance 1e-8, the same optimal values: maxG55 (order 5000) 46 → 1.5 s, maxG60 (7000) 79 → 3.3 s,
  G55mc 39 → 1.8 s, checker_1.5 23.5 → 0.9 s, ice_2.0 119 → 4.4 s, p_auss2_3.0 127 → 8.9 s; a
  matrix-completion problem with 59,937 constraints 53 → 0.4 s. Against LoRADS (its CPU release, on
  the classes of its paper; BRISK at `-tol 1e-6` with a certified bound): matrix completion 4-9 times
  faster (1.5 million constraints in 9.6 s, where LoRADS did not finish in 900 s), max-cut on Gset
  graphs 2.3 times faster in the geometric mean, Mittelmann's checker_1.5, ice_2.0 and p_auss2_3.0
  6-41 times faster, and planar graphs of 65,536 to 262,144 nodes solved in 5 to 14 minutes.
- **`-blkscale 1`**: a diagonal congruence scaling of the semidefinite blocks (X = D X' D, D from
  Ruiz iterations on the data, powers of two), off by default.

## Version 1.4.1 (October 2026)

- **macOS: the dense Cholesky factorization runs as a task graph again.** The OpenMP runtime that
  BRISK bundles for compilers without one (Apple's clang on macOS, and the MATLAB MEX file there)
  now supports task dependences (`depend` clauses). The macOS build of 1.4 used the
  factorization's panel loop instead (the updates of each panel in parallel, the panels one after
  another); with the task graph the panels overlap again. Measured on Linux with the bundled
  runtime, four threads: factorizations of order 3000 to 11000 take 9–22 % less time than with
  the panel loop and are within 10 % of GCC's runtime, with the same factor bit for bit; nine
  problems with large dense blocks take 0.95 of the panel loop's time in the geometric mean. Builds
  with GCC (Linux) are unchanged.

## Version 1.4 (October 2026)

Faster on large dense blocks and on moment relaxations of polynomial optimization, the same
result for any thread count, and a thread count that follows the free cores of a busy machine.
On 121 problems taken from every class of the development benchmark (SDPLIB, Mittelmann's
benchmark, max-cut and theta families, moment and sum-of-squares relaxations, structural
design, truss and vibration problems, AC optimal power flow, DIMACS, second-order cone and
linear programs; four threads, one machine) 1.4 takes 0.76 of 1.3.2's time in the geometric
mean and 0.46 in total (0.40 of MOSEK 11's in the geometric mean), and solves 119 of the 121
against 117 (MOSEK: 111). No class is slower: the classes near parity, timed four times, are
within 3 % of 1.3.2 in their medians. Five problems end with a better status, one (qap9, at
1.1e-8 against the tolerance 1e-8) with a worse one.

- **Large blocks whose dual matrix stays sparse: dense X, sparse Z.** In max-cut, box-QP and
  structural-design relaxations the dual matrix Z = C − A'(y) keeps the pattern of the data while
  X is dense. In the HKM direction the inverse of Z, its products and the dual step length now
  come from a sparse Cholesky factor of Z under a fill-reducing ordering; X keeps its dense
  factor. Automatic for blocks of order 200 or more whose factor has at most 0.1 n² nonzeros
  (`-zsparse`, `-zsparsefill`).
- **One large block uses all cores, and the result does not depend on how many there are.** The
  dense Cholesky factorization runs as a task graph of tiles that grow with the order; products
  and triangular solves with many right-hand sides, the passes over the n² entries of a block,
  the reduction to tridiagonal form and the dense Schur solves with one right-hand side run on
  the threads. Every sum is taken in a fixed order, so the iterates are the same bits for any
  thread count, also when the count changes during a solve (checked at one to four threads and
  with a count changing from iteration to iteration, on every path:
  dense, sparse and envelope Schur complements, chordal and symmetry reductions, the first-order,
  matrix-free, cone, LP, low-rank and high-precision solvers; decisions taken by the clock, such
  as the time budget of the symmetry search, still depend on the machine's speed and load). With
  the sparse Z above, at four
  threads: maxG32 (order 2000) 21.4 → 3.4 s, G40mc 25.9 → 5.9 s, checker_1.5 143 → 21 s,
  theta12 145 → 76 s, vibra5 506 → 149 s.
- **Moment relaxations of polynomial optimization.** On a user's benchmark of 72 relaxations of
  random polynomial problems over a ball, a box and an ellipsoid (four threads), 1.3.2 took
  1828 s and MOSEK 11 1546 s; 1.4 takes 575 s with fewer iterations than MOSEK (477 against
  532), all solved to 1e-8. What changed:
  - the Schur complement of a localizing block is assembled from the much smaller one of its
    classes of entries (entries that appear in the same constraints with the same values;
    automatic, `-cls`);
  - a relaxation whose largest block is a moment matrix with localizing blocks is solved in the
    self-dual embedding with the NT direction from the centred start X = 0.3 I, Z = I, y = 0
    (`-hsddir`, `-init 3`);
  - the Schur complement is assembled in single precision while the residuals are above 1e-3
    (it was already factored in single precision, with conjugate-gradient refinement), and
    factored in double precision once the residuals and the gap are below 1e-6;
  - at the end, a longer last step when the predicted measures meet the tolerance, and the
    primal residual corrected in the metric of the last iteration with the factor at hand,
    instead of further iterations;
  - a relaxation over an ellipsoid is rewritten in coordinates in which the ellipsoid is the
    unit ball, solved, and its solution mapped back and checked on the problem as given (the
    user's 24 ellipsoid relaxations: 170 → 139 iterations; MOSEK 162).
- **The solve stops on the measures it is judged by.** Once the best iterate meets the tolerance
  in the DIMACS measures from which the status is set, with a factor two to spare, and an
  iteration does not improve on it, the solve returns it. The loop's own measures are scaled
  differently (by the 2-norms of b and C instead of the 1-norms) and could keep it iterating at
  their floor: thetaG51 now takes 14 iterations whatever the last bits of b (13–19 before, with
  b perturbed by a few units in the last place).
- **The thread count follows the free cores; `-threads k` is an upper limit.** On Linux the
  interior-point solve uses fewer threads while other processes occupy its cores, and more again
  when they are free; a cpuset or a container counts only its own cores. On four cores with
  three taken, 29 test problems went from 147 to 61 s. `BRISK_NOBUSY=1` keeps exactly k threads.
- **macOS 15 and later: Accelerate on one thread under BRISK's own.** Accelerate threaded its own
  calls (200–400 % CPU at `-threads 1`, and different iterates from run to run).
- **The end of the solve, option by option.** OPTIONS.md names, for each line at the end of the
  log, the flag that turns it off and how it is given in each interface; the texts of
  `nopolish`, `nopolishr` and `nopolishx` corrected.
- **Fixed:** on large problems under a memory limit (`ulimit -v`) a segmentation fault where
  "out of memory" was meant.

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
