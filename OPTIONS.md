# BRISK options

Every option of the command line can be set from every interface. This file has three parts:

1. how to pass options in each interface;
2. the options most users need, explained;
3. the full list, generated from the solver itself (`./brisk` without arguments prints it too).

Defaults are automatic choices, so most problems need no options at all.

## 1. Passing options

An option `-name value` of the command line is the key `name` with that value. A flag such as
`-nohsd` takes `true` (or `1`); `false` (or `0`) leaves it out. An underscore in a key stands
for a dash: `certify_y` is `-certify-y`.

| interface | example: chordal decomposition off, high accuracy, 60 s, 2 threads |
|---|---|
| command line | `./brisk problem.dat-s -chordal 0 -acc high -timelimit 60 -threads 2` |
| Python, low level | `brisk.solve_file("problem.dat-s", options={"chordal": 0, "acc": "high", "timelimit": 60, "threads": 2})` |
| Python, CVXPY | `prob.solve(solver=brisk.BRISK(), chordal=0, acc="high", timelimit=60, threads=2)` |
| Julia, JuMP | `set_attribute(model, "chordal", 0); set_attribute(model, "acc", "high")`; time limit and threads also through `set_time_limit_sec` and `MOI.NumberOfThreads()` |
| Julia, low level | `Brisk.solve_sdpa("problem.dat-s"; chordal = 0, acc = "high", timelimit = 60)` |
| MATLAB / Octave | `opts.chordal = 0; opts.acc = 'high'; opts.timelimit = 60; opts.threads = 2; brisk_sdpa('problem.dat-s', opts)` (same `opts` for `brisk_sedumi`) |

Further forms:
- Python's `options` also takes a list (`["-chordal", "0"]`) or a string (`"-chordal 0"`).
- CVXPY's `options=[...]` passes raw strings.
- MATLAB's `opts.args` takes a string or a cell array of raw options.

An unknown option or a bad value is an error ("unknown option ..."), never silently ignored.

Interface-specific settings, handled by the interface rather than the solver:

| interface | settings |
|---|---|
| Python | `verbose=` (low level), CVXPY's `verbose`, `form` (`"auto"`/`"kernel"`/`"image"`), `soc`, `write_sdpa`, `inaccurate_tol`, `bound` (`"p"`/`"d"`/`"sos"`), `certify` |
| Julia | `"output"` (`:julia`, `:stdout`, `:silent`), `"form"`, `"write_sdpa"`, `"bound"`, `set_silent` |
| MATLAB | `opts.verbose` (0 quiet, 1 summary, 2 iterations), `opts.bound` (`'p'`, `'d'`, `'sos'`), `opts.sym` (also `true`/`false`) |

Output files (`-x`, `-y`, `-z`) and the stand-alone certificate checks (`-certify-x`,
`-certify-y`) belong to the command line. The interfaces return the solution in memory.

## 2. The options most users need

### Accuracy, limits, output

| option | meaning | default |
|---|---|---|
| `acc` | `low`, `default` or `high`: tolerance 1e-6 / 1e-8 / 1e-10 (at `low`, the first-order engine may run first: `fomrace`) | `default` |
| `tol` | the tolerance itself (overrides `acc`) | 1e-8 |
| `maxit` | iteration limit | 100 |
| `timelimit` | wall-clock limit in seconds | none |
| `threads` | OpenMP threads. With the default, tiny problems use one thread and (Linux) the interior-point solve uses only the cores that are free when it starts. | `OMP_NUM_THREADS`, else all cores |
| `q`, `v` | quiet, verbose (command line) | |
| `returnx` | library calls (MATLAB, Python, Julia): the dense X of a chordal-decomposed block is returned unless that needs more than 30% of the memory (`-1`), never (`0`; the point is verified on the cliques, fastest), always (`1`). A block that is not returned is empty (`None` in Python). | `-1` |

### Presolve

| option | meaning | default |
|---|---|---|
| `chordal` | chordal decomposition of sparse blocks: `-1` automatic (when it lowers the cost per iteration), `0` off, `1` force | `-1` |
| `cliquemax`, `chordalmin` | largest clique accepted; smallest block tried | 160; 100 |
| `sym` | exact symmetry reduction: `auto` (sign symmetries and permutation automorphisms found and used), `none`, or a file of generators | `auto` |
| `symmin` | apply the reduction only when it shrinks the problem by this factor | 1.5 |
| `symtime`, `symnodes` | caps of the automatic search | 20 s, 500 nodes |
| `symalg` | block structure shared by all the data matrices of an SDP block. The matrix *-algebra that C and the constraint matrices of the block generate is block-diagonalised numerically: the block splits into its components, and equal copies are kept once. This finds a symmetry group that leaves every constraint matrix unchanged (any such group, not only permutations, also in a rotated basis), commuting data (the block becomes an LP block) and hidden direct sums. It does not find a symmetry that maps the constraints onto each other; for permutations `sym` does that. −1 auto: blocks up to `symalgmax`, applied when the gain is ≥ `symmin`; an O(nnz) test first skips the blocks it proves irreducible. 0 off; 1 every block, any gain. `sym none` turns it off too. | −1 |
| `symalgmax` | largest block `symalg −1` checks. The check costs one eigendecomposition and two n × n products: 0.2 s at n = 1000. | 1000 |
| `nofr` / `fr` | facial reduction off / whatever its depth | on (skipped at depth ≥ 2 when it saves little) |
| `freeelim` | eliminate split free variables (kernel-form SOS): `-1` auto, `0` off, `1` on | `-1` |
| `dualize` | solve the dual form when its Schur complement is at most half the size: `-1` auto, `0` off, `1` force | `-1` |
| `tracebound` | trace bound with the free elimination | `-1` auto |

### Method

| option | meaning | default |
|---|---|---|
| `hsd` / `nohsd` | force / forbid the homogeneous self-dual embedding | automatic, each method the other's fallback |
| `hsdfirst` | automatic order: `1` the embedding first where it is cheap | 1 |
| `dir` | search direction `auto`, `hkm`, `nt` | `auto` |
| `dual` | sparse dual-scaling method (falls back to the primal-dual method unless certified) | off |
| `fom` | first-order engine: `-1` automatic (dense Schur complement does not fit), `0` off, `1` force | `-1` |
| `fomrace` | at tolerance ≥ 1e-6 (`acc low`): share of the estimated interior-point time the first-order engine gets *first* on few-large-dense-block problems with a sparse Gram matrix A A' (theta-type; interior-point estimate ≥ 20 s); the interior-point method follows if it misses the tolerance; `0` off | 0.2 |
| `mfipm` | matrix-free interior-point method (problems with a low-rank optimal side) | 0 |
| `lralm` | low-rank augmented Lagrangian (experimental; very large sparse SDPs such as AC-OPF) | 0 |
| `crossover` | SDP crossover after the solve: `0` off, `1` on, `-1` when cheap | 0 |

### High precision

| option | meaning | default |
|---|---|---|
| `prec` | `dd` (double-double, about 32 digits), `qd` (quad-double, about 64 digits) or a number of decimal digits (variable precision). The interior-point method runs in that precision; with `fom = 1` the augmented Lagrangian / semismooth Newton method, with `lralm = 1` the low-rank augmented Lagrangian method. The data of a file are read exactly as written. Blocks whose data are block diagonal after a permutation are split; a presolve in the working precision removes split free variables and simple faces; the solve starts from a double solve by the default method; the result and its errors are those of the problem as read. | off |
| `hpext` | interior-point method: when the precision is exhausted short of the tolerance, the solve continues in the next precision (dd, qd, then more bits), at most this many times and while that gains a digit; 0: off. The tolerance and the digits returned stay those of `prec` | 2 |
| `hptol` | tolerance of the high-precision solve | the solve aims at 1e-24 (dd), 1e-48 (qd), 1e-77 (100 digits) and a result within 1e-19, 1e-38, 1e-62 counts as optimal; `fom` / `lralm` aim at the latter |

Problems without an interior that the presolve does not reduce reach about half the digits
of a precision; the extension levels (`hpext`) then continue in the next one. The status
`PRECISION LIMIT` means that the solve ended above the tolerance all the same.
The command line prints the objective with all digits and `-x`, `-y`, `-z` write them; in
Python `brisk.hp_solution()` and in Julia `Brisk.hp_solution()` return the solution with all
digits (the usual result holds it rounded to doubles).

### Bounds and certificates

| option | meaning | default |
|---|---|---|
| `bound` | `p`: best feasible X (upper bound); `d`: best feasible y (lower bound) | off |
| `certify` | rigorous check of the bound (interval arithmetic) | off |
| `bound` with `prec` | the certificate is built and checked in high precision on the problem as read; the rigorous bound has all the digits of `prec`, and the certificate is returned (and written by `-x` / `-y`) with all the digits of its check. `certify` is then implied. `-certify-x` / `-certify-y` with `prec` check a given certificate | – |
| `boundtol`, `boundmargin` | residual and eigenvalue margin required of a certificate | 1e-10, 1e-13 |

## 3. Full list

Generated by `tools/gen_options_md.sh` from `./brisk` (do not edit by hand).

<!-- BEGIN GENERATED -->
```text
usage: brisk problem.dat-s [options]
output
  -q | -v          quiet | verbose   (Ctrl-C: stop and return the current point; twice: abort)
  -y <file>        write y (SDPA primal x = -y), one value per line, original numbering
  -x <file>        write X of the original problem (SDPA sparse-like: block i j value, upper triangle)
  -z <file>        write Z = C - A'y of the original problem, same format
termination
  -acc <level>     accuracy: low (tol 1e-6), default (1e-8) or high (1e-10). low also scales the
                   re-solve thresholds (10 tol, 100 tol after the embedding) and the reduced-
                   accuracy class (100 tol); high keeps the default ones. -tol/-retryacc override it
  -tol <t>         relative gap / infeasibility tolerance (1e-8)
  -maxit <k>       iteration limit (100)
  -timelimit <s>   wall-clock limit in seconds (0 = none)
  -bound p|d       return the best GUARANTEED bound instead of the best-balanced pair:
                   p: a feasible X of (P) (A(X) = b to -boundtol, X > 0 with a margin), <C,X> an
                   upper bound on the optimal value even when the solve does not converge; d: a
                   feasible y (Z = C - A'y > 0 verified), b'y a lower bound. (P) is the SDPA dual
                   (min <C,X>, C = -F0): SOS certificates from SeDuMi/GloptiPoly are -bound p;
                   relaxations of minimisation problems take the lower bound from -bound d.
                   The certificate replaces its side of the returned pair (-x / -y write it);
                   -certify makes the bound rigorous. See README (bound mode).
  -boundtol <t>    relative residual required of a -bound p certificate (1e-10)
  -boundmargin <e> eigenvalue margin of the certificate, relative to 1 + max diagonal; at least
                   8 n^2 u per block of order n, what a rigorous check needs (1e-13)
  -boundtrack <t>  residual below which iterates are tracked as candidates (100 x -boundtol)
  -certify         with -bound, check the certificate rigorously on the
                   problem as read (the file's decimals enclosed in intervals, directed rounding,
                   a verified lambda_min) and print the rigorous bound
  -certify-x <f>   no solve: rigorously check the X in f (-x format: block i j value) as a
  -certify-y <f>   ... p certificate, or the y in f (-y format) as a d certificate; exit code 0
                   when certified, 20 when not (certificates of any solver in these formats)
  -stallwin <k>    non-improving iterations tolerated once below 1e-6 (5)
high precision
  -prec <p>        solve in high precision: dd (double-double, about 32 digits), qd (quad-double,
                   about 64 digits) or a number of decimal digits (variable precision; up to 31
                   digits is dd, up to 63 qd). The data are read exactly as written in the file.
                   Blocks whose data are block diagonal after a permutation are split. A
                   presolve in the working precision plus guard digits removes split free
                   variables and simple faces (rank-one, diagonal, LP certificates); the result
                   and its errors are those of the problem as read. The solve starts from a
                   double solve by the default method and goes up a ladder dd -> qd -> variable
                   precision; the interior-point method ends quadratically (centring steps, then
                   one long step per level). With -fom 1 the augmented Lagrangian / semismooth
                   Newton method runs in high precision instead, with -lralm 1 the low-rank
                   (X = R R') augmented Lagrangian method. -x/-y/-z write all the digits.
                   Status PRECISION LIMIT: the accuracy the precision reaches on the problem
                   With -bound p|d the certificate of that side is built in high precision on the
                   problem as read and checked rigorously (in the precision of -prec plus guard
                   limbs, with a-priori error bounds of the arithmetic); the rigorous bound is
                   printed with all digits, and -x / -y write the certificate with all the digits
                   of its check. -certify-x / -certify-y with -prec check a given certificate
                   the same way (p needs an interior of (P), d one of (D))
  -hpext <k>       interior-point method: when the precision is exhausted short of the tolerance
                   (problems without an interior that the presolve does not reduce reach about
                   half the digits), the solve continues in the next precision (dd -> qd -> more
                   bits), at most k times and while that gains a digit (2; 0: off). The tolerance
                   and the digits written stay those of -prec
  -hptol <t>       tolerance of the high-precision solve. Default: the solve aims at 2^(-0.75 bits)
                   (1e-24 for dd, 1e-48 for qd, 1e-77 for 100 digits) and a result within
                   2^(-0.6 bits) (1e-19, 1e-38, 1e-62) counts as OPTIMAL; -fom / -lralm aim at the latter
method
  -dual            dual-scaling method (DSDP-style: potential reduction, correctors, verified
                   certificates); falls back to the primal-dual method unless certified
  -hsd | -nohsd    force | forbid the self-dual embedding (default: automatic - the embedding first
                   on most problems, the infeasible start first on large dense blocks with a cheap
                   Schur complement; each method is the other's fallback)
  -hsdfirst <0|1>  automatic method: the embedding first when its extra dense work is small or the
                   problem tiny (1; 0: standard first); many split free pairs and the chordal moment
                   form use the embedding either way
  -hsdfirstc <c>   ... i.e. when c * sum n^3 <= m^3/3 (12)
  -stdslow <k>     standard method: stop for the retry when the best score has not halved in k iterations (10; 0 off)
  -retryacc <a>    retry with the embedding only if the first result is worse than a (1e-7)
  -warm <l>        re-solve from l x (first point) + (1 - l) x (standard start) (0 = off; measured slower)
  -dir auto|hkm|nt search direction (auto)
  -ntfast <k>      NT: form dX on the pattern of dZ (1) or by dense transforms (0, default)
  -nodd            no double-double endgame
  -ddfactor <f>    endgame work budget as a multiple of the solve's work (1.0)
  -ddminwork <w>   ... but at least this many flop-equivalents (2e9)
  -ddmaxm <k>      size limit for the endgame (1200)
  -ddit <k>        endgame iteration limit (40)
  -nopolish        do not polish reduced-accuracy solutions
  -crossover <k>   SDP crossover after the solve (Newton on the rank-revealed KKT system; kept only if
                   the errors on the original data fall): 0 off (default), 1 on, -1 when cheap
presolve
  -nofr            no facial-reduction presolve
  -fr              facial reduction whatever its depth (default: skipped when of depth >= 2 and saving < frgain)
  -frgain <g>      per-iteration saving that justifies a facial reduction of depth >= 2 (4)
  -nofrretry       do not re-solve without it when the removed duals are not recovered
  -freeelim <k>    eliminate split free-variable pairs by pivoting: -1 auto (kernel-form SOS
                   problems: few SDP blocks, pairs >= 1% of the rows), 0 off, 1 on
  -tracebound <R>  with the elimination: the row sum tr(X) + s = R that keeps the Gram side
                   bounded (-1 auto: 1e7 (1 + max|b|), 0 off)
  -dualize <k>     solve the dual form (Z + sum y_i A_i = C entrywise, y eliminated) when its
                   Schur complement is at most half the size: -1 auto, 0 off, 1 force
  -mfipm <k>       matrix-free interior-point method: NT Mehrotra steps, Newton systems
                   by preconditioned CG, no Schur complement (1 on, 0 off); for problems with
                   a low-rank optimal side (moment-SOS with few atoms). Its options:
  -mfrho <r>, -mfrmax <k>, -mfdrop <d>, -mfkmax <k>   preconditioner: bulk spread (10), outlier
                   eigenvectors per block (4), dropped pairs (0.5), largest capacitance (8000)
  -mfcgmax <k>, -mfcgtol <t>, -mfcgtolmax <t>   CG: steps per solve (3000), tolerance
                   clamp(1e-2 mu/mu0, t, tmax) (1e-10, 1e-3)
  -mfwarm <k>, -mfstall <k>, -mfdiag <k>   corrector CG started from the predictor (1);
                   stop after k iterations without progress once CG hit its budget, best
                   iterate returned (5); exact diag(M) as the base of the preconditioner (0)
  -fom <k>         first-order engine (dual splitting + augmented Lagrangian / semismooth
                   Newton-CG; no Schur complement): -1 auto (when the dense Schur complement
                   does not fit in memory; a very sparse large block waits for the
                   chordal conversion first), 0 off, 1 force (-1)
  -lralm <k>       experimental: low-rank augmented Lagrangian (Burer-Monteiro, X = R R')
                   for large sparse SDPs such as AC-OPF relaxations: Newton-CG on R with a sparse
                   n x n preconditioner, inequality slacks in closed form, rank 1 first and
                   escapes along negative eigenvectors of Z; no presolve; Z = C - A'y checked by
                   a sparse Cholesky, so b'y with a positive semidefinite Z is a lower bound
                   (1 on, 0 off, the default). Its options: -lrrank (1), -lrrmax (32), -lrsigma (10),
                   -lrtol (1e-6), -lrouter (500), -lrnewton (Newton steps a subproblem, 60; 0: L-BFGS),
                   -lrescape (1), -lrtrace R (with tr X <= R: report b'y - R lambda as a lower bound,
                   Z >= -lambda I by a sparse Cholesky ladder); SDP blocks above n = 46 340 are
                   accepted on this path
  -fomrace <f>     at a tolerance of 1e-6 or looser (-acc low), on problems with few large dense
                   and a sparse Gram matrix A A' (theta-type problems)
                   that the presolve leaves as they are (no chordal conversion, no dual form)
                   and whose interior-point solve is expected to take 20 s or more, the
                   first-order engine runs first for the fraction f of that time (0.2; 0 off);
                   the interior-point method follows if it has not reached the tolerance
  -fomstart-x <f>  start the first-order engine from the X in f (the -x format) ...
  -fomstart-y <f>  ... and the y in f (the -y format): continuing a long run (needs -nosym
                   when a symmetry was found, and a problem the presolve keeps as read)
  -fomhalpern <k>  ... Halpern-restarted Peaceman-Rachford (1) or ADMM with step 1.618 (0)
  -fommaxit <k>    ... iteration limit (100000)
  -fomtol <t>      ... stopping tolerance on the residuals (max(tol, 1e-6))
  -fomsigma <s>    ... fixed penalty parameter (0: adaptive)
  -fomsigma0 <s>   ... starting penalty when adaptive (1); -fomaasafe <f>: an Anderson step
                   is rejected when its residual exceeds f times the last (1; 2 on the
                   degenerate moment relaxations); -fomaa <k>: Anderson memory (25, up to 32,
                   capped by a 1.5 GB history budget); -fomaadr <k>: the acceleration on the
                   Douglas-Rachford variable X/sigma + A*y - C, step 1 (1), or on (X, Z), step 1.618 (0)
  -sym auto|none|<file>  exact symmetry reduction (default auto): sign symmetries and
                   permutation automorphisms of the data are found, the constraints aggregated over
                   the orbits and the blocks split by irreducible representation; the point is
                   mapped back to the file. <file>: generators as index permutations (line 1: SDP
                   block sizes; one per line, global 0-based SDP indices). -nosym = -sym none
  -symtime <s>, -symnodes <k>   caps of the automatic search (20 s, 500 nodes); an incomplete
                   search only reduces less, never wrongly
  -symbd <k>       ... symmetry-adapted basis: the blocks split into one block per irreducible
                   representation (1 on, 0 off)
  -symsign <k>     ... -sym auto: sign symmetries (D A D = ±A, D diagonal ±1) first: blocks split by
                   sign pattern, odd constraints dropped (1 on, 0 off)
  -symsigned <k>   ... -sym auto: signed permutations (x_i -> ±x_g(i)): a second search on the
                   absolute values of the data, signs lifted over GF(2), exact (1 on, 0 off)
  -symmin <f>      ... -sym auto applies the reduction only when the constraints or the block
                   algebra shrink by this factor (1.5; 1: always search and apply); a file's
                   generators are always applied
  -symalg <k>      ... then block structure shared by all data matrices of an SDP block: the
                   *-algebra they generate is block-diagonalised numerically (blocks split into
                   components, equal copies kept once). Finds a group that fixes every constraint
                   matrix (any group, also in a rotated basis), commuting data, hidden direct sums;
                   not a symmetry that maps the constraints onto each other (-sym does, for
                   permutations). -1 auto (blocks up to -symalgmax; an O(nnz) test skips the blocks
                   it proves irreducible), 0 off, 1 every block (any gain); -nosym turns it off too
  -symalgmax <n>   ... -symalg -1: largest block tried (1000; the check costs about one
                   eigendecomposition and two n x n products per block: 0.2 s at n = 1000,
                   1.3 s at n = 1860)
  -chordal <k>     chordal decomposition: -1 auto, 0 off, 1 force
  -chordalmin <n>  only blocks at least this large (100)
  -returnx <k>     library calls (MATLAB, Python, Julia): X of a chordal-decomposed block is
                   -1 returned as a dense matrix unless that needs more than 30% of the memory
                   (then the point is verified on the cliques and no X is returned), 0 never
                   returned (fastest), 1 always returned. The command line returns X with -x
  -cliquemax <c>   largest clique accepted (160)
Schur complement and linear algebra
  -sparse <k>      sparse (envelope/supernodal) Schur complement: -1 auto, 0 off, 1 force
  -lowrank <k>     low-rank Schur route: -1 auto, 0 off, 1 force
  -dict <k>        dictionary (shared-vector) Schur route: -1 auto, 0 off, 1 force
  -mixed <k>       single-precision Schur factor + PCG: -1 auto (m>=1000), 0 off, 1 on
  -route <r>       force Schur route of sparse constraints: 0 sparse-sparse, 1 row-product, 3 row-product on the union pattern
  -threads <k>     OpenMP threads for this run (default: OMP_NUM_THREADS, else all cores; with
                   the default, tiny problems use one thread and, on Linux, the interior-point
                   solve leaves out the cores other processes are using when it starts)
  -parblocks <k>   threads over blocks: -1 auto (several blocks, all n <= 256), 0 off, 1 on
  -cblas <c>       routing cost of a BLAS-3 flop relative to a sparse flop (0.05)
  -densemem <MB>   memory for dense constraint copies (768)
  -pivtol <p>      squared-pivot threshold of the Schur factor (1e-14)
  -regill <r>      regularization used when it is violated (1e-10)
iteration details
  -gmax <g>        step-length factor cap (0.99)
  -lanczos <k>     max Lanczos steps for step lengths (30)
  -sigma <k>       centering rule: 0 SDPT3 adaptive, 1 Mehrotra cubic, 2 cubic+safeguard
  -sigmamin <s>    floor for sigma (0)
  -corr <k>        max centrality correctors per iteration (0)
  -init <k>        start: 0 SDPT3-style, 1 least-squares+shift, 2 +balancing (1)
  -balance <b>     raise sigma when infeasibility > b * complementarity (0 = off)
  -feasgrow <f>    cap primal steps that grow the residual (0 = off)
  -hsdsig <k>      self-dual embedding: sigma candidates (3)
  -hsdpat <k>      self-dual embedding: products on the data pattern (1) or dense (0)
  -hsdbeta <b>     self-dual embedding: neighbourhood floor (1e-3)
  -hsdcorr <k>     self-dual embedding: centrality correctors per iteration (3)
  -hsdcfrac <f>    ... their time budget as a fraction of Schur assembly+factorization (1)
  -hsdcbmin <b>, -hsdcbmax <b>   ... target box for the scaled products (0.1, 10)
  -hsdrefine <k>   self-dual embedding: passes enforcing the primal Newton equation (2)
  -hsdfree <k>     self-dual embedding: free variables given as split LP pairs:
                   -1 auto, 0 keep split, 1 augmented Lagrangian, 3 saddle factorization (-1)
  -hsddir <k>      self-dual embedding direction: 0 HKM, 1 NT (auto: NT on the chordal moment form, HKM otherwise)
  -hsdpivtol <p>   self-dual embedding: squared-pivot threshold of the Schur factor (1e-20)
  -hsdstall <k>    self-dual embedding: iterations without a better iterate before stopping (12)
  -split / -nosplit, -ntls <k>, -ntlsgap <g>   experimental NT residual split
dual method
  -dualrho <r>     potential parameter rho (3)
  -dualcert <k>    certificate search steps (12)
  -dualmargin <f>  back-off factor from the certificate boundary (1.5)
  -dualgiveup <k>  iterations without a certificate before falling back (40)
  -dualold         the first dual-scaling code (for comparison)
  -dualtau <t>     corrector target: Newton decrement (0.9)
  -dualtaunb <t>   decrement that sets mu while no bound is certified (2)
  -dualtauub <t>   with a bound, mu may go below gap/(rho n) down to this decrement (1; 0 off)
  -dualcorr <k>    corrector steps per Schur factorization (12)
  -dualmudrop <f>  mu drops at most by this factor per iteration once a bound exists (0.1)
  -dualgstart <0|1> Gershgorin diagonal start when every diagonal is in the range of A' (1)
  -dualcorrauto <0|1> limit correctors when Z^-1 is as costly as the Schur matrix (1)
  -dualsparse <n>  sparse Cholesky of Z for sparse SDP blocks with n >= this (300; 0 off)
  -dualverifyn <n> blocks this large verify certificates only now and then (500)
  -dualls <0|1>    potential line search over several step lengths (1)
  -dualmurule <k>  0 potential reduction (default), 1 adaptive long step
  -dualtaulong <t> decrement of the long step (murule 1)
  -dualmured <0|1> old heuristic mu reduction without a bound (0)
  -dualgamma <g>   phase-1 objective eps b'y - r, eps = 1/g (0: off)
expert and tuning options (listed for completeness; the defaults are the tested ones)
  -vv              very verbose (per-iteration internals)
  -knownfeas       a solution is known to exist: no infeasibility exits
  -nopolishx       no X-metric polish of the returned point (the Euclidean polish stays)
  -mixedfrac <f>   float Schur factor for 1000 <= m < 4000 when the factorization is at least
                   this fraction of an iteration's work (0.4)
  -freefill <f>    free elimination: no pivot whose Markowitz count exceeds this (1e4)
  -chordalform <k> 1 moment form (shared entries; default), 0 overlap equalities
  -chordalsup <k>  constraint supports up to this size join the pattern as cliques (-1 auto: tried
                   at 8, 4, 12, 0; 0 off)
  -chordalmerge <f> merge a clique into its parent when the separator is at least this fraction (0.6)
  -chordaldens <f> blocks denser than this are not tried (0.3)
  -ddbudget <n>, -ddtime <s>   double-double endgame: work per iteration (1e9), time cap (60 s)
  -hsdrefine <k>, -hsdstall <k>, -hsdbatch <0|1>, -hsdreghint <0|1>, -hsdqscale <0|1|2>,
  -hsdsoltol <t>, -hsdcacc <f>, -hsdcgain <f>, -hsdctarget <0|1>, -hsdctrace <0|1>,
  -hsdcucap <f>, -hsdfomega <f>, -hsdpshrink <f>   embedding internals: refinement passes (2),
                   stall window (12), batched solves (1), regularization hint (1), scaling of
                   the kappa/tau rows (0), solve tolerance (0 auto), corrector acceptance (1.02)
                   and gain (0), corrector target (0 trial mean, 1 sigma mu), trace-free shift (1),
                   downward cap (0 auto), free-pair weight (1e2), free-pair shrink (0)
  -fomsigint <k>, -fomsigrule <k>, -fomsigmax <f>   first-order penalty: checked every k (20),
                   rule (2), largest change factor (2)
  -fomsingle <r>   first-order projections in single precision above this residual (1e-4; 0 never)
  -fomssn <0|1>    first-order phase II (ALM + semismooth Newton-CG) (1); -fomssnafter <k> (300),
                   -fomssnres <r> (1e-3): when it starts; -fomssnrho (3), -fomssnsig0 (10),
                   -fomssnprec (1 Gram), -fomssneta (0.1), -fomssnwarm (0), -fomssnouter (200),
                   -fomssnnewton (30), -fomssncg (200), -fomssnstall (6)
  -fombm <0|1>     first-order kernel B (Burer-Monteiro ALM after the splitting; experimental, 0);
                   -fombmrank0 (10), -fombmouter (100), -fombminner (300), -fombmrho (1),
                   -fombmgtol (1e-7), -fombmnegtol (1e-6)
  -mfcgtime <s>, -mfrecycle <k>   matrix-free IPM: time cap of one CG solve (0 none), recycled
                   Ritz vectors (0)
  -lrinner <k>, -lrprec <0|1>   low-rank ALM with -lrnewton 0 (L-BFGS): steps per subproblem
                   (2000), diagonal preconditioner (1)
exit code: 0 optimal, 10 reduced accuracy, 11 primal infeasible, 12 dual infeasible,
           13 iteration limit, 14 numerical difficulties, 15 time limit, 1 usage, 2 input error
```
<!-- END GENERATED -->
