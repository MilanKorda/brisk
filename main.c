#ifdef __linux__
#define _GNU_SOURCE          /* sched_setaffinity, CPU_ALLOC (the re-exec in main) */
#include <sched.h>
#endif
#include <unistd.h>
#include "brisk.h"
#include <stdio.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef _OPENMP
/* 4.25: without OpenMP, a static SuiteSparse built with it (SuiteSparse_toc/time) still
 * references omp_get_wtime and omp_get_max_threads: provide them (weak, so an OpenMP runtime linked in wins) */
#include <time.h>
double omp_get_wtime(void) __attribute__((weak));
double omp_get_wtime(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int omp_get_max_threads(void) __attribute__((weak));
int omp_get_max_threads(void) { return 1; }
void omp_set_num_threads(int) __attribute__((weak));
void omp_set_num_threads(int n) { (void)n; }
#else
#include <omp.h>
#endif

static const char *status_str[] = {
    "OPTIMAL", "PRIMAL INFEASIBLE", "DUAL INFEASIBLE", "ITERATION LIMIT",
    "NUMERICAL DIFFICULTIES", "SOLVED TO REDUCED ACCURACY", "TIME LIMIT"
};
/* process exit codes: distinct per status (0 = optimal) */
static const int exit_code[] = { 0, 11, 12, 13, 14, 10, 15 };

static void usage(void) {
#ifdef BRISK_LIBRARY
    fprintf(stderr, "brisk: the options are listed by the command-line solver (./brisk with no arguments)\n");
    return;
#endif
    fprintf(stderr,
        "usage: brisk problem.dat-s [options]      (an SDPA sparse file)\n"
        "       brisk problem.mat [options]        (SeDuMi format: a MAT-file with A (or At), b, c, K;\n"
        "                                           K.f, K.l, K.q, K.r, K.s; MAT versions 5 to 7)\n"
        "       brisk problem.mps [options]        (a linear program: MPS, free or fixed format)\n"
        "       brisk problem.cbf[.gz] [options]   (CBF, the format of CBLIB: linear and second-order cones;\n"
        "                                           integer variables are relaxed)\n"
        "A linear program read from an MPS file is presolved and solved by the LP interior-point method\n"
        "(bounds kept as bounds, normal equations); the cone solver takes over when that method gives up\n"
        "(infeasible or unbounded problems: it returns the certificates). Its options: -tol, -maxit,\n"
        "-timelimit, -q, -v, -x <file> (x of the problem as read, one value per line), and\n"
        "  -lppresolve <k>  0 none, 1 rows and columns (fixed, singleton, redundant, forcing, dual fixing),\n"
        "                   2 (default) also substitutions: free column singletons, opposite column pairs,\n"
        "                   implied free columns by their equality rows\n"
        "  -lpmethod <k>    1 (default) the LP interior-point method, 0 the cone solver directly\n"
        "  -lpcorr <k>      centrality correctors per iteration (default: 0-3 from the cost of a\n"
        "                   factorization against a solve)\n"
        "A problem in SeDuMi format without semidefinite blocks (a linear or second-order cone program)\n"
        "is solved by the second-order cone solver: its options are -tol, -acc, -maxit, -timelimit, -q,\n"
        "-v, and -x, -y, -z (x, y and z = c - A'y of the SeDuMi form, one value per line). With\n"
        "semidefinite blocks, or with -conesolver 0, -prec, -bound, -fom, -mfipm, -lralm, the problem\n"
        "goes to the semidefinite solver (second-order cones as arrow blocks) with all the options below.\n"
        "output\n"
        "  -q | -v          quiet | verbose   (Ctrl-C: stop and return the current point; twice: abort)\n"
        "  -y <file>        write y (SDPA primal x = -y), one value per line, original numbering\n"
        "  -x <file>        write X of the original problem (SDPA sparse-like: block i j value, upper triangle)\n"
        "  -z <file>        write Z = C - A'y of the original problem, same format\n"
        "termination\n"
        "  -acc <level>     accuracy: low (tol 1e-6), default (1e-8) or high (1e-10). low also scales the\n"
        "                   re-solve thresholds (10 tol, 100 tol after the embedding) and the reduced-\n"
        "                   accuracy class (100 tol); high keeps the default ones. -tol/-retryacc override it\n"
        "  -tol <t>         relative gap / infeasibility tolerance (1e-8)\n"
        "  -maxit <k>       iteration limit (100)\n"
        "  -timelimit <s>   wall-clock limit in seconds (0 = none)\n"
        "  -bound p|d       return the best GUARANTEED bound instead of the best-balanced pair:\n"
        "                   p: a feasible X of (P) (A(X) = b to -boundtol, X > 0 with a margin), <C,X> an\n"
        "                   upper bound on the optimal value even when the solve does not converge; d: a\n"
        "                   feasible y (Z = C - A'y > 0 verified), b'y a lower bound. (P) is the SDPA dual\n"
        "                   (min <C,X>, C = -F0): SOS certificates from SeDuMi/GloptiPoly are -bound p;\n"
        "                   relaxations of minimisation problems take the lower bound from -bound d.\n"
        "                   The certificate replaces its side of the returned pair (-x / -y write it);\n"
        "                   -certify makes the bound rigorous. See README (bound mode).\n"
        "  -boundanchor <f> with -bound d: a y (the -y format, as read) whose Z = C - A'y is positive\n"
        "                   definite, e.g. from a solve of the problem with the largest margin; a\n"
        "                   candidate whose Z is not positive semidefinite is blended with it\n"
        "  -boundtol <t>    relative residual required of a -bound p certificate (1e-10)\n"
        "  -boundmargin <e> eigenvalue margin of the certificate, relative to 1 + max diagonal; at least\n"
        "                   8 n^2 u per block of order n, what a rigorous check needs (1e-13)\n"
        "  -boundtrack <t>  residual below which iterates are tracked as candidates (100 x -boundtol)\n"
        "  -certify         with -bound, check the certificate rigorously on the\n"
        "                   problem as read (the file's decimals enclosed in intervals, directed rounding,\n"
        "                   a verified lambda_min) and print the rigorous bound\n"
        "  -certify-x <f>   no solve: rigorously check the X in f (-x format: block i j value) as a\n"
        "  -certify-y <f>   ... p certificate, or the y in f (-y format) as a d certificate; exit code 0\n"
        "                   when certified, 20 when not (certificates of any solver in these formats)\n"
        "  -stallwin <k>    non-improving iterations tolerated once below 1e-6 (5)\n"
        "high precision\n"
        "  -prec <p>        solve in high precision: dd (double-double, about 32 digits), qd (quad-double,\n"
        "                   about 64 digits) or a number of decimal digits (variable precision; up to 31\n"
        "                   digits is dd, up to 63 qd). The data are read exactly as written in the file.\n"
        "                   Blocks whose data are block diagonal after a permutation are split. The\n"
        "                   reductions of the double solver are applied to the data in the working\n"
        "                   precision: sign and permutation symmetry (the group verified entry by entry),\n"
        "                   the block diagonalisation of the data (-symalg), the chordal decomposition\n"
        "                   where the banded factorization can use it (not with -bound, -fom, -lralm). A\n"
        "                   presolve in the working precision plus guard digits removes split free\n"
        "                   variables and simple faces (rank-one, diagonal, LP certificates); the result\n"
        "                   and its errors are those of the problem as read. The solve starts from a\n"
        "                   double solve by the default method and goes up a ladder dd -> qd -> variable\n"
        "                   precision; the interior-point method ends quadratically (centring steps, then\n"
        "                   one long step per level). With -fom 1 the augmented Lagrangian / semismooth\n"
        "                   Newton method runs in high precision instead, with -lralm 1 the low-rank\n"
        "                   (X = R R') augmented Lagrangian method. -x/-y/-z write all the digits.\n"
        "                   Status PRECISION LIMIT: the accuracy the precision reaches on the problem\n"
        "                   With -bound p|d the certificate of that side is built in high precision on the\n"
        "                   problem as read and checked rigorously (in the precision of -prec plus guard\n"
        "                   limbs, with a-priori error bounds of the arithmetic); the rigorous bound is\n"
        "                   printed with all digits, and -x / -y write the certificate with all the digits\n"
        "                   of its check. -certify-x / -certify-y with -prec check a given certificate\n"
        "                   the same way (p needs an interior of (P), d one of (D))\n"
        "  -hpext <k>       interior-point method: when the precision is exhausted short of the tolerance\n"
        "                   (problems without an interior that the presolve does not reduce reach about\n"
        "                   half the digits), the solve continues in the next precision (dd -> qd -> more\n"
        "                   bits), at most k times and while that gains a digit (2; 0: off). The tolerance\n"
        "                   and the digits written stay those of -prec\n"
        "  -hptol <t>       tolerance of the high-precision solve. Default: the solve aims at 2^(-0.75 bits)\n"
        "                   (1e-24 for dd, 1e-48 for qd, 1e-77 for 100 digits) and a result within\n"
        "                   2^(-0.6 bits) (1e-19, 1e-38, 1e-62) counts as OPTIMAL; -fom / -lralm aim at the latter\n"
        "method\n"
        "  -conesolver 0    a problem in SeDuMi format without semidefinite blocks: by the semidefinite solver\n"
        "                   (second-order cones as arrow blocks) instead of the cone solver (default 1)\n"
        "  -dual            dual-scaling method (DSDP-style: potential reduction, correctors, verified\n"
        "                   certificates); falls back to the primal-dual method unless certified\n"
        "  -hsd | -nohsd    force | forbid the self-dual embedding (default: automatic - the embedding first\n                   on most problems, the infeasible start first on large dense blocks with a cheap\n                   Schur complement; each method is the other's fallback)\n  -hsdfirst <0|1>  automatic method: the embedding first when its extra dense work is small or the\n                   problem tiny (1; 0: standard first); many split free pairs and the chordal moment\n                   form use the embedding either way\n  -hsdfirstc <c>   ... i.e. when c * sum n^3 <= m^3/3 (12),\n  -hsdfirstasm <f> ... or <= f * (m^3/3 + the estimated work of the Schur assembly) (0.5; 0: off)\n  -stdslow <k>     standard method: stop for the retry when the best score has not halved in k iterations (10; 0 off)\n"
        "  -retryacc <a>    retry with the embedding only if the first result is worse than a (1e-7)\n"
        "  -warm <l>        re-solve from l x (first point) + (1 - l) x (standard start) (0 = off; measured slower)\n"
        "  -dir auto|hkm|nt search direction (auto)\n"
        "  -ntfast <k>      NT: form dX on the pattern of dZ (1) or by dense transforms (0, default)\n"
        "  -nodd            no double-double endgame\n"
        "  -ddfactor <f>    endgame work budget as a multiple of the solve's work (1.0)\n"
        "  -ddminwork <w>   ... but at least this many flop-equivalents (2e9)\n"
        "  -ddmaxm <k>      size limit for the endgame (1200)\n"
        "  -ddit <k>        endgame iteration limit (40)\n"
        "  -nopolish        do not polish reduced-accuracy solutions\n  -crossover <k>   SDP crossover after the solve (Newton on the rank-revealed KKT system; kept only if\n                   the errors on the original data fall): 0 off (default), 1 on, -1 when cheap\n"
        "presolve\n"
        "  -nofr            no facial-reduction presolve\n  -fr              facial reduction whatever its depth (default: skipped when of depth >= 2 and saving < frgain)\n  -frgain <g>      per-iteration saving that justifies a facial reduction of depth >= 2 (4)\n"
        "  -nofrretry       do not re-solve without it when the removed duals are not recovered\n"
        "  -freeelim <k>    eliminate split free-variable pairs by pivoting: -1 auto (kernel-form SOS\n"
        "                   problems: few SDP blocks, pairs >= 1%% of the rows), 0 off, 1 on\n"
        "  -tracebound <R>  with the elimination: the row sum tr(X) + s = R that keeps the Gram side\n"
        "                   bounded (-1 auto: 1e7 (1 + max|b|), 0 off)\n"
        "  -dualize <k>     solve the dual form (Z + sum y_i A_i = C entrywise, y eliminated) when its\n"
        "                   Schur complement is at most half the size: -1 auto, 0 off, 1 force\n"
        "  -mfipm <k>       matrix-free interior-point method: NT Mehrotra steps, Newton systems\n"
        "                   by preconditioned CG, no Schur complement (1 on, 0 off); for problems with\n"
        "                   a low-rank optimal side (moment-SOS with few atoms). 2: the hybrid - the\n"
        "                   matrix-free iteration until its merit is below -mfhand (1e-3) or an\n"
        "                   iteration needs more than -mfhandcg (600) CG steps, then the standard\n"
        "                   method from that iterate (a few Schur factorizations instead of thirty;\n"
        "                   truss topology problems with m = 7,000-14,000: 2.5-3x faster). Its options:\n"
        "  -mftry <k>       the matrix-free method first, automatically (1; 0 off): on problems with one\n"
        "                   to four dense blocks of order >= 100 whose Schur factorization is\n"
        "                   estimated at 3.5 times a matrix-free solve or more (at least equal when\n"
        "                   every constraint has an entry in an LP block), and that the presolve\n"
        "                   leaves unchanged. The attempt ends when a solve away from the\n"
        "                   optimum needs more CG steps than a sixth of a Schur factorization costs,\n"
        "                   or the attempt more than a quarter of the standard solve; unless it\n"
        "                   ends OPTIMAL, the standard method follows (from the attempt's iterate\n"
        "                   of merit 1e-3 if it got that far); also when the Schur complement does\n"
        "                   not fit in memory (the first-order engine follows then)\n"
        "  -mfrho <r>, -mfrmax <k>, -mfdrop <d>, -mfkmax <k>   preconditioner: bulk spread (10), outlier\n"
        "                   eigenvectors per block (4), dropped pairs (0.5), largest capacitance (8000)\n"
        "  -mfproj <k>, -mfeta <e>   0 (default): no projection of the primal direction, CG until its\n"
        "                   residual is below e (0.1) times the primal infeasibility; 1: the direction\n"
        "                   projected onto A(dX) = Rp by a second CG (the earlier scheme)\n"
        "  -mfcgmax <k>, -mfcgtol <t>, -mfcgtolmax <t>   CG: steps per solve (3000), tolerance\n"
        "                   clamp(1e-2 mu/mu0, t, tmax) (1e-10, 1e-3)\n"
        "  -mfwarm <k>, -mfstall <k>, -mfdiag <k>   corrector CG started from the predictor (1);\n"
        "                   stop after k iterations without progress once CG hit its budget, best\n"
        "                   iterate returned (5); exact diag(M) as the base of the preconditioner (0)\n"
        "  -fom <k>         first-order engine (dual splitting + augmented Lagrangian / semismooth\n"
        "                   Newton-CG; no Schur complement): -1 auto (when the dense Schur complement\n"
        "                   does not fit in memory; a very sparse large block waits for the\n"
        "                   chordal conversion first), 0 off, 1 force (-1)\n"
        "  -lralm <k>       experimental: low-rank augmented Lagrangian (Burer-Monteiro, X = R R')\n"
        "                   for large sparse SDPs such as AC-OPF relaxations: Newton-CG on R with a sparse\n"
        "                   n x n preconditioner, inequality slacks in closed form, rank 1 first and\n"
        "                   escapes along negative eigenvectors of Z; no presolve; Z = C - A'y checked by\n"
        "                   a sparse Cholesky, so b'y with a positive semidefinite Z is a lower bound\n"
        "                   (1 on, 0 off, the default). Its options: -lrrank (1), -lrrmax (32), -lrsigma (10),\n"
        "                   -lrtol (1e-6), -lrouter (500), -lrnewton (Newton steps a subproblem, 60; 0: L-BFGS),\n"
        "                   -lrescape (1), -lrtrace R (with tr X <= R: report b'y - R lambda as a lower bound,\n"
        "                   Z >= -lambda I by a sparse Cholesky ladder); SDP blocks above n = 46 340 are\n"
        "                   accepted on this path\n"
        "  -fomrace <f>     at a tolerance of 1e-6 or looser (-acc low), on problems with few large dense\n"
        "                   and a sparse Gram matrix A A' (theta-type problems)\n"
        "                   that the presolve leaves as they are (no chordal conversion, no dual form)\n"
        "                   and whose interior-point solve is expected to take 20 s or more, the\n"
        "                   first-order engine runs first for the fraction f of that time (0.2; 0 off);\n"
        "                   the interior-point method follows if it has not reached the tolerance\n"
        "  -fomstart-x <f>  start the first-order engine from the X in f (the -x format) ...\n"
        "  -fomstart-y <f>  ... and the y in f (the -y format): continuing a long run (needs -nosym\n"
        "                   when a symmetry was found, and a problem the presolve keeps as read)\n"
        "  -fomhalpern <k>  ... Halpern-restarted Peaceman-Rachford (1) or ADMM with step 1.618 (0)\n"
        "  -fommaxit <k>    ... iteration limit (100000)\n"
        "  -fomtol <t>      ... stopping tolerance on the residuals (max(tol, 1e-6))\n"
        "  -fomsigma <s>    ... fixed penalty parameter (0: adaptive)\n"
        "  -fomsigma0 <s>   ... starting penalty when adaptive (1); -fomaasafe <f>: an Anderson step\n"
        "                   is rejected when its residual exceeds f times the last (1; 2 on the\n"
        "                   degenerate moment relaxations); -fomaa <k>: Anderson memory (25, up to 32,\n"
        "                   capped by a 1.5 GB history budget); -fomaadr <k>: the acceleration on the\n"
        "                   Douglas-Rachford variable X/sigma + A*y - C, step 1 (1), or on (X, Z), step 1.618 (0)\n"
        "  -sym auto|none|<file>  exact symmetry reduction (default auto): sign symmetries and\n"
        "                   permutation automorphisms of the data are found, the constraints aggregated over\n"
        "                   the orbits and the blocks split by irreducible representation; the point is\n"
        "                   mapped back to the file. <file>: generators as index permutations (line 1: SDP\n"
        "                   block sizes; one per line, global 0-based SDP indices). -nosym = -sym none\n"
        "  -symtime <s>, -symnodes <k>   caps of the automatic search (20 s, 500 nodes); an incomplete\n"
        "                   search only reduces less, never wrongly\n"
        "  -symbd <k>       ... symmetry-adapted basis: the blocks split into one block per irreducible\n"
        "                   representation (1 on, 0 off)\n"
        "  -symsign <k>     ... -sym auto: sign symmetries (D A D = ±A, D diagonal ±1) first: blocks split by\n"
        "                   sign pattern, odd constraints dropped (1 on, 0 off)\n"
        "  -symsigned <k>   ... -sym auto: signed permutations (x_i -> ±x_g(i)): a second search on the\n"
        "                   absolute values of the data, signs lifted over GF(2), exact (1 on, 0 off)\n"
        "  -symmin <f>      ... -sym auto applies the reduction only when the constraints, the block\n"
        "                   algebra or (at most 4 blocks, m <= 20000) the work m^3/3 + 30 sum n^3\n"
        "                   shrink by this factor (1.5; 1: always search and apply); a file's\n"
        "                   generators are always applied\n"
        "  -symalg <k>      ... then block structure shared by all data matrices of an SDP block: the\n"
        "                   *-algebra they generate is block-diagonalised numerically (blocks split into\n"
        "                   components, equal copies kept once). Finds a group that fixes every constraint\n"
        "                   matrix (any group, also in a rotated basis), commuting data, hidden direct sums;\n"
        "                   not a symmetry that maps the constraints onto each other (-sym does, for\n"
        "                   permutations). -1 auto (blocks up to -symalgmax; an O(nnz) test skips the blocks\n"
        "                   it proves irreducible), 0 off, 1 every block (any gain); -nosym turns it off too\n"
        "  -symalgmax <n>   ... -symalg -1: largest block tried (1000; the check costs about one\n"
        "                   eigendecomposition and two n x n products per block: 0.2 s at n = 1000,\n"
        "                   1.3 s at n = 1860)\n"        "  -chordal <k>     chordal decomposition: -1 auto, 0 off, 1 force\n"
        "  -chordalmin <n>  only blocks at least this large (100)\n"
        "  -returnx <k>     library calls (MATLAB, Python, Julia): X of a chordal-decomposed block is\n"
        "                   -1 returned as a dense matrix unless that needs more than 30%% of the memory\n"
        "                   (then the point is verified on the cliques and no X is returned), 0 never\n"
        "                   returned (fastest), 1 always returned. The command line returns X with -x\n"
        "  -cliquemax <c>   largest clique accepted (160)\n"
        "Schur complement and linear algebra\n"
        "  -sparse <k>      sparse (envelope/supernodal) Schur complement: -1 auto, 0 off, 1 force\n"
        "  -lowrank <k>     low-rank Schur route: -1 auto, 0 off, 1 force\n"
        "  -dict <k>        dictionary (shared-vector) Schur route: -1 auto, 0 off, 1 force\n"
        "  -mixed <k>       single-precision Schur factor + PCG: -1 auto (m>=1000), 0 off, 1 on\n"
        "  -route <r>       force Schur route of sparse constraints: 0 sparse-sparse, 1 row-product, 3 row-product on the union pattern\n"
        "  -threads <k>     OpenMP threads for this run (default: OMP_NUM_THREADS, else all cores; with\n"
        "                   the default, tiny problems use one thread and, on Linux, the interior-point\n"
        "                   solve leaves out the cores other processes are using when it starts)\n"
        "  -parblocks <k>   threads over blocks: -1 auto (several blocks, all n <= 256), 0 off, 1 on\n"
        "  -cblas <c>       routing cost of a BLAS-3 flop relative to a sparse flop (0.05)\n"
        "  -densemem <MB>   memory for dense constraint copies (768)\n"
        "  -pivtol <p>      squared-pivot threshold of the Schur factor (1e-14)\n"
        "  -regill <r>      regularization used when it is violated (1e-10)\n"
        "iteration details\n"
        "  -gmax <g>        step-length factor cap (0.99)\n"
        "  -lanczos <k>     max Lanczos steps for step lengths (30)\n"
        "  -sigma <k>       centering rule: 0 SDPT3 adaptive, 1 Mehrotra cubic, 2 cubic+safeguard\n"
        "  -sigmamin <s>    floor for sigma (0)\n"
        "  -corr <k>        max centrality correctors per iteration (0)\n"
        "  -init <k>        start: 0 SDPT3-style, 1 least-squares+shift, 2 +balancing (1)\n"
        "  -balance <b>     raise sigma when infeasibility > b * complementarity (0 = off)\n"
        "  -feasgrow <f>    cap primal steps that grow the residual (0 = off)\n"
        "  -hsdsig <k>      self-dual embedding: sigma candidates (3)\n"
        "  -hsdpat <k>      self-dual embedding: products on the data pattern (1) or dense (0)\n"
        "  -hsdbeta <b>     self-dual embedding: neighbourhood floor (1e-3)\n"
        "  -hsdcorr <k>     self-dual embedding: centrality correctors per iteration (3)\n"
        "  -hsdhoc <k>      self-dual embedding: passes that re-evaluate the second-order term at the chosen\n"
        "                   direction, kept when they predict a larger reduction (4; 0 = off)\n"
        "  -hsdnb <b>       self-dual embedding: after kept passes the step may go to 0.9995 of the boundary\n"
        "                   while lambda_min(XZ) >= b mu (0.3; 0 = off)\n"
        "  -hsdcfrac <f>    ... their time budget as a fraction of Schur assembly+factorization (1)\n"
        "  -hsdcbmin <b>, -hsdcbmax <b>   ... target box for the scaled products (0.1, 10)\n"
        "  -hsdrefine <k>   self-dual embedding: passes enforcing the primal Newton equation (2)\n"
        "  -hsdfree <k>     self-dual embedding: free variables given as split LP pairs:\n"
        "                   -1 auto, 0 keep split, 1 augmented Lagrangian, 3 saddle factorization (-1)\n"
        "  -hsddir <k>      self-dual embedding direction: 0 HKM, 1 NT (auto: NT on the chordal moment form, HKM otherwise)\n"
        "  -hsdpivtol <p>   self-dual embedding: squared-pivot threshold of the Schur factor (1e-20)\n"
        "  -hsdstall <k>    self-dual embedding: iterations without a better iterate before stopping (12)\n"
        "  -split / -nosplit, -ntls <k>, -ntlsgap <g>   experimental NT residual split\n"
        "dual method\n"
        "  -dualrho <r>     potential parameter rho (3)\n"
        "  -dualcert <k>    certificate search steps (12)\n"
        "  -dualmargin <f>  back-off factor from the certificate boundary (1.5)\n"
        "  -dualgiveup <k>  iterations without a certificate before falling back (40)\n"
        "  -dualold         the first dual-scaling code (for comparison)\n"
        "  -dualtau <t>     corrector target: Newton decrement (0.9)\n"
        "  -dualtaunb <t>   decrement that sets mu while no bound is certified (2)\n"
        "  -dualtauub <t>   with a bound, mu may go below gap/(rho n) down to this decrement (1; 0 off)\n"
        "  -dualcorr <k>    corrector steps per Schur factorization (12)\n"
        "  -dualmudrop <f>  mu drops at most by this factor per iteration once a bound exists (0.1)\n"
        "  -dualgstart <0|1> Gershgorin diagonal start when every diagonal is in the range of A' (1)\n"
        "  -dualcorrauto <0|1> limit correctors when Z^-1 is as costly as the Schur matrix (1)\n"
        "  -dualsparse <n>  sparse Cholesky of Z for sparse SDP blocks with n >= this (300; 0 off)\n"
        "  -dualverifyn <n> blocks this large verify certificates only now and then (500)\n"
        "  -dualls <0|1>    potential line search over several step lengths (1)\n"
        "  -dualmurule <k>  0 potential reduction (default), 1 adaptive long step\n"
        "  -dualtaulong <t> decrement of the long step (murule 1)\n"
        "  -dualmured <0|1> old heuristic mu reduction without a bound (0)\n"
        "  -dualgamma <g>   phase-1 objective eps b'y - r, eps = 1/g (0: off)\n"
        "expert and tuning options (listed for completeness; the defaults are the tested ones)\n"
        "  -vv              very verbose (per-iteration internals)\n"
        "  -knownfeas       a solution is known to exist: no infeasibility exits\n"
        "  -nopolishr       no restoration of the primal feasibility of a point that missed the tolerance\n"
        "                   (the correction in the square-root metric of X; the older polishes stay)\n"
        "  -nopolishx       no X-metric polish of the returned point (the Euclidean polish stays)\n"
        "  -mixedfrac <f>   float Schur factor for 1000 <= m < 4000 when the factorization is at least\n"
        "                   this fraction of an iteration's work (0.4)\n"
        "  -freefill <f>    free elimination: no pivot whose Markowitz count exceeds this (1e4)\n"
        "  -chordalform <k> 1 moment form (shared entries; default), 0 overlap equalities\n"
        "  -chordalsup <k>  constraint supports up to this size join the pattern as cliques (-1 auto: tried\n"
        "                   at 8, 4, 12, 0; 0 off)\n"
        "  -chordalmerge <f> merge a clique into its parent when the separator is at least this fraction (0.6)\n"
        "  -chordaldens <f> blocks denser than this are not tried (0.3)\n"
        "  -ddbudget <n>, -ddtime <s>   double-double endgame: work per iteration (1e9), time cap (60 s)\n"
        "  -hsdrefine <k>, -hsdstall <k>, -hsdbatch <0|1>, -hsdreghint <0|1>, -hsdqscale <0|1|2>,\n"
        "  -hsdsoltol <t>, -hsdcacc <f>, -hsdcgain <f>, -hsdctarget <0|1>, -hsdctrace <0|1>,\n"
        "  -hsdcucap <f>, -hsdfomega <f>, -hsdpshrink <f>   embedding internals: refinement passes (2),\n"
        "                   stall window (12), batched solves (1), regularization hint (1), scaling of\n"
        "                   the kappa/tau rows (0), solve tolerance (0 auto), corrector acceptance (1.02)\n"
        "                   and gain (0), corrector target (0 trial mean, 1 sigma mu), trace-free shift (1),\n"
        "                   downward cap (0 auto), free-pair weight (1e2), free-pair shrink (0)\n"
        "  -fomsigint <k>, -fomsigrule <k>, -fomsigmax <f>   first-order penalty: checked every k (20),\n"
        "                   rule (2), largest change factor (2)\n"
        "  -fomsingle <r>   first-order projections in single precision above this residual (1e-4; 0 never)\n"
        "  -fomssn <0|1>    first-order phase II (ALM + semismooth Newton-CG) (1); -fomssnafter <k> (300),\n"
        "                   -fomssnres <r> (1e-3): when it starts (phase II begins as soon as the\n"
        "                   residual is below -fomssnres, whatever -fomssnafter says; to delay it,\n"
        "                   lower both); -fomssnrho (3), -fomssnsig0 (10),\n"
        "                   -fomssnprec (1 Gram), -fomssneta (0.1), -fomssnwarm (0), -fomssnouter (200),\n"
        "                   -fomssnnewton (30), -fomssncg (200), -fomssnstall (6)\n"
        "  -fombm <0|1>     first-order kernel B (Burer-Monteiro ALM after the splitting; experimental, 0);\n"
        "                   -fombmrank0 (10), -fombmouter (100), -fombminner (300), -fombmrho (1),\n"
        "                   -fombmgtol (1e-7), -fombmnegtol (1e-6)\n"
        "  -mfcgtime <s>, -mfrecycle <k>   matrix-free IPM: time cap of one CG solve (0 none), recycled\n"
        "                   Ritz vectors (0)\n"
        "  -lrinner <k>, -lrprec <0|1>   low-rank ALM with -lrnewton 0 (L-BFGS): steps per subproblem\n"
        "                   (2000), diagonal preconditioner (1)\n"
        "exit code: 0 optimal, 10 reduced accuracy, 11 primal infeasible, 12 dual infeasible,\n"
        "           13 iteration limit, 14 numerical difficulties, 15 time limit, 1 usage, 2 input error\n");
}

static int need(int i, int argc, const char *opt) {
    if (i + 1 >= argc) { fprintf(stderr, "brisk: option %s needs a value\n", opt); exit(1); }
    return i + 1;
}
static double num(const char *s, const char *opt) {
    char *e;
    double v = strtod(s, &e);
    if (e == s || *e) { fprintf(stderr, "brisk: invalid value '%s' for %s\n", s, opt); exit(1); }
    return v;
}

static void write_matrix_file(const char *fname, const PSOrig *O, double **M) {
    FILE *f = fopen(fname, "w");
    if (!f) { fprintf(stderr, "brisk: cannot write %s\n", fname); return; }
    for (int k = 0; k < O->nblk; k++) {
        int n = abs(O->bs[k]);
        if (O->bs[k] < 0) {
            for (int i = 0; i < n; i++) if (M[k][i] != 0) fprintf(f, "%d %d %d %.17g\n", k + 1, i + 1, i + 1, M[k][i]);
        } else {
            for (int j = 0; j < n; j++)
                for (int i = 0; i <= j; i++) {
                    double v = M[k][i + (size_t)j * n];
                    if (v != 0) fprintf(f, "%d %d %d %.17g\n", k + 1, i + 1, j + 1, v);
                }
        }
    }
    fclose(f);
}

/* binary X for the benchmark harness (BRISK_XOUT): int nblk, int 1, int 0; per block
 * int type, int n, then n (LP) or n(n+1)/2 values (lower triangle by columns) */
static void write_x_binary(const char *fname, const PSOrig *O, double **X) {
    FILE *f = fopen(fname, "wb");
    if (!f) return;
    int hdr[3] = { O->nblk, 1, 0 };
    fwrite(hdr, sizeof(int), 3, f);
    for (int k = 0; k < O->nblk; k++) {
        int n = abs(O->bs[k]);
        int th[2] = { O->bs[k] < 0 ? BLK_LP : BLK_SDP, n };
        fwrite(th, sizeof(int), 2, f);
        if (O->bs[k] < 0) fwrite(X[k], sizeof(double), n, f);
        else for (int j = 0; j < n; j++) fwrite(X[k] + (size_t)j * n + j, sizeof(double), n - j, f);
    }
    fclose(f);
}


typedef struct {
    Problem P;
    PSOrig *O;
    PSOrig *Ofile;          /* the problem as read, when free variables were eliminated (O is then the reduced one) */
    PSOrig *Otb;            /* 4.30: the problem before the trace bound row was added (O then has it) */
    double tbR, tbw, tbs;   /* the bound, its dual and its slack */
    PSFreeElim *fe;
    PSOrig *Odual;          /* 4.30: the problem as read, when the dual form was solved (Ofile/Otb/O are then of the dual form) */
    PSDual *dual;
    PSOrig *Osym;           /* 4.32: the problem as read, when the symmetry reduction was applied (everything else is of the reduced one) */
    PSSym *sym;
    PSOrig *Osign;          /* 4.33: the problem as read, when the sign-symmetry reduction was applied (before the permutation one) */
    PSSign *sign;
    PSOrig *Oalg;           /* 4.41: the problem before the *-algebra block diagonalisation (after the exact symmetry reductions) */
    PSAlg *alg;
    PSOrig *Ofin;           /* the problem (Xo, yo) refer to after run_finish: one of O, Otb, Ofile, Odual (not owned) */
    int nXo;                /* blocks of Xo */
    Result R;
    PSResult pr;
    double *y, **X, **Xo, *yo;
    int status, measured, fr_removed;
    double acc, tread, tpre;
    BoundCert cert;         /* 4.37 bound mode: the best certificate of this run, on the problem as read */
    int fom_check;          /* 4.39: the first-order decision deferred until after the dual form and free elimination */
    int fom_chordal;        /* 4.40: ... until after the chordal conversion (a very sparse large block) */
} Run;

static void run_free(Run *u) {
    if (u->Xo) { for (int k = 0; k < u->nXo; k++) free(u->Xo[k]); free(u->Xo); }
    if (u->X) { for (int k = 0; k < u->P.nblk; k++) free(u->X[k]); free(u->X); }
    free(u->y); free(u->yo);
    ps_free(u->P.ps); u->P.ps = NULL;
    ps_orig_free(u->O);
    if (u->Ofile) ps_orig_free(u->Ofile);
    if (u->Otb) ps_orig_free(u->Otb);
    if (u->Odual) ps_orig_free(u->Odual);
    dualize_free(u->dual);
    if (u->Osym) ps_orig_free(u->Osym);
    sym_free(u->sym);
    if (u->Osign) ps_orig_free(u->Osign);
    sign_free(u->sign);
    if (u->Oalg) ps_orig_free(u->Oalg);
    alg_free(u->alg);
    free_elim_free(u->fe);
    bound_cert_free(&u->cert);
    problem_free(&u->P);
    memset(u, 0, sizeof(*u));
}

/* read, presolve, solve and postsolve; the status is set from the errors of the returned
 * point measured on the problem as read */
/* 4.25: no constraints (m = 0, as read or after presolve): min <C,X> over X >= 0 has the
 * optimum X = 0 when C >= 0; otherwise it is unbounded along v v' for an eigenvector v of C
 * with a negative eigenvalue (the interior-point code needs m >= 1: LAPACK rejects lda = 0) */
static void solve_m0(Problem *P, Result *R, double **X) {
    memset(R, 0, sizeof(*R));
    int kb = -1, jb = -1;
    double lb = 0, cn = 0;
    double **V = calloc(P->nblk + 1, sizeof(double *));
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        const int n = B->n;
        const SpSym *C = &B->C;
        if (B->type == BLK_LP) {
            for (int q = 0; q < C->nnz; q++) {
                cn = fmax(cn, fabs(C->val[q]));
                if (C->val[q] < lb) { lb = C->val[q]; kb = k; jb = C->row[q]; }
            }
            continue;
        }
        double *A = calloc((size_t)n * n + 1, sizeof(double)), *ev = malloc(sizeof(double) * (n + 1));
        for (int q = 0; q < C->nnz; q++) {
            A[C->row[q] + (size_t)C->col[q] * n] = C->val[q];
            A[C->col[q] + (size_t)C->row[q] * n] = C->val[q];
            cn = fmax(cn, fabs(C->val[q]));
        }
        int lw = -1, info = 0; double wq = 0;
        BL(dsyev_)("V", "L", &n, A, &n, ev, &wq, &lw, &info);
        lw = (int)wq + 1;
        double *wk = malloc(sizeof(double) * lw);
        BL(dsyev_)("V", "L", &n, A, &n, ev, wk, &lw, &info);
        free(wk);
        if (info == 0 && ev[0] < lb) { lb = ev[0]; kb = k; jb = 0; }
        V[k] = A;
        free(ev);
    }
    if (kb >= 0 && lb < -1e-12 * fmax(cn, 1.0)) {
        /* the ray, scaled to <C,X> = -1 */
        const Block *B = &P->blk[kb];
        const int n = B->n;
        if (B->type == BLK_LP) X[kb][jb] = 1.0 / -lb;
        else {
            const double *v = V[kb];
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) X[kb][i + (size_t)j * n] = v[i] * v[j] / -lb;
        }
        R->status = ST_DINFEAS;
        R->pobj = -1;
    } else R->status = ST_OPTIMAL;
    for (int k = 0; k < P->nblk; k++) free(V[k]);
    free(V);
}

/* 4.30: (Xo, yo) back to the problem as read: the trace-bound row and slack dropped, the
 * eliminated free values and pivot-row duals restored (free_elim_post), the dual form mapped
 * back; the errors and the status are then those of the returned point on the file data
 * (until 4.29 the free-elimination runs reported the errors of the reduced problem). */
/* the chain of run_finish on a pair (Xo, yo) of u->O: the trace-bound row and slack
 * dropped, the free elimination, the dual form, the symmetry and sign reductions undone;
 * returns the problem the pair then refers to. main_res: the run's own result (the trace
 * bound's slack and dual are recorded, an infeasibility status follows the dual form). */
static PSOrig *map_back(Run *u, const Params *par, double ***pXo, double **pyo, int *pnXo, int main_res) {
    const int infeas = u->status == ST_PINFEAS || u->status == ST_DINFEAS;
    PSOrig *O = u->O;
    double **Xo = *pXo, *yo = *pyo;
    if (u->Otb) {
        if (main_res) {
            u->tbw = yo[O->m - 1];
            u->tbs = Xo ? Xo[O->nblk - 1][0] : 0.0;
        }
        if (Xo) { free(Xo[O->nblk - 1]); Xo[O->nblk - 1] = NULL; *pnXo = O->nblk - 1; }
        O = u->Otb;
        if (main_res) {
            u->tbR = u->P.tbR > 0 ? u->P.tbR : u->tbR;
            if (par->verbose > 0) printf("   trace bound: slack %.3g of %.3g, dual %.3e\n", u->tbs, u->tbR, u->tbw);
        }
    }
    if (u->fe) {
        double *yf = calloc(u->Ofile->m + 1, sizeof(double));
        free_elim_post(u->fe, u->Ofile, Xo, yo, yf);
        if (Xo) free_elim_refine(u->fe, u->Ofile, Xo, main_res ? par->verbose : 0);
        free(yo); yo = yf;
        O = u->Ofile;
    }
    if (u->dual) {
        double **Xd = NULL, *yd = calloc(u->Odual->m + 1, sizeof(double));
        dualize_unmap(u->dual, Xo, yo, &Xd, yd);
        if (Xo) { for (int k = 0; k < *pnXo; k++) free(Xo[k]); free(Xo); }
        free(yo);
        Xo = Xd; yo = yd; *pnXo = u->Odual->nblk;
        O = u->Odual;
        if (main_res && infeas) u->status = u->status == ST_PINFEAS ? ST_DINFEAS : ST_PINFEAS;
    }
    if (u->alg) {
        /* 4.41: X = sum_c Q_c (I (x) X_c) Q_c' on the blocks before the algebra reduction, y unchanged */
        double *ys = calloc(u->Oalg->m + 1, sizeof(double)), **Xs = NULL;
        if (Xo) {
            Xs = malloc(sizeof(double *) * u->Oalg->nblk);
            for (int k = 0; k < u->Oalg->nblk; k++) { const int n = abs(u->Oalg->bs[k]); Xs[k] = calloc((u->Oalg->bs[k] < 0 ? (size_t)n : (size_t)n * n) + 1, sizeof(double)); }
        }
        alg_unmap(u->alg, Xo, yo, Xs, ys);
        if (Xo) { for (int k = 0; k < *pnXo; k++) free(Xo[k]); free(Xo); }
        free(yo);
        Xo = Xs; yo = ys; *pnXo = u->Oalg->nblk;
        O = u->Oalg;
    }
    if (u->sym) {
        /* 4.32: y expanded over the orbits, X averaged over the group (the blocks are those of the file) */
        double *ys = calloc(u->Osym->m + 1, sizeof(double)), **Xs = NULL;
        if (Xo) {
            Xs = malloc(sizeof(double *) * u->Osym->nblk);
            for (int k = 0; k < u->Osym->nblk; k++) { const int n = abs(u->Osym->bs[k]); Xs[k] = calloc((u->Osym->bs[k] < 0 ? (size_t)n : (size_t)n * n) + 1, sizeof(double)); }
        }
        sym_unmap(u->sym, Xo, yo, Xs, ys, u->Osym->bs);
        if (Xo) { for (int k = 0; k < *pnXo; k++) free(Xo[k]); free(Xo); }
        free(yo);
        Xo = Xs; yo = ys; *pnXo = u->Osym->nblk;
        O = u->Osym;
    }
    if (u->sign) {
        /* 4.33: X embedded into the file's blocks (zero between sign classes), y = 0 on the dropped constraints */
        double *ys = calloc(u->Osign->m + 1, sizeof(double)), **Xs = NULL;
        if (Xo) {
            Xs = malloc(sizeof(double *) * u->Osign->nblk);
            for (int k = 0; k < u->Osign->nblk; k++) { const int n = abs(u->Osign->bs[k]); Xs[k] = calloc((u->Osign->bs[k] < 0 ? (size_t)n : (size_t)n * n) + 1, sizeof(double)); }
        }
        sign_unmap(u->sign, Xo, yo, Xs, ys);
        if (Xo) { for (int k = 0; k < *pnXo; k++) free(Xo[k]); free(Xo); }
        free(yo);
        Xo = Xs; yo = ys; *pnXo = u->Osign->nblk;
        O = u->Osign;
    }
    *pXo = Xo; *pyo = yo;
    return O;
}

static void run_finish(Run *u, const Params *par) {
    if (!u->Otb && !u->fe && !u->dual && !u->alg && !u->sym && !u->sign) return;
    const int infeas = u->status == ST_PINFEAS || u->status == ST_DINFEAS;
    PSOrig *O = map_back(u, par, &u->Xo, &u->yo, &u->nXo, 1);
    double **Xo = u->Xo, *yo = u->yo;
    u->Ofin = O;
    if (infeas || !u->measured || !Xo) return;
    PSResult pr;
    ps_measure(O, Xo, yo, &pr);
    if (getenv("BRISK_FEDBG")) printf("   [finish: reduced pobj %.12e dobj %.12e gap %.2e | file pobj %.12e dobj %.12e gap %.2e]\n",
                                       u->pr.pobj, u->pr.dobj, u->pr.err[5], pr.pobj, pr.dobj, pr.err[5]);
    pr.recovered = u->pr.recovered; pr.unrecovered = u->pr.unrecovered; pr.t += u->pr.t;
    u->pr = pr;
    const double acc = fmax(fmax(pr.err[1], pr.err[2]), fmax(pr.err[4], fmax(fabs(pr.err[5]), fabs(pr.err[6]))));
    int status = u->status;
    if (status == ST_OPTIMAL || status == ST_REDUCED || status == ST_MAXIT || status == ST_NUMERIC || status == ST_TIME) {
        if (acc <= par->tol) status = ST_OPTIMAL;
        else if (acc <= par->red_acc) status = ST_REDUCED;
        else if (status == ST_OPTIMAL || status == ST_REDUCED) status = ST_NUMERIC;
    }
    if (par->verbose > 0 && (u->fe || u->dual || u->alg || u->sym || u->sign))
        printf("   (errors on the file data: %.1e %.1e %.1e %.1e %.1e %.1e; reduced problem %.1e)\n",
               pr.err[1], pr.err[2], pr.err[3], pr.err[4], pr.err[5], pr.err[6], u->acc);
    u->status = status;
    u->acc = acc;
}

/* 4.37: the problem of brisk_run_data; its "file name" is this string, compared by address */
static const BriskData *g_brisk_data = NULL;
static const char g_brisk_data_name[] = "(problem in memory)";

/* 4.39: -fomstart-x / -fomstart-y: a starting point for the first-order engine in the formats
 * that -x and -y write (X: "block i j value" upper triangle; y: one value per line), mapped
 * to the engine's problem. Only when the presolve kept the problem as read up to the scaling
 * (no symmetry or sign reduction, dual form, free elimination, trace bound, facial reduction
 * or chordal conversion): X = X_file / bs, y_i = y_file,i / (cs d_i); a split free pair f =
 * x+ - x- is spread as (f/2, -f/2), the engine's form. Use: continuing a long run. */
static void fom_start_map(Params *par, Run *u, Problem *P) {
    if (!par->fom_start_x && !par->fom_start_y) return;
    const PSOrig *O = u->O;
    int same = !u->alg && !u->sym && !u->sign && !u->dual && !u->fe && u->tbR == 0 && P->ps && P->ps->nrec == 0 && !P->ps->chordal && O && O->m == P->m && O->nblk == P->nblk;
    for (int k = 0; same && k < P->nblk; k++) same = abs(O->bs[k]) == P->blk[k].n && ((O->bs[k] < 0) == (P->blk[k].type == BLK_LP));
    if (!same) { if (par->verbose >= 0) printf("first-order start: ignored (the presolve changed the problem: use -nosym and a problem without free elimination / facial reduction)\n"); return; }
    const int m = P->m, nb = P->nblk;
    if (par->fom_start_y) {
        FILE *f = fopen(par->fom_start_y, "r");
        if (!f) { fprintf(stderr, "brisk: cannot read %s\n", par->fom_start_y); return; }
        double *y = calloc(m + 1, sizeof(double)); int n = 0;
        while (n < m && fscanf(f, "%lf", &y[n]) == 1) n++;
        fclose(f);
        if (n != m) { fprintf(stderr, "brisk: %s has %d values, the problem %d rows: start ignored\n", par->fom_start_y, n, m); free(y); return; }
        for (int i = 0; i < m; i++) y[i] /= P->cs * P->d[i];
        par->fom_y0 = y;
    }
    if (par->fom_start_x) {
        FILE *f = fopen(par->fom_start_x, "r");
        if (!f) { fprintf(stderr, "brisk: cannot read %s\n", par->fom_start_x); return; }
        double **X = calloc(nb + 1, sizeof(double *));
        for (int k = 0; k < nb; k++) X[k] = calloc(P->blk[k].type == BLK_LP ? (size_t)P->blk[k].n : (size_t)P->blk[k].n * P->blk[k].n, sizeof(double));
        int k1, i1, j1; double v; long cnt = 0, bad = 0;
        while (fscanf(f, "%d %d %d %lf", &k1, &i1, &j1, &v) == 4) {
            const int k = k1 - 1, i = i1 - 1, j = j1 - 1;
            if (k < 0 || k >= nb || i < 0 || j < 0 || i >= P->blk[k].n || j >= P->blk[k].n) { bad++; continue; }
            const int n = P->blk[k].n;
            if (P->blk[k].type == BLK_LP) X[k][i] = v / P->bs;
            else { X[k][i + (size_t)j * n] = v / P->bs; X[k][j + (size_t)i * n] = v / P->bs; }
            cnt++;
        }
        fclose(f);
        if (bad) fprintf(stderr, "brisk: %s: %ld entries outside the blocks ignored\n", par->fom_start_x, bad);
        FreePair *fp = NULL; const int nf = free_pairs_detect(P, &fp);
        for (int q = 0; q < nf; q++) { double *x = X[fp[q].blk]; const double fv = x[fp[q].ip] - x[fp[q].im]; x[fp[q].ip] = 0.5 * fv; x[fp[q].im] = -0.5 * fv; }
        free(fp);
        par->fom_X0 = X;
        if (par->verbose >= 0) printf("first-order start: %ld entries of X read from %s%s\n", cnt, par->fom_start_x, par->fom_y0 ? ", y read" : "");
    }
}

static int g_run_no = 0;     /* 4.37: attempts (run_pipeline calls) of this solve */
static int run_pipeline(const char *fname, Params *par, int do_fr, int force_route, Run *u) {
    int race_want = 0; double race_s = 0;        /* 4.42: the first-order engine first (decided after the presolve) */
    memset(u, 0, sizeof(*u));
    g_run_no++;
    Problem *P = &u->P;
    double t0 = wtime();
    if (g_brisk_data && fname == g_brisk_data_name) { if (problem_from_sdpa_data(g_brisk_data, P) != 0) return 2; }   /* 4.37: brisk_run_data */
    else if (read_sdpa(fname, P) != 0) return 2;
    {   /* 4.26 (testing): BRISK_PERTURB=k perturbs b by a few ulps (relative 1e-16 k, signs
         * from a fixed hash): a rounding-sensitivity probe that leaves the problem unchanged
         * far below any tolerance */
        const char *e = getenv("BRISK_PERTURB");
        if (e && atoi(e) != 0) {
            const int k = atoi(e);
            for (int i = 0; i < P->m; i++) {
                unsigned h = (unsigned)(i + 1) * 2654435761u ^ (unsigned)k * 40503u;
                const double sg = (h >> 7) & 1 ? 1.0 : -1.0;
                P->b[i] *= 1.0 + sg * 1e-16 * k;
            }
        }
    }
    u->tread = wtime() - t0;
    double tpre0 = wtime();
    /* 5.7: the matrix-free interior-point method first, on the problems it is made for: one to
     * four dense blocks of order >= 100 and a Schur complement whose factorization costs far
     * more than a matrix-free solve (the truss and vibration problems: tru11 3 s against 100 s).
     * The decision is taken after the presolve, as for the first-order engine below: a problem
     * that the symmetry reduction, the dual form or the chordal conversion changes is the
     * standard method's. The attempt stops as soon as a solve away from the optimum needs more CG
     * steps than the standard method's iteration is worth, or the whole attempt more than a quarter
     * of the standard solve (mfipm.c), and the standard method runs if it does not end OPTIMAL. */
    int mf_want = 0; const int mf_m0 = P->m, mf_nb0 = P->nblk; double mf_s3 = 0, mf_budget = 0, mf_cgcap = 0, mf_prodcap = 0;
    if (par->mf_try > 0 && !par->mfipm && par->fom <= 0 && !par->lralm && !par->dual && !par->bound_side && !par->crossover
        && !par->warm_X && !par->hp_kind && par->dualize <= 0 && !par->fom_start_x && !par->fom_start_y && force_route < 0) {
        int nsdp = 0, maxn = 0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; mf_s3 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n; }
        /* the two costs in the units of the first-order rule below (seconds on its reference
         * machine): the dense Schur factor and block algebra; about 250 products of 4 n^3 flops
         * plus the data (a nonzero counted as 20 n^3-flops: lyap80_big, 1e6 nonzeros, spends its
         * products there) */
        double nz = 0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) for (int t = 0; t < P->blk[k].ncon; t++) nz += P->blk[k].A[t].nnz;
        const double ipm_s = ((double)P->m * P->m * P->m / 3.0 + 20.0 * mf_s3) / 6e9, mf_s = 250.0 * (mf_s3 + 20.0 * nz) / 1e9;
        static double rmin = -1, smin = -1; if (rmin < 0) { const char *e = getenv("BRISK_MFTRYR"); rmin = e ? atof(e) : 3.5; e = getenv("BRISK_MFTRYS"); smin = e ? atof(e) : 5.0; }
        /* lower thresholds (1 and 1.5) when every constraint has an entry in an LP block: the LP
         * part of the Schur complement is then a positive diagonal in every row, which the
         * preconditioner's base has exactly (the truss and vibration problems with their 2m
         * bounds: tru9 and vib9, m = 3,240, 11 and 15 s by the standard method against 1.1 and
         * 4.4 s). Of the problems on which a failed attempt costs 3-20 % at these thresholds
         * (mc_100x80, rose15, rabmo, butcher) none has such a block on all constraints. */
        double rm = rmin, sm = smin;
        if (!getenv("BRISK_MFTRYR") && nsdp >= 1 && nsdp <= 4 && maxn >= 100) {
            char *has = calloc((size_t)P->m + 1, 1); int all = has != NULL, nlp = 0;
            for (int k = 0; k < P->nblk && has; k++) if (P->blk[k].type == BLK_LP) { nlp++; for (int t = 0; t < P->blk[k].ncon; t++) if (P->blk[k].A[t].nnz > 0) has[P->blk[k].con[t]] = 1; }
            for (int c = 0; c < P->m && all; c++) if (!has[c]) all = 0;
            free(has);
            if (all && nlp > 0) { rm = 1.0; sm = 1.5; }
        }
        /* 5.8 (a user's benchmark): also when the dense Schur complement does not fit — the
         * alternative is then the first-order engine, which the matrix-free method beats on
         * these problems (sos_planted_n30_d2, m = 46,375: 25 s against 72 s or a time limit;
         * tru15: 23 s against a segmentation fault in 1.2.1) */
        mf_want = nsdp >= 1 && nsdp <= 4 && maxn >= 100 && ipm_s >= sm && mf_s * rm <= ipm_s;
        if (mf_want) {
            /* not when the dual form will be taken (its rule, as in the first-order decision below)
             * or a large block is the chordal conversion's case (pattern density <= 1 %): those are
             * the standard method's problems, and their symmetry search keeps its budget */
            FreePair *prs = NULL; const int np = free_pairs_detect(P, &prs); free(prs);
            long N = 0; int nlp = 0;
            for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) N += (long)P->blk[k].n * (P->blk[k].n + 1) / 2; else nlp += P->blk[k].n; }
            const long mK = P->m - np, mD = N + (nlp - 2L * np) + np - P->m;
            if (par->dualize < 0 && par->free_elim != 0 && mD >= 1 && mD <= mK / 2 && mK >= 20) mf_want = 0;
            for (int k = 0; k < P->nblk && mf_want; k++) {
                const Block *B = &P->blk[k];
                if (B->type != BLK_SDP || B->n < 200 || par->chordal == 0) continue;
                double pat = 0;
                for (int t = -1; t < B->ncon; t++) { const SpSym *S = t < 0 ? &B->C : &B->A[t]; for (int q = 0; q < S->nnz; q++) pat += S->row[q] != S->col[q]; }
                if (pat / (0.5 * (double)B->n * (B->n + 1.0)) <= 0.01) mf_want = 0;
            }
        }
        /* the attempt's limits, in work rather than time: per solve the CG steps that cost a
         * sixth of one Schur factorization (two solves an iteration, about twice the iterations,
         * and a margin: beyond that the standard method is the cheaper one), and 45 times that
         * in all (a quarter of thirty factorizations; mfipm.c). The time limit, four times the
         * standard solve's estimate, is a backstop. */
        mf_budget = 4.0 * ipm_s;      /* (the estimate is in the units of a faster machine: vib9 has 1.9 and takes 4.4 s matrix-free, 15 s otherwise) */
        mf_cgcap = (double)P->m * P->m * P->m / 3.0 / (6.0 * (4.0 * mf_s3 + 80.0 * nz));
        mf_prodcap = 45.0 * mf_cgcap;      /* 7.5 factorizations, whatever the clamp below does to the cap of a solve */
        if (mf_cgcap < 100) mf_cgcap = 100;
        if (mf_cgcap > 3000) mf_cgcap = 3000;
    }
    if (par->symfile && par->symsign && !strcmp(par->symfile, "auto")) {
        /* 4.33: sign symmetries first (symred.c) */
        u->Osign = ps_orig_build(P);
        u->sign = sign_reduce(P, par->verbose, 1, par->symmin, par->chordal != 0 ? par->chordal_minn : 1 << 30, par->chordal_density);
        if (!u->sign) { ps_orig_free(u->Osign); u->Osign = NULL; }
    }
    if (par->symfile) {
        /* 4.32: exact symmetry reduction from the generators in the file (symred.c) */
        u->Osym = ps_orig_build(P);
        { /* 5.7: with -mfipm 1 the search's budget is a share of the matrix-free solve (about 50
           * products of 4 n^3 flops and 50 eigenvalue decompositions), not of the dense Schur
           * complement's: on the truss problem tru11 the search took 8.9 s of 11.8 s. */
          extern double g_sym_est_cap; extern int g_sym_cap_bound; g_sym_est_cap = 0; g_sym_cap_bound = 0;
          if (mf_want) g_sym_est_cap = 250.0 * mf_s3 / 1e9;
          if (par->mfipm == 1) { double s3 = 0; for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) s3 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n; g_sym_est_cap = 250.0 * s3 / 1e9; } }
        u->sym = sym_reduce(P, par->symfile, par->verbose, par->symtime, par->symnodes, par->symbd, par->symmin, par->chordal != 0 ? par->chordal_minn : 1 << 30, par->chordal_density, par->symsigned);
        if (!u->sym) { ps_orig_free(u->Osym); u->Osym = NULL; }
        if (getenv("BRISK_SETUPT")) printf("   [setup: symmetry reduction %.2fs]\n", wtime() - tpre0);
        if (par->symalg) {
            /* 4.41: hidden and non-permutation symmetry: the *-algebra of each block (symalg.c) */
            u->Oalg = ps_orig_build(P);
            u->alg = alg_reduce(P, par->symalg, par->symalgmax, par->symmin, par->verbose);
            if (!u->alg) { ps_orig_free(u->Oalg); u->Oalg = NULL; }
        }
        if (getenv("BRISK_SYMONLY")) { { printf("SYMONLY m %d -> %d order %d bd %d blocks", u->Osym ? u->Osym->m : P->m, P->m, u->sym ? u->sym->order : 1, u->sym ? u->sym->bd : 0); for (int k = 0; k < P->nblk; k++) printf(" %d", P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n); printf(" time %.2f\n", wtime() - tpre0); } exit(0); }
    }
    if (par->fom < 0) {
        /* 4.31: automatic first-order engine: only when the interior-point method cannot run
         * here - a dense Schur complement (few large blocks: the sparse Schur paths serve the
         * many-small-block models) that does not fit in memory */
        const double ram = brisk_mem_limit();
        int maxn = 0, nsdp = 0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; }
        int use = nsdp <= 64 && maxn >= 200 && 8.0 * (double)P->m * (double)P->m > 0.5 * ram;
        if (!use && par->fom_race > 0 && par->tol >= 0.99e-6 && nsdp <= 64 && maxn >= 200 && !par->mfipm && !par->dual
            && !par->bound_side && !par->crossover && !par->fom_start_x && !par->fom_start_y) {
            /* 4.42: at a tolerance of 1e-6 or looser, the first-order engine is tried first on
             * the problems it is made for (few large dense blocks) when the interior-point
             * solve is expected to be long: theta12 takes it 8 s against 305 s, 1zc.1024 40 s
             * against 278 s. It gets a share (-fomrace, 0.2) of the expected interior-point
             * time; if it has not reached the tolerance by then, the interior-point method
             * runs as before (1et.1024, 1tc.1024: 10 s lost of 140 and 110). A large block
             * with a very sparse pattern is the chordal conversion's case and is left alone.
             * The estimate is that of the symmetry search (dense Schur factor and block
             * algebra over ~20 iterations), calibrated on theta12, 1et.1024 and G55mc. */
            int sparse_big = 0;
            for (int k = 0; k < P->nblk && !sparse_big; k++) {
                const Block *B = &P->blk[k];
                if (B->type != BLK_SDP || B->n < 200) continue;
                double pat = 0;
                for (int t = -1; t < B->ncon; t++) { const SpSym *S = t < 0 ? &B->C : &B->A[t]; for (int q = 0; q < S->nnz; q++) pat += S->row[q] != S->col[q]; }
                if (par->chordal != 0 && pat / (0.5 * (double)B->n * (B->n + 1.0)) <= 0.01) sparse_big = 1;
            }
            double est = (double)P->m * P->m * P->m / 3.0, sp = P->m;
            for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) sp += (double)P->blk[k].ncon * P->blk[k].ncon;
            if (pow(sp, 1.5) < est) est = pow(sp, 1.5);
            for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) est += 20.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
            const double ipm_s = est / 6e9;
            /* the first-order engine's own memory (Anderson history and work matrices, about
             * 400 bytes per entry of the blocks) must fit comfortably: it exits when it does
             * not (shmup5, four blocks of 1861) */
            double sn2 = 0;
            for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) sn2 += (double)P->blk[k].n * P->blk[k].n;
            /* the decision is taken after the presolve (below): a problem the dual form or the
             * chordal conversion changes is the interior-point method's (pglib case793, moment
             * form: 1.8 s) */
            /* ... and its Gram matrix A A' must be sparse. A dense one costs the engine 16 m^2
             * bytes and an m^3/3 factorization, which is the interior-point method's own cost
             * per iteration: neosfbr25 took 3.6 GB (the interior-point method 2.4) and made one
             * iteration in 20 s; thetaG51 spent 4.3 s of its 4.4 s share on the factor.
             * Constraint pairs that share a position bound the off-diagonal count. */
            int gram_dense = 0;
            if (!sparse_big && ipm_s >= 20.0) {
                double pairs = 0;
                for (int k = 0; k < P->nblk; k++) {
                    const Block *B = &P->blk[k];
                    const size_t np = B->type == BLK_SDP ? (size_t)B->n * (B->n + 1) / 2 : (size_t)B->n;
                    int *cnt = (int *)calloc(np ? np : 1, sizeof(int));
                    if (!cnt) { pairs = 1e300; break; }
                    for (int t = 0; t < B->ncon; t++) {
                        const SpSym *S = &B->A[t];
                        for (int q = 0; q < S->nnz; q++) {
                            const size_t r = S->row[q] >= S->col[q] ? S->row[q] : S->col[q], c = S->row[q] >= S->col[q] ? S->col[q] : S->row[q];
                            const size_t id = B->type == BLK_SDP ? r * (r + 1) / 2 + c : r;
                            if (id < np) pairs += cnt[id]++;
                        }
                    }
                    free(cnt);
                }
                gram_dense = pairs > 0.01 * (double)P->m * P->m;
            }
            if (!sparse_big && ipm_s >= 20.0 && !gram_dense && 400.0 * sn2 <= 0.3 * ram) { race_want = 1; race_s = ipm_s; }
        }
        if (use && par->chordal != 0 && !getenv("BRISK_FOMNODEFER")) {
            /* 4.40: a large block whose aggregate pattern is very sparse (AC-OPF: density
             * ~1e-3) is a case for the chordal conversion and the sparse Schur complement,
             * which the 8 m^2 rule does not see: the decision waits until after the
             * conversion, and the pipeline restarts with the first-order engine if it did
             * not convert (rc 4). Off for patterns above 1% (max-cut/theta-type blocks). */
            for (int k = 0; k < P->nblk && use; k++) {
                const Block *B = &P->blk[k];
                if (B->type != BLK_SDP || B->n < par->chordal_minn || B->n < 200) continue;
                double pat = 0;
                for (int t = -1; t < B->ncon; t++) { const SpSym *S = t < 0 ? &B->C : &B->A[t]; for (int q = 0; q < S->nnz; q++) pat += S->row[q] != S->col[q]; }
                const double dens = pat / (0.5 * (double)B->n * (B->n + 1.0));
                if (dens <= 0.01) {
                    use = 0; u->fom_chordal = 1;
                    if (par->verbose >= 0) printf("first-order engine: deferred until after the chordal conversion (m = %d, a block of n = %d with pattern density %.1e)\n", P->m, B->n, dens);
                }
            }
        }
        if (use && (par->dualize != 0 || par->free_elim != 0) && nsdp <= 32 && !getenv("BRISK_FOMNODEFER")) {
            /* 4.39 (ISSUES 22): the rows the dual form or the free elimination would leave (their
             * own rules, dualize.c / freeelim.c); when those fit, the decision waits until after
             * them, and the pipeline restarts with the first-order engine if they did not */
            FreePair *pr = NULL; const int np = free_pairs_detect(P, &pr); free(pr);
            long N = 0; int nlp = 0;
            for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) N += (long)P->blk[k].n * (P->blk[k].n + 1) / 2; else nlp += P->blk[k].n; }
            const long mK = P->m - np, mD = N + (nlp - 2L * np) + np - P->m;
            const int dual_rule = par->dualize > 0 || (par->dualize < 0 && par->free_elim != 0 && mD >= 1 && mD <= mK / 2 && mK >= 20);
            const double mp = dual_rule ? (double)mD : (par->free_elim != 0 ? (double)mK : (double)P->m);
            if (8.0 * mp * mp <= 0.5 * ram) {
                use = 0; u->fom_check = 1;
                if (par->verbose >= 0) printf("first-order engine: deferred (m = %d as read; the %s should leave about %.0f rows)\n", P->m, dual_rule ? "dual form" : "free elimination", mp);
            }
        }
        par->fom = use ? 1 : 0;
        if (use) {
            if (par->free_elim < 0) par->free_elim = 0;
            if (par->dualize < 0) par->dualize = 0;
            if (par->tracebound < 0) par->tracebound = 0;
            if (par->chordal < 0) par->chordal = 0;
            if (par->verbose >= 0) printf("first-order engine: the dense Schur complement (m = %d, %.1f GB) does not fit in memory\n", P->m, 8.0 * (double)P->m * P->m / 1e9);
        }
    }
    if (par->dualize > 0 || (par->dualize < 0 && par->free_elim != 0)) {
        /* 4.30: the dual form when its Schur complement is the smaller one (SOS programs
         * given in the wrong form); its y pairs are then eliminated below */
        PSOrig *Of = ps_orig_build(P);
        u->dual = dualize_apply(P, par->dualize < 0, par->verbose >= 0 ? par->verbose : 0);
        if (u->dual) u->Odual = Of; else ps_orig_free(Of);
        if (getenv("BRISK_SETUPT")) printf("   [setup: dual form %.2fs]\n", wtime() - tpre0);
    }
    u->O = ps_orig_build(P);
    if (par->free_elim) {
        PSFreeElim *F = free_eliminate(P, par->verbose >= 0 ? par->verbose : 0, par->free_elim_fill, par->free_elim < 0);
        if (getenv("BRISK_SETUPT")) printf("   [setup: free elimination done at %.2fs]\n", wtime() - tpre0);
        if (F) { u->fe = F; u->Ofile = u->O; u->O = ps_orig_build(P); }
    }
    if (u->fom_check && 8.0 * (double)P->m * (double)P->m > 0.5 * brisk_mem_limit()) {
        if (par->verbose >= 0) printf("first-order engine: m = %d after the dual form and free elimination still does not fit; restarting with the first-order engine\n", P->m);
        return 4;
    }
    if (par->tracebound > 0 || (par->tracebound < 0 && u->fe && !u->fe->cheap)) {
        double bmax = 0;
        for (int i = 0; i < P->m; i++) bmax = fmax(bmax, fabs(P->b[i]));
        u->tbR = par->tracebound > 0 ? par->tracebound : 1e7 * (1.0 + bmax);
        tracebound_add(P, u->tbR);
        u->Otb = u->O; u->O = ps_orig_build(P);
        if (par->verbose >= 0) printf("presolve: trace bound sum tr(X) + s = %.3g added (row %d)\n", u->tbR, P->m);
    }
    P->ps = ps_new(P);
    if (getenv("BRISK_SETUPT")) printf("   [setup: orig copy + free elimination %.2fs]\n", wtime() - tpre0);
    if (do_fr) {
        double c0 = (double)P->m * P->m * P->m / 3.0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) c0 += 10.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        Problem Pc; problem_clone(P, &Pc);       /* 4.30: restored if the reduction is rejected */
        facial_reduction(P, par->verbose);
        double c1 = (double)P->m * P->m * P->m / 3.0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) c1 += 10.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        /* 4.20: chains of dependent reductions (depth >= 2) mean a singularity degree >= 2;
         * the removed duals then often cannot be recovered (the dual optimum of the problem
         * as read is not attained) and the problem is solved a second time without facial
         * reduction. Unless the reduction makes the solve much cheaper (e_moment: 145x),
         * solve without it from the start (rose13, taha1a, roa_acrobot: 1.9-3.4x). */
        if (do_fr == 1 && P->fr_depth >= 2 && P->ps->nrec > 0 && c0 < par->fr_gain * c1) {
            if (par->verbose >= 0)
                printf("presolve: facial reduction of depth %d saves only %.1fx per iteration; solving without it\n", P->fr_depth, c0 / c1);
            ps_free(P->ps);
            problem_free(P);
            *P = Pc;
            P->ps = ps_new(P);
        } else problem_free(&Pc);
    }
    u->fr_removed = P->ps->nrec;
    if (getenv("BRISK_SETUPT")) printf("   [setup: facial reduction done at %.2fs]\n", wtime() - tpre0);
    if (par->chordal != 0) {
        int nc = chordal_convert(P, par, par->verbose ? 1 : 0);
        if (nc && P->ps->mom) {
            /* the moment form has the original y as free pairs: embedding, pairs native
             * (augmented Lagrangian on the sparse Schur complement) */
            if (!getenv("BRISK_MOM_NOHSD")) par->hsd = 1;
            /* NT: on the moment form HKM stalled more often (case300: 1e-6 against 3e-8) */
            /* (with the inequality multipliers eliminated, HKM is faster and as accurate:
             * case240/300/500/793 1.2-1.4x) */
            if (par->hsd_dir < 0) par->hsd_dir = P->ps->mom->rowmap ? 0 : 1;
        }
        if (nc) {
            P->ps->chordal = 1;
            if (par->verbose >= 0)
                printf("chordal decomposition: %d block(s) split into cliques, m = %d\n", nc, P->m);
        }
    }
    if (getenv("BRISK_MFTRYDRY")) { extern int g_sym_cap_bound; printf("MFTRY want %d chordal %d dual %d sym %d m %d->%d search cut %d\n", mf_want, P->ps->chordal, u->dual != NULL, u->sym || u->sign || u->alg, mf_m0, P->m, g_sym_cap_bound); exit(0); }
    if (mf_want && !P->ps->chordal && !u->dual && !u->sym && !u->sign && !u->alg && P->m == mf_m0 && P->nblk == mf_nb0) {
        if (par->verbose >= 0) printf("matrix-free interior-point method first: m = %d and few dense blocks (-mftry 0 turns this off)\n", P->m);
        par->mf_try_budget = mf_budget;
        par->mf_trial = (int)mf_cgcap;
        par->mf_trial_total = (long)fmin(mf_prodcap, 1e15);
        par->mf_try_fits = 8.0 * (double)P->m * (double)P->m <= 0.5 * brisk_mem_limit();
        return 7;
    }
    { /* the attempt is not made after all (the presolve changed the problem), and the symmetry
       * search had the matrix-free budget and ran into it: once more, as without the rule */
      extern int g_sym_cap_bound;
      if (mf_want && g_sym_cap_bound && !u->sym) { g_sym_cap_bound = 0; return 8; } }
    if (race_want && !P->ps->chordal && !u->dual) {
        par->fom_race_budget = fmax(3.0, par->fom_race * race_s);
        if (par->verbose >= 0) printf("first-order engine first: tolerance %.0e and an interior-point solve of about %.0f s expected; it gets %.0f s (-fomrace 0 turns this off)\n", par->tol, race_s, par->fom_race_budget);
        return 6;
    }
    if (par->chordal_need && !P->ps->chordal && 8.0 * (double)P->m * (double)P->m > 0.5 * brisk_mem_limit()) {
        /* 4.40: a chordal re-solve whose form does not convert would solve the unconverted
         * problem, which does not fit (dense case13659: the overlap-equality form declined,
         * the unconverted solve ran out of memory and the first result was lost) */
        if (par->verbose >= 0) printf("chordal: this form does not convert and the unconverted problem (m = %d) does not fit; re-solve skipped\n", P->m);
        return 5;
    }
    if (u->fom_chordal && !P->ps->chordal && 8.0 * (double)P->m * (double)P->m > 0.5 * brisk_mem_limit()) {
        if (par->verbose >= 0) printf("first-order engine: no chordal conversion and m = %d does not fit; restarting with the first-order engine\n", P->m);
        return 4;
    }
    if (getenv("BRISK_WRITE_CONV")) {          /* the presolved/converted problem as SDPA */
        FILE *f = fopen(getenv("BRISK_WRITE_CONV"), "w");
        if (f) {
            fprintf(f, "\" converted by brisk (facial reduction, chordal decomposition)\n%d\n%d\n", P->m, P->nblk);
            for (int k = 0; k < P->nblk; k++) fprintf(f, "%d ", P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n);
            fprintf(f, "\n");
            for (int i = 0; i < P->m; i++) fprintf(f, "%.17g ", P->b[i]);
            fprintf(f, "\n");
            for (int k = 0; k < P->nblk; k++) {
                const Block *B = &P->blk[k];
                for (int t = -1; t < B->ncon; t++) {
                    const SpSym *S = t < 0 ? &B->C : &B->A[t];
                    const int row = t < 0 ? 0 : B->con[t] + 1;
                    for (int q = 0; q < S->nnz; q++) {
                        int i = S->row[q], j = B->type == BLK_LP ? S->row[q] : S->col[q];
                        if (i > j) { int tt = i; i = j; j = tt; }
                        double v = t < 0 ? -S->val[q] : S->val[q];
                        if (v != 0) fprintf(f, "%d %d %d %d %.17g\n", row, k + 1, i + 1, j + 1, v);
                    }
                }
            }
            fclose(f);
        }
        exit(0);
    }
    if (getenv("BRISK_SETUPT")) printf("   [setup: conversion done at %.2fs]\n", wtime() - tpre0);
    problem_prepare(P, par);
    u->tpre = wtime() - tpre0;
    if (getenv("BRISK_SETUPT")) printf("   [setup: problem_prepare done at %.2fs]\n", u->tpre);
    long nroute[4] = { 0, 0, 0, 0 };
    int maxn = 0, nsdp = 0, nlp = 0, nlr = 0, ndict = 0, dq = 0, dstr = 0;
    long lrR = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        if (B->type == BLK_LP) { nlp += B->n; continue; }
        nsdp++;
        if (B->n > maxn) maxn = B->n;
        if (B->lowrank) { nlr++; lrR += B->lrR; }
        if (B->dict) { ndict++; dq += B->dnu + B->dnv; dstr += B->dstruct; }
        for (int t = 0; t < B->ncon; t++) {
            if (force_route >= 0 && B->route[t] != 2) B->route[t] = (unsigned char)force_route;
            nroute[B->route[t]]++;
        }
    }
    if (par->verbose) {
        printf("BRISK %s interior-point SDP solver\n", BRISK_VERSION);
        printf("problem %s: m = %d, %d SDP blocks (max n = %d), %d LP variables, read %.2fs\n",
               fname, P->m, nsdp, maxn, nlp, u->tread);
        if (par->verbose >= 2) printf("memory limit: %.1f GB (the machine's memory, or the container's or the process's limit when lower)\n", brisk_mem_limit() / 1e9);
        printf("Schur routes: %ld sparse-sparse, %ld row-product, %ld pattern row-product, %ld dense-BLAS3; %d low-rank block(s), total rank %ld\n",
               nroute[0], nroute[1], nroute[3], nroute[2], nlr, lrR);
        if (ndict)
            printf("Dictionary route: %d block(s), lifted size %d, %d constraint parts through shared vectors\n",
                   ndict, dq, dstr);
    }
    Result *R = &u->R;
    u->y = calloc(P->m + 1, sizeof(double));
    u->X = malloc(sizeof(double *) * (P->nblk + 1));
    for (int k = 0; k < P->nblk; k++) {
        size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
        u->X[k] = calloc(len + 1, sizeof(double));
    }
    int st_ = -1;
    double tdual = 0;
    /* 4.37 bound mode: the tracker of the side asked for, on this problem (the dual form
     * exchanges the sides); one X and one y per slot, best and anchor */
    BoundTrack bt;
    memset(&bt, 0, sizeof(bt));
    if (par->bound_side) {
        bt.side = u->dual ? 3 - par->bound_side : par->bound_side;
        bt.tol = par->bound_track_tol > 0 ? par->bound_track_tol : 100.0 * par->bound_tol;
        bt.tol_a = fmax(bt.tol, 1e-2);     /* anchors: the barrier-metric step of bound.c makes them feasible */
        bt.m = P->m; bt.nblk = P->nblk;
        double bytes = 8.0 * (BOUND_NANCHOR + 1) * (P->m + 1);
        for (int k = 0; k < P->nblk; k++) bytes += 8.0 * (BOUND_NANCHOR + 1) * (double)(P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n);
        if (bytes > fmin(1024.0 * 1048576.0 * brisk_mem_scale(), 0.15 * brisk_mem_limit())) {
            bt.disabled = 1;
            if (par->verbose >= 0) printf("bound: the candidates (%.0f MB) do not fit; only the returned points are certified\n", bytes / 1048576.0);
        } else {
            for (int s = 0; s <= BOUND_NANCHOR; s++) {
                double **X = calloc(P->nblk + 1, sizeof(double *));
                for (int k = 0; k < P->nblk; k++) {
                    const size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
                    X[k] = calloc(len + 1, sizeof(double));
                }
                double *yv = calloc(P->m + 1, sizeof(double));
                if (s == 0) { bt.X = X; bt.y = yv; } else { bt.Xa[s - 1] = X; bt.ya[s - 1] = yv; }
            }
            par->btrack = &bt;
        }
    }
    if (P->m > 0 && !par->fom && !par->mfipm && !par->lralm && !par->dual && par->sparse_schur != 0 && !par->mf_trial
        && 8.0 * (double)P->m * (double)P->m > 0.5 * brisk_mem_limit() && schur_dense_needed(P, par)) {
        /* 5.8 (a user's report): the pattern looked sparse to the early routing but neither
         * the sparse factorization nor the envelope takes it, and the dense Schur complement
         * does not fit: the first-order engine instead of an exit with advice (pglib case89
         * at minimal order, m = 34,898) */
        if (par->verbose >= 0) printf("the Schur complement would be dense (m = %d, %.1f GB) and does not fit in memory; restarting with the first-order engine\n", P->m, 8.0 * (double)P->m * P->m / 1e9);
        return 4;
    }
    if (P->m == 0) { solve_m0(P, R, u->X); st_ = 0; }
    else if (par->mfipm == 2) {
        /* 5.4: the hybrid: the matrix-free method until its CG struggles or the merit is below
         * mf_hand, then the standard method from that iterate (a few factorizations of the
         * Schur complement instead of thirty) */
        const int ntb = brisk_threads_busy(par->verbose); brisk_log_threads(par->verbose);
        double **Xi = malloc(sizeof(double *) * (P->nblk + 1)), **Zi = malloc(sizeof(double *) * (P->nblk + 1)), *yi = calloc(P->m + 1, sizeof(double));
        for (int k = 0; k < P->nblk; k++) {
            const size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
            Xi[k] = calloc(len + 1, sizeof(double)); Zi[k] = calloc(len + 1, sizeof(double));
        }
        const double tm0 = wtime();
        const int hand = mfipm_solve_hand(P, par, R, Xi, Zi, yi, par->mf_hand, par->mf_handcg);
        const double tmf = wtime() - tm0;
        if (ntb > 0) omp_set_num_threads(ntb);
        if (hand == 2 && par->verbose >= 0) printf("matrix-free IPM: CG gave up far from the optimum; the standard method starts cold\n");
        if (hand == 1) {
            par->warm_X = Xi; par->warm_Z = Zi; par->warm_y = yi;
            if (!(par->warm_lam > 0)) par->warm_lam = 1.0;
            memset(R, 0, sizeof(*R));
            brisk_solve(P, par, R, u->y, u->X);
            R->t_total += tmf;
            par->warm_X = par->warm_Z = NULL; par->warm_y = NULL;
            st_ = 0;
        } else {
            /* ended by itself (optimal, or no progress): its result stands */
            if (par->verbose >= 0 && hand == 0) printf("matrix-free IPM: ended without a hand-off (status %d); re-solving by the standard method\n", R->status);
            if (hand == 0 || hand == 2) { memset(R, 0, sizeof(*R)); brisk_solve(P, par, R, u->y, u->X); R->t_total += tmf; st_ = 0; }
            else st_ = -1;
        }
        for (int k = 0; k < P->nblk; k++) { free(Xi[k]); free(Zi[k]); }
        free(Xi); free(Zi); free(yi);
    }
    else if (par->mfipm > 0) {
        /* 4.34: the matrix-free interior-point method (mfipm.c); its result is what the run
         * returns, measured on the file data */
        const int ntb = brisk_threads_busy(par->verbose); brisk_log_threads(par->verbose);   /* 4.42: only the free cores (solver.c) */
        if (par->mf_trial && !getenv("BRISK_MFTRYNOWARM")) {
            /* 5.7: the automatic first attempt: if it does not end OPTIMAL but came below a
             * merit of 1e-3, the standard method starts from that iterate (the hybrid's
             * hand-off; tru13: 9 factorizations instead of 30) */
            double **Xi = malloc(sizeof(double *) * (P->nblk + 1)), **Zi = malloc(sizeof(double *) * (P->nblk + 1)), *yi = calloc(P->m + 1, sizeof(double));
            for (int k = 0; k < P->nblk; k++) {
                const size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
                Xi[k] = calloc(len + 1, sizeof(double)); Zi[k] = calloc(len + 1, sizeof(double));
            }
            int taken = 0; const double tm0 = wtime();
            st_ = mfipm_solve_snap(P, par, R, u->y, u->X, Xi, Zi, yi, par->mf_hand, &taken);
            const double tmf = wtime() - tm0;
            if (ntb > 0) omp_set_num_threads(ntb);
            if (st_ >= 0 && R->status != ST_OPTIMAL && taken && par->mf_try_fits && !(par->mf_try_tl > 0 && wtime() - par->t_start >= par->mf_try_tl)) {
                if (par->verbose >= 0) printf("matrix-free interior-point method first: not optimal after %.1f s; the standard method from its iterate of merit %.0e\n", tmf, par->mf_hand);
                brisk_note("matrix-free interior-point method first (%.1f s, not optimal), then the standard method from its iterate of merit %.0e", tmf, par->mf_hand);
                const double tl = par->timelimit; const int tr = par->mf_trial; const double wl = par->warm_lam;
                par->timelimit = par->mf_try_tl; par->mf_trial = 0;
                par->warm_X = Xi; par->warm_Z = Zi; par->warm_y = yi;
                if (!(par->warm_lam > 0)) par->warm_lam = 1.0;
                /* the attempt's own result is kept for the case that the standard method is cut
                 * (time limit) before it is better */
                const Result R0 = *R;
                double **X0 = malloc(sizeof(double *) * (P->nblk + 1)), *y0 = malloc(sizeof(double) * (P->m + 1));
                memcpy(y0, u->y, sizeof(double) * P->m);
                for (int k = 0; k < P->nblk; k++) {
                    const size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
                    X0[k] = malloc(sizeof(double) * (len + 1)); memcpy(X0[k], u->X[k], sizeof(double) * len);
                }
                memset(R, 0, sizeof(*R));
                brisk_solve(P, par, R, u->y, u->X);
                R->t_total += tmf;
                if (R->status != ST_OPTIMAL && !(fmax(fabs(R->relgap), fmax(R->pinf, R->dinf)) < fmax(fabs(R0.relgap), fmax(R0.pinf, R0.dinf)))) {
                    const double tt = R->t_total;
                    *R = R0; R->t_total = tt;
                    memcpy(u->y, y0, sizeof(double) * P->m);
                    for (int k = 0; k < P->nblk; k++) memcpy(u->X[k], X0[k], sizeof(double) * (P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n));
                }
                for (int k = 0; k < P->nblk; k++) free(X0[k]);
                free(X0); free(y0);
                par->warm_X = par->warm_Z = NULL; par->warm_y = NULL; par->warm_lam = wl;
                par->timelimit = tl; par->mf_trial = tr;
                st_ = 0;
            }
            for (int k = 0; k < P->nblk; k++) { free(Xi[k]); free(Zi[k]); }
            free(Xi); free(Zi); free(yi);
        } else {
            st_ = mfipm_solve(P, par, R, u->y, u->X);
            if (ntb > 0) omp_set_num_threads(ntb);
        }
    }
    else if (par->fom > 0) {
        /* 4.31: the first-order engine (fom.c); no interior-point fallback: its result is
         * what the run returns, measured on the file data like any other */
        fom_start_map(par, u, P);
        const int ntb = brisk_threads_busy(par->verbose); brisk_log_threads(par->verbose);
        st_ = fom_solve(P, par, R, u->y, u->X);
        if (st_ == -7) {
            /* 5.7: neither the Schur complement nor the factor of A A' fits in memory: the
             * matrix-free interior-point method needs no m x m matrix (tru15, m = 25,200: the
             * run ended in a segmentation fault; now 23 s to 6.6e-8) */
            if (par->verbose >= 0) printf("the factor of A A' does not fit in memory either: the matrix-free interior-point method\n");
            st_ = mfipm_solve(P, par, R, u->y, u->X);
        }
        if (ntb > 0) omp_set_num_threads(ntb);
        if (par->fom_X0) { for (int k = 0; k < P->nblk; k++) free(par->fom_X0[k]); free(par->fom_X0); par->fom_X0 = NULL; }
        free(par->fom_y0); par->fom_y0 = NULL;
    }
    else if (par->dual) {
        double t0d = wtime();
        const int ntb = brisk_threads_busy(par->verbose); brisk_log_threads(par->verbose);
        st_ = par->dual == 2 ? dual_solve(P, par, R, u->y, u->X) : dsdp_solve(P, par, R, u->y, u->X);
        if (ntb > 0) omp_set_num_threads(ntb);
        tdual = wtime() - t0d;
        R->t_total = tdual;
        /* The dual method is a specialist (max-cut, theta, sparse relaxations). Anything
         * it does not certify as optimal is re-solved by the primal-dual method, so -dual
         * never returns a worse answer than the default - it only costs time. */
        /* a reduced-accuracy result within retry_acc (1e-7) is kept, as in the default method */
        const int dual_good = R->status == ST_OPTIMAL ||
            (R->status == ST_REDUCED && fmax(R->relgap, R->pinf) <= par->retry_acc);
        if (st_ >= 0 && !dual_good && !getenv("BRISK_DUAL_NOFALLBACK")) {
            if (par->verbose >= 0)
                printf("dual method: not certified (status %d), falling back to the primal-dual method\n", R->status);
            memset(R, 0, sizeof(*R));
            st_ = -1;
        }
    }
    if (st_ < 0) { brisk_solve(P, par, R, u->y, u->X); R->t_total += tdual; }
    par->btrack = NULL;
    /* ---- postsolve: the solution of the problem as read, and its errors */
    u->yo = calloc(u->O->m + 1, sizeof(double));
    PSResult *pr = &u->pr;
    postsolve(P, u->O, u->X, u->y, &u->Xo, u->yo, pr, par->verbose);
    int status = R->status;
    double acc = 0;
    u->measured = pr->have_x;
    if (u->measured) {
        acc = fmax(fmax(pr->err[1], pr->err[2]), fmax(pr->err[4], fmax(fabs(pr->err[5]), fabs(pr->err[6]))));
        /* the status reflects the returned point as measured on the original data */
        if (status == ST_OPTIMAL || status == ST_REDUCED || status == ST_MAXIT || status == ST_NUMERIC || status == ST_TIME) {
            if (acc <= par->tol) status = ST_OPTIMAL;
            else if (acc <= par->red_acc) status = ST_REDUCED;
            else if (status == ST_OPTIMAL || status == ST_REDUCED) status = ST_NUMERIC;
        }
    } else {
        /* chordal conversion: primal errors from the converted problem, dual side measured */
        acc = fmax(fmax(R->relgap, R->pinf), fmax(pr->err[4], R->relcomp));
        if ((status == ST_OPTIMAL || status == ST_REDUCED) && pr->err[4] > par->red_acc) status = ST_NUMERIC;
    }
    u->status = status;
    u->acc = acc;
    u->nXo = u->O->nblk;
    u->Ofin = u->O;
    run_finish(u, par);
    if (par->bound_side) {
        /* 4.37: certificates from the tracked candidate (with its anchor) and from the
         * returned point, on the problem as read; the better one is the run's */
        const double tb0 = wtime();
        double **Xt = NULL, *yt = NULL, **XA[BOUND_NANCHOR], *YA[BOUND_NANCHOR];
        int nXt = 0, nXA[BOUND_NANCHOR], na = 0;
        for (int s = -1; s < bt.na; s++) {
            if (s < 0 && !bt.have) continue;
            double **Xm = NULL, *ym = calloc(u->O->m + 1, sizeof(double));
            PSResult prc;
            postsolve(P, u->O, s < 0 ? bt.X : bt.Xa[s], s < 0 ? bt.y : bt.ya[s], &Xm, ym, &prc, 0);
            int nX = u->O->nblk;
            map_back(u, par, &Xm, &ym, &nX, 0);
            if (s < 0) { Xt = Xm; yt = ym; nXt = nX; }
            else { XA[na] = Xm; YA[na] = ym; nXA[na] = nX; na++; }
        }
        PSOrig *Of = u->Ofin ? u->Ofin : u->O;
        if (par->bound_anchor && par->bound_side == 2 && na < BOUND_NANCHOR) {
            /* a y given by the user as the anchor of the dual side (the problem as read) */
            FILE *fa = fopen(par->bound_anchor, "r");
            if (!fa) fprintf(stderr, "brisk: cannot read %s\n", par->bound_anchor);
            else {
                double *ya = calloc(Of->m + 1, sizeof(double)); int k = 0;
                while (k < Of->m && fscanf(fa, "%lf", &ya[k]) == 1) k++;
                fclose(fa);
                if (k < Of->m) { fprintf(stderr, "brisk: %s has %d values, the problem %d rows: anchor ignored\n", par->bound_anchor, k, Of->m); free(ya); }
                else { XA[na] = NULL; YA[na] = ya; nXA[na] = 0; na++; }
            }
        }
        BoundCert c1, c2;
        memset(&c1, 0, sizeof(c1)); memset(&c2, 0, sizeof(c2));
        char src[48];
        snprintf(src, sizeof(src), "iteration %d", bt.it);
        { extern double g_bound_deadline; g_bound_deadline = par->timelimit > 0 ? par->t_start + par->timelimit : 0; }
        if (bt.have) bound_certify(Of, par->bound_side, Xt, yt, na, XA, YA, par->bound_margin, par->bound_tol, src, par->verbose, &c1);
        bound_certify(Of, par->bound_side, u->Xo, u->yo, na, XA, YA, par->bound_margin, par->bound_tol, "the returned point", par->verbose, &c2);
        if (bound_better(&c1, &c2)) { u->cert = c1; bound_cert_free(&c2); } else { u->cert = c2; bound_cert_free(&c1); }
        u->cert.attempt = g_run_no;
        if (par->verbose > 0 && u->cert.have)
            printf("   bound: %s from %s: %.10e (residual %.1e, lambda_min %.1e rel; %d candidates offered, %d anchors, %.2fs)\n",
                   par->bound_side == 1 ? "upper" : "lower", u->cert.src, u->cert.value, u->cert.resid, u->cert.lammin, bt.offers, na, wtime() - tb0);
        bound_free_blocks(Xt, nXt); free(yt);
        for (int a = 0; a < na; a++) { bound_free_blocks(XA[a], nXA[a]); free(YA[a]); }
    }
    if (bt.X) {
        bound_free_blocks(bt.X, bt.nblk); free(bt.y);
        for (int s = 0; s < BOUND_NANCHOR; s++) { bound_free_blocks(bt.Xa[s], bt.nblk); free(bt.ya[s]); }
    }
    return 0;
}

static const Params *g_rank_par = NULL;     /* 4.37: bound mode ranks runs by their certificates */
/* ranking of a finished run: lower is better */
static double run_rank(const Run *u) {
    switch (u->status) {
    case ST_OPTIMAL: return 0 + fmin(u->acc, 1.0);
    case ST_PINFEAS: case ST_DINFEAS: return 0.5;
    case ST_REDUCED: return 1 + fmin(u->acc, 1.0);
    default: return 2 + fmin(u->acc, 1.0);
    }
}

/* 5.8: the events of a run (method switches, the endgame, crossover, bounds, ...) for the
 * summary at the end of the log; cleared by brisk_main */
#include <stdarg.h>
static char g_notes[24][200]; static int g_nnotes = 0;
void brisk_note(const char *fmt, ...) {
    if (g_nnotes >= 24) return;
    va_list ap; va_start(ap, fmt); vsnprintf(g_notes[g_nnotes], sizeof g_notes[0], fmt, ap); va_end(ap);
    g_nnotes++;
}
static void notes_clear(void) { g_nnotes = 0; }
static int g_acc_high = 0;      /* -acc high was given: the hint below names the high-precision solver only */
static int g_has_bound = 0, g_has_certify = 0;   /* -bound / -certify were given: the hint on bounds says only what is left */
/* what the advice at the end of the log need not repeat (the options from argv[i0] on) */
static void advice_scan(int argc, char **argv, int i0) {
    g_acc_high = g_has_bound = g_has_certify = 0;
    for (int i = i0; i < argc; i++) {
        if (!argv[i]) continue;
        if (i + 1 < argc && !strcmp(argv[i], "-acc") && argv[i + 1] && !strcmp(argv[i + 1], "high")) g_acc_high = 1;
        if (!strcmp(argv[i], "-bound")) g_has_bound = 1;
        if (!strcmp(argv[i], "-certify")) g_has_certify = 1;
    }
}

/* ---- who called ---------------------------------------------------------------------------
 * The advice at the end of the log is written in the syntax of the caller: the command line,
 * a C program, or one of BRISK's interfaces and the command of it that was used. An interface
 * says so by the option pair "-caller <tag>" (python:solve_sdpa, python:solve_file,
 * python:solve_sedumi, cvxpy, julia:solve_sdpa, julia:solve_sdpa_data, julia:solve_sedumi, jump,
 * matlab:brisk_sdpa, matlab:brisk_sedumi), which the library's entry points (brisk_run,
 * brisk_run_sedumi) take out of the option list; without it a library call is a C program's,
 * and the executable's is the command line (brisk_caller NULL). An unknown tag gets the
 * command-line syntax. */
const char *brisk_caller = NULL;
#ifdef BRISK_LIBRARY
static int g_caller_depth = 0;
static char g_caller_buf[64];
static char **caller_enter(int *argc, char **argv, const char *deflt) {
    char **av = malloc(sizeof(char *) * ((size_t)(*argc > 0 ? *argc : 0) + 1));
    const char *tag = NULL;
    int n = 0;
    if (!av) return NULL;
    for (int i = 0; i < *argc; i++) {
        if (argv[i] && !strcmp(argv[i], "-caller") && i + 1 < *argc) { tag = argv[++i]; continue; }
        av[n++] = argv[i];
    }
    av[n] = NULL; *argc = n;
    if (g_caller_depth++ == 0) {            /* (a nested call - the semidefinite solver under a SeDuMi problem - keeps the tag) */
        if (tag || deflt) { snprintf(g_caller_buf, sizeof g_caller_buf, "%s", tag ? tag : deflt); brisk_caller = g_caller_buf; }
        else brisk_caller = NULL;
    }
    return av;
}
static void caller_leave(char **av) { free(av); if (--g_caller_depth <= 0) { g_caller_depth = 0; brisk_caller = NULL; } }
#endif

enum { CK_CLI, CK_C, CK_PY, CK_CVXPY, CK_JL, CK_JUMP, CK_ML };
static int caller_kind(const char **cmd) {
    const char *t = brisk_caller;
    *cmd = "";
    if (!t) return CK_CLI;
    if (!strcmp(t, "c")) return CK_C;
    if (!strcmp(t, "cvxpy")) return CK_CVXPY;
    if (!strcmp(t, "jump")) return CK_JUMP;
    if (!strncmp(t, "python:", 7) && t[7]) { *cmd = t + 7; return CK_PY; }
    if (!strncmp(t, "julia:", 6) && t[6]) { *cmd = t + 6; return CK_JL; }
    if (!strncmp(t, "matlab:", 7) && t[7]) { *cmd = t + 7; return CK_ML; }
    return CK_CLI;
}
/* The pieces of the advice in the caller's syntax (each returns one of twelve rotating buffers):
 * syn_kv: the option `name` with the value `v` (a string if q; v NULL: a flag that is set) as
 * it is typed in the place syn_head names; syn_val: a string value. */
static char *syn_buf(void) { static char buf[12][160]; static int ib = 0; return buf[ib = (ib + 1) % 12]; }
#define SYN_L 160
static const char *syn_kv(const char *name, const char *v, int q) {
    char *o = syn_buf();
    const char *cmd;
    switch (caller_kind(&cmd)) {
    case CK_C:
        if (v) snprintf(o, SYN_L, "\"-%s\", \"%s\"", name, v); else snprintf(o, SYN_L, "\"-%s\"", name);
        break;
    case CK_PY:
        if (!v) snprintf(o, SYN_L, "\"%s\": True", name);
        else if (q) snprintf(o, SYN_L, "\"%s\": \"%s\"", name, v);
        else snprintf(o, SYN_L, "\"%s\": %s", name, v);
        break;
    case CK_CVXPY:
        if (!v) snprintf(o, SYN_L, "%s=True", name);
        else if (q) snprintf(o, SYN_L, "%s=\"%s\"", name, v);
        else snprintf(o, SYN_L, "%s=%s", name, v);
        break;
    case CK_JL:
        if (!v) snprintf(o, SYN_L, "%s = true", name);
        else if (q) snprintf(o, SYN_L, "%s = \"%s\"", name, v);
        else snprintf(o, SYN_L, "%s = %s", name, v);
        break;
    case CK_JUMP:
        if (!v) snprintf(o, SYN_L, "set_attribute(model, \"%s\", true)", name);
        else if (q) snprintf(o, SYN_L, "set_attribute(model, \"%s\", \"%s\")", name, v);
        else snprintf(o, SYN_L, "set_attribute(model, \"%s\", %s)", name, v);
        break;
    case CK_ML:
        if (!v) snprintf(o, SYN_L, "opts.%s = true", name);
        else if (q) snprintf(o, SYN_L, "opts.%s = '%s'", name, v);
        else snprintf(o, SYN_L, "opts.%s = %s", name, v);
        break;
    default:
        if (v) snprintf(o, SYN_L, "-%s %s", name, v); else snprintf(o, SYN_L, "-%s", name);
    }
    return o;
}
static const char *syn_val(const char *v) {
    char *o = syn_buf();
    const char *cmd; const int k = caller_kind(&cmd);
    snprintf(o, SYN_L, k == CK_CLI ? "%s" : k == CK_ML ? "'%s'" : "\"%s\"", v);
    return o;
}
/* where the options are typed */
static const char *syn_head(void) {
    char *o = syn_buf();
    const char *cmd;
    switch (caller_kind(&cmd)) {
    case CK_C:     snprintf(o, SYN_L, ", in the option strings of the call"); break;
    case CK_PY:    snprintf(o, SYN_L, ", in brisk.%s(..., options={...})", cmd); break;
    case CK_CVXPY: snprintf(o, SYN_L, ", in prob.solve(solver=brisk.BRISK(), ...)"); break;
    case CK_JL:    snprintf(o, SYN_L, ", as keywords of Brisk.%s(...; ...)", cmd); break;
    case CK_JUMP:  snprintf(o, SYN_L, ", as attributes of the JuMP model"); break;
    case CK_ML:    snprintf(o, SYN_L, ", in the options of %s(..., opts)", cmd); break;
    default:       o[0] = 0;
    }
    return o;
}
/* 5.9: the last lines of a standard-precision log say how to get more accuracy; 5.10: in the
 * caller's syntax, and how to get a guaranteed bound (also at the end of a high-precision log).
 * Each line is an option as it is typed, in a column, and what it does. */
static void bound_hint(int hp) {
    const char *cmd; const int k = caller_kind(&cmd);
    const int model = k == CK_CVXPY || k == CK_JUMP;      /* the modelling layers name the sides of the model */
    const char *kc = syn_kv("certify", NULL, 0);
    const char *head = hp ? syn_head() : "";              /* (after the lines on accuracy the place has just been named) */
    if (g_has_bound && (g_has_certify || hp)) return;
    if (g_has_bound) {
        printf("For a rigorous check of the bound%s:\n  %s   the certificate is checked on the problem as read, with directed rounding\n", head, kc);
        return;
    }
    const char *kb = syn_kv("bound", model ? "dual" : "d", 1);
    int w = (int)strlen(kb);
    if (!hp && (int)strlen(kc) > w) w = (int)strlen(kc);
    printf("For a guaranteed bound on the optimal value, from a certified feasible point%s:\n", head);
    if (model) printf("  %-*s   from a feasible point of the dual (%s: from a feasible point of the model)", w, kb, syn_val("primal"));
    else printf("  %-*s   a lower bound b'y from a feasible y (%s: an upper bound %s)", w, kb, syn_val("p"),
                strstr(cmd, "sedumi") ? "c'x from a feasible x" : "<C,X> from a feasible X");
    if (hp) printf(";\n  %-*s   the certificate is built and checked rigorously in the precision of the solve\n", w, "");
    else printf(";\n  %-*s   with the high-precision solver the certificate is built and checked in its precision\n"
                "  %-*s   with a bound: the certificate is checked rigorously on the problem as read\n", w, "", w, kc);
}
static void accuracy_hint(int lp) {
    if (lp) { printf("For higher accuracy%s:\n  %s   the tolerance (default 1e-8)\n", syn_head(), syn_kv("tol", "<t>", 0)); return; }
    const char *ka = syn_kv("acc", "high", 1), *kp = syn_kv("prec", "dd", 1);
    int w = (int)strlen(kp);
    if (!g_acc_high && (int)strlen(ka) > w) w = (int)strlen(ka);
    printf("For higher accuracy%s:\n", syn_head());
    if (!g_acc_high) printf("  %-*s   tolerance 1e-10, still in double precision\n", w, ka);
    printf("  %-*s   the high-precision solver, about 31 digits (%s: about 63; or a number of digits);\n"
           "  %-*s   slower, the result is checked on the problem as read\n", w, kp, syn_val("qd"), w, "");
    bound_hint(0);
}
static int quiet_summary = 0;   /* BRISK_NOSUMMARY=1: no summary block (a log parsed by tools) */
/* 4.37 (B1): the re-solves of a run, for the note at the end of the log */
typedef struct { const char *name; double t, acc; int kept; } Attempt;
static Attempt g_att[32];
static int g_natt = 0, g_first_internal = 0;
static double g_first_t = 0, g_first_acc = 0;
static void att_log(const char *name, double t, const void *run);

/* 4.37: the second run replaces the first; in bound mode by the better certificate */
static int run_better(const Run *a, const Run *b) {
    /* only a valid certificate decides; otherwise the usual ranking */
    if (g_rank_par && g_rank_par->bound_side && ((a->cert.have && a->cert.valid) || (b->cert.have && b->cert.valid))) {
        if (bound_better(&a->cert, &b->cert)) return 1;
        if (bound_better(&b->cert, &a->cert)) return 0;
    }
    return run_rank(a) < run_rank(b);
}

static void att_log(const char *name, double t, const void *run) {
    if (g_natt >= 32) return;
    const Run *r = (const Run *)run;
    g_att[g_natt].name = name; g_att[g_natt].t = t; g_att[g_natt].acc = r ? r->acc : -1; g_att[g_natt].kept = 0;
    g_natt++;
}

/* 5.8: the summary at the end of the log: the problem as read, every presolve step applied,
 * the engine and the method, every re-solve, endgame, crossover and bound of the run,
 * the time, the six DIMACS errors, and the closing line "Solved to DIMACS error e" (an
 * infeasibility verdict closes with its certificate). The status flags (exit code, the
 * status: line, the library's status) are unchanged. */
static void fmt_count(char *b, size_t n, long v) {   /* 31,465 */
    char t[32]; snprintf(t, sizeof t, "%ld", v); const int L = (int)strlen(t); size_t w = 0;
    for (int i = 0; i < L && w + 2 < n; i++) { if (i && (L - i) % 3 == 0) b[w++] = ','; b[w++] = t[i]; }
    b[w] = 0;
}
static void summary_size(char *b, size_t n, const PSOrig *O) {
    int nsdp = 0, nlp = 0, maxn = 0;
    for (int k = 0; k < O->nblk; k++) { if (O->bs[k] > 0) { nsdp++; if (O->bs[k] > maxn) maxn = O->bs[k]; } else nlp += -O->bs[k]; }
    char cm[32], cl[32]; fmt_count(cm, sizeof cm, O->m); fmt_count(cl, sizeof cl, nlp);
    if (nsdp && nlp) snprintf(b, n, "m = %s, %d SDP block%s (largest %d), %s LP variable%s", cm, nsdp, nsdp > 1 ? "s" : "", maxn, cl, nlp > 1 ? "s" : "");
    else if (nsdp) snprintf(b, n, "m = %s, %d SDP block%s (largest %d)", cm, nsdp, nsdp > 1 ? "s" : "", maxn);
    else snprintf(b, n, "m = %s, %s LP variable%s", cm, cl, nlp > 1 ? "s" : "");
}
/* what a presolve step changed: " (m 31,465 -> 8,425; SDP blocks 1 -> 4, largest 406 -> 136)" */
static void summary_diff(char *d, size_t n, const PSOrig *a, const PSOrig *b) {
    int na = 0, nb = 0, la = 0, lb = 0, ma = 0, mb = 0;
    for (int k = 0; k < a->nblk; k++) { if (a->bs[k] > 0) { na++; if (a->bs[k] > ma) ma = a->bs[k]; } else la += -a->bs[k]; }
    for (int k = 0; k < b->nblk; k++) { if (b->bs[k] > 0) { nb++; if (b->bs[k] > mb) mb = b->bs[k]; } else lb += -b->bs[k]; }
    char s1[32], s2[32]; size_t w = 0; d[0] = 0;
    if (a->m != b->m) { fmt_count(s1, sizeof s1, a->m); fmt_count(s2, sizeof s2, b->m); w += snprintf(d + w, n - w, "%sm %s -> %s", w ? "; " : " (", s1, s2); }
    if (na != nb) w += snprintf(d + w, n - w, "%sSDP blocks %d -> %d", w ? "; " : " (", na, nb);
    if (ma != mb) w += snprintf(d + w, n - w, "%slargest block %d -> %d", w ? "; " : " (", ma, mb);
    if (la != lb) { fmt_count(s1, sizeof s1, la); fmt_count(s2, sizeof s2, lb); w += snprintf(d + w, n - w, "%sLP variables %s -> %s", w ? "; " : " (", s1, s2); }
    if (w) snprintf(d + w, n - w, ")");
}
static void print_summary(const Run *u, const Params *par, const PSOrig *O, int status, const double *err, int infeas,
                          const char *cause, double t_all, double tread, double tpre, int iters, const char *extra) {
    const Problem *PP = &u->P;
    char sz[200];
    printf("------------------------------------------------------------------------------\n");
    summary_size(sz, sizeof sz, O);
    printf("Summary\n  Problem as read:   %s\n", sz);
    /* the presolve chain: each stage's PSOrig is the problem before that step */
    const PSOrig *chain[8]; const char *what[8]; int nc = 0;
    if (u->Osign) { chain[nc] = u->Osign; what[nc++] = "sign-symmetry reduction"; }
    if (u->Osym) { chain[nc] = u->Osym; what[nc++] = "symmetry reduction (automorphisms)"; }
    if (u->Oalg) { chain[nc] = u->Oalg; what[nc++] = "block diagonalisation along the data's *-algebra"; }
    if (u->Odual) { chain[nc] = u->Odual; what[nc++] = "dual form solved"; }
    if (u->Ofile) { chain[nc] = u->Ofile; what[nc++] = "free variables eliminated by substitution"; }
    if (u->Otb) { chain[nc] = u->Otb; what[nc++] = "trace-bound row added"; }
    int npre = 0;
    const char *lab = "  Preprocessing:     ";
    for (int c = 0; c < nc; c++) {
        const PSOrig *before = chain[c], *after = c + 1 < nc ? chain[c + 1] : u->O;
        char d[400]; summary_diff(d, sizeof d, before, after);
        if (strstr(what[c], "trace")) printf("%s%s\n", npre ? "                     " : lab, what[c]);
        else printf("%s%s%s\n", npre ? "                     " : lab, what[c], d);
        npre++;
    }
    if (PP->ps && PP->ps->mom) { printf("%smoment-form conversion of the Schur complement\n", npre ? "                     " : lab); npre++; }
    if (PP->ps && PP->ps->chordal) { printf("%schordal decomposition: %d block(s) split into cliques\n", npre ? "                     " : lab, PP->ps->nch); npre++; }
    if (u->fr_removed > 0) { printf("%sfacial reduction: %d constraint(s) removed\n", npre ? "                     " : lab, u->fr_removed); npre++; }
    if (!npre) printf("%snone applied\n", lab);
    /* the engine */
    const Result *R = &u->R;
    char eng[300];
    if (par->fom > 0) snprintf(eng, sizeof eng, "first-order engine (ADMM / Douglas-Rachford, then augmented Lagrangian + semismooth Newton), %d iterations", iters);
    else if (par->lralm > 0) snprintf(eng, sizeof eng, "low-rank augmented Lagrangian method, %d iterations", iters);
    else if (par->mfipm == 2) snprintf(eng, sizeof eng, "hybrid: matrix-free interior-point iterations, then the standard method from their iterate; %d iterations", iters);
    else if (par->mfipm > 0) snprintf(eng, sizeof eng, "matrix-free interior-point method (NT, preconditioned CG on the Schur system), %d iterations", iters);
    else snprintf(eng, sizeof eng, "interior-point method: %s, %s direction, %d iterations", R->methods == 3 ? "infeasible start and self-dual embedding" : R->methods == 2 || R->retried ? "self-dual embedding" : "infeasible start",
                  R->direction == 1 ? "NT" : "HKM", iters);
    printf("  Solve:             %s\n", eng);
    for (int a = 0; a < g_natt; a++)
        printf("                     re-solve: %s, %.1f s%s\n", g_att[a].name, g_att[a].t, g_att[a].kept ? " (its result is returned)" : "");
    for (int i = 0; i < g_nnotes; i++) printf("                     %s\n", g_notes[i]);
    if (extra && extra[0]) printf("                     %s\n", extra);
    if (status == ST_TIME) printf("                     stopped by the time limit%s\n", brisk_stop_flag ? " (interrupted)" : "");
    else if (status == ST_MAXIT) printf("                     stopped at the iteration limit\n");
    if (par->bound_side && u->cert.have)
        printf("                     %s bound on the optimal value: %s (%s)\n", par->bound_side == 1 ? "primal (upper)" : "dual (lower)", u->cert.valid ? "certificate valid" : "approximate", u->cert.src);
    printf("  Time:              %.2f s (presolve %.2f, solve and re-solves %.2f; reading the file %.2f s more)\n", t_all, tpre, fmax(0.0, t_all - tpre), tread);
    { const int nt = brisk_threads_logged(); if (nt > 0) printf("  Threads:           %d\n", nt); }
    if (infeas) {
        printf("  Verdict:           %s\n", status == ST_PINFEAS ? "the problem is primal infeasible (no X >= 0 with A(X) = b)" : "the problem is dual infeasible (no y with C - A'y >= 0)");
        if (cause && cause[0]) printf("                     %s\n", cause);
        printf("%s\n", status == ST_PINFEAS ? "Primal infeasible" : "Dual infeasible");
        return;
    }
    printf("  DIMACS errors:     pinf %.1e  X>=0 %.1e  dinf %.1e  Z>=0 %.1e  gap %.1e  compl %.1e\n", err[1], err[2], err[3], err[4], err[5], err[6]);
    const double e = fmax(fmax(fabs(err[1]), fabs(err[2])), fmax(fmax(fabs(err[3]), fabs(err[4])), fmax(fabs(err[5]), fabs(err[6]))));
    printf("Solved to DIMACS error %.1e\n", e);
    accuracy_hint(0);
}


BriskResult *g_brisk_result = NULL;   /* 4.30: set by brisk_run */
static void diagnose_cause(const Run *u, const Params *par, const PSOrig *O, double **Xo, const double *yo,
                           const PSResult *pr, double acc, char *out, size_t len);
static int certify_given(const char *fname, const char *cx, const char *cy);

#ifndef BRISK_LIBRARY
/* 4.39: Ctrl-C (SIGINT) on the command line stops the solve as its time limit would: the current
 * point is measured, reported and written (-x / -y); a second Ctrl-C ends the process at once */
static void on_sigint(int sig) {
    (void)sig;
    if (brisk_stop_flag) { signal(SIGINT, SIG_DFL); raise(SIGINT); return; }
    brisk_stop_flag = 1;
    static const char msg[] = "\nbrisk: interrupted - stopping at the next check (Ctrl-C again to abort)\n";
    if (write(2, msg, sizeof msg - 1) < 0) { /* nothing */ }
}
#ifdef __linux__
/* 5.9 (a user's report): with thread binding (OMP_PROC_BIND, OMP_PLACES, GOMP_CPU_AFFINITY) the
 * OpenMP runtime pins the initial thread to its first place before main() runs, and execv
 * keeps that affinity mask: in the new process every place then lay inside it and all threads
 * ran on one core (theta6 with 6 bound threads: 9.7 s against 4.7 s). Before the re-exec the
 * mask is widened again to the union of the runtime's places, which is the set of processors
 * it started from (it builds its places inside the mask it finds, so a taskset or a cpuset
 * of the caller stays in force). Returns 0 when the threads are bound and the places cannot
 * be read or set: the re-exec is then skipped (brisk_threads_init still sets OpenBLAS to one
 * thread; only its idle workers remain). */
static int reexec_affinity(void) {
#if defined(_OPENMP) && _OPENMP >= 201511 && !defined(BRISK_OMP_H)
    if (omp_get_proc_bind() == omp_proc_bind_false) return 1;          /* nothing was pinned */
    const int np = omp_get_num_places();
    int maxid = -1, nids = 0;
    for (int i = 0; i < np; i++) { const int n = omp_get_place_num_procs(i); if (n > nids) nids = n; }
    if (np <= 0 || nids <= 0) return 0;
    int *ids = malloc(sizeof(int) * (size_t)nids);
    for (int i = 0; i < np; i++) { const int n = omp_get_place_num_procs(i); omp_get_place_proc_ids(i, ids); for (int j = 0; j < n; j++) if (ids[j] > maxid) maxid = ids[j]; }
    if (maxid < 0) { free(ids); return 0; }
    cpu_set_t *set = CPU_ALLOC(maxid + 1);
    const size_t sz = CPU_ALLOC_SIZE(maxid + 1);
    if (!set) { free(ids); return 0; }
    CPU_ZERO_S(sz, set);
    for (int i = 0; i < np; i++) { const int n = omp_get_place_num_procs(i); omp_get_place_proc_ids(i, ids); for (int j = 0; j < n; j++) if (ids[j] >= 0) CPU_SET_S(ids[j], sz, set); }
    const int rc = sched_setaffinity(0, sz, set);
    CPU_FREE(set); free(ids);
    return rc == 0;
#else
    /* a runtime without the places interface: no re-exec when binding is asked for */
    return !(getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY") || getenv("KMP_AFFINITY"));
#endif
}
#endif
int main(int argc, char **argv) {
#ifdef __linux__
    /* 4.24: OpenBLAS single-threaded from its start: its pthread workers, created at load,
     * spin next to BRISK's OpenMP threads (case14 0.02 -> 0.13 s at 2 threads even with
     * openblas_set_num_threads(1)); the environment is read at load time, hence the
     * re-exec. BRISK_BLASMT=1 keeps OpenBLAS threading. */
    if (!getenv("OPENBLAS_NUM_THREADS") && !getenv("BRISK_BLASMT") && !getenv("BRISK_NOREEXEC") && reexec_affinity()) {
        setenv("OPENBLAS_NUM_THREADS", "1", 1);
        setenv("BRISK_NOREEXEC", "1", 1);
        execv("/proc/self/exe", argv);
    }
#endif
    {   struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = on_sigint; sigemptyset(&sa.sa_mask); sigaction(SIGINT, &sa, NULL); }
    setvbuf(stdout, NULL, _IOLBF, 0);   /* 4.42: line-buffered also into a file or a pipe: the log of a run that is killed or watched with tail -f is complete */
    return brisk_main(argc, argv);
}
#endif

/* The memory this process may use: the physical RAM, or the lowest cgroup limit above the
 * process when lower (a container). 5.7: the cgroup of the process itself and its ancestors
 * are read (path from /proc/self/cgroup; before, only the root files were, and a limit set on
 * a nested group was missed: the test machine has 8.2 GB of RAM and a limit of 6.27 GB on the
 * process's own group). BRISK_MEMGB overrides. Computed once. */
static double cg_walk(const char *root, const char *path, const char *file, double lim) {
    char p[1024]; snprintf(p, sizeof p, "%s", path ? path : "");
    for (;;) {
        char fn[1280]; snprintf(fn, sizeof fn, "%s%s/%s", root, p, file);
        FILE *f = fopen(fn, "r");
        if (f) { char buf[64] = { 0 }; if (fgets(buf, sizeof buf, f)) { const double v = atof(buf); if (v > 1e8 && v < lim) lim = v; } fclose(f); }
        char *sl = strrchr(p, '/'); if (!sl) break; *sl = 0;
    }
    return lim;
}
double brisk_mem_limit(void) {
    static double cached = 0;
    if (cached > 0) return cached;
    { const char *e = getenv("BRISK_MEMGB"); if (e && atof(e) > 0) return cached = atof(e) * 1e9; }
    double ram = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE);
    ram = cg_walk("/sys/fs/cgroup", "", "memory.max", ram);
    ram = cg_walk("/sys/fs/cgroup/memory", "", "memory.limit_in_bytes", ram);
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f) {
        char ln[1024];
        while (fgets(ln, sizeof ln, f)) {
            ln[strcspn(ln, "\n")] = 0;
            char *c1 = strchr(ln, ':'); if (!c1) continue;
            char *c2 = strchr(c1 + 1, ':'); if (!c2) continue;
            *c2 = 0;
            const char *ctl = c1 + 1, *path = c2 + 1;
            if (path[0] != '/' || strstr(path, "..")) continue;
            if (ctl[0] == 0) ram = cg_walk("/sys/fs/cgroup", path, "memory.max", cg_walk("/sys/fs/cgroup/unified", path, "memory.max", ram));   /* v2 */
            else if (strstr(ctl, "memory")) ram = cg_walk("/sys/fs/cgroup/memory", path, "memory.limit_in_bytes", ram);                           /* v1 */
        }
        fclose(f);
    }
    /* 5.8 (a user's report): an address-space limit of the process (ulimit -v, RLIMIT_DATA)
     * counts too, less 0.3 GB for the mapped code, the stacks and the allocator's slack: under
     * `ulimit -v` every size decision was taken for the physical memory and the run died in an
     * allocation (or, once, took the machine down) */
    { struct rlimit rl;
      if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && (double)rl.rlim_cur - 3e8 > 1e8 && (double)rl.rlim_cur - 3e8 < ram) ram = (double)rl.rlim_cur - 3e8;
      if (getrlimit(RLIMIT_DATA, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && (double)rl.rlim_cur - 3e8 > 1e8 && (double)rl.rlim_cur - 3e8 < ram) ram = (double)rl.rlim_cur - 3e8; }
    return cached = ram;
}
/* 5.7: the fixed caps on index structures and work arrays (entries of a sparse factor, of an
 * adjacency structure, ...) were sized on a machine with 8 GB: they grow in proportion on a
 * larger one, and stay as they are on a smaller one (the allocation checks guard those). */
double brisk_mem_scale(void) { const double s = brisk_mem_limit() / 8.0e9; return s > 1.0 ? s : 1.0; }

extern int g_threads_user;
static int g_hp_inner = 0;   /* 5.0: this run is the double solve inside a high-precision one */
#include <unistd.h>
#include <fcntl.h>
static int hp_discard(const char *s, int is_err) { (void)s; (void)is_err; return 0; }
/* the output of the inner double solve is dropped: the print hook (library) and stdout itself */
static int hp_mute_begin(int (**hook)(const char *, int)) {
    fflush(stdout);
    *hook = brisk_print_hook;
    if (brisk_print_hook) brisk_print_hook = hp_discard;
    const int fd = dup(1), nul = open("/dev/null", O_WRONLY);
    if (nul >= 0) { dup2(nul, 1); close(nul); }
    return fd;
}
static void hp_mute_end(int fd, int (*hook)(const char *, int)) {
    fflush(stdout);
    if (fd >= 0) { dup2(fd, 1); close(fd); }
    brisk_print_hook = hook;
}

/* ---- problems in SeDuMi format (sedumi.c) ------------------------------------------------ */
#include "sedumi.h"
#include "lpio.h"
#include "lpsolve.h"
/* the semidefinite solver on the SDPA form of a SeDuMi problem, by a direct call of the command
 * line (argv: the option words) */
static int sedumi_inner_sdp(const BriskData *d, int argc, char **argv, BriskResult *res) {
    char **av = malloc(sizeof(char *) * ((size_t)argc + 6));
    int ac = 0, hasrx = 0;
    av[ac++] = "brisk"; av[ac++] = (char *)g_brisk_data_name;
    for (int i = 0; i < argc; i++) { av[ac++] = argv[i]; if (argv[i] && !strcmp(argv[i], "-returnx")) hasrx = 1; }
    if (!hasrx) { av[ac++] = "-returnx"; av[ac++] = "1"; }
    av[ac] = NULL;
    memset(res, 0, sizeof(*res)); res->status = -1;
    const BriskData *sd = g_brisk_data; BriskResult *sr = g_brisk_result;
    g_brisk_data = d; g_brisk_result = res;
    const int rc = brisk_main(ac, av);
    g_brisk_data = sd; g_brisk_result = sr;
    free(av);
    return rc;
}
#ifdef BRISK_LIBRARY
static int sedumi_lib_sdp(const BriskData *d, int argc, char **argv, BriskResult *res) {
    char **av = malloc(sizeof(char *) * ((size_t)argc + 6));
    int ac = 0, hasrx = 0;
    av[ac++] = "brisk";
    for (int i = 0; i < argc; i++) { av[ac++] = argv[i]; if (argv[i] && !strcmp(argv[i], "-returnx")) hasrx = 1; }
    if (!hasrx) { av[ac++] = "-returnx"; av[ac++] = "1"; }
    av[ac] = NULL;
    const int rc = brisk_run_data(d, ac, av, res);
    free(av);
    return rc;
}
static void cone_summary(const SedumiProb *P, const SedumiRes *R);
int brisk_run_sedumi(const SedumiProb *P, int argc, char **argv, SedumiRes *R) {
    char **av = caller_enter(&argc, argv, "c");
    if (av) argv = av;
    if (g_caller_depth == 1) notes_clear();
    const int rc = sedumi_solve(P, argc, argv, R, sedumi_lib_sdp);
    /* 5.10: the cone solver's summary also for a problem given in memory (the command line printed
     * it for a file only), so that the interfaces' logs end like the semidefinite solver's */
    if (R->status >= 0 && R->cone && g_caller_depth == 1 && !getenv("BRISK_NOSUMMARY")) {
        int quiet = 0;
        for (int i = 0; i < argc; i++) if (argv[i] && (!strcmp(argv[i], "-q") || !strcmp(argv[i], "-silent"))) quiet = 1;
        if (!quiet) { advice_scan(argc, argv, 0); cone_summary(P, R); }
    }
    caller_leave(av);
    return rc;
}
#else
int brisk_run_sedumi(const SedumiProb *P, int argc, char **argv, SedumiRes *R) { return sedumi_solve(P, argc, argv, R, sedumi_inner_sdp); }
#endif
static void sedumi_write_vec(const char *fname, const double *v, int n) {
    FILE *f = fopen(fname, "w");
    if (!f) { fprintf(stderr, "brisk: cannot write %s\n", fname); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%.17g\n", v[i]);
    fclose(f);
}
/* 5.8: the summary of a cone solve (SeDuMi form without PSD blocks: the cone solver) */
static void cone_summary(const SedumiProb *P, const SedumiRes *R) {
    char cm[32], cn[32]; fmt_count(cm, sizeof cm, P->m); fmt_count(cn, sizeof cn, P->n);
    int qmax = 0; for (int k = 0; k < P->nq; k++) if (P->q[k] > qmax) qmax = P->q[k]; for (int k = 0; k < P->nr; k++) if (P->r[k] > qmax) qmax = P->r[k];
    printf("------------------------------------------------------------------------------\n");
    printf("Summary\n  Problem as read:   m = %s rows, n = %s variables: %d free, %d nonnegative, %d second-order cones, %d rotated (largest %d)\n", cm, cn, P->nf, P->nl, P->nq, P->nr, qmax);
    int np = 0; for (int i = 0; i < g_nnotes; i++) if (!strncmp(g_notes[i], "presolve: ", 10)) { printf("  %s%s\n", np ? "                   " : "Preprocessing:     ", g_notes[i] + 10); np++; }
    if (!np) printf("  Preprocessing:     none applied\n");
    printf("  Solve:             cone solver (homogeneous self-dual model, Nesterov-Todd scaling, L D L' of the reduced system), %d iterations\n", R->iters);
    for (int i = 0; i < g_nnotes; i++) if (strncmp(g_notes[i], "presolve: ", 10)) printf("                     %s\n", g_notes[i]);
    if (R->status == 6) printf("                     stopped by the time limit\n"); else if (R->status == 3) printf("                     stopped at the iteration limit\n");
    printf("  Time:              %.2f s\n  Threads:           1 (the cone solver is sequential)\n", R->time);
    if (R->status == 1 || R->status == 2) { printf("  Verdict:           the problem is %s infeasible (the returned %s is a certificate)\n%s\n", R->status == 1 ? "primal" : "dual", R->status == 1 ? "y" : "x", R->status == 1 ? "Primal infeasible" : "Dual infeasible"); return; }
    printf("  DIMACS errors:     pinf %.1e  x in K %.1e  dinf %.1e  z in K* %.1e  gap %.1e  compl %.1e\n", R->err[0], R->err[1], R->err[2], R->err[3], R->err[4], R->err[5]);
    double e = 0; for (int i = 0; i < 6; i++) e = fmax(e, fabs(R->err[i]));
    printf("Solved to DIMACS error %.1e\n", e);
    accuracy_hint(0);
}
/* the command line on a MAT-file with A (or At), b, c, K: argv[ifile] is the file */
static int brisk_main_sedumi(int argc, char **argv, int ifile) {
    SedumiProb P; char msg[400];
    if (sedumi_read_mat(argv[ifile], &P, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; }
    char **av = malloc(sizeof(char *) * ((size_t)argc + 1));
    int ac = 0, quiet = 0;
    const char *xf = NULL, *yf = NULL, *zf = NULL;
    for (int i = 1; i < argc; i++) {
        if (i == ifile) continue;
        if (i + 1 < argc && (!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z"))) {
            if (argv[i][1] == 'x') xf = argv[i + 1]; else if (argv[i][1] == 'y') yf = argv[i + 1]; else zf = argv[i + 1];
            i++; continue;
        }
        if (!strcmp(argv[i], "-q")) quiet = 1;
        av[ac++] = argv[i];
    }
    if (!quiet) {
        long sd = 0; int smax = 0, qmax = 0;
        for (int k = 0; k < P.ns; k++) { sd += (long)P.s[k] * P.s[k]; if (P.s[k] > smax) smax = P.s[k]; }
        for (int k = 0; k < P.nq; k++) if (P.q[k] > qmax) qmax = P.q[k];
        for (int k = 0; k < P.nr; k++) if (P.r[k] > qmax) qmax = P.r[k];
        printf("SeDuMi problem %s: m = %d, n = %d; K.f = %d, K.l = %d, K.q: %d, K.r: %d (largest %d), K.s: %d (largest %d)\n",
               argv[ifile], P.m, P.n, P.nf, P.nl, P.nq, P.nr, qmax, P.ns, smax);
    }
    SedumiRes R;
    const int rc = sedumi_solve(&P, ac, av, &R, sedumi_inner_sdp);
    if (R.status >= 0) {
        if (!quiet && P.ns > 0) printf("SeDuMi form: c'x = %+.12e   b'y = %+.12e\n", R.pobj, R.dobj);
        if (!quiet && R.cone && !quiet_summary) cone_summary(&P, &R);
        if (xf && R.x && (P.ns == 0 || R.have_x)) sedumi_write_vec(xf, R.x, P.n);
        if (yf && R.y) sedumi_write_vec(yf, R.y, P.m);
        if (zf && R.z) sedumi_write_vec(zf, R.z, P.n);
        if (g_brisk_result) {       /* a library call on the file: status, values and y */
            BriskResult *res = g_brisk_result;
            res->status = R.status; snprintf(res->status_str, sizeof res->status_str, "%s", R.status_str);
            res->exit_code = rc; res->iters = R.iters; res->m = P.m; res->pobj = R.pobj; res->dobj = R.dobj; res->time = R.time;
            for (int i = 0; i < 6; i++) res->err[i + 1] = R.err[i];
            res->y = R.y; R.y = NULL;
            res->sn = P.n; res->have_x = P.ns == 0 || R.have_x; res->sx = R.x; R.x = NULL; res->sz = R.z; R.z = NULL;
        }
    }
    sedumi_result_free(&R); sedumi_free(&P); free(av);
    return rc;
}
/* 5.8: the summary of a linear program (MPS): the presolve, the LP interior-point method or the cone solver */
static void lp_summary(const LpProb *L, const SedumiProb *P, const SedumiRes *R, int lpipm, double eb, double er, double tpre, double ttot) {
    char a[32], b[32], c[32], d[32]; fmt_count(a, sizeof a, L->m); fmt_count(b, sizeof b, L->n); fmt_count(c, sizeof c, P->m); fmt_count(d, sizeof d, P->n);
    printf("------------------------------------------------------------------------------\n");
    printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a, b, L->maxim ? "maximized" : "minimized");
    printf("  Preprocessing:     presolve and standard form: %s rows, %s columns (%.2f s)\n", c, d, tpre);
    int np = 0; for (int i = 0; i < g_nnotes; i++) if (!strncmp(g_notes[i], "presolve: ", 10)) { printf("                     %s\n", g_notes[i] + 10); np++; }
    printf("  Solve:             %s, %d iterations\n", lpipm ? "LP interior-point method (normal equations, Mehrotra predictor-corrector)" : "cone solver (homogeneous self-dual model)", R->iters);
    for (int i = 0; i < g_nnotes; i++) if (strncmp(g_notes[i], "presolve: ", 10)) printf("                     %s\n", g_notes[i]);
    if (R->status == 6) printf("                     stopped by the time limit\n"); else if (R->status == 3) printf("                     stopped at the iteration limit\n");
    printf("  Time:              %.2f s\n  Threads:           1 (the solver for linear programs is sequential)\n", ttot);
    if (R->status == 1 || R->status == 2) { printf("  Verdict:           the problem is %s\n%s\n", R->status == 1 ? "primal infeasible" : "dual infeasible (unbounded)", R->status == 1 ? "Primal infeasible" : "Dual infeasible"); return; }
    printf("  Errors:            on the problem as read: bound violation %.1e, row violation %.1e (relative); of the solve: pinf %.1e  dinf %.1e  gap %.1e\n", eb, er, R->err[0], R->err[2], R->err[4]);
    printf("Solved to relative error %.1e\n", fmax(fmax(eb, er), fmax(fmax(fabs(R->err[0]), fabs(R->err[2])), fabs(R->err[4]))));
    accuracy_hint(1);
}
/* the command line on an MPS file (a linear program): presolve, standard form, the cone solver */
static int brisk_main_lp(int argc, char **argv, int ifile) {
    LpProb L; char msg[400];
    const double t0 = wtime();
    if (lp_read_mps(argv[ifile], &L, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; }
    char **av = malloc(sizeof(char *) * ((size_t)argc + 1));
    int ac = 0, quiet = 0, pre = 2, lpm = 1, lpcorr = -1;
    const char *xf = NULL;
    for (int i = 1; i < argc; i++) {
        if (i == ifile) continue;
        if (i + 1 < argc && !strcmp(argv[i], "-x")) { xf = argv[i + 1]; i++; continue; }
        if (i + 1 < argc && !strcmp(argv[i], "-lppresolve")) { pre = atoi(argv[i + 1]); i++; continue; }
        if (i + 1 < argc && !strcmp(argv[i], "-lpmethod")) { lpm = atoi(argv[i + 1]); i++; continue; }
        if (i + 1 < argc && !strcmp(argv[i], "-lpcorr")) { lpcorr = atoi(argv[i + 1]); i++; continue; }
        if (!strcmp(argv[i], "-q")) quiet = 1;
        av[ac++] = argv[i];
    }
    if (!quiet) printf("LP %s (%s): %d rows, %d columns, %d nonzeros, read %.2fs\n", argv[ifile], L.name[0] ? L.name : "no name", L.m, L.n, L.Ap[L.n], wtime() - t0);
    SedumiProb P; LpMap M;
    const double tp0 = wtime();
    double *ub = NULL;
    if (lpm) lp_to_std(&L, pre, quiet ? 0 : 1, &P, &M, &ub); else lp_to_sedumi(&L, pre, quiet ? 0 : 1, &P, &M);
    const double tpre = wtime() - tp0;
    int rc = 0, lp_done = 0;
    double *x = calloc((size_t)L.n + 1, sizeof(double));
    if (M.status) {
        if (!quiet) printf("status: %s   iterations: 0   time: %.3fs\n", M.status == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE", wtime() - t0);
        rc = M.status == 1 ? 11 : 12;
        if (g_brisk_result) { g_brisk_result->status = M.status; snprintf(g_brisk_result->status_str, sizeof g_brisk_result->status_str, "%s", M.status == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE"); g_brisk_result->exit_code = rc; }
    } else {
        SedumiRes R;
        if (P.n == 0) {          /* everything was fixed by the presolve */
            memset(&R, 0, sizeof R); R.status = 0; snprintf(R.status_str, sizeof R.status_str, "OPTIMAL");
            R.x = calloc(1, sizeof(double)); R.y = calloc((size_t)P.m + 1, sizeof(double)); R.z = calloc(1, sizeof(double));
            if (!quiet) printf("status: OPTIMAL (solved by the presolve)   iterations: 0\n");
        } else {
            int done = 0;
            if (lpm) {
                /* the dedicated interior-point method on the bounded form; the cone solver when it gives up */
                LpIpmOpts o; LpIpmInfo inf; memset(&o, 0, sizeof o);
                o.tol = 1e-8; o.kcorr = lpcorr; o.verbose = quiet ? 0 : 1;
                for (int i = 0; i < ac; i++) {
                    if (i + 1 < ac && !strcmp(av[i], "-tol")) o.tol = atof(av[i + 1]);
                    if (i + 1 < ac && !strcmp(av[i], "-maxit")) o.maxit = atoi(av[i + 1]);
                    if (i + 1 < ac && !strcmp(av[i], "-timelimit")) o.timelimit = atof(av[i + 1]);
                    if (!strcmp(av[i], "-v") || !strcmp(av[i], "-vv")) o.verbose = 2;
                }
                memset(&R, 0, sizeof R);
                R.x = calloc((size_t)P.n + 1, sizeof(double)); R.y = calloc((size_t)P.m + 1, sizeof(double)); R.z = calloc((size_t)P.n + 1, sizeof(double));
                const int st = lpipm_solve(P.m, P.n, P.nf, P.Ap, P.Ai, P.Ax, P.b, P.c, ub, &o, R.x, R.y, R.z, &inf);
                if (st == 0 || st == 3 || st == 6) {
                    done = 1; lp_done = 1; R.status = st; R.iters = inf.iters; R.pobj = inf.pobj; R.dobj = inf.dobj; R.have_x = 1; R.m = P.m; R.n = P.n;
                    R.err[0] = inf.pinf; R.err[2] = inf.dinf; R.err[4] = inf.gap;
                    const double e = fmax(fmax(inf.pinf, inf.dinf), inf.gap);
                    if (st == 0) snprintf(R.status_str, sizeof R.status_str, "OPTIMAL (max rel error %.1e)", e);
                    else snprintf(R.status_str, sizeof R.status_str, "%s", st == 3 ? "ITERATION LIMIT" : "TIME LIMIT");
                    rc = st == 0 ? 0 : st == 3 ? 3 : 6;
                    if (!quiet) {
                        printf("status: %s   iterations: %d   time: %.3fs\n", R.status_str, inf.iters, inf.time);
                        printf("  primal obj c'x = %.12e   dual obj b'y - u'w = %.12e\n", inf.pobj, inf.dobj);
                        printf("  errors: pinf %.1e  dinf %.1e  gap %.1e;  %ld factorizations %.3fs, %ld solves %.3fs (%ld CG steps), %d correctors\n",
                               inf.pinf, inf.dinf, inf.gap, inf.nfact, inf.t_fact, inf.nsolve, inf.t_solve, inf.ncg, inf.ncorr);
                    }
                } else {
                    if (!quiet) printf("  the LP interior-point method stopped after %d iterations (%s): solving with the cone solver (homogeneous model)\n", inf.iters, inf.why);
                    free(R.x); free(R.y); free(R.z); memset(&R, 0, sizeof R);
                    sedumi_free(&P); lp_map_free(&M); free(ub); ub = NULL;
                    lp_to_sedumi(&L, pre, 0, &P, &M);
                }
            }
            if (!done) rc = sedumi_solve(&P, ac, av, &R, sedumi_inner_sdp);
        }
        if (R.status >= 0 && R.x) {
            const double obj = lp_recover(&L, &M, R.x, x);
            if (R.status == 0 || R.status == 5 || R.status == 3 || R.status == 4 || R.status == 6) {
                double eb, er; lp_errors(&L, x, &eb, &er);
                if (!quiet) {
                    printf("objective value (as in the file, %s): %.12e\n", L.maxim ? "maximized" : "minimized", obj);
                    printf("  on the problem as read: bound violation %.1e, row violation %.1e (relative); presolve %.3fs, total %.3fs\n", eb, er, tpre, wtime() - t0);
                    if (!quiet_summary) lp_summary(&L, &P, &R, lp_done, eb, er, tpre, wtime() - t0);
                }
            }
            if (xf) sedumi_write_vec(xf, x, L.n);
            if (g_brisk_result) {
                BriskResult *res = g_brisk_result;
                res->status = R.status; snprintf(res->status_str, sizeof res->status_str, "%s", R.status_str);
                res->exit_code = rc; res->iters = R.iters; res->m = L.m; res->pobj = obj; res->dobj = obj; res->time = wtime() - t0;
                res->sn = L.n; res->have_x = 1; res->sx = x; x = NULL;
            }
        }
        sedumi_result_free(&R);
    }
    free(x); free(ub); sedumi_free(&P); lp_map_free(&M); lp_free(&L); free(av);
    return rc;
}
/* the command line on a CBF file (CBLIB): linear and second-order cones */
static int brisk_main_cbf(int argc, char **argv, int ifile) {
    SedumiProb P; char msg[400]; double c0 = 0; int maxim = 0, nint = 0;
    const double t0 = wtime();
    if (cbf_read(argv[ifile], &P, &c0, &maxim, &nint, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; }
    if (maxim) for (int j = 0; j < P.n; j++) P.c[j] = -P.c[j];
    char **av = malloc(sizeof(char *) * ((size_t)argc + 1));
    int ac = 0, quiet = 0;
    for (int i = 1; i < argc; i++) { if (i == ifile) continue; if (!strcmp(argv[i], "-q")) quiet = 1; av[ac++] = argv[i]; }
    if (!quiet) {
        int qmax = 0; for (int k = 0; k < P.nq; k++) if (P.q[k] > qmax) qmax = P.q[k];
        for (int k = 0; k < P.nr; k++) if (P.r[k] > qmax) qmax = P.r[k];
        printf("CBF problem %s: m = %d, n = %d; free %d, nonnegative %d, second-order cones %d, rotated %d (largest %d), read %.2fs\n", argv[ifile], P.m, P.n, P.nf, P.nl, P.nq, P.nr, qmax, wtime() - t0);
        if (nint) printf("  %d integer variables: relaxed (the continuous relaxation is solved)\n", nint);
    }
    SedumiRes R;
    const int rc = sedumi_solve(&P, ac, av, &R, sedumi_inner_sdp);
    if (R.status >= 0) {
        const double obj = (maxim ? -R.pobj : R.pobj) + c0;
        if (!quiet && R.status != 1 && R.status != 2) printf("objective value (as in the file, %s): %.12e\n", maxim ? "maximized" : "minimized", obj);
        if (!quiet && R.cone && !quiet_summary) cone_summary(&P, &R);
        if (g_brisk_result) {
            BriskResult *res = g_brisk_result;
            res->status = R.status; snprintf(res->status_str, sizeof res->status_str, "%s", R.status_str);
            res->exit_code = rc; res->iters = R.iters; res->m = P.m; res->pobj = obj; res->dobj = (maxim ? -R.dobj : R.dobj) + c0; res->time = R.time;
            for (int i = 0; i < 6; i++) res->err[i + 1] = R.err[i];
            res->y = R.y; R.y = NULL; res->sn = P.n; res->have_x = 1; res->sx = R.x; R.x = NULL; res->sz = R.z; R.z = NULL;
        }
    }
    sedumi_result_free(&R); sedumi_free(&P); free(av);
    return rc;
}
int brisk_main(int argc, char **argv) {
    brisk_threads_init();
    brisk_log_threads_reset();
    if (!g_hp_inner) notes_clear();                                                /* 5.8: the summary's notes */
    quiet_summary = getenv("BRISK_NOSUMMARY") != NULL || g_hp_inner;
    if (!g_hp_inner) advice_scan(argc, argv, 1);
    { extern void (*socp_note)(const char *, ...); socp_note = brisk_note; }
    g_threads_user = getenv("OMP_NUM_THREADS") != NULL;   /* 4.42: set also by -threads: the busy-machine rule then stays out */
    if (argc < 2) { usage(); return 1; }
    for (int i = 1; i < argc; i++) {       /* a MAT-file: a problem in SeDuMi format */
        const size_t l = argv[i] ? strlen(argv[i]) : 0;
        if (l > 4 && argv[i][0] != '-' && !strcmp(argv[i] + l - 4, ".mat")
            && !(i > 1 && (!strcmp(argv[i - 1], "-x") || !strcmp(argv[i - 1], "-y") || !strcmp(argv[i - 1], "-z"))) && sedumi_is_mat(argv[i]))
            return brisk_main_sedumi(argc, argv, i);
        if (l > 4 && argv[i][0] != '-' && lp_is_mps(argv[i]) && !(i > 1 && !strcmp(argv[i - 1], "-x")))
            return brisk_main_lp(argc, argv, i);
        if (l > 4 && argv[i][0] != '-' && cbf_is_file(argv[i]))
            return brisk_main_cbf(argc, argv, i);
    }
    Params par;
    params_default(&par);
    const char *fname = NULL, *yfile = NULL, *xfile = NULL, *zfile = NULL;
    const char *cert_x = NULL, *cert_y = NULL;   /* 4.39: certify a given certificate, no solve */
    int do_certify = 0;                          /* 4.39: -certify (off by default) */
    int force_route = -1, do_fr = 1, fr_retry = 1, acc_set = -1;
    /* -acc first, so that explicit -tol / -retryacc override it wherever they appear */
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "-acc"))
            params_accuracy(&par, !strcmp(argv[i + 1], "low") ? 0 : !strcmp(argv[i + 1], "high") ? 2 : 1);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        {   /* 4.41: a flag may carry an explicit 0/1 (true/false) - the interfaces pass
             * {"nohsd": 1} or opts.nohsd = 1 as "-nohsd 1"; 0 drops the flag */
            static const char *const flags[] = { "-q", "-v", "-vv", "-nofr", "-fr", "-nofrretry", "-nosplit", "-split", "-nopolish",
                                                 "-nopolishx", "-nopolishr", "-nodd", "-hsd", "-nohsd", "-dual", "-dualold", "-nosym", "-certify", NULL };
            int isflag = 0;
            for (int f = 0; flags[f]; f++) if (!strcmp(a, flags[f])) isflag = 1;
            if (isflag && i + 1 < argc) {
                const char *nx = argv[i + 1];
                const int one = !strcmp(nx, "1") || !strcmp(nx, "true") || !strcmp(nx, "True");
                const int zero = !strcmp(nx, "0") || !strcmp(nx, "false") || !strcmp(nx, "False");
                if (one || zero) { i++; if (zero) continue; }
            }
        }
#define OPT_D(name, field) else if (!strcmp(a, name)) { i = need(i, argc, a); par.field = num(argv[i], a); }
#define OPT_I(name, field) else if (!strcmp(a, name)) { i = need(i, argc, a); par.field = (int)num(argv[i], a); }
        if (0) {}
        OPT_D("-tol", tol) OPT_I("-maxit", maxit) OPT_D("-timelimit", timelimit)
        OPT_I("-init", init) OPT_I("-sigma", sigma_rule) OPT_I("-corr", max_correctors)
        OPT_I("-mixed", mixed) OPT_D("-mixedfrac", mixed_frac) OPT_I("-lowrank", lowrank) OPT_I("-dict", dict) OPT_D("-gmax", gamma_max)
        OPT_I("-ntls", nt_ls) OPT_D("-ntlsgap", nt_ls_gap) OPT_I("-stallwin", stall_win)
        OPT_I("-sparse", sparse_schur) OPT_D("-balance", balance) OPT_D("-pivtol", pivtol)
        OPT_D("-feasgrow", feas_grow) OPT_D("-sigmamin", sigma_min) OPT_D("-dualrho", dual_rho)
        OPT_I("-dualcert", dual_cert) OPT_D("-dualmargin", dual_margin) OPT_I("-dualgiveup", dual_giveup) OPT_D("-dualtau", dual_tau) OPT_D("-dualtaulong", dual_tau_long) OPT_D("-dualgamma", dual_gamma) OPT_I("-dualmurule", dual_murule) OPT_I("-dualls", dual_ls) OPT_I("-dualcorr", dual_corr) OPT_I("-dualmured", dual_mured) OPT_D("-dualtaunb", dual_tau_nb) OPT_D("-dualtauub", dual_tau_ub) OPT_I("-dualsparse", dual_sparse_n) OPT_I("-dualcorrauto", dual_corr_auto) OPT_I("-dualgstart", dual_gstart) OPT_D("-dualmudrop", dual_mudrop) OPT_I("-dualverifyn", dual_verify_n)
        OPT_I("-chordal", chordal) OPT_I("-cliquemax", chordal_maxclique) OPT_I("-chordalsup", chordal_sup) OPT_D("-chordalmerge", chordal_merge) OPT_D("-chordaldens", chordal_density) OPT_I("-chordalform", chordal_form) OPT_I("-freeelim", free_elim) OPT_D("-freefill", free_elim_fill) OPT_I("-chordalmin", chordal_minn)
        OPT_I("-hsdsig", hsd_sig) OPT_D("-hsdbeta", hsd_beta) OPT_I("-hsddir", hsd_dir) OPT_I("-hsdcorr", hsd_corr) OPT_I("-hsdrefine", hsd_refine) OPT_I("-hsdfree", hsd_free) OPT_I("-hsdstall", hsd_stall) OPT_D("-hsdpshrink", hsd_pshrink) OPT_I("-hsdreghint", hsd_reghint) OPT_I("-hsdqscale", hsd_qscale) OPT_I("-hsdbatch", hsd_batch) OPT_I("-hsdctrace", hsd_ctrace) OPT_I("-hsdctarget", hsd_ctarget) OPT_D("-hsdcbmin", hsd_cbmin) OPT_D("-hsdcbmax", hsd_cbmax) OPT_D("-hsdcacc", hsd_cacc) OPT_D("-hsdsoltol", hsd_soltol) OPT_D("-hsdcfrac", hsd_cfrac) OPT_D("-hsdfomega", hsd_fomega) OPT_D("-hsdpivtol", hsd_pivtol) OPT_D("-hsdcucap", hsd_cucap) OPT_D("-hsdcgain", hsd_cgain) OPT_I("-ddit", dd_iters) OPT_I("-ddmaxm", dd_maxm)
        OPT_D("-ddbudget", dd_budget) OPT_D("-ddtime", dd_time) OPT_D("-ddfactor", dd_factor)
        OPT_D("-ddminwork", dd_minwork) OPT_D("-regill", reg_ill) OPT_D("-cblas", c_blas)
        OPT_I("-hsdhoc", hsd_hoc) OPT_D("-hsdnb", hsd_nb) OPT_I("-lanczos", lanczos_k) OPT_I("-ntfast", nt_fast) OPT_I("-hsdpat", hsd_pat) OPT_I("-parblocks", par_blocks) OPT_I("-knownfeas", known_feasible) OPT_D("-retryacc", retry_acc) OPT_D("-warm", warm_lam) OPT_D("-tracebound", tracebound) OPT_I("-dualize", dualize) OPT_I("-fom", fom) OPT_I("-fomhalpern", fom_halpern) OPT_I("-fommaxit", fom_maxit) OPT_D("-fomtol", fom_tol) OPT_D("-fomsigma", fom_sigma) OPT_D("-fomsigma0", fom_sigma0) OPT_I("-fomaadr", fom_aadr) OPT_D("-fomsingle", fom_single) OPT_I("-fomssnstall", fom_ssn_stall) OPT_I("-fomsigrule", fom_sigrule) OPT_I("-fomsigint", fom_sigint) OPT_I("-fomaa", fom_aa) OPT_I("-fombm", fom_bm) OPT_I("-fombmrank0", fom_bm_rank0) OPT_I("-fombmouter", fom_bm_outer) OPT_I("-fombminner", fom_bm_inner) OPT_D("-fombmrho", fom_bm_rho) OPT_D("-fombmgtol", fom_bm_gtol) OPT_D("-fombmnegtol", fom_bm_negtol) OPT_I("-fomssn", fom_ssn) OPT_I("-fomssnafter", fom_ssn_after) OPT_D("-fomssnres", fom_ssn_res) OPT_D("-fomssnrho", fom_ssn_rho) OPT_I("-fomssnprec", fom_ssn_prec) OPT_D("-fomssnsig0", fom_ssn_sig0) OPT_D("-fomssneta", fom_ssn_eta) OPT_I("-fomssnwarm", fom_ssn_warm) OPT_I("-fomssnouter", fom_ssn_outer) OPT_I("-fomssnnewton", fom_ssn_newton) OPT_I("-fomssncg", fom_ssn_cg) OPT_D("-fomaasafe", fom_aasafe) OPT_D("-fomsigmax", fom_sigmax) OPT_I("-hsdfirst", hsd_first) OPT_D("-hsdfirstc", hsd_first_c) OPT_D("-hsdfirstasm", hsd_first_asm) OPT_I("-stdslow", std_slow) OPT_I("-crossover", crossover) OPT_D("-frgain", fr_gain)
        else if (!strcmp(a, "-acc")) {
            /* accuracy level: the tolerance and the thresholds derived from it */
            i = need(i, argc, a);
            if (!strcmp(argv[i], "low")) acc_set = 0;
            else if (!strcmp(argv[i], "default")) acc_set = 1;
            else if (!strcmp(argv[i], "high")) acc_set = 2;
            else { fprintf(stderr, "brisk: -acc must be low, default or high\n"); return 1; }
            (void)acc_set;
        }
        else if (!strcmp(a, "-q")) par.verbose = 0;
        else if (!strcmp(a, "-v")) par.verbose = 2;
        else if (!strcmp(a, "-vv")) par.verbose = 3;
        else if (!strcmp(a, "-nofr")) do_fr = 0;
        else if (!strcmp(a, "-fr")) do_fr = 2;          /* facial reduction whatever its depth */
        else if (!strcmp(a, "-nofrretry")) fr_retry = 0;
        else if (!strcmp(a, "-dir")) {
            i = need(i, argc, a);
            if (!strcmp(argv[i], "nt")) par.direction = 1;
            else if (!strcmp(argv[i], "hkm")) par.direction = 0;
            else if (!strcmp(argv[i], "auto")) par.direction = -1;
            else { fprintf(stderr, "brisk: -dir must be auto, hkm or nt\n"); return 1; }
        }
        else if (!strcmp(a, "-nosplit")) par.nt_split = 0;
        else if (!strcmp(a, "-split")) par.nt_split = 1;
        else if (!strcmp(a, "-nopolish")) par.polish = 0;
        else if (!strcmp(a, "-nopolishx")) par.polish_x = 0;
        else if (!strcmp(a, "-nopolishr")) par.polish_r = 0;
        else if (!strcmp(a, "-nodd")) par.dd_end = 0;
        else if (!strcmp(a, "-hsd")) par.hsd = 1;
        else if (!strcmp(a, "-nohsd")) par.hsd = 0;
        else if (!strcmp(a, "-dual")) par.dual = 1;
        else if (!strcmp(a, "-dualold")) par.dual = 2;
        else if (!strcmp(a, "-fomstart-x")) { i = need(i, argc, a); par.fom_start_x = argv[i]; }
        else if (!strcmp(a, "-conesolver")) { i = need(i, argc, a); }   /* problems in SeDuMi format (sedumi.c); no effect on an SDPA problem */
        else if (!strcmp(a, "-fomstart-y")) { i = need(i, argc, a); par.fom_start_y = argv[i]; }
        else if (!strcmp(a, "-boundanchor")) { i = need(i, argc, a); par.bound_anchor = argv[i]; }
        else if (!strcmp(a, "-prec")) {       /* 5.0: high precision */
            i = need(i, argc, a);
            const char *s = argv[i];
            if (!strcmp(s, "dd")) par.hp_kind = 1;
            else if (!strcmp(s, "qd")) par.hp_kind = 2;
            else if (!strcmp(s, "double") || !strcmp(s, "0")) par.hp_kind = 0;
            else { const int nd = (int)num(s, a); if (nd < 1) { fprintf(stderr, "brisk: -prec must be dd, qd or a number of digits\n"); return 1; } par.hp_kind = nd <= 31 ? 1 : nd <= 63 ? 2 : 3; par.hp_digits = nd; }
        }
        OPT_D("-hptol", hp_tol)
        OPT_I("-hpext", hp_ext)
        else if (!strcmp(a, "-y")) { i = need(i, argc, a); yfile = argv[i]; }
        else if (!strcmp(a, "-x")) { i = need(i, argc, a); xfile = argv[i]; }
        else if (!strcmp(a, "-z")) { i = need(i, argc, a); zfile = argv[i]; }
        else if (!strcmp(a, "-sym")) { i = need(i, argc, a); par.symfile = (!strcmp(argv[i], "none") || !strcmp(argv[i], "0")) ? NULL : argv[i]; }
        else if (!strcmp(a, "-nosym")) par.symfile = NULL;
        else if (!strcmp(a, "-certify")) do_certify = 1;
        else if (!strcmp(a, "-certify-x")) { i = need(i, argc, a); cert_x = argv[i]; }
        else if (!strcmp(a, "-certify-y")) { i = need(i, argc, a); cert_y = argv[i]; }
        else if (!strcmp(a, "-bound")) {
            i = need(i, argc, a);
            const char *s = argv[i];
            if (!strcmp(s, "p") || !strcmp(s, "P") || !strcmp(s, "primal") || !strcmp(s, "upper")) par.bound_side = 1;
            else if (!strcmp(s, "d") || !strcmp(s, "D") || !strcmp(s, "dual") || !strcmp(s, "lower")) par.bound_side = 2;
            else if (!strcmp(s, "0") || !strcmp(s, "off") || !strcmp(s, "none")) par.bound_side = 0;
            else { fprintf(stderr, "brisk: -bound must be p or d\n"); return 1; }
        }
        OPT_D("-boundtol", bound_tol) OPT_D("-boundmargin", bound_margin) OPT_D("-boundtrack", bound_track_tol)
        OPT_D("-symtime", symtime) OPT_I("-symnodes", symnodes) OPT_I("-symbd", symbd) OPT_D("-symmin", symmin) OPT_I("-symsign", symsign) OPT_I("-symsigned", symsigned) OPT_D("-fomrace", fom_race) OPT_I("-returnx", returnx) OPT_I("-symalg", symalg) OPT_I("-symalgmax", symalgmax) OPT_I("-mfipm", mfipm) OPT_I("-lralm", lralm) OPT_I("-lrrank", lr_rank) OPT_I("-lrrmax", lr_rmax) OPT_I("-lrouter", lr_outer) OPT_I("-lrinner", lr_inner) OPT_I("-lrescape", lr_escape) OPT_I("-lrprec", lr_prec) OPT_I("-lrnewton", lr_newton) OPT_D("-lrsigma", lr_sigma) OPT_D("-lrtol", lr_tol) OPT_D("-lrtrace", lr_trace) OPT_D("-mfrho", mf_rho) OPT_I("-mfrmax", mf_rmax) OPT_D("-mfdrop", mf_drop) OPT_I("-mfkmax", mf_kmax) OPT_I("-mfcgmax", mf_cgmax) OPT_D("-mfcgtol", mf_cgtol_min) OPT_D("-mfcgtolmax", mf_cgtol_max) OPT_I("-mfdiag", mf_diag) OPT_I("-mfwarm", mf_warm) OPT_I("-mfstall", mf_stall) OPT_D("-mfcgtime", mf_cgtime) OPT_I("-mfrecycle", mf_recycle) OPT_D("-mfhand", mf_hand) OPT_I("-mfhandcg", mf_handcg) OPT_I("-mfproj", mf_proj) OPT_D("-mfeta", mf_eta) OPT_I("-mftry", mf_try)
        else if (!strcmp(a, "-threads")) { i = need(i, argc, a); const int nt = (int)num(argv[i], a); if (nt >= 1) { omp_set_num_threads(nt); g_threads_user = 1; } }   /* 4.32: BRISK's OpenMP threads (default OMP_NUM_THREADS / all cores) */
        else if (!strcmp(a, "-route")) { i = need(i, argc, a); force_route = (int)num(argv[i], a); }
        else if (!strcmp(a, "-densemem")) { i = need(i, argc, a); par.dense_mem = num(argv[i], a) * 1048576.0; }
        else if (a[0] == '-') { fprintf(stderr, "brisk: unknown option %s\n", a); usage(); return 1; }
        else if (!fname) fname = a;
        else { fprintf(stderr, "brisk: more than one input file\n"); return 1; }
#undef OPT_D
#undef OPT_I
    }
    if (!fname) { usage(); return 1; }
    if ((cert_x || cert_y) && par.hp_kind > 0)      /* 5.2: in the master precision of -prec (hpsolve.c) */
        return brisk_hp_certify_given(fname, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result, cert_x, cert_y, par.hp_kind, par.hp_digits, par.verbose);
    if (cert_x || cert_y) return certify_given(fname, cert_x, cert_y);
    { extern int g_ps_need_x; g_ps_need_x = (xfile != NULL || getenv("BRISK_XOUT") != NULL || par.crossover != 0 || par.bound_side != 0 || (g_brisk_result != NULL && par.returnx > 0)) ? 1
                                         : (g_brisk_result != NULL && par.returnx < 0) ? 2 : 0; }
    g_rank_par = &par;
    g_natt = 0; g_run_no = 0;
    if (!(par.tol > 0)) { fprintf(stderr, "brisk: -tol must be positive\n"); return 1; }
    if (par.fom > 0) {
        /* 4.31: the first-order engine works on the problem as read (no free elimination,
         * dual form or trace bound: those serve the Schur complement, which it never forms) */
        if (par.free_elim < 0) par.free_elim = 0;
        if (par.dualize < 0) par.dualize = 0;
        if (par.tracebound < 0) par.tracebound = 0;
        if (par.chordal < 0) par.chordal = 0;     /* the moment form of the conversion is the IPM's; the engine takes the blocks as read */
    }
    if (par.mfipm > 0) {
        /* 4.34: the matrix-free IPM takes the blocks as read (the dual form, the trace bound and
         * the chordal conversion serve the Schur complement, which it never forms); the
         * fill-free elimination of split free pairs is kept */
        par.fom = 0;
        if (par.dualize < 0) par.dualize = 0;
        if (par.tracebound < 0) par.tracebound = 0;
        if (par.chordal < 0) par.chordal = 0;
    }
    par.t_start = wtime();
    if (par.lralm > 0) (void)brisk_threads_busy(par.verbose);   /* (the process ends, or brisk_run restores the count) */
    if (!g_hp_inner) brisk_hp_clear();                                             /* the text of an earlier high-precision solution */
    if (par.hp_kind > 0 && g_hp_inner) { par.hp_kind = 0; par.verbose = -1; }      /* the double solve inside a high-precision one */
    else if (par.hp_kind > 0) {                                                    /* 5.0: high precision */
        /* The high-precision methods start from a double solve of the same problem by the
         * default pipeline (all presolve, any engine): it is run here, silently, with the
         * result on the data as read. Without a result (infeasible, failed) the
         * interior-point method starts cold in its own double level. */
        BriskResult warm; int have_warm = 0;
        memset(&warm, 0, sizeof(warm)); warm.status = -1;
        if (!getenv("BRISK_HPCOLD")) {
            char **av = malloc(sizeof(char *) * (size_t)(argc + 2)); int ac = 0;
            for (int i = 0; i < argc; i++) {
                const char *a = argv[i];
                if (i > 0 && (!strcmp(a, "-prec") || !strcmp(a, "-hptol") || !strcmp(a, "-hpext") || !strcmp(a, "-fom") || !strcmp(a, "-lralm") || !strcmp(a, "-x") || !strcmp(a, "-y") || !strcmp(a, "-z")
                              || !strcmp(a, "-maxit") || !strcmp(a, "-tol") || !strcmp(a, "-bound"))) { i++; continue; }
                if (i > 0 && !strcmp(a, "-certify")) continue;     /* (the certificate is built in high precision: hpcert.inc) */
                av[ac++] = argv[i];
            }
            static char pr[] = "-prec", pd[] = "dd";
            av[ac++] = pr; av[ac++] = pd;                 /* marks the inner run (g_hp_inner) */
            BriskResult *saved = g_brisk_result;
            const BriskData *sdata = g_brisk_data;
            g_brisk_result = &warm; g_hp_inner = 1;
            const double tw = wtime();
            int (*hook)(const char *, int) = NULL;
            const int fd = hp_mute_begin(&hook);
            brisk_main(ac, av);
            hp_mute_end(fd, hook);
            g_hp_inner = 0; g_brisk_result = saved; g_brisk_data = sdata;
            free(av);
            have_warm = warm.status >= 0 && warm.status != 1 && warm.status != 2 && warm.have_x && warm.X && warm.y;   /* (infeasible: the cold start has its own detection) */
            if (par.verbose >= 0) printf("high precision: double solve %s (max rel error %.1e) in %.2fs%s\n", warm.status >= 0 ? warm.status_str : "failed",
                                         fmax(fmax(fabs(warm.err[1]), fabs(warm.err[3])), fmax(fabs(warm.err[5]), fabs(warm.err[6]))), wtime() - tw, have_warm ? "" : "; no X returned: cold start");
        }
        const int rc = brisk_hp_run(fname, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result, have_warm ? &warm : NULL, &par, par.hp_kind, par.hp_digits, par.hp_tol, yfile, xfile, zfile);
        brisk_result_free(&warm);
        /* 5.10: the high-precision log ends with the way to a guaranteed bound (in the caller's syntax) */
        if (par.verbose > 0 && !g_hp_inner && !getenv("BRISK_NOSUMMARY") && (rc == 0 || rc == 10)) bound_hint(1);
        return rc;
    }
    if (par.lralm > 0) return lralm_run(fname, &par, yfile, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result);   /* 4.40: its own reading, measures and report */
    Run run, *u = &run;
    double textra = 0;      /* time of a discarded solve */
    const int hsd_user = par.hsd, hsddir_user = par.hsd_dir;
    {
        int rc = run_pipeline(fname, &par, do_fr, force_route, u);
        if (rc == 8) { run_free(u); par.mf_try = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }      /* 5.7: the symmetry search again with its full budget (run_pipeline) */
        if (rc == 3) { run_free(u); do_fr = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
        if (rc == 4) {   /* 4.39: the deferred first-order decision, on the problem as read */
            run_free(u); par.fom = 1;
            if (par.free_elim < 0) par.free_elim = 0;
            if (par.dualize < 0) par.dualize = 0;
            if (par.tracebound < 0) par.tracebound = 0;
            if (par.chordal < 0) par.chordal = 0;
            rc = run_pipeline(fname, &par, do_fr, force_route, u);
            if (rc == 3) { run_free(u); do_fr = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
        }
        if (rc == 7) {   /* 5.7: the matrix-free method first; the standard method if it does not end OPTIMAL */
            run_free(u);
            Params pr = par;
            pr.mf_try_tl = par.timelimit; pr.mfipm = 1; pr.mf_trial = par.mf_trial; pr.mf_trial_total = par.mf_trial_total; par.mf_trial = 0; par.mf_trial_total = 0; pr.mf_try = 0; pr.fom = 0; pr.fom_race = 0;
            pr.symfile = NULL;            /* the searches found nothing in the first pass */
            if (pr.dualize < 0) pr.dualize = 0;
            if (pr.tracebound < 0) pr.tracebound = 0;
            if (pr.chordal < 0) pr.chordal = 0;
            const double tr0 = wtime();
            /* at most the expected time of the standard solve, and what is left of a time limit */
            { const double lim = (tr0 - par.t_start) + fmax(3.0, par.mf_try_budget); if (!(pr.timelimit > 0) || lim < pr.timelimit) pr.timelimit = lim; }
            rc = run_pipeline(fname, &pr, do_fr, force_route, u);
            /* kept when OPTIMAL, and when the run's own time limit is over (a second start would return nothing) */
            /* 5.8: when the standard method cannot follow (the Schur complement does not fit), a
             * result at reduced accuracy is kept too: the first-order engine would rarely do
             * better, and the matrix-free method would run once more at its end */
            if (rc == 0 && u->measured && (u->status == ST_OPTIMAL || (par.timelimit > 0 && wtime() - par.t_start >= par.timelimit)
                                           || (!pr.mf_try_fits && u->status == ST_REDUCED))) {
                par.mfipm = 1;                               /* kept: the run is a matrix-free run from here on */
                brisk_note("matrix-free interior-point method, tried first automatically (-mftry): its result is returned (%.1f s)", wtime() - tr0);
            } else {
                if (par.verbose >= 0) printf("matrix-free interior-point method first: %.1e after %.1f s, continuing with the standard method\n", rc == 0 ? u->acc : NAN, wtime() - tr0);
                if (rc == 0) { att_log("the matrix-free method (first)", wtime() - tr0, u); textra += wtime() - tr0; }
                if (getenv("BRISK_MFTRYONLY")) { printf("MFTRYONLY failed after %.2f s (since start %.2f)\n", wtime() - tr0, wtime() - par.t_start); exit(0); }
                run_free(u);
                par.mf_try = 0;
                rc = run_pipeline(fname, &par, do_fr, force_route, u);
                if (rc == 3) { run_free(u); do_fr = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
                if (rc == 4) { run_free(u); par.fom = 1; if (par.free_elim < 0) par.free_elim = 0; if (par.dualize < 0) par.dualize = 0; if (par.tracebound < 0) par.tracebound = 0; if (par.chordal < 0) par.chordal = 0;
                               rc = run_pipeline(fname, &par, do_fr, force_route, u);
                               if (rc == 3) { run_free(u); do_fr = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); } }
            }
        }
        if (rc == 6) {   /* 4.42: the first-order engine first, within its budget; the interior-point method if it misses the tolerance */
            run_free(u);
            Params pr = par;
            pr.fom = 1; pr.fom_race = 0;
            if (pr.free_elim < 0) pr.free_elim = 0;
            if (pr.dualize < 0) pr.dualize = 0;
            if (pr.tracebound < 0) pr.tracebound = 0;
            if (pr.chordal < 0) pr.chordal = 0;
            const double tr0 = wtime();
            /* with a time limit, at most the same share of what is left of it */
            if (par.timelimit > 0) par.fom_race_budget = fmin(par.fom_race_budget, fmax(0.0, par.fom_race * (par.timelimit - (tr0 - par.t_start))));
            const double lim = (tr0 - par.t_start) + par.fom_race_budget;
            pr.timelimit = lim;
            rc = run_pipeline(fname, &pr, do_fr, force_route, u);
            if (rc == 0 && u->status == ST_OPTIMAL && u->measured) {
                par.fom = 1;                                 /* kept: the run is a first-order run from here on */
                if (par.verbose >= 0) printf("first-order engine first: reached the tolerance in %.1f s\n", wtime() - tr0);
            } else {
                if (par.verbose >= 0) printf("first-order engine first: %.1e after %.1f s, continuing with the interior-point method\n", rc == 0 ? u->acc : NAN, wtime() - tr0);
                if (rc == 0) { att_log("the first-order engine (first, within its budget)", wtime() - tr0, u); textra += wtime() - tr0; }
                run_free(u);
                par.fom_race = 0;
                rc = run_pipeline(fname, &par, do_fr, force_route, u);
                if (rc == 3) { run_free(u); do_fr = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
                if (rc == 4) { run_free(u); par.fom = 1; if (par.chordal < 0) par.chordal = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
            }
        }
        if (rc != 0) return 2;
        /* 4.37 (B1): the first attempt, and the in-solver method retry as a re-solve */
        g_first_t = u->R.t_total - (u->R.nretry ? u->R.t_retry : 0) + u->tpre + u->pr.t + u->tread;
        g_first_acc = u->R.nretry ? u->R.acc_first : u->acc;
        g_first_internal = u->R.nretry;
        if (u->R.nretry) {
            att_log(u->R.retry_hsd ? "the self-dual embedding" : "the standard method", u->R.t_retry, NULL);
            g_att[g_natt - 1].kept = u->R.retry_kept;
        }
    }
    /* The moment form of the chordal conversion is the accurate one when it converges; on
     * some ill-posed relaxations it stalls (case588_sdet, case1354_pegase: the residual of
     * the moment rows stops decreasing). Then the overlap-equality form is tried, which
     * converges there to ~1e-6. */
    /* first a second moment form: the other clique candidate (support cliques, merging)
     * with NT; it costs one more solve of the converted problem (case588_sdet: 1.5e-6 with
     * the plain extension and HKM, 2.8e-7 this way), against the overlap-equality and
     * unconverted solves below (220 s there) */
    if (u->P.ps && u->P.ps->mom && u->measured && par.verbose > 1)
        printf("   (measured on the original data: DIMACS %.1e %.1e %.1e %.1e %.1e %.1e)\n",
               u->pr.err[1], u->pr.err[2], u->pr.err[3], u->pr.err[4], u->pr.err[5], u->pr.err[6]);
    if (!par.fom && !par.mfipm && u->P.ps && u->P.ps->mom && u->measured && u->status != ST_OPTIMAL && u->acc > par.red_acc
        && par.chordal_sup < 0 && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.3 * par.timelimit))) {
        Params p2 = par;
        p2.chordal_sup = 4; p2.chordal_merge = 0.6;
        p2.hsd = 1;
        p2.hsd_dir = 1;
        p2.chordal_need = 1;
        if (par.verbose >= 0)
            printf("chordal: the moment form ended at %.1e, re-solving with support cliques and NT\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        const int rc2 = run_pipeline(fname, &p2, do_fr, force_route, &second);
        if (rc2 == 5) run_free(&second);
        if (rc2 == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the chordal moment form with support cliques", tsecond, &second);
            if (run_better(&second, u)) {
                g_att[g_natt - 1].kept = 1;
                second.R.iters += u->R.iters;
                run_free(u);
                run = second;
                textra += tfirst;
            } else {
                u->R.iters += second.R.iters;
                run_free(&second);
                textra += tsecond;
            }
        }
    }
    /* near the accuracy floor of the original-space measures (the gap measure is the
     * primal rounding times multipliers of 1e3-1e4 on the AC-OPF data), the slower forms
     * below are not worth it when the unconverted problem costs 20x more per iteration
     * (case588_sdet: 388 s for 8e-7 against 9 s for 2e-6; case1354: 2000 s) */
    int near_floor = 0;
    if (u->P.ps && u->P.ps->mom && u->measured && u->acc > 1e-6 && u->acc <= 1e-5
        && u->P.ps->mom->cost0 > 20.0 * u->P.ps->mom->cost1 && par.chordal < 0 && !getenv("BRISK_FULLFALLBACK")) {
        near_floor = 1;
        if (par.verbose >= 0) printf("chordal: the moment form ended at %.1e; the unconverted problem costs %.0fx more per iteration, not re-solved\n",
                                     u->acc, u->P.ps->mom->cost0 / u->P.ps->mom->cost1);
    }
    if (!par.fom && !par.mfipm && u->P.ps && u->P.ps->mom && u->measured && u->status != ST_OPTIMAL && u->acc > par.red_acc && !near_floor
        && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        p2.chordal_form = 0;
        p2.hsd = 1;
        p2.hsd_dir = hsddir_user;
        p2.chordal_need = 1;
        if (par.verbose >= 0)
            printf("chordal: the moment form ended at %.1e, re-solving with overlap equalities\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        const int rc3 = run_pipeline(fname, &p2, do_fr, force_route, &second);
        if (rc3 == 5) run_free(&second);
        if (rc3 == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the chordal overlap-equality form", tsecond, &second);
            if (run_better(&second, u)) {
                g_att[g_natt - 1].kept = 1;
                second.R.iters += u->R.iters;
                run_free(u);
                run = second;
                textra += tfirst;
            } else {
                u->R.iters += second.R.iters;
                run_free(&second);
                textra += tsecond;
            }
        }
    }
    /* ... and when neither chordal form reaches 1e-6 (both are more fragile than the
     * unconverted problem on ill-posed instances), the unconverted problem is solved: the
     * converted attempts cost a small fraction of that solve. */
    /* 4.22: never the unconverted problem when its dense Schur complement cannot fit in
     * memory (case4661..9241: m = 44k..104k, 15..87 GB; the attempt ended in the OOM killer) */
    const double ram = brisk_mem_limit();
    const int unconv_fits = !(u->O && 8.0 * (double)u->O->m * (double)u->O->m > 0.6 * ram);
    if (!unconv_fits && par.verbose >= 0 && u->P.ps && u->P.ps->chordal && u->acc > 1e-6)
        printf("chordal: the unconverted problem (m = %d) does not fit in memory, not re-solved\n", u->O->m);
    if (!par.fom && !par.mfipm && u->P.ps && u->P.ps->chordal && u->measured && u->status != ST_OPTIMAL && u->acc > par.red_acc && !near_floor && unconv_fits
        && par.chordal < 0 && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.3 * par.timelimit))) {
        Params p2 = par;
        p2.chordal = 0;
        p2.hsd = hsd_user;
        p2.hsd_dir = hsddir_user;
        if (par.verbose >= 0)
            printf("chordal: the converted problem ended at %.1e, solving the unconverted problem\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_pipeline(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the unconverted problem", tsecond, &second);
            if (run_better(&second, u)) {
                g_att[g_natt - 1].kept = 1;
                second.R.iters += u->R.iters;
                run_free(u);
                run = second;
                textra += tfirst;
            } else {
                u->R.iters += second.R.iters;
                run_free(&second);
                textra += tsecond;
            }
        }
    }
    /* 4.29: -acc high after facial reduction: the dual of the removed constraints is rebuilt
     * from the reduced dual slack, and the closer the reduced solve gets to the boundary the
     * less margin that recovery has (gpp500-1: lambda_min(Z) -2e-7 after the default solve,
     * -7.5e-5 after the high one; 8e-9 against 3e-8 on the original data). When the high
     * result misses 1e-8 there, the default-accuracy solve is run as well and the better
     * one kept. */
    if (!par.fom && !par.mfipm && par.acc_level == 2 && do_fr && u->fr_removed > 0 && u->measured && u->acc > 1e-8
        && fmax(fmax(u->R.relgap, u->R.pinf), fmax(u->R.dinf, u->R.relcomp)) <= 1e-8   /* the reduced solve passed 1e-8: */
                                                                                       /* the loss is the recovery's (swissroll: 8e-7 either way, 64 -> 129 s) */
        && u->status != ST_PINFEAS && u->status != ST_DINFEAS && !getenv("BRISK_NOHIFALLBACK")
        && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        params_accuracy(&p2, 1);
        if (par.verbose >= 0)
            printf("postsolve: the high-accuracy result is %.1e on the original data after facial reduction, solving at the default accuracy as well\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_pipeline(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the default accuracy", tsecond, &second);
            if (run_better(&second, u)) {
                g_att[g_natt - 1].kept = 1;
                second.R.iters += u->R.iters;
                run_free(u);
                run = second;
                textra += tfirst;
            } else {
                u->R.iters += second.R.iters;
                run_free(&second);
                textra += tsecond;
            }
        }
    }
    /* (4.29: these fallbacks start above red_acc: 1e-6, and 1e-4 at -acc low, where a result
     * just above 1e-6 is the expected outcome, not a failure: rose15 23 -> 47 s otherwise) */
    /* 4.30: the eliminated kernel form (free pairs pivoted out, trace bound) is 2-4x cheaper
     * per iteration but has a residual floor of 1-2e-6 on a few instances (roa_acrobot_d6_K:
     * 1.7e-6 against 2e-7 with the pairs kept). Just above 1e-6, the problem is solved
     * again as it was read; far above, both paths fail (the ill-posed relaxations). Not
     * after the dual form: the file's own form is the expensive side (roa_vdp_inner_d12_I:
     * 719 s against 6 s). */
    /* 4.30: an infeasibility claim from a presolved problem (free elimination, trace bound,
     * dual form) is not returned as such: the problem is solved again as read and that
     * answer kept (butcher: the eliminated problem with 8059 pairs left claimed primal
     * infeasibility after 4 iterations; as read it is optimal) */
    if ((u->fe || u->dual || u->Otb) && (u->status == ST_PINFEAS || u->status == ST_DINFEAS)
        && par.free_elim < 0 && par.dualize < 0 && par.tracebound < 0) {
        Params p2 = par;
        p2.free_elim = 0; p2.tracebound = 0; p2.dualize = 0;
        if (par.verbose >= 0) printf("the presolved problem looks %s infeasible, re-solving the problem as read\n", u->status == ST_PINFEAS ? "primal" : "dual");
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_pipeline(fname, &p2, do_fr, force_route, &second) == 0) {
            att_log("the problem as read (infeasibility check)", second.R.t_total + second.tpre + second.pr.t + second.tread, &second);
            g_att[g_natt - 1].kept = 1;
            second.R.iters += u->R.iters;
            run_free(u);
            run = second;
            textra += tfirst;
        }
    }
    /* 4.30: the same when the trace bound binds (slack below a thousandth of R) and the
     * result misses the tolerance: the Gram side is unattained, the bound then shifts the
     * value by R times its dual (roa_univar_d16: 8e-4), and only the problem as read can do
     * better (its image form ends at 2e-7 in 22 s) */
    const int tb_binding = u->Otb && u->measured && u->tbs < 1e-3 * u->tbR && u->acc > par.tol;
    if (u->fe && !u->fe->cheap && u->measured && par.free_elim < 0 && par.dualize < 0 && par.tracebound < 0
        && ((!u->dual && u->acc > 1e-6 && u->acc <= 1e-5) || tb_binding)
        && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        p2.free_elim = 0; p2.tracebound = 0; p2.dualize = 0;
        if (par.verbose >= 0) printf("the eliminated problem ended at %.1e on the original data%s, re-solving the problem as read\n", u->acc,
                                     tb_binding ? " with the trace bound binding" : "");
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_pipeline(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the problem as read", tsecond, &second);
            if (run_better(&second, u)) {
                g_att[g_natt - 1].kept = 1;
                second.R.iters += u->R.iters;
                run_free(u);
                run = second;
                textra += tfirst;
            } else {
                u->R.iters += second.R.iters;
                run_free(&second);
                textra += tsecond;
            }
        }
    }
    /* 4.20: a result that misses 1e-6 on the original data and came from one method only
     * (the in-solver score can pass while the measured error does not: hinf11's endgame
     * point has lambda_min(Z) = -1.6e-5) is re-solved with the other method */
    if (hsd_user < 0 && u->measured && u->acc > par.red_acc && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && (u->R.methods == 1 || u->R.methods == 2) && !(u->P.ps && u->P.ps->chordal)
        && !(u->R.free_hsd && u->acc < 1e-4 && !getenv("BRISK_FREERESOLVE"))   /* 4.24: split free pairs: the standard method stalls */
        && !(u->fr_removed > 0 && u->pr.err[4] >= 0.5 * u->acc)      /* the removed duals: see below */
        && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        p2.hsd = u->R.methods == 1 ? 1 : 0;
        p2.no_retry = 1;
        if (par.verbose >= 0)
            printf("the %s ended at %.1e on the original data, re-solving with the %s\n",
                   p2.hsd ? "standard method" : "self-dual embedding", u->acc, p2.hsd ? "self-dual embedding" : "standard method");
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_pipeline(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the other method", tsecond, &second);
            second.R.methods |= u->R.methods;
            if (run_better(&second, u)) {
                g_att[g_natt - 1].kept = 1;
                second.R.iters += u->R.iters;
                run_free(u);
                run = second;
                textra += tfirst;
            } else {
                u->R.iters += second.R.iters;
                u->R.methods |= second.R.methods;
                run_free(&second);
                textra += tsecond;
            }
        }
    }
    (void)hsd_user;
    par.hsd = hsd_user; par.hsd_dir = hsddir_user;     /* the FR fallbacks below start from the user's settings */
    /* Facial reduction can remove constraints whose duals must then be rebuilt through the
     * whole chain of reductions; on problems of high singularity degree (SOS/moment
     * relaxations with hundreds of nested reductions) that reconstruction can lose the
     * dual feasibility the reduced solve had. The returned point is judged on the original
     * data, so when the recovered dual is what spoils it:
     *   1. solve the reduced problem again with the self-dual embedding (unless that is
     *      what produced it): its iterates stay well centred, so the reduced dual slack
     *      keeps the margin the recovery needs (rose13, taha1a);
     *   2. then, if still needed, solve without facial reduction;
     * and keep the best answer. */
    for (int stage = 0; stage < 2; stage++) {
        const int need = do_fr && fr_retry && u->fr_removed > 0 && u->measured && u->status != ST_OPTIMAL
            && u->status != ST_PINFEAS && u->status != ST_DINFEAS
            && u->pr.err[4] > fmax(100.0 * fmax(par.tol, 1e-8), 1e-3 * u->acc) && u->pr.err[4] >= 0.5 * u->acc
            && (!brisk_stop_flag && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit));
        if (!need) break;
        const int used_hsd = par.hsd == 1 || u->R.retried || (u->R.methods & 2);
        if (stage == 0 && used_hsd) continue;
        Params p2 = par;
        p2.dual = 0;               /* the fallbacks use the primal-dual method */
        /* when the reduced solve is good apart from the removed duals, the problem has a
         * solution and infeasibility exits of the new solve are false (rose15 without
         * facial reduction: free variables make it claim dual infeasibility) */
        p2.known_feasible = fmax(fmax(u->pr.err[1], u->pr.err[2]), fmax(fabs(u->pr.err[5]), fabs(u->pr.err[6]))) <= 1e-6;
        int fr2 = 1;
        if (stage == 0) p2.hsd = 1; else fr2 = 0;
        if (par.verbose >= 0)
            printf("postsolve: dual of the removed constraints not recovered to tolerance (err4 %.1e), %s\n",
                   u->pr.err[4], stage == 0 ? "re-solving the reduced problem with the self-dual embedding"
                                            : "re-solving without facial reduction");
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_pipeline(fname, &p2, fr2, force_route, &second) != 0) break;
        const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log(stage == 0 ? "the reduced problem with the self-dual embedding" : "the problem without facial reduction", tsecond, &second);
        /* an infeasibility claim cannot replace a solved reduced problem: the reduced
         * primal solution is feasible for the original problem and its dual bounds the
         * value (rose15 without facial reduction reports a false dual infeasibility) */
        const int second_inf = second.status == ST_PINFEAS || second.status == ST_DINFEAS;
        if (!second_inf && run_better(&second, u)) {
            g_att[g_natt - 1].kept = 1;
            if (par.verbose >= 0) printf("  the new solve is better, keeping it\n");
            second.R.iters += u->R.iters;
            run_free(u);
            run = second;
            textra += tfirst;
        } else {
            if (par.verbose >= 0) printf("  the new solve is not better, keeping the previous one\n");
            u->R.iters += second.R.iters;
            run_free(&second);
            textra += tsecond;
        }
    }
    /* 4.20: SDP crossover on the problem as read (Newton on the rank-revealed KKT system) */
    if (par.crossover != 0 && u->measured && u->Xo && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && u->acc > 1e-15 && u->acc < 1e-3) {
        const double tsolve = u->R.t_total + u->tpre + u->pr.t + textra;
        const double budget = par.crossover > 0 ? fmax(1.0, 3.0 * tsolve) : fmax(0.05, 0.1 * tsolve);
        double eb[7], ea[7], pd[2];
        const double tx0 = wtime();
        if (sdp_crossover(u->Ofin ? u->Ofin : u->O, u->Xo, u->yo, par.verbose, budget, par.crossover > 0 ? 30000.0 : 8000.0, eb, ea, pd)) {
            brisk_note("crossover (Newton on the rank-revealed system): max error %.1e -> %.1e", fmax(fmax(eb[1], eb[2]), fmax(eb[4], fmax(fabs(eb[5]), fabs(eb[6])))), fmax(fmax(ea[1], ea[2]), fmax(ea[4], fmax(fabs(ea[5]), fabs(ea[6])))));
            for (int i = 1; i <= 6; i++) u->pr.err[i] = ea[i];
            u->pr.pobj = pd[0]; u->pr.dobj = pd[1];
            u->acc = fmax(fmax(ea[1], ea[2]), fmax(ea[4], fmax(fabs(ea[5]), fabs(ea[6]))));
            if (u->acc <= par.tol) u->status = ST_OPTIMAL;
            else if (u->acc <= par.red_acc) u->status = ST_REDUCED;
        }
        textra += wtime() - tx0;
    }
    BoundCert *bc = &u->cert;
    /* 4.39: -certify: the rigorous check of the certificate on the problem as read; a
     * certified one counts as valid whatever BRISK's own tolerances say */
    int certified = 0;
    double rigorous = NAN, t_cert = 0;
    char cert_why[160] = "";
    if (do_certify && par.bound_side && bc->have && u->status != ST_PINFEAS && u->status != ST_DINFEAS) {
        const double tc0 = wtime();
        certified = brisk_certify_orig(u->Ofin ? u->Ofin : u->O, par.bound_side, bc->X, bc->y, g_brisk_data != NULL, &rigorous, cert_why, sizeof(cert_why));
        t_cert = wtime() - tc0;
        if (certified && !bc->valid) bc->valid = 1;
    }
    /* 4.37 bound mode: a certificate replaces its side of the returned pair, which is then
     * measured again on the problem as read */
    if (par.bound_side && bc->have && bc->valid && u->status != ST_PINFEAS && u->status != ST_DINFEAS) {
        PSOrig *Ob = u->Ofin ? u->Ofin : u->O;
        int replaced = 0;
        if (par.bound_side == 1 && u->Xo && bc->X) {
            for (int k = 0; k < Ob->nblk; k++) memcpy(u->Xo[k], bc->X[k], sizeof(double) * (Ob->bs[k] < 0 ? (size_t)-Ob->bs[k] : (size_t)Ob->bs[k] * Ob->bs[k]));
            replaced = 1;
        } else if (par.bound_side == 2 && bc->y) { memcpy(u->yo, bc->y, sizeof(double) * Ob->m); replaced = 1; }
        if (replaced && (u->Xo || par.bound_side == 2)) {
            PSResult pb;
            ps_measure(Ob, u->Xo, u->yo, &pb);
            pb.recovered = u->pr.recovered; pb.unrecovered = u->pr.unrecovered; pb.t += u->pr.t;
            if (pb.have_x || !u->measured) {
                u->pr = pb;
                if (pb.have_x) {
                    u->acc = fmax(fmax(pb.err[1], pb.err[2]), fmax(pb.err[4], fmax(fabs(pb.err[5]), fabs(pb.err[6]))));
                    if (u->status == ST_OPTIMAL || u->status == ST_REDUCED || u->status == ST_NUMERIC) {
                        if (u->acc <= par.tol) u->status = ST_OPTIMAL;
                        else if (u->acc <= par.red_acc) u->status = ST_REDUCED;
                        else u->status = ST_NUMERIC;
                    }
                }
            }
        }
    }
    Problem *PP = &u->P;
    PSOrig *O = u->Ofin ? u->Ofin : u->O;
    Result R = u->R;
    PSResult pr = u->pr;
    int status = u->status, measured = u->measured;
    double acc = u->acc, tread = u->tread, tpre = u->tpre;
    double **Xo = u->Xo, *yo = u->yo;
    /* 4.37 (B2a): a result within 100 x red_acc was labelled reduced accuracy with a qualifier.
     * 5.4: the label, the status and the exit code say NUMERICAL DIFFICULTIES again (reduced
     * accuracy promises a max error <= red_acc, and the interfaces report the status without the
     * qualifier); the qualifier stays in the text */
    int near_miss = 0;
    if (status == ST_NUMERIC && acc > par.red_acc && acc <= 100.0 * par.red_acc) near_miss = 1;
    /* 4.37 (B2b): the likely cause of a result that misses the tolerance */
    char cause[320] = "";
    if ((status == ST_REDUCED || status == ST_NUMERIC || status == ST_MAXIT || status == ST_TIME) && acc > 10.0 * par.tol)
        diagnose_cause(u, &par, O, Xo, yo, &pr, acc, cause, sizeof(cause));
    if (near_miss)
        printf("status: %s (max rel error %.1e, within 100x the %.0e reduced-accuracy level)   iterations: %d\n", status_str[status], acc, par.red_acc, R.iters);
    else if (status == ST_REDUCED || status == ST_NUMERIC || status == ST_MAXIT || status == ST_TIME)
        printf("status: %s (max rel error %.1e)   iterations: %d\n", status_str[status], acc, R.iters);
    else
        printf("status: %s   iterations: %d\n", status_str[status], R.iters);
    int infeas = status == ST_PINFEAS || status == ST_DINFEAS;
    if (infeas && par.verbose != 0) {
        /* 5.8 (a user's report): say whether the returned point certifies the verdict on the
         * data as read. After a facial reduction (or on a weakly infeasible problem) it does
         * not: the verdict is that of a reduced problem and no ray need exist. */
        double cobj = 0, cviol = 0;
        const int cert = ps_infeas_check(O, Xo, yo, status == ST_PINFEAS ? 1 : 2, &cobj, &cviol);
        if (cert == 1) snprintf(cause, sizeof cause, "certificate: the returned %s is a ray of the data as read (%s %.1e, cone violation %.1e relative)", status == ST_PINFEAS ? "y" : "X", status == ST_PINFEAS ? "b'y" : "<C,X>", cobj, cviol);
        else if (cert == 0) snprintf(cause, sizeof cause, "no certificate: the returned %s is not a ray of the data as read (%s %.1e, cone violation %.1e relative); the verdict rests on the presolve (a reduced problem is infeasible; a weakly infeasible problem has no ray)", status == ST_PINFEAS ? "y" : "X", status == ST_PINFEAS ? "b'y" : "<C,X>", cobj, cviol);
        else snprintf(cause, sizeof cause, "no certificate: no point was returned");
        printf("  %s\n", cause);
    }
    if (cause[0] && par.verbose != 0 && !infeas) printf("  likely cause: %s\n", cause);
    double pobj = measured && !infeas ? pr.pobj : R.pobj, dobj = infeas ? R.dobj : pr.dobj;
    printf("optimal value (SDPA/SDPLIB convention, max <F0,Y>): %.10e\n", -pobj);
    printf("  primal obj <C,X> = %.10e   dual obj b'y = %.10e\n", pobj, dobj);
    if (measured && !infeas) {
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n",
               pr.err[1], pr.err[2], pr.err[3], pr.err[4], pr.err[5], pr.err[6]);
    } else {
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n",
               R.err[1], R.err[2], R.err[3], infeas ? R.err[4] : pr.err[4], R.err[5], R.err[6]);
    }
    if (par.verbose) {
        printf("  (errors of the %s problem, Z = C - A'y; presolved problem: gap %.2e pinf %.2e dinf %.2e compl %.2e)\n",
               measured ? "original" : "converted", R.relgap, R.pinf, R.dinf, R.relcomp);
        if (PP->ps && PP->ps->nrec)
            printf("  postsolve: %d constraint(s) removed by presolve, %d dual(s) recovered, %d not (%.3fs)\n",
                   PP->ps->nrec, pr.recovered, pr.unrecovered, pr.t);
    }
    printf("direction: %s%s\n", R.direction == 1 ? "NT" : "HKM", R.retried ? " (self-dual embedding)" : "");
    if (par.verbose > 0) printf("(solver setup %.3fs)\n", R.t_setup);
    const double t_all = R.t_total + tpre + pr.t + textra;
    printf("time: total %.3fs (read %.3f, presolve+analysis %.3f) | schur %.3f  chol/solve %.3f  dense %.3f  step %.3f  postsolve %.3f\n",
           t_all, tread, tpre, R.t_schur, R.t_chol, R.t_dense, R.t_step, pr.t);
    /* 4.37 (A1): the bound */
    if (par.bound_side && par.verbose != 0) {
        if (!bc->have) printf("BOUND (%s): no certificate (no admissible candidate)\n", par.bound_side == 1 ? "P" : "D");
        else {
            printf("BOUND (%s): %s bound %s = %.10e  (certificate: %s %.1e rel, lambda_min(%s) %.1e rel; from attempt %d, %s%s)\n",
                   par.bound_side == 1 ? "P" : "D", par.bound_side == 1 ? "upper" : "lower", par.bound_side == 1 ? "<C,X>" : "b'y", bc->value,
                   par.bound_side == 1 ? "|A(X)-b|" : "|C-A'y-Z|", bc->resid, par.bound_side == 1 ? "X" : "Z", bc->lammin, bc->attempt, bc->src,
                   bc->theta > 0 ? ", margin restored" : "");
            printf("           (SDPA convention, max <F0,Y>: %s bound %.10e)\n", par.bound_side == 1 ? "a lower" : "an upper", -bc->value);
            if (!bc->valid && !certified)
                printf("           certificate %s: the bound holds only approximately\n",
                       bc->lammin < 0 ? "not positive semidefinite" : "residual above -boundtol");
            else if (bc->valid == 1)
                printf("           (positive semidefinite, but below the -boundmargin margin: rounding may break it)\n");
        }
    }
    if (do_certify && par.verbose != 0) {
        if (!par.bound_side) printf("(-certify needs -bound p or -bound d)\n");
        else if (certified)
            printf("CERTIFIED (%s): rigorous %s bound %.17g on the optimal value (SDPA convention: %s bound %.17g; %.2fs)\n",
                   par.bound_side == 1 ? "P" : "D", par.bound_side == 1 ? "upper" : "lower", rigorous,
                   par.bound_side == 1 ? "a lower" : "an upper", -rigorous, t_cert);
        else if (bc->have) printf("NOT CERTIFIED (%s): %s\n", par.bound_side == 1 ? "P" : "D", cert_why);
    }
    /* 4.37 (B1): re-solves */
    if (g_natt > 0 && par.verbose != 0) {
        double tre = 0;
        char names[512] = "";
        const char *kept = NULL;
        int hint_method = 0, hint_asread = 0, hint_fr = 0, hint_chordal = 0;
        for (int a = 0; a < g_natt; a++) {
            tre += g_att[a].t;
            if (strlen(names) + strlen(g_att[a].name) + 3 < sizeof(names)) { if (a) strcat(names, "; "); strcat(names, g_att[a].name); }
            if (g_att[a].kept) kept = g_att[a].name;
            if (strstr(g_att[a].name, "embedding") || strstr(g_att[a].name, "method")) hint_method = 1;
            if (strstr(g_att[a].name, "as read")) hint_asread = 1;
            if (strstr(g_att[a].name, "facial")) hint_fr = 1;
            if (strstr(g_att[a].name, "chordal") || strstr(g_att[a].name, "unconverted")) hint_chordal = 1;
        }
        printf("NOTE: %d RE-SOLVE%s %s RUN (%s) - %s TOOK %.1f s OF THE %.1f s TOTAL.\n", g_natt, g_natt > 1 ? "S" : "", g_natt > 1 ? "WERE" : "WAS",
               names, g_natt > 1 ? "THEY" : "IT", tre, t_all);
        printf("      THE FIRST ATTEMPT ALONE ENDED AT %.1e (%s) AFTER %.1f s; THE RETURNED RESULT IS %.1e.\n",
               g_first_acc, g_first_internal ? "internal measure" : "on the file data", g_first_t, acc);
        if (kept) printf("      THE RETURNED RESULT COMES FROM THE RE-SOLVE (%s).\n", kept);
        char hint[256] = "";
        if (hint_method) strcat(hint, "-hsd or -nohsd fixes the method; ");
        if (hint_asread) strcat(hint, "-freeelim 1 keeps the eliminated problem; ");
        if (hint_fr) strcat(hint, "-nofrretry; ");
        if (hint_chordal) strcat(hint, "-chordal 1 keeps the converted problem; ");
        strcat(hint, "-acc low stops earlier");
        printf("      (to skip them: %s)\n", hint);
    }
    if (yfile) {
        FILE *f = fopen(yfile, "w");
        if (f) {
            for (int i = 0; i < O->m; i++) fprintf(f, "%.17g\n", yo[i]);
            fclose(f);
        } else fprintf(stderr, "brisk: cannot write %s\n", yfile);
    }
    if (Xo && xfile) write_matrix_file(xfile, O, Xo);
    if (zfile || g_brisk_result) {
        /* Z = C - A'y on the original data */
        double **Zo = malloc(sizeof(double *) * (O->nblk + 1));
        for (int k = 0; k < O->nblk; k++) {
            size_t n = abs(O->bs[k]);
            /* 4.42: the library does not return the dense Z of a block that would take more
             * than 30% of the memory (as for X; dense case13659: 10 GB for n = 35 503): the
             * block is NULL, Z = C - A'y is given by y and the data. -z writes it anyway. */
            if (!zfile && O->bs[k] > 0 && 8.0 * (double)n * (double)n > 0.3 * brisk_mem_limit()) {
                Zo[k] = NULL;
                if (par.verbose >= 0) printf("Z of block %d (n = %zu) is not returned: its dense form needs %.1f GB (Z = C - A'y from y)\n", k + 1, n, 8.0 * (double)n * (double)n / 1e9);
                continue;
            }
            Zo[k] = calloc(O->bs[k] < 0 ? n : n * n, sizeof(double));
            if (!Zo[k]) { fprintf(stderr, "brisk: out of memory (Z of block %d)\n", k + 1); exit(1); }
        }
        for (size_t q = 0; q < O->nnz; q++) {
            int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
            if (!Zo[k]) continue;
            double c = O->con[q] < 0 ? O->v[q] : -yo[O->con[q]] * O->v[q];
            if (O->bs[k] < 0) Zo[k][i] += c;
            else { Zo[k][i + (size_t)j * n] += c; if (i != j) Zo[k][j + (size_t)i * n] += c; }
        }
        if (zfile) write_matrix_file(zfile, O, Zo);
        if (g_brisk_result) {
            /* 4.30 library interface: the solution of the problem as read, in memory */
            BriskResult *res = g_brisk_result;
            res->status = status; res->exit_code = exit_code[status];
            snprintf(res->status_str, sizeof(res->status_str), "%s", status_str[status]);
            res->iters = R.iters; res->pobj = pobj; res->dobj = dobj;
            res->bound_side = par.bound_side && bc->have ? par.bound_side : 0;
            res->bound_rigorous = NAN;
            res->bound_value = bc->value; res->bound_resid = bc->resid; res->bound_lammin = bc->lammin;
            res->bound_valid = bc->valid; res->bound_certified = certified;
            res->bound_rigorous = rigorous;
            res->n_resolves = g_natt;
            res->t_resolves = 0;
            for (int a = 0; a < g_natt; a++) res->t_resolves += g_att[a].t;
            snprintf(res->cause, sizeof(res->cause), "%s", cause);
            for (int e = 1; e <= 6; e++)
                res->err[e] = measured && !infeas ? pr.err[e] : (e == 4 && !infeas ? pr.err[4] : R.err[e]);
            res->time = R.t_total + tpre + pr.t + textra;
            res->m = O->m; res->nblk = O->nblk;
            res->bs = malloc(sizeof(int) * (O->nblk + 1));
            memcpy(res->bs, O->bs, sizeof(int) * O->nblk);
            res->y = malloc(sizeof(double) * (O->m + 1));
            memcpy(res->y, yo, sizeof(double) * O->m);
            res->Z = Zo; Zo = NULL;
            res->have_x = Xo != NULL;
            if (Xo) {
                res->X = malloc(sizeof(double *) * (O->nblk + 1));
                for (int k = 0; k < O->nblk; k++) {
                    const size_t n = abs(O->bs[k]), len = O->bs[k] < 0 ? n : n * n;
                    res->X[k] = malloc(sizeof(double) * (len + 1));
                    memcpy(res->X[k], Xo[k], sizeof(double) * len);
                }
            }
        }
        if (Zo) { for (int k = 0; k < O->nblk; k++) free(Zo[k]); free(Zo); }
    }
    const char *xo_env = getenv("BRISK_XOUT");
    if (Xo && xo_env) write_x_binary(xo_env, O, Xo);
    if (par.verbose > 0 && !quiet_summary) {
        double e6[7];
        for (int e = 1; e <= 6; e++) e6[e] = measured && !infeas ? pr.err[e] : (e == 4 && !infeas ? pr.err[4] : R.err[e]);
        print_summary(u, &par, O, status, e6, infeas, cause, t_all, tread, tpre, R.iters, NULL);
    }
    run_free(u);
    return exit_code[status];
}

/* 4.39: -certify-x / -certify-y: certify a given certificate (BRISK's -x / -y output, or any
 * solver's in that format) on the problem as read; no solve. Exit code 0 certified, 20 not. */
static int certify_given(const char *fname, const char *cx, const char *cy) {
    Problem P;
    memset(&P, 0, sizeof(P));
    if (g_brisk_data && fname == g_brisk_data_name) { if (problem_from_sdpa_data(g_brisk_data, &P) != 0) return 2; }
    else if (read_sdpa(fname, &P) != 0) return 2;
    PSOrig *O = ps_orig_build(&P);
    const int side = cx ? 1 : 2;
    double **X = NULL, *y = NULL;
    FILE *f = fopen(cx ? cx : cy, "r");
    if (!f) { fprintf(stderr, "brisk: cannot read %s\n", cx ? cx : cy); ps_orig_free(O); problem_free(&P); return 2; }
    if (side == 1) {
        X = bound_alloc_blocks(O);
        int k, i, j; double v;
        while (fscanf(f, "%d %d %d %lf", &k, &i, &j, &v) == 4) {
            if (k < 1 || k > O->nblk) continue;
            const int n = abs(O->bs[k - 1]);
            if (i < 1 || j < 1 || i > n || j > n) continue;
            if (i > j) { const int t = i; i = j; j = t; }
            if (O->bs[k - 1] < 0) X[k - 1][i - 1] = v;
            else { X[k - 1][(i - 1) + (size_t)(j - 1) * n] = v; X[k - 1][(j - 1) + (size_t)(i - 1) * n] = v; }
        }
    } else {
        y = calloc(O->m + 1, sizeof(double));
        for (int i = 0; i < O->m; i++) if (fscanf(f, "%lf", &y[i]) != 1) break;
    }
    fclose(f);
    const double t0 = wtime();
    double B = NAN;
    char why[160];
    const int ok = brisk_certify_orig(O, side, X, y, g_brisk_data != NULL, &B, why, sizeof(why));
    if (ok) printf("CERTIFIED (%s): rigorous %s bound %.17g on the optimal value of (P) (SDPA convention: %s bound %.17g; %.2fs)\n",
                   side == 1 ? "P" : "D", side == 1 ? "upper" : "lower", B, side == 1 ? "a lower" : "an upper", -B, wtime() - t0);
    else printf("NOT CERTIFIED (%s): %s\n", side == 1 ? "P" : "D", why);
    if (g_brisk_result) {
        BriskResult *res = g_brisk_result;
        res->status = -1; res->exit_code = ok ? 0 : 20;
        snprintf(res->status_str, sizeof(res->status_str), "%s", ok ? "CERTIFIED" : "NOT CERTIFIED");
        res->m = O->m; res->nblk = O->nblk;
        res->bs = malloc(sizeof(int) * (O->nblk + 1));
        memcpy(res->bs, O->bs, sizeof(int) * O->nblk);
        res->bound_side = side; res->bound_certified = ok; res->bound_rigorous = B;
        res->bound_value = NAN; res->bound_valid = -1;
        snprintf(res->cause, sizeof(res->cause), "%s", ok ? "" : why);
    }
    if (X) bound_free_blocks(X, O->nblk);
    free(y);
    ps_orig_free(O);
    problem_free(&P);
    return ok ? 0 : 20;
}

/* 4.37 (B2b): the first likely cause of a result that misses the tolerance */
static void diagnose_cause(const Run *u, const Params *par, const PSOrig *O, double **Xo, const double *yo,
                           const PSResult *pr, double acc, char *out, size_t len) {
    double trX = 0, ynorm = 0;
    if (Xo)
        for (int k = 0; k < O->nblk; k++) {
            const int n = abs(O->bs[k]);
            for (int i = 0; i < n; i++) trX += O->bs[k] < 0 ? Xo[k][i] : Xo[k][i + (size_t)i * n];
        }
    if (yo) for (int i = 0; i < O->m; i++) ynorm += fabs(yo[i]);
    const double nb1 = O->nb1, nC1 = O->nC1;
    const double obj = fabs(pr->pobj) > fabs(pr->dobj) ? pr->pobj : pr->dobj;
    const int tb_bind = u->Otb && u->tbs < 1e-3 * u->tbR;
    const double gap = fabs(pr->err[5]);
    const double lamz = pr->err[4] * (1 + nC1), lamx = pr->err[2] * (1 + nb1);
    if (tb_bind || (Xo && trX > 1e5 * (1 + nb1) && trX > 1e4 * (1 + fabs(obj)))) {
        snprintf(out, len, "the (P) side appears unbounded (tr X = %.1e%s): the optimum may not be attained, i.e. (D) may have no strictly feasible point. "
                 "Accuracy is limited in double precision; <C,X> is still an upper bound (-bound p; with -tracebound R each solve is well posed)",
                 trX, tb_bind ? ", the trace bound binds" : "");
        return;
    }
    if (yo && ynorm > 1e5 * (1 + nC1) && ynorm > 1e4 * (1 + fabs(obj))) {
        snprintf(out, len, "the (D) side appears unbounded (|y|_1 = %.1e): the optimum may not be attained, i.e. (P) may have no strictly feasible point. "
                 "Accuracy is limited in double precision; b'y is still a lower bound (-bound d)", ynorm);
        return;
    }
    if (Xo && trX * lamz / (1 + fabs(obj)) > 10 * fmax(gap, par->tol)) {
        snprintf(out, len, "large multipliers (tr X = %.1e) make the objective sensitive to feasibility errors of %.0e (lambda_min(Z)); the objectives agree to %.0e but may be off by ~%.0e",
                 trX, lamz, gap, trX * lamz / (1 + fabs(obj)));
        return;
    }
    if (yo && ynorm * lamx / (1 + fabs(obj)) > 10 * fmax(gap, par->tol)) {
        snprintf(out, len, "large multipliers (|y|_1 = %.1e) make the objective sensitive to feasibility errors of %.0e (lambda_min(X)); the objectives agree to %.0e but may be off by ~%.0e",
                 ynorm, lamx, gap, ynorm * lamx / (1 + fabs(obj)));
        return;
    }
    if (u->R.nreg >= 5) {
        snprintf(out, len, "ill-conditioned Newton systems (%d regularized factorizations%s)", u->R.nreg, u->R.npcg ? ", PCG refinement" : "");
        return;
    }
    if (Xo) {
        /* 4.39 (B2b, the spec's cause 3): split free variables whose two halves have both grown,
         * x+ ~ x- >> |x+ - x-|: the cancellation costs digits of the free value */
        int *P1 = NULL, *P2 = NULL, *PK = NULL;
        const int np = bound_find_pairs(O, &P1, &P2, &PK);
        double worst = 0, wf = 0, wm = 0;
        for (int q = 0; q < np; q++) {
            const double a = Xo[PK[q]][P1[q]], b2 = Xo[PK[q]][P2[q]], mn = fmin(a, b2), f = fabs(a - b2);
            const double rr = mn / (1 + f);
            if (rr > worst) { worst = rr; wf = f; wm = mn; }
        }
        free(P1); free(P2); free(PK);
        if (np > 0 && worst > 1e4 && wm > 1e3 * (1 + nb1)) {
            snprintf(out, len, "split free variables have grown (x+ and x- both ~%.1e for a difference of %.1e): the cancellation limits the accuracy of the free values; a native or eliminated free variable (-freeelim 1) avoids it", wm, wf);
            return;
        }
    }
    if (acc > par->red_acc) snprintf(out, len, "cause not identified (max rel error %.1e); try -acc high or -crossover 1", acc);
}

/* ---- 4.30 library interface -------------------------------------------------------- */
int (*brisk_print_hook)(const char *s, int is_err) = NULL;

void brisk_result_free(BriskResult *res) {
    if (!res) return;
    if (res->X) { for (int k = 0; k < res->nblk; k++) free(res->X[k]); free(res->X); }
    if (res->Z) { for (int k = 0; k < res->nblk; k++) free(res->Z[k]); free(res->Z); }
    free(res->y); free(res->bs); free(res->sx); free(res->sz);
    memset(res, 0, sizeof(*res));
    res->status = -1;
}

#ifdef BRISK_LIBRARY
#include <setjmp.h>
#include <stdarg.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif
#undef printf
#undef fprintf
#undef exit
static jmp_buf g_brisk_jmp;
static int g_brisk_jmp_on = 0;

/* exit() anywhere in the solver (an out-of-memory, an invalid option) returns to brisk_run
 * instead of ending the host process; the memory of the aborted solve is not reclaimed */
void brisk_exit(int code) {
    if (g_brisk_jmp_on) longjmp(g_brisk_jmp, code == 0 ? 1000 : code);
    exit(code);
}

static int brisk_vprint(FILE *f, const char *fmt, va_list ap) {
    if (!brisk_print_hook || (f != stdout && f != stderr)) return vfprintf(f, fmt, ap);
    char buf[2048];
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(buf, sizeof(buf), fmt, aq);
    va_end(aq);
    if (n >= (int)sizeof(buf)) {
        char *big = malloc((size_t)n + 1);
        if (!big) return n;
        vsnprintf(big, (size_t)n + 1, fmt, ap);
        brisk_print_hook(big, f == stderr);
        free(big);
    } else if (n > 0) brisk_print_hook(buf, f == stderr);
    return n;
}
int brisk_printf(const char *fmt, ...) { va_list ap; va_start(ap, fmt); int n = brisk_vprint(stdout, fmt, ap); va_end(ap); return n; }
int brisk_fprintf(FILE *f, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int n = brisk_vprint(f, fmt, ap); va_end(ap); return n; }

/* 4.37: the in-memory problem is given to brisk_main as the "file" g_brisk_data_name */
int brisk_run_data(const BriskData *d, int argc, char **argv, BriskResult *res) {
    char **av = malloc(sizeof(char *) * ((size_t)(argc > 1 ? argc : 1) + 2));
    if (!av) { memset(res, 0, sizeof(*res)); res->status = -1; return -1; }
    av[0] = "brisk";
    av[1] = (char *)g_brisk_data_name;
    for (int i = 1; i < argc; i++) av[i + 1] = argv[i];
    const int ac = (argc > 1 ? argc : 1) + 1;
    av[ac] = NULL;
    g_brisk_data = d;
    const int rc = brisk_run(ac, av, res);
    g_brisk_data = NULL;
    free(av);
    return rc;
}

int brisk_run(int argc, char **argv, BriskResult *res) {
    memset(res, 0, sizeof(*res));
    res->status = -1;
    char **const av_caller = caller_enter(&argc, argv, "c");
    if (av_caller) argv = av_caller;
#if defined(__x86_64__) || defined(__i386__)
    const unsigned int csr = _mm_getcsr();     /* the solver sets flush-to-zero: restore the host's mode */
#endif
    g_brisk_result = res;
    g_brisk_jmp_on = 1;
    volatile int rc;
    const int j = setjmp(g_brisk_jmp);
    /* 4.42: the thread count of the host process is restored after the call, also when the
     * solve left through brisk_exit (the tiny-problem and busy-machine rules lower it for a
     * solve). An explicit -threads k keeps its documented effect on later calls. */
    int nt_saved = -1;
    { int has = 0; for (int i = 1; i < argc; i++) if (argv[i] && !strcmp(argv[i], "-threads")) has = 1;
      if (!has) nt_saved = omp_get_max_threads(); }
    if (j == 0) rc = brisk_main(argc, argv);
    else rc = j == 1000 ? -1000 : -j;
    if (nt_saved > 0) omp_set_num_threads(nt_saved);
    g_brisk_jmp_on = 0;
    g_brisk_result = NULL;
    caller_leave(av_caller);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(csr);
#endif
    return rc;
}
#endif
