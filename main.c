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

/* 5.16: the density of a block's aggregate pattern: off-diagonal entries of C and the A_i, each
 * entry once, over n(n+1)/2. Until 5.15 the three "large sparse block" tests below counted an
 * entry once per constraint that touches it, so a block whose entries several constraints share
 * read as denser than it is (AC-OPF with lifted cuts, case2000_goc: past 1 %, the first-order
 * engine was tried before the chordal conversion and the solve did not finish in 600 s). */
static int cmp_u64_main(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static double blk_pattern_density(const Block *B) {
    size_t ne = B->C.nnz, off = 0;
    for (int t = -1; t < B->ncon; t++) {
        const SpSym *S = t < 0 ? &B->C : &B->A[t];
        if (t >= 0) ne += S->nnz;
        for (int q = 0; q < S->nnz; q++) off += S->row[q] != S->col[q];
    }
    const double tri = 0.5 * (double)B->n * (B->n + 1.0), d0 = (double)off / tri;
    /* the count with repetitions bounds the union from above: at or below 1 % the decisions
     * are those of 5.15 without sorting; above 5e7 entries the sort is not worth it */
    if (d0 <= 0.01 || ne > 50000000) return d0;
    uint64_t *key = (uint64_t *)malloc(sizeof(uint64_t) * (ne + 1));
    if (!key) return d0;
    size_t q0 = 0;
    for (int t = -1; t < B->ncon; t++) {
        const SpSym *S = t < 0 ? &B->C : &B->A[t];
        for (int q = 0; q < S->nnz; q++) {
            int i = S->row[q], j = S->col[q];
            if (i == j) continue;
            if (i > j) { const int tt = i; i = j; j = tt; }
            key[q0++] = ((uint64_t)i << 32) | (uint64_t)j;
        }
    }
    qsort(key, q0, sizeof(uint64_t), cmp_u64_main);
    size_t nd = 0;
    for (size_t q = 0; q < q0; q++) if (q == 0 || key[q] != key[q - 1]) nd++;
    free(key);
    return (double)nd / tri;
}

static void usage(void) {
#ifdef BRISK_LIBRARY
    fprintf(stderr, "brisk: the options are listed by the command-line solver (./brisk with no arguments)\n");
    return;
#endif
    fprintf(stderr,
        "usage: brisk problem.dat-s [options]      (an SDPA sparse file)\n"
        "       brisk problem.mat [options]        (SeDuMi format: a MAT-file with A (or At), b, c, K;\n"
        "                                           K.f, K.l, K.q, K.r, K.s; MAT versions 5 to 7)\n"
        "       brisk problem.mps [options]        (a linear program: MPS, free or fixed format; problem.mps.gz is read through gzip)\n"
        "       brisk problem.cbf[.gz] [options]   (CBF, the format of CBLIB: linear and second-order cones;\n"
        "                                           integer variables are relaxed; -x <file> writes x in the\n"
        "                                           file's own variables, one value per line)\n"
        "       brisk problem.qps [options]        (a convex quadratic program: an MPS file with a QUADOBJ,\n"
        "                                           QMATRIX or QSECTION section; QCMATRIX: quadratic constraints)\n"
        "A convex QP read from a QPS file is solved by the QP solver: by the dual active-set method of Goldfarb\n"
        "and Idnani when it is small and its quadratic term positive definite, by the QP interior-point method\n"
        "otherwise (-qpas, -qpipm: that method). A convex QCQP is solved by the SOCP solver, each quadratic\n"
        "constraint a second-order cone, and its point polished to the QCQP's own optimality conditions. A QP\n"
        "that is not convex is refused. Their options: -qpas, -qpipm, -tol, -maxit, -timelimit, -threads,\n"
        "-x <file> (x, one value per line), -q, -v.\n"
        "A linear program read from an MPS file is presolved and solved by the LP interior-point method\n"
        "(bounds kept as bounds, normal equations); when that method stops near the optimum its point goes\n"
        "to the crossover, and the SOCP solver takes over when it gives up and the crossover cannot finish\n"
        "(infeasible or unbounded problems: it returns the certificates). Its options: -tol, -maxit,\n"
        "-timelimit, -q, -v, -x <file> (x of the problem as read, one value per line), and\n"
        "  -lppresolve <k>  0 none, 1 rows and columns (fixed, singleton, redundant, forcing, dual fixing;\n"
        "                   linearly dependent equality rows),\n"
        "                   2 also substitutions: free column singletons, opposite column pairs, implied free\n"
        "                   columns by their equality rows; 3 (default) with -lpmethod 1, first parallel rows\n"
        "                   and columns, doubleton equations and slack columns, then as 2\n"
        "  -lpmethod <k>    not given (default): the LP interior-point method and, with two threads or more, the\n"
        "                   dual simplex method beside it on one thread; the first to end with a result gives it, which\n"
        "                   can then differ from run to run (-threads 1: the interior-point method alone, the same result\n"
        "                   every run); a result at reduced accuracy waits for the dual simplex method at most twice the\n"
        "                   time taken; 1 the LP interior-point method alone, 0 the SOCP solver directly (also for an\n"
        "                   LP given in SeDuMi form, from CVXPY or JuMP, which goes to the LP interior-point method first),\n"
        "                   2 the dual simplex method, 3 the primal simplex method, 4 the interior-point method\n"
        "                   followed by a crossover, 5 chosen automatically between 2 and 4 (2-5: an optimal basic solution;\n"
        "                   with 5 and two threads or more, 2 and 4 run at the same time and the first to finish gives\n"
        "                   the result, which can then differ from run to run; -threads 1 gives the same result every run)\n"
        "  -lpcorr <k>      the interior-point method's centrality correctors per iteration (-lpmethod 1, 4\n"
        "                   and 5; default: 0-3 from the cost of a factorization against a solve)\n"
        "A problem in SeDuMi format without semidefinite blocks (a linear or second-order cone program)\n"
        "is solved by the second-order cone solver: its options are -tol, -acc, -maxit, -timelimit, -q,\n"
        "-v, and -x, -y, -z (x, y and z = c - A'y of the SeDuMi form, one value per line). With\n"
        "semidefinite blocks, or with -conesolver 0, -prec, -bound, -fom, -mfipm, -lralm, the problem\n"
        "goes to the semidefinite solver (second-order cones as arrow blocks, which its self-dual embedding\n"
        "solves as second-order cones again: -soc) with all the options below. A problem whose semidefinite\n"
        "blocks are all of order 1 or 2 or have arrow data (an SOCP written in semidefinite form) is again\n"
        "solved by the SOCP solver, a block of order 2 as a second-order cone of dimension 3 (-conesolver 0: not).\n"
        "output\n"
        "  -q | -v          quiet | the diagnosis: every step of the solver (the default log shows the problem, the\n"
        "                   presolve's changes, the method, one line per iteration and the Summary)\n"
        "                   (Ctrl-C: stop and return the current point; twice: abort)\n"
        "  -y <file>        write y (SDPA primal x = -y), one value per line, original numbering\n"
        "  -x <file>        write X of the original problem (SDPA sparse-like: block i j value, upper triangle)\n"
        "  -z <file>        write Z = C - A'y of the original problem, same format\n"
        "termination\n"
        "  -acc <level>     accuracy: low (tol 1e-6), default (1e-8) or high (1e-10). low also scales the\n"
        "                   re-solve thresholds (10 tol, 100 tol after the embedding) and the reduced-\n"
        "                   accuracy class (100 tol); high keeps the default ones. -tol/-retryacc override it.\n"
        "                   For a second-order cone program, at 1e-6 or looser the solve also stops at the limit\n"
        "                   of double precision (reduced accuracy, the best point); at default and high it goes on,\n"
        "                   on the normal equations when every free variable was substituted away, otherwise in\n"
        "                   double-double arithmetic\n"
        "  -tol <t>         relative gap / infeasibility tolerance (1e-8); a looser one moves the re-solve and reduced-accuracy\n                   thresholds with it (10 t and 100 t)\n"
        "  -maxit <k>       iteration limit (100); below 100 it is the run's budget: no re-solve after it\n"
        "  -timelimit <s>   wall-clock limit in seconds (0 = none)\n"
        "  -bound p|d|pd    also return GUARANTEED bounds on the optimal value, on the sides asked for:\n"
        "                   p: a feasible X of (P) (A(X) = b to -boundtol, X > 0 with a margin), <C,X> an\n"
        "                   upper bound on the optimal value even when the solve does not converge; d: a\n"
        "                   feasible y (Z = C - A'y > 0 verified), b'y a lower bound; pd: both, in one\n"
        "                   solve. (P) is the SDPA dual (min <C,X>, C = -F0): SOS certificates from\n"
        "                   SeDuMi/GloptiPoly are -bound p; relaxations of minimisation problems take the\n"
        "                   lower bound from -bound d. Each certificate is checked rigorously on the\n"
        "                   problem as read; only a certified one is a bound. The returned point, its\n"
        "                   status and -x / -y stay the solve's own (-boundx / -boundy write the\n"
        "                   certificates). Only the sides asked for are evaluated. See README (bound mode).\n"
        "  -boundx <f>      with -bound p or pd: write the certificate X (the -x format)\n"
        "  -boundy <f>      with -bound d or pd: write the certificate y (the -y format)\n"
        "  -boundanchor <f> with -bound d: a y (the -y format, as read) whose Z = C - A'y is positive\n"
        "                   definite, e.g. from a solve of the problem with the largest margin; a\n"
        "                   candidate whose Z is not positive semidefinite is blended with it\n"
        "  -boundtol <t>    relative residual required of a -bound p certificate (1e-10)\n"
        "  -boundmargin <e> eigenvalue margin of the certificate, relative to 1 + max diagonal; at least\n"
        "                   8 n^2 u per block of order n, what a rigorous check needs (1e-13)\n"
        "  -boundtrack <t>  residual below which iterates are tracked as candidates (100 x -boundtol)\n"
        "  -boundprec <k>   the lambda_min checks of the bound mode beyond double, when only their rounding\n"
        "                   allowance fails them (about n^2 u on a unit diagonal): -1 automatic (192 bits,\n"
        "                   then 384), 0 double only, > 0 that many bits (mpx arithmetic, a-priori error bounds)\n"
        "  -boundprior <k>  when a certificate has no margin (the other side without an interior): a rigorous\n"
        "                   bound from a-priori bounds of the other side found from the problem (1, the\n"
        "                   default; 0 off): y bounded through Z(y) >= 0 (moment relaxations with a ball or\n"
        "                   box constraint), the diagonal of X through equations of positive diagonal entries;\n"
        "                   2: that bound first, on the returned point, and a certificate only when there is\n"
        "                   none (cheaper where the a-priori bounds exist; the bound is weaker by about\n"
        "                   |lambda_min| times the a-priori bound, the certificate is the returned point)\n"
        "  -certify         (implied by -bound) the rigorous check of a certificate on the problem as\n"
        "                   read: the file's decimals enclosed in intervals, directed rounding, a\n"
        "                   verified lambda_min; the rigorous bound is printed\n"
        "  -certify-x <f>   no solve: rigorously check the X in f (-x format: block i j value) as a\n"
        "  -certify-y <f>   ... p certificate, or the y in f (-y format) as a d certificate; exit code 0\n"
        "                   when certified, 20 when not (certificates of any solver in these formats)\n"
        "  -stallwin <k>    non-improving iterations tolerated once below 1e-6 (5)\n"
        "high precision\n"
        "  -prec <p>        solve in high precision: dd (double-double, about 32 digits), qd (quad-double,\n"
        "                   about 64 digits) or a number of decimal digits (variable precision; up to 31\n"
        "                   digits is dd, up to 63 qd; the tolerance is limited to 1e-290, the measured\n"
        "                   errors being double-precision numbers). The data are read exactly as written in the file.\n"
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
        "                   With -bound p|d|pd the certificate of each side is built in high precision on\n"
        "                   the problem as read and checked rigorously (in the precision of -prec plus guard\n"
        "                   limbs, with a-priori error bounds of the arithmetic); the rigorous bound is\n"
        "                   printed with all digits, and -boundx / -boundy write the certificates with all\n"
        "                   the digits of their check (-x / -y: the solution, with the digits of -prec).\n"
        "                   -certify-x / -certify-y with -prec check a given certificate\n"
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
        "  -conesolver 0    a problem in SeDuMi format without semidefinite blocks, or one whose semidefinite\n"
        "                   blocks are all of order 1 or 2 or have arrow data: by the semidefinite solver\n"
        "                   (second-order cones as arrow blocks) instead of the SOCP solver (default 1)\n"
        "  -dual            dual-scaling method (DSDP-style: potential reduction, correctors, verified\n"
        "                   certificates); falls back to the primal-dual method unless certified\n"
        "  -hsd | -nohsd    force | forbid the self-dual embedding (default: automatic - the embedding first\n                   on most problems, the infeasible start first on large dense blocks with a cheap\n                   Schur complement; each method is the other's fallback)\n  -hsdfirst <0|1>  automatic method: the embedding first when its extra dense work is small or the\n                   problem tiny (1; 0: the infeasible start first); many split free pairs and the chordal moment\n                   form use the embedding either way\n  -hsdfirstc <c>   ... i.e. when c * sum n^3 <= m^3/3 (12),\n  -hsdfirstasm <f> ... or <= f * (m^3/3 + the estimated work of the Schur assembly) (0.5; 0: off)\n  -stdslow <k>     infeasible-start method: stop for the retry when the best score has not halved in k iterations (10; 0 off)\n"
        "  -hsdx0 <f>       the self-dual embedding's primal start scaled by f (default: as computed). When the embedding\n"
        "                   ends above 1e-5 with its primal iterate at least 10 times its start's size (Frobenius norm), the\n"
        "                   second solve is the embedding again with its start at that size (the log names the factor)\n"
        "  -retryacc <a>    retry with the embedding only if the first result is worse than a (1e-7)\n"
        "  -warm <l>        re-solve from l x (first point) + (1 - l) x (standard start) (0 = off; measured slower)\n"
        "  -dir auto|hkm|nt search direction of the interior-point method, with either start (auto;\n                   -hsddir sets the self-dual embedding's alone)\n"
        "  -zsparse <n>     HKM direction: blocks of order >= n whose dual matrix Z keeps a sparse\n"
        "                   pattern (max-cut, structural designs) use a sparse Cholesky factor of Z for\n"
        "                   Z^{-1}, its products and the dual step length; X stays dense (200; 0 off)\n"
        "  -zsparsefill <f> ... when the factor has at most f n^2 nonzeros (0.1)\n"
        "  -ntfast <k>      NT: form dX on the pattern of dZ (1) or by dense transforms (0, default)\n"
        "  -nodd            no double-double endgame (nor the end phase on the augmented Newton system)\n"
        "  -ddfactor <f>    endgame work budget as a multiple of the solve's work (1.0); also the time of the end\n"
        "                   phase on the augmented Newton system, where the double-double endgame is too large\n"
        "  -ddminwork <w>   ... but at least this many flop-equivalents (1e11)\n"
        "  -ddmaxm <k>      size limit for the endgame (1200)\n  -ddmaxn2 <s>     ... and the largest sum of the squares of the block orders (0: from the memory limit)\n"
        "  -ddit <k>        endgame iteration limit (40)\n"
        "  -nopolish        no polishing: the last iterate is returned as it is (not the best one), without\n"
        "                   the corrections at the end and the polish probes of the embedding (-nopolishx:\n"
        "                   also the X-metric probe; -nodd: also the double-double endgame)\n  -crossover <k>   SDP crossover after the solve (Newton on the rank-revealed KKT system; kept only if\n                   the errors on the original data fall): 0 off (default), 1 on, -1 when cheap\n"
        "presolve\n"
        "  -nofr            no facial-reduction presolve\n  -frdepth <k>     facial reduction cut at depth k: the reductions resting on k earlier ones are not made\n                   (default: no limit; 0: none, as -nofr); a shallower reduction keeps more of the dual (the\n                   multipliers, e.g. the moments of a moment relaxation) recoverable\n  -fr              facial reduction whatever its depth (default: skipped when of depth >= 2 and saving < frgain)\n  -frgain <g>      per-iteration saving that justifies a facial reduction of depth >= 2 (4)\n"
        "  -soc <0|1>       semidefinite blocks whose data are arrow matrices (second-order cones written\n"
        "                   as matrix inequalities; the SeDuMi reader writes cones so next to semidefinite\n"
        "                   blocks) solved as second-order cones in the self-dual embedding (1)\n"
        "  -socmin <n>      ... blocks of at least this order (3)\n"
        "  -nofrretry       no re-solve when the removed duals are not recovered and the reduced solve misses the tolerance\n"
        "  -frdual <k>      1: when the facial reduction's duals cannot be rebuilt, a second solve finds them on the problem as\n                   read: the dual of least violation whose objective equals the optimal value of the first solve (X and the\n                   value are kept; about one solve of the problem as read; 0, the default: the log says that the dual is lost)\n"
        "  -frfallback <f>  removed duals not rebuilt though the reduced solve is good: with f > 0 the problem is solved again\n                   without facial reduction, within f times the solve's time (0, the default: no re-solve; the reduced\n                   solve is returned as SOLVED TO REDUCED ACCURACY, the log says that the recovery failed, what is\n                   correct, and how to solve again without the reduction for the dual)\n"
        "  -symfallback <f> a solve after an automorphism reduction that ends worse than reduced accuracy is repeated\n                   without the reduction, within this multiple of the solve's time (4; 0 = never)\n"
        "  -deprows <k>     remove the equality constraints that are linear combinations of others (their dual values\n"
        "                   are 0): 1 on (default), 0 off\n"
        "  -freeelim <k>    eliminate split free-variable pairs by pivoting: -1 auto (kernel-form SOS\n"
        "                   problems: few SDP blocks, pairs >= 1%% of the rows), 0 off, 1 on\n"
        "  -tracebound <R>  with the elimination: the row sum tr(X) + s = R that keeps the Gram side\n"
        "                   bounded (-1 auto: 1e7 (1 + max|b|), 0 off)\n"
        "  -dualize <k>     solve the dual form (Z + sum y_i A_i = C entrywise, y eliminated) when its\n"
        "                   Schur complement is at most half the size and the problem has an objective\n"
        "                   (not for a feasibility problem): -1 auto, 0 off, 1 force\n"
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
        "                   every constraint has an entry in an LP block and each LP variable is in\n"
        "                   one constraint at most: a diagonal LP part), and that the presolve\n"
        "                   leaves unchanged; not on moment relaxations (see -hsddir). The attempt\n"
        "                   ends when a solve away from the\n"
        "                   optimum needs more CG steps than a sixth of a Schur factorization costs,\n"
        "                   or the steps of a solve grow geometrically (three times the last\n"
        "                   iteration's, or the next solve past that bound at the same growth),\n"
        "                   or the attempt more than a quarter of the solve with the factored Schur\n"
        "                   complement; unless it ends OPTIMAL, the interior-point method with the\n"
        "                   factored Schur complement follows (from the attempt's best iterate\n"
        "                   between merits 1e-3 and 1e-5, if it got below 1e-3); also when the Schur complement does\n"
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
        "  -bm <k>          the Burer-Monteiro solver (X = R R', nothing of size n^2 formed) for large sparse\n"
        "                   SDPs with low-rank solutions, one semidefinite block (and LP blocks):\n"
        "                   1 always, 0 never, -1 (default) automatically when the problem has one block of\n"
        "                   order n >= 500 and every constraint fixes a diagonal entry, once the linearly\n"
        "                   dependent constraints are removed (max-cut and its\n"
        "                   relatives; then coordinate steps on the spheres and a Riemannian trust region,\n"
        "                   stopped on a certified bound; the interior-point method follows if it misses),\n"
        "                   or when every constraint fixes one entry (matrix completion) and the dense\n"
        "                   Schur complement does not fit in memory (the first-order engine if it misses).\n"
        "                   Other constraints: an augmented Lagrangian with L-BFGS and an exact line search.\n"
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
        "                   first-order engine runs first for the fraction f of that time (default 0: off;\n"
        "                   it lost wherever it ran on the sets, e.g. 0.2);\n"
        "                   the interior-point method follows if it has not reached the tolerance\n"
        "  -fombound <0|1>  constraints g X_ij + a s = b whose extra variable s >= 0 appears in no\n"
        "                   other constraint are bounds on single entries of X (doubly nonnegative\n"
        "                   programs: k-means and theta-plus relaxations). With 1 (the default) the\n"
        "                   first-order engine solves the problem without them, with the bounds as a\n"
        "                   cone of their own on X; when at least half the constraints are such bounds\n"
        "                   and there are 4,000 or more, it runs first, within an iteration cap of about\n"
        "                   twice the interior-point method's estimated time, and the interior-point method\n"
        "                   follows if it has not reached the tolerance (not with -acc high). 0: off\n"
        "  -fomstart-x <f>  start the first-order engine from the X in f (the -x format) ...\n"
        "  -fomstart-y <f>  ... and the y in f (the -y format): continuing a long run (needs -nosym\n"
        "                   when a symmetry was found, and a problem the presolve keeps as read)\n"
        "  -fomhalpern <k>  ... Halpern-restarted Peaceman-Rachford (1) or ADMM with step 1.618 (0)\n"
        "  -fommaxit <k>    ... iteration limit (100000)\n"
        "  -fomtol <t>      ... stopping tolerance on the residuals (max(tol, 1e-6))\n"
        "  -fomsigma <s>    ... fixed penalty parameter (0: adaptive)\n"
        "  -fomsigma0 <s>   ... starting penalty when adaptive (1); -fomaasafe <f>: an Anderson step\n"
        "                   is rejected when its residual exceeds f times the last (2); -fomaa <k>:\n"
        "                   Anderson memory (25, up to 32,\n"
        "                   capped by a 1.5 GB history budget); -fomaadr <k>: the acceleration on the\n"
        "                   Douglas-Rachford variable X/sigma + A*y - C, step 1 (1), or on (X, Z), step 1.618 (0)\n"
        "  -sym auto|none|<file>  exact symmetry reduction (default auto): sign symmetries and\n"
        "                   permutation automorphisms of the data are found, the constraints aggregated over\n"
        "                   the orbits and the blocks split by irreducible representation; the point is\n"
        "                   mapped back to the file. <file>: generators as index permutations (line 1: SDP\n"
        "                   block sizes; one per line, global 0-based SDP indices). -nosym = -sym none\n"
        "  -symtime <s>, -symnodes <k>   the automatic search's budget: at least s (20 s) and 1 %% of the\n"
        "                   expected solve up to 900 s, at most 15 %% of it; counted in operations, not on the\n"
        "                   clock; 500 nodes; an incomplete search only reduces less, never wrongly\n"
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
        "                   1.3 s at n = 1860)\n"
        "  -blkscale <k>    diagonal congruence scaling of the semidefinite blocks, X = D X' D with D from the\n"
        "                   data in powers of two: 2 by the size of the data at each index (D = 1 / the largest\n"
        "                   |coefficient| there, within 2^-20..2^20), 1 by Ruiz iterations; -2, -1 the same when\n"
        "                   a block's factors spread by 4 or more; 0 off (the default). When the scaled solve\n"
        "                   misses the tolerance the problem is solved as given and the better answer kept\n"        "  -chordal <k>     chordal decomposition: -1 auto, 0 off, 1 force\n"
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
        "  -cls <k>         entry-class Schur route (entries in the same constraints with the same\n"
        "                   values grouped: localizing matrices): -1 auto, 0 off, 1 force\n"
        "  -mixed <k>       single-precision Schur factor + PCG: -1 auto (m>=1000), 0 off, 1 on\n"
        "  -route <r>       force Schur route of sparse constraints: 0 sparse-sparse, 1 row-product, 3 row-product on the union pattern\n"
        "  -threads <k>     OpenMP threads for this solve: k >= 1 forces k (no automatic reduction);\n"
        "                   0 or absent: automatic (OMP_NUM_THREADS, else all cores within the container's\n"
        "                   CPU quota (cgroup cpu.max, Linux); problems too small\n"
        "                   for threads run on one; on Linux fewer while other processes occupy the\n"
        "                   cores, BRISK_NOBUSY=1 keeps the count). For this solve only: a later solve of\n"
        "                   the same process starts again from the automatic count. The log states it\n"
        "                   (\"Number of threads: k\", \"(forced by user)\", \"(problem too small for multithreading)\")\n"
        "  -parblocks <k>   threads over blocks: -1 auto (several blocks, all n <= 256), 0 off, 1 on\n"
        "  -cblas <c>       routing cost of a BLAS-3 flop relative to a sparse flop (0.05)\n"
        "  -densemem <MB>   memory for dense constraint copies and the low-rank routes' products (768)\n"
        "  -pivtol <p>      squared-pivot threshold of the Schur factor (1e-14)\n"
        "  -regill <r>      fixed lowest regularization when it is violated (0: automatic: from m times\n"
        "                   the unit roundoff, raised after short steps)\n"
        "iteration details\n"
        "  -gmax <g>        step-length factor cap (0.99)\n"
        "  -lanczos <k>     max Lanczos steps for step lengths (30)\n"
        "  -sigma <k>       centering rule: 0 SDPT3 adaptive, 1 Mehrotra cubic, 2 cubic+safeguard\n"
        "  -sigmamin <s>    floor for sigma (0)\n"
        "  -corr <k>        max centrality correctors per iteration (0)\n"
        "  -init <k>        start: 0 SDPT3-style, 1 least-squares+shift, 2 +balancing, 3 the centred start\n"
        "                   0.3 I, I, y = 0 (1; 3 automatically on moment relaxations, see -hsddir)\n"
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
        "  -hsddir <k>      self-dual embedding direction: 0 HKM, 1 NT (auto: NT on the chordal moment form\n"
        "                   and, with the centred start -init 3, on moment relaxations as given (the\n"
        "                   largest block a moment matrix, a localizing block, no LP part, free\n"
        "                   variables or symmetry reduction); HKM otherwise)\n"
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
        "  -vv | -vvv       per-iteration internals | everything\n"
        "  -knownfeas       a solution is known to exist: no infeasibility exits\n"
        "  -nopolishr       no restoration of the primal feasibility of a point that missed the tolerance\n"
        "                   (the log's \"square-root-metric polishing\"; the other corrections stay)\n"
        "                   (the correction in the square-root metric of X; the older polishes stay)\n"
        "  -nopolishx       no X-metric polish of the returned point and no X-metric probe during the\n"
        "                   iterations (the log's \"X-metric polishing\"; the Euclidean polish stays)\n"
        "  -mixedfrac <f>   float Schur factor for 1000 <= m < 4000 when the factorization is at least\n"
        "                   this fraction of an iteration's work (0.4)\n"
        "  -freefill <f>    free elimination: no pivot whose Markowitz count exceeds this (1e4)\n"
        "  -chordalform <k> 1 moment form (shared entries; default), 0 overlap equalities\n"
        "  -chordalsup <k>  constraint supports up to this size join the pattern as cliques (-1 auto: tried\n"
        "                   at 8, 4, 12, 0; 0 off)\n"
        "  -chordalmerge <f> merge a clique into its parent when the separator is at least this fraction (0.6)\n"
        "  -chordaldens <f> blocks denser than this are not tried (0.3)\n"
        "  -ddtime <s>      double-double endgame: time cap in seconds (0 = none, the default; the budget\n"
        "                   is the work of -ddfactor / -ddminwork); -ddbudget <n> is accepted and has no effect\n"
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
        "           13 iteration limit, 14 numerical difficulties, 15 time limit, 16 solved but a file of the\n"
        "           solution (-x, -y, -z) not written completely (a full disk: said in the Summary), 1 usage,\n"
        "           2 input error\n");
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
    FILE *f = brisk_out_open(fname);
    if (!f) return;
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
    brisk_out_close(f, fname);
}

/* binary X for the benchmark harness (BRISK_XOUT): int nblk, int 1, int 0; per block
 * int type, int n, then n (LP) or n(n+1)/2 values (lower triangle by columns) */
static void write_x_binary(const char *fname, const PSOrig *O, double **X) {
    FILE *f = fopen(fname, "wb");   /* (binary: closed through brisk_out_close) */
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
    brisk_out_close(f, fname);
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
    BlkScale *aff;          /* 5.14: the semidefinite blocks in scaled coordinates (blkscale.c; an option) */
    MomAff *maff;           /* 5.12: a moment relaxation over an ellipsoid, solved in the coordinates of the unit ball */
    double aff_tol0;        /* ... the tolerance asked for (the solve in those coordinates runs to a tighter one) */
    Params *aff_par;
    PSOrig *Oaff;           /* ... and the problem as read then (everything else is of the scaled problem) */
    PSAlg *alg;
    PSOrig *Ofin;           /* the problem (Xo, yo) refer to after run_finish: one of O, Otb, Ofile, Odual (not owned) */
    int nXo;                /* blocks of Xo */
    Result R;
    PSResult pr;
    double *y, **X, **Xo, *yo;
    int *xc_p, *xc_o;       /* 23 October 2026: the blocks of X (xc_p, the problem solved) and of Xo (xc_o, the problem as read)
                             * held as cone vectors: the hub, or -1 (brisk.h, xc_index); NULL: none */
    int status, measured, fr_removed;
    int fr_faces;           /* 23 October 2026: of fr_removed, the faces of a lifted relaxation (lifted_faces), not constraints of the file */
    /* 23 October 2026: run_pipeline's own re-solve with the second-order cones as semidefinite blocks (both runs, which
     * one this Run holds): the caller's entry in the Summary lists the two runs under its name, once each */
    int soc2_done, soc2_kept;
    double soc2_t1, soc2_acc1, soc2_t2, soc2_acc2;
    int dep_removed;        /* 20 October 2026: linearly dependent equality constraints removed in the presolve */
    int nfree_read;         /* 20 October 2026: free variables of the problem as read (split pairs), for the Summary when they are not eliminated */
    double acc, tread, tpre;
    double t_run0;          /* 26 October 2026: the clock at the end of the reading: the Summary's total is the wall time from here */
    BoundCert cert[2];      /* 4.37 bound mode: the best certificate of this run per side ([0] p, [1] d), on the problem as read */
    int fom_check;          /* 4.39: the first-order decision deferred until after the dual form and free elimination */
    int fom_chordal;        /* 4.40: ... until after the chordal conversion (a very sparse large block) */
    double fr_cost;      /* per-iteration cost of the problem as read over that of the reduced problem (the model of the fr_gain rule) */
} Run;

/* 23 October 2026: whether the trace bound changed the answer. The bound row adds R y_R to the value; of that,
 * s y_R is the row's complementarity (it vanishes at an optimum where the bound is slack) and (R - s) y_R = tr(X) y_R
 * moves the value: the bound binds when its slack is below a thousandth of R or when that shift, relative to the
 * objectives, exceeds the tolerance and a tenth of the error measured on the problem as read (roa_univar_d16_I,
 * occupation measures, SDP: slack 1.005e-3 of R, shift 8.4e-4, the whole gap error; until then slack below 1e-3 R only) */
static int tb_binds(const Run *u, double acc, double tol) {
    if (!u->Otb || !(u->tbR > 0)) return 0;
    if (u->tbs < 1e-3 * u->tbR) return 1;
    if (getenv("BRISK_TBSLACKONLY")) return 0;
    const double shift = fabs(u->tbw) * fmax(0.0, u->tbR - u->tbs) / (1.0 + fabs(u->pr.pobj) + fabs(u->pr.dobj));
    return shift > tol && shift > 0.1 * acc;
}

static void run_free(Run *u) {
    if (u->Xo) { for (int k = 0; k < u->nXo; k++) free(u->Xo[k]); free(u->Xo); }
    free(u->xc_p); free(u->xc_o);
    if (u->X) { for (int k = 0; k < u->P.nblk; k++) free(u->X[k]); free(u->X); }
    free(u->y); free(u->yo);
    ps_free(u->P.ps); u->P.ps = NULL;
    ps_orig_free(u->O);
    if (u->Ofile) ps_orig_free(u->Ofile);
    if (u->Otb) ps_orig_free(u->Otb);
    if (u->Odual) ps_orig_free(u->Odual);
    dualize_free(u->dual);
    if (u->Oaff) ps_orig_free(u->Oaff);
    if (u->aff_par) { u->aff_par->tol = u->aff_tol0; u->aff_par = NULL; }
    momaff_free(u->maff); u->maff = NULL;
    blkscale_free(u->aff); u->aff = NULL; u->Oaff = NULL;
    if (u->Osym) ps_orig_free(u->Osym);
    sym_free(u->sym);
    if (u->Osign) ps_orig_free(u->Osign);
    sign_free(u->sign);
    if (u->Oalg) ps_orig_free(u->Oalg);
    alg_free(u->alg);
    free_elim_free(u->fe);
    bound_cert_free(&u->cert[0]); bound_cert_free(&u->cert[1]);
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
/* 24 October 2026 (ISSUES 284): the seconds the run has left, for the postsolve (INFINITY without a limit) */
static double ps_time_left(const Params *par) { return par->timelimit > 0 ? par->timelimit - (wtime() - par->t_start) : INFINITY; }
static PSOrig *map_back(Run *u, const Params *par, double ***pXo, double **pyo, int *pnXo, int main_res, const int *xc) {
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
        const double tfe0 = wtime();
        free_elim_post(u->fe, u->Ofile, Xo, yo, yf, xc);
        if (getenv("BRISK_SETUPT")) printf("   [finish: free elimination undone %.3fs]\n", wtime() - tfe0);
        if (Xo) free_elim_refine(u->fe, u->Ofile, Xo, main_res ? par->verbose : 0, xc);
        if (getenv("BRISK_SETUPT")) printf("   [finish: free values refined %.3fs]\n", wtime() - tfe0);
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
    if (u->aff || u->maff) {
        /* 5.14: from the scaled coordinates (5.12: of the unit ball) back to those of the file (the blocks and the rows are the same) */
        double **Xa = NULL, *ya = calloc(u->Oaff->m + 1, sizeof(double));
        if (u->maff) momaff_unmap(u->maff, Xo, yo, Xo ? &Xa : NULL, ya);
        else blkscale_unmap(u->aff, Xo, yo, Xo ? &Xa : NULL, ya);
        if (Xo) { for (int k = 0; k < *pnXo; k++) free(Xo[k]); free(Xo); }
        free(yo);
        Xo = Xa; yo = ya; *pnXo = u->Oaff->nblk;
        O = u->Oaff;
    }
    *pXo = Xo; *pyo = yo;
    return O;
}

/* 23 October 2026: which blocks of this run's X are held as cone vectors (brisk.h, xc_index), or NULL. A second-order
 * cone written as a semidefinite block with arrow data (as sedumi.c writes SeDuMi's K.q, and users write norm
 * constraints) is solved as a cone, but its dense completion made the postsolve, the measures and the result passes over
 * n^2 entries: the owner's dg3d_1000 (SDP with SOCP, one cone of order 17,012) spent 92 of its 300 s after the
 * iterations, at 9.2 GB. The vector is kept where nothing between the solve and the problem as read maps the block: the
 * interior-point route that solves it as a cone, no reduction that maps blocks (the symmetry and sign reductions, the
 * *-algebra, the dual form, the free-variable elimination, the trace bound, the scaling, a chordal or moment-form
 * conversion), the block kept as read by the presolve (its basis the identity) with arrow data on the problem as read,
 * and no bound mode or crossover (which read X densely). The writers and the library complete the block when they
 * need the matrix (xc_expand). BRISK_XCONE=0: dense as before. */
static int *xcone_decide(const Run *u, const Params *par, const Problem *P) {
    static int off = -1; if (off < 0) { const char *e = getenv("BRISK_XCONE"); off = e && atoi(e) == 0; }
    if (off || P->m <= 0 || !P->ps || !u->O) return NULL;
    if (!par->soc || par->dual || par->mfipm || par->fom || par->lralm || par->bound_side || par->btrack || par->warm_X || par->hsd == 0
        || par->crossover || par->hp_kind) return NULL;
    if (u->Otb || u->dual || u->alg || u->sym || u->sign || u->aff || u->maff) return NULL;
    /* (26 October 2026, ISSUES 292: the free-variable elimination maps the free values and the LP blocks only; its map
     * back reads a cone-vector block's entries from the vector, free_elim_post. The block must have arrow data with the
     * same hub on the problem as read too) */
    const PSOrig *Of = u->fe ? u->Ofile : NULL;
    if (u->fe && (!Of || Of->nblk != u->O->nblk)) return NULL;
    const Postsolve *ps = P->ps;
    if (ps->mom || ps->chordal || ps->nblk_after != P->nblk || !ps->cur2orig || !ps->basis) return NULL;
    int *xc = NULL, *ohub = NULL, *fhub = NULL;      /* (the hubs of the data, all blocks in one pass when first needed) */
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_SDP || B->n < par->soc_minn || B->n < 2) continue;
        const int h = brisk_arrow_hub(B);
        if (h < 0) continue;
        const int o = ps->cur2orig[k];
        if (o < 0 || o >= u->O->nblk || !ps->basis[o].identity || u->O->bs[o] != B->n) continue;
        if (!ohub) {
            ohub = malloc(sizeof(int) * ((size_t)u->O->nblk + 1));
            if (!ohub || ps_arrow_hubs(u->O, NULL, ohub)) { free(ohub); free(xc); return NULL; }
        }
        if (ohub[o] != h) continue;
        if (Of) {
            if (Of->bs[o] != B->n) continue;
            if (!fhub) {
                fhub = malloc(sizeof(int) * ((size_t)Of->nblk + 1));
                if (!fhub || ps_arrow_hubs(Of, NULL, fhub)) { free(fhub); free(ohub); free(xc); return NULL; }
            }
            if (fhub[o] != h) continue;
        }
        if (!xc) { xc = malloc(sizeof(int) * (P->nblk + 1)); for (int q = 0; q < P->nblk; q++) xc[q] = -1; }
        xc[k] = h;
    }
    free(ohub); free(fhub);
    return xc;
}

/* the cone-vector blocks of Xo on O: u->xc_o when O is the problem they were made for (u->O, or the problem as read after
 * the free elimination's map back), else NULL */
static int *xc_on(Run *u, const PSOrig *O) { return (O == u->O || (u->fe && O == u->Ofile)) ? u->xc_o : NULL; }

static void run_finish(Run *u, const Params *par) {
    if (!u->Otb && !u->fe && !u->dual && !u->alg && !u->sym && !u->sign && !u->aff && !u->maff) return;
    const int infeas = u->status == ST_PINFEAS || u->status == ST_DINFEAS;
    const double tf0 = wtime(); const int setupt = getenv("BRISK_SETUPT") != NULL;
    PSOrig *O = map_back(u, par, &u->Xo, &u->yo, &u->nXo, 1, u->xc_o);
    if (setupt) printf("   [finish: map back %.3fs]\n", wtime() - tf0);
    double **Xo = u->Xo, *yo = u->yo;
    u->Ofin = O;
    if (infeas || !u->measured || !Xo) return;
    PSResult pr;
    ps_measure_xc(O, Xo, yo, &pr, u->xc_o);
    if (setupt) printf("   [finish: errors on the problem as read %.3fs]\n", wtime() - tf0);
    if (getenv("BRISK_FEDBG")) printf("   [finish: reduced pobj %.12e dobj %.12e gap %.2e | file pobj %.12e dobj %.12e gap %.2e]\n",
                                       u->pr.pobj, u->pr.dobj, u->pr.err[5], pr.pobj, pr.dobj, pr.err[5]);
    pr.recovered = u->pr.recovered; pr.unrecovered = u->pr.unrecovered; pr.fr_corrected = u->pr.fr_corrected; pr.t += u->pr.t;
    u->pr = pr;
    const double acc = brisk_acc_err(pr.err);
    int status = u->status;
    if (status == ST_OPTIMAL || status == ST_REDUCED || status == ST_MAXIT || status == ST_NUMERIC || status == ST_TIME) {
        if (acc <= par->tol) status = ST_OPTIMAL;
        else if (acc <= par->red_acc) status = ST_REDUCED;
        else if (status == ST_OPTIMAL || status == ST_REDUCED) status = ST_NUMERIC;
    }
    if (par->verbose > 0 && (u->fe || u->dual || u->alg || u->sym || u->sign || u->aff || u->maff))
        printf("   (errors on the file data: %.1e %.1e %.1e %.1e %.1e %.1e; reduced problem %.1e)\n",
               pr.err[1], pr.err[2], pr.err[3], pr.err[4], pr.err[5], pr.err[6], u->acc);
    u->status = status;
    u->acc = acc;
}

/* 24 October 2026 (ISSUES 266): the solve chooses among the points of its end (the method's iterate, the polishes, the end
 * phase on the augmented Newton system) by their errors on the problem it solved; the run is judged on the problem as read,
 * and the reductions between the two can rank them the other way. The duals the facial reduction removed are rebuilt from
 * the reduced dual slack, which needs a margin that a more accurate reduced point can lack (acopf_case118_ieee__api_WSLJGC,
 * SDP with SOCP: the end phase took the reduced errors from 1.7e-8 to 8.5e-9, those on the problem as read from 3.2e-7 to
 * 7.4e-6, the removed duals lost). When the returned point misses the tolerance on the problem as read, the solve's other
 * point (brisk_alt_take) goes through the same postsolve and the same map back, and the better of the two by the run's
 * measure is returned: one postsolve more, only for a run that missed the tolerance and had a second point. */
static void alt_judge(Run *u, const Params *par) {
    Result Ra; double *ya = NULL, **Xa = NULL; const char *what = "";
    if (!brisk_alt_take(&Ra, &ya, &Xa, &what)) return;
    Problem *P = &u->P;
    Result *R = &u->R;
    int used = 0;
    if (u->measured && u->X && u->Xo && u->status != ST_OPTIMAL && u->status != ST_PINFEAS && u->status != ST_DINFEAS) {
        double *yo2 = calloc(u->O->m + 1, sizeof(double)), **Xo2 = NULL;
        int *xc2 = u->xc_p ? malloc(sizeof(int) * (u->O->nblk + 1)) : NULL;
        PSResult pr2;
        postsolve_set_time_left(ps_time_left(par)); postsolve_set_tol(par->tol);
        postsolve(P, u->O, Xa, ya, &Xo2, yo2, &pr2, par->verbose > 1 ? par->verbose : -1, u->xc_p, xc2);
        if (xc2) { int any = 0; for (int o = 0; o < u->O->nblk; o++) any |= xc2[o] >= 0; if (!any) { free(xc2); xc2 = NULL; } }
        int nXo2 = u->O->nblk;
        PSOrig *Of = u->O;
        const double tbw2 = u->Otb ? yo2[u->O->m - 1] : 0, tbs2 = u->Otb && Xo2 ? Xo2[u->O->nblk - 1][0] : 0;
        if (pr2.have_x && Xo2 && (u->Otb || u->fe || u->dual || u->alg || u->sym || u->sign || u->aff || u->maff)) {
            Of = map_back(u, par, &Xo2, &yo2, &nXo2, 0, xc2);   /* (as run_finish maps the returned point) */
            PSResult prf;
            ps_measure_xc(Of, Xo2, yo2, &prf, xc2);
            prf.recovered = pr2.recovered; prf.unrecovered = pr2.unrecovered; prf.fr_corrected = pr2.fr_corrected; prf.t += pr2.t;
            pr2 = prf;
        }
        const double acc2 = pr2.have_x && Xo2 ? brisk_acc_err(pr2.err) : INFINITY;
        used = acc2 < u->acc;
        if (par->verbose > (used ? 0 : 1))
            printf("   the solve's second point (%s): %.1e on the problem as read against %.1e, %s\n",
                   what, acc2, u->acc, used ? "returned" : "not returned");
        if (used) {
            brisk_note_diag("%s returned instead: %.1e on the problem as read, against %.1e", what, acc2, u->acc);
            if (u->Xo) { for (int k = 0; k < u->nXo; k++) free(u->Xo[k]); free(u->Xo); }
            u->Xo = Xo2; u->nXo = nXo2; Xo2 = NULL;
            free(u->yo); u->yo = yo2; yo2 = NULL;
            free(u->xc_o); u->xc_o = xc2; xc2 = NULL;
            u->pr = pr2; u->Ofin = Of;
            if (u->Otb) { u->tbw = tbw2; u->tbs = tbs2; }
            for (int k = 0; k < P->nblk; k++) free(u->X[k]);
            free(u->X); u->X = Xa; Xa = NULL;
            free(u->y); u->y = ya; ya = NULL;
            /* the solve's measures of the returned point (its times and counts stay) */
            R->pobj = Ra.pobj; R->dobj = Ra.dobj; R->pinf = Ra.pinf; R->dinf = Ra.dinf; R->relgap = Ra.relgap; R->relcomp = Ra.relcomp;
            memcpy(R->err, Ra.err, sizeof(R->err));
            int status = u->status;
            if (status == ST_OPTIMAL || status == ST_REDUCED || status == ST_MAXIT || status == ST_NUMERIC || status == ST_TIME) {
                if (acc2 <= par->tol) status = ST_OPTIMAL;
                else if (acc2 <= par->red_acc) status = ST_REDUCED;
                else if (status == ST_OPTIMAL || status == ST_REDUCED) status = ST_NUMERIC;
            }
            u->status = status; u->acc = acc2;
        }
        if (Xo2) { for (int k = 0; k < nXo2; k++) free(Xo2[k]); free(Xo2); }
        free(yo2); free(xc2);
    }
    if (!used) { if (Xa) { for (int k = 0; k < P->nblk; k++) free(Xa[k]); free(Xa); } free(ya); }
}

/* 27 October 2026: the method's iterate before the restoration and the polishes (brisk_bpt_take), postsolved and mapped
 * as the returned point, for the a-priori bound of side p: NULL when there is none or it maps to another problem than
 * the returned point's (the bound compares candidates on one problem) */
static double **bpt_mapped(Run *u, const Params *par, const PSOrig *Op, int *pn) {
    double *ya = NULL, **Xa = NULL;
    *pn = 0;
    if (!brisk_bpt_take(&ya, &Xa)) return NULL;
    Problem *P = &u->P;
    double **Xo2 = NULL, *yo2 = calloc(u->O->m + 1, sizeof(double));
    int *xc2 = u->xc_p ? malloc(sizeof(int) * (u->O->nblk + 1)) : NULL;
    PSResult pr2;
    postsolve_set_time_left(ps_time_left(par)); postsolve_set_tol(par->tol);
    postsolve(P, u->O, Xa, ya, &Xo2, yo2, &pr2, -1, u->xc_p, xc2);
    for (int k = 0; k < P->nblk; k++) free(Xa[k]);
    free(Xa); free(ya);
    int n = u->O->nblk;
    const PSOrig *Of = u->O;
    if (pr2.have_x && Xo2 && (u->Otb || u->fe || u->dual || u->alg || u->sym || u->sign || u->aff || u->maff))
        Of = map_back(u, par, &Xo2, &yo2, &n, 0, xc2);   /* (as run_finish maps the returned point, with the cone map) */
    free(yo2); free(xc2);
    if (!pr2.have_x || !Xo2 || Of != Op) { if (Xo2) { for (int k = 0; k < n; k++) free(Xo2[k]); free(Xo2); } return NULL; }
    *pn = n;
    return Xo2;
}

/* <C,X> on the problem as read, and the relative residual ||A(X) - b|| / (1 + ||b||_1) of the certificates */
static double orig_cx(const PSOrig *O, double **X, double *resid) {
    double cx = 0, *ax = calloc(O->m + 1, sizeof(double));
    for (size_t q = 0; q < O->nnz; q++) {
        const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
        const double x = O->bs[k] < 0 ? X[k][i] : X[k][i + (size_t)j * n];
        const double c = (O->bs[k] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
        if (O->con[q] < 0) cx += c; else if (ax) ax[O->con[q]] += c;
    }
    double r2 = 0, b1 = 0;
    for (int i = 0; ax && i < O->m; i++) { const double v = ax[i] - O->b[i]; r2 += v * v; b1 += fabs(O->b[i]); }
    *resid = ax ? sqrt(r2) / (1 + b1) : NAN;
    free(ax);
    return cx;
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
    int same = !u->aff && !u->maff && !u->alg && !u->sym && !u->sign && !u->dual && !u->fe && u->tbR == 0 && P->ps && P->ps->nrec == 0 && !P->ps->chordal && O && O->m == P->m && O->nblk == P->nblk;
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
static int g_pre_logged = 0; /* ISSUES 725: the problem and its presolve printed before the iterations (once a run) */
static int g_dup_inexact_any(void) { extern int g_dup_inexact; return g_dup_inexact; }
static int g_infeas_unconf = 0;  /* (audit A2) an infeasibility verdict without a certificate: 1, 2 when the facial reduction removed constraints */
static void summary_size(char *b, size_t n, const PSOrig *O, int nfree);
static void print_preprocessing(const Run *u, const char *lab, const char *cont);
static int lowrank_used(const Problem *PP, long *ncl, long *rtot);
static int run_pipeline_core(const char *fname, Params *par, int do_fr, int force_route, Run *u);
static double g_bs_probe = 0;   /* the spread of the factors -blkscale -2 would apply (blkscale_probe; the advice) */
static double g_mf_hand_t = -1;   /* the matrix-free attempt that handed its iterate to the factored method: its time (-1: none) */
static int g_mf_hand_kept = 0;                        /* and the factored solve's result from that iterate is the one returned */
/* 23 October 2026: the solve that writes the run's X, always in the layout the postsolve reads it in (u->xc_p: the blocks
 * held as cone vectors). par->xcone is set for the call only: it points into the run, which run_free releases. Until now
 * only the default path set it, and the calls after the matrix-free method read the field as params_default had left it,
 * unset: tru11 (POEMA, SDP) ended in a segmentation fault in 2 of 4 runs after the matrix-free method's hand-over */
static void run_solve(Problem *P, Params *par, Result *R, Run *u) {
    par->xcone = u->xc_p;
    brisk_solve(P, par, R, u->y, u->X);
    par->xcone = NULL;
}
/* 4 November 2026 (ISSUES 759): the facial reduction's cost ratio for the symmetry search's budget (symred.c calls it once,
 * when a search is about to start with a budget of half a second or more): the reduction run on a copy of the problem as
 * it stands before the search, after the free elimination that precedes the reduction in the pipeline (a constraint on a
 * free variable's split pair is no candidate: on the user's 3x3 grid the copy found 5.3x without it, 2,844x with it), and
 * kept by the rule that decides below whether the reduction is kept (a reduction the pipeline would undo divides nothing).
 * The per-iteration cost model is the one of that rule. Returns 1 when nothing is reduced */
static const Problem *g_symfr_P = NULL; static const Params *g_symfr_par = NULL; static int g_symfr_do = 0;
static double symfr_ratio(void) {
    const Problem *P = g_symfr_P; const Params *par = g_symfr_par;
    extern int fr_has_candidate(const Problem *P);
    if (!P || !par || !fr_has_candidate(P)) return 1.0;
    const double tc0 = wtime();
    double c0 = (double)P->m * P->m * P->m / 3.0;
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) c0 += 10.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
    Problem Pc; problem_clone(P, &Pc);
    if (par->free_elim && par->dualize <= 0) { PSFreeElim *F = free_eliminate(&Pc, 0, par->free_elim_fill, par->free_elim < 0); if (F) free_elim_free(F); }
    Pc.ps = ps_new(&Pc);
    { extern int g_fr_maxdepth; g_fr_maxdepth = par->fr_maxdepth; }
    facial_reduction(&Pc, 0);
    double c1 = (double)Pc.m * Pc.m * Pc.m / 3.0;
    for (int k = 0; k < Pc.nblk; k++) if (Pc.blk[k].type == BLK_SDP) c1 += 10.0 * (double)Pc.blk[k].n * Pc.blk[k].n * Pc.blk[k].n;
    const int kept = Pc.ps->nrec > 0 && !(g_symfr_do == 1 && Pc.fr_depth >= 2 && c0 < par->fr_gain * c1);
    const double r = kept && c1 > 0 && c0 > c1 ? c0 / c1 : 1.0;
    ps_free(Pc.ps); Pc.ps = NULL;
    problem_free(&Pc);
    if (par->verbose > 0) printf("   symmetry search: the facial reduction on a copy of the problem divides the expected solve by %.1f (%.2f s)\n", r, wtime() - tc0);
    return r;
}
/* 4 November 2026 (ISSUES 760): the facial reduction's gain on a copy of the problem, with its free variables eliminated
 * (elim = 1, forced) or kept as split pairs: c0 / c1 in the per-iteration cost model of the rule that keeps a reduction
 * (m^3/3 + 10 sum n^3), 1 when nothing is reduced or the rule would undo it. A constraint holding a free variable's split
 * pair is never a candidate of the single-constraint reduction (its LP entries have both signs); the elimination's
 * combinations of rows are, so the elimination can expose faces the pairs hide. */
static double fe_fr_gain(const Problem *P, const Params *par, int elim, int do_fr, int maxdepth) {
    const double tg0 = wtime();
    Problem Pc; problem_clone(P, &Pc);
    const double tg1 = wtime();
    if (elim) { PSFreeElim *F = free_eliminate(&Pc, 0, par->free_elim_fill, 0); if (F) free_elim_free(F); }
    const double tg2 = wtime();
    double c0 = (double)Pc.m * Pc.m * Pc.m / 3.0;
    for (int k = 0; k < Pc.nblk; k++) if (Pc.blk[k].type == BLK_SDP) c0 += 10.0 * (double)Pc.blk[k].n * Pc.blk[k].n * Pc.blk[k].n;
    double r = 1.0;
    extern int fr_has_candidate(const Problem *P);
    if (fr_has_candidate(&Pc)) {
        Pc.ps = ps_new(&Pc);
        { extern int g_fr_maxdepth; g_fr_maxdepth = maxdepth; }
        facial_reduction(&Pc, 0);
        { extern int g_fr_maxdepth; g_fr_maxdepth = par->fr_maxdepth; }
        double c1 = (double)Pc.m * Pc.m * Pc.m / 3.0;
        for (int k = 0; k < Pc.nblk; k++) if (Pc.blk[k].type == BLK_SDP) c1 += 10.0 * (double)Pc.blk[k].n * Pc.blk[k].n * Pc.blk[k].n;
        const int kept = Pc.ps->nrec > 0 && (maxdepth == 1 || !(do_fr == 1 && Pc.fr_depth >= 2 && c0 < par->fr_gain * c1));
        if (kept && c1 > 0 && c0 > c1) r = c0 / c1;
        ps_free(Pc.ps); Pc.ps = NULL;
    }
    problem_free(&Pc);
    if (par->verbose > 1) printf("   [free elimination test: copy %.2f s, elimination %.2f s, facial reduction %.2f s, gain %.2f]\n", tg1 - tg0, tg2 - tg1, wtime() - tg2, r);
    return r;
}
static int run_pipeline_core(const char *fname, Params *par, int do_fr, int force_route, Run *u) {
    if (par->fr_maxdepth == 0) do_fr = 0;   /* -frdepth 0: no facial reduction, as -nofr (4 November 2026) */
    int race_want = 0; double race_s = 0;        /* 4.42: the first-order engine first (decided after the presolve) */
    memset(u, 0, sizeof(*u));
    g_run_no++;
    brisk_alt_clear();                           /* (an earlier run's point from before its end phase: not this run's) */
    Problem *P = &u->P;
    double t0 = wtime();
    if (g_brisk_data && fname == g_brisk_data_name) { if (problem_from_sdpa_data(g_brisk_data, P) != 0) return 2; }   /* 4.37: brisk_run_data */
    else if (read_sdpa(fname, P) != 0) return 2;
    {   /* 27 October 2026 (ISSUES 292): a block above the readers' cap of 46,340 is a long cone that must stay a cone vector
         * (xcone_decide, below); the options under which no block does are refused here, before any presolve step treats
         * it as a semidefinite block (-soc 0: the chordal conversion's clique search on its star pattern) */
        extern int g_read_maxn;
        const int no_cone = !par->soc || par->dual || par->mfipm || par->fom > 0 || par->lralm || par->bound_side || par->hsd == 0
                            || par->crossover || par->hp_kind;
        for (int k = 0; no_cone && k < P->nblk; k++)
            if (P->blk[k].type == BLK_SDP && P->blk[k].n > g_read_maxn) {
                fprintf(stderr, "brisk: block %d, a second-order cone of order %d, can only be solved as a cone (the default) by the interior-point method, not with the options given (its %d x %d matrix would be needed)\n",
                        k + 1, P->blk[k].n, P->blk[k].n, P->blk[k].n);
                return 2;
            }
    }
    { FreePair *fp = NULL; u->nfree_read = free_pairs_detect(P, &fp); free(fp); }   /* (the Summary counts them as free variables) */
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
    u->t_run0 = tpre0;
    {   /* 5.14: diagonal congruence scaling of the semidefinite blocks (blkscale.c; -blkscale) */
        const char *e = getenv("BRISK_BLKSCALE");
        const int mode = e ? atoi(e) : par->blkscale;
        if (getenv("BRISK_BLKSCALEMIN")) g_blkscale_min = atof(getenv("BRISK_BLKSCALEMIN"));
        if (getenv("BRISK_BLKSCALECAP")) g_blkscale_cap = atof(getenv("BRISK_BLKSCALECAP"));
        PSOrig *Of = mode ? ps_orig_build(P) : NULL;
        g_bs_probe = mode ? 0 : blkscale_probe(P);   /* (for the advice at the end of the log) */
        u->aff = mode ? blkscale_apply(P, mode, par->verbose) : NULL;
        if (u->aff) u->Oaff = Of; else if (Of) ps_orig_free(Of);
        if (!u->aff) {   /* 5.12: a moment relaxation over an ellipsoid in the coordinates of the unit ball (momaff.c; BRISK_NOAFF off) */
            PSOrig *Om = ps_orig_build(P);
            u->maff = momaff_apply(P, par->verbose);
            if (u->maff) {
                u->Oaff = Om;
                /* the map back multiplies the last digits by up to |B| |B^-1|: a tighter tolerance there (ellipsoid (4,8) of a
                 * user: 4.9e-8 on the problem as read at 1e-8, 5.1e-10 at 1e-11, one iteration more) */
                u->aff_tol0 = par->tol; u->aff_par = par;
                par->tol = fmax(1e-12, par->tol * fmin(1.0, 10.0 / momaff_amplification(u->maff)));
            } else ps_orig_free(Om);
        }
        if (getenv("BRISK_SETUPT")) printf("   [setup: block scaling %.2fs]\n", wtime() - tpre0);
    }
    /* 5.7: the matrix-free interior-point method first, on the problems it is made for: one to
     * four dense blocks of order >= 100 and a Schur complement whose factorization costs far
     * more than a matrix-free solve (the truss and vibration problems: tru11 3 s against 100 s).
     * The decision is taken after the presolve, as for the first-order engine below: a problem
     * that the symmetry reduction, the dual form or the chordal conversion changes is the
     * standard method's. The attempt stops as soon as a solve away from the optimum needs more CG
     * steps than the standard method's iteration is worth, or the whole attempt more than a quarter
     * of the standard solve (mfipm.c), and the standard method runs if it does not end OPTIMAL. */
    /* 25 October 2026: a doubly nonnegative program (the k-means and theta-plus relaxations) states each bound on an
     * entry of X as a row g X_ij + alpha s = b with an LP variable s of that row alone, so m grows as n^2/2 and the
     * interior-point method factors a dense Schur complement of that order (kmeans_n200_k10: m = 20,101, 583 s). When at
     * least half the rows are such bounds and m >= 4000, the first-order engine solves the problem first with the bounds
     * as a cone of their own on X (fom.c, fom_bound_solve: a linear system of the other rows only), within an iteration
     * cap of twice the interior-point method's estimated time; if it misses the tolerance, the interior-point method
     * solves the problem (main, rc 6). Measured (four threads, OPTIMAL at the default tolerance either way): the engine
     * from m = 4951 on (thetap_100 4.5 s against 7.0 s, thetap_180 20 s against 86 s, kmeans_n110_k6 0.56 s against
     * 18.5 s), the interior-point method below (thetap_80, m = 3161: 1.4 s against 3.0 s). Not at tolerances below 1e-8
     * (-acc high), nor before the matrix-free attempt, which it replaces here (kmeans_n240_k12: 136 s of matrix-free
     * steps in 1.5.5), and after it the interior-point method makes no matrix-free attempt either (fom_bound_maxit -1:
     * the class where it failed, ISSUES 86). */
    int bnd_want = 0;
    if (getenv("BRISK_BOUNDSCAN")) { printf("BOUNDSCAN m %d bounds %d\n", P->m, fom_bound_rows(P)); exit(0); }   /* (test: the rows the rule counts) */
    if (par->fom < 0 && par->fom_bound > 0 && par->fom_bound_maxit == 0 && P->m >= 4000 && par->tol >= 0.99e-8 && !par->mfipm
        && !par->lralm && !par->dual && !par->bound_side && !par->crossover && !par->warm_X && !par->hp_kind && !par->fom_start_x
        && !par->fom_start_y && force_route < 0) {
        double s3 = 0, s2 = 0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) { const double n = P->blk[k].n; s3 += n * n * n; s2 += n * n; }
        const int np = s3 > 0 ? fom_bound_rows(P) : 0;
        if (2.0 * np >= P->m) {
            /* the cap: twice the interior-point method's time as the rules below estimate it (the dense factor and the
             * block algebra, calibrated on theta12, 1et.1024 and G55mc; 6.8 s for thetap_100, which took 7.0 s), over
             * the engine's time per iteration on these files (4 threads: 1.3 ms at n = 110 to 9 ms at n = 300 for
             * k-means, 2.9 to 12 ms at n = 100 to 180 for theta-plus), modelled as 2e-7 n^2 + 2e-9 n^3 s */
            const double ipm_s = ((double)P->m * P->m * P->m / 3.0 + 20.0 * s3) / 6e9, it_s = 2e-7 * s2 + 2e-9 * s3;
            bnd_want = 1;
            par->fom_bound_maxit = (long)fmin(100000.0, fmax(2000.0, 2.0 * ipm_s / it_s));
            if (par->verbose > 0) printf("first-order engine: %d of the %d constraints are bounds on single entries of X; iteration cap %ld\n", np, P->m, par->fom_bound_maxit);
        }
    }
    int mf_want = 0; const int mf_m0 = P->m, mf_nb0 = P->nblk; double mf_s3 = 0, mf_budget = 0, mf_cgcap = 0, mf_prodcap = 0, mf_rm = 0, mf_sm = 0;
    if (par->mf_try > 0 && !bnd_want && par->fom_bound_maxit != -1 && !par->mfipm && par->fom <= 0 && !par->lralm && !par->dual && !par->bound_side && !par->crossover
        && !par->warm_X && !par->hp_kind && par->dualize <= 0 && !par->fom_start_x && !par->fom_start_y && force_route < 0) {
        int nsdp = 0, maxn = 0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; mf_s3 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n; }
        /* the two costs in the units of the first-order rule below (seconds on its reference
         * machine): the dense Schur factor and block algebra; about 250 products of 4 n^3 flops
         * plus the data (a nonzero counted as 20 n^3-flops: lyap80_big, 1e6 nonzeros, spends its
         * products there). 27 October 2026: the data of every block, as each product passes over it (A'v and A(.)
         * of an LP block too); counted over the semidefinite blocks only, deg_lpsdp_n200 (SDP: 720,000 of its 765,000
         * nonzeros in an LP block of 4,000 variables) read as 45,000, its cap per solve came out at 1,137 steps
         * instead of 435, and the attempt ran 17 iterations (39 s) where the factored method alone takes 25 s */
        double nz = 0;
        for (int k = 0; k < P->nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) nz += P->blk[k].A[t].nnz;
        const double ipm_s = ((double)P->m * P->m * P->m / 3.0 + 20.0 * mf_s3) / 6e9, mf_s = 250.0 * (mf_s3 + 20.0 * nz) / 1e9;
        static double rmin = -1, smin = -1; if (rmin < 0) { const char *e = getenv("BRISK_MFTRYR"); rmin = e ? atof(e) : 3.5; e = getenv("BRISK_MFTRYS"); smin = e ? atof(e) : 5.0; }
        /* lower thresholds (1 and 1.5) when every constraint has an entry in an LP block (and each LP variable is in
         * one constraint at most, 18 October 2026): the LP
         * part of the Schur complement is then a positive diagonal in every row, which the
         * preconditioner's base has exactly (the truss and vibration problems with their 2m
         * bounds: tru9 and vib9, m = 3,240, 11 and 15 s by the standard method against 1.1 and
         * 4.4 s). Of the problems on which a failed attempt costs 3-20 % at these thresholds
         * (mc_100x80, rose15, rabmo, butcher) none has such a block on all constraints. */
        double rm = rmin, sm = smin;
        if (!getenv("BRISK_MFTRYR") && nsdp >= 1 && nsdp <= 4 && maxn >= 100) {
            /* (18 October 2026) and each LP variable in one constraint at most, as the diagonal needs: a random
             * LP part with five constraints a column (the owner's benchmark, an SDP with 20,000 LP variables) has
             * an entry in every row but a sparse, not a diagonal, LP part, and the attempt failed (3.6 s of 7.4) */
            char *has = calloc((size_t)P->m + 1, 1); int all = has != NULL, nlp = 0;
            for (int k = 0; k < P->nblk && has && all; k++) if (P->blk[k].type == BLK_LP) {
                nlp++;
                char *own = calloc((size_t)P->blk[k].n + 1, 1);
                for (int t = 0; t < P->blk[k].ncon && all; t++) {
                    const SpSym *A = &P->blk[k].A[t];
                    if (A->nnz > 0) has[P->blk[k].con[t]] = 1;
                    for (int q = 0; q < A->nnz && own; q++) { if (own[A->row[q]]) { all = 0; break; } own[A->row[q]] = 1; }
                }
                free(own);
            }
            for (int c = 0; c < P->m && all; c++) if (!has[c]) all = 0;
            free(has);
            if (all && nlp > 0) { rm = 1.0; sm = 1.5; }
        }
        /* 5.8 (a user's benchmark): also when the dense Schur complement does not fit — the
         * alternative is then the first-order engine, which the matrix-free method beats on
         * these problems (sos_planted_n30_d2, m = 46,375: 25 s against 72 s or a time limit;
         * tru15: 23 s against a segmentation fault in 1.2.1) */
        mf_want = nsdp >= 1 && nsdp <= 4 && maxn >= 100 && ipm_s >= sm && mf_s * rm <= ipm_s;
        mf_rm = rm; mf_sm = sm;
        if (mf_want) {
            /* not when the dual form will be taken (its rule, as in the first-order decision below)
             * or a large block is the chordal conversion's case (pattern density <= 1 %): those are
             * the standard method's problems, and their symmetry search keeps its budget */
            FreePair *prs = NULL; const int np = free_pairs_detect(P, &prs); free(prs);
            long N = 0; int nlp = 0;
            for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) N += (long)P->blk[k].n * (P->blk[k].n + 1) / 2; else nlp += P->blk[k].n; }
            const long mK = P->m - np, mD = N + (nlp - 2L * np) + np - P->m;
            if (par->dualize < 0 && par->free_elim != 0 && mD >= 1 && mD <= mK / 2 && mK >= 20 && dualize_has_objective(P)) mf_want = 0;
            for (int k = 0; k < P->nblk && mf_want; k++) {
                const Block *B = &P->blk[k];
                if (B->type != BLK_SDP || B->n < 200 || par->chordal == 0) continue;
                if (blk_pattern_density(B) <= 0.01) mf_want = 0;
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
        if (getenv("BRISK_SETUPT")) printf("   [setup: sign symmetry %.2fs]\n", wtime() - tpre0);
    }
    if (par->symfile) {
        /* 4.32: exact symmetry reduction from the generators in the file (symred.c) */
        u->Osym = ps_orig_build(P);
        { /* 5.7: with -mfipm 1 the search's budget is a share of the matrix-free solve (about 50
           * products of 4 n^3 flops and 50 eigenvalue decompositions), not of the dense Schur
           * complement's: on the truss problem tru11 the search took 8.9 s of 11.8 s. */
          extern double g_sym_est_cap; extern int g_sym_cap_bound; g_sym_est_cap = 0; g_sym_cap_bound = 0;
          if (mf_want) g_sym_est_cap = 250.0 * mf_s3 / 1e9;
          if (par->mfipm == 1) { double s3 = 0; for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) s3 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n; g_sym_est_cap = 250.0 * s3 / 1e9; }
          /* 22 October 2026: the time left under -timelimit (hamming_11_2, DIMACS, SDP: after 485 s of the matrix-free
           * method the search got 597 s of a 900 s limit) */
          extern double g_sym_time_left; g_sym_time_left = par->timelimit > 0 ? fmax(0.0, par->timelimit - (wtime() - par->t_start)) : 0; }
        const double tsy0 = wtime();
        {   /* 4 November 2026 (ISSUES 759): the facial reduction that follows, measured on a copy when the search is about to
             * start with a budget of half a second or more (symred.c, g_sym_fr_div); BRISK_SYMFRDIV=0: off */
            extern double g_sym_est_div; extern double (*g_sym_fr_div)(void);
            static int sfd = -1; if (sfd < 0) { const char *e = getenv("BRISK_SYMFRDIV"); sfd = !(e && atoi(e) == 0); }
            g_sym_est_div = 1.0; g_sym_fr_div = NULL;
            if (sfd && do_fr && !strcmp(par->symfile, "auto")) { g_symfr_P = P; g_symfr_par = par; g_symfr_do = do_fr; g_sym_est_div = 0; g_sym_fr_div = symfr_ratio; }
        }
        u->sym = sym_reduce(P, par->symfile, par->verbose, par->symtime, par->symnodes, par->symbd, par->symmin, par->chordal != 0 ? par->chordal_minn : 1 << 30, par->chordal_density, par->symsigned);
        { extern double (*g_sym_fr_div)(void); g_sym_fr_div = NULL; g_symfr_P = NULL; }
        if (getenv("BRISK_SYMSCAN")) { extern double g_sym_est_div; fprintf(stderr, "symscan div %.4g sym %d sign %d time %.3f\n", g_sym_est_div, u->sym != NULL, u->sign != NULL, wtime() - tsy0); exit(0); }   /* (experiments: the step alone) */
        if (!u->sym) { extern double g_dec_sym_t; g_dec_sym_t += wtime() - tsy0; }   /* (4 November 2026: a search that found nothing, for the advice) */
        if (!u->sym) { ps_orig_free(u->Osym); u->Osym = NULL; }
        if (getenv("BRISK_SETUPT")) printf("   [setup: symmetry reduction %.2fs]\n", wtime() - tpre0);
        if (par->symalg) {
            /* 4.41: hidden and non-permutation symmetry: the *-algebra of each block (symalg.c) */
            u->Oalg = ps_orig_build(P);
            u->alg = alg_reduce(P, par->symalg, par->symalgmax, par->symmin, par->verbose, par->soc ? par->soc_minn : 0);
            if (!u->alg) { ps_orig_free(u->Oalg); u->Oalg = NULL; }
        }
        if (getenv("BRISK_SYMONLY")) { { printf("SYMONLY m %d -> %d order %d bd %d blocks", u->Osym ? u->Osym->m : P->m, P->m, u->sym ? u->sym->order : 1, u->sym ? u->sym->bd : 0); for (int k = 0; k < P->nblk; k++) printf(" %d", P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n); printf(" time %.2f\n", wtime() - tpre0); } exit(0); }
    }
    if (par->fom < 0) {
        /* 4.31: automatic first-order engine: only when the interior-point method cannot run
         * here - a dense Schur complement (few large blocks: the sparse Schur paths serve the
         * many-small-block models) that does not fit in memory */
        const double ram = brisk_mem_limit();
        /* 27 October 2026 (ISSUES 297): a block the solve takes as a second-order cone (arrow data, brisk_block_as_cone)
         * is not a large semidefinite block for these rules: its Schur term can be sparse (the long cones' low-rank form),
         * and the first-order engine is made for large semidefinite blocks. r2_port_n11000_k60 (SDP with SOCP: an arrow
         * block of order 11,002 on 23,898 constraints) went to the engine and ran out of memory under 9 GB, where the
         * interior-point method solves it in 7.7 s with a sparse factor of 3.2e6 nonzeros */
        char *iscone = calloc((size_t)P->nblk + 1, 1);
        for (int k = 0; k < P->nblk && iscone; k++) iscone[k] = brisk_block_as_cone(par, &P->blk[k]) != 0;
#define FOM_SDPBLK(k) (P->blk[k].type == BLK_SDP && !(iscone && iscone[k]))
        int maxn = 0, nsdp = 0;
        for (int k = 0; k < P->nblk; k++) if (FOM_SDPBLK(k)) { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; }
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
                if (!FOM_SDPBLK(k) || B->n < 200) continue;
                if (par->chordal != 0 && blk_pattern_density(B) <= 0.01) sparse_big = 1;
            }
            double est = (double)P->m * P->m * P->m / 3.0, sp = P->m;
            for (int k = 0; k < P->nblk; k++) if (FOM_SDPBLK(k)) sp += (double)P->blk[k].ncon * P->blk[k].ncon;
            if (pow(sp, 1.5) < est) est = pow(sp, 1.5);
            for (int k = 0; k < P->nblk; k++) if (FOM_SDPBLK(k)) est += 20.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
            const double ipm_s = est / 6e9;
            /* the first-order engine's own memory (Anderson history and work matrices, about
             * 400 bytes per entry of the blocks) must fit comfortably: it exits when it does
             * not (shmup5, four blocks of 1861) */
            double sn2 = 0;
            for (int k = 0; k < P->nblk; k++) if (FOM_SDPBLK(k)) sn2 += (double)P->blk[k].n * P->blk[k].n;
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
                if (!FOM_SDPBLK(k) || B->n < par->chordal_minn || B->n < 200) continue;
                const double dens = blk_pattern_density(B);
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
            const int dual_rule = par->dualize > 0 || (par->dualize < 0 && par->free_elim != 0 && mD >= 1 && mD <= mK / 2 && mK >= 20 && dualize_has_objective(P));
            const double mp = dual_rule ? (double)mD : (par->free_elim != 0 ? (double)mK : (double)P->m);
            if (8.0 * mp * mp <= 0.5 * ram) {
                use = 0; u->fom_check = 1;
                if (par->verbose >= 0) printf("first-order engine: deferred (m = %d as read; the %s should leave about %.0f rows)\n", P->m, dual_rule ? "dual form" : "free elimination", mp);
            }
        }
        if (!use && bnd_want) race_want = 2;   /* the engine with the bounds first (above) */
        else if (bnd_want) par->fom_bound_maxit = 0;   /* the engine alone: m does not fit the interior-point method */
        par->fom = use ? 1 : 0;
        if (use) {
            if (par->free_elim < 0) par->free_elim = 0;
            if (par->dualize < 0) par->dualize = 0;
            if (par->tracebound < 0) par->tracebound = 0;
            if (par->chordal < 0) par->chordal = 0;
            if (par->verbose >= 0) printf("first-order engine: the dense Schur complement (m = %d, %.1f GB) does not fit in memory\n", P->m, 8.0 * (double)P->m * P->m / 1e9);
        }
        free(iscone);
#undef FOM_SDPBLK
    }
    /* what the problem before an automatic dual form needs beyond the postsolve's copy of its data (u->Odual), to rebuild
     * it when the form is undone (below): its right-hand side as it stands and its row map (27 October 2026) */
    double *dual0_b = NULL; int *dual0_orig = NULL, dual0_m = P->m, dual0_morig = P->morig; const double dual0_off = P->obj_off;
    if (par->dualize < 0 && par->free_elim != 0) {
        dual0_b = malloc(sizeof(double) * (P->m + 1)); memcpy(dual0_b, P->b, sizeof(double) * P->m);
        dual0_orig = malloc(sizeof(int) * (P->m + 1)); memcpy(dual0_orig, P->orig, sizeof(int) * P->m);
    }
    if (par->dualize > 0 || (par->dualize < 0 && par->free_elim != 0)) {
        /* 4.30: the dual form when its Schur complement is the smaller one (SOS programs
         * given in the wrong form); its y pairs are then eliminated below */
        PSOrig *Of = NULL;
        u->dual = dualize_apply(P, par->dualize < 0, par->verbose >= 0 ? par->verbose : 0, &Of);
        if (u->dual) u->Odual = Of; else if (Of) ps_orig_free(Of);
        if (getenv("BRISK_SETUPT")) printf("   [setup: dual form %.2fs]\n", wtime() - tpre0);
    }
    u->O = ps_orig_build(P);
    if (par->free_elim) {
        PSFreeElim *F = free_eliminate(P, par->verbose >= 0 ? par->verbose : 0, par->free_elim_fill, par->free_elim < 0);
        /* 4 November 2026 (ISSUES 760, BRISK_FEFR=1, test): an elimination refused on its estimated cost alone is weighed with
         * the facial reduction it makes possible: the estimate divided by the reduction's gain with the elimination and
         * multiplied by its gain without (both measured on copies). roa_acrobot_d6_K: estimate 1.03, the reduction removes
         * 1,221 of the rows after the elimination and none before; the run 17.3 -> 3.9 s */
        { extern double g_fe_refused_cost; static int fefr = -1; if (fefr < 0) { const char *e = getenv("BRISK_FEFR"); fefr = e ? atoi(e) : 0; }
          extern long g_fe_refused_cand;
          const double est0 = g_fe_refused_cost;   /* (read first: the copies' eliminations reset it) */
          /* (the copies only when the eliminated rows hold a candidate of the facial reduction: on the power-flow
           * relaxations the forced elimination of the copy cost up to 7.3 s and the reduction found nothing) */
          if (!F && fefr && par->free_elim < 0 && do_fr && est0 > 1.0 && g_fe_refused_cand > 0) {
              /* (the gains are at least 1: a first level that reduces nothing ends the test; the copy without the
               * elimination only when the gain with it could bring the estimate to 1) */
              const double tq = wtime(), g1 = fe_fr_gain(P, par, 1, do_fr, 1);
              const double ge = g1 > 1.0 ? fe_fr_gain(P, par, 1, do_fr, par->fr_maxdepth) : 1.0;
              const double gk = ge >= est0 ? fe_fr_gain(P, par, 0, do_fr, par->fr_maxdepth) : 1.0, est = est0 * gk / ge;
              if (par->verbose > 0) printf("presolve: free elimination: the facial reduction divides the solve by %.2f eliminated and by %.2f kept: estimate %.2f -> %.2f (%.2f s)\n", ge, gk, est0, est, wtime() - tq);
              if (est <= 1.0) F = free_eliminate(P, par->verbose >= 0 ? par->verbose : 0, par->free_elim_fill, 0);
          } }
        if (getenv("BRISK_SETUPT")) printf("   [setup: free elimination done at %.2fs]\n", wtime() - tpre0);
        if (F) { u->fe = F; u->Ofile = u->O; u->O = ps_orig_build(P); }
    }
    if (dual0_b) {
        /* 27 October 2026: the automatic dual form pays through the free elimination of its pairs of y (its rule counts the
         * rows left after it). When the elimination refuses (fill), the dual form keeps all its rows and its pairs, and it
         * is kept only when its Schur complement is still the smaller one: the entries the semidefinite blocks give it,
         * the sum over the blocks of the square of the number of constraints on each (each block couples all of its
         * constraints), against the same count for the problem before. A KYP lemma's LMI (r2_brl_n100, SDP: 5,051
         * constraints on a block of order 104) has 5,460 rows in its dual form, each on the block: the dual form was
         * solved with 5,051 split pairs, its solve crept at short steps and stopped at 2.6e-5. A quadratic knapsack's
         * relaxation with its bound rows (qkp_rlt_n70_d075, SDP) keeps the dual form: 2,556 of its 9,872 rows are on the
         * block against all 7,387 of the problem before (the rows of the LP variables are sparse in the Schur complement) */
        int undo = 0; double s0 = 0, s1 = 0;
        if (u->dual && !u->fe && u->Odual) {
            const PSOrig *O0 = u->Odual;
            int *cnt = calloc((size_t)O0->nblk + 1, sizeof(int)), *last = malloc(sizeof(int) * (O0->nblk + 1));
            for (int k = 0; k < O0->nblk; k++) last[k] = -1;
            for (size_t q = 0; q < O0->nnz; q++) {        /* the constraints on each block: their entries come in constraint order */
                const int k = O0->blk[q], c = O0->con[q];
                if (c >= 0 && c != last[k]) { cnt[k]++; last[k] = c; }
            }
            for (int k = 0; k < O0->nblk; k++) if (O0->bs[k] > 0) s0 += (double)cnt[k] * cnt[k];
            free(cnt); free(last);
            for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) s1 += (double)P->blk[k].ncon * P->blk[k].ncon;
            undo = s1 >= s0;
        }
        if (undo) {
            if (par->verbose >= 0) printf("presolve: dual form not used: its pairs of y were not eliminated, and its Schur complement would not be the smaller one (%.3g entries from the semidefinite blocks against %.3g)\n", s1, s0);
            const PSOrig *O0 = u->Odual;
            problem_free(P);
            problem_from_trips(P, O0->m, O0->nblk, O0->bs, O0->b, O0->nnz, O0->con, O0->blk, O0->ii, O0->jj, O0->v);
            memcpy(P->b, dual0_b, sizeof(double) * dual0_m); memcpy(P->orig, dual0_orig, sizeof(int) * dual0_m);
            P->morig = dual0_morig; P->obj_off = dual0_off;
            dualize_free(u->dual); u->dual = NULL;
            ps_orig_free(u->Odual); u->Odual = NULL;
            ps_orig_free(u->O); u->O = ps_orig_build(P);
            PSFreeElim *F = free_eliminate(P, par->verbose >= 0 ? par->verbose : 0, par->free_elim_fill, par->free_elim < 0);
            if (F) { u->fe = F; u->Ofile = u->O; u->O = ps_orig_build(P); }
        }
        free(dual0_b); free(dual0_orig);
    }
    if (u->fom_check && 8.0 * (double)P->m * (double)P->m > 0.5 * brisk_mem_limit()) {
        if (par->verbose >= 0) printf("first-order engine: m = %d after the dual form and free elimination still does not fit; restarting with the first-order engine\n", P->m);
        return 4;
    }
    /* (26 October 2026: not added by default to a problem the SOCP solver will take, brisk_problem_as_socp: the bound is a
     * device of the self-dual embedding, and its row of right-hand side 1e7 (1 + |b|) set the SOCP solver's relative
     * accuracy on the scale of its slack, 2.7e-8 on the problem as read for a CVXPY model that solves to 1.6e-9 without) */
    if (par->tracebound > 0 || (par->tracebound < 0 && u->fe && !u->fe->cheap && !u->fe->bycost && !brisk_problem_as_socp(par, P))) {
        double bmax = 0;
        for (int i = 0; i < P->m; i++) bmax = fmax(bmax, fabs(P->b[i]));
        u->tbR = par->tracebound > 0 ? par->tracebound : 1e7 * (1.0 + bmax);
        tracebound_add(P, u->tbR);
        u->Otb = u->O; u->O = ps_orig_build(P);
        if (par->verbose >= 0) printf("presolve: trace bound sum tr(X) + s = %.3g added (row %d)\n", u->tbR, P->m);
    }
    P->ps = ps_new(P);
    if (getenv("BRISK_SETUPT")) printf("   [setup: orig copy + free elimination %.2fs]\n", wtime() - tpre0);
    if (par->deprows > 0 && !(u->tbR > 0)) {
        /* 20 October 2026: linearly dependent equality constraints (presolve.c), before the facial reduction (which
         * would otherwise find their consequences, with duals it cannot always recover) */
        static double dtol = -1; if (dtol < 0) dtol = getenv("BRISK_DEPTOL") ? atof(getenv("BRISK_DEPTOL")) : 1e-8;
        int ninc = 0;
        const int m0 = P->m; const double tdep = wtime();
        u->dep_removed = remove_dependent_rows(P, par->verbose, dtol, &ninc);
        if (getenv("BRISK_DEPSCAN")) { printf("depscan m %d removed %d inconsistent %d time %.3f presolve %.2f\n", m0, u->dep_removed, ninc, wtime() - tdep, wtime() - tpre0); fflush(stdout); exit(0); }   /* (experiments: the check alone) */
        if (getenv("BRISK_SETUPT")) printf("   [setup: dependent constraints done at %.2fs]\n", wtime() - tpre0);
    }
    if (do_fr) {
        double c0 = (double)P->m * P->m * P->m / 3.0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) c0 += 10.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        /* 4.30: restored if the reduction is rejected; 5.18: not copied when nothing can be reduced */
        extern int fr_has_candidate(const Problem *P);
        const int frc = getenv("BRISK_FRCLONE") ? 1 : fr_has_candidate(P);
        const double tfr0 = wtime();   /* (4 November 2026: a reduction made and then undone is a decision's cost, for the advice) */
        Problem Pc; if (frc) problem_clone(P, &Pc);
        /* 23 October 2026: the faces a lifted relaxation's linear constraints expose (presolve.c), after the clone: a
         * reduction undone below takes them away too */
        const int nlf = lifted_faces(P, par->verbose);
        /* 3 November 2026 (ISSUES 745): the faces fixed by small index sets (their dependent constraints are removed on
         * the face itself, by the facial reduction) */
        {   const int ncf = clique_faces(P, par->verbose);
            if (getenv("BRISK_CFSCAN")) { extern int g_cf_vectors, g_cf_sets; printf("cfscan m %d faces %d vectors %d sets %d\n", P->m, ncf, g_cf_vectors, g_cf_sets); fflush(stdout); exit(0); }   /* (experiments: the detection alone) */
        }
        if (getenv("BRISK_LFSCAN")) { extern double g_lf_time; printf("lfscan m %d faces %d time %.4f\n", P->m, nlf, g_lf_time); fflush(stdout); exit(0); }   /* (experiments: the detection alone) */
        { extern int g_fr_maxdepth; g_fr_maxdepth = par->fr_maxdepth; }
        facial_reduction(P, par->verbose);
        if (nlf > 0 && par->deprows > 0 && !(u->tbR > 0)) {
            /* on the face, constraints that were independent can depend on the others (a QAP relaxation's row and
             * column sums): the check again, on the reduced problem */
            static double dtol2 = -1; if (dtol2 < 0) dtol2 = getenv("BRISK_DEPTOL") ? atof(getenv("BRISK_DEPTOL")) : 1e-8;
            int ninc2 = 0;
            u->dep_removed += remove_dependent_rows(P, par->verbose, dtol2, &ninc2);
        }
        double c1 = (double)P->m * P->m * P->m / 3.0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) c1 += 10.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        /* 4.20: chains of dependent reductions (depth >= 2) mean a singularity degree >= 2;
         * the removed duals then often cannot be recovered (the dual optimum of the problem
         * as read is not attained) and the problem is solved a second time without facial
         * reduction. Unless the reduction makes the solve much cheaper (e_moment: 145x),
         * solve without it from the start (rose13, taha1a, roa_acrobot: 1.9-3.4x). */
        u->fr_cost = c1 > 0 ? c0 / c1 : 1.0;
        if (par->verbose >= 1 && P->ps->nrec > 0) printf("presolve: facial reduction: the per-iteration cost model gives %.2fx for the problem as read\n", c0 / c1);
        if (frc && do_fr == 1 && P->fr_depth >= 2 && P->ps->nrec > 0 && c0 < par->fr_gain * c1) {
            if (par->verbose >= 0)
                printf("presolve: facial reduction of depth %d saves only %.1fx per iteration; solving without it\n", P->fr_depth, c0 / c1);
            { extern double g_dec_fr_t; g_dec_fr_t += wtime() - tfr0; }
            ps_free(P->ps);
            problem_free(P);
            *P = Pc;
            P->ps = ps_new(P);
        } else if (frc) problem_free(&Pc);
    }
    u->fr_removed = P->ps->nrec;
    u->fr_faces = 0; for (int r = 0; r < P->ps->nrec; r++) if (P->ps->rec[r].con <= -2) u->fr_faces++;
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
        if (getenv("BRISK_CLIQUEHASH")) { printf("CLIQUEHASH m %d blocks %d split %d presolve %.3f\n", P->m, P->nblk, nc, wtime() - tpre0); fflush(stdout); exit(0); }   /* (experiments: the conversion alone) */
    }
    {   /* 5.12 (BRISK_MOMRULE=0: off) a moment relaxation as given: its largest block a moment
         * matrix (every entry in exactly one constraint matrix, the entry (b, c) in the constraint
         * of the monomial b + c), at least one more block (a localizing matrix), and no LP part,
         * dual form, free variables, trace bound, chordal conversion or symmetry reduction (the
         * kernel and image forms of SOS programs go through those): the NT direction in the
         * embedding and the centred start 0.3 I, I (IMPLEMENTATION Section 13; a user's 72
         * relaxations 602 -> 514 iterations, 1151 -> 1054 s; no OPTIMAL lost under perturbations) */
        static int momrule = -1; if (momrule < 0) { const char *e = getenv("BRISK_MOMRULE"); momrule = e ? atoi(e) : 1; }
        int has_lp = 0; for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_LP) has_lp = 1;
        if (momrule && par->hsd_dir < 0 && par->init == 1 && !u->dual && !u->fe && u->tbR == 0 && !P->ps->chordal && !has_lp
            && !u->sym && !u->sign && !u->alg) {
            int found = 0, nsdp = 0, nmax = 0;
            for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > nmax) nmax = P->blk[k].n; }
            /* the moment matrix is the largest block and there is at least one more (a localizing
             * matrix): the relaxations of polynomial problems over a set with inequality constraints */
            for (int k = 0; k < P->nblk && !found && nsdp >= 2; k++) {
                const Block *B = &P->blk[k];
                if (B->type != BLK_SDP || B->n < 20 || B->n > 4000 || B->n < nmax) continue;
                const int n = B->n; const size_t nt2 = (size_t)n * (n + 1) / 2;
                unsigned char *cnt = calloc(nt2, 1); int bad = 0; size_t cov = 0;
                for (int t = 0; t < B->ncon && !bad; t++) {
                    const SpSym *A = &B->A[t];
                    for (int q = 0; q < A->nnz; q++) {
                        int r = A->row[q], c = A->col[q]; if (r > c) { int w = r; r = c; c = w; }
                        const size_t id = (size_t)c * (c + 1) / 2 + r;
                        if (cnt[id]) { bad = 1; break; }
                        cnt[id] = 1; cov++;
                    }
                }
                free(cnt);
                if (getenv("BRISK_MOMDBG")) printf("   [moment rule: block %d n %d: %s, %zu of %zu entries]\n", k + 1, n, bad ? "an entry in two constraints" : "one constraint an entry", cov, nt2);
                /* (the constant monomial's entry may be in C only); several entries a constraint on average
                 * (a Hankel-type moment matrix: the entries of a monomial b + c), not one (a free symmetric
                 * matrix written entry by entry, as the Lyapunov variables of the control problems) */
                int ncb = 0; for (int t = 0; t < B->ncon; t++) if (B->A[t].nnz > 0) ncb++;
                if (!bad && cov + 1 >= nt2 && ncb > 0 && (double)cov >= 2.0 * ncb) found = 1;
            }
            if (found) {
                par->hsd_dir = 1; par->init = 3;
                /* and no matrix-free attempt first: on these its CG steps grow from the first
                 * iterations (a user's 72: it ran on 9 and failed on all, 1-6 s each) */
                if (!getenv("BRISK_MOMMF")) mf_want = 0;
                if (par->verbose >= 0) printf("moment relaxation: NT direction and the centred start\n");
            }
        }
    }
    /* 1 November 2026 (the problems written differently, ISSUES 742): a removal of linearly dependent rows leaves the
     * problem the same one (their dual values are 0), so it does not count as a change of the problem below; the
     * attempt's cost model and caps, computed with the rows as read, are computed again on the rows that remain. An SOS
     * polynomial SDP with 2,529 of its 15,178 rows repeated lost the attempt (P->m no longer the m of the decision) and
     * took the factored embedding: 15.3 s against 5.1 s for the problem written without them. BRISK_MFDEP=0: as before */
    const int mf_dep_ok = u->dep_removed > 0 && P->m + u->dep_removed == mf_m0 && !(getenv("BRISK_MFDEP") && atoi(getenv("BRISK_MFDEP")) == 0);
    if (mf_want && mf_dep_ok) {
        double s3 = 0, nz = 0;
        for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) s3 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
            for (int t = 0; t < P->blk[k].ncon; t++) nz += P->blk[k].A[t].nnz; }
        const double ipm_s = ((double)P->m * P->m * P->m / 3.0 + 20.0 * s3) / 6e9, mf_s = 250.0 * (s3 + 20.0 * nz) / 1e9;
        mf_want = ipm_s >= mf_sm && mf_s * mf_rm <= ipm_s;
        mf_budget = 4.0 * ipm_s;
        mf_cgcap = (double)P->m * P->m * P->m / 3.0 / (6.0 * (4.0 * s3 + 80.0 * nz));
        mf_prodcap = 45.0 * mf_cgcap;
        if (mf_cgcap < 100) mf_cgcap = 100;
        if (mf_cgcap > 3000) mf_cgcap = 3000;
    }
    if (getenv("BRISK_MFTRYDRY")) { extern int g_sym_cap_bound; printf("MFTRY want %d chordal %d dual %d sym %d m %d->%d search cut %d\n", mf_want, P->ps->chordal, u->dual != NULL, u->sym || u->sign || u->alg, mf_m0, P->m, g_sym_cap_bound); exit(0); }
    if (mf_want && !P->ps->chordal && !u->dual && !u->sym && !u->sign && !u->alg && (P->m == mf_m0 || mf_dep_ok) && P->nblk == mf_nb0) {
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
    if (race_want == 2 && !P->ps->chordal && !u->dual && !u->sym && !u->sign && !u->alg) {   /* (a symmetry reduction sums the bound rows over orbits) */
        if (par->verbose >= 0) printf("first-order engine first: most constraints are bounds on single entries of X; it gets at most %ld iterations (-fombound 0 turns this off)\n", par->fom_bound_maxit);
        return 6;
    }
    if (race_want == 2) { race_want = 0; par->fom_bound_maxit = -1; }
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
    if (g_log_brief && !g_pre_logged && par->verbose > 0) {   /* (ISSUES 725: the problem and the presolve before the iterations, once a run) */
        g_pre_logged = 1;
        char sz[200]; summary_size(sz, sizeof sz, u->Ofile ? u->Ofile : u->O, u->fe ? u->fe->np : u->nfree_read);
        g_log_user++;
        const PSOrig *O0 = u->Oaff ? u->Oaff : u->Osign ? u->Osign : u->Osym ? u->Osym : u->Oalg ? u->Oalg : u->Odual ? u->Odual : u->Ofile ? u->Ofile : u->Otb ? u->Otb : u->O;
        summary_size(sz, sizeof sz, O0, u->fe ? u->fe->np : u->nfree_read);
        printf("Problem:  %s\n", sz);
        /* (31 October 2026, the owner: the steps that changed the problem, then, in one line, the searches that found
         * nothing: symmetry, chordal structure, facial reduction; "off" when an option turned one off) */
        long lr_c, lr_r; const int lr_any = lowrank_used(&u->P, &lr_c, &lr_r) > 0;
        const int applied = u->Oaff || u->Osign || u->Osym || u->Oalg || u->Odual || u->Ofile || (u->P.ps && (u->P.ps->mom || u->P.ps->chordal)) || u->dep_removed > 0 || u->fr_removed > 0 || lr_any;
        if (applied) print_preprocessing(u, "Presolve: ", "          ");
        char nf[200] = ""; size_t w = 0;
        if (!(u->Osign || u->Osym || u->Oalg)) w += snprintf(nf + w, sizeof nf - w, "%ssymmetry: %s", w ? "; " : "", par->symfile || par->symalg ? "none" : "off");
        if (!(u->P.ps && (u->P.ps->mom || u->P.ps->chordal))) w += snprintf(nf + w, sizeof nf - w, "%schordal decomposition: %s", w ? "; " : "", par->chordal != 0 ? "none" : "off");
        if (!(u->fr_removed > 0)) w += snprintf(nf + w, sizeof nf - w, "%sfacial reduction: %s", w ? "; " : "", do_fr ? "none" : "off");
        if (!lr_any) w += snprintf(nf + w, sizeof nf - w, "%slow-rank data: %s", w ? "; " : "", par->lowrank != 0 ? "none" : "off");
        if (w) printf("%s%s\n", applied ? "          " : "Presolve: ", nf);
        g_log_user--;
    }
    if (getenv("BRISK_SETUPT")) printf("   [setup: problem_prepare done at %.2fs]\n", u->tpre);
    long nroute[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, nweyl = 0;
    int maxn = 0, nsdp = 0, nlp = 0, nlr = 0, ndict = 0, dq = 0, dstr = 0, ncls = 0;
    long lrR = 0, ncK = 0, ncC = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        if (B->type == BLK_LP) { nlp += B->n; continue; }
        nsdp++;
        if (B->n > maxn) maxn = B->n;
        if (B->lowrank) { nlr++; lrR += B->lrR; }
        if (B->dict) { ndict++; dq += B->dnu + B->dnv; dstr += B->dstruct; }
        if (B->cls) { ncls++; ncK += B->cb->ncon; ncC += B->ncon; }
        if (B->allweyl) nweyl += B->ns;
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
        printf("Schur routes: %ld sparse-sparse, %ld row-product, %ld pattern row-product, %ld dense-BLAS3%s; %d low-rank block(s), total rank %ld\n",
               nroute[0], nroute[1], nroute[3], nroute[2], "", nlr, lrR);
        /* (23 October 2026: the Weyl route's pairs are Walsh-Hadamard transforms, not gathers; the log had said gathers) */
        if (nroute[5] > nweyl) printf("Schur routes: %ld monomial constraints in batched products (one BLAS-3 call per chunk, the pairs by gathers)\n", nroute[5] - nweyl);
        if (nweyl) printf("Schur routes: %ld Pauli (Weyl) constraints in batched products, the pairs by Walsh-Hadamard transforms\n", nweyl);
        if (ndict)
            printf("Dictionary route: %d block(s), lifted size %d, %d constraint parts through shared vectors\n",
                   ndict, dq, dstr);
        if (ncls)
            printf("Entry-class route: %d block(s), %ld entry classes for %ld constraint matrices\n", ncls, ncK, ncC);
    }
    Result *R = &u->R;
    u->y = calloc(P->m + 1, sizeof(double));
    u->X = malloc(sizeof(double *) * (P->nblk + 1));
    free(u->xc_p); u->xc_p = xcone_decide(u, par, P);
    {   /* 27 October 2026 (ISSUES 292): a block above the readers' cap of 46,340 is accepted only as a long cone that comes
         * back as its cone vector (problem.c, big_blocks_ok); where the rule above keeps blocks dense (a reduction that
         * maps blocks, the bound mode, -soc 0, ...), its n x n matrix cannot be held: the run stops before the solve */
        extern int g_read_maxn;
        for (int k = 0; k < P->nblk; k++)
            if (P->blk[k].type == BLK_SDP && P->blk[k].n > g_read_maxn && !(u->xc_p && u->xc_p[k] >= 0)) {
                fprintf(stderr, "brisk: block %d, a second-order cone of order %d, cannot be kept as a cone vector with the options or the reductions of this run (its %d x %d matrix would be needed)\n",
                        k + 1, P->blk[k].n, P->blk[k].n, P->blk[k].n);
                return 2;
            }
    }
    for (int k = 0; k < P->nblk; k++) {
        size_t len = P->blk[k].type == BLK_SDP && !(u->xc_p && u->xc_p[k] >= 0) ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
        u->X[k] = calloc(len + 1, sizeof(double));
    }
    if (u->xc_p && par->verbose > 1) { int c = 0; for (int k = 0; k < P->nblk; k++) c += u->xc_p[k] >= 0; printf("   %d block(s) with arrow data returned as cone vectors (no dense completion)\n", c); }
    int st_ = -1;
    double tdual = 0;
    /* 4.37 bound mode: a tracker per side asked for, on this problem (the dual form exchanges
     * the sides); one X and one y per slot, best and anchor */
    BoundTrack bt[2];
    memset(bt, 0, sizeof(bt));
    par->btrack = par->btrack2 = NULL;
    if (par->bound_side) {
        const int nside = (par->bound_side & 1) + ((par->bound_side >> 1) & 1);
        for (int sd = 1; sd <= 2; sd++) {
            if (!(par->bound_side & sd)) continue;
            BoundTrack *T = &bt[sd - 1];
            T->side = u->dual ? 3 - sd : sd;
            /* (1e-7 at the default bound_tol: the certificate projects X onto A(X) = b anyway, and a
             * late iterate at 2e-8 can be several units better than every one at 1e-8) */
            T->tol = par->bound_track_tol > 0 ? par->bound_track_tol : 1000.0 * par->bound_tol;
            T->tol_a = fmax(T->tol, 1e-2);     /* anchors: the barrier-metric step of bound.c makes them feasible */
            T->m = P->m; T->nblk = P->nblk;
            double bytes = 8.0 * (BOUND_NANCHOR + 1) * (P->m + 1);
            for (int k = 0; k < P->nblk; k++) bytes += 8.0 * (BOUND_NANCHOR + 1) * (double)(P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n);
            if (nside * bytes > fmin(1024.0 * 1048576.0 * brisk_mem_scale(), 0.15 * brisk_mem_limit())) {
                T->disabled = 1;
                if (par->verbose >= 0) printf("bound: the candidates (%.0f MB) do not fit; only the returned points are certified\n", nside * bytes / 1048576.0);
                continue;
            }
            for (int s = 0; s <= BOUND_NANCHOR; s++) {
                double **X = calloc(P->nblk + 1, sizeof(double *));
                for (int k = 0; k < P->nblk; k++) {
                    const size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n;
                    X[k] = calloc(len + 1, sizeof(double));
                }
                double *yv = calloc(P->m + 1, sizeof(double));
                if (s == 0) { T->X = X; T->y = yv; } else { T->Xa[s - 1] = X; T->ya[s - 1] = yv; }
            }
            if (!par->btrack) par->btrack = T; else par->btrack2 = T;
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
    /* 25 October 2026 (ISSUES 290): a method that hands over to another sets the solve time from its own clock, so each
     * hand-over below adds the time spent before it (the first-order engine's 65 s before the matrix-free method on
     * pglib_opf_case4661_sdet, forming A A' before finding that it does not fit, were missing from the total) */
    const double t_disp0 = wtime();
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
        if (hand == 2 && par->verbose >= 0) printf("matrix-free IPM: CG gave up far from the optimum; the interior-point method with the factored Schur complement starts cold\n");
        if (hand == 1) {
            par->warm_X = Xi; par->warm_Z = Zi; par->warm_y = yi;
            if (!(par->warm_lam > 0)) par->warm_lam = 1.0;
            memset(R, 0, sizeof(*R));
            run_solve(P, par, R, u);
            R->t_total += tmf;
            par->warm_X = par->warm_Z = NULL; par->warm_y = NULL;
            st_ = 0;
        } else {
            /* ended by itself (optimal, or no progress): its result stands */
            if (par->verbose >= 0 && hand == 0) printf("matrix-free IPM: ended without a hand-off (status %d); re-solving by the interior-point method with the factored Schur complement\n", R->status);
            if (hand == 0 || hand == 2) { memset(R, 0, sizeof(*R)); run_solve(P, par, R, u); R->t_total += tmf; st_ = 0; }
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
                if (par->verbose >= 0) printf("matrix-free interior-point method first: not optimal after %.1f s; the interior-point method with the factored Schur complement from its iterate of merit %.1e\n", tmf, mfipm_snap_merit());
                /* 23 October 2026: two solves, listed as two in the Summary (main): the matrix-free one and the factored one
                 * started from its iterate (tru11, POEMA, SDP: "Solve: matrix-free interior-point method, 5 iterations" had
                 * named the matrix-free method for the factored self-dual embedding's 5 iterations) */
                g_mf_hand_t = tmf;     /* (no error: its errors leave out the cone, and its last X can be indefinite: tru11) */
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
                run_solve(P, par, R, u);
                R->t_total += tmf;
                if (R->status != ST_OPTIMAL && !(fmax(fabs(R->relgap), fmax(R->pinf, R->dinf)) < fmax(fabs(R0.relgap), fmax(R0.pinf, R0.dinf)))) {
                    const double tt = R->t_total;
                    *R = R0; R->t_total = tt;
                    g_mf_hand_t = -1;        /* the matrix-free solve's own result is returned (the Summary's note says so) */
                    brisk_alt_clear();       /* (the factored solve's earlier point is not this result's) */
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
            const double t_fom = wtime() - t_disp0;
            st_ = mfipm_solve(P, par, R, u->y, u->X);
            R->t_total += t_fom;
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
    if (st_ < 0) { const double t_before = wtime() - t_disp0; run_solve(P, par, R, u); R->t_total += t_before; }   /* (the dual method's time, or a failed matrix-free or first-order attempt's) */
    par->btrack = par->btrack2 = NULL;
    if (u->aff_par) { par->tol = u->aff_tol0; u->aff_par = NULL; }   /* (the tolerance asked for, on the problem as read) */
    /* ---- postsolve: the solution of the problem as read, and its errors */
    u->yo = calloc(u->O->m + 1, sizeof(double));
    PSResult *pr = &u->pr;
    free(u->xc_o); u->xc_o = u->xc_p ? malloc(sizeof(int) * (u->O->nblk + 1)) : NULL;
    postsolve_set_time_left(ps_time_left(par)); postsolve_set_tol(par->tol);
    { extern void brisk_memtrace(const char *); brisk_memtrace("postsolve start"); }
    postsolve(P, u->O, u->X, u->y, &u->Xo, u->yo, pr, par->verbose, u->xc_p, u->xc_o);
    { extern void brisk_memtrace(const char *); brisk_memtrace("postsolve end"); }
    if (u->xc_o) { int any = 0; for (int o = 0; o < u->O->nblk; o++) any |= u->xc_o[o] >= 0; if (!any) { free(u->xc_o); u->xc_o = NULL; } }
    int status = R->status;
    double acc = 0;
    u->measured = pr->have_x;
    if (u->measured) {
        acc = brisk_acc_err(pr->err);
        /* the status reflects the returned point as measured on the original data */
        if (status == ST_OPTIMAL || status == ST_REDUCED || status == ST_MAXIT || status == ST_NUMERIC || status == ST_TIME) {
            if (acc <= par->tol) status = ST_OPTIMAL;
            else if (acc <= par->red_acc) status = ST_REDUCED;
            else if (status == ST_OPTIMAL || status == ST_REDUCED) status = ST_NUMERIC;
        }
    } else {
        /* chordal conversion: primal errors from the converted problem, dual side measured */
        acc = brisk_nmax(brisk_nmax(R->relgap, R->pinf), brisk_nmax(pr->err[4], R->relcomp));
        if ((status == ST_OPTIMAL || status == ST_REDUCED) && !(pr->err[4] <= par->red_acc && acc < INFINITY)) status = ST_NUMERIC;   /* (a measure that is NaN fails, audit A11) */
    }
    u->status = status;
    u->acc = acc;
    u->nXo = u->O->nblk;
    u->Ofin = u->O;
    run_finish(u, par);
    alt_judge(u, par);
    for (int sd = 1; sd <= 2; sd++) {
        if (!(par->bound_side & sd)) continue;
        /* 4.37: certificates from the tracked candidate (with its anchor) and from the
         * returned point, on the problem as read; the better one is the run's (per side) */
        BoundTrack *T = &bt[sd - 1];
        const double tb0 = wtime();
        if (par->bound_prior == 2) {
            /* -boundprior 2: the a-priori route first, on the returned point. Building a
             * certificate (projections and margin checks of the candidates, in extended
             * precision) can cost several times the solve; it is built only when this fails. */
            const PSOrig *Op = u->Ofin ? u->Ofin : u->O;
            double B; int which = 0; char w2[160];
            /* (side p: also the method's iterate before the restoration and the polishes, positive definite) */
            int nbp = 0;
            double **Xbp = sd == 1 ? bpt_mapped(u, par, Op, &nbp) : NULL;
            const int got = brisk_prior_bound(Op, sd, Xbp, sd == 1 ? u->Xo : NULL, NULL, sd == 2 ? u->yo : NULL,
                                              g_brisk_data != NULL, par->verbose, &B, &which, w2, sizeof(w2));
            if (got) {
                BoundCert *uc = &u->cert[sd - 1];
                bound_cert_free(uc);
                memset(uc, 0, sizeof(*uc));
                uc->side = sd; uc->have = 1; uc->valid = 1; uc->prior_first = 1;
                uc->value = B; uc->lammin = NAN; uc->value0 = NAN;
                { double **Xc = which == 1 && Xbp ? Xbp : u->Xo; if (Xc) (void)orig_cx(Op, Xc, &uc->resid); }
                uc->nblk = Op->nblk; uc->m = Op->m;
                /* the iterate stays as the bound's candidate: the certification below takes the a-priori route on it
                 * and on the returned point again (prior_first), the better one kept */
                if (Xbp && nbp == Op->nblk) { uc->X = Xbp; Xbp = NULL; }
                snprintf(uc->src, sizeof(uc->src), which == 1 ? "the method's iterate before the polishes (a-priori)" : "the returned point (a-priori)");
                uc->attempt = g_run_no;
                if (par->verbose > 0)
                    printf("   bound: %s from %s by a-priori bounds: %.10e (%.2fs; no certificate built)\n",
                           sd == 1 ? "upper" : "lower", which == 1 ? "the method's iterate before the polishes" : "the returned point", B, wtime() - tb0);
                if (Xbp) { for (int k = 0; k < nbp; k++) free(Xbp[k]); free(Xbp); }
                continue;
            }
            if (Xbp) { for (int k = 0; k < nbp; k++) free(Xbp[k]); free(Xbp); }
            if (par->verbose > 0) printf("   bound: a-priori bounds of the returned point: %s; building a certificate\n", w2);
        }
        double **Xt = NULL, *yt = NULL, **XA[BOUND_NANCHOR], *YA[BOUND_NANCHOR];
        int nXt = 0, nXA[BOUND_NANCHOR], na = 0;
        for (int s = -1; s < T->na; s++) {
            if (s < 0 && !T->have) continue;
            double **Xm = NULL, *ym = calloc(u->O->m + 1, sizeof(double));
            PSResult prc;
            postsolve_set_time_left(ps_time_left(par)); postsolve_set_tol(par->tol);
            postsolve(P, u->O, s < 0 ? T->X : T->Xa[s], s < 0 ? T->y : T->ya[s], &Xm, ym, &prc, 0, NULL, NULL);
            int nX = u->O->nblk;
            map_back(u, par, &Xm, &ym, &nX, 0, NULL);
            if (s < 0) { Xt = Xm; yt = ym; nXt = nX; }
            else { XA[na] = Xm; YA[na] = ym; nXA[na] = nX; na++; }
        }
        PSOrig *Of = u->Ofin ? u->Ofin : u->O;
        if (par->bound_anchor && sd == 2 && na < BOUND_NANCHOR) {
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
        snprintf(src, sizeof(src), "iteration %d", T->it);
        { extern double g_bound_deadline; g_bound_deadline = par->timelimit > 0 ? par->t_start + par->timelimit : 0; }
        if (T->have) bound_certify(Of, sd, Xt, yt, na, XA, YA, par->bound_margin, par->bound_tol, src, par->verbose, &c1);
        bound_certify(Of, sd, u->Xo, u->yo, na, XA, YA, par->bound_margin, par->bound_tol, "the returned point", par->verbose, &c2);
        BoundCert *uc = &u->cert[sd - 1];
        {
            /* the candidate before its margin was restored goes with the kept certificate when it
             * is the better of the two: the a-priori route (main.c) may certify it */
            BoundCert *win = bound_better(&c1, &c2) ? &c1 : &c2, *lose = win == &c1 ? &c2 : &c1;
            const int lhas = sd == 1 ? lose->X0 != NULL : lose->y0 != NULL, whas = sd == 1 ? win->X0 != NULL : win->y0 != NULL;
            const double wv = whas ? win->value0 : win->value;
            if (lhas && win->have && (sd == 1 ? lose->value0 < wv : lose->value0 > wv)) {
                bound_free_blocks(win->X0, win->nblk); free(win->y0);
                win->X0 = lose->X0; win->y0 = lose->y0; win->value0 = lose->value0;
                lose->X0 = NULL; lose->y0 = NULL;
            }
            *uc = *win; bound_cert_free(lose);
        }
        uc->attempt = g_run_no;
        if (par->verbose > 0 && uc->have)
            printf("   bound: %s from %s: %.10e (residual %.1e, lambda_min %.1e rel; %d candidates offered, %d anchors, %.2fs)\n",
                   sd == 1 ? "upper" : "lower", uc->src, uc->value, uc->resid, uc->lammin, T->offers, na, wtime() - tb0);
        bound_free_blocks(Xt, nXt); free(yt);
        for (int a2 = 0; a2 < na; a2++) { bound_free_blocks(XA[a2], nXA[a2]); free(YA[a2]); }
    }
    for (int sd = 0; sd < 2; sd++) {
        BoundTrack *T = &bt[sd];
        if (!T->X) continue;
        bound_free_blocks(T->X, T->nblk); free(T->y);
        for (int s = 0; s < BOUND_NANCHOR; s++) { bound_free_blocks(T->Xa[s], T->nblk); free(T->ya[s]); }
    }
    return 0;
}

static const Params *g_rank_par = NULL;     /* 4.37: bound mode: the certificates follow the run that is kept */
/* ranking of a finished run: lower is better.
 * 24 October 2026: the runs that end with a point and without a verdict (reduced accuracy, numerical difficulties, the
 * iteration or the time limit) form one class, ordered by the run's error. A status of that class says why the run
 * stopped, the error how good its point is; the statuses follow the error on the problem as read (OPTIMAL at the
 * tolerance, REDUCED at the reduced-accuracy level, else the reason it stopped), except where a status is kept for what it
 * reports: a reduced solve whose facial-reduction duals were not rebuilt is REDUCED at its dual error, up to 1.0. Ranked
 * below the time limit's class, it was kept against the re-solve without the reduction that the time cap stopped at a
 * better point (pop_wood_1000, the owner's benchmark, SDP, -frfallback 4: 1.0 returned against 1.6e-6). */
static double run_rank(const Run *u) {
    switch (u->status) {
    case ST_OPTIMAL: return 0 + fmin(u->acc, 1.0);
    case ST_PINFEAS: case ST_DINFEAS: return 0.5;
    case ST_REDUCED: case ST_NUMERIC: case ST_MAXIT: case ST_TIME: return 1 + fmin(u->acc, 1.0);
    default: return 2 + fmin(u->acc, 1.0);
    }
}

/* 5.8: the events of a run (method switches, the endgame, crossover, bounds, ...) for the
 * summary at the end of the log; cleared by brisk_main */
#include <stdarg.h>
static char g_notes[24][512]; static int g_nnotes = 0;
static int g_note_diag[24];   /* (ISSUES 725) a note of the solver's internal steps: in the Summary of the diagnosis (-v) only */
void brisk_note(const char *fmt, ...) {
    if (g_nnotes >= 24) return;
    g_note_diag[g_nnotes] = 0;
    va_list ap; va_start(ap, fmt); vsnprintf(g_notes[g_nnotes], sizeof g_notes[0], fmt, ap); va_end(ap);
    /* (24 October 2026, the SOCP developer) a note equal to one already kept is not kept again: a second attempt of the
     * SOCP solver repeats its presolve, and the Summary listed "free variables eliminated by substitution" twice */
    for (int i = 0; i < g_nnotes; i++) if (!strcmp(g_notes[i], g_notes[g_nnotes])) return;
    g_nnotes++;
}
static void notes_clear(void) { g_nnotes = 0; }
void brisk_note_diag(const char *fmt, ...) {
    if (g_nnotes >= 24) return;
    char t[512]; va_list ap; va_start(ap, fmt); vsnprintf(t, sizeof t, fmt, ap); va_end(ap);
    const int k = g_nnotes; brisk_note("%s", t);
    if (g_nnotes > k) g_note_diag[k] = 1;
}
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
#ifndef BRISK_LIBRARY
#define g_caller_depth 0           /* (the command line: no library call around the run) */
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
/* 18 October 2026: the first-order engine ran out of memory (its Gram matrix: the large AC-OPF relaxations,
 * whose chordal extension is beyond the interior-point method's clique limit, ended with the bare message);
 * the low-rank augmented Lagrangian forms nothing of size n^2 or m^2: say so, in the caller's syntax */
void brisk_fom_oom_advice(void) {
    printf("The first-order engine needs more memory than there is. For a large sparse semidefinite block with a\n"
           "low-rank solution (relaxations of AC optimal power flow, for example) the low-rank augmented Lagrangian\n"
           "forms nothing of size n^2 or m^2%s:\n  %s   (experimental; the objectives, errors and y, no X)\n", syn_head(), syn_kv("lralm", "1", 0));
    fflush(stdout);
}
/* 5.9: the last lines of a standard-precision log say how to get more accuracy; 5.10: in the
 * caller's syntax, and how to get a guaranteed bound (also at the end of a high-precision log).
 * Each line is an option as it is typed, in a column, and what it does. */
static void bound_hint(int hp) {
    const char *cmd; const int k = caller_kind(&cmd);
    const int model = k == CK_CVXPY || k == CK_JUMP;      /* the modelling layers name the sides of the model */
    const char *kc = syn_kv("certify", NULL, 0);
    const char *head = hp ? syn_head() : "";              /* (after the lines on accuracy the place has just been named) */
    if (g_has_bound) return;          /* (the bound is checked rigorously whenever it is asked for) */
    const char *kb = syn_kv("bound", model ? "dual" : "d", 1);
    const int w = (int)strlen(kb);   /* (4 November 2026, the owner: "a lower bound" too far from 'd': the width no longer counts
                                         the certify option, which is not printed since 31 October) */
    (void)kc;
    /* (31 October 2026, the owner: the log names the option and what it gives; how it works is in the options' text) */
    (void)hp;
    printf("For a guaranteed bound on the optimal value%s:\n", head);
    if (model) printf("  %-*s   a lower bound (%s: an upper bound)\n", w, kb, syn_kv("bound", "primal", 1));
    else printf("  %-*s   a lower bound (%s: an upper bound)\n", w, kb, syn_kv("bound", "p", 1));
    (void)cmd;
}
static void resolve_hint(void);
static void accuracy_hint(int lp) {
    if (lp) { printf("For higher accuracy%s:\n  %s   the tolerance (default 1e-8)\n", syn_head(), syn_kv("tol", "<t>", 0)); return; }
    const char *ka = syn_kv("acc", "high", 1), *kp = syn_kv("prec", "dd", 1);
    int w = (int)strlen(kp);
    if (!g_acc_high && (int)strlen(ka) > w) w = (int)strlen(ka);
    printf("For higher accuracy%s:\n", syn_head());
    if (!g_acc_high) printf("  %-*s   tolerance 1e-10\n", w, ka);
    printf("  %-*s   the high-precision solver, about 31 digits (%s: about 63; a number: that many digits)\n", w, kp, syn_val("qd"));
    bound_hint(0);
}
/* 21 October 2026 (the owner): a solve that misses the tolerance on a problem whose blocks' factors spread by 4 or more
 * (where -blkscale -2 acts) names the option; the scaled solve plus the solve as given costs up to twice the time, and on
 * the sets it gains as often as not, so it is not a default (REPORT_5.21 §5) */
static void blkscale_hint(int status) {
    if (status != ST_REDUCED && status != ST_NUMERIC) return;   /* (not at an iteration or time limit) */
    if (g_bs_probe < g_blkscale_min) return;
    printf("The blocks are badly scaled (their scaling factors spread by %.3g)%s:\n"
           "  %s   scales the blocks by the size of their data\n",
           g_bs_probe, syn_head(), syn_kv("blkscale", "-2", 0));
}
/* 31 October 2026 (ISSUES 724): a result of the problem with its free variables eliminated that ended just above 1e-6 is
 * returned without solving the problem as read again; the option that solves it as read is named */
static int g_fe_band = 0;
static void freeelim_hint(int status) {
    if (!g_fe_band || (status != ST_REDUCED && status != ST_NUMERIC)) return;
    const char *k = syn_kv("freeelim", "0", 0);
    /* (ISSUES 725, the owner: say what the presolve did, not a condition the reader has to check) */
    printf("The presolve eliminated the %d free variables by substitution. To keep them%s:\n"
           "  %s   keeps the free variables (may end more accurately)\n", g_fe_band, syn_head(), k);
}
static void decision_hint(void) {
    extern double g_t_all_note, g_dec_sym_t, g_dec_fr_t;
    const double t = g_t_all_note;
    if (!(t > 0)) return;
    const char *ks = syn_kv("sym", "none", 1), *kf = syn_kv("nofr", NULL, 0);
    /* (5 % of the run, and at least 0.1 s: a search of milliseconds on a tiny problem is not worth an option) */
    const int ws = g_dec_sym_t > 0.05 * t && g_dec_sym_t >= 0.1, wf = g_dec_fr_t > 0.05 * t && g_dec_fr_t >= 0.1;
    if (!ws && !wf) return;
    const int w = (int)((ws && wf) ? (strlen(ks) > strlen(kf) ? strlen(ks) : strlen(kf)) : ws ? strlen(ks) : strlen(kf));
    printf("The presolve spent %.1f s (%.0f %% of the time) on decisions that led to nothing. To skip them%s:\n",
           (ws ? g_dec_sym_t : 0) + (wf ? g_dec_fr_t : 0), 100.0 * ((ws ? g_dec_sym_t : 0) + (wf ? g_dec_fr_t : 0)) / t, syn_head());
    if (ws) printf("  %-*s   no symmetry search (it found none: %.1f s)\n", w, ks, g_dec_sym_t);
    if (wf) printf("  %-*s   no facial reduction (it was undone as not worth its cost: %.1f s)\n", w, kf, g_dec_fr_t);
}
static int quiet_summary = 0;   /* BRISK_NOSUMMARY=1: no summary block (a log parsed by tools) */
/* 4.37 (B1): the re-solves of a run, for the note at the end of the log */
/* 20 October 2026 (the owner): each re-solve also carries the option that selects its method (ok, okv: a flag when okv is
 * NULL), for the advice when its result is returned, and the one that keeps the first attempt's method or form (nk, nkv),
 * for the advice when the first attempt's result is returned; g_first_name names the first attempt when it is known */
typedef struct { const char *name; double t, acc; int kept, before; const char *ok, *okv, *nk, *nkv, *ok2, *okv2; } Attempt;   /* ok2: a second option the method needs (23 October 2026: -soc 0); before: an attempt that ran before the first solve (23 October 2026: the matrix-free and first-order attempts, listed first) */
static Attempt g_att[32];
static int g_natt = 0, g_first_internal = 0;
static int g_att_res = 0;   /* (23 October 2026) the entry att_log made for the result its run holds: the callers mark it kept */
static double g_first_t = 0, g_first_acc = 0;
static const char *g_first_name = NULL;
static int g_fr_dual_lost = 0;   /* the constraints removed by the facial reduction whose duals were not rebuilt (20 October 2026) */
/* 4 November 2026 (the owner: "if there is more than 5% of the total solve time spent in decision making in the presolve (eg
 * symmetry search or cost analysis of methods), give user the advice to force whichever decision was taken (eg no symmetry
 * for charlie's problem)"): the time of the presolve's decisions whose outcome an option can force */
double g_dec_sym_t = 0;   /* symmetry searches that found nothing (-sym none skips them) */
double g_dec_fr_t = 0;    /* facial reductions made and then undone by the cost rule (-nofr skips them) */
double g_t_all_note = 0;   /* the run's total time, for decision_hint */
static char g_amp_warn[480];     /* 2 November 2026 (ISSUES 746): the objective's sensitivity to small errors (amp_warning) */
static void att_log(const char *name, double t, const void *run);
static void att_opt(const char *ok, const char *okv, const char *nk, const char *nkv);
/* 20 October 2026 (the owner): after a re-solve, the option that gives the returned result directly, in the caller's syntax:
 * the option of the re-solve whose result is returned, else the options that keep the first attempt's method or form */
static void resolve_hint(void) {
    if (g_natt <= 0) return;
    const Attempt *k = NULL;
    for (int a = 0; a < g_natt; a++) if (g_att[a].kept) k = &g_att[a];
    const char *opt[8]; int n = 0, before[8] = { 0 };
    if (k && k->ok && k->ok2) {             /* two options on one line, in the caller's syntax */
        char *o = syn_buf(); const char *cmd; const int ck = caller_kind(&cmd);
        snprintf(o, SYN_L, "%s%s%s", syn_kv(k->ok, k->okv, 0), ck == CK_CLI ? " " : ck == CK_ML || ck == CK_JUMP ? "; " : ", ", syn_kv(k->ok2, k->okv2, 0));
        opt[n++] = o;
    } else if (k) { if (k->ok) opt[n++] = syn_kv(k->ok, k->okv, 0); }
    else for (int a = 0; a < g_natt && n < 8; a++) if (g_att[a].nk) {
        const char *o = syn_kv(g_att[a].nk, g_att[a].nkv, 0); int dup = 0;
        for (int b = 0; b < n; b++) if (!strcmp(opt[b], o)) dup = 1;
        if (!dup) { before[n] = g_att[a].before; opt[n++] = o; }
    }
    if (!n) return;
    int w = 0; for (int i = 0; i < n; i++) if ((int)strlen(opt[i]) > w) w = (int)strlen(opt[i]);
    printf("For a faster solve without the re-solve%s:\n", syn_head());
    for (int i = 0; i < n; i++)
        /* (24 October 2026: an attempt made before the solve whose result was returned, the matrix-free or the
         *  Burer-Monteiro method, is skipped by its option: the result's method then runs directly; until then the line
         *  said "keeps the first attempt", which in the Summary is that attempt) */
        printf("  %-*s   %s\n", w, opt[i], k || before[i] ? "solves by the method that gave the result, directly" : "keeps the first attempt, whose result was returned");
}

/* the same problem as read: a certificate of one run is one of the other */
static int ps_same(const PSOrig *a, const PSOrig *b) {
    if (a == b) return 1;
    if (!a || !b || a->m != b->m || a->nblk != b->nblk || a->nnz != b->nnz || a->tbR != b->tbR) return 0;
    if (memcmp(a->bs, b->bs, sizeof(int) * a->nblk) || memcmp(a->b, b->b, sizeof(double) * a->m)) return 0;
    return !memcmp(a->con, b->con, sizeof(int) * a->nnz) && !memcmp(a->blk, b->blk, sizeof(int) * a->nnz)
        && !memcmp(a->ii, b->ii, sizeof(int) * a->nnz) && !memcmp(a->jj, b->jj, sizeof(int) * a->nnz)
        && !memcmp(a->v, b->v, sizeof(double) * a->nnz);
}
/* 4.37: does the second run replace the first? By the returned solution (the usual ranking).
 * Bound mode: the better certificate of each side moves to the run that is kept, when both
 * runs refer to the same problem as read (the returned point is never a certificate) */
static int run_better(Run *a, Run *b) {
    const int r = run_rank(a) < run_rank(b);
    if (g_rank_par && g_rank_par->bound_side) {
        Run *w = r ? a : b, *l = r ? b : a;
        if (ps_same(w->Ofin ? w->Ofin : w->O, l->Ofin ? l->Ofin : l->O))
            for (int s = 0; s < 2; s++)
                if (bound_better(&l->cert[s], &w->cert[s])) { const BoundCert t = w->cert[s]; w->cert[s] = l->cert[s]; l->cert[s] = t; }
    }
    return r;
}

/* Were the duals of the constraints removed by the facial reduction not rebuilt (the dual error on the problem as read is
 * that of the removed rows), and if so, the run's error apart from them: the primal errors on the problem as read, the gap,
 * the complementarity, and the reduced solve's own dual errors. A re-solve that keeps the facial reduction does not aim at
 * those duals, so it is judged by this error (23 October 2026: taha1b, SDP, re-solved without the symmetry reduction for
 * a dual error of 1.4 that the facial reduction caused: 4.4 s for the same result). It can still change them, by where it
 * leaves the dual components the reduced problem does not constrain: acopf_case118 (SDP with SOCP), whose sixth re-solve
 * in 1.5.5 rebuilt them (ISSUES 255; whether to re-solve for them is the owner's decision). BRISK_FRKEEP=0: never. */
static int fr_duals_lost(const Run *u, const Params *par, int do_fr, double *acc_apart) {
    static int frkeep = -1;
    if (frkeep < 0) { const char *e = getenv("BRISK_FRKEEP"); frkeep = e ? atoi(e) != 0 : 1; }
    const int lost = frkeep && do_fr && u->fr_removed > 0 && u->measured && u->status != ST_OPTIMAL
        && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && u->pr.err[4] > fmax(100.0 * fmax(par->tol, 1e-8), 1e-3 * u->acc) && u->pr.err[4] >= 0.5 * u->acc;
    if (acc_apart) {
        const double dred = fmax(u->R.err[3], u->R.err[4]);
        *acc_apart = lost ? brisk_nmax(brisk_nmax(u->pr.err[1], u->pr.err[2]), brisk_nmax(dred, brisk_nmax(fabs(u->pr.err[5]), fabs(u->pr.err[6])))) : u->acc;
    }
    return lost;
}
static double acc_apart_fr(const Run *u, const Params *par, int do_fr) { double a; fr_duals_lost(u, par, do_fr, &a); return a; }

static void att_put(const char *name, double t, double acc) {
    if (g_natt >= 32) return;
    g_att[g_natt].name = name; g_att[g_natt].t = t; g_att[g_natt].acc = acc; g_att[g_natt].kept = 0; g_att[g_natt].before = 0;
    g_att[g_natt].ok = g_att[g_natt].okv = g_att[g_natt].nk = g_att[g_natt].nkv = g_att[g_natt].ok2 = g_att[g_natt].okv2 = NULL;
    g_att_res = g_natt++;
}
/* a re-solve, once: a run that repeated itself with the second-order cones as semidefinite blocks (run_pipeline) gives
 * both of its runs under the name, and g_att_res is the one whose result the run holds (until 23 October 2026 the inner
 * run was logged by itself and the run again under the caller's name: two entries for one solve, the first run missing) */
static void att_log(const char *name, double t, const void *run) {
    const Run *r = (const Run *)run;
    if (r && r->soc2_done && g_natt + 2 <= 32) {
        static char nm[32][200];                     /* (the slots' names, valid until the run's summary) */
        snprintf(nm[g_natt], sizeof nm[0], "%s, second-order cones as cones", name);
        att_put(nm[g_natt], r->soc2_t1, r->soc2_acc1); const int i1 = g_att_res;
        snprintf(nm[g_natt], sizeof nm[0], "%s, second-order cones as semidefinite blocks", name);
        att_put(nm[g_natt], r->soc2_t2, r->soc2_acc2); g_att[g_att_res].ok2 = "soc"; g_att[g_att_res].okv2 = "0";
        if (!r->soc2_kept) g_att_res = i1;
        return;
    }
    att_put(name, t, r ? r->acc : -1);
}
static void att_opt(const char *ok, const char *okv, const char *nk, const char *nkv) {
    if (g_natt <= 0 || g_natt > 32 || g_att_res < 0 || g_att_res >= g_natt) return;
    Attempt *a = &g_att[g_att_res]; a->ok = ok; a->okv = okv; a->nk = nk; a->nkv = nkv;
}


/* 4 November 2026 (the owner: "this must be made to work well"; ISSUES 755 and 757; bench/charlie_1104/DUAL_RESEARCH.md):
 * the dual after a facial reduction whose removed constraints' duals could not be rebuilt. The reduced solve gives X and
 * the optimal value v* correctly (to about 1e-9 on a user's moment relaxations); the problem as read has no strictly
 * feasible point, so its dual optimum need not be attained, and the level-by-level recovery diverges (multipliers of
 * 1e14). A second solve on the problem as read, without the facial reduction, finds the dual of least violation that
 * keeps the value:  max -s  s.t.  C - sum y_i A_i + s I >= 0 (every semidefinite block; the other blocks exact),
 * b'y >= v*. Its dual side has an interior (s free), so the interior-point method converges in 17-21 iterations where the
 * problem as read takes 45-53. In the form of the problem as read (min <C,X>, <A_i,X> = b_i): every A_i gains the entry
 * -b_i in a new nonnegative variable whose cost is -v*, a new constraint has -I on every semidefinite block and right-hand
 * side -1, the other right-hand sides are 0. Measured on the user's 2x2 and 3x3 grids: every DIMACS error <= 3e-8, the
 * largest moment 62 and 629, 9.5 s and 27.7 s after first solves of 2.2 s and 5.5 s (MOSEK 15.6 s and 36.5 s with
 * errors of 2.6e-5 and 9.8e-5). The phase's y replaces the run's when its errors on the problem as read are smaller. */
static int run_resolve(const char *fname, Params *par, int do_fr, int force_route, Run *u);
static int fr_dual_phase(Run *u, const Params *par, int force_route, double *t_used) {
    *t_used = 0;
    int kept = 0;
    const PSOrig *O = u->Ofin ? u->Ofin : u->O;
    if (!O || !u->Xo || !u->yo) return 0;
    const int m = O->m, nb = O->nblk;
    /* the value to hold: <C,X> of the first solve (its X is feasible to its primal errors, so this bounds the optimum from above;
     * b'y of a lost dual bounds nothing), less half the tolerance, so that the gap of the pair ends inside the tolerance */
    const double vstar = u->pr.pobj - 0.5 * par->tol * (1.0 + fabs(u->pr.pobj + O->obj_off));
    size_t nd = 0, nbz = 0;
    for (int k = 0; k < nb; k++) if (O->bs[k] > 0) nd += (size_t)O->bs[k];
    for (int i = 0; i < m; i++) if (O->b[i] != 0) nbz++;
    if (nd == 0) return 0;
    const size_t nn = O->nnz + 1 + nbz + nd;
    int *mat = malloc(sizeof(int) * nn), *blk = malloc(sizeof(int) * nn), *ii = malloc(sizeof(int) * nn), *jj = malloc(sizeof(int) * nn);
    double *v = malloc(sizeof(double) * nn), *c = calloc((size_t)m + 2, sizeof(double));
    int *bs = malloc(sizeof(int) * ((size_t)nb + 1));
    size_t q2 = 0;
    for (size_t q = 0; q < O->nnz; q++) {          /* the problem as read: F0 = -C, F_i = A_i, 1-based */
        mat[q2] = O->con[q] < 0 ? 0 : O->con[q] + 1; blk[q2] = O->blk[q] + 1; ii[q2] = O->ii[q] + 1; jj[q2] = O->jj[q] + 1;
        v[q2] = O->con[q] < 0 ? -O->v[q] : O->v[q]; q2++;
    }
    for (int k = 0; k < nb; k++) bs[k] = O->bs[k];
    bs[nb] = -1;                                    /* the value's nonnegative variable: b'y - v* >= 0 */
    if (vstar != 0) { mat[q2] = 0; blk[q2] = nb + 1; ii[q2] = jj[q2] = 1; v[q2] = vstar; q2++; }
    for (int i = 0; i < m; i++) if (O->b[i] != 0) { mat[q2] = i + 1; blk[q2] = nb + 1; ii[q2] = jj[q2] = 1; v[q2] = -O->b[i]; q2++; }
    for (int k = 0; k < nb; k++) if (O->bs[k] > 0)  /* s: -I on every semidefinite block */
        for (int t = 1; t <= O->bs[k]; t++) { mat[q2] = m + 1; blk[q2] = k + 1; ii[q2] = jj[q2] = t; v[q2] = -1.0; q2++; }
    c[m] = -1.0;                                    /* max -s */
    BriskData d = { m + 1, nb + 1, bs, c, q2, mat, blk, ii, jj, v };
    Params p2 = *par;
    p2.fr_dual = 0; p2.fr_fallback = 0; p2.known_feasible = 1; p2.bound_side = 0;
    if (!u->sym) p2.symfile = NULL;   /* (the first solve's search found no symmetry: the phase's problem, built from it, has none to find) */
    if (par->timelimit > 0) { const double left = par->timelimit - (wtime() - par->t_start); if (left < 1) goto done0; p2.timelimit = wtime() - par->t_start + left; }
    if (par->verbose >= 0) printf("postsolve: solving for the dual on the problem as read with the optimal value %.10e held (-frdual)\n", vstar);
    const double t0 = wtime();
    const BriskData *sd = g_brisk_data;
    g_brisk_data = &d;
    Run second;
    const int rc = run_resolve(g_brisk_data_name, &p2, 0, force_route, &second);
    g_brisk_data = sd;
    *t_used = wtime() - t0;
    if (rc == 0 && second.yo && second.Ofin && second.Ofin->m == m + 1) {
        PSResult pr;
        ps_measure_xc(O, u->Xo, second.yo, &pr, u->xc_o);
        const double acc = brisk_acc_err(pr.err);
        att_log("the dual on the problem as read with the optimal value held", *t_used, &second);
        g_att[g_att_res].acc = acc;
        if (acc < u->acc) {
            g_att[g_att_res].kept = 1;
            memcpy(u->yo, second.yo, sizeof(double) * (size_t)m);
            pr.recovered = u->pr.recovered; pr.unrecovered = u->pr.unrecovered; pr.fr_corrected = u->pr.fr_corrected; pr.t = u->pr.t;
            u->pr = pr; u->acc = acc;
            u->status = acc <= par->tol ? ST_OPTIMAL : acc <= par->red_acc ? ST_REDUCED : ST_NUMERIC;
            kept = 1;
            if (par->verbose >= 0) printf("  the dual of the second solve is kept: DIMACS error %.1e on the problem as read\n", acc);
        } else if (par->verbose >= 0) printf("  the second solve's dual is not better (DIMACS error %.1e); the first solve is returned\n", acc);
    } else if (par->verbose >= 0) printf("  the second solve for the dual did not finish; the first solve is returned\n");
    if (rc == 0 || second.Xo) run_free(&second);
done0:
    free(mat); free(blk); free(ii); free(jj); free(v); free(c); free(bs);
    return kept;
}

/* 5.16: a run whose solve had semidefinite blocks with arrow data as second-order cones (solver.c,
 * -soc) and ended worse than reduced accuracy is repeated with those blocks as semidefinite blocks,
 * the better answer kept: two exact forms of the problem, and the cone form, which converges further,
 * can leave the dual recovery after a facial reduction without margin (an AC-OPF form with cone line
 * limits, REPORT_5.16 §9). The other stages of the run (chordal forms, facial reduction) go through
 * this too. */
static double g_pipe_extra = 0;   /* time of the runs this wrapper discarded (added to the run's total) */
/* 27 October 2026, the night: the memory a solve needs with every block of the problem as read as a dense semidefinite
 * block (the cones solved as cones included): the n x n arrays brisk_solve allocates for each (eleven for the iteration:
 * X, Z, their factors, Z's inverse, the two directions, four work arrays; the residual R0; the correctors' two; the best
 * iterate's two kept for the polish), sixteen, the NT direction's six more not counted. For the re-solves that cannot
 * keep a cone as a cone (r2_port_n11000_k60, SDP with SOCP, a cone of order 11,002: 15.5 GB; eight arrays, the first
 * count, let its re-solve start and the process was killed) */
/* 28 October 2026, the night: the run stopped at the user's iteration limit (params_maxit_is_budget): no re-solve */
static int maxit_spent(const Run *u, const Params *par) { return u->status == ST_MAXIT && params_maxit_is_budget(par); }
extern int g_soc_used;
static double sdp_form_bytes(const PSOrig *O) {
    double need = 0;
    for (int k = 0; k < O->nblk; k++) if (O->bs[k] > 0) need += 16.0 * 8.0 * (double)O->bs[k] * (double)O->bs[k];
    return need;
}
static int run_pipeline(const char *fname, Params *par, int do_fr, int force_route, Run *u) {
    extern int g_soc_used;
    g_soc_used = 0;
    /* 25 October 2026 (ISSUES 290): a call without a result (a request: 3, 4, 6, 7, 8) is discarded by its caller, which
     * runs the pipeline again; the time it spent is the run's (the matrix-free method or the first-order engine tried
     * first: the first presolve; pglib_opf_case4661_sdet: 2 s before "restarting with the first-order engine") */
    const double t_call = wtime();
    const int rc = run_pipeline_core(fname, par, do_fr, force_route, u);
    if (rc != 0) g_pipe_extra += wtime() - t_call;
    if (rc != 0 || !g_soc_used || !par->soc || !u->measured) return rc;
    if (u->status == ST_OPTIMAL || u->status == ST_REDUCED || u->status == ST_PINFEAS || u->status == ST_DINFEAS) return rc;
    if (brisk_stop_flag || maxit_spent(u, par) || (par->timelimit > 0 && wtime() - par->t_start >= 0.5 * par->timelimit)) return rc;
    /* 27 October 2026, the night: not when the semidefinite form cannot fit in memory: every block of the problem as read
     * as a dense block (sdp_form_bytes). A cone of order 45,001 (r2_reglmi_N45k, SDP with SOCP, cut by an iteration
     * limit) needs 16 GB an array: the re-solve ran out of memory, and with the system's overcommit the process was
     * killed, losing the first solve's result. (The readers' cap of 46,340 only bounds the index of an entry.) */
    if (u->O) {
        const double need = sdp_form_bytes(u->O);
        if (need > 0.6 * brisk_mem_limit()) {
            if (par->verbose >= 0) printf("second-order cones: the result is %.1e; with the cones as semidefinite blocks the solve would need %.0f GB: not re-solved\n", u->acc, need / 1e9);
            return rc;
        }
    }
    Params p2 = *par;
    p2.soc = 0;
    if (par->verbose >= 0) printf("second-order cones: the result is %.1e, solving with the cones as semidefinite blocks as well\n", u->acc);
    Run second;
    const double t_second = wtime();
    if (run_pipeline_core(fname, &p2, do_fr, force_route, &second) != 0) { g_pipe_extra += wtime() - t_second; return rc; }
    const double t1 = u->R.t_total + u->tpre + u->pr.t, t2 = second.R.t_total + second.tpre + second.pr.t;
    const double acc1 = u->acc, acc2 = second.acc;
    const int better = run_better(&second, u);
    if (better) {
        if (par->verbose >= 0) printf("  the new solve is better, keeping it\n");
        second.R.iters += u->R.iters;
        g_pipe_extra += t1 + u->tread;
        run_free(u);
        *u = second;
    } else {
        if (par->verbose >= 0) printf("  the new solve is not better, keeping the previous one\n");
        u->R.iters += second.R.iters;
        g_pipe_extra += t2 + second.tread;
        run_free(&second);
    }
    u->soc2_done = 1; u->soc2_kept = better;
    u->soc2_t1 = t1; u->soc2_acc1 = acc1; u->soc2_t2 = t2; u->soc2_acc2 = acc2;
    return rc;
}

/* 22 October 2026 (the owner: "This seems like the problem of the safeguard!! Why did it take it for a failure?").
 * run_pipeline returns a result (0) or a request: 3 solve again without facial reduction, 4 the first-order engine
 * (the Schur complement does not fit), 6 the first-order engine first, 7 the matrix-free method first, 8 the symmetry
 * search again with its full budget; 2 (the file) and 5 (a re-solve declined) give no result. Only the first solve
 * served the requests: every re-solve read any code but 0 as a failure and kept the first answer
 * (e_moment_quadknap_17_100: the check of a presolved "dual infeasible" on the problem as read got 7, and the false
 * verdict stood; ISSUES 232). Every re-solve goes through here: the first attempts of other methods (6, 7, 8) belong
 * to the first solve, so a re-solve goes on without them; 3 and 4 are served as in the first solve. */
static int run_resolve(const char *fname, Params *par, int do_fr, int force_route, Run *u) {
    int rc = run_pipeline(fname, par, do_fr, force_route, u);
    for (int pass = 0; pass < 4 && rc != 0 && rc != 2 && rc != 5; pass++) {
        if (rc == 3) do_fr = 0;
        else if (rc == 4) {
            par->fom = 1;
            if (par->free_elim < 0) par->free_elim = 0;
            if (par->dualize < 0) par->dualize = 0;
            if (par->tracebound < 0) par->tracebound = 0;
            if (par->chordal < 0) par->chordal = 0;
        }
        else if (rc == 6) { par->fom_race = 0; par->fom_bound_maxit = -1; }
        else if (rc == 7 || rc == 8) par->mf_try = 0;
        else break;
        run_free(u);
        rc = run_pipeline(fname, par, do_fr, force_route, u);
    }
    return rc;
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
/* 20 October 2026 (the owner): the sizes as a user wrote the problem: equality constraints, semidefinite blocks, nonnegative
 * variables, and free variables (a free variable arrives as the difference of two nonnegative ones; nfree such pairs) */
static void summary_size(char *b, size_t n, const PSOrig *O, int nfree) {
    int nsdp = 0, nlp = 0, maxn = 0;
    for (int k = 0; k < O->nblk; k++) { if (O->bs[k] > 0) { nsdp++; if (O->bs[k] > maxn) maxn = O->bs[k]; } else nlp += -O->bs[k]; }
    nlp -= 2 * nfree; if (nlp < 0) nlp = 0;
    char cm[32], cl[32]; fmt_count(cm, sizeof cm, O->m); fmt_count(cl, sizeof cl, nlp);
    size_t w = (size_t)snprintf(b, n, "%s equality constraint%s", cm, O->m == 1 ? "" : "s");
    if (nsdp && w < n) w += (size_t)snprintf(b + w, n - w, ", %d semidefinite block%s (largest of order %d)", nsdp, nsdp > 1 ? "s" : "", maxn);
    if (nlp && w < n) w += (size_t)snprintf(b + w, n - w, ", %s nonnegative variable%s", cl, nlp > 1 ? "s" : "");
    if (nfree && w < n) snprintf(b + w, n - w, ", %d free variable%s", nfree, nfree > 1 ? "s" : "");
}
/* what a presolve step changed: " (equality constraints 31,465 -> 8,425; semidefinite blocks 1 -> 4, largest 406 -> 136)" */
static void summary_diff(char *d, size_t n, const PSOrig *a, const PSOrig *b) {
    int na = 0, nb = 0, la = 0, lb = 0, ma = 0, mb = 0;
    for (int k = 0; k < a->nblk; k++) { if (a->bs[k] > 0) { na++; if (a->bs[k] > ma) ma = a->bs[k]; } else la += -a->bs[k]; }
    for (int k = 0; k < b->nblk; k++) { if (b->bs[k] > 0) { nb++; if (b->bs[k] > mb) mb = b->bs[k]; } else lb += -b->bs[k]; }
    char s1[32], s2[32]; size_t w = 0; d[0] = 0;
    if (a->m != b->m) { fmt_count(s1, sizeof s1, a->m); fmt_count(s2, sizeof s2, b->m); w += snprintf(d + w, n - w, "%sconstraints %s -> %s", w ? ", " : ": ", s1, s2); }
    if (na != nb) w += snprintf(d + w, n - w, "%sblocks %d -> %d", w ? ", " : ": ", na, nb);
    if (ma != mb) w += snprintf(d + w, n - w, "%slargest block %d -> %d", w ? ", " : ": ", ma, mb);
    if (la != lb) { fmt_count(s1, sizeof s1, la); fmt_count(s2, sizeof s2, lb); w += snprintf(d + w, n - w, "%snonnegative variables %s -> %s", w ? ", " : ": ", s1, s2); }
    (void)w;
}
/* ISSUES 725: a paragraph of the Summary wrapped at the given width, every line after the indent */
static void print_wrapped(const char *indent, const char *text, int width) {
    const int ind = (int)strlen(indent), room = width - ind > 20 ? width - ind : 20;
    const char *p = text;
    while (*p) {
        while (*p == ' ') p++;
        int n = (int)strlen(p);
        if (n > room) { n = room; while (n > 0 && p[n] != ' ') n--; if (n == 0) n = room; }
        printf("%s%.*s\n", indent, n, p);
        p += n;
    }
}
int g_log_nmethod = 0;   /* (the methods begun in this run: from the second on, "Method (solve k):") */
void brisk_log_method(const char *fmt, ...) {
    if (!g_log_brief) return;
    char t[400]; va_list ap; va_start(ap, fmt); vsnprintf(t, sizeof t, fmt, ap); va_end(ap);
    if (g_log_nmethod++ == 0) log_user("Method: %s\n", t);
    else log_user("Method (solve %d): %s\n", g_log_nmethod, t);
}
void brisk_log_head(int ipm) {
    if (ipm) log_user(" ITE  PRIMAL OBJ        DUAL OBJ           PFEAS    DFEAS    GAP      MU       STEP   TIME\n");
    else log_user(" ITE  PRIMAL OBJ        DUAL OBJ           PFEAS    DFEAS    GAP      TIME\n");
}
void brisk_log_row(int it, double po, double dob, double pinf, double dinf, double gap, double mu, double step, double t) {
    if (mu >= 0) log_user("%4d  %+.8e  %+.8e   %.1e  %.1e  %.1e  %.1e  %.3f %6.2f\n", it, po, dob, pinf, dinf, gap, mu, step, t);
    else log_user("%4d  %+.8e  %+.8e   %.1e  %.1e  %.1e %7.2f\n", it, po, dob, pinf, dinf, gap, t);
}
/* the size of a problem as read (a Problem of the reader, before any presolve step) */
void brisk_log_problem(char *b, size_t n, const void *Pv) {
    const Problem *P = (const Problem *)Pv;
    int nsdp = 0, nlp = 0, maxn = 0;
    for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_LP) nlp += P->blk[k].n; else { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; } }
    char cm[32], cl[32]; fmt_count(cm, sizeof cm, P->m); fmt_count(cl, sizeof cl, nlp);
    size_t w = (size_t)snprintf(b, n, "%s equality constraint%s", cm, P->m == 1 ? "" : "s");
    if (nsdp && w < n) w += (size_t)snprintf(b + w, n - w, ", %d semidefinite block%s (largest of order %d)", nsdp, nsdp > 1 ? "s" : "", maxn);
    if (nlp && w < n) snprintf(b + w, n - w, ", %s nonnegative variable%s", cl, nlp > 1 ? "s" : "");
}
void brisk_log_summary(const char *problem, const char *solve, double t, const char *status, double acc, const char *cause,
                       double pobj, double dobj, const double *err, const char *extra) {
    g_log_user++;
    printf("------------------------------------------------------------------------------------------\n");
    printf("Summary\n  Problem as read:   %s\n", problem);
    printf("  Solve:             %s\n", solve);
    if (extra && extra[0]) { char x[800]; snprintf(x, sizeof x, "%s", extra); print_wrapped("                     ", x, 100); }
    printf("  Time:              %.2f s\n", t);
    { const int nt = brisk_threads_logged(); if (nt > 0) printf("  Number of threads: %d%s\n", nt, brisk_threads_label(nt)); }
    if (!strcmp(status, "OPTIMAL")) printf("  Status:            %s\n", status);
    else printf("  Status:            %s (DIMACS error %.1e)\n", status, acc);
    if (cause && cause[0]) { char c[600]; snprintf(c, sizeof c, "likely cause: %s", cause); print_wrapped("                     ", c, 100); }
    printf("  Objective:         primal <C,X> = %.10e, dual b'y = %.10e\n", pobj, dobj);
    if (err) printf("  DIMACS errors:     pinf %.1e, X>=0 %.1e, dinf %.1e, Z>=0 %.1e, gap %.1e, compl %.1e\n", err[1], err[2], err[3], err[4], err[5], err[6]);
    brisk_out_summary_line();
    printf("Solved to DIMACS error %.1e\n", acc);
    g_log_user--;
}
/* ISSUES 725: the presolve steps that changed the problem, as the user wrote it; before the iterations (the default log's
 * "Presolve:" lines) and in the Summary ("Preprocessing:") */
/* 1 November 2026 (the owner): the constraints a block's Schur complement is formed from through their low-rank factors
 * (the low-rank method, or the dictionary of their factors' columns); returns the number of such blocks */
static int lowrank_used(const Problem *PP, long *ncl, long *rtot) {
    int nbl = 0; *ncl = 0; *rtot = 0;
    char *seen = calloc((size_t)PP->m + 1, 1);   /* (a constraint counted once, however many blocks it touches) */
    for (int k = 0; k < PP->nblk; k++) {
        const Block *B = &PP->blk[k];
        if (B->lowrank) { nbl++; *rtot += B->lrR; for (int t = 0; t < B->ncon; t++) if (seen && !seen[B->con[t]]) { seen[B->con[t]] = 1; (*ncl)++; } }
        else if (B->dict && B->dstruct > 0) { nbl++; *rtot += B->dnv; *ncl += B->dstruct; }   /* (the block's constraints given by their factors) */
    }
    free(seen);
    return nbl;
}
static void print_preprocessing(const Run *u, const char *lab, const char *cont) {
    const Problem *PP = &u->P;
    /* the presolve chain: each stage's PSOrig is the problem before that step */
    const PSOrig *chain[8]; const char *what[8]; int nc = 0;
    if (u->Oaff) { chain[nc] = u->Oaff; what[nc++] = u->maff ? "moment relaxation over an ellipsoid: the coordinates of the unit ball" : "diagonal scaling of the semidefinite blocks"; }
    if (u->Osign) { chain[nc] = u->Osign; what[nc++] = "symmetry reduction (sign changes)"; }
    if (u->Osym) { chain[nc] = u->Osym; what[nc++] = "symmetry reduction"; }
    if (u->Oalg) { chain[nc] = u->Oalg; what[nc++] = "symmetry reduction (block diagonalisation)"; }
    if (u->Odual) { chain[nc] = u->Odual; what[nc++] = "dual form solved"; }
    if (u->Ofile) { chain[nc] = u->Ofile; what[nc++] = "free variables eliminated by substitution"; }
    if (u->Otb) { chain[nc] = u->Otb; what[nc++] = "trace-bound row added"; }
    int npre = 0;
    for (int c = 0; c < nc; c++) {
        const PSOrig *before = chain[c], *after = c + 1 < nc ? chain[c + 1] : u->O;
        char d[400]; summary_diff(d, sizeof d, before, after);
        if (strstr(what[c], "trace")) {   /* (a safeguard of the solve, not a change of the problem: the diagnosis's) */
            if (g_log_brief) continue;
            printf("%s%s\n", npre ? cont : lab, what[c]);
        }
        else if (before == u->Ofile && u->fe) {   /* the free variables as the user wrote them, not the split pairs */
            char s1[32], s2[32]; fmt_count(s1, sizeof s1, before->m); fmt_count(s2, sizeof s2, after->m);
            printf("%s%d of %d free variable%s eliminated by substitution (equality constraints %s -> %s)\n", npre ? cont : lab,
                   u->fe->ns, u->fe->np, u->fe->np > 1 ? "s" : "", s1, s2);
        }
        else printf("%s%s%s\n", npre ? cont : lab, what[c], d);
        npre++;
    }
    if (PP->ps && PP->ps->mom) { printf("%smoment-form conversion of the Schur complement\n", npre ? cont : lab); npre++; }
    if (PP->ps && PP->ps->chordal) {   /* (3 November 2026: the moment form counts its blocks in mom->nd; ps->nch only the overlap form's) */
        const int nb = PP->ps->mom ? PP->ps->mom->nd : PP->ps->nch;
        printf("%schordal decomposition: %d block%s split into cliques\n", npre ? cont : lab, nb, nb == 1 ? "" : "s"); npre++;
    }
    {   /* 1 November 2026 (the owner): low-rank data among the structure found, as the symmetries and the faces are */
        long ncl = 0, rtot = 0;
        if (lowrank_used(PP, &ncl, &rtot)) {
            char s1[32], s2[32]; fmt_count(s1, sizeof s1, ncl); fmt_count(s2, sizeof s2, rtot);
            printf("%slow-rank data: %s constraint%s used through factors of total rank %s\n", npre ? cont : lab, s1, ncl > 1 ? "s" : "", s2);
            npre++;
        }
    }
    if (u->dep_removed > 0) { printf("%s%d linearly dependent equality constraint%s removed (dual value%s 0)\n", npre ? cont : lab, u->dep_removed, u->dep_removed > 1 ? "s" : "", u->dep_removed > 1 ? "s" : ""); npre++; }
    if (u->fr_removed > u->fr_faces) { printf("%sfacial reduction: %d constraint%s removed\n", npre ? cont : lab, u->fr_removed - u->fr_faces, u->fr_removed - u->fr_faces > 1 ? "s" : ""); npre++; }
    if (u->fr_faces > 0) { printf("%sfacial reduction: %d face%s of the lifted relaxation, found from its linear constraints, removed\n", npre ? cont : lab, u->fr_faces, u->fr_faces > 1 ? "s" : ""); npre++; }
    if (!npre) printf("%snone applied\n", lab);
}
/* (3 November 2026, the owner on the bound's log: "very confusing") the certification's reason when it failed and its route
 * (0 the certificate's margin; 1-3 a-priori bounds from the certificate, the returned point, the iterate before the polishes),
 * for the Summary's Bound lines */
static char g_cert_why[2][320];
static int g_cert_prior[2];
/* (3 November 2026, the owner) a bound with 7 significant digits, rounded so that the printed number is still a bound: down
 * for a lower bound (dir -1), up for an upper bound (dir +1) */
void brisk_fmt_bound7(char *buf, size_t sz, double v, int dir) { brisk_fmt_bound(buf, sz, v, dir, 7); }
/* (3 November 2026, the owner: the bound with as many digits as the Objective, 11 significant) a bound with nd significant
 * digits (2-15), rounded outward: down for a lower bound (dir -1), up for an upper one (+1), so that the printed number is
 * still a bound. From v's decimal expansion (60 digits): truncated at nd digits, one unit of the last digit away from zero
 * when the dropped digits are not all zero and the direction asks for a larger magnitude */
void brisk_fmt_bound(char *buf, size_t sz, double v, int dir, int nd) {
    if (!isfinite(v)) { snprintf(buf, sz, "%g", v); return; }
    if (nd < 2) nd = 2; if (nd > 15) nd = 15;
    char w[96]; snprintf(w, sizeof w, "%.60e", fabs(v));
    const int neg = v < 0;
    long long m = w[0] - '0'; int k = 1;
    for (const char *c = w + 2; k < nd; c++, k++) m = m * 10 + (*c - '0');
    int rest = 0; for (const char *c = w + 1 + nd; *c && *c != 'e'; c++) if (*c != '0') { rest = 1; break; }
    int ex = atoi(strchr(w, 'e') + 1);
    const int away = (dir < 0) == neg;   /* a lower bound of a negative value, an upper bound of a positive one: larger magnitude */
    long long p10 = 1; for (int i = 1; i < nd; i++) p10 *= 10;
    if (rest && away) { m++; if (m >= 10 * p10) { m /= 10; ex++; } }
    snprintf(buf, sz, "%s%lld.%0*llde%+03d", neg ? "-" : "", m / p10, nd - 1, m % p10, ex);
}
static void summary_bound_lines(const Run *u, const Params *par) {
    for (int sd = 1; sd <= 2; sd++) {
        if (!(par->bound_side & sd)) continue;
        const BoundCert *bc = &u->cert[sd - 1];
        const char *ul = sd == 1 ? "upper" : "lower";
        /* (3 November 2026, the owner: one line, no wording about the certificate; the options' description explains it) */
        if (bc->have && bc->certified && isfinite(bc->rigorous)) {
            char v[48]; brisk_fmt_bound(v, sizeof v, bc->rigorous, sd == 1 ? +1 : -1, 11);
            printf("  Bound: guaranteed %s bound %s (see the option bound in OPTIONS.md)\n", ul, v);
        } else if (u->status == ST_PINFEAS || u->status == ST_DINFEAS) {
            printf("  Bound: none, the problem was found infeasible\n");
        } else {
            char t[480];
            snprintf(t, sizeof t, "no guaranteed %s bound: %s", ul, !bc->have ? "no admissible certificate was found" : g_cert_why[sd - 1][0] ? g_cert_why[sd - 1] : "the rigorous check failed");
            print_wrapped("  Bound: ", t, 100);
        }
    }
}
static void print_summary_lines(const Run *u, const Params *par, const PSOrig *O, int status, const double *err, int infeas,
                                const char *cause, double t_all, double tread, double tpre, int iters, const char *extra,
                                double pobj, double dobj, int near_miss);
/* ISSUES 725: the Summary is the user's (the default log prints it); a few of its lines, about the solver's internal
 * steps, print in the diagnosis (-v) only */
static void print_summary(const Run *u, const Params *par, const PSOrig *O, int status, const double *err, int infeas,
                          const char *cause, double t_all, double tread, double tpre, int iters, const char *extra,
                          double pobj, double dobj, int near_miss) {
    g_log_user++;
    print_summary_lines(u, par, O, status, err, infeas, cause, t_all, tread, tpre, iters, extra, pobj, dobj, near_miss);
    g_log_user--;
}
static void print_summary_lines(const Run *u, const Params *par, const PSOrig *O, int status, const double *err, int infeas,
                                const char *cause, double t_all, double tread, double tpre, int iters, const char *extra,
                                double pobj, double dobj, int near_miss) {
    char sz[200];
    printf("------------------------------------------------------------------------------------------\n");
    summary_size(sz, sizeof sz, O, u->fe ? u->fe->np : u->nfree_read);
    printf("Summary\n  Problem as read:   %s\n", sz);
    print_preprocessing(u, "  Preprocessing:     ", "                     ");
    /* the engine */
    const Result *R = &u->R;
    char eng[300];
    extern int g_fom_bound_used;
    if (par->fom > 0 && g_fom_bound_used) snprintf(eng, sizeof eng, "first-order method with bounds on the entries of X (ADMM; %d constraints taken as bounds), %d iterations", g_fom_bound_used, iters);
    else if (par->fom > 0) snprintf(eng, sizeof eng, "first-order method (ADMM, then semismooth Newton steps), %d iterations", iters);
    else if (par->lralm > 0) snprintf(eng, sizeof eng, "low-rank augmented Lagrangian method, %d iterations", iters);
    else if (par->mfipm == 2) snprintf(eng, sizeof eng, "matrix-free interior-point iterations, then the interior-point method from their iterate; %d iterations", iters);
    else if (par->mfipm > 0 && !g_mf_hand_kept) snprintf(eng, sizeof eng, "matrix-free interior-point method (conjugate gradients), %d iterations", iters);
    else if (R->methods == 4) snprintf(eng, sizeof eng, "the SOCP solver (each semidefinite block is equivalent to a second-order cone), %d iterations", iters);
    else snprintf(eng, sizeof eng, "interior-point method: %s, %s direction, %d iterations", R->methods == 3 ? "infeasible start, then self-dual embedding" : R->methods == 2 || R->retried ? "self-dual embedding" : "infeasible start",
                  R->direction == 1 ? "NT" : "HKM", iters);
    if (g_natt == 0) printf("  Solve:             %s\n", eng);
    else {
        /* 20 October 2026 (the owner): the solves in order, each with its time and where it ended */
        char first[300]; int kept_any = 0;
        for (int a = 0; a < g_natt; a++) kept_any |= g_att[a].kept;
        snprintf(first, sizeof first, "%s", g_first_name ? g_first_name : eng);
        if (!g_first_name) { char *it = strrchr(first, ','); if (it && strstr(it, "iterations")) *it = 0; }
        if (g_mf_hand_kept) { const size_t l = strlen(first); snprintf(first + l, sizeof first - l, ", started from the matrix-free solve's iterate"); }
        int last_kept = -1;                 /* (a re-solve kept and later replaced by another: only the last one's result is returned) */
        for (int a = 0; a < g_natt; a++) if (g_att[a].kept) last_kept = a;
        /* 23 October 2026: in the order they ran: the attempts made before the first solve (the matrix-free method and the
         * first-order method tried first) ahead of it; they had been listed after it (kmeans_n170_k10, SDP: "Solve 2:
         * matrix-free interior-point method" for the 44 s that came before the factored solve) */
        int no = 1;
        #define ATT_PRINT(a) do { char ended[40] = ""; \
            if (g_att[a].acc >= 0) snprintf(ended, sizeof ended, ", ended at %.1e", g_att[a].acc); \
            printf("  Solve %d:%*s%s (%.1f s%s)%s\n", no, no < 10 ? 11 : 10, "", g_att[a].name, g_att[a].t, ended, (a) == last_kept ? "; returned" : ""); \
            no++; } while (0)
        for (int a = 0; a < g_natt; a++) if (g_att[a].before) ATT_PRINT(a);
        printf("  Solve %d:%*s%s (%.1f s, ended at %.1e)%s\n", no, no < 10 ? 11 : 10, "", first, g_first_t, g_first_acc, kept_any ? "" : "; returned");
        no++;
        for (int a = 0; a < g_natt; a++) if (!g_att[a].before) ATT_PRINT(a);
        #undef ATT_PRINT
        printf("  Method:            %s\n", eng);
    }
    if (g_fr_dual_lost) printf("  Dual:              not recovered for the %d constraints removed by the facial reduction\n", g_fr_dual_lost);
    for (int i = 0; i < g_nnotes; i++) if (!(g_log_brief && g_note_diag[i])) printf("                     %s\n", g_notes[i]);
    if (extra && extra[0]) printf("                     %s\n", extra);
    if (status == ST_TIME) printf("                     stopped by the time limit%s\n", brisk_stop_flag ? " (interrupted)" : "");
    else if (status == ST_MAXIT) printf("                     stopped at the iteration limit\n");
    g_t_all_note = t_all;
    if (g_log_brief) printf("  Time:              %.2f s (presolve %.2f s, solve %.2f s)\n", t_all, tpre, fmax(0.0, t_all - tpre));
    else printf("  Time:              %.2f s (presolve %.2f, solve and re-solves %.2f; reading the file %.2f s more)\n", t_all, tpre, fmax(0.0, t_all - tpre), tread);
    /* 31 October 2026 (ISSUES 727): the line of the log's start, the same words and reason (the owner's wording) */
    { const int nt = brisk_threads_logged(); if (nt > 0) printf("  Number of threads: %d%s\n", nt, brisk_threads_label(nt)); }
    if (infeas) {
        printf("  Verdict:           %s\n", status == ST_PINFEAS ? "the problem is primal infeasible (no X >= 0 with A(X) = b)" : "the problem is dual infeasible (no y with C - A'y >= 0)");
        if (cause && cause[0]) { char t[600]; snprintf(t, sizeof t, "%s", cause); print_wrapped("                     ", t, 100); }
        printf("  Status:            %s\n", status == ST_PINFEAS ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE");
        return;
    }
    const double e = brisk_nmax(brisk_nmax(fabs(err[1]), fabs(err[2])), brisk_nmax(brisk_nmax(fabs(err[3]), fabs(err[4])), brisk_nmax(fabs(err[5]), fabs(err[6]))));
    /* ISSUES 725: the status and the objective values in the Summary (the default log has no other place for them) */
    if (status == ST_OPTIMAL) printf("  Status:            %s\n", status_str[status]);
    else printf("  Status:            %s (DIMACS error %.1e)\n", status_str[status], e);
    if (cause && cause[0] && strcmp(cause, "not identified")) { char t[600]; snprintf(t, sizeof t, "likely cause: %s", cause); print_wrapped("                     ", t, 100); }
    if (g_amp_warn[0]) { char t[600]; snprintf(t, sizeof t, "warning: %s", g_amp_warn); print_wrapped("                     ", t, 100); }
    printf("  Objective:         primal <C,X> = %.10e, dual b'y = %.10e\n", pobj, dobj);
    printf("  DIMACS errors:     pinf %.1e, X>=0 %.1e, dinf %.1e, Z>=0 %.1e, gap %.1e, compl %.1e\n", err[1], err[2], err[3], err[4], err[5], err[6]);
    summary_bound_lines(u, par);
    brisk_out_summary_line();
    printf("Solved to DIMACS error %.1e\n", e);
    resolve_hint();
    if (g_infeas_unconf == 2)   /* (audit A2: the verdict may rest on the facial reduction) */
        printf("The facial reduction removed constraints before the verdict. To solve without it%s:\n  %s   solves without the facial reduction\n", syn_head(), syn_kv("nofr", NULL, 0));
    blkscale_hint(status);
    freeelim_hint(status);
    decision_hint();
    if (g_fr_dual_lost) {   /* the fix for the dual first; more accuracy would not give it */
        /* (4 November 2026, the owner) a shallower reduction first: its duals are rebuilt where the deep chain's are not */
        extern int g_fr_depth_used;
        char kf[64], kd[64]; snprintf(kf, sizeof kf, "%s", syn_kv("frdual", "1", 0)); snprintf(kd, sizeof kd, "%s", syn_kv("frdepth", "1", 0));
        const char *k = syn_kv("nofr", NULL, 0);
        /* (4 November 2026) the second solve for the dual first: the dual of least violation with the optimal value held,
         * about one solve of the problem as read (a user's moment relaxations: every DIMACS error <= 3e-8) */
        size_t w = strlen(kf); if (strlen(k) > w) w = strlen(k); if (g_fr_depth_used >= 2 && strlen(kd) > w) w = strlen(kd);
        /* (5 November 2026, the owner: "Dual recovery after facial reduction failed: modeling problems? Try -frdual 1 or -nofr")
         * Both are offered: on the files where the dual was lost (bench/frdual_1104) the second solve gave a feasible dual on
         * every one, while the solve without the reduction was OPTIMAL and faster on most benchmark files (pop_wood,
         * exp_v4) but slow and less accurate on a user's Navier-Stokes relaxations; the share of constraints the reduction
         * removed does not tell them apart (exp_v4: 96 %, the faster without; Navier-Stokes: 72 %, the slower). */
        printf("Dual recovery after the facial reduction failed: the problem has no strictly feasible point,\n"
               "often a sign that the model can be written smaller (variables forced to zero by the constraints).\n"
               "To recover the dual%s:\n  %-*s   a second solve for the dual, the optimal value of this one held\n", syn_head(), (int)w, kf);
        printf("  %-*s   solves without the facial reduction (often faster, on some problems\n  %-*s   much slower)\n", (int)w, k, (int)w, "");
        if (g_fr_depth_used >= 2)
            printf("  %-*s   a facial reduction of depth 1 (this one: depth %d), whose duals are\n  %-*s   rebuilt more often\n", (int)w, kd, g_fr_depth_used, (int)w, "");
        bound_hint(0);
    } else if (par->mfipm == 1 && status != ST_OPTIMAL && status != ST_TIME) {
        /* (3 November 2026) the matrix-free method asked for missed the tolerance: the method that factors the Schur complement,
         * not more accuracy (theta6, SDP: "For higher accuracy: -acc high" after a stall at 7.4e-5) */
        printf("For the default interior-point method%s:\n  %s\n", syn_head(), syn_kv("mfipm", "0", 0));
        bound_hint(0);
    } else if (status == ST_MAXIT) {
        printf("For a result%s:\n  %s   a larger iteration limit\n", syn_head(), syn_kv("maxit", "<k>", 0));
        bound_hint(0);
    } else if (!g_infeas_unconf) accuracy_hint(0);   /* (none on accuracy after an unconfirmed verdict; 1 November 2026, the owner: after an optimal solve too, in the default log as well, so that the high-precision solver is known) */
}


BriskResult *g_brisk_result = NULL;   /* 4.30: set by brisk_run */
static void diagnose_cause(const Run *u, const Params *par, const PSOrig *O, double **Xo, const int *xc, const double *yo,
                           const PSResult *pr, double acc, char *out, size_t len);
static void amp_warning(const PSOrig *O, double **Xo, const int *xc, const double *yo, const PSResult *pr, double acc,
                        double tol, char *out, size_t len);
static int certify_given(const char *fname, const char *cx, const char *cy);
static int g_certify_prior = 1, g_certify_verbose = 1;     /* -boundprior and -v for -certify-x / -certify-y */

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
#ifndef BRISK_NO_MAIN      /* (22 October 2026: the test drivers link main.c without its main, tests/misocp_bb) */
int main(int argc, char **argv) {
    /* 27 October 2026 (a user's report, ISSUES 708): MKL in its reproducible mode, read at its first call; without it
     * MKL chooses kernels by the arrays' alignment and the results changed from run to run at more than one thread
     * (sparsechol.c, mkl_single). Nothing for other BLAS libraries; a user's own setting is kept. */
    if (!getenv("MKL_CBWR")) setenv("MKL_CBWR", "AUTO", 1);
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
extern int g_threads_forced;
/* the count of -threads k in the arguments (the last one), 0 when absent or 0 (automatic) */
/* 31 October 2026 (ISSUES 726, the owner): "BRISK <version>" as the log's first line, so that a log says which version
 * wrote it; once per run (the command line, the outermost library call), not in the quiet mode (-q, -silent) */
static int g_hp_inner;       /* (defined below) */
static int g_version_logged = 0;   /* (once per run: nested calls of the same run do not print it again; reset per library call) */
static void log_version(int argc, char **argv) {
    if (g_caller_depth > 1 || g_hp_inner || g_version_logged) return;
    g_version_logged = 1;
    for (int i = 1; i < argc; i++) if (argv[i] && (!strcmp(argv[i], "-q") || !strcmp(argv[i], "-silent"))) return;
    log_user("BRISK %s\n", BRISK_VERSION);
}
static int threads_arg(int argc, char **argv) {
    int k = 0;
    for (int i = 1; i + 1 < argc; i++) if (argv[i] && argv[i + 1] && !strcmp(argv[i], "-threads")) { const int t = atoi(argv[i + 1]); k = t > 0 ? t : 0; }
    return k;
}
/* 31 October 2026 (ISSUES 727, the owner's agreed behaviour): the thread count of a library call is the call's own. A
 * positive -threads k is used for this solve only; without it (or with 0) the solve starts from the automatic count of
 * the process, whatever earlier calls set; on every return (normal, error, interruption through brisk_exit) the count
 * the caller had is restored. Until 1.6.1 an explicit count was kept for later calls on purpose, so MATLAB's
 * installation test (test_brisk, threads = 1) left every later solve of the session on one thread. The automatic count
 * is the runtime's own (OMP_NUM_THREADS, else the cores; the performance cores on Apple silicon), read once, at the
 * first call of the process, before any call set it. Only the outermost call does this (nested calls run inside it). */
#ifdef BRISK_LIBRARY
static int g_nt_default = 0;
static int nt_scope_begin(int argc, char **argv) {
    if (g_caller_depth <= 1) g_version_logged = 0;        /* (a new library call: its log starts with the version) */
#ifdef _OPENMP
    if (g_caller_depth > 1) return -1;
    if (g_nt_default <= 0) g_nt_default = omp_get_max_threads();
    const int saved = omp_get_max_threads();
    const int k = threads_arg(argc, argv);
    g_threads_forced = k > 0;
    omp_set_num_threads(k > 0 ? k : g_nt_default);
    return saved;
#else
    (void)argc; (void)argv; return -1;
#endif
}
static void nt_scope_end(int saved) {
#ifdef _OPENMP
    if (saved > 0) omp_set_num_threads(saved);
#endif
    if (g_caller_depth <= 1) g_threads_forced = 0;
}
#endif
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
/* 23 October 2026: a silent double solve of a problem given in memory by the default pipeline, as the one that warms
 * the high-precision solver; for the ray problems of its certificate check (hpsolve.c, ISSUES 256). Returns the status */
int brisk_double_solve_data(const BriskData *d, BriskResult *res) {
    static char a0[] = "brisk", pr[] = "-prec", pd[] = "dd";
    char *av[5] = { a0, (char *)g_brisk_data_name, pr, pd, NULL };    /* (-prec with g_hp_inner: the inner double solve) */
    BriskResult *saved = g_brisk_result;
    const BriskData *sdata = g_brisk_data;
    const int sinner = g_hp_inner;
    memset(res, 0, sizeof(*res)); res->status = -1;
    g_brisk_result = res; g_brisk_data = d; g_hp_inner = 1;
    /* (26 October 2026: the caller reads X as matrices: no cone vectors from the inner solve, and the outer run's record
     * of them, g_xcone_res, left as it was; see the warm start below) */
    const int xacc = g_xcone_accept; int *xres = g_xcone_res; g_xcone_accept = 0; g_xcone_res = NULL;
    int (*hook)(const char *, int) = NULL;
    const int fd = hp_mute_begin(&hook);
    brisk_main(4, av);
    hp_mute_end(fd, hook);
    free(g_xcone_res); g_xcone_accept = xacc; g_xcone_res = xres;
    g_hp_inner = sinner; g_brisk_result = saved; g_brisk_data = sdata;
    return res->status;
}

/* ---- problems in SeDuMi format (sedumi.c) ------------------------------------------------ */
#include <pthread.h>
#include "sedumi.h"
extern _Thread_local volatile int *socp_stop_call;     /* (socp.h: the SOCP solver's second stop flag, the calling thread's) */
#include "lpio.h"
#include "lpsolve.h"
#include "simplex.h"
#include "simplex_pre.h"
#include "qp.h"
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
static int cone_lpm0 = 0;   /* (3 November 2026: -lpmethod 0 among the run's options: the SOCP solver directly; set by the callers) */
static int args_lpm0(int argc, char **argv) { for (int i = 0; i + 1 < argc; i++) if (argv[i] && !strcmp(argv[i], "-lpmethod") && atoi(argv[i + 1]) == 0) return 1; return 0; }
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
static void cone_preamble(const SedumiProb *P);
static int log_brief_wanted(int argc, char **argv);
extern int g_cone_brief;   /* (sedumi.c) */
static void cone_problem_line(const SedumiProb *P);
int brisk_run_sedumi(const SedumiProb *P, int argc, char **argv, SedumiRes *R) {
    char **av = caller_enter(&argc, argv, "c");
    if (av) argv = av;
    if (g_caller_depth == 1) notes_clear();
    const int nt_saved = nt_scope_begin(argc, argv);     /* (31 October 2026, ISSUES 727: as brisk_run) */
    log_version(argc, argv);
    const int brief0 = g_log_brief, user0 = g_log_user;   /* (restored at the end; the merge of 5 November 2026) */
    int quiet0 = 0; for (int i = 0; i < argc; i++) if (argv[i] && (!strcmp(argv[i], "-q") || !strcmp(argv[i], "-silent"))) quiet0 = 1;
    const int brief = !quiet0 && P->ns == 0 && g_caller_depth == 1 && log_brief_wanted(argc, argv);   /* (2 November 2026) */
    if (brief) { cone_lpm0 = args_lpm0(argc, argv); cone_preamble(P); }
    const int cb0 = g_cone_brief; g_cone_brief = brief;
    const int rc = sedumi_solve(P, argc, argv, R, sedumi_lib_sdp);
    g_cone_brief = cb0;
    /* 5.10: the SOCP solver's summary also for a problem given in memory (the command line printed
     * it for a file only), so that the interfaces' logs end like the semidefinite solver's */
    if (R->status >= 0 && R->cone && g_caller_depth == 1 && !getenv("BRISK_NOSUMMARY")) {
        int quiet = 0;
        for (int i = 0; i < argc; i++) if (argv[i] && (!strcmp(argv[i], "-q") || !strcmp(argv[i], "-silent"))) quiet = 1;
        if (!quiet) { advice_scan(argc, argv, 0); cone_summary(P, R); }
    }
    g_log_brief = brief0; g_log_user = user0;
    nt_scope_end(nt_saved);
    caller_leave(av);
    return rc;
}
#else
int brisk_run_sedumi(const SedumiProb *P, int argc, char **argv, SedumiRes *R) { return sedumi_solve(P, argc, argv, R, sedumi_inner_sdp); }
#endif
static void sedumi_write_vec(const char *fname, const double *v, int n) {
    FILE *f = brisk_out_open(fname);
    if (!f) return;
    for (int i = 0; i < n; i++) fprintf(f, "%.17g\n", v[i]);
    brisk_out_close(f, fname);
}
/* 5.8: the summary of a cone solve (SeDuMi form without PSD blocks: the SOCP solver) */
/* 17 October 2026: the summary's thread line of the SOCP solver: its large factor blocks take the threads
 * (schol_set_panel_threads in socp_solve), the rest of it is sequential */
static void cone_threads_line(void) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    { const char *e = getenv("BRISK_CONEPANEL"); if (e && atoi(e) == 0) nth = 1; }
    { extern int socp_threads_used; if (socp_threads_used > 0) nth = socp_threads_used; }   /* (31 October 2026: the count the solve used, as its log line) */
    printf("  Number of threads: %d%s\n", nth, brisk_threads_label(nth));
}
static double g_cone_tread = 0;     /* (23 October 2026) seconds of reading the file of the problem the SOCP solver solves (0: given in memory) */
/* 2 November 2026: the default log of the SOCP solver as the other solvers': the problem, the threads and the method before
 * the solve (cone_preamble), the status and the objective in the Summary; the solver's own lines are the diagnosis (-v) */
extern int g_cone_brief;   /* (sedumi.c) */
static char g_cone_objline[200] = "";
static int g_cone_fobj_set = 0, g_cone_fobj_max = 0; static double g_cone_fobj = 0, g_cone_fdual = 0;   /* (a CBF file's objective as written; the SOCP developer's, kept by the merge of 5 November 2026) */
/* (the merge of 5 November 2026) the SOCP developer's problem line of the default log: the problem line is cone_preamble's here */
static void cone_problem_line(const SedumiProb *P) { (void)P; }
/* the problem's line of the SOCP solver's log: the counts in words, the kinds that are absent left out */
static void cone_desc(char *buf, size_t sz, const SedumiProb *P) {
    char cm[32], cn[32]; fmt_count(cm, sizeof cm, P->m); fmt_count(cn, sizeof cn, P->n);
    int qmax = 0; for (int k = 0; k < P->nq; k++) if (P->q[k] > qmax) qmax = P->q[k]; for (int k = 0; k < P->nr; k++) if (P->r[k] > qmax) qmax = P->r[k];
    int o = snprintf(buf, sz, "%s row%s, %s variable%s:", cm, P->m == 1 ? "" : "s", cn, P->n == 1 ? "" : "s");
    const char *sep = " ";
    if (P->nf && o < (int)sz) { o += snprintf(buf + o, sz - o, "%s%d free", sep, P->nf); sep = ", "; }
    if (P->nl && o < (int)sz) { o += snprintf(buf + o, sz - o, "%s%d nonnegative", sep, P->nl); sep = ", "; }
    if (P->nq && o < (int)sz) { o += snprintf(buf + o, sz - o, "%s%d second-order cone%s", sep, P->nq, P->nq == 1 ? "" : "s"); sep = ", "; }
    if (P->nr && o < (int)sz) { o += snprintf(buf + o, sz - o, "%s%d rotated cone%s", sep, P->nr, P->nr == 1 ? "" : "s"); sep = ", "; }
    if ((P->nq || P->nr) && o < (int)sz) snprintf(buf + o, sz - o, " (largest of order %d)", qmax);
}
static void cone_preamble(const SedumiProb *P) {
    char d[256]; cone_desc(d, sizeof d, P);
    printf("Problem:  %s\n", d);
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    { const char *e = getenv("BRISK_CONEPANEL"); if (e && atoi(e) == 0) nth = 1; }
    printf("Number of threads: %d%s\n", nth, brisk_threads_label(nth));
    /* (3 November 2026) a problem without cones: the LP interior-point method (sedumi.c), the SOCP solver if it does not end
     * optimal; the Summary says which gave the result */
    if (P->nq == 0 && P->nr == 0 && P->m > 0 && P->n > 0 && !g_sedumi_nolpm && !cone_lpm0) printf("Method:   interior-point method for linear programs\n");
    else printf("Method:   SOCP solver (homogeneous self-dual model, Nesterov-Todd scaling)\n");
}
static void cone_summary(const SedumiProb *P, const SedumiRes *R) {
    char d[256]; cone_desc(d, sizeof d, P);
    printf("------------------------------------------------------------------------------------------\n");
    printf("Summary\n  Problem as read:   %s\n", d);
    int np = 0; for (int i = 0; i < g_nnotes; i++) if (!strncmp(g_notes[i], "presolve: ", 10)) { printf("  %s%s\n", np ? "                   " : "Preprocessing:     ", g_notes[i] + 10); np++; }
    if (!np) printf("  Preprocessing:     none applied\n");
    if (R->lpm == 1) printf("  Solve:             LP interior-point method (normal equations, Mehrotra predictor-corrector), %d iterations\n", R->iters);
    else if (R->lpm == 2) {
        printf("  Solve 1:           LP interior-point method, stopped without an optimal point (%.2f s)\n", R->lpm_time);
        printf("  Solve 2:           SOCP solver (homogeneous self-dual model), %d iterations (%.2f s)\n", R->iters, R->time - R->lpm_time);
    } else printf("  Solve:             SOCP solver (homogeneous self-dual model), %d iterations\n", R->iters);
    for (int i = 0; i < g_nnotes; i++) if (strncmp(g_notes[i], "presolve: ", 10)) printf("                     %s\n", g_notes[i]);
    if (R->status == 6) printf("                     stopped by the time limit\n"); else if (R->status == 3) printf("                     stopped at the iteration limit\n");
    /* (23 October 2026) the total time of the run, reading included (2013_firL1, SOCP: 11.7 s of reading were not counted) */
    if (g_cone_tread > 0) printf("  Time:              %.2f s (reading the file %.2f s, the solve %.2f s)\n", g_cone_tread + R->time, g_cone_tread, R->time);
    else printf("  Time:              %.2f s\n", R->time);
    cone_threads_line();
    if (R->status == 1 || R->status == 2) { printf("  Verdict:           the problem is %s infeasible (the returned %s is a certificate)\n  Status:            %s\n", R->status == 1 ? "primal" : "dual", R->status == 1 ? "y" : "x", R->status == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE"); return; }
    printf("  Status:            %s\n", R->status_str);
    if (g_cone_objline[0]) printf("  Objective:         %s\n", g_cone_objline);
    else printf("  Objective:         primal c'x = %.10e, dual b'y = %.10e\n", R->pobj, R->dobj);
    printf("  DIMACS errors:     pinf %.1e, x in K %.1e, dinf %.1e, z in K* %.1e, gap %.1e, compl %.1e\n", R->err[0], R->err[1], R->err[2], R->err[3], R->err[4], R->err[5]);
    double e = 0; for (int i = 0; i < 6; i++) e = fmax(e, fabs(R->err[i]));
    brisk_out_summary_line();
    printf("Solved to DIMACS error %.1e\n", e);
    /* (3 November 2026) after the LP method's attempt, how to go to the SOCP solver at once (the owner's rule: the advice
     * says how to avoid a re-solve) */
    if (R->lpm == 2) printf("To solve it by the SOCP solver at once%s:\n  %s   the SOCP solver directly\n", syn_head(), syn_kv("lpmethod", "0", 0));
    else accuracy_hint(0);
}
/* the command line on a MAT-file with A (or At), b, c, K: argv[ifile] is the file */
static int log_brief_wanted(int argc, char **argv);
static int brisk_main_sedumi(int argc, char **argv, int ifile) {
    SedumiProb P; char msg[400];
    { const double t0r = wtime(); if (sedumi_read_mat(argv[ifile], &P, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; } g_cone_tread = wtime() - t0r; }
    if (P.ns > 0) g_log_brief = log_brief_wanted(argc, argv);   /* (ISSUES 725: semidefinite blocks: the semidefinite solver's default log; the SOCP route's default log is cone_preamble's, g_cone_brief: the merge of 5 November 2026) */
    { extern void socp_log_reset(void); socp_log_reset(); }
    { extern void (*socp_note_diag)(const char *, ...); socp_note_diag = brisk_note_diag; }
    char **av = malloc(sizeof(char *) * ((size_t)argc + 1));
    int ac = 0, quiet = 0;
    const char *xf = NULL, *yf = NULL, *zf = NULL, *bxf = NULL, *byf = NULL;
    for (int i = 1; i < argc; i++) {
        if (i == ifile) continue;
        if (i + 1 < argc && (!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z"))) {
            if (argv[i][1] == 'x') xf = argv[i + 1]; else if (argv[i][1] == 'y') yf = argv[i + 1]; else zf = argv[i + 1];
            i++; continue;
        }
        if (i + 1 < argc && (!strcmp(argv[i], "-boundx") || !strcmp(argv[i], "-boundy"))) {    /* in SeDuMi's layout, below */
            if (argv[i][6] == 'x') bxf = argv[i + 1]; else byf = argv[i + 1];
            i++; continue;
        }
        if (!strcmp(argv[i], "-q")) quiet = 1;
        av[ac++] = argv[i];
    }
    const int brief = !quiet && P.ns == 0 && log_brief_wanted(argc, argv);   /* (2 November 2026: the SOCP solver's default log) */
    if (brief) { cone_lpm0 = args_lpm0(argc, argv); cone_preamble(&P); }
    else if (!quiet) {
        long sd = 0; int smax = 0, qmax = 0;
        for (int k = 0; k < P.ns; k++) { sd += (long)P.s[k] * P.s[k]; if (P.s[k] > smax) smax = P.s[k]; }
        for (int k = 0; k < P.nq; k++) if (P.q[k] > qmax) qmax = P.q[k];
        for (int k = 0; k < P.nr; k++) if (P.r[k] > qmax) qmax = P.r[k];
        printf("SeDuMi problem %s: m = %d, n = %d; K.f = %d, K.l = %d, K.q: %d, K.r: %d (largest %d), K.s: %d (largest %d)\n",
               argv[ifile], P.m, P.n, P.nf, P.nl, P.nq, P.nr, qmax, P.ns, smax);
        cone_problem_line(&P);
    }
    SedumiRes R;
    const int cb0 = g_cone_brief; g_cone_brief = brief;
    const int rc = sedumi_solve(&P, ac, av, &R, sedumi_inner_sdp);
    g_cone_brief = cb0;
    if (R.status >= 0) {
        if (!quiet && P.ns > 0 && !g_log_brief) printf("SeDuMi form: c'x = %+.12e   b'y = %+.12e\n", R.pobj, R.dobj);
        if (!quiet && R.cone && !quiet_summary) cone_summary(&P, &R);
        if (xf && R.x && (P.ns == 0 || R.have_x)) sedumi_write_vec(xf, R.x, P.n);
        if (yf && R.y) sedumi_write_vec(yf, R.y, P.m);
        if (zf && R.z) sedumi_write_vec(zf, R.z, P.n);
        if (bxf) { if (R.bnd[0].x) sedumi_write_vec(bxf, R.bnd[0].x, P.n); else fprintf(stderr, "brisk: -boundx: no certificate x to write (it needs -bound p or pd and an admissible candidate)\n"); }
        if (byf) { if (R.bnd[1].y) sedumi_write_vec(byf, R.bnd[1].y, P.m); else fprintf(stderr, "brisk: -boundy: no certificate y to write (it needs -bound d or pd and an admissible candidate)\n"); }
        if (g_brisk_result) {       /* a library call on the file: status, values and y */
            BriskResult *res = g_brisk_result;
            res->status = R.status; snprintf(res->status_str, sizeof res->status_str, "%s", R.status_str);
            res->exit_code = rc; res->iters = R.iters; res->m = P.m; res->pobj = R.pobj; res->dobj = R.dobj; res->time = R.time;
            for (int i = 0; i < 6; i++) res->err[i + 1] = R.err[i];
            res->y = R.y; R.y = NULL;
            res->sn = P.n; res->have_x = P.ns == 0 || R.have_x; res->sx = R.x; R.x = NULL; res->sz = R.z; R.z = NULL;
            for (int s = 0; s < 2; s++) {        /* the bounds: the certificate x (SeDuMi's layout) and y */
                res->bnd[s].asked = R.bnd[s].asked; res->bnd[s].have = R.bnd[s].have; res->bnd[s].valid = R.bnd[s].valid; res->bnd[s].certified = R.bnd[s].certified;
                res->bnd[s].value = R.bnd[s].value; res->bnd[s].resid = R.bnd[s].resid; res->bnd[s].lammin = R.bnd[s].lammin; res->bnd[s].rigorous = R.bnd[s].rigorous;
                memcpy(res->bnd[s].src, R.bnd[s].src, sizeof(res->bnd[s].src));
                res->bnd[s].sx = R.bnd[s].x; R.bnd[s].x = NULL; res->bnd[s].y = R.bnd[s].y; R.bnd[s].y = NULL;
            }
            {
                const int s0 = R.bnd[0].asked ? 0 : 1;
                res->bound_side = R.bnd[s0].asked && R.bnd[s0].have ? s0 + 1 : 0;
                res->bound_value = R.bnd[s0].value; res->bound_resid = R.bnd[s0].resid; res->bound_lammin = R.bnd[s0].lammin;
                res->bound_valid = R.bnd[s0].valid; res->bound_certified = R.bnd[s0].certified; res->bound_rigorous = R.bnd[s0].rigorous;
            }
        }
    }
    sedumi_result_free(&R); sedumi_free(&P); free(av);
    return rc;
}
/* (22 October 2026) a solve of the LP interior-point method that gave up before the solve whose result is returned:
 * the Summary lists them as Solve 1, Solve 2 (the owner's rule of 20 October 2026: the re-solves below the line) */
typedef struct { int sxpre, iters; double t; char why[192]; int verdict; } LpAttempt;   /* (verdict: 1 or 2, a presolve found the problem infeasible or unbounded) */
/* 5.8: the summary of a linear program (MPS): the presolve, the LP interior-point method or the SOCP solver */
/* (31 October 2026, ISSUES 637) the interior-point method's three errors, or "not measured" when it stopped before its first
 * iterate (they are then infinite: lpipm_core) */
static const char *lp_errtxt(char *b, size_t nb, double p, double d, double g) {
    if (isinf(p) && isinf(d) && isinf(g)) snprintf(b, nb, "not measured (no iterate)");
    else snprintf(b, nb, "pinf %.1e  dinf %.1e  gap %.1e", p, d, g);
    return b;
}
/* (31 October 2026, the owner's rule of ISSUES 725 for every solver's log, applied to the LP method: ISSUES 639) in the default
 * log (g_log_brief) the Summary is the user's: the methods named in words, the status, the objective and the errors on the
 * problem as read; the solve's internal errors and the method's details stay for -v */
static int g_lp_nt_total = 0;     /* (the threads of the LP route, the interior-point method's and the simplex method's beside it) */
static void lp_summary(const LpProb *L, const SedumiProb *P, const SedumiRes *R, int lpipm, double eb, double er, double tpre, double ttot,
                       int natt, const LpAttempt *att, int sxpre, double tlast, double obj) {
    const int brief = g_log_brief; if (brief) g_log_user++;
    char a[32], b[32], c[32], d[32]; fmt_count(a, sizeof a, L->m); fmt_count(b, sizeof b, L->n); fmt_count(c, sizeof c, P->m); fmt_count(d, sizeof d, P->n);
    printf("------------------------------------------------------------------------------------------\n");
    printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a, b, L->maxim ? "maximized" : "minimized");
    printf("  Preprocessing:     presolve and standard form: %s rows, %s columns (%.2f s)\n", c, d, tpre);
    int np = 0; for (int i = 0; i < g_nnotes; i++) if (!strncmp(g_notes[i], "presolve: ", 10)) { printf("                     %s\n", g_notes[i] + 10); np++; }
    static const char *const pre_x = " after the presolve of parallel rows and columns, doubleton equations and slack columns", *pre_s = " after the standard presolve", *pre_r = " on the problem as read";
    if (natt == 0 && brief)
        printf("  Solve:             %s, %d iterations\n", lpipm ? "interior-point method" : "SOCP solver (homogeneous self-dual model)", R->iters);
    else if (natt == 0)
        printf("  Solve:             %s%s, %d iterations\n", lpipm ? "LP interior-point method (normal equations, Mehrotra predictor-corrector)" : "SOCP solver (homogeneous self-dual model)",
               sxpre == 1 ? pre_x : sxpre == 2 ? pre_r : "", R->iters);
    else {
        for (int a = 0; a < natt; a++)
            if (att[a].verdict)
                printf("  Solve %d:           %s (%.2f s): found the problem %s%s%s%s; the problem as read is solved %s\n", a + 1,
                       att[a].sxpre == 1 ? "the presolve of parallel rows and columns, doubleton equations and slack columns" : "the standard presolve", att[a].t,
                       att[a].verdict == 1 ? "infeasible" : "unbounded", att[a].why[0] ? " (" : "", att[a].why, att[a].why[0] ? ")" : "",
                       att[a].sxpre == 1 ? "after the standard presolve" : "without the presolve");
            else
            printf("  Solve %d:           LP interior-point method%s, %d iterations (%.2f s), stopped: %s\n", a + 1, att[a].sxpre == 1 ? pre_x : att[a].sxpre == 2 ? pre_r : pre_s, att[a].iters, att[a].t, att[a].why);
        printf("  Solve %d:           %s%s, %d iterations (%.2f s); its result is returned\n", natt + 1, lpipm ? "LP interior-point method" : "SOCP solver (homogeneous self-dual model)",
               lpipm ? (sxpre == 1 ? pre_x : sxpre == 2 ? pre_r : pre_s) : "", R->iters, tlast);
    }
    for (int i = 0; i < g_nnotes; i++) if (strncmp(g_notes[i], "presolve: ", 10)) printf("                     %s\n", g_notes[i]);
    if (R->status == 6) printf("                     stopped by the time limit\n"); else if (R->status == 3) printf("                     stopped at the iteration limit\n");
    {   int nth = 1;
#ifdef _OPENMP
        nth = omp_get_max_threads();
#endif
        if (brief && g_lp_nt_total > 0) nth = g_lp_nt_total;    /* (the default log: all the threads of the route, as at its start) */
        if (lpipm) printf("  Time:              %.2f s\n  Number of threads: %d%s\n", ttot, nth, brisk_threads_label(nth));
        else { printf("  Time:              %.2f s\n", ttot); cone_threads_line(); }
    }
    if (R->status == 1 || R->status == 2) { printf("  Verdict:           the problem is %s\n  Status:            %s\n", R->status == 1 ? "primal infeasible" : "dual infeasible (unbounded)", R->status == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE"); if (brief) g_log_user--; return; }
    if (brief) {
        const char *ss = R->status_str[0] ? R->status_str : "STOPPED"; const char *pp = strstr(ss, " (");
        printf("  Status:            %.*s\n", pp ? (int)(pp - ss) : (int)strlen(ss), ss);
        printf("  Objective:         %.10e\n", obj);
        /* (3 November 2026) the dual error and the gap too, so that the last line's error is the largest number shown here
         * (afiro, LP: row violation 1.8e-14 here, "Solved to relative error 5.2e-13" below, the dual error unshown) */
        if (isinf(R->err[0])) printf("  Errors:            variable bounds violated by %.1e, constraints by %.1e (relative)\n", eb, er);
        else printf("  Errors:            variable bounds violated by %.1e, constraints by %.1e, dual constraints by %.1e; gap %.1e (relative)\n",
                    eb, er, fabs(R->err[2]), fabs(R->err[4]));
    } else { char eb3[96]; printf("  Errors:            variable bounds violated by %.1e, constraints by %.1e (relative); of the solve: %s\n", eb, er, lp_errtxt(eb3, sizeof eb3, R->err[0], R->err[2], R->err[4])); }
    /* (31 October 2026, ISSUES 637) "solved" only for an optimal or reduced-accuracy result; a point returned at a limit or
     * after a failure is said to be one, with its error on the problem as read (Q003_lattice_equalities, LP: "Solved to
     * relative error 6.9e-01" after the time limit) */
    if (R->status == 0 || R->status == 5)
        printf("Solved to relative error %.1e\n", fmax(fmax(eb, er), isinf(R->err[0]) ? 0.0 : fmax(fmax(fabs(R->err[0]), fabs(R->err[2])), fabs(R->err[4]))));
    else
        printf("Not solved (%s): the point returned has a relative error of %.1e on the problem as read\n", R->status_str[0] ? R->status_str : "stopped", fmax(eb, er));
    if (natt > 0 && att[0].verdict)
        printf("To start %s%s:\n  %s   %s\n", att[0].sxpre == 1 ? "with the standard presolve" : "without the presolve", syn_head(), syn_kv("lppresolve", att[0].sxpre == 1 ? "2" : "0", 0),
               att[0].sxpre == 1 ? "without the presolve of parallel rows and columns, doubleton equations and slack columns" : "no presolve");
    else if (natt > 0 && !lpipm) printf("To start with the SOCP solver%s:\n  %s   the SOCP solver directly\n", syn_head(), syn_kv("lpmethod", "0", 0));
    else if (natt > 0) printf("To start with the standard presolve%s:\n  %s   without the presolve of parallel rows and columns, doubleton equations and slack columns\n", syn_head(), syn_kv("lppresolve", "2", 0));
    if (R->status == 6 && !strcmp(R->status_str, "TIME LIMIT")) printf("For a result%s:\n  %s   a longer time limit (seconds)\n", syn_head(), syn_kv("timelimit", "<s>", 0));
    else if (R->status == 3) printf("For a result%s:\n  %s   a larger iteration limit\n", syn_head(), syn_kv("maxit", "<k>", 0));
    else if (R->status == 5 || (R->status == 0 && !brief)) {
        /* (31 October 2026, ISSUES 635) a result at reduced accuracy: the method that gives an optimal basic solution first
         * (fir, LP: the tolerance does not help a method that stalled) */
        if (R->status == 5) printf("For an optimal basic solution%s:\n  %s   the simplex method, or the interior-point method followed by a crossover\n", syn_head(), syn_kv("lpmethod", "5", 0));
        accuracy_hint(1);    /* (no advice after an optimal result in the default log) */
    }
    if (brief) g_log_user--;
}
/* the command line on an MPS file with -lpmethod 2 to 5: the simplex method (simplex.c) on the problem as read */
static int brisk_main_lp_simplex(LpProb *L, int ac, char **av, const char *fname, int method, int quiet, const char *xf, double t0, int lpcorr) {
    SimplexOpts o; memset(&o, 0, sizeof o);
    o.method = method; o.verbose = quiet ? 0 : 1;
    /* (25 October 2026) -lpcorr for the interior-point method of -lpmethod 4 and 5 too: it was read only by -lpmethod 1 */
    if (lpcorr >= 0) o.ipm_kcorr = lpcorr + 1;
    for (int i = 0; i < ac; i++) {
        if (i + 1 < ac && !strcmp(av[i], "-tol")) o.tol_primal = o.tol_dual = atof(av[i + 1]);
        if (i + 1 < ac && !strcmp(av[i], "-maxit")) o.maxit = atol(av[i + 1]);
        if (i + 1 < ac && !strcmp(av[i], "-timelimit")) o.timelimit = atof(av[i + 1]);
        if (!strcmp(av[i], "-v") || !strcmp(av[i], "-vv")) o.verbose = 2;
    }
    if (o.verbose >= 2) printf("Number of threads: %d%s\n", omp_get_max_threads(), brisk_threads_label(omp_get_max_threads()));     /* (RELEASE_CHECKS R10) */
    if (g_log_brief) {      /* (31 October 2026, ISSUES 639: the default log's lines before the solve) */
        log_user("Number of threads: %d%s\n", omp_get_max_threads(), brisk_threads_label(omp_get_max_threads()));
        log_user("Method:   %s\n", method == 1 ? "dual simplex method" : method == 2 ? "primal simplex method" : method == 3 ? "interior-point method, then a crossover to an optimal basis"
                                    : "chosen automatically between the dual simplex method and the interior-point method with a crossover");
    }
    if (getenv("BRISK_SXNOSCALE")) o.noscale = 1;
    if (getenv("BRISK_SXNOPERTURB")) o.noperturb = 1;
    double *x = calloc((size_t)L->n + 1, sizeof(double)), *y = calloc((size_t)L->m + 1, sizeof(double)), *z = calloc((size_t)L->n + 1, sizeof(double));
    SimplexInfo inf;
    /* (3 November 2026) the default log's tables: the simplex method's when it runs alone (methods 1, 2), the interior-point
     * method's before its crossover (3); none for the automatic choice, whose two methods may run at the same time */
    extern int g_stab;
    const int st0 = g_stab, it0 = g_itab; g_stab = g_log_brief && (method == 1 || method == 2); g_itab = g_log_brief && method == 3;
    const int st = simplex_solve(L, &o, x, NULL, y, z, NULL, NULL, &inf);
    g_stab = st0; g_itab = it0;
    static const char *sname[] = {"OPTIMAL", "PRIMAL INFEASIBLE", "DUAL INFEASIBLE", "ITERATION LIMIT", "NUMERICAL DIFFICULTIES", "", "TIME LIMIT"};
    const char *ss = st >= 0 && st <= 6 ? sname[st] : "UNKNOWN";
    const int rc = st == 0 ? 0 : st == 1 ? 11 : st == 2 ? 12 : st == 3 ? 3 : st == 6 ? 6 : 4;
    if (!quiet) {
        printf("status: %s   iterations: %ld   time: %.3fs\n", ss, inf.iters, inf.time);
        if (st == 0 || st == 3 || st == 6) printf("objective value (as in the file, %s): %.12e\n", L->maxim ? "maximized" : "minimized", inf.obj);
        printf("  violations on the problem as read: primal %.1e, dual %.1e (absolute)\n", inf.pinf, inf.dinf);
        if (!quiet_summary) {
            char a[32], b[32]; fmt_count(a, sizeof a, L->m); fmt_count(b, sizeof b, L->n);
            const int brief = g_log_brief; if (brief) g_log_user++;
            printf("------------------------------------------------------------------------------------------\n");
            printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a, b, L->maxim ? "maximized" : "minimized");
            /* (27 October 2026, the night) the presolved problem infeasible or unbounded: the problem as read solved again */
            if (inf.asread_check)
                printf("  Solve 1:           on the presolved problem (%.2f s): %s; the problem as read is solved without the presolve\n",
                       inf.t_presolved, inf.st_presolved == 1 ? "found infeasible" : "found unbounded");
            if (inf.asread_check)
                printf("  Solve 2:           %s simplex method on the problem as read, %ld iterations (%ld dual, %ld primal; %ld in phase 1), %d factorizations (%.2f s); its result is returned\n",
                       method == 2 ? "primal" : "dual", inf.iters, inf.iters_dual, inf.iters_primal, inf.iters_phase1, inf.nfactor, inf.time - inf.t_presolved);
            else if (inf.auto_ipm && inf.raced)
                printf("  Solve 1:           dual simplex method, %ld iterations (%.2f s), stopped: the interior-point method beside it finished first\n",
                       inf.iters_first, inf.t_first);
            else if (inf.auto_ipm)
                printf("  Solve 1:           dual simplex method, %ld iterations (%.2f s), stopped at the predicted work of the interior-point method\n",
                       inf.iters_first, inf.t_first);
            if (inf.asread_check) ;
            else if (inf.ipm_iters > 0)
                printf("  Solve %d:           interior-point method, %d iterations (%.2f s), then crossover to a basis: %ld simplex iterations\n",
                       inf.auto_ipm ? 2 : 1, inf.ipm_iters, inf.t_ipm, inf.iters - inf.iters_first);
            else
                printf("  Solve 1:           %s simplex method, %ld iterations (%ld dual, %ld primal; %ld in phase 1), %d factorizations (%.2f s)\n",
                       method == 2 ? "primal" : "dual", inf.iters, inf.iters_dual, inf.iters_primal, inf.iters_phase1, inf.nfactor, inf.time);
            if (brief) {
                printf("  Time:              %.2f s\n  Number of threads: %d\n  Status:            %s\n", wtime() - t0, omp_get_max_threads(), ss);
                if (st == 0 || st == 3 || st == 6) printf("  Objective:         %.10e\n", inf.obj);
                if (st == 0 || st == 3 || st == 6) printf("  Errors:            on the problem as read: primal %.1e, dual %.1e (absolute)\n", inf.pinf, inf.dinf);
                if (st == 0) printf("Solved to an optimal basic solution\n");
                else if (st == 1 || st == 2) printf("  Status:            %s\n", st == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE");
                else printf("Not solved (%s)\n", ss);
                if (st == 6) printf("For a result%s:\n  %s   a longer time limit (seconds)\n", syn_head(), syn_kv("timelimit", "<s>", 0));
                else if (st == 3) printf("For a result%s:\n  %s   a larger iteration limit\n", syn_head(), syn_kv("maxit", "<k>", 0));
            } else {
                printf("  Time:              %.2f s (factorizations %.2f, solves %.2f, pricing %.2f)\n", wtime() - t0, inf.t_factor, inf.t_ftran + inf.t_btran, inf.t_price);
                printf("%s\n", ss);
            }
            if (inf.auto_ipm && st == 0)
                printf("To start with the interior-point method%s:\n  %s   the interior-point method, then a crossover\n", syn_head(), syn_kv("lpmethod", "4", 0));
            if (brief) g_log_user--;
        }
    }
    if (xf) sedumi_write_vec(xf, x, L->n);
    if (g_brisk_result) {
        BriskResult *res = g_brisk_result;
        res->status = st; snprintf(res->status_str, sizeof res->status_str, "%s", ss);
        res->exit_code = rc; res->iters = (int)inf.iters; res->m = L->m; res->pobj = inf.obj; res->dobj = inf.obj; res->time = wtime() - t0;
        res->sn = L->n; res->have_x = 1; res->sx = x; x = NULL;
    }
    (void)fname;
    free(x); free(y); free(z);
    return rc;
}
/* the command line on an MPS file (a linear program): presolve, standard form, the SOCP solver */
/* (22 October 2026, the night) the constant that the standard form of L dropped, in the minimizing orientation: the
 * objective as given at the standard form's origin (lp_recover of zeros, after lp_to_std of L); the LP method measures its
 * gap against |c'x + this| */
/* (22 October 2026) the key of a pair of bounds (a row's sides, a column's bounds) for the symmetry check of the problem as
 * read: both values' bits mixed into a finite double. The former key lo + 3.1 hi, the infinite side replaced by 1e20,
 * lost the finite side of every row or column with one infinite side (1 + 3.1e20 = 2 + 3.1e20): lp_cheb_10000x300's
 * 20,000 rows started in one class instead of 20,000, and rows with different sides could be called equivalent */
static double lp_bounds_key(double lo, double hi) {
    double l = lo > -1e30 ? lo : -INFINITY, h = hi < 1e30 ? hi : INFINITY;
    if (l == 0) l = 0;
    if (h == 0) h = 0;
    uint64_t a, b; memcpy(&a, &l, 8); memcpy(&b, &h, 8);
    uint64_t z = (a * 0x9e3779b97f4a7c15ull) ^ ((b + 0x632be59bd9b4e019ull) * 0xbf58476d1ce4e5b9ull);
    z ^= z >> 29; z &= 0x3fffffffffffffffull;
    double v; memcpy(&v, &z, 8);
    return v;
}
static double lp_std_objoff(LpProb *L, const LpMap *M, int nstd) {
    double *z = calloc((size_t)nstd + 1, sizeof(double)), *x = calloc((size_t)L->n + 1, sizeof(double));
    const double v = lp_recover(L, M, z, x);
    free(z); free(x);
    return L->maxim ? -v : v;
}
/* (27 October 2026, the night) the default LP route with two threads or more: the dual simplex method (simplex_solve,
 * method 1, with its own presolve) on one thread beside the interior-point method on the others; the first to end with a
 * result (optimal, infeasible, unbounded) gives it, and the other is stopped. An independent test of 1.6 (320 generated
 * LPs against Gurobi) measured the interior-point method alone as the default: 7.5 times Gurobi's time on LPs with dense
 * columns, where BRISK's dual simplex method took 0.34 of it; the faster of the two per LP, 0.64 of Gurobi's time in the
 * geometric mean against 0.85 for the default. Gurobi's and CPLEX's defaults for an LP run the methods at once in the same
 * way. The result depends on which ends first (the race by the clock; the owner, 27 October 2026: the rule of the same
 * bits on every run does not apply to it); at one thread, or with -lpmethod 1, the interior-point method alone.
 * BRISK_LPNOCONC=1: off */
typedef struct {
    pthread_t th; int started;
    LpProb L; SimplexOpts o; SimplexInfo inf; double *x;
    volatile int stop;          /* (set by the caller: the interior-point method's result is taken) */
    volatile int done, won;     /* (set by the thread: ended; ended with a result to take) */
    volatile int nthr;          /* (the threads it may use: 1, all of them once the interior-point method has given up) */
    volatile int *anywon;       /* (28 October 2026: set too when it wins: the interior-point method's stop flag) */
    void *peer;                 /* (the other simplex job, stopped when this one wins: the bounded substitution's) */
    int st; double t0, t;
    SimplexProgress prog;       /* (4 November 2026: its progress, for the race's log) */
} LpConc;
static void *lp_conc_run(void *p) {
    LpConc *c = p;
    omp_set_num_threads(1);
    c->st = simplex_solve(&c->L, &c->o, c->x, NULL, NULL, NULL, NULL, NULL, &c->inf);
    c->t = wtime() - c->t0;
    if (c->st >= 0 && c->st <= 2 && !c->stop) {
        c->won = 1;
        if (c->anywon) *c->anywon = 1;
        if (c->peer) ((LpConc *)c->peer)->stop = 1;
    }
    __sync_synchronize(); c->done = 1;
    return NULL;
}
static int lp_conc_start(LpConc *c, const LpProb *L, double tlim, int aggb, volatile int *anywon) {   /* (aggb: 1 the bounded substitution's job, 2 the direct job, no route through the dual problem) */
    LpProb *D = &c->L; *D = *L;
    const int n = L->n, m = L->m, nz = L->Ap[n];
#define LPC_CP(f, cnt, T) do { D->f = malloc(sizeof(T) * ((size_t)(cnt) + 1)); memcpy(D->f, L->f, sizeof(T) * (size_t)(cnt)); } while (0)
    LPC_CP(Ap, n + 1, int); LPC_CP(Ai, nz, int); LPC_CP(Ax, nz, double); LPC_CP(c, n, double); LPC_CP(l, n, double); LPC_CP(u, n, double);
    LPC_CP(rl, m, double); LPC_CP(ru, m, double);
#undef LPC_CP
    D->rname = NULL; D->cname = NULL;
    memset(&c->o, 0, sizeof c->o);
    c->o.method = 1; c->o.timelimit = tlim; c->o.stop = &c->stop;
    c->nthr = 1; c->o.threads = &c->nthr;
    c->o.aggb_only = aggb == 1; c->o.no_dual_route = aggb == 2; c->anywon = anywon;
    memset(&c->prog, 0, sizeof c->prog); c->o.prog = &c->prog;
    c->x = calloc((size_t)n + 1, sizeof(double));
    c->t0 = wtime();
    if (pthread_create(&c->th, NULL, lp_conc_run, c)) { free(c->x); c->x = NULL; lp_free(D); return 0; }
    c->started = 1;
    return 1;
}
/* (4 November 2026, the owner: "No iterates displayed. Try to display the race side by side with only partial info (less
 * columns) per method"; and "I want the logs to stay essentially cost-free for each method") the race's log in the default
 * log: the interior-point method's iterations (its number, primal objective and largest relative error, values it has)
 * beside the dual simplex method's iteration count (one plain store per iteration: SimplexProgress) and, when it ends
 * optimal, its objective and largest violation on the problem as read. Only the interior-point method's thread prints: a
 * row at each of its first 10 iterations, every 5th up to 50, every 10th after, and the last row at the join; no thread or
 * lock exists for the log, and the simplex thread only stores. -v: the detailed lines instead, as before */
extern int (*brisk_print_hook)(const char *s, int is_err);
typedef struct {
    int on, maxim; double t0;
    int ipm_it; double ipm_obj, ipm_err;
    LpConc *cc;
} RaceLog;
static _Thread_local RaceLog *g_racelog = NULL;
static void race_out(const char *s) {
#ifdef BRISK_LIBRARY
    if (brisk_print_hook) { brisk_print_hook(s, 0); return; }
#endif
    fputs(s, stdout); fflush(stdout);
}
static void race_row(RaceLog *R, int last) {
    if (!R->on) return;
    char a[64], b[64], line[200];
    const int it = R->ipm_it;
    if (it >= 0) snprintf(a, sizeof a, "%5d  %+.8e   %.1e", it, R->ipm_obj, R->ipm_err); else snprintf(a, sizeof a, "%5s  %15s   %7s", "-", "", "");
    const SimplexProgress *P = &R->cc->prog;
    const long sit = P->it;
    if (last && R->cc->won) snprintf(b, sizeof b, "%8ld  %+.8e   %.1e", sit, R->cc->inf.obj, R->cc->inf.pinf);
    else if (P->started && sit > 0) snprintf(b, sizeof b, "%8ld", sit);
    else snprintf(b, sizeof b, "%8s", "-");
    snprintf(line, sizeof line, "%6.2f %s  | %s\n", wtime() - R->t0, a, b);
    race_out(line);
    if (last) R->on = 0;
}
static void race_ipm_cb(void *ctx, int it, double pobj, double err) {
    RaceLog *R = ctx;
    R->ipm_it = it; R->ipm_obj = R->maxim ? -pobj : pobj; R->ipm_err = err;
    if (it < 10 || (it < 50 && it % 5 == 0) || it % 10 == 0) race_row(R, 0);
}
static void race_begin(RaceLog *R, LpConc *cc, double t0, int maxim) {
    memset(R, 0, sizeof *R);
    R->on = 1; R->cc = cc; R->t0 = t0; R->maxim = maxim; R->ipm_it = -1;
    char h[256];
    snprintf(h, sizeof h, "%-41s| %s\n%6s %5s  %-15s   %-7s  | %8s  %-15s   %s\n", "              interior-point method", "dual simplex method",
             "TIME", "ITE", "PRIMAL OBJ", "ERROR", "ITER", "OBJECTIVE", "VIOLATION");
    race_out(h);
    g_racelog = R;
}
static void race_end(RaceLog *R) {
    if (!R || !R->cc) return;
    if (R->on) race_row(R, 1);
    if (g_racelog == R) g_racelog = NULL;
}
/* stops the thread unless take: waits for it either way */
static void lp_conc_join(LpConc *c, int stop) {
    if (c->started != 1) return;      /* (not started, or joined already) */
    if (stop) c->stop = 1;
    const double tj = wtime();
    pthread_join(c->th, NULL);
    if (g_racelog && g_racelog->cc == c) race_end(g_racelog);   /* (the race's log: its last row) */
    if (getenv("BRISK_LPDBG")) printf("   [the dual simplex method beside it: joined after %.4f s (stop %d), its status %d after %.4f s, ended at %.3f of the clock]\n", wtime() - tj, stop, c->st, c->t, c->t0 + c->t);
    c->started = 2;
}
static void lp_conc_free(LpConc *c) {
    if (c->started == 1) lp_conc_join(c, 1);
    if (c->started) { lp_free(&c->L); free(c->x); c->x = NULL; c->started = 0; }
}
/* (28 October 2026) the two simplex jobs (the problem, and the bounded substitution's when started, b): stop both and
 * wait (stop), or wait for the first to end with a result, or for both to end, then stop the other. Afterwards a holds the
 * job whose result is the earlier one (the structs swapped when it is b's) */
static void lp_conc_join2(LpConc *a, LpConc *b, int stop) {
    if (b->started != 1) { lp_conc_join(a, stop); if (b->started == 2 && b->won && (!a->won || b->t < a->t)) { LpConc t = *a; *a = *b; *b = t; } return; }
    if (stop) { a->stop = 1; b->stop = 1; }
    else {
        const struct timespec ms = { 0, 1000000 };
        while (!(a->won || b->won) && !(a->done && b->done)) nanosleep(&ms, NULL);
        if (!a->won) a->stop = 1;
        if (!b->won) b->stop = 1;
    }
    lp_conc_join(a, a->stop); lp_conc_join(b, b->stop);
    if (b->won && (!a->won || b->t < a->t)) { LpConc t = *a; *a = *b; *b = t; }
}
/* the point of the problem as read from the method's (through the standard form's map, and the simplex method's
 * postsolve when its presolve ran: every reduced status basic); returns the objective as in the file */
static double lp_point_as_read(const LpProb *L, const LpProb *Lw, const LpMap *M, const SxPre *SP, const double *Rx, double *x) {
    if (!SP) return lp_recover((LpProb *)L, (LpMap *)M, Rx, x);
    const int mr = Lw->m, nr = Lw->n;
    double *xr = calloc((size_t)nr + 1, sizeof(double)), *yo = calloc((size_t)L->m + 1, sizeof(double));
    int *csr = calloc((size_t)nr + 1, sizeof(int)), *rsr = calloc((size_t)mr + 1, sizeof(int)), *cso = malloc(sizeof(int) * ((size_t)L->n + 1)), *rso = malloc(sizeof(int) * ((size_t)L->m + 1));
    lp_recover((LpProb *)Lw, (LpMap *)M, Rx, xr);                     /* (statuses SX_BASIC = 0 from calloc) */
    if (getenv("BRISK_LPDBG")) { double eb_, er_; lp_errors((LpProb *)Lw, xr, &eb_, &er_); printf("   [the reduced problem: bound violation %.1e, row violation %.1e]\n", eb_, er_); }
    sxpre_postsolve(SP, xr, NULL, csr, rsr, x, yo, cso, rso);
    double obj = L->c0; for (int j = 0; j < L->n; j++) obj += L->c[j] * x[j];
    free(xr); free(yo); free(csr); free(rsr); free(cso); free(rso);
    return obj;
}
/* (1 November 2026) the point and the row duals of the problem as read from the method's (the crossover's start: x and y as
 * the simplex method returns them; through the simplex method's postsolve when its presolve ran) */
static void lp_point_y_as_read(const LpProb *L, const LpProb *Lw, const LpMap *M, const SxPre *SP, const double *Rx, const double *Ry, double *x, double *y) {
    if (!SP) {
        lp_recover((LpProb *)L, (LpMap *)M, Rx, x);
        for (int i = 0; i < L->m; i++) y[i] = M->rpos[i] >= 0 ? M->rsgn[i] * Ry[M->rpos[i]] : 0.0;
        return;
    }
    const int mr = Lw->m, nr = Lw->n;
    double *xr = calloc((size_t)nr + 1, sizeof(double)), *yr = calloc((size_t)mr + 1, sizeof(double));
    int *csr = calloc((size_t)nr + 1, sizeof(int)), *rsr = calloc((size_t)mr + 1, sizeof(int)), *cso = malloc(sizeof(int) * ((size_t)L->n + 1)), *rso = malloc(sizeof(int) * ((size_t)L->m + 1));
    lp_recover((LpProb *)Lw, (LpMap *)M, Rx, xr);
    for (int i = 0; i < mr; i++) yr[i] = M->rpos[i] >= 0 ? M->rsgn[i] * Ry[M->rpos[i]] : 0.0;
    sxpre_postsolve(SP, xr, yr, csr, rsr, x, y, cso, rso);
    free(xr); free(yr); free(csr); free(rsr); free(cso); free(rso);
}
static int brisk_main_lp(int argc, char **argv, int ifile) {
    LpProb L; char msg[400];
    const double t0 = wtime();
    if (lp_read_mps(argv[ifile], &L, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; }
    char **av = malloc(sizeof(char *) * ((size_t)argc + 1));
    int ac = 0, quiet = 0, pre = 3, lpm = 1, lpcorr = -1, lpm_given = 0;
    double tlim_lp = 0;
    const char *xf = NULL;
    for (int i = 1; i < argc; i++) {
        if (i == ifile) continue;
        if (i + 1 < argc && !strcmp(argv[i], "-x")) { xf = argv[i + 1]; i++; continue; }
        if (i + 1 < argc && !strcmp(argv[i], "-lppresolve")) { pre = atoi(argv[i + 1]); i++; continue; }
        if (i + 1 < argc && !strcmp(argv[i], "-lpmethod")) { lpm = atoi(argv[i + 1]); lpm_given = 1; i++; continue; }
        if (i + 1 < argc && !strcmp(argv[i], "-timelimit")) tlim_lp = atof(argv[i + 1]);
        if (i + 1 < argc && !strcmp(argv[i], "-lpcorr")) { lpcorr = atoi(argv[i + 1]); i++; continue; }
        if (!strcmp(argv[i], "-q")) quiet = 1;
        /* (23 October 2026, the night) -threads for every LP method: only the interior-point method's branch read it, so
         * -lpmethod 2 to 5 ran on all cores whatever was asked */
        if (i + 1 < argc && !strcmp(argv[i], "-threads") && atoi(argv[i + 1]) >= 1) { omp_set_num_threads(atoi(argv[i + 1])); g_threads_user = 1; }
        av[ac++] = argv[i];
    }
    /* (31 October 2026, ISSUES 639) the default log is the user's (the owner's rule of ISSUES 725): the problem, the threads,
     * the method, the Summary; every other line of the default level moves to -v */
    if (!g_hp_inner) g_log_brief = !quiet && log_brief_wanted(argc, argv);
    g_lp_nt_total = 0;
    if (!quiet) printf("LP %s (%s): %d rows, %d columns, %d nonzeros, read %.2fs\n", argv[ifile], L.name[0] ? L.name : "no name", L.m, L.n, L.Ap[L.n], wtime() - t0);
    if (g_log_brief) {
        char a_[32], b_[32], c_[32]; fmt_count(a_, sizeof a_, L.m); fmt_count(b_, sizeof b_, L.n); fmt_count(c_, sizeof c_, L.Ap[L.n]);
        log_user("Problem:  %s rows, %s columns, %s nonzeros (linear program, %s)\n", a_, b_, c_, L.maxim ? "maximized" : "minimized");
    }
    /* (3 November 2026) dual infeasibility decided before any method runs, by the bounds on the duals that the rows' sides
     * and the columns alone in their rows imply (lp_ray_singletons, lpio.c), with its primal ray checked on the data and
     * returned as the certificate: neos-3283608-agout and neos-3230376-yser (LPs) went through the presolve, the folding and
     * the interior-point method (0.7 and 4 s; Gurobi's presolve decides them in 0.02 and 0.2 s). BRISK_LPDUALBOX=0: off */
    if (L.m > 0 && L.n > 0) {
        double *dr = malloc(sizeof(double) * ((size_t)L.n + 1));
        const int kr = dr ? lp_ray_singletons(&L, dr) : -1;
        if (kr >= 0) {
            const char *cn = L.cname && L.cname[kr] ? L.cname[kr] : "(unnamed)";
            if (!quiet) {
                printf("status: DUAL INFEASIBLE (presolve: column %s improves the objective along a ray of the problem)   iterations: 0   time: %.3fs\n", cn, wtime() - t0);
                if (!quiet_summary) {
                    char a_[32], b_[32]; fmt_count(a_, sizeof a_, L.m); fmt_count(b_, sizeof b_, L.n);
                    if (g_log_brief) g_log_user++;
                    printf("------------------------------------------------------------------------------\n");
                    printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a_, b_, L.maxim ? "maximized" : "minimized");
                    printf("  Presolve:          column %s can improve the objective without limit: no dual point gives its reduced cost\n"
                           "                     the sign it needs, by the bounds on the duals that the rows' sides and the columns\n"
                           "                     alone in their rows imply; the ray (checked on the data) is returned as the certificate\n", cn);
                    printf("  Time:              %.2f s\n", wtime() - t0);
                    /* (the ray proves that no dual point exists; unbounded only if the problem also has a feasible point,
                     * which is not checked here: netlib's cplex1 has a ray and no feasible point) */
                    printf("  Verdict:           the problem is dual infeasible: it has no optimum (unbounded if it has a feasible point)\nDual infeasible\n");
                    if (g_log_brief) g_log_user--;
                }
            }
            if (xf) sedumi_write_vec(xf, dr, L.n);
            if (g_brisk_result) {
                BriskResult *res = g_brisk_result;
                res->status = 2; snprintf(res->status_str, sizeof res->status_str, "DUAL INFEASIBLE");
                res->exit_code = 12; res->iters = 0; res->m = L.m; res->pobj = res->dobj = 0; res->time = wtime() - t0;
                res->sn = L.n; res->have_x = 1; res->sx = dr; dr = NULL;
            }
            free(dr); lp_free(&L); free(av);
            return 12;
        }
        free(dr);
    }
    if (lpm >= 2 && lpm <= 5) {
        const int rcs = brisk_main_lp_simplex(&L, ac, av, argv[ifile], lpm == 5 ? 0 : lpm == 4 ? 3 : lpm == 3 ? 2 : 1, quiet, xf, t0, lpcorr);
        lp_free(&L); free(av);
        return rcs;
    }
    /* (27 October 2026, the night) the dual simplex method beside the interior-point method (lp_conc_run) */
    LpConc cc, cc2; memset(&cc, 0, sizeof cc); memset(&cc2, 0, sizeof cc2);
    volatile int conc_any = 0;      /* (a simplex job ended with a result: the interior-point method's stop flag; the jobs are joined before this function returns) */
    char ipm_line[1024] = "";     /* (the method's status lines: printed once its result is the one taken) */
    const int nt_all = omp_get_max_threads();
    if (getenv("BRISK_LPDBG")) printf("   [t %.3f s: the race starts; t0 %.3f of the clock]\n", wtime() - t0, t0);
    if (lpm == 1 && !lpm_given && nt_all >= 2 && L.m > 0 && L.n > 0 && !getenv("BRISK_LPNOCONC") && lp_conc_start(&cc, &L, tlim_lp, 0, &conc_any)) {
        /* with four threads or more, when the simplex job would go through the dual problem of a tall problem, a second
         * simplex job solving the problem itself, the interior-point method on one thread fewer (the race pairs them
         * too: on 12 held-out LPs the route through the dual problem was taken on 5, three to four times slower on
         * s1234 and 1.7 on stockholm, 4.5 times faster on momentum2). BRISK_LPCONC2=1; not by default (2 November 2026):
         * on the 55 development LPs where it starts, total time 379 -> 279 s, the geometric mean at the noise of the
         * untouched LPs (neos-2075418-temuka 86 -> 23 s, rmine13 86 -> 26 s, LPs), but on a fresh held-out set (the
         * solve call on the problem in memory) 4 % slower in geometric mean (H_svm_2 6.0 -> 9.6 s: the interior-point
         * method on one thread fewer where it wins); its wins come from the first job's choice of the route through the
         * dual problem, which is where they are to be fixed (ISSUES 612). BRISK_LPCONC2=2: when the bounded substitution can apply, the second job solves the problem with it
         * instead (satellites2-25, LP: 9.2 -> 5.4 s), not by default: on the 97 LPs where it starts, geometric mean 1.041
         * against the job off (the interior-point method on one thread fewer: rail02, LP, 42.9 -> 61.3 s); run by the
         * first simplex job itself instead, 1.013 (ofi, LP, 30 s -> the time limit of 120 s; tried and dropped) */
        if (nt_all >= 4 && getenv("BRISK_LPCONC2") && atoi(getenv("BRISK_LPCONC2")) > 0) {
            const char *e2 = getenv("BRISK_LPCONC2");
            const int kind = atoi(e2) == 2 && simplex_aggb_possible(&L) ? 1 : simplex_dual_route_possible(&L) ? 2 : 0;
            if (kind && lp_conc_start(&cc2, &L, tlim_lp, kind, &conc_any)) { cc.peer = &cc2; cc2.peer = &cc; }
            if (getenv("BRISK_LPDBG")) fprintf(stderr, "   [the second simplex job: kind %d (1 the bounded substitution, 2 the direct solve), %s]\n", kind, cc2.started ? "started" : "not started");
        }
        omp_set_num_threads(nt_all - 1 - (cc2.started ? 1 : 0));
    }
    const int nt_ipm = nt_all - (cc.started ? 1 : 0) - (cc2.started ? 1 : 0);     /* (the interior-point method's threads, for the Summary) */
    g_lp_nt_total = nt_all;
    if (g_log_brief) {
        log_user("Number of threads: %d\n", nt_all);
        if (lpm) log_user("Method:   interior-point method%s\n", cc.started ? " and dual simplex method in parallel (the first to finish gives the result)" : "");
        else log_user("Method:   SOCP solver (homogeneous self-dual model)\n");
    }
    /* (4 November 2026) the race's log (race_row above): the default log only, not with -v or -q */
    RaceLog rlog; memset(&rlog, 0, sizeof rlog);
    {   int vflag = 0; for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "-vv")) vflag = 1;
        static int rlo = -1; if (rlo < 0) { const char *e_ = getenv("BRISK_LPRACELOG"); rlo = !(e_ && atoi(e_) == 0); }
        if (rlo && g_log_brief && !quiet && !vflag && lpm && cc.started) race_begin(&rlog, &cc, t0, L.maxim); }
    SedumiProb P; LpMap M;
    const double tp0 = wtime();
    double *ub = NULL;
    /* 21 October 2026 (the night): the interior-point method on the problem the simplex method's presolve leaves
     * (simplex_pre.c: parallel rows and columns, doubleton equations, slack columns, forcing and redundant rows), then
     * lpio.c's own reductions on it. The postsolve carries the primal point back with every reduced status basic: each
     * removed column then takes a value that keeps its row and its bounds (only x is returned on this route).
     * BRISK_LPSXPRE=0 or -lppresolve 2 (22 October 2026: level 3 is the default and has it): lpio.c's presolve alone, as before. */
    LpProb Lr; SxPre *SP = NULL; LpProb *Lw = &L;
    /* (the same night) not when the problem as read folds to at most half its rows and columns: the folding (lpsolve.c,
     * the coarsest equitable partition) is then the larger reduction, and the simplex method's presolve, which merges
     * parallel columns in part of an orbit only, destroys it (ex9: 33,846 rows folded to 8,415, 0.8 s; after that
     * presolve no partition, 15 s). A problem that folds less keeps the presolve (irish-electricity folds by a fifth and
     * needs its reductions to converge). The partition of the problem as read: rows by their two sides, columns by cost
     * and bounds. (22 October 2026: only when that presolve would run; at -lppresolve 2 it cost lp_cheb_10000x300, 6.04
     * million nonzeros, 0.35 s for an answer not used, ISSUES 421) */
    int lpsym = 0, lpsym_keepdep = 0;
    /* (4 November 2026) the folding below stops when the dual simplex method beside it has its result (its stop flag) */
    const volatile int *lpsf0 = lp_get_stop(); if (cc.started) lp_set_stop(&conc_any);
    if (lpm == 1 && pre >= 3 && !getenv("BRISK_LPNOFOLD") && !(getenv("BRISK_LPSXPRE") && atoi(getenv("BRISK_LPSXPRE")) == 0) && L.m > 0 && L.n > 0) {
        double *hb = malloc(sizeof(double) * ((size_t)L.m + 1)), *hc = malloc(sizeof(double) * ((size_t)L.n + 1)), *hu = malloc(sizeof(double) * ((size_t)L.n + 1));
        int *rcl = malloc(sizeof(int) * ((size_t)L.m + 1)), *ccl = malloc(sizeof(int) * ((size_t)L.n + 1)), nrc = 0, ncc = 0;
        for (int i = 0; i < L.m; i++) hb[i] = lp_bounds_key(L.rl[i], L.ru[i]);
        for (int j = 0; j < L.n; j++) { hc[j] = L.c[j]; hu[j] = lp_bounds_key(L.l[j], L.u[j]); }
        lpsym = (getenv("BRISK_LPFOLDOLD") ? lp_fold_partition(L.m, L.n, 0, L.Ap, L.Ai, L.Ax, hb, hc, hu, rcl, ccl, &nrc, &ncc)
                 : lp_fold_partition_stop(L.m, L.n, 0, L.Ap, L.Ai, L.Ax, hb, hc, hu, rcl, ccl, &nrc, &ncc, 0.5)) && (double)nrc + ncc <= 0.5 * ((double)L.m + L.n);
        if (getenv("BRISK_LPDBG")) printf("   [the problem as read: %d row and %d column classes of %d and %d%s]\n", nrc, ncc, L.m, L.n, lpsym ? ": folded first, the simplex method's presolve skipped" : "");
        /* (28 October 2026, ISSUES 626) the dependent equality rows when the problem folds: dropping them may break the partition
         * (qap15, LP: one dependent row of a class of equivalent rows dropped, the fold 6,330 -> 2,121 rows lost, 1.3 -> 33 s) or
         * restore it (copies of quantile regressions with redundant equations, LP: 1,341 row classes with them, 442 without, the
         * original's). The partition without them decides: the standard presolve drops them unless it has more classes */
        if (lpsym && !(getenv("BRISK_LPDEPROWS") && atoi(getenv("BRISK_LPDEPROWS")) == 0)) {
            int me = 0; for (int i = 0; i < L.m; i++) if (L.rl[i] == L.ru[i] && L.rl[i] > -LP_INF) me++;
            lpsym_keepdep = 1;
            if (me >= 2) {
                int *ei = malloc(sizeof(int) * ((size_t)L.m + 1)), *erow = malloc(sizeof(int) * (size_t)me); long *ep = calloc((size_t)me + 2, sizeof(long));
                for (int i = 0, t = 0; i < L.m; i++) { ei[i] = -1; if (L.rl[i] == L.ru[i] && L.rl[i] > -LP_INF) { ei[i] = t; erow[t++] = i; } }
                for (int p = 0; p < L.Ap[L.n]; p++) if (ei[L.Ai[p]] >= 0 && L.Ax[p] != 0) ep[ei[L.Ai[p]] + 2]++;
                for (int t = 0; t < me; t++) ep[t + 2] += ep[t + 1];
                int *ej = malloc(sizeof(int) * ((size_t)ep[me + 1] + 1)); double *ev = malloc(sizeof(double) * ((size_t)ep[me + 1] + 1)), *eb = malloc(sizeof(double) * (size_t)me);
                for (int j = 0; j < L.n; j++) for (int p = L.Ap[j]; p < L.Ap[j + 1]; p++) if (ei[L.Ai[p]] >= 0 && L.Ax[p] != 0) { const long q = ep[ei[L.Ai[p]] + 1]++; ej[q] = j; ev[q] = L.Ax[p]; }
                for (int t = 0; t < me; t++) eb[t] = L.rl[erow[t]];
                char *dep = malloc((size_t)me);
                const int nd = lp_dependent_rows(me, L.n, ep, ej, ev, eb, dep, NULL);
                if (nd == 0) lpsym_keepdep = 0;
                else {
                    /* the partition of the problem without them (rows renumbered, their entries left out) */
                    int *rn = malloc(sizeof(int) * ((size_t)L.m + 1)), m2 = 0;
                    for (int i = 0; i < L.m; i++) rn[i] = (ei[i] >= 0 && dep[ei[i]]) ? -1 : m2++;
                    int *Ap2 = malloc(sizeof(int) * ((size_t)L.n + 1)), *Ai2 = malloc(sizeof(int) * ((size_t)L.Ap[L.n] + 1)); double *Ax2 = malloc(sizeof(double) * ((size_t)L.Ap[L.n] + 1));
                    double *hb2 = malloc(sizeof(double) * ((size_t)m2 + 1)), *hc2 = malloc(sizeof(double) * ((size_t)L.n + 1)), *hu2 = malloc(sizeof(double) * ((size_t)L.n + 1));
                    int *rcl2 = malloc(sizeof(int) * ((size_t)m2 + 1)), *ccl2 = malloc(sizeof(int) * ((size_t)L.n + 1)), nrc2 = 0, ncc2 = 0, w2 = 0;
                    for (int j = 0; j < L.n; j++) { Ap2[j] = w2; for (int p = L.Ap[j]; p < L.Ap[j + 1]; p++) if (rn[L.Ai[p]] >= 0) { Ai2[w2] = rn[L.Ai[p]]; Ax2[w2++] = L.Ax[p]; } hc2[j] = L.c[j]; hu2[j] = lp_bounds_key(L.l[j], L.u[j]); }
                    Ap2[L.n] = w2;
                    for (int i = 0; i < L.m; i++) if (rn[i] >= 0) hb2[rn[i]] = lp_bounds_key(L.rl[i], L.ru[i]);
                    const int ok2 = lp_fold_partition_stop(m2, L.n, 0, Ap2, Ai2, Ax2, hb2, hc2, hu2, rcl2, ccl2, &nrc2, &ncc2, 0.5);
                    if (ok2 && (double)nrc2 + ncc2 <= (double)nrc + ncc) lpsym_keepdep = 0;
                    if (getenv("BRISK_LPDBG")) printf("   [the problem as read without its %d dependent equations: %d row and %d column classes: they are %s]\n", nd, ok2 ? nrc2 : m2, ok2 ? ncc2 : L.n, lpsym_keepdep ? "kept (the fold)" : "dropped");
                    free(rn); free(Ap2); free(Ai2); free(Ax2); free(hb2); free(hc2); free(hu2); free(rcl2); free(ccl2);
                }
                free(ei); free(erow); free(ep); free(ej); free(ev); free(eb); free(dep);
            }
        }
        free(hb); free(hc); free(hu); free(rcl); free(ccl);
    }
    lp_set_stop(lpsf0);
    if (getenv("BRISK_LPDBG")) printf("   [t %.3f s: the folding check of the problem as read done]\n", wtime() - t0);
    int pv_st = 0, pv_sx = 0; double pv_t = 0; char pv_why[96] = "";   /* (a presolve's verdict, checked on the problem as read) */
    if (lpm == 1 && pre >= 3 && !lpsym && !(getenv("BRISK_LPSXPRE") && atoi(getenv("BRISK_LPSXPRE")) == 0)) {
        char why[160] = "";
        /* (28 October 2026, the night) the presolves on this thread stop when the dual simplex method beside it has its
         * result (their flags per thread, restored after each call: neos-4984115-canche, LP, the standard presolve 45 s
         * after the simplex method's result at 14 s) */
        const volatile int *sxs0 = sxpre_get_stop(); if (cc.started) sxpre_set_stop(&conc_any);
        const int ps = sxpre_presolve(&L, &Lr, &SP, quiet ? 0 : 1, why, sizeof why);
        sxpre_set_stop(sxs0);
        /* (27 October 2026, the night) its verdict of infeasible or unbounded is checked on the problem as read with the
         * standard presolve (Solve 1 and 2 in the Summary): the presolve decides in exact arithmetic on data rounded to the
         * file's digits and gives no certificate (dell, LP: found infeasible, optimal within 4e-9 for Gurobi, HiGHS and
         * the simplex method on the problem as read). BRISK_LPNOASREADCHECK=1: the verdict returned, as before */
        /* (28 October 2026, the night) an unbounded verdict stands without that solve when the problem as read has a column
         * that is a ray by the signs of its data alone (lp_ray_column: exact; the solve to confirm took 10-64 s on unbounded
         * held-out LPs) */
        int rayj = -1;
        if (ps == 2 && (rayj = lp_ray_column(&L)) >= 0)
            snprintf(why, sizeof why, "column %s of the problem as read improves the objective without limit, and no row or bound stops it",
                     L.cname && L.cname[rayj] ? L.cname[rayj] : "(unnamed)");
        if ((ps == 1 || ps == 2) && rayj < 0 && !getenv("BRISK_LPNOASREADCHECK")) {
            pv_st = ps; pv_sx = 1; pv_t = wtime() - tp0; snprintf(pv_why, sizeof pv_why, "%s", why);
            if (!quiet) printf("  the presolve of parallel rows and columns, doubleton equations and slack columns found the problem %s (%s): solving the problem as read with the standard presolve, to confirm\n",
                               ps == 1 ? "infeasible" : "unbounded", why);
            sxpre_free(SP); SP = NULL;
        } else if (ps == 1 || ps == 2) {
            if (!quiet) printf("status: %s (presolve: %s)   iterations: 0   time: %.3fs\n", ps == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE", why, wtime() - t0);
            const int rcp = ps == 1 ? 11 : 12;
            if (g_brisk_result) { g_brisk_result->status = ps; snprintf(g_brisk_result->status_str, sizeof g_brisk_result->status_str, "%s", ps == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE"); g_brisk_result->exit_code = rcp; }
            sxpre_free(SP); lp_conc_free(&cc); lp_conc_free(&cc2); lp_free(&L); free(av);
            return rcp;
        }
        if (ps == 0) Lw = &Lr; else { sxpre_free(SP); SP = NULL; }
        /* (31 October 2026, ISSUES 638) the Summary says when equations were combined: the problem is the same, its form not */
        { extern int sxpre_nsparsify(const SxPre *P);
          if (SP && sxpre_nsparsify(SP) > 0 && !quiet) brisk_note("the presolve replaced %d equations by their difference with another (entries that cancel exactly)", sxpre_nsparsify(SP)); }
        if (ps == 0 && getenv("BRISK_LPDBG")) {
            double bm = 0, rm = 0; for (int j = 0; j < Lr.n; j++) { if (Lr.l[j] > -1e30) bm = fmax(bm, fabs(Lr.l[j])); if (Lr.u[j] < 1e30) bm = fmax(bm, fabs(Lr.u[j])); }
            for (int i = 0; i < Lr.m; i++) { if (Lr.rl[i] > -1e30) rm = fmax(rm, fabs(Lr.rl[i])); if (Lr.ru[i] < 1e30) rm = fmax(rm, fabs(Lr.ru[i])); }
            double am = 0; long nb = 0; for (int p = 0; p < Lr.Ap[Lr.n]; p++) { am = fmax(am, fabs(Lr.Ax[p])); nb += fabs(Lr.Ax[p]) > 1e12; }
            printf("   [the reduced problem: largest finite bound %.2e, largest finite side %.2e, largest entry %.2e (%ld above 1e12)]\n", bm, rm, am, nb);
        }
    }
    const volatile int *lps0 = lp_get_stop(); if (cc.started) lp_set_stop(&conc_any);
    /* (28 October 2026, ISSUES 626) no search for dependent rows when the simplex method's presolve removed them, nor when the
     * problem as read folds and dropping them would break the partition (lpsym_keepdep, above) */
    lp_set_nodeprows(SP != NULL || lpsym_keepdep);
    if (lpm) lp_to_std(Lw, pre, quiet ? 0 : 1, &P, &M, &ub); else lp_to_sedumi(Lw, pre, quiet ? 0 : 1, &P, &M);
    lp_set_nodeprows(0);
    lp_set_stop(lps0);
    if (getenv("BRISK_LPDBG")) printf("   [t %.3f s: the standard presolve done]\n", wtime() - t0);
    int asread_raw = 0;   /* (the problem as read without presolve: after a decline, or a presolve's verdict) */
    /* (27 October 2026, the night) the standard presolve's verdict of infeasible or unbounded: the problem as read is solved
     * without it, as the simplex method's presolve's above */
    if (lpm && pre > 0 && (M.status == 1 || M.status == 2) && !pv_st && !getenv("BRISK_LPNOASREADCHECK") && !(M.status == 2 && lp_ray_column(&L) >= 0)) {
        pv_st = M.status; pv_sx = 0; pv_t = wtime() - tp0; pv_why[0] = 0;
        if (!quiet) printf("  the standard presolve found the problem %s: solving the problem as read without the presolve, to confirm\n", M.status == 1 ? "infeasible" : "unbounded");
        sedumi_free(&P); lp_map_free(&M); free(ub); ub = NULL;
        if (SP) { lp_free(&Lr); sxpre_free(SP); SP = NULL; }
        Lw = &L; asread_raw = 1;
        lp_to_std(&L, 0, quiet ? 0 : 1, &P, &M, &ub);
    }
    const double tpre = wtime() - tp0;
    int rc = 0, lp_done = 0, natt = 0, sxpre_last = SP != NULL;
    LpAttempt att[3]; double tlast = 0;
    memset(att, 0, sizeof att);
    if (pv_st) { att[0].sxpre = pv_sx; att[0].verdict = pv_st; att[0].t = pv_t; snprintf(att[0].why, sizeof att[0].why, "%s", pv_why); natt = 1; }
    double *x = calloc((size_t)L.n + 1, sizeof(double));
    int xo_taken = 0; SimplexInfo xo_inf; memset(&xo_inf, 0, sizeof xo_inf); double xo_t = 0, xo_tipm = 0;    /* (the crossover after a late loss, below) */
    if (M.status) {
        lp_conc_join2(&cc, &cc2, 1);
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
                /* the dedicated interior-point method on the bounded form; the SOCP solver when it gives up */
                LpIpmOpts o; LpIpmInfo inf; memset(&o, 0, sizeof o);
                o.tol = 1e-8; o.kcorr = lpcorr; o.verbose = quiet ? 0 : 1;
                if (cc.started) o.stop = &conc_any;
                if (rlog.on) { o.iter_cb = race_ipm_cb; o.iter_ctx = &rlog; }
                ipm_line[0] = 0;
                for (int i = 0; i < ac; i++) {
                    if (i + 1 < ac && !strcmp(av[i], "-tol")) o.tol = atof(av[i + 1]);
                    if (i + 1 < ac && !strcmp(av[i], "-maxit")) o.maxit = atoi(av[i + 1]);
                    if (i + 1 < ac && !strcmp(av[i], "-timelimit")) o.timelimit = atof(av[i + 1]);
                    if (i + 1 < ac && !strcmp(av[i], "-threads") && atoi(av[i + 1]) >= 1) omp_set_num_threads(atoi(av[i + 1]) - (cc.started ? 1 : 0) - (cc2.started ? 1 : 0));   /* (the factorizations run on the threads; one less beside the dual simplex method) */
                    if (!strcmp(av[i], "-v") || !strcmp(av[i], "-vv")) o.verbose = 2;
                }
                /* (21 October 2026, the night) when the method gives up on the problem the simplex method's presolve left,
                 * it runs again on the problem as read with lpio.c's presolve alone (on 194 files: triptim1, rail01,
                 * k1mushroom, splice1k1, neos-5107597-kakapo and neos-5052403-cygnet failed in the first form and not in
                 * the second); the SOCP solver when it gives up there too */
                for (int attempt = 0; ; attempt++) {
                    memset(&R, 0, sizeof R);
                    R.x = calloc((size_t)P.n + 1, sizeof(double)); R.y = calloc((size_t)P.m + 1, sizeof(double)); R.z = calloc((size_t)P.n + 1, sizeof(double));
                    o.objoff = getenv("BRISK_LPNOOBJOFF") ? 0.0 : lp_std_objoff(Lw, &M, P.n);
                    const double ta = wtime();
                    /* (24 October 2026) on the problem the simplex method's presolve left, the method's own analysis is
                     * compared with that of the problem as read, as on the interior-point route of the simplex method: when
                     * the presolved iteration is expensive and the problem as read's costs at most half (the presolve's
                     * substitutions can fill the normal equations), the method declines and runs on the problem as read
                     * (rail02, LP: 1.05e11 against 3.47e10 flops an iteration; 549 s at four threads, Gurobi 43 s).
                     * BRISK_LPASREAD=0: off */
                    void *arx = NULL;
                    if (SP && attempt == 0 && !(getenv("BRISK_LPASREAD") && atoi(getenv("BRISK_LPASREAD")) == 0)) {
                        arx = sx_asread_new(&L, Lw, quiet ? 0 : 1);
                        o.analyzed = sx_asread_cb; o.analyzed_ctx = arx;
                    }
                    /* (28 October 2026, the night) each attempt within what is left of the run's time limit: each had the whole
                     * limit (shs1014, LP, with 60 s: two attempts of 28 and 18 s, then the SOCP solver's 60 s, 118.7 s in all) */
                    if (tlim_lp > 0) o.timelimit = fmax(tlim_lp - (wtime() - t0), 1e-3);
                    if (getenv("BRISK_LPDBG")) printf("   [t %.3f s: the interior-point method starts]\n", wtime() - t0);
                    const int itab0 = g_itab; g_itab = g_log_brief && !quiet;   /* (3 November 2026: the default log's iteration table) */
                    int st = lpipm_solve(P.m, P.n, P.nf, P.Ap, P.Ai, P.Ax, P.b, P.c, ub, &o, R.x, R.y, R.z, &inf), viol_next = 0;
                    g_itab = itab0;
                    if (getenv("BRISK_LPDBG")) printf("   [t %.3f s: the interior-point method ends, status %d]\n", wtime() - t0, st);
                    /* (2 November 2026) an optimal point of the first attempt (after the presolve of parallel rows, doubleton
                     * equations and slack columns) that violates a row of the problem as read beyond 1e-6 of its activity is a
                     * failed attempt like a failed factorization: the next attempt after the standard presolve, the dual simplex
                     * method still beside it. Before, such a point sent the run to wait for the simplex method, while a failed
                     * factorization of the same attempt went on to the standard presolve: satellites2-60-fs (LP) 2.9 s as written
                     * (the factorization failed, the second attempt 1.5 s), 11-12 s on permuted copies (a point 6.6e-6 off).
                     * BRISK_LPVIOLNEXT=0: as before */
                    {   static int vnext = -1; if (vnext < 0) { const char *e_ = getenv("BRISK_LPVIOLNEXT"); vnext = !(e_ && atoi(e_) == 0); }
                        if (vnext && st == 0 && SP && attempt == 0 && R.x && !(arx && sx_asread_declined(arx))) {
                            lp_point_as_read(&L, Lw, &M, SP, R.x, x);
                            const double v = lp_rowviol_act(&L, x);
                            memset(x, 0, sizeof(double) * (size_t)L.n);
                            if (v > 1e-6) { st = 4; viol_next = 1; snprintf(inf.why, sizeof inf.why, "its point violates a row of the problem as read by %.1e of its activity", v); }
                        }
                    }
                    const int declined = arx && sx_asread_declined(arx);
                    o.analyzed = NULL; o.analyzed_ctx = NULL; sx_asread_free(arx);
                    if (declined) { inf.iters = 0; snprintf(inf.why, sizeof inf.why, "the problem as read costs less an iteration"); }
                    /* (27 October 2026, the night) stopped by the dual simplex method beside it, which ended with a result:
                     * nothing of this attempt is reported */
                    if (cc.started && conc_any) {
                        done = 1; R.status = 6; snprintf(R.status_str, sizeof R.status_str, "STOPPED"); tlast = wtime() - ta;
                        break;
                    }
                    /* (24 October 2026) the last attempt of the method stopped without progress within a hundred times the
                     * tolerance: its best point is returned at reduced accuracy (the advice gives the simplex method's optimal
                     * basic solution) instead of the SOCP solver's solve from the start (rail02, LP: stopped after 54
                     * iterations at 1e-6, then 372 s in the SOCP solver). BRISK_LPNEARKEEP=0: the SOCP solver, as before */
                    /* (31 October 2026, ISSUES 640) not when the dense columns' correction lost its accuracy: the SOCP solver keeps
                     * them apart in its augmented system, stably (l1_regression P316, LP: "reduced accuracy" at a gap of 6.2e-7
                     * here, OPTIMAL in 0.03 s by the SOCP solver) */
                    /* (1 November 2026, ISSUES 811) the dense columns' correction lost near the optimum (every error within a
                     * hundred times the tolerance): the point goes to the crossover on the problem as read, whose pushes and
                     * simplex iterations from a basis next to the optimal one cost a fraction of the SOCP solver's solve from
                     * the start. Q083_energy (LP, 242 dense columns): lost at iteration 45 (gap 4.8e-8), the crossover 6.7 s
                     * (101,174 pushes, 229 simplex iterations), where the SOCP solver took 17 s an iteration until the dual
                     * simplex method beside it ended at 701 s. When the crossover does not end optimal, the SOCP solver as
                     * before. BRISK_LPXOLOST=0: the SOCP solver always */
                    /* (1 November 2026) and likewise a point where the method stalled near the optimum for another reason (no
                     * progress, the step lengths): neos-5052403-cygnet (LP) stalled twice at a gap of 2e-8 (22 iterations, 8.5 s,
                     * then 20 s after the standard presolve) and its point was returned at reduced accuracy after the dual simplex
                     * method beside it had its twice the time, 90 s in all. When the crossover does not end optimal, the route
                     * goes on as before (the second attempt, or the point at reduced accuracy). BRISK_LPXOSTALL=0: only after the
                     * loss of the dense columns' correction */
                    static int xostall = -1; if (xostall < 0) { const char *e = getenv("BRISK_LPXOSTALL"); xostall = !(e && atoi(e) == 0); }
                    static int xobud = -1; if (xobud < 0) { const char *e = getenv("BRISK_LPXOBUDGET"); xobud = !(e && atoi(e) == 0); }
                    const double xo_err = brisk_nmax(brisk_nmax(inf.pinf, inf.dinf), inf.gap);
                    const int xo_band = xo_err <= 100.0 * o.tol;
                    /* (2 November 2026) after a loss of the dense columns' correction outside the band (errors up to 1e-2),
                     * the crossover with a time budget of what the interior-point method has spent, about what the SOCP
                     * solver's new start costs; the SOCP solver as before when it does not finish within it (the run then
                     * longer by at most that time). On the 54 LPs with dense columns and the held-out energy LPs
                     * (-lpmethod 1): 563 -> 523 s, no status changed; rmine13 88.6 -> 54.1 s, ns1905797 0.97 -> 0.63 s,
                     * neos-4306827-ravan 2.9 -> 4.1 s (the attempt fails). BRISK_LPXOBUDGET=0: not */
                    const int xo_budgeted = !xo_band && xobud && inf.dense_lost && xo_err <= 1e-2;
                    /* (3 November 2026) and a point optimal by the method's own measures that violates a row of the problem as
                     * read beyond 1e-6 of its activity (viol_next): the crossover from it first, the second attempt after the
                     * standard presolve only when the crossover does not end optimal. The method's measures are relative to
                     * the largest right-hand side, so on badly scaled data a row of small activity can be off after the
                     * postsolve while the point is optimal to 1e-8 by them: ofi (LP, right-hand sides to 6e11) converged in 35
                     * iterations, its point 1.9e-6 off a row, and the second attempt and the SOCP solver then took 100 s more.
                     * BRISK_LPVIOLXO=0: the second attempt at once, as before */
                    static int violxo = -1; if (violxo < 0) { const char *e = getenv("BRISK_LPVIOLXO"); violxo = !(e && atoi(e) == 0); }
                    if (!declined && st == 4 && (!viol_next || violxo) && (inf.dense_lost || xostall || (viol_next && violxo)) && (xo_band || xo_budgeted)
                        && !(getenv("BRISK_LPXOLOST") && atoi(getenv("BRISK_LPXOLOST")) == 0)) {
                        double *yL = calloc((size_t)L.m + 1, sizeof(double)), *x0 = calloc((size_t)L.n + 1, sizeof(double));
                        lp_point_y_as_read(&L, Lw, &M, SP, R.x, R.y, x0, yL);
                        SimplexOpts so; memset(&so, 0, sizeof so);
                        so.xstart = x0; so.ystart = yL; so.nopresolve = 1; so.verbose = o.verbose >= 2 ? 1 : 0;
                        if (tlim_lp > 0) so.timelimit = fmax(tlim_lp - (wtime() - t0), 1e-3);
                        if (xo_budgeted) so.timelimit = so.timelimit > 0 ? fmin(so.timelimit, fmax(wtime() - ta, 1.0)) : fmax(wtime() - ta, 1.0);
                        if (cc.started) so.stop = &conc_any;
                        if (!quiet) printf("  the LP interior-point method stopped after %d iterations near the optimum (%s; errors pinf %.1e dinf %.1e gap %.1e): its point goes to the crossover\n",
                                           inf.iters, inf.why, inf.pinf, inf.dinf, inf.gap);
                        xo_tipm = wtime() - ta;
                        const double tx = wtime();
                        /* (4 November 2026) a tall problem's crossover through its dual problem (simplex_crossover_dual): a basis
                         * of as many rows as it has columns, where the problem's own needs a dual push for every row whose dual is
                         * not zero (rmine13, LP: 163,343 dual pushes, 14.5 s). Not applied, or not optimal: the crossover on the
                         * problem as read, as before. BRISK_LPXODUAL=0: off */
                        static int xod = -1; if (xod < 0) { const char *e_ = getenv("BRISK_LPXODUAL"); xod = !(e_ && atoi(e_) == 0); }
                        int xst = -1, xo_out = 0;
                        /* (4 November 2026) the crossover on the reduced problem the method solved (the simplex method's presolve), from
                         * its point there, before the postsolve: the point is accurate there (its own errors) and the problem smaller;
                         * the basis is then postsolved and the simplex method finishes on the problem as read from it, as the simplex
                         * method's own presolved route does. On the problem as read, from the postsolved point: ofi (LP), 422,587 rows
                         * against 104,822 reduced, 106,092 pushes and 13,629 simplex iterations, 7.8 s (Gurobi's crossover on its
                         * presolved problem 0.7 s). Not optimal: the crossover on the problem as read, as before.
                         * BRISK_LPXOREDUCED=0: off */
                        static int xored = -1; if (xored < 0) { const char *e_ = getenv("BRISK_LPXOREDUCED"); xored = !(e_ && atoi(e_) == 0); }
                        /* (a tall reduced problem: the crossover through the dual problem below, on the problem as read: rmine13, LP,
                         * 3.65 s that way, 14.9 s on its reduced problem directly) */
                        if (xored && SP && Lw != &L && !(xod && simplex_dual_route_possible(Lw))) {
                            const int mr = Lw->m, nr = Lw->n;
                            double *xr0 = calloc((size_t)nr + 1, sizeof(double)), *yr0 = calloc((size_t)mr + 1, sizeof(double));
                            double *xr = calloc((size_t)nr + 1, sizeof(double)), *yr = calloc((size_t)mr + 1, sizeof(double));
                            int *csr = calloc((size_t)nr + 1, sizeof(int)), *rsr = calloc((size_t)mr + 1, sizeof(int));
                            lp_recover((LpProb *)Lw, &M, R.x, xr0);
                            for (int i = 0; i < mr; i++) yr0[i] = M.rpos[i] >= 0 ? M.rsgn[i] * R.y[M.rpos[i]] : 0.0;
                            SimplexOpts sor = so; sor.xstart = xr0; sor.ystart = yr0; sor.nopresolve = 1;
                            SimplexInfo ir; memset(&ir, 0, sizeof ir);
                            const int s1 = simplex_solve(Lw, &sor, xr, NULL, yr, NULL, csr, rsr, &ir);
                            if (!quiet && so.verbose) printf("  the crossover on the reduced problem: status %d, %ld pushes, %ld iterations (%.2f s)\n", s1, ir.pushes, ir.iters, wtime() - tx);
                            if (s1 == 0) {
                                double *xa = calloc((size_t)L.n + 1, sizeof(double)), *ya = calloc((size_t)L.m + 1, sizeof(double));
                                int *cs = calloc((size_t)L.n + 1, sizeof(int)), *rs = calloc((size_t)L.m + 1, sizeof(int));
                                sxpre_postsolve(SP, xr, yr, csr, rsr, xa, ya, cs, rs);
                                /* (the postsolved solution taken when optimal on the problem as read, as the simplex methods'
                                 * presolved route does: ofi, LP, the solve from its basis was 4 iterations and 0.47 s of
                                 * set-up on 422,587 rows. BRISK_LPXODIRECT=0: always that solve) */
                                static int xodir = -1; if (xodir < 0) { const char *e_ = getenv("BRISK_LPXODIRECT"); xodir = !(e_ && atoi(e_) == 0); }
                                SimplexInfo idr;
                                if (xodir && simplex_postsolved_check(&L, xa, ya, cs, rs, x, &idr)) {
                                    xo_inf = idr; xo_inf.pushes = ir.pushes; xo_inf.iters = ir.iters; xst = 0;
                                    if (!quiet && so.verbose) printf("  the postsolved basis is optimal on the problem as read (%.2f s)\n", wtime() - tx);
                                    free(xa); free(ya); free(cs); free(rs);
                                    goto xored_done;
                                }
                                SimplexOpts sof = so; sof.xstart = NULL; sof.ystart = NULL; sof.cstat0 = cs; sof.rstat0 = rs; sof.nopresolve = 1;
                                if (tlim_lp > 0) sof.timelimit = fmax(tlim_lp - (wtime() - t0), 1e-3);
                                SimplexInfo i2; memset(&i2, 0, sizeof i2);
                                const int s2 = simplex_solve(&L, &sof, x, NULL, NULL, NULL, NULL, NULL, &i2);
                                if (!quiet && so.verbose) printf("  the problem as read from its postsolved basis: status %d, %ld iterations (%.2f s)\n", s2, i2.iters, wtime() - tx);
                                if (s2 == 0) { xo_inf = i2; xo_inf.pushes = ir.pushes; xo_inf.iters += ir.iters; xst = 0; }
                                else memset(x, 0, sizeof(double) * (size_t)L.n);
                                if (s2 == 6) xo_out = 1;
                                free(xa); free(ya); free(cs); free(rs);
                            }
                          xored_done:
                            /* (the time limit or the stop flag reached: no second crossover, which had the same time again:
                             * neos-2974461-ibar, LP, -timelimit 150 ended at 239 s) */
                            if (s1 == 6) xo_out = 1;
                            free(xr0); free(yr0); free(xr); free(yr); free(csr); free(rsr);
                            if (xo_out) { xst = 6; memset(&xo_inf, 0, sizeof xo_inf); xo_inf.status = 6; }
                            else if (tlim_lp > 0) so.timelimit = fmax(tlim_lp - (wtime() - t0), 1e-3);
                        }
                        if (xst < 0 && xod && simplex_dual_route_possible(&L)) {
                            xst = simplex_crossover_dual(&L, &so, x0, yL, x, &xo_inf);
                            if (!quiet && so.verbose) printf("  the crossover through the dual problem: status %d (%.2f s)\n", xst, wtime() - tx);
                            if (xst != 0) { memset(x, 0, sizeof(double) * (size_t)L.n); memset(&xo_inf, 0, sizeof xo_inf); xst = -1; }
                        }
                        if (xst < 0) xst = simplex_solve(&L, &so, x, NULL, NULL, NULL, NULL, NULL, &xo_inf);
                        xo_t = wtime() - tx;
                        free(yL); free(x0);
                        if (xst == 0) {
                            done = 1; lp_done = 1; xo_taken = 1; R.status = -1; rc = 0; tlast = wtime() - ta; sxpre_last = 0;
                            if (natt < 3) { att[natt].sxpre = SP != NULL ? 1 : asread_raw ? 2 : 0; att[natt].iters = inf.iters; att[natt].t = xo_tipm; snprintf(att[natt].why, sizeof att[natt].why, "%s", inf.why); natt++; }
                            break;
                        }
                        if (!quiet) printf("  the crossover ended with status %d (%.2f s)\n", xst, xo_t);
                        if (xo_budgeted) {   /* (the Summary says it: the attempt's time is in the method's line) */
                            const size_t lw = strlen(inf.why);
                            snprintf(inf.why + lw, sizeof inf.why - lw, "; the crossover from its point did not finish in %.1f s", xo_t);
                        }
                        memset(x, 0, sizeof(double) * (size_t)L.n);
                    }
                    const int nearkeep = !declined && st == 4 && !inf.dense_lost && !(SP && attempt == 0) && brisk_nmax(brisk_nmax(inf.pinf, inf.dinf), inf.gap) <= 100.0 * o.tol
                                         && !(getenv("BRISK_LPNEARKEEP") && atoi(getenv("BRISK_LPNEARKEEP")) == 0);
                    if (nearkeep) {
                        done = 1; lp_done = 1; R.status = 5; R.iters = inf.iters; R.pobj = inf.pobj; R.dobj = inf.dobj; R.have_x = 1; R.m = P.m; R.n = P.n;
                        R.err[0] = inf.pinf; R.err[2] = inf.dinf; R.err[4] = inf.gap;
                        snprintf(R.status_str, sizeof R.status_str, "SOLVED TO REDUCED ACCURACY");
                        rc = 10; tlast = wtime() - ta; sxpre_last = SP != NULL ? 1 : asread_raw ? 2 : 0;
                        if (!quiet) snprintf(ipm_line + strlen(ipm_line), sizeof ipm_line - strlen(ipm_line), "status: %s (%s; errors pinf %.1e dinf %.1e gap %.1e)   iterations: %d   time: %.3fs\n"
                                           "  (an optimal basic solution by the simplex method: -lpmethod 5)\n", R.status_str, inf.why, inf.pinf, inf.dinf, inf.gap, inf.iters, inf.time);
                        break;
                    }
                    /* (3 November 2026) the method's iterates diverged along a ray (lpsolve.c, lp_ray_check): dual infeasible,
                     * without the SOCP solver's homogeneous model for the same verdict (neos-3230376-yser, LP: 9 s of 18.7) */
                    if (!declined && st == 2) {
                        done = 1; lp_done = 1; R.status = 2; R.iters = inf.iters; R.have_x = 1; R.m = P.m; R.n = P.n;
                        snprintf(R.status_str, sizeof R.status_str, "DUAL INFEASIBLE");
                        rc = 12; tlast = wtime() - ta; sxpre_last = SP != NULL ? 1 : asread_raw ? 2 : 0;
                        if (!quiet) snprintf(ipm_line + strlen(ipm_line), sizeof ipm_line - strlen(ipm_line), "status: DUAL INFEASIBLE (%s)   iterations: %d   time: %.3fs\n", inf.why, inf.iters, inf.time);
                        break;
                    }
                    if (!declined && (st == 0 || st == 3 || st == 6)) {
                        done = 1; lp_done = 1; R.status = st; R.iters = inf.iters; R.pobj = inf.pobj; R.dobj = inf.dobj; R.have_x = 1; R.m = P.m; R.n = P.n;
                        R.err[0] = inf.pinf; R.err[2] = inf.dinf; R.err[4] = inf.gap;
                        const double e = brisk_nmax(brisk_nmax(inf.pinf, inf.dinf), inf.gap);
                        (void)e;   /* (3 November 2026: the status alone, as for the other solvers; the error is in the Summary and the result) */
                        if (st == 0) snprintf(R.status_str, sizeof R.status_str, "OPTIMAL");
                        else snprintf(R.status_str, sizeof R.status_str, "%s", st == 3 ? "ITERATION LIMIT" : "TIME LIMIT");
                        rc = st == 0 ? 0 : st == 3 ? 3 : 6; tlast = wtime() - ta; sxpre_last = SP != NULL ? 1 : asread_raw ? 2 : 0;
                        if (!quiet) {
                            const size_t q0 = strlen(ipm_line);
                            char eb3[96];
                            snprintf(ipm_line + q0, sizeof ipm_line - q0, "status: %s   iterations: %d   time: %.3fs\n  primal obj c'x = %.12e   dual obj b'y - u'w = %.12e\n"
                                     "  errors: %s;  %ld factorizations %.3fs, %ld solves %.3fs (%ld CG steps), %d correctors\n",
                                     R.status_str, inf.iters, inf.time, inf.pobj, inf.dobj, lp_errtxt(eb3, sizeof eb3, inf.pinf, inf.dinf, inf.gap), inf.nfact, inf.t_fact, inf.nsolve, inf.t_solve, inf.ncg, inf.ncorr);
                        }
                        break;
                    }
                    free(R.x); free(R.y); free(R.z); memset(&R, 0, sizeof R);
                    sedumi_free(&P); lp_map_free(&M); free(ub); ub = NULL;
                    if (natt < 3) { att[natt].sxpre = SP != NULL ? 1 : asread_raw ? 2 : 0; att[natt].iters = inf.iters; att[natt].t = wtime() - ta; snprintf(att[natt].why, sizeof att[natt].why, "%s", inf.why); natt++; }
                    /* (31 October 2026, ISSUES 640) and no second attempt after the standard presolve when the dense columns' correction
                     * lost its accuracy: the same columns meet the same end (neos-914441, LP: both attempts lost it, 45 and 45
                     * iterations, then the SOCP solver) */
                    /* (2 November 2026) nor when the free columns stalled with their weight at its largest and split as well */
                    if (SP && attempt == 0 && !inf.dense_lost && inf.free_stalled != 2) {
                        SedumiProb P2; LpMap M2; double *ub2 = NULL;
                        /* (24 October 2026) after a decline the problem as read in the form its cost was estimated in (no
                         * standard presolve): with it rail02 (LP) became 62,030 rows and 1.33e11 flops an iteration, against
                         * 3.41e10 estimated and 3.30e10 on the simplex method's interior-point route */
                        lp_to_std(&L, declined ? 0 : pre, 0, &P2, &M2, &ub2);
                        if (declined) asread_raw = 1;
                        if (!M2.status && P2.n > 0) {
                            if (!quiet) printf("  the LP interior-point method stopped after %d iterations (%s): solving again %s\n", inf.iters, inf.why, declined ? "on the problem as read" : "after the standard presolve");
                            P = P2; M = M2; ub = ub2; lp_free(&Lr); sxpre_free(SP); SP = NULL; Lw = &L;
                            continue;
                        }
                        sedumi_free(&P2); lp_map_free(&M2); free(ub2);
                    }
                    if (!quiet) printf("  the LP interior-point method stopped after %d iterations (%s): solving with the SOCP solver (homogeneous model)\n", inf.iters, inf.why);
                    lp_to_sedumi(Lw, pre, 0, &P, &M);
                    break;
                }
            }
            /* (27 October 2026, the night; superseded the next night, below) the interior-point method gave up beside the dual
             * simplex method: the SOCP solver (which cannot be stopped from here) only if the simplex method ended without a
             * result, then on every thread */
            /* (28 October 2026, the night) when the simplex method is still running, the SOCP solver starts at once on the
             * interior-point method's threads, beside it: the interior-point method gives up mostly on infeasible or
             * unbounded problems, whose certificate the SOCP solver's self-dual model gives in a few iterations, while the
             * simplex method can run to the time limit (neos-3283608-agout and neos-3230376-yser, LPs of the held-out
             * set: the SOCP solver's certificate after 0.4 and 2.6 s, the run 60 and 66 s, Gurobi 0.03 and 0.26 s). The
             * first to end with a result gives it (below); the SOCP solver cannot be stopped from here, so a simplex
             * method that ends first is taken once it has ended (superseded later that night: the SOCP solver stops when
             * the simplex method has its result, socp_stop_call below). BRISK_LPSOCPWAIT=1: the SOCP solver after the
             * simplex method, as before */
            if (!done && cc.started && (cc.done || getenv("BRISK_LPSOCPWAIT"))) {
                cc.nthr = cc2.started == 1 ? (nt_all > 3 ? nt_all / 2 : 1) : nt_all;     /* (28 October 2026: the interior-point method's threads to the simplex method) */
                cc2.nthr = cc.nthr;
                lp_conc_join2(&cc, &cc2, 0);
                if (cc.won) { done = 1; R.status = 6; snprintf(R.status_str, sizeof R.status_str, "STOPPED"); }
                else omp_set_num_threads(nt_all);
            }
            /* (28 October 2026, later in the night) the SOCP solver stops when the simplex method beside it has its result
             * (socp_stop_call: the flag the simplex job raises; Ctrl-C's flag stays the solver's own) */
            if (!done) {
                volatile int *const ssc0 = socp_stop_call; if (cc.started) socp_stop_call = &conc_any;
                /* (28 October 2026, the night) within what is left of the run's time limit (above): its options with the
                 * time limit replaced */
                char tlbuf[32]; char **av2 = av;
                if (tlim_lp > 0 && (av2 = malloc(sizeof(char *) * ((size_t)ac + 1)))) {
                    memcpy(av2, av, sizeof(char *) * (size_t)ac); av2[ac] = NULL;
                    snprintf(tlbuf, sizeof tlbuf, "%.3f", fmax(tlim_lp - (wtime() - t0), 1e-3));
                    for (int i = 0; i + 1 < ac; i++) if (!strcmp(av2[i], "-timelimit")) av2[i + 1] = tlbuf;
                } else av2 = av;
                const int nolpm0 = g_sedumi_nolpm, itab1 = g_itab; g_sedumi_nolpm = 1; g_itab = g_log_brief && !quiet;   /* (3 November 2026: the LP method has had its turn above, or -lpmethod 0) */
                const double ta = wtime(); rc = sedumi_solve(&P, ac, av2, &R, sedumi_inner_sdp); tlast = wtime() - ta; sxpre_last = 0;
                g_sedumi_nolpm = nolpm0; g_itab = itab1;
                if (av2 != av) free(av2);
                socp_stop_call = ssc0;
            }
        }
        /* (27 October 2026, the night) the dual simplex method beside it: its result when it ended first with one, or when
         * the interior-point method's is not optimal (it is then waited for, within the time limit) */
        int sx_taken = 0, ipm_nores = 0;     /* (ipm_nores: the interior-point method ended first, without a result to take) */
        int x_have = 0; double obj_have = 0; double ipm_viol = 0;
        if (cc.started) {
            /* (the first to end with a result: the simplex method's when it ended before this one did; an optimal point
             * that violates a row of the problem as read beyond 1e-6 of its activity is not one: the check below would
             * report it at reduced accuracy, ken-13, LP) */
            int ipm_ok = (R.status == 0 && R.x) || R.status == 1 || R.status == 2 || xo_taken;
            if (R.status == 0 && R.x) {
                obj_have = lp_point_as_read(&L, Lw, &M, SP, R.x, x); x_have = 1;
                if ((ipm_viol = lp_rowviol_act(&L, x)) > 1e-6) ipm_ok = 0;
            }
            const double t_ipm = wtime() - cc.t0;
            /* (28 October 2026) the interior-point method ended without a result to take: the simplex method, waited for,
             * gets its threads (the simplex method reads the count at every iteration, SimplexOpts.threads; its result
             * does not depend on it). rmine13 (LP): the interior-point method at reduced accuracy after 50 s, the dual
             * simplex method on one thread not done at the 60 s limit; the race of -lpmethod 5 on four threads 26 s */
            if (!ipm_ok) { cc.nthr = cc2.started == 1 ? (nt_all > 3 ? nt_all / 2 : 1) : nt_all; cc2.nthr = cc.nthr; }
            /* (31 October 2026, ISSUES 635) a point at reduced accuracy (within 100 times the tolerance) is a result the user can
             * have: the simplex method then gets twice the time the race has taken, on every thread, so that the call takes at
             * most three times the time at which that result was ready (an optimal basic solution is worth more than the point:
             * with as long again, neos-4531126-vouga's simplex result came 0.06 s after the bound); its optimal basic solution
             * when it ends within that, else the point at reduced accuracy with the advice for an optimal basic solution. fir
             * (LP): the point at 7.7 s, the simplex method to the time limit, 90-100 s in all; rmine13 (LP): the point after
             * 50 s, the simplex method's result at 77-81 s, taken. BRISK_LPWAITX: the factor (2; 0 or negative: no bound) */
            int waited_out = 0;
            if (!ipm_ok && R.status == 5 && cc.started == 1 && !cc.done) {
                static double wx = -2; if (wx < -1) { const char *e_ = getenv("BRISK_LPWAITX"); wx = e_ ? atof(e_) : 2.0; }
                if (wx > 0) {
                    const double tend = cc.t0 + (1.0 + wx) * t_ipm;
                    while (!cc.done && !(cc2.started == 1 && cc2.done && cc2.won) && wtime() < tend) { struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL); }
                    if (!cc.done && !(cc2.started == 1 && cc2.done && cc2.won)) { cc.stop = 1; cc2.stop = 1; waited_out = 1; }
                }
            }
            lp_conc_join2(&cc, &cc2, ipm_ok);
            sx_taken = cc.won && (!ipm_ok || cc.t < t_ipm);
            ipm_nores = sx_taken && !ipm_ok && cc.t >= t_ipm && ipm_viol <= 1e-6;
            /* (28 October 2026) stopped by this side only when its result was taken; otherwise it ended without one (shs1014,
             * LP: at its time limit, reported as stopped by the interior-point method) */
            if (!sx_taken && cc.done && !quiet && !quiet_summary)
                brisk_note("the dual simplex method ran beside it on one thread (%ld iterations, %.2f s), %s", cc.inf.iters, cc.t,
                           waited_out ? "stopped: given twice the time the interior-point method had taken, it had no result" :
                           cc.stop && xo_taken ? "stopped: the crossover from the interior-point method's point finished first" : cc.stop ? "stopped: the interior-point method finished first" : tlim_lp > 0 && cc.t >= 0.99 * tlim_lp ? "stopped by the time limit" : "ended without a result");
        }
        if (!sx_taken && ipm_line[0]) printf("%s", ipm_line);
        ipm_line[0] = 0;
        if (sx_taken) {
            const SimplexInfo *si = &cc.inf; double eb = 0, er = 0;
            static const char *const snm[] = {"OPTIMAL", "PRIMAL INFEASIBLE", "DUAL INFEASIBLE"};
            rc = cc.st == 0 ? 0 : cc.st == 1 ? 11 : 12;
            if (cc.st == 0) lp_errors(&L, cc.x, &eb, &er);
            if (!quiet) {
                printf("status: %s   iterations: %ld   time: %.3fs (the dual simplex method, beside the interior-point method)\n", snm[cc.st], si->iters, cc.t);
                if (cc.st == 0) {
                    printf("objective value (as in the file, %s): %.12e\n", L.maxim ? "maximized" : "minimized", si->obj);
                    printf("  on the problem as read: bound violation %.1e, row violation %.1e (relative); total %.3fs\n", eb, er, wtime() - t0);
                }
                if (!quiet_summary && g_log_brief) {     /* (31 October 2026, ISSUES 639: the default log's Summary) */
                    char a_[32], b_[32]; fmt_count(a_, sizeof a_, L.m); fmt_count(b_, sizeof b_, L.n);
                    g_log_user++;
                    printf("------------------------------------------------------------------------------------------\n");
                    printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a_, b_, L.maxim ? "maximized" : "minimized");
                    printf("  Solve:             dual simplex method, %ld iterations, beside the interior-point method%s\n", si->iters,
                           ipm_viol > 1e-6 || ipm_nores ? " (which ended first without a result to take)" : " (stopped: the dual simplex method finished first)");
                    for (int i = 0; i < g_nnotes; i++) printf("                     %s\n", g_notes[i]);
                    printf("  Time:              %.2f s\n  Number of threads: %d\n", wtime() - t0, nt_all);
                    if (cc.st == 0) {
                        printf("  Status:            OPTIMAL\n  Objective:         %.10e\n", si->obj);
                        printf("  Errors:            variable bounds violated by %.1e, constraints by %.1e (relative)\nSolved to relative error %.1e\n", eb, er, fmax(eb, er));
                    } else printf("  Verdict:           the problem is %s\n  Status:            %s\n", cc.st == 1 ? "primal infeasible" : "dual infeasible (unbounded)", snm[cc.st]);
                    g_log_user--;
                } else if (!quiet_summary) {
                    char a_[32], b_[32]; fmt_count(a_, sizeof a_, L.m); fmt_count(b_, sizeof b_, L.n);
                    printf("------------------------------------------------------------------------------------------\n");
                    printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a_, b_, L.maxim ? "maximized" : "minimized");
                    if (ipm_viol > 1e-6)
                        printf("  Solve 1:           LP interior-point method on %d threads: its point violates a row of the problem as read by %.1e of its activity; the dual simplex method beside it was waited for\n", nt_ipm, ipm_viol);
                    else if (ipm_nores)
                        printf("  Solve 1:           LP interior-point method on %d threads: %s, not a result to take; the dual simplex method beside it was waited for\n", nt_ipm,
                               R.status == 5 ? "solved to reduced accuracy" : R.status == 3 ? "at its iteration limit" : R.status == 6 && !strcmp(R.status_str, "TIME LIMIT") ? "at the time limit" : "it gave up");
                    else
                        printf("  Solve 1:           LP interior-point method on %d threads, stopped: the dual simplex method beside it ended first\n", nt_ipm);
                    printf("  Solve 2:           dual simplex method beside it%s (one thread%s), %ld iterations (%ld dual, %ld primal), %d factorizations (%.2f s); its result is returned\n",
                           cc.o.aggb_only ? " on the problem with the bounded substitution of columns without cost" : cc.o.no_dual_route ? " on the problem itself (not through its dual problem)" : "",
                           ipm_nores ? ", more once the interior-point method had ended" : "", si->iters, si->iters_dual, si->iters_primal, si->nfactor, cc.t);
                    printf("  Time:              %.2f s\n", wtime() - t0);
                    if (cc.st == 0) printf("  Status:            OPTIMAL\n  Errors:            variable bounds violated by %.1e, constraints by %.1e (relative)\nSolved to relative error %.1e\n", eb, er, fmax(eb, er));
                    else printf("  Verdict:           the problem is %s\n  Status:            %s\n", cc.st == 1 ? "primal infeasible" : "dual infeasible (unbounded)", cc.st == 1 ? "PRIMAL INFEASIBLE" : "DUAL INFEASIBLE");
                }
            }
            if (xf) sedumi_write_vec(xf, cc.x, L.n);
            if (g_brisk_result) {
                BriskResult *res = g_brisk_result;
                res->status = cc.st; snprintf(res->status_str, sizeof res->status_str, "%s", snm[cc.st]);
                res->exit_code = rc; res->iters = (int)si->iters; res->m = L.m; res->pobj = si->obj; res->dobj = si->obj; res->time = wtime() - t0;
                res->sn = L.n; res->have_x = 1; res->sx = cc.x; cc.x = NULL;
            }
        }
        if (xo_taken && !sx_taken) {    /* (1 November 2026) the crossover's optimal basic solution after the late loss */
            const SimplexInfo *si = &xo_inf; double eb = 0, er = 0;
            lp_errors(&L, x, &eb, &er);
            if (!quiet) {
                printf("status: OPTIMAL   iterations: %ld   time: %.3fs (the crossover from the interior-point method's point)\n", si->iters, xo_t);
                printf("objective value (as in the file, %s): %.12e\n", L.maxim ? "maximized" : "minimized", si->obj);
                printf("  on the problem as read: bound violation %.1e, row violation %.1e (relative); total %.3fs\n", eb, er, wtime() - t0);
                if (!quiet_summary) {
                    char a_[32], b_[32], pc_[32]; fmt_count(a_, sizeof a_, L.m); fmt_count(b_, sizeof b_, L.n); fmt_count(pc_, sizeof pc_, (long)si->pushes);
                    if (g_log_brief) g_log_user++;
                    printf("------------------------------------------------------------------------------------------\n");
                    printf("Summary\n  Problem as read:   %s rows, %s columns (linear program, %s)\n", a_, b_, L.maxim ? "maximized" : "minimized");
                    printf("  Solve 1:           LP interior-point method, %d iterations (%.2f s), stopped near the optimum: %s\n", att[natt - 1].iters, xo_tipm, att[natt - 1].why);
                    printf("  Solve 2:           crossover from its point to an optimal basis: %s pushes, %ld simplex iterations (%.2f s); its result is returned\n", pc_, si->iters, xo_t);
                    for (int i = 0; i < g_nnotes; i++) printf("                     %s\n", g_notes[i]);
                    printf("  Time:              %.2f s\n", wtime() - t0);
                    printf("  Status:            OPTIMAL\n  Objective:         %.10e\n", si->obj);
                    printf("  Errors:            variable bounds violated by %.1e, constraints by %.1e (relative)\nSolved to relative error %.1e\n", eb, er, fmax(eb, er));
                    if (g_log_brief) g_log_user--;
                }
            }
            if (xf) sedumi_write_vec(xf, x, L.n);
            if (g_brisk_result) {
                BriskResult *res = g_brisk_result;
                res->status = 0; snprintf(res->status_str, sizeof res->status_str, "OPTIMAL");
                res->exit_code = 0; res->iters = (int)si->iters; res->m = L.m; res->pobj = si->obj; res->dobj = si->obj; res->time = wtime() - t0;
                res->sn = L.n; res->have_x = 1; res->sx = malloc(sizeof(double) * ((size_t)L.n + 1)); memcpy(res->sx, x, sizeof(double) * (size_t)L.n);
            }
        }
        if (!sx_taken && R.status >= 0 && R.x) {
            const double obj = x_have ? obj_have : lp_point_as_read(&L, Lw, &M, SP, R.x, x);
            if (R.status == 0 || R.status == 5 || R.status == 3 || R.status == 4 || R.status == 6) {
                double eb, er; lp_errors(&L, x, &eb, &er);
                /* (21 October 2026, the night) not optimal when a row of the problem as read is violated by more than
                 * 100 times the tolerance of its activity: the SOCP solver, when it takes over, measures its errors
                 * against norms that rows of size 1e3 can drown in (piperout-08 as read: 2.3e-3, its objective off by
                 * 1.3e-3; triptim1 after the simplex method's presolve: 1e-3), and called such points optimal */
                if (R.status == 0) {
                    const double ea = lp_rowviol_act(&L, x);
                    if (ea > 1e-6) {
                        R.status = 5; snprintf(R.status_str, sizeof R.status_str, "SOLVED TO REDUCED ACCURACY"); rc = 10;
                        if (!quiet) printf("status: SOLVED TO REDUCED ACCURACY: a row of the problem as read is violated by %.1e of its activity\n"
                                           "  (an optimal basic solution by the simplex method: -lpmethod 5)\n", ea);
                    }
                }
                if (!quiet) {
                    printf("objective value (as in the file, %s): %.12e\n", L.maxim ? "maximized" : "minimized", obj);
                    printf("  on the problem as read: bound violation %.1e, row violation %.1e (relative); presolve %.3fs, total %.3fs\n", eb, er, tpre, wtime() - t0);
                    if (!quiet_summary) lp_summary(&L, &P, &R, lp_done, eb, er, tpre, wtime() - t0, natt, att, sxpre_last, tlast, obj);
                }
            } else if ((R.status == 1 || R.status == 2) && !quiet && !quiet_summary)
                lp_summary(&L, &P, &R, lp_done, 0, 0, tpre, wtime() - t0, natt, att, sxpre_last, tlast, 0.0);   /* (its verdict; 27 October 2026: none was printed) */
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
    lp_conc_free(&cc); lp_conc_free(&cc2);
    free(x); free(ub); sedumi_free(&P); lp_map_free(&M); lp_free(&L); free(av);
    if (SP) { lp_free(&Lr); sxpre_free(SP); }
    return rc;
}
/* the command line on a CBF file (CBLIB): linear and second-order cones */
/* (22 October 2026, the SOCP developer) the command line on a QPS file: a QP (a quadratic objective, QUADOBJ, QMATRIX
 * or QSECTION) by the QP solver (by default its active-set method for a small QP with H positive definite, its
 * interior-point method otherwise; -qpas, -qpipm: that method), a QCQP (quadratic constraints,
 * QCMATRIX) by the SOCP solver through its reformulation (qcqp.c) */
extern int (*qcqp_socp)(const SedumiProb *P, int argc, char **argv, SedumiRes *R);
static int qcqp_socp_cli(const SedumiProb *P, int argc, char **argv, SedumiRes *R) { return sedumi_solve(P, argc, argv, R, sedumi_inner_sdp); }
/* (22 October 2026) a QP or QCQP in memory (brisk_run_qp, the library's brisk_solve_qp): its name in argv stands for it */
static const QpProb *g_qp_data = NULL;
static const char g_qp_data_name[] = "(QP in memory)";
/* (22 October 2026) a library call's result from a QP or QCQP solve: the status, the value, x, the rows' y, the bounds'
 * z (taken over from R), the errors; the one solve's (brisk_main_qp) and the solver object's re-solves' (capi.c) */
void brisk_qp_fill_result(BriskResult *res, int n, int m, QpRes *R, double t) {
    const int st = R->status;
    res->status = st == QP_OPTIMAL ? 0 : st == QP_PRIMAL_INFEASIBLE ? 1 : st == QP_DUAL_INFEASIBLE ? 2 : st == QP_MAXIT ? 3 : st == QP_REDUCED ? 5 : st == QP_TIME ? 6 : st == QP_INPUT ? -1 : 4;      /* (24 October 2026: an input refused, e.g. not convex: -1, not solved; was 4) */
    snprintf(res->status_str, sizeof res->status_str, "%s", qp_status_name(st));
    res->exit_code = st == QP_OPTIMAL || st == QP_REDUCED ? 0 : 1; res->iters = R->iter; res->m = m; res->pobj = R->obj; res->dobj = R->obj; res->time = t;
    res->err[1] = R->pres; res->err[3] = R->dres; res->err[5] = R->gap;
    res->sn = n; res->have_x = 1; res->sx = R->x; R->x = NULL; res->y = R->y; R->y = NULL; res->sz = R->z; R->z = NULL;
}
/* (23 October 2026, the SOCP developer) the advice after a QP solve that ran twice by the default choice: the active-set
 * method first, not optimal, then the interior-point method; how to ask for the interior-point method at once, in the
 * caller's syntax (a CVXPY caller here is the QP solver class BRISK_QP) */
static void qp_advice_ipm(void) {
    const char *cmd; const int k = caller_kind(&cmd);
    if (k == CK_CVXPY) { printf("To use the interior-point method at once: prob.solve(solver=brisk.BRISK_QP(), active_set=False)\n"); return; }
    if (k == CK_PY) { printf("To use the interior-point method at once: options=[\"-qpipm\"] in brisk.%s(...)\n", cmd); return; }
    printf("To use the interior-point method at once: %s%s\n", syn_kv("qpipm", NULL, 0), syn_head());
}
static int brisk_main_qp(int argc, char **argv, int ifile) {
    const double t0 = wtime();
    QpOpts o; qp_default_opts(&o); o.verbose = 1;
    const char *xout = NULL;
    for (int i = 1; i < argc; i++) {
        if (i == ifile) continue;
        if (!strcmp(argv[i], "-qpas")) o.method = QP_METHOD_ACTIVESET;
        else if (!strcmp(argv[i], "-qpipm")) o.method = QP_METHOD_IPM;
        else if (!strcmp(argv[i], "-tol") && i + 1 < argc) o.tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "-maxit") && i + 1 < argc) o.maxit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-timelimit") && i + 1 < argc) o.timelimit = atof(argv[++i]);
        else if (!strcmp(argv[i], "-threads") && i + 1 < argc) { const int t = atoi(argv[++i]); if (t > 0) { omp_set_num_threads(t); g_threads_user = 1; } }
        else if (!strcmp(argv[i], "-x") && i + 1 < argc) xout = argv[++i];
        else if (!strcmp(argv[i], "-q")) o.verbose = 0;
        else if (!strcmp(argv[i], "-v")) o.verbose = 2;
        else if (!strcmp(argv[i], "-vv")) o.verbose = 3;
        else if (!strcmp(argv[i], "-qpauto")) o.method = QP_METHOD_AUTO;
        else fprintf(stderr, "brisk: option %s is not used for a QP or QCQP file (its options: -qpas -qpipm -tol -maxit -timelimit -threads -x -q -v -vv)\n", argv[i]);
    }
    QpProb P; char msg[512];
    const int inmem = g_qp_data && argv[ifile] == g_qp_data_name;
    if (inmem) P = *g_qp_data;
    else if (qp_read_qps(argv[ifile], &P, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; }
    const double tread = wtime() - t0;
    /* 2 November 2026: the default log as the other solvers': the problem, the threads and the method before the solve, the
     * status in the Summary and its last line; the solvers' own lines are the diagnosis (-v) */
    const int brief = o.verbose == 1 && log_brief_wanted(argc, argv);
    if (brief) {
        /* (3 November 2026) the problem line in the other solvers' words; the method used is printed once it is known (the solve
         * prints nothing in the default log), not the rule that chooses it */
        if (P.nqc) printf("Problem:  %d row%s, %d variables, %d nonzeros, %d in the quadratic objective (convex QCQP: %d quadratic constraint%s)\n",
                          P.m, P.m == 1 ? "" : "s", P.n, P.Ap[P.n], P.Hp ? P.Hp[P.n] : 0, P.nqc, P.nqc == 1 ? "" : "s");
        else printf("Problem:  %d row%s, %d variables, %d nonzeros, %d in the quadratic objective (convex QP, minimized)\n",
                    P.m, P.m == 1 ? "" : "s", P.n, P.Ap[P.n], P.Hp ? P.Hp[P.n] : 0);
        int nth = 1;
#ifdef _OPENMP
        nth = omp_get_max_threads();
#endif
        printf("Number of threads: %d%s\n", nth, brisk_threads_label(nth));
        o.verbose = 0;
    } else if (o.verbose) printf("%s problem %s (%s): n = %d, m = %d, nnz(A) = %d, nnz(H, lower triangle) = %d; read in %.2f s\n",
                          P.nqc ? "QCQP" : "QP", argv[ifile], P.name[0] ? P.name : "-", P.n, P.m, P.Ap[P.n], P.Hp ? P.Hp[P.n] : 0, tread);
    if (P.nqc && o.verbose) printf("  %d quadratic constraints\n", P.nqc);
    qcqp_socp = qcqp_socp_cli;
    qp_printf = brisk_printf;        /* (the output hook; 31 October 2026, ISSUES 725: the command line too, so that the default log filters it) */
    QpRes R;
    extern int g_stab;
    if (brief) {   /* (3 November 2026) the Method line above the iteration table: the choice qp_solve makes (qp.c) */
        const int m1 = P.nqc ? -1 : o.method == QP_METHOD_AUTO ? (qp_as_small(&P) ? QP_METHOD_ACTIVESET : QP_METHOD_IPM) : o.method;
        printf("Method:   %s\n", m1 < 0 ? "SOCP solver (each quadratic constraint as a rotated second-order cone)"
               : m1 == QP_METHOD_ACTIVESET ? "QP active-set method (the dual method of Goldfarb and Idnani)" : "QP interior-point method");
    }
    const int itab0 = g_itab, stab0 = g_stab; g_itab = brief; g_stab = brief;   /* (3 November 2026: the default log's iteration
                                                                                  * tables: the interior-point and the active-set methods) */
    qp_solve(&P, &o, &R);
    g_itab = itab0; g_stab = stab0;
    const int st = R.status;
    if (o.verbose || brief) {   /* (26 October 2026, the merge for 1.6: -q quiet as for the other formats) */
        printf("------------------------------------------------------------------------------------------\n");
        printf("Summary\n");
        if (st == QP_INPUT) {      /* (24 October 2026) refused before any method (e.g. not convex: the reason above) */
            printf("  Problem as read:   %s, n = %d variables, m = %d rows\n", P.nqc ? "QCQP" : "QP", P.n, P.m);
            printf("  Solve:             none: the problem was refused (the reason is printed above)\n");
            printf("  Time:              %.2f s\n", wtime() - t0);
            printf("%s\n", qp_status_name(st));
            goto summary_done;
        }
        printf("  Problem as read:   %d row%s, %d variables (%s)\n", P.m, P.m == 1 ? "" : "s", P.n, P.nqc ? "convex QCQP" : "convex QP");
        const int used = R.method ? R.method : o.method;
        printf("  Solve:             %s, %d iterations\n", P.nqc ? "SOCP solver (each quadratic constraint as a rotated second-order cone)"
               : used == QP_METHOD_ACTIVESET ? "QP active-set method (the dual method of Goldfarb and Idnani)" : "QP interior-point method", R.iter);
        if (R.t_as > 0) printf("                     Solve 1: QP active-set method, not optimal (%.2f s)\n"
                               "                     Solve 2: QP interior-point method (%.2f s)\n", R.t_as, R.time);
        if (R.resolves == 2) printf("                     Solve 1: SOCP solver at the tolerance %.0e (%.2f s)\n"
                                    "                     Solve 2: SOCP solver at the tolerance %.1e, which the errors on the problem as read\n"
                                    "                     needed (%.2f s)%s\n", o.tol > 0 ? o.tol : 1e-8, R.t_first, R.tol2, R.time - R.t_first,
                                    R.kept2 ? "" : "; not better, the first solve's result is returned");
        else if (R.resolves) printf("                     Solve 1: QP interior-point method (%.2f s)\n"
                               "                     Solve 2: QP interior-point method, other linearly dependent equality rows left out, chosen by their\n"
                               "                     activity at the first solve's point (%.2f s)\n", R.t_first, R.time - R.t_first);
        printf("  Time:              %.2f s\n", wtime() - t0);   /* (3 November 2026: two decimals, as the other solvers) */
        { int nth = 1;
#ifdef _OPENMP
          nth = omp_get_max_threads();
#endif
          printf("  Number of threads: %d%s\n", nth, brisk_threads_label(nth)); }
        printf("  Status:            %s\n", qp_status_name(st));
        printf("  Objective:         %.10e\n", R.obj);
        printf("  Errors:            primal %.1e  dual %.1e  gap %.1e (relative, on the problem as read)\n", R.pres, R.dres, R.gap);
        const double qe = fmax(R.pres, fmax(R.dres, fabs(R.gap)));
        if (st == QP_OPTIMAL || st == QP_REDUCED) printf("Solved to relative error %.1e\n", qe); else printf("%s\n", qp_status_name(st));
        if (R.t_as > 0) qp_advice_ipm();
    }
summary_done:
    if (xout) { FILE *f = brisk_out_open(xout); if (f) { for (int j = 0; j < P.n; j++) fprintf(f, "%.17g\n", R.x[j]); brisk_out_close(f, xout); } }
    if (g_brisk_result) brisk_qp_fill_result(g_brisk_result, P.n, P.m, &R, wtime() - t0);   /* (a library call on a QP or QCQP file) */
    qp_res_free(&R); if (!inmem) qp_prob_free(&P);
    return st == QP_OPTIMAL ? 0 : st == QP_REDUCED ? 0 : 1;
}
static int brisk_main_cbf(int argc, char **argv, int ifile) {
    SedumiProb P; char msg[400]; double c0 = 0; int maxim = 0, nint = 0, nvar = 0, *vmap = NULL; signed char *vmsgn = NULL;
    const char *xout = NULL;     /* (27 October 2026) -x <file>: x in the file's own variables, one value per line */
    for (int i = 1; i + 1 < argc; i++) if (i != ifile && !strcmp(argv[i], "-x")) xout = argv[i + 1];
    const double t0 = wtime();
    if (cbf_read_vmap(argv[ifile], &P, &c0, &maxim, &nint, xout ? &vmap : NULL, xout ? &vmsgn : NULL, &nvar, msg, sizeof msg)) { fprintf(stderr, "brisk: %s\n", msg); return 2; }
    g_cone_tread = wtime() - t0;
    if (maxim) for (int j = 0; j < P.n; j++) P.c[j] = -P.c[j];
    char **av = malloc(sizeof(char *) * ((size_t)argc + 1));
    int ac = 0, quiet = 0;
    for (int i = 1; i < argc; i++) { if (i == ifile) continue; if (!strcmp(argv[i], "-q")) quiet = 1; av[ac++] = argv[i]; }
    const int brief = !quiet && P.ns == 0 && log_brief_wanted(argc, argv);   /* (2 November 2026: the default log, cone_preamble) */
    if (brief) {
        if (nint) printf("Note: %d integer variables are relaxed (the continuous relaxation is solved)\n", nint);
        cone_lpm0 = args_lpm0(argc, argv); cone_preamble(&P);
    } else if (!quiet) {
        int qmax = 0; for (int k = 0; k < P.nq; k++) if (P.q[k] > qmax) qmax = P.q[k];
        for (int k = 0; k < P.nr; k++) if (P.r[k] > qmax) qmax = P.r[k];
        printf("CBF problem %s: m = %d, n = %d; free %d, nonnegative %d, second-order cones %d, rotated %d (largest %d), read %.2fs\n", argv[ifile], P.m, P.n, P.nf, P.nl, P.nq, P.nr, qmax, wtime() - t0);
        cone_problem_line(&P);
        if (nint) { if (g_log_brief) log_user("          %d integer variables: relaxed (the continuous relaxation is solved)\n", nint); else printf("  %d integer variables: relaxed (the continuous relaxation is solved)\n", nint); }
    }
    SedumiRes R;
    const int cb0 = g_cone_brief; g_cone_brief = brief;
    const int rc = sedumi_solve(&P, ac, av, &R, sedumi_inner_sdp);
    g_cone_brief = cb0;
    if (R.status >= 0) {
        const double obj = (maxim ? -R.pobj : R.pobj) + c0;
        if (!quiet && !brief && R.status != 1 && R.status != 2) printf("objective value (as in the file, %s): %.12e\n", maxim ? "maximized" : "minimized", obj);
        snprintf(g_cone_objline, sizeof g_cone_objline, "%.10e (as in the file, %s)", obj, maxim ? "maximized" : "minimized");
        if (xout && vmap && R.x) {     /* (an unbounded problem: the certificate x; an infeasible one has no x to write) */
            FILE *fx = brisk_out_open(xout);
            if (fx) { for (int j = 0; j < nvar; j++) fprintf(fx, "%.17g\n", vmap[j] >= 0 ? vmsgn[j] * R.x[vmap[j]] : 0.0); brisk_out_close(fx, xout); }
        }
        g_cone_fobj_set = 1; g_cone_fobj_max = maxim; g_cone_fobj = obj; g_cone_fdual = (maxim ? -R.dobj : R.dobj) + c0;
        if (!quiet && R.cone && !quiet_summary) cone_summary(&P, &R);
        g_cone_objline[0] = 0;
        if (g_brisk_result) {
            BriskResult *res = g_brisk_result;
            res->status = R.status; snprintf(res->status_str, sizeof res->status_str, "%s", R.status_str);
            res->exit_code = rc; res->iters = R.iters; res->m = P.m; res->pobj = obj; res->dobj = (maxim ? -R.dobj : R.dobj) + c0; res->time = R.time;
            for (int i = 0; i < 6; i++) res->err[i + 1] = R.err[i];
            res->y = R.y; R.y = NULL; res->sn = P.n; res->have_x = 1; res->sx = R.x; R.x = NULL; res->sz = R.z; R.z = NULL;
        }
    }
    sedumi_result_free(&R); sedumi_free(&P); free(av); free(vmap); free(vmsgn);
    return rc;
}
static int brisk_main_run(int argc, char **argv);
/* ISSUES 725: the default log's mode is the run's own; a nested run (the semidefinite solver under a SeDuMi problem, the
 * double solve inside a high-precision one) leaves the outer run's mode as it found it */
static double g_brisk_deadline = 0;   /* (brisk.h: one deadline for the whole solve) */
double brisk_deadline(void) { return g_brisk_deadline; }
double brisk_time_left(void) { return g_brisk_deadline > 0 ? g_brisk_deadline - wtime() : INFINITY; }
double brisk_deadline_tighten(double t_abs) {
    const double prev = g_brisk_deadline;
    if (t_abs > 0 && !(g_brisk_deadline > 0 && g_brisk_deadline <= t_abs)) g_brisk_deadline = t_abs;
    return prev;
}
void brisk_deadline_restore(double prev) { g_brisk_deadline = prev; }
int brisk_main(int argc, char **argv) {
    const int brief0 = g_log_brief, user0 = g_log_user;
    const double dl0 = brisk_deadline();   /* (restored on return: the deadline is the call's) */
    if (!g_hp_inner) { extern int g_dup_inexact; g_dup_inexact = 0; g_out_fail = 0; g_out_said = 0; }   /* (audit A20: per run, before the problem is read) */
    int rc = brisk_main_run(argc, argv);
    /* (5 November 2026, ISSUES 820) a file of the solution not written completely: said in the Summary where the
     * Summary follows the writing, else here after it; exit code 16 in place of 0 or 10 (the other codes say more) */
    if (!g_hp_inner && g_out_fail) {
        if (!g_out_said) { g_log_user++; brisk_out_summary_line(); g_log_user--; }
        if (rc == 0 || rc == 10) rc = 16;
    }
    g_log_brief = brief0; g_log_user = user0;
    brisk_deadline_restore(dl0);
    return rc;
}
/* ISSUES 725: the default log (only the user's lines) unless -q, -v, -vv or -vvv is given, or BRISK_LOGDIAG is set */
static int log_brief_wanted_in(int argc, char **argv);
static int log_brief_wanted(int argc, char **argv) {
    int q = 0; for (int i = 1; i < argc; i++) if (argv[i] && (!strcmp(argv[i], "-q") || !strcmp(argv[i], "-silent"))) q = 1;
    const int r = log_brief_wanted_in(argc, argv);
    g_log_wrap = r || q;   /* (the diagnosis log is not wrapped) */
    return r;
}
static int log_brief_wanted_in(int argc, char **argv) {
    if (getenv("BRISK_LOGDIAG")) return 0;
    for (int i = 0; i < argc; i++) {        /* (from 0: a library call's options start there, the command line's program name never matches) */
        const char *a = argv[i];
        if (!a) continue;
        if (!strcmp(a, "-q") || !strcmp(a, "-silent")) return 0;
        if (!strcmp(a, "-v") || !strcmp(a, "-vv") || !strcmp(a, "-vvv"))
            return i + 1 < argc && argv[i + 1] && !strcmp(argv[i + 1], "0");   /* (the interfaces' "-v 0" drops the flag) */
    }
    return 1;
}
/* 31 October 2026 (the SOCP audit's A3, handed to the integrator): the options of a call checked by the one parser, without
 * solving; the SOCP route of a SeDuMi problem (sedumi.c socp_options) had its own short parser that skipped every word it
 * did not know, so a mistyped option ({"threds": 4}) was dropped without a word. Returns 0 when every option is known
 * (the parser prints the message otherwise). */
int brisk_fault(const char *name) {
    static const char *f = (const char *)1;
    if (f == (const char *)1) f = getenv("BRISK_FAULT");
    return f && strstr(f, name) != NULL;
}
static int g_opts_only = 0;
int brisk_options_valid(int argc, char **argv) {
    char **av = malloc(sizeof(char *) * ((size_t)argc + 2));
    if (!av) return 0;
    av[0] = "brisk";
    for (int i = 0; i < argc; i++) av[i + 1] = argv[i];
    g_opts_only = 1;
    const int rc = brisk_main_run(argc + 1, av);
    g_opts_only = 0;
    free(av);
    return rc == 0;
}
static int brisk_main_run(int argc, char **argv) {
    if (g_opts_only) goto opts_only;
    brisk_threads_init();
    brisk_log_threads_reset();
    if (!g_hp_inner) notes_clear();                                                /* 5.8: the summary's notes */
    if (!g_hp_inner) { lowrank_rng_reset(); symred_rng_reset(); }                  /* 5.15: each solve as in a fresh process */
    quiet_summary = getenv("BRISK_NOSUMMARY") != NULL || g_hp_inner;
    if (!g_hp_inner) advice_scan(argc, argv, 1);
    { extern void (*socp_note)(const char *, ...); socp_note = brisk_note; }
    g_threads_user = getenv("OMP_NUM_THREADS") != NULL;   /* 4.42: set also by -threads */
    /* 31 October 2026 (ISSUES 727): -threads k (k >= 1) forces k threads for this solve, no automatic reduction; set
     * here for the command line and the outermost library call (nested calls keep the outer solve's choice) */
    if (g_caller_depth <= 1 && !g_hp_inner) g_threads_forced = threads_arg(argc, argv) > 0;
    /* (31 October 2026, the LP developer; PLAN_lp_1031.md P1.5) the automatic count within the container's CPU quota
     * (brisk_cpu_quota, solver.c): the runtime counts the CPUs the process may run on, not the time a quota lets it run
     * (the owner's benchmarks of 29 October: 5 CPUs visible, a quota of 4, 5 busy threads). Not when OMP_NUM_THREADS or
     * -threads gives the count; a library call's count is given back on return (nt_scope_end) */
    if (g_caller_depth <= 1 && !g_hp_inner && !g_threads_user && !g_threads_forced) {
        const int q = brisk_cpu_quota();
        if (q > 0 && q < omp_get_max_threads()) omp_set_num_threads(q);
    }
    if (argc < 2) { usage(); return 1; }
    log_version(argc, argv);
    for (int i = 1; i < argc; i++) if (g_qp_data && argv[i] == g_qp_data_name) return brisk_main_qp(argc, argv, i);
    for (int i = 1; i < argc; i++) {       /* a MAT-file: a problem in SeDuMi format */
        const size_t l = argv[i] ? strlen(argv[i]) : 0;
        if (l > 4 && argv[i][0] != '-' && !strcmp(argv[i] + l - 4, ".mat")
            && !(i > 1 && (!strcmp(argv[i - 1], "-x") || !strcmp(argv[i - 1], "-y") || !strcmp(argv[i - 1], "-z"))) && sedumi_is_mat(argv[i]))
            return brisk_main_sedumi(argc, argv, i);
        if (l > 4 && argv[i][0] != '-' && !(i > 1 && !strcmp(argv[i - 1], "-x"))
            && (!strcmp(argv[i] + l - 4, ".qps") || !strcmp(argv[i] + l - 4, ".QPS") || (lp_is_mps(argv[i]) && qp_is_qps(argv[i]))))
            return brisk_main_qp(argc, argv, i);
        if (l > 4 && argv[i][0] != '-' && lp_is_mps(argv[i]) && !(i > 1 && !strcmp(argv[i - 1], "-x")))
            return brisk_main_lp(argc, argv, i);
        if (l > 4 && argv[i][0] != '-' && cbf_is_file(argv[i]))
            return brisk_main_cbf(argc, argv, i);
    }
opts_only:;
    Params par;
    params_default(&par);
    if (!g_hp_inner && !g_opts_only) g_log_brief = log_brief_wanted(argc, argv);   /* (ISSUES 725: the semidefinite solver's route) */
    const char *fname = NULL, *yfile = NULL, *xfile = NULL, *zfile = NULL, *boundxfile = NULL, *boundyfile = NULL;
    const char *cert_x = NULL, *cert_y = NULL;   /* 4.39: certify a given certificate, no solve */
    int do_certify = 0;                          /* 4.39: -certify (off by default) */
    int force_route = -1, do_fr = 1, fr_retry = 1, acc_set = -1;
    /* -acc first, so that explicit -tol / -retryacc override it wherever they appear */
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "-acc"))
            params_accuracy(&par, !strcmp(argv[i + 1], "low") ? 0 : !strcmp(argv[i + 1], "high") ? 2 : 1);
    int dir_cli = -2;      /* -dir as given (-2: not given) */
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        {   /* 4.41: a flag may carry an explicit 0/1 (true/false) - the interfaces pass
             * {"nohsd": 1} or opts.nohsd = 1 as "-nohsd 1"; 0 drops the flag */
            static const char *const flags[] = { "-q", "-v", "-vv", "-vvv", "-nofr", "-fr", "-nofrretry", "-nosplit", "-split", "-nopolish",
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
        OPT_I("-mixed", mixed) OPT_D("-mixedfrac", mixed_frac) OPT_I("-lowrank", lowrank) OPT_I("-dict", dict) OPT_I("-cls", cls) OPT_D("-gmax", gamma_max)
        OPT_I("-ntls", nt_ls) OPT_D("-ntlsgap", nt_ls_gap) OPT_I("-stallwin", stall_win)
        OPT_I("-sparse", sparse_schur) OPT_D("-balance", balance) OPT_D("-pivtol", pivtol)
        OPT_D("-feasgrow", feas_grow) OPT_D("-sigmamin", sigma_min) OPT_D("-dualrho", dual_rho)
        OPT_I("-dualcert", dual_cert) OPT_D("-dualmargin", dual_margin) OPT_I("-dualgiveup", dual_giveup) OPT_D("-dualtau", dual_tau) OPT_D("-dualtaulong", dual_tau_long) OPT_D("-dualgamma", dual_gamma) OPT_I("-dualmurule", dual_murule) OPT_I("-dualls", dual_ls) OPT_I("-dualcorr", dual_corr) OPT_I("-dualmured", dual_mured) OPT_D("-dualtaunb", dual_tau_nb) OPT_D("-dualtauub", dual_tau_ub) OPT_I("-dualsparse", dual_sparse_n) OPT_I("-zsparse", zsparse_n) OPT_D("-zsparsefill", zsparse_fill) OPT_I("-dualcorrauto", dual_corr_auto) OPT_I("-dualgstart", dual_gstart) OPT_D("-dualmudrop", dual_mudrop) OPT_I("-dualverifyn", dual_verify_n)
        OPT_I("-chordal", chordal) OPT_I("-cliquemax", chordal_maxclique) OPT_I("-chordalsup", chordal_sup) OPT_D("-chordalmerge", chordal_merge) OPT_D("-chordaldens", chordal_density) OPT_I("-chordalform", chordal_form) OPT_I("-freeelim", free_elim) OPT_I("-deprows", deprows) OPT_D("-freefill", free_elim_fill) OPT_I("-chordalmin", chordal_minn)
        OPT_I("-hsdsig", hsd_sig) OPT_D("-hsdbeta", hsd_beta) OPT_I("-hsddir", hsd_dir) OPT_I("-hsdcorr", hsd_corr) OPT_I("-hsdrefine", hsd_refine) OPT_I("-hsdfree", hsd_free) OPT_I("-hsdstall", hsd_stall) OPT_D("-hsdpshrink", hsd_pshrink) OPT_I("-hsdreghint", hsd_reghint) OPT_D("-hsdx0", hsd_x0) OPT_I("-hsdqscale", hsd_qscale) OPT_I("-hsdbatch", hsd_batch) OPT_I("-hsdctrace", hsd_ctrace) OPT_I("-hsdctarget", hsd_ctarget) OPT_D("-hsdcbmin", hsd_cbmin) OPT_D("-hsdcbmax", hsd_cbmax) OPT_D("-hsdcacc", hsd_cacc) OPT_D("-hsdsoltol", hsd_soltol) OPT_D("-hsdcfrac", hsd_cfrac) OPT_D("-hsdfomega", hsd_fomega) OPT_D("-hsdpivtol", hsd_pivtol) OPT_D("-hsdcucap", hsd_cucap) OPT_D("-hsdcgain", hsd_cgain) OPT_I("-ddit", dd_iters) OPT_I("-ddmaxm", dd_maxm) OPT_D("-ddmaxn2", dd_maxn2)
        OPT_D("-ddbudget", dd_budget) OPT_D("-ddtime", dd_time) OPT_D("-ddfactor", dd_factor)
        OPT_D("-ddminwork", dd_minwork) OPT_D("-regill", reg_ill) OPT_D("-cblas", c_blas)
        OPT_I("-hsdhoc", hsd_hoc) OPT_D("-hsdnb", hsd_nb) OPT_I("-lanczos", lanczos_k) OPT_I("-ntfast", nt_fast) OPT_I("-hsdpat", hsd_pat) OPT_I("-parblocks", par_blocks) OPT_I("-knownfeas", known_feasible) OPT_D("-retryacc", retry_acc) OPT_D("-warm", warm_lam) OPT_D("-tracebound", tracebound) OPT_I("-dualize", dualize) OPT_I("-fom", fom) OPT_I("-fomhalpern", fom_halpern) OPT_I("-fommaxit", fom_maxit) OPT_D("-fomtol", fom_tol) OPT_D("-fomsigma", fom_sigma) OPT_D("-fomsigma0", fom_sigma0) OPT_I("-fomaadr", fom_aadr) OPT_D("-fomsingle", fom_single) OPT_I("-fomssnstall", fom_ssn_stall) OPT_I("-fomsigrule", fom_sigrule) OPT_I("-fomsigint", fom_sigint) OPT_I("-fomaa", fom_aa) OPT_I("-fombm", fom_bm) OPT_I("-fombmrank0", fom_bm_rank0) OPT_I("-fombmouter", fom_bm_outer) OPT_I("-fombminner", fom_bm_inner) OPT_D("-fombmrho", fom_bm_rho) OPT_D("-fombmgtol", fom_bm_gtol) OPT_D("-fombmnegtol", fom_bm_negtol) OPT_I("-fomssn", fom_ssn) OPT_I("-fomssnafter", fom_ssn_after) OPT_D("-fomssnres", fom_ssn_res) OPT_D("-fomssnrho", fom_ssn_rho) OPT_I("-fomssnprec", fom_ssn_prec) OPT_D("-fomssnsig0", fom_ssn_sig0) OPT_D("-fomssneta", fom_ssn_eta) OPT_I("-fomssnwarm", fom_ssn_warm) OPT_I("-fomssnouter", fom_ssn_outer) OPT_I("-fomssnnewton", fom_ssn_newton) OPT_I("-fomssncg", fom_ssn_cg) OPT_D("-fomaasafe", fom_aasafe) OPT_D("-fomsigmax", fom_sigmax) OPT_I("-hsdfirst", hsd_first) OPT_D("-hsdfirstc", hsd_first_c) OPT_D("-hsdfirstasm", hsd_first_asm) OPT_I("-stdslow", std_slow) OPT_I("-crossover", crossover) OPT_D("-frgain", fr_gain) OPT_I("-frdepth", fr_maxdepth) OPT_I("-frdual", fr_dual) OPT_D("-frfallback", fr_fallback) OPT_D("-symfallback", sym_fallback) OPT_I("-soc", soc) OPT_I("-socmin", soc_minn)
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
        /* ISSUES 725 (31 October 2026): -v the diagnosis (the default log until then), -vv and -vvv the levels -v and -vv were */
        else if (!strcmp(a, "-v")) par.verbose = 1;
        else if (!strcmp(a, "-vv")) par.verbose = 2;
        else if (!strcmp(a, "-vvv")) par.verbose = 3;
        else if (!strcmp(a, "-nofr")) do_fr = 0;
        else if (!strcmp(a, "-fr")) do_fr = 2;          /* facial reduction whatever its depth */
        else if (!strcmp(a, "-nofrretry")) fr_retry = 0;
        else if (!strcmp(a, "-dir")) {
            i = need(i, argc, a);
            if (!strcmp(argv[i], "nt")) par.direction = 1;
            else if (!strcmp(argv[i], "hkm")) par.direction = 0;
            else if (!strcmp(argv[i], "auto")) par.direction = -1;
            else { fprintf(stderr, "brisk: -dir must be auto, hkm or nt\n"); return 1; }
            dir_cli = par.direction;
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
        else if (!strcmp(a, "-conesolver")) { i = need(i, argc, a); par.conesolver = atoi(argv[i]) != 0; }   /* (SeDuMi format: sedumi.c; 26 October 2026: also a problem whose blocks are cones in disguise, solver.c socp_route) */
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
        else if (!strcmp(a, "-boundx")) { i = need(i, argc, a); boundxfile = argv[i]; }
        else if (!strcmp(a, "-boundy")) { i = need(i, argc, a); boundyfile = argv[i]; }
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
            else if (!strcmp(s, "pd") || !strcmp(s, "dp") || !strcmp(s, "PD") || !strcmp(s, "both")) par.bound_side = 3;
            else if (!strcmp(s, "0") || !strcmp(s, "off") || !strcmp(s, "none")) par.bound_side = 0;
            else { fprintf(stderr, "brisk: -bound must be p, d or pd\n"); return 1; }
        }
        OPT_I("-boundprior", bound_prior) OPT_I("-boundprec", bound_prec) OPT_D("-boundtol", bound_tol) OPT_D("-boundmargin", bound_margin) OPT_D("-boundtrack", bound_track_tol)
        OPT_D("-symtime", symtime) OPT_I("-symnodes", symnodes) OPT_I("-symbd", symbd) OPT_D("-symmin", symmin) OPT_I("-symsign", symsign) OPT_I("-symsigned", symsigned) OPT_D("-fomrace", fom_race) OPT_I("-fombound", fom_bound) OPT_I("-returnx", returnx) OPT_I("-symalg", symalg) OPT_I("-symalgmax", symalgmax) OPT_I("-blkscale", blkscale) OPT_I("-bm", bm) OPT_I("-mfipm", mfipm) OPT_I("-lralm", lralm) OPT_I("-lrrank", lr_rank) OPT_I("-lrrmax", lr_rmax) OPT_I("-lrouter", lr_outer) OPT_I("-lrinner", lr_inner) OPT_I("-lrescape", lr_escape) OPT_I("-lrprec", lr_prec) OPT_I("-lrnewton", lr_newton) OPT_D("-lrsigma", lr_sigma) OPT_D("-lrtol", lr_tol) OPT_D("-lrtrace", lr_trace) OPT_D("-mfrho", mf_rho) OPT_I("-mfrmax", mf_rmax) OPT_D("-mfdrop", mf_drop) OPT_I("-mfkmax", mf_kmax) OPT_I("-mfcgmax", mf_cgmax) OPT_D("-mfcgtol", mf_cgtol_min) OPT_D("-mfcgtolmax", mf_cgtol_max) OPT_I("-mfdiag", mf_diag) OPT_I("-mfwarm", mf_warm) OPT_I("-mfstall", mf_stall) OPT_D("-mfcgtime", mf_cgtime) OPT_I("-mfrecycle", mf_recycle) OPT_D("-mfhand", mf_hand) OPT_I("-mfhandcg", mf_handcg) OPT_I("-mfproj", mf_proj) OPT_D("-mfeta", mf_eta) OPT_I("-mftry", mf_try)
        else if (!strcmp(a, "-threads")) { i = need(i, argc, a); const int nt = (int)num(argv[i], a); if (nt >= 1) { omp_set_num_threads(nt); g_threads_user = 1; } }   /* 4.32: BRISK's OpenMP threads (default OMP_NUM_THREADS / all cores) */
        else if (!strcmp(a, "-route")) { i = need(i, argc, a); force_route = (int)num(argv[i], a); }
        else if (!strcmp(a, "-densemem")) { i = need(i, argc, a); par.dense_mem = num(argv[i], a) * 1048576.0; }
        else if (a[0] == '-') { fprintf(stderr, "brisk: unknown option %s\n", a); usage(); return 1; }
        else if (!fname) fname = a;
        else { fprintf(stderr, "brisk: more than one input file\n"); return 1; }
#undef OPT_D
#undef OPT_I
    }
    if (g_opts_only) return 0;
    {   /* 4 November 2026 (ISSUES 758): a tolerance looser than the accuracy level's moves the thresholds derived from it in
         * the levels' own proportions (re-solve above 10 tol, reduced accuracy and the embedding-first re-solve above 100 tol,
         * the slow-progress stop below 1000 tol). Until then -tol set the tolerance alone: a user's moment relaxation (SDP,
         * -nofr -tol 1e-4) ended the embedding at 5.8e-5, inside the tolerance asked for, and was re-solved with the
         * infeasible-start method because the re-solve threshold stayed at 1e-6: 71.4 s, against 25.7 s for -tol 1e-5. A
         * tighter tolerance keeps the level's thresholds (scaled down they re-solved more for no gain, params_accuracy). An
         * explicit -retryacc stays as given. */
        int tol_given = 0, retry_given = 0;
        for (int i = 1; i + 1 < argc; i++) { if (!strcmp(argv[i], "-tol")) tol_given = 1; if (!strcmp(argv[i], "-retryacc")) retry_given = 1; }
        if (tol_given && par.tol > 0) {
            if (!retry_given && 10.0 * par.tol > par.retry_acc) par.retry_acc = 10.0 * par.tol;
            if (100.0 * par.tol > par.red_acc) par.red_acc = 100.0 * par.tol;
            if (100.0 * par.tol > par.retry_hsd1) par.retry_hsd1 = 100.0 * par.tol;
            if (1000.0 * par.tol > par.slow_gate) par.slow_gate = 1000.0 * par.tol;
        }
    }
    /* 20 October 2026: -dir sets the direction of the self-dual embedding too (it reached only the infeasible-start
     * method, so -dir nt did nothing when the embedding ran); -hsddir still sets the embedding's alone */
    if (dir_cli >= 0 && par.hsd_dir < 0) par.hsd_dir = dir_cli;
    if (!fname) { usage(); return 1; }
    par.bound_xfile = boundxfile; par.bound_yfile = boundyfile;
    brisk_bound_prec_set(par.bound_prec, par.verbose);
    if ((cert_x || cert_y) && par.hp_kind > 0)      /* 5.2: in the master precision of -prec (hpsolve.c) */
        return brisk_hp_certify_given(fname, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result, cert_x, cert_y, par.hp_kind, par.hp_digits, par.verbose);
    g_certify_prior = par.bound_prior; g_certify_verbose = par.verbose;
    if (cert_x || cert_y) return certify_given(fname, cert_x, cert_y);
    { extern int g_ps_fr_correct; const char *ef = getenv("BRISK_FRFIX"); g_ps_fr_correct = ef ? atoi(ef) != 0 : 0; }   /* 5.17: opt-in (BRISK_FRFIX=1): the default path pays nothing and says plainly that the recovery failed */
    { extern int g_ps_need_x; g_ps_need_x = (xfile != NULL || getenv("BRISK_XOUT") != NULL || par.crossover != 0 || par.bound_side != 0 || (g_brisk_result != NULL && par.returnx > 0)) ? 1
                                         : (g_brisk_result != NULL && par.returnx < 0) ? 2 : 0; }
    g_rank_par = &par;
    g_natt = 0; g_run_no = 0; g_pre_logged = 0; g_infeas_unconf = 0; { extern int g_dup_inexact; g_dup_inexact = 0; } { extern int g_log_nmethod; g_log_nmethod = 0; } g_fr_dual_lost = 0; { extern int g_fr_depth_used; g_fr_depth_used = 0; } g_dec_sym_t = 0; g_dec_fr_t = 0; g_t_all_note = 0; g_amp_warn[0] = 0; g_fe_band = 0; g_first_name = NULL; g_mf_hand_t = -1; g_mf_hand_kept = 0;
    { extern int g_soc_used; g_soc_used = 0; }
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
    { extern double g_tab_t0; g_tab_t0 = par.t_start; }   /* (the SDP table's clock, for the lines of the steps after it) */
    if (par.timelimit > 0) (void)brisk_deadline_tighten(par.t_start + par.timelimit);   /* (restored by brisk_main) */
    if (par.lralm > 0 || par.bm > 0) (void)brisk_threads_busy(par.verbose);   /* (the process ends, or brisk_run restores the count) */
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
                              || !strcmp(a, "-maxit") || !strcmp(a, "-tol") || !strcmp(a, "-bound") || !strcmp(a, "-boundx") || !strcmp(a, "-boundy"))) { i++; continue; }
                if (i > 0 && !strcmp(a, "-certify")) continue;     /* (the certificate is built in high precision: hpcert.inc) */
                av[ac++] = argv[i];
            }
            static char pr[] = "-prec", pd[] = "dd";
            av[ac++] = pr; av[ac++] = pd;                 /* marks the inner run (g_hp_inner) */
            BriskResult *saved = g_brisk_result;
            const BriskData *sdata = g_brisk_data;
            g_brisk_result = &warm; g_hp_inner = 1;
            /* (26 October 2026: the high-precision solver reads the warm X as matrices. With the free-variable elimination
             * no longer excluding them, the inner solve of a SeDuMi problem (whose caller accepts cone vectors) returned
             * its arrow blocks as vectors and left g_xcone_res set, and the SeDuMi path then read the high-precision
             * solver's matrices as vectors: a cone of dimension 3 came out at half its value. The inner solve now returns
             * matrices and leaves the record as it was) */
            const int xacc = g_xcone_accept; int *xres = g_xcone_res; g_xcone_accept = 0; g_xcone_res = NULL;
            const double tw = wtime();
            int (*hook)(const char *, int) = NULL;
            const int fd = hp_mute_begin(&hook);
            brisk_main(ac, av);
            hp_mute_end(fd, hook);
            free(g_xcone_res); g_xcone_accept = xacc; g_xcone_res = xres;
            g_hp_inner = 0; g_brisk_result = saved; g_brisk_data = sdata;
            free(av);
            have_warm = warm.status >= 0 && warm.status != 1 && warm.status != 2 && warm.have_x && warm.X && warm.y;   /* (infeasible: the cold start has its own detection) */
            if (par.verbose >= 0) printf("high precision: double solve %s (max rel error %.1e) in %.2fs%s\n", warm.status >= 0 ? warm.status_str : "failed",
                                         fmax(fmax(fabs(warm.err[1]), fabs(warm.err[3])), fmax(fabs(warm.err[5]), fabs(warm.err[6]))), wtime() - tw, have_warm ? "" : "; no X returned: cold start");
        }
        const int rc = brisk_hp_run(fname, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result, have_warm ? &warm : NULL, &par, par.hp_kind, par.hp_digits, par.hp_tol, yfile, xfile, zfile);
        brisk_result_free(&warm);
        /* 5.10: the high-precision log ends with the way to a guaranteed bound (in the caller's syntax) */
        if (par.verbose > 0 && !g_hp_inner && !getenv("BRISK_NOSUMMARY") && (rc == 0 || rc == 10) && !(g_log_brief && rc == 0)) { g_log_user++; bound_hint(1); g_log_user--; }
        return rc;
    }
    if (par.bm > 0) return bm_run_x(fname, &par, yfile, xfile, zfile, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result, 0);   /* 5.14 */
    double t_bm_miss = 0;   /* (the time of a Burer-Monteiro attempt that missed) */
    {   /* 5.14: one block of order >= 500 whose constraints fix its diagonal (max-cut and its relatives): the
         * Burer-Monteiro method first, to the tolerance with a certified bound; the interior-point method if it misses */
        const BriskData *bd = g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL;
        const int bmc = bm_auto_candidate(fname, bd, &par, g_brisk_result != NULL || xfile || zfile);
        if (bmc) {
            /* 31 October 2026: the busy-machine rule before the attempt too (it held only for an explicit -bm): the method's
             * many small parallel loops wait at every barrier for a thread without a core; maxG11 (SDPLIB, SDP) beside
             * another 4-thread process on 4 cores: 93 s at 4 threads, 0.87 s at 2, 0.75 s on an idle machine */
            const int ntb = brisk_threads_busy(par.verbose);
            const double tb = wtime();
            const int rc = bm_run_x(fname, &par, yfile, xfile, zfile, bd, g_brisk_result, bmc);
#ifdef _OPENMP
            if (ntb > 0) omp_set_num_threads(ntb);   /* (the method that follows checks the machine itself) */
#endif
            if (rc == 0) return 0;
            if (rc != -1) {
                /* 24 October 2026: the attempt in the Summary and in the total (they left it out), and the time limit not
                 * shortened by it: the limit counts from the run's start (par.t_start, before the attempt), so its time
                 * was counted twice (after 100 s of the attempt under a 900 s limit, the interior-point method stopped
                 * at 800 s) */
                t_bm_miss = wtime() - tb;
                if (par.verbose >= 0) printf("Burer-Monteiro: the tolerance was not reached in %.1fs; the interior-point method with the factored Schur complement follows\n", t_bm_miss);
                att_put("Burer-Monteiro method", t_bm_miss, -1); g_att[g_att_res].before = 1; att_opt(NULL, NULL, "bm", "0");
                if (g_brisk_result) { brisk_result_free(g_brisk_result); memset(g_brisk_result, 0, sizeof(*g_brisk_result)); }
            }
        }
    }
    if (par.lralm > 0) return lralm_run(fname, &par, yfile, g_brisk_data && fname == g_brisk_data_name ? g_brisk_data : NULL, g_brisk_result);   /* 4.40: its own reading, measures and report */
    Run run, *u = &run;
    double textra = t_bm_miss;      /* time of a discarded solve */
    g_pipe_extra = 0;
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
            /* 23 October 2026: and a result at reduced accuracy when the time left of a time limit is less than the standard
             * solve's expected time (this attempt's budget): the standard solve would start from nothing and stop at the
             * limit (tru15, POEMA: 1.7e-8 after 722 s of 900 discarded, the embedding stopped at the limit at 1.9, TIME
             * LIMIT with 5.8e-1 returned) */
            const int short_left = par.timelimit > 0 && par.timelimit - (wtime() - par.t_start) < par.mf_try_budget;
            if (rc == 0 && u->measured && (u->status == ST_OPTIMAL || (par.timelimit > 0 && wtime() - par.t_start >= par.timelimit)
                                           || (!pr.mf_try_fits && u->status == ST_REDUCED) || (short_left && u->status == ST_REDUCED))) {
                par.mfipm = 1;                               /* kept: the run is a matrix-free run from here on */
                if (short_left && u->status == ST_REDUCED && par.verbose >= 0)
                    printf("matrix-free interior-point method first: %.1e after %.1f s; the factored method is not expected to finish in the %.0f s left of the time limit: this result is kept\n",
                           u->acc, wtime() - tr0, par.timelimit - (wtime() - par.t_start));
                if (g_mf_hand_t >= 0) {   /* two solves: the matrix-free one, then the factored one from its iterate (whose result this is) */
                    att_put("matrix-free interior-point method, not optimal", g_mf_hand_t, -1); g_att[g_att_res].before = 1; g_mf_hand_kept = 1;
                } else brisk_note("matrix-free interior-point method, tried first automatically (-mftry): its result is returned (%.1f s)", wtime() - tr0);
            } else {
                if (par.verbose >= 0) printf("matrix-free interior-point method first: %.1e after %.1f s, continuing with the interior-point method with the factored Schur complement\n", rc == 0 ? u->acc : NAN, wtime() - tr0);
                if (rc == 0) { att_log(g_mf_hand_t >= 0 ? "matrix-free interior-point method, then the interior-point method with the factored Schur complement from its iterate" : "matrix-free interior-point method", wtime() - tr0, u);
                               g_att[g_att_res].before = 1; att_opt(NULL, NULL, "mftry", "0"); textra += wtime() - tr0; }
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
            const int bnd = par.fom_bound_maxit > 0;   /* 25 October 2026: the engine with the bounds on X's entries, by an iteration cap */
            pr.fom = 1; pr.fom_race = 0;
            if (pr.free_elim < 0) pr.free_elim = 0;
            if (pr.dualize < 0) pr.dualize = 0;
            if (pr.tracebound < 0) pr.tracebound = 0;
            if (pr.chordal < 0) pr.chordal = 0;
            const double tr0 = wtime();
            if (bnd) pr.fom_maxit = (int)(par.fom_maxit > 0 && par.fom_maxit < par.fom_bound_maxit ? par.fom_maxit : par.fom_bound_maxit);   /* (-fommaxit lowers it: the test of the hand-over) */
            else {
                /* with a time limit, at most the same share of what is left of it */
                if (par.timelimit > 0) par.fom_race_budget = fmin(par.fom_race_budget, fmax(0.0, par.fom_race * (par.timelimit - (tr0 - par.t_start))));
                const double lim = (tr0 - par.t_start) + par.fom_race_budget;
                pr.timelimit = lim;
            }
            rc = run_pipeline(fname, &pr, do_fr, force_route, u);
            if (rc == 0 && u->status == ST_OPTIMAL && u->measured) {
                par.fom = 1;                                 /* kept: the run is a first-order run from here on */
                if (par.verbose >= 0) printf("first-order engine first: reached the tolerance in %.1f s\n", wtime() - tr0);
            } else {
                if (par.verbose >= 0) printf("first-order engine first: %.1e after %.1f s, continuing with the interior-point method\n", rc == 0 ? u->acc : NAN, wtime() - tr0);
                if (rc == 0) { att_log(bnd ? "first-order method with entrywise bounds on X (within its iteration budget)" : "first-order method (within its time budget)", wtime() - tr0, u); g_att[g_att_res].before = 1; att_opt(NULL, NULL, bnd ? "fombound" : "fomrace", "0"); textra += wtime() - tr0; }
                run_free(u);
                par.fom_race = 0; par.fom_bound_maxit = -1;
                rc = run_pipeline(fname, &par, do_fr, force_route, u);
                if (rc == 3) { run_free(u); do_fr = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
                if (rc == 4) { run_free(u); par.fom = 1; if (par.chordal < 0) par.chordal = 0; rc = run_pipeline(fname, &par, do_fr, force_route, u); }
            }
        }
        if (rc != 0) return 2;
        /* 5.14: a solve in scaled coordinates whose solution, mapped back, misses the tolerance on the problem as
         * read: the problem is solved again as given (blkscale.c); 5.20: the better of the two answers is kept
         * (both are measured on the problem as read; the scaled one is often the better, static21compl, sat_roa) */
        if ((u->aff || u->maff) && u->measured && u->status != ST_OPTIMAL && !brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit)) {
            const double ta = u->R.t_total + u->tpre + u->pr.t + u->tread;   /* (the kept run's: the discarded attempts are in textra and g_pipe_extra) */
            if (par.verbose >= 0) printf("%s: %.1e on the problem as read; solving the problem as given\n", u->maff ? "ellipsoid coordinates" : "scaled blocks", u->acc);
            Run second;
            const double tb0 = wtime();
            g_blkscale_off = 1; g_momaff_off = 1;
            Params pb = par;
            const int rc2 = run_resolve(fname, &pb, do_fr, force_route, &second);
            g_blkscale_off = 0; g_momaff_off = 0;
            if (rc2 == 5) run_free(&second);
            if (rc2 == 0) {
                if (run_better(&second, u)) {
                    att_log(u->maff ? "the relaxation in the coordinates of the unit ball" : "the problem with scaled semidefinite blocks", ta, u);
                    if (!u->maff) att_opt(NULL, NULL, "blkscale", "0");
                    second.R.iters += u->R.iters;
                    run_free(u);
                    run = second;
                    textra += ta;
                } else {
                    if (par.verbose >= 0) printf("  the problem as given ended at %.1e; the solve in scaled coordinates is kept\n", second.acc);
                    att_log("the problem without the scaling", wtime() - tb0, &second);
                    u->R.iters += second.R.iters;
                    run_free(&second);
                    textra += wtime() - tb0;
                }
            }
        }
        /* 4.37 (B1): the first attempt, and the in-solver method retry as a re-solve */
        g_first_t = u->R.t_total - (u->R.nretry ? u->R.t_retry : 0) + u->tpre + u->pr.t + u->tread;
        if (g_mf_hand_kept) g_first_t -= g_mf_hand_t;      /* (the matrix-free solve before it is listed by itself) */
        g_first_acc = u->R.nretry ? u->R.acc_first : u->acc;
        g_first_internal = u->R.nretry;
        /* (23 October 2026) the first solve named by its chordal form when it had one, and its own re-solve with the cones as
         * semidefinite blocks listed after it (the first solve's time and accuracy are then those of its first run) */
        if (u->P.ps && u->P.ps->chordal) g_first_name = u->P.ps->mom ? "chordal decomposition, moment form" : "chordal decomposition, overlap equalities";
        if (u->soc2_done) {
            g_first_t = u->soc2_t1 + u->tread; g_first_acc = u->soc2_acc1;
            att_put("second-order cones solved as semidefinite blocks", u->soc2_t2, u->soc2_acc2);
            g_att[g_att_res].kept = u->soc2_kept; att_opt("soc", "0", NULL, NULL);
        }
        if (u->R.nretry && u->R.retry_hsd == 2) {
            /* 23 October 2026: the embedding again with its primal start at the scale the first one reached (solver.c); the
             * advice names that start (-hsdx0) when its result is returned, -hsd (no second solve) when the first one's is */
            static char x0v[32];
            snprintf(x0v, sizeof x0v, "%.2g", u->R.retry_x0);
            att_log("interior-point method, self-dual embedding with a rescaled start", u->R.t_retry, NULL);
            g_att[g_att_res].kept = u->R.retry_kept;
            if (u->R.retry_kept) g_att[g_att_res].acc = u->acc;
            att_opt("hsdx0", x0v, "hsd", NULL);
            g_first_name = "interior-point method, self-dual embedding";
        } else if (u->R.nretry) {
            att_log(u->R.retry_hsd ? "interior-point method, self-dual embedding" : "interior-point method, infeasible start", u->R.t_retry, NULL);
            g_att[g_att_res].kept = u->R.retry_kept;
            if (u->R.retry_kept) g_att[g_att_res].acc = u->acc;
            att_opt(u->R.retry_hsd ? "hsd" : "nohsd", NULL, u->R.retry_hsd ? "nohsd" : "hsd", NULL);
            g_first_name = u->R.retry_hsd ? "interior-point method, infeasible start" : "interior-point method, self-dual embedding";
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
    if (!par.fom && !par.mfipm && u->P.ps && u->P.ps->mom && u->measured && u->status != ST_OPTIMAL && acc_apart_fr(u, &par, do_fr) > par.red_acc
        && par.chordal_sup < 0 && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.3 * par.timelimit))) {
        Params p2 = par;
        p2.chordal_sup = 4; p2.chordal_merge = 0.6;
        p2.hsd = 1;
        p2.hsd_dir = 1;
        p2.chordal_need = 1;
        if (par.verbose >= 0)
            printf("chordal: the moment form ended at %.1e, re-solving with support cliques and NT\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        const int rc2 = run_resolve(fname, &p2, do_fr, force_route, &second);
        if (rc2 == 5) run_free(&second);
        if (rc2 == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("chordal decomposition, moment form with support cliques", tsecond, &second); att_opt("chordalsup", "4", NULL, NULL);
            if (run_better(&second, u)) {
                g_att[g_att_res].kept = 1;
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
    if (!par.fom && !par.mfipm && u->P.ps && u->P.ps->mom && u->measured && u->status != ST_OPTIMAL && acc_apart_fr(u, &par, do_fr) > par.red_acc && !near_floor
        && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        p2.chordal_form = 0;
        p2.hsd = 1;
        p2.hsd_dir = hsddir_user;
        p2.chordal_need = 1;
        if (par.verbose >= 0)
            printf("chordal: the moment form ended at %.1e, re-solving with overlap equalities\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        const int rc3 = run_resolve(fname, &p2, do_fr, force_route, &second);
        if (rc3 == 5) run_free(&second);
        if (rc3 == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("chordal decomposition, overlap equalities", tsecond, &second); att_opt("chordalform", "0", NULL, NULL);
            if (run_better(&second, u)) {
                g_att[g_att_res].kept = 1;
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
    if (!par.fom && !par.mfipm && u->P.ps && u->P.ps->chordal && u->measured && u->status != ST_OPTIMAL && acc_apart_fr(u, &par, do_fr) > par.red_acc && !near_floor && unconv_fits
        && par.chordal < 0 && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.3 * par.timelimit))) {
        Params p2 = par;
        p2.chordal = 0;
        p2.hsd = hsd_user;
        p2.hsd_dir = hsddir_user;
        if (par.verbose >= 0)
            printf("chordal: the converted problem ended at %.1e, solving the unconverted problem\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_resolve(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the problem without chordal decomposition", tsecond, &second); att_opt("chordal", "0", "chordal", "1");
            if (run_better(&second, u)) {
                g_att[g_att_res].kept = 1;
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
        && brisk_nmax(brisk_nmax(u->R.relgap, u->R.pinf), brisk_nmax(u->R.dinf, u->R.relcomp)) <= 1e-8   /* the reduced solve passed 1e-8: */
                                                                                       /* the loss is the recovery's (swissroll: 8e-7 either way, 64 -> 129 s) */
        && u->status != ST_PINFEAS && u->status != ST_DINFEAS && !getenv("BRISK_NOHIFALLBACK")
        && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        params_accuracy(&p2, 1);
        if (par.verbose >= 0)
            printf("postsolve: the high-accuracy result is %.1e on the original data after facial reduction, solving at the default accuracy as well\n", u->acc);
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_resolve(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("the default accuracy", tsecond, &second);
            if (run_better(&second, u)) {
                g_att[g_att_res].kept = 1;
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
        if (run_resolve(fname, &p2, do_fr, force_route, &second) == 0) {
            att_log("the problem without the elimination of free variables (infeasibility check)", second.R.t_total + second.tpre + second.pr.t + second.tread, &second); att_opt("freeelim", "0", "freeelim", "1");
            g_att[g_att_res].kept = 1;
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
    /* 31 October 2026 (ISSUES 724): just above 1e-6 without the trace bound binding, the eliminated result is returned and the
     * advice names -freeelim 0. Both forms meet the same limit there, the problem's (an unattained optimum: tr X 7e6, tau to
     * 5e-5 in both); the re-solve fired on two of the 150 development files where the elimination applies and gained a
     * factor of about 2 for twice the time (the owner's region of attraction, SDP: 3.1e-6 -> 1.6e-6, 21.7 s against
     * 9.5 s for the problem as read alone; roa_vdp_inner_d14_K: 1.2e-6 -> 7.2e-7, 34 s against 15 s; MOSEK 1.4e-4).
     * roa_acrobot_d6_K, the case of 4.30, no longer ends in the band. BRISK_FEBAND=1: the re-solve as before.
     * The same in the band when the trace bound binds: its shift of the value is then of the order of the error, which is
     * measured on the file's data and so already holds it (the owner's file through brisk_sedumi: shift 8e-7 relative
     * against 3.3e-6, counted as binding, and solved again for 3.3e-6 -> 2.6e-6; from the command line 2.9e-7, not
     * counted). The binding bound's re-solve is for results far above the band (roa_univar_d16_I: 8.4e-4 -> 2.9e-7) */
    const int tb_binding = u->measured && u->acc > par.tol && tb_binds(u, u->acc, par.tol);
    const int in_band = u->measured && u->acc > 1e-6 && u->acc <= 1e-5;
    const int fe_band = u->fe && !u->fe->cheap && u->measured && par.free_elim < 0 && par.dualize < 0 && par.tracebound < 0
        && !u->dual && in_band && u->status != ST_PINFEAS && u->status != ST_DINFEAS;
    const int fe_band_resolve = fe_band && ENV_ON("BRISK_FEBAND");
    if (fe_band && !fe_band_resolve) g_fe_band = u->fe->ns > 0 ? u->fe->ns : 1;   /* (the count, for the advice) */
    if (u->fe && !u->fe->cheap && u->measured && par.free_elim < 0 && par.dualize < 0 && par.tracebound < 0
        && (fe_band_resolve || (tb_binding && (!in_band || ENV_ON("BRISK_FEBAND"))))
        && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        Params p2 = par;
        p2.free_elim = 0; p2.tracebound = 0; p2.dualize = 0;
        if (par.verbose >= 0) printf("the eliminated problem ended at %.1e on the original data%s, re-solving the problem as read\n", u->acc,
                                     tb_binding ? " with the trace bound binding" : "");
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_resolve(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log("without eliminating free variables", tsecond, &second); att_opt("freeelim", "0", "freeelim", "1");
            if (run_better(&second, u)) {
                g_att[g_att_res].kept = 1;
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
    /* 5.16 (the owner's benchmark of 15 October 2026, its issue 2, ISSUES 94): a solve after an automorphism
     * reduction that ends worse than reduced accuracy is repeated without the reduction, within -symfallback
     * times the solve's time, and the better answer kept (two CBLIB clique problems: the reduced problem,
     * m 1,770 -> 900, one block 60 -> 2 x 30, ends at 3e-3 and 2e-6 after 50-60 iterations; without the
     * reduction OPTIMAL at 1e-9 in 12-13; the reduction is exact, so the reduced problem's conditioning is
     * the cause, open). The per-iteration cost model is not consulted: it reads 7.6x on these files (m 1,770
     * against 900, cubed) while the unreduced solve takes half the reduced one's time (the LP part and the
     * sparsity are not in the model); the time cap is the rule.
     * 23 October 2026: judged by the error apart from the duals the facial reduction could not rebuild (fr_duals_lost):
     * the re-solve keeps the facial reduction and cannot change them. */
    if (u->sym && par.symfile && !strcmp(par.symfile, "auto") && par.sym_fallback > 0 && u->measured && acc_apart_fr(u, &par, do_fr) > par.red_acc
        && u->status != ST_PINFEAS && u->status != ST_DINFEAS && u->status != ST_TIME
        && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))) {
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        {
            Params p2 = par;
            p2.symfile = NULL;
            const double lim = (wtime() - par.t_start) + fmax(0.5, par.sym_fallback * fmax(tfirst, 0.01));
            if (!(p2.timelimit > 0) || lim < p2.timelimit) p2.timelimit = lim;
            if (par.verbose >= 0) printf("symmetry reduction: the solve ended at %.1e on the original data, solving without the reduction (at most %gx the solve's time; -symfallback)\n", u->acc, par.sym_fallback);
            Run second;
            if (run_resolve(fname, &p2, do_fr, force_route, &second) == 0) {
                const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
                att_log("the problem without the symmetry reduction", tsecond, &second); att_opt("nosym", NULL, "symfallback", "0");
                if (run_better(&second, u)) {
                    g_att[g_att_res].kept = 1;
                    if (par.verbose >= 0) printf("  the new solve is better, keeping it\n");
                    brisk_note("symmetry reduction: the reduced solve ended at %.1e; the problem was solved again without the reduction and that solve is returned", u->acc);
                    second.R.iters += u->R.iters;
                    run_free(u);
                    run = second;
                    textra += tfirst;
                } else {
                    if (par.verbose >= 0) printf("  the new solve is not better%s, keeping the previous one\n", second.status == ST_TIME ? " (it did not finish within its time)" : "");
                    u->R.iters += second.R.iters;
                    run_free(&second);
                    textra += tsecond;
                }
            }
        }
    }
    /* 4.20: a result that misses 1e-6 on the original data and came from one method only
     * (the in-solver score can pass while the measured error does not: hinf11's endgame
     * point has lambda_min(Z) = -1.6e-5) is re-solved with the other method */
    /* 23 October 2026: not for an embedding result at reduced accuracy by its own measure as well (within ten times
     * red_acc on the problem as read and above red_acc in the solver's score, the cone violations included): the
     * embedding reached its floor there, and the in-solver retry no longer runs in that band (solver.c, rgate); a
     * result the solver scored below red_acc and the problem as read does not (hinf11's case) is still re-solved.
     * BRISK_RETRYBAND=0: as before */
    const double s_in = brisk_nmax(brisk_nmax(brisk_nmax(u->R.relgap, u->R.relcomp), brisk_nmax(u->R.pinf, u->R.dinf)), brisk_nmax(u->R.err[2], u->R.err[4]));
    const int hsd_band = u->R.methods == 2 && u->acc <= 10.0 * par.red_acc && s_in > par.red_acc
                         && !(getenv("BRISK_RETRYBAND") && atoi(getenv("BRISK_RETRYBAND")) == 0);
    /* (27 October 2026, the night: the infeasible-start method takes no cone as a cone; a cone of order 45,001 as a dense
     * block ran out of memory and the process was killed, losing the embedding's result) */
    const int hsd_cones_too_big = u->R.methods == 2 && g_soc_used && u->O && sdp_form_bytes(u->O) > 0.6 * brisk_mem_limit();
    if (hsd_cones_too_big && hsd_user < 0 && u->measured && acc_apart_fr(u, &par, do_fr) > par.red_acc && par.verbose >= 0)
        printf("the self-dual embedding ended at %.1e on the original data; the infeasible-start method would need its cones as semidefinite blocks (%.0f GB): not re-solved\n", u->acc, sdp_form_bytes(u->O) / 1e9);
    if (hsd_user < 0 && u->measured && acc_apart_fr(u, &par, do_fr) > par.red_acc && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && (u->R.methods == 1 || u->R.methods == 2) && !hsd_band && !(u->P.ps && u->P.ps->chordal)
        && !(u->R.free_hsd && u->acc < 1e-4 && !getenv("BRISK_FREERESOLVE"))   /* 4.24: split free pairs: the standard method stalls */
        && !(u->fr_removed > 0 && u->pr.err[4] >= 0.5 * u->acc)      /* the removed duals: see below */
        && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit))
        && !hsd_cones_too_big) {
        Params p2 = par;
        p2.hsd = u->R.methods == 1 ? 1 : 0;
        p2.no_retry = 1;
        if (par.verbose >= 0)
            printf("the %s ended at %.1e on the original data, re-solving with the %s\n",
                   p2.hsd ? "infeasible-start method" : "self-dual embedding", u->acc, p2.hsd ? "self-dual embedding" : "infeasible-start method");
        Run second;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        if (run_resolve(fname, &p2, do_fr, force_route, &second) == 0) {
            const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log(p2.hsd ? "interior-point method, self-dual embedding" : "interior-point method, infeasible start", tsecond, &second);
            att_opt(p2.hsd ? "hsd" : "nohsd", NULL, p2.hsd ? "nohsd" : "hsd", NULL);
            g_first_name = p2.hsd ? "interior-point method, infeasible start" : "interior-point method, self-dual embedding";
            second.R.methods |= u->R.methods;
            if (run_better(&second, u)) {
                g_att[g_att_res].kept = 1;
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
    /* 5.16 (the owner's decision of 12 October 2026): when the duals of the removed constraints
     * cannot be rebuilt and the reduced solve is good in itself, it is returned at once, without
     * the re-solves below. X is that of the problem as read; the removed rows have b_r = 0, so b'y
     * is the reduced solve's, and the optimal value lies in [b'y, <C,X>] (weak duality on the
     * reduced problem, whose face holds every feasible X); y is not dual feasible for the removed
     * constraints (a note says so). The status follows the primal errors on the problem as read,
     * the gap and complementarity, and the reduced solve's own dual errors. Until 5.15 the reduced
     * problem was solved again with the embedding and then the problem without facial reduction
     * (a user's mbb14compl_msos_tspmin1__d3: 3.6 s and a value right to 1e-8, then 66 s and a value
     * 1.3e-6 off returned). BRISK_FRKEEP=0: the re-solves as before.
     * 5.16 then solved the problem again without the reduction within -frfallback (4) times the solve's time;
     * 5.17 (the owner, 16 October 2026): no re-solve by default (-frfallback 0): the status is not OPTIMAL, the
     * log and the note say that the recovery failed, what is correct, and how to solve again without the
     * reduction in the caller's syntax; -frfallback f > 0 gives the re-solve of 5.16. The correction of the
     * dual through complementarity with the primal (postsolve.c) is opt-in, BRISK_FRFIX=1. */
    int fr_kept = 0;
    {
        static int frkeep = -1;
        if (frkeep < 0) { const char *e = getenv("BRISK_FRKEEP"); frkeep = e ? atoi(e) != 0 : 1; }
        double acc_k;
        const int lost = fr_duals_lost(u, &par, do_fr, &acc_k);
        if (lost) {
            if (acc_k <= par.red_acc) {
                /* 5.16 (the owner's benchmark of 15 October 2026, ISSUES 93): never OPTIMAL with a dual that is
                 * infeasible on the problem as read (taha1b and the chained-Wood SOS relaxations ended OPTIMAL at a
                 * dual error of 1.0 while MOSEK returns a dual at 2e-10); the reduced solve's accuracy is kept for
                 * the record, the run's accuracy is that of the returned point. The reduced solve's dual need not
                 * extend to the removed constraints even when the dual optimum of the problem as read is attained:
                 * the reduced dual optimal face is larger than the projection of the original one (the coupling
                 * of a removed index with the kept ones must lie in the range of the reduced slack, which the
                 * reduced solve knows nothing about), so the re-solve below is the way to that dual. */
                /* 5.17 (the owner, 16 October 2026): no re-solve by default; the user is told that the recovery failed,
                 * what is correct, and how to solve again without the reduction, in the caller's syntax (the command
                 * line, Python, CVXPY, Julia, JuMP, MATLAB: the machinery of the advice at the end of the log) */
                const int resolve_follows = fr_retry && par.fr_fallback > 0;
                if (par.verbose >= 0) {
                    /* 20 October 2026 (the owner): said once here; the summary has one line, the advice the option */
                    printf("postsolve: facial reduction: the duals of the %d removed constraint(s) could not be rebuilt; X and the optimal value\n"
                           "  are correct to %.1e, y is not dual feasible (dual error %.1e on the problem as read)\n", u->fr_removed, acc_k, u->pr.err[4]);
                }
                if (!resolve_follows) g_fr_dual_lost = u->fr_removed;
                u->status = ST_REDUCED;
                u->acc = fmax(acc_k, fmin(u->pr.err[4], 1.0));
                fr_kept = 1;
                if (par.fr_dual > 0 && !resolve_follows) {   /* -frdual: the dual on the problem as read, the value held */
                    double tph = 0;
                    if (fr_dual_phase(u, &par, force_route, &tph)) g_fr_dual_lost = 0;
                    textra += tph;
                }
            }
        }
        /* 5.17 (ISSUES 93): the removed duals were rebuilt through complementarity with the primal, but not
         * to the tolerance (pop_wood: 1e-7 where the status needs 1e-8): the point is returned as it is,
         * with the status its errors give, and the re-solve without the reduction is tried within the same
         * budget for the last decade */
        if (!lost && frkeep && do_fr && u->fr_removed > 0 && u->measured && u->status != ST_OPTIMAL
            && u->status != ST_PINFEAS && u->status != ST_DINFEAS && u->pr.fr_corrected
            && u->pr.err[4] > par.tol && u->pr.err[4] >= 0.5 * u->acc) {
            if (par.verbose >= 0)
                printf("postsolve: the duals of the %d removed constraint(s) are rebuilt through complementarity with the primal to a dual error of %.1e on the problem as read (the certificate recovery failed)\n", u->fr_removed, u->pr.err[4]);
            brisk_note("facial reduction: the duals of the %d removed constraint(s) could not be rebuilt from the reduction certificates; y was corrected through "
                       "complementarity with the primal to a dual error of %.1e on the problem as read", u->fr_removed, u->pr.err[4]);
            fr_kept = 1;
        }
    }
    for (int stage = 0; stage < 2; stage++) {
        const int need = do_fr && fr_retry && u->fr_removed > 0 && u->measured && u->status != ST_OPTIMAL
            && u->status != ST_PINFEAS && u->status != ST_DINFEAS
            && u->pr.err[4] > (fr_kept ? fmax(par.tol, 1e-3 * u->acc) : fmax(100.0 * fmax(par.tol, 1e-8), 1e-3 * u->acc)) && u->pr.err[4] >= 0.5 * u->acc
            && (!fr_kept || (stage == 1 && par.fr_fallback > 0))
            && (!brisk_stop_flag && !maxit_spent(u, &par) && (par.timelimit <= 0 || wtime() - par.t_start < 0.5 * par.timelimit));
        if (!need) { if (fr_kept && stage == 0) continue; break; }
        const int used_hsd = par.hsd == 1 || u->R.retried || (u->R.methods & 2);
        if (stage == 0 && used_hsd) continue;
        Params p2 = par;
        p2.dual = 0;               /* the fallbacks use the primal-dual method */
        /* when the reduced solve is good apart from the removed duals, the problem has a
         * solution and infeasibility exits of the new solve are false (rose15 without
         * facial reduction: free variables make it claim dual infeasibility) */
        p2.known_feasible = brisk_nmax(brisk_nmax(u->pr.err[1], u->pr.err[2]), brisk_nmax(fabs(u->pr.err[5]), fabs(u->pr.err[6]))) <= 1e-6;
        int fr2 = 1;
        if (stage == 0) p2.hsd = 1; else fr2 = 0;
        const double tfirst = u->R.t_total + u->tpre + u->pr.t + u->tread;
        const int nrem = u->fr_removed;
        /* the reduced solve is good in itself: the re-solve without the reduction is worth a bounded
         * multiple of its time (the owner's benchmark: 1.7-3.9x on taha1b and pop_wood; a user's
         * mbb14compl 18x, where the reduced solve is returned instead) */
        if (fr_kept) {
            /* (the per-iteration cost model of the fr_gain rule, u->fr_cost, does not predict the re-solve's
             * time: pop_wood_20 reads 16x and the re-solve takes 6x, a user's mbb14compl reads small and the
             * re-solve does not finish within 4x; the time cap is the bound) */
            const double lim = (wtime() - par.t_start) + fmax(0.5, par.fr_fallback * fmax(tfirst, 0.01));
            if (!(p2.timelimit > 0) || lim < p2.timelimit) p2.timelimit = lim;
        }
        if (par.verbose >= 0) {
            if (fr_kept) printf("postsolve: re-solving without facial reduction for the dual (at most %gx the solve's time; -frfallback)\n", par.fr_fallback);
            else printf("postsolve: dual of the removed constraints not recovered to tolerance (err4 %.1e), %s\n",
                        u->pr.err[4], stage == 0 ? "re-solving the reduced problem with the self-dual embedding"
                                                 : "re-solving without facial reduction");
        }
        Run second;
        if (run_resolve(fname, &p2, fr2, force_route, &second) != 0) break;
        const double tsecond = second.R.t_total + second.tpre + second.pr.t + second.tread;
            att_log(stage == 0 ? "the facially reduced problem with the self-dual embedding" : "the problem without facial reduction", tsecond, &second);
            if (stage == 0) att_opt("hsd", NULL, NULL, NULL); else att_opt("nofr", NULL, "nofrretry", NULL);
        /* an infeasibility claim cannot replace a solved reduced problem: the reduced
         * primal solution is feasible for the original problem and its dual bounds the
         * value (rose15 without facial reduction reports a false dual infeasibility) */
        const int second_inf = second.status == ST_PINFEAS || second.status == ST_DINFEAS;
        if (!second_inf && run_better(&second, u)) {
            g_att[g_att_res].kept = 1;
            if (par.verbose >= 0) printf("  the new solve is better, keeping it\n");
            second.R.iters += u->R.iters;
            run_free(u);
            run = second;
            textra += tfirst;
            if (fr_kept) { fr_kept = 0; brisk_note("facial reduction: the reduced solve's dual did not extend to the %d removed constraint(s); the problem was solved again without the reduction and that solve is returned", nrem); }
        } else {
            if (par.verbose >= 0) {
                if (fr_kept && second.status == ST_TIME) printf("  the re-solve did not finish within %gx the solve's time; the reduced solve is returned\n", par.fr_fallback);
                else printf("  the new solve is not better, keeping the previous one\n");
            }
            u->R.iters += second.R.iters;
            run_free(&second);
            textra += tsecond;
        }
    }
    /* 4.20: SDP crossover on the problem as read (Newton on the rank-revealed KKT system) */
    if (par.crossover != 0 && !fr_kept && u->measured && u->Xo && u->status != ST_PINFEAS && u->status != ST_DINFEAS
        && u->acc > 1e-15 && u->acc < 1e-3) {
        const double tsolve = u->R.t_total + u->tpre + u->pr.t + textra;
        const double budget = par.crossover > 0 ? fmax(1.0, 3.0 * tsolve) : fmax(0.05, 0.1 * tsolve);
        double eb[7], ea[7], pd[2];
        const double tx0 = wtime();
        if (sdp_crossover(u->Ofin ? u->Ofin : u->O, u->Xo, u->yo, par.verbose, budget, par.crossover > 0 ? 30000.0 : 8000.0, eb, ea, pd)) {
            brisk_note("crossover (Newton on the rank-revealed system): max error %.1e -> %.1e", brisk_acc_err(eb), brisk_acc_err(ea));
            for (int i = 1; i <= 6; i++) u->pr.err[i] = ea[i];
            u->pr.pobj = pd[0]; u->pr.dobj = pd[1];
            u->acc = brisk_acc_err(ea);
            if (u->acc <= par.tol) u->status = ST_OPTIMAL;
            else if (u->acc <= par.red_acc) u->status = ST_REDUCED;
        }
        textra += wtime() - tx0;
    }
    /* Bound mode: each side asked for has its certificate checked rigorously on the problem as
     * read (4.39 -certify, now always): only a certified certificate is a bound. The returned pair
     * stays the solve's own point. Up to 1.5.1 the certificate replaced its side of the pair, which
     * was then measured again; it can come from an earlier iterate than the returned point, or be
     * the returned point moved to a margin, and the pair was then unbalanced (ni35 order 2: X of
     * iteration 48 with the final y, gap 8.6e-7 for a pair at 9.7e-10; arch0 -bound p: OPTIMAL
     * 9.7e-9 became REDUCED ACCURACY 7.2e-7). */
    int certified[2] = { 0, 0 };
    double rigorous[2] = { NAN, NAN }, t_cert[2] = { 0, 0 };
    char cert_why[2][320] = { "", "" };
    int prior[2] = { 0, 0 };                     /* certified by a-priori bounds: from the certificate (1), the returned point (2) or the iterate before the polishes (3) */
    for (int sd = 1; sd <= 2; sd++) {
        BoundCert *bc = &u->cert[sd - 1];
        bc->certified = 0; bc->rigorous = NAN;
        if (!(par.bound_side & sd) || !bc->have || u->status == ST_PINFEAS || u->status == ST_DINFEAS) continue;
        const double tc0 = wtime();
        { extern int g_dup_inexact;   /* (audit A20: entries given in parts whose exact sum is not a double) */
          if (g_dup_inexact) {
              certified[sd - 1] = 0; rigorous[sd - 1] = NAN; t_cert[sd - 1] = 0;
              snprintf(cert_why[sd - 1], sizeof(cert_why[0]), "%d entr%s of the data given in parts whose exact sum is not a double: the problem as given cannot be represented exactly, so no bound is certified for it",
                       g_dup_inexact, g_dup_inexact == 1 ? "y" : "ies");
              continue;
          } }
        /* (prior first: no certificate was built; the a-priori route below certifies the returned point) */
        certified[sd - 1] = bc->prior_first ? 0 :
            brisk_certify_orig(u->Ofin ? u->Ofin : u->O, sd, bc->X, bc->y, g_brisk_data != NULL, &rigorous[sd - 1], cert_why[sd - 1], sizeof(cert_why[0]));
        t_cert[sd - 1] = wtime() - tc0;
        if (!certified[sd - 1]) rigorous[sd - 1] = NAN;      /* (computed before the check failed: not a bound) */
        /* the candidate before its margin was restored, when the restoration moved it: blending
         * with an anchor can give away most of the objective (ni35 order 2 with the objective cut:
         * the anchor blend 1.0 certified 1941.608 for a point at 1940.571). The a-priori route
         * certifies points without a margin, so it is tried on that candidate too, and the better
         * rigorous bound is kept. */
        const int has0 = sd == 1 ? bc->X0 != NULL : bc->y0 != NULL;
        const int loss0 = certified[sd - 1] && has0 &&
            (sd == 1 ? rigorous[sd - 1] - bc->value0 : bc->value0 - rigorous[sd - 1]) > 1e-7 * (1 + fabs(rigorous[sd - 1]));
        if ((!certified[sd - 1] || loss0) && par.bound_prior) {
            /* no margin (or a margin that cost objective): a bound from a-priori bounds of the other
             * side (boundcert.c), for the certificate (before its margin was restored, when it was
             * moved) or the returned point, whichever gives the better one */
            double B; int which = 0; char w2[160] = "";
            const double tp0 = wtime();
            const double margin_bound = rigorous[sd - 1];
            /* (-boundprior 2: the run certified the returned point by a-priori bounds before any certificate was built;
             * its bound is used as it is, not computed again) */
            const int okp = bc->prior_first ? (B = bc->value, which = strncmp(bc->src, "the method's iterate", 20) ? 2 : 3, 1)
                : brisk_prior_bound(u->Ofin ? u->Ofin : u->O, sd, sd == 1 ? (bc->X0 ? bc->X0 : bc->X) : NULL, sd == 1 ? u->Xo : NULL,
                                    sd == 2 ? (bc->y0 ? bc->y0 : bc->y) : NULL, sd == 2 ? u->yo : NULL,
                                    g_brisk_data != NULL, par.verbose, &B, &which, w2, sizeof(w2));
            if (okp && (!certified[sd - 1] || (sd == 1 ? B < margin_bound : B > margin_bound))) {
                if (certified[sd - 1])
                    snprintf(cert_why[sd - 1], sizeof(cert_why[0]), "better than %.17g from the certificate with the margin restored (%s)", margin_bound, bc->src);
                certified[sd - 1] = 1; rigorous[sd - 1] = B; prior[sd - 1] = which;
                if (which == 1 && has0) {
                    /* the certificate is the candidate as it was before the restoration */
                    if (sd == 1) { bound_free_blocks(bc->X, bc->nblk); bc->X = bc->X0; bc->X0 = NULL; }
                    else { free(bc->y); bc->y = bc->y0; bc->y0 = NULL; }
                    bc->value = bc->value0; bc->resid = bc->resid0; bc->lammin = bc->lammin0; bc->theta = 0;
                }
                else if (which == 2) {
                    /* (19 October 2026, the review S6) the returned point is what was certified: the certificate written by
                     * -boundx / -boundy and returned by the library is that point, with its value and measures on the data
                     * as read (before, the margin candidate stayed beside a bound it had not given) */
                    const PSOrig *Oc = u->Ofin ? u->Ofin : u->O;
                    if (sd == 1 && u->Xo) {
                        double **Xc = bound_alloc_blocks(Oc);
                        for (int k = 0; k < Oc->nblk; k++) {
                            const size_t len = Oc->bs[k] < 0 ? (size_t)-Oc->bs[k] : (size_t)Oc->bs[k] * Oc->bs[k];
                            memcpy(Xc[k], u->Xo[k], sizeof(double) * len);
                        }
                        if (bc->X) bound_free_blocks(bc->X, bc->nblk);
                        bc->X = Xc; bc->nblk = Oc->nblk;
                        bc->value = u->pr.pobj; bc->resid = u->pr.err[1]; bc->lammin = -u->pr.err[2];
                    } else if (sd == 2 && u->yo) {
                        double *yc = malloc(sizeof(double) * (Oc->m + 1));
                        if (yc) {
                            memcpy(yc, u->yo, sizeof(double) * Oc->m);
                            free(bc->y); bc->y = yc;
                            bc->value = u->pr.dobj; bc->resid = 0; bc->lammin = -u->pr.err[4];
                        }
                    }
                    bc->theta = 0;
                }
            } else if (!certified[sd - 1]) {
                const size_t l = strlen(cert_why[sd - 1]);
                snprintf(cert_why[sd - 1] + l, sizeof(cert_why[0]) - l, "; a-priori bounds: %s", w2);
            }
            t_cert[sd - 1] += wtime() - tp0;
        }
        /* 27 October 2026 (ISSUES 709): side p on the default route too, the method's iterate before the restoration and
         * the polishes by the a-priori route, the better rigorous bound kept (a user's patch gave the iterate to
         * -boundprior 2 only; a PolyOpt cut SDP, cutsdp_2: 4.0e-5 above b'y from the certificate, 2.0e-7 from the
         * iterate). The certificate is then that iterate, as the bound's point. */
        if (sd == 1 && par.bound_prior && !bc->prior_first) {
            const PSOrig *Op = u->Ofin ? u->Ofin : u->O;
            const double ti0 = wtime();
            int nbp = 0;
            double **Xbp = bpt_mapped(u, &par, Op, &nbp);
            double B; int which = 0; char w2[160] = "";
            if (Xbp && brisk_prior_bound(Op, 1, Xbp, NULL, NULL, NULL, g_brisk_data != NULL, par.verbose, &B, &which, w2, sizeof(w2))
                && (!certified[0] || B < rigorous[0]) && nbp == Op->nblk) {
                if (certified[0]) snprintf(cert_why[0], sizeof(cert_why[0]), "better than %.17g from %s", rigorous[0], prior[0] == 2 ? "the returned point" : "the certificate");
                certified[0] = 1; rigorous[0] = B; prior[0] = 3;
                if (bc->X) bound_free_blocks(bc->X, bc->nblk);
                bc->X = Xbp; bc->nblk = nbp; Xbp = NULL;
                bc->value = orig_cx(Op, bc->X, &bc->resid); bc->lammin = NAN; bc->theta = 0;
                snprintf(bc->src, sizeof(bc->src), "the method's iterate before the polishes");
            }
            if (Xbp) { for (int k = 0; k < nbp; k++) free(Xbp[k]); free(Xbp); }
            t_cert[0] += wtime() - ti0;
        }
        bc->certified = certified[sd - 1]; bc->rigorous = rigorous[sd - 1];
        if (certified[sd - 1] && !bc->valid) bc->valid = 1;
        snprintf(g_cert_why[sd - 1], sizeof g_cert_why[0], "%s", cert_why[sd - 1]); g_cert_prior[sd - 1] = prior[sd - 1];
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
    /* 31 October 2026 (the owner's external SDP benchmark, audit A2; the owner's decision): an infeasibility verdict is
     * returned only when the returned point certifies it on the problem as read. Otherwise the status is NUMERICAL
     * DIFFICULTIES and the Summary says the infeasibility is not confirmed: the verdict may rest on a presolve step (a
     * reduced problem infeasible), the problem may be weakly infeasible (no certificate exists), or it may be wrong. Until
     * then the check only wrote a line of the log, and a program that branched on the status took a false verdict. */
    if (status == ST_PINFEAS || status == ST_DINFEAS) {
        double cobj = 0, cviol = 0;
        const int cert = ps_infeas_check(O, Xo, yo, status == ST_PINFEAS ? 1 : 2, &cobj, &cviol, xc_on(u, O));
        if (cert != 1) {
            const int pinf_ = status == ST_PINFEAS;
            if (cert == 0) snprintf(cause, sizeof cause, "infeasibility not confirmed: the solve found the problem %s infeasible, but the returned %s does not prove it on the problem as read (violation %.1e); a presolve step may have changed the problem, or the problem may be infeasible without a certificate",
                                    pinf_ ? "primal" : "dual", pinf_ ? "y" : "X", cviol);
            else snprintf(cause, sizeof cause, "infeasibility not confirmed: the solve found the problem %s infeasible, but returned no point that proves it on the problem as read", pinf_ ? "primal" : "dual");
            status = ST_NUMERIC; u->status = ST_NUMERIC; g_infeas_unconf = 1 + (u->fr_removed > 0);
        }
    }
    if (fr_kept || g_infeas_unconf) { /* 5.17: the cause is known and said above the status (the diagnosis below would blame the multipliers) */ }
    else if (par.mfipm == 1 && status != ST_OPTIMAL && mfipm_stalled())   /* (3 November 2026) the matrix-free method asked for stalled */
        snprintf(cause, sizeof cause, "the matrix-free method stalled (its conjugate gradient solves hit their step limit); its best iterate is returned");
    else if ((status == ST_REDUCED || status == ST_NUMERIC || status == ST_MAXIT || status == ST_TIME) && acc > 10.0 * par.tol) {
        diagnose_cause(u, &par, O, Xo, xc_on(u, O), yo, &pr, acc, cause, sizeof(cause));
    }
    /* (ISSUES 746) the objective's sensitivity, for every result whose cause is not already said */
    if (!cause[0] && measured && status != ST_PINFEAS && status != ST_DINFEAS) amp_warning(O, Xo, xc_on(u, O), yo, &pr, acc, par.tol, g_amp_warn, sizeof g_amp_warn);
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
        const int cert = ps_infeas_check(O, Xo, yo, status == ST_PINFEAS ? 1 : 2, &cobj, &cviol, xc_on(u, O));
        /* 22 October 2026: the violation of the certificate normalized to b'y = 1 or <C,X> = -1 (postsolve.c) */
        /* (31 October 2026, ISSUES 725: in words a user reads; the conditions themselves in the diagnosis, -v) */
        if (cert == 1) snprintf(cause, sizeof cause, "certified: the returned %s proves it on the problem as read (%s, to a relative error of %.1e)", status == ST_PINFEAS ? "y" : "X", status == ST_PINFEAS ? "b'y > 0, A'y <= 0" : "<C,X> < 0, A(X) = 0, X >= 0", cviol);
        else if (cert == 0) snprintf(cause, sizeof cause, "not certified: the returned %s does not prove it on the problem as read (relative error %.1e); the verdict may rest on a presolve step, the problem may be infeasible without a certificate (weakly infeasible), or the verdict may be wrong", status == ST_PINFEAS ? "y" : "X", cviol);
        else snprintf(cause, sizeof cause, "no certificate: no point was returned");
        printf("  %s\n", cause);
    }
    if (cause[0] && par.verbose != 0 && !infeas) printf("  likely cause: %s\n", cause);
    if (g_amp_warn[0] && par.verbose != 0 && !infeas) printf("  warning: %s\n", g_amp_warn);
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
        if (PP->ps && PP->ps->nrec && pr.recovered + pr.unrecovered > 0)   /* (20 October 2026: no line that counts nothing) */
            printf("  postsolve: %d constraint(s) removed by presolve, %d dual(s) recovered, %d not (%.3fs)%s\n",
                   PP->ps->nrec, pr.recovered, pr.unrecovered, pr.t, pr.late ? "; the time limit stopped the recovery" : "");
    }
    printf("direction: %s%s\n", R.direction == 1 ? "NT" : "HKM", R.retried ? " (self-dual embedding)" : "");
    if (par.verbose > 0) printf("(solver setup %.3fs)\n", R.t_setup);
    textra += g_pipe_extra; g_pipe_extra = 0;   /* 5.16: the runs run_pipeline discarded */
    /* 26 October 2026: the total is the wall time from the end of the reading, which holds every step; the sum of the
     * parts (solve, presolve, postsolve, discarded attempts) left out what lies between them: 77 s of tiny_soc_N50000's
     * 99 s (SDP with SOCP: a test of the cones' data, once per block over all the entries) and the guaranteed-bound
     * step (18.7 s after a solve of 3.3 s on frame24_nmt_ad__077, SDP). The parts still explain it ("other" below) */
    const double t_sum = R.t_total + tpre + pr.t + textra;
    const double t_all = u->t_run0 > 0 ? fmax(t_sum, wtime() - u->t_run0) : t_sum;
    /* 23 October 2026: the setup and what no part covers (the end phase's attempts, the polish probes, a method tried
     * first, the re-solves), so that the parts add up to the total: the probes' 14 s of metric_50's 35 s were found
     * only by stack samples */
    const double t_other = t_all - tpre - pr.t - R.t_setup - R.t_schur - R.t_chol - R.t_dense - R.t_step;
    printf("time: total %.3fs (read %.3f, presolve+analysis %.3f) | schur %.3f  chol/solve %.3f  dense %.3f  step %.3f  postsolve %.3f  setup %.3f  other %.3f\n",
           t_all, tread, tpre, R.t_schur, R.t_chol, R.t_dense, R.t_step, pr.t, R.t_setup, t_other > 0 ? t_other : 0.0);
    /* 4.37 (A1): the bound of each side asked for; only a certified one is a bound */
    g_log_user++;   /* (ISSUES 725: the bound the user asked for is the user's) */
    for (int sd = 1; sd <= 2 && par.verbose != 0 && !g_log_brief; sd++) {   /* (3 November 2026: the default log has them in the Summary) */
        if (!(par.bound_side & sd)) continue;
        const BoundCert *bc = &u->cert[sd - 1];
        const char *SD = sd == 1 ? "P" : "D", *ul = sd == 1 ? "upper" : "lower";
        if (!bc->have) { printf("BOUND (%s): no certificate (no admissible candidate): no %s bound\n", SD, ul); continue; }
        if (bc->prior_first || isnan(bc->lammin))      /* (27 October 2026: the a-priori route's points: their value and residual; the bound charges lambda_min itself) */
            printf("BOUND (%s): %s bound %s = %.10e  (certified by a-priori bounds: %s %.1e rel; from attempt %d, %s)\n",
                   SD, ul, sd == 1 ? "<C,X>" : "b'y", bc->value, sd == 1 ? "|A(X)-b|" : "|C-A'y-Z|", bc->resid, bc->attempt, bc->src);
        else
        printf("BOUND (%s): %s bound %s = %.10e  (certificate: %s %.1e rel, lambda_min(%s) %.1e rel; from attempt %d, %s%s)\n",
               SD, ul, sd == 1 ? "<C,X>" : "b'y", bc->value,
               sd == 1 ? "|A(X)-b|" : "|C-A'y-Z|", bc->resid, sd == 1 ? "X" : "Z", bc->lammin, bc->attempt, bc->src,
               bc->theta > 0 ? ", margin restored" : "");
        if (certified[sd - 1] && prior[sd - 1])
            printf("CERTIFIED (%s): rigorous %s bound %.17g on the optimal value (%.2fs)\n"
                   "           by a-priori bounds of %s found from the problem (%s%s); from %s\n",
                   SD, ul, rigorous[sd - 1], t_cert[sd - 1],
                   sd == 1 ? "y" : "the diagonal of X", strncmp(cert_why[sd - 1], "better than", 11) ? "the margin check failed: " : "", cert_why[sd - 1],
                   prior[sd - 1] == 1 ? "the certificate" : prior[sd - 1] == 3 ? "the method's iterate before the polishes" : "the returned point");
        else if (certified[sd - 1])
            printf("CERTIFIED (%s): rigorous %s bound %.17g on the optimal value (%.2fs)\n", SD, ul, rigorous[sd - 1], t_cert[sd - 1]);
        else if (u->status == ST_PINFEAS || u->status == ST_DINFEAS)
            printf("NOT CERTIFIED (%s): the problem was found infeasible: no %s bound\n", SD, ul);
        else
            printf("NOT CERTIFIED (%s): %s: no guaranteed %s bound (the value above holds only approximately)\n", SD, cert_why[sd - 1], ul);
    }
    if (do_certify && !par.bound_side && par.verbose != 0) printf("(-certify needs -bound p, d or pd)\n");
    g_log_user--;
    /* 4.37 (B1): the re-solves are reported in the summary (Solve 1, Solve 2, ...) and the advice says how to avoid
     * them (20 October 2026, the owner: no note above the line) */
    if (yfile) {
        FILE *f = brisk_out_open(yfile);
        if (f) {
            for (int i = 0; i < O->m; i++) fprintf(f, "%.17g\n", yo[i]);
            brisk_out_close(f, yfile);
        }
    }
    /* (23 October 2026: the writers of the matrices complete a cone-vector block: they write X itself) */
    /* (27 October 2026, the night: a completion that does not fit in memory leaves the block a vector, which the writers
     * would read as a matrix; then the file is not written, and it is said) */
    if (Xo && xfile) {
        if (xc_expand(O, Xo, xc_on(u, O))) fprintf(stderr, "brisk: not enough memory for the matrix X of a block solved as a second-order cone: %s not written\n", xfile);
        else write_matrix_file(xfile, O, Xo);
    }
    /* the certificates of the bound mode, in the formats of -x and -y */
    if (boundxfile) {
        if (!(par.bound_side & 1) || !u->cert[0].have || !u->cert[0].X) fprintf(stderr, "brisk: -boundx: no certificate X to write (it needs -bound p or pd and an admissible candidate)\n");
        else write_matrix_file(boundxfile, O, u->cert[0].X);
    }
    if (boundyfile) {
        if (!(par.bound_side & 2) || !u->cert[1].have || !u->cert[1].y) fprintf(stderr, "brisk: -boundy: no certificate y to write (it needs -bound d or pd and an admissible candidate)\n");
        else {
            FILE *f = brisk_out_open(boundyfile);
            if (f) { for (int i = 0; i < O->m; i++) fprintf(f, "%.17g\n", u->cert[1].y[i]); brisk_out_close(f, boundyfile); }
        }
    }
    if (zfile || g_brisk_result) {
        /* Z = C - A'y on the original data */
        double **Zo = malloc(sizeof(double *) * (O->nblk + 1));
        for (int k = 0; k < O->nblk; k++) {
            size_t n = abs(O->bs[k]);
            /* 4.42: the library does not return the dense Z of a block that would take more
             * than 30% of the memory (as for X; dense case13659: 10 GB for n = 35 503): the
             * block is NULL, Z = C - A'y is given by y and the data. -z writes it anyway. */
            /* (23 October 2026: a caller that maps the cone-vector blocks itself, sedumi.c, forms z = c - A'y on its own) */
            if (!zfile && g_xcone_accept && xc_on(u, O) && u->xc_o[k] >= 0) { Zo[k] = NULL; continue; }
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
            /* the bound of each side asked for, with its certificate (a copy); the legacy fields
             * are those of the first side asked for */
            for (int sd = 1; sd <= 2; sd++) {
                const BoundCert *bc = &u->cert[sd - 1];
                memset(&res->bnd[sd - 1], 0, sizeof(res->bnd[0]));
                res->bnd[sd - 1].rigorous = NAN; res->bnd[sd - 1].value = NAN;
                if (!(par.bound_side & sd)) continue;
                res->bnd[sd - 1].asked = 1;
                if (!bc->have) continue;
                res->bnd[sd - 1].have = 1; res->bnd[sd - 1].valid = bc->valid; res->bnd[sd - 1].certified = certified[sd - 1];
                res->bnd[sd - 1].value = bc->value; res->bnd[sd - 1].resid = bc->resid; res->bnd[sd - 1].lammin = bc->lammin;
                res->bnd[sd - 1].rigorous = rigorous[sd - 1];
                snprintf(res->bnd[sd - 1].src, sizeof(res->bnd[0].src), "%s%s", prior[sd - 1] ? "a-priori bounds; " : "", prior[sd - 1] == 2 ? "the returned point" : bc->src);
                if (sd == 1 && bc->X) {      /* (allocations checked: a copy that does not fit is left out, the bound kept) */
                    res->bnd[0].X = calloc(O->nblk + 1, sizeof(double *));
                    for (int k = 0; res->bnd[0].X && k < O->nblk; k++) {
                        const size_t len = O->bs[k] < 0 ? (size_t)-O->bs[k] : (size_t)O->bs[k] * O->bs[k];
                        res->bnd[0].X[k] = malloc(sizeof(double) * (len + 1));
                        if (!res->bnd[0].X[k]) { for (int q = 0; q < k; q++) free(res->bnd[0].X[q]); free(res->bnd[0].X); res->bnd[0].X = NULL; break; }
                        memcpy(res->bnd[0].X[k], bc->X[k], sizeof(double) * len);
                    }
                }
                if (sd == 2 && bc->y && (res->bnd[1].y = malloc(sizeof(double) * (O->m + 1)))) memcpy(res->bnd[1].y, bc->y, sizeof(double) * O->m);
            }
            {
                const int s0 = (par.bound_side & 1) ? 0 : 1;
                res->bound_side = par.bound_side && res->bnd[s0].have ? s0 + 1 : 0;
                res->bound_value = res->bnd[s0].value; res->bound_resid = res->bnd[s0].resid; res->bound_lammin = res->bnd[s0].lammin;
                res->bound_valid = res->bnd[s0].valid; res->bound_certified = res->bnd[s0].certified;
                res->bound_rigorous = res->bnd[s0].rigorous;
            }
            res->n_resolves = g_natt;
            res->t_resolves = 0;
            for (int a = 0; a < g_natt; a++) res->t_resolves += g_att[a].t;
            snprintf(res->cause, sizeof(res->cause), "%s", cause);
            for (int e = 1; e <= 6; e++)
                res->err[e] = measured && !infeas ? pr.err[e] : (e == 4 && !infeas ? pr.err[4] : R.err[e]);
            res->time = t_all;        /* (the Summary's total: the wall time of the run) */
            res->m = O->m; res->nblk = O->nblk;
            res->bs = malloc(sizeof(int) * (O->nblk + 1));
            memcpy(res->bs, O->bs, sizeof(int) * O->nblk);
            res->y = malloc(sizeof(double) * (O->m + 1));
            memcpy(res->y, yo, sizeof(double) * O->m);
            res->Z = Zo; Zo = NULL;
            res->have_x = Xo != NULL;
            /* 23 October 2026: the cone-vector blocks stay vectors for a caller that maps them itself (sedumi.c reads
             * g_xcone_res); any other caller gets the matrix */
            const int *xcr = xc_on(u, O);
            free(g_xcone_res); g_xcone_res = NULL;
            if (Xo && xcr && g_xcone_accept) { g_xcone_res = malloc(sizeof(int) * (O->nblk + 1)); for (int k = 0; k < O->nblk; k++) g_xcone_res[k] = xcr[k]; }
            int xfit = 1;
            if (Xo && xcr && !g_xcone_res && xc_expand(O, Xo, u->xc_o)) {
                xfit = 0;
                fprintf(stderr, "brisk: not enough memory for the matrix X of a block solved as a second-order cone; y and the objectives are returned\n");
            }
            xcr = NULL;
            if (Xo && !xfit) res->have_x = 0;
            if (Xo && xfit) {
                /* 5.16: the copy can fail on large blocks (two cones of 8192 read as arrow blocks: 1 GB
                 * more; under a 5 GB cap memcpy wrote through NULL): then no X, said so, y kept */
                res->X = calloc((size_t)O->nblk + 1, sizeof(double *));
                int ok = res->X != NULL;
                for (int k = 0; k < O->nblk && ok; k++) {
                    const size_t n = abs(O->bs[k]), len = O->bs[k] < 0 || (g_xcone_res && g_xcone_res[k] >= 0) ? n : n * n;
                    res->X[k] = malloc(sizeof(double) * (len + 1));
                    if (!res->X[k]) { ok = 0; fprintf(stderr, "brisk: not enough memory to return X (block %d of order %zu: %.2f GB); y and the objectives are returned\n", k + 1, n, 8e-9 * (double)len); break; }
                    memcpy(res->X[k], Xo[k], sizeof(double) * len);
                }
                if (!ok) {
                    if (res->X) for (int k = 0; k < O->nblk; k++) free(res->X[k]);
                    free(res->X); res->X = NULL; res->have_x = 0;
                    free(g_xcone_res); g_xcone_res = NULL;
                }
            }
        }
        if (Zo) { for (int k = 0; k < O->nblk; k++) free(Zo[k]); free(Zo); }
    }
    const char *xo_env = getenv("BRISK_XOUT");
    if (Xo && xo_env) {
        if (xc_expand(O, Xo, xc_on(u, O))) fprintf(stderr, "brisk: not enough memory for the matrix X of a block solved as a second-order cone: %s not written\n", xo_env);
        else write_x_binary(xo_env, O, Xo);
    }
    if (par.verbose > 0 && !quiet_summary) {
        double e6[7];
        for (int e = 1; e <= 6; e++) e6[e] = measured && !infeas ? pr.err[e] : (e == 4 && !infeas ? pr.err[4] : R.err[e]);
        print_summary(u, &par, O, status, e6, infeas, cause, t_all, tread, tpre, R.iters, NULL, pobj, dobj, near_miss);
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
    char why[400];
    int ok = brisk_certify_orig(O, side, X, y, g_brisk_data != NULL, &B, why, sizeof(why)), prior = 0;
    { extern int g_dup_inexact;   /* (audit A20) */
      if (g_dup_inexact) { ok = 0; B = NAN; snprintf(why, sizeof why, "%d entr%s of the data given in parts whose exact sum is not a double: the problem as given cannot be represented exactly, so no bound is certified for it", g_dup_inexact, g_dup_inexact == 1 ? "y" : "ies"); } }
    if (!ok && !g_dup_inexact_any()) {
        /* no margin: a-priori bounds of the other side (boundcert.c) */
        double B2; char w2[160];
        if (g_certify_prior && brisk_prior_bound(O, side, X, NULL, y, NULL, g_brisk_data != NULL, g_certify_verbose, &B2, NULL, w2, sizeof(w2))) { ok = prior = 1; B = B2; }
        else { B = NAN; if (g_certify_prior) { const size_t l = strlen(why); snprintf(why + l, sizeof(why) - l, "; a-priori bounds: %s", w2); } }
    }
    g_log_user++;   /* (ISSUES 725: the certification asked for is the user's) */
    if (ok) printf("CERTIFIED (%s): a guaranteed %s bound on the optimal value: %.17g (%.2fs)%s\n",
                   side == 1 ? "P" : "D", side == 1 ? "upper" : "lower", B, wtime() - t0,
                   prior ? (side == 1 ? "\n           by a-priori bounds of y found from the problem (the certificate has no margin)" : "\n           by a-priori bounds of the diagonal of X found from the problem (the certificate has no margin)") : "");
    else printf("NOT CERTIFIED (%s): %s\n", side == 1 ? "P" : "D", why);
    g_log_user--;
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

/* 2 November 2026 (the owner: "does BRISK warn the user that the solution may be inaccurate even if the residuals are
 * small if the norm of the solution is large?"; ISSUES 746): the objective's sensitivity to the returned errors, for
 * every result, OPTIMAL included. Each error moves the objective by at most its size times the size of the solution it
 * acts on: the primal residual A(X) - b by |y| (|y' r| <= |y|_1 |r|), the dual residual C - A'y - Z by tr X (|<X, R>| <=
 * tr X |R|_F for X >= 0), X's negative eigenvalue by |y|_1 and Z's by tr X. When the largest of these, relative to the
 * objective, exceeds ten times the larger of the tolerance and the result's own error, the small errors do not make the
 * value accurate: a solution of a huge norm (Kocvara's moment-SOS hierarchies, arXiv 2610.17261, its amplification factor;
 * MOSEK's OPTIMAL on diamond_patch: a primal residual of 1.6e-7 against a dual of norm 5e6). The status is not changed;
 * the Summary says it. Empty when nothing is to be said. */
static void amp_warning(const PSOrig *O, double **Xo, const int *xc, const double *yo, const PSResult *pr, double acc,
                        double tol, char *out, size_t len) {
    out[0] = 0;
    if (!Xo || !yo) return;
    if (xc) for (int k = 0; k < O->nblk; k++) if (xc[k] >= 0) return;     /* (cone-vector blocks: not measured here) */
    /* the first-order effect of each error on the objective, from C = Z + A'y + R_d:
     *  - X moved back onto A(X) = b by D with A(D) = -r: <C, D> = <Z, D> - y'r, so about |y'r|;
     *  - X's negative eigenvalue -e lifted by e I (and the rows corrected as above): about e tr Z;
     *  - Z's negative eigenvalue -e lifted by e I through y: about e tr X;
     *  - the dual residual: |<X, R_d>| <= tr X |R_d|_F (X >= 0). */
    double *ax = calloc(O->m + 1, sizeof(double));
    double trX = 0, trC = 0, trAy = 0;
    for (size_t q = 0; q < O->nnz; q++) {
        const int o = O->blk[q], i = O->ii[q], j = O->jj[q], no = abs(O->bs[o]), r = O->con[q];
        const double xv = O->bs[o] < 0 ? Xo[o][i] : Xo[o][i + (size_t)j * no];
        if (r >= 0) ax[r] += (O->bs[o] > 0 && i != j) ? 2.0 * O->v[q] * xv : O->v[q] * xv;
        if (i == j || O->bs[o] < 0) { if (r < 0) trC += O->v[q]; else trAy += yo[r] * O->v[q]; }
    }
    for (int k = 0; k < O->nblk; k++) {
        const int n = abs(O->bs[k]);
        for (int i = 0; i < n; i++) trX += O->bs[k] < 0 ? Xo[k][i] : Xo[k][i + (size_t)i * n];
    }
    double yr = 0, ynorm = 0;
    for (int i = 0; i < O->m; i++) { yr += yo[i] * (ax[i] - O->b[i]); ynorm += fabs(yo[i]); }
    free(ax);
    const double trZ = fabs(trC - trAy);
    const double nb1 = O->nb1, nC1 = O->nC1;
    const double obj = fabs(pr->pobj) > fabs(pr->dobj) ? pr->pobj : pr->dobj, den = 1.0 + fabs(obj);
    const double rd = fabs(pr->err[3]) * (1 + nC1), lx = fabs(pr->err[2]) * (1 + nb1), lz = fabs(pr->err[4]) * (1 + nC1);
    /* (the eigenvalue violations are not used: lifting X's or Z's negative direction costs its eigenvalue times v'Zv or
     * v'Xv, which needs the eigenvector; tr Z and tr X overstate it up to the block's order, and the measured eigenvalue
     * near zero sits at a rounding floor of 1e-12 of the norm: on 38 Mittelmann files those terms flagged 11 results whose
     * objectives are accurate, the two first-order terms none. Those violations are DIMACS errors 2 and 4 as they are.) */
    (void)trZ; (void)lx; (void)lz;
    const double t[2] = { fabs(yr) / den, trX * rd / den };
    if (getenv("BRISK_AMPDBG")) printf("   [objective sensitivity: y'r %.1e, <X,Rd> %.1e (relative); tol %.0e, error %.1e]\n", t[0], t[1], tol, acc);
    const int w = t[1] > t[0];
    if (!(t[w] > 10.0 * fmax(tol, acc))) return;
    if (w == 0)
        snprintf(out, len, "the objective may be off by about %.0e (relative) although the errors are small: y is large "
                 "(sum of |y_i| %.1e), and with the primal residual y'(A(X) - b) = %.1e", t[0], ynorm, yr);
    else
        snprintf(out, len, "the objective may be off by about %.0e (relative) although the errors are small: X is large "
                 "(trace %.1e), and the dual residual %.0e times that can move the objective that much", t[1], trX, rd);
}

/* 4.37 (B2b): the first likely cause of a result that misses the tolerance */
/* (27 October 2026, the night: xc, the blocks of Xo kept as cone vectors; their diagonal is read from the vector. The
 * diagnosis completed them to matrices first, and a cone of order 100,001 (r2_reglmi_N100k_p100, SDP with SOCP, cut by
 * an iteration limit) needs 80 GB: the completion failed, the block stayed a vector, and this loop read it as n^2
 * entries: a segmentation fault) */
static void diagnose_cause(const Run *u, const Params *par, const PSOrig *O, double **Xo, const int *xc, const double *yo,
                           const PSResult *pr, double acc, char *out, size_t len) {
    double trX = 0, ynorm = 0;
    if (Xo)
        for (int k = 0; k < O->nblk; k++) {
            const int n = abs(O->bs[k]);
            if (xc && xc[k] >= 0) { const double f = xc_entry_f(n, Xo[k]); for (int i = 0; i < n; i++) trX += xc_entry(n, xc[k], Xo[k], f, i, i); continue; }
            for (int i = 0; i < n; i++) trX += O->bs[k] < 0 ? Xo[k][i] : Xo[k][i + (size_t)i * n];
        }
    if (yo) for (int i = 0; i < O->m; i++) ynorm += fabs(yo[i]);
    const double nb1 = O->nb1, nC1 = O->nC1;
    const double obj = fabs(pr->pobj) > fabs(pr->dobj) ? pr->pobj : pr->dobj;
    const int tb_bind = tb_binds(u, acc, par->tol);
    const double gap = fabs(pr->err[5]);
    const double lamz = pr->err[4] * (1 + nC1), lamx = pr->err[2] * (1 + nb1);
    if (tb_bind || (Xo && trX > 1e5 * (1 + nb1) && trX > 1e4 * (1 + fabs(obj)))) {
        /* (31 October 2026, ISSUES 725: the causes in words a user reads without the solver's internals; the options are
         * in the advice below the Summary, in the syntax of the interface used) */
        snprintf(out, len, "the optimum may not be attained: the trace of X grows without bound (%.1e here), so the dual problem may have "
                 "no strictly feasible point",
                 trX);
        (void)tb_bind;
        return;
    }
    if (yo && ynorm > 1e5 * (1 + nC1) && ynorm > 1e4 * (1 + fabs(obj))) {
        snprintf(out, len, "the optimum may not be attained: y grows without bound (sum of |y_i| %.1e here), so the primal problem may have "
                 "no strictly feasible point", ynorm);
        return;
    }
    if (Xo && trX * lamz / (1 + fabs(obj)) > 10 * fmax(gap, par->tol)) {
        snprintf(out, len, "X is large (trace %.1e), so the small violation %.0e of Z >= 0 can move the objective by about %.0e, although the two objective values agree to %.0e",
                 trX, lamz, trX * lamz / (1 + fabs(obj)), gap);
        return;
    }
    if (yo && ynorm * lamx / (1 + fabs(obj)) > 10 * fmax(gap, par->tol)) {
        snprintf(out, len, "y is large (sum of |y_i| %.1e), so the small violation %.0e of X >= 0 can move the objective by about %.0e, although the two objective values agree to %.0e",
                 ynorm, lamx, ynorm * lamx / (1 + fabs(obj)), gap);
        return;
    }
    if (u->R.nreg >= 5) {
        snprintf(out, len, "the problem is ill-conditioned: the method's linear systems had to be regularized %d times", u->R.nreg);
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
            snprintf(out, len, "free variables given as differences of two nonnegative ones have grown (both parts about %.1e for a difference of %.1e); the cancellation limits the accuracy of the free values (%s avoids it)", wm, wf, syn_kv("freeelim", "1", 0));
            return;
        }
    }
    if (acc > par->red_acc) snprintf(out, len, "not identified");
}

/* (3 November 2026, the owner: "Give them the iteration tables, make it every n steps for very small problems with n chosen
 * based on iteration speed") the default log's iteration table of the LP, SOCP and QP interior-point methods, in the SDP
 * solver's columns. g_itab: set by the routes around the method whose table the default log shows (0 elsewhere, so that
 * inner and repeated solves print nothing). The rows: the first ten iterations, then one every n, where n is set at the
 * tenth from the time an iteration has taken so far, so that a row comes about every 0.05 s (1, 2, 5, 10, 20, 50, ...);
 * the last iteration always (brisk_itab_end). One table at a time (the methods print from one thread). */
int g_itab = 0;
static struct { int on, n, last, printed; double t0, r[8]; } g_it;
double g_tab_t0 = 0;   /* (the clock of the SDP solver's table: the run's start; the other tables use g_it.t0) */
static void itab_print(int it, const double *r) {
    log_user("%4d  %+15.8e  %+15.8e   %7.1e  %7.1e  %7.1e  %7.1e  %5.3f %6.2f\n", it, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);   /* (widths: a nan keeps the columns) */
}
void brisk_itab_row(int it, double pobj, double dobj, double pinf, double dinf, double gap, double mu, double step) {
    if (!g_itab) return;
    const double now = wtime();
    if (!g_it.on || it == 0) {
        if (g_it.on && g_it.printed != g_it.last) itab_print(g_it.last, g_it.r);   /* (a restart of the method: the previous table's last row) */
        g_it.on = 1; g_it.n = 1; g_it.printed = -1; g_it.t0 = now;
        log_user(" ITE  PRIMAL OBJ        DUAL OBJ           PFEAS    DFEAS    GAP      MU       STEP   TIME\n");
    }
    if (it == 10 && g_it.n == 1) {
        const double ti = (now - g_it.t0) / 10.0;
        double n = ti > 0 ? 0.05 / ti : 1e9;
        long k = 1; while (k * 10 <= n) k *= 10;
        g_it.n = (int)(n >= 5 * k ? 5 * k : n >= 2 * k ? 2 * k : k);
    }
    g_it.r[0] = pobj; g_it.r[1] = dobj; g_it.r[2] = pinf; g_it.r[3] = dinf; g_it.r[4] = gap; g_it.r[5] = mu; g_it.r[6] = step; g_it.r[7] = now - g_it.t0;
    g_it.last = it;
    if (it < 10 || it % g_it.n == 0) { itab_print(it, g_it.r); g_it.printed = it; }
}
/* (3 November 2026) the same for the simplex methods, whose iterations are many and cheap: a row at the first iteration,
 * then every n, n set at the 100th iteration from their speed (a row about every 0.05 s, 1, 2, 5, 10, ...), and the last.
 * The columns: the iteration, the method and phase, the objective, the sum of the primal infeasibilities and their count. */
static struct { int on; long n, last, printed; double t0, obj, pinf; int ninf; char what[16]; } g_st;
static void stab_print(void) {
    log_user("%8ld  %-14s %+.10e   %.2e %7d  %7.2f\n", g_st.last, g_st.what, g_st.obj, g_st.pinf, g_st.ninf, wtime() - g_st.t0);
}
int g_stab = 0;   /* (the simplex method's table: set only where it runs alone, never beside the interior-point method) */
int brisk_stab_due(long it) {   /* (whether the next row is to be printed: the caller computes the infeasibility only then) */
    if (!g_stab) return 0;
    if (!g_st.on || it < 10) return 1;   /* (the first ten iterations always, as in the interior-point methods' table) */
    if (it == 100 && g_st.n == 0) {
        const double ti = (wtime() - g_st.t0) / 100.0; const double n = ti > 0 ? 0.05 / ti : 1e9;
        long k = 1; while (k * 10 <= n) k *= 10;
        g_st.n = n >= 5 * k ? 5 * k : n >= 2 * k ? 2 * k : k;
    }
    return g_st.n > 0 && it % g_st.n == 0;
}
void brisk_stab_row(long it, const char *what, double obj, double pinf, int ninf) {
    if (!g_stab) return;
    if (g_st.on && it == g_st.printed && !strcmp(what, g_st.what)) return;   /* (the same row twice: a phase's end and the due row) */
    if (!g_st.on) {
        g_st.on = 1; g_st.n = 0; g_st.printed = -1; g_st.t0 = wtime();
        log_user("    ITER  METHOD          OBJECTIVE          PRIMAL INF   ROWS     TIME\n");
    }
    g_st.last = it; g_st.obj = obj; g_st.pinf = pinf; g_st.ninf = ninf; snprintf(g_st.what, sizeof g_st.what, "%s", what);
    stab_print(); g_st.printed = it;
}
void brisk_stab_end(long it, const char *what, double obj, double pinf, int ninf) {   /* (the last iteration's row) */
    if (g_stab && g_st.on && g_st.printed != it) brisk_stab_row(it, what, obj, pinf, ninf);
    g_st.on = 0;
}
void brisk_itab_end(void) {
    if (g_itab && g_it.on && g_it.printed != g_it.last) itab_print(g_it.last, g_it.r);
    g_it.on = 0;
}
/* (3 November 2026, the owner: "the logs do not seem to show the polishing or dd steps. They should!") the steps after the
 * iterations in the default log: each polish and end phase on one line, its error (the largest of the relative errors the
 * solve measures) before and after, and whether its point was kept; the double-double end phase's iterations as rows of the
 * table. The diagnosis log (-v) keeps its own detailed lines. */
int g_log_step_off = 0;
void brisk_log_step(const char *what, double before, double after, int kept) {
    if (!g_log_brief || g_log_step_off) return;
    const double t0c = g_it.t0 > 0 ? g_it.t0 : g_tab_t0, t = t0c > 0 ? wtime() - t0c : 0;   /* (the table's clock, as its TIME column) */
    brisk_itab_end();
    log_user("      %-43s error %.1e -> %.1e %-8s%6.2f\n", what, before, after, kept ? "kept" : "not kept", t);
}
static int g_dd_rows = 0;
void brisk_ddtab_row(int it, double pobj, double dobj, double pinf, double dinf, double gap, double mu, double step) {
    if (!g_log_brief) return;
    if (it == 0) { brisk_itab_end(); log_user("      end phase in double-double arithmetic (about 31 digits), its iterations:\n"); g_dd_rows = 1; }
    const double r[8] = { pobj, dobj, pinf, dinf, gap, mu, step, 0 };
    (void)g_dd_rows;
    log_user("%4d  %+15.8e  %+15.8e   %7.1e  %7.1e  %7.1e  %7.1e  %5.3f %6.2f\n", it, r[0], r[1], r[2], r[3], r[4], r[5], r[6],
             g_it.t0 > 0 ? wtime() - g_it.t0 : g_tab_t0 > 0 ? wtime() - g_tab_t0 : 0);
}
/* ---- 4.30 library interface -------------------------------------------------------- */
int (*brisk_print_hook)(const char *s, int is_err) = NULL;
int g_log_brief = 0, g_log_user = 0;   /* (ISSUES 725, brisk.h: the default log prints only the user's lines) */
/* a line to stdout is dropped in the default log unless it is a user line */
#define LOG_DROPPED(f) ((f) == stdout && g_log_brief && g_log_user <= 0)
/* (3 November 2026, the owner: "correct the kind of formatting problems ... everywhere") the default log's lines are at most
 * LOG_WIDTH columns: a longer line is broken at spaces, its continuation under the start of its text (after "Label: " or
 * after the gap of two spaces or more that follows a label or an option). Not in the diagnosis log (-v, BRISK_LOGDIAG),
 * whose long lines are read by scripts. g_log_col: the column the last text left stdout at */
#define LOG_WIDTH 100
int g_log_wrap = 1;
static int g_log_col = 0;
static int log_vis(const char *p, int n) { int w = 0; for (int i = 0; i < n; i++) if (((unsigned char)p[i] & 0xC0) != 0x80) w++; return w; }
static void log_wrapped(const char *txt, void (*sink)(const char *, int, void *), void *ctx) {
    const char *p = txt;
    while (*p) {
        const char *nl = strchr(p, '\n');
        const int len = nl ? (int)(nl - p) : (int)strlen(p);
        /* (a table's header and rows are never broken: a row starts with its number, a header with " ITE" or "    ITER") */
        int tab = 0; { int k = 0; while (k < len && p[k] == ' ') k++; tab = (k < len && p[k] >= '0' && p[k] <= '9') || (len >= 4 && !strncmp(p + k, "ITE", 3)); }
        if (!g_log_wrap || g_log_col > 0 || tab || log_vis(p, len) <= LOG_WIDTH) {
            sink(p, len + (nl ? 1 : 0), ctx);
            g_log_col = nl ? 0 : g_log_col + log_vis(p, len);
        } else {
            /* the continuation's indent: after "Label: " followed by spaces, else after the first gap of 2+ spaces past the
             * leading ones, else after ": ", else the leading spaces plus two */
            int lead = 0; while (lead < len && p[lead] == ' ') lead++;
            int ind = -1;
            for (int i = lead; i + 1 < len && i < 60; i++) if (p[i] == ' ' && p[i + 1] == ' ') { int j = i; while (j < len && p[j] == ' ') j++; ind = j; break; }
            if (ind < 0) for (int i = lead; i + 1 < len && i < 60; i++) if (p[i] == ':' && p[i + 1] == ' ') { ind = i + 2; break; }
            if (ind < 0 || ind > 60) ind = lead + 2;
            char pad[64]; memset(pad, ' ', sizeof pad); 
            int pos = 0, first = 1;
            while (pos < len) {
                const int avail = LOG_WIDTH - (first ? 0 : ind);
                int rem = len - pos, take = rem;
                if (log_vis(p + pos, rem) > avail) {
                    take = 0; int brk = -1;
                    int pbrk = -1, obrk = -1;   /* (a break after a comma or a semicolon is preferred: "gap 1e-9, compl 1e-9" stays
                                                  * together; then a space outside parentheses and quotes: the code of an option
                                                  * in the advice, set_attribute(model, "bound", "primal"), is not split) */
                    int depth = 0, quote = 0;
                    for (int i = 0; i < rem && log_vis(p + pos, i) <= avail; i++) {
                        const char ch = p[pos + i];
                        if (ch == '"') quote = !quote; else if (!quote && ch == '(') depth++; else if (!quote && ch == ')' && depth > 0) depth--;
                        if (ch == ' ' && i > 0) { brk = i; if (!depth && !quote) { obrk = i; if (p[pos + i - 1] == ',' || p[pos + i - 1] == ';') pbrk = i; } }
                    }
                    if (pbrk > avail / 2) brk = pbrk; else if (obrk > avail / 2) brk = obrk;
                    take = brk > 0 ? brk : rem;   /* (a word longer than the line: left whole) */
                }
                if (!first) sink(pad, ind, ctx);
                sink(p + pos, take, ctx); sink("\n", 1, ctx);
                pos += take; while (pos < len && p[pos] == ' ') pos++;
                first = 0;
            }
            if (!nl) { /* (no newline at the end of the text: the last piece stays open) */ }
            g_log_col = 0;
        }
        if (!nl) break;
        p = nl + 1;
    }
}
#ifndef BRISK_LIBRARY
#undef printf
#undef fprintf
static void sink_stdout(const char *p, int n, void *ctx) { (void)ctx; fwrite(p, 1, (size_t)n, stdout); }
int brisk_printf(const char *fmt, ...) {
    if (LOG_DROPPED(stdout)) return 0;
    char buf[4096]; va_list ap; va_start(ap, fmt); const int n = vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    if (n < (int)sizeof buf) { log_wrapped(buf, sink_stdout, NULL); return n; }
    char *big = malloc((size_t)n + 1); if (!big) return n;
    va_start(ap, fmt); vsnprintf(big, (size_t)n + 1, fmt, ap); va_end(ap);
    log_wrapped(big, sink_stdout, NULL); free(big); return n;
}
int brisk_fprintf(FILE *f, const char *fmt, ...) {
    if (LOG_DROPPED(f)) return 0;
    va_list ap; va_start(ap, fmt); const int n = vfprintf(f, fmt, ap); va_end(ap); return n;
}
#define printf(...) brisk_printf(__VA_ARGS__)
#define fprintf(...) brisk_fprintf(__VA_ARGS__)
#endif

void brisk_result_free(BriskResult *res) {
    if (!res) return;
    if (res->X) { for (int k = 0; k < res->nblk; k++) free(res->X[k]); free(res->X); }
    if (res->Z) { for (int k = 0; k < res->nblk; k++) free(res->Z[k]); free(res->Z); }
    if (res->bnd[0].X) { for (int k = 0; k < res->nblk; k++) free(res->bnd[0].X[k]); free(res->bnd[0].X); }
    for (int sd = 0; sd < 2; sd++) { free(res->bnd[sd].y); free(res->bnd[sd].sx); }
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

typedef struct { char *b; size_t n, cap; } LogAcc;
static void sink_acc(const char *p, int n, void *ctx) {
    LogAcc *a = (LogAcc *)ctx;
    if (a->n + (size_t)n + 1 > a->cap) { size_t c = (a->n + (size_t)n + 1) * 2; char *nb = realloc(a->b, c); if (!nb) return; a->b = nb; a->cap = c; }
    memcpy(a->b + a->n, p, (size_t)n); a->n += (size_t)n; a->b[a->n] = 0;
}
static int brisk_vprint(FILE *f, const char *fmt, va_list ap) {
    if (LOG_DROPPED(f)) return 0;
    if (f == stdout && g_log_wrap) {   /* (the default log's width, as the command line's) */
        va_list aq; va_copy(aq, ap);
        const int n0 = vsnprintf(NULL, 0, fmt, aq); va_end(aq);
        char *t = malloc((size_t)(n0 > 0 ? n0 : 0) + 1); if (!t) return n0;
        vsnprintf(t, (size_t)n0 + 1, fmt, ap);
        LogAcc a = { NULL, 0, 0 }; log_wrapped(t, sink_acc, &a); free(t);
        if (a.b) { if (brisk_print_hook) brisk_print_hook(a.b, 0); else fputs(a.b, stdout); free(a.b); }
        return n0;
    }
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

/* (22 October 2026) BRISK_LIBRARY: the QP or QCQP P with the command-line options argv[1..argc-1] (argv[0] ignored) */
int brisk_run_qp(const QpProb *P, int argc, char **argv, BriskResult *res) {
    char **av = malloc(sizeof(char *) * ((size_t)(argc > 1 ? argc : 1) + 2));
    if (!av) { memset(res, 0, sizeof(*res)); res->status = -1; return -1; }
    av[0] = "brisk"; av[1] = (char *)g_qp_data_name;
    for (int i = 1; i < argc; i++) av[i + 1] = argv[i];
    const int ac = (argc > 1 ? argc : 1) + 1;
    av[ac] = NULL;
    g_qp_data = P;
    const int rc = brisk_run(ac, av, res);
    g_qp_data = NULL;
    free(av);
    return rc;
}

/* (28 October 2026, the SOCP developer) a QP in memory solved without brisk_main: the QP options only (-qpas -qpipm -qpauto
 * -tol -maxit -timelimit -threads -q; anything else: -2, nothing done, the caller takes the general route), qp_solve, and
 * x (n), y (m), z (n) and info[8] (status as brisk_result_status, iterations, objective, the primal, dual and gap errors,
 * the time, the method: 1 active-set, 2 interior-point) written to the caller's arrays. For small QPs, whose solve takes
 * microseconds: the whole program around it (brisk_main's setup, its option parser, the result object) cost several times
 * the solve, most of all on cold caches (the owner's independent QP benchmark: 325 us a call of a QP of ten variables, the
 * solve 20 us). The same solve as brisk_run_qp's: the same options, the same qp_solve; no log (quiet only) */
int brisk_qp_run_lean(const QpProb *P, int nopt, const char *const *opts, double *x, double *y, double *z, double *info) {
    QpOpts o; qp_default_opts(&o); o.verbose = 0;
    int nt = 0;
    for (int i = 0; i < nopt; i++) {
        const char *a = opts[i];
        if (!a) continue;
        if (!strcmp(a, "-qpas")) o.method = QP_METHOD_ACTIVESET;
        else if (!strcmp(a, "-qpipm")) o.method = QP_METHOD_IPM;
        else if (!strcmp(a, "-qpauto")) o.method = QP_METHOD_AUTO;
        else if (!strcmp(a, "-q")) o.verbose = 0;
        else if (!strcmp(a, "-tol") && i + 1 < nopt) o.tol = atof(opts[++i]);
        else if (!strcmp(a, "-maxit") && i + 1 < nopt) o.maxit = atoi(opts[++i]);
        else if (!strcmp(a, "-timelimit") && i + 1 < nopt) o.timelimit = atof(opts[++i]);
        else if (!strcmp(a, "-threads") && i + 1 < nopt) nt = atoi(opts[++i]);
        else if (!strcmp(a, "-caller") && i + 1 < nopt) ++i;
        else return -2;
    }
#if defined(__x86_64__) || defined(__i386__)
    const unsigned int csr = _mm_getcsr();
#endif
    /* (one thread unless -threads asks for more: the callers are small QPs, which the tiny-problem rule of the general route
     *  gives one thread too; the method's results are the same at any count) */
    const int nt0 = omp_get_max_threads();
    omp_set_num_threads(nt > 0 ? nt : 1);
#if defined(BRISK_LIBRARY) && !defined(BRISK_NO_IO_MACROS)
    qp_printf = brisk_printf;
#endif
    qcqp_socp = qcqp_socp_cli;
    QpRes R; memset(&R, 0, sizeof R);
    g_brisk_jmp_on = 1;
    const int j = setjmp(g_brisk_jmp);
    volatile int rc = 0;
    if (j == 0) {
        const double t0 = wtime();
        qp_solve(P, &o, &R);
        const int st = R.status;
        info[0] = st == QP_OPTIMAL ? 0 : st == QP_PRIMAL_INFEASIBLE ? 1 : st == QP_DUAL_INFEASIBLE ? 2 : st == QP_MAXIT ? 3 : st == QP_REDUCED ? 5 : st == QP_TIME ? 6 : st == QP_INPUT ? -1 : 4;
        info[1] = R.iter; info[2] = R.obj; info[3] = R.pres; info[4] = R.dres; info[5] = R.gap; info[6] = wtime() - t0;
        info[7] = R.method == QP_METHOD_ACTIVESET ? 1 : 2;
        if (R.x) memcpy(x, R.x, sizeof(double) * (size_t)P->n); else memset(x, 0, sizeof(double) * (size_t)P->n);
        if (R.y) memcpy(y, R.y, sizeof(double) * (size_t)P->m); else memset(y, 0, sizeof(double) * (size_t)P->m);
        if (R.z) memcpy(z, R.z, sizeof(double) * (size_t)P->n); else memset(z, 0, sizeof(double) * (size_t)P->n);
        rc = st == QP_OPTIMAL || st == QP_REDUCED ? 0 : 1;
    } else rc = j == 1000 ? -1000 : -j;
    g_brisk_jmp_on = 0;
    qp_res_free(&R);
    omp_set_num_threads(nt0);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(csr);
#endif
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
    /* 4.42: the thread count of the host process is restored after the call, also when the solve left through
     * brisk_exit (27 October 2026: read before the setjmp). 31 October 2026 (ISSUES 727): also after an explicit
     * -threads k, which is for this call only, and the call starts from the automatic count (nt_scope_begin) */
    volatile int nt_saved = nt_scope_begin(argc, argv);
    g_brisk_jmp_on = 1;
    volatile int rc;
    volatile int log_brief0 = g_log_brief, log_user0 = g_log_user;   /* (ISSUES 725: also after a solve left through brisk_exit) */
    const int j = setjmp(g_brisk_jmp);
    if (j == 0) rc = brisk_main(argc, argv);
    else rc = j == 1000 ? -1000 : -j;
    g_log_brief = log_brief0; g_log_user = log_user0;
    nt_scope_end(nt_saved);
    g_brisk_jmp_on = 0;
    g_brisk_result = NULL;
    caller_leave(av_caller);
    if (g_caller_depth == 0) brisk_threads_restore();     /* and its OpenBLAS thread count (not for a nested call) */
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(csr);
#endif
    return rc;
}
#endif
