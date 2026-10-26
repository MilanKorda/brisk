#include <unistd.h>
/* qpmain.c - the command line of the QP solver (the SOCP developer, 22 October 2026):
 *     brisk_qp file.qps [-ipm | -as | -auto] [-tol t] [-maxit k] [-v | -vv | -q] [-threads k] [-timelimit s] [-x file]
 * reads a QPS file (MPS with a quadratic objective), solves it and prints the status, the objective, the errors on
 * the problem as read and the total time. */
#include "qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
#ifdef __linux__
    /* (23 October 2026) OpenBLAS single-threaded from its start, as brisk's own main does (main.c): with its threads the
     * bits depended on the cores the process could see, even at -threads 1 (QPILOTNO, QP: 42 iterations on one core, 43
     * on two); the environment is read when the library loads, hence the re-exec */
    if (!getenv("OPENBLAS_NUM_THREADS") && !getenv("BRISK_BLASMT") && !getenv("BRISK_NOREEXEC")) {
        setenv("OPENBLAS_NUM_THREADS", "1", 1); setenv("BRISK_NOREEXEC", "1", 1);
        execv("/proc/self/exe", argv);
    }
#endif
    const double t0 = now();
    if (argc < 2) {
        printf("usage: brisk_qp file.qps [options]\n"
               "  solves min 1/2 x'Hx + c'x s.t. rl <= Ax <= ru, xl <= x <= xu (H positive semidefinite) from a QPS file\n"
               "  -auto           the active-set method for a small QP with H positive definite (its dense work 2n^3 +\n"
               "                  2n(nnz(A) + n + m) at most 4e6 flops), the interior-point method otherwise (default)\n"
               "  -ipm            the interior-point method\n"
               "  -as             the active-set method (the dual method of Goldfarb and Idnani)\n"
               "  -tol t          relative tolerance of the residuals and the gap (1e-8)\n"
               "  -maxit k        iterations\n"
               "  -v, -vv, -q     the iteration log, details, quiet (-vvv: each step's blocking variables and products)\n"
               "  -threads k      threads of the factorization\n"
               "  -timelimit s    seconds\n"
               "  -x file         write the solution (one value per line)\n"
               "  -w file         write the problem as read as a free-format QPS file with plain names, and stop\n"
               "  -wt file        write the problem as read as plain triplets (the comparison harness's input), and stop\n");
        return 1;
    }
    QpOpts o; qp_default_opts(&o);
    const char *fname = NULL, *xout = NULL, *wout = NULL, *wtout = NULL;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-ipm")) o.method = QP_METHOD_IPM;
        else if (!strcmp(argv[a], "-auto")) o.method = QP_METHOD_AUTO;
        else if (!strcmp(argv[a], "-as")) o.method = QP_METHOD_ACTIVESET;
        else if (!strcmp(argv[a], "-tol") && a + 1 < argc) o.tol = atof(argv[++a]);
        else if (!strcmp(argv[a], "-maxit") && a + 1 < argc) o.maxit = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-v")) o.verbose = 2;
        else if (!strcmp(argv[a], "-vv")) o.verbose = 3;
        else if (!strcmp(argv[a], "-vvv")) o.verbose = 4;
        else if (!strcmp(argv[a], "-q")) o.verbose = 0;
        else if (!strcmp(argv[a], "-threads") && a + 1 < argc) o.threads = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-timelimit") && a + 1 < argc) o.timelimit = atof(argv[++a]);
        else if (!strcmp(argv[a], "-x") && a + 1 < argc) xout = argv[++a];
        else if (!strcmp(argv[a], "-w") && a + 1 < argc) wout = argv[++a];
        else if (!strcmp(argv[a], "-wt") && a + 1 < argc) wtout = argv[++a];
        else if (argv[a][0] == '-') { fprintf(stderr, "brisk_qp: unknown option %s\n", argv[a]); return 1; }
        else fname = argv[a];
    }
#ifdef _OPENMP
    if (o.threads > 0) omp_set_num_threads(o.threads);
#endif
    QpProb P; char msg[512];
    const double tr = now();
    if (qp_read_qps(fname, &P, msg, sizeof msg)) { fprintf(stderr, "brisk_qp: %s\n", msg); return 1; }
    const double tread = now() - tr;
    if (wtout) { const int e = qp_write_text(&P, wtout); qp_prob_free(&P); if (e) fprintf(stderr, "brisk_qp: cannot write %s\n", wtout); return e; }
    if (wout) { const int e = qp_write_qps(&P, wout); qp_prob_free(&P); if (e) fprintf(stderr, "brisk_qp: cannot write %s\n", wout); return e; }
    if (o.verbose) printf("QPS problem %s (%s): n = %d, m = %d, nnz(A) = %d, nnz(H lower) = %d, read %.2fs\n", fname, P.name, P.n, P.m,
                          P.Ap[P.n], P.Hp ? P.Hp[P.n] : 0, tread);
    QpRes R;
    qp_solve(&P, &o, &R);
    printf("status: %s   iterations: %d   objective: %.12e\n", qp_status_name(R.status), R.iter, R.obj);
    printf("errors on the problem as read: primal %.1e, dual %.1e, gap %.1e\n", R.pres, R.dres, R.gap);
    printf("Time: %.3f s\n", now() - t0);
    if (xout) { FILE *f = fopen(xout, "w"); if (f) { for (int j = 0; j < P.n; j++) fprintf(f, "%.17g\n", R.x[j]); fclose(f); } }
    const int st = R.status;
    qp_res_free(&R); qp_prob_free(&P);
    return st == QP_OPTIMAL ? 0 : 2;
}
