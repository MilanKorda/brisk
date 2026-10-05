/* capi.c - BRISK 4.37: a flat C interface for foreign-function callers (Julia's ccall,
 * Python's ctypes, ...), built into the shared library libbrisk (make libbrisk).
 *
 * Only plain C types cross the boundary: the problem is the numbers of an SDPA sparse file
 * (brisk_solve_data) or a file name (brisk_solve_file), options are command-line strings,
 * and the result is an opaque handle read through accessor functions, so a caller never
 * depends on the layout of BriskResult.
 *
 *   BriskResult *r = brisk_result_new();
 *   const char *opts[] = { "-acc", "high" };
 *   int rc = brisk_solve_data(m, nblk, bs, c, nnz, mat, blk, i, j, v, 2, opts, r);
 *   brisk_result_status(r), brisk_result_y(r), brisk_result_X(r, k), ...
 *   brisk_result_delete(r);
 *
 * Conventions are those of the command line (brisk.h): (P) min <C,X> s.t. <A_i,X> = b_i,
 * X in K; (D) max b'y s.t. C - sum y_i A_i = Z in K, with C = -F0, A_i = F_i, b = c. The SDPA
 * primal variable is x = -y. X and Z blocks are n x n column-major (SDP) or n (LP).
 *
 * Not reentrant: the solver has process-wide state (options, the print hook, the exit jump).
 * Callers must serialize solves (the Julia package holds a lock). */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BRISK_NO_IO_MACROS
#include "brisk.h"
#include "sedumi.h"

#if defined(__GNUC__)
#define BRISK_API __attribute__((visibility("default")))
#else
#define BRISK_API
#endif

/* ---- output ------------------------------------------------------------------------ */
static int g_out_mode = 0;                                  /* 0 stdout/stderr, 1 silent (errors only), 2 callback */
static int (*g_out_cb)(const char *s, int is_err) = NULL;
static pthread_t g_caller;                                  /* the thread inside brisk_solve_* */

static int capi_print(const char *s, int is_err) {
    /* the callback only on the caller's thread (a runtime such as Julia's must not be
     * entered from a solver worker thread); anything else goes to stdout/stderr */
    if (g_out_mode == 2 && g_out_cb && pthread_equal(pthread_self(), g_caller)) return g_out_cb(s, is_err);
    if (g_out_mode == 1 && !is_err) return 0;
    FILE *f = is_err ? stderr : stdout;
    fputs(s, f);
    return 0;
}

/* mode 0: printed output to the C stdout/stderr (flushed after each solve); 1: nothing but
 * error messages; 2: every piece of output to cb(s, is_err), called only from the thread
 * that called brisk_solve_* (output from any other thread goes to stdout/stderr) */
BRISK_API void brisk_set_output(int mode, int (*cb)(const char *s, int is_err)) {
    g_out_mode = mode;
    g_out_cb = cb;
}

BRISK_API const char *brisk_version(void) { return BRISK_VERSION; }

/* ---- results ----------------------------------------------------------------------- */
BRISK_API BriskResult *brisk_result_new(void) {
    BriskResult *r = calloc(1, sizeof(BriskResult));
    if (r) r->status = -1;
    return r;
}
BRISK_API void brisk_result_delete(BriskResult *r) {
    if (!r) return;
    brisk_result_free(r);
    free(r);
}
BRISK_API int brisk_result_status(const BriskResult *r) { return r ? r->status : -1; }
BRISK_API const char *brisk_result_status_str(const BriskResult *r) { return r && r->status >= 0 ? r->status_str : "NOT SOLVED"; }
BRISK_API int brisk_result_exit_code(const BriskResult *r) { return r ? r->exit_code : -1; }
BRISK_API int brisk_result_iterations(const BriskResult *r) { return r ? r->iters : 0; }
BRISK_API int brisk_result_m(const BriskResult *r) { return r ? r->m : 0; }
BRISK_API int brisk_result_nblk(const BriskResult *r) { return r ? r->nblk : 0; }
BRISK_API int brisk_result_blocksize(const BriskResult *r, int k) { return r && r->bs && k >= 0 && k < r->nblk ? r->bs[k] : 0; }
BRISK_API double brisk_result_pobj(const BriskResult *r) { return r ? r->pobj : 0; }
BRISK_API double brisk_result_dobj(const BriskResult *r) { return r ? r->dobj : 0; }
BRISK_API double brisk_result_time(const BriskResult *r) { return r ? r->time : 0; }
BRISK_API int brisk_result_have_x(const BriskResult *r) { return r ? r->have_x : 0; }
/* DIMACS errors 1..6 on the data as given (err[0..5]) */
BRISK_API void brisk_result_dimacs(const BriskResult *r, double *err) {
    for (int e = 0; e < 6; e++) err[e] = r ? r->err[e + 1] : 0;
}
/* 4.37 bound mode (-bound p|d): out = { value, resid, lammin, valid, certified, rigorous }
 * (6 doubles; rigorous: the -certify bound, NaN when not run); returns the side (0 none,
 * 1 p: an upper bound <C,X>, 2 d: a lower bound b'y) */
BRISK_API int brisk_result_bound(const BriskResult *r, double *out) {
    if (!r) return 0;
    if (out) { out[0] = r->bound_value; out[1] = r->bound_resid; out[2] = r->bound_lammin; out[3] = r->bound_valid; out[4] = r->bound_certified; out[5] = r->bound_rigorous; }
    return r->bound_side;
}
BRISK_API int brisk_result_bound_side(const BriskResult *r) { return r ? r->bound_side : 0; }
/* 4.37: re-solves run and their time (s) */
BRISK_API int brisk_result_resolves(const BriskResult *r, double *t) { if (t) *t = r ? r->t_resolves : 0; return r ? r->n_resolves : 0; }
/* 4.37: the likely cause of a result that misses the tolerance ("" when none) */
BRISK_API const char *brisk_result_cause(const BriskResult *r) { return r ? r->cause : ""; }

/* a problem in SeDuMi format (brisk_solve_sedumi, brisk_solve_file on a MAT-file): n, x (n) and
 * z = c - A'y (n); y is brisk_result_y */
BRISK_API int brisk_result_sn(const BriskResult *r) { return r ? r->sn : 0; }
BRISK_API const double *brisk_result_sx(const BriskResult *r) { return r && r->have_x ? r->sx : NULL; }
BRISK_API const double *brisk_result_sz(const BriskResult *r) { return r ? r->sz : NULL; }

/* pointers into the result (valid until brisk_result_delete or the next solve with r):
 * y (m), X and Z block k (0-based; n*n column-major or n for LP); NULL when absent */
BRISK_API const double *brisk_result_y(const BriskResult *r) { return r ? r->y : NULL; }
BRISK_API const double *brisk_result_X(const BriskResult *r, int k) {
    return r && r->have_x && r->X && k >= 0 && k < r->nblk ? r->X[k] : NULL;
}
BRISK_API const double *brisk_result_Z(const BriskResult *r, int k) {
    return r && r->Z && k >= 0 && k < r->nblk ? r->Z[k] : NULL;
}

/* ---- solving ----------------------------------------------------------------------- */
static char **make_argv(const char *first, int nopt, const char *const *opts, int *argc) {
    char **av = malloc(sizeof(char *) * ((size_t)(nopt > 0 ? nopt : 0) + 3));
    if (!av) return NULL;
    int n = 0;
    av[n++] = "brisk";
    if (first) av[n++] = (char *)first;
    for (int i = 0; i < nopt; i++) av[n++] = (char *)opts[i];
    av[n] = NULL;
    *argc = n;
    return av;
}

static void finish(void) { fflush(stdout); fflush(stderr); brisk_print_hook = NULL; }

/* 4.37: stop the running solve as if its time limit were reached now: the solve returns the
 * current iterate with status TIME LIMIT (6) and brisk_interrupted() = 1. Async-signal-safe
 * and callable from any thread (a signal handler, a watcher thread); the flag is cleared when
 * the next brisk_solve_* starts. */
BRISK_API void brisk_interrupt(void) { brisk_stop_flag = 1; }
BRISK_API int brisk_interrupted(void) { return brisk_stop_flag != 0; }

/* The return value is the command line's exit code (0 optimal, 10 reduced accuracy, 11/12
 * primal/dual infeasible, 13 iteration limit, 14 numerical difficulties, 15 time limit;
 * 1 bad options, 2 unreadable problem) or, when negative, an abort (out of memory, an
 * invalid option found late): the result is then empty. */
BRISK_API int brisk_solve_data(int m, int nblk, const int *bs, const double *c,
                               int64_t nnz, const int *mat, const int *blk, const int *i, const int *j, const double *v,
                               int nopt, const char *const *opts, BriskResult *res) {
    if (!res) return -1;
    brisk_result_free(res);
    if (nnz < 0) return 2;
    BriskData d = { m, nblk, bs, c, (size_t)nnz, mat, blk, i, j, v };
    int argc;
    char **av = make_argv(NULL, nopt, opts, &argc);
    if (!av) return -1;
    g_caller = pthread_self();
    brisk_stop_flag = 0;
    brisk_print_hook = capi_print;
    const int rc = brisk_run_data(&d, argc, av, res);
    finish();
    free(av);
    return rc;
}

BRISK_API int brisk_solve_file(const char *fname, int nopt, const char *const *opts, BriskResult *res) {
    if (!res || !fname) return -1;
    brisk_result_free(res);
    int argc;
    char **av = make_argv(fname, nopt, opts, &argc);
    if (!av) return -1;
    g_caller = pthread_self();
    brisk_stop_flag = 0;
    brisk_print_hook = capi_print;
    const int rc = brisk_run(argc, av, res);
    finish();
    free(av);
    return rc;
}

/* A problem in SeDuMi format:  min c'x  s.t.  A x = b,  x in K;  max b'y  s.t.  c - A'y = z in K*.
 * A is m x n by columns (Ap: n + 1 column starts, Ai: rows, 0-based, Ax); K = nf free variables,
 * nl nonnegative ones, second-order cones q[0..nq-1] (x0 >= |x(1:)|), rotated cones r[0..nr-1]
 * (2 x0 x1 >= |x(2:)|^2), semidefinite blocks s[0..ns-1] (each as its d*d entries by columns),
 * in this order. Without semidefinite blocks the problem goes to the cone solver (socp.c), with
 * them to the semidefinite solver. The result: brisk_result_sx (x), brisk_result_y, brisk_result_sz
 * (z), status, values (pobj = c'x, dobj = b'y) and errors as for the other calls. */
BRISK_API int brisk_solve_sedumi(int m, int n, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c,
                                 int nf, int nl, int nq, const int *q, int nr, const int *r, int ns, const int *s,
                                 int nopt, const char *const *opts, BriskResult *res) {
    if (!res) return -1;
    brisk_result_free(res);
    if (m < 0 || n < 0 || nf < 0 || nl < 0 || nq < 0 || nr < 0 || ns < 0) return 2;
    long tot = (long)nf + nl;
    for (int k = 0; k < nq; k++) { if (q[k] < 1) return 2; tot += q[k]; }
    for (int k = 0; k < nr; k++) { if (r[k] < 2) return 2; tot += r[k]; }
    for (int k = 0; k < ns; k++) { if (s[k] < 1) return 2; tot += (long)s[k] * s[k]; }
    if (tot != n) return 2;
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) if (Ai[p] < 0 || Ai[p] >= m) return 2;
    SedumiProb P = { m, n, (int *)Ap, (int *)Ai, (double *)Ax, (double *)b, (double *)c, nf, nl, nq, (int *)q, nr, (int *)r, ns, (int *)s };
    SedumiRes R;
    g_caller = pthread_self();
    brisk_stop_flag = 0;
    brisk_print_hook = capi_print;
    const int rc = brisk_run_sedumi(&P, nopt, (char **)opts, &R);
    finish();
    if (R.status >= 0) {
        res->status = R.status; snprintf(res->status_str, sizeof res->status_str, "%s", R.status_str);
        res->exit_code = rc; res->iters = R.iters; res->m = m; res->pobj = R.pobj; res->dobj = R.dobj; res->time = R.time;
        for (int i = 0; i < 6; i++) res->err[i + 1] = R.err[i];
        res->y = R.y; res->sn = n; res->sx = R.x; res->sz = R.z; res->have_x = ns == 0 || R.have_x;
        R.x = R.y = R.z = NULL;
    }
    sedumi_result_free(&R);
    return rc;
}
