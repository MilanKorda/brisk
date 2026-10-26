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
#include <math.h>
#define BRISK_NO_IO_MACROS
#include "brisk.h"
#include "sedumi.h"
#include "qp.h"
#include "socpmodel.h"
int brisk_run_qp(const QpProb *P, int argc, char **argv, BriskResult *res);

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
/* bound mode per side (-bound p|d|pd; side 1 p: an upper bound <C,X>, 2 d: a lower bound b'y):
 * out = { value, resid, lammin, valid, certified, rigorous } (6 doubles; only a certified one is a
 * bound, rigorous is NaN otherwise); returns 2 when the side was asked for and has a certificate,
 * 1 when it was asked for without one, 0 when it was not asked for */
BRISK_API int brisk_result_bound_of(const BriskResult *r, int side, double *out) {
    if (!r || side < 1 || side > 2) return 0;
    const int s = side - 1;
    if (out) { out[0] = r->bnd[s].value; out[1] = r->bnd[s].resid; out[2] = r->bnd[s].lammin; out[3] = r->bnd[s].valid; out[4] = r->bnd[s].certified; out[5] = r->bnd[s].rigorous; }
    return r->bnd[s].asked ? 1 + r->bnd[s].have : 0;
}
BRISK_API const char *brisk_result_bound_src(const BriskResult *r, int side) { return r && side >= 1 && side <= 2 ? r->bnd[side - 1].src : ""; }
/* the certificates (pointers into the result): block k of the X of side p (as X; NULL when
 * absent), the y of side d (m), and for a problem in SeDuMi format the x of side p (sn) */
BRISK_API const double *brisk_result_bound_X(const BriskResult *r, int k) {
    return r && r->bnd[0].X && k >= 0 && k < r->nblk ? r->bnd[0].X[k] : NULL;
}
BRISK_API const double *brisk_result_bound_y(const BriskResult *r) { return r ? r->bnd[1].y : NULL; }
BRISK_API const double *brisk_result_bound_sx(const BriskResult *r) { return r ? r->bnd[0].sx : NULL; }
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
 * the next brisk_solve_* starts.
 * (24 October 2026, the SOCP developer; ISSUES 272) A caller that starts the solve in another thread clears the flag
 * itself before it hands the solve over (brisk_clear_interrupt) and asks the entry points to leave it alone
 * (brisk_set_interrupt_owner(1)): otherwise an interrupt that arrives after the hand-off and before the entry point
 * runs is cleared by it, and the solve runs to its end (Python's test of Ctrl-C hung so, twice on 24 October). */
static int g_stop_owner = 0;          /* 1: the caller clears the flag (brisk_clear_interrupt), the entry points do not */
BRISK_API void brisk_interrupt(void) { brisk_stop_flag = 1; }
BRISK_API int brisk_interrupted(void) { return brisk_stop_flag != 0; }
BRISK_API void brisk_clear_interrupt(void) { brisk_stop_flag = 0; }
BRISK_API void brisk_set_interrupt_owner(int caller) { g_stop_owner = caller != 0; }
#define STOP_RESET() do { if (!g_stop_owner) brisk_stop_flag = 0; } while (0)

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
    STOP_RESET();
    brisk_print_hook = capi_print;
    const int rc = brisk_run_data(&d, argc, av, res);
    finish();
    free(av);
    return rc;
}

/* (22 October 2026) A convex QP:  min 1/2 x'Hx + c'x + c0  s.t.  rl <= A x <= ru,  xl <= x <= xu.
 * H: its lower triangle by columns (Hp: n + 1 column starts, Hi: rows >= the column, 0-based, Hx; Hp NULL: H = 0); A: m x n
 * by columns (Ap, Ai, Ax); a bound of magnitude 1e20 or more (or infinite) is absent. Options as on the command line for a
 * QP file (-qpas -qpipm -tol -maxit -timelimit -threads -q -v -vv; by default the active-set method for a small QP
 * with H positive definite, the interior-point method otherwise). The result: brisk_result_sx (x, n), brisk_result_y
 * (the rows' multipliers, m), brisk_result_sz (the bounds', n), status (0 optimal, 1 primal infeasible, 2 dual
 * infeasible, 3 the iteration limit, 4 numerical difficulties, 5 reduced accuracy, 6 the time limit), pobj (the
 * objective), brisk_result_dimacs: [0] the primal, [2] the dual, [4] the gap error on the problem as given. The
 * multipliers satisfy H x + c - A'y - z = 0, y_i > 0 at rl and < 0 at ru, z_j > 0 at xl and < 0 at xu. */
BRISK_API int brisk_solve_qp(int n, int m, const int *Hp, const int *Hi, const double *Hx, const double *c, double c0,
                             const int *Ap, const int *Ai, const double *Ax, const double *rl, const double *ru,
                             const double *xl, const double *xu, int nopt, const char *const *opts, BriskResult *res) {
    if (!res) return -1;
    brisk_result_free(res);
    if (n < 0 || m < 0 || !Ap || !c) return 2;
    QpProb P; memset(&P, 0, sizeof P);
    P.n = n; P.m = m; P.Hp = (int *)Hp; P.Hi = (int *)Hi; P.Hx = (double *)Hx; P.Ap = (int *)Ap; P.Ai = (int *)Ai; P.Ax = (double *)Ax;
    P.c = (double *)c; P.c0 = c0;
    double *b = malloc(sizeof(double) * (2 * (size_t)m + 2 * (size_t)n + 1));
    if (!b) return -1;
    P.rl = b; P.ru = b + m; P.xl = b + 2 * m; P.xu = b + 2 * m + n;
    for (int i = 0; i < m; i++) { const double l = rl ? rl[i] : -QP_INF, u = ru ? ru[i] : QP_INF; P.rl[i] = l <= -QP_INF ? -QP_INF : l; P.ru[i] = u >= QP_INF ? QP_INF : u; }
    for (int j = 0; j < n; j++) { const double l = xl ? xl[j] : -QP_INF, u = xu ? xu[j] : QP_INF; P.xl[j] = l <= -QP_INF ? -QP_INF : l; P.xu[j] = u >= QP_INF ? QP_INF : u; }
    int argc;
    char **av = make_argv(NULL, nopt, opts, &argc);
    if (!av) { free(b); return -1; }
    g_caller = pthread_self();
    STOP_RESET();
    brisk_print_hook = capi_print;
    const int rc = brisk_run_qp(&P, argc, av, res);
    finish();
    free(av); free(b);
    return rc;
}

/* (23 October 2026, the SOCP developer) H's lower triangle for brisk_solve_qp from a CSC matrix with sorted rows and no
 * duplicates in each column (n columns: Hp, Hi, Hx): H symmetric (both triangles, the strictly lower entries equal to the
 * strictly upper ones transposed within 1e-14 of max(1, max |H|)), lower or upper triangular. The lower triangle is written
 * to Lp (n + 1), Li and Lx (room for Hp[n] entries each); returns its number of entries, or -1 when H is none of the three.
 * The interfaces' check in O(nnz) (Python's numpy version took 37 microseconds on a QP of 16 variables). */
BRISK_API long brisk_qp_lower(int n, const int *Hp, const int *Hi, const double *Hx, int *Lp, int *Li, double *Lx) {
    if (n < 0 || !Hp || !Lp) return -1;
    long nlo = 0, nup = 0; double hmax = 0;
    for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) { const int i = Hi[p]; if (i > j) nlo++; else if (i < j) nup++; hmax = fmax(hmax, fabs(Hx[p])); }
    Lp[0] = 0;
    if (nup == 0) {                                   /* lower triangular: as it is */
        for (int j = 0; j < n; j++) Lp[j + 1] = Hp[j + 1];
        for (int p = 0; p < Hp[n]; p++) { Li[p] = Hi[p]; Lx[p] = Hx[p]; }
        return Hp[n];
    }
    /* the transpose of the upper triangle with the diagonal (column i of it: the entries (i, j) of H with i <= j, in
     * increasing j, since the columns are read in order) */
    for (int j = 0; j <= n; j++) Lp[j] = 0;
    for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hi[p] <= j) Lp[Hi[p] + 1]++;
    for (int j = 0; j < n; j++) Lp[j + 1] += Lp[j];
    int *f = malloc(sizeof(int) * ((size_t)n + 1)); if (!f) return -1;
    for (int j = 0; j < n; j++) f[j] = Lp[j];
    for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hi[p] <= j) { const int t = f[Hi[p]]++; Li[t] = j; Lx[t] = Hx[p]; }
    free(f);
    if (nlo == 0) return Lp[n];                       /* upper triangular: its transpose */
    /* symmetric: the strictly lower part of each column against the transposed upper part's (which holds the diagonal
     * first, then the rows below it) */
    const double tol = 1e-14 * fmax(1.0, hmax);
    for (int j = 0; j < n; j++) {
        int p = Hp[j]; while (p < Hp[j + 1] && Hi[p] <= j) p++;      /* (the strictly lower entries of column j) */
        int q = Lp[j]; if (q < Lp[j + 1] && Li[q] == j) q++;          /* (past the diagonal) */
        if (Hp[j + 1] - p != Lp[j + 1] - q) return -1;
        for (; p < Hp[j + 1]; p++, q++) if (Hi[p] != Li[q] || fabs(Hx[p] - Lx[q]) > tol) return -1;
    }
    return Lp[n];
}

/* (22 October 2026) the QP solver object for warm starts (mixed-integer QPs by branch and bound): created on the data of
 * brisk_solve_qp, then bounds, row bounds and the cost changed and the problem re-solved from the previous active set.
 * Options: -qpas (the active-set method, the default here: it carries the active set from solve to solve), -qpipm (each
 * solve cold), -tol, -maxit, -timelimit, -q, -v, -vv. NULL on bad data or an unknown option. */
void brisk_qp_fill_result(BriskResult *res, int n, int m, QpRes *R, double t);
typedef struct { QpSolver *S; int n, m; } BriskQp;
static double qp_bnd(const double v, const int lo) { return lo ? (v <= -QP_INF ? -QP_INF : v) : (v >= QP_INF ? QP_INF : v); }
BRISK_API void *brisk_qp_create(int n, int m, const int *Hp, const int *Hi, const double *Hx, const double *c, double c0,
                                const int *Ap, const int *Ai, const double *Ax, const double *rl, const double *ru,
                                const double *xl, const double *xu, int nopt, const char *const *opts) {
    if (n < 0 || m < 0 || !Ap || !c) return NULL;
    QpOpts o; qp_default_opts(&o); o.method = QP_METHOD_ACTIVESET; o.verbose = 1;
    for (int i = 0; i < nopt; i++) {
        const char *a = opts[i];
        if (!strcmp(a, "-qpas")) o.method = QP_METHOD_ACTIVESET;
        else if (!strcmp(a, "-qpipm")) o.method = QP_METHOD_IPM;
        else if (!strcmp(a, "-tol") && i + 1 < nopt) o.tol = atof(opts[++i]);
        else if (!strcmp(a, "-maxit") && i + 1 < nopt) o.maxit = atoi(opts[++i]);
        else if (!strcmp(a, "-timelimit") && i + 1 < nopt) o.timelimit = atof(opts[++i]);
        else if (!strcmp(a, "-q")) o.verbose = 0;
        else if (!strcmp(a, "-v")) o.verbose = 2;
        else if (!strcmp(a, "-vv")) o.verbose = 3;
        else return NULL;
    }
    QpProb P; memset(&P, 0, sizeof P);
    P.n = n; P.m = m; P.Hp = (int *)Hp; P.Hi = (int *)Hi; P.Hx = (double *)Hx; P.Ap = (int *)Ap; P.Ai = (int *)Ai; P.Ax = (double *)Ax;
    P.c = (double *)c; P.c0 = c0;
    double *b = malloc(sizeof(double) * (2 * (size_t)m + 2 * (size_t)n + 1));
    if (!b) return NULL;
    P.rl = b; P.ru = b + m; P.xl = b + 2 * m; P.xu = b + 2 * m + n;
    for (int i = 0; i < m; i++) { P.rl[i] = rl ? qp_bnd(rl[i], 1) : -QP_INF; P.ru[i] = ru ? qp_bnd(ru[i], 0) : QP_INF; }
    for (int j = 0; j < n; j++) { P.xl[j] = xl ? qp_bnd(xl[j], 1) : -QP_INF; P.xu[j] = xu ? qp_bnd(xu[j], 0) : QP_INF; }
    BriskQp *h = malloc(sizeof *h);
    if (!h) { free(b); return NULL; }
    h->S = qp_create(&P, &o); h->n = n; h->m = m;           /* (the data copied) */
    free(b);
    if (!h->S) { free(h); return NULL; }
    return h;
}
BRISK_API int brisk_qp_set_var_bounds(void *h, int j, double lo, double hi) { return h ? qp_set_var_bounds(((BriskQp *)h)->S, j, qp_bnd(lo, 1), qp_bnd(hi, 0)) : 1; }
BRISK_API int brisk_qp_set_row_bounds(void *h, int i, double lo, double hi) { return h ? qp_set_row_bounds(((BriskQp *)h)->S, i, qp_bnd(lo, 1), qp_bnd(hi, 0)) : 1; }
BRISK_API int brisk_qp_set_cost(void *h, const double *c) { return h && c ? qp_set_cost(((BriskQp *)h)->S, c) : 1; }
/* the re-solve: the result as brisk_solve_qp's (the status code returned as well) */
BRISK_API int brisk_qp_resolve(void *hv, BriskResult *res) {
    if (!hv || !res) return -1;
    BriskQp *h = hv;
    brisk_result_free(res);
    g_caller = pthread_self();
    STOP_RESET();
    brisk_print_hook = capi_print;
#if !defined(BRISK_NO_IO_MACROS)
    qp_printf = brisk_printf;
#endif
    const double t0 = wtime();
    QpRes R; memset(&R, 0, sizeof R);
    qp_resolve(h->S, &R);
    brisk_qp_fill_result(res, h->n, h->m, &R, wtime() - t0);
    qp_res_free(&R);
    finish();
    return res->status;
}
BRISK_API void brisk_qp_destroy(void *h) { if (h) { qp_destroy(((BriskQp *)h)->S); free(h); } }

/* (23 October 2026, the SOCP developer) the SOCP model for warm starts (mixed-integer SOCPs by branch and bound): SeDuMi's
 * primal form min c'x, A x = b, x in K (nf free, nl nonnegative, second-order cones q, rotated cones r, in this order), kept
 * for re-solves after bound changes on any column; each solve after an optimal one starts from it (socpmodel.c). Options:
 * -tol, -maxit, -timelimit, -q, -v; NULL on bad data or an unknown option. The result: brisk_result_sx (x), _y, _sz (z) over
 * the problem as given, status as brisk_solve_sedumi's (and 9: stopped by the cutoff) */
static void socp_fill_result(BriskResult *res, int n, int m, SocpRes *R, double t) {
    const int st = R->status;
    res->status = st == SOCP_OPTIMAL ? 0 : st == SOCP_PINF ? 1 : st == SOCP_DINF ? 2 : st == SOCP_MAXIT ? 3 : st == SOCP_REDUCED ? 5 : st == SOCP_TIME ? 6 : st == SOCP_CUTOFF ? 9 : 4;
    snprintf(res->status_str, sizeof res->status_str, "%s", st == SOCP_CUTOFF ? "STOPPED BY THE CUTOFF" : socp_status_str(st));
    res->exit_code = st == SOCP_OPTIMAL || st == SOCP_REDUCED ? 0 : 1; res->iters = R->iters; res->m = m; res->pobj = R->pobj; res->dobj = R->dobj; res->time = t;
    for (int k = 0; k < 4 && k + 1 < 7; k++) res->err[k + 1] = R->err[k];
    res->sn = n; res->have_x = R->x != NULL; res->sx = R->x; R->x = NULL; res->y = R->y; R->y = NULL; res->sz = R->z; R->z = NULL;
}
typedef struct { SocpModel *M; int n, m; } BriskSocp;
BRISK_API void *brisk_socp_create(int m, int n, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c,
                                  int nf, int nl, int nq, const int *q, int nr, const int *r, int nopt, const char *const *opts) {
    if (m < 0 || n < 1 || !Ap || !c || nf < 0 || nl < 0 || nq < 0 || nr < 0) return NULL;
    long tot = (long)nf + nl; for (int k = 0; k < nq; k++) tot += q[k]; for (int k = 0; k < nr; k++) tot += r[k];
    if (tot != n) return NULL;
    SocpOpts o; socp_default_opts(&o); o.verbose = 0;
    for (int i = 0; i < nopt; i++) {
        const char *a = opts[i];
        if (!strcmp(a, "-tol") && i + 1 < nopt) o.tol = atof(opts[++i]);
        else if (!strcmp(a, "-maxit") && i + 1 < nopt) o.maxit = atoi(opts[++i]);
        else if (!strcmp(a, "-timelimit") && i + 1 < nopt) o.timelimit = atof(opts[++i]);
        else if (!strcmp(a, "-q")) o.verbose = 0;
        else if (!strcmp(a, "-v")) o.verbose = 1;
        else return NULL;
    }
    SocpProb P; memset(&P, 0, sizeof P);
    P.m = m; P.n = n; P.Ap = Ap; P.Ai = Ai; P.Ax = Ax; P.b = b; P.c = c; P.nf = nf; P.nl = nl; P.nq = nq; P.q = q; P.nr = nr; P.r = r;
    BriskSocp *h = malloc(sizeof *h);
    if (!h) return NULL;
    h->M = socpm_create(&P, &o); h->n = n; h->m = m;          /* (the data copied) */
    if (!h->M) { free(h); return NULL; }
    return h;
}
BRISK_API int brisk_socp_set_var_bounds(void *h, int j, double lo, double hi) { return h ? socpm_set_var_bounds(((BriskSocp *)h)->M, j, lo, hi) : 1; }
BRISK_API int brisk_socp_set_cost(void *h, const double *c) { return h && c ? socpm_set_cost(((BriskSocp *)h)->M, c) : 1; }
BRISK_API void brisk_socp_set_cutoff(void *h, double v) { if (h) socpm_set_cutoff(((BriskSocp *)h)->M, v); }
BRISK_API int brisk_socp_resolve(void *hv, BriskResult *res) {
    if (!hv || !res) return -1;
    BriskSocp *h = hv;
    brisk_result_free(res);
    g_caller = pthread_self();
    STOP_RESET();
    brisk_print_hook = capi_print;
    const double t0 = wtime();
    SocpRes R; memset(&R, 0, sizeof R);
    socpm_solve(h->M, &R);
    socp_fill_result(res, h->n, h->m, &R, wtime() - t0);
    socp_result_free(&R);
    finish();
    return res->status;
}
BRISK_API void brisk_socp_destroy(void *h) { if (h) { socpm_destroy(((BriskSocp *)h)->M); free(h); } }
/* the model's warm-start point saved and put back (for example the parent's before each child; NULL: none kept yet) */
BRISK_API void *brisk_socp_save_point(void *h) { return h ? (void *)socpm_save_point(((BriskSocp *)h)->M) : NULL; }
BRISK_API void brisk_socp_restore_point(void *h, void *p) { if (h && p) socpm_restore_point(((BriskSocp *)h)->M, (SocpmPoint *)p); }
BRISK_API void brisk_socp_point_free(void *p) { socpm_point_free((SocpmPoint *)p); }

BRISK_API int brisk_solve_file(const char *fname, int nopt, const char *const *opts, BriskResult *res) {
    if (!res || !fname) return -1;
    brisk_result_free(res);
    int argc;
    char **av = make_argv(fname, nopt, opts, &argc);
    if (!av) return -1;
    g_caller = pthread_self();
    STOP_RESET();
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
 * in this order. Without semidefinite blocks the problem goes to the SOCP solver (socp.c), with
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
    STOP_RESET();
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
