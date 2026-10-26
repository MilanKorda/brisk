/* qpipm.c - the QP solver's interior-point method (the SOCP developer, 22 October 2026).
 *
 * Primal-dual infeasible-start method with Mehrotra's predictor-corrector for
 *     min 1/2 x'Hx + c'x   s.t.  rl <= A x <= ru,  xl <= x <= xu
 * Each finite bound has its own slack and multiplier: x - tl = xl, x + tu = xu, and for an inequality row with s = A x,
 * s - wl = rl, s + wu = ru; equality rows (rl = ru) keep A x = b. The Newton system reduces to the quasi-definite
 *     [ -(H + Dx + rho I)   A'            ] [dx]   [ -R1 ]
 *     [  A                  Ds^-1 + delta ] [dy] = [  R2 ]
 * Dx = zl/tl + zu/tu, Ds = vl/wl + vu/wu (0 rows of Ds^-1 for equalities), factored by BRISK's sparse signed Cholesky
 * (sparsechol.c: the signs prescribed, -1 on x, +1 on the rows) and refined against the matrix with the regularization
 * at 1e-14. Before the iteration: fixed variables substituted, rows without bounds dropped, Ruiz's equilibration of
 * [H A'; A 0] and a scaling of the cost. The measures of the end are computed on the problem as given (qp_errors). */
#include "qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
/* (25 October 2026) dsyrk for the dense assembly of the normal equations, with the build's BLAS prefix (brisk.h's BL) */
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define QP_CAT2(a, b) a##b
#define QP_CAT(a, b) QP_CAT2(a, b)
void QP_CAT(BLAS_PREFIX, dsyrk_)(const char *, const char *, const int *, const int *, const double *, const double *, const int *,
                                 const double *, double *, const int *);
#define qp_dsyrk QP_CAT(BLAS_PREFIX, dsyrk_)
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct SChol SChol;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_signs(SChol *S, const signed char *sgn_orig);
void   schol_set_dynpiv(SChol *S, double p);
void   schol_set_tinypiv(SChol *S, double t);
int    schol_ndyn(const SChol *S);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
void   schol_solve_many(const SChol *S, double *Bm, int nrhs, double *work);
SChol *schol_reanalyze_amalg(SChol *S, double f);
void   schol_set_inplace(SChol *S, int on);
int    schol_inplace(const SChol *S);
static SChol *qp_amalg(SChol *S, int nunk);
size_t schol_nnz(const SChol *S);
double schol_flops(const SChol *S);
void   schol_set_solve_seq(int on);
int    schol_solve_part(SChol *S, double *b, double *work, int nt);
/* (23 October 2026, the SOCP developer) the solve by independent subtrees (sparsechol.c: the same bits at any thread count),
 * a test: BRISK_QPSOLVEPART=1. Not the default: on the 25 QPs over a second at two threads 1.11 of the total time
 * (QPLIB_8500 2.9 with two more iterations, CONT-300 1.35; bench/qp_part/); the cause is not found (the default solve is
 * sequential here too); TRICKS 8 */
static int qp_part_nt(void) {
    static int v = -1; if (v < 0) { const char *e = getenv("BRISK_QPSOLVEPART"); v = e ? atoi(e) : 0; }
    if (!v) return 0;
#ifdef _OPENMP
    return omp_in_parallel() ? 0 : omp_get_max_threads();
#else
    return 1;
#endif
}
int    schol_set_amd(int mode);
void   schol_set_nd(int on);
void   schol_set_nd_budget(double factorizations);
double schol_flops(const SChol *S);
void qp_hmul(const QpProb *P, const double *x, double *y);

/* (23 October 2026, the SOCP developer) a debug switch read once: getenv scans the environment, and the refinement's switches
 * were read in every step (76 lookups a solve on a QP of 16 variables, 3 % of its instructions) */
static int env_on(const char *name, int *cache) { if (*cache < 0) *cache = getenv(name) != NULL; return *cache; }
static int g_refdbg = -1, g_nandbg = -1;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *xm(size_t s) { void *p = malloc(s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static void *xz(size_t s) { void *p = calloc(1, s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static int fin(double v) { return fabs(v) < QP_INF; }
static double nrmi(int n, const double *v) { double s = 0; for (int i = 0; i < n; i++) s = fmax(s, fabs(v[i])); return s; }

/* the reduced, scaled problem the iteration works on */
typedef struct {
    int n, m;                     /* columns (free and bounded, fixed removed), rows (with a finite bound) */
    int *Hp, *Hi; double *Hx;     /* lower triangle */
    int *Ap, *Ai; double *Ax;
    double *c;
    double *xl, *xu, *rl, *ru;
    char *hl, *hu;                /* finite lower / upper bound of each column */
    char *eq, *gl, *gu;           /* equality row; inequality row with a finite lower / upper bound */
    double *D, *E, cs, bs;        /* scaling: x = bs D x~, rows E, cost cs */
    int *col0, *row0;             /* the original index of each column / row */
    int ns, *srow, *scol; double *sa;   /* the singleton rows turned into bounds: row, its column, its entry */
    double *xlp, *xup;            /* the columns' bounds after them (original columns) */
} Red;

void   schol_zero(SChol *S);
void   schol_add(SChol *S, size_t r, size_t c, double v);
void   schol_pivots(const SChol *S, double *d);
static int g_qp_ray = 0;           /* (reduce: 2) the empty column of an unbounded direction, 1-based, signed */
static int g_qp_ndep = 0;          /* dependent equality rows dropped by reduce */
static const double *g_qp_rowprio = NULL;   /* (the second solve) the rows' priorities for the choice of the dropped rows */
static char *g_qp_dead = NULL;              /* the original rows reduce dropped as dependent (when not NULL) */

/* (23 October 2026) the equality rows that are linear combinations of others: nr rows (rp, rj, rv by rows over ncol columns)
 * with right-hand sides b. The method of the LP and SDP presolve (presolve.c, remove_dependent_rows, which the QP method's
 * binary does not link): the Gram matrix G = A A' factored by AMD's order with the tiny-pivot rule (a pivot below tol of
 * its diagonal replaced by a huge one), its flagged rows each checked in the space of the data, not through G (which
 * squares the condition): lambda from G lambda = A a_i with the flagged rows' pivots huge (their lambda 0), v = a_i -
 * A' lambda, a dependence when |v| <= 1e-9 |a_i|, its right-hand side then b_i = lambda' b to 1e-8 of the scale; a flagged
 * row that fails is kept (left out of the span the others are checked against, which can hide a dependence but not create
 * one). Budgets: forming G 2e9 multiply-adds, its factor 2e10 flops, the checks 1e10; past one, nothing is dropped.
 * dead[i] = 1 for the rows dropped; returns their number, *ninc the inconsistent ones (then nothing is dropped). */
/* prio (NULL: none): a priority per row; a dependent row is swapped with the kept row of highest priority in its combination
 * (its coefficient nonzero, so the kept rows span the same space) when that row's priority is higher, and the swapped-out
 * row is dropped instead; a row whose combination uses a row already swapped out keeps its place */
static int qp_dep_rows_core(int nr, int ncol, const int *rp, const int *rj, const double *rv, const double *b, double tol, char *dead, int *ninc, const double *prio) {
    *ninc = 0; for (int i = 0; i < nr; i++) dead[i] = 0;
    if (nr < 2) return 0;
    int *cp = xz(sizeof(int) * (ncol + 2));
    for (int p = 0; p < rp[nr]; p++) cp[rj[p] + 1]++;
    double work = 0; for (int j = 0; j < ncol; j++) work += (double)cp[j + 1] * cp[j + 1];
    /* (23 October 2026) the rows' Gram matrix G = A A' costs its pattern's pairs (the sum of the columns' squared lengths) to
     * build and an analysis and factorization on top: the check is made only while G is not much denser than A itself. With
     * the bound 2e9 alone, HUBER_200 (QP; 200 dense columns over its 20,000 equality rows) built a dense 20,000 x 20,000
     * pattern, 33 s and several GB before the flop cap below refused its factorization (without a memory cap the machine
     * restarted). Dense columns make G dense; the interior-point method keeps such rows (its regularization) */
    if (work > 2e9 || work > 50.0 * (double)rp[nr] + 1e6) { if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent rows: the Gram matrix's pattern %.3g pairs against %d entries: not formed]\n", work, rp[nr]); free(cp); return 0; }
    for (int j = 0; j < ncol; j++) cp[j + 1] += cp[j];
    int *ci = xm(sizeof(int) * ((size_t)rp[nr] + 1)), *f = xm(sizeof(int) * (ncol + 1)); double *cv = xm(sizeof(double) * ((size_t)rp[nr] + 1));
    memcpy(f, cp, sizeof(int) * ncol);
    for (int i = 0; i < nr; i++) for (int p = rp[i]; p < rp[i + 1]; p++) { const int t = f[rj[p]]++; ci[t] = i; cv[t] = rv[p]; }
    free(f);
    /* G's pattern (rows sharing a column) */
    int *mk = xm(sizeof(int) * (nr + 1)), *deg = xz(sizeof(int) * (nr + 1)); for (int i = 0; i < nr; i++) mk[i] = -1;
    long tot = 0;
    for (int i = 0; i < nr; i++) { mk[i] = i; for (int p = rp[i]; p < rp[i + 1]; p++) for (int q = cp[rj[p]]; q < cp[rj[p] + 1]; q++) { const int k = ci[q]; if (mk[k] != i) { mk[k] = i; deg[i]++; } } tot += deg[i]; }
    int **nbr = xm(sizeof(int *) * (nr + 1)), *pool = xm(sizeof(int) * ((size_t)tot + 1));
    { long o = 0; for (int i = 0; i < nr; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; mk[i] = -1; } }
    for (int i = 0; i < nr; i++) { mk[i] = i; for (int p = rp[i]; p < rp[i + 1]; p++) for (int q = cp[rj[p]]; q < cp[rj[p] + 1]; q++) { const int k = ci[q]; if (mk[k] != i) { mk[k] = i; nbr[i][deg[i]++] = k; } } }
    const int oamd = schol_set_amd(2);
    /* (24 October 2026, the SOCP developer) the fill cap from the flop bound below: a factor of N entries over nr columns
     * costs at least N^2 / nr flops (Cauchy-Schwarz on its column counts), so one within 2e10 flops has at most
     * sqrt(2e10 nr) entries. The cap 2^31 let the analysis allocate a factor it was to refuse: QFLOW_100000 (QP; a
     * network flow on a random graph, A A' its Laplacian) stopped with "out of memory" in the analysis */
    SChol *S = schol_analyze_adj(nr, deg, nbr, (size_t)(sqrt(2e10 * (double)nr) + nr));
    schol_set_amd(oamd);
    int removed = 0;
    double *acc = NULL, *d0 = NULL, *piv = NULL, *g = NULL, *u = NULL, *wk = NULL; char *flag = NULL;
    if (!S || schol_flops(S) > 2e10) { if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent rows: the Gram matrix's factor %s (%.3g flops)]\n", S ? "beyond 2e10 flops" : "not analysed", S ? schol_flops(S) : 0.0); goto out; }
    schol_zero(S);
    acc = xz(sizeof(double) * (nr + 1)); d0 = xm(sizeof(double) * (nr + 1));
    for (int i = 0; i < nr; i++) {
        for (int p = rp[i]; p < rp[i + 1]; p++) for (int q = cp[rj[p]]; q < cp[rj[p] + 1]; q++) acc[ci[q]] += rv[p] * cv[q];
        d0[i] = acc[i]; schol_add(S, (size_t)i, (size_t)i, acc[i]); acc[i] = 0;
        for (int c = 0; c < deg[i]; c++) { const int k = nbr[i][c]; if (k > i) schol_add(S, (size_t)k, (size_t)i, acc[k]); acc[k] = 0; }
    }
    schol_set_tinypiv(S, tol);
    if (schol_factor(S, 0.0)) goto out;
    piv = xm(sizeof(double) * (nr + 1)); schol_pivots(S, piv);
    flag = xz(nr + 1); int nf = 0;
    for (int i = 0; i < nr; i++) if (piv[i] >= 1e100 || !(d0[i] > 0)) { flag[i] = 1; nf++; }
    static double chk = -1; if (chk < 0) chk = getenv("BRISK_QPDEPCHK") ? atof(getenv("BRISK_QPDEPCHK")) : 1e10;   /* a test: the checks' budget */
    if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent rows: %d flagged, nnz(L) %.3g, the checks' estimate %.3g]\n", nf, (double)schol_nnz(S), (double)nf * 2.0 * (2.0 * rp[nr] + ncol + 2.0 * (double)schol_nnz(S)));
    const double per = 2.0 * (2.0 * rp[nr] + ncol + 2.0 * (double)schol_nnz(S));      /* (one check: two solves and the products) */
    static int grp = -1; if (grp < 0) { const char *e = getenv("BRISK_QPDEPGRP"); grp = e ? atoi(e) : 1; }
    if (nf == 0 || ((double)nf * per > chk && !(grp && 2.0 * per <= chk))) goto out;
    double bsc = 0; for (int i = 0; i < nr; i++) if (d0[i] > 0) bsc = fmax(bsc, fabs(b[i]) / sqrt(d0[i]));
    const double ctol = 1e-8 * (bsc > 0 ? bsc : 1.0);
    g = xm(sizeof(double) * (nr + 1)); u = xm(sizeof(double) * (ncol + 1)); wk = xm(sizeof(double) * (2 * (size_t)nr + 2));
    if ((double)nf * per > chk) {
        /* (23 October 2026, the SOCP developer; BRISK_QPDEPGRP=0: off) the checks of the flagged rows one by one exceed the
         * budget (QPLIB_8785, QP: 2,328 flagged against a factor of 5.8e6 entries, 5.4e10 flops against 1e10, so nothing was
         * dropped): they are checked as groups. A group G passes when one combination a_w = sum_G w_i a_i with fixed
         * weights w_i in [1, 2) lies in the span of the unflagged rows (v = a_w - A'lambda, lambda from the Gram factor as for
         * one row, |v| <= 1e-9 |a_w|) with a consistent right-hand side (|b_w - lambda'b| within the tolerance of a_w's
         * size); a group whose rows are not all dependent and consistent fails for all but measure-zero weights. A group
         * that fails is halved; a single row that fails is kept (dependent but inconsistent: counted, nothing dropped, as
         * one by one). Its rows dropped as flagged, without the swaps by priority (each needs the row's own combination).
         * The checks within the same budget */
        int *stk = xm(sizeof(int) * (2 * (size_t)nf + 64)), *lst = xm(sizeof(int) * ((size_t)nf + 1)), nl = 0;
        for (int i = 0; i < nr; i++) if (flag[i]) { if (!(d0[i] > 0)) { if (b[i] == 0) dead[i] = 1; else (*ninc)++; } else lst[nl++] = i; }
        int sp = 0; stk[sp++] = 0; stk[sp++] = nl;            /* (ranges [lo, hi) of lst) */
        double spent = 0; int ndg = 0, nbadg = 0, ntest = 0;
        while (sp > 0) {
            const int hi = stk[--sp], lo = stk[--sp];
            if (hi <= lo) continue;
            if (spent + per > chk) break;                        /* (the rest kept) */
            spent += per; ntest++;
            for (int j = 0; j < ncol; j++) u[j] = 0;
            double bw = 0, aw2 = 0;
            for (int t = lo; t < hi; t++) {
                const int i = lst[t]; const double w = 1.0 + (double)(((unsigned)i * 2654435761u) % 1000u) / 1000.0;
                for (int p = rp[i]; p < rp[i + 1]; p++) u[rj[p]] += w * rv[p];
                bw += w * b[i]; aw2 += w * w * d0[i];
            }
            double *aw = xm(sizeof(double) * (ncol + 1)); memcpy(aw, u, sizeof(double) * ncol);
            double *lam = acc; for (int k = 0; k < nr; k++) lam[k] = 0;
            double sn = 0, an = 0; for (int j = 0; j < ncol; j++) an += aw[j] * aw[j]; an = sqrt(an);
            for (int pass = 0; pass < 2; pass++) {
                for (int k = 0; k < nr; k++) { double t2 = 0; if (!flag[k]) for (int p = rp[k]; p < rp[k + 1]; p++) t2 += rv[p] * u[rj[p]]; g[k] = t2; }
                schol_solve(S, g, wk);
                for (int k = 0; k < nr; k++) if (!flag[k]) lam[k] += g[k];
                memcpy(u, aw, sizeof(double) * ncol);
                for (int k = 0; k < nr; k++) if (!flag[k] && lam[k] != 0) for (int p = rp[k]; p < rp[k + 1]; p++) u[rj[p]] -= lam[k] * rv[p];
                double vv = 0; for (int j = 0; j < ncol; j++) vv += u[j] * u[j];
                sn = sqrt(vv) / fmax(an, 1e-300);
                if (sn <= 1e-12) break;
            }
            free(aw);
            const int dep = sn <= 1e-9;
            double r = bw; for (int k = 0; k < nr; k++) if (!flag[k]) r -= lam[k] * b[k];
            const int cons = fabs(r) <= ctol * sqrt(aw2);
            if (dep && cons) { for (int t = lo; t < hi; t++) dead[lst[t]] = 1; ndg += hi - lo; continue; }
            if (hi - lo == 1) { if (dep) nbadg++; continue; }
            const int mid = lo + (hi - lo) / 2;
            stk[sp++] = lo; stk[sp++] = mid; stk[sp++] = mid; stk[sp++] = hi;
        }
        if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent rows by groups: %d of %d flagged dropped, %d inconsistent, %d tests (%.3g of the budget's %.3g flops)]\n", ndg, nf, nbadg, ntest, spent, chk);
        free(stk); free(lst);
        if (nbadg > 0 || *ninc > 0) { *ninc += nbadg; for (int i = 0; i < nr; i++) dead[i] = 0; removed = 0; goto out; }
        removed = 0; for (int i = 0; i < nr; i++) removed += dead[i];
        goto out;
    }
    int nd = 0, nbad = 0;
    /* (23 October 2026, the default; BRISK_QPDEPBATCH=0: one by one) the first pass's solves of the flagged rows together,
     * by chunks of right-hand sides (schol_solve_many, as the SDP presolve verifies its flagged rows): the solves are
     * independent of the decisions (the factor and the unflagged rows are fixed), and one solve per check was 0.33 s of
     * STCQP1's 0.41 s (QP: 1,113 flagged rows against a factor of 2.4e5 entries; now 0.25 s). On the 299 QPs the same
     * iterations and objectives everywhere */
    static int dbat = -1; if (dbat < 0) { const char *e = getenv("BRISK_QPDEPBATCH"); dbat = e ? atoi(e) : 1; }
    int *dfl = NULL, ndfl = 0, dch = 0, dc0 = 0, dc1 = 0, dpos = 0; double *dGB = NULL, *dwk = NULL, *dub = NULL;
    if (dbat) {
        dfl = xm(sizeof(int) * ((size_t)nf + 1));
        for (int i = 0; i < nr; i++) if (flag[i] && d0[i] > 0) dfl[ndfl++] = i;
        dch = (int)((1u << 22) / ((unsigned)nr + 1)); if (dch < 1) dch = 1; if (dch > 256) dch = 256; if (dch > ndfl) dch = ndfl > 0 ? ndfl : 1;
        dGB = xm(sizeof(double) * (size_t)nr * dch); dwk = xm(sizeof(double) * ((size_t)nr * dch + nr + 1)); dub = xz(sizeof(double) * ((size_t)ncol + 1));
    }
    int nsw = 0, swcap = 0, *sw_out = NULL, *sw_in = NULL, *sw_i = NULL; long *sw_p = NULL, swvcap = 0; double *sw_piv = NULL, *sw_v = NULL;   /* (the swaps by priority) */
    for (int i = 0; i < nr; i++) if (flag[i]) {
        if (!(d0[i] > 0)) { if (b[i] == 0) { dead[i] = 1; nd++; } else nbad++; continue; }      /* an empty row: consistent only with b_i = 0 */
        /* lambda: G lambda = A_K a_i, one refinement (G dl = A_K v) */
        for (int j = 0; j < ncol; j++) u[j] = 0;
        for (int p = rp[i]; p < rp[i + 1]; p++) u[rj[p]] = rv[p];
        double *lam = acc; for (int k = 0; k < nr; k++) lam[k] = 0;
        double sn = 0;
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 0 && dbat) {
                if (dpos >= dc1) {                       /* the next chunk: A_K a_i for its rows, then their solves together */
                    dc0 = dpos; dc1 = dc0 + dch < ndfl ? dc0 + dch : ndfl;
                    for (int t = dc0; t < dc1; t++) {
                        const int ii = dfl[t]; double *gc = dGB + (size_t)(t - dc0) * nr;
                        for (int p = rp[ii]; p < rp[ii + 1]; p++) dub[rj[p]] = rv[p];
                        for (int k = 0; k < nr; k++) { double t2 = 0; if (!flag[k]) for (int p = rp[k]; p < rp[k + 1]; p++) t2 += rv[p] * dub[rj[p]]; gc[k] = t2; }
                        for (int p = rp[ii]; p < rp[ii + 1]; p++) dub[rj[p]] = 0;
                    }
                    schol_solve_many(S, dGB, dc1 - dc0, dwk);
                }
                memcpy(g, dGB + (size_t)(dpos - dc0) * nr, sizeof(double) * nr);
            } else {
                for (int k = 0; k < nr; k++) { double t = 0; if (!flag[k]) for (int p = rp[k]; p < rp[k + 1]; p++) t += rv[p] * u[rj[p]]; g[k] = t; }
                schol_solve(S, g, wk);
            }
            for (int k = 0; k < nr; k++) if (!flag[k]) lam[k] += g[k];
            for (int j = 0; j < ncol; j++) u[j] = 0;
            for (int p = rp[i]; p < rp[i + 1]; p++) u[rj[p]] = rv[p];
            for (int k = 0; k < nr; k++) if (!flag[k] && lam[k] != 0) for (int p = rp[k]; p < rp[k + 1]; p++) u[rj[p]] -= lam[k] * rv[p];
            double vv = 0; for (int j = 0; j < ncol; j++) vv += u[j] * u[j];
            sn = sqrt(vv / d0[i]);
            if (sn <= 1e-12) break;
        }
        dpos++;
        if (sn > 1e-9) continue;
        double r = b[i]; for (int k = 0; k < nr; k++) if (!flag[k]) r -= lam[k] * b[k];
        if (fabs(r) / sqrt(d0[i]) > ctol) { nbad++; continue; }
        int drop = i;
        if (prio) {
            /* the combination in the current kept rows: each earlier swap (kout replaced by d0, a_kout = (a_d0 - sum_k mu_k
             * a_k) / mu_kout, mu its own combination at its time) substituted, in their order */
            for (int t = 0; t < nsw; t++) {
                const int ko = sw_out[t]; const double lk = lam[ko];
                if (lk == 0) continue;
                const double c = lk / sw_piv[t];
                lam[ko] = 0; lam[sw_in[t]] += c;
                for (long e = sw_p[t]; e < sw_p[t + 1]; e++) if (sw_i[e] != ko) lam[sw_i[e]] -= c * sw_v[e];
            }
            double lmax = 0; for (int k = 0; k < nr; k++) lmax = fmax(lmax, fabs(lam[k]));
            int kb = -1;
            for (int k = 0; k < nr; k++) if (k != i && !dead[k] && fabs(lam[k]) > 1e-8 * lmax && (kb < 0 || prio[k] > prio[kb])) kb = k;
            if (kb >= 0 && prio[kb] > prio[i]) {
                drop = kb;
                /* the swap kept: a_kb = (a_i - sum_{k != kb} lam_k a_k) / lam_kb */
                long cnt = 0; for (int k = 0; k < nr; k++) if (lam[k] != 0 && fabs(lam[k]) > 1e-14 * lmax) cnt++;
                if (nsw == swcap) { swcap = swcap ? 2 * swcap : 64; sw_out = realloc(sw_out, sizeof(int) * swcap); sw_in = realloc(sw_in, sizeof(int) * swcap); sw_piv = realloc(sw_piv, sizeof(double) * swcap); sw_p = realloc(sw_p, sizeof(long) * (swcap + 1)); if (nsw == 0) sw_p[0] = 0; }
                if (sw_p[nsw] + cnt > swvcap) { while (sw_p[nsw] + cnt > swvcap) swvcap = swvcap ? 2 * swvcap : 4096; sw_i = realloc(sw_i, sizeof(int) * swvcap); sw_v = realloc(sw_v, sizeof(double) * swvcap); }
                long e = sw_p[nsw];
                for (int k = 0; k < nr; k++) if (lam[k] != 0 && fabs(lam[k]) > 1e-14 * lmax) { sw_i[e] = k; sw_v[e] = lam[k]; e++; }
                sw_out[nsw] = kb; sw_in[nsw] = i; sw_piv[nsw] = lam[kb]; sw_p[nsw + 1] = e; nsw++;
            }
            if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent row %d (priority %.2e): %s %d (priority %.2e)]\n", i, prio[i], drop != i ? "dropped instead" : "kept dropped, the best kept", kb, kb >= 0 ? prio[kb] : 0);
        }
        dead[drop] = 1; nd++;
    }
    if (nbad > 0) { *ninc = nbad; for (int i = 0; i < nr; i++) dead[i] = 0; nd = 0; }
    free(sw_out); free(sw_in); free(sw_piv); free(sw_p); free(sw_i); free(sw_v); free(dfl); free(dGB); free(dwk); free(dub);
    removed = nd;
out:
    if (S) schol_free(S);
    free(cp); free(ci); free(cv); free(mk); free(deg); free(nbr); free(pool); free(acc); free(d0); free(piv); free(flag); free(g); free(u); free(wk);
    return removed;
}
/* (23 October 2026, the SOCP developer) the rows that cannot take part in a dependence set aside first, as in the SDP
 * presolve (IMPLEMENTATION 5.14): a row with an entry in a column that no other remaining row has is independent of the
 * rest (its coefficient in a vanishing combination is 0 there); it is set aside, which can leave another row alone in a
 * column, and so on, in O(nnz). Only the rest (the core) goes to the Gram matrix of qp_dep_rows_core, whose guards on
 * its density and cost then see the core alone. QPLIB_8602 (QP): 44,514 equality rows on 26,142 columns, at least 18,372
 * dependent, none found: the Gram matrix of all the rows was refused as too dense. */
static int qp_dep_rows(int nr, int ncol, const int *rp, const int *rj, const double *rv, const double *b, double tol, char *dead, int *ninc, const double *prio) {
    *ninc = 0; for (int i = 0; i < nr; i++) dead[i] = 0;
    if (nr < 2) return 0;
    static int peel = -1; if (peel < 0) { const char *e_ = getenv("BRISK_QPDEPPEEL"); peel = !(e_ && atoi(e_) == 0); }
    if (!peel) return qp_dep_rows_core(nr, ncol, rp, rj, rv, b, tol, dead, ninc, prio);
    int *cc = xz(sizeof(int) * (ncol + 1)), *cp = xz(sizeof(int) * (ncol + 2));
    for (int p = 0; p < rp[nr]; p++) if (rv[p] != 0) { cc[rj[p]]++; cp[rj[p] + 1]++; }
    for (int j = 0; j < ncol; j++) cp[j + 1] += cp[j];
    int *ci = xm(sizeof(int) * ((size_t)cp[ncol] + 1)), *f = xm(sizeof(int) * (ncol + 1));
    memcpy(f, cp, sizeof(int) * ncol);
    for (int i = 0; i < nr; i++) for (int p = rp[i]; p < rp[i + 1]; p++) if (rv[p] != 0) ci[f[rj[p]]++] = i;
    free(f);
    char *aside = xz(nr + 1); int *stk = xm(sizeof(int) * (ncol + 1)), ns = 0, nas = 0;
    for (int j = 0; j < ncol; j++) if (cc[j] == 1) stk[ns++] = j;
    while (ns > 0) {
        const int j = stk[--ns];
        if (cc[j] != 1) continue;
        int i = -1; for (int q = cp[j]; q < cp[j + 1]; q++) if (!aside[ci[q]]) { i = ci[q]; break; }
        if (i < 0) continue;
        aside[i] = 1; nas++;
        for (int p = rp[i]; p < rp[i + 1]; p++) if (rv[p] != 0) { const int c = rj[p]; if (--cc[c] == 1) stk[ns++] = c; }
    }
    free(stk); free(cc); free(cp); free(ci);
    /* (23 October 2026, later) rows of the core that are multiples of another (a_i = alpha a_k: the same pattern, the
     * values over the first one equal to 1e-14 relative) found by a hash and dropped when b_i = alpha b_k to 1e-12 of
     * their size (the one of the larger activity bound when priorities are given, the later one otherwise); an
     * inconsistent pair keeps every row (the interior-point method reports the infeasibility). QPLIB_8602 (QP): 5,621
     * such rows, QPLIB_8785: 1,315 */
    int ndup = 0;
    {   static int dup = -1; if (dup < 0) { const char *e_ = getenv("BRISK_QPDEPDUP"); dup = !(e_ && atoi(e_) == 0); }
        if (dup) {
            /* (23 October 2026, the night) the table sized by the rows, a power of two at least twice their number: a fixed
             * 2^20 entries were set to -1 on every solve, 11 % of a small QP's (QUADCMPC1: 384 equality rows). Rows equal
             * up to scale have the same hash and so the same bucket at any size, in the same order: the same matches */
            int hsz = 64; while (hsz < 2 * nr && hsz < (1 << 20)) hsz <<= 1;
            int *head = xm(sizeof(int) * hsz), *next = xm(sizeof(int) * (nr + 1));
            for (int h = 0; h < hsz; h++) head[h] = -1;
            for (int i = 0; i < nr && !*ninc; i++) {
                if (aside[i] || rp[i + 1] == rp[i] || rv[rp[i]] == 0) continue;
                const double v0 = rv[rp[i]];
                unsigned long long hh = 1469598103934665603ULL ^ (unsigned)(rp[i + 1] - rp[i]);
                for (int p = rp[i]; p < rp[i + 1]; p++) {
                    double q = rv[p] / v0; long long iq = llround(q * 1e8);
                    hh = (hh ^ (unsigned long long)rj[p]) * 1099511628211ULL; hh = (hh ^ (unsigned long long)iq) * 1099511628211ULL;
                }
                const int hb = (int)(hh & (unsigned long long)(hsz - 1));
                int match = -1;
                for (int k = head[hb]; k >= 0; k = next[k]) {
                    if (rp[k + 1] - rp[k] != rp[i + 1] - rp[i]) continue;
                    const double w0 = rv[rp[k]]; int same = 1;
                    for (int t = 0; t < rp[i + 1] - rp[i] && same; t++) {
                        const int pi = rp[i] + t, pk = rp[k] + t;
                        if (rj[pi] != rj[pk] || fabs(rv[pi] / v0 - rv[pk] / w0) > 1e-14 * (fabs(rv[pi] / v0) + 1e-300)) same = 0;
                    }
                    if (same) { match = k; break; }
                }
                if (match < 0) { next[i] = head[hb]; head[hb] = i; continue; }
                const double al = v0 / rv[rp[match]];          /* a_i = al a_match */
                if (fabs(b[i] - al * b[match]) > 1e-12 * (fabs(b[i]) + fabs(al * b[match]) + 1e-300) && fabs(b[i] - al * b[match]) > 0) { *ninc = 1; break; }
                int drop = i;
                if (prio && prio[match] > prio[i]) { drop = match; next[i] = head[hb]; head[hb] = i; }   /* (the kept one stays in its class) */
                dead[drop] = 1; aside[drop] = 2; ndup++;
                if (drop == match) {      /* (match left the class: unlink it) */
                    int *pp = &head[hb]; while (*pp >= 0 && *pp != match) pp = &next[*pp]; if (*pp == match) *pp = next[match];
                }
            }
            free(head); free(next);
            if (*ninc) { for (int i = 0; i < nr; i++) dead[i] = 0; free(aside); return 0; }
        }
    }
    if (nas == 0 && ndup == 0) { free(aside); return qp_dep_rows_core(nr, ncol, rp, rj, rv, b, tol, dead, ninc, prio); }
    const int nc = nr - nas - ndup;
    int removed = 0;
    if (nc >= 2) {
        int *map = xm(sizeof(int) * (nc + 1)), *rp2 = xm(sizeof(int) * (nc + 2));
        long nz = 0; int t = 0;
        for (int i = 0; i < nr; i++) if (!aside[i]) { map[t++] = i; nz += rp[i + 1] - rp[i]; }
        int *rj2 = xm(sizeof(int) * ((size_t)nz + 1)); double *rv2 = xm(sizeof(double) * ((size_t)nz + 1)), *b2 = xm(sizeof(double) * (nc + 1)), *pr2 = prio ? xm(sizeof(double) * (nc + 1)) : NULL;
        long o = 0; rp2[0] = 0;
        for (int k = 0; k < nc; k++) { const int i = map[k]; for (int p = rp[i]; p < rp[i + 1]; p++) { rj2[o] = rj[p]; rv2[o] = rv[p]; o++; } rp2[k + 1] = (int)o; b2[k] = b[i]; if (pr2) pr2[k] = prio[i]; }
        char *dead2 = xz(nc + 1);
        removed = qp_dep_rows_core(nc, ncol, rp2, rj2, rv2, b2, tol, dead2, ninc, pr2);
        if (*ninc == 0) for (int k = 0; k < nc; k++) if (dead2[k]) dead[map[k]] = 1;
        if (*ninc) for (int i = 0; i < nr; i++) dead[i] = 0;
        free(map); free(rp2); free(rj2); free(rv2); free(b2); free(pr2); free(dead2);
    }
    if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent rows: %d of %d rows set aside as independent, %d multiples of another row, a core of %d]\n", nas, nr, ndup, nc);
    free(aside);
    return *ninc ? 0 : removed + ndup;
}
static void red_free(Red *R) {
    free(R->Hp); free(R->Hi); free(R->Hx); free(R->Ap); free(R->Ai); free(R->Ax); free(R->c);
    free(R->xl); free(R->xu); free(R->rl); free(R->ru); free(R->hl); free(R->hu); free(R->eq); free(R->gl); free(R->gu);
    free(R->D); free(R->E); free(R->col0); free(R->row0); free(R->srow); free(R->scol); free(R->sa); free(R->xlp); free(R->xup);
}

/* 0: built; 1: primal infeasible found (a row without entries outside its bounds, a column with xl > xu) */
static int reduce(const QpProb *P, Red *R, double *fixval, double *c0_add) {
    const int n0 = P->n, m0 = P->m;
    memset(R, 0, sizeof(*R));
    int *cmap = xm(sizeof(int) * (n0 + 1)), *rmap = xm(sizeof(int) * (m0 + 1));
    /* (22 October 2026) a row with one entry, rl <= a x_j <= ru, is a bound on x_j: the column's bounds tightened, the
     * row removed, its multiplier recovered from the column's (extract). A test (BRISK_QPSINGLE=1), off: kept as rows */
    double *xl = xm(sizeof(double) * (n0 + 1)), *xu = xm(sizeof(double) * (n0 + 1));
    memcpy(xl, P->xl, sizeof(double) * n0); memcpy(xu, P->xu, sizeof(double) * n0);
    char *srm = xz(m0 + 1);
    {   static int sg = -1; if (sg < 0) { const char *e = getenv("BRISK_QPSINGLE"); sg = e ? atoi(e) : 0; }
        if (sg && P->nqc == 0 && m0 > 0) {
            int *cnt = xz(sizeof(int) * (m0 + 1)), *sj = xm(sizeof(int) * (m0 + 1)); double *sav = xm(sizeof(double) * (m0 + 1));
            for (int j = 0; j < n0; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0) { const int i = P->Ai[p]; cnt[i]++; sj[i] = j; sav[i] = P->Ax[p]; }
            int ns = 0; for (int i = 0; i < m0; i++) ns += cnt[i] == 1 && (fin(P->rl[i]) || fin(P->ru[i]));
            if (ns) {
                R->srow = xm(sizeof(int) * ns); R->scol = xm(sizeof(int) * ns); R->sa = xm(sizeof(double) * ns);
                for (int i = 0; i < m0; i++) if (cnt[i] == 1 && (fin(P->rl[i]) || fin(P->ru[i]))) {
                    const int j = sj[i]; const double a = sav[i];
                    const double lo = fin(P->rl[i]) ? P->rl[i] / a : (a > 0 ? -QP_INF : QP_INF), hi = fin(P->ru[i]) ? P->ru[i] / a : (a > 0 ? QP_INF : -QP_INF);
                    const double l = a > 0 ? lo : hi, u = a > 0 ? hi : lo;
                    if (fin(l) && l > xl[j]) xl[j] = l;
                    if (fin(u) && u < xu[j]) xu[j] = u;
                    srm[i] = 1; R->srow[R->ns] = i; R->scol[R->ns] = j; R->sa[R->ns] = a; R->ns++;
                }
            }
            free(cnt); free(sj); free(sav);
        }
    }
    R->xlp = xl; R->xup = xu;
    /* (25 October 2026, the SOCP developer) a column with no entry in A or H, a cost and no bound in its descent direction
     * is an unbounded direction (H d = 0, A d = 0, c'd < 0, d within the bounds' recession): the unboundedness certificate
     * the iteration checks, found before it (QPUNB_20000, QP: a nearly dense H of 20,000 columns, 60 s an iteration until
     * the iterates grow). Returned as 2 with the column in g_qp_ray (+1 or -1: the direction's sign) */
    if (P->nqc == 0) {
        char *used = xz(n0 + 1);
        for (int j = 0; j < n0; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0) used[j] = 1;
        if (P->Hp) for (int j = 0; j < n0; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) if (P->Hx[p] != 0) { used[j] = 1; used[P->Hi[p]] = 1; }
        for (int j = 0; j < n0; j++) if (!used[j] && P->c[j] != 0 && (P->c[j] < 0 ? !fin(xu[j]) : !fin(xl[j]))) {
            g_qp_ray = P->c[j] < 0 ? j + 1 : -(j + 1); free(used); free(cmap); free(rmap); free(srm); return 2; }
        free(used);
    }
    int n = 0;
    for (int j = 0; j < n0; j++) {
        if (fin(xl[j]) && fin(xu[j]) && xl[j] > xu[j] + 1e-9 * (1 + fabs(xl[j]))) { free(cmap); free(rmap); free(srm); return 1; }
        if (fin(xl[j]) && fin(xu[j]) && xu[j] - xl[j] <= 1e-12 * (1 + fabs(xl[j]))) { cmap[j] = -1; fixval[j] = 0.5 * (xl[j] + xu[j]); }
        else { cmap[j] = n++; fixval[j] = 0; }
    }
    /* the fixed columns' contribution: c0 + c'v + 1/2 v'Hv, the cost of the others c + H v, the rows' bounds - A v */
    double *cn = xz(sizeof(double) * (n0 + 1)), *ash = xz(sizeof(double) * (m0 + 1));
    for (int j = 0; j < n0; j++) cn[j] = P->c[j];
    double add = 0;
    for (int j = 0; j < n0; j++) if (cmap[j] < 0) {
        const double v = fixval[j]; add += P->c[j] * v;
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) ash[P->Ai[p]] += P->Ax[p] * v;
    }
    if (P->Hp) for (int j = 0; j < n0; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) {
        const int i = P->Hi[p]; const double h = P->Hx[p];
        const int fi = cmap[i] < 0, fj = cmap[j] < 0;
        if (fi && fj) add += (i == j ? 0.5 : 1.0) * h * fixval[i] * fixval[j];
        else if (fi) cn[j] += h * fixval[i];
        else if (fj) cn[i] += h * fixval[j];
    }
    *c0_add = add;
    /* rows: count entries in the kept columns */
    int *rcnt = xz(sizeof(int) * (m0 + 1));
    for (int j = 0; j < n0; j++) if (cmap[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0) rcnt[P->Ai[p]]++;
    int m = 0;
    for (int i = 0; i < m0; i++) {
        const double lo = fin(P->rl[i]) ? P->rl[i] - ash[i] : -QP_INF, hi = fin(P->ru[i]) ? P->ru[i] - ash[i] : QP_INF;
        if (srm[i]) { rmap[i] = -1; continue; }
        if (rcnt[i] == 0) {
            if ((fin(lo) && lo > 1e-9 * (1 + fabs(lo))) || (fin(hi) && hi < -1e-9 * (1 + fabs(hi)))) { free(cmap); free(rmap); free(cn); free(ash); free(rcnt); free(srm); return 1; }
            rmap[i] = -1; continue;
        }
        if (!fin(lo) && !fin(hi)) { rmap[i] = -1; continue; }
        rmap[i] = 0;
    }
    /* (23 October 2026) equality rows that are linear combinations of others dropped (their multipliers 0; the dependence
     * test of the LP and SDP presolve, qp_dep_rows, on the kept columns): a network's balance rows, one dependent per
     * connected component, left the normal equations singular along them and the dual residual lost (QPLIB_9002: 199 of
     * 1,649 rows; OPTIMAL in 29 iterations without them). Rows of the quadratic constraints are not considered.
     * BRISK_QPDEP=0: kept, as before */
    g_qp_ndep = 0;
    {   static int qdep = -1; if (qdep < 0) { const char *e = getenv("BRISK_QPDEP"); qdep = !(e && atoi(e) == 0); }
        int neq = 0; char *qr = NULL;
        if (P->nqc) { qr = xz(m0 + 1); for (int k = 0; k < P->nqc; k++) qr[P->qrow[k]] = 1; }
        int *eqi = xm(sizeof(int) * (m0 + 1)), *pos = xm(sizeof(int) * (m0 + 1));
        for (int i = 0; i < m0; i++) {
            pos[i] = -1;
            if (rmap[i] < 0 || (qr && qr[i]) || !fin(P->rl[i]) || !fin(P->ru[i]) || P->rl[i] != P->ru[i]) continue;
            pos[i] = neq; eqi[neq++] = i;
        }
        if (qdep && neq >= 2) {
            int *rp = xz(sizeof(int) * (neq + 2));
            for (int j = 0; j < n0; j++) if (cmap[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0 && pos[P->Ai[p]] >= 0) rp[pos[P->Ai[p]] + 1]++;
            for (int q = 0; q < neq; q++) rp[q + 1] += rp[q];
            int *rj = xm(sizeof(int) * ((size_t)rp[neq] + 1)), *fill = xm(sizeof(int) * (neq + 1)); double *rv = xm(sizeof(double) * ((size_t)rp[neq] + 1));
            memcpy(fill, rp, sizeof(int) * neq);
            for (int j = 0; j < n0; j++) if (cmap[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ax[p] != 0 && pos[P->Ai[p]] >= 0) { const int t = fill[pos[P->Ai[p]]]++; rj[t] = cmap[j]; rv[t] = P->Ax[p]; }
            double *bq = xm(sizeof(double) * (neq + 1)); for (int q = 0; q < neq; q++) bq[q] = P->rl[eqi[q]] - ash[eqi[q]];
            char *dead = xz(neq + 1); int ninc = 0;
            static double dtol = -1; if (dtol < 0) dtol = getenv("BRISK_DEPTOL") ? atof(getenv("BRISK_DEPTOL")) : 1e-8;
            const double tdep = now();
            double *pq = NULL;
            if (g_qp_rowprio) { pq = xm(sizeof(double) * (neq + 1)); for (int q = 0; q < neq; q++) pq[q] = g_qp_rowprio[eqi[q]]; }
            else {   /* the first solve: each row's activity bounded by its columns' finite bounds (1 for a free column);
                      * a dropped row inherits its kept rows' rounding, small against its own terms only when they are
                      * as large (QPLIB_9002, QP: one solve instead of a diverging one and a second, 1.9 -> 0.25 s) */
                pq = xz(sizeof(double) * (neq + 1));
                for (int j = 0; j < n0; j++) if (cmap[j] >= 0) {
                    double sz = 0; if (fin(P->xl[j])) sz = fabs(P->xl[j]); if (fin(P->xu[j])) sz = fmax(sz, fabs(P->xu[j])); if (sz == 0) sz = 1;
                    for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (pos[P->Ai[p]] >= 0) pq[pos[P->Ai[p]]] += fabs(P->Ax[p]) * sz;
                }
            }
            const int nd = qp_dep_rows(neq, n, rp, rj, rv, bq, dtol, dead, &ninc, pq);
            free(pq);
            if (getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[dependent rows: %d of %d equality rows, %d inconsistent, %.3f s]\n", nd, neq, ninc, now() - tdep);
            /* inconsistent rows (a dependent row whose right-hand side is not the same combination): every row is kept and
             * the interior-point method's certificate test decides, which returns the Farkas multipliers with the status */
            if (getenv("BRISK_QPDEPDBG") && getenv("BRISK_QPDEPDBG")[0] == '2') for (int q = 0; q < neq; q++) if (dead[q]) fprintf(stderr, "[dropped row %d]\n", eqi[q]);
            if (ninc == 0) { for (int q = 0; q < neq; q++) if (dead[q]) { rmap[eqi[q]] = -1; if (g_qp_dead) g_qp_dead[eqi[q]] = 1; } g_qp_ndep = nd; }
            free(rp); free(rj); free(fill); free(rv); free(bq); free(dead);
        }
        free(eqi); free(pos); free(qr);
    }
    for (int i = 0; i < m0; i++) if (rmap[i] >= 0) rmap[i] = m++;
    R->n = n; R->m = m;
    R->c = xm(sizeof(double) * (n + 1)); R->xl = xm(sizeof(double) * (n + 1)); R->xu = xm(sizeof(double) * (n + 1));
    R->hl = xz(n + 1); R->hu = xz(n + 1); R->col0 = xm(sizeof(int) * (n + 1));
    for (int j = 0; j < n0; j++) if (cmap[j] >= 0) {
        const int k = cmap[j]; R->c[k] = cn[j]; R->xl[k] = xl[j]; R->xu[k] = xu[j]; R->col0[k] = j;
        R->hl[k] = fin(xl[j]); R->hu[k] = fin(xu[j]);
    }
    R->rl = xm(sizeof(double) * (m + 1)); R->ru = xm(sizeof(double) * (m + 1)); R->row0 = xm(sizeof(int) * (m + 1));
    R->eq = xz(m + 1); R->gl = xz(m + 1); R->gu = xz(m + 1);
    for (int i = 0; i < m0; i++) if (rmap[i] >= 0) {
        const int k = rmap[i]; R->row0[k] = i;
        const double lo = fin(P->rl[i]) ? P->rl[i] - ash[i] : -QP_INF, hi = fin(P->ru[i]) ? P->ru[i] - ash[i] : QP_INF;
        R->rl[k] = lo; R->ru[k] = hi;
        if (fin(lo) && fin(hi) && hi - lo <= 1e-12 * (1 + fabs(lo))) { R->eq[k] = 1; R->rl[k] = R->ru[k] = 0.5 * (lo + hi); }
        else { R->gl[k] = fin(lo); R->gu[k] = fin(hi); }
    }
    /* A of the kept columns and rows */
    long na = 0; for (int j = 0; j < n0; j++) if (cmap[j] >= 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (rmap[P->Ai[p]] >= 0 && P->Ax[p] != 0) na++;
    R->Ap = xz(sizeof(int) * (n + 1)); R->Ai = xm(sizeof(int) * (na + 1)); R->Ax = xm(sizeof(double) * (na + 1));
    long w = 0;
    for (int j = 0; j < n0; j++) if (cmap[j] >= 0) {
        const int k = cmap[j]; R->Ap[k] = (int)w;
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int r = rmap[P->Ai[p]]; if (r >= 0 && P->Ax[p] != 0) { R->Ai[w] = r; R->Ax[w] = P->Ax[p]; w++; } }
    }
    R->Ap[n] = (int)w;
    /* H of the kept columns (lower triangle; the row indices stay sorted since the map is monotone) */
    long nh = 0; if (P->Hp) for (int j = 0; j < n0; j++) if (cmap[j] >= 0) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) if (cmap[P->Hi[p]] >= 0 && P->Hx[p] != 0) nh++;
    R->Hp = xz(sizeof(int) * (n + 1)); R->Hi = xm(sizeof(int) * (nh + 1)); R->Hx = xm(sizeof(double) * (nh + 1));
    w = 0;
    for (int j = 0; j < n0; j++) if (cmap[j] >= 0) {
        const int k = cmap[j]; R->Hp[k] = (int)w;
        if (P->Hp) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) { const int i = cmap[P->Hi[p]]; if (i >= 0 && P->Hx[p] != 0) { R->Hi[w] = i; R->Hx[w] = P->Hx[p]; w++; } }
    }
    R->Hp[n] = (int)w;
    free(cmap); free(rmap); free(cn); free(ash); free(rcnt); free(srm);
    return 0;
}

/* (25 October 2026, the SOCP developer) the scaling of Curtis and Reid (1972) on K = [H A'; A 0]: the column factors
 * g = log D and the row factors r = log E minimizing sum over the entries of (log|a_ij| + r_i + g_j)^2 and of
 * (log|h_jk| + g_j + g_k)^2, by conjugate gradients with the diagonal as the preconditioner. Its minimizer moves with any
 * scaling of the given rows and columns (a_ij -> s_i a_ij t_j shifts r_i by -log s_i and g_j by -log t_j), so the scaled
 * matrix does not depend on it; Ruiz's equilibration, run after it, has many fixed points and lands on one that depends on
 * the starting point (QPSCALE_GRID_40000, QP, the rows scaled by 10^U(-4, 4) and the columns by 10^U(-3, 3): 176 iterations
 * against its unscaled twin's 16, one row's residual held at 4.5e-5 of its size for 150 iterations). Applied to H and A
 * and returned in D and E; the right-hand sides, bounds and cost are scaled with Ruiz's factors after it */
static void curtis_reid(Red *R, double *D, double *E) {
    const int n = R->n, m = R->m, N = n + m;
    double *g = xz(sizeof(double) * (N + 1)), *res = xm(sizeof(double) * (N + 1)), *z = xm(sizeof(double) * (N + 1)),
           *pd = xz(sizeof(double) * (N + 1)), *q = xm(sizeof(double) * (N + 1)), *dg = xz(sizeof(double) * (N + 1)), *u = xz(sizeof(double) * (N + 1));
    /* the right-hand side -sum s_t log|a_t| and the diagonal sum s_t^2 (unknowns: columns 0..n-1, rows n..n+m-1) */
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) { const double a = fabs(R->Hx[p]); if (a == 0) continue;
        const int k = R->Hi[p]; const double l = log(a);
        if (k == j) { g[j] -= 2 * l; dg[j] += 4; } else { g[j] -= l; g[k] -= l; dg[j] += 1; dg[k] += 1; } }
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const double a = fabs(R->Ax[p]); if (a == 0) continue;
        const int i = n + R->Ai[p]; const double l = log(a); g[j] -= l; g[i] -= l; dg[j] += 1; dg[i] += 1; }
    /* conjugate gradients from u = 0 (an unknown without entries stays 0) */
    double g0 = 0; for (int k = 0; k < N; k++) { res[k] = dg[k] > 0 ? g[k] : 0; g0 += res[k] * res[k]; }
    double rz = 0; for (int k = 0; k < N; k++) { z[k] = dg[k] > 0 ? res[k] / dg[k] : 0; pd[k] = z[k]; rz += res[k] * z[k]; }
    for (int it = 0; it < 500 && g0 > 0; it++) {
        for (int k = 0; k < N; k++) q[k] = 0;
        for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) { if (R->Hx[p] == 0) continue; const int k = R->Hi[p];
            if (k == j) q[j] += 4 * pd[j]; else { const double v = pd[j] + pd[k]; q[j] += v; q[k] += v; } }
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { if (R->Ax[p] == 0) continue; const int i = n + R->Ai[p];
            const double v = pd[j] + pd[i]; q[j] += v; q[i] += v; }
        double pq = 0; for (int k = 0; k < N; k++) pq += pd[k] * q[k];
        if (!(pq > 0)) break;
        const double al = rz / pq; double rr = 0, rz2 = 0;
        for (int k = 0; k < N; k++) { u[k] += al * pd[k]; res[k] -= al * q[k]; rr += res[k] * res[k]; }
        if (rr <= 1e-12 * g0) break;
        for (int k = 0; k < N; k++) { z[k] = dg[k] > 0 ? res[k] / dg[k] : 0; rz2 += res[k] * z[k]; }
        const double be = rz2 / rz; rz = rz2;
        for (int k = 0; k < N; k++) pd[k] = z[k] + be * pd[k];
    }
    for (int j = 0; j < n; j++) D[j] = exp(fmin(fmax(u[j], -230.0), 230.0));
    for (int i = 0; i < m; i++) E[i] = exp(fmin(fmax(u[n + i], -230.0), 230.0));
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) R->Hx[p] *= D[j] * D[R->Hi[p]];
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) R->Ax[p] *= D[j] * E[R->Ai[p]];
    free(g); free(res); free(z); free(pd); free(q); free(dg); free(u);
}

/* Ruiz's equilibration of K = [H A'; A 0]: each column j of [H; A] and each row of A brought to the infinity norm 1 by
 * x = D x~ and the rows scaled by E, ten passes; then the cost scaled so that max(|c|, |diag H|) is about 1 */
static void equilibrate(Red *R, int passes) {
    const int n = R->n, m = R->m;
    R->D = xm(sizeof(double) * (n + 1)); R->E = xm(sizeof(double) * (m + 1));
    for (int j = 0; j < n; j++) R->D[j] = 1;
    for (int i = 0; i < m; i++) R->E[i] = 1;
    { static int cr = -1; if (cr < 0) { const char *e = getenv("BRISK_QPGEOM"); cr = e ? atoi(e) : 0; }
      if (cr == 1 && passes > 0) curtis_reid(R, R->D, R->E);
      if (cr == 2 && passes > 0) {   /* (test) geometric-mean passes: each column of [H; A] and each row of A by 1/sqrt(min max) */
          double *lo = xm(sizeof(double) * (n + m + 1)), *hi = xm(sizeof(double) * (n + m + 1));
          for (int ps = 0; ps < 6; ps++) {
              for (int k = 0; k < n + m; k++) { lo[k] = INFINITY; hi[k] = 0; }
              for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) { const double a = fabs(R->Hx[p]); if (a == 0) continue; const int k = R->Hi[p];
                  lo[j] = fmin(lo[j], a); hi[j] = fmax(hi[j], a); lo[k] = fmin(lo[k], a); hi[k] = fmax(hi[k], a); }
              for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const double a = fabs(R->Ax[p]); if (a == 0) continue; const int i = n + R->Ai[p];
                  lo[j] = fmin(lo[j], a); hi[j] = fmax(hi[j], a); lo[i] = fmin(lo[i], a); hi[i] = fmax(hi[i], a); }
              for (int k = 0; k < n + m; k++) lo[k] = hi[k] > 0 ? 1.0 / sqrt(sqrt(lo[k] * hi[k])) : 1.0;   /* (the square root: K's symmetric scaling d_j d_k) */
              for (int k = n; k < n + m; k++) lo[k] *= lo[k];   /* (a row of A by its own factor, its columns already by theirs) */
              for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) R->Hx[p] *= lo[j] * lo[R->Hi[p]];
              for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) R->Ax[p] *= lo[j] * lo[n + R->Ai[p]];
              for (int j = 0; j < n; j++) R->D[j] *= lo[j];
              for (int i = 0; i < m; i++) R->E[i] *= lo[n + i];
          }
          free(lo); free(hi);
      } }
    double *cn = xm(sizeof(double) * (n + 1)), *rn = xm(sizeof(double) * (m + 1));
    for (int it = 0; it < passes; it++) {
        for (int j = 0; j < n; j++) cn[j] = 0;
        for (int i = 0; i < m; i++) rn[i] = 0;
        for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) {
            const int i = R->Hi[p]; const double v = fabs(R->Hx[p]);
            if (v > cn[j]) cn[j] = v;
            if (v > cn[i]) cn[i] = v;
        }
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) {
            const double v = fabs(R->Ax[p]); const int i = R->Ai[p];
            if (v > cn[j]) cn[j] = v;
            if (v > rn[i]) rn[i] = v;
        }
        double dmax = 0;
        for (int j = 0; j < n; j++) { const double d = cn[j] > 0 ? 1.0 / sqrt(cn[j]) : 1.0; cn[j] = fmin(fmax(d, 1e-4), 1e4); dmax = fmax(dmax, fabs(cn[j] - 1)); }
        for (int i = 0; i < m; i++) { const double d = rn[i] > 0 ? 1.0 / sqrt(rn[i]) : 1.0; rn[i] = fmin(fmax(d, 1e-4), 1e4); dmax = fmax(dmax, fabs(rn[i] - 1)); }
        /* apply: H := d H d, A := e A d */
        for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) R->Hx[p] *= cn[j] * cn[R->Hi[p]];
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) R->Ax[p] *= cn[j] * rn[R->Ai[p]];
        for (int j = 0; j < n; j++) R->D[j] *= cn[j];
        for (int i = 0; i < m; i++) R->E[i] *= rn[i];
        if (dmax < 1e-3) break;
    }
    for (int j = 0; j < n; j++) {
        R->c[j] *= R->D[j];
        if (R->hl[j]) R->xl[j] /= R->D[j];
        if (R->hu[j]) R->xu[j] /= R->D[j];
    }
    for (int i = 0; i < m; i++) { if (fin(R->rl[i])) R->rl[i] *= R->E[i]; if (fin(R->ru[i])) R->ru[i] *= R->E[i]; }
    /* (23 October 2026, a test: BRISK_QPCOLB=1) each variable scaled by its own size, the larger of its finite bounds
     * (when above 1), then the rows equilibrated again: QPLIB_9002's bounds span 1e3 to 1e11 and its solution 0.1 to 8e9,
     * which one primal scale cannot fit */
    { static int colb = -1; if (colb < 0) { const char *e = getenv("BRISK_QPCOLB"); colb = e ? atoi(e) : 0; }
      if (colb) {
        double *sc = malloc(sizeof(double) * (size_t)(n + 1)); int any = 0;
        for (int j = 0; j < n; j++) {
            double b = 0; if (R->hl[j]) b = fmax(b, fabs(R->xl[j])); if (R->hu[j]) b = fmax(b, fabs(R->xu[j]));
            sc[j] = b > 1 ? b : 1.0; any |= sc[j] != 1.0;
        }
        if (any) {
            for (int j = 0; j < n; j++) {
                const double t = sc[j];
                if (t == 1.0) continue;
                R->D[j] *= t; R->c[j] *= t; if (R->hl[j]) R->xl[j] /= t; if (R->hu[j]) R->xu[j] /= t;
                for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) R->Ax[p] *= t;
            }
            for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) R->Hx[p] *= sc[j] * sc[R->Hi[p]];
            double *rm = calloc((size_t)m + 1, sizeof(double));
            for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) rm[R->Ai[p]] = fmax(rm[R->Ai[p]], fabs(R->Ax[p]));
            for (int i = 0; i < m; i++) if (rm[i] > 0) { rm[i] = 1.0 / rm[i]; R->E[i] *= rm[i]; if (fin(R->rl[i])) R->rl[i] *= rm[i]; if (fin(R->ru[i])) R->ru[i] *= rm[i]; } else rm[i] = 1.0;
            for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) R->Ax[p] *= rm[R->Ai[p]];
            free(rm);
        }
        free(sc);
      } }
    /* the size of the primal variables (22 October 2026): x = bs x^ with bs the largest right-hand side of a row (the
     * bounds when no row has one), so that the scaled problem's right-hand sides and its solution are of order one.
     * The matrix scaling leaves the solution's size alone; on QSHELL (right-hand sides 5e4, a solution up to 3e5) the
     * iterates moved by 1e5-1e6 a step, and the regularization's error rho |dx| in the dual residual was 1e-2 a step */
    /* (22 October 2026) and not beyond the variables' own bounds, the direct evidence of their size: BOYD2's nine right-hand
     * sides reach 2.2e11 while every variable lies in [0, 0.1]; scaled by 2e7, the bounds were 5e-9, the cost scale
     * 2.5e-15, and the iteration ended at a gap of 0.14 with mu at 1e-21. b_s = max(1, min(the largest right-hand side,
     * the largest finite bound)) */
    double bm = 0, bb = 0; int anyb = 0;
    for (int i = 0; i < m; i++) { if (fin(R->rl[i])) bm = fmax(bm, fabs(R->rl[i])); if (fin(R->ru[i])) bm = fmax(bm, fabs(R->ru[i])); }
    for (int j = 0; j < n; j++) { if (R->hl[j]) { bb = fmax(bb, fabs(R->xl[j])); anyb = 1; } if (R->hu[j]) { bb = fmax(bb, fabs(R->xu[j])); anyb = 1; } }
    static int bsr = -1; if (bsr < 0) { const char *e = getenv("BRISK_QPBSRULE"); bsr = e ? atoi(e) : 0; }   /* (0, the default: the right-hand sides; tests: 1 and the bounds, 2 and the minimizer's size -- on the 138 Maros-Meszaros QPs 2,382 -> 2,681 iterations, the LP-like ones lost most: QSHARE1B 18 -> 56, POWELL20 30 -> 88; BOYD1 39 -> 19) */
    if (bm == 0) bm = bb; else if (anyb && bsr >= 1) bm = fmin(bm, bb);
    /* and not beyond the size of the unconstrained minimizer, max|c| / max H_jj (H != 0): scaling x by b_s multiplies H by
     * b_s^2 (QPLIB_9002: bounds to 3.4e8, b_s at its cap 1e8, and the cost scale fell to 1e-16) */
    {   double cm = 0, hm = 0;
        for (int j = 0; j < n; j++) cm = fmax(cm, fabs(R->c[j]));
        for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) if (R->Hi[p] == j) hm = fmax(hm, fabs(R->Hx[p]));
        if (bsr >= 2 && hm > 0 && bm > 0) bm = fmin(bm, fmax(1.0, cm / hm)); }
    { static int bon = -1; if (bon < 0) { const char *e = getenv("BRISK_QPBSCALE"); bon = e ? atoi(e) : 1; } if (!bon) bm = 0; }
    R->bs = bm > 1 ? fmin(bm, 1e8) : 1.0;
    if (R->bs != 1) {
        for (int j = 0; j < n; j++) { if (R->hl[j]) R->xl[j] /= R->bs; if (R->hu[j]) R->xu[j] /= R->bs; R->c[j] *= R->bs; }
        for (int i = 0; i < m; i++) { if (fin(R->rl[i])) R->rl[i] /= R->bs; if (fin(R->ru[i])) R->ru[i] /= R->bs; }
        for (int p = 0; p < R->Hp[n]; p++) R->Hx[p] *= R->bs * R->bs;
    }
    /* the cost */
    double cm = 0, hm = 0;
    for (int j = 0; j < n; j++) cm = fmax(cm, fabs(R->c[j]));
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) if (R->Hi[p] == j) hm = fmax(hm, fabs(R->Hx[p]));
    { static int csr = -1; if (csr < 0) { const char *e = getenv("BRISK_QPCSRULE"); csr = e ? atoi(e) : 1; }
      if (csr == 1) {   /* (the default from 22 October 2026; 0: the largest entry) H's size as the mean of its columns' largest entries, as OSQP's cost scaling: on the 288 QPs it keeps BOYD2 and QPCBOEI2 */
          double *cn2 = calloc((size_t)n, sizeof(double)), sm = 0;
          for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) { double a = fabs(R->Hx[p]); int i = R->Hi[p]; cn2[j] = fmax(cn2[j], a); cn2[i] = fmax(cn2[i], a); }
          for (int j = 0; j < n; j++) sm += cn2[j];
          hm = n ? sm / n : 0; free(cn2);
      } }
    double s = fmax(cm, hm); R->cs = s > 0 ? 1.0 / fmax(s, 1e-6) : 1.0;
    if (R->cs > 1e6) R->cs = 1e6;
    for (int j = 0; j < n; j++) R->c[j] *= R->cs;
    for (int p = 0; p < R->Hp[n]; p++) R->Hx[p] *= R->cs;
    free(cn); free(rn);
}

/* the iteration's state and work */
typedef struct {
    Red *R;
    int n, m, N;
    SChol *K;
    size_t *offHd, *offH, *offA, *offR;   /* value offsets: H's diagonal (per column), H's entries, A's entries, rows' diagonal */
    signed char *sgn;
    double rho, delta, rho_ref, delta_ref, delta_ineq, delta_ineq_ref;   /* (delta_ineq: on the inequality rows) */
    double tol;                             /* the stop's tolerance (the refinement's floor, BRISK_QPREFINE=3) */
    double rf_du, rf_pr, rf_gap;            /* the refinement's targets (BRISK_QPREFINE=3): dual and primal equations, gap */
    int fullref;                            /* the full refinement from here on (the targets' progress stalled) */
    double e_prev;                          /* the last iteration's largest error on the problem as given */
    int nundo, rmu_hold;                    /* (BRISK_QPUNDO) steps undone; iterations the regularization stays raised */
    int redo;                               /* (23 October 2026) this iteration repeats an undone step (the stall rule skips it) */
    double epr_prev;                        /* the last iteration's scaled primal residual */
    int refmu;                              /* (BRISK_QPREFMU) the targets in the neighbourhood; 0 once the progress stalled */
    double rpn_prev, rdn_prev, rpn0, rdn0, mu0;   /* (BRISK_QPREFMU) the last iteration's and the start's residuals (scaled units, absolute) and the start's mu */
    double *x, *y, *s, *tl, *tu, *zl, *zu, *wl, *wu, *vl, *vu;
    double *dx, *dy, *ds, *dtl, *dtu, *dzl, *dzu, *dwl, *dwu, *dvl, *dvu;
    double *rd, *rs, *rp, *rtl, *rtu, *rwl, *rwu;
    double *ctl, *ctu, *cwl, *cwu;          /* the complementarity right-hand sides */
    double *Dx, *Ds, *Hdiag;
    double *rhs, *sol, *res, *work, *tmp, *hx;
    double *gV, *gZ;                        /* (BRISK_QPGMRES) the Krylov basis, (gm + 1) N, and its preconditioned vectors, gm N, allocated at the first use */
    double *sv;                             /* a copy of the direction and of its complementarity right-hand sides */
    int nfact, nsolve, nref, ncorr;
    double tfact, tsolve;                   /* seconds in the factorizations and in the solves (with the refinement) */
    int kcorr;                              /* Gondzio's correctors at most per iteration */
    /* the normal equations (H diagonal): M = A diag(d) A' + E, d_j = 1 / (H_jj + Dx_j (+ rho for a column with neither)), E the
     * inequality rows' Ds^-1 (0 on the equality rows) */
    SChol *M; size_t *offMd, *offMp; int *mpst; double *dne;
    int dasm, wdc; double *Md, *Wd; size_t *dmo;   /* M by dsyrk on dense panels of A D^1/2 (wdc columns, Wd) into a dense Md; dmo: the places of its entries below the diagonal, by columns */
    int *grp, *grq, *grj, *gmp, *gmk; size_t *gmo; double *gw;   /* M formed row by row: A's sparse part by rows (grp, grq: CSC positions), M's entries below the diagonal by rows (gmp, gmk) and their places in the factor (gmo), a dense row (gw) */
    int nerho;                              /* rho on every column of the normal equations (after a breakdown) */
    /* (22 October 2026) the dense columns kept as unknowns: [M_s A_d; A_d' -D_d^-1] (dy; dx_d), M_s from the sparse
     * columns; offsets of A_d's entries (by A's index) and of the dense unknowns' diagonals */
    int nd, *dcol; char *isd; size_t *offAd, *offDd;
    SChol *Kspare;                          /* the augmented system's analysis, kept for a breakdown of the normal equations */
    double *st0;                            /* the iterate before the last step (x, s, tl, tu, zl, zu; y, wl, wu, vl, vu) */
    /* (24 October 2026) the normal equations solved by conjugate gradients, preconditioned by their spanning tree (pcg = 1:
     * every column of A has at most two entries and no factor fits; ne_pcg_*) */
    int pcg, *tord, *tpar, *tcol, *ecol, *uf; double *tL, *tD, *tE, *tw, *cgw; long ncg;
} Ipm;

/* the iterate (x, tl, tu, zl, zu; y, s, wl, wu, vl, vu) copied to I->st0 or back */
static void state_copy(Ipm *I, int back) {
    double *nv[] = { I->x, I->tl, I->tu, I->zl, I->zu }, *mv[] = { I->y, I->s, I->wl, I->wu, I->vl, I->vu };
    if (!I->st0) I->st0 = malloc(sizeof(double) * (5 * (size_t)I->n + 6 * (size_t)I->m + 1));
    double *p = I->st0;
    for (int k = 0; k < 5; k++, p += I->n) { if (back) memcpy(nv[k], p, sizeof(double) * I->n); else memcpy(p, nv[k], sizeof(double) * I->n); }
    for (int k = 0; k < 6; k++, p += I->m) { if (back) memcpy(mv[k], p, sizeof(double) * I->m); else memcpy(p, mv[k], sizeof(double) * I->m); }
}
/* the direction (dx, dtl, dtu, dzl, dzu, ctl, ctu; dy, ds, dwl, dwu, dvl, dvu, cwl, cwu) copied to I->sv or back */
static void dir_copy(Ipm *I, int back) {
    double *nv[] = { I->dx, I->dtl, I->dtu, I->dzl, I->dzu, I->ctl, I->ctu }, *mv[] = { I->dy, I->ds, I->dwl, I->dwu, I->dvl, I->dvu, I->cwl, I->cwu };
    double *p = I->sv;
    for (int k = 0; k < 7; k++, p += I->n) { if (back) memcpy(nv[k], p, sizeof(double) * I->n); else memcpy(p, nv[k], sizeof(double) * I->n); }
    for (int k = 0; k < 8; k++, p += I->m) { if (back) memcpy(mv[k], p, sizeof(double) * I->m); else memcpy(p, mv[k], sizeof(double) * I->m); }
}

/* (24 October 2026, the SOCP developer) the factor computed in place over the assembled values (BRISK_QPINPLACE=0: on a
 * copy, as before): the copy is the size of the factor (QPLIB_8547, QP: 1.1 GB); its fill positions then hold the last
 * factor, so the values are zeroed before they are filled again (a memset in place of the copy's memcpy). With the
 * normal equations' tiny-pivot rule, sparsechol.c saves the original diagonals the rule reads */
static double *qp_values(SChol *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_QPINPLACE"); on = e ? atoi(e) : 1; }
    if (on) { schol_set_inplace(S, 1); if (schol_inplace(S)) schol_zero(S); }
    return schol_values(S);
}
/* (24 October 2026, the SOCP developer) the normal equations M = A D A' + E of a problem whose columns of A have at most
 * two entries each (a network's node-arc incidence matrix, scaled: the rows are nodes, a column of two entries an arc), when
 * no factor of M or of the augmented system fits in memory (QFLOW_100000, QP: a flow on a random graph, whose factor fills
 * at any order). M is the weighted graph matrix of the arcs; it is solved by conjugate gradients preconditioned by T, the
 * same matrix on a spanning forest of maximum weight (the arc j's weight d_j |a_ij a_kj|) together with the diagonal of the
 * columns of one entry and of E: a tree's matrix factors without fill, so T is factored and solved in linear time; as the
 * iterates approach a vertex the weights of the basic arcs dominate and T approaches M (the network interior-point methods
 * of Resende and Veiga, 1993, and Portugal, Resende, Veiga and Judice, 2000). A component of the forest whose matrix is
 * singular (no diagonal term: the rows of a connected network without a slack are dependent) has its root's pivot set to
 * its largest weight, so that T is positive definite; M's null vector there is in the kernel of A', and the iterations,
 * started from 0 with a right-hand side in M's range, stay in that range. The method's refinement (GMRES on the whole system
 * with the solve as its preconditioner, the preconditioned vectors kept) corrects the inexact solves. */
static const double *qp_cg_w;      /* (the weights qsort compares) */
static int g_cgprec = -1;            /* (BRISK_QPCGPREC, a test: 1 the diagonal of M instead of the tree, 2 the tree over M's diagonal) */
static int qp_cg_cmp(const void *a, const void *b) {
    const int i = *(const int *)a, j = *(const int *)b;
    if (qp_cg_w[i] != qp_cg_w[j]) return qp_cg_w[i] > qp_cg_w[j] ? -1 : 1;
    return i - j;
}
static int uf_root(int *uf, int i) { while (uf[i] != i) { uf[i] = uf[uf[i]]; i = uf[i]; } return i; }
/* T for the current weights: the forest by Kruskal's rule (the arcs by decreasing weight, ties by index), then its LDL'
 * from the leaves (I->tord: the rows, each after its children; tpar the parent row, tcol the arc to it) */
static void ne_pcg_factor(Ipm *I) {
    Red *R = I->R; const int n = I->n, m = I->m;
    { static int all = -1; if (all < 0) { const char *e = getenv("BRISK_QPNERHO"); all = e ? atoi(e) : 0; }
      for (int j = 0; j < n; j++) { const double dd = I->Hdiag[j] + I->Dx[j]; I->dne[j] = (all || I->nerho) ? 1.0 / (dd + I->rho) : 1.0 / (dd > 0 ? dd : I->rho); } }
    for (int i = 0; i < m; i++) I->tE[i] = R->eq[i] ? 0.0 : 1.0 / I->Ds[i];
    /* T's diagonal: E, the columns of one entry */
    double *g = I->tD; for (int i = 0; i < m; i++) g[i] = I->tE[i];
    int na = 0;
    for (int j = 0; j < n; j++) {
        const int c = R->Ap[j + 1] - R->Ap[j];
        if (c == 1) { const int p = R->Ap[j]; g[R->Ai[p]] += I->dne[j] * R->Ax[p] * R->Ax[p]; }
        else if (c == 2) { const int p = R->Ap[j]; I->tw[j] = I->dne[j] * fabs(R->Ax[p] * R->Ax[p + 1]); I->ecol[na++] = j; }
    }
    {   double *dm = I->tw + n; for (int i = 0; i < m; i++) dm[i] = I->tE[i];
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) dm[R->Ai[p]] += I->dne[j] * R->Ax[p] * R->Ax[p];
        for (int i = 0; i < m; i++) if (!(dm[i] > 0)) dm[i] = 1.0; }
    qp_cg_w = I->tw; qsort(I->ecol, (size_t)na, sizeof(int), qp_cg_cmp);
    for (int i = 0; i < m; i++) I->uf[i] = i;
    /* the forest's adjacency: each row's tree arcs (at most m - 1 of them), as a list per row */
    int *deg = calloc((size_t)m + 1, sizeof(int)), *tarc = malloc(sizeof(int) * ((size_t)m + 1)), nt = 0;
    for (int t = 0; t < na; t++) {
        const int j = I->ecol[t], p = R->Ap[j], a = uf_root(I->uf, R->Ai[p]), b = uf_root(I->uf, R->Ai[p + 1]);
        if (a == b) continue;
        I->uf[a] = b; tarc[nt++] = j; deg[R->Ai[p]]++; deg[R->Ai[p + 1]]++;
    }
    int *st = malloc(sizeof(int) * ((size_t)m + 2)); st[0] = 0; for (int i = 0; i < m; i++) st[i + 1] = st[i] + deg[i];
    int *adj = malloc(sizeof(int) * (2 * (size_t)nt + 1)); for (int i = 0; i < m; i++) deg[i] = st[i];
    for (int t = 0; t < nt; t++) { const int j = tarc[t], p = R->Ap[j]; adj[deg[R->Ai[p]]++] = j; adj[deg[R->Ai[p + 1]]++] = j; }
    /* breadth first from each component's first row: tord the visiting order (a parent before its children) */
    for (int i = 0; i < m; i++) I->tpar[i] = -2;
    int qh = 0, qt = 0;
    for (int r = 0; r < m; r++) {
        if (I->tpar[r] != -2) continue;
        I->tpar[r] = -1; I->tcol[r] = -1; I->tord[qt++] = r;
        while (qh < qt) {
            const int i = I->tord[qh++];
            for (int q = st[i]; q < st[i + 1]; q++) {
                const int j = adj[q], p = R->Ap[j], k = R->Ai[p] == i ? R->Ai[p + 1] : R->Ai[p];
                if (I->tpar[k] != -2) continue;
                I->tpar[k] = i; I->tcol[k] = j; I->tord[qt++] = k;
            }
        }
    }
    free(deg); free(tarc); free(st); free(adj);
    /* the tree arcs' terms in T: T_ii += w a_i^2, T_pp += w a_p^2, T_ip = w a_i a_p; then the elimination, children first */
    for (int i = 0; i < m; i++) {
        const int j = I->tcol[i]; if (j < 0) { I->tL[i] = 0; continue; }
        const int p = R->Ap[j]; const double ai = R->Ai[p] == i ? R->Ax[p] : R->Ax[p + 1], ap = R->Ai[p] == i ? R->Ax[p + 1] : R->Ax[p];
        g[i] += I->dne[j] * ai * ai; g[I->tpar[i]] += I->dne[j] * ap * ap; I->tL[i] = I->dne[j] * ai * ap;   /* (T_ip for now) */
    }
    /* (BRISK_QPCGPREC=2, a test) the tree's off-diagonal entries over M's whole diagonal */
    if (g_cgprec < 0) { const char *e = getenv("BRISK_QPCGPREC"); g_cgprec = e ? atoi(e) : 2; }
    if (g_cgprec == 2) for (int i = 0; i < m; i++) g[i] = I->tw[n + i];
    /* each component's largest diagonal, for a singular component's root */
    double *cmax = I->cgw; for (int i = 0; i < m; i++) cmax[i] = 0;
    for (int t = m - 1; t >= 0; t--) {
        const int i = I->tord[t], pr = I->tpar[i];
        cmax[i] = fmax(cmax[i], g[i]);
        if (pr < 0) { if (!(g[i] > 1e-14 * cmax[i])) g[i] = cmax[i] > 0 ? cmax[i] : 1.0; continue; }
        cmax[pr] = fmax(cmax[pr], cmax[i]);
        const double l = I->tL[i] / g[i];
        g[pr] -= l * I->tL[i]; I->tL[i] = l;        /* (tD = g holds the pivots, tL the multipliers L_pi) */
    }
}
/* z = T^-1 r: L D L' with L's entry L_pi below each row i's parent */
static void ne_pcg_tsolve(const Ipm *I, const double *r, double *z) {
    const int m = I->m;
    if (g_cgprec < 0) { const char *e = getenv("BRISK_QPCGPREC"); g_cgprec = e ? atoi(e) : 2; }
    if (g_cgprec == 1) { for (int i = 0; i < m; i++) z[i] = r[i] / I->tw[I->n + i]; return; }
    memcpy(z, r, sizeof(double) * m);
    for (int t = m - 1; t >= 0; t--) { const int i = I->tord[t], p = I->tpar[i]; if (p >= 0) z[p] -= I->tL[i] * z[i]; }
    for (int i = 0; i < m; i++) z[i] /= I->tD[i];
    for (int t = 0; t < m; t++) { const int i = I->tord[t], p = I->tpar[i]; if (p >= 0) z[i] -= I->tL[i] * z[p]; }
}
/* y = M v = A diag(dne) A' v + E v */
static void ne_pcg_mul(const Ipm *I, const double *v, double *y) {
    const Red *R = I->R; const int n = I->n, m = I->m;
    for (int i = 0; i < m; i++) y[i] = I->tE[i] * v[i];
    for (int j = 0; j < n; j++) {
        double t = 0; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) t += R->Ax[p] * v[R->Ai[p]];
        t *= I->dne[j];
        for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) y[R->Ai[p]] += R->Ax[p] * t;
    }
}
/* x = M^-1 b by preconditioned conjugate gradients from 0, in place on b; to a relative residual of BRISK_QPCGTOL (1e-10)
 * or 500 iterations */
static void ne_pcg_solve_n(Ipm *I, double *b, int cap);
static void ne_pcg_solve(Ipm *I, double *b) { ne_pcg_solve_n(I, b, 0); }
/* (cap > 0: at most cap iterations, the trial of the choice below) */
static void ne_pcg_solve_n(Ipm *I, double *b, int cap) {
    const int m = I->m;
    double *x = I->cgw, *r = x + m, *z = r + m, *p = z + m, *q = p + m;
    static double tol = -1; static int maxit = -1;
    if (tol < 0) { const char *e = getenv("BRISK_QPCGTOL"); tol = e ? atof(e) : 1e-10; e = getenv("BRISK_QPCGMAX"); maxit = e ? atoi(e) : 500; }
    double bn = 0; for (int i = 0; i < m; i++) { x[i] = 0; r[i] = b[i]; bn += b[i] * b[i]; }
    bn = sqrt(bn);
    if (!(bn > 0)) return;
    ne_pcg_tsolve(I, r, z);
    double rz = 0; for (int i = 0; i < m; i++) { p[i] = z[i]; rz += r[i] * z[i]; }
    int it = 0;
    const int mx = cap > 0 && cap < maxit ? cap : maxit;
    for (; it < mx; it++) {
        ne_pcg_mul(I, p, q);
        double pq = 0; for (int i = 0; i < m; i++) pq += p[i] * q[i];
        if (!(pq > 0)) break;
        const double a = rz / pq;
        double rn = 0; for (int i = 0; i < m; i++) { x[i] += a * p[i]; r[i] -= a * q[i]; rn += r[i] * r[i]; }
        if (sqrt(rn) <= tol * bn) { it++; break; }
        ne_pcg_tsolve(I, r, z);
        double rz2 = 0; for (int i = 0; i < m; i++) rz2 += r[i] * z[i];
        const double be = rz2 / rz; rz = rz2;
        for (int i = 0; i < m; i++) p[i] = z[i] + be * p[i];
    }
    I->ncg += it;
    if (getenv("BRISK_QPCGDBG")) qp_printf("       [cg %d iterations]\n", it);
    memcpy(b, x, sizeof(double) * m);
}
/* the method's arrays for the conjugate gradients (every column of A at most two entries: 1, else 0) */
static int ne_pcg_setup(Ipm *I) {
    const Red *R = I->R; const int n = I->n, m = I->m;
    for (int j = 0; j < n; j++) if (R->Ap[j + 1] - R->Ap[j] > 2) return 0;
    I->pcg = 1; I->nd = 0;
    I->tord = xm(sizeof(int) * (m + 1)); I->tpar = xm(sizeof(int) * (m + 1)); I->tcol = xm(sizeof(int) * (m + 1)); I->uf = xm(sizeof(int) * (m + 1));
    I->ecol = xm(sizeof(int) * (n + 1)); I->tL = xm(sizeof(double) * (m + 1)); I->tD = xm(sizeof(double) * (m + 1)); I->tE = xm(sizeof(double) * (m + 1));
    I->tw = xm(sizeof(double) * ((size_t)n + m + 1)); I->cgw = xm(sizeof(double) * (5 * (size_t)m + 1));
    if (!I->dne) I->dne = xm(sizeof(double) * (n + 1));
    return 1;
}

/* the factored matrix's values for the current Dx, Ds */
static int ne_factor(Ipm *I) {
    Red *R = I->R; const int n = I->n, m = I->m;
    double *v = qp_values(I->M);
    { static int all = -1; if (all < 0) { const char *e = getenv("BRISK_QPNERHO"); all = e ? atoi(e) : 0; }
      for (int j = 0; j < n; j++) { const double dd = I->Hdiag[j] + I->Dx[j]; I->dne[j] = (all || I->nerho) ? 1.0 / (dd + I->rho) : 1.0 / (dd > 0 ? dd : I->rho); } }
    for (int a = 0; a < I->nd; a++) {
        const int j = I->dcol[a];
        v[I->offDd[a]] = -1.0 / I->dne[j];
        for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) v[I->offAd[p]] = R->Ax[p];
    }
    if (I->dasm) {     /* (25 October 2026) dense panels of A D^1/2 by dsyrk, M's entries copied from the dense result */
        const double one = 1.0; double beta = 0.0; const int md = m;
        for (int c0 = 0; c0 < n; c0 += I->wdc) {
            const int kk = n - c0 < I->wdc ? n - c0 : I->wdc;
            memset(I->Wd, 0, sizeof(double) * (size_t)m * (size_t)kk);
            for (int t = 0; t < kk; t++) {
                const int j = c0 + t; const double f = sqrt(I->dne[j]); double *wc = I->Wd + (size_t)t * m;
                for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) wc[R->Ai[p]] += f * R->Ax[p];
            }
            qp_dsyrk("L", "N", &md, &kk, &one, I->Wd, &md, &beta, I->Md, &md);
            beta = 1.0;
        }
        if (beta == 0.0) for (int i = 0; i < m; i++) memset(I->Md + (size_t)i * m + i, 0, sizeof(double) * (size_t)(m - i));
        size_t t = 0;
        for (int i = 0; i < m; i++) {
            const double *col = I->Md + (size_t)i * m;
            v[I->offMd[i]] = col[i] + (R->eq[i] ? 0.0 : 1.0 / I->Ds[i]);
            for (int k = i + 1; k < m; k++) v[I->dmo[t++]] = col[k];
        }
        goto formed;
    }
    if (I->gmp) {      /* row i of A D A': a_ij d_j times column j below row i, gathered into M's entries of column i */
        double *w = I->gw;
        for (int i = 0; i < m; i++) {
            double di = R->eq[i] ? 0.0 : 1.0 / I->Ds[i];
            for (int t = I->grp[i]; t < I->grp[i + 1]; t++) {
                const int p = I->grq[t], j = I->grj[t]; const double d = R->Ax[p] * I->dne[j];
                di += d * R->Ax[p];
                for (int q = p + 1; q < R->Ap[j + 1]; q++) w[R->Ai[q]] += d * R->Ax[q];
            }
            v[I->offMd[i]] = di;
            for (int t = I->gmp[i]; t < I->gmp[i + 1]; t++) { const int k = I->gmk[t]; v[I->gmo[t]] = w[k]; w[k] = 0; }
        }
        goto formed;
    }
    for (int i = 0; i < m; i++) v[I->offMd[i]] = R->eq[i] ? 0.0 : 1.0 / I->Ds[i];
    for (int j = 0; j < n; j++) for (int q = I->mpst[j]; q < I->mpst[j + 1]; q++) v[I->offMp[q]] = 0.0;
    for (int j = 0; j < n; j++) {
        if (I->nd && I->isd[j]) continue;
        const double dj = I->dne[j]; int q = I->mpst[j];
        for (int a = R->Ap[j]; a < R->Ap[j + 1]; a++) {
            v[I->offMd[R->Ai[a]]] += R->Ax[a] * R->Ax[a] * dj;
            for (int b = R->Ap[j]; b < a; b++) v[I->offMp[q++]] += R->Ax[a] * R->Ax[b] * dj;
        }
    }
formed:
    I->nfact++;
    const double t0 = now();
    int rc = schol_factor(I->M, 0.0);
    I->tfact += now() - t0;
    return rc;
}
static int ipm_factor(Ipm *I) {
    if (I->pcg) { const double t0 = now(); ne_pcg_factor(I); I->nfact++; I->tfact += now() - t0; return 0; }
    if (I->M) return ne_factor(I);
    Red *R = I->R; const int n = I->n, m = I->m;
    double *v = qp_values(I->K);
    for (int j = 0; j < n; j++) v[I->offHd[j]] = -(I->Hdiag[j] + I->Dx[j] + I->rho);
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) if (R->Hi[p] != j) v[I->offH[p]] = -R->Hx[p];
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) v[I->offA[p]] = R->Ax[p];
    /* (22 October 2026) the regularization delta on the equality rows only: an inequality row's diagonal Ds^-1 > 0 keeps
     * the matrix quasi-definite already, and delta = 1e-8 against Ds^-1 near 1e-16 at the end made the refinement contract
     * at a rate close to 1, so the step's error grew like 1/mu (LASER: the dual residual 4e-7 -> 5e2 in five iterations) */
    for (int i = 0; i < m; i++) v[I->offR[i]] = R->eq[i] ? I->delta : 1.0 / I->Ds[i] + I->delta_ineq;
    I->nfact++;
    const double t0 = now();
    const int rc = schol_factor(I->K, 0.0);
    I->tfact += now() - t0;
    return rc;
}

/* w = K_ref v with the reference regularization */
static void kmul(const Ipm *I, const double *v, double *w) {
    const Red *R = I->R; const int n = I->n, m = I->m;
    for (int j = 0; j < n; j++) w[j] = -(I->Hdiag[j] + I->Dx[j] + I->rho_ref) * v[j];
    for (int i = 0; i < m; i++) w[n + i] = (R->eq[i] ? I->delta_ref : 1.0 / I->Ds[i] + I->delta_ineq_ref) * v[n + i];
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) {
        const int i = R->Hi[p]; if (i == j) continue;
        w[i] -= R->Hx[p] * v[j]; w[j] -= R->Hx[p] * v[i];
    }
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) {
        const int i = n + R->Ai[p];
        w[i] += R->Ax[p] * v[j]; w[j] += R->Ax[p] * v[i];
    }
}

/* sol = K^-1 rhs, refined against K_ref (at most 6 steps, kept while the residual falls) */
/* the factored system's solve, in place on b (n + m): the augmented system's factor, or the normal equations:
 * dy = M^-1 (r2 + A D r1), dx = D (A'dy - r1) for [-D^-1 A'; A E] (dx; dy) = (r1; r2) */
static void kbase(Ipm *I, double *b) {
    if (!I->M && !I->pcg) { const int pn = qp_part_nt(); if (!(pn > 0 && schol_solve_part(I->K, b, I->work, pn) == 0)) schol_solve(I->K, b, I->work); return; }
    const Red *R = I->R; const int n = I->n, m = I->m;
    double *r2 = I->work;                 /* (work holds 2 (n + m) + 2 m + 2 doubles: r2 m + nd, the solve's scratch after it) */
    const int nd = I->nd;
    for (int i = 0; i < m; i++) r2[i] = b[n + i];
    for (int j = 0; j < n; j++) { if (nd && I->isd[j]) continue; const double t = I->dne[j] * b[j]; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) r2[R->Ai[p]] += R->Ax[p] * t; }
    for (int a = 0; a < nd; a++) r2[m + a] = b[I->dcol[a]];
    if (I->pcg) ne_pcg_solve(I, r2);
    else { const int pn = qp_part_nt(); if (!(pn > 0 && schol_solve_part(I->M, r2, I->work + m + nd, pn) == 0)) schol_solve(I->M, r2, I->work + m + nd); }
    for (int a = 0; a < nd; a++) b[I->dcol[a]] = r2[m + a];
    for (int j = 0; j < n; j++) { if (nd && I->isd[j]) continue; double a = 0; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) a += R->Ax[p] * r2[R->Ai[p]]; b[j] = I->dne[j] * (a - b[j]); }
    for (int i = 0; i < m; i++) b[n + i] = r2[i];
}
static void ksolve_(Ipm *I);
static void ksolve(Ipm *I) { const double t0 = now(); ksolve_(I); I->tsolve += now() - t0; }
/* (22 October 2026, the default with m = 10; BRISK_QPGMRES=0: the plain refinement) the refinement by restarted GMRES(m) on K_ref sol = rhs, preconditioned on the
 * right by the factored matrix (kbase), from sol and its residual res: it minimizes the residual over the Krylov space of
 * the steps the plain refinement takes one by one, so it contracts at least as fast, and much faster when the factored
 * matrix differs from K_ref in a few directions (the regularization against tiny diagonals: the plain refinement
 * contracted by 1.01-1.25 a step on QCAPRI, QP). The targets as the plain refinement's (BRISK_QPREFINE=3); otherwise on
 * until a cycle no longer halves the residual; at most 60 products in all. Returns the residual's largest entry. */
static int targets_met(const Ipm *I) {
    if (I->rf_du <= 0 || I->fullref) return 0;
    double eg = 0; for (int j = 0; j < I->n; j++) eg += fabs(I->x[j] * I->res[j]);
    for (int i = 0; i < I->m; i++) eg += fabs(I->y[i] * I->res[I->n + i]);
    return nrmi(I->n, I->res) <= I->rf_du && nrmi(I->m, I->res + I->n) <= I->rf_pr && eg <= I->rf_gap;
}
static double kgmres(Ipm *I, int gm, double rn, int rm3) {
    const int N = I->n + I->m;
    if (!I->gV) { I->gV = xm(sizeof(double) * (size_t)(gm + 1) * N); I->gZ = xm(sizeof(double) * (size_t)gm * N); }
    double *V = I->gV, *Z = I->gZ, *w = I->tmp;
    double H[41 * 40], cs[40], sn[40], g[41], yv[40];
    int nmv = 0;
    while (nmv < 60) {
        if (rm3 && targets_met(I)) break;
        double beta = 0; for (int k = 0; k < N; k++) beta += I->res[k] * I->res[k]; beta = sqrt(beta);
        if (!(beta > 0)) break;
        for (int k = 0; k < N; k++) V[k] = I->res[k] / beta;
        for (int i = 0; i <= gm; i++) g[i] = 0;
        g[0] = beta;
        const double tin = rm3 && !I->fullref ? 0.5 * fmin(I->rf_du, I->rf_pr) : 1e-3 * beta;
        int k = 0;
        for (; k < gm && nmv < 60; k++) {
            double *vk = V + (size_t)k * N, *vn = V + (size_t)(k + 1) * N, *zk = Z + (size_t)k * N;
            memcpy(zk, vk, sizeof(double) * N); kbase(I, zk); I->nsolve++; I->nref++;      /* z_k = K~^-1 v_k, kept */
            kmul(I, zk, w); nmv++;
            for (int i = 0; i <= k; i++) { const double *vi = V + (size_t)i * N; double h = 0; for (int q = 0; q < N; q++) h += w[q] * vi[q]; H[i * gm + k] = h; for (int q = 0; q < N; q++) w[q] -= h * vi[q]; }
            double hn = 0; for (int q = 0; q < N; q++) hn += w[q] * w[q]; hn = sqrt(hn);
            for (int i = 0; i < k; i++) { const double a = H[i * gm + k], b = H[(i + 1) * gm + k]; H[i * gm + k] = cs[i] * a + sn[i] * b; H[(i + 1) * gm + k] = -sn[i] * a + cs[i] * b; }
            const double a = H[k * gm + k], r = hypot(a, hn);
            cs[k] = r > 0 ? a / r : 1; sn[k] = r > 0 ? hn / r : 0; H[k * gm + k] = r;
            g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
            if (!(hn > 0)) { k++; break; }
            for (int q = 0; q < N; q++) vn[q] = w[q] / hn;
            if (fabs(g[k + 1]) <= tin) { k++; break; }
            /* (BRISK_QPGMSTAG) and a cycle whose estimate has not halved in its last three steps: a plateau */
            { static double gst2 = -1; if (gst2 < 0) { const char *e = getenv("BRISK_QPGMSTAG"); gst2 = e ? atof(e) : 1.5; }
              if (gst2 > 0 && k >= 3 && fabs(g[k + 1]) > 0.5 * fabs(g[k - 2])) { k++; break; } }
        }
        if (k == 0) break;
        for (int i = k - 1; i >= 0; i--) { double t = g[i]; for (int j = i + 1; j < k; j++) t -= H[i * gm + j] * yv[j]; yv[i] = H[i * gm + i] != 0 ? t / H[i * gm + i] : 0; }
        double *u = w;                               /* (w is free now) */
        for (int q = 0; q < N; q++) u[q] = 0;
        for (int i = 0; i < k; i++) { const double *zi = Z + (size_t)i * N; for (int q = 0; q < N; q++) u[q] += yv[i] * zi[q]; }   /* K~^-1 V y from the kept z_i: no solve */
        for (int q = 0; q < N; q++) V[q] = I->sol[q] + u[q];      /* (V's first column is free now: the candidate solution) */
        kmul(I, V, u); nmv++;
        for (int q = 0; q < N; q++) u[q] = I->rhs[q] - u[q];
        const double rn2 = nrmi(N, u);
        if (!(rn2 < rn)) break;
        memcpy(I->sol, V, sizeof(double) * N); memcpy(I->res, u, sizeof(double) * N);
        const double ratio = rn / rn2; rn = rn2;
        if (env_on("BRISK_QPREFDBG", &g_refdbg)) qp_printf("       [gmres %d] -> %.2e (x%.2f)\n", k, rn, ratio);
        if (!(rm3 && !I->fullref) && ratio < 2) break;
        /* (23 October 2026; the default from 24 October, r = 1.5; BRISK_QPGMSTAG=0: off) the targets' refinement ends once a whole restart cycle contracts the
         * residual by less than r: GMRES has stagnated at the accuracy the residual's computation allows, and further
         * cycles only spend solves (QPLIB_8500, QP, with BRISK_QPREFMU=1: nine solves of 60 products each at x1.00) */
        { static double gst = -1; if (gst < 0) { const char *e = getenv("BRISK_QPGMSTAG"); gst = e ? atof(e) : 1.5; }
          if (gst > 0 && ratio < gst) break; }
    }
    return 0;
}
static void ksolve_(Ipm *I) {
    const int N = I->n + I->m;
    memcpy(I->sol, I->rhs, sizeof(double) * N);
    kbase(I, I->sol);
    I->nsolve++;
    if (env_on("BRISK_QPNANDBG", &g_nandbg)) {
        int br = 0, bs = 0; double mx = 0; for (int k = 0; k < N; k++) { br += !isfinite(I->rhs[k]); bs += !isfinite(I->sol[k]); mx = fmax(mx, fabs(I->rhs[k])); }
        double dmax = 0, dmin = 1e300; for (int j = 0; j < I->n; j++) { dmax = fmax(dmax, I->Dx[j]); dmin = fmin(dmin, I->Dx[j]); }
        double smax = 0, smin = 1e300; for (int i = 0; i < I->m; i++) if (!I->R->eq[i]) { smax = fmax(smax, I->Ds[i]); smin = fmin(smin, I->Ds[i]); }
        if (br || bs) qp_printf("     [nan] rhs non-finite %d, solution non-finite %d, |rhs| %.1e, Dx in [%.1e, %.1e], Ds in [%.1e, %.1e]\n", br, bs, mx, dmin, dmax, smin, smax);
    }
    const double bn = nrmi(N, I->rhs) + 1e-300;
    kmul(I, I->sol, I->res);
    for (int k = 0; k < N; k++) I->res[k] = I->rhs[k] - I->res[k];
    double rn = nrmi(N, I->res);
    /* (as long as the residual falls: near the end the regularization delta is far larger than an active row's own
     * diagonal Ds^-1 and the refinement contracts slowly; stopped after six steps it left y inconsistent with the slacks'
     * multipliers and the dual residual stalled at 6.6e-8 on QPTEST) */
    static int rmode = -1; if (rmode < 0) { const char *e = getenv("BRISK_QPREFINE"); rmode = e ? atoi(e) : 3; }
    const double rstop = rmode == 1 ? 1e-12 + 1e-13 * bn : rmode == 2 ? 1e-12 * bn : 0;
    /* (BRISK_QPREFINE=3, the default from 22 October 2026: on the 288 QPs 126,508 -> 28,319 refinement steps; 0 as before) an inexact Newton step: the solve's error in the dual (x) and the primal (row) equations
     * goes into the next residuals, (1 - alpha) r + alpha e, so it is refined until it is below a tenth of the present
     * dual and primal residuals (floors 1e-2 tol), not as far as the refinement goes */
    /* and the gap: the error e enters it as x'e_x + y'e_y (UBH1: an error of 1e-11 in the dual equations against x of
     * size 1e5 kept the gap at 1e-8 while mu fell to 1e-69) */
    { static int gm = -1; if (gm < 0) { const char *e = getenv("BRISK_QPGMRES"); gm = e ? atoi(e) : 10; if (gm > 40) gm = 40; }
      if (gm > 0 && (rmode == 0 || rmode == 3)) { kgmres(I, gm, rn, rmode == 3); return; } }
    for (int it = 0; it < (rmode == 1 ? 10 : 60) && rn > rstop; it++) {
        if (rmode == 3 && I->rf_du > 0 && !I->fullref) {
            double eg = 0; for (int j = 0; j < I->n; j++) eg += fabs(I->x[j] * I->res[j]);
            for (int i = 0; i < I->m; i++) eg += fabs(I->y[i] * I->res[I->n + i]);
            if (nrmi(I->n, I->res) <= I->rf_du && nrmi(I->m, I->res + I->n) <= I->rf_pr && eg <= I->rf_gap) break;
        }
        memcpy(I->tmp, I->res, sizeof(double) * N);
        kbase(I, I->tmp); I->nsolve++; I->nref++;
        for (int k = 0; k < N; k++) I->tmp[k] += I->sol[k];
        kmul(I, I->tmp, I->res);
        for (int k = 0; k < N; k++) I->res[k] = I->rhs[k] - I->res[k];
        const double rn2 = nrmi(N, I->res);
        if (!(rn2 < rn)) break;
        memcpy(I->sol, I->tmp, sizeof(double) * N);
        const double ratio = rn / rn2; rn = rn2;
        if ((rmode == 1 && ratio < 5) || (rmode == 2 && ratio < 1.1)) break;     /* contracting too slowly to be worth its solves */
        /* (22 October 2026, a test: BRISK_QPREFSTAG=r) the refinement ends once a step contracts the residual by less than r
         * (QCAPRI, QP: 60 steps a solve at 1.01-1.25 after the full refinement began) */
        { static double stag = -1; if (stag < 0) { const char *e = getenv("BRISK_QPREFSTAG"); stag = e ? atof(e) : 0; }
          if (stag > 0 && ratio < stag) break; }
        if (env_on("BRISK_QPREFDBG", &g_refdbg)) qp_printf("       [refine %d] %.2e -> %.2e (x%.2f)\n", it, rn * ratio, rn, ratio);
    }
}

/* the direction for the complementarity right-hand sides ctl.. (sigma mu - t z - corrections), given the residuals */
static void direction(Ipm *I) {
    const Red *R = I->R; const int n = I->n, m = I->m;
    /* R1 = -rd + (ctl - zl rtl)/tl - (ctu + zu rtu)/tu;  rhs_x = -R1 */
    for (int j = 0; j < n; j++) {
        double r1 = -I->rd[j];
        if (R->hl[j]) r1 += (I->ctl[j] - I->zl[j] * I->rtl[j]) / I->tl[j];
        if (R->hu[j]) r1 -= (I->ctu[j] + I->zu[j] * I->rtu[j]) / I->tu[j];
        I->rhs[j] = -r1;
    }
    /* gs = -rs + (cwl - vl rwl)/wl - (cwu + vu rwu)/wu;  R2 = gs/Ds - rp (inequality), -rp (equality) */
    for (int i = 0; i < m; i++) {
        if (R->eq[i]) { I->rhs[n + i] = -I->rp[i]; continue; }
        double gs = -I->rs[i];
        if (R->gl[i]) gs += (I->cwl[i] - I->vl[i] * I->rwl[i]) / I->wl[i];
        if (R->gu[i]) gs -= (I->cwu[i] + I->vu[i] * I->rwu[i]) / I->wu[i];
        I->tmp[i] = gs;               /* kept for ds below (tmp is reused by ksolve: copied first) */
        I->rhs[n + i] = gs / I->Ds[i] - I->rp[i];
    }
    double *gsv = I->hx;              /* the gs values (hx is free here) */
    for (int i = 0; i < m; i++) gsv[i] = R->eq[i] ? 0 : I->tmp[i];
    ksolve(I);
    for (int j = 0; j < n; j++) I->dx[j] = I->sol[j];
    for (int i = 0; i < m; i++) I->dy[i] = I->sol[n + i];
    /* ds = A dx + rp (inequality rows) */
    for (int i = 0; i < m; i++) I->ds[i] = I->rp[i];
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->ds[R->Ai[p]] += R->Ax[p] * I->dx[j];
    for (int i = 0; i < m; i++) {
        if (R->eq[i]) { I->ds[i] = 0; continue; }
        if (R->gl[i]) { I->dwl[i] = I->ds[i] + I->rwl[i]; I->dvl[i] = (I->cwl[i] - I->vl[i] * I->dwl[i]) / I->wl[i]; }
        if (R->gu[i]) { I->dwu[i] = -I->rwu[i] - I->ds[i]; I->dvu[i] = (I->cwu[i] - I->vu[i] * I->dwu[i]) / I->wu[i]; }
    }
    for (int j = 0; j < n; j++) {
        if (R->hl[j]) { I->dtl[j] = I->dx[j] + I->rtl[j]; I->dzl[j] = (I->ctl[j] - I->zl[j] * I->dtl[j]) / I->tl[j]; }
        if (R->hu[j]) { I->dtu[j] = -I->rtu[j] - I->dx[j]; I->dzu[j] = (I->ctu[j] - I->zu[j] * I->dtu[j]) / I->tu[j]; }
    }
}

static double maxstep(const double *v, const double *dv, const char *on, int k) {
    double a = 1e300;
    for (int i = 0; i < k; i++) if (on[i] && dv[i] < 0) { const double t = -v[i] / dv[i]; if (t < a) a = t; }
    return a;
}
static double step_to_boundary(const Ipm *I) {
    const Red *R = I->R; const int n = I->n, m = I->m;
    double a = 1e300;
    a = fmin(a, maxstep(I->tl, I->dtl, R->hl, n)); a = fmin(a, maxstep(I->zl, I->dzl, R->hl, n));
    a = fmin(a, maxstep(I->tu, I->dtu, R->hu, n)); a = fmin(a, maxstep(I->zu, I->dzu, R->hu, n));
    a = fmin(a, maxstep(I->wl, I->dwl, R->gl, m)); a = fmin(a, maxstep(I->vl, I->dvl, R->gl, m));
    a = fmin(a, maxstep(I->wu, I->dwu, R->gu, m)); a = fmin(a, maxstep(I->vu, I->dvu, R->gu, m));
    return a;
}

static void hmul_red(const Red *R, const double *x, double *y) {
    for (int j = 0; j < R->n; j++) y[j] = 0;
    for (int j = 0; j < R->n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) {
        const int i = R->Hi[p]; const double v = R->Hx[p];
        y[i] += v * x[j]; if (i != j) y[j] += v * x[i];
    }
}
static void compute_residuals(Ipm *I) {
    const Red *R = I->R; const int n = I->n, m = I->m;
    hmul_red(R, I->x, I->hx);
    for (int j = 0; j < n; j++) {
        double r = I->hx[j] + R->c[j];
        if (R->hl[j]) r -= I->zl[j];
        if (R->hu[j]) r += I->zu[j];
        I->rd[j] = r;
    }
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->rd[j] -= R->Ax[p] * I->y[R->Ai[p]];
    for (int i = 0; i < m; i++) I->rp[i] = 0;
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->rp[R->Ai[p]] += R->Ax[p] * I->x[j];
    for (int i = 0; i < m; i++) {
        if (R->eq[i]) { I->rp[i] -= R->rl[i]; I->rs[i] = 0; continue; }
        I->rp[i] -= I->s[i];
        double r = I->y[i];
        if (R->gl[i]) { r -= I->vl[i]; I->rwl[i] = 0; }
        if (R->gu[i]) { r += I->vu[i]; I->rwu[i] = 0; }
        I->rs[i] = r;
    }
    for (int j = 0; j < n; j++) {
        if (R->hl[j]) I->rtl[j] = 0;          /* (the slacks are x - xl and xu - x exactly) */
        if (R->hu[j]) I->rtu[j] = 0;
    }
}
static double complementarity(const Ipm *I) {
    const Red *R = I->R; double s = 0;
    for (int j = 0; j < I->n; j++) { if (R->hl[j]) s += I->tl[j] * I->zl[j]; if (R->hu[j]) s += I->tu[j] * I->zu[j]; }
    for (int i = 0; i < I->m; i++) { if (R->gl[i]) s += I->wl[i] * I->vl[i]; if (R->gu[i]) s += I->wu[i] * I->vu[i]; }
    return s;
}
/* the complementarity after a step of length a along the current direction */
static double compl_after(const Ipm *I, double a) {
    const Red *R = I->R; double s = 0;
    for (int j = 0; j < I->n; j++) {
        if (R->hl[j]) s += (I->tl[j] + a * I->dtl[j]) * (I->zl[j] + a * I->dzl[j]);
        if (R->hu[j]) s += (I->tu[j] + a * I->dtu[j]) * (I->zu[j] + a * I->dzu[j]);
    }
    for (int i = 0; i < I->m; i++) {
        if (R->gl[i]) s += (I->wl[i] + a * I->dwl[i]) * (I->vl[i] + a * I->dvl[i]);
        if (R->gu[i]) s += (I->wu[i] + a * I->dwu[i]) * (I->vu[i] + a * I->dvu[i]);
    }
    return s;
}
/* the point of the problem as given from the iteration's (unscaled, the fixed columns back, their multipliers from the
 * stationarity) */
static void extract(const QpProb *P, const Red *R, const Ipm *I, const double *fixval, QpRes *Res) {
    const int n0 = P->n, n = R->n, m = R->m;
    for (int j = 0; j < n0; j++) Res->x[j] = fixval[j];
    for (int k = 0; k < n; k++) {
        const int j = R->col0[k];
        Res->x[j] = R->bs * R->D[k] * I->x[k];
        double z = (R->hl[k] ? I->zl[k] : 0) - (R->hu[k] ? I->zu[k] : 0);
        Res->z[j] = z / (R->cs * R->bs * R->D[k]);
    }
    for (int k = 0; k < m; k++) Res->y[R->row0[k]] = R->E[k] * I->y[k] / (R->cs * R->bs);
    for (int t = 0; t < R->ns; t++) Res->y[R->srow[t]] = 0;          /* (extract runs every iteration: no stale value) */
    /* the fixed columns' multipliers: z_j = (H x + c - A'y)_j */
    {
        double *hx = xz(sizeof(double) * (n0 + 1)); qp_hmul(P, Res->x, hx);
        for (int j = 0; j < n0; j++) {
            int kept = 0; (void)kept;
            const double xlj = R->xlp ? R->xlp[j] : P->xl[j], xuj = R->xup ? R->xup[j] : P->xu[j];
            if (fin(xlj) && fin(xuj) && xuj - xlj <= 1e-12 * (1 + fabs(xlj))) {
                double r = hx[j] + P->c[j];
                for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) r -= P->Ax[p] * Res->y[P->Ai[p]];
                Res->z[j] = r;
            }
        }
        free(hx);
    }
    /* the singleton rows: a column's multiplier z_j belongs to the bound that binds, the column's own or a row's (then
     * y_i = z_j / a and z_j = 0); ties to the column's own bound */
    for (int t = 0; t < R->ns; t++) {
        const int i = R->srow[t], j = R->scol[t]; const double a = R->sa[t], z = Res->z[j];
        if (getenv("BRISK_QPSDBG")) qp_printf("     [singleton row %d: column %d, a %g, z %g, rl %g ru %g, xl %g xu %g -> %g %g, x %g]\n", i, j, a, z, P->rl[i], P->ru[i], P->xl[j], P->xu[j], R->xlp[j], R->xup[j], Res->x[j]);
        if (z == 0) continue;
        const double lo = fin(P->rl[i]) ? P->rl[i] / a : (a > 0 ? -QP_INF : QP_INF), hi = fin(P->ru[i]) ? P->ru[i] / a : (a > 0 ? QP_INF : -QP_INF);
        const double l = a > 0 ? lo : hi, u = a > 0 ? hi : lo;
        const double tl = 1e-12 * (1 + fabs(R->xlp[j])), tu = 1e-12 * (1 + fabs(R->xup[j]));
        if (z > 0 && fin(l) && fabs(l - R->xlp[j]) <= tl && !(fin(P->xl[j]) && P->xl[j] >= l - tl)) { Res->y[i] = z / a; Res->z[j] = 0; }
        else if (z < 0 && fin(u) && fabs(u - R->xup[j]) <= tu && !(fin(P->xu[j]) && P->xu[j] <= u + tu)) { Res->y[i] = z / a; Res->z[j] = 0; }
    }
}

/* (22 October 2026) the nested dissection of sparsechol.c as a candidate order for the QP method's system (it runs only
 * where AMD's flops are large for the graph's edges, within a time budget). BRISK_QPND: 0 never; 1 (the default) on the
 * system chosen, after the choice between the normal equations and the augmented system was made on AMD's counts, with
 * a budget of a fifth of the factorizations a QP solve takes (25: five factorizations' worth); 2 in every analysis of the
 * augmented system with the LP method's budget (the first test: QPLIB_8500 spent 16.6 s on the dissection of a system
 * that the normal equations then replaced). On the 17 of the 288 QPs where it runs: the shifted geometric mean of the
 * time 6.00 -> 5.34 s (QPLIB_8602 66.3 -> 47.6 s, the CVXQP*_L family 1.5-1.7 times faster; NNLS_100, whose augmented
 * system is nearly dense, 26.4 -> 29.3 s), the same statuses and iterations on all 288 */
static int qp_nd_now = 0;
static int k_build_offsets(Ipm *I);
/* (24 October 2026, the SOCP developer) the largest factor the machine can hold: its physical memory at 12 bytes an entry
 * (the value and its row index), or the process's own memory limit when smaller; an analysis beyond it fails, and the method says so, instead of the allocation stopping
 * the process ("out of memory": QFLOW_100000, QP, a network flow on a random graph whose factor fills at any order) */
static size_t qp_fill_cap(void) {
    static size_t cap = 0;
    if (!cap) { const long pg = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGESIZE); double mem = pg > 0 && ps > 0 ? (double)pg * ps : 8e9;
        struct rlimit rl; if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) mem = fmin(mem, (double)rl.rlim_cur);   /* (a process's own limit, ulimit -v) */
        cap = (size_t)fmin(mem / 12.0, (double)((size_t)1 << 32)); }
    return cap;
}
/* the nested dissection's budget in factorizations' worth (BRISK_QPNDB, a test) */
static double qp_nd_budget(void) { static double b = -1; if (b < 0) { const char *e = getenv("BRISK_QPNDB"); b = e ? atof(e) : 5.0; } return b; }
static int qp_nd_mode(void) { static int m_ = -1; if (m_ < 0) { const char *e = getenv("BRISK_QPND"); m_ = e ? atoi(e) : 1; } return m_; }

/* the augmented system's analysis and value offsets (I->K; 0, or 1 when the analysis fails) */
static int k_build(Ipm *I) {
    Red *R = I->R; const int n = I->n, m = I->m, N = n + m;
    /* the pattern of K: H's off-diagonal entries and A's entries */
    int *deg = xz(sizeof(int) * (N + 1));
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) if (R->Hi[p] != j) { deg[j]++; deg[R->Hi[p]]++; }
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { deg[j]++; deg[n + R->Ai[p]]++; }
    int **nbr = xm(sizeof(int *) * (N + 1)); long tot = 0; for (int k = 0; k < N; k++) tot += deg[k];
    int *pool = xm(sizeof(int) * (tot + 1)); { long o = 0; for (int k = 0; k < N; k++) { nbr[k] = pool + o; o += deg[k]; deg[k] = 0; } }
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) { const int i = R->Hi[p]; if (i != j) { nbr[j][deg[j]++] = i; nbr[i][deg[i]++] = j; } }
    for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int i = n + R->Ai[p]; nbr[j][deg[j]++] = i; nbr[i][deg[i]++] = j; }
    /* (the nested dissection of sparsechol.c tried against AMD when the caller asks: qp_nd_mode below) */
    if (qp_nd_now) schol_set_nd(1);
    /* (22 October 2026) AMD alone, as for the normal equations: on the 288 QPs the own orderings of sparsechol.c never gave
     * fewer flops than AMD (3 ties, 15 failures at the flop cap), and a failure costs about five factorizations' worth
     * (their explicit elimination graph's work grows with the flops they reach: CVXQP1_L 2.4 s, QPLIB_8567 2.1 s of a 7 s
     * solve). BRISK_QPOWN=1: as before */
    static int qown = -1; if (qown < 0) { const char *e = getenv("BRISK_QPOWN"); qown = e && atoi(e) == 1; }
    const int ao_ = qown ? -2 : schol_set_amd(3);
    I->K = N > 0 ? schol_analyze_adj(N, deg, nbr, qp_fill_cap()) : NULL;
    if (ao_ != -2) schol_set_amd(ao_);
    schol_set_nd(0);
    I->K = qp_amalg(I->K, N);
    free(deg); free(nbr); free(pool);
    if (N > 0 && !I->K) return 1;
    return k_build_offsets(I);
}

/* (23 October 2026, the night, the default; BRISK_QPAMALG=0: exact supernodes; =f > 0 and < 1: that fraction whatever the
 * factor) a factor with long columns
 * (at least 100 entries of L per unknown) analysed again, its ordering kept, with its supernodes relaxed (a node merged
 * into its parent while the explicit zeros stay below 0.3 of the merged panel, 64 columns at most), the SOCP solver's rule:
 * the QP method ran on the exact supernodes, narrow on its augmented systems (QPLIB_8602, QP: 18 factorizations of 8.8e9
 * flops in 37.6 s, 4.2 GF/s; relaxed 10.0 s, 47.6 -> 22.7 s in total); where the columns are short (ISO_100000, OBST_300:
 * 2 and 33 entries per unknown) the merge cost 4-10 %. On the 299 QPs no status changed, the iterations the same (but
 * QPLIB_8602 17 -> 18); paired times (one thread): QPLIB_8567 and CVXQP3_L 3.7 -> 2.3-2.5 s, QPLIB_8559 2.8 -> 1.9, NNLS_50
 * 5.1 -> 3.9 (4.1 by the rule), NNLS_100 25.9 -> 19.6, QPLIB_8785 2.3 -> 1.8; QPLIB_8500 (108 entries per unknown) 28.5 ->
 * 30.4. An analysis without its own ordering (a dense factor) is left as it is */
static SChol *qp_amalg(SChol *S, int nunk) {
    static double qam = -1; if (qam < 0) { const char *e = getenv("BRISK_QPAMALG"); qam = e ? atof(e) : 1; }
    if (!S || !(qam > 0) || nunk <= 0) return S;
    const double f = qam >= 1 ? 0.3 : qam;
    if (qam >= 1 && (double)schol_nnz(S) < 100.0 * nunk) return S;
    return schol_reanalyze_amalg(S, f);
}
/* the augmented system's signs and value offsets for the analysis in I->K */
static int k_build_offsets(Ipm *I) {
    Red *R = I->R; const int n = I->n, m = I->m, N = n + m;
    if (!I->sgn) { I->sgn = xm(N + 1); for (int k = 0; k < N; k++) I->sgn[k] = k < n ? -1 : 1; }
    if (I->K) { schol_set_signs(I->K, I->sgn); schol_set_dynpiv(I->K, 1e-8); }
    if (!I->offHd) {
        I->offHd = xm(sizeof(size_t) * (n + 1)); I->offH = xm(sizeof(size_t) * ((size_t)R->Hp[n] + 1));
        I->offA = xm(sizeof(size_t) * ((size_t)R->Ap[n] + 1)); I->offR = xm(sizeof(size_t) * (m + 1));
    }
    if (I->K) {
        for (int j = 0; j < n; j++) I->offHd[j] = schol_offset(I->K, j, j);
        for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) I->offH[p] = schol_offset(I->K, R->Hi[p], j);
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->offA[p] = schol_offset(I->K, n + R->Ai[p], j);
        for (int i = 0; i < m; i++) I->offR[i] = schol_offset(I->K, n + i, n + i);
    }
    return 0;
}

/* the normal equations given up for the augmented system, analysed now when it was not at the start (BRISK_QPNEFALL=0:
 * never); returns 1 when the augmented system is in use */
static int ne_to_augmented(Ipm *I, int *k_lazy) {
    static int fall = -1; if (fall < 0) { const char *e = getenv("BRISK_QPNEFALL"); fall = e ? atoi(e) : 1; }
    if (!fall) return 0;
    if (!I->Kspare && *k_lazy) {         /* (not analysed at the start: now) */
        SChol *Ms = I->M; I->M = NULL;
        if (k_build(I) == 0) I->Kspare = I->K; else if (I->K) { schol_free(I->K); }
        I->K = NULL; I->M = Ms; *k_lazy = 0;
    }
    if (!I->Kspare) return 0;
    schol_free(I->M); I->M = NULL; I->K = I->Kspare; I->Kspare = NULL;
    return 1;
}
static int qp_solve_ipm_once(const QpProb *P, const QpOpts *opt, QpRes *Res);
/* (23 October 2026, the SOCP developer) the interior-point method, and once more with other dependent rows dropped when the
 * first solve's largest primal error is on a dropped row. A dropped row's residual is the combination of its kept rows'
 * residuals; those are at rounding level against their own terms, so a dropped row whose terms are small against theirs
 * inherits an error of order one against its own (QPLIB_9002, QP: flows to 8e9, the dropped rows' terms 1e-6 to 1e-2, the
 * primal error stuck at 1e-6 while the scaled problem converged). The second solve drops, in each dependence, the row
 * with the largest terms at the first solve's point (qp_dep_rows' priorities); the better result is returned */
int qp_solve_ipm(const QpProb *P, const QpOpts *opt, QpRes *Res) {
    const double t0 = now();
    const int m0 = P->m, n0 = P->n;
    char *dead = calloc((size_t)m0 + 1, 1);
    g_qp_dead = dead;
    int st = qp_solve_ipm_once(P, opt, Res);
    g_qp_dead = NULL;
    int any = 0; for (int i = 0; i < m0 && dead; i++) any |= dead[i];
    if (dead && any && Res->x && (st == QP_REDUCED || st == QP_NUMERICAL || st == QP_MAXIT)) {
        /* the rows' terms at the point, and the row of the largest primal error (as qp_errors measures it) */
        double *ax = calloc((size_t)m0 + 1, sizeof(double)), *at = calloc((size_t)m0 + 1, sizeof(double));
        for (int j = 0; j < n0; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ax[P->Ai[p]] += P->Ax[p] * Res->x[j]; at[P->Ai[p]] += fabs(P->Ax[p] * Res->x[j]); }
        double ew = 0; int iw = -1;
        for (int i = 0; i < m0; i++) {
            double v = 0;
            if (fin(P->rl[i]) && ax[i] < P->rl[i]) v = (P->rl[i] - ax[i]) / (1 + fmax(fabs(P->rl[i]), at[i]));
            if (fin(P->ru[i]) && ax[i] > P->ru[i]) v = (ax[i] - P->ru[i]) / (1 + fmax(fabs(P->ru[i]), at[i]));
            if (v > ew) { ew = v; iw = i; }
        }
        if (iw >= 0 && dead[iw]) {
            if (opt && opt->verbose) qp_printf("the largest primal error (%.1e) on row %d, dropped as dependent: solved again with the dependent rows chosen by their activity\n", ew, iw);
            QpRes R2; memset(&R2, 0, sizeof R2);
            g_qp_rowprio = at;
            const int st2 = qp_solve_ipm_once(P, opt, &R2);
            g_qp_rowprio = NULL;
            const double e1 = fmax(Res->pres, fmax(Res->dres, Res->gap)), e2 = fmax(R2.pres, fmax(R2.dres, R2.gap));
            const double t1 = Res->time;
            if (st2 == QP_OPTIMAL || (st2 == QP_REDUCED && e2 < e1)) { qp_res_free(Res); *Res = R2; st = st2; }
            else qp_res_free(&R2);
            Res->resolves = 1; Res->t_first = t1;
        }
        free(ax); free(at);
    }
    free(dead);
    Res->time = now() - t0;
    return st;
}
static int qp_solve_ipm_once(const QpProb *P, const QpOpts *opt, QpRes *Res) {
    const double t0 = now();
    QpOpts od; if (!opt) { qp_default_opts(&od); opt = &od; }
    const int verbose = opt->verbose, maxit = opt->maxit > 0 ? opt->maxit : 200;
    const double tol = opt->tol > 0 ? opt->tol : 1e-8;
    memset(Res, 0, sizeof(*Res));
    const int n0 = P->n, m0 = P->m;
    Res->x = xz(sizeof(double) * (n0 + 1)); Res->y = xz(sizeof(double) * (m0 + 1)); Res->z = xz(sizeof(double) * (n0 + 1));
    double *fixval = xz(sizeof(double) * (n0 + 1)), c0add = 0;
    Red Rd; Red *R = &Rd;
    const int rrc = reduce(P, R, fixval, &c0add);
    if (rrc == 2) {
        const int j = abs(g_qp_ray) - 1; Res->x[j] = g_qp_ray > 0 ? 1.0 : -1.0;
        Res->status = QP_DUAL_INFEASIBLE; Res->time = now() - t0; free(fixval);
        if (verbose) qp_printf("presolve: column %s has no entries in the rows or the quadratic term, a cost of %.3g and no bound in its descent\n  direction: unbounded (the solution holds the direction)\n",
                               P->cname && P->cname[j] ? P->cname[j] : "?", P->c[j]);
        return Res->status;
    }
    if (rrc) {
        Res->status = QP_PRIMAL_INFEASIBLE; Res->time = now() - t0; free(fixval);
        if (verbose) qp_printf("presolve: a row or a column with contradictory bounds: primal infeasible\n");
        return Res->status;
    }
    const int n = R->n, m = R->m;
    { const char *e = getenv("BRISK_QPNOSCALE"); equilibrate(R, e ? 0 : 20); }
    if (verbose) {
        long nh = R->Hp[n], na = R->Ap[n]; int neq = 0, nb = 0; for (int i = 0; i < m; i++) neq += R->eq[i]; for (int j = 0; j < n; j++) nb += R->hl[j] + R->hu[j];
        qp_printf("BRISK QP interior-point method: n = %d, m = %d (%d equalities), %ld entries in H (lower), %ld in A, %d finite bounds; %d fixed columns substituted, %d rows without bounds or entries dropped",
               n, m, neq, nh, na, nb, n0 - n, m0 - m - (g_qp_ndep > 0 ? g_qp_ndep : 0));
        if (g_qp_ndep > 0) qp_printf(", %d equality rows dropped as linear combinations of others", g_qp_ndep);
        qp_printf("\n");
        double dmn = 1e300, dmx = 0, emn = 1e300, emx = 0;
        for (int j = 0; j < n; j++) { dmn = fmin(dmn, R->D[j]); dmx = fmax(dmx, R->D[j]); }
        for (int i = 0; i < m; i++) { emn = fmin(emn, R->E[i]); emx = fmax(emx, R->E[i]); }
        qp_printf("scaling: columns [%.1e, %.1e], rows [%.1e, %.1e], primal variables %.1e, cost %.1e\n", dmn, dmx, emn, emx, R->bs, R->cs);
    }
    Ipm I_; Ipm *I = &I_; memset(I, 0, sizeof(*I));
    I->R = R; I->n = n; I->m = m;
    const int N = n + m;
    I->N = 0; for (int j = 0; j < n; j++) I->N += R->hl[j] + R->hu[j]; for (int i = 0; i < m; i++) I->N += R->gl[i] + R->gu[i];
    double **vecs[] = { &I->x, &I->dx, &I->tl, &I->tu, &I->zl, &I->zu, &I->dtl, &I->dtu, &I->dzl, &I->dzu, &I->rd, &I->rtl, &I->rtu, &I->ctl, &I->ctu, &I->Dx, &I->Hdiag };
    for (size_t k = 0; k < sizeof(vecs) / sizeof(vecs[0]); k++) *vecs[k] = xz(sizeof(double) * (n + 1));
    double **vecr[] = { &I->y, &I->dy, &I->s, &I->ds, &I->wl, &I->wu, &I->vl, &I->vu, &I->dwl, &I->dwu, &I->dvl, &I->dvu, &I->rs, &I->rp, &I->rwl, &I->rwu, &I->cwl, &I->cwu, &I->Ds };
    for (size_t k = 0; k < sizeof(vecr) / sizeof(vecr[0]); k++) *vecr[k] = xz(sizeof(double) * (m + 1));
    I->rhs = xz(sizeof(double) * (N + 1)); I->sol = xz(sizeof(double) * (N + 1)); I->res = xz(sizeof(double) * (N + 1));
    I->work = xz(sizeof(double) * (2 * (size_t)N + 2 * (size_t)m + 2)); I->tmp = xz(sizeof(double) * (N + 1)); I->hx = xz(sizeof(double) * (N + 1));
    I->sv = xz(sizeof(double) * (7 * (size_t)n + 8 * (size_t)m + 1));
    for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) if (R->Hi[p] == j) I->Hdiag[j] += R->Hx[p];
    const double ta = now();
    /* the triangular solves in sequence: the task-parallel solve allocates per node, and a QP's KKT factor has mostly
     * nodes of one column (BOYD1: 93,279 unknowns; 7 s an iteration with the tasks) */
    { static int seq = -1; if (seq < 0) { const char *e = getenv("BRISK_QPSOLVEPAR"); seq = !(e && atoi(e)); } schol_set_solve_seq(seq); }
    /* the augmented system K is built here unless the normal equations are chosen below without it (22 October 2026:
     * when M's factor is at most twice K's own entries, a lower bound of K's factor; QPLIB_9008, a million variables:
     * K's analysis ran out of memory at 8 GB, and PORTF_100's took 9 s for a factor not used) */
    int k_lazy = 0;
    /* (22 October 2026) the normal equations when H is diagonal: M = A (H + Dx)^-1 A' + E is positive definite, needs no
     * regularization for a column with a bound or a quadratic term nor for an equality row, and a tiny pivot is replaced
     * by a huge one as in the LP method (the direction's component along a dependent row dropped). The augmented system's
     * regularization 1e-8 on the columns and the equality rows had to be undone by the refinement, which contracts slowly
     * where it dominates: QPLIB_10034 (40,200 equality rows) stalled at a primal residual of 3.5e-9 with 4,632 refinement
     * steps. Chosen when H is diagonal, no column has more than 200 entries and M's factor is at most twice the augmented
     * system's (BRISK_QPNE=0: never; 2: whenever possible) */
    {
        static int nemode = -1; if (nemode < 0) { const char *e = getenv("BRISK_QPNE"); nemode = e ? atoi(e) : 1; }
        int diag = 1, maxc = 0;
        for (int j = 0; j < n && diag; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) if (R->Hi[p] != j) { diag = 0; break; }
        /* (22 October 2026) columns of more than 200 entries are dense: kept out of M and added back by the Woodbury
         * formula (as the SOCP solver's dense linear columns), at most 300 of them, and only when every equality row has
         * an entry in a sparse column (M_s then stays positive definite: an inequality row has Ds^-1 > 0). An SVM with
         * 100 dense features (10,000 samples): the augmented system's factor had 2 million entries, M_s is diagonal.
         * BRISK_QPNEDENSE=0: the normal equations refused when a column has more than 200 entries, as before */
        static int nedense = -1; if (nedense < 0) { const char *e = getenv("BRISK_QPNEDENSE"); nedense = e ? atoi(e) : 1; }
        char *isd = xz(n + 1); int nd = 0;
        for (int j = 0; j < n; j++) { const int c = R->Ap[j + 1] - R->Ap[j]; if (c > 200) { isd[j] = 1; nd++; } else if (c > maxc) maxc = c; }
        if (nd && nedense && nd <= 300) {
            char *rs = xz(m + 1);
            for (int j = 0; j < n; j++) if (!isd[j]) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) rs[R->Ai[p]] = 1;
            for (int i = 0; i < m; i++) if (R->eq[i] && !rs[i]) { nd = -1; break; }
            free(rs);
        } else if (nd) nd = -1;
        /* (25 October 2026) when the dense columns cannot be kept apart (more than 300, or an equality row only in dense
         * columns), all of them go into M, and the comparison of M's factor with the augmented system's entries below
         * decides, as for any other M: a column of more than 200 entries is dense only against a large m. The step of the
         * sequential quadratic programming on PORTMR_5000x20x30 (QCQP; 621 rows, 5,600 columns of about 200 entries
         * each): the augmented system's factor took 0.22 s at each of 13 factorizations, M is a dense 621 x 621 matrix.
         * BRISK_QPNEDENSE=2: the normal equations refused in these cases, as before */
        if (nd < 0 && nedense == 1) {
            /* (not when M is sure to be refused: a column of c entries makes a clique, nnz(L) >= c (c + 1) / 2, against the
             * comparison's bound below, twice the augmented system's own entries; DUALC8, QP: 8 columns of 503 entries, M's
             * pattern of 1e6 pairs built for nothing, 18.5 -> 24.1 ms) */
            long cmx = 0; for (int j = 0; j < n; j++) { const long c_ = R->Ap[j + 1] - R->Ap[j]; if (c_ > cmx) cmx = c_; }
            if (0.5 * (double)cmx * (cmx + 1) <= 2.0 * ((double)n + m + R->Ap[n] + R->Hp[n])) {
                if (verbose >= 2) qp_printf("  [dense columns all in the normal equations]\n");
                nd = 0; memset(isd, 0, n);
            }
        }
        const int ne_ok = nemode && diag && nd >= 0 && m > 0;
        if (!ne_ok) { if (k_build(I)) { if (verbose) qp_printf("the factorization's analysis failed (the fill exceeds its cap)\n"); free(isd); Res->status = QP_NUMERICAL; goto done; } }
        if (ne_ok) {
            /* (25 October 2026) M by dense products when they are cheaper, the LP method's cost model: the row-by-row
             * assembly costs sum_j c_j (c_j + 1) / 2 scattered updates (c_j: column j's entries), dsyrk n m (m + 1) / 2
             * flops at BLAS-3 speed, up to 20 times the rate of a scattered update (below); at most 8192 rows (a dense M of 512 MB),
             * no dense columns kept apart. M's pattern is then full (building it entry by entry costs as much as the
             * assembly). The step of the sequential quadratic programming on PORTMR_5000x20x30 (QCQP; 621 rows, 5,600
             * columns of about 200 entries): 1.1e8 updates an iteration, and as many for the pattern. BRISK_QPDASM=0/1:
             * off/forced (within 8192 rows and without dense columns) */
            int dasm = 0;
            {   double ws = 0; for (int j = 0; j < n; j++) { const double l = R->Ap[j + 1] - R->Ap[j]; ws += 0.5 * l * (l + 1); }
                const double wd = 0.5 * (double)n * (double)m * (m + 1.0);
                const char *e = getenv("BRISK_QPDASM");
                /* (the dense flops' rate against a scattered update's grows with the rows, measured at equal work on this
                 * host: 0.7 at 2 rows, 1.4 at 8, 2.1 at 22, 4.1 at 52, 9.9 at 128; about m / 13, and 20 at about 300 rows,
                 * the LP method's measurement: so m / 13, at most 20. With 20 at any m, HUES-MOD, QP, 2 rows, lost 7 ms in 60) */
                dasm = nd == 0 && m <= 8192 && (e ? atoi(e) != 0 : wd < fmin(20.0, m / 13.0) * ws);
                if (dasm && verbose >= 2) qp_printf("  M by dense products (%.3g flops against %.3g scattered updates)\n", wd, ws); }
            /* M's pattern: rows sharing a sparse column */
            int *rp = xz(sizeof(int) * (m + 2)), *rj;
            for (int j = 0; j < n; j++) if (!isd[j]) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) rp[R->Ai[p] + 1]++;
            for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
            rj = xm(sizeof(int) * ((size_t)rp[m] + 1));
            { int *f = xm(sizeof(int) * (m + 1)); memcpy(f, rp, sizeof(int) * m); for (int j = 0; j < n; j++) if (!isd[j]) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) rj[f[R->Ai[p]]++] = j; free(f); }
            /* the unknowns: the rows (0..m-1), then the dense columns (m..m+nd-1) */
            const int nd0 = nd > 0 ? nd : 0, mm = m + nd0;
            int *dpos = xm(sizeof(int) * (n + 1)); { int a = 0; for (int j = 0; j < n; j++) dpos[j] = isd[j] ? m + a++ : -1; }
            int *mk = xm(sizeof(int) * (mm + 1)), *dg = xz(sizeof(int) * (mm + 1)); for (int i = 0; i < m; i++) mk[i] = -1;
            long tn = 0;
            if (dasm) { for (int i = 0; i < m; i++) dg[i] = m - 1; tn = (long)m * (m - 1); }
            else for (int i = 0; i < m; i++) { mk[i] = i; for (int q = rp[i]; q < rp[i + 1]; q++) { const int j = rj[q]; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int k = R->Ai[p]; if (mk[k] != i) { mk[k] = i; dg[i]++; } } } tn += dg[i]; }
            for (int j = 0; j < n; j++) if (dpos[j] >= 0) { const int c = R->Ap[j + 1] - R->Ap[j]; dg[dpos[j]] += c; tn += 2 * c; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) dg[R->Ai[p]]++; }
            int **nb = xm(sizeof(int *) * (mm + 1)), *pl = xm(sizeof(int) * (tn + 1));
            { long o = 0; for (int i = 0; i < mm; i++) { nb[i] = pl + o; o += dg[i]; dg[i] = 0; } for (int i = 0; i < m; i++) mk[i] = -1; }
            if (dasm) { for (int i = 0; i < m; i++) for (int k = 0; k < m; k++) if (k != i) nb[i][dg[i]++] = k; }
            else for (int i = 0; i < m; i++) { mk[i] = i; for (int q = rp[i]; q < rp[i + 1]; q++) { const int j = rj[q]; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int k = R->Ai[p]; if (mk[k] != i) { mk[k] = i; nb[i][dg[i]++] = k; } } } }
            for (int j = 0; j < n; j++) if (dpos[j] >= 0) { const int d = dpos[j]; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int i = R->Ai[p]; nb[d][dg[d]++] = i; nb[i][dg[i]++] = d; } }
            free(dpos);
            /* (BRISK_QPCG=2, a test: the conjugate gradients wherever every column has at most two entries) */
            { const char *e = getenv("BRISK_QPCG"); const int cg2 = e && atoi(e) == 2;      /* (read at each solve: the test sets it) */
              if (cg2 && nd <= 0 && ne_pcg_setup(I)) {
                  if (verbose) qp_printf("normal equations: conjugate gradients preconditioned by the spanning tree of the columns' weights over the matrix's\n  diagonal (BRISK_QPCG=2)\n");
                  free(rp); free(rj); free(mk); free(dg); free(nb); free(pl);
                  goto pcg_ready;
              } }
            const double tm0 = now();
            if (verbose >= 2) qp_printf("  the augmented system's analysis %.3f s; M's pattern %.3f s\n", tm0 - ta, 0.0);
            /* (22 October 2026) M analysed under the fill the choice below accepts: twice K's own entries (K then not
             * analysed at all); beyond it K is analysed and M again under twice K's factor. Uncapped, M's analysis ran to
             * the end before the comparison: QPLIB_8785 (QP), AMD's 1.6e10 flops against K's 1.9e9, 5.1 s of an analysis
             * thrown away (BRISK_QPNECAP=0: uncapped, as before) */
            static int necap = -1; if (necap < 0) { const char *e = getenv("BRISK_QPNECAP"); necap = !(e && atoi(e) == 0); }
            size_t ktril = (size_t)n + m + R->Ap[n]; for (int j = 0; j < n; j++) for (int p = R->Hp[j]; p < R->Hp[j + 1]; p++) ktril += R->Hi[p] != j;
            /* (AMD alone under the cap, as the LP method's cost model: when AMD stops at the cap the own orderings would
             * run on to it, 7.6 s on QPLIB_8785) */
            const int amd_old = (necap && nemode != 2) ? schol_set_amd(3) : -2;
            SChol *Mc = schol_analyze_adj(mm, dg, nb, (necap && nemode != 2) ? 2 * ktril : (size_t)1 << 32);
            if (verbose >= 2) qp_printf("  M's analysis %.3f s%s\n", now() - tm0, Mc ? "" : " (beyond twice the augmented system's own entries)");
            {   /* K's own entries (its lower triangle): a lower bound of its factor */
                if (Mc && (nemode == 2 || schol_nnz(Mc) <= 2 * ktril)) k_lazy = 1;
                else if (k_build(I)) {
                    if (I->K) { schol_free(I->K); I->K = NULL; }
                    const char *ecg = getenv("BRISK_QPCG"); const int cgon = ecg ? atoi(ecg) : 1;   /* (0: never) */
                    if (!Mc && nd <= 0 && cgon && ne_pcg_setup(I)) {
                        if (verbose) qp_printf("normal equations: no factor fits (the fill exceeds its cap); every column has at most two entries: conjugate\n"
                                               "  gradients preconditioned by the spanning tree of the columns' weights over the matrix's diagonal\n");
                        free(rp); free(rj); free(mk); free(dg); free(nb); free(pl);
                        goto pcg_ready;
                    }
                    if (verbose) qp_printf("the factorization's analysis failed (the fill exceeds its cap)\n");
                    if (Mc) schol_free(Mc);
                    free(rp); free(rj); free(mk); free(dg); free(nb); free(pl); free(isd); Res->status = QP_NUMERICAL; goto done; }
                else if (!Mc && necap && I->K && schol_nnz(I->K) > ktril) {
                    const double tm1 = now();
                    Mc = schol_analyze_adj(mm, dg, nb, 2 * schol_nnz(I->K));
                    if (verbose >= 2) qp_printf("  M's analysis under twice the augmented system's factor %.3f s%s\n", now() - tm1, Mc ? "" : " (beyond it)");
                }
            }
            if (amd_old != -2) schol_set_amd(amd_old);
            Mc = qp_amalg(Mc, mm);
            /* (22 October 2026) the choice by the factorizations' flops, the cost of each iteration, not by their entries:
             * QPLIB_8602 without its duplicate rows took M for 66.9 against 34.6 million entries, 1.7e11 against 5.2e10 flops
             * (BRISK_QPNEFLOPS=0: by the entries, as before) */
            static int nefl = -1; if (nefl < 0) { const char *e = getenv("BRISK_QPNEFLOPS"); nefl = !(e && atoi(e) == 0); }
            if (Mc && (nemode == 2 || k_lazy || (nefl ? schol_flops(Mc) <= 2 * schol_flops(I->K) : schol_nnz(Mc) <= 2 * schol_nnz(I->K)))) {
                if (qp_nd_mode() == 1) {          /* the dissection on the system chosen (above) */
                    const int ao = schol_set_amd(3); schol_set_nd(1); schol_set_nd_budget(qp_nd_budget());
                    const double tn = now();
                    SChol *M2 = schol_analyze_adj(mm, dg, nb, schol_nnz(Mc) + (size_t)mm);
                    schol_set_nd(0); schol_set_nd_budget(0); schol_set_amd(ao);
                    M2 = qp_amalg(M2, mm);
                    if (verbose >= 2) qp_printf("  the normal equations' nested dissection %.3f s: %.3g flops against AMD's %.3g\n", now() - tn, M2 ? schol_flops(M2) : -1.0, schol_flops(Mc));
                    if (M2 && schol_flops(M2) < schol_flops(Mc)) { schol_free(Mc); Mc = M2; } else if (M2) schol_free(M2);
                }
                I->M = Mc; schol_set_tinypiv(I->M, 1e-16);
                if (nd) {
                    I->nd = nd; I->isd = isd; isd = NULL; I->dcol = xm(sizeof(int) * (nd + 1));
                    { int a = 0; for (int j = 0; j < n; j++) if (I->isd[j]) I->dcol[a++] = j; }
                    signed char *sg = xm(mm + 1); for (int k = 0; k < mm; k++) sg[k] = k < m ? 1 : -1;
                    schol_set_signs(I->M, sg); free(sg);
                    I->offDd = xm(sizeof(size_t) * (nd + 1)); I->offAd = xm(sizeof(size_t) * ((size_t)R->Ap[n] + 1));
                    for (int a = 0; a < nd; a++) {
                        const int j = I->dcol[a]; I->offDd[a] = schol_offset(I->M, m + a, m + a);
                        for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->offAd[p] = schol_offset(I->M, m + a, R->Ai[p]);
                    }
                    if (verbose) qp_printf("normal equations: %d dense columns (more than 200 entries) kept as unknowns\n", nd);
                }
                I->offMd = xm(sizeof(size_t) * (m + 1)); I->dne = xm(sizeof(double) * (n + 1));
                for (int i = 0; i < m; i++) I->offMd[i] = schol_offset(I->M, i, i);
                if (dasm) {
                    I->dasm = 1; I->wdc = (int)fmin((double)n, fmax(1.0, (double)(64 << 20) / (8.0 * m)));
                    I->Md = xm(sizeof(double) * (size_t)m * (size_t)m); I->Wd = xm(sizeof(double) * (size_t)m * (size_t)I->wdc);
                    I->dmo = xm(sizeof(size_t) * ((size_t)m * (m - 1) / 2 + 1));
                    size_t t = 0; for (int i = 0; i < m; i++) for (int k = i + 1; k < m; k++) I->dmo[t++] = schol_offset(I->M, k, i);
                } else {
                /* (22 October 2026) M formed row by row (each row of A D A' in a dense vector, gathered into M's own
                 * entries): a place in the factor for each entry of M, where the table of a place for every pair of
                 * entries of a column of A took 0.84 s and 100 MB for PORTF_100 (QP: 12.9 million pairs, M of 5,151
                 * entries). The columns' row indices must be increasing; BRISK_QPMTAB=1: the table, as before */
                int sorted = 1;
                for (int j = 0; j < n && sorted; j++) for (int p = R->Ap[j] + 1; p < R->Ap[j + 1]; p++) if (R->Ai[p] <= R->Ai[p - 1]) { sorted = 0; break; }
                if (sorted && !getenv("BRISK_QPMTAB")) {
                    I->grp = xz(sizeof(int) * (m + 2));
                    for (int j = 0; j < n; j++) if (!(I->nd && I->isd[j])) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->grp[R->Ai[p] + 1]++;
                    for (int i = 0; i < m; i++) I->grp[i + 1] += I->grp[i];
                    I->grq = xm(sizeof(int) * ((size_t)I->grp[m] + 1)); I->grj = xm(sizeof(int) * ((size_t)I->grp[m] + 1));
                    { int *f = xm(sizeof(int) * (m + 1)); memcpy(f, I->grp, sizeof(int) * m);
                      for (int j = 0; j < n; j++) if (!(I->nd && I->isd[j])) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int t = f[R->Ai[p]]++; I->grq[t] = p; I->grj[t] = j; }
                      free(f); }
                    /* M's entries below the diagonal, row by row: the rows k > i sharing a sparse column with row i */
                    int *mk2 = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) mk2[i] = -1;
                    size_t cap = (size_t)m + 16, ne = 0; I->gmp = xm(sizeof(int) * (m + 2)); I->gmk = xm(sizeof(int) * cap);
                    for (int i = 0; i < m; i++) {
                        I->gmp[i] = (int)ne;
                        for (int t = I->grp[i]; t < I->grp[i + 1]; t++) { const int p = I->grq[t], j = I->grj[t];
                            for (int q = p + 1; q < R->Ap[j + 1]; q++) { const int k = R->Ai[q]; if (mk2[k] != i) { mk2[k] = i; if (ne == cap) { cap *= 2; I->gmk = realloc(I->gmk, sizeof(int) * cap); } I->gmk[ne++] = k; } } }
                    }
                    I->gmp[m] = (int)ne; free(mk2);
                    I->gmo = xm(sizeof(size_t) * (ne + 1));
                    for (int i = 0; i < m; i++) for (int t = I->gmp[i]; t < I->gmp[i + 1]; t++) I->gmo[t] = schol_offset(I->M, I->gmk[t], i);
                    I->gw = xz(sizeof(double) * (m + 1));
                } else {
                I->mpst = xm(sizeof(int) * (n + 1)); long np = 0;
                for (int j = 0; j < n; j++) { I->mpst[j] = (int)np; const long c = (I->nd && I->isd[j]) ? 0 : R->Ap[j + 1] - R->Ap[j]; np += c * (c - 1) / 2; }
                I->mpst[n] = (int)np;
                I->offMp = xm(sizeof(size_t) * (np + 1));
                for (int j = 0; j < n; j++) { if (I->nd && I->isd[j]) continue; int q = I->mpst[j]; for (int a = R->Ap[j]; a < R->Ap[j + 1]; a++) for (int b = R->Ap[j]; b < a; b++) { const int ia = R->Ai[a], ib = R->Ai[b]; I->offMp[q++] = schol_offset(I->M, ia > ib ? ia : ib, ia > ib ? ib : ia); } }
                }
                }   /* (the dense assembly's else) */
                if (verbose && I->K) qp_printf("normal equations: %d rows, nnz(L) = %zu (the augmented system's: %zu)\n", m, schol_nnz(I->M), schol_nnz(I->K));
                else if (verbose) qp_printf("normal equations: %d rows, nnz(L) = %zu (the augmented system not analysed: its own entries %zu)\n", m, schol_nnz(I->M), (size_t)n + m + R->Ap[n] + R->Hp[n]);
                { static int fall = -1; if (fall < 0) { const char *e = getenv("BRISK_QPNEFALL"); fall = e ? atoi(e) : 1; }
                  if (fall) I->Kspare = I->K; else schol_free(I->K); }
                I->K = NULL;
            } else if (Mc) schol_free(Mc);
            free(rp); free(rj); free(mk); free(dg); free(nb); free(pl);
        }
      pcg_ready:
        free(isd);
        /* the dissection on the augmented system when it is the one chosen */
        if (qp_nd_mode() == 1 && I->K && !I->M && schol_flops(I->K) >= 1e9) {
            SChol *K0 = I->K; I->K = NULL;
            const int ao = schol_set_amd(3); qp_nd_now = 1; schol_set_nd_budget(qp_nd_budget());
            const double tn = now();
            const int rk = k_build(I);
            qp_nd_now = 0; schol_set_nd_budget(0); schol_set_amd(ao);
            if (verbose >= 2) qp_printf("  the augmented system's nested dissection %.3f s: %.3g flops against AMD's %.3g\n", now() - tn, (!rk && I->K) ? schol_flops(I->K) : -1.0, schol_flops(K0));
            if (!rk && I->K && schol_flops(I->K) < schol_flops(K0)) schol_free(K0);
            else { if (I->K) schol_free(I->K); I->K = NULL; I->K = K0; k_build_offsets(I); }
        }
    }
    {   SChol *Fk = I->M ? I->M : I->K;
        const double ff = Fk ? schol_flops(Fk) : 0, fs = Fk ? 4.0 * (double)schol_nnz(Fk) + 4.0 * ((double)R->Ap[n] + (double)R->Hp[n]) : 1;
        const double ratio = ff / fmax(fs, 1.0);
        /* 3 correctors (BRISK_QPCORR=k: k; -1: the LP method's rule by flop ratios 10/30/50, a test: on the 138 Maros-Meszaros
         * QPs it gave 3,173 iterations against 2,382, the small factorizations getting none; the flop ratio also overstates
         * a factorization against a memory-bound solve, QPLIB_10038: 93 by flops, 9 measured) */
        I->kcorr = 3;
        const char *e = getenv("BRISK_QPCORR");
        if (e) I->kcorr = atoi(e) >= 0 ? atoi(e) : ratio > 50 ? 3 : ratio > 30 ? 2 : ratio > 10 ? 1 : 0;
        if (verbose && I->pcg) qp_printf("normal equations: %d rows, solved by conjugate gradients; analysis %.3fs; %d corrector%s\n", m, now() - ta, I->kcorr, I->kcorr == 1 ? "" : "s");
        else if (verbose) qp_printf("factorization: %d unknowns, nnz(L) = %zu, %.2e flops (%.0f solves), analysis %.3fs; %d corrector%s\n", I->M ? m : N, Fk ? schol_nnz(Fk) : (size_t)0,
                            ff, ratio, now() - ta, I->kcorr, I->kcorr == 1 ? "" : "s"); }
    I->rho = 1e-8; I->delta = 1e-8; I->rho_ref = 1e-14; I->delta_ref = 1e-14; I->tol = tol; I->e_prev = HUGE_VAL;
    { const char *e = getenv("BRISK_QPREG"); if (e) sscanf(e, "%lf,%lf", &I->rho, &I->delta); }
    { const char *e = getenv("BRISK_QPDALL"); I->delta_ineq = e ? atof(e) : 0; I->delta_ineq_ref = 0; }
    { const char *e = getenv("BRISK_QPREFREG"); if (e && atoi(e)) { I->rho_ref = I->rho; I->delta_ref = I->delta; I->delta_ineq_ref = I->delta_ineq; } }

    /* ---- the starting point: x from [-(H + I) A'; A I] [x; y] = [c; b], s = A x, slacks and multipliers shifted */
    for (int j = 0; j < n; j++) I->Dx[j] = 1.0;
    for (int i = 0; i < m; i++) I->Ds[i] = 1.0;
    /* (24 October 2026) the normal equations of a network QP (every column of A at most two entries) by the factor or by
     * the conjugate gradients, whichever costs less work an iteration: the factorization's flops at 2e10 a second and
     * s = 2 + correctors solves of 4 nnz(L) each, against s solves of k conjugate-gradient iterations of 4 nnz(A) + 2 n +
     * 20 m each, the solves at 1e9 a second (memory-bound); k from one solve of the start's system (Dx = Ds = 1). Rates
     * of one core and counts of work, not measured times: the choice is the same on every run and at every thread
     * count. BRISK_QPCG=0: never; 3: the factor whenever it fits (the conjugate gradients only where none does) */
    {   const char *ecg = getenv("BRISK_QPCG"); const int cgm = ecg ? atoi(ecg) : 1;
        if (I->M && !I->nd && cgm == 1 && !I->pcg) {
            int ok2 = 1; for (int j = 0; j < n && ok2; j++) if (R->Ap[j + 1] - R->Ap[j] > 2) ok2 = 0;
            if (ok2 && ne_pcg_setup(I)) {
                ne_pcg_factor(I);
                double *r2 = I->tmp;              /* the start's right-hand side in the normal equations' form */
                for (int i = 0; i < m; i++) { const double lo = R->rl[i], hi = R->ru[i]; r2[i] = R->eq[i] ? lo : (fin(lo) && fin(hi)) ? 0.5 * (lo + hi) : fin(lo) ? lo : hi; }
                for (int j = 0; j < n; j++) { const double t = I->dne[j] * R->c[j]; for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) r2[R->Ai[p]] += R->Ax[p] * t; }
                const double sv = 2.0 + I->kcorr, cit = (4.0 * (double)R->Ap[n] + 2.0 * n + 20.0 * m) / 1e9;
                const double tf = schol_flops(I->M) / 2e10 + sv * 4.0 * (double)schol_nnz(I->M) / 1e9;
                /* (the trial stops where the conjugate gradients can no longer win: its cost stays below a fifth of the
                 * factor's work an iteration) */
                const int kcap = (int)fmin(1e9, tf / (sv * cit)) + 1;
                const long k0 = I->ncg; ne_pcg_solve_n(I, r2, kcap); const double k = (double)(I->ncg - k0); I->ncg = k0;
                const double tc = sv * fmax(k, 1.0) * cit;
                if (verbose >= 2) qp_printf("  the normal equations: the factor's work an iteration %.3g s, the conjugate gradients' %.3g s (%.0f iterations a solve)\n", tf, tc, k);
                if (tc < tf) {
                    schol_free(I->M); I->M = NULL; if (I->Kspare) { schol_free(I->Kspare); I->Kspare = NULL; }
                    if (verbose) qp_printf("normal equations: by conjugate gradients preconditioned by the spanning tree of the columns' weights over\n"
                                           "  the matrix's diagonal (every column has at most two entries), less work an iteration than the factor\n");
                } else {
                    I->pcg = 0;
                    free(I->tord); free(I->tpar); free(I->tcol); free(I->ecol); free(I->uf); free(I->tL); free(I->tD); free(I->tE); free(I->tw); free(I->cgw);
                    I->tord = I->tpar = I->tcol = I->ecol = I->uf = NULL; I->tL = I->tD = I->tE = I->tw = I->cgw = NULL;
                }
            }
        }
    }
    {
        int fail = (I->K || I->M || I->pcg) ? ipm_factor(I) : 0;
        if (fail) { if (verbose) qp_printf("the first factorization failed\n"); Res->status = QP_NUMERICAL; goto done; }
        for (int j = 0; j < n; j++) I->rhs[j] = R->c[j];
        for (int i = 0; i < m; i++) {
            const double lo = R->rl[i], hi = R->ru[i];
            I->rhs[n + i] = R->eq[i] ? lo : (fin(lo) && fin(hi)) ? 0.5 * (lo + hi) : fin(lo) ? lo : hi;
        }
        /* the system solved here has Ds^-1 = 1 on the inequality rows and delta on the equalities */
        if (N > 0) { memcpy(I->sol, I->rhs, sizeof(double) * N); kbase(I, I->sol); }
        for (int j = 0; j < n; j++) I->x[j] = I->sol[j];
        for (int i = 0; i < m; i++) I->y[i] = I->sol[n + i];
        for (int i = 0; i < m; i++) I->s[i] = 0;
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) I->s[R->Ai[p]] += R->Ax[p] * I->x[j];
        hmul_red(R, I->x, I->hx);
        double *zt = I->rd;              /* z = H x + c - A'y */
        for (int j = 0; j < n; j++) zt[j] = I->hx[j] + R->c[j];
        for (int j = 0; j < n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) zt[j] -= R->Ax[p] * I->y[R->Ai[p]];
        /* (22 October 2026) the slacks are functions of x and s, t = x - xl, w = s - rl, exactly: x and s are moved into
         * the interior of their bounds here and kept there by the step rule. As separate variables with their own
         * residuals x - t - xl, the residuals' rounding (1e-16 |x|) divided by slacks near 1e-17 at the end put errors
         * of order one into the dual step: the dual residual grew by about 40 times an iteration (QSCTAP1, LASER) */
        static int smode = -1; if (smode < 0) { const char *e = getenv("BRISK_QPSTART"); smode = e ? atoi(e) : 0; }
        if (smode == 1 && I->N > 0) {
            /* Mehrotra's starting point (Mehrotra 1992, sec. 7), for bounds and ranged rows: the least-squares point's
             * slacks a (x - xl, xu - x, s - rl, ru - s) and multipliers b (z and -z of the bounds, y and -y of the rows)
             * shifted by 1.5 times their most negative entry, then by 1/2 a'b / sum(b) and 1/2 a'b / sum(a): the start's
             * complementarity has the scale of the problem's gap. With a fixed margin and a multiplier shift of 1e-2 the
             * start's complementarity was far too small for the distance to the solution, and the iterates crawled at
             * steps near 0.1 (YAO: 107 iterations, Gurobi's barrier 25) */
            double amin = 1e300, bmin = 1e300;
            for (int j = 0; j < n; j++) {
                if (R->hl[j]) { amin = fmin(amin, I->x[j] - R->xl[j]); bmin = fmin(bmin, zt[j]); }
                if (R->hu[j]) { amin = fmin(amin, R->xu[j] - I->x[j]); bmin = fmin(bmin, -zt[j]); }
            }
            for (int i = 0; i < m; i++) {
                if (R->gl[i]) { amin = fmin(amin, I->s[i] - R->rl[i]); bmin = fmin(bmin, I->y[i]); }
                if (R->gu[i]) { amin = fmin(amin, R->ru[i] - I->s[i]); bmin = fmin(bmin, -I->y[i]); }
            }
            const double dp = fmax(-1.5 * amin, 0), dd = fmax(-1.5 * bmin, 0);
            double ab = 0, sa = 0, sb = 0;
#define MACC(A, B) { const double a_ = (A) + dp, b_ = (B) + dd; ab += a_ * b_; sa += a_; sb += b_; }
            for (int j = 0; j < n; j++) { if (R->hl[j]) MACC(I->x[j] - R->xl[j], zt[j]); if (R->hu[j]) MACC(R->xu[j] - I->x[j], -zt[j]); }
            for (int i = 0; i < m; i++) { if (R->gl[i]) MACC(I->s[i] - R->rl[i], I->y[i]); if (R->gu[i]) MACC(R->ru[i] - I->s[i], -I->y[i]); }
#undef MACC
            double Dp = dp + 0.5 * ab / fmax(sb, 1e-300), Dd = dd + 0.5 * ab / fmax(sa, 1e-300);
            if (!(Dp > 0) || !isfinite(Dp)) Dp = 1;
            if (!(Dd > 0) || !isfinite(Dd)) Dd = 1;
            /* x and s shifted by Dp into their bounds (a column or row bounded on both sides: kept a quarter of its range
             * inside each bound), the multipliers by Dd */
            for (int j = 0; j < n; j++) {
                if (R->hl[j] && R->hu[j]) { const double g = fmin(Dp, 0.25 * (R->xu[j] - R->xl[j])); I->x[j] = fmin(fmax(I->x[j], R->xl[j] + g), R->xu[j] - g); }
                else if (R->hl[j]) I->x[j] += Dp;
                else if (R->hu[j]) I->x[j] -= Dp;
                if (R->hl[j]) { I->tl[j] = I->x[j] - R->xl[j]; I->zl[j] = zt[j] + Dd; }
                if (R->hu[j]) { I->tu[j] = R->xu[j] - I->x[j]; I->zu[j] = -zt[j] + Dd; }
            }
            for (int i = 0; i < m; i++) {
                if (R->eq[i]) continue;
                if (R->gl[i] && R->gu[i]) { const double g = fmin(Dp, 0.25 * (R->ru[i] - R->rl[i])); I->s[i] = fmin(fmax(I->s[i], R->rl[i] + g), R->ru[i] - g); }
                else if (R->gl[i]) I->s[i] += Dp;
                else if (R->gu[i]) I->s[i] -= Dp;
                if (R->gl[i]) { I->wl[i] = I->s[i] - R->rl[i]; I->vl[i] = I->y[i] + Dd; }
                if (R->gu[i]) { I->wu[i] = R->ru[i] - I->s[i]; I->vu[i] = -I->y[i] + Dd; }
            }
            /* a pair whose multiplier is still not positive (a two-sided bound kept inside its range): set to Dd */
            for (int j = 0; j < n; j++) { if (R->hl[j] && !(I->zl[j] > 0)) I->zl[j] = Dd; if (R->hu[j] && !(I->zu[j] > 0)) I->zu[j] = Dd; }
            for (int i = 0; i < m; i++) { if (R->gl[i] && !(I->vl[i] > 0)) I->vl[i] = Dd; if (R->gu[i] && !(I->vu[i] > 0)) I->vu[i] = Dd; }
            for (int i = 0; i < m; i++) if (!R->eq[i]) I->y[i] = (R->gl[i] ? I->vl[i] : 0) - (R->gu[i] ? I->vu[i] : 0);
            if (verbose > 2) qp_printf("Mehrotra's start: primal shift %.2e, dual shift %.2e\n", Dp, Dd);
        } else {
        static double mg = -1; if (mg < 0) { const char *e = getenv("BRISK_QPMARGIN"); mg = e ? atof(e) : 1.0; }
        for (int j = 0; j < n; j++) {
            if (R->hl[j] && R->hu[j]) { const double g = fmin(mg, 0.25 * (R->xu[j] - R->xl[j])); I->x[j] = fmin(fmax(I->x[j], R->xl[j] + g), R->xu[j] - g); }
            else if (R->hl[j]) I->x[j] = fmax(I->x[j], R->xl[j] + mg);
            else if (R->hu[j]) I->x[j] = fmin(I->x[j], R->xu[j] - mg);
        }
        for (int i = 0; i < m; i++) {
            if (R->eq[i]) continue;
            if (R->gl[i] && R->gu[i]) { const double g = fmin(mg, 0.25 * (R->ru[i] - R->rl[i])); I->s[i] = fmin(fmax(I->s[i], R->rl[i] + g), R->ru[i] - g); }
            else if (R->gl[i]) I->s[i] = fmax(I->s[i], R->rl[i] + mg);
            else if (R->gu[i]) I->s[i] = fmin(I->s[i], R->ru[i] - mg);
        }
        double mind = 1e300;
        for (int j = 0; j < n; j++) {
            if (R->hl[j]) { I->tl[j] = I->x[j] - R->xl[j]; I->zl[j] = fmax(zt[j], 0); mind = fmin(mind, I->zl[j]); }
            if (R->hu[j]) { I->tu[j] = R->xu[j] - I->x[j]; I->zu[j] = fmax(-zt[j], 0); mind = fmin(mind, I->zu[j]); }
        }
        for (int i = 0; i < m; i++) {
            if (R->eq[i]) continue;
            if (R->gl[i]) { I->wl[i] = I->s[i] - R->rl[i]; I->vl[i] = fmax(I->y[i], 0); mind = fmin(mind, I->vl[i]); }
            if (R->gu[i]) { I->wu[i] = R->ru[i] - I->s[i]; I->vu[i] = fmax(-I->y[i], 0); mind = fmin(mind, I->vu[i]); }
        }
        if (I->N > 0) {
            /* the multipliers shifted (Mehrotra's rule on the dual side: positive, then balanced against the slacks) */
            double sp = 0, pd = 0;
            for (int j = 0; j < n; j++) {
                if (R->hl[j]) { I->zl[j] += 1e-2; sp += I->tl[j]; pd += I->tl[j] * I->zl[j]; }
                if (R->hu[j]) { I->zu[j] += 1e-2; sp += I->tu[j]; pd += I->tu[j] * I->zu[j]; }
            }
            for (int i = 0; i < m; i++) {
                if (R->gl[i]) { I->vl[i] += 1e-2; sp += I->wl[i]; pd += I->wl[i] * I->vl[i]; }
                if (R->gu[i]) { I->vu[i] += 1e-2; sp += I->wu[i]; pd += I->wu[i] * I->vu[i]; }
            }
            static double dsh = -1; if (dsh < 0) { const char *e = getenv("BRISK_QPDSHIFT"); dsh = e ? atof(e) : 1e-2; }
            const double ed = 0.5 * pd / fmax(sp, 1e-300) + dsh;
            for (int j = 0; j < n; j++) { if (R->hl[j]) I->zl[j] += ed; if (R->hu[j]) I->zu[j] += ed; }
            for (int i = 0; i < m; i++) { if (R->gl[i]) I->vl[i] += ed; if (R->gu[i]) I->vu[i] += ed; }
            (void)mind;
        }
        /* the rows' multiplier consistent with the slacks' multipliers */
        for (int i = 0; i < m; i++) if (!R->eq[i]) I->y[i] = (R->gl[i] ? I->vl[i] : 0) - (R->gu[i] ? I->vu[i] : 0);
        }
    }
    double bn = 1, cnrm = 1 + nrmi(n, R->c);
    for (int i = 0; i < m; i++) { if (fin(R->rl[i])) bn = fmax(bn, fabs(R->rl[i])); if (fin(R->ru[i])) bn = fmax(bn, fabs(R->ru[i])); }
    for (int j = 0; j < n; j++) { if (R->hl[j]) bn = fmax(bn, fabs(R->xl[j])); if (R->hu[j]) bn = fmax(bn, fabs(R->xu[j])); }
    if (verbose > 1) qp_printf("%4s %15s %15s %8s %8s %8s %8s %6s\n", "it", "primal obj", "dual obj", "pres", "dres", "gap", "mu", "step");
    int it, status = QP_MAXIT, nretry = 0;
    /* the best point seen (by its largest error on the problem as given): returned when the iteration ends without
     * reaching the tolerance (a breakdown at the end, QPCBOEI2: the gap 1.9e-8 one iteration before) */
    double best = INFINITY, *bx = xm(sizeof(double) * (n0 + 1)), *by = xm(sizeof(double) * (m0 + 1)), *bz = xm(sizeof(double) * (n0 + 1));
    double pobj = 0, dobj = 0, epr = 1, edu = 1, egp = 1, mu = 0, alpha = 0, edu_prev = 1;
    for (it = 0; it <= maxit; it++) {
        compute_residuals(I);
        /* objectives (scaled): pobj = 1/2 x'Hx + c'x, dobj = -1/2 x'Hx + rows' and bounds' terms */
        double xhx = 0, cx = 0; for (int j = 0; j < n; j++) { xhx += I->x[j] * I->hx[j]; cx += R->c[j] * I->x[j]; }
        pobj = 0.5 * xhx + cx; dobj = -0.5 * xhx;
        for (int j = 0; j < n; j++) { if (R->hl[j]) dobj += I->zl[j] * R->xl[j]; if (R->hu[j]) dobj -= I->zu[j] * R->xu[j]; }
        for (int i = 0; i < m; i++) {
            if (R->eq[i]) { dobj += I->y[i] * R->rl[i]; continue; }
            if (R->gl[i]) dobj += I->vl[i] * R->rl[i];
            if (R->gu[i]) dobj -= I->vu[i] * R->ru[i];
        }
        double rpn = nrmi(m, I->rp);
        for (int j = 0; j < n; j++) { if (R->hl[j]) rpn = fmax(rpn, fabs(I->rtl[j])); if (R->hu[j]) rpn = fmax(rpn, fabs(I->rtu[j])); }
        for (int i = 0; i < m; i++) { if (R->gl[i]) rpn = fmax(rpn, fabs(I->rwl[i])); if (R->gu[i]) rpn = fmax(rpn, fabs(I->rwu[i])); }
        double rdn = fmax(nrmi(n, I->rd), nrmi(m, I->rs));
        /* (22 October 2026, a test: BRISK_QPMEAS=1) the measures in the scaled problem as the images of those of the
         * problem as read: its 1 is 1/b_s in the rows, c_s b_s in the dual equations and c_s in the objective, and the
         * objective includes the constant c0. With 1 in the scaled units the measures were absolute where c_s |f| is far
         * below 1 (BOYD2, QP: c_s = 2.5e-15, the refinement's gap target 4e4 on the problem as read; QPLIB_9002: c_s =
         * 1.3e-16) and too loose where c0 cancels most of the objective (LISWET1: c0 = 2,525, f = 36) */
        static int meas = -1; if (meas < 0) { const char *e = getenv("BRISK_QPMEAS"); meas = e ? atoi(e) : 0; }
        const double one_p = meas ? 1.0 / R->bs : 1, one_d = meas ? R->cs * R->bs : 1, one_g = meas ? R->cs : 1, c0s = meas ? R->cs * (P->c0 + c0add) : 0;
        const double bnp = meas ? bn - 1 : bn;     /* (bn starts at 1) */
        const double dsz = fmax(cnrm - 1, fmax(nrmi(n, I->hx), fmax(nrmi(n, I->zl), fmax(nrmi(n, I->zu), nrmi(m, I->y)))));
        epr = rpn / (one_p + bnp + nrmi(n, I->x)); edu = rdn / (one_d + dsz); egp = fabs(pobj - dobj) / (one_g + fabs(pobj + c0s));
        /* (23 October 2026; the default from 24 October, BRISK_QPREFMU=0: a tenth of the present residuals) the refinement's targets held in the infeasible-start method's neighbourhood
         * |r_k| / mu_k <= |r_0| / mu_0 (Kojima, Megiddo and Mizuno; Wright 1997, chapter 6) instead of a tenth of the present
         * residuals: the residuals need only fall as fast as mu, and a full step (alpha = 1) leaves the solve's error as the
         * next residual, which the tenth then demanded to refine at every solve (QUADCMPC1, QP: primal residual 4e-9 at
         * iteration 1 with mu = 1.5; 58 refinement steps in 95 solves). The checks for a residual's growth compare with
         * what the last targets allowed (the last residual plus the solve's permitted error), not with the last residual */
        static int refmu_env = -1; if (refmu_env < 0) { const char *e = getenv("BRISK_QPREFMU"); refmu_env = e ? atoi(e) : 1; }
        if (it == 0) I->refmu = refmu_env;
        const int refmu = I->refmu;
        double epr_allow = I->epr_prev, edu_allow = edu_prev;
        const double mu_now = I->N > 0 ? complementarity(I) / I->N : 0;
        if (refmu && it > 0) {
            epr_allow = fmax(epr_allow, (I->rpn_prev + I->rf_pr) / (one_p + bnp + nrmi(n, I->x)));
            edu_allow = fmax(edu_allow, (I->rdn_prev + I->rf_du) / (one_d + dsz));
        }
        if (it == 0) { I->rpn0 = rpn; I->rdn0 = rdn; I->mu0 = mu_now; }
        /* the refinement's targets for the next directions (BRISK_QPREFINE=3): a tenth of the present residuals, at least
         * 1e-2 of the tolerance in the stop's measures */
        I->rf_pr = 0.1 * fmax(rpn, 1e-2 * tol * (one_p + bnp + nrmi(n, I->x)));
        I->rf_du = 0.1 * fmax(rdn, 1e-2 * tol * (one_d + dsz));
        I->rf_gap = 0.1 * fmax(fabs(pobj - dobj), 1e-2 * tol * (one_g + fabs(pobj + c0s)));
        if (refmu && I->mu0 > 0 && mu_now > 0 && mu_now < I->mu0) {
            I->rf_pr = fmax(I->rf_pr, 0.1 * I->rpn0 * mu_now / I->mu0);
            I->rf_du = fmax(I->rf_du, 0.1 * I->rdn0 * mu_now / I->mu0);
        }
        I->rpn_prev = rpn; I->rdn_prev = rdn;
        /* (22 October 2026) the normal equations' breakdown: the dual residual a thousand times the last one after a step
         * (STADAT1: 2e-10 -> 4e-3 at iteration 44, the columns inside their bounds without a quadratic term at D near 1e12,
         * their tiny pivots replaced): the step undone, rho on every column from here on */
        /* (22 October 2026, a test: BRISK_QPUNDO=1) on the augmented system too: a step after which a residual is a thousand
         * times the last one is undone, and the direction recomputed from a factorization with 100 times the regularization
         * and the full refinement (QPLIB_8515, QP: a direction whose solve GMRES could not refine took the primal residual
         * from 1.9e-12 to 1.4 at a step of 1e-4); at most three times in a solve */
        { static int undo = -1; if (undo < 0) { const char *e = getenv("BRISK_QPUNDO"); undo = e ? atoi(e) : 0; }
          if (undo && !I->M && I->K && it > 0 && I->nundo < 3 && (edu > 1e3 * fmax(edu_allow, 1e-14) || epr > 1e3 * fmax(epr_allow, 1e-14))) {
              state_copy(I, 1); I->nundo++; I->redo = 1;
              I->rho = fmin(1e-4, I->rho * 100); I->delta = fmin(1e-4, I->delta * 100); I->fullref = 1; I->rmu_hold = 3;
              if (verbose) qp_printf("     a residual grew a thousandfold (dual %.1e -> %.1e, primal %.1e -> %.1e): the step undone, the regularization raised to %.0e, the full refinement\n", edu_prev, edu, I->epr_prev, epr, I->rho);
              it--; continue; } }
        if (I->M && !I->nerho && it > 0 && edu > 1e3 * fmax(edu_allow, 1e-14)) {
            state_copy(I, 1); I->redo = 1;
            /* (22 October 2026) the augmented system from here on, whose analysis was kept: rho on every column of the
             * normal equations made the dual residual grow thirtyfold an iteration on QPCBOEI2 (4.6e-12 -> 1e227);
             * BRISK_QPNEFALL=0: rho on the columns, as before */
            if (ne_to_augmented(I, &k_lazy)) {
                if (verbose) qp_printf("     the normal equations lost the dual residual (%.1e -> %.1e): the step undone, the augmented system from here on\n", edu_prev, edu);
            } else {
                I->nerho = 1;
                if (verbose) qp_printf("     the normal equations lost the dual residual (%.1e -> %.1e): the step undone, the columns regularized from here on\n", edu_prev, edu);
            }
            it--; continue;
        }
        /* (BRISK_QPREFINE=3) a residual that grows under the cheap solves: the full refinement from here on (STADAT1: the
         * scaled dual residual 3.4e-3 -> 3.5e-2 -> 0.3 -> 930 in steps below the breakdown's thousandfold) */
        { static int rm3 = -1; if (rm3 < 0) { const char *e_ = getenv("BRISK_QPREFINE"); rm3 = !e_ || atoi(e_) == 3; }
          if (rm3 && !I->fullref && it > 0 && (edu > 2 * fmax(edu_allow, 1e-14) || epr > 2 * fmax(epr_allow, 1e-14)) && fmax(edu, epr) > tol) {
              I->fullref = 1; if (verbose > 1) qp_printf("     the full refinement from here on (a residual grew: dual %.1e -> %.1e, primal %.1e -> %.1e)\n", edu_prev, edu, I->epr_prev, epr); }
          I->epr_prev = epr; }
        edu_prev = edu;
        mu = I->N > 0 ? complementarity(I) / I->N : 0;
        if (verbose > 1) qp_printf("%4d %+15.8e %+15.8e %8.1e %8.1e %8.1e %8.1e %6.3f\n", it, pobj / R->cs, dobj / R->cs, epr, edu, egp, mu, alpha);
        /* the stop on the errors of the problem as given (the reported ones), every iteration (O(nnz)): the scaled
         * problem's own measures stay above the tolerance at optimal points (YAO, LISWET1: 1e-11 and below) and fall below
         * it at points that are not (HS268: the gap 3e-6) */
        extract(P, R, I, fixval, Res);
        double e1, e2, e3; qp_errors(P, Res->x, Res->y, Res->z, &e1, &e2, &e3);
        if (verbose > 2) qp_printf("     errors of the problem as given: primal %.1e, dual %.1e, gap %.1e\n", e1, e2, e3);
        if (e1 <= tol && e2 <= tol && e3 <= tol) { status = QP_OPTIMAL; break; }
        /* (24 October 2026) the iterate's multipliers as a proof of primal infeasibility, at every iteration and any size
         * (qp_farkas_proof: exact where the bounds r needs are finite); BRISK_QPFARKAS=0: only the growth test below */
        { static int fk = -1; if (fk < 0) { const char *e = getenv("BRISK_QPFARKAS"); fk = e ? atoi(e) : 1; }
          if (fk && it > 0 && qp_farkas_proof(P, Res->y, Res->z)) {
              if (verbose > 1) qp_printf("     the multipliers prove the problem infeasible (a Farkas certificate within the bounds)\n");
              status = QP_PRIMAL_INFEASIBLE; break; } }
        /* (22 October 2026) the certificates, once the iterates grow (multipliers or x beyond 1e6 times the data): a
         * Farkas certificate's residual within tol of its value, or an unbounded direction's conditions within tol of
         * its descent */
        {   double sy = 0, sx = 0;
            for (int i = 0; i < m0; i++) sy = fmax(sy, fabs(Res->y[i]));
            for (int j = 0; j < n0; j++) { sy = fmax(sy, fabs(Res->z[j])); sx = fmax(sx, fabs(Res->x[j])); }
            const double big = 1e6 * (1 + qp_scale_of(P));
            if (sy > big || sx > big) {
                double pv, pr, dv, dr; qp_infeas_measures(P, Res->x, Res->y, Res->z, &pv, &pr, &dv, &dr);
                if (verbose > 2) qp_printf("     certificates: primal infeasibility value %.1e residual %.1e; unboundedness descent %.1e residual %.1e\n", pv, pr, dv, dr);
                /* both parts relative: the normalized certificate's residual at most tol, its value at least tol times the
                 * size of the data (the value carries the bounds' size: QPLIB_9002, bounds to 3.4e8, an exploding iterate
                 * gave a value of 4.1e9 with a residual of 1.0, taken for a certificate when measured against the value) */
                const double sd = 1 + qp_scale_of(P);
                if (sy > big && pr <= tol && pv >= tol * sd) { status = QP_PRIMAL_INFEASIBLE; break; }
                if (sx > big && dr <= tol && dv >= tol * (1 + qp_cost_scale(P))) { status = QP_DUAL_INFEASIBLE; break; }
            }
        }
        /* (BRISK_QPREFINE=3) the cheap solves' progress stalls near the end (UBH1: the gap 4.6e-6 -> 4.4e-6 an iteration
         * at steps of 0.995 with mu at 1e-27): the full refinement from then on. (22 October 2026) A stall says the solves
         * are inaccurate only after a long step, one that should have cut the error at least fourfold (alpha >= 0.75):
         * after a step of 0.3 an error falls by 0.7 with exact solves (LISWET1, QP: the rule fired at iteration 11 and the
         * full refinement took 750 of 1,258 solves). BRISK_QPSTALLA=a: the least step (0: any, as before) */
        static double stalla = -1; if (stalla < 0) { const char *e = getenv("BRISK_QPSTALLA"); stalla = e ? atof(e) : 0.75; }
        { const double e = fmax(e1, fmax(e2, e3));
          /* (23 October 2026) not on the iteration that repeats an undone step: its error is the one before the undone step by
           * construction, and the rule read it as a stall after a long step (TV1D_100000, QP: the normal equations' breakdown
           * at iteration 8 undone, then the full refinement for the rest, 45 refinement steps in 93 solves) */
          if (I->redo) I->redo = 0;
          /* (BRISK_QPREFMU=1) a stall first withdraws the neighbourhood's looser targets (a tenth of the present residuals
           * from here on), and only a second stall the full refinement: with the residuals held at the neighbourhood's bound
           * the largest error on the problem as given can stall while the solves are accurate (QSCFXM1, QP: the full
           * refinement from iteration 20, 372 solves against 142) */
          else if (!I->fullref && it >= 3 && e < 1e-3 && e > 0.5 * I->e_prev && alpha >= stalla && I->refmu) { I->refmu = 0; if (verbose > 1) qp_printf("     the refinement's targets a tenth of the present residuals from here on (the largest error %.1e -> %.1e)\n", I->e_prev, e); }
          else if (!I->fullref && it >= 3 && e < 1e-3 && e > 0.5 * I->e_prev && alpha >= stalla) { I->fullref = 1; if (verbose > 1) qp_printf("     the full refinement from here on (the largest error %.1e -> %.1e)\n", I->e_prev, e); }
          I->e_prev = e; }
        { const double e = fmax(e1, fmax(e2, e3)); if (e < best) { best = e; memcpy(bx, Res->x, sizeof(double) * n0); memcpy(by, Res->y, sizeof(double) * m0); memcpy(bz, Res->z, sizeof(double) * n0); } }
        if (it == maxit) break;
        if (opt->timelimit > 0 && now() - t0 > opt->timelimit) { status = QP_TIME; break; }
        /* the scaling matrices */
        for (int j = 0; j < n; j++) I->Dx[j] = (R->hl[j] ? I->zl[j] / I->tl[j] : 0) + (R->hu[j] ? I->zu[j] / I->tu[j] : 0);
        for (int i = 0; i < m; i++) I->Ds[i] = R->eq[i] ? 1 : (R->gl[i] ? I->vl[i] / I->wl[i] : 0) + (R->gu[i] ? I->vu[i] / I->wu[i] : 0);
        /* (22 October 2026, a test: BRISK_QPDMU=1) the equality rows' regularization with mu: delta = max(1e-12, min(1e-8,
         * mu)); with delta fixed at 1e-8 the primal residual of QPLIB_10034 (QP, 40,200 equality rows) stalled at 3.5e-9 and
         * the gap at 1.5e-5 while mu fell to 1e-14 */
        { static int dmu = -1; if (dmu < 0) { const char *e = getenv("BRISK_QPDMU"); dmu = e ? atoi(e) : 0; }
          if (dmu && nretry == 0) I->delta = fmax(1e-12, fmin(1e-8, mu)); }
        /* (22 October 2026, the default; BRISK_QPRMU=0: fixed at 1e-8) both regularizations with mu, rho = delta = max(rmin, min(1e-8, mu))
         * (rmin: BRISK_QPRMIN, 1e-12), as in the primal-dual regularized methods of Friedlander and Orban (2012): a static
         * 1e-8 against scaled entries of order one made the refinement contract by 1.15 a step at the end (UBH1, QP: 444
         * refinement steps in 476 solves), 1e-12 from the start broke the normal equations (QSCRS8: the dual residual 2e-9
         * -> 7.8e-3 at iteration 16) */
        { static int rmu = -1; static double rmin = 1e-12;
          if (rmu < 0) { const char *e = getenv("BRISK_QPRMU"); rmu = e ? atoi(e) : 1; e = getenv("BRISK_QPRMIN"); if (e) rmin = atof(e); }
          if (rmu && nretry == 0 && it > 0 && I->rmu_hold > 0) I->rmu_hold--;
          else if (rmu && nretry == 0 && it > 0) {
              I->rho = I->delta = fmax(rmin, fmin(1e-8, mu));
              { const char *e = getenv("BRISK_QPREFREG"); if (e && atoi(e)) { I->rho_ref = I->rho; I->delta_ref = I->delta; } } } }
    refactor:
        if ((I->K || I->M || I->pcg) && ipm_factor(I)) {
            /* (22 October 2026) a failed factorization of the normal equations: the augmented system from here on, as
             * after their breakdown (BOYD2, QP: the normal equations' factorization failed at mu 1e-17, the solve ended
             * with NUMERICAL DIFFICULTIES at a gap of 1.2e-3 while the augmented system was at hand) */
            if (I->M && ne_to_augmented(I, &k_lazy)) {
                if (verbose) qp_printf("     the normal equations' factorization failed at iteration %d: the augmented system from here on\n", it);
                goto refactor;
            }
            I->rho *= 100; I->delta *= 100;
            if (I->rho > 1e-2 || ipm_factor(I)) { if (verbose) qp_printf("the factorization failed at iteration %d\n", it); status = QP_NUMERICAL; break; }
        }
        /* predictor */
        for (int j = 0; j < n; j++) { I->ctl[j] = R->hl[j] ? -I->tl[j] * I->zl[j] : 0; I->ctu[j] = R->hu[j] ? -I->tu[j] * I->zu[j] : 0; }
        for (int i = 0; i < m; i++) { I->cwl[i] = R->gl[i] ? -I->wl[i] * I->vl[i] : 0; I->cwu[i] = R->gu[i] ? -I->wu[i] * I->vu[i] : 0; }
        direction(I);
        double aaff = I->N > 0 ? fmin(1.0, step_to_boundary(I)) : 1.0;
        double sigma = 0;
        if (I->N > 0) {
            const double muaff = compl_after(I, aaff) / I->N;
            sigma = pow(fmax(muaff, 0) / fmax(mu, 1e-300), 3); if (sigma > 1) sigma = 1;
            /* corrector: sigma mu - t z - dt_aff dz_aff */
            const double sm = sigma * mu;
            for (int j = 0; j < n; j++) {
                if (R->hl[j]) I->ctl[j] = sm - I->tl[j] * I->zl[j] - I->dtl[j] * I->dzl[j];
                if (R->hu[j]) I->ctu[j] = sm - I->tu[j] * I->zu[j] - I->dtu[j] * I->dzu[j];
            }
            for (int i = 0; i < m; i++) {
                if (R->gl[i]) I->cwl[i] = sm - I->wl[i] * I->vl[i] - I->dwl[i] * I->dvl[i];
                if (R->gu[i]) I->cwu[i] = sm - I->wu[i] * I->vu[i] - I->dwu[i] * I->dvu[i];
            }
            direction(I);
        }
        double amax = I->N > 0 ? step_to_boundary(I) : 1e300;
        /* Gondzio's multiple centrality correctors (Gondzio 1996; Colombo and Gondzio 2008): at the trial step
         * a~ = min(1, 1.5 a + 0.1) the products t z outside [0.1 sigma mu, 10 sigma mu] are pulled back into it by a
         * correction of the complementarity right-hand sides (the residuals' part unchanged: one more solve with the
         * same factor), kept while it lengthens the step by 1% at least. Against steps blocked by a few pairs far from
         * the centre (YAO: 107 iterations at steps near 0.1) */
        {
            /* (the number by the cost of a factorization against a solve, the LP method's rule: 0-3 at flop ratios
             * 10/30/50; QPLIB_10038: a factorization 1.1 s, a solve 0.12 s, and the solves 2.5 times the factorizations) */
            const int kmax = I->kcorr;
            const double sm = sigma * mu, bl = 0.1 * sm, bu = 10 * sm;
            for (int k = 0; k < kmax && I->N > 0 && amax < 1.0 / 0.995; k++) {
                const double at = fmin(1.0, 1.5 * fmin(amax, 1.0) + 0.1);
                dir_copy(I, 0);
#define GCORR(T, DT, Z, DZ, C) { const double q = (T + at * DT) * (Z + at * DZ); C += q < bl ? bl - q : q > bu ? fmax(bu - q, -bu) : 0; }
                for (int j = 0; j < n; j++) {
                    if (R->hl[j]) GCORR(I->tl[j], I->dtl[j], I->zl[j], I->dzl[j], I->ctl[j]);
                    if (R->hu[j]) GCORR(I->tu[j], I->dtu[j], I->zu[j], I->dzu[j], I->ctu[j]);
                }
                for (int i = 0; i < m; i++) {
                    if (R->gl[i]) GCORR(I->wl[i], I->dwl[i], I->vl[i], I->dvl[i], I->cwl[i]);
                    if (R->gu[i]) GCORR(I->wu[i], I->dwu[i], I->vu[i], I->dvu[i], I->cwu[i]);
                }
#undef GCORR
                direction(I);
                const double anew = step_to_boundary(I);
                if (!(anew >= 1.01 * amax)) { dir_copy(I, 1); break; }
                amax = anew; I->ncorr++;
            }
        }
        if (verbose > 3) {
            int jm = 0; for (int j = 1; j < n; j++) if (fabs(I->rd[j]) > fabs(I->rd[jm])) jm = j;
            int im = 0; for (int i = 1; i < m; i++) if (fabs(I->rs[i]) > fabs(I->rs[im])) im = i;
            if (n) qp_printf("     largest rd[%d] = %.2e (x %.3e, bounds %d%d, tl %.2e zl %.2e tu %.2e zu %.2e, Hdiag %.1e, Dx %.1e); rs[%d] = %.2e\n", jm, I->rd[jm], I->x[jm], R->hl[jm], R->hu[jm],
                          R->hl[jm] ? I->tl[jm] : 0, R->hl[jm] ? I->zl[jm] : 0, R->hu[jm] ? I->tu[jm] : 0, R->hu[jm] ? I->zu[jm] : 0, I->Hdiag[jm], I->Dx[jm], im, m ? I->rs[im] : 0);
        }
        if (verbose > 3 && I->N > 0) {
            double lo = 1e300, hi = 0; int nlo = 0;
            for (int j = 0; j < n; j++) { if (R->hl[j]) { double q = I->tl[j] * I->zl[j] / mu; lo = fmin(lo, q); hi = fmax(hi, q); nlo += q < 1e-2; } if (R->hu[j]) { double q = I->tu[j] * I->zu[j] / mu; lo = fmin(lo, q); hi = fmax(hi, q); nlo += q < 1e-2; } }
            for (int i = 0; i < m; i++) { if (R->gl[i]) { double q = I->wl[i] * I->vl[i] / mu; lo = fmin(lo, q); hi = fmax(hi, q); nlo += q < 1e-2; } if (R->gu[i]) { double q = I->wu[i] * I->vu[i] / mu; lo = fmin(lo, q); hi = fmax(hi, q); nlo += q < 1e-2; } }
            qp_printf("     products / mu: min %.1e max %.1e, %d below 1e-2; aaff %.3f sigma %.2e; steps tl %.2e zl %.2e tu %.2e zu %.2e wl %.2e vl %.2e wu %.2e vu %.2e\n", lo, hi, nlo, aaff, sigma,
                   maxstep(I->tl, I->dtl, R->hl, n), maxstep(I->zl, I->dzl, R->hl, n), maxstep(I->tu, I->dtu, R->hu, n), maxstep(I->zu, I->dzu, R->hu, n),
                   maxstep(I->wl, I->dwl, R->gl, m), maxstep(I->vl, I->dvl, R->gl, m), maxstep(I->wu, I->dwu, R->gu, m), maxstep(I->vu, I->dvu, R->gu, m));
        }
        alpha = fmin(1.0, 0.995 * amax);
        int bad = 0;
        for (int j = 0; j < n && !bad; j++) if (!isfinite(I->dx[j])) bad = 1;
        for (int i = 0; i < m && !bad; i++) if (!isfinite(I->dy[i])) bad = 1;
        if (bad || !isfinite(alpha)) {
            /* the factor too inaccurate for the solves: refactored with 100 times the regularization (twice at most).
             * (23 October 2026, the SOCP developer) The inequality rows get it as well: their diagonal 1/Ds + delta_ineq is
             * numerically zero at an active row late in the solve (QPCBOEI2, QP, with a larger dual shift at the start:
             * 4.8e-19 against x diagonals to 9.4e13; 15 pivots of the wrong sign replaced and the solve not finite), and a
             * retry that raised rho and delta alone factored the same matrix three times */
            if ((I->K || I->M || I->pcg) && nretry < 2 && I->rho < 1e-4) {
                I->rho *= 100; I->delta *= 100; nretry++;
                I->delta_ineq = fmax(I->delta_ineq, I->delta);
                if (verbose) qp_printf("the direction is not finite at iteration %d: refactored with the regularization %.0e\n", it, I->rho);
                goto refactor;
            }
            if (verbose) qp_printf("the direction is not finite at iteration %d\n", it);
            status = QP_NUMERICAL; break;
        }
        { static int undo2 = -1; if (undo2 < 0) { const char *e = getenv("BRISK_QPUNDO"); undo2 = e ? atoi(e) : 0; }
          if ((I->M && !I->nerho) || (undo2 && I->K)) state_copy(I, 0); }
        for (int j = 0; j < n; j++) {
            I->x[j] += alpha * I->dx[j];
            if (R->hl[j]) { I->tl[j] = fmax(I->x[j] - R->xl[j], (1 - alpha) * I->tl[j] * 1e-3); I->zl[j] += alpha * I->dzl[j]; }
            if (R->hu[j]) { I->tu[j] = fmax(R->xu[j] - I->x[j], (1 - alpha) * I->tu[j] * 1e-3); I->zu[j] += alpha * I->dzu[j]; }
        }
        for (int i = 0; i < m; i++) {
            I->y[i] += alpha * I->dy[i];
            if (R->eq[i]) continue;
            I->s[i] += alpha * I->ds[i];
            if (R->gl[i]) { I->wl[i] = fmax(I->s[i] - R->rl[i], (1 - alpha) * I->wl[i] * 1e-3); I->vl[i] += alpha * I->dvl[i]; }
            if (R->gu[i]) { I->wu[i] = fmax(R->ru[i] - I->s[i], (1 - alpha) * I->wu[i] * 1e-3); I->vu[i] += alpha * I->dvu[i]; }
        }
        (void)sigma;
    }
    Res->iter = it;
    extract(P, R, I, fixval, Res);
    Res->status = status;
    Res->obj = qp_objective(P, Res->x);
    qp_errors(P, Res->x, Res->y, Res->z, &Res->pres, &Res->dres, &Res->gap);
    if (status == QP_OPTIMAL && fmax(Res->pres, fmax(Res->dres, Res->gap)) > 10 * tol) status = Res->status = QP_REDUCED;
    if (status != QP_OPTIMAL && best < fmax(Res->pres, fmax(Res->dres, Res->gap))) {
        memcpy(Res->x, bx, sizeof(double) * n0); memcpy(Res->y, by, sizeof(double) * m0); memcpy(Res->z, bz, sizeof(double) * n0);
        Res->obj = qp_objective(P, Res->x);
        qp_errors(P, Res->x, Res->y, Res->z, &Res->pres, &Res->dres, &Res->gap);
        if (verbose) qp_printf("  the best point seen returned (largest error %.1e)\n", best);
    }
    if (status != QP_OPTIMAL && fmax(Res->pres, fmax(Res->dres, Res->gap)) <= fmax(1e-6, 100 * tol)) status = Res->status = QP_REDUCED;
    if (verbose) qp_printf("  %d factorizations (%.2f s), %d solves (%d refinement steps; %.2f s), %d centrality corrections\n", I->nfact, I->tfact, I->nsolve, I->nref, I->tsolve, I->ncorr);
    if (verbose && I->pcg) qp_printf("  the normal equations' solves: %ld conjugate-gradient iterations in all\n", I->ncg);
    free(bx); free(by); free(bz);
done:
    Res->time = now() - t0;
    if (I->K) schol_free(I->K);
    if (I->M) schol_free(I->M);
    if (I->Kspare) schol_free(I->Kspare);
    free(I->dcol); free(I->isd); free(I->offAd); free(I->offDd);
    free(I->offMd); free(I->offMp); free(I->mpst); free(I->dne); free(I->st0);
    free(I->grp); free(I->grq); free(I->grj); free(I->gmp); free(I->gmk); free(I->gmo); free(I->gw); free(I->Md); free(I->Wd); free(I->dmo);
    free(I->tord); free(I->tpar); free(I->tcol); free(I->ecol); free(I->uf); free(I->tL); free(I->tD); free(I->tE); free(I->tw); free(I->cgw);
    for (size_t k = 0; k < sizeof(vecs) / sizeof(vecs[0]); k++) free(*vecs[k]);
    for (size_t k = 0; k < sizeof(vecr) / sizeof(vecr[0]); k++) free(*vecr[k]);
    free(I->rhs); free(I->sol); free(I->res); free(I->work); free(I->tmp); free(I->hx); free(I->sv); free(I->gV); free(I->gZ);
    free(I->sgn); free(I->offHd); free(I->offH); free(I->offA); free(I->offR);
    red_free(R); free(fixval);
    return Res->status;
}
