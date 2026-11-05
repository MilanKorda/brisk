/* qp.c - the QP solver's common parts (the SOCP developer, 22 October 2026): options, results, the objective and the
 * errors on the problem as given, the choice of the method, and the solver object for warm starts (qp.h). */
#include "qp.h"
#include "brisk.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qp_fminmax.h"
#include "blas_one.h"
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define QP_CAT2(a, b) a##b
#define QP_CAT(a, b) QP_CAT2(a, b)
void QP_CAT(BLAS_PREFIX, dpotrf_)(const char *, const int *, double *, const int *, int *);
#ifdef _OPENMP
#include <omp.h>
#endif
static double qp_clock(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

void qp_default_opts(QpOpts *o) { o->tol = 1e-8; o->maxit = 0; o->verbose = 1; o->timelimit = 0; o->method = QP_METHOD_AUTO; o->threads = 0; }

void qp_prob_free(QpProb *P) {
    if (!P) return;
    free(P->Hp); free(P->Hi); free(P->Hx); free(P->Ap); free(P->Ai); free(P->Ax);
    free(P->c); free(P->rl); free(P->ru); free(P->xl); free(P->xu);
    if (P->rname) { for (int i = 0; i < P->m; i++) free(P->rname[i]); free(P->rname); }
    if (P->cname) { for (int j = 0; j < P->n; j++) free(P->cname[j]); free(P->cname); }
    free(P->qrow); free(P->qs); free(P->qi); free(P->qj); free(P->qv);
    memset(P, 0, sizeof(*P));
}
int (*qp_printf)(const char *fmt, ...) = printf;
void qp_res_free(QpRes *R) { if (!R) return; free(R->x); free(R->y); free(R->z); memset(R, 0, sizeof(*R)); }

const char *qp_status_name(int st) {
    switch (st) {
        case QP_OPTIMAL: return "OPTIMAL";
        case QP_PRIMAL_INFEASIBLE: return "PRIMAL INFEASIBLE";
        case QP_DUAL_INFEASIBLE: return "DUAL INFEASIBLE (unbounded)";
        case QP_MAXIT: return "ITERATION LIMIT";
        case QP_NUMERICAL: return "NUMERICAL DIFFICULTIES";
        case QP_TIME: return "TIME LIMIT";
        case QP_REDUCED: return "SOLVED TO REDUCED ACCURACY";
        case QP_INPUT: return "INPUT ERROR";
    }
    return "UNKNOWN";
}

/* y = H x with H given by its lower triangle */
void qp_hmul(const QpProb *P, const double *x, double *y) {
    for (int j = 0; j < P->n; j++) y[j] = 0;
    if (!P->Hp) return;
    for (int j = 0; j < P->n; j++)
        for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) {
            const int i = P->Hi[p]; const double v = P->Hx[p];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
}

double qp_objective(const QpProb *P, const double *x) {
    double *hx = malloc(sizeof(double) * ((size_t)P->n + 1));
    qp_hmul(P, x, hx);
    double f = P->c0;
    for (int j = 0; j < P->n; j++) f += x[j] * (0.5 * hx[j] + P->c[j]);
    free(hx);
    return P->maxim ? -f : f;
}

static int fin(double v) { return fabs(v) < QP_INF; }

/* the errors of (x, y, z) on P (the internal minimization):
 *   primal: max over rows and bounds of the violation / (1 + |bound|)
 *   dual:   || H x + c - A'y - z ||_inf / (1 + ||c||_inf), with y, z of the right sign at the bounds they belong to
 *   gap:    |f - d| / (1 + |f|), f the primal objective, d the dual objective
 *           d = -1/2 x'Hx + sum_rows (y+ rl - y- ru) + sum_cols (z+ xl - z- xu) + c0 (an infinite bound's term: 0, and its
 *           multiplier is counted as a dual violation) */
int qp_errors_pabs(void) { static int pabs = -1; if (pabs < 0) { const char *e = getenv("BRISK_QPPABS"); pabs = e ? atoi(e) : 0; } return pabs; }
static int qp_errdbg(void) { static int d = -1; if (d < 0) d = getenv("BRISK_QPERRDBG") ? 1 + (getenv("BRISK_QPERRDBG")[0] == '2') : 0; return d; }
/* (28 October 2026, the SOCP developer) the errors from the products: ax = A x (with the quadratic rows' parts), at the sums
 * |a_ij x_j| of a row (needed only at a violated row; NULL with BRISK_QPPABS=1), nt the rows' numbers of terms (NULL unless
 * BRISK_QPPABS=0), r = H x (overwritten), aty = A'y (with y_k Q_k x), dq; fobj (if not NULL) the objective f of the internal
 * minimization. The active-set method passes its own products (dense ones by dgemv): one definition of the measure */
void qp_errors_core(const QpProb *P, const double *x, const double *y, const double *z, const double *ax, const double *at,
                    const int *nt, double *r, const double *aty, double dq, double *epri, double *edual, double *egap, double *fobj) {
    const int n = P->n, m = P->m;
    double ep = 0;
    const double uu = 1.1102230246251565e-16;      /* (the unit roundoff of double precision, 2^-53) */
    for (int i = 0; i < m; i++) {
        if (nt) {          /* (the violation beyond the rounding floor of the row's sum, against 1 + |b|) */
            const double ku = (nt[i] + 1) * uu, fl = (ku < 1 ? ku / (1 - ku) : 1.0) * at[i];
            if (fin(P->rl[i]) && ax[i] < P->rl[i]) ep = fmax(ep, fmax(0.0, P->rl[i] - ax[i] - fl) / (1 + fabs(P->rl[i])));
            if (fin(P->ru[i]) && ax[i] > P->ru[i]) ep = fmax(ep, fmax(0.0, ax[i] - P->ru[i] - fl) / (1 + fabs(P->ru[i])));
            continue;
        }
        const double ts = at ? at[i] : 0;
        if (fin(P->rl[i]) && ax[i] < P->rl[i]) ep = fmax(ep, (P->rl[i] - ax[i]) / (1 + fmax(fabs(P->rl[i]), ts)));
        if (fin(P->ru[i]) && ax[i] > P->ru[i]) ep = fmax(ep, (ax[i] - P->ru[i]) / (1 + fmax(fabs(P->ru[i]), ts)));
    }
    if (qp_errdbg()) { double er = 0, eb = 0; int ir = -1, jb = -1;
        for (int i = 0; i < m; i++) { const double ts = at ? at[i] : 0; double v = 0;
            if (fin(P->rl[i]) && ax[i] < P->rl[i]) v = (P->rl[i] - ax[i]) / (1 + fmax(fabs(P->rl[i]), ts));
            if (fin(P->ru[i]) && ax[i] > P->ru[i]) v = (ax[i] - P->ru[i]) / (1 + fmax(fabs(P->ru[i]), ts));
            if (v > er) { er = v; ir = i; } }
        for (int j = 0; j < n; j++) { double v = 0;
            if (fin(P->xl[j]) && x[j] < P->xl[j]) v = (P->xl[j] - x[j]) / (1 + fabs(P->xl[j]));
            if (fin(P->xu[j]) && x[j] > P->xu[j]) v = (x[j] - P->xu[j]) / (1 + fabs(P->xu[j]));
            if (v > eb) { eb = v; jb = j; } }
        if (ir >= 0 && qp_errdbg() == 2) for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (P->Ai[p] == ir) fprintf(stderr, "        [column %d: a %g, x %.6e, bounds %.3e %.3e]\n", j, P->Ax[p], x[j], P->xl[j], P->xu[j]);
        fprintf(stderr, "     [primal: row %d %.1e (activity %.6e, bounds %.6e %.6e, terms %.3e); column %d %.1e (x %.9e, bounds %.9e %.9e)]\n", ir, er, ir >= 0 ? ax[ir] : 0, ir >= 0 ? P->rl[ir] : 0, ir >= 0 ? P->ru[ir] : 0, ir >= 0 && at ? at[ir] : 0, jb, eb, jb >= 0 ? x[jb] : 0, jb >= 0 ? P->xl[jb] : 0, jb >= 0 ? P->xu[jb] : 0); }
    for (int j = 0; j < n; j++) {
        if (fin(P->xl[j]) && x[j] < P->xl[j]) ep = fmax(ep, (P->xl[j] - x[j]) / (1 + fabs(P->xl[j])));
        if (fin(P->xu[j]) && x[j] > P->xu[j]) ep = fmax(ep, (x[j] - P->xu[j]) / (1 + fabs(P->xu[j])));
    }
    double xhx = 0, cn = 0; for (int j = 0; j < n; j++) { xhx += x[j] * r[j]; cn = fmax(cn, fabs(P->c[j])); }
    /* the dual residual relative to the size of its terms, |Hx + c - A'y - z| / (1 + max(|Hx|, |c|, |A'y|, |z|)), the
     * definition of MOSEK, Gurobi, OSQP and PIQP: where the optimal multipliers are unbounded (a variable forced to its
     * bound by the constraints) the iterates' multipliers grow, 1e9 on QSCFXM2, and their rounding alone is 1e-7 */
    double tn = cn;
    for (int j = 0; j < n; j++) { tn = fmax(tn, fabs(r[j])); tn = fmax(tn, fabs(z[j])); r[j] += P->c[j]; }
    for (int j = 0; j < n; j++) { const double a = aty[j]; r[j] -= a; tn = fmax(tn, fabs(a)); }
    double dsum = P->c0 - 0.5 * xhx + dq, dviol = 0;
    for (int j = 0; j < n; j++) {
        r[j] -= z[j];
        if (z[j] > 0) { if (fin(P->xl[j])) dsum += z[j] * P->xl[j]; else dviol = fmax(dviol, z[j]); }
        if (z[j] < 0) { if (fin(P->xu[j])) dsum += z[j] * P->xu[j]; else dviol = fmax(dviol, -z[j]); }
    }
    for (int i = 0; i < m; i++) {
        if (y[i] > 0) { if (fin(P->rl[i])) dsum += y[i] * P->rl[i]; else dviol = fmax(dviol, y[i]); }
        if (y[i] < 0) { if (fin(P->ru[i])) dsum += y[i] * P->ru[i]; else dviol = fmax(dviol, -y[i]); }
    }
    double ed = 0; int jed = -1; for (int j = 0; j < n; j++) if (fabs(r[j]) > ed) { ed = fabs(r[j]); jed = j; }
    if (qp_errdbg() && jed >= 0) qp_printf("     [errors] the largest dual residual %.3e at column %d (z %.3e)\n", ed, jed, z[jed]);
    ed = fmax(ed, dviol) / (1 + tn);
    double f = P->c0 + 0.5 * xhx; for (int j = 0; j < n; j++) f += P->c[j] * x[j];
    *epri = ep; *edual = ed; *egap = fabs(f - dsum) / (1 + fabs(f));
    if (fobj) *fobj = f;
    /* (1 November 2026, the SOCP developer; the owner's QP benchmark, audit A1) a point or a product that is not finite has
     * infinite errors here, in the one definition every method uses: the maxima above take fmax, which drops a NaN, and
     * the active-set method, which calls this core with its own products, returned OPTIMAL with a NaN x for a NaN cost */
    {   int bad = !isfinite(f) || !isfinite(dsum) || isnan(ep) || isnan(ed);
        for (int j = 0; j < n && !bad; j++) bad = !isfinite(x[j]) || !isfinite(z[j]) || !isfinite(r[j]);
        for (int i = 0; i < m && !bad; i++) bad = !isfinite(y[i]) || !isfinite(ax[i]);
        if (bad) { *epri = *edual = *egap = INFINITY; } }
    if (qp_errdbg()) {
        double zb = 0, yb = 0, zfree = 0;
        for (int j = 0; j < n; j++) { if (z[j] > 0 && fin(P->xl[j])) zb += z[j] * P->xl[j]; if (z[j] < 0 && fin(P->xu[j])) zb += z[j] * P->xu[j]; if (fin(P->xl[j]) && fin(P->xu[j]) && P->xl[j] == P->xu[j]) zfree += z[j] * P->xl[j]; }
        for (int i = 0; i < m; i++) { if (y[i] > 0 && fin(P->rl[i])) yb += y[i] * P->rl[i]; if (y[i] < 0 && fin(P->ru[i])) yb += y[i] * P->ru[i]; }
        qp_printf("     [errors] f %.10e d %.10e: c0 %.4e, -1/2 x'Hx %.10e, bounds %.10e (fixed %.4e), rows %.10e\n", f, dsum, P->c0, -0.5 * xhx, zb, zfree, yb);
    }
}
void qp_errors(const QpProb *P, const double *x, const double *y, const double *z, double *epri, double *edual, double *egap) {
    const int n = P->n, m = P->m;
    /* (22 October 2026) a point with a non-finite entry has infinite errors: the maxima below take fmax, which drops a
     * NaN, and a NaN step of the QCQP polishing (QPLIB_2456) read as errors 0, 0, NaN and was returned as OPTIMAL */
    {   int bad = 0;
        for (int j = 0; j < n && !bad; j++) bad = !isfinite(x[j]) || !isfinite(z[j]);
        for (int i = 0; i < m && !bad; i++) bad = !isfinite(y[i]);
        if (bad) { *epri = *edual = *egap = INFINITY; return; } }
    double *ax = calloc((size_t)m + 1, sizeof(double)), *r = malloc(sizeof(double) * ((size_t)n + 1));
    /* (24 October 2026, the SOCP developer) a row's violation against 1 + |b|, as the measure is defined, less what the
     * rounding of the row's own sum can produce: a computed sum of k terms is wrong by at most gamma_k sum |terms|,
     * gamma_k = k u / (1 - k u) (u the unit roundoff; Higham, Accuracy and Stability of Numerical Algorithms, 3.1). So
     * max(0, |a'x - b| - gamma_k sum |a_j x_j|) / (1 + |b|). QPLIB_9002 (QP: a network, right-hand sides 0, flows up to 8e9)
     * cannot get below 1e-6 against 1 + |b| alone (137 iterations, numerical difficulties); the measure of 23 October,
     * |a'x - b| / (1 + max(|b|, sum |a_j x_j|)), let QSCFXM3 (QP) stop with violations of 4.5e-6 against 1 + |b| (two
     * iterations early). BRISK_QPPABS=1: against 1 + |b| alone; 2: the measure of 23 October */
    const int pabs = qp_errors_pabs();
    double *at = pabs == 1 ? NULL : calloc((size_t)m + 1, sizeof(double));
    int *nt = pabs == 0 ? calloc((size_t)m + 1, sizeof(int)) : NULL;      /* (the rows' numbers of terms) */
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { ax[P->Ai[p]] += P->Ax[p] * x[j]; if (at) at[P->Ai[p]] += fabs(P->Ax[p] * x[j]); if (nt) nt[P->Ai[p]]++; }
    /* the quadratic constraints' parts: 1/2 x'Q_k x in the row's activity, y_k Q_k x in the gradient of y'(row) */
    double *qx = P->nqc ? calloc((size_t)n + 1, sizeof(double)) : NULL, dq = 0;   /* dq: 1/2 sum_k y_k x'Q_k x, in the dual value */
    for (int k = 0; k < P->nqc; k++) {
        const int i = P->qrow[k]; double xq = 0;
        for (int e = P->qs[k]; e < P->qs[k + 1]; e++) {
            const int a = P->qi[e], b = P->qj[e]; const double v = P->qv[e];
            xq += (a == b ? 0.5 : 1.0) * v * x[a] * x[b];
            if (at) at[i] += fabs((a == b ? 0.5 : 1.0) * v * x[a] * x[b]);
            if (nt) nt[i]++;
            qx[a] += y[i] * v * x[b]; if (a != b) qx[b] += y[i] * v * x[a];
        }
        ax[i] += xq; dq += y[i] * xq;
    }
    qp_hmul(P, x, r);
    double *aty = malloc(sizeof(double) * ((size_t)n + 1));
    for (int j = 0; j < n; j++) { double a = qx ? qx[j] : 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) a += P->Ax[p] * y[P->Ai[p]]; aty[j] = a; }
    qp_errors_core(P, x, y, z, ax, at, nt, r, aty, dq, epri, edual, egap, NULL);
    free(ax); free(at); free(nt); free(r); free(qx); free(aty);
}

/* (23 October 2026) the doubleton equations substituted out first (qpdbl.c), the reduced QP solved by the interior-point
 * method and its solution mapped back; the errors and the objective are those of the problem as given. When the reduced
 * solve is not optimal, or the errors of the mapped point miss the tolerance, the problem as given is solved instead
 * (certificates of infeasibility come from it). BRISK_QPNODBL=1: off */
typedef struct QpDbl QpDbl;
QpDbl *qpdbl_presolve(const QpProb *P, QpProb *P2, int *nelim);
void qpdbl_postsolve(const QpDbl *D, const QpProb *P, const double *x2, const double *y2, const double *z2, double *x, double *y, double *z);
void qpdbl_free(QpDbl *D);
/* (1 November 2026, the owner's QP benchmark, audit A14) also the persistent model's interior-point route (qpas.c), which
 * had called qp_solve_ipm directly and so solved a problem with doubleton rows differently from the one-shot call; the
 * time is the whole call's (it was the reduced problem's solve alone) */
int qp_solve_ipm_dbl(const QpProb *P, const QpOpts *o, QpRes *R) {
    const double t0d = qp_clock();
    static int off = -1; if (off < 0) { const char *e = getenv("BRISK_QPNODBL"); off = e && atoi(e) == 1; }
    const double tdb_ = qp_clock();
    QpProb P2; int ne = 0; QpDbl *D = off ? NULL : qpdbl_presolve(P, &P2, &ne);
    if (getenv("BRISK_QPSTAMP")) fprintf(stderr, "[qp stamp: the doubleton presolve %.3f s]\n", qp_clock() - tdb_);
    if (!D) return qp_solve_ipm(P, o, R);
    if (o->verbose) qp_printf("presolve: %d doubleton equality rows substituted out (columns %d -> %d, rows %d -> %d)\n", ne, P->n, P2.n, P->m, P2.m);
    QpRes R2; memset(&R2, 0, sizeof R2);
    const int st = qp_solve_ipm(&P2, o, &R2);
    int ok = 0;
    memset(R, 0, sizeof *R);
    if (st == QP_OPTIMAL && R2.x && R2.y && R2.z) {
        R->x = calloc((size_t)P->n + 1, sizeof(double)); R->y = calloc((size_t)P->m + 1, sizeof(double)); R->z = calloc((size_t)P->n + 1, sizeof(double));
        if (R->x && R->y && R->z) {
            qpdbl_postsolve(D, P, R2.x, R2.y, R2.z, R->x, R->y, R->z);
            double e1, e2, e3; qp_errors(P, R->x, R->y, R->z, &e1, &e2, &e3);
            const double tol = o->tol > 0 ? o->tol : 1e-8;
            if (e1 <= tol && e2 <= tol && e3 <= tol) {
                ok = 1; R->status = QP_OPTIMAL; R->pres = e1; R->dres = e2; R->gap = e3; R->obj = qp_objective(P, R->x);
                R->iter = R2.iter; R->time = qp_clock() - t0d; R->resolves = R2.resolves; R->t_first = R2.t_first;
            } else if (o->verbose) qp_printf("presolve: the mapped point misses the tolerance on the problem as given (primal %.1e, dual %.1e, gap %.1e): solved as given\n", e1, e2, e3);
        }
    }
    qp_res_free(&R2); qp_prob_free(&P2); qpdbl_free(D);
    if (!ok) { qp_res_free(R); return qp_solve_ipm(P, o, R); }
    return R->status;
}
/* (23 October 2026, the SOCP developer) the default method for a small strictly convex QP: the dense active-set method
 * when H is positive definite (its own Cholesky test) and its work is small: 2 n^3 (the Cholesky factor of H and J =
 * L^-T, the updates of J and R as constraints enter, at most about n of them) plus 2 n (nnz(A) + n + m) (each entry's
 * search for a violated constraint) at most 4e6 flops (2e6 until 24 October 2026). Below that the interior-point method's own cost (the analysis of
 * its system, a dozen factorizations and their solves) is the larger; on the QPs of n <= 300 of the five QP sets the
 * active-set method's time over the interior-point method's, both in C on one thread, was 0.01-0.9 on every QP within
 * the bound (LIPMWALK 0.04, WHLIPBAL 0.3, DUALC 0.02-0.05, KSIP 0.03, the HS problems 0.01-0.07; all QPs) and above 1
 * beyond it from 2e6 on (CVXQP2_S 1.4 at 2.0e6, QISRAEL 1.9 at 6.5e6, DPKLO1 4.0 at 5.2e6). A positive semidefinite H
 * stays with the interior-point method (the active-set method needs its proximal-point loop for it, with no bound on
 * its solves). When the active-set method does not end optimal, the interior-point method solves the problem (the
 * Summary says so). -qpipm or -qpas: that method */
static int as_small(const QpProb *P, double *wipm_out);
/* the default's size bound alone (the QP model's automatic choice, qpas.c, which has no work budget): within the bound, not
 * by the cost model's budgeted attempt */
int qp_as_small(const QpProb *P) { double w = 0; return as_small(P, &w) && w == 0; }
static int as_small(const QpProb *P, double *wipm_out) {
    const double n = P->n, w = 2 * n * n * n + 2 * n * ((double)P->Ap[P->n] + n + P->m);
    *wipm_out = 0;
    /* (24 October 2026) 4e6 instead of 2e6: measured again after the active-set method's J = L^-T was made three times
     * cheaper (bench/qp/small/small_ab2.txt, 129 QPs of n <= 300, both methods in C): the rule's choice is the faster
     * method on 123 of them at 4e6 against 117 at 2e6, the sum of the chosen times 293 against 301 ms; between the two
     * bounds the active-set method takes EQQP_100 (0.79 against 2.4 ms) and DUAL3 (0.65 against 3.8 ms; QPs), the
     * others there have H only semidefinite. No status changed on the 288 QPs */
    static double cap = -1; if (cap < 0) { const char *e = getenv("BRISK_QPASAUTO"); cap = e ? atof(e) : 4e6; }   /* (0: off) */
    if (P->n <= 0) return 0;
    if (w <= cap) return 1;
    /* (28 October 2026, the SOCP developer, after the owner's independent QP benchmark: B005_A_mpc_long_condensed, a dense
     * condensed MPC QP of n = 1,920 and 7,704 rows, 48.6 s by the interior-point method (14 iterations, each factoring a
     * system of 9,600 unknowns) against 3.2 s by the active-set method) beyond the bound, a cost model in operations: the
     * interior-point method about 15 iterations, each factoring at least the denser of the rows' cliques and H's pattern,
     * W_ipm = 100 max(sum_i r_i^2, sum_j h_j^2) + 1000 size (r_i the entries of row i of A, h_j those of column j of H; the
     * factors measured, below; 5 max(...) until the night of 28 October); the active-set
     * method's setup (the factor of H and J = L^-T, the rows' products) W_as = 2 n^3 / 3 + 2 n nnz(A). When W_as <= W_ipm the
     * active-set method is tried with W_ipm as its budget, counted in its iterations as they run (qp_solve_activeset_budget:
     * the iterations a budget of W_ipm - W_as allows, each charged its own work); when they run out, the interior-point
     * method solves the problem. (A first form predicted the iterations from the constraints the unconstrained minimum
     * violates: hundreds on condensed MPC QPs that the method solves in one or two, B010_R_mpc_integrator 22.6 ms by the
     * interior-point method against 1.8 ms.)
     * BRISK_QPASAUTOIPM=0: the bound alone */
    static int ipmr = -1; if (ipmr < 0) { const char *e = getenv("BRISK_QPASAUTOIPM"); ipmr = !(e && atoi(e) == 0); }
    if (!ipmr || cap <= 0 || n * n > 4e8) return 0;
    double sr = 0, sh = 0;
    { int *rc = calloc((size_t)P->m + 1, sizeof(int));
      if (!rc) return 0;
      for (int p = 0; p < P->Ap[P->n]; p++) rc[P->Ai[p]]++;
      for (int i = 0; i < P->m; i++) sr += (double)rc[i] * rc[i];
      free(rc); }
    if (P->Hp) {      /* (the columns of the whole H, both triangles, from its lower triangle) */
        int *hc = calloc((size_t)P->n + 1, sizeof(int));
        if (!hc) return 0;
        for (int j = 0; j < P->n; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) { hc[j]++; if (P->Hi[p] != j) hc[P->Hi[p]]++; }
        for (int j = 0; j < P->n; j++) sh += (double)hc[j] * hc[j];
        free(hc);
    }
    /* (28 October 2026, the night) the estimate calibrated on the owner's benchmark: both methods measured on its 534 QPs of
     * n <= 2,000 (one thread, the medians of 3-9 solves; bench/qp_ext/ab2.py): the interior-point method's iteration costs
     * 0.24-1.1 ns per unit of max(sum r_i^2, sum h_j^2) and a fixed part per unit of size, against about 0.4 ns per counted
     * operation of the active-set method; 100 max(...) + 1000 (n + m + nnz(A) + nnz(H)) for its 6-15 iterations. Simulated
     * on the 534 QPs with the budget below: the chosen method's time over the faster one's, geometric mean, 1.219 with
     * 5 max(...) (B010_R_mpc_integrator: 10.0 ms by the interior-point method against 1.1 ms) -> 1.048, six attempts of the
     * active-set method abandoned to the interior-point method */
    const double size = n + (double)P->m + (double)P->Ap[P->n] + (P->Hp ? (double)P->Hp[P->n] : 0.0);
    double wipm = 100.0 * fmax(sr, sh) + 1000.0 * size; const double wel = qp_elim_setup_work(P);
    /* (2 November 2026, the SOCP developer) a dense H (its pattern a quarter of the pairs or more): the interior-point method
     * factors a dense system at the dense kernels' rate, which the fit above (on sparse factorizations) overstates by 1.2-4
     * times, more with n; measured on the owner's benchmark's dense-H QPs (bench/run_1101n/choice_1102.jsonl) and the SVM
     * dual of 3,000 variables (QP: the active-set method 4,541 iterations, 27.6 s, against the interior-point method's 7.1 s),
     * its time in the active-set method's units is about 25 (n+m)^3 + 8500 (n+m)^2 (n = 128: 2.1e8, 3,000: 7.5e11), the
     * estimate's upper end taken; so the active-set method's budget, a failed attempt included, stays near the interior-point
     * method's own time. BRISK_QPDENSEEST=0: off */
    { static int dest = -1; if (dest < 0) { const char *e = getenv("BRISK_QPDENSEEST"); dest = !(e && atoi(e) == 0); }
      if (dest && sh >= 0.25 * n * n * n) { const double N = n + (double)P->m, wd = 25.0 * N * N * N + 8500.0 * N * N; if (wd < wipm) wipm = wd; } }
    double was = 2.0 * n * n * n / 3.0 + 2.0 * n * (double)P->Ap[P->n];
    if (wel >= 0 && wel < was) was = wel;      /* (the equality rows eliminated: the sparse form of an MPC QP) */
    if (was > wipm) return 0;
    *wipm_out = wipm;
    return 1;
}
/* (24 October 2026, the SOCP developer) the exact part of the test for a positive semidefinite matrix sg M, M symmetric
 * given by ne entries (qi[e], qj[e], qv[e]) of its lower triangle (duplicates summed): such a matrix has no negative
 * diagonal entry, and a zero diagonal entry only in a zero row (its 2 x 2 principal minors are not negative). Returns 0
 * when M passes, 1 for a negative diagonal entry (*j1 its column), 2 for a zero diagonal entry *j1 whose row holds the
 * entry of column *j2. No tolerance: the matrix as read; a positive diagonal with off-diagonal entries too large for it
 * is left to the methods (the QCQP's factorization tests its matrices) */
static int cmp_int_qp(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }
/* (24 October 2026, the SOCP developer) the test on a workspace: d (n, zero) and part (n, -1) on entry and on return, only
 * the matrix's own entries touched; the touched columns checked in ascending order, as a scan of all n would (the same first
 * column reported). A scan of all n for each quadratic row was 4e9 steps on a QCQP of 44,700 rows over 91,650 variables
 * (5 s of a 17 s run, LOSSFLOW_150) */
static int psd_exact_ws(int ne, const int *qi, const int *qj, const double *qv, double sg, int *j1, int *j2, double *d, int *part, int *tl) {
    int nt = 0;
    for (int e = 0; e < ne; e++) {
        const int i = qi[e], j = qj[e];
        if (d[i] == 0 && part[i] < 0) tl[nt++] = i;
        if (j != i && d[j] == 0 && part[j] < 0) tl[nt++] = j;
        if (i == j) d[j] += sg * qv[e];
        else if (qv[e] != 0) { if (part[i] < 0) part[i] = j; if (part[j] < 0) part[j] = i; }
    }
    /* (a column whose diagonal sums to exactly 0 and has no product may appear twice: harmless) */
    qsort(tl, (size_t)nt, sizeof(int), cmp_int_qp);
    int rc = 0;
    for (int t = 0; t < nt && !rc; t++) {
        const int j = tl[t];
        if (d[j] < 0) { rc = 1; *j1 = j; *j2 = -1; }
        else if (d[j] == 0 && part[j] >= 0) { rc = 2; *j1 = j; *j2 = part[j]; }
    }
    for (int t = 0; t < nt; t++) { d[tl[t]] = 0; part[tl[t]] = -1; }
    return rc;
}
int qp_psd_exact(int n, int ne, const int *qi, const int *qj, const double *qv, double sg, int *j1, int *j2) {
    double *d = calloc((size_t)n + 1, sizeof(double)); int *part = malloc(sizeof(int) * ((size_t)n + 1)), *tl = malloc(sizeof(int) * (2 * (size_t)ne + 1));
    if (!d || !part || !tl) { free(d); free(part); free(tl); return 0; }
    for (int j = 0; j < n; j++) part[j] = -1;
    const int rc = psd_exact_ws(ne, qi, qj, qv, sg, j1, j2, d, part, tl);
    free(d); free(part); free(tl);
    return rc;
}
/* (1 November 2026, the SOCP developer; the owner's QP benchmark, audit A3) the sufficient part of the test: the Cholesky
 * factorization of H + 1e-9 diag(|H|) (each diagonal entry raised by 1e-9 of itself: the test of the unit-diagonal scaling
 * D^-1/2 H D^-1/2 for its smallest eigenvalue >= -1e-9, whatever the units of the variables; the rounding of the
 * factorization is about n u of the scaled matrix, far below the margin). Only for an H with products (the necessary
 * part decides a diagonal H exactly) and within 1e11 flops of the factor (a few seconds; one factorization of the
 * interior-point method's system costs at least as much); beyond that the test is not made and said so under -v. H =
 * [1 2; 2 1] (eigenvalues -1 and 3) had passed the necessary part and been solved to a saddle point reported OPTIMAL.
 * 0 positive semidefinite within the margin, 1 not, -1 not tested */
static int qp_psd_chol(int n, const int *Hp, const int *Hi, const double *Hx) {
    int *deg = calloc((size_t)n + 1, sizeof(int)), *mk = malloc(sizeof(int) * ((size_t)n + 1));
    if (!deg || !mk) { free(deg); free(mk); return -1; }
    long off = 0;
    for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hi[p] != j && Hx[p] != 0) { deg[j]++; deg[Hi[p]]++; off++; }
    if (off == 0) { free(deg); free(mk); return 0; }
    /* (1 November 2026, the SOCP developer; the owner's QP benchmark, the small MPC QPs) up to 128 columns the same test on a
     * dense array (the same shift, a pivot that is not positive fails): through the sparse factorization's pattern, analysis
     * and scatter it took a fifth of the solve of a quadrotor MPC QP of 92 variables (its reduced Hessian: 20 x 20). Positive
     * semidefiniteness does not depend on the order; only the rounding at the margin can differ */
    /* (1 November 2026, 00:10) and a dense H of up to 2,048 columns (its products a quarter of the pairs or more) by LAPACK's
     * blocked Cholesky factorization (OpenBLAS on one thread, blas_one.h: the verdict at the margin independent of the host's
     * thread count; info > 0, a leading minor not positive definite, is the failure of a pivot): the sparse route's minimum-
     * degree ordering and analysis of the complete graph cost 4.5 times its factorization (the benchmark's portfolio QP of
     * 512 variables, a dense H: 107 million instructions, a fifth of the interior-point solve) */
    const int dense_ = n <= 128 || (n <= 2048 && 2.0 * (double)off >= 0.25 * (double)n * (n - 1));
    if (dense_ && !getenv("BRISK_QPPSDSPARSE")) {
        free(deg); free(mk);
        double *L = calloc((size_t)n * n + 1, sizeof(double)); if (!L) return -1;
        for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) { const int i = Hi[p]; if (i >= j) L[i + (size_t)j * n] += Hx[p]; else L[j + (size_t)i * n] += Hx[p]; }
        for (int j = 0; j < n; j++) { double *d = &L[j + (size_t)j * n]; *d += 1e-9 * (fabs(*d) > 0 ? fabs(*d) : 1.0); }
        int rc = 0;
        if (n > 128) {
            int info = 0;
            blas_one_begin(); QP_CAT(BLAS_PREFIX, dpotrf_)("L", &n, L, &n, &info); blas_one_end();
            free(L);
            return info > 0 ? 1 : info < 0 ? -1 : 0;
        }
        for (int j = 0; j < n && !rc; j++) {
            double *cj = L + (size_t)j * n;
            for (int k = 0; k < j; k++) { const double f = L[j + (size_t)k * n]; if (f == 0) continue; const double *ck = L + (size_t)k * n; for (int i = j; i < n; i++) cj[i] -= f * ck[i]; }
            const double d = cj[j];
            if (!(d > 0)) { rc = 1; break; }
            const double s = sqrt(d), is = 1.0 / s; cj[j] = s;
            for (int i = j + 1; i < n; i++) cj[i] *= is;
        }
        free(L);
        return rc;
    }
    int **nbr = malloc(sizeof(int *) * ((size_t)n + 1)), *pool = malloc(sizeof(int) * (2 * (size_t)off + 1));
    if (!nbr || !pool) { free(deg); free(mk); free(nbr); free(pool); return -1; }
    { long o = 0; for (int j = 0; j < n; j++) { nbr[j] = pool + o; o += deg[j]; deg[j] = 0; } }
    for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hi[p] != j && Hx[p] != 0) { const int i = Hi[p]; nbr[j][deg[j]++] = i; nbr[i][deg[i]++] = j; }
    /* (duplicate entries of a column give a repeated neighbour: removed, the analysis expects a set) */
    for (int j = 0; j < n; j++) mk[j] = -1;
    for (int j = 0; j < n; j++) { int d = 0; for (int c = 0; c < deg[j]; c++) { const int i = nbr[j][c]; if (mk[i] != j) { mk[i] = j; nbr[j][d++] = i; } } deg[j] = d; }
    /* first the minimum-degree ordering alone, stopped once its running count of the work passes the budget (its count is
     * that of an LU factorization, about twice the Cholesky factorization's): beyond it no analysis is made (QPUNB_20000,
     * QP: 20,000 columns, 1.3 million entries in H, the analysis 6 s to learn that the factor exceeds the budget) */
    {   int schol_amd_estimate_cap(int m, const int *deg, int *const *nbr, const volatile double *cap, double *flops_out, double *nnz_out);
        const volatile double acap = 2e11; double afl = 0, anz = 0;
        const int e_ = schol_amd_estimate_cap(n, deg, nbr, &acap, &afl, &anz);
        if (e_ == 2 || (e_ == 0 && afl > 2e11)) { free(deg); free(mk); free(nbr); free(pool); return -1; } }
    const int oamd = schol_set_amd(2);
    SChol *S = schol_analyze_adj(n, deg, nbr, (size_t)(sqrt(1e11 * (double)n) + n));
    schol_set_amd(oamd);
    int rc = -1;
    if (S && schol_flops(S) <= 1e11) {
        schol_zero(S);
        for (int j = 0; j < n; j++) for (int p = Hp[j]; p < Hp[j + 1]; p++) if (Hx[p] != 0 || Hi[p] == j) schol_add(S, (size_t)Hi[p], (size_t)j, Hx[p]);
        rc = schol_factor(S, 1e-9) ? 1 : 0;
    }
    if (S) schol_free(S);
    free(deg); free(mk); free(nbr); free(pool);
    return rc;
}
/* the objective's H and the constraints' Q_k (on the side each row bounds) by that test: 0, or QP_INPUT with the message */
int qp_convex_exact(const QpProb *P) {
    const int n = P->n;
    int j1 = -1, j2 = -1, rc = 0;
    const char *what = "the objective's quadratic part";
    char row[96] = "";
    if (P->Hp && P->Hp[n] > 0) {
        const int ne = P->Hp[n];
        int *hj = malloc(sizeof(int) * ((size_t)ne + 1));
        if (hj) { for (int j = 0; j < n; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) hj[p] = j; rc = qp_psd_exact(n, ne, P->Hi, hj, P->Hx, 1.0, &j1, &j2); free(hj); }
        if (!rc) {
            const int c_ = qp_psd_chol(n, P->Hp, P->Hi, P->Hx);
            if (c_ == 1) { fprintf(stderr, "brisk: the problem is not convex: the objective's quadratic part is not positive semidefinite (its Cholesky factorization fails)\n"); return QP_INPUT; }
            if (c_ < 0 && getenv("BRISK_QPDEPDBG")) fprintf(stderr, "[convexity: the Cholesky test of H not made (beyond 1e11 flops)]\n");
        }
    }
    double *dw = NULL; int *pw = NULL, *tw = NULL;
    if (P->nqc > 0 && !rc) {
        int mx = 0; for (int k = 0; k < P->nqc; k++) if (P->qs[k + 1] - P->qs[k] > mx) mx = P->qs[k + 1] - P->qs[k];
        dw = calloc((size_t)n + 1, sizeof(double)); pw = malloc(sizeof(int) * ((size_t)n + 1)); tw = malloc(sizeof(int) * (2 * (size_t)mx + 1));
        if (!dw || !pw || !tw) { free(dw); free(pw); free(tw); return 0; }
        for (int j = 0; j < n; j++) pw[j] = -1;
    }
    for (int k = 0; k < P->nqc && !rc; k++) {
        const int i = P->qrow[k];
        if (fin(P->rl[i]) == fin(P->ru[i])) continue;        /* (a free row; two finite sides: the QCQP refuses it itself) */
        rc = psd_exact_ws(P->qs[k + 1] - P->qs[k], P->qi + P->qs[k], P->qj + P->qs[k], P->qv + P->qs[k], fin(P->ru[i]) ? 1.0 : -1.0, &j1, &j2, dw, pw, tw);
        if (rc) { what = "the quadratic part of row"; snprintf(row, sizeof row, " %s%s", P->rname && P->rname[i] ? P->rname[i] : "?", fin(P->ru[i]) ? "" : " (its lower side, negated)"); }
    }
    free(dw); free(pw); free(tw);
    if (!rc) return 0;
    /* (the columns by name, or by their index from 0 for a problem given without names) */
    char c1[80], c2[80];
    if (P->cname && P->cname[j1]) snprintf(c1, sizeof c1, "%s", P->cname[j1]); else snprintf(c1, sizeof c1, "%d (from 0)", j1);
    if (j2 >= 0 && P->cname && P->cname[j2]) snprintf(c2, sizeof c2, "%s", P->cname[j2]); else snprintf(c2, sizeof c2, "%d (from 0)", j2);
    if (rc == 1) fprintf(stderr, "brisk: the problem is not convex: %s%s has a negative square term (column %s)\n", what, row, c1);
    else fprintf(stderr, "brisk: the problem is not convex: %s%s has no square term of column %s but a product of it with column %s\n",
                 what, row, c1, c2);
    return QP_INPUT;
}

/* (1 November 2026, the SOCP developer; the owner's QP benchmark, audits A2 and A6) one deadline for a whole QP solve: the
 * outermost entry (qp_solve, qp_resolve, a method called directly) sets it from its time limit, the inner solves (the
 * active-set method tried first, the doubleton presolve's reduced problem and its fallback, the interior-point method's
 * re-solve with other dependent rows dropped, the QCQP's rounds) only read it, so that their own clocks no longer each
 * start the full limit again; the methods test it inside their iterations (the active-set method's inner loop too,
 * which had tested its limit only between proximal-point solves: a 500-variable box QP ran 0.41 s at a limit of 1 ms) */
static _Thread_local double g_qp_deadline = 0; static _Thread_local int g_qp_dl_depth = 0;
/* (1 November 2026, the SOCP developer; the owner's QP benchmark, the mass-spring MPC plants) the processor flushes subnormal
 * results to zero and reads subnormal operands as zero (x86: MXCSR's FTZ and DAZ bits) for the whole solve, on the calling
 * thread and on the OpenMP threads, the mode before restored at the end. A product of two tiny normal numbers (the exact
 * zero-order hold of a stiff spring chain has entries down to 1e-300) is a subnormal number, and arithmetic on those takes a
 * microcode assist of about a hundred cycles: the 128-mass plant (QP, 51,456 variables) factored at 3.5 GFlop/s against 9.4
 * on the 96-mass plant. Values below 2.2e-308 are zero for every quantity of the solve (the tolerances are relative and
 * above 1e-16). The mode is the same at every thread count, so the same bits at any thread count still hold. BRISK_QPFTZ=0:
 * off */
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define QP_FTZ_BITS 0x8040u
static _Thread_local unsigned g_qp_csr0 = 0;
static int qp_ftz_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("BRISK_QPFTZ"); v = e ? atoi(e) : 1; } return v; }
static void qp_ftz_enter(void) {
    if (!qp_ftz_on()) return;
    g_qp_csr0 = _mm_getcsr();
    if ((g_qp_csr0 & QP_FTZ_BITS) == QP_FTZ_BITS) return;
    _mm_setcsr(g_qp_csr0 | QP_FTZ_BITS);
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        #pragma omp parallel
        { _mm_setcsr(_mm_getcsr() | QP_FTZ_BITS); }
    }
#endif
}
static void qp_ftz_leave(void) {
    if (!qp_ftz_on() || (g_qp_csr0 & QP_FTZ_BITS) == QP_FTZ_BITS) return;
    const unsigned clr = QP_FTZ_BITS & ~g_qp_csr0;
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        #pragma omp parallel
        { _mm_setcsr(_mm_getcsr() & ~clr); }
    }
#endif
    _mm_setcsr(_mm_getcsr() & ~clr);
}
#else
static void qp_ftz_enter(void) {}
static void qp_ftz_leave(void) {}
#endif
/* (1 November 2026) the subnormal results flushed only for a problem of 1e5 entries or more (A's, H's, the rows and the
 * columns): the mode is set on every OpenMP thread, two parallel regions a call, which woke the thread pool at every solve of
 * a QP of tens of variables solved in tens of microseconds (the owner's QP benchmark's small MPC, regression and portfolio
 * QPs); a rule by size alone, so that the mode is the same at any thread count */
static _Thread_local int g_qp_ftz_set = 0;
void qp_deadline_enter(const QpOpts *o, const QpProb *P) {
    if (g_qp_dl_depth++ == 0) {
        g_qp_deadline = o && o->timelimit > 0 ? qp_clock() + o->timelimit : 0;
        const double sz = P ? (double)P->n + P->m + (P->Ap ? P->Ap[P->n] : 0) + (P->Hp ? P->Hp[P->n] : 0) : 0;
        g_qp_ftz_set = sz >= 1e5;
        if (g_qp_ftz_set) qp_ftz_enter();
    }
}
void qp_deadline_leave(void) { if (g_qp_dl_depth > 0 && --g_qp_dl_depth == 0) { g_qp_deadline = 0; if (g_qp_ftz_set) qp_ftz_leave(); g_qp_ftz_set = 0; } }
int qp_time_up(void) { return g_qp_deadline > 0 && qp_clock() > g_qp_deadline; }
/* (1 November 2026, the SOCP developer; the owner's QP benchmark, audit A1 and A15) the data of a QP or QCQP checked once
 * at each entry (the one-shot call, the persistent model, the lean route; the file reader's numbers by qpio.c): the
 * sizes, the column structures (pointers from 0, nondecreasing, row indices in range; a triangle's entries on or below
 * the diagonal), every coefficient finite (H, A, c, c0, the quadratic rows' entries), and no bound NaN (an infinite bound
 * is a missing side). 0, or QP_INPUT with the reason in msg. A NaN cost had given OPTIMAL with a NaN x */
static int qp_csc_ok(int n, int m, const int *p, const int *ix, const double *v, int lower, const char *what, char *msg, size_t nmsg) {
    if (p[0] != 0) { snprintf(msg, nmsg, "%s: the column pointers do not start at 0", what); return 0; }
    for (int j = 0; j < n; j++) {
        if (p[j + 1] < p[j]) { snprintf(msg, nmsg, "%s: the column pointers decrease at column %d", what, j + 1); return 0; }
        for (int q = p[j]; q < p[j + 1]; q++) {
            if (ix[q] < 0 || ix[q] >= m || (lower && ix[q] < j)) { snprintf(msg, nmsg, "%s: the entry %d of column %d has row index %d %s", what, q, j + 1, ix[q], lower && ix[q] >= 0 && ix[q] < j ? "(above the diagonal: the lower triangle is expected)" : "(out of range)"); return 0; }
            if (!isfinite(v[q])) { snprintf(msg, nmsg, "%s: the entry in row %d, column %d is %s", what, ix[q] + 1, j + 1, isnan(v[q]) ? "NaN" : "infinite"); return 0; }
        }
    }
    return 1;
}
int qp_input_check(const QpProb *P, char *msg, size_t nmsg) {
    const int n = P->n, m = P->m;
    if (n < 0 || m < 0) { snprintf(msg, nmsg, "negative sizes (n = %d, m = %d)", n, m); return QP_INPUT; }
    if (P->Hp && !qp_csc_ok(n, n, P->Hp, P->Hi, P->Hx, 1, "the objective's quadratic part H", msg, nmsg)) return QP_INPUT;
    if (m > 0 && !P->Ap) { snprintf(msg, nmsg, "%d rows but no constraint matrix", m); return QP_INPUT; }
    if (P->Ap && !qp_csc_ok(n, m, P->Ap, P->Ai, P->Ax, 0, "the constraint matrix A", msg, nmsg)) return QP_INPUT;
    if (!isfinite(P->c0)) { snprintf(msg, nmsg, "the objective's constant is %s", isnan(P->c0) ? "NaN" : "infinite"); return QP_INPUT; }
    for (int j = 0; j < n; j++) {
        if (P->c && !isfinite(P->c[j])) { snprintf(msg, nmsg, "the cost of variable %d is %s", j + 1, isnan(P->c[j]) ? "NaN" : "infinite"); return QP_INPUT; }
        if ((P->xl && isnan(P->xl[j])) || (P->xu && isnan(P->xu[j]))) { snprintf(msg, nmsg, "a bound of variable %d is NaN", j + 1); return QP_INPUT; }
    }
    for (int i = 0; i < m; i++)
        if ((P->rl && isnan(P->rl[i])) || (P->ru && isnan(P->ru[i]))) { snprintf(msg, nmsg, "a bound of row %d is NaN", i + 1); return QP_INPUT; }
    for (int k = 0; k < P->nqc; k++) {
        if (P->qrow[k] < 0 || P->qrow[k] >= m) { snprintf(msg, nmsg, "quadratic constraint %d names row %d (out of range)", k + 1, P->qrow[k] + 1); return QP_INPUT; }
        for (int q = P->qs[k]; q < P->qs[k + 1]; q++) {
            if (P->qi[q] < 0 || P->qi[q] >= n || P->qj[q] < 0 || P->qj[q] >= n) { snprintf(msg, nmsg, "quadratic constraint %d: an entry's index is out of range", k + 1); return QP_INPUT; }
            if (!isfinite(P->qv[q])) { snprintf(msg, nmsg, "quadratic constraint %d: the entry (%d, %d) is %s", k + 1, P->qi[q] + 1, P->qj[q] + 1, isnan(P->qv[q]) ? "NaN" : "infinite"); return QP_INPUT; }
        }
    }
    return 0;
}
static int qp_input_refuse(const QpProb *P, QpRes *R, const char *msg) {
    memset(R, 0, sizeof(*R));
    R->x = calloc((size_t)(P->n > 0 ? P->n : 0) + 1, sizeof(double)); R->y = calloc((size_t)(P->m > 0 ? P->m : 0) + 1, sizeof(double)); R->z = calloc((size_t)(P->n > 0 ? P->n : 0) + 1, sizeof(double));
    fprintf(stderr, "brisk: invalid problem data: %s\n", msg);
    R->status = QP_INPUT; return R->status;
}
/* (1 November 2026, the SOCP developer; the owner's QP benchmark, the large mass-spring MPC plants) entries of H and A below
 * the smallest normal double (2.2e-308, subnormal numbers) are zero for every purpose: their product with any variable
 * within the bounds' range (1e20) is below 1e-287, far below the rounding of any row or column with a normal entry; but
 * arithmetic on them is slow (each operation a microcode assist on x86) and the factorizations' fill spreads them. The
 * exact zero-order hold of a stiff spring chain gives them (the 128-mass plant: 276,960 in its dynamics matrix; BRISK's
 * solve 80 s, 31 s with the processor flushing them to zero). Dropped in place from a column structure; returns the count */
long qp_csc_drop_subnormal(int n, int *p, int *ix, double *v) {
    long nd = 0, w = 0;
    for (int j = 0; j < n; j++) {
        const int a = p[j], b = p[j + 1]; p[j] = (int)w;
        for (int q = a; q < b; q++) { if (v[q] != 0 && fabs(v[q]) < 2.2250738585072014e-308) { nd++; continue; } ix[w] = ix[q]; v[w] = v[q]; w++; }
    }
    p[n] = (int)w;
    return nd;
}
static long qp_count_subnormal(const int *p, const double *v, int n) { long c = 0; for (long q = 0; q < p[n]; q++) if (v[q] != 0 && fabs(v[q]) < 2.2250738585072014e-308) c++; return c; }
static int qp_solve_body(const QpProb *P, const QpOpts *o, QpRes *R);
static int qp_solve_body0(const QpProb *P, const QpOpts *o, QpRes *R) {
    const long sa = P->Ap ? qp_count_subnormal(P->Ap, P->Ax, P->n) : 0, sh = P->Hp ? qp_count_subnormal(P->Hp, P->Hx, P->n) : 0;
    if (sa == 0 && sh == 0) return qp_solve_body(P, o, R);
    QpProb C = *P; int *ap = NULL, *ai = NULL, *hp = NULL, *hi = NULL; double *ax = NULL, *hx = NULL;
    if (sa) {
        const long nz = P->Ap[P->n];
        ap = malloc(sizeof(int) * ((size_t)P->n + 1)); ai = malloc(sizeof(int) * ((size_t)nz + 1)); ax = malloc(sizeof(double) * ((size_t)nz + 1));
        if (ap && ai && ax) { memcpy(ap, P->Ap, sizeof(int) * ((size_t)P->n + 1)); memcpy(ai, P->Ai, sizeof(int) * (size_t)nz); memcpy(ax, P->Ax, sizeof(double) * (size_t)nz); qp_csc_drop_subnormal(P->n, ap, ai, ax); C.Ap = ap; C.Ai = ai; C.Ax = ax; }
    }
    if (sh) {
        const long nz = P->Hp[P->n];
        hp = malloc(sizeof(int) * ((size_t)P->n + 1)); hi = malloc(sizeof(int) * ((size_t)nz + 1)); hx = malloc(sizeof(double) * ((size_t)nz + 1));
        if (hp && hi && hx) { memcpy(hp, P->Hp, sizeof(int) * ((size_t)P->n + 1)); memcpy(hi, P->Hi, sizeof(int) * (size_t)nz); memcpy(hx, P->Hx, sizeof(double) * (size_t)nz); qp_csc_drop_subnormal(P->n, hp, hi, hx); C.Hp = hp; C.Hi = hi; C.Hx = hx; }
    }
    if (o->verbose > 1) qp_printf("%ld entries of A and %ld of H below the smallest normal number (2.2e-308) dropped\n", sa, sh);
    const int st = qp_solve_body(&C, o, R);
    free(ap); free(ai); free(ax); free(hp); free(hi); free(hx);
    return st;
}
int qp_solve(const QpProb *P, const QpOpts *o, QpRes *R) {
    { char msg[200]; if (qp_input_check(P, msg, sizeof msg)) return qp_input_refuse(P, R, msg); }
    QpOpts od; if (!o) { qp_default_opts(&od); o = &od; }
    qp_deadline_enter(o, P);
    const int st = qp_solve_body0(P, o, R);
    qp_deadline_leave();
    return st;
}
static int qp_solve_body(const QpProb *P, const QpOpts *o, QpRes *R) {
    /* (28 October 2026, the SOCP developer) the default choice's active-set method first, before the exact test of
     * convexity: when H's Cholesky factor exists (positive definite, eps = 0) it proves the convexity the test checks (a
     * positive definite H has a positive diagonal); the test, which allocates and sorts H's entries, was a part of the solve
     * of a small QP (the owner's independent QP benchmark: QPs of 10 to 64 variables solved in tens of microseconds). For a
     * semidefinite H the active-set method runs the test itself before its proximal-point loop (as_solve_body), and a model
     * that eliminates equality rows runs it when it is made; so a result (st >= 0) has passed it. Every other route takes the
     * test first, as before */
    int proven = 0; double t_as = 0;
    if (P->nqc == 0 && o->method == QP_METHOD_AUTO) {
        double wipm = 0;
        if (as_small(P, &wipm)) {
            const int st = qp_solve_activeset_budget(P, o, R, wipm);
            if (st == QP_OPTIMAL) { R->method = QP_METHOD_ACTIVESET; return st; }
            if (st >= 0) {
                proven = 1;
                t_as = R->time > 0 ? R->time : 1e-9;
                if (o->verbose) qp_printf("the active-set method ended %s: the interior-point method solves the problem\n", qp_status_name(st));
                qp_res_free(R);
            }
        }
    }
    if (!proven && qp_convex_exact(P)) {
        memset(R, 0, sizeof(*R));
        R->x = calloc((size_t)P->n + 1, sizeof(double)); R->y = calloc((size_t)P->m + 1, sizeof(double)); R->z = calloc((size_t)P->n + 1, sizeof(double));
        R->status = QP_INPUT; return R->status;
    }
    if (P->nqc > 0) return qcqp_solve(P, o, R);
    if (o->method == QP_METHOD_ACTIVESET) { const int st = qp_solve_activeset(P, o, R); R->method = QP_METHOD_ACTIVESET; return st; }
    const int st = qp_solve_ipm_dbl(P, o, R);
    R->method = QP_METHOD_IPM; R->t_as = t_as;
    return st;
}

/* (22 October 2026) certificates of infeasibility read off a point, for an iteration whose iterates diverge.
 * Primal infeasibility (Farkas): multipliers (y, z) with A'y + z = 0, the right sign at each finite bound and zero at an
 * infinite one, and the bound value sum_i y_i (rl_i or ru_i) + sum_j z_j (xl_j or xu_j) > 0; returned: that value over
 * the size of (y, z), and the residual |A'y + z| (plus the wrong-sign parts) over the same size (pinf_val, pinf_res).
 * Dual infeasibility (unboundedness): a direction d = x / |x| with H d = 0, c'd < 0 and d in the recession cone of the
 * constraints (A d >= 0 where only rl is finite, <= 0 where only ru is, = 0 where both are; likewise for the bounds);
 * returned: -c'd and the largest violation of those conditions (|H d| included), each relative (dinf_val, dinf_res). */
/* (24 October 2026, the SOCP developer) a proof of primal infeasibility from multipliers (y, z) of any size: with the
 * multipliers of sides that have no finite bound set to 0 (still multipliers), every x with rl <= Ax <= ru and
 * xl <= x <= xu gives r'x >= val for r = A'y + z and val = sum of y_i times the side it bounds (rl_i for y_i > 0, ru_i for
 * y_i < 0) plus the same for z; and r'x <= sum_j max(r_j xl_j, r_j xu_j) over the box. A value above that maximum (by more
 * than the rounding of the sums) proves that no such x exists. Exact where the bounds that r needs are finite (every
 * column of r_j != 0); no threshold on the multipliers' growth (the iterates of an infeasible QP whose variables are all
 * bounded grow slowly: BCTRLI_200, QP, 130 iterations to the growth test, MOSEK 7). 1: proven */
int qp_farkas_proof(const QpProb *P, const double *y, const double *z) {
    const int n = P->n, m = P->m;
    double *r = calloc((size_t)n + 1, sizeof(double));
    if (!r) return 0;
    double val = 0, mag = 0;
    for (int j = 0; j < n; j++) {
        double zj = z[j];
        if (zj > 0 && !fin(P->xl[j])) zj = 0;
        if (zj < 0 && !fin(P->xu[j])) zj = 0;
        if (zj > 0) { val += zj * P->xl[j]; mag += fabs(zj * P->xl[j]); }
        if (zj < 0) { val += zj * P->xu[j]; mag += fabs(zj * P->xu[j]); }
        r[j] = zj;
    }
    for (int i = 0; i < m; i++) {
        double yi = y[i];
        if (yi > 0 && !fin(P->rl[i])) yi = 0;
        if (yi < 0 && !fin(P->ru[i])) yi = 0;
        if (yi == 0) continue;
        const double side = yi > 0 ? P->rl[i] : P->ru[i];
        val += yi * side; mag += fabs(yi * side);
    }
    for (int j = 0; j < n; j++) {
        double a = 0, aa = 0;
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) {
            const int i = P->Ai[p]; double yi = y[i];
            if (yi > 0 && !fin(P->rl[i])) yi = 0;
            if (yi < 0 && !fin(P->ru[i])) yi = 0;
            a += P->Ax[p] * yi; aa += fabs(P->Ax[p] * yi);
        }
        r[j] += a;
        const double bnd = fmax(fin(P->xl[j]) ? fabs(P->xl[j]) : 0, fin(P->xu[j]) ? fabs(P->xu[j]) : 0);
        mag += (aa + fabs(z[j])) * bnd;
    }
    double smax = 0; int ok = 1;
    for (int j = 0; j < n && ok; j++) {
        if (r[j] > 0) { if (!fin(P->xu[j])) ok = 0; else smax += r[j] * P->xu[j]; }
        else if (r[j] < 0) { if (!fin(P->xl[j])) ok = 0; else smax += r[j] * P->xl[j]; }
    }
    free(r);
    /* (the margin: the sums' rounding, at most a few units in the last place of their terms' magnitude per term; taken
     * as 1e-10 of it, far above that and far below a certificate's value) */
    return ok && val - smax > 1e-10 * (mag + fabs(val) + fabs(smax)) && val - smax > 0;
}

void qp_infeas_measures(const QpProb *P, const double *x, const double *y, const double *z,
                        double *pinf_val, double *pinf_res, double *dinf_val, double *dinf_res) {
    const int n = P->n, m = P->m;
    double sy = 0; for (int i = 0; i < m; i++) sy = fmax(sy, fabs(y[i])); for (int j = 0; j < n; j++) sy = fmax(sy, fabs(z[j]));
    double val = 0, res = 0;
    if (sy > 0) {
        for (int i = 0; i < m; i++) {
            if (y[i] > 0) { if (fin(P->rl[i])) val += y[i] * P->rl[i]; else res = fmax(res, y[i]); }
            if (y[i] < 0) { if (fin(P->ru[i])) val += y[i] * P->ru[i]; else res = fmax(res, -y[i]); }
        }
        for (int j = 0; j < n; j++) {
            if (z[j] > 0) { if (fin(P->xl[j])) val += z[j] * P->xl[j]; else res = fmax(res, z[j]); }
            if (z[j] < 0) { if (fin(P->xu[j])) val += z[j] * P->xu[j]; else res = fmax(res, -z[j]); }
        }
        for (int j = 0; j < n; j++) { double a = z[j]; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) a += P->Ax[p] * y[P->Ai[p]]; res = fmax(res, fabs(a)); }
        val /= sy; res /= sy;
    }
    *pinf_val = val; *pinf_res = res;
    double sx = 0; for (int j = 0; j < n; j++) sx = fmax(sx, fabs(x[j]));
    double dv = 0, dr = 0;
    if (sx > 0) {
        double *hd = calloc((size_t)n + 1, sizeof(double)), *ad = calloc((size_t)m + 1, sizeof(double));
        qp_hmul(P, x, hd);
        for (int j = 0; j < n; j++) { dv -= P->c[j] * x[j] / sx; dr = fmax(dr, fabs(hd[j]) / sx); }
        for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) ad[P->Ai[p]] += P->Ax[p] * x[j];
        for (int i = 0; i < m; i++) {
            const double a = ad[i] / sx;
            if (fin(P->rl[i]) && a < 0) dr = fmax(dr, -a);
            if (fin(P->ru[i]) && a > 0) dr = fmax(dr, a);
        }
        for (int j = 0; j < n; j++) {
            const double a = x[j] / sx;
            if (fin(P->xl[j]) && a < 0) dr = fmax(dr, -a);
            if (fin(P->xu[j]) && a > 0) dr = fmax(dr, a);
        }
        free(hd); free(ad);
    }
    *dinf_val = dv; *dinf_res = dr;
}
/* the size of the data (bounds, right-hand sides, cost): for the certificates' test of divergence */
double qp_scale_of(const QpProb *P) {
    double s = 0;
    for (int i = 0; i < P->m; i++) { if (fin(P->rl[i])) s = fmax(s, fabs(P->rl[i])); if (fin(P->ru[i])) s = fmax(s, fabs(P->ru[i])); }
    for (int j = 0; j < P->n; j++) { s = fmax(s, fabs(P->c[j])); if (fin(P->xl[j])) s = fmax(s, fabs(P->xl[j])); if (fin(P->xu[j])) s = fmax(s, fabs(P->xu[j])); }
    return s;
}
/* the cost's size, max |c_j| */
double qp_cost_scale(const QpProb *P) { double s = 0; for (int j = 0; j < P->n; j++) s = fmax(s, fabs(P->c[j])); return s; }
