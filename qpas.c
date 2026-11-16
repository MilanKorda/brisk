/* qpas.c - the QP solver's active-set method (the SOCP developer, 22 October 2026).
 *
 * The dual active-set method of Goldfarb and Idnani (Math. Programming 27, 1983), in the form of its classical
 * implementation (QuadProg): for min 1/2 x'Gx + c'x with G positive definite and constraints n_k'x >= b_k (inequalities)
 * and n_k'x = b_k (equalities), starting from the unconstrained minimum and adding violated constraints one at a time,
 * always dual feasible, dropping a constraint when its multiplier would turn negative. G = L L' (dense Cholesky); the
 * matrix J = L^-T Q and the upper triangle R of the QR factorization of L^-1 N_A (N_A the active normals) are updated
 * by Givens rotations when a constraint enters or leaves.
 *
 * The problem of qp.h is brought to this form: a row rl <= a'x <= ru gives a'x >= rl and -a'x >= -ru (one of them can
 * be active), rl = ru an equality; a bound xl <= x_j <= xu gives e_j'x >= xl and -e_j'x >= -xu. A positive
 * semidefinite H (an LP, a least-squares problem with fewer data than unknowns) is handled by a proximal-point outer
 * loop: G = H + eps I and c - eps x_k, re-solved from the previous active set until x_k stops moving (each solve is a
 * warm start: the previous active set is dual feasible for the shifted cost when the shift is small).
 *
 * Dense: G, J and R are n x n; the method is for small and medium QPs (model predictive control, the nodes of a branch
 * and bound), where it is warm-started; the interior-point method (qpipm.c) is the one for large sparse QPs.
 *
 * Warm start: qp_as_solve takes an initial working set (the constraints believed active, by index and side); it is
 * made dual feasible (constraints with negative multipliers dropped) and the method continues from there. */
#include "qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qp_fminmax.h"
#include <time.h>
#include "blas_one.h"
/* (28 October 2026, the SOCP developer) the dense algebra by BLAS and LAPACK, with the build's prefix (brisk.h's BL); OpenBLAS on
 * one thread during a solve (blas_one.h), so that the bits do not depend on the host's thread count */
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define AS_CAT2(a, b) a##b
#define AS_CAT(a, b) AS_CAT2(a, b)
void AS_CAT(BLAS_PREFIX, dpotrf_)(const char *, const int *, double *, const int *, int *);
void AS_CAT(BLAS_PREFIX, dtrtri_)(const char *, const char *, const int *, double *, const int *, int *);
void AS_CAT(BLAS_PREFIX, dgemv_)(const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
void AS_CAT(BLAS_PREFIX, dtrsv_)(const char *, const char *, const char *, const int *, const double *, const int *, double *, const int *);
#define as_dpotrf AS_CAT(BLAS_PREFIX, dpotrf_)
#define as_dtrtri AS_CAT(BLAS_PREFIX, dtrtri_)
#define as_dgemv AS_CAT(BLAS_PREFIX, dgemv_)
#define as_dtrsv AS_CAT(BLAS_PREFIX, dtrsv_)
void AS_CAT(BLAS_PREFIX, dgetrf_)(const int *, const int *, double *, const int *, int *, int *);
void AS_CAT(BLAS_PREFIX, dtrsm_)(const char *, const char *, const char *, const char *, const int *, const int *, const double *, const double *, const int *, double *, const int *);
void AS_CAT(BLAS_PREFIX, dgemm_)(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
#define as_dgetrf AS_CAT(BLAS_PREFIX, dgetrf_)
#define as_dtrsm AS_CAT(BLAS_PREFIX, dtrsm_)
#define as_dgemm AS_CAT(BLAS_PREFIX, dgemm_)
static const int as_one = 1; static const double as_d1 = 1.0;
/* (28 October 2026) below 64 x 64 entries (LAPACK's block size, under which it runs unblocked code anyway) the products and
 * triangular solves as plain loops: a BLAS call's dispatch cost as much as the arithmetic of a QP of a few dozen variables
 * (the owner's independent QP benchmark: an integrator MPC QP of n = 34, 25 iterations, a quarter of the time in BLAS calls) */
#define AS_SMALL 4096
static void mv(const char *tr, int M, int N, const double *A, int lda, const double *x, double beta, double *y) {
    if ((double)M * N >= AS_SMALL) { as_dgemv(tr, &M, &N, &as_d1, A, &lda, x, &as_one, &beta, y, &as_one); return; }
    if (tr[0] == 'N') {           /* y = beta y + A x, A M x N by columns */
        if (beta == 0.0) for (int i = 0; i < M; i++) y[i] = 0.0;
        for (int j = 0; j < N; j++) { const double a = x[j]; if (a != 0.0) { const double *Aj = A + (size_t)j * lda; for (int i = 0; i < M; i++) y[i] += a * Aj[i]; } }
    } else {                      /* y = beta y + A' x */
        for (int j = 0; j < N; j++) { const double *Aj = A + (size_t)j * lda; double s = 0; for (int i = 0; i < M; i++) s += Aj[i] * x[i]; y[j] = (beta == 0.0 ? 0.0 : beta * y[j]) + s; }
    }
}
static void trsv_u(const char *tr, int q, const double *R, int ld, double *x) {
    if ((double)q * q >= AS_SMALL) { as_dtrsv("U", tr, "N", &q, R, &ld, x, &as_one); return; }
    if (tr[0] == 'N') for (int i = q - 1; i >= 0; i--) { double s = x[i]; for (int k = i + 1; k < q; k++) s -= R[i + (size_t)k * ld] * x[k]; x[i] = s / R[i + (size_t)i * ld]; }
    else for (int i = 0; i < q; i++) { double s = x[i]; const double *Ri = R + (size_t)i * ld; for (int k = 0; k < i; k++) s -= Ri[k] * x[k]; x[i] = s / Ri[i]; }
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *xm(size_t s) { void *p = malloc(s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static void *xz(size_t s) { void *p = calloc(1, s ? s : 1); if (!p) { fprintf(stderr, "brisk_qp: out of memory\n"); exit(2); } return p; }
static int fin(double v) { return fabs(v) < QP_INF; }
void qp_hmul(const QpProb *P, const double *x, double *y);

/* the constraints: kind 0 a row of A, 1 a bound; side +1: n = +a (>= lower), -1: n = -a (>= -upper); eq: an equality */
typedef struct { int kind, idx, side, eq; double b; } Con;

typedef struct {
    int n, m, nc;
    const QpProb *P;
    int *Rp, *Ri; double *Rx;     /* A by rows */
    Con *con;
    double *G;                    /* H + eps I, dense, column-major */
    double *L;                    /* its Cholesky factor (lower) */
    double *J, *R;                /* J (n x n), R (n x n, upper, the first q columns used) */
    int *act;                     /* the active constraints, in the order of R's columns */
    int q;                        /* their number */
    double *u;                    /* the multipliers of the active constraints */
    char *isact;
    double *x, *cc, *d, *z, *r, *s, *wv, *nv, *av;     /* (av: a row of A scattered, for J'n by dgemv) */
    double *ax;                   /* A x (m), for the search of a violated constraint */
    int axv;                      /* 1: ax is A x at the current x (the last search found no violated constraint) */
    double *AT;                   /* A' dense (n x m by columns: row i of A in column i) when A is dense, else NULL */
    double eps;
    long iters;
} AS;

/* d = J' n_k */
static void jt_n(const AS *S, int k, double *d) {
    const int n = S->n; const Con *c = &S->con[k];
    if (c->kind == 1) { const int j = c->idx; for (int i = 0; i < n; i++) d[i] = c->side * S->J[j + (size_t)i * n]; return; }
    const int p0 = S->Rp[c->idx], p1 = S->Rp[c->idx + 1];
    if ((p1 - p0) * 32 >= n) {      /* (a dense row: d = J'a by dgemv, J read by columns; the loop below reads J by rows, n apart) */
        double *a = S->av; memset(a, 0, sizeof(double) * n);
        for (int p = p0; p < p1; p++) a[S->Ri[p]] = c->side * S->Rx[p];
        mv("T", n, n, S->J, n, a, 0.0, d);
        return;
    }
    for (int i = 0; i < n; i++) d[i] = 0;
    for (int p = p0; p < p1; p++) {
        const int j = S->Ri[p]; const double a = c->side * S->Rx[p];
        const double *Jrow = S->J + j;
        for (int i = 0; i < n; i++) d[i] += a * Jrow[(size_t)i * n];
    }
}
/* n_k' v */
/* row i of A times v, in four sums (a fixed order: the same bits wherever it is called; one sum waited on each addition, a
 * dense row of 240 entries 4 cycles an entry) */
static double row_dot(const AS *S, int i, const double *v) {
    const int p0 = S->Rp[i], p1 = S->Rp[i + 1]; const int *Ri = S->Ri; const double *Rx = S->Rx;
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0; int p = p0;
    for (; p + 3 < p1; p += 4) { s0 += Rx[p] * v[Ri[p]]; s1 += Rx[p + 1] * v[Ri[p + 1]]; s2 += Rx[p + 2] * v[Ri[p + 2]]; s3 += Rx[p + 3] * v[Ri[p + 3]]; }
    for (; p < p1; p++) s0 += Rx[p] * v[Ri[p]];
    return (s0 + s1) + (s2 + s3);
}
static double n_dot(const AS *S, int k, const double *v) {
    const Con *c = &S->con[k];
    if (c->kind == 1) return c->side * v[c->idx];
    return c->side * row_dot(S, c->idx, v);
}
/* the slack n_k'x - b_k */
static double slack(const AS *S, int k) { return n_dot(S, k, S->x) - S->con[k].b; }

/* Givens rotation zeroing b in (a, b) */
static void givens(double a, double b, double *cs, double *sn, double *r) {
    if (b == 0) { *cs = 1; *sn = 0; *r = a; return; }
    const double h = hypot(a, b); *cs = a / h; *sn = b / h; *r = h;
}

/* add constraint k (d = J' n_k computed): rotate d[q..n-1] to zero below q (columns of J rotated), R's column q = d[0..q] */
static void add_con(AS *S, int k, double *d) {
    const int n = S->n, q = S->q;
    for (int i = n - 1; i > q; i--) {
        double cs, sn, rr; givens(d[i - 1], d[i], &cs, &sn, &rr);
        if (sn == 0) continue;
        d[i - 1] = rr; d[i] = 0;
        double *ja = S->J + (size_t)(i - 1) * n, *jb = S->J + (size_t)i * n;
        for (int t = 0; t < n; t++) { const double a = ja[t], b = jb[t]; ja[t] = cs * a + sn * b; jb[t] = -sn * a + cs * b; }
    }
    for (int i = 0; i <= q; i++) S->R[i + (size_t)q * n] = d[i];
    S->act[q] = k; S->isact[k] = 1; S->q = q + 1;
}
/* drop the active constraint in position l: R's columns shifted, re-triangularized by rotations of rows (and of J's columns) */
static void drop_con(AS *S, int l) {
    const int n = S->n, q = S->q;
    S->isact[S->act[l]] = 0;
    for (int c = l; c < q - 1; c++) {
        memcpy(S->R + (size_t)c * n, S->R + (size_t)(c + 1) * n, sizeof(double) * n);
        S->act[c] = S->act[c + 1]; S->u[c] = S->u[c + 1];
    }
    S->q = q - 1;
    /* column c (l <= c < q-1) now has a nonzero at row c+1: rotate rows c, c+1 */
    for (int c = l; c < q - 1; c++) {
        double *Rc = S->R + (size_t)c * n;
        double cs, sn, rr; givens(Rc[c], Rc[c + 1], &cs, &sn, &rr);
        Rc[c] = rr; Rc[c + 1] = 0;
        for (int c2 = c + 1; c2 < q - 1; c2++) { double *R2 = S->R + (size_t)c2 * n; const double a = R2[c], b = R2[c + 1]; R2[c] = cs * a + sn * b; R2[c + 1] = -sn * a + cs * b; }
        double *ja = S->J + (size_t)c * n, *jb = S->J + (size_t)(c + 1) * n;
        for (int t = 0; t < n; t++) { const double a = ja[t], b = jb[t]; ja[t] = cs * a + sn * b; jb[t] = -sn * a + cs * b; }
    }
    for (int i = 0; i < n; i++) S->R[i + (size_t)(q - 1) * n] = 0;
}

/* dense Cholesky of G (lower, column-major) */
static int chol(int n, const double *G, double *L) {
    memcpy(L, G, sizeof(double) * (size_t)n * n);
    { static int lap = -1; if (lap < 0) { const char *e = qp_getenv("BRISK_QPASLAPACK"); lap = !(e && atoi(e) == 0); }
      if (lap) {          /* (28 October 2026: LAPACK's dpotrf, G = U'U with U upper (L = U' is not formed: J = L^-T = U^-1, init_J),
                           * the strict lower triangle set to zero; the diagonal is L's) */
          int info = 0; as_dpotrf("U", &n, L, &n, &info);
          if (info) return 1;
          for (int j = 0; j < n; j++) memset(L + (size_t)j * n + j + 1, 0, sizeof(double) * (n - 1 - j));
          return 0;
      } }
    for (int j = 0; j < n; j++) {
        double *Lj = L + (size_t)j * n;
        for (int k = 0; k < j; k++) { const double f = L[j + (size_t)k * n]; if (f != 0) { const double *Lk = L + (size_t)k * n; for (int i = j; i < n; i++) Lj[i] -= f * Lk[i]; } }
        if (!(Lj[j] > 0)) return 1;
        const double d = sqrt(Lj[j]); Lj[j] = d;
        for (int i = j + 1; i < n; i++) Lj[i] /= d;
        for (int i = 0; i < j; i++) Lj[i] = 0;
    }
    return 0;
}
/* J = L^-T (upper triangular): column i of J solves L' J_i = e_i. (23 October 2026) Column i is zero below row i, so the
 * back substitution runs over rows i..0 and its sums over k <= i only: n^3/6 multiply-adds instead of n^3/2, the same
 * bits (the terms left out are products with exact zeros); it was 87 % of an active-set solve of WHLIPBAL0 (QP, n = 50) */
static void init_J(AS *S) {
    const int n = S->n;
    { static int lap = -1; if (lap < 0) { const char *e = qp_getenv("BRISK_QPASLAPACK"); lap = !(e && atoi(e) == 0); }
      if (lap) {          /* (28 October 2026: J = L^-T = U^-1 by LAPACK's dtrtri on chol's U; its strict lower triangle is zero) */
          memcpy(S->J, S->L, sizeof(double) * (size_t)n * n);
          int info = 0; as_dtrtri("U", "N", &n, S->J, &n, &info);
          if (!info) return;
          memcpy(S->J, S->L, sizeof(double) * (size_t)n * n);    /* (singular: J left as U, the caller's factor test has passed) */
          return;
      } }
    memset(S->J, 0, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) {
        double *Ji = S->J + (size_t)i * n;
        Ji[i] = 1.0;
        for (int r = i; r >= 0; r--) {               /* back substitution with L' (upper) */
            double v = Ji[r];
            const double *Lr = S->L + (size_t)r * n;
            for (int k = r + 1; k <= i; k++) v -= Lr[k] * Ji[k];
            Ji[r] = v / Lr[r];
        }
    }
}

/* x = the minimum of 1/2 x'Gx + cc'x subject to the active constraints as equalities, and their multipliers u:
 * x = -J2 J2' cc + J1 R^-T b_A,  u = R^-1 J1' (G x + cc) */
static void eq_solve(AS *S) {
    const int n = S->n, q = S->q;
    double *t = S->wv, *v = S->nv;
    /* (28 October 2026: the products by dgemv, the triangular solves by dtrsv; the same formulas) */
    /* t = J' cc */
    mv("T", n, n, S->J, n, S->cc, 0.0, t);
    /* v (q) = R^-T b_A */
    for (int i = 0; i < q; i++) v[i] = S->con[S->act[i]].b;
    if (q > 0) trsv_u("T", q, S->R, n, v);
    /* x = J w, w = (v, -t[q..n-1]) */
    for (int i = q; i < n; i++) v[i] = -t[i];
    mv("N", n, n, S->J, n, v, 0.0, S->x);
    /* u = R^-1 J1' (G x + cc) */
    double *g = S->d;
    memcpy(g, S->cc, sizeof(double) * n);
    mv("N", n, n, S->G, n, S->x, 1.0, g);
    if (q > 0) {
        mv("T", n, q, S->J, n, g, 0.0, S->u);
        trsv_u("N", q, S->R, n, S->u);
    }
}

/* (28 October 2026, the SOCP developer, after the owner's independent QP benchmark: a weakly convex H, n = 16, was returned
 * OPTIMAL with errors 6.7e-6 at tolerance 1e-8: eq_solve's x and u carry the rounding of the dense factors, of the order of
 * eps cond(H)) one step of iterative refinement of the equality-constrained solution on the active set: the residuals
 * rd = G x + cc - N u and rp = b_A - N'x in double, and the correction (dx, du) from the same factors (eq_solve with cc := rd,
 * b_A := rp gives G dx + rd = N du, N'dx = rp), so that x + dx, u + du solve the system to the factors' backward error */
static void as_refine(AS *S) {
    const int n = S->n, q = S->q;
    S->axv = 0;
    double *x0 = xm(sizeof(double) * (2 * (size_t)n + q + 1)), *cc0 = x0 + n, *u0 = cc0 + n;
    double *b0 = xm(sizeof(double) * ((size_t)q + 1));
    memcpy(x0, S->x, sizeof(double) * n); memcpy(cc0, S->cc, sizeof(double) * n); memcpy(u0, S->u, sizeof(double) * q);
    double *rd = S->cc;                                   /* rd = G x + cc - N u, in place of cc */
    for (int j = 0; j < n; j++) { const double xj = x0[j]; if (xj != 0) { const double *Gj = S->G + (size_t)j * n; for (int k = 0; k < n; k++) rd[k] += Gj[k] * xj; } }
    for (int i = 0; i < q; i++) {
        const Con *c = &S->con[S->act[i]]; const double a = c->side * u0[i];
        if (c->kind == 1) rd[c->idx] -= a; else for (int p = S->Rp[c->idx]; p < S->Rp[c->idx + 1]; p++) rd[S->Ri[p]] -= a * S->Rx[p];
    }
    for (int i = 0; i < q; i++) { const int k = S->act[i]; b0[i] = S->con[k].b; S->con[k].b = b0[i] - n_dot(S, k, x0); }
    eq_solve(S);
    for (int j = 0; j < n; j++) S->x[j] = x0[j] + S->x[j];      /* (eq_solve solves G v + rd = N w, N'v = rp: x + v, u + w) */
    for (int i = 0; i < q; i++) S->u[i] = u0[i] + S->u[i];
    for (int i = 0; i < q; i++) S->con[S->act[i]].b = b0[i];
    memcpy(S->cc, cc0, sizeof(double) * n);
    free(x0); free(b0);
}
/* the method of Goldfarb and Idnani from the current active set (made dual feasible first); returns 0 optimal,
 * 1 infeasible, 3 iteration limit */
/* (3 November 2026, the owner: the default log's iteration tables) the active-set method's row: the objective at x (of the
 * proximal-point subproblem when eps > 0: 1/2 x'Gx + cc'x), the largest relative violation of a constraint and how many are
 * violated (a dual method: x is optimal for the active constraints, the violations fall to 0) */
void brisk_stab_row(long it, const char *what, double obj, double pinf, int ninf);
void brisk_stab_end(long it, const char *what, double obj, double pinf, int ninf);
int brisk_stab_due(long it);
static void as_tab(AS *S, long it, int last, double tol) {
    if (!last && !brisk_stab_due(it)) return;
    const int n = S->n;
    double f = 0;
    for (int i = 0; i < n; i++) { const double *Gi = S->G + (size_t)i * n; double s = 0; for (int k = 0; k < n; k++) s += Gi[k] * S->x[k]; f += S->x[i] * (0.5 * s + S->cc[i]); }
    double w = 0; int nv = 0;
    for (int k = 0; k < S->nc; k++) {
        if (S->isact[k]) continue;
        const double sk = slack(S, k), sc = 1 + fabs(S->con[k].b);
        const double v = S->con[k].eq ? fabs(sk) / sc : -sk / sc;
        if (v > tol) { nv++; if (v > w) w = v; }
    }
    const double obj = (S->P->maxim ? -1 : 1) * (f + S->P->c0);
    if (last) brisk_stab_end(it, "active set", obj, w, nv); else brisk_stab_row(it, "active set", obj, w, nv);
}
static int gi_solve(AS *S, long maxit, double tol, int verbose) {
    const int n = S->n;
    S->axv = 0;
    /* dual feasibility of the starting set: drop inequalities with negative multipliers, one at a time */
    for (;;) {
        eq_solve(S);
        int worst = -1; double wu = -1e-14;
        for (int i = 0; i < S->q; i++) if (!S->con[S->act[i]].eq && S->u[i] < wu) { wu = S->u[i]; worst = i; }
        if (worst < 0) break;
        drop_con(S, worst);
    }
    for (long it = 0; it < maxit; it++) {
        as_tab(S, S->iters, 0, tol);
        S->iters++;
        if ((it & 15) == 15 && qp_time_up()) return 5;     /* (1 November 2026, audit A2: the solve's deadline inside the loop; every 16 iterations) */
        /* the most violated constraint (relative to its normal's size); equalities not yet active come first. The row
         * products A x once (both sides of a row from the same product; was one product a side: 2 nnz(A) an iteration) */
        if (S->AT) mv("T", n, S->m, S->AT, n, S->x, 0.0, S->ax);
        else for (int i = 0; i < S->m; i++) S->ax[i] = row_dot(S, i, S->x);
        int p = -1; double worst = 0;
        for (int k = 0; k < S->nc; k++) {
            if (S->isact[k]) continue;
            const Con *ck = &S->con[k];
            const double sk = (ck->kind == 1 ? ck->side * S->x[ck->idx] : ck->side * S->ax[ck->idx]) - ck->b;
            const Con *c = &S->con[k];
            const double scale = 1 + fabs(c->b);
            if (c->eq) { if (fabs(sk) > tol * scale) { const double v = fabs(sk) / scale * 1e6; if (v > worst) { worst = v; p = k; } } }
            else if (sk < -tol * scale) { const double v = -sk / scale; if (v > worst) { worst = v; p = k; } }
        }
        if (p < 0) { S->axv = 1; return 0; }
        /* an equality with a positive slack enters with the opposite side (n'x = b: the side does not matter) */
        if (S->con[p].eq && slack(S, p) > 0) { S->con[p].side = -S->con[p].side; S->con[p].b = -S->con[p].b; }
        double up = 0;               /* the multiplier of the entering constraint */
        for (;;) {
            jt_n(S, p, S->d);
            const int q = S->q;
            /* z = J2 d2 (the primal step), r = R^-1 d1 (the change of the active multipliers) */
            { const int nq = n - q;
              if (nq > 0) mv("N", n, nq, S->J + (size_t)q * n, n, S->d + q, 0.0, S->z);
              else memset(S->z, 0, sizeof(double) * n); }
            if (q > 0) { memcpy(S->r, S->d, sizeof(double) * q); trsv_u("N", q, S->R, n, S->r); }
            /* the partial step (a multiplier of an active inequality reaching zero) */
            double t1 = INFINITY; int l = -1;
            for (int i = 0; i < q; i++) if (!S->con[S->act[i]].eq && S->r[i] > 0) { const double t = S->u[i] / S->r[i]; if (t < t1) { t1 = t; l = i; } }
            /* the full step (the entering constraint becomes active) */
            const double zn = n_dot(S, p, S->z);
            double t2 = INFINITY;
            if (fabs(zn) > 1e-14 * (1 + fabs(S->d[0]))) t2 = -slack(S, p) / zn;
            if (!isfinite(t1) && !isfinite(t2)) return 1;      /* no step: the constraints are inconsistent */
            if (!isfinite(t2)) {                                 /* a step in the dual only */
                for (int i = 0; i < q; i++) S->u[i] -= t1 * S->r[i];
                up += t1;
                drop_con(S, l);
                continue;
            }
            const double t = fmin(t1, t2);
            for (int k = 0; k < n; k++) S->x[k] += t * S->z[k];
            for (int i = 0; i < q; i++) S->u[i] -= t * S->r[i];
            up += t;
            if (t2 <= t1) {                                      /* the full step: p enters */
                jt_n(S, p, S->d);
                add_con(S, p, S->d);
                S->u[S->q - 1] = up;
                break;
            }
            drop_con(S, l);                                      /* the partial step: l leaves, p is tried again */
        }
        if (verbose > 2) qp_printf("  [active set: %d constraints]\n", S->q);
    }
    return 3;
}

/* the constraints of P */
static void build_cons(AS *S, const QpProb *P) {
    const int n = P->n, m = P->m;
    S->con = xm(sizeof(Con) * (2 * (size_t)(n + m) + 1)); int k = 0;
    for (int i = 0; i < m; i++) {
        const double lo = P->rl[i], hi = P->ru[i];
        if (fin(lo) && fin(hi) && hi - lo <= 1e-12 * (1 + fabs(lo))) { S->con[k++] = (Con){ 0, i, +1, 1, lo }; continue; }
        if (fin(lo)) S->con[k++] = (Con){ 0, i, +1, 0, lo };
        if (fin(hi)) S->con[k++] = (Con){ 0, i, -1, 0, -hi };
    }
    for (int j = 0; j < n; j++) {
        const double lo = P->xl[j], hi = P->xu[j];
        if (fin(lo) && fin(hi) && hi - lo <= 1e-12 * (1 + fabs(lo))) { S->con[k++] = (Con){ 1, j, +1, 1, lo }; continue; }
        if (fin(lo)) S->con[k++] = (Con){ 1, j, +1, 0, lo };
        if (fin(hi)) S->con[k++] = (Con){ 1, j, -1, 0, -hi };
    }
    S->nc = k;
}

struct QpSolver {
    QpProb P;                     /* a copy */
    QpOpts o;
    AS S;
    int have;                     /* a previous solution (active set) exists */
    int built;
    int convex;                   /* 0: not yet tested, 1: passed qp_convex_exact, -1: failed (every solve refused) */
    int dirty;                    /* bounds changed since the constraints were built: brought up to date at the next solve */
    int failed;                   /* the factorization of H failed (qp_prepare) */
    struct QpElim *E;             /* the equality rows eliminated (qp_elim_build), or NULL */
    int elim;                     /* 0: not tried, 1: in use (E), -1: not used */
    void *memo;                   /* (1 November 2026) the interior-point method's analysis kept between solves (qpipm.c) */
    /* (14 November 2026, the QP developer) the model's own choice of method (QP_METHOD_AUTO), made once (qp.c's
     * qp_as_stream): chosen, the choice is made; auto_as, the active-set method; wstep, the work a solve may use before the
     * interior-point method takes it over (0: none, the one-shot rule's own budget); over, the solves in a row that did */
    int chosen, auto_as, over; double wstep;
};
static int qp_auto_as(QpSolver *Q) {
    if (!Q->chosen) { Q->chosen = 1; Q->auto_as = qp_as_stream(&Q->P, &Q->wstep); }
    return Q->auto_as;
}

static void as_alloc(AS *S, const QpProb *P) {
    const int n = P->n, m = P->m;
    memset(S, 0, sizeof(*S));
    S->n = n; S->m = m; S->P = P;
    /* A by rows */
    S->Rp = xz(sizeof(int) * (m + 1)); const int nz = P->Ap[n];
    S->Ri = xm(sizeof(int) * (nz + 1)); S->Rx = xm(sizeof(double) * (nz + 1));
    for (int p = 0; p < nz; p++) S->Rp[P->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) S->Rp[i + 1] += S->Rp[i];
    int *pos = xm(sizeof(int) * (m + 1)); memcpy(pos, S->Rp, sizeof(int) * m);
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int q = pos[P->Ai[p]]++; S->Ri[q] = j; S->Rx[q] = P->Ax[p]; }
    free(pos);
    build_cons(S, P);
    const size_t nn = (size_t)n * n;
    S->G = xz(sizeof(double) * (nn + 1)); S->L = xm(sizeof(double) * (nn + 1)); S->J = xm(sizeof(double) * (nn + 1)); S->R = xz(sizeof(double) * (nn + 1));
    S->act = xm(sizeof(int) * (n + 1)); S->u = xz(sizeof(double) * (n + 1)); S->isact = xz(S->nc + 1);
    S->x = xz(sizeof(double) * (n + 1)); S->cc = xz(sizeof(double) * (n + 1)); S->d = xz(sizeof(double) * (n + 1));
    S->z = xz(sizeof(double) * (n + 1)); S->r = xz(sizeof(double) * (n + 1)); S->s = xz(sizeof(double) * (n + 1));
    S->wv = xz(sizeof(double) * (n + 1)); S->nv = xz(sizeof(double) * (n + 1)); S->av = xz(sizeof(double) * (n + 1));
    S->ax = xz(sizeof(double) * ((size_t)m + 1));
    /* (28 October 2026) a dense A (a quarter of its entries or more, as a condensed MPC QP's) also as a dense A', for A x by
     * one product a search of a violated constraint (the four sums of row_dot: 4 cycles an entry; dgemv: under 1) */
    if ((double)m * n <= 4e6 && 4.0 * nz >= (double)m * n && m > 0) {
        S->AT = xz(sizeof(double) * ((size_t)m * n + 1));
        for (int i = 0; i < m; i++) for (int p = S->Rp[i]; p < S->Rp[i + 1]; p++) S->AT[S->Ri[p] + (size_t)i * n] = S->Rx[p];
    }
}
static void as_free(AS *S) {
    free(S->Rp); free(S->Ri); free(S->Rx); free(S->con); free(S->G); free(S->L); free(S->J); free(S->R);
    free(S->act); free(S->u); free(S->isact); free(S->x); free(S->cc); free(S->d); free(S->z); free(S->r); free(S->s); free(S->wv); free(S->nv); free(S->av); free(S->ax); free(S->AT);
}
/* G = H + eps I and its factorization; eps = 0 when H is positive definite (the Cholesky succeeds with a margin) */
static int as_factor(AS *S) {
    const QpProb *P = S->P; const int n = S->n;
    memset(S->G, 0, sizeof(double) * (size_t)n * n);
    double hmax = 0;
    if (P->Hp) for (int j = 0; j < n; j++) for (int p = P->Hp[j]; p < P->Hp[j + 1]; p++) {
        const int i = P->Hi[p]; const double v = P->Hx[p];
        S->G[i + (size_t)j * n] += v; if (i != j) S->G[j + (size_t)i * n] += v;
        if (i == j) hmax = fmax(hmax, fabs(v));
    }
    S->eps = 0;
    if (!chol(n, S->G, S->L)) {
        /* positive definite: keep eps = 0 unless the factor is badly conditioned */
        double dmin = INFINITY, dmax = 0; for (int j = 0; j < n; j++) { const double d = S->L[j + (size_t)j * n]; dmin = fmin(dmin, d); dmax = fmax(dmax, d); }
        if (dmin * dmin >= 1e-12 * dmax * dmax) return 0;      /* (J = L^-T is set by as_reset, which every caller runs next) */
    }
    S->eps = 1e-6 * fmax(hmax, 1.0);
    for (int j = 0; j < n; j++) S->G[j + (size_t)j * n] += S->eps;
    if (chol(n, S->G, S->L)) return 1;
    return 0;
}
/* restart from an empty active set (J = L^-T, R empty) */
static void as_reset(AS *S) {
    init_J(S); S->q = 0; memset(S->isact, 0, S->nc); memset(S->R, 0, sizeof(double) * (size_t)S->n * S->n);
}
/* reinstate an active set (the constraints' indices): J reset and the constraints added one by one; equalities
 * first; a constraint whose normal is dependent on the active ones (no rotation room) is skipped */
static void as_reinstate(AS *S, const int *list, int cnt) {
    as_reset(S);
    for (int pass = 0; pass < 2; pass++)
        for (int t = 0; t < cnt; t++) {
            const int k = list[t]; if (k < 0 || k >= S->nc || S->isact[k]) continue;
            if ((pass == 0) != (S->con[k].eq != 0)) continue;
            if (S->q >= S->n) break;
            jt_n(S, k, S->d);
            double rest = 0; for (int i = S->q; i < S->n; i++) rest = fmax(rest, fabs(S->d[i]));
            if (rest < 1e-10) continue;           /* dependent on the active normals */
            add_con(S, k, S->d);
        }
}

/* Res's x, y, z from S's iterate and active multipliers, and its errors on the problem as given; returns the largest */
/* (28 October 2026, the SOCP developer) the errors and the objective of Res's point by qp_errors_core, from this method's
 * products: A x and A'y by dgemv on the dense A' when it is kept (else by rows and by columns), H x by dgemv on G when
 * G = H (eps = 0), the sums |a_ij x_j| only at the violated rows (the only ones the measure reads). The sparse products of
 * qp_errors and qp_objective cost 0.41 ms of a 0.9 ms warm re-solve (a condensed MPC QP of 320 variables and 328 rows) */
static void as_errors(const AS *S, QpRes *Res) {
    const QpProb *P = S->P; const int n = S->n, m = S->m;
    const double *x = Res->x, *y = Res->y;
    double *w = xm(sizeof(double) * (2 * (size_t)n + 2 * (size_t)m + 2)), *hx = w, *aty = hx + n, *ax = aty + n, *at = ax + m;
    /* A x: the last search's when it was made at this point (the same products, the same bits); A'y from the rows with a
     * multiplier (the active ones: y is zero elsewhere) */
    if (S->axv) memcpy(ax, S->ax, sizeof(double) * m);
    else if (S->AT) mv("T", n, m, S->AT, n, x, 0.0, ax);
    else for (int i = 0; i < m; i++) ax[i] = row_dot(S, i, x);
    memset(aty, 0, sizeof(double) * n);
    for (int i = 0; i < m; i++) { const double yi = y[i]; if (yi != 0) for (int p = S->Rp[i]; p < S->Rp[i + 1]; p++) aty[S->Ri[p]] += yi * S->Rx[p]; }
    if (S->eps == 0) mv("N", n, n, S->G, n, x, 0.0, hx); else qp_hmul(P, x, hx);
    const int pabs = qp_errors_pabs();
    int *nt = pabs == 0 ? xm(sizeof(int) * ((size_t)m + 1)) : NULL;
    for (int i = 0; i < m; i++) {
        at[i] = 0;
        if (nt) nt[i] = S->Rp[i + 1] - S->Rp[i];
        if ((fin(P->rl[i]) && ax[i] < P->rl[i]) || (fin(P->ru[i]) && ax[i] > P->ru[i]))
            for (int p = S->Rp[i]; p < S->Rp[i + 1]; p++) at[i] += fabs(S->Rx[p] * x[S->Ri[p]]);
    }
    double f;
    qp_errors_core(P, x, y, Res->z, ax, pabs == 1 ? NULL : at, nt, hx, aty, 0.0, &Res->pres, &Res->dres, &Res->gap, &f);
    Res->obj = P->maxim ? -f : f;
    free(w); free(nt);
}
static double as_result(const AS *S, QpRes *Res) {
    const int n = S->n, m = S->m;
    memset(Res->y, 0, sizeof(double) * (m + 1)); memset(Res->z, 0, sizeof(double) * (n + 1));
    memcpy(Res->x, S->x, sizeof(double) * n);
    for (int i = 0; i < S->q; i++) {
        const Con *c = &S->con[S->act[i]]; const double u = S->u[i] * c->side;
        if (c->kind == 0) Res->y[c->idx] += u; else Res->z[c->idx] += u;
    }
    as_errors(S, Res);
    return fmax(Res->pres, fmax(Res->dres, Res->gap));
}
/* the full solve with the proximal-point loop; S's active set is the start */
static int as_run(AS *S, const QpOpts *o, QpRes *Res, double t0) {
    const QpProb *P = S->P; const int n = S->n;
    const double tol = o->tol > 0 ? o->tol : 1e-8;
    const long maxit = o->maxit > 0 ? o->maxit : 100L * (n + S->nc) + 1000;
    double *xk = xz(sizeof(double) * (n + 1));
    int st = 0, outer = 0;
    const long it0 = S->iters;
    for (int j = 0; j < n; j++) xk[j] = S->x[j];
    for (;;) {
        for (int j = 0; j < n; j++) S->cc[j] = P->c[j] - S->eps * xk[j];
        /* (28 October 2026) the iterations counted over the whole solve, the proximal-point loop's solves together (it was
         * each solve's own count: maxit iterations up to 10,000 times) */
        const long left = maxit - (S->iters - it0);
        if (left <= 0) { st = 3; break; }
        st = gi_solve(S, left, tol, o->verbose);
        outer++;
        if (st || S->eps == 0) break;
        double dx = 0, xn = 0; for (int j = 0; j < n; j++) { dx = fmax(dx, fabs(S->x[j] - xk[j])); xn = fmax(xn, fabs(S->x[j])); }
        for (int j = 0; j < n; j++) xk[j] = S->x[j];
        if (dx <= 1e-2 * tol * (1 + xn) || outer >= 10000) break;
        if ((o->timelimit > 0 && now() - t0 > o->timelimit) || qp_time_up()) { st = 5; break; }
    }
    free(xk);
    as_tab(S, S->iters, 1, tol);   /* (the table's last row, whatever ended the solve) */
    /* the result; at an optimal active set, up to three steps of refinement (as_refine) while the errors on the problem as
     * given are above the tolerance and fall; then OPTIMAL only within ten times the tolerance, the interior-point method's
     * rule (it was a thousand times, 28 October 2026). BRISK_QPASREFINE=0: no refinement */
    static int nref = -1; if (nref < 0) { const char *e = qp_getenv("BRISK_QPASREFINE"); nref = e ? atoi(e) : 3; }
    double *xs = NULL, *us = NULL; double ebest = INFINITY;
    for (int pass = 0; ; pass++) {
        const double e = as_result(S, Res);
        if (pass > 0 && !(e < ebest)) {             /* no gain: the previous point back */
            memcpy(S->x, xs, sizeof(double) * n); memcpy(S->u, us, sizeof(double) * S->q); S->axv = 0; as_result(S, Res); break;
        }
        ebest = e;
        if (st != 0 || S->eps != 0 || e <= tol || pass >= nref) break;
        if (!xs) { xs = xm(sizeof(double) * (n + 1)); us = xm(sizeof(double) * ((size_t)S->q + 1)); }
        memcpy(xs, S->x, sizeof(double) * n); memcpy(us, S->u, sizeof(double) * S->q);
        as_refine(S);
        if (o->verbose > 1) qp_printf("  [active-set method: refinement %d from errors %.1e]\n", pass + 1, e);
    }
    free(xs); free(us);
    Res->iter = (int)S->iters;
    Res->status = st == 0 ? QP_OPTIMAL : st == 1 ? QP_PRIMAL_INFEASIBLE : st == 5 ? QP_TIME : QP_MAXIT;
    /* (1 November 2026, the owner's QP benchmark, audit A13) REDUCED only within max(1e-6, 100 tol), as the interior-point
     * method's; above it a failure (the default's choice then solves the problem by the interior-point method) */
    if (Res->status == QP_OPTIMAL) { const double e_ = fmax(Res->pres, fmax(Res->dres, Res->gap)); if (e_ > 10 * tol) Res->status = e_ <= fmax(1e-6, 100 * tol) ? QP_REDUCED : QP_NUMERICAL; }
    if (o->verbose) qp_printf("  active-set method: %ld iterations, %d proximal-point solves (eps %.1e), %d active constraints\n", S->iters, outer, S->eps, S->q);
    Res->time = now() - t0;
    return Res->status;
}

/* need_pd: only when H is positive definite (the method's own test: the Cholesky factor of H with a margin, eps = 0);
 * otherwise -1 and nothing solved (Res empty) */
static int as_solve_body(const QpProb *P, const QpOpts *o, QpRes *Res, int need_pd, double wipm);
static int as_solve(const QpProb *P, const QpOpts *o, QpRes *Res, int need_pd) { qp_deadline_enter(o, P); blas_one_begin(); const int st = as_solve_body(P, o, Res, need_pd, 0); blas_one_end(); qp_deadline_leave(); return st; }
int qp_solve_activeset_budget(const QpProb *P, const QpOpts *o, QpRes *Res, double wipm) { qp_deadline_enter(o, P); blas_one_begin(); const int st = as_solve_body(P, o, Res, 1, wipm); blas_one_end(); qp_deadline_leave(); return st; }
static int qp_elim_wanted(const QpProb *P);
static int qp_resolve_body(QpSolver *Q, QpRes *Res);
static void qp_elim_budget(QpSolver *Q, double wipm);
static int as_solve_body(const QpProb *P, const QpOpts *o, QpRes *Res, int need_pd, double wipm) {
    const double t0 = now();
    QpOpts od; if (!o) { qp_default_opts(&od); o = &od; }
    memset(Res, 0, sizeof(*Res));
    /* (28 October 2026, the SOCP developer) a QP whose equality rows can be eliminated (qp_elim_build: the sparse form of an
     * MPC QP) solved through a model that eliminates them: the dense method on all n variables added the k equalities one at
     * a time (B003_R_mpc_mass_spring of the owner's benchmark, n = 1,224, k = 984: 813 ms against the interior-point
     * method's 14 ms). The budget of the automatic choice (wipm) is counted on the reduced QP */
    if (qp_elim_wanted(P)) {
        QpOpts ob = *o; ob.method = QP_METHOD_ACTIVESET;
        QpSolver *Q = qp_create(P, &ob);
        if (Q->elim == 1) {
            if (wipm > 0) qp_elim_budget(Q, wipm);
            const int st = qp_resolve_body(Q, Res);
            qp_destroy(Q);
            Res->time = now() - t0;
            return st;
        }
        qp_destroy(Q);
    }
    if (need_pd && (double)P->n * P->n > 4e8) return -1;
    Res->x = xz(sizeof(double) * (P->n + 1)); Res->y = xz(sizeof(double) * (P->m + 1)); Res->z = xz(sizeof(double) * (P->n + 1));
    if ((double)P->n * P->n > 4e8) { if (o->verbose) qp_printf("the active-set method is dense: n = %d is too large (the interior-point method is the one for it)\n", P->n); Res->status = QP_INPUT; return Res->status; }
    AS S; as_alloc(&S, P);
    const int fr = as_factor(&S);
    /* (28 October 2026) the automatic choice (need_pd) also takes a semidefinite H (S.eps > 0: the proximal-point loop), after
     * the exact test of convexity (H + eps I can be definite while H is not semidefinite) and with its iterations capped:
     * by the budget beyond the fixed bound, by 2 (n + constraints) + 50 within it. The owner's independent QP benchmark:
     * the weakly convex QPs and the quadrotor MPC QPs (H semidefinite), the active-set method 3-6 times faster than the
     * interior-point method in 5-18 iterations (B002_A_weak_hessian 0.22 against 1.37 ms) */
    int psd = 0;
    if (need_pd && !fr && S.eps != 0) psd = !qp_convex_exact(P);
    if (need_pd && (fr || (S.eps != 0 && !psd))) { as_free(&S); qp_res_free(Res); memset(Res, 0, sizeof(*Res)); return -1; }
    if (o->verbose) qp_printf("BRISK QP active-set method (the dual method of Goldfarb and Idnani): n = %d, %d constraints (rows and bounds as one-sided inequalities)\n", P->n, S.nc);
    if (fr) { Res->status = QP_NUMERICAL; as_free(&S); return Res->status; }
    as_reset(&S);
    QpOpts ob = *o;
    if (psd && wipm <= 0) { const long cap = 2L * (P->n + S.nc) + 50; if (ob.maxit <= 0 || ob.maxit > cap) ob.maxit = (int)cap; }
    if (wipm > 0) {
        /* (the budget, qp_solve_activeset_budget: the interior-point method's estimated work, less this method's setup, in
         * iterations, each charged its own work: the rotations and products of J and R, about 8 n^2, and the search for a
         * violated constraint, about 2 nnz(A); rated at the cost of the interior-point method's blocked factorization, as
         * measured on the benchmark's portfolio, regression and weak-Hessian QPs (n = 128-512: about 1 ns an iteration and
         * n^2 against 0.12 ns an operation of the factorization, a factor 8). When the iterations run out the method stops
         * with ITERATION LIMIT and the interior-point method solves the problem. B010_R_mpc_integrator: the active-set method
         * in 1 iteration (1.8 ms) where a count of the constraints the unconstrained minimum violates predicted hundreds) */
        const double n = P->n, was = 2.0 * n * n * n / 3.0 + 2.0 * n * (double)P->Ap[P->n], wit = 8.0 * 8.0 * n * n + 2.0 * (double)P->Ap[P->n];
        const double cap = fmin(1e9, fmax(10.0, (wipm - was) / wit));
        if (ob.maxit <= 0 || ob.maxit > cap) ob.maxit = (int)cap;
        if (o->verbose > 1) qp_printf("  [active-set method: at most %ld iterations, the interior-point method's estimated work %.3g]\n", (long)ob.maxit, wipm);
    }
    as_run(&S, &ob, Res, t0);
    as_free(&S);
    return Res->status;
}
int qp_solve_activeset(const QpProb *P, const QpOpts *o, QpRes *Res) { return as_solve(P, o, Res, 0); }
int qp_solve_activeset_pd(const QpProb *P, const QpOpts *o, QpRes *Res) { return as_solve(P, o, Res, 1); }

/* ---- the solver object: warm starts */
static QpProb copy_prob(const QpProb *P) {
    QpProb C; memset(&C, 0, sizeof(C));
    const int n = P->n, m = P->m;
    C.n = n; C.m = m; C.c0 = P->c0; C.maxim = P->maxim; memcpy(C.name, P->name, sizeof C.name);
#define DUP(dst, src, cnt, T) do { dst = xm(sizeof(T) * ((size_t)(cnt) + 1)); if (cnt) memcpy(dst, src, sizeof(T) * (size_t)(cnt)); } while (0)
    if (P->Hp) { DUP(C.Hp, P->Hp, n + 1, int); DUP(C.Hi, P->Hi, P->Hp[n], int); DUP(C.Hx, P->Hx, P->Hp[n], double); }
    DUP(C.Ap, P->Ap, n + 1, int); DUP(C.Ai, P->Ai, P->Ap[n], int); DUP(C.Ax, P->Ax, P->Ap[n], double);
    DUP(C.c, P->c, n, double); DUP(C.rl, P->rl, m, double); DUP(C.ru, P->ru, m, double); DUP(C.xl, P->xl, n, double); DUP(C.xu, P->xu, n, double);
#undef DUP
    return C;
}
static void qp_prepare(QpSolver *Q);
/* (2 November 2026, the SOCP developer) set by qp_elim_build around the creation of its reduced model: the reduced Hessian
 * Z'HZ of an H that passed the exact convexity test is positive semidefinite (for any Z), so the reduced model does not
 * test it again (a dense test of an r x r matrix the model's own Cholesky factorization repeats at once: 5 % of the call of
 * the benchmark's network-flow QP of 80 variables, QP) */
static _Thread_local int g_qp_convex_known = 0;
QpSolver *qp_create(const QpProb *P, const QpOpts *o) {
    QpSolver *Q = xz(sizeof(QpSolver));
    if (g_qp_convex_known) Q->convex = 1;
    Q->P = copy_prob(P);
    /* (1 November 2026) subnormal entries of the copy dropped, as qp_solve does (qp.c, qp_csc_drop_subnormal) */
    if (Q->P.Ap) qp_csc_drop_subnormal(Q->P.n, Q->P.Ap, Q->P.Ai, Q->P.Ax);
    if (Q->P.Hp) qp_csc_drop_subnormal(Q->P.n, Q->P.Hp, Q->P.Hi, Q->P.Hx);
    if (o) Q->o = *o; else qp_default_opts(&Q->o);
    qp_prepare(Q);
    return Q;
}
const QpProb *qp_problem(const QpSolver *Q) { return &Q->P; }
/* a changed bound or row bound: the constraints are rebuilt (their indices change when a side appears or disappears),
 * and the active set is carried over by the identity of the constraint (row or column, side).
 * (28 October 2026, the SOCP developer, after the owner's independent QP benchmark: each setter rebuilt the constraints and
 * the working set's factors at once, 61.8 ms to set one row bound of an MPC QP to its own value, 10.96 s an update of an H40
 * mass-spring MPC against a 4.8 ms solve.) Now a setter only marks the solver, and the constraints are brought up to date once
 * at the next solve (update_cons): when the new list has the same structure (the same constraints in the same order: rows and
 * columns, finite sides and equalities as before), only the right-hand sides change, and the active set, its factors J and R
 * (which depend on the constraints' normals alone) and the iterate are kept: the solve's first step (eq_solve) recomputes x
 * and the multipliers from the active set at the new right-hand sides. A change of structure rebuilds as before. */
static void rebuild(QpSolver *Q) {
    if (!Q->built) return;
    AS *S = &Q->S;
    int *keys = xm(sizeof(int) * (S->q + 1)), cnt = S->q;
    for (int i = 0; i < S->q; i++) { const Con *c = &S->con[S->act[i]]; keys[i] = (c->kind * 2 + (c->side < 0)) + 4 * c->idx; }
    free(S->con); free(S->isact);
    build_cons(S, &Q->P);
    S->isact = xz(S->nc + 1);
    int *list = xm(sizeof(int) * (cnt + 1)), nl = 0;
    for (int t = 0; t < cnt; t++) {
        const int kind = (keys[t] % 4) / 2, neg = keys[t] % 2, idx = keys[t] / 4;
        for (int k = 0; k < S->nc; k++) if (S->con[k].kind == kind && S->con[k].idx == idx && ((S->con[k].side < 0) == neg || S->con[k].eq)) { list[nl++] = k; break; }
    }
    as_reinstate(S, list, nl);
    free(list); free(keys);
}
static void update_cons(QpSolver *Q) {
    if (!Q->built || !Q->dirty) return;
    Q->dirty = 0;
    AS *S = &Q->S; AS T; memset(&T, 0, sizeof(T));
    static int fast = -1; if (fast < 0) { const char *e = qp_getenv("BRISK_QPASFASTB"); fast = !(e && atoi(e) == 0); }
    build_cons(&T, &Q->P);
    int same = fast && T.nc == S->nc;
    for (int k = 0; same && k < T.nc; k++) {
        const Con *a = &S->con[k], *b = &T.con[k];
        if (a->kind != b->kind || a->idx != b->idx || a->eq != b->eq || (!a->eq && a->side != b->side)) same = 0;
    }
    if (same) {      /* (an equality entered with its side reversed keeps it: b = side lo) */
        for (int k = 0; k < T.nc; k++) S->con[k].b = S->con[k].eq ? S->con[k].side * T.con[k].b : T.con[k].b;
        free(T.con);
        return;
    }
    free(T.con);
    rebuild(Q);
}
int qp_set_var_bounds(QpSolver *Q, int j, double lo, double hi) {
    if (j < 0 || j >= Q->P.n || isnan(lo) || isnan(hi)) return 1;      /* (1 November 2026, audit A1: no NaN bound) */
    Q->P.xl[j] = lo; Q->P.xu[j] = hi; Q->dirty = 1; return 0;
}
int qp_set_row_bounds(QpSolver *Q, int i, double lo, double hi) {
    if (i < 0 || i >= Q->P.m || isnan(lo) || isnan(hi)) return 1;
    Q->P.rl[i] = lo; Q->P.ru[i] = hi; Q->dirty = 1; return 0;
}
int qp_set_cost(QpSolver *Q, const double *c) { for (int j = 0; j < Q->P.n; j++) if (!isfinite(c[j])) return 1; memcpy(Q->P.c, c, sizeof(double) * Q->P.n); return 0; }   /* (1 November 2026, audit A1: a cost not finite refused, the model unchanged) */
static int qp_resolve_body(QpSolver *Q, QpRes *Res);
int qp_resolve(QpSolver *Q, QpRes *Res) { qp_deadline_enter(&Q->o, qp_problem(Q)); blas_one_begin(); const int st = qp_resolve_body(Q, Res); blas_one_end(); qp_deadline_leave(); return st; }
static int qp_elim_solve(QpSolver *Q, QpRes *Res, double t0);
static int qp_elim_same(const QpSolver *Q);
static void qp_elim_free(QpSolver *Q);
void *qp_ipm_memo_new(void); void qp_ipm_memo_use(void *p);
/* (1 November 2026) the model's interior-point solve with its analysis memo (qpipm.c): the analysis of the normal equations
 * kept from the model's previous solve when the pattern is the same */
static int qp_resolve_ipm(QpSolver *Q, const QpProb *P, QpRes *Res) {
    if (!Q->memo) Q->memo = qp_ipm_memo_new();
    qp_ipm_memo_use(Q->memo);
    const int st = qp_solve_ipm_dbl(P, &Q->o, Res);
    qp_ipm_memo_use(NULL);
    return st;
}
static void qp_elim_step_budget(QpSolver *Q, double wstep);
static int qp_resolve_body(QpSolver *Q, QpRes *Res) {
    const double t0 = now();
    const QpProb *P = &Q->P;
    if (!Res->x) { Res->x = xz(sizeof(double) * (P->n + 1)); Res->y = xz(sizeof(double) * (P->m + 1)); Res->z = xz(sizeof(double) * (P->n + 1)); }
    if (!Q->convex) Q->convex = qp_convex_exact(P) ? -1 : 1;        /* (H does not change between solves) */
    if (Q->convex < 0) { Res->status = QP_INPUT; Res->iter = 0; Res->time = now() - t0; return Res->status; }
    /* (26 October 2026, the integrator) QP_METHOD_AUTO (-qpauto): the choice of the one-shot solve (qp_solve), so that a
     * caller that does not choose (JuMP) is never held to the active-set method: within the default's size bound
     * (qp_as_small) the active-set method from the previous active set, else, for an H that is not positive definite (its
     * Cholesky factor fails; H does not change between solves, so for good) and after an active-set solve that does not end
     * optimal, the interior-point method from the start. (31 October 2026, the merge: the build through qp_prepare, which
     * also makes the equality rows' elimination) */
    const int autom = Q->o.method == QP_METHOD_AUTO;
    if (Q->o.method == QP_METHOD_IPM || (autom && !qp_auto_as(Q))) { qp_res_free(Res); return qp_resolve_ipm(Q, P, Res); }
    if (autom) { qp_prepare(Q); if (Q->failed || !Q->auto_as) { if (Q->failed) Q->o.method = QP_METHOD_IPM; qp_res_free(Res); return qp_resolve_ipm(Q, P, Res); } }
    if (Q->elim == 1 && !qp_elim_same(Q)) { qp_elim_free(Q); Q->elim = -1; }      /* (the equality rows changed: the full method) */
    /* (14 November 2026) a model that chose the active-set method because the equality rows' elimination made it cheap lives
     * on the elimination. Without it (the equality rows changed, or its point missed the tolerance: the states' solution
     * operator loses accuracy over a long horizon of unstable dynamics) the dense method on every variable is not the
     * method that was chosen, unless the problem is within the fixed bound for it: the interior-point method keeps the
     * model. (Until now the dense method was tried on all variables at every later solve before the interior-point method
     * solved the problem.) */
    if (autom && Q->elim != 1 && qp_elim_setup_work(P) >= 0 && !qp_as_dense_small(P)) { Q->auto_as = 0; qp_res_free(Res); return qp_resolve_ipm(Q, P, Res); }
    if (Q->elim == 1) {
        if (autom && Q->wstep > 0) qp_elim_step_budget(Q, Q->wstep);
        const int st = qp_elim_solve(Q, Res, t0);
        if (autom && Q->wstep > 0) {
            if (st == QP_OPTIMAL) { Q->over = 0; return st; }
            /* the budget ran out, or the solve ended otherwise: the interior-point method solves this problem; after
             * three such solves in a row the stream is not one the active-set method serves, and it keeps the model */
            if (++Q->over >= 3) Q->auto_as = 0;
            if (qp_getenv("BRISK_QPSTREAMDBG")) fprintf(stderr, "[stream: the active-set solve ended %d (%s) after %d iterations, budget %d; over %d]\n", st, st >= 0 ? qp_status_name(st) : "missed the tolerance", Res->iter, Q->E ? 0 : -1, Q->over);
            if (Q->o.verbose) qp_printf("  [active-set method: this solve handed to the interior-point method%s]\n", Q->auto_as ? "" : ", and the model from here on");
            qp_res_free(Res); return qp_resolve_ipm(Q, P, Res);
        }
        if (st >= 0) return st;
        qp_elim_free(Q); Q->elim = -1;                    /* (its point missed the tolerance on the problem as given) */
        if (Q->o.verbose) qp_printf("  [active-set method: the equality rows' elimination missed the tolerance; the full problem solved]\n");
        if (autom && !qp_as_dense_small(P)) { Q->auto_as = 0; qp_res_free(Res); return qp_resolve_ipm(Q, P, Res); }   /* (as above) */
    }
    update_cons(Q);
    qp_prepare(Q);
    if (!Q->built) { Res->status = QP_NUMERICAL; return Res->status; }
    Q->S.P = P; Q->S.iters = 0;
    const int st = as_run(&Q->S, &Q->o, Res, t0);
    if (autom && st != QP_OPTIMAL && st >= 0) { qp_res_free(Res); return qp_resolve_ipm(Q, P, Res); }
    return st;
}
/* (28 October 2026, the SOCP developer) the equality rows eliminated for the active-set method of a QP model: the sparse
 * formulation of a model-predictive-control QP (the states and inputs of every stage as variables, the dynamics as
 * equality rows, bounds on the variables) has k equality rows of full rank among n variables, and the dense method on all n
 * (J and R n x n) added the k equalities one at a time at O(n^2) each: 1.3 s for the first solve of the mass-spring MPC
 * QP of 1224 variables and 984 equalities of the owner's independent QP benchmark, against DAQP's 19 ms. Here
 * E = the equality rows (k x n); the LU factorization with partial pivoting of E' (n x k: P' E' = [L1; L2] U, LAPACK dgetrf)
 * picks k basic variables (the first k of the permuted order) whose columns B of E are independent; with N the others,
 * B = U' L1', N = U' L2' and B^-1 N = L1^-T L2' =: M (k x r, r = n - k; U cancels). Every solution of E x = b is
 * x_B = x_pB - M w, x_N = w, with x_pB = B^-1 b = L1^-T U^-T b: the QP in w (r variables) has the Hessian Z'HZ
 * (Z = [-M; I] in the permuted order), the cost Z'(H x_p + c), the other rows a_i Z and the bounds of the basic variables
 * as rows -M w within [lb - x_pB, ub - x_pB]. This is the condensed form of the MPC QP (the inputs as the variables when
 * the basis is the states); the reduced QP is a QP model of its own, warm-started from solve to solve, and only x_p and the
 * reduced bounds and cost change when the right-hand sides, bounds or cost change. The point is mapped back (x; the
 * multipliers of the other rows and of the bounds from the reduced ones; those of the equalities from B'y_E = g_B, g the
 * gradient H x + c less the other rows' and the bounds' terms) and its errors measured on the problem as given: when they
 * miss ten times the tolerance, the model drops the elimination and solves the full problem from then on.
 * Used when k >= 8 and k >= n / 8, H is not empty, the problem has no quadratic constraints and E has full rank (its LU's
 * pivots above 1e-9 of the largest); the setup is k^2 n for the LU and 2 n r^2 for Z'HZ (dgemm), at most the dense
 * method's own n^3. BRISK_QPASELIM=0: off */
typedef struct QpElim {
    int n, m, k, r, mi, nbb, mr;
    int *perm, *iperm;            /* the permuted order (the k basic variables first) and its inverse */
    int *eqrow;                   /* the equality rows, in the order of E's rows */
    int *irow;                    /* the other rows (mi), the reduced rows 0..mi-1 */
    int *bb;                      /* the basic variables (their positions 0..k-1) with a finite bound: reduced rows mi.. */
    char *iseq;                   /* the rows' pattern: 1 equality (for the test that the pattern is unchanged) */
    double *LU;                   /* the factored E' (n x k; freed once M is made) */
    int *Lp, *Li, *Up, *Ui; double *Lx, *Ux;   /* L1 (strictly lower, unit diagonal) and U (with its diagonal) by columns:
                                    their nonzeros only (0.2-5 % of the entries on the benchmark's sparse MPC QPs) */
    double *M;                    /* B^-1 N (k x r, by columns) */
    double *xb, *g, *ax, *cr, *t; /* work: x_pB (k), gradients (n), A x_p (m), the reduced cost (r), (n) */
    QpSolver *in;                 /* the reduced QP's model */
} QpElim;
static int fin_e(double v) { return fabs(v) < QP_INF; }
static int is_eqrow(const QpProb *P, int i) { return fin_e(P->rl[i]) && fin_e(P->ru[i]) && P->rl[i] == P->ru[i]; }
static void qp_elim_free(QpSolver *Q) {
    QpElim *E = Q->E; if (!E) return;
    free(E->perm); free(E->iperm); free(E->eqrow); free(E->irow); free(E->bb); free(E->iseq); free(E->LU); free(E->M);
    free(E->Lp); free(E->Li); free(E->Lx); free(E->Up); free(E->Ui); free(E->Ux);
    free(E->xb); free(E->g); free(E->ax); free(E->cr); free(E->t);
    if (E->in) qp_destroy(E->in);
    free(E); Q->E = NULL;
}
/* the solves with B = U' L1' and B' = L1 U by the factors' columns: x := B^-1 x and x := B^-T x */
static void elim_binv(const QpElim *E, double *x) {
    const int k = E->k;
    for (int i = 0; i < k; i++) {             /* U' s = b: s_i = (b_i - sum_{j<i} U_ji s_j) / U_ii */
        double v = x[i]; int p = E->Up[i];
        for (; p < E->Up[i + 1] - 1; p++) v -= E->Ux[p] * x[E->Ui[p]];
        x[i] = v / E->Ux[p];                  /* (the diagonal last in its column) */
    }
    for (int i = k - 1; i >= 0; i--) {        /* L1' x = s: x_i = s_i - sum_{j>i} L_ji x_j */
        double v = x[i];
        for (int p = E->Lp[i]; p < E->Lp[i + 1]; p++) v -= E->Lx[p] * x[E->Li[p]];
        x[i] = v;
    }
}
static void elim_btinv(const QpElim *E, double *x) {
    const int k = E->k;
    for (int i = 0; i < k; i++) { const double v = x[i]; if (v != 0) for (int p = E->Lp[i]; p < E->Lp[i + 1]; p++) x[E->Li[p]] -= E->Lx[p] * v; }   /* L1 v = g */
    for (int i = k - 1; i >= 0; i--) {        /* U y = v */
        const int pd = E->Up[i + 1] - 1; const double v = x[i] / E->Ux[pd]; x[i] = v;
        if (v != 0) for (int p = E->Up[i]; p < pd; p++) x[E->Ui[p]] -= E->Ux[p] * v;
    }
}
static int qp_elim_same(const QpSolver *Q) {
    const QpElim *E = Q->E; const QpProb *P = &Q->P;
    for (int i = 0; i < P->m; i++) if (is_eqrow(P, i) != E->iseq[i]) return 0;
    return 1;
}
/* the reduced cost, row bounds and variable bounds from P's current data (x_p from the equalities' right-hand sides) */
static void qp_elim_data(QpSolver *Q, double *cr, double *rl, double *ru, double *xl, double *xu) {
    QpElim *E = Q->E; const QpProb *P = &Q->P; const int n = E->n, k = E->k, r = E->r;
    for (int t = 0; t < k; t++) E->xb[t] = P->rl[E->eqrow[t]];
    elim_binv(E, E->xb);                                           /* x_pB = B^-1 b */
    double *xp = E->t; memset(xp, 0, sizeof(double) * n);
    for (int i = 0; i < k; i++) xp[E->perm[i]] = E->xb[i];
    qp_hmul(P, xp, E->g);
    for (int j = 0; j < n; j++) E->g[j] += P->c[j];
    double *gb = E->ax; for (int i = 0; i < k; i++) gb[i] = E->g[E->perm[i]];      /* (ax as work: k <= m) */
    for (int j = 0; j < r; j++) cr[j] = E->g[E->perm[k + j]];
    const double m1 = -1.0;
    if (k > 0 && r > 0) as_dgemv("T", &k, &r, &m1, E->M, &k, gb, &as_one, &as_d1, cr, &as_one);     /* Z'g = g_N - M'g_B */
    memset(E->ax, 0, sizeof(double) * ((size_t)P->m + 1));
    for (int i = 0; i < k; i++) { const int j = E->perm[i]; const double v = xp[j]; if (v != 0) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) E->ax[P->Ai[p]] += P->Ax[p] * v; }
    for (int t = 0; t < E->mi; t++) {
        const int i = E->irow[t];
        rl[t] = fin_e(P->rl[i]) ? P->rl[i] - E->ax[i] : -QP_INF; ru[t] = fin_e(P->ru[i]) ? P->ru[i] - E->ax[i] : QP_INF;
    }
    for (int t = 0; t < E->nbb; t++) {
        const int i = E->bb[t], j = E->perm[i];
        rl[E->mi + t] = fin_e(P->xl[j]) ? P->xl[j] - E->xb[i] : -QP_INF; ru[E->mi + t] = fin_e(P->xu[j]) ? P->xu[j] - E->xb[i] : QP_INF;
    }
    for (int j = 0; j < r; j++) { xl[j] = P->xl[E->perm[k + j]]; xu[j] = P->xu[E->perm[k + j]]; }
}
static int qp_elim_wanted(const QpProb *P) {
    { static int off = -1; if (off < 0) { const char *e = qp_getenv("BRISK_QPASELIM"); off = e && atoi(e) == 0; } if (off) return 0; }
    if (P->nqc > 0 || !P->Hp || P->m == 0) return 0;
    int k = 0; for (int i = 0; i < P->m; i++) k += is_eqrow(P, i);
    return k >= 8 && 8 * k >= P->n && k < P->n;
}
/* (1 November 2026, the SOCP developer) whether qp_elim_build's peeling covers the equality rows (a triangular basis: no LU),
 * by the same rule on the sparse rows (a variable in only one remaining equality row, its coefficient at least a tenth of
 * that row's largest), O(nnz): for the automatic choice's cost model */
static int qp_eq_peel(const QpProb *P, int *piv, int *ordr);
static int qp_eq_peelable(const QpProb *P) { return qp_eq_peel(P, NULL, NULL); }
/* the peeling itself: piv[i] the basic variable of equality row i, ordr[t] the row peeled t-th (both may be NULL) */
static int qp_eq_peel(const QpProb *P, int *piv, int *ordr) {
    { static int pon = -1; if (pon < 0) { const char *e_ = qp_getenv("BRISK_QPELIMPEEL"); pon = !(e_ && atoi(e_) == 0); } if (!pon) return 0; }
    const int n = P->n, m = P->m;
    int *rp = calloc((size_t)m + 2, sizeof(int)), *cnt = calloc((size_t)n + 1, sizeof(int)), *stk = malloc(sizeof(int) * ((size_t)n + 1));
    double *rmax = calloc((size_t)m + 1, sizeof(double)); char *dead = calloc((size_t)m + 1, 1), *used = calloc((size_t)n + 1, 1);
    int *rj = NULL; double *rv = NULL; int ok = 0;
    if (!rp || !cnt || !stk || !rmax || !dead || !used) goto out;
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (is_eqrow(P, P->Ai[p]) && P->Ax[p] != 0) { rp[P->Ai[p] + 1]++; cnt[j]++; if (fabs(P->Ax[p]) > rmax[P->Ai[p]]) rmax[P->Ai[p]] = fabs(P->Ax[p]); }
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    rj = malloc(sizeof(int) * ((size_t)rp[m] + 1)); rv = malloc(sizeof(double) * ((size_t)rp[m] + 1));
    if (!rj || !rv) goto out;
    { int *f = malloc(sizeof(int) * ((size_t)m + 1)); if (!f) goto out; memcpy(f, rp, sizeof(int) * (size_t)m);
      for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (is_eqrow(P, P->Ai[p]) && P->Ax[p] != 0) { const int q = f[P->Ai[p]]++; rj[q] = j; rv[q] = P->Ax[p]; }
      free(f); }
    int k = 0; for (int i = 0; i < m; i++) k += is_eqrow(P, i);
    int ns = 0, done = 0; for (int j = 0; j < n; j++) if (cnt[j] == 1) stk[ns++] = j;
    while (ns > 0) {
        const int j = stk[--ns]; if (used[j] || cnt[j] != 1) continue;
        int c = -1; double a = 0;
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int i = P->Ai[p]; if (is_eqrow(P, i) && !dead[i] && P->Ax[p] != 0) { c = i; a = P->Ax[p]; break; } }
        if (c < 0 || !(fabs(a) >= 0.1 * rmax[c])) continue;
        used[j] = 1; dead[c] = 1; if (piv) piv[c] = j; if (ordr) ordr[done] = c; done++;
        for (int q = rp[c]; q < rp[c + 1]; q++) { const int jj = rj[q]; if (!used[jj] && --cnt[jj] == 1) stk[ns++] = jj; }
    }
    ok = done == k;
  out:
    free(rp); free(cnt); free(stk); free(rmax); free(dead); free(used); free(rj); free(rv);
    return ok;
}
/* the setup of the active-set method with the equality rows eliminated, in the units of qp.c's cost model (the LU k^2 n,
 * Z'HZ 2 n r^2, the reduced QP's dense setup 2 r^3 / 3 + 2 r nnz, its rows dense: the other rows and the basic variables'
 * bounds); -1 when the elimination does not apply */
double qp_elim_setup_work(const QpProb *P) {
    if (!qp_elim_wanted(P)) return -1;
    double k = 0, nbb = 0;
    for (int i = 0; i < P->m; i++) k += is_eqrow(P, i);
    const double n = P->n, r = n - k, mr = (P->m - k) + fmin(k, n);
    (void)nbb;
    /* (1 November 2026) the LU's k^2 n only when the peeling does not give a triangular basis; with it, the dense array's
     * scans (about 4 n k) */
    const double wlu = qp_eq_peelable(P) ? 4 * n * k : k * k * n;
    return wlu + 2 * n * r * r + 2 * r * r * r / 3 + 2 * r * (mr * r);
}
/* the automatic choice's budget on the reduced QP: its setup (the LU k^2 n, Z'HZ 2 n r^2, the dense method's own on r
 * variables) and the iterations left, as qp_solve_activeset_budget counts them for the full QP */
static void qp_elim_budget(QpSolver *Q, double wipm) {
    const QpElim *E = Q->E; QpSolver *I = E->in;
    const double n = E->n, k = E->k, r = E->r, nza = (double)I->P.Ap[I->P.n];
    const double was = k * k * n + 2 * n * r * r + 2 * r * r * r / 3 + 2 * r * nza, wit = 64 * r * r + 2 * nza;
    const double cap = fmin(1e9, fmax(10.0, (wipm - was) / wit));
    if (I->o.maxit <= 0 || I->o.maxit > cap) I->o.maxit = (int)cap;
}
/* (14 November 2026) a solve's budget in a stream (qp_as_stream): the changes of the active set that the work of one
 * interior-point solve pays beyond the solve's own 2 r^2 + 2 nnz of the reduced rows, each 10 r^2, at least 10 */
static void qp_elim_step_budget(QpSolver *Q, double wstep) {
    QpSolver *I = Q->E->in; const double r = Q->E->r, wbase = 2.0 * r * r + 2.0 * (double)I->P.Ap[I->P.n];
    I->o.maxit = (int)fmin(1e9, fmax(10.0, (wstep - wbase) / (10.0 * r * r)));
}
static int qp_elim_build(QpSolver *Q) {
    const QpProb *P = &Q->P; const int n = P->n, m = P->m;
    if (!qp_elim_wanted(P)) return 0;
    int k = 0; for (int i = 0; i < m; i++) k += is_eqrow(P, i);
    QpElim *E = xz(sizeof(QpElim));
    E->n = n; E->m = m; E->k = k; E->r = n - k;
    const int r = E->r;
    E->eqrow = xm(sizeof(int) * (k + 1)); E->irow = xm(sizeof(int) * ((size_t)m - k + 1)); E->iseq = xz(m + 1);
    int *col = xm(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { if (is_eqrow(P, i)) { col[i] = E->k - k; E->eqrow[E->k - k] = i; k--; E->iseq[i] = 1; } else { col[i] = -1; E->irow[E->mi++] = i; } }
    k = E->k;
    E->LU = xz(sizeof(double) * ((size_t)n * k + 1));
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (col[P->Ai[p]] >= 0) E->LU[j + (size_t)col[P->Ai[p]] * n] += P->Ax[p];
    free(col);
    int *ipiv = xm(sizeof(int) * (k + 1)), info = 0;
    /* (1 November 2026, the SOCP developer; the owner's QP benchmark, the small sparse MPC QPs) a triangular basis by peeling: a
     * variable in only one of the remaining equality rows, with a coefficient at least a tenth of that row's largest, is that
     * row's basic variable; the row is removed and the peeling goes on. When it covers every row, the basic block, ordered by
     * the peeling, is lower triangular with those coefficients D on its diagonal: E = [L1; L2] U with U = D,
     * L1 = E_B D^-1, L2 = E_N D^-1, so no factorization is needed (a receding-horizon plant's dynamics: the last state is in
     * the last row only, then the one before, ...). The dense LU took most of the solve of a quadrotor MPC QP of 92 variables
     * and 72 equality rows (0.5 million flops through LAPACK on blocks of a few columns). Otherwise the dense LU as before.
     * BRISK_QPELIMPEEL=0: always the dense LU */
    int peeled_ = 0;
    {   int *piv = xm(sizeof(int) * ((size_t)m + 1)), *ordr = xm(sizeof(int) * ((size_t)k + 1));
        if (qp_eq_peel(P, piv, ordr)) {
            /* the rows in the order of the peeling (a row's basic variable is in it and in rows peeled before it only), the
             * basic variables in the same order, then the others; E's column t is row ordr[t], its entries divided by the
             * pivot below the diagonal (L), the pivot on it (U = D); an entry above the diagonal: not triangular */
            int *vperm = xm(sizeof(int) * (n + 1)), *vpos = xm(sizeof(int) * (n + 1)), *rpos = xm(sizeof(int) * ((size_t)m + 1)), *eq2 = xm(sizeof(int) * (k + 1));
            for (int j = 0; j < n; j++) vpos[j] = -1;
            for (int i = 0; i < m; i++) rpos[i] = -1;
            for (int t = 0; t < k; t++) { vperm[t] = piv[ordr[t]]; vpos[piv[ordr[t]]] = t; rpos[ordr[t]] = t; eq2[t] = ordr[t]; }
            { int q = k; for (int j = 0; j < n; j++) if (vpos[j] < 0) { vperm[q] = j; vpos[j] = q++; } }
            double *dg = xz(sizeof(double) * (k + 1));
            for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int t = rpos[P->Ai[p]]; if (t >= 0 && vpos[j] == t) dg[t] += P->Ax[p]; }
            memset(E->LU, 0, sizeof(double) * ((size_t)n * k));
            int ok_ = 1;
            for (int j = 0; j < n && ok_; j++) { const int i = vpos[j];
                for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { const int t = rpos[P->Ai[p]]; if (t < 0 || P->Ax[p] == 0) continue;
                    if (i < t) { ok_ = 0; break; }
                    if (i > t) E->LU[i + (size_t)t * n] += P->Ax[p] / dg[t]; } }
            for (int t = 0; t < k && ok_; t++) { if (!(dg[t] != 0)) ok_ = 0; else E->LU[t + (size_t)t * n] = dg[t]; }
            if (ok_) {
                memcpy(E->eqrow, eq2, sizeof(int) * (size_t)k);
                E->perm = xm(sizeof(int) * (n + 1)); E->iperm = xm(sizeof(int) * (n + 1));
                for (int j = 0; j < n; j++) { E->perm[j] = vperm[j]; E->iperm[vperm[j]] = j; }
                peeled_ = 1;
            } else {          /* (back to E's entries by rows as built above, for the LU) */
                memset(E->LU, 0, sizeof(double) * ((size_t)n * k));
                int *colr = xm(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) colr[i] = -1; for (int t = 0; t < k; t++) colr[E->eqrow[t]] = t;
                for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (colr[P->Ai[p]] >= 0) E->LU[j + (size_t)colr[P->Ai[p]] * n] += P->Ax[p];
                free(colr);
            }
            free(vperm); free(vpos); free(rpos); free(eq2); free(dg);
        }
        free(piv); free(ordr);
    }
    if (peeled_) { /* (factored: L1, L2 and U = D in place, perm and iperm set) */ }
    else if ((double)n * k >= AS_SMALL) as_dgetrf(&n, &k, E->LU, &n, ipiv, &info);
    else {      /* (below 4096 entries the LU as a plain loop, LAPACK's partial pivoting: the first entry of largest magnitude;
                 * dgetrf's blocking cost more than its arithmetic on a QP of 64 variables and 44 equalities) */
        double *a = E->LU;
        for (int j = 0; j < k; j++) {
            int pr = j; double best = fabs(a[j + (size_t)j * n]);
            for (int i = j + 1; i < n; i++) { const double v = fabs(a[i + (size_t)j * n]); if (v > best) { best = v; pr = i; } }
            ipiv[j] = pr + 1;
            if (best == 0) { if (!info) info = j + 1; continue; }
            if (pr != j) for (int c = 0; c < k; c++) { const double t = a[j + (size_t)c * n]; a[j + (size_t)c * n] = a[pr + (size_t)c * n]; a[pr + (size_t)c * n] = t; }
            const double d = a[j + (size_t)j * n];
            for (int i = j + 1; i < n; i++) a[i + (size_t)j * n] /= d;
            for (int c = j + 1; c < k; c++) { const double f = a[j + (size_t)c * n]; if (f != 0) for (int i = j + 1; i < n; i++) a[i + (size_t)c * n] -= a[i + (size_t)j * n] * f; }
        }
    }
    double umax = 0, umin = INFINITY;
    for (int i = 0; i < k; i++) { const double d = fabs(E->LU[i + (size_t)i * n]); umax = fmax(umax, d); umin = fmin(umin, d); }
    if (info != 0 || !(umin > 1e-9 * umax)) { free(ipiv); Q->E = E; qp_elim_free(Q); return 0; }
    if (!peeled_) {
    E->perm = xm(sizeof(int) * (n + 1)); E->iperm = xm(sizeof(int) * (n + 1));
    for (int j = 0; j < n; j++) E->perm[j] = j;
    for (int i = 0; i < k; i++) { const int q = ipiv[i] - 1, a = E->perm[i]; E->perm[i] = E->perm[q]; E->perm[q] = a; }
    for (int j = 0; j < n; j++) E->iperm[E->perm[j]] = j;
    }
    free(ipiv);
    /* M = L1^-T L2' */
    E->M = xm(sizeof(double) * ((size_t)k * r + 1));
    for (int j = 0; j < r; j++) for (int i = 0; i < k; i++) E->M[i + (size_t)j * k] = E->LU[(k + j) + (size_t)i * n];
    as_dtrsm("L", "L", "T", "U", &k, &r, &as_d1, E->LU, &n, E->M, &k);
    /* L1 and U by columns, their nonzeros (U's diagonal last in each column), then the dense factor freed */
    { long nl = 0, nu = 0;
      for (int j = 0; j < k; j++) { for (int i = j + 1; i < k; i++) nl += E->LU[i + (size_t)j * n] != 0; for (int i = 0; i <= j; i++) nu += E->LU[i + (size_t)j * n] != 0 || i == j; }
      E->Lp = xm(sizeof(int) * (k + 1)); E->Li = xm(sizeof(int) * (nl + 1)); E->Lx = xm(sizeof(double) * (nl + 1));
      E->Up = xm(sizeof(int) * (k + 1)); E->Ui = xm(sizeof(int) * (nu + 1)); E->Ux = xm(sizeof(double) * (nu + 1));
      long a = 0, b = 0;
      for (int j = 0; j < k; j++) {
          E->Lp[j] = (int)a; E->Up[j] = (int)b;
          for (int i = j + 1; i < k; i++) { const double v = E->LU[i + (size_t)j * n]; if (v != 0) { E->Li[a] = i; E->Lx[a] = v; a++; } }
          for (int i = 0; i <= j; i++) { const double v = E->LU[i + (size_t)j * n]; if (v != 0 || i == j) { E->Ui[b] = i; E->Ux[b] = v; b++; } }
      }
      E->Lp[k] = (int)a; E->Up[k] = (int)b; }
    free(E->LU); E->LU = NULL;
    /* Z'HZ: HZ by columns (z_j = e_N(j) - sum_i M_ij e_B(i)), then H_NN-part less M' (HZ)_B */
    double *z = xm(sizeof(double) * (n + 1)), *hz = xm(sizeof(double) * (n + 1));
    double *HB = xm(sizeof(double) * ((size_t)k * r + 1)), *Hr = xm(sizeof(double) * ((size_t)r * r + 1));
    for (int j = 0; j < r; j++) {
        memset(z, 0, sizeof(double) * n);
        z[E->perm[k + j]] = 1.0;
        for (int i = 0; i < k; i++) z[E->perm[i]] = -E->M[i + (size_t)j * k];
        qp_hmul(P, z, hz);
        for (int i = 0; i < k; i++) HB[i + (size_t)j * k] = hz[E->perm[i]];
        for (int i = 0; i < r; i++) Hr[i + (size_t)j * r] = hz[E->perm[k + i]];
    }
    { const double m1 = -1.0; as_dgemm("T", "N", &r, &r, &k, &m1, E->M, &k, HB, &k, &as_d1, Hr, &r); }
    free(HB); free(z); free(hz);
    /* the reduced rows: the other rows a_i Z, then -M(i,:) for each basic variable with a finite bound */
    E->bb = xm(sizeof(int) * (k + 1));
    for (int i = 0; i < k; i++) { const int j = E->perm[i]; if (fin_e(P->xl[j]) || fin_e(P->xu[j])) E->bb[E->nbb++] = i; }
    E->mr = E->mi + E->nbb;
    const int mr = E->mr;
    double *Ar = xz(sizeof(double) * ((size_t)mr * r + 1));          /* (dense, mr x r by columns) */
    { int *rowpos = xm(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) rowpos[i] = -1; for (int t = 0; t < E->mi; t++) rowpos[E->irow[t]] = t;
      for (int jj = 0; jj < n; jj++) {
          const int pos = E->iperm[jj];
          for (int p = P->Ap[jj]; p < P->Ap[jj + 1]; p++) {
              const int t = rowpos[P->Ai[p]]; if (t < 0) continue;
              const double a = P->Ax[p];
              if (pos >= k) Ar[t + (size_t)(pos - k) * mr] += a;
              else for (int j = 0; j < r; j++) Ar[t + (size_t)j * mr] -= a * E->M[pos + (size_t)j * k];
          }
      }
      free(rowpos); }
    for (int t = 0; t < E->nbb; t++) { const int i = E->bb[t]; for (int j = 0; j < r; j++) Ar[E->mi + t + (size_t)j * mr] = -E->M[i + (size_t)j * k]; }
    /* the reduced problem (dense CSC: H's lower triangle, symmetrized; A's nonzeros) */
    QpProb R; memset(&R, 0, sizeof R);
    R.n = r; R.m = mr; R.c0 = 0;
    R.Hp = xm(sizeof(int) * (r + 1)); R.Hi = xm(sizeof(int) * ((size_t)r * (r + 1) / 2 + 1)); R.Hx = xm(sizeof(double) * ((size_t)r * (r + 1) / 2 + 1));
    { long q = 0; for (int j = 0; j < r; j++) { R.Hp[j] = (int)q; for (int i = j; i < r; i++) { const double v = 0.5 * (Hr[i + (size_t)j * r] + Hr[j + (size_t)i * r]); if (v != 0 || i == j) { R.Hi[q] = i; R.Hx[q] = v; q++; } } } R.Hp[r] = (int)q; }
    free(Hr);
    long nza = 0; for (size_t e = 0; e < (size_t)mr * r; e++) nza += Ar[e] != 0;
    R.Ap = xm(sizeof(int) * (r + 1)); R.Ai = xm(sizeof(int) * (nza + 1)); R.Ax = xm(sizeof(double) * (nza + 1));
    { long q = 0; for (int j = 0; j < r; j++) { R.Ap[j] = (int)q; for (int t = 0; t < mr; t++) { const double v = Ar[t + (size_t)j * mr]; if (v != 0) { R.Ai[q] = t; R.Ax[q] = v; q++; } } } R.Ap[r] = (int)q; }
    free(Ar);
    E->xb = xm(sizeof(double) * (k + 1)); E->g = xm(sizeof(double) * (n + 1)); E->ax = xm(sizeof(double) * ((size_t)m + 1));
    E->cr = xm(sizeof(double) * (r + 1)); E->t = xm(sizeof(double) * (n + 1));
    R.c = xm(sizeof(double) * (r + 1)); R.rl = xm(sizeof(double) * (mr + 1)); R.ru = xm(sizeof(double) * (mr + 1));
    R.xl = xm(sizeof(double) * (r + 1)); R.xu = xm(sizeof(double) * (r + 1));
    Q->E = E;
    qp_elim_data(Q, R.c, R.rl, R.ru, R.xl, R.xu);
    QpOpts o = Q->o; o.verbose = Q->o.verbose > 1 ? Q->o.verbose - 1 : 0;
    g_qp_convex_known = Q->convex > 0; E->in = qp_create(&R, &o); g_qp_convex_known = 0;
    qp_prob_free(&R);
    if (!E->in || !E->in->built || E->in->convex < 0) { qp_elim_free(Q); return 0; }
    if (Q->o.verbose) qp_printf("  [active-set method: %d equality rows eliminated, %d variables and %d rows left]\n", k, r, mr);
    return 1;
}
/* the solve through the reduced model; -1 when the point misses the tolerance on the problem as given */
static int qp_elim_solve(QpSolver *Q, QpRes *Res, double t0) {
    QpElim *E = Q->E; const QpProb *P = &Q->P; const int n = E->n, m = E->m, k = E->k, r = E->r, mr = E->mr;
    QpSolver *I = E->in;
    qp_elim_data(Q, I->P.c, I->P.rl, I->P.ru, I->P.xl, I->P.xu);
    I->dirty = 1;
    QpRes R; memset(&R, 0, sizeof R);
    const int st = qp_resolve_body(I, &R);
    /* x */
    memcpy(E->t, E->xb, sizeof(double) * k);
    const double m1 = -1.0;
    as_dgemv("N", &k, &r, &m1, E->M, &k, R.x, &as_one, &as_d1, E->t, &as_one);         /* x_B = x_pB - M w */
    for (int i = 0; i < k; i++) Res->x[E->perm[i]] = E->t[i];
    for (int j = 0; j < r; j++) Res->x[E->perm[k + j]] = R.x[j];
    /* the multipliers: the other rows', the bounds', then the equalities' from B'y_E = g_B */
    memset(Res->y, 0, sizeof(double) * (m + 1)); memset(Res->z, 0, sizeof(double) * (n + 1));
    for (int t = 0; t < E->mi; t++) Res->y[E->irow[t]] = R.y[t];
    for (int t = 0; t < E->nbb; t++) Res->z[E->perm[E->bb[t]]] = R.y[E->mi + t];
    for (int j = 0; j < r; j++) Res->z[E->perm[k + j]] = R.z[j];
    double *g = E->g;
    qp_hmul(P, Res->x, g);
    for (int j = 0; j < n; j++) g[j] += P->c[j] - Res->z[j];
    for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) if (!E->iseq[P->Ai[p]]) g[j] -= P->Ax[p] * Res->y[P->Ai[p]];
    double *yb = E->cr; (void)mr;
    double *v = E->ax; for (int i = 0; i < k; i++) v[i] = g[E->perm[i]];
    elim_btinv(E, v);                                              /* B'y_E = g_B */
    for (int t = 0; t < k; t++) Res->y[E->eqrow[t]] = v[t];
    (void)yb;
    qp_errors(P, Res->x, Res->y, Res->z, &Res->pres, &Res->dres, &Res->gap);
    Res->obj = qp_objective(P, Res->x);
    Res->iter = R.iter; Res->method = QP_METHOD_ACTIVESET;
    const double tol = Q->o.tol > 0 ? Q->o.tol : 1e-8, e = fmax(Res->pres, fmax(Res->dres, Res->gap));
    int s = st;
    if (st == QP_OPTIMAL || st == QP_REDUCED) s = e <= 10 * tol ? QP_OPTIMAL : -1;
    qp_res_free(&R);
    Res->status = s; Res->time = now() - t0;
    return s;
}
/* (28 October 2026, the SOCP developer) the active-set method's setup when the model is made (the test of convexity, the
 * dense factor of H and J): the setup of a receding-horizon controller, apart from its first solve, as other solvers do it
 * (DAQP factors at its setup; the owner's independent QP benchmark times the setup and the solves apart). Run again by
 * a solve when it has not been done (the factorization failed: the solve then reports numerical difficulties) */
static int qp_elim_build(QpSolver *Q);
static void qp_prepare(QpSolver *Q) {
    if ((Q->o.method != QP_METHOD_ACTIVESET && Q->o.method != QP_METHOD_AUTO) || Q->built || Q->failed) return;
    if (Q->o.method == QP_METHOD_AUTO && !qp_auto_as(Q)) return;      /* (the automatic choice takes the interior-point method) */
    if (!Q->convex) Q->convex = qp_convex_exact(&Q->P) ? -1 : 1;
    if (Q->convex < 0) return;
    if (Q->elim == 0) { Q->elim = qp_elim_build(Q) ? 1 : -1; if (Q->elim == 1) return; }
    if (Q->elim == 1) return;
    if (Q->o.method == QP_METHOD_AUTO && qp_elim_setup_work(&Q->P) >= 0 && !qp_as_dense_small(&Q->P)) { Q->auto_as = 0; return; }   /* (chosen for the elimination, which could not be built) */
    as_alloc(&Q->S, &Q->P);
    if (as_factor(&Q->S)) { as_free(&Q->S); Q->failed = 1; return; }
    as_reset(&Q->S);
    Q->built = 1; Q->dirty = 0;
}
void qp_ipm_memo_free(void *p);
void qp_destroy(QpSolver *Q) { if (!Q) return; qp_ipm_memo_free(Q->memo); qp_elim_free(Q); if (Q->built) as_free(&Q->S); qp_prob_free(&Q->P); free(Q); }
