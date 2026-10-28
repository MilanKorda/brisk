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
#include <time.h>

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
    double *x, *cc, *d, *z, *r, *s, *wv, *nv;
    double eps;
    long iters;
} AS;

/* d = J' n_k */
static void jt_n(const AS *S, int k, double *d) {
    const int n = S->n; const Con *c = &S->con[k];
    if (c->kind == 1) { const int j = c->idx; for (int i = 0; i < n; i++) d[i] = c->side * S->J[j + (size_t)i * n]; return; }
    for (int i = 0; i < n; i++) d[i] = 0;
    for (int p = S->Rp[c->idx]; p < S->Rp[c->idx + 1]; p++) {
        const int j = S->Ri[p]; const double a = c->side * S->Rx[p];
        const double *Jrow = S->J + j;
        for (int i = 0; i < n; i++) d[i] += a * Jrow[(size_t)i * n];
    }
}
/* n_k' v */
static double n_dot(const AS *S, int k, const double *v) {
    const Con *c = &S->con[k];
    if (c->kind == 1) return c->side * v[c->idx];
    double s = 0; for (int p = S->Rp[c->idx]; p < S->Rp[c->idx + 1]; p++) s += S->Rx[p] * v[S->Ri[p]];
    return c->side * s;
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
    /* t = J' cc */
    for (int i = 0; i < n; i++) { const double *Ji = S->J + (size_t)i * n; double s = 0; for (int k = 0; k < n; k++) s += Ji[k] * S->cc[k]; t[i] = s; }
    /* v (q) = R^-T b_A */
    for (int i = 0; i < q; i++) { double s = S->con[S->act[i]].b; for (int k = 0; k < i; k++) s -= S->R[k + (size_t)i * n] * v[k]; v[i] = s / S->R[i + (size_t)i * n]; }
    for (int k = 0; k < n; k++) S->x[k] = 0;
    for (int i = q; i < n; i++) { const double *Ji = S->J + (size_t)i * n; const double a = -t[i]; for (int k = 0; k < n; k++) S->x[k] += a * Ji[k]; }
    for (int i = 0; i < q; i++) { const double *Ji = S->J + (size_t)i * n; const double a = v[i]; for (int k = 0; k < n; k++) S->x[k] += a * Ji[k]; }
    /* u = R^-1 J1' (G x + cc) */
    double *g = S->d;
    for (int k = 0; k < n; k++) g[k] = S->cc[k];
    for (int j = 0; j < n; j++) { const double xj = S->x[j]; if (xj != 0) { const double *Gj = S->G + (size_t)j * n; for (int k = 0; k < n; k++) g[k] += Gj[k] * xj; } }
    for (int i = 0; i < q; i++) { const double *Ji = S->J + (size_t)i * n; double s = 0; for (int k = 0; k < n; k++) s += Ji[k] * g[k]; t[i] = s; }
    for (int i = q - 1; i >= 0; i--) { double s = t[i]; for (int k = i + 1; k < q; k++) s -= S->R[i + (size_t)k * n] * S->u[k]; S->u[i] = s / S->R[i + (size_t)i * n]; }
}

/* the method of Goldfarb and Idnani from the current active set (made dual feasible first); returns 0 optimal,
 * 1 infeasible, 3 iteration limit */
static int gi_solve(AS *S, long maxit, double tol, int verbose) {
    const int n = S->n;
    /* dual feasibility of the starting set: drop inequalities with negative multipliers, one at a time */
    for (;;) {
        eq_solve(S);
        int worst = -1; double wu = -1e-14;
        for (int i = 0; i < S->q; i++) if (!S->con[S->act[i]].eq && S->u[i] < wu) { wu = S->u[i]; worst = i; }
        if (worst < 0) break;
        drop_con(S, worst);
    }
    for (long it = 0; it < maxit; it++) {
        S->iters++;
        /* the most violated constraint (relative to its normal's size); equalities not yet active come first */
        int p = -1; double worst = 0;
        for (int k = 0; k < S->nc; k++) {
            if (S->isact[k]) continue;
            const double sk = slack(S, k);
            const Con *c = &S->con[k];
            const double scale = 1 + fabs(c->b);
            if (c->eq) { if (fabs(sk) > tol * scale) { const double v = fabs(sk) / scale * 1e6; if (v > worst) { worst = v; p = k; } } }
            else if (sk < -tol * scale) { const double v = -sk / scale; if (v > worst) { worst = v; p = k; } }
        }
        if (p < 0) return 0;
        /* an equality with a positive slack enters with the opposite side (n'x = b: the side does not matter) */
        if (S->con[p].eq && slack(S, p) > 0) { S->con[p].side = -S->con[p].side; S->con[p].b = -S->con[p].b; }
        double up = 0;               /* the multiplier of the entering constraint */
        for (;;) {
            jt_n(S, p, S->d);
            const int q = S->q;
            /* z = J2 d2 (the primal step), r = R^-1 d1 (the change of the active multipliers) */
            for (int k = 0; k < n; k++) S->z[k] = 0;
            for (int i = q; i < n; i++) { const double a = S->d[i]; if (a != 0) { const double *Ji = S->J + (size_t)i * n; for (int k = 0; k < n; k++) S->z[k] += a * Ji[k]; } }
            for (int i = q - 1; i >= 0; i--) { double s = S->d[i]; for (int k = i + 1; k < q; k++) s -= S->R[i + (size_t)k * n] * S->r[k]; S->r[i] = s / S->R[i + (size_t)i * n]; }
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
};

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
    S->wv = xz(sizeof(double) * (n + 1)); S->nv = xz(sizeof(double) * (n + 1));
}
static void as_free(AS *S) {
    free(S->Rp); free(S->Ri); free(S->Rx); free(S->con); free(S->G); free(S->L); free(S->J); free(S->R);
    free(S->act); free(S->u); free(S->isact); free(S->x); free(S->cc); free(S->d); free(S->z); free(S->r); free(S->s); free(S->wv); free(S->nv);
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

/* the full solve with the proximal-point loop; S's active set is the start */
static int as_run(AS *S, const QpOpts *o, QpRes *Res, double t0) {
    const QpProb *P = S->P; const int n = S->n, m = S->m;
    const double tol = o->tol > 0 ? o->tol : 1e-8;
    const long maxit = o->maxit > 0 ? o->maxit : 100L * (n + S->nc) + 1000;
    double *xk = xz(sizeof(double) * (n + 1));
    int st = 0, outer = 0;
    for (int j = 0; j < n; j++) xk[j] = S->x[j];
    for (;;) {
        for (int j = 0; j < n; j++) S->cc[j] = P->c[j] - S->eps * xk[j];
        st = gi_solve(S, maxit, tol, o->verbose);
        outer++;
        if (st || S->eps == 0) break;
        double dx = 0, xn = 0; for (int j = 0; j < n; j++) { dx = fmax(dx, fabs(S->x[j] - xk[j])); xn = fmax(xn, fabs(S->x[j])); }
        for (int j = 0; j < n; j++) xk[j] = S->x[j];
        if (dx <= 1e-2 * tol * (1 + xn) || outer >= 10000) break;
        if (o->timelimit > 0 && now() - t0 > o->timelimit) { st = 5; break; }
    }
    free(xk);
    /* the result */
    memset(Res->y, 0, sizeof(double) * (m + 1)); memset(Res->z, 0, sizeof(double) * (n + 1));
    memcpy(Res->x, S->x, sizeof(double) * n);
    for (int i = 0; i < S->q; i++) {
        const Con *c = &S->con[S->act[i]]; const double u = S->u[i] * c->side;
        if (c->kind == 0) Res->y[c->idx] += u; else Res->z[c->idx] += u;
    }
    Res->iter = (int)S->iters;
    Res->status = st == 0 ? QP_OPTIMAL : st == 1 ? QP_PRIMAL_INFEASIBLE : st == 5 ? QP_TIME : QP_MAXIT;
    Res->obj = qp_objective(P, Res->x);
    qp_errors(P, Res->x, Res->y, Res->z, &Res->pres, &Res->dres, &Res->gap);
    if (Res->status == QP_OPTIMAL && fmax(Res->pres, fmax(Res->dres, Res->gap)) > 1e3 * tol) Res->status = QP_REDUCED;
    if (o->verbose) qp_printf("  active-set method: %ld iterations, %d proximal-point solves (eps %.1e), %d active constraints\n", S->iters, outer, S->eps, S->q);
    Res->time = now() - t0;
    return Res->status;
}

/* need_pd: only when H is positive definite (the method's own test: the Cholesky factor of H with a margin, eps = 0);
 * otherwise -1 and nothing solved (Res empty) */
static int as_solve(const QpProb *P, const QpOpts *o, QpRes *Res, int need_pd) {
    const double t0 = now();
    QpOpts od; if (!o) { qp_default_opts(&od); o = &od; }
    memset(Res, 0, sizeof(*Res));
    if (need_pd && (double)P->n * P->n > 4e8) return -1;
    Res->x = xz(sizeof(double) * (P->n + 1)); Res->y = xz(sizeof(double) * (P->m + 1)); Res->z = xz(sizeof(double) * (P->n + 1));
    if ((double)P->n * P->n > 4e8) { if (o->verbose) qp_printf("the active-set method is dense: n = %d is too large (the interior-point method is the one for it)\n", P->n); Res->status = QP_INPUT; return Res->status; }
    AS S; as_alloc(&S, P);
    const int fr = as_factor(&S);
    if (need_pd && (fr || S.eps != 0)) { as_free(&S); qp_res_free(Res); memset(Res, 0, sizeof(*Res)); return -1; }
    if (o->verbose) qp_printf("BRISK QP active-set method (the dual method of Goldfarb and Idnani): n = %d, %d constraints (rows and bounds as one-sided inequalities)\n", P->n, S.nc);
    if (fr) { Res->status = QP_NUMERICAL; as_free(&S); return Res->status; }
    as_reset(&S);
    as_run(&S, o, Res, t0);
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
QpSolver *qp_create(const QpProb *P, const QpOpts *o) {
    QpSolver *Q = xz(sizeof(QpSolver));
    Q->P = copy_prob(P);
    if (o) Q->o = *o; else qp_default_opts(&Q->o);
    return Q;
}
const QpProb *qp_problem(const QpSolver *Q) { return &Q->P; }
/* a changed bound or row bound: the constraints are rebuilt (their indices change when a side appears or disappears),
 * and the active set is carried over by the identity of the constraint (row or column, side) */
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
int qp_set_var_bounds(QpSolver *Q, int j, double lo, double hi) {
    if (j < 0 || j >= Q->P.n) return 1;
    Q->P.xl[j] = lo; Q->P.xu[j] = hi; rebuild(Q); return 0;
}
int qp_set_row_bounds(QpSolver *Q, int i, double lo, double hi) {
    if (i < 0 || i >= Q->P.m) return 1;
    Q->P.rl[i] = lo; Q->P.ru[i] = hi; rebuild(Q); return 0;
}
int qp_set_cost(QpSolver *Q, const double *c) { memcpy(Q->P.c, c, sizeof(double) * Q->P.n); return 0; }
int qp_resolve(QpSolver *Q, QpRes *Res) {
    const double t0 = now();
    const QpProb *P = &Q->P;
    if (!Res->x) { Res->x = xz(sizeof(double) * (P->n + 1)); Res->y = xz(sizeof(double) * (P->m + 1)); Res->z = xz(sizeof(double) * (P->n + 1)); }
    if (!Q->convex) Q->convex = qp_convex_exact(P) ? -1 : 1;        /* (H does not change between solves) */
    if (Q->convex < 0) { Res->status = QP_INPUT; Res->iter = 0; Res->time = now() - t0; return Res->status; }
    /* (26 October 2026, the integrator) QP_METHOD_AUTO (-qpauto): the choice of the one-shot solve (qp_solve), so that a
     * caller that does not choose (JuMP) is never held to the active-set method: within the default's size bound
     * (qp_as_small) the active-set method from the previous active set, else, for an H that is not positive definite (its
     * Cholesky factor fails; H does not change between solves, so for good) and after an active-set solve that does not end
     * optimal, the interior-point method from the start */
    const int autom = Q->o.method == QP_METHOD_AUTO;
    if (Q->o.method == QP_METHOD_IPM || (autom && !qp_as_small(P))) { qp_res_free(Res); return qp_solve_ipm(P, &Q->o, Res); }
    if (!Q->built) {
        as_alloc(&Q->S, P);
        if (as_factor(&Q->S)) {
            if (!autom) { Res->status = QP_NUMERICAL; return Res->status; }
            as_free(&Q->S); Q->o.method = QP_METHOD_IPM;
            qp_res_free(Res); return qp_solve_ipm(P, &Q->o, Res);
        }
        as_reset(&Q->S);
        Q->built = 1;
    }
    Q->S.P = P; Q->S.iters = 0;
    const int st = as_run(&Q->S, &Q->o, Res, t0);
    if (autom && st != QP_OPTIMAL && st >= 0) { qp_res_free(Res); return qp_solve_ipm(P, &Q->o, Res); }
    return st;
}
void qp_destroy(QpSolver *Q) { if (!Q) return; if (Q->built) as_free(&Q->S); qp_prob_free(&Q->P); free(Q); }
