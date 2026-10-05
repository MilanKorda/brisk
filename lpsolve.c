/* lpsolve.c - a primal-dual interior-point method for linear programs in bounded standard form
 *
 *     min c'x,   A x = b,   0 <= x_j <= u_j   (u_j = +inf: no upper bound; no free variables)
 *
 * Mehrotra's predictor-corrector from an infeasible start, with
 *   - upper bounds kept as bounds (x + s = u, multipliers w): the normal equations are
 *     A Theta A' dy = r with Theta_j = 1 / (z_j / x_j + w_j / s_j): no rows for the bounds;
 *   - the normal equations factored by the supernodal sparse Cholesky code (sparsechol.c: AMD
 *     order, BLAS panels, dependent rows by tiny-pivot replacement), assembled row by row into
 *     the pattern of A A' of the sparse columns;
 *   - dense columns (long against the others) left out of the factored matrix and brought back
 *     by the Schur complement of the Sherman-Morrison-Woodbury formula;
 *   - every solve refined by preconditioned CG on the true matrix A Theta A' (one or two steps
 *     in the regular case, more when pivots were replaced or dense columns are ill-conditioned);
 *   - separate primal and dual step lengths, Gondzio's multiple centrality correctors (their
 *     number from the cost of a factorization against a solve, by flop counts);
 *   - Ruiz equilibration of A; termination on the residuals and the gap of the problem as given.
 * The method has no certificate of infeasibility: when the iteration diverges or stalls the
 * caller hands the problem to the cone solver (homogeneous self-dual model).                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "brisk.h"
#include "lpsolve.h"

typedef struct SChol SChol;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_tinypiv(SChol *S, double t);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
size_t schol_nnz(const SChol *S);
int    schol_set_amd(int mode);
void   schol_zero(SChol *S);
double schol_flops(const SChol *S);
int    schol_ntiny(const SChol *S);
void   schol_set_solve_seq(int on);
void   schol_set_perm(const int *perm);
void   schol_set_nd(int on);

static double lp_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *xm(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (LP solver)\n"); exit(1); } return p; }
static void *xz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (LP solver)\n"); exit(1); } return p; }
static double dot(int n, const double *a, const double *b) { double s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static double nrminf(int n, const double *a) { double s = 0; for (int i = 0; i < n; i++) { const double v = fabs(a[i]); if (v > s) s = v; } return s; }

typedef struct {
    int m, n;
    int *Ap, *Ai; double *Ax;          /* scaled A by columns (rows ascending in a column) */
    int *Rp, *Rj, *Rq; double *Rx;     /* the sparse columns by rows; Rq: the position of the entry in its column */
    double *b, *c, *u; char *bd, *fr;  /* scaled data; bd[j]: x_j has an upper bound; fr[j]: x_j is free */
    double *rs, *cs;                   /* row and column scales: A_scaled = diag(rs) A diag(cs) */
    int nd, *dcol; char *isd;          /* dense columns */
    SChol *sc; int *Mp, *Mi; size_t *Mo;   /* the lower pattern of A_s A_s' by columns: rows Mi >= the column, offsets Mo */
    double *V, *G, *gw;                /* dense columns: V = M_s^-1 A_d (m x nd), G = Theta_d^-1 + A_d' V = L L' */
    double *theta;
    double *w1, *w2, *w3, *w4, *wn, *acc; long nref; double worst;   /* work: m, m, m, n, m */
    double *pr, *pz, *pp, *pq;         /* CG work (m) */
    double reg;
    long nfact, nsolve, ncg; int ntiny;
    double t_fact, t_solve;
} LpS;

static void A_mul(const LpS *S, const double *x, double *y) {           /* y = A x */
    for (int i = 0; i < S->m; i++) y[i] = 0;
    for (int j = 0; j < S->n; j++) { const double xj = x[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) y[S->Ai[p]] += S->Ax[p] * xj; }
}
static void At_mul(const LpS *S, const double *y, double *x) {          /* x = A' y */
    for (int j = 0; j < S->n; j++) { double s = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * y[S->Ai[p]]; x[j] = s; }
}
/* q = A Theta A' v (the true matrix: the regularization is in the factor, the preconditioner, only) */
static void M_mul(LpS *S, const double *v, double *q) {
    At_mul(S, v, S->wn);
    for (int j = 0; j < S->n; j++) S->wn[j] *= S->theta[j];
    A_mul(S, S->wn, q);
}
/* the factorization of A_s Theta A_s' + reg (and the Schur complement of the dense columns); 0: done */
static int ne_factor(LpS *S) {
    const double t0 = lp_time();
    const int m = S->m;
    schol_zero(S->sc);
    double *pm = schol_values(S->sc), *acc = S->acc;
    for (int i = 0; i < m; i++) {
        for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) acc[S->Mi[e]] = 0.0;
        for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) {
            const int j = S->Rj[q]; const double t = S->Rx[q] * S->theta[j];
            for (int p = S->Rq[q]; p < S->Ap[j + 1]; p++) acc[S->Ai[p]] += t * S->Ax[p];
        }
        for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) pm[S->Mo[e]] = acc[S->Mi[e]];
        pm[S->Mo[S->Mp[i]]] += S->reg;                    /* (the diagonal is the first entry of the column) */
    }
    schol_set_tinypiv(S->sc, getenv("BRISK_LPTINY") ? atof(getenv("BRISK_LPTINY")) : 1e-20);
    const int bad = schol_factor(S->sc, 0.0);
    S->ntiny += schol_ntiny(S->sc);
    S->nfact++;
    if (bad) { S->t_fact += lp_time() - t0; return 1; }
    if (S->nd > 0) {
        const int nd = S->nd;
        for (int d = 0; d < nd; d++) {
            const int j = S->dcol[d]; double *v = S->V + (size_t)d * m;
            for (int i = 0; i < m; i++) v[i] = 0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v[S->Ai[p]] = S->Ax[p];
            schol_solve(S->sc, v, S->w3);
        }
        for (int d = 0; d < nd; d++) {
            const double *v = S->V + (size_t)d * m;
            for (int e = 0; e <= d; e++) {
                const int j = S->dcol[e]; double s = 0;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * v[S->Ai[p]];
                S->G[d + (size_t)e * nd] = s; S->G[e + (size_t)d * nd] = s;
            }
            S->G[d + (size_t)d * nd] += 1.0 / S->theta[S->dcol[d]];
        }
        int info = 0;
        BL(dpotrf_)("L", &nd, S->G, &nd, &info);
        if (info) { S->t_fact += lp_time() - t0; return 1; }
    }
    S->t_fact += lp_time() - t0;
    return 0;
}
/* z = P^-1 r: the factored sparse part and the dense columns by the Woodbury formula */
static void ne_prec(LpS *S, const double *r, double *z) {
    const int m = S->m;
    memcpy(z, r, sizeof(double) * (size_t)m);
    schol_solve(S->sc, z, S->w3);
    if (S->nd > 0) {
        const int nd = S->nd, one = 1; int info = 0;
        for (int d = 0; d < nd; d++) {
            const int j = S->dcol[d]; double s = 0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * z[S->Ai[p]];
            S->gw[d] = s;
        }
        BL(dpotrs_)("L", &nd, &one, S->G, &nd, S->gw, &nd, &info);
        for (int d = 0; d < nd; d++) { const double g = S->gw[d]; const double *v = S->V + (size_t)d * m; if (g != 0.0) for (int i = 0; i < m; i++) z[i] -= g * v[i]; }
    }
}
/* (A Theta A') x = r: the preconditioner, then CG on the true matrix until the residual is below
 * rtol |r| (at most maxcg steps); returns the relative residual */
static double ne_cg(LpS *S, double *x, double nr, double rn, double rtol, int maxcg);
static double ne_solve(LpS *S, const double *r, double *x, double rtol, int maxcg) {
    const double t0 = lp_time();
    const int m = S->m;
    double *res = S->pr, *q = S->pq;
    const double nr = sqrt(dot(m, r, r));
    S->nsolve++;
    if (nr == 0) { memset(x, 0, sizeof(double) * (size_t)m); S->t_solve += lp_time() - t0; return 0; }
    ne_prec(S, r, x);
    M_mul(S, x, q);
    for (int i = 0; i < m; i++) res[i] = r[i] - q[i];
    const double rn0 = sqrt(dot(m, res, res));
    if (rn0 <= rtol * nr || maxcg <= 0) { S->t_solve += lp_time() - t0; return rn0 / nr; }
    const double out = ne_cg(S, x, nr, rn0, rtol, maxcg);
    S->t_solve += lp_time() - t0;
    return out;
}
/* CG on A Theta A' from x with the residual in S->pr (norm rn; the right-hand side's norm nr):
 * the best iterate is left in x; returns its relative residual */
static double ne_cg(LpS *S, double *x, double nr, double rn, double rtol, int maxcg) {
    const int m = S->m;
    double *res = S->pr, *z = S->pz, *p = S->pp, *q = S->pq;
    ne_prec(S, res, z);
    memcpy(p, z, sizeof(double) * (size_t)m);
    double rz = dot(m, res, z), best = rn;
    double rprev2 = rn, rprev1 = rn;
    double *xb = S->w2; memcpy(xb, x, sizeof(double) * (size_t)m);
    for (int k = 0; k < maxcg; k++) {
        M_mul(S, p, q);
        const double pq = dot(m, p, q);
        if (!(pq > 0) || !isfinite(pq)) break;
        const double al = rz / pq;
        for (int i = 0; i < m; i++) { x[i] += al * p[i]; res[i] -= al * q[i]; }
        S->ncg++;
        rn = sqrt(dot(m, res, res));
        if (rn < best) { best = rn; memcpy(xb, x, sizeof(double) * (size_t)m); }
        if (rn <= rtol * nr) break;
        if (rn > 1e3 * best) break;
        /* no progress in two steps (replaced pivots: the preconditioner is singular there): the direct solve stands */
        if (k >= 1 && rn > 0.5 * rprev2) break;
        rprev2 = rprev1; rprev1 = rn;
        ne_prec(S, res, z);
        const double rz2 = dot(m, res, z), be = rz2 / rz; rz = rz2;
        for (int i = 0; i < m; i++) p[i] = z[i] + be * p[i];
    }
    memcpy(x, xb, sizeof(double) * (size_t)m);
    if (best / nr > S->worst) S->worst = best / nr;
    return best / nr;
}

static void lps_free(LpS *S) {
    free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Rj); free(S->Rq); free(S->Rx); free(S->b); free(S->c); free(S->u); free(S->bd); free(S->fr);
    free(S->rs); free(S->cs); free(S->dcol); free(S->isd); if (S->sc) schol_free(S->sc); free(S->Mp); free(S->Mi); free(S->Mo);
    free(S->V); free(S->G); free(S->gw); free(S->theta); free(S->w1); free(S->w2); free(S->w4); free(S->w3); free(S->wn); free(S->acc);
    free(S->pr); free(S->pz); free(S->pp); free(S->pq);
}

/* the dense columns, the sparse columns by rows, the pattern of A_s A_s' and its analysis
 * (again, without dense columns, when the Woodbury correction loses its accuracy); 0: done */
static int ne_setup(LpS *S, int usedense) {
    const int m = S->m, n = S->n, nz = S->Ap[n];
    free(S->Rp); free(S->Rj); free(S->Rq); free(S->Rx); free(S->Mp); free(S->Mi); free(S->Mo); free(S->V); free(S->G); free(S->gw);
    S->Rp = S->Rj = S->Rq = S->Mp = S->Mi = NULL; S->Rx = S->V = S->G = S->gw = NULL; S->Mo = NULL;
    if (S->sc) { schol_free(S->sc); S->sc = NULL; }
    if (!S->isd) { S->isd = xz(n + 1); S->dcol = xm(sizeof(int) * (n + 1)); }
    memset(S->isd, 0, n + 1); S->nd = 0;
    /* dense columns: long against the average and against the rows, and few */
    if (usedense && m >= 500) {
        double avg = (double)nz / fmax(n, 1);
        const double thr = fmax(fmax(50.0, 10.0 * avg), 0.1 * m);
        int nd = 0;
        for (int j = 0; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] > thr) nd++;
        if (nd > 0 && nd <= 400 && nd < n / 4) { for (int j = 0; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] > thr) { S->isd[j] = 1; S->dcol[S->nd++] = j; } }
    }
    /* the sparse columns by rows */
    S->Rp = xz(sizeof(int) * (m + 2));
    int nzs = 0;
    for (int j = 0; j < n; j++) if (!S->isd[j]) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { S->Rp[S->Ai[p] + 2]++; nzs++; }
    for (int i = 0; i < m; i++) S->Rp[i + 2] += S->Rp[i + 1];
    S->Rj = xm(sizeof(int) * (nzs + 1)); S->Rq = xm(sizeof(int) * (nzs + 1)); S->Rx = xm(sizeof(double) * (nzs + 1));
    for (int j = 0; j < n; j++) if (!S->isd[j]) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = S->Rp[S->Ai[p] + 1]++; S->Rj[q] = j; S->Rq[q] = p; S->Rx[q] = S->Ax[p]; }
    /* the pattern of A_s A_s' (lower, by columns; the diagonal first) */
    int *mark = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) mark[i] = -1;
    long cap = 4L * nzs + m + 16, w = 0;
    int *Mi = xm(sizeof(int) * cap); S->Mp = xm(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) {
        S->Mp[i] = (int)w;
        if (w + m + 1 > cap) { cap = 2 * cap + m; Mi = realloc(Mi, sizeof(int) * cap); }
        Mi[w++] = i; mark[i] = i;
        for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) {
            const int j = S->Rj[q];
            for (int p = S->Rq[q] + 1; p < S->Ap[j + 1]; p++) { const int k = S->Ai[p]; if (mark[k] != i) { mark[k] = i; Mi[w++] = k; } }
        }
    }
    S->Mp[m] = (int)w; S->Mi = Mi;
    free(mark);
    int *deg = xz(sizeof(int) * (m + 1)), **nbr = xm(sizeof(int *) * (m + 1));
    for (int i = 0; i < m; i++) for (int e = S->Mp[i] + 1; e < S->Mp[i + 1]; e++) { deg[i]++; deg[Mi[e]]++; }
    int *pool = xm(sizeof(int) * (2 * (w - m) + 1));
    { long o = 0; for (int i = 0; i < m; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; } }
    for (int i = 0; i < m; i++) for (int e = S->Mp[i] + 1; e < S->Mp[i + 1]; e++) { const int k = Mi[e]; nbr[i][deg[i]++] = k; nbr[k][deg[k]++] = i; }
    if (getenv("BRISK_LPDUMP")) {          /* (experiments: the pattern of the normal equations, one row "i: neighbours") */
        FILE *fd = fopen(getenv("BRISK_LPDUMP"), "w");
        if (fd) { fprintf(fd, "%d\n", m); for (int i = 0; i < m; i++) { for (int e = 0; e < deg[i]; e++) fprintf(fd, "%d ", nbr[i][e]); fprintf(fd, "\n"); } fclose(fd); }
    }
    int *fperm = NULL;
    if (getenv("BRISK_LPPERM")) {          /* (experiments: an ordering from a file, perm[new] = old) */
        FILE *fd = fopen(getenv("BRISK_LPPERM"), "r");
        if (fd) { fperm = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) if (fscanf(fd, "%d", &fperm[i]) != 1) { free(fperm); fperm = NULL; break; } fclose(fd); }
        if (fperm) schol_set_perm(fperm);
    }
    const int om = schol_set_amd(getenv("BRISK_LPAMD") ? atoi(getenv("BRISK_LPAMD")) : 2);
    schol_set_nd(getenv("BRISK_LPND") ? atoi(getenv("BRISK_LPND")) : 1);
    S->sc = schol_analyze_adj(m, deg, nbr, (size_t)1 << 31);
    schol_set_nd(0);
    if (fperm) { schol_set_perm(NULL); free(fperm); }
    schol_set_amd(om);
    free(deg); free(nbr); free(pool);
    if (!S->sc) return 1;
    S->Mo = xm(sizeof(size_t) * (w + 1));
    for (int i = 0; i < m; i++) for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) {
        S->Mo[e] = schol_offset(S->sc, (size_t)Mi[e], (size_t)i);
        if (S->Mo[e] == (size_t)-1) return 1;
    }
    if (S->nd) { S->V = xm(sizeof(double) * (size_t)m * S->nd); S->G = xm(sizeof(double) * (size_t)S->nd * S->nd); S->gw = xm(sizeof(double) * (S->nd + 1)); }
    return 0;
}

static void kkt_solve(LpS *S, const double *g, const double *rb, double *dx, double *dy, double rtol, int maxcg);
/* the direction for the right-hand sides (rb, ru, rc, rxz, rsw):
 *   A dx = rb, dx + ds = ru, A'dy + dz - dw = rc, z dx + x dz = rxz, w ds + s dw = rsw */
static void direction(LpS *S, const double *x, const double *s, const double *z, const double *w,
                      const double *rb, const double *ru, const double *rc, const double *rxz, const double *rsw,
                      double *dx, double *dy, double *dz, double *ds, double *dw, double *g, double rtol, int maxcg) {
    const int n = S->n;
    /* g = rc - rxz / x + (rsw - w ru) / s;  dx = Theta (A'dy - g);  A Theta A' dy = rb + A Theta g */
    for (int j = 0; j < n; j++) {
        double v = rc[j];
        if (!S->fr[j]) v -= rxz[j] / x[j];
        if (S->bd[j]) v += (rsw[j] - w[j] * ru[j]) / s[j];
        g[j] = v;
    }
    kkt_solve(S, g, rb, dx, dy, rtol, maxcg);
    for (int j = 0; j < n; j++) {
        dz[j] = S->fr[j] ? 0.0 : (rxz[j] - z[j] * dx[j]) / x[j];
        if (S->bd[j]) { ds[j] = ru[j] - dx[j]; dw[j] = (rsw[j] - w[j] * ds[j]) / s[j]; } else { ds[j] = 0; dw[j] = 0; }
    }
}
/* dx = Theta (A'dy - g), A dx = rb, by the normal equations */
static void kkt_solve(LpS *S, const double *g, const double *rb, double *dx, double *dy, double rtol, int maxcg) {
    const int m = S->m, n = S->n;
    for (int j = 0; j < n; j++) dx[j] = S->theta[j] * g[j];
    A_mul(S, dx, S->w1);
    for (int i = 0; i < m; i++) S->w1[i] += rb[i];
    /* the factor's solve; its residual is rb - A dx (the same vector as rhs - A Theta A'dy, and
     * dx is needed anyway: one product with A instead of two more with A and A') */
    int have_res = 0;
    {
        const double t0 = lp_time();
        const double nr = sqrt(dot(m, S->w1, S->w1));
        S->nsolve++;
        if (nr == 0) memset(dy, 0, sizeof(double) * (size_t)m); else ne_prec(S, S->w1, dy);
        At_mul(S, dy, dx);
        for (int j = 0; j < n; j++) dx[j] = S->theta[j] * (dx[j] - g[j]);
        A_mul(S, dx, S->pr);
        for (int i = 0; i < m; i++) S->pr[i] = rb[i] - S->pr[i];
        const double rn = sqrt(dot(m, S->pr, S->pr));
        have_res = 1;
        if (nr > 0 && rn > rtol * nr && maxcg > 0) {
            ne_cg(S, dy, nr, rn, rtol, maxcg);
            At_mul(S, dy, dx);
            for (int j = 0; j < n; j++) dx[j] = S->theta[j] * (dx[j] - g[j]);
            have_res = 0;
        }
        S->t_solve += lp_time() - t0;
    }
    /* A dx = rb to the accuracy the step needs (the right-hand side rb + A Theta g loses it by
     * cancellation when Theta spreads): correct dy on the residual */
    { const double nrb = nrminf(m, rb);
      for (int pass = 0; pass < 3; pass++) {
          if (have_res) { memcpy(S->w1, S->pr, sizeof(double) * (size_t)m); have_res = 0; }
          else { A_mul(S, dx, S->w1); for (int i = 0; i < m; i++) S->w1[i] = rb[i] - S->w1[i]; }
          const double nr = nrminf(m, S->w1);
          if (!(nr > fmax(1e-3 * nrb, 1e-13)) || !isfinite(nr)) break;
          double *ddy = S->w4;
          ne_solve(S, S->w1, ddy, rtol, maxcg);
          At_mul(S, ddy, S->wn);
          double nn = 0;
          for (int j = 0; j < n; j++) { S->wn[j] *= S->theta[j]; }
          A_mul(S, S->wn, S->acc);
          for (int i = 0; i < m; i++) { const double v = fabs(S->w1[i] - S->acc[i]); if (v > nn) nn = v; S->acc[i] = 0; }
          if (!(nn < 0.5 * nr)) break;                    /* (no gain: the factorization is at its limit) */
          for (int j = 0; j < n; j++) dx[j] += S->wn[j];
          for (int i = 0; i < m; i++) dy[i] += ddy[i];
          S->nref++;
      } }
}
static void step_max(const LpS *S, const double *x, const double *s, const double *z, const double *w,
                     const double *dx, const double *ds, const double *dz, const double *dw, double *ap, double *ad) {
    double a = 1e300, d = 1e300;
    for (int j = 0; j < S->n; j++) {
        if (S->fr[j]) continue;
        if (dx[j] < 0) { const double t = -x[j] / dx[j]; if (t < a) a = t; }
        if (dz[j] < 0) { const double t = -z[j] / dz[j]; if (t < d) d = t; }
        if (S->bd[j]) {
            if (ds[j] < 0) { const double t = -s[j] / ds[j]; if (t < a) a = t; }
            if (dw[j] < 0) { const double t = -w[j] / dw[j]; if (t < d) d = t; }
        }
    }
    *ap = a; *ad = d;
}

int lpipm_solve(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info) {
    const double t0 = lp_time();
    const double tol = opt->tol > 0 ? opt->tol : 1e-8;
    const int maxit = opt->maxit > 0 ? opt->maxit : 200, verbose = opt->verbose;
    memset(info, 0, sizeof(*info));
    LpS S_; LpS *S = &S_; memset(S, 0, sizeof(*S));
    S->m = m; S->n = n;
    const int nz = Ap0[n];
    S->Ap = xm(sizeof(int) * (n + 1)); S->Ai = xm(sizeof(int) * (nz + 1)); S->Ax = xm(sizeof(double) * (nz + 1));
    memcpy(S->Ap, Ap0, sizeof(int) * (n + 1)); memcpy(S->Ai, Ai0, sizeof(int) * nz); memcpy(S->Ax, Ax0, sizeof(double) * nz);
    /* rows ascending within a column */
    for (int j = 0; j < n; j++) {
        const int a = S->Ap[j], e = S->Ap[j + 1];
        for (int p = a + 1; p < e; p++) { const int r = S->Ai[p]; const double v = S->Ax[p]; int q = p - 1; while (q >= a && S->Ai[q] > r) { S->Ai[q + 1] = S->Ai[q]; S->Ax[q + 1] = S->Ax[q]; q--; } S->Ai[q + 1] = r; S->Ax[q + 1] = v; }
    }
    const double tph0 = lp_time();
    /* ---- Ruiz equilibration */
    S->rs = xm(sizeof(double) * (m + 1)); S->cs = xm(sizeof(double) * (n + 1));
    for (int i = 0; i < m; i++) S->rs[i] = 1.0;
    for (int j = 0; j < n; j++) S->cs[j] = 1.0;
    { double *rmax = xm(sizeof(double) * (m + 1));
      for (int pass = 0; pass < 10; pass++) {
          for (int i = 0; i < m; i++) rmax[i] = 0;
          double dev = 0;
          for (int j = 0; j < n; j++) {
              double cmx = 0;
              for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > cmx) cmx = v; if (v > rmax[S->Ai[p]]) rmax[S->Ai[p]] = v; }
              if (cmx > 0) { const double f = 1.0 / sqrt(cmx); dev = fmax(dev, fabs(1.0 - cmx)); S->cs[j] *= f; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= f; }
          }
          for (int i = 0; i < m; i++) rmax[i] = 0;
          for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > rmax[S->Ai[p]]) rmax[S->Ai[p]] = v; }
          for (int i = 0; i < m; i++) if (rmax[i] > 0) { dev = fmax(dev, fabs(1.0 - rmax[i])); rmax[i] = 1.0 / sqrt(rmax[i]); S->rs[i] *= rmax[i]; } else rmax[i] = 1.0;
          for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= rmax[S->Ai[p]];
          if (dev < 1e-2) break;
      }
      free(rmax); }
    S->b = xm(sizeof(double) * (m + 1)); S->c = xm(sizeof(double) * (n + 1)); S->u = xm(sizeof(double) * (n + 1)); S->bd = xz(n + 1); S->fr = xz(n + 1);
    for (int j = 0; j < nfree && j < n; j++) S->fr[j] = 1;
    for (int i = 0; i < m; i++) S->b[i] = b0[i] * S->rs[i];
    int nbd = 0;
    for (int j = 0; j < n; j++) { S->c[j] = c0[j] * S->cs[j]; S->bd[j] = !S->fr[j] && u0 && u0[j] < 1e29; S->u[j] = S->bd[j] ? u0[j] / S->cs[j] : 0.0; nbd += S->bd[j]; }
    /* scalar scaling of b (with u) and c: the iterates are then O(1) */
    double bsc = fmax(nrminf(m, S->b), 1e-300), csc = fmax(nrminf(n, S->c), 1e-300);
    { double um = 0; for (int j = 0; j < n; j++) if (S->bd[j] && S->u[j] > um) um = S->u[j]; if (um > bsc) bsc = um; }
    if (!(bsc > 1e-12)) bsc = 1.0;
    if (!(csc > 1e-12)) csc = 1.0;
    for (int i = 0; i < m; i++) S->b[i] /= bsc;
    for (int j = 0; j < n; j++) { S->c[j] /= csc; S->u[j] /= bsc; }
    const double tph1 = lp_time();
    /* the normal equations (free variables enter with the weight of a proximal term) */
    if (ne_setup(S, getenv("BRISK_LPNODENSE") ? 0 : 1)) { lps_free(S); info->status = 4; snprintf(info->why, sizeof info->why, "the factor of the normal equations does not fit"); return 4; }
    SChol *F = S->sc;
    info->lnz = (long)schol_nnz(F); info->ndense = S->nd;
    S->theta = xm(sizeof(double) * (n + 1));
    S->w1 = xm(sizeof(double) * (m + 1)); S->w2 = xm(sizeof(double) * (m + 1)); S->w4 = xm(sizeof(double) * (m + 1)); S->w3 = xm(sizeof(double) * (2 * (size_t)m + 2)); S->wn = xm(sizeof(double) * (n + 1)); S->acc = xz(sizeof(double) * (m + 1));
    S->pr = xm(sizeof(double) * (m + 1)); S->pz = xm(sizeof(double) * (m + 1)); S->pp = xm(sizeof(double) * (m + 1)); S->pq = xm(sizeof(double) * (m + 1));
    S->reg = getenv("BRISK_LPREG") ? atof(getenv("BRISK_LPREG")) : 1e-14;
    schol_set_solve_seq(1);
    /* the number of centrality correctors: from the cost of a factorization against a solve */
    int kcorr;
    { const double ff = schol_flops(F) + (double)S->nd * 2.0 * (double)schol_nnz(F), fs = 4.0 * (double)schol_nnz(F) + 4.0 * nz;
      const double ratio = ff / fmax(fs, 1.0);
      double t3 = 50, t2 = 30, t1 = 10;
      if (getenv("BRISK_LPCORRT")) sscanf(getenv("BRISK_LPCORRT"), "%lf,%lf,%lf", &t3, &t2, &t1);
      kcorr = ratio > t3 ? 3 : ratio > t2 ? 2 : ratio > t1 ? 1 : 0;
      if (opt->kcorr >= 0) kcorr = opt->kcorr; }
    if (verbose > 0)
        printf("BRISK LP interior-point solver: %d rows, %d columns (%d with upper bounds), %d nonzeros; normal equations: nnz(L) = %ld, %.2e flops, %d dense column%s; %d corrector%s\n",
               m, n, nbd, nz, info->lnz, schol_flops(F), S->nd, S->nd == 1 ? "" : "s", kcorr, kcorr == 1 ? "" : "s");
    if (verbose > 0) printf("Number of threads: 1\n");   /* (this solver is sequential) */
    double *x = xz(sizeof(double) * (n + 1)), *s = xz(sizeof(double) * (n + 1)), *z = xz(sizeof(double) * (n + 1)), *w = xz(sizeof(double) * (n + 1)), *y = xz(sizeof(double) * (m + 1));
    double *dx = xz(sizeof(double) * (n + 1)), *ds = xz(sizeof(double) * (n + 1)), *dz = xz(sizeof(double) * (n + 1)), *dw = xz(sizeof(double) * (n + 1)), *dy = xz(sizeof(double) * (m + 1));
    double *dx2 = xz(sizeof(double) * (n + 1)), *ds2 = xz(sizeof(double) * (n + 1)), *dz2 = xz(sizeof(double) * (n + 1)), *dw2 = xz(sizeof(double) * (n + 1)), *dy2 = xz(sizeof(double) * (m + 1));
    double *rb = xm(sizeof(double) * (m + 1)), *ru = xz(sizeof(double) * (n + 1)), *rc = xm(sizeof(double) * (n + 1)), *rxz = xm(sizeof(double) * (n + 1)), *rsw = xz(sizeof(double) * (n + 1));
    double *g = xm(sizeof(double) * (n + 1)), *zr = xz(sizeof(double) * (n + 1)), *zm = xz(sizeof(double) * (m + 1));
    int status = 3, it = 0;
    const double tph2 = lp_time(); double tph3 = tph2;
    /* ---- starting point (Mehrotra's, with the bounds): least-norm x and least-squares y */
    for (int j = 0; j < n; j++) S->theta[j] = S->bd[j] ? 0.5 : 1.0;      /* (free columns: 1, the least-norm start) */
    if (ne_factor(S)) { status = 4; snprintf(info->why, sizeof info->why, "the first factorization failed"); goto done; }
    {
        for (int j = 0; j < n; j++) x[j] = S->bd[j] ? 0.5 * S->u[j] : 0.0;
        A_mul(S, x, rb);
        for (int i = 0; i < m; i++) rb[i] = S->b[i] - rb[i];
        kkt_solve(S, zr, rb, dx, dy, 1e-10, 20);                 /* the least-norm correction: A dx = b - A x */
        for (int j = 0; j < n; j++) { x[j] += dx[j]; s[j] = S->bd[j] ? S->u[j] - x[j] : 0.0; }
        kkt_solve(S, S->c, zm, dx, y, 1e-10, 20);                /* A Theta (A'y - c) = 0 */
        for (int j = 0; j < n; j++) { const double r = -dx[j] / S->theta[j]; if (S->fr[j]) { z[j] = 0; w[j] = 0; } else if (S->bd[j]) { z[j] = 0.5 * r; w[j] = -0.5 * r; } else { z[j] = r; w[j] = 0; } }
        double xmin = 1e300, zmin = 1e300;
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; xmin = fmin(xmin, x[j]); zmin = fmin(zmin, z[j]); if (S->bd[j]) { xmin = fmin(xmin, s[j]); zmin = fmin(zmin, w[j]); } }
        const double dxs = fmax(-1.5 * xmin, 0.0), dzs = fmax(-1.5 * zmin, 0.0);
        double xz_ = 0, sx = 0, sz = 0;
        for (int j = 0; j < n; j++) {
            if (S->fr[j]) continue;
            x[j] += dxs; z[j] += dzs; xz_ += x[j] * z[j]; sx += x[j]; sz += z[j];
            if (S->bd[j]) { s[j] += dxs; w[j] += dzs; xz_ += s[j] * w[j]; sx += s[j]; sz += w[j]; }
        }
        double dx3 = sz > 0 ? 0.5 * xz_ / sz : 0.0, dz3 = sx > 0 ? 0.5 * xz_ / sx : 0.0;
        if (!(dx3 > 1e-8)) dx3 = fmax(dx3, 1e-2);         /* (a start at zero: push inside) */
        if (!(dz3 > 1e-8)) dz3 = fmax(dz3, 1e-2);
        if (getenv("BRISK_LPSTART")) { const double f = atof(getenv("BRISK_LPSTART")); dx3 *= f; dz3 *= f; }
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; x[j] += dx3; z[j] += dz3; if (S->bd[j]) { s[j] += dx3; w[j] += dz3; } }
    }
    const double nb0 = nrminf(m, b0), nc0 = nrminf(n, c0);
    double mu0 = -1, best = 1e300; int nstall = 0, nsafe = 0;
    const int steprule = getenv("BRISK_LPSTEP") ? atoi(getenv("BRISK_LPSTEP")) : 1;
    const double frth = getenv("BRISK_LPFREETH") ? atof(getenv("BRISK_LPFREETH")) : 1e6;
    if (verbose > 1) printf("  it      pobj             dobj         pinf     dinf     gap      mu      alp_p alp_d sigma  cg  solres tiny\n");
    tph3 = lp_time();
    for (it = 0; ; it++) {
        /* ---- residuals (scaled problem) and the errors of the problem as given */
        A_mul(S, x, rb);
        for (int i = 0; i < m; i++) rb[i] = S->b[i] - rb[i];
        At_mul(S, y, rc);
        double mu = 0; int ncomp = 0;
        for (int j = 0; j < n; j++) {
            rc[j] = S->c[j] - rc[j] - z[j] + w[j];
            if (S->fr[j]) continue;
            mu += x[j] * z[j]; ncomp++;
            if (S->bd[j]) { ru[j] = S->u[j] - x[j] - s[j]; mu += s[j] * w[j]; ncomp++; }
        }
        mu /= fmax(ncomp, 1);
        if (mu0 < 0) mu0 = mu;
        double ep = 0, ed = 0, pobj = 0, dobj = 0;
        for (int i = 0; i < m; i++) { const double v = fabs(rb[i]) * bsc / S->rs[i]; if (v > ep) ep = v; dobj += S->b[i] * y[i]; }
        for (int j = 0; j < n; j++) {
            const double v = fabs(rc[j]) * csc / S->cs[j]; if (v > ed) ed = v;
            pobj += S->c[j] * x[j];
            if (S->bd[j]) { const double q = fabs(ru[j]) * bsc * S->cs[j]; if (q > ep) ep = q; dobj -= S->u[j] * w[j]; }
        }
        pobj *= bsc * csc; dobj *= bsc * csc;
        const double e_p = ep / (1.0 + nb0), e_d = ed / (1.0 + nc0), e_g = fabs(pobj - dobj) / (1.0 + fabs(pobj));
        const double err = fmax(fmax(e_p, e_d), e_g);
        info->pobj = pobj; info->dobj = dobj; info->pinf = e_p; info->dinf = e_d; info->gap = e_g; info->iters = it;
        if (verbose > 1 && it == 0) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e\n", it, pobj, dobj, e_p, e_d, e_g, mu);
        if (getenv("BRISK_LPDBG")) {
            int im = 0; double vm = 0; for (int i = 0; i < m; i++) { const double v = fabs(rb[i]) / S->rs[i]; if (v > vm) { vm = v; im = i; } }
            double sa = 0, xm_ = 0, ym = nrminf(m, y); for (int j = 0; j < n; j++) { xm_ = fmax(xm_, fabs(x[j])); for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (S->Ai[p] == im) sa += fabs(S->Ax[p] * x[j]); }
            printf("     [dbg: worst row %d: rb %.2e (scaled; max scaled %.2e), rs %.2e, sum|a x| %.2e, b %.2e; max|x| %.2e max|y| %.2e]\n", im, rb[im], nrminf(m, rb), S->rs[im], sa, S->b[im], xm_, ym);
        }
        if (!isfinite(pobj) || !isfinite(dobj) || !isfinite(mu) || !isfinite(ep) || !isfinite(ed)) { status = 4; snprintf(info->why, sizeof info->why, "not finite"); break; }
        if (err <= tol) { status = 0; break; }
        if (it >= maxit) { status = 3; snprintf(info->why, sizeof info->why, "iteration limit"); break; }
        if (opt->timelimit > 0 && lp_time() - t0 > opt->timelimit) { status = 6; break; }
        if (opt->stop && *opt->stop) { status = 6; break; }
        /* divergence or stall: the caller's homogeneous model decides (infeasible or unbounded problems) */
        if (err < 0.9 * best) { best = err; nstall = 0; } else nstall++;
        if (!isfinite(err) || !isfinite(mu)) { status = 4; snprintf(info->why, sizeof info->why, "not finite"); break; }
        if (nstall >= 25 || mu > 1e12 * fmax(mu0, 1.0) || nrminf(n, x) > 1e14 || nrminf(m, y) > 1e14) { status = 4; snprintf(info->why, sizeof info->why, "%s", nstall >= 25 ? "no progress in 25 iterations" : "the iterates diverge"); break; }
        /* ---- factorization */
        double thmax = 0;
        for (int j = 0; j < n; j++) if (!S->fr[j]) { S->theta[j] = 1.0 / (z[j] / x[j] + (S->bd[j] ? w[j] / s[j] : 0.0)); if (S->theta[j] > thmax) thmax = S->theta[j]; }
        /* free variables: a proximal term rho |dx|^2 / 2, Theta = 1 / rho (the Newton step of the regularized problem) */
        { const double thf = getenv("BRISK_LPFREEREL") ? fmin(fmax(frth, atof(getenv("BRISK_LPFREEREL")) * thmax), 1e12) : frth; for (int j = 0; j < nfree; j++) S->theta[j] = thf; }
        if (ne_factor(S)) { status = 4; snprintf(info->why, sizeof info->why, "factorization failed"); break; }
        const double rtol = 1e-10; const int maxcg = getenv("BRISK_LPCG") ? atoi(getenv("BRISK_LPCG")) : 30;
        const long cg0 = S->ncg;
        /* ---- predictor */
        for (int j = 0; j < n; j++) { rxz[j] = S->fr[j] ? 0.0 : -x[j] * z[j]; rsw[j] = S->bd[j] ? -s[j] * w[j] : 0.0; }
        S->worst = 0;
        direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        if (S->nd > 0 && S->worst > 1e-6) {
            /* the Woodbury correction of the dense columns has lost its accuracy: all columns in the factor */
            if (verbose > 1) printf("     (dense columns: residual %.1e, they join the factored matrix)\n", S->worst);
            if (ne_setup(S, 0) || ne_factor(S)) { status = 4; snprintf(info->why, sizeof info->why, "the factor with the dense columns does not fit"); break; }
            info->ndense = 0; info->lnz = (long)schol_nnz(S->sc); S->worst = 0;
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        double ap, ad;
        step_max(S, x, s, z, w, dx, ds, dz, dw, &ap, &ad);
        ap = fmin(1.0, ap); ad = fmin(1.0, ad);
        double mua = 0;
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; mua += (x[j] + ap * dx[j]) * (z[j] + ad * dz[j]); if (S->bd[j]) mua += (s[j] + ap * ds[j]) * (w[j] + ad * dw[j]); }
        mua /= fmax(ncomp, 1);
        double sigma = pow(mua / mu, 3.0);
        if (sigma > 1.0) sigma = 1.0;
        if (sigma < 1e-6) sigma = 1e-6;
        /* ---- corrector */
        for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu - x[j] * z[j] - dx[j] * dz[j]; rsw[j] = S->bd[j] ? sigma * mu - s[j] * w[j] - ds[j] * dw[j] : 0.0; }
        direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        step_max(S, x, s, z, w, dx, ds, dz, dw, &ap, &ad);
        /* a blocked step: the second-order term was wrong for this point - a centering step instead */
        if (fmin(ap, ad) < 0.1 && !getenv("BRISK_LPNOSAFE")) {
            sigma = fmax(sigma, 0.5);
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu - x[j] * z[j]; rsw[j] = S->bd[j] ? sigma * mu - s[j] * w[j] : 0.0; }
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            double bp, bdl;
            step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &bp, &bdl);
            if (fmin(bp, bdl) > fmin(ap, ad)) {
                memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                ap = bp; ad = bdl; nsafe++;
            }
        }
        /* ---- Gondzio's centrality correctors */
        for (int k = 0; k < kcorr; k++) {
            const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
            if (a1 >= 0.999 && a2 >= 0.999) break;
            const double tp = fmin(1.0, 1.08 * a1 + 0.08), td = fmin(1.0, 1.08 * a2 + 0.08);
            const double mt = sigma * mu, lo = 0.1 * mt, hi = 10.0 * mt;
            for (int j = 0; j < n; j++) {
                if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; }
                double v = (x[j] + tp * dx[j]) * (z[j] + td * dz[j]);
                double t = v < lo ? lo - v : v > hi ? hi - v : 0.0;
                if (t < -hi) t = -hi;
                rxz[j] = t;
                if (S->bd[j]) { v = (s[j] + tp * ds[j]) * (w[j] + td * dw[j]); t = v < lo ? lo - v : v > hi ? hi - v : 0.0; if (t < -hi) t = -hi; rsw[j] = t; } else rsw[j] = 0;
            }
            direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            for (int j = 0; j < n; j++) { dx2[j] += dx[j]; dz2[j] += dz[j]; ds2[j] += ds[j]; dw2[j] += dw[j]; }
            for (int i = 0; i < m; i++) dy2[i] += dy[i];
            double bp, bdl;
            step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &bp, &bdl);
            if (fmin(1.0, bp) >= a1 + 0.01 * 0.08 || fmin(1.0, bdl) >= a2 + 0.01 * 0.08) {
                if (!(fmin(1.0, bp) + fmin(1.0, bdl) > a1 + a2)) break;
                memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                ap = bp; ad = bdl; info->ncorr++;
            } else break;
        }
        /* ---- step lengths. Mehrotra's rule: the blocking component keeps a complementarity product of
         * gamma_f times the mean after the full step, and the step is at least 0.9 of the way to the boundary */
        if (steprule == 1) {
            const double gf = 0.01, ga = 0.1;
            const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
            double muf = 0;
            for (int j = 0; j < n; j++) { if (S->fr[j]) continue; muf += (x[j] + a1 * dx[j]) * (z[j] + a2 * dz[j]); if (S->bd[j]) muf += (s[j] + a1 * ds[j]) * (w[j] + a2 * dw[j]); }
            muf = fmax(muf, 0.0) / fmax(ncomp, 1);
            /* the blocking components */
            int kp = -1, kd = -1, tp_ = 0, td_ = 0; double bp_ = 1e300, bd_ = 1e300;
            for (int j = 0; j < n; j++) {
                if (S->fr[j]) continue;
                if (dx[j] < 0) { const double t = -x[j] / dx[j]; if (t < bp_) { bp_ = t; kp = j; tp_ = 0; } }
                if (dz[j] < 0) { const double t = -z[j] / dz[j]; if (t < bd_) { bd_ = t; kd = j; td_ = 0; } }
                if (S->bd[j]) {
                    if (ds[j] < 0) { const double t = -s[j] / ds[j]; if (t < bp_) { bp_ = t; kp = j; tp_ = 1; } }
                    if (dw[j] < 0) { const double t = -w[j] / dw[j]; if (t < bd_) { bd_ = t; kd = j; td_ = 1; } }
                }
            }
            double fp = 1.0, fd = 1.0;
            if (kp >= 0 && ap < 1.0) {
                const double xk = tp_ ? s[kp] : x[kp], dxk = tp_ ? ds[kp] : dx[kp], zk = tp_ ? w[kp] + a2 * dw[kp] : z[kp] + a2 * dz[kp];
                fp = zk > 0 ? (gf * muf / zk - xk) / (ap * dxk) : 1.0 - ga;
                fp = fmin(fmax(fp, 1.0 - ga), 0.999999);
            }
            if (kd >= 0 && ad < 1.0) {
                const double zk = td_ ? w[kd] : z[kd], dzk = td_ ? dw[kd] : dz[kd], xk = td_ ? s[kd] + a1 * ds[kd] : x[kd] + a1 * dx[kd];
                fd = xk > 0 ? (gf * muf / xk - zk) / (ad * dzk) : 1.0 - ga;
                fd = fmin(fmax(fd, 1.0 - ga), 0.999999);
            }
            ap = fmin(1.0, fp * ap); ad = fmin(1.0, fd * ad);
        } else {
            const double eta = fmax(0.9, fmin(0.99995, 1.0 - 0.1 * mu / fmax(mu0, 1e-300)));
            const double etaf = steprule == 3 ? 0.99 : steprule == 2 ? eta : mu < 1e-6 * mu0 ? 0.9999 : fmax(eta, 0.995);
            ap = fmin(1.0, etaf * ap); ad = fmin(1.0, etaf * ad);
        }
        if (steprule == 3) { ap = ad = fmin(ap, ad); }
        if (nfree == n) { ap = 1.0; ad = 1.0; }
        for (int j = 0; j < n; j++) { x[j] += ap * dx[j]; z[j] += ad * dz[j]; if (S->bd[j]) { s[j] += ap * ds[j]; w[j] += ad * dw[j]; } }
        for (int i = 0; i < m; i++) y[i] += ad * dy[i];
        if (verbose > 1) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %5.3f %5.3f %6.0e %3ld %7.0e %d\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, ap, ad, sigma, S->ncg - cg0, S->worst, schol_ntiny(S->sc));
    }
    (void)nsafe;
done:
    if (getenv("BRISK_LPDBG")) printf("   [LP phases: copy %.3fs, scaling %.3fs, pattern and analysis %.3fs, start %.3fs, iterations %.3fs (factorizations %.3fs, solves %.3fs)]\n", tph0 - t0, tph1 - tph0, tph2 - tph1, tph3 - tph2, lp_time() - tph3, S->t_fact, S->t_solve);
    info->status = status; info->iters = it; info->time = lp_time() - t0;
    info->nfact = S->nfact; info->nsolve = S->nsolve; info->ncg = S->ncg; info->ntiny = S->ntiny; info->t_fact = S->t_fact; info->t_solve = S->t_solve;
    /* the solution of the problem as given */
    for (int j = 0; j < n; j++) { xout[j] = x[j] * S->cs[j] * bsc; if (zout) zout[j] = (z[j] - w[j]) * csc / S->cs[j]; }
    for (int i = 0; i < m; i++) yout[i] = y[i] * S->rs[i] * csc;
    schol_set_solve_seq(0);
    free(x); free(s); free(z); free(w); free(y); free(dx); free(ds); free(dz); free(dw); free(dy); free(dx2); free(ds2); free(dz2); free(dw2); free(dy2);
    free(rb); free(ru); free(rc); free(rxz); free(rsw); free(g); free(zr); free(zm);
    lps_free(S);
    return status;
}
