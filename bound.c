/* bound.c - BRISK 4.37: the one-sided bound mode (-bound p|d).
 *
 * BRISK's (P) is min <C,X> s.t. <A_i,X> = b_i, X in K and (D) max b'y s.t. C - sum y_i A_i
 * = Z in K. Any X feasible for (P) gives the upper bound <C,X> on the optimal value, any y
 * feasible for (D) the lower bound b'y, whatever happened to the other side. With -bound p
 * (resp. d) the solver keeps the best candidate of that side during the solve (solver.c,
 * BoundTrack), and here the candidates, mapped back to the problem as read, are turned into
 * certificates:
 *
 *   p: X is projected onto {A(X) = b} (X += A*(w), (A A*) w = b - A(X), two to four passes),
 *      then made positive definite with a margin: blended with an interior anchor (an
 *      earlier, more central iterate, projected the same way: every blend satisfies the
 *      equations), or moved along the null-space projection of the identity when that is
 *      positive definite;
 *   d: Z = C - A'y is recomputed from y (the equations hold by construction); a negative
 *      lambda_min(Z) is repaired by blending with the anchor's y or along y - t e, where
 *      A'e is the projection of the identity onto the range of A' (max-cut: A'e = I).
 *
 * The margin (per block: boundmargin * (1 + max diagonal)) is what lets a rigorous check
 * (-certify, boundcert.c) round the certificate without losing definiteness.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "brisk.h"

static size_t bsz_o(int bs) { return bs < 0 ? (size_t)(-bs) : (size_t)bs * (size_t)bs; }

double **bound_alloc_blocks(const PSOrig *O) {
    double **X = calloc(O->nblk + 1, sizeof(double *));
    for (int k = 0; k < O->nblk; k++) X[k] = calloc(bsz_o(O->bs[k]) + 1, sizeof(double));
    return X;
}
void bound_free_blocks(double **X, int nblk) {
    if (!X) return;
    for (int k = 0; k < nblk; k++) free(X[k]);
    free(X);
}
static void copy_blocks(const PSOrig *O, double **D, double **S) {
    for (int k = 0; k < O->nblk; k++) memcpy(D[k], S[k], sizeof(double) * bsz_o(O->bs[k]));
}

/* out_i = <A_i, X>; returns <C, X> */
static double opA(const PSOrig *O, double **X, double *out) {
    double cx = 0;
    memset(out, 0, sizeof(double) * (O->m + 1));
    for (size_t q = 0; q < O->nnz; q++) {
        const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
        const double x = O->bs[k] < 0 ? X[k][i] : X[k][i + (size_t)j * n];
        const double c = (O->bs[k] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
        if (O->con[q] < 0) cx += c; else out[O->con[q]] += c;
    }
    return cx;
}
/* X += alpha sum_i w_i A_i (+ beta C) */
static void opAT(const PSOrig *O, const double *w, double alpha, double beta, double **X) {
    for (size_t q = 0; q < O->nnz; q++) {
        const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
        const double c = O->con[q] < 0 ? beta * O->v[q] : alpha * w[O->con[q]] * O->v[q];
        if (c == 0) continue;
        if (O->bs[k] < 0) X[k][i] += c;
        else { X[k][i + (size_t)j * n] += c; if (i != j) X[k][j + (size_t)i * n] += c; }
    }
}

/* ---- the Gram matrix G = A A* (dense Cholesky when it fits, else PCG) ------------------ */
typedef struct { int m, dense; double *L, *diag; double ridge; } Gram0;

static const PSOrig *g_sortO;
static int cmp_ent(const void *a, const void *b) {
    const size_t p = *(const size_t *)a, q = *(const size_t *)b;
    const PSOrig *O = g_sortO;
    if (O->blk[p] != O->blk[q]) return O->blk[p] < O->blk[q] ? -1 : 1;
    if (O->jj[p] != O->jj[q]) return O->jj[p] < O->jj[q] ? -1 : 1;
    if (O->ii[p] != O->ii[q]) return O->ii[p] < O->ii[q] ? -1 : 1;
    return 0;
}

static void gram_init(const PSOrig *O, Gram0 *G, int verbose) {
    memset(G, 0, sizeof(*G));
    const int m = O->m;
    G->m = m;
    G->diag = calloc(m + 1, sizeof(double));
    for (size_t q = 0; q < O->nnz; q++) {
        if (O->con[q] < 0) continue;
        const int k = O->blk[q];
        const double w = (O->bs[k] > 0 && O->ii[q] != O->jj[q]) ? 2.0 : 1.0;
        G->diag[O->con[q]] += w * O->v[q] * O->v[q];
    }
    const double ram = brisk_mem_limit();
    if (!(m > 0 && m <= 8000 && 8.0 * (double)m * m < 0.2 * ram)) return;
    /* entries grouped by position: G_ab += w v_a v_b over the constraints sharing an entry */
    size_t na = 0;
    size_t *idx = malloc(sizeof(size_t) * (O->nnz + 1));
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) idx[na++] = q;
    g_sortO = O;
    qsort(idx, na, sizeof(size_t), cmp_ent);
    double work = 0;
    for (size_t s = 0; s < na;) {
        size_t e = s + 1;
        while (e < na && cmp_ent(&idx[s], &idx[e]) == 0) e++;
        work += (double)(e - s) * (double)(e - s);
        s = e;
    }
    if (work > 2e9) { free(idx); return; }
    double *L = calloc((size_t)m * m, sizeof(double));
    for (size_t s = 0; s < na;) {
        size_t e = s + 1;
        while (e < na && cmp_ent(&idx[s], &idx[e]) == 0) e++;
        const int k = O->blk[idx[s]];
        const double w = (O->bs[k] > 0 && O->ii[idx[s]] != O->jj[idx[s]]) ? 2.0 : 1.0;
        for (size_t a = s; a < e; a++)
            for (size_t b = s; b < e; b++) {
                const int ca = O->con[idx[a]], cb = O->con[idx[b]];
                if (ca >= cb) L[ca + (size_t)cb * m] += w * O->v[idx[a]] * O->v[idx[b]];
            }
        s = e;
    }
    free(idx);
    double dmax = 0;
    for (int i = 0; i < m; i++) dmax = fmax(dmax, L[i + (size_t)i * m]);
    double *F = malloc(sizeof(double) * (size_t)m * m);
    int info = 1;
    for (int t = 0; t < 6 && info != 0; t++) {
        const double ridge = t == 0 ? 0.0 : dmax * pow(10.0, -15.0 + 2.0 * t);
        memcpy(F, L, sizeof(double) * (size_t)m * m);
        for (int i = 0; i < m; i++) F[i + (size_t)i * m] += ridge + (F[i + (size_t)i * m] == 0 ? 1.0 : 0.0);
        BL(dpotrf_)("L", &m, F, &m, &info);
        G->ridge = ridge;
    }
    free(L);
    if (info != 0) { free(F); return; }
    G->L = F;
    G->dense = 1;
    if (verbose > 1) printf("   bound: Gram matrix m = %d factorized (ridge %.1e)\n", m, G->ridge);
}
static void gram_free(Gram0 *G) { free(G->L); free(G->diag); memset(G, 0, sizeof(*G)); }

/* 4.42: the Gram factor is kept between the calls of one problem (one call per candidate
 * point), and the dual side builds it only when it is needed (identity_dir). On
 * nn_n10_m100_K (m = 7 810 as read) every call built and factored the 7 810 x 7 810 matrix:
 * 238 s of certificate work after a 13 s solve. ps_orig_free drops the cache. */
double g_bound_deadline = 0;      /* 4.42: wall-clock time after which the costly dual-side repairs are skipped (0: none) */
/* only where the repair is costly (m above 2000: seconds per factorization); below, it is always done */
static int bound_late(const PSOrig *O) { return O->m > 2000 && g_bound_deadline > 0 && wtime() > g_bound_deadline; }
static struct { const PSOrig *O; int m; size_t nnz; Gram0 G; } g_gc;
void bound_gram_drop(const PSOrig *O) {
    if (g_gc.O && (!O || g_gc.O == O)) { gram_free(&g_gc.G); g_gc.O = NULL; }
}
static const Gram0 *gram_get(const PSOrig *O, int verbose) {
    if (g_gc.O == O && g_gc.m == O->m && g_gc.nnz == O->nnz) return &g_gc.G;
    bound_gram_drop(NULL);
    gram_init(O, &g_gc.G, verbose);
    g_gc.O = O; g_gc.m = O->m; g_gc.nnz = O->nnz;
    return &g_gc.G;
}

/* w = G^{-1} r (approximately: the projection passes refine it) */
static void gram_solve(const PSOrig *O, const Gram0 *G, const double *r, double *w) {
    const int m = G->m, one = 1;
    if (G->dense) {
        memcpy(w, r, sizeof(double) * m);
        int info;
        BL(dpotrs_)("L", &m, &one, G->L, &m, w, &m, &info);
        return;
    }
    /* PCG (Jacobi) on G w = r, matrix-free: G v = A(A*(v)) */
    double **T = bound_alloc_blocks(O);
    double *res = malloc(sizeof(double) * (m + 1)), *z = malloc(sizeof(double) * (m + 1));
    double *p = malloc(sizeof(double) * (m + 1)), *Gp = malloc(sizeof(double) * (m + 1));
    memset(w, 0, sizeof(double) * m);
    memcpy(res, r, sizeof(double) * m);
    double r0 = 0;
    for (int i = 0; i < m; i++) r0 += r[i] * r[i];
    r0 = sqrt(r0);
    double rz = 0;
    for (int i = 0; i < m; i++) { z[i] = G->diag[i] > 0 ? res[i] / G->diag[i] : res[i]; rz += res[i] * z[i]; }
    memcpy(p, z, sizeof(double) * m);
    for (int it = 0; it < 500 && r0 > 0; it++) {
        for (int k = 0; k < O->nblk; k++) memset(T[k], 0, sizeof(double) * bsz_o(O->bs[k]));
        opAT(O, p, 1.0, 0.0, T);
        opA(O, T, Gp);
        double pGp = 0;
        for (int i = 0; i < m; i++) pGp += p[i] * Gp[i];
        if (!(pGp > 0)) break;
        const double a = rz / pGp;
        double rn = 0;
        for (int i = 0; i < m; i++) { w[i] += a * p[i]; res[i] -= a * Gp[i]; rn += res[i] * res[i]; }
        if (sqrt(rn) < 1e-15 * r0) break;
        double rz2 = 0;
        for (int i = 0; i < m; i++) { z[i] = G->diag[i] > 0 ? res[i] / G->diag[i] : res[i]; rz2 += res[i] * z[i]; }
        const double bta = rz2 / rz;
        rz = rz2;
        for (int i = 0; i < m; i++) p[i] = z[i] + bta * p[i];
    }
    free(res); free(z); free(p); free(Gp);
    bound_free_blocks(T, O->nblk);
}

/* 5.2 (hpsolve.c): w = G^{-1} r with the cached factor of G = A A* (dense with a ridge, or
 * PCG): the projection onto A(X) = b of the high-precision certificates is refined with it */
void bound_gram_apply(const PSOrig *O, const double *r, double *w) { gram_solve(O, gram_get(O, 0), r, w); }

static double norm1_b(const PSOrig *O) { double s = 0; for (int i = 0; i < O->m; i++) s += fabs(O->b[i]); return s; }

/* relative primal residual ||A(X) - b|| / (1 + ||b||_1) */
static double resid_p(const PSOrig *O, double **X, double *tmp) {
    opA(O, X, tmp);
    double r2 = 0;
    for (int i = 0; i < O->m; i++) { const double v = tmp[i] - O->b[i]; r2 += v * v; }
    return sqrt(r2) / (1 + norm1_b(O));
}

/* passes of X += A*(G^{-1}(b - A(X))) */
static double project_p(const PSOrig *O, const Gram0 *G, double **X) {
    const int m = O->m;
    double *r = malloc(sizeof(double) * (m + 1)), *w = malloc(sizeof(double) * (m + 1));
    double prev = INFINITY, res = 0;
    for (int pass = 0; pass < 4; pass++) {
        opA(O, X, r);
        double r2 = 0;
        for (int i = 0; i < m; i++) { r[i] = O->b[i] - r[i]; r2 += r[i] * r[i]; }
        res = sqrt(r2) / (1 + norm1_b(O));
        if (res == 0 || res > 0.5 * prev) break;
        prev = res;
        gram_solve(O, G, r, w);
        opAT(O, w, 1.0, 0.0, X);
    }
    res = resid_p(O, X, r);
    free(r); free(w);
    return res;
}

/* The step in the geometry of X (the Newton step for A(X) = b of the log-barrier):
 * X += X A*(w) X with M w = b - A(X), M_ij = <A_i, X A_j X>. A(X) is then b up to
 * rounding, and X stays positive definite when ||X^1/2 A*(w) X^1/2|| < 1: unlike the
 * Euclidean projection, the correction is measured against X's own eigenvalues (the
 * iterates of an infeasible interior-point method reach feasibility at the rate of mu,
 * truss4 / qap5: the Euclidean correction then exceeds lambda_min(X)). Returns 0 when not
 * done (too expensive, X not positive definite, M singular). */
static int scaled_step(const PSOrig *O, double **X, int verbose) {
    const int m = O->m;
    if (m <= 0) return 0;
    double cost = 0;
    size_t *cnt = calloc(m + 1, sizeof(size_t));
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) cnt[O->con[q]]++;
    for (size_t q = 0; q < O->nnz; q++)
        if (O->con[q] >= 0 && O->bs[O->blk[q]] > 0) cost += 2.0 * (double)O->bs[O->blk[q]] * O->bs[O->blk[q]];
    cost += (double)m * (double)O->nnz;
    free(cnt);
    if (cost > 4e10 || 8.0 * (double)m * m > 0.2 * brisk_mem_limit()) return 0;
    for (int k = 0; k < O->nblk; k++)       /* X must be positive definite */
        if (O->bs[k] > 0) {
            const int n = O->bs[k]; int info;
            double *W = malloc(sizeof(double) * (size_t)n * n);
            memcpy(W, X[k], sizeof(double) * (size_t)n * n);
            BL(dpotrf_)("L", &n, W, &n, &info);
            free(W);
            if (info != 0) return 0;
        } else for (int i = 0; i < -O->bs[k]; i++) if (!(X[k][i] > 0)) return 0;
    /* constraint -> its entries */
    size_t *ptr = calloc(m + 2, sizeof(size_t)), *ix;
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) ptr[O->con[q] + 1]++;
    for (int i = 0; i < m; i++) ptr[i + 1] += ptr[i];
    ix = malloc(sizeof(size_t) * (ptr[m] + 1));
    size_t *fill = calloc(m + 1, sizeof(size_t));
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) { const int i = O->con[q]; ix[ptr[i] + fill[i]++] = q; }
    free(fill);
    double **P = bound_alloc_blocks(O);
    double *M = calloc((size_t)m * m, sizeof(double)), *col = malloc(sizeof(double) * (m + 1));
    for (int j = 0; j < m; j++) {
        /* P = X A_j X, block by block, from the entries of A_j */
        for (int k = 0; k < O->nblk; k++) memset(P[k], 0, sizeof(double) * bsz_o(O->bs[k]));
        for (size_t a = ptr[j]; a < ptr[j + 1]; a++) {
            const size_t q = ix[a];
            const int k = O->blk[q], p = O->ii[q], r = O->jj[q];
            const double v = O->v[q];
            if (O->bs[k] < 0) { P[k][p] += v * X[k][p] * X[k][p]; continue; }
            const int n = O->bs[k];
            const double *xp = X[k] + (size_t)p * n, *xr = X[k] + (size_t)r * n;   /* columns p, r (X symmetric) */
            for (int c = 0; c < n; c++) {
                const double a1 = v * xr[c], a2 = p != r ? v * xp[c] : 0.0;
                double *Pc = P[k] + (size_t)c * n;
                for (int i2 = 0; i2 < n; i2++) Pc[i2] += xp[i2] * a1 + xr[i2] * a2;
            }
        }
        opA(O, P, col);
        for (int i = 0; i < m; i++) M[i + (size_t)j * m] = col[i];
    }
    free(ptr); free(ix); free(col);
    /* symmetrize, factor (ridge if needed) and solve M w = b - A(X) */
    for (int j = 0; j < m; j++) for (int i = j + 1; i < m; i++) { const double s = 0.5 * (M[i + (size_t)j * m] + M[j + (size_t)i * m]); M[i + (size_t)j * m] = M[j + (size_t)i * m] = s; }
    double *r = malloc(sizeof(double) * (m + 1)), *w = malloc(sizeof(double) * (m + 1));
    opA(O, X, r);
    for (int i = 0; i < m; i++) r[i] = O->b[i] - r[i];
    double dmax = 0;
    for (int i = 0; i < m; i++) dmax = fmax(dmax, M[i + (size_t)i * m]);
    int info = 1;
    double *F = malloc(sizeof(double) * (size_t)m * m);
    for (int t = 0; t < 4 && info != 0; t++) {
        memcpy(F, M, sizeof(double) * (size_t)m * m);
        const double ridge = t == 0 ? 0 : dmax * pow(10.0, -14.0 + 2.0 * t);
        for (int i = 0; i < m; i++) F[i + (size_t)i * m] += ridge;
        BL(dpotrf_)("L", &m, F, &m, &info);
    }
    int ok = 0;
    if (info == 0) {
        const int one = 1;
        memcpy(w, r, sizeof(double) * m);
        BL(dpotrs_)("L", &m, &one, F, &m, w, &m, &info);
        /* D = X S X with S = A*(w) */
        for (int k = 0; k < O->nblk; k++) memset(P[k], 0, sizeof(double) * bsz_o(O->bs[k]));
        opAT(O, w, 1.0, 0.0, P);
        for (int k = 0; k < O->nblk; k++) {
            if (O->bs[k] < 0) { for (int i = 0; i < -O->bs[k]; i++) X[k][i] += X[k][i] * P[k][i] * X[k][i]; continue; }
            const int n = O->bs[k];
            const double one_ = 1.0, zero = 0.0;
            double *T1 = malloc(sizeof(double) * (size_t)n * n), *T2 = malloc(sizeof(double) * (size_t)n * n);
            BL(dgemm_)("N", "N", &n, &n, &n, &one_, X[k], &n, P[k], &n, &zero, T1, &n);
            BL(dgemm_)("N", "N", &n, &n, &n, &one_, T1, &n, X[k], &n, &zero, T2, &n);
            for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) X[k][i + (size_t)j * n] += 0.5 * (T2[i + (size_t)j * n] + T2[j + (size_t)i * n]);
            free(T1); free(T2);
        }
        ok = 1;
        if (verbose > 1) printf("   bound: scaled (barrier-metric) projection step, m = %d\n", m);
    }
    free(F); free(M); free(r); free(w);
    bound_free_blocks(P, O->nblk);
    return ok;
}

/* ---- definiteness ----------------------------------------------------------------------- */
static double maxdiag(int bs, const double *M) {
    const int n = abs(bs);
    double d = 0;
    for (int i = 0; i < n; i++) d = fmax(d, fabs(bs < 0 ? M[i] : M[i + (size_t)i * n]));
    return d;
}
/* M - shift I positive definite (Cholesky) */
static int pd_shift(int bs, const double *M, double shift, double *work) {
    const int n = abs(bs);
    if (bs < 0) { for (int i = 0; i < n; i++) if (!(M[i] > shift)) return 0; return 1; }
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) work[i + (size_t)i * n] -= shift;
    int info;
    BL(dpotrf_)("L", &n, work, &n, &info);
    return info == 0;
}
/* the smallest eigenvalue (dsyevd on a copy; LP: the smallest entry) */
static double lam_min(int bs, const double *M, double *work) {
    const int n = abs(bs);
    if (n == 0) return 0;
    if (bs < 0) { double v = M[0]; for (int i = 1; i < n; i++) v = fmin(v, M[i]); return v; }
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    double *ev = malloc(sizeof(double) * (n + 1));
    int lw = -1, liw = -1, iwq = 0, info = 0;
    double wq = 0;
    BL(dsyevd_)("N", "L", &n, work, &n, ev, &wq, &lw, &iwq, &liw, &info);
    lw = (int)wq + 1; liw = iwq + 1;
    double *wk = malloc(sizeof(double) * lw);
    int *iw = malloc(sizeof(int) * liw);
    BL(dsyevd_)("N", "L", &n, work, &n, ev, wk, &lw, iw, &liw, &info);
    const double v = info == 0 ? ev[0] : -INFINITY;
    free(wk); free(iw); free(ev);
    return v;
}
static double *work_for(const PSOrig *O) {
    size_t mx = 1;
    for (int k = 0; k < O->nblk; k++) if (O->bs[k] > 0 && bsz_o(O->bs[k]) > mx) mx = bsz_o(O->bs[k]);
    return malloc(sizeof(double) * mx);
}
/* LP entries that are halves of split pairs (d side: Z there is +-(an equality), zero at a
 * feasible y), exempt from the margin; NULL when none */
static char **g_pairmask = NULL;
/* the margin of a block, relative to 1 + max diagonal: at least what a rigorous check of
 * lambda_min >= 0 in double precision needs (boundcert.c: a Cholesky's rounding is
 * bounded by ~3 n^2 u max diagonal), and at least -boundmargin */
/* 4.39 (ISSUES 26): the d-side barrier step, the mirror of scaled_step. With Zs = Z + delta I
 * positive definite and S = Zs^-1, the y-step whose A*dy is closest to -delta I in the metric of
 * Zs (min |S^1/2 (-delta I - A*dy) S^1/2|_F): M dy = -delta A(S S), M_ij = <A_i, S A_j S>; then
 * Z - A*dy is Zs minus the part of delta I the range of A* misses, measured where Zs is small.
 * Returns 1 when the step was taken (y updated). */
static int scaled_step_d(const PSOrig *O, double *y, double **Z, double delta, int verbose) {
    const int m = O->m;
    if (m <= 0) return 0;
    double cost = (double)m * (double)O->nnz;
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0 && O->bs[O->blk[q]] > 0) cost += 2.0 * (double)O->bs[O->blk[q]] * O->bs[O->blk[q]];
    if (cost > 4e10 || 8.0 * (double)m * m > 0.2 * brisk_mem_limit()) return 0;
    double **S = bound_alloc_blocks(O);
    for (int k = 0; k < O->nblk; k++) {
        if (O->bs[k] < 0) {
            for (int i = 0; i < -O->bs[k]; i++) {
                if (g_pairmask && g_pairmask[k] && g_pairmask[k][i]) { S[k][i] = 0; continue; }   /* equalities: no barrier */
                const double z = Z[k][i] + delta; if (!(z > 0)) { bound_free_blocks(S, O->nblk); return 0; } S[k][i] = 1.0 / z;
            }
            continue;
        }
        const int n = O->bs[k]; int info;
        memcpy(S[k], Z[k], sizeof(double) * (size_t)n * n);
        for (int i = 0; i < n; i++) S[k][i + (size_t)i * n] += delta;
        BL(dpotrf_)("L", &n, S[k], &n, &info);
        if (info == 0) BL(dpotri_)("L", &n, S[k], &n, &info);
        if (info != 0) { bound_free_blocks(S, O->nblk); return 0; }
        for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) S[k][i + (size_t)j * n] = S[k][j + (size_t)i * n];
    }
    size_t *ptr = calloc(m + 2, sizeof(size_t)), *ix;
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) ptr[O->con[q] + 1]++;
    for (int i = 0; i < m; i++) ptr[i + 1] += ptr[i];
    ix = malloc(sizeof(size_t) * (ptr[m] + 1));
    size_t *fill = calloc(m + 1, sizeof(size_t));
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) { const int i = O->con[q]; ix[ptr[i] + fill[i]++] = q; }
    free(fill);
    double **P = bound_alloc_blocks(O);
    double *M = calloc((size_t)m * m, sizeof(double)), *col = malloc(sizeof(double) * (m + 1));
    for (int j = 0; j < m; j++) {
        for (int k = 0; k < O->nblk; k++) memset(P[k], 0, sizeof(double) * bsz_o(O->bs[k]));
        for (size_t a = ptr[j]; a < ptr[j + 1]; a++) {
            const size_t q = ix[a];
            const int k = O->blk[q], p = O->ii[q], r = O->jj[q];
            const double v = O->v[q];
            if (O->bs[k] < 0) { P[k][p] += v * S[k][p] * S[k][p]; continue; }
            const int n = O->bs[k];
            const double *sp = S[k] + (size_t)p * n, *sr = S[k] + (size_t)r * n;
            for (int c = 0; c < n; c++) {
                const double a1 = v * sr[c], a2 = p != r ? v * sp[c] : 0.0;
                double *Pc = P[k] + (size_t)c * n;
                for (int i2 = 0; i2 < n; i2++) Pc[i2] += sp[i2] * a1 + sr[i2] * a2;
            }
        }
        opA(O, P, col);
        for (int i = 0; i < m; i++) M[i + (size_t)j * m] = col[i];
    }
    free(ptr); free(ix); free(col);
    for (int j = 0; j < m; j++) for (int i = j + 1; i < m; i++) { const double t = 0.5 * (M[i + (size_t)j * m] + M[j + (size_t)i * m]); M[i + (size_t)j * m] = M[j + (size_t)i * m] = t; }
    /* rhs = -delta A(S S) */
    for (int k = 0; k < O->nblk; k++) {
        if (O->bs[k] < 0) { for (int i = 0; i < -O->bs[k]; i++) P[k][i] = S[k][i] * S[k][i]; continue; }
        const int n = O->bs[k]; const double one = 1.0, zero = 0.0;
        BL(dgemm_)("N", "N", &n, &n, &n, &one, S[k], &n, S[k], &n, &zero, P[k], &n);
    }
    double *rhs = malloc(sizeof(double) * (m + 1));
    opA(O, P, rhs);
    for (int i = 0; i < m; i++) rhs[i] *= -delta;
    double dmax = 0; for (int i = 0; i < m; i++) dmax = fmax(dmax, M[i + (size_t)i * m]);
    double *F = malloc(sizeof(double) * (size_t)m * m); int info = 1;
    for (int t = 0; t < 4 && info != 0; t++) {
        memcpy(F, M, sizeof(double) * (size_t)m * m);
        const double ridge = t == 0 ? 0 : dmax * pow(10.0, -14.0 + 2.0 * t);
        for (int i = 0; i < m; i++) F[i + (size_t)i * m] += ridge;
        BL(dpotrf_)("L", &m, F, &m, &info);
    }
    int ok = 0;
    if (info == 0) {
        const int one = 1;
        BL(dpotrs_)("L", &m, &one, F, &m, rhs, &m, &info);
        if (info == 0) { for (int i = 0; i < m; i++) y[i] += rhs[i]; ok = 1; }
        if (ok && verbose > 1) printf("   bound: d-side barrier-metric step (delta %.2e), m = %d\n", delta, m);
    }
    free(F); free(M); free(rhs);
    bound_free_blocks(P, O->nblk); bound_free_blocks(S, O->nblk);
    return ok;
}

static double blk_margin(int bs, double margin) {
    const double n = abs(bs);
    return bs < 0 ? fmax(margin, 8.0 * 1.1102230246251565e-16) : fmax(margin, 8.0 * n * n * 1.1102230246251565e-16);
}
/* every block of M has lambda_min >= its margin * (1 + max diagonal) */
static int margin_ok(const PSOrig *O, double **M, double margin, double *work) {
    for (int k = 0; k < O->nblk; k++) {
        if (O->bs[k] < 0 && g_pairmask && g_pairmask[k]) {
            const int n = -O->bs[k];
            double d = 0;
            for (int i = 0; i < n; i++) if (!g_pairmask[k][i]) d = fmax(d, fabs(M[k][i]));
            for (int i = 0; i < n; i++) if (!g_pairmask[k][i] && !(M[k][i] > blk_margin(O->bs[k], margin) * (1 + d))) return 0;
            continue;
        }
        if (!pd_shift(O->bs[k], M[k], blk_margin(O->bs[k], margin) * (1 + maxdiag(O->bs[k], M[k])), work)) return 0;
    }
    return 1;
}
/* min over blocks of lambda_min / (1 + max diagonal) */
static double lam_rel(const PSOrig *O, double **M, double *work) {
    double v = INFINITY;
    for (int k = 0; k < O->nblk; k++) {
        if (O->bs[k] < 0 && g_pairmask && g_pairmask[k]) {
            const int n = -O->bs[k];
            double d = 0, mn = INFINITY;
            for (int i = 0; i < n; i++) if (!g_pairmask[k][i]) { d = fmax(d, fabs(M[k][i])); mn = fmin(mn, M[k][i]); }
            if (mn < INFINITY) v = fmin(v, mn / (1 + d));
            continue;
        }
        v = fmin(v, lam_min(O->bs[k], M[k], work) / (1 + maxdiag(O->bs[k], M[k])));
    }
    return O->nblk ? v : 0;
}
static void blend(const PSOrig *O, double **D, double **A, double **B, double theta) {
    for (int k = 0; k < O->nblk; k++) {
        const size_t len = bsz_o(O->bs[k]);
        for (size_t q = 0; q < len; q++) D[k][q] = (1 - theta) * A[k][q] + theta * B[k][q];
    }
}
/* the smallest theta in [0, 1] with margin_ok((1 - theta) A + theta B) (B must pass) */
static double bisect_theta(const PSOrig *O, double **A, double **B, double margin, double **T, double *work) {
    double lo = 0, hi = 1;
    for (int s = 0; s < 14; s++) {
        const double mid = 0.5 * (lo + hi);
        blend(O, T, A, B, mid);
        if (margin_ok(O, T, margin, work)) hi = mid; else lo = mid;
    }
    return hi;
}

/* split pairs: LP positions (k, i1), (k, i2) whose columns (C and every A_j) are exact
 * negatives. Returns the number of pairs; P1[p], P2[p], PK[p] their positions. */
typedef struct { int k, i; size_t s, e; } LPCol;
static const PSOrig *g_pO;
static int cmp_q_lp(const void *a, const void *b) {
    const size_t p = *(const size_t *)a, q = *(const size_t *)b;
    const PSOrig *O = g_pO;
    if (O->blk[p] != O->blk[q]) return O->blk[p] < O->blk[q] ? -1 : 1;
    if (O->ii[p] != O->ii[q]) return O->ii[p] < O->ii[q] ? -1 : 1;
    if (O->con[p] != O->con[q]) return O->con[p] < O->con[q] ? -1 : 1;
    return 0;
}
static int cols_negated(const PSOrig *O, const size_t *idx, const LPCol *a, const LPCol *b) {
    if (a->e - a->s != b->e - b->s || a->e == a->s) return 0;
    for (size_t t = 0; t < a->e - a->s; t++) {
        const size_t p = idx[a->s + t], q = idx[b->s + t];
        if (O->con[p] != O->con[q] || O->v[p] != -O->v[q]) return 0;
    }
    return 1;
}
static double col_key(const PSOrig *O, const size_t *idx, const LPCol *c, int *sg) {
    /* sign-normalized hash: the first value made positive */
    const double s0 = O->v[idx[c->s]] > 0 ? 1.0 : -1.0;
    double h = 0;
    for (size_t t = c->s; t < c->e; t++) h = h * 1.000000119 + (O->con[idx[t]] + 2.0) * 0.6180339887 + s0 * O->v[idx[t]];
    *sg = s0 > 0;
    return h;
}
static int g_nkey;
static double *g_keys;
static int cmp_key(const void *a, const void *b) {
    const int p = *(const int *)a, q = *(const int *)b;
    return g_keys[p] < g_keys[q] ? -1 : g_keys[p] > g_keys[q] ? 1 : 0;
}
int bound_find_pairs(const PSOrig *O, int **P1, int **P2, int **PK) {
    size_t na = 0;
    size_t *idx = malloc(sizeof(size_t) * (O->nnz + 1));
    for (size_t q = 0; q < O->nnz; q++) if (O->bs[O->blk[q]] < 0) idx[na++] = q;
    g_pO = O;
    qsort(idx, na, sizeof(size_t), cmp_q_lp);
    LPCol *cols = malloc(sizeof(LPCol) * (na + 1));
    int nc = 0;
    for (size_t s = 0; s < na;) {
        size_t e = s + 1;
        while (e < na && O->blk[idx[e]] == O->blk[idx[s]] && O->ii[idx[e]] == O->ii[idx[s]]) e++;
        cols[nc].k = O->blk[idx[s]]; cols[nc].i = O->ii[idx[s]]; cols[nc].s = s; cols[nc].e = e; nc++;
        s = e;
    }
    double *keys = malloc(sizeof(double) * (nc + 1));
    int *sg = malloc(sizeof(int) * (nc + 1)), *ord = malloc(sizeof(int) * (nc + 1)), *used = calloc(nc + 1, sizeof(int));
    for (int c = 0; c < nc; c++) { keys[c] = col_key(O, idx, &cols[c], &sg[c]) + 1e6 * cols[c].k; ord[c] = c; }
    g_keys = keys; g_nkey = nc;
    qsort(ord, nc, sizeof(int), cmp_key);
    int np = 0;
    *P1 = malloc(sizeof(int) * (nc / 2 + 1)); *P2 = malloc(sizeof(int) * (nc / 2 + 1)); *PK = malloc(sizeof(int) * (nc / 2 + 1));
    for (int a = 0; a < nc; a++) {
        const int ca = ord[a];
        if (used[ca]) continue;
        for (int b2 = a + 1; b2 < nc && keys[ord[b2]] == keys[ca]; b2++) {
            const int cb = ord[b2];
            if (used[cb] || sg[cb] == sg[ca] || cols[cb].k != cols[ca].k) continue;
            if (cols_negated(O, idx, &cols[ca], &cols[cb])) {
                used[ca] = used[cb] = 1;
                (*P1)[np] = cols[ca].i; (*P2)[np] = cols[cb].i; (*PK)[np] = cols[ca].k; np++;
                break;
            }
        }
    }
    free(idx); free(cols); free(keys); free(sg); free(ord); free(used);
    return np;
}
/* y += the least-squares correction making Z zero on the first half of every pair
 * (a_e'y = c_e: the equalities the pairs stand for); dense normal equations */
static void project_pairs(const PSOrig *O, int np, const int *P1, const int *PK, double *y) {
    if (np <= 0) return;
    const int m = O->m;
    double *Ap = calloc((size_t)np * m, sizeof(double)), *ce = calloc(np + 1, sizeof(double));
    /* row p of Ap: the coefficients of y in Z at (PK[p], P1[p]); ce: C there */
    int *pos = malloc(sizeof(int) * (np + 1));
    for (int p = 0; p < np; p++) pos[p] = p;
    for (size_t q = 0; q < O->nnz; q++) {
        const int k = O->blk[q];
        if (O->bs[k] >= 0) continue;
        for (int p = 0; p < np; p++)
            if (PK[p] == k && P1[p] == O->ii[q]) {
                if (O->con[q] < 0) ce[p] += O->v[q]; else Ap[p + (size_t)O->con[q] * np] += O->v[q];
            }
    }
    free(pos);
    for (int pass = 0; pass < 3; pass++) {
        /* residual: Z_e = ce - Ap y  (want 0) */
        double *r = malloc(sizeof(double) * (np + 1));
        for (int p = 0; p < np; p++) { double s = ce[p]; for (int i = 0; i < m; i++) s -= Ap[p + (size_t)i * np] * y[i]; r[p] = s; }
        double *G = calloc((size_t)np * np, sizeof(double));
        for (int i = 0; i < m; i++)
            for (int a = 0; a < np; a++) { const double va = Ap[a + (size_t)i * np]; if (va == 0) continue; for (int b = 0; b < np; b++) G[a + (size_t)b * np] += va * Ap[b + (size_t)i * np]; }
        double dmax = 0;
        for (int a = 0; a < np; a++) dmax = fmax(dmax, G[a + (size_t)a * np]);
        for (int a = 0; a < np; a++) G[a + (size_t)a * np] += 1e-14 * dmax + (G[a + (size_t)a * np] == 0 ? 1.0 : 0.0);
        int info, one = 1;
        BL(dpotrf_)("L", &np, G, &np, &info);
        if (info == 0) {
            BL(dpotrs_)("L", &np, &one, G, &np, r, &np, &info);
            for (int i = 0; i < m; i++) { double s = 0; for (int p = 0; p < np; p++) s += Ap[p + (size_t)i * np] * r[p]; y[i] += s; }
        }
        free(G); free(r);
        if (info != 0) break;
    }
    free(Ap); free(ce);
}

/* Z = C - sum y_i A_i */
static void build_Z(const PSOrig *O, const double *y, double **Z) {
    for (int k = 0; k < O->nblk; k++) memset(Z[k], 0, sizeof(double) * bsz_o(O->bs[k]));
    opAT(O, y, -1.0, 1.0, Z);
}

/* e = G^{-1} A(I): A'e is the projection of the identity onto the range of A' */
static void identity_dir(const PSOrig *O, const Gram0 *G, double *e) {
    double **I = bound_alloc_blocks(O);
    for (int k = 0; k < O->nblk; k++) {
        const int n = abs(O->bs[k]);
        for (int i = 0; i < n; i++) { if (O->bs[k] < 0) I[k][i] = 1; else I[k][i + (size_t)i * n] = 1; }
    }
    double *r = malloc(sizeof(double) * (O->m + 1));
    opA(O, I, r);
    gram_solve(O, G, r, e);
    /* one refinement pass */
    double **T = bound_alloc_blocks(O), *r2 = malloc(sizeof(double) * (O->m + 1)), *de = malloc(sizeof(double) * (O->m + 1));
    opAT(O, e, 1.0, 0.0, T);
    opA(O, T, r2);
    for (int i = 0; i < O->m; i++) r2[i] = r[i] - r2[i];
    gram_solve(O, G, r2, de);
    for (int i = 0; i < O->m; i++) e[i] += de[i];
    bound_free_blocks(T, O->nblk); bound_free_blocks(I, O->nblk);
    free(r); free(r2); free(de);
}

void bound_cert_free(BoundCert *c) {
    if (!c) return;
    bound_free_blocks(c->X, c->nblk);
    free(c->y);
    memset(c, 0, sizeof(*c));
}

/* 1 if a is a better certificate than b for this side */
int bound_better(const BoundCert *a, const BoundCert *b) {
    if (!a->have) return 0;
    if (!b->have) return 1;
    if (a->valid != b->valid) return a->valid > b->valid;
    /* neither is a certificate: the objective of an infeasible point means nothing; the one
     * closer to feasibility (larger lambda_min, then smaller residual) */
    if (!a->valid) return a->lammin != b->lammin ? a->lammin > b->lammin : a->resid < b->resid;
    return a->side == 1 ? a->value < b->value : a->value > b->value;
}

/* A certificate from a candidate (X for p, y for d) on the problem as read, with an
 * optional anchor (a more central candidate of the same side). Returns 1 when one was made. */
int bound_certify(const PSOrig *O, int side, double **X, const double *y, int na, double ***XA, double **YA,
                  double margin, double tol, const char *src, int verbose, BoundCert *out) {
    memset(out, 0, sizeof(*out));
    out->side = side;
    out->nblk = O->nblk; out->m = O->m;
    snprintf(out->src, sizeof(out->src), "%s", src);
    double *work = work_for(O);
    const Gram0 *G = side == 1 ? gram_get(O, verbose) : NULL;
    double **T = bound_alloc_blocks(O);
    double *tmp = malloc(sizeof(double) * (O->m + 1));
    if (side == 1) {
        if (!X) { free(work); bound_free_blocks(T, O->nblk); free(tmp); return 0; }
        double **Xp = bound_alloc_blocks(O);
        copy_blocks(O, Xp, X);
        const double r0 = resid_p(O, Xp, tmp);
        scaled_step(O, Xp, verbose);
        out->resid = project_p(O, G, Xp);
        if (verbose > 1) printf("   bound (P) %s: residual %.1e -> %.1e after projection\n", src, r0, out->resid);
        double theta = 0, t_id = 0;
        const char *how = "";
        if (!margin_ok(O, Xp, margin, work)) {
            int done = 0;
            for (int a = na - 1; a >= 0 && !done; a--) {      /* the most recent anchor first */
                if (!XA[a]) continue;
                double **Xap = bound_alloc_blocks(O);
                copy_blocks(O, Xap, XA[a]);
                scaled_step(O, Xap, 0);
                project_p(O, G, Xap);
                if (getenv("BRISK_BOUNDDBG")) printf("   [bound anchor %d/%d: lambda_min rel %.2e, <C,X> %.10e]\n", a, na, lam_rel(O, Xap, work), opA(O, Xap, tmp));
                if (margin_ok(O, Xap, margin, work)) {
                    theta = bisect_theta(O, Xp, Xap, margin, T, work);
                    blend(O, Xp, Xp, Xap, theta);
                    done = 1; how = "blended with an anchor";
                }
                bound_free_blocks(Xap, O->nblk);
            }
            if (!done) {
                /* X + t (I - A*(G^{-1} A(I))): the null-space part of the identity */
                double *e = malloc(sizeof(double) * (O->m + 1));
                identity_dir(O, G, e);
                double **N = bound_alloc_blocks(O);
                for (int k = 0; k < O->nblk; k++) {
                    const int n = abs(O->bs[k]);
                    for (int i = 0; i < n; i++) { if (O->bs[k] < 0) N[k][i] = 1; else N[k][i + (size_t)i * n] = 1; }
                }
                opAT(O, e, -1.0, 0.0, N);
                if (margin_ok(O, N, 1e-6, work)) {
                    double t = 1e-12 * (1 + maxdiag(O->bs[0], Xp[0]));
                    for (int s = 0; s < 80; s++) {
                        for (int k = 0; k < O->nblk; k++) { const size_t len = bsz_o(O->bs[k]); for (size_t q = 0; q < len; q++) T[k][q] = Xp[k][q] + t * N[k][q]; }
                        if (margin_ok(O, T, margin, work)) break;
                        t *= 2;
                    }
                    if (margin_ok(O, T, margin, work)) { copy_blocks(O, Xp, T); t_id = t; done = 1; how = "moved along the null-space identity"; }
                }
                bound_free_blocks(N, O->nblk);
                free(e);
            }
            if (!done && X) {
                /* X + eps I is positive definite; the barrier-metric step from there puts it
                 * back on A(X) = b (the correction is measured against X + eps I) */
                double lneg = 0;
                for (int k = 0; k < O->nblk; k++) lneg = fmin(lneg, lam_min(O->bs[k], X[k], work));
                double **Xr = bound_alloc_blocks(O);
                for (double f = 3.0; f <= 3e4 && !done; f *= 10.0) {
                    copy_blocks(O, Xr, X);
                    double eps = 0;
                    for (int k = 0; k < O->nblk; k++) eps = fmax(eps, margin * (1 + maxdiag(O->bs[k], X[k])));
                    eps = f * fmax(-lneg, eps);
                    for (int k = 0; k < O->nblk; k++) {
                        const int n = abs(O->bs[k]);
                        for (int i = 0; i < n; i++) { if (O->bs[k] < 0) Xr[k][i] += eps; else Xr[k][i + (size_t)i * n] += eps; }
                    }
                    if (!scaled_step(O, Xr, 0)) { if (getenv("BRISK_BOUNDDBG")) printf("   [bound eps %.2e: scaled step not possible]\n", eps); break; }
                    project_p(O, G, Xr);
                    if (getenv("BRISK_BOUNDDBG")) printf("   [bound eps %.2e: lambda_min rel %.2e, <C,X> %.10e]\n", eps, lam_rel(O, Xr, work), opA(O, Xr, tmp));
                    if (margin_ok(O, Xr, margin, work)) { copy_blocks(O, Xp, Xr); done = 1; theta = eps; how = "shifted by eps I and stepped back onto A(X) = b"; }
                }
                bound_free_blocks(Xr, O->nblk);
            }
            if (!done) how = "margin not reached";
            out->resid = resid_p(O, Xp, tmp);
        }
        out->value = opA(O, Xp, tmp);
        out->lammin = lam_rel(O, Xp, work);
        out->theta = theta > 0 ? theta : t_id;
        out->valid = out->lammin >= 0 && out->resid <= tol ? (margin_ok(O, Xp, 0.99 * margin, work) ? 2 : 1) : 0;
        out->X = Xp;
        out->have = 1;
        if (verbose > 1 && *how) printf("   bound (P) %s: %s (%.2e)\n", src, how, out->theta);
    } else {
        if (!y) { free(work); bound_free_blocks(T, O->nblk); free(tmp); return 0; }
        double *yc = malloc(sizeof(double) * (O->m + 1));
        memcpy(yc, y, sizeof(double) * O->m);
        int *P1 = NULL, *P2 = NULL, *PK = NULL;
        const int np = bound_find_pairs(O, &P1, &P2, &PK);
        if (np > 0) {
            g_pairmask = calloc(O->nblk + 1, sizeof(char *));
            for (int p = 0; p < np; p++) {
                if (!g_pairmask[PK[p]]) g_pairmask[PK[p]] = calloc(-O->bs[PK[p]] + 1, 1);
                g_pairmask[PK[p]][P1[p]] = g_pairmask[PK[p]][P2[p]] = 1;
            }
            project_pairs(O, np, P1, PK, yc);
            if (verbose > 1) printf("   bound (D) %s: %d split pairs (equalities) projected\n", src, np);
        }
        double **Z = bound_alloc_blocks(O);
        build_Z(O, yc, Z);
        double theta = 0;
        const char *how = "";
        if (!margin_ok(O, Z, margin, work)) {
            int done = 0;
            for (int a = na - 1; a >= 0 && !done; a--) {
                if (!YA[a]) continue;
                const double *ya = YA[a];
                double **Za = bound_alloc_blocks(O);
                build_Z(O, ya, Za);
                if (margin_ok(O, Za, margin, work)) {
                    theta = bisect_theta(O, Z, Za, margin, T, work);
                    for (int i = 0; i < O->m; i++) yc[i] = (1 - theta) * yc[i] + theta * ya[i];
                    build_Z(O, yc, Z);
                    done = margin_ok(O, Z, 0.5 * margin, work);
                    how = "blended with an anchor";
                }
                bound_free_blocks(Za, O->nblk);
            }
            if (!done && !bound_late(O)) {   /* 4.42: not past the time limit when m > 2000 (each costs an m x m factorization) */
                /* y - t e with A'e the range projection of the identity: Z + t A'e */
                double *e = malloc(sizeof(double) * (O->m + 1));
                G = gram_get(O, verbose);
                identity_dir(O, G, e);
                double **D = bound_alloc_blocks(O);
                opAT(O, e, 1.0, 0.0, D);
                if (margin_ok(O, D, 1e-6, work)) {
                    double t = 1e-14 * (1 + maxdiag(O->bs[0], Z[0]));
                    double *yt = malloc(sizeof(double) * (O->m + 1));
                    for (int s = 0; s < 90; s++) {
                        for (int i = 0; i < O->m; i++) yt[i] = yc[i] - t * e[i];
                        build_Z(O, yt, T);
                        if (margin_ok(O, T, margin, work)) break;
                        t *= 2;
                    }
                    if (margin_ok(O, T, margin, work)) { memcpy(yc, yt, sizeof(double) * O->m); build_Z(O, yc, Z); done = 1; theta = t; how = "moved along y - t e (A'e ~ I)"; }
                    free(yt);
                }
                bound_free_blocks(D, O->nblk);
                free(e);
            }
            if (!done && !bound_late(O)) {   /* 4.42: not past the time limit when m > 2000 (each costs an m x m factorization) */
                /* 4.39: Z + delta I and the barrier-metric y-step back (ISSUES 26: truss2-4 have
                 * no A'e ~ I and no anchors) */
                double lneg = 0;
                for (int k = 0; k < O->nblk; k++) lneg = fmin(lneg, lam_min(O->bs[k], Z[k], work));
                double eps0 = 0;
                for (int k = 0; k < O->nblk; k++) eps0 = fmax(eps0, margin * (1 + maxdiag(O->bs[k], Z[k])));
                double *yt = malloc(sizeof(double) * (O->m + 1));
                for (double f = 3.0; f <= 3e4 && !done; f *= 10.0) {
                    memcpy(yt, yc, sizeof(double) * O->m);
                    const double delta = f * fmax(-lneg, eps0);
                    if (!scaled_step_d(O, yt, Z, delta, 0)) break;
                    if (np > 0) project_pairs(O, np, P1, PK, yt);
                    build_Z(O, yt, T);
                    if (getenv("BRISK_BOUNDDBG")) { double by = 0; for (int i = 0; i < O->m; i++) by += O->b[i] * yt[i]; printf("   [bound d delta %.2e: lambda_min rel %.2e, b'y %.10e]\n", delta, lam_rel(O, T, work), by); }
                    if (margin_ok(O, T, margin, work)) { memcpy(yc, yt, sizeof(double) * O->m); copy_blocks(O, Z, T); done = 1; theta = delta; how = "Z + delta I and the barrier-metric y-step back"; }
                }
                free(yt);
            }
            if (!done) how = "margin not reached";
        }
        if (np > 0) {   /* the anchors and shifts above may have moved the equalities */
            project_pairs(O, np, P1, PK, yc);
            build_Z(O, yc, Z);
        }
        double by = 0;
        for (int i = 0; i < O->m; i++) by += O->b[i] * yc[i];
        out->value = by;
        out->resid = 0;
        out->lammin = lam_rel(O, Z, work);
        out->theta = theta;
        out->valid = out->lammin >= 0 ? (margin_ok(O, Z, 0.99 * margin, work) ? 2 : 1) : 0;
        out->y = yc;
        out->have = 1;
        bound_free_blocks(Z, O->nblk);
        if (g_pairmask) { for (int k = 0; k < O->nblk; k++) free(g_pairmask[k]); free(g_pairmask); g_pairmask = NULL; }
        free(P1); free(P2); free(PK);
        if (verbose > 1 && *how) printf("   bound (D) %s: %s (%.2e)\n", src, how, theta);
    }
    bound_free_blocks(T, O->nblk);
    free(tmp); free(work);
    return 1;
}
