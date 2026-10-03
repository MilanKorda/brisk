/* SDP crossover (4.20): from an interior-point solution to a low-rank KKT point.
 *
 * On the problem as read (PSOrig: min <C,X> s.t. <A_i,X> = b_i, X >= 0, Z = C - A'y >= 0),
 * each SDP block of the returned X is factored X_k = U_k U_k' with the rank read from
 * complementarity (an eigenvector q of X is kept when its X eigenvalue, relative to the
 * largest, dominates its Z Rayleigh quotient q'Zq relative to the largest), LP blocks keep
 * their support with x_j = u_j^2. Then Newton steps (least squares; the gauge U -> UQ is
 * left to the rank-revealing solve) on
 *
 *     A(U U') - b = 0,      Z(y) U_k = 0  (every SDP block),   z_j u_j = 0  (LP support),
 *
 * whose solutions have exact complementarity <X,Z> = 0 and X = UU' PSD by construction.
 * What the data leave open is lambda_min(Z) on the complement of range(U): strictly
 * complementary, nondegenerate problems converge quadratically to rounding level; on
 * degenerate ones the system is singular and the result is usually no better than the
 * input. The result is kept only if the largest DIMACS error on the original data falls.
 */
#include "brisk.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void BL(dgelsy_)(const int *, const int *, const int *, double *, const int *, double *, const int *,
                 int *, const double *, int *, double *, const int *, int *);
void BL(dsyev_)(const char *, const char *, const int *, double *, const int *, double *, double *,
                const int *, int *);

static const int IONE_ = 1;
static double xsq_(int n, const double *x) { double s = 0; for (int i = 0; i < n; i++) s += x[i] * x[i]; return s; }
static double xdot_(int n, const double *x, const double *y) { double s = 0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s; }
static double xo_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static size_t bsz_o(int b) { return b < 0 ? (size_t)(-b) : (size_t)b * b; }

/* eigenvalues (ascending) and optionally vectors of a symmetric n x n matrix (copy) */
static int sym_eig(int n, const double *A, double *w, double *V, int vec) {
    double *T = V ? V : malloc(sizeof(double) * (size_t)n * n);
    memcpy(T, A, sizeof(double) * (size_t)n * n);
    int lwork = -1, info;
    double wq;
    BL(dsyev_)(vec ? "V" : "N", "L", &n, T, &n, w, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    double *work = malloc(sizeof(double) * lwork);
    BL(dsyev_)(vec ? "V" : "N", "L", &n, T, &n, w, work, &lwork, &info);
    free(work);
    if (!V) free(T);
    return info;
}

typedef struct {
    const PSOrig *O;
    int m, nb;
    int *start, *cnt;       /* entries of O grouped by block: idx[start[k] .. + cnt[k]] */
    size_t *idx;
    double **C;             /* dense C per block */
} XOData;

static void xo_Z(const XOData *D, const double *y, double **Z) {
    const PSOrig *O = D->O;
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]);
        memcpy(Z[k], D->C[k], sizeof(double) * bsz_o(O->bs[k]));
        for (int e = 0; e < D->cnt[k]; e++) {
            const size_t q = D->idx[D->start[k] + e];
            if (O->con[q] < 0) continue;
            const double c = -y[O->con[q]] * O->v[q];
            const int i = O->ii[q], j = O->jj[q];
            if (O->bs[k] < 0) Z[k][i] += c;
            else { Z[k][i + (size_t)j * n] += c; if (i != j) Z[k][j + (size_t)i * n] += c; }
        }
    }
}

/* DIMACS errors 1..6 on O (err[3] = 0: Z is C - A'y exactly) */
static double g_xo_pd[2], g_xo_floor;
static void xo_errors(const XOData *D, double **X, const double *y, double **Z, double *err) {
    const PSOrig *O = D->O;
    const int m = D->m;
    double *r = calloc(m + 1, sizeof(double));
    double pobj = 0, dobj = 0, xz = 0, lx = 0, lz = 0;
    for (int i = 0; i < m; i++) dobj += O->b[i] * y[i];
    for (size_t q = 0; q < O->nnz; q++) {
        const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
        const double x = O->bs[k] < 0 ? X[k][i] : X[k][i + (size_t)j * n];
        const double c = (O->bs[k] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
        if (O->con[q] < 0) pobj += c; else r[O->con[q]] += c;
    }
    double r2 = 0;
    for (int i = 0; i < m; i++) { double v = r[i] - O->b[i]; r2 += v * v; }
    double axz = 0, aby = 0;
    for (int i = 0; i < m; i++) aby += fabs(O->b[i] * y[i]);
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]);
        const size_t len = bsz_o(O->bs[k]);
        for (size_t t = 0; t < len; t++) { xz += X[k][t] * Z[k][t]; axz += fabs(X[k][t] * Z[k][t]); }
        if (O->bs[k] < 0) {
            for (int i = 0; i < n; i++) { lx = fmin(lx, X[k][i]); lz = fmin(lz, Z[k][i]); }
        } else {
            double *w = malloc(sizeof(double) * n);
            if (sym_eig(n, X[k], w, NULL, 0) == 0) lx = fmin(lx, w[0]);
            if (sym_eig(n, Z[k], w, NULL, 0) == 0) lz = fmin(lz, w[0]);
            free(w);
        }
    }
    const double den = 1 + fabs(pobj) + fabs(dobj);
    err[0] = 0;
    err[1] = sqrt(r2) / (1 + O->nb1);
    err[2] = fmax(0.0, -lx) / (1 + O->nb1);
    err[3] = 0;
    err[4] = fmax(0.0, -lz) / (1 + O->nC1);
    err[5] = (pobj - dobj) / den;
    err[6] = xz / den;
    g_xo_pd[0] = pobj; g_xo_pd[1] = dobj;
    /* rounding floor of the gap measures: with huge duals (gpp250-1: |y| 4e6) <X,Z> and
     * b'y cancel, and differences below this are noise (4.20) */
    g_xo_floor = 1e-14 * (axz + aby) / den;
    free(r);
}
static double xo_maxerr(const double *e) {
    double v = 0;
    for (int i = 1; i <= 6; i++) v = fmax(v, fabs(e[i]));
    return v;
}

typedef struct {
    int nb, m, nU;
    int *r;                 /* SDP: rank; LP: support size */
    int **sup;              /* LP support */
    int *off;               /* offset of block k's unknowns */
} XOShape;

/* residual F (m + nU) at unknowns v = [U_k (col-major n x r) | u_LP | y] */
static void xo_build_X(const XOData *D, const XOShape *S, const double *v, double **X) {
    const PSOrig *O = D->O;
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]), r = S->r[k];
        const double *U = v + S->off[k];
        memset(X[k], 0, sizeof(double) * bsz_o(O->bs[k]));
        if (O->bs[k] < 0) { for (int t = 0; t < r; t++) X[k][S->sup[k][t]] = U[t] * U[t]; continue; }
        if (r == 0) continue;
        const double one = 1.0, zero = 0.0;
        BL(dgemm_)("N", "T", &n, &n, &r, &one, U, &n, U, &n, &zero, X[k], &n);
    }
}
static double xo_residual(const XOData *D, const XOShape *S, const double *v, double *F, double **X, double **Z) {
    const PSOrig *O = D->O;
    const int m = D->m;
    const double *y = v + S->nU;
    xo_build_X(D, S, v, X);
    xo_Z(D, y, Z);
    memset(F, 0, sizeof(double) * m);
    for (size_t q = 0; q < O->nnz; q++) {
        if (O->con[q] < 0) continue;
        const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
        const double x = O->bs[k] < 0 ? X[k][i] : X[k][i + (size_t)j * n];
        F[O->con[q]] += (O->bs[k] > 0 && i != j) ? 2.0 * O->v[q] * x : O->v[q] * x;
    }
    for (int i = 0; i < m; i++) F[i] -= O->b[i];
    double *Fp = F + m;
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]), r = S->r[k];
        const double *U = v + S->off[k];
        if (O->bs[k] < 0) {
            for (int t = 0; t < r; t++) Fp[t] = Z[k][S->sup[k][t]] * U[t];
            Fp += r;
            continue;
        }
        if (r == 0) continue;
        const double one = 1.0, zero = 0.0;
        BL(dgemm_)("N", "N", &n, &r, &n, &one, Z[k], &n, U, &n, &zero, Fp, &n);
        Fp += (size_t)n * r;
    }
    double s = 0;
    for (int i = 0; i < m + S->nU; i++) s += F[i] * F[i];
    return sqrt(s);
}

#if 0  /* the dense Jacobian of the first version (dgelsy): 10-15x slower */
/* dense Jacobian, rows m + nU, columns nU + m (column-major, leading dimension NR) */
static void xo_jacobian(const XOData *D, const XOShape *S, const double *v, double **Z, double *J, int NR) {
    const PSOrig *O = D->O;
    const int m = D->m, nU = S->nU;
    memset(J, 0, sizeof(double) * (size_t)NR * (nU + m));
    int rowb = m;
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]), r = S->r[k], col = S->off[k];
        const double *U = v + col;
        if (O->bs[k] < 0) {
            int *pos = malloc(sizeof(int) * n);
            for (int i = 0; i < n; i++) pos[i] = -1;
            for (int t = 0; t < r; t++) pos[S->sup[k][t]] = t;
            for (int e = 0; e < D->cnt[k]; e++) {
                const size_t q = D->idx[D->start[k] + e];
                const int i = O->ii[q], t = pos[i];
                if (t < 0) continue;
                if (O->con[q] >= 0) {
                    J[O->con[q] + (size_t)(col + t) * NR] += 2.0 * O->v[q] * U[t];
                    J[(rowb + t) + (size_t)(nU + O->con[q]) * NR] += -O->v[q] * U[t];
                }
            }
            for (int t = 0; t < r; t++) J[(rowb + t) + (size_t)(col + t) * NR] += Z[k][S->sup[k][t]];
            free(pos);
            rowb += r;
            continue;
        }
        for (int e = 0; e < D->cnt[k]; e++) {
            const size_t q = D->idx[D->start[k] + e];
            const int con = O->con[q];
            if (con < 0) continue;
            const int i = O->ii[q], j = O->jj[q];
            const double a = O->v[q];
            for (int c = 0; c < r; c++) {
                /* d <A,UU'> / dU[i,c] += 2 a U[j,c]; (A U)[i,c] += a U[j,c] */
                J[con + (size_t)(col + c * n + i) * NR] += 2.0 * a * U[j + (size_t)c * n];
                J[(rowb + c * n + i) + (size_t)(nU + con) * NR] += -a * U[j + (size_t)c * n];
                if (i != j) {
                    J[con + (size_t)(col + c * n + j) * NR] += 2.0 * a * U[i + (size_t)c * n];
                    J[(rowb + c * n + j) + (size_t)(nU + con) * NR] += -a * U[i + (size_t)c * n];
                }
            }
        }
        /* d(ZU)/dU = I (x) Z */
        for (int c = 0; c < r; c++)
            for (int qq = 0; qq < n; qq++) {
                double *Jc = J + (size_t)(col + c * n + qq) * NR + rowb + c * n;
                const double *zc = Z[k] + (size_t)qq * n;
                for (int p = 0; p < n; p++) Jc[p] += zc[p];
            }
        rowb += n * r;
    }
}


#endif

/* B (nU x m, column-major): column i = vec over the blocks of (A_i U_k) (SDP) and
 * (A_i)_jj u_t (LP support); the Jacobian is J = [2 B', 0; Zb, -B] with Zb =
 * blockdiag(I_r (x) Z_k, diag(z_sup)).                                                 */
static void xo_B(const XOData *D, const XOShape *S, const double *v, double *B) {
    const PSOrig *O = D->O;
    const int nU = S->nU;
    memset(B, 0, sizeof(double) * (size_t)nU * D->m);
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]), r = S->r[k], col = S->off[k];
        const double *U = v + col;
        if (O->bs[k] < 0) {
            int *pos = malloc(sizeof(int) * (n + 1));
            for (int i = 0; i < n; i++) pos[i] = -1;
            for (int t = 0; t < r; t++) pos[S->sup[k][t]] = t;
            for (int e = 0; e < D->cnt[k]; e++) {
                const size_t q = D->idx[D->start[k] + e];
                if (O->con[q] < 0) continue;
                const int t = pos[O->ii[q]];
                if (t >= 0) B[(col + t) + (size_t)O->con[q] * nU] += O->v[q] * U[t];
            }
            free(pos);
            continue;
        }
        for (int e = 0; e < D->cnt[k]; e++) {
            const size_t q = D->idx[D->start[k] + e];
            const int con = O->con[q];
            if (con < 0) continue;
            const int i = O->ii[q], j = O->jj[q];
            const double a = O->v[q];
            double *Bc = B + (size_t)con * nU + col;
            for (int c = 0; c < r; c++) {
                Bc[c * n + i] += a * U[j + (size_t)c * n];
                if (i != j) Bc[c * n + j] += a * U[i + (size_t)c * n];
            }
        }
    }
}
/* out = Zb x (x of length nU), or for each of ncol columns (leading dimension nU) */
static void xo_Zb(const XOData *D, const XOShape *S, double **Z, const double *x, double *out, int ncol) {
    const PSOrig *O = D->O;
    const int nU = S->nU;
    for (int k = 0; k < D->nb; k++) {
        const int n = abs(O->bs[k]), r = S->r[k], col = S->off[k];
        if (O->bs[k] < 0) {
            for (int cc = 0; cc < ncol; cc++)
                for (int t = 0; t < r; t++) out[col + t + (size_t)cc * nU] = Z[k][S->sup[k][t]] * x[col + t + (size_t)cc * nU];
            continue;
        }
        if (r == 0) continue;
        /* each (column c of U_k, rhs cc) slab: n x 1; together n x (r ncol) with stride */
        for (int cc = 0; cc < ncol; cc++) {
            const double one = 1.0, zero = 0.0;
            BL(dgemm_)("N", "N", &n, &r, &n, &one, Z[k], &n, x + col + (size_t)cc * nU, &n, &zero, out + col + (size_t)cc * nU, &n);
        }
    }
}

/* y = J x  (x = [dU; dy], y = [2 B' dU; Zb dU - B dy]) and x = J' w */
static void xo_Jmv(const XOData *D, const XOShape *S, double **Z, const double *Bm, const double *x, double *y, double *tmp) {
    const int m = D->m, nU = S->nU;
    const double two = 2.0, zero = 0.0, mone = -1.0, one = 1.0;
    if (nU == 0) { memset(y, 0, sizeof(double) * m); return; }
    BL(dgemv_)("T", &nU, &m, &two, Bm, &nU, x, &IONE_, &zero, y, &IONE_);
    xo_Zb(D, S, Z, x, y + m, 1);
    BL(dgemv_)("N", &nU, &m, &mone, Bm, &nU, x + nU, &IONE_, &one, y + m, &IONE_);
    (void)tmp;
}
static void xo_JTmv(const XOData *D, const XOShape *S, double **Z, const double *Bm, const double *w, double *x, double *tmp) {
    const int m = D->m, nU = S->nU;
    const double two = 2.0, zero = 0.0, mone = -1.0;
    if (nU == 0) { memset(x, 0, sizeof(double) * (m + nU)); return; }
    BL(dgemv_)("N", &nU, &m, &two, Bm, &nU, w, &IONE_, &zero, x, &IONE_);
    xo_Zb(D, S, Z, w + m, tmp, 1);
    for (int p = 0; p < nU; p++) x[p] += tmp[p];
    BL(dgemv_)("T", &nU, &m, &mone, Bm, &nU, w + m, &IONE_, &zero, x + nU, &IONE_);
}

void BL(dsysv_)(const char *, const int *, const int *, double *, const int *, int *, double *, const int *,
                double *, const int *, int *);
/* Newton direction by the reduced system (4.20). Per SDP block, Z = V diag(lam) V' with the
 * r smallest eigenvalues first ("small" rows: the range of U at a solution), dU = V W,
 * P_j = V' A_j U, R = V' F2. The rows of Z dU - (A'dy) U = -F2 on the n - r "big"
 * eigenvalues give W_b = lam_b^{-1} (sum_j dy_j P_j,b - R_b) explicitly; what remains is
 *   [ K     2 Ps' ] [dy ]   [ -F1 + 2 sum <P_i,b, lam_b^{-1} R_b> ]
 *   [ 2 Ps -2 lam_s] [W_s] = [ 2 R_s                              ]
 * with K_ij = 2 sum <P_i,b, lam_b^{-1} P_j,b>: size m + sum r^2 instead of m + sum n r
 * (mcp250-1: 875 against 6,500). LP support entries are "small" rows with V = I.
 * The gauge U -> U Q (null directions W_s = C S, S skew) is fixed by a tiny
 * regularization and removed by refinement against the unregularized matrix.        */
static int xo_reduced_dir(const XOData *D, const XOShape *S, const double *v, const double *F,
                          double **Z, const double *Bm, double *dir, int verbose) {
    const PSOrig *O = D->O;
    const int m = D->m, nb = D->nb, nU = S->nU;
    int *soff = malloc(sizeof(int) * (nb + 1));
    int Ns = 0;
    for (int k = 0; k < nb; k++) { soff[k] = Ns; Ns += O->bs[k] < 0 ? S->r[k] : S->r[k] * S->r[k]; }
    const int N = m + Ns;
    double *Kr = calloc((size_t)N * N, sizeof(double)), *rhs = calloc(N, sizeof(double));
    double **P = calloc(nb, sizeof(double *)), **Vk = calloc(nb, sizeof(double *)), **lam = calloc(nb, sizeof(double *)), **Rk = calloc(nb, sizeof(double *));
    const double one = 1.0, zero = 0.0, two = 2.0;
    int rc = 0;
    for (int i = 0; i < m; i++) rhs[i] = -F[i];
    const double *F2 = F + m;
    for (int k = 0; k < nb && !rc; k++) {
        const int n = abs(O->bs[k]), r = S->r[k], col = S->off[k];
        if (r == 0) continue;
        if (O->bs[k] < 0) {           /* LP support: small rows, V = I */
            for (int t = 0; t < r; t++) {
                const int row = m + soff[k] + t;
                for (int j = 0; j < m; j++) Kr[row + (size_t)j * N] = 2.0 * Bm[(col + t) + (size_t)j * nU];
                Kr[row + (size_t)row * N] = -2.0 * Z[k][S->sup[k][t]];
                rhs[row] = 2.0 * F2[col + t];
            }
            continue;
        }
        Vk[k] = malloc(sizeof(double) * (size_t)n * n); lam[k] = malloc(sizeof(double) * n);
        if (sym_eig(n, Z[k], lam[k], Vk[k], 1) != 0) { rc = 1; break; }
        /* P = V' [A_j U] for all j (n x r m), R = V' F2_k (n x r) */
        const size_t nrm = (size_t)n * r * m;
        double *Bc = malloc(sizeof(double) * nrm);
        for (int j = 0; j < m; j++) memcpy(Bc + (size_t)j * n * r, Bm + (size_t)j * nU + col, sizeof(double) * (size_t)n * r);
        P[k] = malloc(sizeof(double) * nrm);
        const int rm = r * m;
        BL(dgemm_)("T", "N", &n, &rm, &n, &one, Vk[k], &n, Bc, &n, &zero, P[k], &n);
        free(Bc);
        Rk[k] = malloc(sizeof(double) * (size_t)n * r);
        BL(dgemm_)("T", "N", &n, &r, &n, &one, Vk[k], &n, F2 + col, &n, &zero, Rk[k], &n);
        const int nbg = n - r;
        if (nbg > 0) {
            /* Q = lam_b^{-1/2} P_b ((n-r) r x m), K += 2 Q'Q, rhs1 += 2 Q' (lam_b^{-1/2} R_b) */
            const int nq = nbg * r;
            double *Q = malloc(sizeof(double) * (size_t)nq * m), *q = malloc(sizeof(double) * nq);
            int bad = 0;
            for (int a = r; a < n; a++) if (!(lam[k][a] > 0)) bad = 1;
            if (bad) { free(Q); free(q); rc = 2; break; }
            for (int j = 0; j < m; j++)
                for (int c = 0; c < r; c++)
                    for (int a = r; a < n; a++)
                        Q[(a - r) + (size_t)nbg * c + (size_t)nq * j] = P[k][a + (size_t)n * (c + (size_t)r * j)] / sqrt(lam[k][a]);
            for (int c = 0; c < r; c++)
                for (int a = r; a < n; a++) q[(a - r) + nbg * c] = Rk[k][a + (size_t)n * c] / sqrt(lam[k][a]);
            BL(dsyrk_)("L", "T", &m, &nq, &two, Q, &nq, &one, Kr, &N);
            BL(dgemv_)("T", &nq, &m, &two, Q, &nq, q, &IONE_, &one, rhs, &IONE_);
            free(Q); free(q);
        }
        /* small rows */
        for (int c = 0; c < r; c++)
            for (int a = 0; a < r; a++) {
                const int row = m + soff[k] + a + r * c;
                for (int j = 0; j < m; j++) Kr[row + (size_t)j * N] = 2.0 * P[k][a + (size_t)n * (c + (size_t)r * j)];
                Kr[row + (size_t)row * N] = -2.0 * lam[k][a];
                rhs[row] = 2.0 * Rk[k][a + (size_t)n * c];
            }
    }
    if (!rc) {
        /* symmetric (lower triangle filled): regularize, factor, refine */
        for (int j = 0; j < N; j++) for (int i = 0; i < j; i++) Kr[i + (size_t)j * N] = Kr[j + (size_t)i * N];
        double *K0 = malloc(sizeof(double) * (size_t)N * N);
        memcpy(K0, Kr, sizeof(double) * (size_t)N * N);
        double dmax = 0;
        for (int i = 0; i < N; i++) dmax = fmax(dmax, fabs(Kr[i + (size_t)i * N]));
        const double eps = 1e-14 * dmax + 1e-300;
        for (int i = 0; i < N; i++) Kr[i + (size_t)i * N] += i < m ? eps : -eps;
        int *ipiv = malloc(sizeof(int) * N), info, lw = -1, nr1 = 1;
        double wq;
        double *x = malloc(sizeof(double) * N), *res = malloc(sizeof(double) * N);
        memcpy(x, rhs, sizeof(double) * N);
        BL(dsysv_)("L", &N, &nr1, Kr, &N, ipiv, x, &N, &wq, &lw, &info);
        lw = (int)wq + 1;
        double *work = malloc(sizeof(double) * lw);
        BL(dsysv_)("L", &N, &nr1, Kr, &N, ipiv, x, &N, work, &lw, &info);
        if (info != 0) rc = 3;
        for (int pass = 0; pass < 3 && !rc; pass++) {
            memcpy(res, rhs, sizeof(double) * N);
            const double mone = -1.0;
            BL(dgemv_)("N", &N, &N, &mone, K0, &N, x, &IONE_, &one, res, &IONE_);
            void BL(dsytrs_)(const char *, const int *, const int *, const double *, const int *, const int *, double *, const int *, int *);
            BL(dsytrs_)("L", &N, &nr1, Kr, &N, ipiv, res, &N, &info);
            for (int i = 0; i < N; i++) x[i] += res[i];
        }
        free(work); free(ipiv); free(res); free(K0);
        if (!rc) {
            /* dy, then dU per block */
            memcpy(dir + nU, x, sizeof(double) * m);
            for (int k = 0; k < nb; k++) {
                const int n = abs(O->bs[k]), r = S->r[k], col = S->off[k];
                if (r == 0) continue;
                if (O->bs[k] < 0) { for (int t = 0; t < r; t++) dir[col + t] = x[m + soff[k] + t]; continue; }
                double *W = malloc(sizeof(double) * (size_t)n * r);
                for (int c = 0; c < r; c++) {
                    for (int a = 0; a < r; a++) W[a + (size_t)n * c] = x[m + soff[k] + a + r * c];
                    for (int a = r; a < n; a++) {
                        double g = -Rk[k][a + (size_t)n * c];
                        for (int j = 0; j < m; j++) g += x[j] * P[k][a + (size_t)n * (c + (size_t)r * j)];
                        W[a + (size_t)n * c] = g / lam[k][a];
                    }
                }
                BL(dgemm_)("N", "N", &n, &r, &n, &one, Vk[k], &n, W, &n, &zero, dir + col, &n);
                free(W);
                /* remove the gauge component U S (S skew, least squares): the exact Newton
                 * solution is only defined up to it, and a large one ruins the full step */
                if (r > 1) {
                    const double *U = v + col;
                    double *G = malloc(sizeof(double) * (size_t)r * r), *Hm = malloc(sizeof(double) * (size_t)r * r);
                    double *ev = malloc(sizeof(double) * r), *Qg = malloc(sizeof(double) * (size_t)r * r), *T = malloc(sizeof(double) * (size_t)r * r);
                    BL(dgemm_)("T", "N", &r, &r, &n, &one, U, &n, U, &n, &zero, G, &r);
                    BL(dgemm_)("T", "N", &r, &r, &n, &one, U, &n, dir + col, &n, &zero, Hm, &r);
                    for (int b = 0; b < r; b++) for (int a = 0; a < r; a++) T[a + (size_t)r * b] = Hm[a + (size_t)r * b] - Hm[b + (size_t)r * a];
                    if (sym_eig(r, G, ev, Qg, 1) == 0) {
                        /* G S + S G = T - T' ... in the eigenbasis of G: S~_ab = T~_ab / (g_a + g_b) */
                        BL(dgemm_)("T", "N", &r, &r, &r, &one, Qg, &r, T, &r, &zero, Hm, &r);
                        BL(dgemm_)("N", "N", &r, &r, &r, &one, Hm, &r, Qg, &r, &zero, T, &r);
                        for (int b = 0; b < r; b++) for (int a = 0; a < r; a++) {
                            const double den = ev[a] + ev[b];
                            T[a + (size_t)r * b] = den > 0 ? T[a + (size_t)r * b] / den : 0.0;
                        }
                        BL(dgemm_)("N", "N", &r, &r, &r, &one, Qg, &r, T, &r, &zero, Hm, &r);
                        BL(dgemm_)("N", "T", &r, &r, &r, &one, Hm, &r, Qg, &r, &zero, T, &r);   /* S */
                        const double mone = -1.0;
                        BL(dgemm_)("N", "N", &n, &r, &r, &mone, U, &n, T, &r, &one, dir + col, &n);
                    }
                    free(G); free(Hm); free(ev); free(Qg); free(T);
                }
            }
        }
        free(x);
    }
    if (rc && verbose > 0) printf("crossover: reduced system failed (%d)\n", rc);
    for (int k = 0; k < nb; k++) { free(P[k]); free(Vk[k]); free(lam[k]); free(Rk[k]); }
    free(P); free(Vk); free(lam); free(Rk); free(Kr); free(rhs); free(soff);
    return rc;
}

int sdp_crossover(const PSOrig *O, double **Xo, double *yo, int verbose, double max_seconds,
                  double max_unknowns, double *err_before, double *err_after, double *pd) {
    const double t0 = xo_now();
    const int m = O->m, nb = O->nblk;
    XOData D = { O, m, nb, NULL, NULL, NULL, NULL };
    D.start = calloc(nb + 1, sizeof(int)); D.cnt = calloc(nb + 1, sizeof(int));
    D.idx = malloc(sizeof(size_t) * (O->nnz + 1));
    for (size_t q = 0; q < O->nnz; q++) D.cnt[O->blk[q]]++;
    for (int k = 1; k < nb; k++) D.start[k] = D.start[k - 1] + D.cnt[k - 1];
    { int *fill = calloc(nb + 1, sizeof(int));
      for (size_t q = 0; q < O->nnz; q++) { const int k = O->blk[q]; D.idx[D.start[k] + fill[k]++] = q; }
      free(fill); }
    D.C = malloc(sizeof(double *) * nb);
    double **Z = malloc(sizeof(double *) * nb), **Xn = malloc(sizeof(double *) * nb), **Zn = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) {
        const size_t len = bsz_o(O->bs[k]);
        D.C[k] = calloc(len, sizeof(double));
        Z[k] = malloc(sizeof(double) * len); Xn[k] = malloc(sizeof(double) * len); Zn[k] = malloc(sizeof(double) * len);
        const int n = abs(O->bs[k]);
        for (int e = 0; e < D.cnt[k]; e++) {
            const size_t q = D.idx[D.start[k] + e];
            if (O->con[q] >= 0) continue;
            const int i = O->ii[q], j = O->jj[q];
            if (O->bs[k] < 0) D.C[k][i] += O->v[q];
            else { D.C[k][i + (size_t)j * n] += O->v[q]; if (i != j) D.C[k][j + (size_t)i * n] += O->v[q]; }
        }
    }
    int accepted = 0;
    {   /* the rank decisions cost an eigendecomposition per block (~10 n^3) */
        double e3 = 0;
        for (int k = 0; k < nb; k++) if (O->bs[k] > 0) e3 += 10.0 * (double)O->bs[k] * O->bs[k] * O->bs[k];
        if (e3 > max_seconds * 3e10 || (double)m * m > max_unknowns * max_unknowns) {
            if (verbose > 0) printf("crossover: skipped (eigendecompositions too costly for the budget)\n");
            memcpy(err_after, err_before, sizeof(double) * 7);
            for (int k = 0; k < nb; k++) { free(D.C[k]); free(Z[k]); free(Xn[k]); free(Zn[k]); }
            free(D.C); free(Z); free(Xn); free(Zn); free(D.start); free(D.cnt); free(D.idx);
            err_before[0] = -1;
            return 0;
        }
    }
    xo_Z(&D, yo, Z);
    xo_errors(&D, Xo, yo, Z, err_before);
    const double e0 = xo_maxerr(err_before), floor0 = g_xo_floor;
    /* ranks from complementarity */
    XOShape S = { nb, m, 0, calloc(nb, sizeof(int)), calloc(nb, sizeof(int *)), calloc(nb + 1, sizeof(int)) };
    double **Uinit = calloc(nb, sizeof(double *));
    int ok = 1;
    for (int k = 0; k < nb && ok; k++) {
        const int n = abs(O->bs[k]);
        S.off[k] = S.nU;
        if (O->bs[k] < 0) {
            double xm = 0, zm = 0;
            for (int i = 0; i < n; i++) { xm = fmax(xm, Xo[k][i]); zm = fmax(zm, fabs(Z[k][i])); }
            xm = fmax(xm, 1e-300); zm = fmax(zm, 1e-300);
            S.sup[k] = malloc(sizeof(int) * (n + 1));
            Uinit[k] = malloc(sizeof(double) * (n + 1));
            int r = 0;
            for (int i = 0; i < n; i++)
                if (Xo[k][i] / xm > fmax(Z[k][i], 0.0) / zm) { S.sup[k][r] = i; Uinit[k][r] = sqrt(fmax(Xo[k][i], 0.0)); r++; }
            S.r[k] = r; S.nU += r;
            continue;
        }
        double *w = malloc(sizeof(double) * n), *Q = malloc(sizeof(double) * (size_t)n * n);
        if (sym_eig(n, Xo[k], w, Q, 1) != 0) { ok = 0; free(w); free(Q); break; }
        const double wmax = fmax(w[n - 1], 1e-300);
        double *zq = malloc(sizeof(double) * n), *t = malloc(sizeof(double) * n), zmax = 1e-300;
        for (int c = 0; c < n; c++) {
            const double *qc = Q + (size_t)c * n;
            double s = 0;
            for (int j = 0; j < n; j++) {
                double a = 0;
                const double *zj = Z[k] + (size_t)j * n;
                for (int i = 0; i < n; i++) a += zj[i] * qc[i];
                t[j] = a;
            }
            for (int j = 0; j < n; j++) s += qc[j] * t[j];
            zq[c] = s; zmax = fmax(zmax, fabs(s));
        }
        int r = 0;
        for (int c = 0; c < n; c++) if (w[c] / wmax > fmax(zq[c], 0.0) / zmax) r++;
        if (r < 1) r = 1;
        Uinit[k] = malloc(sizeof(double) * (size_t)n * r);
        for (int c = 0; c < r; c++) {            /* the r largest eigenpairs */
            const int src = n - 1 - c;
            const double s = sqrt(fmax(w[src], 0.0));
            for (int i = 0; i < n; i++) Uinit[k][i + (size_t)c * n] = Q[i + (size_t)src * n] * s;
        }
        S.r[k] = r; S.nU += n * r;
        free(w); free(Q); free(zq); free(t);
    }
    const int N = S.nU + m, NR = m + S.nU;
    double Nred = m, fpk = 0;
    for (int k = 0; k < nb; k++) {
        const double n = abs(O->bs[k]), r = S.r[k];
        Nred += O->bs[k] < 0 ? r : r * r;
        if (O->bs[k] > 0) fpk += n * n * r * m + (n - r) * r * (double)m * m + 10.0 * n * n * n;
    }
    const double flops = getenv("BRISK_XO_GN") ? (double)S.nU * S.nU * m + (double)N * N * N / 3.0 + (double)m * m * S.nU
                                               : fpk + Nred * Nred * Nred / 3.0 * 2.0;
    if (verbose > 0) {
        int nr = 0; long rs = 0;
        for (int k = 0; k < nb; k++) if (O->bs[k] > 0) { nr++; rs += S.r[k]; }
        if (verbose > 1) printf("crossover: %d unknowns (%ld rank over %d SDP blocks, m %d), %.1e flops per Newton step\n", N, rs, nr, m, flops);
    }
    if (!ok || N > max_unknowns || flops > max_seconds * 3e10) {
        if (verbose > 0) printf("crossover: skipped (%s)\n", ok ? "too large for the time budget" : "eigendecomposition failed");
        goto done;
    }
    {
        double *v = malloc(sizeof(double) * N), *vn = malloc(sizeof(double) * N), *vb = malloc(sizeof(double) * N);
        double *F = malloc(sizeof(double) * NR), *Fn = malloc(sizeof(double) * NR);
        int reduced = !getenv("BRISK_XO_GN");
        /* the Gauss-Newton fallback (when the reduced Newton steps are cut short: truss5)
         * needs the N x N normal matrix */
        const double gnflops = (double)S.nU * S.nU * m + (double)N * N * N / 3.0 + (double)m * m * S.nU;
        const int gn_ok = N <= max_unknowns && 3.0 * gnflops <= max_seconds * 3e10 && (double)N * N * 16 < 4e9;   /* ~3 steps in budget */
        const size_t NH = (!reduced || gn_ok) ? (size_t)N * N : 1;
        double *H = malloc(sizeof(double) * NH), *rhs = malloc(sizeof(double) * (size_t)N), *g = malloc(sizeof(double) * (size_t)N);
        double *Bm = malloc(sizeof(double) * ((size_t)S.nU * m + 1)), *ZB = malloc(sizeof(double) * ((size_t)S.nU * m + 1));
        int maxn = 1; for (int k = 0; k < nb; k++) if (O->bs[k] > maxn) maxn = O->bs[k];
        double *Z2 = malloc(sizeof(double) * (size_t)maxn * maxn);
        double *H0 = malloc(sizeof(double) * NH), *dLM = malloc(sizeof(double) * N);
        double *pc_r = malloc(sizeof(double) * 6 * (size_t)(N + NR)), *pc_z = pc_r + (N + NR), *pc_p = pc_z + (N + NR),
               *pc_q = pc_p + (N + NR), *pc_t = pc_q + (N + NR), *pc_w = pc_t + (N + NR);
        for (int k = 0; k < nb; k++)
            memcpy(v + S.off[k], Uinit[k], sizeof(double) * (O->bs[k] < 0 ? (size_t)S.r[k] : (size_t)abs(O->bs[k]) * S.r[k]));
        memcpy(v + S.nU, yo, sizeof(double) * m);
        double nF = xo_residual(&D, &S, v, F, Xn, Zn);
        if (getenv("BRISK_XODBG")) {
            double a = 0, bq = 0;
            for (int i = 0; i < m; i++) a += F[i] * F[i];
            for (int i = m; i < NR; i++) bq += F[i] * F[i];
            printf("crossover: initial |A(UU')-b| %.2e |ZU| %.2e\n", sqrt(a), sqrt(bq));
        }
        double best = INFINITY, ebest[7], bfloor = 0;
        memcpy(vb, v, sizeof(double) * N);
        const double scale = 1.0 + O->nb1 + O->nC1;
        int slow = 0;
        for (int it = 0; it < 12; it++) {
            if (xo_now() - t0 > max_seconds) break;
            /* Gauss-Newton normal equations H d = -J'F, H = J'J (structured):
             *   H_UU = 4 B B' + Zb^2,  H_Uy = -Zb B,  H_yy = B'B,
             *   J'F  = [2 B F1 + Zb F2 ; -B' F2]
             * plus a tiny multiple of the diagonal for the gauge directions (U -> UQ).  */
            xo_B(&D, &S, v, Bm);
            if (reduced) {
                if (xo_reduced_dir(&D, &S, v, F, Zn, Bm, rhs, verbose)) break;
                if (verbose > 1) {
                    double *tq = pc_t;
                    xo_Jmv(&D, &S, Zn, Bm, rhs, tq, NULL);
                    double a1 = 0, a2 = 0, f1 = 0, f2 = 0;
                    for (int i = 0; i < m; i++) { a1 += (F[i] + tq[i]) * (F[i] + tq[i]); f1 += F[i] * F[i]; }
                    for (int i = m; i < NR; i++) { a2 += (F[i] + tq[i]) * (F[i] + tq[i]); f2 += F[i] * F[i]; }
                    printf("crossover: linearized residual |F1+J1d| %.2e (|F1| %.2e), |F2+J2d| %.2e (|F2| %.2e)\n", sqrt(a1), sqrt(f1), sqrt(a2), sqrt(f2));
                }
                goto have_dir;
            }
            {
                const double four = 4.0, one = 1.0, zero = 0.0, mone = -1.0, two = 2.0;
                const int nU = S.nU;
                memset(H, 0, sizeof(double) * (size_t)N * N);
                if (nU > 0) BL(dsyrk_)("L", "N", &nU, &m, &four, Bm, &nU, &zero, H, &N);
                /* Zb^2 on the diagonal blocks */
                for (int k = 0; k < nb; k++) {
                    const int n = abs(O->bs[k]), r = S.r[k], col = S.off[k];
                    if (O->bs[k] < 0) { for (int t = 0; t < r; t++) { double z = Zn[k][S.sup[k][t]]; H[(col + t) + (size_t)(col + t) * N] += z * z; } continue; }
                    if (r == 0) continue;
                    BL(dsyrk_)("L", "N", &n, &n, &one, Zn[k], &n, &zero, Z2, &n);   /* Z Z' = Z^2 */
                    for (int c = 0; c < r; c++)
                        for (int jj = 0; jj < n; jj++)
                            for (int ii = jj; ii < n; ii++) H[(col + c * n + ii) + (size_t)(col + c * n + jj) * N] += Z2[ii + (size_t)jj * n];
                }
                /* H_yU = -(Zb B)' : rows nU.., columns 0..nU-1 */
                xo_Zb(&D, &S, Zn, Bm, ZB, m);
                for (int i = 0; i < m; i++)
                    for (int p = 0; p < nU; p++) H[(nU + i) + (size_t)p * N] = -ZB[p + (size_t)i * nU];
                if (nU > 0) BL(dsyrk_)("L", "T", &m, &nU, &one, Bm, &nU, &zero, H + nU + (size_t)nU * N, &N);
                /* g = J'F */
                const double *F1 = F, *F2 = F + m;
                if (nU > 0) {
                    BL(dgemv_)("N", &nU, &m, &two, Bm, &nU, F1, &IONE_, &zero, g, &IONE_);
                    xo_Zb(&D, &S, Zn, F2, ZB, 1);
                    for (int p = 0; p < nU; p++) g[p] += ZB[p];
                    BL(dgemv_)("T", &nU, &m, &mone, Bm, &nU, F2, &IONE_, &zero, g + nU, &IONE_);
                } else memset(g, 0, sizeof(double) * N);
            }
            /* regularization for the gauge directions: as small as the factorization allows;
             * PCG on J'J (preconditioned by this factor) then removes its effect */
            double dmax = 0;
            for (int i = 0; i < N; i++) dmax = fmax(dmax, H[i + (size_t)i * N]);
            int info = 1;
            double lam = 1e-15 * dmax;
            for (int tr = 0; tr < 4 && info != 0; tr++, lam *= 100) {
                if (tr) for (int j = 0; j < N; j++) memcpy(H + (size_t)j * N + j, H0 + (size_t)j * N + j, sizeof(double) * (N - j));
                else for (int j = 0; j < N; j++) memcpy(H0 + (size_t)j * N + j, H + (size_t)j * N + j, sizeof(double) * (N - j));
                for (int i = 0; i < N; i++) H[i + (size_t)i * N] += lam + 1e-300;
                BL(dpotrf_)("L", &N, H, &N, &info);
            }
            if (info != 0) { if (verbose > 0) printf("crossover: normal equations not positive definite (%d)\n", info); break; }
            /* PCG on J'J d = -J'F */
            {
                int one = 1, npcg = 0;
                double *rr = pc_r, *zz = pc_z, *pp = pc_p, *qq = pc_q, *tq = pc_t;
                for (int i = 0; i < N; i++) rr[i] = -g[i];
                memset(rhs, 0, sizeof(double) * N);
                const double ng = sqrt(xsq_(N, rr));
                memcpy(zz, rr, sizeof(double) * N);
                BL(dpotrs_)("L", &N, &one, H, &N, zz, &N, &info);
                memcpy(dLM, zz, sizeof(double) * N);          /* the Levenberg-Marquardt step */
                memcpy(pp, zz, sizeof(double) * N);
                double rz = xdot_(N, rr, zz);
                for (int k = 0; k < 20; k++) {
                    xo_Jmv(&D, &S, Zn, Bm, pp, tq, NULL);
                    xo_JTmv(&D, &S, Zn, Bm, tq, qq, pc_w);
                    const double pq = xdot_(N, pp, qq);
                    if (!(pq > 0)) break;
                    const double al = rz / pq;
                    for (int i = 0; i < N; i++) { rhs[i] += al * pp[i]; rr[i] -= al * qq[i]; }
                    npcg++;
                    if (sqrt(xsq_(N, rr)) < 1e-15 * ng) break;
                    memcpy(zz, rr, sizeof(double) * N);
                    BL(dpotrs_)("L", &N, &one, H, &N, zz, &N, &info);
                    const double rz2 = xdot_(N, rr, zz), be = rz2 / rz;
                    rz = rz2;
                    for (int i = 0; i < N; i++) pp[i] = zz[i] + be * pp[i];
                }
                if (verbose > 1) printf("crossover: %d PCG steps (lambda %.0e)\n", npcg, lam / 100 / dmax);
            }
            {
            /* the refined (Gauss-Newton) step converges quadratically near a solution; the
             * regularized one is the more robust far from it (theta3): take the better */
                for (int i = 0; i < N; i++) vn[i] = v[i] + dLM[i];
                const double fl = xo_residual(&D, &S, vn, Fn, Xn, Zn);
                for (int i = 0; i < N; i++) vn[i] = v[i] + rhs[i];
                const double fr = xo_residual(&D, &S, vn, Fn, Xn, Zn);
                if (!(fr <= fl)) memcpy(rhs, dLM, sizeof(double) * N);
                if (verbose > 1) printf("crossover: |F| after the regularized step %.2e, refined %.2e\n", fl, fr);
            }
        have_dir:;
            const int rank = N;
            double step = 1.0, nFn = INFINITY;
            for (int ls = 0; ls < 20; ls++) {
                for (int i = 0; i < N; i++) vn[i] = v[i] + step * rhs[i];
                nFn = xo_residual(&D, &S, vn, Fn, Xn, Zn);
                if (nFn < nF) break;
                step *= 0.5;
            }
            if (reduced && step < 0.1 && gn_ok && best > 1e-12) {
                /* discard the step and redo it by Gauss-Newton (the last iterate is kept) */
                if (verbose > 0) printf("crossover: reduced Newton step cut to %.2g, Gauss-Newton from here\n", step);
                reduced = 0;
                xo_residual(&D, &S, v, F, Xn, Zn);          /* restore Xn, Zn, F at v */
                continue;
            }
            if (!(nFn < nF)) break;
            memcpy(v, vn, sizeof(double) * N);
            memcpy(F, Fn, sizeof(double) * NR);
            const double prev = nF;
            nF = nFn;
            double e[7];
            xo_errors(&D, Xn, v + S.nU, Zn, e);
            const double me = xo_maxerr(e);
            if (verbose > 1) printf("crossover: Newton %d |F| %.2e -> %.2e (step %.2g, rank %d of %d) max error %.1e\n", it, prev, nF, step, rank, N, me);
            if (me < best) { best = me; bfloor = g_xo_floor; memcpy(ebest, e, sizeof(e)); memcpy(vb, v, sizeof(double) * N); pd[0] = g_xo_pd[0]; pd[1] = g_xo_pd[1]; }
            if (nF < 1e-15 * scale) break;
            if (nF > 0.9 * prev) { if (++slow >= 2) break; } else slow = 0;
        }
        if (best < e0 && e0 - best > 2.0 * fmax(bfloor, floor0)) {
            xo_build_X(&D, &S, vb, Xo);
            memcpy(yo, vb + S.nU, sizeof(double) * m);
            memcpy(err_after, ebest, sizeof(ebest));
            accepted = 1;
        }
        free(v); free(vn); free(vb); free(F); free(Fn); free(H); free(rhs); free(g); free(Bm); free(ZB); free(Z2); free(H0); free(dLM); free(pc_r);
    }
done:
    if (!accepted) memcpy(err_after, err_before, sizeof(double) * 7);
    if (verbose >= 0 && (verbose > 0 || accepted))
        printf("crossover: max error %.1e -> %.1e %s (%.2fs)\n", e0, xo_maxerr(err_after), accepted ? "accepted" : "not improved, input kept", xo_now() - t0);
    for (int k = 0; k < nb; k++) { free(D.C[k]); free(Z[k]); free(Xn[k]); free(Zn[k]); free(Uinit[k]); free(S.sup[k]); }
    free(D.C); free(Z); free(Xn); free(Zn); free(Uinit); free(S.sup); free(S.r); free(S.off);
    free(D.start); free(D.cnt); free(D.idx);
    return accepted;
}
