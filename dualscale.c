/*
 * Sparse dual-scaling method (DSDP style), option -dual.
 *
 * Works only with y: Z(y) = C - A*(y) > 0, maximize b'y + mu log det Z(y). The primal
 * matrix is never formed during the iteration. Newton system:
 *     M dy = b/mu - a0,   a0 = A(Z^-1),   M_ij = tr(A_i Z^-1 A_j Z^-1)
 *
 * General block structure: Z is block-diagonal and an LP block is a diagonal SDP block, so
 * one internal block type (with O(n) arithmetic for LP) covers every problem.
 *
 * No strictly feasible start is needed: when C - A*(0) is not positive definite a
 * penalty variable r >= 0 enters as
 *     Z = C - A*(y) + r I,   maximize b'y - Gamma r
 * which is just one more constraint (-I in every block, plus a one-variable LP block for
 * r >= 0) with b = -Gamma. In primal terms it is the bound tr(X) <= Gamma, exact whenever
 * the optimal trace is below Gamma; if it stays active, Gamma grows and the solve goes on.
 *
 * Ideas that make it fast:
 *  - the Newton direction is affine in 1/mu, dy(mu) = M^-1 b / mu - M^-1 a0, so two solves
 *    give it for every mu;
 *  - optimal certificate: the implied primal point X(mu) satisfies A(X) = b exactly, its
 *    gap is linear in mu and it is PSD exactly when Z + A*(dy(mu)) is, so mu is searched
 *    for the tightest certified bound at each iterate, independently of the step;
 *  - lazy certificates: the stored bound is monotone, so the search runs only when the
 *    central-path gap estimate could improve it;
 *  - potential reduction: step toward mu = gap/(rho n) with a line search on
 *    rho n log(ub - b'y) - log det Z, log det free from the factors;
 *  - Z assembled directly into sparse Cholesky factors, Z^-1 by blocked solves;
 *  - Schur matrix factored in single precision and refined in double (the certificate
 *    needs accurate solves, and refinement gets them at half the factorization cost);
 *  - Hadamard fast path when every constraint of a block is one diagonal entry.
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int IONE = 1;
static const double DONE_ = 1.0, DZERO_ = 0.0;

static void *dm(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }
static int dchol(int n, double *A) { int info = 0; BL(dpotrf_)("L", &n, A, &n, &info); return info != 0; }

/* ------------------------------------------------------------------------ */
typedef struct {
    int lp, n, ncon;
    int *con;                  /* global ids (m = penalty) */
    SpSym **A;                 /* constraint matrices (borrowed or owned) */
    const SpSym *C;
    int alldiag;               /* every non-penalty constraint is one diagonal entry */
    int *dg_p;                 /* ... its position */
    double *dg_a;              /* ... and value */
    SChol *SC;                 /* sparse factor (SDP) or NULL (dense) */
    double *Z, *L, *Zi, *W, *T;
    int *vptr, *vcon;          /* LP: constraints by variable */
    double *vval;
    double logdet;
} DB;

static SpSym *spsym_diag(int n, double v) {
    SpSym *S = dm(sizeof(SpSym));
    S->nnz = S->ef = n;
    S->row = dm(sizeof(int) * n); S->col = dm(sizeof(int) * n); S->val = dm(sizeof(double) * n);
    S->fr = dm(sizeof(int) * n); S->fc = dm(sizeof(int) * n); S->fv = dm(sizeof(double) * n);
    for (int i = 0; i < n; i++) { S->row[i] = S->col[i] = S->fr[i] = S->fc[i] = i; S->val[i] = S->fv[i] = v; }
    return S;
}
static void spsym_diag_free(SpSym *S) {
    free(S->row); free(S->col); free(S->val); free(S->fr); free(S->fc); free(S->fv); free(S);
}

/* aggregate sparsity pattern of an SDP block -> sparse Cholesky analysis */
static SChol *db_pattern(const DB *d) {
    const int n = d->n;
    int *deg = dm(sizeof(int) * n), *cap = dm(sizeof(int) * n);
    int **nbr = dm(sizeof(int *) * n);
    for (int i = 0; i < n; i++) { cap[i] = 8; nbr[i] = dm(sizeof(int) * 8); }
    char *mark = dm(n);
    for (int t = -1; t < d->ncon; t++) {
        const SpSym *S = t < 0 ? d->C : d->A[t];
        for (int q = 0; q < S->nnz; q++) {
            int a = S->row[q], b = S->col[q];
            if (a == b) continue;
            int dup = 0;
            for (int z = 0; z < deg[a]; z++) if (nbr[a][z] == b) { dup = 1; break; }
            if (dup) continue;
            for (int s = 0; s < 2; s++) {
                int x = s ? b : a, w = s ? a : b;
                if (deg[x] == cap[x]) { cap[x] *= 2; nbr[x] = realloc(nbr[x], sizeof(int) * cap[x]); }
                nbr[x][deg[x]++] = w;
            }
        }
    }
    SChol *SC = schol_analyze_adj(n, deg, nbr, (size_t)(0.3 * (double)n * n) + 1000);
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(deg); free(cap); free(mark);
    return SC;
}

/* Z of block d at y (dense for SDP without a sparse factor, vector for LP) */
static void db_Zdense(const DB *d, const double *y, double *Z) {
    const int n = d->n;
    if (d->lp) {
        for (int p = 0; p < n; p++) Z[p] = 0;
        for (int q = 0; q < d->C->nnz; q++) Z[d->C->row[q]] += d->C->val[q];
        for (int t = 0; t < d->ncon; t++) {
            const double yt = y[d->con[t]];
            if (yt == 0) continue;
            const SpSym *S = d->A[t];
            for (int q = 0; q < S->nnz; q++) Z[S->row[q]] -= yt * S->val[q];
        }
        return;
    }
    memset(Z, 0, sizeof(double) * (size_t)n * n);
    for (int q = 0; q < d->C->ef; q++) Z[d->C->fr[q] + (size_t)d->C->fc[q] * n] += d->C->fv[q];
    for (int t = 0; t < d->ncon; t++) {
        const double yt = y[d->con[t]];
        if (yt == 0) continue;
        const SpSym *S = d->A[t];
        for (int q = 0; q < S->ef; q++) Z[S->fr[q] + (size_t)S->fc[q] * n] -= yt * S->fv[q];
    }
}

/* factor Z(y) of one block; optionally form Z^{-1}. Returns 1 if positive definite. */
static int db_factor(DB *d, const double *y, int want_inv) {
    const int n = d->n;
    const size_t nn = (size_t)n * n;
    if (d->lp) {
        db_Zdense(d, y, d->Z);
        double ld = 0;
        for (int p = 0; p < n; p++) { if (!(d->Z[p] > 0)) return 0; ld += log(d->Z[p]); }
        d->logdet = ld;
        if (want_inv) for (int p = 0; p < n; p++) d->Zi[p] = 1.0 / d->Z[p];
        return 1;
    }
    if (d->SC) {
        schol_zero(d->SC);
        for (int q = 0; q < d->C->nnz; q++) schol_add(d->SC, d->C->row[q], d->C->col[q], d->C->val[q]);
        for (int t = 0; t < d->ncon; t++) {
            const double yt = y[d->con[t]];
            if (yt == 0) continue;
            const SpSym *S = d->A[t];
            for (int q = 0; q < S->nnz; q++) schol_add(d->SC, S->row[q], S->col[q], -yt * S->val[q]);
        }
        if (schol_factor(d->SC, 0.0)) return 0;
        d->logdet = schol_logdet(d->SC);
        if (want_inv) {
            memset(d->Zi, 0, sizeof(double) * nn);
            for (int j = 0; j < n; j++) d->Zi[j + (size_t)j * n] = 1.0;
            schol_solve_many(d->SC, d->Zi, n, d->W);
        }
        return 1;
    }
    db_Zdense(d, y, d->Z);
    memcpy(d->L, d->Z, sizeof(double) * nn);
    if (dchol(n, d->L)) return 0;
    double ld = 0;
    for (int i = 0; i < n; i++) ld += 2.0 * log(d->L[i + (size_t)i * n]);
    d->logdet = ld;
    if (want_inv) {
        int info = 0;
        memcpy(d->Zi, d->L, sizeof(double) * nn);
        BL(dpotri_)("L", &n, d->Zi, &n, &info);
        if (info) return 0;
        for (int j = 0; j < n; j++)
            for (int i = j + 1; i < n; i++) d->Zi[j + (size_t)i * n] = d->Zi[i + (size_t)j * n];
    }
    return 1;
}

static int pd_all(DB *db, int nb, const double *y, double *logdet) {
    double ld = 0;
    for (int k = 0; k < nb; k++) { if (!db_factor(&db[k], y, 0)) return 0; ld += db[k].logdet; }
    if (logdet) *logdet = ld;
    return 1;
}

/* a0 = A(Z^{-1}) and M = tr(A_i Zi A_j Zi) accumulated over blocks */
static void assemble(DB *db, int nb, int mm, double *a0, double *M) {
    memset(a0, 0, sizeof(double) * mm);
    memset(M, 0, sizeof(double) * (size_t)mm * mm);
#define MADDX(i, j, v) do { int i_ = (i), j_ = (j); \
        M[(i_ > j_ ? i_ : j_) + (size_t)(i_ > j_ ? j_ : i_) * mm] += (v); } while (0)
    for (int k = 0; k < nb; k++) {
        DB *d = &db[k];
        const int n = d->n;
        const double *Zi = d->Zi;
        if (d->lp) {
            for (int t = 0; t < d->ncon; t++) {
                const SpSym *S = d->A[t];
                double s = 0;
                for (int q = 0; q < S->nnz; q++) s += S->val[q] * Zi[S->row[q]];
                a0[d->con[t]] += s;
            }
            for (int p = 0; p < n; p++) {
                const double w = Zi[p] * Zi[p];
                for (int a = d->vptr[p]; a < d->vptr[p + 1]; a++)
                    for (int b = a; b < d->vptr[p + 1]; b++) {
                        double v = d->vval[a] * d->vval[b] * w;
                        if (d->vcon[a] == d->vcon[b] && a != b) v *= 2;
                        MADDX(d->vcon[a], d->vcon[b], v);
                    }
            }
            continue;
        }
        for (int t = 0; t < d->ncon; t++) {
            const SpSym *S = d->A[t];
            double s = 0;
            for (int q = 0; q < S->ef; q++) s += S->fv[q] * Zi[S->fr[q] + (size_t)S->fc[q] * n];
            a0[d->con[t]] += s;
        }
        /* Hadamard fast path: single diagonal entries -> M_tu = a_t a_u Zi[p,q]^2 */
        if (d->alldiag) {
            for (int t = 0; t < d->ncon; t++) {
                if (d->dg_p[t] < 0) continue;
                const int pt = d->dg_p[t];
                const double at = d->dg_a[t];
                const double *zc = Zi + (size_t)pt * n;
                for (int u = t; u < d->ncon; u++) {
                    if (d->dg_p[u] < 0) continue;
                    const double z = zc[d->dg_p[u]];
                    MADDX(d->con[t], d->con[u], at * d->dg_a[u] * z * z);
                }
            }
        }
        for (int t = 0; t < d->ncon; t++) {
            const SpSym *St = d->A[t];
            if (d->alldiag && d->dg_p[t] >= 0) {
                /* pairs with non-diagonal (penalty) constraints only */
                for (int u = 0; u < d->ncon; u++) {
                    if (d->dg_p[u] >= 0) continue;
                    const SpSym *Su = d->A[u];
                    const int p = d->dg_p[t];
                    double s = 0;
                    for (int qb = 0; qb < Su->ef; qb++)
                        s += Su->fv[qb] * Zi[p + (size_t)Su->fr[qb] * n] * Zi[Su->fc[qb] + (size_t)p * n];
                    MADDX(d->con[t], d->con[u], d->dg_a[t] * s);
                }
                continue;
            }
            if (St->ef > n) {
                /* dense constraint: G = Zi A Zi by BLAS, then M_tu = <A_u, G> */
                memset(d->W, 0, sizeof(double) * (size_t)n * n);
                for (int q = 0; q < St->ef; q++) d->W[St->fr[q] + (size_t)St->fc[q] * n] = St->fv[q];
                xsymm("L", "L", &n, &n, &DONE_, Zi, &n, d->W, &n, &DZERO_, d->T, &n);
                xsymm("R", "L", &n, &n, &DONE_, Zi, &n, d->T, &n, &DZERO_, d->W, &n);
                for (int u = 0; u < d->ncon; u++) {
                    const SpSym *Su = d->A[u];
                    if (Su->ef > n && u < t) continue;
                    if (d->alldiag && d->dg_p[u] >= 0) continue;          /* done above */
                    double s = 0;
                    for (int q = 0; q < Su->ef; q++) s += Su->fv[q] * d->W[Su->fr[q] + (size_t)Su->fc[q] * n];
                    MADDX(d->con[t], d->con[u], s);
                }
                continue;
            }
            for (int u = t; u < d->ncon; u++) {
                const SpSym *Su = d->A[u];
                if (Su->ef > n) continue;
                if (d->alldiag && d->dg_p[u] >= 0) continue;
                double s = 0;
                for (int qa = 0; qa < St->ef; qa++) {
                    const int p = St->fr[qa], q = St->fc[qa];
                    double inner = 0;
                    for (int qb = 0; qb < Su->ef; qb++)
                        inner += Su->fv[qb] * Zi[q + (size_t)Su->fr[qb] * n] * Zi[Su->fc[qb] + (size_t)p * n];
                    s += St->fv[qa] * inner;
                }
                MADDX(d->con[t], d->con[u], s);
            }
        }
    }
#undef MADDX
    for (int j = 0; j < mm; j++)
        for (int i = j + 1; i < mm; i++) M[j + (size_t)i * mm] = M[i + (size_t)j * mm];
}

/* ---- Schur solves: single-precision factor, double-precision refinement */
typedef struct { int mm, is_float; double *M, *Md; float *Mf; double *r, *t; float *fr; } MS;

static int ms_factor(MS *S, int use_float) {
    const int mm = S->mm;
    const size_t sz = (size_t)mm * mm;
    S->is_float = 0;
    if (use_float) {
        for (size_t q = 0; q < sz; q++) S->Mf[q] = (float)S->M[q];
        int info = 0;
        BL(spotrf_)("L", &mm, S->Mf, &mm, &info);
        if (!info) { S->is_float = 1; return 0; }
    }
    double reg = 0;
    for (;;) {
        memcpy(S->Md, S->M, sizeof(double) * sz);
        if (reg > 0) for (int i = 0; i < mm; i++) S->Md[i + (size_t)i * mm] *= (1 + reg);
        if (!dchol(mm, S->Md)) return 0;
        reg = reg == 0 ? 1e-12 : reg * 100;
        if (reg > 1e-2) return 1;
    }
}
static void ms_base(MS *S, double *x) {
    int info = 0;
    if (S->is_float) {
        for (int i = 0; i < S->mm; i++) S->fr[i] = (float)x[i];
        BL(spotrs_)("L", &S->mm, &IONE, S->Mf, &S->mm, S->fr, &S->mm, &info);
        for (int i = 0; i < S->mm; i++) x[i] = S->fr[i];
    } else BL(dpotrs_)("L", &S->mm, &IONE, S->Md, &S->mm, x, &S->mm, &info);
}
/* solve M x = rhs to near machine precision; returns the final relative residual */
static double ms_solve(MS *S, const double *rhs, double *x) {
    const int mm = S->mm;
    memcpy(x, rhs, sizeof(double) * mm);
    ms_base(S, x);
    double nb = 0, nr = 0;
    for (int i = 0; i < mm; i++) nb += rhs[i] * rhs[i];
    nb = sqrt(nb) + 1e-300;
    for (int pass = 0; pass < 8; pass++) {
        const double mone = -1.0;
        memcpy(S->r, rhs, sizeof(double) * mm);
        BL(dsymv_)("L", &mm, &mone, S->M, &mm, x, &IONE, &DONE_, S->r, &IONE);
        nr = 0;
        for (int i = 0; i < mm; i++) nr += S->r[i] * S->r[i];
        nr = sqrt(nr);
        if (nr <= 1e-14 * nb) break;
        ms_base(S, S->r);
        for (int i = 0; i < mm; i++) x[i] += S->r[i];
    }
    return nr / nb;
}

/* LP blocks: constraint lists by variable (rebuilt when the phase-1 penalty is dropped,
 * otherwise they would still reference constraint index m) */
static void lp_incidence(DB *d) {
    free(d->vptr); free(d->vcon); free(d->vval);
    int *cnt = dm(sizeof(int) * (d->n + 1));
    memset(cnt, 0, sizeof(int) * (d->n + 1));
    for (int t = 0; t < d->ncon; t++) for (int q = 0; q < d->A[t]->nnz; q++) cnt[d->A[t]->row[q] + 1]++;
    for (int p = 0; p < d->n; p++) cnt[p + 1] += cnt[p];
    d->vptr = dm(sizeof(int) * (d->n + 1));
    memcpy(d->vptr, cnt, sizeof(int) * (d->n + 1));
    d->vcon = dm(sizeof(int) * (cnt[d->n] + 1));
    d->vval = dm(sizeof(double) * (cnt[d->n] + 1));
    int *f = dm(sizeof(int) * (d->n + 1));
    memcpy(f, cnt, sizeof(int) * d->n);
    for (int t = 0; t < d->ncon; t++)
        for (int q = 0; q < d->A[t]->nnz; q++) {
            int p = d->A[t]->row[q];
            d->vcon[f[p]] = d->con[t];
            d->vval[f[p]] = d->A[t]->val[q];
            f[p]++;
        }
    free(cnt); free(f);
}

/* implied primal point X = mu (Zi + Zi A*(dy) Zi): PSD and A(X) = b? (the certificate) */
static int verify(DB *db, int nb, int mm, const double *b, const double *dy, double mu,
                  const double *y, double gap, double *resid_out, double **Xcap, int ncap) {
    double *ax = dm(sizeof(double) * mm);
    int ok = 1;
    for (int k = 0; k < nb && ok; k++) {
        DB *d = &db[k];
        const int n = d->n;
        if (d->lp) {
            for (int p = 0; p < n; p++) {
                double s = 0;
                for (int a = d->vptr[p]; a < d->vptr[p + 1]; a++) s += d->vval[a] * dy[d->vcon[a]];
                const double x = mu * (d->Zi[p] + d->Zi[p] * d->Zi[p] * s);
                if (Xcap && k < ncap) Xcap[k][p] = x;
                if (x < 0) { ok = 0; break; }
                for (int a = d->vptr[p]; a < d->vptr[p + 1]; a++) ax[d->vcon[a]] += d->vval[a] * x;
            }
            continue;
        }
        const size_t nn = (size_t)n * n;
        double *Ad = d->W, *T = d->T, *X = d->L;
        memset(Ad, 0, sizeof(double) * nn);
        for (int t = 0; t < d->ncon; t++) {
            const SpSym *S = d->A[t];
            const double v = dy[d->con[t]];
            if (v == 0) continue;
            for (int q = 0; q < S->ef; q++) Ad[S->fr[q] + (size_t)S->fc[q] * n] += v * S->fv[q];
        }
        xsymm("L", "L", &n, &n, &DONE_, d->Zi, &n, Ad, &n, &DZERO_, T, &n);
        xsymm("R", "L", &n, &n, &DONE_, d->Zi, &n, T, &n, &DZERO_, X, &n);
        for (size_t q = 0; q < nn; q++) X[q] = mu * (d->Zi[q] + X[q]);
        if (Xcap && k < ncap) memcpy(Xcap[k], X, sizeof(double) * nn);
        for (int t = 0; t < d->ncon; t++) {
            const SpSym *S = d->A[t];
            double s = 0;
            for (int q = 0; q < S->ef; q++) s += S->fv[q] * X[S->fr[q] + (size_t)S->fc[q] * n];
            ax[d->con[t]] += s;
        }
        memcpy(T, X, sizeof(double) * nn);
        if (dchol(n, T)) ok = 0;
    }
    double res = 0, ny = 0, nbb = 0;
    for (int i = 0; i < mm; i++) { double r = ax[i] - b[i]; res += r * r; ny += y[i] * y[i]; nbb += b[i] * b[i]; }
    res = sqrt(res);
    if (resid_out) *resid_out = res;
    free(ax);
    /* the residual perturbs the bound by ||A(X)-b|| ||y||: keep it well below the gap */
    if (res * sqrt(ny) > 0.05 * fabs(gap) || res > 1e-6 * (1 + sqrt(nbb))) ok = 0;
    return ok;
}

/* ------------------------------------------------------------------------ */
int dual_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout) {
    const int m = P->m;
    const double sc = P->bs * P->cs;

    /* ---- internal blocks */
    int nb0 = P->nblk;
    DB *db = dm(sizeof(DB) * (nb0 + 1));
    for (int k = 0; k < nb0; k++) {
        const Block *B = &P->blk[k];
        DB *d = &db[k];
        d->lp = (B->type == BLK_LP);
        d->n = B->n;
        d->ncon = B->ncon;
        d->con = dm(sizeof(int) * (B->ncon + 1));
        d->A = dm(sizeof(SpSym *) * (B->ncon + 1));
        for (int t = 0; t < B->ncon; t++) { d->con[t] = B->con[t]; d->A[t] = &B->A[t]; }
        d->C = &B->C;
    }
    int nb = nb0;

    /* ---- a start: y = 0 if Z(0) > 0; for single-block diagonal problems a Gershgorin
     * shift of y; otherwise the trace penalty with r0 from a Gershgorin bound.       */
    double *y = dm(sizeof(double) * (m + 2));
    int penalty = 0;
    for (int k = 0; k < nb; k++) {
        DB *d = &db[k];
        size_t sz = d->lp ? (size_t)d->n : (size_t)d->n * d->n;
        d->Z = dm(sizeof(double) * sz); d->L = dm(sizeof(double) * sz); d->Zi = dm(sizeof(double) * sz);
        d->W = dm(sizeof(double) * sz); d->T = dm(sizeof(double) * sz);
    }
    int feasible0 = 1;
    for (int k = 0; k < nb && feasible0; k++) if (!db_factor(&db[k], y, 0)) feasible0 = 0;
    if (!feasible0 && nb == 1 && !db[0].lp) {
        DB *d = &db[0];
        int alld = 1;
        for (int t = 0; t < d->ncon; t++) if (d->A[t]->nnz != 1 || d->A[t]->row[0] != d->A[t]->col[0]) alld = 0;
        if (alld) {
            const int n = d->n;
            db_Zdense(d, y, d->Z);
            for (int t = 0; t < d->ncon; t++) {
                int i = d->A[t]->row[0];
                double rs = 0;
                for (int j = 0; j < n; j++) if (j != i) rs += fabs(d->Z[i + (size_t)j * n]);
                y[d->con[t]] = (d->Z[i + (size_t)i * n] - (rs + 1.0)) / d->A[t]->val[0];
            }
            feasible0 = db_factor(d, y, 0);
        }
    }
    double Gamma = 0;
    int mm = m;
    SpSym **pen = NULL;
    SpSym *slackA = NULL, *emptyC = NULL;
    if (!feasible0) {
        memset(y, 0, sizeof(double) * (m + 2));
        penalty = 1;
        mm = m + 1;
        /* Gershgorin lower bound on lambda_min(C) over all blocks */
        double gl = 1e300;
        for (int k = 0; k < nb; k++) {
            DB *d = &db[k];
            const int n = d->n;
            db_Zdense(d, y, d->Z);
            for (int i = 0; i < n; i++) {
                double v;
                if (d->lp) v = d->Z[i];
                else {
                    double rs = 0;
                    for (int j = 0; j < n; j++) if (j != i) rs += fabs(d->Z[i + (size_t)j * n]);
                    v = d->Z[i + (size_t)i * n] - rs;
                }
                if (v < gl) gl = v;
            }
        }
        y[m] = fmax(0.0, -gl) + 1.0;                     /* r0 */
        double ntr = 0;
        for (int k = 0; k < nb; k++) ntr += db[k].n;
        Gamma = 1e3 * (1.0 + ntr);
        pen = dm(sizeof(SpSym *) * nb);
        for (int k = 0; k < nb; k++) {
            DB *d = &db[k];
            pen[k] = spsym_diag(d->n, -1.0);
            d->A[d->ncon] = pen[k];
            d->con[d->ncon] = m;
            d->ncon++;
        }
        /* one-variable LP block for r >= 0: z = r */
        DB *s = &db[nb];
        memset(s, 0, sizeof(DB));
        s->lp = 1; s->n = 1; s->ncon = 1;
        s->con = dm(sizeof(int)); s->con[0] = m;
        s->A = dm(sizeof(SpSym *));
        slackA = spsym_diag(1, -1.0);
        s->A[0] = slackA;
        emptyC = dm(sizeof(SpSym));
        s->C = emptyC;
        s->Z = dm(sizeof(double)); s->L = dm(sizeof(double)); s->Zi = dm(sizeof(double));
        s->W = dm(sizeof(double)); s->T = dm(sizeof(double));
        nb++;
    }
    double *b = dm(sizeof(double) * (mm + 1));
    memcpy(b, P->b, sizeof(double) * m);
    if (penalty) b[m] = -Gamma;

    /* ---- per-block structure: sparse factors, LP incidence, diagonal fast path */
    double ntot = 0;
    for (int k = 0; k < nb; k++) {
        DB *d = &db[k];
        ntot += d->n;
        if (d->lp) {
            lp_incidence(d);
            continue;
        }
        d->alldiag = 1;
        d->dg_p = dm(sizeof(int) * (d->ncon + 1));
        d->dg_a = dm(sizeof(double) * (d->ncon + 1));
        int ndg = 0;
        for (int t = 0; t < d->ncon; t++) {
            const SpSym *S = d->A[t];
            if (d->con[t] == m && penalty) { d->dg_p[t] = -1; continue; }
            if (S->nnz == 1 && S->row[0] == S->col[0]) { d->dg_p[t] = S->row[0]; d->dg_a[t] = S->val[0]; ndg++; }
            else { d->alldiag = 0; d->dg_p[t] = -1; }
        }
        if (!ndg) d->alldiag = 0;
        if (!d->alldiag) for (int t = 0; t < d->ncon; t++) d->dg_p[t] = -1;
        if (d->n >= 50 && !getenv("BRISK_DUALDENSE")) d->SC = db_pattern(d);
    }
    if (par->verbose >= 0)
        printf("dual method: %d block(s), m = %d%s\n", nb, m,
               penalty ? " (trace penalty: no strictly feasible start available)" : "");

    /* ---- workspace */
    MS S = { 0 };
    S.mm = mm;
    S.M = dm(sizeof(double) * (size_t)mm * mm);
    S.Md = dm(sizeof(double) * (size_t)mm * mm);
    S.Mf = dm(sizeof(float) * (size_t)mm * mm);
    S.r = dm(sizeof(double) * mm); S.t = dm(sizeof(double) * mm); S.fr = dm(sizeof(float) * mm);
    double *a0 = dm(sizeof(double) * mm), *db_ = dm(sizeof(double) * mm), *da = dm(sizeof(double) * mm);
    double *dy = dm(sizeof(double) * mm), *ytry = dm(sizeof(double) * mm), *dycert = dm(sizeof(double) * mm);
    const int use_float = (mm >= 400) && !getenv("BRISK_DUALDOUBLE");

    double mu = 0;
    for (int k = 0; k < nb; k++) {
        DB *d = &db[k];
        db_factor(d, y, 0);
        db_Zdense(d, y, d->Z);
        for (int i = 0; i < d->n; i++) mu += d->lp ? d->Z[i] : d->Z[i + (size_t)i * d->n];
    }
    mu /= ntot;

    /* ---- phase 1: maximize -r with r free, stopping as soon as r < 0. Then the original
     * Z = C - A*(y) = Z_aug - r I is strictly positive definite, the penalty is dropped
     * and the method starts from an interior point. Only the sign of r matters, so this
     * is usually a handful of iterations; if r never goes negative the dual has no
     * interior and the trace penalty stays. */
    int phase1_its = 0, phase1_ok = 0;
    if (penalty && !getenv("BRISK_NOPHASE1")) {
        const int nb1 = nb - 1;                           /* without the r >= 0 slack */
        double *b1 = dm(sizeof(double) * mm);
        b1[m] = -1.0;
        double mu1 = mu;
        for (phase1_its = 0; phase1_its < 80; phase1_its++) {
            if (y[m] < -1e-8 * (1 + fabs(y[m]))) { phase1_ok = 1; break; }
            int okf = 1;
            for (int k = 0; k < nb1 && okf; k++) okf = db_factor(&db[k], y, 1);
            if (!okf) break;
            assemble(db, nb1, mm, a0, S.M);
            if (ms_factor(&S, use_float)) break;
            ms_solve(&S, b1, db_);
            ms_solve(&S, a0, da);
            for (int i = 0; i < mm; i++) dy[i] = db_[i] / mu1 - da[i];
            double amax = 1.0;
            int tries = 0;
            for (; tries < 60; tries++) {
                for (int i = 0; i < mm; i++) ytry[i] = y[i] + amax * dy[i];
                if (pd_all(db, nb1, ytry, NULL)) break;
                amax *= 0.6;
            }
            if (tries == 60) break;
            double a = 0.9 * amax;
            for (int i = 0; i < mm; i++) y[i] += a * dy[i];
            mu1 *= (a > 0.8 ? 0.2 : a > 0.4 ? 0.5 : 0.9);
        }
        free(b1);
        if (phase1_ok) {
            y[m] = 0;                                     /* Z_orig = Z_aug - r I > 0 */
            for (int k = 0; k < nb - 1; k++) {
                db[k].ncon--;                             /* the penalty was appended last */
                if (db[k].lp) lp_incidence(&db[k]);
            }
            nb--;
            mm = m;
            S.mm = m;
            penalty = 0;
            ntot = 0;
            for (int k = 0; k < nb; k++) ntot += db[k].n;
            mu = 0;
            for (int k = 0; k < nb; k++) {
                DB *dd = &db[k];
                db_Zdense(dd, y, dd->Z);
                for (int i = 0; i < dd->n; i++) mu += dd->lp ? dd->Z[i] : dd->Z[i + (size_t)i * dd->n];
            }
            mu /= ntot;
            memcpy(b, P->b, sizeof(double) * m);
        } else {
            if (y[m] < 1e-12) y[m] = 1e-12;               /* back inside r >= 0 */
        }
        if (par->verbose >= 0)
            printf("dual method: phase 1 %s after %d iterations\n",
                   phase1_ok ? "found a strictly feasible dual point" : "found no dual interior, keeping the trace penalty",
                   phase1_its);
    }
    const double rho_n = par->dual_rho * ntot;
    double ub = 0, mucert = 0, bty = 0, gap = 0, ubnow = 0;
    double last_res = 0;
    int have_ub = 0, status = 3, it, ncert = 0, nskip = 0, ngamma = 0, gaveup = 0;
    if (!phase1_ok && penalty) {                          /* phase 1 moved y: refresh mu */
        mu = 0;
        for (int k = 0; k < nb; k++) {
            DB *dd = &db[k];
            if (!db_factor(dd, y, 0)) { mu = -1; break; }
            db_Zdense(dd, y, dd->Z);
            for (int i = 0; i < dd->n; i++) mu += dd->lp ? dd->Z[i] : dd->Z[i + (size_t)i * dd->n];
        }
        if (mu < 0) {                                     /* not interior: restart from y = 0 */
            memset(y, 0, sizeof(double) * (m + 2));
            double gl = 1e300;
            for (int k = 0; k < nb - 1; k++) {
                DB *dd = &db[k];
                db_Zdense(dd, y, dd->Z);
                for (int i = 0; i < dd->n; i++) {
                    double v;
                    if (dd->lp) v = dd->Z[i];
                    else { double rs = 0; for (int j = 0; j < dd->n; j++) if (j != i) rs += fabs(dd->Z[i + (size_t)j * dd->n]); v = dd->Z[i + (size_t)i * dd->n] - rs; }
                    if (v < gl) gl = v;
                }
            }
            y[m] = fmax(0.0, -gl) + 1.0;
            mu = 1.0;
        } else mu /= ntot;
    }

    for (it = 0; it <= par->maxit; it++) {
        double logdet = 0;
        int okf = 1;
        for (int k = 0; k < nb && okf; k++) { okf = db_factor(&db[k], y, 1); logdet += db[k].logdet; }
        if (!okf) { status = 4; break; }
        bty = 0;
        for (int i = 0; i < mm; i++) bty += b[i] * y[i];
        assemble(db, nb, mm, a0, S.M);
        if (ms_factor(&S, use_float)) { status = 4; break; }
        ms_solve(&S, b, db_);
        ms_solve(&S, a0, da);
        double a_db = 0, a_da = 0;
        for (int i = 0; i < mm; i++) { a_db += a0[i] * db_[i]; a_da += a0[i] * da[i]; }
        const double slope = ntot - a_da;

        /* ---- certificate (lazy): gap(mu) = mu*slope + a_db, certified when
         * Z(y - dy(mu)) > 0; search for the tightest bound unless it cannot help */
        int cert_now = 0;
        const double est = mu * ntot;
        const int want = !have_ub || it < 3 || est < 0.8 * (ub - bty) ||
                         sc * est / (1 + fabs(sc * bty)) < 10 * par->tol;
        if (want) {
            for (int attempt = 0; attempt < 2 && !cert_now; attempt++) {
                double lo = 1e-16 * (1 + fabs(bty)), hi = fmax(mu * 100.0, 1.0), mtest = mu;
                int steps = par->dual_cert;
                if (attempt == 0) {
                    if (!(mucert > 0 && it > 0)) continue;
                    lo = fmax(lo, mucert / 16.0);
                    mtest = fmax(mtest, mucert / 4.0);
                    steps = par->dual_cert / 2 + 2;
                }
                double gbest = -1, mubest = 0;
#define FEAS(MU) (({ for (int i2 = 0; i2 < mm; i2++) ytry[i2] = y[i2] - (db_[i2] / (MU) - da[i2]); \
                     pd_all(db, nb, ytry, NULL); }))
                int f0 = FEAS(mtest);
                if (f0) {
                    double g0 = mtest * slope + a_db;
                    if (g0 > 0) { gbest = g0; mubest = mtest; }
                    if (slope > 0) {
                        double fl = lo, fh = mtest;
                        for (int s2 = 0; s2 < steps; s2++) {
                            double mid = sqrt(fl * fh);
                            if (FEAS(mid)) { fh = mid; double g2 = mid * slope + a_db; if (g2 > 0 && g2 < gbest) { gbest = g2; mubest = mid; } }
                            else fl = mid;
                        }
                    } else {
                        double fl = mtest, fh = hi;
                        for (int s2 = 0; s2 < steps; s2++) {
                            double mid = sqrt(fl * fh);
                            if (FEAS(mid)) { fl = mid; double g2 = mid * slope + a_db; if (g2 > 0 && g2 < gbest) { gbest = g2; mubest = mid; } }
                            else fh = mid;
                        }
                    }
                } else {
                    double mt = mtest;
                    for (int s2 = 0; s2 < steps && !f0; s2++) { mt *= 3.0; f0 = FEAS(mt); }
                    if (f0) { double g0 = mt * slope + a_db; if (g0 > 0) { gbest = g0; mubest = mt; } }
                }
#undef FEAS
                /* back off the cone boundary, where X is numerically singular */
                if (gbest > 0 && slope > 0) {
                    double mm2 = mubest * par->dual_margin;
                    if (mm2 <= mtest) { mubest = mm2; gbest = mm2 * slope + a_db; }
                }
                ncert++;
                if (gbest > 0) {
                    cert_now = 1;
                    ubnow = bty + gbest;
                    if (!have_ub || ubnow < ub) { ub = ubnow; have_ub = 1; }
                    for (int i = 0; i < mm; i++) dycert[i] = db_[i] / mubest - da[i];
                    mucert = mubest;
                }
            }
        } else nskip++;
        gap = have_ub ? ub - bty : mu * ntot;
        const double rel = sc * gap / (1 + fabs(sc * bty));
        if (par->verbose)
            printf("%3d  dual obj %+.10e   gap %.2e   mu %.2e%s%s\n", it, sc * bty, rel, mu,
                   have_ub ? "" : " (no certificate)", penalty ? "" : "");
        if (cert_now && sc * (ubnow - bty) / (1 + fabs(sc * bty)) < par->tol) {
            double res = 0;
            if (verify(db, nb, mm, b, dycert, mucert, y, ubnow - bty, &res, Xout, nb0)) {
                /* with the penalty, the bound is for the original problem only when the
                 * trace bound is inactive, i.e. r has gone to zero */
                double rsc = penalty ? y[m] * sqrt(ntot) / (1 + P->normC2) : 0;
                last_res = res;
                if (!penalty || rsc < par->tol) { gap = ubnow - bty; status = 0; break; }
                if (ngamma < 6) {
                    Gamma *= 100; b[m] = -Gamma; have_ub = 0; ngamma++;
                    if (par->verbose) printf("   trace bound active: Gamma -> %.1e\n", Gamma);
                    continue;
                }
                status = 4; break;                     /* bound keeps binding: no certificate of unboundedness */
            }
        }
        if (it == par->maxit) { status = 3; break; }
        /* Dual scaling is weak on ill-conditioned families (control, hinf, arch, qap):
         * there the iterates stay far from the central path and no certificate ever
         * appears. Give up early so the caller can fall back to the primal-dual method. */
        if (!have_ub && it >= par->dual_giveup) { status = 4; gaveup = 1; break; }

        /* ---- potential-reduction step */
        /* Toward mu = gap/(rho n) when a certified bound exists. Without one, the mu of
         * the direction stays put and is reduced after the step according to how far the
         * step went: shrinking it by a fixed factor regardless of progress let mu
         * collapse to 1e-11 while y had barely moved (control1, hinf1).               */
        double muk = have_ub ? fmax(gap / rho_n, 1e-14 * (1 + fabs(bty))) : mu;
        mu = muk;
        for (int i = 0; i < mm; i++) dy[i] = db_[i] / muk - da[i];
        double amax = 1.0;
        int tries = 0;
        for (; tries < 60; tries++) {
            for (int i = 0; i < mm; i++) ytry[i] = y[i] + amax * dy[i];
            if (pd_all(db, nb, ytry, NULL)) break;
            amax *= 0.6;
        }
        if (tries == 60) { status = 4; break; }
        const double fr[4] = { 0.98, 0.8, 0.5, 0.25 };
        double phibest = 1e300, abest = 0;
        for (int c = 0; c < 4; c++) {
            double a = amax * fr[c], bt = 0, ld = 0;
            for (int i = 0; i < mm; i++) { ytry[i] = y[i] + a * dy[i]; bt += b[i] * ytry[i]; }
            if (have_ub && ub - bt <= 0) continue;
            if (!pd_all(db, nb, ytry, &ld)) continue;
            double phi = (have_ub ? rho_n * log(ub - bt) : -rho_n * bt / (1 + fabs(bt))) - ld;
            if (phi < phibest) { phibest = phi; abest = a; }
        }
        if (abest == 0) abest = amax * 0.5;
        for (int i = 0; i < mm; i++) y[i] += abest * dy[i];
        if (!have_ub) mu *= (abest > 0.9 ? 0.3 : abest > 0.5 ? 0.6 : 0.9);
    }

    /* ---- report (original problem) */
    double bto = 0;
    for (int i = 0; i < m; i++) bto += P->b[i] * y[i];
    R->iters = it;
    R->pobj = sc * (bty + gap);
    R->dobj = sc * bto;
    R->relgap = sc * gap / (1 + fabs(sc * bty + P->obj_off));
    R->relcomp = R->relgap;
    R->pinf = last_res * P->bs / (1 + P->normb2);   /* residual of the certificate X (scaled rows) */
    R->dinf = penalty ? y[m] * sqrt(ntot) / (1 + P->normC2) : 0;
    R->err[1] = R->pinf; R->err[2] = 0; R->err[3] = 0; R->err[4] = R->dinf;
    R->err[5] = R->relgap; R->err[6] = R->relgap;
    if (yout) for (int i = 0; i < m; i++) yout[i] = P->cs * P->d[i] * y[i];
    if (Xout && status == 0)                             /* certificate X, original scale */
        for (int k = 0; k < nb0; k++) {
            const Block *B = &P->blk[k];
            size_t len = B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
            for (size_t q = 0; q < len; q++) Xout[k][q] *= P->bs;
        }
    /* main prints R->status: an earlier version never set it, so every dual run was
     * reported as OPTIMAL whatever happened */
    if (status == 0 && R->relgap > par->tol) status = 5;
    R->status = status;
    (void)gaveup;
    if (par->verbose)
        printf("   (dual method: %d certificate searches, %d skipped, %s Schur factor, %d Gamma increases)\n",
               ncert, nskip, use_float ? "single-precision" : "double", ngamma);

    const int nb_alloc = nb0 + (slackA != NULL);   /* the phase-1 slack block may have been dropped */
    for (int k = 0; k < nb_alloc; k++) {
        DB *d = &db[k];
        schol_free(d->SC);
        free(d->Z); free(d->L); free(d->Zi); free(d->W); free(d->T);
        free(d->con); free(d->A); free(d->dg_p); free(d->dg_a);
        free(d->vptr); free(d->vcon); free(d->vval);
    }
    if (pen) { for (int k = 0; k < nb0; k++) spsym_diag_free(pen[k]); free(pen); }
    if (slackA) spsym_diag_free(slackA);
    free(emptyC);
    free(db); free(y); free(b);
    free(S.M); free(S.Md); free(S.Mf); free(S.r); free(S.t); free(S.fr);
    free(a0); free(db_); free(da); free(dy); free(ytry); free(dycert);
    return status;
}
