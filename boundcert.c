/* boundcert.c - BRISK 4.39: rigorous certification of a one-sided bound (-certify).
 *
 * The C replacement of 4.38's tools/certify.py (removed). Conventions of brisk.h:
 *   (P) min <C,X> s.t. <A_i,X> = b_i, X >= 0      (D) max b'y s.t. Z = C - sum y_i A_i >= 0.
 * The problem is the one as read (a PSOrig). The data are enclosed in intervals: values read
 * from a file are correctly rounded decimals (strtod), so the exact decimal lies within one ulp
 * of the double; data given in memory (brisk_run_data) are exact. The certificate (X~ or y~) is
 * taken exactly as the doubles it is.
 *
 * p: r = b - A(X~) is enclosed with directed rounding; the exact point
 *    X* = X~ + A*(G^{-1} r), G = A A*, satisfies the equations and
 *    ||X* - X~||_F = sqrt(r' G^{-1} r) <= ||r|| / sqrt(lambda_min(G)) = delta.
 *    Certified when lambda_min(X~_k) - delta >= 0 for every block; then
 *    optimum <= <C,X~> + ||C||_F delta = U (rounded up).
 * d: Z = C - A'y~ is enclosed entrywise, Z~ its midpoint, rho its radius; split pairs (the
 *    equalities of (D)) are corrected exactly as for p (delta from their Gram matrix), their
 *    effect on block k bounded by delta (sum_i ||A_ik||_F^2)^(1/2). Certified when
 *    lambda_min(Z~_k) - ||rho_k||_F - delta N_k >= 0 for every block; then
 *    optimum >= b'y~ - ||b|| delta = L (rounded down).
 * lambda_min lower bound of a symmetric double matrix M: if LAPACK's Cholesky of
 * A = fl(M - cI) (round to nearest) succeeds, then A + E = R'R with |E| <= g |R'||R|,
 * g = gamma_{3n} (Demmel's backward error with room for blocking), ||E||_2 <= g ||R||_F^2 <=
 * g tr(A) / (1 - g), and with the rounding of the diagonal shift (u max|A_ii|):
 *    lambda_min(M) >= c - g tr(A) / (1 - g) - u max|A_ii|,
 * every term evaluated with directed rounding. Underflow is ignored (the bound adds 1e-300).
 *
 * This file is compiled with -frounding-math (Makefile): the rounding mode is changed here.
 */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "brisk.h"


static const double U = DBL_EPSILON / 2;      /* unit roundoff */

static size_t bsz_c(int bs) { return bs < 0 ? (size_t)(-bs) : (size_t)bs * (size_t)bs; }

/* the enclosure of a data value */
static inline void dint(double v, int exact, double *lo, double *hi) {
    if (exact || v == 0) { *lo = *hi = v; return; }
    *lo = nextafter(v, -INFINITY); *hi = nextafter(v, INFINITY);
}
/* [lo, hi] * x for an exact double x, with the current rounding mode set by the caller:
 * lower bound (downward mode) or upper bound (upward mode) */
static inline double mul_lo(double lo, double hi, double x) { return x >= 0 ? lo * x : hi * x; }
static inline double mul_hi(double lo, double hi, double x) { return x >= 0 ? hi * x : lo * x; }

/* a rigorous lower bound on lambda_min of the symmetric n x n double matrix M (column-major) */
static double lammin_lower(int n, const double *M) {
    if (n <= 0) return 0;
    if (n == 1) return M[0];
    fesetround(FE_TONEAREST);
    double *W = malloc(sizeof(double) * (size_t)n * n), *ev = malloc(sizeof(double) * (n + 1));
    memcpy(W, M, sizeof(double) * (size_t)n * n);
    int lw = -1, liw = -1, iwq = 0, info = 0;
    double wq = 0;
    BL(dsyevd_)("N", "L", &n, W, &n, ev, &wq, &lw, &iwq, &liw, &info);
    lw = (int)wq + 1; liw = iwq + 1;
    double *wk = malloc(sizeof(double) * lw);
    int *iw = malloc(sizeof(int) * liw);
    BL(dsyevd_)("N", "L", &n, W, &n, ev, wk, &lw, iw, &liw, &info);
    const double lam = info == 0 ? ev[0] : -INFINITY;
    free(wk); free(iw); free(ev);
    double dmax0 = 0;
    for (int i = 0; i < n; i++) dmax0 = fmax(dmax0, fabs(M[i + (size_t)i * n]));
    const double g = 3.0 * n * U / (1 - 3.0 * n * U);
    double best = -INFINITY;
    static const double fr[] = { 0.5, 0.9, 0.99, 0.0, -0.5 };
    for (int t = 0; t < 5; t++) {
        const double c = lam > 0 ? lam * fr[t] : lam * (1 + fabs(fr[t])) - 1e-12 * dmax0;
        if (!isfinite(c)) break;
        fesetround(FE_TONEAREST);
        memcpy(W, M, sizeof(double) * (size_t)n * n);
        for (int i = 0; i < n; i++) W[i + (size_t)i * n] -= c;
        BL(dpotrf_)("L", &n, W, &n, &info);
        if (info != 0) continue;
        /* the bound, with the subtracted terms rounded up and the result rounded down */
        double tr = 0, dm = 0;
        fesetround(FE_UPWARD);
        for (int i = 0; i < n; i++) {
            const double a = M[i + (size_t)i * n] - c;          /* upper bound of fl(M_ii - c) up to u: covered below */
            tr += fabs(a); dm = fmax(dm, fabs(a));
        }
        tr = tr * (1 + 4 * U) + 1e-300;
        const double e1 = g * tr / (1 - g), e2 = U * dm * (1 + 4 * U) + 1e-300;
        const double err = e1 + e2;
        fesetround(FE_DOWNWARD);
        const double bnd = c - err;
        fesetround(FE_TONEAREST);
        if (bnd > best) best = bnd;
        if (bnd > 0) break;
    }
    fesetround(FE_TONEAREST);
    free(W);
    return best;
}

/* ||M||_F rounded up */
static double fro_up(size_t len, const double *M) {
    fesetround(FE_UPWARD);
    double s = 0;
    for (size_t q = 0; q < len; q++) s += M[q] * M[q];
    s = sqrt(s) * (1 + 2 * U) + 1e-300;
    fesetround(FE_TONEAREST);
    return s;
}

/* a lower bound on lambda_min(G) for G = A W A' (W = 2 on off-diagonal SDP entries), over the
 * constraints listed in rows (all when rows == NULL), from a double G and an entrywise error
 * bound that covers the data intervals. Returns -inf when too large (m > 8000). */
static double gram_lammin_lower(const PSOrig *O, int exact, int nr, const int *rows) {
    const int m = O->m;
    if (nr > 6000) return -INFINITY;
    int *pos = malloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) pos[i] = -1;
    if (rows) for (int a = 0; a < nr; a++) pos[rows[a]] = a;
    else for (int i = 0; i < m; i++) pos[i] = i;
    /* entries grouped by position */
    size_t na = 0, *idx = malloc(sizeof(size_t) * (O->nnz + 1));
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0 && pos[O->con[q]] >= 0) idx[na++] = q;
    /* sort by (blk, jj, ii): simple insertion into buckets via qsort on a key array */
    long long *key = malloc(sizeof(long long) * (na + 1));
    size_t *ord = malloc(sizeof(size_t) * (na + 1));
    for (size_t t = 0; t < na; t++) {
        const size_t q = idx[t];
        key[t] = ((long long)O->blk[q] << 42) ^ ((long long)O->jj[q] << 21) ^ (long long)O->ii[q];
        ord[t] = t;
    }
    /* counting-free sort */
    {
        /* shell sort on ord by key (na is at most a few million) */
        size_t gap = 1;
        while (gap < na / 3) gap = 3 * gap + 1;
        for (; gap > 0; gap /= 3)
            for (size_t i = gap; i < na; i++) {
                const size_t v = ord[i]; size_t j = i;
                while (j >= gap && key[ord[j - gap]] > key[v]) { ord[j] = ord[j - gap]; j -= gap; }
                ord[j] = v;
            }
    }
    double *G = calloc((size_t)nr * nr + 1, sizeof(double)), *Ga = calloc((size_t)nr * nr + 1, sizeof(double));
    int kmax = 1;
    int *cnt = calloc(nr + 1, sizeof(int));
    for (size_t t = 0; t < na; t++) cnt[pos[O->con[idx[t]]]]++;
    for (int a = 0; a < nr; a++) if (cnt[a] > kmax) kmax = cnt[a];
    free(cnt);
    for (size_t s = 0; s < na;) {
        size_t e = s + 1;
        while (e < na && key[ord[e]] == key[ord[s]]) e++;
        const size_t q0 = idx[ord[s]];
        const double w = (O->bs[O->blk[q0]] > 0 && O->ii[q0] != O->jj[q0]) ? 2.0 : 1.0;
        for (size_t a = s; a < e; a++)
            for (size_t b = s; b < e; b++) {
                const size_t qa = idx[ord[a]], qb = idx[ord[b]];
                const int ca = pos[O->con[qa]], cb = pos[O->con[qb]];
                G[ca + (size_t)cb * nr] += w * O->v[qa] * O->v[qb];
                Ga[ca + (size_t)cb * nr] += w * fabs(O->v[qa]) * fabs(O->v[qb]);
            }
        s = e;
    }
    free(idx); free(key); free(ord); free(pos);
    /* ||G~ - G||_2 <= ||G~ - G||_F <= (gamma_{k+2} + 6u) ||(|A| W |A'|)||_F (data intervals: 2u
     * relative per factor; the sums: gamma of the longest inner product) */
    const double gk = (kmax + 2.0) * U / (1 - (kmax + 2.0) * U);
    const double dG = (gk + 6 * U + (exact ? 0 : 4 * U)) * fro_up((size_t)nr * nr, Ga) * 1.01;
    free(Ga);
    const double lg = lammin_lower(nr, G);
    free(G);
    fesetround(FE_DOWNWARD);
    const double r = lg - dG;
    fesetround(FE_TONEAREST);
    return r;
}

/* 5.2 (hpsolve.c): the two rigorous lower bounds that the high-precision certifier takes
 * from double arithmetic. The data of O are the doubles nearest to the decimals of the
 * file (exact = 0) or the data themselves (exact = 1).
 * lambda_min(A A*) over all rows; lambda_min of the Gram matrix of the pair equalities
 * (the coefficients of y at the LP entries (PK[p], P1[p])). <= 0: no bound. */
double brisk_gram_lammin_lower(const PSOrig *O, int exact) {
    const int rm0 = fegetround();
    fesetround(FE_TONEAREST);
    const double r = gram_lammin_lower(O, exact, O->m, NULL);
    fesetround(rm0);
    return r;
}
double brisk_pairs_lammin_lower(const PSOrig *O, int exact, int np, const int *PK, const int *P1) {
    const int m = O->m;
    if (np <= 0) return INFINITY;
    if (np > 6000) return -INFINITY;
    const int rm0 = fegetround();
    fesetround(FE_TONEAREST);
    /* (blk, index) -> pair: a sorted key list */
    long long *key = malloc(sizeof(long long) * (np + 1)); int *ord = malloc(sizeof(int) * (np + 1));
    for (int p = 0; p < np; p++) { key[p] = ((long long)PK[p] << 32) | (unsigned)P1[p]; ord[p] = p; }
    for (int a = 1; a < np; a++) { const int v = ord[a]; int b = a; while (b > 0 && key[ord[b - 1]] > key[v]) { ord[b] = ord[b - 1]; b--; } ord[b] = v; }
    /* the rows by constraint: column lists (constraint i: pairs and values) */
    int *cnt = calloc(m + 2, sizeof(int)), *rowc = calloc(np + 1, sizeof(int));
    size_t ne = 0;
    int *ep = malloc(sizeof(int) * (O->nnz + 1)), *ec = malloc(sizeof(int) * (O->nnz + 1)); double *ev = malloc(sizeof(double) * (O->nnz + 1));
    for (size_t q = 0; q < O->nnz; q++) {
        const int k = O->blk[q];
        if (O->bs[k] >= 0 || O->con[q] < 0) continue;
        const long long kq = ((long long)k << 32) | (unsigned)O->ii[q];
        int lo = 0, hi = np - 1, f = -1;
        while (lo <= hi) { const int mid = (lo + hi) / 2; if (key[ord[mid]] == kq) { f = ord[mid]; break; } if (key[ord[mid]] < kq) lo = mid + 1; else hi = mid - 1; }
        if (f < 0) continue;
        ep[ne] = f; ec[ne] = O->con[q]; ev[ne] = O->v[q]; ne++; cnt[O->con[q] + 1]++; rowc[f]++;
    }
    free(key); free(ord);
    for (int i = 0; i < m; i++) cnt[i + 1] += cnt[i];
    int *pos = malloc(sizeof(int) * (m + 1)), *sp = malloc(sizeof(int) * (ne + 1)); double *sv = malloc(sizeof(double) * (ne + 1));
    memcpy(pos, cnt, sizeof(int) * (m + 1));
    for (size_t e = 0; e < ne; e++) { sp[pos[ec[e]]] = ep[e]; sv[pos[ec[e]]] = ev[e]; pos[ec[e]]++; }
    free(ep); free(ec); free(ev); free(pos);
    double *Gp = calloc((size_t)np * np + 1, sizeof(double)), *Ga = calloc((size_t)np * np + 1, sizeof(double));
    int kmax = 1;
    for (int p = 0; p < np; p++) if (rowc[p] > kmax) kmax = rowc[p];
    free(rowc);
    for (int i = 0; i < m; i++)
        for (int a = cnt[i]; a < cnt[i + 1]; a++)
            for (int b = cnt[i]; b < cnt[i + 1]; b++) {
                Gp[sp[a] + (size_t)sp[b] * np] += sv[a] * sv[b];
                Ga[sp[a] + (size_t)sp[b] * np] += fabs(sv[a] * sv[b]);
            }
    free(cnt); free(sp); free(sv);
    const double gk = (kmax + 2.0) * U / (1 - (kmax + 2.0) * U);
    const double dG = (gk + 6 * U + (exact ? 0 : 4 * U)) * fro_up((size_t)np * np, Ga) * 1.01;
    const double lg0 = lammin_lower(np, Gp);
    free(Gp); free(Ga);
    fesetround(FE_DOWNWARD);
    const double lg = lg0 - dG;
    fesetround(rm0);
    return lg;
}

int brisk_certify_orig(const PSOrig *O, int side, double **X, const double *y, int exact,
                       double *bound, char *reason, size_t rlen) {
    const int m = O->m;
    *bound = side == 1 ? INFINITY : -INFINITY;
    if (reason && rlen) reason[0] = 0;
    const int rm0 = fegetround();
    if (side == 1) {
        if (!X) { if (reason) snprintf(reason, rlen, "no X"); return 0; }
        /* the certificate is the symmetric matrix of its upper triangle (what -x writes and
         * what A(X) reads); the lower triangle is overwritten in a copy */
        double **Xs = bound_alloc_blocks(O);
        for (int k = 0; k < O->nblk; k++) {
            const int n = abs(O->bs[k]);
            memcpy(Xs[k], X[k], sizeof(double) * bsz_c(O->bs[k]));
            if (O->bs[k] > 0) for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) Xs[k][j + (size_t)i * n] = Xs[k][i + (size_t)j * n];
        }
        X = Xs;
        /* r = b - A(X~): lower and upper bounds */
        double *rl = malloc(sizeof(double) * (m + 1)), *ru = malloc(sizeof(double) * (m + 1));
        double cxu = 0;
        for (int dir = 0; dir < 2; dir++) {
            fesetround(dir == 0 ? FE_DOWNWARD : FE_UPWARD);
            double *r = dir == 0 ? rl : ru;
            for (int i = 0; i < m; i++) {
                double lo, hi;
                dint(O->b[i], exact, &lo, &hi);
                r[i] = dir == 0 ? lo : hi;
            }
            double cx = 0;
            for (size_t q = 0; q < O->nnz; q++) {
                const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
                const double x = O->bs[k] < 0 ? X[k][i] : X[k][i + (size_t)j * n];
                const double w = (O->bs[k] > 0 && i != j) ? 2.0 : 1.0;
                double lo, hi;
                dint(O->v[q], exact, &lo, &hi);
                if (O->con[q] < 0) {
                    /* C = -F0 is how PSOrig stores C? PSOrig's v for con < 0 is C itself */
                    if (dir == 1) cx += w * mul_hi(lo, hi, x);
                } else {
                    /* r_i - w a x: lower bound subtracts the upper product, upper the lower */
                    if (dir == 0) r[O->con[q]] -= w * mul_hi(lo, hi, x);
                    else r[O->con[q]] -= w * mul_lo(lo, hi, x);
                }
            }
            if (dir == 1) cxu = cx;
        }
        fesetround(FE_UPWARD);
        double r2 = 0;
        for (int i = 0; i < m; i++) { const double a = fmax(fabs(rl[i]), fabs(ru[i])); r2 += a * a; }
        const double rn = sqrt(r2) * (1 + 2 * U);
        fesetround(FE_TONEAREST);
        free(rl); free(ru);
        double delta = 0;
        if (rn > 0) {
            const double lg = gram_lammin_lower(O, exact, m, NULL);
            if (!(lg > 0)) {
                if (reason) snprintf(reason, rlen, "lambda_min(A A*) not bounded away from 0 (dependent constraints, or m > 6000)");
                fesetround(rm0);
                bound_free_blocks(Xs, O->nblk);
                return 0;
            }
            fesetround(FE_UPWARD);
            delta = rn / sqrt(fmax(lg, 0) * (1 - 2 * U)) * (1 + 4 * U);
            fesetround(FE_TONEAREST);
        }
        double worst = INFINITY;
        for (int k = 0; k < O->nblk; k++) {
            const int n = abs(O->bs[k]);
            double lb;
            if (O->bs[k] < 0) { lb = INFINITY; for (int i = 0; i < n; i++) lb = fmin(lb, X[k][i]); }
            else lb = lammin_lower(n, X[k]);
            fesetround(FE_DOWNWARD);
            const double v = lb - delta;
            fesetround(FE_TONEAREST);
            worst = fmin(worst, v);
        }
        /* ||C||_F upper bound */
        fesetround(FE_UPWARD);
        double c2 = 0;
        for (size_t q = 0; q < O->nnz; q++)
            if (O->con[q] < 0) {
                const int k = O->blk[q];
                const double w = (O->bs[k] > 0 && O->ii[q] != O->jj[q]) ? 2.0 : 1.0;
                const double a = fabs(O->v[q]) * (exact ? 1 : 1 + 2 * U);
                c2 += w * a * a;
            }
        const double cF = sqrt(c2) * (1 + 2 * U);
        const double Ub = cxu + cF * delta;
        fesetround(FE_TONEAREST);
        *bound = Ub;
        fesetround(rm0);
        bound_free_blocks(Xs, O->nblk);
        if (!(worst >= 0)) { if (reason) snprintf(reason, rlen, "lambda_min(X) - delta = %.2e < 0", worst); return 0; }
        return 1;
    }
    /* ---- d ---- */
    if (!y) { if (reason) snprintf(reason, rlen, "no y"); return 0; }
    /* split pairs and the equalities they stand for */
    int *P1 = NULL, *P2 = NULL, *PK = NULL;
    const int np = bound_find_pairs(O, &P1, &P2, &PK);
    char **pm = NULL;
    if (np > 0) {
        pm = calloc(O->nblk + 1, sizeof(char *));
        for (int p = 0; p < np; p++) {
            if (!pm[PK[p]]) pm[PK[p]] = calloc(-O->bs[PK[p]] + 1, 1);
            pm[PK[p]][P1[p]] = pm[PK[p]][P2[p]] = 1;
        }
    }
    /* Z enclosures */
    double **Zl = bound_alloc_blocks(O), **Zu = bound_alloc_blocks(O);
    for (int dir = 0; dir < 2; dir++) {
        fesetround(dir == 0 ? FE_DOWNWARD : FE_UPWARD);
        double **Zd = dir == 0 ? Zl : Zu;
        for (size_t q = 0; q < O->nnz; q++) {
            const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
            double lo, hi, c;
            dint(O->v[q], exact, &lo, &hi);
            if (O->con[q] < 0) c = dir == 0 ? lo : hi;                      /* + C */
            else c = dir == 0 ? -mul_hi(lo, hi, y[O->con[q]]) : -mul_lo(lo, hi, y[O->con[q]]);   /* - y_i A_i */
            if (O->bs[k] < 0) Zd[k][i] += c;
            else { Zd[k][i + (size_t)j * n] += c; if (i != j) Zd[k][j + (size_t)i * n] += c; }
        }
    }
    fesetround(FE_TONEAREST);
    /* the pair equalities: residuals of Z at the first half (want 0), their Gram matrix */
    double delta = 0;
    if (np > 0) {
        fesetround(FE_UPWARD);
        double r2 = 0;
        for (int p = 0; p < np; p++) {
            const int k = PK[p], i = P1[p];
            const double a = fmax(fabs(Zl[k][i]), fabs(Zu[k][i]));
            r2 += a * a;
        }
        const double rn = sqrt(r2) * (1 + 2 * U);
        fesetround(FE_TONEAREST);
        if (rn > 0) {
            /* rows of the pair equalities: build a small PSOrig-free Gram: coefficients of y at
             * (PK[p], P1[p]) */
            double *Ap = calloc((size_t)np * m + 1, sizeof(double));
            for (size_t q = 0; q < O->nnz; q++) {
                const int k = O->blk[q];
                if (O->bs[k] >= 0 || O->con[q] < 0) continue;
                for (int p = 0; p < np; p++)
                    if (PK[p] == k && P1[p] == O->ii[q]) Ap[p + (size_t)O->con[q] * np] += O->v[q];
            }
            double *Gp = calloc((size_t)np * np + 1, sizeof(double)), *Ga = calloc((size_t)np * np + 1, sizeof(double));
            int kmax = 1;
            for (int p = 0; p < np; p++) { int c = 0; for (int i = 0; i < m; i++) if (Ap[p + (size_t)i * np] != 0) c++; if (c > kmax) kmax = c; }
            for (int i = 0; i < m; i++)
                for (int a = 0; a < np; a++) {
                    const double va = Ap[a + (size_t)i * np];
                    if (va == 0) continue;
                    for (int b = 0; b < np; b++) { Gp[a + (size_t)b * np] += va * Ap[b + (size_t)i * np]; Ga[a + (size_t)b * np] += fabs(va * Ap[b + (size_t)i * np]); }
                }
            free(Ap);
            const double gk = (kmax + 2.0) * U / (1 - (kmax + 2.0) * U);
            const double dG = (gk + 6 * U + (exact ? 0 : 4 * U)) * fro_up((size_t)np * np, Ga) * 1.01;
            const double lg0 = lammin_lower(np, Gp);
            free(Gp); free(Ga);
            fesetround(FE_DOWNWARD);
            const double lg = lg0 - dG;
            fesetround(FE_TONEAREST);
            if (!(lg > 0)) {
                if (reason) snprintf(reason, rlen, "the pair equalities are dependent");
                goto fail_d;
            }
            fesetround(FE_UPWARD);
            delta = rn / sqrt(lg * (1 - 2 * U)) * (1 + 4 * U);
            fesetround(FE_TONEAREST);
        }
    }
    {
        /* per block: N_k = (sum_i ||A_ik||_F^2)^(1/2) (LP: per entry) */
        double *Nk = calloc(O->nblk + 1, sizeof(double));
        double **Ne = calloc(O->nblk + 1, sizeof(double *));
        fesetround(FE_UPWARD);
        for (size_t q = 0; q < O->nnz; q++) {
            if (O->con[q] < 0) continue;
            const int k = O->blk[q];
            const double a = fabs(O->v[q]) * (exact ? 1 : 1 + 2 * U);
            if (O->bs[k] < 0) {
                if (!Ne[k]) Ne[k] = calloc(-O->bs[k] + 1, sizeof(double));
                Ne[k][O->ii[q]] += a * a;
            } else Nk[k] += (O->ii[q] != O->jj[q] ? 2.0 : 1.0) * a * a;
        }
        fesetround(FE_TONEAREST);
        double worst = INFINITY;
        for (int k = 0; k < O->nblk && worst >= 0; k++) {
            const int n = abs(O->bs[k]);
            if (O->bs[k] < 0) {
                for (int i = 0; i < n; i++) {
                    if (pm && pm[k] && pm[k][i]) continue;      /* exactly zero at the corrected y */
                    fesetround(FE_UPWARD);
                    const double sh = delta * sqrt(Ne[k] ? Ne[k][i] : 0.0) * (1 + 2 * U);
                    fesetround(FE_DOWNWARD);
                    const double v = Zl[k][i] - sh;
                    fesetround(FE_TONEAREST);
                    worst = fmin(worst, v);
                }
                continue;
            }
            const size_t len = bsz_c(O->bs[k]);
            double *Zm = malloc(sizeof(double) * len), *rho = malloc(sizeof(double) * len);
            fesetround(FE_TONEAREST);
            for (size_t q = 0; q < len; q++) Zm[q] = 0.5 * Zl[k][q] + 0.5 * Zu[k][q];   /* any double works: rho covers it */
            fesetround(FE_UPWARD);
            for (size_t q = 0; q < len; q++) rho[q] = fmax(fabs(Zu[k][q] - Zm[q]), fabs(Zm[q] - Zl[k][q]));
            fesetround(FE_TONEAREST);
            /* the midpoint must be symmetric: Zl/Zu are, and so is the midpoint */
            const double lb = lammin_lower(n, Zm);
            const double rf = fro_up(len, rho);
            fesetround(FE_UPWARD);
            const double sh = rf + delta * sqrt(Nk[k]) * (1 + 2 * U);
            fesetround(FE_DOWNWARD);
            const double v = lb - sh;
            fesetround(FE_TONEAREST);
            worst = fmin(worst, v);
            free(Zm); free(rho);
        }
        free(Nk);
        for (int k = 0; k < O->nblk; k++) free(Ne[k]);
        free(Ne);
        /* L = b'y - ||b|| delta, rounded down */
        fesetround(FE_DOWNWARD);
        double by = 0;
        for (int i = 0; i < m; i++) { double lo, hi; dint(O->b[i], exact, &lo, &hi); by += mul_lo(lo, hi, y[i]); }
        fesetround(FE_UPWARD);
        double b2 = 0;
        for (int i = 0; i < m; i++) { const double a = fabs(O->b[i]) * (exact ? 1 : 1 + 2 * U); b2 += a * a; }
        const double sb = sqrt(b2) * (1 + 2 * U) * delta;
        fesetround(FE_DOWNWARD);
        *bound = by - sb;
        fesetround(FE_TONEAREST);
        bound_free_blocks(Zl, O->nblk); bound_free_blocks(Zu, O->nblk);
        if (pm) { for (int k = 0; k < O->nblk; k++) free(pm[k]); free(pm); }
        free(P1); free(P2); free(PK);
        fesetround(rm0);
        if (!(worst >= 0)) { if (reason) snprintf(reason, rlen, "lambda_min(Z) lower bound %.2e < 0", worst); return 0; }
        return 1;
    }
fail_d:
    bound_free_blocks(Zl, O->nblk); bound_free_blocks(Zu, O->nblk);
    if (pm) { for (int k = 0; k < O->nblk; k++) free(pm[k]); free(pm); }
    free(P1); free(P2); free(PK);
    fesetround(rm0);
    return 0;
}
