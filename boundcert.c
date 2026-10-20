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
static double g_lam_est = NAN;      /* the last lammin_lower's eigenvalue estimate (LAPACK, double) */
static double lammin_lower(int n, const double *M) {
    g_lam_est = n == 1 ? M[0] : NAN;
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
    g_lam_est = lam;
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

/* The same bound for a large sparse G (all rows; m > 6000): G and the error matrix |A| W |A'|
 * are formed by rows on their sparsity pattern, lambda_min is estimated by inverse iteration with
 * a sparse Cholesky factor (sparsechol.c: fill-reducing order, supernodes, BLAS panels) and the
 * bound is the dense one: the factorization of fl(G~ - c I) runs to completion without any pivot
 * replacement, so G~ - c I + E is positive semidefinite with ||E||_2 <= g tr / (1 - g),
 * g = gamma_{3m} (the bound on |E| <= g |R'||R| holds for any order of the operations and any
 * elimination order). Returns -inf when the factor does not fit (fill above 3e8 entries). */
/* Beyond double (-boundprec): a check that fails only by the rounding allowance of the double
 * Cholesky (about gamma_{3n}(u) tr|M|, 2e-9 at n = 2544 on a unit diagonal) is repeated in mpx
 * arithmetic (hpsolve.c, brisk_lammin_ge_mp): automatically 192 bits, then 384 when that
 * allowance still matters; or the bits asked for; 0: double only. */
static int g_bprec = -1, g_bverb = 0;
void brisk_bound_prec_set(int bits, int verbose) {
    /* (19 October 2026) the multiprecision lambda_min proof needs at least 128 bits (below, its rounding unit is 1 and
     * the proof would hold for anything): -boundprec 1..127 is raised to 128 with a note */
    if (bits > 0 && bits < 128) {
        if (verbose >= 0) printf("note: -boundprec %d is below 128 bits, the least a lambda_min proof can use; 128 bits are used\n", bits);
        bits = 128;
    }
    g_bprec = bits; g_bverb = verbose;
}
/* the rounding allowance of a lambda_min proof of M at `bits` (0: double), roughly (a choice, not a bound) */
static double lam_allow(int n, const double *M, int bits) {
    double tr = 0;
    for (int i = 0; i < n; i++) tr += fabs(M[i + (size_t)i * n]);
    const double u = bits > 0 ? ldexp(1.0, -64 * ((bits + 63) / 64 - 1)) : U;
    return 2.5 * (3.0 * n + 10.0) * u * tr;
}
/* lambda_min(M) >= need proved beyond double; est is the double estimate of lambda_min (the
 * proof is not tried when est is below need by more than its error, about n u tr|M|: then
 * lambda_min < need in any precision) */
static int lammin_ge_hi(int n, const double *M, double need, double est, const char *what) {
    if (g_bprec == 0 || n <= 1 || !(need > -INFINITY)) return 0;
    if (!(est + 2.0 * lam_allow(n, M, 0) >= need)) return 0;
    const int bl[2] = { g_bprec > 0 ? g_bprec : 192, 384 };
    const int nb = g_bprec > 0 ? 1 : 2;
    for (int a = 0; a < nb; a++) {
        if (a > 0 && lam_allow(n, M, bl[a - 1]) < 1e-3 * fabs(need) && need != 0) break;   /* more bits cannot help */
        const double t0 = wtime();
        const int r = brisk_lammin_ge_mp(n, M, need, bl[a]);
        if (g_bverb > 1) printf("     %s (n = %d): lambda_min >= %.3e in %d-bit arithmetic: %s (%.2fs)\n", what, n, need, bl[a],
                                r > 0 ? "proved" : r < 0 ? "does not fit in the memory" : "not proved", wtime() - t0);
        if (r != 0) return r > 0;
    }
    return 0;
}
/* the best lower bound of lambda_min(M) below 0 that the higher precision proves: 0, or a small
 * negative value, else lb (the double one) */
static double lammin_lower_hi(int n, const double *M, double lb, double est, const char *what) {
    if (lb >= 0 || g_bprec == 0) return lb;
    double s = 0;
    for (int i = 0; i < n; i++) s = fmax(s, fabs(M[i + (size_t)i * n]));
    const double tries[4] = { 0.0, -ldexp(s, -60), -ldexp(s, -40), -ldexp(s, -24) };
    for (int t = 0; t < 4; t++) {
        if (!(tries[t] > lb)) break;
        if (lammin_ge_hi(n, M, tries[t], est, what)) return tries[t];
    }
    return lb;
}

static double gram_lammin_lower_sparse(const PSOrig *O, int exact) {
    const int m = O->m;
    size_t na = 0, *idx = malloc(sizeof(size_t) * (O->nnz + 1));
    for (size_t q = 0; q < O->nnz; q++) if (O->con[q] >= 0) idx[na++] = q;
    long long *key = malloc(sizeof(long long) * (na + 1));
    size_t *ord = malloc(sizeof(size_t) * (na + 1));
    for (size_t t = 0; t < na; t++) {
        const size_t q = idx[t];
        key[t] = ((long long)O->blk[q] << 42) ^ ((long long)O->jj[q] << 21) ^ (long long)O->ii[q];
        ord[t] = t;
    }
    {
        size_t gap = 1;
        while (gap < na / 3) gap = 3 * gap + 1;
        for (; gap > 0; gap /= 3)
            for (size_t i = gap; i < na; i++) {
                const size_t v = ord[i]; size_t j = i;
                while (j >= gap && key[ord[j - gap]] > key[v]) { ord[j] = ord[j - gap]; j -= gap; }
                ord[j] = v;
            }
    }
    /* groups (positions): gs[p] .. gs[p+1] in ord; the entries of a row: (group, value) */
    size_t ng = 0, *gs = malloc(sizeof(size_t) * (na + 2)), *gof = malloc(sizeof(size_t) * (na + 1));
    for (size_t s = 0; s < na;) {
        size_t e = s + 1;
        while (e < na && key[ord[e]] == key[ord[s]]) e++;
        for (size_t a = s; a < e; a++) gof[ord[a]] = ng;
        gs[ng++] = s; s = e;
    }
    gs[ng] = na;
    free(key);
    size_t *rp = calloc((size_t)m + 2, sizeof(size_t)), *rt = malloc(sizeof(size_t) * (na + 1));
    for (size_t t = 0; t < na; t++) rp[O->con[idx[t]] + 2]++;
    int kmax = 1;
    for (int i = 0; i < m; i++) { if ((long long)rp[i + 2] > kmax) kmax = (int)rp[i + 2]; rp[i + 2] += rp[i + 1]; }
    for (size_t t = 0; t < na; t++) rt[rp[O->con[idx[t]] + 1]++] = t;
    /* the lower triangle by rows: columns b < a, values of G and of |A| W |A'|; the diagonal apart */
    size_t cap = 4 * (size_t)m + 1024, nz = 0;
    size_t *lp = malloc(sizeof(size_t) * ((size_t)m + 1));
    int *lj = malloc(sizeof(int) * cap);
    double *lv = malloc(sizeof(double) * cap), *dg = calloc((size_t)m + 1, sizeof(double));
    double *acc = calloc((size_t)m + 1, sizeof(double)), *aca = calloc((size_t)m + 1, sizeof(double));
    int *mark = malloc(sizeof(int) * ((size_t)m + 1)), *list = malloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    double fa2 = 0;                       /* ||(|A| W |A'|)||_F^2, rounded up (the sums of each entry to nearest: covered by the factor below) */
    for (int a = 0; a < m; a++) {
        lp[a] = nz;
        int nl = 0;
        for (size_t u = rp[a]; u < rp[a + 1]; u++) {
            const size_t t = rt[u], q = idx[t], g = gof[t];
            const double w = (O->bs[O->blk[q]] > 0 && O->ii[q] != O->jj[q]) ? 2.0 : 1.0, va = O->v[q];
            for (size_t e = gs[g]; e < gs[g + 1]; e++) {
                const size_t qb = idx[ord[e]];
                const int b = O->con[qb];
                if (b > a) continue;
                if (mark[b] != a) { mark[b] = a; list[nl++] = b; acc[b] = 0; aca[b] = 0; }
                acc[b] += w * va * O->v[qb]; aca[b] += w * fabs(va) * fabs(O->v[qb]);
            }
        }
        if (nz + (size_t)nl + 1 > cap) { cap = 2 * cap + (size_t)nl; lj = realloc(lj, sizeof(int) * cap); lv = realloc(lv, sizeof(double) * cap); }
        fesetround(FE_UPWARD);
        for (int c = 0; c < nl; c++) {
            const int b = list[c];
            if (b == a) { dg[a] = acc[b]; fa2 += aca[b] * aca[b]; }
            else { lj[nz] = b; lv[nz] = acc[b]; nz++; fa2 += 2.0 * (aca[b] * aca[b]); }
        }
        fesetround(FE_TONEAREST);
    }
    lp[m] = nz;
    free(acc); free(aca); free(mark); free(list); free(rp); free(rt); free(gs); free(gof); free(idx); free(ord);
    fesetround(FE_UPWARD);
    const double faF = sqrt(fa2) * (1 + 2 * U) + 1e-300;
    fesetround(FE_TONEAREST);
    const double gk = (kmax + 2.0) * U / (1 - (kmax + 2.0) * U);
    const double dG = (gk + 6 * U + (exact ? 0 : 4 * U)) * faF * 1.01;
    /* the adjacency (both triangles) and the analysis */
    int *deg = calloc((size_t)m + 1, sizeof(int)), **nbr = malloc(sizeof(int *) * ((size_t)m + 1));
    for (int a = 0; a < m; a++) for (size_t u = lp[a]; u < lp[a + 1]; u++) { deg[a]++; deg[lj[u]]++; }
    int *pool = malloc(sizeof(int) * (2 * nz + 1));
    { size_t o = 0; for (int a = 0; a < m; a++) { nbr[a] = pool + o; o += (size_t)deg[a]; deg[a] = 0; } }
    for (int a = 0; a < m; a++) for (size_t u = lp[a]; u < lp[a + 1]; u++) { const int b = lj[u]; nbr[a][deg[a]++] = b; nbr[b][deg[b]++] = a; }
    SChol *S = schol_analyze_adj(m, deg, nbr, (size_t)3e8);
    free(deg); free(nbr); free(pool);
    double best = -INFINITY;
    if (S) {
        schol_set_tinypiv(S, 0.0);
        double dmax0 = 0, lam = 0;
        for (int i = 0; i < m; i++) dmax0 = fmax(dmax0, fabs(dg[i]));
        const double g = 3.0 * m * U / (1 - 3.0 * m * U);
        static const double fr[] = { 0.0, 0.5, 0.2, 0.05 };       /* 0: the factor for the estimate */
        for (int t = 0; t < 4; t++) {
            const double c = lam * fr[t];
            if (t > 0 && !(c > 0)) break;
            schol_zero(S);
            for (int a = 0; a < m; a++) {
                schol_add(S, (size_t)a, (size_t)a, dg[a] - c);
                for (size_t u = lp[a]; u < lp[a + 1]; u++) schol_add(S, (size_t)a, (size_t)lj[u], lv[u]);
            }
            if (schol_factor(S, 0.0) != 0 || schol_ntiny(S) != 0) { if (t == 0) break; continue; }
            if (t == 0) {       /* inverse iteration: lam ~ lambda_min (an estimate; nothing below relies on it) */
                double *x = malloc(sizeof(double) * ((size_t)m + 1)), *wk = malloc(sizeof(double) * ((size_t)m + 1));
                unsigned long long sd = 88172645463325252ULL;
                for (int i = 0; i < m; i++) { sd ^= sd << 13; sd ^= sd >> 7; sd ^= sd << 17; x[i] = (double)(sd % 2001) / 1000.0 - 1.0; }
                for (int it = 0; it < 40; it++) {
                    double nx = 0; for (int i = 0; i < m; i++) nx += x[i] * x[i];
                    nx = sqrt(nx); if (!(nx > 0)) break;
                    for (int i = 0; i < m; i++) x[i] /= nx;
                    schol_solve(S, x, wk);
                    double ny = 0; for (int i = 0; i < m; i++) ny += x[i] * x[i];
                    lam = 1.0 / sqrt(ny);
                }
                free(x); free(wk);
                if (!(lam > 0) || !isfinite(lam)) break;
                continue;
            }
            double tr = 0, dm = 0;
            fesetround(FE_UPWARD);
            for (int i = 0; i < m; i++) { const double a = dg[i] - c; tr += fabs(a); dm = fmax(dm, fabs(a)); }
            tr = tr * (1 + 4 * U) + 1e-300;
            const double e1 = g * tr / (1 - g), e2 = U * dm * (1 + 4 * U) + 1e-300;
            const double err = e1 + e2 + dG;
            fesetround(FE_DOWNWARD);
            const double bnd = c - err;
            fesetround(FE_TONEAREST);
            if (getenv("BRISK_BOUNDDBG")) printf("   [sparse Gram bound: m %d, nnz(G) %zu, nnz(L) %zu, lambda_min ~ %.3e, shift %.3e, error terms %.2e %.2e %.2e -> %.3e]\n", m, nz + (size_t)m, schol_nnz(S), lam, c, e1, e2, dG, bnd);
            if (bnd > best) best = bnd;
            if (bnd > 0) break;
        }
        schol_free(S);
    }
    free(lp); free(lj); free(lv); free(dg);
    return best;
}

/* a lower bound on lambda_min(G) for G = A W A' (W = 2 on off-diagonal SDP entries), over the
 * constraints listed in rows (all when rows == NULL), from a double G and an entrywise error
 * bound that covers the data intervals. Above 6000 rows: the sparse path when all the
 * constraints are taken, -inf for a subset. */
static double gram_lammin_lower(const PSOrig *O, int exact, int nr, const int *rows) {
    const int m = O->m;
    if (nr > 6000 || (getenv("BRISK_GRAMSPARSE") && rows == NULL && nr == m)) return rows == NULL && nr == m ? gram_lammin_lower_sparse(O, exact) : -INFINITY;
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
                    /* (19 October 2026: the product of the negated interval [-hi, -lo] added, so that it rounds in the
                     * pass's direction; "r -= mul_hi" under FE_DOWNWARD rounded the upper product down, an enclosure that
                     * could miss r by an ulp per term) */
                    if (dir == 0) r[O->con[q]] += w * mul_lo(-hi, -lo, x);
                    else r[O->con[q]] += w * mul_hi(-hi, -lo, x);
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
                if (reason) snprintf(reason, rlen, "lambda_min(A A*) not bounded away from 0 (dependent constraints, or a sparse factor that does not fit)");
                fesetround(rm0);
                bound_free_blocks(Xs, O->nblk);
                return 0;
            }
            fesetround(FE_UPWARD);
            delta = rn / sqrt(fmax(lg, 0) * (1 - 2 * U)) * (1 + 4 * U);
            fesetround(FE_TONEAREST);
        }
        double worst = INFINITY;
        double *lbk = malloc(sizeof(double) * (O->nblk + 1)), *estk = malloc(sizeof(double) * (O->nblk + 1));
        int hopeless = 0;
        for (int k = 0; k < O->nblk; k++) {
            const int n = abs(O->bs[k]);
            if (O->bs[k] < 0) { lbk[k] = INFINITY; for (int i = 0; i < n; i++) lbk[k] = fmin(lbk[k], X[k][i]); estk[k] = lbk[k]; }
            else { lbk[k] = lammin_lower(n, X[k]); estk[k] = g_lam_est; }
            /* (a block that fails beyond its rounding error fails in any precision) */
            if (lbk[k] < delta && (O->bs[k] < 0 || !(estk[k] + 2.0 * lam_allow(n, X[k], 0) >= delta))) hopeless = 1;
        }
        for (int k = 0; k < O->nblk; k++) {
            const int n = abs(O->bs[k]);
            fesetround(FE_DOWNWARD);
            double v = lbk[k] - delta;
            fesetround(FE_TONEAREST);
            if (v < 0 && !hopeless && O->bs[k] > 0 && lammin_ge_hi(n, X[k], delta, estk[k], "certificate (P), margin")) v = 0;
            worst = fmin(worst, v);
        }
        free(lbk); free(estk);
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
            else c = dir == 0 ? mul_lo(-hi, -lo, y[O->con[q]]) : mul_hi(-hi, -lo, y[O->con[q]]);   /* - y_i A_i (the negated
                                                                     * interval, so that the product rounds outward) */
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
        /* two passes: in double, then (when every failing block may pass with more digits)
         * the blocks that failed by their rounding allowance only, in more precision */
        unsigned char *retry = calloc(O->nblk + 1, 1);
        int hopeless = 0;
        for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) { if (hopeless || worst >= 0 || g_bprec == 0) break; worst = INFINITY; }
        for (int k = 0; k < O->nblk && (pass == 1 || !hopeless); k++) {
            const int n = abs(O->bs[k]);
            if (pass == 1 && O->bs[k] < 0) continue;            /* (passed in the first pass) */
            if (O->bs[k] < 0) {
                for (int i = 0; i < n; i++) {
                    if (pm && pm[k] && pm[k][i]) continue;      /* exactly zero at the corrected y */
                    fesetround(FE_UPWARD);
                    const double sh = delta * sqrt(Ne[k] ? Ne[k][i] : 0.0) * (1 + 2 * U);
                    fesetround(FE_DOWNWARD);
                    const double v = Zl[k][i] - sh;
                    fesetround(FE_TONEAREST);
                    worst = fmin(worst, v);
                    if (v < 0) hopeless = 1;
                }
                continue;
            }
            if (pass == 1 && !retry[k]) continue;
            const size_t len = bsz_c(O->bs[k]);
            double *Zm = malloc(sizeof(double) * len), *rho = malloc(sizeof(double) * len);
            fesetround(FE_TONEAREST);
            for (size_t q = 0; q < len; q++) Zm[q] = 0.5 * Zl[k][q] + 0.5 * Zu[k][q];   /* any double works: rho covers it */
            fesetround(FE_UPWARD);
            for (size_t q = 0; q < len; q++) rho[q] = fmax(fabs(Zu[k][q] - Zm[q]), fabs(Zm[q] - Zl[k][q]));
            fesetround(FE_TONEAREST);
            /* the midpoint must be symmetric: Zl/Zu are, and so is the midpoint */
            const double lb = lammin_lower(n, Zm), est = g_lam_est;
            const double rf = fro_up(len, rho);
            fesetround(FE_UPWARD);
            const double sh = rf + delta * sqrt(Nk[k]) * (1 + 2 * U);
            fesetround(FE_DOWNWARD);
            double v = lb - sh;
            fesetround(FE_TONEAREST);
            if (pass == 0 && v < 0) { if (!(est + 2.0 * lam_allow(n, Zm, 0) >= sh)) hopeless = 1; else retry[k] = 1; }
            if (pass == 1) { if (lammin_ge_hi(n, Zm, sh, est, "certificate (D), margin")) v = 0; else hopeless = 1; }
            worst = fmin(worst, v);
            free(Zm); free(rho);
            if (pass == 1 && hopeless) break;
        }
        }
        /* (after the second pass worst is the smallest v of the retried blocks; the others passed) */
        free(retry);
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

/* ---------------------------------------------------------------------------------------
 * A-priori bounds for the bound mode (used when the check above fails), after C. Jansson,
 * D. Chaykin, C. Keil, "Rigorous error bounds for the optimal value in semidefinite
 * programming", SIAM J. Numer. Anal. 46 (2007):
 * p: for every y feasible for (D) and any symmetric X~, with r = b - A(X~),
 *      b'y = <C,X~> - <X~, Z(y)> + r'y <= <C,X~> - sum_k min(0, lambda_min(X~_k)) tr Z_k(y) + sum_i |r_i| |y_i|,
 *    so with a-priori bounds |y_i| <= yb_i and tr Z_k(y) <= zb_k on the whole feasible set of (D)
 *      d* <= <C,X~> + sum_i |r_i| yb_i - sum_k min(0, lambda_min(X~_k)) zb_k
 *    (LP entries: - sum_j min(0, x_j) zb_j). No margin of X~ is needed, so the bound also holds
 *    when (D) has no interior and the optimal X is not attained. It bounds the optimal value of
 *    (D), d* = sup b'y (that of (P) as well when there is no duality gap).
 * d: for every X feasible for (P), <C,X> = b'y~ + <Z(y~), X> >= b'y~ + sum_k min(0, lambda_min(Z_k(y~))) tr X_k,
 *    so with tr X_k <= xb_k on the feasible set of (P): p* = inf <C,X> >= b'y~ + sum_k min(0, ...) xb_k.
 * The a-priori bounds are found from the problem (simple cases only):
 *   (D): interval bounds of y propagated through what Z(y) >= 0 implies, Z_jj(y) >= 0 (linear in
 *        y) and |Z_jk(y)| <= sqrt(max Z_jj max Z_kk) over the current bounds, to a fixed point;
 *        tr Z_k <= the sum of the upper bounds of its diagonal. A moment relaxation with a ball
 *        or box constraint among its localizing matrices (the Archimedean condition written in
 *        the relaxation) has bounded moments that this finds (the localizing diagonal
 *        1 - y_{2e_i} >= 0, then y_{2a+2e_i} <= y_{2a}, and the moment matrix's 2 x 2 minors).
 *   (P): the equations whose entries are all diagonal (or linear) entries with positive
 *        coefficients bound each of them by b_i / a (max-cut: diag(X) = 1; tr X = 1).
 * Everything is evaluated with directed rounding on the data enclosures of dint(). */

typedef struct { int blk, i, j, diag; size_t t0, t1; double clo, chi; } PrPos;
static int g_prior_cmp_mode;
static const PSOrig *g_prior_O;
static int prior_cmp(const void *a, const void *b) {
    const PSOrig *O = g_prior_O;
    const size_t p = *(const size_t *)a, q = *(const size_t *)b;
    const int dp = O->bs[O->blk[p]] < 0 || O->ii[p] == O->jj[p], dq = O->bs[O->blk[q]] < 0 || O->ii[q] == O->jj[q];
    (void)g_prior_cmp_mode;
    if (dp != dq) return dq - dp;                       /* the diagonals (and linear entries) first */
    if (O->blk[p] != O->blk[q]) return O->blk[p] - O->blk[q];
    if (O->ii[p] != O->ii[q]) return O->ii[p] - O->ii[q];
    if (O->jj[p] != O->jj[q]) return O->jj[p] - O->jj[q];
    return p < q ? -1 : p > q;
}
/* a in [alo, ahi] (one sign), y in [l, u]: the range of a y in the current rounding mode */
static inline double tmin(double alo, double ahi, double l, double u) { return alo > 0 ? (l >= 0 ? alo * l : ahi * l) : (u >= 0 ? alo * u : ahi * u); }
static inline double tmax(double alo, double ahi, double l, double u) { return alo > 0 ? (u >= 0 ? ahi * u : alo * u) : (l >= 0 ? ahi * l : alo * l); }

/* interval products, differences and quotients with directed rounding (finite operands) */
static void iv_mul(double al, double ah, double bl, double bh, double *lo, double *hi) {
    fesetround(FE_DOWNWARD);
    *lo = fmin(fmin(al * bl, al * bh), fmin(ah * bl, ah * bh));
    fesetround(FE_UPWARD);
    *hi = fmax(fmax(al * bl, al * bh), fmax(ah * bl, ah * bh));
}
static void iv_sub(double al, double ah, double bl, double bh, double *lo, double *hi) {
    fesetround(FE_DOWNWARD); *lo = al - bh;
    fesetround(FE_UPWARD); *hi = ah - bl;
}
/* [al, ah] / [dl, dh] for a divisor interval of one sign */
static void iv_div(double al, double ah, double dl, double dh, double *lo, double *hi) {
    fesetround(FE_DOWNWARD); const double rl = 1.0 / dh;
    fesetround(FE_UPWARD); const double rh = 1.0 / dl;
    iv_mul(al, ah, rl, rh, lo, hi);
}
typedef struct { int va, vb; size_t pi; } PrPair;
static int prpair_cmp(const void *a, const void *b) {
    const PrPair *p = a, *q = b;
    if (p->va != q->va) return p->va < q->va ? -1 : 1;
    if (p->vb != q->vb) return p->vb < q->vb ? -1 : 1;
    return p->pi < q->pi ? -1 : p->pi > q->pi;
}
/* The bounds of y over the feasible set of (D): yl, yu (m), and the upper bounds of the
 * diagonals: zdiag[k][i] (per block, n entries; LP blocks: per entry). Returns the number of
 * y with both bounds finite. */
static int prior_propagate(const PSOrig *O, int exact, int verbose, double *yl, double *yu, double **zdiag) {
    const int m = O->m;
    const double t0 = wtime();
    for (int i = 0; i < m; i++) { yl[i] = -INFINITY; yu[i] = INFINITY; }
    for (int k = 0; k < O->nblk; k++) { const int n = abs(O->bs[k]); for (int i = 0; i < n; i++) zdiag[k][i] = INFINITY; }
    size_t *ord = malloc(sizeof(size_t) * (O->nnz + 1));
    PrPos *P = malloc(sizeof(PrPos) * (O->nnz + 1));
    int *tc = malloc(sizeof(int) * (O->nnz + 1));
    double *tlo = malloc(sizeof(double) * (O->nnz + 1)), *thi = malloc(sizeof(double) * (O->nnz + 1));
    if (!ord || !P || !tc || !tlo || !thi) {          /* (the review, 19 October 2026: allocations checked) no bounds */
        free(ord); free(P); free(tc); free(tlo); free(thi);
        if (verbose > 1) printf("     a-priori bounds: out of memory\n");
        return 0;
    }
    for (size_t q = 0; q < O->nnz; q++) ord[q] = q;
    g_prior_O = O;
    qsort(ord, O->nnz, sizeof(size_t), prior_cmp);
    /* positions and their terms */
    size_t np = 0, nt = 0, kmax = 1;
    for (size_t a = 0; a < O->nnz;) {
        size_t e = a;
        const int k = O->blk[ord[a]], i = O->ii[ord[a]], j = O->jj[ord[a]];
        PrPos *p = &P[np++];
        p->blk = k; p->i = i; p->j = j; p->diag = O->bs[k] < 0 || i == j; p->t0 = nt; p->clo = 0; p->chi = 0;
        fesetround(FE_DOWNWARD); double cl = 0; fesetround(FE_UPWARD); double ch = 0;
        while (e < O->nnz && O->blk[ord[e]] == k && O->ii[ord[e]] == i && O->jj[ord[e]] == j) {
            const size_t q = ord[e++];
            double lo, hi;
            dint(O->v[q], exact, &lo, &hi);
            if (O->con[q] < 0) { fesetround(FE_DOWNWARD); cl += lo; fesetround(FE_UPWARD); ch += hi; }
            else { tc[nt] = O->con[q]; tlo[nt] = lo; thi[nt] = hi; nt++; }
        }
        fesetround(FE_TONEAREST);
        p->clo = cl; p->chi = ch; p->t1 = nt;
        if (p->t1 - p->t0 > kmax) kmax = p->t1 - p->t0;
        a = e;
    }
    free(ord);
    double *mn_dn = malloc(sizeof(double) * kmax), *mx_up = malloc(sizeof(double) * kmax);
    double *mn_up = malloc(sizeof(double) * kmax), *mx_dn = malloc(sizeof(double) * kmax);
    double *rmin = malloc(sizeof(double) * kmax), *rmax = malloc(sizeof(double) * kmax);
    PrPair *pr = malloc(sizeof(PrPair) * (np + 1));
    if (!mn_dn || !mx_up || !mn_up || !mx_dn || !rmin || !rmax || !pr) {
        free(P); free(tc); free(tlo); free(thi); free(mn_dn); free(mx_up); free(mn_up); free(mx_dn); free(rmin); free(rmax); free(pr);
        for (int k = 0; k < O->nblk; k++) { const int n = abs(O->bs[k]); for (int i = 0; i < n; i++) zdiag[k][i] = INFINITY; }
        if (verbose > 1) printf("     a-priori bounds: out of memory\n");
        return 0;
    }
    /* positions without data are zero: a diagonal without data is 0 */
    for (int k = 0; k < O->nblk; k++) if (O->bs[k] > 0) { const int n = O->bs[k]; for (int i = 0; i < n; i++) zdiag[k][i] = 0; }
    for (size_t pi = 0; pi < np; pi++) if (P[pi].diag) zdiag[P[pi].blk][P[pi].i] = INFINITY;
    /* Off-diagonal positions with two terms, grouped by their pair of y. The permutation-symmetry
     * split writes moments that appear nowhere else as y_a + y_b in one block and y_a - y_b in
     * the other: each entry alone has two unbounded terms and the row-wise propagation below
     * bounds neither, but two such entries form a 2 x 2 system that bounds both (ni35 order 3:
     * 240 of 30305 y left unbounded, and with them the bound of side p). */
    size_t npr = 0;
    for (size_t pi = 0; pi < np; pi++) {
        const PrPos *p = &P[pi];
        if (p->diag || p->t1 - p->t0 != 2) continue;
        int a = tc[p->t0], b = tc[p->t0 + 1];
        if (a == b) continue;
        if (a > b) { const int t = a; a = b; b = t; }
        pr[npr].va = a; pr[npr].vb = b; pr[npr].pi = pi; npr++;
    }
    qsort(pr, npr, sizeof(PrPair), prpair_cmp);
    int pass;
    size_t nchg = 0;
    for (pass = 0; pass < 60; pass++) {
        nchg = 0;
        double big = 0;
        for (size_t pi = 0; pi < np; pi++) {
            PrPos *p = &P[pi];
            const size_t k0 = p->t0, kn = p->t1 - p->t0;
            double L = -INFINITY, Ub = INFINITY;
            if (p->diag) Ub = p->chi;                                  /* sum a y <= c (Z_jj >= 0) */
            else {
                const double zi = zdiag[p->blk][p->i], zj = zdiag[p->blk][p->j];
                if (!(zi < INFINITY && zj < INFINITY)) continue;
                fesetround(FE_UPWARD);
                const double s = sqrt(fmax(zi, 0) * fmax(zj, 0));
                Ub = p->chi + s;
                fesetround(FE_DOWNWARD);
                L = p->clo - s;
            }
            if (kn == 0) {                                              /* a constant entry: a diagonal is its value */
                fesetround(FE_TONEAREST);
                if (p->diag && p->chi < zdiag[p->blk][p->i]) { zdiag[p->blk][p->i] = p->chi; nchg++; }
                continue;
            }
            /* the ranges of the terms and their sums, in both directions */
            int ninf_lo = 0, ninf_hi = 0; size_t ilo = 0, ihi = 0;
            fesetround(FE_DOWNWARD);
            double Smin = 0;
            for (size_t t = 0; t < kn; t++) {
                const int v = tc[k0 + t];
                mn_dn[t] = tmin(tlo[k0 + t], thi[k0 + t], yl[v], yu[v]); mx_dn[t] = tmax(tlo[k0 + t], thi[k0 + t], yl[v], yu[v]);
                if (isinf(mn_dn[t])) { ninf_lo++; ilo = t; } else Smin += mn_dn[t];
            }
            fesetround(FE_UPWARD);
            double Smax = 0;
            for (size_t t = 0; t < kn; t++) {
                const int v = tc[k0 + t];
                mn_up[t] = tmin(tlo[k0 + t], thi[k0 + t], yl[v], yu[v]); mx_up[t] = tmax(tlo[k0 + t], thi[k0 + t], yl[v], yu[v]);
                if (isinf(mx_up[t])) { ninf_hi++; ihi = t; } else Smax += mx_up[t];
            }
            if (p->diag && ninf_lo == 0) {                              /* max Z_jj = c - min sum a y */
                const double z = p->chi - Smin;
                double *zd = &zdiag[p->blk][p->i];
                if (z < *zd) { if (*zd < INFINITY) big = fmax(big, (*zd - z) / (1 + fabs(z))); else big = 1; *zd = z; nchg++; }
            }
            /* the rest of the sum without term t */
            fesetround(FE_DOWNWARD);
            for (size_t t = 0; t < kn; t++)
                rmin[t] = ninf_lo == 0 ? Smin - mn_up[t] : (ninf_lo == 1 && ilo == t ? Smin : -INFINITY);
            fesetround(FE_UPWARD);
            for (size_t t = 0; t < kn; t++)
                rmax[t] = ninf_hi == 0 ? Smax - mx_dn[t] : (ninf_hi == 1 && ihi == t ? Smax : INFINITY);
            /* upper bounds of y (rounded up): from sum <= Ub with a > 0, from sum >= L with a < 0 */
            for (size_t t = 0; t < kn; t++) {
                const int v = tc[k0 + t]; const double alo = tlo[k0 + t], ahi = thi[k0 + t];
                double nb = INFINITY;
                if (alo > 0 && Ub < INFINITY && rmin[t] > -INFINITY) { const double R = Ub - rmin[t]; nb = R >= 0 ? R / alo : R / ahi; }
                else if (ahi < 0 && L > -INFINITY && rmax[t] < INFINITY) {
                    fesetround(FE_DOWNWARD); const double R = L - rmax[t]; fesetround(FE_UPWARD);
                    nb = R >= 0 ? R / alo : R / ahi;
                }
                if (nb < yu[v]) { if (yu[v] < INFINITY) big = fmax(big, (yu[v] - nb) / (1 + fabs(nb))); else big = 1; yu[v] = nb; nchg++; }
            }
            /* lower bounds of y (rounded down) */
            fesetround(FE_DOWNWARD);
            for (size_t t = 0; t < kn; t++) {
                const int v = tc[k0 + t]; const double alo = tlo[k0 + t], ahi = thi[k0 + t];
                double nb = -INFINITY;
                if (alo > 0 && L > -INFINITY && rmax[t] < INFINITY) { const double R = L - rmax[t]; nb = R >= 0 ? R / ahi : R / alo; }
                else if (ahi < 0 && Ub < INFINITY && rmin[t] > -INFINITY) {
                    fesetround(FE_UPWARD); const double R = Ub - rmin[t]; fesetround(FE_DOWNWARD);
                    nb = R >= 0 ? R / ahi : R / alo;
                }
                if (nb > yl[v]) { if (yl[v] > -INFINITY) big = fmax(big, (nb - yl[v]) / (1 + fabs(nb))); else big = 1; yl[v] = nb; nchg++; }
            }
            fesetround(FE_TONEAREST);
        }
        /* the 2 x 2 systems: a1 u + a2 v in E_p and b1 u + b2 v in E_q, with E = c +- sqrt(max Z_ii max Z_jj) */
        for (size_t g = 0; g < npr;) {
            size_t h = g;
            while (h < npr && pr[h].va == pr[g].va && pr[h].vb == pr[g].vb) h++;
            const int va = pr[g].va, vb = pr[g].vb;
            int done = 0;
            const size_t hc = h - g > 32 ? g + 32 : h;               /* at most 32 x 31 / 2 systems per pair of y */
            for (size_t x = g; x < hc && !done; x++) for (size_t z = x + 1; z < hc && !done; z++) {
                double co[2][4], el[2], eh[2];
                int ok = 1;
                for (int r = 0; r < 2 && ok; r++) {
                    const PrPos *p = &P[pr[r == 0 ? x : z].pi];
                    const double zi = zdiag[p->blk][p->i], zj = zdiag[p->blk][p->j];
                    if (!(zi < INFINITY && zj < INFINITY)) { ok = 0; break; }
                    fesetround(FE_UPWARD); const double sq = sqrt(fmax(zi, 0) * fmax(zj, 0)); eh[r] = p->chi + sq;
                    fesetround(FE_DOWNWARD); el[r] = p->clo - sq;
                    const int first_a = tc[p->t0] == va;
                    const size_t ta = p->t0 + (first_a ? 0 : 1), tb = p->t0 + (first_a ? 1 : 0);
                    co[r][0] = tlo[ta]; co[r][1] = thi[ta]; co[r][2] = tlo[tb]; co[r][3] = thi[tb];
                }
                fesetround(FE_TONEAREST);
                if (!ok) continue;
                double d1l, d1h, d2l, d2h, dl, dh;
                iv_mul(co[0][0], co[0][1], co[1][2], co[1][3], &d1l, &d1h);
                iv_mul(co[0][2], co[0][3], co[1][0], co[1][1], &d2l, &d2h);
                iv_sub(d1l, d1h, d2l, d2h, &dl, &dh);
                fesetround(FE_TONEAREST);
                if (!(dl > 0 || dh < 0)) continue;                     /* singular (or not provably regular) */
                double t1l, t1h, t2l, t2h, nl, nh, ul, uh, vl, vh;
                iv_mul(co[1][2], co[1][3], el[0], eh[0], &t1l, &t1h);   /* u = (b2 E_p - a2 E_q) / det */
                iv_mul(co[0][2], co[0][3], el[1], eh[1], &t2l, &t2h);
                iv_sub(t1l, t1h, t2l, t2h, &nl, &nh);
                iv_div(nl, nh, dl, dh, &ul, &uh);
                iv_mul(co[0][0], co[0][1], el[1], eh[1], &t1l, &t1h);   /* v = (a1 E_q - b1 E_p) / det */
                iv_mul(co[1][0], co[1][1], el[0], eh[0], &t2l, &t2h);
                iv_sub(t1l, t1h, t2l, t2h, &nl, &nh);
                iv_div(nl, nh, dl, dh, &vl, &vh);
                fesetround(FE_TONEAREST);
                const int vv[2] = { va, vb };
                const double lo[2] = { ul, vl }, hi[2] = { uh, vh };
                for (int r = 0; r < 2; r++) {
                    const int v = vv[r];
                    if (!(isfinite(lo[r]) && isfinite(hi[r]))) continue;
                    if (hi[r] < yu[v]) { if (yu[v] < INFINITY) big = fmax(big, (yu[v] - hi[r]) / (1 + fabs(hi[r]))); else big = 1; yu[v] = hi[r]; nchg++; }
                    if (lo[r] > yl[v]) { if (yl[v] > -INFINITY) big = fmax(big, (lo[r] - yl[v]) / (1 + fabs(lo[r]))); else big = 1; yl[v] = lo[r]; nchg++; }
                }
                done = 1;
            }
            g = h;
        }
        fesetround(FE_TONEAREST);
        if (nchg == 0 || big < 1e-9) break;
        if (wtime() - t0 > 60) {                                       /* every pass leaves valid bounds: stop on time */
            if (verbose > 1) printf("     a-priori bounds: propagation stopped after %d passes (60 s)\n", pass + 1);
            break;
        }
    }
    free(pr);
    fesetround(FE_TONEAREST);
    int nfin = 0;
    for (int i = 0; i < m; i++) nfin += yl[i] > -INFINITY && yu[i] < INFINITY;
    if (verbose > 1) {
        double ym = 0; for (int i = 0; i < m; i++) if (yl[i] > -INFINITY && yu[i] < INFINITY) ym = fmax(ym, fmax(fabs(yl[i]), fabs(yu[i])));
        printf("     a-priori bounds: %d of %d y bounded (largest |y| %.2e), %zu positions, %d passes, %.2fs\n", nfin, m, ym, np, pass + 1, wtime() - t0);
    }
    free(P); free(tc); free(tlo); free(thi); free(mn_dn); free(mx_up); free(mn_up); free(mx_dn); free(rmin); free(rmax);
    return nfin;
}

/* side p with given a-priori bounds: the bound for X~, or +inf with the reason */
static double prior_eval_p(const PSOrig *O, double **X, int exact, const double *yl, const double *yu, double **zd,
                           int verbose, char *reason, size_t rlen) {
    const int m = O->m;
    double **Xs = bound_alloc_blocks(O);
    for (int k = 0; k < O->nblk; k++) {
        const int n = abs(O->bs[k]);
        memcpy(Xs[k], X[k], sizeof(double) * bsz_c(O->bs[k]));
        if (O->bs[k] > 0) for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) Xs[k][j + (size_t)i * n] = Xs[k][i + (size_t)j * n];
    }
    double *rl = malloc(sizeof(double) * (m + 1)), *ru = malloc(sizeof(double) * (m + 1));
    double cxu = 0;
    for (int dir = 0; dir < 2; dir++) {
        fesetround(dir == 0 ? FE_DOWNWARD : FE_UPWARD);
        double *r = dir == 0 ? rl : ru;
        for (int i = 0; i < m; i++) { double lo, hi; dint(O->b[i], exact, &lo, &hi); r[i] = dir == 0 ? lo : hi; }
        double cx = 0;
        for (size_t q = 0; q < O->nnz; q++) {
            const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
            const double x = O->bs[k] < 0 ? Xs[k][i] : Xs[k][i + (size_t)j * n];
            const double w = (O->bs[k] > 0 && i != j) ? 2.0 : 1.0;
            double lo, hi;
            dint(O->v[q], exact, &lo, &hi);
            if (O->con[q] < 0) { if (dir == 1) cx += w * mul_hi(lo, hi, x); }
            else if (dir == 0) r[O->con[q]] += w * mul_lo(-hi, -lo, x);     /* (the negated interval: rounds outward) */
            else r[O->con[q]] += w * mul_hi(-hi, -lo, x);
        }
        if (dir == 1) cxu = cx;
    }
    fesetround(FE_UPWARD);
    double Ub = cxu;
    int ok = 1;
    for (int i = 0; i < m && ok; i++) {
        const double a = fmax(fabs(rl[i]), fabs(ru[i]));
        if (a == 0) continue;
        const double yb = fmax(fabs(yl[i]), fabs(yu[i]));
        if (!(yb < INFINITY)) { ok = 0; if (reason) snprintf(reason, rlen, "no a-priori bound of y_%d (its residual is %.1e)", i + 1, a); break; }
        Ub += a * yb;
    }
    const double rterm = Ub - cxu;
    fesetround(FE_TONEAREST);
    double lmin_worst = INFINITY;
    for (int k = 0; k < O->nblk && ok; k++) {
        const int n = abs(O->bs[k]);
        if (O->bs[k] < 0) {
            for (int i = 0; i < n && ok; i++) {
                if (Xs[k][i] >= 0) continue;
                if (!(zd[k][i] < INFINITY)) { ok = 0; if (reason) snprintf(reason, rlen, "no a-priori bound of entry %d of linear block %d", i + 1, k + 1); break; }
                fesetround(FE_UPWARD); Ub += -Xs[k][i] * fmax(zd[k][i], 0); fesetround(FE_TONEAREST);
            }
            continue;
        }
        /* In the diagonal congruence S X~ S, S = diag(2^-e_j) with 4^e_j about X~_jj (exact in
         * binary): <X~, Z> = <S X~ S, S^-1 Z S^-1> >= lambda_min(S X~ S) sum_j 4^e_j Z_jj, and the
         * rigorous lambda_min of S X~ S has an error of the order of n u, not of u tr X~ (a
         * certificate without a margin is large: tr X~ = 7e8 on a moment relaxation of m = 15,026) */
        const size_t len = bsz_c(O->bs[k]);
        int *e = malloc(sizeof(int) * (n + 1));
        double *W = malloc(sizeof(double) * len);
        for (int i = 0; i < n; i++) { const double d = Xs[k][i + (size_t)i * n]; e[i] = d > 0 && isfinite(d) ? (int)lround(0.5 * log2(d)) : 0; }
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) W[i + (size_t)j * n] = ldexp(Xs[k][i + (size_t)j * n], -e[i] - e[j]);
        const double lb0 = lammin_lower(n, W), est = g_lam_est;
        const double lb = lammin_lower_hi(n, W, lb0, est, "a-priori bound (P)");
        free(W);
        lmin_worst = fmin(lmin_worst, lb);
        if (lb >= 0) { free(e); continue; }
        fesetround(FE_UPWARD);
        double zb = 0;
        for (int i = 0; i < n; i++) zb += ldexp(fmax(zd[k][i], 0), 2 * e[i]);
        fesetround(FE_TONEAREST);
        free(e);
        if (!(zb < INFINITY)) { ok = 0; if (reason) snprintf(reason, rlen, "no a-priori bound of tr Z of block %d", k + 1); break; }
        fesetround(FE_UPWARD); Ub += -lb * zb; fesetround(FE_TONEAREST);
    }
    if (verbose > 1 && ok) printf("     a-priori bound (P): %.17g = <C,X~> %.17g + residual term %.3e + eigenvalue term %.3e (lambda_min(S X~ S) >= %.2e)\n",
                                  Ub, cxu, rterm, Ub - cxu - rterm, lmin_worst);
    free(rl); free(ru);
    bound_free_blocks(Xs, O->nblk);
    return ok ? Ub : INFINITY;
}
/* side d with given bounds of the diagonal of X: the bound for y~, or -inf with the reason */
static double prior_eval_d(const PSOrig *O, const double *y, int exact, double **xu, int verbose, char *reason, size_t rlen) {
    const int m = O->m;
    double **Zl = bound_alloc_blocks(O), **Zu = bound_alloc_blocks(O);
    for (int dir = 0; dir < 2; dir++) {
        fesetround(dir == 0 ? FE_DOWNWARD : FE_UPWARD);
        double **Zd = dir == 0 ? Zl : Zu;
        for (size_t q = 0; q < O->nnz; q++) {
            const int k = O->blk[q], i = O->ii[q], j = O->jj[q], n = abs(O->bs[k]);
            double lo, hi, c;
            dint(O->v[q], exact, &lo, &hi);
            if (O->con[q] < 0) c = dir == 0 ? lo : hi;
            else c = dir == 0 ? mul_lo(-hi, -lo, y[O->con[q]]) : mul_hi(-hi, -lo, y[O->con[q]]);   /* (rounds outward) */
            if (O->bs[k] < 0) Zd[k][i] += c;
            else { Zd[k][i + (size_t)j * n] += c; if (i != j) Zd[k][j + (size_t)i * n] += c; }
        }
    }
    fesetround(FE_DOWNWARD);
    double Lb = 0;
    for (int i = 0; i < m; i++) { double lo, hi; dint(O->b[i], exact, &lo, &hi); Lb += mul_lo(lo, hi, y[i]); }
    fesetround(FE_TONEAREST);
    int ok = 1;
    for (int k = 0; k < O->nblk && ok; k++) {
        const int n = abs(O->bs[k]);
        if (O->bs[k] < 0) {
            for (int i = 0; i < n && ok; i++) {
                if (Zl[k][i] >= 0) continue;
                if (!(xu[k][i] < INFINITY)) { ok = 0; if (reason) snprintf(reason, rlen, "no a-priori bound of entry %d of linear block %d of X", i + 1, k + 1); break; }
                fesetround(FE_DOWNWARD); Lb += Zl[k][i] * xu[k][i]; fesetround(FE_TONEAREST);
            }
            continue;
        }
        const size_t len = bsz_c(O->bs[k]);
        double *Zm = malloc(sizeof(double) * len), *rho = malloc(sizeof(double) * len);
        for (size_t q = 0; q < len; q++) Zm[q] = 0.5 * Zl[k][q] + 0.5 * Zu[k][q];
        fesetround(FE_UPWARD);
        for (size_t q = 0; q < len; q++) rho[q] = fmax(fabs(Zu[k][q] - Zm[q]), fabs(Zm[q] - Zl[k][q]));
        fesetround(FE_TONEAREST);
        double lb0 = lammin_lower(n, Zm);
        const double est = g_lam_est, rf = fro_up(len, rho);
        if (lb0 < rf) { if (lammin_ge_hi(n, Zm, rf, est, "a-priori bound (D)")) lb0 = rf; else lb0 = lammin_lower_hi(n, Zm, lb0, est, "a-priori bound (D)"); }
        free(Zm); free(rho);
        fesetround(FE_DOWNWARD);
        const double lb = lb0 - rf;
        fesetround(FE_TONEAREST);
        if (lb >= 0) continue;
        fesetround(FE_UPWARD);
        double xb = 0;
        for (int i = 0; i < n; i++) xb += xu[k][i];
        fesetround(FE_TONEAREST);
        if (!(xb < INFINITY)) { ok = 0; if (reason) snprintf(reason, rlen, "no a-priori bound of tr X of block %d", k + 1); break; }
        fesetround(FE_DOWNWARD); Lb += lb * xb; fesetround(FE_TONEAREST);
    }
    bound_free_blocks(Zl, O->nblk); bound_free_blocks(Zu, O->nblk);
    if (verbose > 1 && ok) printf("     a-priori bound (D): %.17g\n", Lb);
    return ok ? Lb : -INFINITY;
}
/* The a-priori bound of side p (candidates X1, X2) or d (y1, y2); NULL candidates are skipped.
 * Returns 1 with the better finite bound in *bound and the candidate it came from in *which
 * (1 or 2), 0 with the reason otherwise. */
int brisk_prior_bound(const PSOrig *O, int side, double **X1, double **X2, const double *y1, const double *y2, int exact,
                      int verbose, double *bound, int *which, char *reason, size_t rlen) {
    const int m = O->m, rm0 = fegetround();
    *bound = side == 1 ? INFINITY : -INFINITY;
    if (which) *which = 0;
    if (reason && rlen) reason[0] = 0;
    if (side == 1) {
        if (!X1 && !X2) { if (reason) snprintf(reason, rlen, "no X"); return 0; }
        double *yl = malloc(sizeof(double) * (m + 1)), *yu = malloc(sizeof(double) * (m + 1));
        double **zd = calloc(O->nblk + 1, sizeof(double *));
        int okm = yl && yu && zd;
        for (int k = 0; okm && k < O->nblk; k++) okm = (zd[k] = malloc(sizeof(double) * (abs(O->bs[k]) + 1))) != NULL;
        if (!okm) {
            free(yl); free(yu);
            if (zd) for (int k = 0; k < O->nblk; k++) free(zd[k]);
            free(zd);
            if (reason) snprintf(reason, rlen, "out of memory");
            return 0;
        }
        prior_propagate(O, exact, verbose, yl, yu, zd);
        for (int c = 0; c < 2; c++) {
            double **X = c == 0 ? X1 : X2;
            if (!X) continue;
            const double B = prior_eval_p(O, X, exact, yl, yu, zd, verbose, c == 0 || !(*bound < INFINITY) ? reason : NULL, rlen);
            if (B < *bound) { *bound = B; if (which) *which = c + 1; }
        }
        free(yl); free(yu);
        for (int k = 0; k < O->nblk; k++) free(zd[k]);
        free(zd);
        fesetround(rm0);
        return *bound < INFINITY;
    }
    if (!y1 && !y2) { if (reason) snprintf(reason, rlen, "no y"); return 0; }
    /* bounds of the diagonal of X from the equations with positive diagonal (or linear) entries only */
    unsigned char *rowok = malloc(m + 1);
    memset(rowok, 1, m + 1);
    for (size_t q = 0; q < O->nnz; q++) {
        if (O->con[q] < 0) continue;
        const int k = O->blk[q];
        if ((O->bs[k] > 0 && O->ii[q] != O->jj[q]) || !(O->v[q] > 0)) rowok[O->con[q]] = 0;
    }
    double **xu = calloc(O->nblk + 1, sizeof(double *));
    for (int k = 0; k < O->nblk; k++) { const int n = abs(O->bs[k]); xu[k] = malloc(sizeof(double) * (n + 1)); for (int i = 0; i < n; i++) xu[k][i] = INFINITY; }
    fesetround(FE_UPWARD);
    for (size_t q = 0; q < O->nnz; q++) {
        if (O->con[q] < 0 || !rowok[O->con[q]]) continue;
        double lo, hi, blo, bhi;
        dint(O->v[q], exact, &lo, &hi); dint(O->b[O->con[q]], exact, &blo, &bhi);
        if (!(bhi >= 0)) continue;
        const double v = bhi / lo;                  /* a x <= b, every other term of the row >= 0 */
        double *x = &xu[O->blk[q]][O->ii[q]];
        if (v < *x) *x = v;
    }
    fesetround(FE_TONEAREST);
    free(rowok);
    for (int c = 0; c < 2; c++) {
        const double *y = c == 0 ? y1 : y2;
        if (!y) continue;
        const double B = prior_eval_d(O, y, exact, xu, verbose, c == 0 || !(*bound > -INFINITY) ? reason : NULL, rlen);
        if (B > *bound) { *bound = B; if (which) *which = c + 1; }
    }
    for (int k = 0; k < O->nblk; k++) free(xu[k]);
    free(xu);
    fesetround(rm0);
    return *bound > -INFINITY;
}
