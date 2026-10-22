/*
 * High-precision endgame.
 *
 * On degenerate or ill-posed problems the Schur complement becomes so ill conditioned
 * that the search direction can no longer reduce the primal residual: on SDPLIB's hinf
 * family the double-precision iteration stalls with |A(X) - b| ~ 1e-7 while mu keeps
 * falling, so the two objectives converge to different values (a relative gap of up to
 * 3e-4). No amount of refinement in double precision helps, because the assembled Schur
 * matrix itself carries an error of eps * cond(M).
 *
 * This module therefore redoes the whole linear algebra in double-double arithmetic
 * (two doubles per number, about 32 digits): Z^{-1}, the Schur complement, its Cholesky
 * factorization, the direction and the step lengths. It runs a few HKM predictor-corrector
 * iterations starting from the double solver's best iterate, and only when
 *   - that iterate missed the tolerance, and
 *   - the problem is small enough (m <= dd_maxm and sum n^2 <= dd_maxn2),
 * so the cost (about 45x the flop cost of double arithmetic, on problems that take
 * milliseconds) is irrelevant. The caller accepts the result only if the measured
 * accuracy improves.
 *
 * Everything here works on the scaled problem, exactly as the main iteration does, and
 * uses plain dense per-block storage with generic sparse loops over the constraints: no
 * routes, no low-rank, no envelope. Small problems only.
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "amd/amd.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double dd_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ---------------------------------------------------------------- dd arithmetic */
typedef struct { double h, l; } dd;

static inline dd dd_qts(double a, double b) {          /* quick two-sum, |a| >= |b| */
    dd r;
    r.h = a + b;
    r.l = b - (r.h - a);
    return r;
}
static inline dd dd_ts(double a, double b) {           /* two-sum */
    double s = a + b, bb = s - a;
    dd r;
    r.h = s;
    r.l = (a - (s - bb)) + (b - bb);
    return r;
}
static inline dd dd_tp(double a, double b) {           /* two-product (fma) */
    dd r;
    r.h = a * b;
    r.l = fma(a, b, -r.h);
    return r;
}
static inline dd dd_of(double a) { dd r = { a, 0.0 }; return r; }
static inline double dd_d(dd a) { return a.h + a.l; }
static inline dd dd_add(dd a, dd b) {
    dd s = dd_ts(a.h, b.h), t = dd_ts(a.l, b.l);
    s.l += t.h;
    s = dd_qts(s.h, s.l);
    s.l += t.l;
    return dd_qts(s.h, s.l);
}
/* 5.16: the accumulation add (one two-sum, the low parts added in double: error 2^-104 relative,
 * half the operations of dd_add; the Schur assembly's inner loop) */
static inline dd dd_adds(dd a, dd b) { dd r = dd_ts(a.h, b.h); r.l += a.l + b.l; return dd_qts(r.h, r.l); }
static inline dd dd_neg(dd a) { dd r = { -a.h, -a.l }; return r; }
static inline dd dd_sub(dd a, dd b) { return dd_add(a, dd_neg(b)); }
static inline dd dd_mul(dd a, dd b) {
    dd p = dd_tp(a.h, b.h);
    p.l += a.h * b.l + a.l * b.h;
    return dd_qts(p.h, p.l);
}
static inline dd dd_muld(dd a, double b) {
    dd p = dd_tp(a.h, b);
    p.l += a.l * b;
    return dd_qts(p.h, p.l);
}
static inline dd dd_div(dd a, dd b) {
    double q1 = a.h / b.h;
    dd r = dd_sub(a, dd_muld(b, q1));
    double q2 = dd_d(r) / b.h;
    return dd_qts(q1, q2);
}
static inline dd dd_sqrt(dd a) {
    if (a.h <= 0) return dd_of(0.0);
    double x = 1.0 / sqrt(a.h);
    double ax = a.h * x;
    dd t = dd_sub(a, dd_tp(ax, ax));
    return dd_qts(ax, dd_d(t) * x * 0.5);
}
static inline int dd_gt(dd a, dd b) { dd d = dd_sub(a, b); return d.h > 0 || (d.h == 0 && d.l > 0); }

/* ---------------------------------------------------------------- block workspace */
typedef struct {
    int type, n;
    size_t len;                 /* n*n (SDP) or n (LP) */
    dd *X, *Z, *Zi, *dX, *dZ, *L, *T1, *T2, *T3, *Rd, *Q;
    char *fr;                   /* 5.9 (LP blocks): 1 = the + slot of a free variable given as a split pair (X holds its value,
                                 * Z is 0), 2 = its - slot (X and Z are 0); NULL: no pair in this block */
} DBlk;

static dd *dalloc(size_t n) {
    dd *p = calloc(n ? n : 1, sizeof(dd));
    if (!p) { fprintf(stderr, "brisk: out of memory (dd endgame)\n"); exit(1); }
    return p;
}

/* C = A B (n x n, dd) */
/* 5.9: the dense kernels on two loops the compiler turns into SIMD code (the kernels of the
 * high-precision solver, hpsolve.c: "sloppy" double-double sums, enough for factorizations and
 * products): y += a x and x'y on contiguous vectors. The scalar triple loops they replace
 * cost about 5 ns per element; these about 1. */
#define DD_NL 8
static void dd_axpy(size_t n, dd a, const dd *x, dd *y) {
    const double ah = a.h, al = a.l;
#pragma omp simd
    for (size_t i = 0; i < n; i++) {
        const double xh = x[i].h, xl = x[i].l, yh = y[i].h;
        const double p = ah * xh, e = fma(ah, xh, -p) + (ah * xl + al * xh);
        const double s = yh + p, bb = s - yh;
        const double t = ((yh - (s - bb)) + (p - bb)) + (y[i].l + e);
        const double h = s + t;
        y[i].l = t - (h - s); y[i].h = h;
    }
}
static dd dd_dot(size_t n, const dd *x, const dd *y) {
    double sh[DD_NL] = { 0 }, sl[DD_NL] = { 0 };
    size_t i = 0;
    for (; i + DD_NL <= n; i += DD_NL) {
#pragma omp simd
        for (int l = 0; l < DD_NL; l++) {
            const double ah = x[i + l].h, al = x[i + l].l, bh = y[i + l].h, bl = y[i + l].l;
            const double p = ah * bh, e = fma(ah, bh, -p) + (ah * bl + al * bh);
            const double s = sh[l] + p, bb = s - sh[l];
            sl[l] += ((sh[l] - (s - bb)) + (p - bb)) + e;
            sh[l] = s;
        }
    }
    dd r = dd_of(0.0);
    for (int l = 0; l < DD_NL; l++) { dd v; v = dd_ts(sh[l], sl[l]); r = dd_add(r, v); }
    for (; i < n; i++) r = dd_add(r, dd_mul(x[i], y[i]));
    return r;
}
/* C = A B (column-major): column j of C is the combination of the columns of A with column j of B */
/* (21 October 2026: the columns on the threads, each column's sums in the same order, so the bits are those of one
 * thread; the endgame's block algebra was single-threaded and cost 830 s for one step on a structural design problem
 * with four blocks of order 1,861, ISSUES 228) */
#define DD_PAR_N 128
/* (21 October 2026) the eigenvalue step only where the bisection costs real time: 28 double-double Cholesky factorizations
 * of n^3/3 a call are about 1 s at n = 500; below, the bisection's path is kept (roa_vdp_inner_d14_K, a block of order 148:
 * the eigenvalue step moved its endgame from 9.0e-5 to 1.05e-4 and into a re-solve, 24.6 -> 34.6 s) */
#define DD_EIG_N 512
static void dd_gemm(int n, const dd *A, const dd *B, dd *C) {
    memset(C, 0, sizeof(dd) * (size_t)n * n);
    #pragma omp parallel for schedule(static) if (n >= DD_PAR_N)
    for (int j = 0; j < n; j++) {
        dd *Cj = C + (size_t)j * n;
        for (int k = 0; k < n; k++) { const dd bkj = B[k + (size_t)j * n]; if (bkj.h == 0 && bkj.l == 0) continue; dd_axpy((size_t)n, bkj, A + (size_t)k * n, Cj); }
    }
}
static void dd_sym(int n, dd *A) {
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++) {
            dd v = dd_muld(dd_add(A[i + (size_t)j * n], A[j + (size_t)i * n]), 0.5);
            A[i + (size_t)j * n] = A[j + (size_t)i * n] = v;
        }
}
/* lower Cholesky of the lower triangle of A into L (left-looking by columns); returns 1 on failure */
/* (21 October 2026, ISSUES 228: for a large block the rows of each column on the threads in chunks: every entry's
 * updates in the same order over k, so the bits are those of one thread) */
static int dd_chol(int n, const dd *A, dd *L) {
    memset(L, 0, sizeof(dd) * (size_t)n * n);
    for (int j = 0; j < n; j++) {
        dd *Lj = L + (size_t)j * n;
        memcpy(Lj + j, A + (size_t)j * n + j, sizeof(dd) * (size_t)(n - j));
        const int len = n - j;
        if (n >= DD_PAR_N && len >= 64 && j >= 16) {
            const int nch = (len + 63) / 64;
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < nch; c++) {
                const int i0 = j + 64 * c, cl = i0 + 64 <= n ? 64 : n - i0;
                for (int k = 0; k < j; k++) { const dd v = L[j + (size_t)k * n]; if (v.h == 0 && v.l == 0) continue; dd_axpy((size_t)cl, dd_neg(v), L + (size_t)k * n + i0, Lj + i0); }
            }
        } else
            for (int k = 0; k < j; k++) { const dd v = L[j + (size_t)k * n]; if (v.h == 0 && v.l == 0) continue; dd_axpy((size_t)len, dd_neg(v), L + (size_t)k * n + j, Lj + j); }
        if (!(Lj[j].h > 0)) return 1;
        const dd d = dd_sqrt(Lj[j]), inv = dd_div(dd_of(1.0), d);
        Lj[j] = d;
        for (int i = j + 1; i < n; i++) Lj[i] = dd_mul(Lj[i], inv);
    }
    return 0;
}
static void dd_solve_chol(int m, const dd *L, dd *x) {
    for (int j = 0; j < m; j++) {                          /* forward, by columns */
        x[j] = dd_div(x[j], L[j + (size_t)j * m]);
        if (x[j].h != 0 || x[j].l != 0) dd_axpy((size_t)(m - j - 1), dd_neg(x[j]), L + (size_t)j * m + j + 1, x + j + 1);
    }
    for (int i = m - 1; i >= 0; i--) {                     /* backward: dots with the columns */
        const dd s = dd_sub(x[i], dd_dot((size_t)(m - i - 1), L + (size_t)i * m + i + 1, x + i + 1));
        x[i] = dd_div(s, L[i + (size_t)i * m]);
    }
}
/* inverse of a symmetric positive definite matrix from its Cholesky factor */
static void dd_inv_from_chol(int n, const dd *L, dd *Inv, dd *col) {
    if (n >= DD_PAR_N) {            /* (21 October 2026, ISSUES 228: the columns are independent: on the threads, solved in place) */
        (void)col;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++) {
            dd *cj = Inv + (size_t)j * n;
            for (int i = 0; i < n; i++) cj[i] = dd_of(i == j ? 1.0 : 0.0);
            dd_solve_chol(n, L, cj);
        }
        return;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = dd_of(i == j ? 1.0 : 0.0);
        dd_solve_chol(n, L, col);
        for (int i = 0; i < n; i++) Inv[i + (size_t)j * n] = col[i];
    }
}

/* ---------------------------------------------------------------- operators */
/* out_i += w * <A_i, D> over the constraints of this block */
static void dd_Aop(const Block *B, const DBlk *b, const dd *D, dd *out, double w) {
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *S = &B->A[t];
        dd s = dd_of(0.0);
        if (B->type == BLK_LP)
            for (int q = 0; q < S->nnz; q++) s = dd_add(s, dd_muld(D[S->row[q]], S->val[q]));
        else
            for (int q = 0; q < S->ef; q++)
                s = dd_add(s, dd_muld(D[S->fr[q] + (size_t)S->fc[q] * b->n], S->fv[q]));
        out[B->con[t]] = dd_add(out[B->con[t]], dd_muld(s, w));
    }
}
/* F = sum_i y_i A_i  (block part) */
static void dd_ATy(const Block *B, const DBlk *b, const dd *y, dd *F) {
    memset(F, 0, sizeof(dd) * b->len);
    for (int t = 0; t < B->ncon; t++) {
        dd yt = y[B->con[t]];
        if (yt.h == 0 && yt.l == 0) continue;
        const SpSym *S = &B->A[t];
        if (B->type == BLK_LP)
            for (int q = 0; q < S->nnz; q++) {
                size_t i = S->row[q];
                F[i] = dd_add(F[i], dd_muld(yt, S->val[q]));
            }
        else
            for (int q = 0; q < S->ef; q++) {
                size_t i = S->fr[q] + (size_t)S->fc[q] * b->n;
                F[i] = dd_add(F[i], dd_muld(yt, S->fv[q]));
            }
    }
}
/* F += w * (sparse matrix S) */
static void dd_add_sp(const Block *B, const DBlk *b, const SpSym *S, double w, dd *F) {
    if (B->type == BLK_LP)
        for (int q = 0; q < S->nnz; q++) F[S->row[q]] = dd_add(F[S->row[q]], dd_of(w * S->val[q]));
    else
        for (int q = 0; q < S->ef; q++) {
            size_t i = S->fr[q] + (size_t)S->fc[q] * b->n;
            F[i] = dd_add(F[i], dd_of(w * S->fv[q]));
        }
}
/* M += lower triangle of the block's Schur contribution M_tu = tr(A_t X A_u Zi).
 * SDP blocks: H_t = X A_t Zi on the union pattern of the block's constraints
 * (Y = A_t Zi on the rows of A_t, then H = X[:, rows] Y), and M_tu = <A_u, H_t>.
 * Cost per constraint: n ef_t + |pattern| r_t + sum_u ef_u dd operations, instead of
 * sum_u ef_t ef_u for the pairwise loop this replaces (90x less on dense 12x12 LMIs). */
static double g_dds_prod = 0, g_dds_gath = 0, g_dds_lp = 0;   /* 5.16: the assembly's parts, under BRISK_DDTIME */
static double g_ddf_fact = 0, g_ddp_panel = 0, g_ddp_syrk = 0;  /* the factorization, and the blocked factor's panels and trailing updates */
static void dd_schur(const Block *B, const DBlk *b, int m, dd *M) {
    const int n = b->n;
    const double t0 = dd_now();
    if (B->type == BLK_LP) {
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *St = &B->A[t];
            const int it = B->con[t];
            for (int u = t; u < B->ncon; u++) {
                const SpSym *Su = &B->A[u];
                const int iu = B->con[u];
                dd s = dd_of(0.0);
                int qa = 0, qb = 0;
                while (qa < St->nnz && qb < Su->nnz) {
                    int ra = St->row[qa], rb = Su->row[qb];
                    if (ra < rb) qa++;
                    else if (rb < ra) qb++;
                    else {
                        dd v = dd_mul(b->X[ra], b->Zi[ra]);
                        s = dd_add(s, dd_muld(v, St->val[qa] * Su->val[qb]));
                        qa++; qb++;
                    }
                }
                int r = it > iu ? it : iu, c = it > iu ? iu : it;
                M[r + (size_t)c * m] = dd_add(M[r + (size_t)c * m], s);
            }
        }
        g_dds_lp += dd_now() - t0;
        return;
    }
    const int unf = B->unf;
    const int *ufr = B->ufr, *ufc = B->ufc;
    /* 5.16: the constraints over the threads (each H_t and its row of M are one thread's; the same
     * bits for any thread count), with per-thread work space */
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#endif
    if (B->ncon < 64) nthr = 1;
    int *rposall = malloc(sizeof(int) * (size_t)n * nthr);
    dd *Yall = dalloc((size_t)n * n * nthr);          /* rows of A_t Zi, compact (r x n) */
    dd *Hall = dalloc((size_t)n * n * nthr);          /* H_t on the union pattern */
    /* 5.16: H_t is needed only at the entries of the constraints u >= t (the gather below): the
     * union pattern ordered by the last constraint touching each entry, so that the live entries
     * at constraint t are a prefix (half the products on average; the products were 86 % of the
     * assembly on an occupation-measure file) */
    int *last = malloc(sizeof(int) * (size_t)unf), *order = malloc(sizeof(int) * (size_t)unf), *cnt = calloc((size_t)B->ncon + 2, sizeof(int));
    {
        int *pos = malloc(sizeof(int) * (size_t)n * n);
        for (int e = 0; e < unf; e++) { pos[ufr[e] + (size_t)ufc[e] * n] = e; last[e] = -1; }
        for (int u = 0; u < B->ncon; u++) { const SpSym *Su = &B->A[u]; for (int q = 0; q < Su->ef; q++) last[pos[Su->fr[q] + (size_t)Su->fc[q] * n]] = u; }
        free(pos);
        for (int e = 0; e < unf; e++) if (last[e] >= 0) cnt[last[e]]++;             /* counting sort, descending in last */
        int *start = malloc(sizeof(int) * (size_t)(B->ncon + 1)); int acc = 0;
        for (int u = B->ncon - 1; u >= 0; u--) { start[u] = acc; acc += cnt[u]; }
        for (int e = 0; e < unf; e++) if (last[e] >= 0) order[start[last[e]]++] = e;
        /* cnt[t] = entries with last >= t (prefix length at constraint t) */
        acc = 0; for (int u = B->ncon - 1; u >= 0; u--) { acc += cnt[u]; cnt[u] = acc; }
        free(start);
    }
    /* the last constraint touching each column of the union pattern: a dense column product (SIMD
     * axpy over the column) where the live entries fill the live columns, the entry loop otherwise */
    int *colLast = malloc(sizeof(int) * (size_t)n);
    for (int j = 0; j < n; j++) colLast[j] = -1;
    for (int e = 0; e < unf; e++) if (last[e] > colLast[ufc[e]]) colLast[ufc[e]] = last[e];
    int *ncolLive = calloc((size_t)B->ncon + 1, sizeof(int));
    /* 5.16: the gather transposed — for each live entry, its constraints u >= t (lists by entry, CSR over
     * the union pattern: H read once in order, the sums in a row accumulator in L1) */
    int *eptr = calloc((size_t)unf + 1, sizeof(int)), *econ = NULL; double *eval = NULL;
    {
        int *pos = malloc(sizeof(int) * (size_t)n * n);
        for (int e = 0; e < unf; e++) pos[ufr[e] + (size_t)ufc[e] * n] = e;
        for (int u = 0; u < B->ncon; u++) { const SpSym *Su = &B->A[u]; for (int q = 0; q < Su->ef; q++) eptr[pos[Su->fr[q] + (size_t)Su->fc[q] * n] + 1]++; }
        for (int e = 0; e < unf; e++) eptr[e + 1] += eptr[e];
        econ = malloc(sizeof(int) * (size_t)eptr[unf]); eval = malloc(sizeof(double) * (size_t)eptr[unf]);
        int *fill = malloc(sizeof(int) * (size_t)unf); memcpy(fill, eptr, sizeof(int) * (size_t)unf);
        for (int u = 0; u < B->ncon; u++) { const SpSym *Su = &B->A[u]; for (int q = 0; q < Su->ef; q++) { const int e = pos[Su->fr[q] + (size_t)Su->fc[q] * n]; econ[fill[e]] = u; eval[fill[e]] = Su->fv[q]; fill[e]++; } }
        free(fill); free(pos);
    }
    dd *accall = dalloc((size_t)B->ncon * nthr);
    static int gathT = -1; if (gathT < 0) gathT = !getenv("BRISK_DDGATH0");
    for (int j = 0; j < n; j++) if (colLast[j] >= 0) ncolLive[colLast[j]]++;
    for (int u = B->ncon - 2; u >= 0; u--) ncolLive[u] += ncolLive[u + 1];      /* columns with colLast >= u */
    double tq = dd_now();
    #pragma omp parallel for schedule(dynamic, 4) if (nthr > 1)
    for (int t = 0; t < B->ncon; t++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        int *rpos = rposall + (size_t)tid * n;
        dd *Y = Yall + (size_t)tid * n * n, *H = Hall + (size_t)tid * n * n;
        const SpSym *St = &B->A[t];
        const int it = B->con[t], r = St->nr;
        for (int k = 0; k < r; k++) rpos[St->rows[k]] = k;
        memset(Y, 0, sizeof(dd) * (size_t)r * n);
        for (int q = 0; q < St->ef; q++) {
            const int p = rpos[St->fr[q]], c = St->fc[q];
            const double v = St->fv[q];
            dd *yr = Y + (size_t)p;                     /* row p, stride r */
            const dd *zc = b->Zi + (size_t)c * n;        /* Zi[c, :] = Zi[:, c] */
            for (int j = 0; j < n; j++) yr[(size_t)j * r] = dd_adds(yr[(size_t)j * r], dd_muld(zc[j], v));
        }
        const int live = cnt[t];
        if ((double)live >= 0.25 * (double)n * (double)ncolLive[t]) {
            for (int j = 0; j < n; j++) {
                if (colLast[j] < t) continue;
                dd *Hj = H + (size_t)j * n;
                memset(Hj, 0, sizeof(dd) * (size_t)n);
                for (int k = 0; k < r; k++) dd_axpy((size_t)n, Y[k + (size_t)j * r], b->X + (size_t)St->rows[k] * n, Hj);
            }
        } else {
            for (int o = 0; o < live; o++) {
                const int e = order[o], i = ufr[e], j = ufc[e];
                dd s = dd_mul(b->X[i + (size_t)St->rows[0] * n], Y[(size_t)j * r]);
                for (int k = 1; k < r; k++)
                    s = dd_adds(s, dd_mul(b->X[i + (size_t)St->rows[k] * n], Y[k + (size_t)j * r]));
                H[i + (size_t)j * n] = s;
            }
        }
        if (nthr == 1) { const double t1 = dd_now(); g_dds_prod += t1 - tq; tq = t1; }
        if (gathT) {
            dd *acc = accall + (size_t)tid * B->ncon;
            for (int u = t; u < B->ncon; u++) acc[u] = dd_of(0.0);
            for (int o = 0; o < live; o++) {
                const int e = order[o];
                const dd h = H[ufr[e] + (size_t)ufc[e] * n];
                for (int p = eptr[e + 1] - 1; p >= eptr[e] && econ[p] >= t; p--)      /* the lists ascend in u */
                    acc[econ[p]] = dd_adds(acc[econ[p]], dd_muld(h, eval[p]));
            }
            for (int u = t; u < B->ncon; u++) {
                const int iu = B->con[u];
                int rr = it > iu ? it : iu, c = it > iu ? iu : it;
                M[rr + (size_t)c * m] = dd_add(M[rr + (size_t)c * m], acc[u]);
            }
        } else
        for (int u = t; u < B->ncon; u++) {
            const SpSym *Su = &B->A[u];
            const int iu = B->con[u];
            dd s = dd_of(0.0);
            for (int q = 0; q < Su->ef; q++)
                s = dd_adds(s, dd_muld(H[Su->fr[q] + (size_t)Su->fc[q] * n], Su->fv[q]));
            int rr = it > iu ? it : iu, c = it > iu ? iu : it;
            M[rr + (size_t)c * m] = dd_add(M[rr + (size_t)c * m], s);
        }
        if (nthr == 1) { const double t2 = dd_now(); g_dds_gath += t2 - tq; tq = t2; }
    }
    free(rposall); free(Yall); free(Hall); free(last); free(order); free(cnt); free(colLast); free(ncolLive); free(eptr); free(econ); free(eval); free(accall);
}


/* Search directions from dy:
 *   dZ = Rd - A'(dy)
 *   dX = -sym(Zi dZ X) - X            (predictor, corr = 0)
 *   dX = smu Zi - Q - sym(Zi dZ X) - X (corrector, corr = 1; Q = sym(Zi dZa dXa))   */
static void dd_dir(const Block *B, DBlk *b, const dd *dy, dd smu, int corr) {
    const int n = b->n;
    dd_ATy(B, b, dy, b->T1);
    for (size_t i = 0; i < b->len; i++) b->dZ[i] = dd_sub(b->Rd[i], b->T1[i]);
    if (b->type == BLK_LP) {
        for (int i = 0; i < n; i++) {
            dd v = dd_mul(b->Zi[i], dd_mul(b->dZ[i], b->X[i]));
            dd x = dd_sub(dd_neg(v), b->X[i]);
            if (corr) x = dd_add(x, dd_sub(dd_mul(b->Zi[i], smu), b->Q[i]));
            b->dX[i] = x;
        }
        return;
    }
    dd_gemm(n, b->Zi, b->dZ, b->T2);
    dd_gemm(n, b->T2, b->X, b->T3);
    dd_sym(n, b->T3);
    for (size_t i = 0; i < b->len; i++) {
        dd x = dd_sub(dd_neg(b->T3[i]), b->X[i]);
        if (corr) x = dd_sub(dd_add(x, dd_mul(b->Zi[i], smu)), b->Q[i]);
        b->dX[i] = x;
    }
}

/* largest steps keeping X + a dX and Z + a dZ positive definite (bisection + Cholesky) */
static int g_lim_p = -1, g_lim_d = -1;   /* the block that set the last step bound on each side (diagnostics, BRISK_DDDIAG) */
/* diagnostics (BRISK_DDDIAG, 14 October): the smallest eigenvalues of X and Z of a block, in double */
static void dd_blk_lmin(const DBlk *b, double *lx, double *lz) {
    const int n = b->n; *lx = *lz = 0;
    double *T = malloc(sizeof(double) * b->len), *ev = malloc(sizeof(double) * n), wq; int lwork = -1, info;
    for (int pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < b->len; i++) T[i] = dd_d(pass ? b->Z[i] : b->X[i]);
        BL(dsyev_)("N", "L", &n, T, &n, ev, &wq, &lwork, &info); lwork = (int)wq + 1;
        double *work = malloc(sizeof(double) * lwork);
        BL(dsyev_)("N", "L", &n, T, &n, ev, work, &lwork, &info); free(work); lwork = -1;
        if (info == 0) { if (pass) *lz = ev[0]; else *lx = ev[0]; }
    }
    free(T); free(ev);
}
/* diagnostics: the entry of an LP block that bounds the step, with its values */
static void dd_lp_limiter(const DBlk *b, int side, double *ratio, int *idx, double *v, double *d, double *v2, double *d2) {
    const dd *V = side ? b->Z : b->X, *D = side ? b->dZ : b->dX, *W = side ? b->X : b->Z, *E = side ? b->dX : b->dZ;
    *ratio = 1e300; *idx = -1;
    for (int i = 0; i < b->n; i++) {
        if (b->fr && b->fr[i]) continue;
        if (dd_d(D[i]) < 0) { const double r = -dd_d(V[i]) / dd_d(D[i]); if (r < *ratio) { *ratio = r; *idx = i; } }
    }
    if (*idx >= 0) { *v = dd_d(V[*idx]); *d = dd_d(D[*idx]); *v2 = dd_d(W[*idx]); *d2 = dd_d(E[*idx]); }
}
/* (21 October 2026, ISSUES 228) the largest a with V + a D >= 0, in double: V = L L' (dpotrf), W = L^-1 D L^-T (two dtrsm),
 * a = -1 / lambda_min(W) (dsyevr, the smallest eigenvalue only); 1e300 when lambda_min >= 0; -1 when V is not positive
 * definite in double or LAPACK fails. */
static double dd_eig_step(int n, const dd *V, const dd *D) {
    const size_t len = (size_t)n * n;
    double *L = malloc(sizeof(double) * len), *W = malloc(sizeof(double) * len);
    if (!L || !W) { free(L); free(W); return -1; }
    for (size_t i = 0; i < len; i++) { L[i] = dd_d(V[i]); W[i] = dd_d(D[i]); }
    int info = 0; double res = -1;
    BL(dpotrf_)("L", &n, L, &n, &info);
    if (info == 0) {
        const double one = 1.0;
        BL(dtrsm_)("L", "L", "N", "N", &n, &n, &one, L, &n, W, &n);
        BL(dtrsm_)("R", "L", "T", "N", &n, &n, &one, L, &n, W, &n);
        for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) { const double v = 0.5 * (W[i + (size_t)j * n] + W[j + (size_t)i * n]); W[i + (size_t)j * n] = W[j + (size_t)i * n] = v; }
        int il = 1, iu = 1, mfound = 0, lwork = -1, liwork = -1, iwq = 0; double vl = 0, vu = 0, abstol = 0, ev[2], wq = 0, z = 0; int ldz = 1, isup[2];
        BL(dsyevr_)("N", "I", "L", &n, W, &n, &vl, &vu, &il, &iu, &abstol, &mfound, ev, &z, &ldz, isup, &wq, &lwork, &iwq, &liwork, &info);
        lwork = (int)wq + 1; liwork = iwq + 1;
        double *work = malloc(sizeof(double) * lwork); int *iwork = malloc(sizeof(int) * liwork);
        if (work && iwork) {
            BL(dsyevr_)("N", "I", "L", &n, W, &n, &vl, &vu, &il, &iu, &abstol, &mfound, ev, &z, &ldz, isup, work, &lwork, iwork, &liwork, &info);
            if (info == 0 && mfound >= 1) res = ev[0] >= 0 ? 1e300 : -1.0 / ev[0];
        }
        free(work); free(iwork);
    }
    free(L); free(W);
    return res;
}
static void dd_steps(const Problem *P, DBlk *b, int nb, dd *apo, dd *ado) {
    dd ap = dd_of(1.0), ad = dd_of(1.0);
    g_lim_p = g_lim_d = -1;
    for (int k = 0; k < nb; k++) {
        const int n = b[k].n;
        for (int side = 0; side < 2; side++) {
            const dd *V = side ? b[k].Z : b[k].X, *D = side ? b[k].dZ : b[k].dX;
            dd lo = dd_of(0.0), hi = side ? ad : ap;
            /* 21 October 2026 (ISSUES 228): a large block's bound from the eigenvalues of L^-1 D L^-T in double (V = L L'),
             * for blocks of order >= DD_EIG_N, one Cholesky, two triangular solves and the smallest eigenvalue on the threaded BLAS, confirmed by one
             * double-double Cholesky at the bound less 2^-14; the bisection (14 double-double Cholesky factorizations a
             * side, 80 % of an endgame iteration on blocks of order 1,861) only when the double estimate fails or is not
             * confirmed. BRISK_DDBISECT=1: the bisection always. */
            static int bisect_only = -1; if (bisect_only < 0) bisect_only = getenv("BRISK_DDBISECT") != NULL;
            if (b[k].type != BLK_LP && n >= DD_EIG_N && !bisect_only) {
                const double a_est = dd_eig_step(n, V, D);
                if (a_est > 0) {
                    const double a = fmin(dd_d(hi), a_est) * (1.0 - 1.0 / 16384.0);
                    const dd ad_ = dd_of(a);
                    for (size_t i = 0; i < b[k].len; i++) b[k].T1[i] = dd_add(V[i], dd_mul(ad_, D[i]));
                    if (!dd_chol(n, b[k].T1, b[k].T2)) {
                        lo = ad_;
                        if (side) { if (dd_d(lo) < 0.999 * dd_d(ad)) g_lim_d = k; ad = lo; } else { if (dd_d(lo) < 0.999 * dd_d(ap)) g_lim_p = k; ap = lo; }
                        continue;
                    }
                    hi = ad_;            /* (not confirmed: the bisection below the estimate) */
                }
            }
            for (int s = 0; s < 14; s++) {                /* 2^-14: the step is scaled by 0.9-0.99 anyway */
                dd mid = dd_muld(dd_add(lo, hi), 0.5);
                int ok = 1;
                if (b[k].type == BLK_LP) {
                    for (int i = 0; i < n && ok; i++) {
                        if (b[k].fr && b[k].fr[i]) continue;            /* (a free variable has no bound) */
                        if (!dd_gt(dd_add(V[i], dd_mul(mid, D[i])), dd_of(0.0))) ok = 0;
                    }
                } else {
                    for (size_t i = 0; i < b[k].len; i++) b[k].T1[i] = dd_add(V[i], dd_mul(mid, D[i]));
                    ok = !dd_chol(n, b[k].T1, b[k].T2);
                }
                if (ok) lo = mid; else hi = mid;
            }
            if (side) { if (dd_d(lo) < 0.999 * dd_d(ad)) g_lim_d = k; ad = lo; } else { if (dd_d(lo) < 0.999 * dd_d(ap)) g_lim_p = k; ap = lo; }   /* (the bisection stops 2^-14 below the bound) */
        }
    }
    (void)P;
    *apo = ap; *ado = ad;
}

/* 5.9: the Schur complement factored as a sparse matrix. The endgame used a dense Cholesky
 * factorization in dd (m^3/3 operations at about 45 flops each: 2.3 s for m = 387 on a problem
 * that double solves in 0.2 s, so the work budget allowed two iterations or none). The pattern
 * of M is that of the double solver's Schur complement, usually sparse where the endgame is
 * needed (many small blocks): an AMD order of the pattern found at the first assembly, then an
 * up-looking Cholesky with explicit row and column lists (dense storage of L, m <= dd_maxm).
 * The work is that of the sparse factorization: sum over the columns of (entries below)^2. */
typedef struct {
    int dense;                  /* the factor is nearly full: the dense kernels (LD, column-major) */
    dd *LD;
    int m; int *perm;           /* row i of L is constraint perm[i] */
    dd *L;                      /* row-major, lower triangle, permuted */
    int **rs, *rn, *rcap;       /* columns j < i with L[i,j] != 0, increasing */
    int **cs, *cn, *ccap;       /* rows k > j with L[k,j] != 0, increasing */
    char *mark; dd *w, *t;
    int ndep;                   /* rows found dependent in the last factorization */
    double flops;               /* of the last factorization */
} DSp;
static void dsp_free(DSp *S) {
    if (!S->perm) return;
    if (S->rs) for (int i = 0; i < S->m; i++) { free(S->rs[i]); free(S->cs[i]); }
    free(S->perm); free(S->L); free(S->LD); free(S->rs); free(S->rn); free(S->rcap); free(S->cs); free(S->cn); free(S->ccap); free(S->mark); free(S->w); free(S->t);
    memset(S, 0, sizeof *S);
}
/* the order: AMD on the pattern of M (full symmetric storage) */
static void dsp_init(DSp *S, int m, const dd *M) {
    memset(S, 0, sizeof *S);
    S->m = m; S->perm = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    int *ap = (int *)calloc((size_t)m + 1, sizeof(int)); size_t nz = 0;
    for (int j = 0; j < m; j++) { for (int i = 0; i < m; i++) if (i != j && (M[i + (size_t)j * m].h != 0 || M[i + (size_t)j * m].l != 0)) nz++; ap[j + 1] = (int)nz; }
    int *ai = (int *)malloc(sizeof(int) * (nz + 1)); nz = 0;
    for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) if (i != j && (M[i + (size_t)j * m].h != 0 || M[i + (size_t)j * m].l != 0)) ai[nz++] = i;
    if (getenv("BRISK_DDDENSE") || amd_order(m, ap, ai, S->perm, NULL, NULL) < 0) for (int i = 0; i < m; i++) S->perm[i] = i;
    free(ap); free(ai);
    /* the work of the sparse factorization, counted on the pattern; beyond a twelfth of the
     * dense count (the dense kernels are vectorized, about 5 times faster per entry, and the
     * sparse code pays for its lists) the factor is dense */
    {
        const double lim = getenv("BRISK_DDSPARSE") ? 1e300 : (double)m * m * m / 6.0 / 12.0;
        double fl = 0;
        int **cs = (int **)calloc((size_t)m + 1, sizeof(int *)), *cn = (int *)calloc((size_t)m + 1, sizeof(int)), *cc = (int *)calloc((size_t)m + 1, sizeof(int));
        char *mk = (char *)calloc((size_t)m + 1, 1);
        for (int i = 0; i < m && fl <= lim; i++) {
            const dd *Mi = M + (size_t)S->perm[i] * m;
            for (int j = 0; j < i; j++) { const dd v = Mi[S->perm[j]]; mk[j] = v.h != 0 || v.l != 0; }
            for (int j = 0; j < i; j++) {
                if (!mk[j]) continue;
                mk[j] = 0;
                for (int q = 0; q < cn[j]; q++) mk[cs[j][q]] = 1;
                fl += cn[j] + 1;
                if (cn[j] == cc[j]) { cc[j] = 2 * cc[j] + 8; cs[j] = (int *)realloc(cs[j], sizeof(int) * (size_t)cc[j]); }
                cs[j][cn[j]++] = i;
            }
        }
        for (int i = 0; i < m; i++) free(cs[i]);
        free(cs); free(cn); free(cc); free(mk);
        S->dense = getenv("BRISK_DDDENSE") ? 1 : fl > lim;
    }
    if (S->dense) { S->LD = (dd *)calloc((size_t)m * m + 1, sizeof(dd)); S->t = (dd *)calloc((size_t)m + 1, sizeof(dd)); return; }
    S->L = (dd *)calloc((size_t)m * m + 1, sizeof(dd));
    S->rs = (int **)calloc((size_t)m + 1, sizeof(int *)); S->rn = (int *)calloc((size_t)m + 1, sizeof(int)); S->rcap = (int *)calloc((size_t)m + 1, sizeof(int));
    S->cs = (int **)calloc((size_t)m + 1, sizeof(int *)); S->cn = (int *)calloc((size_t)m + 1, sizeof(int)); S->ccap = (int *)calloc((size_t)m + 1, sizeof(int));
    S->mark = (char *)calloc((size_t)m + 1, 1); S->w = (dd *)calloc((size_t)m + 1, sizeof(dd)); S->t = (dd *)calloc((size_t)m + 1, sizeof(dd));
}
/* Linearly dependent constraints (the Schur complement is singular): a pivot below 1e-30 of its
 * diagonal entry (the rounding level of double-double: near the end the pivots of independent
 * rows legitimately range over many orders of magnitude, and 1e-22 took such rows out on the
 * option-pricing relaxations) marks the row dependent; its pivot is made huge, which sets that component of
 * every solve to zero (the rule of the high-precision solver). */
#define DSP_DEP 1e-30
/* ---------------------------------------------------------------- 5.16: the dense dd Cholesky, blocked,
 * its trailing updates through exact double products (Ozaki's scheme). Each dd entry of the panel is
 * split into pieces of s bits relative to its row's largest magnitude (x = x_1 + ... + x_np, the last
 * piece the rest), so that every piece product summed over the panel's width is exact in double
 * (2s + log2(width) <= 53); the np^2 piece products and the two cross terms with the low parts are
 * dgemms, their results added to the trailing matrix in dd. Measured against binary128 (14 October):
 * four pieces of 17 bits are as accurate as the scalar dd update (5.6e-24 against 1.4e-23 of a matrix of
 * scale 2.5e8), 6.8x faster at n = 2000 with the reference BLAS at one thread; the products being exact,
 * the result does not depend on the BLAS's summation order or thread count. */
/* three pieces of 23 bits by default (11 dgemms; the error 2^-99 of the row scales: the same results as four
 * pieces of 17 on qap9, qap10 and an m = 6,034 file, whose endgame is 1.5x faster, 262 -> 171 s at two threads);
 * BRISK_DDPIECES=4 gives four of 17 (18 dgemms, 2^-104) */
static int ddp_np(void) { static int np = -1; if (np < 0) { const char *e = getenv("BRISK_DDPIECES"); np = e && atoi(e) == 4 ? 4 : 3; } return np; }
#define DDP_NP ddp_np()
#define DDP_S (DDP_NP == 3 ? 23 : 17)
static void ddp_split(int n, int k, const double *X, int ldx, double **Xp, int ldp) {
    for (int i = 0; i < n; i++) {
        double mx = 0; for (int p = 0; p < k; p++) { const double v = fabs(X[i + (size_t)p * ldx]); if (v > mx) mx = v; }
        int e; frexp(mx > 0 ? mx : 1.0, &e);
        for (int p = 0; p < k; p++) {
            double r = X[i + (size_t)p * ldx];
            for (int q = 0; q < DDP_NP - 1; q++) {
                const double sc = ldexp(1.0, DDP_S * (q + 1) - e);
                const double y = floor(r * sc + 0.5) / sc;
                Xp[q][i + (size_t)p * ldp] = y; r -= y;
            }
            Xp[DDP_NP - 1][i + (size_t)p * ldp] = r;
        }
    }
}
/* C (n x n lower, ldc) -= A A' (A: n x k dd, lda); work: (2 + DDP_NP) n k + n cb doubles */
static void ddp_syrk(int n, int k, const dd *A, int lda, dd *C, int ldc, double *work) {
    if (n <= 0 || k <= 0) return;
    double *Ah = work, *Al = Ah + (size_t)n * k, *Xp[4];
    for (int q = 0; q < DDP_NP; q++) Xp[q] = Al + (size_t)n * k * (q + 1);
    double *P = Al + (size_t)n * k * (DDP_NP + 1);
    const int cb = n < 512 ? n : 512;                /* the product in column blocks */
    for (int p = 0; p < k; p++) for (int i = 0; i < n; i++) { Ah[i + (size_t)p * n] = A[i + (size_t)p * lda].h; Al[i + (size_t)p * n] = A[i + (size_t)p * lda].l; }
    ddp_split(n, k, Ah, n, Xp, n);
    const double one = 1.0, zero = 0.0;
    for (int c0 = 0; c0 < n; c0 += cb) {
        const int nc = n - c0 < cb ? n - c0 : cb, nr = n - c0;     /* rows c0.. (the lower triangle) */
        for (int a = 0; a < DDP_NP + 1; a++) for (int b = 0; b < DDP_NP + 1; b++) {
            const double *X, *Y;
            if (a < DDP_NP && b < DDP_NP) { X = Xp[a]; Y = Xp[b]; }
            else if (a == DDP_NP && b == DDP_NP) continue;        /* Al Al': below 2^-106 */
            else { X = a == DDP_NP ? Al : Ah; Y = b == DDP_NP ? Al : Ah; if (a < DDP_NP && b == DDP_NP) { X = Ah; } }
            /* the cross terms once each: (Ah, Al) and (Al, Ah) */
            if ((a == DDP_NP) != (b == DDP_NP)) { if (a < DDP_NP && a != 0) continue; if (b < DDP_NP && b != 0) continue; X = a == DDP_NP ? Al : Ah; Y = b == DDP_NP ? Al : Ah; }
            pdgemm("N", "T", &nr, &nc, &k, &one, X + c0, &n, Y + c0, &n, &zero, P, &nr);   /* the solver's threaded tiles (the products are exact: the same bits for any tiling) */
            #pragma omp parallel for schedule(static) if (nr >= 512)
            for (int j = 0; j < nc; j++) for (int i = j; i < nr; i++) {     /* each entry one thread's: the same bits */
                dd *c = C + (c0 + i) + (size_t)(c0 + j) * ldc; const double v = -P[i + (size_t)j * nr];
                dd t = dd_ts(c->h, v); t.l += c->l; *c = dd_qts(t.h, t.l);
            }
        }
    }
}
/* the blocked Cholesky of the lower triangle of L (n x n dd, column-major); the panels by the column
 * algorithm (dd_axpy), the trailing updates by ddp_syrk; dependent pivots as in the column code */
static int ddp_potrf(int n, dd *L, const dd *M, int *ndep) {
    const int nb = 128;
    double *work = (double *)malloc(sizeof(double) * ((size_t)(2 + DDP_NP) * n * nb + (size_t)n * (n < 512 ? n : 512)));
    for (int j0 = 0; j0 < n; j0 += nb) {
        const int b = n - j0 < nb ? n - j0 : nb;
        const double tp = dd_now();
        for (int j = j0; j < j0 + b; j++) {
            dd *Lj = L + (size_t)j * n;
            for (int k = j0; k < j; k++) { const dd v = L[j + (size_t)k * n]; if (v.h == 0 && v.l == 0) continue; dd_axpy((size_t)(n - j), dd_neg(v), L + (size_t)k * n + j, Lj + j); }
            const double ajj = M[j + (size_t)j * n].h;
            if (!(Lj[j].h > DSP_DEP * ajj)) {
                if (!(ajj > 0) || Lj[j].h < -1e-12 * ajj) { free(work); return 1; }
                Lj[j] = dd_of(1e150 * (ajj > 0 ? ajj : 1.0)); for (int i = j + 1; i < n; i++) Lj[i] = dd_of(0.0);
                (*ndep)++; continue;
            }
            const dd d = dd_sqrt(Lj[j]), inv = dd_div(dd_of(1.0), d);
            Lj[j] = d;
            for (int i = j + 1; i < n; i++) Lj[i] = dd_mul(Lj[i], inv);
        }
        const int n2 = n - j0 - b;
        const double ts = dd_now(); g_ddp_panel += ts - tp;
        if (n2 > 0) ddp_syrk(n2, b, L + (j0 + b) + (size_t)j0 * n, n, L + (j0 + b) + (size_t)(j0 + b) * n, n, work);
        g_ddp_syrk += dd_now() - ts;
    }
    free(work);
    return 0;
}
static int dsp_factor(DSp *S, const dd *M) {
    S->ndep = 0;
    if (S->dense) {
        const int n = S->m; dd *L = S->LD;
        S->flops = (double)n * n * n / 6.0;
        memset(L, 0, sizeof(dd) * (size_t)n * n);
        /* the blocked factor from n = 1024 (below it the column code with its SIMD axpy is the faster one:
         * SDPLIB's endgames +11 % with the blocked one everywhere; above, the trailing dgemms dominate and
         * scale with the threads: an m = 6,034 factor 77 -> 40 s at one thread); BRISK_DDBLK=<n> sets it */
        static int blk_from = -1; if (blk_from < 0) { const char *e = getenv("BRISK_DDBLK"); blk_from = e ? atoi(e) : 1024; }
        if (blk_from > 0 && n >= blk_from) {
            for (int j = 0; j < n; j++) memcpy(L + (size_t)j * n + j, M + (size_t)j * n + j, sizeof(dd) * (size_t)(n - j));
            return ddp_potrf(n, L, M, &S->ndep);
        }
        for (int j = 0; j < n; j++) {
            dd *Lj = L + (size_t)j * n;
            memcpy(Lj + j, M + (size_t)j * n + j, sizeof(dd) * (size_t)(n - j));
            for (int k = 0; k < j; k++) { const dd v = L[j + (size_t)k * n]; if (v.h == 0 && v.l == 0) continue; dd_axpy((size_t)(n - j), dd_neg(v), L + (size_t)k * n + j, Lj + j); }
            const double ajj = M[j + (size_t)j * n].h;
            if (!(Lj[j].h > DSP_DEP * ajj)) {
                if (!(ajj > 0) || Lj[j].h < -1e-12 * ajj) return 1;
                Lj[j] = dd_of(1e150 * (ajj > 0 ? ajj : 1.0)); for (int i = j + 1; i < n; i++) Lj[i] = dd_of(0.0);
                S->ndep++; continue;
            }
            const dd d = dd_sqrt(Lj[j]), inv = dd_div(dd_of(1.0), d);
            Lj[j] = d;
            for (int i = j + 1; i < n; i++) Lj[i] = dd_mul(Lj[i], inv);
        }
        return 0;
    }
    const int m = S->m; const int *p = S->perm;
    dd *L = S->L, *w = S->w; char *mark = S->mark;
    double fl = 0;
    for (int i = 0; i < m; i++) { S->rn[i] = 0; S->cn[i] = 0; }
    for (int i = 0; i < m; i++) {
        const dd *Mi = M + (size_t)p[i] * m;        /* (M is symmetric: column p[i] = row p[i]) */
        for (int j = 0; j < i; j++) { w[j] = Mi[p[j]]; mark[j] = w[j].h != 0 || w[j].l != 0; }
        dd d2 = dd_of(0.0);
        for (int j = 0; j < i; j++) {
            if (!mark[j]) continue;
            mark[j] = 0;
            const dd x = dd_div(w[j], L[(size_t)j * m + j]);
            L[(size_t)i * m + j] = x; d2 = dd_add(d2, dd_mul(x, x));
            const int *c = S->cs[j]; const int nc = S->cn[j];
            for (int q = 0; q < nc; q++) { const int k = c[q]; w[k] = dd_sub(w[k], dd_mul(L[(size_t)k * m + j], x)); mark[k] = 1; }
            fl += nc + 1;
            if (S->cn[j] == S->ccap[j]) { S->ccap[j] = 2 * S->ccap[j] + 8; S->cs[j] = (int *)realloc(S->cs[j], sizeof(int) * (size_t)S->ccap[j]); }
            S->cs[j][S->cn[j]++] = i;
            if (S->rn[i] == S->rcap[i]) { S->rcap[i] = 2 * S->rcap[i] + 8; S->rs[i] = (int *)realloc(S->rs[i], sizeof(int) * (size_t)S->rcap[i]); }
            S->rs[i][S->rn[i]++] = j;
        }
        const dd d = dd_sub(Mi[p[i]], d2);
        if (!(d.h > DSP_DEP * Mi[p[i]].h)) {
            if (!(Mi[p[i]].h > 0) || d.h < -1e-12 * Mi[p[i]].h) { S->flops = fl; return 1; }
            /* dependent: the row leaves the factor (its entries to zero, a huge pivot) */
            for (int q = 0; q < S->rn[i]; q++) { const int j = S->rs[i][q]; L[(size_t)i * m + j] = dd_of(0.0); S->cn[j]--; }
            S->rn[i] = 0;
            L[(size_t)i * m + i] = dd_of(1e150 * Mi[p[i]].h); S->ndep++; continue;
        }
        L[(size_t)i * m + i] = dd_sqrt(d);
    }
    S->flops = fl;
    return 0;
}
static void dsp_solve(const DSp *S, dd *x) {
    if (S->dense) { dd_solve_chol(S->m, S->LD, x); return; }
    const int m = S->m; const int *p = S->perm; dd *t = S->t; const dd *L = S->L;
    for (int i = 0; i < m; i++) t[i] = x[p[i]];
    for (int i = 0; i < m; i++) {
        dd s = t[i]; const int *r = S->rs[i]; const dd *Li = L + (size_t)i * m;
        for (int q = 0; q < S->rn[i]; q++) s = dd_sub(s, dd_mul(Li[r[q]], t[r[q]]));
        t[i] = dd_div(s, Li[i]);
    }
    for (int i = m - 1; i >= 0; i--) {
        const int *r = S->rs[i]; const dd *Li = L + (size_t)i * m;
        t[i] = dd_div(t[i], Li[i]);
        for (int q = 0; q < S->rn[i]; q++) t[r[q]] = dd_sub(t[r[q]], dd_mul(Li[r[q]], t[i]));
    }
    for (int i = 0; i < m; i++) x[p[i]] = t[i];
}

/* 5.9: the bordered solve for free variables. On entry dy = M^-1 rhs; on exit dy and f_new solve
 * M dy + A_f f_new = rhs, A_f' dy = g, with V = M^-1 A_f and the Cholesky factor of A_f' V. */
static void dd_border(int m, int nf, const dd *Af, const dd *Vf, const dd *LSf, const dd *g, dd *dy, dd *fnew) {
    for (int f = 0; f < nf; f++) {
        dd s = dd_of(0.0);
        for (int i = 0; i < m; i++) s = dd_add(s, dd_mul(Af[i + (size_t)f * m], dy[i]));
        fnew[f] = dd_sub(s, g[f]);
    }
    dd_solve_chol(nf, LSf, fnew);
    for (int f = 0; f < nf; f++) for (int i = 0; i < m; i++) dy[i] = dd_sub(dy[i], dd_mul(Vf[i + (size_t)f * m], fnew[f]));
}
/* the direction of the free variables: the + slot moves to f_new, nothing else of a pair moves */
static void dd_free_dir(DBlk *b, const FreePair *prs, int nf, const dd *fnew) {
    for (int f = 0; f < nf; f++) {
        DBlk *bk = &b[prs[f].blk];
        bk->dX[prs[f].ip] = dd_sub(fnew[f], bk->X[prs[f].ip]); bk->dX[prs[f].im] = dd_of(0.0);
        bk->dZ[prs[f].ip] = dd_of(0.0); bk->dZ[prs[f].im] = dd_of(0.0);
    }
}

/* ---------------------------------------------------------------- endgame */
/* ---------------------------------------------------------------- 5.16 (experiment, BRISK_DDQR=1):
 * the Schur system of the endgame solved with a double-precision square-root factor instead of the
 * dd assembly and dd Cholesky: the Gram vectors g_i = vec(L_X' A_i R_Z) (X = L_X L_X', Z^-1 = R_Z R_Z',
 * from the dd iterate rounded to double), column-equilibrated, their QR by row chunks (dgeqrf on
 * [R; chunk]), and each solve refined BRISK_DDQRREF times (default 3) with the residual of the dd
 * system computed through the operator, r = rhs - A(X A'(dy) Z^-1), in double-double (2n^3 dd per
 * block per pass). The factor's error is kappa(G) u (REPORT_5.16 section 15), the refinement brings
 * the direction to the accuracy of the dd system; the cost is 2.7 N m^2 in double against the dd
 * assembly and Cholesky (45x each). */
typedef struct { int m; size_t N; double *G, *D, *Lq; } DDQR;
static const int DDQR_ONE = 1;
extern void BL(dgeqrf_)(const int *, const int *, double *, const int *, double *, double *, const int *, int *);
static int ddqr_build(const Problem *P, const DBlk *b, int nb, int m, DDQR *q, int verbose) {
    const double t0 = dd_now();
    size_t N = 0;
    for (int k = 0; k < nb; k++) { if (P->blk[k].type == BLK_SOC) return 1; N += b[k].len; }
    double *G = calloc(N * (size_t)m, sizeof(double));
    if (!G) return 1;
    size_t off = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; const int n = b[k].n;
        if (B->type == BLK_LP) {
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *A = &B->A[t]; double *g = G + off + (size_t)B->con[t] * N;
                for (int p = 0; p < A->nnz; p++) { const int r = A->row[p]; g[r] = A->val[p] * sqrt(dd_d(b[k].X[r]) * dd_d(b[k].Zi[r])); }
            }
            off += (size_t)n; continue;
        }
        int info;
        double *LX = malloc(sizeof(double) * b[k].len), *RZ = malloc(sizeof(double) * b[k].len);
        double *Ad = malloc(sizeof(double) * b[k].len);
        for (size_t i = 0; i < b[k].len; i++) { LX[i] = dd_d(b[k].X[i]); RZ[i] = dd_d(b[k].Zi[i]); }
        BL(dpotrf_)("L", &n, LX, &n, &info); if (info) { free(LX); free(RZ); free(Ad); free(G); return 1; }
        BL(dpotrf_)("L", &n, RZ, &n, &info); if (info) { free(LX); free(RZ); free(Ad); free(G); return 1; }
        const double one = 1.0;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t];
            memset(Ad, 0, sizeof(double) * b[k].len);
            for (int p = 0; p < A->nnz; p++) { const int r = A->row[p], c = A->col[p]; Ad[r + (size_t)c * n] = A->val[p]; Ad[c + (size_t)r * n] = A->val[p]; }
            BL(dtrmm_)("L", "L", "T", "N", &n, &n, &one, LX, &n, Ad, &n);
            BL(dtrmm_)("R", "L", "N", "N", &n, &n, &one, RZ, &n, Ad, &n);
            memcpy(G + off + (size_t)B->con[t] * N, Ad, sizeof(double) * b[k].len);
        }
        free(LX); free(RZ); free(Ad);
        off += b[k].len;
    }
    /* column equilibration */
    double *D = malloc(sizeof(double) * m);
    for (int j = 0; j < m; j++) {
        double s = 0; const double *g = G + (size_t)j * N;
        for (size_t p = 0; p < N; p++) s += g[p] * g[p];
        D[j] = s > 0 ? 1.0 / sqrt(s) : 1.0;
        double *gw = G + (size_t)j * N; for (size_t p = 0; p < N; p++) gw[p] *= D[j];
    }
    /* the QR by row chunks */
    const int c = (int)fmin((double)N, fmax((double)m, 512.0)), ld = m + c;
    double *St = calloc((size_t)ld * m, sizeof(double)), *tau = malloc(sizeof(double) * m);
    int lwork = -1, info; double wq;
    BL(dgeqrf_)(&ld, &m, St, &ld, tau, &wq, &lwork, &info);
    lwork = (int)wq + 1; double *work = malloc(sizeof(double) * lwork);
    for (size_t r0 = 0; r0 < N; r0 += c) {
        const int rows = (int)fmin((double)c, (double)(N - r0)), tot = m + rows;
        for (int j = 0; j < m; j++) {
            double *col = St + (size_t)j * ld;
            for (int i = j + 1; i < m; i++) col[i] = 0.0;
            memcpy(col + m, G + r0 + (size_t)j * N, sizeof(double) * rows);
        }
        BL(dgeqrf_)(&tot, &m, St, &ld, tau, work, &lwork, &info);
        if (info) { free(St); free(tau); free(work); free(G); free(D); return 1; }
    }
    double *Lq = calloc((size_t)m * m, sizeof(double));
    for (int j = 0; j < m; j++) for (int i = j; i < m; i++) Lq[i + (size_t)j * m] = St[j + (size_t)i * ld];
    free(St); free(tau); free(work);
    free(q->G); free(q->D); free(q->Lq);
    q->m = m; q->N = N; q->G = G; q->D = D; q->Lq = Lq;
    if (verbose > 1) printf("   dd endgame: square-root factor, N %zu, m %d, %.3f s\n", N, m, dd_now() - t0);
    return 0;
}
/* out = M v through the operator in dd: (M v)_t = <A_t, X A'(v) Z^-1> */
static void ddqr_mv(const Problem *P, DBlk *b, int nb, const dd *v, dd *out) {
    const int m = P->m;
    memset(out, 0, sizeof(dd) * m);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; const int n = b[k].n;
        dd_ATy(B, &b[k], v, b[k].T1);
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) b[k].T3[i] = dd_mul(b[k].X[i], dd_mul(b[k].T1[i], b[k].Zi[i]));
        } else {
            dd_gemm(n, b[k].X, b[k].T1, b[k].T2);
            dd_gemm(n, b[k].T2, b[k].Zi, b[k].T3);
        }
        dd_Aop(B, &b[k], b[k].T3, out, 1.0);
    }
}
static void ddqr_solve(const Problem *P, DBlk *b, int nb, const DDQR *q, const dd *rhs, dd *x, int verbose) {
    const int m = q->m;
    static int nref = -1; if (nref < 0) { const char *e = getenv("BRISK_DDQRREF"); nref = e ? atoi(e) : 3; }
    double *w = malloc(sizeof(double) * m); dd *r = dalloc(m), *t = dalloc(m);
    for (int i = 0; i < m; i++) w[i] = dd_d(rhs[i]) * q->D[i];
    BL(dtrsv_)("L", "N", "N", &m, q->Lq, &m, w, &DDQR_ONE); BL(dtrsv_)("L", "T", "N", &m, q->Lq, &m, w, &DDQR_ONE);
    for (int i = 0; i < m; i++) x[i] = dd_of(w[i] * q->D[i]);
    for (int pass = 0; pass < nref; pass++) {
        ddqr_mv(P, b, nb, x, t);
        double rn = 0, bn = 0;
        for (int i = 0; i < m; i++) { r[i] = dd_sub(rhs[i], t[i]); rn += dd_d(r[i]) * dd_d(r[i]); bn += dd_d(rhs[i]) * dd_d(rhs[i]); }
        if (getenv("BRISK_DDREFDBG")) printf("      dd square-root solve: pass %d residual %.2e\n", pass, sqrt(rn / (bn > 0 ? bn : 1)));
        for (int i = 0; i < m; i++) w[i] = dd_d(r[i]) * q->D[i];
        BL(dtrsv_)("L", "N", "N", &m, q->Lq, &m, w, &DDQR_ONE); BL(dtrsv_)("L", "T", "N", &m, q->Lq, &m, w, &DDQR_ONE);
        for (int i = 0; i < m; i++) x[i] = dd_add(x[i], dd_of(w[i] * q->D[i]));
    }
    free(w); free(r); free(t);
}
/* 5.16: the Schur complement assembled, factored, and the free-pair border prepared (Vf = M^-1 Af,
 * S = Af' Vf, its dd Cholesky). Returns 1 on failure (message printed). */
static int dd_build_factor(const Problem *P, const DBlk *b, int nb, int m, dd *M, DSp *SP, int nf, const dd *Af, dd *Vf, dd *Sf, dd *LSf, int verbose) {
    memset(M, 0, sizeof(dd) * (size_t)m * m);
    for (int k = 0; k < nb; k++) dd_schur(&P->blk[k], &b[k], m, M);
    for (int j = 0; j < m; j++)
        for (int i = j + 1; i < m; i++) M[j + (size_t)i * m] = M[i + (size_t)j * m];
    if (!SP->perm) dsp_init(SP, m, M);
    const double tf = dd_now();
    const int rf = dsp_factor(SP, M);
    g_ddf_fact += dd_now() - tf;
    if (rf) {                                  /* tiny shift if needed */
        for (int i = 0; i < m; i++) M[i + (size_t)i * m] = dd_add(M[i + (size_t)i * m], dd_muld(M[i + (size_t)i * m], 1e-24));
        if (dsp_factor(SP, M)) { if (verbose > 0) printf("   dd endgame: the Schur complement is not positive definite: stopping\n"); return 1; }
    }
    if (nf > 0) {
        memcpy(Vf, Af, sizeof(dd) * (size_t)m * nf);
        for (int f = 0; f < nf; f++) dsp_solve(SP, Vf + (size_t)f * m);
        for (int f = 0; f < nf; f++) for (int g = 0; g <= f; g++) {
            dd s = dd_of(0.0);
            for (int i = 0; i < m; i++) s = dd_add(s, dd_mul(Af[i + (size_t)f * m], Vf[i + (size_t)g * m]));
            Sf[f + (size_t)g * nf] = s; Sf[g + (size_t)f * nf] = s;
        }
        if (dd_chol(nf, Sf, LSf)) { if (verbose > 0) printf("   dd endgame: the free variables' system is not positive definite: stopping\n"); return 1; }
    }
    return 0;
}
/* 5.16: the (bordered) Newton system [M Af; Af' 0] [dy; f] = [rhs; g] solved with the factor SP (and
 * Vf, LSf) and refined with the residual of the dd system computed through the operator (ddqr_mv:
 * A(X A'(dy) Z^-1), 2n^3 dd per block per pass). With a fresh factor one or two passes restore the
 * digits the factorization's rounding took; with the factor of an earlier iteration (reuse) the
 * passes converge at the rate ||I - M_old^-1 M||, each costing a small fraction of an assembly.
 * Returns the passes used, or -1 when the residual stopped falling (the factor is too stale). */
static int dd_solve_b(const Problem *P, DBlk *b, int nb, const DSp *SP, int nf, const dd *Af, const dd *Vf, const dd *LSf,
                      const dd *rhs, const dd *g, dd *dy, dd *fnew, int maxpass, double *rel_out) {
    const int m = P->m;
    memcpy(dy, rhs, sizeof(dd) * m);
    dsp_solve(SP, dy);
    if (nf > 0) dd_border(m, nf, Af, Vf, LSf, g, dy, fnew);
    if (maxpass <= 0) { if (rel_out) *rel_out = -1; return 0; }
    dd *r = dalloc(m), *t = dalloc(m), *r2 = nf > 0 ? dalloc(nf) : NULL, *df = nf > 0 ? dalloc(nf) : NULL;
    double bn = 0; for (int i = 0; i < m; i++) bn += dd_d(rhs[i]) * dd_d(rhs[i]); bn = sqrt(bn);
    double prev = -1, rel = -1; int pass = 0, ok = 1;
    for (; pass < maxpass; pass++) {
        ddqr_mv(P, b, nb, dy, t);
        double rn = 0;
        for (int i = 0; i < m; i++) {
            dd v = dd_sub(rhs[i], t[i]);
            for (int f = 0; f < nf; f++) v = dd_sub(v, dd_mul(Af[i + (size_t)f * m], fnew[f]));
            r[i] = v; rn += dd_d(v) * dd_d(v);
        }
        for (int f = 0; f < nf; f++) {
            dd v = g[f];
            for (int i = 0; i < m; i++) v = dd_sub(v, dd_mul(Af[i + (size_t)f * m], dy[i]));
            r2[f] = v; rn += dd_d(v) * dd_d(v);
        }
        rel = bn > 0 ? sqrt(rn) / bn : 0;
        if (getenv("BRISK_DDREFDBG")) printf("        [refine pass %d: %.2e]\n", pass, rel);
        if (rel <= 1e-24) break;
        if (prev >= 0 && rel > 0.9 * prev) { ok = 0; break; }      /* not converging: the factor is too stale */
        prev = rel;
        dsp_solve(SP, r);
        if (nf > 0) dd_border(m, nf, Af, Vf, LSf, r2, r, df);
        for (int i = 0; i < m; i++) dy[i] = dd_add(dy[i], r[i]);
        for (int f = 0; f < nf; f++) fnew[f] = dd_add(fnew[f], df[f]);
    }
    free(r); free(t); free(r2); free(df);
    if (rel_out) *rel_out = rel;
    return ok ? pass : -1;
}
int dd_endgame(const Problem *P, double **Xio, double **Zio, double *y, const Params *par,
               int maxit, double tol, int verbose, brisk_dd_measure meas, void *mctx) {
    const int m = P->m, nb = P->nblk;
    const double t_start = dd_now(), t_limit = par->dd_time;
    DBlk *b = calloc(nb, sizeof(DBlk));
    double ndim = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        b[k].type = B->type;
        b[k].n = B->n;
        b[k].len = B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
        size_t len = b[k].len;
        dd **arr[] = { &b[k].X, &b[k].Z, &b[k].Zi, &b[k].dX, &b[k].dZ, &b[k].L,
                       &b[k].T1, &b[k].T2, &b[k].T3, &b[k].Rd, &b[k].Q };
        for (size_t q = 0; q < sizeof(arr) / sizeof(arr[0]); q++) *arr[q] = dalloc(len);
        for (size_t i = 0; i < len; i++) { b[k].X[i] = dd_of(Xio[k][i]); b[k].Z[i] = dd_of(Zio[k][i]); }
        ndim += B->n;
    }
    /* 5.9: free variables given as split pairs that the presolve kept (two LP columns a, -a with
     * costs c, -c). As two nonnegative variables they have no interior: the embedding handles them
     * by a saddle factorization and returns slacks of opposite signs, and the Schur complement
     * built from x/z was not positive definite here (the POEMA option-pricing relaxations: the
     * endgame stopped in its first iteration). Now each pair is one free variable f = x+ - x-:
     * its slots stay out of the cone terms, and the Newton system is bordered,
     *     M dy + A_f f_new = rhs,   A_f' dy = c_f - A_f' y,
     * solved through S = A_f' M^-1 A_f (nf x nf, Cholesky in dd). */
    FreePair *prs = NULL; const int nfree = getenv("BRISK_DDNOFREE") ? 0 : free_pairs_detect(P, &prs);
    dd *Af = NULL, *Vf = NULL, *Sf = NULL, *LSf = NULL, *fnew = NULL, *gf = NULL;
    if (nfree > 0) {
        for (int f = 0; f < nfree; f++) {
            DBlk *bk = &b[prs[f].blk];
            if (!bk->fr) bk->fr = (char *)calloc((size_t)bk->n + 1, 1);
            bk->fr[prs[f].ip] = 1; bk->fr[prs[f].im] = 2;
            bk->X[prs[f].ip] = dd_sub(bk->X[prs[f].ip], bk->X[prs[f].im]); bk->X[prs[f].im] = dd_of(0.0);
            bk->Z[prs[f].ip] = dd_of(0.0); bk->Z[prs[f].im] = dd_of(0.0);
        }
        ndim -= 2.0 * nfree; if (ndim < 1) ndim = 1;
        Af = dalloc((size_t)m * nfree); Vf = dalloc((size_t)m * nfree); Sf = dalloc((size_t)nfree * nfree); LSf = dalloc((size_t)nfree * nfree);
        fnew = dalloc(nfree); gf = dalloc(nfree);
        for (int f = 0; f < nfree; f++) {            /* the column of the + slot */
            const Block *B = &P->blk[prs[f].blk];
            for (int t = 0; t < B->ncon; t++) { const SpSym *S = &B->A[t]; for (int q = 0; q < S->nnz; q++) if (S->row[q] == prs[f].ip) Af[B->con[t] + (size_t)f * m] = dd_add(Af[B->con[t] + (size_t)f * m], dd_of(S->val[q])); }
        }
    }
    static int ddqr_on = -1; if (ddqr_on < 0) ddqr_on = getenv("BRISK_DDQR") != NULL;
    /* 5.16: each solve refined with two passes against the dd system through the operator (dd_solve_b) */
    static int reuse_on = 1;
    int n_asm = 0;
    dd *M = ddqr_on ? NULL : dalloc((size_t)m * m);
    DSp SP; memset(&SP, 0, sizeof SP);
    DDQR QRF; memset(&QRF, 0, sizeof QRF);
    /* best iterate so far (the time limit may stop the loop at a worse point) */
    double **Xb = malloc(sizeof(double *) * nb), **Zb = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) { Xb[k] = malloc(sizeof(double) * b[k].len); Zb[k] = malloc(sizeof(double) * b[k].len); }
    double *yb = malloc(sizeof(double) * (m + 1));
    double **Xc = malloc(sizeof(double *) * nb), **Zc = malloc(sizeof(double *) * nb), *yc = malloc(sizeof(double) * (m + 1));
    for (int k = 0; k < nb; k++) { Xc[k] = malloc(sizeof(double) * b[k].len); Zc[k] = malloc(sizeof(double) * b[k].len); }
    double best_d = 1e300, sd0 = 1e300; int nostall_d = 0, improved_d = 0;
    double last_ap = 0, last_ad = 0;   /* the last corrector step's lengths (printed) */
    /* BRISK_DDDIAG (14 October): sigma, mu_aff/mu and the block (and LP entry) that bounds each step, with its
     * smallest eigenvalues: the creeping of the endgame on blocks heading for a face was diagnosed with it
     * (ISSUES 86) */
    double last_sig = 0, last_frac = 0; int last_lp = -1, last_ld = -1;
    const int dddiag = getenv("BRISK_DDDIAG") != NULL;
    dd *rp = dalloc(m), *rhs = dalloc(m), *dy = dalloc(m), *yv = dalloc(m);
    /* `col` is used as a per-block scratch column: size it by the largest block, not m
     * (ss30 has m = 132 and a block of 294, which overran the buffer) */
    size_t colmax = (size_t)m + 1;
    for (int k = 0; k < nb; k++) if ((size_t)P->blk[k].n + 1 > colmax) colmax = (size_t)P->blk[k].n + 1;
    dd *a0 = dalloc(m), *hv = dalloc(m), *qv = dalloc(m), *col = dalloc(colmax);
    for (int i = 0; i < m; i++) yv[i] = dd_of(y[i]);
    const double sc = P->bs * P->cs;
    double best = 1e300;
    int used = 0, improved = 0;

    double score0 = -1;
    int nostall = 0;
    /* 5.16: the endgame starts inside the cone. The double iterate it is given (the embedding's probe
     * candidate, a least-squares correction) can lie outside the cone by rounding, 1e-9 to 1e-8 in
     * lambda_min (qap6: mu = <X,Z>/n = -1.1e-9 in dd, hinf1: X not positive definite in double): the
     * endgame then stopped in its first iteration, and the 4.14 gains on the qap family had been lost.
     * A block whose smallest eigenvalue is negative is shifted by twice it (X += 2|lambda_min| I, the
     * same for Z: lambda_min becomes |lambda_min|), a perturbation of the residuals of that size that
     * the first dd iterations remove. SDPLIB OPTIMAL 70 -> 72, occupation measures +2, option pricing
     * unchanged (14 October). BRISK_DDSHIFT=<f> sets the factor, 0 turns it off; -vv prints lambda_min. */
    {
        static double shift_on = -1; if (shift_on < 0) { const char *e = getenv("BRISK_DDSHIFT"); shift_on = e ? atof(e) : 2.0; }   /* the factor on |lambda_min|; default 2 (5.16), 0 off */
        for (int k = 0; k < nb; k++) {
            const int n = b[k].n; double lx = 1e300, lz = 1e300;
            if (b[k].type == BLK_LP) {
                for (int i = 0; i < n; i++) { if (b[k].fr && b[k].fr[i]) continue; if (dd_d(b[k].X[i]) < lx) lx = dd_d(b[k].X[i]); if (dd_d(b[k].Z[i]) < lz) lz = dd_d(b[k].Z[i]); }
            } else {
                double *T = malloc(sizeof(double) * b[k].len), *ev = malloc(sizeof(double) * n), wq; int lwork = -1, info;
                for (int pass = 0; pass < 2; pass++) {
                    for (size_t i = 0; i < b[k].len; i++) T[i] = dd_d(pass ? b[k].Z[i] : b[k].X[i]);
                    BL(dsyev_)("N", "L", &n, T, &n, ev, &wq, &lwork, &info); lwork = (int)wq + 1;
                    double *work = malloc(sizeof(double) * lwork);
                    BL(dsyev_)("N", "L", &n, T, &n, ev, work, &lwork, &info); free(work); lwork = -1;
                    if (info == 0) { if (pass) lz = ev[0]; else lx = ev[0]; }
                }
                free(T); free(ev);
            }
            if (verbose > 1) printf("   dd endgame start: block %d lambda_min(X) %.3e lambda_min(Z) %.3e\n", k + 1, lx, lz);
            if (shift_on) {
                if (lx < 0) { const dd d = dd_of(-shift_on * lx); if (b[k].type == BLK_LP) { for (int i = 0; i < n; i++) if (!(b[k].fr && b[k].fr[i])) b[k].X[i] = dd_add(b[k].X[i], d); } else for (int i = 0; i < n; i++) b[k].X[i + (size_t)i * n] = dd_add(b[k].X[i + (size_t)i * n], d); if (verbose > 0) printf("   dd endgame: X of block %d shifted into the cone by %.2e\n", k + 1, -shift_on * lx); }
                if (lz < 0) { const dd d = dd_of(-shift_on * lz); if (b[k].type == BLK_LP) { for (int i = 0; i < n; i++) if (!(b[k].fr && b[k].fr[i])) b[k].Z[i] = dd_add(b[k].Z[i], d); } else for (int i = 0; i < n; i++) b[k].Z[i + (size_t)i * n] = dd_add(b[k].Z[i + (size_t)i * n], d); if (verbose > 0) printf("   dd endgame: Z of block %d shifted into the cone by %.2e\n", k + 1, -shift_on * lz); }
            }
        }
    }
    double best_inf = 1e300, prev_score = 1e300;
    int started = 0, extra = 0;
    const int maxit0 = maxit;
    static int ddtime = -1; if (ddtime < 0) ddtime = getenv("BRISK_DDTIME") != NULL;
    double tph[8] = {0,0,0,0,0,0,0,0}, tp0 = dd_now();
    #define DDPH(i) do { if (ddtime) { const double _t = dd_now(); tph[i] += _t - tp0; tp0 = _t; } } while (0)
    for (int it = 0; it <= maxit; it++) {
        if (t_limit > 0 && dd_now() - t_start > t_limit) {
            if (verbose > 0) printf("   dd endgame: time limit (%.1fs) reached\n", t_limit);
            break;
        }
        if (brisk_stop_flag || (par->timelimit > 0 && dd_now() - par->t_start > par->timelimit)) break;
        /* ---- residuals, objectives */
        for (int i = 0; i < m; i++) rp[i] = dd_of(P->b[i]);
        dd pobj = dd_of(0.0), dobj = dd_of(0.0), xz = dd_of(0.0);
        double rd2 = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            dd_Aop(B, &b[k], b[k].X, rp, -1.0);
            dd_ATy(B, &b[k], yv, b[k].Rd);
            for (size_t i = 0; i < b[k].len; i++) b[k].Rd[i] = dd_neg(dd_add(b[k].Rd[i], b[k].Z[i]));
            dd_add_sp(B, &b[k], &B->C, 1.0, b[k].Rd);                    /* Rd = C - Z - A'y */
            for (size_t i = 0; i < b[k].len; i++) {
                double v = dd_d(b[k].Rd[i]);
                rd2 += v * v;
                xz = dd_add(xz, dd_mul(b[k].X[i], b[k].Z[i]));
            }
            dd s = dd_of(0.0);
            dd_add_sp(B, &b[k], &B->C, 1.0, b[k].T1);                    /* <C, X> */
            for (size_t i = 0; i < b[k].len; i++) s = dd_add(s, dd_mul(b[k].T1[i], b[k].X[i]));
            memset(b[k].T1, 0, sizeof(dd) * b[k].len);
            pobj = dd_add(pobj, s);
        }
        for (int i = 0; i < m; i++) dobj = dd_add(dobj, dd_mul(dd_of(P->b[i]), yv[i]));
        double rpo = 0;
        for (int i = 0; i < m; i++) { double v = dd_d(rp[i]) * P->du[i]; rpo += v * v; }
        rpo = sqrt(rpo);
        double po = sc * dd_d(pobj), dob = sc * dd_d(dobj), den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        double pinf = rpo / (1 + P->normb2), dinf = P->cs * sqrt(rd2) / (1 + P->normC2);
        double gap = fabs(po - dob) / den, comp = fabs(sc * dd_d(xz)) / den;
        double score = fmax(fmax(pinf, dinf), fmax(gap, comp));
        double mu = dd_d(xz) / ndim;
        if (verbose > 0)
            printf("   dd %2d  pinf %.2e  dinf %.2e  gap %.2e  compl %.2e  mu %.2e  steps %.3f %.3f\n",
                   it, pinf, dinf, gap, comp, mu, last_ap, last_ad);
        if (verbose > 0 && dddiag && it > 0) {
            printf("      (sigma %.3g, mu_aff/mu %.3g; the step bounded by block %d on X, block %d on Z)\n", last_sig, last_frac, last_lp + 1, last_ld + 1);
            for (int side = 0; side < 2; side++) {
                const int k = side ? last_ld : last_lp;
                if (k >= 0 && b[k].type != BLK_LP) { double lx, lz; dd_blk_lmin(&b[k], &lx, &lz); printf("      (SDP block %d of order %d bounds the %s step: now lambda_min(X) %.3e, lambda_min(Z) %.3e, mu %.3e)\n", k + 1, b[k].n, side ? "Z" : "X", lx, lz, mu); }
                if (k >= 0 && b[k].type == BLK_LP) {
                    double r, v, d, v2, d2; int i; dd_lp_limiter(&b[k], side, &r, &i, &v, &d, &v2, &d2);
                    if (i >= 0) printf("      (LP block %d, %s side: entry %d of %d bounds the step at %.3g: %s %.3e d%s %.3e, %s %.3e d%s %.3e)\n", k + 1, side ? "Z" : "X", i, b[k].n, r, side ? "z" : "x", v, side ? "z" : "x", d, side ? "x" : "z", v2, side ? "x" : "z", d2);
                }
            }
        }
        if (score0 < 0) score0 = score;
        /* stall exit: stop after 6 iterations that do not improve the best score by 10%
         * (the endgame is also bounded by its work budget; on the LMI families it
         * converges slowly but steadily, which a stricter rule cuts off)              */
        /* progress also counts when infeasibility falls while the gap is still opening
         * up (hinf9: 13 such iterations, then quadratic convergence)                  */
        const double inf = fmax(pinf, dinf);
        if (score < 0.9 * best || inf < 0.9 * best_inf) nostall = 0; else if (it > 0) nostall++;
        if (inf < best_inf) best_inf = inf;
        /* 5.16: the iterate returned is the best by the solver's own double-precision measure (meas), not by
         * the dd score: past the point where X's small eigenvalues fall below double resolution the dd score
         * keeps falling while the double-rounded point gets worse (attr_vdp_d14_I: dd 5 at compl 6e-9 measures
         * 7e-8 in double and polishes to OPTIMAL, dd 7 at 4e-10 measures 8e-7); the endgame stops once that
         * measure has not improved for three iterations */
        int keep = 0; double sd_it = 1e300;
        if (meas) {
            for (int k = 0; k < nb; k++)
                for (size_t i = 0; i < b[k].len; i++) { Xc[k][i] = dd_d(b[k].X[i]); Zc[k][i] = dd_d(b[k].Z[i]); }
            for (int f = 0; f < nfree; f++) {
                const double v = dd_d(b[prs[f].blk].X[prs[f].ip]);
                Xc[prs[f].blk][prs[f].ip] = v > 0 ? v : 0.0; Xc[prs[f].blk][prs[f].im] = v < 0 ? -v : 0.0;
            }
            for (int i = 0; i < m; i++) yc[i] = dd_d(yv[i]);
            const double sd = meas(mctx, Xc, Zc, yc); sd_it = sd;
            if (verbose > 1) printf("      (the double-precision measure of this iterate: %.2e)\n", sd);
            if (it == 0) sd0 = sd;
            if (sd < best_d * (1 - 1e-12)) { best_d = sd; keep = 1; nostall_d = 0; if (sd < sd0) improved_d = 1; }
            else nostall_d = nostall > 0 ? nostall_d + 1 : 0;      /* a stall only where the dd rule sees none of its progress either (infeasibility falling while the gap opens) */
        } else if (score < best * (1 - 1e-12)) keep = 1;
        if (score < best * (1 - 1e-12)) { best = score; if (score < score0) improved = 1; }
        if (keep) {
            for (int k = 0; k < nb; k++)
                for (size_t i = 0; i < b[k].len; i++) { Xb[k][i] = dd_d(b[k].X[i]); Zb[k][i] = dd_d(b[k].Z[i]); }
            for (int f = 0; f < nfree; f++) {        /* the free value as its two parts */
                const double v = dd_d(b[prs[f].blk].X[prs[f].ip]);
                Xb[prs[f].blk][prs[f].ip] = v > 0 ? v : 0.0; Xb[prs[f].blk][prs[f].im] = v < 0 ? -v : 0.0;
            }
            for (int i = 0; i < m; i++) yb[i] = dd_d(yv[i]);
        }
        if (meas && best_d < tol) { used = 1; break; }
        /* 5.16: a first step that worsens the double measure means the start is not in the endgame's regime (a
         * user's nishioka file: the gap 2.6e-5 -> 1.4e-2 at the first step, then three more iterations of 70 s
         * for nothing); the start's transient after the shift is in sd0 already (measured after the shift) */
        /* (15 October 2026, the owner's benchmark, ISSUES 99: only a step that worsens the measure tenfold; a first
         * step that leaves it where it was (hinf12: 9.1e-2 from 9.1e-2, hinf5, truss7, hinf1 "x -> x rejected") is
         * the start of a run that reached 5.8e-5 on hinf12 before the rule; the stall rule below ends it otherwise) */
        if (meas && it == 1 && sd0 < 1e300 && sd_it > 10.0 * sd0) { if (verbose > 0) printf("   dd endgame: the first step worsens the double-precision measure (%.1e from %.1e): stopping\n", sd_it, sd0); break; }
        if (meas && nostall_d >= 3 && it >= 3) { if (verbose > 1) printf("   dd endgame: the double-precision measure stopped improving: stopping\n"); used = 1; break; }
        if (!meas && score < tol) { used = 1; break; }
        /* 5.9: at the end of the budget, up to half as many iterations more while each one halves
         * the score (the budget ended runs that were three iterations from the tolerance) */
        if (it == maxit && it > 0 && extra < maxit0 / 2 + 1 && score < 0.5 * prev_score && mu > 0) { maxit++; extra++; }
        prev_score = score;
        if (!(mu > 0) || it == maxit || nostall >= 6) break;

        DDPH(0);
        /* ---- Z^{-1}, Schur complement, Cholesky */
        int fail = 0;
        for (int k = 0; k < nb && !fail; k++) {
            const int n = b[k].n;
            if (b[k].type == BLK_LP) {
                for (int i = 0; i < n; i++) b[k].Zi[i] = b[k].fr && b[k].fr[i] ? dd_of(0.0) : dd_div(dd_of(1.0), b[k].Z[i]);
            } else {
                if (dd_chol(n, b[k].Z, b[k].L)) { fail = 1; if (verbose > 0) printf("   dd endgame: Z of block %d is not positive definite in double-double: stopping\n", k + 1); break; }
                dd_inv_from_chol(n, b[k].L, b[k].Zi, col);
                dd_sym(n, b[k].Zi);
            }
        }
        if (fail) break;
        DDPH(1);
        const int stale = 0;   /* (reuse of the factor across iterations was tried on 14 October: the matrix changes by orders of magnitude between dd iterations, the refinement never converged) */
        if (ddqr_on) {
            if (ddqr_build(P, b, nb, m, &QRF, verbose)) { if (verbose > 0) printf("   dd endgame: the square-root factor failed: stopping\n"); break; }
            if (nfree > 0) {
                memcpy(Vf, Af, sizeof(dd) * (size_t)m * nfree);
                for (int f = 0; f < nfree; f++) { memcpy(qv, Af + (size_t)f * m, sizeof(dd) * m); ddqr_solve(P, b, nb, &QRF, qv, Vf + (size_t)f * m, verbose); }
                for (int f = 0; f < nfree; f++) for (int g = 0; g <= f; g++) {
                    dd s = dd_of(0.0);
                    for (int i = 0; i < m; i++) s = dd_add(s, dd_mul(Af[i + (size_t)f * m], Vf[i + (size_t)g * m]));
                    Sf[f + (size_t)g * nfree] = s; Sf[g + (size_t)f * nfree] = s;
                }
                if (dd_chol(nfree, Sf, LSf)) { if (verbose > 0) printf("   dd endgame: the free variables' system is not positive definite: stopping\n"); break; }
            }
        } else if (!stale) {
            n_asm++;
            if (dd_build_factor(P, b, nb, m, M, &SP, nfree, Af, Vf, Sf, LSf, verbose)) break;
            DDPH(2);
        }
        /* the dual residual of the free variables: c_f - A_f' y (Rd of the + slot, Z = 0 there) */
        for (int f = 0; f < nfree; f++) gf[f] = b[prs[f].blk].Rd[prs[f].ip];

        DDPH(3);
        /* ---- right-hand side pieces: a0 = A(Zi), h = A(Zi Rd X) */
        memset(a0, 0, sizeof(dd) * m);
        memset(hv, 0, sizeof(dd) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            const int n = b[k].n;
            dd_Aop(B, &b[k], b[k].Zi, a0, 1.0);
            if (b[k].type == BLK_LP)
                for (int i = 0; i < n; i++) b[k].T1[i] = dd_mul(b[k].Zi[i], dd_mul(b[k].Rd[i], b[k].X[i]));
            else {
                dd_gemm(n, b[k].Zi, b[k].Rd, b[k].T2);
                dd_gemm(n, b[k].T2, b[k].X, b[k].T1);
                dd_sym(n, b[k].T1);
            }
            dd_Aop(B, &b[k], b[k].T1, hv, 1.0);
        }

        DDPH(4);
        /* ---- predictor: M dy = b + h ; dZ = Rd - A'dy ; dX = -sym(Zi dZ X) - X */
        for (int i = 0; i < m; i++) rhs[i] = dd_add(dd_of(P->b[i]), hv[i]);
        if (ddqr_on) { ddqr_solve(P, b, nb, &QRF, rhs, dy, verbose); if (nfree > 0) dd_border(m, nfree, Af, Vf, LSf, gf, dy, fnew); }
        else {
            dd_solve_b(P, b, nb, &SP, nfree, Af, Vf, LSf, rhs, gf, dy, fnew, reuse_on ? 2 : 0, NULL);
        }
        for (int k = 0; k < nb; k++) dd_dir(&P->blk[k], &b[k], dy, dd_of(0.0), 0);
        if (nfree > 0) dd_free_dir(b, prs, nfree, fnew);
        dd ap, ad;
        DDPH(5);
        dd_steps(P, b, nb, &ap, &ad);
        DDPH(6);

        /* ---- centering parameter from the affine step */
        dd xza = dd_of(0.0);
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < b[k].len; i++) {
                dd xa = dd_add(b[k].X[i], dd_mul(ap, b[k].dX[i]));
                dd za = dd_add(b[k].Z[i], dd_mul(ad, b[k].dZ[i]));
                xza = dd_add(xza, dd_mul(xa, za));
            }
        double mua = dd_d(xza) / ndim, frac = mua / mu;
        double mn0 = fmin(dd_d(ap), dd_d(ad));
        double expo = gap > 1e-6 ? fmax(1.0, 3.0 * mn0 * mn0) : 3.0;
        double sigma = fmin(1.0, pow(frac, expo));
        last_sig = sigma; last_frac = frac;
        /* (a pure centering first iteration was tried on 14 October for the creeping runs: sigma is already ~1
         * there, the affine step being short; the obstacle is the boundary, not the centrality) */
        dd smu = dd_muld(dd_of(mu), sigma);

        DDPH(4);
        /* ---- corrector: Q = sym(Zi dZa dXa); rhs = b - sigma mu a0 + A(Q) + h */
        memset(qv, 0, sizeof(dd) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            const int n = b[k].n;
            if (b[k].type == BLK_LP)
                for (int i = 0; i < n; i++) b[k].Q[i] = dd_mul(b[k].Zi[i], dd_mul(b[k].dZ[i], b[k].dX[i]));
            else {
                dd_gemm(n, b[k].Zi, b[k].dZ, b[k].T2);
                dd_gemm(n, b[k].T2, b[k].dX, b[k].Q);
                dd_sym(n, b[k].Q);
            }
            dd_Aop(B, &b[k], b[k].Q, qv, 1.0);
        }
        for (int i = 0; i < m; i++)
            rhs[i] = dd_add(dd_sub(dd_of(P->b[i]), dd_mul(a0[i], smu)), dd_add(qv[i], hv[i]));
        if (ddqr_on) { ddqr_solve(P, b, nb, &QRF, rhs, dy, verbose); if (nfree > 0) dd_border(m, nfree, Af, Vf, LSf, gf, dy, fnew); }
        else dd_solve_b(P, b, nb, &SP, nfree, Af, Vf, LSf, rhs, gf, dy, fnew, reuse_on ? 2 : 0, NULL);
        for (int k = 0; k < nb; k++) dd_dir(&P->blk[k], &b[k], dy, smu, 1);
        if (nfree > 0) dd_free_dir(b, prs, nfree, fnew);
        dd_steps(P, b, nb, &ap, &ad);
        DDPH(7);   /* (21 October 2026: the corrector's direction and steps were booked nowhere) */

        double mn = fmin(dd_d(ap), dd_d(ad));
        double gam = 0.9 + (par->gamma_max - 0.9) * mn;
        dd sp = dd_muld(ap, gam), sdd = dd_muld(ad, gam);
        if (dd_d(sp) > 1) sp = dd_of(1.0);
        if (dd_d(sdd) > 1) sdd = dd_of(1.0);
        if (dd_d(sp) < 1e-10 && dd_d(sdd) < 1e-10) { if (verbose > 0) printf("   dd endgame: the step collapsed: stopping\n"); break; }
        for (int k = 0; k < nb; k++) {
            for (size_t i = 0; i < b[k].len; i++) {
                b[k].X[i] = dd_add(b[k].X[i], dd_mul(sp, b[k].dX[i]));
                b[k].Z[i] = dd_add(b[k].Z[i], dd_mul(sdd, b[k].dZ[i]));
            }
            if (b[k].type != BLK_LP) { dd_sym(b[k].n, b[k].X); dd_sym(b[k].n, b[k].Z); }
        }
        last_ap = dd_d(ap); last_ad = dd_d(ad); last_lp = g_lim_p; last_ld = g_lim_d;
        for (int i = 0; i < m; i++) yv[i] = dd_add(yv[i], dd_mul(sdd, dy[i]));
        used = 1; started = 1;
    }

    if (improved) {
        for (int k = 0; k < nb; k++) {
            memcpy(Xio[k], Xb[k], sizeof(double) * b[k].len);
            memcpy(Zio[k], Zb[k], sizeof(double) * b[k].len);
        }
        memcpy(y, yb, sizeof(double) * m);
    }
    for (int k = 0; k < nb; k++) { free(Xb[k]); free(Zb[k]); }
    free(Xb); free(Zb); free(yb);
    for (int k = 0; k < nb; k++) {
        free(b[k].X); free(b[k].Z); free(b[k].Zi); free(b[k].dX); free(b[k].dZ); free(b[k].L);
        free(b[k].T1); free(b[k].T2); free(b[k].T3); free(b[k].Rd); free(b[k].Q);
    }
    for (int k = 0; k < nb; k++) free(b[k].fr);
    free(prs); free(Af); free(Vf); free(Sf); free(LSf); free(fnew); free(gf);
    dsp_free(&SP);
    if (ddtime) printf("   [dd endgame: %d assemblies; assembly: products %.2f, gather %.2f, LP blocks %.2f s; factorization %.2f s (blocked: panels %.2f, trailing updates %.2f)]\n", n_asm, g_dds_prod, g_dds_gath, g_dds_lp, g_ddf_fact, g_ddp_panel, g_ddp_syrk);
    if (ddtime) printf("   [dd phases: residuals %.2f, Zi+chol %.2f, assembly %.2f, factor %.2f, rhs %.2f, predictor %.2f, steps %.2f, corrector %.2f s]\n", tph[0], tph[1], tph[2], tph[3], tph[4], tph[5], tph[6], tph[7]);
    #undef DDPH
    for (int k = 0; k < nb; k++) { free(Xc[k]); free(Zc[k]); }
    free(Xc); free(Zc); free(yc);
    free(b); free(M); free(rp); free(rhs); free(dy); free(yv);
    free(QRF.G); free(QRF.D); free(QRF.Lq);
    free(a0); free(hv); free(qv); free(col);
    if (meas) improved = improved_d;
    return used && improved ? 1 : started ? 0 : -1;       /* 1: an improved point is returned; -1: no step could be taken */
}
