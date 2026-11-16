/* simplex_lu.c - the factorization of the simplex basis and its updates (simplex_int.h).
 *
 * Factorization: right-looking sparse Gaussian elimination with Markowitz pivot choice and threshold
 * partial pivoting (Suhl and Suhl 1990: columns and rows searched by increasing count, a candidate
 * accepted when |a_ij| >= u max_k |a_kj|, the search stopped after a few candidates). Logical columns
 * are pivoted first in bulk; singletons have Markowitz count 0 and are taken before the kernel without
 * a special path (no fill). Columns that cannot be pivoted (structurally or numerically singular basis)
 * are replaced by the logicals of the rows left over.
 *
 * Solves: column etas of L forward, the row etas R of the updates, U backward by columns (FTRAN);
 * U^T forward by rows, R^T backward, L^T backward by rows (BTRAN). When the right-hand side is sparse
 * and the result is expected sparse, the nonzero pattern of the result is found first by a depth-first
 * search over the graph of the factor (Gilbert and Peierls 1988), and only the reached pivots are
 * processed (hypersparse solves, Hall and McKinnon 2005).
 *
 * Updates: Forrest and Tomlin (1972). The entering column's partial FTRAN (after L and R: the spike)
 * replaces the column of the leaving position in U; that position's pivot moves to the end of the
 * order, and its row is eliminated by a row eta whose entries come from the partial BTRAN of the
 * leaving row (after U^T), which the simplex iteration computes anyway. The new diagonal entry is
 * checked against u_rr * alpha_rq (the determinant identity). */
#include "simplex_int.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif

/* (27 October 2026) the simplex family's settings (environment variables BRISK_... and SX...) looked up in a table of
 * those entries alone, one per thread, built again when a new solve has started (simplex_solve calls sx_env_refresh):
 * getenv scans the whole environment, 166 entries in the benchmark's container, and the simplex method reads its
 * settings on every solve and every factorization; on sc205 (LP) its getenv calls were 8.6 % of the solve's
 * instructions. Changes of the environment between solves are seen (the Python tests set variables in the process);
 * other names go to getenv. */
extern char **environ;
#include <time.h>
static double lu_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }   /* (as simplex.c's now) */
static atomic_long g_sx_env_gen = 1;
void sx_env_refresh(void) { atomic_fetch_add_explicit(&g_sx_env_gen, 1, memory_order_relaxed); }
static int sx_env_ours(const char *s) { return (s[0] == 'B' && !strncmp(s, "BRISK_", 6)) || (s[0] == 'S' && s[1] == 'X'); }
#define SX_ENV_MAX 64
static _Thread_local struct { long gen; int n, over; const char *ent[SX_ENV_MAX]; } g_sx_env;
const char *sx_getenv(const char *name) {
    if (!sx_env_ours(name)) return getenv(name);
    const long gen = atomic_load_explicit(&g_sx_env_gen, memory_order_relaxed);
    if (g_sx_env.gen != gen) {
        g_sx_env.gen = gen; g_sx_env.n = 0; g_sx_env.over = 0;
        for (char **e = environ; e && *e; e++) if (sx_env_ours(*e)) { if (g_sx_env.n < SX_ENV_MAX) g_sx_env.ent[g_sx_env.n++] = *e; else g_sx_env.over = 1; }
    }
    if (g_sx_env.over) return getenv(name);
    const size_t l = strlen(name);
    for (int k = 0; k < g_sx_env.n; k++) if (!strncmp(g_sx_env.ent[k], name, l) && g_sx_env.ent[k][l] == '=') return g_sx_env.ent[k] + l + 1;
    return NULL;
}
#define getenv(n) sx_getenv(n)

#define SX_TINY 1e-14          /* entries below this are dropped from solve results */
#define SX_PIVTOL 1e-11        /* absolute pivot tolerance of the factorization */
#define SX_DENSE_MIN 48        /* the dense kernel: at least this many active rows and columns ... */
#define SX_DENSE_FRAC 0.7      /* ... and this fraction of their entries nonzero */
#define SX_DENSE_FRAC_BLK 0.3  /* (the fraction with the blocked dense kernel, dense_kernel_blocked) */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xrealloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }

/* ---------------------------------------------------------------- vectors */
void sxv_init(SxVec *v, int n) {
    v->n = n; v->cnt = 0;
    v->idx = xmalloc(sizeof(int) * ((size_t)n + 8));      /* (8: room for a whole vector block stored at the list's end) */
    v->val = xcalloc((size_t)n + 1, sizeof(double));
    v->mark = xcalloc((size_t)n + 1, 1);
}
static void sxv_prune(SxVec *v);
void sxv_free(SxVec *v) { free(v->idx); free(v->val); free(v->mark); memset(v, 0, sizeof *v); }
#ifndef SX_CLEAR_DIV
#define SX_CLEAR_DIV 8      /* a vector listed with fewer entries than its length over this is cleared by its list */
#endif
void sxv_clear(SxVec *v) {
    if (v->cnt >= 0 && v->cnt < v->n / SX_CLEAR_DIV) for (int k = 0; k < v->cnt; k++) v->val[v->idx[k]] = 0;
    else memset(v->val, 0, sizeof(double) * (size_t)v->n);
    v->cnt = 0;
}
void sxv_reindex(SxVec *v) {
    /* (24 October 2026, the night) without a branch per entry (each index written, the count advanced by whether its entry
     * is nonzero), then the tiny entries dropped from the list: the same list (as the pricing's listing pass, simplex.c).
     * (27 October 2026) One pass that lists the entries not below SX_TINY and zeroes the others, eight at a time where the
     * processor has AVX-512 (the kept positions compressed in a register and stored whole: the list has room for eight
     * more); the same list and values as the two passes (a NaN is kept, as before). The listing loop was the costliest line
     * of scfxm3's solve (LP, 8.7 % of its instructions, 2.5 calls an iteration), the dropping pass another 5 % */
    const int n = v->n; int *restrict ix = v->idx; double *restrict val = v->val;
    int c = 0, i = 0;
#if defined(__AVX512F__) && defined(__AVX512VL__)
    const __m512i absm = _mm512_set1_epi64(0x7fffffffffffffffLL);
    const __m512d tiny = _mm512_set1_pd(SX_TINY), zero = _mm512_setzero_pd();
    __m256i id = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); const __m256i eight = _mm256_set1_epi32(8);
    for (; i + 8 <= n; i += 8) {
        const __m512d x = _mm512_loadu_pd(val + i);
        const __m512d ax = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x), absm));
        const __mmask8 keep = _mm512_cmp_pd_mask(ax, tiny, _CMP_NLT_UQ);
        const __mmask8 drop = _mm512_cmp_pd_mask(x, zero, _CMP_NEQ_UQ) & (__mmask8)~keep;
        if (drop) _mm512_mask_storeu_pd(val + i, drop, zero);
        _mm256_storeu_si256((__m256i *)(ix + c), _mm256_maskz_compress_epi32(keep, id));
        c += __builtin_popcount((unsigned)keep);
        id = _mm256_add_epi32(id, eight);
    }
#endif
    for (; i < n; i++) { if (fabs(val[i]) < SX_TINY) { val[i] = 0; continue; } ix[c++] = i; }
    v->cnt = c;
}
void sxv_copy(SxVec *dst, const SxVec *src) {
    sxv_clear(dst);
    if (src->cnt >= 0) { for (int k = 0; k < src->cnt; k++) { const int i = src->idx[k]; dst->val[i] = src->val[i]; dst->idx[k] = i; } dst->cnt = src->cnt; }
    else { memcpy(dst->val, src->val, sizeof(double) * (size_t)src->n); sxv_reindex(dst); }
}
/* drop tiny entries from the list of a vector solved in hypersparse mode */
static void sxv_prune(SxVec *v) {
    int c = 0;
    for (int k = 0; k < v->cnt; k++) {
        const int i = v->idx[k];
        if (fabs(v->val[i]) < SX_TINY) { v->val[i] = 0; continue; }
        v->idx[c++] = i;
    }
    v->cnt = c;
}

/* ---------------------------------------------------------------- the active matrix of the elimination */
typedef struct {
    int m;
    int *cs, *cl, *cc, *ci; double *cv; int cend, ccap;     /* columns (basis positions): rows and values */
    int *rs, *rl, *rc, *ri; int rend, rcap;                 /* rows: positions (pattern); rl: the active entries' count */
    int *rp;                                                /* (24 October 2026) the pattern's length: entries of inactive columns stay until the row is compacted */
    int *chead, *cnext, *cprev, *rhead, *rnext, *rprev;     /* count lists */
    double *cmax;                                           /* largest |entry| of a column, < 0: stale */
    char *cact, *ract;
    double ptol;                                            /* the absolute pivot tolerance */
    double *cmax0, canc;                                    /* largest |entry| of each column in B; a pivot below canc times it is refused */
    int nrej; long nrejnz;                                  /* columns dropped by the search (no acceptable pivot) and their entries */
    int kmax;                                               /* no count exceeds it (the count lists above it stay empty) */
    int nccomp, nrcomp;                                     /* compactions of the column and row storage (new arrays) */
} Act;

static void lst_ins(int *head, int *next, int *prev, int k, int j) { next[j] = head[k]; prev[j] = -1; if (head[k] >= 0) prev[head[k]] = j; head[k] = j; }
static void lst_del(int *head, int *next, int *prev, int k, int j) {
    if (prev[j] >= 0) next[prev[j]] = next[j]; else head[k] = next[j];
    if (next[j] >= 0) prev[next[j]] = prev[j];
}
static void col_compact(Act *A, int extra) {
    long tot = 0;
    for (int j = 0; j < A->m; j++) if (A->cact[j]) tot += A->cl[j] + 2;
    long cap = 2 * tot + extra + 4 * (long)A->m + 16;
    if (cap < A->ccap) cap = A->ccap;
    int *ni = xmalloc(sizeof(int) * cap); double *nv = xmalloc(sizeof(double) * cap);
    int e = 0;
    for (int j = 0; j < A->m; j++) {
        if (!A->cact[j]) { A->cs[j] = 0; A->cc[j] = 0; continue; }
        memcpy(ni + e, A->ci + A->cs[j], sizeof(int) * A->cl[j]); memcpy(nv + e, A->cv + A->cs[j], sizeof(double) * A->cl[j]);
        A->cs[j] = e; A->cc[j] = A->cl[j] + 2; e += A->cc[j];
    }
    free(A->ci); free(A->cv); A->ci = ni; A->cv = nv; A->cend = e; A->ccap = (int)cap; A->nccomp++;
}
static void col_reserve(Act *A, int j, int need) {
    if (A->cl[j] + need <= A->cc[j]) return;
    const int nc = 2 * (A->cl[j] + need) + 4;
    if ((long)A->cend + nc > A->ccap) col_compact(A, nc);
    memmove(A->ci + A->cend, A->ci + A->cs[j], sizeof(int) * A->cl[j]);
    memmove(A->cv + A->cend, A->cv + A->cs[j], sizeof(double) * A->cl[j]);
    A->cs[j] = A->cend; A->cc[j] = nc; A->cend += nc;
}
static void row_compact(Act *A, int extra) {
    long tot = 0;
    for (int i = 0; i < A->m; i++) if (A->ract[i]) tot += A->rl[i] + 2;
    long cap = 2 * tot + extra + 4 * (long)A->m + 16;
    if (cap < A->rcap) cap = A->rcap;
    int *ni = xmalloc(sizeof(int) * cap);
    int e = 0;
    for (int i = 0; i < A->m; i++) {
        if (!A->ract[i]) { A->rs[i] = 0; A->rc[i] = 0; A->rp[i] = 0; continue; }
        int k = 0;
        for (int p = A->rs[i]; p < A->rs[i] + A->rp[i]; p++) if (A->cact[A->ri[p]]) ni[e + k++] = A->ri[p];
        A->rp[i] = k; A->rs[i] = e; A->rc[i] = k + 2; e += A->rc[i];
    }
    free(A->ri); A->ri = ni; A->rend = e; A->rcap = (int)cap; A->nrcomp++;
}
static void row_reserve(Act *A, int i, int need) {
    if (A->rp[i] + need <= A->rc[i]) return;
    if (A->rp[i] > A->rl[i]) {      /* the entries of inactive columns out first, in place */
        int k = A->rs[i];
        for (int p = A->rs[i]; p < A->rs[i] + A->rp[i]; p++) if (A->cact[A->ri[p]]) A->ri[k++] = A->ri[p];
        A->rp[i] = k - A->rs[i];
        if (A->rp[i] + need <= A->rc[i]) return;
    }
    const int nc = 2 * (A->rp[i] + need) + 4;
    if ((long)A->rend + nc > A->rcap) row_compact(A, nc);
    memmove(A->ri + A->rend, A->ri + A->rs[i], sizeof(int) * A->rp[i]);
    A->rs[i] = A->rend; A->rc[i] = nc; A->rend += nc;
}
static double col_max(Act *A, int j) {
    if (A->cmax[j] >= 0) return A->cmax[j];
    double mx = 0; const int s = A->cs[j], e = s + A->cl[j];
    for (int p = s; p < e; p++) { const double a = fabs(A->cv[p]); if (a > mx) mx = a; }
    return A->cmax[j] = mx;
}
static int col_find(const Act *A, int j, int i) {
    const int s = A->cs[j], e = s + A->cl[j];
    for (int p = s; p < e; p++) if (A->ci[p] == i) return p;
    return -1;
}
/* column j (made inactive by the caller, for good) out of row i: by its count only, the entry staying in the pattern until
 * the row is compacted, and the readers skipping inactive columns (24 October 2026: the search of the row for j was half of
 * square41's time, LP, rows of thousands of entries after its dense columns; BRISK_SXLULAZY=0: the search) */
/* (25 October 2026) BRISK_SXLU1PASS=0: the column updates in three passes as before (the same factors either way) */
static int lu_onepass(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXLU1PASS"); c = !(e && atoi(e) == 0); } return c; }
static int lu_lazy(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXLULAZY"); c = !(e && atoi(e) == 0); } return c; }
static void row_remove(Act *A, int i, int j) {
    if (lu_lazy()) { A->rl[i]--; return; }
    const int s = A->rs[i], e = s + A->rp[i];
    for (int p = s; p < e; p++) if (A->ri[p] == j) { A->ri[p] = A->ri[e - 1]; A->rl[i]--; A->rp[i]--; return; }
}

/* Markowitz search: returns 1 and (r, c) or 0 when no acceptable pivot remains */
static int find_pivot(Act *A, double u, int *pr, int *pc) {
    long best = -1; double bestv = 0; int br = -1, bc = -1, nsearch = 0;
    const int lim = 4;
    /* (23 October 2026) at most 8 rows of a count examined without an acceptable entry (all of them below the threshold of
     * their column), then the next count: on a singular or nearly singular basis such rows stay in the lists and were
     * searched again at every pivot, each entry with a search of its column (chromaticindex1024-7, LP: the crossover's
     * start basis, 12,451 columns replaced, 121 million column maxima, 6.7 s; with the cap 0.8 s, the problem 8.9 -> 2.7
     * s; chromaticindex512-7 2.2 -> 1.1 s; the dual simplex method on dfl001, pilot87, sing44 the same). A column always
     * has an acceptable entry (its largest), so the search still ends. BRISK_SXLUROWLIM: the cap (0: none) */
    static int rowlim = -1; if (rowlim < 0) { const char *e = getenv("BRISK_SXLUROWLIM"); rowlim = e ? atoi(e) : 8; }
    for (int k = 1; k <= A->kmax; k++) {
        for (int j = A->chead[k], jn; j >= 0; j = jn) {
            jn = A->cnext[j];
            const double cm = col_max(A, j);
            const int s = A->cs[j], e = s + A->cl[j];
            if (cm < A->ptol || cm < A->canc * A->cmax0[j]) {
                /* (22 October 2026) no entry can ever be its pivot (eliminations only shrink what is left of it): out of
                 * the active matrix at once; a logical takes its position at the end. Left in, it was searched again at
                 * every pivot: qap15's crossover start (1,781 such columns of 6,330) took 0.9 s to factorize */
                lst_del(A->chead, A->cnext, A->cprev, A->cl[j], j);
                for (int p = s; p < e; p++) {
                    const int i = A->ci[p];
                    lst_del(A->rhead, A->rnext, A->rprev, A->rl[i], i);
                    row_remove(A, i, j);
                    lst_ins(A->rhead, A->rnext, A->rprev, A->rl[i], i);
                }
                A->nrej++; A->nrejnz += A->cl[j];
                A->cact[j] = 0; A->cl[j] = 0;
                continue;
            }
            for (int p = s; p < e; p++) {
                const double a = fabs(A->cv[p]);
                if (a < u * cm || a < A->ptol || a < A->canc * A->cmax0[j]) continue;
                const long mer = (long)(k - 1) * (A->rl[A->ci[p]] - 1);
                if (best < 0 || mer < best || (mer == best && a > bestv)) { best = mer; bestv = a; br = A->ci[p]; bc = j; }
            }
            if (best >= 0 && (best <= (long)(k - 1) * (k - 1) || ++nsearch >= lim)) goto done;
        }
        int nfail = 0;
        for (int i = A->rhead[k]; i >= 0; i = A->rnext[i]) {
            const int s = A->rs[i], e = s + A->rp[i];
            int ok = 0;
            for (int p = s; p < e; p++) {
                const int j = A->ri[p];
                if (!A->cact[j]) continue;
                const int q = col_find(A, j, i);
                const double a = fabs(A->cv[q]);
                if (a < u * col_max(A, j) || a < A->ptol || a < A->canc * A->cmax0[j]) continue;
                ok = 1;
                const long mer = (long)(k - 1) * (A->cl[j] - 1);
                if (best < 0 || mer < best || (mer == best && a > bestv)) { best = mer; bestv = a; br = i; bc = j; }
            }
            if (!ok && rowlim > 0 && ++nfail >= rowlim) break;
            if (best >= 0 && (best <= (long)k * (k - 1) || ++nsearch >= lim)) goto done;
        }
        if (best >= 0 && best <= (long)k * k) goto done;
    }
done:
    if (best < 0) return 0;
    *pr = br; *pc = bc; return 1;
}

/* the U entries recorded during the elimination: (row r pivoted, position j, value) */
typedef struct { int *r, *j; double *v; int n, cap; } Trip;
static void __attribute__((noinline)) trip_grow(Trip *T, long need) {
    long cap = 2 * (long)T->cap + 1024; if (cap < need) cap = need;
    T->cap = (int)cap; T->r = xrealloc(T->r, sizeof(int) * T->cap); T->j = xrealloc(T->j, sizeof(int) * T->cap); T->v = xrealloc(T->v, sizeof(double) * T->cap);
}
static inline void trip_add(Trip *T, int r, int j, double v) {
    if (T->n == T->cap) trip_grow(T, 0);
    T->r[T->n] = r; T->j[T->n] = j; T->v[T->n] = v; T->n++;
}
static void l_add(SxLU *F, int i, double v) {   /* to the column being built: lstart[nl] its start, lstart[nl + 1] its end */
    if (F->lstart[F->nl + 1] + 1 > F->lcap) { F->lcap = 2 * F->lcap + 1024; F->lidx = xrealloc(F->lidx, sizeof(int) * F->lcap); F->lval = xrealloc(F->lval, sizeof(double) * F->lcap); }
    const int p = F->lstart[F->nl + 1]++;
    F->lidx[p] = i; F->lval[p] = v;
}

/* (22 October 2026) a copy of a factorization with its updates (lpmodel.c: the state before a trial of bound changes,
 * restored without a new factorization); dst's arrays are freed first */
struct SxFws; static void fws_free(struct SxFws *W);
static void *dup_arr(const void *p, size_t bytes) { if (!p) return NULL; void *q = xmalloc(bytes); memcpy(q, p, bytes); return q; }
void sxlu_copy(SxLU *dst, const SxLU *src) {
    sxlu_free(dst);
    *dst = *src;
    dst->fws = NULL;      /* (the factorization's work arrays: its own, made at its first factorization) */
    const size_t m = (size_t)src->m, I = sizeof(int), D = sizeof(double);
    dst->stack = dup_arr(src->stack, I * (2 * m + 2)); dst->wmark = dup_arr(src->wmark, I * (m + 1)); dst->wlist = dup_arr(src->wlist, I * (m + 1));
    dst->wval = dup_arr(src->wval, D * (m + 1)); dst->dfs_pos = dup_arr(src->dfs_pos, I * (m + 1)); dst->dmark = dup_arr(src->dmark, m + 1);
    dst->lpos = dup_arr(src->lpos, I * (m + 1)); dst->lcs_row = dup_arr(src->lcs_row, I * (m + 1)); dst->lcl_row = dup_arr(src->lcl_row, I * (m + 1));
    dst->lrstart = dup_arr(src->lrstart, I * (m + 2)); dst->lrlen = dup_arr(src->lrlen, I * (m + 1)); dst->lrorder = dup_arr(src->lrorder, I * (m + 1));
    dst->lrow = dup_arr(src->lrow, I * (m + 1)); dst->lstart = dup_arr(src->lstart, I * (m + 2));
    dst->uordpos = dup_arr(src->uordpos, I * (m + 1)); dst->udiag = dup_arr(src->udiag, D * (m + 1)); dst->urecip = dup_arr(src->urecip, D * (m + 1)); dst->uzpos = dup_arr(src->uzpos, I * (m + 1));
    dst->ucs = dup_arr(src->ucs, I * (m + 1)); dst->ucl = dup_arr(src->ucl, I * (m + 1)); dst->ucc = dup_arr(src->ucc, I * (m + 1));
    dst->urs = dup_arr(src->urs, I * (m + 1)); dst->url = dup_arr(src->url, I * (m + 1)); dst->urc = dup_arr(src->urc, I * (m + 1));
    dst->lidx = dup_arr(src->lidx, I * (size_t)src->lcap); dst->lval = dup_arr(src->lval, D * (size_t)src->lcap);
    const size_t nlz = src->lstart ? (size_t)src->lstart[src->nl] + 1 : 0;
    dst->lridx = dup_arr(src->lridx, I * nlz); dst->lrval = dup_arr(src->lrval, D * nlz);
    dst->uord = dup_arr(src->uord, I * (size_t)src->ordcap); dst->uz = dup_arr(src->uz, I * (size_t)src->uzcap);
    dst->uci = dup_arr(src->uci, I * (size_t)src->ucap); dst->ucv = dup_arr(src->ucv, D * (size_t)src->ucap);
    dst->uri = dup_arr(src->uri, I * (size_t)src->urcap); dst->urv = dup_arr(src->urv, D * (size_t)src->urcap);
    dst->rpiv = dup_arr(src->rpiv, I * (size_t)src->rcapn); dst->rstart = dup_arr(src->rstart, I * ((size_t)src->rcapn + 1));
    dst->ridx = dup_arr(src->ridx, I * (size_t)src->rcap); dst->rval = dup_arr(src->rval, D * (size_t)src->rcap);
    dst->rthead = dup_arr(src->rthead, I * (m + 1)); dst->rtcnt = dup_arr(src->rtcnt, I * (m + 1));
    dst->rnext = dup_arr(src->rnext, I * (size_t)src->rcap); dst->reta = dup_arr(src->reta, I * (size_t)src->rcap);
    /* (1 November 2026) the sweep's bitmaps (kept zero between solves): fresh ones, not the source's (shared, they were freed
     * twice: lpmodel.c's trials, a saved factorization restored and freed; the re-solve interface's test aborted) */
    dst->ubm = src->ubm ? xcalloc(2 * m / 64 + 2, sizeof(unsigned long long)) : NULL;
    dst->ubs = src->ubs ? xcalloc(2 * m / 4096 + 2, sizeof(unsigned long long)) : NULL;
}
void sxlu_free(SxLU *F) {
    free(F->lrow); free(F->lstart); free(F->lidx); free(F->lval); free(F->lrstart); free(F->lridx); free(F->lrval); free(F->lpos);
    free(F->uord); free(F->uordpos); free(F->uz); free(F->uzpos); free(F->udiag); free(F->urecip); free(F->ucs); free(F->ucl); free(F->ucc); free(F->uci); free(F->ucv);
    free(F->urs); free(F->url); free(F->urc); free(F->uri); free(F->urv);
    free(F->rpiv); free(F->rstart); free(F->ridx); free(F->rval); free(F->rthead); free(F->rtcnt); free(F->rnext); free(F->reta);
    free(F->stack); free(F->wmark); free(F->wlist); free(F->wval); free(F->dfs_pos); free(F->dmark); free(F->ubm); free(F->ubs);
    free(F->lcs_row); free(F->lcl_row); free(F->lrorder); free(F->lrlen); fws_free(F->fws);
    memset(F, 0, sizeof *F);
}

/* the dense kernel: once the active matrix is dense, it is copied into a dense array (row-major) and factorized by
 * LU with partial pivoting, column by column; the pivots, the L columns and the U entries are recorded as in the
 * sparse elimination. A column without an entry above the absolute pivot tolerance is left (its position gets a
 * logical at the end, as in the sparse case). */
/* (25 October 2026) the dense kernel by blocks: LU with partial pivoting (the largest entry of the column, as below) on a
 * column-major copy, panels of SX_DENSE_NB columns factored column by column, then the block row by a triangular solve
 * and the trailing block by one product. The pivots, the L columns (the multipliers) and the U rows are recorded as by
 * the column kernel. Square active matrices only, and every pivot must pass the column kernel's tests (the absolute
 * tolerance, the cancellation test); otherwise 0 is returned and the column kernel factorizes, skipping the columns
 * without a pivot as before. The column kernel's rank-one updates of a row-major array cost n^3/3 multiply-adds without
 * blocking: 31 ms for qap15's active matrix of 638 (LP, the crossover), which kept the switch to it late. */
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define SXL_CAT2(a, b) a##b
#define SXL_CAT(a, b) SXL_CAT2(a, b)
#define SXL_BL(name) SXL_CAT(BLAS_PREFIX, name)
void SXL_BL(dgemm_)(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *,
                    const double *, const int *, const double *, double *, const int *);
void SXL_BL(dtrsm_)(const char *, const char *, const char *, const char *, const int *, const int *, const double *, const double *,
                    const int *, double *, const int *);
#define SX_DENSE_NB 128      /* (10 November 2026) the panel's columns: 128 from 2,048 active (a rank-48 update of a large trailing
                                matrix is bound by memory), 48 below (a block narrower than the panel is factored by the panel's
                                unblocked loop alone: supportcase34, LP, 1.11 -> 1.27 s with 128 throughout) */
/* BRISK_SXDENSEBLK=0: the column kernel only, and the switch at SX_DENSE_FRAC */
/* (10 November 2026) the dense kernel's parallel blocks: not inside another parallel region (the race's jobs run the simplex
 * method on one thread of their own), BRISK_SXDENSEPAR=0: off */
static int dense_par(void) {
    static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXDENSEPAR"); c = !(e && atoi(e) == 0); }
#ifdef _OPENMP
    return c && !omp_in_parallel() && omp_get_max_threads() > 1;
#else
    return 0;
#endif
}
/* (10 November 2026) the largest dense array of the blocked kernel: a twentieth of the physical memory in doubles (15 GB: 94
 * million entries, a matrix of 9,700), at least the 25 million of before. With the old fixed cap a dense active matrix
 * larger than 5,000 went on by the sparse elimination at 30-70 % density: the crossover's first factorization of the outside
 * LP benchmark's regression LP E0000_5 took 142.6 s, 120 s of it in the 642 sparse pivots from 5,642 active rows down to the
 * cap's 5,000. BRISK_SXDENSECAP=<entries>: the cap */
static double dense_cap(void) {
    static double c = -1;
    if (c < 0) {
        const char *e = getenv("BRISK_SXDENSECAP");
        if (e) c = atof(e);
        else { const double mem = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE); c = mem > 0 ? fmax(2.5e7, 0.05 * mem / 8.0) : 2.5e7; }
    }
    return c;
}
static int dense_blk_on(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXDENSEBLK"); c = !(e && atoi(e) == 0); } return c; }
static int dense_kernel_blocked(SxLU *F, Act *A, Trip *T, int *rowof, int *prow, double *pval, int *np) {
    const int m = A->m;
    int n = 0, nc = 0;
    for (int i = 0; i < m; i++) n += A->ract[i] != 0;
    for (int j = 0; j < m; j++) nc += A->cact[j] != 0;
    if (n != nc || n == 0) { if (getenv("SXDBG") && atoi(getenv("SXDBG")) > 1) printf("    [the blocked dense kernel declined: %d active rows, %d columns]\n", n, nc); return 0; }
    int *rid = xmalloc(sizeof(int) * (n + 1)), *cid = xmalloc(sizeof(int) * (n + 1)), *rmap = xmalloc(sizeof(int) * (m + 1)), *perm = xmalloc(sizeof(int) * (n + 1));
    int k = 0;
    for (int i = 0; i < m; i++) { rmap[i] = -1; if (A->ract[i]) { rmap[i] = k; rid[k++] = i; } }
    k = 0; for (int j = 0; j < m; j++) if (A->cact[j]) cid[k++] = j;
    struct timespec tdk0_; clock_gettime(CLOCK_MONOTONIC, &tdk0_);
    double *D = xcalloc((size_t)n * n, sizeof(double));
    #define DE(i, j) D[(size_t)(j) * n + (i)]
    for (int jj = 0; jj < n; jj++) { const int j = cid[jj]; for (int p = A->cs[j]; p < A->cs[j] + A->cl[j]; p++) DE(rmap[A->ci[p]], jj) = A->cv[p]; }
    for (int i = 0; i < n; i++) perm[i] = i;
    int ok = 1, nce = n;      /* (nce: the columns still active; a column without a pivot goes to the end, as the column kernel
                                * skips it: left active, its position a logical later) */
    const double mone = -1.0, one = 1.0;
    int piv_rows[SX_DENSE_NB];
    const int NB = n >= 2048 ? SX_DENSE_NB : 48;
    for (int k0 = 0; k0 < nce && ok; k0 += NB) {
        int kb = nce - k0 < NB ? nce - k0 : NB, k1 = k0 + kb;
        for (int j = k0; j < k1; j++) {
            int pr = -1; double best = 0;
            for (;;) {
                pr = -1; best = 0;
                for (int i = j; i < n; i++) { const double a = fabs(DE(i, j)); if (a > best) { best = a; pr = i; } }
                if (!(pr < 0 || best < A->ptol || best < A->canc * A->cmax0[cid[j]])) break;
                /* (10 November 2026) no pivot in this column (a singular active matrix: the crossover's start basis of the
                 * outside LP benchmark's regression LP E0000_5, 5,642 active, the column kernel took 18 s): it goes to the end
                 * and the last active column takes its place, after the panel's swaps and eliminations so far (the earlier
                 * panels' updates it has, from the trailing part); declined only when that column is inside the panel */
                const int last = nce - 1;
                if (last < k1) {
                    /* (the last active column inside the panel: it has the panel's swaps and eliminations already; the panel
                     * ends there) */
                    for (int i = 0; i < n; i++) { const double t = DE(i, j); DE(i, j) = DE(i, last); DE(i, last) = t; }
                    { const int t = cid[j]; cid[j] = cid[last]; cid[last] = t; }
                    nce--; k1 = nce; kb = k1 - k0;
                    if (j >= k1) break;
                    continue;
                }
                for (int i = 0; i < n; i++) { const double t = DE(i, j); DE(i, j) = DE(i, last); DE(i, last) = t; }
                { const int t = cid[j]; cid[j] = cid[last]; cid[last] = t; }
                nce--;
                double *col = &DE(0, j);
                for (int t = k0; t < j; t++) { const int a = t, b = piv_rows[t - k0]; if (a != b) { const double v = col[a]; col[a] = col[b]; col[b] = v; } }
                for (int t = k0; t < j; t++) { const double xt = col[t]; if (xt != 0) for (int i = t + 1; i < n; i++) col[i] -= DE(i, t) * xt; }
            }
            if (j >= k1) break;            /* (no active column left) */
            /* (10 November 2026) the swap within the panel's columns only; the other columns take the panel's swaps once
             * after it (below), column by column: a swap across all n columns of the column-major array was a strided pass
             * per pivot */
            if (pr != j) { for (int c = k0; c < k1; c++) { const double t = DE(j, c); DE(j, c) = DE(pr, c); DE(pr, c) = t; } const int t = perm[j]; perm[j] = perm[pr]; perm[pr] = t; }
            piv_rows[j - k0] = pr;
            const double rp = 1.0 / DE(j, j);
            for (int i = j + 1; i < n; i++) DE(i, j) *= rp;
            for (int c = j + 1; c < k1; c++) { const double ujc = DE(j, c); if (ujc != 0) for (int i = j + 1; i < n; i++) DE(i, c) -= DE(i, j) * ujc; }
        }
        if (!ok) break;
        {   /* the panel's swaps to the active columns before it and after it, in the panel's order */
            const int kbn = k1 - k0;
            #pragma omp parallel for schedule(static) if (n >= 1024 && dense_par())
            for (int c = 0; c < nce; c++) {
                if (c >= k0 && c < k1) continue;
                double *col = &DE(0, c);
                for (int t = 0; t < kbn; t++) { const int a = k0 + t, b = piv_rows[t]; if (a != b) { const double v = col[a]; col[a] = col[b]; col[b] = v; } }
            }
        }
        if (k1 >= nce) break;
        const int nr2 = n - k1, wc = nce - k1;
        /* (10 November 2026) the block row's solve and the trailing update by chunks of columns on the threads: each column's
         * arithmetic is that of one call (the same operations in the same order), so the result does not depend on the thread
         * count. A dense active matrix of 5,000 on one thread took 7.6 s (the outside LP benchmark's regression LP E0000_5,
         * its crossover's first factorization) */
        const int CH = 256, nch = (wc + CH - 1) / CH;
        #pragma omp parallel for schedule(static) if (wc >= 1024 && dense_par())
        for (int ch = 0; ch < nch; ch++) {
            const int c0 = k1 + ch * CH, w = nce - c0 < CH ? nce - c0 : CH;
            SXL_BL(dtrsm_)("L", "L", "N", "U", &kb, &w, &one, &DE(k0, k0), &n, &DE(k0, c0), &n);
            SXL_BL(dgemm_)("N", "N", &nr2, &w, &kb, &mone, &DE(k1, k0), &n, &DE(k0, c0), &n, &one, &DE(k1, c0), &n);
        }
    }
    if (!ok) { free(rid); free(cid); free(rmap); free(perm); free(D); return 0; }
    struct timespec tdk1_; clock_gettime(CLOCK_MONOTONIC, &tdk1_);
    for (int kk = 0; kk < nce; kk++) {      /* (the rows and columns left without a pivot stay active, as the column kernel leaves them) */
        const int r = rid[perm[kk]], c = cid[kk];
        const double piv = DE(kk, kk);
        int nl = 0;
        for (int i = kk + 1; i < n; i++) if (DE(i, kk) != 0) nl++;
        if (nl) {
            F->lrow[F->nl] = r; F->lstart[F->nl + 1] = F->lstart[F->nl];
            for (int i = kk + 1; i < n; i++) if (DE(i, kk) != 0) l_add(F, rid[perm[i]], DE(i, kk));
            F->nl++;
        }
        for (int jj = kk + 1; jj < nce; jj++) if (DE(kk, jj) != 0) trip_add(T, r, cid[jj], DE(kk, jj));
        F->factor_work += (double)(n - kk - 1) * (n - kk);
        A->ract[r] = 0; A->cact[c] = 0;
        rowof[c] = r; prow[*np] = r; pval[*np] = piv; (*np)++;
    }
    #undef DE
    if (getenv("SXDBG") && atoi(getenv("SXDBG")) > 1) { struct timespec t2_; clock_gettime(CLOCK_MONOTONIC, &t2_);
        printf("    [the blocked dense kernel, wall: the copy and the factorization %.3f s, the factors recorded %.3f s]\n",
               (tdk1_.tv_sec - tdk0_.tv_sec) + 1e-9 * (tdk1_.tv_nsec - tdk0_.tv_nsec), (t2_.tv_sec - tdk1_.tv_sec) + 1e-9 * (t2_.tv_nsec - tdk1_.tv_nsec)); }
    free(rid); free(cid); free(rmap); free(perm); free(D);
    return 1;
}
static void dense_kernel(SxLU *F, Act *A, Trip *T, double u, int *rowof, int *prow, double *pval, int *np) {
    (void)u;
    if (dense_blk_on() && dense_kernel_blocked(F, A, T, rowof, prow, pval, np)) return;
    const int m = A->m;
    int nr = 0, nc = 0;
    int *rid = xmalloc(sizeof(int) * (m + 1)), *cid = xmalloc(sizeof(int) * (m + 1)), *rmap = xmalloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) { rmap[i] = -1; if (A->ract[i]) { rmap[i] = nr; rid[nr++] = i; } }
    for (int j = 0; j < m; j++) if (A->cact[j]) cid[nc++] = j;
    double *D = xcalloc((size_t)nr * nc, sizeof(double));
    for (int jj = 0; jj < nc; jj++) { const int j = cid[jj]; for (int p = A->cs[j]; p < A->cs[j] + A->cl[j]; p++) D[(size_t)rmap[A->ci[p]] * nc + jj] = A->cv[p]; }
    int *left = xmalloc(sizeof(int) * (nr + 1)); int nleft = nr;
    for (int k = 0; k < nr; k++) left[k] = k;
    int *lr = xmalloc(sizeof(int) * (nr + 1)); double *lv = xmalloc(sizeof(double) * (nr + 1));
    for (int jj = 0; jj < nc && nleft > 0; jj++) {
        int kb = -1; double best = 0;
        for (int k = 0; k < nleft; k++) { const double a = fabs(D[(size_t)left[k] * nc + jj]); if (a > best) { best = a; kb = k; } }
        if (kb < 0 || best < A->ptol || best < A->canc * A->cmax0[cid[jj]]) continue;
        const int ii = left[kb]; left[kb] = left[--nleft];
        const double *prw = D + (size_t)ii * nc;
        const double piv = prw[jj];
        /* L column */
        int nl = 0;
        for (int k = 0; k < nleft; k++) { const int i = left[k]; const double a = D[(size_t)i * nc + jj]; if (a != 0) { lr[nl] = i; lv[nl] = a / piv; nl++; } }
        if (nl) {
            F->lrow[F->nl] = rid[ii]; F->lstart[F->nl + 1] = F->lstart[F->nl];
            for (int k = 0; k < nl; k++) l_add(F, rid[lr[k]], lv[k]);
            F->nl++;
        }
        /* U row: the later columns */
        for (int kk = jj + 1; kk < nc; kk++) if (prw[kk] != 0) trip_add(T, rid[ii], cid[kk], prw[kk]);
        /* the update of the rows left */
        for (int k = 0; k < nl; k++) {
            double *row = D + (size_t)lr[k] * nc; const double l = lv[k];
            for (int kk = jj + 1; kk < nc; kk++) row[kk] -= l * prw[kk];
        }
        F->factor_work += (double)nl * (nc - jj);
        const int r = rid[ii], c = cid[jj];
        A->ract[r] = 0; A->cact[c] = 0;
        rowof[c] = r; prow[*np] = r; pval[*np] = piv; (*np)++;
    }
    free(rid); free(cid); free(rmap); free(D); free(left); free(lr); free(lv);
}

/* the columns set aside at the start of the elimination (positions defj) back into the active matrix: each column is
 * taken from A without its entries in the rows of logicals and singletons (in U already), updated by the L columns of the
 * elimination in their order (x_i -= l_ik x_r for the pivot row r of L column k); its entry in an eliminated row r is
 * then its U entry, those in the rows still active its active column */
static void defer_back(SxLU *F, Act *A, Trip *T, int m, const int *Ap, const int *Ai, const double *Ax, const int *var,
                       const char *rdone, const int *defj, int ndef, int *nca, long *nzact) {
    double *x = xcalloc((size_t)m + 1, sizeof(double));
    for (int t = 0; t < ndef; t++) {
        const int j = defj[t], v = var[j];
        for (int p = Ap[v]; p < Ap[v + 1]; p++) if (Ax[p] != 0 && !rdone[Ai[p]]) x[Ai[p]] = Ax[p];
        for (int k = 0; k < F->nl; k++) {
            const double xr = x[F->lrow[k]];
            if (xr == 0) continue;
            for (int q = F->lstart[k]; q < F->lstart[k + 1]; q++) x[F->lidx[q]] -= F->lval[q] * xr;
        }
        F->factor_work += m + F->lstart[F->nl];
        int cnt = 0;
        for (int i = 0; i < m; i++) if (x[i] != 0 && A->ract[i]) cnt++;
        A->cl[j] = 0; col_reserve(A, j, cnt);
        for (int i = 0; i < m; i++) {
            if (x[i] == 0) continue;
            if (A->ract[i]) {
                const int q = A->cs[j] + A->cl[j]++;
                A->ci[q] = i; A->cv[q] = x[i];
                lst_del(A->rhead, A->rnext, A->rprev, A->rl[i], i);
                row_reserve(A, i, 1); A->ri[A->rs[i] + A->rp[i]++] = j; A->rl[i]++;
                lst_ins(A->rhead, A->rnext, A->rprev, A->rl[i], i);
            } else if (!rdone[i]) trip_add(T, i, j, x[i]);
            x[i] = 0;
        }
        A->cact[j] = 1; A->cmax[j] = -1;
        lst_ins(A->chead, A->cnext, A->cprev, A->cl[j], j);
        (*nca)++; *nzact += A->cl[j];
    }
    free(x);
}
/* (5 November 2026) the factorization's work arrays, kept between factorizations of the same dimension (SxLU.fws): the
 * arrays of the rows and positions, the active matrix's lists and storage, the U entries recorded. They were allocated,
 * zeroed and freed at every factorization, and most passes over them ran over all m rows and positions; on savsched1's
 * dual (LP, 247,801 rows) a factorization of the logicals' basis took 6-8 ms, more than half of it in allocation and
 * passes over rows that the logicals had already pivoted. Now the passes after the logicals run over the structural
 * positions and the rows left, and the passes over all rows that remain are plain fills. The same pivots, factors and
 * order of entries as before */
struct SxFws {
    int m;
    int *rowof, *sj, *apos, *arow, *cnt, *rptr, *rcnt, *rfill, *stk, *upos;
    double *pval;
    char *rdone, *umk;
    unsigned long long *bits;
    int *rj; long rjcap;
    int *bp, *bi, *apt; double *bx; long bcap;
    Trip T;
    /* the active matrix's arrays (Act), kept with their storage */
    int *cs, *cl, *cc, *rs, *rl, *rc, *rp, *chead, *cnext, *cprev, *rhead, *rnext, *rprev;
    double *cmax, *cmax0;
    char *cact, *ract;
    int *ci, *ri; double *cv; long cicap, ricap;
};
static void fws_free(struct SxFws *W) {
    if (!W) return;
    free(W->rcnt); free(W->rfill); free(W->rowof); free(W->sj); free(W->apos); free(W->arow); free(W->cnt); free(W->rptr); free(W->stk); free(W->upos);
    free(W->pval); free(W->rdone); free(W->umk); free(W->bits); free(W->rj); free(W->bp); free(W->bi); free(W->bx); free(W->apt); free(W->T.r); free(W->T.j); free(W->T.v);
    free(W->cs); free(W->cl); free(W->cc); free(W->rs); free(W->rl); free(W->rc); free(W->rp); free(W->chead); free(W->cnext); free(W->cprev);
    free(W->rhead); free(W->rnext); free(W->rprev); free(W->cmax); free(W->cmax0); free(W->cact); free(W->ract); free(W->ci); free(W->ri); free(W->cv);
    free(W);
}
static struct SxFws *fws_get(SxLU *F, int m) {
    if (F->fws && F->fws->m == m) return F->fws;
    fws_free(F->fws);
    struct SxFws *W = xcalloc(1, sizeof *W);
    const size_t M = (size_t)m + 2;
    W->m = m;
    W->rcnt = xmalloc(sizeof(int) * M); W->rfill = xmalloc(sizeof(int) * M); W->rowof = xmalloc(sizeof(int) * M); W->sj = xmalloc(sizeof(int) * M);
    W->apos = xmalloc(sizeof(int) * M); W->arow = xmalloc(sizeof(int) * M); W->cnt = xmalloc(sizeof(int) * M); W->rptr = xmalloc(sizeof(int) * M);
    W->stk = xmalloc(sizeof(int) * M); W->upos = xmalloc(sizeof(int) * M); W->bp = xmalloc(sizeof(int) * M); W->apt = xmalloc(sizeof(int) * M);
    W->pval = xmalloc(sizeof(double) * M); W->rdone = xcalloc(M, 1); W->umk = xcalloc(M, 1);
    W->bits = xcalloc(M / 64 + 2, sizeof(unsigned long long));
    W->cs = xmalloc(sizeof(int) * M); W->cl = xmalloc(sizeof(int) * M); W->cc = xmalloc(sizeof(int) * M);
    W->rs = xmalloc(sizeof(int) * M); W->rl = xmalloc(sizeof(int) * M); W->rc = xmalloc(sizeof(int) * M); W->rp = xmalloc(sizeof(int) * M);
    W->chead = xmalloc(sizeof(int) * M); W->cnext = xmalloc(sizeof(int) * M); W->cprev = xmalloc(sizeof(int) * M);
    W->rhead = xmalloc(sizeof(int) * M); W->rnext = xmalloc(sizeof(int) * M); W->rprev = xmalloc(sizeof(int) * M);
    W->cmax = xmalloc(sizeof(double) * M); W->cmax0 = xmalloc(sizeof(double) * M); W->cact = xcalloc(M, 1); W->ract = xcalloc(M, 1);
    F->fws = W;
    return W;
}
int sxlu_factor(SxLU *F, int m, int n, const int *Ap, const int *Ai, const double *Ax, int *var, int *out) {
    if (F->m != m) { const double pt = F->pivthr, pc = F->pivcanc, dl = F->deadline; const volatile int *sf = F->stopf; sxlu_free(F); F->m = m; F->pivthr = pt; F->pivcanc = pc; F->deadline = dl; F->stopf = sf; }   /* (the settings kept, the stop and the deadline too: ISSUES 884) */
    const double u = F->pivthr > 0 ? F->pivthr : 0.1, ptol = SX_PIVTOL;
    /* work arrays kept between factorizations */
    if (!F->stack) {
        F->stack = xmalloc(sizeof(int) * (2 * (size_t)m + 2)); F->wmark = xcalloc((size_t)m + 1, sizeof(int));
        F->wlist = xmalloc(sizeof(int) * ((size_t)m + 1)); F->wval = xcalloc((size_t)m + 1, sizeof(double)); F->dfs_pos = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->dmark = xcalloc((size_t)m + 1, 1);
        F->ubm = xcalloc(2 * (size_t)m / 64 + 2, sizeof(unsigned long long)); F->ubs = xcalloc(2 * (size_t)m / 4096 + 2, sizeof(unsigned long long));
        F->lpos = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lcs_row = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lcl_row = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->lrstart = xmalloc(sizeof(int) * ((size_t)m + 2)); F->lrlen = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lrorder = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->lrow = xmalloc(sizeof(int) * ((size_t)m + 1)); F->lstart = xmalloc(sizeof(int) * ((size_t)m + 2));
        F->uordpos = xmalloc(sizeof(int) * ((size_t)m + 1)); F->udiag = xmalloc(sizeof(double) * ((size_t)m + 1)); F->urecip = xmalloc(sizeof(double) * ((size_t)m + 1));
        F->ucs = xmalloc(sizeof(int) * ((size_t)m + 1)); F->ucl = xmalloc(sizeof(int) * ((size_t)m + 1)); F->ucc = xmalloc(sizeof(int) * ((size_t)m + 1));
        F->urs = xmalloc(sizeof(int) * ((size_t)m + 1)); F->url = xmalloc(sizeof(int) * ((size_t)m + 1)); F->urc = xmalloc(sizeof(int) * ((size_t)m + 1));
    }
    struct SxFws *W = fws_get(F, m);
    /* the pivot rows in pivot order (prow) are stored as U's order at once */
    if (F->ordcap < 2 * m + 16) { F->ordcap = 2 * m + 16; F->uord = xrealloc(F->uord, sizeof(int) * F->ordcap); }
    int *prow = F->uord, *rowof = W->rowof; double *pval = W->pval;
    char *rdone = W->rdone;
    int *sj = W->sj, *apos = W->apos, *arow = W->arow;   /* the structural positions, the positions left to the elimination, its rows */
    int np = 0, ns = 0, nap = 0, nar = 0;
    F->nl = 0; F->lstart[0] = 0; F->nr = 0; F->nupd = 0; F->rank_deficient = 0; F->factor_work = 0;
    Trip T = W->T; T.n = 0;
    memset(rowof, 0xff, sizeof(int) * (size_t)m);       /* pivot row of each position (-1: none yet) */
    memset(rdone, 0, (size_t)m);
    /* 1. logicals first (bulk): their rows leave the active matrix; entries of other columns there go to U. A second
     * logical of the same row (singular) is left to the elimination as an empty column (apos, merged below) */
    int *dupl = W->upos, ndup = 0;     /* (upos is free until the elimination) */
    for (int j = 0; j < m; j++) {
        if (var[j] < n) { sj[ns++] = j; continue; }
        const int i = var[j] - n;
        if (rdone[i]) { dupl[ndup++] = j; continue; }
        rdone[i] = 1; rowof[j] = i; prow[np] = i; np++;     /* (its pivot 1: pval is not set for the first nlog) */
    }
    const int nlog = np;
    /* 1b. column singletons (structural columns with one entry in the rows left), in bulk like the logicals: no
     * elimination, their rows leave the active matrix (entries of other columns there go to U, the column's own entries
     * in rows pivoted before it too), and the counts of the columns with an entry in such a row drop, which can make
     * more singletons. Counts and a row-wise copy of the structural entries in the rows left make it linear in the
     * basis' entries. */
    /* (5 November 2026) the basis' structural columns first copied together, without explicit zeros (bp, bi, bx: column t
     * is position sj[t]): the passes below read them in one block instead of the columns of A scattered over its storage */
    long nzs = 0;
    int *bp = W->bp, *bi; double *bx;
    {
        for (int t = 0; t < ns; t++) { const int v = var[sj[t]]; nzs += Ap[v + 1] - Ap[v]; }
        if (W->bcap < nzs + 1) { W->bcap = nzs + 1 + nzs / 4; free(W->bi); free(W->bx); W->bi = xmalloc(sizeof(int) * W->bcap); W->bx = xmalloc(sizeof(double) * W->bcap); }
        bi = W->bi; bx = W->bx;
    }
    {
        int *cnt = W->cnt, *stk = W->stk; int nst = 0;     /* cnt by t */
        /* the rows left after the logicals, in increasing order (a word of eight rows skipped at once when all are done);
         * their counts, starts and fill pointers in the row-wise copy (rcnt, rbeg, fill: set for these rows only) */
        {
            const unsigned long long all = 0x0101010101010101ULL;
            int i = 0;
            for (; i + 8 <= m; i += 8) {
                unsigned long long w; memcpy(&w, rdone + i, 8);
                if (w == all) continue;
                for (int k = i; k < i + 8; k++) if (!rdone[k]) arow[nar++] = k;
            }
            for (; i < m; i++) if (!rdone[i]) arow[nar++] = i;
        }
        int *rcnt = W->rcnt, *rbeg = W->rptr, *fill = W->rfill;
        for (int t = 0; t < nar; t++) rcnt[arow[t]] = 0;
        long nzb = 0; int q = 0;
        bp[0] = 0;
        for (int t = 0; t < ns; t++) {
            const int v = var[sj[t]]; int c = 0;
            for (int p = Ap[v]; p < Ap[v + 1]; p++) {
                if (Ax[p] == 0) continue;
                const int i = Ai[p];
                bi[q] = i; bx[q] = Ax[p]; q++;
                if (!rdone[i]) { rcnt[i]++; c++; }
            }
            bp[t + 1] = q; cnt[t] = c; nzb += c;
        }
        { int run = 0; for (int t = 0; t < nar; t++) { const int i = arow[t]; rbeg[i] = run; fill[i] = run; run += rcnt[i]; } }
        if (W->rjcap < nzb + 1) { W->rjcap = nzb + 1 + nzb / 4; free(W->rj); W->rj = xmalloc(sizeof(int) * W->rjcap); }
        int *rj = W->rj;      /* the columns (t) of each row left */
        for (int t = 0; t < ns; t++) {
            for (int p = bp[t]; p < bp[t + 1]; p++) if (!rdone[bi[p]]) rj[fill[bi[p]]++] = t;
            if (cnt[t] == 1) stk[nst++] = t;
        }
        while (nst > 0) {
            const int t = stk[--nst], j = sj[t];
            if (rowof[j] >= 0 || cnt[t] != 1) continue;
            int i = -1; double a = 0;
            for (int p = bp[t]; p < bp[t + 1]; p++) if (!rdone[bi[p]]) { i = bi[p]; a = bx[p]; break; }
            if (i < 0 || fabs(a) < ptol) continue;    /* (left to the elimination) */
            if (F->pivcanc > 0) { double cm = 0; for (int p = bp[t]; p < bp[t + 1]; p++) cm = fmax(cm, fabs(bx[p])); if (fabs(a) < F->pivcanc * cm) continue; }
            if (T.n + (bp[t + 1] - bp[t]) > T.cap) trip_grow(&T, T.n + (bp[t + 1] - bp[t]));
            for (int p = bp[t]; p < bp[t + 1]; p++) if (bi[p] != i) { T.r[T.n] = bi[p]; T.j[T.n] = j; T.v[T.n] = bx[p]; T.n++; }
            rdone[i] = 1; rowof[j] = i; prow[np] = i; pval[np] = a; np++;
            for (int p = rbeg[i]; p < rbeg[i] + rcnt[i]; p++) { const int k = rj[p]; if (rowof[sj[k]] < 0 && --cnt[k] == 1) stk[nst++] = k; }
        }
        /* the positions left, in increasing order: the structural ones not pivoted (apt: their t) and the duplicate logicals
         * (apt -1) */
        int a = 0, b = 0;
        while (a < ns || b < ndup) {
            if (b >= ndup || (a < ns && sj[a] < dupl[b])) { const int j = sj[a++]; if (rowof[j] >= 0) continue; W->apt[nap] = a - 1; apos[nap++] = j; }
            else { W->apt[nap] = -1; apos[nap++] = dupl[b++]; }
        }
        /* the rows left after the singletons */
        int k = 0;
        for (int t = 0; t < nar; t++) if (!rdone[arow[t]]) arow[k++] = arow[t];
        nar = k;
    }
    /* 2. the active matrix: structural columns, entries in rows not yet pivoted */
    Act A; memset(&A, 0, sizeof A); A.m = m; A.ptol = ptol; A.canc = F->pivcanc; A.cmax0 = W->cmax0;
    A.ccap = (int)(2 * nzs + 4 * (long)m + 16);
    if (W->cicap < A.ccap) { free(W->ci); free(W->cv); W->cicap = A.ccap; W->ci = xmalloc(sizeof(int) * W->cicap); W->cv = xmalloc(sizeof(double) * W->cicap); }
    A.ci = W->ci; A.cv = W->cv;
    A.rcap = A.ccap;
    if (W->ricap < A.rcap) { free(W->ri); W->ricap = A.rcap; W->ri = xmalloc(sizeof(int) * W->ricap); }
    A.ri = W->ri;
    A.cs = W->cs; A.cl = W->cl; A.cc = W->cc; A.rs = W->rs; A.rl = W->rl; A.rc = W->rc; A.rp = W->rp;
    A.chead = W->chead; A.cnext = W->cnext; A.cprev = W->cprev; A.rhead = W->rhead; A.rnext = W->rnext; A.rprev = W->rprev;
    A.cmax = W->cmax; A.cact = W->cact; A.ract = W->ract;
    memset(A.cact, 0, (size_t)m); memset(A.ract, 0, (size_t)m);
    for (int t = 0; t < nar; t++) A.rl[arow[t]] = 0;
    int e = 0;
    for (int t = 0; t < nap; t++) {
        const int j = apos[t];
        A.cs[j] = e; A.cmax[j] = -1; A.cmax0[j] = 0; A.cl[j] = 0;
        const int bt = W->apt[t];
        /* (28 October 2026) the largest entries of the columns left to the elimination only (the searches read no other):
         * the pass over every column of the basis was 3.5 % of the dual simplex method's time on the tester's largest
         * coupled network (LP, 575,040 rows), whose columns are mostly pivoted in bulk above */
        if (bt < 0) { /* a duplicate logical: an empty active column */ A.cact[j] = 1; A.cc[j] = 2; e += 2; continue; }
        for (int p = bp[bt]; p < bp[bt + 1]; p++) A.cmax0[j] = fmax(A.cmax0[j], fabs(bx[p]));
        A.cact[j] = 1;
        if (T.n + (bp[bt + 1] - bp[bt]) > T.cap) trip_grow(&T, T.n + (bp[bt + 1] - bp[bt]));
        for (int p = bp[bt]; p < bp[bt + 1]; p++) {
            const int i = bi[p];
            if (rdone[i]) { T.r[T.n] = i; T.j[T.n] = j; T.v[T.n] = bx[p]; T.n++; }
            else { A.ci[e] = i; A.cv[e] = bx[p]; e++; A.cl[j]++; A.rl[i]++; }
        }
        A.cc[j] = A.cl[j] + 2; e += 2;
    }
    A.cend = e;
    /* 2b. long columns set aside (22 October 2026): every pivot whose row meets a column updates it, at a cost of the
     * column's length, so a column of length L costs about L^2 over the elimination; set aside, it is brought back once
     * the other columns are pivoted, at the cost of one pass over L (sxlu_defer_back). A column whose L^2 exceeds the
     * active rows and entries is set aside: on lp_densecols_small (20 dense columns of 10,000 rows) the factorizations
     * took 1.2 s each, the searches and scatters of the dense columns at every pivot */
    int ndef = 0, *defj = NULL;
    {
        long nza = 0; const int nr0 = nar;
        for (int t = 0; t < nap; t++) nza += A.cl[apos[t]];
        for (int t = 0; t < nap; t++) {
            const int j = apos[t];
            if (var[j] >= n || A.cl[j] < SX_DENSE_MIN) continue;
            if ((double)A.cl[j] * A.cl[j] <= (double)nr0 + (double)nza) continue;
            if (!defj) defj = xmalloc(sizeof(int) * m);
            defj[ndef++] = j; A.cact[j] = 0;
            for (int p = A.cs[j]; p < A.cs[j] + A.cl[j]; p++) A.rl[A.ci[p]]--;
            A.cl[j] = 0;
        }
    }
    /* no column count exceeds the rows left, no row count the positions left */
    A.kmax = nar > nap ? nar : nap;
    for (int k = 0; k <= A.kmax; k++) A.chead[k] = A.rhead[k] = -1;
    e = 0;
    for (int t = 0; t < nar; t++) { const int i = arow[t]; A.rs[i] = e; A.ract[i] = 1; A.rc[i] = A.rl[i] + 2; e += A.rc[i]; A.rl[i] = 0; }
    A.rend = e;
    for (int t = 0; t < nap; t++) { const int j = apos[t]; if (A.cact[j]) for (int p = A.cs[j]; p < A.cs[j] + A.cl[j]; p++) { const int i = A.ci[p]; A.ri[A.rs[i] + A.rl[i]++] = j; } }
    for (int t = 0; t < nar; t++) A.rp[arow[t]] = A.rl[arow[t]];
    /* the entries, rows and columns of the active matrix, for the switch to the dense kernel */
    long nzact = 0; int nca = 0, nra = nar;
    for (int t = 0; t < nap; t++) { const int j = apos[t]; if (A.cact[j]) { lst_ins(A.chead, A.cnext, A.cprev, A.cl[j], j); nca++; nzact += A.cl[j]; } }
    for (int t = 0; t < nar; t++) lst_ins(A.rhead, A.rnext, A.rprev, A.rl[arow[t]], arow[t]);
    /* 3. elimination */
    int *wpos = F->wmark;          /* row -> position in the column being updated (+1; 0 none) */
    int *lrows = F->wlist; double *lmul = F->wval; int *ucols = F->lrorder;   /* (lrorder is rebuilt after the elimination) */
    /* (25 October 2026) the column updates in one pass over each column (lu_onepass): wpos holds the pivot column's rows
     * (k + 1 for lrows[k]) for the whole pivot step, umk marks the rows of the column met, upos the positions to update */
    const int onepass = lu_onepass();
    char *umk = W->umk; int *upos = W->upos;     /* (umk is kept zero) */
    int r, c;
    const char *de = getenv("SXDENSE");      /* (experiments; read at each factorization: no static, ISSUES 422) */
    unsigned lu_chk = 0; int lu_stopped = 0;
    for (;;) {
        /* (25 October 2026) with the blocked dense kernel the switch at 30 % instead of 70 %: a blocked step costs about a tenth
         * of a sparse elimination step per multiply-add, and the sparse work falls with the square of the density, so the
         * two meet near sqrt(0.1); pilot87's crossover (LP) 1.12 -> 0.84 s at 20 %, qap15's 4.62 -> 4.48 s */
        /* (the dense array at most 25 million entries, 200 MB, three times the active entries at 30 %; above, as before) */
        const double dfrac = de ? atof(de) : dense_blk_on() && (double)nca * nra <= dense_cap() ? SX_DENSE_FRAC_BLK : SX_DENSE_FRAC;
        if (nca >= SX_DENSE_MIN && nra >= SX_DENSE_MIN && (double)nzact >= dfrac * (double)nca * nra) {
            if (ndef) { defer_back(F, &A, &T, m, Ap, Ai, Ax, var, rdone, defj, ndef, &nca, &nzact); ndef = 0; }
            { const double td = (double)clock() / CLOCKS_PER_SEC;
              dense_kernel(F, &A, &T, u, rowof, prow, pval, &np);
              if (getenv("SXDBG") && atoi(getenv("SXDBG")) > 1) printf("    [dense kernel: %d x %d active, %.3f s]\n", nra, nca, (double)clock() / CLOCKS_PER_SEC - td); }
            break;
        }
        /* (14 November 2026, ISSUES 884) the stop request and the deadline, every 4 pivots: the crossover's start basis on
         * highschool1-aigio (LP, 72,004 rows from a list ahead of 192,975 candidates) was factorized for over 70 s past the
         * time limit, which the crossover checks only between its pushes; stopped, the columns left are replaced by logicals */
        if (!lu_stopped && (F->stopf || F->deadline > 0) && (++lu_chk & 3) == 0) {
            if ((F->stopf && __atomic_load_n(F->stopf, __ATOMIC_ACQUIRE)) || (F->deadline > 0 && lu_now() > F->deadline)) lu_stopped = 1;
        }
        const int fp = lu_stopped ? 0 : find_pivot(&A, u, &r, &c);
        nca -= A.nrej; nzact -= A.nrejnz; A.nrej = 0; A.nrejnz = 0;
        if (!fp) {
            if (ndef) {     /* the columns set aside come back, updated by L, for the rest of the elimination */
                defer_back(F, &A, &T, m, Ap, Ai, Ax, var, rdone, defj, ndef, &nca, &nzact);
                ndef = 0; continue;
            }
            break;
        }
        nca--; nra--; nzact -= A.cl[c];
        const int pp = col_find(&A, c, r);
        const double piv = A.cv[pp];
        /* out of the count lists: the pivot row and column, the rows of column c, the columns of row r */
        lst_del(A.chead, A.cnext, A.cprev, A.cl[c], c); lst_del(A.rhead, A.rnext, A.rprev, A.rl[r], r);
        int nlr = 0;
        for (int p = A.cs[c]; p < A.cs[c] + A.cl[c]; p++) {
            const int i = A.ci[p];
            if (i == r) continue;
            lst_del(A.rhead, A.rnext, A.rprev, A.rl[i], i);
            lrows[nlr] = i; lmul[nlr] = A.cv[p] / piv; nlr++;
            row_remove(&A, i, c);
        }
        A.cact[c] = 0;            /* (row r stays active until its pattern has been used: a compaction keeps it) */
        /* L column */
        if (nlr) {
            F->lrow[F->nl] = r; F->lstart[F->nl + 1] = F->lstart[F->nl];
            for (int k = 0; k < nlr; k++) l_add(F, lrows[k], lmul[k]);
            F->nl++;
        }
        /* U row and the update of the columns of row r (its pattern copied: compactions move the row file) */
        int nur = 0;
        for (int p = A.rs[r]; p < A.rs[r] + A.rp[r]; p++) if (A.ri[p] != c && A.cact[A.ri[p]]) ucols[nur++] = A.ri[p];
        if (onepass) {
            /* (25 October 2026) each column of row r in one pass: its entry in row r found and its entries in the pivot
             * column's rows collected together; then those updated, the row-r entry removed (the last entry in its
             * place), the fill appended in the order of the pivot column. The same entries, values and order as the
             * three passes before (the search for row r, the positions scattered, cleared again), so the same factors:
             * on pilot87 (LP) the three passes over long columns were 2.0e9 of the crossover's 5.5e9 instructions */
            for (int k = 0; k < nlr; k++) wpos[lrows[k]] = k + 1;
            for (int t0 = 0; t0 < nur; t0++) {
                const int j = ucols[t0];
                lst_del(A.chead, A.cnext, A.cprev, A.cl[j], j);
                const int s = A.cs[j], e = s + A.cl[j];
                int q = -1, nu = 0;
                for (int t = s; t < e; t++) { const int i = A.ci[t]; if (i == r) q = t; else if (wpos[i]) { upos[nu++] = t; umk[i] = 1; } }
                const double urv = A.cv[q];
                for (int z = 0; z < nu; z++) { const int t = upos[z]; A.cv[t] -= lmul[wpos[A.ci[t]] - 1] * urv; }
                A.ci[q] = A.ci[e - 1]; A.cv[q] = A.cv[e - 1]; A.cl[j]--; nzact--;
                trip_add(&T, r, j, urv);
                F->factor_work += nlr;
                if (nlr > nu) {
                    col_reserve(&A, j, nlr - nu);
                    for (int k = 0; k < nlr; k++) {
                        const int i = lrows[k];
                        if (umk[i]) { umk[i] = 0; continue; }
                        const int t = A.cs[j] + A.cl[j]++; nzact++;
                        A.ci[t] = i; A.cv[t] = -(lmul[k] * urv);
                        row_reserve(&A, i, 1); A.ri[A.rs[i] + A.rp[i]++] = j; A.rl[i]++;
                    }
                } else for (int k = 0; k < nlr; k++) umk[lrows[k]] = 0;
                A.cmax[j] = -1;
                lst_ins(A.chead, A.cnext, A.cprev, A.cl[j], j);
            }
            for (int k = 0; k < nlr; k++) wpos[lrows[k]] = 0;
        } else
        for (int t0 = 0; t0 < nur; t0++) {
            const int j = ucols[t0];
            lst_del(A.chead, A.cnext, A.cprev, A.cl[j], j);
            const int q = col_find(&A, j, r);
            const double urv = A.cv[q];
            A.ci[q] = A.ci[A.cs[j] + A.cl[j] - 1]; A.cv[q] = A.cv[A.cs[j] + A.cl[j] - 1]; A.cl[j]--; nzact--;
            trip_add(&T, r, j, urv);
            F->factor_work += nlr;
            if (nlr) {
                col_reserve(&A, j, nlr);
                const int s = A.cs[j];
                for (int t = s; t < s + A.cl[j]; t++) wpos[A.ci[t]] = t + 1;
                for (int k = 0; k < nlr; k++) {
                    const int i = lrows[k];
                    const double d = lmul[k] * urv;
                    if (wpos[i]) A.cv[wpos[i] - 1] -= d;
                    else {
                        const int t = A.cs[j] + A.cl[j]++; nzact++;
                        A.ci[t] = i; A.cv[t] = -d;
                        row_reserve(&A, i, 1); A.ri[A.rs[i] + A.rp[i]++] = j; A.rl[i]++;
                    }
                }
                for (int t = A.cs[j]; t < A.cs[j] + A.cl[j]; t++) wpos[A.ci[t]] = 0;
            }
            A.cmax[j] = -1;
            lst_ins(A.chead, A.cnext, A.cprev, A.cl[j], j);
        }
        A.ract[r] = 0;
        for (int k = 0; k < nlr; k++) lst_ins(A.rhead, A.rnext, A.rprev, A.rl[lrows[k]], lrows[k]);
        rowof[c] = r; prow[np] = r; pval[np] = piv; np++;
    }
    /* 4. a singular basis: the positions left get the logicals of the rows left */
    int nout = 0;
    char *repl = NULL;
    if (np < m) {
        /* rows left: neither a logical's row (rdone) nor pivoted in the elimination (inactive) */
        repl = xcalloc(m, 1);
        int ir = 0;
        for (int j = 0; j < m; j++) {
            if (rowof[j] >= 0) continue;
            while (ir < m && (rdone[ir] || !A.ract[ir])) ir++;
            if (ir == m) break;
            const int i = ir++;
            out[nout++] = var[j]; repl[j] = 1;
            var[j] = n + i; rowof[j] = i; A.ract[i] = 0;
            prow[np] = i; pval[np] = 1.0; np++;
        }
        F->rank_deficient = nout;
    }
    /* 6. U: columns and rows keyed by the pivot rows; its order and diagonal (the rows of L below need the order) */
    if (F->uzcap < 2 * m + 16) { F->uzcap = 2 * m + 16; F->uz = xrealloc(F->uz, sizeof(int) * F->uzcap); }
    if (!F->uzpos) F->uzpos = xmalloc(sizeof(int) * ((size_t)m + 1));
    int nu = 0;
    memset(F->ucl, 0, sizeof(int) * (size_t)m); memset(F->url, 0, sizeof(int) * (size_t)m);
    /* drop the entries of replaced positions; translate positions to their pivot rows; count */
    if (!repl) {
        for (int t = 0; t < T.n; t++) { const int rc = rowof[T.j[t]]; T.j[t] = rc; F->ucl[rc]++; F->url[T.r[t]]++; }
        nu = T.n;
    } else
        for (int t = 0; t < T.n; t++) {
            const int j = T.j[t];
            if (repl[j]) continue;      /* a replaced column: the logical has no entries in other rows */
            const int rc = rowof[j];
            T.j[nu] = rc; T.r[nu] = T.r[t]; T.v[nu] = T.v[t]; F->ucl[rc]++; F->url[T.r[nu]]++; nu++;
        }
    const int need = 2 * nu + 8 * m + 64;
    if (F->ucap < need) { F->ucap = need; F->uci = xrealloc(F->uci, sizeof(int) * need); F->ucv = xrealloc(F->ucv, sizeof(double) * need); }
    if (F->urcap < need) { F->urcap = need; F->uri = xrealloc(F->uri, sizeof(int) * need); F->urv = xrealloc(F->urv, sizeof(double) * need); }
    /* (3 November 2026) the columns stored in pivot order and without spare room: an update never adds to a column (its
     * spike is a new column, stored at the end of the storage, which is also the end of the order), so the forward
     * solve with U reads the columns in one backward sweep through memory. The row lists keep their room (the spikes
     * add to them). satellites2-25 (LP), dual simplex method: the forward solves 3.27 -> 2.88 s, the same bits */
    /* the pivots that do something in the forward solve with U (a column or a diagonal other than 1), in pivot order */
    F->nord = np; F->nuz = 0;
    if (np < m) memset(F->uzpos, 0xff, sizeof(int) * (size_t)m);
    {
        int ec = 0;
        for (int k = 0; k < np; k++) {
            const int i = prow[k], c0 = F->ucl[i];
            const double d = k < nlog ? 1.0 : pval[k];
            F->uordpos[i] = k; F->udiag[i] = d; F->urecip[i] = 1.0 / d;
            F->ucs[i] = ec; F->ucc[i] = c0; ec += c0; F->ucl[i] = 0;
            if (c0 > 0 || d != 1.0) { F->uzpos[i] = F->nuz; F->uz[F->nuz++] = i; } else F->uzpos[i] = -1;
        }
        F->ucend = ec;
    }
    /* 5. L by rows, and the per-row access of L columns (for the searches) */
    memset(F->lpos, 0xff, sizeof(int) * (size_t)m); memset(F->lcs_row, 0, sizeof(int) * (size_t)m); memset(F->lcl_row, 0, sizeof(int) * (size_t)m);
    for (int k = 0; k < F->nl; k++) { const int rr = F->lrow[k]; F->lpos[rr] = k; F->lcs_row[rr] = F->lstart[k]; F->lcl_row[rr] = F->lstart[k + 1] - F->lstart[k]; }
    const int nlz = F->lstart[F->nl];
    F->lridx = xrealloc(F->lridx, sizeof(int) * (nlz + 1)); F->lrval = xrealloc(F->lrval, sizeof(double) * (nlz + 1));
    {
        int *cnt = F->dfs_pos, *lrr = W->sj, nlrr = 0;     /* lrr: the rows with entries in L (sj is free now) */
        memset(F->lrstart, 0, sizeof(int) * ((size_t)m + 1));
        for (int p = 0; p < nlz; p++) { const int i = F->lidx[p]; if (F->lrstart[i + 1]++ == 0) lrr[nlrr++] = i; }
        { int run = 0; for (int i = 0; i < m; i++) { const int c0 = F->lrstart[i + 1]; F->lrlen[i] = c0; cnt[i] = run; run += c0; F->lrstart[i + 1] = run; } }
        for (int k = 0; k < F->nl; k++)
            for (int p = F->lstart[k]; p < F->lstart[k + 1]; p++) { const int i = F->lidx[p]; F->lridx[cnt[i]] = F->lrow[k]; F->lrval[cnt[i]] = F->lval[p]; cnt[i]++; }
        /* rows with entries in L, by decreasing pivot step: their steps marked in a bitmap, read from the top */
        int nro = 0;
        if (np == m) {
            unsigned long long *bits = W->bits;
            for (int t = 0; t < nlrr; t++) { const int k = F->uordpos[lrr[t]]; bits[k >> 6] |= 1ULL << (k & 63); }
            for (int w = (np - 1) >> 6; w >= 0; w--) {
                unsigned long long b = bits[w];
                if (!b) continue;
                bits[w] = 0;
                while (b) { const int h = 63 - __builtin_clzll(b); F->lrorder[nro++] = prow[(w << 6) + h]; b &= ~(1ULL << h); }
            }
        } else
            for (int k = np - 1; k >= 0; k--) { const int rr = prow[k]; if (F->lrstart[rr + 1] > F->lrstart[rr]) F->lrorder[nro++] = rr; }
        F->nlrorder = nro;
    }
    /* 6b. U's rows; the entries into the columns and rows */
    {
        int er = 0;
        for (int i = 0; i < m; i++) {
            F->urs[i] = er; F->urc[i] = F->url[i] + 4; er += F->urc[i]; F->url[i] = 0;
        }
        F->urend = er;
        for (int t = 0; t < nu; t++) {
            const int rc = T.j[t], rr = T.r[t];
            int p = F->ucs[rc] + F->ucl[rc]++; F->uci[p] = rr; F->ucv[p] = T.v[t];
            p = F->urs[rr] + F->url[rr]++; F->uri[p] = rc; F->urv[p] = T.v[t];
        }
    }
    /* 7. the basis reordered: the position of each column is its pivot row */
    {
        int *nv = F->dfs_pos;
        for (int j = 0; j < m; j++) nv[rowof[j]] = var[j];
        memcpy(var, nv, sizeof(int) * m);
    }
    F->nnz_l = nlz; F->nnz_u0 = nu; F->nnz_u = nu; F->nnz_r = 0;
    /* the work of the factorization in entries touched: the elimination's updates (counted above), and the passes
     * over the basis' entries and over the rows that every factorization makes (setting up the active matrix and the
     * count lists, the transposes of L and U, the reordering: about a dozen per row and four per entry) */
    F->factor_work += 4.0 * (nzs + nlz + nu) + 12.0 * m;
    if (!F->rstart) { F->rcapn = 256; F->rpiv = xmalloc(sizeof(int) * F->rcapn); F->rstart = xmalloc(sizeof(int) * (F->rcapn + 1)); }
    F->rstart[0] = 0;
    if (!F->ridx) { F->rcap = 4 * m + 1024; F->ridx = xmalloc(sizeof(int) * F->rcap); F->rval = xmalloc(sizeof(double) * F->rcap); }
    if (!F->rnext) { F->rnext = xmalloc(sizeof(int) * F->rcap); F->reta = xmalloc(sizeof(int) * F->rcap); }
    if (!F->rthead) { F->rthead = xmalloc(sizeof(int) * ((size_t)m + 1)); F->rtcnt = xmalloc(sizeof(int) * ((size_t)m + 1)); }
    memset(F->rthead, 0xff, sizeof(int) * (size_t)m); memset(F->rtcnt, 0, sizeof(int) * (size_t)m);
    memset(F->wmark, 0, sizeof(int) * (size_t)m);
    /* the work arrays back to the store (the recorded entries and the active matrix's storage may have moved) */
    W->T = T;
    if (A.nccomp) { W->ci = A.ci; W->cv = A.cv; W->cicap = A.ccap; }
    if (A.nrcomp) { W->ri = A.ri; W->ricap = A.rcap; }
    free(defj); free(repl);
    return nout;
}

/* ---------------------------------------------------------------- solves */
/* the density below which a solve is hypersparse (a depth-first search for the reach, then the work on it alone); the
 * switch BRISK_SXHYPER: an experiment (23 October 2026) */
/* (read once: called for every solve, its getenv was 3 % of ken-18's instructions, LP, 24 October 2026) */
static double hyper_thr(void) { static double t = -1; if (t < 0) { const char *e = getenv("BRISK_SXHYPER"); t = e ? atof(e) : 0.1; } return t; }
/* (24 October 2026, the night) the hypersparse solve costs its searches' edges, the sweep a pass over the pivots (the
 * columns of L, the pivots of U with work, the rows); the numeric work is the same. A direction whose recent hypersparse
 * solves examined more edges than a quarter of that pass sweeps instead, with a hypersparse trial every 32 solves (to
 * follow the basis). Only the solves with the factorization's own scratch space (one at a time, so the same choice on
 * every run). sing44 (LP): the searches were more than half the dual method's time. BRISK_SXHYCOST=0: the density
 * threshold alone */
static int hy_cost_on(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXHYCOST"); c = !(e && atoi(e) == 0); } return c; }
static int hy_allow(SxLU *F, int d) {
    if (!hy_cost_on()) return 1;
    const double pass = (double)F->nl + F->nuz + F->m;
    if (F->hy_e[d] <= 0.25 * pass) return 1;
    if (--F->hy_p[d] <= 0) { F->hy_p[d] = 32; return 1; }
    return 0;
}
static void hy_note(SxLU *F, int d, long edges) { F->hy_e[d] = F->hy_e[d] > 0 ? 0.9 * F->hy_e[d] + 0.1 * (double)edges : (double)edges + 1e-9; }
/* depth-first search from the nonzeros of x over the graph (start/len/adj keyed by row id); the reached
 * rows in postorder in F->wlist (count returned); F->wmark used and cleared by the caller */
static int dfs(SxWs *W, const SxVec *x, const int *start, const int *len, const int *adj) {
    unsigned char *mark = W->dmark; int *stk = W->stack, *pos = W->dfs_pos, *outl = W->wlist;
    int nout = 0; long ed = 0;
    for (int k = 0; k < x->cnt; k++) {
        const int r0 = x->idx[k];
        if (mark[r0]) continue;
        int top = 0; stk[0] = r0; pos[0] = 0; mark[r0] = 1; ed += len[r0];
        while (top >= 0) {
            const int r = stk[top];
            const int s = start[r], l = len[r];
            int p = pos[top];
            while (p < l && mark[adj[s + p]]) p++;
            if (p < l) {
                const int nx = adj[s + p];
                pos[top] = p + 1;
                /* (31 October 2026) the next candidate's start and length fetched ahead: each row reached costs two cache
                 * misses there, and the search was 31 % of a multicommodity flow's dual simplex method (S_mc2, LP) */
                if (p + 1 < l) { const int nn = adj[s + p + 1]; __builtin_prefetch(start + nn); __builtin_prefetch(len + nn); }
                mark[nx] = 1; top++; stk[top] = nx; pos[top] = 0; ed += len[nx];
            } else { outl[nout++] = r; top--; }
        }
    }
    W->nedges += ed;
    return nout;
}
static void unmark(SxWs *W, int nout) { for (int k = 0; k < nout; k++) W->dmark[W->wlist[k]] = 0; }
/* (14 November 2026, ISSUES 881; the dual simplex method's sparsest-member rule, BRISK_SXDEGBOUND=1) the number of nonzeros of
 * B^-1 a for the column whose rows are seed[0..ns-1], counted on the patterns (no cancellation): the rows reached through L's
 * pattern, then the row etas' pivot rows reached (in their order), then the rows reached through U's pattern. Each stage's set
 * holds the one before, so the count can stop once it passes bound: the result is then bound + 1 (a member that cannot be the
 * sparsest is known as such for the work of reaching bound + 1 rows, where the numeric solve reaches all of them) */
long sxlu_count_bounded(SxLU *F, SxWs *W, const int *seed, int ns, long bound) {
    unsigned char *mark = W->dmark; int *stk = W->stack, *pos = W->dfs_pos, *vis = W->wlist;
    long nv = 0, ed = 0;
    /* a depth-first search from root over (start, len, adj), marking and listing what it reaches; returns 1 when past bound */
#define CB_DFS(root, start, len, adj) do { \
        int top_ = 0; stk[0] = (root); pos[0] = 0; \
        while (top_ >= 0) { \
            const int r_ = stk[top_]; const int s_ = (start)[r_], l_ = (len)[r_]; int p_ = pos[top_]; \
            while (p_ < l_ && mark[(adj)[s_ + p_]]) p_++; \
            ed += p_ - pos[top_] + 1; \
            if (p_ < l_) { const int nx_ = (adj)[s_ + p_]; pos[top_] = p_ + 1; mark[nx_] = 1; vis[nv++] = nx_; if (nv > bound) goto cb_done; top_++; stk[top_] = nx_; pos[top_] = 0; } \
            else top_--; \
        } } while (0)
    for (int k = 0; k < ns; k++) {
        const int r0 = seed[k]; if (mark[r0]) continue;
        mark[r0] = 1; vis[nv++] = r0; if (nv > bound) goto cb_done;
        CB_DFS(r0, F->lcs_row, F->lcl_row, F->lidx);
    }
    for (int e = 0; e < F->nr; e++) {
        const int r = F->rpiv[e]; if (mark[r]) continue;
        for (int p = F->rstart[e]; p < F->rstart[e + 1]; p++) if (mark[F->ridx[p]]) { mark[r] = 1; vis[nv++] = r; break; }
        ed += F->rstart[e + 1] - F->rstart[e];
        if (nv > bound) goto cb_done;
    }
    {   const long n1 = nv;
        for (long k = 0; k < n1; k++) CB_DFS(vis[k], F->ucs, F->ucl, F->uci);
    }
cb_done:
#undef CB_DFS
    for (long k = 0; k < nv; k++) mark[vis[k]] = 0;
    W->nedges += ed;
    return nv > bound ? bound + 1 : nv;
}
/* the factorization's own scratch space, as a workspace */
static SxWs own_ws(SxLU *F) { SxWs W; W.m = F->m; W.stack = F->stack; W.wmark = F->wmark; W.wlist = F->wlist; W.dfs_pos = F->dfs_pos; W.dmark = F->dmark; W.nflops = 0; W.nedges = 0; W.ubm = F->ubm; W.ubs = F->ubs; return W; }
void sxws_init(SxWs *W, int m) {
    W->m = m; W->stack = malloc(sizeof(int) * (2 * (size_t)m + 2)); W->wmark = calloc((size_t)m + 1, sizeof(int));
    W->wlist = malloc(sizeof(int) * ((size_t)m + 1)); W->dfs_pos = malloc(sizeof(int) * ((size_t)m + 1)); W->dmark = calloc((size_t)m + 1, 1); W->nflops = 0; W->nedges = 0;
    W->ubm = calloc(2 * (size_t)m / 64 + 2, sizeof(unsigned long long)); W->ubs = calloc(2 * (size_t)m / 4096 + 2, sizeof(unsigned long long));
    if (!W->stack || !W->wmark || !W->wlist || !W->dfs_pos || !W->dmark) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); }
}
void sxws_free(SxWs *W) { free(W->stack); free(W->wmark); free(W->wlist); free(W->dfs_pos); free(W->dmark); free(W->ubm); free(W->ubs); memset(W, 0, sizeof *W); }

/* the forward solve keeps its list of nonzeros as it goes when the pivots of U with work are fewer than half the rows
 * (a basis mostly of logicals: a pass over all rows to find the nonzeros would cost more than the solve); otherwise
 * one pass at the end (cheaper than a test per update) */
#define TRACK(F) ((F)->nuz < (F)->m / 2)
/* x <- L^{-1} x */
static void ftran_l(SxLU *F, SxWs *W, SxVec *x, int hyper) {
    double *v = x->val;
    if (hyper) {
        const int nr = dfs(W, x, F->lcs_row, F->lcl_row, F->lidx);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            const double xr = v[r];
            if (xr != 0 && F->lpos[r] >= 0) {
                const int l = F->lpos[r];
                W->nflops += F->lstart[l + 1] - F->lstart[l];
            for (int p = F->lstart[l]; p < F->lstart[l + 1]; p++) v[F->lidx[p]] -= F->lval[p] * xr;
            }
        }
        for (int k = 0; k < nr; k++) x->idx[k] = W->wlist[k];
        x->cnt = nr; unmark(W, nr);
        W->nflops += nr;
    } else if (TRACK(F)) {
        /* (the list of nonzeros kept as entries appear; one that cancels keeps 1e-300 and is pruned afterwards) */
        int *idx = x->idx, cnt = x->cnt;
        for (int l = 0; l < F->nl; l++) {
            const double xr = v[F->lrow[l]];
            if (xr == 0) continue;
            W->nflops += F->lstart[l + 1] - F->lstart[l];
            for (int p = F->lstart[l]; p < F->lstart[l + 1]; p++) {
                const int i = F->lidx[p];
                if (v[i] == 0) idx[cnt++] = i;
                v[i] -= F->lval[p] * xr;
                if (v[i] == 0) v[i] = 1e-300;
            }
        }
        x->cnt = cnt;
        W->nflops += F->nl;
    } else {
        for (int l = 0; l < F->nl; l++) {
            const double xr = v[F->lrow[l]];
            if (xr == 0) continue;
            W->nflops += F->lstart[l + 1] - F->lstart[l];
            for (int p = F->lstart[l]; p < F->lstart[l + 1]; p++) v[F->lidx[p]] -= F->lval[p] * xr;
        }
        x->cnt = -1;
        W->nflops += F->nl;
    }
}
/* x <- R_k ... R_1 x. (24 October 2026, the night) With a sparse x, only the etas that read its nonzeros: from each
 * nonzero's list of readers, in the order of the etas (a heap), an eta that makes a new nonzero bringing in that row's later
 * readers; the others would add zero. The same etas with a nonzero sum, in the same order, so the same bits. The pass over
 * every eta was 45 microseconds a solve on lp_beckmann_300's crossover (LP, a basis of 90,000 rows), 12 % of ken-18's dual
 * method. Used when the readers of x's nonzeros are fewer than a quarter of the file's entries; the heap in W->stack, the
 * marks in W->dmark, the etas taken in W->wlist (all free between the solves with L and with U; at most m etas). */
static void ftran_r_hyper(SxLU *F, SxWs *W, SxVec *x) {
    double *v = x->val;
    int *heap = W->stack, *took = W->wlist, nh = 0, nt = 0;
    unsigned char *seen = W->dmark;
    long work = 0;
#define RH_PUSH(e0) do { int q_ = nh++; const int e_ = (e0); while (q_ > 0 && heap[(q_ - 1) / 2] > e_) { heap[q_] = heap[(q_ - 1) / 2]; q_ = (q_ - 1) / 2; } heap[q_] = e_; } while (0)
    const int c0 = x->cnt;
    for (int k = 0; k < c0; k++) {
        const int i = x->idx[k];
        for (int p = F->rthead[i]; p >= 0; p = F->rnext[p]) { const int e = F->reta[p]; work++; if (!seen[e]) { seen[e] = 1; RH_PUSH(e); } }
    }
    while (nh > 0) {
        const int e = heap[0];
        {   /* pop */
            const int last = heap[--nh]; int q = 0;
            for (;;) { int c = 2 * q + 1; if (c >= nh) break; if (c + 1 < nh && heap[c + 1] < heap[c]) c++; if (heap[c] >= last) break; heap[q] = heap[c]; q = c; }
            if (nh > 0) heap[q] = last;
        }
        took[nt++] = e;
        double s = 0;
        for (int p = F->rstart[e]; p < F->rstart[e + 1]; p++) s += F->rval[p] * v[F->ridx[p]];
        work += F->rstart[e + 1] - F->rstart[e];
        if (s != 0) {
            const int r = F->rpiv[e];
            if (v[r] == 0) {
                x->idx[x->cnt++] = r;
                for (int p = F->rthead[r]; p >= 0 && F->reta[p] > e; p = F->rnext[p]) { const int e2 = F->reta[p]; work++; if (!seen[e2]) { seen[e2] = 1; RH_PUSH(e2); } }
            }
            v[r] -= s;
            if (v[r] == 0) v[r] = 1e-300;   /* keeps the index valid; pruned at the end */
        }
    }
#undef RH_PUSH
    for (int k = 0; k < nt; k++) seen[took[k]] = 0;
    W->nflops += work;
}
static void ftran_r(SxLU *F, SxWs *W, SxVec *x) {
    double *v = x->val;
    if (x->cnt >= 0 && F->nr <= F->m && F->rthead) {
        long est = 0; const long tot = F->rstart[F->nr];
        for (int k = 0; k < x->cnt && 4 * est < tot; k++) est += F->rtcnt[x->idx[k]];
        if (4 * est < tot) { ftran_r_hyper(F, W, x); return; }
    }
    for (int e = 0; e < F->nr; e++) {
        double s = 0;
        for (int p = F->rstart[e]; p < F->rstart[e + 1]; p++) s += F->rval[p] * v[F->ridx[p]];
        if (s != 0) {
            const int r = F->rpiv[e];
            if (x->cnt >= 0 && v[r] == 0) x->idx[x->cnt++] = r;
            v[r] -= s;
            if (v[r] == 0) v[r] = 1e-300;   /* keeps the index valid; pruned at the end */
        }
    }
    W->nflops += F->rstart[F->nr];
}
/* (31 October 2026) x <- U^{-1} x by a sweep over the positions of U's order, downward, through a bitmap of the rows reached
 * (and a bitmap of its nonzero words): a row is taken once every later pivot has added to it (the rows a column reaches lie
 * before it, so one downward sweep serves), and its column is applied only when its value is not zero, so a part of the
 * reach where the contributions cancel exactly is not entered. A heap in place of the bitmap cost more than the search it
 * saved (S_mc2's solves 19.6 -> 26.5 s). On a network basis (entries of one size and both signs) the
 * solve for an arc reaches the paths from its two ends to the root of the spanning tree, and above the point where they
 * meet the two contributions cancel: the depth-first search visited about twice the nonzeros it produced (S_mc2, a
 * multicommodity flow, LP: 1,900-2,900 rows reached, 890 nonzero), at about 20 ns an edge, 31 % of the dual simplex
 * method's time. Each row's sum is taken in the order of the positions, the same on every run. */
static void ftran_u_bits(SxLU *F, SxWs *W, SxVec *x) {
    double *v = x->val; unsigned long long *bm = W->ubm, *bs = W->ubs; int *out = W->wlist;
    int nout = 0, shi = -1; long work = 0;
#define UB_SET(pos_) do { const int p_ = (pos_), w_ = p_ >> 6; const unsigned long long b_ = 1ULL << (p_ & 63); \
        if (!(bm[w_] & b_)) { if (!bm[w_]) { bs[w_ >> 6] |= 1ULL << (w_ & 63); if ((w_ >> 6) > shi) shi = w_ >> 6; } bm[w_] |= b_; } } while (0)
    for (int k = 0; k < x->cnt; k++) UB_SET(F->uordpos[x->idx[k]]);
    for (int sw = shi; sw >= 0; sw--)
        while (bs[sw]) {
            const int wb = 63 - __builtin_clzll(bs[sw]), w = (sw << 6) | wb;
            while (bm[w]) {
                const int b = 63 - __builtin_clzll(bm[w]);
                bm[w] &= ~(1ULL << b);
                const int r = F->uord[(w << 6) | b];
                out[nout++] = r;
                if (v[r] == 0) continue;
                const double xr = v[r] * F->urecip[r];
                v[r] = xr;
                const int s0 = F->ucs[r], l = F->ucl[r];
                work += l;
                for (int p = s0; p < s0 + l; p++) { const int i = F->uci[p]; UB_SET(F->uordpos[i]); v[i] -= F->ucv[p] * xr; }
            }
            bs[sw] &= ~(1ULL << wb);
        }
#undef UB_SET
    for (int k = 0; k < nout; k++) x->idx[k] = out[k];
    x->cnt = nout; sxv_prune(x);
    W->nflops += work + nout;
}
static int uheap_on(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXUHEAP"); c = !(e && atoi(e) == 0); } return c; }
/* x <- U^{-1} x */
static void ftran_u(SxLU *F, SxWs *W, SxVec *x, int hyper, int track) {
    double *v = x->val;
    /* (31 October 2026) the sweep while the depth-first solves find a third or more of their reach ending zero;
     * every 16th solve by the depth-first search, to follow the basis (BRISK_SXUHEAP=0: never) */
    if (hyper && x->cnt >= 0 && uheap_on() && W->ubm && F->nord <= 2 * F->m && F->hy_canc >= 0.3 && --F->hy_cp > 0) { ftran_u_bits(F, W, x); return; }
    if (hyper) {
        const int nr = dfs(W, x, F->ucs, F->ucl, F->uci);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            if (v[r] == 0) continue;
            const double xr = v[r] * F->urecip[r];
            v[r] = xr;
            W->nflops += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) v[F->uci[p]] -= F->ucv[p] * xr;
        }
        for (int k = 0; k < nr; k++) x->idx[k] = W->wlist[k];
        x->cnt = nr; unmark(W, nr); sxv_prune(x);
        W->nflops += nr;
        if (nr >= 64) { F->hy_canc = 0.9 * F->hy_canc + 0.1 * (double)(nr - x->cnt) / nr; F->hy_cp = 16; }
    } else if (track && TRACK(F) && x->cnt >= 0) {
        /* the pivots with an empty column and a unit diagonal (the logicals) change nothing: only the others (uz) */
        int *idx = x->idx, cnt = x->cnt;
        for (int k = F->nuz - 1; k >= 0; k--) {
            const int r = F->uz[k];
            if (r < 0 || v[r] == 0) continue;
            const double xr = v[r] * F->urecip[r];
            v[r] = xr;
            W->nflops += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) {
                const int i = F->uci[p];
                if (v[i] == 0) idx[cnt++] = i;
                v[i] -= F->ucv[p] * xr;
                if (v[i] == 0) v[i] = 1e-300;
            }
        }
        x->cnt = cnt; sxv_prune(x);
        W->nflops += F->nuz;
    } else {
        for (int k = F->nuz - 1; k >= 0; k--) {
            const int r = F->uz[k];
            if (r < 0 || v[r] == 0) continue;
            const double xr = v[r] * F->urecip[r];
            v[r] = xr;
            W->nflops += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) v[F->uci[p]] -= F->ucv[p] * xr;
        }
        sxv_reindex(x);
        W->nflops += F->nuz;
    }
}
/* (8 November 2026, a diagnostic: BRISK_SXFTPARTS=1) the forward solves' work and time by part (L, the row etas, U), at exit */
static double ftp_w[3], ftp_t[3], ftp_in[3], ftp_out[3]; static long ftp_n, ftp_hyp;
static void ftp_print(void) { fprintf(stderr, "[ftparts] %ld solves (%ld hypersparse): L work %.3g time %.3f s, in %.3g; R work %.3g time %.3f s, in %.3g; U work %.3g time %.3f s, in %.3g out %.3g\n",
    ftp_n, ftp_hyp, ftp_w[0], ftp_t[0], ftp_in[0], ftp_w[1], ftp_t[1], ftp_in[1], ftp_w[2], ftp_t[2], ftp_in[2], ftp_out[2]); }
static int ftp_on(void) { static int c = -1; if (c < 0) { c = getenv("BRISK_SXFTPARTS") != NULL; if (c) atexit(ftp_print); } return c; }
static double ftp_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
void sxlu_ftran_ws(SxLU *F, SxWs *W, SxVec *x, double expdens, SxVec *spike) {
    const int m = F->m;
    if (x->cnt < 0) sxv_reindex(x);
    const double ht = hyper_thr();
    const int own = W->stack == F->stack;
    const int hyper = x->cnt < ht * m && expdens < ht && (!own || hy_allow(F, 0));
    const long e0 = W->nedges;
    const int fp = ftp_on(); double t_ = 0; long w_ = 0;
    if (fp) { ftp_n++; ftp_hyp += hyper; t_ = ftp_now(); w_ = W->nflops + W->nedges; ftp_in[0] += x->cnt; }
    ftran_l(F, W, x, hyper);
    if (x->cnt >= 0) sxv_prune(x);
    if (F->nr && x->cnt < 0) sxv_reindex(x);
    if (fp) { const double t2 = ftp_now(); const long w2 = W->nflops + W->nedges; ftp_t[0] += t2 - t_; ftp_w[0] += w2 - w_; t_ = t2; w_ = w2; ftp_in[1] += x->cnt >= 0 ? x->cnt : m; }
    if (F->nr) ftran_r(F, W, x);
    if (x->cnt < 0) sxv_reindex(x);
    else sxv_prune(x);
    if (fp) { const double t2 = ftp_now(); const long w2 = W->nflops + W->nedges; ftp_t[1] += t2 - t_; ftp_w[1] += w2 - w_; t_ = t2; w_ = w2; ftp_in[2] += x->cnt; }
    if (spike) sxv_copy(spike, x);
    /* (22 October 2026) the result's nonzeros tracked during the solve only when it is expected sparse; a dense one (the
     * entering columns of lp_setcover_5k, LP: 59 % of the rows) takes the plain loop and one pass to list it */
    ftran_u(F, W, x, hyper && x->cnt < ht * m, expdens < 0.3);
    if (fp) { ftp_t[2] += ftp_now() - t_; ftp_w[2] += W->nflops + W->nedges - w_; ftp_out[2] += x->cnt >= 0 ? x->cnt : m; }
    if (own && hyper) hy_note(F, 0, W->nedges - e0);
}

void sxlu_ftran(SxLU *F, SxVec *x, double expdens, SxVec *spike) {
    SxWs W = own_ws(F);
    sxlu_ftran_ws(F, &W, x, expdens, spike);
    F->nflops += W.nflops; F->nedges += W.nedges;
}
/* (3 November 2026) two forward solves through the factor in one pass over L and one over U, for the dual simplex method's
 * entering column and the steepest-edge weights' vector, when neither solve is hypersparse. Each vector goes through the
 * same operations in the same order as in its own solve (sxlu_ftran_ws: L, then the row etas R, then U), so the results
 * and the counted work are the same bits; what is shared is the pass over the factor: its indices and values are read
 * once, and the pivot loop's bookkeeping is done once. On satellites2-25 (LP) both vectors are dense (the entering column
 * a third of the rows, the weights' vector three quarters) and the two forward solves were 43 % of the dual method's time,
 * most of it in the loops over U's pivots, about two entries a column. */
static void ftran_l_pair(SxLU *F, SxWs *W, SxVec *x1, SxVec *x2) {
    double *v1 = x1->val, *v2 = x2->val;
    const int *lrow = F->lrow, *lstart = F->lstart, *lidx = F->lidx; const double *lval = F->lval;
    long fl = 0;
    if (TRACK(F)) {
        int *i1 = x1->idx, *i2 = x2->idx, c1 = x1->cnt, c2 = x2->cnt;
        for (int l = 0; l < F->nl; l++) {
            const double a = v1[lrow[l]], b = v2[lrow[l]];
            const int s = lstart[l], e = lstart[l + 1];
            if (a != 0 && b != 0) {
                fl += 2L * (e - s);
                for (int p = s; p < e; p++) {
                    const int i = lidx[p]; const double lv = lval[p];
                    if (v1[i] == 0) i1[c1++] = i;
                    v1[i] -= lv * a; if (v1[i] == 0) v1[i] = 1e-300;
                    if (v2[i] == 0) i2[c2++] = i;
                    v2[i] -= lv * b; if (v2[i] == 0) v2[i] = 1e-300;
                }
            } else if (a != 0) {
                fl += e - s;
                for (int p = s; p < e; p++) { const int i = lidx[p]; if (v1[i] == 0) i1[c1++] = i; v1[i] -= lval[p] * a; if (v1[i] == 0) v1[i] = 1e-300; }
            } else if (b != 0) {
                fl += e - s;
                for (int p = s; p < e; p++) { const int i = lidx[p]; if (v2[i] == 0) i2[c2++] = i; v2[i] -= lval[p] * b; if (v2[i] == 0) v2[i] = 1e-300; }
            }
        }
        x1->cnt = c1; x2->cnt = c2;
    } else {
        for (int l = 0; l < F->nl; l++) {
            const double a = v1[lrow[l]], b = v2[lrow[l]];
            const int s = lstart[l], e = lstart[l + 1];
            if (a != 0 && b != 0) {
                fl += 2L * (e - s);
                for (int p = s; p < e; p++) { const int i = lidx[p]; const double lv = lval[p]; v1[i] -= lv * a; v2[i] -= lv * b; }
            } else if (a != 0) {
                fl += e - s;
                for (int p = s; p < e; p++) v1[lidx[p]] -= lval[p] * a;
            } else if (b != 0) {
                fl += e - s;
                for (int p = s; p < e; p++) v2[lidx[p]] -= lval[p] * b;
            }
        }
        x1->cnt = -1; x2->cnt = -1;
    }
    W->nflops += fl + 2L * F->nl;
}
/* U for both vectors, the plain sweep (track 0) or the sweep that keeps the lists (track 1), as ftran_u does for each */
static void ftran_u_pair(SxLU *F, SxWs *W, SxVec *x1, SxVec *x2, int track) {
    double *v1 = x1->val, *v2 = x2->val;
    const int *uz = F->uz, *ucs = F->ucs, *ucl = F->ucl, *uci = F->uci; const double *ucv = F->ucv, *urc = F->urecip;
    long fl = 0;
    if (track) {
        int *i1 = x1->idx, *i2 = x2->idx, c1 = x1->cnt, c2 = x2->cnt;
        for (int k = F->nuz - 1; k >= 0; k--) {
            const int r = uz[k];
            if (r < 0) continue;
            double a = v1[r], b = v2[r];
            if (a == 0 && b == 0) continue;
            const int s = ucs[r], e = s + ucl[r];
            if (a != 0) { a *= urc[r]; v1[r] = a; fl += e - s; }
            if (b != 0) { b *= urc[r]; v2[r] = b; fl += e - s; }
            if (a != 0 && b != 0) {
                for (int p = s; p < e; p++) {
                    const int i = uci[p]; const double uv = ucv[p];
                    if (v1[i] == 0) i1[c1++] = i;
                    v1[i] -= uv * a; if (v1[i] == 0) v1[i] = 1e-300;
                    if (v2[i] == 0) i2[c2++] = i;
                    v2[i] -= uv * b; if (v2[i] == 0) v2[i] = 1e-300;
                }
            } else if (a != 0) {
                for (int p = s; p < e; p++) { const int i = uci[p]; if (v1[i] == 0) i1[c1++] = i; v1[i] -= ucv[p] * a; if (v1[i] == 0) v1[i] = 1e-300; }
            } else {
                for (int p = s; p < e; p++) { const int i = uci[p]; if (v2[i] == 0) i2[c2++] = i; v2[i] -= ucv[p] * b; if (v2[i] == 0) v2[i] = 1e-300; }
            }
        }
        x1->cnt = c1; x2->cnt = c2; sxv_prune(x1); sxv_prune(x2);
    } else {
        for (int k = F->nuz - 1; k >= 0; k--) {
            const int r = uz[k];
            if (r < 0) continue;
            double a = v1[r], b = v2[r];
            if (a == 0 && b == 0) continue;
            const int s = ucs[r], e = s + ucl[r];
            if (a != 0 && b != 0) {
                a *= urc[r]; b *= urc[r]; v1[r] = a; v2[r] = b; fl += 2L * (e - s);
                for (int p = s; p < e; p++) { const int i = uci[p]; const double uv = ucv[p]; v1[i] -= uv * a; v2[i] -= uv * b; }
            } else if (a != 0) {
                a *= urc[r]; v1[r] = a; fl += e - s;
                for (int p = s; p < e; p++) v1[uci[p]] -= ucv[p] * a;
            } else {
                b *= urc[r]; v2[r] = b; fl += e - s;
                for (int p = s; p < e; p++) v2[uci[p]] -= ucv[p] * b;
            }
        }
        sxv_reindex(x1); sxv_reindex(x2);
    }
    W->nflops += fl + 2L * F->nuz;
}
/* both vectors as sxlu_ftran(x1, d1, spike1) then sxlu_ftran(x2, d2, NULL) would solve them; in one pass when neither is
 * hypersparse (BRISK_SXFTRANPAIR=0: always the two solves) */
static int ftran_pair_on(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXFTRANPAIR"); c = !(e && atoi(e) == 0); } return c; }
void sxlu_ftran_pair(SxLU *F, SxVec *x1, double d1, SxVec *spike1, SxVec *x2, double d2) {
    const int m = F->m; const double ht = hyper_thr();
    if (x1->cnt < 0) sxv_reindex(x1);
    if (x2->cnt < 0) sxv_reindex(x2);
    /* (the hypersparse test of sxlu_ftran_ws, its cheap part first, so that its cost rule is not consulted: when it is
     * false for both, neither solve would be hypersparse, and the rule's counters are left as the two solves leave them) */
    if (!ftran_pair_on() || (x1->cnt < ht * m && d1 < ht) || (x2->cnt < ht * m && d2 < ht)) {
        sxlu_ftran(F, x1, d1, spike1); sxlu_ftran(F, x2, d2, NULL); return;
    }
    SxWs W = own_ws(F);
    ftran_l_pair(F, &W, x1, x2);
    SxVec *xs[2] = { x1, x2 };
    for (int t = 0; t < 2; t++) {
        SxVec *x = xs[t];
        if (x->cnt >= 0) sxv_prune(x);
        if (F->nr && x->cnt < 0) sxv_reindex(x);
        if (F->nr) ftran_r(F, &W, x);
        if (x->cnt < 0) sxv_reindex(x); else sxv_prune(x);
    }
    if (spike1) sxv_copy(spike1, x1);
    const int tr1 = d1 < 0.3 && TRACK(F) && x1->cnt >= 0, tr2 = d2 < 0.3 && TRACK(F) && x2->cnt >= 0;
    if (tr1 == tr2) ftran_u_pair(F, &W, x1, x2, tr1);
    else { ftran_u(F, &W, x1, 0, d1 < 0.3); ftran_u(F, &W, x2, 0, d2 < 0.3); }
    F->nflops += W.nflops; F->nedges += W.nedges;
}
/* a depth-first search that can be advanced by steps (sxlu_ftran_part runs two at the same pace) */
typedef struct {
    const int *start, *len, *adj, *seed;
    int nseed, k, top, nout;
    unsigned char *mark; int *stk, *pos, *out;
    long edges;
} Dfs;
static void dfs_init(Dfs *D, SxWs *W, const int *start, const int *len, const int *adj, const int *seed, int nseed) {
    D->start = start; D->len = len; D->adj = adj; D->seed = seed; D->nseed = nseed; D->k = 0; D->top = -1; D->nout = 0;
    D->mark = W->dmark; D->stk = W->stack; D->pos = W->dfs_pos; D->out = W->wlist; D->edges = 0;
}
/* advances by about `steps` entries examined; returns 1 when the search is complete (the reached nodes in postorder in
 * out[0..nout-1]) */
static int dfs_step(Dfs *D, long steps) {
    unsigned char *mark = D->mark; int *stk = D->stk, *pos = D->pos;
    const int *adj = D->adj;
    long budget = D->edges + steps;
    for (;;) {
        if (D->top < 0) {
            while (D->k < D->nseed && mark[D->seed[D->k]]) D->k++;
            if (D->k == D->nseed) return 1;
            const int r0 = D->seed[D->k++];
            D->top = 0; stk[0] = r0; pos[0] = 0; mark[r0] = 1;
        }
        while (D->top >= 0) {
            const int r = stk[D->top];
            const int s = D->start[r], l = D->len[r];
            int p = pos[D->top];
            while (p < l && mark[adj[s + p]]) p++;
            D->edges += p - pos[D->top] + 1;
            if (p < l) {
                const int nx = adj[s + p];
                pos[D->top] = p + 1;
                mark[nx] = 1; D->top++; stk[D->top] = nx; pos[D->top] = 0;
            } else { D->out[D->nout++] = r; D->top--; }
            if (D->edges >= budget) return D->top < 0 && D->k == D->nseed;
        }
    }
}
/* the marks of an unfinished search out */
static void dfs_unmark(Dfs *D) {
    for (int k = 0; k < D->nout; k++) D->mark[D->out[k]] = 0;
    for (int t = 0; t <= D->top; t++) D->mark[D->stk[t]] = 0;
}
/* a solve wanted at a few positions only (the steepest-edge weights read B^{-1} rho_r where the entering column is
 * nonzero). L and R as in the full solve; then, in U x = y, x_i = (y_i - sum_{j after i} u_ij x_j) / u_ii depends only on
 * the entries of row i of U: the wanted positions and the rows they reach through the rows of U (a depth-first search
 * over the row lists, in postorder: each row after the rows it depends on) can be computed by dot products, and nothing
 * else. Which of the two is cheaper depends on the basis: the full solve reaches the rows that y's nonzeros reach through
 * the columns of U, the restricted one the rows the wanted positions reach through the rows of U (a fifth of the rows
 * against a few hundred on buildingenergy; the other way round on the long chains of a network basis). With a sparse y
 * both searches run at the same pace and the one that ends first is used (the work is at most twice the cheaper one's);
 * with a dense y the restricted solve is used. W2: the second search's scratch space. */
int sxlu_ftran_part(SxLU *F, SxWs *W2, SxVec *x, double expdens, const int *pat, int npat) {
    SxWs W0 = own_ws(F), *W = &W0;
    const int m = F->m;
    double *v = x->val;
    if (x->cnt < 0) sxv_reindex(x);
    const double ht = hyper_thr(); const int hyper = x->cnt < ht * m && expdens < ht;
    ftran_l(F, W, x, hyper);
    if (x->cnt >= 0) sxv_prune(x);
    if (F->nr && x->cnt < 0) sxv_reindex(x);
    if (F->nr) ftran_r(F, W, x);
    if (x->cnt >= 0) sxv_prune(x);
    const int nlr = x->cnt;
    Dfs A, B;       /* A: the rows the wanted positions depend on; B: the reach of y's nonzeros (the full solve) */
    dfs_init(&A, W, F->urs, F->url, F->uri, pat, npat);
    int full = 0;
    if (x->cnt >= 0 && x->cnt < ht * m) {
        dfs_init(&B, W2, F->ucs, F->ucl, F->uci, x->idx, x->cnt);
        for (;;) {
            if (dfs_step(&A, 256)) { dfs_unmark(&B); break; }
            if (dfs_step(&B, 256)) { dfs_unmark(&A); full = 1; break; }
        }
        W->nflops += B.edges;
    } else dfs_step(&A, (long)1 << 62);
    W->nflops += A.edges;
    if (full) {
        /* the full solve on B's reach (as ftran_u's hypersparse solve) */
        const int nr = B.nout;
        long fl = nr;
        for (int k = nr - 1; k >= 0; k--) {
            const int r = B.out[k];
            if (v[r] == 0) continue;
            const double xr = v[r] * F->urecip[r];
            v[r] = xr;
            fl += F->ucl[r];
            for (int p = F->ucs[r]; p < F->ucs[r] + F->ucl[r]; p++) v[F->uci[p]] -= F->ucv[p] * xr;
        }
        for (int k = 0; k < nr; k++) { x->idx[k] = B.out[k]; B.mark[B.out[k]] = 0; }
        x->cnt = nr; sxv_prune(x);
        W->nflops += fl;
    } else {
        const int nr = A.nout;
        long fl = nr;
        for (int k = 0; k < nr; k++) {
            const int r = A.out[k];
            double sum = v[r];
            const int e = F->urs[r] + F->url[r];
            for (int p = F->urs[r]; p < e; p++) sum -= F->urv[p] * v[F->uri[p]];
            fl += F->url[r];
            v[r] = sum * F->urecip[r];
        }
        /* the other entries of y out (the marks of the search tell the computed ones) */
        if (x->cnt >= 0) { for (int k = 0; k < x->cnt; k++) { const int i = x->idx[k]; if (!A.mark[i]) v[i] = 0; } }
        else for (int i = 0; i < m; i++) if (!A.mark[i]) v[i] = 0;
        for (int k = 0; k < nr; k++) { x->idx[k] = A.out[k]; A.mark[A.out[k]] = 0; }
        x->cnt = nr;
        W->nflops += fl;
    }
    F->nflops += W0.nflops; F->nedges += W0.nedges;
    return nlr;
}

/* y <- U^{-T} y */
static void btran_u(SxLU *F, SxWs *W, SxVec *y, int hyper) {
    double *v = y->val;
    if (hyper) {
        const int nr = dfs(W, y, F->urs, F->url, F->uri);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            if (v[r] == 0) continue;
            const double yr = v[r] * F->urecip[r];
            v[r] = yr;
            W->nflops += F->url[r];
            for (int p = F->urs[r]; p < F->urs[r] + F->url[r]; p++) v[F->uri[p]] -= F->urv[p] * yr;
        }
        for (int k = 0; k < nr; k++) y->idx[k] = W->wlist[k];
        y->cnt = nr; unmark(W, nr); sxv_prune(y);
        W->nflops += nr;
    } else {
        for (int k = 0; k < F->nord; k++) {
            const int r = F->uord[k];
            if (r < 0 || v[r] == 0) continue;
            const double yr = v[r] * F->urecip[r];
            v[r] = yr;
            W->nflops += F->url[r];
            for (int p = F->urs[r]; p < F->urs[r] + F->url[r]; p++) v[F->uri[p]] -= F->urv[p] * yr;
        }
        sxv_reindex(y);
        W->nflops += F->nord;
    }
}
/* y <- R_1^T ... R_k^T y */
static void btran_r(SxLU *F, SxWs *W, SxVec *y) {
    double *v = y->val;
    for (int e = F->nr - 1; e >= 0; e--) {
        const double yr = v[F->rpiv[e]];
        if (yr == 0) continue;
        for (int p = F->rstart[e]; p < F->rstart[e + 1]; p++) {
            const int j = F->ridx[p];
            if (y->cnt >= 0 && v[j] == 0) y->idx[y->cnt++] = j;
            v[j] -= F->rval[p] * yr;
            if (v[j] == 0) v[j] = 1e-300;
        }
    }
    W->nflops += F->rstart[F->nr];
}
/* y <- L^{-T} y */
static void btran_l(SxLU *F, SxWs *W, SxVec *y, int hyper) {
    double *v = y->val;
    if (hyper) {
        const int nr = dfs(W, y, F->lrstart, F->lrlen, F->lridx);
        for (int k = nr - 1; k >= 0; k--) {
            const int r = W->wlist[k];
            const double yr = v[r];
            if (yr == 0) continue;
            W->nflops += F->lrstart[r + 1] - F->lrstart[r];
            for (int p = F->lrstart[r]; p < F->lrstart[r + 1]; p++) v[F->lridx[p]] -= F->lrval[p] * yr;
        }
        for (int k = 0; k < nr; k++) y->idx[k] = W->wlist[k];
        y->cnt = nr; unmark(W, nr); sxv_prune(y);
        W->nflops += nr;
    } else {
        for (int k = 0; k < F->nlrorder; k++) {
            const int r = F->lrorder[k];
            const double yr = v[r];
            if (yr == 0) continue;
            W->nflops += F->lrstart[r + 1] - F->lrstart[r];
            for (int p = F->lrstart[r]; p < F->lrstart[r + 1]; p++) v[F->lridx[p]] -= F->lrval[p] * yr;
        }
        sxv_reindex(y);
        W->nflops += F->nlrorder;
    }
}
void sxlu_btran(SxLU *F, SxVec *y, double expdens, SxVec *upart) {
    SxWs W0 = own_ws(F), *W = &W0;
    const int m = F->m;
    if (y->cnt < 0) sxv_reindex(y);
    const double ht = hyper_thr(); const int hyper = y->cnt < ht * m && expdens < ht && hy_allow(F, 1);
    btran_u(F, W, y, hyper);
    if (upart) sxv_copy(upart, y);
    if (F->nr) btran_r(F, W, y);
    if (y->cnt >= 0) sxv_prune(y);
    btran_l(F, W, y, hyper && y->cnt >= 0 && y->cnt < ht * m);
    if (hyper) hy_note(F, 1, W0.nedges);
    F->nflops += W0.nflops; F->nedges += W0.nedges;
}

/* y <- B^{-T} y with the given scratch space, the sparse or the full sweeps chosen by the expected density alone (the
 * factorization's statistics of past solves neither read nor changed: the solve can run beside another one, and it
 * does not steer that one's choices); the counts in W */
void sxlu_btran_ws(SxLU *F, SxWs *W, SxVec *y, double expdens) {
    const int m = F->m;
    if (y->cnt < 0) sxv_reindex(y);
    const double ht = hyper_thr(); const int hyper = y->cnt < ht * m && expdens < ht;
    btran_u(F, W, y, hyper);
    if (F->nr) btran_r(F, W, y);
    if (y->cnt >= 0) sxv_prune(y);
    btran_l(F, W, y, hyper && y->cnt >= 0 && y->cnt < ht * m);
}

/* only the part of y <- B^{-T} y that the Forrest-Tomlin update needs (the solve with U^T, copied to upart), when the
 * whole row of B^{-1} is not wanted (the crossover's primal pushes); y is left partly solved */
void sxlu_btran_upart(SxLU *F, SxVec *y, double expdens, SxVec *upart) {
    SxWs W0 = own_ws(F), *W = &W0;
    const int m = F->m;
    if (y->cnt < 0) sxv_reindex(y);
    const double ht = hyper_thr(); const int hyper = y->cnt < ht * m && expdens < ht;
    btran_u(F, W, y, hyper);
    sxv_copy(upart, y);
    F->nflops += W0.nflops; F->nedges += W0.nedges;
}

/* ---------------------------------------------------------------- the Forrest-Tomlin update */
static void ucol_del(SxLU *F, int c, int i) {   /* entry of row i out of column c */
    const int s = F->ucs[c], e = s + F->ucl[c];
    for (int p = s; p < e; p++) if (F->uci[p] == i) { F->uci[p] = F->uci[e - 1]; F->ucv[p] = F->ucv[e - 1]; F->ucl[c]--; return; }
}
static void urow_del(SxLU *F, int r, int j) {
    const int s = F->urs[r], e = s + F->url[r];
    for (int p = s; p < e; p++) if (F->uri[p] == j) { F->uri[p] = F->uri[e - 1]; F->urv[p] = F->urv[e - 1]; F->url[r]--; return; }
}
static void ucompact(SxLU *F, int rows, int extra) {   /* rows: the row lists, else the column lists */
    const int m = F->m;
    int *s = rows ? F->urs : F->ucs, *l = rows ? F->url : F->ucl, *c = rows ? F->urc : F->ucc;
    int *idx = rows ? F->uri : F->uci; double *val = rows ? F->urv : F->ucv;
    long tot = 0; for (int i = 0; i < m; i++) tot += l[i] + 4;
    long cap = 2 * tot + extra + 64; const int oldcap = rows ? F->urcap : F->ucap; if (cap < oldcap) cap = oldcap;
    int *ni = xmalloc(sizeof(int) * cap); double *nv = xmalloc(sizeof(double) * cap);
    int e = 0;
    if (rows) for (int i = 0; i < m; i++) { memcpy(ni + e, idx + s[i], sizeof(int) * l[i]); memcpy(nv + e, val + s[i], sizeof(double) * l[i]); s[i] = e; c[i] = l[i] + 4; e += c[i]; }
    else for (int k = 0; k < F->nord; k++) {   /* (the columns in pivot order, without spare room: see the factorization) */
        const int i = F->uord[k]; if (i < 0) continue;
        memcpy(ni + e, idx + s[i], sizeof(int) * l[i]); memcpy(nv + e, val + s[i], sizeof(double) * l[i]); s[i] = e; c[i] = l[i]; e += c[i];
    }
    free(idx); free(val);
    if (rows) { F->uri = ni; F->urv = nv; F->urend = e; F->urcap = (int)cap; } else { F->uci = ni; F->ucv = nv; F->ucend = e; F->ucap = (int)cap; }
}
static void urow_add(SxLU *F, int r, int j, double v) {
    if (F->url[r] == F->urc[r]) {
        const int nc = 2 * F->urc[r] + 4;
        if ((long)F->urend + nc > F->urcap) ucompact(F, 1, nc);
        memmove(F->uri + F->urend, F->uri + F->urs[r], sizeof(int) * F->url[r]); memmove(F->urv + F->urend, F->urv + F->urs[r], sizeof(double) * F->url[r]);
        F->urs[r] = F->urend; F->urc[r] = nc; F->urend += nc;
    }
    const int p = F->urs[r] + F->url[r]++;
    F->uri[p] = j; F->urv[p] = v;
}
int sxlu_update(SxLU *F, int r, SxVec *spike, SxVec *upart, double alpha) {
    const int m = F->m;
    const double urr = F->udiag[r];
    /* the row eta: eta_j = -u_rr z_j (z: the partial btran of e_r), j != r */
    if (F->nr == F->rcapn) { F->rcapn *= 2; F->rpiv = xrealloc(F->rpiv, sizeof(int) * F->rcapn); F->rstart = xrealloc(F->rstart, sizeof(int) * (F->rcapn + 1)); }
    if (upart->cnt < 0) sxv_reindex(upart);
    if (spike->cnt < 0) sxv_reindex(spike);
    int p0 = F->rstart[F->nr];
    if (p0 + upart->cnt + 1 > F->rcap) { F->rcap = 2 * (p0 + upart->cnt) + 1024; F->ridx = xrealloc(F->ridx, sizeof(int) * F->rcap); F->rval = xrealloc(F->rval, sizeof(double) * F->rcap);
        F->rnext = xrealloc(F->rnext, sizeof(int) * F->rcap); F->reta = xrealloc(F->reta, sizeof(int) * F->rcap); }
    double newd = spike->val[r];
    int p = p0;
    for (int k = 0; k < upart->cnt; k++) {
        const int j = upart->idx[k];
        if (j == r) continue;
        const double z = upart->val[j];
        if (fabs(z) < SX_TINY) continue;
        const double eta = -urr * z;
        F->ridx[p] = j; F->rval[p] = eta; F->reta[p] = F->nr; F->rnext[p] = F->rthead[j]; F->rthead[j] = p; F->rtcnt[j]++; p++;
        newd -= eta * spike->val[j];
    }
    F->rpiv[F->nr] = r; F->rstart[F->nr + 1] = p; F->nr++;
    F->nnz_r += p - p0;
    /* U: column r and row r out */
    F->nnz_u -= F->ucl[r] + F->url[r];
    for (int q = F->ucs[r]; q < F->ucs[r] + F->ucl[r]; q++) urow_del(F, F->uci[q], r);
    F->ucl[r] = 0;
    for (int q = F->urs[r]; q < F->urs[r] + F->url[r]; q++) ucol_del(F, F->uri[q], r);
    F->url[r] = 0;
    /* the spike as the new column r (the last pivot) */
    int ns = 0;
    for (int k = 0; k < spike->cnt; k++) { const int i = spike->idx[k]; if (i != r && fabs(spike->val[i]) >= SX_TINY) ns++; }
    if (F->ucc[r] < ns) {
        const int nc = 2 * ns + 4;
        if ((long)F->ucend + nc > F->ucap) ucompact(F, 0, nc);
        F->ucs[r] = F->ucend; F->ucc[r] = nc; F->ucend += nc;
    }
    for (int k = 0; k < spike->cnt; k++) {
        const int i = spike->idx[k];
        const double s = spike->val[i];
        if (i == r || fabs(s) < SX_TINY) continue;
        const int q = F->ucs[r] + F->ucl[r]++;
        F->uci[q] = i; F->ucv[q] = s;
        urow_add(F, i, r, s);
    }
    F->udiag[r] = newd; F->urecip[r] = 1.0 / newd;
    F->nnz_u += F->ucl[r];
    /* the order: r moves to the end */
    F->uord[F->uordpos[r]] = -1;
    if (F->nord == F->ordcap) {   /* compact the order */
        int e = 0; for (int k = 0; k < F->nord; k++) if (F->uord[k] >= 0) { F->uord[e] = F->uord[k]; F->uordpos[F->uord[e]] = e; e++; }
        F->nord = e;
        if (F->nord == F->ordcap) { F->ordcap *= 2; F->uord = xrealloc(F->uord, sizeof(int) * F->ordcap); }
    }
    F->uord[F->nord] = r; F->uordpos[r] = F->nord; F->nord++;
    if (F->uzpos[r] >= 0) F->uz[F->uzpos[r]] = -1;
    if (F->nuz == F->uzcap) {
        int e = 0; for (int k = 0; k < F->nuz; k++) if (F->uz[k] >= 0) { F->uz[e] = F->uz[k]; F->uzpos[F->uz[e]] = e; e++; }
        F->nuz = e;
        if (F->nuz == F->uzcap) { F->uzcap *= 2; F->uz = xrealloc(F->uz, sizeof(int) * F->uzcap); }
    }
    F->uz[F->nuz] = r; F->uzpos[r] = F->nuz; F->nuz++;
    F->nupd++;
    (void)m;
    /* stability: the new diagonal against u_rr alpha_rq */
    const double pred = urr * alpha;
    if (!(fabs(newd) > 0) || fabs(newd - pred) > 1e-8 * (1 + fabs(pred))) return 1;
    return 0;
}
