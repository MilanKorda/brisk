#include <time.h>
/*
 * Sparse Cholesky factorization of the Schur complement.
 *
 * The envelope (RCM profile) path only pays when the pattern is banded. Problems with
 * many small blocks - finite-element models, LP-heavy models, and above all the output of
 * the chordal decomposition - have patterns whose fill under a minimum-degree ordering is
 * far smaller than their profile. On the 793-bus AC-OPF relaxation the decomposition
 * leaves a 13158 x 13158 complement that the dense path factors in 7.6e11 flops; with a
 * fill-reducing ordering the same matrix costs orders of magnitude less.
 *
 * The ordering is minimum degree on the explicit elimination graph, which produces the
 * pattern of L as a by-product (column j of L is the clique of j at elimination time), so
 * the symbolic phase also yields the exact factorization flop count
 * (sum_j |L(:,j)|^2), which the route selection uses instead of an estimate.
 *
 * The numeric phase is a left-looking scalar factorization with a dense accumulator and
 * the usual linked lists of pending column updates.
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static double omp_get_wtime(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int __attribute__((unused)) omp_in_parallel(void) { return 0; }
#endif

struct SChol {
    long uid;                        /* unique per analysis (caches keyed on it) */
    int m;
    int *perm, *iperm;      /* perm[new] = old */
    int *cp;                /* column pointers of L (size m+1), in permuted order */
    int *ci;                /* row indices of L, sorted, diagonal first (during the analysis only) */

    double flops;           /* exact factorization cost */
    size_t nnz;
    /* workspace */
    double *acc;
    int *head, *next, *mark, *stack;
    double *acc2;            /* block-solve workspace */
    /* supernodal layout: columns [ss[k], ss[k+1]) share the row pattern of column ss[k];
     * supernode k is a dense (pnr[k] x width) column-major panel at pbase[k]          */
    int ns, *ss, *snof, *prow, *prp, *pnr;
    size_t *pbase, pansz;
    double *pan, *pmat;
    int inplace;             /* 4.22: factor over the assembled values (pan == pmat) */
    int *relind, *relidx, *snhead, *snnext, *snptr;
    double *upd;
    size_t updcap;
    uint64_t *hkey;          /* assembly cache: (row, col) -> panel offset */
    size_t *hval, hmask;
    double *wrk;             /* solve workspace */
    /* the matrix's own pattern (lower triangle, permuted columns): M x without the fill
     * entries of L and without per-entry branches; values live in pmat at moff */
    int *mcp, *mri;
    size_t *moff;
    double tinypiv;                  /* > 0: pivots below tinypiv * original diagonal are
                                      * replaced by a huge value (that direction is dropped) */
    int ntiny;
    int tinydag;                     /* 18 October 2026 (LP/SOCP): with tinypiv, large nodes by the tile task graph too
                                      * (fnode_dag, the tiny-pivot rule in its diagonal tiles); set by the LP method only */
    double dynpiv;                   /* > 0 (signed factorization): a pivot of the wrong sign is replaced by
                                      * sign * dynpiv and counted, instead of failing */
    int ndyn;
    double *mval;                    /* compact copy of the M values (valid if mval_ok) */
    size_t *mrq, *mdg; int mdirect;  /* 17 October 2026: the runs' and the diagonal's offsets in pmat (mdirect: no mval copy) */
    int an_hit; size_t an_tot; double an_fl, an_dense; unsigned an_set;   /* 17 October 2026: the analysis's caps (an_over) and settings */
    int an_hit_other, an_own, an_amdcmp; size_t an_amd_tot, an_own_tot; double an_amd_fl, an_own_fl;
    int mval_ok;
    int *mrun, nrun;                 /* runs of consecutive rows in the off-diagonal pattern: (q0, i0, len) triples (4.20) */
    int *mrp;                        /* per column: first run */
    int run_state;                   /* 0 not analysed, 1 use runs, -1 entry loop */
    double *t2; size_t t2cap;        /* schol_solve2 workspace */
    long fgen;                       /* 5.12: factorizations so far */
    double *tinv; long tinv_gen; int tinv_kt, tinv_kt2;   /* (L_TT L_TT')^{-1} of the top (schol_inverse), for factor tinv_gen */
    double *tl; long tl_gen; int tl_kt, tl_kt2;           /* L_TT, dense lower triangle (the top's solves), for factor tl_gen */
    /* quasi-definite mode: M = L S L' with S = diag(+-1) prescribed per (permuted) column;
     * NULL = ordinary Cholesky */
    signed char *sgn;
    double *sw;              /* scaled copy for the signed updates */
    size_t swcap;
    /* 4.24: canonical (thread-count independent) factorization: static update lists per
     * target supernode (sources in increasing order), supernodal tree, work estimates */
    int *ul_ptr, *ul_j, *ul_p, *ul_a;
    int mv_np, *mv_c; size_t *mv_zo; double *mv_z;
    double *mva, *mv_za;     /* 4.27: |x|, |M||x| accumulators of schol_mv_abs, and its chunk tails */
    double *mv2w, *mv_z2;    /* the second column's vectors and chunk tails of schol_mv2 */
    int *sch_ptr, *sch_idx;  /* 4.24: children of each supernode, ascending */   /* 4.24: chunked M x (fixed chunks) */
    int *spar, *sfd;         /* parent supernode (-1: root), first descendant (postorder) */
    double *swk, *ssub;      /* node work estimate, subtree work */
    int fail;
};

static const double DONE = 1.0, DZERO = 0.0;
static const int IONE = 1;

static void *sx(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }

struct SAdj { int *a; int n, cap; };
typedef struct SAdj SAdj;
static void sadj_add(SAdj *A, int x) {
    if (A->n == A->cap) { A->cap = A->cap ? 2 * A->cap : 8; A->a = realloc(A->a, sizeof(int) * A->cap); }
    A->a[A->n++] = x;
}

/* Minimum-degree ordering of the pattern given as adjacency lists (upper/lower symmetric,
 * no diagonal). Returns the permutation and the per-column patterns of L. Fill is capped:
 * if it exceeds `fillcap` entries the routine gives up (the caller falls back).        */
static SChol *schol_analyze(int m, SAdj *adj, size_t fillcap);
/* (sparsechol-dev) ascending sort of n distinct integers in [0, m): insertion below 48 entries, else radix by bytes
 * (the digits that m needs) with the buffer t of n entries; the same order as qsort with cmp_i */
static void sc_isort(int *a, int n, int *t, int m) {
    if (n < 48) {
        for (int i = 1; i < n; i++) { const int v = a[i]; int q = i - 1; while (q >= 0 && a[q] > v) { a[q + 1] = a[q]; q--; } a[q + 1] = v; }
        return;
    }
    int *src = a, *dst = t;
    for (int sh = 0; sh < 32 && (m - 1) >> sh; sh += 8) {
        int cnt[257] = { 0 };
        for (int i = 0; i < n; i++) cnt[((src[i] >> sh) & 255) + 1]++;
        for (int d = 0; d < 256; d++) cnt[d + 1] += cnt[d];
        for (int i = 0; i < n; i++) dst[cnt[(src[i] >> sh) & 255]++] = src[i];
        int *x = src; src = dst; dst = x;
    }
    if (src != a) memcpy(a, src, sizeof(int) * (size_t)n);
}
static int cmp_ll(const void *a, const void *b) { const long long x = *(const long long *)a, y = *(const long long *)b; return (x > y) - (x < y); }

/* optional cap on the factorization flops (sum of squared column counts): orderings stop
 * as soon as the factor cannot beat the dense path, instead of finishing the analysis
 * only to be rejected (e_moment_stable_17: 0.8 s per analysis) */
static double g_flopcap = 0;
void schol_set_flopcap(double cap) { g_flopcap = cap; }
/* 17 October 2026: what the caps did in an analysis (schol_analyze_sym): whether an attempt stopped at one, and the
 * largest fill and flops an attempt reached; an analysis that met no cap is the one any caps above those give
 * (schol_analysis_fits) */
static int g_an_hit; static size_t g_an_tot, g_an_fillcap; static double g_an_fl;
/* ... by phase (g_an_ph: 1 an AMD attempt, 2 the own orderings after a failed AMD attempt, 0 the rest): a caller's cap
 * met outside the AMD attempts, where the AMD attempts stopped, and what the own orderings reached */
static int g_an_ph, g_an_hit_other, g_an_own; static size_t g_an_amd_tot, g_an_own_tot; static double g_an_amd_fl, g_an_own_fl;
/* ... and a limit of the analysis's own (not the caller's): the second ordering stops once its flops pass those of the
 * first, which it then cannot beat (schol_analyze) */
static double g_flopcap_own = 0;
static inline int an_over(size_t total, double flops, size_t fillcap) {
    if (total > g_an_tot) g_an_tot = total;
    if (flops > g_an_fl) g_an_fl = flops;
    if (g_an_ph == 2) { if (total > g_an_own_tot) g_an_own_tot = total; if (flops > g_an_own_fl) g_an_own_fl = flops; }
    if (total > g_an_fillcap || (g_flopcap > 0 && flops > g_flopcap)) {     /* (the caller's caps) */
        g_an_hit = 1;
        if (g_an_ph != 1) g_an_hit_other = 1;
        else { if (total < g_an_amd_tot) g_an_amd_tot = total; if (flops < g_an_amd_fl) g_an_amd_fl = flops; }   /* (where it stopped) */
    }
    return total > fillcap || (g_flopcap > 0 && flops > g_flopcap) || (g_flopcap_own > 0 && flops > g_flopcap_own);
}
/* the ordering mode of the next analyses (BRISK_AMD: 0 own, 1 automatic, 2 AMD alone); returns the previous one (-1: not yet set) */
static int g_amd;
int schol_set_amd(int mode) { const int old = g_amd; g_amd = mode; return old; }
/* AMD's dense-row parameter for the next analyses (0: its default, 10): rows with more than this times sqrt(n) entries are ordered last */
static double g_amd_dense = 0;
/* an ordering given by the caller for the next analysis in AMD mode (perm[new] = old; NULL: AMD) */
static const int *g_force_perm = NULL;
static int g_peo_try = 0, g_zf = 0, g_an_peo = 0;   /* (the chordal ordering's switch, its check mode, its use: see mcs_order) */
void schol_set_perm(const int *perm) { g_force_perm = perm; }
void schol_set_amd_dense(double d) { g_amd_dense = d; }

/* Minimum degree on supervariables. Nodes with identical closed neighbourhoods are
 * indistinguishable and are eliminated together (the Gram entries of one SOS multiplier,
 * the kernel coordinates of one SOS constraint: the Schur pattern of a moment-SOS program
 * has a few dozen such classes among thousands of constraints). Weighted minimum degree
 * runs on the compressed graph, so the analysis costs O(pattern) instead of the O(flops)
 * of the explicit elimination graph. Returns 2 when the compression is too weak (the
 * caller uses the node-by-node ordering), 1 when the fill cap is exceeded. */
static int schol_order_super(int m, SAdj *adj, size_t fillcap, int **perm_out, int **cpat_len,
                             int ***cpat, double *flops_out, size_t *nnz_out) {
    uint64_t *h = sx(sizeof(uint64_t) * m);
    for (int i = 0; i < m; i++) {
        uint64_t a = (uint64_t)i * 0x9e3779b97f4a7c15ULL;
        a ^= a >> 29;
        uint64_t hs = a * 0xbf58476d1ce4e5b9ULL;
        for (int q = 0; q < adj[i].n; q++) {
            uint64_t b = (uint64_t)adj[i].a[q] * 0x9e3779b97f4a7c15ULL;
            b ^= b >> 29;
            hs += b * 0xbf58476d1ce4e5b9ULL;
        }
        h[i] = hs ^ ((uint64_t)adj[i].n << 40);
    }
    /* group by hash: sort node ids by (hash) */
    int *ord = sx(sizeof(int) * m);
    for (int i = 0; i < m; i++) ord[i] = i;
    /* insertion into buckets by hash via sorting */
    {
        /* simple merge sort on keys */
        int *tmp = sx(sizeof(int) * m);
        for (int w = 1; w < m; w *= 2)
            for (int lo = 0; lo < m; lo += 2 * w) {
                int mid = lo + w < m ? lo + w : m, hi = lo + 2 * w < m ? lo + 2 * w : m;
                int a = lo, b = mid, k = lo;
                while (a < mid && b < hi) tmp[k++] = h[ord[a]] <= h[ord[b]] ? ord[a++] : ord[b++];
                while (a < mid) tmp[k++] = ord[a++];
                while (b < hi) tmp[k++] = ord[b++];
                memcpy(ord + lo, tmp + lo, sizeof(int) * (hi - lo));
            }
        free(tmp);
    }
    int *sv = sx(sizeof(int) * m);
    for (int i = 0; i < m; i++) sv[i] = -1;
    char *mk = sx(m);
    int nsv = 0;
    int *rep = sx(sizeof(int) * m);
    for (int a = 0; a < m; ) {
        int b = a + 1;
        while (b < m && h[ord[b]] == h[ord[a]]) b++;
        /* verify candidates of equal hash exactly: closed neighbourhoods equal */
        for (int p = a; p < b; p++) {
            int i = ord[p];
            if (sv[i] >= 0) continue;
            sv[i] = nsv;
            rep[nsv] = i;
            for (int q = 0; q < adj[i].n; q++) mk[adj[i].a[q]] = 1;
            mk[i] = 1;
            for (int p2 = p + 1; p2 < b; p2++) {
                int j = ord[p2];
                if (sv[j] >= 0 || adj[j].n != adj[i].n || !mk[j]) continue;
                int same = 1;
                for (int q = 0; q < adj[j].n && same; q++) if (!mk[adj[j].a[q]]) same = 0;
                if (same) sv[j] = nsv;
            }
            for (int q = 0; q < adj[i].n; q++) mk[adj[i].a[q]] = 0;
            mk[i] = 0;
            nsv++;
        }
        a = b;
    }
    free(h); free(ord);
    if (nsv > m / 2) { free(sv); free(mk); free(rep); return 2; }
    /* members of each supervariable */
    int *wt = sx(sizeof(int) * nsv), *mptr = sx(sizeof(int) * (nsv + 1)), *mem = sx(sizeof(int) * m);
    for (int i = 0; i < m; i++) wt[sv[i]]++;
    for (int k = 0; k < nsv; k++) mptr[k + 1] = mptr[k] + wt[k];
    {
        int *fill = sx(sizeof(int) * nsv);
        for (int i = 0; i < m; i++) { int k = sv[i]; mem[mptr[k] + fill[k]++] = i; }
        free(fill);
    }
    /* compressed adjacency */
    SAdj *ca = sx(sizeof(SAdj) * nsv);
    char *cm = sx(nsv);
    for (int k = 0; k < nsv; k++) {
        int r = rep[k];
        for (int q = 0; q < adj[r].n; q++) {
            int t = sv[adj[r].a[q]];
            if (t != k && !cm[t]) { cm[t] = 1; sadj_add(&ca[k], t); }
        }
        for (int q = 0; q < ca[k].n; q++) cm[ca[k].a[q]] = 0;
    }
    free(rep); free(mk);
    char *gone = sx(nsv);
    double *deg = sx(sizeof(double) * nsv);
    for (int k = 0; k < nsv; k++) { double d = wt[k] - 1; for (int q = 0; q < ca[k].n; q++) d += wt[ca[k].a[q]]; deg[k] = d; }
    int *perm = sx(sizeof(int) * m), *plen = sx(sizeof(int) * m), **pat = sx(sizeof(int *) * m);
    int *nbuf = sx(sizeof(int) * nsv), *ebuf = sx(sizeof(int) * m);
    size_t total = 0;
    double flops = 0, work = 0;
    int step = 0, rc = 0;
    for (int it = 0; it < nsv; it++) {
        int best = -1;
        for (int k = 0; k < nsv; k++) if (!gone[k] && (best < 0 || deg[k] < deg[best])) best = k;
        work += nsv;
        if (work > 1e9) { rc = 1; break; }          /* weakly compressed, dense graph */
        /* live neighbours and their members */
        int nn = 0, ne = 0;
        for (int q = 0; q < ca[best].n; q++) {
            int t = ca[best].a[q];
            if (gone[t]) continue;
            nbuf[nn++] = t;
            for (int z = mptr[t]; z < mptr[t + 1]; z++) ebuf[ne++] = mem[z];
        }
        const int w = wt[best];
        for (int c = 0; c < w; c++) {
            const int len = (w - c) + ne;
            total += (size_t)len;
            flops += (double)len * len;
            if (an_over(total, flops, fillcap)) { rc = 1; break; }
            int *pp = malloc(sizeof(int) * len);
            for (int c2 = c; c2 < w; c2++) pp[c2 - c] = mem[mptr[best] + c2];
            memcpy(pp + (w - c), ebuf, sizeof(int) * ne);
            perm[step] = mem[mptr[best] + c];
            pat[step] = pp;
            plen[step] = len;
            step++;
        }
        if (rc) break;
        gone[best] = 1;
        /* the neighbours become a clique */
        for (int a = 0; a < nn; a++) {
            int u = nbuf[a];
            work += ca[u].n + nn;
            for (int q = 0; q < ca[u].n; q++) cm[ca[u].a[q]] = 1;
            for (int b = 0; b < nn; b++) {
                int v = nbuf[b];
                if (v != u && !cm[v]) { sadj_add(&ca[u], v); cm[v] = 1; }
            }
            for (int q = 0; q < ca[u].n; q++) cm[ca[u].a[q]] = 0;
            double d = wt[u] - 1;
            for (int q = 0; q < ca[u].n; q++) if (!gone[ca[u].a[q]]) d += wt[ca[u].a[q]];
            deg[u] = d;
        }
    }
    for (int k = 0; k < nsv; k++) free(ca[k].a);
    free(ca); free(cm); free(gone); free(deg); free(nbuf); free(ebuf);
    free(wt); free(mptr); free(mem); free(sv);
    if (rc || step != m) {
        for (int i = 0; i < step; i++) free(pat[i]);
        free(perm); free(plen); free(pat);
        return 1;
    }
    *perm_out = perm;
    *cpat = pat;
    *cpat_len = plen;
    *flops_out = flops;
    *nnz_out = total;
    return 0;
}

static int schol_order(int m, SAdj *adj, size_t fillcap, int **perm_out, int **cpat_len,
                       int ***cpat, double *flops_out, size_t *nnz_out) {
    char *gone = sx(m);
    int *deg = sx(sizeof(int) * m);
    char *mark = sx(m);
    int *perm = sx(sizeof(int) * m);
    int **pat = sx(sizeof(int *) * m);
    int *plen = sx(sizeof(int) * m);
    int *buf = sx(sizeof(int) * m);
    size_t total = 0;
    double flops = 0, work = 0;
    /* Work budget: minimum degree on an explicit elimination graph can blow up on
     * irregular patterns with heavy fill (a random-coupled 9000-constraint instance ran
     * for minutes). Give up and let the caller fall back instead of hanging.         */
    const double workcap = 5e8;
    /* degree buckets */
    for (int i = 0; i < m; i++) deg[i] = adj[i].n;
    /* degree buckets: picking the minimum-degree node by scanning all nodes is O(m^2)
     * (0.35 s at m = 7200, seconds at m = 20000); the bucket queue makes it near-linear */
    int *bnext = sx(sizeof(int) * m), *bprev = sx(sizeof(int) * m), *bhead = sx(sizeof(int) * (m + 1));
    for (int i = 0; i <= m; i++) bhead[i] = -1;
    for (int i = 0; i < m; i++) {
        int d = deg[i] < m ? deg[i] : m;
        bnext[i] = bhead[d];
        bprev[i] = -1;
        if (bhead[d] >= 0) bprev[bhead[d]] = i;
        bhead[d] = i;
    }
    int *bin = sx(sizeof(int) * m);
    for (int i = 0; i < m; i++) bin[i] = deg[i] < m ? deg[i] : m;
    #define BUCK_DROP(x) do { int d_ = bin[x]; \
        if (bprev[x] >= 0) bnext[bprev[x]] = bnext[x]; else bhead[d_] = bnext[x]; \
        if (bnext[x] >= 0) bprev[bnext[x]] = bprev[x]; } while (0)
    #define BUCK_PUT(x, d) do { int d_ = (d) < m ? (d) : m; bin[x] = d_; \
        bnext[x] = bhead[d_]; bprev[x] = -1; \
        if (bhead[d_] >= 0) { bprev[bhead[d_]] = x; } \
        bhead[d_] = x; } while (0)
    int mind = 0;
    for (int step = 0; step < m; step++) {
        int best = -1;
        while (mind <= m && bhead[mind] < 0) mind++;
        /* 4.20: once the remaining graph is (nearly) complete, the rest of the elimination is
         * one dense block: its patterns follow directly, without the explicit clique updates
         * that made the analysis cost O(flops) (e_moment_stable_17: 0.9 s, most of it on the
         * final dense part) */
        {
            const int r = m - step;
            if (r > 64 && mind <= m && (double)mind >= 0.9 * (r - 1)) {
                int *rem = malloc(sizeof(int) * r), nr_ = 0;
                for (int i = 0; i < m; i++) if (!gone[i]) rem[nr_++] = i;
                double fl2 = 0; size_t tot2 = 0;
                for (int t = 0; t < nr_; t++) { const double len = nr_ - t; fl2 += len * len; tot2 += (size_t)(nr_ - t); }
                if (an_over(total + tot2, flops + fl2, fillcap)) {
                    free(rem);
                    for (int i = 0; i < step; i++) free(pat[i]);
                    free(gone); free(deg); free(mark); free(perm); free(pat); free(plen); free(buf);
                    free(bnext); free(bprev); free(bhead); free(bin);
                    return 1;
                }
                for (int t = 0; t < nr_; t++) {
                    const int len = nr_ - t;
                    pat[step + t] = malloc(sizeof(int) * len);
                    memcpy(pat[step + t], rem + t, sizeof(int) * len);
                    plen[step + t] = len;
                    perm[step + t] = rem[t];
                }
                total += tot2; flops += fl2;
                free(rem);
                step = m;
                break;
            }
        }
        if (mind > m) {
            for (int d = 0; d <= m && best < 0; d++) if (bhead[d] >= 0) { mind = d; }
            if (mind > m) break;
        }
        best = bhead[mind];
        if (best < 0) break;
        BUCK_DROP(best);
        if (mind > 0) mind--;                       /* degrees can only drop by one step */
        /* clique of `best`: its live neighbours */
        int nb = 0;
        for (int q = 0; q < adj[best].n; q++) {
            int w = adj[best].a[q];
            if (!gone[w] && !mark[w]) { mark[w] = 1; buf[nb++] = w; }
        }
        for (int q = 0; q < nb; q++) mark[buf[q]] = 0;
        total += (size_t)nb + 1;
        flops += (double)(nb + 1) * (nb + 1);
        work += (double)nb * nb;
        if (an_over(total, flops, fillcap) || work > workcap) {
            for (int i = 0; i < m; i++) free(pat[i]);
            free(gone); free(deg); free(mark); free(perm); free(pat); free(plen); free(buf);
            free(bnext); free(bprev); free(bhead); free(bin);
            return 1;
        }
        pat[step] = malloc(sizeof(int) * (nb + 1));
        pat[step][0] = best;
        memcpy(pat[step] + 1, buf, sizeof(int) * nb);
        plen[step] = nb + 1;
        perm[step] = best;
        /* fill: connect the clique */
        for (int a = 0; a < nb; a++) {
            int u = buf[a];
            for (int q = 0; q < adj[u].n; q++) if (!gone[adj[u].a[q]]) mark[adj[u].a[q]] = 1;
            mark[u] = 1;
            for (int b = 0; b < nb; b++) {
                int v = buf[b];
                if (mark[v]) continue;
                sadj_add(&adj[u], v);
                deg[u]++;
                mark[v] = 1;
            }
            for (int q = 0; q < adj[u].n; q++) mark[adj[u].a[q]] = 0;
            mark[u] = 0;
            deg[u]--;                                   /* `best` leaves the graph */
            if (deg[u] < 0) deg[u] = 0;
            BUCK_DROP(u);
            BUCK_PUT(u, deg[u]);
            if (deg[u] < mind) mind = deg[u];
        }
        gone[best] = 1;
    }
    free(gone); free(deg); free(mark); free(buf);
    free(bnext); free(bprev); free(bhead); free(bin);
    #undef BUCK_DROP
    #undef BUCK_PUT
    *perm_out = perm;
    *cpat = pat;
    *cpat_len = plen;
    *flops_out = flops;
    *nnz_out = total;
    return 0;
}


/* Build the factorization structure from a pattern given as adjacency lists.
 * Returns NULL if the ordering gives up (fill cap) or the matrix is trivial.        */
/* entry point: pattern given as per-node neighbour arrays (no diagonal) */
/* panel offset of entry (r, c) (original indices), or SIZE_MAX outside the pattern */
/* 4.22: by binary search in the supernode's (sorted) row list; the hash table it replaces
 * took 3 slots x 16 bytes per panel entry (4.3 GB for the case9241 Schur pattern) */
static inline size_t sc_off(const SChol *S, int i, int j) {
    if (i < j) { const int t = i; i = j; j = t; }
    const int k = S->snof[j], cc = j - S->ss[k], nr = S->pnr[k];
    const int *rw = S->prow + S->prp[k];
    int lo = cc, hi = nr - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1, v = rw[mid];
        if (v < i) lo = mid + 1; else if (v > i) hi = mid - 1;
        else return S->pbase[k] + (size_t)cc * nr + mid;
    }
    return SIZE_MAX;
}
size_t schol_offset(const SChol *S, size_t r, size_t c) {
    return sc_off(S, S->iperm[r], S->iperm[c]);
}
static void sc_ensure(const SChol *Sc) {
    SChol *S = (SChol *)Sc;
    if (!S->pmat) S->pmat = sx(sizeof(double) * (S->pansz ? S->pansz : 1));
}
/* 4.22: the numeric factor's storage back to the system (recreated by the next factorization) */
size_t schol_pansz(const SChol *S) { return S->pansz; }
void schol_release_factor(SChol *S) { if (S->pan != S->pmat) free(S->pan); S->pan = NULL; }
/* (in place, the products read the compact copy mval, gathered through moff: not once moff is gone, sc_drop_moff) */
void schol_set_inplace(SChol *S, int on) { S->inplace = on && S->moff; }
/* 18 October 2026: when the products read the factor's input directly (mdirect) and the factor has its own storage,
 * nothing reads M's offsets after the first factorization: freed before that storage is allocated (8 bytes per
 * entry of M; broyden25 155 MB) */
static void sc_drop_moff(SChol *S) { if (S->mdirect && !S->inplace && S->moff) { free(S->moff); S->moff = NULL; } }
double *schol_values(SChol *S) { sc_ensure(S); S->mval_ok = 0; return S->pmat; }
/* prescribe the signs of the pivots (original indexing): +1 positive definite part, -1
 * negative definite part of a quasi-definite matrix [H A; A' -D]. Any symmetric ordering
 * of such a matrix has an L S L' factorization without pivoting. */
void schol_set_signs(SChol *S, const signed char *sgn_orig) {
    if (!sgn_orig) { free(S->sgn); S->sgn = NULL; return; }
    if (!S->sgn) S->sgn = malloc(S->m + 1);
    for (int j = 0; j < S->m; j++) S->sgn[j] = sgn_orig[S->perm[j]] < 0 ? -1 : 1;
}

static double g_amalg = 0.0;
double schol_set_amalg(double f) { const double o = g_amalg; g_amalg = f; return o; }
static unsigned an_settings(void);
static int an_amd_competes(int m);
/* 17 October 2026: the ordering and the symbolic factorization alone (an estimate of the cost needs no more);
 * schol_attach_pattern adds the pattern of M and its offsets, which the numeric work needs */
SChol *schol_analyze_sym(int m, int *deg, int **nbr, size_t fillcap) {
    SAdj *adj = sx(sizeof(SAdj) * m);
    for (int i = 0; i < m; i++) {
        adj[i].n = adj[i].cap = deg[i];
        adj[i].a = malloc(sizeof(int) * (deg[i] ? deg[i] : 1));
        memcpy(adj[i].a, nbr[i], sizeof(int) * deg[i]);
    }
    SChol *S = schol_analyze(m, adj, fillcap);
    if (S) { S->an_hit = g_an_hit; S->an_tot = g_an_tot; S->an_fl = g_an_fl; S->an_dense = g_amd_dense; S->an_set = an_settings();
             S->an_hit_other = g_an_hit_other; S->an_own = g_an_own; S->an_amd_tot = g_an_amd_tot; S->an_amd_fl = g_an_amd_fl;
             S->an_own_tot = g_an_own_tot; S->an_own_fl = g_an_own_fl; S->an_amdcmp = an_amd_competes(m); }
    for (int i = 0; i < m; i++) free(adj[i].a);
    free(adj);
    return S;
}
/* the analysis S (any caps) is the one schol_analyze_sym gives under (fillcap, flopcap; 0: none) and the present
 * settings: none of its attempts met a cap and none reached these */
int schol_analysis_fits(const SChol *S, size_t fillcap, double flopcap) {
    if (S && getenv("BRISK_ICDBG")) printf("   [fits: hit %d (other %d), fill %zu / %zu, flops %.3g / %.3g, settings %u / %u, dense %g / %g; own %d (%zu, %.3g) against AMD's stop (%zu, %.3g), competes %d]\n", S->an_hit, S->an_hit_other, S->an_tot, fillcap, S->an_fl, flopcap, S->an_set, an_settings(), S->an_dense, g_amd_dense, S->an_own, S->an_own_tot, S->an_own_fl, S->an_amd_tot, S->an_amd_fl, S->an_amdcmp);
    /* an AMD attempt stopped at a cap and the own ordering, which ran within every cap, was kept: with caps above
     * these, AMD fails again (the own ordering runs as it did) or completes with at least the fill and flops it had
     * reached; the own ordering then runs against it (capped at its flops and the caller's fill) as it did, and is
     * kept, with fewer flops. So the result is the same when the own ordering ran within the caps and its flops are
     * below AMD's at its stop (below the order where AMD runs without a competitor) */
    if (S && S->an_hit && !S->an_hit_other && S->an_own && S->an_amdcmp && S->an_set == an_settings() && S->an_dense == g_amd_dense
        && !getenv("BRISK_OWNFILL") && S->an_own_tot <= fillcap && (flopcap <= 0 || S->an_own_fl <= flopcap) && S->an_own_fl < S->an_amd_fl) return 1;
    return S && !S->an_hit && S->an_tot <= fillcap && (flopcap <= 0 || S->an_fl <= flopcap)
             && S->an_set == an_settings() && S->an_dense == g_amd_dense;
}
/* a key of the adjacency lists in their order (the analysis reads them in that order) */
uint64_t schol_adj_key(int m, const int *deg, int *const *nbr) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t)m;
    for (int i = 0; i < m; i++) {
        h = (h ^ (uint64_t)deg[i]) * 1099511628211ull;
        for (int q = 0; q < deg[i]; q++) h = (h ^ (uint64_t)nbr[i][q]) * 1099511628211ull;
    }
    return h;
}
static void mv_runs(const SChol *S);
void schol_attach_pattern(SChol *S, int *deg, int **nbr) {
    const int m = S->m;
    {
        /* pattern of M: for permuted column j, rows i >= j (permuted) with their offsets */
        size_t tot = (size_t)m;
        for (int r = 0; r < m; r++) tot += deg[r];
        S->mcp = sx(sizeof(int) * (m + 1));
        S->mri = sx(sizeof(int) * (tot / 2 + m + 1));
        S->moff = sx(sizeof(size_t) * (tot / 2 + m + 1));
        int *cnt = sx(sizeof(int) * (m + 1));
        for (int r = 0; r < m; r++) {
            const int pr = S->iperm[r];
            cnt[pr]++;                                            /* diagonal */
            for (int q = 0; q < deg[r]; q++) { const int pc = S->iperm[nbr[r][q]]; if (pc < pr) cnt[pc]++; }
        }
        S->mcp[0] = 0;
        for (int j = 0; j < m; j++) S->mcp[j + 1] = S->mcp[j] + cnt[j];
        memset(cnt, 0, sizeof(int) * (m + 1));
        /* rows visited in increasing permuted order: every column comes out sorted (the
         * per-column qsort was 5% of attr_vdp_d10_I) */
        for (int pr = 0; pr < m; pr++) {
            const int r = S->perm[pr];
            for (int q = 0; q < deg[r]; q++) { const int pc = S->iperm[nbr[r][q]]; if (pc < pr) S->mri[S->mcp[pc] + cnt[pc]++] = pr; }
            /* the diagonal is the first entry of its own column: all rows visited before pr
             * that land in column pr are < pr, which cannot happen (pc < pr only) */
            S->mri[S->mcp[pr] + cnt[pr]++] = pr;
        }
        free(cnt);
        /* 17 October 2026: the offsets by one merge of the column's rows (sorted) with its supernode's row list
         * (sorted, from the column's own position) instead of a binary search per entry: the same offsets
         * (broyden25: 20 million entries, 1.2 s -> see TRICKS) */
        for (int j = 0; j < m; j++) {
            const int k = S->snof[j], cc = j - S->ss[k], nr = S->pnr[k];
            const int *rw = S->prow + S->prp[k];
            const size_t base = S->pbase[k] + (size_t)cc * nr;
            int p = cc;
            for (int q = S->mcp[j]; q < S->mcp[j + 1]; q++) {
                const int i = S->mri[q];
                while (p < nr && rw[p] < i) p++;
                S->moff[q] = (p < nr && rw[p] == i) ? base + p : SIZE_MAX;
            }
        }
    }
    S->run_state = 0; mv_runs(S);
}
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap) {
    SChol *S = schol_analyze_sym(m, deg, nbr, fillcap);
    if (S) schol_attach_pattern(S, deg, nbr);
    return S;
}


#ifdef HAVE_AMD
#include "amd/amd.h"          /* 4.32: AMD is bundled (amd/, BSD-3-clause) */
/* 4.22: approximate minimum degree (SuiteSparse AMD, quotient graph: O(pattern) time and
 * memory) and a symbolic factorization along its elimination tree. The explicit-graph
 * minimum degree above costs O(flops) and gave up on the Schur patterns of the large
 * AC-OPF moment forms (case6468: m = 77,459). Same outputs as schol_order.            */
static int schol_symb_perm(int m, SAdj *adj, size_t fillcap, int *P, int **perm_out, int **cpat_len,
                           int ***cpat, double *flops_out, size_t *nnz_out);
static int g_amd = -1;
static int schol_order_amd(int m, SAdj *adj, size_t fillcap, int **perm_out, int **cpat_len,
                           int ***cpat, double *flops_out, size_t *nnz_out) {
    size_t nz = 0;
    for (int i = 0; i < m; i++) nz += adj[i].n;
    int *Ap = malloc(sizeof(int) * (m + 1)), *Ai = malloc(sizeof(int) * (nz + 1)), *P = malloc(sizeof(int) * (m + 1));
    if (!Ap || !Ai || !P) { free(Ap); free(Ai); free(P); return 1; }
    Ap[0] = 0;
    for (int i = 0; i < m; i++) { memcpy(Ai + Ap[i], adj[i].a, sizeof(int) * adj[i].n); Ap[i + 1] = Ap[i] + adj[i].n; }
    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_defaults(Control);
    if (g_amd_dense != 0) Control[AMD_DENSE] = g_amd_dense;
    Control[AMD_AGGRESSIVE] = 1;
    int st = AMD_OK;
    if (g_force_perm) memcpy(P, g_force_perm, sizeof(int) * (size_t)m);       /* the caller's ordering */
    else st = amd_order(m, Ap, Ai, P, Control, Info);
    free(Ap); free(Ai);
    if (st != AMD_OK && st != AMD_OK_BUT_JUMBLED) { free(P); return 1; }
    return schol_symb_perm(m, adj, fillcap, P, perm_out, cpat_len, cpat, flops_out, nnz_out);
}
#endif
/* symbolic factorization along the elimination order P (P[k] = vertex eliminated k-th;
 * takes P); same outputs as schol_order */
__attribute__((unused)) static int schol_symb_perm(int m, SAdj *adj, size_t fillcap, int *P, int **perm_out, int **cpat_len,
                           int ***cpat, double *flops_out, size_t *nnz_out) {
    int *ip = sx(sizeof(int) * m), *mark = sx(sizeof(int) * m), *head = sx(sizeof(int) * (m + 1)), *nxt = sx(sizeof(int) * (m + 1));
    int *plen = sx(sizeof(int) * m), **pat = sx(sizeof(int *) * m), *buf = sx(sizeof(int) * (m + 1));
    for (int j = 0; j < m; j++) { ip[P[j]] = j; mark[j] = -1; head[j] = -1; }
    size_t total = 0;
    double flops = 0;
    int rc = 0;
    for (int j = 0; j < m && !rc; j++) {
        const int v = P[j];
        int len = 0;
        buf[len++] = v; mark[v] = j;
        for (int q = 0; q < adj[v].n; q++) { const int u = adj[v].a[q]; if (ip[u] > j && mark[u] != j) { mark[u] = j; buf[len++] = u; } }
        const int len_adj = len;
        for (int c = head[j]; c >= 0; c = nxt[c])
            for (int q = 1; q < plen[c]; q++) { const int u = pat[c][q]; if (mark[u] != j) { mark[u] = j; buf[len++] = u; } }
        if (g_zf && len != len_adj) { rc = 1; break; }     /* (the chordality check: a fill entry) */
        total += (size_t)len;
        flops += (double)len * len;
        if (an_over(total, flops, fillcap)) { rc = 1; break; }
        pat[j] = sx(sizeof(int) * len);   /* 5.12: checked (a NULL here crashed tru19 under a memory cap) */
        memcpy(pat[j], buf, sizeof(int) * len);
        plen[j] = len;
        int par = m;
        for (int q = 1; q < len; q++) if (ip[buf[q]] < par) par = ip[buf[q]];
        if (par < m) { nxt[j] = head[par]; head[par] = j; }
    }
    free(ip); free(mark); free(head); free(nxt); free(buf);
    if (rc) { for (int j = 0; j < m; j++) free(pat[j]); free(pat); free(plen); free(P); return 1; }
    *perm_out = P; *cpat = pat; *cpat_len = plen; *flops_out = flops; *nnz_out = total;
    return 0;
}

int g_nd_block = 0;          /* 4.22: set around analyses of quasi-definite (signed) matrices */
#ifdef HAVE_CHOLMOD
#include <suitesparse/cholmod.h>
/* 4.22: CHOLMOD's nested dissection (METIS node bisection recursively, CCOLAMD on the
 * separator tree) with large leaves (nd_small 5000). On the AC-OPF Schur patterns it beats
 * AMD from about m = 30000: case9241 6.2e10 against 8.3e10 flops, case6468 1.58e10 against
 * 1.69e10, case2869 1.02e9 against 1.11e9 (plain METIS_NodeND was 5-60% worse than AMD). */
static int schol_order_nesdis(int m, SAdj *adj, size_t fillcap, int **perm_out, int **cpat_len,
                              int ***cpat, double *flops_out, size_t *nnz_out) {
    size_t nz = 0;
    for (int i = 0; i < m; i++) for (int q = 0; q < adj[i].n; q++) if (adj[i].a[q] > i) nz++;
    if (nz + (size_t)m >= (size_t)INT32_MAX) return 1;
    cholmod_common c;
    cholmod_start(&c);
    c.print = 0;
    cholmod_sparse *A = cholmod_allocate_sparse(m, m, nz + m, 0, 1, -1, CHOLMOD_PATTERN, &c);
    if (!A) { cholmod_finish(&c); return 1; }
    int *Ap = A->p, *Ai = A->i;
    size_t w = 0;
    for (int j = 0; j < m; j++) {
        Ap[j] = (int)w; Ai[w++] = j;
        for (int q = 0; q < adj[j].n; q++) if (adj[j].a[q] > j) Ai[w++] = adj[j].a[q];
    }
    Ap[m] = (int)w;
    cholmod_sort(A, &c);      /* METIS's partitions depend on the adjacency order */
    static int nds = -1;
    if (nds < 0) { const char *e = getenv("BRISK_NDSMALL"); nds = e ? atoi(e) : 5000; }
    c.current = 0;
    c.method[0].ordering = CHOLMOD_NESDIS;
    c.method[0].nd_small = nds;
    c.method[0].nd_camd = 1;
    int *Perm = malloc(sizeof(int) * (m + 1)), *CP = malloc(sizeof(int) * (m + 1)), *CM = malloc(sizeof(int) * (m + 1));
    const int64_t nc = (Perm && CP && CM) ? cholmod_nested_dissection(A, NULL, 0, Perm, CP, CM, &c) : -1;
    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);
    free(CP); free(CM);
    if (nc < 0) { free(Perm); return 1; }
    return schol_symb_perm(m, adj, fillcap, Perm, perm_out, cpat_len, cpat, flops_out, nnz_out);
}
#endif

#ifdef HAVE_METIS
#include <metis.h>
/* 4.22: nested dissection (METIS) on large patterns; kept when its factorization is
 * cheaper than AMD's (the AC-OPF Schur graphs come from near-planar networks) */
static int schol_order_metis(int m, SAdj *adj, size_t fillcap, int **perm_out, int **cpat_len,
                             int ***cpat, double *flops_out, size_t *nnz_out) {
    size_t nz = 0;
    for (int i = 0; i < m; i++) nz += adj[i].n;
    if (nz >= (size_t)INT32_MAX) return 1;
    idx_t *xadj = malloc(sizeof(idx_t) * (m + 1)), *adjncy = malloc(sizeof(idx_t) * (nz + 1));
    idx_t *perm = malloc(sizeof(idx_t) * (m + 1)), *iperm = malloc(sizeof(idx_t) * (m + 1));
    if (!xadj || !adjncy || !perm || !iperm) { free(xadj); free(adjncy); free(perm); free(iperm); return 1; }
    size_t w = 0;
    xadj[0] = 0;
    for (int i = 0; i < m; i++) {
        for (int q = 0; q < adj[i].n; q++) if (adj[i].a[q] != i) adjncy[w++] = adj[i].a[q];
        xadj[i + 1] = (idx_t)w;
    }
    idx_t n = m, options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    const int st = METIS_NodeND(&n, xadj, adjncy, NULL, options, perm, iperm);
    free(xadj); free(adjncy); free(iperm);
    if (st != METIS_OK) { if (getenv("BRISK_AMDBG")) printf("   [METIS_NodeND status %d]\n", st); free(perm); return 1; }
    int *P = malloc(sizeof(int) * (m + 1));
    for (int k = 0; k < m; k++) P[k] = (int)perm[k];
    free(perm);
    return schol_symb_perm(m, adj, fillcap, P, perm_out, cpat_len, cpat, flops_out, nnz_out);
}
#endif

/* 5.5: the own nested dissection (nd.c); the order is kept when its factorization is cheaper than AMD's */
int nd_order(int n, const int *xadj, const int *adjncy, int *perm);
static int g_nd_try = 0, g_nd_used = 0;
void schol_set_nd(int on) { g_nd_try = on; }
extern int g_nd_block;
static unsigned an_settings(void) { return (unsigned)(g_amd + 2) | (unsigned)(g_nd_try != 0) << 8 | (unsigned)(g_nd_block != 0) << 9 | (unsigned)(g_force_perm != NULL) << 10 | (unsigned)(g_peo_try != 0) << 11; }
int schol_nd_used(void) { return g_nd_used; }
static int schol_order_nd(int m, SAdj *adj, size_t fillcap, int **perm_out, int **cpat_len,
                          int ***cpat, double *flops_out, size_t *nnz_out) {
    size_t nz = 0;
    for (int i = 0; i < m; i++) nz += adj[i].n;
    if (nz >= (size_t)INT32_MAX) return 1;
    int *xadj = malloc(sizeof(int) * (m + 1)), *adjncy = malloc(sizeof(int) * (nz + 1)), *P = malloc(sizeof(int) * (m + 1));
    if (!xadj || !adjncy || !P) { free(xadj); free(adjncy); free(P); return 1; }
    size_t w = 0;
    xadj[0] = 0;
    for (int i = 0; i < m; i++) {
        for (int q = 0; q < adj[i].n; q++) if (adj[i].a[q] != i) adjncy[w++] = adj[i].a[q];
        xadj[i + 1] = (int)w;
    }
    const int st = nd_order(m, xadj, adjncy, P);
    free(xadj); free(adjncy);
    if (st) { free(P); return 1; }
    return schol_symb_perm(m, adj, fillcap, P, perm_out, cpat_len, cpat, flops_out, nnz_out);
}
#ifdef HAVE_AMD
static int g_amd_big = -1;
#endif
/* (the own ordering competes with a completed AMD order: automatic mode below BRISK_AMDBIG) */
static int an_amd_competes(int m) {
#ifdef HAVE_AMD
    return g_amd == 1 && g_amd_big > 0 && m < g_amd_big;
#else
    (void)m; return 0;
#endif
}
/* 18 October 2026: a chordal pattern needs no fill-reducing heuristic. Maximum cardinality search (Tarjan and
 * Yannakakis 1984) numbers the vertices in linear time; on a chordal graph the reverse of its visit order is a perfect
 * elimination ordering, so the factor has exactly the pattern of M (zero fill, which no ordering beats). The symbolic
 * factorization along that order checks it and stops at the first fill entry (g_zf); the other orderings then run as
 * before. The Schur complement of an SDP is a union of cliques (each block's constraints, each LP variable's), chordal
 * when they form a tree with the running-intersection property: chained SOS relaxations, chordal conversions. Only
 * the callers that ask (schol_set_peo: the SDP solver's Schur complement); the LP and cone solvers are unchanged. */
int schol_set_peo(int on) { const int o = g_peo_try; g_peo_try = on; return o; }
int schol_peo_used(void) { return g_an_peo; }
static int *mcs_order(int m, const SAdj *adj) {
    int *w = sx(sizeof(int) * (size_t)m), *nx = sx(sizeof(int) * (size_t)m), *pv = sx(sizeof(int) * (size_t)m);
    int *head = sx(sizeof(int) * ((size_t)m + 1)), *P = sx(sizeof(int) * (size_t)m);
    char *done = sx((size_t)m);
    for (int b = 0; b <= m; b++) head[b] = -1;
    for (int v = m - 1; v >= 0; v--) { nx[v] = head[0]; pv[v] = -1; if (head[0] >= 0) pv[head[0]] = v; head[0] = v; }
    int maxw = 0;
    for (int i = m - 1; i >= 0; i--) {
        while (maxw > 0 && head[maxw] < 0) maxw--;
        const int v = head[maxw];
        head[maxw] = nx[v]; if (nx[v] >= 0) pv[nx[v]] = -1;
        done[v] = 1; P[i] = v;
        for (int q = 0; q < adj[v].n; q++) {
            const int u = adj[v].a[q];
            if (done[u]) continue;
            if (pv[u] >= 0) nx[pv[u]] = nx[u]; else head[w[u]] = nx[u];      /* out of its bucket */
            if (nx[u] >= 0) pv[nx[u]] = pv[u];
            w[u]++;
            nx[u] = head[w[u]]; pv[u] = -1; if (head[w[u]] >= 0) pv[head[w[u]]] = u; head[w[u]] = u;
            if (w[u] > maxw) maxw = w[u];
        }
    }
    free(w); free(nx); free(pv); free(head); free(done);
    return P;
}
static SChol *schol_analyze(int m, SAdj *adj, size_t fillcap) {
    g_an_hit = 0; g_an_tot = 0; g_an_fl = 0; g_an_fillcap = fillcap;
    g_an_ph = 0; g_an_hit_other = 0; g_an_own = 0; g_an_amd_tot = SIZE_MAX; g_an_amd_fl = 1e300; g_an_own_tot = 0; g_an_own_fl = 0;
    int *perm = NULL, *plen = NULL, **pat = NULL;
    double flops = 0;
    size_t nnz = 0;
    int ord_ok = 0;
    g_an_peo = 0;
    if (g_peo_try && !g_force_perm && !g_nd_block && m >= 64) {
        int *P = mcs_order(m, adj);
        g_zf = 1;
        const int rs = schol_symb_perm(m, adj, fillcap, P, &perm, &plen, &pat, &flops, &nnz);    /* (takes P) */
        g_zf = 0;
        if (rs == 0) {
            /* the columns of each fundamental supernode (a clique with the same later neighbours: any order of them is
             * still a perfect elimination ordering) by the first column whose pattern reaches them (ties by vertex
             * number). The sets of rows that the descendants send into a supernode are then contiguous where they
             * are nested (a chain of cliques: the grandchild's rows in a node are part of the child's), and the
             * updates are one dgemm into the target instead of a product and a scatter (broyden25 in the search's
             * order: the scatter 0.28 -> 0.76 s, the factorizations +5-10 %) */
            int *ip = sx(sizeof(int) * (size_t)m), *P2 = sx(sizeof(int) * (size_t)m), *key = sx(sizeof(int) * (size_t)m);
            for (int j = 0; j < m; j++) { ip[perm[j]] = j; key[j] = j; }
            for (int j = 0; j < m; j++) for (int q = 1; q < plen[j]; q++) { const int r = ip[pat[j][q]]; if (j < key[r]) key[r] = j; }
            int j0 = 0;
            while (j0 < m) {
                int j1 = j0 + 1;
                while (j1 < m && plen[j1] == plen[j1 - 1] - 1 && plen[j1 - 1] > 1) {
                    int par = m; for (int q = 1; q < plen[j1 - 1]; q++) { const int r = ip[pat[j1 - 1][q]]; if (r < par) par = r; }
                    if (par != j1) break;
                    j1++;
                }
                /* (key, vertex) pairs of the run, sorted */
                long long *kv = sx(sizeof(long long) * (size_t)(j1 - j0));
                for (int t = j0; t < j1; t++) kv[t - j0] = (long long)key[t] * (m + 1LL) + perm[t];
                qsort(kv, (size_t)(j1 - j0), sizeof(long long), cmp_ll);
                for (int t = j0; t < j1; t++) P2[t] = (int)(kv[t - j0] % (m + 1LL));
                free(kv);
                j0 = j1;
            }
            free(key);
            free(ip);
            for (int j = 0; j < m; j++) free(pat[j]);
            free(pat); free(plen); free(perm); pat = NULL; plen = NULL; perm = NULL;
            g_zf = 1;
            const int rs2 = schol_symb_perm(m, adj, fillcap, P2, &perm, &plen, &pat, &flops, &nnz);
            g_zf = 0;
            if (rs2 == 0) { ord_ok = 1; g_an_peo = 1; goto ordered; }
        }
    }
#ifdef HAVE_AMD
    /* BRISK_AMD: 0 never, 1 when the own ordering gives up (default), 2 always */
    if (g_amd < 0) { const char *e = getenv("BRISK_AMD"); g_amd = e ? atoi(e) : 1; }
    if (g_amd_big < 0) { const char *e = getenv("BRISK_AMDBIG"); g_amd_big = e ? atoi(e) : 20000; }
    g_an_ph = 1;
    if (g_amd == 2 && m >= 64 && !schol_order_amd(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) ord_ok = 1;
    g_an_ph = 0;
    g_nd_used = 0;
    /* (only where the factorization costs: the dissection itself takes the time of a few of them.
     * 5.7: from 1e9 flops instead of 2e8: with relaxed supernodes and the left-looking code a
     * factorization of that size takes half the time it did, and the dissection, 0.4-0.5 s on
     * dfl001 and pds-10, no longer pays for itself there: 2.59 -> 2.23 s and 1.11 -> 0.96 s) */
    /* (18 October 2026, LP/SOCP: not when AMD's factor already fills a quarter of the triangle: a graph that fills like
     * that has no small separators, and the dissection only costs its time (set covering 5,000 rows, 89 % full: 6 s of
     * two dissections for the same flops; min-cost flow on a random graph, 37 %: 12 s; the meshes of DC power flow,
     * 13-15 %, gain 2.6-5.4 times in flops). The dissection serves the LP method only.) */
    if (ord_ok && g_nd_try && m >= 1000 && flops >= (getenv("BRISK_NDMINFLOPS") ? atof(getenv("BRISK_NDMINFLOPS")) : 1e9) && !g_force_perm
        && (double)nnz <= 0.25 * 0.5 * (double)m * (m + 1.0)) {
        int *perm2 = NULL, *plen2 = NULL, **pat2 = NULL;
        double flops2 = 0; size_t nnz2 = 0;
        const double tm = (double)clock() / CLOCKS_PER_SEC;
        const int rs = schol_order_nd(m, adj, (size_t)(1.2 * (double)nnz) + (size_t)m, &perm2, &plen2, &pat2, &flops2, &nnz2);
        if (getenv("BRISK_AMDBG")) printf("   [ordering m %d: AMD %.3g flops %zu nnz; nested dissection %s %.3g flops %zu nnz (%.2fs)]\n", m, flops, nnz, rs ? "failed/over" : "ok", flops2, nnz2, (double)clock() / CLOCKS_PER_SEC - tm);
        if (rs == 0 && flops2 < 0.9 * flops) {
            for (int j = 0; j < m; j++) free(pat[j]);
            free(pat); free(plen); free(perm);
            perm = perm2; plen = plen2; pat = pat2; flops = flops2; nnz = nnz2; g_nd_used = 1;
        } else {
            if (rs == 0) { for (int j = 0; j < m; j++) free(pat2[j]); free(pat2); free(plen2); free(perm2); }
            /* 18 October 2026 (LP/SOCP): the dissection's hub rule (vertices of more than five times the mean degree
             * ordered last, unsearched) puts every period's resource rows of a staircase LP into the top separator: the
             * dissection's fill passes AMD's. Without the rule it finds the periods' separators (staircase 50 x 500:
             * 3.85e10 flops by AMD, 1.11e10 by the dissection, 1.27e10 by MOSEK's graph partitioning; 200 x 500:
             * 2.37e12 -> 5.55e10), but it takes longer (6 and 23 s against 0.15 s). So the second trial is made when
             * the first did not beat AMD and AMD's factorization costs 1e10 flops or more: at 30 factorizations, tens of
             * seconds that a better order recovers. (The dissection serves the LP method only.) BRISK_NDHUB2=0: off. */
            extern void nd_set_hubs(int on);
            if (flops >= 1e10 && !(getenv("BRISK_NDHUB2") && atoi(getenv("BRISK_NDHUB2")) == 0)) {
                perm2 = NULL; plen2 = NULL; pat2 = NULL; flops2 = 0; nnz2 = 0;
                const double tm2 = (double)clock() / CLOCKS_PER_SEC;
                nd_set_hubs(0);
                const int rs3 = schol_order_nd(m, adj, (size_t)(1.2 * (double)nnz) + (size_t)m, &perm2, &plen2, &pat2, &flops2, &nnz2);
                nd_set_hubs(1);
                if (getenv("BRISK_AMDBG")) printf("   [ordering m %d: nested dissection without hubs %s %.3g flops %zu nnz (%.2fs)]\n", m, rs3 ? "failed/over" : "ok", flops2, nnz2, (double)clock() / CLOCKS_PER_SEC - tm2);
                if (rs3 == 0 && flops2 < 0.9 * flops) {
                    for (int j = 0; j < m; j++) free(pat[j]);
                    free(pat); free(plen); free(perm);
                    perm = perm2; plen = plen2; pat = pat2; flops = flops2; nnz = nnz2; g_nd_used = 1;
                } else if (rs3 == 0) { for (int j = 0; j < m; j++) free(pat2[j]); free(pat2); free(plen2); free(perm2); }
            }
        }
    }
    /* auto (4.22): AMD first (fast); on large patterns it is kept (TSSOS case2869: 0.89 against
     * 1.3 GF per factorization, and the own ordering took 1.6 s), on small ones the own
     * ordering runs as well, capped at AMD's fill, and the cheaper factorization wins */
    size_t fillcap_own = fillcap;
    int have_amd = 0;
    g_an_ph = 1;
    const int amd_ok = g_amd == 1 && m >= 64 && !schol_order_amd(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz);
    g_an_ph = 0;
    if (amd_ok) {
        have_amd = 1;
        if (m >= g_amd_big) {
            ord_ok = 1;
#ifdef HAVE_CHOLMOD
            static int g_nd = -1, g_ndmin = -1;
            if (g_nd < 0) { const char *e = getenv("BRISK_NESDIS"); g_nd = e ? atoi(e) : 1; e = getenv("BRISK_NDMIN"); g_ndmin = e ? atoi(e) : 20000; }
            if (g_nd && m >= g_ndmin && !g_nd_block) {
                int *perm2 = NULL, *plen2 = NULL, **pat2 = NULL;
                double flops2 = 0; size_t nnz2 = 0;
                const double tm = (double)clock() / CLOCKS_PER_SEC;
                const int rs = schol_order_nesdis(m, adj, (size_t)(1.05 * (double)nnz) + (size_t)m, &perm2, &plen2, &pat2, &flops2, &nnz2);
                if (getenv("BRISK_AMDBG")) printf("   [ordering m %d: AMD %.3g flops %zu nnz; NESDIS %s %.3g flops %zu nnz (%.2fs)]\n", m, flops, nnz, rs ? "failed/over" : "ok", flops2, nnz2, (double)clock() / CLOCKS_PER_SEC - tm);
                if (rs == 0 && flops2 < 0.97 * flops) {
                    for (int j = 0; j < m; j++) free(pat[j]);
                    free(pat); free(plen); free(perm);
                    perm = perm2; plen = plen2; pat = pat2; flops = flops2; nnz = nnz2;
                } else if (rs == 0) {
                    for (int j = 0; j < m; j++) free(pat2[j]);
                    free(pat2); free(plen2); free(perm2);
                }
            }
#endif
#ifdef HAVE_METIS
            static int g_metis = -1;
            if (g_metis < 0) { const char *e = getenv("BRISK_METIS"); g_metis = e ? atoi(e) : 0;   /* measured worse than AMD on the AC-OPF Schur graphs (case2869: 1.5-1.7e9 against 0.9-1.1e9 flops) */ }
            if (g_metis) {
                int *perm2 = NULL, *plen2 = NULL, **pat2 = NULL;
                double flops2 = 0; size_t nnz2 = 0;
                const double tm = (double)clock() / CLOCKS_PER_SEC;
                const int rs = schol_order_metis(m, adj, getenv("BRISK_METISCAP") ? (size_t)-1 : (size_t)(1.05 * (double)nnz) + (size_t)m, &perm2, &plen2, &pat2, &flops2, &nnz2);
                if (getenv("BRISK_AMDBG")) printf("   [ordering m %d: AMD %.3g flops %zu nnz; METIS %s %.3g flops %zu nnz (%.2fs)]\n", m, flops, nnz, rs ? "failed" : "ok", flops2, nnz2, (double)clock() / CLOCKS_PER_SEC - tm);
                if (rs == 0 && flops2 < flops) {
                    for (int j = 0; j < m; j++) free(pat[j]);
                    free(pat); free(plen); free(perm);
                    perm = perm2; plen = plen2; pat = pat2; flops = flops2; nnz = nnz2;
                } else if (rs == 0) {
                    for (int j = 0; j < m; j++) free(pat2[j]);
                    free(pat2); free(plen2); free(perm2);
                }
            }
#endif
        }
        /* (17 October 2026: the own ordering's limit is AMD's flops, the measure of the choice (g_flopcap_own, below), and
         * the caller's fill cap; its fill was capped at 1.1 x AMD's before, which could reject an order with fewer
         * flops. BRISK_OWNFILL=1: that cap) */
        else if (getenv("BRISK_OWNFILL")) fillcap_own = (size_t)(1.1 * (double)nnz) + (size_t)m;
    }
    if (!ord_ok && have_amd) {
        int *perm2 = NULL, *plen2 = NULL, **pat2 = NULL;
        double flops2 = 0; size_t nnz2 = 0;
        /* 17 October 2026: it is kept only with fewer flops than AMD's, and its flops only grow: it stops once they
         * pass AMD's (broyden25: it ran on to 1.4e11 against AMD's 2.7e10). The same choice, sooner. BRISK_OWNCAP=0 off */
        {   static int oc = -1; if (oc < 0) { const char *e = getenv("BRISK_OWNCAP"); oc = !(e && atoi(e) == 0); }
            g_flopcap_own = oc ? flops : 0; }
        int rs = schol_order_super(m, adj, fillcap_own, &perm2, &plen2, &pat2, &flops2, &nnz2);
        if (rs == 2 && schol_order(m, adj, fillcap_own, &perm2, &plen2, &pat2, &flops2, &nnz2)) rs = 1;
        else if (rs == 2) rs = 0;
        g_flopcap_own = 0;
        if (getenv("BRISK_AMDBG")) printf("   [ordering m %d: AMD %.3g flops %zu nnz; own %s %.3g flops %zu nnz]\n", m, flops, nnz, rs ? "failed" : "ok", flops2, nnz2);
        if (rs == 0 && flops2 < flops) {
            for (int j = 0; j < m; j++) free(pat[j]);
            free(pat); free(plen); free(perm);
            perm = perm2; plen = plen2; pat = pat2; flops = flops2; nnz = nnz2;
        } else if (rs == 0) {
            for (int j = 0; j < m; j++) free(pat2[j]);
            free(pat2); free(plen2); free(perm2);
        }
        ord_ok = 1;
    }
#endif
    if (!ord_ok) {
        /* the own orderings: minimum degree with least-fill ties (better than AMD on the
         * clique-structured moment-SOS patterns: attr_vdp_d10_I 4.1e8 flops against a
         * rejected factor), but O(flops) work; on large patterns they give up */
        g_an_ph = 2;
        int rs = schol_order_super(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz);
        if (rs == 0) ord_ok = 1;
        else if (rs == 2 && !schol_order(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) ord_ok = 1;
        g_an_ph = 0;
        g_an_own = ord_ok;
    }
#ifdef HAVE_AMD
    g_an_ph = 1;
    if (!ord_ok && g_amd == 1 && m >= 64 && !schol_order_amd(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) ord_ok = 1;
    g_an_ph = 0;
#endif
ordered:
    if (!ord_ok) return NULL;
    {
        /* postorder the elimination tree (same fill): a parent then follows its last
         * child, which the relaxed amalgamation below needs; the child with the longest
         * column goes last so the heavy chains become contiguous */
        int *ip = sx(sizeof(int) * m), *par = sx(sizeof(int) * m);
        for (int i = 0; i < m; i++) ip[perm[i]] = i;
        for (int j = 0; j < m; j++) {
            int p = m;
            for (int q = 0; q < plen[j]; q++) { const int r = ip[pat[j][q]]; if (r > j && r < p) p = r; }
            par[j] = p < m ? p : -1;
        }
        int *head = sx(sizeof(int) * (m + 1)), *nxt = sx(sizeof(int) * m), *stk = sx(sizeof(int) * (m + 1));
        for (int i = 0; i <= m; i++) head[i] = -1;
        /* children lists sorted by column length descending when pushed (so popped ascending) */
        int *ordc = sx(sizeof(int) * m);
        for (int j = 0; j < m; j++) ordc[j] = j;
        /* counting sort by plen */
        int maxl = 0; for (int j = 0; j < m; j++) if (plen[j] > maxl) maxl = plen[j];
        int *cntl = sx(sizeof(int) * (maxl + 2));
        for (int j = 0; j < m; j++) cntl[plen[j]]++;
        for (int l = 1; l <= maxl + 1; l++) cntl[l] += cntl[l - 1];
        for (int j = m - 1; j >= 0; j--) ordc[--cntl[plen[j]]] = j;   /* ascending length */
        free(cntl);
        /* insert at list heads in ascending order -> each list is descending by length */
        for (int t = 0; t < m; t++) { const int j = ordc[t], p = par[j] < 0 ? m : par[j]; nxt[j] = head[p]; head[p] = j; }
        free(ordc);
        int *post = sx(sizeof(int) * m), np = 0;
        /* iterative DFS from the virtual root m; children visited in list order (longest
         * first), the node emitted after all of its children: longest child ends up last?
         * no: emitted order = order visited; reverse the list so the longest is visited last */
        for (int p = 0; p <= m; p++) {                  /* reverse each child list */
            int prev = -1, c = head[p];
            while (c >= 0) { const int nx = nxt[c]; nxt[c] = prev; prev = c; c = nx; }
            head[p] = prev;
        }
        int top = 0; stk[top++] = m;
        int *it = sx(sizeof(int) * (m + 1));
        for (int i = 0; i <= m; i++) it[i] = head[i];
        while (top) {
            const int v = stk[top - 1];
            if (it[v] >= 0) { const int c = it[v]; it[v] = nxt[c]; stk[top++] = c; }
            else { top--; if (v < m) post[np++] = v; }
        }
        free(it); free(head); free(nxt); free(stk); free(par); free(ip);
        int *perm2 = sx(sizeof(int) * m), *plen2 = sx(sizeof(int) * m), **pat2 = sx(sizeof(int *) * m);
        for (int i = 0; i < m; i++) { perm2[i] = perm[post[i]]; plen2[i] = plen[post[i]]; pat2[i] = pat[post[i]]; }
        free(post); free(perm); free(plen); free(pat);
        perm = perm2; plen = plen2; pat = pat2;
    }
    SChol *S = sx(sizeof(SChol));
    { static long g_uid = 0;
      #pragma omp atomic capture
      S->uid = ++g_uid; }
    S->m = m;
    S->perm = perm;
    S->iperm = sx(sizeof(int) * m);
    for (int i = 0; i < m; i++) S->iperm[perm[i]] = i;
    S->cp = sx(sizeof(int) * (m + 1));
    S->ci = sx(sizeof(int) * (nnz ? nnz : 1));

    size_t p = 0;
    int maxl = 1; for (int j = 0; j < m; j++) if (plen[j] > maxl) maxl = plen[j];
    int *stmp = malloc(sizeof(int) * (size_t)maxl);
    for (int j = 0; j < m; j++) {
        S->cp[j] = (int)p;
        /* column j in permuted indices: its clique, mapped and sorted, diagonal first (sc_isort: distinct integers, so
         * the order qsort gave; qsort was 6 % of the cone solver's lp_2000x6000) */
        int len = plen[j];
        int *col = S->ci + p;
        int k = 0;
        for (int q = 0; q < len; q++) {
            int r = S->iperm[pat[j][q]];
            if (r >= j) col[k++] = r;
        }
        sc_isort(col, k, stmp, m);
        p += (size_t)k;
        free(pat[j]);
    }
    free(stmp);
    S->cp[m] = (int)p;
    S->nnz = p;
    S->flops = flops;
    free(pat); free(plen);
    /* ---- supernodes: consecutive columns whose patterns nest exactly */
    S->ss = sx(sizeof(int) * (m + 1));
    S->snof = sx(sizeof(int) * m);
    int ns = 0;
    for (int j = 0; j < m; ) {
        int e = j + 1;
        while (e < m) {
            int lj = S->cp[e] - S->cp[e - 1], le = S->cp[e + 1] - S->cp[e];
            if (le != lj - 1) break;
            int ok = 1;
            for (int q = 0; q < le; q++)
                if (S->ci[S->cp[e] + q] != S->ci[S->cp[e - 1] + 1 + q]) { ok = 0; break; }
            if (!ok) break;
            e++;
        }
        S->ss[ns++] = j;
        for (int c2 = j; c2 < e; c2++) S->snof[c2] = ns - 1;
        j = e;
    }
    S->ss[ns] = m;
    /* ---- relaxed amalgamation: a supernode whose last column has the first column of the
     * next supernode as its etree parent is merged into it when the explicit zeros stay
     * below a fraction of the merged panel (rows = own columns + the next node's rows).
     * Small exact supernodes (width ~8 on the AC-OPF Schur patterns) ran the dgemm updates
     * and the solves at a fraction of the BLAS rate.                                     */
    int *slast = sx(sizeof(int) * (ns + 1));          /* last merged exact supernode start */
    {
        /* 5.7: the fraction is set by the caller (schol_set_amalg: 0.3 for the normal equations of the LP
         * method, where the exact supernodes are narrow and the factorization ran at an eighth of the
         * BLAS rate; 0 = exact supernodes for the Schur complements of the SDP path, where it measured
         * neutral); BRISK_AMALG overrides */
        static double aenv = -2; static int amaxw = 0;
        if (aenv < -1) { const char *e = getenv("BRISK_AMALG"); aenv = e ? atof(e) : -1;
                         const char *w = getenv("BRISK_AMALGW"); amaxw = w ? atoi(w) : 64; }
        const double afrac = aenv >= 0 ? aenv : g_amalg;
        int *ss2 = sx(sizeof(int) * (ns + 1)), n2 = 0;
        int cj0 = S->ss[0], clast = S->ss[0];
        double ctrue = 0;
        for (int c = S->ss[0]; c < S->ss[1]; c++) ctrue += S->cp[c + 1] - S->cp[c];
        for (int k = 1; k <= ns; k++) {
            int merged = 0;
            if (k < ns && afrac > 0) {
                const int e = S->ss[k], lc = e - 1;
                const int par = (S->cp[lc + 1] - S->cp[lc] > 1) ? S->ci[S->cp[lc] + 1] : -1;
                if (par == e) {
                    const int wn = S->ss[k + 1] - cj0, nrn = (e - cj0) + (S->cp[e + 1] - S->cp[e]);
                    double tn = ctrue;
                    for (int c = e; c < S->ss[k + 1]; c++) tn += S->cp[c + 1] - S->cp[c];
                    const double store = (double)wn * nrn - 0.5 * (double)wn * (wn - 1);
                    if (wn <= amaxw && store - tn <= afrac * store) { merged = 1; clast = e; ctrue = tn; }
                }
            }
            if (!merged) {
                ss2[n2] = cj0; slast[n2] = clast; n2++;
                if (k < ns) {
                    cj0 = clast = S->ss[k]; ctrue = 0;
                    for (int c = S->ss[k]; c < S->ss[k + 1]; c++) ctrue += S->cp[c + 1] - S->cp[c];
                }
            }
        }
        ss2[n2] = m;
        if (getenv("BRISK_AMDBG")) {
            int npar = 0;
            for (int k = 1; k < ns; k++) { const int e = S->ss[k], lc = e - 1; if (S->cp[lc + 1] - S->cp[lc] > 1 && S->ci[S->cp[lc] + 1] == e) npar++; }
            printf("   [amalgamation: %d -> %d supernodes (m %d), %d adjacent parents]\n", ns, n2, m, npar);
        }
        memcpy(S->ss, ss2, sizeof(int) * (n2 + 1));
        free(ss2);
        ns = n2;
        for (int k = 0; k < ns; k++) for (int c2 = S->ss[k]; c2 < S->ss[k + 1]; c2++) S->snof[c2] = k;
    }
    S->ns = ns;
    S->prp = sx(sizeof(int) * (ns + 1));
    S->pnr = sx(sizeof(int) * ns);
    S->pbase = sx(sizeof(size_t) * (ns + 1));
    size_t rtot = 0, ptot = 0;
    for (int k = 0; k < ns; k++) {
        int j0 = S->ss[k], w = S->ss[k + 1] - j0, sl = slast[k], nr = (sl - j0) + (S->cp[sl + 1] - S->cp[sl]);
        S->pnr[k] = nr;
        S->prp[k] = (int)rtot;
        rtot += nr;
        S->pbase[k] = ptot;
        ptot += (size_t)nr * w;
    }
    S->prp[ns] = (int)rtot;
    S->pbase[ns] = ptot;
    S->prow = sx(sizeof(int) * (rtot ? rtot : 1));
    for (int k = 0; k < ns; k++) {
        int j0 = S->ss[k], sl = slast[k], q = 0;
        for (int c = j0; c < sl; c++) S->prow[S->prp[k] + q++] = c;
        for (int t = S->cp[sl]; t < S->cp[sl + 1]; t++) S->prow[S->prp[k] + q++] = S->ci[t];
    }
    free(slast);
    /* 18 October 2026: the column-wise row indices end with the analysis (the supernodes' row lists replace them):
     * 4 bytes per entry of L (broyden25: 77 MB) */
    free(S->ci); S->ci = NULL;
    /* numeric storage on first use (4.22): the chordal conversion analyzes candidate
     * patterns only for their cost (two 2.1 GB arrays per candidate on case9241) */
    S->pan = S->pmat = NULL;
    S->pansz = ptot;
    S->relind = sx(sizeof(int) * m);
    S->snhead = sx(sizeof(int) * (ns + 1));
    S->snnext = sx(sizeof(int) * (ns + 1));
    S->snptr = sx(sizeof(int) * (ns + 1));
    S->wrk = sx(sizeof(double) * m);
    /* ---- assembly cache: panel offset of every lower-triangle panel position */
    S->hkey = NULL; S->hval = NULL; S->hmask = 0;     /* (offsets by binary search, sc_off) */
    S->acc = sx(sizeof(double) * m);
    S->acc2 = NULL;
    S->head = sx(sizeof(int) * m);
    S->next = sx(sizeof(int) * m);
    S->mark = sx(sizeof(int) * m);
    S->stack = sx(sizeof(int) * m);
    return S;
}

void schol_free(SChol *S) {
    if (!S) return;
    free(S->perm); free(S->iperm); free(S->cp); free(S->ci); free(S->mcp); free(S->mri); free(S->moff); free(S->mval); free(S->mrq); free(S->mdg); free(S->mrun); free(S->mrp); free(S->t2); free(S->sgn); free(S->sw);
    free(S->acc); free(S->acc2); free(S->head); free(S->next); free(S->mark); free(S->stack); free(S->tinv); free(S->tl);
    free(S->ss); free(S->snof); free(S->pbase); free(S->prow); free(S->prp); free(S->pnr);
    if (S->pan != S->pmat) free(S->pan);
    free(S->pmat); free(S->relind); free(S->relidx); free(S->upd); free(S->wrk);
    free(S->snhead); free(S->snnext); free(S->snptr); free(S->hkey); free(S->hval);
    free(S->mv_c); free(S->mv_zo); free(S->mv_z); free(S->mva); free(S->mv_za); free(S->mv2w); free(S->mv_z2); free(S->sch_ptr); free(S->sch_idx);
    free(S->ul_ptr); free(S->ul_j); free(S->ul_p); free(S->ul_a); free(S->spar); free(S->sfd); free(S->swk); free(S->ssub);
    free(S);
}

void schol_zero(SChol *S) {
    sc_ensure(S); S->mval_ok = 0;
    const size_t n = S->pansz, ch = 1 << 20, nc = (n + ch - 1) / ch;
    #pragma omp parallel for schedule(static) if (nc > 4 && !omp_in_parallel())
    for (size_t c = 0; c < nc; c++) { const size_t a = c * ch, b = a + ch < n ? a + ch : n; memset(S->pmat + a, 0, sizeof(double) * (b - a)); }
}

/* Load the matrix from a dense lower triangle M (original indices, column-major m x m,
 * entry (r, c) with r >= c at M[r + c m]). Used when the Schur complement is assembled
 * densely: the per-entry hashed writes of schol_add cost more than the factorization on
 * dense-ish patterns (moment-SOS image forms: 15-30% of the entries). */
void schol_gather_dense(SChol *S, const double *M) {
    sc_ensure(S);
    S->mval_ok = 0;
    const size_t m = (size_t)S->m;
    for (int k = 0; k < S->ns; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const int *rw = S->prow + S->prp[k];
        double *P = S->pmat + S->pbase[k];
        for (int cc = 0; cc < w; cc++) {
            const size_t c = (size_t)S->perm[j0 + cc];
            for (int q = cc; q < nr; q++) {
                const size_t r = (size_t)S->perm[rw[q]];
                P[(size_t)cc * nr + q] = r >= c ? M[r + c * m] : M[c + r * m];
            }
        }
    }
}

/* Accumulate a value at (r, c) of the matrix (original indices). Every panel position was
 * hashed during analysis, so this is a constant-time lookup instead of a binary search
 * through a factor column. */
void schol_add(SChol *S, size_t r, size_t c, double v) {
    sc_ensure(S);
    const size_t o = sc_off(S, S->iperm[r], S->iperm[c]);
    if (o != SIZE_MAX) { S->pmat[o] += v; if (S->mval_ok) S->mval_ok = 0; }   /* (no write of the shared field per entry) */
}

/* value at (r, c) of the assembled matrix (0 outside the pattern) */
double schol_get(const SChol *S, size_t r, size_t c) {
    sc_ensure(S);
    const size_t o = sc_off(S, S->iperm[r], S->iperm[c]);
    return o != SIZE_MAX ? S->pmat[o] : 0.0;
}

#define MV_P 32
static int g_mvold = -1;
/* y = M x (symmetric), original indexing, over the pattern of M */
double g_tmv[2], g_tmvg; long g_nmv[2], g_nmvg;   /* 4.26 debug: time and calls of schol_mv / schol_solve_tp */
static void schol_mv_(const SChol *S, const double *x, double *y);
void schol_mv(const SChol *S, const double *x, double *y) { double t0 = omp_get_wtime(); schol_mv_(S, x, y); g_tmv[0] += omp_get_wtime() - t0; g_nmv[0]++; }
/* runs of consecutive row indices: dense-ish patterns (moment-SOS Schur complements with a few wide supernodes) turn
 * the gather/scatter loop of the products into dot/axpy on contiguous segments. 18 October 2026: decided when the
 * pattern is attached (was: at the first product), and the row indices of M freed when the runs are taken (no product
 * reads them then: 4 bytes per entry of M, before the factor's storage is allocated) */
static void mv_runs(const SChol *S) {
    const int m = S->m;
    /* runs of consecutive row indices: dense-ish patterns (moment-SOS Schur complements
     * with a few wide supernodes) turn the gather/scatter loop into dot/axpy on
     * contiguous segments */
    SChol *Sm = (SChol *)S;
    const size_t nq = (size_t)S->mcp[m];
    long nr = 0;
    for (int j = 0; j < m; j++)
        for (int q = S->mcp[j] + 1; q < S->mcp[j + 1]; q++)
            if (q == S->mcp[j] + 1 || S->mri[q] != S->mri[q - 1] + 1) nr++;
    const double offd = (double)nq - m;
    if (getenv("BRISK_RUNDBG")) printf("   [schol_mv: %ld runs, %.1f entries per run]\n", nr, nr ? offd / nr : 0.0);
    static double runmin = -1;
    if (runmin < 0) { const char *e = getenv("BRISK_RUNMIN"); runmin = e ? atof(e) : 6.0; }
    if (nr > 0 && offd / nr >= runmin) {
        Sm->mrun = sx(sizeof(int) * 3 * (nr + 1));
        Sm->mrp = sx(sizeof(int) * (m + 1));
        int t = 0;
        for (int j = 0; j < m; j++) {
            Sm->mrp[j] = t;
            for (int q = S->mcp[j] + 1; q < S->mcp[j + 1]; q++) {
                if (q == S->mcp[j] + 1 || S->mri[q] != S->mri[q - 1] + 1) { Sm->mrun[3 * t] = q; Sm->mrun[3 * t + 1] = S->mri[q]; Sm->mrun[3 * t + 2] = 0; t++; }
                Sm->mrun[3 * (t - 1) + 2]++;
            }
        }
        Sm->mrp[m] = t;
        Sm->nrun = t;
        Sm->run_state = 1;
        /* 17 October 2026: when every run is contiguous in pmat (consecutive rows of M are consecutive rows of
         * the panel), the products read pmat directly and the copy mval, a gather of the whole pattern after
         * every factorization (broyden25: 0.1 s each, 2.3 s of 3.9 s of residual products), is not made.
         * The values are the same, so are the products. BRISK_MVDIRECT=0 off. */
        {   const char *e = getenv("BRISK_MVDIRECT"); int ok = !(e && atoi(e) == 0);
            for (int r = 0; r < t && ok; r++) { const size_t q = (size_t)Sm->mrun[3 * r]; const int len = Sm->mrun[3 * r + 2]; if (S->moff[q + len - 1] - S->moff[q] != (size_t)(len - 1)) ok = 0; }
            if (ok) {
                Sm->mrq = sx(sizeof(size_t) * ((size_t)t + 1)); Sm->mdg = sx(sizeof(size_t) * ((size_t)m + 1));
                for (int r = 0; r < t; r++) Sm->mrq[r] = S->moff[Sm->mrun[3 * r]];
                for (int j = 0; j < m; j++) Sm->mdg[j] = S->moff[S->mcp[j]];
                Sm->mdirect = 1;
                if (S->pan != S->pmat) Sm->mval_ok = 1;
            }
        }
    } else Sm->run_state = -1;
    if (S->run_state == 1) { free(((SChol *)S)->mri); ((SChol *)S)->mri = NULL; }
}
static void mv_prepare(const SChol *S) {
    if (g_mvold < 0) g_mvold = getenv("BRISK_MVOLD") != NULL;
    sc_ensure(S);
    const int m = S->m;
    if (S->run_state == 0) mv_runs(S);
    if (!S->mval_ok && S->mdirect && S->pan != S->pmat) ((SChol *)S)->mval_ok = 1;   /* (the runs read pmat itself) */
    if (!S->mval_ok) {
        SChol *Sm = (SChol *)S;
        const size_t nq = (size_t)S->mcp[m];
        const double tg0 = omp_get_wtime();
        if (!Sm->mval) Sm->mval = sx(sizeof(double) * (nq + 1));
        for (size_t q = 0; q < nq; q++) Sm->mval[q] = S->pmat[S->moff[q]];
        Sm->mval_ok = 1;
        g_tmvg += omp_get_wtime() - tg0; g_nmvg++;
    }
    if (S->mcp[m] >= 200000 && !g_mvold) {
        SChol *Sm = (SChol *)S;
        if (!Sm->mv_c) {
            const int np = MV_P;
            Sm->mv_c = sx(sizeof(int) * (np + 1)); Sm->mv_zo = sx(sizeof(size_t) * (np + 1));
            const double tot = (double)S->mcp[m];
            int p = 0; Sm->mv_c[0] = 0;
            for (int j = 0; j < m && p < np - 1; j++) if ((double)S->mcp[j + 1] >= tot * (p + 1) / np) Sm->mv_c[++p] = j + 1;
            while (p < np) Sm->mv_c[++p] = m;
            for (int q = 0; q < np; q++) Sm->mv_zo[q + 1] = Sm->mv_zo[q] + (size_t)(m - Sm->mv_c[q]);
            Sm->mv_z = sx(sizeof(double) * (Sm->mv_zo[np] + 1));
            Sm->mv_np = np;
        }
    }
}
static void schol_mv_(const SChol *S, const double *x, double *y) {
    mv_prepare(S);
    const int m = S->m;
    double *xp = S->wrk, *yp = S->acc;
    for (int i = 0; i < m; i++) { xp[i] = x[S->perm[i]]; yp[i] = 0; }
    const double *mv = S->mval;
    const int dir = S->mdirect && S->pan != S->pmat; const double *pm = S->pmat;   /* (see mv_prepare) */
    if (S->mcp[m] >= 200000 && !g_mvold) {
        /* 4.24: fixed column chunks (independent of the thread count): each chunk gathers
         * its own rows and scatters the transposed part into a private tail vector; the
         * results are summed per row in chunk order */
        const int np = S->mv_np, runs = S->run_state == 1;
        #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
        for (int p = 0; p < np; p++) {
            const int c0 = S->mv_c[p], c1 = S->mv_c[p + 1];
            double *z = S->mv_z + S->mv_zo[p] - c0;          /* z[i], i >= c0 */
            memset(z + c0, 0, sizeof(double) * (size_t)(m - c0));
            for (int j = c0; j < c1; j++) {
                const double xj = xp[j];
                double sj = (dir ? pm[S->mdg[j]] : mv[S->mcp[j]]) * xj;
                if (runs) {
                    for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                        const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                        const double *vq = dir ? pm + S->mrq[t] : mv + q, *xq = xp + i0;
                        double *zq = z + i0, acc = 0;
                        #pragma omp simd reduction(+:acc)
                        for (int e = 0; e < len; e++) { acc += vq[e] * xq[e]; zq[e] += vq[e] * xj; }
                        sj += acc;
                    }
                } else {
                    for (int q = S->mcp[j] + 1; q < S->mcp[j + 1]; q++) { const double v = mv[q]; const int i = S->mri[q]; sj += v * xp[i]; z[i] += v * xj; }
                }
                yp[j] = sj;
            }
        }
        #pragma omp parallel for schedule(static) if (!omp_in_parallel())
        for (int i = 0; i < m; i++) {
            double acc = yp[i];
            for (int p = 0; p < np && S->mv_c[p] <= i; p++) acc += S->mv_z[S->mv_zo[p] + (size_t)(i - S->mv_c[p])];
            y[S->perm[i]] = acc;
        }
        return;
    }
    if (S->run_state == 1) {
        for (int j = 0; j < m; j++) {
            const double xj = xp[j];
            double sj = (dir ? pm[S->mdg[j]] : mv[S->mcp[j]]) * xj;
            for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                const double *vq = dir ? pm + S->mrq[t] : mv + q;
                const double *xq = xp + i0;
                double *yq = yp + i0;
                double acc = 0;
                #pragma omp simd reduction(+:acc)
                for (int e = 0; e < len; e++) { acc += vq[e] * xq[e]; yq[e] += vq[e] * xj; }
                sj += acc;
            }
            yp[j] += sj;
        }
        for (int i = 0; i < m; i++) y[S->perm[i]] = yp[i];
        return;
    }
    for (int j = 0; j < m; j++) {
        const int q0 = S->mcp[j], q1 = S->mcp[j + 1];
        const double xj = xp[j];
        double sj = mv[q0] * xj;                            /* diagonal first */
        for (int q = q0 + 1; q < q1; q++) {
            const double v = mv[q];
            const int i = S->mri[q];
            sj += v * xp[i];
            yp[i] += v * xj;
        }
        yp[j] += sj;
    }
    for (int i = 0; i < m; i++) y[S->perm[i]] = yp[i];
}

/* 18 October 2026: y1 = M x1 and y2 = M x2 in one pass over M. The product is bound by the memory traffic of M
 * (broyden25: 162 MB, 7.5 ms at the host's 24 GB/s), and the multi-column checks and PCG steps of the solves took
 * one pass per column. The same chunks and the same operations per column as schol_mv. */
void schol_mv2(const SChol *S, const double *x1, const double *x2, double *y1, double *y2) {
    const int m = S->m;
    if (!(S->mcp[m] >= 200000 && !g_mvold)) { schol_mv(S, x1, y1); schol_mv(S, x2, y2); return; }
    double t0 = omp_get_wtime();
    SChol *Sm = (SChol *)S;
    mv_prepare(S);
    if (!Sm->mv2w) Sm->mv2w = sx(sizeof(double) * 2 * ((size_t)m + 1));
    const int np = S->mv_np;
    if (!Sm->mv_z2) Sm->mv_z2 = sx(sizeof(double) * (S->mv_zo[np] + 1));
    double *xp = S->wrk, *yp = S->acc, *xq2 = Sm->mv2w, *yp2 = Sm->mv2w + m + 1;
    for (int i = 0; i < m; i++) { xp[i] = x1[S->perm[i]]; xq2[i] = x2[S->perm[i]]; yp[i] = 0; yp2[i] = 0; }
    const double *mv = S->mval;
    const int dir = S->mdirect && S->pan != S->pmat; const double *pm = S->pmat;   /* (see mv_prepare) */
    const int runs = S->run_state == 1;
    #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
    for (int p = 0; p < np; p++) {
        const int c0 = S->mv_c[p], c1 = S->mv_c[p + 1];
        double *z = S->mv_z + S->mv_zo[p] - c0, *z2 = S->mv_z2 + S->mv_zo[p] - c0;
        memset(z + c0, 0, sizeof(double) * (size_t)(m - c0));
        memset(z2 + c0, 0, sizeof(double) * (size_t)(m - c0));
        for (int j = c0; j < c1; j++) {
            const double xj = xp[j], xj2 = xq2[j];
            const double dj = dir ? pm[S->mdg[j]] : mv[S->mcp[j]];
            double sj = dj * xj, sj2 = dj * xj2;
            if (runs) {
                for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                    const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                    const double *vq = dir ? pm + S->mrq[t] : mv + q, *xq = xp + i0, *xr = xq2 + i0;
                    double *zq = z + i0, *zr = z2 + i0, acc = 0, acc2 = 0;
                    #pragma omp simd reduction(+:acc,acc2)
                    for (int e = 0; e < len; e++) { const double v = vq[e]; acc += v * xq[e]; zq[e] += v * xj; acc2 += v * xr[e]; zr[e] += v * xj2; }
                    sj += acc; sj2 += acc2;
                }
            } else {
                for (int q = S->mcp[j] + 1; q < S->mcp[j + 1]; q++) {
                    const double v = mv[q]; const int i = S->mri[q];
                    sj += v * xp[i]; z[i] += v * xj; sj2 += v * xq2[i]; z2[i] += v * xj2;
                }
            }
            yp[j] = sj; yp2[j] = sj2;
        }
    }
    #pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (int i = 0; i < m; i++) {
        double acc = yp[i], acc2 = yp2[i];
        for (int p = 0; p < np && S->mv_c[p] <= i; p++) {
            acc += S->mv_z[S->mv_zo[p] + (size_t)(i - S->mv_c[p])];
            acc2 += S->mv_z2[S->mv_zo[p] + (size_t)(i - S->mv_c[p])];
        }
        y1[S->perm[i]] = acc; y2[S->perm[i]] = acc2;
    }
    g_tmv[0] += omp_get_wtime() - t0; g_nmv[0] += 2;
}

/* 4.27: y = M x and ya = |M| |x| in one pass over M (the same memory traffic as schol_mv;
 * the residual check of a solve gets the rounding floor u |M||x| of its residual with it).
 * The same chunking as schol_mv, so the result does not depend on the thread count. */
void schol_mv_abs(const SChol *S, const double *x, double *y, double *ya) {
    double t0 = omp_get_wtime();
    SChol *Sm = (SChol *)S;
    mv_prepare(S);
    const int m = S->m;
    if (!Sm->mva) Sm->mva = sx(sizeof(double) * 2 * ((size_t)m + 1));
    double *xp = S->wrk, *yp = S->acc, *xa = Sm->mva, *ap = Sm->mva + m + 1;
    for (int i = 0; i < m; i++) { xp[i] = x[S->perm[i]]; xa[i] = fabs(xp[i]); yp[i] = 0; ap[i] = 0; }
    const double *mv = S->mval;
    const int dir = S->mdirect && S->pan != S->pmat; const double *pm = S->pmat;   /* (see mv_prepare) */
    const int runs = S->run_state == 1;
    if (S->mcp[m] >= 200000 && !g_mvold) {
        const int np = S->mv_np;
        if (!Sm->mv_za) Sm->mv_za = sx(sizeof(double) * (S->mv_zo[np] + 1));
        #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
        for (int p = 0; p < np; p++) {
            const int c0 = S->mv_c[p], c1 = S->mv_c[p + 1];
            double *z = S->mv_z + S->mv_zo[p] - c0, *za = S->mv_za + S->mv_zo[p] - c0;
            memset(z + c0, 0, sizeof(double) * (size_t)(m - c0));
            memset(za + c0, 0, sizeof(double) * (size_t)(m - c0));
            for (int j = c0; j < c1; j++) {
                const double xj = xp[j], aj = xa[j];
                const double dj = dir ? pm[S->mdg[j]] : mv[S->mcp[j]]; double sj = dj * xj, tj = fabs(dj) * aj;
                if (runs) {
                    for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                        const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                        const double *vq = dir ? pm + S->mrq[t] : mv + q, *xq = xp + i0, *aq = xa + i0;
                        double *zq = z + i0, *zaq = za + i0, acc = 0, acca = 0;
                        #pragma omp simd reduction(+:acc,acca)
                        for (int e = 0; e < len; e++) {
                            const double v = vq[e], av = fabs(v);
                            acc += v * xq[e]; zq[e] += v * xj;
                            acca += av * aq[e]; zaq[e] += av * aj;
                        }
                        sj += acc; tj += acca;
                    }
                } else {
                    for (int q = S->mcp[j] + 1; q < S->mcp[j + 1]; q++) {
                        const double v = mv[q], av = fabs(v); const int i = S->mri[q];
                        sj += v * xp[i]; z[i] += v * xj; tj += av * xa[i]; za[i] += av * aj;
                    }
                }
                yp[j] = sj; ap[j] = tj;
            }
        }
        #pragma omp parallel for schedule(static) if (!omp_in_parallel())
        for (int i = 0; i < m; i++) {
            double acc = yp[i], acca = ap[i];
            for (int p = 0; p < np && S->mv_c[p] <= i; p++) {
                acc += S->mv_z[S->mv_zo[p] + (size_t)(i - S->mv_c[p])];
                acca += S->mv_za[S->mv_zo[p] + (size_t)(i - S->mv_c[p])];
            }
            y[S->perm[i]] = acc; ya[S->perm[i]] = acca;
        }
    } else {
        for (int j = 0; j < m; j++) {
            const double xj = xp[j], aj = xa[j];
            const double dj = dir ? pm[S->mdg[j]] : mv[S->mcp[j]]; double sj = dj * xj, tj = fabs(dj) * aj;
            if (runs) {
                for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                    const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                    const double *vq = dir ? pm + S->mrq[t] : mv + q, *xq = xp + i0, *aq = xa + i0;
                    double *yq = yp + i0, *apq = ap + i0, acc = 0, acca = 0;
                    #pragma omp simd reduction(+:acc,acca)
                    for (int e = 0; e < len; e++) {
                        const double v = vq[e], av = fabs(v);
                        acc += v * xq[e]; yq[e] += v * xj;
                        acca += av * aq[e]; apq[e] += av * aj;
                    }
                    sj += acc; tj += acca;
                }
            } else {
                for (int q = S->mcp[j] + 1; q < S->mcp[j + 1]; q++) {
                    const double v = mv[q], av = fabs(v); const int i = S->mri[q];
                    sj += v * xp[i]; yp[i] += v * xj; tj += av * xa[i]; ap[i] += av * aj;
                }
            }
            yp[j] += sj; ap[j] += tj;
        }
        for (int i = 0; i < m; i++) { y[S->perm[i]] = yp[i]; ya[S->perm[i]] = ap[i]; }
    }
    g_tmv[0] += omp_get_wtime() - t0; g_nmv[0]++;
}

/* Supernodal left-looking Cholesky: each supernode is factored as a dense panel, and the
 * updates from earlier supernodes are one dgemm each, scattered into the panel. */
static size_t g_small_upd = (size_t)-1;
static int g_nocontig_neg = -1;
/* -1 if every column of the supernode is negative, +1 if every one is positive, 0 mixed */
static int panel_sign_uniform(const SChol *S, int j0, int w) {
    if (!S->sgn) return 1;
    int neg = 0;
    for (int c = 0; c < w; c++) neg += S->sgn[j0 + c] < 0;
    return neg == w ? -1 : neg == 0 ? 1 : 0;
}
static int g_fstat = -1; static long g_nupd = 0, g_nsmall = 0; static double g_updfl = 0;
void schol_fstat_print(void) { if (g_fstat > 0) printf("   [factor updates: %ld (%ld with tot*a < 256), %.2e update flops]\n", g_nupd, g_nsmall, g_updfl); }


/* (sparsechol-dev) The panel kernels of the factorization: B := B L^{-T} (sc_trsm: B m x n, L n x n lower) and the
 * Cholesky factor of a diagonal block (sc_potrf, LAPACK's info), by recursion on halves of the columns (the first
 * half a multiple of 8): the halves are coupled by one dgemm (dsyrk), the leaves of at most SC_TL (SC_PL) columns are
 * column loops in C. OpenBLAS's dtrsm and dpotrf run at 15-20 GF/s on the panels of the sparse factor (8 to 132
 * columns, a few hundred rows below; at n = 132 for any number of rows) where its dgemm runs at 50-70, and a node of
 * a few columns paid two library calls for a few hundred flops (case2000_goc: 12,000 of its 13,500 nodes have at most
 * 8 columns). On panels of 4 to 132 columns the two together take 0.40-0.47 of the time of dpotrf and dtrsm
 * (bench/sparsechol_dev/panel.c). It is the blocked Cholesky with its operations in another order, backward stable as
 * the column loop; the blocks depend on the sizes alone, so the results are the same at any thread count. The leaves
 * scale by the reciprocal of the pivot, as OpenBLAS's kernels do. Measured on the sets with the same classes (and one
 * more OPTIMAL among the occupation measures), the factorizations 5-34 % faster (REPORT_sparsechol_1). The dense
 * Cholesky's task graph and panel loop use them too (in both precisions, sc_*_s; the dense Schur factorizations of
 * the structural and occupation-measure sets 10-15 % faster). BRISK_PANELLIB=1: dpotrf and dtrsm as before. */
#define SC_TL 4                 /* (sc_trsm's leaf is written out for at most four columns) */
_Static_assert(SC_TL == 4, "sc_trsm's leaf handles 1 to 4 columns");
#define SC_PL 8
static int g_panellib = -1;
static inline int sc_panellib(void) { if (g_panellib < 0) g_panellib = getenv("BRISK_PANELLIB") != NULL; return g_panellib; }
#define SC_KERNELS(SUF, T, GEMM, SYRK, SQRT, TRSM, POTRF)                                                                         \
static void sc_trsm##SUF(int m, int n, const T *L, int ldl, T *B, int ldb) {                                          \
    if (m <= 0 || n <= 0) return;                                                                                    \
    if (sc_panellib()) { const T one = 1; TRSM("R", "L", "T", "N", &m, &n, &one, L, &ldl, B, &ldb); return; }       \
    if (n <= SC_TL) {                   /* (a row's n columns in registers, one pass; each entry the operations of    \
                                         * the column loop in its order: bc -= L_ck bk over k ascending, then *= 1/L_cc) */ \
        T *restrict b0 = B;                                                                                          \
        const T i0 = (T)1 / L[0];                                                                                    \
        if (n == 1) { for (int r = 0; r < m; r++) b0[r] *= i0; return; }                                             \
        T *restrict b1 = B + ldb;                                                                                    \
        const T l10 = L[1], i1 = (T)1 / L[1 + ldl];                                                                  \
        if (n == 2) { for (int r = 0; r < m; r++) { const T x0 = b0[r] * i0; T x1 = b1[r]; x1 -= l10 * x0; b0[r] = x0; b1[r] = x1 * i1; } return; } \
        T *restrict b2 = B + 2 * (size_t)ldb;                                                                        \
        const T l20 = L[2], l21 = L[2 + ldl], i2 = (T)1 / L[2 + 2 * (size_t)ldl];                                    \
        if (n == 3) { for (int r = 0; r < m; r++) { const T x0 = b0[r] * i0; T x1 = b1[r]; x1 -= l10 * x0; x1 *= i1; T x2 = b2[r]; x2 -= l20 * x0; x2 -= l21 * x1; b0[r] = x0; b1[r] = x1; b2[r] = x2 * i2; } return; } \
        T *restrict b3 = B + 3 * (size_t)ldb;                                                                        \
        const T l30 = L[3], l31 = L[3 + ldl], l32 = L[3 + 2 * (size_t)ldl], i3 = (T)1 / L[3 + 3 * (size_t)ldl];      \
        for (int r = 0; r < m; r++) {                                                                                \
            const T x0 = b0[r] * i0;                                                                                 \
            T x1 = b1[r]; x1 -= l10 * x0; x1 *= i1;                                                                  \
            T x2 = b2[r]; x2 -= l20 * x0; x2 -= l21 * x1; x2 *= i2;                                                  \
            T x3 = b3[r]; x3 -= l30 * x0; x3 -= l31 * x1; x3 -= l32 * x2;                                            \
            b0[r] = x0; b1[r] = x1; b2[r] = x2; b3[r] = x3 * i3;                                                     \
        }                                                                                                            \
        return;                                                                                                      \
    }                                                                                                                \
    int n1 = ((n + 1) / 2 + 7) / 8 * 8; if (n1 >= n) n1 = n / 2;                                                     \
    const int n2 = n - n1; const T one = 1, mone = -1;                                                               \
    sc_trsm##SUF(m, n1, L, ldl, B, ldb);                                                                             \
    GEMM("N", "T", &m, &n2, &n1, &mone, B, &ldb, L + n1, &ldl, &one, B + (size_t)n1 * ldb, &ldb);                    \
    sc_trsm##SUF(m, n2, L + n1 + (size_t)n1 * ldl, ldl, B + (size_t)n1 * ldb, ldb);                                  \
}                                                                                                                    \
static int sc_potrf##SUF(int n, T *A, int lda) {                                                                     \
    if (n <= 0) return 0;                                                                                            \
    if (sc_panellib()) { int info = 0; POTRF("L", &n, A, &lda, &info); return info; }                               \
    if (n <= SC_PL) {                                                                                                \
        for (int c = 0; c < n; c++) {                                                                                \
            T *restrict ac = A + (size_t)c * lda;                                                                    \
            for (int k = 0; k < c; k++) { const T f = A[c + (size_t)k * lda]; const T *restrict ak = A + (size_t)k * lda; for (int r = c; r < n; r++) ac[r] -= f * ak[r]; } \
            const T d = ac[c];                                                                                       \
            if (!(d > 0)) return c + 1;                                                                              \
            const T l = SQRT(d), il = (T)1 / l;                                                                      \
            ac[c] = l;                                                                                               \
            for (int r = c + 1; r < n; r++) ac[r] *= il;                                                             \
        }                                                                                                            \
        return 0;                                                                                                    \
    }                                                                                                                \
    int n1 = ((n + 1) / 2 + 7) / 8 * 8; if (n1 >= n) n1 = n / 2;                                                     \
    const int n2 = n - n1; const T one = 1, mone = -1;                                                               \
    int info = sc_potrf##SUF(n1, A, lda);                                                                            \
    if (info) return info;                                                                                           \
    sc_trsm##SUF(n2, n1, A, lda, A + n1, lda);                                                                       \
    SYRK("L", "N", &n2, &n1, &mone, A + n1, &lda, &one, A + n1 + (size_t)n1 * lda, &lda);                            \
    info = sc_potrf##SUF(n2, A + n1 + (size_t)n1 * lda, lda);                                                        \
    return info ? info + n1 : 0;                                                                                     \
}
SC_KERNELS(, double, BL(dgemm_), BL(dsyrk_), sqrt, BL(dtrsm_), BL(dpotrf_))

/* Cholesky of the w x w diagonal block (lower, leading dimension ld) with replacement of the
 * pivots that are not positive (the dynamic regularization of the signed factorization) */
static int chol_dyn_block(double *P, int ld, int w, double piv) {
    int nrep = 0;
    for (int c = 0; c < w; c++) {
        double *col = P + (size_t)c * ld;
        for (int k2 = 0; k2 < c; k2++) {
            const double *ck = P + (size_t)k2 * ld;
            const double f = ck[c];
            if (f == 0) continue;
            for (int r = c; r < w; r++) col[r] -= f * ck[r];
        }
        double d = col[c];
        if (!(d > 0)) { d = piv; nrep++; }
        const double l = sqrt(d), il = 1.0 / l;
        col[c] = l;
        for (int r = c + 1; r < w; r++) col[r] *= il;
    }
    return nrep;
}
/* dpotrf of the diagonal block of a panel; with dynpiv a failed factorization is repeated from a
 * copy with replaced pivots. 0: done */
static int panel_potrf_dyn(SChol *S, double *Pk, int nr, int w, double **buf, size_t *cap) {
    int info = 0;
    if (!(S->dynpiv > 0)) { info = sc_potrf(w, Pk, nr); return info != 0; }
    const size_t nw = (size_t)w * w;
    if (nw > *cap) { *cap = 2 * nw; free(*buf); *buf = (double *)malloc(sizeof(double) * *cap); }
    double *T = *buf;
    for (int c = 0; c < w; c++) memcpy(T + (size_t)c * w + c, Pk + (size_t)c * nr + c, sizeof(double) * (size_t)(w - c));
    info = sc_potrf(w, Pk, nr);
    if (info == 0) return 0;
    for (int c = 0; c < w; c++) memcpy(Pk + (size_t)c * nr + c, T + (size_t)c * w + c, sizeof(double) * (size_t)(w - c));
    const int nrep = chol_dyn_block(Pk, nr, w, S->dynpiv);
#pragma omp atomic
    S->ndyn += nrep;
    return 0;
}


/* Signed Cholesky P = L S L' of the w x w diagonal block of a panel with columns of both signs
 * (lower triangle, leading dimension ld; sg: the signs), by blocks: the diagonal block of SB columns
 * by the column loop, the rows below by dtrsm and a sign scaling, the trailing update by two dsyrk
 * (the positive and the negative columns). A pivot of the wrong sign is replaced by sign * dynpiv
 * when dynpiv > 0 (counted in *nrep); otherwise 1 is returned. */
#define SIGNED_SB 48         /* (17 October 2026: 96, 128 and 192 measured, no gain on lasso_5000) */
/* 17 October 2026: a team of threads for the large nodes of a factorization whose tree runs sequentially (the cone
 * solver, which stays on one thread: schol_set_panel_threads; 0 or 1: none). The work it is given is cut into
 * fixed tiles, so the arithmetic does not depend on its size. */
static int g_panel_nt = 0;
void schol_set_panel_threads(int nt) { g_panel_nt = nt; }
int schol_panel_threads(void) { return g_panel_nt > 1 ? g_panel_nt : 1; }
static inline int panel_team(double work) {
#ifdef _OPENMP
    return (g_panel_nt > 1 && !omp_in_parallel() && work >= 4e6) ? g_panel_nt : 1;
#else
    (void)work; return 1;
#endif
}
#define SIGNED_TR 256
static int signed_chol_blocked(double *P, int ld, int w, const signed char *sg, double dynpiv, int *nrep) {
    double *buf = NULL;
    if (w > 2 * SIGNED_SB && dynpiv > 0) buf = (double *)malloc(sizeof(double) * (size_t)w * SIGNED_SB);   /* (blocks for the cone solver; the column loop, as before, otherwise) */
    const int sb = buf ? SIGNED_SB : w;
    for (int j0 = 0; j0 < w; j0 += sb) {
        const int jb = w - j0 < sb ? w - j0 : sb, j1 = j0 + jb;
        for (int c = j0; c < j1; c++) {
            double *col = P + (size_t)c * ld;
            for (int k2 = j0; k2 < c; k2++) {
                const double *ck = P + (size_t)k2 * ld;
                const double f = ck[c] * sg[k2];
                if (f == 0) continue;
                for (int r = c; r < j1; r++) col[r] -= f * ck[r];
            }
            double d = col[c]; const double sc = sg[c];
            if (!(d * sc > 0)) { if (!(dynpiv > 0)) { free(buf); return 1; } d = sc * dynpiv; (*nrep)++; }
            const double l = sqrt(fabs(d)), il = 1.0 / (l * sc);
            col[c] = l;
            for (int r = c + 1; r < j1; r++) col[r] *= il;
        }
        const int below = w - j1;
        if (below <= 0) continue;
        double *L11 = P + (size_t)j0 * ld + j0, *P21 = P + (size_t)j0 * ld + j1;
        double *P22 = P + (size_t)j1 * ld + j1;
        const double mone = -1.0, one = 1.0;
        if (buf) {
            /* 17 October 2026: the rows below and the trailing update by fixed tiles of SIGNED_TR rows (by tile pairs
             * below the diagonal), on the team of panel_team for a large node; the same arithmetic for any team */
            const int nti = (below + SIGNED_TR - 1) / SIGNED_TR, tn = panel_team((double)below * below * jb);
            int npos = 0, nneg = 0;
            for (int c = 0; c < jb; c++) { if (sg[j0 + c] < 0) nneg++; else npos++; }
            #pragma omp parallel for schedule(static) num_threads(tn) if (tn > 1 && nti > 1)
            for (int t = 0; t < nti; t++) {
                const int r0 = t * SIGNED_TR, rr = below - r0 < SIGNED_TR ? below - r0 : SIGNED_TR;
                sc_trsm(rr, jb, L11, ld, P21 + r0, ld);
                for (int c = 0; c < jb; c++) if (sg[j0 + c] < 0) { double *col = P21 + (size_t)c * ld + r0; for (int r = 0; r < rr; r++) col[r] = -col[r]; }
                if (npos != jb && nneg != jb) {          /* the positive and the negative columns side by side in buf */
                    int a = 0, b = npos;
                    for (int c = 0; c < jb; c++) memcpy(buf + (size_t)(sg[j0 + c] > 0 ? a++ : b++) * below + r0, P21 + (size_t)c * ld + r0, sizeof(double) * (size_t)rr);
                }
            }
            const int npair = nti * (nti + 1) / 2;
            #pragma omp parallel for schedule(dynamic, 1) num_threads(tn) if (tn > 1 && nti > 1)
            for (int q = 0; q < npair; q++) {
                int ti = 0; while ((ti + 1) * (ti + 2) / 2 <= q) ti++;
                const int tj = q - ti * (ti + 1) / 2;
                const int i0 = ti * SIGNED_TR, ni = below - i0 < SIGNED_TR ? below - i0 : SIGNED_TR;
                const int c0 = tj * SIGNED_TR, nc = below - c0 < SIGNED_TR ? below - c0 : SIGNED_TR;
                double *C = P22 + (size_t)c0 * ld + i0;
                const double *Pp = npos == jb || nneg == jb ? P21 : buf, *Pn = npos == jb || nneg == jb ? P21 : buf + (size_t)npos * below;
                const int lp = npos == jb || nneg == jb ? ld : below;
                const int kp = nneg == jb ? 0 : npos, kn = npos == jb ? 0 : nneg;
                if (ti == tj) {
                    if (kp) BL(dsyrk_)("L", "N", &ni, &kp, &mone, Pp + i0, &lp, &one, C, &ld);
                    if (kn) BL(dsyrk_)("L", "N", &ni, &kn, &one, Pn + i0, &lp, &one, C, &ld);
                } else {
                    if (kp) BL(dgemm_)("N", "T", &ni, &nc, &kp, &mone, Pp + i0, &lp, Pp + c0, &lp, &one, C, &ld);
                    if (kn) BL(dgemm_)("N", "T", &ni, &nc, &kn, &one, Pn + i0, &lp, Pn + c0, &lp, &one, C, &ld);
                }
            }
            continue;
        }
        sc_trsm(below, jb, L11, ld, P21, ld);
        int npos = 0, nneg = 0;
        for (int c = 0; c < jb; c++) {
            double *col = P21 + (size_t)c * ld;
            if (sg[j0 + c] < 0) { for (int r = 0; r < below; r++) col[r] = -col[r]; nneg++; } else npos++;
        }
        if (npos == jb) BL(dsyrk_)("L", "N", &below, &jb, &mone, P21, &ld, &one, P22, &ld);
        else if (nneg == jb) BL(dsyrk_)("L", "N", &below, &jb, &one, P21, &ld, &one, P22, &ld);
        else {
            int a = 0;
            for (int c = 0; c < jb; c++) if (sg[j0 + c] > 0) memcpy(buf + (size_t)(a++) * below, P21 + (size_t)c * ld, sizeof(double) * (size_t)below);
            BL(dsyrk_)("L", "N", &below, &npos, &mone, buf, &below, &one, P22, &ld);
            a = 0;
            for (int c = 0; c < jb; c++) if (sg[j0 + c] < 0) memcpy(buf + (size_t)(a++) * below, P21 + (size_t)c * ld, sizeof(double) * (size_t)below);
            BL(dsyrk_)("L", "N", &below, &nneg, &one, buf, &below, &one, P22, &ld);
        }
    }
    free(buf);
    return 0;
}

/* (sparsechol-dev) the left-looking step of the column loops of the tiny-pivot mode: col[r] -= f_k P[r + k ld] over
 * the earlier columns k in [k0, c) with f_k = P[c + k ld] != 0, in ascending order, for r in [r0, r1); four columns per
 * pass over col. Each entry receives the same operations in the same order as in the loop column by column (the
 * same bits), with a quarter of the loads and stores of col (the loop was 4 % of the LP pds-20). */
static inline void sc_colupd(double *restrict col, const double *P, int ld, int k0, int c, int r0, int r1) {
    const double *ck[4]; double f[4]; int nk = 0;
    for (int k = k0; k < c; k++) {
        const double *p = P + (size_t)k * ld; const double fk = p[c];
        if (fk == 0) continue;
        ck[nk] = p; f[nk] = fk;
        if (++nk == 4) {
            const double *a0 = ck[0], *a1 = ck[1], *a2 = ck[2], *a3 = ck[3]; const double f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3];
            for (int r = r0; r < r1; r++) { double t = col[r]; t -= f0 * a0[r]; t -= f1 * a1[r]; t -= f2 * a2[r]; t -= f3 * a3[r]; col[r] = t; }
            nk = 0;
        }
    }
    for (int q = 0; q < nk; q++) { const double *a = ck[q]; const double fq = f[q]; for (int r = r0; r < r1; r++) col[r] -= fq * a[r]; }
}
static int g_tinyloop = -1;          /* BRISK_TINYLOOP: the column loop for every width (read once) */

/* Blocked Cholesky of a column panel (nr x w, leading dimension ld, lower part used) with
 * tiny-pivot replacement (4.20): a pivot that has lost all but tinypiv of its original
 * diagonal d0[c] becomes 1e128 and its column of L is zeroed. Diagonal blocks of nb
 * columns by the checked column loop, the rows below by dtrsm, the trailing columns by one
 * dgemm: BLAS-3 speed instead of the column loop over the whole panel (roa_vdp_d8_I, 4
 * supernodes of up to 2,000 columns: that loop was 38% of the run). Returns the number
 * of replaced pivots.                                                                 */
int chol_tiny_blocked(int nr, int w, double *P, int ld, const double *d0, int d0stride, double tinypiv) {
    const int nb = 64;
    int ntiny = 0;
    for (int c0 = 0; c0 < w; c0 += nb) {
        const int b = w - c0 < nb ? w - c0 : nb;
        /* diagonal block: rows c0 .. c0+b, columns c0 .. c0+b (already updated by the
         * earlier blocks) */
        for (int c = c0; c < c0 + b; c++) {
            double *col = P + (size_t)c * ld;
            sc_colupd(col, P, ld, c0, c, c, c0 + b);
            double d = col[c];
            const double dd = fabs(d0[(size_t)c * d0stride]);
            int tiny = 0;
            if (!(d > tinypiv * dd) || !(d > 0)) { d = 1e128; ntiny++; tiny = 1; }
            const double l = sqrt(d), il = 1.0 / l;
            col[c] = l;
            if (tiny) { for (int r = c + 1; r < c0 + b; r++) col[r] = 0; }
            else for (int r = c + 1; r < c0 + b; r++) col[r] *= il;
        }
        const int below = nr - (c0 + b);
        if (below > 0) {
            sc_trsm(below, b, P + (size_t)c0 * ld + c0, ld, P + (size_t)c0 * ld + c0 + b, ld);
            /* columns of dropped pivots: exactly zero below (as the column loop gave) */
            for (int c = c0; c < c0 + b; c++)
                if (P[(size_t)c * ld + c] >= 1e63) { double *col = P + (size_t)c * ld + c0 + b; for (int r = 0; r < below; r++) col[r] = 0; }
        }
        const int tw = w - (c0 + b);
        if (tw > 0 && below > 0) {
            /* P[c0+b : nr, c0+b : w] -= L[c0+b : nr, c0 : c0+b] L[c0+b : w, c0 : c0+b]':
             * the square part by dsyrk (lower triangle), the rows below w by dgemm */
            const double mone = -1.0;
            const double *L1 = P + (size_t)c0 * ld + c0 + b;
            BL(dsyrk_)("L", "N", &tw, &b, &mone, L1, &ld, &DONE, P + (size_t)(c0 + b) * ld + c0 + b, &ld);
            const int rb = nr - w;
            if (rb > 0)
                BL(dgemm_)("N", "T", &rb, &tw, &b, &mone, P + (size_t)c0 * ld + w, &ld,
                           L1, &ld, &DONE, P + (size_t)(c0 + b) * ld + w, &ld);
        }
    }
    return ntiny;
}

static int schol_factor_ll(SChol *S, double shift) {
    if (g_fstat < 0) g_fstat = getenv("BRISK_FSTAT") != NULL;
    if (g_small_upd == (size_t)-1) { const char *e = getenv("BRISK_SMALLUPD"); g_small_upd = e ? (size_t)atol(e) : 0; }
    if (g_nocontig_neg < 0) g_nocontig_neg = getenv("BRISK_NOCONTIGNEG") != NULL;
    if (!S->relidx) S->relidx = sx(sizeof(int) * (S->m + 1));
    const int ns = S->ns;
    S->mval_ok = 0;
    sc_ensure(S);
    if (S->inplace && S->tinypiv <= 0) {
        /* the caller refills the values before every factorization */
        if (S->pan && S->pan != S->pmat) free(S->pan);
        S->pan = S->pmat;
    } else {
        sc_drop_moff(S);
        if (!S->pan || S->pan == S->pmat) S->pan = malloc(sizeof(double) * (S->pansz ? S->pansz : 1));
        if (!S->pan) { fprintf(stderr, "brisk: out of memory (the copy of a sparse factor, %.1f GB)\n", 8.0 * (double)S->pansz / 1e9); exit(1); }
        memcpy(S->pan, S->pmat, sizeof(double) * S->pansz);
    }
    for (int k = 0; k <= ns; k++) { S->snhead[k] = -1; S->snptr[k] = 0; }
    for (int k = 0; k < ns; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const int *rw = S->prow + S->prp[k];
        double *Pk = S->pan + S->pbase[k];
        if (shift != 0)
            for (int cc = 0; cc < w; cc++) {
                double *d = &Pk[(size_t)cc * nr + cc];
                *d += shift * (fabs(*d) > 0 ? fabs(*d) : 1.0);
            }
        for (int q = 0; q < nr; q++) S->relind[rw[q]] = q;
        /* updates from earlier supernodes */
        for (int j = S->snhead[k]; j != -1; ) {
            const int jnext = S->snnext[j];
            const int nrj = S->pnr[j], wj = S->ss[j + 1] - S->ss[j];
            const int *rwj = S->prow + S->prp[j];
            int p = S->snptr[j];
            int a = 0;
            while (p + a < nrj && rwj[p + a] < S->ss[k + 1]) a++;
            const int tot = nrj - p;
            if (a > 0 && g_fstat) { g_nupd++; g_updfl += 2.0 * tot * a * wj; if (tot * a < 256) g_nsmall++; }
            if (a > 0) {
                size_t need = (size_t)tot * a;
                if (need > S->updcap) { S->updcap = 2 * need; S->upd = realloc(S->upd, sizeof(double) * S->updcap); }
                const double *U = S->pan + S->pbase[j] + p;          /* (tot x wj), ld = nrj */
                int neg = 0;
                if (S->sgn) for (int c = 0; c < wj; c++) if (S->sgn[S->ss[j] + c] < 0) { neg = 1; break; }
                /* contiguous target (rows and columns land on consecutive panel positions,
                 * common after the postorder): one dgemm straight into the panel, no scatter
                 * (the scatter cost 60% of the dgemm time on the TSSOS Schur patterns) */
                {
                    const int r0 = S->relind[rwj[p]], c0 = rwj[p];
                    int contig = (rwj[p + a - 1] == c0 + a - 1);
                    for (int q = 1; contig && q < tot; q++) contig = S->relind[rwj[p + q]] == r0 + q;
                    if (contig && !neg) {
                        const double mone = -1.0;
                        BL(dgemm_)("N", "T", &tot, &a, &wj, &mone, U, &nrj, U, &nrj, &DONE, Pk + (size_t)(c0 - j0) * nr + r0, &nr);
                        goto advance;
                    }
                    if (contig && !g_nocontig_neg) {
                        /* 4.22: signed source (quasi-definite bordered system): U S U_top' straight
                         * into the panel as well, with the top rows sign-scaled */
                        size_t nw = (size_t)a * wj;
                        if (nw > S->swcap) { S->swcap = 2 * nw; S->sw = realloc(S->sw, sizeof(double) * S->swcap); }
                        for (int c = 0; c < wj; c++) {
                            const double sc = S->sgn[S->ss[j] + c];
                            for (int r = 0; r < a; r++) S->sw[r + (size_t)c * a] = sc * U[r + (size_t)c * nrj];
                        }
                        const double mone = -1.0;
                        BL(dgemm_)("N", "T", &tot, &a, &wj, &mone, U, &nrj, S->sw, &a, &DONE, Pk + (size_t)(c0 - j0) * nr + r0, &nr);
                        goto advance;
                    }
                }
                /* small update: straight into the panel (no dgemm call, no buffer); on the
                 * TSSOS Schur patterns 70% of the updates have tot * a < 256 */
                if ((size_t)tot * a * wj <= g_small_upd) {
                    int *restrict ri = S->relidx;
                    for (int q = 0; q < tot; q++) ri[q] = S->relind[rwj[p + q]];
                    for (int cc = 0; cc < a; cc++) {
                        double *restrict dst = Pk + (size_t)(rwj[p + cc] - j0) * nr;
                        for (int c = 0; c < wj; c++) {
                            const double *restrict col = U + (size_t)c * nrj;
                            const double u = neg ? col[cc] * S->sgn[S->ss[j] + c] : col[cc];
                            if (u == 0) continue;
                            for (int q = cc; q < tot; q++) dst[ri[q]] -= u * col[q];
                        }
                    }
                    goto advance;
                }
                if (!neg) BL(dgemm_)("N", "T", &tot, &a, &wj, &DONE, U, &nrj, U, &nrj, &DZERO, S->upd, &tot);
                else {
                    /* U S U_top': the top a rows scaled by the column signs */
                    size_t nw = (size_t)a * wj;
                    if (nw > S->swcap) { S->swcap = 2 * nw; S->sw = realloc(S->sw, sizeof(double) * S->swcap); }
                    for (int c = 0; c < wj; c++) {
                        const double sc = S->sgn[S->ss[j] + c];
                        for (int r = 0; r < a; r++) S->sw[r + (size_t)c * a] = sc * U[r + (size_t)c * nrj];
                    }
                    BL(dgemm_)("N", "T", &tot, &a, &wj, &DONE, U, &nrj, S->sw, &a, &DZERO, S->upd, &tot);
                }
                for (int cc = 0; cc < a; cc++) {
                    const int lc = rwj[p + cc] - j0;                 /* local column in k */
                    double *dst = Pk + (size_t)lc * nr;
                    const double *src = S->upd + (size_t)cc * tot;
                    for (int q = cc; q < tot; q++) dst[S->relind[rwj[p + q]]] -= src[q];
                }
            }
            advance:
            /* advance this source to its next target supernode */
            S->snptr[j] = p + a;
            if (S->snptr[j] < nrj) {
                int t = S->snof[rwj[S->snptr[j]]];
                S->snnext[j] = S->snhead[t];
                S->snhead[t] = j;
            }
            j = jnext;
        }
        /* factor the panel */
        int neg = 0;
        if (S->sgn) for (int c = 0; c < w; c++) if (S->sgn[j0 + c] < 0) { neg = 1; break; }
        int rest = nr - w;
        if (!neg && S->tinypiv > 0) {
            /* unblocked column Cholesky of the panel with tiny-pivot replacement (Wright's
             * trick for degenerate IPM normal equations): a pivot that has lost all but
             * tinypiv of its original diagonal becomes 1e128, its column of L ~ 0 */
            const double *P0 = S->pmat + S->pbase[k];
            /* blocked for wide panels; its pivots carry more cancellation noise (blocked
             * sums), so the drop threshold sits 10x higher (at 1e-14 the drop decisions were
             * noise: attr_lorenz_d8_I 36 -> 75 iterations) */
            if (g_tinyloop < 0) g_tinyloop = getenv("BRISK_TINYLOOP") != NULL;
            if (w > 96 && !g_tinyloop) S->ntiny += chol_tiny_blocked(nr, w, Pk, nr, P0, nr + 1, 10.0 * S->tinypiv);
            else
            for (int c = 0; c < w; c++) {
                double *col = Pk + (size_t)c * nr;
                sc_colupd(col, Pk, nr, 0, c, c, nr);
                double d = col[c];
                const double d0 = fabs(P0[(size_t)c * nr + c]);
                if (!(d > S->tinypiv * d0) || !(d > 0)) { d = 1e128; S->ntiny++; for (int r = c + 1; r < nr; r++) col[r] = 0; }
                const double l = sqrt(d), il = 1.0 / l;
                col[c] = l;
                for (int r = c + 1; r < nr; r++) col[r] *= il;
            }
        } else if (!neg) {
            if (panel_potrf_dyn(S, Pk, nr, w, &S->sw, &S->swcap)) return 1;
            if (rest > 0)
                sc_trsm(rest, w, Pk, nr, Pk + w, nr);
        } else if (panel_sign_uniform(S, j0, w) < 0) {
            /* all columns negative: -P = L L' by dpotrf on the negated panel gives the same
             * factor as the signed column loop (l_cc = sqrt(|d|), L_rc = -P_rc / l_cc), at
             * the BLAS rate (the loop made the bordered factorization 2.3x slower) */
            for (int c = 0; c < w; c++) { double *col = Pk + (size_t)c * nr; for (int r = c; r < nr; r++) col[r] = -col[r]; }
            if (panel_potrf_dyn(S, Pk, nr, w, &S->sw, &S->swcap)) return 1;
            if (rest > 0) sc_trsm(rest, w, Pk, nr, Pk + w, nr);
        } else {
            /* signed column Cholesky of the whole panel: l_cc = sqrt(|d|), sign prescribed */
            /* diagonal block by the column loop, the rows below by one dtrsm and a column
             * sign scaling: P21 = L21 S L11' gives L21 = P21 L11^{-T} S */
            { int nrep = 0; if (signed_chol_blocked(Pk, nr, w, S->sgn + j0, S->dynpiv, &nrep)) return 1; S->ndyn += nrep; }
            if (rest > 0) {
                /* L11 (lower part of the diagonal block) copied to a square buffer */
                size_t nw = (size_t)w * w;
                if (nw > S->swcap) { S->swcap = 2 * nw; S->sw = realloc(S->sw, sizeof(double) * S->swcap); }
                double *T = S->sw;
                for (int c = 0; c < w; c++)
                    for (int r = 0; r < w; r++) T[r + (size_t)c * w] = r < c ? 0.0 : Pk[r + (size_t)c * nr];
                sc_trsm(rest, w, T, w, Pk + w, nr);
                for (int c = 0; c < w; c++) {
                    const double sc = S->sgn[j0 + c];
                    if (sc < 0) { double *col = Pk + (size_t)c * nr + w; for (int r = 0; r < rest; r++) col[r] = -col[r]; }
                }
            }
        }
        S->snptr[k] = w;
        if (w < nr) {
            int t = S->snof[rw[w]];
            S->snnext[k] = S->snhead[t];
            S->snhead[t] = k;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------------------
 * 4.24: tree- and node-parallel supernodal factorization, bitwise independent of the
 * number of threads.
 *
 * The arithmetic is fixed by the structure alone:
 *  - every target supernode receives the updates of its sources in increasing source
 *    order (static lists built once per pattern), in row blocks of FB_RB local rows;
 *  - panels wider than FB_NB are factored by a tiled right-looking Cholesky with FB_NB
 *    column tiles and FB_RB-row chunks.
 * Scheduling (which thread runs which unit, and when) never changes a single operation:
 * independent subtrees run concurrently, and a large node's row blocks and tiles run as
 * OpenMP tasks. BLAS runs single-threaded inside.
 * ------------------------------------------------------------------------------------ */
static int FB_RB = 256, FB_NB = 128, g_fcb = -1;   /* 17 October 2026: FB_RB 512 -> 256 with the column tiles of fnode (FB_CB) */   /* fixed per run (BRISK_FRB, BRISK_FNB): they define the arithmetic */
#if defined(__APPLE__) || defined(BRISK_NO_OPENBLAS)
static void (*const BL(openblas_set_num_threads))(int) = 0;
static int (*const BL(openblas_get_num_threads))(void) = 0;
#else
extern void BL(openblas_set_num_threads)(int) __attribute__((weak));
extern int BL(openblas_get_num_threads)(void) __attribute__((weak));
#endif
typedef struct { int *relind; double *upd; size_t updcap; double *sw; size_t swcap; double *ib; size_t ibcap; double *br; size_t brcap; double *bc; size_t bccap; double *ism; size_t ismcap; int *ri; size_t ricap; } FScr;
static FScr *g_fscr = NULL; static int g_nfscr = 0, g_fscr_m = 0;
static int g_fpar = -1;             /* BRISK_FACTLL=1: the previous left-looking code */
static double g_fcut = -1;
static FScr *fscr_get(void) {
#ifdef _OPENMP
    return &g_fscr[omp_get_thread_num()];
#else
    return &g_fscr[0];
#endif
}
static void fscr_ensure(int nt, int m) {
    if (nt > g_nfscr || m > g_fscr_m) {
        for (int t = 0; t < g_nfscr; t++) { free(g_fscr[t].relind); free(g_fscr[t].upd); free(g_fscr[t].sw); free(g_fscr[t].ib); free(g_fscr[t].br); free(g_fscr[t].bc); free(g_fscr[t].ism); free(g_fscr[t].ri); }
        free(g_fscr);
        g_nfscr = nt > g_nfscr ? nt : g_nfscr; g_fscr_m = m > g_fscr_m ? m : g_fscr_m;
        g_fscr = sx(sizeof(FScr) * g_nfscr);
        for (int t = 0; t < g_nfscr; t++) g_fscr[t].relind = sx(sizeof(int) * (g_fscr_m + 1));
    }
}
static double *fscr_buf(double **b, size_t *cap, size_t need) {
    if (need > *cap) { *cap = 2 * need; *b = realloc(*b, sizeof(double) * *cap); }
    return *b;
}
/* static structure: update lists, supernodal tree, work */
static void sc_build_tree(SChol *S) {
    const int ns = S->ns;
    int *cnt = sx(sizeof(int) * (ns + 1));
    for (int j = 0; j < ns; j++) {
        const int wj = S->ss[j + 1] - S->ss[j], nrj = S->pnr[j];
        const int *rw = S->prow + S->prp[j];
        for (int p = wj; p < nrj; ) { const int t = S->snof[rw[p]]; int a = 0; while (p + a < nrj && rw[p + a] < S->ss[t + 1]) a++; cnt[t]++; p += a; }
    }
    S->ul_ptr = sx(sizeof(int) * (ns + 1));
    for (int k = 0; k < ns; k++) S->ul_ptr[k + 1] = S->ul_ptr[k] + cnt[k];
    const int nu = S->ul_ptr[ns];
    S->ul_j = sx(sizeof(int) * (nu + 1)); S->ul_p = sx(sizeof(int) * (nu + 1)); S->ul_a = sx(sizeof(int) * (nu + 1));
    for (int k = 0; k < ns; k++) cnt[k] = S->ul_ptr[k];
    S->spar = sx(sizeof(int) * (ns + 1)); S->sfd = sx(sizeof(int) * (ns + 1));
    S->swk = sx(sizeof(double) * (ns + 1)); S->ssub = sx(sizeof(double) * (ns + 1));
    for (int j = 0; j < ns; j++) {
        const int wj = S->ss[j + 1] - S->ss[j], nrj = S->pnr[j];
        const int *rw = S->prow + S->prp[j];
        S->spar[j] = nrj > wj ? S->snof[rw[wj]] : -1;
        for (int p = wj; p < nrj; ) {
            const int t = S->snof[rw[p]]; int a = 0;
            while (p + a < nrj && rw[p + a] < S->ss[t + 1]) a++;
            const int u = cnt[t]++;
            S->ul_j[u] = j; S->ul_p[u] = p; S->ul_a[u] = a;
            S->swk[t] += 2.0 * (double)(nrj - p) * a * wj;        /* update flops into t */
            p += a;
        }
        S->swk[j] += (double)wj * wj * wj / 3.0 + (double)(nrj - wj) * wj * wj;
    }
    for (int k = 0; k < ns; k++) S->sfd[k] = k;
    for (int k = 0; k < ns; k++) { S->ssub[k] += S->swk[k]; if (S->spar[k] >= 0) { S->ssub[S->spar[k]] += S->ssub[k]; if (S->sfd[k] < S->sfd[S->spar[k]]) S->sfd[S->spar[k]] = S->sfd[k]; } }
    free(cnt);
}
/* updates of target k from all its sources, restricted to local rows [R0, R1) */
#define SMALL_CB 256
#define BATCH_K 512
static double g_batchd = 0.9;
static int g_batchw = -1;       /* (18 October 2026, LP/SOCP: the widest source batched, cone solver only; BRISK_BATCHW, 4 as before) */
static int g_ftime = -1; static double g_ft[6], g_pansz_dbg, g_ftd[2];   /* (g_ftd: thread seconds of fnode_dag's trsm and gemm tasks) */   /* BRISK_FTIME: seconds in the update products, their scatter, the panel factor */
void schol_ftime_print(void) { if (g_ftime > 0) printf("   [sparse factor: update products %.2fs, scatter %.2fs, panels %.2fs (thread seconds; the task graph: trsm %.2fs, gemm %.2fs); wall: the calls %.2fs, their parallel regions %.2fs, the copy of the values %.2fs; pansz %.3g]\n", g_ft[0], g_ft[1], g_ft[2], g_ftd[0], g_ftd[1], g_ft[3], g_ft[4], g_ft[5], g_pansz_dbg); }
static void fnode_upd(SChol *S, int k, int R0, int R1, int C0, int C1, FScr *w) {
    if (g_ftime < 0) g_ftime = getenv("BRISK_FTIME") ? atoi(getenv("BRISK_FTIME")) : 0;
    if (g_batchw < 0) { const char *e = getenv("BRISK_BATCHW"); g_batchw = e ? atoi(e) : 64; e = getenv("BRISK_BATCHD"); g_batchd = e ? atof(e) : 0.9; }
    const int j0 = S->ss[k], nr = S->pnr[k];
    const int *rw = S->prow + S->prp[k];
    double *Pk = S->pan + S->pbase[k];
    int *relind = w->relind;
    for (int q = 0; q < nr; q++) relind[rw[q]] = q;
    const double mone = -1.0;
    int *small = NULL, nsmall = 0, *batch = NULL, nbatch = 0, nbcols = 0;
    const int wk = S->ss[k + 1] - j0;
    for (int u = S->ul_ptr[k]; u < S->ul_ptr[k + 1]; u++) {
        const int j = S->ul_j[u], p = S->ul_p[u], a = S->ul_a[u];
        const int nrj = S->pnr[j], wj = S->ss[j + 1] - S->ss[j], tot = nrj - p;
        const int *rwj = S->prow + S->prp[j] + p;           /* rows of the source from p */
        const double *U = S->pan + S->pbase[j] + p;          /* (tot x wj), ld = nrj */
        /* source rows whose target rows fall in [R0, R1): relind[rwj[q]] increases with q */
        int lo = 0, hi = tot;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (relind[rwj[mid]] < R0) lo = mid + 1; else hi = mid; }
        const int q0 = lo;
        hi = tot;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (relind[rwj[mid]] < R1) lo = mid + 1; else hi = mid; }
        int q1 = lo, nq = q1 - q0;
        if (nq <= 0) continue;
        /* 17 October 2026: the target columns [C0, C1) only (fnode tiles the update by rows and columns): the
         * source columns cc0 .. cc1 land there (rwj ascending), and rows above cc0 add nothing to them */
        int cc0 = 0, cc1 = a;
        if (C0 > 0 || C1 < wk) {
            lo = 0; hi = a; while (lo < hi) { const int mid = (lo + hi) >> 1; if (rwj[mid] - j0 < C0) lo = mid + 1; else hi = mid; }
            cc0 = lo; hi = a; while (lo < hi) { const int mid = (lo + hi) >> 1; if (rwj[mid] - j0 < C1) lo = mid + 1; else hi = mid; }
            cc1 = lo;
            if (cc1 <= cc0) continue;
        }
        int neg = 0;
        if (S->sgn) for (int c = 0; c < wj; c++) if (S->sgn[S->ss[j] + c] < 0) { neg = 1; break; }
        const double *V = U; int ldv = nrj;
        if (neg) {
            double *sw = fscr_buf(&w->sw, &w->swcap, (size_t)a * wj);
            for (int c = 0; c < wj; c++) {
                const double sc = S->sgn[S->ss[j] + c];
                for (int r = 0; r < a; r++) sw[r + (size_t)c * a] = sc * U[r + (size_t)c * nrj];
            }
            V = sw; ldv = a;
        }
        const int r0f = relind[rwj[q0]], c0 = rwj[0];
        int contig = (rwj[a - 1] == c0 + a - 1) && (relind[rwj[q1 - 1]] == r0f + nq - 1);
        const int q0c = q0 > cc0 ? q0 : cc0, nqc = q1 - q0c, ac = cc1 - cc0;     /* the tile's rows and columns */
        if (nqc <= 0) continue;
        const int r0 = relind[rwj[q0c]];
        /* a source of a few columns that is dense into the target tile: its columns go to a batch
         * and the whole batch is one dgemm (a dense block of order w fed by thousands of rows of
         * the data: 20,000 rank-one updates of a 200 x 200 block cost 0.2 s as scatter or as
         * dgemm with k = 1, and 0.01 s as one product) */
        if (wj <= g_batchw && S->dynpiv > 0 && a >= 8 && nq >= 8 && (wj <= 4 || contig) && (double)a * nq >= (wj <= 4 ? 0.08 : g_batchd) * (double)wk * (R1 - R0)) {
            if (!batch) batch = (int *)fscr_buf(&w->ib, &w->ibcap, 4 * (size_t)(S->ul_ptr[k + 1] - S->ul_ptr[k]) + 8);
            batch[4 * nbatch] = u; batch[4 * nbatch + 1] = q0; batch[4 * nbatch + 2] = q1; batch[4 * nbatch + 3] = 0;
            nbatch++; nbcols += wj;
            continue;
        }
        if (contig) {
            const double tf0 = g_ftime > 0 ? omp_get_wtime() : 0;
            BL(dgemm_)("N", "T", &nqc, &ac, &wj, &mone, U + q0c, &nrj, V + cc0, &ldv, &DONE, Pk + (size_t)(rwj[cc0] - j0) * nr + r0, &nr);
            if (g_ftime > 0) { const double tf1 = omp_get_wtime();
                #pragma omp atomic
                g_ft[0] += tf1 - tf0; }
            continue;
        }
        if (wj <= 4 && S->dynpiv > 0) {
            /* (the cone solver's factorizations only: the order of the updates, and so the rounding, of
             * the SDP solver's factorizations is left as it was)
             * a few columns: the update straight into the target (lower part only), without the
             * product and its scatter (a linear variable, or a variable and its row, in the cone
             * solver's systems: tens of thousands of such updates of a few hundred rows each).
             * They are done after the loop, by blocks of target columns. */
            if (!small) { small = (int *)fscr_buf(&w->ism, &w->ismcap, 2 * (size_t)(S->ul_ptr[k + 1] - S->ul_ptr[k]) + 2); }
            small[4 * nsmall] = u; small[4 * nsmall + 1] = q0; small[4 * nsmall + 2] = q1; small[4 * nsmall + 3] = 0;
            nsmall++;
            continue;
        }
        double *upd = fscr_buf(&w->upd, &w->updcap, (size_t)nqc * ac);
        const double tf0 = g_ftime > 0 ? omp_get_wtime() : 0;
        BL(dgemm_)("N", "T", &nqc, &ac, &wj, &DONE, U + q0c, &nrj, V + cc0, &ldv, &DZERO, upd, &nqc);
        const double tf1 = g_ftime > 0 ? omp_get_wtime() : 0;
        /* (sparsechol-dev) the target rows once for all the columns, and two columns per pass: each entry receives the
         * same subtraction as before (the same bits); the scatter read rwj and relind again for every column */
        if ((size_t)nqc > w->ricap) { w->ricap = 2 * (size_t)nqc; free(w->ri); w->ri = malloc(sizeof(int) * w->ricap); }
        int *restrict ri = w->ri;
        for (int q = q0c; q < q1; q++) ri[q - q0c] = relind[rwj[q]];
        int cc = cc0;
        for (; cc + 1 < cc1; cc += 2) {
            double *restrict d0 = Pk + (size_t)(rwj[cc] - j0) * nr, *restrict d1 = Pk + (size_t)(rwj[cc + 1] - j0) * nr;
            const double *s0 = upd + (size_t)(cc - cc0) * nqc, *s1 = s0 + nqc;      /* (entry q at q - q0c) */
            const int qa = q0c > cc ? q0c : cc, qb = q0c > cc + 1 ? q0c : cc + 1;
            for (int q = qa; q < qb && q < q1; q++) d0[ri[q - q0c]] -= s0[q - q0c];
            for (int q = qb; q < q1; q++) { const int i = ri[q - q0c]; d0[i] -= s0[q - q0c]; d1[i] -= s1[q - q0c]; }
        }
        if (cc < cc1) {
            double *dst = Pk + (size_t)(rwj[cc] - j0) * nr;
            const double *src = upd + (size_t)(cc - cc0) * nqc;
            for (int q = (q0c > cc ? q0c : cc); q < q1; q++) dst[ri[q - q0c]] -= src[q - q0c];
        }
        if (g_ftime > 0) { const double tf2 = omp_get_wtime();
            #pragma omp atomic
            g_ft[0] += tf1 - tf0;
            #pragma omp atomic
            g_ft[1] += tf2 - tf1; }
    }
    if (nsmall) {
        /* the small sources, by blocks of target columns: the block of the target (rows of the tile
         * x SMALL_CB columns) stays in the cache while the sources pass over it (the scatter into a
         * wide panel was bound by the memory: 4 ns per entry on the dense block of a lasso problem) */
        const int wk = S->ss[k + 1] - j0;
        const int cbw = (double)(R1 - R0) * wk * 8.0 > 1.5e6 ? SMALL_CB : wk;
        for (int cb0 = 0; cb0 < wk; cb0 += cbw) {
            const int chi = j0 + (cb0 + cbw < wk ? cb0 + cbw : wk);          /* target columns below chi */
            for (int t = 0; t < nsmall; t++) {
                int *e = small + 4 * t;
                const int u = e[0], q0 = e[1], q1 = e[2];
                const int j = S->ul_j[u], p = S->ul_p[u], a = S->ul_a[u];
                int cc = e[3];
                /* (sparsechol-dev: a column at or past q1 meets no row of the tile, since its rows start at max(q0, cc):
                 * those columns are left out, which skips only empty loops; on the root of lasso_5000, a dense node of
                 * 1502 columns fed by one-column sources, about 40 % of the visits) */
                if (cc >= a || cc >= q1) continue;
                const int nrj = S->pnr[j], wj = S->ss[j + 1] - S->ss[j];
                const int *rwj = S->prow + S->prp[j] + p;
                if (rwj[cc] >= chi) continue;
                const double *U = S->pan + S->pbase[j] + p;
                const double *U1 = U + nrj, *U2 = U + 2 * (size_t)nrj, *U3 = U + 3 * (size_t)nrj;
                double sg[4] = { 1.0, 1.0, 1.0, 1.0 };
                if (S->sgn) for (int c = 0; c < wj; c++) sg[c] = S->sgn[S->ss[j] + c];
                for (; cc < a && cc < q1 && rwj[cc] < chi; cc++) {
                    double *dst = Pk + (size_t)(rwj[cc] - j0) * nr;
                    const int qs = q0 > cc ? q0 : cc;
                    const double f0 = sg[0] * U[cc];
                    if (wj == 1) {
                        if (f0 == 0) continue;
                        for (int q = qs; q < q1; q++) dst[relind[rwj[q]]] -= f0 * U[q];
                    } else if (wj == 2) {
                        const double f1 = sg[1] * U1[cc];
                        for (int q = qs; q < q1; q++) dst[relind[rwj[q]]] -= f0 * U[q] + f1 * U1[q];
                    } else if (wj == 3) {
                        const double f1 = sg[1] * U1[cc], f2 = sg[2] * U2[cc];
                        for (int q = qs; q < q1; q++) dst[relind[rwj[q]]] -= f0 * U[q] + f1 * U1[q] + f2 * U2[q];
                    } else {
                        const double f1 = sg[1] * U1[cc], f2 = sg[2] * U2[cc], f3 = sg[3] * U3[cc];
                        for (int q = qs; q < q1; q++) dst[relind[rwj[q]]] -= f0 * U[q] + f1 * U1[q] + f2 * U2[q] + f3 * U3[q];
                    }
                }
                e[3] = cc;
            }
        }
    }
    if (nbatch) {
        /* the batch: Urow (tile rows x K) and Ucol (w x K) zero-padded, signs on Ucol, one dgemm per
         * chunk of K <= BATCH_K columns: P[R0:R1, :] -= Urow Ucol' */
        const int ntile = R1 - R0;
        int t = 0;
        while (t < nbatch) {
            int K = 0, t1 = t;
            while (t1 < nbatch && K + (S->ss[S->ul_j[batch[4 * t1]] + 1] - S->ss[S->ul_j[batch[4 * t1]]]) <= BATCH_K) { K += S->ss[S->ul_j[batch[4 * t1]] + 1] - S->ss[S->ul_j[batch[4 * t1]]]; t1++; }
            if (t1 == t) { K = S->ss[S->ul_j[batch[4 * t]] + 1] - S->ss[S->ul_j[batch[4 * t]]]; t1 = t + 1; }
            double *Ur = fscr_buf(&w->br, &w->brcap, (size_t)ntile * K), *Uc = fscr_buf(&w->bc, &w->bccap, (size_t)wk * K);
            memset(Ur, 0, sizeof(double) * (size_t)ntile * K); memset(Uc, 0, sizeof(double) * (size_t)wk * K);
            int col = 0;
            for (int b_ = t; b_ < t1; b_++) {
                const int u = batch[4 * b_], q0 = batch[4 * b_ + 1], q1 = batch[4 * b_ + 2];
                const int j = S->ul_j[u], p = S->ul_p[u], a = S->ul_a[u];
                const int nrj = S->pnr[j], wj = S->ss[j + 1] - S->ss[j];
                const int *rwj = S->prow + S->prp[j] + p;
                const double *U = S->pan + S->pbase[j] + p;
                for (int c = 0; c < wj; c++, col++) {
                    const double *Uj = U + (size_t)c * nrj;
                    const double sg = S->sgn ? (double)S->sgn[S->ss[j] + c] : 1.0;
                    double *ur = Ur + (size_t)col * ntile, *uc = Uc + (size_t)col * wk;
                    for (int q = q0; q < q1; q++) ur[relind[rwj[q]] - R0] = Uj[q];
                    for (int cc = 0; cc < a; cc++) uc[rwj[cc] - j0] = sg * Uj[cc];
                }
            }
            BL(dgemm_)("N", "T", &ntile, &wk, &K, &mone, Ur, &ntile, Uc, &wk, &DONE, Pk + R0, &nr);
            t = t1;
        }
    }
}
/* tiled right-looking Cholesky of the (nr x w) panel P (ld nr), lower part */
static int fpanel_tiled(double *P, int nr, int w, int par) {
    int bad = 0;
    for (int c0 = 0; c0 < w && !bad; c0 += FB_NB) {
        int cb = (w - c0 < FB_NB) ? w - c0 : FB_NB;
        double *D = P + (size_t)c0 * nr + c0;
        int info = 0;
        info = sc_potrf(cb, D, nr);
        if (info) { bad = 1; break; }
        const int rb0 = c0 + cb;
        for (int r = rb0; r < nr; r += FB_RB) {
            int rr = (nr - r < FB_RB) ? nr - r : FB_RB;
            double *B = P + (size_t)c0 * nr + r;
            if (par) {
                #pragma omp task firstprivate(B, rr) shared(D, nr)
                sc_trsm(rr, cb, D, nr, B, nr);
            } else sc_trsm(rr, cb, D, nr, B, nr);
        }
        if (par) {
            #pragma omp taskwait
        }
        const double mone = -1.0;
        for (int c1 = rb0; c1 < w; c1 += FB_NB) {
            int cb1 = (w - c1 < FB_NB) ? w - c1 : FB_NB;
            const double *A1 = P + (size_t)c0 * nr + c1;         /* rows c1.., columns of tile c0 */
            double *Cd = P + (size_t)c1 * nr + c1;
            if (par) {
                #pragma omp task firstprivate(A1, Cd, cb1) shared(nr)
                BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &nr, &DONE, Cd, &nr);
            } else BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &nr, &DONE, Cd, &nr);
            for (int r = c1 + cb1; r < nr; r += FB_RB) {
                int rr = (nr - r < FB_RB) ? nr - r : FB_RB;
                const double *Ar = P + (size_t)c0 * nr + r;
                double *Cr = P + (size_t)c1 * nr + r;
                if (par) {
                    #pragma omp task firstprivate(A1, Ar, Cr, rr, cb1) shared(nr)
                    BL(dgemm_)("N", "T", &rr, &cb1, &cb, &mone, Ar, &nr, A1, &nr, &DONE, Cr, &nr);
                } else BL(dgemm_)("N", "T", &rr, &cb1, &cb, &mone, Ar, &nr, A1, &nr, &DONE, Cr, &nr);
            }
        }
        if (par) {
            #pragma omp taskwait
        }
    }
    return bad;
}
/* panel factorization of supernode k (same numerics as the left-looking code for narrow
 * or special panels; tiled for wide uniform-sign ones) */
static int g_w1lib = -1;
static int fnode_factor(SChol *S, int k, FScr *wsc, int par) {
    if (g_w1lib < 0) g_w1lib = getenv("BRISK_W1LIB") != NULL;
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    double *Pk = S->pan + S->pbase[k];
    int neg = 0;
    if (S->sgn) for (int c = 0; c < w; c++) if (S->sgn[j0 + c] < 0) { neg = 1; break; }
    if (!neg && S->tinypiv > 0) {
        const double *P0 = S->pmat + S->pbase[k];
        int nt = 0;
        if (g_tinyloop < 0) g_tinyloop = getenv("BRISK_TINYLOOP") != NULL;
        if (w > 96 && !g_tinyloop) nt = chol_tiny_blocked(nr, w, Pk, nr, P0, nr + 1, 10.0 * S->tinypiv);
        else
        for (int c = 0; c < w; c++) {
            double *col = Pk + (size_t)c * nr;
            sc_colupd(col, Pk, nr, 0, c, c, nr);
            double d = col[c];
            const double d0 = fabs(P0[(size_t)c * nr + c]);
            if (!(d > S->tinypiv * d0) || !(d > 0)) { d = 1e128; nt++; for (int r = c + 1; r < nr; r++) col[r] = 0; }
            const double l = sqrt(d), il = 1.0 / l;
            col[c] = l;
            for (int r = c + 1; r < nr; r++) col[r] *= il;
        }
        if (nt) {
            #pragma omp atomic
            S->ntiny += nt;
        }
        return 0;
    }
    const int uni = neg ? panel_sign_uniform(S, j0, w) : 1;
    if (uni != 0) {
        if (uni < 0) for (int c = 0; c < w; c++) { double *col = Pk + (size_t)c * nr; for (int r = c; r < nr; r++) col[r] = -col[r]; }
        if (w == 1 && !g_w1lib) {
            /* 17 October 2026: a one-column node without the two library calls (dpotrf of 1 x 1, dtrsm of one column):
             * the cone solver's factorizations have tens of thousands of them (robustlp_400: 61,500), where those
             * calls were most of the time of the node. The pivot is replaced as in chol_dyn_block. BRISK_W1LIB=1: the calls */
            double d = Pk[0];
            if (!(d > 0)) {
                if (!(S->dynpiv > 0)) return 1;
                d = S->dynpiv;
#pragma omp atomic
                S->ndyn += 1;
            }
            const double l = sqrt(d), il = 1.0 / l;
            Pk[0] = l;
            for (int r = 1; r < nr; r++) Pk[r] *= il;
            return 0;
        }
        if (w > FB_NB && !(S->dynpiv > 0)) return fpanel_tiled(Pk, nr, w, par);
        if (panel_potrf_dyn(S, Pk, nr, w, &wsc->sw, &wsc->swcap)) return 1;
        if (rest > 0) sc_trsm(rest, w, Pk, nr, Pk + w, nr);
        return 0;
    }
    /* mixed signs: signed Cholesky by blocks */
    { int nrep = 0;
      if (signed_chol_blocked(Pk, nr, w, S->sgn + j0, S->dynpiv, &nrep)) return 1;
      if (nrep) {
#pragma omp atomic
          S->ndyn += nrep;
      } }
    if (rest > 0) {
        double *T = fscr_buf(&wsc->sw, &wsc->swcap, (size_t)w * w);
        for (int c = 0; c < w; c++)
            for (int r = 0; r < w; r++) T[r + (size_t)c * w] = r < c ? 0.0 : Pk[r + (size_t)c * nr];
        sc_trsm(rest, w, T, w, Pk + w, nr);
        for (int c = 0; c < w; c++) {
            const double sc = S->sgn[j0 + c];
            if (sc < 0) { double *col = Pk + (size_t)c * nr + w; for (int r = 0; r < rest; r++) col[r] = -col[r]; }
        }
    }
    return 0;
}

/* 4.24: dense Cholesky of an n x n matrix (lower, ld lda): LAPACK below FB_DENSE_MIN, the
 * tiled task-parallel form above (fixed tiles: the same arithmetic for any thread count) */
#define FB_DENSE_MIN 512
static int g_pdoff = -1;
static int getenv_cached_pd(void) { if (g_pdoff < 0) g_pdoff = getenv("BRISK_NOPD") != NULL; return g_pdoff; }
extern void BL(spotrf_)(const char *, const int *, float *, const int *, int *);
extern void BL(strsm_)(const char *, const char *, const char *, const char *, const int *, const int *, const float *, const float *, const int *, float *, const int *);
extern void BL(ssyrk_)(const char *, const char *, const int *, const int *, const float *, const float *, const int *, const float *, float *, const int *);
extern void BL(sgemm_)(const char *, const char *, const int *, const int *, const int *, const float *, const float *, const int *, const float *, const int *, const float *, float *, const int *);
/* (sparsechol-dev) the panel kernels in single precision, and both with the calling sequences of dpotrf / dtrsm (lower;
 * right, lower, transposed: the only forms used here) for the dense task graph's macro */
SC_KERNELS(_s, float, BL(sgemm_), BL(ssyrk_), sqrtf, BL(strsm_), BL(spotrf_))
/* 19 October 2026 (the integration, public 1.5.3): the dense Cholesky of the Schur complement keeps the library's dpotrf
 * and dtrsm by default; BRISK_DENSEKERN=1 takes the panel kernels there too. With the kernels SDPLIB's dense files
 * moved both ways (control10 REDUCED -> OPTIMAL, 30 -> 23 iterations; qap9 16 -> 23 iterations, 0.6 -> 4.0 s; equalG51
 * 19 -> 22) for a dense factor up to 15 % faster; the sparse factor's gain (a fifth of its factorizations) does not
 * depend on it. Left for a judgement on more sets (IDEAS). */
static int g_densekern = -1;
static inline int sc_densekern(void) {
    if (g_densekern < 0) { const char *e = getenv("BRISK_DENSEKERN"); g_densekern = e && atoi(e) > 0; }
    return g_densekern && !sc_panellib();
}
static void sc_dpotrf_w(const char *u, const int *n, double *A, const int *lda, int *info) {
    if (!sc_densekern()) { BL(dpotrf_)(u, n, A, lda, info); return; }
    *info = sc_potrf(*n, A, *lda);
}
static void sc_spotrf_w(const char *u, const int *n, float *A, const int *lda, int *info) {
    if (!sc_densekern()) { BL(spotrf_)(u, n, A, lda, info); return; }
    *info = sc_potrf_s(*n, A, *lda);
}
static void sc_dtrsm_w(const char *sd, const char *u, const char *t, const char *dg, const int *m, const int *n, const double *al, const double *L, const int *ldl, double *B, const int *ldb) {
    if (*al != 1 || !sc_densekern()) { BL(dtrsm_)(sd, u, t, dg, m, n, al, L, ldl, B, ldb); return; }
    sc_trsm(*m, *n, L, *ldl, B, *ldb);
}
static void sc_strsm_w(const char *sd, const char *u, const char *t, const char *dg, const int *m, const int *n, const float *al, const float *L, const int *ldl, float *B, const int *ldb) {
    if (*al != 1 || !sc_densekern()) { BL(strsm_)(sd, u, t, dg, m, n, al, L, ldl, B, ldb); return; }
    sc_trsm_s(*m, *n, L, *ldl, B, *ldb);
}
/* 4.25: row tiles of the dense (Schur, X, Z) Cholesky: 2048 rows instead of the sparse
 * factor's 512 (fewer OpenBLAS packings; still fixed, so the same bits at any thread count).
 * 5.12: 512 again. With 2048 the triangular solve of each panel was one task up to order 2176
 * and the updates few and long: at four threads order 2000 took 48-63 ms (double) and 26-28 ms
 * (single) against 27 and 15 ms with 512, slower than at two threads; order 3500 115-130 against
 * 94 ms, 6910 651 / 338 against 628 / 304 ms; one thread at 6910 about 10 % slower
 * (bench/dense_5.12/potrf_bench.c, BRISK_FDRB). Below order 1536 the same tiles run on one
 * thread: the tasks cost more than they share (order 1000: 10.6 ms at four threads, 9.1 at one). */
static int FD_RB = 512;
#define FD_PAR_MIN 1536
static void pd_env(void);
static int fpanel_tiled_ld(double *P, int nr, int w, int ld, int par) {
    int bad = 0;
    const double mone = -1.0;
    for (int c0 = 0; c0 < w && !bad; c0 += FB_NB) {
        int cb = (w - c0 < FB_NB) ? w - c0 : FB_NB;
        double *D = P + (size_t)c0 * ld + c0;
        int info = 0; sc_dpotrf_w("L", &cb, D, &ld, &info);
        if (info) { bad = 1; break; }
        for (int r = c0 + cb; r < nr; r += FD_RB) {
            int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
            double *B = P + (size_t)c0 * ld + r;
            if (par) {
                #pragma omp task firstprivate(B, rr) shared(D, ld)
                sc_dtrsm_w("R", "L", "T", "N", &rr, &cb, &DONE, D, &ld, B, &ld);
            } else sc_dtrsm_w("R", "L", "T", "N", &rr, &cb, &DONE, D, &ld, B, &ld);
        }
        if (par) {
            #pragma omp taskwait
        }
        for (int c1 = c0 + cb; c1 < w; c1 += FB_NB) {
            int cb1 = (w - c1 < FB_NB) ? w - c1 : FB_NB;
            const double *A1 = P + (size_t)c0 * ld + c1;
            double *Cd = P + (size_t)c1 * ld + c1;
            if (par) {
                #pragma omp task firstprivate(A1, Cd, cb1) shared(ld)
                BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &DONE, Cd, &ld);
            } else BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &DONE, Cd, &ld);
            for (int r = c1 + cb1; r < nr; r += FD_RB) {
                int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
                const double *Ar = P + (size_t)c0 * ld + r;
                double *Cr = P + (size_t)c1 * ld + r;
                if (par) {
                    #pragma omp task firstprivate(A1, Ar, Cr, rr, cb1) shared(ld)
                    BL(dgemm_)("N", "T", &rr, &cb1, &cb, &mone, Ar, &ld, A1, &ld, &DONE, Cr, &ld);
                } else BL(dgemm_)("N", "T", &rr, &cb1, &cb, &mone, Ar, &ld, A1, &ld, &DONE, Cr, &ld);
            }
        }
        if (par) {
            #pragma omp taskwait
        }
    }
    return bad;
}
static int fpanel_tiled_s(float *P, int nr, int w, int ld, int par) {
    int bad = 0;
    const float one = 1.0f, mone = -1.0f;
    for (int c0 = 0; c0 < w && !bad; c0 += FB_NB) {
        int cb = (w - c0 < FB_NB) ? w - c0 : FB_NB;
        float *D = P + (size_t)c0 * ld + c0;
        int info = 0; sc_spotrf_w("L", &cb, D, &ld, &info);
        if (info) { bad = 1; break; }
        for (int r = c0 + cb; r < nr; r += FD_RB) {
            int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
            float *B = P + (size_t)c0 * ld + r;
            if (par) {
                #pragma omp task firstprivate(B, rr) shared(D, ld)
                sc_strsm_w("R", "L", "T", "N", &rr, &cb, &one, D, &ld, B, &ld);
            } else sc_strsm_w("R", "L", "T", "N", &rr, &cb, &one, D, &ld, B, &ld);
        }
        if (par) {
            #pragma omp taskwait
        }
        for (int c1 = c0 + cb; c1 < w; c1 += FB_NB) {
            int cb1 = (w - c1 < FB_NB) ? w - c1 : FB_NB;
            const float *A1 = P + (size_t)c0 * ld + c1;
            float *Cd = P + (size_t)c1 * ld + c1;
            if (par) {
                #pragma omp task firstprivate(A1, Cd, cb1) shared(ld)
                BL(ssyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &one, Cd, &ld);
            } else BL(ssyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &one, Cd, &ld);
            for (int r = c1 + cb1; r < nr; r += FD_RB) {
                int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
                const float *Ar = P + (size_t)c0 * ld + r;
                float *Cr = P + (size_t)c1 * ld + r;
                if (par) {
                    #pragma omp task firstprivate(A1, Ar, Cr, rr, cb1) shared(ld)
                    BL(sgemm_)("N", "T", &rr, &cb1, &cb, &mone, Ar, &ld, A1, &ld, &one, Cr, &ld);
                } else BL(sgemm_)("N", "T", &rr, &cb1, &cb, &mone, Ar, &ld, A1, &ld, &one, Cr, &ld);
            }
        }
        if (par) {
            #pragma omp taskwait
        }
    }
    return bad;
}
/* returns LAPACK-style info (0 ok, > 0 not positive definite) */
/* 5.12: the tiled Cholesky as a task graph: column blocks of FB_NB, row groups of FD_RB fixed by
 * the order (aligned at multiples of FD_RB), one dependence token per (row group, column
 * block). A column block's factorization starts as soon as its own updates are done, while
 * the rest of the trailing matrix is still being updated (lookahead without a schedule of its
 * own); each tile receives its updates in panel order through the chain of its token, so the
 * result does not depend on the thread count. Called inside a single region. */
#define FD_DAG(NAME, T, POTRF, TRSM, SYRK, GEMM)                                                     \
static int NAME(T *A, int n, int ld, int NB, int RB) {                                              \
    const int nJ = (n + NB - 1) / NB, nG = (n + RB - 1) / RB;                                        \
    char *tok = calloc((size_t)nG * nJ + 1, 1);                                                      \
    int bad = 0;                                                                                     \
    const T one = 1, mone = -1;                                                                      \
    for (int J = 0; J < nJ; J++) {                                                                   \
        int c0 = J * NB, cb = c0 + NB < n ? NB : n - c0, gd = c0 / RB;                         \
        T *D = A + c0 + (size_t)c0 * ld;                                                             \
        char *tdj = tok + (size_t)gd * nJ + J; (void)tdj;                                            \
        _Pragma("omp task depend(inout: tdj[0]) shared(bad)")                                       \
        { int info = 0; if (!bad) { POTRF("L", &cb, D, &ld, &info); if (info) { _Pragma("omp atomic write") bad = 1; } } } \
        for (int G = gd; G < nG; G++) {                         /* the column block below its diagonal */ \
            int r0 = G == gd ? c0 + cb : G * RB, r1 = (G + 1) * RB < n ? (G + 1) * RB : n;      \
            if (r0 >= r1) continue;                                                                  \
            int rr = r1 - r0;                                                                  \
            T *B = A + r0 + (size_t)c0 * ld;                                                         \
            char *tg = tok + (size_t)G * nJ + J; (void)tg;                                           \
            if (G == gd) {                                      /* (the diagonal block's own group) */ \
                _Pragma("omp task depend(inout: tdj[0]) shared(bad)")                               \
                { if (!bad) TRSM("R", "L", "T", "N", &rr, &cb, &one, D, &ld, B, &ld); }              \
            } else {                                                                                 \
                _Pragma("omp task depend(in: tdj[0]) depend(inout: tg[0]) shared(bad)")             \
                { if (!bad) TRSM("R", "L", "T", "N", &rr, &cb, &one, D, &ld, B, &ld); }              \
            }                                                                                        \
        }                                                                                            \
        for (int K = J + 1; K < nJ; K++) {                      /* the updates of column block K */  \
            int k0 = K * NB, kb = k0 + NB < n ? NB : n - k0, gk = k0 / RB;                     \
            const T *AK = A + k0 + (size_t)c0 * ld;                     /* rows of K in panel J */  \
            char *tkj = tok + (size_t)gk * nJ + J, *tkk = tok + (size_t)gk * nJ + K; (void)tkj; (void)tkk; \
            T *DK = A + k0 + (size_t)k0 * ld;                                                        \
            _Pragma("omp task depend(in: tkj[0]) depend(inout: tkk[0]) shared(bad)")                \
            { if (!bad) SYRK("L", "N", &kb, &cb, &mone, AK, &ld, &one, DK, &ld); }                   \
            for (int G = gk; G < nG; G++) {                                                          \
                int r0 = G == gk ? k0 + kb : G * RB, r1 = (G + 1) * RB < n ? (G + 1) * RB : n;  \
                if (r0 >= r1) continue;                                                              \
                int rr = r1 - r0;                                                              \
                const T *AR = A + r0 + (size_t)c0 * ld;                                              \
                T *CR = A + r0 + (size_t)k0 * ld;                                                    \
                char *tgj = tok + (size_t)G * nJ + J, *tgk = tok + (size_t)G * nJ + K; (void)tgj; (void)tgk; \
                _Pragma("omp task depend(in: tgj[0], tkj[0]) depend(inout: tgk[0]) shared(bad)")    \
                { if (!bad) GEMM("N", "T", &rr, &kb, &cb, &mone, AR, &ld, AK, &ld, &one, CR, &ld); } \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
    _Pragma("omp taskwait")                                                                          \
    free(tok);                                                                                       \
    return bad;                                                                                      \
}
FD_DAG(fd_dag_d, double, sc_dpotrf_w, sc_dtrsm_w, BL(dsyrk_), BL(dgemm_))
FD_DAG(fd_dag_s, float, sc_spotrf_w, sc_strsm_w, BL(ssyrk_), BL(sgemm_))
static int g_fd_dag = -1, g_fd_fixed = 0;
/* 5.12: the task graph's tiles grow with the order: 128 x 512 below 4500, 192 x 1152 below
 * 11000, 256 x 1536 above (the inner dimension of the updates is the column block: 128 kept the
 * double updates at 150 GF/s where dgemm gives 215). They depend on n only: the same bits for any
 * thread count. Measured on a 4-core Cascade Lake, four threads (bench/dense_5.12/potrf_tiles_cl_T4.txt):
 * double 150 -> 196 GF/s at n = 14949, 160 -> 185 at 8000; single 305 -> 386 at 14949; below
 * 4500 the small tiles stay best. BRISK_FNB / BRISK_FDRB (test): fixed tiles for every order. */
static void fd_tiles(int n, int *nb, int *rb) {
    if (g_fd_fixed || n < 4500) { *nb = FB_NB; *rb = FD_RB; }
    else if (n < 11000) { *nb = 192; *rb = 1152; }
    else { *nb = 256; *rb = 1536; }
}
int brisk_dpotrf(int n, double *A, int lda) {
    if (n <= 0) return 0;                      /* 4.25: m = 0 (LAPACK rejects lda = 0) */
    pd_env();
    int info = 0;
    if (n < FB_DENSE_MIN) { BL(dpotrf_)("L", &n, A, &lda, &info); return info; }
    int nt = 1;
#ifdef _OPENMP
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    if (n < FD_PAR_MIN) return fpanel_tiled_ld(A, n, n, lda, 0);
    if (g_fd_dag < 0) g_fd_dag = !ENV_ON("BRISK_FDNODAG");
    int fnb, frb; fd_tiles(n, &fnb, &frb);
    if (nt <= 1) return g_fd_dag ? fd_dag_d(A, n, lda, fnb, frb) : fpanel_tiled_ld(A, n, n, lda, 0);   /* (the same tasks, in order: the same bits) */
    int bad = 0;
    #pragma omp parallel
    #pragma omp single
    bad = g_fd_dag ? fd_dag_d(A, n, lda, fnb, frb) : fpanel_tiled_ld(A, n, n, lda, 1);
    return bad;
}
int brisk_spotrf(int n, float *A, int lda) {
    if (n <= 0) return 0;
    pd_env();
    int info = 0;
    if (n < FB_DENSE_MIN) { BL(spotrf_)("L", &n, A, &lda, &info); return info; }
    int nt = 1;
#ifdef _OPENMP
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    if (n < FD_PAR_MIN) return fpanel_tiled_s(A, n, n, lda, 0);
    if (g_fd_dag < 0) g_fd_dag = !ENV_ON("BRISK_FDNODAG");
    int fnb, frb; fd_tiles(n, &fnb, &frb);
    if (nt <= 1) return g_fd_dag ? fd_dag_s(A, n, lda, fnb, frb) : fpanel_tiled_s(A, n, n, lda, 0);   /* (the same tasks, in order: the same bits) */
    int bad = 0;
    #pragma omp parallel
    #pragma omp single
    bad = g_fd_dag ? fd_dag_s(A, n, lda, fnb, frb) : fpanel_tiled_s(A, n, n, lda, 1);
    return bad;
}
/* 18 October 2026: the Bunch-Kaufman factorization A = L D L' (lower) of LAPACK's dsytrf with its trailing updates
 * on the threads. dsytrf runs on one core here (OpenBLAS is single-threaded in BRISK, and its own threads would
 * change the bits with their number): the saddle system of the free variables of a sphere SOS (order 4,998) took
 * 1.5 s a factorization against 0.27 s for the threaded Cholesky of M alone. The blocked algorithm of dsytrf /
 * dlasyf (lower, blocks of 64 columns) with the same BLAS calls in the same order; the update of the trailing
 * matrix by the block's columns, A22 -= L21 W', is one block column of 64 at a time in dlasyf, and the block
 * columns write disjoint parts of A, so they run in parallel, each one single-threaded call as before: the factor
 * is the same at any thread count. The panel (the pivot search and its level-2 updates) stays sequential. The
 * result is in dsytrf's format (dsytrs solves with it). Returns LAPACK's info. */
extern void BL(dcopy_)(const int *, const double *, const int *, double *, const int *);
extern void BL(dswap_)(const int *, double *, const int *, double *, const int *);
extern int BL(idamax_)(const int *, const double *, const int *);
extern void BL(dsytf2_)(const char *, const int *, double *, const int *, int *, int *);
extern void BL(dscal_)(const int *, const double *, double *, const int *);
#define SY_NB 64
/* y -= A x (A len x kc, column-major) by fixed chunks of rows on the threads: each row's sum is formed by the same
 * single-threaded call on its chunk, in the same order (checked: the factor equals LAPACK's bit for bit) */
static void sy_gemv_rows(int len, int kc, const double *A, int lda, const double *x, int incx, double *y, int par) {
    const double one = 1.0, mone = -1.0;
    if (kc <= 0) return;
    const int ch = 1024, nch = (len + ch - 1) / ch;
    #pragma omp parallel for schedule(static) if (par && nch > 1 && (double)len * kc >= 2e5)
    for (int c = 0; c < nch; c++) {
        const int r0 = c * ch, rl = len - r0 < ch ? len - r0 : ch;
        BL(dgemv_)("N", &rl, &kc, &mone, A + r0, &lda, x, &incx, &one, y + r0, &IONE);
    }
}
/* dlasyf, lower: factor up to nb columns of the n x n submatrix A (with W, ldw >= n), update the trailing part */
double g_sy_t[3];
static int sy_lasyf(int n, int nb, int *kb_out, double *A, int lda, int *ipiv, double *W, int ldw, int par) {
    const double one = 1.0, mone = -1.0, alpha = (1.0 + sqrt(17.0)) / 8.0;
    int info = 0, k = 1;     /* (1-based as in LAPACK) */
#define AA(i, j) A[((i) - 1) + (size_t)((j) - 1) * lda]
#define WW(i, j) W[((i) - 1) + (size_t)((j) - 1) * ldw]
    for (;;) {
        if ((k >= nb && nb < n) || k > n) break;
        int len = n - k + 1, km1 = k - 1, kstep = 1, kp, imax = 0;
        BL(dcopy_)(&len, &AA(k, k), &IONE, &WW(k, k), &IONE);
        sy_gemv_rows(len, km1, &AA(k, 1), lda, &WW(k, 1), ldw, &WW(k, k), par);
        const double absakk = fabs(WW(k, k));
        double colmax = 0;
        if (k < n) { const int l = n - k; imax = k + BL(idamax_)(&l, &WW(k + 1, k), &IONE); colmax = fabs(WW(imax, k)); }
        if (fmax(absakk, colmax) == 0) { if (info == 0) info = k; kp = k; }
        else {
            if (absakk >= alpha * colmax) kp = k;
            else {
                int l1 = imax - k, l2 = n - imax + 1;
                BL(dcopy_)(&l1, &AA(imax, k), &lda, &WW(k, k + 1), &IONE);
                BL(dcopy_)(&l2, &AA(imax, imax), &IONE, &WW(imax, k + 1), &IONE);
                sy_gemv_rows(len, km1, &AA(k, 1), lda, &WW(imax, 1), ldw, &WW(k, k + 1), par);
                int jmax = k - 1 + BL(idamax_)(&l1, &WW(k, k + 1), &IONE);
                double rowmax = fabs(WW(jmax, k + 1));
                if (imax < n) { const int l3 = n - imax; jmax = imax + BL(idamax_)(&l3, &WW(imax + 1, k + 1), &IONE); rowmax = fmax(rowmax, fabs(WW(jmax, k + 1))); }
                if (absakk >= alpha * colmax * (colmax / rowmax)) kp = k;
                else if (fabs(WW(imax, k + 1)) >= alpha * rowmax) { kp = imax; BL(dcopy_)(&len, &WW(k, k + 1), &IONE, &WW(k, k), &IONE); }
                else { kp = imax; kstep = 2; }
            }
            const int kk = k + kstep - 1;
            if (kp != kk) {
                AA(kp, kp) = AA(kk, kk);
                int l4 = kp - kk - 1;
                BL(dcopy_)(&l4, &AA(kk + 1, kk), &IONE, &AA(kp, kk + 1), &lda);
                if (kp < n) { int l5 = n - kp; BL(dcopy_)(&l5, &AA(kp + 1, kk), &IONE, &AA(kp + 1, kp), &IONE); }
                if (k > 1) BL(dswap_)(&km1, &AA(kk, 1), &lda, &AA(kp, 1), &lda);
                int l6 = kk;
                BL(dswap_)(&l6, &WW(kk, 1), &ldw, &WW(kp, 1), &ldw);
            }
            if (kstep == 1) {
                BL(dcopy_)(&len, &WW(k, k), &IONE, &AA(k, k), &IONE);
                if (k < n) { const double r1 = one / AA(k, k); int l7 = n - k; BL(dscal_)(&l7, &r1, &AA(k + 1, k), &IONE); }
            } else {
                if (k < n - 1) {
                    double d21 = WW(k + 1, k);
                    const double d11 = WW(k + 1, k + 1) / d21, d22 = WW(k, k) / d21, t = one / (d11 * d22 - one);
                    d21 = t / d21;
                    for (int j = k + 2; j <= n; j++) {
                        AA(j, k) = d21 * (d11 * WW(j, k) - WW(j, k + 1));
                        AA(j, k + 1) = d21 * (d22 * WW(j, k + 1) - WW(j, k));
                    }
                }
                AA(k, k) = WW(k, k); AA(k + 1, k) = WW(k + 1, k); AA(k + 1, k + 1) = WW(k + 1, k + 1);
            }
        }
        if (kstep == 1) ipiv[k - 1] = kp; else { ipiv[k - 1] = -kp; ipiv[k] = -kp; }
        k += kstep;
    }
    extern double g_sy_t[3]; const double tsy0 = omp_get_wtime();
    /* A22 -= L21 W' by blocks of nb columns: each block column on its own (the diagonal block's lower triangle by
     * one dgemv a column, the rest by one dgemm), independent of the others */
    {
        const int kk1 = k - 1, nblk = (n - k + nb) / nb;
        #pragma omp parallel for schedule(dynamic, 1) if (par && nblk > 1)
        for (int bj = 0; bj < nblk; bj++) {
            const int j = k + bj * nb, jb = nb < n - j + 1 ? nb : n - j + 1;
            for (int jj = j; jj <= j + jb - 1; jj++) {
                int l = j + jb - jj;
                if (kk1 > 0) BL(dgemv_)("N", &l, &kk1, &mone, &AA(jj, 1), &lda, &WW(jj, 1), &ldw, &one, &AA(jj, jj), &IONE);
            }
            if (j + jb <= n && kk1 > 0) { int r = n - j - jb + 1; BL(dgemm_)("N", "T", &r, &jb, &kk1, &mone, &AA(j + jb, 1), &lda, &WW(j, 1), &ldw, &one, &AA(j + jb, j), &lda); }
        }
    }
    g_sy_t[1] += omp_get_wtime() - tsy0;
    /* L21 in standard form: the interchanges of rows in columns 1:k-1 partly undone, from k-1 back */
    {
        int j = k - 1;
        while (j > 1) {
            const int jj = j; int jp = ipiv[j - 1];
            if (jp < 0) { jp = -jp; j--; }
            j--;
            if (jp != jj && j >= 1) BL(dswap_)(&j, &AA(jp, 1), &lda, &AA(jj, 1), &lda);
        }
    }
#undef AA
#undef WW
    *kb_out = k - 1;
    return info;
}
int brisk_dsytrf(int n, double *A, int lda, int *ipiv) {
    if (n <= 0) return 0;
    const int nb = SY_NB;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    if (nb >= n) { int info = 0; BL(dsytf2_)("L", &n, A, &lda, ipiv, &info); return info; }
    double *W = malloc(sizeof(double) * (size_t)n * nb);
    if (!W) { int info = 0; BL(dsytf2_)("L", &n, A, &lda, ipiv, &info); return info; }
    int info = 0, k = 1;
    const double tsy = omp_get_wtime(); g_sy_t[1] = 0;
    while (k <= n) {
        int kb, iinfo = 0, m = n - k + 1;
        double *Akk = A + (size_t)(k - 1) + (size_t)(k - 1) * lda;
        if (k <= n - nb) iinfo = sy_lasyf(m, nb, &kb, Akk, lda, ipiv + k - 1, W, n, nt > 1);
        else { BL(dsytf2_)("L", &m, Akk, &lda, ipiv + k - 1, &iinfo); kb = m; }
        if (info == 0 && iinfo > 0) info = iinfo + k - 1;
        for (int j = k; j <= k + kb - 1; j++) { if (ipiv[j - 1] > 0) ipiv[j - 1] += k - 1; else ipiv[j - 1] -= k - 1; }
        k += kb;
    }
    free(W);
    if (getenv("BRISK_SYTIME")) printf("     [brisk_dsytrf N %d: %.3f s, of it the trailing updates %.3f s]\n", n, omp_get_wtime() - tsy, g_sy_t[1]);
    return info;
}
/* 18 October 2026: dsytrf's factor (lower) turned into P A P' = L D L' with one permutation P and a unit lower L, as
 * LAPACK's dsyconv ("C") does for dsytrs2: the off-diagonal entries of the 2 x 2 blocks of D into e (n), the rows
 * of L permuted. Then brisk_dsytrs1 solves with two threaded triangular solves (the panels and fixed chunks of
 * brisk_dtrsv1: the same bits at any thread count) instead of dsytrs's rank-one updates on one core. */
void brisk_dsyconv(int n, double *A, int lda, const int *ipiv, double *e) {
#define AA(i, j) A[(i) + (size_t)(j) * lda]
    e[n - 1] = 0;
    for (int i = 0; i < n; i++) {
        if (i < n - 1 && ipiv[i] < 0) { e[i] = AA(i + 1, i); e[i + 1] = 0; AA(i + 1, i) = 0; i++; }
        else e[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        if (ipiv[i] > 0) {
            const int ip = ipiv[i] - 1;
            for (int j = 0; j < i; j++) { const double t = AA(ip, j); AA(ip, j) = AA(i, j); AA(i, j) = t; }
        } else {
            const int ip = -ipiv[i] - 1;
            for (int j = 0; j < i; j++) { const double t = AA(ip, j); AA(ip, j) = AA(i + 1, j); AA(i + 1, j) = t; }
            i++;
        }
    }
#undef AA
}
static void dtrsv1_d(int trans, const char *dg, int n, const double *L, int lda, double *x);
void brisk_dsytrs1(int n, const double *A, int lda, const int *ipiv, const double *e, double *x) {
#define AA(i, j) A[(i) + (size_t)(j) * lda]
    for (int k = 0; k < n; ) {                                  /* P' x */
        if (ipiv[k] > 0) { const int kp = ipiv[k] - 1; if (kp != k) { const double t = x[k]; x[k] = x[kp]; x[kp] = t; } k++; }
        else { const int kp = -ipiv[k + 1] - 1; if (kp == -ipiv[k] - 1) { const double t = x[k + 1]; x[k + 1] = x[kp]; x[kp] = t; } k += 2; }
    }
    dtrsv1_d(0, "U", n, A, lda, x);
    for (int i = 0; i < n; i++) {                               /* D \ x */
        if (ipiv[i] > 0) x[i] *= 1.0 / AA(i, i);
        else {
            const double akm1k = e[i], akm1 = AA(i, i) / akm1k, ak = AA(i + 1, i + 1) / akm1k, denom = akm1 * ak - 1.0;
            const double bkm1 = x[i] / akm1k, bk = x[i + 1] / akm1k;
            x[i] = (ak * bkm1 - bk) / denom; x[i + 1] = (akm1 * bk - bkm1) / denom;
            i++;
        }
    }
    dtrsv1_d(1, "U", n, A, lda, x);
    for (int k = n - 1; k >= 0; ) {                             /* P x */
        if (ipiv[k] > 0) { const int kp = ipiv[k] - 1; if (kp != k) { const double t = x[k]; x[k] = x[kp]; x[kp] = t; } k--; }
        else { const int kp = -ipiv[k] - 1; if (k > 0 && kp == -ipiv[k - 1] - 1) { const double t = x[k]; x[k] = x[kp]; x[kp] = t; } k -= 2; }   /* (the pair's second row, as swapped forward) */
    }
#undef AA
}

/* 5.11 (a user's patch): L L' x = b, one right-hand side, L lower: the dense Schur solves.
 * OpenBLAS's level-2 solves run on one core (m = 6264: about 19 ms per pair). Here each sweep
 * goes over panels of FB_NB columns; the rows below a panel go in fixed chunks of PS_RB, summed
 * in chunk order: the same bits for any thread count (6 threads: 5.6 ms). Below 2 FB_DENSE_MIN
 * the two level-2 solves (BRISK_TRSV=1: always). Measured on a 4-core host
 * (bench/threads_5.11/potrs_bench.c): 0.4-0.56 times the two trsv at four threads from n = 3,000,
 * 1.0-1.2 times on one thread; in the solver 2 % less time per iteration at four threads,
 * nothing measurable on one (REPORT_5.11 section 3). */
#define PS_RB 256
#define BRISK_POTRS1(NAME, T, TRSV, GEMV)                                                     \
void NAME(int n, const T *L, int lda, T *x) {                                               \
    if (n < 2 * FB_DENSE_MIN || ENV_ON("BRISK_TRSV")) {                                     \
        TRSV("L", "N", "N", &n, L, &lda, x, &IONE); TRSV("L", "T", "N", &n, L, &lda, x, &IONE); return; } \
    const T one = 1, mone = -1, zero = 0;                                                   \
    T *wk = malloc(sizeof(T) * (size_t)((n + PS_RB - 1) / PS_RB) * FB_NB);                  \
    _Pragma("omp parallel if (!omp_in_parallel())")                                         \
    {                                                                                       \
        for (int j = 0; j < n; j += FB_NB) {                                                \
            const int w = j + FB_NB < n ? FB_NB : n - j, r0 = j + w, nch = (n - r0 + PS_RB - 1) / PS_RB; \
            _Pragma("omp single")                                                           \
            TRSV("L", "N", "N", &w, L + j + (size_t)j * lda, &lda, x + j, &IONE);           \
            _Pragma("omp for schedule(static)")                                             \
            for (int c = 0; c < nch; c++) {                                                 \
                const int a = r0 + c * PS_RB, len = a + PS_RB <= n ? PS_RB : n - a;         \
                GEMV("N", &len, &w, &mone, L + a + (size_t)j * lda, &lda, x + j, &IONE, &one, x + a, &IONE); \
            }                                                                               \
        }                                                                                   \
        for (int j = (n - 1) / FB_NB * FB_NB; j >= 0; j -= FB_NB) {                         \
            const int w = j + FB_NB < n ? FB_NB : n - j, r0 = j + w, nch = (n - r0 + PS_RB - 1) / PS_RB; \
            _Pragma("omp for schedule(static)")                                             \
            for (int c = 0; c < nch; c++) {                                                 \
                const int a = r0 + c * PS_RB, len = a + PS_RB <= n ? PS_RB : n - a;         \
                GEMV("T", &len, &w, &one, L + a + (size_t)j * lda, &lda, x + a, &IONE, &zero, wk + (size_t)c * FB_NB, &IONE); \
            }                                                                               \
            _Pragma("omp single")                                                           \
            {                                                                               \
                for (int c = 0; c < nch; c++) for (int i = 0; i < w; i++) x[j + i] -= wk[(size_t)c * FB_NB + i]; \
                TRSV("L", "T", "N", &w, L + j + (size_t)j * lda, &lda, x + j, &IONE);       \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    free(wk);                                                                               \
}
BRISK_POTRS1(brisk_dpotrs1, double, BL(dtrsv_), BL(dgemv_))
/* 5.12: one sweep of the above (trans 0: L x = b, 1: L' x = b), for the Lanczos step lengths on
 * dense blocks (L^{-1} D L^{-T} v): the same panels and fixed chunks, so the same bits for any
 * thread count. Below 2 FB_DENSE_MIN, and inside a parallel region, the level-2 solve. */
static void dtrsv1_d(int trans, const char *dg, int n, const double *L, int lda, double *x);
void brisk_dtrsv1(int trans, int n, const double *L, int lda, double *x) { dtrsv1_d(trans, "N", n, L, lda, x); }
/* (dg "U": L with a unit diagonal, the solves of brisk_dsytrs1) */
static void dtrsv1_d(int trans, const char *dg, int n, const double *L, int lda, double *x) {
    if (n < 2 * FB_DENSE_MIN || ENV_ON("BRISK_TRSV")) { BL(dtrsv_)("L", trans ? "T" : "N", dg, &n, L, &lda, x, &IONE); return; }
    const double one = 1, mone = -1, zero = 0;
    double *wk = trans ? malloc(sizeof(double) * (size_t)((n + PS_RB - 1) / PS_RB) * FB_NB) : NULL;
    /* 5.12: the same panels inside a parallel region, on one thread there (it fell back to
     * dtrsv: other bits than at the top level) */
    #pragma omp parallel if (!omp_in_parallel())
    {
        if (!trans) for (int j = 0; j < n; j += FB_NB) {
            const int w = j + FB_NB < n ? FB_NB : n - j, r0 = j + w, nch = (n - r0 + PS_RB - 1) / PS_RB;
            #pragma omp single
            BL(dtrsv_)("L", "N", dg, &w, L + j + (size_t)j * lda, &lda, x + j, &IONE);
            #pragma omp for schedule(static)
            for (int c = 0; c < nch; c++) {
                const int a = r0 + c * PS_RB, len = a + PS_RB <= n ? PS_RB : n - a;
                BL(dgemv_)("N", &len, &w, &mone, L + a + (size_t)j * lda, &lda, x + j, &IONE, &one, x + a, &IONE);
            }
        }
        else for (int j = (n - 1) / FB_NB * FB_NB; j >= 0; j -= FB_NB) {
            const int w = j + FB_NB < n ? FB_NB : n - j, r0 = j + w, nch = (n - r0 + PS_RB - 1) / PS_RB;
            #pragma omp for schedule(static)
            for (int c = 0; c < nch; c++) {
                const int a = r0 + c * PS_RB, len = a + PS_RB <= n ? PS_RB : n - a;
                BL(dgemv_)("T", &len, &w, &one, L + a + (size_t)j * lda, &lda, x + a, &IONE, &zero, wk + (size_t)c * FB_NB, &IONE);
            }
            #pragma omp single
            {
                for (int c = 0; c < nch; c++) for (int i = 0; i < w; i++) x[j + i] -= wk[(size_t)c * FB_NB + i];
                BL(dtrsv_)("L", "T", dg, &w, L + j + (size_t)j * lda, &lda, x + j, &IONE);
            }
        }
    }
    free(wk);
}
BRISK_POTRS1(brisk_spotrs1, float, BL(strsv_), BL(sgemv_))
/* 5.12: L L' X = B for k right-hand sides (X n x k, leading dimension ldx): the panels and fixed
 * chunks of brisk_*potrs1 with a product of k columns per chunk, so that L is read once for all the
 * columns instead of once per column by a level-2 solve on one thread (the Schur solves of the
 * embedding with two or three right-hand sides). The same bits for any thread count. */
#define BRISK_POTRSK(NAME, T, TRSM, GEMM, POTRS1)                                             \
void NAME(int n, int k, const T *L, int lda, T *X, int ldx) {                               \
    if (k <= 0) return;                                                                     \
    if (k == 1) { POTRS1(n, L, lda, X); return; }                                           \
    const T one = 1, mone = -1, zero = 0;                                                   \
    if (n < 2 * FB_DENSE_MIN || ENV_ON("BRISK_TRSV")) {                                     \
        TRSM("L", "L", "N", "N", &n, &k, &one, L, &lda, X, &ldx);                           \
        TRSM("L", "L", "T", "N", &n, &k, &one, L, &lda, X, &ldx); return; }                 \
    T *wk = malloc(sizeof(T) * (size_t)((n + PS_RB - 1) / PS_RB) * FB_NB * k);              \
    _Pragma("omp parallel if (!omp_in_parallel())")                                         \
    {                                                                                       \
        for (int j = 0; j < n; j += FB_NB) {                                                \
            const int w = j + FB_NB < n ? FB_NB : n - j, r0 = j + w, nch = (n - r0 + PS_RB - 1) / PS_RB; \
            _Pragma("omp single")                                                           \
            TRSM("L", "L", "N", "N", &w, &k, &one, L + j + (size_t)j * lda, &lda, X + j, &ldx); \
            _Pragma("omp for schedule(static)")                                             \
            for (int c = 0; c < nch; c++) {                                                 \
                const int a = r0 + c * PS_RB, len = a + PS_RB <= n ? PS_RB : n - a;         \
                GEMM("N", "N", &len, &k, &w, &mone, L + a + (size_t)j * lda, &lda, X + j, &ldx, &one, X + a, &ldx); \
            }                                                                               \
        }                                                                                   \
        for (int j = (n - 1) / FB_NB * FB_NB; j >= 0; j -= FB_NB) {                         \
            const int w = j + FB_NB < n ? FB_NB : n - j, r0 = j + w, nch = (n - r0 + PS_RB - 1) / PS_RB; \
            _Pragma("omp for schedule(static)")                                             \
            for (int c = 0; c < nch; c++) {                                                 \
                const int a = r0 + c * PS_RB, len = a + PS_RB <= n ? PS_RB : n - a;         \
                GEMM("T", "N", &w, &k, &len, &one, L + a + (size_t)j * lda, &lda, X + a, &ldx, &zero, wk + (size_t)c * FB_NB * k, &w); \
            }                                                                               \
            _Pragma("omp single")                                                           \
            {                                                                               \
                for (int c = 0; c < nch; c++)                                               \
                    for (int q = 0; q < k; q++) {                                           \
                        const T *wc = wk + (size_t)c * FB_NB * k + (size_t)q * w; T *xq = X + j + (size_t)q * ldx; \
                        for (int i = 0; i < w; i++) xq[i] -= wc[i];                         \
                    }                                                                       \
                TRSM("L", "L", "T", "N", &w, &k, &one, L + j + (size_t)j * lda, &lda, X + j, &ldx); \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    free(wk);                                                                               \
}
BRISK_POTRSK(brisk_dpotrsk, double, BL(dtrsm_), BL(dgemm_), brisk_dpotrs1)
BRISK_POTRSK(brisk_spotrsk, float, BL(strsm_), BL(sgemm_), brisk_spotrs1)
/* 5.12: the full inverse of an SPD matrix from its Cholesky factor (the HKM method's Z^{-1}),
 * from n = 768 outside parallel regions: column panels of 128, W = L^{-1} by triangular solves of
 * the identity's panels, then Z^{-1} = W' W by a triangular product per panel, both panel loops
 * on the threads (each panel by single-threaded BLAS on one thread: the same bits for any thread
 * count), and the upper triangle copied. LAPACK's dpotri ran on one core: at n = 2000 115 ms
 * against 44 ms at four threads (141 ms on one; bench/dense_5.12/potri_bench.c); the results
 * agree to 3e-15 relative. A: the factor in the lower triangle on entry, the full inverse on
 * exit. Returns LAPACK's info (from dpotri below the threshold). */
int brisk_dpotri_full(int n, double *A, int lda) {
    if (n < 768 || lda != n || ENV_ON("BRISK_POTRI")) {
        int info;
        BL(dpotri_)("L", &n, A, &lda, &info);
        if (info == 0) for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) A[j + (size_t)i * lda] = A[i + (size_t)j * lda];
        return info;
    }
    for (int j = 0; j < n; j++) if (!(A[j + (size_t)j * lda] != 0)) return j + 1;
    /* 5.16 (ISSUES 83): the same panels inside a parallel region (a loop over blocks runs them on
     * one thread, with a work array of its own). Until 5.15 LAPACK's dpotri ran there: a block of
     * order >= 768 among several blocks got other bits at one thread (no active region) than at
     * two or more (a user's f24_fixed__007: 32 iterations at one thread, 28 at two). */
    static double *Ws = NULL; static size_t wn = 0;
    const int inner = omp_in_parallel();
    double *W;
    if (inner) { W = malloc(sizeof(double) * (size_t)n * n); if (!W) { int info; BL(dpotri_)("L", &n, A, &lda, &info); return info; } }
    else {
        if ((size_t)n * n > wn) { free(Ws); wn = (size_t)n * n; Ws = malloc(sizeof(double) * wn); if (!Ws) { wn = 0; int info; BL(dpotri_)("L", &n, A, &lda, &info); return info; } }
        W = Ws;
    }
    const double one = 1.0;
    const int nb = 128, np = (n + nb - 1) / nb;
    #pragma omp parallel for schedule(dynamic, 1) if (!inner)
    for (int p = 0; p < np; p++) {             /* W(j0:, J) = L(j0:, j0:)^{-1} I(j0:, J) */
        const int j0 = p * nb, nj = j0 + nb < n ? nb : n - j0, r = n - j0;
        for (int j = j0; j < j0 + nj; j++) { double *w = W + (size_t)j * n; memset(w + j0, 0, sizeof(double) * r); w[j] = 1.0; }
        BL(dtrsm_)("L", "L", "N", "N", &r, &nj, &one, A + j0 + (size_t)j0 * n, &n, W + j0 + (size_t)j0 * n, &n);
    }
    #pragma omp parallel for schedule(dynamic, 1) if (!inner)
    for (int p = 0; p < np; p++) {             /* A(j0:, J) = W(j0:, j0:)' W(j0:, J) */
        const int j0 = p * nb, nj = j0 + nb < n ? nb : n - j0, r = n - j0;
        for (int j = j0; j < j0 + nj; j++) memcpy(A + j0 + (size_t)j * n, W + j0 + (size_t)j * n, sizeof(double) * r);
        BL(dtrmm_)("L", "L", "T", "N", &r, &nj, &one, W + j0 + (size_t)j0 * n, &n, A + j0 + (size_t)j0 * n, &n);
    }
    #pragma omp parallel for schedule(static) if (!inner)
    for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) A[j + (size_t)i * n] = A[i + (size_t)j * n];
    if (inner) free(W);
    return 0;
}
/* 5.12: Householder tridiagonalization of a symmetric matrix (lower triangle), LAPACK's dsytrd
 * with uplo "L" (the same reflectors, d, e, tau, so that dormtr applies them), with its two costly
 * parts on the threads: the product of the trailing matrix with each Householder vector (half
 * of the 4/3 n^3 flops, memory-bound: one thread was 93 ms at n = 1001) in fixed column panels of
 * equal area whose partial vectors are added in panel order, and the rank-2k update after each
 * panel of 32 columns in fixed column blocks of 128. The panels and blocks depend on the sizes
 * only, so the bits do not depend on the thread count. Below order 256 LAPACK's dsytrd. */
extern void BL(dlarfg_)(const int *, double *, double *, const int *, double *);
extern void BL(dsytd2_)(const char *, const int *, double *, const int *, double *, double *, double *, int *);
extern void BL(dsyr2k_)(const char *, const char *, const int *, const int *, const double *, const double *, const int *,
                        const double *, const int *, const double *, double *, const int *);
extern void BL(dscal_)(const int *, const double *, double *, const int *);
#define TRD_NB 32
#define TRD_NX 128
#define TRD_PMAX 32
/* All the functions below run inside one parallel region per tridiagonalization (orphaned
 * work-sharing): a region per column cost more than the column's work when the caller is itself
 * in a parallel region (an inactive one, a loop over blocks with one block): libgomp creates the
 * threads of a nested team anew for each region (swissroll, n = 799: 120 ms a reduction against
 * 31 ms outside such a loop). */
/* y = A x for A s x s symmetric (lower, ld lda); buf: TRD_PMAX * s doubles */
static void trd_symv(int s, const double *A, int lda, const double *x, double *y, double *buf) {
    const double one = 1.0, zero = 0.0;
    if (s < 512) {
        #pragma omp single
        BL(dsymv_)("L", &s, &one, A, &lda, x, &IONE, &zero, y, &IONE);
        return;
    }
    int P = s / 128; if (P < 4) P = 4; if (P > TRD_PMAX) P = TRD_PMAX;
    int cut[TRD_PMAX + 1];
    for (int b = 0; b <= P; b++) cut[b] = (int)(s - s * sqrt(1.0 - (double)b / P) + 0.5);
    cut[0] = 0; cut[P] = s;
    #pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < P; b++) {
        const int c0 = cut[b], c1 = cut[b + 1], nc = c1 - c0, nr = s - c1;
        double *w = buf + (size_t)b * s;
        if (nc <= 0) continue;
        BL(dsymv_)("L", &nc, &one, A + c0 + (size_t)c0 * lda, &lda, x + c0, &IONE, &zero, w + c0, &IONE);
        if (nr > 0) {
            BL(dgemv_)("N", &nr, &nc, &one, A + c1 + (size_t)c0 * lda, &lda, x + c0, &IONE, &zero, w + c1, &IONE);
            BL(dgemv_)("T", &nr, &nc, &one, A + c1 + (size_t)c0 * lda, &lda, x + c1, &IONE, &one, w + c0, &IONE);
        }
    }
    #pragma omp for schedule(static)
    for (int i = 0; i < s; i++) {
        double t = 0;
        for (int b = 0; b < P && cut[b] <= i; b++) if (cut[b + 1] > cut[b]) t += buf[(size_t)b * s + i];
        y[i] = t;
    }
}
/* C -= V W' + W V' (lower, s x s), V and W s x k: fixed column blocks of 128 */
static void trd_syr2k(int s, int k, const double *V, int ldv, const double *W, int ldw, double *C, int ldc) {
    const double one = 1.0, mone = -1.0;
    const int nbk = 128, np = (s + nbk - 1) / nbk;
    #pragma omp for schedule(dynamic, 1)
    for (int p = 0; p < np; p++) {
        const int j0 = p * nbk, nj = s - j0 < nbk ? s - j0 : nbk, nr = s - j0 - nj;
        BL(dsyr2k_)("L", "N", &nj, &k, &mone, V + j0, &ldv, W + j0, &ldw, &one, C + j0 + (size_t)j0 * ldc, &ldc);
        if (nr > 0) {
            BL(dgemm_)("N", "T", &nr, &nj, &k, &mone, V + j0 + nj, &ldv, W + j0, &ldw, &one, C + j0 + nj + (size_t)j0 * ldc, &ldc);
            BL(dgemm_)("N", "T", &nr, &nj, &k, &mone, W + j0 + nj, &ldw, V + j0, &ldv, &one, C + j0 + nj + (size_t)j0 * ldc, &ldc);
        }
    }
}
/* LAPACK's dlatrd, uplo "L": the first nb columns of the k x k matrix A, W k x nb */
static void trd_latrd(int k, int nb, double *A, int lda, double *e, double *tau, double *W, int ldw, double *buf) {
    const double one = 1.0, zero = 0.0, mone = -1.0;
    for (int i = 0; i < nb; i++) {
        const int r = k - i;
        int q = k - i - 1;
        double *ai = A + (i + 1) + (size_t)i * lda, *wi = W + (i + 1) + (size_t)i * ldw, *wt = W + (size_t)i * ldw;
        #pragma omp single
        {
            if (i > 0) {
                BL(dgemv_)("N", &r, &i, &mone, A + i, &lda, W + i, &ldw, &one, A + i + (size_t)i * lda, &IONE);
                BL(dgemv_)("N", &r, &i, &mone, W + i, &ldw, A + i, &lda, &one, A + i + (size_t)i * lda, &IONE);
            }
            if (i < k - 1) {
                BL(dlarfg_)(&q, ai, A + (i + 2 < k ? i + 2 : k - 1) + (size_t)i * lda, &IONE, tau + i);
                e[i] = ai[0]; ai[0] = 1.0;
            }
        }
        if (i < k - 1) {
            trd_symv(q, A + (i + 1) + (size_t)(i + 1) * lda, lda, ai, wi, buf);
            #pragma omp single
            {
                if (i > 0) {
                    BL(dgemv_)("T", &q, &i, &one, W + (i + 1), &ldw, ai, &IONE, &zero, wt, &IONE);
                    BL(dgemv_)("N", &q, &i, &mone, A + (i + 1), &lda, wt, &IONE, &one, wi, &IONE);
                    BL(dgemv_)("T", &q, &i, &one, A + (i + 1), &lda, ai, &IONE, &zero, wt, &IONE);
                    BL(dgemv_)("N", &q, &i, &mone, W + (i + 1), &ldw, wt, &IONE, &one, wi, &IONE);
                }
                BL(dscal_)(&q, tau + i, wi, &IONE);
                const double alpha = -0.5 * tau[i] * BL(ddot_)(&q, wi, &IONE, ai, &IONE);
                BL(daxpy_)(&q, &alpha, ai, &IONE, wi, &IONE);
            }
        }
    }
}
int brisk_dsytrd_low(int n, double *A, int lda, double *d, double *e, double *tau) {
    int info = 0;
    if (n < 256) {
        int lwork = -1; double wq;
        BL(dsytrd_)("L", &n, A, &lda, d, e, tau, &wq, &lwork, &info);
        lwork = (int)wq + 1;
        double *work = malloc(sizeof(double) * (size_t)lwork);
        BL(dsytrd_)("L", &n, A, &lda, d, e, tau, work, &lwork, &info);
        free(work);
        return info;
    }
    double *W = malloc(sizeof(double) * (size_t)n * TRD_NB), *buf = malloc(sizeof(double) * (size_t)n * TRD_PMAX);
    if (!W || !buf) { free(W); free(buf); return -1; }
    int iend = 0;
    #pragma omp parallel if (!omp_in_parallel())
    {
        int i = 0;
        for (; i + 1 <= n - TRD_NX; i += TRD_NB) {
            const int k = n - i;
            trd_latrd(k, TRD_NB, A + i + (size_t)i * lda, lda, e + i, tau + i, W, k, buf);
            trd_syr2k(k - TRD_NB, TRD_NB, A + (i + TRD_NB) + (size_t)i * lda, lda, W + TRD_NB, k, A + (i + TRD_NB) + (size_t)(i + TRD_NB) * lda, lda);
            #pragma omp single
            for (int j = i; j < i + TRD_NB; j++) { A[(j + 1) + (size_t)j * lda] = e[j]; d[j] = A[j + (size_t)j * lda]; }
        }
        #pragma omp single
        iend = i;
    }
    const int nl = n - iend;
    BL(dsytd2_)("L", &nl, A + iend + (size_t)iend * lda, &lda, d + iend, e + iend, tau + iend, &info);
    free(W); free(buf);
    return info;
}

/* 4.24: BRISK owns the threads: OpenBLAS runs single-threaded (its pthread workers spin
 * next to the OpenMP ones: case14 0.02 -> 0.14 s at 2 threads; its threaded kernels also
 * change the results with the thread count). BRISK_BLASMT=1 keeps OpenBLAS threading. */
static int blas_T_host = 0;      /* the OpenBLAS thread count found at the first call of a run */
#ifdef __APPLE__
/* 5.11 (a user's report, 8 October 2026): on macOS the BLAS is Accelerate, which runs threads of
 * its own: trto5 at -threads 1 used 200 % of the CPU on an M1 and 400 % on an Intel i9, and on
 * the M1 the iteration count changed from run to run (its threaded kernels add in an order of
 * their own). BLASSetThreading (macOS 15; the setting is per thread) asks for one thread:
 * value 1 (BLAS_THREADING_SINGLE_THREADED of vecLib's thread_api.h; 0 lets Accelerate decide).
 * Looked up at run time, so that older systems and other BLAS libraries are left as they are;
 * set on the calling thread and on every OpenMP thread, and the caller's setting is given back
 * at the end of a library call (BLASGetThreading).
 * 5.15: before macOS 15 (no BLASSetThreading) VECLIB_MAXIMUM_THREADS=1 is set in the process,
 * unless it is set already. Accelerate reads it at its first threaded call, so setting it here,
 * before BRISK's first BLAS call, is enough (the variable outside the process is not needed).
 * Without it, on an M3 under macOS 14 at -threads 1, 3-5 % of the runs of theta4 took another
 * path (Accelerate's LAPACK ran on 7 dispatch threads); with it, 200 of 200 runs were the same.
 * In a library host the setting stays for the rest of the process (as when the user sets it). */
#include <dlfcn.h>
static int (*accel_set)(unsigned) = NULL;
static unsigned (*accel_get)(void) = NULL;
static int accel_looked = 0;
static long accel_host = -1;          /* the caller's setting, to give back (-1: none saved) */
static void accel_single(void) {
    if (!accel_looked) {
        accel_looked = 1;
        accel_set = (int (*)(unsigned))dlsym(RTLD_DEFAULT, "BLASSetThreading");
        accel_get = (unsigned (*)(void))dlsym(RTLD_DEFAULT, "BLASGetThreading");
    }
    if (!accel_set) { if (!getenv("VECLIB_MAXIMUM_THREADS")) setenv("VECLIB_MAXIMUM_THREADS", "1", 1); return; }
    if (accel_get && accel_host < 0) accel_host = (long)accel_get();
    (void)accel_set(1u);
    #pragma omp parallel
    (void)accel_set(1u);
}
#endif
void brisk_threads_init(void) {
    if (getenv("BRISK_BLASMT")) return;
#ifdef __APPLE__
    accel_single();
#endif
    if (BL(openblas_set_num_threads)) {
        if (BL(openblas_get_num_threads) && blas_T_host == 0) { const int t = BL(openblas_get_num_threads)(); if (t > 1) blas_T_host = t; }
        BL(openblas_set_num_threads)(1);
    }
}
/* Library interfaces: the host gets its OpenBLAS thread count back when the call returns.
 * A host that shares this OpenBLAS (Python with SciPy and BLAS=scipy, Octave) was left with
 * one BLAS thread for the rest of the process: scipy dgemm, n = 1500, 0.06 -> 0.13 s on two
 * cores after the first solve. */
void brisk_threads_restore(void) {
#ifdef __APPLE__
    if (accel_set && accel_host >= 0) (void)accel_set((unsigned)accel_host);
    accel_host = -1;
#endif
    if (blas_T_host > 1 && BL(openblas_set_num_threads)) BL(openblas_set_num_threads)(blas_T_host);
    blas_T_host = 0;
}

/* 4.24: tiled dgemm / dsymm for large products, parallel outside parallel regions. The
 * tiles are fixed by the sizes, so the arithmetic is the same for any thread count;
 * below PD_MIN (m n k) the plain BLAS call. */
#define PD_MIN 8e6
/* 4.25: 2048 x 512 tiles (1024 x 256 in 4.24 cost 29% against one dgemm at n = 5000 - A was
 * packed once per column tile - and made maxG55/60 10-15% slower than 4.13; now 11%) */
static int PD_RB = 2048, PD_CB = 512;   /* fixed tiles (BRISK_PDRB, BRISK_PDCB): they define the arithmetic */
static int pd_cb_fixed = 0;
static void pd_env(void) {
    /* 5.16: called from several threads at once (pdgemm inside the parallel Schur loops; a
     * ThreadSanitizer run reported it): the settings are read once, under a lock, and published
     * with release/acquire order, so no thread sees the flag before the values */
    static int done = 0;
    if (__atomic_load_n(&done, __ATOMIC_ACQUIRE)) return;
    #pragma omp critical(brisk_pd_env)
    if (!__atomic_load_n(&done, __ATOMIC_RELAXED)) {
        const char *e;
        if ((e = getenv("BRISK_PDRB"))) PD_RB = atoi(e);
        if ((e = getenv("BRISK_PDCB"))) { PD_CB = atoi(e); pd_cb_fixed = 1; }
        if ((e = getenv("BRISK_FNB"))) { FB_NB = atoi(e); g_fd_fixed = 1; }
        if ((e = getenv("BRISK_FDRB"))) { FD_RB = atoi(e); g_fd_fixed = 1; }
        __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
    }
}
/* 5.12: the reference tiles and verified fast cuts. The fixed tiles below (pd_ref_*) define the
 * arithmetic: they depend on the sizes only, so the bits do not depend on the thread count. Two
 * faster ways give the same bits whenever the BLAS computes every entry the same way wherever
 * it sits in the cut (OpenBLAS does from order ~1000, not always below): one call on one thread
 * (the reference tiles pack the large operand once per tile: 17-19 % more at n = 1761 ... 3000)
 * and one tile per thread otherwise (better balanced than the fixed tiles). Neither is assumed:
 * for each kind and size at a thread count both ways are computed once on random data of that
 * size, leading dimensions and alignment (pd_verify), and the fast way is used only if the bits
 * are the same (OpenBLAS: yes at n = 1000-3000 for every cut on multiples of 16; at n = 500 not
 * for three tiles). Two extra products per size and thread count; above 256 MB of result no
 * check (the reference). Side "R" products are cut in row tiles of PD_CB (512) since 5.12: with tiles of
 * PD_RB (2048) a product of order below 2048 was one tile, on one thread (maxG32: half of the
 * solve). BRISK_PDFIXED=1: always the reference; BRISK_PDDBG=1: the verdicts. */
/* 5.12: below 1024 columns the column tiles are 128 wide (were 512: a product of order 462 was one
 * tile, on one thread; bench/dense_5.12/trsm_blocked.txt). By the size only: the same bits for
 * any thread count. */
static inline int pd_cb(int len) { return (len < 1024 && !pd_cb_fixed) ? 128 : PD_CB; }
static void pd_ref_gemm(const char *ta, const char *tb, int m, int n, const int *k_, const double *alpha, const double *A, int lda,
                        const double *B, int ldb, const double *beta, double *C, int ldc) {
    const int tA = ta[0] == 'T' || ta[0] == 't', tB = tb[0] == 'T' || tb[0] == 't';
    const int CB = pd_cb(n);
    const int nr = (m + PD_RB - 1) / PD_RB, nc = (n + CB - 1) / CB, nt = nr * nc;
    #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel() && nt > 1)
    for (int t = 0; t < nt; t++) {
        const int i0 = (t % nr) * PD_RB, j0 = (t / nr) * CB;
        const int mi = m - i0 < PD_RB ? m - i0 : PD_RB, nj = n - j0 < CB ? n - j0 : CB;
        const double *Ai = tA ? A + (size_t)i0 * lda : A + i0;
        const double *Bj = tB ? B + j0 : B + (size_t)j0 * ldb;
        BL(dgemm_)(ta, tb, &mi, &nj, k_, alpha, Ai, &lda, Bj, &ldb, beta, C + i0 + (size_t)j0 * ldc, &ldc);
    }
}
static void pd_ref_symm(const char *side, const char *uplo, int m, int n, const double *alpha, const double *A, const int *lda_,
                        const double *B, int ldb, const double *beta, double *C, int ldc) {
    const int left = side[0] == 'L' || side[0] == 'l';
    const int CB = pd_cb(left ? n : m);
    const int nt = left ? (n + CB - 1) / CB : (m + CB - 1) / CB;
    #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel() && nt > 1)
    for (int t = 0; t < nt; t++) {
        const int a0 = t * CB, len = (left ? n : m) - a0 < CB ? (left ? n : m) - a0 : CB;
        if (left) BL(dsymm_)(side, uplo, &m, &len, alpha, A, lda_, B + (size_t)a0 * ldb, &ldb, beta, C + (size_t)a0 * ldc, &ldc);    /* C(:, J) = A B(:, J) */
        else BL(dsymm_)(side, uplo, &len, &n, alpha, A, lda_, B + a0, &ldb, beta, C + a0, &ldc);                                    /* C(I, :) = B(I, :) A */
    }
}
/* the fast way: one call for T = 1, else T tiles of the longer side (gemm) or of the side the
 * symm keeps whole (columns for "L", rows for "R"), cut on multiples of 16 */
static void pd_fast(int symm, const char *a, const char *b, int m, int n, const int *k_, const double *alpha, const double *A, const int *lda_,
                    const double *B, int ldb, const double *beta, double *C, int ldc, int T) {
    const int lda = *lda_;
    const int rows = symm ? !(a[0] == 'L' || a[0] == 'l') : m >= n;
    if (T <= 1) {
        if (symm) BL(dsymm_)(a, b, &m, &n, alpha, A, lda_, B, &ldb, beta, C, &ldc);
        else BL(dgemm_)(a, b, &m, &n, k_, alpha, A, lda_, B, &ldb, beta, C, &ldc);
        return;
    }
    const int tA = !symm && (a[0] == 'T' || a[0] == 't'), tB = !symm && (b[0] == 'T' || b[0] == 't');
    const int len = rows ? m : n, u = (len + 15) / 16;
    #pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < T; t++) {
        int a0 = (int)((long)u * t / T) * 16, a1 = (int)((long)u * (t + 1) / T) * 16;
        if (a1 > len) a1 = len;
        if (a0 >= a1) continue;
        const int l = a1 - a0;
        if (symm) {
            if (rows) BL(dsymm_)(a, b, &l, &n, alpha, A, lda_, B + a0, &ldb, beta, C + a0, &ldc);
            else BL(dsymm_)(a, b, &m, &l, alpha, A, lda_, B + (size_t)a0 * ldb, &ldb, beta, C + (size_t)a0 * ldc, &ldc);
        } else {
            if (rows) BL(dgemm_)(a, b, &l, &n, k_, alpha, tA ? A + (size_t)a0 * lda : A + a0, lda_, B, &ldb, beta, C + a0, &ldc);
            else BL(dgemm_)(a, b, &m, &l, k_, alpha, A, lda_, tB ? B + a0 : B + (size_t)a0 * ldb, &ldb, beta, C + (size_t)a0 * ldc, &ldc);
        }
    }
}
#define PD_NV 64
static struct { int kind, m, n, k, T; signed char ok; } pd_ver[PD_NV];
static int pd_nver = 0;
/* -1 unknown, 0 the reference, 1 the fast way (for this kind, size and thread count) */
static int pd_lookup(int kind, int m, int n, int k, int T) {
    for (int i = 0; i < pd_nver; i++)
        if (pd_ver[i].kind == kind && pd_ver[i].m == m && pd_ver[i].n == n && pd_ver[i].k == k && pd_ver[i].T == T) return pd_ver[i].ok;
    return -1;
}
/* does the fast way give the reference's bits for products of this kind and size? Checked on
 * random data with the same sizes, leading dimensions and alignments (mod 64 bytes) as the call:
 * the solver's own first product can have a special structure (Z^{-1} nearly diagonal at the
 * start: entries that are single exact terms), and mcp500-1 passed the check on it and then
 * differed at three threads. */
static int pd_verify(int symm, const char *a, const char *b, int m, int n, const int *k_, int lda, int ldb, int ldc,
                     const void *pA, const void *pB, const void *pC, int T) {
    const int left = symm && (a[0] == 'L' || a[0] == 'l');
    const int tA = !symm && (a[0] == 'T' || a[0] == 't'), tB = !symm && (b[0] == 'T' || b[0] == 't');
    const int k = symm ? (left ? m : n) : *k_;
    const size_t la = (size_t)lda * (symm ? k : (tA ? m : k)), lb = (size_t)ldb * (symm ? n : (tB ? k : n)), lc = (size_t)ldc * n;
    const size_t oa = (uintptr_t)pA % 64, ob = (uintptr_t)pB % 64, oc = (uintptr_t)pC % 64;
    char *bA = malloc(sizeof(double) * la + 128), *bB = malloc(sizeof(double) * lb + 128);
    char *bC = malloc(sizeof(double) * lc + 128), *bD = malloc(sizeof(double) * lc + 128);
    if (!bA || !bB || !bC || !bD) { free(bA); free(bB); free(bC); free(bD); return 0; }
    double *A = (double *)(bA + (64 - (uintptr_t)bA % 64) % 64 + oa), *B = (double *)(bB + (64 - (uintptr_t)bB % 64) % 64 + ob);
    double *C1 = (double *)(bC + (64 - (uintptr_t)bC % 64) % 64 + oc), *C2 = (double *)(bD + (64 - (uintptr_t)bD % 64) % 64 + oc);
    uint64_t r = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)m << 32) ^ (uint64_t)n;
    for (size_t i = 0; i < la; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; A[i] = (double)(r >> 11) * 0x1.0p-53 - 0.5; }
    for (size_t i = 0; i < lb; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; B[i] = (double)(r >> 11) * 0x1.0p-53 - 0.5; }
    const double one = 1.0, zero = 0.0;
    if (symm) pd_ref_symm(a, b, m, n, &one, A, &lda, B, ldb, &zero, C1, ldc);
    else pd_ref_gemm(a, b, m, n, k_, &one, A, lda, B, ldb, &zero, C1, ldc);
    pd_fast(symm, a, b, m, n, k_, &one, A, &lda, B, ldb, &zero, C2, ldc, T);
    int same = 1;
    for (int j = 0; j < n && same; j++) same = memcmp(C1 + (size_t)j * ldc, C2 + (size_t)j * ldc, sizeof(double) * m) == 0;
    free(bA); free(bB); free(bC); free(bD);
    return same;
}
/* the product: the fast way where it was verified to give the reference's bits for this kind,
 * size and thread count, else the reference tiles */
static void pd_run(int symm, const char *a, const char *b, int m, int n, const int *k_, const double *alpha, const double *A, const int *lda_,
                   const double *B, int ldb, const double *beta, double *C, int ldc) {
    static int fixed = -1, dbg = -1;
    if (fixed < 0) { fixed = getenv("BRISK_PDFIXED") != NULL; dbg = getenv("BRISK_PDDBG") != NULL; }
    int T = 1;
#ifdef _OPENMP
    T = omp_in_parallel() ? 0 : omp_get_max_threads();       /* inside a region: the reference, serially */
#endif
    const int kind = (symm ? 8 : 0) + 2 * (a[0] == 'T' || a[0] == 't' || a[0] == 'R' || a[0] == 'r') + (b[0] == 'T' || b[0] == 't' || b[0] == 'U' || b[0] == 'u');
    const int k = symm ? 0 : *k_;
    int v = (fixed || T == 0) ? 0 : pd_lookup(kind, m, n, k, T);
    if (v < 0) {
        if (pd_nver >= PD_NV || (double)ldc * n * 8.0 > 256e6) v = 0;
        else {
            v = pd_verify(symm, a, b, m, n, k_, *lda_, ldb, ldc, A, B, C, T);
            pd_ver[pd_nver].kind = kind; pd_ver[pd_nver].m = m; pd_ver[pd_nver].n = n; pd_ver[pd_nver].k = k; pd_ver[pd_nver].T = T; pd_ver[pd_nver].ok = (signed char)v;
            pd_nver++;
            if (dbg) printf("   [pd: %s %c%c %d x %d x %d at %d threads: %s]\n", symm ? "dsymm" : "dgemm", a[0], b[0], m, n, symm ? (a[0] == 'L' || a[0] == 'l' ? m : n) : k, T,
                            v ? "the fast way gives the same bits" : "the reference tiles");
        }
    }
    if (v == 1) { pd_fast(symm, a, b, m, n, k_, alpha, A, lda_, B, ldb, beta, C, ldc, T); return; }
    if (symm) pd_ref_symm(a, b, m, n, alpha, A, lda_, B, ldb, beta, C, ldc);
    else pd_ref_gemm(a, b, m, n, k_, alpha, A, *lda_, B, ldb, beta, C, ldc);
}
void pdgemm(const char *ta, const char *tb, const int *m_, const int *n_, const int *k_, const double *alpha,
            const double *A, const int *lda_, const double *B, const int *ldb_, const double *beta, double *C, const int *ldc_) {
    const int m = *m_, n = *n_, k = *k_;
    pd_env();
    if ((double)m * n * k < PD_MIN || getenv_cached_pd()) { BL(dgemm_)(ta, tb, m_, n_, k_, alpha, A, lda_, B, ldb_, beta, C, ldc_); return; }
    pd_run(0, ta, tb, m, n, k_, alpha, A, lda_, B, *ldb_, beta, C, *ldc_);
}
void pdsymm(const char *side, const char *uplo, const int *m_, const int *n_, const double *alpha,
            const double *A, const int *lda_, const double *B, const int *ldb_, const double *beta, double *C, const int *ldc_) {
    const int m = *m_, n = *n_;
    const int left = side[0] == 'L' || side[0] == 'l', k = left ? m : n;
    pd_env();
    if ((double)m * n * k < PD_MIN || getenv_cached_pd()) { BL(dsymm_)(side, uplo, m_, n_, alpha, A, lda_, B, ldb_, beta, C, ldc_); return; }
    pd_run(1, side, uplo, m, n, NULL, alpha, A, lda_, B, *ldb_, beta, C, *ldc_);
}
/* 5.12: triangular solves and products with many right-hand sides on the threads. The
 * right-hand side splits into independent fixed panels (128 columns of B for side "L", 128
 * rows for side "R"), each one single-threaded BLAS call; the panels are the same inside a
 * parallel region (computed one after the other there), so the bits do not depend on the
 * thread count. Below PD_MIN flops (m n k) one call as before. The corrector of the embedding
 * spent 3.8 of thetaG51's 28 s in these calls on one thread. */
#define PT_W 128
/* 5.12: one panel by blocks of PT_NB: the triangle's diagonal blocks by dtrsm / dtrmm, the rest by
 * dgemm updates (L lower, non-unit). OpenBLAS's dtrsm on these shapes (order 462-792, 128 columns)
 * ran at 12-20 GF/s on one core, the blocked form at 25-34 (bench/dense_5.12/trsm_blocked.txt);
 * the blocks are fixed, so the bits do not depend on the thread count. BRISK_PTBLK0=1: dtrsm. */
#define PT_NB 64
static int g_ptblk = -1;
static void pt_blocked(int solve, int left, int tr, int m, int n, double alpha, const double *L, int ldl, double *B, int ldb) {
    const double one = 1.0, mone = -1.0;
    const int k = left ? m : n, nbk = (k + PT_NB - 1) / PT_NB;
    if (alpha != 1.0) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) B[i + (size_t)j * ldb] *= alpha;
    /* forward (block 0 first) or backward, by the variant: see the comments of each case */
    for (int s = 0; s < nbk; s++) {
        int q;
        if (solve) q = (left ? !tr : tr) ? s : nbk - 1 - s;      /* L^{-1}B, B L^{-T}: forward; L^{-T}B, B L^{-1}: backward */
        else q = (left ? tr : !tr) ? s : nbk - 1 - s;             /* L'B, B L: forward (they read the rows/columns not yet changed) */
        const int j = q * PT_NB, b = k - j < PT_NB ? k - j : PT_NB, r = k - j - b;
        const double *Ljj = L + j + (size_t)j * ldl;
        if (left) {
            double *Bj = B + j;
            if (solve) {
                BL(dtrsm_)("L", "L", tr ? "T" : "N", "N", &b, &n, &one, Ljj, &ldl, Bj, &ldb);
                if (!tr && r > 0) BL(dgemm_)("N", "N", &r, &n, &b, &mone, Ljj + b, &ldl, Bj, &ldb, &one, Bj + b, &ldb);   /* B_below -= L_below,j X_j */
                if (tr && j > 0) BL(dgemm_)("T", "N", &j, &n, &b, &mone, L + j, &ldl, Bj, &ldb, &one, B, &ldb);         /* B_above -= L_j,above' X_j */
            } else {
                BL(dtrmm_)("L", "L", tr ? "T" : "N", "N", &b, &n, &one, Ljj, &ldl, Bj, &ldb);
                if (tr && r > 0) BL(dgemm_)("T", "N", &b, &n, &r, &one, Ljj + b, &ldl, Bj + b, &ldb, &one, Bj, &ldb);    /* + L_below,j' B_below */
                if (!tr && j > 0) BL(dgemm_)("N", "N", &b, &n, &j, &one, L + j, &ldl, B, &ldb, &one, Bj, &ldb);           /* + L_j,above B_above */
            }
        } else {
            double *Bj = B + (size_t)j * ldb;
            if (solve) {
                BL(dtrsm_)("R", "L", tr ? "T" : "N", "N", &m, &b, &one, Ljj, &ldl, Bj, &ldb);
                if (!tr && j > 0) BL(dgemm_)("N", "N", &m, &j, &b, &mone, Bj, &ldb, L + j, &ldl, &one, B, &ldb);           /* B_left -= X_j L_j,left */
                if (tr && r > 0) BL(dgemm_)("N", "T", &m, &r, &b, &mone, Bj, &ldb, Ljj + b, &ldl, &one, Bj + (size_t)b * ldb, &ldb);  /* B_right -= X_j L_right,j' */
            } else {
                BL(dtrmm_)("R", "L", tr ? "T" : "N", "N", &m, &b, &one, Ljj, &ldl, Bj, &ldb);
                if (!tr && r > 0) BL(dgemm_)("N", "N", &m, &b, &r, &one, Bj + (size_t)b * ldb, &ldb, Ljj + b, &ldl, &one, Bj, &ldb);  /* + B_right L_right,j */
                if (tr && j > 0) BL(dgemm_)("N", "T", &m, &b, &j, &one, B, &ldb, L + j, &ldl, &one, Bj, &ldb);           /* + B_left L_j,left' */
            }
        }
    }
}
static void pdtri(int solve, const char *side, const char *uplo, const char *ta, const char *diag, const int *m_, const int *n_,
                  const double *alpha, const double *A, const int *lda_, double *B, const int *ldb_) {
    const int m = *m_, n = *n_, left = side[0] == 'L' || side[0] == 'l', k = left ? m : n;
    pd_env();
    if ((double)m * n * k < PD_MIN || getenv_cached_pd() || (left ? n : m) <= PT_W) {
        if (solve) BL(dtrsm_)(side, uplo, ta, diag, m_, n_, alpha, A, lda_, B, ldb_);
        else BL(dtrmm_)(side, uplo, ta, diag, m_, n_, alpha, A, lda_, B, ldb_);
        return;
    }
    const int ldb = *ldb_, np = ((left ? n : m) + PT_W - 1) / PT_W;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    if (g_ptblk < 0) g_ptblk = !ENV_ON("BRISK_PTBLK0");
    const int blk = g_ptblk && (uplo[0] == 'L' || uplo[0] == 'l') && (diag[0] == 'N' || diag[0] == 'n') && k > 2 * PT_NB;
    #pragma omp parallel for schedule(dynamic, 1) if (nt > 1)
    for (int p = 0; p < np; p++) {
        const int r0 = p * PT_W, w = ((left ? n : m) - r0) < PT_W ? ((left ? n : m) - r0) : PT_W;
        double *Bp = left ? B + (size_t)r0 * ldb : B + r0;
        const int mm = left ? m : w, nn = left ? w : n;
        if (blk) pt_blocked(solve, left, ta[0] == 'T' || ta[0] == 't', mm, nn, *alpha, A, *lda_, Bp, ldb);
        else if (solve) BL(dtrsm_)(side, uplo, ta, diag, &mm, &nn, alpha, A, lda_, Bp, ldb_);
        else BL(dtrmm_)(side, uplo, ta, diag, &mm, &nn, alpha, A, lda_, Bp, ldb_);
    }
}
void pdtrsm(const char *side, const char *uplo, const char *ta, const char *diag, const int *m_, const int *n_,
            const double *alpha, const double *A, const int *lda_, double *B, const int *ldb_) {
    pdtri(1, side, uplo, ta, diag, m_, n_, alpha, A, lda_, B, ldb_);
}
void pdtrmm(const char *side, const char *uplo, const char *ta, const char *diag, const int *m_, const int *n_,
            const double *alpha, const double *A, const int *lda_, double *B, const int *ldb_) {
    pdtri(0, side, uplo, ta, diag, m_, n_, alpha, A, lda_, B, ldb_);
}
/* 18 October 2026: a large node as one task graph over square tiles of g_ftile (the rows of its own columns tiled as
 * the columns, the rows below from w): each tile's update from the earlier nodes (fnode_upd, the sources in their
 * order), then the right-looking Cholesky of the panel (potrf, trsm, syrk/gemm per tile), every task waiting only
 * for the tiles it reads. Before, the updates and each column step of the panel were separate phases of tasks with
 * a barrier between them, and the update tiles (256 x 256, the last 24 columns wide on broyden25's nodes of 792)
 * left threads idle: a node of broyden25's chain (792 columns, 1715 rows) took 16 ms, 12 ms as one graph. The order
 * of the updates of a tile is fixed by the graph, so the result does not depend on the thread count.
 * BRISK_FTILE=0: the phases. */
static int g_ftile = -1;
/* (sparsechol-dev) the tile is twice g_ftile on a node of at least g_ftilew such double tiles (264 columns from a
 * width of 792): the inner dimension of the graph's products doubles where three column tiles still give the graph
 * enough tasks (broyden25's nodes of 792 columns: its factorizations 4.73 -> 4.45 s, mbb14weight's of 968 and 3528
 * 4.72 -> 4.13 s; a tile of 264 for every node made the AC-OPF relaxations' nodes of 133-264 columns leave the graph,
 * and two tiles (from 528 columns) slowed a node of 569 columns). It depends on the node's width only: the same bits
 * at any thread count. BRISK_FTILEW (3; 0 = g_ftile everywhere). */
static int g_ftilew = -1;
static int g_ftile_lp = -1;     /* (18 October 2026, LP/SOCP: the tiles of the LP method's nodes, tinydag; BRISK_LPFTILE.
                                 * 264: the LP's nodes are few and large (set covering 8,000 rows: factorizations 29 ->
                                 * 24 s against 132; min-cost flow 110,000 arcs: 107 -> 81 s; 396 no better). The LP's
                                 * rule comes first; the width rule above is for the other factorizations) */
static int fnode_dag(SChol *S, int k, int par) {
    if (g_ftilew < 0) { const char *e = getenv("BRISK_FTILEW"); g_ftilew = e ? atoi(e) : 3; }
    if (g_ftile_lp < 0) { const char *e = getenv("BRISK_LPFTILE"); g_ftile_lp = e ? atoi(e) : 264; }
    const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
    const int nb = S->tinydag && g_ftile_lp > 0 ? g_ftile_lp : g_ftilew > 0 && w >= g_ftilew * 2 * g_ftile ? 2 * g_ftile : g_ftile;
    double *Pk = S->pan + S->pbase[k];
    const int nct = (w + nb - 1) / nb, nrt = nct + (nr - w + nb - 1) / nb;
    int *r0 = malloc(sizeof(int) * (size_t)(nrt + 1));
    for (int i = 0; i < nct; i++) r0[i] = i * nb;
    for (int i = nct; i < nrt; i++) r0[i] = w + (i - nct) * nb;
    r0[nrt] = nr;
    int bad = 0;
    #define FT(i, j) (Pk + (size_t)r0[j] * nr + r0[i])
    for (int j = 0; j < nct; j++)
        for (int i = j; i < nrt; i++) {
            double *Cij = FT(i, j);
            const int R0 = r0[i], R1 = r0[i + 1], C0 = r0[j], C1 = r0[j + 1];
            (void)Cij;                                   /* (read by the depend clause only) */
            #pragma omp task if(par) depend(out: Cij[0]) firstprivate(R0, R1, C0, C1)
            fnode_upd(S, k, R0, R1, C0, C1, fscr_get());
        }
    for (int c = 0; c < nct; c++) {
        double *D = FT(c, c);
        const int cb = r0[c + 1] - r0[c];
        if (S->tinypiv > 0) {
            /* (the LP's tiny-pivot rule, as chol_tiny_blocked: a pivot below 10 tinypiv of its original diagonal becomes
             * 1e128 and its column of L is zero; the original diagonal of the tile from pmat) */
            const double *D0 = S->pmat + S->pbase[k] + (size_t)r0[c] * nr + r0[c];
            #pragma omp task if(par) depend(inout: D[0]) firstprivate(D, D0, cb)
            { const int nt = chol_tiny_blocked(cb, cb, D, nr, D0, nr + 1, 10.0 * S->tinypiv);
              if (nt) {
                #pragma omp atomic
                S->ntiny += nt; } }
        } else {
        #pragma omp task if(par) depend(inout: D[0]) firstprivate(D, cb) shared(bad)
        { const int info = sc_potrf(cb, D, nr); if (info) {
            #pragma omp atomic write
            bad = 1; } }
        }
        for (int i = c + 1; i < nrt; i++) {
            double *Ti = FT(i, c);
            const int rr = r0[i + 1] - r0[i];
            #pragma omp task if(par) depend(in: D[0]) depend(inout: Ti[0]) firstprivate(D, Ti, rr, cb)
            { const double t0 = g_ftime > 0 ? omp_get_wtime() : 0; sc_trsm(rr, cb, D, nr, Ti, nr);
              if (S->tinypiv > 0) for (int q = 0; q < cb; q++) if (D[(size_t)q * nr + q] >= 1e63) { double *col = Ti + (size_t)q * nr; for (int r = 0; r < rr; r++) col[r] = 0; }
              if (g_ftime > 0) { const double t1 = omp_get_wtime();
                #pragma omp atomic
                g_ftd[0] += t1 - t0; } }
        }
        for (int j = c + 1; j < nct; j++) {
            double *Ajc = FT(j, c);
            const int cj = r0[j + 1] - r0[j];
            for (int i = j; i < nrt; i++) {
                double *Aic = FT(i, c), *Cij = FT(i, j);
                const int rr = r0[i + 1] - r0[i];
                if (i == j) {
                    #pragma omp task if(par) depend(in: Ajc[0]) depend(inout: Cij[0]) firstprivate(Ajc, Cij, cj, cb)
                    { const double mone = -1.0; BL(dsyrk_)("L", "N", &cj, &cb, &mone, Ajc, &nr, &DONE, Cij, &nr); }
                } else {
                    #pragma omp task if(par) depend(in: Aic[0], Ajc[0]) depend(inout: Cij[0]) firstprivate(Aic, Ajc, Cij, rr, cj, cb)
                    { const double mone = -1.0, t0 = g_ftime > 0 ? omp_get_wtime() : 0; BL(dgemm_)("N", "T", &rr, &cj, &cb, &mone, Aic, &nr, Ajc, &nr, &DONE, Cij, &nr);
                      if (g_ftime > 0) { const double t1 = omp_get_wtime();
                        #pragma omp atomic
                        g_ftd[1] += t1 - t0; } }
                }
            }
        }
    }
    #undef FT
    #pragma omp taskwait
    free(r0);
    return bad;
}
/* one node: shift, updates (row blocks, as tasks when par), panel factorization */
static void fnode(SChol *S, int k, double shift, int par) {
    if (S->fail) return;
    const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
    double *Pk = S->pan + S->pbase[k];
    if (shift != 0)
        for (int cc = 0; cc < w; cc++) { double *d = &Pk[(size_t)cc * nr + cc]; *d += shift * (fabs(*d) > 0 ? fabs(*d) : 1.0); }
    const int nblk = (nr + FB_RB - 1) / FB_RB;
    /* 17 October 2026: the updates also by blocks of FB_CB target columns (the SDP path's factorizations; the cone
     * solver's, with dynamic pivots, keep whole rows of tiles). A chain of large supernodes (the dense cliques of a
     * chained SOS relaxation: broyden25, 19 nodes of about 840 columns) gave a node's update from its one child
     * as three row tasks for four threads. The tiles are fixed, so the arithmetic is the same at any thread count
     * (BRISK_FCB: their width, 0 = whole rows as before). */
    if (g_fcb < 0) { const char *e = getenv("BRISK_FCB"); g_fcb = e ? atoi(e) : 256; }
    if (g_ftile < 0) { const char *e = getenv("BRISK_FTILE"); g_ftile = e ? atoi(e) : 132; }
    /* (every such node, not only those of the parallel part of the tree, which depends on the thread count: on the
     * others the tasks run in place, in the order made, so the arithmetic is the same) */
    if (g_ftile > 0 && w > g_ftile && !(S->dynpiv > 0) && (!(S->tinypiv > 0) || S->tinydag)) {
        int neg = 0;
        if (S->sgn) for (int c = 0; c < w; c++) if (S->sgn[S->ss[k] + c] < 0) { neg = 1; break; }
        if (!neg) {
            const double tp0 = g_ftime > 0 ? omp_get_wtime() : 0;
            if (fnode_dag(S, k, par)) S->fail = 1;
            if (g_ftime > 0) { const double tp1 = omp_get_wtime();
                #pragma omp atomic
                g_ft[2] += tp1 - tp0; }
            return;
        }
    }
    const int cbw = (g_fcb > 0 && !(S->dynpiv > 0)) ? g_fcb : w, ncb = (w + cbw - 1) / cbw;
    if (par && nblk * ncb > 1) {
        for (int b = 0; b < nblk; b++)
            for (int cb = 0; cb < ncb; cb++) {
                #pragma omp task firstprivate(b, cb)
                fnode_upd(S, k, b * FB_RB, (b + 1) * FB_RB < nr ? (b + 1) * FB_RB : nr, cb * cbw, (cb + 1) * cbw < w ? (cb + 1) * cbw : w, fscr_get());
            }
        #pragma omp taskwait
    } else {
        /* (17 October 2026: a large node's tiles on the team of panel_team, when the tree runs sequentially; the
         * tiles write disjoint parts of the node, so the results are those of the loop) */
        const int tn = nblk * ncb > 1 ? panel_team(S->swk[k]) : 1;
        if (tn > 1) {                                   /* (a team only when there is one: a region per node costs) */
            #pragma omp parallel for schedule(dynamic, 1) num_threads(tn)
            for (int bc = 0; bc < nblk * ncb; bc++) {
                const int b = bc / ncb, cb = bc % ncb;
                fnode_upd(S, k, b * FB_RB, (b + 1) * FB_RB < nr ? (b + 1) * FB_RB : nr, cb * cbw, (cb + 1) * cbw < w ? (cb + 1) * cbw : w, fscr_get());
            }
        } else
            for (int b = 0; b < nblk; b++)
                for (int cb = 0; cb < ncb; cb++)
                    fnode_upd(S, k, b * FB_RB, (b + 1) * FB_RB < nr ? (b + 1) * FB_RB : nr, cb * cbw, (cb + 1) * cbw < w ? (cb + 1) * cbw : w, fscr_get());
    }
    const double tp0 = g_ftime > 0 ? omp_get_wtime() : 0;
    if (fnode_factor(S, k, fscr_get(), par)) S->fail = 1;
    if (g_ftime > 0) { const double tp1 = omp_get_wtime();
        #pragma omp atomic
        g_ft[2] += tp1 - tp0; }
}
/* task driver: a finished node releases its parent once all children are done */
static int *g_fcnt = NULL;
static void fnode_chain(SChol *S, int k, double shift, const char *big) {
    for (;;) {
        fnode(S, k, shift, big[k]);
        const int p = S->spar[k];
        if (p < 0) return;
        int left;
        #pragma omp flush
        #pragma omp atomic capture
        left = --g_fcnt[p];
        if (left != 0) return;
        #pragma omp flush
        k = p;                                   /* the last child continues with the parent */
    }
}
static void fsubtree(SChol *S, int u, double shift) {
    for (int k = S->sfd[u]; k <= u; k++) fnode(S, k, shift, 0);
}
static int g_solve_seq;       /* (defined with the solve routines) */
static int schol_factor_(SChol *S, double shift);
int schol_factor(SChol *S, double shift) {
    if (g_ftime < 0) g_ftime = getenv("BRISK_FTIME") ? atoi(getenv("BRISK_FTIME")) : 0;
    const double t0 = g_ftime > 0 ? omp_get_wtime() : 0;
    const int r = schol_factor_(S, shift);
    if (g_ftime > 0) g_ft[3] += omp_get_wtime() - t0;
    return r;
}
static int schol_factor_(SChol *S, double shift) {
    S->fgen++;
    if (g_fpar < 0) {
        g_fpar = getenv("BRISK_FACTLL") == NULL; const char *e = getenv("BRISK_FCUT"); g_fcut = e ? atof(e) : 16.0;
        if ((e = getenv("BRISK_FRB"))) FB_RB = atoi(e);
        if ((e = getenv("BRISK_FNB"))) FB_NB = atoi(e);
    }
    /* 5.7: also for a caller that works sequentially (the LP method: schol_set_solve_seq): at one thread
     * the plain left-looking code is 25-30 % faster than the tree-parallel one on its patterns */
    if (!g_fpar || g_solve_seq) return schol_factor_ll(S, shift);
    if (g_fstat < 0) g_fstat = getenv("BRISK_FSTAT") != NULL;
    const int ns = S->ns;
    S->mval_ok = 0;
    sc_ensure(S);
    if (S->inplace && S->tinypiv <= 0) {
        if (S->pan && S->pan != S->pmat) free(S->pan);
        S->pan = S->pmat;
    } else {
        const double tc0 = g_ftime > 0 ? omp_get_wtime() : 0;
        sc_drop_moff(S);
        if (!S->pan || S->pan == S->pmat) S->pan = malloc(sizeof(double) * (S->pansz ? S->pansz : 1));
        if (!S->pan) { fprintf(stderr, "brisk: out of memory (the copy of a sparse factor, %.1f GB)\n", 8.0 * (double)S->pansz / 1e9); exit(1); }
        {   /* 17 October 2026: the copy of the values by chunks on the threads (the first one also faults the new
             * storage in on all of them: broyden25, 219 MB, 1.3-1.9 s on one thread on a host with dear page faults) */
            const size_t n = S->pansz, ch = (size_t)1 << 18, nch = (n + ch - 1) / ch;
            double *dst = S->pan; const double *src = S->pmat;
            #pragma omp parallel for schedule(static) if (n >= ((size_t)1 << 21) && !omp_in_parallel())
            for (size_t h = 0; h < nch; h++) { const size_t a = h * ch, l = a + ch < n ? ch : n - a; memcpy(dst + a, src + a, sizeof(double) * l); }
        }
        if (g_ftime > 0) { g_ft[5] += omp_get_wtime() - tc0; g_pansz_dbg = (double)S->pansz; if (g_ftime > 1) printf("   [copy %p <- %p: %.3f s]\n", (void *)S->pan, (void *)S->pmat, omp_get_wtime() - tc0); }
    }
    if (!S->ul_ptr) sc_build_tree(S);
    S->fail = 0;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (omp_in_parallel()) nt = 1;
#endif
    fscr_ensure(nt > g_panel_nt ? nt : g_panel_nt, S->m);
    const double tot = ns > 0 ? S->ssub[ns - 1] + 0.0 : 0.0;
    double totw = 0; for (int k = 0; k < ns; k++) if (S->spar[k] < 0) totw += S->ssub[k];
    (void)tot;
    if (g_ftime < 0) g_ftime = getenv("BRISK_FTIME") ? atoi(getenv("BRISK_FTIME")) : 0;
    if (g_ftime > 0) { static int once = 0; if (!once++) { double mx = 0; for (int k = 0; k < ns; k++) if (S->swk[k] > mx) mx = S->swk[k]; printf("   [sparse factor: nt %d, ns %d, totw %.3g, largest node work %.3g]\n", nt, ns, totw, mx); } }
    if (nt <= 1 || totw < 2e7) {
        for (int k = 0; k < ns && !S->fail; k++) fnode(S, k, shift, 0);
        return S->fail;
    }
    int blas_prev = 0;
    if (BL(openblas_get_num_threads) && BL(openblas_set_num_threads)) { blas_prev = BL(openblas_get_num_threads)(); if (blas_prev > 1) BL(openblas_set_num_threads)(1); }
    /* units: maximal subtrees below the cut (work < totw / (fcut * nt)), run whole; nodes
     * above it run one by one, large ones with task-parallel row blocks and tiles */
    const double cut = totw / (g_fcut * nt);
    char *big = sx(ns + 1);
    g_fcnt = sx(sizeof(int) * (ns + 1));
    for (int k = 0; k < ns; k++) big[k] = S->ssub[k] >= cut;
    for (int k = 0; k < ns; k++) if (S->spar[k] >= 0 && big[S->spar[k]]) g_fcnt[S->spar[k]]++;
    /* a node is "big" for intra-node tasks only if its own work is worth splitting */
    char *nodepar = sx(ns + 1);
    for (int k = 0; k < ns; k++) nodepar[k] = big[k] && S->swk[k] >= 4e6;
    int nunit = 0;
    int *units = sx(sizeof(int) * (ns + 1));
    for (int k = 0; k < ns; k++) if (!big[k] && (S->spar[k] < 0 || big[S->spar[k]])) units[nunit++] = k;
    /* largest units first */
    for (int i = 1; i < nunit; i++) { int u = units[i], q = i; while (q > 0 && S->ssub[units[q - 1]] < S->ssub[u]) { units[q] = units[q - 1]; q--; } units[q] = u; }
    /* big leaves (no children among the big nodes and no units below) start directly */
    const double tpr0 = g_ftime > 0 ? omp_get_wtime() : 0;
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < nunit; i++) {
            int u = units[i];
            #pragma omp task firstprivate(u)
            {
                fsubtree(S, u, shift);
                const int p = S->spar[u];
                if (p >= 0) {
                    int left;
                    #pragma omp flush
                    #pragma omp atomic capture
                    left = --g_fcnt[p];
                    #pragma omp flush
                    if (left == 0) fnode_chain(S, p, shift, nodepar);
                }
            }
        }
        for (int k = 0; k < ns; k++)
            if (big[k] && S->sfd[k] == k) {             /* a big leaf (no descendants at all) */
                int kk = k;
                #pragma omp task firstprivate(kk)
                fnode_chain(S, kk, shift, nodepar);
            }
        /* big nodes whose children are all big and have no units: covered by the chain */
    }
    if (g_ftime > 0) g_ft[4] += omp_get_wtime() - tpr0;
    free(big); free(nodepar); free(units); free(g_fcnt); g_fcnt = NULL;
    if (blas_prev > 1) BL(openblas_set_num_threads)(blas_prev);
    return S->fail;
}

/* ------------------------------------------------------------------------------------
 * 4.24: tree-parallel triangular solves, bitwise independent of the thread count.
 * Forward (multifrontal form): a node subtracts its children's contribution vectors from
 * its columns and adds the rest to its own, children in increasing order, solves its
 * diagonal block, and adds L21 x to its contribution vector (read by its parent only), so
 * concurrent subtrees never write the same entry. Backward is the gather form (a node
 * reads its ancestors' final values). Right-hand sides are held transposed (k x m).
 * ------------------------------------------------------------------------------------ */
static int g_tps = -1;
typedef struct { double *pool; size_t cap, top; } SPool;
static SPool *g_spool = NULL; static int g_nspool = 0;
static double **g_ext = NULL; static int g_next = 0; static char *g_extp = NULL;   /* g_extp[k]: vector on a pool */
static int *g_scnt = NULL;
static void sc_build_children(SChol *S) {
    const int ns = S->ns;
    int *cnt = sx(sizeof(int) * (ns + 2));
    for (int k = 0; k < ns; k++) if (S->spar[k] >= 0) cnt[S->spar[k] + 1]++;
    S->sch_ptr = sx(sizeof(int) * (ns + 1));
    for (int k = 0; k < ns; k++) S->sch_ptr[k + 1] = S->sch_ptr[k] + cnt[k + 1];
    S->sch_idx = sx(sizeof(int) * (S->sch_ptr[ns] + 1));
    int *f = sx(sizeof(int) * (ns + 1)); memcpy(f, S->sch_ptr, sizeof(int) * ns);
    for (int k = 0; k < ns; k++) if (S->spar[k] >= 0) S->sch_idx[f[S->spar[k]]++] = k;   /* ascending */
    free(f); free(cnt);
}
static double *spool_alloc(SPool *p, size_t n) {
    if (p->top + n > p->cap) return NULL;
    double *r = p->pool + p->top; p->top += n; return r;
}
/* e[q] += sum_{c0 <= c < c1} L[q + c ld] x[c] for q < n, the terms in ascending c as by one column at a time, by four
 * columns per pass over e (the same operations on each entry, so the same bits as the column loop; e is read and
 * written a quarter as often) */
static inline void tps_lmv(double *restrict e, const double *L, size_t ld, const double *x, int c0, int c1, int n) {
    int c = c0;
    for (; c + 4 <= c1; c += 4) {
        const double *restrict l0 = L + (size_t)c * ld, *restrict l1 = l0 + ld, *restrict l2 = l1 + ld, *restrict l3 = l2 + ld;
        const double v0 = x[c], v1 = x[c + 1], v2 = x[c + 2], v3 = x[c + 3];
        for (int q = 0; q < n; q++) { double s = e[q]; s += l0[q] * v0; s += l1[q] * v1; s += l2[q] * v2; s += l3[q] * v3; e[q] = s; }
    }
    for (; c < c1; c++) { const double *restrict l0 = L + (size_t)c * ld; const double v0 = x[c]; for (int q = 0; q < n; q++) e[q] += l0[q] * v0; }
}
/* v[c - c0] = x[c] - sum_q L[q + c ld] t[q] for c0 <= c < c1 <= c0 + 4, each a chain over ascending q as by one column
 * at a time, the four chains in one pass (the same bits; a single chain waits on each subtraction) */
static inline void tps_ldots(double *v, const double *L, size_t ld, const double *x, const double *t, int c0, int c1, int n) {
    if (c1 - c0 == 4) {
        const double *restrict l0 = L + (size_t)c0 * ld, *restrict l1 = l0 + ld, *restrict l2 = l1 + ld, *restrict l3 = l2 + ld;
        double v0 = x[c0], v1 = x[c0 + 1], v2 = x[c0 + 2], v3 = x[c0 + 3];
        for (int q = 0; q < n; q++) { const double tq = t[q]; v0 -= l0[q] * tq; v1 -= l1[q] * tq; v2 -= l2[q] * tq; v3 -= l3[q] * tq; }
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
        return;
    }
    for (int c = c0; c < c1; c++) { const double *restrict l0 = L + (size_t)c * ld; double v0 = x[c]; for (int q = 0; q < n; q++) v0 -= l0[q] * t[q]; v[c - c0] = v0; }
}
/* a[c - c0] = sum_q L[q + c ld] t[q] for c0 <= c < c1 <= c0 + 4 and n < 32: tps_dot's short loop (a chain over
 * ascending q from 0) for the columns in one pass */
static inline void tps_dots(double *a, const double *L, size_t ld, const double *t, int c0, int c1, int n) {
    if (c1 - c0 == 4) {
        const double *restrict l0 = L + (size_t)c0 * ld, *restrict l1 = l0 + ld, *restrict l2 = l1 + ld, *restrict l3 = l2 + ld;
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int q = 0; q < n; q++) { const double tq = t[q]; s0 += l0[q] * tq; s1 += l1[q] * tq; s2 += l2[q] * tq; s3 += l3[q] * tq; }
        a[0] = s0; a[1] = s1; a[2] = s2; a[3] = s3;
        return;
    }
    for (int c = c0; c < c1; c++) { const double *restrict l0 = L + (size_t)c * ld; double s0 = 0; for (int q = 0; q < n; q++) s0 += l0[q] * t[q]; a[c - c0] = s0; }
}
/* forward step of node k on xt (k rhs, transposed); ext[k] receives its contribution */
/* the dot of a column's off-diagonal part with t in the backward solve (one function for the sequential and the
 * threaded node, so that both form it alike) */
static __attribute__((noinline)) double tps_dot(const double *a, const double *b, int n) {
    double s = 0;
    if (n >= 32) {
        #pragma omp simd reduction(+:s)
        for (int q = 0; q < n; q++) s += a[q] * b[q];
    } else for (int q = 0; q < n; q++) s += a[q] * b[q];
    return s;
}
/* 17 October 2026: the backward triangle of a wide supernode (w >= TPS_TB2) by fixed blocks of TPS_TB columns from the
 * right: for a block [c0, c1) the tails sum_{i >= c1} L_ic x_i of its columns first (independent: on the threads when
 * par), then the block's columns in turn, x_c = (x_c - a1_c - (tail_c + sum_{c < i < c1} L_ic x_i)) / L_cc. The
 * blocks are fixed, so is the arithmetic, at any thread count and in both branches of the solve (the
 * tree-parallel one calls it with par = 0). a1: the off-diagonal dots L21' t. */
#define TPS_TB 128
#define TPS_TB2 256
static int g_solvepar = -1;
static inline int tps_tri_node(int w) {
    if (g_solvepar < 0) { const char *e = getenv("BRISK_SOLVEPAR"); g_solvepar = e ? atoi(e) : 1; }
    return g_solvepar && w >= TPS_TB2;
}
static void tps_bwd_tri(const double *P, int nr, int w, double *xs, const double *a1, int par) {
    double tl[TPS_TB];
    for (int c1 = w; c1 > 0; c1 -= TPS_TB) {
        const int c0 = c1 - TPS_TB > 0 ? c1 - TPS_TB : 0;
        #pragma omp parallel for schedule(static) if (par)
        for (int c = c0; c < c1; c++) tl[c - c0] = tps_dot(P + (size_t)c * nr + c1, xs + c1, w - c1);
        for (int c = c1 - 1; c >= c0; c--) {
            const double *col = P + (size_t)c * nr;
            const double a2 = tl[c - c0] + tps_dot(col + c + 1, xs + c + 1, c1 - c - 1);
            xs[c] = (xs[c] - a1[c] - a2) / col[c];
        }
    }
}
static inline __attribute__((always_inline)) void tps_fwd_node_i(const SChol *S, int k, double *xt, const int nr_, SPool *pool, int *relind) {
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    const int *rw = S->prow + S->prp[k];
    const double *P = S->pan + S->pbase[k];
    double *xs = xt + (size_t)j0 * nr_;
    double *e = NULL;
    if (rest > 0) {
        const size_t need = (size_t)rest * nr_;
        e = pool ? spool_alloc(pool, need) : NULL;
        g_extp[k] = e != NULL;
        if (!e) e = malloc(sizeof(double) * need);
        memset(e, 0, sizeof(double) * need);
        for (int q = w; q < nr; q++) relind[rw[q]] = q - w;
    }
    for (int u = S->sch_ptr[k]; u < S->sch_ptr[k + 1]; u++) {
        const int c = S->sch_idx[u];
        const int wc = S->ss[c + 1] - S->ss[c], rc = S->pnr[c] - wc;
        const int *rwc = S->prow + S->prp[c] + wc;
        const double *ec = g_ext[c];
        for (int q = 0; q < rc; q++) {
            const int R = rwc[q];
            const double *eq = ec + (size_t)q * nr_;
            if (R < j0 + w) { double *d = xs + (size_t)(R - j0) * nr_; for (int r = 0; r < nr_; r++) d[r] -= eq[r]; }
            else { double *d = e + (size_t)relind[R] * nr_; for (int r = 0; r < nr_; r++) d[r] += eq[r]; }
        }
        if (!g_extp[c]) free(g_ext[c]);
        g_ext[c] = NULL; g_extp[c] = 0;
    }
    for (int c = 0; c < w; c++) {
        const double *col = P + (size_t)c * nr;
        double *xc = xs + (size_t)c * nr_;
        const double id = 1.0 / col[c];
        for (int r = 0; r < nr_; r++) xc[r] *= id;
        for (int i = c + 1; i < w; i++) { double *xi = xs + (size_t)i * nr_; const double l = col[i]; for (int r = 0; r < nr_; r++) xi[r] -= l * xc[r]; }
        if (rest > 0 && nr_ > 1) {
            const double *lo = col + w;
            for (int q = 0; q < rest; q++) { double *eq = e + (size_t)q * nr_; const double l = lo[q]; for (int r = 0; r < nr_; r++) eq[r] += l * xc[r]; }
        }
    }
    if (rest > 0 && nr_ == 1) tps_lmv(e, P + w, (size_t)nr, xs, 0, w, rest);   /* (after the triangle: e and xs apart) */
    g_ext[k] = e;
}
static inline __attribute__((always_inline)) void tps_bwd_node_i(const SChol *S, int k, double *xt, const int nr_, double *t) {
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    const int *rw = S->prow + S->prp[k];
    const double *P = S->pan + S->pbase[k];
    double *xs = xt + (size_t)j0 * nr_;
    if (nr_ == 1) for (int q = 0; q < rest; q++) t[q] = xt[rw[w + q]];
    else for (int q = 0; q < rest; q++) { const double *src = xt + (size_t)rw[w + q] * nr_; double *dst = t + (size_t)q * nr_; for (int r = 0; r < nr_; r++) dst[r] = src[r]; }
    if (nr_ == 1 && tps_tri_node(w)) {                  /* (the arithmetic of tps_bwd_node_par, on this thread) */
        double *a1 = malloc(sizeof(double) * (size_t)w);
        for (int c = 0; c < w; c++) a1[c] = tps_dot(P + (size_t)c * nr + w, t, rest);
        tps_bwd_tri(P, nr, w, xs, a1, 0);
        free(a1);
        return;
    }
    if (nr_ == 1 && rest < 32) {                        /* tps_dot's short loop for four columns at once: four chains */
        for (int c1 = w; c1 > 0; c1 -= 4) {
            const int c0 = c1 - 4 > 0 ? c1 - 4 : 0;
            double a[4];
            tps_dots(a, P + w, (size_t)nr, t, c0, c1, rest);
            for (int c = c1 - 1; c >= c0; c--) {
                const double *col = P + (size_t)c * nr;
                double a2 = 0;
                if (w - c > 32) {
                    #pragma omp simd reduction(+:a2)
                    for (int i = c + 1; i < w; i++) a2 += col[i] * xs[i];
                } else for (int i = c + 1; i < w; i++) a2 += col[i] * xs[i];
                xs[c] = (xs[c] - a[c - c0] - a2) / col[c];
            }
        }
        return;
    }
    for (int c = w - 1; c >= 0; c--) {
        const double *col = P + (size_t)c * nr;
        double *xc = xs + (size_t)c * nr_;
        const double *lo = col + w;
        if (nr_ == 1) {
            double a1 = tps_dot(lo, t, rest), a2 = 0;
            if (w - c > 32) {
                #pragma omp simd reduction(+:a2)
                for (int i = c + 1; i < w; i++) a2 += col[i] * xs[i];
            } else for (int i = c + 1; i < w; i++) a2 += col[i] * xs[i];
            xc[0] = (xc[0] - a1 - a2) / col[c];
        } else {
            for (int q = 0; q < rest; q++) { const double l = lo[q]; const double *tq = t + (size_t)q * nr_; for (int r = 0; r < nr_; r++) xc[r] -= l * tq[r]; }
            for (int i = c + 1; i < w; i++) { const double l = col[i]; const double *xi = xs + (size_t)i * nr_; for (int r = 0; r < nr_; r++) xc[r] -= l * xi[r]; }
            const double id = 1.0 / col[c];
            for (int r = 0; r < nr_; r++) xc[r] *= id;
        }
    }
}
/* 17 October 2026: one right-hand side through a large supernode on the threads. When the tree has few supernodes
 * (schol_solve_tp's serial branch, ns < 64: the dense cliques of SOS relaxations, broyden25's 19 supernodes and 20
 * million factor entries), one thread streamed the whole factor. The off-diagonal part of a node (rest x w, the
 * larger part) is split by rows (forward: e = L21 x1) or by columns (backward: the dots L21' t) across the
 * threads; the diagonal triangle stays sequential. Every entry is formed by the same operations in the same order
 * as in tps_fwd_node_i / tps_bwd_node_i (e[q] adds lo[q] x_c over c ascending; the dots by tps_dot), so the
 * results are the same at any thread count and the same as the sequential code's. BRISK_SOLVEPAR=0 off. */
static int tps_par_node(const SChol *S, int k) {
    const int w = S->ss[k + 1] - S->ss[k], rest = S->pnr[k] - w;
    return tps_tri_node(w) || (g_solvepar && rest >= 256 && (double)rest * w >= 65536.0);
}
static void tps_fwd_node_par(const SChol *S, int k, double *xt, int *relind) {
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    const int *rw = S->prow + S->prp[k];
    const double *P = S->pan + S->pbase[k];
    double *xs = xt + j0;
    double *e = calloc((size_t)rest, sizeof(double));
    g_extp[k] = 0;
    for (int q = w; q < nr; q++) relind[rw[q]] = q - w;
    for (int u = S->sch_ptr[k]; u < S->sch_ptr[k + 1]; u++) {
        const int c = S->sch_idx[u];
        const int wc = S->ss[c + 1] - S->ss[c], rc = S->pnr[c] - wc;
        const int *rwc = S->prow + S->prp[c] + wc;
        const double *ec = g_ext[c];
        for (int q = 0; q < rc; q++) {
            const int R = rwc[q];
            if (R < j0 + w) xs[R - j0] -= ec[q]; else e[relind[R]] += ec[q];
        }
        if (!g_extp[c]) free(g_ext[c]);
        g_ext[c] = NULL; g_extp[c] = 0;
    }
    if (tps_tri_node(w)) {
        /* 17 October 2026: the triangle too, by blocks of TPS_TB columns: a block's own triangle in turn, then every
         * row below it (the triangle's and L21's, by chunks of 256 on the threads) takes the block's columns in
         * ascending order. Each entry receives its terms in the order of tps_fwd_node_i, so the results are its. */
        for (int c0 = 0; c0 < w; c0 += TPS_TB) {
            const int c1 = c0 + TPS_TB < w ? c0 + TPS_TB : w;
            for (int c = c0; c < c1; c++) {
                const double *col = P + (size_t)c * nr;
                xs[c] *= 1.0 / col[c];
                const double v = xs[c];
                for (int i = c + 1; i < c1; i++) xs[i] -= col[i] * v;
            }
            const int nb = nr - c1, nch = (nb + 255) / 256;
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < nch; h++) {
                const int i0 = c1 + h * 256, i1 = i0 + 256 < nr ? i0 + 256 : nr;
                for (int c = c0; c < c1; c++) {
                    const double v = xs[c], *col = P + (size_t)c * nr;
                    const int ia = i0, ib = i1 < w ? i1 : w;     /* the triangle's rows */
                    for (int i = ia; i < ib; i++) xs[i] -= col[i] * v;
                }
                const int ja = i0 > w ? i0 : w;                 /* L21's rows */
                if (ja < i1) tps_lmv(e + (ja - w), P + ja, (size_t)nr, xs, c0, c1, i1 - ja);
            }
        }
        g_ext[k] = e;
        return;
    }
    for (int c = 0; c < w; c++) {                       /* the triangle, as in tps_fwd_node_i */
        const double *col = P + (size_t)c * nr;
        xs[c] *= 1.0 / col[c];
        const double v = xs[c];
        for (int i = c + 1; i < w; i++) xs[i] -= col[i] * v;
    }
    const int nch = (rest + 255) / 256;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < nch; h++) {                     /* e = L21 x1, rows by chunks; e[q] over c ascending */
        const int q0 = h * 256, q1 = q0 + 256 < rest ? q0 + 256 : rest;
        tps_lmv(e + q0, P + w + q0, (size_t)nr, xs, 0, w, q1 - q0);
    }
    g_ext[k] = e;
}
static void tps_bwd_node_par(const SChol *S, int k, double *xt, double *t) {
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    const int *rw = S->prow + S->prp[k];
    const double *P = S->pan + S->pbase[k];
    double *xs = xt + j0;
    for (int q = 0; q < rest; q++) t[q] = xt[rw[w + q]];
    double *a1 = malloc(sizeof(double) * (size_t)w);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < w; c++) a1[c] = tps_dot(P + (size_t)c * nr + w, t, rest);
    if (tps_tri_node(w)) { tps_bwd_tri(P, nr, w, xs, a1, 1); free(a1); return; }
    for (int c = w - 1; c >= 0; c--) {
        const double *col = P + (size_t)c * nr;
        double a2 = 0;
        if (w - c > 32) {
            #pragma omp simd reduction(+:a2)
            for (int i = c + 1; i < w; i++) a2 += col[i] * xs[i];
        } else for (int i = c + 1; i < w; i++) a2 += col[i] * xs[i];
        xs[c] = (xs[c] - a1[c] - a2) / col[c];
    }
    free(a1);
}
static void tps_ensure(const SChol *S, int nt, int nr_) {
    if (g_nspool < nt) {
        for (int t = 0; t < g_nspool; t++) free(g_spool[t].pool);
        free(g_spool); g_spool = sx(sizeof(SPool) * nt); g_nspool = nt;
    }
    if (g_next < S->ns) { free(g_ext); g_ext = sx(sizeof(double *) * (S->ns + 1)); free(g_scnt); g_scnt = sx(sizeof(int) * (S->ns + 1)); free(g_extp); g_extp = sx(S->ns + 1); g_next = S->ns; }
    (void)nr_;
}
/* unit: whole subtree of u, right-looking into x for rows inside the subtree (its columns
 * are the contiguous range [ss[sfd[u]], ss[u+1]) in postorder); rows outside go to the
 * unit's contribution vector, in node order */
static inline __attribute__((always_inline)) void tps_fwd_unit_i(const SChol *S, int u, double *xt, const int nr_) {
    int *relind = fscr_get()->relind;
    const int wu = S->ss[u + 1] - S->ss[u], ru = S->pnr[u] - wu, hi = S->ss[u + 1];
    const int *rwu = S->prow + S->prp[u] + wu;
    double *e = NULL;
    if (ru > 0) { e = calloc((size_t)ru * nr_, sizeof(double)); for (int q = 0; q < ru; q++) relind[rwu[q]] = q; }
    static __thread double *t = NULL; static __thread size_t tcap = 0;
    for (int k = S->sfd[u]; k <= u; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *P = S->pan + S->pbase[k];
        double *xs = xt + (size_t)j0 * nr_;
        if ((size_t)rest * nr_ > tcap) { free(t); tcap = (size_t)rest * nr_ + 64; t = malloc(sizeof(double) * tcap); }
        if (rest > 0) memset(t, 0, sizeof(double) * (size_t)rest * nr_);
        for (int c = 0; c < w; c++) {
            const double *col = P + (size_t)c * nr;
            double *xc = xs + (size_t)c * nr_;
            const double id = 1.0 / col[c];
            for (int r = 0; r < nr_; r++) xc[r] *= id;
            if (nr_ == 1) {
                const double v = xc[0];
                for (int i = c + 1; i < w; i++) xs[i] -= col[i] * v;
            } else {
                for (int i = c + 1; i < w; i++) { double *xi = xs + (size_t)i * nr_; const double l = col[i]; for (int r = 0; r < nr_; r++) xi[r] -= l * xc[r]; }
                const double *lo = col + w;
                for (int q = 0; q < rest; q++) { double *tq = t + (size_t)q * nr_; const double l = lo[q]; for (int r = 0; r < nr_; r++) tq[r] += l * xc[r]; }
            }
        }
        if (nr_ == 1 && rest > 0) tps_lmv(t, P + w, (size_t)nr, xs, 0, w, rest);
        for (int q = 0; q < rest; q++) {
            const int R = rw[w + q];
            const double *tq = t + (size_t)q * nr_;
            if (R < hi) { double *d = xt + (size_t)R * nr_; for (int r = 0; r < nr_; r++) d[r] -= tq[r]; }
            else { double *d = e + (size_t)relind[R] * nr_; for (int r = 0; r < nr_; r++) d[r] += tq[r]; }
        }
    }
    g_ext[u] = e; g_extp[u] = 0;
}
#define TPS_DISPATCH(call1, callk) do { switch (nr_) { case 1: call1(1); break; case 2: call1(2); break; case 3: call1(3); break; case 4: call1(4); break; default: callk; } } while (0)
static void tps_fwd_node(const SChol *S, int k, double *xt, int nr_, SPool *pool, int *relind) {
#define F1(K) tps_fwd_node_i(S, k, xt, K, pool, relind)
    TPS_DISPATCH(F1, tps_fwd_node_i(S, k, xt, nr_, pool, relind));
#undef F1
}
static void tps_bwd_node(const SChol *S, int k, double *xt, int nr_, double *t) {
#define F1(K) tps_bwd_node_i(S, k, xt, K, t)
    TPS_DISPATCH(F1, tps_bwd_node_i(S, k, xt, nr_, t));
#undef F1
}
static void tps_fwd_unit(const SChol *S, int u, double *xt, int nr_) {
#define F1(K) tps_fwd_unit_i(S, u, xt, K)
    TPS_DISPATCH(F1, tps_fwd_unit_i(S, u, xt, nr_));
#undef F1
}
static void tps_fwd_chain(const SChol *S, int k, double *xt, int nr_) {
    for (;;) {
        tps_fwd_node(S, k, xt, nr_, NULL, fscr_get()->relind);
        const int p = S->spar[k];
        if (p < 0) return;
        int left;
        #pragma omp flush
        #pragma omp atomic capture
        left = --g_scnt[p];
        if (left != 0) return;
        #pragma omp flush
        k = p;
    }
}
static void tps_bwd_unit(const SChol *S, int u, double *xt, int nr_, double *t) {
    for (int k = u; k >= S->sfd[u]; k--) tps_bwd_node(S, k, xt, nr_, t);
}
static void tps_bwd_task(const SChol *S, int k, double *xt, int nr_, const char *big, size_t tmax) {
    for (;;) {
        double *t = malloc(sizeof(double) * tmax);
        tps_bwd_node(S, k, xt, nr_, t);
        free(t);
        int next = -1;
        for (int u = S->sch_ptr[k]; u < S->sch_ptr[k + 1]; u++) {
            int c = S->sch_idx[u];
            if (!big[c]) {
                #pragma omp task firstprivate(c)
                { double *tt = malloc(sizeof(double) * tmax); tps_bwd_unit(S, c, xt, nr_, tt); free(tt); }
            } else if (next < 0) next = c;
            else {
                #pragma omp task firstprivate(c)
                tps_bwd_task(S, c, xt, nr_, big, tmax);
            }
        }
        if (next < 0) return;
        k = next;
    }
}
/* B: m x nr_ (column-major, original indexing), solved in place */
static void schol_solve_tp_(const SChol *S, double *B, int nr_);
static void schol_solve_tp(const SChol *S, double *B, int nr_) { double t0 = omp_get_wtime(); schol_solve_tp_(S, B, nr_); g_tmv[1] += omp_get_wtime() - t0; g_nmv[1] += nr_; }
static void schol_solve_tp_(const SChol *S, double *B, int nr_) {
    const int m = S->m, ns = S->ns;
    SChol *Sm = (SChol *)S;
    if (!Sm->sch_ptr) sc_build_children(Sm);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    fscr_ensure(nt, m);
    tps_ensure(S, nt, nr_);
    double *xt = malloc(sizeof(double) * (size_t)m * nr_);
    for (int r = 0; r < nr_; r++) for (int i = 0; i < m; i++) xt[(size_t)i * nr_ + r] = B[(size_t)S->perm[i] + (size_t)r * m];
    size_t tmax = 1;
    for (int k = 0; k < ns; k++) { const size_t rr = (size_t)(S->pnr[k] - (S->ss[k + 1] - S->ss[k])) * nr_; if (rr > tmax) tmax = rr; }
    double totw = 0; for (int k = 0; k < ns; k++) if (S->spar[k] < 0) totw += S->ssub[k];
    /* the units are fixed (not by the thread count): they define the order of additions */
    const double cut = totw / 256.0;
    char *big = sx(ns + 1);
    for (int k = 0; k < ns; k++) { big[k] = S->ssub[k] >= cut; g_scnt[k] = 0; }
    for (int k = 0; k < ns; k++) if (S->spar[k] >= 0 && big[S->spar[k]]) g_scnt[S->spar[k]]++;
    static int tdbg = -1; if (tdbg < 0) tdbg = getenv("BRISK_TPSDBG") != NULL;
    double tt0 = tdbg ? omp_get_wtime() : 0, tt1 = 0, tt2 = 0;
    if (tdbg) { int nbig = 0; for (int k = 0; k < ns; k++) nbig += big[k]; static int once = 0; if (!once++) printf("   [tps: %d big nodes of %d]\n", nbig, ns); }
    if (nt <= 1 || ns < 64) {
        const int npar = nt > 1 && nr_ == 1;          /* (one right-hand side: the nodes' off-diagonal parts on the threads) */
        for (int k = 0; k < ns; k++) {
            if (!big[k] && (S->spar[k] < 0 || big[S->spar[k]])) tps_fwd_unit(S, k, xt, nr_);
            else if (big[k]) { if (npar && tps_par_node(S, k)) tps_fwd_node_par(S, k, xt, fscr_get()->relind); else tps_fwd_node(S, k, xt, nr_, NULL, fscr_get()->relind); }
        }
        if (S->sgn) for (int i = 0; i < m; i++) if (S->sgn[i] < 0) for (int r = 0; r < nr_; r++) xt[(size_t)i * nr_ + r] = -xt[(size_t)i * nr_ + r];
        if (tdbg) tt1 = omp_get_wtime();
        double *t = malloc(sizeof(double) * tmax);
        for (int k = ns - 1; k >= 0; k--) { if (npar && tps_par_node(S, k)) tps_bwd_node_par(S, k, xt, t); else tps_bwd_node(S, k, xt, nr_, t); }
        free(t);
        if (tdbg) { tt2 = omp_get_wtime(); static int cnt = 0; if (cnt++ < 3) printf("   [tps serial: fwd %.1f ms bwd %.1f ms]\n", 1e3 * (tt1 - tt0), 1e3 * (tt2 - tt1)); }
        for (int k = 0; k < ns; k++) if (S->spar[k] < 0 && g_ext[k]) { free(g_ext[k]); g_ext[k] = NULL; }
    } else {
        #pragma omp parallel
        #pragma omp single
        {
            for (int k = 0; k < ns; k++) {
                if (!big[k] && (S->spar[k] < 0 || big[S->spar[k]])) {
                    int u = k;
                    #pragma omp task firstprivate(u)
                    {
                        tps_fwd_unit(S, u, xt, nr_);
                        const int p = S->spar[u];
                        if (p >= 0) { int left;
                            #pragma omp flush
                            #pragma omp atomic capture
                            left = --g_scnt[p];
                            #pragma omp flush
                            if (left == 0) tps_fwd_chain(S, p, xt, nr_); }
                    }
                } else if (big[k] && S->sfd[k] == k) {
                    int kk = k;
                    #pragma omp task firstprivate(kk)
                    tps_fwd_chain(S, kk, xt, nr_);
                }
            }
        }
        for (int k = 0; k < ns; k++) if (S->spar[k] < 0 && g_ext[k]) { free(g_ext[k]); g_ext[k] = NULL; }
        if (S->sgn) for (int i = 0; i < m; i++) if (S->sgn[i] < 0) for (int r = 0; r < nr_; r++) xt[(size_t)i * nr_ + r] = -xt[(size_t)i * nr_ + r];
        #pragma omp parallel
        #pragma omp single
        {
            for (int k = 0; k < ns; k++) if (S->spar[k] < 0) {
                int kk = k;
                if (big[kk]) {
                    #pragma omp task firstprivate(kk)
                    tps_bwd_task(S, kk, xt, nr_, big, tmax);
                } else {
                    #pragma omp task firstprivate(kk)
                    { double *tt = malloc(sizeof(double) * tmax); tps_bwd_unit(S, kk, xt, nr_, tt); free(tt); }
                }
            }
        }
    }
    free(big);
    for (int r = 0; r < nr_; r++) for (int i = 0; i < m; i++) B[(size_t)S->perm[i] + (size_t)r * m] = xt[(size_t)i * nr_ + r];
    free(xt);
}

static size_t g_solve_blas = 0;     /* panels above this many entries use BLAS in the solve */
/* solve M x = b in place (original indexing) */
/* 1: single right-hand sides by the sequential supernode loops (the cone solver: 1.2 to 1.5 times faster on one thread) */
static int g_solve_seq = 0;
void schol_set_solve_seq(int on) { g_solve_seq = on; }
void schol_solve(const SChol *S, double *b, double *work) {
    if (!S->pan) { static int w = 0; if (!w++) fprintf(stderr, "brisk: internal: sparse solve without a factor\n"); memset(b, 0, sizeof(double) * S->m); return; }
    if (g_tps < 0) g_tps = getenv("BRISK_SOLVEOLD") == NULL;
    if (g_tps && S->spar && !g_solve_seq) { schol_solve_tp(S, b, 1); return; }
    const int m = S->m;
    double *x = work;
    for (int i = 0; i < m; i++) x[i] = b[S->perm[i]];
    if (!g_solve_blas) { const char *e = getenv("BRISK_SOLVEBLAS"); g_solve_blas = e ? (size_t)atol(e) : 4096; }
    /* plain loops per supernode: the four BLAS calls per supernode (dtrsv, dgemv each
     * way) cost more in call overhead than in work on patterns with thousands of narrow
     * supernodes (15% of the run on the 1354-bus TSSOS model) */
    double *restrict t = S->acc;
    for (int k = 0; k < S->ns; k++) {                                /* forward */
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *restrict Pk = S->pan + S->pbase[k];
        double *restrict xs = x + j0;
        if ((size_t)w * nr > g_solve_blas) {
            BL(dtrsv_)("L", "N", "N", &w, Pk, &nr, xs, &IONE);
            if (rest > 0) {
                BL(dgemv_)("N", &rest, &w, &DONE, Pk + w, &nr, xs, &IONE, &DZERO, t, &IONE);
                for (int q = 0; q < rest; q++) x[rw[w + q]] -= t[q];
            }
            continue;
        }
        if (w == 1) {                                                /* one column: scatter directly (no pass over t) */
            const double xc = xs[0] / Pk[0];
            xs[0] = xc;
            const double *restrict lo = Pk + 1; const int *restrict rwr = rw + 1;
            for (int q = 0; q < rest; q++) x[rwr[q]] -= lo[q] * xc;
            continue;
        }
        for (int q = 0; q < rest; q++) t[q] = 0;
        for (int c = 0; c < w; c++) {
            const double *restrict col = Pk + (size_t)c * nr;
            const double xc = xs[c] / col[c];
            xs[c] = xc;
            for (int r = c + 1; r < w; r++) xs[r] -= col[r] * xc;
        }
        tps_lmv(t, Pk + w, (size_t)nr, xs, 0, w, rest);
        for (int q = 0; q < rest; q++) x[rw[w + q]] -= t[q];
    }
    if (S->sgn) for (int i = 0; i < m; i++) if (S->sgn[i] < 0) x[i] = -x[i];
    for (int k = S->ns - 1; k >= 0; k--) {                           /* backward */
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *restrict Pk = S->pan + S->pbase[k];
        double *restrict xs = x + j0;
        if (w == 1 && (size_t)nr <= g_solve_blas) {                 /* one column: the dot directly */
            double v = xs[0];
            const double *restrict lo = Pk + 1; const int *restrict rwr = rw + 1;
            for (int q = 0; q < rest; q++) v -= lo[q] * x[rwr[q]];
            xs[0] = v / Pk[0];
            continue;
        }
        for (int q = 0; q < rest; q++) t[q] = x[rw[w + q]];
        if ((size_t)w * nr > g_solve_blas) {
            double mone = -1.0;
            if (rest > 0) BL(dgemv_)("T", &rest, &w, &mone, Pk + w, &nr, t, &IONE, &DONE, xs, &IONE);
            BL(dtrsv_)("L", "T", "N", &w, Pk, &nr, xs, &IONE);
            continue;
        }
        for (int c1 = w; c1 > 0; c1 -= 4) {                        /* the dots with t by four columns: four chains */
            const int c0 = c1 - 4 > 0 ? c1 - 4 : 0;
            double v[4];
            tps_ldots(v, Pk + w, (size_t)nr, xs, t, c0, c1, rest);
            for (int c = c1 - 1; c >= c0; c--) {
                const double *restrict col = Pk + (size_t)c * nr;
                double vc = v[c - c0];
                for (int r = c + 1; r < w; r++) vc -= col[r] * xs[r];
                xs[c] = vc / col[c];
            }
        }
    }
    for (int i = 0; i < m; i++) b[S->perm[i]] = x[i];
}


/* Two right-hand sides at once (b1, b2 in original ordering, overwritten), reading L once:
 * the solves are memory-bound on L (4.22; the paired Schur solves of the embedding were
 * two full passes). work: 2 m doubles; S->acc must hold 2 * max rest (allocated here). */
void schol_solve2(const SChol *S, double *b1, double *b2, double *work) {
    if (!S->pan) { static int w = 0; if (!w++) fprintf(stderr, "brisk: internal: sparse solve without a factor\n"); memset(b1, 0, sizeof(double) * S->m); memset(b2, 0, sizeof(double) * S->m); return; }
    if (g_tps < 0) g_tps = getenv("BRISK_SOLVEOLD") == NULL;
    if (g_tps && S->spar) {
        const int m = S->m;
        double *B2 = malloc(sizeof(double) * 2 * (size_t)m);
        memcpy(B2, b1, sizeof(double) * m); memcpy(B2 + m, b2, sizeof(double) * m);
        schol_solve_tp(S, B2, 2);
        memcpy(b1, B2, sizeof(double) * m); memcpy(b2, B2 + m, sizeof(double) * m);
        free(B2); return;
    }
    const int m = S->m;
    double *x = work, *y = work + m;
    for (int i = 0; i < m; i++) { x[i] = b1[S->perm[i]]; y[i] = b2[S->perm[i]]; }
    if (!g_solve_blas) { const char *e = getenv("BRISK_SOLVEBLAS"); g_solve_blas = e ? (size_t)atol(e) : 4096; }
    SChol *Sm = (SChol *)S;
    {
        size_t need = 0;
        for (int k = 0; k < S->ns; k++) { size_t r = (size_t)(S->pnr[k] - (S->ss[k + 1] - S->ss[k])); if (r > need) need = r; }
        need = 2 * need + 2;
        if (need > Sm->t2cap) { free(Sm->t2); Sm->t2 = sx(sizeof(double) * need); Sm->t2cap = need; }
    }
    double *restrict t = Sm->t2;
    const int two = 2;
    for (int k = 0; k < S->ns; k++) {                                /* forward */
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *restrict Pk = S->pan + S->pbase[k];
        double *restrict xs = x + j0, *restrict ys = y + j0;
        if ((size_t)w * nr > g_solve_blas) {
            BL(dtrsm_)("L", "L", "N", "N", &w, &two, &DONE, Pk, &nr, xs, &m);
            if (rest > 0) {
                BL(dgemm_)("N", "N", &rest, &two, &w, &DONE, Pk + w, &nr, xs, &m, &DZERO, t, &rest);
                for (int q = 0; q < rest; q++) { x[rw[w + q]] -= t[q]; y[rw[w + q]] -= t[rest + q]; }
            }
            continue;
        }
        double *restrict t1 = t, *restrict t2 = t + rest;
        for (int q = 0; q < rest; q++) { t1[q] = 0; t2[q] = 0; }
        for (int c = 0; c < w; c++) {
            const double *restrict col = Pk + (size_t)c * nr;
            const double ic = 1.0 / col[c];
            const double xc = xs[c] * ic, yc = ys[c] * ic;
            xs[c] = xc; ys[c] = yc;
            for (int r = c + 1; r < w; r++) { xs[r] -= col[r] * xc; ys[r] -= col[r] * yc; }
            const double *restrict lo = col + w;
            for (int q = 0; q < rest; q++) { t1[q] += lo[q] * xc; t2[q] += lo[q] * yc; }
        }
        for (int q = 0; q < rest; q++) { x[rw[w + q]] -= t1[q]; y[rw[w + q]] -= t2[q]; }
    }
    if (S->sgn) for (int i = 0; i < m; i++) if (S->sgn[i] < 0) { x[i] = -x[i]; y[i] = -y[i]; }
    for (int k = S->ns - 1; k >= 0; k--) {                           /* backward */
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *restrict Pk = S->pan + S->pbase[k];
        double *restrict xs = x + j0, *restrict ys = y + j0;
        double *restrict t1 = t, *restrict t2 = t + rest;
        for (int q = 0; q < rest; q++) { t1[q] = x[rw[w + q]]; t2[q] = y[rw[w + q]]; }
        if ((size_t)w * nr > g_solve_blas) {
            double mone = -1.0;
            if (rest > 0) BL(dgemm_)("T", "N", &w, &two, &rest, &mone, Pk + w, &nr, t, &rest, &DONE, xs, &m);
            BL(dtrsm_)("L", "L", "T", "N", &w, &two, &DONE, Pk, &nr, xs, &m);
            continue;
        }
        for (int c = w - 1; c >= 0; c--) {
            const double *restrict col = Pk + (size_t)c * nr;
            double v1 = xs[c], v2 = ys[c];
            const double *restrict lo = col + w;
            for (int q = 0; q < rest; q++) { v1 -= lo[q] * t1[q]; v2 -= lo[q] * t2[q]; }
            for (int r = c + 1; r < w; r++) { v1 -= col[r] * xs[r]; v2 -= col[r] * ys[r]; }
            const double ic = 1.0 / col[c];
            xs[c] = v1 * ic; ys[c] = v2 * ic;
        }
    }
    for (int i = 0; i < m; i++) { b1[S->perm[i]] = x[i]; b2[S->perm[i]] = y[i]; }
}

/* 5.12: M Y = B for many right-hand sides held as rows: T is nrow x m (column-major, leading
 * dimension ldt) and its column i is row i of B (original indexing); on return column i of T
 * is row i of Y. The rows of T go in fixed chunks of SR_CB through a permuted copy that stays
 * in the core's cache, and every operation is an axpy over the SR_CB entries of a chunk, so
 * the result does not depend on the thread count or the chunking. The factor is read entry by
 * entry (the panels of a graph pattern are narrow: a BLAS call per supernode costs more than
 * its work). For Z^{-1} times a dense matrix on a sparse Z: 4 nnz(L) nrow flops against the
 * 2 m^2 nrow of a product with a dense Z^{-1}. */
#define SR_CB 64
/* one chunk of CB lanes (CB a constant after inlining: the lane loops stay in registers) */
#define SR_MAXT 256
static double *g_sr_buf[SR_MAXT];
static size_t g_sr_cap[SR_MAXT];
static int g_sr_wb = 0;      /* supernodes at least this wide go through BLAS (dtrsm, dgemm on the chunk) */
/* a > 0 (the inverse): entries below a are zero on entry and not wanted on exit, so the work
 * starts at a's supernode, inside it at column a; mk != NULL: the forward visits only the
 * supernodes marked stamp (the ancestors of the chunk's indices: L^{-1} e_j lives on the
 * path from j to the root) */
/* kt < kt2: the supernodes kt..kt2-1 (the top, solved in panels: sr_top) are skipped; dir: 1
 * the forward solve only, 2 the backward only, 3 both */
static inline __attribute__((always_inline)) void sr_chunk(const SChol *S, double *restrict Wc, double *restrict A, const int CB,
                                                           const int a, const int *mk, const int stamp, const int kt, const int kt2, const int dir) {
    const int k0 = a ? S->snof[a] : 0, ns = S->ns;
    const double one = 1.0, zero = 0.0, mone = -1.0;
    if (dir & 1)
    for (int k = k0; k < ns; k++) {                             /* forward: L x = b */
        if (k == kt) { k = kt2 - 1; continue; }
        if (mk && mk[k] != stamp) continue;
        const int c0 = k == k0 ? a - S->ss[k] : 0;
        const int j0 = S->ss[k] + c0, w = S->ss[k + 1] - j0, nr = S->pnr[k] - c0;
        const int *rw = S->prow + S->prp[k] + c0;
        const double *Pk = S->pan + S->pbase[k] + c0 + (size_t)c0 * S->pnr[k];
        const int ld = S->pnr[k];
        double *restrict X0 = Wc + (size_t)j0 * CB;
        if (w >= g_sr_wb) {
            /* the chunk's columns of the supernode as a CB x w matrix X (ld CB): X <- X L11^{-T}, U = X L21' */
            const int rest = nr - w;
            BL(dtrsm_)("R", "L", "T", "N", &CB, &w, &one, Pk, &ld, X0, &CB);
            if (rest > 0) {
                BL(dgemm_)("N", "T", &CB, &rest, &w, &one, X0, &CB, Pk + w, &ld, &zero, A, &CB);
                for (int q = 0; q < rest; q++) {
                    double *restrict xt = Wc + (size_t)rw[w + q] * CB;
                    const double *restrict u = A + (size_t)q * CB;
                    for (int r = 0; r < CB; r++) xt[r] -= u[r];
                }
            }
            continue;
        }
        for (int cc = 0; cc < w; cc++) {                        /* the diagonal block */
            const double *col = Pk + (size_t)cc * ld;
            double *restrict xc = X0 + (size_t)cc * CB;
            const double d = col[cc];
            for (int r = 0; r < CB; r++) xc[r] /= d;
            for (int q = cc + 1; q < w; q++) {
                double *restrict xt = X0 + (size_t)q * CB;
                const double l = col[q];
                for (int r = 0; r < CB; r++) xt[r] -= l * xc[r];
            }
        }
        if (w == 1)                                             /* (most of a sparse forest) */
            for (int q = 1; q < nr; q++) {
                double *restrict xt = Wc + (size_t)rw[q] * CB;
                const double l = Pk[q];
                for (int r = 0; r < CB; r++) xt[r] -= l * X0[r];
            }
        else
        for (int q = w; q < nr; q++) {                          /* each row below: all w columns at once */
            double *restrict xt = Wc + (size_t)rw[q] * CB;
            double s[CB];
            for (int r = 0; r < CB; r++) s[r] = xt[r];
            for (int cc = 0; cc < w; cc++) {
                const double l = Pk[q + (size_t)cc * ld];
                const double *restrict xc = X0 + (size_t)cc * CB;
                for (int r = 0; r < CB; r++) s[r] -= l * xc[r];
            }
            for (int r = 0; r < CB; r++) xt[r] = s[r];
        }
    }
    if (dir & 2)
    for (int k = ns - 1; k >= k0; k--) {                        /* backward: L' y = x */
        if (k == kt2 - 1 && kt < kt2) { k = kt; continue; }
        const int c0 = k == k0 ? a - S->ss[k] : 0;              /* columns >= c0 need only columns >= c0 */
        const int j0 = S->ss[k] + c0, w = S->ss[k + 1] - j0, nr = S->pnr[k] - c0;
        const int *rw = S->prow + S->prp[k] + c0;
        const double *Pk = S->pan + S->pbase[k] + c0 + (size_t)c0 * S->pnr[k];
        const int ld = S->pnr[k];
        double *restrict X0 = Wc + (size_t)j0 * CB;
        if (w >= g_sr_wb) {                                     /* X <- (X - G L21) L11^{-1}, G = rows below */
            const int rest = nr - w;
            if (rest > 0) {
                for (int q = 0; q < rest; q++) memcpy(A + (size_t)q * CB, Wc + (size_t)rw[w + q] * CB, sizeof(double) * CB);
                BL(dgemm_)("N", "N", &CB, &w, &rest, &mone, A, &CB, Pk + w, &ld, &one, X0, &CB);
            }
            BL(dtrsm_)("R", "L", "N", "N", &CB, &w, &one, Pk, &ld, X0, &CB);
            continue;
        }
        if (w == 1) {                                           /* (most of a sparse forest; the same operations) */
            for (int q = 1; q < nr; q++) {
                const double *restrict xt = Wc + (size_t)rw[q] * CB;
                const double l = Pk[q];
                for (int r = 0; r < CB; r++) X0[r] -= l * xt[r];
            }
            const double d = Pk[0];
            for (int r = 0; r < CB; r++) X0[r] /= d;
            continue;
        }
        for (size_t i = 0; i < (size_t)w * CB; i++) A[i] = X0[i];
        for (int q = w; q < nr; q++) {                          /* each row below read once for all w columns */
            const double *restrict xt = Wc + (size_t)rw[q] * CB;
            for (int cc = 0; cc < w; cc++) {
                const double l = Pk[q + (size_t)cc * ld];
                double *restrict ac = A + (size_t)cc * CB;
                for (int r = 0; r < CB; r++) ac[r] -= l * xt[r];
            }
        }
        for (int cc = w - 1; cc >= 0; cc--) {
            const double *col = Pk + (size_t)cc * ld;
            double *restrict ac = A + (size_t)cc * CB;
            for (int q = cc + 1; q < w; q++) {
                const double l = col[q];
                const double *restrict xq = X0 + (size_t)q * CB;
                for (int r = 0; r < CB; r++) ac[r] -= l * xq[r];
            }
            const double d = col[cc];
            double *restrict xc = X0 + (size_t)cc * CB;
            for (int r = 0; r < CB; r++) xc[r] = ac[r] / d;
        }
    }
}
static void sr_chunk64(const SChol *S, double *Wc, double *A, int kt, int kt2, int dir) { sr_chunk(S, Wc, A, 64, 0, NULL, 0, kt, kt2, dir); }
static void sr_chunk32(const SChol *S, double *Wc, double *A, int kt, int kt2, int dir) { sr_chunk(S, Wc, A, 32, 0, NULL, 0, kt, kt2, dir); }
static void sr_chunk16(const SChol *S, double *Wc, double *A, int kt, int kt2, int dir) { sr_chunk(S, Wc, A, 16, 0, NULL, 0, kt, kt2, dir); }
#define SR_CHUNK(S_, Wc_, A_, kt_, kt2_, dir_) do { if (CB == 64) sr_chunk64(S_, Wc_, A_, kt_, kt2_, dir_); else if (CB == 32) sr_chunk32(S_, Wc_, A_, kt_, kt2_, dir_); \
                                                    else sr_chunk16(S_, Wc_, A_, kt_, kt2_, dir_); } while (0)
static void sr_ichunk64(const SChol *S, double *Wc, double *A, int a, const int *mk, int st, int kt, int kt2, int dir) { sr_chunk(S, Wc, A, 64, a, mk, st, kt, kt2, dir); }
static void sr_ichunk32(const SChol *S, double *Wc, double *A, int a, const int *mk, int st, int kt, int kt2, int dir) { sr_chunk(S, Wc, A, 32, a, mk, st, kt, kt2, dir); }
/* 5.12: the top of the factor: supernodes kt..kt2-1, columns m0..m1-1, the root of the tree
 * of the widest supernode and the run of wide supernodes before it in the postorder whose
 * parents are in the run (the dense separators at the root of a nested dissection; an
 * upward-closed set). Between the forward and the backward solve of the rest, the top's own
 * forward and backward solve runs for panels of PR right-hand sides at once (sr_top_solve:
 * blocks of 128 columns, dtrsm on the diagonal block and dgemm with the rows below; L_TT
 * gathered once per factorization). OpenBLAS's dtrsm on these shapes runs at 25-38 GF/s a
 * core (1427 x 1427 triangle, 32-256 right-hand sides) against 47-68 for dgemm. */
static int g_sr_top = -1, g_sr_pr = 256;
static int sr_parent(const SChol *S, int k) {
    const int w = S->ss[k + 1] - S->ss[k];
    return S->pnr[k] > w ? S->snof[S->prow[S->prp[k] + w]] : -1;
}
static int sr_top_range(const SChol *S, int *kt2) {
    if (g_sr_top < 0) { const char *e = getenv("BRISK_SRTOP"); g_sr_top = e ? atoi(e) : 256; if ((e = getenv("BRISK_SRPR"))) g_sr_pr = atoi(e); }
    const int ns = S->ns;
    *kt2 = ns;
    if (g_sr_top <= 0 || ns < 2) return ns;
    int kw = 0;
    for (int k = 1; k < ns; k++) if (S->ss[k + 1] - S->ss[k] > S->ss[kw + 1] - S->ss[kw]) kw = k;
    int r = kw;
    for (int p; (p = sr_parent(S, r)) >= 0; ) r = p;
    int kt = r;
    while (kt > 0 && S->ss[kt] - S->ss[kt - 1] >= 64) {
        const int p = sr_parent(S, kt - 1);
        if (p < kt || p > r) break;
        kt--;
    }
    for (int k = kt; k <= r; k++) if (S->ss[k + 1] - S->ss[k] < 64) return ns;   /* (a narrow one between) */
    if (S->ss[r + 1] - S->ss[kt] < g_sr_top) return ns;
    *kt2 = r + 1;
    return kt;
}
/* S->tl = L_TT (dense lower triangle, ld mt) for the top kt..kt2-1 of the current factor */
static int sr_top_factor(SChol *S, int kt, int kt2) {
    if (S->tl && S->tl_gen == S->fgen && S->tl_kt == kt && S->tl_kt2 == kt2) return 1;
    const int m0 = S->ss[kt], mt = S->ss[kt2] - m0;
    if (!S->tl || S->tl_kt != kt || S->tl_kt2 != kt2) {
        free(S->tl); S->tl = NULL;
        if (posix_memalign((void **)&S->tl, 64, sizeof(double) * (size_t)mt * mt)) { S->tl = NULL; return 0; }
        S->tl_kt = kt; S->tl_kt2 = kt2;
    }
    double *Lt = S->tl;
    memset(Lt, 0, sizeof(double) * (size_t)mt * mt);
    for (int k = kt; k < kt2; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const int *rw = S->prow + S->prp[k];
        const double *Pk = S->pan + S->pbase[k];
        for (int c = 0; c < w; c++) {
            double *col = Lt + (size_t)(j0 - m0 + c) * mt;
            for (int q = c; q < nr; q++) col[rw[q] - m0] = Pk[q + (size_t)c * nr];
        }
    }
    S->tl_gen = S->fgen;
    return 1;
}
/* S->tinv = (L_TT L_TT')^{-1}: the top's own columns of the inverse (schol_inverse) */
static int sr_top_inverse(SChol *S, int kt, int kt2) {
    if (S->tinv && S->tinv_gen == S->fgen && S->tinv_kt == kt && S->tinv_kt2 == kt2) return 1;
    if (!sr_top_factor(S, kt, kt2)) return 0;
    const int mt = S->ss[kt2] - S->ss[kt];
    if (!S->tinv || S->tinv_kt != kt || S->tinv_kt2 != kt2) {
        free(S->tinv); S->tinv = NULL;
        if (posix_memalign((void **)&S->tinv, 64, sizeof(double) * (size_t)mt * mt)) { S->tinv = NULL; return 0; }
        S->tinv_kt = kt; S->tinv_kt2 = kt2;
    }
    memcpy(S->tinv, S->tl, sizeof(double) * (size_t)mt * mt);
    if (brisk_dpotri_full(mt, S->tinv, mt) != 0) { S->tinv_gen = -1; return 0; }
    S->tinv_gen = S->fgen;
    return 1;
}
/* the top's forward and backward solve for a panel X (pw x mt, ld pw; the right-hand sides
 * as rows): X <- X L^{-T} L^{-1}, blocked by 128 columns: a triangular solve on the diagonal
 * block and a product with the rows below it (most of the flops in dgemm; backward stable,
 * unlike a product with the explicit inverse, which cost thetaG51 two iterations at the end) */
static int g_sr_nb = 0;
static void sr_top_solve(const double *Lt, int mt, double *X, int pw) {
    if (!g_sr_nb) { const char *e = getenv("BRISK_SRNB"); g_sr_nb = e && atoi(e) > 0 ? atoi(e) : 128; }
    const int NB = g_sr_nb;
    const double one = 1.0, mone = -1.0;
    for (int j0 = 0; j0 < mt; j0 += NB) {                  /* X <- X L^{-T} */
        const int nb = j0 + NB < mt ? NB : mt - j0, rest = mt - j0 - nb;
        BL(dtrsm_)("R", "L", "T", "N", &pw, &nb, &one, Lt + j0 + (size_t)j0 * mt, &mt, X + (size_t)j0 * pw, &pw);
        if (rest > 0) BL(dgemm_)("N", "T", &pw, &rest, &nb, &mone, X + (size_t)j0 * pw, &pw, Lt + (j0 + nb) + (size_t)j0 * mt, &mt, &one, X + (size_t)(j0 + nb) * pw, &pw);
    }
    for (int j0 = (mt - 1) / NB * NB; j0 >= 0; j0 -= NB) { /* X <- X L^{-1} */
        const int nb = j0 + NB < mt ? NB : mt - j0, rest = mt - j0 - nb;
        if (rest > 0) BL(dgemm_)("N", "N", &pw, &nb, &rest, &mone, X + (size_t)(j0 + nb) * pw, &pw, Lt + (j0 + nb) + (size_t)j0 * mt, &mt, &one, X + (size_t)j0 * pw, &pw);
        BL(dtrsm_)("R", "L", "N", "N", &pw, &nb, &one, Lt + j0 + (size_t)j0 * mt, &mt, X + (size_t)j0 * pw, &pw);
    }
}
static double *g_sr_xt = NULL; static size_t g_sr_xtcap = 0;
static double g_srti[4];
void schol_solve_rows(const SChol *S, double *T, int ldt, int nrow) {
    const int m = S->m, ns = S->ns;
    if (!S->pan) { static int w = 0; if (!w++) fprintf(stderr, "brisk: internal: sparse solve without a factor\n"); return; }
    /* lanes per chunk: the chunk (m x CB doubles) within about 1 MB of cache */
    static int cbx = -1; if (cbx < 0) { const char *e = getenv("BRISK_SRCB"); cbx = e ? atoi(e) : 0; }
    const int CB = cbx == 16 || cbx == 32 || cbx == 64 ? cbx : (size_t)m * 64 * 8 <= (1u << 20) ? 64 : 32;
    const int nch = (nrow + CB - 1) / CB;
    if (!g_sr_wb) { const char *e = getenv("BRISK_SRWB"); g_sr_wb = e ? atoi(e) : 16; if (g_sr_wb < 1) g_sr_wb = 1; }
    int wmax = 1;                       /* scratch: w x CB (narrow backward), rest x CB (BLAS updates) */
    for (int k = 0; k < ns; k++) { if (S->ss[k + 1] - S->ss[k] > wmax) wmax = S->ss[k + 1] - S->ss[k]; if (S->pnr[k] > wmax) wmax = S->pnr[k]; }
    static int dbg = -1; if (dbg < 0) dbg = getenv("BRISK_SRDBG") != NULL;
    if (dbg == 1) {
        dbg = 2;
        double h[5] = { 0 }, hd = 0, tot = 0;     /* off-diagonal entries by supernode width 1, 2-3, 4-7, 8-31, 32+ */
        for (int k = 0; k < ns; k++) {
            const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
            const double off = (double)w * (nr - w), dia = 0.5 * w * (w + 1);
            h[w == 1 ? 0 : w < 4 ? 1 : w < 8 ? 2 : w < 32 ? 3 : 4] += off; hd += dia; tot += off + dia;
        }
        printf("   [solve_rows: m %d, %d supernodes, wmax %d, CB %d; entries: diagonal blocks %.0f%%, below by width 1: %.0f%% 2-3: %.0f%% 4-7: %.0f%% 8-31: %.0f%% 32+: %.0f%%]\n",
               m, ns, wmax, CB, 100 * hd / tot, 100 * h[0] / tot, 100 * h[1] / tot, 100 * h[2] / tot, 100 * h[3] / tot, 100 * h[4] / tot);
        /* the top set (supernodes of width >= W and their ancestors): columns, supernodes, entries */
        char *top = malloc(ns);
        for (int W = 32; W <= 1024; W *= 2) {
            memset(top, 0, ns);
            for (int k = 0; k < ns; k++) {
                const int w = S->ss[k + 1] - S->ss[k];
                if (w >= W) top[k] = 1;
                if (top[k] && S->pnr[k] > w) top[S->snof[S->prow[S->prp[k] + w]]] = 1;
            }
            double te = 0; int tc = 0, tk = 0;
            for (int k = 0; k < ns; k++) if (top[k]) {
                const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
                te += (double)w * (nr - w) + 0.5 * w * (w + 1); tc += w; tk++;
            }
            printf("   [top set from width %4d: %d supernodes, %d columns, %.0f%% of the entries]\n", W, tk, tc, 100 * te / tot);
        }
        free(top);
        {   /* the rest below the top of sr_top_range: supernodes, entries, rows below that are in the top */
            int kt2, kt = sr_top_range(S, &kt2), nsf = 0, nbl = 0; double ef = 0, eb = 0, et = 0;
            for (int k = 0; k < ns; k++) {
                if (k >= kt && k < kt2) continue;
                const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
                nsf++; if (w >= g_sr_wb) nbl++;
                ef += 0.5 * w * (w + 1) + (double)w * (nr - w); eb += (double)w * (nr - w);
                for (int q = w; q < nr; q++) { const int r = S->prow[S->prp[k] + q]; if (kt < ns && r >= S->ss[kt] && r < S->ss[kt2]) et += w; }
            }
            printf("   [below the top (supernodes %d..%d): %d supernodes (%d through BLAS), %.0f entries, %.0f below the diagonal blocks, %.0f of them in the top's rows]\n",
                   kt, kt2 - 1, nsf, nbl, ef, eb, et);
        }
    }
    /* called from a parallel loop over blocks: one thread per call, the caller's thread
     * number names its buffer (inside the inactive inner region the number would be 0) */
    const int outer = omp_in_parallel();
#ifdef _OPENMP
    const int otid = outer ? omp_get_thread_num() : 0;
#else
    const int otid = 0;
#endif
    /* 5.12: the top (dense separators at the root) in panels of PR right-hand sides on the
     * threads, between the forward and the backward solve of the chunks below it; the panels
     * are fixed by nrow, so the bits do not depend on the thread count */
    int kt2 = ns, kt = outer || nch < 2 ? ns : sr_top_range(S, &kt2);
    if (kt < kt2 && !sr_top_factor((SChol *)S, kt, kt2)) kt = kt2 = ns;
    const int PR = g_sr_pr > CB ? g_sr_pr / CB * CB : CB;
    const int m0 = S->ss[kt], mt = S->ss[kt2] - m0, nrp = nch * CB, npan = (nrp + PR - 1) / PR;
    double *XT = NULL, *XF = NULL;       /* the top rows by panels; the other rows by chunks (contiguous) */
    if (kt < ns) {
        const size_t need = (size_t)m * nrp;
        if (need > g_sr_xtcap) { free(g_sr_xt); g_sr_xt = NULL; g_sr_xtcap = 0; if (!posix_memalign((void **)&g_sr_xt, 64, sizeof(double) * need)) g_sr_xtcap = need; }
        if (g_sr_xt) { XT = g_sr_xt; XF = XT + (size_t)mt * nrp; } else kt = kt2 = ns;
    }
    const int *iperm = S->iperm;
    static int srt = -1; if (srt < 0) srt = getenv("BRISK_SRTIME") != NULL;
    double tq[4] = { omp_get_wtime(), 0, 0, 0 };
    #pragma omp parallel if (nch > 1 && !outer)
    {
        /* per-thread buffers kept between calls: a fresh 0.5 MB block per call and thread was
         * an mmap, page faults and an munmap with its TLB shootdown, every product */
#ifdef _OPENMP
        const int tid = outer ? otid : omp_get_thread_num();
#else
        const int tid = otid;
#endif
        double *Wc = NULL, *A = NULL;
        if (tid < SR_MAXT) {
            const size_t need = (size_t)m * CB + (size_t)wmax * CB;
            if (need > g_sr_cap[tid]) {
                free(g_sr_buf[tid]); g_sr_buf[tid] = NULL; g_sr_cap[tid] = 0;
                if (!posix_memalign((void **)&g_sr_buf[tid], 64, sizeof(double) * need)) g_sr_cap[tid] = need;
                else g_sr_buf[tid] = NULL;
            }
            if (g_sr_buf[tid]) { Wc = g_sr_buf[tid]; A = Wc + (size_t)m * CB; }
        }
        double *own = NULL;
        if (!Wc) { own = sx(sizeof(double) * ((size_t)m * CB + (size_t)wmax * CB)); Wc = own; A = own + (size_t)m * CB; }
        #pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < nch; c++) {
            const int r0 = c * CB, cb = nrow - r0 < CB ? nrow - r0 : CB;
            double ti0 = srt ? omp_get_wtime() : 0;
            for (int j = 0; j < m; j++) {                       /* (T's columns in order: its pages in order) */
                double *xi = Wc + (size_t)iperm[j] * CB;
                const double *src = T + r0 + (size_t)j * ldt;
                int r = 0;
                for (; r < cb; r++) xi[r] = src[r];
                for (; r < CB; r++) xi[r] = 0;
            }
            if (srt && omp_get_thread_num() == 0) { double t_ = omp_get_wtime(); g_srti[0] += t_ - ti0; ti0 = t_; }
            if (kt == ns) {
                SR_CHUNK(S, Wc, A, -1, -1, 3);
                for (int j = 0; j < m; j++) {
                    const double *xi = Wc + (size_t)iperm[j] * CB;
                    double *dst = T + r0 + (size_t)j * ldt;
                    for (int r = 0; r < cb; r++) dst[r] = xi[r];
                }
                continue;
            }
            /* the forward solve below the top; the top rows to their panel (lanes r0 % PR on) */
            SR_CHUNK(S, Wc, A, kt, kt2, 1);
            if (srt && omp_get_thread_num() == 0) { double t_ = omp_get_wtime(); g_srti[1] += t_ - ti0; ti0 = t_; }
            double *xf = XF + (size_t)c * (m - mt) * CB;
            memcpy(xf, Wc, sizeof(double) * (size_t)m0 * CB);
            memcpy(xf + (size_t)m0 * CB, Wc + (size_t)(m0 + mt) * CB, sizeof(double) * (size_t)(m - m0 - mt) * CB);
            const int pp = r0 / PR, pw = (pp + 1) * PR <= nrp ? PR : nrp - pp * PR, off = r0 - pp * PR;
            double *Xp = XT + (size_t)pp * PR * mt;
            for (int i = m0; i < m0 + mt; i++) memcpy(Xp + (size_t)(i - m0) * pw + off, Wc + (size_t)i * CB, sizeof(double) * CB);
            if (srt && omp_get_thread_num() == 0) { double t_ = omp_get_wtime(); g_srti[2] += t_ - ti0; g_srti[3] += 1; }
        }
        if (kt < ns) {
            #pragma omp master
            tq[1] = omp_get_wtime();
            #pragma omp for schedule(dynamic, 1)
            for (int pp = 0; pp < npan; pp++) {
                const int pw = (pp + 1) * PR <= nrp ? PR : nrp - pp * PR;
                sr_top_solve(S->tl, mt, XT + (size_t)pp * PR * mt, pw);
            }
            #pragma omp master
            tq[2] = omp_get_wtime();
            #pragma omp for schedule(dynamic, 1)
            for (int c = 0; c < nch; c++) {                     /* the backward solve below the top */
                const int r0 = c * CB, cb = nrow - r0 < CB ? nrow - r0 : CB;
                const double *xf = XF + (size_t)c * (m - mt) * CB;
                memcpy(Wc, xf, sizeof(double) * (size_t)m0 * CB);
                memcpy(Wc + (size_t)(m0 + mt) * CB, xf + (size_t)m0 * CB, sizeof(double) * (size_t)(m - m0 - mt) * CB);
                const int pp = r0 / PR, pw = (pp + 1) * PR <= nrp ? PR : nrp - pp * PR, off = r0 - pp * PR;
                const double *Xp = XT + (size_t)pp * PR * mt;
                for (int i = m0; i < m0 + mt; i++) memcpy(Wc + (size_t)i * CB, Xp + (size_t)(i - m0) * pw + off, sizeof(double) * CB);
                SR_CHUNK(S, Wc, A, kt, kt2, 2);
                for (int j = 0; j < m; j++) {
                    const double *xi = Wc + (size_t)iperm[j] * CB;
                    double *dst = T + r0 + (size_t)j * ldt;
                    for (int r = 0; r < cb; r++) dst[r] = xi[r];
                }
            }
        }
        free(own);
    }
    if (srt && kt < ns) { fprintf(stderr, "   [solve_rows top %d of %d: forward %.1f ms, top %.1f ms, backward %.1f ms; thread 0 forward: gather %.1f, solve %.1f, store %.1f ms, %.0f chunks]\n", mt, m, 1e3 * (tq[1] - tq[0]), 1e3 * (tq[2] - tq[1]), 1e3 * (omp_get_wtime() - tq[2]), 1e3 * g_srti[0], 1e3 * g_srti[1], 1e3 * g_srti[2], g_srti[3]); memset(g_srti, 0, sizeof g_srti); }
}

/* M^{-1} (full, symmetric, original indexing) into Zi (ld ldz). The columns of the identity go
 * in chunks of consecutive permuted indices [a, a + CB): the forward solve of e_j touches only
 * the ancestors of j (L^{-1} e_j lives on the path from j to the root), and of the result only
 * the entries at permuted indices >= j are kept (the others are the mirror of an entry found
 * from a smaller index), so the backward stops at a's supernode. For one dense supernode this
 * is (1/3 + 1/3) n^3 flops against 2 n^3 for the solve with the identity. The upper half is
 * then filled by a tiled transpose. The chunks are fixed, so the bits do not depend on the
 * thread count. */
static int parent_sn(const SChol *S, int k) {
    const int w = S->ss[k + 1] - S->ss[k];
    return S->pnr[k] > w ? S->snof[S->prow[S->prp[k] + w]] : -1;
}
void schol_inverse(const SChol *S, double *Zi, int ldz) {
    const int m = S->m, ns = S->ns;
    if (!S->pan) { static int w = 0; if (!w++) fprintf(stderr, "brisk: internal: sparse inverse without a factor\n"); return; }
    const int CB = (size_t)m * 64 * 8 <= (1u << 20) ? 64 : 32;
    if (!g_sr_wb) { const char *e = getenv("BRISK_SRWB"); g_sr_wb = e ? atoi(e) : 16; if (g_sr_wb < 1) g_sr_wb = 1; }
    int wmax = 1;
    for (int k = 0; k < ns; k++) { if (S->ss[k + 1] - S->ss[k] > wmax) wmax = S->ss[k + 1] - S->ss[k]; if (S->pnr[k] > wmax) wmax = S->pnr[k]; }
    int *par = sx(sizeof(int) * (size_t)ns);
    for (int k = 0; k < ns; k++) par[k] = parent_sn(S, k);
    const int *perm = S->perm, *iperm = S->iperm;
    const int outer = omp_in_parallel();
#ifdef _OPENMP
    const int otid = outer ? omp_get_thread_num() : 0;
#else
    const int otid = 0;
#endif
    /* the top (sr_top_range): its own columns are the inverse of its Schur complement (S->tinv);
     * the columns below it take the top's part of the solve as a product with S->tinv in
     * panels of PR (as schol_solve_rows); the chunks: [0, m0) and [m1, m) in steps of CB */
    int kt2 = ns, kt = outer ? ns : sr_top_range(S, &kt2);
    if (kt < kt2 && !sr_top_inverse((SChol *)S, kt, kt2)) kt = kt2 = ns;
    if (kt < ns) {
        const size_t need = (size_t)m * (((size_t)S->ss[kt] + CB - 1) / CB * CB);
        if (need > g_sr_xtcap) { free(g_sr_xt); g_sr_xt = NULL; g_sr_xtcap = 0; if (!posix_memalign((void **)&g_sr_xt, 64, sizeof(double) * need)) g_sr_xtcap = need; }
        if (!g_sr_xt) kt = kt2 = ns;
    }
    const int top = kt < ns, m0 = S->ss[kt], m1 = S->ss[kt2], mt = m1 - m0;
    const int nf = (m0 + CB - 1) / CB, nch = nf + (m - m1 + CB - 1) / CB;
    const int PR = g_sr_pr > CB ? g_sr_pr / CB * CB : CB, nrp = nf * CB, npan = (nrp + PR - 1) / PR;
    double *XT = top ? g_sr_xt : NULL, *XF = top ? g_sr_xt + (size_t)mt * nrp : NULL;
    #pragma omp parallel if (nch > 1 && !outer)
    {
#ifdef _OPENMP
        const int tid = outer ? otid : omp_get_thread_num();
#else
        const int tid = otid;
#endif
        double *Wc = NULL, *A = NULL;
        if (tid < SR_MAXT) {
            const size_t need = (size_t)m * CB + (size_t)wmax * CB;
            if (need > g_sr_cap[tid]) {
                free(g_sr_buf[tid]); g_sr_buf[tid] = NULL; g_sr_cap[tid] = 0;
                if (!posix_memalign((void **)&g_sr_buf[tid], 64, sizeof(double) * need)) g_sr_cap[tid] = need;
                else g_sr_buf[tid] = NULL;
            }
            if (g_sr_buf[tid]) { Wc = g_sr_buf[tid]; A = Wc + (size_t)m * CB; }
        }
        double *own = NULL;
        if (!Wc) { own = sx(sizeof(double) * ((size_t)m * CB + (size_t)wmax * CB)); Wc = own; A = own + (size_t)m * CB; }
        int *mk = sx(sizeof(int) * (size_t)ns);
        for (int k = 0; k < ns; k++) mk[k] = -1;
        #pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < nch; c++) {
            const int a = c < nf ? c * CB : m1 + (c - nf) * CB, e = c < nf ? (a + CB < m0 ? a + CB : m0) : (a + CB < m ? a + CB : m);
            const int cb = e - a, i0 = S->ss[S->snof[a]];
            for (int i = a; i < e; i++)                         /* the ancestors of the chunk's indices */
                for (int k = S->snof[i]; k >= 0 && mk[k] != c; k = par[k]) mk[k] = c;
            memset(Wc + (size_t)i0 * CB, 0, sizeof(double) * (size_t)(m - i0) * CB);
            for (int r = 0; r < cb; r++) Wc[(size_t)(a + r) * CB + r] = 1.0;
            if (top && c < nf) {                                /* forward below the top; the top's rows to their panel */
                if (CB == 64) sr_ichunk64(S, Wc, A, a, mk, c, kt, kt2, 1); else sr_ichunk32(S, Wc, A, a, mk, c, kt, kt2, 1);
                double *xf = XF + (size_t)c * (m - mt) * CB;
                memcpy(xf, Wc, sizeof(double) * (size_t)m0 * CB);
                memcpy(xf + (size_t)m0 * CB, Wc + (size_t)m1 * CB, sizeof(double) * (size_t)(m - m1) * CB);
                const int r0 = c * CB, pp = r0 / PR, pw = (pp + 1) * PR <= nrp ? PR : nrp - pp * PR, off = r0 - pp * PR;
                double *Xp = XT + (size_t)pp * PR * mt;
                for (int i = m0; i < m1; i++) memcpy(Xp + (size_t)(i - m0) * pw + off, Wc + (size_t)i * CB, sizeof(double) * CB);
                continue;
            }
            if (CB == 64) sr_ichunk64(S, Wc, A, a, mk, c, -1, -1, 3); else sr_ichunk32(S, Wc, A, a, mk, c, -1, -1, 3);
            for (int r = 0; r < cb; r++) {                      /* column perm[j], rows perm[i] for i >= j */
                const int j = a + r;
                double *restrict col = Zi + (size_t)perm[j] * ldz;
                for (int i = j; i < m; i++) col[perm[i]] = Wc[(size_t)i * CB + r];
            }
        }
        if (top) {
            #pragma omp for schedule(dynamic, 1)
            for (int pp = 0; pp < npan; pp++) {
                const int pw = (pp + 1) * PR <= nrp ? PR : nrp - pp * PR;
                sr_top_solve(S->tl, mt, XT + (size_t)pp * PR * mt, pw);
            }
            #pragma omp for schedule(dynamic, 1)
            for (int c = 0; c < nf; c++) {                      /* backward below the top */
                const int a = c * CB, e = a + CB < m0 ? a + CB : m0, cb = e - a;
                const double *xf = XF + (size_t)c * (m - mt) * CB;
                memcpy(Wc, xf, sizeof(double) * (size_t)m0 * CB);
                memcpy(Wc + (size_t)m1 * CB, xf + (size_t)m0 * CB, sizeof(double) * (size_t)(m - m1) * CB);
                const int r0 = c * CB, pp = r0 / PR, pw = (pp + 1) * PR <= nrp ? PR : nrp - pp * PR, off = r0 - pp * PR;
                const double *Xp = XT + (size_t)pp * PR * mt;
                for (int i = m0; i < m1; i++) memcpy(Wc + (size_t)i * CB, Xp + (size_t)(i - m0) * pw + off, sizeof(double) * CB);
                if (CB == 64) sr_ichunk64(S, Wc, A, a, NULL, 0, kt, kt2, 2); else sr_ichunk32(S, Wc, A, a, NULL, 0, kt, kt2, 2);
                for (int r = 0; r < cb; r++) {
                    const int j = a + r;
                    double *restrict col = Zi + (size_t)perm[j] * ldz;
                    for (int i = j; i < m; i++) col[perm[i]] = Wc[(size_t)i * CB + r];
                }
            }
            #pragma omp for schedule(static)
            for (int j = m0; j < m1; j++) {                     /* the top's own columns */
                double *restrict col = Zi + (size_t)perm[j] * ldz;
                const double *tj = S->tinv + (size_t)(j - m0) * mt;
                for (int i = j; i < m1; i++) col[perm[i]] = tj[i - m0];
                for (int i = m1; i < m; i++) col[perm[i]] = 0.0;
            }
        }
        free(mk);
        free(own);
    }
    free(par);
    /* the other half: entry (r, c) is set iff iperm[r] >= iperm[c]; tiles of 64 x 64 */
    const int TB = 64, nt = (m + TB - 1) / TB;
    #pragma omp parallel for schedule(dynamic, 1) if (nt > 4 && !outer)
    for (int tj = 0; tj < nt; tj++) {
        const int c0 = tj * TB, c1 = c0 + TB < m ? c0 + TB : m;
        for (int ti = 0; ti <= tj; ti++) {
            const int r0 = ti * TB, r1 = r0 + TB < m ? r0 + TB : m;
            for (int c = c0; c < c1; c++) {
                const int pc = iperm[c], re = ti == tj ? c : r1;
                double *restrict zc = Zi + (size_t)c * ldz;
                for (int r = r0; r < re; r++) {
                    double *zr = Zi + (size_t)r * ldz + c;     /* (c, r) */
                    if (iperm[r] >= pc) *zr = zc[r]; else zc[r] = *zr;
                }
            }
        }
    }
}

/* 5.12: the halves of a solve, in the permuted ordering (x[i] belongs to original index
 * perm[i]): L x = b and L' x = b in place. With M = P' L L' P the eigenvalues of
 * L^{-1} P D P' L^{-T} are those of M^{-1} D (the dual step lengths on a sparse Z). */
void schol_lsolve(const SChol *S, double *x) {
    for (int k = 0; k < S->ns; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const int *rw = S->prow + S->prp[k];
        const double *Pk = S->pan + S->pbase[k];
        for (int cc = 0; cc < w; cc++) {
            const double *col = Pk + (size_t)cc * nr;
            const double xc = x[j0 + cc] / col[cc];
            x[j0 + cc] = xc;
            for (int q = cc + 1; q < nr; q++) x[rw[q]] -= col[q] * xc;
        }
    }
}
void schol_ltsolve(const SChol *S, double *x) {
    for (int k = S->ns - 1; k >= 0; k--) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const int *rw = S->prow + S->prp[k];
        const double *Pk = S->pan + S->pbase[k];
        for (int cc = w - 1; cc >= 0; cc--) {
            const double *col = Pk + (size_t)cc * nr;
            double v = x[j0 + cc];
            for (int q = cc + 1; q < nr; q++) v -= col[q] * x[rw[q]];
            x[j0 + cc] = v / col[cc];
        }
    }
}

/* Solve M X = B for many right-hand sides at once (B is m x nrhs, column-major, original
 * ordering). One dtrsm/dgemm per supernode instead of per column: computing Z^{-1} column
 * by column paid the per-supernode call overhead n times over. */
void schol_solve_many(const SChol *S, double *Bm, int nrhs, double *work) {
    if (!S->pan) { static int w = 0; if (!w++) fprintf(stderr, "brisk: internal: sparse solve without a factor\n"); memset(Bm, 0, sizeof(double) * S->m * (size_t)nrhs); return; }
    if (g_tps < 0) g_tps = getenv("BRISK_SOLVEOLD") == NULL;
    if (g_tps && S->spar) { schol_solve_tp(S, Bm, nrhs); return; }
    /* The right-hand sides are held transposed (work = X', nrhs x m, column i = row i of
     * X), so the updates between supernodes are contiguous vectors of length nrhs instead
     * of nrhs scattered entries: with the small supernodes of a graph pattern the
     * scattered version was memory-bound (maxG/qpG: Z^{-1} 3-5x slower than this). */
    const int m = S->m;
    {
        SChol *Sm = (SChol *)S;
        size_t need = 0;
        for (int k = 0; k < S->ns; k++) {
            size_t r = (size_t)(S->pnr[k] - (S->ss[k + 1] - S->ss[k])) * nrhs;
            if (r > need) need = r;
        }
        Sm->acc2 = realloc(Sm->acc2, sizeof(double) * (need ? need : 1));
    }
    const double DONE_ = 1.0, DZERO_ = 0.0, DMONE_ = -1.0;
    const size_t R = (size_t)nrhs;
    for (int i = 0; i < m; i++) {
        const size_t oi = (size_t)S->perm[i];
        double *xi = work + (size_t)i * R;
        for (int c = 0; c < nrhs; c++) xi[c] = Bm[oi + (size_t)c * m];
    }
    for (int k = 0; k < S->ns; k++) {                       /* forward: X' <- X' L^{-T} */
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *Pk = S->pan + S->pbase[k];
        double *Xj = work + (size_t)j0 * R;
        BL(dtrsm_)("R", "L", "T", "N", &nrhs, &w, &DONE_, Pk, &nr, Xj, &nrhs);
        if (rest > 0) {
            double *T = S->acc2;                              /* T' = X'_j L21' (nrhs x rest) */
            BL(dgemm_)("N", "T", &nrhs, &rest, &w, &DONE_, Xj, &nrhs, Pk + w, &nr, &DZERO_, T, &nrhs);
            for (int q = 0; q < rest; q++) {
                double *dst = work + (size_t)rw[w + q] * R;
                const double *src = T + (size_t)q * R;
                for (int c = 0; c < nrhs; c++) dst[c] -= src[c];
            }
        }
    }
    if (S->sgn)
        for (int i = 0; i < m; i++) if (S->sgn[i] < 0) { double *xi = work + (size_t)i * R; for (int c = 0; c < nrhs; c++) xi[c] = -xi[c]; }
    for (int k = S->ns - 1; k >= 0; k--) {                  /* backward: X' <- X' L^{-1} */
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
        const int *rw = S->prow + S->prp[k];
        const double *Pk = S->pan + S->pbase[k];
        double *Xj = work + (size_t)j0 * R;
        if (rest > 0) {
            double *T = S->acc2;
            for (int q = 0; q < rest; q++) memcpy(T + (size_t)q * R, work + (size_t)rw[w + q] * R, sizeof(double) * R);
            BL(dgemm_)("N", "N", &nrhs, &w, &rest, &DMONE_, T, &nrhs, Pk + w, &nr, &DONE_, Xj, &nrhs);
        }
        BL(dtrsm_)("R", "L", "N", "N", &nrhs, &w, &DONE_, Pk, &nr, Xj, &nrhs);
    }
    for (int i = 0; i < m; i++) {
        const size_t oi = (size_t)S->perm[i];
        const double *xi = work + (size_t)i * R;
        for (int c = 0; c < nrhs; c++) Bm[oi + (size_t)c * m] = xi[c];
    }
}


/* Panel offsets of a clique of original indices con[0..nl) (4.20): pos[a*nl+b] (both
 * triangles) and the lower-triangle entries in increasing offset order (soff, sidx =
 * a*nl+b), computed from the supernodal layout directly instead of nl^2 hash lookups and a
 * sort (schur_pos_setup: 0.15 s on attr_vdp_d10_I). Returns 0, or 1 if an entry is
 * outside the pattern of L.                                                            */
static int cmp_pi(const void *a, const void *b) { const int *x = a, *y = b; return x[0] - y[0]; }
int schol_clique_offsets(const SChol *S, int nl, const int *con, size_t *pos, size_t *soff, int *sidx) {
    int *pc = malloc(sizeof(int) * 2 * (nl + 1));
    for (int a = 0; a < nl; a++) { pc[2 * a] = S->iperm[con[a]]; pc[2 * a + 1] = a; }
    qsort(pc, nl, 2 * sizeof(int), cmp_pi);
    /* 4.23: the row map persists (all -1 between calls; only the rows of the supernodes
     * visited are set and reset): an O(m) fill per clique cost ~1 s of setup on TSSOS
     * case6468 (20,000 cliques, m = 76,000) */
    static __thread int *pm = NULL; static __thread int pm_m = 0;
    if (pm_m < S->m) {
        pm = realloc(pm, sizeof(int) * (S->m + 1));
        for (int i = pm_m; i <= S->m; i++) pm[i] = -1;
        pm_m = S->m;
    }
    int cur = -1, w = 0, rc = 0;
    for (int bb = 0; bb < nl && !rc; bb++) {
        const int j = pc[2 * bb], b = pc[2 * bb + 1];
        const int k = S->snof[j];
        if (k != cur) {
            if (cur >= 0) for (int q = 0; q < S->pnr[cur]; q++) pm[S->prow[S->prp[cur] + q]] = -1;
            for (int q = 0; q < S->pnr[k]; q++) pm[S->prow[S->prp[k] + q]] = q;
            cur = k;
        }
        const size_t base = S->pbase[k] + (size_t)(j - S->ss[k]) * S->pnr[k];
        for (int aa = bb; aa < nl; aa++) {
            const int r = pc[2 * aa], a = pc[2 * aa + 1];
            const int q = pm[r];
            if (q < 0) { rc = 1; break; }
            const size_t o = base + q;
            pos[(size_t)a * nl + b] = pos[(size_t)b * nl + a] = o;
            if (soff) { soff[w] = o; sidx[w] = a >= b ? b * nl + a : a * nl + b; w++;   /* 4.23: (lo, hi) layout: a row of pairs is contiguous */ }
        }
    }
    if (cur >= 0) for (int q = 0; q < S->pnr[cur]; q++) pm[S->prow[S->prp[cur] + q]] = -1;
    free(pc);
    return rc;
}

/* 18 October 2026: the same offsets as schol_clique_offsets in O(nl x supernodes) memory instead of nl^2 (the
 * solver's per-block squares were a third of the peak memory on broyden25): the rows of a clique in a column of
 * supernode k sit at the positions of supernode k's row list, the same for every column of k, so one row of
 * positions per supernode the clique's columns fall in, and a base per column. NULL if an entry is outside the
 * pattern of L (as schol_clique_offsets's 1). */
CliqueTab *schol_clique_table(const SChol *S, int nl, const int *con) {
    int *pc = malloc(sizeof(int) * 2 * (nl + 1));
    for (int a = 0; a < nl; a++) { pc[2 * a] = S->iperm[con[a]]; pc[2 * a + 1] = a; }
    qsort(pc, nl, 2 * sizeof(int), cmp_pi);
    int ns = 0;
    for (int bb = 0; bb < nl; bb++) if (bb == 0 || S->snof[pc[2 * bb]] != S->snof[pc[2 * bb - 2]]) ns++;
    CliqueTab *T = calloc(1, sizeof(CliqueTab));
    T->nl = nl; T->nslot = ns;
    T->e = malloc(sizeof(CliqueEnt) * (nl + 1)); T->rp = malloc(sizeof(int) * ((size_t)ns * nl + 1));
    static __thread int *pm = NULL; static __thread int pm_m = 0;   /* (as in schol_clique_offsets) */
    if (pm_m < S->m) {
        pm = realloc(pm, sizeof(int) * (S->m + 1));
        for (int i = pm_m; i <= S->m; i++) pm[i] = -1;
        pm_m = S->m;
    }
    int cur = -1, slot = -1, rc = 0;
    for (int bb = 0; bb < nl && !rc; bb++) {
        const int j = pc[2 * bb], b = pc[2 * bb + 1];
        const int k = S->snof[j];
        if (k != cur) {
            if (cur >= 0) for (int q = 0; q < S->pnr[cur]; q++) pm[S->prow[S->prp[cur] + q]] = -1;
            for (int q = 0; q < S->pnr[k]; q++) pm[S->prow[S->prp[k] + q]] = q;
            cur = k; slot++;
            int *rw = T->rp + (size_t)slot * nl;
            for (int aa = 0; aa < nl; aa++) {
                const int q = pm[pc[2 * aa]];
                rw[pc[2 * aa + 1]] = q;
                if (aa >= bb && q < 0) rc = 1;   /* a row at or below this column outside the pattern */
            }
        }
        T->e[b].pl = j; T->e[b].sl = slot;
        T->e[b].cb = S->pbase[k] + (size_t)(j - S->ss[k]) * S->pnr[k];
    }
    if (cur >= 0) for (int q = 0; q < S->pnr[cur]; q++) pm[S->prow[S->prp[cur] + q]] = -1;
    free(pc);
    if (rc) { clique_tab_free(T); return NULL; }
    return T;
}
void clique_tab_free(CliqueTab *T) { if (T) { free(T->e); free(T->rp); free(T); } }
double schol_flops(const SChol *S) { return S ? S->flops : 0; }

/* factorization cost in cost units: flops in wide supernodes (>= 16 columns) run as
 * dgemm/dpotrf panels (c_wide per flop), the rest at the scalar rate c_narrow */
double schol_cost(const SChol *S, double c_narrow, double c_wide) {
    if (!S) return 0;
    double c = 0;
    for (int k = 0; k < S->ns; k++) {
        const int j0 = S->ss[k], j1 = S->ss[k + 1];
        double f = 0;
        for (int j = j0; j < j1; j++) { double l = S->cp[j + 1] - S->cp[j]; f += l * l; }
        c += f * (j1 - j0 >= 16 ? c_wide : c_narrow);
    }
    return c;
}

/* log det of the factored matrix = 2 sum log(diag L) */
double schol_logdet(const SChol *S) {
    if (!S->pan) return NAN;
    double ld = 0;
    for (int k = 0; k < S->ns; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const double *Pk = S->pan + S->pbase[k];
        for (int cc = 0; cc < w; cc++) ld += log(Pk[(size_t)cc * nr + cc]);
    }
    return 2.0 * ld;
}
long schol_uid(const SChol *S) { return S ? S->uid : 0; }
const int *schol_perm(const SChol *S) { return S->perm; }      /* perm[new] = old */
void schol_set_tinypiv(SChol *S, double t) { S->tinypiv = t; S->ntiny = 0; }
void schol_set_tinydag(SChol *S, int on) { S->tinydag = on; }
void schol_set_dynpiv(SChol *S, double p) { S->dynpiv = p; S->ndyn = 0; }
int schol_ndyn(const SChol *S) { return S->ndyn; }
int schol_ntiny(const SChol *S) { return S->ntiny; }
size_t schol_nnz(const SChol *S) { return S ? S->nnz : 0; }
int schol_nsuper(const SChol *S) { return S ? S->ns : 0; }
/* off-diagonal entries of the matrix pattern (lower triangle), as given to the analysis */
size_t schol_mpat_nnz(const SChol *S) { return S && S->mcp ? (size_t)S->mcp[S->m] - (size_t)S->m : 0; }

/* 4.24 diagnostic: parallelism in the supernodal elimination tree (BRISK_TREEDBG) */
void schol_tree_stats(const SChol *S) {
    const int ns = S->ns;
    int *par = malloc(sizeof(int) * (ns + 1));
    double *c = calloc(ns + 1, sizeof(double)), *sub = calloc(ns + 1, sizeof(double));
    int *nch = calloc(ns + 1, sizeof(int));
    double tot = 0, cmax = 0;
    for (int k = 0; k < ns; k++) {
        const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
        par[k] = nr > w ? S->snof[S->prow[S->prp[k] + w]] : -1;
        c[k] = (double)nr * nr * w;      /* ~ its factor + outgoing update work */
        tot += c[k]; if (c[k] > cmax) cmax = c[k];
    }
    int nroot = 0;
    for (int k = 0; k < ns; k++) { sub[k] += c[k]; if (par[k] >= 0) { sub[par[k]] += sub[k]; nch[par[k]]++; } else nroot++; }
    /* critical path: longest root-leaf chain of c */
    double *cp = calloc(ns + 1, sizeof(double)), crit = 0;
    for (int k = ns - 1; k >= 0; k--) { cp[k] = c[k] + (par[k] >= 0 ? cp[par[k]] : 0); if (cp[k] > crit) crit = cp[k]; }
    {   /* postorder check: subtree of k = [k - size + 1, k] */
        int *sz = calloc(ns + 1, sizeof(int)), *fd = malloc(sizeof(int) * (ns + 1)), bad = 0;
        for (int k = 0; k < ns; k++) fd[k] = k;
        for (int k = 0; k < ns; k++) { sz[k]++; if (par[k] >= 0) { sz[par[k]] += sz[k]; if (fd[k] < fd[par[k]]) fd[par[k]] = fd[k]; } }
        for (int k = 0; k < ns; k++) if (k - fd[k] + 1 != sz[k]) bad++;
        printf("   [tree: postorder %s (%d violations)]\n", bad ? "NO" : "yes", bad);
        int idx[12]; double wv[12]; int ni = 0;
        for (int k = 0; k < ns; k++) {
            int pos = ni < 12 ? ni++ : 11;
            if (ni == 12 && pos == 11 && c[k] <= wv[11]) continue;
            idx[pos] = k; wv[pos] = c[k];
            for (int q = pos; q > 0 && wv[q] > wv[q - 1]; q--) { double t = wv[q]; wv[q] = wv[q - 1]; wv[q - 1] = t; int ti = idx[q]; idx[q] = idx[q - 1]; idx[q - 1] = ti; }
        }
        for (int q = 0; q < ni; q++) printf("     [node %d: w %d nr %d, %.1f%%, subtree %.1f%%]\n", idx[q], S->ss[idx[q] + 1] - S->ss[idx[q]], S->pnr[idx[q]], 100 * wv[q] / tot, 100 * sub[idx[q]] / tot);
        free(sz); free(fd);
    }
    printf("   [tree: %d supernodes, %d roots, work %.3g, largest node %.1f%%, critical path %.1f%% (max speedup %.1f)]\n",
           ns, nroot, tot, 100 * cmax / tot, 100 * crit / tot, tot / crit);
    /* Geist-Ng style split: repeatedly open the heaviest subtree; report the top fraction */
    for (int T = 2; T <= 64; T *= 2) {
        char *top = calloc(ns + 1, 1); int *heap = malloc(sizeof(int) * (ns + 1)); int nh = 0;
        for (int k = 0; k < ns; k++) if (par[k] < 0) heap[nh++] = k;
        int **ch = NULL; (void)ch;
        double topw = 0;
        for (int it = 0; it < 100000; it++) {
            int bi = -1; double bw = 0;
            for (int h = 0; h < nh; h++) if (sub[heap[h]] > bw) { bw = sub[heap[h]]; bi = h; }
            if (bi < 0 || bw <= tot / (4.0 * T)) break;
            const int k = heap[bi]; heap[bi] = heap[--nh]; top[k] = 1; topw += c[k];
            for (int j = 0; j < ns; j++) if (par[j] == k) heap[nh++] = j;   /* O(ns) per step: diagnostic only */
        }
        double big = 0; for (int h = 0; h < nh; h++) if (sub[heap[h]] > big) big = sub[heap[h]];
        /* LPT bound for the subtree phase */
        double sw = tot - topw, ph = fmax(sw / T, big);
        printf("   [tree T=%d: %d subtrees, top %.1f%% of the work, subtree phase >= %.1f%% -> speedup <= %.2f (top serial) / %.2f (top at T x BLAS)]\n",
               T, nh, 100 * topw / tot, 100 * ph / tot, tot / (ph + topw), tot / (ph + topw / T));
        free(top); free(heap);
    }
    free(par); free(c); free(sub); free(nch); free(cp);
}

/* 20 October 2026: the pivots of the last factorization (the squares of the diagonal of L), by original index; a
 * pivot replaced by the tiny-pivot rule reads 1e128 */
void schol_pivots(const SChol *S, double *d) {
    for (int k = 0; k < S->ns; k++) {
        const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k];
        const double *Pk = S->pan + S->pbase[k];
        for (int cc = 0; cc < w; cc++) { const double l = Pk[(size_t)cc * nr + cc]; d[S->perm[j0 + cc]] = l * l; }
    }
}
/* 20 October 2026: schol_solve_many by the transposed right-hand sides (the path BRISK_SOLVEOLD selects), whatever the
 * default: many right-hand sides on a factor with a few large supernodes (the presolve's dependent constraints: 276 on
 * a dense factor of order 2,304, 1.0 -> 0.2 s). Single-threaded callers only (it switches the path for the call). */
void schol_solve_many_t(const SChol *S, double *Bm, int nrhs, double *work) {
    const int o = g_tps; g_tps = 0;
    schol_solve_many(S, Bm, nrhs, work);
    g_tps = o;
}
