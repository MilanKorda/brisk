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
    int *ci;                /* row indices of L, sorted, diagonal first */

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
    double dynpiv;                   /* > 0 (signed factorization): a pivot of the wrong sign is replaced by
                                      * sign * dynpiv and counted, instead of failing */
    int ndyn;
    double *mval;                    /* compact copy of the M values (valid if mval_ok) */
    int mval_ok;
    int *mrun, nrun;                 /* runs of consecutive rows in the off-diagonal pattern: (q0, i0, len) triples (4.20) */
    int *mrp;                        /* per column: first run */
    int run_state;                   /* 0 not analysed, 1 use runs, -1 entry loop */
    double *t2; size_t t2cap;        /* schol_solve2 workspace */
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
static int cmp_i(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

/* optional cap on the factorization flops (sum of squared column counts): orderings stop
 * as soon as the factor cannot beat the dense path, instead of finishing the analysis
 * only to be rejected (e_moment_stable_17: 0.8 s per analysis) */
static double g_flopcap = 0;
void schol_set_flopcap(double cap) { g_flopcap = cap; }
/* the ordering mode of the next analyses (BRISK_AMD: 0 own, 1 automatic, 2 AMD alone); returns the previous one (-1: not yet set) */
static int g_amd;
int schol_set_amd(int mode) { const int old = g_amd; g_amd = mode; return old; }
/* AMD's dense-row parameter for the next analyses (0: its default, 10): rows with more than this times sqrt(n) entries are ordered last */
static double g_amd_dense = 0;
/* an ordering given by the caller for the next analysis in AMD mode (perm[new] = old; NULL: AMD) */
static const int *g_force_perm = NULL;
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
            if (total > fillcap || (g_flopcap > 0 && flops > g_flopcap)) { rc = 1; break; }
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
                if (total + tot2 > fillcap || (g_flopcap > 0 && flops + fl2 > g_flopcap)) {
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
        if (total > fillcap || work > workcap || (g_flopcap > 0 && flops > g_flopcap)) {
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
void schol_set_inplace(SChol *S, int on) { S->inplace = on; }
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
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap) {
    SAdj *adj = sx(sizeof(SAdj) * m);
    for (int i = 0; i < m; i++) {
        adj[i].n = adj[i].cap = deg[i];
        adj[i].a = malloc(sizeof(int) * (deg[i] ? deg[i] : 1));
        memcpy(adj[i].a, nbr[i], sizeof(int) * deg[i]);
    }
    SChol *S = schol_analyze(m, adj, fillcap);
    if (S) {
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
        for (int j = 0; j < m; j++) {
            for (int q = S->mcp[j]; q < S->mcp[j + 1]; q++) {
                size_t off = schol_offset(S, (size_t)S->perm[S->mri[q]], (size_t)S->perm[j]);
                S->moff[q] = off;
            }
        }
    }
    for (int i = 0; i < m; i++) free(adj[i].a);
    free(adj);
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
    int *ip = malloc(sizeof(int) * m), *mark = malloc(sizeof(int) * m), *head = malloc(sizeof(int) * (m + 1)), *nxt = malloc(sizeof(int) * (m + 1));
    int *plen = malloc(sizeof(int) * m), **pat = calloc(m, sizeof(int *)), *buf = malloc(sizeof(int) * (m + 1));
    for (int j = 0; j < m; j++) { ip[P[j]] = j; mark[j] = -1; head[j] = -1; }
    size_t total = 0;
    double flops = 0;
    int rc = 0;
    for (int j = 0; j < m && !rc; j++) {
        const int v = P[j];
        int len = 0;
        buf[len++] = v; mark[v] = j;
        for (int q = 0; q < adj[v].n; q++) { const int u = adj[v].a[q]; if (ip[u] > j && mark[u] != j) { mark[u] = j; buf[len++] = u; } }
        for (int c = head[j]; c >= 0; c = nxt[c])
            for (int q = 1; q < plen[c]; q++) { const int u = pat[c][q]; if (mark[u] != j) { mark[u] = j; buf[len++] = u; } }
        total += (size_t)len;
        flops += (double)len * len;
        if (total > fillcap || (g_flopcap > 0 && flops > g_flopcap)) { rc = 1; break; }
        pat[j] = malloc(sizeof(int) * len);
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
static SChol *schol_analyze(int m, SAdj *adj, size_t fillcap) {
    int *perm = NULL, *plen = NULL, **pat = NULL;
    double flops = 0;
    size_t nnz = 0;
    int ord_ok = 0;
#ifdef HAVE_AMD
    /* BRISK_AMD: 0 never, 1 when the own ordering gives up (default), 2 always */
    if (g_amd < 0) { const char *e = getenv("BRISK_AMD"); g_amd = e ? atoi(e) : 1; }
    if (g_amd_big < 0) { const char *e = getenv("BRISK_AMDBIG"); g_amd_big = e ? atoi(e) : 20000; }
    if (g_amd == 2 && m >= 64 && !schol_order_amd(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) ord_ok = 1;
    g_nd_used = 0;
    /* (only where the factorization costs: the dissection itself takes the time of a few of them.
     * 5.7: from 1e9 flops instead of 2e8: with relaxed supernodes and the left-looking code a
     * factorization of that size takes half the time it did, and the dissection, 0.4-0.5 s on
     * dfl001 and pds-10, no longer pays for itself there: 2.59 -> 2.23 s and 1.11 -> 0.96 s) */
    if (ord_ok && g_nd_try && m >= 1000 && flops >= (getenv("BRISK_NDMINFLOPS") ? atof(getenv("BRISK_NDMINFLOPS")) : 1e9) && !g_force_perm) {
        int *perm2 = NULL, *plen2 = NULL, **pat2 = NULL;
        double flops2 = 0; size_t nnz2 = 0;
        const double tm = (double)clock() / CLOCKS_PER_SEC;
        const int rs = schol_order_nd(m, adj, (size_t)(1.2 * (double)nnz) + (size_t)m, &perm2, &plen2, &pat2, &flops2, &nnz2);
        if (getenv("BRISK_AMDBG")) printf("   [ordering m %d: AMD %.3g flops %zu nnz; nested dissection %s %.3g flops %zu nnz (%.2fs)]\n", m, flops, nnz, rs ? "failed/over" : "ok", flops2, nnz2, (double)clock() / CLOCKS_PER_SEC - tm);
        if (rs == 0 && flops2 < 0.9 * flops) {
            for (int j = 0; j < m; j++) free(pat[j]);
            free(pat); free(plen); free(perm);
            perm = perm2; plen = plen2; pat = pat2; flops = flops2; nnz = nnz2; g_nd_used = 1;
        } else if (rs == 0) {
            for (int j = 0; j < m; j++) free(pat2[j]);
            free(pat2); free(plen2); free(perm2);
        }
    }
    /* auto (4.22): AMD first (fast); on large patterns it is kept (TSSOS case2869: 0.89 against
     * 1.3 GF per factorization, and the own ordering took 1.6 s), on small ones the own
     * ordering runs as well, capped at AMD's fill, and the cheaper factorization wins */
    size_t fillcap_own = fillcap;
    int have_amd = 0;
    if (g_amd == 1 && m >= 64 && !schol_order_amd(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) {
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
        else fillcap_own = (size_t)(1.1 * (double)nnz) + (size_t)m;
    }
    if (!ord_ok && have_amd) {
        int *perm2 = NULL, *plen2 = NULL, **pat2 = NULL;
        double flops2 = 0; size_t nnz2 = 0;
        int rs = schol_order_super(m, adj, fillcap_own, &perm2, &plen2, &pat2, &flops2, &nnz2);
        if (rs == 2 && schol_order(m, adj, fillcap_own, &perm2, &plen2, &pat2, &flops2, &nnz2)) rs = 1;
        else if (rs == 2) rs = 0;
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
        int rs = schol_order_super(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz);
        if (rs == 0) ord_ok = 1;
        else if (rs == 2 && !schol_order(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) ord_ok = 1;
    }
#ifdef HAVE_AMD
    if (!ord_ok && g_amd == 1 && m >= 64 && !schol_order_amd(m, adj, fillcap, &perm, &plen, &pat, &flops, &nnz)) ord_ok = 1;
#endif
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
    for (int j = 0; j < m; j++) {
        S->cp[j] = (int)p;
        /* column j in permuted indices: its clique, mapped and sorted, diagonal first */
        int len = plen[j];
        int *tmp = malloc(sizeof(int) * len);
        int k = 0;
        for (int q = 0; q < len; q++) {
            int r = S->iperm[pat[j][q]];
            if (r >= j) tmp[k++] = r;
        }
        qsort(tmp, k, sizeof(int), cmp_i);
        for (int q = 0; q < k; q++) S->ci[p++] = tmp[q];
        free(tmp);
        free(pat[j]);
    }
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
    free(S->perm); free(S->iperm); free(S->cp); free(S->ci); free(S->mcp); free(S->mri); free(S->moff); free(S->mval); free(S->mrun); free(S->mrp); free(S->t2); free(S->sgn); free(S->sw);
    free(S->acc); free(S->acc2); free(S->head); free(S->next); free(S->mark); free(S->stack);
    free(S->ss); free(S->snof); free(S->pbase); free(S->prow); free(S->prp); free(S->pnr);
    if (S->pan != S->pmat) free(S->pan);
    free(S->pmat); free(S->relind); free(S->relidx); free(S->upd); free(S->wrk);
    free(S->snhead); free(S->snnext); free(S->snptr); free(S->hkey); free(S->hval);
    free(S->mv_c); free(S->mv_zo); free(S->mv_z); free(S->mva); free(S->mv_za); free(S->sch_ptr); free(S->sch_idx);
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
    if (o != SIZE_MAX) { S->pmat[o] += v; S->mval_ok = 0; }
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
double g_tmv[2]; long g_nmv[2];   /* 4.26 debug: time and calls of schol_mv / schol_solve_tp */
static void schol_mv_(const SChol *S, const double *x, double *y);
void schol_mv(const SChol *S, const double *x, double *y) { double t0 = omp_get_wtime(); schol_mv_(S, x, y); g_tmv[0] += omp_get_wtime() - t0; g_nmv[0]++; }
static void mv_prepare(const SChol *S) {
    if (g_mvold < 0) g_mvold = getenv("BRISK_MVOLD") != NULL;
    sc_ensure(S);
    const int m = S->m;
    if (!S->mval_ok) {
        SChol *Sm = (SChol *)S;
        const size_t nq = (size_t)S->mcp[m];
        if (!Sm->mval) Sm->mval = sx(sizeof(double) * (nq + 1));
        for (size_t q = 0; q < nq; q++) Sm->mval[q] = S->pmat[S->moff[q]];
        Sm->mval_ok = 1;
    }
    if (S->run_state == 0) {
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
        } else Sm->run_state = -1;
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
                double sj = mv[S->mcp[j]] * xj;
                if (runs) {
                    for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                        const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                        const double *vq = mv + q, *xq = xp + i0;
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
            double sj = mv[S->mcp[j]] * xj;
            for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                const double *vq = mv + q;
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
                double sj = mv[S->mcp[j]] * xj, tj = fabs(mv[S->mcp[j]]) * aj;
                if (runs) {
                    for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                        const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                        const double *vq = mv + q, *xq = xp + i0, *aq = xa + i0;
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
            double sj = mv[S->mcp[j]] * xj, tj = fabs(mv[S->mcp[j]]) * aj;
            if (runs) {
                for (int t = S->mrp[j]; t < S->mrp[j + 1]; t++) {
                    const int q = S->mrun[3 * t], i0 = S->mrun[3 * t + 1], len = S->mrun[3 * t + 2];
                    const double *vq = mv + q, *xq = xp + i0, *aq = xa + i0;
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
    if (!(S->dynpiv > 0)) { BL(dpotrf_)("L", &w, Pk, &nr, &info); return info != 0; }
    const size_t nw = (size_t)w * w;
    if (nw > *cap) { *cap = 2 * nw; free(*buf); *buf = (double *)malloc(sizeof(double) * *cap); }
    double *T = *buf;
    for (int c = 0; c < w; c++) memcpy(T + (size_t)c * w + c, Pk + (size_t)c * nr + c, sizeof(double) * (size_t)(w - c));
    BL(dpotrf_)("L", &w, Pk, &nr, &info);
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
#define SIGNED_SB 48
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
        BL(dtrsm_)("R", "L", "T", "N", &below, &jb, &DONE, L11, &ld, P21, &ld);
        int npos = 0, nneg = 0;
        for (int c = 0; c < jb; c++) {
            double *col = P21 + (size_t)c * ld;
            if (sg[j0 + c] < 0) { for (int r = 0; r < below; r++) col[r] = -col[r]; nneg++; } else npos++;
        }
        double *P22 = P + (size_t)j1 * ld + j1;
        const double mone = -1.0, one = 1.0;
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
            for (int k2 = c0; k2 < c; k2++) {
                const double *ck = P + (size_t)k2 * ld;
                const double f = ck[c];
                if (f == 0) continue;
                for (int r = c; r < c0 + b; r++) col[r] -= f * ck[r];
            }
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
            BL(dtrsm_)("R", "L", "T", "N", &below, &b, &DONE, P + (size_t)c0 * ld + c0, &ld, P + (size_t)c0 * ld + c0 + b, &ld);
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
            if (w > 96 && !getenv("BRISK_TINYLOOP")) S->ntiny += chol_tiny_blocked(nr, w, Pk, nr, P0, nr + 1, 10.0 * S->tinypiv);
            else
            for (int c = 0; c < w; c++) {
                double *col = Pk + (size_t)c * nr;
                for (int k2 = 0; k2 < c; k2++) {
                    const double *ck = Pk + (size_t)k2 * nr;
                    const double f = ck[c];
                    if (f == 0) continue;
                    for (int r = c; r < nr; r++) col[r] -= f * ck[r];
                }
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
                BL(dtrsm_)("R", "L", "T", "N", &rest, &w, &DONE, Pk, &nr, Pk + w, &nr);
        } else if (panel_sign_uniform(S, j0, w) < 0) {
            /* all columns negative: -P = L L' by dpotrf on the negated panel gives the same
             * factor as the signed column loop (l_cc = sqrt(|d|), L_rc = -P_rc / l_cc), at
             * the BLAS rate (the loop made the bordered factorization 2.3x slower) */
            for (int c = 0; c < w; c++) { double *col = Pk + (size_t)c * nr; for (int r = c; r < nr; r++) col[r] = -col[r]; }
            if (panel_potrf_dyn(S, Pk, nr, w, &S->sw, &S->swcap)) return 1;
            if (rest > 0) BL(dtrsm_)("R", "L", "T", "N", &rest, &w, &DONE, Pk, &nr, Pk + w, &nr);
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
                BL(dtrsm_)("R", "L", "T", "N", &rest, &w, &DONE, T, &w, Pk + w, &nr);
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
static int FB_RB = 512, FB_NB = 128;   /* fixed per run (BRISK_FRB, BRISK_FNB): they define the arithmetic */
#if defined(__APPLE__) || defined(BRISK_NO_OPENBLAS)
static void (*const BL(openblas_set_num_threads))(int) = 0;
static int (*const BL(openblas_get_num_threads))(void) = 0;
#else
extern void BL(openblas_set_num_threads)(int) __attribute__((weak));
extern int BL(openblas_get_num_threads)(void) __attribute__((weak));
#endif
typedef struct { int *relind; double *upd; size_t updcap; double *sw; size_t swcap; double *ib; size_t ibcap; double *br; size_t brcap; double *bc; size_t bccap; double *ism; size_t ismcap; } FScr;
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
        for (int t = 0; t < g_nfscr; t++) { free(g_fscr[t].relind); free(g_fscr[t].upd); free(g_fscr[t].sw); free(g_fscr[t].ib); free(g_fscr[t].br); free(g_fscr[t].bc); free(g_fscr[t].ism); }
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
static void fnode_upd(SChol *S, int k, int R0, int R1, FScr *w) {
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
        const int q1 = lo, nq = q1 - q0;
        if (nq <= 0) continue;
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
        const int r0 = relind[rwj[q0]], c0 = rwj[0];
        int contig = (rwj[a - 1] == c0 + a - 1) && (relind[rwj[q1 - 1]] == r0 + nq - 1);
        /* a source of a few columns that is dense into the target tile: its columns go to a batch
         * and the whole batch is one dgemm (a dense block of order w fed by thousands of rows of
         * the data: 20,000 rank-one updates of a 200 x 200 block cost 0.2 s as scatter or as
         * dgemm with k = 1, and 0.01 s as one product) */
        if (wj <= 4 && S->dynpiv > 0 && a >= 8 && nq >= 8 && (double)a * nq >= 0.08 * (double)wk * (R1 - R0)) {
            if (!batch) batch = (int *)fscr_buf(&w->ib, &w->ibcap, 4 * (size_t)(S->ul_ptr[k + 1] - S->ul_ptr[k]) + 8);
            batch[4 * nbatch] = u; batch[4 * nbatch + 1] = q0; batch[4 * nbatch + 2] = q1; batch[4 * nbatch + 3] = 0;
            nbatch++; nbcols += wj;
            continue;
        }
        if (contig) {
            BL(dgemm_)("N", "T", &nq, &a, &wj, &mone, U + q0, &nrj, V, &ldv, &DONE, Pk + (size_t)(c0 - j0) * nr + r0, &nr);
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
        double *upd = fscr_buf(&w->upd, &w->updcap, (size_t)nq * a);
        BL(dgemm_)("N", "T", &nq, &a, &wj, &DONE, U + q0, &nrj, V, &ldv, &DZERO, upd, &nq);
        for (int cc = 0; cc < a; cc++) {
            double *dst = Pk + (size_t)(rwj[cc] - j0) * nr;
            const double *src = upd + (size_t)cc * nq;
            for (int q = (q0 > cc ? q0 : cc); q < q1; q++) dst[relind[rwj[q]]] -= src[q - q0];
        }
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
                if (cc >= a) continue;
                const int nrj = S->pnr[j], wj = S->ss[j + 1] - S->ss[j];
                const int *rwj = S->prow + S->prp[j] + p;
                if (rwj[cc] >= chi) continue;
                const double *U = S->pan + S->pbase[j] + p;
                const double *U1 = U + nrj, *U2 = U + 2 * (size_t)nrj, *U3 = U + 3 * (size_t)nrj;
                double sg[4] = { 1.0, 1.0, 1.0, 1.0 };
                if (S->sgn) for (int c = 0; c < wj; c++) sg[c] = S->sgn[S->ss[j] + c];
                for (; cc < a && rwj[cc] < chi; cc++) {
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
        const int cb = (w - c0 < FB_NB) ? w - c0 : FB_NB;
        double *D = P + (size_t)c0 * nr + c0;
        int info = 0;
        BL(dpotrf_)("L", &cb, D, &nr, &info);
        if (info) { bad = 1; break; }
        const int rb0 = c0 + cb;
        for (int r = rb0; r < nr; r += FB_RB) {
            const int rr = (nr - r < FB_RB) ? nr - r : FB_RB;
            double *B = P + (size_t)c0 * nr + r;
            if (par) {
                #pragma omp task firstprivate(B, rr) shared(D, nr, cb)
                BL(dtrsm_)("R", "L", "T", "N", &rr, &cb, &DONE, D, &nr, B, &nr);
            } else BL(dtrsm_)("R", "L", "T", "N", &rr, &cb, &DONE, D, &nr, B, &nr);
        }
        if (par) {
            #pragma omp taskwait
        }
        const double mone = -1.0;
        for (int c1 = rb0; c1 < w; c1 += FB_NB) {
            const int cb1 = (w - c1 < FB_NB) ? w - c1 : FB_NB;
            const double *A1 = P + (size_t)c0 * nr + c1;         /* rows c1.., columns of tile c0 */
            double *Cd = P + (size_t)c1 * nr + c1;
            if (par) {
                #pragma omp task firstprivate(A1, Cd, cb1) shared(nr, cb)
                BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &nr, &DONE, Cd, &nr);
            } else BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &nr, &DONE, Cd, &nr);
            for (int r = c1 + cb1; r < nr; r += FB_RB) {
                const int rr = (nr - r < FB_RB) ? nr - r : FB_RB;
                const double *Ar = P + (size_t)c0 * nr + r;
                double *Cr = P + (size_t)c1 * nr + r;
                if (par) {
                    #pragma omp task firstprivate(A1, Ar, Cr, rr, cb1) shared(nr, cb)
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
static int fnode_factor(SChol *S, int k, FScr *wsc, int par) {
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    double *Pk = S->pan + S->pbase[k];
    int neg = 0;
    if (S->sgn) for (int c = 0; c < w; c++) if (S->sgn[j0 + c] < 0) { neg = 1; break; }
    if (!neg && S->tinypiv > 0) {
        const double *P0 = S->pmat + S->pbase[k];
        int nt = 0;
        if (w > 96 && !getenv("BRISK_TINYLOOP")) nt = chol_tiny_blocked(nr, w, Pk, nr, P0, nr + 1, 10.0 * S->tinypiv);
        else
        for (int c = 0; c < w; c++) {
            double *col = Pk + (size_t)c * nr;
            for (int k2 = 0; k2 < c; k2++) {
                const double *ck = Pk + (size_t)k2 * nr;
                const double f = ck[c];
                if (f == 0) continue;
                for (int r = c; r < nr; r++) col[r] -= f * ck[r];
            }
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
        if (w > FB_NB && !(S->dynpiv > 0)) return fpanel_tiled(Pk, nr, w, par);
        if (panel_potrf_dyn(S, Pk, nr, w, &wsc->sw, &wsc->swcap)) return 1;
        if (rest > 0) BL(dtrsm_)("R", "L", "T", "N", &rest, &w, &DONE, Pk, &nr, Pk + w, &nr);
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
        BL(dtrsm_)("R", "L", "T", "N", &rest, &w, &DONE, T, &w, Pk + w, &nr);
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
/* 4.25: row tiles of the dense (Schur, X, Z) Cholesky: 2048 rows instead of the sparse
 * factor's 512 (fewer OpenBLAS packings; still fixed, so the same bits at any thread count) */
static int FD_RB = 2048;
static void pd_env(void);
static int fpanel_tiled_ld(double *P, int nr, int w, int ld, int par) {
    int bad = 0;
    const double mone = -1.0;
    for (int c0 = 0; c0 < w && !bad; c0 += FB_NB) {
        const int cb = (w - c0 < FB_NB) ? w - c0 : FB_NB;
        double *D = P + (size_t)c0 * ld + c0;
        int info = 0;
        BL(dpotrf_)("L", &cb, D, &ld, &info);
        if (info) { bad = 1; break; }
        for (int r = c0 + cb; r < nr; r += FD_RB) {
            const int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
            double *B = P + (size_t)c0 * ld + r;
            if (par) {
                #pragma omp task firstprivate(B, rr) shared(D, ld, cb)
                BL(dtrsm_)("R", "L", "T", "N", &rr, &cb, &DONE, D, &ld, B, &ld);
            } else BL(dtrsm_)("R", "L", "T", "N", &rr, &cb, &DONE, D, &ld, B, &ld);
        }
        if (par) {
            #pragma omp taskwait
        }
        for (int c1 = c0 + cb; c1 < w; c1 += FB_NB) {
            const int cb1 = (w - c1 < FB_NB) ? w - c1 : FB_NB;
            const double *A1 = P + (size_t)c0 * ld + c1;
            double *Cd = P + (size_t)c1 * ld + c1;
            if (par) {
                #pragma omp task firstprivate(A1, Cd, cb1) shared(ld, cb)
                BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &DONE, Cd, &ld);
            } else BL(dsyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &DONE, Cd, &ld);
            for (int r = c1 + cb1; r < nr; r += FD_RB) {
                const int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
                const double *Ar = P + (size_t)c0 * ld + r;
                double *Cr = P + (size_t)c1 * ld + r;
                if (par) {
                    #pragma omp task firstprivate(A1, Ar, Cr, rr, cb1) shared(ld, cb)
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
        const int cb = (w - c0 < FB_NB) ? w - c0 : FB_NB;
        float *D = P + (size_t)c0 * ld + c0;
        int info = 0;
        BL(spotrf_)("L", &cb, D, &ld, &info);
        if (info) { bad = 1; break; }
        for (int r = c0 + cb; r < nr; r += FD_RB) {
            const int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
            float *B = P + (size_t)c0 * ld + r;
            if (par) {
                #pragma omp task firstprivate(B, rr) shared(D, ld, cb, one)
                BL(strsm_)("R", "L", "T", "N", &rr, &cb, &one, D, &ld, B, &ld);
            } else BL(strsm_)("R", "L", "T", "N", &rr, &cb, &one, D, &ld, B, &ld);
        }
        if (par) {
            #pragma omp taskwait
        }
        for (int c1 = c0 + cb; c1 < w; c1 += FB_NB) {
            const int cb1 = (w - c1 < FB_NB) ? w - c1 : FB_NB;
            const float *A1 = P + (size_t)c0 * ld + c1;
            float *Cd = P + (size_t)c1 * ld + c1;
            if (par) {
                #pragma omp task firstprivate(A1, Cd, cb1) shared(ld, cb, one, mone)
                BL(ssyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &one, Cd, &ld);
            } else BL(ssyrk_)("L", "N", &cb1, &cb, &mone, A1, &ld, &one, Cd, &ld);
            for (int r = c1 + cb1; r < nr; r += FD_RB) {
                const int rr = (nr - r < FD_RB) ? nr - r : FD_RB;
                const float *Ar = P + (size_t)c0 * ld + r;
                float *Cr = P + (size_t)c1 * ld + r;
                if (par) {
                    #pragma omp task firstprivate(A1, Ar, Cr, rr, cb1) shared(ld, cb, one, mone)
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
int brisk_dpotrf(int n, double *A, int lda) {
    if (n <= 0) return 0;                      /* 4.25: m = 0 (LAPACK rejects lda = 0) */
    pd_env();
    int info = 0;
    if (n < FB_DENSE_MIN) { BL(dpotrf_)("L", &n, A, &lda, &info); return info; }
    int nt = 1;
#ifdef _OPENMP
    nt = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    if (nt <= 1) return fpanel_tiled_ld(A, n, n, lda, 0);
    int bad = 0;
    #pragma omp parallel
    #pragma omp single
    bad = fpanel_tiled_ld(A, n, n, lda, 1);
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
    if (nt <= 1) return fpanel_tiled_s(A, n, n, lda, 0);
    int bad = 0;
    #pragma omp parallel
    #pragma omp single
    bad = fpanel_tiled_s(A, n, n, lda, 1);
    return bad;
}
/* 4.24: BRISK owns the threads: OpenBLAS runs single-threaded (its pthread workers spin
 * next to the OpenMP ones: case14 0.02 -> 0.14 s at 2 threads; its threaded kernels also
 * change the results with the thread count). BRISK_BLASMT=1 keeps OpenBLAS threading. */
void brisk_threads_init(void) {
    if (getenv("BRISK_BLASMT")) return;
    if (BL(openblas_set_num_threads)) BL(openblas_set_num_threads)(1);
}

/* 4.24: tiled dgemm / dsymm for large products, parallel outside parallel regions. The
 * tiles are fixed by the sizes, so the arithmetic is the same for any thread count;
 * below PD_MIN (m n k) the plain BLAS call. */
#define PD_MIN 8e6
/* 4.25: 2048 x 512 tiles (1024 x 256 in 4.24 cost 29% against one dgemm at n = 5000 - A was
 * packed once per column tile - and made maxG55/60 10-15% slower than 4.13; now 11%) */
static int PD_RB = 2048, PD_CB = 512;   /* fixed tiles (BRISK_PDRB, BRISK_PDCB): they define the arithmetic */
static void pd_env(void) {
    static int done = 0;
    if (done) return;
    const char *e;
    if ((e = getenv("BRISK_PDRB"))) PD_RB = atoi(e);
    if ((e = getenv("BRISK_PDCB"))) PD_CB = atoi(e);
    if ((e = getenv("BRISK_FNB"))) FB_NB = atoi(e);
    if ((e = getenv("BRISK_FDRB"))) FD_RB = atoi(e);
    done = 1;
}
void pdgemm(const char *ta, const char *tb, const int *m_, const int *n_, const int *k_, const double *alpha,
            const double *A, const int *lda_, const double *B, const int *ldb_, const double *beta, double *C, const int *ldc_) {
    const int m = *m_, n = *n_, k = *k_, lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    pd_env();
    if ((double)m * n * k < PD_MIN || getenv_cached_pd()) { BL(dgemm_)(ta, tb, m_, n_, k_, alpha, A, lda_, B, ldb_, beta, C, ldc_); return; }
    const int tA = ta[0] == 'T' || ta[0] == 't', tB = tb[0] == 'T' || tb[0] == 't';
    const int nr = (m + PD_RB - 1) / PD_RB, nc = (n + PD_CB - 1) / PD_CB, nt = nr * nc;
    #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel() && nt > 1)
    for (int t = 0; t < nt; t++) {
        const int i0 = (t % nr) * PD_RB, j0 = (t / nr) * PD_CB;
        const int mi = m - i0 < PD_RB ? m - i0 : PD_RB, nj = n - j0 < PD_CB ? n - j0 : PD_CB;
        const double *Ai = tA ? A + (size_t)i0 * lda : A + i0;
        const double *Bj = tB ? B + j0 : B + (size_t)j0 * ldb;
        BL(dgemm_)(ta, tb, &mi, &nj, k_, alpha, Ai, lda_, Bj, ldb_, beta, C + i0 + (size_t)j0 * ldc, ldc_);
    }
}
void pdsymm(const char *side, const char *uplo, const int *m_, const int *n_, const double *alpha,
            const double *A, const int *lda_, const double *B, const int *ldb_, const double *beta, double *C, const int *ldc_) {
    const int m = *m_, n = *n_, ldb = *ldb_, ldc = *ldc_;
    const int left = side[0] == 'L' || side[0] == 'l', k = left ? m : n;
    pd_env();
    if ((double)m * n * k < PD_MIN || getenv_cached_pd()) { BL(dsymm_)(side, uplo, m_, n_, alpha, A, lda_, B, ldb_, beta, C, ldc_); return; }
    if (left) {        /* C(:, J) = A B(:, J) */
        const int nc = (n + PD_CB - 1) / PD_CB;
        #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel() && nc > 1)
        for (int t = 0; t < nc; t++) {
            const int j0 = t * PD_CB, nj = n - j0 < PD_CB ? n - j0 : PD_CB;
            BL(dsymm_)(side, uplo, m_, &nj, alpha, A, lda_, B + (size_t)j0 * ldb, ldb_, beta, C + (size_t)j0 * ldc, ldc_);
        }
    } else {           /* C(I, :) = B(I, :) A */
        const int nr = (m + PD_RB - 1) / PD_RB;
        #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel() && nr > 1)
        for (int t = 0; t < nr; t++) {
            const int i0 = t * PD_RB, mi = m - i0 < PD_RB ? m - i0 : PD_RB;
            BL(dsymm_)(side, uplo, &mi, n_, alpha, A, lda_, B + i0, ldb_, beta, C + i0, ldc_);
        }
    }
}
/* one node: shift, updates (row blocks, as tasks when par), panel factorization */
static void fnode(SChol *S, int k, double shift, int par) {
    if (S->fail) return;
    const int w = S->ss[k + 1] - S->ss[k], nr = S->pnr[k];
    double *Pk = S->pan + S->pbase[k];
    if (shift != 0)
        for (int cc = 0; cc < w; cc++) { double *d = &Pk[(size_t)cc * nr + cc]; *d += shift * (fabs(*d) > 0 ? fabs(*d) : 1.0); }
    const int nblk = (nr + FB_RB - 1) / FB_RB;
    if (par && nblk > 1) {
        for (int b = 0; b < nblk; b++) {
            #pragma omp task firstprivate(b)
            fnode_upd(S, k, b * FB_RB, (b + 1) * FB_RB < nr ? (b + 1) * FB_RB : nr, fscr_get());
        }
        #pragma omp taskwait
    } else
        for (int b = 0; b < nblk; b++) fnode_upd(S, k, b * FB_RB, (b + 1) * FB_RB < nr ? (b + 1) * FB_RB : nr, fscr_get());
    if (fnode_factor(S, k, fscr_get(), par)) S->fail = 1;
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
int schol_factor(SChol *S, double shift) {
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
        if (!S->pan || S->pan == S->pmat) S->pan = malloc(sizeof(double) * (S->pansz ? S->pansz : 1));
        if (!S->pan) { fprintf(stderr, "brisk: out of memory (the copy of a sparse factor, %.1f GB)\n", 8.0 * (double)S->pansz / 1e9); exit(1); }
        memcpy(S->pan, S->pmat, sizeof(double) * S->pansz);
    }
    if (!S->ul_ptr) sc_build_tree(S);
    S->fail = 0;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (omp_in_parallel()) nt = 1;
#endif
    fscr_ensure(nt, S->m);
    const double tot = ns > 0 ? S->ssub[ns - 1] + 0.0 : 0.0;
    double totw = 0; for (int k = 0; k < ns; k++) if (S->spar[k] < 0) totw += S->ssub[k];
    (void)tot;
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
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < nunit; i++) {
            const int u = units[i];
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
                const int kk = k;
                #pragma omp task firstprivate(kk)
                fnode_chain(S, kk, shift, nodepar);
            }
        /* big nodes whose children are all big and have no units: covered by the chain */
    }
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
/* forward step of node k on xt (k rhs, transposed); ext[k] receives its contribution */
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
        if (rest > 0) {
            const double *lo = col + w;
            if (nr_ == 1) { const double v = xc[0]; for (int q = 0; q < rest; q++) e[q] += lo[q] * v; }
            else for (int q = 0; q < rest; q++) { double *eq = e + (size_t)q * nr_; const double l = lo[q]; for (int r = 0; r < nr_; r++) eq[r] += l * xc[r]; }
        }
    }
    g_ext[k] = e;
}
static inline __attribute__((always_inline)) void tps_bwd_node_i(const SChol *S, int k, double *xt, const int nr_, double *t) {
    const int j0 = S->ss[k], w = S->ss[k + 1] - j0, nr = S->pnr[k], rest = nr - w;
    const int *rw = S->prow + S->prp[k];
    const double *P = S->pan + S->pbase[k];
    double *xs = xt + (size_t)j0 * nr_;
    if (nr_ == 1) for (int q = 0; q < rest; q++) t[q] = xt[rw[w + q]];
    else for (int q = 0; q < rest; q++) { const double *src = xt + (size_t)rw[w + q] * nr_; double *dst = t + (size_t)q * nr_; for (int r = 0; r < nr_; r++) dst[r] = src[r]; }
    for (int c = w - 1; c >= 0; c--) {
        const double *col = P + (size_t)c * nr;
        double *xc = xs + (size_t)c * nr_;
        const double *lo = col + w;
        if (nr_ == 1) {
            double a1 = 0, a2 = 0;
            if (rest >= 32) {
                #pragma omp simd reduction(+:a1)
                for (int q = 0; q < rest; q++) a1 += lo[q] * t[q];
            } else for (int q = 0; q < rest; q++) a1 += lo[q] * t[q];
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
                const double *lo = col + w;
                for (int q = 0; q < rest; q++) t[q] += lo[q] * v;
            } else {
                for (int i = c + 1; i < w; i++) { double *xi = xs + (size_t)i * nr_; const double l = col[i]; for (int r = 0; r < nr_; r++) xi[r] -= l * xc[r]; }
                const double *lo = col + w;
                for (int q = 0; q < rest; q++) { double *tq = t + (size_t)q * nr_; const double l = lo[q]; for (int r = 0; r < nr_; r++) tq[r] += l * xc[r]; }
            }
        }
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
            const int c = S->sch_idx[u];
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
        for (int k = 0; k < ns; k++) {
            if (!big[k] && (S->spar[k] < 0 || big[S->spar[k]])) tps_fwd_unit(S, k, xt, nr_);
            else if (big[k]) tps_fwd_node(S, k, xt, nr_, NULL, fscr_get()->relind);
        }
        if (S->sgn) for (int i = 0; i < m; i++) if (S->sgn[i] < 0) for (int r = 0; r < nr_; r++) xt[(size_t)i * nr_ + r] = -xt[(size_t)i * nr_ + r];
        if (tdbg) tt1 = omp_get_wtime();
        double *t = malloc(sizeof(double) * tmax);
        for (int k = ns - 1; k >= 0; k--) tps_bwd_node(S, k, xt, nr_, t);
        free(t);
        if (tdbg) { tt2 = omp_get_wtime(); static int cnt = 0; if (cnt++ < 3) printf("   [tps serial: fwd %.1f ms bwd %.1f ms]\n", 1e3 * (tt1 - tt0), 1e3 * (tt2 - tt1)); }
        for (int k = 0; k < ns; k++) if (S->spar[k] < 0 && g_ext[k]) { free(g_ext[k]); g_ext[k] = NULL; }
    } else {
        #pragma omp parallel
        #pragma omp single
        {
            for (int k = 0; k < ns; k++) {
                if (!big[k] && (S->spar[k] < 0 || big[S->spar[k]])) {
                    const int u = k;
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
                    const int kk = k;
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
                const int kk = k;
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
            const double *restrict lo = col + w;
            for (int q = 0; q < rest; q++) t[q] += lo[q] * xc;
        }
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
        for (int c = w - 1; c >= 0; c--) {
            const double *restrict col = Pk + (size_t)c * nr;
            double v = xs[c];
            const double *restrict lo = col + w;
            for (int q = 0; q < rest; q++) v -= lo[q] * t[q];
            for (int r = c + 1; r < w; r++) v -= col[r] * xs[r];
            xs[c] = v / col[c];
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
