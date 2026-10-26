/* nd.c - a nested dissection ordering for sparse Cholesky factorizations
 *
 *     int nd_order(int n, const int *xadj, const int *adjncy, int *perm);     perm[new] = old; 0: done
 *
 * The graph (symmetric adjacency lists, no self loops needed) is split recursively by a vertex
 * separator; the two parts are ordered first, the separator last, and parts below a leaf size by
 * AMD on their own subgraph. A separator comes from a multilevel edge bisection:
 *   - coarsening by heavy-edge matching (vertices in a random order, each matched with the
 *     unmatched neighbour of the heaviest edge) down to about a hundred vertices;
 *   - on the coarsest graph, regions grown breadth-first from several seeds to half the weight;
 *   - at every level a refinement in the manner of Fiduccia and Mattheyses: boundary vertices
 *     move by gain from heaps, with moves of negative gain allowed and undone back to the best
 *     cut of the pass;
 *   - the edge cut becomes a vertex separator by a minimum vertex cover of the bipartite graph of
 *     the cut edges (a maximum matching and Koenig's construction).
 * Deterministic (its own random numbers). The caller compares the factorization this order
 * gives with AMD's and keeps the cheaper one: nested dissection wins on meshes and networks
 * (grids, multicommodity flows), AMD on most other patterns.                                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include "amd/amd.h"

typedef struct { int n; int *xadj, *adj, *ew, *vw; long tvw; } Gr;

/* (21 October 2026) a time budget: once past it the rest is ordered by AMD and nd_order reports failure (the caller keeps its
 * own order); 0: none */
/* (23 October 2026, the night) the budget in work, not in seconds: the edges each phase goes over (coarsening, the
 * refinements' passes, the vertex cover, the leaves' AMD), so the order is the same on every run and at any load; the
 * clock made it depend on the host and on the other threads (triptim1, LP: factors of 7.8 and 14.1 million entries from
 * one run to the next). The caller's seconds are converted at ND_RATE units a second; thread-local, since orderings run
 * on several threads at once. BRISK_NDCLOCK=1: the wall clock as before. */
#define ND_RATE 2e8
static _Thread_local double g_nd_wdead = 0, g_nd_work = 0, g_nd_dead = 0; static _Thread_local int g_nd_abort = 0;
/* (24 October 2026, the night) the flops of the order the dissection must beat (AMD's), for the bounds on its first
 * bisection (bisect, nd_rec); 0: none. g_nd_first: the first bisection not yet made; g_nd_top: the vertices already put
 * last above it (hubs) */
static _Thread_local double g_nd_target = 0; static _Thread_local int g_nd_first = 0, g_nd_top = 0;
/* a first separator of s vertices that cannot lead to an order beating the target: ordered last, it fills into a dense
 * block whose factorization alone costs s^3/3 flops. (A first try that does not separate is no reason to stop: a later
 * try may; stopping there lost opm2-z10-s4's order, LP, 1.14e10 flops against AMD's 1.76e10.) */
static int nd_first_hopeless(double s, double small) {
    (void)small;
    if (!g_nd_first || g_nd_target <= 0) return 0;
    const double st = s + g_nd_top;
    return st * st * st / 3.0 > g_nd_target;
}
static double nd_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int nd_clock(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_NDCLOCK"); c = e && atoi(e) == 1; } return c; }
void nd_set_deadline(double seconds) {
    g_nd_wdead = seconds > 0 ? g_nd_work + seconds * ND_RATE : 0; g_nd_dead = seconds > 0 ? nd_now() + seconds : 0; g_nd_abort = 0;
}
double nd_work_count(void) { return g_nd_work; }
double nd_work_rate(void) { return ND_RATE; }
static int nd_past(void) { return nd_clock() ? (g_nd_dead > 0 && nd_now() > g_nd_dead) : (g_nd_wdead > 0 && g_nd_work > g_nd_wdead); }
/* (22 October 2026, the SOCP developer, for the QP method) a budget of work set by the caller (sparsechol.c, when the
 * QP method sets the ordering's budget): the vertices and edges the coarsening and the refinement passes go through,
 * counted as on its branch (ND_WORK), so that the QP method's orders are those it was measured with; 0: none. Merged on
 * 26 October 2026 beside the LP developer's counted deadline above (g_nd_work, in its own units) */
static _Thread_local double g_nd_qwork = 0, g_nd_wlim = 0;
void nd_set_work_budget(double units) { g_nd_wlim = units > 0 ? units : 0; if (units > 0) g_nd_qwork = 0; g_nd_abort = 0; }   /* (the count kept after a 0, for the log) */
double nd_work_done(void) { return g_nd_qwork; }
int nd_aborted(void) { return g_nd_abort; }      /* (the last nd_order stopped at its budget) */
static int nd_over(void) { return nd_past() || (g_nd_wlim > 0 && g_nd_qwork > g_nd_wlim); }
#define ND_WORK(units) (g_nd_qwork += (double)(units))
static _Thread_local unsigned int nd_seed;     /* (per thread: orderings run on several threads at once) */
static unsigned int nd_rand(void) { nd_seed = nd_seed * 1664525u + 1013904223u; return nd_seed >> 8; }
static void gr_free(Gr *g) { free(g->xadj); free(g->adj); free(g->ew); free(g->vw); }

/* one level of coarsening; cmap[v]: the coarse vertex of v. 0: done, 1: no memory */
static int coarsen(const Gr *g, Gr *c, int *cmap, long maxvw) {
    const int n = g->n;
    g_nd_work += 2.0 * g->xadj[n] + n;
    ND_WORK(n + g->xadj[n]);
    int *match = malloc(sizeof(int) * (n + 1)), *ord = malloc(sizeof(int) * (n + 1));
    if (!match || !ord) { free(match); free(ord); return 1; }
    for (int i = 0; i < n; i++) { match[i] = -1; ord[i] = i; }
    for (int i = n - 1; i > 0; i--) { const int j = (int)(nd_rand() % (unsigned)(i + 1)), t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    /* (21 October 2026) the random order sorted by degree, ascending and stable, as METIS's sorted heavy-edge matching:
     * the vertices of few neighbours match first, while their neighbours are free (rail01 1.12e10 -> 1.07e10 flops,
     * triptim1 6.5e9 -> 6.0e9, pds-20 1.52e9 -> 1.35e9; over 14 LP normal equations 0.673 -> 0.663 of AMD's flops) */
    {
        int maxd = 0; for (int v = 0; v < n; v++) { const int d = g->xadj[v + 1] - g->xadj[v]; if (d > maxd) maxd = d; }
        int *cnt = calloc((size_t)maxd + 2, sizeof(int)), *o2 = malloc(sizeof(int) * (n + 1));
        if (cnt && o2) {
            for (int k = 0; k < n; k++) cnt[g->xadj[ord[k] + 1] - g->xadj[ord[k]] + 1]++;
            for (int d = 0; d <= maxd; d++) cnt[d + 1] += cnt[d];
            for (int k = 0; k < n; k++) { const int v = ord[k]; o2[cnt[g->xadj[v + 1] - g->xadj[v]]++] = v; }
            memcpy(ord, o2, sizeof(int) * n);
        }
        free(cnt); free(o2);
    }
    int nc = 0;
    for (int k = 0; k < n; k++) {
        const int v = ord[k];
        if (match[v] >= 0) continue;
        int best = -1, bw = -1;
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
            const int u = g->adj[p];
            if (match[u] < 0 && u != v && g->ew[p] > bw && g->vw[v] + g->vw[u] <= maxvw) { bw = g->ew[p]; best = u; }
        }
        if (best >= 0) { match[v] = best; match[best] = v; cmap[v] = cmap[best] = nc++; }
        else { match[v] = v; cmap[v] = nc++; }
    }
    c->n = nc; c->tvw = g->tvw;
    c->xadj = malloc(sizeof(int) * (nc + 1)); c->vw = calloc((size_t)nc + 1, sizeof(int));
    c->adj = malloc(sizeof(int) * (g->xadj[n] + 1)); c->ew = malloc(sizeof(int) * (g->xadj[n] + 1));
    int *pos = malloc(sizeof(int) * (nc + 1));
    if (!c->xadj || !c->vw || !c->adj || !c->ew || !pos) { free(match); free(ord); free(pos); gr_free(c); return 1; }
    for (int i = 0; i < nc; i++) pos[i] = -1;
    int w = 0, cv = 0;
    for (int k = 0; k < n; k++) {
        const int v = ord[k];
        if (cmap[v] != cv) continue;                 /* (coarse vertices were numbered in this order: first member) */
        c->xadj[cv] = w;
        for (int t = 0; t < 2; t++) {
            const int x = t == 0 ? v : match[v];
            if (t == 1 && x == v) break;
            c->vw[cv] += g->vw[x];
            for (int p = g->xadj[x]; p < g->xadj[x + 1]; p++) {
                const int cu = cmap[g->adj[p]];
                if (cu == cv) continue;
                if (pos[cu] >= c->xadj[cv]) c->ew[pos[cu]] += g->ew[p];
                else { pos[cu] = w; c->adj[w] = cu; c->ew[w] = g->ew[p]; w++; }
            }
        }
        cv++;
    }
    c->xadj[nc] = w;
    free(match); free(ord); free(pos);
    return 0;
}

/* a binary max-heap of (gain, vertex) with lazy deletion */
typedef struct { int *g, *v; int n, cap; } Heap;
static void hpush(Heap *h, int gain, int v) {
    if (h->n == h->cap) { h->cap = 2 * h->cap + 64; h->g = realloc(h->g, sizeof(int) * h->cap); h->v = realloc(h->v, sizeof(int) * h->cap); }
    int i = h->n++;
    while (i > 0) { const int p = (i - 1) >> 1; if (h->g[p] >= gain) break; h->g[i] = h->g[p]; h->v[i] = h->v[p]; i = p; }
    h->g[i] = gain; h->v[i] = v;
}
static void hpop(Heap *h) {
    const int gain = h->g[--h->n], v = h->v[h->n];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= h->n) break;
        if (c + 1 < h->n && h->g[c + 1] > h->g[c]) c++;
        if (h->g[c] <= gain) break;
        h->g[i] = h->g[c]; h->v[i] = h->v[c]; i = c;
    }
    h->g[i] = gain; h->v[i] = v;
}

/* (24 October 2026) the same heap indexed by vertex: a vertex is in it at most once and a new gain updates its entry in
 * place, where the plain heap took a new entry and left the old one to be popped as stale (a vertex pushed once for every
 * neighbour that moved: on rail01, LP, 14 million pops in the analysis, a sixth of its time). The choices are those of
 * the plain heap up to the order among equal gains. BRISK_NDIHEAP=0: the plain heap */
static int nd_iheap(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_NDIHEAP"); c = !(e && atoi(e) == 0); } return c; }
typedef struct { int *g, *v, *pos; int n, cap; } IHeap;
static void ih_init(IHeap *h, int nv) { h->g = h->v = NULL; h->n = h->cap = 0; h->pos = malloc(sizeof(int) * ((size_t)nv + 1)); for (int i = 0; i < nv; i++) h->pos[i] = -1; }
static void ih_free(IHeap *h) { free(h->g); free(h->v); free(h->pos); }
static void ih_clear(IHeap *h) { for (int i = 0; i < h->n; i++) h->pos[h->v[i]] = -1; h->n = 0; }
static void ih_up(IHeap *h, int i) {
    const int gain = h->g[i], v = h->v[i];
    while (i > 0) { const int p = (i - 1) >> 1; if (h->g[p] >= gain) break; h->g[i] = h->g[p]; h->v[i] = h->v[p]; h->pos[h->v[i]] = i; i = p; }
    h->g[i] = gain; h->v[i] = v; h->pos[v] = i;
}
static void ih_down(IHeap *h, int i) {
    const int gain = h->g[i], v = h->v[i];
    for (;;) {
        int c = 2 * i + 1;
        if (c >= h->n) break;
        if (c + 1 < h->n && h->g[c + 1] > h->g[c]) c++;
        if (h->g[c] <= gain) break;
        h->g[i] = h->g[c]; h->v[i] = h->v[c]; h->pos[h->v[i]] = i; i = c;
    }
    h->g[i] = gain; h->v[i] = v; h->pos[v] = i;
}
static void ih_set(IHeap *h, int gain, int v) {
    int i = h->pos[v];
    if (i >= 0) { const int old = h->g[i]; h->g[i] = gain; if (gain > old) ih_up(h, i); else if (gain < old) ih_down(h, i); return; }
    if (h->n == h->cap) { h->cap = 2 * h->cap + 64; h->g = realloc(h->g, sizeof(int) * h->cap); h->v = realloc(h->v, sizeof(int) * h->cap); }
    i = h->n++; h->g[i] = gain; h->v[i] = v; h->pos[v] = i; ih_up(h, i);
}
static void ih_pop(IHeap *h) {
    h->pos[h->v[0]] = -1;
    if (--h->n == 0) return;
    h->g[0] = h->g[h->n]; h->v[0] = h->v[h->n]; h->pos[h->v[0]] = 0; ih_down(h, 0);
}

/* the heap of the refinements: indexed or plain (nd_iheap) */
typedef struct { Heap h; IHeap ih; int idx; } PQ;
static void pq_init(PQ *q, int nv) { q->idx = nd_iheap(); q->h.g = q->h.v = NULL; q->h.n = q->h.cap = 0; if (q->idx) ih_init(&q->ih, nv); }
static void pq_free(PQ *q) { if (q->idx) ih_free(&q->ih); else { free(q->h.g); free(q->h.v); } }
static void pq_clear(PQ *q) { if (q->idx) ih_clear(&q->ih); else q->h.n = 0; }
static inline int pq_n(const PQ *q) { return q->idx ? q->ih.n : q->h.n; }
static inline int pq_v(const PQ *q) { return q->idx ? q->ih.v[0] : q->h.v[0]; }
static inline int pq_g(const PQ *q) { return q->idx ? q->ih.g[0] : q->h.g[0]; }
static inline void pq_push(PQ *q, int gain, int v) { if (q->idx) ih_set(&q->ih, gain, v); else hpush(&q->h, gain, v); }
static inline void pq_pop(PQ *q) { if (q->idx) ih_pop(&q->ih); else hpop(&q->h); }

/* refinement of the bisection part[] (0/1); returns the cut */
static long refine(const Gr *g, char *part, long maxside, int npass) {
    const int n = g->n;
    g_nd_work += (double)(npass + 1) * (g->xadj[n] + n);
    ND_WORK((double)npass * (n + g->xadj[n]));
    int *ed = malloc(sizeof(int) * (n + 1)), *id = malloc(sizeof(int) * (n + 1)), *moved = malloc(sizeof(int) * (n + 1));
    char *lock = malloc((size_t)n + 1);
    PQ H[2]; pq_init(&H[0], n); pq_init(&H[1], n);
    long cut = 0, w[2] = { 0, 0 };
    for (int v = 0; v < n; v++) {
        int e = 0, i = 0;
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { if (part[g->adj[p]] != part[v]) e += g->ew[p]; else i += g->ew[p]; }
        ed[v] = e; id[v] = i; cut += e; w[(int)part[v]] += g->vw[v];
    }
    cut /= 2;
    for (int pass = 0; pass < npass; pass++) {
        memset(lock, 0, (size_t)n);
        pq_clear(&H[0]); pq_clear(&H[1]);
        for (int v = 0; v < n; v++) if (ed[v] > 0 || g->xadj[v + 1] == g->xadj[v]) pq_push(&H[(int)part[v]], ed[v] - id[v], v);
        const int ok0 = w[0] <= maxside && w[1] <= maxside;
        long best = ok0 ? cut : LONG_MAX, cur = cut, bestimb = labs(w[0] - w[1]);
        int nm = 0, bestk = 0, bad = 0;
        const int limit = 100 + n / 50;
        while (bad < limit) {
            /* the side to move from: the one over its limit, else the better gain */
            int side = -1;
            for (int s = 0; s < 2; s++) while (pq_n(&H[s]) > 0 && (lock[pq_v(&H[s])] || part[pq_v(&H[s])] != s || pq_g(&H[s]) != ed[pq_v(&H[s])] - id[pq_v(&H[s])])) pq_pop(&H[s]);
            if (w[0] > maxside && pq_n(&H[0]) > 0) side = 0;
            else if (w[1] > maxside && pq_n(&H[1]) > 0) side = 1;
            else if (pq_n(&H[0]) > 0 && pq_n(&H[1]) > 0) side = pq_g(&H[0]) > pq_g(&H[1]) ? 0 : pq_g(&H[0]) < pq_g(&H[1]) ? 1 : (w[0] >= w[1] ? 0 : 1);
            else if (pq_n(&H[0]) > 0) side = 0;
            else if (pq_n(&H[1]) > 0) side = 1;
            if (side < 0) break;
            const int v = pq_v(&H[side]), gain = pq_g(&H[side]);
            pq_pop(&H[side]);
            if (w[1 - side] + g->vw[v] > maxside && w[side] <= maxside) { lock[v] = 1; continue; }     /* (would unbalance: this vertex stays) */
            part[v] = (char)(1 - side); lock[v] = 1;
            w[side] -= g->vw[v]; w[1 - side] += g->vw[v];
            cur -= gain;
            { const int t = ed[v]; ed[v] = id[v]; id[v] = t; }
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
                const int u = g->adj[p], ww = g->ew[p];
                if (part[u] == side) { ed[u] += ww; id[u] -= ww; } else { ed[u] -= ww; id[u] += ww; }
                if (!lock[u] && ed[u] > 0) pq_push(&H[(int)part[u]], ed[u] - id[u], u);
            }
            moved[nm++] = v;
            const long imb = labs(w[0] - w[1]);
            const int ok = w[0] <= maxside && w[1] <= maxside;
            if (ok && (cur < best || (cur == best && imb < bestimb))) { best = cur; bestimb = imb; bestk = nm; bad = 0; }
            else bad++;
        }
        /* back to the best point of the pass */
        for (int k = nm - 1; k >= bestk; k--) {
            const int v = moved[k], side = part[v];
            part[v] = (char)(1 - side);
            w[side] -= g->vw[v]; w[1 - side] += g->vw[v];
            { const int t = ed[v]; ed[v] = id[v]; id[v] = t; }
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
                const int u = g->adj[p], ww = g->ew[p];
                if (part[u] == side) { ed[u] += ww; id[u] -= ww; } else { ed[u] -= ww; id[u] += ww; }
            }
        }
        if (bestk == 0) break;                      /* (no move kept) */
        const int improved = best < cut || !ok0;
        cut = best;
        if (!improved) break;
    }
    free(ed); free(id); free(moved); free(lock); pq_free(&H[0]); pq_free(&H[1]);
    return cut;
}

static void vertex_cover(const Gr *g, const char *part, char *sep);
static void node_refine(const Gr *g, char *st, int npass);
/* a separator of g: part[] = 0, 1 (the sides) or 2 (the separator). 0: done */
static int bisect(const Gr *g0, char *part0, int mode) {
    enum { MAXLEV = 48, CO = 120 };
    Gr lev[MAXLEV]; int *cmaps[MAXLEV]; int nl = 0;
    const Gr *g = g0;
    while (g->n > CO && nl < MAXLEV) {
        int *cmap = malloc(sizeof(int) * (g->n + 1));
        if (!cmap) break;
        long maxvw = (long)(1.5 * (double)g->tvw / CO); if (maxvw < 2) maxvw = 2;
        if (coarsen(g, &lev[nl], cmap, maxvw)) { free(cmap); break; }
        if (lev[nl].n > 0.95 * g->n) { gr_free(&lev[nl]); free(cmap); break; }       /* (nothing left to match) */
        cmaps[nl] = cmap; g = &lev[nl]; nl++;
        if (nd_over()) { g_nd_abort = 1; break; }
    }
    if (g_nd_abort) { for (int k = 0; k < nl; k++) { gr_free(&lev[k]); free(cmaps[k]); } return 1; }   /* (past the time budget) */
    if (getenv("BRISK_NDDBG") && atoi(getenv("BRISK_NDDBG")) >= 9) { fprintf(stderr, "      [bisect n %d: levels", g0->n); for (int k = 0; k < nl; k++) fprintf(stderr, " %d", lev[k].n); fprintf(stderr, "]\n"); }
    /* the coarsest graph: regions grown from several seeds */
    const int n = g->n;
    const long maxside = (long)(0.6 * (double)g0->tvw) + 1;
    char *part = malloc((size_t)n + 1), *bestp = malloc((size_t)n + 1);
    int *queue = malloc(sizeof(int) * (n + 1));
    long bestcut = -1;
    const int ntry = n <= 2 ? 1 : 10;
    for (int t = 0; t < ntry; t++) {
        if (t > 0 && nd_over()) { g_nd_abort = 1; break; }
        memset(part, 1, (size_t)n);
        long wa = 0; int qh = 0, qt = 0, next = 0;
        const int seed = (int)(nd_rand() % (unsigned)(n > 0 ? n : 1));
        queue[qt++] = seed; part[seed] = 0; wa += g->vw[seed];
        while (2 * wa < g->tvw) {
            if (qh == qt) {                           /* (another component) */
                while (next < n && part[next] == 0) next++;
                if (next >= n) break;
                part[next] = 0; wa += g->vw[next]; queue[qt++] = next; continue;
            }
            const int v = queue[qh++];
            for (int p = g->xadj[v]; p < g->xadj[v + 1] && 2 * wa < g->tvw; p++) {
                const int u = g->adj[p];
                if (part[u]) { part[u] = 0; wa += g->vw[u]; queue[qt++] = u; }
            }
        }
        long cut = refine(g, part, maxside, 8);
        if (mode == 1) {
            /* judged by the separator it gives (weight, and the heavier side) */
            char *sepc = malloc((size_t)n + 1);
            vertex_cover(g, part, sepc);
            for (int v = 0; v < n; v++) if (sepc[v]) part[v] = 2;
            free(sepc);
            node_refine(g, part, 6);
            long ww[3] = { 0, 0, 0 };
            for (int v = 0; v < n; v++) ww[(int)part[v]] += g->vw[v];
            const long big = ww[0] > ww[1] ? ww[0] : ww[1];
            cut = ww[2] * 100 + (ww[2] + 1) * (100 * big / (g->tvw > 0 ? g->tvw : 1) - 50);
            if (ww[0] == 0 || ww[1] == 0) cut = LONG_MAX / 2;
        }
        if (bestcut < 0 || cut < bestcut) { bestcut = cut; memcpy(bestp, part, (size_t)n); }
    }
    free(queue);
    char *cur = bestp;
    if (mode == 0) {
        /* the edge bisection refined at every level; its vertex cover, refined, at the end */
        for (int k = nl - 1; k >= 0; k--) {
            const Gr *f = k == 0 ? g0 : &lev[k - 1];
            char *pf = k == 0 ? part0 : malloc((size_t)f->n + 1);
            for (int v = 0; v < f->n; v++) pf[v] = cur[cmaps[k][v]];
            if (!g_nd_abort && nd_over()) g_nd_abort = 1;
            if (!g_nd_abort) refine(f, pf, maxside, 6);
            if (cur != bestp) free(cur);
            cur = pf;
        }
        if (nl == 0) memcpy(part0, bestp, (size_t)g0->n);
        /* (24 October 2026, the night) before the matching (most of the time of a dense cut): any cover of the cut edges has
         * at least (cut edges) / (the most cut edges at one vertex) vertices; at the first bisection, a bound that leaves no
         * hope stops the dissection (lp_setcover_8k, LP: 20 s for AMD's own flops) */
        if (g_nd_first && g_nd_target > 0) {
            long ce = 0, cmax = 0, w0 = 0, w1 = 0;
            for (int v = 0; v < g0->n; v++) {
                long c = 0;
                for (int p = g0->xadj[v]; p < g0->xadj[v + 1]; p++) c += part0[g0->adj[p]] != part0[v];
                ce += c; if (c > cmax) cmax = c;
                if (part0[v] == 0) w0++; else w1++;
            }
            g_nd_work += (double)g0->xadj[g0->n] + g0->n;
            ce /= 2;
            if (cmax > 0 && nd_first_hopeless((double)ce / (double)cmax, (double)(w0 < w1 ? w0 : w1))) {
                g_nd_abort = 1;
                for (int k = 0; k < nl; k++) { gr_free(&lev[k]); free(cmaps[k]); }
                free(bestp); free(part);
                return 1;
            }
        }
        char *sepc = malloc((size_t)g0->n + 1);
        vertex_cover(g0, part0, sepc);
        for (int v = 0; v < g0->n; v++) if (sepc[v]) part0[v] = 2;
        free(sepc);
        if (!g_nd_abort) node_refine(g0, part0, 8);
    } else {
        /* the separator of the coarsest bisection, projected (the members of a coarse separator
         * vertex are separator vertices) and refined at every level */
        for (int k = nl - 1; k >= 0; k--) {
            const Gr *f = k == 0 ? g0 : &lev[k - 1];
            char *pf = k == 0 ? part0 : malloc((size_t)f->n + 1);
            for (int v = 0; v < f->n; v++) pf[v] = cur[cmaps[k][v]];
            if (!g_nd_abort && nd_over()) g_nd_abort = 1;
            if (!g_nd_abort) node_refine(f, pf, 6);
            if (cur != bestp) free(cur);
            cur = pf;
        }
        if (nl == 0) memcpy(part0, bestp, (size_t)g0->n);
    }
    free(part); free(bestp);
    for (int k = 0; k < nl; k++) { gr_free(&lev[k]); free(cmaps[k]); }
    return g_nd_abort ? 1 : 0;
}

/* maximum matching in the bipartite graph of the cut edges (Kuhn's augmenting paths) */
static int aug(int a, const Gr *g, const char *part, const int *bidx, int *mb, int *seen, int stamp, const int *alist) {
    const int v = alist[a];
    /* (24 October 2026, the night: the edges each search goes over counted in the work: uncounted, the matching of a set
     * covering's dense cut, most of its dissection's time, ran past the budget: lp_setcover_8k, LP, 4.3e8 units counted in
     * 15.7 s) */
    for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
        g_nd_work += 1.0;
        const int u = g->adj[p];
        if (part[u] == part[v]) continue;
        const int b = bidx[u];
        if (seen[b] == stamp) continue;
        seen[b] = stamp;
        if (mb[b] < 0 || aug(mb[b], g, part, bidx, mb, seen, stamp, alist)) { mb[b] = a; return 1; }
    }
    return 0;
}
/* the separator of the bisection: sep[v] = 1 for the vertices of a minimum vertex cover of the cut edges */
static void vertex_cover(const Gr *g, const char *part, char *sep) {
    const int n = g->n;
    g_nd_work += 2.0 * g->xadj[n] + n;
    int *bidx = malloc(sizeof(int) * (n + 1)), *alist = malloc(sizeof(int) * (n + 1)), *blist = malloc(sizeof(int) * (n + 1));
    int na = 0, nb = 0;
    for (int v = 0; v < n; v++) {
        sep[v] = 0; bidx[v] = -1;
        int b = 0;
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) if (part[g->adj[p]] != part[v]) { b = 1; break; }
        if (!b) continue;
        if (part[v] == 0) { bidx[v] = na; alist[na++] = v; } else { bidx[v] = nb; blist[nb++] = v; }
    }
    int *mb = malloc(sizeof(int) * (nb + 1)), *seen = calloc((size_t)nb + 1, sizeof(int)), *ma = malloc(sizeof(int) * (na + 1));
    for (int b = 0; b < nb; b++) mb[b] = -1;
    if (na > 20000) {                               /* (deep recursion: the smaller side instead) */
        if (na <= nb) for (int a = 0; a < na; a++) sep[alist[a]] = 1; else for (int b = 0; b < nb; b++) sep[blist[b]] = 1;
        free(bidx); free(alist); free(blist); free(mb); free(seen); free(ma);
        return;
    }
    for (int a = 0; a < na; a++) {
        if ((a & 31) == 0 && nd_over()) {      /* (past the time budget: the smaller side) */
            g_nd_abort = 1;
            if (na <= nb) for (int a2 = 0; a2 < na; a2++) sep[alist[a2]] = 1; else for (int b = 0; b < nb; b++) sep[blist[b]] = 1;
            free(bidx); free(alist); free(blist); free(mb); free(seen); free(ma);
            return;
        }
        aug(a, g, part, bidx, mb, seen, a + 1, alist);
    }
    for (int a = 0; a < na; a++) ma[a] = -1;
    for (int b = 0; b < nb; b++) if (mb[b] >= 0) ma[mb[b]] = b;
    /* Z: reachable from the unmatched A vertices by alternating paths; cover = (A \ Z) + (B in Z) */
    char *za = calloc((size_t)na + 1, 1), *zb = calloc((size_t)nb + 1, 1);
    int *st = malloc(sizeof(int) * (na + 1)), ns = 0;
    for (int a = 0; a < na; a++) if (ma[a] < 0) { za[a] = 1; st[ns++] = a; }
    while (ns > 0) {
        const int a = st[--ns], v = alist[a];
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
            const int u = g->adj[p];
            if (part[u] == part[v]) continue;
            const int b = bidx[u];
            if (zb[b]) continue;
            zb[b] = 1;
            if (mb[b] >= 0 && !za[mb[b]]) { za[mb[b]] = 1; st[ns++] = mb[b]; }
        }
    }
    for (int a = 0; a < na; a++) if (!za[a]) sep[alist[a]] = 1;
    for (int b = 0; b < nb; b++) if (zb[b]) sep[blist[b]] = 1;
    free(bidx); free(alist); free(blist); free(mb); free(seen); free(ma); free(za); free(zb); free(st);
}

/* refinement of the vertex separator: st[v] = 0, 1 (the sides) or 2 (separator). A separator
 * vertex moves to a side and draws its neighbours of the other side into the separator; the
 * gain is its weight minus theirs. Moves by gain from a heap per side, moves of negative gain
 * allowed, back to the smallest separator of the pass. */
/* (sparsechol-dev) the gain from the weights of each vertex's neighbours on either side, pw[2 v + s], kept current by
 * nd_setst at every change of a state: the same numbers as the scan of the neighbours at the same moment (so the
 * same moves and the same order), for O(1) instead of O(degree) per gain (node_refine was 7 % of the LP pds-20).
 * The LP/SOCP branch made the same change (nbw_add; a DC power flow's normal equations 3 s of 4.7 in the scans);
 * at the merge of 19 October 2026 this version was kept, the same gains and orders as either. */
static inline int node_gain(const Gr *g, const int *pw, int v, int side) { return g->vw[v] - pw[2 * v + 1 - side]; }
static inline void nd_setst(const Gr *g, char *st, int *pw, int v, int s) {
    const int a = st[v], w = g->vw[v];
    if (a == s) return;
    st[v] = (char)s;
    for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
        int *q = pw + 2 * g->adj[p];
        if (a < 2) q[a] -= w;
        if (s < 2) q[s] += w;
    }
}
/* (21 October 2026) the one-sided refinement of a vertex separator (the moves of METIS's one-sided node refinement): in a pass the separator
 * vertices move into one side only, each drawing its neighbours of the other side into the separator, and a vertex drawn
 * in may move on in the same pass, so the separator can sweep across many layers; the passes alternate the sides, each
 * back to the smallest separator it passed (ties: the better balance); the end when two passes in a row gain nothing.
 * The two-sided refinement below locks a vertex once moved, so a separator shifts by about one layer a pass. Over 14 LP
 * normal equations 0.663 -> 0.623 of AMD's flops in the geometric mean (dfl001 7.8e8 -> 5.0e8, rail01 1.07e10 -> 9.1e9,
 * s250r10 1.21e9 -> 9.8e8), the order no slower. BRISK_NDREF=0: the two-sided refinement (experiments). */
static void node_refine1(const Gr *g, char *st, int niter) {
    const int n = g->n;
    g_nd_work += (double)(niter + 1) * (g->xadj[n] + n);
    ND_WORK((double)niter * (n + g->xadj[n]));
    int *pw = calloc(2 * (size_t)n + 2, sizeof(int));
    for (int v = 0; v < n; v++)
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { const int u = g->adj[p]; if (st[u] < 2) pw[2 * v + st[u]] += g->vw[u]; }
    int *mv = malloc(sizeof(int) * (n + 1)), *mp = malloc(sizeof(int) * (n + 2));
    int *pulled = NULL, npl = 0, plcap = 0;
    PQ H; pq_init(&H, n);
    long w[3] = { 0, 0, 0 };
    for (int v = 0; v < n; v++) w[(int)st[v]] += g->vw[v];
    const long maxside = (long)(0.6 * (double)g->tvw) + 1;
    int to = w[0] < w[1] ? 1 : 0, idle = 0;
    for (int pass = 0; pass < 2 * niter && idle < 2; pass++) {
        const int other = to; to = 1 - to;
        pq_clear(&H); npl = 0;
        int nsep = 0;
        for (int v = 0; v < n; v++) if (st[v] == 2) { pq_push(&H, g->vw[v] - pw[2 * v + other], v); nsep++; }
        const long init = w[2];
        long best = w[2], bestdiff = labs(w[0] - w[1]);
        int nm = 0, bestk = 0;
        const int limit = nsep < 100 ? 3 * nsep + 10 : (3 * nsep < 300 ? 3 * nsep : 300);
        mp[0] = 0;
        while (pq_n(&H) > 0) {
            const int v = pq_v(&H);
            if (st[v] != 2) { pq_pop(&H); continue; }
            const int gn = g->vw[v] - pw[2 * v + other];
            if (gn != pq_g(&H)) { if (H.idx) pq_push(&H, gn, v); else { pq_pop(&H); pq_push(&H, gn, v); } continue; }
            if (w[to] + g->vw[v] > maxside) break;
            pq_pop(&H);
            nd_setst(g, st, pw, v, to); w[2] -= g->vw[v]; w[to] += g->vw[v];
            mv[nm] = v;
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
                const int u = g->adj[p];
                if (st[u] != other) continue;
                nd_setst(g, st, pw, u, 2); w[other] -= g->vw[u]; w[2] += g->vw[u];
                if (npl == plcap) { plcap = 2 * plcap + 256; pulled = realloc(pulled, sizeof(int) * plcap); }
                pulled[npl++] = u;
                pq_push(&H, g->vw[u] - pw[2 * u + other], u);
                /* its separator neighbours lose a neighbour on the other side: they move into `to` for more */
                for (int q = g->xadj[u]; q < g->xadj[u + 1]; q++) { const int x = g->adj[q]; if (st[x] == 2 && x != u) pq_push(&H, g->vw[x] - pw[2 * x + other], x); }
            }
            nm++; mp[nm] = npl;
            const long diff = labs(w[0] - w[1]);
            if (w[2] < best || (w[2] == best && diff < bestdiff)) { best = w[2]; bestdiff = diff; bestk = nm; }
            else if (nm - bestk > 3 * limit || (nm - bestk > limit && w[2] > 1.1 * best)) break;
        }
        for (int k = nm - 1; k >= bestk; k--) {
            const int v = mv[k];
            for (int q = mp[k]; q < mp[k + 1]; q++) { const int u = pulled[q]; nd_setst(g, st, pw, u, other); w[other] += g->vw[u]; w[2] -= g->vw[u]; }
            nd_setst(g, st, pw, v, 2); w[to] -= g->vw[v]; w[2] += g->vw[v];
        }
        idle = best < init ? 0 : idle + 1;
    }
    free(pw); free(mv); free(mp); free(pulled); pq_free(&H);
}
static void node_refine2(const Gr *g, char *st, int npass);
static void node_refine(const Gr *g, char *st, int npass) {
    static int r1 = -1; if (r1 < 0) { const char *e = getenv("BRISK_NDREF"); r1 = !(e && atoi(e) == 0); }
    if (r1) node_refine1(g, st, 10); else node_refine2(g, st, npass);
}
static void node_refine2(const Gr *g, char *st, int npass) {
    const int n = g->n;
    g_nd_work += (double)(npass + 1) * (g->xadj[n] + n);
    ND_WORK((double)npass * (n + g->xadj[n]));
    int *pw = calloc(2 * (size_t)n + 2, sizeof(int));
    for (int v = 0; v < n; v++)
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { const int u = g->adj[p]; if (st[u] < 2) pw[2 * v + st[u]] += g->vw[u]; }
    char *lock = malloc((size_t)n + 1);
    int *mv = malloc(sizeof(int) * (n + 1)), *ms = malloc(sizeof(int) * (n + 1)), *mp = malloc(sizeof(int) * (n + 2));
    int *pulled = NULL, npl = 0, plcap = 0;
    Heap H[2] = { { NULL, NULL, 0, 0 }, { NULL, NULL, 0, 0 } };
    long w[3] = { 0, 0, 0 };
    for (int v = 0; v < n; v++) w[(int)st[v]] += g->vw[v];
    const long maxside = (long)(0.62 * (double)g->tvw) + 1;
    for (int pass = 0; pass < npass; pass++) {
        memset(lock, 0, (size_t)n);
        H[0].n = H[1].n = 0; npl = 0;
        for (int v = 0; v < n; v++) if (st[v] == 2) { hpush(&H[0], node_gain(g, pw, v, 0), v); hpush(&H[1], node_gain(g, pw, v, 1), v); }
        long cur = w[2], best = w[2];
        int nm = 0, bestk = 0, bad = 0;
        const int limit = 60 + n / 100;
        while (bad < limit) {
            for (int s = 0; s < 2; s++)
                while (H[s].n > 0) {
                    const int v = H[s].v[0];
                    if (st[v] != 2 || lock[v]) { hpop(&H[s]); continue; }
                    const int gn = node_gain(g, pw, v, s);
                    if (gn != H[s].g[0]) { hpop(&H[s]); hpush(&H[s], gn, v); continue; }
                    break;
                }
            int side = -1;
            if (H[0].n > 0 && H[1].n > 0) side = H[0].g[0] > H[1].g[0] ? 0 : H[0].g[0] < H[1].g[0] ? 1 : (w[0] <= w[1] ? 0 : 1);
            else if (H[0].n > 0) side = 0;
            else if (H[1].n > 0) side = 1;
            if (side < 0) break;
            /* the lighter side when the better one is full */
            if (w[side] + g->vw[H[side].v[0]] > maxside) { if (H[1 - side].n > 0 && w[1 - side] + g->vw[H[1 - side].v[0]] <= maxside) side = 1 - side; else break; }
            const int v = H[side].v[0], gain = H[side].g[0];
            hpop(&H[side]);
            nd_setst(g, st, pw, v, side); lock[v] = 1; w[2] -= g->vw[v]; w[side] += g->vw[v];
            mv[nm] = v; ms[nm] = side; mp[nm] = npl;
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
                const int u = g->adj[p];
                if (st[u] != 1 - side) continue;
                nd_setst(g, st, pw, u, 2); w[1 - side] -= g->vw[u]; w[2] += g->vw[u];
                if (npl == plcap) { plcap = 2 * plcap + 256; pulled = realloc(pulled, sizeof(int) * plcap); }
                pulled[npl++] = u;
                if (!lock[u]) { hpush(&H[0], node_gain(g, pw, u, 0), u); hpush(&H[1], node_gain(g, pw, u, 1), u); }
                /* the separator neighbours of u no longer draw it in when they go to the side of v */
                for (int q = g->xadj[u]; q < g->xadj[u + 1]; q++) { const int x = g->adj[q]; if (st[x] == 2 && !lock[x] && x != u) hpush(&H[side], node_gain(g, pw, x, side), x); }
            }
            /* the separator neighbours of v can now go to its side for less */
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { const int u = g->adj[p]; if (st[u] == 2 && !lock[u]) hpush(&H[side], node_gain(g, pw, u, side), u); }
            nm++; mp[nm] = npl;
            cur -= gain;
            if (cur < best || (cur == best && labs(w[0] - w[1]) < 0)) { best = cur; bestk = nm; bad = 0; } else bad++;
        }
        for (int k = nm - 1; k >= bestk; k--) {
            const int v = mv[k], side = ms[k];
            for (int q = mp[k]; q < mp[k + 1]; q++) { const int u = pulled[q]; nd_setst(g, st, pw, u, 1 - side); w[1 - side] += g->vw[u]; w[2] -= g->vw[u]; }
            nd_setst(g, st, pw, v, 2); w[side] -= g->vw[v]; w[2] += g->vw[v];
        }
        if (bestk == 0) break;
    }
    free(pw); free(lock); free(mv); free(ms); free(mp); free(pulled); free(H[0].g); free(H[0].v); free(H[1].g); free(H[1].v);
}

/* AMD on the graph g; its vertices' original names in name[]; appended to perm at *np */
static void leaf_order(const Gr *g, const int *name, int *perm, int *np) {
    const int n = g->n;
    g_nd_work += 10.0 * (g->xadj[n] + n);
    int *P = malloc(sizeof(int) * (n + 1));
    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_defaults(Control);
    int st = n > 2 ? amd_order(n, g->xadj, g->adj, P, Control, Info) : AMD_INVALID;
    if (st != AMD_OK && st != AMD_OK_BUT_JUMBLED) for (int i = 0; i < n; i++) P[i] = i;
    for (int i = 0; i < n; i++) perm[(*np)++] = name[P[i]];
    free(P);
}

/* 18 October 2026 (LP/SOCP): 0 = no hub rule (a second trial of the LP method's analysis, sparsechol.c) */
static _Thread_local int g_nd_hubs = 1;
void nd_set_hubs(int on) { g_nd_hubs = on; }
void nd_set_target(double flops) { g_nd_target = flops; }
/* (20 October 2026, LP/SOCP) the number of bisections per level and the leaf size, set by a caller for one order (-1: the
 * defaults, 4 or 2 and 200, or BRISK_NDTRY / BRISK_NDLEAF); the cone solver's full system takes one try and leaves of
 * 5,000: the prism's order in 0.9 s instead of 10 s at 0.63 of AMD's work instead of 0.43 */
static int g_nd_try_set = -1, g_nd_leaf_set = -1;
void nd_set_params(int ntry, int leaf) { g_nd_try_set = ntry; g_nd_leaf_set = leaf; }
static void nd_rec(Gr *g, int *name, int *perm, int *np, int leaf, int depth) {
    const int n = g->n;
    ND_WORK(n);
    if (!g_nd_abort && nd_over()) g_nd_abort = 1;
    if (g_nd_abort) { leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    if (n <= leaf || depth > 60) { leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    /* hubs (vertices of many times the mean degree) go last without a search: they are in every
     * good separator of what they connect, and an edge bisection cannot afford to cut their edges */
    {
        static double hubf = -1;
        if (hubf < 0) hubf = getenv("BRISK_NDHUB") ? atof(getenv("BRISK_NDHUB")) : 5;
        if (!g_nd_hubs) goto nohubs;
        const double avg = (double)g->xadj[n] / n;
        int nh = 0;
        if (hubf > 0) for (int v = 0; v < n; v++) if (g->xadj[v + 1] - g->xadj[v] > hubf * avg && g->xadj[v + 1] - g->xadj[v] > 16) nh++;
        if (nh > 0 && nh <= n / 8) {
            int *loc = malloc(sizeof(int) * (n + 1)), *hubs = malloc(sizeof(int) * (nh + 1)), k = 0, h = 0, nz = 0;
            for (int v = 0; v < n; v++) { if (g->xadj[v + 1] - g->xadj[v] > hubf * avg && g->xadj[v + 1] - g->xadj[v] > 16) { loc[v] = -1; hubs[h++] = name[v]; } else { loc[v] = k++; nz += g->xadj[v + 1] - g->xadj[v]; } }
            Gr r; r.n = k; r.tvw = k;
            r.xadj = malloc(sizeof(int) * (k + 1)); r.adj = malloc(sizeof(int) * (nz + 1)); r.ew = malloc(sizeof(int) * (nz + 1)); r.vw = malloc(sizeof(int) * (k + 1));
            int *rname = malloc(sizeof(int) * (k + 1)), w = 0; k = 0;
            for (int v = 0; v < n; v++) if (loc[v] >= 0) {
                r.xadj[k] = w; r.vw[k] = 1; rname[k] = name[v]; k++;
                for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) if (loc[g->adj[p]] >= 0) { r.adj[w] = loc[g->adj[p]]; r.ew[w] = 1; w++; }
            }
            r.xadj[k] = w;
            if (getenv("BRISK_NDDBG") && depth <= atoi(getenv("BRISK_NDDBG"))) fprintf(stderr, "   [nd depth %d: n %d, %d hubs (degree > %.0f) last]\n", depth, n, nh, hubf * avg);
            free(loc); gr_free(g); free(name);
            if (g_nd_first) g_nd_top += nh;
            nd_rec(&r, rname, perm, np, leaf, depth + 1);
            for (int i = 0; i < nh; i++) perm[(*np)++] = hubs[i];
            free(hubs);
            return;
        }
    }
nohubs:;
    /* (21 October 2026) a graph of several connected components: each is ordered on its own, with no separator between
     * them (neos-2075418-temuka's normal equations without their hub rows: 147 components, which the bisections split
     * with empty separators at the cost of a multilevel bisection each) */
    {
        int *cc = malloc(sizeof(int) * (n + 1)), *qq = malloc(sizeof(int) * (n + 1)), ncc = 0;
        if (cc && qq) {
            for (int v = 0; v < n; v++) cc[v] = -1;
            for (int s0 = 0; s0 < n; s0++) if (cc[s0] < 0) {
                int h = 0, t = 0; qq[t++] = s0; cc[s0] = ncc;
                while (h < t) { const int v = qq[h++]; for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) if (cc[g->adj[p]] < 0) { cc[g->adj[p]] = ncc; qq[t++] = g->adj[p]; } }
                ncc++;
            }
        }
        if (ncc > 1) {
            int *csz = calloc((size_t)ncc + 1, sizeof(int)), *cnz = calloc((size_t)ncc + 1, sizeof(int)), *loc = malloc(sizeof(int) * (n + 1));
            for (int v = 0; v < n; v++) { loc[v] = csz[cc[v]]++; cnz[cc[v]] += g->xadj[v + 1] - g->xadj[v]; }
            Gr *sub = malloc(sizeof(Gr) * ncc); int **sname = malloc(sizeof(int *) * ncc), *fill = calloc((size_t)ncc + 1, sizeof(int)), *wpos = calloc((size_t)ncc + 1, sizeof(int));
            for (int c = 0; c < ncc; c++) {
                sub[c].n = csz[c]; sub[c].tvw = csz[c];
                sub[c].xadj = malloc(sizeof(int) * (csz[c] + 1)); sub[c].adj = malloc(sizeof(int) * (cnz[c] + 1)); sub[c].ew = malloc(sizeof(int) * (cnz[c] + 1)); sub[c].vw = malloc(sizeof(int) * (csz[c] + 1));
                sname[c] = malloc(sizeof(int) * (csz[c] + 1));
            }
            for (int v = 0; v < n; v++) {
                const int c = cc[v], k = fill[c]++;
                sub[c].xadj[k] = wpos[c]; sub[c].vw[k] = 1; sname[c][k] = name[v];
                for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { sub[c].adj[wpos[c]] = loc[g->adj[p]]; sub[c].ew[wpos[c]] = 1; wpos[c]++; }
            }
            for (int c = 0; c < ncc; c++) sub[c].xadj[csz[c]] = wpos[c];
            free(cc); free(qq); free(csz); free(cnz); free(loc); free(fill); free(wpos);
            gr_free(g); free(name);
            for (int c = 0; c < ncc; c++) nd_rec(&sub[c], sname[c], perm, np, leaf, depth + 1);
            free(sub); free(sname);
            return;
        }
        free(cc); free(qq);
    }
    char *part = malloc((size_t)n + 1), *sep = malloc((size_t)n + 1), *bpart = malloc((size_t)n + 1), *bsep = malloc((size_t)n + 1);
    if (!part || !sep || !bpart || !bsep) { free(part); free(sep); free(bpart); free(bsep); leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    /* several bisections (the matching and the seeds are random), the smallest separator kept */
    /* (the trial without hubs, 18 October 2026: two attempts; on staircase 50 x 500 they gave 9.7e9 flops in 2.5 s
     * against 1.11e10 in 5.6 s with four, one gave 1.35e10) */
    const int ntry = g_nd_try_set > 0 ? g_nd_try_set : getenv("BRISK_NDTRY") ? atoi(getenv("BRISK_NDTRY")) : g_nd_hubs ? 4 : 2;
    long bestscore = -1;
    for (int t = 0; t < ntry; t++) {
        if (t > 0 && nd_over()) { g_nd_abort = 1; break; }   /* (past the time budget: no more tries) */
        const double wt0 = g_nd_work;
        if (bisect(g, part, t & 1)) break;
        const double wtry = g_nd_work - wt0;
        long c[3] = { 0, 0, 0 };
        for (int v = 0; v < n; v++) c[(int)part[v]]++;
        if (c[0] == 0 || c[1] == 0) continue;
        /* (the separator, with a penalty for imbalance: the larger side decides the work below) */
        const long big = c[0] > c[1] ? c[0] : c[1];
        const long score = c[2] * 100 + (c[2] + 1) * (100 * big / (n > 0 ? n : 1) - 50);
        if (bestscore < 0 || score < bestscore) { bestscore = score; for (int v = 0; v < n; v++) { bsep[v] = part[v] == 2; bpart[v] = part[v] == 2 ? 0 : part[v]; } }
        /* (24 October 2026, the night) the first bisection's separator, with the hubs put last above it, is ordered last and
         * fills into a dense block: its factorization alone costs (s^3)/3 flops. When that passes the order to beat, no
         * dissection beats it: stop here (the caller keeps its own order). lp_setcover_8k (LP; AMD's factor 81 % full): the
         * dissection found AMD's flops after 15 s at one thread; ex9's dual form (36 % full) has small separators */
        if (nd_first_hopeless((double)c[2], (double)(c[0] < c[1] ? c[0] : c[1]))) { g_nd_abort = 1; break; }
        /* (24 October 2026, the night) another try only where the separator is worth it: the best separator's s vertices
         * fill into a dense block of s^3/3 flops a factorization, and a try costs about the work of the one just made. A
         * try made when the block costs less than 10 times that work rarely pays over the interior-point method's 20 to 30
         * factorizations (the dissection's counted work runs at about 5e8 units a second, the factorization at 1e11 flops
         * on four threads): deep in the recursion and on graphs with small separators one try, at the top of graphs with
         * large separators the four. On ten LP normal equations, ordering plus 25 factorizations at 1e11 flops a second:
         * 78.1 -> 68.4 s (mzzv11 1.45 -> 0.71, triptim1 3.52 -> 2.20, rail01 3.76 -> 2.68, rail02 12.3 -> 9.7; the
         * orderings 1.2 -> 0.4, 2.0 -> 0.6, 1.5 -> 0.4, 4.1 -> 1.2 s, the flops up 1 to 13 %). BRISK_NDTRYC: the factor
         * (negative: every try) */
        {
            static double tc = -2; if (tc < -1) { const char *e = getenv("BRISK_NDTRYC"); tc = e ? atof(e) : 10.0; }
            if (tc >= 0 && t + 1 < ntry) {
                long bs = 0; for (int v = 0; v < n; v++) bs += bsep[v];
                const double sb = (double)bs;
                if (sb * sb * sb / 3.0 < tc * wtry) break;
            }
        }
    }
    g_nd_first = 0;
    if (g_nd_abort) { free(part); free(sep); free(bpart); free(bsep); leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    if (bestscore < 0) { free(part); free(sep); free(bpart); free(bsep); leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    memcpy(part, bpart, (size_t)n); memcpy(sep, bsep, (size_t)n);
    free(bpart); free(bsep);
    int cnt[3] = { 0, 0, 0 };
    for (int v = 0; v < n; v++) cnt[sep[v] ? 2 : (int)part[v]]++;
    if (getenv("BRISK_NDDBG") && depth <= atoi(getenv("BRISK_NDDBG"))) fprintf(stderr, "   [nd depth %d: n %d -> %d + %d, separator %d]\n", depth, n, cnt[0], cnt[1], cnt[2]);
    /* a separator that does not separate, or is no smaller than what it leaves: this part by AMD */
    if (cnt[0] == 0 || cnt[1] == 0 || cnt[2] > (cnt[0] < cnt[1] ? cnt[0] : cnt[1])) { free(part); free(sep); leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    int *loc = malloc(sizeof(int) * (n + 1));
    Gr sub[2]; int *sname[2];
    for (int s = 0; s < 2; s++) {
        int k = 0, nz = 0;
        for (int v = 0; v < n; v++) if (!sep[v] && part[v] == s) { loc[v] = k++; nz += g->xadj[v + 1] - g->xadj[v]; }
        sub[s].n = k; sub[s].tvw = k;
        sub[s].xadj = malloc(sizeof(int) * (k + 1)); sub[s].adj = malloc(sizeof(int) * (nz + 1)); sub[s].ew = malloc(sizeof(int) * (nz + 1)); sub[s].vw = malloc(sizeof(int) * (k + 1));
        sname[s] = malloc(sizeof(int) * (k + 1));
        int w = 0; k = 0;
        for (int v = 0; v < n; v++) if (!sep[v] && part[v] == s) {
            sub[s].xadj[k] = w; sub[s].vw[k] = 1; sname[s][k] = name[v]; k++;
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { const int u = g->adj[p]; if (!sep[u] && part[u] == s) { sub[s].adj[w] = u; sub[s].ew[w] = 1; w++; } }
        }
        sub[s].xadj[k] = w;
        for (int p = 0; p < w; p++) sub[s].adj[p] = loc[sub[s].adj[p]];
    }
    int *sepname = malloc(sizeof(int) * (cnt[2] + 1)), ns = 0;
    for (int v = 0; v < n; v++) if (sep[v]) sepname[ns++] = name[v];
    free(part); free(sep); free(loc);
    gr_free(g); free(name);                          /* (this level's graph is no longer needed) */
    for (int s = 0; s < 2; s++) nd_rec(&sub[s], sname[s], perm, np, leaf, depth + 1);
    for (int i = 0; i < ns; i++) perm[(*np)++] = sepname[i];
    free(sepname);
}

int nd_order(int n, const int *xadj, const int *adjncy, int *perm) {
    if (n <= 0) return 1;
    nd_seed = 12345u;
    g_nd_first = 1; g_nd_top = 0;
    Gr g; g.n = n; g.tvw = n;
    const int nz = xadj[n];
    g.xadj = malloc(sizeof(int) * (n + 1)); g.adj = malloc(sizeof(int) * (nz + 1)); g.ew = malloc(sizeof(int) * (nz + 1)); g.vw = malloc(sizeof(int) * (n + 1));
    int *name = malloc(sizeof(int) * (n + 1));
    if (!g.xadj || !g.adj || !g.ew || !g.vw || !name) { gr_free(&g); free(name); return 1; }
    int w = 0;
    for (int v = 0; v < n; v++) {
        g.xadj[v] = w; g.vw[v] = 1; name[v] = v;
        for (int p = xadj[v]; p < xadj[v + 1]; p++) if (adjncy[p] != v) { g.adj[w] = adjncy[p]; g.ew[w] = 1; w++; }
    }
    g.xadj[n] = w;
    int np = 0;
    nd_rec(&g, name, perm, &np, g_nd_leaf_set > 0 ? g_nd_leaf_set : getenv("BRISK_NDLEAF") ? atoi(getenv("BRISK_NDLEAF")) : 200, 0);
    /* (23 October 2026, a test: BRISK_NDPARTIAL=1) past the budget the rest is ordered by AMD (leaf_order): that order, the
     * top levels' separators and AMD below, is a valid permutation, offered to the caller, who keeps it when its factor is
     * cheaper than AMD's; without it the dissection's work up to the budget is thrown away */
    { static int part = -1; if (part < 0) { const char *e = getenv("BRISK_NDPARTIAL"); part = e && atoi(e) == 1; }
      if (part && np == n) return 0; }
    return np == n && !g_nd_abort ? 0 : 1;
}
