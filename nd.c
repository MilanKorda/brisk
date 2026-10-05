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
#include "amd/amd.h"

typedef struct { int n; int *xadj, *adj, *ew, *vw; long tvw; } Gr;

static unsigned int nd_seed;
static unsigned int nd_rand(void) { nd_seed = nd_seed * 1664525u + 1013904223u; return nd_seed >> 8; }
static void gr_free(Gr *g) { free(g->xadj); free(g->adj); free(g->ew); free(g->vw); }

/* one level of coarsening; cmap[v]: the coarse vertex of v. 0: done, 1: no memory */
static int coarsen(const Gr *g, Gr *c, int *cmap, long maxvw) {
    const int n = g->n;
    int *match = malloc(sizeof(int) * (n + 1)), *ord = malloc(sizeof(int) * (n + 1));
    if (!match || !ord) { free(match); free(ord); return 1; }
    for (int i = 0; i < n; i++) { match[i] = -1; ord[i] = i; }
    for (int i = n - 1; i > 0; i--) { const int j = (int)(nd_rand() % (unsigned)(i + 1)), t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
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

/* refinement of the bisection part[] (0/1); returns the cut */
static long refine(const Gr *g, char *part, long maxside, int npass) {
    const int n = g->n;
    int *ed = malloc(sizeof(int) * (n + 1)), *id = malloc(sizeof(int) * (n + 1)), *moved = malloc(sizeof(int) * (n + 1));
    char *lock = malloc((size_t)n + 1);
    Heap H[2] = { { NULL, NULL, 0, 0 }, { NULL, NULL, 0, 0 } };
    long cut = 0, w[2] = { 0, 0 };
    for (int v = 0; v < n; v++) {
        int e = 0, i = 0;
        for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { if (part[g->adj[p]] != part[v]) e += g->ew[p]; else i += g->ew[p]; }
        ed[v] = e; id[v] = i; cut += e; w[(int)part[v]] += g->vw[v];
    }
    cut /= 2;
    for (int pass = 0; pass < npass; pass++) {
        memset(lock, 0, (size_t)n);
        H[0].n = H[1].n = 0;
        for (int v = 0; v < n; v++) if (ed[v] > 0 || g->xadj[v + 1] == g->xadj[v]) hpush(&H[(int)part[v]], ed[v] - id[v], v);
        const int ok0 = w[0] <= maxside && w[1] <= maxside;
        long best = ok0 ? cut : LONG_MAX, cur = cut, bestimb = labs(w[0] - w[1]);
        int nm = 0, bestk = 0, bad = 0;
        const int limit = 100 + n / 50;
        while (bad < limit) {
            /* the side to move from: the one over its limit, else the better gain */
            int side = -1;
            for (int s = 0; s < 2; s++) while (H[s].n > 0 && (lock[H[s].v[0]] || part[H[s].v[0]] != s || H[s].g[0] != ed[H[s].v[0]] - id[H[s].v[0]])) hpop(&H[s]);
            if (w[0] > maxside && H[0].n > 0) side = 0;
            else if (w[1] > maxside && H[1].n > 0) side = 1;
            else if (H[0].n > 0 && H[1].n > 0) side = H[0].g[0] > H[1].g[0] ? 0 : H[0].g[0] < H[1].g[0] ? 1 : (w[0] >= w[1] ? 0 : 1);
            else if (H[0].n > 0) side = 0;
            else if (H[1].n > 0) side = 1;
            if (side < 0) break;
            const int v = H[side].v[0], gain = H[side].g[0];
            hpop(&H[side]);
            if (w[1 - side] + g->vw[v] > maxside && w[side] <= maxside) { lock[v] = 1; continue; }     /* (would unbalance: this vertex stays) */
            part[v] = (char)(1 - side); lock[v] = 1;
            w[side] -= g->vw[v]; w[1 - side] += g->vw[v];
            cur -= gain;
            { const int t = ed[v]; ed[v] = id[v]; id[v] = t; }
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
                const int u = g->adj[p], ww = g->ew[p];
                if (part[u] == side) { ed[u] += ww; id[u] -= ww; } else { ed[u] -= ww; id[u] += ww; }
                if (!lock[u] && ed[u] > 0) hpush(&H[(int)part[u]], ed[u] - id[u], u);
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
    free(ed); free(id); free(moved); free(lock); free(H[0].g); free(H[0].v); free(H[1].g); free(H[1].v);
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
    }
    if (getenv("BRISK_NDDBG") && atoi(getenv("BRISK_NDDBG")) >= 9) { fprintf(stderr, "      [bisect n %d: levels", g0->n); for (int k = 0; k < nl; k++) fprintf(stderr, " %d", lev[k].n); fprintf(stderr, "]\n"); }
    /* the coarsest graph: regions grown from several seeds */
    const int n = g->n;
    const long maxside = (long)(0.6 * (double)g0->tvw) + 1;
    char *part = malloc((size_t)n + 1), *bestp = malloc((size_t)n + 1);
    int *queue = malloc(sizeof(int) * (n + 1));
    long bestcut = -1;
    const int ntry = n <= 2 ? 1 : 10;
    for (int t = 0; t < ntry; t++) {
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
            refine(f, pf, maxside, 6);
            if (cur != bestp) free(cur);
            cur = pf;
        }
        if (nl == 0) memcpy(part0, bestp, (size_t)g0->n);
        char *sepc = malloc((size_t)g0->n + 1);
        vertex_cover(g0, part0, sepc);
        for (int v = 0; v < g0->n; v++) if (sepc[v]) part0[v] = 2;
        free(sepc);
        node_refine(g0, part0, 8);
    } else {
        /* the separator of the coarsest bisection, projected (the members of a coarse separator
         * vertex are separator vertices) and refined at every level */
        for (int k = nl - 1; k >= 0; k--) {
            const Gr *f = k == 0 ? g0 : &lev[k - 1];
            char *pf = k == 0 ? part0 : malloc((size_t)f->n + 1);
            for (int v = 0; v < f->n; v++) pf[v] = cur[cmaps[k][v]];
            node_refine(f, pf, 6);
            if (cur != bestp) free(cur);
            cur = pf;
        }
        if (nl == 0) memcpy(part0, bestp, (size_t)g0->n);
    }
    free(part); free(bestp);
    for (int k = 0; k < nl; k++) { gr_free(&lev[k]); free(cmaps[k]); }
    return 0;
}

/* maximum matching in the bipartite graph of the cut edges (Kuhn's augmenting paths) */
static int aug(int a, const Gr *g, const char *part, const int *bidx, int *mb, int *seen, int stamp, const int *alist) {
    const int v = alist[a];
    for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
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
    for (int a = 0; a < na; a++) aug(a, g, part, bidx, mb, seen, a + 1, alist);
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
static int node_gain(const Gr *g, const char *st, int v, int side) {
    int pull = 0;
    for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) if (st[g->adj[p]] == 1 - side) pull += g->vw[g->adj[p]];
    return g->vw[v] - pull;
}
static void node_refine(const Gr *g, char *st, int npass) {
    const int n = g->n;
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
        for (int v = 0; v < n; v++) if (st[v] == 2) { hpush(&H[0], node_gain(g, st, v, 0), v); hpush(&H[1], node_gain(g, st, v, 1), v); }
        long cur = w[2], best = w[2];
        int nm = 0, bestk = 0, bad = 0;
        const int limit = 60 + n / 100;
        while (bad < limit) {
            for (int s = 0; s < 2; s++)
                while (H[s].n > 0) {
                    const int v = H[s].v[0];
                    if (st[v] != 2 || lock[v]) { hpop(&H[s]); continue; }
                    const int gn = node_gain(g, st, v, s);
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
            st[v] = (char)side; lock[v] = 1; w[2] -= g->vw[v]; w[side] += g->vw[v];
            mv[nm] = v; ms[nm] = side; mp[nm] = npl;
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) {
                const int u = g->adj[p];
                if (st[u] != 1 - side) continue;
                st[u] = 2; w[1 - side] -= g->vw[u]; w[2] += g->vw[u];
                if (npl == plcap) { plcap = 2 * plcap + 256; pulled = realloc(pulled, sizeof(int) * plcap); }
                pulled[npl++] = u;
                if (!lock[u]) { hpush(&H[0], node_gain(g, st, u, 0), u); hpush(&H[1], node_gain(g, st, u, 1), u); }
                /* the separator neighbours of u no longer draw it in when they go to the side of v */
                for (int q = g->xadj[u]; q < g->xadj[u + 1]; q++) { const int x = g->adj[q]; if (st[x] == 2 && !lock[x] && x != u) hpush(&H[side], node_gain(g, st, x, side), x); }
            }
            /* the separator neighbours of v can now go to its side for less */
            for (int p = g->xadj[v]; p < g->xadj[v + 1]; p++) { const int u = g->adj[p]; if (st[u] == 2 && !lock[u]) hpush(&H[side], node_gain(g, st, u, side), u); }
            nm++; mp[nm] = npl;
            cur -= gain;
            if (cur < best || (cur == best && labs(w[0] - w[1]) < 0)) { best = cur; bestk = nm; bad = 0; } else bad++;
        }
        for (int k = nm - 1; k >= bestk; k--) {
            const int v = mv[k], side = ms[k];
            for (int q = mp[k]; q < mp[k + 1]; q++) { const int u = pulled[q]; st[u] = (char)(1 - side); w[1 - side] += g->vw[u]; w[2] -= g->vw[u]; }
            st[v] = 2; w[side] -= g->vw[v]; w[2] += g->vw[v];
        }
        if (bestk == 0) break;
    }
    free(lock); free(mv); free(ms); free(mp); free(pulled); free(H[0].g); free(H[0].v); free(H[1].g); free(H[1].v);
}

/* AMD on the graph g; its vertices' original names in name[]; appended to perm at *np */
static void leaf_order(const Gr *g, const int *name, int *perm, int *np) {
    const int n = g->n;
    int *P = malloc(sizeof(int) * (n + 1));
    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_defaults(Control);
    int st = n > 2 ? amd_order(n, g->xadj, g->adj, P, Control, Info) : AMD_INVALID;
    if (st != AMD_OK && st != AMD_OK_BUT_JUMBLED) for (int i = 0; i < n; i++) P[i] = i;
    for (int i = 0; i < n; i++) perm[(*np)++] = name[P[i]];
    free(P);
}

static void nd_rec(Gr *g, int *name, int *perm, int *np, int leaf, int depth) {
    const int n = g->n;
    if (n <= leaf || depth > 60) { leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    /* hubs (vertices of many times the mean degree) go last without a search: they are in every
     * good separator of what they connect, and an edge bisection cannot afford to cut their edges */
    {
        static double hubf = -1;
        if (hubf < 0) hubf = getenv("BRISK_NDHUB") ? atof(getenv("BRISK_NDHUB")) : 5;
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
            nd_rec(&r, rname, perm, np, leaf, depth + 1);
            for (int i = 0; i < nh; i++) perm[(*np)++] = hubs[i];
            free(hubs);
            return;
        }
    }
    char *part = malloc((size_t)n + 1), *sep = malloc((size_t)n + 1), *bpart = malloc((size_t)n + 1), *bsep = malloc((size_t)n + 1);
    if (!part || !sep || !bpart || !bsep) { free(part); free(sep); free(bpart); free(bsep); leaf_order(g, name, perm, np); gr_free(g); free(name); return; }
    /* several bisections (the matching and the seeds are random), the smallest separator kept */
    const int ntry = getenv("BRISK_NDTRY") ? atoi(getenv("BRISK_NDTRY")) : 4;
    long bestscore = -1;
    for (int t = 0; t < ntry; t++) {
        if (bisect(g, part, t & 1)) break;
        long c[3] = { 0, 0, 0 };
        for (int v = 0; v < n; v++) c[(int)part[v]]++;
        if (c[0] == 0 || c[1] == 0) continue;
        /* (the separator, with a penalty for imbalance: the larger side decides the work below) */
        const long big = c[0] > c[1] ? c[0] : c[1];
        const long score = c[2] * 100 + (c[2] + 1) * (100 * big / (n > 0 ? n : 1) - 50);
        if (bestscore < 0 || score < bestscore) { bestscore = score; for (int v = 0; v < n; v++) { bsep[v] = part[v] == 2; bpart[v] = part[v] == 2 ? 0 : part[v]; } }
    }
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
    nd_rec(&g, name, perm, &np, getenv("BRISK_NDLEAF") ? atoi(getenv("BRISK_NDLEAF")) : 200, 0);
    return np == n ? 0 : 1;
}
