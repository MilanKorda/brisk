/*
 * Chordal (clique) decomposition.
 *
 * A large SDP block whose aggregate sparsity pattern (the union of the patterns of C and
 * all A_i) is sparse can be replaced by one block per clique of a chordal extension of
 * that pattern, plus equality constraints tying overlapping entries together: X is
 * positive semidefinite completable exactly when every clique submatrix is.
 *
 * On the 793-bus AC-OPF relaxation the moment block is 1781 x 1781 but only 0.35% dense,
 * and a min-degree extension gives cliques of at most 23, so the per-iteration dense work
 * drops from 4 n^3 = 2.3e10 flops to sum 4 c^3 = 2.8e6. The price is the overlap
 * equalities, which enlarge m; clique merging trades the two against each other and the
 * gates below refuse the conversion when it would not pay.
 *
 * The rewritten problem is an ordinary SDP with many small blocks, which the rest of the
 * solver already handles well (sparse envelope Schur complement, small-block kernels).
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* clique-tree construction helpers (file scope: nested functions are a GCC extension) */
typedef struct { int a, b, w; } Edge;
static int cmp_edge(const void *A2, const void *B2) {
    const Edge *x = A2, *y2 = B2;
    if (x->a != y2->a) return x->a - y2->a;
    return x->b - y2->b;
}
static int cmp_w(const void *A2, const void *B2) { return ((const Edge *)B2)->w - ((const Edge *)A2)->w; }
static int find2(int *u2, int x) { while (u2[x] != x) { u2[x] = u2[u2[x]]; x = u2[x]; } return x; }

typedef struct { int *v, n, cap, alive, parent, rep; } Clique;

/* 4.42: every allocation of this file is checked (iter_cost dereferenced a failed realloc on
 * dense case13659); a failure ends the solve with a message instead of a segfault */
static void *ck_malloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (chordal conversion, %zu bytes)\n", n); exit(1); } return p; }
static void *ck_calloc(size_t a, size_t b) { void *p = calloc(a ? a : 1, b ? b : 1); if (!p) { fprintf(stderr, "brisk: out of memory (chordal conversion, %zu bytes)\n", a * b); exit(1); } return p; }
static void *ck_realloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (chordal conversion, %zu bytes)\n", n); exit(1); } return p; }
#define malloc(n) ck_malloc(n)
#define calloc(a, b) ck_calloc(a, b)
#define realloc(q, n) ck_realloc(q, n)
static void *cm(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

static int cl_has(const Clique *c, int x) {
    int lo = 0, hi = c->n - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (c->v[mid] == x) return mid; if (c->v[mid] < x) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

/* adjacency with dynamic arrays */
typedef struct { int *a, n, cap; } Adj;
static void adj_add(Adj *A, int x) {
    for (int i = 0; i < A->n; i++) if (A->a[i] == x) return;
    if (A->n == A->cap) { A->cap = A->cap ? 2 * A->cap : 8; A->a = realloc(A->a, sizeof(int) * A->cap); }
    A->a[A->n++] = x;
}

/* Convert one SDP block. Returns the number of cliques, or 0 if the block is left alone.
 * On success: *cliq receives the cliques, *repof[v] the clique index that owns node v. */
static int chordal_cliques(const Block *B, const Params *par, Clique **cliq_out, int **repof_out,
                           int *nmerge_out, int **pos_out) {
    const int n = B->n;
    Adj *adj = cm(sizeof(Adj) * n);
    size_t pat = 0;
    for (int t = -1; t < B->ncon; t++) {
        const SpSym *S = t < 0 ? &B->C : &B->A[t];
        for (int q = 0; q < S->nnz; q++) {
            int i = S->row[q], j = S->col[q];
            if (i == j) continue;
            adj_add(&adj[i], j);
            adj_add(&adj[j], i);
            pat++;
        }
    }
    double dens = (double)pat / (0.5 * n * (double)(n + 1));
    /* Constraint supports as cliques (correlative sparsity). Without them a constraint
     * whose entries form a star (the power balance of a bus: V_i V_j for every neighbour
     * j) is split over several cliques, and every clique it touches couples it to all of
     * that clique's constraints in the Schur complement: on the 793-bus AC-OPF block each
     * constraint touched ~9 cliques and the Schur pattern was 13% dense. With the support
     * of each constraint inside one clique, every constraint lives in exactly one block and
     * the Schur complement follows the clique tree. */
    if (par->chordal_sup > 0) {
        int *sup = cm(sizeof(int) * n);
        char *in = cm(n);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            int ns2 = 0;
            for (int q = 0; q < S->nnz && ns2 <= par->chordal_sup; q++) {
                int a2[2] = { S->row[q], S->col[q] };
                for (int e = 0; e < 2; e++) if (!in[a2[e]]) { in[a2[e]] = 1; sup[ns2++] = a2[e]; if (ns2 > par->chordal_sup) break; }
            }
            if (ns2 <= par->chordal_sup)
                for (int a = 0; a < ns2; a++)
                    for (int b3 = a + 1; b3 < ns2; b3++) { adj_add(&adj[sup[a]], sup[b3]); adj_add(&adj[sup[b3]], sup[a]); }
            for (int a = 0; a < ns2; a++) in[sup[a]] = 0;
        }
        free(sup); free(in);
    }
    if (dens > par->chordal_density) {
        for (int i = 0; i < n; i++) free(adj[i].a);
        free(adj);
        return 0;
    }
    /* ---- min-degree elimination with fill -> chordal extension and cliques */
    char *gone = cm(n);
    int *pos = cm(sizeof(int) * n), *deg = cm(sizeof(int) * n);
    Clique *cl = cm(sizeof(Clique) * n);
    int *buf = cm(sizeof(int) * n);
    char *mark = cm(n);
    int maxc = 0;
    static int mfill = -1;
    if (mfill < 0) { const char *e = getenv("BRISK_CH_MINFILL"); mfill = e ? atoi(e) : 3; }
    int *cand = cm(sizeof(int) * n), *mk2 = cm(sizeof(int) * n);
    for (int i = 0; i < n; i++) mk2[i] = -1;
    for (int step = 0; step < n; step++) {
        int best = -1, bd = 1 << 30, nc = 0;
        for (int i = 0; i < n; i++) {
            if (gone[i]) continue;
            int d = 0;
            for (int q = 0; q < adj[i].n; q++) if (!gone[adj[i].a[q]]) d++;
            deg[i] = d;
            if (d < bd) { bd = d; best = i; }
        }
        /* candidates: degree within mfill_slack of the minimum (at most 64) */
        for (int i = 0; i < n && nc < 64; i++) if (!gone[i] && deg[i] <= bd + (mfill > 1 ? mfill - 1 : 0)) cand[nc++] = i;
        /* ties of the minimum degree broken by the fill they create (TSSOS's MF chordal
         * extension is minimum fill; plain minimum degree gave cliques up to 32 against
         * 26 on the 162-bus AC-OPF, 1.8x the Schur factorization) */
        if (mfill && nc > 1 && bd > 1) {
            long bf = -1;
            for (int c = 0; c < nc && bf != 0; c++) {
                const int v = cand[c];
                int nb2 = 0;
                for (int q = 0; q < adj[v].n; q++) if (!gone[adj[v].a[q]]) buf[nb2++] = adj[v].a[q];
                long fill = 0;
                for (int a = 0; a < nb2; a++) {
                    const int u = buf[a];
                    for (int q = 0; q < adj[u].n; q++) mk2[adj[u].a[q]] = step * 4 + 1 + (c & 1);   /* stamp */
                    const int stamp = step * 4 + 1 + (c & 1);
                    for (int b2 = a + 1; b2 < nb2; b2++) if (mk2[buf[b2]] != stamp) fill++;
                    for (int q = 0; q < adj[u].n; q++) mk2[adj[u].a[q]] = -1;
                    if (bf >= 0 && fill >= bf) break;
                }
                if (bf < 0 || fill < bf) { bf = fill; best = v; }
            }
        }
        if (best < 0) break;
        int nb = 0;
        for (int q = 0; q < adj[best].n; q++) if (!gone[adj[best].a[q]]) buf[nb++] = adj[best].a[q];
        if (nb + 1 > par->chordal_maxclique) {                 /* treewidth too large */
            for (int i = 0; i < n; i++) free(adj[i].a);
            for (int i = 0; i < n; i++) free(cl[i].v);
            free(adj); free(gone); free(pos); free(deg); free(cl); free(buf); free(mark); free(cand); free(mk2);
            return 0;
        }
        for (int a = 0; a < nb; a++)                            /* fill */
            for (int b2 = a + 1; b2 < nb; b2++) { adj_add(&adj[buf[a]], buf[b2]); adj_add(&adj[buf[b2]], buf[a]); }
        cl[step].v = malloc(sizeof(int) * (nb + 1));
        cl[step].v[0] = best;
        memcpy(cl[step].v + 1, buf, sizeof(int) * nb);
        cl[step].n = nb + 1;
        qsort(cl[step].v, cl[step].n, sizeof(int), cmp_int);
        cl[step].alive = 1;
        if (cl[step].n > maxc) maxc = cl[step].n;
        pos[best] = step;
        gone[best] = 1;
    }
    free(cand); free(mk2);
    const int ncl0 = n;
    /* ---- maximality: C_v \ {v} is contained in the clique of its earliest member */
    int *repv = cm(sizeof(int) * n);                            /* node -> clique index */
    for (int s = 0; s < ncl0; s++) { cl[s].rep = s; cl[s].parent = -1; }
    for (int s = 0; s < ncl0; s++) {
        int v = -1;
        for (int q = 0; q < cl[s].n; q++) if (pos[cl[s].v[q]] == s) { v = cl[s].v[q]; break; }
        if (v < 0) continue;
        repv[v] = s;
        int par_node = -1, bp = 1 << 30;
        for (int q = 0; q < cl[s].n; q++) {
            int w = cl[s].v[q];
            if (w == v) continue;
            if (pos[w] < bp) { bp = pos[w]; par_node = w; }
        }
        cl[s].parent = par_node >= 0 ? pos[par_node] : -1;
        /* C_v \ {v} is contained in C_p, so |C_v| - 1 == |C_p| means C_p is contained in
         * C_v: the *child* absorbs the parent (an earlier version walked the other way and
         * mapped nodes to cliques that did not contain them) */
        if (par_node >= 0 && cl[s].n - 1 == cl[pos[par_node]].n) {
            cl[pos[par_node]].alive = 0;
            cl[pos[par_node]].rep = s;
        }
    }
    /* flatten the absorber chains */
    for (int s = 0; s < ncl0; s++) {
        int r = s, guard = 0;
        while (!cl[r].alive && cl[r].rep != r && guard++ < ncl0) r = cl[r].rep;
        cl[s].rep = r;
    }
    for (int v = 0; v < n; v++) repv[v] = cl[repv[v]].rep;
    /* parents of live cliques */
    for (int s = 0; s < ncl0; s++)
        if (cl[s].alive && cl[s].parent >= 0) {
            int p = cl[cl[s].parent].rep;
            cl[s].parent = (p != s) ? p : -1;
        }
    /* ---- merge a child into its parent when the separator is large */
    int nmerge = 0;
    for (int s = ncl0 - 1; s >= 0; s--) {
        if (!cl[s].alive || cl[s].parent < 0) continue;
        int p = cl[s].parent;
        while (p >= 0 && !cl[p].alive) p = cl[p].parent;
        if (p < 0) { cl[s].parent = -1; continue; }
        cl[s].parent = p;
        int sep = 0;
        for (int q = 0; q < cl[s].n; q++) if (cl_has(&cl[p], cl[s].v[q]) >= 0) sep++;
        int uni = cl[s].n + cl[p].n - sep;
        int mn = cl[s].n < cl[p].n ? cl[s].n : cl[p].n;
        if (uni <= par->chordal_maxclique && sep >= par->chordal_merge * mn) {
            int *u = malloc(sizeof(int) * uni), k = 0;
            memset(mark, 0, n);
            for (int q = 0; q < cl[p].n; q++) { u[k++] = cl[p].v[q]; mark[cl[p].v[q]] = 1; }
            for (int q = 0; q < cl[s].n; q++) if (!mark[cl[s].v[q]]) u[k++] = cl[s].v[q];
            qsort(u, k, sizeof(int), cmp_int);
            free(cl[p].v);
            cl[p].v = u; cl[p].n = k;
            cl[s].alive = 0; cl[s].rep = p;
            for (int v = 0; v < n; v++) if (repv[v] == s) repv[v] = p;
            for (int t = 0; t < ncl0; t++) if (cl[t].alive && cl[t].parent == s) cl[t].parent = p;
            nmerge++;
        }
    }
    /* Rebuild the clique tree after merging as a junction tree: a maximum-weight spanning
     * forest over the live cliques with weights |C_i ∩ C_j|. This restores the running
     * intersection property, which the elimination-order parent rule loses once cliques
     * are merged (an earlier version produced far too few overlap equalities, silently
     * relaxing the problem).                                                          */
    {
        int nlive = 0;
        for (int s = 0; s < ncl0; s++) if (cl[s].alive) nlive++;
        int *lst = malloc(sizeof(int) * (nlive ? nlive : 1)), li = 0;
        for (int s = 0; s < ncl0; s++) if (cl[s].alive) { lst[li] = s; cl[s].parent = -1; li++; }
        int *where = malloc(sizeof(int) * ncl0);
        for (int s = 0; s < ncl0; s++) where[s] = -1;
        for (int i = 0; i < nlive; i++) where[lst[i]] = i;
        /* node -> live cliques containing it */
        int *cnt2 = calloc(n + 1, sizeof(int));
        for (int i = 0; i < nlive; i++)
            for (int q = 0; q < cl[lst[i]].n; q++) cnt2[cl[lst[i]].v[q]]++;
        int *ptr = malloc(sizeof(int) * (n + 1));
        int tot = 0;
        for (int v2 = 0; v2 < n; v2++) { ptr[v2] = tot; tot += cnt2[v2]; }
        ptr[n] = tot;
        int *lists = malloc(sizeof(int) * (tot ? tot : 1));
        int *fill2 = calloc(n + 1, sizeof(int));
        for (int i = 0; i < nlive; i++)
            for (int q = 0; q < cl[lst[i]].n; q++) {
                int v2 = cl[lst[i]].v[q];
                lists[ptr[v2] + fill2[v2]++] = i;
            }
        /* candidate edges with weights */
        size_t ecap = 1024, ne2 = 0;
        Edge *E2 = malloc(sizeof(Edge) * ecap);
        for (int v2 = 0; v2 < n; v2++)
            for (int x = ptr[v2]; x < ptr[v2 + 1]; x++)
                for (int y2 = x + 1; y2 < ptr[v2 + 1]; y2++) {
                    if (ne2 == ecap) { ecap *= 2; E2 = realloc(E2, sizeof(Edge) * ecap); }
                    E2[ne2].a = lists[x]; E2[ne2].b = lists[y2]; E2[ne2].w = 1;
                    ne2++;
                }
        /* accumulate duplicate pairs */
        for (size_t q = 0; q < ne2; q++) if (E2[q].a > E2[q].b) { int t2 = E2[q].a; E2[q].a = E2[q].b; E2[q].b = t2; }
        qsort(E2, ne2, sizeof(Edge), cmp_edge);
        size_t w2 = 0;
        for (size_t q = 0; q < ne2; q++) {
            if (w2 > 0 && E2[w2 - 1].a == E2[q].a && E2[w2 - 1].b == E2[q].b) E2[w2 - 1].w++;
            else E2[w2++] = E2[q];
        }
        ne2 = w2;
        qsort(E2, ne2, sizeof(Edge), cmp_w);
        int *uf = malloc(sizeof(int) * nlive);
        for (int i = 0; i < nlive; i++) uf[i] = i;
        /* Kruskal: keep the chosen edges, then orient them by BFS so that every tree edge
         * becomes exactly one parent link (overwriting parents drops edges and their
         * equalities, which silently relaxes the problem).                            */
        int *ea = malloc(sizeof(int) * (nlive ? nlive : 1)), *eb = malloc(sizeof(int) * (nlive ? nlive : 1));
        int nte = 0;
        for (size_t q = 0; q < ne2 && nte < nlive - 1; q++) {
            int ra = find2(uf, E2[q].a), rb = find2(uf, E2[q].b);
            if (ra == rb) continue;
            uf[rb] = ra;
            ea[nte] = E2[q].a; eb[nte] = E2[q].b; nte++;
        }
        int *deg2 = calloc(nlive + 1, sizeof(int));
        for (int q = 0; q < nte; q++) { deg2[ea[q]]++; deg2[eb[q]]++; }
        int *off = malloc(sizeof(int) * (nlive + 1));
        int acc = 0;
        for (int i = 0; i < nlive; i++) { off[i] = acc; acc += deg2[i]; }
        off[nlive] = acc;
        int *nbr = malloc(sizeof(int) * (acc ? acc : 1));
        int *fl = calloc(nlive + 1, sizeof(int));
        for (int q = 0; q < nte; q++) {
            nbr[off[ea[q]] + fl[ea[q]]++] = eb[q];
            nbr[off[eb[q]] + fl[eb[q]]++] = ea[q];
        }
        char *seen = calloc(nlive + 1, 1);
        int *queue = malloc(sizeof(int) * (nlive ? nlive : 1));
        for (int r = 0; r < nlive; r++) {
            if (seen[r]) continue;
            int qh = 0, qt = 0;
            queue[qt++] = r; seen[r] = 1;
            cl[lst[r]].parent = -1;                     /* root of this component */
            while (qh < qt) {
                int x = queue[qh++];
                for (int q = off[x]; q < off[x + 1]; q++) {
                    int w3 = nbr[q];
                    if (seen[w3]) continue;
                    seen[w3] = 1;
                    cl[lst[w3]].parent = lst[x];        /* child -> parent */
                    queue[qt++] = w3;
                }
            }
        }
        free(ea); free(eb); free(deg2); free(off); free(nbr); free(fl); free(seen); free(queue);
        free(lst); free(where); free(cnt2); free(ptr); free(lists); free(fill2); free(E2); free(uf);
    }
    *cliq_out = cl;
    *repof_out = repv;
    *nmerge_out = nmerge;
    for (int i = 0; i < n; i++) free(adj[i].a);
    free(adj); free(gone); free(deg); free(buf); free(mark);
    *pos_out = pos;
    int live = 0;
    for (int s = 0; s < ncl0; s++) if (cl[s].alive) live++;
    return live;
}

/* Undo information of a trial conversion: the original block array (not freed) and the
 * data that the conversion reallocates. */
typedef struct { Block *blk; int nblk, m, nch, nconv, ncl; double *b, *b0; int *orig; int *conv; } ChUndo;

static int chordal_convert_ex(Problem *P, const Params *par, int verbose, ChUndo *u);

/* Rewrite the problem with chordal decomposition. Returns the number of blocks converted. */
static int chordal_convert_ex(Problem *P, const Params *par, int verbose, ChUndo *u) {
    if (par->chordal == 0) return 0;
    int converted = 0;
    if (u) {
        memset(u, 0, sizeof(*u));
        u->blk = P->blk; u->nblk = P->nblk; u->m = P->m;
        u->nch = P->ps ? P->ps->nch : 0;
        u->b = malloc(sizeof(double) * (P->m + 1)); memcpy(u->b, P->b, sizeof(double) * P->m);
        u->b0 = malloc(sizeof(double) * (P->m + 1)); memcpy(u->b0, P->b0, sizeof(double) * P->m);
        u->orig = malloc(sizeof(int) * (P->m + 1)); memcpy(u->orig, P->orig, sizeof(int) * P->m);
        u->conv = malloc(sizeof(int) * (P->nblk + 1));
    }
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        if (B->type != BLK_SDP || B->n < par->chordal_minn) continue;
        Clique *cl = NULL;
        int *repv = NULL, nmerge = 0, *epos = NULL;
        int live = chordal_cliques(B, par, &cl, &repv, &nmerge, &epos);
        if (live <= 1) { if (cl) { for (int s = 0; s < B->n; s++) free(cl[s].v); free(cl); free(repv); } free(epos); continue; }
        const int n = B->n, ncl0 = n;
        /* index of each live clique, and its position among the new blocks */
        int *idx = cm(sizeof(int) * ncl0);
        int nb2 = 0;
        double c3 = 0, csum = 0;
        for (int s = 0; s < ncl0; s++) {
            idx[s] = -1;
            if (!cl[s].alive) continue;
            idx[s] = nb2++;
            c3 += (double)cl[s].n * cl[s].n * cl[s].n;
            csum += cl[s].n;
        }
        /* overlap equalities: one per entry of each clique-tree separator */
        long extra = 0;
        for (int s = 0; s < ncl0; s++) {
            if (!cl[s].alive || cl[s].parent < 0) continue;
            int p = cl[s].parent, sep = 0;
            for (int q = 0; q < cl[s].n; q++) if (cl_has(&cl[p], cl[s].v[q]) >= 0) sep++;
            extra += (long)sep * (sep + 1) / 2;
        }
        double n3 = (double)n * n * n;
        /* Cost of one iteration, dense work plus the Schur factorization. The conversion
         * shrinks the first and grows the second (m gains one row per separator entry),
         * so it only pays when the cliques are small *and* the overlaps few. On the
         * 793-bus AC-OPF block the dense work falls from 2.3e10 to 1.5e6 flops but m goes
         * 7019 -> 13158, i.e. m^3/3 from 1.2e11 to 7.6e11: a net loss, and measured as
         * such (17 s per iteration against 8-12 s). */
        double m0d = (double)P->m, mnd = m0d + (double)extra;
        double cost_old = 4.0 * n3 + m0d * m0d * m0d / 3.0;
        double cost_new = 4.0 * c3 + mnd * mnd * mnd / 3.0;
        if (par->chordal > 0) cost_new = 0;                    /* forced */
        if (cost_new > par->chordal_gain2 * cost_old ||
            c3 > par->chordal_gain * n3 || extra > par->chordal_maxextra * (double)P->m) {
            if (verbose)
                printf("chordal: block %d (n = %d) not converted: %d cliques, dense work %.1e -> %.1e, "
                       "but %ld extra constraints make the Schur factorization %.1e -> %.1e "
                       "(total %.1e -> %.1e flops/iteration)\n",
                       k + 1, n, nb2, 4 * n3, 4 * c3, extra,
                       m0d * m0d * m0d / 3.0, mnd * mnd * mnd / 3.0, cost_old, cost_new);
            for (int s = 0; s < ncl0; s++) free(cl[s].v);
            free(cl); free(repv); free(idx); free(epos);
            continue;
        }
        /* ---- build the new blocks */
        const int m0 = P->m, mnew = m0 + (int)extra;
        Block *nb_arr = cm(sizeof(Block) * (P->nblk - 1 + nb2));
        int put = 0;
        for (int j = 0; j < P->nblk; j++) if (j != k) nb_arr[put++] = P->blk[j];
        const int base = put;
        for (int s = 0; s < ncl0; s++) {
            if (!cl[s].alive) continue;
            Block *NB = &nb_arr[base + idx[s]];
            NB->type = BLK_SDP;
            NB->n = cl[s].n;
            NB->ncon = 0;
            NB->con = NULL;
            NB->A = NULL;
        }
        /* entries: (p,q) goes to the clique owning the earlier-eliminated endpoint;
         * both endpoints are in that clique by the elimination property              */
        typedef struct { int blk, i, j; double v; } Ent2;
        int *cnt = cm(sizeof(int) * nb2);
        /* first pass: count entries per (clique, matrix) to size the SpSym arrays */
        Ent2 *tmp = malloc(sizeof(Ent2) * 4096);
        size_t tcap = 4096;
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *S = t < 0 ? &B->C : &B->A[t];
            if ((size_t)S->nnz > tcap) { tcap = 2 * S->nnz; tmp = realloc(tmp, sizeof(Ent2) * tcap); }
            memset(cnt, 0, sizeof(int) * nb2);
            size_t ne = 0;
            /* the whole constraint in one clique: the clique of its first-eliminated node
             * contains every node of a clique of the chordal extension */
            int whole = -1;
            if (S->nnz > 0) {
                int v0 = S->row[0];
                for (int q = 0; q < S->nnz; q++) {
                    if (epos[S->row[q]] < epos[v0]) v0 = S->row[q];
                    if (epos[S->col[q]] < epos[v0]) v0 = S->col[q];
                }
                whole = repv[v0];
                for (int q = 0; q < S->nnz && whole >= 0; q++)
                    if (cl_has(&cl[whole], S->row[q]) < 0 || cl_has(&cl[whole], S->col[q]) < 0) whole = -1;
            }
            for (int q = 0; q < S->nnz; q++) {
                int i = S->row[q], j = S->col[q];
                int owner = whole >= 0 ? whole : repv[i];
                if (cl_has(&cl[owner], j) < 0) owner = repv[j];
                if (cl_has(&cl[owner], i) < 0 || cl_has(&cl[owner], j) < 0) { owner = -1; }
                if (owner < 0) {               /* should not happen: fall back, no conversion */
                    if (verbose) {             /* report before anything is freed */
                        int oi = repv[i], oj = repv[j];
                        printf("chordal: entry (%d,%d) not covered: repv[i]=%d (alive %d, n %d, has i %d, has j %d), "
                               "repv[j]=%d (alive %d, n %d)\n", i, j, oi, cl[oi].alive, cl[oi].n,
                               cl_has(&cl[oi], i), cl_has(&cl[oi], j), oj, cl[oj].alive, cl[oj].n);
                    }
                    for (int s2 = 0; s2 < nb2; s2++) {       /* clique blocks built so far */
                        Block *NB = &nb_arr[base + s2];
                        for (int t2 = 0; t2 < NB->ncon; t2++) { free(NB->A[t2].row); free(NB->A[t2].col); free(NB->A[t2].val); }
                        free(NB->A); free(NB->con);
                        free(NB->C.row); free(NB->C.col); free(NB->C.val);
                    }
                    free(tmp); free(cnt); free(nb_arr);
                    for (int s2 = 0; s2 < ncl0; s2++) free(cl[s2].v);
                    free(cl); free(repv); free(idx); free(epos);
                    goto next_block;
                }
                tmp[ne].blk = idx[owner];
                tmp[ne].i = cl_has(&cl[owner], i);
                tmp[ne].j = cl_has(&cl[owner], j);
                tmp[ne].v = S->val[q];
                cnt[tmp[ne].blk]++;
                ne++;
            }
            for (int s = 0; s < nb2; s++) {
                if (!cnt[s]) continue;
                Block *NB = &nb_arr[base + s];
                if (t < 0) {
                    NB->C.nnz = 0;
                    NB->C.row = malloc(sizeof(int) * cnt[s]);
                    NB->C.col = malloc(sizeof(int) * cnt[s]);
                    NB->C.val = malloc(sizeof(double) * cnt[s]);
                } else {
                    NB->A = realloc(NB->A, sizeof(SpSym) * (NB->ncon + 1));
                    NB->con = realloc(NB->con, sizeof(int) * (NB->ncon + 1));
                    memset(&NB->A[NB->ncon], 0, sizeof(SpSym));
                    NB->A[NB->ncon].row = malloc(sizeof(int) * cnt[s]);
                    NB->A[NB->ncon].col = malloc(sizeof(int) * cnt[s]);
                    NB->A[NB->ncon].val = malloc(sizeof(double) * cnt[s]);
                    NB->con[NB->ncon] = B->con[t];
                    NB->ncon++;
                }
            }
            for (size_t q = 0; q < ne; q++) {
                Block *NB = &nb_arr[base + tmp[q].blk];
                SpSym *T = (t < 0) ? &NB->C : &NB->A[NB->ncon - 1];
                int i = tmp[q].i, j = tmp[q].j;
                T->row[T->nnz] = i < j ? i : j;
                T->col[T->nnz] = i < j ? j : i;
                T->val[T->nnz] = tmp[q].v;
                T->nnz++;
            }
        }
        free(tmp);
        free(cnt);
        /* empty C for cliques without objective entries */
        for (int s = 0; s < nb2; s++) {
            Block *NB = &nb_arr[base + s];
            if (!NB->C.row) { NB->C.row = malloc(sizeof(int)); NB->C.col = malloc(sizeof(int)); NB->C.val = malloc(sizeof(double)); NB->C.nnz = 0; }
        }
        /* ---- overlap equalities */
        double *b2 = realloc(P->b, sizeof(double) * (mnew + 1));
        int *orig2 = realloc(P->orig, sizeof(int) * (mnew + 1));
        double *b02 = realloc(P->b0, sizeof(double) * (mnew + 1));
        int row = m0;
        for (int s = 0; s < ncl0; s++) {
            if (!cl[s].alive || cl[s].parent < 0) continue;
            int p = cl[s].parent;
            Block *Bc = &nb_arr[base + idx[s]], *Bp = &nb_arr[base + idx[p]];
            for (int a = 0; a < cl[s].n; a++) {
                int va = cl[s].v[a];
                if (cl_has(&cl[p], va) < 0) continue;
                for (int b3 = a; b3 < cl[s].n; b3++) {
                    int vb = cl[s].v[b3];
                    if (cl_has(&cl[p], vb) < 0) continue;
                    b2[row] = 0.0; b02[row] = 0.0; orig2[row] = -1;
                    for (int side = 0; side < 2; side++) {
                        Block *NB = side ? Bp : Bc;
                        int ia = side ? cl_has(&cl[p], va) : a, ib = side ? cl_has(&cl[p], vb) : b3;
                        NB->A = realloc(NB->A, sizeof(SpSym) * (NB->ncon + 1));
                        NB->con = realloc(NB->con, sizeof(int) * (NB->ncon + 1));
                        SpSym *T = &NB->A[NB->ncon];
                        memset(T, 0, sizeof(SpSym));
                        T->nnz = 1;
                        T->row = malloc(sizeof(int)); T->col = malloc(sizeof(int)); T->val = malloc(sizeof(double));
                        T->row[0] = ia < ib ? ia : ib;
                        T->col[0] = ia < ib ? ib : ia;
                        T->val[0] = side ? -1.0 : 1.0;
                        NB->con[NB->ncon] = row;
                        NB->ncon++;
                    }
                    row++;
                }
            }
        }
        for (int s = 0; s < nb2; s++) {
            Block *NB = &nb_arr[base + s];
            spsym_finish(&NB->C, NB->n, 0);
            for (int t = 0; t < NB->ncon; t++) spsym_finish(&NB->A[t], NB->n, 0);
        }
        P->b = b2; P->b0 = b02; P->orig = orig2;
        P->m = mnew;
        if (P->ps) {                          /* record for the primal postsolve (completion) */
            Postsolve *ps = P->ps;
            ps->ch = realloc(ps->ch, sizeof(PSChordal) * (ps->nch + 1));
            PSChordal *r = &ps->ch[ps->nch++];
            r->pos = k; r->n = n; r->ncl = nb2;
            r->cptr = malloc(sizeof(int) * (nb2 + 1));
            r->par = malloc(sizeof(int) * nb2);
            r->cptr[0] = 0;
            for (int s = 0; s < ncl0; s++) if (cl[s].alive) r->cptr[idx[s] + 1] = cl[s].n;
            for (int t = 0; t < nb2; t++) r->cptr[t + 1] += r->cptr[t];
            r->cv = malloc(sizeof(int) * (r->cptr[nb2] + 1));
            for (int s = 0; s < ncl0; s++) {
                if (!cl[s].alive) continue;
                memcpy(r->cv + r->cptr[idx[s]], cl[s].v, sizeof(int) * cl[s].n);
                r->par[idx[s]] = cl[s].parent >= 0 ? idx[cl[s].parent] : -1;
            }
        }
        if (u) {                              /* trial: keep the original array and block */
            /* position of block k in the original array: unconverted originals keep their
             * relative order and come first */
            int seen = -1, q0 = 0;
            for (q0 = 0; q0 < u->nblk; q0++) {
                int isconv = 0;
                for (int z = 0; z < u->nconv; z++) if (u->conv[z] == q0) isconv = 1;
                if (!isconv) seen++;
                if (seen == k) break;
            }
            u->conv[u->nconv++] = q0;
            u->ncl += nb2;
            if (P->blk != u->blk) free(P->blk);
        } else {
            block_free_contents(&P->blk[k]);     /* the converted block's own data */
            free(P->blk);
        }
        P->blk = nb_arr;
        P->nblk = P->nblk - 1 + nb2;
        if (verbose)
        {
            int mx = 0; double s2 = 0;
            for (int s = 0; s < nb2; s++) { const Block *NB = &P->blk[P->nblk - nb2 + s]; if (NB->n > mx) mx = NB->n; s2 += (double)NB->ncon * NB->ncon; }
            printf("chordal: block (n = %d) -> %d cliques (max %d, %d merges), %ld overlap constraints, "
                   "dense work %.1e -> %.1e flops/iteration, sum ncon^2/2 %.2e\n",
                   n, nb2, mx, nmerge, extra, 4 * n3, 4 * c3, 0.5 * s2);
        }
        converted++;
        for (int s = 0; s < ncl0; s++) free(cl[s].v);
        free(cl); free(repv); free(idx); free(epos);
        k = -1;                       /* block list changed: restart the scan */
        continue;
    next_block:
        continue;
    }
    return converted;
}

/* Positive semidefinite completion of the clique blocks of one conversion, as a factor
 * X = V V' built along the clique tree from the roots. For a clique C with separator S
 * (nodes already filled) and new nodes N:
 *     V[N] = X_C[N,S] pinv(V[S]')          (so that V[N] V[S]' = X_C[N,S] on range V[S]),
 *     R    = X_C[N,N] - V[N] V[N]',  V[N] gets new columns sqrt(R_+).
 * This is the maximum-determinant completion when the overlap copies agree, and it is
 * positive semidefinite by construction when they do not (an interior iterate agrees only
 * up to the primal infeasibility, and near a low-rank optimum pinv(X[S,S]) amplifies that
 * disagreement: the explicit completion formula lost 1e-5..1e-4 of positive
 * semidefiniteness on AC-OPF). Eigenvalues of V[S]V[S]' below the disagreement level count
 * as zero. Cost sum_C c^2 r + n^2 r, r = number of columns of V (<= n).                 */
double *chordal_complete(const PSChordal *r, double *const *Xc) {
    const int n = r->n, ncl = r->ncl;
    char *filled = calloc(n + 1, 1);
    int *cnt = calloc(ncl + 1, sizeof(int)), *cp = malloc(sizeof(int) * (ncl + 1)), *ch = malloc(sizeof(int) * (ncl + 1));
    for (int t = 0; t < ncl; t++) if (r->par[t] >= 0) cnt[r->par[t]]++;
    cp[0] = 0;
    for (int t = 0; t < ncl; t++) cp[t + 1] = cp[t] + cnt[t];
    memset(cnt, 0, sizeof(int) * (ncl + 1));
    for (int t = 0; t < ncl; t++) if (r->par[t] >= 0) { int p = r->par[t]; ch[cp[p] + cnt[p]++] = t; }
    int *ord = malloc(sizeof(int) * (ncl + 1)), no = 0;
    for (int t = 0; t < ncl; t++) {
        if (r->par[t] >= 0) continue;
        int h = no; ord[no++] = t;
        while (h < no) { int u = ord[h++]; for (int q = cp[u]; q < cp[u + 1]; q++) ord[no++] = ch[q]; }
    }
    int cmax = 1;
    for (int t = 0; t < ncl; t++) if (r->cptr[t + 1] - r->cptr[t] > cmax) cmax = r->cptr[t + 1] - r->cptr[t];
    const size_t c2 = (size_t)cmax * cmax + 1;
    /* Make the overlap copies agree while keeping every clique block positive semidefinite:
     * adding a PSD matrix to a principal submatrix keeps a block PSD, so first (children
     * before parents) each parent's separator part is raised by the negative part of
     * X_P[S,S] - X_C[S,S], which makes it dominate every child's copy, then (parents
     * before children) each child's separator part is overwritten by its parent's, again
     * a PSD increase. The changes are of the size of the disagreement, and afterwards the
     * completion is exact. (Completing the disagreeing copies directly loses up to
     * sqrt(disagreement) near a low-rank optimum, where pinv(X[S,S]) amplifies it.) */
    double **Xw = malloc(sizeof(double *) * (ncl + 1));
    for (int t = 0; t < ncl; t++) {
        const int c = r->cptr[t + 1] - r->cptr[t];
        Xw[t] = malloc(sizeof(double) * ((size_t)c * c + 1));
        memcpy(Xw[t], Xc[t], sizeof(double) * (size_t)c * c);
    }
    /* The returned blocks can be slightly indefinite (the polishing step trades a little
     * positive semidefiniteness for feasibility; -5e-9 relative on AC-OPF), and the
     * completion amplifies any indefiniteness down the clique tree. Every block gets its
     * eigenvalues floored at floor_rel * lambda_max first (a change of that size in the
     * pattern entries): positive definite blocks with bounded condition numbers. */
    {
        const char *fs = getenv("BRISK_COMPL_FLOOR");
        const double floor_rel = fs ? atof(fs) : 1e-12;
        double *A = malloc(sizeof(double) * c2), *ev = malloc(sizeof(double) * (cmax + 1));
        int lw4 = 3 * cmax * cmax + 64;
        double *wk = malloc(sizeof(double) * lw4);
        for (int t = 0; t < ncl; t++) {
            const int c = r->cptr[t + 1] - r->cptr[t];
            memcpy(A, Xw[t], sizeof(double) * (size_t)c * c);
            int info = 0;
            BL(dsyev_)("V", "L", &c, A, &c, ev, wk, &lw4, &info);
            if (info != 0) continue;
            const double fl0 = floor_rel * fmax(ev[c - 1], 0.0);
            if (ev[0] >= fl0) continue;
            for (int a = 0; a < c; a++) for (int b = 0; b < c; b++) {
                double s = 0;
                for (int k = 0; k < c; k++) { const double lk = ev[k] < fl0 ? fl0 : ev[k]; s += lk * A[a + (size_t)k * c] * A[b + (size_t)k * c]; }
                Xw[t][a + (size_t)b * c] = s;
            }
        }
        free(A); free(ev); free(wk);
    }
    {
        int *lp = malloc(sizeof(int) * (n + 1));
        for (int i = 0; i < n; i++) lp[i] = -1;
        int *ic = malloc(sizeof(int) * (cmax + 1)), *ip = malloc(sizeof(int) * (cmax + 1));
        double *D = malloc(sizeof(double) * ((size_t)cmax * cmax + 1)), *ev = malloc(sizeof(double) * (cmax + 1));
        int lw2 = 3 * cmax * cmax + 64;
        double *wk2 = malloc(sizeof(double) * lw2);
        for (int pass = 0; pass < 2; pass++)
            for (int oo = 0; oo < no; oo++) {
                const int t = pass == 0 ? ord[no - 1 - oo] : ord[oo], p = r->par[t];
                if (p < 0) continue;
                const int c = r->cptr[t + 1] - r->cptr[t], cpn = r->cptr[p + 1] - r->cptr[p];
                const int *v = r->cv + r->cptr[t], *vp = r->cv + r->cptr[p];
                for (int a = 0; a < cpn; a++) lp[vp[a]] = a;
                int ns = 0;
                for (int a = 0; a < c; a++) if (lp[v[a]] >= 0) { ic[ns] = a; ip[ns] = lp[v[a]]; ns++; }
                for (int a = 0; a < cpn; a++) lp[vp[a]] = -1;
                if (ns == 0) continue;
                double *XC = Xw[t], *XP = Xw[p];
                if (pass == 0) {
                    for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++)
                        D[a + (size_t)b * ns] = XP[ip[a] + (size_t)ip[b] * cpn] - XC[ic[a] + (size_t)ic[b] * c];
                    int info = 0;
                    BL(dsyev_)("V", "L", &ns, D, &ns, ev, wk2, &lw2, &info);
                    if (info != 0) continue;
                    for (int k = 0; k < ns; k++) {
                        if (ev[k] >= 0) continue;
                        const double w = -ev[k];
                        for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++)
                            XP[ip[a] + (size_t)ip[b] * cpn] += w * D[a + (size_t)k * ns] * D[b + (size_t)k * ns];
                    }
                } else {
                    for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++)
                        XC[ic[a] + (size_t)ic[b] * c] = XP[ip[a] + (size_t)ip[b] * cpn];
                }
            }
        free(lp); free(ic); free(ip); free(D); free(ev); free(wk2);
    }
    if (getenv("BRISK_CHK_COMPL")) {
        double wmin0 = 1e300, wmin1 = 1e300, rel0 = 1e300;
        for (int t = 0; t < ncl; t++) {
            const int c = r->cptr[t + 1] - r->cptr[t];
            for (int pass = 0; pass < 2; pass++) {
                double *A = malloc(sizeof(double) * c * c), *ev = malloc(sizeof(double) * c), *wk = malloc(sizeof(double) * (3 * c * c + 64));
                memcpy(A, pass ? Xw[t] : Xc[t], sizeof(double) * c * c);
                int lw3 = 3 * c * c + 64, info = 0;
                BL(dsyev_)("N", "L", &c, A, &c, ev, wk, &lw3, &info);
                if (pass == 0) { wmin0 = fmin(wmin0, ev[0]); rel0 = fmin(rel0, ev[0] / fmax(ev[c - 1], 1e-300)); } else wmin1 = fmin(wmin1, ev[0]);
                free(A); free(ev); free(wk);
            }
        }
        double dmax = 0;
        for (int t = 0; t < ncl; t++) { const int c = r->cptr[t + 1] - r->cptr[t]; for (size_t q = 0; q < (size_t)c * c; q++) dmax = fmax(dmax, fabs(Xw[t][q] - Xc[t][q])); }
        printf("completion: clique lambda_min before repair %.2e (relative %.2e), after %.2e; max change %.2e\n", wmin0, rel0, wmin1, dmax);
    }
    Xc = (double *const *)Xw;
    /* Explicit maximum-determinant completion in extended precision: for a clique C with
     * separator S and new nodes N, X[N,R] = X_C[N,S] X_C[S,S]^-1 X[S,R] for the filled
     * nodes R outside C. The pattern entries are the (consistent) clique entries. Near a
     * low-rank optimum the separators have condition numbers of 1e12-1e14, and rounding in
     * the entries already completed is amplified from clique to clique down the tree: in
     * double precision lambda_min(X) ended at -1e-5 on AC-OPF (and a diagonal shift, or a
     * factor form X = V V', only traded that for errors in the pattern entries). With the
     * 64-bit mantissa of long double the same recursion stays at rounding level.        */
    typedef long double LD;
    LD *XL = calloc((size_t)n * n + 1, sizeof(LD));
    int *fl = malloc(sizeof(int) * (n + 1)), nf = 0, *loc = malloc(sizeof(int) * (n + 1)), *Rn = malloc(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) loc[i] = -1;
    LD *Ch = malloc(sizeof(LD) * c2), *Wt = malloc(sizeof(LD) * c2);
    int *Sl = malloc(sizeof(int) * (cmax + 1)), *Nl = malloc(sizeof(int) * (cmax + 1));
    LD *G = NULL;
    size_t gcap = 0;
    int rk = 0;
    for (int o = 0; o < no; o++) {
        const int t = ord[o], c = r->cptr[t + 1] - r->cptr[t];
        const int *v = r->cv + r->cptr[t];
        const double *Xt = Xc[t];
        int ns = 0, nn = 0;
        for (int a = 0; a < c; a++) { if (filled[v[a]]) Sl[ns++] = a; else Nl[nn++] = a; }
        if (nn == 0) continue;
        for (int a = 0; a < nn; a++) {
            const int ia = v[Nl[a]];
            for (int b = 0; b < c; b++) {
                const LD x = Xt[Nl[a] + (size_t)b * c];
                XL[ia + (size_t)v[b] * n] = x;
                XL[v[b] + (size_t)ia * n] = x;
            }
        }
        for (int a = 0; a < c; a++) loc[v[a]] = a;
        int nr = 0;
        for (int q = 0; q < nf; q++) if (loc[fl[q]] < 0) Rn[nr++] = fl[q];
        for (int a = 0; a < c; a++) loc[v[a]] = -1;
        if (ns > 0 && nr > 0) {
            /* Cholesky of X_C[S,S]; a non-positive pivot drops that direction (pinv-like) */
            for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++) Ch[a + (size_t)b * ns] = Xt[Sl[a] + (size_t)Sl[b] * c];
            LD dmax = 0;
            for (int a = 0; a < ns; a++) if (Ch[a + (size_t)a * ns] > dmax) dmax = Ch[a + (size_t)a * ns];
            char *dead = calloc(ns + 1, 1);
            for (int k = 0; k < ns; k++) {
                LD d = Ch[k + (size_t)k * ns];
                for (int q = 0; q < k; q++) d -= Ch[k + (size_t)q * ns] * Ch[k + (size_t)q * ns];
                if (d <= 1e-30L * dmax) { dead[k] = 1; for (int a = k; a < ns; a++) Ch[a + (size_t)k * ns] = 0; continue; }
                const LD l = sqrtl(d);
                Ch[k + (size_t)k * ns] = l;
                for (int a = k + 1; a < ns; a++) {
                    LD s = Ch[a + (size_t)k * ns];
                    for (int q = 0; q < k; q++) s -= Ch[a + (size_t)q * ns] * Ch[k + (size_t)q * ns];
                    Ch[a + (size_t)k * ns] = s / l;
                }
            }
            /* Wt = X_C[S,S]^-1 X_C[S,N]  (ns x nn) */
            for (int a = 0; a < nn; a++) {
                LD *w = Wt + (size_t)a * ns;
                for (int b = 0; b < ns; b++) w[b] = Xt[Sl[b] + (size_t)Nl[a] * c];
                for (int k = 0; k < ns; k++) {               /* L y = w */
                    if (dead[k]) { w[k] = 0; continue; }
                    LD s = w[k];
                    for (int q = 0; q < k; q++) s -= Ch[k + (size_t)q * ns] * w[q];
                    w[k] = s / Ch[k + (size_t)k * ns];
                }
                for (int k = ns - 1; k >= 0; k--) {          /* L' x = y */
                    if (dead[k]) { w[k] = 0; continue; }
                    LD s = w[k];
                    for (int q = k + 1; q < ns; q++) s -= Ch[q + (size_t)k * ns] * w[q];
                    w[k] = s / Ch[k + (size_t)k * ns];
                }
            }
            free(dead);
            if ((size_t)ns * nr > gcap) { gcap = (size_t)ns * nr; G = realloc(G, sizeof(LD) * gcap); }
            for (int q = 0; q < nr; q++) for (int b = 0; b < ns; b++) G[b + (size_t)q * ns] = XL[v[Sl[b]] + (size_t)Rn[q] * n];
            for (int q = 0; q < nr; q++) {
                const LD *g = G + (size_t)q * ns;
                LD *col = XL + (size_t)Rn[q] * n;
                for (int a = 0; a < nn; a++) {
                    const LD *w = Wt + (size_t)a * ns;
                    LD s = 0;
                    for (int b = 0; b < ns; b++) s += w[b] * g[b];
                    col[v[Nl[a]]] = s;
                    XL[Rn[q] + (size_t)v[Nl[a]] * n] = s;
                }
            }
        }
        for (int a = 0; a < nn; a++) { filled[v[Nl[a]]] = 1; fl[nf++] = v[Nl[a]]; }
    }
    double *X = malloc(sizeof(double) * ((size_t)n * n + 1));
    for (size_t q = 0; q < (size_t)n * n; q++) X[q] = (double)XL[q];
    free(XL); free(fl); free(loc); free(Rn); free(G); free(Ch); free(Wt); free(Sl); free(Nl);
    if (getenv("BRISK_CHK_COMPL")) {
        double mx = 0, mxr = 0; int tw = -1;
        for (int t = 0; t < ncl; t++) {
            const int c = r->cptr[t + 1] - r->cptr[t]; const int *v = r->cv + r->cptr[t];
            for (int a = 0; a < c; a++) for (int b = 0; b < c; b++) {
                double d = fabs(X[v[a] + (size_t)v[b] * n] - Xc[t][a + (size_t)b * c]);
                if (d > mx) { mx = d; tw = t; }
                mxr = fmax(mxr, fabs(Xc[t][a + (size_t)b * c]));
            }
        }
        printf("completion: n %d rank %d, max |X - X_C| %.2e (clique %d, par %d), max |X_C| %.2e\n", n, rk, mx, tw, tw >= 0 ? r->par[tw] : -2, mxr);
    }
    for (int t = 0; t < ncl; t++) free(Xw[t]);
    free(Xw);
    free(filled); free(cnt); free(cp); free(ch); free(ord);
    return X;
}

static void chordal_undo(Problem *P, ChUndo *u) {
    const int nnew = P->nblk;
    for (int k = nnew - u->ncl; k < nnew; k++) block_free_contents(&P->blk[k]);
    if (P->blk != u->blk) free(P->blk);
    P->blk = u->blk; P->nblk = u->nblk; P->m = u->m;
    memcpy(P->b, u->b, sizeof(double) * u->m);
    memcpy(P->b0, u->b0, sizeof(double) * u->m);
    memcpy(P->orig, u->orig, sizeof(int) * u->m);
    if (P->ps) {
        for (int q = u->nch; q < P->ps->nch; q++) { free(P->ps->ch[q].cptr); free(P->ps->ch[q].cv); free(P->ps->ch[q].par); }
        P->ps->nch = u->nch;
    }
    free(u->b); free(u->b0); free(u->orig); free(u->conv);
    memset(u, 0, sizeof(*u));
}
static void chordal_commit(Problem *P, ChUndo *u) {
    if (P->blk != u->blk) {
        for (int z = 0; z < u->nconv; z++) block_free_contents(&u->blk[u->conv[z]]);
        free(u->blk);
    }
    free(u->b); free(u->b0); free(u->orig); free(u->conv);
    memset(u, 0, sizeof(*u));
}

/* Estimated cost of one interior-point iteration (flop-equivalents): dense block work at
 * BLAS-3 rate, small blocks at a fifth of it, and the Schur complement from the symbolic
 * analysis of its actual pattern (sparse factorization plus ~10 solves per iteration,
 * memory-bound), or dense m^3/3 when the sparse analysis gives up. The chordal gate needs
 * the real pattern: the dense estimate m^3/3 of the converted problem rejected the
 * conversion of the AC-OPF moment block although its Schur complement factors in 4e8
 * flops against 7.6e11 dense. */
void chordal_keep_analysis(SChol *S, int m);
/* the analysis of the last estimate, kept for the solver (see chordal_take_analysis) */
static SChol *g_keep_sc = NULL;
static int g_keep_on = 0;
static double iter_cost(const Problem *P, double *sp_flops, double budget);
static double iter_cost_keep(const Problem *P, double *sp_flops, double budget, int keep) {
    g_keep_on = keep;
    double c = iter_cost(P, sp_flops, budget);
    g_keep_on = 0;
    return c;
}
/* 4.22: fill limit of the Schur factor, ~64 bytes per entry (panel, assembled copy, hash):
 * 6e7 is ~3.8 GB (2e7 rejected every AC-OPF case from 6468 buses up) */
double g_fillmax = 6e7;
static double g_ic_t0;
static double iter_cost(const Problem *P, double *sp_flops, double budget) {
    g_ic_t0 = wtime();
    { const char *e = getenv("BRISK_FILLMAX"); g_fillmax = e ? atof(e) : 6e7 * brisk_mem_scale(); }   /* 5.7: 6e7 per 8 GB */
    const int m = P->m;
    double dense = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_SDP) continue;
        const double n3 = (double)B->n * B->n * B->n;
        dense += B->n >= 64 ? 4.0 * n3 : 20.0 * n3;
    }
    /* Schur pattern: SDP blocks make cliques of their constraints, LP variables of theirs */
    int *mark = malloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    Adj *ad = cm(sizeof(Adj) * (m + 1));
    double pat = 0;
    int *lst = malloc(sizeof(int) * (m + 1));
    double lmax = 0;   /* 4.42: the largest constraint clique (an SDP block's constraints, or an LP variable's) */
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_SDP) {
            pat += (double)B->ncon * B->ncon;
            if (B->ncon > lmax) lmax = B->ncon;
            if (pat > 0.6 * (double)m * m) break;
        }
    }
    if (pat <= 0.6 * (double)m * m && pat <= 2e8 * brisk_mem_scale()) {
        /* 4.42: the LP variables' cliques count too (they were left out of the 2e8 cap of 4.40) */
        for (int k = 0; k < P->nblk && pat <= 2e8 * brisk_mem_scale(); k++) {
            const Block *B = &P->blk[k];
            if (B->type != BLK_LP || B->n <= 0) continue;
            int *cntv = calloc(B->n + 1, sizeof(int));
            if (!cntv) { pat = 1e300; break; }
            for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) cntv[B->A[t].row[q]]++;
            for (int p2 = 0; p2 < B->n; p2++) { pat += (double)cntv[p2] * cntv[p2]; if (cntv[p2] > lmax) lmax = cntv[p2]; }
            free(cntv);
        }
    }
    /* 4.42: a clique of l constraints is a dense l x l part of the Schur factor (4 l^2 bytes
     * in the lower triangle): a candidate whose largest clique does not fit in half the
     * memory can never run (the overlap-equality form of dense case13659: one 2 x 2 clique
     * block touched by 79 226 constraints, 25 GB) */
    {
        extern double brisk_mem_limit(void);
        if (budget > 0 && 4.0 * lmax * lmax > 0.5 * brisk_mem_limit()) {
            if (getenv("BRISK_ICDBG")) printf("   [iter_cost: largest constraint clique %.0f: its dense factor block (%.1f GB) does not fit: not a candidate]\n", lmax, 4.0 * lmax * lmax / 1e9);
            free(mark); free(ad); free(lst);
            if (sp_flops) *sp_flops = -1;
            return 1e300;
        }
    }
    /* 4.25: a lower bound first. The constraints of an SDP block are a clique of the Schur
     * pattern, so the factor costs at least s^3/3 flops (sched >= s^3) for the largest block
     * constraint count s; over the budget, the pattern is not built (the moment form of
     * mcp500: 0.46 s for an estimate rejected anyway) */
    if (budget > 0) {
        double smax = 0;
        for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP && P->blk[k].ncon > smax) smax = P->blk[k].ncon;
        /* (the dense fallback m^3/3 + 20 m^2 caps what the estimate can return) */
        if (dense + fmin(smax * smax * smax, (double)m * m * m / 3.0 + 20.0 * (double)m * m) > budget) {
            free(mark); free(ad); free(lst);
            if (sp_flops) *sp_flops = -1;
            return 1e300;
        }
    }
    double sched = 0;
    /* 4.40: at most 2e8 adjacency entries (0.8 GB): the dense pglib AC-OPF block (49 000
     * constraints) asked for 2.4e9 and crashed on the failed allocation; a pattern that full
     * factors densely anyway */
    if (pat <= 0.6 * (double)m * m && pat <= 2e8 * brisk_mem_scale() && m >= 200) {
        /* cliques of constraints: SDP blocks, and per LP variable the constraints touching
         * it; node -> cliques, then each node's neighbours deduplicated with a marker */
        int ncq = 0; size_t totq = 0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            if (B->type == BLK_SDP) { ncq++; totq += B->ncon; } else { ncq += B->n; for (int t = 0; t < B->ncon; t++) totq += B->A[t].nnz; }
        }
        int *qp = calloc(ncq + 2, sizeof(int)), *qv = malloc(sizeof(int) * (totq + 1));
        {
            int cq = 0; size_t pos2 = 0;
            for (int k = 0; k < P->nblk; k++) {
                const Block *B = &P->blk[k];
                if (B->type == BLK_SDP) { qp[cq] = (int)pos2; for (int t = 0; t < B->ncon; t++) qv[pos2++] = B->con[t]; cq++; }
                else {
                    int *cntv = calloc(B->n + 1, sizeof(int));
                    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) cntv[B->A[t].row[q]]++;
                    int *st = malloc(sizeof(int) * (B->n + 1));
                    for (int p = 0; p < B->n; p++) { qp[cq + p] = (int)pos2; st[p] = (int)pos2; pos2 += cntv[p]; }
                    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) qv[st[B->A[t].row[q]]++] = B->con[t];
                    cq += B->n;
                    free(cntv); free(st);
                }
            }
            qp[cq] = (int)pos2;
        }
        int *np2 = calloc(m + 2, sizeof(int));
        for (int cq = 0; cq < ncq; cq++) for (int q = qp[cq]; q < qp[cq + 1]; q++) np2[qv[q] + 1]++;
        for (int i = 0; i < m; i++) np2[i + 1] += np2[i];
        int *nq = malloc(sizeof(int) * (np2[m] + 1)), *fillp = malloc(sizeof(int) * (m + 1));
        memcpy(fillp, np2, sizeof(int) * m);
        for (int cq = 0; cq < ncq; cq++) for (int q = qp[cq]; q < qp[cq + 1]; q++) nq[fillp[qv[q]]++] = cq;
        free(fillp);
        for (int i = 0; i < m; i++) {
            for (int z = np2[i]; z < np2[i + 1]; z++) {
                const int cq = nq[z];
                for (int q = qp[cq]; q < qp[cq + 1]; q++) {
                    const int j = qv[q];
                    if (j == i || mark[j] == i) continue;
                    mark[j] = i;
                    if (ad[i].n == ad[i].cap) {
                        ad[i].cap = ad[i].cap ? 2 * ad[i].cap : 8;
                        int *na = realloc(ad[i].a, sizeof(int) * ad[i].cap);
                        if (!na) { fprintf(stderr, "brisk: out of memory (chordal cost estimate)\n"); exit(1); }   /* 4.42: was dereferenced unchecked */
                        ad[i].a = na;
                    }
                    ad[i].a[ad[i].n++] = j;
                }
            }
        }
        free(qp); free(qv); free(np2); free(nq);
        for (int k = 0; k < 0; k++) {
            const Block *B = &P->blk[k];
            if (B->type == BLK_SDP) {
            } else {
                /* variable p -> constraints */
                int **vc = cm(sizeof(int *) * (B->n + 1)), *vn = cm(sizeof(int) * (B->n + 1)), *vcap = cm(sizeof(int) * (B->n + 1));
                for (int t = 0; t < B->ncon; t++)
                    for (int q = 0; q < B->A[t].nnz; q++) {
                        int p = B->A[t].row[q];
                        if (vn[p] == vcap[p]) { vcap[p] = vcap[p] ? 2 * vcap[p] : 4; vc[p] = realloc(vc[p], sizeof(int) * vcap[p]); }
                        vc[p][vn[p]++] = B->con[t];
                    }
                for (int p = 0; p < B->n; p++) {
                    for (int a = 0; a < vn[p]; a++) for (int b = 0; b < vn[p]; b++) if (vc[p][a] != vc[p][b]) adj_add(&ad[vc[p][a]], vc[p][b]);
                    free(vc[p]);
                }
                free(vc); free(vn); free(vcap);
            }
        }
        /* 4.25: second lower bound, before the ordering: the factor has at least the
         * pattern's nnz(L) >= |E| + m entries, and sum of squared column counts >= nnz(L)^2 / m
         * (Cauchy-Schwarz). buck3: 4 candidates of 0.35 s of AMD each rejected anyway */
        {
            double ne = 0;
            for (int i = 0; i < m; i++) ne += ad[i].n;
            const double nl = 0.5 * ne + m, lbf = nl * nl / m;
            if (getenv("BRISK_ICDBG")) printf("   [iter_cost: m %d, adjacency %.0f entries built at %.3fs, flop bound %.2e]\n", m, ne, wtime() - g_ic_t0, lbf);
            if (budget > 0 && dense + fmin(3.0 * lbf, (double)m * m * m / 3.0 + 20.0 * (double)m * m) > budget) {
                for (int i = 0; i < m; i++) free(ad[i].a);
                free(ad); free(mark); free(lst);
                if (sp_flops) *sp_flops = -1;
                return 1e300;
            }
        }
        int *deg = malloc(sizeof(int) * (m + 1));
        int **nbr = malloc(sizeof(int *) * (m + 1));
        for (int i = 0; i < m; i++) { deg[i] = ad[i].n; nbr[i] = ad[i].a ? ad[i].a : lst; }
        const size_t cap = (size_t)fmin(g_fillmax, 0.45 * (double)m * m) + 1000;
        double fcap = (double)m * m * m / 3.0 * 0.5;
        if (budget > 0 && budget / 3.0 < fcap) fcap = budget / 3.0;
        schol_set_flopcap(fcap);
        SChol *SC = schol_analyze_adj(m, deg, nbr, cap);
        schol_set_flopcap(0);
        if (getenv("BRISK_ICDBG")) printf("   [iter_cost: analysis %s at %.3fs (flop cap %.1e)]\n", SC ? "done" : "stopped", wtime() - g_ic_t0, fcap);
        if (!SC && budget > 0 && budget / 3.0 < (double)m * m * m / 6.0) {     /* over the budget: not a candidate */
            for (int i = 0; i < m; i++) free(ad[i].a);
            free(ad); free(mark); free(lst); free(deg); free(nbr);
            if (sp_flops) *sp_flops = -1;
            return 1e300;
        }
        if (SC) {
            sched = 3.0 * schol_flops(SC) + 200.0 * (double)schol_nnz(SC);
            if (sp_flops) *sp_flops = schol_flops(SC);
            if (g_keep_on) { schol_free(g_keep_sc); g_keep_sc = SC; }
            else schol_free(SC);
        }
        free(deg); free(nbr);
    }
    if (sched == 0) { sched = (double)m * m * m / 3.0 + 20.0 * (double)m * m; if (sp_flops) *sp_flops = -1; }
    for (int i = 0; i < m; i++) free(ad[i].a);
    free(ad); free(mark); free(lst);
    return dense + sched;
}

/* Chordal conversion with the choice made on the estimated iteration cost: the support-
 * clique threshold is tried at a few values (0 = entries only), and the cheapest
 * conversion is kept if it beats the unconverted problem by the chordal_gain2 factor. */
static int chordal_convert_domain(Problem *P, const Params *par, int verbose) {
    if (par->chordal == 0) return 0;
    int any = 0;
    for (int k = 0; k < P->nblk; k++)
        if (P->blk[k].type == BLK_SDP && P->blk[k].n >= par->chordal_minn) any = 1;
    if (!any) return 0;
    Params q = *par;
    q.chordal = 1;                               /* the gate below decides */
    const double c0 = par->chordal > 0 ? 1e300 : iter_cost(P, NULL, 0);
    const int cand_auto[4] = { 8, 4, 12, 0 };
    const int *cand = par->chordal_sup >= 0 ? &par->chordal_sup : cand_auto;
    const int ncand = par->chordal_sup >= 0 ? 1 : 4;
    double best = 1e300; int bsup = -1;
    for (int ci = 0; ci < ncand; ci++) {
        q.chordal_sup = cand[ci];
        ChUndo u;
        int nc = chordal_convert_ex(P, &q, 0, &u);
        if (nc == 0) { chordal_undo(P, &u); continue; }
        double spf = 0;
        double c1 = (ncand == 1 && par->chordal > 0) ? 0 : iter_cost(P, &spf, c0 < 1e300 ? par->chordal_gain2 * c0 : 0);
        chordal_undo(P, &u);
        if (verbose) printf("chordal: support cliques <= %d: estimated cost %.2e per iteration (Schur factor %.1e flops; unconverted %.2e)\n",
                            cand[ci], c1, spf, c0 < 1e300 ? c0 : -1.0);
        if (c1 < best) { best = c1; bsup = cand[ci]; }
    }
    if (bsup < 0 || c0 < 0 || (par->chordal < 0 && best > par->chordal_gain2 * c0)) return 0;
    q.chordal_sup = bsup;
    ChUndo u;
    int nc = chordal_convert_ex(P, &q, verbose, &u);
    chordal_commit(P, &u);
    if (getenv("BRISK_DUMP_CLIQUES") && P->ps && P->ps->nch > 0) {
        FILE *f = fopen(getenv("BRISK_DUMP_CLIQUES"), "w");
        if (f) {
            const PSChordal *r = &P->ps->ch[0];
            fprintf(f, "%d %d %d\n", r->pos, r->n, r->ncl);
            for (int t = 0; t < r->ncl; t++) {
                fprintf(f, "%d", r->cptr[t + 1] - r->cptr[t]);
                for (int q2 = r->cptr[t]; q2 < r->cptr[t + 1]; q2++) fprintf(f, " %d", r->cv[q2]);
                fprintf(f, "\n");
            }
            fclose(f);
        }
    }
    return nc;
}

/* ======================================================================== */
/* Moment (range-space) form.
 *
 * The overlap-equality form above makes the dual non-unique: the multipliers of the
 * equalities between clique copies can trade against each other, so the problem is
 * primal degenerate and the Schur complement becomes ill-conditioned like 1/mu^2 (on the
 * AC-OPF relaxations the primal residual stalled at 1e-7, for MOSEK too). The moment form
 * shares the entries instead. It is the dual,
 *     max b'y  s.t.  sum_k P_k' Z_k P_k + A'y = C (on the chordal pattern), Z_k >= 0,
 * written as a primal: one row per moment (entry of the pattern, entry of a small SDP
 * block, LP variable), the clique blocks Z_k, the LP slacks, and y as free split pairs
 * (handled natively by the embedding). Its non-uniqueness is in the split of Z among the
 * cliques, i.e. in the primal of the new problem, which is harmless for an interior-point
 * method; the moments (the original X) are its unique dual. MOSEK on the same form:
 * 2.4e-11 on case793 against 6e-7 with overlap equalities.                            */

/* cliques of block B packed as a PSChordal (pos, n set by the caller); 0 if none */
static int cliques_pack(const Block *B, const Params *q, PSChordal *r, int *maxc) {
    Clique *cl = NULL; int *repv = NULL, nmerge = 0, *epos = NULL;
    int live = chordal_cliques(B, q, &cl, &repv, &nmerge, &epos);
    if (live <= 1) {
        if (cl) { for (int s = 0; s < B->n; s++) free(cl[s].v); free(cl); free(repv); }
        free(epos);
        return 0;
    }
    const int n = B->n;
    int *idx = malloc(sizeof(int) * n), nb2 = 0;
    for (int s = 0; s < n; s++) idx[s] = cl[s].alive ? nb2++ : -1;
    r->n = n; r->ncl = nb2;
    r->cptr = malloc(sizeof(int) * (nb2 + 1));
    r->par = malloc(sizeof(int) * (nb2 + 1));
    r->cptr[0] = 0;
    for (int s = 0; s < n; s++) if (cl[s].alive) r->cptr[idx[s] + 1] = cl[s].n;
    for (int t = 0; t < nb2; t++) r->cptr[t + 1] += r->cptr[t];
    r->cv = malloc(sizeof(int) * (r->cptr[nb2] + 1));
    *maxc = 0;
    for (int s = 0; s < n; s++) {
        if (!cl[s].alive) continue;
        memcpy(r->cv + r->cptr[idx[s]], cl[s].v, sizeof(int) * cl[s].n);
        r->par[idx[s]] = cl[s].parent >= 0 ? idx[cl[s].parent] : -1;
        if (cl[s].n > *maxc) *maxc = cl[s].n;
    }
    for (int s = 0; s < n; s++) free(cl[s].v);
    free(cl); free(repv); free(epos); free(idx);
    return nb2;
}

/* open-addressing map (p, q) -> row */
typedef struct { uint64_t *key; int *val; size_t mask; } EMap;
static void emap_init(EMap *h, size_t n) {
    size_t sz = 16; while (sz < 2 * n + 16) sz <<= 1;
    h->key = malloc(sizeof(uint64_t) * sz); h->val = malloc(sizeof(int) * sz); h->mask = sz - 1;
    for (size_t i = 0; i < sz; i++) h->key[i] = UINT64_MAX;
}
static int *emap_slot(EMap *h, uint64_t k, int *isnew) {
    size_t i = (k * 0x9e3779b97f4a7c15ULL) & h->mask;
    while (h->key[i] != UINT64_MAX && h->key[i] != k) i = (i + 1) & h->mask;
    *isnew = h->key[i] == UINT64_MAX;
    if (*isnew) h->key[i] = k;
    return &h->val[i];
}
static int emap_get(const EMap *h, uint64_t k) {
    size_t i = (k * 0x9e3779b97f4a7c15ULL) & h->mask;
    while (h->key[i] != UINT64_MAX) { if (h->key[i] == k) return h->val[i]; i = (i + 1) & h->mask; }
    return -1;
}

typedef struct { size_t n, cap; int *con, *blk, *i, *j; double *v; } TripList;
static void tl_add(TripList *T, int con, int blk, int i, int j, double v) {
    if (v == 0) return;
    if (T->n == T->cap) {
        T->cap = T->cap ? 2 * T->cap : 4096;
        T->con = realloc(T->con, sizeof(int) * T->cap); T->blk = realloc(T->blk, sizeof(int) * T->cap);
        T->i = realloc(T->i, sizeof(int) * T->cap); T->j = realloc(T->j, sizeof(int) * T->cap);
        T->v = realloc(T->v, sizeof(double) * T->cap);
    }
    T->con[T->n] = con; T->blk[T->n] = blk; T->i[T->n] = i; T->j[T->n] = j; T->v[T->n] = v; T->n++;
}
static void tl_free(TripList *T) { free(T->con); free(T->blk); free(T->i); free(T->j); free(T->v); }

/* Build the moment form of P with the blocks kcs[0..nd) decomposed by the cliques in rs
 * (not modified: P stays as it is). Every other SDP block is written entrywise (one row
 * per entry). Returns the new problem in *Q and the postsolve record in *M, or 0 if a
 * block not decomposed is larger than maxsmall. 4.34: any number of decomposed blocks
 * (until 4.33 exactly one, with every other SDP block at most 64). */
static void moment_record_free(PSMoment *M);
static int moment_build(const Problem *P, int nd, const int *kcs, const PSChordal *rs, Problem *Q, PSMoment *M, int maxsmall) {
    const int m0 = P->m, nb = P->nblk;
    int *dof = malloc(sizeof(int) * (nb + 1));           /* block -> index among the decomposed, -1 */
    for (int k = 0; k < nb; k++) dof[k] = -1;
    for (int q = 0; q < nd; q++) dof[kcs[q]] = q;
    for (int k = 0; k < nb; k++)
        if (dof[k] < 0 && P->blk[k].type == BLK_SDP && P->blk[k].n > maxsmall) { free(dof); return 0; }
    memset(M, 0, sizeof(*M));
    M->m0 = m0; M->nblk0 = nb; M->nd = nd;
    M->orig0 = malloc(sizeof(int) * (m0 + 1)); memcpy(M->orig0, P->orig, sizeof(int) * m0);
    M->bs0 = malloc(sizeof(int) * (nb + 1));
    for (int k = 0; k < nb; k++) M->bs0[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    M->dpos = malloc(sizeof(int) * (nd + 1)); M->dch = calloc(nd + 1, sizeof(PSChordal));
    M->dcrow = calloc(nd + 1, sizeof(int *)); M->dcoff = calloc(nd + 1, sizeof(int *));
    EMap *hs = calloc(nd + 1, sizeof(EMap));
    int nrow = 0, ncl = 0;
    for (int q = 0; q < nd; q++) {
        const PSChordal *r = &rs[q];
        M->dpos[q] = kcs[q];
        PSChordal *c0 = &M->dch[q];
        *c0 = *r; c0->pos = kcs[q];
        c0->cptr = malloc(sizeof(int) * (r->ncl + 1)); memcpy(c0->cptr, r->cptr, sizeof(int) * (r->ncl + 1));
        c0->cv = malloc(sizeof(int) * (r->cptr[r->ncl] + 1)); memcpy(c0->cv, r->cv, sizeof(int) * r->cptr[r->ncl]);
        c0->par = malloc(sizeof(int) * (r->ncl + 1)); memcpy(c0->par, r->par, sizeof(int) * r->ncl);
        const int nq = r->n;
        /* rows: pattern entries of the decomposed block */
        size_t tot = 0;
        for (int t = 0; t < r->ncl; t++) { size_t c = r->cptr[t + 1] - r->cptr[t]; tot += c * (c + 1) / 2; }
        M->dcoff[q] = malloc(sizeof(int) * (r->ncl + 1));
        M->dcrow[q] = malloc(sizeof(int) * (tot + 1));
        emap_init(&hs[q], tot);
        size_t pos = 0;
        for (int t = 0; t < r->ncl; t++) {
            const int c = r->cptr[t + 1] - r->cptr[t], *v = r->cv + r->cptr[t];
            M->dcoff[q][t] = (int)pos;
            for (int b = 0; b < c; b++)
                for (int a = 0; a <= b; a++) {
                    int p = v[a] < v[b] ? v[a] : v[b], q2 = v[a] < v[b] ? v[b] : v[a];
                    int isnew;
                    int *slot = emap_slot(&hs[q], (uint64_t)p * (uint64_t)nq + (uint64_t)q2, &isnew);
                    if (isnew) *slot = nrow++;
                    M->dcrow[q][pos++] = *slot;
                }
        }
        M->dcoff[q][r->ncl] = (int)pos;
        ncl += r->ncl;
    }
    M->chpos = nd > 0 ? M->dpos[0] : -1;
    if (nd > 0) { M->ch = M->dch[0]; M->crow = M->dcrow[0]; M->coff = M->dcoff[0]; }
    /* rows of the other blocks */
    M->brow0 = malloc(sizeof(int) * (nb + 1));
    int nslack = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (dof[k] >= 0) { M->brow0[k] = -1; continue; }
        M->brow0[k] = nrow;
        if (B->type == BLK_SDP) nrow += B->n * (B->n + 1) / 2;
        else { nrow += B->n; nslack += B->n; }
    }
    M->nrow = nrow;
    /* new blocks: cliques, small SDP blocks, one LP block [slacks | y+ | y-] */
    int nsmall = 0;
    for (int k = 0; k < nb; k++) if (dof[k] < 0 && P->blk[k].type == BLK_SDP) nsmall++;
    const int nbq = ncl + nsmall + 1;
    int *bsz = malloc(sizeof(int) * (nbq + 1));
    int *cl0 = malloc(sizeof(int) * (nd + 1));       /* first new block of each decomposed block's cliques */
    { int t0 = 0; for (int q = 0; q < nd; q++) { cl0[q] = t0; for (int t = 0; t < rs[q].ncl; t++) bsz[t0++] = rs[q].cptr[t + 1] - rs[q].cptr[t]; } }
    int *tblk = malloc(sizeof(int) * (nb + 1));      /* original block -> new block (small SDP) */
    {
        int s = ncl;
        for (int k = 0; k < nb; k++) { tblk[k] = -1; if (dof[k] < 0 && P->blk[k].type == BLK_SDP) { tblk[k] = s; bsz[s++] = P->blk[k].n; } }
    }
    const int tlp = nbq - 1, off = nslack;
    bsz[tlp] = -(nslack + 2 * m0);
    M->tlp = tlp; M->off = off;
    double *bq = calloc(nrow + 1, sizeof(double));
    TripList T = { 0 };
    /* clique blocks */
    for (int d = 0; d < nd; d++)
        for (int t = 0; t < rs[d].ncl; t++) {
            const int c = rs[d].cptr[t + 1] - rs[d].cptr[t];
            size_t q = M->dcoff[d][t];
            for (int b = 0; b < c; b++)
                for (int a = 0; a <= b; a++) tl_add(&T, M->dcrow[d][q++], cl0[d] + t, a, b, a == b ? 1.0 : 0.5);
        }
    /* small SDP blocks and LP slacks; row of entry (p, q) of block k: brow0 + q(q+1)/2 + p */
    int slack0 = 0;
    int *slackof = malloc(sizeof(int) * (nb + 1));
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        slackof[k] = -1;
        if (dof[k] >= 0) continue;
        if (B->type == BLK_SDP) {
            for (int q2 = 0; q2 < B->n; q2++)
                for (int p = 0; p <= q2; p++) tl_add(&T, M->brow0[k] + q2 * (q2 + 1) / 2 + p, tblk[k], p, q2, p == q2 ? 1.0 : 0.5);
        } else {
            slackof[k] = slack0;
            for (int j = 0; j < B->n; j++) tl_add(&T, M->brow0[k] + j, tlp, slack0 + j, slack0 + j, 1.0);
            slack0 += B->n;
        }
    }
    /* row of an entry of block k */
#define MROW(k, p, q2) (dof[k] >= 0 ? emap_get(&hs[dof[k]], (uint64_t)((p) < (q2) ? (p) : (q2)) * (uint64_t)rs[dof[k]].n + (uint64_t)((p) < (q2) ? (q2) : (p))) \
                     : P->blk[k].type == BLK_LP ? M->brow0[k] + (p) \
                     : M->brow0[k] + ((p) < (q2) ? (q2) * ((q2) + 1) / 2 + (p) : (p) * ((p) + 1) / 2 + (q2)))
    int ok = 1;
    for (int k = 0; k < nb && ok; k++) {
        const Block *B = &P->blk[k];
        /* rhs: C */
        for (int e = 0; e < B->C.nnz; e++) {
            int rr = MROW(k, B->C.row[e], B->type == BLK_LP ? B->C.row[e] : B->C.col[e]);
            if (rr < 0) { ok = 0; break; }
            bq[rr] += B->C.val[e];
        }
        /* free pairs: column of y_i = entries of A_i */
        for (int t = 0; t < B->ncon && ok; t++) {
            const SpSym *S = &B->A[t];
            const int i = B->con[t];
            for (int e = 0; e < S->nnz; e++) {
                int rr = MROW(k, S->row[e], B->type == BLK_LP ? S->row[e] : S->col[e]);
                if (rr < 0) { ok = 0; break; }
                tl_add(&T, rr, tlp, off + i, off + i, S->val[e]);
                tl_add(&T, rr, tlp, off + m0 + i, off + m0 + i, -S->val[e]);
            }
        }
    }
#undef MROW
    /* objective: min -b'y+ + b'y- */
    for (int i = 0; i < m0; i++) {
        tl_add(&T, -1, tlp, off + i, off + i, -P->b[i]);
        tl_add(&T, -1, tlp, off + m0 + i, off + m0 + i, P->b[i]);
    }
    /* ---- eliminate the inequality multipliers (TSSOS writes them as nonnegative scalars):
     * a slack column j of an original LP block that appears in exactly one constraint i
     * (coefficient a, cost 0) gives the row z_j + a y_i = 0, so y_i = -z_j / a >= 0 on the
     * right side; the pair of y_i and the row are dropped and z_j takes y_i's column / (-a).
     * On the AC-OPF moment forms this removes 5432 of 7019 free pairs (case793). */
    static int noelim = -1;
    if (noelim < 0) noelim = getenv("BRISK_MOM_NOELIM") != NULL;
    if (ok && nslack > 0 && !noelim) {
        int *cnt = calloc(nslack + 1, sizeof(int)), *who = malloc(sizeof(int) * (nslack + 1));
        double *wv = malloc(sizeof(double) * (nslack + 1)), *cs = calloc(nslack + 1, sizeof(double));
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            if (dof[k] >= 0 || B->type != BLK_LP) continue;
            for (int e = 0; e < B->C.nnz; e++) cs[slackof[k] + B->C.row[e]] += B->C.val[e];
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *S = &B->A[t];
                for (int e = 0; e < S->nnz; e++) {
                    const int sidx = slackof[k] + S->row[e];
                    if (S->val[e] == 0) continue;
                    cnt[sidx]++; who[sidx] = B->con[t]; wv[sidx] = S->val[e];
                }
            }
        }
        int *ey = malloc(sizeof(int) * (m0 + 1));           /* y_i -> slack, -1 */
        double *ea = malloc(sizeof(double) * (m0 + 1));
        for (int i = 0; i < m0; i++) ey[i] = -1;
        int ne = 0;
        for (int sidx = 0; sidx < nslack; sidx++)
            if (cnt[sidx] == 1 && cs[sidx] == 0 && ey[who[sidx]] < 0) { ey[who[sidx]] = sidx; ea[who[sidx]] = wv[sidx]; ne++; }
        if (ne > 0) {
            /* row of each slack (as built) */
            int *srow = malloc(sizeof(int) * (nslack + 1));
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                if (dof[k] >= 0 || B->type != BLK_LP) continue;
                for (int j = 0; j < B->n; j++) srow[slackof[k] + j] = M->brow0[k] + j;
            }
            char *drop = calloc(nrow + 1, 1);
            for (int i = 0; i < m0; i++) if (ey[i] >= 0) drop[srow[ey[i]]] = 1;
            M->rowmap = malloc(sizeof(int) * (nrow + 1));
            int nr2 = 0;
            for (int r0 = 0; r0 < nrow; r0++) M->rowmap[r0] = drop[r0] ? -1 : nr2++;
            M->ymap = malloc(sizeof(int) * (m0 + 1));
            int np = 0;
            for (int i = 0; i < m0; i++) M->ymap[i] = ey[i] >= 0 ? -1 : np++;
            M->np = np; M->ne = ne;
            M->e_i = malloc(sizeof(int) * ne); M->e_s = malloc(sizeof(int) * ne); M->e_row = malloc(sizeof(int) * ne);
            M->e_a = malloc(sizeof(double) * ne); M->e_b = malloc(sizeof(double) * ne);
            M->e_ptr = calloc(ne + 1, sizeof(int));
            int *eix = malloc(sizeof(int) * (m0 + 1));
            {
                int q = 0;
                for (int i = 0; i < m0; i++) if (ey[i] >= 0) {
                    eix[i] = q; M->e_i[q] = i; M->e_s[q] = ey[i]; M->e_a[q] = ea[i]; M->e_b[q] = P->b[i]; M->e_row[q] = srow[ey[i]]; q++;
                } else eix[i] = -1;
            }
            /* column scaling of the multipliers (what TSSOS's normal=true does to the
             * constraints): y_i = yhat_i / |A_i|_max, z_s = f zhat_s with f = |a| / |A_i|_max */
            static int noscale = -1;
            if (noscale < 0) noscale = getenv("BRISK_MOM_NOSCALE") != NULL;
            double *ysc = malloc(sizeof(double) * (m0 + 1));
            for (int i = 0; i < m0; i++) ysc[i] = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                for (int t = 0; t < B->ncon; t++) {
                    const SpSym *S = &B->A[t];
                    for (int e = 0; e < S->nnz; e++) ysc[B->con[t]] = fmax(ysc[B->con[t]], fabs(S->val[e]));
                }
            }
            for (int i = 0; i < m0; i++) ysc[i] = (noscale || !(ysc[i] > 0)) ? 1.0 : 1.0 / ysc[i];
            M->ysc = malloc(sizeof(double) * (m0 + 1));
            memcpy(M->ysc, ysc, sizeof(double) * m0);
            /* rewrite the triplets */
            size_t w = 0;
            for (size_t e = 0; e < T.n; e++) {
                int c = T.con[e];
                if (c >= 0 && drop[c]) continue;
                int nc = c >= 0 ? M->rowmap[c] : -1;
                if (T.blk[e] == tlp && T.i[e] >= off) {
                    int col = T.i[e] - off, neg = col >= m0;
                    const int i = neg ? col - m0 : col;
                    if (ey[i] >= 0) {
                        if (neg) continue;                    /* y = y+ - y-: substitute once */
                        const double v = -T.v[e] / ea[i] * fabs(ea[i]) * ysc[i];
                        const int sidx = ey[i];
                        if (c >= 0) M->e_ptr[eix[i] + 1]++;
                        T.con[w] = nc; T.blk[w] = tlp; T.i[w] = T.j[w] = sidx; T.v[w] = v; w++;
                        continue;
                    }
                    const int ni = off + (neg ? np : 0) + M->ymap[i];
                    T.con[w] = nc; T.blk[w] = tlp; T.i[w] = T.j[w] = ni; T.v[w] = T.v[e] * ysc[i]; w++;
                    continue;
                }
                T.con[w] = nc; T.blk[w] = T.blk[e]; T.i[w] = T.i[e]; T.j[w] = T.j[e]; T.v[w] = T.v[e]; w++;
            }
            T.n = w;
            /* columns of the eliminated y (rows != the dropped one, solved numbering) */
            for (int q = 0; q < ne; q++) M->e_ptr[q + 1] += M->e_ptr[q];
            M->e_r = malloc(sizeof(int) * (M->e_ptr[ne] + 1)); M->e_v = malloc(sizeof(double) * (M->e_ptr[ne] + 1));
            int *fill = malloc(sizeof(int) * (ne + 1));
            memcpy(fill, M->e_ptr, sizeof(int) * ne);
            for (size_t e = 0; e < T.n; e++) {
                if (T.blk[e] != tlp || T.i[e] >= off || T.con[e] < 0) continue;
                /* a slack column now carrying an eliminated y: its entries are -A_ir / a */
                const int sidx = T.i[e];
                const int i = (cnt[sidx] == 1 && cs[sidx] == 0) ? who[sidx] : -1;
                if (i < 0 || ey[i] != sidx) continue;
                const int q = eix[i];
                M->e_r[fill[q]] = T.con[e]; M->e_v[fill[q]] = -T.v[e] / (fabs(ea[i]) * ysc[i]) * ea[i]; fill[q]++;
            }
            free(fill); free(eix); free(drop); free(srow); free(ysc);
            /* bq to the new rows */
            double *bq2 = calloc(nr2 + 1, sizeof(double));
            for (int r0 = 0; r0 < nrow; r0++) if (M->rowmap[r0] >= 0) bq2[M->rowmap[r0]] = bq[r0];
            free(bq); bq = bq2;
            M->nrow = nrow;           /* as built (postsolve maps through rowmap) */
            nrow = nr2;
            bsz[tlp] = -(nslack + 2 * np);
        }
        free(cnt); free(who); free(wv); free(cs); free(ey); free(ea);
    }
    if (ok) problem_from_trips(Q, nrow, nbq, bsz, bq, T.n, T.con, T.blk, T.i, T.j, T.v);
    tl_free(&T);
    free(bsz); free(tblk); free(bq); free(slackof); free(cl0);
    for (int q = 0; q < nd; q++) { free(hs[q].key); free(hs[q].val); }
    free(hs); free(dof);
    if (!ok) { moment_record_free(M); memset(M, 0, sizeof(*M)); return 0; }
    return 1;
}

static void moment_record_free(PSMoment *M) {
    if (!M) return;
    free(M->rowmap); free(M->ymap); free(M->ysc); free(M->e_i); free(M->e_s); free(M->e_row); free(M->e_a); free(M->e_b);
    free(M->e_ptr); free(M->e_r); free(M->e_v);
    free(M->orig0); free(M->bs0); free(M->brow0);
    for (int q = 0; q < M->nd; q++) { free(M->dch[q].cptr); free(M->dch[q].cv); free(M->dch[q].par); free(M->dcrow[q]); free(M->dcoff[q]); }
    free(M->dpos); free(M->dch); free(M->dcrow); free(M->dcoff);
}
void ps_moment_free(PSMoment *M) { moment_record_free(M); free(M); }

static void problem_free_data(Problem *P) {
    for (int k = 0; k < P->nblk; k++) block_free_contents(&P->blk[k]);
    free(P->blk); free(P->b); free(P->b0); free(P->orig); free(P->d); free(P->du);
    P->blk = NULL; P->b = P->b0 = NULL; P->orig = NULL; P->d = NULL; P->du = NULL;
}

/* the moment form, the support-clique threshold chosen on the estimated iteration cost */
static int chordal_convert_moment(Problem *P, const Params *par, int verbose) {
    /* 4.34: every SDP block of at least chordal_minn indices, and every block above the
     * entrywise limit (64) that has a clique decomposition, is decomposed (until 4.33 one
     * block only, and any other block above 64 made the form inapplicable) */
    const int maxsmall = 64;
    int any = 0;
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP && P->blk[k].n >= par->chordal_minn) any = 1;
    if (!any) return 0;
    Params q = *par;
    q.chordal = 1;
    const double c0 = par->chordal > 0 ? 1e300 : iter_cost(P, NULL, 0);
    /* candidates (support-clique threshold, merge ratio). With the inequality multipliers
     * eliminated, the plain chordal extension without support cliques or merging gives
     * the TSSOS structure (case793: 1346 cliques, 8589 rows against TSSOS's 1352 / 8546)
     * and usually the cheapest iteration; support cliques are the alternative when single
     * constraints straddle many cliques. */
    const int cand_auto[2] = { 0, 4 };
    const double merge_auto[2] = { 1.01, 0.6 };
    const int *cand = par->chordal_sup >= 0 ? &par->chordal_sup : cand_auto;
    /* one candidate by default (it won on every AC-OPF case by 15-40%, and each estimate
     * costs a symbolic analysis); the support-clique form is the retry in main.c */
    const int ncand = par->chordal_sup >= 0 ? 1 : (getenv("BRISK_MOM_CAND2") ? 2 : 1);
    double best = 1e300; int bsup = -1, bmaxc = 0;
    SChol *best_sc = NULL;
    Problem Q; memset(&Q, 0, sizeof(Q));
    PSMoment *M = calloc(1, sizeof(PSMoment));
    const int nb = P->nblk;
    int *ord = malloc(sizeof(int) * (nb + 1)), nord = 0;
    for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP && (P->blk[k].n >= par->chordal_minn || P->blk[k].n > maxsmall)) ord[nord++] = k;
    for (int i = 1; i < nord; i++) { const int v = ord[i]; int j = i; while (j > 0 && P->blk[ord[j - 1]].n < P->blk[v].n) { ord[j] = ord[j - 1]; j--; } ord[j] = v; }   /* largest first */
    PSChordal *rs = calloc(nord + 1, sizeof(PSChordal));
    int *kcs = malloc(sizeof(int) * (nord + 1));
    for (int ci = 0; ci < ncand; ci++) {
        q.chordal_sup = cand[ci];
        if (par->chordal_sup < 0) q.chordal_merge = merge_auto[ci];
        int nd = 0, maxc = 0, ok = 1;
        for (int i = 0; i < nord && ok; i++) {
            const int k = ord[i]; int mc = 0;
            PSChordal r;
            if (!cliques_pack(&P->blk[k], &q, &r, &mc)) {
                if (verbose) printf("chordal (moment form): no clique decomposition of block %d (n = %d; clique limit %d)\n", k + 1, P->blk[k].n, q.chordal_maxclique);
                if (P->blk[k].n > maxsmall) ok = 0;       /* too large to be written entrywise */
                continue;
            }
            rs[nd] = r; kcs[nd] = k; nd++;
            if (mc > maxc) maxc = mc;
        }
        Problem Qc; PSMoment Mc;
        if (!ok || nd == 0 || !moment_build(P, nd, kcs, rs, &Qc, &Mc, maxsmall)) {
            if (verbose) printf("chordal (moment form): not in moment form (%d block(s) decomposed; a block above %d without a decomposition)\n", nd, maxsmall);
        } else {
            double spf = 0, tic = wtime();
            double bud = c0 < 1e300 ? par->chordal_gain2 * c0 : 0;
            if (best < 1e300 && (bud == 0 || best < bud)) bud = best;       /* cannot win: stop early */
            double c1 = (ncand == 1 && par->chordal > 0) ? 0 : iter_cost_keep(&Qc, &spf, bud, 1);
            if (verbose) printf("   (cost estimate %.3fs)\n", wtime() - tic);
            int ncl = 0; for (int d = 0; d < nd; d++) ncl += rs[d].ncl;
            if (verbose) printf("chordal (moment form): support cliques <= %d: %d block(s), %d cliques (max %d), m = %d, estimated cost %.2e per iteration (Schur factor %.1e flops; unconverted %.2e)\n",
                                cand[ci], nd, ncl, maxc, Qc.m, c1, spf, c0 < 1e300 ? c0 : -1.0);
            if (c1 < best) {            /* keep the best candidate (no rebuild) and its analysis */
                if (bsup >= 0) { problem_free_data(&Q); moment_record_free(M); }
                best = c1; bsup = cand[ci]; bmaxc = maxc; Q = Qc; *M = Mc;
                schol_free(best_sc); best_sc = g_keep_sc; g_keep_sc = NULL;
            } else { problem_free_data(&Qc); moment_record_free(&Mc); schol_free(g_keep_sc); g_keep_sc = NULL; }
        }
        for (int d = 0; d < nd; d++) { free(rs[d].cptr); free(rs[d].cv); free(rs[d].par); }
    }
    free(rs); free(kcs); free(ord);
    if (bsup < 0 || c0 < 0 || (par->chordal < 0 && best > par->chordal_gain2 * c0)) {
        if (bsup >= 0) { problem_free_data(&Q); moment_record_free(M); }
        free(M); schol_free(best_sc);
        return 0;
    }
    chordal_keep_analysis(best_sc, Q.m);
    M->cost0 = c0 < 1e300 ? c0 : -1.0; M->cost1 = best;
    const int maxc = bmaxc;
    if (verbose)
        for (int d = 0; d < M->nd; d++)
            printf("chordal (moment form): block %d (n = %d) -> %d cliques%s\n", M->dpos[d] + 1, P->blk[M->dpos[d]].n, M->dch[d].ncl, d == 0 ? "" : " (several blocks)");
    if (verbose) printf("chordal (moment form): %d block(s), max clique %d, %d moment rows, %d free pairs\n",
                        M->nd, maxc, Q.m, M->rowmap ? M->np : M->m0);
    const int kc = M->chpos;
    /* swap the new problem in; the postsolve record keeps what is needed to map back */
    struct Postsolve *ps = P->ps;
    const int morig = P->morig, frr = P->fr_removed;
    problem_free_data(P);
    *P = Q;
    P->ps = ps; P->morig = morig; P->fr_removed = frr;
    if (getenv("BRISK_DUMP_CLIQUES")) {
        FILE *f = fopen(getenv("BRISK_DUMP_CLIQUES"), "w");
        if (f) {
            const PSChordal *r = &M->ch;
            fprintf(f, "%d %d %d\n", kc, r->n, r->ncl);
            for (int t = 0; t < r->ncl; t++) {
                fprintf(f, "%d", r->cptr[t + 1] - r->cptr[t]);
                for (int q2 = r->cptr[t]; q2 < r->cptr[t + 1]; q2++) fprintf(f, " %d", r->cv[q2]);
                fprintf(f, "\n");
            }
            fclose(f);
        }
    }
    const int ndec = M->nd;
    if (ps) ps->mom = M; else ps_moment_free(M);
    return ndec;
}

int chordal_convert(Problem *P, const Params *par, int verbose) {
    if (par->chordal == 0) return 0;
    if (par->chordal_form == 1) {
        int r = chordal_convert_moment(P, par, verbose);
        if (r >= 0) return r;
    }
    return chordal_convert_domain(P, par, verbose);
}

/* The symbolic analysis of the chosen conversion's Schur pattern, handed to the solver so
 * it is not redone (0.3-0.7 s on the AC-OPF moment forms): taken once, if m matches. */
static SChol *g_cache_sc = NULL;
static int g_cache_m = -1;
void chordal_keep_analysis(SChol *S, int m) { schol_free(g_cache_sc); g_cache_sc = S; g_cache_m = S ? m : -1; }
SChol *chordal_take_analysis(int m, size_t patnnz) {
    SChol *S = NULL;
    if (g_cache_sc && g_cache_m == m && schol_mpat_nnz(g_cache_sc) == patnnz) S = g_cache_sc;
    else schol_free(g_cache_sc);
    g_cache_sc = NULL; g_cache_m = -1;
    return S;
}
