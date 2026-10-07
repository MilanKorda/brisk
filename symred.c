/* symred.c - BRISK 4.32/4.33: exact symmetry reduction from generators of the automorphism
 * group: constraint aggregation over the orbits (4.32) and the symmetry-adapted basis that
 * splits the blocks into their irreducible components (4.33, below).
 *
 * Input: a file with permutations of the SDP indices (global 0-based over the SDP blocks in
 * file order; blocks of equal size may be permuted; LP indices are fixed). The first line
 * lists the SDP block sizes (checked), each further line one generator.
 *
 * Each generator g must map the data onto itself: an entry (i, j) of A_c with value v goes to
 * an entry (g i, g j) of some A_c' with the same value, every entry of A_c goes to the same
 * c', b_c' = b_c, and C is invariant. The group G is generated (closure, bounded), the
 * constraints fall into orbits, and the problem is replaced by the aggregated one
 *
 *     <sum_{c in O} A_c, X> = sum_{c in O} b_c        for each orbit O,
 *
 * which has the same optimal value: a feasible X of the original is feasible here; a feasible
 * X here averaged over G (X_bar = |G|^-1 sum_g P_g X P_g') keeps the objective (C invariant),
 * stays feasible for the aggregate, and on an invariant matrix every constraint of an orbit
 * takes the same value, so X_bar satisfies each original constraint (b is orbit-constant).
 * The dual y_c = ybar_orbit(c) is feasible for the original with the same objective.
 *
 * Postsolve: y expanded over the orbits; X averaged over the group (block k index i goes to
 * block k' index i'); the free split pair is untouched. Example 8.1.3 (Klep-Magron-Volcic-
 * Wang) d = 4 with the S3 of the input relabellings: 64 878 -> 11 462 rows, 140 s to 5.3e-9
 * on two cores with the adapted basis (390 s without) where the unreduced problem does not
 * fit in memory.                                                                          */
#include "brisk.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
double wtime(void);

static void *sx(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (symmetry reduction)\n"); exit(1); } return p; }

typedef struct { uint64_t key; int con; double v; } SEnt;
static int cmp_sent(const void *a, const void *b) { const uint64_t x = ((const SEnt *)a)->key, y = ((const SEnt *)b)->key; return x < y ? -1 : x > y; }

static int64_t find_key(const SEnt *E, size_t ne, uint64_t key) {
    size_t lo = 0, hi = ne;
    while (lo < hi) { const size_t mid = (lo + hi) / 2; if (E[mid].key < key) lo = mid + 1; else hi = mid; }
    return (lo < ne && E[lo].key == key) ? (int64_t)lo : -1;
}


/* ---- automatic search for automorphisms (-sym auto) --------------------------------
 * Colour refinement on the index/constraint structure, then individualisation of index
 * vertices down to discrete colourings (leaves); a leaf's certificate is the hash of the data
 * relabelled by the colouring. Two leaves with equal certificates differ by a permutation
 * that is checked on the data before it is kept (so hash collisions and an incomplete search
 * are harmless: any subgroup of the automorphism group gives an exact reduction).           */
typedef struct {
    int N, NI, m, nsdp;
    size_t ne;
    int *ea, *eb, *ec; double *ev; unsigned char *ed;   /* entries a <= b, constraint node c (>= NI), value, diagonal */
    int *adjp, *adj;                /* CSR: node -> entry ids */
    int *col, *ncol;                /* colourings */
    uint64_t *hk; int *ord;         /* scratch */
    long nodes, maxnodes;
    double t0, tmax;
    int *gens, *gcps; int ngen, gcap;   /* found generators, NI each, with their constraint permutations (m + 1 each) */
    int *ufp;                       /* union-find over index nodes (orbits of the found group) */
    uint64_t cert_ref; int *leaf_ref, *inv_ref;
    int verbose;
    int absval;                     /* 4.37: the search on |data| (signed permutations: the signs are lifted afterwards) */
} ASearch;

static uint64_t h64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
static uint64_t hval(double v) { union { double d; uint64_t u; } w; w.d = v; return h64(w.u + 0x1234567ull); }
static int cmp_u64(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
typedef struct { uint64_t key; int node; } KN;
static int cmp_kn(const void *a, const void *b) { const KN *x = a, *y = b; return x->key < y->key ? -1 : x->key > y->key ? 1 : x->node - y->node; }

/* refine col in place to a stable colouring (colours 0..K-1 ranked canonically); returns K */
static int as_refine(ASearch *A, int *col) {
    const int N = A->N;
    KN *kn = sx(sizeof(KN) * N);
    int K = 0;
    { uint64_t *tmp = sx(sizeof(uint64_t) * N); for (int u = 0; u < N; u++) tmp[u] = col[u]; qsort(tmp, N, sizeof(uint64_t), cmp_u64); for (int u = 0; u < N; u++) if (u == 0 || tmp[u] != tmp[u - 1]) K++; free(tmp); }
    for (int round = 0; round < 1000; round++) {
        /* per node: sorted list of item hashes */
        for (int u = 0; u < N; u++) {
            const int d0 = A->adjp[u], d1 = A->adjp[u + 1];
            uint64_t *hk = A->hk;
            for (int q = d0; q < d1; q++) {
                const int e = A->adj[q];
                const int a = A->ea[e], b = A->eb[e], c = A->ec[e];
                uint64_t h;
                if (u == c) { const uint64_t ca = col[a], cb = col[b]; h = h64((ca < cb ? ca : cb) * 1000003ull + (ca < cb ? cb : ca)) ^ hval(A->ev[e]) ^ (A->ed[e] ? 0x5bd1e995ull : 0); }
                else { const int o = (u == a) ? b : a; h = h64((uint64_t)col[o] * 7919ull + (c >= 0 ? (uint64_t)col[c] * 104729ull + 17 : 0x3c6ef372fe94f82bull)) ^ hval(A->ev[e]) ^ (A->ed[e] ? 0x5bd1e995ull : 0); }
                hk[q - d0] = h;
            }
            /* order-independent combination (a sum of hashes): linear in the degree, where a
             * sorted chain cost d log d per node and round (a 500 000-entry constraint made the
             * search of equalG51 miss its time cap by minutes) */
            uint64_t acc = 1469598103934665603ull + (uint64_t)(d1 - d0) * 0x9e3779b97f4a7c15ull;
            for (int q = 0; q < d1 - d0; q++) acc += h64(hk[q] + 0x632be59bd9b4e019ull);
            kn[u].key = ((uint64_t)col[u] << 40) ^ (h64(acc) >> 24);   /* old colour first: refinement, not re-colouring */
            kn[u].node = u;
        }
        qsort(kn, N, sizeof(KN), cmp_kn);
        int K2 = 0;
        for (int q = 0; q < N; q++) { if (q > 0 && kn[q].key != kn[q - 1].key) K2++; A->ncol[kn[q].node] = K2; }
        K2++;
        memcpy(col, A->ncol, sizeof(int) * N);
        if (K2 == K) break;
        K = K2;
    }
    free(kn);
    return K;
}
static int as_individualize(ASearch *A, const int *col, int v, int *out) {
    const int N = A->N;
    for (int u = 0; u < N; u++) out[u] = 2 * col[u] + (col[u] == col[v] && u != v ? 1 : 0);
    return as_refine(A, out);
}
/* target cell: the smallest non-singleton cell among the index nodes, ties by the colour
 * (a canonical choice: colours are canonical ranks, so the same cell is chosen in every
 * branch related by an automorphism); returns its size, members in cell[] */
static int as_target(const ASearch *A, const int *col, int *cell) {
    const int NI = A->NI;
    int *cnt = calloc(A->N, sizeof(int));
    for (int u = 0; u < NI; u++) cnt[col[u]]++;
    int best = -1;
    for (int u = 0; u < NI; u++) { const int c = col[u]; if (cnt[c] > 1 && (best < 0 || cnt[c] < cnt[best] || (cnt[c] == cnt[best] && c < best))) best = c; }
    int n = 0;
    if (best >= 0) for (int u = 0; u < NI; u++) if (col[u] == best) cell[n++] = u;
    free(cnt);
    return n;
}
static uint64_t as_cert(const ASearch *A, const int *col) {
    uint64_t *hk = sx(sizeof(uint64_t) * A->ne);
    for (size_t e = 0; e < A->ne; e++) {
        uint64_t pa = col[A->ea[e]], pb = col[A->eb[e]], pc = A->ec[e] >= 0 ? (uint64_t)col[A->ec[e]] : 0xfffffffull;
        if (pa > pb) { const uint64_t t = pa; pa = pb; pb = t; }
        hk[e] = h64(pa * 1000003ull + pb) ^ h64(pc * 31ull + 7) ^ hval(A->ev[e]);
    }
    uint64_t acc = 1469598103934665603ull;
    for (size_t e = 0; e < A->ne; e++) acc += h64(hk[e] + 0x632be59bd9b4e019ull);
    free(hk);
    return h64(acc);
}
static int uf_find(int *p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }

/* union-find with parity (moved up in 4.37: the signed symmetries use it too) */
typedef struct { int *p, *par; } PUF;
static int cmp_int(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return x < y ? -1 : x > y; }
static int puf_find(PUF *U, int x, int *parity) {
    int r = x, q = 0; while (U->p[r] != r) { q ^= U->par[r]; r = U->p[r]; }
    int y = x, acc = q;                       /* compress the path, keeping each node's parity to the root */
    while (y != r && U->p[y] != r) { const int nx = U->p[y], py = U->par[y]; U->p[y] = r; U->par[y] = acc; acc ^= py; y = nx; }
    *parity = q; return r;
}
/* union with s_a + s_b = d; returns 0 on contradiction */
static int puf_union(PUF *U, int a, int b2, int d) {
    int pa, pb; const int ra = puf_find(U, a, &pa), rb = puf_find(U, b2, &pb);
    if (ra == rb) return ((pa ^ pb) == d);
    U->p[ra] = rb; U->par[ra] = pa ^ pb ^ d; return 1;
}

/* ---- 4.37: signed permutations -------------------------------------------------------------
 * A signed permutation P e_i = s_i e_{g(i)} (s_i = +-1; +1 on LP indices, where -1 would not map
 * the orthant onto itself) is an automorphism of the data when P A_c P' = e_c A_{cp(c)} for every
 * constraint, with b_{cp(c)} = e_c b_c, and P C P' = C; entrywise (A_{cp(c)})_{g i, g j} =
 * s_i s_j e_c (A_c)_{ij}. For an invariant X (P X P' = X) the constraints of an orbit agree up
 * to their signs, and a constraint the group maps to its own negative vanishes (b_c = 0 then).
 * Example 8.1.3: the exchange of the parties A and C with the outputs of both flipped.
 *
 * The search runs on |data| (an automorphism of the data is one of |data|); each candidate
 * (g, cp) is lifted by solving for the signs over GF(2):
 *     sgn(v') + sgn(v) = s_i + s_j + e_c   for every entry (i, j, c, v) -> (g i, g j, cp c, v'),
 * e = 0 for C, e_c fixed by the signs of b_c and b_{cp c} when b_c != 0, s = 0 on LP indices.
 * The system is the sign stage's with a right-hand side: union-find with parity for the
 * two-variable equations (C, diagonal entries, constraints whose e_c is known), rounds (an entry
 * inside one component fixes e_c), then a sparse elimination with right-hand sides over the
 * components for the rest; free variables 0; every equation is checked on the result.       */
typedef struct { int *c; int n; int rhs; } LRow;
static int sign_lift(int NI, int m, const char *islp, const SEnt *E, size_t ne, const int *g, const int *cp, const double *b, signed char *sg, signed char *ce) {
    unsigned char *pq = sx(ne + 1);
    for (size_t q = 0; q < ne; q++) {
        const int i = (int)(E[q].key >> 32), j = (int)(E[q].key & 0xffffffffu);
        int a = g[i], bb = g[j]; if (a > bb) { const int t = a; a = bb; bb = t; }
        const uint64_t key = ((uint64_t)a << 32) | (uint64_t)bb;
        const int c2 = E[q].con < 0 ? -1 : cp[E[q].con];
        int found = 0;
        for (int64_t pp = find_key(E, ne, key); pp >= 0 && pp < (int64_t)ne && E[pp].key == key; pp++)
            if (E[pp].con == c2 && fabs(fabs(E[pp].v) - fabs(E[q].v)) <= 1e-12 * (1 + fabs(E[q].v))) { pq[q] = (E[pp].v < 0) != (E[q].v < 0); found = 1; break; }
        if (!found) { free(pq); return 0; }
    }
    /* the entries of each constraint (CSR) */
    int *ptr = calloc((size_t)m + 2, sizeof(int)); size_t nce = 0;
    for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) { ptr[E[q].con + 1]++; nce++; }
    for (int c = 0; c < m; c++) ptr[c + 1] += ptr[c];
    size_t *ent = sx(sizeof(size_t) * (nce + 1));
    { int *fill = sx(sizeof(int) * (m + 1)); memcpy(fill, ptr, sizeof(int) * m); for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) ent[fill[E[q].con]++] = q; free(fill); }
    #define EI(q) ((int)(E[q].key >> 32))
    #define EJ(q) ((int)(E[q].key & 0xffffffffu))
    const int ZERO = NI;
    PUF U; U.p = sx(sizeof(int) * (NI + 1)); U.par = calloc(NI + 1, sizeof(int)); for (int i = 0; i <= NI; i++) U.p[i] = i;
    int ok = 1;
    for (int i = 0; i < NI && ok; i++) if (islp[i]) ok = puf_union(&U, i, ZERO, 0);
    int *e = sx(sizeof(int) * (m + 1));
    for (int c = 0; c < m; c++) e[c] = -1;
    for (int c = 0; c < m && ok; c++) if (b[c] != 0) { if (fabs(fabs(b[cp[c]]) - fabs(b[c])) > 1e-12 * (1 + fabs(b[c]))) ok = 0; e[c] = (b[cp[c]] < 0) != (b[c] < 0); }
    for (size_t q = 0; q < ne && ok; q++) {
        if (E[q].con < 0) ok = puf_union(&U, EI(q), EJ(q), pq[q]);          /* C: s_i + s_j = p */
        else if (EI(q) == EJ(q)) { const int c = E[q].con; if (e[c] < 0) e[c] = pq[q]; else if ((e[c] & 1) != pq[q]) ok = 0; }   /* diagonal / LP: e_c = p */
    }
    int changed = 1, rounds = 0;
    while (changed && ok && rounds < 1000) {
        changed = 0; rounds++;
        for (int c = 0; c < m && ok; c++) {
            if (e[c] < 0)
                for (int w = ptr[c]; w < ptr[c + 1]; w++) { const size_t q = ent[w]; int pi, pj; if (puf_find(&U, EI(q), &pi) == puf_find(&U, EJ(q), &pj)) { e[c] = pq[q] ^ pi ^ pj; break; } }
            if (e[c] >= 0 && !(e[c] & 2)) {
                for (int w = ptr[c]; w < ptr[c + 1] && ok; w++) { const size_t q = ent[w]; ok = puf_union(&U, EI(q), EJ(q), pq[q] ^ (e[c] & 1)); }
                e[c] |= 2; changed = 1;
            }
        }
    }
    int *root = sx(sizeof(int) * (NI + 2)), *rpar = sx(sizeof(int) * (NI + 2)), *rid = sx(sizeof(int) * (NI + 2)); int K = 0;
    int *x = NULL;
    if (ok) {
        for (int i = 0; i <= NI; i++) { root[i] = puf_find(&U, i, &rpar[i]); rid[i] = -1; }
        for (int i = 0; i <= NI; i++) if (rid[root[i]] < 0) rid[root[i]] = K++;
        /* sparse elimination with right-hand sides over the K component variables */
        LRow *prow = calloc((size_t)K + 1, sizeof(LRow)); int npiv = 0; int *pivof = sx(sizeof(int) * (K + 1)); for (int r = 0; r < K; r++) pivof[r] = -1;
        int *wa = sx(sizeof(int) * (K + 8)), *wb = sx(sizeof(int) * (K + 8)); size_t fill = 0; const size_t fillmax = (size_t)50 << 20;
        #define LADD(cols_, n_, rhs_) do { \
            int wn = (n_), rh = (rhs_); memcpy(wa, (cols_), sizeof(int) * wn); qsort(wa, wn, sizeof(int), cmp_int); \
            { int w2 = 0; for (int q_ = 0; q_ < wn; q_++) { if (w2 && wa[w2 - 1] == wa[q_]) w2--; else wa[w2++] = wa[q_]; } wn = w2; } \
            while (wn > 0 && pivof[wa[0]] >= 0) { const LRow *R_ = &prow[pivof[wa[0]]]; int i1 = 0, i2 = 0, w2 = 0; rh ^= R_->rhs; \
                while (i1 < wn || i2 < R_->n) { if (i2 >= R_->n || (i1 < wn && wa[i1] < R_->c[i2])) wb[w2++] = wa[i1++]; else if (i1 >= wn || R_->c[i2] < wa[i1]) wb[w2++] = R_->c[i2++]; else { i1++; i2++; } } \
                memcpy(wa, wb, sizeof(int) * w2); wn = w2; } \
            if (wn == 0) { if (rh) ok = 0; } \
            else { prow[npiv].c = sx(sizeof(int) * wn); memcpy(prow[npiv].c, wa, sizeof(int) * wn); prow[npiv].n = wn; prow[npiv].rhs = rh; pivof[wa[0]] = npiv++; fill += wn; if (fill > fillmax) ok = 0; } \
        } while (0)
        { int cc[1] = { rid[root[ZERO]] }; LADD(cc, 1, rpar[ZERO]); }            /* x_root(ZERO) + parity = 0 */
        for (int c = 0; c < m && ok; c++) {
            if (e[c] >= 0 || ptr[c + 1] - ptr[c] < 2) continue;
            const size_t q0 = ent[ptr[c]]; const int i0 = EI(q0), j0 = EJ(q0);
            for (int w = ptr[c] + 1; w < ptr[c + 1] && ok; w++) {
                const size_t q = ent[w]; const int i = EI(q), j = EJ(q);
                const int cc[4] = { rid[root[i]], rid[root[j]], rid[root[i0]], rid[root[j0]] };
                LADD(cc, 4, pq[q] ^ pq[q0] ^ rpar[i] ^ rpar[j] ^ rpar[i0] ^ rpar[j0]);
            }
        }
        #undef LADD
        if (ok) {
            /* back-substitution from the largest pivot column: the other columns of a row are larger */
            x = calloc(K + 1, sizeof(int));
            KN *kn = sx(sizeof(KN) * (npiv + 1)); for (int pv = 0; pv < npiv; pv++) { kn[pv].key = (uint64_t)prow[pv].c[0]; kn[pv].node = pv; }
            qsort(kn, npiv, sizeof(KN), cmp_kn);
            for (int o = npiv - 1; o >= 0; o--) { const LRow *R = &prow[kn[o].node]; int v = R->rhs; for (int q = 1; q < R->n; q++) v ^= x[R->c[q]]; x[R->c[0]] = v; }
            free(kn);
        }
        for (int pv = 0; pv < npiv; pv++) free(prow[pv].c);
        free(prow); free(pivof); free(wa); free(wb);
    }
    if (ok) {
        for (int i = 0; i < NI; i++) sg[i] = (x[rid[root[i]]] ^ rpar[i]) ? -1 : 1;
        for (int c = 0; c < m; c++) {
            int ec = e[c] >= 0 ? (e[c] & 1) : 0;
            if (e[c] < 0 && ptr[c + 1] > ptr[c]) { const size_t q0 = ent[ptr[c]]; ec = pq[q0] ^ (sg[EI(q0)] < 0) ^ (sg[EJ(q0)] < 0); }
            ce[c] = ec ? -1 : 1;
        }
        /* the check, entry by entry */
        for (size_t q = 0; q < ne && ok; q++) {
            const int ec = E[q].con < 0 ? 0 : ce[E[q].con] < 0;
            if ((int)pq[q] != (((sg[EI(q)] < 0) ^ (sg[EJ(q)] < 0) ^ ec) & 1)) ok = 0;
        }
        for (int i = 0; i < NI && ok; i++) if (islp[i] && sg[i] < 0) ok = 0;
        for (int c = 0; c < m && ok; c++) if (b[c] != 0 && ((b[cp[c]] < 0) != (b[c] < 0)) != (ce[c] < 0)) ok = 0;
    }
    #undef EI
    #undef EJ
    free(x); free(root); free(rpar); free(rid); free(e); free(U.p); free(U.par); free(ptr); free(ent); free(pq);
    return ok;
}

/* the constraint map (cp, ce) of a signed index permutation (g, sg; sg NULL: +1), with the
 * automorphism check: every entry (i, j, c, v) has the image (g i, g j, cp c, s_i s_j ce_c v),
 * C maps onto C, b_{cp c} = ce_c b_c, cp a bijection. Returns NULL or the reason it fails. */
static const char *sym_induced(int m, const SEnt *E, size_t ne, const int *cnt, const double *b, const int *g, const signed char *sg, int *cp, signed char *ce) {
    int *ptr = calloc((size_t)m + 2, sizeof(int));
    for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) ptr[E[q].con + 1]++;
    for (int c = 0; c < m; c++) ptr[c + 1] += ptr[c];
    size_t *ent = sx(sizeof(size_t) * ((size_t)ptr[m] + 1));
    { int *fill = sx(sizeof(int) * (m + 1)); memcpy(fill, ptr, sizeof(int) * m); for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) ent[fill[E[q].con]++] = q; free(fill); }
    char *used = calloc((size_t)m + 1, 1);
    const char *why = NULL;
    #define SG(i) (sg ? (double)sg[i] : 1.0)
    /* the image of one entry (value w expected at constraint c2): found? */
    #define HAS(i_, j_, c2_, w_, found_) do { int a_ = g[i_], b_ = g[j_]; if (a_ > b_) { const int t_ = a_; a_ = b_; b_ = t_; } \
        const uint64_t k_ = ((uint64_t)a_ << 32) | (uint64_t)b_; (found_) = 0; \
        for (int64_t pp_ = find_key(E, ne, k_); pp_ >= 0 && pp_ < (int64_t)ne && E[pp_].key == k_; pp_++) \
            if (E[pp_].con == (c2_) && fabs(E[pp_].v - (w_)) <= 1e-12 * (1 + fabs(w_))) { (found_) = 1; break; } } while (0)
    for (size_t q = 0; q < ne && !why; q++) if (E[q].con < 0) { const int i = (int)(E[q].key >> 32), j = (int)(E[q].key & 0xffffffffu); int f; HAS(i, j, -1, SG(i) * SG(j) * E[q].v, f); if (!f) why = "C is not invariant"; }
    for (int c = 0; c < m && !why; c++) {
        if (ptr[c + 1] == ptr[c]) { why = "a constraint without entries"; break; }
        const size_t q0 = ent[ptr[c]]; const int i0 = (int)(E[q0].key >> 32), j0 = (int)(E[q0].key & 0xffffffffu);
        int a = g[i0], bb = g[j0]; if (a > bb) { const int t = a; a = bb; bb = t; }
        const uint64_t key = ((uint64_t)a << 32) | (uint64_t)bb;
        int done = 0;
        for (int64_t pp = find_key(E, ne, key); !done && pp >= 0 && pp < (int64_t)ne && E[pp].key == key; pp++) {
            const int c2 = E[pp].con; if (c2 < 0 || used[c2] || cnt[c2] != cnt[c]) continue;
            const double w0 = SG(i0) * SG(j0) * E[q0].v; if (fabs(fabs(E[pp].v) - fabs(w0)) > 1e-12 * (1 + fabs(w0))) continue;
            const double e2 = (E[pp].v < 0) == (w0 < 0) ? 1.0 : -1.0;
            if (fabs(b[c2] - e2 * b[c]) > 1e-12 * (1 + fabs(b[c]))) continue;
            int all = 1;
            for (int w = ptr[c] + 1; w < ptr[c + 1] && all; w++) { const size_t q = ent[w]; const int i = (int)(E[q].key >> 32), j = (int)(E[q].key & 0xffffffffu); HAS(i, j, c2, e2 * SG(i) * SG(j) * E[q].v, all); }
            if (all) { cp[c] = c2; ce[c] = e2 < 0 ? -1 : 1; used[c2] = 1; done = 1; }
        }
        if (!done) why = "a constraint does not map onto another one";
    }
    #undef HAS
    #undef SG
    free(used); free(ptr); free(ent);
    return why;
}

/* a candidate permutation g of the index nodes with its constraint permutation cp: is it an
 * automorphism? Every entry (i, j, c, v) must have the entry (g i, g j, cp c, v) in the data
 * (C to C), with b and the entry counts invariant. */
static int as_verify(const ASearch *A, const int *g, const int *cp, const double *b, const SEnt *E, size_t ne, const int *cnt) {
    const int m = A->m;
    for (size_t q = 0; q < ne; q++) {
        const int i = (int)(E[q].key >> 32), j = (int)(E[q].key & 0xffffffffu);
        int a = g[i], bb = g[j]; if (a > bb) { const int t = a; a = bb; bb = t; }
        int64_t p = find_key(E, ne, ((uint64_t)a << 32) | (uint64_t)bb);
        if (p < 0) return 0;
        const int c2 = E[q].con < 0 ? -1 : cp[E[q].con];
        int found = 0;
        for (int64_t pp = p; pp < (int64_t)ne && E[pp].key == (((uint64_t)a << 32) | (uint64_t)bb); pp++)
            if (E[pp].con == c2 && (A->absval ? fabs(fabs(E[pp].v) - fabs(E[q].v)) : fabs(E[pp].v - E[q].v)) <= 1e-12 * (1 + fabs(E[q].v))) { found = 1; break; }
        if (!found) return 0;
    }
    for (int c = 0; c < m; c++) if (cp[c] < 0 || cp[c] >= m || cnt[c] != cnt[cp[c]] || (A->absval ? fabs(fabs(b[c]) - fabs(b[cp[c]])) : fabs(b[c] - b[cp[c]])) > 1e-12 * (1 + fabs(b[c]))) return 0;
    { char *seen = calloc((size_t)m + 1, 1); for (int c = 0; c < m; c++) { if (seen[cp[c]]) { free(seen); return 0; } seen[cp[c]] = 1; } free(seen); }
    return 1;
}
/* the constraint permutation of a leaf relative to the reference leaf: constraints paired in
 * the order of their colours (constraints of one colour at a discrete leaf are identical) */
static void as_conperm(const ASearch *A, const int *col, int *cp) {
    const int m = A->m, NI = A->NI;
    KN *a = sx(sizeof(KN) * m), *r = sx(sizeof(KN) * m);
    for (int c = 0; c < m; c++) { a[c].key = (uint64_t)col[NI + c]; a[c].node = c; r[c].key = (uint64_t)A->leaf_ref[NI + c]; r[c].node = c; }
    qsort(a, m, sizeof(KN), cmp_kn); qsort(r, m, sizeof(KN), cmp_kn);
    for (int q = 0; q < m; q++) cp[a[q].node] = r[q].node;
    free(a); free(r);
}
static void as_dfs(ASearch *A, const int *col, int depth, const double *b, const SEnt *E, size_t ne, const int *cnt, int *path) {
    if (A->nodes > A->maxnodes || (wtime() - A->t0 > A->tmax || brisk_stop_flag)) return;
    int *cell = sx(sizeof(int) * A->NI);
    const int nc = as_target(A, col, cell);
    if (nc == 0) {
        /* leaf */
        if (A->leaf_ref) {
            if (as_cert(A, col) == A->cert_ref) {
                int *g = sx(sizeof(int) * A->NI), *cp = sx(sizeof(int) * (A->m + 1));
                for (int u = 0; u < A->NI; u++) g[u] = A->inv_ref[col[u]];
                as_conperm(A, col, cp);
                int ident = 1; for (int u = 0; u < A->NI && ident; u++) if (g[u] != u) ident = 0;
                if (!ident && as_verify(A, g, cp, b, E, ne, cnt)) {
                    if (A->ngen == A->gcap) { A->gcap *= 2; A->gens = realloc(A->gens, sizeof(int) * (size_t)A->gcap * A->NI); A->gcps = realloc(A->gcps, sizeof(int) * (size_t)A->gcap * (A->m + 1)); }
                    memcpy(A->gens + (size_t)A->ngen * A->NI, g, sizeof(int) * A->NI);
                    memcpy(A->gcps + (size_t)A->ngen * (A->m + 1), cp, sizeof(int) * A->m); A->ngen++;
                    for (int u = 0; u < A->NI; u++) { const int x = uf_find(A->ufp, u), y = uf_find(A->ufp, g[u]); if (x != y) A->ufp[x] = y; }
                    if (A->verbose > 1) printf("   [symmetry search: automorphism at depth %d]\n", depth);
                }
                free(g); free(cp);
            }
        }
        free(cell);
        return;
    }
    int *child = sx(sizeof(int) * A->N);
    /* orbit pruning with the generators that fix the path pointwise (a subgroup of the
     * stabilizer of this node: pruning by its orbits is valid and conservative) */
    int *uf = sx(sizeof(int) * A->NI);
    for (int q = 0; q < nc; q++) {
        const int w = cell[q];
        if (q > 0 && (A->nodes > A->maxnodes || (wtime() - A->t0 > A->tmax || brisk_stop_flag))) break;
        if (q > 0) {
            for (int u = 0; u < A->NI; u++) uf[u] = u;
            for (int s2 = 0; s2 < A->ngen; s2++) {
                const int *g = A->gens + (size_t)s2 * A->NI;
                int fixes = 1; for (int d = 0; d < depth && fixes; d++) if (g[path[d]] != path[d]) fixes = 0;
                if (!fixes) continue;
                for (int u = 0; u < A->NI; u++) { const int x = uf_find(uf, u), y = uf_find(uf, g[u]); if (x != y) uf[x] = y; }
            }
            int skip = 0;
            for (int r = 0; r < q && !skip; r++) if (uf_find(uf, cell[r]) == uf_find(uf, w)) skip = 1;
            if (skip) continue;
        }
        as_individualize(A, col, w, child); A->nodes++;
        path[depth] = w;
        if (!A->leaf_ref && q == 0) {
            /* the reference leaf: leftmost path */
            int *cur = sx(sizeof(int) * A->N); memcpy(cur, child, sizeof(int) * A->N);
            int *cc = sx(sizeof(int) * A->NI);
            for (;;) { const int k = as_target(A, cur, cc); if (k == 0) break; if (A->nodes > A->maxnodes || (wtime() - A->t0 > A->tmax || brisk_stop_flag)) { A->nodes = A->maxnodes + 1; break; } int *nx = sx(sizeof(int) * A->N); as_individualize(A, cur, cc[0], nx); A->nodes++; memcpy(cur, nx, sizeof(int) * A->N); free(nx); }
            if (A->nodes > A->maxnodes) { free(cur); free(cc); break; }   /* capped inside the reference descent: no leaf, no automorphisms */
            A->leaf_ref = cur; A->cert_ref = as_cert(A, cur);
            A->inv_ref = sx(sizeof(int) * A->N); for (int u = 0; u < A->N; u++) A->inv_ref[u] = -1;
            for (int u = 0; u < A->NI; u++) A->inv_ref[cur[u]] = u;
            free(cc);
        }
        as_dfs(A, child, depth + 1, b, E, ne, cnt, path);
        if (A->nodes > A->maxnodes || (wtime() - A->t0 > A->tmax || brisk_stop_flag)) break;
    }
    free(child); free(cell); free(uf);
}

/* returns generators (ngen x NI, malloc) of a subgroup of the automorphism group */
/* 5.5: the gain of a reduction in work when the dense Schur complement dominates: with few
 * blocks and m far above their orders an iteration costs m^3/3 (the factorization) plus about
 * 30 sum n^3 (the block operations), and constraints shrinking by 1.2 are worth 1.7 there
 * (a moment relaxation of order 2 in 15 variables: m = 3,739, n = 135). Zero when the rule
 * does not apply (many blocks: the Schur complement is sparse and m^3 is not its cost). */
static double sym_cost_gain(const Problem *P, int m, int mr, double c0, double c1) {
    int nsdp = 0;
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) nsdp++;
    if (nsdp > 4 || m > 20000 || mr <= 0 || getenv("BRISK_NOSYMCOST")) return 0.0;
    const double t0 = (double)m * m * m / 3.0 + 30.0 * c0, t1 = (double)mr * mr * mr / 3.0 + 30.0 * c1;
    return t0 / fmax(t1, 1.0);
}
static int *sym_auto(const Problem *P, const int *off, int NI, int nsdp, const SEnt *E, size_t ne, int *ngen_out, int **gcps_out, int verbose, double tmax, long maxnodes, double symmin, int absval) {
    const int m = P->m, nb = P->nblk;
    ASearch A; memset(&A, 0, sizeof(A)); A.absval = absval;
    A.N = NI + m; A.NI = NI; A.m = m; A.nsdp = nsdp; A.verbose = verbose; A.t0 = wtime(); A.tmax = tmax; A.maxnodes = maxnodes;
    /* entries (constraints only) for the refinement */
    size_t nc = 0; for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) nc++;
    A.ne = ne;
    A.ea = sx(sizeof(int) * ne); A.eb = sx(sizeof(int) * ne); A.ec = sx(sizeof(int) * ne); A.ev = sx(sizeof(double) * ne); A.ed = sx(ne);
    for (size_t q = 0; q < ne; q++) { A.ea[q] = (int)(E[q].key >> 32); A.eb[q] = (int)(E[q].key & 0xffffffffu); A.ec[q] = E[q].con >= 0 ? NI + E[q].con : -1; A.ev[q] = absval ? fabs(E[q].v) : E[q].v; A.ed[q] = A.ea[q] == A.eb[q]; }
    /* adjacency: constraint entries (index-index-constraint) and C entries (index-index) */
    A.adjp = calloc(A.N + 1, sizeof(int));
    for (size_t q = 0; q < ne; q++) { A.adjp[A.ea[q] + 1]++; if (A.eb[q] != A.ea[q]) A.adjp[A.eb[q] + 1]++; if (A.ec[q] >= 0) A.adjp[A.ec[q] + 1]++; }
    for (int u = 0; u < A.N; u++) A.adjp[u + 1] += A.adjp[u];
    A.adj = sx(sizeof(int) * (A.adjp[A.N] + 1));
    { int *fill = sx(sizeof(int) * A.N); memcpy(fill, A.adjp, sizeof(int) * A.N);
      for (size_t q = 0; q < ne; q++) { A.adj[fill[A.ea[q]]++] = (int)q; if (A.eb[q] != A.ea[q]) A.adj[fill[A.eb[q]]++] = (int)q; if (A.ec[q] >= 0) A.adj[fill[A.ec[q]]++] = (int)q; }
      free(fill); }
    int maxdeg = 0; for (int u = 0; u < A.N; u++) if (A.adjp[u + 1] - A.adjp[u] > maxdeg) maxdeg = A.adjp[u + 1] - A.adjp[u];
    A.hk = sx(sizeof(uint64_t) * (maxdeg + 1)); A.ncol = sx(sizeof(int) * A.N);
    /* initial colouring: index nodes by (block size, LP, C entries), constraints by b */
    int *col = sx(sizeof(int) * A.N);
    {   KN *kn = sx(sizeof(KN) * A.N);
        uint64_t *ch = calloc(NI, sizeof(uint64_t));
        /* 5.9: a sum over the entries, the same for both indices of an off-diagonal entry. (The
         * chained hash with a different term for the larger index gave an index a colour that
         * depended on how many of its C entries pair it with a smaller index: with a dense C
         * every index had its own colour and no permutation was ever found - the theta problem
         * of a cycle, or any problem with C = J.) */
        for (size_t q = 0; q < ne; q++) if (E[q].con < 0) { const uint64_t h = h64(hval(A.ev[q]) + (A.ea[q] == A.eb[q] ? 0x1111ull : 0x77ull)); ch[A.ea[q]] += h; if (A.eb[q] != A.ea[q]) ch[A.eb[q]] += h; }
        for (int k = 0; k < nb; k++) for (int i = off[k]; i < off[k + 1]; i++) { kn[i].key = h64((uint64_t)P->blk[k].n * 3 + (P->blk[k].type == BLK_LP)) ^ ch[i]; kn[i].node = i; }
        for (int c = 0; c < m; c++) { kn[NI + c].key = 0x8000000000000000ull | (hval(absval ? fabs(P->b[c]) : P->b[c]) >> 1); kn[NI + c].node = NI + c; }
        qsort(kn, A.N, sizeof(KN), cmp_kn);
        int K = 0; for (int q = 0; q < A.N; q++) { if (q > 0 && kn[q].key != kn[q - 1].key) K++; col[kn[q].node] = K; }
        free(kn); free(ch);
    }
    const int K0 = as_refine(&A, col);
    int ncls = 0; { int *seen = calloc(A.N, sizeof(int)); for (int u = 0; u < NI; u++) if (!seen[col[u]]++) ncls++; free(seen); }
    if (verbose > 0) printf("   symmetry search: refinement %d classes (%d among the %d indices), %.2fs\n", K0, ncls, NI, wtime() - A.t0);
    /* automorphisms preserve the colours, so the orbits refine the classes: the constraint
     * orbits number at least the constraint classes, and the trivial component of a block at
     * least its index classes. When neither bound leaves a gain of symmin, no search */
    if (symmin > 1) {
        int *seen = calloc(A.N, sizeof(int)); int ccls = 0;
        for (int c = 0; c < m; c++) if (!seen[col[NI + c]]++) ccls++;
        int can = (double)m / ccls >= symmin;
        /* 4.34: the whole cube sum (a block's trivial component alone has at least its number
         * of index classes): sum n^3 / sum classes^3, not the best single block */
        double c0 = 0, c1 = 0;
        for (int k = 0; k < nb && !can; k++) {
            if (P->blk[k].type != BLK_SDP) continue;
            memset(seen, 0, sizeof(int) * A.N); int bc = 0;
            for (int i = off[k]; i < off[k + 1]; i++) if (!seen[col[i]]++) bc++;
            c0 += pow((double)P->blk[k].n, 3.0); c1 += pow((double)bc, 3.0);
        }
        if (!can && c1 > 0 && c0 / c1 >= symmin) can = 1;
        free(seen);
        if (!can) {
            if (verbose > 0) printf("   symmetry search: no automorphism group can give a gain of %g (%d constraint classes, index classes per block): no search\n", symmin, ccls);
            ncls = NI;   /* no search */
        }
    }
    int *cnt = calloc(m, sizeof(int)); for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) cnt[E[q].con]++;
    A.ufp = sx(sizeof(int) * NI); for (int u = 0; u < NI; u++) A.ufp[u] = u;
    A.gcap = 8; A.gens = sx(sizeof(int) * (size_t)A.gcap * NI); A.gcps = sx(sizeof(int) * (size_t)A.gcap * (m + 1));
    if (ncls < NI) { int *path = sx(sizeof(int) * (NI + 1)); as_dfs(&A, col, 0, P->b, E, ne, cnt, path); free(path); }
    int norb = 0; for (int u = 0; u < NI; u++) if (uf_find(A.ufp, u) == u) norb++;
    if (verbose > 0) printf("   symmetry search: %ld nodes, %d generator(s), %d index orbits, %.2fs\n", A.nodes, A.ngen, norb, wtime() - A.t0);
    *ngen_out = A.ngen; *gcps_out = A.gcps;
    int *gens = A.gens;
    free(A.ea); free(A.eb); free(A.ec); free(A.ev); free(A.ed); free(A.adjp); free(A.adj); free(A.hk); free(A.ncol); free(col); free(cnt); free(A.ufp); free(A.leaf_ref); free(A.inv_ref);
    return gens;
}


/* ---- would a reduction take the chordal decomposition away? ------------------------------
 * The moment form of the chordal conversion handles one large sparse block; a reduction that
 * leaves two or more blocks of at least chordal_minn indices, cut from sparse blocks (pattern
 * density at most chordal_density), is not applied in automatic mode (AC-OPF case300: the
 * sign split 739 -> 139 + 600 lost the conversion, 0.9 -> 28 s). */
static void sym_block_density(const Problem *P, double *dens) {
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; dens[k] = 1.0;
        if (B->type != BLK_SDP || B->n < 2) continue;
        size_t ne = B->C.nnz; for (int t = 0; t < B->ncon; t++) ne += B->A[t].nnz;
        uint64_t *key = sx(sizeof(uint64_t) * (ne + 1)); size_t q0 = 0;
        for (int t = -1; t < B->ncon; t++) { const SpSym *A = t < 0 ? &B->C : &B->A[t];
            for (int q = 0; q < A->nnz; q++) { int i = A->row[q], j = A->col[q]; if (i > j) { const int tt = i; i = j; j = tt; } key[q0++] = ((uint64_t)i << 32) | (uint64_t)j; } }
        qsort(key, q0, sizeof(uint64_t), cmp_u64);
        size_t nd = 0; for (size_t q = 0; q < q0; q++) if (q == 0 || key[q] != key[q - 1]) nd++;
        dens[k] = (double)nd / ((double)B->n * (B->n + 1) / 2);
        free(key);
    }
}
/* nred reduced SDP blocks of sizes nr[] cut from the original blocks parent[] */
static int sym_chordal_conflict(const Problem *P, int nred, const int *nr, const int *parent, int minn, double maxdens) {
    /* 4.34: the moment form takes any number of blocks (ISSUES 1 closed): no conflict unless
     * asked for (BRISK_SYMCONF, the 4.33 rule, kept for comparisons) */
    if (!getenv("BRISK_SYMCONF")) return 0;
    /* the moment form takes one block of at least minn indices (sparse) with every other SDP
     * block of at most 64 indices: a conflict is losing that shape */
    const int maxsmall = 64;
    double *dens = sx(sizeof(double) * (P->nblk + 1));
    sym_block_density(P, dens);
    int big0 = 0, other0 = 0;
    for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type != BLK_SDP) continue; if (P->blk[k].n >= minn && dens[k] <= maxdens) big0++; else if (P->blk[k].n > maxsmall) other0++; }
    const int orig_mom = big0 == 1 && other0 == 0;
    int big1 = 0, other1 = 0, any1 = 0;
    for (int r = 0; r < nred; r++) { if (nr[r] >= minn) any1 = 1; if (nr[r] >= minn && dens[parent[r]] <= maxdens) big1++; else if (nr[r] > maxsmall) other1++; }
    const int red_mom = big1 == 1 && other1 == 0;
    free(dens);
    return orig_mom && any1 && !red_mom;
}

/* ---- 4.33: symmetry-adapted basis (block diagonalisation) ------------------------------
 * An invariant X (P_g X P_g' = X for all g) lies in the commutant of the permutation
 * representation, which an orthogonal change of basis T brings to block-diagonal form:
 * T'XT = (+)_rho X_rho (x) I_{d_rho}, one block X_rho of size m_rho (the multiplicity of the
 * irreducible representation rho) repeated d_rho (its dimension) times. The basis is built
 * orbit by orbit, so it is sparse (a vector lives on one orbit) and the transformed
 * constraints stay sparse: on each orbit a random symmetric element M of the commutant
 * (constant on the orbitals, the orbits of G on the pairs) is diagonalised; its eigenspaces
 * are irreducible invariant subspaces (generically, checked: invariance under the generators
 * and a one-dimensional self-commutant, i.e. irreducible of real type). Copies of the same
 * irreducible type are aligned by the equivariant isometry from a reference copy (Schur's
 * lemma: unique up to scale), so that the d_rho rows of every copy transform by the same
 * matrices and the (i, i) diagonal blocks of T'AT coincide for invariant A while the (i, j)
 * blocks vanish. The reduced problem keeps one row: block (k, rho) of size m_{k,rho} with
 * data sum_i W_i' A W_i (the factor d_rho is inside the sum), same y, X_k = sum_rho sum_i
 * W_i X_rho W_i' on the way back. Exact up to the rounding of the basis (measured on the file
 * data by the postsolve). Not applied (the orbit aggregation alone remains) when an orbit
 * spans two blocks, an orbit is larger than SYM_BDMAXORB, an irreducible representation has
 * dimension above SYM_BDMAXD, or a representation is not of real type.                     */
#define SYM_BDMAXORB 4000
#define SYM_BDMAXD 24
#define SYM_BDMAXGEN 32    /* generators used for the basis (a subgroup is exact; the search can return hundreds) */
static uint64_t bd_rng = 0x9E3779B97F4A7C15ull;
static double bd_rand(void) { bd_rng ^= bd_rng << 13; bd_rng ^= bd_rng >> 7; bd_rng ^= bd_rng << 17; return (double)(bd_rng >> 11) * (2.0 / 9007199254740992.0) - 1.0; }

/* nullity of F (d x d) -> [R_s F - F Q_s]_s and, when it is one, the isometric kernel vector */
static int bd_intertwiner(int d, int ngen, double **R, double **Q, double *F) {
    const int dd = d * d, rows = ngen * dd;
    double *K = calloc((size_t)rows * dd, sizeof(double));
    for (int s = 0; s < ngen; s++) for (int j = 0; j < d; j++) for (int i = 0; i < d; i++) {
        const size_t row = (size_t)s * dd + i + (size_t)j * d;
        for (int l = 0; l < d; l++) K[row + (size_t)(l + j * d) * rows] += R[s][i + l * d];   /* (R F)_ij: F(l,j) */
        for (int l = 0; l < d; l++) K[row + (size_t)(i + l * d) * rows] -= Q[s][l + j * d];   /* (F Q)_ij: F(i,l) */
    }
    const int mn = dd;
    double *sv = sx(sizeof(double) * mn), *VT = sx(sizeof(double) * (size_t)mn * dd), *U = sx(sizeof(double) * (size_t)rows * mn);
    int lwork = -1, info = 0, *iwork = sx(sizeof(int) * 8 * mn); double wq;
    BL(dgesdd_)("S", &rows, &dd, K, &rows, sv, U, &rows, VT, &mn, &wq, &lwork, iwork, &info);
    lwork = (int)wq + 1; double *work = sx(sizeof(double) * lwork);
    BL(dgesdd_)("S", &rows, &dd, K, &rows, sv, U, &rows, VT, &mn, work, &lwork, iwork, &info);
    int nul = 0;
    if (info == 0) { const double tol = 1e-8 * (sv[0] > 1 ? sv[0] : 1); for (int i = 0; i < mn; i++) if (sv[i] <= tol) nul++; }
    else nul = -1;
    if (nul == 1 && F) {
        double nrm = 0;
        for (int j = 0; j < dd; j++) { F[j] = VT[(mn - 1) + (size_t)j * mn]; nrm += F[j] * F[j]; }
        nrm = sqrt(nrm / d);
        for (int j = 0; j < dd; j++) F[j] /= nrm;          /* F'F = c I -> isometry */
    }
    free(K); free(sv); free(VT); free(U); free(iwork); free(work);
    return nul;
}

typedef struct { int k, t, d, s, r, p; int *idx; double *U; } BDCopy;   /* copy of an irreducible of dimension d (type t of block k) on an orbit of s local indices; U s x d */

/* the irreducible decomposition of one block under the generators hg (local permutations of
 * its n indices): copies appended to cp, types (reference matrices) local to the block.
 * Returns NULL or the reason it is not possible. */
static const char *bd_block(int k, int n, int nh, int **hg, signed char **hs, BDCopy **cpp, int *ncp, int *cpcap, int *ntype_out, int **dt_out, int verbose) {
    const char *why = NULL;
    int *par = sx(sizeof(int) * (n + 1)); for (int i = 0; i < n; i++) par[i] = i;
    for (int s = 0; s < nh; s++) for (int i = 0; i < n; i++) { const int a = uf_find(par, i), b2 = uf_find(par, hg[s][i]); if (a != b2) par[a] = b2; }
    int *ord = sx(sizeof(int) * (n + 1)), *root = sx(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) { root[i] = uf_find(par, i); ord[i] = i; }
    { int *cnt = calloc(n + 1, sizeof(int)); for (int i = 0; i < n; i++) cnt[root[i] + 1]++;
      for (int i = 0; i < n; i++) cnt[i + 1] += cnt[i];
      for (int i = 0; i < n; i++) ord[cnt[root[i]]++] = i;
      free(cnt); }
    int norb = 0, maxs = 0;
    int *ostart = sx(sizeof(int) * (n + 2));
    for (int q = 0; q < n; q++) if (q == 0 || root[ord[q]] != root[ord[q - 1]]) ostart[norb++] = q;
    ostart[norb] = n;
    for (int o = 0; o < norb; o++) { const int s = ostart[o + 1] - ostart[o]; if (s > maxs) maxs = s; }
    if (maxs > SYM_BDMAXORB) why = "an orbit is too large";
    int ntype = 0, tcap = 16; int *dt = sx(sizeof(int) * tcap); double ***Rref = sx(sizeof(double **) * tcap);
    int *pos = sx(sizeof(int) * (n + 1));
    double *M = sx(sizeof(double) * ((size_t)maxs * maxs + 1)), *lam = sx(sizeof(double) * (maxs + 1));
    double *PU = sx(sizeof(double) * ((size_t)maxs * SYM_BDMAXD + 1)), *F = sx(sizeof(double) * SYM_BDMAXD * SYM_BDMAXD);
    double **R = sx(sizeof(double *) * (nh + 1)); for (int s = 0; s < nh; s++) R[s] = sx(sizeof(double) * SYM_BDMAXD * SYM_BDMAXD);
    int *par2 = sx(sizeof(int) * ((size_t)maxs * maxs + 1)); double *rr = sx(sizeof(double) * ((size_t)maxs * maxs + 1));
    int *pp2 = calloc((size_t)maxs * maxs + 1, sizeof(int)); char *z2 = calloc((size_t)maxs * maxs + 1, 1);   /* 4.37: parity of the pair orbits, sign-inconsistent orbitals */
    int lwork = -1; double *work = NULL;
    { double wq; int info, nn = maxs > 1 ? maxs : 1; BL(dsyev_)("V", "U", &nn, M, &nn, lam, &wq, &lwork, &info); lwork = (int)wq + 1; work = sx(sizeof(double) * lwork); }
    for (int o = 0; o < norb && !why; o++) {
        const int s = ostart[o + 1] - ostart[o];
        const int *idx = ord + ostart[o];
        for (int q = 0; q < s; q++) pos[idx[q]] = q;
        const size_t ss = (size_t)s * s;
        for (size_t p = 0; p < ss; p++) { par2[p] = (int)p; pp2[p] = 0; z2[p] = 0; }
        /* the orbitals (orbits on the pairs), with parity for signed generators (4.37): M in the
         * commutant has M(g a, g b) = s_a s_b M(a, b); an orbital that meets itself with the sign
         * -1 carries 0 */
        {   PUF U2; U2.p = par2; U2.par = pp2;
            for (int g = 0; g < nh; g++) for (int a = 0; a < s; a++) { const int ga = pos[hg[g][idx[a]]]; const int sa = hs && hs[g] && hs[g][idx[a]] < 0; for (int b2 = 0; b2 < s; b2++) {
                const int gb = pos[hg[g][idx[b2]]]; const int sb = hs && hs[g] && hs[g][idx[b2]] < 0;
                if (!puf_union(&U2, a + b2 * s, ga + gb * s, sa ^ sb)) z2[a + b2 * s] = 1; } }
            for (size_t p = 0; p < ss; p++) if (z2[p]) { int pr; z2[puf_find(&U2, (int)p, &pr)] = 2; }
        }
        int attempt, ok = 0;
        const int ncp0 = *ncp, ntype0 = ntype;
        for (attempt = 0; attempt < 4 && !ok; attempt++) {
            *ncp = ncp0; ntype = ntype0;
            for (size_t p = 0; p < ss; p++) rr[p] = bd_rand();
            {   PUF U2; U2.p = par2; U2.par = pp2;
                for (int a = 0; a < s; a++) for (int b2 = 0; b2 < s; b2++) {
                    int p1, p2; const int r1 = puf_find(&U2, a + b2 * s, &p1), r2 = puf_find(&U2, b2 + a * s, &p2);
                    M[a + (size_t)b2 * s] = (z2[r1] == 2 ? 0.0 : (p1 ? -rr[r1] : rr[r1])) + (z2[r2] == 2 ? 0.0 : (p2 ? -rr[r2] : rr[r2])); }
            }
            int info, nn = s; BL(dsyev_)("V", "U", &nn, M, &nn, lam, work, &lwork, &info);
            if (info) break;
            double lmax = 0; for (int a = 0; a < s; a++) if (fabs(lam[a]) > lmax) lmax = fabs(lam[a]);
            const double gap = 1e-7 * (1 + lmax);
            ok = 1;
            for (int j0 = 0; j0 < s && ok; ) {
                int j1 = j0 + 1; while (j1 < s && lam[j1] - lam[j1 - 1] <= gap) j1++;
                const int d = j1 - j0;
                if (d > SYM_BDMAXD) { why = "an irreducible representation is too large"; ok = 0; break; }
                const double *U = M + (size_t)j0 * s;
                for (int g = 0; g < nh && ok; g++) {
                    for (int a = 0; a < s; a++) { const int ga = pos[hg[g][idx[a]]]; const double sa = hs && hs[g] ? (double)hs[g][idx[a]] : 1.0; for (int i = 0; i < d; i++) PU[ga + (size_t)i * s] = sa * U[a + (size_t)i * s]; }
                    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) { double t = 0; for (int a = 0; a < s; a++) t += U[a + (size_t)i * s] * PU[a + (size_t)j * s]; R[g][i + j * d] = t; }
                    double res = 0;
                    for (int j = 0; j < d; j++) for (int a = 0; a < s; a++) { double t = PU[a + (size_t)j * s]; for (int i = 0; i < d; i++) t -= U[a + (size_t)i * s] * R[g][i + j * d]; res += t * t; }
                    if (res > 1e-14 * d) ok = 0;
                }
                if (!ok) break;
                if (nh > 0 && bd_intertwiner(d, nh, R, R, NULL) != 1) { ok = 0; break; }
                int t = -1;
                for (int u = 0; u < ntype && t < 0; u++) {
                    if (dt[u] != d) continue;
                    /* equivalent representations have equal characters: a cheap filter on the generators */
                    int same = 1;
                    for (int g = 0; g < nh && same; g++) { double tr = 0, tr2 = 0; for (int i = 0; i < d; i++) { tr += R[g][i + i * d]; tr2 += Rref[u][g][i + i * d]; } if (fabs(tr - tr2) > 1e-6) same = 0; }
                    if (!same) continue;
                    const int nul = nh > 0 ? bd_intertwiner(d, nh, R, Rref[u], F) : 1;
                    if (nh == 0) for (int i = 0; i < d * d; i++) F[i] = i % (d + 1) == 0;
                    if (nul == 1) t = u; else if (nul != 0) { ok = 0; break; }
                }
                if (!ok) break;
                if (*ncp == *cpcap) { *cpcap *= 2; *cpp = realloc(*cpp, sizeof(BDCopy) * *cpcap); }
                BDCopy *c = &(*cpp)[(*ncp)++];
                c->k = k; c->s = s; c->d = d; c->idx = sx(sizeof(int) * s); memcpy(c->idx, idx, sizeof(int) * s); c->U = sx(sizeof(double) * (size_t)s * d);
                if (t < 0) {
                    if (ntype == tcap) { tcap *= 2; dt = realloc(dt, sizeof(int) * tcap); Rref = realloc(Rref, sizeof(double **) * tcap); }
                    t = ntype++; dt[t] = d; Rref[t] = sx(sizeof(double *) * (nh + 1));
                    for (int g = 0; g < nh; g++) { Rref[t][g] = sx(sizeof(double) * d * d); memcpy(Rref[t][g], R[g], sizeof(double) * d * d); }
                    memcpy(c->U, U, sizeof(double) * (size_t)s * d);
                } else {
                    for (int j = 0; j < d; j++) for (int a = 0; a < s; a++) { double w = 0; for (int i = 0; i < d; i++) w += U[a + (size_t)i * s] * F[i + j * d]; c->U[a + (size_t)j * s] = w; }
                }
                c->t = t;
                j0 = j1;
            }
            /* 5.8 (a user's fix): the counters go back too, or the caller frees cp[ncp0 .. ncp) a
             * second time (a double free on three POEMA relaxations) */
            if (!ok) { for (int q = ncp0; q < *ncp; q++) { free((*cpp)[q].U); free((*cpp)[q].idx); } *ncp = ncp0; }
            if (!ok) { for (int u = ntype0; u < ntype; u++) { for (int g = 0; g < nh; g++) free(Rref[u][g]); free(Rref[u]); } ntype = ntype0; }
        }
        if (!ok && !why) why = "no generic decomposition of an orbit was found";
    }
    free(M); free(lam); free(PU); free(F); free(work); free(par2); free(rr); free(pp2); free(z2);
    for (int s = 0; s < nh; s++) free(R[s]);
    free(R);
    for (int u = 0; u < ntype; u++) { for (int g = 0; g < nh; g++) free(Rref[u][g]); free(Rref[u]); }
    free(Rref); free(pos); free(par); free(ord); free(root); free(ostart);
    if (why) { free(dt); *ntype_out = 0; *dt_out = NULL; }
    else { *ntype_out = ntype; *dt_out = dt; }
    (void)verbose;
    return why;
}

/* triplets sorted, duplicates summed, entries below 1e-13 of the largest of their constraint dropped */
typedef struct { int c, b, i, j; double v; } STrip;
static int cmp_strip(const void *x, const void *y) { const STrip *a = x, *b = y;
    if (a->c != b->c) return a->c < b->c ? -1 : 1;
    if (a->b != b->b) return a->b < b->b ? -1 : 1;
    if (a->j != b->j) return a->j < b->j ? -1 : 1;
    if (a->i != b->i) return a->i < b->i ? -1 : 1;
    return 0; }
/* the reduced blocks and the transformed triplets; 0 when not applicable (nothing allocated).
 * Blocks exchanged by the group: one representative per block orbit is kept (its data times
 * the orbit size: <A, X> = sum_b <A_b, X_b> = |orbit| <A_b1, X_b1> for invariant A and X),
 * decomposed under the stabilizer of the block (Schreier generators from a transversal);
 * the other blocks are recovered as X_b = P_u X_b1 P_u' on the way back. */
static int sym_blockdiag(const Problem *P, PSSym *S, const int *cls, const int *off, int NI, int nsdp, int ngen, int **gen, const int *gblk, int verbose,
                         size_t *nt_out, int **tc_out, int **tb_out, int **ti_out, int **tj_out, double **tv_out, int *nbr_out, int **bsr_out) {
    const int nb = P->nblk;
    const double t0 = wtime();
    /* block orbits (an index permutation maps whole blocks onto blocks of equal size) */
    int *bpar = sx(sizeof(int) * (nb + 1)); for (int k = 0; k < nb; k++) bpar[k] = k;
    for (int s = 0; s < ngen; s++) for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) { const int a = uf_find(bpar, k), b2 = uf_find(bpar, gblk[gen[s][off[k]]]); if (a != b2) bpar[a] = b2; }
    int *brep = sx(sizeof(int) * (nb + 1)), *bscale = calloc(nb + 1, sizeof(int)), **bmap = calloc(nb + 1, sizeof(int *));
    signed char **bsgn = NULL;       /* 4.37: signs of the exchanged blocks' maps */
    for (int k = 0; k < nb; k++) brep[k] = -1;
    for (int k = 0; k < nb; k++) { const int r = uf_find(bpar, k); if (brep[r] < 0) brep[r] = k; brep[k] = brep[r]; }   /* the smallest block of the orbit */
    for (int k = 0; k < nb; k++) bscale[brep[k]]++;
    const char *why = NULL;
    int ncp = 0, cpcap = 64; BDCopy *cp = sx(sizeof(BDCopy) * cpcap);
    int *nbt = calloc(nb + 1, sizeof(int)), **bdt = calloc(nb + 1, sizeof(int *));
    int *tmp = sx(sizeof(int) * (NI + 1)), *inv = sx(sizeof(int) * (NI + 1));
    for (int k = 0; k < nb && !why; k++) {
        if (brisk_stop_flag) { why = "interrupted"; break; }   /* 4.42: Ctrl-C / brisk_interrupt(): the basis is abandoned (the orbit aggregation stays) */
        if (P->blk[k].type != BLK_SDP || brep[k] != k) continue;
        const int n = P->blk[k].n;
        /* transversal u_b (b1 -> b) over the block orbit, then the Schreier generators
         * u_{s b}^-1 s u_b restricted to the block */
        /* 4.37: signed generators: the transversal and the Schreier generators carry signs
         * (composition i -> g2(g1 i) with the sign s1(i) s2(g1 i); the inverse of (u, su) maps
         * u(j) -> j with the sign su(j)) */
        const signed char *const *gsg = (const signed char *const *)S->gsg;
        int *orb = sx(sizeof(int) * (nb + 1)), no = 0; int **u = calloc(nb + 1, sizeof(int *)); signed char **su = calloc(nb + 1, sizeof(signed char *));
        u[k] = sx(sizeof(int) * NI); su[k] = sx(NI + 1); for (int i = 0; i < NI; i++) { u[k][i] = i; su[k][i] = 1; } orb[no++] = k;
        for (int q = 0; q < no; q++) for (int s = 0; s < ngen; s++) {
            const int b = orb[q], b2 = gblk[gen[s][off[b]]];
            if (u[b2]) continue;
            u[b2] = sx(sizeof(int) * NI); su[b2] = sx(NI + 1);
            for (int i = 0; i < NI; i++) { u[b2][i] = gen[s][u[b][i]]; su[b2][i] = (signed char)(su[b][i] * (gsg && gsg[s] ? gsg[s][u[b][i]] : 1)); }
            orb[no++] = b2;
        }
        int nh = 0, hcap = no * ngen + 1; int **hg = sx(sizeof(int *) * hcap); signed char **hs = calloc(hcap, sizeof(signed char *)); uint64_t *hh = sx(sizeof(uint64_t) * hcap);
        signed char *tsg = sx(NI + 1);
        for (int q = 0; q < no; q++) for (int s = 0; s < ngen; s++) {
            const int b = orb[q], b2 = gblk[gen[s][off[b]]];
            for (int i = 0; i < NI; i++) inv[u[b2][i]] = i;
            int anyneg = 0;
            for (int i = 0; i < NI; i++) { const int ui = u[b][i], gi = gen[s][ui], j = inv[gi]; tmp[i] = j; tsg[i] = (signed char)(su[b][i] * (gsg && gsg[s] ? gsg[s][ui] : 1) * su[b2][j]); }
            for (int i = 0; i < n; i++) if (tsg[off[k] + i] < 0) anyneg = 1;
            int ident = !anyneg; for (int i = 0; i < n && ident; i++) if (tmp[off[k] + i] != off[k] + i) ident = 0;
            if (ident) continue;
            uint64_t h = 1469598103934665603ull; for (int i = 0; i < n; i++) { h ^= (uint64_t)tmp[off[k] + i] + (tsg[off[k] + i] < 0 ? 0x5555ull : 0); h *= 1099511628211ull; }
            int dup = 0; for (int t = 0; t < nh && !dup; t++) if (hh[t] == h) { dup = 1; for (int i = 0; i < n; i++) if (hg[t][i] != tmp[off[k] + i] - off[k] || (hs[t] ? hs[t][i] : 1) != tsg[off[k] + i]) { dup = 0; break; } }
            if (dup) continue;
            hg[nh] = sx(sizeof(int) * n); for (int i = 0; i < n; i++) hg[nh][i] = tmp[off[k] + i] - off[k];
            if (anyneg) { hs[nh] = sx(n + 1); for (int i = 0; i < n; i++) hs[nh][i] = tsg[off[k] + i]; }
            hh[nh] = h; nh++;
        }
        free(tsg);
        for (int q = 1; q < no; q++) { const int b = orb[q]; bmap[b] = sx(sizeof(int) * n); for (int i = 0; i < n; i++) bmap[b][i] = u[b][off[k] + i] - off[b];
            int anyneg = 0; for (int i = 0; i < n; i++) if (su[b][off[k] + i] < 0) anyneg = 1;
            if (anyneg) { if (!bsgn) bsgn = calloc(nb + 1, sizeof(signed char *)); bsgn[b] = sx(n + 1); for (int i = 0; i < n; i++) bsgn[b][i] = su[b][off[k] + i]; } }
        for (int q = 0; q < no; q++) { free(u[orb[q]]); free(su[orb[q]]); }
        free(u); free(su); free(orb);
        why = bd_block(k, n, nh, hg, hs, &cp, &ncp, &cpcap, &nbt[k], &bdt[k], verbose);
        for (int t = 0; t < nh; t++) { free(hg[t]); free(hs[t]); }
        free(hg); free(hs); free(hh);
    }
    free(tmp); free(inv);
    int ret = 0;
    if (why) {
        if (verbose >= 0) printf("presolve: symmetry-adapted basis not applied (%s)\n", why);
        for (int k = 0; k < nb; k++) free(bmap[k]);
        free(bmap);
        if (bsgn) { for (int k = 0; k < nb; k++) free(bsgn[k]); free(bsgn); bsgn = NULL; }
    } else {
        /* reduced blocks: (k, t) for the representative SDP blocks in that order, LP blocks in place */
        int nbr = 0; for (int k = 0; k < nb; k++) { if (P->blk[k].type == BLK_LP) nbr++; else if (brep[k] == k) nbr += nbt[k]; }
        int *bsr = sx(sizeof(int) * (nbr + 1)), *lpr = sx(sizeof(int) * (nb + 1)), **rkt = calloc(nb + 1, sizeof(int *));
        S->nsb = nbr; S->sb = calloc(nbr + 1, sizeof(SymBlk));
        int r = 0;
        for (int k = 0; k < nb; k++) {
            if (P->blk[k].type == BLK_LP) { lpr[k] = r; bsr[r] = -P->blk[k].n; S->sb[r].k = k; S->sb[r].t = -1; S->sb[r].n = P->blk[k].n; r++; continue; }
            if (brep[k] != k) continue;
            rkt[k] = sx(sizeof(int) * (nbt[k] + 1));
            for (int t = 0; t < nbt[k]; t++) { rkt[k][t] = r; S->sb[r].k = k; S->sb[r].t = t; S->sb[r].d = bdt[k][t]; S->sb[r].n = 0; r++; }
        }
        for (int q = 0; q < ncp; q++) { const int rr = rkt[cp[q].k][cp[q].t]; cp[q].r = rr; cp[q].p = S->sb[rr].n++; }
        /* the reduced problem's blocks: the SDP ones of size >= 2, the LP blocks, then one LP
         * block holding every 1 x 1 piece (a 1 x 1 SDP block is an LP variable) */
        int rb = 0, nlp1 = 0;
        for (r = 0; r < nbr; r++) { SymBlk *B = &S->sb[r]; B->lp = B->t >= 0 && B->n == 1; if (B->lp) B->p = nlp1++; }
        for (r = 0; r < nbr; r++) if (S->sb[r].t >= 0 && !S->sb[r].lp) S->sb[r].rblk = rb++;
        for (r = 0; r < nbr; r++) if (S->sb[r].t < 0) S->sb[r].rblk = rb++;
        const int lpblk = nlp1 ? rb++ : -1;
        for (r = 0; r < nbr; r++) if (S->sb[r].lp) S->sb[r].rblk = lpblk;
        free(bsr); bsr = sx(sizeof(int) * (rb + 1));
        for (r = 0; r < nbr; r++) { SymBlk *B = &S->sb[r]; if (B->lp) continue; bsr[B->rblk] = B->t < 0 ? -B->n : B->n; }
        if (lpblk >= 0) bsr[lpblk] = -nlp1;
        S->lpblk = lpblk;
        for (r = 0; r < nbr; r++) { SymBlk *B = &S->sb[r]; if (B->t >= 0) B->W = calloc((size_t)P->blk[B->k].n * B->n * B->d + 1, sizeof(double)); }
        for (int q = 0; q < ncp; q++) {
            const BDCopy *c = &cp[q]; const int nk = P->blk[c->k].n, nr = S->sb[c->r].n, d = c->d;
            double *W = S->sb[c->r].W;
            for (int i = 0; i < d; i++) for (int a = 0; a < c->s; a++) W[(size_t)i * nk * nr + c->idx[a] + (size_t)c->p * nk] = c->U[a + (size_t)i * c->s];
        }
        /* per global index: the (r, p, w[0..d)) it touches (representative blocks only) */
        int *lp = calloc(NI + 2, sizeof(int));
        for (int q = 0; q < ncp; q++) for (int a = 0; a < cp[q].s; a++) lp[off[cp[q].k] + cp[q].idx[a] + 1]++;
        for (int a = 0; a < NI; a++) lp[a + 1] += lp[a];
        const int nl = lp[NI];
        int *lr = sx(sizeof(int) * (nl + 1)), *lpp = sx(sizeof(int) * (nl + 1)), *ld = sx(sizeof(int) * (nl + 1)); double *lw = sx(sizeof(double) * ((size_t)nl * SYM_BDMAXD + 1));
        { int *w = sx(sizeof(int) * (NI + 1)); memcpy(w, lp, sizeof(int) * NI);
          for (int q = 0; q < ncp; q++) { const BDCopy *c = &cp[q]; const int d = c->d; for (int a = 0; a < c->s; a++) { const int e = w[off[c->k] + c->idx[a]]++; lr[e] = c->r; lpp[e] = c->p; ld[e] = d; for (int i = 0; i < d; i++) lw[(size_t)e * SYM_BDMAXD + i] = c->U[a + (size_t)i * c->s]; } }
          free(w); }
        /* 4.37: the transformed triplets class by class (the aggregated constraint of an orbit),
         * each class merged (duplicates summed, entries below 1e-13 of its largest dropped) before
         * it is appended: the unmerged list of the signed d = 5 reduction does not fit in memory */
        size_t nt = 0, ct = 1024;
        int *tc = sx(sizeof(int) * ct), *tb = sx(sizeof(int) * ct), *ti = sx(sizeof(int) * ct), *tj = sx(sizeof(int) * ct); double *tv = sx(sizeof(double) * ct);
        const double droptol = getenv("BRISK_SYMDROP") ? atof(getenv("BRISK_SYMDROP")) : 1e-13;
        size_t nl_ = 0, cl_ = 1024; STrip *L_ = sx(sizeof(STrip) * cl_);
        #define EMIT(c_, b_, i_, j_, v_) do { if (nl_ == cl_) { cl_ *= 2; L_ = realloc(L_, sizeof(STrip) * cl_); if (!L_) { fprintf(stderr, "brisk: out of memory (symmetry reduction)\n"); exit(1); } } \
            { int i2_ = (i_), j2_ = (j_); if (i2_ > j2_) { const int t_ = i2_; i2_ = j2_; j2_ = t_; } L_[nl_].c = (c_); L_[nl_].b = (b_); L_[nl_].i = i2_; L_[nl_].j = j2_; L_[nl_].v = (v_); nl_++; } } while (0)
        #define FLUSH() do { qsort(L_, nl_, sizeof(STrip), cmp_strip); size_t w_ = 0; \
            for (size_t q_ = 0; q_ < nl_; q_++) { if (w_ && L_[w_ - 1].b == L_[q_].b && L_[w_ - 1].i == L_[q_].i && L_[w_ - 1].j == L_[q_].j) L_[w_ - 1].v += L_[q_].v; else L_[w_++] = L_[q_]; } \
            double mx_ = 0; for (size_t q_ = 0; q_ < w_; q_++) if (fabs(L_[q_].v) > mx_) mx_ = fabs(L_[q_].v); \
            for (size_t q_ = 0; q_ < w_; q_++) { if (!(fabs(L_[q_].v) > droptol * mx_)) continue; \
                if (nt == ct) { ct *= 2; tc = realloc(tc, sizeof(int) * ct); tb = realloc(tb, sizeof(int) * ct); ti = realloc(ti, sizeof(int) * ct); tj = realloc(tj, sizeof(int) * ct); tv = realloc(tv, sizeof(double) * ct); \
                                 if (!tc || !tb || !ti || !tj || !tv) { fprintf(stderr, "brisk: out of memory (symmetry reduction)\n"); exit(1); } } \
                tc[nt] = L_[q_].c; tb[nt] = L_[q_].b; ti[nt] = L_[q_].i; tj[nt] = L_[q_].j; tv[nt] = L_[q_].v; nt++; } \
            nl_ = 0; } while (0)
        /* members of each class: (block, constraint slot) of the representative and LP blocks */
        int *mst = calloc((size_t)S->mr + 2, sizeof(int));
        for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; if (B->type != BLK_LP && brep[k] != k) continue; for (int t = 0; t < B->ncon; t++) { const int c = cls[B->con[t]]; if (c >= 0) mst[c + 1]++; } }
        for (int c = 0; c < S->mr; c++) mst[c + 1] += mst[c];
        int *mk_ = sx(sizeof(int) * ((size_t)mst[S->mr] + 1)), *mt_ = sx(sizeof(int) * ((size_t)mst[S->mr] + 1));
        { int *fill = sx(sizeof(int) * ((size_t)S->mr + 1)); memcpy(fill, mst, sizeof(int) * S->mr);
          for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; if (B->type != BLK_LP && brep[k] != k) continue; for (int t = 0; t < B->ncon; t++) { const int c = cls[B->con[t]]; if (c >= 0) { mk_[fill[c]] = k; mt_[fill[c]] = t; fill[c]++; } } }
          free(fill); }
        for (int c = -1; c < S->mr; c++) {
            const int w0 = c < 0 ? 0 : mst[c], w1 = c < 0 ? nb : mst[c + 1];
            for (int w = w0; w < w1; w++) {
                const int k = c < 0 ? w : mk_[w], t = c < 0 ? -1 : mt_[w];
                const Block *B = &P->blk[k]; const int islp = B->type == BLK_LP;
                if (!islp && brep[k] != k) continue;
                const double scale = islp ? 1.0 : (double)bscale[k];
                const SpSym *A = t < 0 ? &B->C : &B->A[t];
                const double sgc = t < 0 ? 1.0 : (double)S->csg[B->con[t]];
                for (int q = 0; q < A->nnz; q++) {
                    if (islp) { EMIT(c, S->sb[lpr[k]].rblk, A->row[q], A->row[q], sgc * A->val[q]); continue; }
                    const int a = off[k] + A->row[q], b2 = off[k] + A->col[q]; const double v = sgc * A->val[q] * scale;
                    for (int ea = lp[a]; ea < lp[a + 1]; ea++) for (int eb = lp[b2]; eb < lp[b2 + 1]; eb++) {
                        if (lr[ea] != lr[eb]) continue;
                        const int d = ld[ea]; double wv = 0;
                        for (int i = 0; i < d; i++) wv += lw[(size_t)ea * SYM_BDMAXD + i] * lw[(size_t)eb * SYM_BDMAXD + i];
                        wv *= v;
                        const SymBlk *SB = &S->sb[lr[ea]];
                        if (SB->lp) { EMIT(c, lpblk, SB->p, SB->p, a != b2 ? 2 * wv : wv); continue; }
                        const int p = lpp[ea], pq = lpp[eb];
                        if (p <= pq) EMIT(c, SB->rblk, p, pq, wv);
                        if (a != b2 && pq <= p) EMIT(c, SB->rblk, pq, p, wv);
                    }
                }
            }
            FLUSH();
        }
        free(L_); free(mst); free(mk_); free(mt_);
        #undef FLUSH
        #undef EMIT
        free(lp); free(lr); free(lpp); free(ld); free(lw); free(lpr);
        for (int k = 0; k < nb; k++) free(rkt[k]);
        free(rkt);
        if (verbose >= 0) {
            printf("presolve: symmetry-adapted basis: blocks");
            for (int k = 0; k < nb; k++) { if (P->blk[k].type == BLK_LP || brep[k] != k) continue; printf(" %d%s ->", P->blk[k].n, bscale[k] > 1 ? "(x)" : ""); int n1 = 0; for (r = 0; r < nbr; r++) if (S->sb[r].k == k) { if (S->sb[r].n == 1) n1++; else printf(" %d%s", S->sb[r].n, S->sb[r].d > 1 ? "*" : ""); } if (n1) printf(" %dx1", n1); printf(";"); }
            printf(" (x): one of %s exchanged blocks; *: representation of dimension > 1; 1 x 1 pieces are LP variables (%.2f s)\n", "several", wtime() - t0);
        }
        S->bd = 1; S->nbr = rb; S->bsr = bsr; S->brep = brep; S->bmap = bmap; S->bsgn = bsgn; brep = NULL; bmap = NULL; bsgn = NULL;
        *nt_out = nt; *tc_out = tc; *tb_out = tb; *ti_out = ti; *tj_out = tj; *tv_out = tv; *nbr_out = rb; *bsr_out = bsr;
        ret = 1;
    }
    for (int q = 0; q < ncp; q++) { free(cp[q].U); free(cp[q].idx); }
    for (int k = 0; k < nb; k++) free(bdt[k]);
    free(cp); free(bpar); free(brep); free(bscale); free(nbt); free(bdt);
    return ret;
}

static size_t sym_merge_trips(size_t nt, int *tc, int *tb, int *ti, int *tj, double *tv, double droptol) {
    STrip *T = sx(sizeof(STrip) * (nt + 1));
    for (size_t q = 0; q < nt; q++) { int a = ti[q], b2 = tj[q]; if (a > b2) { const int t = a; a = b2; b2 = t; } T[q].c = tc[q]; T[q].b = tb[q]; T[q].i = a; T[q].j = b2; T[q].v = tv[q]; }
    qsort(T, nt, sizeof(STrip), cmp_strip);
    size_t w = 0;
    for (size_t q = 0; q < nt; q++) {
        if (w && T[w - 1].c == T[q].c && T[w - 1].b == T[q].b && T[w - 1].i == T[q].i && T[w - 1].j == T[q].j) T[w - 1].v += T[q].v;
        else T[w++] = T[q];
    }
    nt = w; w = 0;
    for (size_t q0 = 0; q0 < nt; ) {
        size_t q1 = q0; double mx = 0;
        while (q1 < nt && T[q1].c == T[q0].c) { if (fabs(T[q1].v) > mx) mx = fabs(T[q1].v); q1++; }
        for (size_t q = q0; q < q1; q++) if (fabs(T[q].v) > droptol * mx) T[w++] = T[q];
        q0 = q1;
    }
    for (size_t q = 0; q < w; q++) { tc[q] = T[q].c; tb[q] = T[q].b; ti[q] = T[q].i; tj[q] = T[q].j; tv[q] = T[q].v; }
    free(T);
    return w;
}

double g_sym_est_cap = 0;
int g_sym_cap_bound = 0;      /* the last search stopped on the capped budget without a generator */
PSSym *sym_reduce(Problem *P, const char *fname, int verbose, double symtime, long symnodes, int symbd, double symmin, int chordal_minn, double chordal_density, int symsigned) {
    const int m = P->m, nb = P->nblk;
    const int auto_mode = !strcmp(fname, "auto");
    /* global SDP indices */
    int *off = sx(sizeof(int) * (nb + 1)); off[0] = 0;
    int nsdp = 0;
    for (int k = 0; k < nb; k++) { off[k + 1] = off[k] + P->blk[k].n; if (P->blk[k].type == BLK_SDP) nsdp += P->blk[k].n; }
    const int NI = off[nb];
    /* a generator file numbers the SDP indices in block order: it needs the SDP blocks first */
    if (!auto_mode) { int seen_lp = 0; for (int k = 0; k < nb; k++) { if (P->blk[k].type == BLK_LP) seen_lp = 1; else if (seen_lp) { fprintf(stderr, "brisk: -sym <file> needs the SDP blocks before the LP blocks in the file\n"); exit(2); } } }
    /* entries: key = (min, max) global; value; constraint (-1 for C) - built first, the automatic search needs them */
    size_t ne = 0;
    for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; ne += B->C.nnz; for (int t = 0; t < B->ncon; t++) ne += B->A[t].nnz; }
    SEnt *E = sx(sizeof(SEnt) * (ne ? ne : 1));
    {   size_t q0 = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k]; const int lp = B->type == BLK_LP;
            for (int t = -1; t < B->ncon; t++) {
                const SpSym *S = t < 0 ? &B->C : &B->A[t];
                for (int q = 0; q < S->nnz; q++) {
                    int i = off[k] + S->row[q], j = off[k] + (lp ? S->row[q] : S->col[q]);
                    if (i > j) { const int tt = i; i = j; j = tt; }
                    E[q0].key = ((uint64_t)i << 32) | (uint64_t)j; E[q0].con = t < 0 ? -1 : B->con[t]; E[q0].v = S->val[q]; q0++;
                }
            }
        }
        qsort(E, ne, sizeof(SEnt), cmp_sent);
        for (size_t q = 1; q < ne; q++) if (E[q].key == E[q - 1].key && E[q].con == E[q - 1].con) { fprintf(stderr, "brisk: symmetry reduction: duplicate entries in the data\n"); exit(2); }
    }
    int ngen = 0, cap = 4, **autocp = NULL;
    int **gen = sx(sizeof(int *) * cap);
    signed char **gsg = calloc(cap, sizeof(signed char *)), **gce = calloc(cap, sizeof(signed char *));   /* 4.37: signs (NULL: +1) */
    #define GEN_GROW() do { if (ngen == cap) { cap *= 2; gen = realloc(gen, sizeof(int *) * cap); gsg = realloc(gsg, sizeof(signed char *) * cap); gce = realloc(gce, sizeof(signed char *) * cap); \
                             if (autocp) autocp = realloc(autocp, sizeof(int *) * cap); } gsg[ngen] = NULL; gce[ngen] = NULL; } while (0)
    if (!strcmp(fname, "auto")) {
        int na = 0, *gc = NULL;
        /* the search's time budget: a share of the expected solve (dense Schur factor and
         * block algebra over ~20 iterations at ~1 GFlop/s), at least 1 s, at most symtime */
        double est = (double)m * m * m / 3.0, sp = m;
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) sp += (double)P->blk[k].ncon * P->blk[k].ncon;
        if (pow(sp, 1.5) < est) est = pow(sp, 1.5);     /* a sparse Schur complement (many small blocks) factors for about nnz^1.5 */
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) est += 20.0 * (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        est /= 1e9;
        const double est_full = est;
        if (g_sym_est_cap > 0 && est > g_sym_est_cap) est = g_sym_est_cap;   /* 5.7: the matrix-free method was asked for (main.c) */
        /* 4.34: at least symtime, more on problems whose solve is long (Example 8.1.3 d = 5:
         * 7 search nodes of 12 s each on 15 255 indices and 1.35 M rows) */
        const double tb = fmin(fmax(symtime, fmin(900.0, 0.01 * est)), fmax(0.05, 0.15 * est));
        if (est < 0.2 && symmin > 1) {   /* a solve of a fraction of a second: not worth a search (-symmin 1: always) */
            if (verbose > 0) printf("   symmetry search skipped (expected solve %.2f s)\n", est);
            free(gen); free(gsg); free(gce); free(off); free(E); return NULL;
        }
        if (verbose > 0) printf("   symmetry search: time budget %.2f s (expected solve ~%.1f s)\n", tb, est);
        const double tsearch0 = wtime();
        int *ga = sym_auto(P, off, NI, nsdp, E, ne, &na, &gc, verbose, tb, symnodes, auto_mode ? symmin : 0.0, 0);
        /* a search that ran into the budget of the matrix-free estimate and found nothing: with the
         * standard method's budget it might have (main.c runs the pipeline again if the matrix-free
         * attempt is not made after all) */
        g_sym_cap_bound = est < est_full && na == 0 && wtime() - tsearch0 >= 0.5 * tb;
        autocp = sx(sizeof(int *) * cap);
        for (int s = 0; s < na; s++) { GEN_GROW(); gen[ngen] = sx(sizeof(int) * NI); memcpy(gen[ngen], ga + (size_t)s * NI, sizeof(int) * NI);
                                       autocp[ngen] = sx(sizeof(int) * (m + 1)); memcpy(autocp[ngen], gc + (size_t)s * (m + 1), sizeof(int) * m); ngen++; }
        free(ga); free(gc);
        /* 4.37: signed permutations: a second search on |data|, each generator's signs lifted
         * over GF(2) (sign_lift); generators that do not lift are tried in pairs (their
         * product may). Only when the data have negative values (otherwise |data| = data). */
        int hasneg = 0; for (size_t q = 0; q < ne && !hasneg; q++) if (E[q].v < 0) hasneg = 1;
        for (int c = 0; c < m && !hasneg; c++) if (P->b[c] < 0) hasneg = 1;
        if (symsigned && hasneg) {
            const double tb2 = fmax(0.05, tb - 0.5 * (wtime() - tsearch0));
            int nab = 0, *gcb = NULL;
            int *gab = sym_auto(P, off, NI, nsdp, E, ne, &nab, &gcb, verbose, tb2, symnodes, auto_mode ? symmin : 0.0, 1);
            if (est < est_full && na == 0 && nab == 0 && wtime() - tsearch0 >= 0.5 * tb) g_sym_cap_bound = 1;      /* (the same for the signed search) */
            char *islp = sx(NI + 1); for (int k = 0; k < nb; k++) for (int i = off[k]; i < off[k + 1]; i++) islp[i] = P->blk[k].type == BLK_LP;
            signed char *sg = sx(NI + 1), *ce = sx(m + 1);
            int *nl = sx(sizeof(int) * (nab + 1)), nnl = 0, nlift = 0, nsgn = 0;
            int *gg = sx(sizeof(int) * (NI + 1)), *cc = sx(sizeof(int) * (m + 1));
            const double tl0 = wtime();
            for (int s = 0; s < nab + nnl * (nnl + 1) / 2 + 1; s++) {
                const int *g, *cpp;
                if (s < nab) { g = gab + (size_t)s * NI; cpp = gcb + (size_t)s * (m + 1); }
                else {   /* the products of pairs of non-liftable generators (s - nab enumerates a <= b) */
                    int t = s - nab, a = 0; while (a < nnl && t >= nnl - a) { t -= nnl - a; a++; }
                    if (a >= nnl) break;
                    const int bq = a + t; const int *ga1 = gab + (size_t)nl[a] * NI, *gb1 = gab + (size_t)nl[bq] * NI, *ca1 = gcb + (size_t)nl[a] * (m + 1), *cb1 = gcb + (size_t)nl[bq] * (m + 1);
                    for (int i = 0; i < NI; i++) gg[i] = ga1[gb1[i]];
                    for (int c = 0; c < m; c++) cc[c] = ca1[cb1[c]];
                    int ident = 1; for (int i = 0; i < NI && ident; i++) if (gg[i] != i) ident = 0;
                    if (ident) continue;
                    g = gg; cpp = cc;
                }
                if (!sign_lift(NI, m, islp, E, ne, g, cpp, P->b, sg, ce)) { if (s < nab) nl[nnl++] = s; continue; }
                int pure = 1; for (int i = 0; i < NI && pure; i++) if (sg[i] < 0) pure = 0;
                for (int c = 0; c < m && pure; c++) if (ce[c] < 0) pure = 0;
                GEN_GROW(); gen[ngen] = sx(sizeof(int) * NI); memcpy(gen[ngen], g, sizeof(int) * NI);
                autocp[ngen] = sx(sizeof(int) * (m + 1)); memcpy(autocp[ngen], cpp, sizeof(int) * m);
                if (!pure) { gsg[ngen] = sx(NI + 1); memcpy(gsg[ngen], sg, NI); gce[ngen] = sx(m + 1); memcpy(gce[ngen], ce, m); nsgn++; }
                ngen++; nlift++;
            }
            if (verbose > 0) printf("   symmetry search on |data|: %d generator(s), %d lift to signed automorphisms (%d with signs), %d do not (%.2fs)\n", nab, nlift, nsgn, nnl, wtime() - tl0);
            free(gab); free(gcb); free(islp); free(sg); free(ce); free(nl); free(gg); free(cc);
        }
        {   /* 4.37 (research hook, inert): the generators found, for offline analysis */
            const char *dp = getenv("BRISK_SYMDUMP");
            FILE *fd = dp ? fopen(dp, "w") : NULL;
            if (fd) { fprintf(fd, "%d %d %d\n", NI, m, ngen);
                      for (int s = 0; s < ngen; s++) { for (int u = 0; u < NI; u++) fprintf(fd, gsg[s] && gsg[s][u] < 0 ? "-%d " : "%d ", gen[s][u]); fprintf(fd, "\n"); for (int c = 0; c < m; c++) fprintf(fd, gce[s] && gce[s][c] < 0 ? "-%d " : "%d ", autocp[s][c]); fprintf(fd, "\n"); }
                      fclose(fd); }
        }
        if (ngen == 0) { if (verbose >= 0) printf("presolve: symmetry search found no automorphism\n"); free(gen); free(gsg); free(gce); free(autocp); free(off); free(E); return NULL; }
    } else {
    FILE *f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "brisk: cannot open the symmetry file %s\n", fname); exit(2); }
    /* line 1: block sizes */
    { int k = 0, n;
      for (int q = 0; q < nb; q++) if (P->blk[q].type == BLK_SDP) { if (fscanf(f, "%d", &n) != 1 || n != P->blk[q].n) { fprintf(stderr, "brisk: symmetry file: block sizes do not match the problem\n"); exit(2); } k++; }
      (void)k; }
    for (;;) {
        /* 4.37: an entry "-j" is the index j with the sign -1 (a signed permutation) */
        int *g = sx(sizeof(int) * NI); signed char *sgn = sx(NI + 1); int anyneg = 0;
        int q = 0; char tok[64];
        while (q < nsdp && fscanf(f, "%63s", tok) == 1) { const int neg = tok[0] == '-'; g[q] = atoi(tok + neg); sgn[q] = neg ? -1 : 1; anyneg |= neg; q++; }
        if (q == 0) { free(g); free(sgn); break; }
        if (q != nsdp) { fprintf(stderr, "brisk: symmetry file: generator with %d entries, %d expected\n", q, nsdp); exit(2); }
        for (q = nsdp; q < NI; q++) { g[q] = q; sgn[q] = 1; }
        /* a permutation? */
        { char *seen = calloc(NI, 1); for (q = 0; q < NI; q++) { if (g[q] < 0 || g[q] >= NI || seen[g[q]]) { fprintf(stderr, "brisk: symmetry file: not a permutation\n"); exit(2); } seen[g[q]] = 1; } free(seen); }
        GEN_GROW();
        gen[ngen] = g; if (anyneg) gsg[ngen] = sgn; else free(sgn); ngen++;
    }
    fclose(f);
    }
    if (ngen == 0) { free(gen); free(gsg); free(gce); free(off); free(E); return NULL; }
    #undef GEN_GROW
    /* which block a global index belongs to (SDP indices map within SDP blocks of equal size) */
    int *gblk = sx(sizeof(int) * NI);
    for (int k = 0; k < nb; k++) for (int i = off[k]; i < off[k + 1]; i++) gblk[i] = k;
    /* the induced constraint permutation of an index permutation, with the automorphism check */
    int *cnt = calloc(m, sizeof(int));
    for (size_t q = 0; q < ne; q++) if (E[q].con >= 0) cnt[E[q].con]++;
    /* the constraint permutation induced by an index permutation g: from the entries whose
     * image has a single candidate first, then the rest (first unused candidate); then every
     * entry is checked against it */
    #define INDUCED(g, cp, fail) do { \
        for (int c = 0; c < m; c++) (cp)[c] = -1; \
        char *used_ = calloc(m, 1); \
        (fail) = NULL; \
        for (int pass_ = 0; pass_ < 2 && !(fail); pass_++) for (size_t q = 0; q < ne && !(fail); q++) { \
            const int c = E[q].con; if (c < 0 || (cp)[c] >= 0) continue; \
            const int i = (int)(E[q].key >> 32), j = (int)(E[q].key & 0xffffffffu); \
            int a = (g)[i], bb = (g)[j]; if (a > bb) { const int tt = a; a = bb; bb = tt; } \
            const uint64_t key_ = ((uint64_t)a << 32) | (uint64_t)bb; \
            const int64_t p = find_key(E, ne, key_); \
            if (p < 0) { (fail) = "an entry has no image"; break; } \
            int ncand_ = 0, cand_ = -1; \
            for (int64_t pp = p; pp < (int64_t)ne && E[pp].key == key_; pp++) \
                if (E[pp].con >= 0 && fabs(E[pp].v - E[q].v) <= 1e-12 * (1 + fabs(E[q].v))) { ncand_++; if (cand_ < 0 && !used_[E[pp].con]) cand_ = E[pp].con; } \
            if (ncand_ == 0) { (fail) = "an entry value is not invariant"; break; } \
            if (pass_ == 0 && ncand_ > 1) continue; \
            if (cand_ < 0) { (fail) = "a constraint has no free image"; break; } \
            (cp)[c] = cand_; used_[cand_] = 1; \
        } \
        free(used_); \
        if (!(fail)) for (int c = 0; c < m; c++) if ((cp)[c] < 0) { (fail) = "a constraint without entries"; break; } \
        if (!(fail)) for (size_t q = 0; q < ne && !(fail); q++) { \
            const int i = (int)(E[q].key >> 32), j = (int)(E[q].key & 0xffffffffu); \
            int a = (g)[i], bb = (g)[j]; if (a > bb) { const int tt = a; a = bb; bb = tt; } \
            const uint64_t key_ = ((uint64_t)a << 32) | (uint64_t)bb; \
            const int c2 = E[q].con < 0 ? -1 : (cp)[E[q].con]; int found_ = 0; \
            for (int64_t pp = find_key(E, ne, key_); pp >= 0 && pp < (int64_t)ne && E[pp].key == key_; pp++) \
                if (E[pp].con == c2 && fabs(E[pp].v - E[q].v) <= 1e-12 * (1 + fabs(E[q].v))) { found_ = 1; break; } \
            if (!found_) (fail) = E[q].con < 0 ? "C is not invariant" : "a constraint does not map onto its image"; \
        } \
        if (!(fail)) for (int c = 0; c < m; c++) { \
            if (cnt[c] != cnt[(cp)[c]] || fabs(P->b[c] - P->b[(cp)[c]]) > 1e-12 * (1 + fabs(P->b[c]))) { (fail) = "b or the entry counts are not invariant"; break; } \
        } \
    } while (0)
    /* the generators are checked to be automorphisms (with their induced constraint
     * permutations); the closure composes both index and constraint permutations, so every
     * element is an automorphism without further checks */
    int **gcp = sx(sizeof(int *) * ngen);
    for (int s = 0; s < ngen; s++) {
        gcp[s] = sx(sizeof(int) * m);
        const char *why;
        if (autocp) { memcpy(gcp[s], autocp[s], sizeof(int) * m); why = NULL; }   /* verified by the search (as_verify) and the sign lifting */
        else if (!gsg[s]) INDUCED(gen[s], gcp[s], why);
        else { gce[s] = sx(m + 1); why = sym_induced(m, E, ne, cnt, P->b, gen[s], gsg[s], gcp[s], gce[s]); }   /* 4.37: a signed generator */
        if (why) { fprintf(stderr, "brisk: symmetry file: generator %d is not an automorphism of the data (%s); the reduction is not applied\n", s + 1, why); exit(2); }
    }
    if (autocp) { for (int s = 0; s < ngen; s++) free(autocp[s]); free(autocp); }
    int nsigned = 0; for (int s = 0; s < ngen; s++) if (gsg[s] || gce[s]) nsigned++;
    const int GMAX = NI > 20000 ? 1 : 512;     /* the closure is only reported (the orbits and the averaging use the generators); skipped on large index sets */
    /* the closure (reported only; the orbits and the averaging use the generators): signed
     * permutations (g, s) composed as i -> g2(g1 i) with the sign s1(i) s2(g1 i) */
    int **G = sx(sizeof(int *) * GMAX), **CP = sx(sizeof(int *) * GMAX); int ng = 0;
    signed char **GS = calloc(GMAX, sizeof(signed char *));
    int *id = sx(sizeof(int) * NI); for (int i = 0; i < NI; i++) id[i] = i;
    G[ng] = id; CP[ng] = NULL; ng++;
    /* membership by hashing the index permutation and its signs */
    uint64_t *hash = sx(sizeof(uint64_t) * GMAX);
    #define PHASH(g, sg_, h) do { uint64_t hh = 1469598103934665603ull; for (int i_ = 0; i_ < NI; i_++) { hh ^= (uint64_t)(g)[i_] + ((sg_) && (sg_)[i_] < 0 ? 0x5555ull : 0) + 0x9e3779b97f4a7c15ull; hh *= 1099511628211ull; } (h) = hh; } while (0)
    PHASH(id, (signed char *)NULL, hash[0]);
    int *tmp = sx(sizeof(int) * NI); signed char *tsg = sx(NI + 1);
    int truncated = 0;
    for (int front = 0; front < ng && !truncated; front++) {
        for (int s = 0; s < ngen; s++) {
            int anyneg = 0;
            for (int i = 0; i < NI; i++) { const int gi = G[front][i]; tmp[i] = gen[s][gi]; tsg[i] = (signed char)((GS[front] ? GS[front][i] : 1) * (gsg[s] ? gsg[s][gi] : 1)); anyneg |= tsg[i] < 0; }
            uint64_t h; PHASH(tmp, anyneg ? tsg : (signed char *)NULL, h);
            int found = 0;
            for (int t = 0; t < ng && !found; t++) if (hash[t] == h && !memcmp(G[t], tmp, sizeof(int) * NI)) {
                found = 1; for (int i = 0; i < NI && found; i++) if ((GS[t] ? GS[t][i] : 1) != (anyneg ? tsg[i] : 1)) found = 0; }
            if (found) continue;
            if (ng == GMAX) { truncated = 1; break; }
            G[ng] = sx(sizeof(int) * NI); memcpy(G[ng], tmp, sizeof(int) * NI); CP[ng] = NULL;
            if (anyneg) { GS[ng] = sx(NI + 1); memcpy(GS[ng], tsg, NI); }
            hash[ng] = h; ng++;
        }
    }
    if (truncated && verbose > 0) printf("   symmetry reduction: the group has more than %d elements (the orbits come from the generators and are exact)\n", GMAX);
    for (int t = 0; t < ng; t++) free(GS[t]);
    free(GS); free(tsg);
    /* orbits of the constraints: union-find with parity (4.37): c ~ cp(c) with the sign ce(c);
     * an orbit on which a constraint meets its own negative vanishes on invariant X (b = 0
     * there) and is dropped; the others are aggregated with their signs relative to the root */
    PUF CU; CU.p = sx(sizeof(int) * (m + 1)); CU.par = calloc(m + 1, sizeof(int)); for (int c = 0; c < m; c++) CU.p[c] = c;
    int *par = CU.p;
    char *czero = calloc(m + 1, 1);
    for (int s = 0; s < ngen; s++) for (int c = 0; c < m; c++) if (!puf_union(&CU, c, gcp[s][c], gce[s] ? gce[s][c] < 0 : 0)) czero[c] = 1;
    for (int s = 0; s < ngen; s++) { free(gcp[s]); free(gce[s]); }
    free(gcp); free(gce);
    int *cls = sx(sizeof(int) * m), *root2cls = sx(sizeof(int) * m); for (int c = 0; c < m; c++) root2cls[c] = -1;
    signed char *csg = sx(m + 1);
    {   char *rzero = calloc(m + 1, 1); int pr;
        for (int c = 0; c < m; c++) if (czero[c]) rzero[puf_find(&CU, c, &pr)] = 1;
        for (int c = 0; c < m; c++) { const int r = puf_find(&CU, c, &pr); csg[c] = pr ? -1 : 1; if (rzero[r]) root2cls[r] = -2; }
        free(rzero); }
    int mr = 0, mdrop = 0;
    for (int c = 0; c < m; c++) {
        int pr; const int r = puf_find(&CU, c, &pr);
        if (root2cls[r] == -2) { cls[c] = -1; mdrop++; if (P->b[c] != 0) { fprintf(stderr, "brisk: symmetry reduction: a vanishing constraint has b != 0 (not an automorphism)\n"); exit(2); } continue; }
        if (root2cls[r] < 0) root2cls[r] = mr++;
        cls[c] = root2cls[r];
    }
    free(czero); free(CU.par);
    if (verbose >= 0) {
        printf("presolve: symmetry reduction: %d generator(s)", ngen);
        if (nsigned) printf(" (%d signed)", nsigned);
        printf(", group of order %s%d, %d constraints -> %d orbits", truncated ? "> " : "", ng, m, mr);
        if (mdrop) printf(" (%d constraints vanish on invariant X)", mdrop);
        printf("\n");
    }
    /* keep what the postsolve needs, then rebuild P */
    PSSym *S = calloc(1, sizeof(PSSym));
    S->m = m; S->mr = mr; S->order = ng; S->NI = NI; S->nblk = nb; S->cls = cls; S->csg = csg; S->gsg = gsg;
    S->G = G; S->off = off; S->gblk = gblk;
    S->ngen = ngen; S->gen = gen;
    S->b0 = sx(sizeof(double) * m); memcpy(S->b0, P->b, sizeof(double) * m);
    /* the aggregated problem as triplets: in the symmetry-adapted basis (4.33) when possible */
    size_t nt = 0; int *tc = NULL, *tb = NULL, *ti = NULL, *tj = NULL; double *tv = NULL;
    int nbq = nb, *bsz = NULL;
    /* automatic mode: the basis is worth computing only if some block can shrink enough: the
     * trivial component alone has the size of the number of index orbits o_k, so the cube
     * sum is at least o_k^3 and the gain at most (n_k / o_k)^3; with the constraints not
     * shrinking either, the reduction would not be applied anyway */
    int try_bd = symbd;
    if (auto_mode && (double)m / mr < symmin) {
        int *pi = sx(sizeof(int) * (NI + 1)); for (int i = 0; i < NI; i++) pi[i] = i;
        for (int s = 0; s < ngen; s++) for (int i = 0; i < NI; i++) { const int a = uf_find(pi, i), b2 = uf_find(pi, gen[s][i]); if (a != b2) pi[a] = b2; }
        int can = 0;
        for (int k = 0; k < nb && !can; k++) {
            if (P->blk[k].type != BLK_SDP) continue;
            int o = 0; for (int i = off[k]; i < off[k + 1]; i++) if (uf_find(pi, i) == i) o++;
            if (pow((double)P->blk[k].n / o, 3.0) >= symmin) can = 1;
        }
        free(pi);
        if (!can) {     /* (5.5: or the factorization of the Schur complement alone gains that much, the blocks unchanged) */
            double c0b = 0; for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) c0b += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
            if (sym_cost_gain(P, m, mr, c0b, c0b) >= symmin) can = 1;
        }
        if (!can) {
            if (verbose >= 0) printf("presolve: symmetry reduction not applied: too small a gain (%d -> %d constraints, no block can shrink by %g)\n", m, mr, symmin);
            for (int t = 0; t < ng; t++) free(CP[t]);
            free(CP); free(E); free(cnt); free(par); free(root2cls); free(tmp); free(hash);
            sym_free(S);
            return NULL;
        }
    }
    if (!(try_bd && sym_blockdiag(P, S, cls, off, NI, nsdp, ngen < SYM_BDMAXGEN ? ngen : SYM_BDMAXGEN, gen, gblk, verbose, &nt, &tc, &tb, &ti, &tj, &tv, &nbq, &bsz))) {
        size_t ct = ne + 16;
        tc = sx(sizeof(int) * ct); tb = sx(sizeof(int) * ct); ti = sx(sizeof(int) * ct); tj = sx(sizeof(int) * ct); tv = sx(sizeof(double) * ct);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k]; const int lp = B->type == BLK_LP;
            for (int t = -1; t < B->ncon; t++) {
                const SpSym *A = t < 0 ? &B->C : &B->A[t];
                if (t >= 0 && cls[B->con[t]] < 0) continue;      /* 4.37: vanishes on invariant X */
                const double sgc = t < 0 ? 1.0 : (double)csg[B->con[t]];
                for (int q = 0; q < A->nnz; q++) { tc[nt] = t < 0 ? -1 : cls[B->con[t]]; tb[nt] = k; ti[nt] = A->row[q]; tj[nt] = lp ? A->row[q] : A->col[q]; tv[nt] = sgc * A->val[q]; nt++; }
            }
        }
        bsz = sx(sizeof(int) * nb);
        for (int k = 0; k < nb; k++) bsz[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    }
    if (!S->bd) nt = sym_merge_trips(nt, tc, tb, ti, tj, tv, 0.0);      /* 4.37: the adapted-basis path merges class by class */
    /* worth it? (automatic mode) the constraints or the block algebra must shrink by symmin */
    if (auto_mode) {
        double c0 = 0, c1 = 0;
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) c0 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        for (int k = 0; k < nbq; k++) if (bsz[k] > 0) c1 += (double)bsz[k] * bsz[k] * bsz[k];
        const double gain = fmax(fmax((double)m / (mr > 0 ? mr : 1), c0 / (c1 > 0 ? c1 : 1)), sym_cost_gain(P, m, mr, c0, c1));
        int conflict = 0;
        if (gain >= symmin) {
            int *nr = sx(sizeof(int) * (nbq + 1)), *parent = sx(sizeof(int) * (nbq + 1)), nred = 0;
            if (S->bd) { for (int r = 0; r < S->nsb; r++) if (S->sb[r].t >= 0 && !S->sb[r].lp) { nr[nred] = S->sb[r].n; parent[nred] = S->sb[r].k; nred++; } }
            else for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) { nr[nred] = P->blk[k].n; parent[nred] = k; nred++; }
            conflict = sym_chordal_conflict(P, nred, nr, parent, chordal_minn, chordal_density);
            free(nr); free(parent);
        }
        if (gain < symmin || conflict) {
            if (verbose >= 0) printf(conflict ? "presolve: symmetry reduction not applied: it would take the chordal conversion away (one large sparse block with small ones)\n"
                                              : "presolve: symmetry reduction not applied: too small a gain (%.2f < %g)\n", gain, symmin);
            free(tc); free(tb); free(ti); free(tj); free(tv); if (!S->bd) free(bsz);
            for (int t = 0; t < ng; t++) free(CP[t]);
            free(CP); free(E); free(cnt); free(par); free(root2cls); free(tmp); free(hash);
            sym_free(S);
            return NULL;
        }
    }
    double *bn = calloc(mr, sizeof(double));
    for (int c = 0; c < m; c++) if (cls[c] >= 0) bn[cls[c]] += csg[c] * P->b[c];
    /* 4.37: the problem as read is freed before the reduced one is built (d = 5: both do not fit) */
    const double tbR0 = P->tbR, off0 = P->obj_off;
    free(E); E = NULL;
    problem_free(P);
    Problem Q; memset(&Q, 0, sizeof(Q));
    problem_from_trips(&Q, mr, nbq, bsz, bn, nt, tc, tb, ti, tj, tv);
    Q.tbR = tbR0; Q.obj_off = off0; Q.morig = mr;
    for (int i = 0; i < mr; i++) Q.orig[i] = i;      /* the reduced problem is the base of the downstream postsolve */
    *P = Q;
    free(tc); free(tb); free(ti); free(tj); free(tv); free(bn); if (!S->bd) free(bsz);
    for (int t = 0; t < ng; t++) free(CP[t]);
    free(CP);
    free(E); free(cnt); free(par); free(root2cls); free(tmp); free(hash);
    return S;
}

/* the solution of the aggregated problem -> the file's: y expanded, X averaged over the
 * group. The average is the mean over the orbit of each entry; the pair orbits are closed
 * under the generators by union-find (O(ngen n^2), independent of the group order). */
void sym_unmap(const PSSym *S, double **Xred0, const double *yred, double **Xo, double *yo, const int *bs) {
    for (int c = 0; c < S->m; c++) yo[c] = S->cls[c] < 0 ? 0.0 : (S->csg ? S->csg[c] : 1) * yred[S->cls[c]];   /* 4.37: signs; dropped constraints y = 0 */
    if (!Xred0 || !Xo) return;
    const int nb = S->nblk, NI = S->NI;
    for (int k = 0; k < nb; k++) { const int n = abs(bs[k]); memset(Xo[k], 0, sizeof(double) * (bs[k] < 0 ? (size_t)n : (size_t)n * n)); }
    double **Xred = Xred0, **Xbd = NULL;
    if (S->bd) {
        /* 4.33: back from the symmetry-adapted basis: X_k = sum_r sum_i W_i X_r W_i' (the blocks of the file) */
        Xbd = sx(sizeof(double *) * nb);
        for (int k = 0; k < nb; k++) { const int n = abs(bs[k]); Xbd[k] = calloc((bs[k] < 0 ? (size_t)n : (size_t)n * n) + 1, sizeof(double)); }
        for (int r = 0; r < S->nsb; r++) {
            const SymBlk *B = &S->sb[r]; const int k = B->k, nk = abs(bs[k]), nr = B->n;
            if (B->t < 0) { memcpy(Xbd[k], Xred0[B->rblk], sizeof(double) * nk); continue; }
            const int d = B->d; const double one = 1.0;
            const double *Xr = B->lp ? Xred0[S->lpblk] + B->p : Xred0[B->rblk];
            double *T = sx(sizeof(double) * (size_t)nk * nr);
            for (int i = 0; i < d; i++) {
                const double *W = B->W + (size_t)i * nk * nr;
                BL(dsymm_)("R", "U", &nk, &nr, &one, Xr, &nr, W, &nk, &(double){0.0}, T, &nk);
                BL(dgemm_)("N", "T", &nk, &nk, &nr, &one, T, &nk, W, &nk, &one, Xbd[k], &nk);
            }
            free(T);
        }
        /* the exchanged blocks: X_b = P_u X_b1 P_u' */
        for (int k = 0; k < nb; k++) {
            if (bs[k] < 0 || S->brep[k] == k) continue;
            const int k1 = S->brep[k], n = abs(bs[k]); const int *u = S->bmap[k];
            const signed char *su = S->bsgn ? S->bsgn[k] : NULL;
            for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Xbd[k][u[i] + (size_t)u[j] * n] = (su ? (double)(su[i] * su[j]) : 1.0) * Xbd[k1][i + (size_t)j * n];
        }
        Xred = Xbd;
    }
    /* global dense index of an SDP entry (i, j), i <= j, both in the same block: pairs of
     * different blocks never carry data. Use a per-block pair numbering. */
    size_t *poff = sx(sizeof(size_t) * (nb + 1)); poff[0] = 0;
    for (int k = 0; k < nb; k++) poff[k + 1] = poff[k] + (bs[k] < 0 ? (size_t)abs(bs[k]) : (size_t)abs(bs[k]) * abs(bs[k]));
    const size_t np = poff[nb];
    /* 4.37: with parity (signed generators): the entry (i, j) goes to (g i, g j) times s_i s_j;
     * an entry orbit that meets itself with the sign -1 is 0 on invariant X */
    PUF PU_; PU_.p = sx(sizeof(int) * np); PU_.par = calloc(np + 1, sizeof(int));
    int *par = PU_.p;
    for (size_t p = 0; p < np; p++) par[p] = (int)p;
    char *pz = calloc(np + 1, 1);
    #define PFIND(x, r) do { int pr_; (r) = puf_find(&PU_, (x), &pr_); } while (0)
    #define PUNION(a_, b_, d_) do { if (!puf_union(&PU_, (a_), (b_), (d_))) pz[(a_)] = 1; } while (0)
    for (int s = 0; s < S->ngen; s++) {
        const int *g = S->gen[s]; const signed char *gs = S->gsg ? S->gsg[s] : NULL;
        for (int k = 0; k < nb; k++) {
            const int n = abs(bs[k]);
            if (bs[k] < 0) continue;
            for (int j = 0; j < n; j++) {
                const int gj = g[S->off[k] + j], kj = S->gblk[gj], jj = gj - S->off[kj];
                for (int i = 0; i <= j; i++) {
                    const int gi = g[S->off[k] + i], ki = S->gblk[gi], ii = gi - S->off[ki];
                    if (ki != kj) continue;
                    const int a = (int)(poff[k] + (size_t)i + (size_t)j * n);
                    const int lo = ii < jj ? ii : jj, hi = ii < jj ? jj : ii;
                    const int b2 = (int)(poff[ki] + (size_t)lo + (size_t)hi * abs(bs[ki]));
                    PUNION(a, b2, gs ? ((gs[S->off[k] + i] < 0) ^ (gs[S->off[k] + j] < 0)) : 0);
                }
            }
        }
    }
    /* LP indices: the generators may permute them (automatic search); average over their orbits */
    for (int s = 0; s < S->ngen; s++) {
        const int *g = S->gen[s];
        for (int k = 0; k < nb; k++) {
            if (bs[k] >= 0) continue;
            for (int i = 0; i < abs(bs[k]); i++) {
                const int gi = g[S->off[k] + i], ki = S->gblk[gi], ii = gi - S->off[ki];
                if (bs[ki] >= 0) continue;
                PUNION((int)(poff[k] + i), (int)(poff[ki] + ii), 0);
            }
        }
    }
    for (size_t p = 0; p < np; p++) if (pz[p] == 1) { int r; PFIND((int)p, r); pz[r] = 2; }
    #define PFINDP(x, r, s_) do { int pr_; (r) = puf_find(&PU_, (x), &pr_); (s_) = pr_ ? -1.0 : 1.0; } while (0)
    double *sum = calloc(np, sizeof(double)); int *cnt = calloc(np, sizeof(int));
    for (int k = 0; k < nb; k++) {
        const int n = abs(bs[k]);
        if (bs[k] < 0) { for (int i = 0; i < n; i++) { int r; double sg; PFINDP((int)(poff[k] + i), r, sg); sum[r] += sg * Xred[k][i]; cnt[r]++; } continue; }
        for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { int r; double sg; PFINDP((int)(poff[k] + i + (size_t)j * n), r, sg); sum[r] += sg * Xred[k][i + (size_t)j * n]; cnt[r]++; }
    }
    for (int k = 0; k < nb; k++) {
        const int n = abs(bs[k]);
        if (bs[k] < 0) { for (int i = 0; i < n; i++) { int r; double sg; PFINDP((int)(poff[k] + i), r, sg); Xo[k][i] = pz[r] == 2 ? 0.0 : sg * sum[r] / cnt[r]; } continue; }
        for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { int r; double sg; PFINDP((int)(poff[k] + i + (size_t)j * n), r, sg); const double v = pz[r] == 2 ? 0.0 : sg * sum[r] / cnt[r]; Xo[k][i + (size_t)j * n] = v; Xo[k][j + (size_t)i * n] = v; }
    }
    #undef PFINDP
    #undef PUNION
    (void)NI;
    free(par); free(PU_.par); free(pz); free(sum); free(cnt); free(poff);
    if (Xbd) { for (int k = 0; k < nb; k++) free(Xbd[k]); free(Xbd); }
}

void sym_free(PSSym *S) {
    if (!S) return;
    for (int t = 0; t < S->order; t++) free(S->G[t]);
    for (int s = 0; s < S->ngen; s++) free(S->gen[s]);
    if (S->gsg) { for (int s = 0; s < S->ngen; s++) free(S->gsg[s]); free(S->gsg); }
    if (S->bsgn) { for (int k = 0; k < S->nblk; k++) free(S->bsgn[k]); free(S->bsgn); }
    free(S->csg);
    for (int r = 0; r < S->nsb; r++) free(S->sb[r].W);
    if (S->bmap) for (int k = 0; k < S->nblk; k++) free(S->bmap[k]);
    free(S->sb); free(S->bsr); free(S->brep); free(S->bmap);
    free(S->gen); free(S->G); free(S->cls); free(S->off); free(S->gblk); free(S->b0); free(S);
}

/* ---- 4.33: sign symmetries ----------------------------------------------------------------
 * Diagonal sign matrices D = diag(s), s in {±1}^n on the SDP indices, with D A_c D = ± A_c for
 * every c (b_c = 0 where the sign is -), D C D = C: the group of the even/odd structure of a
 * polynomial problem (x_i -> -x_i maps every monomial to ± itself, so the moment/SOS matrices
 * are invariant up to signs). An invariant X (D X D = X for every D of the group) vanishes
 * between indices of different sign patterns, so every block splits into one block per
 * pattern, and the constraints with D A_c D = -A_c drop (they hold on an invariant X: <A_c,
 * X> = -<A_c, X> = 0 = b_c) with y_c = 0. Exact: X of the reduced problem embedded is
 * invariant and feasible; the average of any feasible X over the group is block-diagonal
 * with the same objective. The paper's sign reduction of Example 8.1.3, found from the
 * data: 445 -> 130 + 105 + 105 + 105, 11 346 -> 3 018 rows.
 *
 * The signs solve a linear system over GF(2): s_i + s_j = e_c for every entry (i, j) of
 * A_c, s_i + s_j = 0 for C's entries, e_c = 0 when b_c != 0 or A_c has a diagonal or LP
 * entry. Union-find with parity absorbs the two-variable equations, repeatedly (an entry
 * inside one component fixes e_c and turns the constraint's other entries into two-variable
 * equations); what remains is eliminated densely over the components. The kernel's basis
 * gives the generators; the sign pattern of an index over them is its block.               */
PSSign *sign_reduce(Problem *P, int verbose, int auto_mode, double symmin, int chordal_minn, double chordal_density) {
    const int m = P->m, nb = P->nblk;
    int *off = sx(sizeof(int) * (nb + 1)); off[0] = 0;
    for (int k = 0; k < nb; k++) off[k + 1] = off[k] + P->blk[k].n;
    const int NI = off[nb];
    char *issdp = sx(NI + 1);
    for (int k = 0; k < nb; k++) for (int i = off[k]; i < off[k + 1]; i++) issdp[i] = P->blk[k].type == BLK_SDP;
    PUF U; U.p = sx(sizeof(int) * (NI + 1)); U.par = calloc(NI + 1, sizeof(int)); for (int i = 0; i < NI; i++) U.p[i] = i;
    /* the entries of each constraint over all blocks (global indices), CSR */
    int *cp = calloc(m + 2, sizeof(int));
    for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; for (int t = 0; t < B->ncon; t++) cp[B->con[t] + 1] += B->A[t].nnz; }
    for (int c = 0; c < m; c++) cp[c + 1] += cp[c];
    const int nce = cp[m];
    int *ci = sx(sizeof(int) * (nce + 1)), *cj = sx(sizeof(int) * (nce + 1));
    { int *fill = sx(sizeof(int) * (m + 1)); memcpy(fill, cp, sizeof(int) * m);
      for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int lp = B->type == BLK_LP;
          for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; const int c = B->con[t];
              for (int q = 0; q < A->nnz; q++) { const int w = fill[c]++; ci[w] = off[k] + A->row[q]; cj[w] = off[k] + (lp ? A->row[q] : A->col[q]); } } }
      free(fill); }
    /* e_c: -1 unknown, 0, 1 (bit 2: its entries are in the union-find) */
    int *e = sx(sizeof(int) * (m + 1)); for (int c = 0; c < m; c++) e[c] = fabs(P->b[c]) > 0 ? 0 : -1;
    int ok = 1;
    for (int k = 0; k < nb && ok; k++) {
        const Block *B = &P->blk[k]; const int lp = B->type == BLK_LP;
        for (int q = 0; q < B->C.nnz && ok; q++) { const int i = off[k] + B->C.row[q], j = off[k] + (lp ? B->C.row[q] : B->C.col[q]); if (!lp && i != j) ok = puf_union(&U, i, j, 0); }
    }
    for (int c = 0; c < m; c++) for (int q = cp[c]; q < cp[c + 1]; q++) if (ci[q] == cj[q] || !issdp[ci[q]]) { e[c] = 0; break; }   /* a diagonal or LP entry */
    /* rounds: known e_c -> two-variable equations; an entry within a component -> e_c known */
    int changed = 1, rounds = 0;
    while (changed && ok && rounds < 1000) {
        changed = 0; rounds++;
        for (int c = 0; c < m && ok; c++) {
            if (e[c] < 0) {
                for (int q = cp[c]; q < cp[c + 1]; q++) { int pi, pj; if (puf_find(&U, ci[q], &pi) == puf_find(&U, cj[q], &pj)) { e[c] = pi ^ pj; break; } }
            }
            if (e[c] >= 0 && !(e[c] & 2)) {
                for (int q = cp[c]; q < cp[c + 1] && ok; q++) if (issdp[ci[q]] && issdp[cj[q]]) ok = puf_union(&U, ci[q], cj[q], e[c] & 1);
                e[c] |= 2; changed = 1;
            }
        }
    }
    /* the components (roots) and the constraints still unknown */
    int *root = sx(sizeof(int) * (NI + 1)), *rpar = sx(sizeof(int) * (NI + 1)), *rid = sx(sizeof(int) * (NI + 1)); int K = 0;
    for (int i = 0; i < NI; i++) { root[i] = puf_find(&U, i, &rpar[i]); rid[i] = -1; }
    for (int i = 0; i < NI; i++) if (rid[root[i]] < 0) rid[root[i]] = K++;
    /* LP indices are fixed (+1): each LP index is its own component with s = 0: equations r = 0 */
    /* sparse GF(2) elimination on the K root variables (rows as sorted column lists; the
     * systems are local, dense bitsets cost K^2 on TSSOS files with 20 000 components) */
    const double tsg0 = wtime(); const int sgdbg = getenv("BRISK_SETUPT") != NULL;
    typedef struct { int *c; int n; } SRow;
    SRow *prow = sx(sizeof(SRow) * (K + 1)); int npiv = 0;
    int *pivof = sx(sizeof(int) * (K + 1)); for (int r = 0; r < K; r++) pivof[r] = -1;   /* column -> pivot row */
    size_t fill = 0; const size_t fillmax = (size_t)50 << 20;
    int *wa = sx(sizeof(int) * (K + 8)), *wb = sx(sizeof(int) * (K + 8));   /* working rows */
    #define ADDROW(cols_, n_) do { \
        int wn = (n_); memcpy(wa, (cols_), sizeof(int) * wn); qsort(wa, wn, sizeof(int), cmp_int); \
        { int w2 = 0; for (int q = 0; q < wn; q++) { if (w2 && wa[w2 - 1] == wa[q]) w2--; else wa[w2++] = wa[q]; } wn = w2; }   /* x + x = 0 */ \
        while (wn > 0 && pivof[wa[0]] >= 0) {                     /* reduce the leading column by its pivot row (merge = xor) */ \
            const SRow *R = &prow[pivof[wa[0]]]; int i1 = 0, i2 = 0, w2 = 0; \
            while (i1 < wn || i2 < R->n) { \
                if (i2 >= R->n || (i1 < wn && wa[i1] < R->c[i2])) wb[w2++] = wa[i1++]; \
                else if (i1 >= wn || R->c[i2] < wa[i1]) wb[w2++] = R->c[i2++]; \
                else { i1++; i2++; } } \
            memcpy(wa, wb, sizeof(int) * w2); wn = w2; \
        } \
        if (wn > 0) { prow[npiv].c = sx(sizeof(int) * wn); memcpy(prow[npiv].c, wa, sizeof(int) * wn); prow[npiv].n = wn; pivof[wa[0]] = npiv++; fill += wn; if (fill > fillmax) ok = 0; } \
    } while (0)
    if (ok) {
        for (int i = 0; i < NI && ok; i++) if (!issdp[i]) { int cc[1] = { rid[root[i]] }; ADDROW(cc, 1); }
        for (int c = 0; c < m && ok; c++) {
            if (e[c] >= 0) continue;
            for (int q = cp[c] + 1; q < cp[c + 1] && ok; q++) {
                const int cc[4] = { rid[root[ci[q - 1]]], rid[root[cj[q - 1]]], rid[root[ci[q]]], rid[root[cj[q]]] };
                ADDROW(cc, 4);
            }
        }
    }
    #undef ADDROW
    if (sgdbg) printf("   [sign: union-find %d rounds, %d components, elimination %d pivots, fill %zu at %.2fs]\n", rounds, K, npiv, fill, wtime() - tsg0);
    /* reduced echelon form: from the last pivot (largest column) down, remove the later pivot
     * columns of each row (their rows are already reduced: pivot + free columns only) */
    if (ok) {
        int *order = sx(sizeof(int) * (npiv + 1)); for (int pv = 0; pv < npiv; pv++) order[pv] = pv;
        /* rows in increasing pivot column */
        { KN *kn = sx(sizeof(KN) * (npiv + 1)); for (int pv = 0; pv < npiv; pv++) { kn[pv].key = (uint64_t)prow[pv].c[0]; kn[pv].node = pv; } qsort(kn, npiv, sizeof(KN), cmp_kn); for (int pv = 0; pv < npiv; pv++) order[pv] = kn[pv].node; free(kn); }
        for (int o = npiv - 1; o >= 0 && ok; o--) {
            SRow *R = &prow[order[o]];
            int wn = R->n; memcpy(wa, R->c, sizeof(int) * wn);
            for (;;) {
                int hit = -1; for (int q = 1; q < wn; q++) if (pivof[wa[q]] >= 0) { hit = q; break; }
                if (hit < 0) break;
                const SRow *T = &prow[pivof[wa[hit]]]; int i1 = 0, i2 = 0, w2 = 0;
                while (i1 < wn || i2 < T->n) {
                    if (i2 >= T->n || (i1 < wn && wa[i1] < T->c[i2])) wb[w2++] = wa[i1++];
                    else if (i1 >= wn || T->c[i2] < wa[i1]) wb[w2++] = T->c[i2++];
                    else { i1++; i2++; } }
                memcpy(wa, wb, sizeof(int) * w2); wn = w2;
                fill += w2; if (fill > fillmax) { ok = 0; break; }
            }
            if (wn != R->n) { free(R->c); R->c = sx(sizeof(int) * wn); memcpy(R->c, wa, sizeof(int) * wn); R->n = wn; }
        }
        free(order);
    }
    /* the sign pattern of a component over the kernel basis (one vector per free column f:
     * x_f = 1, x_pivot = the row's coefficient at f): a free column is its own pattern, a
     * pivot column the set of free columns in its reduced row; hashed */
    int ngen = 0;
    uint64_t *cpat = sx(sizeof(uint64_t) * (K + 1));
    if (ok) {
        ngen = K - npiv;
        for (int r = 0; r < K; r++) {
            uint64_t h = 1469598103934665603ull;
            if (pivof[r] < 0) { h ^= (uint64_t)r + 0x9e3779b97f4a7c15ull; h = h64(h); }
            else { const SRow *R = &prow[pivof[r]]; for (int q = 1; q < R->n; q++) { h ^= (uint64_t)R->c[q] + 0x9e3779b97f4a7c15ull; h *= 1099511628211ull; h = h64(h); } }
            cpat[r] = h;
        }
    }
    for (int pv = 0; pv < npiv; pv++) free(prow[pv].c);
    free(prow); free(pivof); free(wa); free(wb);
    if (sgdbg) printf("   [sign: %d generators at %.2fs]\n", ngen, wtime() - tsg0);
    PSSign *S = NULL;
    if (ok && ngen > 0) {
        /* sign patterns -> classes per block; constraints kept when D A_c D = A_c for every generator */
        /* the sign pattern of an index: that of its component (a hash collision only merges
         * two classes: still exact, since constraints are kept only when all their entries
         * stay inside one class) */
        uint64_t *pat = sx(sizeof(uint64_t) * (NI + 1));
        for (int i = 0; i < NI; i++) pat[i] = rpar[i] ? h64(cpat[rid[root[i]]] ^ 0x5bd1e9955bd1e995ull) : cpat[rid[root[i]]];   /* (parity 1: the complementary pattern; never here) */
        int *cls = sx(sizeof(int) * (NI + 1)); int ncls = 0;
        int *bcls0 = sx(sizeof(int) * (nb + 1));
        for (int k = 0; k < nb; k++) {
            bcls0[k] = ncls;
            if (P->blk[k].type == BLK_LP) { for (int i = off[k]; i < off[k + 1]; i++) cls[i] = ncls; ncls++; continue; }
            /* distinct patterns in the block, numbered in order of first appearance */
            { const int n = off[k + 1] - off[k]; KN *kn = sx(sizeof(KN) * (n + 1));
              for (int i = 0; i < n; i++) { kn[i].key = pat[off[k] + i]; kn[i].node = off[k] + i; }
              qsort(kn, n, sizeof(KN), cmp_kn);
              int *first = sx(sizeof(int) * (n + 1)); int ng2 = 0;
              for (int q = 0; q < n; q++) { if (q == 0 || kn[q].key != kn[q - 1].key) first[ng2++] = kn[q].node; cls[kn[q].node] = first[ng2 - 1]; }   /* class = its first index for now */
              /* renumber by first appearance */
              int *ord2 = sx(sizeof(int) * (ng2 + 1)); memcpy(ord2, first, sizeof(int) * ng2);
              /* 4.42: tmpn is indexed by the class's first index, inside this block: a block-sized
               * array (an NI-sized one per block made the step quadratic: 100 000 blocks 0.67 s) */
              { const int o0 = off[k]; int *tmpn = sx(sizeof(int) * (n + 1)); qsort(ord2, ng2, sizeof(int), cmp_int);
                for (int q = 0; q < ng2; q++) tmpn[ord2[q] - o0] = ncls + q;
                for (int i = off[k]; i < off[k + 1]; i++) cls[i] = tmpn[cls[i] - o0];
                free(tmpn); }
              ncls += ng2; free(kn); free(first); free(ord2); }
        }
        int *csz = calloc(ncls + 1, sizeof(int)), *cpos = sx(sizeof(int) * (NI + 1));
        for (int i = 0; i < NI; i++) cpos[i] = csz[cls[i]]++;
        char *keep = sx(m + 1); int mk = 0;
        for (int c = 0; c < m; c++) keep[c] = 1;
        for (int c = 0; c < m; c++) for (int q = cp[c]; q < cp[c + 1]; q++) if (pat[ci[q]] != pat[cj[q]]) { keep[c] = 0; break; }
        for (int c = 0; c < m; c++) mk += keep[c];
        double c0 = 0, c1 = 0;
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) c0 += (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        for (int q = 0; q < ncls; q++) c1 += (double)csz[q] * csz[q] * csz[q];
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_LP) c1 -= (double)P->blk[k].n * P->blk[k].n * P->blk[k].n;
        const double gain = fmax(fmax((double)m / (mk > 0 ? mk : 1), c0 / (c1 > 0 ? c1 : 1)), sym_cost_gain(P, m, mk, c0, c1));
        int conflict = 0;
        if (auto_mode && gain >= symmin) {
            int *nr = sx(sizeof(int) * (ncls + 1)), *parent = sx(sizeof(int) * (ncls + 1)), nred = 0;
            for (int k = 0; k < nb; k++) { if (P->blk[k].type == BLK_LP) continue; for (int q = bcls0[k]; q < (k + 1 < nb ? bcls0[k + 1] : ncls); q++) { nr[nred] = csz[q]; parent[nred] = k; nred++; } }
            conflict = sym_chordal_conflict(P, nred, nr, parent, chordal_minn, chordal_density);
            free(nr); free(parent);
        }
        if (auto_mode && (gain < symmin || conflict)) {
            if (verbose >= 0) printf(conflict ? "presolve: sign symmetry: %d generator(s), reduction not applied: it would take the chordal conversion away (one large sparse block with small ones)\n"
                                              : "presolve: sign symmetry: %d generator(s), reduction not applied: too small a gain (%.2f < %g)\n", ngen, gain, symmin);
        } else {
            S = calloc(1, sizeof(PSSign));
            S->m = m; S->mk = mk; S->nblk = nb; S->ncls = ncls; S->ngen = ngen; S->NI = NI;
            S->keep = keep; keep = NULL; S->cls = cls; cls = NULL; S->cpos = cpos; cpos = NULL; S->csz = csz; csz = NULL; S->off = off; off = NULL;
            S->bs = sx(sizeof(int) * nb); for (int k = 0; k < nb; k++) S->bs[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
            /* the reduced problem's blocks: the classes of size >= 2 of the SDP blocks, the LP
             * blocks, then one LP block with every class of size 1 (an LP variable) */
            int *qblk = sx(sizeof(int) * (ncls + 1)), *qpos0 = calloc(ncls + 1, sizeof(int)); int rb = 0, n1 = 0;
            for (int q = 0; q < ncls; q++) qblk[q] = -1;
            for (int k = 0; k < nb; k++) { if (S->bs[k] < 0) continue; for (int q = bcls0[k]; q < (k + 1 < nb ? bcls0[k + 1] : ncls); q++) if (S->csz[q] >= 2) qblk[q] = rb++; }
            for (int k = 0; k < nb; k++) if (S->bs[k] < 0) qblk[bcls0[k]] = rb++;
            const int lpblk = rb;
            for (int q = 0; q < ncls; q++) if (qblk[q] < 0) { qblk[q] = lpblk; qpos0[q] = n1++; }
            if (n1) rb++;
            S->nbr = rb; S->qblk = qblk; S->qpos0 = qpos0;
            int *newc = sx(sizeof(int) * (m + 1)); { int q = 0; for (int c = 0; c < m; c++) newc[c] = S->keep[c] ? q++ : -1; }
            size_t nt = 0, ne = 0;
            for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; ne += B->C.nnz; for (int t = 0; t < B->ncon; t++) ne += B->A[t].nnz; }
            int *tc = sx(sizeof(int) * (ne + 1)), *tb = sx(sizeof(int) * (ne + 1)), *ti = sx(sizeof(int) * (ne + 1)), *tj = sx(sizeof(int) * (ne + 1)); double *tv = sx(sizeof(double) * (ne + 1));
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; const int lp = B->type == BLK_LP;
                for (int t = -1; t < B->ncon; t++) {
                    const SpSym *A = t < 0 ? &B->C : &B->A[t];
                    const int c = t < 0 ? -1 : newc[B->con[t]]; if (t >= 0 && c < 0) continue;
                    for (int q = 0; q < A->nnz; q++) {
                        const int i = S->off[k] + A->row[q], j = S->off[k] + (lp ? A->row[q] : A->col[q]);
                        if (S->cls[i] != S->cls[j]) continue;   /* C: never (invariant); kept constraints: never */
                        const int qq = S->cls[i];
                        tc[nt] = c; tb[nt] = qblk[qq]; ti[nt] = S->cpos[i] + qpos0[qq]; tj[nt] = S->cpos[j] + qpos0[qq]; tv[nt] = A->val[q]; nt++;
                    }
                }
            }
            int *bsz = sx(sizeof(int) * (rb + 1));
            for (int k = 0; k < nb; k++) for (int q = bcls0[k]; q < (k + 1 < nb ? bcls0[k + 1] : ncls); q++) { if (S->bs[k] < 0) bsz[qblk[q]] = -S->csz[q]; else if (S->csz[q] >= 2) bsz[qblk[q]] = S->csz[q]; }
            if (n1) bsz[lpblk] = -n1;
            double *bn = sx(sizeof(double) * (mk + 1)); for (int c = 0; c < m; c++) if (S->keep[c]) bn[newc[c]] = P->b[c];
            Problem Q; memset(&Q, 0, sizeof(Q));
            problem_from_trips(&Q, mk, rb, bsz, bn, nt, tc, tb, ti, tj, tv);
            Q.tbR = P->tbR; Q.obj_off = P->obj_off; Q.morig = mk;
            for (int i = 0; i < mk; i++) Q.orig[i] = i;
            problem_free(P); *P = Q;
            if (verbose >= 0) {
                printf("presolve: sign symmetry: %d generator(s), %d -> %d constraints, blocks", ngen, m, mk);
                for (int k = 0; k < nb; k++) { if (S->bs[k] < 0) continue; printf(" %d ->", S->bs[k]); int c1 = 0; for (int q = bcls0[k]; q < (k + 1 < nb ? bcls0[k + 1] : ncls); q++) { if (S->csz[q] >= 2) printf(" %d", S->csz[q]); else c1++; } if (c1) printf(" %dx1", c1); printf(k + 1 < nb && S->bs[k + 1] > 0 ? "," : ""); }
                printf("%s\n", n1 ? " (1 x 1 pieces are LP variables)" : "");
            }
            free(newc); free(tc); free(tb); free(ti); free(tj); free(tv); free(bsz); free(bn);
        }
        free(pat); free(cls); free(cpos); free(csz); free(keep); free(bcls0);
    } else if (verbose > 0 && !ok) printf("   sign symmetry: the system has no solution beyond the identity\n");
    free(cpat); free(U.p); free(U.par); free(e); free(root); free(rpar); free(rid); free(off); free(cp); free(ci); free(cj); free(issdp);
    return S;
}

/* back to the file's blocks: X embedded (zero between sign classes), y = 0 on the dropped constraints */
void sign_unmap(const PSSign *S, double **Xred, const double *yred, double **Xo, double *yo) {
    { int q = 0; for (int c = 0; c < S->m; c++) yo[c] = S->keep[c] ? yred[q++] : 0.0; }
    if (!Xred || !Xo) return;
    for (int k = 0; k < S->nblk; k++) {
        const int n = abs(S->bs[k]);
        if (S->bs[k] < 0) { for (int i = 0; i < n; i++) { const int q = S->cls[S->off[k] + i]; Xo[k][i] = Xred[S->qblk[q]][S->cpos[S->off[k] + i] + S->qpos0[q]]; } continue; }
        memset(Xo[k], 0, sizeof(double) * (size_t)n * n);
        for (int j = 0; j < n; j++) { const int cj = S->cls[S->off[k] + j], pj = S->cpos[S->off[k] + j];
            if (S->csz[cj] == 1) { Xo[k][j + (size_t)j * n] = Xred[S->qblk[cj]][S->qpos0[cj]]; continue; }
            for (int i = 0; i < n; i++) { const int ci = S->cls[S->off[k] + i]; if (ci != cj) continue;
                Xo[k][i + (size_t)j * n] = Xred[cj >= 0 ? S->qblk[cj] : 0][S->cpos[S->off[k] + i] + (size_t)pj * S->csz[cj]]; } }
    }
}

void sign_free(PSSign *S) { if (!S) return; free(S->keep); free(S->cls); free(S->cpos); free(S->csz); free(S->off); free(S->bs); free(S->qblk); free(S->qpos0); free(S); }
