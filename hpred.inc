/* SPDX-License-Identifier: Apache-2.0
 * hpred.inc (5.9): the reductions of the double pipeline in the high-precision solver.
 *
 * The high-precision driver had a presolve of its own (blocks split by their pattern, the dual
 * form, eliminations and simple faces: hppre.inc) but none of the reductions that make the
 * double solver fast on structured problems: its iterations ran on the problem as read. Here
 * the same reductions are applied to the data in the master precision, in the order of the
 * double pipeline:
 *
 *   1. sign symmetry (symred.c: sign_reduce): the classes of indices and the constraints that
 *      vanish on invariant X are combinatorial; the reduced problem is a selection of entries,
 *      exact.
 *   2. permutation symmetry (symred.c: sym_reduce, without its adapted basis): the constraints
 *      of an orbit are summed, exact; X of the aggregated problem is averaged over the group at
 *      the end (the mean over the orbits of the entries), exact as well.
 *   3. the block diagonalisation of the matrix *-algebra the data of a block generate (as in
 *      symalg.c, which is run on a double copy as the test for structure). The basis is
 *      computed here in the master precision: the eigenspaces of a random combination S1 of
 *      the data, joined into components and aligned by a second one S2, verified on a third.
 *      The reduced data W A W' are then exact to the master precision (the target and 192 guard
 *      bits), like the eliminations of hppre.inc.
 *   4. the chordal decomposition (chordal.c, overlap equalities): the cliques are taken from the
 *      conversion of a double copy; the entries are copied, the overlap rows are +-1: exact.
 *      X of the problem as read is the positive semidefinite completion of the clique blocks,
 *      computed in the master precision.
 *
 * The detection (groups, classes, cliques) runs on a double copy of the current problem by
 * the code of the double pipeline, which keeps its rules and its messages; only the maps are
 * taken from it. The solution is mapped back through the stages and measured on the problem
 * as read by the driver; should a reduction not hold in the working precision (a symmetry of
 * the rounded data only), that measure shows it and the driver solves again without. */

/* ---- a double copy of a job */
static void hpred_shadow(const HPJob *J, Problem *P) {
    const size_t Wm = (size_t)J->Lm + 2;
    const size_t nt = J->nnzA + J->nnzC;
    int *tc = (int *)malloc(sizeof(int) * (nt + 1)), *tb = (int *)malloc(sizeof(int) * (nt + 1)), *ti = (int *)malloc(sizeof(int) * (nt + 1)), *tj = (int *)malloc(sizeof(int) * (nt + 1));
    double *tv = (double *)malloc(sizeof(double) * (nt + 1)), *bd = (double *)malloc(sizeof(double) * ((size_t)J->m + 1));
    size_t w = 0;
    for (int k = 0; k < J->nblk; k++) {
        const int lp = J->bs[k] < 0;
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) { tc[w] = -1; tb[w] = k; ti[w] = J->ci[e]; tj[w] = lp ? J->ci[e] : J->cj[e]; tv[w] = mpx_getd(J->Cv + e * Wm); if (tv[w] == 0.0 && mpx_sgn(J->Cv + e * Wm) != 0) tv[w] = mpx_sgn(J->Cv + e * Wm) * 1e-300; w++; }
        for (int tt = 0; tt < J->ncon[k]; tt++)
            for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) { tc[w] = J->con[k][tt]; tb[w] = k; ti[w] = J->ei[e]; tj[w] = lp ? J->ei[e] : J->ej[e]; tv[w] = mpx_getd(J->Av + e * Wm); if (tv[w] == 0.0 && mpx_sgn(J->Av + e * Wm) != 0) tv[w] = mpx_sgn(J->Av + e * Wm) * 1e-300; w++; }
    }
    for (int p = 0; p < J->m; p++) { bd[p] = mpx_getd(J->b + (size_t)p * Wm); if (bd[p] == 0.0 && mpx_sgn(J->b + (size_t)p * Wm) != 0) bd[p] = mpx_sgn(J->b + (size_t)p * Wm) * 1e-300; }   /* (nonzero stays nonzero) */
    problem_from_trips(P, J->m, J->nblk, J->bs, bd, w, tc, tb, ti, tj, tv);
    free(tc); free(tb); free(ti); free(tj); free(tv); free(bd);
}

/* ---- a list of entries with values in the master precision, and the job built from it */
typedef struct { HEnt *E; uint64_t *v; size_t n, cap, Wm; } HTl;
static void htl_init(HTl *T, size_t Wm, size_t cap) { T->Wm = Wm; T->n = 0; T->cap = cap < 64 ? 64 : cap; T->E = (HEnt *)malloc(sizeof(HEnt) * T->cap); T->v = (uint64_t *)malloc(sizeof(uint64_t) * T->cap * Wm); }
static void htl_free(HTl *T) { free(T->E); free(T->v); T->E = NULL; T->v = NULL; }
/* adds the entry (i, j of an SDP block in any order) and returns its value slot */
static uint64_t *htl_add(HTl *T, int con, int blk, int i, int j) {
    if (T->n == T->cap) { T->cap *= 2; T->E = (HEnt *)realloc(T->E, sizeof(HEnt) * T->cap); T->v = (uint64_t *)realloc(T->v, sizeof(uint64_t) * T->cap * T->Wm); if (!T->E || !T->v) { fprintf(stderr, "brisk: out of memory (high precision)\n"); exit(1); } }
    HEnt *e = &T->E[T->n];
    e->con = con; e->blk = blk; e->i = i < j ? i : j; e->j = i < j ? j : i; e->idx = T->n;
    return T->v + (T->n++) * T->Wm;
}
/* the job of the list (bs and b are taken over); merge: entries given more than once are
 * summed, sums that are exactly zero dropped. Returns 0 on success. */
static int htl_build(HTl *T, HPJob *Jn, int Lm, int m, int nb, int *bs, uint64_t *b, int merge) {
    const size_t Wm = T->Wm;
    if (merge && T->n > 1) {
        qsort(T->E, T->n, sizeof(HEnt), cmp_hent);
        size_t w = 0;
        for (size_t q = 0; q < T->n; ) {
            size_t q1 = q + 1;
            uint64_t *v0 = T->v + T->E[q].idx * Wm;
            while (q1 < T->n && cmp_hent(&T->E[q1], &T->E[q]) == 0) { mpx_add(v0, v0, T->v + T->E[q1].idx * Wm); q1++; }
            if (mpx_sgn(v0) != 0) T->E[w++] = T->E[q];
            q = q1;
        }
        T->n = w;
    }
    memset(Jn, 0, sizeof(*Jn));
    Jn->Lm = Lm; Jn->m = m; Jn->nblk = nb; Jn->bs = bs; Jn->b = b;
    return hp_build(Jn, T->E, T->n, T->v);
}

/* ---- the stages */
typedef struct { int k, d, mult, rblk, rpos; uint64_t *W; } HACmp;   /* a component of block k: mult copies of dimension d; W: (mult d) rows of length n_k, row r d + i: copy r, index i; rblk: its block (-1: dropped, X = 0 there), rpos: its place in the LP block (d = 1) */
typedef struct {
    int kind;                    /* 1 sign symmetry, 2 orbits, 3 algebra, 4 chordal, 5 empty rows removed */
    int m0, nb0, *bs0;           /* the layout before the stage */
    int m1, nb1, *bs1;           /* and after */
    PSSign *sg; PSSym *sy;
    int *brep, **bu, *bnew; signed char **bsg;     /* orbits: the blocks the group exchanges: block -> its representative; index of the representative -> index of the block, its sign; block -> block after (-1: not kept) */
    int nc; HACmp *cmp; int *bmap;                 /* algebra: components; block before -> after (-1: decomposed) */
    int *cmap;                                     /* empty rows: constraint before -> after (-1: removed) */
    int nch; PSChordal *ch; int *cbs;              /* chordal: the records in order; cbs: block sizes after the stage */
} HPStage;
static void hpst_free(HPStage *S);
#define HPRED_MAX 8
typedef struct { int n; HPStage st[HPRED_MAX]; HPJob Jfull; } HPRed;

static void hpst_layout(HPStage *S, const HPJob *J0) {
    S->m0 = J0->m; S->nb0 = J0->nblk;
    S->bs0 = (int *)malloc(sizeof(int) * ((size_t)J0->nblk + 1)); memcpy(S->bs0, J0->bs, sizeof(int) * (size_t)J0->nblk);
}
static uint64_t *hpred_bcopy(const HPJob *J, int m) { const size_t Wm = (size_t)J->Lm + 2; uint64_t *b = (uint64_t *)calloc(((size_t)m + 1) * Wm, sizeof(uint64_t)); return b; }

/* -- 1. sign symmetry */
static int hpred_sign(HPStage *S, const HPJob *J, HPJob *Jn, const Params *par) {
    if (!(par->symfile && par->symsign && !strcmp(par->symfile, "auto"))) return 0;
    Problem P; hpred_shadow(J, &P);
    PSSign *G = sign_reduce(&P, par->verbose, 1, par->symmin, par->chordal != 0 ? par->chordal_minn : 1 << 30, par->chordal_density);
    if (!G) { problem_free(&P); return 0; }
    const size_t Wm = (size_t)J->Lm + 2;
    const int nbr = P.nblk, mk = P.m;
    int *bs = (int *)malloc(sizeof(int) * ((size_t)nbr + 1));
    for (int k = 0; k < nbr; k++) bs[k] = P.blk[k].type == BLK_LP ? -P.blk[k].n : P.blk[k].n;
    problem_free(&P);
    int *newc = (int *)malloc(sizeof(int) * ((size_t)J->m + 1)); { int q = 0; for (int c = 0; c < J->m; c++) newc[c] = G->keep[c] ? q++ : -1; }
    uint64_t *b = hpred_bcopy(J, mk);
    for (int c = 0; c < J->m; c++) if (newc[c] >= 0) memcpy(b + (size_t)newc[c] * Wm, J->b + (size_t)c * Wm, Wm * sizeof(uint64_t));
    HTl T; htl_init(&T, Wm, J->nnzA + J->nnzC + 1);
    int bad = 0;
    for (int k = 0; k < J->nblk && !bad; k++) {
        const int lp = J->bs[k] < 0;
        for (int tt = -1; tt < J->ncon[k] && !bad; tt++) {
            const int c = tt < 0 ? -1 : newc[J->con[k][tt]];
            if (tt >= 0 && c < 0) continue;
            const size_t e0 = tt < 0 ? J->cptr[k] : J->ptr[k][tt], e1 = tt < 0 ? J->cptr[k + 1] : J->ptr[k][tt + 1];
            for (size_t e = e0; e < e1; e++) {
                const int i = G->off[k] + (tt < 0 ? J->ci[e] : J->ei[e]), j = lp ? i : G->off[k] + (tt < 0 ? J->cj[e] : J->ej[e]);
                if (G->cls[i] != G->cls[j]) { bad = 1; break; }
                const int q = G->cls[i], rb = G->qblk[q];
                const int ni = G->cpos[i] + G->qpos0[q], nj = G->cpos[j] + G->qpos0[q];
                memcpy(htl_add(&T, c, rb, ni, bs[rb] < 0 ? ni : nj), (tt < 0 ? J->Cv : J->Av) + e * Wm, Wm * sizeof(uint64_t));
            }
        }
    }
    free(newc);
    if (bad || htl_build(&T, Jn, J->Lm, mk, nbr, bs, b, 0)) { if (bad) { free(bs); free(b); } else hp_free(Jn); htl_free(&T); sign_free(G); return 0; }
    htl_free(&T);
    memset(S, 0, sizeof(*S)); S->kind = 1; S->sg = G; hpst_layout(S, J); S->m1 = mk; S->nb1 = nbr;
    return 1;
}
static void hpred_sign_back(const HPStage *S, uint64_t *const *X1, const uint64_t *y1, uint64_t **X0, uint64_t *y0) {
    const PSSign *G = S->sg; const size_t Wm = (size_t)mpx_W;
    { int q = 0; for (int c = 0; c < S->m0; c++) { if (G->keep[c]) { memcpy(y0 + (size_t)c * Wm, y1 + (size_t)q * Wm, Wm * sizeof(uint64_t)); q++; } } }
    for (int k = 0; k < S->nb0; k++) {
        const int n = abs(S->bs0[k]);
        if (S->bs0[k] < 0) { for (int i = 0; i < n; i++) { const int q = G->cls[G->off[k] + i]; memcpy(X0[k] + (size_t)i * Wm, X1[G->qblk[q]] + (size_t)(G->cpos[G->off[k] + i] + G->qpos0[q]) * Wm, Wm * sizeof(uint64_t)); } continue; }
        for (int i = 0; i < n; i++) {
            const int qi = G->cls[G->off[k] + i], pi = G->cpos[G->off[k] + i];
            if (G->csz[qi] == 1) { memcpy(X0[k] + ((size_t)i * n + i) * Wm, X1[G->qblk[qi]] + (size_t)G->qpos0[qi] * Wm, Wm * sizeof(uint64_t)); continue; }
            for (int j = 0; j < n; j++) {
                if (G->cls[G->off[k] + j] != qi) continue;
                memcpy(X0[k] + ((size_t)i * n + j) * Wm, X1[G->qblk[qi]] + ((size_t)pi * G->csz[qi] + G->cpos[G->off[k] + j]) * Wm, Wm * sizeof(uint64_t));
            }
        }
    }
}

/* -- 2. the constraints of an orbit summed */
static int hpred_orbit(HPStage *S, const HPJob *J, HPJob *Jn, const Params *par) {
    if (!par->symfile) return 0;
    Problem P; hpred_shadow(J, &P);
    /* without the adapted basis (stage 3 computes it in the master precision), and applied
     * whenever the group is not trivial: fewer rows at no cost */
    PSSym *G = sym_reduce(&P, par->symfile, par->verbose, par->symtime, par->symnodes, 0, 1.0, 1 << 30, par->chordal_density, par->symsigned);
    if (!G) { problem_free(&P); return 0; }
    const int mr = P.m;
    problem_free(&P);
    if (G->bd || mr > J->m) { sym_free(G); return 0; }
    const size_t Wm = (size_t)J->Lm + 2; const int nb = J->nblk;
    /* SDP blocks that the group exchanges: X_k = P X_rep P' on invariant X, and the aggregated
     * data of block k are those of its representative permuted: only the representative is
     * kept, its data times the number of blocks in its orbit */
    int *brep = (int *)malloc(sizeof(int) * ((size_t)nb + 1)), *bnew = (int *)malloc(sizeof(int) * ((size_t)nb + 1)), *bcnt = (int *)calloc((size_t)nb + 1, sizeof(int));
    int **bu = (int **)calloc((size_t)nb + 1, sizeof(int *)); signed char **bsg = (signed char **)calloc((size_t)nb + 1, sizeof(signed char *));
    for (int k = 0; k < nb; k++) brep[k] = -1;
    int okb = 1, nex = 0;
    if (getenv("BRISK_HPNOEXCH")) okb = 0;
    for (int k0 = 0; k0 < nb && okb; k0++) {
        if (J->bs[k0] < 0 || brep[k0] >= 0) continue;
        const int n = J->bs[k0];
        brep[k0] = k0;
        int *queue = (int *)malloc(sizeof(int) * ((size_t)nb + 1)); int qh = 0, qt = 0; queue[qt++] = k0;
        while (qh < qt && okb) {
            const int a = queue[qh++];
            for (int s2 = 0; s2 < G->ngen && okb; s2++) {
                const int *g = G->gen[s2]; const signed char *gs = G->gsg ? G->gsg[s2] : NULL;
                const int bb = G->gblk[g[G->off[a]]];
                if (J->bs[bb] != n) { okb = 0; break; }
                for (int i = 0; i < n; i++) if (G->gblk[g[G->off[a] + i]] != bb) { okb = 0; break; }
                if (!okb || brep[bb] >= 0) { if (okb && brep[bb] != k0) okb = 0; continue; }
                brep[bb] = k0;
                bu[bb] = (int *)malloc(sizeof(int) * (size_t)n); bsg[bb] = (signed char *)malloc((size_t)n);
                for (int i = 0; i < n; i++) {
                    const int ia = a == k0 ? i : bu[a][i];
                    bu[bb][i] = g[G->off[a] + ia] - G->off[bb];
                    bsg[bb][i] = (signed char)((a == k0 ? 1 : bsg[a][i]) * (gs ? gs[G->off[a] + ia] : 1));
                }
                queue[qt++] = bb; nex++;
            }
        }
        free(queue);
    }
    if (!okb) { for (int k = 0; k < nb; k++) { free(bu[k]); free(bsg[k]); bu[k] = NULL; bsg[k] = NULL; brep[k] = k; } nex = 0; }
    for (int k = 0; k < nb; k++) { if (J->bs[k] < 0) brep[k] = k; bcnt[brep[k]]++; }
    int nbn = 0;
    for (int k = 0; k < nb; k++) { if (brep[k] == k) bnew[k] = nbn++; else bnew[k] = -1; }
    /* the aggregated problem on all blocks first: its invariance under the generators is then
     * checked entry by entry in the master precision (the search ran on the data rounded to
     * doubles: 0.3 and 0.30000000000000000001 are the same number there) */
    int valid = 1;
    uint64_t *b = hpred_bcopy(J, mr);
    {
        char *seen = (char *)calloc((size_t)mr + 1, 1); uint64_t t[MPX_MAXL + 2];
        for (int c = 0; c < J->m && valid; c++) {
            if (G->cls[c] < 0) continue;
            uint64_t *d = b + (size_t)G->cls[c] * Wm;
            if (G->csg && G->csg[c] < 0) mpx_neg(t, J->b + (size_t)c * Wm); else mpx_copy(t, J->b + (size_t)c * Wm);
            if (!seen[G->cls[c]]) { seen[G->cls[c]] = 1; mpx_copy(d, t); }
            else if (mpx_cmp(d, t) != 0) valid = 0;                       /* the right-hand sides of an orbit differ */
        }
        free(seen);
        /* (the right-hand side of the sum over the orbit: its size times the common value) */
        int *cnt = (int *)calloc((size_t)mr + 1, sizeof(int));
        for (int c = 0; c < J->m; c++) if (G->cls[c] >= 0) cnt[G->cls[c]]++;
        for (int q = 0; q < mr; q++) if (cnt[q] > 1) { mpx_muld(t, b + (size_t)q * Wm, (double)cnt[q]); mpx_copy(b + (size_t)q * Wm, t); }
        free(cnt);
    }
    HPJob Ja; memset(&Ja, 0, sizeof(Ja));
    if (valid) {
        HTl T; htl_init(&T, Wm, J->nnzA + J->nnzC + 1);
        for (int k = 0; k < nb; k++) {
            const int lp = J->bs[k] < 0;
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) memcpy(htl_add(&T, -1, k, J->ci[e], lp ? J->ci[e] : J->cj[e]), J->Cv + e * Wm, Wm * sizeof(uint64_t));
            for (int tt = 0; tt < J->ncon[k]; tt++) {
                const int c0 = J->con[k][tt], c = G->cls[c0];
                if (c < 0) continue;
                const int neg = G->csg && G->csg[c0] < 0;
                for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                    uint64_t *v = htl_add(&T, c, k, J->ei[e], lp ? J->ei[e] : J->ej[e]);
                    if (neg) mpx_neg(v, J->Av + e * Wm); else memcpy(v, J->Av + e * Wm, Wm * sizeof(uint64_t));
                }
            }
        }
        int *bsa = (int *)malloc(sizeof(int) * ((size_t)nb + 1)); memcpy(bsa, J->bs, sizeof(int) * (size_t)nb);
        if (htl_build(&T, &Ja, J->Lm, mr, nb, bsa, b, 1)) valid = 0;
        b = NULL;                                                         /* (owned by Ja) */
        htl_free(&T);
    }
    for (int s2 = 0; s2 < G->ngen && valid; s2++) {
        const int *g = G->gen[s2]; const signed char *gs = G->gsg ? G->gsg[s2] : NULL;
        uint64_t t[MPX_MAXL + 2];
        for (int k = 0; k < nb && valid; k++) {
            const int lp = Ja.bs[k] < 0;
            for (int tt = -1; tt < Ja.ncon[k] && valid; tt++) {
                const size_t e0 = tt < 0 ? Ja.cptr[k] : Ja.ptr[k][tt], e1 = tt < 0 ? Ja.cptr[k + 1] : Ja.ptr[k][tt + 1];
                const uint64_t *V = tt < 0 ? Ja.Cv : Ja.Av; const int *I = tt < 0 ? Ja.ci : Ja.ei, *Jx = tt < 0 ? Ja.cj : Ja.ej;
                for (size_t e = e0; e < e1; e++) {
                    const int i = I[e], j = lp ? i : Jx[e];
                    const int gi = g[G->off[k] + i], gj = g[G->off[k] + j], kb = G->gblk[gi];
                    if (G->gblk[gj] != kb || (Ja.bs[kb] < 0) != lp) { valid = 0; break; }
                    int i2 = gi - G->off[kb], j2 = gj - G->off[kb]; if (i2 > j2) { const int x = i2; i2 = j2; j2 = x; }
                    /* the entry (i2, j2) of the same matrix in block kb */
                    size_t f0, f1;
                    if (tt < 0) { f0 = Ja.cptr[kb]; f1 = Ja.cptr[kb + 1]; }
                    else {
                        const int c = Ja.con[k][tt]; int lo = 0, hi = Ja.ncon[kb] - 1, at = -1;
                        while (lo <= hi) { const int mid = (lo + hi) / 2; if (Ja.con[kb][mid] == c) { at = mid; break; } if (Ja.con[kb][mid] < c) lo = mid + 1; else hi = mid - 1; }
                        if (at < 0) { valid = 0; break; }
                        f0 = Ja.ptr[kb][at]; f1 = Ja.ptr[kb][at + 1];
                    }
                    size_t lo = f0, hi = f1, at = (size_t)-1;
                    while (lo < hi) { const size_t mid = (lo + hi) / 2; const int a = I[mid], b2 = Ja.bs[kb] < 0 ? I[mid] : Jx[mid]; if (a == i2 && b2 == j2) { at = mid; break; } if (a < i2 || (a == i2 && b2 < j2)) lo = mid + 1; else hi = mid; }
                    if (at == (size_t)-1) { valid = 0; break; }
                    if (gs && !lp && gs[G->off[k] + i] * gs[G->off[k] + j] < 0) mpx_neg(t, V + e * Wm); else mpx_copy(t, V + e * Wm);
                    if (mpx_cmp(t, V + at * Wm) != 0) { valid = 0; break; }
                }
            }
        }
    }
    if (!valid) {
        if (par->verbose >= 0) printf("presolve: the symmetry found on the data rounded to doubles does not hold in the working precision: not used\n");
        if (Ja.bs) hp_free(&Ja);
        free(b); sym_free(G); for (int k = 0; k < nb; k++) { free(bu[k]); free(bsg[k]); } free(bu); free(bsg); free(brep); free(bnew); free(bcnt);
        return 0;
    }
    if (nex == 0) { *Jn = Ja; free(bcnt); }
    else {
        /* one block per orbit of blocks, its data times the size of the orbit */
        int *bs = (int *)malloc(sizeof(int) * ((size_t)nb + 1));
        for (int k = 0; k < nb; k++) if (bnew[k] >= 0) bs[bnew[k]] = Ja.bs[k];
        uint64_t *b2 = hpred_bcopy(J, mr); memcpy(b2, Ja.b, sizeof(uint64_t) * (size_t)mr * Wm);
        HTl T; htl_init(&T, Wm, Ja.nnzA + Ja.nnzC + 1);
        for (int k = 0; k < nb; k++) {
            if (bnew[k] < 0) continue;
            const int lp = Ja.bs[k] < 0; const double mult = (double)bcnt[k];
            for (size_t e = Ja.cptr[k]; e < Ja.cptr[k + 1]; e++) mpx_muld(htl_add(&T, -1, bnew[k], Ja.ci[e], lp ? Ja.ci[e] : Ja.cj[e]), Ja.Cv + e * Wm, mult);
            for (int tt = 0; tt < Ja.ncon[k]; tt++) for (size_t e = Ja.ptr[k][tt]; e < Ja.ptr[k][tt + 1]; e++) mpx_muld(htl_add(&T, Ja.con[k][tt], bnew[k], Ja.ei[e], lp ? Ja.ei[e] : Ja.ej[e]), Ja.Av + e * Wm, mult);
        }
        free(bcnt);
        const int rcb = htl_build(&T, Jn, J->Lm, mr, nbn, bs, b2, 0);
        htl_free(&T); hp_free(&Ja);
        if (rcb) { hp_free(Jn); sym_free(G); for (int k = 0; k < nb; k++) { free(bu[k]); free(bsg[k]); } free(bu); free(bsg); free(brep); free(bnew); return 0; }
    }
    memset(S, 0, sizeof(*S)); S->kind = 2; S->sy = G; hpst_layout(S, J); S->m1 = mr; S->nb1 = nbn;
    S->brep = brep; S->bu = bu; S->bsg = bsg; S->bnew = bnew;
    if (nex > 0 && par->verbose >= 0) printf("presolve: %d SDP block(s) are images of others under the group: one block of each orbit kept (%d -> %d blocks)\n", nex, nb, nbn);
    return 1;
}
/* union-find with parity (the signs of signed permutations) */
typedef struct { int *p; unsigned char *par; } HPuf;
static int hpuf_find(HPuf *U, int x, int *parity) {
    int r = x, pr = 0;
    while (U->p[r] != r) { pr ^= U->par[r]; r = U->p[r]; }
    int cur = x, cp = pr;
    while (U->p[cur] != cur) { const int nx = U->p[cur], np = cp ^ U->par[cur]; U->p[cur] = r; U->par[cur] = (unsigned char)cp; cur = nx; cp = np; }
    *parity = pr; return r;
}
static int hpuf_union(HPuf *U, int a, int b, int d) {
    int pa, pb; const int ra = hpuf_find(U, a, &pa), rb = hpuf_find(U, b, &pb);
    if (ra == rb) return (pa ^ pb) == d;
    U->p[ra] = rb; U->par[ra] = (unsigned char)(pa ^ pb ^ d);
    return 1;
}
/* y expanded over the orbits; X averaged over the group: the mean over the orbit of each entry
 * (the orbits of the index pairs closed under the generators; an orbit that meets itself with
 * the sign -1 is zero on invariant X) */
static void hpred_orbit_back(const HPStage *S, uint64_t *const *X1in, const uint64_t *y1, uint64_t **X0, uint64_t *y0) {
    uint64_t *const *X1 = X1in;
    const PSSym *G = S->sy; const size_t Wm = (size_t)mpx_W;
    const int nb = S->nb0; const int *bs = S->bs0;
    for (int c = 0; c < S->m0; c++) {
        if (G->cls[c] < 0) continue;
        if (G->csg && G->csg[c] < 0) mpx_neg(y0 + (size_t)c * Wm, y1 + (size_t)G->cls[c] * Wm); else memcpy(y0 + (size_t)c * Wm, y1 + (size_t)G->cls[c] * Wm, Wm * sizeof(uint64_t));
    }
    /* the blocks as before the stage: the kept ones, and the images X_k = P X_rep P' */
    uint64_t **Xe = (uint64_t **)calloc((size_t)nb + 1, sizeof(uint64_t *));
    for (int k = 0; k < nb; k++) {
        if (S->bnew[k] >= 0) { Xe[k] = X1[S->bnew[k]]; continue; }
        const int n = bs[k]; const uint64_t *Xr = X1[S->bnew[S->brep[k]]]; const int *uu = S->bu[k]; const signed char *su = S->bsg[k];
        Xe[k] = (uint64_t *)calloc(((size_t)n * n + 1) * Wm, sizeof(uint64_t));
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            uint64_t *d = Xe[k] + ((size_t)uu[i] * n + uu[j]) * Wm;
            if (su[i] * su[j] < 0) mpx_neg(d, Xr + ((size_t)i * n + j) * Wm); else memcpy(d, Xr + ((size_t)i * n + j) * Wm, Wm * sizeof(uint64_t));
        }
    }
    X1 = Xe;
    size_t *poff = (size_t *)malloc(sizeof(size_t) * ((size_t)nb + 1)); poff[0] = 0;
    for (int k = 0; k < nb; k++) poff[k + 1] = poff[k] + (bs[k] < 0 ? (size_t)abs(bs[k]) : (size_t)abs(bs[k]) * abs(bs[k]));
    const size_t np = poff[nb];
    HPuf U; U.p = (int *)malloc(sizeof(int) * (np + 1)); U.par = (unsigned char *)calloc(np + 1, 1);
    char *pz = (char *)calloc(np + 1, 1);
    for (size_t p = 0; p < np; p++) U.p[p] = (int)p;
    for (int s = 0; s < G->ngen; s++) {
        const int *g = G->gen[s]; const signed char *gs = G->gsg ? G->gsg[s] : NULL;
        for (int k = 0; k < nb; k++) {
            const int n = abs(bs[k]);
            if (bs[k] < 0) {
                for (int i = 0; i < n; i++) { const int gi = g[G->off[k] + i], ki = G->gblk[gi]; if (bs[ki] >= 0) continue; (void)hpuf_union(&U, (int)(poff[k] + i), (int)(poff[ki] + (size_t)(gi - G->off[ki])), 0); }
                continue;
            }
            for (int j = 0; j < n; j++) {
                const int gj = g[G->off[k] + j], kj = G->gblk[gj], jj = gj - G->off[kj];
                for (int i = 0; i <= j; i++) {
                    const int gi = g[G->off[k] + i], ki = G->gblk[gi], ii = gi - G->off[ki];
                    if (ki != kj) continue;
                    const int a = (int)(poff[k] + (size_t)i + (size_t)j * n);
                    const int lo = ii < jj ? ii : jj, hi = ii < jj ? jj : ii;
                    const int b2 = (int)(poff[ki] + (size_t)lo + (size_t)hi * abs(bs[ki]));
                    if (!hpuf_union(&U, a, b2, gs ? ((gs[G->off[k] + i] < 0) ^ (gs[G->off[k] + j] < 0)) : 0)) pz[a] = 1;
                }
            }
        }
    }
    for (size_t p = 0; p < np; p++) if (pz[p] == 1) { int pr; pz[hpuf_find(&U, (int)p, &pr)] = 2; }
    uint64_t *sum = (uint64_t *)calloc((np + 1) * Wm, sizeof(uint64_t)); int *cnt = (int *)calloc(np + 1, sizeof(int));
    for (int k = 0; k < nb; k++) {
        const int n = abs(bs[k]);
        for (int j = 0; j < n; j++) for (int i = (bs[k] < 0 ? j : 0); i <= j; i++) {
            const size_t pp = bs[k] < 0 ? poff[k] + j : poff[k] + (size_t)i + (size_t)j * n;
            const uint64_t *x = bs[k] < 0 ? X1[k] + (size_t)j * Wm : X1[k] + ((size_t)i * n + j) * Wm;
            int pr; const int r = hpuf_find(&U, (int)pp, &pr);
            if (pr) mpx_sub(sum + (size_t)r * Wm, sum + (size_t)r * Wm, x); else mpx_add(sum + (size_t)r * Wm, sum + (size_t)r * Wm, x);
            cnt[r]++;
        }
    }
    uint64_t t[MPX_MAXL + 2], cn[MPX_MAXL + 2];
    for (size_t r = 0; r < np; r++) if (cnt[r] > 1 && U.p[r] == (int)r) { mpx_setd(cn, (double)cnt[r]); mpx_div(t, sum + r * Wm, cn); mpx_copy(sum + r * Wm, t); }
    for (int k = 0; k < nb; k++) {
        const int n = abs(bs[k]);
        for (int j = 0; j < n; j++) for (int i = (bs[k] < 0 ? j : 0); i <= j; i++) {
            const size_t pp = bs[k] < 0 ? poff[k] + j : poff[k] + (size_t)i + (size_t)j * n;
            int pr; const int r = hpuf_find(&U, (int)pp, &pr);
            if (pz[r] == 2) mpx_setd(t, 0.0); else if (pr) mpx_neg(t, sum + (size_t)r * Wm); else mpx_copy(t, sum + (size_t)r * Wm);
            if (bs[k] < 0) mpx_copy(X0[k] + (size_t)j * Wm, t);
            else { mpx_copy(X0[k] + ((size_t)i * n + j) * Wm, t); mpx_copy(X0[k] + ((size_t)j * n + i) * Wm, t); }
        }
    }
    free(poff); free(U.p); free(U.par); free(pz); free(sum); free(cnt);
    for (int k = 0; k < nb; k++) if (S->bnew[k] < 0) free(Xe[k]);
    free(Xe);
}

/* -- 3. the *-algebra of a block in the master precision */
static uint64_t hpalg_rng(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 33; }
/* S (n x n, row-major, full) = a random combination of C and the A_p of block k, each scaled
 * by a power of two to its size (exact) */
static void hpalg_comb(const HPJob *J, int k, uint64_t seed, uint64_t *S) {
    const size_t Wm = (size_t)J->Lm + 2; const int n = J->bs[k];
    memset(S, 0, sizeof(uint64_t) * (size_t)n * n * Wm);
    uint64_t t[MPX_MAXL + 2];
    for (int tt = -1; tt < J->ncon[k]; tt++) {
        const size_t e0 = tt < 0 ? J->cptr[k] : J->ptr[k][tt], e1 = tt < 0 ? J->cptr[k + 1] : J->ptr[k][tt + 1];
        const uint64_t *V = tt < 0 ? J->Cv : J->Av; const int *I = tt < 0 ? J->ci : J->ei, *Jx = tt < 0 ? J->cj : J->ej;
        double mx = 0; for (size_t e = e0; e < e1; e++) { const double v = fabs(mpx_getd(V + e * Wm)); if (v > mx) mx = v; }
        uint64_t sd = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tt < 0 ? 0x7fffffff : J->con[k][tt] + 1));
        (void)hpalg_rng(&sd);
        int ex = 0; if (mx > 0) (void)frexp(mx, &ex);
        const double cf = ldexp((double)(1024 + hpalg_rng(&sd) % 1047552) * (hpalg_rng(&sd) & 1 ? 1.0 : -1.0), -ex - 20);
        for (size_t e = e0; e < e1; e++) {
            const int i = I[e], j = Jx[e];
            mpx_muld(t, V + e * Wm, cf);
            mpx_add(S + ((size_t)i * n + j) * Wm, S + ((size_t)i * n + j) * Wm, t);
            if (i != j) mpx_add(S + ((size_t)j * n + i) * Wm, S + ((size_t)j * n + i) * Wm, t);
        }
    }
}
static int hpalg_uf(int *p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
static int g_hpred_bits = 106;       /* the target precision of the solve (set by the driver) */
/* y = S x for the combination S of hpalg_comb with this seed (from the entries: O(nnz)) */
static void hpalg_apply(const HPJob *J, int k, uint64_t seed, const uint64_t *x, uint64_t *y) {
    const size_t Wm = (size_t)J->Lm + 2; const int n = J->bs[k];
    memset(y, 0, sizeof(uint64_t) * (size_t)n * Wm);
    uint64_t t[MPX_MAXL + 2];
    for (int tt = -1; tt < J->ncon[k]; tt++) {
        const size_t e0 = tt < 0 ? J->cptr[k] : J->ptr[k][tt], e1 = tt < 0 ? J->cptr[k + 1] : J->ptr[k][tt + 1];
        const uint64_t *V = tt < 0 ? J->Cv : J->Av; const int *I = tt < 0 ? J->ci : J->ei, *Jx = tt < 0 ? J->cj : J->ej;
        double mx = 0; for (size_t e = e0; e < e1; e++) { const double v = fabs(mpx_getd(V + e * Wm)); if (v > mx) mx = v; }
        uint64_t sd = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tt < 0 ? 0x7fffffff : J->con[k][tt] + 1));
        (void)hpalg_rng(&sd);
        int ex = 0; if (mx > 0) (void)frexp(mx, &ex);
        const double cf = ldexp((double)(1024 + hpalg_rng(&sd) % 1047552) * (hpalg_rng(&sd) & 1 ? 1.0 : -1.0), -ex - 20);
        for (size_t e = e0; e < e1; e++) {
            const int i = I[e], j = Jx[e];
            mpx_muld(t, V + e * Wm, cf);
            mpx_addmul(y + (size_t)i * Wm, t, x + (size_t)j * Wm);
            if (i != j) mpx_addmul(y + (size_t)j * Wm, t, x + (size_t)i * Wm);
        }
    }
}
/* The components of block k: returns their number (0: no structure, or the analysis failed its
 * verification) and the components in *out.
 *   1. structure in double-double: the eigenvectors of S1 from LAPACK refined (eigref), the
 *      clusters of equal eigenvalues, their couplings through S2, the components and, for
 *      those of multiplicity >= 2, a spanning tree of the strongest couplings;
 *   2. the basis in the precision the target needs (quad-double for a double-double solve,
 *      the master precision otherwise): refinement and alignment along the tree (algbuild);
 *   3. the verification in the master precision on a third combination, by products with
 *      random vectors: S3 must map each copy of a component into itself, the same way. */
static int hpalg_block(const HPJob *J, int k, HACmp **out, int verbose) {
    const size_t W = (size_t)mpx_W; const int n = J->bs[k]; const size_t N = (size_t)n * n;
    const double t0 = wtime();
    uint64_t *S1 = (uint64_t *)malloc(sizeof(uint64_t) * (N + 1) * W), *S2 = (uint64_t *)malloc(sizeof(uint64_t) * (N + 1) * W), *Q = (uint64_t *)calloc((N + 1) * W, sizeof(uint64_t));
    dd_t *A1 = (dd_t *)malloc(sizeof(dd_t) * (N + 1)), *Vd = (dd_t *)malloc(sizeof(dd_t) * (N + 1)), *Sd = (dd_t *)malloc(sizeof(dd_t) * (N + 1)), *Rd = (dd_t *)malloc(sizeof(dd_t) * (N + 1)), *Td = (dd_t *)malloc(sizeof(dd_t) * (N + 1)), *ld = (dd_t *)malloc(sizeof(dd_t) * ((size_t)n + 1));
    double *Ad = (double *)malloc(sizeof(double) * (N + 1)), *wd = (double *)malloc(sizeof(double) * ((size_t)n + 1));
    int *cl = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *cs = (int *)malloc(sizeof(int) * ((size_t)n + 2)), *uf = NULL, *comp = NULL, *ord = NULL, *par = NULL, *order = NULL, *perm = NULL;
    double *cn = NULL; char *fail = NULL, *aln = NULL;
    HACmp *cmp = NULL; int nc = 0, ok = 0, ncl = 0, nfail = 0;
    double emax = 0, worst = 0, tstruct = 0, tbasis = 0;
    if (!S1 || !S2 || !Q || !A1 || !Vd || !Sd || !Rd || !Td || !Ad) goto done;
    hpalg_comb(J, k, 0x1234567ull + (uint64_t)k * 7919u, S1);
    hpalg_comb(J, k, 0x7654321ull + (uint64_t)k * 104729u, S2);
    /* ---- 1. the structure in double-double */
    for (size_t q = 0; q < N; q++) { A1[q] = mpx_get_dd(S1 + q * W); Ad[q] = A1[q].h; }
    { int lw = 3 * n + 64, info = 0; double *work = (double *)malloc(sizeof(double) * (size_t)lw);
      BL(dsyev_)("V", "U", &n, Ad, &n, wd, work, &lw, &info); free(work);
      if (info != 0) goto done; }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { Vd[(size_t)i * n + j].h = Ad[j + (size_t)i * n]; Vd[(size_t)i * n + j].l = 0.0; }   /* row i: eigenvector i (increasing) */
    const double scale = fmax(fabs(wd[0]), fabs(wd[n - 1]));
    { int c = 0; for (int i = 0; i < n; i++) { if (i > 0 && wd[i] - wd[i - 1] > 1e-9 * scale) c++; cl[i] = c; } }
    for (int it = 0; it < 4; it++) { const double e = eigref_dd(n, A1, Vd, cl, ld, Sd, Rd, Td); if (e < 1e-26) break; }
    (void)eigref_dd(n, A1, Vd, cl, ld, Sd, Rd, Td);            /* (the eigenvalues of the refined vectors) */
    /* rows by increasing eigenvalue within the provisional clusters, then the clusters to 1e-20 */
    perm = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = 1; i < n; i++) { const int pi = perm[i]; int j = i; while (j > 0 && cl[perm[j - 1]] == cl[pi] && (ld[perm[j - 1]].h > ld[pi].h || (ld[perm[j - 1]].h == ld[pi].h && ld[perm[j - 1]].l > ld[pi].l))) { perm[j] = perm[j - 1]; j--; } perm[j] = pi; }
    for (int i = 0; i < n; i++) memcpy(Td + (size_t)i * n, Vd + (size_t)perm[i] * n, sizeof(dd_t) * (size_t)n);
    memcpy(Vd, Td, sizeof(dd_t) * N);
    cs[0] = 0;
    for (int i = 1; i < n; i++) { const double gap = (ld[perm[i]].h - ld[perm[i - 1]].h) + (ld[perm[i]].l - ld[perm[i - 1]].l); if (fabs(gap) > 1e-20 * scale) cs[++ncl] = i; }
    cs[++ncl] = n;
    /* the couplings T2 = V S2 V' */
    for (size_t q = 0; q < N; q++) A1[q] = mpx_get_dd(S2 + q * W);
    gemm_rows_dd(n, n, n, Vd, A1, Td); gemm_abt_dd(n, n, n, Td, Vd, Sd);
    double s2n = 0; for (size_t q = 0; q < N; q++) if (fabs(Sd[q].h) > s2n) s2n = fabs(Sd[q].h);
    const double thr2 = 1e-18 * s2n;
    cn = (double *)calloc((size_t)ncl * ncl + 1, sizeof(double)); uf = (int *)malloc(sizeof(int) * ((size_t)ncl + 1));
    for (int a = 0; a < ncl; a++) uf[a] = a;
    for (int a = 0; a < ncl; a++) for (int b = a + 1; b < ncl; b++) {
        double s = 0;
        for (int i = cs[a]; i < cs[a + 1]; i++) for (int j = cs[b]; j < cs[b + 1]; j++) { const double v = Sd[(size_t)i * n + j].h; s += v * v; }
        s = sqrt(s); cn[(size_t)a * ncl + b] = cn[(size_t)b * ncl + a] = s;
        if (s > thr2) { const int ra = hpalg_uf(uf, a), rb = hpalg_uf(uf, b); if (ra != rb) uf[ra] = rb; }
    }
    comp = (int *)malloc(sizeof(int) * ((size_t)ncl + 1)); ord = (int *)malloc(sizeof(int) * ((size_t)ncl + 1));
    par = (int *)malloc(sizeof(int) * ((size_t)ncl + 1)); order = (int *)malloc(sizeof(int) * ((size_t)ncl + 1));
    fail = (char *)calloc((size_t)ncl + 1, 1);
    for (int a = 0; a < ncl; a++) { comp[a] = -1; par[a] = -1; }
    for (int a = 0; a < ncl; a++) { const int r = hpalg_uf(uf, a); if (comp[r] < 0) comp[r] = nc++; comp[a] = comp[r]; }
    aln = (char *)calloc((size_t)nc + 1, 1);
    int no = 0;
    for (int c = 0; c < nc; c++) {
        int nk = 0, mm = -1, same = 1;
        for (int a = 0; a < ncl; a++) if (comp[a] == c) { ord[nk++] = a; const int sz = cs[a + 1] - cs[a]; if (mm < 0) mm = sz; else if (sz != mm) same = 0; }
        aln[c] = same && mm >= 2;
        if (!aln[c] || nk == 1) { for (int x = 0; x < nk; x++) order[no++] = ord[x]; continue; }
        /* Prim from the first cluster by the strongest coupling */
        char *in = (char *)calloc((size_t)nk + 1, 1);
        in[0] = 1; order[no++] = ord[0];
        for (int step = 1; step < nk; step++) {
            int pa = -1, pb = -1; double best = -1;
            for (int x = 0; x < nk; x++) if (in[x]) for (int y = 0; y < nk; y++) if (!in[y] && cn[(size_t)ord[x] * ncl + ord[y]] > best) { best = cn[(size_t)ord[x] * ncl + ord[y]]; pa = x; pb = y; }
            par[ord[pb]] = ord[pa]; in[pb] = 1; order[no++] = ord[pb];
        }
        free(in);
    }
    tstruct = wtime() - t0;
    /* ---- 2. the basis: error 2^-100 doubled per step, to 1.25 times the target and a margin */
    {
        int iters = 1; while (100.0 * ldexp(1.0, iters) < 1.25 * g_hpred_bits + 16.0) iters++;
        const double orthtol = ldexp(1.0, -(int)(1.1 * g_hpred_bits));
        if (g_hpred_bits <= 106) algbuild_qd(J, n, S1, S2, Vd, NULL, 0, ncl, cs, par, order, iters, Q, fail, orthtol, &emax);
        else {
            /* the first step in quad-double (four times faster), the others in the master precision */
            uint64_t *V1 = (uint64_t *)malloc(sizeof(uint64_t) * (N + 1) * W);
            algbuild_qd(J, n, S1, S2, Vd, NULL, 1, ncl, cs, par, order, 1, V1, fail, orthtol, &emax);
            algbuild_mp(J, n, S1, S2, Vd, V1, 0, ncl, cs, par, order, iters - 1, Q, fail, orthtol, &emax);
            free(V1);
        }
    }
    tbasis = wtime() - t0 - tstruct;
    /* the components: rows of Qc in the order (copy, index) */
    cmp = (HACmp *)calloc((size_t)nc + 1, sizeof(HACmp));
    {
        uint64_t *Qc = (uint64_t *)malloc(sizeof(uint64_t) * (N + 1) * W); int qrow = 0;
        for (int c = 0; c < nc; c++) {
            int nk = 0, mm = 0, bad = 0;
            for (int a = 0; a < ncl; a++) if (comp[a] == c) { ord[nk++] = a; mm = cs[a + 1] - cs[a]; if (fail[a]) bad = 1; }
            HACmp *cc = &cmp[c];
            cc->k = k; cc->rblk = -1; cc->rpos = -1; cc->W = Qc + (size_t)qrow * n * W;
            if (aln[c] && !bad) {
                cc->d = nk; cc->mult = mm;
                for (int r = 0; r < mm; r++) for (int x = 0; x < nk; x++) { memcpy(Qc + (size_t)qrow * n * W, Q + (size_t)(cs[ord[x]] + r) * n * W, sizeof(uint64_t) * (size_t)n * W); qrow++; }
            } else {
                if (aln[c]) nfail++;
                cc->d = 0; cc->mult = 1;
                for (int x = 0; x < nk; x++) for (int i = cs[ord[x]]; i < cs[ord[x] + 1]; i++) { memcpy(Qc + (size_t)qrow * n * W, Q + (size_t)i * n * W, sizeof(uint64_t) * (size_t)n * W); qrow++; cc->d++; }
            }
        }
        free(Q); Q = Qc;
    }
    if (nc == 1 && cmp[0].mult == 1) { nc = 0; goto done; }       /* no structure */
    /* ---- 3. the verification on a third combination, by random vectors */
    {
        const uint64_t seed3 = 0xabcdef1ull + (uint64_t)k * 1299709u; uint64_t rs = 0x5deece66dull + (uint64_t)k;
        uint64_t *x = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)n + 1) * W), *y = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)n + 1) * W), *z = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)n + 1) * W);
        uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2];
        /* orthonormal rows: Q'(Q z) = z */
        for (int i = 0; i < n; i++) mpx_setd(z + (size_t)i * W, 0.5 + (double)(hpalg_rng(&rs) % 1000003) / 1000003.0);
        memset(y, 0, sizeof(uint64_t) * (size_t)n * W);
        for (int r = 0; r < n; r++) { mpx_dot((size_t)n, Q + (size_t)r * n * W, z, t); axpy_mp((size_t)n, t, Q + (size_t)r * n * W, y); }
        for (int i = 0; i < n; i++) { mpx_sub(t, y + (size_t)i * W, z + (size_t)i * W); const double v = fabs(mpx_getd(t)); if (v > worst) worst = v; }
        for (int c = 0; c < nc; c++) {
            const HACmp *cc = &cmp[c]; const int d = cc->d, mm = cc->mult;
            double *xi = (double *)malloc(sizeof(double) * ((size_t)d + 1));
            uint64_t *beta = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)d + 1) * W);
            for (int i = 0; i < d; i++) xi[i] = 0.5 + (double)(hpalg_rng(&rs) % 1000003) / 1000003.0;
            for (int pass = 0; pass < (mm > 1 ? 2 : 1); pass++) {
                /* pass 0: the first copy (this defines beta = B xi); pass 1: a random combination of the copies */
                memset(x, 0, sizeof(uint64_t) * (size_t)n * W); memset(z, 0, sizeof(uint64_t) * (size_t)n * W);
                for (int r = 0; r < (pass ? mm : 1); r++) {
                    const double al = pass ? 0.5 + (double)(hpalg_rng(&rs) % 1000003) / 1000003.0 : 1.0;
                    for (int i = 0; i < d; i++) { mpx_setd(u, al); mpx_setd(t, xi[i]); mpx_mul(t, t, u); axpy_mp((size_t)n, t, cc->W + (size_t)(r * d + i) * n * W, x); }
                    if (pass) for (int i = 0; i < d; i++) { mpx_setd(u, al); mpx_mul(t, u, beta + (size_t)i * W); axpy_mp((size_t)n, t, cc->W + (size_t)(r * d + i) * n * W, z); }
                }
                hpalg_apply(J, k, seed3, x, y);
                if (!pass) for (int i = 0; i < d; i++) { mpx_dot((size_t)n, cc->W + (size_t)i * n * W, y, beta + (size_t)i * W); axpy_mp((size_t)n, beta + (size_t)i * W, cc->W + (size_t)i * n * W, z); }
                double yn = 0, rn = 0;
                for (int i = 0; i < n; i++) { const double v = fabs(mpx_getd(y + (size_t)i * W)); if (v > yn) yn = v; mpx_sub(t, y + (size_t)i * W, z + (size_t)i * W); const double w = fabs(mpx_getd(t)); if (w > rn) rn = w; }
                if (yn > 0 && rn / yn > worst) worst = rn / yn;
            }
            free(xi); free(beta);
        }
        free(x); free(y); free(z);
        ok = worst <= ldexp(1.0, -(int)(1.2 * g_hpred_bits));
    }
    if (verbose > 1 || getenv("BRISK_HPDBG")) printf("   [algebra, block %d (n = %d): %d cluster(s), %d component(s), %d not aligned; last correction %.1e, verification %.1e: %s; structure %.2fs, basis %.2fs, check %.2fs]\n",
                                                     k + 1, n, ncl, nc, nfail, emax, worst, ok ? "passed" : "failed", tstruct, tbasis, wtime() - t0 - tstruct - tbasis);
done:
    free(S1); free(S2); free(A1); free(Vd); free(Sd); free(Rd); free(Td); free(ld); free(Ad); free(wd); free(cl); free(cs); free(uf); free(comp); free(ord); free(par); free(order); free(perm); free(cn); free(fail); free(aln);
    if (!ok || nc == 0) { free(Q); free(cmp); *out = NULL; return 0; }
    *out = cmp;                                                  /* (cmp[0].W = Q owns the rows) */
    return nc;
}
static void hpred_alg_freecmp(HPStage *S) { if (!S->cmp) return; int kprev = -1; for (int c = 0; c < S->nc; c++) { if (S->cmp[c].k != kprev) { free(S->cmp[c].W); kprev = S->cmp[c].k; } } free(S->cmp); S->cmp = NULL; }
static int hpred_alg(HPStage *S, const HPJob *J, HPJob *Jn, const Params *par) {
    if (!par->symfile || !par->symalg) return 0;
    const size_t Wm = (size_t)J->Lm + 2; const int nb = J->nblk;
    /* the double analysis says which blocks have structure worth the reduction */
    char *want = (char *)calloc((size_t)nb + 1, 1); int any = 0;
    {
        Problem P; hpred_shadow(J, &P);
        PSAlg *A = alg_reduce(&P, par->symalg, par->symalgmax, par->symmin, -1);
        if (A) {
            /* Is the basis worth its cost in this precision? In multiply-adds of the target
             * arithmetic: an iteration costs about 5 n^3 of block algebra and, per constraint
             * with nnz entries in the block, 2 min(nnz, n) n^2 + m nnz / 2 of assembly (the
             * rule of symalg.c); after the split 5 sum d^3 and 2 sum d^3 + m sum d^2 / 2. The
             * analysis, its verification and the map of X back cost about 150 n^3 for a
             * double-double solve (14 products in double-double, 5 in quad-double, n^3 in the
             * master precision), 500 for quad-double, 1000 above; the solve has about 8
             * double-double iterations, 15 quad-double ones at 13.5 times the cost, and more
             * above. Measured: SDPLIB thetaG11 (n = 801, four orbits: nothing to assemble)
             * 50 s as read, 97 s with the basis (22 + 68 s of analysis, 40 s for X); neu2
             * (n = 252, 113 dense orbit matrices) 487 s against 6 s. */
            const double wit = g_hpred_bits <= 106 ? 8.0 : g_hpred_bits <= 212 ? 8.0 + 15.0 * 13.5 : 8.0 + 15.0 * 13.5 + 700.0;
            const double can = g_hpred_bits <= 106 ? 150.0 : g_hpred_bits <= 212 ? 500.0 : 1000.0;
            for (int k = 0; k < nb; k++) {
                if (J->bs[k] < 2) continue;
                double sd2 = 0, sd3 = 0; int has = 0;
                for (int c = 0; c < A->ncomp; c++) if (A->ck[c] == k) { const double d = A->cd[c]; sd2 += d * d; sd3 += d * d * d; has = 1; }
                if (!has) continue;
                const double n = J->bs[k], n3 = n * n * n, mm = J->m;
                double before = 5.0 * n3, after = 5.0 * sd3;
                for (int tt = 0; tt < J->ncon[k]; tt++) { const double nz = (double)(J->ptr[k][tt + 1] - J->ptr[k][tt]); before += 2.0 * fmin(nz, n) * n * n + 0.5 * mm * nz; after += 2.0 * sd3 + 0.5 * mm * sd2; }
                const int pays = wit * (before - after) >= 1.5 * can * n3 || getenv("BRISK_HPREDSMALL") != NULL;
                if (getenv("BRISK_HPDBG")) printf("   [algebra, block %d (n = %d): work per iteration %.2e -> %.2e, analysis %.2e: %s]\n", k + 1, J->bs[k], before, after, can * n3, pays ? "worth it" : "not worth it");
                if (pays) { want[k] = 1; any = 1; }
            }
            alg_free(A);
        }
        problem_free(&P);
    }
    if (!any) { free(want); return 0; }
    memset(S, 0, sizeof(*S)); S->kind = 3; hpst_layout(S, J);
    S->bmap = (int *)malloc(sizeof(int) * ((size_t)nb + 1));
    int nc = 0, cap = 16; S->cmp = (HACmp *)malloc(sizeof(HACmp) * (size_t)cap);
    int *c0 = (int *)calloc((size_t)nb + 2, sizeof(int));
    any = 0;
    for (int k = 0; k < nb; k++) {
        c0[k] = nc;
        if (!want[k] || J->bs[k] < 2) continue;
        HACmp *cm = NULL; const int n1 = hpalg_block(J, k, &cm, par->verbose);
        if (n1 <= 0) continue;
        if (nc + n1 > cap) { cap = 2 * (nc + n1); S->cmp = (HACmp *)realloc(S->cmp, sizeof(HACmp) * (size_t)cap); }
        memcpy(S->cmp + nc, cm, sizeof(HACmp) * (size_t)n1); nc += n1; free(cm); any = 1;
    }
    c0[nb] = nc; S->nc = nc;
    free(want);
    if (!any) { free(S->bmap); free(S->cmp); free(S->bs0); free(c0); return 0; }
    /* the reduced data: mult W1 A W1' per component (W1: its first copy), entries below
     * 2^-(1.2 target bits) of the largest dropped (structural zeros of the reduced matrices) */
    HTl T; htl_init(&T, Wm, J->nnzA + J->nnzC + 1);
    int *nbsz = (int *)malloc(sizeof(int) * ((size_t)nb + nc + 2)); int rb = 0;
    for (int k = 0; k < nb; k++) { if (c0[k + 1] > c0[k]) { S->bmap[k] = -1; continue; } S->bmap[k] = rb; nbsz[rb++] = J->bs[k]; }
    for (int k = 0; k < nb; k++) {
        if (S->bmap[k] < 0) continue;
        const int lp = J->bs[k] < 0;
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) memcpy(htl_add(&T, -1, S->bmap[k], J->ci[e], lp ? J->ci[e] : J->cj[e]), J->Cv + e * Wm, Wm * sizeof(uint64_t));
        for (int tt = 0; tt < J->ncon[k]; tt++) for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) memcpy(htl_add(&T, J->con[k][tt], S->bmap[k], J->ei[e], lp ? J->ei[e] : J->ej[e]), J->Av + e * Wm, Wm * sizeof(uint64_t));
    }
    const double drop = ldexp(1.0, -(int)(1.2 * g_hpred_bits));
    int nlp = 0;
    const double td0 = wtime();
    /* per component: the matrices of C and of every constraint of its block, then its place */
    for (int c = 0; c < nc; c++) {
        HACmp *cc = &S->cmp[c]; const int k = cc->k, n = J->bs[k], d = cc->d;
        const uint64_t *W1 = cc->W;
        const int nmat = J->ncon[k] + 1;
        uint64_t *Bv = (uint64_t *)calloc(((size_t)nmat * d * d + 1) * Wm, sizeof(uint64_t));
        double *bmax = (double *)calloc((size_t)nmat + 1, sizeof(double));
#pragma omp parallel for schedule(dynamic, 4) if (nmat > 8)
        for (int tq = 0; tq < nmat; tq++) {
            const int tt = tq - 1;
            const size_t e0 = tt < 0 ? J->cptr[k] : J->ptr[k][tt], e1 = tt < 0 ? J->cptr[k + 1] : J->ptr[k][tt + 1];
            const uint64_t *Vv = tt < 0 ? J->Cv : J->Av; const int *I = tt < 0 ? J->ci : J->ei, *Jx = tt < 0 ? J->cj : J->ej;
            uint64_t *B = Bv + (size_t)tq * d * d * Wm;
            uint64_t t[MPX_MAXL + 2], t2[MPX_MAXL + 2];
            if ((e1 - e0) < (size_t)n) {
                for (size_t e = e0; e < e1; e++) {
                    const int i = I[e], j = Jx[e]; const uint64_t *v = Vv + e * Wm;
                    for (int a = 0; a < d; a++) {
                        mpx_mul(t, v, W1 + ((size_t)a * n + i) * Wm); mpx_mul(t2, v, W1 + ((size_t)a * n + j) * Wm);
                        for (int b = a; b < d; b++) { mpx_addmul(B + ((size_t)a * d + b) * Wm, t, W1 + ((size_t)b * n + j) * Wm); if (i != j) mpx_addmul(B + ((size_t)a * d + b) * Wm, t2, W1 + ((size_t)b * n + i) * Wm); }
                    }
                }
            } else {
                uint64_t *Y = (uint64_t *)calloc(((size_t)n + 1) * Wm, sizeof(uint64_t));
                for (int b = 0; b < d; b++) {
                    memset(Y, 0, sizeof(uint64_t) * (size_t)n * Wm);
                    for (size_t e = e0; e < e1; e++) { const int i = I[e], j = Jx[e]; mpx_addmul(Y + (size_t)i * Wm, Vv + e * Wm, W1 + ((size_t)b * n + j) * Wm); if (i != j) mpx_addmul(Y + (size_t)j * Wm, Vv + e * Wm, W1 + ((size_t)b * n + i) * Wm); }
                    for (int a = 0; a <= b; a++) mpx_dot((size_t)n, W1 + (size_t)a * n * Wm, Y, B + ((size_t)a * d + b) * Wm);
                }
                free(Y);
            }
            double mx = 0; for (int a = 0; a < d; a++) for (int b = a; b < d; b++) { const double v = fabs(mpx_getd(B + ((size_t)a * d + b) * Wm)); if (v > mx) mx = v; }
            double dmx = 0; for (size_t e = e0; e < e1; e++) { const double v = fabs(mpx_getd(Vv + e * Wm)); if (v > dmx) dmx = v; }
            bmax[tq] = mx > drop * dmx ? mx : 0.0;             /* 0: the matrix vanishes on this component */
        }
        int allzero = 1; for (int tq = 0; tq < nmat; tq++) if (bmax[tq] > 0) allzero = 0;
        if (!allzero) {
            if (d >= 2) { cc->rblk = rb; nbsz[rb++] = d; } else cc->rpos = nlp++;
            uint64_t t[MPX_MAXL + 2];
            for (int tq = 0; tq < nmat; tq++) {
                if (!(bmax[tq] > 0)) continue;
                const int con = tq == 0 ? -1 : J->con[k][tq - 1];
                const uint64_t *B = Bv + (size_t)tq * d * d * Wm;
                for (int a = 0; a < d; a++) for (int b = a; b < d; b++) {
                    if (fabs(mpx_getd(B + ((size_t)a * d + b) * Wm)) <= drop * bmax[tq]) continue;
                    mpx_muld(t, B + ((size_t)a * d + b) * Wm, (double)cc->mult);
                    /* (LP pieces get their block number below: -2 - place for now) */
                    mpx_copy(htl_add(&T, con, d >= 2 ? cc->rblk : -2 - cc->rpos, d >= 2 ? a : 0, d >= 2 ? b : 0), t);
                }
            }
        }
        free(Bv); free(bmax);
    }
    if (getenv("BRISK_HPDBG")) printf("   [algebra: reduced data %.2fs]\n", wtime() - td0);
    if (nlp > 0) { for (size_t q = 0; q < T.n; q++) if (T.E[q].blk <= -2) { const int p = -2 - T.E[q].blk; T.E[q].blk = rb; T.E[q].i = T.E[q].j = p; } nbsz[rb++] = -nlp; }
    for (int c = 0; c < nc; c++) if (S->cmp[c].rpos >= 0) S->cmp[c].rblk = rb - 1;
    free(c0);
    uint64_t *b = hpred_bcopy(J, J->m); memcpy(b, J->b, sizeof(uint64_t) * (size_t)J->m * Wm);
    if (htl_build(&T, Jn, J->Lm, J->m, rb, nbsz, b, 0)) { hp_free(Jn); htl_free(&T); hpred_alg_freecmp(S); free(S->bmap); free(S->bs0); return 0; }
    htl_free(&T);
    S->m1 = J->m; S->nb1 = rb;
    if (par->verbose >= 0) {
        printf("presolve: block structure of the data (matrix *-algebra, basis in the working precision):");
        int shown = 0;
        for (int k = 0; k < nb && shown < 6; k++) {
            if (S->bmap[k] >= 0) continue;
            printf("%s %d ->", shown ? ";" : "", J->bs[k]);
            int ncmp = 0; for (int c = 0; c < nc; c++) if (S->cmp[c].k == k) { if (ncmp < 12) { if (S->cmp[c].rblk < 0) printf(" (%d dropped)", S->cmp[c].d * S->cmp[c].mult); else if (S->cmp[c].mult > 1) printf(" %dx%d", S->cmp[c].mult, S->cmp[c].d); else printf(" %d", S->cmp[c].d); } ncmp++; }
            if (ncmp > 12) printf(" ... (%d components)", ncmp);
            shown++;
        }
        printf("\n");
    }
    return 1;
}
/* X_k = sum_c sum_r W_cr' X_c W_cr (rows of W: the basis); y unchanged */
static void hpred_alg_back(const HPStage *S, uint64_t *const *X1, const uint64_t *y1, uint64_t **X0, uint64_t *y0) {
    const size_t Wm = (size_t)mpx_W;
    memcpy(y0, y1, sizeof(uint64_t) * (size_t)S->m0 * Wm);
    for (int k = 0; k < S->nb0; k++) {
        if (S->bmap[k] < 0) continue;
        const size_t len = S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k];
        memcpy(X0[k], X1[S->bmap[k]], sizeof(uint64_t) * len * Wm);
    }
    for (int c = 0; c < S->nc; c++) {
        const HACmp *cc = &S->cmp[c];
        if (cc->rblk < 0) continue;
        const int k = cc->k, n = S->bs0[k], d = cc->d;
        const uint64_t *Xc = d >= 2 ? X1[cc->rblk] : X1[cc->rblk] + (size_t)cc->rpos * Wm;
        uint64_t *Tm = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)d * n + 1) * Wm);
        for (int r = 0; r < cc->mult; r++) {
            const uint64_t *Wr = cc->W + (size_t)r * d * n * Wm;
            gemm_rows_mp(d, d, n, Xc, Wr, Tm);                                   /* T = X_c W_r (d x n) */
#pragma omp parallel for schedule(static) if ((double)n * n * d > 20000.0)
            for (int i = 0; i < n; i++) for (int a = 0; a < d; a++) axpy_mp((size_t)n, Wr + ((size_t)a * n + i) * Wm, Tm + (size_t)a * n * Wm, X0[k] + (size_t)i * n * Wm);
        }
        free(Tm);
    }
    for (int k = 0; k < S->nb0; k++) if (S->bmap[k] < 0) symm_mp(S->bs0[k], X0[k]);
}

/* -- 5. rows without entries (after a reduction): removed when b = 0 (y = 0) */
static int hpred_empty(HPStage *S, const HPJob *J, HPJob *Jn, const Params *par) {
    const size_t Wm = (size_t)J->Lm + 2; const int m = J->m;
    char *has = (char *)calloc((size_t)m + 1, 1);
    for (int k = 0; k < J->nblk; k++) for (int tt = 0; tt < J->ncon[k]; tt++) if (J->ptr[k][tt + 1] > J->ptr[k][tt]) has[J->con[k][tt]] = 1;
    int nrem = 0, nbad = 0;
    for (int c = 0; c < m; c++) if (!has[c]) { if (mpx_sgn(J->b + (size_t)c * Wm) == 0) nrem++; else nbad++; }
    if (!nrem) { free(has); return 0; }
    int *cmap = (int *)malloc(sizeof(int) * ((size_t)m + 1)); int mk = 0;
    for (int c = 0; c < m; c++) cmap[c] = has[c] || mpx_sgn(J->b + (size_t)c * Wm) != 0 ? mk++ : -1;
    free(has);
    (void)nbad;
    int *bs = (int *)malloc(sizeof(int) * ((size_t)J->nblk + 1)); memcpy(bs, J->bs, sizeof(int) * (size_t)J->nblk);
    uint64_t *b = hpred_bcopy(J, mk);
    for (int c = 0; c < m; c++) if (cmap[c] >= 0) memcpy(b + (size_t)cmap[c] * Wm, J->b + (size_t)c * Wm, Wm * sizeof(uint64_t));
    HTl T; htl_init(&T, Wm, J->nnzA + J->nnzC + 1);
    for (int k = 0; k < J->nblk; k++) {
        const int lp = J->bs[k] < 0;
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) memcpy(htl_add(&T, -1, k, J->ci[e], lp ? J->ci[e] : J->cj[e]), J->Cv + e * Wm, Wm * sizeof(uint64_t));
        for (int tt = 0; tt < J->ncon[k]; tt++) for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) memcpy(htl_add(&T, cmap[J->con[k][tt]], k, J->ei[e], lp ? J->ei[e] : J->ej[e]), J->Av + e * Wm, Wm * sizeof(uint64_t));
    }
    if (htl_build(&T, Jn, J->Lm, mk, J->nblk, bs, b, 0)) { hp_free(Jn); htl_free(&T); free(cmap); return 0; }
    htl_free(&T);
    memset(S, 0, sizeof(*S)); S->kind = 5; S->cmap = cmap; hpst_layout(S, J); S->m1 = mk; S->nb1 = J->nblk;
    if (par->verbose >= 0) printf("presolve: %d constraint(s) without entries after the reductions removed\n", nrem);
    return 1;
}

/* -- 4. the chordal decomposition (overlap equalities) */
/* The cliques are those of the double pipeline's conversion (its moment form, the default,
 * run on a double copy: its gate decides); the problem built here ties the clique blocks by
 * equalities on the entries a clique shares with its parent in the clique tree. Every entry
 * goes to one clique that contains both its indices. The dual form of the driver then turns
 * this into the form with shared entries when that has fewer rows. */
static int hpred_chordal(HPStage *S, const HPJob *J, HPJob *Jn, const Params *par) {
    if (par->chordal == 0) return 0;
    const size_t Wm = (size_t)J->Lm + 2; const int m0 = J->m, nb0 = J->nblk;
    Problem P; hpred_shadow(J, &P);
    P.ps = ps_new(&P);
    Params p2 = *par; p2.chordal_form = 1;
    const int nconv = chordal_convert(&P, &p2, getenv("BRISK_HPDBG") ? 1 : 0);
    Postsolve *ps = P.ps;
    const PSMoment *M = ps ? ps->mom : NULL;
    if (!nconv || !M || M->nd <= 0) { ps_free(ps); P.ps = NULL; problem_free(&P); return 0; }
    memset(S, 0, sizeof(*S));
    S->nch = M->nd; S->ch = (PSChordal *)calloc((size_t)M->nd + 1, sizeof(PSChordal));
    int ok = 1;
    int *dq = (int *)malloc(sizeof(int) * ((size_t)nb0 + 1));          /* block -> its record, -1 */
    for (int k = 0; k < nb0; k++) dq[k] = -1;
    for (int q = 0; q < M->nd; q++) {
        const PSChordal *r = &M->dch[q]; PSChordal *d = &S->ch[q];
        d->pos = M->dpos[q]; d->n = r->n; d->ncl = r->ncl;
        d->cptr = (int *)malloc(sizeof(int) * ((size_t)r->ncl + 1)); memcpy(d->cptr, r->cptr, sizeof(int) * ((size_t)r->ncl + 1));
        d->cv = (int *)malloc(sizeof(int) * ((size_t)r->cptr[r->ncl] + 1)); memcpy(d->cv, r->cv, sizeof(int) * (size_t)r->cptr[r->ncl]);
        d->par = (int *)malloc(sizeof(int) * ((size_t)r->ncl + 1)); memcpy(d->par, r->par, sizeof(int) * (size_t)r->ncl);
        if (d->pos < 0 || d->pos >= nb0 || J->bs[d->pos] != d->n || dq[d->pos] >= 0) ok = 0; else dq[d->pos] = q;
    }
    ps_free(ps); P.ps = NULL; problem_free(&P);
    /* the blocks: the ones kept, in their order, then the cliques record by record */
    S->bmap = (int *)malloc(sizeof(int) * ((size_t)nb0 + 1));
    int nb1 = 0, ncl = 0, cmax = 0;
    for (int q = 0; q < S->nch; q++) ncl += S->ch[q].ncl;
    int *bs = (int *)malloc(sizeof(int) * ((size_t)nb0 + ncl + 1)), *c0 = (int *)malloc(sizeof(int) * ((size_t)S->nch + 1));
    for (int k = 0; k < nb0; k++) { if (dq[k] >= 0) { S->bmap[k] = -1; continue; } S->bmap[k] = nb1; bs[nb1++] = J->bs[k]; }
    for (int q = 0; q < S->nch; q++) { c0[q] = nb1; for (int t = 0; t < S->ch[q].ncl; t++) { const int c = S->ch[q].cptr[t + 1] - S->ch[q].cptr[t]; bs[nb1++] = c; if (c > cmax) cmax = c; } }
    HTl T; htl_init(&T, Wm, J->nnzA + J->nnzC + 1);
    long nover = 0; int m1 = m0;
    for (int k = 0; k < nb0 && ok; k++) {
        const int lp = J->bs[k] < 0;
        if (dq[k] < 0) {
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) memcpy(htl_add(&T, -1, S->bmap[k], J->ci[e], lp ? J->ci[e] : J->cj[e]), J->Cv + e * Wm, Wm * sizeof(uint64_t));
            for (int tt = 0; tt < J->ncon[k]; tt++) for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) memcpy(htl_add(&T, J->con[k][tt], S->bmap[k], J->ei[e], lp ? J->ei[e] : J->ej[e]), J->Av + e * Wm, Wm * sizeof(uint64_t));
            continue;
        }
        const PSChordal *r = &S->ch[dq[k]]; const int n = r->n, q0 = c0[dq[k]];
        /* node -> the cliques that contain it (increasing), with its place in each */
        int *np = (int *)calloc((size_t)n + 2, sizeof(int));
        for (int t = 0; t < r->ncl; t++) for (int a = r->cptr[t]; a < r->cptr[t + 1]; a++) np[r->cv[a] + 1]++;
        for (int i = 0; i < n; i++) np[i + 1] += np[i];
        int *nc = (int *)malloc(sizeof(int) * ((size_t)np[n] + 1)), *nl = (int *)malloc(sizeof(int) * ((size_t)np[n] + 1)), *fill = (int *)malloc(sizeof(int) * ((size_t)n + 1));
        memcpy(fill, np, sizeof(int) * (size_t)n);
        for (int t = 0; t < r->ncl; t++) for (int a = r->cptr[t]; a < r->cptr[t + 1]; a++) { const int i = r->cv[a]; nc[fill[i]] = t; nl[fill[i]] = a - r->cptr[t]; fill[i]++; }
        for (int tt = -1; tt < J->ncon[k] && ok; tt++) {
            const size_t e0 = tt < 0 ? J->cptr[k] : J->ptr[k][tt], e1 = tt < 0 ? J->cptr[k + 1] : J->ptr[k][tt + 1];
            const uint64_t *V = tt < 0 ? J->Cv : J->Av; const int *I = tt < 0 ? J->ci : J->ei, *Jx = tt < 0 ? J->cj : J->ej;
            for (size_t e = e0; e < e1; e++) {
                const int i = I[e], j = Jx[e];
                int a = np[i], b = np[j], hit = 0;
                while (a < np[i + 1] && b < np[j + 1]) { if (nc[a] == nc[b]) { hit = 1; break; } if (nc[a] < nc[b]) a++; else b++; }
                if (!hit) { ok = 0; break; }                       /* (an entry outside the cliques: not a cover of the pattern) */
                memcpy(htl_add(&T, tt < 0 ? -1 : J->con[k][tt], q0 + nc[a], nl[a], nl[b]), V + e * Wm, Wm * sizeof(uint64_t));
            }
        }
        /* the entries a clique shares with its parent: X_t[a, b] - X_p[a, b] = 0 */
        int *lpn = (int *)malloc(sizeof(int) * ((size_t)n + 1));
        for (int i = 0; i < n; i++) lpn[i] = -1;
        for (int t = 0; t < r->ncl && ok; t++) {
            const int p = r->par[t];
            if (p < 0) continue;
            for (int a = r->cptr[p]; a < r->cptr[p + 1]; a++) lpn[r->cv[a]] = a - r->cptr[p];
            for (int a = r->cptr[t]; a < r->cptr[t + 1]; a++) {
                if (lpn[r->cv[a]] < 0) continue;
                for (int b = a; b < r->cptr[t + 1]; b++) {
                    if (lpn[r->cv[b]] < 0) continue;
                    mpx_setd(htl_add(&T, m1, q0 + t, a - r->cptr[t], b - r->cptr[t]), 1.0);
                    mpx_setd(htl_add(&T, m1, q0 + p, lpn[r->cv[a]], lpn[r->cv[b]]), -1.0);
                    m1++; nover++;
                }
            }
            for (int a = r->cptr[p]; a < r->cptr[p + 1]; a++) lpn[r->cv[a]] = -1;
        }
        free(np); free(nc); free(nl); free(fill); free(lpn);
    }
    free(dq); free(c0);
    if (!ok) { htl_free(&T); free(bs); hpst_free(S); return 0; }
    uint64_t *bv = hpred_bcopy(J, m1); memcpy(bv, J->b, sizeof(uint64_t) * (size_t)m0 * Wm);
    for (int c = m0; c < m1; c++) mpx_setd(bv + (size_t)c * Wm, 0.0);
    if (htl_build(&T, Jn, J->Lm, m1, nb1, bs, bv, 0)) { hp_free(Jn); htl_free(&T); hpst_free(S); return 0; }
    htl_free(&T);
    /* Worth it for this solver? Its Schur complement is factored within an envelope (hp_schur_order,
     * no general sparse factorization): about the sum of the squared row widths, env^2 / m,
     * against 3 n^3 of block algebra per iteration (measured on maxG11: 800 -> 598 cliques
     * of at most 24 gives rows about 1000 wide, 13 s per double-double iteration against 2.6).
     * Applied when the estimate is at most half. (-chordal 1 applies it always.) */
    {
        double c0 = 0, c1 = 0;
        for (int k = 0; k < nb0; k++) if (J->bs[k] > 0) c0 += 3.0 * (double)J->bs[k] * J->bs[k] * J->bs[k];
        for (int k = 0; k < nb1; k++) if (bs[k] > 0) c1 += 3.0 * (double)bs[k] * bs[k] * bs[k];
        const double e0 = hp_envelope(J, J->mpos, NULL), e1 = hp_envelope(Jn, Jn->mpos, NULL);
        c0 += e0 * e0 / (m0 > 0 ? m0 : 1); c1 += e1 * e1 / (m1 > 0 ? m1 : 1);
        if (getenv("BRISK_HPDBG")) printf("   [chordal: %d block(s) -> %d cliques (largest %d), m %d -> %d; estimated work per iteration %.2e -> %.2e]\n", S->nch, ncl, cmax, m0, m1, c0, c1);
        if (par->chordal < 0 && c1 > 0.5 * c0) { hp_free(Jn); hpst_free(S); return 0; }
    }
    S->kind = 4; hpst_layout(S, J); S->m1 = m1; S->nb1 = nb1;
    if (par->verbose >= 0) printf("presolve: chordal decomposition: %d block(s) -> %d clique blocks (largest %d), %ld equalities on shared entries (m %d -> %d)\n", S->nch, ncl, cmax, nover, m0, m1);
    return 1;
}
/* The positive semidefinite completion of the clique blocks of one conversion (n x n, in the
 * master precision), as chordal_complete does it in double: the overlap copies are first made
 * to agree by positive semidefinite increases (children before parents: the parent's separator
 * part raised by the negative part of the difference; then parents before children: the copy
 * of the child overwritten), then X[N, R] = X_C[N, S] X_C[S, S]^-1 X[S, R] down the clique tree. */
static uint64_t *hpred_complete(const PSChordal *r, uint64_t *const *Xc) {
    const size_t W = (size_t)mpx_W; const int n = r->n, ncl = r->ncl;
    int *cnt = (int *)calloc((size_t)ncl + 1, sizeof(int)), *cp = (int *)malloc(sizeof(int) * ((size_t)ncl + 1)), *ch = (int *)malloc(sizeof(int) * ((size_t)ncl + 1));
    for (int t = 0; t < ncl; t++) if (r->par[t] >= 0) cnt[r->par[t]]++;
    cp[0] = 0; for (int t = 0; t < ncl; t++) cp[t + 1] = cp[t] + cnt[t];
    memset(cnt, 0, sizeof(int) * ((size_t)ncl + 1));
    for (int t = 0; t < ncl; t++) if (r->par[t] >= 0) { const int p = r->par[t]; ch[cp[p] + cnt[p]++] = t; }
    int *ord = (int *)malloc(sizeof(int) * ((size_t)ncl + 1)), no = 0;
    for (int t = 0; t < ncl; t++) { if (r->par[t] >= 0) continue; int h = no; ord[no++] = t; while (h < no) { const int u = ord[h++]; for (int q = cp[u]; q < cp[u + 1]; q++) ord[no++] = ch[q]; } }
    int cmax = 1; for (int t = 0; t < ncl; t++) if (r->cptr[t + 1] - r->cptr[t] > cmax) cmax = r->cptr[t + 1] - r->cptr[t];
    const size_t c2 = (size_t)cmax * cmax + 1;
    uint64_t **Xw = (uint64_t **)malloc(sizeof(uint64_t *) * ((size_t)ncl + 1));
    for (int t = 0; t < ncl; t++) { const size_t c = (size_t)(r->cptr[t + 1] - r->cptr[t]); Xw[t] = (uint64_t *)malloc(sizeof(uint64_t) * (c * c + 1) * W); memcpy(Xw[t], Xc[t], sizeof(uint64_t) * c * c * W); }
    int *lp = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *ic = (int *)malloc(sizeof(int) * ((size_t)cmax + 1)), *ip = (int *)malloc(sizeof(int) * ((size_t)cmax + 1));
    for (int i = 0; i < n; i++) lp[i] = -1;
    uint64_t *D = (uint64_t *)malloc(sizeof(uint64_t) * c2 * W), *Ve = (uint64_t *)malloc(sizeof(uint64_t) * c2 * W), *Bw = (uint64_t *)malloc(sizeof(uint64_t) * c2 * W), *Tw = (uint64_t *)malloc(sizeof(uint64_t) * c2 * W), *ev = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)cmax + 1) * W);
    uint64_t t1[MPX_MAXL + 2], t2[MPX_MAXL + 2];
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
            uint64_t *XC = Xw[t], *XP = Xw[p];
            if (pass == 0) {
                int any = 0;
                for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++) { uint64_t *d = D + ((size_t)a * ns + b) * W; mpx_sub(d, XP + ((size_t)ip[a] * cpn + ip[b]) * W, XC + ((size_t)ic[a] * c + ic[b]) * W); if (mpx_sgn(d) != 0) any = 1; }
                if (!any || eig_mp(ns, D, Ve, ev, Bw, Tw) < 0) continue;
                for (int k = 0; k < ns; k++) {
                    if (mpx_sgn(ev + (size_t)k * W) >= 0) continue;
                    for (int a = 0; a < ns; a++) {
                        mpx_mul(t1, ev + (size_t)k * W, Ve + ((size_t)k * ns + a) * W);
                        for (int b = 0; b < ns; b++) mpx_submul(XP + ((size_t)ip[a] * cpn + ip[b]) * W, t1, Ve + ((size_t)k * ns + b) * W);
                    }
                }
            } else {
                for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++) mpx_copy(XC + ((size_t)ic[a] * c + ic[b]) * W, XP + ((size_t)ip[a] * cpn + ip[b]) * W);
            }
        }
    free(D); free(Ve); free(Bw); free(Tw); free(ev); free(lp); free(ic); free(ip);
    uint64_t *XL = (uint64_t *)calloc(((size_t)n * n + 1) * W, sizeof(uint64_t));
    char *filled = (char *)calloc((size_t)n + 1, 1);
    int *fl = (int *)malloc(sizeof(int) * ((size_t)n + 1)), nf = 0, *loc = (int *)malloc(sizeof(int) * ((size_t)n + 1)), *Rn = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    for (int i = 0; i < n; i++) loc[i] = -1;
    uint64_t *Ch = (uint64_t *)malloc(sizeof(uint64_t) * c2 * W), *Wt = (uint64_t *)malloc(sizeof(uint64_t) * c2 * W), *G = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)cmax + 1) * W);
    int *Sl = (int *)malloc(sizeof(int) * ((size_t)cmax + 1)), *Nl = (int *)malloc(sizeof(int) * ((size_t)cmax + 1));
    char *dead = (char *)malloc((size_t)cmax + 1);
    const double deadtol = ldexp(1.0, -(mpx_bits() - 32));
    for (int o = 0; o < no; o++) {
        const int t = ord[o], c = r->cptr[t + 1] - r->cptr[t];
        const int *v = r->cv + r->cptr[t];
        const uint64_t *Xt = Xw[t];
        int ns = 0, nn = 0;
        for (int a = 0; a < c; a++) { if (filled[v[a]]) Sl[ns++] = a; else Nl[nn++] = a; }
        if (nn == 0) continue;
        for (int a = 0; a < nn; a++) {
            const int ia = v[Nl[a]];
            for (int b = 0; b < c; b++) { const uint64_t *x = Xt + ((size_t)Nl[a] * c + b) * W; mpx_copy(XL + ((size_t)ia * n + v[b]) * W, x); mpx_copy(XL + ((size_t)v[b] * n + ia) * W, x); }
        }
        for (int a = 0; a < c; a++) loc[v[a]] = a;
        int nr = 0;
        for (int q = 0; q < nf; q++) if (loc[fl[q]] < 0) Rn[nr++] = fl[q];
        for (int a = 0; a < c; a++) loc[v[a]] = -1;
        if (ns > 0 && nr > 0) {
            /* Cholesky of X_C[S, S] (lower, row-major); a pivot at the rounding level drops its direction */
            double dmax = 0;
            for (int a = 0; a < ns; a++) { for (int b = 0; b < ns; b++) mpx_copy(Ch + ((size_t)a * ns + b) * W, Xt + ((size_t)Sl[a] * c + Sl[b]) * W); const double d = mpx_getd(Ch + ((size_t)a * ns + a) * W); if (d > dmax) dmax = d; }
            for (int k = 0; k < ns; k++) {
                dead[k] = 0;
                mpx_copy(t1, Ch + ((size_t)k * ns + k) * W);
                for (int q = 0; q < k; q++) mpx_submul(t1, Ch + ((size_t)k * ns + q) * W, Ch + ((size_t)k * ns + q) * W);
                if (!(mpx_getd(t1) > deadtol * dmax)) { dead[k] = 1; for (int a = k; a < ns; a++) mpx_setd(Ch + ((size_t)a * ns + k) * W, 0.0); continue; }
                mpx_sqrt(t2, t1); mpx_copy(Ch + ((size_t)k * ns + k) * W, t2);
                for (int a = k + 1; a < ns; a++) {
                    mpx_copy(t1, Ch + ((size_t)a * ns + k) * W);
                    for (int q = 0; q < k; q++) mpx_submul(t1, Ch + ((size_t)a * ns + q) * W, Ch + ((size_t)k * ns + q) * W);
                    mpx_div(Ch + ((size_t)a * ns + k) * W, t1, t2);
                }
            }
            /* Wt (nn x ns, rows) = X_C[N, S] X_C[S, S]^-1 */
            for (int a = 0; a < nn; a++) {
                uint64_t *w = Wt + (size_t)a * ns * W;
                for (int b = 0; b < ns; b++) mpx_copy(w + (size_t)b * W, Xt + ((size_t)Sl[b] * c + Nl[a]) * W);
                for (int k = 0; k < ns; k++) {
                    if (dead[k]) { mpx_setd(w + (size_t)k * W, 0.0); continue; }
                    mpx_copy(t1, w + (size_t)k * W);
                    for (int q = 0; q < k; q++) mpx_submul(t1, Ch + ((size_t)k * ns + q) * W, w + (size_t)q * W);
                    mpx_div(w + (size_t)k * W, t1, Ch + ((size_t)k * ns + k) * W);
                }
                for (int k = ns - 1; k >= 0; k--) {
                    if (dead[k]) { mpx_setd(w + (size_t)k * W, 0.0); continue; }
                    mpx_copy(t1, w + (size_t)k * W);
                    for (int q = k + 1; q < ns; q++) mpx_submul(t1, Ch + ((size_t)q * ns + k) * W, w + (size_t)q * W);
                    mpx_div(w + (size_t)k * W, t1, Ch + ((size_t)k * ns + k) * W);
                }
            }
            for (int q = 0; q < nr; q++) {
                for (int b = 0; b < ns; b++) mpx_copy(G + (size_t)b * W, XL + ((size_t)v[Sl[b]] * n + Rn[q]) * W);
                for (int a = 0; a < nn; a++) {
                    mpx_dot((size_t)ns, Wt + (size_t)a * ns * W, G, t1);
                    mpx_copy(XL + ((size_t)Rn[q] * n + v[Nl[a]]) * W, t1); mpx_copy(XL + ((size_t)v[Nl[a]] * n + Rn[q]) * W, t1);
                }
            }
        }
        for (int a = 0; a < nn; a++) { filled[v[Nl[a]]] = 1; fl[nf++] = v[Nl[a]]; }
    }
    for (int t = 0; t < ncl; t++) free(Xw[t]);
    free(Xw); free(filled); free(fl); free(loc); free(Rn); free(Ch); free(Wt); free(G); free(Sl); free(Nl); free(dead); free(cnt); free(cp); free(ch); free(ord);
    return XL;
}
static void hpred_chordal_back(const HPStage *S, uint64_t *const *X1, const uint64_t *y1, uint64_t **X0, uint64_t *y0) {
    const size_t Wm = (size_t)mpx_W;
    memcpy(y0, y1, sizeof(uint64_t) * (size_t)S->m0 * Wm);
    int nk = 0;
    for (int k = 0; k < S->nb0; k++) {
        if (S->bmap[k] < 0) continue;
        const size_t len = S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k];
        memcpy(X0[k], X1[S->bmap[k]], sizeof(uint64_t) * len * Wm); nk++;
    }
    int q0 = nk;
    for (int q = 0; q < S->nch; q++) {
        const PSChordal *r = &S->ch[q];
        uint64_t *D = hpred_complete(r, X1 + q0);
        memcpy(X0[r->pos], D, sizeof(uint64_t) * (size_t)r->n * r->n * Wm);
        free(D);
        q0 += r->ncl;
    }
}

static void hpst_free(HPStage *S) {
    if (S->bu) for (int k = 0; k < S->nb0; k++) { free(S->bu[k]); free(S->bsg[k]); }
    free(S->bu); free(S->bsg); free(S->brep); free(S->bnew);
    free(S->bs0); free(S->bs1); free(S->bmap); free(S->cmap);
    if (S->sg) sign_free(S->sg);
    if (S->sy) sym_free(S->sy);
    hpred_alg_freecmp(S);
    for (int q = 0; q < S->nch; q++) { free(S->ch[q].cptr); free(S->ch[q].cv); free(S->ch[q].par); }
    free(S->ch); free(S->cbs);
    memset(S, 0, sizeof(*S));
}
static void hp_red_free(HPRed *R) { for (int s = 0; s < R->n; s++) hpst_free(&R->st[s]); R->n = 0; }

/* The reductions applied to *J: on success (return 1) *J is the reduced job and R->Jfull the
 * job it came from (to measure the solution on at the end). */
static int hp_reduce(HPJob *J, HPRed *R, const Params *par) {
    memset(R, 0, sizeof(*R));
    HPJob cur = *J; int owned = 0;
    const char *off = getenv("BRISK_HPNORED");          /* a list of the stages to leave out (1 sign, 2 orbits, 3 algebra, 4 chordal), or "all" */
    if (off && !strcmp(off, "all")) return 0;
#define HPRED_OFF(c) (off && strchr(off, (c)))
#define HPRED_TAKE(Jn_) do { HPStage *S_ = &R->st[R->n++]; S_->bs1 = (int *)malloc(sizeof(int) * ((size_t)(Jn_).nblk + 1)); memcpy(S_->bs1, (Jn_).bs, sizeof(int) * (size_t)(Jn_).nblk); \
                             if (owned) { hp_free(&cur); } cur = (Jn_); owned = 1; } while (0)
    HPJob Jn;
    if (!HPRED_OFF('1') && hpred_sign(&R->st[R->n], &cur, &Jn, par)) HPRED_TAKE(Jn);
    /* the search for a group, the analysis of the algebra and the cliques cost a few hundredths
     * of a second to a second: not on a problem that is solved in less (sum n^3 + m^3 / 3
     * below 2e7: n = 270, or m = 390; the theta problem of the 131-cycle, group of order 262:
     * 0.25 s as read, 1.1 s with 0.74 s of analysis) */
    { double w = (double)cur.m * cur.m * cur.m / 3.0; for (int k = 0; k < cur.nblk; k++) if (cur.bs[k] > 0) w += (double)cur.bs[k] * cur.bs[k] * cur.bs[k];
      if (w < 2e7 && !getenv("BRISK_HPREDSMALL")) off = "234"; }
    if (!HPRED_OFF('2') && hpred_orbit(&R->st[R->n], &cur, &Jn, par)) {
        /* the orbits are kept when they pay by themselves (rows by -symmin, or blocks removed)
         * or when the block structure of the aggregated data is then used */
        HPStage *So = &R->st[R->n];
        HPJob Ja;
        const int alg = !HPRED_OFF('3') && hpred_alg(&R->st[R->n + 1], &Jn, &Ja, par);
        if (alg || (double)So->m0 >= par->symmin * (double)So->m1 || So->nb1 < So->nb0) {
            HPRED_TAKE(Jn);
            if (alg) HPRED_TAKE(Ja);
        } else {
            if (par->verbose >= 0) printf("presolve: symmetry reduction not applied: too small a gain (%d -> %d constraints, the blocks unchanged)\n", So->m0, So->m1);
            hp_free(&Jn); hpst_free(So);
        }
    } else if (!HPRED_OFF('3') && hpred_alg(&R->st[R->n], &cur, &Jn, par)) HPRED_TAKE(Jn);
    if (!HPRED_OFF('4') && hpred_chordal(&R->st[R->n], &cur, &Jn, par)) HPRED_TAKE(Jn);
    if (R->n > 0 && hpred_empty(&R->st[R->n], &cur, &Jn, par)) HPRED_TAKE(Jn);
#undef HPRED_OFF
#undef HPRED_TAKE
    if (R->n == 0) return 0;
    R->Jfull = *J;
    const double ts = J->tstart;
    *J = cur; J->tstart = ts;
    return 1;
}
/* A double point (X, y, Z of the layout before the reductions) in the reduced layout: the start
 * of the high-precision solve. Returns 0 when a stage has no such map (the chordal
 * decomposition: Z would have to be split among the cliques). X1, Z1, y1 are allocated. */
static int hp_red_fwd(const HPRed *R, double *const *Xin, double *const *Zin, const double *yin, double ***Xout, double ***Zout, double **yout) {
    for (int s = 0; s < R->n; s++) if (R->st[s].kind == 4) return 0;
    double **X0 = (double **)Xin, **Z0 = (double **)Zin; double *y0 = (double *)yin; int own = 0;
    for (int s = 0; s < R->n; s++) {
        const HPStage *S = &R->st[s];
        double **X1 = (double **)calloc((size_t)S->nb1 + 1, sizeof(double *)), **Z1 = Z0 ? (double **)calloc((size_t)S->nb1 + 1, sizeof(double *)) : NULL;
        for (int k = 0; k < S->nb1; k++) { const size_t len = S->bs1[k] < 0 ? (size_t)(-S->bs1[k]) : (size_t)S->bs1[k] * S->bs1[k]; X1[k] = (double *)calloc(len + 1, sizeof(double)); if (Z1) Z1[k] = (double *)calloc(len + 1, sizeof(double)); }
        double *y1 = (double *)calloc((size_t)S->m1 + 1, sizeof(double));
        for (int w = 0; w < (Z0 ? 2 : 1); w++) {
            double **A0 = w ? Z0 : X0, **A1 = w ? Z1 : X1;
            if (S->kind == 1) {
                const PSSign *G = S->sg;
                for (int k = 0; k < S->nb0; k++) {
                    const int n = abs(S->bs0[k]);
                    if (S->bs0[k] < 0) { for (int i = 0; i < n; i++) { const int q = G->cls[G->off[k] + i]; A1[G->qblk[q]][G->cpos[G->off[k] + i] + G->qpos0[q]] = A0[k][i]; } continue; }
                    for (int i = 0; i < n; i++) {
                        const int qi = G->cls[G->off[k] + i], pi = G->cpos[G->off[k] + i];
                        if (G->csz[qi] == 1) { A1[G->qblk[qi]][G->qpos0[qi]] = A0[k][(size_t)i * n + i]; continue; }
                        for (int j = 0; j < n; j++) if (G->cls[G->off[k] + j] == qi) A1[G->qblk[qi]][(size_t)pi * G->csz[qi] + G->cpos[G->off[k] + j]] = A0[k][(size_t)i * n + j];
                    }
                }
            } else if (S->kind == 2) {
                int *cnt = (int *)calloc((size_t)S->nb0 + 1, sizeof(int));
                for (int k = 0; k < S->nb0; k++) cnt[S->brep[k]]++;
                for (int k = 0; k < S->nb0; k++) {
                    if (S->bnew[k] < 0) continue;
                    const size_t len = S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k];
                    const double f = w ? (double)cnt[k] : 1.0;              /* (the dual slack of the kept block: its data were scaled) */
                    for (size_t q = 0; q < len; q++) A1[S->bnew[k]][q] = f * A0[k][q];
                }
                free(cnt);
            } else if (S->kind == 3) {
                for (int k = 0; k < S->nb0; k++) { if (S->bmap[k] < 0) continue; const size_t len = S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k]; memcpy(A1[S->bmap[k]], A0[k], sizeof(double) * len); }
                for (int c = 0; c < S->nc; c++) {
                    const HACmp *cc = &S->cmp[c];
                    if (cc->rblk < 0) continue;
                    const int n = S->bs0[cc->k], d = cc->d; const size_t Wm = (size_t)mpx_W;
                    double *Wd = (double *)malloc(sizeof(double) * ((size_t)d * n + 1)), *Td = (double *)calloc((size_t)d * n + 1, sizeof(double));
                    for (size_t q = 0; q < (size_t)d * n; q++) Wd[q] = mpx_getd(cc->W + q * Wm);
                    for (int a = 0; a < d; a++) for (int i = 0; i < n; i++) { const double wv = Wd[(size_t)a * n + i]; if (wv == 0.0) continue; const double *row = A0[cc->k] + (size_t)i * n; double *t = Td + (size_t)a * n; for (int j = 0; j < n; j++) t[j] += wv * row[j]; }
                    const double f = w ? (double)cc->mult : 1.0;
                    for (int a = 0; a < d; a++) for (int b = 0; b < d; b++) {
                        double v = 0; for (int j = 0; j < n; j++) v += Td[(size_t)a * n + j] * Wd[(size_t)b * n + j];
                        if (d >= 2) A1[cc->rblk][(size_t)a * d + b] = f * v; else A1[cc->rblk][cc->rpos] = f * v;
                    }
                    free(Wd); free(Td);
                }
            } else if (S->kind == 5) {
                for (int k = 0; k < S->nb0; k++) { const size_t len = S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k]; memcpy(A1[k], A0[k], sizeof(double) * len); }
            }
        }
        if (S->kind == 1) { int q = 0; for (int c = 0; c < S->m0; c++) if (S->sg->keep[c]) y1[q++] = y0[c]; }
        else if (S->kind == 2) {
            const PSSym *G = S->sy; int *cnt = (int *)calloc((size_t)S->m1 + 1, sizeof(int));
            for (int c = 0; c < S->m0; c++) if (G->cls[c] >= 0) { y1[G->cls[c]] += (G->csg && G->csg[c] < 0 ? -1.0 : 1.0) * y0[c]; cnt[G->cls[c]]++; }
            for (int q = 0; q < S->m1; q++) if (cnt[q] > 1) y1[q] /= cnt[q];
            free(cnt);
        }
        else if (S->kind == 3) memcpy(y1, y0, sizeof(double) * (size_t)S->m0);
        else if (S->kind == 5) { for (int c = 0; c < S->m0; c++) if (S->cmap[c] >= 0) y1[S->cmap[c]] = y0[c]; }
        if (own) { for (int k = 0; k < S->nb0; k++) { free(X0[k]); if (Z0) free(Z0[k]); } free(X0); free(Z0); free(y0); }
        X0 = X1; Z0 = Z1; y0 = y1; own = 1;
    }
    *Xout = X0; *Zout = Z0; *yout = y0;
    return 1;
}
/* the solution (X, y) of the reduced job in the layout of R->Jfull (allocated here) */
static void hp_red_back(const HPRed *R, uint64_t *const *Xr, const uint64_t *yr, uint64_t ***Xf, uint64_t **yf) {
    const size_t Wm = (size_t)mpx_W;
    uint64_t **X1 = (uint64_t **)Xr; uint64_t *y1 = (uint64_t *)yr; int own = 0;
    for (int s = R->n - 1; s >= 0; s--) {
        const HPStage *S = &R->st[s];
        uint64_t **X0 = (uint64_t **)calloc((size_t)S->nb0 + 1, sizeof(uint64_t *));
        for (int k = 0; k < S->nb0; k++) X0[k] = (uint64_t *)calloc(((S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k]) + 1) * Wm, sizeof(uint64_t));
        uint64_t *y0 = (uint64_t *)calloc(((size_t)S->m0 + 1) * Wm, sizeof(uint64_t));
        if (S->kind == 1) hpred_sign_back(S, X1, y1, X0, y0);
        else if (S->kind == 2) hpred_orbit_back(S, X1, y1, X0, y0);
        else if (S->kind == 3) hpred_alg_back(S, X1, y1, X0, y0);
        else if (S->kind == 4) hpred_chordal_back(S, X1, y1, X0, y0);
        else if (S->kind == 5) {
            for (int c = 0; c < S->m0; c++) if (S->cmap[c] >= 0) memcpy(y0 + (size_t)c * Wm, y1 + (size_t)S->cmap[c] * Wm, Wm * sizeof(uint64_t));
            for (int k = 0; k < S->nb0; k++) memcpy(X0[k], X1[k], sizeof(uint64_t) * (S->bs0[k] < 0 ? (size_t)(-S->bs0[k]) : (size_t)S->bs0[k] * S->bs0[k]) * Wm);
        }
        if (own) { for (int k = 0; k < S->nb1; k++) free(X1[k]); free(X1); free(y1); }
        X1 = X0; y1 = y0; own = 1;
    }
    *Xf = X1; *yf = y1;
}
