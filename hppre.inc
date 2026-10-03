/* SPDX-License-Identifier: Apache-2.0
 * hppre.inc (5.1): presolve and postsolve of the high-precision path, included once by
 * hpsolve.c. All arithmetic is in mpx at the master precision of the solve (the target
 * precision plus guard limbs), so that the reduced problem is the original one to the working
 * precision; the decisions (which pivot, which pair) are taken on doubles.
 *
 * Steps (each is undone exactly by hp_postsolve):
 *  1. Free variables. Two LP variables whose columns and costs are exact negatives of each
 *     other are a split free variable f = x+ - x-. It is eliminated by a pivot on one of its
 *     constraints (row p: f = (b_p - <rest of row p>) / a_pf): the row is removed, the other
 *     rows and the objective lose their f entries by row operations. The dual of a split pair
 *     has no interior (z+ = -z- = 0), which is what limits the accuracy of the interior-point
 *     method on moment relaxations in the kernel form.
 *  2. Facial reduction for a constraint <A_p, X> = 0 whose matrix is semidefinite of a simple
 *     form: rank one in one SDP block (X a = 0: the block is restricted to the complement of
 *     a with the basis v_k = e_k - (a_k / a_piv) e_piv), a same-sign diagonal (those rows and
 *     columns of X vanish), same-sign LP coefficients (those variables vanish).
 *
 * The problem is held as sparse rows over the positions (block, i, j), i <= j: row 0 is C,
 * row c + 1 is A_c. */

typedef struct { int n, cap; uint64_t *key, *val; } PRow;       /* sorted by key; val: n * Wm words */
#define PKEY(k, i, j) (((uint64_t)(k) << 44) | ((uint64_t)(i) << 22) | (uint64_t)(j))
#define PKEY_BLK(key) ((int)((key) >> 44))
#define PKEY_I(key) ((int)(((key) >> 22) & 0x3fffff))
#define PKEY_J(key) ((int)((key) & 0x3fffff))

typedef struct {
    int kind;                 /* 1 free variable, 2 rank-one face, 3 deleted indices (diagonal / LP face) */
    int blk, ip, im;          /* free: the block and the two LP variables (f = x_ip - x_im); face: the block, pivot index */
    int bm;                   /* free: the block of the minus variable */
    int prow;                 /* the constraint removed (original number) */
    PRow row;                 /* free: the pivot row at its stage */
    uint64_t *bp, *apf, *g;   /* free: its right-hand side, the pivot, cost / pivot */
    int nl, *lc; uint64_t *lv; /* free: rows changed (original numbers, -1 = objective) and their multipliers */
    int na, *ai; uint64_t *av; /* rank-one face: support of a and r_k = a_k / a_piv (k != piv); deleted: the indices */
    double sgn;               /* face: the sign of the certificate */
} PRec;

typedef struct {
    int nrec, cap; PRec *rec;
    int m0, mr;               /* rows of the original and of the reduced problem */
    int *rowmap;              /* reduced row -> original row */
    int *blkmap;              /* original block -> reduced block, -1 when it vanished */
    int **idxmap;             /* per original block: index -> reduced index, -1 when removed */
    int nfree, nface, ndrop;  /* free variables eliminated, face reductions, empty rows dropped */
    int *frow, nfrow;         /* rows removed by face reductions (original numbers) */
} HPPre;

static void prow_free(PRow *R) { free(R->key); free(R->val); R->key = NULL; R->val = NULL; R->n = R->cap = 0; }
static void prow_push(PRow *R, uint64_t key, const uint64_t *v) {
    const size_t Wm = (size_t)mpx_W;
    if (R->n == R->cap) {
        R->cap = R->cap ? 2 * R->cap : 8;
        R->key = (uint64_t *)realloc(R->key, sizeof(uint64_t) * (size_t)R->cap);
        R->val = (uint64_t *)realloc(R->val, sizeof(uint64_t) * (size_t)R->cap * Wm);
    }
    R->key[R->n] = key; memcpy(R->val + (size_t)R->n * Wm, v, sizeof(uint64_t) * Wm); R->n++;
}
static int prow_find(const PRow *R, uint64_t key) {
    int lo = 0, hi = R->n - 1;
    while (lo <= hi) { const int mid = (lo + hi) / 2; if (R->key[mid] == key) return mid; if (R->key[mid] < key) lo = mid + 1; else hi = mid - 1; }
    return -1;
}
static void prow_del(PRow *R, int pos) {
    const size_t Wm = (size_t)mpx_W;
    memmove(R->key + pos, R->key + pos + 1, sizeof(uint64_t) * (size_t)(R->n - pos - 1));
    memmove(R->val + (size_t)pos * Wm, R->val + (size_t)(pos + 1) * Wm, sizeof(uint64_t) * Wm * (size_t)(R->n - pos - 1));
    R->n--;
}
static void prow_copy(PRow *D, const PRow *S) {
    const size_t Wm = (size_t)mpx_W;
    D->n = D->cap = S->n;
    D->key = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(S->n + 1)); D->val = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)S->n + 1) * Wm);
    memcpy(D->key, S->key, sizeof(uint64_t) * (size_t)S->n); memcpy(D->val, S->val, sizeof(uint64_t) * (size_t)S->n * Wm);
}
/* R <- R - l S; an entry that cancels to the rounding level of the master precision is dropped;
 * skip: a key of S that is not carried over (its entry in R is deleted) */
static void prow_axpy(PRow *R, const uint64_t *l, const PRow *S, uint64_t skip, double droptol) {
    const size_t Wm = (size_t)mpx_W;
    PRow N; memset(&N, 0, sizeof(N));
    N.cap = R->n + S->n + 1;
    N.key = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)N.cap); N.val = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)N.cap * Wm);
    int a = 0, b = 0;
    uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2];
    while (a < R->n || b < S->n) {
        const uint64_t ka = a < R->n ? R->key[a] : UINT64_MAX, kb = b < S->n ? S->key[b] : UINT64_MAX;
        if (ka < kb) { if (ka != skip) { N.key[N.n] = ka; memcpy(N.val + (size_t)N.n * Wm, R->val + (size_t)a * Wm, sizeof(uint64_t) * Wm); N.n++; } a++; continue; }
        mpx_mul(t, l, S->val + (size_t)b * Wm);
        if (kb < ka) {
            if (kb != skip && mpx_sgn(t) != 0) { mpx_neg(u, t); N.key[N.n] = kb; memcpy(N.val + (size_t)N.n * Wm, u, sizeof(uint64_t) * Wm); N.n++; }
            b++; continue;
        }
        if (ka != skip) {
            mpx_sub(u, R->val + (size_t)a * Wm, t);
            const double big = fabs(mpx_getd(R->val + (size_t)a * Wm)) + fabs(mpx_getd(t));
            if (mpx_sgn(u) != 0 && fabs(mpx_getd(u)) > droptol * big) { N.key[N.n] = ka; memcpy(N.val + (size_t)N.n * Wm, u, sizeof(uint64_t) * Wm); N.n++; }
        }
        a++; b++;
    }
    free(R->key); free(R->val);
    *R = N;
}

static void hp_pre_free(HPPre *P, int nblk0) {
    if (!P) return;
    for (int r = 0; r < P->nrec; r++) {
        PRec *q = &P->rec[r];
        prow_free(&q->row); free(q->bp); free(q->apf); free(q->g); free(q->lc); free(q->lv); free(q->ai); free(q->av);
    }
    free(P->rec); free(P->rowmap); free(P->blkmap); free(P->frow);
    if (P->idxmap) { for (int k = 0; k < nblk0; k++) free(P->idxmap[k]); free(P->idxmap); }
    free(P);
}
static PRec *pre_rec(HPPre *P) {
    if (P->nrec == P->cap) { P->cap = P->cap ? 2 * P->cap : 64; P->rec = (PRec *)realloc(P->rec, sizeof(PRec) * (size_t)P->cap); }
    PRec *q = &P->rec[P->nrec++];
    memset(q, 0, sizeof(*q));
    return q;
}

static const uint64_t *g_pre_keys;
static int cmp_pre_ord(const void *a, const void *b) { const uint64_t x = g_pre_keys[*(const int *)a], y = g_pre_keys[*(const int *)b]; return x < y ? -1 : x > y; }
typedef struct { int blk, i; uint64_t h; } PVar;
static int cmp_pvar(const void *a, const void *b) { const uint64_t x = ((const PVar *)a)->h, y = ((const PVar *)b)->h; return x < y ? -1 : x > y; }

/* Presolve of J0 (built) into Jr (built). Returns NULL when nothing was reduced. */
static HPPre *hp_presolve(const HPJob *J0, HPJob *Jr, int verbose, int allow_face) {
    const size_t Wm = (size_t)J0->Lm + 2;
    const int m0 = J0->m, nb0 = J0->nblk;
    const double droptol = ldexp(1.0, -(64 * (J0->Lm - 1) - 40));
    const int do_free = !getenv("BRISK_HPNOFREE"), do_face = allow_face && !getenv("BRISK_HPNOFACE");
    uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2];
    /* rows */
    PRow *rows = (PRow *)calloc((size_t)m0 + 1, sizeof(PRow));
    uint64_t *b = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)m0 * Wm);
    memcpy(b, J0->b, sizeof(uint64_t) * (size_t)m0 * Wm);
    char *alive = (char *)malloc((size_t)m0 + 1);
    memset(alive, 1, (size_t)m0 + 1);
    for (int k = 0; k < nb0; k++) {
        for (size_t e = J0->cptr[k]; e < J0->cptr[k + 1]; e++) prow_push(&rows[0], PKEY(k, J0->ci[e], J0->bs[k] < 0 ? J0->ci[e] : J0->cj[e]), J0->Cv + e * Wm);
        for (int tt = 0; tt < J0->ncon[k]; tt++)
            for (size_t e = J0->ptr[k][tt]; e < J0->ptr[k][tt + 1]; e++)
                prow_push(&rows[J0->con[k][tt] + 1], PKEY(k, J0->ei[e], J0->bs[k] < 0 ? J0->ei[e] : J0->ej[e]), J0->Av + e * Wm);
    }
    HPPre *P = (HPPre *)calloc(1, sizeof(HPPre));
    P->m0 = m0;
    /* removed LP variables / SDP indices */
    char **gone = (char **)malloc(sizeof(char *) * (size_t)nb0);
    for (int k = 0; k < nb0; k++) gone[k] = (char *)calloc((size_t)abs(J0->bs[k]) + 1, 1);

    /* ---- 1. split free variables ---- */
    if (do_free) {
        int nv = 0;
        for (int k = 0; k < nb0; k++) if (J0->bs[k] < 0) nv += -J0->bs[k];
        PVar *V = (PVar *)malloc(sizeof(PVar) * (size_t)(nv + 1));
        nv = 0;
        for (int k = 0; k < nb0; k++) {
            if (J0->bs[k] >= 0) continue;
            const int n = -J0->bs[k];
            for (int i = 0; i < n; i++) {
                uint64_t h = 1469598103934665603ULL;
                for (int q = J0->lpptr[k][i]; q < J0->lpptr[k][i + 1]; q++) {
                    const double d = fabs(mpx_getd(J0->Av + (size_t)J0->lpe[k][q] * Wm));
                    uint64_t bits; memcpy(&bits, &d, 8);
                    h = (h ^ (uint64_t)J0->lpc[k][q]) * 1099511628211ULL; h = (h ^ bits) * 1099511628211ULL;
                }
                if (J0->lpptr[k][i + 1] == J0->lpptr[k][i]) continue;        /* in no constraint: left alone */
                V[nv].blk = k; V[nv].i = i; V[nv].h = h; nv++;
            }
        }
        qsort(V, (size_t)nv, sizeof(PVar), cmp_pvar);
        /* cost of an LP variable */
        #define PCOST(k, i) (prow_find(&rows[0], PKEY(k, i, i)))
        int npair = 0, *pa = (int *)malloc(sizeof(int) * (size_t)(nv + 1)), *pb = (int *)malloc(sizeof(int) * (size_t)(nv + 1));
        char *used = (char *)calloc((size_t)nv + 1, 1);
        for (int x = 0; x < nv; x++) {
            if (used[x]) continue;
            for (int y = x + 1; y < nv && V[y].h == V[x].h; y++) {
                if (used[y]) continue;
                const int k1 = V[x].blk, i1 = V[x].i, k2 = V[y].blk, i2 = V[y].i;
                const int n1 = J0->lpptr[k1][i1 + 1] - J0->lpptr[k1][i1];
                if (n1 != J0->lpptr[k2][i2 + 1] - J0->lpptr[k2][i2]) continue;
                int ok = 1;
                for (int q = 0; q < n1 && ok; q++) {
                    const int q1 = J0->lpptr[k1][i1] + q, q2 = J0->lpptr[k2][i2] + q;
                    if (J0->lpc[k1][q1] != J0->lpc[k2][q2]) { ok = 0; break; }
                    mpx_add(t, J0->Av + (size_t)J0->lpe[k1][q1] * Wm, J0->Av + (size_t)J0->lpe[k2][q2] * Wm);
                    if (mpx_sgn(t) != 0) ok = 0;
                }
                if (!ok) continue;
                const int c1 = PCOST(k1, i1), c2 = PCOST(k2, i2);
                if ((c1 < 0) != (c2 < 0)) continue;
                if (c1 >= 0) { mpx_add(t, rows[0].val + (size_t)c1 * Wm, rows[0].val + (size_t)c2 * Wm); if (mpx_sgn(t) != 0) continue; }
                pa[npair] = x; pb[npair] = y; npair++; used[x] = used[y] = 1;
                break;
            }
        }
        free(used);
        /* elimination, shortest columns first (static order), pivot: the shortest row whose entry
         * is at least 1% of the largest in the column */
        int *cnt = (int *)malloc(sizeof(int) * (size_t)(m0 + 2));
        for (int z = 0; z < npair; z++) {
            const int k = V[pa[z]].blk, i = V[pa[z]].i, k2 = V[pb[z]].blk, i2 = V[pb[z]].i;
            const uint64_t key = PKEY(k, i, i), key2 = PKEY(k2, i2, i2);
            int nc = 0, p = -1; double amax = 0;
            for (int c = 1; c <= m0; c++) {
                if (!alive[c]) continue;
                const int pos = prow_find(&rows[c], key);
                if (pos < 0) continue;
                cnt[nc++] = c;
                const double a = fabs(mpx_getd(rows[c].val + (size_t)pos * Wm));
                if (a > amax) amax = a;
            }
            if (!nc) continue;                                           /* only a cost: left as a pair */
            double fill = 0;
            for (int q = 0; q < nc; q++) {
                const int c = cnt[q];
                const double a = fabs(mpx_getd(rows[c].val + (size_t)prow_find(&rows[c], key) * Wm));
                if (a >= 0.01 * amax && (p < 0 || rows[c].n < rows[p].n)) p = c;
            }
            fill = (double)rows[p].n * nc;
            if (fill > 5e6) continue;
            /* the minus variable leaves every row (its entries are the negatives) */
            for (int c = 0; c <= m0; c++) { if (!alive[c]) continue; const int pos = prow_find(&rows[c], key2); if (pos >= 0) prow_del(&rows[c], pos); }
            PRec *R = pre_rec(P);
            R->kind = 1; R->blk = k; R->ip = i; R->bm = k2; R->im = i2; R->prow = p - 1;
            prow_copy(&R->row, &rows[p]);
            R->bp = (uint64_t *)malloc(sizeof(uint64_t) * Wm); R->apf = (uint64_t *)malloc(sizeof(uint64_t) * Wm); R->g = (uint64_t *)calloc(Wm, sizeof(uint64_t));
            mpx_copy(R->bp, b + (size_t)(p - 1) * Wm);
            mpx_copy(R->apf, rows[p].val + (size_t)prow_find(&rows[p], key) * Wm);
            R->lc = (int *)malloc(sizeof(int) * (size_t)(nc + 1)); R->lv = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(nc + 1) * Wm);
            for (int q = -1; q < nc; q++) {
                const int c = q < 0 ? 0 : cnt[q];
                if (c == p) continue;
                const int pos = prow_find(&rows[c], key);
                if (pos < 0) continue;
                mpx_div(t, rows[c].val + (size_t)pos * Wm, R->apf);
                prow_axpy(&rows[c], t, &rows[p], key, droptol);
                if (c == 0) mpx_copy(R->g, t);
                else {
                    mpx_mul(u, t, R->bp); mpx_sub(b + (size_t)(c - 1) * Wm, b + (size_t)(c - 1) * Wm, u);
                    R->lc[R->nl] = c - 1; mpx_copy(R->lv + (size_t)R->nl * Wm, t); R->nl++;
                }
            }
            alive[p] = 0; prow_free(&rows[p]);
            gone[k][i] = 1; gone[k2][i2] = 1;
            P->nfree++;
        }
        free(cnt); free(pa); free(pb); free(V);
        #undef PCOST
    }

    /* ---- 2. facial reduction: rows with b = 0 whose matrix is semidefinite of a simple form ---- */
    if (do_face) {
        int maxn = 1;
        for (int k = 0; k < nb0; k++) if (abs(J0->bs[k]) > maxn) maxn = abs(J0->bs[k]);
        int *supp = (int *)malloc(sizeof(int) * (size_t)(maxn + 1)), *pos = (int *)malloc(sizeof(int) * (size_t)(maxn + 1));
        int *Lidx = (int *)malloc(sizeof(int) * (size_t)(maxn + 1));
        uint64_t *rv = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(maxn + 1) * Wm), *Lval = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(maxn + 1) * Wm);
        int *touch = (int *)calloc((size_t)maxn + 1, sizeof(int));
        P->frow = (int *)malloc(sizeof(int) * (size_t)(m0 + 1));
        double bmax = 0;
        for (int c = 0; c < m0; c++) { const double d = fabs(mpx_getd(b + (size_t)c * Wm)); if (d > bmax) bmax = d; }
        int progress = 1;
        while (progress) {
            progress = 0;
            for (int c = 1; c <= m0; c++) {
                if (!alive[c] || rows[c].n == 0) continue;
                if (fabs(mpx_getd(b + (size_t)(c - 1) * Wm)) > droptol * 1e10 * (1 + bmax)) continue;
                const PRow *R = &rows[c];
                /* every block part: same-sign diagonal (SDP or LP) or rank one; one sign overall */
                int ok = 1, sg = 0, npart = 0, e0 = 0;
                int pk[64], pkind[64], pe0[64], pe1[64];
                while (e0 < R->n && ok) {
                    const int k = PKEY_BLK(R->key[e0]);
                    int e1 = e0;
                    while (e1 < R->n && PKEY_BLK(R->key[e1]) == k) e1++;
                    int diag = 1, s1 = 0;
                    for (int e = e0; e < e1; e++) {
                        const int i = PKEY_I(R->key[e]), j = PKEY_J(R->key[e]);
                        if (i != j) { diag = 0; continue; }
                        const int sv = mpx_sgn(R->val + (size_t)e * Wm);
                        if (s1 && sv != s1) ok = 0;
                        s1 = sv;
                    }
                    if (!ok || !s1 || (sg && s1 != sg)) { ok = 0; break; }
                    sg = s1;
                    int kind = 3;
                    if (!diag) {
                        /* rank one: the support S, all |S|(|S|+1)/2 entries present, A_ij A_pp = A_ip A_jp */
                        if (J0->bs[k] < 0) { ok = 0; break; }
                        int ns = 0;
                        for (int e = e0; e < e1; e++) { const int i = PKEY_I(R->key[e]), j = PKEY_J(R->key[e]); if (i == j) supp[ns++] = i; }
                        if ((e1 - e0) != ns * (ns + 1) / 2) { ok = 0; break; }
                        const int p0 = supp[0];
                        const int epp = prow_find(R, PKEY(k, p0, p0));
                        for (int e = e0; e < e1 && ok; e++) {
                            const int i = PKEY_I(R->key[e]), j = PKEY_J(R->key[e]);
                            const int eip = prow_find(R, PKEY(k, p0 < i ? p0 : i, p0 < i ? i : p0)), ejp = prow_find(R, PKEY(k, p0 < j ? p0 : j, p0 < j ? j : p0));
                            if (eip < 0 || ejp < 0) { ok = 0; break; }
                            mpx_mul(t, R->val + (size_t)e * Wm, R->val + (size_t)epp * Wm);
                            mpx_mul(u, R->val + (size_t)eip * Wm, R->val + (size_t)ejp * Wm);
                            const double big = fabs(mpx_getd(t)) + fabs(mpx_getd(u));
                            mpx_sub(t, t, u);
                            if (fabs(mpx_getd(t)) > droptol * 1e6 * big) ok = 0;
                        }
                        kind = 2;
                    }
                    if (npart == 64) { ok = 0; break; }
                    pk[npart] = k; pkind[npart] = kind; pe0[npart] = e0; pe1[npart] = e1; npart++;
                    e0 = e1;
                }
                if (!ok || !npart) continue;
                /* apply, part by part (the row itself is read from a copy) */
                PRow RC; prow_copy(&RC, R);
                for (int q = 0; q < npart; q++) {
                    const int k = pk[q];
                    PRec *Q = pre_rec(P);
                    Q->blk = k; Q->prow = c - 1; Q->sgn = sg;
                    if (pkind[q] == 3) {
                        Q->kind = 3;
                        Q->na = pe1[q] - pe0[q];
                        Q->ai = (int *)malloc(sizeof(int) * (size_t)(Q->na + 1));
                        for (int e = pe0[q]; e < pe1[q]; e++) { Q->ai[e - pe0[q]] = PKEY_I(RC.key[e]); gone[k][PKEY_I(RC.key[e])] = 1; }
                        /* the positions of the deleted indices leave every row */
                        for (int c2 = 0; c2 <= m0; c2++) {
                            if (!alive[c2] || c2 == c) continue;
                            PRow *T = &rows[c2];
                            int w = 0;
                            for (int e = 0; e < T->n; e++) {
                                const uint64_t key = T->key[e];
                                if (PKEY_BLK(key) == k && (gone[k][PKEY_I(key)] || gone[k][PKEY_J(key)])) continue;
                                if (w != e) { T->key[w] = key; memcpy(T->val + (size_t)w * Wm, T->val + (size_t)e * Wm, sizeof(uint64_t) * Wm); }
                                w++;
                            }
                            T->n = w;
                        }
                        continue;
                    }
                    /* rank one: support, pivot = the support index touched by the fewest entries of the other rows */
                    int ns = 0;
                    for (int e = pe0[q]; e < pe1[q]; e++) { const int i = PKEY_I(RC.key[e]), j = PKEY_J(RC.key[e]); if (i == j) supp[ns++] = i; }
                    for (int a = 0; a < ns; a++) touch[supp[a]] = 0;
                    const uint64_t klo = PKEY(k, 0, 0), khi = PKEY(k + 1, 0, 0);
                    for (int c2 = 0; c2 <= m0; c2++) {
                        if (!alive[c2] || c2 == c) continue;
                        const PRow *T = &rows[c2];
                        for (int e = 0; e < T->n; e++) { if (T->key[e] < klo || T->key[e] >= khi) continue; touch[PKEY_I(T->key[e])]++; if (PKEY_I(T->key[e]) != PKEY_J(T->key[e])) touch[PKEY_J(T->key[e])]++; }
                    }
                    int piv = supp[0];
                    for (int a = 1; a < ns; a++) if (touch[supp[a]] < touch[piv]) piv = supp[a];
                    for (int a = 0; a < maxn; a++) touch[a] = 0;
                    /* r_k = A_k,piv / A_piv,piv */
                    const int epp = prow_find(&RC, PKEY(k, piv, piv));
                    Q->kind = 2; Q->ip = piv; Q->na = 0;
                    Q->ai = (int *)malloc(sizeof(int) * (size_t)(ns + 1)); Q->av = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(ns + 1) * Wm);
                    for (int a = 0; a < ns; a++) {
                        const int i = supp[a];
                        if (i == piv) continue;
                        const int e = prow_find(&RC, PKEY(k, piv < i ? piv : i, piv < i ? i : piv));
                        mpx_div(Q->av + (size_t)Q->na * Wm, RC.val + (size_t)e * Wm, RC.val + (size_t)epp * Wm);
                        Q->ai[Q->na++] = i;
                    }
                    gone[k][piv] = 1;
                    /* every other row: M' = V' M V on the indices != piv */
                    for (int c2 = 0; c2 <= m0; c2++) {
                        if (!alive[c2] || c2 == c) continue;
                        PRow *T = &rows[c2];
                        /* the entries of row / column piv */
                        int nL = 0, have_pp = 0, w = 0;
                        uint64_t mpp[MPX_MAXL + 2];
                        for (int e = 0; e < T->n; e++) {
                            const uint64_t key = T->key[e];
                            const int i = PKEY_I(key), j = PKEY_J(key);
                            if (PKEY_BLK(key) == k && (i == piv || j == piv)) {
                                if (i == j) { have_pp = 1; mpx_copy(mpp, T->val + (size_t)e * Wm); }
                                else { Lidx[nL] = i == piv ? j : i; mpx_copy(Lval + (size_t)nL * Wm, T->val + (size_t)e * Wm); nL++; }
                                continue;
                            }
                            if (w != e) { T->key[w] = key; memcpy(T->val + (size_t)w * Wm, T->val + (size_t)e * Wm, sizeof(uint64_t) * Wm); }
                            w++;
                        }
                        T->n = w;
                        if (!nL && !have_pp) continue;
                        /* additions, gathered in a dense strip over (support' x indices) via a list */
                        PRow A; memset(&A, 0, sizeof(A));
                        for (int a = 0; a < Q->na; a++) {
                            const int kk = Q->ai[a];
                            const uint64_t *rk = Q->av + (size_t)a * Wm;
                            for (int z2 = 0; z2 < nL; z2++) {
                                const int l = Lidx[z2];
                                mpx_mul(t, rk, Lval + (size_t)z2 * Wm);
                                if (kk == l) mpx_add(t, t, t);
                                mpx_neg(t, t);
                                prow_push(&A, PKEY(k, kk < l ? kk : l, kk < l ? l : kk), t);
                            }
                            if (have_pp) for (int a2 = a; a2 < Q->na; a2++) {
                                mpx_mul(t, rk, Q->av + (size_t)a2 * Wm); mpx_mul(t, t, mpp);
                                prow_push(&A, PKEY(k, kk < Q->ai[a2] ? kk : Q->ai[a2], kk < Q->ai[a2] ? Q->ai[a2] : kk), t);
                            }
                        }
                        /* sort by key and combine equal keys (insertion order index sort) */
                        if (A.n > 1) {
                            int *ord = (int *)malloc(sizeof(int) * (size_t)A.n);
                            for (int e = 0; e < A.n; e++) ord[e] = e;
                            g_pre_keys = A.key;
                            qsort(ord, (size_t)A.n, sizeof(int), cmp_pre_ord);
                            PRow B2; memset(&B2, 0, sizeof(B2));
                            for (int e = 0; e < A.n; e++) {
                                const int o = ord[e];
                                if (B2.n && B2.key[B2.n - 1] == A.key[o]) mpx_add(B2.val + (size_t)(B2.n - 1) * Wm, B2.val + (size_t)(B2.n - 1) * Wm, A.val + (size_t)o * Wm);
                                else prow_push(&B2, A.key[o], A.val + (size_t)o * Wm);
                            }
                            free(ord); prow_free(&A); A = B2;
                        }
                        mpx_setd(t, -1.0);
                        prow_axpy(T, t, &A, UINT64_MAX, droptol);        /* T <- T + A */
                        prow_free(&A);
                    }
                }
                prow_free(&RC);
                alive[c] = 0; prow_free(&rows[c]);
                P->frow[P->nfrow++] = c - 1;
                P->nface++;
                progress = 1;
            }
        }
        free(supp); free(pos); free(Lidx); free(rv); free(Lval); free(touch);
    }

    /* rows that became empty: 0 = b. With b = 0 the row is dropped (it was dependent). */
    for (int c = 1; c <= m0; c++)
        if (alive[c] && rows[c].n == 0 && (P->nfree || P->nface)) {
            const double bd = fabs(mpx_getd(b + (size_t)(c - 1) * Wm));
            if (bd <= droptol * 1e10) { alive[c] = 0; P->ndrop++; }
        }
    if (!P->nfree && !P->nface) {
        for (int c = 0; c <= m0; c++) prow_free(&rows[c]);
        free(rows); free(b); free(alive);
        for (int k = 0; k < nb0; k++) free(gone[k]);
        free(gone); hp_pre_free(P, nb0);
        return NULL;
    }
    /* ---- the reduced problem ---- */
    P->blkmap = (int *)malloc(sizeof(int) * (size_t)nb0);
    P->idxmap = (int **)malloc(sizeof(int *) * (size_t)nb0);
    memset(Jr, 0, sizeof(*Jr));
    Jr->Lm = J0->Lm;
    Jr->bs = (int *)malloc(sizeof(int) * (size_t)nb0);
    int nbr = 0;
    for (int k = 0; k < nb0; k++) {
        const int n = abs(J0->bs[k]);
        P->idxmap[k] = (int *)malloc(sizeof(int) * (size_t)(n + 1));
        int w = 0;
        for (int i = 0; i < n; i++) P->idxmap[k][i] = gone[k][i] ? -1 : w++;
        if (w == 0) { P->blkmap[k] = -1; continue; }
        P->blkmap[k] = nbr; Jr->bs[nbr++] = J0->bs[k] < 0 ? -w : w;
    }
    Jr->nblk = nbr;
    P->rowmap = (int *)malloc(sizeof(int) * (size_t)(m0 + 1));
    int mr = 0; size_t ne = 0;
    for (int c = 1; c <= m0; c++) if (alive[c]) { P->rowmap[mr++] = c - 1; ne += (size_t)rows[c].n; }
    ne += (size_t)rows[0].n;
    P->mr = mr; Jr->m = mr;
    Jr->b = (uint64_t *)calloc((size_t)(mr ? mr : 1) * Wm, sizeof(uint64_t));
    HEnt *E = (HEnt *)malloc(sizeof(HEnt) * (ne + 1));
    uint64_t *vals = (uint64_t *)malloc(sizeof(uint64_t) * (ne + 1) * Wm);
    size_t q = 0;
    for (int r = -1; r < mr; r++) {
        const PRow *R = r < 0 ? &rows[0] : &rows[P->rowmap[r] + 1];
        if (r >= 0) mpx_copy(Jr->b + (size_t)r * Wm, b + (size_t)P->rowmap[r] * Wm);
        for (int e = 0; e < R->n; e++) {
            const int k = PKEY_BLK(R->key[e]), i = PKEY_I(R->key[e]), j = PKEY_J(R->key[e]);
            if (P->blkmap[k] < 0 || P->idxmap[k][i] < 0 || P->idxmap[k][j] < 0) continue;      /* (a removed position: cannot occur) */
            E[q].con = r; E[q].blk = P->blkmap[k]; E[q].i = P->idxmap[k][i]; E[q].j = P->idxmap[k][j]; E[q].idx = q;
            memcpy(vals + q * Wm, R->val + (size_t)e * Wm, sizeof(uint64_t) * Wm);
            q++;
        }
    }
    const int rc = mr > 0 && nbr > 0 ? hp_build(Jr, E, q, vals) : 1;
    free(E); free(vals);
    for (int c = 0; c <= m0; c++) prow_free(&rows[c]);
    free(rows); free(b); free(alive);
    for (int k = 0; k < nb0; k++) free(gone[k]);
    free(gone);
    if (rc) { hp_pre_free(P, nb0); return NULL; }
    if (verbose) {
        printf("presolve (in %d bits):", 64 * (J0->Lm - 1));
        if (P->nfree) printf(" %d free variable(s) eliminated;", P->nfree);
        if (P->nface) printf(" %d face reduction(s);", P->nface);
        if (P->ndrop) printf(" %d dependent constraint(s) dropped;", P->ndrop);
        printf(" m %d -> %d, nonzeros %zu -> %zu\n", m0, mr, J0->nnzA + J0->nnzC, Jr->nnzA + Jr->nnzC);
    }
    return P;
}

/* <row, X> for a row over positions (X: per original block, dense or LP), mpx */
static void pre_rowdot(const HPJob *J0, const PRow *R, uint64_t *const *X, uint64_t skip, uint64_t *res) {
    const size_t Wm = (size_t)mpx_W;
    uint64_t t[MPX_MAXL + 2];
    mpx_setd(res, 0.0);
    for (int e = 0; e < R->n; e++) {
        if (R->key[e] == skip) continue;
        const int k = PKEY_BLK(R->key[e]), i = PKEY_I(R->key[e]), j = PKEY_J(R->key[e]);
        const int n = abs(J0->bs[k]);
        const uint64_t *x = J0->bs[k] < 0 ? X[k] + (size_t)i * Wm : X[k] + ((size_t)i * n + j) * Wm;
        mpx_mul(t, R->val + (size_t)e * Wm, x);
        if (J0->bs[k] > 0 && i != j) mpx_add(t, t, t);
        mpx_add(res, res, t);
    }
}

/* The solution of the original problem from the one of the reduced problem. Xr, yr: reduced
 * (blocks as Jr). X0 (allocated, per original block) and y0 (m0) are filled. */
static void hp_postsolve(const HPPre *P, const HPJob *J0, const HPJob *Jr, uint64_t *const *Xr, const uint64_t *yr, uint64_t **X0, uint64_t *y0, double tface) {
    const size_t Wm = (size_t)mpx_W;
    const int nb0 = J0->nblk, m0 = J0->m;
    uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2];
    /* X: positions that survived */
    for (int k = 0; k < nb0; k++) {
        const int n = abs(J0->bs[k]), kr = P->blkmap[k];
        const size_t len = J0->bs[k] < 0 ? (size_t)n : (size_t)n * n;
        memset(X0[k], 0, len * Wm * sizeof(uint64_t));
        if (kr < 0) continue;
        const int nr = abs(Jr->bs[kr]);
        if (J0->bs[k] < 0) { for (int i = 0; i < n; i++) if (P->idxmap[k][i] >= 0) mpx_copy(X0[k] + (size_t)i * Wm, Xr[kr] + (size_t)P->idxmap[k][i] * Wm); continue; }
        for (int i = 0; i < n; i++) {
            if (P->idxmap[k][i] < 0) continue;
            for (int j = 0; j < n; j++) if (P->idxmap[k][j] >= 0) mpx_copy(X0[k] + ((size_t)i * n + j) * Wm, Xr[kr] + ((size_t)P->idxmap[k][i] * nr + P->idxmap[k][j]) * Wm);
        }
    }
    /* y on the rows that survived */
    memset(y0, 0, (size_t)m0 * Wm * sizeof(uint64_t));
    for (int r = 0; r < P->mr; r++) mpx_copy(y0 + (size_t)P->rowmap[r] * Wm, yr + (size_t)r * Wm);
    /* the records, last first */
    for (int z = P->nrec - 1; z >= 0; z--) {
        const PRec *R = &P->rec[z];
        if (R->kind == 2) {
            /* X = V W V': row and column piv from W (the other indices) and r */
            const int k = R->blk, n = abs(J0->bs[k]), piv = R->ip;
            uint64_t *X = X0[k], acc[MPX_MAXL + 2];
            for (int l = 0; l < n; l++) {
                if (l == piv) continue;
                mpx_setd(acc, 0.0);
                for (int a = 0; a < R->na; a++) { mpx_mul(u, R->av + (size_t)a * Wm, X + ((size_t)R->ai[a] * n + l) * Wm); mpx_sub(acc, acc, u); }
                mpx_copy(X + ((size_t)piv * n + l) * Wm, acc); mpx_copy(X + ((size_t)l * n + piv) * Wm, acc);
            }
            mpx_setd(acc, 0.0);
            for (int a = 0; a < R->na; a++) { mpx_mul(u, R->av + (size_t)a * Wm, X + ((size_t)piv * n + R->ai[a]) * Wm); mpx_sub(acc, acc, u); }
            mpx_copy(X + ((size_t)piv * n + piv) * Wm, acc);
            mpx_setd(t, -R->sgn * tface); mpx_copy(y0 + (size_t)R->prow * Wm, t);
        } else if (R->kind == 3) {
            mpx_setd(t, -R->sgn * tface); mpx_copy(y0 + (size_t)R->prow * Wm, t);
        }
        if (R->kind == 1) {
            /* f = (b_p - <row p without f, X>) / a_pf, kept (signed) at the plus position for
             * the rows of earlier records, which contain it */
            const uint64_t key = PKEY(R->blk, R->ip, R->ip);
            pre_rowdot(J0, &R->row, X0, key, t);
            mpx_sub(t, R->bp, t); mpx_div(u, t, R->apf);
            mpx_copy(X0[R->blk] + (size_t)R->ip * Wm, u);
            /* w_p = g - sum_c l_c w_c */
            mpx_copy(t, R->g);
            for (int q = 0; q < R->nl; q++) { mpx_mul(u, R->lv + (size_t)q * Wm, y0 + (size_t)R->lc[q] * Wm); mpx_sub(t, t, u); }
            mpx_copy(y0 + (size_t)R->prow * Wm, t);
        }
    }
    /* the free values as pairs: x+ = max(f, 0), x- = max(-f, 0) */
    for (int z = 0; z < P->nrec; z++) {
        const PRec *R = &P->rec[z];
        if (R->kind != 1) continue;
        uint64_t *xp = X0[R->blk] + (size_t)R->ip * Wm, *xm = X0[R->bm] + (size_t)R->im * Wm;
        if (mpx_sgn(xp) < 0) { mpx_neg(xm, xp); mpx_setd(xp, 0.0); } else mpx_setd(xm, 0.0);
    }
}

/* The errors of (X, y) on the problem J (mpx). Z = C - A*(y) is formed into Z (allocated).
 * dinf is the smallest shift d of a grid for which Z + d I has a Cholesky factor (LP: the most
 * negative entry), relative to 1 + |C|; tolgoal sets the start of the grid. */
static void hp_measure(HPJob *J, uint64_t *const *X, const uint64_t *y, uint64_t **Z, double tolgoal) {
    const size_t Wm = (size_t)mpx_W;
    const int nb = J->nblk, m = J->m;
    uint64_t t[MPX_MAXL + 2], u[MPX_MAXL + 2], acc[MPX_MAXL + 2], pobj[MPX_MAXL + 2], dobj[MPX_MAXL + 2], xz[MPX_MAXL + 2], nb2[MPX_MAXL + 2], nc2[MPX_MAXL + 2], rp2[MPX_MAXL + 2];
    uint64_t *ax = (uint64_t *)calloc((size_t)m * Wm, sizeof(uint64_t));
    mpx_setd(pobj, 0.0); mpx_setd(dobj, 0.0); mpx_setd(xz, 0.0); mpx_setd(nb2, 0.0); mpx_setd(nc2, 0.0); mpx_setd(rp2, 0.0);
    for (int k = 0; k < nb; k++) {
        const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
        const size_t len = lp ? (size_t)n : (size_t)n * n;
        memset(Z[k], 0, len * Wm * sizeof(uint64_t));
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
            const int i = J->ci[e], j = lp ? i : J->cj[e];
            const uint64_t *v = J->Cv + e * Wm;
            mpx_mul(t, v, v); if (!lp && i != j) mpx_add(t, t, t); mpx_add(nc2, nc2, t);
            if (lp) { mpx_copy(Z[k] + (size_t)i * Wm, v); mpx_mul(t, v, X[k] + (size_t)i * Wm); }
            else {
                mpx_copy(Z[k] + ((size_t)i * n + j) * Wm, v); mpx_copy(Z[k] + ((size_t)j * n + i) * Wm, v);
                mpx_mul(t, v, X[k] + ((size_t)i * n + j) * Wm); if (i != j) mpx_add(t, t, t);
            }
            mpx_add(pobj, pobj, t);
        }
        for (int tt = 0; tt < J->ncon[k]; tt++) {
            const int c = J->con[k][tt];
            const uint64_t *yc = y + (size_t)c * Wm;
            mpx_setd(acc, 0.0);
            for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                const int i = J->ei[e], j = lp ? i : J->ej[e];
                const uint64_t *v = J->Av + e * Wm;
                if (lp) { mpx_mul(t, v, X[k] + (size_t)i * Wm); mpx_submul(Z[k] + (size_t)i * Wm, yc, v); }
                else {
                    mpx_mul(t, v, X[k] + ((size_t)i * n + j) * Wm); if (i != j) mpx_add(t, t, t);
                    mpx_submul(Z[k] + ((size_t)i * n + j) * Wm, yc, v);
                    if (i != j) mpx_submul(Z[k] + ((size_t)j * n + i) * Wm, yc, v);
                }
                mpx_add(acc, acc, t);
            }
            mpx_add(ax + (size_t)c * Wm, ax + (size_t)c * Wm, acc);
        }
        for (size_t i = 0; i < len; i++) { mpx_mul(t, X[k] + i * Wm, Z[k] + i * Wm); mpx_add(xz, xz, t); }
    }
    for (int c = 0; c < m; c++) {
        const uint64_t *bc = J->b + (size_t)c * Wm;
        mpx_sub(t, bc, ax + (size_t)c * Wm); mpx_mul(u, t, t); mpx_add(rp2, rp2, u);
        mpx_mul(u, bc, bc); mpx_add(nb2, nb2, u);
        mpx_mul(u, bc, y + (size_t)c * Wm); mpx_add(dobj, dobj, u);
    }
    free(ax);
    mpx_sqrt(t, nb2); const double nbn = mpx_getd(t);
    mpx_sqrt(t, nc2); const double ncn = mpx_getd(t);
    mpx_sqrt(t, rp2); const double pinf = mpx_getd(t) / (1 + nbn);
    /* dual infeasibility: Cholesky of Z + d I */
    double dinf = 0;
    int nmax = 1;
    if (tolgoal < 0) nmax = 0;                      /* (the dual check is not wanted) */
    for (int k = 0; k < nb && tolgoal >= 0; k++) if (J->bs[k] > nmax) nmax = J->bs[k];
    uint64_t *W = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)nmax * nmax + 1) * Wm);
    for (int k = 0; k < nb && tolgoal >= 0; k++) {
        const int n = abs(J->bs[k]);
        if (J->bs[k] < 0) { for (int i = 0; i < n; i++) { const double z = mpx_getd(Z[k] + (size_t)i * Wm); if (-z / (1 + ncn) > dinf) dinf = -z / (1 + ncn); } continue; }
        double d = 0;
        for (int tries = 0; tries < 40; tries++) {
            memcpy(W, Z[k], sizeof(uint64_t) * (size_t)n * n * Wm);
            if (d > 0) { mpx_setd(u, d * (1 + ncn)); for (int i = 0; i < n; i++) mpx_add(W + ((size_t)i * n + i) * Wm, W + ((size_t)i * n + i) * Wm, u); }
            if (cholr_mp(n, W, 0.0, NULL, NULL, 0, NULL, NULL) == 0) break;
            d = d > 0 ? d * (tries < 30 ? 4.0 : 1e3) : 1e-3 * tolgoal;
            if (tries == 39) d = 1.0;
        }
        if (d > dinf) dinf = d;
    }
    free(W);
    const double po = mpx_getd(pobj), dob = mpx_getd(dobj), den = 1 + fabs(po) + fabs(dob);
    mpx_sub(t, pobj, dobj);
    J->best_pinf = pinf; J->best_dinf = dinf; J->best_gap = mpx_getd(t) / den; J->best_compl = mpx_getd(xz) / den;
    J->best_err = fmax(fmax(pinf, dinf), fmax(fabs(J->best_gap), fabs(J->best_compl)));
    mpx_copy(J->pobj, pobj); mpx_copy(J->dobj, dobj);
}

/* The dual form. (D) max b'y, Z = C - sum y_p A_p >= 0 is written as a problem in the standard
 * form on Z with free variables y: one row per position (k, i <= j) of the cone,
 *     Z_ij + sum_p y_p (A_p)_ij = C_ij,      min -b'y,
 * the y as split pairs in a last LP block (the presolve eliminates them, which leaves
 * (positions - m) rows). Only copies, signs and the factor 1/2 are involved: the form is exact.
 * Its solution gives the original one: X = the dual slack of the new problem, y = y+ - y-,
 * Z = its matrix variable. Worth it when m is close to the number of positions (image-form
 * relaxations: m = 865 becomes 379). Returns 0 on success. */
static int hp_dualize(const HPJob *J0, HPJob *Jd) {
    const size_t Wm = (size_t)J0->Lm + 2;
    const int nb0 = J0->nblk, m0 = J0->m;
    memset(Jd, 0, sizeof(*Jd));
    Jd->Lm = J0->Lm;
    size_t nrow = 0;
    size_t *off = (size_t *)malloc(sizeof(size_t) * (size_t)(nb0 + 1));
    for (int k = 0; k < nb0; k++) { off[k] = nrow; const size_t n = (size_t)abs(J0->bs[k]); nrow += J0->bs[k] < 0 ? n : n * (n + 1) / 2; }
    off[nb0] = nrow;
    if (nrow > 2000000000u) { free(off); return 1; }
    #define DROW(k, n, i, j) (off[k] + (J0->bs[k] < 0 ? (size_t)(i) : (size_t)(i) * (size_t)(n) - (size_t)(i) * ((size_t)(i) - 1) / 2 + (size_t)((j) - (i))))
    Jd->m = (int)nrow; Jd->nblk = nb0 + 1;
    Jd->bs = (int *)malloc(sizeof(int) * (size_t)(nb0 + 1));
    memcpy(Jd->bs, J0->bs, sizeof(int) * (size_t)nb0); Jd->bs[nb0] = -2 * m0;
    Jd->b = (uint64_t *)calloc(nrow * Wm, sizeof(uint64_t));
    const size_t ne = nrow + 2 * J0->nnzA + 2 * (size_t)m0;
    HEnt *E = (HEnt *)malloc(sizeof(HEnt) * (ne + 1));
    uint64_t *vals = (uint64_t *)malloc(sizeof(uint64_t) * (ne + 1) * Wm);
    size_t q = 0;
    for (int k = 0; k < nb0; k++) {
        const int n = abs(J0->bs[k]), lp = J0->bs[k] < 0;
        for (int i = 0; i < n; i++)
            for (int j = i; j < (lp ? i + 1 : n); j++) {
                E[q].con = (int)DROW(k, n, i, j); E[q].blk = k; E[q].i = i; E[q].j = j; E[q].idx = q;
                mpx_setd(vals + q * Wm, i == j ? 1.0 : 0.5); q++;
            }
        for (size_t e = J0->cptr[k]; e < J0->cptr[k + 1]; e++) {
            const int i = J0->ci[e], j = lp ? i : J0->cj[e];
            mpx_copy(Jd->b + DROW(k, n, i, j) * Wm, J0->Cv + e * Wm);
        }
        for (int t = 0; t < J0->ncon[k]; t++) {
            const int pcon = J0->con[k][t];
            for (size_t e = J0->ptr[k][t]; e < J0->ptr[k][t + 1]; e++) {
                const int i = J0->ei[e], j = lp ? i : J0->ej[e];
                const int r = (int)DROW(k, n, i, j);
                E[q].con = r; E[q].blk = nb0; E[q].i = E[q].j = 2 * pcon; E[q].idx = q; mpx_copy(vals + q * Wm, J0->Av + e * Wm); q++;
                E[q].con = r; E[q].blk = nb0; E[q].i = E[q].j = 2 * pcon + 1; E[q].idx = q; mpx_neg(vals + q * Wm, J0->Av + e * Wm); q++;
            }
        }
    }
    for (int pcon = 0; pcon < m0; pcon++) {
        const uint64_t *bp = J0->b + (size_t)pcon * Wm;
        if (mpx_sgn(bp) == 0) continue;
        E[q].con = -1; E[q].blk = nb0; E[q].i = E[q].j = 2 * pcon; E[q].idx = q; mpx_neg(vals + q * Wm, bp); q++;
        E[q].con = -1; E[q].blk = nb0; E[q].i = E[q].j = 2 * pcon + 1; E[q].idx = q; mpx_copy(vals + q * Wm, bp); q++;
    }
    #undef DROW
    free(off);
    const int rc = hp_build(Jd, E, q, vals);
    free(E); free(vals);
    return rc;
}

/* ---- 5.3: blocks split by their aggregate sparsity pattern ----
 * An SDP block whose pattern (C and all A_p together) is block diagonal after a permutation is
 * replaced by its diagonal blocks; components of one index become LP variables (one LP block
 * per original block), indices with no entry at all are dropped. No arithmetic is involved:
 * the split problem has the same y and Z, and a solution X of it gives the block-diagonal
 * solution of the problem as read. (SDPLIB qpG: a block of 1600 = one of 800 and 800 LP
 * variables.) */
typedef struct {
    int nb0, *bs0;             /* the blocks as read */
    int **nblk, **nidx;        /* per original block and index: the new block and the index in it (-1: dropped) */
    int nsplit, nlp, ndrop;    /* blocks that were split, LP variables made, indices dropped */
} HPSplit;
static void hp_split_free(HPSplit *sp) {
    for (int k = 0; k < sp->nb0; k++) { if (sp->nblk) free(sp->nblk[k]); if (sp->nidx) free(sp->nidx[k]); }
    free(sp->nblk); free(sp->nidx); free(sp->bs0);
    memset(sp, 0, sizeof(*sp));
}
static int split_find(int *p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
/* returns 1 and fills Js (built) and sp when some block splits */
static int hp_split(const HPJob *J0, HPJob *Js, HPSplit *sp) {
    const size_t Wm = (size_t)J0->Lm + 2;
    const int nb0 = J0->nblk;
    memset(sp, 0, sizeof(*sp));
    sp->nb0 = nb0;
    sp->bs0 = (int *)malloc(sizeof(int) * (size_t)(nb0 + 1)); memcpy(sp->bs0, J0->bs, sizeof(int) * (size_t)nb0);
    sp->nblk = (int **)calloc((size_t)nb0 + 1, sizeof(int *)); sp->nidx = (int **)calloc((size_t)nb0 + 1, sizeof(int *));
    int nbn = 0, any = 0, cap = nb0 + 8;
    int *bsn = (int *)malloc(sizeof(int) * (size_t)cap);
    for (int k = 0; k < nb0; k++) {
        const int n = abs(J0->bs[k]);
        sp->nblk[k] = (int *)malloc(sizeof(int) * (size_t)n); sp->nidx[k] = (int *)malloc(sizeof(int) * (size_t)n);
        if (J0->bs[k] < 0) {
            if (nbn + 2 > cap) { cap *= 2; bsn = (int *)realloc(bsn, sizeof(int) * (size_t)cap); }
            for (int i = 0; i < n; i++) { sp->nblk[k][i] = nbn; sp->nidx[k][i] = i; }
            bsn[nbn++] = -n;
            continue;
        }
        int *par = (int *)malloc(sizeof(int) * (size_t)n), *size = (int *)calloc((size_t)n, sizeof(int));
        char *has = (char *)calloc((size_t)n + 1, 1);
        for (int i = 0; i < n; i++) par[i] = i;
        for (size_t e = J0->cptr[k]; e < J0->cptr[k + 1]; e++) { const int a = split_find(par, J0->ci[e]), b = split_find(par, J0->cj[e]); has[J0->ci[e]] = has[J0->cj[e]] = 1; if (a != b) par[a] = b; }
        for (size_t e = J0->ptr[k][0]; J0->ncon[k] > 0 && e < J0->ptr[k][J0->ncon[k]]; e++) { const int a = split_find(par, J0->ei[e]), b = split_find(par, J0->ej[e]); has[J0->ei[e]] = has[J0->ej[e]] = 1; if (a != b) par[a] = b; }
        for (int i = 0; i < n; i++) size[split_find(par, i)]++;
        int ncomp = 0, nsing = 0, nnone = 0;
        for (int i = 0; i < n; i++) if (par[i] == i) { if (size[i] > 1) ncomp++; else if (has[i]) nsing++; else nnone++; }
        if (ncomp == 1 && nsing == 0 && nnone == 0) {
            if (nbn + 2 > cap) { cap *= 2; bsn = (int *)realloc(bsn, sizeof(int) * (size_t)cap); }
            for (int i = 0; i < n; i++) { sp->nblk[k][i] = nbn; sp->nidx[k][i] = i; }
            bsn[nbn++] = n;
            free(par); free(size); free(has);
            continue;
        }
        any = 1; sp->nsplit++; sp->nlp += nsing; sp->ndrop += nnone;
        if (nbn + ncomp + 2 > cap) { cap = 2 * cap + ncomp + 2; bsn = (int *)realloc(bsn, sizeof(int) * (size_t)cap); }
        /* the components of two or more indices, in the order of their first index */
        int *cb = (int *)malloc(sizeof(int) * (size_t)n), *cn = (int *)calloc((size_t)n, sizeof(int));
        for (int i = 0; i < n; i++) cb[i] = -1;
        for (int i = 0; i < n; i++) {
            const int r = split_find(par, i);
            if (size[r] <= 1) continue;
            if (cb[r] < 0) { cb[r] = nbn; bsn[nbn++] = size[r]; }
            sp->nblk[k][i] = cb[r]; sp->nidx[k][i] = cn[r]++;
        }
        int nl = 0;
        for (int i = 0; i < n; i++) {
            const int r = split_find(par, i);
            if (size[r] > 1) continue;
            if (!has[i]) { sp->nblk[k][i] = -1; sp->nidx[k][i] = -1; continue; }
            sp->nblk[k][i] = nbn; sp->nidx[k][i] = nl++;
        }
        if (nl > 0) bsn[nbn++] = -nl;
        free(cb); free(cn); free(par); free(size); free(has);
    }
    if (!any || nbn == 0) { free(bsn); hp_split_free(sp); return 0; }
    memset(Js, 0, sizeof(*Js));
    Js->Lm = J0->Lm; Js->m = J0->m; Js->nblk = nbn; Js->bs = bsn;
    Js->b = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)J0->m + 1) * Wm);
    memcpy(Js->b, J0->b, sizeof(uint64_t) * (size_t)J0->m * Wm);
    const size_t ne = J0->nnzA + J0->nnzC;
    HEnt *E = (HEnt *)malloc(sizeof(HEnt) * (ne + 1));
    uint64_t *vals = (uint64_t *)malloc(sizeof(uint64_t) * (ne + 1) * Wm);
    size_t w = 0;
    for (int k = 0; k < nb0; k++) {
        const int lp = J0->bs[k] < 0;
        for (size_t e = J0->cptr[k]; e < J0->cptr[k + 1]; e++) {
            const int i = J0->ci[e], j = lp ? i : J0->cj[e];
            E[w].con = -1; E[w].blk = sp->nblk[k][i]; E[w].i = sp->nidx[k][i]; E[w].j = sp->nidx[k][j]; E[w].idx = w;
            memcpy(vals + w * Wm, J0->Cv + e * Wm, Wm * sizeof(uint64_t)); w++;
        }
        for (int tt = 0; tt < J0->ncon[k]; tt++)
            for (size_t e = J0->ptr[k][tt]; e < J0->ptr[k][tt + 1]; e++) {
                const int i = J0->ei[e], j = lp ? i : J0->ej[e];
                E[w].con = J0->con[k][tt]; E[w].blk = sp->nblk[k][i]; E[w].i = sp->nidx[k][i]; E[w].j = sp->nidx[k][j]; E[w].idx = w;
                memcpy(vals + w * Wm, J0->Av + e * Wm, Wm * sizeof(uint64_t)); w++;
            }
    }
    const int rc = hp_build(Js, E, w, vals);
    free(E); free(vals);
    if (rc) { hp_free(Js); hp_split_free(sp); return 0; }
    return 1;
}
/* a double matrix set of the problem as read (blocks n x n or LP vectors) in the split layout */
static double **hp_split_fwd_d(const HPSplit *sp, const HPJob *Js, double *const *X0) {
    double **X = (double **)calloc((size_t)Js->nblk + 1, sizeof(double *));
    for (int k = 0; k < Js->nblk; k++) X[k] = (double *)calloc((Js->bs[k] < 0 ? (size_t)(-Js->bs[k]) : (size_t)Js->bs[k] * Js->bs[k]) + 1, sizeof(double));
    for (int k = 0; k < sp->nb0; k++) {
        const int n = abs(sp->bs0[k]);
        if (sp->bs0[k] < 0) { for (int i = 0; i < n; i++) X[sp->nblk[k][i]][sp->nidx[k][i]] = X0[k][i]; continue; }
        for (int i = 0; i < n; i++) {
            const int kn = sp->nblk[k][i];
            if (kn < 0) continue;
            if (Js->bs[kn] < 0) { X[kn][sp->nidx[k][i]] = X0[k][(size_t)i * n + i]; continue; }
            const int nn = Js->bs[kn];
            for (int j = 0; j < n; j++) if (sp->nblk[k][j] == kn) X[kn][(size_t)sp->nidx[k][i] * nn + sp->nidx[k][j]] = X0[k][(size_t)i * n + j];
        }
    }
    return X;
}
/* an mpx matrix set of the split problem in the layout as read (allocated here) */
static uint64_t **hp_split_back(const HPSplit *sp, const HPJob *Js, uint64_t *const *Xs) {
    const size_t Wm = (size_t)Js->Lm + 2;
    uint64_t **X0 = (uint64_t **)calloc((size_t)sp->nb0 + 1, sizeof(uint64_t *));
    for (int k = 0; k < sp->nb0; k++) {
        const int n = abs(sp->bs0[k]);
        X0[k] = (uint64_t *)calloc(((sp->bs0[k] < 0 ? (size_t)n : (size_t)n * n) + 1) * Wm, sizeof(uint64_t));
        if (sp->bs0[k] < 0) { for (int i = 0; i < n; i++) memcpy(X0[k] + (size_t)i * Wm, Xs[sp->nblk[k][i]] + (size_t)sp->nidx[k][i] * Wm, Wm * sizeof(uint64_t)); continue; }
        for (int i = 0; i < n; i++) {
            const int kn = sp->nblk[k][i];
            if (kn < 0) continue;
            if (Js->bs[kn] < 0) { memcpy(X0[k] + ((size_t)i * n + i) * Wm, Xs[kn] + (size_t)sp->nidx[k][i] * Wm, Wm * sizeof(uint64_t)); continue; }
            const int nn = Js->bs[kn];
            for (int j = 0; j < n; j++) if (sp->nblk[k][j] == kn) memcpy(X0[k] + ((size_t)i * n + j) * Wm, Xs[kn] + ((size_t)sp->nidx[k][i] * nn + sp->nidx[k][j]) * Wm, Wm * sizeof(uint64_t));
        }
    }
    return X0;
}
