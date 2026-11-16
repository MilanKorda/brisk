/* rowspace.c: the equality constraints in reduced row echelon form, so that the sign-symmetry search sees the
 * symmetries that recombined rows hide (16 November 2026).
 *
 * A sign symmetry is a diagonal matrix D = diag(+-1) on the positions of the variables (entries of the semidefinite
 * blocks, the LP variables) under which the problem is invariant: D C = C and the row space of the constraint matrix A
 * is mapped onto itself, with b following. The search of symred.c (sign_reduce) asks more: each row a_i of A must be
 * even or odd under D (D a_i = +a_i or -a_i). A row space can be invariant without that: a row that is the sum of an
 * even and an odd row, or a dependent row that is such a sum, breaks the row-wise property, and the search finds
 * nothing.
 *
 * The reduced row echelon form E of A (the columns in a fixed order) is unique for its row space, and in it every
 * symmetry of the row space is row-wise: E D has the same row space as E, and dividing each row of E D by the sign
 * of its pivot column gives a reduced row echelon form of that space with the same pivot columns, hence E itself; so
 * each row e_q satisfies e_q D = +-e_q. The constraints E X = T b (T the combination with E = T A on the
 * independent rows, the dependent rows dropped) have the same solutions as A X = b, and a dual y_E of them gives the
 * dual y = T' y_E of the original rows (A' y = E' y_E: the same dual slack).
 *
 * The form is computed by sparse Gauss-Jordan elimination over the positions in increasing order, with the pivot
 * of largest magnitude relative to its row (scaled partial pivoting) among the rows not yet pivots, every other row
 * eliminated in the pivot column (also the earlier pivot rows), and T carried as further columns of the rows. The
 * elimination stops (nothing applied) when its work or its fill passes a budget proportional to the entries of A. */
#include "brisk.h"

struct PSRref {
    int m0, m1;          /* the rows of the problem before and after */
    int *tp, *ti;        /* T by rows of the new problem: entries tp[q] .. tp[q + 1] - 1, rows ti of the problem before */
    double *tv;
};

typedef struct { int len, cap; int *ix; double *v; } RRow;

static void rr_reserve(RRow *r, int n) {
    if (n <= r->cap) return;
    int c = r->cap ? r->cap : 4;
    while (c < n) c *= 2;
    r->ix = realloc(r->ix, sizeof(int) * (size_t)c);
    r->v = realloc(r->v, sizeof(double) * (size_t)c);
    r->cap = c;
}
static double rr_get(const RRow *r, int c, int *pos) {   /* the row's value in column c (0 when absent) */
    int lo = 0, hi = r->len - 1;
    while (lo <= hi) { const int mid = (lo + hi) >> 1; if (r->ix[mid] < c) lo = mid + 1; else if (r->ix[mid] > c) hi = mid - 1; else { if (pos) *pos = mid; return r->v[mid]; } }
    if (pos) *pos = -1;
    return 0.0;
}

typedef struct { size_t g; int row; double v; } RREnt;
static int rrent_cmp(const void *a, const void *b) {
    const RREnt *x = a, *y = b;
    if (x->g != y->g) return x->g < y->g ? -1 : 1;
    return (x->row > y->row) - (x->row < y->row);
}

static int rref_envi(const char *name, int dflt) { const char *e = getenv(name); return e ? atoi(e) : dflt; }

void rref_free(PSRref *R) { if (!R) return; free(R->tp); free(R->ti); free(R->tv); free(R); }

int rref_rows(const PSRref *R, int *m0) { if (m0) *m0 = R ? R->m0 : 0; return R ? R->m1 : 0; }

void rref_unmap_y(const PSRref *R, const double *yq, double *yp) {
    for (int i = 0; i < R->m0; i++) yp[i] = 0.0;
    for (int q = 0; q < R->m1; q++) for (int e = R->tp[q]; e < R->tp[q + 1]; e++) yp[R->ti[e]] += R->tv[e] * yq[q];
}

/* A sign symmetry can exist only when the objective leaves the signs free: D C D = C ties s_i = s_j on every
 * off-diagonal entry (i, j) of C, and a sign constant on a block acts as the identity on it, so with C's pattern
 * connected on every semidefinite block and no free variables (the LP variables are even) the only sign symmetry is
 * the trivial one, whatever the rows. Rows are not looked at: they are what the reduced form recombines. */
static int fd_find(int *p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
static int rref_sign_possible(const Problem *P) {
    FreePair *prs = NULL; const int np = free_pairs_detect(P, &prs); free(prs);
    if (np > 0) return 1;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_SDP || B->n < 2) continue;
        int *p = malloc(sizeof(int) * (size_t)B->n), nc = B->n;
        for (int i = 0; i < B->n; i++) p[i] = i;
        for (int q = 0; q < B->C.nnz && nc > 1; q++) {
            if (B->C.row[q] == B->C.col[q] || B->C.val[q] == 0.0) continue;
            const int x = fd_find(p, B->C.row[q]), y = fd_find(p, B->C.col[q]);
            if (x != y) { p[x] = y; nc--; }
        }
        free(p);
        if (nc > 1) return 1;
    }
    return 0;
}

/* The reduced row echelon form of P's constraints as a new problem Q (C, the blocks and the objective constant those
 * of P; Q's rows identified with themselves: orig the identity), and T. NULL (Q untouched) when the problem is not one
 * of semidefinite and LP blocks as read, when the elimination changes nothing beyond a scaling of the rows, when it
 * passes its budget, or when the rows are inconsistent (left to the solver, which reports it). */
PSRref *rowspace_rref(const Problem *P, Problem *Q, int verbose) {
    const double t0 = wtime();
    const int m = P->m, nblk = P->nblk;
    if (m < 2 || P->ps || P->scaled || P->tbR != 0) return NULL;
    for (int k = 0; k < nblk; k++) if (P->blk[k].type != BLK_SDP && P->blk[k].type != BLK_LP) return NULL;
    if (P->orig) { if (P->morig != m) return NULL; for (int i = 0; i < m; i++) if (P->orig[i] != i) return NULL; }
    if (!rref_sign_possible(P)) {
        if (getenv("BRISK_RREFDBG")) fprintf(stderr, "rref: m %d skipped: no sign symmetry possible (the objective's pattern connects every block, no free variables)\n", m);
        return NULL;
    }
    size_t *off = malloc(sizeof(size_t) * ((size_t)nblk + 1)), R = 0;
    for (int k = 0; k < nblk; k++) { off[k] = R; const size_t n = (size_t)P->blk[k].n; R += P->blk[k].type == BLK_SDP ? n * (n + 1) / 2 : n; }
    off[nblk] = R;
    size_t ne = 0;
    for (int k = 0; k < nblk; k++) for (int t = 0; t < P->blk[k].ncon; t++) ne += (size_t)P->blk[k].A[t].nnz;
    const size_t necap = (size_t)rref_envi("BRISK_RREFMAXNNZ", 20000000);
    if (ne == 0 || ne > necap || ne + 2 * (size_t)m >= ((size_t)1 << 31)) { free(off); return NULL; }
    /* the entries by position, then row; the positions used become the columns 0 .. nc - 1 in increasing order */
    RREnt *E = malloc(sizeof(RREnt) * ne);
    size_t w = 0;
    for (int k = 0; k < nblk; k++) {
        const Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            for (int e = 0; e < S->nnz; e++) {
                if (S->val[e] == 0.0) continue;
                const size_t r = (size_t)S->row[e], c = (size_t)S->col[e];
                E[w].g = off[k] + (B->type == BLK_SDP ? c * (c + 1) / 2 + r : r);
                E[w].row = B->con[t]; E[w].v = S->val[e]; w++;
            }
        }
    }
    ne = w;
    qsort(E, ne, sizeof(RREnt), rrent_cmp);
    { size_t z = 0; for (size_t q = 0; q < ne; q++) { if (z > 0 && E[z - 1].g == E[q].g && E[z - 1].row == E[q].row) E[z - 1].v += E[q].v; else E[z++] = E[q]; } ne = z; }
    size_t *gpos = malloc(sizeof(size_t) * (ne + 1));
    int nc = 0;
    RRow *rw = calloc((size_t)m, sizeof(RRow));
    {   int *cnt = calloc((size_t)m, sizeof(int));
        for (size_t q = 0; q < ne; q++) { if (q == 0 || E[q].g != E[q - 1].g) gpos[nc++] = E[q].g; cnt[E[q].row]++; }
        for (int i = 0; i < m; i++) rr_reserve(&rw[i], cnt[i] + 1);
        free(cnt); }
    /* the column lists: a pool of (row, next) nodes; a node can be stale (its entry cancelled later), checked on use */
    size_t npool = 0, cappool = ne + (size_t)m + 16;
    int *nrow = malloc(sizeof(int) * cappool), *nnext = malloc(sizeof(int) * cappool), *head = malloc(sizeof(int) * ((size_t)nc + 1));
    for (int c = 0; c < nc; c++) head[c] = -1;
    {   int c = -1;
        for (size_t q = 0; q < ne; q++) {
            if (q == 0 || E[q].g != E[q - 1].g) c++;
            RRow *r = &rw[E[q].row];
            r->ix[r->len] = c; r->v[r->len] = E[q].v; r->len++;   /* (in increasing column order: E is sorted by position) */
            nrow[npool] = E[q].row; nnext[npool] = head[c]; head[c] = (int)npool; npool++;
        }
    }
    free(E);
    double *b = malloc(sizeof(double) * (size_t)m), *sc = malloc(sizeof(double) * (size_t)m), *sb = malloc(sizeof(double) * (size_t)m);
    for (int i = 0; i < m; i++) {
        RRow *r = &rw[i];
        double s = 0; for (int e = 0; e < r->len; e++) s = fmax(s, fabs(r->v[e]));
        sc[i] = s > 0 ? s : 1.0;
        b[i] = P->b[i]; sb[i] = fabs(b[i]);
        r->ix[r->len] = nc + i; r->v[r->len] = 1.0; r->len++;    /* T: the identity, as columns nc + i */
    }
    int *pivcol = malloc(sizeof(int) * (size_t)m);
    for (int i = 0; i < m; i++) pivcol[i] = -1;
    const double wmax = (double)rref_envi("BRISK_RREFWORK", 20) * (double)ne + 2e6, fmaxe = 4.0 * (double)ne + 2.0 * m + 1e5;
    double work = 0, stored = (double)ne + m;
    const double ctol = 1e-11, ptol = 1e-9;
    int nelim = 0, fail = 0, *cand = malloc(sizeof(int) * (size_t)m), *oth = malloc(sizeof(int) * (size_t)m);
    char *seen = calloc((size_t)m, 1);
    RRow tmp = {0, 0, NULL, NULL};
    for (int c = 0; c < nc && !fail; c++) {
        /* the rows with an entry in column c (the stale nodes and the repeated ones skipped) */
        int no = 0, best = -1; double bestr = 0;
        for (int nd = head[c]; nd >= 0; nd = nnext[nd]) {
            const int i = nrow[nd];
            work += 1;
            if (seen[i]) continue;
            int pos; const double v = rr_get(&rw[i], c, &pos);
            if (pos < 0) continue;
            seen[i] = 1; oth[no++] = i;
            if (pivcol[i] < 0) { const double rel = fabs(v) / sc[i]; if (rel > ptol && (rel > bestr || (rel == bestr && i < best))) { bestr = rel; best = i; } }
        }
        for (int q = 0; q < no; q++) seen[oth[q]] = 0;
        if (best < 0) continue;
        const int p = best;
        RRow *rp = &rw[p];
        {   const double pv = rr_get(rp, c, NULL);
            for (int e = 0; e < rp->len; e++) rp->v[e] = rp->ix[e] == c ? 1.0 : rp->v[e] / pv;
            b[p] /= pv; sc[p] /= fabs(pv); sb[p] /= fabs(pv);
            pivcol[p] = c; }
        for (int q = 0; q < no; q++) {
            const int i = oth[q];
            if (i == p) continue;
            RRow *ri = &rw[i];
            const double f = rr_get(ri, c, NULL);
            if (f == 0.0) continue;
            /* ri <- ri - f rp, merged in column order; cancellations (relative to the terms) dropped */
            rr_reserve(&tmp, ri->len + rp->len);
            int a = 0, d = 0, o = 0;
            while (a < ri->len || d < rp->len) {
                int col; double nv, mag;
                if (d >= rp->len || (a < ri->len && ri->ix[a] < rp->ix[d])) { col = ri->ix[a]; nv = ri->v[a]; a++; tmp.ix[o] = col; tmp.v[o] = nv; o++; continue; }
                if (a >= ri->len || rp->ix[d] < ri->ix[a]) {
                    col = rp->ix[d]; nv = -f * rp->v[d]; d++;
                    if (col < nc) {   /* a new entry: row i joins column col's list */
                        if (npool == cappool) { cappool *= 2; nrow = realloc(nrow, sizeof(int) * cappool); nnext = realloc(nnext, sizeof(int) * cappool); }
                        nrow[npool] = i; nnext[npool] = head[col]; head[col] = (int)npool; npool++;
                    }
                    tmp.ix[o] = col; tmp.v[o] = nv; o++; continue;
                }
                col = ri->ix[a]; { const double t1 = ri->v[a], t2 = f * rp->v[d]; nv = t1 - t2; mag = fmax(fabs(t1), fabs(t2)); } a++; d++;
                if (col == c || fabs(nv) <= ctol * mag) continue;
                tmp.ix[o] = col; tmp.v[o] = nv; o++;
            }
            work += ri->len + rp->len;
            stored += o - ri->len;
            { int *ti = ri->ix; double *tv = ri->v; const int cp = ri->cap; ri->ix = tmp.ix; ri->v = tmp.v; ri->cap = tmp.cap; ri->len = o; tmp.ix = ti; tmp.v = tv; tmp.cap = cp; tmp.len = 0; }
            b[i] -= f * b[p];
            sc[i] = fmax(sc[i], fabs(f) * sc[p]); sb[i] = fmax(sb[i], fabs(f) * sb[p]);
            nelim++;
        }
        if (work > wmax || stored > fmaxe) fail = 1;
    }
    free(tmp.ix); free(tmp.v); free(cand); free(oth); free(seen); free(nrow); free(nnext); free(head);
    int m1 = 0, ndep = 0, ninc = 0;
    if (!fail) for (int i = 0; i < m; i++) {
        if (pivcol[i] >= 0) { m1++; continue; }
        ndep++;
        if (fabs(b[i]) > 1e-8 * fmax(sb[i], 1.0)) ninc++;
    }
    PSRref *Rr = NULL;
    if (fail || ninc > 0 || (nelim == 0 && ndep == 0)) {
        if (verbose >= 2) {
            if (fail) printf("presolve: reduced row echelon form of the constraints not computed (its work or fill passed its budget; %.2fs)\n", wtime() - t0);
            else if (ninc) printf("presolve: reduced row echelon form of the constraints: %d inconsistent dependent rows, not applied\n", ninc);
            else printf("presolve: the constraints are in reduced row echelon form up to scaling (%.2fs)\n", wtime() - t0);
        }
    } else {
        /* the new rows in the order of their pivot columns */
        int *ord = malloc(sizeof(int) * (size_t)(m1 + 1)), *byc = malloc(sizeof(int) * ((size_t)nc + 1));
        for (int c = 0; c < nc; c++) byc[c] = -1;
        for (int i = 0; i < m; i++) if (pivcol[i] >= 0) byc[pivcol[i]] = i;
        { int q = 0; for (int c = 0; c < nc; c++) if (byc[c] >= 0) ord[q++] = byc[c]; }
        free(byc);
        size_t nA = 0, nT = 0, nCe = 0;
        for (int q = 0; q < m1; q++) { const RRow *r = &rw[ord[q]]; for (int e = 0; e < r->len; e++) { if (r->ix[e] < nc) nA++; else nT++; } }
        for (int k = 0; k < nblk; k++) nCe += (size_t)P->blk[k].C.nnz;
        const size_t nt = nA + nCe;
        int *tm = malloc(sizeof(int) * (nt + 1)), *tb = malloc(sizeof(int) * (nt + 1)), *tii = malloc(sizeof(int) * (nt + 1)), *tj = malloc(sizeof(int) * (nt + 1));
        double *tvv = malloc(sizeof(double) * (nt + 1)), *bq = malloc(sizeof(double) * (size_t)(m1 + 1));
        int *bsz = malloc(sizeof(int) * (size_t)nblk);
        for (int k = 0; k < nblk; k++) bsz[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
        size_t z = 0;
        for (int k = 0; k < nblk; k++) {
            const SpSym *C = &P->blk[k].C;
            for (int q = 0; q < C->nnz; q++) { tm[z] = 0; tb[z] = k + 1; tii[z] = C->row[q] + 1; tj[z] = C->col[q] + 1; tvv[z] = -C->val[q]; z++; }
        }
        Rr = calloc(1, sizeof(PSRref));
        Rr->m0 = m; Rr->m1 = m1;
        Rr->tp = malloc(sizeof(int) * (size_t)(m1 + 1)); Rr->ti = malloc(sizeof(int) * (nT + 1)); Rr->tv = malloc(sizeof(double) * (nT + 1));
        size_t zt = 0; int kb = 0;
        for (int q = 0; q < m1; q++) {
            const RRow *r = &rw[ord[q]];
            Rr->tp[q] = (int)zt;
            bq[q] = b[ord[q]];
            for (int e = 0; e < r->len; e++) {
                const int col = r->ix[e];
                if (col >= nc) { Rr->ti[zt] = col - nc; Rr->tv[zt] = r->v[e]; zt++; continue; }
                const size_t g = gpos[col];
                if (g < off[kb] || g >= off[kb + 1]) { int lo = 0, hi = nblk - 1; while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (off[mid] <= g) lo = mid; else hi = mid - 1; } kb = lo; }
                const size_t d = g - off[kb];
                size_t rr, cc;
                if (P->blk[kb].type == BLK_SDP) {
                    cc = (size_t)((sqrt(8.0 * (double)d + 1.0) - 1.0) / 2.0);
                    while (cc * (cc + 1) / 2 > d) cc--;
                    while ((cc + 1) * (cc + 2) / 2 <= d) cc++;
                    rr = d - cc * (cc + 1) / 2;
                } else rr = cc = d;
                tm[z] = q + 1; tb[z] = kb + 1; tii[z] = (int)rr + 1; tj[z] = (int)cc + 1; tvv[z] = r->v[e]; z++;
            }
        }
        Rr->tp[m1] = (int)zt;
        BriskData dd = { m1, nblk, bsz, bq, z, tm, tb, tii, tj, tvv };
        if (problem_from_sdpa_data(&dd, Q) != 0) { rref_free(Rr); Rr = NULL; }
        else Q->obj_off = P->obj_off;
        free(tm); free(tb); free(tii); free(tj); free(tvv); free(bq); free(bsz); free(ord);
        if (Rr && verbose >= 2)
            printf("presolve: reduced row echelon form of the constraints: %d -> %d rows (%d dependent), entries %zu -> %zu, T %zu entries (%.2fs)\n",
                   m, m1, ndep, ne, nA, nT, wtime() - t0);
    }
    for (int i = 0; i < m; i++) { free(rw[i].ix); free(rw[i].v); }
    free(rw); free(b); free(sc); free(sb); free(pivcol); free(gpos); free(off);
    if (getenv("BRISK_RREFDBG")) fprintf(stderr, "rref: m %d nc %d ne %zu elim %d work %.3g (cap %.3g) fill %.3g (cap %.3g) dep %d inc %d fail %d applied %d %.3fs\n",
                                         m, nc, ne, nelim, work, wmax, stored, fmaxe, ndep, ninc, fail, Rr != NULL, wtime() - t0);
    return Rr;
}
