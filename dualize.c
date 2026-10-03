/* BRISK 4.30: the dual form.
 *
 * An SOS program comes in two SDPA files: the kernel form (Gram matrices as the cone
 * variable, one row per coefficient equation, the multipliers as split free pairs) and the
 * image form (the Gram matrices parametrized by free variables, one row per Gram entry).
 * They are the same primal-dual pair written from the two sides, and the interior-point
 * cost is very different: the Schur complement has the size of the row count, which is m
 * in one form and N - m in the other (N the cone dimension). MOSEK picks the cheaper side
 * (its automatic dualizer); until 4.29 BRISK solved the file as given, which made the
 * "better form per solver" comparison a comparison of files.
 *
 * Here the problem as read,  (P) min <C,X>  s.t. A(X) = b, X in K, x_f free,
 * is replaced by its dual written as a primal,
 *   (P') min -b'y  s.t.  Z_ij + sum_t y_t (A_t)_ij = C_ij  for every entry (i <= j) of every
 *                        block, z_l + sum_t y_t (A_t)_l = c_l for every nonneg LP variable,
 *                        sum_t y_t (A_t)_f = c_f for every free pair,  Z in K, y free,
 * with y as split pairs. Free elimination (freeelim.c) then removes the m pairs of y, one
 * row each, leaving N + n_f - m rows over Z: the other form of the file. The solution maps
 * back exactly: y of (P) is the free values, Z of (P) the cone variable of (P'), and X of
 * (P) comes from the multipliers w of the entry rows, X_ii = -w_ii, X_ij = -w_ij / 2.
 * Everything downstream (trace bound, facial reduction, the solver, the fallbacks) sees an
 * ordinary problem. The rule (automatic): the class of free elimination (at most 32 SDP
 * blocks) and N + n_f - m <= (m - n_f) / 2, i.e. the dual form has at most half the rows.
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void *dx(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }

PSDual *dualize_apply(Problem *P, int automatic, int verbose) {
    const int m = P->m, nb = P->nblk;
    FreePair *pr = NULL;
    const int np = free_pairs_detect(P, &pr);
    /* sizes: N (cone dimension), nonneg LP variables, SDP blocks */
    long N = 0; int nsdp = 0, nlp = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_SDP) { N += (long)B->n * (B->n + 1) / 2; nsdp++; }
        else nlp += B->n;
    }
    const long mK = m - np, mD = N + (nlp - 2L * np) + np - m;
    if (automatic) {
        if (nsdp > 32 || mD < 1 || mD > mK / 2 || mK < 20) { free(pr); return NULL; }
    } else if (mD < 1) { free(pr); return NULL; }
    /* pair slot map: (block, slot) -> pair index or -1 */
    int **pslot = dx(sizeof(int *) * (nb + 1));
    for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_LP) { pslot[k] = dx(sizeof(int) * (P->blk[k].n + 1)); for (int l = 0; l < P->blk[k].n; l++) pslot[k][l] = -1; }
    for (int p = 0; p < np; p++) { pslot[pr[p].blk][pr[p].ip] = p; pslot[pr[p].blk][pr[p].im] = p; }
    PSDual *D = dx(sizeof(PSDual));
    D->m = m; D->nblk = nb; D->np = np; D->pr = pr;
    D->bs = dx(sizeof(int) * (nb + 1));
    D->sdpmap = dx(sizeof(int) * (nb + 1));
    D->rowbase = dx(sizeof(int) * (nb + 1));
    D->lpbase = dx(sizeof(int) * (nb + 1));
    D->lpslot = dx(sizeof(int) * (nlp + 1));
    D->prow = dx(sizeof(int) * (np + 1));
    /* dual-form blocks: the SDP blocks in order, the nonneg LP block, the y block */
    int nbp = 0, row = 0, lpoff = 0, nz = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        D->bs[k] = B->type == BLK_LP ? -B->n : B->n;
        if (B->type == BLK_SDP) { D->sdpmap[k] = nbp++; D->rowbase[k] = row; row += B->n * (B->n + 1) / 2; }
        else { D->sdpmap[k] = -1; D->lpbase[k] = lpoff; lpoff += B->n; }
    }
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_LP) continue;
        for (int l = 0; l < B->n; l++) D->lpslot[D->lpbase[k] + l] = pslot[k][l] >= 0 ? -1 : row + nz++;
    }
    row += nz;
    for (int p = 0; p < np; p++) D->prow[p] = row++;
    D->mp = row;
    D->nz = nz;
    D->zblk = nz > 0 ? nbp : -1;
    D->yblk = nz > 0 ? nbp + 1 : nbp;
    const int nbp_all = D->yblk + 1;
    /* triplets of the dual form */
    size_t nt = 0, ct = 4096;
    int *tc = malloc(sizeof(int) * ct), *tb = malloc(sizeof(int) * ct), *ti = malloc(sizeof(int) * ct), *tj = malloc(sizeof(int) * ct);
    double *tv = malloc(sizeof(double) * ct);
#define TPUT(con_, bk_, i_, j_, v_) do { \
        if (nt == ct) { ct *= 2; tc = realloc(tc, sizeof(int) * ct); tb = realloc(tb, sizeof(int) * ct); ti = realloc(ti, sizeof(int) * ct); tj = realloc(tj, sizeof(int) * ct); tv = realloc(tv, sizeof(double) * ct); } \
        tc[nt] = (con_); tb[nt] = (bk_); ti[nt] = (i_); tj[nt] = (j_); tv[nt] = (v_); nt++; } while (0)
    double *bp = dx(sizeof(double) * (D->mp + 1));
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_SDP) {
            const int n = B->n, kp = D->sdpmap[k], base = D->rowbase[k];
            /* cone entries: Z_ij (value 1/2 off the diagonal: <E, Z> = Z_ij) */
            for (int j = 0; j < n; j++)
                for (int i = 0; i <= j; i++) TPUT(base + j * (j + 1) / 2 + i, kp, i, j, i == j ? 1.0 : 0.5);
            for (int q = 0; q < B->C.nnz; q++) {
                const int i = B->C.row[q], j = B->C.col[q];      /* i <= j */
                bp[base + j * (j + 1) / 2 + i] += B->C.val[q];
            }
            for (int t = 0; t < B->ncon; t++) {
                const int c = B->con[t];
                const SpSym *S = &B->A[t];
                for (int q = 0; q < S->nnz; q++) {
                    const int i = S->row[q], j = S->col[q], r = base + j * (j + 1) / 2 + i;
                    TPUT(r, D->yblk, c, c, S->val[q]);
                    TPUT(r, D->yblk, m + c, m + c, -S->val[q]);
                }
            }
        } else {
            const int n = B->n;
            for (int l = 0; l < n; l++) {
                const int r = D->lpslot[D->lpbase[k] + l];
                if (r >= 0) { const int z = r - (D->mp - nz - np); TPUT(r, D->zblk, z, z, 1.0); }
            }
            for (int q = 0; q < B->C.nnz; q++) {
                const int l = B->C.row[q], p = pslot[k][l];
                if (p >= 0) { if (l == pr[p].ip) bp[D->prow[p]] += B->C.val[q]; }
                else bp[D->lpslot[D->lpbase[k] + l]] += B->C.val[q];
            }
            for (int t = 0; t < B->ncon; t++) {
                const int c = B->con[t];
                const SpSym *S = &B->A[t];
                for (int q = 0; q < S->nnz; q++) {
                    const int l = S->row[q], p = pslot[k][l];
                    int r;
                    if (p >= 0) { if (l != pr[p].ip) continue; r = D->prow[p]; }
                    else r = D->lpslot[D->lpbase[k] + l];
                    TPUT(r, D->yblk, c, c, S->val[q]);
                    TPUT(r, D->yblk, m + c, m + c, -S->val[q]);
                }
            }
        }
    }
    /* objective: min -b'y */
    for (int i = 0; i < m; i++) if (P->b[i] != 0) { TPUT(-1, D->yblk, i, i, -P->b[i]); TPUT(-1, D->yblk, m + i, m + i, P->b[i]); }
    int *bsz = dx(sizeof(int) * (nbp_all + 1));
    for (int k = 0; k < nb; k++) if (D->sdpmap[k] >= 0) bsz[D->sdpmap[k]] = P->blk[k].n;
    if (nz > 0) bsz[D->zblk] = -nz;
    bsz[D->yblk] = -2 * m;
    Problem Q;
    problem_from_trips(&Q, D->mp, nbp_all, bsz, bp, nt, tc, tb, ti, tj, tv);
    for (int k = 0; k < nb; k++) block_free_contents(&P->blk[k]);
    free(P->blk); free(P->b); free(P->b0); free(P->orig);
    P->blk = Q.blk; P->b = Q.b; P->b0 = Q.b0; P->m = D->mp; P->nblk = nbp_all;
    P->orig = malloc(sizeof(int) * (D->mp + 1));
    for (int i = 0; i < D->mp; i++) P->orig[i] = i;
    P->morig = D->mp;
    P->tbR = 0;
    free(Q.orig);
    free(tc); free(tb); free(ti); free(tj); free(tv); free(bsz); free(bp);
    for (int k = 0; k < nb; k++) free(pslot[k]);
    free(pslot);
    if (verbose >= 0)
        printf("presolve: dual form: %d rows over the cone entries (%ld would remain of %d), %d free pairs of y\n",
               D->mp, mD, m, m);
    return D;
}

/* (Xp, yp): the cone blocks and the row multipliers of the dual form; (Xo, yo): the problem
 * as read. Xo is allocated here (full storage for SDP blocks). */
void dualize_unmap(const PSDual *D, double **Xp, const double *yp, double ***Xo_out, double *yo) {
    const int m = D->m, nb = D->nblk;
    for (int i = 0; i < m; i++) yo[i] = Xp ? Xp[D->yblk][i] - Xp[D->yblk][m + i] : 0.0;
    double **Xo = malloc(sizeof(double *) * (nb + 1));
    for (int k = 0; k < nb; k++) {
        const int n = abs(D->bs[k]);
        if (D->bs[k] > 0) {
            Xo[k] = dx(sizeof(double) * ((size_t)n * n + 1));
            const int base = D->rowbase[k];
            for (int j = 0; j < n; j++)
                for (int i = 0; i <= j; i++) {
                    const double w = yp[base + j * (j + 1) / 2 + i];
                    const double x = i == j ? -w : -0.5 * w;
                    Xo[k][i + (size_t)j * n] = x; Xo[k][j + (size_t)i * n] = x;
                }
        } else {
            Xo[k] = dx(sizeof(double) * (n + 1));
            for (int l = 0; l < n; l++) {
                const int r = D->lpslot[D->lpbase[k] + l];
                if (r >= 0) Xo[k][l] = -yp[r];
            }
        }
    }
    for (int p = 0; p < D->np; p++) {
        const double xf = -yp[D->prow[p]];
        Xo[D->pr[p].blk][D->pr[p].ip] = xf > 0 ? xf : 0.0;
        Xo[D->pr[p].blk][D->pr[p].im] = xf < 0 ? -xf : 0.0;
    }
    *Xo_out = Xo;
}

void dualize_free(PSDual *D) {
    if (!D) return;
    free(D->bs); free(D->sdpmap); free(D->rowbase); free(D->lpbase); free(D->lpslot); free(D->prow); free(D->pr);
    free(D);
}
