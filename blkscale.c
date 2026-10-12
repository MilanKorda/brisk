/* blkscale.c - 5.14: diagonal congruence scaling of the semidefinite blocks (presolve).
 *
 * Each SDP block k gets a diagonal D_k > 0 and the solver works with X'_k, where X_k = D_k X'_k D_k:
 *   A'_i = D A_i D,  C' = D C D  (entry (p,q) of every data matrix times d_p d_q),  b' = b,
 * so <A'_i, X'> = <A_i, X>, <C', X'> = <C, X>, y' = y and Z = D^-1 Z' D^-1. The cone is preserved
 * (D X' D is positive semidefinite exactly when X' is), the constraint values, the objective, the
 * multipliers and <X, Z> are those of the problem as given; only the coordinates of the blocks change.
 * An interior-point method is invariant under such a change in exact arithmetic except for its start
 * and its scale-dependent tests (the start X = I, Z = I is D^2, D^-2 in the problem as given); a
 * first-order method is not invariant at all, and there the scaling is a preconditioner.
 *
 * D from Ruiz iterations on the 2-norms of the data: with constraint factors r_i (used only to
 * balance the rows while D is computed, not applied: the solver scales its rows itself),
 *   rho_i = |r_i D A_i D|_F,  kappa_p = (sum_i |row p of r_i D A_i D|^2 / m)^{1/2},
 *   r_i <- r_i / sqrt(rho_i),  d_p <- d_p / sqrt(kappa_p),
 * then each block's D normalized to geometric mean 1 and rounded to powers of two (the scaling is
 * exact in floating point, and indices that a symmetry exchanges keep equal factors). LP blocks are
 * left alone (split free pairs stay opposite). Not applied when every block's factors stay within a
 * factor 2 of each other (nothing to gain).                                                     */
#include "brisk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

double g_blkscale_min = 4.0;   /* auto (-blkscale -1): applied when some block's factors spread by at least this */
int g_blkscale_off = 0;          /* set by the driver for the re-solve of the problem as given */

struct BlkScale {
    int nblk, m, *bs;
    double **d;          /* per SDP block: its n factors (NULL for LP blocks and unscaled blocks) */
    double spread;       /* the largest max/min ratio of a block's factors */
};

static double p2(double x) { return ldexp(1.0, (int)lround(log2(x))); }

BlkScale *blkscale_apply(Problem *P, int mode, int verbose) {
    if (mode == 0 || g_blkscale_off) return NULL;
    const int m = P->m, nb = P->nblk;
    int nsdp = 0;
    for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP && P->blk[k].n > 1) nsdp++;
    if (!nsdp || m < 1) return NULL;
    double **d = calloc(nb, sizeof(double *)), **acc = calloc(nb, sizeof(double *));
    for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP && P->blk[k].n > 1) {
        d[k] = malloc(sizeof(double) * P->blk[k].n); acc[k] = malloc(sizeof(double) * P->blk[k].n);
        for (int p = 0; p < P->blk[k].n; p++) d[k][p] = 1.0;
    }
    double *r = malloc(sizeof(double) * m), *rho = malloc(sizeof(double) * m);
    for (int i = 0; i < m; i++) r[i] = 1.0;
    for (int it = 0; it < 10; it++) {
        memset(rho, 0, sizeof(double) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *A = &B->A[t]; const int i = B->con[t]; double s = 0;
                for (int q = 0; q < A->nnz; q++) {
                    const double v = A->val[q] * (d[k] ? d[k][A->row[q]] * d[k][A->col[q]] : 1.0);
                    s += (A->row[q] == A->col[q] ? 1.0 : 2.0) * v * v;
                }
                rho[i] += s;
            }
        }
        for (int i = 0; i < m; i++) if (rho[i] > 0) r[i] /= sqrt(sqrt(rho[i]));
        for (int k = 0; k < nb; k++) {
            if (!d[k]) continue;
            const Block *B = &P->blk[k]; const int n = B->n;
            memset(acc[k], 0, sizeof(double) * n);
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *A = &B->A[t]; const double ri = r[B->con[t]];
                for (int q = 0; q < A->nnz; q++) {
                    const int a = A->row[q], c = A->col[q];
                    const double v = ri * A->val[q] * d[k][a] * d[k][c], v2 = v * v;
                    acc[k][a] += v2; if (a != c) acc[k][c] += v2;
                }
            }
            for (int p = 0; p < n; p++) if (acc[k][p] > 0) d[k][p] /= sqrt(sqrt(acc[k][p] / m));
        }
    }
    double spread = 1.0;
    for (int k = 0; k < nb; k++) {
        if (!d[k]) continue;
        const int n = P->blk[k].n; double lg = 0;
        for (int p = 0; p < n; p++) lg += log(d[k][p]);
        const double g = exp(lg / n);
        double lo = 1e300, hi = 0;
        for (int p = 0; p < n; p++) { d[k][p] = p2(d[k][p] / g); lo = fmin(lo, d[k][p]); hi = fmax(hi, d[k][p]); }
        if (hi / lo < 2.0) { free(d[k]); d[k] = NULL; continue; }   /* (within a factor 2: left alone) */
        spread = fmax(spread, hi / lo);
    }
    free(r); free(rho);
    for (int k = 0; k < nb; k++) free(acc[k]);
    free(acc);
    int any = 0; for (int k = 0; k < nb; k++) if (d[k]) any = 1;
    if (!any || (mode < 0 && spread < g_blkscale_min)) { for (int k = 0; k < nb; k++) free(d[k]); free(d); return NULL; }
    /* apply: the stored (upper) entries and the full expansions of A_i and C */
    for (int k = 0; k < nb; k++) {
        if (!d[k]) continue;
        Block *B = &P->blk[k]; const double *dk = d[k];
        for (int t = 0; t <= B->ncon; t++) {
            SpSym *A = t < B->ncon ? &B->A[t] : &B->C;
            for (int q = 0; q < A->nnz; q++) A->val[q] *= dk[A->row[q]] * dk[A->col[q]];
            if (A->fv) for (int q = 0; q < A->ef; q++) A->fv[q] *= dk[A->fr[q]] * dk[A->fc[q]];
        }
    }
    BlkScale *S = calloc(1, sizeof(BlkScale));
    S->nblk = nb; S->m = m; S->d = d; S->spread = spread;
    S->bs = malloc(sizeof(int) * nb);
    for (int k = 0; k < nb; k++) S->bs[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    if (verbose >= 0) {
        int ns = 0; for (int k = 0; k < nb; k++) if (d[k]) ns++;
        printf("presolve: diagonal scaling of %d semidefinite block%s (factors within %.3g)\n", ns, ns == 1 ? "" : "s", spread);
    }
    return S;
}

/* the solution back to the coordinates of the problem as read: X = D X' D, y unchanged. Xp may be NULL. */
void blkscale_unmap(const BlkScale *S, double **Xp, const double *yp, double ***Xo, double *yo) {
    memcpy(yo, yp, sizeof(double) * S->m);
    if (!Xp || !Xo) return;
    double **X = calloc(S->nblk + 1, sizeof(double *));
    for (int k = 0; k < S->nblk; k++) {
        const int n = abs(S->bs[k]); const size_t sz = S->bs[k] < 0 ? (size_t)n : (size_t)n * n;
        X[k] = malloc(sizeof(double) * sz + 8);
        memcpy(X[k], Xp[k], sizeof(double) * sz);
        if (!S->d[k]) continue;
        const double *dk = S->d[k];
        for (int c = 0; c < n; c++) for (int r = 0; r < n; r++) X[k][r + (size_t)c * n] *= dk[r] * dk[c];
    }
    *Xo = X;
}

double blkscale_spread(const BlkScale *S) { return S ? S->spread : 1.0; }

void blkscale_free(BlkScale *S) {
    if (!S) return;
    for (int k = 0; k < S->nblk; k++) free(S->d[k]);
    free(S->d); free(S->bs); free(S);
}
