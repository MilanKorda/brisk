/* SPDX-License-Identifier: Apache-2.0
 * hpalg.inc (5.9): two kernels of the block diagonalisation in high precision (hpred.inc),
 * compiled per precision after hpfom.inc.
 *
 * eigref: one step of the refinement of an approximate eigenvector matrix (Ogita and Aishima,
 * Japan J. Indust. Appl. Math. 35 (2018) and 36 (2019), the version for multiple eigenvalues):
 * with R = I - V V', S = V A V' and lambda_i = S_ii / (1 - R_ii), the correction
 *     E_ij = (S_ij + lambda_j R_ij) / (lambda_j - lambda_i)    (i, j in different clusters),
 *     E_ij = R_ij / 2                                           (same cluster, and i = j),
 * V <- V + E' V squares the error of the invariant subspaces of the clusters and of the
 * orthonormality: four matrix products per step, where a Jacobi sweep in the same precision
 * costs about ten and three are needed. The rows of V are the vectors.
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
static double HPN(eigref)(int n, const HPT *A, HPT *V, const int *cl, HPT *lam, HPT *S, HPT *R, HPT *T) {
    HPN(gemm_rows)(n, n, n, V, A, T);
    HPN(gemm_abt)(n, n, n, T, V, S);
    HPN(gemm_abt)(n, n, n, V, V, R);
    double emax = 0;
    for (int i = 0; i < n; i++) {
        TMP(one); TMP(d); HP_SETD(one, 1.0);
        for (int j = 0; j < n; j++) { HPT *r = EL(R, (size_t)i * n + j); HP_NEG(r, r); if (i == j) HP_ADD(r, r, one); }
        HP_SUB(d, one, EL(R, (size_t)i * n + i));
        HP_DIV(EL(lam, i), EL(S, (size_t)i * n + i), d);
    }
#pragma omp parallel for schedule(static) reduction(max : emax) if (n >= 64)
    for (int i = 0; i < n; i++) {
        TMP(half); TMP(num); TMP(den); HP_SETD(half, 0.5);
        for (int j = 0; j < n; j++) {
            HPT *e = EL(S, (size_t)i * n + j); const HPT *r = EL(R, (size_t)i * n + j);
            if (i == j || cl[i] == cl[j]) HP_MUL(e, r, half);
            else { HP_MUL(num, EL(lam, j), r); HP_ADD(num, num, e); HP_SUB(den, EL(lam, j), EL(lam, i)); HP_DIV(e, num, den); }
            const double v = fabs(HP_GETD(e)); if (v > emax) emax = v;
        }
    }
    HPN(transpose)(n, S);
    HPN(gemm_rows)(n, n, n, S, V, T);
    for (size_t q = 0; q < (size_t)n * n; q++) HP_ADD(EL(V, q), EL(V, q), EL(T, q));
    return emax;
}
/* The aligned bases of the clusters of S1 (hpred.inc): V (rows, from the double-double analysis)
 * refined by `iters` steps, then along the tree `par` (clusters in `order`, parents first)
 * O_b = G' / gamma with G = V_b S2 W_a' for the aligned rows W_a of the parent: a scalar
 * multiple of an orthogonal matrix for a component of real type (fail[b] = 1 otherwise, and
 * the rows of the cluster are left as they are). Qm: the rows in the master precision.
 * Vstart: the start in the master precision instead of Vdd; refine_only: no alignment, Qm = the
 * refined vectors. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))              /* (compiled for four precisions, used in three) */
#endif
static void HPN(algbuild)(const HPJob *J, int n, const uint64_t *S1m, const uint64_t *S2m, const dd_t *Vdd, const uint64_t *Vstart, int refine_only, int ncl, const int *cs, const int *par, const int *order,
                          int iters, uint64_t *Qm, char *fail, double orthtol, double *emax) {
    const size_t N = (size_t)n * n, Wm = (size_t)J->Lm + 2;
    HPT *A = HPN(alloc)(N), *V = HPN(alloc)(N), *S = HPN(alloc)(N), *R = HPN(alloc)(N), *T = HPN(alloc)(N), *lam = HPN(alloc)((size_t)n + 1);
    int *cl = (int *)malloc(sizeof(int) * ((size_t)n + 1));
    for (int a = 0; a < ncl; a++) for (int i = cs[a]; i < cs[a + 1]; i++) cl[i] = a;
    for (size_t q = 0; q < N; q++) {
        HP_IMPORT(EL(A, q), S1m + q * Wm, J);
        if (Vstart) HP_IMPORT(EL(V, q), Vstart + q * Wm, J);
        else { TMP(t); HP_SETD(EL(V, q), Vdd[q].h); HP_SETD(t, Vdd[q].l); HP_ADD(EL(V, q), EL(V, q), t); }
    }
    *emax = 0;
    for (int it = 0; it < iters; it++) *emax = HPN(eigref)(n, A, V, cl, lam, S, R, T);
    if (refine_only) {                                       /* the refined vectors only (the start of a higher precision) */
        for (size_t q = 0; q < N; q++) HP_EXPORT(Qm + q * Wm, EL(V, q), J);
        free(A); free(V); free(S); free(R); free(T); free(lam); free(cl);
        return;
    }
    for (size_t q = 0; q < N; q++) HP_IMPORT(EL(A, q), S2m + q * Wm, J);
    HPN(gemm_rows)(n, n, n, V, A, T);                        /* T = V S2 */
    HPT *Q = S;
    HPN(zero)(N, Q);
    for (int o = 0; o < ncl; o++) {
        const int b = order[o], mm = cs[b + 1] - cs[b], a = par[b];
        fail[b] = 0;
        if (a >= 0 && fail[a]) fail[b] = 1;
        if (a < 0 || fail[b]) { HPN(copy)((size_t)mm * n, EL(V, (size_t)cs[b] * n), EL(Q, (size_t)cs[b] * n)); continue; }
        HPT *G = R, *O = EL(R, (size_t)mm * mm);
        TMP(g2); TMP(t); TMP(u); HP_SETD(g2, 0.0);
        for (int s = 0; s < mm; s++) for (int r = 0; r < mm; r++) {
            HPT *g = EL(G, (size_t)s * mm + r);
            HPN(dot)((size_t)n, EL(T, (size_t)(cs[b] + s) * n), EL(Q, (size_t)(cs[a] + r) * n), g);
            HP_MUL(t, g, g); HP_ADD(g2, g2, t);
        }
        HP_SETD(u, (double)mm); HP_DIV(t, g2, u);
        if (HP_SGN(t) <= 0) fail[b] = 1;
        else {
            HP_SQRT(u, t);
            for (int r = 0; r < mm; r++) for (int s = 0; s < mm; s++) HP_DIV(EL(O, (size_t)r * mm + s), EL(G, (size_t)s * mm + r), u);
            for (int r = 0; r < mm && !fail[b]; r++) for (int r2 = r; r2 < mm; r2++) {
                HPN(dot)((size_t)mm, EL(O, (size_t)r * mm), EL(O, (size_t)r2 * mm), t);
                if (fabs(HP_GETD(t) - (r == r2 ? 1.0 : 0.0)) > orthtol) { fail[b] = 1; break; }
            }
        }
        if (fail[b]) { HPN(copy)((size_t)mm * n, EL(V, (size_t)cs[b] * n), EL(Q, (size_t)cs[b] * n)); continue; }
        for (int r = 0; r < mm; r++) for (int s = 0; s < mm; s++) HPN(axpy)((size_t)n, EL(O, (size_t)r * mm + s), EL(V, (size_t)(cs[b] + s) * n), EL(Q, (size_t)(cs[b] + r) * n));
    }
    for (size_t q = 0; q < N; q++) HP_EXPORT(Qm + q * Wm, EL(Q, q), J);
    free(A); free(V); free(S); free(R); free(T); free(lam); free(cl);
}
