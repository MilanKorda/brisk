/* SPDX-License-Identifier: Apache-2.0
 * hpipm.inc (5.0): the high-precision interior-point method, written once and compiled for
 * double, double-double, quad-double and mpx (hpsolve.c includes this file four times).
 *
 * The including file defines: HPT (storage word), HPW (words per element), HPWMAX, HPN(name),
 * and the element operations on pointers: HP_SETD HP_GETD HP_COPY HP_ADD HP_SUB HP_MUL HP_DIV
 * HP_SQRT HP_NEG HP_SGN HP_ADDMUL HP_SUBMUL HP_IMPORT HP_EXPORT, and HP_EPS (unit roundoff).
 *
 * Matrices are dense, row-major; symmetric ones are stored in full. All dense kernels reduce
 * to axpy and dot over contiguous rows, which are the two loops to specialise per type.
 */
#define EL(A, i) ((A) + (size_t)(i) * (size_t)HPW)
#define TMP(t) HPT t[HPWMAX]

static HPT *HPN(alloc)(size_t n) {
    HPT *p = (HPT *)calloc((n ? n : 1) * (size_t)HPW, sizeof(HPT));
    if (!p) { fprintf(stderr, "brisk: out of memory (high precision)\n"); exit(1); }
    return p;
}
static inline void HPN(zero)(size_t n, HPT *x) { memset(x, 0, n * (size_t)HPW * sizeof(HPT)); }
static inline void HPN(copy)(size_t n, const HPT *x, HPT *y) { memcpy(y, x, n * (size_t)HPW * sizeof(HPT)); }

#ifndef HP_OWN_KERNELS
/* y += a x */
static void HPN(axpy)(size_t n, const HPT *a, const HPT *x, HPT *y) {
    for (size_t i = 0; i < n; i++) HP_ADDMUL(EL(y, i), a, EL(x, i));
}
/* r = x . y */
static void HPN(dot)(size_t n, const HPT *x, const HPT *y, HPT *r) {
    TMP(s); HP_SETD(s, 0.0);
    for (size_t i = 0; i < n; i++) HP_ADDMUL(s, EL(x, i), EL(y, i));
    HP_COPY(r, s);
}
#endif
static void HPN(scal)(size_t n, const HPT *a, HPT *x) { for (size_t i = 0; i < n; i++) HP_MUL(EL(x, i), EL(x, i), a); }

/* Cholesky in place, lower triangle, row-major (Crout: dots over contiguous rows); 0 = ok */
#define HPN_chol(n, A) HPN(cholr)(n, A, 0.0, NULL, NULL, 0, NULL, NULL)
/* Dependent rows (the Schur complement of a problem with linearly dependent constraints is
 * singular). A row flagged in mask is skipped: its pivot is made huge, which sets that
 * component of every solve to zero. mode 1 (detection, at a well-conditioned point): a pivot
 * below reg times its diagonal entry flags the row. mode 2 (a new precision): a flagged row
 * whose pivot is above reg times its diagonal entry is not dependent after all and loses the
 * flag. reg > 0 without a mask: the plain threshold (BRISK_HPREG). *nskip counts the skips. */
static int HPN(cholr)(int n, HPT *A, double reg, int *nskip, int *mask, int mode, double *minrat, const int *fstin) {
    /* above the blocking threshold: HP_KB columns at a time; the rows below the block take
     * their entries of these columns in one visit (the same dots in the same order) */
    const int kb = (double)n * n * sizeof(HPT) > 2000000.0 ? 16 : n;
    /* The envelope (5.3): fst[i] is the first nonzero column of row i; the factor has zeros
     * before it too, so every dot starts at the later of the two rows' first columns and the
     * entries before fst[i] are not visited.  A Schur complement of many small blocks that
     * couple neighbouring constraints (sparse polynomial optimization) is banded in this sense
     * and its factorization costs m w^2 instead of m^3 / 3.  Used when the envelope holds less
     * than 70% of the triangle; otherwise fst = 0 and nothing changes. */
    int *fst = NULL;
    if (fstin) { fst = (int *)malloc(sizeof(int) * (size_t)n); memcpy(fst, fstin, sizeof(int) * (size_t)n); }   /* known from the structure */
    else if (n >= 64) {
        fst = (int *)malloc(sizeof(int) * (size_t)n);
        double env = 0.0;
        for (int i = 0; i < n; i++) {
            const HPT *Ai = EL(A, (size_t)i * n);
            int f = 0;
            while (f < i && HP_SGN(EL(Ai, f)) == 0) f++;
            fst[i] = f; env += (double)(i - f);
        }
        if (env > 0.35 * (double)n * n) { free(fst); fst = NULL; }
    }
#define HP_FST(i) (fst ? fst[i] : 0)
    for (int j0 = 0; j0 < n; j0 += kb) {
    const int j1 = j0 + kb < n ? j0 + kb : n;
    for (int j = j0; j < j1; j++) {
        HPT *Lj = EL(A, (size_t)j * n);
        TMP(d); TMP(inv); TMP(one);
        HPN(dot)((size_t)(j - HP_FST(j)), EL(Lj, HP_FST(j)), EL(Lj, HP_FST(j)), d);
        const double ajj = HP_GETD(EL(Lj, j));
        HP_SUB(d, EL(Lj, j), d);
        int skip = 0;
        if (mask) {
            const int small = !(HP_GETD(d) > reg * ajj);
            if (mode == 1 && small) mask[j] = 1;
            if (mode == 2 && mask[j] && !small) mask[j] = 0;
            skip = mask[j];
        } else if (reg > 0 && !(HP_GETD(d) > reg * ajj)) skip = 1;
        if (!skip && minrat && ajj > 0) { const double q = HP_GETD(d) / ajj; if (q < *minrat) *minrat = q; }
        if (skip) {
            HP_SETD(d, (ajj > 0 ? ajj : 1.0) * 1e60);
            if (nskip) (*nskip)++;
        }
        if (HP_SGN(d) <= 0) { free(fst); return j + 1; }
        HP_SQRT(EL(Lj, j), d);
        HP_SETD(one, 1.0); HP_DIV(inv, one, EL(Lj, j));
#pragma omp parallel for schedule(static) if ((double)(j1 - j) * j > 20000.0)
        for (int i = j + 1; i < j1; i++) {
            HPT *Li = EL(A, (size_t)i * n);
            TMP(t);
            if (HP_FST(i) > j) continue;
            const int s0 = HP_FST(i) > HP_FST(j) ? HP_FST(i) : HP_FST(j);
            HPN(dot)((size_t)(j - s0), EL(Li, s0), EL(Lj, s0), t);
            HP_SUB(t, EL(Li, j), t);
            HP_MUL(EL(Li, j), t, inv);
        }
    }
#pragma omp parallel for schedule(static) if (n - j1 >= 2)
    for (int i = j1; i < n; i++) {
        HPT *Li = EL(A, (size_t)i * n);
        TMP(t); TMP(inv); TMP(one);
        HP_SETD(one, 1.0);
        for (int j = j0 > HP_FST(i) ? j0 : HP_FST(i); j < j1; j++) {
            const HPT *Lj = EL(A, (size_t)j * n);
            const int s0 = HP_FST(i) > HP_FST(j) ? HP_FST(i) : HP_FST(j);
            HPN(dot)((size_t)(j - s0), EL(Li, s0), EL(Lj, s0), t);
            HP_SUB(t, EL(Li, j), t);
            HP_DIV(inv, one, EL(Lj, j));
            HP_MUL(EL(Li, j), t, inv);
        }
    }
    }
#undef HP_FST
    free(fst);
    return 0;
}
/* L x = b in place */
static void HPN(fwd)(int n, const HPT *L, HPT *x) {
    for (int i = 0; i < n; i++) {
        const HPT *Li = EL(L, (size_t)i * n);
        TMP(t);
        HPN(dot)((size_t)i, Li, x, t);
        HP_SUB(t, EL(x, i), t);
        HP_DIV(EL(x, i), t, EL(Li, i));
    }
}
/* L' x = b in place */
static void HPN(bwd)(int n, const HPT *L, HPT *x) {
    for (int k = n - 1; k >= 0; k--) {
        const HPT *Lk = EL(L, (size_t)k * n);
        TMP(t);
        HP_DIV(EL(x, k), EL(x, k), EL(Lk, k));
        HP_NEG(t, EL(x, k));
        HPN(axpy)((size_t)k, t, Lk, x);
    }
}
#ifndef HP_OWN_GEMM
/* C = A B (n x n).  Above HP_BLK_MIN the sum over k runs in blocks of HP_KB rows of B, which then
 * stay in the second-level cache while every row of C takes its updates from them (5.3: the
 * plain row-by-row form streams all of B once per row of C; 1.3-1.7 times slower from order
 * 500 on).  The order of the additions is the same, so the result is too. */
#define HP_KB 16
#define HP_BLK_BYTES 2000000.0           /* (the Cholesky factorization above has the same two numbers) */
static void HPN(gemm)(int n, const HPT *A, const HPT *B, HPT *C) {
    const int kb = (double)n * n * sizeof(HPT) > HP_BLK_BYTES ? HP_KB : n;
    HPN(zero)((size_t)n * n, C);
    for (int k0 = 0; k0 < n; k0 += kb) {
        const int k1 = k0 + kb < n ? k0 + kb : n;
#pragma omp parallel for schedule(static) if (n >= 24)
        for (int i = 0; i < n; i++) {
            HPT *Ci = EL(C, (size_t)i * n);
            for (int k = k0; k < k1; k++) {
                const HPT *a = EL(A, (size_t)i * n + k);
                if (HP_SGN(a) != 0) HPN(axpy)((size_t)n, a, EL(B, (size_t)k * n), Ci);
            }
        }
    }
}
#endif
static void HPN(transpose)(int n, HPT *A) {
    TMP(t);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            HPT *a = EL(A, (size_t)i * n + j), *b = EL(A, (size_t)j * n + i);
            HP_COPY(t, a); HP_COPY(a, b); HP_COPY(b, t);
        }
}
/* C = A B for symmetric A and B where B may be sparse (dZ and Rd have the aggregate sparsity
 * pattern of the block: max-cut, truss, ...): with less than a quarter of B nonzero the
 * product is formed as (B A)' from the nonzeros of B, nnz(B) n operations instead of n^3 */
static void HPN(gemm_rs)(int n, const HPT *A, const HPT *B, HPT *C) {
    size_t nz = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) nz += HP_SGN(EL(B, i)) != 0;
    if (4 * nz > (size_t)n * n) { HPN(gemm)(n, A, B, C); return; }
#pragma omp parallel for schedule(static) if (n >= 24)
    for (int i = 0; i < n; i++) {
        HPT *Ci = EL(C, (size_t)i * n);
        HPN(zero)((size_t)n, Ci);
        for (int k = 0; k < n; k++) {
            const HPT *b = EL(B, (size_t)i * n + k);
            if (HP_SGN(b) != 0) HPN(axpy)((size_t)n, b, EL(A, (size_t)k * n), Ci);
        }
    }
    HPN(transpose)(n, C);
}
/* A <- (A + A') / 2 */
static void HPN(symm)(int n, HPT *A) {
    TMP(h); HP_SETD(h, 0.5);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            HPT *a = EL(A, (size_t)i * n + j), *b = EL(A, (size_t)j * n + i);
            HP_ADD(a, a, b); HP_MUL(a, a, h); HP_COPY(b, a);
        }
}
/* each row r of R <- (L^{-1} r')' , i.e. R <- R L^{-T}.  Above the blocking threshold the rows
 * are solved in groups of HP_KB, row i of L serving the whole group before the next one. */
static void HPN(trsm_rows)(int n, const HPT *L, HPT *R) {
    if (!((double)n * n * sizeof(HPT) > HP_BLK_BYTES)) {
#pragma omp parallel for schedule(static) if (n >= 24)
        for (int r = 0; r < n; r++) HPN(fwd)(n, L, EL(R, (size_t)r * n));
        return;
    }
#pragma omp parallel for schedule(static)
    for (int r0 = 0; r0 < n; r0 += HP_KB) {
        const int r1 = r0 + HP_KB < n ? r0 + HP_KB : n;
        for (int i = 0; i < n; i++) {
            const HPT *Li = EL(L, (size_t)i * n);
            for (int r = r0; r < r1; r++) {
                HPT *x = EL(R, (size_t)r * n);
                TMP(t);
                HPN(dot)((size_t)i, Li, x, t);
                HP_SUB(t, EL(x, i), t);
                HP_DIV(EL(x, i), t, EL(Li, i));
            }
        }
    }
}
/* the same for the leading r + 1 entries of row r only (the lower triangle of a result that
 * is symmetric: a third of the work) */
static void HPN(trsm_rows_lower)(int n, const HPT *L, HPT *R) {
    const int g = (double)n * n * sizeof(HPT) > HP_BLK_BYTES ? HP_KB : 1, chunk = g > 1 ? 1 : 8;
#pragma omp parallel for schedule(dynamic, chunk) if (n >= 24)
    for (int r0 = 0; r0 < n; r0 += g) {
        const int r1 = r0 + g < n ? r0 + g : n;
        for (int i = 0; i < r1; i++) {
            const HPT *Li = EL(L, (size_t)i * n);
            for (int r = i > r0 ? i : r0; r < r1; r++) {
                HPT *x = EL(R, (size_t)r * n);
                TMP(t);
                HPN(dot)((size_t)i, Li, x, t);
                HP_SUB(t, EL(x, i), t);
                HP_DIV(EL(x, i), t, EL(Li, i));
            }
        }
    }
}
/* Inv = (L L')^{-1} from the Cholesky factor L (lower); W is n x n work */
static void HPN(inv_from_chol)(int n, const HPT *L, HPT *Inv, HPT *W) {
    /* W = L^{-1} by rows: row_i = (e_i - sum_{k<i} L[i,k] row_k) / L[i,i]; in blocks of rows k
     * above the blocking threshold (a finished block updates all the rows below it) */
    const int kb = (double)n * n * sizeof(HPT) > HP_BLK_BYTES ? HP_KB : n;
    HPN(zero)((size_t)n * n, W);
    for (int k0 = 0; k0 < n; k0 += kb) {
        const int k1 = k0 + kb < n ? k0 + kb : n;
        for (int i = k0; i < k1; i++) {
            HPT *Wi = EL(W, (size_t)i * n);
            const HPT *Li = EL(L, (size_t)i * n);
            TMP(t); TMP(one);
            for (int k = k0; k < i; k++) { HP_NEG(t, EL(Li, k)); HPN(axpy)((size_t)k + 1, t, EL(W, (size_t)k * n), Wi); }
            HP_SETD(one, 1.0); HP_ADD(EL(Wi, i), EL(Wi, i), one);
            HP_DIV(t, one, EL(Li, i));
            HPN(scal)((size_t)i + 1, t, Wi);
        }
#pragma omp parallel for schedule(static) if (n - k1 >= 24)
        for (int i = k1; i < n; i++) {
            HPT *Wi = EL(W, (size_t)i * n);
            const HPT *Li = EL(L, (size_t)i * n);
            TMP(t);
            for (int k = k0; k < k1; k++) { HP_NEG(t, EL(Li, k)); HPN(axpy)((size_t)k + 1, t, EL(W, (size_t)k * n), Wi); }
        }
    }
    /* Inv = W' W: lower triangle by rank-one updates of rows (the same blocks), then mirrored */
    HPN(zero)((size_t)n * n, Inv);
    for (int k0 = 0; k0 < n; k0 += kb) {
        const int k1 = k0 + kb < n ? k0 + kb : n;
#pragma omp parallel for schedule(dynamic, 4) if (n >= 24)
        for (int i = 0; i < k1; i++) {
            HPT *Ii = EL(Inv, (size_t)i * n);
            for (int k = i > k0 ? i : k0; k < k1; k++) {
                const HPT *Wk = EL(W, (size_t)k * n);
                HPN(axpy)((size_t)i + 1, EL(Wk, i), Wk, Ii);
            }
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) HP_COPY(EL(Inv, (size_t)j * n + i), EL(Inv, (size_t)i * n + j));
}

/* ------------------------------------------------------------------ the solver state */
typedef struct {
    HPJob *J;
    int m, nb;
    HPT *Av, *Cv, *b;
    int *n, *lp;
    size_t *len;
    HPT **X, **Z, **Zi, **dX, **dZ, **dXa, **dZa, **Lx, **Lz, **K, **T1, **T2, **Rd, **Xb, **Zb, **Xh, **Zh;
    HPT *y, *dy, *yb, *yh, *rp, *rhs, *M, *v1, *v2, *vp;
    double *wd, *ev, *lw;
    int *mask;                   /* rows skipped in the current Schur factorization */
} HPN(st);

/* out_p = <A_p, D> for all p (D need not be symmetric) */
static void HPN(Aop)(const HPN(st) *S, HPT *const *D, HPT *out) {
    const HPJob *J = S->J;
    HPN(zero)((size_t)S->m, out);
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        for (int t = 0; t < J->ncon[k]; t++) {
            HPT *o = EL(out, J->con[k][t]);
            for (size_t e = J->ptr[k][t]; e < J->ptr[k][t + 1]; e++) {
                const int i = J->ei[e], j = J->ej[e];
                const HPT *v = EL(S->Av, e);
                if (S->lp[k]) HP_ADDMUL(o, v, EL(D[k], i));
                else if (i == j) HP_ADDMUL(o, v, EL(D[k], (size_t)i * n + i));
                else { TMP(s); HP_ADD(s, EL(D[k], (size_t)i * n + j), EL(D[k], (size_t)j * n + i)); HP_ADDMUL(o, v, s); }
            }
        }
    }
}
/* F = sum_p y_p A_p */
static void HPN(ATy)(const HPN(st) *S, const HPT *y, HPT **F) {
    const HPJob *J = S->J;
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        HPN(zero)(S->len[k], F[k]);
        for (int t = 0; t < J->ncon[k]; t++) {
            const HPT *yp = EL(y, J->con[k][t]);
            if (HP_SGN(yp) == 0) continue;
            for (size_t e = J->ptr[k][t]; e < J->ptr[k][t + 1]; e++) {
                const int i = J->ei[e], j = J->ej[e];
                const HPT *v = EL(S->Av, e);
                if (S->lp[k]) HP_ADDMUL(EL(F[k], i), yp, v);
                else {
                    HP_ADDMUL(EL(F[k], (size_t)i * n + j), yp, v);
                    if (i != j) HP_ADDMUL(EL(F[k], (size_t)j * n + i), yp, v);
                }
            }
        }
    }
}
/* F += s C  (s = +1 or -1) */
static void HPN(addC)(const HPN(st) *S, HPT **F, int s) {
    const HPJob *J = S->J;
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
            const int i = J->ci[e], j = J->cj[e];
            TMP(v); HP_COPY(v, EL(S->Cv, e)); if (s < 0) HP_NEG(v, v);
            if (S->lp[k]) HP_ADD(EL(F[k], i), EL(F[k], i), v);
            else {
                HP_ADD(EL(F[k], (size_t)i * n + j), EL(F[k], (size_t)i * n + j), v);
                if (i != j) HP_ADD(EL(F[k], (size_t)j * n + i), EL(F[k], (size_t)j * n + i), v);
            }
        }
    }
}
/* r = sum_k <A_k, B_k> (elementwise) */
static void HPN(inner)(const HPN(st) *S, HPT *const *A, HPT *const *B, HPT *r) {
    TMP(s); TMP(t); HP_SETD(s, 0.0);
    for (int k = 0; k < S->nb; k++) { HPN(dot)(S->len[k], A[k], B[k], t); HP_ADD(s, s, t); }
    HP_COPY(r, s);
}
/* (the root is taken in the working precision: the square of an error can be below the double range) */
static double HPN(norm)(size_t n, const HPT *x) { TMP(t); HPN(dot)(n, x, x, t); if (HP_SGN(t) <= 0) return 0.0; HP_SQRT(t, t); return HP_GETD(t); }
static double HPN(normb)(const HPN(st) *S, HPT *const *A) { TMP(t); HPN(inner)(S, A, A, t); if (HP_SGN(t) <= 0) return 0.0; HP_SQRT(t, t); return HP_GETD(t); }

/* the Schur complement M_pq = sum_blocks tr(A_p X A_q Z^{-1}), lower triangle, row-major */
static void HPN(schur)(HPN(st) *S) {
    const HPJob *J = S->J;
    const int m = S->m;
    if (J->mfst) { for (int i = 0; i < m; i++) HPN(zero)((size_t)(i + 1 - J->mfst[i]), EL(S->M, (size_t)i * m + J->mfst[i])); }
    else HPN(zero)((size_t)m * m, S->M);
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k], nc = J->ncon[k];
        if (S->lp[k]) {
            for (int i = 0; i < n; i++) {
                TMP(w); HP_DIV(w, EL(S->X[k], i), EL(S->Z[k], i));
                for (int a = J->lpptr[k][i]; a < J->lpptr[k][i + 1]; a++) {
                    TMP(wa); HP_MUL(wa, w, EL(S->Av, J->lpe[k][a]));
                    const int p = J->lpc[k][a];
                    for (int c = J->lpptr[k][i]; c <= a; c++) {
                        const int q = J->lpc[k][c];
                        const int pr = J->mpos ? J->mpos[p] : p, qr = J->mpos ? J->mpos[q] : q;
                        HPT *o = pr >= qr ? EL(S->M, (size_t)pr * m + qr) : EL(S->M, (size_t)qr * m + pr);
                        HP_ADDMUL(o, wa, EL(S->Av, J->lpe[k][c]));
                    }
                }
            }
            continue;
        }
        const int *ord = J->kord[k], *cp = J->kcolptr[k], *cl = J->kcol[k];
        double *suf = (double *)malloc(sizeof(double) * (size_t)(nc + 1));
        suf[nc] = 0;
        for (int r = nc - 1; r >= 0; r--) suf[r] = suf[r + 1] + 2.0 * (double)(J->ptr[k][ord[r] + 1] - J->ptr[k][ord[r]]);
        const HPT *X = S->X[k], *Zi = S->Zi[k];
        size_t maxlen = 0;
        for (int t = 0; t < nc; t++) if ((size_t)(J->ptr[k][t + 1] - J->ptr[k][t]) > maxlen) maxlen = J->ptr[k][t + 1] - J->ptr[k][t];
#pragma omp parallel
        {
            HPT *Tc = HPN(alloc)((size_t)n * n), *U = HPN(alloc)((size_t)n * n);
            HPT *TcT = HPN(alloc)((size_t)n * n), *ZcT = HPN(alloc)((size_t)n * n), *buf = HPN(alloc)(maxlen + 1);
            int *cpos = (int *)malloc(sizeof(int) * (size_t)n);
#pragma omp for schedule(dynamic, 1)
            for (int r = 0; r < nc; r++) {
                const int t = ord[r], q = J->con[k][t], ncol = cp[t + 1] - cp[t];
                const int *cols = cl + cp[t];
                HPN(zero)((size_t)ncol * n, Tc);
                for (int c = 0; c < ncol; c++) cpos[cols[c]] = c;
                for (size_t e = J->ptr[k][t]; e < J->ptr[k][t + 1]; e++) {
                    const int i = J->ei[e], j = J->ej[e];
                    const HPT *v = EL(S->Av, e);
                    HPN(axpy)((size_t)n, v, EL(X, (size_t)i * n), EL(Tc, (size_t)cpos[j] * n));
                    if (i != j) HPN(axpy)((size_t)n, v, EL(X, (size_t)j * n), EL(Tc, (size_t)cpos[i] * n));
                }
                const int dense = (double)n * n * ncol + suf[r] < (double)ncol * suf[r];
                if (dense)
                    for (int i = 0; i < n; i++) {
                        HPT *Ui = EL(U, (size_t)i * n);
                        HPN(zero)((size_t)n, Ui);
                        for (int c = 0; c < ncol; c++) HPN(axpy)((size_t)n, EL(Tc, (size_t)c * n + i), EL(Zi, (size_t)cols[c] * n), Ui);
                    }
                /* The sums over the entries of the other constraints are dot products of
                 * contiguous vectors (5.3; they were scalar loops, which in double-double cost
                 * 7 ns per term against 0.8 in the vector kernel and were two thirds of the
                 * solve time of the control and moment problems):
                 *   dense: U <- U + U' off the diagonal, the entries gathered into buf;
                 *   sparse, 4 columns or more: Tc and the rows of Zi transposed (n x ncol), so
                 *   that entry (i, j) is the dot of row i of one with row j of the other. */
                const int vec = !dense && ncol >= 4;
                if (dense)
                    for (int i = 0; i < n; i++)
                        for (int j = i + 1; j < n; j++) {
                            HPT *a = EL(U, (size_t)i * n + j), *b = EL(U, (size_t)j * n + i);
                            HP_ADD(a, a, b); HP_COPY(b, a);
                        }
                else if (vec)
                    for (int i = 0; i < n; i++)
                        for (int c = 0; c < ncol; c++) {
                            HP_COPY(EL(TcT, (size_t)i * ncol + c), EL(Tc, (size_t)c * n + i));
                            HP_COPY(EL(ZcT, (size_t)i * ncol + c), EL(Zi, (size_t)i * n + cols[c]));
                        }
                for (int r2 = r; r2 < nc; r2++) {
                    const int t2 = ord[r2], p = J->con[k][t2];
                    const size_t e0 = J->ptr[k][t2], len = J->ptr[k][t2 + 1] - e0;
                    TMP(acc); HP_SETD(acc, 0.0);
                    if (dense && len >= 4) {
                        for (size_t e = 0; e < len; e++) HP_COPY(EL(buf, e), EL(U, (size_t)J->ei[e0 + e] * n + J->ej[e0 + e]));
                        HPN(dot)(len, EL(S->Av, e0), buf, acc);
                    } else if (dense) {
                        for (size_t e = e0; e < e0 + len; e++) HP_ADDMUL(acc, EL(S->Av, e), EL(U, (size_t)J->ei[e] * n + J->ej[e]));
                    } else if (vec) {
                        for (size_t e = 0; e < len; e++) {
                            const int i = J->ei[e0 + e], j = J->ej[e0 + e];
                            HPN(dot)((size_t)ncol, EL(TcT, (size_t)i * ncol), EL(ZcT, (size_t)j * ncol), EL(buf, e));
                            if (i != j) { TMP(u); HPN(dot)((size_t)ncol, EL(TcT, (size_t)j * ncol), EL(ZcT, (size_t)i * ncol), u); HP_ADD(EL(buf, e), EL(buf, e), u); }
                        }
                        if (len >= 4) HPN(dot)(len, EL(S->Av, e0), buf, acc);
                        else for (size_t e = 0; e < len; e++) HP_ADDMUL(acc, EL(S->Av, e0 + e), EL(buf, e));
                    } else
                        for (size_t e = e0; e < e0 + len; e++) {
                            const int i = J->ei[e], j = J->ej[e];
                            TMP(u); HP_SETD(u, 0.0);
                            for (int c = 0; c < ncol; c++) {
                                HP_ADDMUL(u, EL(Tc, (size_t)c * n + i), EL(Zi, (size_t)cols[c] * n + j));
                                if (i != j) HP_ADDMUL(u, EL(Tc, (size_t)c * n + j), EL(Zi, (size_t)cols[c] * n + i));
                            }
                            HP_ADDMUL(acc, EL(S->Av, e), u);
                        }
                    const int pr = J->mpos ? J->mpos[p] : p, qr = J->mpos ? J->mpos[q] : q;
                    HPT *o = pr >= qr ? EL(S->M, (size_t)pr * m + qr) : EL(S->M, (size_t)qr * m + pr);
                    HP_ADD(o, o, acc);
                }
            }
            free(TcT); free(ZcT); free(buf);
            free(Tc); free(U); free(cpos);
        }
        free(suf);
    }
}

/* lambda_min(L^{-1} D L^{-T}) by the Lanczos method, for the step length of a large block:
 * a step costs two triangular solves and one product with D (2 n^2 operations in the working
 * precision) against 2/3 n^3 for the matrix itself.  The vectors are kept in double with full
 * reorthogonalization (the operator is applied in the working precision, so the accuracy is
 * that of the dense path, which finds the eigenvalues in double too).  Stops when the residual
 * bound delta of the smallest Ritz value theta is 1e-10 of it (the predictor decides whether
 * the full step is feasible, to 1e-9) and returns theta - delta.  *ok = 0: not converged in HP_LANMAX
 * steps or a spectrum too wide for double; the caller then takes the dense path. */
#define HP_LANMAX 60
static double HPN(lanczos_min)(HPN(st) *S, int k, const HPT *L, const HPT *D, int *ok) {
    const int n = S->n[k], mx = n - 1 < HP_LANMAX ? n - 1 : HP_LANMAX;
    double *Q = S->wd, al[HP_LANMAX + 1], be[HP_LANMAX + 1], T[(HP_LANMAX + 1) * (HP_LANMAX + 1)], ev[HP_LANMAX + 1], wk[3 * HP_LANMAX + 16];
    HPT *u = HPN(alloc)((size_t)n), *w = HPN(alloc)((size_t)n);
    double *r = (double *)malloc(sizeof(double) * (size_t)n), res = 0.0;
    uint64_t sd = 0x9E3779B97F4A7C15ULL;
    double nr = 0.0;
    const char *et = getenv("BRISK_HPLANTOL"); const double ltol = et ? atof(et) : 1e-10;
    *ok = 0;
    for (int i = 0; i < n; i++) { sd = sd * 6364136223846793005ULL + 1442695040888963407ULL; Q[i] = (double)(sd >> 11) / 9007199254740992.0 - 0.5; nr += Q[i] * Q[i]; }
    nr = sqrt(nr); for (int i = 0; i < n; i++) Q[i] /= nr;
    for (int j = 0; j < mx; j++) {
        const double *q = Q + (size_t)j * n;
        for (int i = 0; i < n; i++) HP_SETD(EL(u, i), q[i]);
        HPN(bwd)(n, L, u);
#pragma omp parallel for schedule(static) if (n >= 200)
        for (int i = 0; i < n; i++) HPN(dot)((size_t)n, EL(D, (size_t)i * n), u, EL(w, i));
        HPN(fwd)(n, L, w);
        for (int i = 0; i < n; i++) r[i] = HP_GETD(EL(w, i));
        double a = 0.0;
        for (int i = 0; i < n; i++) a += q[i] * r[i];
        al[j] = a;
        for (int pass = 0; pass < 2; pass++)
            for (int l = 0; l <= j; l++) {
                const double *ql = Q + (size_t)l * n;
                double c = 0.0;
                for (int i = 0; i < n; i++) c += ql[i] * r[i];
                for (int i = 0; i < n; i++) r[i] -= c * ql[i];
            }
        double b = 0.0;
        for (int i = 0; i < n; i++) b += r[i] * r[i];
        b = sqrt(b); be[j] = b;
        if (!(b == b) || !(a == a)) break;
        /* the tridiagonal matrix of order j + 1: smallest and largest Ritz values, residual bound */
        const int t = j + 1;
        memset(T, 0, sizeof(double) * (size_t)t * t);
        for (int i = 0; i < t; i++) { T[(size_t)i * t + i] = al[i]; if (i + 1 < t) T[(size_t)i * t + i + 1] = T[(size_t)(i + 1) * t + i] = be[i]; }
        int info = 0, lw = 3 * HP_LANMAX + 16;
        BL(dsyev_)("V", "U", &t, T, &t, ev, wk, &lw, &info);
        if (info) break;
        const double th = ev[0], dl = b * fabs(T[t - 1]);          /* last entry of the first eigenvector */
        if (fmax(fabs(ev[t - 1]), fabs(th)) >= 1e9 * fmax(fabs(th), 1e-3)) break;
        if (dl <= ltol * fmax(fabs(th - dl), 0.1) || b <= 1e-13 * fmax(fabs(ev[t - 1]), fabs(th))) { *ok = 1; res = th - dl; if (getenv("BRISK_HPDBG")) printf("       [Lanczos: %d steps]\n", t); break; }
        double *qn = Q + (size_t)(j + 1) * n;
        for (int i = 0; i < n; i++) qn[i] = r[i] / b;
    }
    free(u); free(w); free(r);
    return res;
}
/* largest alpha with X + alpha D >= 0: -1 / lambda_min(L^{-1} D L^{-T}); the eigenvalue in
 * double (the matrix is formed in the working precision) */
static double HPN(maxstep)(HPN(st) *S, int k, const HPT *L, const HPT *D, const HPT *x) {
    const int n = S->n[k];
    if (S->lp[k]) {
        double a = 1e300;
        for (int i = 0; i < n; i++)
            if (HP_SGN(EL(D, i)) < 0) { TMP(t); HP_DIV(t, EL(x, i), EL(D, i)); const double r = -HP_GETD(t); if (r < a) a = r; }
        return a;
    }
    {
        const char *el = getenv("BRISK_HPLAN");
        const int nl = el ? atoi(el) : 200;                  /* 0: never */
        if (nl > 0 && n >= nl && n >= 8) {
            int ok;
            const double lam = HPN(lanczos_min)(S, k, L, D, &ok);
            if (getenv("BRISK_HPDBG")) printf("       [maxstep blk %d: Lanczos ok %d lmin %.3e]\n", k, ok, lam);
            if (ok) return lam < 0 ? -1.0 / lam : 1e300;
        }
    }
    HPT *R = S->T2[k];
    HPN(copy)((size_t)n * n, D, R);
    HPN(trsm_rows)(n, L, R);
    HPN(transpose)(n, R);
    HPN(trsm_rows_lower)(n, L, R);                      /* (dsyev below reads this triangle only) */
    for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) S->wd[(size_t)i * n + j] = HP_GETD(EL(R, (size_t)i * n + j));
    int info = 0, lw = 3 * n + 8;
    BL(dsyev_)("N", "U", &n, S->wd, &n, S->ev, S->lw, &lw, &info);
    const double lmin = S->ev[0], lmax = fmax(fabs(S->ev[n - 1]), fabs(S->ev[0]));
    if (getenv("BRISK_HPDBG")) printf("       [maxstep blk %d: info %d lmin %.3e lmax %.3e]\n", k, info, lmin, S->ev[n - 1]);
    if (info == 0 && lmin == lmin && lmax < 1e9 * fmax(fabs(lmin), 1e-3)) return lmin < 0 ? -1.0 / lmin : 1e300;
    /* the spectrum is too wide for double (a degenerate problem: huge positive eigenvalues
     * beside the one that matters): bisection on alpha with Cholesky in the working precision */
    {
        const double mx = getenv("BRISK_HPNOBIS") ? 0.0 : 1.0;
        if (mx == 0.0) return info == 0 && lmin < 0 ? -1.0 / lmin : info == 0 ? 1e300 : 0.0;
        double lo = 0.0, hi = 1.0 / 0.9;                     /* alpha beyond 1/gamma_min is never used */
        TMP(a); TMP(t);
        for (int pass = 0; pass < 14; pass++) {
            const double al = pass == 0 ? hi : 0.5 * (lo + hi);
            HP_SETD(a, al);
            for (size_t i = 0; i < (size_t)n * n; i++) { HP_MUL(t, a, EL(D, i)); HP_ADD(EL(R, i), EL(x, i), t); }
            if (HPN_chol(n, R) == 0) { lo = al; if (pass == 0) return 1e300; } else hi = al;
        }
        S->J->nbis++;
        return lo;
    }
}

/* x <- M^{-1} x with the Schur factor, which is stored in the rows of J->mpos; within the
 * envelope when the first columns are known (J->mfst) */
static void HPN(msolve)(HPN(st) *S, HPT *x) {
    const int m = S->m, *pos = S->J->mpos, *fst = S->J->mfst;
    HPT *v = x;
    if (pos) { v = S->vp; for (int p = 0; p < m; p++) HP_COPY(EL(v, pos[p]), EL(x, p)); }
    if (!fst) { HPN(fwd)(m, S->M, v); HPN(bwd)(m, S->M, v); }
    else {
        for (int i = 0; i < m; i++) {
            const HPT *Li = EL(S->M, (size_t)i * m);
            TMP(t);
            HPN(dot)((size_t)(i - fst[i]), EL(Li, fst[i]), EL(v, fst[i]), t);
            HP_SUB(t, EL(v, i), t);
            HP_DIV(EL(v, i), t, EL(Li, i));
        }
        for (int k = m - 1; k >= 0; k--) {
            const HPT *Lk = EL(S->M, (size_t)k * m);
            TMP(t);
            HP_DIV(EL(v, k), EL(v, k), EL(Lk, k));
            HP_NEG(t, EL(v, k));
            HPN(axpy)((size_t)(k - fst[k]), t, EL(Lk, fst[k]), EL(v, fst[k]));
        }
    }
    if (pos) for (int p = 0; p < m; p++) HP_COPY(EL(x, p), EL(S->vp, pos[p]));
}
/* 1 if W + a dW is positive definite for every block (T2 is work) */
static int HPN(pdtest)(HPN(st) *S, HPT **W, HPT **dW, const HPT *a) {
    for (int k = 0; k < S->nb; k++) {
        TMP(t);
        for (size_t i = 0; i < S->len[k]; i++) { HP_MUL(t, a, EL(dW[k], i)); HP_ADD(EL(S->T2[k], i), EL(W[k], i), t); }
        if (S->lp[k]) { for (int i = 0; i < S->n[k]; i++) if (HP_SGN(EL(S->T2[k], i)) <= 0) return 0; }
        else if (HPN_chol(S->n[k], S->T2[k])) return 0;
    }
    return 1;
}
/* dX (into S->dX) for a given dy, with sigma*mu (smu) and the Mehrotra term (corr):
 *   dZ = Rd - A*(dy);  dX = smu Zi - X - sym(X dZ Zi) - [corr] sym(dXa dZa Zi) */
static void HPN(direction)(HPN(st) *S, const HPT *dy, const HPT *smu, int corr) {
    HPN(ATy)(S, dy, S->dZ);
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        TMP(t);
        for (size_t i = 0; i < S->len[k]; i++) HP_SUB(EL(S->dZ[k], i), EL(S->Rd[k], i), EL(S->dZ[k], i));
        if (S->lp[k]) {
            for (int i = 0; i < n; i++) {
                TMP(u);
                HP_MUL(u, EL(S->X[k], i), EL(S->dZ[k], i));                 /* x dz */
                if (corr) HP_ADDMUL(u, EL(S->dXa[k], i), EL(S->dZa[k], i)); /* + dxa dza */
                HP_SUB(u, smu, u);
                HP_DIV(u, u, EL(S->Z[k], i));
                HP_SUB(EL(S->dX[k], i), u, EL(S->X[k], i));
            }
            continue;
        }
        /* T1 = X dZ (+ dXa dZa), dX = smu Zi - X - sym(T1 Zi) */
        HPN(gemm_rs)(n, S->X[k], S->dZ[k], S->T1[k]);
        if (corr) for (size_t i = 0; i < S->len[k]; i++) HP_ADD(EL(S->T1[k], i), EL(S->T1[k], i), EL(S->K[k], i));   /* K = dXa dZa (make_rhs) */
        HPN(gemm)(n, S->T1[k], S->Zi[k], S->dX[k]);
        HPN(symm)(n, S->dX[k]);
        for (size_t i = 0; i < S->len[k]; i++) {
            HP_MUL(t, smu, EL(S->Zi[k], i));
            HP_SUB(t, t, EL(S->X[k], i));
            HP_SUB(EL(S->dX[k], i), t, EL(S->dX[k], i));
        }
    }
}
/* rhs of the Schur system: b - A(smu Zi) + A(X Rd Zi) [K, precomputed] + [corr] A(dXa dZa Zi) */
static void HPN(make_rhs)(HPN(st) *S, const HPT *smu, int corr, const HPT *AK) {
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        if (S->lp[k]) {
            for (int i = 0; i < n; i++) {
                TMP(u); HP_NEG(u, smu);
                if (corr) HP_ADDMUL(u, EL(S->dXa[k], i), EL(S->dZa[k], i));
                HP_DIV(EL(S->T1[k], i), u, EL(S->Z[k], i));
            }
            continue;
        }
        if (corr) {
            HPN(gemm_rs)(n, S->dXa[k], S->dZa[k], S->K[k]);         /* kept for the direction */
            HPN(gemm)(n, S->K[k], S->Zi[k], S->T1[k]);
        } else HPN(zero)(S->len[k], S->T1[k]);
        for (size_t i = 0; i < S->len[k]; i++) HP_SUBMUL(EL(S->T1[k], i), smu, EL(S->Zi[k], i));
    }
    HPN(Aop)(S, S->T1, S->rhs);
    for (int p = 0; p < S->m; p++) {
        HP_ADD(EL(S->rhs, p), EL(S->rhs, p), EL(S->b, p));
        HP_ADD(EL(S->rhs, p), EL(S->rhs, p), EL(AK, p));
    }
}

/* Refinement of a Schur solve: A(dX) = rp is the Schur system itself, so its residual is read
 * off the direction (a sparse product); the correction is solved with the factor at hand and
 * the direction recomputed, until the residual (relative to 1 + |b|) is below rtol, stops
 * halving, or maxref rounds are done. Returns the last residual. */
static double HPN(refine)(HPN(st) *S, const HPT *smu, int corr, int maxref, double rtol, double nb_) {
    const int m = S->m;
    double eprev = 1e300, en = 0;
    for (int ref = 0; ; ref++) {
        HPN(Aop)(S, S->dX, S->v1);
        for (int p = 0; p < m; p++) HP_SUB(EL(S->v1, p), EL(S->rp, p), EL(S->v1, p));
        en = HPN(norm)((size_t)m, S->v1) / (1 + nb_);
        if (getenv("BRISK_HPDBG")) printf("       [schur residual %.2e (target %.2e)]\n", en, rtol);
        if (en <= rtol || ref == maxref || en >= 0.5 * eprev) break;
        eprev = en;
        HPN(msolve)(S, S->v1);
        TMP(one_); HP_SETD(one_, 1.0);
        HPN(axpy)((size_t)m, one_, S->v1, S->dy);
        HPN(direction)(S, S->dy, smu, corr);
    }
    return en;
}
static const char *HPN(name) = HP_NAME;

/* Runs the method in this precision from the exchange iterate of J (or from a cold start).
 * Returns 0: tolerance reached; 1: this precision is exhausted (or its level target is
 * reached) and the best iterate is exported; 2: iteration or time limit; 3: failure. */
static int HPN(ipm)(HPJob *J) {
    HPN(st) S_, *S = &S_;
    memset(S, 0, sizeof(*S));
    S->J = J; S->m = J->m; S->nb = J->nblk;
    const int m = S->m, nb = S->nb;
    const size_t Wm = (size_t)J->Lm + 2;
    S->Av = HPN(alloc)(J->nnzA); S->Cv = HPN(alloc)(J->nnzC); S->b = HPN(alloc)((size_t)m);
    for (size_t e = 0; e < J->nnzA; e++) HP_IMPORT(EL(S->Av, e), J->Av + e * Wm, J);
    for (size_t e = 0; e < J->nnzC; e++) HP_IMPORT(EL(S->Cv, e), J->Cv + e * Wm, J);
    for (int p = 0; p < m; p++) HP_IMPORT(EL(S->b, p), J->b + (size_t)p * Wm, J);
    S->n = (int *)malloc(sizeof(int) * (size_t)nb); S->lp = (int *)malloc(sizeof(int) * (size_t)nb);
    S->len = (size_t *)malloc(sizeof(size_t) * (size_t)nb);
    HPT ***arr[] = { &S->X, &S->Z, &S->Zi, &S->dX, &S->dZ, &S->dXa, &S->dZa, &S->Lx, &S->Lz, &S->K, &S->T1, &S->T2, &S->Rd, &S->Xb, &S->Zb, &S->Xh, &S->Zh };
    const int narr = (int)(sizeof(arr) / sizeof(arr[0]));
    int nmax = 1; double N = 0;
    for (int a = 0; a < narr; a++) *arr[a] = (HPT **)malloc(sizeof(HPT *) * (size_t)nb);
    for (int k = 0; k < nb; k++) {
        S->lp[k] = J->bs[k] < 0; S->n[k] = abs(J->bs[k]);
        S->len[k] = S->lp[k] ? (size_t)S->n[k] : (size_t)S->n[k] * S->n[k];
        if (!S->lp[k] && S->n[k] > nmax) nmax = S->n[k];
        N += S->n[k];
        for (int a = 0; a < narr; a++) (*arr[a])[k] = HPN(alloc)(S->len[k]);
    }
    S->y = HPN(alloc)((size_t)m); S->dy = HPN(alloc)((size_t)m); S->yb = HPN(alloc)((size_t)m); S->yh = HPN(alloc)((size_t)m);
    S->rp = HPN(alloc)((size_t)m); S->rhs = HPN(alloc)((size_t)m); S->v1 = HPN(alloc)((size_t)m); S->v2 = HPN(alloc)((size_t)m);
    S->M = HPN(alloc)((size_t)m * m); S->vp = HPN(alloc)((size_t)m);
    S->mask = (int *)calloc((size_t)m + 1, sizeof(int));
    S->wd = (double *)malloc(sizeof(double) * (size_t)nmax * nmax);
    S->ev = (double *)malloc(sizeof(double) * (size_t)nmax);
    S->lw = (double *)malloc(sizeof(double) * (size_t)(3 * nmax + 8));

    /* norms of the data */
    const double nb_ = HPN(norm)((size_t)m, S->b), nC = HPN(norm)(J->nnzC, S->Cv);

    if (J->have_iter) {
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < S->len[k]; i++) { HP_IMPORT(EL(S->X[k], i), J->X[k] + i * Wm, J); HP_IMPORT(EL(S->Z[k], i), J->Z[k] + i * Wm, J); }
        for (int p = 0; p < m; p++) HP_IMPORT(EL(S->y, p), J->y + (size_t)p * Wm, J);
    } else {
        /* cold start (SDPT3's rule): X = xi I, Z = eta I per block, y = 0 */
        for (int k = 0; k < nb; k++) {
            const int n = S->n[k];
            double xi = 0, eta = 0;
            for (int t = 0; t < J->ncon[k]; t++) {
                double s = 0;
                for (size_t e = J->ptr[k][t]; e < J->ptr[k][t + 1]; e++) { const double v = HP_GETD(EL(S->Av, e)); s += (J->ei[e] == J->ej[e] ? 1.0 : 2.0) * v * v; }
                s = sqrt(s);
                const double bp = fabs(HP_GETD(EL(S->b, J->con[k][t])));
                if ((1 + bp) / (1 + s) > xi) xi = (1 + bp) / (1 + s);
                if (s > eta) eta = s;
            }
            double sc = 0;
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) { const double v = HP_GETD(EL(S->Cv, e)); sc += (J->ci[e] == J->cj[e] ? 1.0 : 2.0) * v * v; }
            if (sqrt(sc) > eta) eta = sqrt(sc);
            const double sq = S->lp[k] ? 1.0 : sqrt((double)n);
            xi = fmax(10.0, fmax(sq, sq * xi)); eta = fmax(10.0, fmax(sq, eta));
            for (int i = 0; i < n; i++) {
                const size_t d = S->lp[k] ? (size_t)i : (size_t)i * n + i;
                HP_SETD(EL(S->X[k], d), xi); HP_SETD(EL(S->Z[k], d), eta);
            }
        }
    }
    int rc = 2;
    /* factors of the starting point */
    for (int k = 0; k < nb; k++) {
        if (S->lp[k]) { for (int i = 0; i < S->n[k]; i++) if (HP_SGN(EL(S->X[k], i)) <= 0 || HP_SGN(EL(S->Z[k], i)) <= 0) rc = 3; continue; }
        HPN(copy)(S->len[k], S->X[k], S->Lx[k]); HPN(copy)(S->len[k], S->Z[k], S->Lz[k]);
        if (HPN_chol(S->n[k], S->Lx[k]) || HPN_chol(S->n[k], S->Lz[k])) rc = 3;
    }
    if (rc == 3) { if (J->verbose) printf("high precision (%s): the starting point is not positive definite\n", HPN(name)); goto done; }

    TMP(mu); TMP(pobj); TMP(dobj); TMP(smu); TMP(t); TMP(zero_); TMP(xz);
    HP_SETD(zero_, 0.0);
    double best = 1e300, ap = 0, ad = 0, sigma = 0;
    int nbad = 0, have_best = 0, it, stall_it = 0, cen_left = 0, cen_done = 0, cen_ok = 1;
    double cen_retry = 0, last_delta = 1.0, sn3 = 0;
    double cen_rho = 0;
    int redo = 0, frozen_ok = 1, cen_gaveup = 0, nsick = 0, have_h = 0, at_floor = 0;
    double besth = 1e300, hm[5] = { 0, 0, 0, 0, 0 };
    TMP(hpo); TMP(hdo); HP_SETD(hpo, 0.0); HP_SETD(hdo, 0.0);
    J->sick_exit = 0; J->chol_exit = 0;
    for (int k = 0; k < nb; k++) if (!S->lp[k]) sn3 += (double)S->n[k] * S->n[k] * S->n[k];
    double stall_ref = 1e300, min_err = 1e300;
    for (it = 0; ; it++) {
        /* residuals, measures */
        HPN(Aop)(S, S->X, S->v1);
        for (int p = 0; p < m; p++) HP_SUB(EL(S->rp, p), EL(S->b, p), EL(S->v1, p));
        HPN(ATy)(S, S->y, S->Rd);
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < S->len[k]; i++) { HP_ADD(EL(S->Rd[k], i), EL(S->Rd[k], i), EL(S->Z[k], i)); HP_NEG(EL(S->Rd[k], i), EL(S->Rd[k], i)); }
        HPN(addC)(S, S->Rd, +1);                                   /* Rd = C - A*(y) - Z */
        HPN(inner)(S, S->X, S->Z, xz);
        TMP(nn); HP_SETD(nn, N); HP_DIV(mu, xz, nn);
        HPN(dot)((size_t)m, S->b, S->y, dobj);
        HP_SETD(pobj, 0.0);                                        /* <C, X> from the sparse C */
        for (int k = 0; k < nb; k++)
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
                const int i = J->ci[e], j = J->cj[e], n = S->n[k];
                if (S->lp[k]) HP_ADDMUL(pobj, EL(S->Cv, e), EL(S->X[k], i));
                else if (i == j) HP_ADDMUL(pobj, EL(S->Cv, e), EL(S->X[k], (size_t)i * n + i));
                else { TMP(u); HP_ADD(u, EL(S->X[k], (size_t)i * n + j), EL(S->X[k], (size_t)j * n + i)); HP_ADDMUL(pobj, EL(S->Cv, e), u); }
            }
        const double po = HP_GETD(pobj), dob = HP_GETD(dobj);
        const double pinf = HPN(norm)((size_t)m, S->rp) / (1 + nb_), dinf = HPN(normb)(S, S->Rd) / (1 + nC);
        HP_SUB(t, pobj, dobj);
        const double den = 1 + fabs(po) + fabs(dob), gap = HP_GETD(t) / den, compl = HP_GETD(xz) / den;
        const double err = fmax(fmax(pinf, dinf), fmax(fabs(gap), compl));
        J->pinf = pinf; J->dinf = dinf; J->gap = gap; J->compl = compl; J->err = err;
        if (J->verbose > 1 && !redo)
            printf("%4d %-3s %+.14e %+.14e %8.1e %8.1e %9.1e %8.1e %6.3f %6.3f %7.1e %7.1fs\n", J->iters, HPN(name), po, dob, pinf, dinf, gap,
                   HP_GETD(mu), ap, ad, sigma, wtime() - J->tstart);
        if (err < best) {
            best = err; have_best = 1;
            for (int k = 0; k < nb; k++) { HPN(copy)(S->len[k], S->X[k], S->Xb[k]); HPN(copy)(S->len[k], S->Z[k], S->Zb[k]); }
            HPN(copy)((size_t)m, S->y, S->yb);
            HP_EXPORT(J->pobj, pobj, J); HP_EXPORT(J->dobj, dobj, J);
            J->best_pinf = pinf; J->best_dinf = dinf; J->best_gap = gap; J->best_compl = compl; J->best_err = err;
        }
        /* The handover iterate (when a higher precision can follow): the best one whose error
         * still is its complementarity (<X,Z> at least a tenth of the error, and no negative
         * gap to speak of). On a
         * problem without an interior, or with multipliers that diverge (qap), a level ends
         * with mu below the infeasibility it cannot remove and a gap that is y'(b - A(X)); the
         * next precision makes no step to speak of from such a point. */
        if (J->more && compl >= 0.1 * err && gap > -0.1 * compl && err < besth) {
            besth = err; have_h = 1;
            for (int k = 0; k < nb; k++) { HPN(copy)(S->len[k], S->X[k], S->Xh[k]); HPN(copy)(S->len[k], S->Z[k], S->Zh[k]); }
            HPN(copy)((size_t)m, S->y, S->yh);
            HP_COPY(hpo, pobj); HP_COPY(hdo, dobj);
            hm[0] = pinf; hm[1] = dinf; hm[2] = gap; hm[3] = compl; hm[4] = err;
        }
        /* Anchors of the ex-post certificate (hpcert.inc): iterates on a ladder of errors, each
         * 1e4 below the last, kept while the error still is the complementarity (central
         * points: their X and Z(y) are positive definite with room). The newest HP_NANC. */
        if (J->anc && compl >= 0.1 * err && gap > -0.1 * compl && err <= J->anc_thr && pinf <= 0.1 * err && dinf <= 0.1 * err) {
            if (J->na == HP_NANC) { const struct HPChamp f = J->anc[0]; for (int a = 1; a < HP_NANC; a++) J->anc[a - 1] = J->anc[a]; J->anc[HP_NANC - 1] = f; J->na--; }
            struct HPChamp *Cc = &J->anc[J->na++];
            if (!Cc->y) {
                Cc->X = (uint64_t **)calloc((size_t)nb, sizeof(uint64_t *));
                for (int k = 0; k < nb; k++) Cc->X[k] = (uint64_t *)calloc(S->len[k] * Wm, sizeof(uint64_t));
                Cc->y = (uint64_t *)calloc((size_t)m * Wm + 1, sizeof(uint64_t));
            }
            for (int k = 0; k < nb; k++) for (size_t i = 0; i < S->len[k]; i++) HP_EXPORT(Cc->X[k] + i * Wm, EL(S->X[k], i), J);
            for (int p = 0; p < m; p++) HP_EXPORT(Cc->y + (size_t)p * Wm, EL(S->y, p), J);
            Cc->m[4] = err; Cc->have = 1;
            J->anc_thr = 1e-4 * err;
        }
        /* infeasibility: an improving ray. (P) infeasible: b'y > 0 with A*(y) + Z ~ 0;
         * (D) infeasible: <C,X> < 0 with A(X) ~ 0 */
        /* (5.3) Not after this level has been within 1e-5 of a solution: both problems then have
         * points feasible to that accuracy, and an iterate that runs away along a ray afterwards
         * is a breakdown of the iteration on an ill-posed problem (Mittelmann's rose15: error
         * 3e-7, then <C,X> = -2e10), which ends the level with its best iterate. */
        double ray_p = 1e300, ray_d = 1e300;
        if (err < min_err) min_err = err;
        const int was_near = min_err < 1e-5;
        if (err > 1e-6) {
            if (dob > 0) {
                for (int k = 0; k < nb; k++) for (size_t i = 0; i < S->len[k]; i++) HP_NEG(EL(S->T1[k], i), EL(S->Rd[k], i));
                HPN(addC)(S, S->T1, +1);                               /* C - Rd = A*(y) + Z */
                ray_d = HPN(normb)(S, S->T1) / dob / (1 + nC);
                if (ray_d < 1e-8) { if (was_near) rc = 1; else { rc = 4; J->have_iter = 0; } break; }
            }
            if (po < 0) {
                for (int p = 0; p < m; p++) HP_SUB(EL(S->v2, p), EL(S->b, p), EL(S->rp, p));   /* A(X) */
                ray_p = HPN(norm)((size_t)m, S->v2) / (-po) / (1 + nb_);
                if (ray_p < 1e-8) { if (was_near) rc = 1; else { rc = 5; J->have_iter = 0; } break; }
            }
        }
        if (err <= J->tol) { rc = 0; break; }
        /* a level below the last one goes as deep as its precision allows (its iterations
         * are several times cheaper than those of the next): it ends at 30 times its floor
         * eps^0.85, the goal of the centring cycles, or when it is exhausted (below) */
        if (err <= fmin(J->level_tol, 30.0 * pow(HP_EPS, 0.85))) { rc = 1; at_floor = 1; break; }
        if (it > 0) {
            /* exhausted: no step to speak of twice in a row, or the best error has not halved
             * in the last 8 iterations of this level */
            if (fmax(ap, ad) < 1e-3 && !cen_gaveup) nbad++; else if (!cen_gaveup) nbad = 0;
            cen_gaveup = 0;
            if (err < 0.5 * stall_ref) { stall_ref = err; stall_it = it; }
            /* (BRISK_HPSICK: the level ends once mu has outrun the error twice in a row; off) */
            if (J->sick > 0 && J->more && have_h && err > J->accept && compl < J->sick * err) { if (++nsick >= 2) { rc = 1; break; } } else nsick = 0;
            /* (once the error is within 10 times the tolerance, which the driver accepts, two
             * iterations without halving it are enough: the precision's floor for the residuals) */
            if (nbad >= 2 || it - stall_it >= (stall_ref <= 10.0 * J->tol ? 2 : 8) || !(err == err)) {
                /* a level that stops far from a solution along a ray that is improving to
                 * 1e-4 (the steps vanish before the ray reaches the 1e-8 of the test above) */
                rc = 1;
                if (!was_near && err > 1e-2 && ray_d < 1e-4 && ray_d <= ray_p) { rc = 4; J->have_iter = 0; }
                else if (!was_near && err > 1e-2 && ray_p < 1e-4) { rc = 5; J->have_iter = 0; }
                break;
            }
        }
        if (J->iters >= J->maxit || (J->timelimit > 0 && wtime() - J->tstart > J->timelimit) || brisk_stop_flag) { rc = 2; break; }
        J->iters++; redo = 0;

        /* Z^{-1} (after the tests: the last pass of a level does not need it) */
        for (int k = 0; k < nb; k++) {
            if (S->lp[k]) continue;
            HPN(inv_from_chol)(S->n[k], S->Lz[k], S->Zi[k], S->T1[k]);
        }
        /* Schur complement and its factor. An iteration that follows a centring step from a
         * point already close to the path (delta < 0.05) keeps the factor of that step:
         * the two Schur matrices differ by about delta, and the refinement below (the true
         * residual through the direction, corrected with the factor at hand) makes up for it
         * at the rate delta per round. Used where the factorization is the cost (m^3 / 3 above
         * 12 sum n^3: theta, control); elsewhere a fresh factor is as cheap. */
        const int frozen = cen_done && last_delta < 0.05 && frozen_ok && it > 0 && HP_EPS < 1e-20 && !(J->pivreg > 0)
                           && (double)m * m * m / 3.0 > 12.0 * sn3 && !getenv("BRISK_HPNOFROZEN");
        int nskip = 0, cfail = 0, ndyn = 0;
        if (!frozen) {
            HPN(schur)(S);
            if (getenv("BRISK_HPDBG")) { double dmin = 1e300, dmax = 0; for (int p = 0; p < m; p++) { const double d = HP_GETD(EL(S->M, (size_t)p * m + p)); if (d < dmin) dmin = d; if (d > dmax) dmax = d; } printf("       [M diag %.2e .. %.2e]\n", dmin, dmax); }
            if (!J->dep) cfail = HPN(cholr)(m, S->M, J->pivreg > 0 ? pow(HP_EPS, J->pivreg) : 0.0, &nskip, NULL, 0, &J->minpiv, J->mfst);
            else {
                /* the first factorization of a level: at the cold start the dependent rows are
                 * found; a higher precision checks the flags it inherits. With pivreg, rows whose
                 * pivot falls below eps^pivreg later are skipped as well (for that factorization). */
                int *mk = J->dep;
                if (J->pivreg > 0 && (it > 0 || J->dep_known)) { mk = S->mask; memcpy(mk, J->dep, sizeof(int) * (size_t)m); }
                if (it == 0 && !(J->pivreg > 0 && J->dep_known)) {
                    double mp = 1e300;
                    cfail = HPN(cholr)(m, S->M, pow(HP_EPS, J->dep_known ? 0.6 : 0.7), &nskip, J->dep, J->dep_known ? 2 : 1, &mp, J->mfst);
                    if (J->dep_known && mp < pow(HP_EPS, 0.9)) J->nwould++;
                    J->dep_known = 1;
                    if (nskip && J->verbose > 1) printf("     %s: %d linearly dependent constraint(s) left out of the Schur complement\n", HPN(name), nskip);
                } else if (J->pivreg > 0) {
                    cfail = HPN(cholr)(m, S->M, pow(HP_EPS, J->pivreg), &nskip, mk, 1, NULL, J->mfst);
                    for (int p = 0; p < m; p++) ndyn += mk[p] && !J->dep[p];
                    J->ndyn += ndyn;
                } else {
                    double mp = 1e300;
                    cfail = HPN(cholr)(m, S->M, 0.0, &nskip, J->dep, 0, &mp, J->mfst);
                    if (mp < J->minpiv) J->minpiv = mp;
                    if (mp < pow(HP_EPS, 0.9)) J->nwould++;      /* the skipping variant would differ here */
                }
            }
            if (getenv("BRISK_HPDBG")) printf("       [%d pivots skipped]\n", nskip);
        }
        if (J->deponly) { rc = 9; have_best = 0; break; }          /* only the dependent rows were wanted (hpsolve.c) */
        if (cfail) { rc = 1; J->chol_exit = 1; if (J->verbose > 1) printf("     %s: the Schur complement is not positive definite in this precision\n", HPN(name)); break; }
        /* K = X Rd Zi and A(K), unless Rd = 0 */
        const int have_rd = dinf > 100.0 * HP_EPS;
        HPN(zero)((size_t)m, S->v2);
        /* a dual residual at the rounding level is left as it is: dropped from dZ as well,
         * since X Rd Z^{-1} would amplify it by 1/mu in the right-hand side */
        if (!have_rd) for (int k = 0; k < nb; k++) HPN(zero)(S->len[k], S->Rd[k]);
        if (have_rd) {
            for (int k = 0; k < nb; k++) {
                const int n = S->n[k];
                if (S->lp[k]) { for (int i = 0; i < n; i++) { HP_MUL(t, EL(S->X[k], i), EL(S->Rd[k], i)); HP_DIV(EL(S->K[k], i), t, EL(S->Z[k], i)); } continue; }
                HPN(gemm_rs)(n, S->X[k], S->Rd[k], S->T1[k]);
                HPN(gemm)(n, S->T1[k], S->Zi[k], S->K[k]);
            }
            HPN(Aop)(S, S->K, S->v2);
        }
        TMP(a1); TMP(a2); TMP(u1); TMP(u2); TMP(u3);
        double apa = 1.0, ada = 1.0, muaff = 0, amin = 1.0, rho = 1.0, gam = 1.0, omega = 0.0;
        int fin_corr = 0; TMP(fsmu); HP_SETD(fsmu, 0.0);
        const double mud = HP_GETD(mu);
        int centring = cen_left > 0;
        const int after_cen = cen_done;
        /* goal of this level in terms of mu: the final tolerance, or as deep as this precision
         * goes when a higher one follows (its iterations cost several times more) */
        const double goal_err = fmax(0.3 * J->tol, J->level_tol > J->tol ? pow(HP_EPS, 0.85) : 0.0);
        const double drho = goal_err * den / N / mud;             /* mu_goal / mu */
        if (!centring) {
            /* predictor (sigma = 0) */
            HPN(make_rhs)(S, zero_, 0, S->v2);
            HPN(copy)((size_t)m, S->rhs, S->dy);
            HPN(msolve)(S, S->dy);
            HPN(direction)(S, S->dy, zero_, 0);
            if (frozen) HPN(refine)(S, zero_, 0, 14, 1e-2 * fmin(goal_err, compl), nb_);   /* mu_aff decides the jump: the predictor must be the true one */
            apa = 1e300; ada = 1e300;
            for (int k = 0; k < nb; k++) {
                apa = fmin(apa, HPN(maxstep)(S, k, S->Lx[k], S->dX[k], S->X[k]));
                ada = fmin(ada, HPN(maxstep)(S, k, S->Lz[k], S->dZ[k], S->Z[k]));
            }
            /* the step lengths come from a double eigensolver: within 1e-9 of the full step
             * they are taken as the full step (the factorization of the new point decides) */
            if (apa > 1.0 - 1e-9) apa = 1.0;
            if (ada > 1.0 - 1e-9) ada = 1.0;
            /* mu after the predictor step: <X + apa dX, Z + ada dZ> / N */
            HPN(inner)(S, S->dX, S->Z, u1); HPN(inner)(S, S->X, S->dZ, u2); HPN(inner)(S, S->dX, S->dZ, u3);
            {   /* in the working precision: at the quadratic end mu_aff / mu is far below 1e-16 */
                TMP(w1); TMP(w2); TMP(acc);
                HP_SETD(w1, apa); HP_SETD(w2, ada);
                HP_COPY(acc, xz); HP_ADDMUL(acc, w1, u1); HP_ADDMUL(acc, w2, u2);
                HP_MUL(w1, w1, w2); HP_ADDMUL(acc, w1, u3);
                muaff = HP_GETD(acc) / N;
            }
            amin = fmin(apa, ada);
            const double ex = amin > 0.57735 ? fmax(1.0, 3.0 * amin * amin) : 1.0;
            sigma = fmin(1.0, pow(fmax(muaff, 0.0) / mud, ex));
            if (!(sigma >= 0)) sigma = 0.5;
            for (int k = 0; k < nb; k++) { HPN(copy)(S->len[k], S->dX[k], S->dXa[k]); HPN(copy)(S->len[k], S->dZ[k], S->dZa[k]); }
            /* corrector */
            HP_SETD(t, sigma); HP_MUL(smu, mu, t);
            HPN(make_rhs)(S, smu, 1, S->v2);
            HPN(copy)((size_t)m, S->rhs, S->dy);
            HPN(msolve)(S, S->dy);
            HPN(direction)(S, S->dy, smu, 1);
            fin_corr = 1; HP_COPY(fsmu, smu);
            ap = 1e300; ad = 1e300;
            for (int k = 0; k < nb; k++) {
                ap = fmin(ap, HPN(maxstep)(S, k, S->Lx[k], S->dX[k], S->X[k]));
                ad = fmin(ad, HPN(maxstep)(S, k, S->Lz[k], S->dZ[k], S->Z[k]));
            }
            /* Safeguard: on degenerate problems the second-order term can spoil the direction (a
             * step of a few percent, after which the iteration loses the path: SDPLIB hinf1 at
             * mu = 4e-10 in quad-double). Then the plain centred Newton direction with the same
             * factor is tried, and the longer of the two is kept. */
            if (fmin(ap, ad) < 0.25) {
                for (int k = 0; k < nb; k++) { HPN(copy)(S->len[k], S->dX[k], S->dXa[k]); HPN(copy)(S->len[k], S->dZ[k], S->dZa[k]); }
                HPN(copy)((size_t)m, S->dy, S->v1);
                const double sig2 = fmax(sigma, 0.3);
                HP_SETD(t, sig2); HP_MUL(smu, mu, t);
                HPN(make_rhs)(S, smu, 0, S->v2);
                HPN(copy)((size_t)m, S->rhs, S->dy);
                HPN(msolve)(S, S->dy);
                HPN(direction)(S, S->dy, smu, 0);
                double ap2 = 1e300, ad2 = 1e300;
                for (int k = 0; k < nb; k++) {
                    ap2 = fmin(ap2, HPN(maxstep)(S, k, S->Lx[k], S->dX[k], S->X[k]));
                    ad2 = fmin(ad2, HPN(maxstep)(S, k, S->Lz[k], S->dZ[k], S->Z[k]));
                }
                if (fmin(ap2, ad2) > fmin(ap, ad)) { ap = ap2; ad = ad2; sigma = sig2; fin_corr = 0; HP_COPY(fsmu, smu); }
                else {
                    for (int k = 0; k < nb; k++) { HPN(copy)(S->len[k], S->dXa[k], S->dX[k]); HPN(copy)(S->len[k], S->dZa[k], S->dZ[k]); }
                    HPN(copy)((size_t)m, S->v1, S->dy);
                }
            }
            /* step fraction: SDPT3's 0.9 + 0.09 alpha; once the predictor alone takes (almost) the
             * full step and leaves mu_aff / mu = rho small, 1 - gamma follows rho, which gives the
             * superlinear end (mu falls by about rho per iteration, and rho falls with mu) */
            rho = fmax(muaff, 0.0) / mud;
            gam = 0.9 + 0.09 * fmin(fmin(ap, ad), 1.0);
            if (J->fast && amin > 0.9 && rho < 0.01) {
                /* 1 - gamma can be far below 1e-16: it is kept apart and the step 1 - omega is
                 * formed in the working precision */
                const double om = fmax(fmax(J->fast * rho, 4.0 * HP_EPS), fmin(drho, 0.1));
                gam = fmax(gam, 1.0 - om);
                if (om < 1e-8 && ap >= 1.0 - 1e-9 && ad >= 1.0 - 1e-9) omega = om;
            }
            /* centring pays where the predictor is good (non-degenerate end); a predictor
             * that is poor right after centring switches it off for this level */
            /* (tried again once mu is 1000 times smaller: far from the solution the curvature of
             * the path limits the predictor, not the centring) */
            if (after_cen && !(amin > 0.9 && rho < 0.05)) { cen_ok = 0; cen_retry = 1e-3 * mud; }
            else if (!cen_ok && mud < cen_retry) cen_ok = 1;
            /* In the higher precisions an iteration is a centring step unless the goal of the
             * level is within reach of this step (tested by the factors of the new point).
             * The same Schur factor serves: nothing has moved yet. */
            if (J->ncen > 0 && HP_EPS < 1e-20 && cen_ok && !after_cen && drho < 1e-3) {
                int jump = 0;
                /* (only in the last level: a level below it centres first in any case, so that
                 * its end point is close to the path and the next, dearer level can jump at once) */
                if (omega > 0 && omega <= 10.0 * drho && !(J->level_tol > J->tol)) {
                    TMP(o_); HP_SETD(o_, omega); HP_SETD(a1, 1.0); HP_SUB(a1, a1, o_);
                    jump = HPN(pdtest)(S, S->X, S->dX, a1) && HPN(pdtest)(S, S->Z, S->dZ, a1);
                }
                if (!jump) { centring = 1; cen_left = J->ncen; omega = 0; cen_rho = rho; }
            }
            if (getenv("BRISK_HPDBG")) printf("       [predictor: amin %.6f rho %.2e drho %.2e]\n", amin, rho, drho);
        }
        cen_done = centring;
        if (centring) {
            /* Centring step: Newton on X Z = mu I, no predictor, the full step when it stays
             * inside. The distance to the path is read from the step to the boundary (about
             * 1 / delta); a Newton step squares it. Centring goes on until the predictor that
             * follows can reach the goal: from a point at distance delta its mu_aff / mu is
             * about delta, so that one iteration then gains log10(1 / delta) digits. */
            sigma = 1.0;
            HPN(make_rhs)(S, mu, 0, S->v2);
            HPN(copy)((size_t)m, S->rhs, S->dy);
            HPN(msolve)(S, S->dy);
            HPN(direction)(S, S->dy, mu, 0);
            fin_corr = 0; HP_COPY(fsmu, mu);
            ap = 1e300; ad = 1e300;
            for (int k = 0; k < nb; k++) {
                ap = fmin(ap, HPN(maxstep)(S, k, S->Lx[k], S->dX[k], S->X[k]));
                ad = fmin(ad, HPN(maxstep)(S, k, S->Lz[k], S->dZ[k], S->Z[k]));
            }
            const double delta = 1.0 / fmin(ap, ad);
            last_delta = delta;
            /* a centring direction that hardly moves (a point far from the path whose gap is
             * mostly infeasibility: qap5 from the double start): centring is given up until mu
             * is 1000 times smaller, and the step is not counted as a failed one */
            if (fmin(ap, ad) < 0.05) { cen_ok = 0; cen_retry = 1e-3 * mud; cen_left = 1; cen_gaveup = 1; }
            gam = delta < 0.5 ? 1.0 : 0.98;
            if (delta < 0.5) { ap = fmin(ap, 1.0); ad = fmin(ad, 1.0); }
            cen_left--; stall_it++;
            /* enough when the distance after this step (delta^2) is a tenth of what the
             * predictor can use: the goal, or 1e-4 of the mu_aff / mu that the predictor of this
             * cycle reached (5.3; fewer centring steps far from the goal) */
            if (delta < 0.5 && delta * delta < 0.1 * fmax(drho, 1e-4 * cen_rho)) cen_left = 0;
            if (getenv("BRISK_HPDBG")) printf("       [centring: delta %.2e, goal %.2e, left %d]\n", delta, drho, cen_left);
        }
        /* Refinement of the Schur solve. A(dX) = rp is the Schur system itself, so its residual
         * is read off the direction (a sparse product). When that residual would leave a primal
         * infeasibility above a hundredth of the current gap (the factor of an ill-conditioned
         * Schur matrix: control problems near the end of a precision), the correction is solved
         * with the same factor and the direction recomputed; at most twice, while it helps. */
        if (HP_EPS < 1e-20 && !getenv("BRISK_HPNOREF")) {
            const double rtol = frozen ? 1e-2 * fmin(goal_err, compl) : 1e-2 * compl;
            const double en = HPN(refine)(S, fsmu, fin_corr, frozen ? 14 : 2, rtol, nb_);
            if (frozen && en > 100.0 * rtol) {
                /* the kept factor does not solve this system (the refinement stalls far above
                 * the target: control7 in quad-double took the step with a residual of 1e-15
                 * and lost 25 digits of feasibility): the iteration is done again from the
                 * same point with a fresh factor, and the factor is not kept again in this level */
                frozen_ok = 0;
                if (centring) cen_left++;
                J->iters--; it--; redo = 1;
                continue;
            }
        }
        ap = fmin(1.0, gam * ap); ad = fmin(1.0, gam * ad);
        /* update, with the new factors; shorter steps while a factor fails */
        double om_p = omega, om_d = omega;
        int ok = 0;
        for (int tries = 0; tries < 80 && !ok; tries++) {
            ok = 1;
            HP_SETD(a1, ap);
            if (om_p > 0) { TMP(o_); HP_SETD(o_, om_p); HP_SETD(a1, 1.0); HP_SUB(a1, a1, o_); }
            for (int k = 0; k < nb && ok; k++) {
                for (size_t i = 0; i < S->len[k]; i++) { HP_MUL(t, a1, EL(S->dX[k], i)); HP_ADD(EL(S->T1[k], i), EL(S->X[k], i), t); }
                if (S->lp[k]) { for (int i = 0; i < S->n[k]; i++) if (HP_SGN(EL(S->T1[k], i)) <= 0) ok = 0; }
                else { HPN(copy)(S->len[k], S->T1[k], S->Lx[k]); if (HPN_chol(S->n[k], S->Lx[k])) ok = 0; }
                if (ok) HPN(copy)(S->len[k], S->T1[k], S->K[k]);       /* K holds the accepted X of block k */
            }
            if (!ok) { if (om_p > 0) { om_p = pow(om_p, 0.8); if (om_p > 1e-3) { ap = 1.0 - om_p; om_p = 0; } } else ap *= 0.8; }
        }
        if (!ok) { rc = 1; break; }
        for (int k = 0; k < nb; k++) HPN(copy)(S->len[k], S->K[k], S->X[k]);
        ok = 0;
        for (int tries = 0; tries < 80 && !ok; tries++) {
            ok = 1;
            HP_SETD(a2, ad);
            if (om_d > 0) { TMP(o_); HP_SETD(o_, om_d); HP_SETD(a2, 1.0); HP_SUB(a2, a2, o_); }
            for (int k = 0; k < nb && ok; k++) {
                for (size_t i = 0; i < S->len[k]; i++) { HP_MUL(t, a2, EL(S->dZ[k], i)); HP_ADD(EL(S->T1[k], i), EL(S->Z[k], i), t); }
                if (S->lp[k]) { for (int i = 0; i < S->n[k]; i++) if (HP_SGN(EL(S->T1[k], i)) <= 0) ok = 0; }
                else { HPN(copy)(S->len[k], S->T1[k], S->Lz[k]); if (HPN_chol(S->n[k], S->Lz[k])) ok = 0; }
                if (ok) HPN(copy)(S->len[k], S->T1[k], S->K[k]);
            }
            if (!ok) { if (om_d > 0) { om_d = pow(om_d, 0.8); if (om_d > 1e-3) { ad = 1.0 - om_d; om_d = 0; } } else ad *= 0.8; }
        }
        if (!ok) { rc = 1; break; }
        for (int k = 0; k < nb; k++) HPN(copy)(S->len[k], S->K[k], S->Z[k]);
        if (getenv("BRISK_HPDBG") && omega > 0) printf("       [jump: omega %.2e -> %.2e, %.2e]\n", omega, om_p, om_d);
        HPN(axpy)((size_t)m, a2, S->dy, S->y);
    }
    J->level_best = have_best ? best : 1e300;
    /* export the best iterate, or the handover iterate when the best one is of no use to the
     * next precision */
    if (have_best && rc == 1 && !at_floor && J->more && have_h && J->best_err > J->accept && besth > J->best_err && besth < 1e4 * J->best_err && J->champ) {
        struct HPChamp *Cc = J->champ;
        if (!Cc->have || J->best_err < Cc->m[4]) {
            if (!Cc->y) {
                for (int k = 0; k < nb; k++) { Cc->X[k] = (uint64_t *)calloc(S->len[k] * Wm, sizeof(uint64_t)); Cc->Z[k] = (uint64_t *)calloc(S->len[k] * Wm, sizeof(uint64_t)); }
                Cc->y = (uint64_t *)calloc((size_t)m * Wm + 1, sizeof(uint64_t)); Cc->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); Cc->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
            }
            for (int k = 0; k < nb; k++)
                for (size_t i = 0; i < S->len[k]; i++) { HP_EXPORT(Cc->X[k] + i * Wm, EL(S->Xb[k], i), J); HP_EXPORT(Cc->Z[k] + i * Wm, EL(S->Zb[k], i), J); }
            for (int p = 0; p < m; p++) HP_EXPORT(Cc->y + (size_t)p * Wm, EL(S->yb, p), J);
            memcpy(Cc->pobj, J->pobj, Wm * sizeof(uint64_t)); memcpy(Cc->dobj, J->dobj, Wm * sizeof(uint64_t));
            Cc->m[0] = J->best_pinf; Cc->m[1] = J->best_dinf; Cc->m[2] = J->best_gap; Cc->m[3] = J->best_compl; Cc->m[4] = J->best_err;
            Cc->have = 1;
        }
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < S->len[k]; i++) { HP_EXPORT(J->X[k] + i * Wm, EL(S->Xh[k], i), J); HP_EXPORT(J->Z[k] + i * Wm, EL(S->Zh[k], i), J); }
        for (int p = 0; p < m; p++) HP_EXPORT(J->y + (size_t)p * Wm, EL(S->yh, p), J);
        HP_EXPORT(J->pobj, hpo, J); HP_EXPORT(J->dobj, hdo, J);
        if (J->verbose > 1) printf("     %s: exhausted at the error %.1e; the iterate handed over is the one at %.1e, where the error still was the complementarity\n", HPN(name), J->best_err, hm[4]);
        J->best_pinf = hm[0]; J->best_dinf = hm[1]; J->best_gap = hm[2]; J->best_compl = hm[3]; J->best_err = hm[4];
        J->have_iter = 1; J->sick_exit = 1;
    } else if (have_best) {
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < S->len[k]; i++) { HP_EXPORT(J->X[k] + i * Wm, EL(S->Xb[k], i), J); HP_EXPORT(J->Z[k] + i * Wm, EL(S->Zb[k], i), J); }
        for (int p = 0; p < m; p++) HP_EXPORT(J->y + (size_t)p * Wm, EL(S->yb, p), J);
        J->have_iter = 1;
        /* the handover iterate is kept as a spare: a best iterate at the rounding level of
         * this precision can fail the Cholesky test of the next one (hpsolve.c) */
        if (rc == 1 && J->more && have_h && besth > J->best_err && J->spare) {
            struct HPChamp *Cc = J->spare;
            if (!Cc->y) {
                for (int k = 0; k < nb; k++) { Cc->X[k] = (uint64_t *)calloc(S->len[k] * Wm, sizeof(uint64_t)); Cc->Z[k] = (uint64_t *)calloc(S->len[k] * Wm, sizeof(uint64_t)); }
                Cc->y = (uint64_t *)calloc((size_t)m * Wm + 1, sizeof(uint64_t)); Cc->pobj = (uint64_t *)calloc(Wm, sizeof(uint64_t)); Cc->dobj = (uint64_t *)calloc(Wm, sizeof(uint64_t));
            }
            for (int k = 0; k < nb; k++)
                for (size_t i = 0; i < S->len[k]; i++) { HP_EXPORT(Cc->X[k] + i * Wm, EL(S->Xh[k], i), J); HP_EXPORT(Cc->Z[k] + i * Wm, EL(S->Zh[k], i), J); }
            for (int p = 0; p < m; p++) HP_EXPORT(Cc->y + (size_t)p * Wm, EL(S->yh, p), J);
            HP_EXPORT(Cc->pobj, hpo, J); HP_EXPORT(Cc->dobj, hdo, J);
            for (int q = 0; q < 5; q++) Cc->m[q] = hm[q];
            Cc->have = 1;
        }
    }
done:
    for (int a = 0; a < narr; a++) { for (int k = 0; k < nb; k++) free((*arr[a])[k]); free(*arr[a]); }
    free(S->Av); free(S->Cv); free(S->b); free(S->n); free(S->lp); free(S->len);
    free(S->y); free(S->dy); free(S->yb); free(S->yh); free(S->rp); free(S->rhs); free(S->v1); free(S->v2); free(S->M); free(S->vp);
    free(S->wd); free(S->ev); free(S->lw); free(S->mask);
    return rc;
}
