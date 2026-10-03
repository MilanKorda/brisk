/* SPDX-License-Identifier: Apache-2.0
 * hplr.inc (5.0): the low-rank (Burer-Monteiro) augmented Lagrangian method in high precision,
 * compiled for each arithmetic after hpipm.inc and hpfom.inc (hpsolve.c).
 *
 *   X_k = R_k R_k' (R_k: n_k x r_k; an LP block is r = 1 with only its diagonal used),
 *   L(R, y) = <C, RR'> - y'(A(RR') - b) + sigma/2 |A(RR') - b|^2,
 * minimised over R by Newton-CG with an exact line search (L is a quartic along a direction),
 * then y <- y - sigma (A(RR') - b). Nothing of size n x n is formed in the iteration: the
 * operators work on the union sparsity pattern of C and the A_p (built once, hpsolve.c):
 *   row dots  g_ij = R_i . R_j  on the pattern,  A(RR')_p = sum_e v_e g_e,
 *   S = C - A*(w) on the pattern,  (S D)_i = sum_j S_ij D_j.
 * The dual matrix Z = C - A*(y) is checked by a Cholesky factorization of Z + delta I in the
 * working precision (dense, once per outer iteration). */

typedef struct {
    HPJob *J;
    int m, nb, *n, *lp, *r, *cap;
    HPT *Av, *Cv, *b;
    HPT **R, **G, **D, **P, **Q, **Rs, **Rb;       /* n x cap, row-major with r columns in use */
    HPT **sv, **tv, **gv, **svb;                   /* values on the pattern */
    HPT *y, *u, *w, *q1, *q2, *yb;
} HPN(ls);

/* out_s = A_i . B_j (A == B), or A_i . B_j + A_j . B_i */
static void HPN(lr_dots)(const HPN(ls) *L, int k, const HPT *A, const HPT *B, HPT *out) {
    const HPJob *J = L->J;
    const int r = L->r[k], np = J->lr_np[k];
    const int *pi = J->lr_pi[k], *pj = J->lr_pj[k];
#pragma omp parallel for schedule(static) if ((double)np * r > 20000.0)
    for (int s = 0; s < np; s++) {
        HPN(dot)((size_t)r, EL(A, (size_t)pi[s] * r), EL(B, (size_t)pj[s] * r), EL(out, s));
        if (A != B) { TMP(t); HPN(dot)((size_t)r, EL(A, (size_t)pj[s] * r), EL(B, (size_t)pi[s] * r), t); HP_ADD(EL(out, s), EL(out, s), t); }
    }
}
/* out_p = <A_p, G> and c = <C, G> for the pattern values gv of a symmetric G */
static void HPN(lr_A)(const HPN(ls) *L, HPT *const *gv, HPT *out, HPT *c) {
    const HPJob *J = L->J;
    HPN(zero)((size_t)L->m, out);
    if (c) HP_SETD(c, 0.0);
    for (int k = 0; k < L->nb; k++) {
        for (int t = 0; t < J->ncon[k]; t++) {
            HPT *o = EL(out, J->con[k][t]);
            for (size_t e = J->ptr[k][t]; e < J->ptr[k][t + 1]; e++) {
                TMP(x); HP_MUL(x, EL(L->Av, e), EL(gv[k], J->lr_es[e]));
                if (J->ei[e] != J->ej[e]) HP_ADD(x, x, x);
                HP_ADD(o, o, x);
            }
        }
        if (c) for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
            TMP(x); HP_MUL(x, EL(L->Cv, e), EL(gv[k], J->lr_cs[e]));
            if (J->ci[e] != J->cj[e]) HP_ADD(x, x, x);
            HP_ADD(c, c, x);
        }
    }
}
/* vals = [C] + sgn sum_p w_p A_p on the pattern */
static void HPN(lr_S)(const HPN(ls) *L, const HPT *w, int withC, int sgn, HPT **vals) {
    const HPJob *J = L->J;
    for (int k = 0; k < L->nb; k++) {
        HPN(zero)((size_t)J->lr_np[k], vals[k]);
        if (withC) for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) HP_ADD(EL(vals[k], J->lr_cs[e]), EL(vals[k], J->lr_cs[e]), EL(L->Cv, e));
        for (int t = 0; t < J->ncon[k]; t++) {
            const HPT *wp = EL(w, J->con[k][t]);
            if (HP_SGN(wp) == 0) continue;
            for (size_t e = J->ptr[k][t]; e < J->ptr[k][t + 1]; e++) {
                if (sgn > 0) HP_ADDMUL(EL(vals[k], J->lr_es[e]), wp, EL(L->Av, e));
                else HP_SUBMUL(EL(vals[k], J->lr_es[e]), wp, EL(L->Av, e));
            }
        }
    }
}
/* Out = S1 D1 [+ S2 D2] for pattern values S1, S2 (block k) */
static void HPN(lr_SD)(const HPN(ls) *L, int k, const HPT *s1, const HPT *D1, const HPT *s2, const HPT *D2, HPT *Out) {
    const HPJob *J = L->J;
    const int n = L->n[k], r = L->r[k];
    const int *rp = J->lr_rp[k], *rc = J->lr_rc[k], *rs = J->lr_rs[k];
#pragma omp parallel for schedule(static) if ((double)rp[n] * r > 20000.0)
    for (int i = 0; i < n; i++) {
        HPT *o = EL(Out, (size_t)i * r);
        HPN(zero)((size_t)r, o);
        for (int a = rp[i]; a < rp[i + 1]; a++) {
            HPN(axpy)((size_t)r, EL(s1, rs[a]), EL(D1, (size_t)rc[a] * r), o);
            if (s2) HPN(axpy)((size_t)r, EL(s2, rs[a]), EL(D2, (size_t)rc[a] * r), o);
        }
    }
}
static void HPN(lr_inner)(const HPN(ls) *L, HPT *const *A, HPT *const *B, HPT *res) {
    TMP(t); HP_SETD(res, 0.0);
    for (int k = 0; k < L->nb; k++) { HPN(dot)((size_t)L->n[k] * L->r[k], A[k], B[k], t); HP_ADD(res, res, t); }
}
/* half Hessian times P: Q = S P + sigma A*(A(R P' + P R')) R */
static void HPN(lr_hess)(HPN(ls) *L, const HPT *sig, HPT **P, HPT **Q) {
    for (int k = 0; k < L->nb; k++) HPN(lr_dots)(L, k, L->R[k], P[k], L->gv[k]);
    HPN(lr_A)(L, L->gv, L->q1, NULL);
    HPN(scal)((size_t)L->m, sig, L->q1);
    HPN(lr_S)(L, L->q1, 0, +1, L->tv);
    for (int k = 0; k < L->nb; k++) HPN(lr_SD)(L, k, L->sv[k], P[k], L->tv[k], L->R[k], Q[k]);
}
/* the dense S of block k from its pattern values (symmetric, full storage) */
static void HPN(lr_dense)(const HPN(ls) *L, int k, const HPT *vals, HPT *Z) {
    const HPJob *J = L->J;
    const int n = L->n[k];
    HPN(zero)((size_t)n * n, Z);
    for (int s = 0; s < J->lr_np[k]; s++) {
        const int i = J->lr_pi[k][s], j = J->lr_pj[k][s];
        HP_COPY(EL(Z, (size_t)i * n + j), EL(vals, s));
        HP_COPY(EL(Z, (size_t)j * n + i), EL(vals, s));
    }
}
/* minimiser t > 0 of a1 t + a2 t^2 + a3 t^3 + a4 t^4 (a1 < 0, a4 >= 0), coefficients scaled by |a1| */
static double HPN(lr_quartic)(double a2, double a3, double a4) {
    /* derivative -1 + 2 a2 t + 3 a3 t^2 + 4 a4 t^3: its first positive root */
    double hi = 1.0;
    int k = 0;
    while (-1.0 + hi * (2 * a2 + hi * (3 * a3 + hi * 4 * a4)) < 0 && k < 60) { hi *= 2; k++; }
    if (k == 60) return 1.0;
    double lo = 0.0;
    /* the first root: step up from 0 on a grid, then bisect */
    for (int g = 1; g <= 32; g++) {
        const double t = hi * g / 32.0;
        if (-1.0 + t * (2 * a2 + t * (3 * a3 + t * 4 * a4)) >= 0) { hi = t; break; }
        lo = t;
    }
    for (int it = 0; it < 100; it++) {
        const double t = 0.5 * (lo + hi);
        if (-1.0 + t * (2 * a2 + t * (3 * a3 + t * 4 * a4)) < 0) lo = t; else hi = t;
    }
    return 0.5 * (lo + hi);
}

static int HPN(lr)(HPJob *J) {
    HPN(ls) L_, *L = &L_;
    memset(L, 0, sizeof(*L));
    L->J = J; L->m = J->m; L->nb = J->nblk;
    const int m = L->m, nb = L->nb;
    const size_t Wm = (size_t)J->Lm + 2;
    L->Av = HPN(alloc)(J->nnzA); L->Cv = HPN(alloc)(J->nnzC); L->b = HPN(alloc)((size_t)m);
    for (size_t e = 0; e < J->nnzA; e++) HP_IMPORT(EL(L->Av, e), J->Av + e * Wm, J);
    for (size_t e = 0; e < J->nnzC; e++) HP_IMPORT(EL(L->Cv, e), J->Cv + e * Wm, J);
    for (int p = 0; p < m; p++) HP_IMPORT(EL(L->b, p), J->b + (size_t)p * Wm, J);
    L->n = (int *)malloc(sizeof(int) * (size_t)nb); L->lp = (int *)malloc(sizeof(int) * (size_t)nb);
    L->r = (int *)malloc(sizeof(int) * (size_t)nb); L->cap = (int *)malloc(sizeof(int) * (size_t)nb);
    int nmax = 1;
    for (int k = 0; k < nb; k++) {
        L->lp[k] = J->bs[k] < 0; L->n[k] = abs(J->bs[k]);
        if (!L->lp[k] && L->n[k] > nmax) nmax = L->n[k];
    }
    double *Xd = (double *)malloc(sizeof(double) * (size_t)nmax * nmax), *wd = (double *)malloc(sizeof(double) * (size_t)nmax);
    const int lwork = 3 * nmax + 64;
    double *work = (double *)malloc(sizeof(double) * (size_t)lwork);
    HPT *Zd = NULL;                                 /* dense n x n, for the check of Z and the export */
    /* ranks: carried from the level below, or from the eigenvalues of the starting X */
    for (int k = 0; k < nb; k++) {
        const int n = L->n[k];
        if (L->lp[k]) { L->r[k] = 1; L->cap[k] = 1; continue; }
        if (J->lr_have) { L->r[k] = J->lr_rk[k]; L->cap[k] = J->lr_cap[k]; continue; }
        int r = 1;
        if (J->have_iter) {
            /* rank of X from a pair that is nearly complementary: with the eigenvalues of X
             * decreasing and those of Z increasing (each scaled by its largest), the leading
             * ones of X that exceed their partner in Z */
            int info = 0;
            double *zd = (double *)malloc(sizeof(double) * (size_t)n), zmax = 0;
            for (size_t i = 0; i < (size_t)n * n; i++) Xd[i] = mpx_getd(J->Z[k] + i * Wm);
            BL(dsyev_)("N", "U", &n, Xd, &n, zd, work, &lwork, &info);
            zmax = fmax(fabs(zd[n - 1]), fabs(zd[0]));
            for (size_t i = 0; i < (size_t)n * n; i++) Xd[i] = mpx_getd(J->X[k] + i * Wm);
            BL(dsyev_)("V", "U", &n, Xd, &n, wd, work, &lwork, &info);
            const double thr = getenv("BRISK_HPLRTHR") ? atof(getenv("BRISK_HPLRTHR")) : 1e-5;
            r = 0;
            for (int i = 0; i < n; i++) {
                const double xi = wd[n - 1 - i] / wd[n - 1], zi = zmax > 0 ? fmax(zd[i], 0.0) / zmax : 0.0;
                if (zmax > 0 ? xi > 1e-3 * zi && xi > 1e-12 : xi > thr) r++; else break;
            }
            free(zd);
            if (r < 1) r = 1;
        } else {
            r = (int)ceil(sqrt(2.0 * J->ncon[k])) + 1;
            if (r > n) r = n;
        }
        L->r[k] = r; L->cap[k] = r + 8 < n ? r + 8 : n;
        J->lr_cap[k] = L->cap[k];
        /* R = V sqrt(lambda) of the leading r pairs, into the exchange (columns of dsyev's Xd) */
        J->lr_R[k] = (uint64_t *)calloc((size_t)n * L->cap[k] * Wm, sizeof(uint64_t));
        for (int i = 0; i < n; i++)
            for (int c = 0; c < r; c++) {
                const double v = J->have_iter ? Xd[(size_t)(n - 1 - c) * n + i] * sqrt(fmax(wd[n - 1 - c], 0.0)) : (i == c ? 1.0 : 0.01 * sin(1.0 + i * (c + 1.0)));
                mpx_setd(J->lr_R[k] + ((size_t)i * L->cap[k] + c) * Wm, v);
            }
    }
    for (int k = 0; k < nb; k++) if (L->lp[k] && !J->lr_have) {
        const int n = L->n[k];
        J->lr_cap[k] = 1;
        J->lr_R[k] = (uint64_t *)calloc((size_t)n * Wm, sizeof(uint64_t));
        for (int i = 0; i < n; i++) mpx_setd(J->lr_R[k] + (size_t)i * Wm, J->have_iter ? sqrt(fmax(mpx_getd(J->X[k] + (size_t)i * Wm), 0.0)) : 1.0);
    }
    HPT ***arr[] = { &L->R, &L->G, &L->D, &L->P, &L->Q, &L->Rs, &L->Rb };
    const int narr = (int)(sizeof(arr) / sizeof(arr[0]));
    for (int a = 0; a < narr; a++) {
        *arr[a] = (HPT **)malloc(sizeof(HPT *) * (size_t)nb);
        for (int k = 0; k < nb; k++) (*arr[a])[k] = HPN(alloc)((size_t)L->n[k] * L->cap[k]);
    }
    HPT ***pat[] = { &L->sv, &L->tv, &L->gv, &L->svb };
    for (int a = 0; a < 4; a++) {
        *pat[a] = (HPT **)malloc(sizeof(HPT *) * (size_t)nb);
        for (int k = 0; k < nb; k++) (*pat[a])[k] = HPN(alloc)((size_t)J->lr_np[k]);
    }
    HPT **vec[] = { &L->y, &L->u, &L->w, &L->q1, &L->q2, &L->yb };
    for (int a = 0; a < 6; a++) *vec[a] = HPN(alloc)((size_t)m);
    Zd = HPN(alloc)((size_t)nmax * nmax);
    /* import R (stored in the exchange with cap columns; here r columns contiguous) and y */
    for (int k = 0; k < nb; k++)
        for (int i = 0; i < L->n[k]; i++)
            for (int c = 0; c < L->r[k]; c++) HP_IMPORT(EL(L->R[k], (size_t)i * L->r[k] + c), J->lr_R[k] + ((size_t)i * L->cap[k] + c) * Wm, J);
    if (J->have_iter) for (int p = 0; p < m; p++) HP_IMPORT(EL(L->y, p), J->y + (size_t)p * Wm, J);
    J->lr_have = 1;
    for (int k = 0; k < nb; k++) J->lr_rk[k] = L->r[k];

    const double nb_ = HPN(norm)((size_t)m, L->b), nC = HPN(norm)(J->nnzC, L->Cv);
    double sigma = J->sigma > 0 ? J->sigma : 10.0;
    TMP(sig); TMP(c0); TMP(t); TMP(x); TMP(pobj); TMP(dobj); TMP(rr); TMP(rr2); TMP(al); TMP(be); TMP(pq);
    int rc = 2, have_best = 0, nbad = 0;
    int *rbest = (int *)malloc(sizeof(int) * (size_t)nb);
    for (int k = 0; k < nb; k++) rbest[k] = L->r[k];
    double best = 1e300, prev_err = 1.0, prev_pinf = 1e300, delta_best = 0;
    /* u = A(RR') - b, c0 = <C, RR'> */
#define LR_EVAL() do { \
        for (int k_ = 0; k_ < nb; k_++) HPN(lr_dots)(L, k_, L->R[k_], L->R[k_], L->gv[k_]); \
        HPN(lr_A)(L, L->gv, L->u, c0); \
        for (int p_ = 0; p_ < m; p_++) HP_SUB(EL(L->u, p_), EL(L->u, p_), EL(L->b, p_)); } while (0)
    LR_EVAL();
    if (J->have_iter) prev_err = fmin(1.0, fmax(HPN(norm)((size_t)m, L->u) / (1 + nb_), 1e-12));   /* the inner tolerance follows the residual of the start */
    for (int outer = 0; ; outer++) {
        HP_SETD(sig, sigma);
        const double intol = fmax(0.3 * J->tol, 1e-2 * fmin(prev_err, 1.0));
        int newt = 0, cgtot = 0;
        double gn = 0, grel = 0, grel0 = 0;
        for (;; newt++) {
            /* w = y - sigma u, S = C - A*(w), G = S R (half the gradient) */
            for (int p = 0; p < m; p++) { HP_MUL(t, sig, EL(L->u, p)); HP_SUB(EL(L->w, p), EL(L->y, p), t); }
            HPN(lr_S)(L, L->w, 1, -1, L->sv);
            double rn2 = 0;
            for (int k = 0; k < nb; k++) {
                HPN(lr_SD)(L, k, L->sv[k], L->R[k], NULL, NULL, L->G[k]);
                HPN(dot)((size_t)L->n[k] * L->r[k], L->R[k], L->R[k], t); rn2 += HP_GETD(t);
            }
            HPN(lr_inner)(L, L->G, L->G, rr);
            gn = sqrt(fmax(HP_GETD(rr), 0.0));
            grel = gn / ((1 + nC) * (1 + sqrt(rn2)));
            if (newt == 0) grel0 = grel;
            if (grel <= intol || newt >= 60) break;
            if ((J->timelimit > 0 && wtime() - J->tstart > J->timelimit) || brisk_stop_flag) break;
            /* CG on H D = -G; a direction of non-positive curvature ends it */
            const double eta = fmin(0.1, sqrt(grel)) * gn;
            const int cgmax = J->cgmax > 0 ? J->cgmax : 2000;
            int negc = 0;
            for (int k = 0; k < nb; k++) {
                const size_t len = (size_t)L->n[k] * L->r[k];
                HPN(zero)(len, L->D[k]);
                for (size_t i = 0; i < len; i++) { HP_NEG(EL(L->Rs[k], i), EL(L->G[k], i)); HP_COPY(EL(L->P[k], i), EL(L->Rs[k], i)); }
            }
            for (int cg = 0; cg < cgmax; cg++) {
                HPN(lr_hess)(L, sig, L->P, L->Q);
                HPN(lr_inner)(L, L->P, L->Q, pq);
                cgtot++;
                if (HP_SGN(pq) <= 0) { negc = 1; if (cg == 0) for (int k = 0; k < nb; k++) HPN(copy)((size_t)L->n[k] * L->r[k], L->P[k], L->D[k]); break; }
                HP_DIV(al, rr, pq);
                HP_NEG(t, al);
                for (int k = 0; k < nb; k++) {
                    const size_t len = (size_t)L->n[k] * L->r[k];
                    HPN(axpy)(len, al, L->P[k], L->D[k]); HPN(axpy)(len, t, L->Q[k], L->Rs[k]);
                }
                HPN(lr_inner)(L, L->Rs, L->Rs, rr2);
                const double rn = HP_GETD(rr2);
                if (!(rn > 0) || sqrt(rn) <= eta) break;
                HP_DIV(be, rr2, rr); HP_COPY(rr, rr2);
                for (int k = 0; k < nb; k++) {
                    const size_t len = (size_t)L->n[k] * L->r[k];
                    for (size_t i = 0; i < len; i++) { HPT *pp = EL(L->P[k], i); HP_MUL(pp, pp, be); HP_ADD(pp, pp, EL(L->Rs[k], i)); }
                }
            }
            (void)negc;
            /* exact line search: L(R + t D) - L(R) = a1 t + a2 t^2 + a3 t^3 + a4 t^4 */
            TMP(c1); TMP(c2); TMP(a1); TMP(a2); TMP(a3); TMP(a4); TMP(half);
            HP_SETD(half, 0.5);
            for (int k = 0; k < nb; k++) HPN(lr_dots)(L, k, L->R[k], L->D[k], L->gv[k]);
            HPN(lr_A)(L, L->gv, L->q1, c1);
            for (int k = 0; k < nb; k++) HPN(lr_dots)(L, k, L->D[k], L->D[k], L->gv[k]);
            HPN(lr_A)(L, L->gv, L->q2, c2);
            HPN(dot)((size_t)m, L->y, L->q1, t); HP_SUB(a1, c1, t);
            HPN(dot)((size_t)m, L->u, L->q1, t); HP_MUL(t, t, sig); HP_ADD(a1, a1, t);
            HPN(dot)((size_t)m, L->y, L->q2, t); HP_SUB(a2, c2, t);
            HPN(dot)((size_t)m, L->u, L->q2, t); HPN(dot)((size_t)m, L->q1, L->q1, x); HP_MUL(x, x, half); HP_ADD(t, t, x); HP_MUL(t, t, sig); HP_ADD(a2, a2, t);
            HPN(dot)((size_t)m, L->q1, L->q2, t); HP_MUL(a3, t, sig);
            HPN(dot)((size_t)m, L->q2, L->q2, t); HP_MUL(t, t, half); HP_MUL(a4, t, sig);
            if (HP_SGN(a1) >= 0) break;                      /* not a descent direction: the inner iteration ends */
            TMP(na1); HP_NEG(na1, a1);
            HP_DIV(t, a2, na1); const double b2 = HP_GETD(t);
            HP_DIV(t, a3, na1); const double b3 = HP_GETD(t);
            HP_DIV(t, a4, na1); const double b4 = HP_GETD(t);
            const double step = HPN(lr_quartic)(b2, b3, b4);
            if (!(step > 0)) break;
            HP_SETD(t, step);
            for (int k = 0; k < nb; k++) HPN(axpy)((size_t)L->n[k] * L->r[k], t, L->D[k], L->R[k]);
            LR_EVAL();
        }
        J->iters += newt;
        /* multiplier update and the measures of (RR', y, S(y)) */
        const double pinf = HPN(norm)((size_t)m, L->u) / (1 + nb_);
        HPN(copy)((size_t)m, L->y, L->q2);                   /* kept: a rank change takes the update back */
        HPN(copy)((size_t)m, L->w, L->y);                    /* y - sigma u, with S(y) = sv and G = S R */
        HPN(dot)((size_t)m, L->b, L->y, dobj); HP_COPY(pobj, c0);
        HPN(lr_inner)(L, L->R, L->G, t);
        const double po = HP_GETD(pobj), dob = HP_GETD(dobj), den = 1 + fabs(po) + fabs(dob);
        HP_SUB(x, pobj, dobj);
        const double gap = HP_GETD(x) / den, compl = fabs(HP_GETD(t)) / den;
        /* smallest eigenvalue of S: in double first */
        double lmin = 1e300, nS = 0; int kmin = -1;
        for (int k = 0; k < nb; k++) {
            const int n = L->n[k];
            if (L->lp[k]) { for (int i = 0; i < n; i++) { const double v = HP_GETD(EL(L->sv[k], J->lr_diag[k][i])); if (v < lmin) { lmin = v; kmin = k; } if (fabs(v) > nS) nS = fabs(v); } continue; }
            memset(Xd, 0, sizeof(double) * (size_t)n * n);
            for (int s = 0; s < J->lr_np[k]; s++) { const double v = HP_GETD(EL(L->sv[k], s)); Xd[(size_t)J->lr_pi[k][s] * n + J->lr_pj[k][s]] = v; Xd[(size_t)J->lr_pj[k][s] * n + J->lr_pi[k][s]] = v; }
            int info = 0;
            BL(dsyev_)("N", "U", &n, Xd, &n, wd, work, &lwork, &info);
            if (wd[0] < lmin) { lmin = wd[0]; kmin = k; }
            if (fabs(wd[n - 1]) > nS) nS = fabs(wd[n - 1]);
            if (fabs(wd[0]) > nS) nS = fabs(wd[0]);
        }
        double dinf = fmax(0.0, -lmin) / (1 + nC), delta = fmax(0.0, -lmin);
        const double others = fmax(fmax(pinf, fabs(gap)), compl);
        if (-lmin < 1e-11 * (1 + nS)) {
            /* at the rounding level of the double eigenvalues: Cholesky of S + delta I in the
             * working precision, delta from the other errors upwards */
            double dl = fmax(others, 0.1 * J->tol) * (1 + nC);
            int ok = 0;
            for (int tries = 0; tries < 6 && !ok; tries++, dl *= 100.0) {
                ok = 1;
                HP_SETD(x, dl);
                for (int k = 0; k < nb && ok; k++) {
                    const int n = L->n[k];
                    if (L->lp[k]) { for (int i = 0; i < n; i++) { HP_ADD(t, EL(L->sv[k], J->lr_diag[k][i]), x); if (HP_SGN(t) <= 0) ok = 0; } continue; }
                    HPN(lr_dense)(L, k, L->sv[k], Zd);
                    for (int i = 0; i < n; i++) HP_ADD(EL(Zd, (size_t)i * n + i), EL(Zd, (size_t)i * n + i), x);
                    if (HPN_chol(n, Zd)) ok = 0;
                }
                if (ok) { delta = dl; dinf = dl / (1 + nC); }
            }
            if (!ok) { delta = fmax(1e-11 * (1 + nS), -lmin); dinf = delta / (1 + nC); }
        }
        const double err = fmax(fmax(pinf, dinf), fmax(fabs(gap), compl));
        J->pinf = pinf; J->dinf = dinf; J->gap = gap; J->compl = compl; J->err = err;
        if (J->verbose > 1) {
            int rmax = 0; for (int k = 0; k < nb; k++) if (!L->lp[k] && L->r[k] > rmax) rmax = L->r[k];
            printf("%4d %-3s %+.14e %+.14e %8.1e %8.1e %9.1e  rank %3d  sigma %7.1e  newton %2d cg %5d %7.1fs\n", outer, HPN(name), po, dob, pinf, dinf, gap, rmax, sigma, newt, cgtot, wtime() - J->tstart);
        }
        if (err < best) {
            best = err; have_best = 1; delta_best = delta;
            for (int k = 0; k < nb; k++) { HPN(copy)((size_t)L->n[k] * L->r[k], L->R[k], L->Rb[k]); HPN(copy)((size_t)J->lr_np[k], L->sv[k], L->svb[k]); rbest[k] = L->r[k]; }
            HPN(copy)((size_t)m, L->y, L->yb);
            HP_EXPORT(J->pobj, pobj, J); HP_EXPORT(J->dobj, dobj, J);
            J->best_pinf = pinf; J->best_dinf = dinf; J->best_gap = gap; J->best_compl = compl; J->best_err = err;
            nbad = 0;
        } else nbad++;
        if (err <= J->tol) { rc = 0; break; }
        if (err <= J->level_tol) { rc = 1; break; }
        if (nbad >= 4 || !(err == err)) { rc = 1; break; }
        if (J->iters >= J->maxit || (J->timelimit > 0 && wtime() - J->tstart > J->timelimit) || brisk_stop_flag) { rc = 2; break; }
        /* rank: a clearly negative eigenvalue of S at a stationary point with a small residual
         * means that a column is missing; the eigenvector enters as a small new column */
        if (kmin >= 0 && !L->lp[kmin] && dinf > 1e-9 && dinf > 10.0 * fmax(pinf, compl) && L->r[kmin] < L->cap[kmin]) {
            const int k = kmin, n = L->n[k], r = L->r[k];
            memset(Xd, 0, sizeof(double) * (size_t)n * n);
            for (int s = 0; s < J->lr_np[k]; s++) { const double v = HP_GETD(EL(L->sv[k], s)); Xd[(size_t)J->lr_pi[k][s] * n + J->lr_pj[k][s]] = v; Xd[(size_t)J->lr_pj[k][s] * n + J->lr_pi[k][s]] = v; }
            int info = 0;
            BL(dsyev_)("V", "U", &n, Xd, &n, wd, work, &lwork, &info);
            double rn2 = 0;
            HPN(dot)((size_t)n * r, L->R[k], L->R[k], t); rn2 = HP_GETD(t);
            const double sc = 1e-3 * sqrt(rn2 / r);
            int add = 1;
            while (add < 4 && r + add < L->cap[k] && wd[add] < 0.3 * wd[0]) add++;
            HPN(copy)((size_t)n * r, L->R[k], L->Rs[k]);
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < r; c++) HP_COPY(EL(L->R[k], (size_t)i * (r + add) + c), EL(L->Rs[k], (size_t)i * r + c));
                for (int c = 0; c < add; c++) HP_SETD(EL(L->R[k], (size_t)i * (r + add) + r + c), sc * Xd[(size_t)c * n + i]);
            }
            L->r[k] = r + add;
            if (J->verbose > 1) printf("       rank of block %d raised to %d (smallest eigenvalue of Z %.1e)\n", k + 1, r + add, lmin);
            HPN(copy)((size_t)m, L->q2, L->y);
            LR_EVAL();
            prev_err = fmin(prev_err, err); prev_pinf = 1e300;
            continue;
        }
        /* penalty: raised while the residual falls by less than a factor 10 per outer iteration */
        const double fac = J->sigfac > 0 ? J->sigfac : 10.0;
        if (pinf > (getenv("BRISK_HPLRRATE") ? atof(getenv("BRISK_HPLRRATE")) : 0.1) * prev_pinf && sigma < 1e12 * (J->sigma > 0 ? J->sigma : 10.0)) sigma *= fac;
        prev_pinf = pinf; prev_err = fmin(prev_err, err);
        (void)grel0;
    }
#undef LR_EVAL
    J->sigma = sigma;
    if (have_best) {
        /* exchange: R and y for the next level; X = RR' and Z = S(y) dense for the result */
        TMP(z);
        for (int k = 0; k < nb; k++) {
            const int n = L->n[k], r = rbest[k];             /* the rank the best factor was stored with */
            J->lr_rk[k] = r;
            memset(J->lr_R[k], 0, (size_t)n * L->cap[k] * Wm * sizeof(uint64_t));
            for (int i = 0; i < n; i++) for (int c = 0; c < r; c++) HP_EXPORT(J->lr_R[k] + ((size_t)i * L->cap[k] + c) * Wm, EL(L->Rb[k], (size_t)i * r + c), J);
            if (L->lp[k]) {
                for (int i = 0; i < n; i++) {
                    HP_MUL(z, EL(L->Rb[k], i), EL(L->Rb[k], i)); HP_EXPORT(J->X[k] + (size_t)i * Wm, z, J);
                    HP_EXPORT(J->Z[k] + (size_t)i * Wm, EL(L->svb[k], J->lr_diag[k][i]), J);
                }
                continue;
            }
            HPN(gemm_abt)(n, n, r, L->Rb[k], L->Rb[k], Zd);
            for (size_t i = 0; i < (size_t)n * n; i++) HP_EXPORT(J->X[k] + i * Wm, EL(Zd, i), J);
            HPN(lr_dense)(L, k, L->svb[k], Zd);
            for (size_t i = 0; i < (size_t)n * n; i++) HP_EXPORT(J->Z[k] + i * Wm, EL(Zd, i), J);
        }
        for (int p = 0; p < m; p++) HP_EXPORT(J->y + (size_t)p * Wm, EL(L->yb, p), J);
        J->have_iter = 1; J->lr_delta = delta_best;
        J->pinf = J->best_pinf; J->dinf = J->best_dinf; J->gap = J->best_gap; J->compl = J->best_compl; J->err = J->best_err;
    }
    free(L->Av); free(L->Cv); free(L->b);
    for (int a = 0; a < narr; a++) { for (int k = 0; k < nb; k++) free((*arr[a])[k]); free(*arr[a]); }
    for (int a = 0; a < 4; a++) { for (int k = 0; k < nb; k++) free((*pat[a])[k]); free(*pat[a]); }
    for (int a = 0; a < 6; a++) free(*vec[a]);
    free(rbest);
    free(L->n); free(L->lp); free(L->r); free(L->cap); free(Xd); free(wd); free(work); free(Zd);
    return rc;
}
#undef EL
#undef TMP
