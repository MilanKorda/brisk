/* SPDX-License-Identifier: Apache-2.0
 * hpfom.inc (5.0): first-order methods in high precision, compiled per precision after
 * hpipm.inc (same macros, same kernels).
 *
 * 1. A symmetric eigensolver: LAPACK in double, the eigenvectors re-orthogonalised in the
 *    working precision (Newton-Schulz), then cyclic Jacobi sweeps on the nearly diagonal
 *    V A V', which converge quadratically and do not mind clusters.
 * 2. The augmented Lagrangian method on the dual with semismooth Newton-CG inner solves
 *    (Zhao, Sun and Toh's SDPNAL scheme): no Schur complement is formed, the outer iteration
 *    gains a factor proportional to 1/sigma per step, the inner one converges quadratically.
 */

/* C = A B, A: ra x k (row-major), B: k x n; the rows of C are combinations of the rows of B */
static void HPN(gemm_rows)(int ra, int k, int n, const HPT *A, const HPT *B, HPT *C) {
#pragma omp parallel for schedule(static) if ((double)ra * k * n > 20000.0)
    for (int i = 0; i < ra; i++) {
        HPT *Ci = EL(C, (size_t)i * n);
        HPN(zero)((size_t)n, Ci);
        for (int l = 0; l < k; l++) HPN(axpy)((size_t)n, EL(A, (size_t)i * k + l), EL(B, (size_t)l * n), Ci);
    }
}
/* C = A B', A: ra x n, B: rb x n, C: ra x rb */
static void HPN(gemm_abt)(int ra, int rb, int n, const HPT *A, const HPT *B, HPT *C) {
#pragma omp parallel for schedule(static) if ((double)ra * rb * n > 20000.0)
    for (int i = 0; i < ra; i++)
        for (int j = 0; j < rb; j++) HPN(dot)((size_t)n, EL(A, (size_t)i * n), EL(B, (size_t)j * n), EL(C, (size_t)i * rb + j));
}
#ifndef HP_OWN_ROT
/* (x, y) <- (c x - s y, s x + c y) */
static void HPN(rot)(size_t n, const HPT *c, const HPT *s, HPT *x, HPT *y) {
    for (size_t i = 0; i < n; i++) {
        TMP(a); TMP(b); TMP(t);
        HP_MUL(a, c, EL(x, i)); HP_SUBMUL(a, s, EL(y, i));
        HP_MUL(b, s, EL(x, i)); HP_MUL(t, c, EL(y, i)); HP_ADD(b, b, t);
        HP_COPY(EL(x, i), a); HP_COPY(EL(y, i), b);
    }
}
#endif
/* Eigen-decomposition of the symmetric A (n x n): on return the rows of V are orthonormal
 * eigenvectors and w the eigenvalues, sorted in decreasing order. B and T are n x n work.
 * Returns the number of Jacobi sweeps, or -1 on failure. */
static int HPN(eig)(int n, const HPT *A, HPT *V, HPT *w, HPT *B, HPT *T) {
    double *Ad = (double *)malloc(sizeof(double) * (size_t)n * n), *wd = (double *)malloc(sizeof(double) * (size_t)n);
    int lw = 3 * n + 64, info = 0;
    double *work = (double *)malloc(sizeof(double) * (size_t)lw);
    for (size_t i = 0; i < (size_t)n * n; i++) Ad[i] = HP_GETD(EL(A, i));
    BL(dsyev_)("V", "U", &n, Ad, &n, wd, work, &lw, &info);
    free(work);
    if (info != 0) { free(Ad); free(wd); return -1; }
    /* rows of V = eigenvectors, largest eigenvalue first */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) HP_SETD(EL(V, (size_t)(n - 1 - j) * n + i), Ad[i + (size_t)j * n]);
    free(Ad); free(wd);
    TMP(half); HP_SETD(half, 0.5);
    /* orthonormal rows in the working precision: V <- V + (I - V V') V / 2, quadratic */
    for (int it = 0; it < 12; it++) {
        HPN(gemm_abt)(n, n, n, V, V, B);
        double e2 = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                HPT *b = EL(B, (size_t)i * n + j);
                HP_NEG(b, b);
                if (i == j) { TMP(one); HP_SETD(one, 1.0); HP_ADD(b, b, one); }
                const double d = HP_GETD(b); e2 += d * d;
                HP_MUL(b, b, half);
            }
        if (sqrt(e2) <= 8.0 * HP_EPS * n) break;
        HPN(gemm_rows)(n, n, n, B, V, T);
        for (size_t i = 0; i < (size_t)n * n; i++) HP_ADD(EL(V, i), EL(V, i), EL(T, i));
        if (e2 < 1e-60 && HP_EPS > 1e-40) break;                 /* quadratic: the next error is e2 */
    }
    /* B = V A V' */
    HPN(gemm_rows)(n, n, n, V, A, T);
    HPN(gemm_abt)(n, n, n, T, V, B);
    HPN(symm)(n, B);
    int sweeps = 0;
    for (; sweeps < 30; sweeps++) {
        double off = 0, dia = 0;
        for (int i = 0; i < n; i++) {
            const double d = HP_GETD(EL(B, (size_t)i * n + i)); dia += d * d;
            for (int j = i + 1; j < n; j++) { const double o = HP_GETD(EL(B, (size_t)i * n + j)); off += o * o; }
        }
        const double thr = 4.0 * HP_EPS * sqrt(dia + 2 * off) / n;
        if (sqrt(off) <= 4.0 * HP_EPS * sqrt(dia + 2 * off)) break;
        int nrot = 0;
        for (int p = 0; p < n - 1; p++)
            for (int q = p + 1; q < n; q++) {
                HPT *bpq = EL(B, (size_t)p * n + q), *bpp = EL(B, (size_t)p * n + p), *bqq = EL(B, (size_t)q * n + q);
                if (fabs(HP_GETD(bpq)) <= thr) continue;
                TMP(th); TMP(t); TMP(c); TMP(s); TMP(u); TMP(one);
                HP_SETD(one, 1.0);
                HP_SUB(th, bqq, bpp); HP_ADD(u, bpq, bpq); HP_DIV(th, th, u);         /* theta = (bqq - bpp) / (2 bpq) */
                HP_MUL(u, th, th); HP_ADD(u, u, one); HP_SQRT(u, u);                 /* sqrt(1 + theta^2) */
                if (HP_SGN(th) >= 0) { HP_ADD(u, th, u); HP_DIV(t, one, u); }
                else { HP_SUB(u, u, th); HP_DIV(t, one, u); HP_NEG(t, t); }
                HP_MUL(u, t, t); HP_ADD(u, u, one); HP_SQRT(u, u); HP_DIV(c, one, u);
                HP_MUL(s, t, c);
                /* the 2 x 2 block, then rows p and q, then the columns by symmetry */
                TMP(tb); HP_MUL(tb, t, bpq);
                TMP(npp); TMP(nqq); HP_SUB(npp, bpp, tb); HP_ADD(nqq, bqq, tb);
                HPN(rot)((size_t)n, c, s, EL(B, (size_t)p * n), EL(B, (size_t)q * n));
                HP_COPY(bpp, npp); HP_COPY(bqq, nqq); HP_SETD(bpq, 0.0); HP_SETD(EL(B, (size_t)q * n + p), 0.0);
                for (int k = 0; k < n; k++) {
                    if (k == p || k == q) continue;
                    HP_COPY(EL(B, (size_t)k * n + p), EL(B, (size_t)p * n + k));
                    HP_COPY(EL(B, (size_t)k * n + q), EL(B, (size_t)q * n + k));
                }
                HPN(rot)((size_t)n, c, s, EL(V, (size_t)p * n), EL(V, (size_t)q * n));
                nrot++;
            }
        if (!nrot) break;
    }
    /* sort decreasing (the order can change only locally) */
    for (int i = 0; i < n; i++) HP_COPY(EL(w, i), EL(B, (size_t)i * n + i));
    int *perm = (int *)malloc(sizeof(int) * (size_t)n), sorted = 1;
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = 1; i < n; i++) {
        const int pi = perm[i]; int j = i;
        while (j > 0) { TMP(d); HP_SUB(d, EL(w, perm[j - 1]), EL(w, pi)); if (HP_SGN(d) >= 0) break; perm[j] = perm[j - 1]; j--; sorted = 0; }
        perm[j] = pi;
    }
    if (!sorted) {
        HPN(copy)((size_t)n * n, V, T);
        for (int i = 0; i < n; i++) { HPN(copy)((size_t)n, EL(T, (size_t)perm[i] * n), EL(V, (size_t)i * n)); HP_COPY(EL(B, i), EL(w, perm[i])); }
        HPN(copy)((size_t)n, B, w);
    }
    free(perm);
    return sweeps;
}

/* ------------------------------------------------------------------ ALM with semismooth Newton */
typedef struct {
    HPN(st) S;                   /* the data and block layout, for Aop / ATy / addC */
    HPT **X, **W, **V, **lam, **Xp, **H, **G, **Mm, **Nn, **T1, **T2, **Zb, **Xb;
    int *na;                     /* positive eigenvalues of W per block (rows 0..na-1 of V) */
    HPT *y, *g, *d, *r, *p, *q, *yt, *yb;
    HPT sig[HPWMAX], isig[HPWMAX];
} HPN(fs);

/* W = X + sigma (A*(y) - C), its eigen-decomposition, and phi = -b'y + sum_{lambda>0} lambda^2 / (2 sigma) */
static int HPN(fom_eval)(HPN(fs) *F, const HPT *y, HPT *phi) {
    HPN(st) *S = &F->S;
    HPN(ATy)(S, y, F->H);
    HPN(addC)(S, F->H, -1);
    TMP(acc); TMP(t); HP_SETD(acc, 0.0);
    int bad = 0;
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        for (size_t i = 0; i < S->len[k]; i++) { HP_MUL(t, F->sig, EL(F->H[k], i)); HP_ADD(EL(F->W[k], i), EL(F->X[k], i), t); }
        if (S->lp[k]) {
            for (int i = 0; i < n; i++) if (HP_SGN(EL(F->W[k], i)) > 0) HP_ADDMUL(acc, EL(F->W[k], i), EL(F->W[k], i));
            continue;
        }
        if (HPN(eig)(n, F->W[k], F->V[k], F->lam[k], F->T1[k], F->T2[k]) < 0) bad = 1;
        int na = 0;
        while (na < n && HP_SGN(EL(F->lam[k], na)) > 0) { HP_ADDMUL(acc, EL(F->lam[k], na), EL(F->lam[k], na)); na++; }
        F->na[k] = na;
    }
    TMP(h); HP_SETD(h, 0.5); HP_MUL(acc, acc, h); HP_MUL(acc, acc, F->isig);
    HPN(dot)((size_t)S->m, S->b, y, t);
    HP_SUB(phi, acc, t);
    return bad;
}
/* Xp = the positive part of W, from the current decomposition (the smaller of the two sides) */
static void HPN(fom_xp)(HPN(fs) *F) {
    HPN(st) *S = &F->S;
    for (int k = 0; k < S->nb; k++) {
        const int n = S->n[k];
        if (S->lp[k]) { for (int i = 0; i < n; i++) { if (HP_SGN(EL(F->W[k], i)) > 0) HP_COPY(EL(F->Xp[k], i), EL(F->W[k], i)); else HP_SETD(EL(F->Xp[k], i), 0.0); } continue; }
        const int na = F->na[k], pos = na <= n - na, o = pos ? 0 : na, cnt = pos ? na : n - na;
        const HPT *Vo = EL(F->V[k], (size_t)o * n);
        HPT *T = F->T1[k], *Xp = F->Xp[k];
        for (int a = 0; a < cnt; a++) {                       /* T = Lambda V (the chosen rows) */
            HPN(copy)((size_t)n, EL(Vo, (size_t)a * n), EL(T, (size_t)a * n));
            HPN(scal)((size_t)n, EL(F->lam[k], o + a), EL(T, (size_t)a * n));
        }
#pragma omp parallel for schedule(static) if ((double)n * n * cnt > 20000.0)
        for (int i = 0; i < n; i++) {
            HPT *Xi = EL(Xp, (size_t)i * n);
            HPN(zero)((size_t)n, Xi);
            for (int a = 0; a < cnt; a++) HPN(axpy)((size_t)n, EL(Vo, (size_t)a * n + i), EL(T, (size_t)a * n), Xi);
        }
        if (!pos) for (size_t i = 0; i < S->len[k]; i++) HP_SUB(EL(Xp, i), EL(F->W[k], i), EL(Xp, i));   /* W - (negative part) */
        HPN(symm)(n, Xp);
    }
}
/* Out = P[H], the generalised Jacobian of the projection at W applied to the symmetric H (block k):
 * Q (Omega o (Q' H Q)) Q', through the smaller of the two eigenvalue sets */
static void HPN(fom_P)(HPN(fs) *F, int k, const HPT *H, HPT *Out) {
    HPN(st) *S = &F->S;
    const int n = S->n[k];
    if (S->lp[k]) { for (int i = 0; i < n; i++) { if (HP_SGN(EL(F->W[k], i)) > 0) HP_COPY(EL(Out, i), EL(H, i)); else HP_SETD(EL(Out, i), 0.0); } return; }
    const int na = F->na[k], pos = na <= n - na, o = pos ? 0 : na, cnt = pos ? na : n - na;
    if (cnt == 0) { if (pos) HPN(zero)(S->len[k], Out); else HPN(copy)(S->len[k], H, Out); return; }
    const HPT *V = F->V[k], *Vo = EL(V, (size_t)o * n), *lam = F->lam[k];
    HPT *G = F->G[k], *M = F->Mm[k], *N = F->Nn[k];
    HPN(gemm_rows)(cnt, n, n, Vo, H, G);                      /* G = V_o H        (cnt x n) */
    HPN(gemm_abt)(cnt, n, n, G, V, M);                        /* M = G V'         (cnt x n) */
    TMP(half); HP_SETD(half, 0.5);
    for (int a = 0; a < cnt; a++)
        for (int b = 0; b < n; b++) {
            HPT *e = EL(M, (size_t)a * n + b);
            const int inside = pos ? b < na : b >= na;
            if (inside) HP_MUL(e, e, half);
            else {
                /* pos: Omega = lam_a / (lam_a - lam_b), a positive, b not; else 1 - Omega = -lam_c / (lam_b - lam_c), c = o + a */
                TMP(den); TMP(w);
                if (pos) { HP_SUB(den, EL(lam, a), EL(lam, b)); HP_DIV(w, EL(lam, a), den); }
                else { HP_SUB(den, EL(lam, b), EL(lam, o + a)); HP_DIV(w, EL(lam, o + a), den); HP_NEG(w, w); }
                HP_MUL(e, e, w);
            }
        }
    HPN(gemm_rows)(cnt, n, n, M, V, N);                       /* N = M V          (cnt x n) */
#pragma omp parallel for schedule(static) if ((double)n * n * cnt > 20000.0)
    for (int i = 0; i < n; i++) {
        HPT *Oi = EL(Out, (size_t)i * n);
        HPN(zero)((size_t)n, Oi);
        for (int a = 0; a < cnt; a++) HPN(axpy)((size_t)n, EL(Vo, (size_t)a * n + i), EL(N, (size_t)a * n), Oi);
    }
    /* Out <- Out + Out' (and H - that, for the negative side) */
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            HPT *x = EL(Out, (size_t)i * n + j), *z = EL(Out, (size_t)j * n + i);
            TMP(s); HP_ADD(s, x, z);
            if (!pos) HP_SUB(s, EL(H, (size_t)i * n + j), s);
            HP_COPY(x, s); HP_COPY(z, s);
        }
}
/* out = sigma A(P[A*(d)]) + eps d */
static void HPN(fom_Vd)(HPN(fs) *F, const HPT *d, const HPT *eps, HPT *out) {
    HPN(st) *S = &F->S;
    HPN(ATy)(S, d, F->H);
    for (int k = 0; k < S->nb; k++) HPN(fom_P)(F, k, F->H[k], F->T2[k]);
    HPN(Aop)(S, F->T2, out);
    for (int p = 0; p < S->m; p++) { HPT *o = EL(out, p); HP_MUL(o, o, F->sig); HP_ADDMUL(o, eps, EL(d, p)); }
}

/* Returns as HPN(ipm): 0 tolerance reached, 1 level done or exhausted, 2 limit, 3 failure. */
static int HPN(alm)(HPJob *J) {
    HPN(fs) F_, *F = &F_;
    memset(F, 0, sizeof(*F));
    HPN(st) *S = &F->S;
    S->J = J; S->m = J->m; S->nb = J->nblk;
    const int m = S->m, nb = S->nb;
    const size_t Wm = (size_t)J->Lm + 2;
    S->Av = HPN(alloc)(J->nnzA); S->Cv = HPN(alloc)(J->nnzC); S->b = HPN(alloc)((size_t)m);
    for (size_t e = 0; e < J->nnzA; e++) HP_IMPORT(EL(S->Av, e), J->Av + e * Wm, J);
    for (size_t e = 0; e < J->nnzC; e++) HP_IMPORT(EL(S->Cv, e), J->Cv + e * Wm, J);
    for (int p = 0; p < m; p++) HP_IMPORT(EL(S->b, p), J->b + (size_t)p * Wm, J);
    S->n = (int *)malloc(sizeof(int) * (size_t)nb); S->lp = (int *)malloc(sizeof(int) * (size_t)nb); S->len = (size_t *)malloc(sizeof(size_t) * (size_t)nb);
    F->na = (int *)calloc((size_t)nb, sizeof(int));
    HPT ***arr[] = { &F->X, &F->W, &F->V, &F->lam, &F->Xp, &F->H, &F->G, &F->Mm, &F->Nn, &F->T1, &F->T2, &F->Zb, &F->Xb };
    const int narr = (int)(sizeof(arr) / sizeof(arr[0]));
    for (int a = 0; a < narr; a++) *arr[a] = (HPT **)malloc(sizeof(HPT *) * (size_t)nb);
    for (int k = 0; k < nb; k++) {
        S->lp[k] = J->bs[k] < 0; S->n[k] = abs(J->bs[k]);
        S->len[k] = S->lp[k] ? (size_t)S->n[k] : (size_t)S->n[k] * S->n[k];
        for (int a = 0; a < narr; a++) (*arr[a])[k] = HPN(alloc)(S->len[k]);
    }
    HPT **vec[] = { &F->y, &F->g, &F->d, &F->r, &F->p, &F->q, &F->yt, &F->yb };
    for (int a = 0; a < 8; a++) *vec[a] = HPN(alloc)((size_t)m);
    const double nb_ = HPN(norm)((size_t)m, S->b), nC = HPN(norm)(J->nnzC, S->Cv);
    if (J->have_iter) {
        for (int k = 0; k < nb; k++) for (size_t i = 0; i < S->len[k]; i++) HP_IMPORT(EL(F->X[k], i), J->X[k] + i * Wm, J);
        for (int p = 0; p < m; p++) HP_IMPORT(EL(F->y, p), J->y + (size_t)p * Wm, J);
    }
    double sigma = J->sigma > 0 ? J->sigma : 1.0;
    TMP(one); TMP(phi); TMP(phit); TMP(t); TMP(u); TMP(eps); TMP(pobj); TMP(dobj);
    HP_SETD(one, 1.0);
    int rc = 2, have_best = 0, nbad = 0;
    double best = 1e300, prev_err = 1.0, pinf = 1e300;
    HP_SETD(F->sig, sigma); HP_DIV(F->isig, one, F->sig);
    if (HPN(fom_eval)(F, F->y, phi)) { rc = 3; goto done; }
    HPN(fom_xp)(F);
    HPN(Aop)(S, F->Xp, F->g);
    for (int p = 0; p < m; p++) HP_SUB(EL(F->g, p), EL(F->g, p), EL(S->b, p));
    for (int outer = 0; ; outer++) {
        /* inner: semismooth Newton on phi(y) until the primal residual is a fraction of the last error */
        const double intol = fmax(0.5 * J->tol, 1e-2 * fmin(prev_err, 1.0));
        int newt = 0, cgtot = 0;
        for (;; newt++) {
            const double gn = HPN(norm)((size_t)m, F->g);
            pinf = gn / (1 + nb_);
            if (pinf <= intol || newt >= 50) break;
            if ((J->timelimit > 0 && wtime() - J->tstart > J->timelimit) || brisk_stop_flag) break;
            /* CG on (V + eps I) d = -g */
            HP_SETD(eps, sigma * fmin(1e-4, 0.1 * pinf));
            HPN(zero)((size_t)m, F->d);
            for (int p = 0; p < m; p++) { HP_NEG(EL(F->r, p), EL(F->g, p)); HP_COPY(EL(F->p, p), EL(F->r, p)); }
            TMP(rr); TMP(rr2); TMP(al); TMP(be); TMP(pq);
            HPN(dot)((size_t)m, F->r, F->r, rr);
            const double eta = fmin(1e-2, pinf) * gn;
            const int cgmax = J->cgmax > 0 ? J->cgmax : 500;
            for (int cg = 0; cg < cgmax; cg++) {
                HPN(fom_Vd)(F, F->p, eps, F->q);
                HPN(dot)((size_t)m, F->p, F->q, pq);
                if (HP_SGN(pq) <= 0) break;
                HP_DIV(al, rr, pq);
                HPN(axpy)((size_t)m, al, F->p, F->d);
                HP_NEG(t, al); HPN(axpy)((size_t)m, t, F->q, F->r);
                HPN(dot)((size_t)m, F->r, F->r, rr2);
                cgtot++;
                const double rn = HP_GETD(rr2);
                if (!(rn > 0) || sqrt(rn) <= eta) break;
                HP_DIV(be, rr2, rr); HP_COPY(rr, rr2);
                for (int p = 0; p < m; p++) { HPT *pp = EL(F->p, p); HP_MUL(pp, pp, be); HP_ADD(pp, pp, EL(F->r, p)); }
            }
            /* Armijo line search on phi */
            TMP(gd); HPN(dot)((size_t)m, F->g, F->d, gd);
            if (HP_SGN(gd) >= 0) { for (int p = 0; p < m; p++) HP_NEG(EL(F->d, p), EL(F->g, p)); HPN(dot)((size_t)m, F->g, F->d, gd); }
            double alpha = 1.0; int ok = 0;
            for (int ls = 0; ls < 40; ls++, alpha *= 0.5) {
                TMP(a); HP_SETD(a, alpha);
                HPN(copy)((size_t)m, F->y, F->yt); HPN(axpy)((size_t)m, a, F->d, F->yt);
                if (HPN(fom_eval)(F, F->yt, phit)) { rc = 3; goto done; }
                HP_SETD(u, 1e-4 * alpha); HP_MUL(u, u, gd); HP_ADD(u, u, phi); HP_SUB(u, phit, u);     /* phit - (phi + 1e-4 alpha g'd) */
                if (HP_SGN(u) <= 0) { ok = 1; break; }
            }
            if (!ok) { HPN(fom_eval)(F, F->y, phi); break; }           /* no decrease: the inner iteration ends here */
            HPN(copy)((size_t)m, F->yt, F->y); HP_COPY(phi, phit);
            HPN(fom_xp)(F);
            HPN(Aop)(S, F->Xp, F->g);
            for (int p = 0; p < m; p++) HP_SUB(EL(F->g, p), EL(F->g, p), EL(S->b, p));
        }
        J->iters += newt;
        /* measures of (Xp, y, Z = (Xp - W) / sigma) */
        double dx2 = 0;
        for (int k = 0; k < nb; k++) {
            for (size_t i = 0; i < S->len[k]; i++) { HP_SUB(EL(F->T1[k], i), EL(F->Xp[k], i), EL(F->X[k], i)); }
            HPN(dot)(S->len[k], F->T1[k], F->T1[k], t); dx2 += HP_GETD(t);
        }
        const double dinf = sqrt(fmax(dx2, 0.0)) / sigma / (1 + nC);
        HPN(dot)((size_t)m, S->b, F->y, dobj);
        HP_SETD(pobj, 0.0);
        for (int k = 0; k < nb; k++)
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
                const int i = J->ci[e], j = J->cj[e], n = S->n[k];
                if (S->lp[k]) HP_ADDMUL(pobj, EL(S->Cv, e), EL(F->Xp[k], i));
                else if (i == j) HP_ADDMUL(pobj, EL(S->Cv, e), EL(F->Xp[k], (size_t)i * n + i));
                else { HP_ADD(u, EL(F->Xp[k], (size_t)i * n + j), EL(F->Xp[k], (size_t)j * n + i)); HP_ADDMUL(pobj, EL(S->Cv, e), u); }
            }
        const double po = HP_GETD(pobj), dob = HP_GETD(dobj);
        HP_SUB(t, pobj, dobj);
        const double gap = HP_GETD(t) / (1 + fabs(po) + fabs(dob));
        const double err = fmax(fmax(pinf, dinf), fabs(gap));
        if (J->verbose > 1)
            printf("%4d %-3s %+.14e %+.14e %8.1e %8.1e %9.1e  sigma %7.1e  %2d Newton %4d CG %7.1fs\n", outer, HPN(name), po, dob, pinf, dinf, gap, sigma, newt, cgtot, wtime() - J->tstart);
        J->pinf = pinf; J->dinf = dinf; J->gap = gap; J->compl = 0; J->err = err;
        if (err < best) {
            best = err; have_best = 1;
            for (int k = 0; k < nb; k++) {
                HPN(copy)(S->len[k], F->Xp[k], F->Xb[k]);
                for (size_t i = 0; i < S->len[k]; i++) { HP_SUB(t, EL(F->Xp[k], i), EL(F->W[k], i)); HP_MUL(EL(F->Zb[k], i), t, F->isig); }
            }
            HPN(copy)((size_t)m, F->y, F->yb);
            HP_EXPORT(J->pobj, pobj, J); HP_EXPORT(J->dobj, dobj, J);
            J->best_pinf = pinf; J->best_dinf = dinf; J->best_gap = gap; J->best_compl = 0; J->best_err = err;
        }
        if (err <= J->tol) { rc = 0; break; }
        if (err <= J->level_tol) { rc = 1; break; }
        if (outer > 0) { if (!(err < 0.7 * prev_err)) nbad++; else nbad = 0; if (nbad >= 4) { rc = 1; break; } }
        prev_err = err;
        if (outer >= J->maxit || (J->timelimit > 0 && wtime() - J->tstart > J->timelimit) || brisk_stop_flag) { rc = 2; break; }
        /* X <- Xp; a larger sigma while the dual residual lags */
        for (int k = 0; k < nb; k++) HPN(copy)(S->len[k], F->Xp[k], F->X[k]);
        if (dinf > 0.2 * pinf && sigma < 1e14) sigma *= J->sigfac > 1 ? J->sigfac : 10.0;
        HP_SETD(F->sig, sigma); HP_DIV(F->isig, one, F->sig);
        if (HPN(fom_eval)(F, F->y, phi)) { rc = 3; break; }
        HPN(fom_xp)(F);
        HPN(Aop)(S, F->Xp, F->g);
        for (int p = 0; p < m; p++) HP_SUB(EL(F->g, p), EL(F->g, p), EL(S->b, p));
    }
    J->sigma = sigma;
    if (have_best) {
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < S->len[k]; i++) { HP_EXPORT(J->X[k] + i * Wm, EL(F->Xb[k], i), J); HP_EXPORT(J->Z[k] + i * Wm, EL(F->Zb[k], i), J); }
        for (int p = 0; p < m; p++) HP_EXPORT(J->y + (size_t)p * Wm, EL(F->yb, p), J);
        J->have_iter = 1;
    }
done:
    for (int a = 0; a < narr; a++) { for (int k = 0; k < nb; k++) free((*arr[a])[k]); free(*arr[a]); }
    for (int a = 0; a < 8; a++) free(*vec[a]);
    free(S->Av); free(S->Cv); free(S->b); free(S->n); free(S->lp); free(S->len); free(F->na);
    return rc;
}
