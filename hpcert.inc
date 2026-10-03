/* SPDX-License-Identifier: Apache-2.0
 * hpcert.inc (5.2): ex-post certification in high precision (-prec with -bound p|d,
 * -certify-x, -certify-y). Included by hpsolve.c after hppre.inc.
 *
 * The problem is the one as read: (P) min <C,X>, A(X) = b, X >= 0; (D) max b'y,
 * Z = C - A*(y) >= 0. A certificate is a matrix X~ (side p) or a vector y~ (side d) in mpx at
 * the master precision; the check proves that a feasible point lies within a computed
 * distance of it and returns a bound on the optimal value that holds for the exact data.
 *
 * The arithmetic. mpx truncates. With L limbs an addition or subtraction errs by a unit or
 * two of the last limb of the larger operand, a product by less than L units of its last
 * limb, a quotient or square root (Newton) by a few: of the order 2^(-64 (L - 1)) L relative
 * to the larger operand (sums) or to the result (the others). The bounds below use the
 * model unit
 *        u = 2^(-64 (L - 2)),
 * one whole limb above that, for every operation, in the form without a guard digit:
 *        fl(a +- b) = a (1 + e1) +- b (1 + e2),   fl(a op b) = (a op b)(1 + e),   |e| <= u.
 * A decimal of the file converted by mpx_from_str is within u (relative) of its value
 * (a few dozen operations). tools/mpx_check.py tests these statements against exact
 * integer arithmetic. Quantities that only enter the bounds (sums of absolute values,
 * norms) are evaluated in double and inflated by 1 %.
 *
 * p. r^ = fl(b - A(X~)); |r^_i - r_i| <= (k_i + 8) u (|b_i| + sum |a| |x|) = E_i, so
 *    ||r|| <= || |r^| + E ||. The point X* = X~ + A*(G^{-1} r), G = A A*, is feasible and
 *    ||X* - X~||_F <= ||r|| / sqrt(lambda_min(G)) = delta; lambda_min(G) is bounded below in
 *    double with directed rounding (boundcert.c: the doubles of the data are within one ulp
 *    of the decimals). Certified when lambda_min(X~_k) >= delta for every block (LP: every
 *    entry); then  optimum <= <C,X~> + E_C + ||C||_F delta.
 * d. Z^ = fl(C - A*(y~)), entrywise within (k + 8) u (|C| + sum |y| |A|); its Frobenius norm
 *    over block k is rho_k. Split pairs (LP columns that are exact negatives, cost included)
 *    are equalities of (D): their residuals are removed by y* = y~ - Ap'(Gp^{-1} z_p), a
 *    change of at most delta = ||z_p|| / sqrt(lambda_min(Gp)) in norm, which moves Z_k by at
 *    most delta N_k, N_k = (sum_i ||A_ik||_F^2)^(1/2). Certified when
 *    lambda_min(Z^_k) >= rho_k + delta N_k; then  optimum >= b'y~ - E_b - ||b|| delta.
 * lambda_min(M) >= c for a symmetric mpx matrix M: the Cholesky factorization of
 *    W = fl(M - c' I) runs to the end, where c' = c + 2 (g tr(M) + u max |M_ii|),
 *    g = gamma_{3n+10}(u). Then W + E = L L' with |E| <= g |L| |L'|, ||E||_2 <= g tr(W) / (1 - g),
 *    and lambda_min(M) >= c' - g tr(W) / (1 - g) - u max(|M_ii|, c') >= c.
 */

#define HPC_U(J) ldexp(1.0, -64 * ((J)->Lm - 2))

/* an upper bound of |a| as a double */
static double hpc_abs(const uint64_t *a) {
    if (mpx_sgn(a) == 0) return 0.0;
    const double d = fabs(mpx_getd(a));
    if (d == 0.0) return 1e-300;
    return d * (1 + 1e-12);
}

/* (sum a_i^2)^(1/2) of nonnegative doubles without underflow of the squares, rounded up */
static double hpc_norm(size_t n, const double *a) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) if (a[i] > mx) mx = a[i];
    if (mx == 0 || !isfinite(mx)) return mx;
    double s2 = 0;
    for (size_t i = 0; i < n; i++) { const double q = a[i] / mx; s2 += q * q; }
    return mx * sqrt(s2) * 1.01;
}

/* ---- split pairs: LP variables whose columns (cost included) are exact negatives ---- */
typedef struct { int np, *K1, *I1, *K2, *I2; char **mark; int nblk; } HPPairs;
static void hp_pairs_free(HPPairs *Q) {
    if (Q->mark) { for (int k = 0; k < Q->nblk; k++) free(Q->mark[k]); free(Q->mark); }
    free(Q->K1); free(Q->I1); free(Q->K2); free(Q->I2);
    memset(Q, 0, sizeof(*Q));
}
static void hp_find_pairs(const HPJob *J, HPPairs *Q) {
    const size_t Wm = (size_t)mpx_W;
    const int nb = J->nblk;
    memset(Q, 0, sizeof(*Q));
    Q->nblk = nb;
    Q->mark = (char **)calloc((size_t)nb + 1, sizeof(char *));
    int nv = 0;
    for (int k = 0; k < nb; k++) if (J->bs[k] < 0) { nv += -J->bs[k]; Q->mark[k] = (char *)calloc((size_t)(-J->bs[k]) + 1, 1); }
    if (!nv) return;
    /* cost entry of each LP variable */
    long **ce = (long **)calloc((size_t)nb, sizeof(long *));
    for (int k = 0; k < nb; k++) {
        if (J->bs[k] >= 0) continue;
        ce[k] = (long *)malloc(sizeof(long) * (size_t)(-J->bs[k]));
        for (int i = 0; i < -J->bs[k]; i++) ce[k][i] = -1;
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) ce[k][J->ci[e]] = (long)e;
    }
    PVar *V = (PVar *)malloc(sizeof(PVar) * (size_t)(nv + 1));
    nv = 0;
    for (int k = 0; k < nb; k++) {
        if (J->bs[k] >= 0) continue;
        for (int i = 0; i < -J->bs[k]; i++) {
            if (J->lpptr[k][i + 1] == J->lpptr[k][i]) continue;
            uint64_t h = 1469598103934665603ULL;
            for (int q = J->lpptr[k][i]; q < J->lpptr[k][i + 1]; q++) {
                const double d = fabs(mpx_getd(J->Av + (size_t)J->lpe[k][q] * Wm));
                uint64_t bits; memcpy(&bits, &d, 8);
                h = (h ^ (uint64_t)J->lpc[k][q]) * 1099511628211ULL; h = (h ^ bits) * 1099511628211ULL;
            }
            V[nv].blk = k; V[nv].i = i; V[nv].h = h; nv++;
        }
    }
    qsort(V, (size_t)nv, sizeof(PVar), cmp_pvar);
    Q->K1 = (int *)malloc(sizeof(int) * (size_t)(nv + 1)); Q->I1 = (int *)malloc(sizeof(int) * (size_t)(nv + 1));
    Q->K2 = (int *)malloc(sizeof(int) * (size_t)(nv + 1)); Q->I2 = (int *)malloc(sizeof(int) * (size_t)(nv + 1));
    char *used = (char *)calloc((size_t)nv + 1, 1);
    uint64_t t[MPX_MAXL + 2];
    for (int x = 0; x < nv; x++) {
        if (used[x]) continue;
        for (int z = x + 1; z < nv && V[z].h == V[x].h; z++) {
            if (used[z]) continue;
            const int k1 = V[x].blk, i1 = V[x].i, k2 = V[z].blk, i2 = V[z].i;
            const int n1 = J->lpptr[k1][i1 + 1] - J->lpptr[k1][i1];
            if (n1 != J->lpptr[k2][i2 + 1] - J->lpptr[k2][i2]) continue;
            int ok = 1;
            for (int q = 0; q < n1 && ok; q++) {
                const int q1 = J->lpptr[k1][i1] + q, q2 = J->lpptr[k2][i2] + q;
                if (J->lpc[k1][q1] != J->lpc[k2][q2]) { ok = 0; break; }
                mpx_add(t, J->Av + (size_t)J->lpe[k1][q1] * Wm, J->Av + (size_t)J->lpe[k2][q2] * Wm);
                if (mpx_sgn(t) != 0) ok = 0;
            }
            if (!ok) continue;
            const long c1 = ce[k1][i1], c2 = ce[k2][i2];
            if ((c1 < 0) != (c2 < 0)) continue;
            if (c1 >= 0) { mpx_add(t, J->Cv + (size_t)c1 * Wm, J->Cv + (size_t)c2 * Wm); if (mpx_sgn(t) != 0) continue; }
            Q->K1[Q->np] = k1; Q->I1[Q->np] = i1; Q->K2[Q->np] = k2; Q->I2[Q->np] = i2; Q->np++;
            Q->mark[k1][i1] = Q->mark[k2][i2] = 1;
            used[x] = used[z] = 1;
            break;
        }
    }
    free(used); free(V);
    for (int k = 0; k < nb; k++) free(ce[k]);
    free(ce);
}

/* Z = C - A*(y) (Z allocated, full symmetric blocks). Sabs (optional): per block, the same
 * positions with |C| + sum |y| |A| as doubles (upper bounds). */
static void hp_formZ(const HPJob *J, const uint64_t *y, uint64_t **Z, double **Sabs) {
    const size_t Wm = (size_t)mpx_W;
    for (int k = 0; k < J->nblk; k++) {
        const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
        const size_t len = lp ? (size_t)n : (size_t)n * n;
        memset(Z[k], 0, len * Wm * sizeof(uint64_t));
        if (Sabs) memset(Sabs[k], 0, len * sizeof(double));
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
            const int i = J->ci[e], j = lp ? i : J->cj[e];
            const uint64_t *v = J->Cv + e * Wm;
            const size_t p1 = lp ? (size_t)i : (size_t)i * n + j, p2 = lp ? (size_t)i : (size_t)j * n + i;
            mpx_copy(Z[k] + p1 * Wm, v); if (p2 != p1) mpx_copy(Z[k] + p2 * Wm, v);
            if (Sabs) { const double a = hpc_abs(v); Sabs[k][p1] += a; if (p2 != p1) Sabs[k][p2] += a; }
        }
        for (int tt = 0; tt < J->ncon[k]; tt++) {
            const uint64_t *yc = y + (size_t)J->con[k][tt] * Wm;
            if (mpx_sgn(yc) == 0) continue;
            const double ya = Sabs ? hpc_abs(yc) : 0.0;
            for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                const int i = J->ei[e], j = lp ? i : J->ej[e];
                const uint64_t *v = J->Av + e * Wm;
                const size_t p1 = lp ? (size_t)i : (size_t)i * n + j, p2 = lp ? (size_t)i : (size_t)j * n + i;
                mpx_submul(Z[k] + p1 * Wm, yc, v); if (p2 != p1) mpx_submul(Z[k] + p2 * Wm, yc, v);
                if (Sabs) { const double a = ya * hpc_abs(v); Sabs[k][p1] += a; if (p2 != p1) Sabs[k][p2] += a; }
            }
        }
    }
}

/* 1 when lambda_min(M) >= need is proved for the symmetric n x n mpx matrix whose upper
 * triangle (row-major, i <= j) is in M; W is n x n work. */
static int hpc_lammin_ge(const HPJob *J, int n, const uint64_t *M, double need, uint64_t *W) {
    const size_t Wm = (size_t)mpx_W;
    const double u = HPC_U(J);
    double tr = 0, dmax = 0;
    for (int i = 0; i < n; i++) {
        const uint64_t *d = M + ((size_t)i * n + i) * Wm;
        if (mpx_sgn(d) <= 0) return 0;
        const double a = hpc_abs(d);
        tr += a; if (a > dmax) dmax = a;
    }
    if (!isfinite(tr) || !isfinite(need)) return 0;
    const double g = (3.0 * n + 10.0) * u / (1 - (3.0 * n + 10.0) * u);
    const double c = (need + 2.0 * (g * tr + u * dmax)) * 1.01 + 1e-290;
    uint64_t cm[MPX_MAXL + 2];
    mpx_setd(cm, c);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            /* lower triangle of W from the upper triangle of M */
            if (i == j) mpx_sub(W + ((size_t)i * n + i) * Wm, M + ((size_t)i * n + i) * Wm, cm);
            else mpx_copy(W + ((size_t)i * n + j) * Wm, M + ((size_t)j * n + i) * Wm);
        }
    return cholr_mp(n, W, 0.0, NULL, NULL, 0, NULL, NULL) == 0;
}

typedef struct {
    int ok;
    char why[200];
    uint64_t value[MPX_MAXL + 2];     /* <C,X~> or b'y~ as computed */
    uint64_t bound[MPX_MAXL + 2];     /* the rigorous bound (meaningful also when not certified: -/+ slack) */
    double resid;                     /* p: ||b - A(X~)|| / (1 + ||b||); d: the pair equalities' residual */
    double delta;                     /* the distance to the feasible point */
} HPCertOut;

/* The check. side 1: X (per block; the certificate is the symmetric matrix of the upper
 * triangles); side 2: y. lgG / lgP: the lower bounds of lambda_min(A A*) and of the pair
 * equalities' Gram matrix from boundcert.c (<= 0 or NaN: none). Returns 1 when certified. */
static int hp_cert_check(const HPJob *J, const HPPairs *Q, double lgG, double lgP, int side,
                         uint64_t *const *X, const uint64_t *y, HPCertOut *out) {
    const size_t Wm = (size_t)mpx_W;
    const int nb = J->nblk, m = J->m;
    const double u = HPC_U(J);
    uint64_t t[MPX_MAXL + 2], s[MPX_MAXL + 2];
    memset(out, 0, sizeof(*out));
    int nmax = 1;
    for (int k = 0; k < nb; k++) if (J->bs[k] > nmax) nmax = J->bs[k];
    uint64_t *W = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)nmax * nmax + 1) * Wm);
    int ok = 1;
    if (side == 1) {
        /* r^ = b - A(X~), the absolute sums and the term counts */
        uint64_t *ax = (uint64_t *)calloc((size_t)m * Wm, sizeof(uint64_t));
        double *S = (double *)calloc((size_t)m + 1, sizeof(double));
        int *cnt = (int *)calloc((size_t)m + 1, sizeof(int));
        uint64_t cx[MPX_MAXL + 2];
        mpx_setd(cx, 0.0);
        double Scx = 0, c2 = 0; size_t ncx = 0;
        for (int k = 0; k < nb; k++) {
            const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
                const int i = J->ci[e], j = lp ? i : J->cj[e];
                const uint64_t *v = J->Cv + e * Wm, *x = lp ? X[k] + (size_t)i * Wm : X[k] + ((size_t)i * n + j) * Wm;
                const double w = (!lp && i != j) ? 2.0 : 1.0, a = hpc_abs(v);
                mpx_mul(t, v, x); if (w > 1) mpx_add(t, t, t);
                mpx_add(cx, cx, t);
                Scx += w * a * hpc_abs(x); c2 += w * a * a; ncx++;
            }
            for (int tt = 0; tt < J->ncon[k]; tt++) {
                const int c = J->con[k][tt];
                for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                    const int i = J->ei[e], j = lp ? i : J->ej[e];
                    const uint64_t *v = J->Av + e * Wm, *x = lp ? X[k] + (size_t)i * Wm : X[k] + ((size_t)i * n + j) * Wm;
                    const double w = (!lp && i != j) ? 2.0 : 1.0;
                    mpx_mul(t, v, x); if (w > 1) mpx_add(t, t, t);
                    mpx_add(ax + (size_t)c * Wm, ax + (size_t)c * Wm, t);
                    S[c] += w * hpc_abs(v) * hpc_abs(x); cnt[c] += 2;
                }
            }
        }
        double b2 = 0;
        for (int c = 0; c < m; c++) {
            const uint64_t *bc = J->b + (size_t)c * Wm;
            mpx_sub(t, bc, ax + (size_t)c * Wm);
            const double ba = hpc_abs(bc), E = (cnt[c] + 8.0) * u * (ba + S[c]) * 1.01;
            S[c] = hpc_abs(t) + E; b2 += ba * ba;          /* (S reused: the bound of |r_c|) */
        }
        const double rn = hpc_norm((size_t)m, S);
        free(ax); free(S); free(cnt);
        out->resid = rn / (1 + sqrt(b2));
        double delta = 0;
        if (rn > 0) {
            if (!(lgG > 0)) { snprintf(out->why, sizeof(out->why), "lambda_min(A A*) is not bounded away from 0 in double (dependent constraints, or m > 6000)"); ok = 0; delta = INFINITY; }
            else delta = rn / sqrt(lgG) * 1.01 + 1e-290;
        }
        out->delta = delta;
        for (int k = 0; k < nb && ok; k++) {
            const int n = abs(J->bs[k]);
            if (J->bs[k] < 0) {
                mpx_setd(t, delta * 1.01 + 1e-290);
                for (int i = 0; i < n && ok; i++)
                    if (mpx_cmp(X[k] + (size_t)i * Wm, t) < 0) { snprintf(out->why, sizeof(out->why), "an entry of the LP block %d of X is below the distance %.1e to the feasible point", k + 1, delta); ok = 0; }
            } else if (!hpc_lammin_ge(J, n, X[k], delta, W)) {
                snprintf(out->why, sizeof(out->why), "lambda_min of block %d of X is not above the distance %.1e to the feasible point", k + 1, delta); ok = 0;
            }
        }
        mpx_copy(out->value, cx);
        const double slack = ((double)ncx * 2 + 8.0) * u * Scx * 1.01 + (sqrt(c2) + 1e-140) * 1.01 * (isfinite(delta) ? delta : 0.0) * 1.01;
        mpx_setd(s, slack + 1e-290); mpx_add(out->bound, cx, s);
    } else {
        /* Z^ and its absolute sums */
        uint64_t **Z = (uint64_t **)calloc((size_t)nb + 1, sizeof(uint64_t *));
        double **Sa = (double **)calloc((size_t)nb + 1, sizeof(double *));
        for (int k = 0; k < nb; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            Z[k] = (uint64_t *)malloc(sizeof(uint64_t) * (len + 1) * Wm); Sa[k] = (double *)malloc(sizeof(double) * (len + 1));
        }
        hp_formZ(J, y, Z, Sa);
        /* the pair equalities */
        double delta = 0;
        double *rp = (double *)calloc((size_t)Q->np + 1, sizeof(double));
        for (int p = 0; p < Q->np; p++) {
            const int k = Q->K1[p], i = Q->I1[p];
            rp[p] = hpc_abs(Z[k] + (size_t)i * Wm) + (J->ncon[k] * 2 + 8.0) * u * Sa[k][i] * 1.01;
        }
        const double rn = hpc_norm((size_t)Q->np, rp);
        free(rp);
        out->resid = rn;
        if (rn > 0) {
            if (!(lgP > 0)) { snprintf(out->why, sizeof(out->why), "the equalities of (D) (split free variables) are dependent, or their Gram matrix is not bounded away from 0 in double"); ok = 0; delta = INFINITY; }
            else delta = rn / sqrt(lgP) * 1.01 + 1e-290;
        }
        out->delta = delta;
        for (int k = 0; k < nb && ok; k++) {
            const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
            const double ku = (J->ncon[k] * 2 + 8.0) * u * 1.01;
            if (lp) {
                /* N per entry: (sum_i a_i^2)^(1/2) */
                double *Ne = (double *)calloc((size_t)n + 1, sizeof(double));
                if (delta > 0) for (int tt = 0; tt < J->ncon[k]; tt++) for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) { const double a = hpc_abs(J->Av + e * Wm); Ne[J->ei[e]] += a * a; }
                for (int i = 0; i < n && ok; i++) {
                    if (Q->mark[k] && Q->mark[k][i]) continue;           /* zero at the corrected y */
                    mpx_setd(t, (ku * Sa[k][i] + delta * (sqrt(Ne[i]) + 1e-140) * 1.01) * 1.01 + 1e-290);
                    if (mpx_cmp(Z[k] + (size_t)i * Wm, t) < 0) { snprintf(out->why, sizeof(out->why), "an entry of the LP block %d of Z = C - A*(y) is negative, or below its error bound", k + 1); ok = 0; }
                }
                free(Ne);
                continue;
            }
            double N2 = 0;
            const double rho = hpc_norm((size_t)n * n, Sa[k]);
            if (delta > 0) for (int tt = 0; tt < J->ncon[k]; tt++) for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) { const double a = hpc_abs(J->Av + e * Wm); N2 += (J->ei[e] != J->ej[e] ? 2.0 : 1.0) * a * a; }
            const double need = ku * rho * 1.01 + delta * (sqrt(N2) + 1e-140) * 1.01;          /* (1e-140: squares that underflowed) */
            if (!hpc_lammin_ge(J, n, Z[k], need, W)) { snprintf(out->why, sizeof(out->why), "block %d of Z = C - A*(y) is not positive definite beyond its error bound %.1e", k + 1, need); ok = 0; }
        }
        for (int k = 0; k < nb; k++) { free(Z[k]); free(Sa[k]); }
        free(Z); free(Sa);
        uint64_t by[MPX_MAXL + 2];
        mpx_setd(by, 0.0);
        double Sby = 0, b2 = 0;
        for (int c = 0; c < m; c++) {
            const uint64_t *bc = J->b + (size_t)c * Wm;
            mpx_mul(t, bc, y + (size_t)c * Wm); mpx_add(by, by, t);
            const double ba = hpc_abs(bc);
            Sby += ba * hpc_abs(y + (size_t)c * Wm); b2 += ba * ba;
        }
        mpx_copy(out->value, by);
        const double slack = (2.0 * m + 8.0) * u * Sby * 1.01 + (sqrt(b2) + 1e-140) * 1.01 * (isfinite(delta) ? delta : 0.0) * 1.01;
        mpx_setd(s, slack + 1e-290); mpx_sub(out->bound, by, s);
    }
    free(W);
    out->ok = ok;
    return ok;
}

/* The problem J as doubles (a PSOrig, for the two bounds that come from double arithmetic and
 * the Gram solve of the projection): each value is the double nearest to the mpx datum up to
 * rounding of the conversion, within one ulp of the decimal, as boundcert.c assumes of data
 * read from a file. */
static double hpc_round_d(const uint64_t *a) {
    uint64_t t[MPX_MAXL + 2], r[MPX_MAXL + 2];
    const double d = mpx_getd(a);
    mpx_setd(t, d); mpx_sub(r, a, t);
    return d + mpx_getd(r);
}
static PSOrig *hp_psorig(const HPJob *J) {
    const size_t Wm = (size_t)mpx_W;
    PSOrig *O = (PSOrig *)calloc(1, sizeof(PSOrig));
    O->m = J->m; O->nblk = J->nblk;
    O->bs = (int *)malloc(sizeof(int) * (size_t)(J->nblk + 1)); memcpy(O->bs, J->bs, sizeof(int) * (size_t)J->nblk);
    O->b = (double *)malloc(sizeof(double) * (size_t)(J->m + 1));
    double nb1 = 0, nC1 = 0;
    for (int c = 0; c < J->m; c++) { O->b[c] = hpc_round_d(J->b + (size_t)c * Wm); nb1 += fabs(O->b[c]); }
    const size_t nnz = J->nnzA + J->nnzC;
    O->nnz = nnz;
    O->con = (int *)malloc(sizeof(int) * (nnz + 1)); O->blk = (int *)malloc(sizeof(int) * (nnz + 1));
    O->ii = (int *)malloc(sizeof(int) * (nnz + 1)); O->jj = (int *)malloc(sizeof(int) * (nnz + 1));
    O->v = (double *)malloc(sizeof(double) * (nnz + 1));
    size_t w = 0;
    for (int k = 0; k < J->nblk; k++) {
        const int lp = J->bs[k] < 0;
        for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) {
            O->con[w] = -1; O->blk[w] = k; O->ii[w] = J->ci[e]; O->jj[w] = lp ? J->ci[e] : J->cj[e]; O->v[w] = hpc_round_d(J->Cv + e * Wm);
            nC1 += fabs(O->v[w]) * (O->ii[w] == O->jj[w] ? 1.0 : 2.0); w++;
        }
        for (int tt = 0; tt < J->ncon[k]; tt++)
            for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                O->con[w] = J->con[k][tt]; O->blk[w] = k; O->ii[w] = J->ei[e]; O->jj[w] = lp ? J->ei[e] : J->ej[e]; O->v[w] = hpc_round_d(J->Av + e * Wm); w++;
            }
    }
    O->nb1 = nb1; O->nC1 = nC1;
    return O;
}

/* ---- construction of the certificate ---- */

/* p: X onto A(X) = b. The correction A*(w), G w = b - A(X), is solved with the double factor
 * of G = A A* (bound.c) and refined: the residual is formed in mpx. Returns the relative
 * residual reached. */
static double hp_project_p(const HPJob *J, const PSOrig *O, uint64_t **X) {
    const size_t Wm = (size_t)mpx_W;
    const int nb = J->nblk, m = J->m;
    uint64_t t[MPX_MAXL + 2], wv[MPX_MAXL + 2];
    uint64_t *ax = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)m + 1) * Wm);
    double *r = (double *)malloc(sizeof(double) * ((size_t)m + 1)), *w = (double *)malloc(sizeof(double) * ((size_t)m + 1));
    double prev = 1e300, rel = 1e300, bn = 0;
    for (int c = 0; c < m; c++) { const double a = mpx_getd(J->b + (size_t)c * Wm); bn += a * a; }
    bn = sqrt(bn);
    for (int pass = 0; pass < 60; pass++) {
        memset(ax, 0, sizeof(uint64_t) * (size_t)m * Wm);
        for (int k = 0; k < nb; k++) {
            const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
            for (int tt = 0; tt < J->ncon[k]; tt++) {
                uint64_t *a = ax + (size_t)J->con[k][tt] * Wm;
                for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                    const int i = J->ei[e], j = lp ? i : J->ej[e];
                    mpx_mul(t, J->Av + e * Wm, lp ? X[k] + (size_t)i * Wm : X[k] + ((size_t)i * n + j) * Wm);
                    if (!lp && i != j) mpx_add(t, t, t);
                    mpx_add(a, a, t);
                }
            }
        }
        double rn = 0;
        for (int c = 0; c < m; c++) { mpx_sub(t, J->b + (size_t)c * Wm, ax + (size_t)c * Wm); r[c] = mpx_getd(t); rn += r[c] * r[c]; }
        rel = sqrt(rn) / (1 + bn);
        if (!(rel < 0.5 * prev) || rel == 0) break;
        prev = rel;
        bound_gram_apply(O, r, w);
        for (int k = 0; k < nb; k++) {
            const int n = abs(J->bs[k]), lp = J->bs[k] < 0;
            for (int tt = 0; tt < J->ncon[k]; tt++) {
                const double wc = w[J->con[k][tt]];
                if (wc == 0 || !isfinite(wc)) continue;
                mpx_setd(wv, wc);
                for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) {
                    const int i = J->ei[e], j = lp ? i : J->ej[e];
                    if (lp) { mpx_addmul(X[k] + (size_t)i * Wm, wv, J->Av + e * Wm); continue; }
                    mpx_addmul(X[k] + ((size_t)i * n + j) * Wm, wv, J->Av + e * Wm);
                    if (i != j) mpx_copy(X[k] + ((size_t)j * n + i) * Wm, X[k] + ((size_t)i * n + j) * Wm);
                }
            }
        }
    }
    free(ax); free(r); free(w);
    return fmin(rel, prev);
}

/* d: y onto the pair equalities z(K1, I1) = 0 (normal equations in double, refined in mpx) */
static void hp_project_pairs(const HPJob *J, const HPPairs *Q, uint64_t *y) {
    const size_t Wm = (size_t)mpx_W;
    const int np = Q->np, m = J->m;
    if (np <= 0 || np > 4000) return;
    uint64_t t[MPX_MAXL + 2], z[MPX_MAXL + 2], wv[MPX_MAXL + 2];
    /* rows of the equalities: entries (constraint, value) of the column (K1, I1) */
    double *G = (double *)calloc((size_t)np * np + 1, sizeof(double));
    int *last = (int *)malloc(sizeof(int) * ((size_t)m + 1));
    {
        /* G = Ap Ap': by constraint, the pairs that have it */
        int *cnt = (int *)calloc((size_t)m + 2, sizeof(int));
        for (int p = 0; p < np; p++) { const int k = Q->K1[p], i = Q->I1[p]; for (int q = J->lpptr[k][i]; q < J->lpptr[k][i + 1]; q++) cnt[J->lpc[k][q] + 1]++; }
        for (int c = 0; c < m; c++) cnt[c + 1] += cnt[c];
        const int tot = cnt[m];
        int *pp = (int *)malloc(sizeof(int) * ((size_t)tot + 1)); double *pv = (double *)malloc(sizeof(double) * ((size_t)tot + 1));
        memcpy(last, cnt, sizeof(int) * (size_t)m);
        for (int p = 0; p < np; p++) { const int k = Q->K1[p], i = Q->I1[p]; for (int q = J->lpptr[k][i]; q < J->lpptr[k][i + 1]; q++) { const int c = J->lpc[k][q]; pp[last[c]] = p; pv[last[c]] = mpx_getd(J->Av + (size_t)J->lpe[k][q] * Wm); last[c]++; } }
        for (int c = 0; c < m; c++) for (int a = cnt[c]; a < cnt[c + 1]; a++) for (int b = cnt[c]; b < cnt[c + 1]; b++) G[pp[a] + (size_t)pp[b] * np] += pv[a] * pv[b];
        free(cnt); free(pp); free(pv);
    }
    free(last);
    int info = 0;
    BL(dpotrf_)("L", &np, G, &np, &info);
    if (info != 0) { free(G); return; }
    double *r = (double *)malloc(sizeof(double) * ((size_t)np + 1));
    double prev = 1e300;
    for (int pass = 0; pass < 40; pass++) {
        double rn = 0;
        for (int p = 0; p < np; p++) {
            const int k = Q->K1[p], i = Q->I1[p];
            mpx_setd(z, 0.0);
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) if (J->ci[e] == i) mpx_copy(z, J->Cv + e * Wm);
            for (int q = J->lpptr[k][i]; q < J->lpptr[k][i + 1]; q++) { mpx_mul(t, J->Av + (size_t)J->lpe[k][q] * Wm, y + (size_t)J->lpc[k][q] * Wm); mpx_sub(z, z, t); }
            r[p] = mpx_getd(z); rn += r[p] * r[p];
        }
        rn = sqrt(rn);
        if (!(rn < 0.5 * prev) || rn == 0) break;
        prev = rn;
        const int one = 1;
        BL(dpotrs_)("L", &np, &one, G, &np, r, &np, &info);
        for (int p = 0; p < np; p++) {
            if (r[p] == 0 || !isfinite(r[p])) continue;
            const int k = Q->K1[p], i = Q->I1[p];
            mpx_setd(wv, r[p]);
            for (int q = J->lpptr[k][i]; q < J->lpptr[k][i + 1]; q++) mpx_addmul(y + (size_t)J->lpc[k][q] * Wm, wv, J->Av + (size_t)J->lpe[k][q] * Wm);
        }
    }
    free(r); free(G);
}

typedef struct { uint64_t **X; uint64_t *y; double err; } HPPoint;

static uint64_t **hpc_alloc_blocks(const HPJob *J) {
    const size_t Wm = (size_t)mpx_W;
    uint64_t **X = (uint64_t **)calloc((size_t)J->nblk + 1, sizeof(uint64_t *));
    for (int k = 0; k < J->nblk; k++) X[k] = (uint64_t *)calloc(((J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]) + 1) * Wm, sizeof(uint64_t));
    return X;
}
static void hpc_free_blocks(uint64_t **X, int nb) { if (!X) return; for (int k = 0; k < nb; k++) free(X[k]); free(X); }
/* D = A + s (B - A) + t Dir over the certificate's side */
static void hpc_combine(const HPJob *J, int side, uint64_t **DX, uint64_t *dy, uint64_t *const *AX, const uint64_t *ay, uint64_t *const *BX, const uint64_t *by2,
                        const uint64_t *s, uint64_t *const *DirX, const uint64_t *diry, double tdir) {
    const size_t Wm = (size_t)mpx_W;
    uint64_t t[MPX_MAXL + 2], td[MPX_MAXL + 2];
    mpx_setd(td, tdir);
    if (side == 1) {
        for (int k = 0; k < J->nblk; k++) {
            const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k];
            for (size_t i = 0; i < len; i++) {
                uint64_t *d = DX[k] + i * Wm;
                mpx_sub(t, BX[k] + i * Wm, AX[k] + i * Wm); mpx_mul(t, t, s); mpx_add(d, AX[k] + i * Wm, t);
                if (DirX && tdir != 0) mpx_addmul(d, td, DirX[k] + i * Wm);
            }
        }
    } else {
        for (int c = 0; c < J->m; c++) {
            uint64_t *d = dy + (size_t)c * Wm;
            mpx_sub(t, by2 + (size_t)c * Wm, ay + (size_t)c * Wm); mpx_mul(t, t, s); mpx_add(d, ay + (size_t)c * Wm, t);
            if (diry && tdir != 0) mpx_addmul(d, td, diry + (size_t)c * Wm);
        }
    }
}

/* The certificate of one side on the problem as read J (final point in J->X, J->y), from the
 * returned point, the anchors anc[0..na-1] (older first; earlier, more central iterates of the
 * solve mapped to the problem as read) and, after a face reduction, the direction that the
 * multiplier of the reduction's certificate adds (DirX on side p, diry on side d). On success
 * the certificate replaces J->X (p) or J->y (d) and 1 is returned. */
static int hp_bound(HPJob *J, const PSOrig *O, int exact, int side, int na, HPPoint *anc, uint64_t **DirX, const uint64_t *diry,
                    int verbose, HPCertOut *out, char *src, size_t srclen) {
    const size_t Wm = (size_t)mpx_W;
    const int nb = J->nblk, m = J->m;
    HPPairs Q;
    hp_find_pairs(J, &Q);
    double lgG = NAN, lgP = NAN;
    if (side == 1) lgG = brisk_gram_lammin_lower(O, exact);
    else if (Q.np > 0) lgP = brisk_pairs_lammin_lower(O, exact, Q.np, Q.K1, Q.I1);
    if (verbose > 1) {
        if (side == 1) printf("     certificate (P): lambda_min(A A*) >= %.2e (double, rigorous); %d split pair(s)\n", lgG, Q.np);
        else printf("     certificate (D): %d equalities (split pairs)%s\n", Q.np, Q.np ? (lgP > 0 ? ", independent" : ", not shown independent") : "");
    }
    /* the candidates, the returned point first, then the anchors from the most recent */
    const int nc = na + 2, isyn = na + 1;
    HPPoint *C = (HPPoint *)calloc((size_t)nc, sizeof(HPPoint));
    /* (a copy of the returned point: the preparation below changes the candidates) */
    C[0].err = J->best_err;
    if (side == 1) { C[0].X = hpc_alloc_blocks(J); for (int k = 0; k < nb; k++) { const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]; memcpy(C[0].X[k], J->X[k], len * Wm * sizeof(uint64_t)); } }
    else { C[0].y = (uint64_t *)malloc(((size_t)m + 1) * Wm * sizeof(uint64_t)); memcpy(C[0].y, J->y, (size_t)m * Wm * sizeof(uint64_t)); }
    for (int a = 0; a < na; a++) C[1 + a] = anc[na - 1 - a];
    /* The last candidate: the returned point moved along the identity. p: X + s I, which the
     * projection below turns into X + s (I - A*(G^{-1} A(I))), the part of the identity in the
     * null space of A. d: y - s e with e = G^{-1} A(I) (double), so that Z gains s A*(e), the
     * part of the identity in the range of A* (the identity itself when a constraint is the
     * trace: max-cut, theta). It serves the points that are not interior (the first-order
     * methods' projections) and the solves without anchors. */
    C[isyn].err = NAN;
    if (side == 1) {
        double sx = 1.0;
        for (int k = 0; k < nb; k++) { const int n = abs(J->bs[k]); for (int i = 0; i < n; i++) { const double a = fabs(mpx_getd(J->bs[k] < 0 ? J->X[k] + (size_t)i * Wm : J->X[k] + ((size_t)i * n + i) * Wm)); if (a > sx) sx = a; } }
        uint64_t sv[MPX_MAXL + 2];
        mpx_setd(sv, sx);
        C[isyn].X = hpc_alloc_blocks(J);
        for (int k = 0; k < nb; k++) {
            const int n = abs(J->bs[k]); const size_t len = J->bs[k] < 0 ? (size_t)n : (size_t)n * n;
            memcpy(C[isyn].X[k], J->X[k], len * Wm * sizeof(uint64_t));
            for (int i = 0; i < n; i++) { uint64_t *d = J->bs[k] < 0 ? C[isyn].X[k] + (size_t)i * Wm : C[isyn].X[k] + ((size_t)i * n + i) * Wm; mpx_add(d, d, sv); }
        }
    } else {
        double *ai = (double *)calloc((size_t)m + 1, sizeof(double)), *ev = (double *)calloc((size_t)m + 1, sizeof(double));
        double sc = 1.0;
        for (int k = 0; k < nb; k++) {
            const int lp = J->bs[k] < 0;
            for (size_t e = J->cptr[k]; e < J->cptr[k + 1]; e++) { const double a = fabs(mpx_getd(J->Cv + e * Wm)); if (a > sc) sc = a; }
            for (int tt = 0; tt < J->ncon[k]; tt++) for (size_t e = J->ptr[k][tt]; e < J->ptr[k][tt + 1]; e++) if (lp || J->ei[e] == J->ej[e]) ai[J->con[k][tt]] += mpx_getd(J->Av + e * Wm);
        }
        bound_gram_apply(O, ai, ev);
        C[isyn].y = (uint64_t *)malloc(((size_t)m + 1) * Wm * sizeof(uint64_t));
        memcpy(C[isyn].y, J->y, (size_t)m * Wm * sizeof(uint64_t));
        uint64_t tv[MPX_MAXL + 2];
        for (int c = 0; c < m; c++) { if (!isfinite(ev[c]) || ev[c] == 0) continue; mpx_setd(tv, sc * ev[c]); mpx_sub(C[isyn].y + (size_t)c * Wm, C[isyn].y + (size_t)c * Wm, tv); }
        free(ai); free(ev);
    }
    uint64_t one[MPX_MAXL + 2], zero[MPX_MAXL + 2];
    mpx_setd(one, 1.0); mpx_setd(zero, 0.0);
    uint64_t **TX = side == 1 ? hpc_alloc_blocks(J) : NULL;
    uint64_t *ty = side == 2 ? (uint64_t *)calloc(((size_t)m + 1) * Wm, sizeof(uint64_t)) : NULL;
    int found = 0;
    src[0] = 0;
    /* After a face reduction the certificate of the reduction (W >= 0, in the range of A* with
     * b'y = 0 on side d; a feasible direction of X on side p) is added with a multiplier t.
     * A larger t is never worse (Z + t W grows with t), so the largest that leaves the
     * arithmetic 30 digits of room below the eigenvalues to be seen is taken: u^(-0.45). */
    const int have_dir = side == 1 ? DirX != NULL : diry != NULL;
    double dscale = 1.0;
    if (have_dir && side == 2) { for (int c = 0; c < m; c++) { const double a = fabs(mpx_getd(J->y + (size_t)c * Wm)); if (a > dscale) dscale = a; } }
    const double tdir = have_dir ? dscale * pow(HPC_U(J), -0.45) : 0.0;
    int pass_a = -1;
    for (int a = 0; a < nc; a++) {
        if (side == 1) {
            /* a split pair's two entries are raised together (their columns cancel) */
            for (int p = 0; p < Q.np; p++) {
                uint64_t *x1 = C[a].X[Q.K1[p]] + (size_t)Q.I1[p] * Wm, *x2 = C[a].X[Q.K2[p]] + (size_t)Q.I2[p] * Wm, sh[MPX_MAXL + 2];
                mpx_setd(sh, 1.0 + fabs(mpx_getd(x1)) + fabs(mpx_getd(x2)));
                mpx_add(x1, x1, sh); mpx_add(x2, x2, sh);
            }
            const double rel = hp_project_p(J, O, C[a].X);
            if (verbose > 1) printf("     certificate (P): candidate %d (error %.1e) projected onto A(X) = b: residual %.1e\n", a, C[a].err, rel);
        } else hp_project_pairs(J, &Q, C[a].y);
        hpc_combine(J, side, TX, ty, C[a].X, C[a].y, C[a].X, C[a].y, zero, DirX, diry, tdir);
        if (hp_cert_check(J, &Q, lgG, lgP, side, TX, ty, out)) {
            pass_a = a;
            if (a == 0) { found = 1; snprintf(src, srclen, "the returned point"); }
            else if (a == isyn) snprintf(src, srclen, "the returned point moved along the identity");
            else snprintf(src, srclen, "an earlier iterate (error %.0e)", C[a].err);
            break;
        }
        if (verbose > 1) printf("     certificate (%s): candidate %d (error %.1e): %s\n", side == 1 ? "P" : "D", a, C[a].err, out->why);
        if (!isfinite(out->delta)) break;          /* no bound from double: the other candidates fail the same way */
    }
    if (!found && pass_a > 0) {
        /* Between the anchor (passes) and the returned point (fails): the smallest weight of
         * the anchor on a geometric grid, refined by bisection of its logarithm. The loss of
         * the bound is that weight times the anchor's gap. */
        uint64_t sm[MPX_MAXL + 2], w[MPX_MAXL + 2];
        HPCertOut o2;
        double lpass = 0.0, lfail = -44.0;         /* log10 of the weight: 1 passes */
        for (int kk = 40; kk >= 4; kk -= 4) {
            mpx_setd(w, pow(10.0, -kk)); mpx_sub(sm, one, w);
            hpc_combine(J, side, TX, ty, C[pass_a].X, C[pass_a].y, C[0].X, C[0].y, sm, DirX, diry, tdir);
            if (hp_cert_check(J, &Q, lgG, lgP, side, TX, ty, &o2)) { lpass = -kk; break; }
            lfail = -kk;
        }
        for (int bis = 0; bis < 5; bis++) {
            const double lm = 0.5 * (lpass + lfail);
            mpx_setd(w, pow(10.0, lm)); mpx_sub(sm, one, w);
            hpc_combine(J, side, TX, ty, C[pass_a].X, C[pass_a].y, C[0].X, C[0].y, sm, DirX, diry, tdir);
            if (hp_cert_check(J, &Q, lgG, lgP, side, TX, ty, &o2)) lpass = lm; else lfail = lm;
        }
        if (lpass < 0) {
            mpx_setd(w, pow(10.0, lpass)); mpx_sub(sm, one, w);
            hpc_combine(J, side, TX, ty, C[pass_a].X, C[pass_a].y, C[0].X, C[0].y, sm, DirX, diry, tdir);
            if (hp_cert_check(J, &Q, lgG, lgP, side, TX, ty, &o2)) {
                *out = o2; found = 1;
                if (pass_a == isyn) snprintf(src, srclen, "the returned point moved along the identity by the weight %.1e", pow(10.0, lpass));
                else snprintf(src, srclen, "the returned point and an earlier iterate (error %.0e) with the weight %.1e", C[pass_a].err, pow(10.0, lpass));
            }
        }
        if (!found) {
            hpc_combine(J, side, TX, ty, C[pass_a].X, C[pass_a].y, C[pass_a].X, C[pass_a].y, zero, DirX, diry, tdir);
            found = hp_cert_check(J, &Q, lgG, lgP, side, TX, ty, out);
        }
    }
    if (found && tdir != 0) { const size_t l = strlen(src); snprintf(src + l, srclen - l, "; multiplier %.0e on the face reduction's certificate", tdir); }
    if (found) {
        if (side == 1) for (int k = 0; k < nb; k++) { const size_t len = J->bs[k] < 0 ? (size_t)(-J->bs[k]) : (size_t)J->bs[k] * J->bs[k]; memcpy(J->X[k], TX[k], len * Wm * sizeof(uint64_t)); }
        else memcpy(J->y, ty, (size_t)m * Wm * sizeof(uint64_t));
    }
    if (side == 1) { hpc_free_blocks(C[0].X, nb); hpc_free_blocks(C[isyn].X, nb); } else { free(C[0].y); free(C[isyn].y); }
    hpc_free_blocks(TX, nb); free(ty); free(C);
    hp_pairs_free(&Q);
    return found;
}

/* the bound as text, rounded outward (dir -1: down, +1: up) at nd digits */
static void hpc_bound_str(char *buf, const uint64_t *b, int nd, int dir) {
    uint64_t t[MPX_MAXL + 2], s[MPX_MAXL + 2];
    const double a = fabs(mpx_getd(b));
    mpx_setd(s, (a > 0 ? a : 1e-290) * 2.0 * pow(10.0, 1.0 - nd));
    if (dir < 0) mpx_sub(t, b, s); else mpx_add(t, b, s);
    mpx_to_str(buf, t, nd);
}
static double hpc_bound_dbl(const uint64_t *b, int dir) {
    double d = mpx_getd(b);
    d = nextafter(nextafter(d, dir < 0 ? -INFINITY : INFINITY), dir < 0 ? -INFINITY : INFINITY);
    return d;
}

/* A point of the reduced problem JR (Xr, yr) on the problem as read J0: through the presolve
 * (pre; tface: the multiplier of the face reductions' certificates) to the base problem JB,
 * and from the dual form (X = the slack of the dual form, y = y+ - y-). X0 (blocks of J0)
 * and y0 are filled. */
static void hp_map_back(const HPPre *pre, const HPJob *JB, const HPJob *JR, int dualized, const HPJob *J0,
                        uint64_t *const *Xr, const uint64_t *yr, double tface, uint64_t **X0, uint64_t *y0) {
    const size_t Wm = (size_t)mpx_W;
    uint64_t **Xb = NULL, *yb = NULL;
    uint64_t *const *XB = Xr; const uint64_t *yB = yr;
    if (pre) {
        Xb = hpc_alloc_blocks(JB); yb = (uint64_t *)calloc(((size_t)JB->m + 1) * Wm, sizeof(uint64_t));
        hp_postsolve(pre, JB, JR, Xr, yr, Xb, yb, tface);
        XB = Xb; yB = yb;
    }
    if (dualized) {
        uint64_t **Zb = hpc_alloc_blocks(JB);
        hp_formZ(JB, yB, Zb, NULL);
        for (int k = 0; k < J0->nblk; k++) { const size_t len = J0->bs[k] < 0 ? (size_t)(-J0->bs[k]) : (size_t)J0->bs[k] * J0->bs[k]; memcpy(X0[k], Zb[k], len * Wm * sizeof(uint64_t)); }
        for (int p = 0; p < J0->m; p++) mpx_sub(y0 + (size_t)p * Wm, XB[J0->nblk] + (size_t)(2 * p) * Wm, XB[J0->nblk] + (size_t)(2 * p + 1) * Wm);
        hpc_free_blocks(Zb, JB->nblk);
    } else {
        for (int k = 0; k < J0->nblk; k++) { const size_t len = J0->bs[k] < 0 ? (size_t)(-J0->bs[k]) : (size_t)J0->bs[k] * J0->bs[k]; memcpy(X0[k], XB[k], len * Wm * sizeof(uint64_t)); }
        memcpy(y0, yB, (size_t)J0->m * Wm * sizeof(uint64_t));
    }
    hpc_free_blocks(Xb, pre ? JB->nblk : 0); free(yb);
}
