/*
 * High-precision endgame.
 *
 * On degenerate or ill-posed problems the Schur complement becomes so ill conditioned
 * that the search direction can no longer reduce the primal residual: on SDPLIB's hinf
 * family the double-precision iteration stalls with |A(X) - b| ~ 1e-7 while mu keeps
 * falling, so the two objectives converge to different values (a relative gap of up to
 * 3e-4). No amount of refinement in double precision helps, because the assembled Schur
 * matrix itself carries an error of eps * cond(M).
 *
 * This module therefore redoes the whole linear algebra in double-double arithmetic
 * (two doubles per number, about 32 digits): Z^{-1}, the Schur complement, its Cholesky
 * factorization, the direction and the step lengths. It runs a few HKM predictor-corrector
 * iterations starting from the double solver's best iterate, and only when
 *   - that iterate missed the tolerance, and
 *   - the problem is small enough (m <= dd_maxm and sum n^2 <= dd_maxn2),
 * so the cost (about 20x the flop cost of double arithmetic, on problems that take
 * milliseconds) is irrelevant. The caller accepts the result only if the measured
 * accuracy improves.
 *
 * Everything here works on the scaled problem, exactly as the main iteration does, and
 * uses plain dense per-block storage with generic sparse loops over the constraints: no
 * routes, no low-rank, no envelope. Small problems only.
 */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double dd_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ---------------------------------------------------------------- dd arithmetic */
typedef struct { double h, l; } dd;

static inline dd dd_qts(double a, double b) {          /* quick two-sum, |a| >= |b| */
    dd r;
    r.h = a + b;
    r.l = b - (r.h - a);
    return r;
}
static inline dd dd_ts(double a, double b) {           /* two-sum */
    double s = a + b, bb = s - a;
    dd r;
    r.h = s;
    r.l = (a - (s - bb)) + (b - bb);
    return r;
}
static inline dd dd_tp(double a, double b) {           /* two-product (fma) */
    dd r;
    r.h = a * b;
    r.l = fma(a, b, -r.h);
    return r;
}
static inline dd dd_of(double a) { dd r = { a, 0.0 }; return r; }
static inline double dd_d(dd a) { return a.h + a.l; }
static inline dd dd_add(dd a, dd b) {
    dd s = dd_ts(a.h, b.h), t = dd_ts(a.l, b.l);
    s.l += t.h;
    s = dd_qts(s.h, s.l);
    s.l += t.l;
    return dd_qts(s.h, s.l);
}
static inline dd dd_neg(dd a) { dd r = { -a.h, -a.l }; return r; }
static inline dd dd_sub(dd a, dd b) { return dd_add(a, dd_neg(b)); }
static inline dd dd_mul(dd a, dd b) {
    dd p = dd_tp(a.h, b.h);
    p.l += a.h * b.l + a.l * b.h;
    return dd_qts(p.h, p.l);
}
static inline dd dd_muld(dd a, double b) {
    dd p = dd_tp(a.h, b);
    p.l += a.l * b;
    return dd_qts(p.h, p.l);
}
static inline dd dd_div(dd a, dd b) {
    double q1 = a.h / b.h;
    dd r = dd_sub(a, dd_muld(b, q1));
    double q2 = dd_d(r) / b.h;
    return dd_qts(q1, q2);
}
static inline dd dd_sqrt(dd a) {
    if (a.h <= 0) return dd_of(0.0);
    double x = 1.0 / sqrt(a.h);
    double ax = a.h * x;
    dd t = dd_sub(a, dd_tp(ax, ax));
    return dd_qts(ax, dd_d(t) * x * 0.5);
}
static inline int dd_gt(dd a, dd b) { dd d = dd_sub(a, b); return d.h > 0 || (d.h == 0 && d.l > 0); }

/* ---------------------------------------------------------------- block workspace */
typedef struct {
    int type, n;
    size_t len;                 /* n*n (SDP) or n (LP) */
    dd *X, *Z, *Zi, *dX, *dZ, *L, *T1, *T2, *T3, *Rd, *Q;
} DBlk;

static dd *dalloc(size_t n) {
    dd *p = calloc(n ? n : 1, sizeof(dd));
    if (!p) { fprintf(stderr, "brisk: out of memory (dd endgame)\n"); exit(1); }
    return p;
}

/* C = A B (n x n, dd) */
static void dd_gemm(int n, const dd *A, const dd *B, dd *C) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            dd s = dd_of(0.0);
            for (int k = 0; k < n; k++) s = dd_add(s, dd_mul(A[i + (size_t)k * n], B[k + (size_t)j * n]));
            C[i + (size_t)j * n] = s;
        }
}
static void dd_sym(int n, dd *A) {
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++) {
            dd v = dd_muld(dd_add(A[i + (size_t)j * n], A[j + (size_t)i * n]), 0.5);
            A[i + (size_t)j * n] = A[j + (size_t)i * n] = v;
        }
}
/* lower Cholesky of the lower triangle of A into L; returns 1 on failure */
static int dd_chol(int n, const dd *A, dd *L) {
    memset(L, 0, sizeof(dd) * (size_t)n * n);
    for (int j = 0; j < n; j++) {
        dd s = A[j + (size_t)j * n];
        for (int k = 0; k < j; k++) { dd v = L[j + (size_t)k * n]; s = dd_sub(s, dd_mul(v, v)); }
        if (!(s.h > 0)) return 1;
        dd d = dd_sqrt(s);
        L[j + (size_t)j * n] = d;
        for (int i = j + 1; i < n; i++) {
            dd t = A[i + (size_t)j * n];
            for (int k = 0; k < j; k++) t = dd_sub(t, dd_mul(L[i + (size_t)k * n], L[j + (size_t)k * n]));
            L[i + (size_t)j * n] = dd_div(t, d);
        }
    }
    return 0;
}
/* inverse of a symmetric positive definite matrix from its Cholesky factor */
static void dd_inv_from_chol(int n, const dd *L, dd *Inv, dd *col) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = dd_of(i == j ? 1.0 : 0.0);
        for (int i = 0; i < n; i++) {                       /* forward */
            dd s = col[i];
            for (int k = 0; k < i; k++) s = dd_sub(s, dd_mul(L[i + (size_t)k * n], col[k]));
            col[i] = dd_div(s, L[i + (size_t)i * n]);
        }
        for (int i = n - 1; i >= 0; i--) {                  /* backward */
            dd s = col[i];
            for (int k = i + 1; k < n; k++) s = dd_sub(s, dd_mul(L[k + (size_t)i * n], col[k]));
            col[i] = dd_div(s, L[i + (size_t)i * n]);
        }
        for (int i = 0; i < n; i++) Inv[i + (size_t)j * n] = col[i];
    }
}
static void dd_solve_chol(int m, const dd *L, dd *x) {
    for (int i = 0; i < m; i++) {
        dd s = x[i];
        for (int k = 0; k < i; k++) s = dd_sub(s, dd_mul(L[i + (size_t)k * m], x[k]));
        x[i] = dd_div(s, L[i + (size_t)i * m]);
    }
    for (int i = m - 1; i >= 0; i--) {
        dd s = x[i];
        for (int k = i + 1; k < m; k++) s = dd_sub(s, dd_mul(L[k + (size_t)i * m], x[k]));
        x[i] = dd_div(s, L[i + (size_t)i * m]);
    }
}

/* ---------------------------------------------------------------- operators */
/* out_i += w * <A_i, D> over the constraints of this block */
static void dd_Aop(const Block *B, const DBlk *b, const dd *D, dd *out, double w) {
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *S = &B->A[t];
        dd s = dd_of(0.0);
        if (B->type == BLK_LP)
            for (int q = 0; q < S->nnz; q++) s = dd_add(s, dd_muld(D[S->row[q]], S->val[q]));
        else
            for (int q = 0; q < S->ef; q++)
                s = dd_add(s, dd_muld(D[S->fr[q] + (size_t)S->fc[q] * b->n], S->fv[q]));
        out[B->con[t]] = dd_add(out[B->con[t]], dd_muld(s, w));
    }
}
/* F = sum_i y_i A_i  (block part) */
static void dd_ATy(const Block *B, const DBlk *b, const dd *y, dd *F) {
    memset(F, 0, sizeof(dd) * b->len);
    for (int t = 0; t < B->ncon; t++) {
        dd yt = y[B->con[t]];
        if (yt.h == 0 && yt.l == 0) continue;
        const SpSym *S = &B->A[t];
        if (B->type == BLK_LP)
            for (int q = 0; q < S->nnz; q++) {
                size_t i = S->row[q];
                F[i] = dd_add(F[i], dd_muld(yt, S->val[q]));
            }
        else
            for (int q = 0; q < S->ef; q++) {
                size_t i = S->fr[q] + (size_t)S->fc[q] * b->n;
                F[i] = dd_add(F[i], dd_muld(yt, S->fv[q]));
            }
    }
}
/* F += w * (sparse matrix S) */
static void dd_add_sp(const Block *B, const DBlk *b, const SpSym *S, double w, dd *F) {
    if (B->type == BLK_LP)
        for (int q = 0; q < S->nnz; q++) F[S->row[q]] = dd_add(F[S->row[q]], dd_of(w * S->val[q]));
    else
        for (int q = 0; q < S->ef; q++) {
            size_t i = S->fr[q] + (size_t)S->fc[q] * b->n;
            F[i] = dd_add(F[i], dd_of(w * S->fv[q]));
        }
}
/* M += lower triangle of the block's Schur contribution M_tu = tr(A_t X A_u Zi).
 * SDP blocks: H_t = X A_t Zi on the union pattern of the block's constraints
 * (Y = A_t Zi on the rows of A_t, then H = X[:, rows] Y), and M_tu = <A_u, H_t>.
 * Cost per constraint: n ef_t + |pattern| r_t + sum_u ef_u dd operations, instead of
 * sum_u ef_t ef_u for the pairwise loop this replaces (90x less on dense 12x12 LMIs). */
static void dd_schur(const Block *B, const DBlk *b, int m, dd *M) {
    const int n = b->n;
    if (B->type == BLK_LP) {
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *St = &B->A[t];
            const int it = B->con[t];
            for (int u = t; u < B->ncon; u++) {
                const SpSym *Su = &B->A[u];
                const int iu = B->con[u];
                dd s = dd_of(0.0);
                int qa = 0, qb = 0;
                while (qa < St->nnz && qb < Su->nnz) {
                    int ra = St->row[qa], rb = Su->row[qb];
                    if (ra < rb) qa++;
                    else if (rb < ra) qb++;
                    else {
                        dd v = dd_mul(b->X[ra], b->Zi[ra]);
                        s = dd_add(s, dd_muld(v, St->val[qa] * Su->val[qb]));
                        qa++; qb++;
                    }
                }
                int r = it > iu ? it : iu, c = it > iu ? iu : it;
                M[r + (size_t)c * m] = dd_add(M[r + (size_t)c * m], s);
            }
        }
        return;
    }
    const int unf = B->unf;
    const int *ufr = B->ufr, *ufc = B->ufc;
    int *rpos = malloc(sizeof(int) * n);
    dd *Y = dalloc((size_t)n * n);          /* rows of A_t Zi, compact (r x n) */
    dd *H = dalloc((size_t)n * n);          /* H_t on the union pattern */
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *St = &B->A[t];
        const int it = B->con[t], r = St->nr;
        for (int k = 0; k < r; k++) rpos[St->rows[k]] = k;
        memset(Y, 0, sizeof(dd) * (size_t)r * n);
        for (int q = 0; q < St->ef; q++) {
            const int p = rpos[St->fr[q]], c = St->fc[q];
            const double v = St->fv[q];
            dd *yr = Y + (size_t)p;                     /* row p, stride r */
            const dd *zc = b->Zi + (size_t)c * n;        /* Zi[c, :] = Zi[:, c] */
            for (int j = 0; j < n; j++) yr[(size_t)j * r] = dd_add(yr[(size_t)j * r], dd_muld(zc[j], v));
        }
        for (int e = 0; e < unf; e++) {
            const int i = ufr[e], j = ufc[e];
            dd s = dd_of(0.0);
            for (int k = 0; k < r; k++)
                s = dd_add(s, dd_mul(b->X[i + (size_t)St->rows[k] * n], Y[k + (size_t)j * r]));
            H[i + (size_t)j * n] = s;
        }
        for (int u = t; u < B->ncon; u++) {
            const SpSym *Su = &B->A[u];
            const int iu = B->con[u];
            dd s = dd_of(0.0);
            for (int q = 0; q < Su->ef; q++)
                s = dd_add(s, dd_muld(H[Su->fr[q] + (size_t)Su->fc[q] * n], Su->fv[q]));
            int rr = it > iu ? it : iu, c = it > iu ? iu : it;
            M[rr + (size_t)c * m] = dd_add(M[rr + (size_t)c * m], s);
        }
    }
    free(rpos); free(Y); free(H);
}


/* Search directions from dy:
 *   dZ = Rd - A'(dy)
 *   dX = -sym(Zi dZ X) - X            (predictor, corr = 0)
 *   dX = smu Zi - Q - sym(Zi dZ X) - X (corrector, corr = 1; Q = sym(Zi dZa dXa))   */
static void dd_dir(const Block *B, DBlk *b, const dd *dy, dd smu, int corr) {
    const int n = b->n;
    dd_ATy(B, b, dy, b->T1);
    for (size_t i = 0; i < b->len; i++) b->dZ[i] = dd_sub(b->Rd[i], b->T1[i]);
    if (b->type == BLK_LP) {
        for (int i = 0; i < n; i++) {
            dd v = dd_mul(b->Zi[i], dd_mul(b->dZ[i], b->X[i]));
            dd x = dd_sub(dd_neg(v), b->X[i]);
            if (corr) x = dd_add(x, dd_sub(dd_mul(b->Zi[i], smu), b->Q[i]));
            b->dX[i] = x;
        }
        return;
    }
    dd_gemm(n, b->Zi, b->dZ, b->T2);
    dd_gemm(n, b->T2, b->X, b->T3);
    dd_sym(n, b->T3);
    for (size_t i = 0; i < b->len; i++) {
        dd x = dd_sub(dd_neg(b->T3[i]), b->X[i]);
        if (corr) x = dd_sub(dd_add(x, dd_mul(b->Zi[i], smu)), b->Q[i]);
        b->dX[i] = x;
    }
}

/* largest steps keeping X + a dX and Z + a dZ positive definite (bisection + Cholesky) */
static void dd_steps(const Problem *P, DBlk *b, int nb, dd *apo, dd *ado) {
    dd ap = dd_of(1.0), ad = dd_of(1.0);
    for (int k = 0; k < nb; k++) {
        const int n = b[k].n;
        for (int side = 0; side < 2; side++) {
            const dd *V = side ? b[k].Z : b[k].X, *D = side ? b[k].dZ : b[k].dX;
            dd lo = dd_of(0.0), hi = side ? ad : ap;
            for (int s = 0; s < 14; s++) {                /* 2^-14: the step is scaled by 0.9-0.99 anyway */
                dd mid = dd_muld(dd_add(lo, hi), 0.5);
                int ok = 1;
                if (b[k].type == BLK_LP) {
                    for (int i = 0; i < n && ok; i++)
                        if (!dd_gt(dd_add(V[i], dd_mul(mid, D[i])), dd_of(0.0))) ok = 0;
                } else {
                    for (size_t i = 0; i < b[k].len; i++) b[k].T1[i] = dd_add(V[i], dd_mul(mid, D[i]));
                    ok = !dd_chol(n, b[k].T1, b[k].T2);
                }
                if (ok) lo = mid; else hi = mid;
            }
            if (side) ad = lo; else ap = lo;
        }
    }
    (void)P;
    *apo = ap; *ado = ad;
}

/* ---------------------------------------------------------------- endgame */
int dd_endgame(const Problem *P, double **Xio, double **Zio, double *y, const Params *par,
               int maxit, double tol, int verbose) {
    const int m = P->m, nb = P->nblk;
    const double t_start = dd_now(), t_limit = par->dd_time;
    DBlk *b = calloc(nb, sizeof(DBlk));
    double ndim = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        b[k].type = B->type;
        b[k].n = B->n;
        b[k].len = B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
        size_t len = b[k].len;
        dd **arr[] = { &b[k].X, &b[k].Z, &b[k].Zi, &b[k].dX, &b[k].dZ, &b[k].L,
                       &b[k].T1, &b[k].T2, &b[k].T3, &b[k].Rd, &b[k].Q };
        for (size_t q = 0; q < sizeof(arr) / sizeof(arr[0]); q++) *arr[q] = dalloc(len);
        for (size_t i = 0; i < len; i++) { b[k].X[i] = dd_of(Xio[k][i]); b[k].Z[i] = dd_of(Zio[k][i]); }
        ndim += B->n;
    }
    dd *M = dalloc((size_t)m * m), *LM = dalloc((size_t)m * m);
    /* best iterate so far (the time limit may stop the loop at a worse point) */
    double **Xb = malloc(sizeof(double *) * nb), **Zb = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) { Xb[k] = malloc(sizeof(double) * b[k].len); Zb[k] = malloc(sizeof(double) * b[k].len); }
    double *yb = malloc(sizeof(double) * (m + 1));
    dd *rp = dalloc(m), *rhs = dalloc(m), *dy = dalloc(m), *yv = dalloc(m);
    /* `col` is used as a per-block scratch column: size it by the largest block, not m
     * (ss30 has m = 132 and a block of 294, which overran the buffer) */
    size_t colmax = (size_t)m + 1;
    for (int k = 0; k < nb; k++) if ((size_t)P->blk[k].n + 1 > colmax) colmax = (size_t)P->blk[k].n + 1;
    dd *a0 = dalloc(m), *hv = dalloc(m), *qv = dalloc(m), *col = dalloc(colmax);
    for (int i = 0; i < m; i++) yv[i] = dd_of(y[i]);
    const double sc = P->bs * P->cs;
    double best = 1e300;
    int used = 0, improved = 0;

    double score0 = -1;
    int nostall = 0;
    double best_inf = 1e300;
    for (int it = 0; it <= maxit; it++) {
        if (t_limit > 0 && dd_now() - t_start > t_limit) {
            if (verbose > 0) printf("   dd endgame: time limit (%.1fs) reached\n", t_limit);
            break;
        }
        if (brisk_stop_flag || (par->timelimit > 0 && dd_now() - par->t_start > par->timelimit)) break;
        /* ---- residuals, objectives */
        for (int i = 0; i < m; i++) rp[i] = dd_of(P->b[i]);
        dd pobj = dd_of(0.0), dobj = dd_of(0.0), xz = dd_of(0.0);
        double rd2 = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            dd_Aop(B, &b[k], b[k].X, rp, -1.0);
            dd_ATy(B, &b[k], yv, b[k].Rd);
            for (size_t i = 0; i < b[k].len; i++) b[k].Rd[i] = dd_neg(dd_add(b[k].Rd[i], b[k].Z[i]));
            dd_add_sp(B, &b[k], &B->C, 1.0, b[k].Rd);                    /* Rd = C - Z - A'y */
            for (size_t i = 0; i < b[k].len; i++) {
                double v = dd_d(b[k].Rd[i]);
                rd2 += v * v;
                xz = dd_add(xz, dd_mul(b[k].X[i], b[k].Z[i]));
            }
            dd s = dd_of(0.0);
            dd_add_sp(B, &b[k], &B->C, 1.0, b[k].T1);                    /* <C, X> */
            for (size_t i = 0; i < b[k].len; i++) s = dd_add(s, dd_mul(b[k].T1[i], b[k].X[i]));
            memset(b[k].T1, 0, sizeof(dd) * b[k].len);
            pobj = dd_add(pobj, s);
        }
        for (int i = 0; i < m; i++) dobj = dd_add(dobj, dd_mul(dd_of(P->b[i]), yv[i]));
        double rpo = 0;
        for (int i = 0; i < m; i++) { double v = dd_d(rp[i]) * P->du[i]; rpo += v * v; }
        rpo = sqrt(rpo);
        double po = sc * dd_d(pobj), dob = sc * dd_d(dobj), den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        double pinf = rpo / (1 + P->normb2), dinf = P->cs * sqrt(rd2) / (1 + P->normC2);
        double gap = fabs(po - dob) / den, comp = fabs(sc * dd_d(xz)) / den;
        double score = fmax(fmax(pinf, dinf), fmax(gap, comp));
        double mu = dd_d(xz) / ndim;
        if (verbose > 0)
            printf("   dd %2d  pinf %.2e  dinf %.2e  gap %.2e  compl %.2e  mu %.2e\n",
                   it, pinf, dinf, gap, comp, mu);
        if (score0 < 0) score0 = score;
        /* stall exit: stop after 6 iterations that do not improve the best score by 10%
         * (the endgame is also bounded by its work budget; on the LMI families it
         * converges slowly but steadily, which a stricter rule cuts off)              */
        /* progress also counts when infeasibility falls while the gap is still opening
         * up (hinf9: 13 such iterations, then quadratic convergence)                  */
        const double inf = fmax(pinf, dinf);
        if (score < 0.9 * best || inf < 0.9 * best_inf) nostall = 0; else if (it > 0) nostall++;
        if (inf < best_inf) best_inf = inf;
        if (score < best * (1 - 1e-12)) {
            best = score;
            if (score < score0) improved = 1;
            for (int k = 0; k < nb; k++)
                for (size_t i = 0; i < b[k].len; i++) { Xb[k][i] = dd_d(b[k].X[i]); Zb[k][i] = dd_d(b[k].Z[i]); }
            for (int i = 0; i < m; i++) yb[i] = dd_d(yv[i]);
        }
        if (score < tol) { used = 1; break; }
        if (!(mu > 0) || it == maxit || nostall >= 6) break;

        /* ---- Z^{-1}, Schur complement, Cholesky */
        int fail = 0;
        for (int k = 0; k < nb && !fail; k++) {
            const int n = b[k].n;
            if (b[k].type == BLK_LP) {
                for (int i = 0; i < n; i++) b[k].Zi[i] = dd_div(dd_of(1.0), b[k].Z[i]);
            } else {
                if (dd_chol(n, b[k].Z, b[k].L)) { fail = 1; break; }
                dd_inv_from_chol(n, b[k].L, b[k].Zi, col);
                dd_sym(n, b[k].Zi);
            }
        }
        if (fail) break;
        memset(M, 0, sizeof(dd) * (size_t)m * m);
        for (int k = 0; k < nb; k++) dd_schur(&P->blk[k], &b[k], m, M);
        for (int j = 0; j < m; j++)
            for (int i = j + 1; i < m; i++) M[j + (size_t)i * m] = M[i + (size_t)j * m];
        if (dd_chol(m, M, LM)) {                                   /* tiny shift if needed */
            for (int i = 0; i < m; i++) M[i + (size_t)i * m] = dd_add(M[i + (size_t)i * m],
                                                                      dd_muld(M[i + (size_t)i * m], 1e-24));
            if (dd_chol(m, M, LM)) break;
        }

        /* ---- right-hand side pieces: a0 = A(Zi), h = A(Zi Rd X) */
        memset(a0, 0, sizeof(dd) * m);
        memset(hv, 0, sizeof(dd) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            const int n = b[k].n;
            dd_Aop(B, &b[k], b[k].Zi, a0, 1.0);
            if (b[k].type == BLK_LP)
                for (int i = 0; i < n; i++) b[k].T1[i] = dd_mul(b[k].Zi[i], dd_mul(b[k].Rd[i], b[k].X[i]));
            else {
                dd_gemm(n, b[k].Zi, b[k].Rd, b[k].T2);
                dd_gemm(n, b[k].T2, b[k].X, b[k].T1);
                dd_sym(n, b[k].T1);
            }
            dd_Aop(B, &b[k], b[k].T1, hv, 1.0);
        }

        /* ---- predictor: M dy = b + h ; dZ = Rd - A'dy ; dX = -sym(Zi dZ X) - X */
        for (int i = 0; i < m; i++) rhs[i] = dd_add(dd_of(P->b[i]), hv[i]);
        memcpy(dy, rhs, sizeof(dd) * m);
        dd_solve_chol(m, LM, dy);
        for (int k = 0; k < nb; k++) dd_dir(&P->blk[k], &b[k], dy, dd_of(0.0), 0);
        dd ap, ad;
        dd_steps(P, b, nb, &ap, &ad);

        /* ---- centering parameter from the affine step */
        dd xza = dd_of(0.0);
        for (int k = 0; k < nb; k++)
            for (size_t i = 0; i < b[k].len; i++) {
                dd xa = dd_add(b[k].X[i], dd_mul(ap, b[k].dX[i]));
                dd za = dd_add(b[k].Z[i], dd_mul(ad, b[k].dZ[i]));
                xza = dd_add(xza, dd_mul(xa, za));
            }
        double mua = dd_d(xza) / ndim, frac = mua / mu;
        double mn0 = fmin(dd_d(ap), dd_d(ad));
        double expo = gap > 1e-6 ? fmax(1.0, 3.0 * mn0 * mn0) : 3.0;
        double sigma = fmin(1.0, pow(frac, expo));
        dd smu = dd_muld(dd_of(mu), sigma);

        /* ---- corrector: Q = sym(Zi dZa dXa); rhs = b - sigma mu a0 + A(Q) + h */
        memset(qv, 0, sizeof(dd) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            const int n = b[k].n;
            if (b[k].type == BLK_LP)
                for (int i = 0; i < n; i++) b[k].Q[i] = dd_mul(b[k].Zi[i], dd_mul(b[k].dZ[i], b[k].dX[i]));
            else {
                dd_gemm(n, b[k].Zi, b[k].dZ, b[k].T2);
                dd_gemm(n, b[k].T2, b[k].dX, b[k].Q);
                dd_sym(n, b[k].Q);
            }
            dd_Aop(B, &b[k], b[k].Q, qv, 1.0);
        }
        for (int i = 0; i < m; i++)
            rhs[i] = dd_add(dd_sub(dd_of(P->b[i]), dd_mul(a0[i], smu)), dd_add(qv[i], hv[i]));
        memcpy(dy, rhs, sizeof(dd) * m);
        dd_solve_chol(m, LM, dy);
        for (int k = 0; k < nb; k++) dd_dir(&P->blk[k], &b[k], dy, smu, 1);
        dd_steps(P, b, nb, &ap, &ad);

        double mn = fmin(dd_d(ap), dd_d(ad));
        double gam = 0.9 + (par->gamma_max - 0.9) * mn;
        dd sp = dd_muld(ap, gam), sdd = dd_muld(ad, gam);
        if (dd_d(sp) > 1) sp = dd_of(1.0);
        if (dd_d(sdd) > 1) sdd = dd_of(1.0);
        if (dd_d(sp) < 1e-10 && dd_d(sdd) < 1e-10) break;
        for (int k = 0; k < nb; k++) {
            for (size_t i = 0; i < b[k].len; i++) {
                b[k].X[i] = dd_add(b[k].X[i], dd_mul(sp, b[k].dX[i]));
                b[k].Z[i] = dd_add(b[k].Z[i], dd_mul(sdd, b[k].dZ[i]));
            }
            if (b[k].type != BLK_LP) { dd_sym(b[k].n, b[k].X); dd_sym(b[k].n, b[k].Z); }
        }
        for (int i = 0; i < m; i++) yv[i] = dd_add(yv[i], dd_mul(sdd, dy[i]));
        used = 1;
    }

    if (improved) {
        for (int k = 0; k < nb; k++) {
            memcpy(Xio[k], Xb[k], sizeof(double) * b[k].len);
            memcpy(Zio[k], Zb[k], sizeof(double) * b[k].len);
        }
        memcpy(y, yb, sizeof(double) * m);
    }
    for (int k = 0; k < nb; k++) { free(Xb[k]); free(Zb[k]); }
    free(Xb); free(Zb); free(yb);
    for (int k = 0; k < nb; k++) {
        free(b[k].X); free(b[k].Z); free(b[k].Zi); free(b[k].dX); free(b[k].dZ); free(b[k].L);
        free(b[k].T1); free(b[k].T2); free(b[k].T3); free(b[k].Rd); free(b[k].Q);
    }
    free(b); free(M); free(LM); free(rp); free(rhs); free(dy); free(yv);
    free(a0); free(hv); free(qv); free(col);
    return used && improved;
}
