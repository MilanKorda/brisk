/* lralm.c - 4.40: a low-rank augmented Lagrangian method (Burer-Monteiro) for large sparse
 * SDPs, `-lralm 1`.
 *
 * Target: problems whose Schur complement cannot be formed or factored at all, in particular
 * the Shor relaxations of large AC-OPF instances (one network-sparse block of n = 40 000 -
 * 160 000 with a rank-one or low-rank optimum, plus LP slacks of the inequalities). The
 * chordal conversion of case19402 already has cliques of 224 and an estimated Schur factor
 * of 1e17 flops per iteration.
 *
 * Method (SDPLR-type, with the inequalities in closed form):
 *   X_k = R_k R_k' per SDP block (R_k n_k x r_k, row-major), LP variables x = v^2;
 *   an LP variable that is a pure slack (cost 0, one constraint, coefficient a) is not a
 *   variable: the augmented Lagrangian is minimized over it in closed form, so constraint
 *   i has the residual t_i = A_i(X) + (other LP terms) - b_i and the penalty
 *       pen_i = -y_i u_i + sigma/2 u_i^2,  u_i = t_i (equality),
 *       u_i = max(t_i, y_i/sigma) (slack with a > 0), min(t_i, y_i/sigma) (a < 0),
 *   which is C^1 in X; the multiplier estimate is w_i = y_i - sigma u_i (sign-clipped for
 *   the slack rows), the multiplier step y <- w.
 *   L(R, v) = <C,X> + c'x + sum_i pen_i, gradient 2 (C - A*(w)) R, 2 (c - A'w) o v.
 *   Inner: L-BFGS (memory 10) with an exact line search: along a direction D every
 *   residual is a quadratic polynomial in the step (t0 + a t1 + a^2 t2, two passes over
 *   the data), so the line function and its derivative cost O(m) per trial step; the
 *   first stationary point is found by bracketing and safeguarded secant steps.
 *   Outer: y <- w; sigma x 4 when |u| did not fall below 1/4 of its last value.
 *   Measures: Z = C - A*y on the data pattern; lambda_min(Z) by a sparse Cholesky ladder
 *   (the network pattern factors cheaply); rank escape along the eigenvector of the most
 *   negative eigenvalue of Z from a Lanczos run when it is clearly negative.
 * No presolve runs on this path (the measures are those of the problem as read, up to the
 * row equilibration, which is undone). Nothing of size n^2 or m^2 is formed.               */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void *lx(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (low-rank ALM, %.2f GB)\n", n / 1e9); exit(1); } return p; }
static void *lz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (low-rank ALM, %.2f GB)\n", n / 1e9); exit(1); } return p; }

/* one SDP block */
typedef struct {
    int k, n, r, rmax;
    int np, *sp, *sj;            /* full symmetric pattern of C and all A_t (CSR, sorted) */
    int *dg;                     /* position of (i,i) in the pattern, -1 if absent */
    double *xs, *ws, *wq;        /* per pattern entry: X (or a bilinear form), S = C - A*(w), A*(q) */
    int *sup, *supp;             /* constraint t: its distinct rows (the support) */
    SChol *F;                    /* preconditioner: 2 S + 4 sigma sum_t (A_t R)(A_t R)' + delta I */
    double fshift, fabs_; int Fr;
    SChol *Fz;                   /* Z's own pattern (the bound test) */
    int ncon; const int *con;
    int *cp, *ep; double *ev;    /* constraint t: full entries -> pattern positions, values */
    int cne, *cep; double *cev;  /* C */
    size_t off;                  /* offset of R in the stacked vector */
} LS;

/* LP variables that remain variables (x = v^2) */
typedef struct { int k, j; int nc; int *ci; double *cv; double c; size_t off; } LV;

typedef struct {
    Problem *P;
    int m, ns, nv;
    LS *S; LV *V;
    size_t N;                    /* length of the stacked vector */
    double *sa;                  /* slack coefficient per row (0: equality) */
    int *sk, *sj2;               /* slack variable (block, index) per row */
    double *b;                   /* scaled rhs */
    double sigma; int verbose;
} LR;

static int cmp_i2(const void *a, const void *b) { const int *x = a, *y = b; return x[0] != y[0] ? x[0] - y[0] : x[1] - y[1]; }
static int spos(const LS *s, int i, int j) {
    int lo = s->sp[i], hi = s->sp[i + 1] - 1;
    while (lo <= hi) { const int mid = (lo + hi) >> 1; if (s->sj[mid] == j) return mid; if (s->sj[mid] < j) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

static void ls_setup(LS *s, const Block *B, int k) {
    const int n = B->n;
    s->k = k; s->n = n; s->ncon = B->ncon; s->con = B->con;
    size_t tot = B->C.ef + (size_t)n; for (int t = 0; t < B->ncon; t++) tot += B->A[t].ef;
    int (*pr)[2] = lx(sizeof(int[2]) * (tot ? tot : 1));
    size_t np = 0;
    for (int i = 0; i < n; i++) { pr[np][0] = i; pr[np][1] = i; np++; }   /* the diagonal always (Lanczos shift, ladder) */
    for (int q = 0; q < B->C.ef; q++) { pr[np][0] = B->C.fr[q]; pr[np][1] = B->C.fc[q]; np++; }
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) { pr[np][0] = B->A[t].fr[q]; pr[np][1] = B->A[t].fc[q]; np++; }
    qsort(pr, np, sizeof(int[2]), cmp_i2);
    size_t nu = 0;
    for (size_t p = 0; p < np; p++) if (!nu || pr[nu - 1][0] != pr[p][0] || pr[nu - 1][1] != pr[p][1]) { pr[nu][0] = pr[p][0]; pr[nu][1] = pr[p][1]; nu++; }
    s->np = (int)nu;
    s->sp = lz(sizeof(int) * (n + 1)); s->sj = lx(sizeof(int) * (nu + 1));
    for (size_t p = 0; p < nu; p++) { s->sp[pr[p][0] + 1]++; s->sj[p] = pr[p][1]; }
    for (int i = 0; i < n; i++) s->sp[i + 1] += s->sp[i];
    free(pr);
    s->dg = lx(sizeof(int) * n);
    for (int i = 0; i < n; i++) s->dg[i] = spos(s, i, i);
    s->cp = lx(sizeof(int) * (B->ncon + 1)); s->cp[0] = 0;
    for (int t = 0; t < B->ncon; t++) s->cp[t + 1] = s->cp[t] + B->A[t].ef;
    s->ep = lx(sizeof(int) * ((size_t)s->cp[B->ncon] + 1)); s->ev = lx(sizeof(double) * ((size_t)s->cp[B->ncon] + 1));
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) { s->ep[s->cp[t] + q] = spos(s, B->A[t].fr[q], B->A[t].fc[q]); s->ev[s->cp[t] + q] = B->A[t].fv[q]; }
    s->cne = B->C.ef; s->cep = lx(sizeof(int) * (s->cne + 1)); s->cev = lx(sizeof(double) * (s->cne + 1));
    for (int q = 0; q < B->C.ef; q++) { s->cep[q] = spos(s, B->C.fr[q], B->C.fc[q]); s->cev[q] = B->C.fv[q]; }
    s->xs = lx(sizeof(double) * (nu + 1)); s->ws = lx(sizeof(double) * (nu + 1)); s->wq = lx(sizeof(double) * (nu + 1));
    /* supports */
    s->supp = lx(sizeof(int) * (B->ncon + 1)); s->supp[0] = 0;
    {   char *on = lz(n + 1); size_t cap = 1024, ns = 0; s->sup = lx(sizeof(int) * cap);
        for (int t = 0; t < B->ncon; t++) {
            for (int q = 0; q < B->A[t].ef; q++) { const int i = B->A[t].fr[q]; if (!on[i]) { on[i] = 1; if (ns == cap) { cap *= 2; s->sup = realloc(s->sup, sizeof(int) * cap); } s->sup[ns++] = i; } }
            for (size_t a = s->supp[t]; a < ns; a++) on[s->sup[a]] = 0;
            s->supp[t + 1] = (int)ns;
        }
        free(on);
    }
}
static void ls_free(LS *s) { free(s->sp); free(s->sj); free(s->dg); free(s->cp); free(s->ep); free(s->ev); free(s->cep); free(s->cev); free(s->xs); free(s->ws); free(s->wq); free(s->sup); free(s->supp); if (s->F) schol_free(s->F); if (s->Fz) schol_free(s->Fz); }

/* xs[p] = (U V' + V U')_{ij} / (1 + same) on the pattern: sym = 0: (U U')_{ij}; 1: U_i.V_j + V_i.U_j */
static void pat_form(const LS *s, const double *U, const double *V, int sym) {
    const int n = s->n, r = s->r;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ui = U + (size_t)i * r, *vi = V ? V + (size_t)i * r : NULL;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
            const int j = s->sj[p]; const double *uj = U + (size_t)j * r;
            double v = 0;
            if (!sym) for (int l = 0; l < r; l++) v += ui[l] * uj[l];
            else { const double *vj = V + (size_t)j * r; for (int l = 0; l < r; l++) v += ui[l] * vj[l] + vi[l] * uj[l]; }
            s->xs[p] = v;
        }
    }
}
/* out[con] += <A_t, xs>, returns <C, xs> */
static double pat_apply(const LS *s, double *out) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < s->ncon; t++) { double v = 0; for (int e = s->cp[t]; e < s->cp[t + 1]; e++) v += s->ev[e] * s->xs[s->ep[e]]; out[s->con[t]] += v; }
    double c = 0; for (int e = 0; e < s->cne; e++) c += s->cev[e] * s->xs[s->cep[e]];
    return c;
}
/* ws = C - A*(w) on the pattern */
static void pat_slack(const LS *s, const double *w) {
    memset(s->ws, 0, sizeof(double) * s->np);
    for (int e = 0; e < s->cne; e++) s->ws[s->cep[e]] += s->cev[e];
    for (int t = 0; t < s->ncon; t++) { const double wt = w[s->con[t]]; if (wt == 0) continue; for (int e = s->cp[t]; e < s->cp[t + 1]; e++) s->ws[s->ep[e]] -= wt * s->ev[e]; }
}
/* O = c * Ws U (pattern SpMV, r columns, row-major) */
static void pat_mul(const LS *s, const double *U, double *O, double c, int r) {
    const int n = s->n;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *oi = O + (size_t)i * r;
        for (int l = 0; l < r; l++) oi[l] = 0;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const double a = s->ws[p]; const double *uj = U + (size_t)s->sj[p] * r; for (int l = 0; l < r; l++) oi[l] += a * uj[l]; }
        for (int l = 0; l < r; l++) oi[l] *= c;
    }
}

/* residuals t = A(X) + LP terms - b and the cost, at z (no slacks) */
static double eval_t(LR *L, const double *z, double *t) {
    const int m = L->m;
    for (int i = 0; i < m; i++) t[i] = -L->b[i];
    double c = 0;
    for (int q = 0; q < L->ns; q++) { LS *s = &L->S[q]; pat_form(s, z + s->off, NULL, 0); c += pat_apply(s, t); }
    for (int q = 0; q < L->nv; q++) { const LV *v = &L->V[q]; const double x = z[v->off] * z[v->off]; c += v->c * x; for (int e = 0; e < v->nc; e++) t[v->ci[e]] += v->cv[e] * x; }
    return c;
}
/* the polynomial coefficients along D: t1, t2 and c1, c2 */
static void eval_dir(LR *L, const double *z, const double *D, double *t1, double *t2, double *c1, double *c2) {
    const int m = L->m;
    memset(t1, 0, sizeof(double) * m); memset(t2, 0, sizeof(double) * m);
    double a1 = 0, a2 = 0;
    for (int q = 0; q < L->ns; q++) {
        LS *s = &L->S[q];
        pat_form(s, z + s->off, D + s->off, 1); a1 += pat_apply(s, t1);
        pat_form(s, D + s->off, NULL, 0); a2 += pat_apply(s, t2);
    }
    for (int q = 0; q < L->nv; q++) {
        const LV *v = &L->V[q]; const double x1 = 2 * z[v->off] * D[v->off], x2 = D[v->off] * D[v->off];
        a1 += v->c * x1; a2 += v->c * x2;
        for (int e = 0; e < v->nc; e++) { t1[v->ci[e]] += v->cv[e] * x1; t2[v->ci[e]] += v->cv[e] * x2; }
    }
    *c1 = a1; *c2 = a2;
}
/* u, w and the penalty sum from t */
static inline double clip_u(double t, double y, double sigma, double a) {
    if (a > 0) { const double q = y / sigma; return t > q ? t : q; }
    if (a < 0) { const double q = y / sigma; return t < q ? t : q; }
    return t;
}
static double pen_sum(const LR *L, const double *t, const double *y, double *w) {
    double f = 0; const double sg = L->sigma;
    for (int i = 0; i < L->m; i++) {
        const double u = clip_u(t[i], y[i], sg, L->sa[i]);
        f += -y[i] * u + 0.5 * sg * u * u;
        if (w) w[i] = y[i] - sg * u;
    }
    return f;
}
/* gradient at z for the multiplier estimate w */
static void grad(LR *L, const double *z, const double *w, double *g) {
    for (int q = 0; q < L->ns; q++) { LS *s = &L->S[q]; pat_slack(s, w); pat_mul(s, z + s->off, g + s->off, 2.0, s->r); }
    for (int q = 0; q < L->nv; q++) { const LV *v = &L->V[q]; double sv = v->c; for (int e = 0; e < v->nc; e++) sv -= w[v->ci[e]] * v->cv[e]; g[v->off] = 2.0 * sv * z[v->off]; }
}
/* phi'(a) along the direction */
static double dphi(const LR *L, double a, const double *t0, const double *t1, const double *t2, const double *y, double c1, double c2) {
    double d = c1 + 2 * c2 * a; const double sg = L->sigma;
    for (int i = 0; i < L->m; i++) {
        const double ti = t0[i] + a * (t1[i] + a * t2[i]);
        const double u = clip_u(ti, y[i], sg, L->sa[i]);
        d -= (y[i] - sg * u) * (t1[i] + 2 * a * t2[i]);
    }
    return d;
}
static double lsearch(const LR *L, const double *t0, const double *t1, const double *t2, const double *y, double c1, double c2, double d0, int *nev) {
    double lo = 0, dlo = d0, hi = 1, dhi = dphi(L, hi, t0, t1, t2, y, c1, c2);
    int k = 1;
    while (dhi < 0 && k < 80) { lo = hi; dlo = dhi; hi *= 2; dhi = dphi(L, hi, t0, t1, t2, y, c1, c2); k++; }
    if (dhi < 0) { *nev += k; return hi; }
    /* Illinois on [lo, hi]; the point with the smallest |phi'| is returned */
    int side = 0; double best = fabs(dlo) < fabs(dhi) ? lo : hi, dbest = fmin(fabs(dlo), fabs(dhi));
    for (int it = 0; it < 60; it++) {
        double a = (lo * dhi - hi * dlo) / (dhi - dlo);
        if (!(a > lo && a < hi)) a = 0.5 * (lo + hi);
        const double da = dphi(L, a, t0, t1, t2, y, c1, c2); k++;
        if (fabs(da) < dbest) { dbest = fabs(da); best = a; }
        if (da < 0) { lo = a; dlo = da; if (side == -1) dhi *= 0.5; side = -1; }
        else { hi = a; dhi = da; if (side == 1) dlo *= 0.5; side = 1; }
        if (hi - lo <= 1e-7 * hi || fabs(da) <= 1e-10 * fabs(d0)) break;
    }
    *nev += k;
    return best > 0 ? best : 0.5 * (lo + hi);
}

/* Lanczos for the smallest eigenpair of Z = ws (pattern) of block s: k steps, full
 * reorthogonalization; returns theta, vector in vout (n) */
void BL(dstev_)(const char *, const int *, double *, double *, double *, const int *, double *, int *);
void BL(dsyev_)(const char *, const char *, const int *, double *, const int *, double *, double *, const int *, int *);
static double lanczos_min(const LS *s, int k, double *vout, unsigned seed) {
    const int n = s->n;
    if (k > n) k = n;
    double *Q = lx(sizeof(double) * (size_t)n * (k + 1)), *al = lx(sizeof(double) * (k + 1)), *be = lx(sizeof(double) * (k + 1)), *w = lx(sizeof(double) * n);
    unsigned h = seed * 2654435761u + 12345u; double nr = 0;
    for (int i = 0; i < n; i++) { h = h * 1103515245u + 12345u; Q[i] = ((h >> 8) & 0xffff) / 65535.0 - 0.5; nr += Q[i] * Q[i]; }
    nr = sqrt(nr); for (int i = 0; i < n; i++) Q[i] /= nr;
    int kk = 0;
    for (int j = 0; j < k; j++) {
        const double *qj = Q + (size_t)j * n;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { double v = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) v += s->ws[p] * qj[s->sj[p]]; w[i] = v; }
        double a = 0; for (int i = 0; i < n; i++) a += w[i] * qj[i];
        al[j] = a; kk = j + 1;
        for (int c = 0; c <= j; c++) { const double *qc = Q + (size_t)c * n; double d = 0; for (int i = 0; i < n; i++) d += w[i] * qc[i]; for (int i = 0; i < n; i++) w[i] -= d * qc[i]; }
        for (int c = 0; c <= j; c++) { const double *qc = Q + (size_t)c * n; double d = 0; for (int i = 0; i < n; i++) d += w[i] * qc[i]; for (int i = 0; i < n; i++) w[i] -= d * qc[i]; }
        double bn = 0; for (int i = 0; i < n; i++) bn += w[i] * w[i]; bn = sqrt(bn);
        be[j] = bn;
        if (bn < 1e-14 || j == k - 1) break;
        double *qn = Q + (size_t)(j + 1) * n; for (int i = 0; i < n; i++) qn[i] = w[i] / bn;
    }
    double *d = lx(sizeof(double) * kk), *e = lx(sizeof(double) * kk), *Zv = lx(sizeof(double) * (size_t)kk * kk), *wk = lx(sizeof(double) * (2 * kk + 2));
    memcpy(d, al, sizeof(double) * kk); for (int i = 0; i + 1 < kk; i++) e[i] = be[i];
    int info = 0; BL(dstev_)("V", &kk, d, e, Zv, &kk, wk, &info);
    const double th = d[0];
    if (vout) { for (int i = 0; i < n; i++) vout[i] = 0; for (int c = 0; c < kk; c++) { const double zc = Zv[c]; const double *qc = Q + (size_t)c * n; for (int i = 0; i < n; i++) vout[i] += zc * qc[i]; } }
    free(Q); free(al); free(be); free(w); free(d); free(e); free(Zv); free(wk);
    return th;
}

/* lambda_min(Z) >= -delta by a sparse Cholesky ladder on the pattern (Z = ws): returns the
 * smallest delta of the ladder (0 if Z is positive definite; > 0 otherwise; -1 if the
 * pattern does not factor) */
static double ladder(const LS *s, double scale) {
    const int n = s->n;
    int *deg = lz(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) deg[i]++;
    int **nbr = lx(sizeof(int *) * (n + 1));
    for (int i = 0; i < n; i++) { nbr[i] = lx(sizeof(int) * (deg[i] + 1)); int c = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) nbr[i][c++] = s->sj[p]; }
    SChol *F = schol_analyze_adj(n, deg, nbr, (size_t)fmin(2e8, 0.3 * (double)n * n) + 1000);
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(deg);
    if (!F) return -1;
    double res = -1;
    for (int at = 0; at < 14; at++) {
        const double sh = at == 0 ? 0.0 : scale * pow(10.0, -13.0 + at);
        schol_zero(F);
        for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const int j = s->sj[p]; if (j > i) continue; schol_add(F, i, j, s->ws[p] + (i == j ? sh : 0.0)); }
        if (schol_factor(F, 0.0) == 0) { res = sh; break; }
    }
    schol_free(F);
    return res;
}

/* Z = ws positive definite (a sparse Cholesky at shift 0 on Z's pattern, analysed once):
 * with the LP duals nonnegative, b'y is then a lower bound */
static int z_posdef(LS *s) {
    const int n = s->n;
    if (!s->Fz) {
        int *deg = lz(sizeof(int) * (n + 1));
        for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) deg[i]++;
        int **nbr = lx(sizeof(int *) * (n + 1));
        for (int i = 0; i < n; i++) { nbr[i] = lx(sizeof(int) * (deg[i] + 1)); int c = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) nbr[i][c++] = s->sj[p]; }
        s->Fz = schol_analyze_adj(n, deg, nbr, (size_t)fmin(2e8, 0.3 * (double)n * n) + 1000);
        for (int i = 0; i < n; i++) free(nbr[i]);
        free(nbr); free(deg);
        if (!s->Fz) return 0;
    }
    schol_zero(s->Fz);
    for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const int j = s->sj[p]; if (j <= i) schol_add(s->Fz, i, j, s->ws[p]); }
    return schol_factor(s->Fz, 0.0) == 0;
}

static double nrm(const double *x, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += x[i] * x[i]; return sqrt(s); }
static double dotv(const double *x, const double *y, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += x[i] * y[i]; return s; }

static void spsym_scale(SpSym *S, double s) { for (int k = 0; k < S->nnz; k++) S->val[k] *= s; for (int k = 0; k < S->ef; k++) S->fv[k] *= s; }
/* row equilibration as problem_prepare does it, without the routing analysis */
static void lr_scale(Problem *P) {
    const int m = P->m;
    double *nrm2 = lz(sizeof(double) * (m + 1)), c2 = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) nrm2[B->con[t]] += B->A[t].fv[q] * B->A[t].fv[q];
        for (int q = 0; q < B->C.ef; q++) c2 += B->C.fv[q] * B->C.fv[q];
    }
    P->normC2 = sqrt(c2);
    P->normb2 = 0; for (int i = 0; i < m; i++) P->normb2 += P->b0[i] * P->b0[i];
    P->normb2 = sqrt(P->normb2);
    P->d = lx(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) P->d[i] = nrm2[i] > 0 ? 1.0 / sqrt(nrm2[i]) : 1.0;
    free(nrm2);
    double nb = 0; for (int i = 0; i < m; i++) nb += P->d[i] * P->b[i] * P->d[i] * P->b[i];
    nb = sqrt(nb); P->bs = nb > 1 ? nb : 1.0;
    for (int k = 0; k < P->nblk; k++) { Block *B = &P->blk[k]; for (int t = 0; t < B->ncon; t++) spsym_scale(&B->A[t], P->d[B->con[t]]); }
    for (int i = 0; i < m; i++) P->b[i] *= P->d[i] / P->bs;
    P->du = lx(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) P->du[i] = P->bs / P->d[i];
    P->cs = P->normC2 > 1 ? P->normC2 : 1.0;
    for (int k = 0; k < P->nblk; k++) spsym_scale(&P->blk[k].C, 1.0 / P->cs);
    P->scaled = 1;
}

typedef struct { double pinf, dinf, gap, compl, pobj, dobj, err[7]; double lmin; int lmin_ok; } LMeas;

/* the measures on the problem as read: X = R R' (exactly PSD), slacks from the ALM's u,
 * Z = C - A*y; dinf from the ladder (when asked) or the Lanczos estimate */
static void measure(LR *L, const double *z, const double *y, const double *t, int full, LMeas *M) {
    Problem *P = L->P; const int m = L->m; const double sg = L->sigma;
    double rp = 0;
    for (int i = 0; i < m; i++) { const double u = clip_u(t[i], y[i], sg, L->sa[i]); const double v = u * P->du[i]; rp += v * v; }
    M->pinf = sqrt(rp) / (1 + P->normb2);
    /* objective: <C,X> + c'x; the slacks have no cost */
    double cx = 0;
    for (int q = 0; q < L->ns; q++) { LS *s = &L->S[q]; pat_form(s, z + s->off, NULL, 0); for (int e = 0; e < s->cne; e++) cx += s->cev[e] * s->xs[s->cep[e]]; }
    for (int q = 0; q < L->nv; q++) cx += L->V[q].c * z[L->V[q].off] * z[L->V[q].off];
    double by = 0; for (int i = 0; i < m; i++) by += L->b[i] * y[i];
    const double sc = P->bs * P->cs;
    M->pobj = sc * cx; M->dobj = sc * by;
    const double den = 1 + fabs(M->pobj) + fabs(M->dobj);
    M->gap = (M->pobj - M->dobj) / den;
    /* <X,Z> = sum over blocks <R R', C - A*y> + LP vars x z + slacks x z */
    double xz = 0, lmin = 0, lpneg = 0;
    for (int q = 0; q < L->ns; q++) {
        LS *s = &L->S[q]; pat_slack(s, y);
        for (int p = 0; p < s->np; p++) xz += s->ws[p] * s->xs[p];
    }
    for (int q = 0; q < L->nv; q++) { const LV *v = &L->V[q]; double zv = v->c; for (int e = 0; e < v->nc; e++) zv -= y[v->ci[e]] * v->cv[e]; xz += zv * z[v->off] * z[v->off]; if (zv < lpneg) lpneg = zv; }
    for (int i = 0; i < m; i++) if (L->sa[i] != 0) {
        /* slack x = (u - t)/a >= 0, its dual -a y >= 0 by the clipping */
        const double u = clip_u(t[i], y[i], sg, L->sa[i]); const double x = (u - t[i]) / L->sa[i], zs = -L->sa[i] * y[i];
        xz += x * zs; if (zs < lpneg) lpneg = zs;
    }
    M->compl = sc * xz / den;
    if (full) {
        lmin = 0; M->lmin_ok = 1;
        for (int q = 0; q < L->ns; q++) {
            LS *s = &L->S[q];   /* ws holds Z of this block from the loop above only for the last block: redo */
            pat_slack(s, y);
            double dmax = 0; for (int i = 0; i < s->n; i++) if (s->dg[i] >= 0) dmax = fmax(dmax, fabs(s->ws[s->dg[i]]));
            const double d = ladder(s, dmax > 0 ? dmax : 1.0);
            if (d < 0) M->lmin_ok = 0; else if (-d < lmin) lmin = -d;
        }
        M->lmin = lmin;
    }
    M->dinf = P->cs * fmax(0.0, -fmin(lmin, lpneg)) / (1 + P->normC2);
    M->err[1] = M->pinf; M->err[2] = 0; M->err[3] = 0; M->err[4] = M->dinf; M->err[5] = M->gap; M->err[6] = M->compl;
}

/* ---- 4.40 Newton-CG inner solver ---------------------------------------------------------
 * Hessian of phi at z (multipliers y, sigma, the clipping of the current t):
 *   H D = 2 S(w) D + 2 A*(q) R,  q = sigma mask o J D,  J D = A(R D' + D R')
 * (LP v: 2 s_j d_j + 2 v_j sum_i a_ij q_i). Preconditioner: per block the sparse Cholesky of
 * M = 2 S(w) + 4 sigma sum_{t unclipped} sum_l (A_t R_l)(A_t R_l)' + delta I on the pattern of
 * the constraint supports - an n x n matrix with the network's chordal structure (the m x m
 * moment-space Schur complement of an interior-point method is never needed).            */
static void pat_adj(const LS *s, const double *q) {
    memset(s->wq, 0, sizeof(double) * s->np);
    for (int t = 0; t < s->ncon; t++) { const double qt = q[s->con[t]]; if (qt == 0) continue; for (int e = s->cp[t]; e < s->cp[t + 1]; e++) s->wq[s->ep[e]] += qt * s->ev[e]; }
}
static void jac_dir(LR *L, const double *z, const double *D, double *t1) {
    memset(t1, 0, sizeof(double) * L->m);
    for (int q = 0; q < L->ns; q++) { LS *s = &L->S[q]; pat_form(s, z + s->off, D + s->off, 1); pat_apply(s, t1); }
    for (int q = 0; q < L->nv; q++) { const LV *v = &L->V[q]; const double x1 = 2 * z[v->off] * D[v->off]; for (int e = 0; e < v->nc; e++) t1[v->ci[e]] += v->cv[e] * x1; }
}
/* out = H D; S(w) must be in s->ws (grad() leaves it there); qb is m scratch */
static void hess_vec(LR *L, const double *z, const double *y, const double *t, const double *w, const double *D, double *out, double *qb) {
    const double sg = L->sigma;
    jac_dir(L, z, D, qb);
    for (int i = 0; i < L->m; i++) { const int act = L->sa[i] == 0 || clip_u(t[i], y[i], sg, L->sa[i]) == t[i]; qb[i] = act ? sg * qb[i] : 0.0; }
    for (int q = 0; q < L->ns; q++) {
        LS *s = &L->S[q]; const int n = s->n, r = s->r; const double *R = z + s->off, *Dd = D + s->off; double *O = out + s->off;
        pat_adj(s, qb);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double *oi = O + (size_t)i * r;
            for (int l = 0; l < r; l++) oi[l] = 0;
            for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
                const double a = 2.0 * s->ws[p], bq = 2.0 * s->wq[p]; const double *dj = Dd + (size_t)s->sj[p] * r, *rj = R + (size_t)s->sj[p] * r;
                for (int l = 0; l < r; l++) oi[l] += a * dj[l] + bq * rj[l];
            }
        }
    }
    for (int q = 0; q < L->nv; q++) {
        const LV *v = &L->V[q]; double sv = v->c, aq = 0;
        for (int e = 0; e < v->nc; e++) { sv -= w[v->ci[e]] * v->cv[e]; aq += v->cv[e] * qb[v->ci[e]]; }
        out[v->off] = 2.0 * sv * D[v->off] + 2.0 * z[v->off] * aq;
    }
}
/* build / refactor the preconditioner of each block at (z, y, t, w); S(w) in ws. The unknowns
 * of a block are R's entries in its row-major order (i r + a); M = 2 S (x) I_r + 4 sigma sum_t
 * vec(A_t R) vec(A_t R)' + delta I, i.e. with the Gauss-Newton coupling between the columns
 * (a per-column preconditioner needed ~100 CG steps a Newton step at rank 2-3 against 1-3 at
 * rank 1). The pattern is analysed again when the rank changes. */
static int npat_built = 0;
static int g_lrdbg = -1;
static void nprec_build(LR *L, const double *z, const double *y, const double *t) {
    Problem *P = L->P; const double sg = L->sigma;
    for (int q = 0; q < L->ns; q++) {
        LS *s = &L->S[q]; const Block *B = &P->blk[s->k]; const int n = s->n, r = s->r; const double *R = z + s->off;
        if (s->F && s->Fr != r) { schol_free(s->F); s->F = NULL; }
        if (!s->F) {
            /* base pattern (i > j): S's and the supports' cliques */
            size_t cap = (size_t)s->np + 16, np2 = 0; int (*pr)[2] = lx(sizeof(int[2]) * cap);
            for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] < i) { pr[np2][0] = i; pr[np2][1] = s->sj[p]; np2++; }
            for (int tt = 0; tt < s->ncon; tt++) {
                const int a0 = s->supp[tt], a1 = s->supp[tt + 1];
                for (int a = a0; a < a1; a++) for (int b2 = a0; b2 < a; b2++) {
                    int i = s->sup[a], j = s->sup[b2]; if (i < j) { const int x = i; i = j; j = x; }
                    if (np2 == cap) { cap *= 2; pr = realloc(pr, sizeof(int[2]) * cap); if (!pr) { fprintf(stderr, "brisk: out of memory (low-rank ALM preconditioner)\n"); exit(1); } }
                    pr[np2][0] = i; pr[np2][1] = j; np2++;
                }
            }
            qsort(pr, np2, sizeof(int[2]), cmp_i2);
            size_t nu = 0; for (size_t p = 0; p < np2; p++) if (!nu || pr[nu - 1][0] != pr[p][0] || pr[nu - 1][1] != pr[p][1]) { pr[nu][0] = pr[p][0]; pr[nu][1] = pr[p][1]; nu++; }
            /* expanded: every base pair gives r x r entries, every diagonal an r x r block */
            const int nn = n * r;
            int *deg = lz(sizeof(int) * (nn + 1));
            for (size_t p = 0; p < nu; p++) for (int a = 0; a < r; a++) { deg[pr[p][0] * r + a] += r; deg[pr[p][1] * r + a] += r; }
            for (int i = 0; i < n; i++) for (int a = 0; a < r; a++) deg[i * r + a] += r - 1;
            size_t tot = 0; for (int i = 0; i < nn; i++) tot += deg[i];
            int **nbr = lx(sizeof(int *) * (nn + 1)), *fl = lz(sizeof(int) * (nn + 1)), *pool = lx(sizeof(int) * (tot + 1));
            size_t o = 0; for (int i = 0; i < nn; i++) { nbr[i] = pool + o; o += deg[i]; }
            for (size_t p = 0; p < nu; p++) { const int i = pr[p][0], j = pr[p][1]; for (int a = 0; a < r; a++) for (int c = 0; c < r; c++) { const int I = i * r + a, J = j * r + c; nbr[I][fl[I]++] = J; nbr[J][fl[J]++] = I; } }
            for (int i = 0; i < n; i++) for (int a = 0; a < r; a++) for (int c = 0; c < r; c++) if (a != c) { const int I = i * r + a; nbr[I][fl[I]++] = i * r + c; }
            free(pr);
            const double t0 = wtime();
            s->F = schol_analyze_adj(nn, deg, nbr, (size_t)4e8);
            free(nbr); free(fl); free(pool); free(deg);
            if (!s->F) { fprintf(stderr, "brisk: low-rank ALM: the preconditioner pattern does not factor\n"); exit(1); }
            if (L->verbose > 0 || !npat_built) printf("lralm: preconditioner of block %d at rank %d: %d unknowns, %zu base pairs, factor %.3g nonzeros, %.3g flops, analysis %.2fs\n", s->k + 1, r, nn, nu, (double)schol_nnz(s->F), schol_flops(s->F), wtime() - t0);
            npat_built++;
            s->Fr = r;
            if (s->fshift <= 0) s->fshift = 1e-8;
        }
        double dmax = 0;
        double *av = lx(sizeof(double) * ((size_t)n * r + 1));   /* (A_t R)_i per support row */
        for (int at = 0; at < 30; at++) {
            schol_zero(s->F);
            for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] <= i) { const int j = s->sj[p]; for (int a = 0; a < r; a++) schol_add(s->F, (size_t)i * r + a, (size_t)j * r + a, 2.0 * s->ws[p]); }
            for (int tt = 0; tt < s->ncon; tt++) {
                const int i0 = B->con[tt];
                if (L->sa[i0] != 0 && clip_u(t[i0], y[i0], sg, L->sa[i0]) != t[i0]) continue;
                const SpSym *A = &B->A[tt]; const int a0 = s->supp[tt], a1 = s->supp[tt + 1];
                for (int c = a0; c < a1; c++) for (int l = 0; l < r; l++) av[(size_t)s->sup[c] * r + l] = 0;
                for (int e = 0; e < A->ef; e++) { const int i = A->fr[e], j = A->fc[e]; for (int l = 0; l < r; l++) av[(size_t)i * r + l] += A->fv[e] * R[(size_t)j * r + l]; }
                for (int c = a0; c < a1; c++) for (int d2 = a0; d2 < a1; d2++) {
                    const int i = s->sup[c], j = s->sup[d2];
                    for (int a = 0; a < r; a++) for (int b2 = 0; b2 < r; b2++) {
                        const size_t I = (size_t)i * r + a, J = (size_t)j * r + b2;
                        if (I < J) continue;
                        schol_add(s->F, I, J, 4.0 * sg * av[(size_t)i * r + a] * av[(size_t)j * r + b2]);
                    }
                }
            }
            if (at == 0) { dmax = 0; for (size_t i = 0; i < (size_t)n * r; i++) dmax = fmax(dmax, fabs(schol_get(s->F, i, i))); if (dmax == 0) dmax = 1; }
            const double sh = s->fshift * dmax;
            for (size_t i = 0; i < (size_t)n * r; i++) schol_add(s->F, i, i, sh);
            if (schol_factor(s->F, 0.0) == 0) { s->fabs_ = sh; if (at == 0 && s->fshift > 1e-12) s->fshift *= 0.1; break; }
            s->fshift = fmin(s->fshift * 4.0, 1.0);
        }
        free(av);
    }
}
/* out = M^-1 v per block (one solve over its n r unknowns), LP entries by their diagonal */
static void nprec_apply(LR *L, const double *z, const double *y, const double *t, const double *w, const double *v, double *out, double *col, double *wk) {
    const double sg = L->sigma; (void)col;
    for (int q = 0; q < L->ns; q++) {
        LS *s = &L->S[q]; const size_t nn = (size_t)s->n * s->r;
        memcpy(out + s->off, v + s->off, sizeof(double) * nn);
        schol_solve(s->F, out + s->off, wk);
    }
    for (int q = 0; q < L->nv; q++) {
        const LV *lv = &L->V[q]; double sv = lv->c, gn = 0; const double x = z[lv->off];
        for (int e = 0; e < lv->nc; e++) { sv -= w[lv->ci[e]] * lv->cv[e]; const int i0 = lv->ci[e]; if (L->sa[i0] == 0 || clip_u(t[i0], y[i0], sg, L->sa[i0]) == t[i0]) gn += 4.0 * lv->cv[e] * lv->cv[e] * x * x; }
        const double h = 2.0 * fabs(sv) + 4.0 * sg * gn;
        out[lv->off] = v[lv->off] / (h > 1e-12 ? h : 1e-12);
    }
}

/* the diagonal of the Gauss-Newton model of L at z for the multipliers y: per entry of R
 * 2 |S_ii| + 4 sigma sum_{rows not clipped} ((A_t R)_{il})^2 (LP v: (2 a v)^2 terms); used
 * as the initial inverse Hessian of the L-BFGS (h^-1, scaled), recomputed periodically */
static void precond(LR *L, const double *z, const double *y, const double *t, const double *w, double *h) {
    Problem *P = L->P; const double sg = L->sigma;
    memset(h, 0, sizeof(double) * L->N);
    for (int q = 0; q < L->ns; q++) {
        LS *s = &L->S[q]; const Block *B = &P->blk[s->k]; const int n = s->n, r = s->r; const double *R = z + s->off; double *H = h + s->off;
        pat_slack(s, w);
        for (int i = 0; i < n; i++) { const double a = s->dg[i] >= 0 ? 2.0 * fabs(s->ws[s->dg[i]]) : 0.0; for (int l = 0; l < r; l++) H[(size_t)i * r + l] = a; }
        double *acc = lz(sizeof(double) * ((size_t)n * r + 1)); int *touch = lx(sizeof(int) * (n + 1)); char *on = lz(n + 1);
        for (int tt = 0; tt < B->ncon; tt++) {
            const int i0 = B->con[tt];
            if (L->sa[i0] != 0 && clip_u(t[i0], y[i0], sg, L->sa[i0]) != t[i0]) continue;   /* clipped: no curvature */
            const SpSym *A = &B->A[tt]; int nt = 0;
            for (int e = 0; e < A->ef; e++) {
                const int i = A->fr[e], j = A->fc[e]; const double v = A->fv[e];
                if (!on[i]) { on[i] = 1; touch[nt++] = i; }
                for (int l = 0; l < r; l++) acc[(size_t)i * r + l] += v * R[(size_t)j * r + l];
            }
            for (int a = 0; a < nt; a++) { const int i = touch[a]; on[i] = 0; for (int l = 0; l < r; l++) { const double v = acc[(size_t)i * r + l]; H[(size_t)i * r + l] += 4.0 * sg * v * v; acc[(size_t)i * r + l] = 0; } }
        }
        free(acc); free(touch); free(on);
    }
    for (int q = 0; q < L->nv; q++) {
        const LV *v = &L->V[q]; double sv = v->c, gn = 0; const double x = z[v->off];
        for (int e = 0; e < v->nc; e++) { sv -= w[v->ci[e]] * v->cv[e]; const int i0 = v->ci[e]; if (L->sa[i0] == 0 || clip_u(t[i0], y[i0], sg, L->sa[i0]) == t[i0]) gn += 4.0 * v->cv[e] * v->cv[e] * x * x; }
        h[v->off] = 2.0 * fabs(sv) + 4.0 * sg * gn;
    }
    double hmax = 0; for (size_t i = 0; i < L->N; i++) hmax = fmax(hmax, h[i]);
    const double fl = 1e-10 * (hmax > 0 ? hmax : 1.0);
    for (size_t i = 0; i < L->N; i++) h[i] = 1.0 / (h[i] > fl ? h[i] : fl);   /* h holds the inverse */
}

int lralm_run(const char *fname, Params *par, const char *yfile, const BriskData *data, BriskResult *res) {
    const double t0 = wtime();
    Problem PP, *P = &PP;
    memset(P, 0, sizeof PP);
    g_read_maxn = 2000000000;   /* no n x n array on this path */
    if (data ? problem_from_sdpa_data(data, P) != 0 : read_sdpa(fname, P) != 0) return 2;
    int *bs0 = lx(sizeof(int) * (P->nblk + 1));
    for (int k = 0; k < P->nblk; k++) bs0[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    {   const char *e = getenv("BRISK_PERTURB");
        if (e && atoi(e) != 0) { const int k = atoi(e); for (int i = 0; i < P->m; i++) { unsigned h = (unsigned)(i + 1) * 2654435761u ^ (unsigned)k * 40503u; P->b[i] *= 1.0 + (((h >> 7) & 1) ? 1.0 : -1.0) * 1e-16 * k; } }
    }
    if (!P->b0) { P->b0 = lx(sizeof(double) * (P->m + 1)); memcpy(P->b0, P->b, sizeof(double) * P->m); }
    const double tread = wtime() - t0;
    lr_scale(P);
    const int m = P->m;
    LR LL, *L = &LL; memset(L, 0, sizeof LL);
    L->P = P; L->m = m; L->b = P->b;
    L->sa = lz(sizeof(double) * (m + 1)); L->sk = lx(sizeof(int) * (m + 1)); L->sj2 = lx(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) L->sk[i] = -1;
    int nsdp = 0, nlp = 0; for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) nsdp++; else nlp += P->blk[k].n; }
    L->S = lz(sizeof(LS) * (nsdp + 1)); L->V = lz(sizeof(LV) * (nlp + 1));
    size_t N = 0;
    const int r0 = par->lr_rank > 0 ? par->lr_rank : 1;
    int nslack = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        if (B->type == BLK_SDP) {
            LS *s = &L->S[L->ns++]; ls_setup(s, B, k);
            s->r = r0 < B->n ? r0 : B->n; s->rmax = par->lr_rmax > 0 ? par->lr_rmax : 32; if (s->rmax > B->n) s->rmax = B->n;
            s->off = N; N += (size_t)B->n * s->rmax;     /* room for the rank to grow */
            continue;
        }
        /* LP: per variable its constraints */
        const int n = B->n;
        int *cnt = lz(sizeof(int) * (n + 1));
        for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) cnt[B->A[t].row[q]]++;
        double *cc = lz(sizeof(double) * (n + 1));
        for (int q = 0; q < B->C.nnz; q++) cc[B->C.row[q]] += B->C.val[q];
        int **ci = lx(sizeof(int *) * (n + 1)); double **cv = lx(sizeof(double *) * (n + 1)); int *fl = lz(sizeof(int) * (n + 1));
        for (int j = 0; j < n; j++) { ci[j] = lx(sizeof(int) * (cnt[j] + 1)); cv[j] = lx(sizeof(double) * (cnt[j] + 1)); }
        for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) { const int j = B->A[t].row[q]; ci[j][fl[j]] = B->con[t]; cv[j][fl[j]] = B->A[t].val[q]; fl[j]++; }
        for (int j = 0; j < n; j++) {
            if (fl[j] == 1 && cc[j] == 0 && cv[j][0] != 0 && L->sa[ci[j][0]] == 0) {
                const int i = ci[j][0]; L->sa[i] = cv[j][0]; L->sk[i] = k; L->sj2[i] = j; nslack++;
                free(ci[j]); free(cv[j]);
            } else if (fl[j] == 0 && cc[j] == 0) { free(ci[j]); free(cv[j]); }
            else { LV *v = &L->V[L->nv++]; v->k = k; v->j = j; v->nc = fl[j]; v->ci = ci[j]; v->cv = cv[j]; v->c = cc[j]; v->off = N; N++; }
        }
        free(cnt); free(cc); free(ci); free(cv); free(fl);
    }
    L->N = N;
    L->sigma = par->lr_sigma > 0 ? par->lr_sigma : 10.0;
    const double sigma0 = L->sigma;
    const double tol = par->lr_tol > 0 ? par->lr_tol : 1e-6;
    const int verbose = par->verbose; L->verbose = verbose;
    g_lrdbg = getenv("BRISK_LRDBG") != NULL;
    if (verbose >= 0) {
        long np = 0; for (int q = 0; q < L->ns; q++) np += L->S[q].np;
        int maxn = 0; for (int q = 0; q < L->ns; q++) if (L->S[q].n > maxn) maxn = L->S[q].n;
        printf("BRISK %s low-rank augmented Lagrangian (-lralm)\n", BRISK_VERSION);
        printf("problem %s: m = %d, %d SDP blocks (max n = %d), %d LP variables (%d slacks in closed form), read %.2fs\n", fname, m, L->ns, maxn, nlp, nslack, tread);
        printf("lralm: rank %d (max %d), pattern %ld entries, sigma0 %.2g, target %.0e\n", r0, L->ns ? L->S[0].rmax : 0, np, L->sigma, tol);
    }
    /* stacked vectors: z (R row-major n x rmax per block, the first r columns used, then LP v) */
    double *z = lz(sizeof(double) * (N + 1)), *g = lz(sizeof(double) * (N + 1)), *d = lz(sizeof(double) * (N + 1)), *gn = lz(sizeof(double) * (N + 1));
    double *y = lz(sizeof(double) * (m + 1)), *w = lx(sizeof(double) * (m + 1)), *t = lx(sizeof(double) * (m + 1)), *t1 = lx(sizeof(double) * (m + 1)), *t2 = lx(sizeof(double) * (m + 1));
    /* R stored compactly per block as n x r (row-major) at s->off; the stride is r */
    {   unsigned h = 987654321u;
        for (int q = 0; q < L->ns; q++) { LS *s = &L->S[q]; for (size_t i = 0; i < (size_t)s->n * s->r; i++) { h = h * 1103515245u + 12345u; z[s->off + i] = (((h >> 8) & 0xffff) / 65535.0 - 0.5) * sqrt(12.0 / s->r); } }
        for (int q = 0; q < L->nv; q++) z[L->V[q].off] = 1.0;
        /* scale so that |A(X0)| matches |b| */
        eval_t(L, z, t);
        double ab = 0, bb = 0; for (int i = 0; i < m; i++) { const double ax = t[i] + L->b[i]; ab += ax * ax; bb += L->b[i] * L->b[i]; }
        if (ab > 0 && bb > 0) { const double f = sqrt(sqrt(bb / ab)); for (size_t i = 0; i < N; i++) z[i] *= f; }
    }
    const int mem = 10;
    double *hinv = lz(sizeof(double) * (N + 1));
    const int newton = par->lr_newton != 0, max_newton = par->lr_newton > 1 ? par->lr_newton : 60, cgmax = 200;
    const double cgtol = getenv("BRISK_LRCGTOL") ? atof(getenv("BRISK_LRCGTOL")) : 1e-3, sigfac = getenv("BRISK_LRSIGFAC") ? atof(getenv("BRISK_LRSIGFAC")) : 4.0;
    const double sredf = getenv("BRISK_LRSRED") ? atof(getenv("BRISK_LRSRED")) : 1e-3, stallsig = getenv("BRISK_LRSTALLSIG") ? atof(getenv("BRISK_LRSTALLSIG")) : 0.0;
    const int stall_solved = getenv("BRISK_LRSTALLANY") ? 0 : 1;   /* count stalls over solved subproblems only (the version that solved case200/500) */
    const double relstop = getenv("BRISK_LRREL") ? atof(getenv("BRISK_LRREL")) : 0.0;
    const double trR = par->lr_trace;
    const int bal = getenv("BRISK_LRBAL") ? atoi(getenv("BRISK_LRBAL")) : 0, bal_newton = getenv("BRISK_LRBALN") ? atoi(getenv("BRISK_LRBALN")) : 10;
    const int sigesc = getenv("BRISK_LRSIGESC") ? atoi(getenv("BRISK_LRSIGESC")) : 0;
    const int guard_y = getenv("BRISK_LRGUARD") ? atoi(getenv("BRISK_LRGUARD")) : 0;
    const int sched = getenv("BRISK_LRSCHED") ? atoi(getenv("BRISK_LRSCHED")) : 0; int nunsolved = 0, nsigstall = 0; double feas_tgt = -1, un_at_sig = 1e300;
    long ncg = 0; int nneg = 0, nrank_dn = 0; const double drop = getenv("BRISK_LRDROP") ? atof(getenv("BRISK_LRDROP")) : 0.0; const int lmshift = getenv("BRISK_LRLM") ? atoi(getenv("BRISK_LRLM")) : 1;
    double *rc = lz(sizeof(double) * (N + 1)), *zc = lz(sizeof(double) * (N + 1)), *pv = lz(sizeof(double) * (N + 1)), *Hp = lz(sizeof(double) * (N + 1)), *qb = lx(sizeof(double) * (m + 1));
    int nmax_ = 1; for (int q = 0; q < L->ns; q++) if (L->S[q].n > nmax_) nmax_ = L->S[q].n;
    int rmx_ = 1; for (int q = 0; q < L->ns; q++) if (L->S[q].rmax > rmx_) rmx_ = L->S[q].rmax;
    double *col = lx(sizeof(double) * (nmax_ + 1)), *wk = lx(sizeof(double) * (2 * (size_t)nmax_ * rmx_ + 2));
    const int use_prec = par->lr_prec != 0;
    /* L-BFGS memory over the full stacked length (rank growth keeps offsets) */
    double *Sm = lz(sizeof(double) * N * mem), *Ym = lz(sizeof(double) * N * mem), rhoM[16], alM[16];
    long ninner = 0; int nev = 0, nrank = 0, outer, status = ST_MAXIT;
    double eta = 1e-1, unorm_prev = 1e300, un_best = 1e300; int nstall = 0;
    double best_lb = -1e300, best_lb_t = 0, *ybest = lz(sizeof(double) * (m + 1)); int best_lb_outer = 0, have_best = 0, nbtest = 0, nbok = 0, bound_ok_now = 0;
    LMeas M; memset(&M, 0, sizeof M);
    const int max_outer = par->lr_outer > 0 ? par->lr_outer : 500, max_inner = par->lr_inner > 0 ? par->lr_inner : 2000;
    double cobj = eval_t(L, z, t);
    for (outer = 1; outer <= max_outer; outer++) {
        /* ---- inner: L-BFGS with the exact line search ---- */
        /* orthogonal columns (R <- R Q, Q the eigenvectors of R'R; X unchanged): the
         * preconditioner treats the columns separately */
        for (int q = 0; q < L->ns; q++) {
            LS *s = &L->S[q]; const int r = s->r, n = s->n; if (r < 2) continue;
            double *R = z + s->off, G2[32 * 32], ev2[32], wk2[32 * 34]; int info = 0, lw = 32 * 34;
            for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) { double v = 0; for (int i = 0; i < n; i++) v += R[(size_t)i * r + a] * R[(size_t)i * r + b]; G2[a + b * r] = v; }
            BL(dsyev_)("V", "L", &r, G2, &r, ev2, wk2, &lw, &info);
            if (info == 0) for (int i = 0; i < n; i++) { double row[32]; for (int b = 0; b < r; b++) { double v = 0; for (int a = 0; a < r; a++) v += R[(size_t)i * r + a] * G2[a + (r - 1 - b) * r]; row[b] = v; } memcpy(R + (size_t)i * r, row, sizeof(double) * r); }
            /* rank reduction: columns (eigenvalues of X) below lr_drop of the largest are
             * dropped - an excess rank makes the minimizer degenerate (the vanishing
             * columns converge like Newton on x^4, linearly) */
            if (info == 0 && drop > 0) {
                int keep = r; while (keep > 1 && ev2[r - keep] < drop * ev2[r - 1]) keep--;
                if (keep < r) {
                    for (int i = 0; i < n; i++) for (int l = 0; l < keep; l++) R[(size_t)i * keep + l] = R[(size_t)i * r + l];
                    memset(R + (size_t)n * keep, 0, sizeof(double) * (size_t)n * (r - keep));
                    if (verbose > 0) printf("   (rank reduction in block %d: %d -> %d)\n", s->k + 1, r, keep);
                    s->r = keep; nrank_dn++;
                }
            }
        }
        double f = cobj + pen_sum(L, t, y, w);
        grad(L, z, w, g);
        if (use_prec && !newton) precond(L, z, y, t, w, hinv);
        double gnorm = nrm(g, N);
        int nm = 0, head = 0, inner;
        const double gstart = gnorm; const long ncg0 = ncg; const int nneg0 = nneg;
        if (newton) {
            nprec_build(L, z, y, t);
            const int nmax_ = bal ? bal_newton : max_newton;
            for (inner = 0; inner < nmax_ && gnorm > eta && !(relstop > 0 && gnorm <= relstop * gstart) && !(bal && gnorm <= 0.1 * gstart); inner++) {
                if (brisk_time_up(par)) break;
                /* PCG on H d = -g */
                memset(d, 0, sizeof(double) * N);
                for (size_t i = 0; i < N; i++) rc[i] = -g[i];
                nprec_apply(L, z, y, t, w, rc, zc, col, wk);
                memcpy(pv, zc, sizeof(double) * N);
                double rz = dotv(rc, zc, N); const double rz0 = rz; int k;
                for (k = 0; k < cgmax; k++) {
                    hess_vec(L, z, y, t, w, pv, Hp, qb);
                    /* the shifted (Levenberg-Marquardt) system H + delta I, delta the shift that
                     * made the preconditioner positive definite: S(w) is indefinite until the
                     * multipliers are near optimal */
                    if (lmshift) for (int q = 0; q < L->ns; q++) { const LS *s = &L->S[q]; const double dl = s->fabs_; const size_t nn = (size_t)s->n * s->r; for (size_t i = 0; i < nn; i++) Hp[s->off + i] += dl * pv[s->off + i]; }
                    const double pHp = dotv(pv, Hp, N);
                    if (!(pHp > 0)) {
                        /* negative curvature: the direction itself (the exact line search on
                         * the quartic follows it), sign for descent */
                        if (k == 0 || !lmshift) { const double gp = dotv(g, pv, N); for (size_t i = 0; i < N; i++) d[i] = gp > 0 ? -pv[i] : pv[i]; }
                        nneg++; break; }
                    const double al = rz / pHp;
                    for (size_t i = 0; i < N; i++) { d[i] += al * pv[i]; rc[i] -= al * Hp[i]; }
                    nprec_apply(L, z, y, t, w, rc, zc, col, wk);
                    const double rzn = dotv(rc, zc, N);
                    if (rzn <= cgtol * cgtol * rz0) { k++; break; }
                    const double be = rzn / rz; rz = rzn;
                    for (size_t i = 0; i < N; i++) pv[i] = zc[i] + be * pv[i];
                }
                ncg += k;
                double gd = dotv(g, d, N);
                if (!(gd < 0)) { nprec_apply(L, z, y, t, w, g, d, col, wk); for (size_t i = 0; i < N; i++) d[i] = -d[i]; gd = dotv(g, d, N); }
                double c1, c2;
                eval_dir(L, z, d, t1, t2, &c1, &c2);
                const double a = lsearch(L, t, t1, t2, y, c1, c2, gd, &nev);
                for (size_t i = 0; i < N; i++) z[i] += a * d[i];
                for (int i = 0; i < m; i++) t[i] += a * (t1[i] + a * t2[i]);
                cobj += a * (c1 + a * c2);
                if ((inner & 7) == 7) cobj = eval_t(L, z, t);
                pen_sum(L, t, y, w);
                grad(L, z, w, g);
                gnorm = nrm(g, N);
                ninner++;
                if (g_lrdbg) {
                    static char *msk = NULL; static int mm = 0; if (!msk) { msk = calloc(m + 1, 1); mm = m; }
                    int flips = 0, act = 0;
                    for (int i = 0; i < mm; i++) { const char c = L->sa[i] == 0 || clip_u(t[i], y[i], L->sigma, L->sa[i]) == t[i]; if (c != msk[i]) flips++; msk[i] = c; act += c; }
                    printf("      newton %d: step %.3e |d| %.2e gd %.2e cg %d |g| %.3e  active %d flips %d\n", inner, a, nrm(d, N), gd, k, gnorm, act, flips);
                }
                if (gnorm > eta) nprec_build(L, z, y, t);   /* cheap next to the CG it saves (the factor is n-sized) */
            }
        } else
        for (inner = 0; inner < max_inner && gnorm > eta; inner++) {
            if ((inner & 15) == 15 && brisk_time_up(par)) break;
            memcpy(d, g, sizeof(double) * N);
            for (int j = 0; j < nm; j++) { const int ix = (head - 1 - j + mem) % mem; alM[ix] = rhoM[ix] * dotv(Sm + (size_t)ix * N, d, N); const double *yj = Ym + (size_t)ix * N; for (size_t i = 0; i < N; i++) d[i] -= alM[ix] * yj[i]; }
            if (use_prec) {
                double gam = 1.0;
                if (nm > 0) { const int ix = (head - 1 + mem) % mem; const double *yj = Ym + (size_t)ix * N; double yhy = 0; for (size_t i = 0; i < N; i++) yhy += yj[i] * hinv[i] * yj[i]; gam = dotv(Sm + (size_t)ix * N, yj, N) / fmax(yhy, 1e-300); }
                for (size_t i = 0; i < N; i++) d[i] *= gam * hinv[i];
            } else if (nm > 0) { const int ix = (head - 1 + mem) % mem; const double sy = dotv(Sm + (size_t)ix * N, Ym + (size_t)ix * N, N), yy = dotv(Ym + (size_t)ix * N, Ym + (size_t)ix * N, N); const double gam = sy / fmax(yy, 1e-300); for (size_t i = 0; i < N; i++) d[i] *= gam; }
            for (int j = nm - 1; j >= 0; j--) { const int ix = (head - 1 - j + mem) % mem; const double be = rhoM[ix] * dotv(Ym + (size_t)ix * N, d, N); const double *sj = Sm + (size_t)ix * N; for (size_t i = 0; i < N; i++) d[i] += (alM[ix] - be) * sj[i]; }
            for (size_t i = 0; i < N; i++) d[i] = -d[i];
            double gd = dotv(g, d, N);
            if (!(gd < 0)) { for (size_t i = 0; i < N; i++) d[i] = -g[i]; gd = -gnorm * gnorm; nm = 0; }
            double c1, c2;
            eval_dir(L, z, d, t1, t2, &c1, &c2);
            const double a = lsearch(L, t, t1, t2, y, c1, c2, gd, &nev);
            for (size_t i = 0; i < N; i++) z[i] += a * d[i];
            for (int i = 0; i < m; i++) t[i] += a * (t1[i] + a * t2[i]);
            cobj += a * (c1 + a * c2);
            if ((inner & 31) == 31) cobj = eval_t(L, z, t);       /* against drift */
            const double fn = cobj + pen_sum(L, t, y, w);
            grad(L, z, w, gn);
            /* memory */
            {   double *sj = Sm + (size_t)head * N, *yj = Ym + (size_t)head * N; double sy = 0;
                for (size_t i = 0; i < N; i++) { sj[i] = a * d[i]; yj[i] = gn[i] - g[i]; sy += sj[i] * yj[i]; }
                if (sy > 1e-16 * nrm(sj, N) * nrm(yj, N)) { rhoM[head] = 1.0 / sy; head = (head + 1) % mem; if (nm < mem) nm++; }
            }
            memcpy(g, gn, sizeof(double) * N); f = fn;
            gnorm = nrm(g, N);
            ninner++;
            if (use_prec && (inner % 100) == 99) { precond(L, z, y, t, w, hinv); nm = 0; }
        }
        (void)f;
        /* ---- multiplier step ---- */
        cobj = eval_t(L, z, t);
        pen_sum(L, t, y, w);
        double un = 0; for (int i = 0; i < m; i++) { const double u = clip_u(t[i], y[i], L->sigma, L->sa[i]); un += u * u; } un = sqrt(un);
        int sched_sig = 0; (void)sched_sig;
        if (sched) {
            /* 4.40 (LANCELOT-type): the multipliers move only after a solved subproblem whose
             * residual met the feasibility target (then the target tightens); a solved one that
             * missed it raises sigma and keeps y; an unsolved one changes nothing (up to 3 in a
             * row) */
            const int solved = gnorm <= eta || nunsolved >= 3;
            if (!solved) nunsolved++;
            else {
                nunsolved = 0;
                if (feas_tgt < 0) feas_tgt = 2.0 * un;
                if (un <= feas_tgt) { memcpy(y, w, sizeof(double) * m); feas_tgt = fmax(0.25 * un, 1e-3 * tol); eta = fmax(0.2 * eta, 0.1 * tol); }
                else {
                    /* the residual of a solved subproblem not falling under a larger penalty: an
                     * infeasible stationary point of the factorized problem - escape (below) */
                    if (un > 0.5 * un_at_sig) nsigstall++; else nsigstall = 0;
                    un_at_sig = un;
                    L->sigma = fmin(L->sigma * sigfac, 1e12); sched_sig = 1;
                }
            }
        } else if (!guard_y || gnorm <= eta || gnorm <= 1e-3 * gstart || un < unorm_prev)
            memcpy(y, w, sizeof(double) * m);   /* (guarded: not after a subproblem that is neither solved nor more feasible) */
        /* ---- measures (the ladder when the cheap ones are near the target) ---- */
        measure(L, z, y, t, 0, &M);
        {   /* 4.40: the best certified lower bound: b'y with Z positive definite on every
             * block (sparse Cholesky) and the LP duals >= 0 (the slack duals by the clipping) */
            int ok = 1; double dl = 0;   /* lambda_min(Z) >= -dl (scaled) */
            for (int q = 0; q < L->ns && ok; q++) {
                LS *s = &L->S[q]; pat_slack(s, y);
                if (z_posdef(s)) continue;
                if (trR > 0 && (outer % 5 == 0 || fmax(M.pinf, fabs(M.gap)) <= 1e-3)) {
                    double dmax = 0; for (int i = 0; i < s->n; i++) if (s->dg[i] >= 0) dmax = fmax(dmax, fabs(s->ws[s->dg[i]]));
                    const double d = ladder(s, dmax > 0 ? dmax : 1.0);
                    if (d >= 0) { dl = fmax(dl, d); continue; }
                }
                ok = 0;
            }
            for (int q = 0; q < L->nv && ok; q++) { const LV *v = &L->V[q]; double zv = v->c; for (int e = 0; e < v->nc; e++) zv -= y[v->ci[e]] * v->cv[e]; if (zv < 0) ok = 0; }
            nbtest++;
            /* with tr X <= R (-lrtrace, the problem as read): <C,X> = b'y + <Z,X> >= b'y - cs dl R */
            const double lb = M.dobj - P->cs * dl * (trR > 0 ? trR : 0.0);
            if (ok) { nbok++; if (lb > best_lb) { best_lb = lb; best_lb_outer = outer; best_lb_t = wtime() - t0; memcpy(ybest, y, sizeof(double) * m); have_best = 1; } }
            bound_ok_now = ok;
        }
        double lz_min = 0;
        int full = 0;
        int escaped = 0;
        const int solved_ = gnorm <= eta || gnorm <= 1e-6 * gstart;
        /* a stall of the residual over solved subproblems at this rank: an infeasible
         * stationary point of the factorized problem (rank 1 is the nonconvex QCQP itself);
         * escape along the most negative eigenvector of Z as at a near-KKT point */
        if (solved_ || !stall_solved) { if (un > 0.75 * un_best) nstall++; else nstall = 0; if (un < un_best) un_best = un; }
        const int near_kkt = (solved_ && fmax(M.pinf, fmax(fabs(M.gap), fabs(M.compl))) <= 10 * tol) || (nstall >= 3 && L->sigma >= stallsig) || ((sched || sigesc) && nsigstall >= 2);
        if ((sched || sigesc) && nsigstall >= 2) { /* Z of the current multiplier estimate: the escape direction of the stalled point */ memcpy(y, w, sizeof(double) * m); nsigstall = 0; un_at_sig = 1e300; feas_tgt = -1; }
        const int bal_esc = bal && (gnorm <= 0.1 * gstart || gnorm <= eta);
        if (fmax(M.pinf, fmax(fabs(M.gap), fabs(M.compl))) <= 10 * tol || outer % 10 == 0 || near_kkt || bal) {
            /* Lanczos estimate of lambda_min(Z) per block; rank escape when clearly negative */
            for (int q = 0; q < L->ns; q++) {
                LS *s = &L->S[q]; pat_slack(s, y);
                double *v = lx(sizeof(double) * s->n);
                const double th = lanczos_min(s, 60, v, (unsigned)outer);
                if (th < lz_min) lz_min = th;
                if ((near_kkt || (bal_esc && th < -fmax(tol, 0.1 * M.pinf))) && th < -tol && s->r < s->rmax && par->lr_escape != 0) {
                    /* grow the rank in place: X + alpha v v' with alpha from the exact line
                     * search (the residuals are linear in alpha), the new column sqrt(alpha) v */
                    const int r = s->r, n = s->n; double *R = z + s->off;
                    for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) s->xs[p] = v[i] * v[s->sj[p]];
                    memset(t1, 0, sizeof(double) * m); memset(t2, 0, sizeof(double) * m);
                    const double c1 = pat_apply(s, t1);
                    double d0 = c1; for (int i = 0; i < m; i++) { const double u = clip_u(t[i], y[i], L->sigma, L->sa[i]); d0 -= (y[i] - L->sigma * u) * t1[i]; }
                    double al = 0;
                    if (d0 < 0) al = lsearch(L, t, t1, t2, y, c1, 0.0, d0, &nev);
                    const double be = al > 0 ? sqrt(al) : 1e-3;
                    for (int i = n - 1; i >= 0; i--) { for (int l = r - 1; l >= 0; l--) R[(size_t)i * (r + 1) + l] = R[(size_t)i * r + l]; R[(size_t)i * (r + 1) + r] = be * v[i]; }
                    s->r = r + 1; nrank++; escaped = 1; nstall = 0; un_best = 1e300;
                    if (verbose > 0) printf("   (rank escape in block %d: lambda_min(Z) %.2e, step %.2e, rank %d)\n", s->k + 1, th, al, s->r);
                }
                free(v);
            }
            M.dinf = P->cs * fmax(0.0, -lz_min) / (1 + P->normC2);
            if (fmax(M.pinf, fmax(fabs(M.gap), fabs(M.compl))) <= tol && M.dinf <= 10 * tol) { measure(L, z, y, t, 1, &M); full = 1; }
            cobj = eval_t(L, z, t);
        }
        if (verbose > 0) printf("LR%4d %16.9e %16.9e  pinf %8.1e gap %9.1e compl %9.1e dinf %8.1e%s  sigma %7.1e  inner %5d (cg %ld, neg %d, shift %.0e) |g| %7.1e  rank %d  %7.1fs  LB %s%.9e\n",
                                outer, M.pobj, M.dobj, M.pinf, M.gap, M.compl, M.dinf, full ? "*" : (lz_min < 0 || outer % 10 == 0 || fmax(M.pinf, fabs(M.gap)) <= 10 * tol ? "~" : " "),
                                L->sigma, inner, ncg - ncg0, nneg - nneg0, L->ns ? L->S[0].fshift : 0.0, gnorm, L->ns ? L->S[0].r : 0, wtime() - t0, bound_ok_now ? "+" : " ", have_best ? best_lb : 0.0), fflush(stdout);
        if (full && fmax(fmax(M.pinf, M.dinf), fmax(fabs(M.gap), fabs(M.compl))) <= tol) { status = ST_OPTIMAL; break; }
        if (escaped && !bal) {
            /* a new face: the penalty that held the old one (often 1e8) would freeze the
             * iterate there; restart it moderate (BRISK_LRESIG: factor, 1e-3) */
            const double f2 = getenv("BRISK_LRESIG") ? atof(getenv("BRISK_LRESIG")) : 1e-3;
            L->sigma = fmax(sigma0, L->sigma * f2); unorm_prev = 1e300; eta = fmax(eta, 1e-3); feas_tgt = -1; un_at_sig = 1e300;
            cobj = eval_t(L, z, t);
            continue;
        }
        if (brisk_time_up(par)) { status = ST_TIME; break; }
        if (bal) {
            /* 4.40 (BRISK_LRBAL): the penalty balances the primal residual against the dual
             * one (lambda_min(Z) from Lanczos) - up x2 when the primal side lags 3x, down /2 when
             * the dual side does; the multipliers move every outer iteration */
            const double pr = M.pinf, du = fmax(M.dinf, 1e-16);
            if (pr > 3 * du) L->sigma = fmin(L->sigma * 2.0, 1e10);
            else if (du > 3 * pr) L->sigma = fmax(L->sigma * 0.5, 1e-2);
            eta = fmax(0.5 * eta, 0.1 * tol);
        } else
        if (!sched) {   /* penalty: x sigfac when |u| did not fall below 1/4 (the L-BFGS inner solver: only
             * after a solved subproblem); inner tolerance tied to the residual */
            const int solved = gnorm <= eta || gnorm <= sredf * gstart || (relstop > 0 && gnorm <= relstop * gstart);
            if (solved && un > 0.25 * unorm_prev) {
                /* the residual of solved subproblems not halving over two penalty increases: an
                 * infeasible stationary point of the factorized problem (BRISK_LRSIGESC) */
                if (sigesc) { if (un > 0.5 * un_at_sig) nsigstall++; else nsigstall = 0; un_at_sig = un; }
                L->sigma = fmin(L->sigma * sigfac, 1e12);
            }
            if (solved || un < unorm_prev) unorm_prev = un;
            eta = fmax(0.2 * eta, 0.1 * tol);
        }
        /* the line search and the L-BFGS pairs belong to the old penalty: memory restarts */
    }
    if (outer > max_outer) outer = max_outer;
    /* final measure with the ladder */
    cobj = eval_t(L, z, t);
    measure(L, z, y, t, 1, &M);
    const double acc = fmax(fmax(M.pinf, M.dinf), fmax(fabs(M.gap), fabs(M.compl)));
    int st = acc <= tol ? ST_OPTIMAL : (acc <= par->red_acc ? ST_REDUCED : (status == ST_TIME ? ST_TIME : ST_MAXIT));
    if (verbose >= 0) {
        const char *sn = st == ST_OPTIMAL ? "OPTIMAL" : st == ST_REDUCED ? "SOLVED TO REDUCED ACCURACY" : st == ST_TIME ? "TIME LIMIT" : "ITERATION LIMIT";
        printf("lralm: %d outer, %ld %s steps (%ld CG, %d negative curvature), %d line-search evaluations, %d rank increases, final rank", outer, ninner, newton ? "Newton" : "L-BFGS", ncg, nneg, nev, nrank);
        for (int q = 0; q < L->ns; q++) printf(" %d", L->S[q].r);
        printf("%s\n", M.lmin_ok ? "" : " (lambda_min(Z) not certified: the pattern did not factor)");
        printf("status: %s (max rel error %.1e)   iterations: %d\n", sn, acc, outer);
        printf("optimal value (SDPA/SDPLIB convention, max <F0,Y>): %.10e\n", -M.pobj);
        printf("  primal obj <C,X> = %.10e   dual obj b'y = %.10e\n", M.pobj, M.dobj);
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n", M.err[1], M.err[2], M.err[3], M.err[4], M.err[5], M.err[6]);
        printf("  (X = R R' is positive semidefinite by construction; err4 from a sparse Cholesky ladder of Z = C - A'y)\n");
        if (have_best) printf("lower bound (b'y%s, Z = C - A'y by sparse Cholesky, LP duals >= 0): %.10e, outer %d at %.1fs; %d of %d iterates passed\n", trR > 0 ? " - lambda_min(Z)- tr-bound" : " with Z positive definite", best_lb, best_lb_outer, best_lb_t, nbok, nbtest);
        else printf("lower bound: no iterate had a positive definite Z\n");
        printf("time: total %.3fs (read %.3f)\n", wtime() - t0, tread);
    }
    if (res) {
        /* the library interface: status, objectives, errors and y of the problem as read; no X
         * (X = R R' is not formed) */
        static const int ecode[7] = { 0, 11, 12, 13, 14, 10, 15 };
        res->status = st; res->exit_code = ecode[st];
        snprintf(res->status_str, sizeof(res->status_str), "%s", st == ST_OPTIMAL ? "OPTIMAL" : st == ST_REDUCED ? "SOLVED TO REDUCED ACCURACY" : st == ST_TIME ? "TIME LIMIT" : "ITERATION LIMIT");
        res->iters = outer; res->pobj = M.pobj; res->dobj = M.dobj;
        for (int e = 1; e <= 6; e++) res->err[e] = M.err[e];
        res->time = wtime() - t0; res->m = m; res->nblk = P->nblk;
        res->bs = malloc(sizeof(int) * (P->nblk + 1)); memcpy(res->bs, bs0, sizeof(int) * P->nblk);
        res->y = malloc(sizeof(double) * (m + 1));
        for (int i = 0; i < m; i++) res->y[i] = P->cs * P->d[i] * y[i];
        res->have_x = 0; res->X = NULL; res->Z = NULL;
        res->bound_side = 0; res->bound_rigorous = NAN;
        snprintf(res->cause, sizeof(res->cause), "%s", st == ST_OPTIMAL ? "" : "low-rank ALM (-lralm, experimental): the outer iteration did not reach the target");
    }
    free(bs0);
    if (yfile) {
        FILE *f = fopen(yfile, "w");
        if (f) { for (int i = 0; i < m; i++) fprintf(f, "%.17g\n", P->cs * P->d[i] * y[i]); fclose(f); }
    }
    for (int q = 0; q < L->ns; q++) ls_free(&L->S[q]);
    for (int q = 0; q < L->nv; q++) { free(L->V[q].ci); free(L->V[q].cv); }
    free(L->S); free(L->V); free(L->sa); free(L->sk); free(L->sj2);
    free(z); free(g); free(d); free(gn); free(y); free(w); free(t); free(t1); free(t2); free(Sm); free(Ym); free(hinv); free(ybest); free(rc); free(zc); free(pv); free(Hp); free(qb); free(col); free(wk);
    problem_free(P);
    return st == ST_OPTIMAL ? 0 : st == ST_REDUCED ? 10 : st == ST_TIME ? 15 : 13;
}
