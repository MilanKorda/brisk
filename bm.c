/* bm.c - 5.14: a Burer-Monteiro solver for large sparse SDPs with low-rank solutions (`-bm 1`).
 *
 *   min <C,X> s.t. <A_i,X> = b_i, X = R R' (one or more semidefinite blocks, R n x r per block)
 *
 * Nothing of size n x n is formed: the data live on the union of their sparsity patterns and every
 * operation is a sparse product with the n x r factor.
 *
 * Two methods, chosen from the data:
 *  - Fixed diagonal (every constraint is a_p X_pp = b_p, as in max-cut and the other relaxations of
 *    problems over {-1,1}^n): the constraints hold exactly on the product of spheres
 *    |R_p|^2 = b_p / a_p, and <C, R R'> is minimized there by block-coordinate steps (each row in turn
 *    moves to the minimizer of the objective over its sphere, R_p = -|R_p| g_p / |g_p| with
 *    g_p = sum_{q != p} C_pq R_q). Rows of the same colour of a greedy colouring of C's graph do not
 *    interact, so each colour is one parallel step (the same result for any thread count).
 *  - General constraints: the augmented Lagrangian
 *        L(R; y, s) = <C, RR'> - y'(A(RR') - b) + s/2 |A(RR') - b|^2,
 *    minimized over R by L-BFGS with an exact line search (along R + a D every residual is a quadratic
 *    in a, so L is a quartic in a and its minimizer is a root of a cubic), y <- y - s (A(RR') - b), s
 *    raised when the residual does not fall fast enough.
 * Both end with a certificate: y from the stationarity conditions (fixed diagonal) or the
 * multipliers, Z = C - A*(y) on the pattern, lambda_min(Z) by Lanczos; a clearly negative eigenvalue
 * adds a column along its eigenvector (the rank grows, as it must when the factorization is too
 * small) and the method continues. The reported errors are the three of the low-rank literature
 * (|A(X) - b|_2 / (1 + |b|_1), max(0, -lambda_min(Z)) / (1 + |C|_1), the relative gap), measured on the
 * problem as read.                                                                               */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void *bx(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (Burer-Monteiro, %.2f GB)\n", n / 1e9); exit(1); } return p; }
static void *bz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (Burer-Monteiro, %.2f GB)\n", n / 1e9); exit(1); } return p; }

static double g_bm_sweeptol = 1e-4;   /* BRISK_BMSWEEPTOL: the sweeps stop when the objective's relative change falls below this */
static double g_bm_omega = 1.9;       /* BRISK_BMOMEGA: over-relaxation of the sweeps */
static int g_bm_newton = 1;           /* BRISK_BMNEWTON=0: no trust region after the sweeps */
static int g_bm_cert = 1;             /* BRISK_BMCERT=0: the fixed-diagonal path stops on the DIMACS errors alone (not the certified gap) */
static double g_bm_sigma0 = 0;        /* BRISK_BMSIGMA: the first penalty of the augmented Lagrangian (0: 1/sqrt(n)) */
static double g_bm_sigfac = 2.0;      /* BRISK_BMSIGFAC: its growth when the residual does not fall to a quarter */
static int g_bm_prec = 0;             /* BRISK_BMPREC=1: a sparse Cholesky preconditioner in the trust region (slower: IMPLEMENTATION 15.4) */
static double g_bm_pmu = 1.0;         /* BRISK_BMPMU: the weight of R's columns in that preconditioner, relative to the mean |Z_pp| (0: none) */
static double g_bm_pdelta = 1e-3;     /* BRISK_BMPDELTA: its shift, relative to the mean |Z_pp| */
static int g_bm_escp = 8;             /* BRISK_BMESCP: the columns an escape adds at most (Ritz vectors of the most negative eigenvalues of Z) */
static int g_bm_escsweep = 0;         /* BRISK_BMESCSWEEP=1: sweeps after an escape (before the multi-column escape: always) */
static int g_bm_warm = 1;             /* BRISK_BMWARM=0: each certificate's Lanczos from a random start */
static double g_bm_rndfac = 1e-1;     /* BRISK_BMRNDFAC: a trust-region round cuts |Z R| by this */
static int g_bm_verify = 1;           /* BRISK_BMVERIFY=0: the bound not checked by a factorization */
static int g_bm_lzk = 100;            /* BRISK_BMLZK: Lanczos steps a cycle (at most; 150 before: 100 was faster on 9 of 9 graphs, 60 lost on a random one) */
static int g_bm_fuse = 1;             /* BRISK_BMFUSE=0: the trust region's CG in separate passes (as before the fused kernels) */
static int g_bm_stall_stop = 0;    /* the automatic attempt: a stall of the primal error ends the augmented Lagrangian path (gen_alm) */

typedef struct {
    int k, n, r;
    int np, *sp, *sj;            /* union of the full patterns of C and the A_i (CSR, sorted, diagonal included) */
    double *cv;                  /* C on the pattern */
    double *ws;                  /* S = C - A*(w) on the pattern */
    double *xs;                  /* a pattern buffer */
    int ncon; const int *con;
    int *cp, *ep; double *ev;    /* constraint t: its full entries as pattern positions and values */
    double *R;                   /* n x r, row-major */
    /* fixed diagonal */
    double *rho;                 /* |R_p|^2 */
    double *adiag;               /* a_p of the row's constraint */
    int *crow;                   /* the row's constraint */
    int ncol, *cptr, *cidx;      /* colouring: the rows of colour c are cidx[cptr[c] .. cptr[c+1]) */
    int vwarm;                   /* the Lanczos vector of the last certificate is a start */
    SChol *Fv; int vstate;       /* the certificate's check: a sparse Cholesky of Z + delta I (vstate 0 untried, 1 on, -1 off) */
    SChol *F; int fstate;        /* preconditioner: a sparse Cholesky of Z + delta I on the pattern (fstate 0 untried, 1 on, -1 off) */
    double *pw, *pb;             /* its work arrays (n x r, column-major) */
    double *pW, *pS; double pmu; /* the deflation of R's columns: W = (Z + delta I)^-1 R, the factor of R'R/pmu + R'W */
    /* 13 November 2026: one constraint a <J, X> = 0 besides the fixed diagonal (J = e e', max-bisection; Mittelmann's
     * sparse benchmark: G40_mb, G60_mb, rendl1, foot, biomedP), by the exact penalty <C + jP J, X>: J R = e (e'R), a
     * rank-one term in every product with C (jsum); jP = 0 when absent */
    double jP, jcoef; int jcon, jt;
    int jlast; double jb;        /* the lift of a <J, X> = b > 0: the last row is the added one (weight -1 in w = (e, -1), |R_n|^2 = b / a) */
    /* 13 November 2026 (the owner: Burer-Monteiro with a few equality constraints besides the fixed diagonal): kx extra
     * rows <A_j, X> = b_j by an augmented Lagrangian, the diagonal kept exact on the product of spheres. cv holds the
     * cost of the current linearization, cv0 - sum w_j A_j (w_j = lambda_j - sigma r_j at the current R), and jP the
     * rank-one part of a J-type row (-w_J a); every product with C then serves the Lagrangian as it stands */
    int kx;                      /* 0: none */
    int *xt, *xcon, *xn;         /* the block's constraint index, the problem's, the count of full entries */
    int **xa, **xb; double **xv; /* sparse rows: full entries (a, b, v) */
    int *xJ; double *xcoef;      /* J-type row (a <J, X>, kept out of the pattern): its value a */
    double *xrhs, *xlam, *xres, *xw, xsig;
    double *cv0;
} BB;
static int g_bb_skip = -1;   /* bb_setup leaves this constraint (the a J row) out of the pattern */
static int g_bm_jrow = 1;   /* BRISK_BMJROW=0: a J row is not taken (the problem goes to the interior-point method, as before) */
static double jsq(const BB *s, const double *U, int r);
static void jsum(const BB *s, const double *U, int r, double *u);
/* the extra rows' residuals r_j = <A_j, U U'> - b_j */
static void x_resid(const BB *s, const double *U, int r) {
    for (int j = 0; j < s->kx; j++) {
        double v = 0;
        if (s->xJ[j]) v = s->xcoef[j] * jsq(s, U, r);
        else for (int e = 0; e < s->xn[j]; e++) { const double *ua = U + (size_t)s->xa[j][e] * r, *ub = U + (size_t)s->xb[j][e] * r; double d = 0; for (int l = 0; l < r; l++) d += ua[l] * ub[l]; v += s->xv[j][e] * d; }
        s->xres[j] = v - s->xrhs[j];
    }
}
/* the linearization at U: w = lambda - sigma r, cv = cv0 - sum w_j A_j (sparse rows), jP = -a w_J (a J row) */
static void x_refresh(BB *s, const double *U, int r) {
    if (!s->kx) return;
    x_resid(s, U, r);
    memcpy(s->cv, s->cv0, sizeof(double) * s->np);
    s->jP = 0;
    for (int j = 0; j < s->kx; j++) {
        const double w = s->xlam[j] - s->xsig * s->xres[j]; s->xw[j] = w;
        if (s->xJ[j]) { s->jP += -s->xcoef[j] * w; continue; }
        const int t = s->xt[j];
        for (int e = s->cp[t]; e < s->cp[t + 1]; e++) s->cv[s->ep[e]] -= w * s->ev[e];
    }
}
/* the augmented Lagrangian at U: <C, UU'> - lambda'r + sigma/2 |r|^2 */
static double x_lag(BB *s, const double *U, int r) {
    const int n = s->n; double f = 0;
    for (int i = 0; i < n; i++) {
        const double *ui = U + (size_t)i * r;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const double c = s->cv0[p]; if (c == 0) continue; const double *uj = U + (size_t)s->sj[p] * r; double d = 0; for (int l = 0; l < r; l++) d += ui[l] * uj[l]; f += c * d; }
    }
    x_resid(s, U, r);
    for (int j = 0; j < s->kx; j++) f += -s->xlam[j] * s->xres[j] + 0.5 * s->xsig * s->xres[j] * s->xres[j];
    return f;
}
/* Q += 2 sigma sum_j <A_j R, P> A_j R (the penalty's second-order term) */
static void x_hess_add(const BB *s, const double *R, const double *Pd, double *Q) {
    const int n = s->n, r = s->r;
    double *u = bx(sizeof(double) * r), *sp = bx(sizeof(double) * r);
    for (int j = 0; j < s->kx; j++) {
        if (s->xJ[j]) {
            jsum(s, R, r, u); jsum(s, Pd, r, sp);
            double d = 0; for (int l = 0; l < r; l++) d += u[l] * sp[l];
            const double c = 2 * s->xsig * s->xcoef[j] * s->xcoef[j] * d;
            for (int i = 0; i < n; i++) { double *qi = Q + (size_t)i * r; for (int l = 0; l < r; l++) qi[l] += c * u[l]; }
            continue;
        }
        double d = 0;
        for (int e = 0; e < s->xn[j]; e++) { const double *rb = R + (size_t)s->xb[j][e] * r, *pa = Pd + (size_t)s->xa[j][e] * r; double q = 0; for (int l = 0; l < r; l++) q += rb[l] * pa[l]; d += s->xv[j][e] * q; }
        const double c = 2 * s->xsig * d;
        for (int e = 0; e < s->xn[j]; e++) { const double *rb = R + (size_t)s->xb[j][e] * r; double *qa = Q + (size_t)s->xa[j][e] * r; const double f = c * s->xv[j][e]; for (int l = 0; l < r; l++) qa[l] += f * rb[l]; }
    }
    free(u); free(sp);
}

/* u = e'U (the column sums of an n x r factor, added in row order: the same bits for any thread count) */
#define JW(s, i) (((s)->jlast && (i) == (s)->n - 1) ? -1.0 : 1.0)   /* the rank-one row's weights: e, or (e, -1) when lifted */
static void jsum(const BB *s, const double *U, int r, double *u) {
    for (int l = 0; l < r; l++) u[l] = 0;
    for (int i = 0; i < s->n; i++) { const double *ui = U + (size_t)i * r; const double wi = JW(s, i); for (int l = 0; l < r; l++) u[l] += wi * ui[l]; }
}
static double jsq(const BB *s, const double *U, int r) {   /* |e'U|^2 */
    double u[512], *up = r <= 512 ? u : bx(sizeof(double) * r), v = 0;
    jsum(s, U, r, up); for (int l = 0; l < r; l++) v += up[l] * up[l];
    if (up != u) free(up);
    return v;
}

static int cmp2(const void *a, const void *b) { const int *x = a, *y = b; return x[0] != y[0] ? x[0] - y[0] : x[1] - y[1]; }
static int ppos(const BB *s, int i, int j) {
    int lo = s->sp[i], hi = s->sp[i + 1] - 1;
    while (lo <= hi) { const int mid = (lo + hi) >> 1; if (s->sj[mid] == j) return mid; if (s->sj[mid] < j) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

static void bb_setup(BB *s, const Block *B, int k) {
    const int n = B->n;
    s->k = k; s->n = n; s->ncon = B->ncon; s->con = B->con;
    size_t tot = B->C.ef + (size_t)n; for (int t = 0; t < B->ncon; t++) tot += B->A[t].ef;
    int (*pr)[2] = bx(sizeof(int[2]) * (tot ? tot : 1));
    size_t np = 0;
    for (int i = 0; i < n; i++) { pr[np][0] = i; pr[np][1] = i; np++; }
    for (int q = 0; q < B->C.ef; q++) { pr[np][0] = B->C.fr[q]; pr[np][1] = B->C.fc[q]; np++; }
    for (int t = 0; t < B->ncon; t++) if (t != g_bb_skip) for (int q = 0; q < B->A[t].ef; q++) { pr[np][0] = B->A[t].fr[q]; pr[np][1] = B->A[t].fc[q]; np++; }
    qsort(pr, np, sizeof(int[2]), cmp2);
    size_t nu = 0;
    for (size_t p = 0; p < np; p++) if (!nu || pr[nu - 1][0] != pr[p][0] || pr[nu - 1][1] != pr[p][1]) { pr[nu][0] = pr[p][0]; pr[nu][1] = pr[p][1]; nu++; }
    s->np = (int)nu;
    s->sp = bz(sizeof(int) * (n + 1)); s->sj = bx(sizeof(int) * (nu + 1));
    for (size_t p = 0; p < nu; p++) { s->sp[pr[p][0] + 1]++; s->sj[p] = pr[p][1]; }
    for (int i = 0; i < n; i++) s->sp[i + 1] += s->sp[i];
    free(pr);
    s->cv = bz(sizeof(double) * (nu + 1)); s->ws = bz(sizeof(double) * (nu + 1)); s->xs = bz(sizeof(double) * (nu + 1));
    for (int q = 0; q < B->C.ef; q++) s->cv[ppos(s, B->C.fr[q], B->C.fc[q])] += B->C.fv[q];
    s->cp = bx(sizeof(int) * (B->ncon + 1)); s->cp[0] = 0;
    for (int t = 0; t < B->ncon; t++) s->cp[t + 1] = s->cp[t] + (t == g_bb_skip ? 0 : B->A[t].ef);
    s->ep = bx(sizeof(int) * ((size_t)s->cp[B->ncon] + 1)); s->ev = bx(sizeof(double) * ((size_t)s->cp[B->ncon] + 1));
    for (int t = 0; t < B->ncon; t++) if (t != g_bb_skip) for (int q = 0; q < B->A[t].ef; q++) { s->ep[s->cp[t] + q] = ppos(s, B->A[t].fr[q], B->A[t].fc[q]); s->ev[s->cp[t] + q] = B->A[t].fv[q]; }
}

static void bb_free(BB *s) {
    free(s->sp); free(s->sj); free(s->cv); free(s->ws); free(s->xs); free(s->cp); free(s->ep); free(s->ev); free(s->R);
    free(s->rho); free(s->adiag); free(s->crow); free(s->cptr); free(s->cidx);
    if (s->F) schol_free(s->F);
    if (s->Fv) schol_free(s->Fv);
    free(s->pw); free(s->pb); free(s->pW); free(s->pS);
    for (int j = 0; j < s->kx; j++) { free(s->xa ? s->xa[j] : NULL); free(s->xb ? s->xb[j] : NULL); free(s->xv ? s->xv[j] : NULL); }
    free(s->xt); free(s->xcon); free(s->xn); free(s->xa); free(s->xb); free(s->xv); free(s->xJ); free(s->xcoef);
    free(s->xrhs); free(s->xlam); free(s->xres); free(s->xw); free(s->cv0);
}

/* xs = (U V')_{ij} on the pattern (V = NULL: U U') */
static void pat_uv(const BB *s, const double *U, const double *V) {
    const int n = s->n, r = s->r;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ui = U + (size_t)i * r;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
            const double *vj = (V ? V : U) + (size_t)s->sj[p] * r;
            double v = 0; for (int l = 0; l < r; l++) v += ui[l] * vj[l];
            s->xs[p] = v;
        }
    }
}
/* out[con] += <A_t, xs> (symmetric part when xs is not symmetric: the full entries cover both (i,j), (j,i)); returns <C, xs> */
static double pat_apply(const BB *s, double *out) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < s->ncon; t++) { double v = 0; for (int e = s->cp[t]; e < s->cp[t + 1]; e++) v += s->ev[e] * s->xs[s->ep[e]]; out[s->con[t]] += v; }
    double c = 0;
    for (int p = 0; p < s->np; p++) c += s->cv[p] * s->xs[p];
    return c;
}
/* ws = C - A*(w) */
static void pat_slack(const BB *s, const double *w) {
    memcpy(s->ws, s->cv, sizeof(double) * s->np);
    for (int t = 0; t < s->ncon; t++) { const double wt = w[s->con[t]]; if (wt == 0) continue; for (int e = s->cp[t]; e < s->cp[t + 1]; e++) s->ws[s->ep[e]] -= wt * s->ev[e]; }
}
/* O = c * W U with W = ws (or cv) on the pattern */
static void pat_mul(const BB *s, const double *W, const double *U, double *O, double c) {
    const int n = s->n, r = s->r;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *oi = O + (size_t)i * r;
        for (int l = 0; l < r; l++) oi[l] = 0;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const double a = W[p]; if (a == 0) continue; const double *uj = U + (size_t)s->sj[p] * r; for (int l = 0; l < r; l++) oi[l] += a * uj[l]; }
        if (c != 1.0) for (int l = 0; l < r; l++) oi[l] *= c;
    }
    if (s->jP != 0 && (W == s->cv || W == s->ws)) {   /* + c jP e (e'U) */
        double u[512], *up = r <= 512 ? u : bx(sizeof(double) * r);
        jsum(s, U, r, up); for (int l = 0; l < r; l++) up[l] *= c * s->jP;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { double *oi = O + (size_t)i * r; const double wi = JW(s, i); for (int l = 0; l < r; l++) oi[l] += wi * up[l]; }
        if (up != u) free(up);
    }
}

/* lambda_min of W (on the pattern) by Lanczos with full reorthogonalization, k steps; v: its vector */
static double g_bm_tsw = 0, g_bm_ttr = 0, g_bm_tlz = 0, g_bm_tvf = 0;
/* the work of the fixed-diagonal method so far in modelled nanoseconds, for the
 * budget of the bound's bisection: a sweep and a product with the Hessian visit the pattern's np
 * entries with r-vectors (about 1 ns a scattered update, IMPLEMENTATION's figure for pair loops).
 * Counted, not timed, so the decision is the same in every run. */
static double g_bm_work = 0;   /* time in the sweeps, the trust region, Lanczos (verbose) */

/* the smallest eigenvalues of W (on the pattern) by Lanczos with full reorthogonalization, k steps;
 * returns the smallest Ritz value; vout (n x nv, column-major, may be NULL) receives the Ritz vectors
 * of the nv smallest Ritz values, th their values, res their residual norms |beta_k s_k|. The
 * reorthogonalization is classical Gram-Schmidt twice, threaded over the basis and over the rows in
 * fixed blocks: the same result for any thread count. */
static double lanczos_low(const BB *s, const double *W, int k, int nv, double *vout, double *th, double *res, unsigned seed, const double *v0) {
    const int n = s->n;
    if (k > n) k = n;
    double *Q = bx(sizeof(double) * (size_t)n * (k + 1)), *al = bx(sizeof(double) * (k + 1)), *be = bx(sizeof(double) * (k + 1)), *w = bx(sizeof(double) * n), *dc = bx(sizeof(double) * (k + 1));
    unsigned h = seed * 2654435761u + 12345u; double nr = 0;
    for (int i = 0; i < n; i++) { h = h * 1103515245u + 12345u; Q[i] = ((h >> 8) & 0xffff) / 65535.0 - 0.5; if (v0) Q[i] = v0[i] + 1e-3 * Q[i]; nr += Q[i] * Q[i]; }
    nr = sqrt(nr); for (int i = 0; i < n; i++) Q[i] /= nr;
    int kk = 0;
    const double t0 = wtime();
    for (int j = 0; j < k; j++) {
        const double *qj = Q + (size_t)j * n;
        double jq = 0;
        if (s->jP != 0 && (W == s->cv || W == s->ws)) { for (int i = 0; i < n; i++) jq += JW(s, i) * qj[i]; jq *= s->jP; }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { double v = JW(s, i) * jq; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) v += W[p] * qj[s->sj[p]]; w[i] = v; }
        double a = 0; for (int i = 0; i < n; i++) a += w[i] * qj[i];
        al[j] = a; kk = j + 1;
        for (int pass = 0; pass < 2; pass++) {
            const int nc = j + 1;
            #pragma omp parallel for schedule(static) if (nc >= 4)
            for (int c = 0; c < nc; c++) { const double *qc = Q + (size_t)c * n; double d = 0; for (int i = 0; i < n; i++) d += w[i] * qc[i]; dc[c] = d; }
            #pragma omp parallel for schedule(static)
            for (int i0 = 0; i0 < n; i0 += 1024) {
                const int i1 = i0 + 1024 < n ? i0 + 1024 : n;
                for (int c = 0; c < nc; c++) { const double d = dc[c]; const double *qc = Q + (size_t)c * n; for (int i = i0; i < i1; i++) w[i] -= d * qc[i]; }
            }
        }
        double bn = 0; for (int i = 0; i < n; i++) bn += w[i] * w[i]; bn = sqrt(bn);
        be[j] = bn;
        if (bn < 1e-14 || j == k - 1) break;
        double *qn = Q + (size_t)(j + 1) * n; for (int i = 0; i < n; i++) qn[i] = w[i] / bn;
    }
    double *d = bx(sizeof(double) * kk), *e = bx(sizeof(double) * (kk + 1)), *Zv = bx(sizeof(double) * (size_t)kk * kk), *wk = bx(sizeof(double) * (2 * kk + 2));
    memcpy(d, al, sizeof(double) * kk); for (int i = 0; i + 1 < kk; i++) e[i] = be[i];
    int info = 0; BL(dstev_)("V", &kk, d, e, Zv, &kk, wk, &info);
    const double th0 = d[0];
    if (nv > kk) nv = kk;
    for (int a = 0; a < nv; a++) {
        if (th) th[a] = d[a];
        if (res) res[a] = fabs(be[kk - 1] * Zv[(size_t)a * kk + kk - 1]);
        if (vout) {
            double *va = vout + (size_t)a * n;
            #pragma omp parallel for schedule(static)
            for (int i0 = 0; i0 < n; i0 += 1024) {
                const int i1 = i0 + 1024 < n ? i0 + 1024 : n;
                for (int i = i0; i < i1; i++) va[i] = 0;
                for (int c = 0; c < kk; c++) { const double zc = Zv[(size_t)a * kk + c]; const double *qc = Q + (size_t)c * n; for (int i = i0; i < i1; i++) va[i] += zc * qc[i]; }
            }
        }
    }
    free(Q); free(al); free(be); free(w); free(dc); free(d); free(e); free(Zv); free(wk);
    g_bm_tlz += wtime() - t0;
    return th0;
}
static double lanczos_min(const BB *s, const double *W, int k, double *vout, unsigned seed, const double *v0) {
    return lanczos_low(s, W, k, 1, vout, NULL, NULL, seed, v0);
}

/* lambda_min with restarts from the last Ritz vector: k steps a cycle, until two cycles agree */
static double lmin_est(const BB *s, const double *W, double *v, double scale) {
    const int n = s->n;
    const int k = n < 200 ? n : (int)fmin(g_bm_lzk, fmax(40, 3e8 / (8.0 * n)));
    double th = lanczos_min(s, W, k, v, 7u, s->vwarm && g_bm_warm ? v : NULL);
    for (int cyc = 0; cyc < 4 && k < n; cyc++) {
        const double th2 = lanczos_min(s, W, k, v, 11u + cyc, v);
        const int done = fabs(th2 - th) <= 1e-2 * fabs(th2) + 1e-12 * scale;
        th = fmin(th, th2);
        if (done) break;
    }
    return th;
}

static double nrm2(const double *x, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += x[i] * x[i]; return sqrt(s); }

/* greedy colouring of the graph of C's off-diagonal pattern (largest degree first) */
static void colour(BB *s) {
    const int n = s->n;
    int *deg = bx(sizeof(int) * n), *ord = bx(sizeof(int) * n), *col = bx(sizeof(int) * n);
    for (int i = 0; i < n; i++) { deg[i] = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i && s->cv[p] != 0) deg[i]++; ord[i] = i; col[i] = -1; }
    /* counting sort by degree, descending */
    int maxd = 0; for (int i = 0; i < n; i++) if (deg[i] > maxd) maxd = deg[i];
    int *cnt = bz(sizeof(int) * (maxd + 2));
    for (int i = 0; i < n; i++) cnt[maxd - deg[i] + 1]++;
    for (int d = 0; d <= maxd; d++) cnt[d + 1] += cnt[d];
    for (int i = 0; i < n; i++) ord[cnt[maxd - deg[i]]++] = i;
    free(cnt);
    int ncol = 0; char *used = bz(maxd + 2);
    for (int a = 0; a < n; a++) {
        const int i = ord[a];
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const int j = s->sj[p]; if (j != i && s->cv[p] != 0 && col[j] >= 0) used[col[j]] = 1; }
        int c = 0; while (used[c]) c++;
        col[i] = c; if (c + 1 > ncol) ncol = c + 1;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const int j = s->sj[p]; if (j != i && s->cv[p] != 0 && col[j] >= 0) used[col[j]] = 0; }
    }
    free(used);
    s->ncol = ncol; s->cptr = bz(sizeof(int) * (ncol + 1)); s->cidx = bx(sizeof(int) * n);
    for (int i = 0; i < n; i++) s->cptr[col[i] + 1]++;
    for (int c = 0; c < ncol; c++) s->cptr[c + 1] += s->cptr[c];
    int *pos = bx(sizeof(int) * (ncol + 1)); memcpy(pos, s->cptr, sizeof(int) * ncol);
    for (int i = 0; i < n; i++) s->cidx[pos[col[i]]++] = i;
    free(pos); free(deg); free(ord); free(col);
}

/* one block-coordinate sweep over the rows (colour by colour); returns the objective <C, R R'> */
static double sweep(BB *s) {
    const int r = s->r;
    if (s->kx) {   /* (the rows' steps on the linearization at the sweep's start) */
        x_refresh(s, s->R, r);
        const int k = s->kx; s->kx = 0;
        sweep(s);
        s->kx = k;
        return x_lag(s, s->R, r);
    }
    if (s->jP != 0) {
        /* (13 November 2026) the term jP J couples every row: the rows in turn, one at a time, e'R kept up to date (the
         * exact block-coordinate step; the colours of C's graph do not separate rows coupled through J) */
        double *u = bx(sizeof(double) * r), *gp = bx(sizeof(double) * r);
        jsum(s, s->R, r, u);
        for (int a = 0; a < s->n; a++) {
            const int i = s->cidx[a]; double *ri = s->R + (size_t)i * r;
            const double wi = JW(s, i);
            for (int l = 0; l < r; l++) gp[l] = s->jP * wi * (u[l] - wi * ri[l]);
            for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
                const int j = s->sj[p]; const double cpq = s->cv[p];
                if (j == i || cpq == 0) continue;
                const double *rj = s->R + (size_t)j * r;
                for (int l = 0; l < r; l++) gp[l] += cpq * rj[l];
            }
            double gn = 0; for (int l = 0; l < r; l++) gn += gp[l] * gp[l]; gn = sqrt(gn);
            if (gn > 0) {
                const double f = -sqrt(s->rho[i]) / gn;
                for (int l = 0; l < r; l++) { const double nv = f * gp[l]; u[l] += wi * (nv - ri[l]); ri[l] = nv; }
            }
        }
        free(u); free(gp);
        pat_uv(s, s->R, NULL);
        double f = 0; for (int p = 0; p < s->np; p++) f += s->cv[p] * s->xs[p];
        return f + s->jP * jsq(s, s->R, r);
    }
    for (int c = 0; c < s->ncol; c++) {
        const int lo = s->cptr[c], hi = s->cptr[c + 1];
        #pragma omp parallel for schedule(dynamic, 256)
        for (int a = lo; a < hi; a++) {
            const int i = s->cidx[a];
            double g[512]; double *gp = r <= 512 ? g : bx(sizeof(double) * r);
            for (int l = 0; l < r; l++) gp[l] = 0;
            for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
                const int j = s->sj[p]; const double cpq = s->cv[p];
                if (j == i || cpq == 0) continue;
                const double *rj = s->R + (size_t)j * r;
                for (int l = 0; l < r; l++) gp[l] += cpq * rj[l];
            }
            double gn = 0; for (int l = 0; l < r; l++) gn += gp[l] * gp[l]; gn = sqrt(gn);
            if (gn > 0) {
                const double f = -sqrt(s->rho[i]) / gn; double *ri = s->R + (size_t)i * r;
                if (g_bm_omega == 1.0) for (int l = 0; l < r; l++) ri[l] = f * gp[l];
                else {   /* over-relaxed: (1 - w) old + w new, back on the sphere */
                    double nr = 0;
                    for (int l = 0; l < r; l++) { ri[l] = (1 - g_bm_omega) * ri[l] + g_bm_omega * f * gp[l]; nr += ri[l] * ri[l]; }
                    const double k = sqrt(s->rho[i] / nr); for (int l = 0; l < r; l++) ri[l] *= k;
                }
            }
            if (gp != g) free(gp);
        }
    }
    pat_uv(s, s->R, NULL);
    double f = 0; for (int p = 0; p < s->np; p++) f += s->cv[p] * s->xs[p];
    return f;
}

/* append a column to R (rank r -> r + 1) holding v * scale, then renormalize the fixed rows */
static void add_col(BB *s, const double *v, double scale) {
    const int n = s->n, r = s->r, r1 = r + 1;
    double *Rn = bx(sizeof(double) * (size_t)n * r1);
    for (int i = 0; i < n; i++) { memcpy(Rn + (size_t)i * r1, s->R + (size_t)i * r, sizeof(double) * r); Rn[(size_t)i * r1 + r] = scale * v[i]; }
    free(s->R); s->R = Rn; s->r = r1;
    if (s->rho) for (int i = 0; i < n; i++) { double *ri = s->R + (size_t)i * r1; const double nr = nrm2(ri, r1); if (nr > 0) { const double f = sqrt(s->rho[i]) / nr; for (int l = 0; l < r1; l++) ri[l] *= f; } }
}

typedef struct { double pobj, dobj, e1, e2, e3, lmin, cgap, lb, vdelta; int cert; } BMMeas;   /* cert: 1 the bound checked by a factorization, 0 lambda_min a Lanczos estimate */

/* mu_p = (C R)_p . R_p / rho_p and G = Z R = C R - Diag(mu) R (half the Riemannian gradient); returns <C, RR'> */
static double fd_grad(const BB *s, const double *R, double *mu, double *G) {
    const int n = s->n, r = s->r;
    if (s->kx) x_refresh((BB *)s, R, r);   /* (the linearization at R: the gradient of the augmented Lagrangian) */
    pat_mul(s, s->cv, R, G, 1.0);
    double f = 0;
    #pragma omp parallel for schedule(static) reduction(+:f)
    for (int i = 0; i < n; i++) {
        const double *ri = R + (size_t)i * r; double *gi = G + (size_t)i * r;
        double d = 0; for (int l = 0; l < r; l++) d += ri[l] * gi[l];
        f += d; mu[i] = d / s->rho[i];
        for (int l = 0; l < r; l++) gi[l] -= mu[i] * ri[l];
    }
    if (s->kx) f = x_lag((BB *)s, R, r);
    return f;
}
/* Q = Proj_T(Z P): Z P = C P - Diag(mu) P, then each row projected off R_p */
static void fd_hess(const BB *s, const double *R, const double *mu, const double *Pd, double *Q) {
    const int n = s->n, r = s->r;
    pat_mul(s, s->cv, Pd, Q, 1.0);
    if (s->kx) x_hess_add(s, R, Pd, Q);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ri = R + (size_t)i * r, *pi = Pd + (size_t)i * r; double *qi = Q + (size_t)i * r;
        double d = 0;
        for (int l = 0; l < r; l++) { qi[l] -= mu[i] * pi[l]; d += qi[l] * ri[l]; }
        d /= s->rho[i];
        for (int l = 0; l < r; l++) qi[l] -= d * ri[l];
    }
}
/* a'b over fixed chunks of 8192, the partial sums added in order: the same bits for any thread count */
#define BM_CH 8192
static double dotp(const double *a, const double *b, size_t n) {
    const size_t nc = (n + BM_CH - 1) / BM_CH;
    if (nc <= 1) { double s = 0; for (size_t i = 0; i < n; i++) s += a[i] * b[i]; return s; }
    double part[1024], *pp = nc <= 1024 ? part : bx(sizeof(double) * nc);
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < nc; c++) { const size_t lo = c * BM_CH, hi = lo + BM_CH < n ? lo + BM_CH : n; double s = 0; for (size_t i = lo; i < hi; i++) s += a[i] * b[i]; pp[c] = s; }
    double s = 0; for (size_t c = 0; c < nc; c++) s += pp[c];
    if (pp != part) free(pp);
    return s;
}
/* y = a x + b y (b = 1: y += a x) */
static void axpby(size_t n, double a, const double *x, double b, double *y) {
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i++) y[i] = b == 0.0 ? a * x[i] : a * x[i] + b * y[i];
}
/* R <- rows of R + D normalized to |R_p|^2 = rho_p */
static void retract(const BB *s, const double *R, const double *D, double a, double *Rn) {
    const int n = s->n, r = s->r;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ri = R + (size_t)i * r, *di = D + (size_t)i * r; double *o = Rn + (size_t)i * r; double nr = 0;
        for (int l = 0; l < r; l++) { o[l] = ri[l] + a * di[l]; nr += o[l] * o[l]; }
        const double f = sqrt(s->rho[i] / nr); for (int l = 0; l < r; l++) o[l] *= f;
    }
}
/* Riemannian trust region with truncated CG (Steihaug-Toint) on the product of spheres, until
 * |grad f| = 2 |Z R| <= gtol. Model: f + <g, e> + <e, H e>/2 with g = 2 Z R, H e = 2 Proj(Z e).
 * Returns the outer steps taken. */
/* the preconditioner: the analysis of Z's pattern once (kept when the factor stays within 30 times the
 * pattern), then each call a factorization of Z + delta I, delta raised until it succeeds */
static int prec_build(BB *s, const double *mu, int verbose) {
    const int n = s->n;
    if (s->fstate < 0) return 0;
    if (s->fstate == 0) {
        s->fstate = -1;
        if (!g_bm_prec) return 0;
        int *deg = bz(sizeof(int) * (n + 1)); int **nbr = bx(sizeof(int *) * (n + 1));
        for (int i = 0; i < n; i++) { for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) deg[i]++; nbr[i] = bx(sizeof(int) * (deg[i] + 1)); int c = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) nbr[i][c++] = s->sj[p]; }
        s->F = schol_analyze_adj(n, deg, nbr, (size_t)30 * s->np + (size_t)n);
        for (int i = 0; i < n; i++) free(nbr[i]);
        free(nbr); free(deg);
        if (!s->F) { if (verbose > 0) printf("bm: no preconditioner (the factor would fill)\n"); return 0; }
        if (verbose >= 0) printf("bm: preconditioner: sparse Cholesky of Z + delta I, %zu nonzeros\n", schol_nnz(s->F));
        s->fstate = 1;
    }
    /* the scale: the mean |Z_pp| */
    double dsum = 0;
    for (int i = 0; i < n; i++) { const int pd = ppos(s, i, i); dsum += fabs((pd >= 0 ? s->cv[pd] : 0.0) - mu[i]); }
    const double dav = dsum / n + 1e-300;
    int ok = 0;
    for (double delta = g_bm_pdelta * dav; delta < 10 * dav; delta *= 10) {
        schol_zero(s->F);
        for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
            const int j = s->sj[p]; if (j > i) continue;
            schol_add(s->F, i, j, s->cv[p] + (i == j ? delta - mu[i] : 0.0));
        }
        if (schol_factor(s->F, 0.0) == 0) { ok = 1; break; }
    }
    if (!ok) return 0;
    /* Z R ~ 0 near a solution: (Z + delta I)^-1 would blow up R's columns, which the projection does not
     * remove; the preconditioner is (Z + delta I + pmu R (R'R)^-1 R')^-1 instead, by Woodbury:
     * A^-1 - W (R'R/pmu + R'W)^-1 W' with W = A^-1 R */
    const int r = s->r;
    if (!s->pb) { s->pb = bx(sizeof(double) * (size_t)n * r); s->pw = bx(sizeof(double) * (size_t)n * r * 2 + 64); }
    if (!s->pW) { s->pW = bx(sizeof(double) * (size_t)n * r); s->pS = bx(sizeof(double) * (size_t)r * r); }
    s->pmu = g_bm_pmu * dav;
    if (s->pmu <= 0) return 1;
    for (int i = 0; i < n; i++) for (int l = 0; l < r; l++) s->pW[i + (size_t)l * n] = s->R[(size_t)i * r + l];
    schol_solve_many(s->F, s->pW, r, s->pw);
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) {
        double v = 0, u = 0; const double *wb = s->pW + (size_t)b * n;
        for (int i = 0; i < n; i++) { const double ria = s->R[(size_t)i * r + a]; v += ria * s->R[(size_t)i * r + b]; u += ria * wb[i]; }
        s->pS[a + (size_t)b * r] = v / s->pmu + u;
    }
    for (int a = 0; a < r; a++) for (int b = a + 1; b < r; b++) { const double m = 0.5 * (s->pS[a + (size_t)b * r] + s->pS[b + (size_t)a * r]); s->pS[a + (size_t)b * r] = s->pS[b + (size_t)a * r] = m; }
    int info = 0; BL(dpotrf_)("L", &r, s->pS, &r, &info);
    if (info) s->pmu = 0;
    return 1;
}
/* Z_out = Proj((Z + delta I)^-1 Proj(Rin)) (row-major n x r) */
static void prec_apply(BB *s, const double *R, const double *Rin, double *Zo) {
    const int n = s->n, r = s->r;
    if (!s->pb) { s->pb = bx(sizeof(double) * (size_t)n * r); s->pw = bx(sizeof(double) * (size_t)n * r * 2 + 64); }
    for (int i = 0; i < n; i++) for (int l = 0; l < r; l++) s->pb[i + (size_t)l * n] = Rin[(size_t)i * r + l];
    schol_solve_many(s->F, s->pb, r, s->pw);
    if (s->pmu > 0) {   /* pb -= W S^-1 (R' pb) */
        double T[64 * 64], *Tp = r <= 64 ? T : bx(sizeof(double) * (size_t)r * r);
        #pragma omp parallel for schedule(static)
        for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) {
            double v = 0; const double *pbb = s->pb + (size_t)b * n;
            for (int i = 0; i < n; i++) v += R[(size_t)i * r + a] * pbb[i];
            Tp[a + (size_t)b * r] = v;
        }
        int info = 0; BL(dpotrs_)("L", &r, &r, s->pS, &r, Tp, &r, &info);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) for (int b = 0; b < r; b++) {
            double v = 0; for (int a = 0; a < r; a++) v += s->pW[i + (size_t)a * n] * Tp[a + (size_t)b * r];
            s->pb[i + (size_t)b * n] -= v;
        }
        if (Tp != T) free(Tp);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ri = R + (size_t)i * r; double *zi = Zo + (size_t)i * r, d = 0;
        for (int l = 0; l < r; l++) { zi[l] = s->pb[i + (size_t)l * n]; d += zi[l] * ri[l]; }
        d /= s->rho[i];
        for (int l = 0; l < r; l++) zi[l] -= d * ri[l];
    }
}

/* E += a P, Rr += a Q; returns |Rr|^2 (chunk sums added in order: the same bits for any thread count) */
static double cg_upd(size_t N, double a, const double *P, const double *Q, double *E, double *Rr) {
    const size_t nc = (N + BM_CH - 1) / BM_CH;
    double part[1024], *pp = nc <= 1024 ? part : bx(sizeof(double) * nc);
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < nc; c++) {
        const size_t lo = c * BM_CH, hi = lo + BM_CH < N ? lo + BM_CH : N; double v = 0;
        for (size_t q = lo; q < hi; q++) { E[q] += a * P[q]; const double t = Rr[q] + a * Q[q]; Rr[q] = t; v += t * t; }
        pp[c] = v;
    }
    double v = 0; for (size_t c = 0; c < nc; c++) v += pp[c];
    if (pp != part) free(pp);
    return v;
}
/* Q = Proj_T(Z P) as fd_hess; returns <P, Q> (row sums added in order) */
static double fd_hess_dot(const BB *s, const double *R, const double *mu, const double *Pd, double *Q, double *rowv) {
    const int n = s->n, r = s->r;
    double ju[512], *jup = r <= 512 ? ju : bx(sizeof(double) * r);
    if (s->jP != 0) { jsum(s, Pd, r, jup); for (int l = 0; l < r; l++) jup[l] *= s->jP; } else for (int l = 0; l < r; l++) jup[l] = 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *qi = Q + (size_t)i * r; const double *ri = R + (size_t)i * r, *pi = Pd + (size_t)i * r;
        { const double wi = JW(s, i); for (int l = 0; l < r; l++) qi[l] = wi * jup[l] - mu[i] * pi[l]; }
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const double a = s->cv[p]; if (a == 0) continue; const double *uj = Pd + (size_t)s->sj[p] * r; for (int l = 0; l < r; l++) qi[l] += a * uj[l]; }
        double d = 0; for (int l = 0; l < r; l++) d += qi[l] * ri[l];
        d /= s->rho[i];
        double v = 0; for (int l = 0; l < r; l++) { qi[l] -= d * ri[l]; v += qi[l] * pi[l]; }
        rowv[i] = v;
    }
    if (jup != ju) free(jup);
    double v = 0; for (int i = 0; i < n; i++) v += rowv[i];
    return v;
}

static int fd_newton(BB *s, double gtol, int maxit, double tlim, double t0, int verbose) {
    const int n = s->n; const size_t N = (size_t)n * s->r;
    double *mu = bx(sizeof(double) * n), *G = bx(sizeof(double) * N), *E = bx(sizeof(double) * N), *Rr = bx(sizeof(double) * N),
           *Pd = bx(sizeof(double) * N), *Q = bx(sizeof(double) * N), *Rn = bx(sizeof(double) * N), *HE = bx(sizeof(double) * N), *Zr = bx(sizeof(double) * N);
    if (s->pb && (size_t)n * s->r > 0) { free(s->pb); free(s->pw); s->pb = s->pw = NULL; }   /* (the rank may have changed) */
    free(s->pW); free(s->pS); s->pW = s->pS = NULL;
    int it = 0; long ncg = 0;
    double f = fd_grad(s, s->R, mu, G);           /* G = Z R: half the gradient */
    double rsum = 0; for (int i = 0; i < n; i++) rsum += s->rho[i];
    const double dmax = sqrt(rsum);
    double D = -1;
    for (; it < maxit; it++) {
        const double gn = 2 * sqrt(dotp(G, G, N));    /* |grad| */
        if (gn <= gtol || (tlim > 0 && wtime() - t0 > tlim)) break;
        const int pc = prec_build(s, mu, verbose);
        double mdec;
        int bound = 0;
        if (!pc && g_bm_fuse && !s->kx) {   /* tCG (Steihaug-Toint) on Proj(Z e) = -G in fused passes; H E is Rr - G */
            double *rowv = Zr;
            #pragma omp parallel for schedule(static) if (N > 65536)
            for (size_t q = 0; q < N; q++) { E[q] = 0; Rr[q] = G[q]; Pd[q] = -G[q]; }
            double rz = dotp(G, G, N); const double r0 = sqrt(rz);
            if (D < 0) D = dmax / 8;
            double ee = 0, ep = 0, pp = rz;
            for (int k = 0; k < 1000; k++) {
                const double pq = fd_hess_dot(s, s->R, mu, Pd, Q, rowv); ncg++; g_bm_work += 2.0 * s->np * s->r;
                const double a = pq > 0 ? rz / pq : 0;
                if (pq <= 0 || ee + 2 * a * ep + a * a * pp >= D * D) {
                    const double tau = (-ep + sqrt(fmax(0, ep * ep + pp * (D * D - ee)))) / pp;
                    cg_upd(N, tau, Pd, Q, E, Rr);
                    bound = 1; break;
                }
                const double rr = cg_upd(N, a, Pd, Q, E, Rr);
                ee = ee + 2 * a * ep + a * a * pp;
                if (sqrt(rr) <= r0 * fmin(0.1, sqrt(r0))) break;
                const double be = rr / rz; rz = rr;
                axpby(N, -1.0, Rr, be, Pd);
                ep = be * (ep + a * pp); pp = rr + be * be * pp;
            }
            /* model decrease of f: -(<g,e> + <e,He>/2) with g = 2G, He = 2(Rr - G) */
            mdec = -(dotp(G, E, N) + dotp(E, Rr, N));
        } else {
        /* tCG (preconditioned, Steihaug-Toint) on Proj(Z e) = -G, the trust region in the preconditioner's norm */
        for (size_t q = 0; q < N; q++) { E[q] = 0; HE[q] = 0; Rr[q] = G[q]; }
        if (pc) prec_apply(s, s->R, Rr, Zr); else memcpy(Zr, Rr, sizeof(double) * N);
        for (size_t q = 0; q < N; q++) Pd[q] = -Zr[q];
        double rz = dotp(Rr, Zr, N); const double r0 = sqrt(dotp(Rr, Rr, N));
        if (D < 0) D = pc ? sqrt(rz) : dmax / 8;
        double ee = 0, ep = 0, pp = rz;
        for (int k = 0; k < 1000; k++) {
            fd_hess(s, s->R, mu, Pd, Q); ncg++; g_bm_work += 2.0 * s->np * s->r;
            const double pq = dotp(Pd, Q, N);
            const double a = pq > 0 ? rz / pq : 0;
            if (pq <= 0 || ee + 2 * a * ep + a * a * pp >= D * D) {
                const double tau = (-ep + sqrt(fmax(0, ep * ep + pp * (D * D - ee)))) / pp;
                axpby(N, tau, Pd, 1.0, E); axpby(N, tau, Q, 1.0, HE);
                bound = 1; break;
            }
            axpby(N, a, Pd, 1.0, E); axpby(N, a, Q, 1.0, HE); axpby(N, a, Q, 1.0, Rr);
            ee = ee + 2 * a * ep + a * a * pp;
            const double rn = sqrt(dotp(Rr, Rr, N));
            if (rn <= r0 * fmin(0.1, sqrt(r0))) break;
            if (pc) prec_apply(s, s->R, Rr, Zr); else memcpy(Zr, Rr, sizeof(double) * N);
            const double rz2 = dotp(Rr, Zr, N);
            const double be = rz2 / rz; rz = rz2;
            axpby(N, -1.0, Zr, be, Pd);
            ep = be * (ep + a * pp); pp = rz2 + be * be * pp;
        }
        /* model decrease of f: -(<g,e> + <e,He>/2) with g = 2G, He = 2 HE */
        mdec = -(2 * dotp(G, E, N) + dotp(E, HE, N));
        }
        retract(s, s->R, E, 1.0, Rn);
        pat_uv(s, Rn, NULL); double fn = 0; for (int p = 0; p < s->np; p++) fn += s->cv[p] * s->xs[p];
        if (s->kx) fn = x_lag(s, Rn, s->r); else if (s->jP != 0) fn += s->jP * jsq(s, Rn, s->r);
        /* the ratio regularized at the round-off of f (as in Manopt): near the end both differences are noise */
        const double reg = 1e3 * 2.220446e-16 * fmax(1.0, fabs(f));
        const double rho = mdec > 0 ? (f - fn + reg) / (mdec + reg) : -1;
        if (rho < 0.25) D *= 0.25; else if (rho > 0.75 && bound) D = 2 * D;
        if (rho > 0.1 || (mdec <= 1e-15 * fabs(f) && fn <= f)) {
            memcpy(s->R, Rn, sizeof(double) * N);
            f = fd_grad(s, s->R, mu, G);
        }
        if (verbose > 1) printf("bm: newton %d  obj %.12e  |grad| %.2e  rho %.2f radius %.2e  cg %ld%s%s\n", it, f, gn, rho, D, ncg, bound ? " (boundary)" : "", pc ? " P" : "");
        if (D < 1e-14 * dmax) break;
    }
    free(mu); free(G); free(E); free(Rr); free(Pd); free(Q); free(Rn); free(HE); free(Zr);
    return it;
}

/* the certificate of the fixed-diagonal method: y from the stationarity conditions (a_p y_p =
 * (C R)_p . R_p / rho_p), Z = C - Diag(a y) on the pattern, lambda_min(Z) by Lanczos (its vector in v);
 * returns |Z R|_F */
static double fd_cert(BB *s, const Problem *P, double *y, double *v, double c1, BMMeas *M) {
    const int n = s->n, r = s->r;
    if (s->kx) x_refresh(s, s->R, r);
    double *G = bx(sizeof(double) * (size_t)n * r);
    pat_mul(s, s->cv, s->R, G, 1.0);
    double f = 0, gg = 0;
    for (int i = 0; i < n; i++) {
        const double *ri = s->R + (size_t)i * r, *gi = G + (size_t)i * r;
        double d = 0; for (int l = 0; l < r; l++) d += ri[l] * gi[l];
        f += d;
        const double ay = d / s->rho[i];
        y[s->crow[i]] = ay / s->adiag[i];
        for (int l = 0; l < r; l++) { const double z = gi[l] - ay * ri[l]; gg += z * z; }
    }
    if (s->kx) for (int j = 0; j < s->kx; j++) f += s->xw[j] * (s->xres[j] + s->xrhs[j]);   /* <C, X> = <C - sum w A, X> + sum w <A, X> */
    if (s->kx) for (int j = 0; j < s->kx; j++) y[s->xcon[j]] = s->xw[j];   /* Z = C - Diag(a y) - sum w_j A_j */
    else if (s->jlast) { const double yn = y[P->m]; y[s->jcon] = s->jP - yn > 0 ? s->jP * yn / (s->jcoef * (s->jP - yn)) : -s->jP / s->jcoef; }   /* (the lift's Schur complement) */
    else if (s->jP != 0) y[s->jcon] = -s->jP / s->jcoef;   /* Z = C - Diag(a y) - y_J a_J J = C + jP J - Diag(a y) */
    double dobj = 0; for (int i = 0; i < P->m; i++) if (!(s->jlast && i == s->jcon)) dobj += P->b[i] * y[i];
    if (s->jlast) dobj += s->jb * y[P->m];   /* (the lifted dual's objective: a lower bound on the common optimal value) */
    memcpy(s->ws, s->cv, sizeof(double) * s->np);
    for (int i = 0; i < n; i++) s->ws[ppos(s, i, i)] -= s->adiag[i] * y[s->crow[i]];
    const double lm = lmin_est(s, s->ws, v, c1 / (s->np > 0 ? s->np : 1));
    s->vwarm = 1;
    M->cert = 0; M->pobj = f; M->dobj = dobj; M->lmin = lm;
    /* (3 November 2026, the owner: "Is it possible that the dimacs errors are 0?") the rows of R are scaled to the fixed
     * diagonal, so A(X) = b up to rounding: measured, not set to 0 */
    double r2 = 0, b1 = 0;
    for (int i = 0; i < P->m; i++) b1 += fabs(P->b[i]);
    for (int i = 0; i < n; i++) {
        const double *ri = s->R + (size_t)i * r; double d = 0; for (int l = 0; l < r; l++) d += ri[l] * ri[l];
        const double e = s->adiag[i] * d - P->b[s->crow[i]]; r2 += e * e;
    }
    if (s->kx) for (int j = 0; j < s->kx; j++) r2 += s->xres[j] * s->xres[j];
    else if (s->jlast) {   /* the row as read: a |e'R_{0..n-1}|^2 - b */
        double u2 = 0; for (int l = 0; l < r; l++) { double v = 0; for (int i = 0; i < n - 1; i++) v += s->R[(size_t)i * r + l]; u2 += v * v; }
        const double e = s->jcoef * u2 - P->b[s->jcon]; r2 += e * e;
    }
    else if (s->jP != 0) { const double e = s->jcoef * jsq(s, s->R, r) - P->b[s->jcon]; r2 += e * e; }
    M->e1 = sqrt(r2) / (1 + b1); M->e2 = fmax(0, -lm) / (1 + c1); M->e3 = fabs(f - dobj) / (1 + fabs(f) + fabs(dobj));
    /* the certified bound: y + min(lambda_min, 0)/a makes Z positive semidefinite, b'y then drops by |lambda_min| sum rho */
    double rs = 0; for (int i = 0; i < n; i++) rs += s->rho[i];
    M->lb = dobj + fmin(lm, 0) * rs;
    M->cgap = (f - M->lb) / (1 + fabs(f) + fabs(M->lb));
    free(G);
    return sqrt(gg);
}

/* the check of a certificate: Z + delta I (Z = s->ws on the pattern) is positive definite when its
 * sparse Cholesky factorization succeeds. The analysis once, kept when the factor has at most
 * max(30 np, 2e7) entries. Returns 1 (it factors), 0 (it does not) or -1 (no factor affordable). */
static int fd_verify_(BB *s, double delta);
static int g_bm_nvf = 0;   /* the checks made (the log) */
static int fd_verify(BB *s, double delta) { const double t = wtime(); const int v = fd_verify_(s, delta); g_bm_tvf += wtime() - t; g_bm_nvf++; return v; }
static int fd_verify_(BB *s, double delta) {
    const int n = s->n;
    if (brisk_fault("bmverify")) return -1;
    int hasJ = s->jP != 0; for (int j = 0; j < s->kx; j++) hasJ |= s->xJ[j];
    if (hasJ) {
        /* (13 November 2026) Z = S + jP J is dense: its Cholesky factorization, dense, up to order 12,000 (1.2 GB) */
        if (n > 12000) return -1;
        double *Zd = malloc(sizeof(double) * ((size_t)n * n + 1));
        if (!Zd) return -1;
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Zd[i + (size_t)j * n] = s->jP * JW(s, i) * JW(s, j);
        for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) Zd[i + (size_t)s->sj[p] * n] += s->ws[p];
        for (int i = 0; i < n; i++) Zd[i + (size_t)i * n] += delta;
        const int info = brisk_dpotrf(n, Zd, n);
        free(Zd);
        return info == 0 ? 1 : 0;
    }
    if (s->vstate < 0) return -1;
    if (s->vstate == 0) {
        s->vstate = -1;
        int *deg = bz(sizeof(int) * (n + 1)); int **nbr = bx(sizeof(int *) * (n + 1));
        for (int i = 0; i < n; i++) { for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) deg[i]++; nbr[i] = bx(sizeof(int) * (deg[i] + 1)); int c = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) if (s->sj[p] != i) nbr[i][c++] = s->sj[p]; }
        const size_t cap = (size_t)fmax(30.0 * s->np + n, 2e7);
        s->Fv = schol_analyze_adj(n, deg, nbr, cap);
        for (int i = 0; i < n; i++) free(nbr[i]);
        free(nbr); free(deg);
        if (!s->Fv) return -1;
        s->vstate = 1;
    }
    schol_zero(s->Fv);
    for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
        const int j = s->sj[p]; if (j > i) continue;
        schol_add(s->Fv, i, j, s->ws[p] + (i == j ? delta : 0.0));
    }
    return schol_factor(s->Fv, 0.0) == 0 ? 1 : 0;
}

/* <C, R R'> for an n x r1 factor (row sums, then added in order) */
static double fd_obj(const BB *s, const double *R, int r1, double *rowf) {
    const int n = s->n;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ri = R + (size_t)i * r1; double v = 0;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const double c = s->cv[p]; if (c == 0) continue; const double *rj = R + (size_t)s->sj[p] * r1; double d = 0; for (int l = 0; l < r1; l++) d += ri[l] * rj[l]; v += c * d; }
        rowf[i] = v;
    }
    double f = 0; for (int i = 0; i < n; i++) f += rowf[i];
    if (s->kx) return x_lag((BB *)s, R, r1);
    if (s->jP != 0) f += s->jP * jsq(s, R, r1);
    return f;
}

/* escape from a point where Z = C - Diag(a y) (in s->ws) has negative eigenvalues: the Ritz vectors of
 * the most negative ones (at most g_bm_escp, each at least a tenth of the most negative) become new
 * columns t V, the rows renormalized; t by a search on the objective over t = 0.5 |R| 2^-q, q = 0..15;
 * no escape (0 returned) when none of them lowers the objective f0 */
static int fd_escape(BB *s, const double *v, double rnorm, double f0, int verbose) {
    const int n = s->n, r = s->r;
    int pm = g_bm_escp < 1 ? 1 : g_bm_escp;
    const int k = n < 200 ? n : (int)fmin(g_bm_lzk, fmax(40, 3e8 / (8.0 * n)));
    if (pm > k / 4) pm = k / 4 > 1 ? k / 4 : 1;
    double *V = bx(sizeof(double) * (size_t)n * pm), *th = bx(sizeof(double) * pm), *res = bx(sizeof(double) * pm);
    int p = 0;
    if (pm > 1) {
        lanczos_low(s, s->ws, k, pm, V, th, res, 13u, v);
        for (int a = 0; a < pm; a++) if (th[a] < 0 && th[a] <= 0.1 * th[0]) { if (a != p) memcpy(V + (size_t)p * n, V + (size_t)a * n, sizeof(double) * n); p++; }
    }
    if (p == 0) { memcpy(V, v, sizeof(double) * n); p = 1; }
    const int r1 = r + p;
    double *Rn = bx(sizeof(double) * (size_t)n * r1), *Rb = bx(sizeof(double) * (size_t)n * r1), *rowf = bx(sizeof(double) * n);
    double fb = 1e300, tb = 0;
    for (int q = 0; q < 16; q++) {
        const double t = 0.5 * rnorm * ldexp(1.0, -q);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double *o = Rn + (size_t)i * r1; memcpy(o, s->R + (size_t)i * r, sizeof(double) * r);
            for (int a = 0; a < p; a++) o[r + a] = t * V[(size_t)a * n + i];
            double nr = 0; for (int l = 0; l < r1; l++) nr += o[l] * o[l];
            const double f = sqrt(s->rho[i] / nr); for (int l = 0; l < r1; l++) o[l] *= f;
        }
        const double f = fd_obj(s, Rn, r1, rowf);
        if (f < fb) { fb = f; tb = t; double *x = Rb; Rb = Rn; Rn = x; }
    }
    if (!(fb < f0)) {
        if (verbose > 0) printf("bm: escape: no step along %d Ritz vector(s) lowers the objective\n", p);
        free(V); free(th); free(res); free(Rn); free(Rb); free(rowf);
        return 0;
    }
    free(s->R); s->R = Rb; s->r = r1;
    if (verbose > 0) printf("bm: escape: %d column(s) (Ritz values %.2e .. %.2e), step %.2e, objective %.10e\n", p, th && pm > 1 ? th[0] : 0.0, pm > 1 ? th[p - 1] : 0.0, tb, fb);
    free(V); free(th); free(res); free(Rn); free(rowf);
    return p;
}

/* the fixed-diagonal method on one block. Returns 0 when the tolerance is met. */
static int fixed_diag(BB *s, const Problem *P, double tol, int verbose, double tlim, double t0, BMMeas *M, double *yout) {
    const int n = s->n;
    colour(s);
    double c1 = 0;
    for (int p = 0; p < s->np; p++) c1 += fabs(s->cv[p]);
    double *v = bx(sizeof(double) * n), *y = bz(sizeof(double) * (P->m + 1));
    double rs = 0; for (int i = 0; i < n; i++) rs += s->rho[i];
    const double rnorm = sqrt(rs);
    g_bm_work = 0;
    double f = sweep(s), fprev = f;
    g_bm_work += (double)s->np * s->r;
    int it = 0, nesc = 0, rc = 1, ntr = 0, nver = 0, nquick = 0, bm_rows = 0;   /* (bm_rows: the default log's rows, ISSUES 725) */
    if (verbose >= 0) printf("bm: fixed diagonal, n = %d, rank %d, %d colours\n", n, s->r, s->ncol);
    int dosweep = 1, fresh = 0;
    for (;;) {
        /* sweeps until the objective settles */
        const double ts = wtime();
        for (int ns = 0; dosweep && ns < (s->kx ? 30 : 100000); ns++) {   /* (with extra rows each sweep moves the linearization: at most 30, the trust region does the rest) */
            fprev = f; f = sweep(s); it++; g_bm_work += (double)s->np * s->r;
            if (fabs(fprev - f) <= g_bm_sweeptol * (1 + fabs(f)) && ns >= 3) break;
            if (tlim > 0 && wtime() - t0 > tlim) break;
        }
        g_bm_tsw += wtime() - ts;
        int escape = 0;
        for (int round = 0; ; round++) {
            const double gn = fd_cert(s, P, y, v, c1, M);
            if (verbose >= 0 && g_log_brief) {   /* (ISSUES 725: the default log's row) */
                if (!bm_rows) log_user(" STEP  OBJECTIVE          RANK  MIN EIG(Z)  ERROR    TIME\n");
                log_user("%4d  %+.10e  %4d  %+.2e  %.1e %6.2f\n", bm_rows++, M->pobj, s->r, M->lmin, M->e2, wtime() - t0);
            } else if (verbose >= 0) printf("bm: %6d sweeps %4d TR  rank %3d  obj %.10e  lambda_min(Z) %.2e  |ZR| %.1e  err2 %.1e  %.2fs\n", it, ntr, s->r, M->pobj, M->lmin, gn, M->e2, wtime() - t0);
            /* (an inner solve done, its Riemannian gradient small against the objective: the multipliers' step; Z need not be
             * positive semidefinite before the multipliers are right) */
            if (s->kx && M->e1 > tol && round > 0 && gn * rnorm <= 1e-6 * (1 + fabs(M->pobj))) { rc = 2; break; }
            if (!s->kx && s->jP != 0 && M->e2 <= tol && M->e3 <= tol && M->e1 > tol) { rc = 2; break; }   /* (the penalty too small: e'Xe stays above 0) */
            if (M->e2 <= tol && M->e3 <= tol && (!g_bm_cert || M->cgap <= tol)) {
                /* check the bound: the smallest delta = max(-lambda_min, 0) + eta 4^q for which Z + delta I
                 * factors (eta: a twentieth of the tolerance's room); a Lanczos estimate can be too high */
                const double eta = 0.05 * tol * (1 + fabs(M->pobj) + fabs(M->lb)) / rs;
                double delta = fmax(-M->lmin, 0) + eta; int vr = -1;
                const double tv0 = wtime(); double dlo = -1;
                int nv = 0;                 /* the checks (factorizations) made for this bound */
                /* (13 November 2026) with a <J, X> row each check is a dense factorization: the shift's search by factors of 4
                 * from eta (up to eleven failed factorizations of order 7000 on G60_mb, SDP) becomes two checks, the Lanczos
                 * estimate's shift and 99 % of the largest shift that certifies at the tolerance (the gap's and Z's limits); when
                 * neither factors the iterations go on (not counted as a failed attempt); after four such misses the full search
                 * as before, so that a bound is still found */
                int hasJc = s->jP != 0; for (int j = 0; j < s->kx; j++) hasJc |= s->xJ[j];
                const int quick = g_bm_cert && g_bm_verify && hasJc && nquick < 4;   /* (after four failed quick attempts, the full search) */
                if (quick) {
                    vr = fd_verify(s, delta); nv++;
                    if (vr == 0) { dlo = delta; const double d2 = 0.99 * fmin((tol * (1 + fabs(M->pobj) + fabs(M->dobj)) - (M->pobj - M->dobj)) / rs, tol * (1 + c1) + eta); if (d2 > delta) { delta = d2; vr = fd_verify(s, delta); nv++; } }
                    if (vr == 0) dlo = -1;   /* (no bisection: the next attempt starts afresh) */
                }
                else if (g_bm_cert && g_bm_verify) for (int q = 0; q < 12; q++, delta = fmax(-M->lmin, 0) + eta * ldexp(1.0, 2 * q)) { vr = fd_verify(s, delta); nv++; if (vr != 0) break; dlo = delta; }
                if (vr == 1 && dlo > 0) {   /* bisect between the last shift that failed and the first that factored, while cheap */
                    /* "cheap": the checks, the next one included, within a tenth of the
                     * method's work, both in modelled nanoseconds: a check is a sparse factorization,
                     * 0.05 ns a flop of schol_flops (IMPLEMENTATION's figure for products in panels) and
                     * 1 ns an entry of its assembly; the method's work is g_bm_work. Until then the rule
                     * compared the elapsed times of the checks and of the method: the number of bisection
                     * steps, and with it the shift, the certified bound, the gap and the errors reported,
                     * changed from run to run with the threads' timing (maxcut mcp500-2_s at eight
                     * threads: three outcomes in ten runs; now the same in every run and at 1-8 threads).
                     * BRISK_BMBISTIME=1: the timed rule (comparisons). */
                    static int timed = -1; if (timed < 0) timed = getenv("BRISK_BMBISTIME") != NULL;
                    const double tf = wtime() - tv0;
                    /* (13 November 2026) with a <J, X> row the check is a dense factorization (fd_verify): n^3/3 flops and n^2
                     * entries, not the sparse factor's count (Fv is not built there; it counted 0 and allowed six dense
                     * bisection steps: G60_mb, SDP, order 7000, 21 s of its 34 s in the checks) */
                    int hasJv = s->jP != 0; for (int j = 0; j < s->kx; j++) hasJv |= s->xJ[j];
                    const double fl = hasJv ? 0.05 * (double)s->n * s->n * s->n / 3.0 + (double)s->n * s->n : 0.05 * schol_flops(s->Fv) + (double)s->np;
                    for (int b = 0; b < 6 && (timed ? (wtime() - tv0) + tf < 0.1 * (wtime() - t0) : (nv + 1) * fl < 0.1 * (g_bm_work + nv * fl)); b++) {
                        const double dm = sqrt(dlo * delta);
                        if (fd_verify(s, dm) == 1) delta = dm; else dlo = dm;
                        nv++;
                    }
                }
                if (vr == 1) {
                    M->cert = 1; M->vdelta = delta; M->lmin = fmin(M->lmin, -delta + eta);
                    M->lb = M->dobj - delta * rs; M->cgap = (M->pobj - M->lb) / (1 + fabs(M->pobj) + fabs(M->lb));
                    M->e2 = fmax(0, delta - eta) / (1 + c1);
                    if (verbose > 0) printf("bm: the bound checked: Z + %.2e I factors\n", delta);
                }
                if (vr == 0 && (quick ? verbose > 0 : verbose >= 0)) printf("bm: Z + %.2e I does not factor: lambda_min is below the Lanczos estimate %.2e%s\n", delta, M->lmin, quick ? " (the iterations go on)" : "");
                if (vr != 0 && M->e2 <= tol && M->cgap <= tol) { rc = 0; break; }
                if (vr == 0) { M->cgap = 1; if (quick) nquick++; else nver++; if (nver >= 3) break; }   /* (the two quick checks of the dense case: not an attempt) */
            }
            if (tlim > 0 && wtime() - t0 > tlim) break;
            /* a saddle (negative curvature dominates the gradient) or a rank too small: escape */
            if (!(fresh && round == 0) && M->lmin < 0 && -M->lmin * rnorm > 10 * gn && nesc < 40) { escape = 1; break; }
            if (!g_bm_newton || round >= (g_bm_rndfac > 0.05 ? 30 : 12)) { escape = M->lmin < 0 && nesc < 40; break; }
            const double tt = wtime();
            ntr += fd_newton(s, g_bm_rndfac * 2 * gn, 200, tlim, t0, verbose);
            g_bm_ttr += wtime() - tt;
        }
        if (rc == 0 || rc == 2 || !escape || (tlim > 0 && wtime() - t0 > tlim)) break;
        if (g_bm_escp > 0) fd_escape(s, v, rnorm, M->pobj, verbose);
        else { add_col(s, v, 0.3 * rnorm); }
        nesc++; fresh = 1;   /* a trust-region round before the next escape (back-to-back escapes looped on amazon0505) */
        dosweep = g_bm_escsweep || g_bm_escp <= 0;
        if (dosweep) f = sweep(s);
    }
    if (verbose > 0) printf("bm: time in the sweeps %.2fs, the trust region %.2fs, Lanczos %.2fs, the bound's factorizations %.2fs (%d); %d escapes\n", g_bm_tsw, g_bm_ttr, g_bm_tlz, g_bm_tvf, g_bm_nvf, nesc);
    memcpy(yout, y, sizeof(double) * P->m);
    free(v); free(y);
    return rc;
}


/* ---- general constraints: augmented Lagrangian, L-BFGS, exact line search ------------------------- */

/* rows of A to unit 2-norm (d_i), b then to unit norm (bs), C to unit Frobenius norm (cs): the problem
 * the method sees; X = bs X~, y_i = cs d_i y~_i, <C,X> = cs bs <C~,X~> */
static void bm_scale(Problem *P) {
    const int m = P->m;
    double *nrm2 = bz(sizeof(double) * (m + 1)), c2 = 0;
    for (int k = 0; k < P->nblk; k++) {
        Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) nrm2[B->con[t]] += B->A[t].fv[q] * B->A[t].fv[q];
        for (int q = 0; q < B->C.ef; q++) c2 += B->C.fv[q] * B->C.fv[q];
    }
    P->normC2 = sqrt(c2);
    P->d = bx(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) P->d[i] = nrm2[i] > 0 ? 1.0 / sqrt(nrm2[i]) : 1.0;
    free(nrm2);
    double nb = 0; for (int i = 0; i < m; i++) nb += P->d[i] * P->b[i] * P->d[i] * P->b[i];
    nb = sqrt(nb); P->bs = nb > 1e-300 ? nb : 1.0;
    for (int k = 0; k < P->nblk; k++) { Block *B = &P->blk[k];
        for (int t = 0; t < B->ncon; t++) { SpSym *A = &B->A[t]; const double f = P->d[B->con[t]]; for (int q = 0; q < A->nnz; q++) A->val[q] *= f; for (int q = 0; q < A->ef; q++) A->fv[q] *= f; } }
    for (int i = 0; i < m; i++) P->b[i] *= P->d[i] / P->bs;
    P->cs = P->normC2 > 1e-300 ? P->normC2 : 1.0;
    for (int k = 0; k < P->nblk; k++) { SpSym *C = &P->blk[k].C; for (int q = 0; q < C->nnz; q++) C->val[q] /= P->cs; for (int q = 0; q < C->ef; q++) C->fv[q] /= P->cs; }
    P->scaled = 1;
}

/* the real roots of c0 + c1 a + c2 a^2 + c3 a^3 (up to 3), returned in r[]; their count */
static int cubic_roots(double c0, double c1, double c2, double c3, double *r) {
    const double sc = fmax(fmax(fabs(c0), fabs(c1)), fmax(fabs(c2), fabs(c3)));
    if (sc == 0) return 0;
    if (fabs(c3) <= 1e-14 * sc) {                       /* quadratic */
        if (fabs(c2) <= 1e-14 * sc) { if (c1 == 0) return 0; r[0] = -c0 / c1; return 1; }
        const double d = c1 * c1 - 4 * c2 * c0;
        if (d < 0) return 0;
        const double q = -0.5 * (c1 + copysign(sqrt(d), c1));
        int k = 0; if (q != 0) r[k++] = c0 / q; r[k++] = q / c2; return k;
    }
    const double a = c2 / c3, b = c1 / c3, c = c0 / c3;
    const double Q = (a * a - 3 * b) / 9, Rr = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
    if (Rr * Rr < Q * Q * Q) {
        const double th = acos(Rr / sqrt(Q * Q * Q)), sq = -2 * sqrt(Q);
        r[0] = sq * cos(th / 3) - a / 3; r[1] = sq * cos((th + 2 * M_PI) / 3) - a / 3; r[2] = sq * cos((th - 2 * M_PI) / 3) - a / 3;
        return 3;
    }
    const double A = -copysign(cbrt(fabs(Rr) + sqrt(Rr * Rr - Q * Q * Q)), Rr), Bq = A != 0 ? Q / A : 0;
    r[0] = A + Bq - a / 3; return 1;
}

typedef struct {
    BB *s; const Problem *P; int m;
    double *y, *t, *t1, *t2, *w, sigma;
    /* LP variables x_j = v_j^2 (stacked after R): their constraints (CSR) and costs */
    int nl, *lp_ptr, *lp_con; double *lp_val, *lp_c;
    size_t N;                 /* the length of R (n r); v at z + N */
} GA;

/* t = A(R R') + A_lp(v^2) - b; returns <C, R R'> + c'v^2 (z = [R; v]) */
static double ga_resid(GA *g, const double *R) {
    BB *s = g->s;
    for (int i = 0; i < g->m; i++) g->t[i] = -g->P->b[i];
    pat_uv(s, R, NULL);
    double f = pat_apply(s, g->t);
    const double *v = R + g->N;
    for (int j = 0; j < g->nl; j++) { const double x = v[j] * v[j]; f += g->lp_c[j] * x; for (int e = g->lp_ptr[j]; e < g->lp_ptr[j + 1]; e++) g->t[g->lp_con[e]] += g->lp_val[e] * x; }
    return f;
}
/* L and its gradient 2 (C - A*(w)) R with w = y - sigma t (t current) */
static double ga_lag(GA *g, double f, const double *R, double *Gr) {
    double L = f;
    for (int i = 0; i < g->m; i++) { L += -g->y[i] * g->t[i] + 0.5 * g->sigma * g->t[i] * g->t[i]; g->w[i] = g->y[i] - g->sigma * g->t[i]; }
    pat_slack(g->s, g->w);
    pat_mul(g->s, g->s->ws, R, Gr, 2.0);
    const double *v = R + g->N; double *gv = Gr + g->N;
    for (int j = 0; j < g->nl; j++) { double sj = g->lp_c[j]; for (int e = g->lp_ptr[j]; e < g->lp_ptr[j + 1]; e++) sj -= g->w[g->lp_con[e]] * g->lp_val[e]; gv[j] = 2 * sj * v[j]; }
    return L;
}
/* the exact step along D: phi(a) = L(R + a D) is a quartic; returns its minimizer over (0, amax] */
static double ga_step(GA *g, const double *R, const double *D, double amax, double *phi0, double *dphi) {
    BB *s = g->s; const int m = g->m;
    memset(g->t1, 0, sizeof(double) * m); memset(g->t2, 0, sizeof(double) * m);
    pat_uv(s, R, D); double o1 = 2 * pat_apply(s, g->t1);
    for (int i = 0; i < m; i++) g->t1[i] *= 2;
    pat_uv(s, D, NULL); double o2 = pat_apply(s, g->t2);
    {   const double *v = R + g->N, *dv = D + g->N;   /* (v + a d)^2 = v^2 + 2 a v d + a^2 d^2 */
        for (int j = 0; j < g->nl; j++) {
            const double x1 = 2 * v[j] * dv[j], x2 = dv[j] * dv[j];
            o1 += g->lp_c[j] * x1; o2 += g->lp_c[j] * x2;
            for (int e = g->lp_ptr[j]; e < g->lp_ptr[j + 1]; e++) { g->t1[g->lp_con[e]] += g->lp_val[e] * x1; g->t2[g->lp_con[e]] += g->lp_val[e] * x2; }
        } }
    double Y1 = 0, Y2 = 0, A0 = 0, A1 = 0, A2 = 0, A3 = 0, R00 = 0;
    for (int i = 0; i < m; i++) {
        const double r0 = g->t[i], r1 = g->t1[i], r2 = g->t2[i];
        Y1 += g->y[i] * r1; Y2 += g->y[i] * r2;
        A0 += r0 * r1; A1 += r1 * r1 + 2 * r0 * r2; A2 += 3 * r1 * r2; A3 += 2 * r2 * r2; R00 += r0 * r0;
    }
    const double sg = g->sigma;
    const double c0 = o1 - Y1 + sg * A0, c1 = 2 * o2 - 2 * Y2 + sg * A1, c2 = sg * A2, c3 = sg * A3;
    *dphi = c0;
    /* phi(a) - phi(0) = c0 a + c1 a^2/2 + c2 a^3/3 + c3 a^4/4 */
    double rt[3]; const int nr = cubic_roots(c0, c1, c2, c3, rt);
    double best = 0, bv = 0;
    for (int k = 0; k < nr; k++) {
        const double a = rt[k];
        if (!(a > 0) || a > amax) continue;
        const double v = c0 * a + c1 * a * a / 2 + c2 * a * a * a / 3 + c3 * a * a * a * a / 4;
        if (v < bv) { bv = v; best = a; }
    }
    if (best == 0 && c0 < 0) {                           /* no interior root (amax reached): the end point if it decreases */
        const double a = amax, v = c0 * a + c1 * a * a / 2 + c2 * a * a * a / 3 + c3 * a * a * a * a / 4;
        if (v < 0) best = a;
    }
    *phi0 = bv;
    return best;
}

/* the general method on one block; returns 0 when the tolerance is met */
static int gen_alm(BB *s, const Problem *P, GA *LPd, double tol, int verbose, double tlim, double t0, BMMeas *M, double *yout) {
    const int n = s->n, m = P->m;
    double c1 = 0, b1 = 0, cF = 0;
    for (int p = 0; p < s->np; p++) { c1 += fabs(s->cv[p]); cF += s->cv[p] * s->cv[p]; }
    for (int i = 0; i < m; i++) b1 += fabs(P->b[i]);
    cF = sqrt(cF);
    double b1o = 0, c1o = c1; if (P->scaled) { for (int i = 0; i < m; i++) b1o += fabs(P->b0[i]); c1o = c1 * P->cs; }
    const double fsc = P->scaled ? P->cs * P->bs : 1.0;
    GA G; memset(&G, 0, sizeof G);
    G.s = s; G.P = P; G.m = m;
    if (LPd) { G.nl = LPd->nl; G.lp_ptr = LPd->lp_ptr; G.lp_con = LPd->lp_con; G.lp_val = LPd->lp_val; G.lp_c = LPd->lp_c; }
    for (int j = 0; j < G.nl; j++) c1 += fabs(G.lp_c[j]);
    G.y = bz(sizeof(double) * (m + 1)); G.t = bx(sizeof(double) * (m + 1)); G.t1 = bx(sizeof(double) * (m + 1)); G.t2 = bx(sizeof(double) * (m + 1)); G.w = bx(sizeof(double) * (m + 1));
    double *v = bx(sizeof(double) * n);
    const int MEM = 8;
    G.N = (size_t)n * s->r;
    size_t N = G.N + G.nl, cap = N + (size_t)n * 8;
    {   double *z = bx(sizeof(double) * cap); memcpy(z, s->R, sizeof(double) * G.N); for (int j = 0; j < G.nl; j++) z[G.N + j] = 1.0; free(s->R); s->R = z; }
    double *Gr = bx(sizeof(double) * cap), *Gn = bx(sizeof(double) * cap), *D = bx(sizeof(double) * cap), *Rn = bx(sizeof(double) * cap);
    double *Sv = bx(sizeof(double) * cap * MEM), *Yv = bx(sizeof(double) * cap * MEM), rhoh[16], alh[16];
    /* scale R so that |A(RR')| ~ |b| */
    {   double f = ga_resid(&G, s->R); (void)f; double ab = 0, bb = 0;
        for (int i = 0; i < m; i++) { const double ax = G.t[i] + P->b[i]; ab += ax * ax; bb += P->b[i] * P->b[i]; }
        if (ab > 0 && bb > 0) { const double k = sqrt(sqrt(bb / ab)); for (size_t q = 0; q < N; q++) s->R[q] *= k; } }
    G.sigma = g_bm_sigma0 > 0 ? g_bm_sigma0 : 1.0 / sqrt((double)n);
    double pinf_prev = 1e300; int rc = 1, nesc = 0, nstall = 0; long nit = 0, nit_prev = -1;
    /* 31 October 2026: the attempt ends when the penalty rises and the primal error does not fall. The escape below adds a
     * column only near feasibility (pinf <= 10 tol), which a factor of too small a rank never reaches: its missing rank
     * shows as a primal error that the penalty cannot lower (the benchmark's MovieLens completions, n = 1,500, m = 37,196:
     * rank 24 against an optimal X of rank 131, pinf 1.7e-3 from the 10th outer iteration to the 182nd, 600 s; four such
     * files 600-1,200 s each). Growing the rank on such a stall was tried: feasible but slow at the end (a synthetic
     * completion of the same structure, n = 1,500: 813 s against 67 s with the stop, 580 s before), so the method that
     * follows takes over instead. The test: over the last five outer iterations the penalty rose 32 times and the primal
     * error did not halve. Only in the automatic attempt (another method follows): with -bm 1 the user asked for this
     * method, and the escapes still make slow progress (a correlation completion, n = 2,000: gap 2.2e-4 after 300 s,
     * rank 26 -> 42). BRISK_BMNOSTALL=1: no test. */
    double hist_p[6], hist_s[6]; int nhist = 0;
    const int stall_test = g_bm_stall_stop && !getenv("BRISK_BMNOSTALL");
    const double sigma0 = G.sigma;
    for (int outer = 0; outer < 300; outer++) {
        /* inner: L-BFGS */
        double f = ga_resid(&G, s->R), L = ga_lag(&G, f, s->R, Gr);
        double g0 = sqrt(dotp(Gr, Gr, N)); int nh = 0, head = 0;
        const double gstop = fmax(1e-2 * g0, 1e-1 * tol * (1 + cF));
        for (int k = 0; k < 500; k++) {
            const double gn = sqrt(dotp(Gr, Gr, N));
            if (gn <= gstop) break;
            /* two-loop recursion: D = -H Gr */
            axpby(N, -1.0, Gr, 0.0, D);
            for (int a = 0; a < nh; a++) { const int j = (head - 1 - a + MEM) % MEM; alh[j] = rhoh[j] * dotp(Sv + (size_t)j * cap, D, N); axpby(N, -alh[j], Yv + (size_t)j * cap, 1.0, D); }
            if (nh) { const int j = (head - 1 + MEM) % MEM; const double gam = dotp(Sv + (size_t)j * cap, Yv + (size_t)j * cap, N) / dotp(Yv + (size_t)j * cap, Yv + (size_t)j * cap, N); axpby(N, 0.0, D, gam, D); }
            for (int a = nh - 1; a >= 0; a--) { const int j = (head - 1 - a + MEM) % MEM; const double be = rhoh[j] * dotp(Yv + (size_t)j * cap, D, N); axpby(N, alh[j] - be, Sv + (size_t)j * cap, 1.0, D); }
            if (dotp(D, Gr, N) >= 0) { axpby(N, -1.0, Gr, 0.0, D); nh = 0; }
            double ph0, dph; const double a = ga_step(&G, s->R, D, 1e6, &ph0, &dph);
            if (a <= 0) { if (nh == 0) break; nh = 0; continue; }
            memcpy(Rn, s->R, sizeof(double) * N); axpby(N, a, D, 1.0, Rn);
            f = ga_resid(&G, Rn); L = ga_lag(&G, f, Rn, Gn);
            {   const int j = head; double *sj = Sv + (size_t)j * cap, *yj = Yv + (size_t)j * cap;
                axpby(N, a, D, 0.0, sj); memcpy(yj, Gn, sizeof(double) * N); axpby(N, -1.0, Gr, 1.0, yj);
                const double sy = dotp(sj, yj, N);
                if (sy > 1e-300) { rhoh[j] = 1 / sy; head = (head + 1) % MEM; if (nh < MEM) nh++; } }
            memcpy(s->R, Rn, sizeof(double) * N); memcpy(Gr, Gn, sizeof(double) * N);
            nit++;
            if (tlim > 0 && wtime() - t0 > tlim) break;
        }
        (void)L;
        /* multipliers */
        double tn = 0; for (int i = 0; i < m; i++) tn += G.t[i] * G.t[i];
        tn = sqrt(tn);
        double pinf = tn / (1 + b1);
        if (P->scaled) { double s2 = 0; for (int i = 0; i < m; i++) { const double r = G.t[i] * P->bs / P->d[i]; s2 += r * r; } pinf = sqrt(s2) / (1 + b1o); }
        for (int i = 0; i < m; i++) G.y[i] -= G.sigma * G.t[i];
        /* measures: Z = C - A*(y) */
        pat_slack(s, G.y);
        double dobj = 0; for (int i = 0; i < m; i++) dobj += P->b[i] * G.y[i];
        const int check = pinf <= 10 * tol || outer % 5 == 4;
        double lm = 0;
        if (check) lm = lmin_est(s, s->ws, v, c1 / (s->np ? s->np : 1));
        double smin = 0;   /* the LP part of the dual: s_j = c_j - (A'y)_j >= 0 */
        for (int j = 0; j < G.nl; j++) { double sj = G.lp_c[j]; for (int e = G.lp_ptr[j]; e < G.lp_ptr[j + 1]; e++) sj -= G.y[G.lp_con[e]] * G.lp_val[e]; if (sj < smin) smin = sj; }
        if (check && smin < lm) lm = smin;
        M->pobj = f * fsc; M->dobj = dobj * fsc; M->e1 = pinf; M->lmin = lm * (P->scaled ? P->cs : 1.0); M->e2 = fmax(0, -M->lmin) / (1 + c1o); M->e3 = fabs(f - dobj) / (1 + fabs(f) + fabs(dobj));
        M->e3 = fabs(M->pobj - M->dobj) / (1 + fabs(M->pobj) + fabs(M->dobj));
        M->lb = M->dobj; M->cgap = M->e3;
        if (verbose >= 0 && g_log_brief) {
            if (outer == 0) log_user(" ITE  OBJECTIVE          RANK  PFEAS    GAP      MIN EIG(Z)  TIME\n");
            log_user("%4d  %+.10e  %4d  %.1e  %.1e  %+.2e %6.2f\n", outer, M->pobj, s->r, pinf, M->e3, M->lmin, wtime() - t0);
        } else if (verbose >= 0) printf("bm: alm %3d  its %6ld  rank %3d  sigma %.1e  obj %.10e  pinf %.1e  gap %.1e  lmin %s%.2e  %.2fs\n", outer, nit, s->r, G.sigma, M->pobj, pinf, M->e3, check ? "" : "~", M->lmin, wtime() - t0);
        if (check && pinf <= tol && M->e2 <= tol && M->e3 <= tol) { rc = 0; break; }
        if (tlim > 0 && wtime() - t0 > tlim) break;
        if (check && pinf <= 10 * tol && lm < 0 && M->e2 > 10 * tol && nesc < 20) {
            /* the rank is too small: a column along the eigenvector */
            const int r = s->r, r1 = r + 1;
            const size_t N1 = (size_t)n * r1 + G.nl;
            double *Rx = bx(sizeof(double) * (N1 > cap ? N1 + (size_t)n * 8 : cap)), rn = sqrt(dotp(s->R, s->R, G.N) / n);
            for (int i = 0; i < n; i++) { memcpy(Rx + (size_t)i * r1, s->R + (size_t)i * r, sizeof(double) * r); Rx[(size_t)i * r1 + r] = 0.1 * rn * sqrt((double)n) * v[i]; }
            memcpy(Rx + (size_t)n * r1, s->R + G.N, sizeof(double) * G.nl);
            free(s->R); s->R = Rx; s->r = r1; G.N = (size_t)n * r1; N = N1; nesc++;
            if (N > cap) { cap = N + (size_t)n * 8; Gr = realloc(Gr, sizeof(double) * cap); Gn = realloc(Gn, sizeof(double) * cap); D = realloc(D, sizeof(double) * cap); Rn = realloc(Rn, sizeof(double) * cap); free(Sv); free(Yv); Sv = bx(sizeof(double) * cap * MEM); Yv = bx(sizeof(double) * cap * MEM); }
            continue;
        }
        /* the stall test over the last five outer iterations (see above) */
        if (nhist == 6) { memmove(hist_p, hist_p + 1, sizeof(double) * 5); memmove(hist_s, hist_s + 1, sizeof(double) * 5); nhist = 5; }
        hist_p[nhist] = pinf; hist_s[nhist] = G.sigma; nhist++;
        if (stall_test && nhist == 6 && G.sigma >= 32.0 * hist_s[0] && pinf >= 0.5 * hist_p[0]) {
            if (verbose >= 0 && !g_log_brief) printf("bm: the primal error stays at %.1e while the penalty rose %.0fx: the rank (%d) is too small for this problem; stopping\n", pinf, G.sigma / hist_s[0], s->r);
            break;
        }
        if (pinf > 0.25 * pinf_prev && G.sigma < 1e8 * sigma0) G.sigma *= g_bm_sigfac;
        if (nit == nit_prev && ++nstall >= 3) break;   /* the inner solver no longer moves */
        if (nit != nit_prev) nstall = 0;
        nit_prev = nit;
        pinf_prev = pinf;
    }
    for (int i = 0; i < m; i++) yout[i] = P->scaled ? P->cs * P->d[i] * G.y[i] : G.y[i];
    free(G.y); free(G.t); free(G.t1); free(G.t2); free(G.w); free(v); free(Gr); free(Gn); free(D); free(Rn); free(Sv); free(Yv);
    return rc;
}

/* 15 November 2026 (the owner: the Burer-Monteiro method for several blocks; ISSUES 898). The general method (the augmented
 * Lagrangian minimized by L-BFGS with the exact line search of gen_alm) on any number of semidefinite blocks and LP
 * variables. Each block k has its own factor R_k (n_k x r_k, its own rank) and its own pattern (bb_setup); the unknowns are
 * one vector z = [R_1 | ... | R_K | v], so the L-BFGS and the line search see one problem: the residual, the objective and
 * the line search's quartic are sums over the blocks, the gradient is block by block (2 (C_k - A_k*(w)) R_k). LP variables
 * are x = v^2; a free variable written as a split pair (columns a, -a, costs c, -c) is one variable x = u with no square
 * (as two squares v+^2 - v-^2 it has no unique minimizer and both parts grow), the '-' slot fixed at 0. The rank of each
 * block grows on its own: near feasibility a block whose dual slack Z_k = C_k - A_k*(y) has a negative eigenvalue gets a
 * column along its eigenvector. The first ranks: min(n_k, ceil(2 ln(m_k + 1)) + 2), m_k the constraints touching the
 * block (BRISK_BMRANK: the same rank everywhere, at most n_k). The measures: the primal residual over all rows, the dual
 * cone error min over the blocks of lambda_min(Z_k) and over the LP slacks, the free variables' dual residuals
 * |c_f - a_f'y| (in the dual error), the relative gap. Reached only by -bm 1 on a problem of several blocks: the automatic
 * route takes one block only, as before, and the one-block path is unchanged. */
typedef struct {
    int nbk; BB **S; size_t *off;     /* block k's R_k at z + off[k] */
    const Problem *P; int m;
    double *y, *t, *t1, *t2, *w, sigma;
    int nl, *lp_ptr, *lp_con; double *lp_val, *lp_c; unsigned char *lp_kind;   /* LP variables: kind 0 x = v^2, 1 free x = v, 2 fixed at 0 */
    size_t N;                         /* sum n_k r_k; v at z + N */
    double so1, so2;                  /* the last line search's objective terms: f(z + a D) = f + a so1 + a^2 so2 */
    double **xs2;                     /* per block: a second pattern buffer (the fused line search) */
    struct GU **U;                    /* per block: the upper triangle of the pattern for the line search (NULL: the full one) */
} GM;
/* 16 November 2026: the line search's two pattern products on the upper triangle of the pattern, each unordered pair
 * once: x1 = u_i'd_j + u_j'd_i (u_i'd_i on the diagonal) and x2 = 2 d_i'd_j (d_i'd_i), against the entries a_ij of
 * i <= j, so that sum_{i<=j} a_ij x1_ij = <A, U D'> and sum a_ij x2_ij = <A, D D'> as the full pattern gives them: three
 * r-vector products per pair instead of four, and half the index and value traffic. BRISK_BMUPPER=0: the full pattern */
typedef struct GU { int n, np, ncon; const int *con; int *sp, *sj, *cp, *ep; double *ev, *cv, *x1, *x2;
                   int *tp, *tc; double *tv; } GU;   /* (tp, tc, tv: the full pattern's entries by position, for the dual slack) */
static GU *gu_build(const BB *s) {
    const int n = s->n;
    int *umap = bx(sizeof(int) * ((size_t)s->np + 1)), *prow = bx(sizeof(int) * ((size_t)s->np + 1));
    GU *u = bz(sizeof(GU)); u->n = n; u->ncon = s->ncon; u->con = s->con;
    u->sp = bz(sizeof(int) * ((size_t)n + 1));
    int nu = 0;
    for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { prow[p] = i; if (s->sj[p] >= i) { umap[p] = nu++; u->sp[i + 1]++; } else umap[p] = -1; }
    for (int i = 0; i < n; i++) u->sp[i + 1] += u->sp[i];
    u->np = nu; u->sj = bx(sizeof(int) * ((size_t)nu + 1)); u->cv = bz(sizeof(double) * ((size_t)nu + 1));
    u->x1 = bx(sizeof(double) * ((size_t)nu + 1)); u->x2 = bx(sizeof(double) * ((size_t)nu + 1));
    for (int p = 0; p < s->np; p++) if (umap[p] >= 0) { u->sj[umap[p]] = s->sj[p]; u->cv[umap[p]] = s->cv[p]; }
    u->cp = bx(sizeof(int) * ((size_t)s->ncon + 1)); u->cp[0] = 0;
    size_t ne = 0; for (int e = 0; e < s->cp[s->ncon]; e++) if (umap[s->ep[e]] >= 0) ne++;
    u->ep = bx(sizeof(int) * (ne + 1)); u->ev = bx(sizeof(double) * (ne + 1));
    size_t w = 0;
    for (int t = 0; t < s->ncon; t++) {
        for (int e = s->cp[t]; e < s->cp[t + 1]; e++) { const int q = umap[s->ep[e]]; if (q < 0) continue; u->ep[w] = q; u->ev[w] = s->ev[e]; w++; }
        u->cp[t + 1] = (int)w;
    }
    free(umap); free(prow);
    /* the dual slack ws = C - sum_t w_t A_t by position, in parallel (pat_slack scatters by constraint, on one thread): the
     * entries of each position in the order of the constraints, so the sum and its bits do not depend on the threads */
    {   const size_t nf = (size_t)s->cp[s->ncon];
        u->tp = bz(sizeof(int) * ((size_t)s->np + 2)); u->tc = bx(sizeof(int) * (nf + 1)); u->tv = bx(sizeof(double) * (nf + 1));
        for (size_t e = 0; e < nf; e++) u->tp[s->ep[e] + 1]++;
        for (int p = 0; p < s->np; p++) u->tp[p + 1] += u->tp[p];
        int *fill = bx(sizeof(int) * ((size_t)s->np + 1)); memcpy(fill, u->tp, sizeof(int) * (size_t)s->np);
        for (int t = 0; t < s->ncon; t++) for (int e = s->cp[t]; e < s->cp[t + 1]; e++) { const int q = fill[s->ep[e]]++; u->tc[q] = s->con[t]; u->tv[q] = s->ev[e]; }
        free(fill); }
    return u;
}
static void gu_slack(const GU *u, const BB *s, const double *w) {
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < s->np; p++) { double v = s->cv[p]; for (int q = u->tp[p]; q < u->tp[p + 1]; q++) v -= w[u->tc[q]] * u->tv[q]; s->ws[p] = v; }
}
static void gu_free(GU *u) { if (!u) return; free(u->tp); free(u->tc); free(u->tv); free(u->sp); free(u->sj); free(u->cp); free(u->ep); free(u->ev); free(u->cv); free(u->x1); free(u->x2); free(u); }
/* out1 += <A_t, U D'>, out2 += <A_t, D D'>; c1 = <C, U D'>, c2 = <C, D D'> (as gm_uv2 and gm_apply2 on the full pattern) */
static void gu_step(const GU *u, int r, const double *U, const double *D, double *out1, double *out2, double *c1, double *c2) {
    const int n = u->n;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        const double *ui = U + (size_t)i * r, *di = D + (size_t)i * r;
        for (int p = u->sp[i]; p < u->sp[i + 1]; p++) {
            const int j = u->sj[p];
            const double *uj = U + (size_t)j * r, *dj = D + (size_t)j * r;
            if (j == i) { double v1 = 0, v2 = 0; for (int l = 0; l < r; l++) { v1 += ui[l] * di[l]; v2 += di[l] * di[l]; } u->x1[p] = v1; u->x2[p] = v2; continue; }
            double v1 = 0, v2 = 0; for (int l = 0; l < r; l++) { v1 += ui[l] * dj[l] + uj[l] * di[l]; v2 += di[l] * dj[l]; }
            u->x1[p] = v1; u->x2[p] = 2 * v2;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < u->ncon; t++) { double v1 = 0, v2 = 0; for (int e = u->cp[t]; e < u->cp[t + 1]; e++) { v1 += u->ev[e] * u->x1[u->ep[e]]; v2 += u->ev[e] * u->x2[u->ep[e]]; } out1[u->con[t]] += v1; out2[u->con[t]] += v2; }
    double a = 0, b = 0;
    for (int p = 0; p < u->np; p++) { a += u->cv[p] * u->x1[p]; b += u->cv[p] * u->x2[p]; }
    *c1 = a; *c2 = b;
}
/* 15 November 2026: the pattern products of a block whose pattern is dense (np >= g_gm_dense n^2, n <= 8,000): U V' by one
 * BLAS-3 product (pdgemm, the same bits at any thread count) and its pattern entries gathered, W U by W scattered into a
 * dense matrix and one product; the sparse loops of pat_uv and pat_mul gather r-vectors for every pattern entry (memory
 * bound). Used by the multi-block method only (the one-block path keeps its arithmetic). BRISK_BMDENSE=<f>: the density
 * from which (0: never) */
static double g_gm_dense = -1;
static double *g_gm_buf = NULL; static size_t g_gm_cap = 0;
static double *gm_buf(size_t n) { if (n > g_gm_cap) { free(g_gm_buf); g_gm_buf = bx(sizeof(double) * n); g_gm_cap = n; } return g_gm_buf; }
static int gm_isdense(const BB *s) {
    if (g_gm_dense < 0) { const char *e = getenv("BRISK_BMDENSE"); g_gm_dense = e ? atof(e) : 0.0; }   /* (off: slower on NH4+, 2,619 against 3,141 L-BFGS steps in 60 s; BRISK_BMDENSE=0.1 on) */
    return g_gm_dense > 0 && s->n >= 32 && s->n <= 8000 && (double)s->np >= g_gm_dense * (double)s->n * s->n;
}
static void gm_uv(const BB *s, const double *U, const double *V) {
    if (!gm_isdense(s)) { pat_uv(s, U, V); return; }
    const int n = s->n, r = s->r; double *X = gm_buf((size_t)n * n);
    const double one = 1.0, zero = 0.0;
    pdgemm("T", "N", &n, &n, &r, &one, U, &r, V ? V : U, &r, &zero, X, &n);   /* X(i, j) = u_i . v_j */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) for (int p = s->sp[i]; p < s->sp[i + 1]; p++) s->xs[p] = X[i + (size_t)s->sj[p] * n];
}
static void gm_mul(const BB *s, const double *W, const double *U, double *O, double c) {
    if (!gm_isdense(s) || s->jP != 0) { pat_mul(s, W, U, O, c); return; }
    const int n = s->n, r = s->r; double *Wd = gm_buf((size_t)n * n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { double *col = Wd + (size_t)i * n; memset(col, 0, sizeof(double) * n); for (int p = s->sp[i]; p < s->sp[i + 1]; p++) col[s->sj[p]] = W[p]; }
    const double zero = 0.0;
    pdgemm("N", "N", &r, &n, &n, &c, U, &r, Wd, &n, &zero, O, &r);            /* O' = c U' W (W symmetric) */
}
/* the line search's two pattern passes in one: xs = (U D')_ij, x2 = (D D')_ij on the pattern (sparse blocks; the dense
 * form makes the two products) */
static void gm_uv2(const BB *s, const double *U, const double *D, double *x2) {
    if (gm_isdense(s)) { gm_uv(s, U, D); double *keep = s->xs; ((BB *)s)->xs = x2; gm_uv(s, D, NULL); ((BB *)s)->xs = keep; return; }
    const int n = s->n, r = s->r;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *ui = U + (size_t)i * r, *di = D + (size_t)i * r;
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) {
            const size_t j = (size_t)s->sj[p] * r; const double *dj = D + j;
            double v1 = 0, v2 = 0; for (int l = 0; l < r; l++) { v1 += ui[l] * dj[l]; v2 += di[l] * dj[l]; }
            s->xs[p] = v1; x2[p] = v2;
        }
    }
}
/* out1 += <A_t, xs>, out2 += <A_t, x2>; returns <C, xs> and <C, x2> */
static void gm_apply2(const BB *s, const double *x2, double *out1, double *out2, double *c1, double *c2) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < s->ncon; t++) { double v1 = 0, v2 = 0; for (int e = s->cp[t]; e < s->cp[t + 1]; e++) { v1 += s->ev[e] * s->xs[s->ep[e]]; v2 += s->ev[e] * x2[s->ep[e]]; } out1[s->con[t]] += v1; out2[s->con[t]] += v2; }
    double a = 0, b = 0;
    for (int p = 0; p < s->np; p++) { a += s->cv[p] * s->xs[p]; b += s->cv[p] * x2[p]; }
    *c1 = a; *c2 = b;
}
static void gm_layout(GM *g) { size_t o = 0; for (int k = 0; k < g->nbk; k++) { g->off[k] = o; o += (size_t)g->S[k]->n * g->S[k]->r; } g->N = o; }
static inline double gm_x(const GM *g, int j, double v) { return g->lp_kind[j] == 0 ? v * v : g->lp_kind[j] == 1 ? v : 0.0; }
static double g_gmt[6];   /* (BRISK_BMPROF=1: resid, lag, step, lambda_min, L-BFGS vector work) */
static double gm_resid(GM *g, const double *z) {
    const double t0_ = wtime();
    for (int i = 0; i < g->m; i++) g->t[i] = -g->P->b[i];
    double f = 0;
    for (int k = 0; k < g->nbk; k++) { gm_uv(g->S[k], z + g->off[k], NULL); f += pat_apply(g->S[k], g->t); }
    const double *v = z + g->N;
    for (int j = 0; j < g->nl; j++) { const double x = gm_x(g, j, v[j]); if (x == 0) continue; f += g->lp_c[j] * x; for (int e = g->lp_ptr[j]; e < g->lp_ptr[j + 1]; e++) g->t[g->lp_con[e]] += g->lp_val[e] * x; }
    g_gmt[0] += wtime() - t0_;
    return f;
}
static double gm_lag(GM *g, double f, const double *z, double *Gr) {
    const double t0_ = wtime();
    double L = f;
    for (int i = 0; i < g->m; i++) { L += -g->y[i] * g->t[i] + 0.5 * g->sigma * g->t[i] * g->t[i]; g->w[i] = g->y[i] - g->sigma * g->t[i]; }
    for (int k = 0; k < g->nbk; k++) { const double ts_ = wtime(); if (g->U && g->U[k]) gu_slack(g->U[k], g->S[k], g->w); else pat_slack(g->S[k], g->w); g_gmt[5] += wtime() - ts_; gm_mul(g->S[k], g->S[k]->ws, z + g->off[k], Gr + g->off[k], 2.0); }
    const double *v = z + g->N; double *gv = Gr + g->N;
    for (int j = 0; j < g->nl; j++) {
        if (g->lp_kind[j] == 2) { gv[j] = 0; continue; }
        double sj = g->lp_c[j]; for (int e = g->lp_ptr[j]; e < g->lp_ptr[j + 1]; e++) sj -= g->w[g->lp_con[e]] * g->lp_val[e];
        gv[j] = g->lp_kind[j] == 1 ? sj : 2 * sj * v[j];
    }
    g_gmt[1] += wtime() - t0_;
    return L;
}
static double gm_step(GM *g, const double *z, const double *D, double amax, double *phi0, double *dphi) {
    const double t0_ = wtime();
    const int m = g->m;
    memset(g->t1, 0, sizeof(double) * m); memset(g->t2, 0, sizeof(double) * m);
    double o1 = 0, o2 = 0;
    for (int k = 0; k < g->nbk; k++) {
        double a1, a2;
        if (g->U && g->U[k]) gu_step(g->U[k], g->S[k]->r, z + g->off[k], D + g->off[k], g->t1, g->t2, &a1, &a2);
        else { gm_uv2(g->S[k], z + g->off[k], D + g->off[k], g->xs2[k]); gm_apply2(g->S[k], g->xs2[k], g->t1, g->t2, &a1, &a2); }
        o1 += 2 * a1; o2 += a2;
    }
    for (int i = 0; i < m; i++) g->t1[i] *= 2;
    {   const double *v = z + g->N, *dv = D + g->N;
        for (int j = 0; j < g->nl; j++) {
            if (g->lp_kind[j] == 2) continue;
            const double x1 = g->lp_kind[j] == 1 ? dv[j] : 2 * v[j] * dv[j], x2 = g->lp_kind[j] == 1 ? 0.0 : dv[j] * dv[j];
            o1 += g->lp_c[j] * x1; o2 += g->lp_c[j] * x2;
            for (int e = g->lp_ptr[j]; e < g->lp_ptr[j + 1]; e++) { g->t1[g->lp_con[e]] += g->lp_val[e] * x1; g->t2[g->lp_con[e]] += g->lp_val[e] * x2; }
        } }
    double Y1 = 0, Y2 = 0, A0 = 0, A1 = 0, A2 = 0, A3 = 0;
    for (int i = 0; i < m; i++) {
        const double r0 = g->t[i], r1 = g->t1[i], r2 = g->t2[i];
        Y1 += g->y[i] * r1; Y2 += g->y[i] * r2;
        A0 += r0 * r1; A1 += r1 * r1 + 2 * r0 * r2; A2 += 3 * r1 * r2; A3 += 2 * r2 * r2;
    }
    const double sg = g->sigma;
    const double c0 = o1 - Y1 + sg * A0, c1 = 2 * o2 - 2 * Y2 + sg * A1, c2 = sg * A2, c3 = sg * A3;
    *dphi = c0;
    double rt[3]; const int nr = cubic_roots(c0, c1, c2, c3, rt);
    double best = 0, bv = 0;
    for (int q = 0; q < nr; q++) {
        const double a = rt[q];
        if (!(a > 0) || a > amax) continue;
        const double v = c0 * a + c1 * a * a / 2 + c2 * a * a * a / 3 + c3 * a * a * a * a / 4;
        if (v < bv) { bv = v; best = a; }
    }
    if (best == 0 && c0 < 0) { const double a = amax, v = c0 * a + c1 * a * a / 2 + c2 * a * a * a / 3 + c3 * a * a * a * a / 4; if (v < 0) best = a; }
    *phi0 = bv;
    g->so1 = o1; g->so2 = o2;
    g_gmt[2] += wtime() - t0_;
    return best;
}
/* the method; on return each S[k]->R holds its factor (n_k x r_k, row-major) and v the LP values; returns 0 at the tolerance */
static int gen_alm_multi(GM *G, double **vout, double tol, int verbose, double tlim, double t0, BMMeas *M, double *yout) {
    const Problem *P = G->P; const int m = G->m, nbk = G->nbk;
    double c1 = 0, b1 = 0, cF = 0;
    for (int k = 0; k < nbk; k++) for (int p = 0; p < G->S[k]->np; p++) { const double c = G->S[k]->cv[p]; c1 += fabs(c); cF += c * c; }
    for (int j = 0; j < G->nl; j++) { c1 += fabs(G->lp_c[j]); cF += G->lp_c[j] * G->lp_c[j]; }
    for (int i = 0; i < m; i++) b1 += fabs(P->b[i]);
    cF = sqrt(cF);
    int nmax = 1; for (int k = 0; k < nbk; k++) if (G->S[k]->n > nmax) nmax = G->S[k]->n;
    G->y = bz(sizeof(double) * (m + 1)); G->t = bx(sizeof(double) * (m + 1)); G->t1 = bx(sizeof(double) * (m + 1)); G->t2 = bx(sizeof(double) * (m + 1)); G->w = bx(sizeof(double) * (m + 1));
    G->xs2 = bz(sizeof(double *) * (nbk + 1)); for (int k = 0; k < nbk; k++) G->xs2[k] = bx(sizeof(double) * ((size_t)G->S[k]->np + 1));
    {   const char *e = getenv("BRISK_BMUPPER"); const int up = !(e && atoi(e) == 0);
        G->U = bz(sizeof(GU *) * (nbk + 1));
        for (int k = 0; k < nbk; k++) if (up && !gm_isdense(G->S[k]) && G->S[k]->jP == 0) G->U[k] = gu_build(G->S[k]); }
    double *vk = bx(sizeof(double) * (size_t)nmax * nbk), *lmk = bx(sizeof(double) * (nbk + 1)); int *gw = bz(sizeof(int) * (nbk + 1)); double *lt = bz(sizeof(double) * (nbk + 1));
    const int MEM = 8;
    gm_layout(G);
    size_t N = G->N + G->nl, cap = N + (size_t)nmax * 8 * nbk;
    double *z = bx(sizeof(double) * cap);
    for (int k = 0; k < nbk; k++) { memcpy(z + G->off[k], G->S[k]->R, sizeof(double) * (size_t)G->S[k]->n * G->S[k]->r); }
    for (int j = 0; j < G->nl; j++) z[G->N + j] = G->lp_kind[j] == 0 ? 1.0 : 0.0;
    double *Gr = bx(sizeof(double) * cap), *Gn = bx(sizeof(double) * cap), *D = bx(sizeof(double) * cap), *Rn = bx(sizeof(double) * cap);
    double *Sv = bx(sizeof(double) * cap * MEM), *Yv = bx(sizeof(double) * cap * MEM), rhoh[16], alh[16];
    {   double f = gm_resid(G, z); (void)f; double ab = 0, bb = 0;   /* scale R so that |A(RR')| ~ |b| */
        for (int i = 0; i < m; i++) { const double ax = G->t[i] + P->b[i]; ab += ax * ax; bb += P->b[i] * P->b[i]; }
        if (ab > 0 && bb > 0) { const double kq = sqrt(sqrt(bb / ab)); for (size_t q = 0; q < G->N; q++) z[q] *= kq; for (int j = 0; j < G->nl; j++) if (G->lp_kind[j] == 0) z[G->N + j] *= kq; } }
    double nsum = 0; for (int k = 0; k < nbk; k++) nsum += G->S[k]->n;
    G->sigma = g_bm_sigma0 > 0 ? g_bm_sigma0 : 1.0 / sqrt(nsum);
    double pinf_prev = 1e300; int rc = 1, nesc = 0, nstall = 0; long nit = 0, nit_prev = -1;
    const double sigma0 = G->sigma;
    for (int outer = 0; outer < 300; outer++) {
        double f = gm_resid(G, z), L = gm_lag(G, f, z, Gr);
        double g0 = sqrt(dotp(Gr, Gr, N)); int nh = 0, head = 0;
        const double gstop = fmax(1e-2 * g0, 1e-1 * tol * (1 + cF));
        for (int it = 0; it < 500; it++) {
            const double gn = sqrt(dotp(Gr, Gr, N));
            if (gn <= gstop) break;
            axpby(N, -1.0, Gr, 0.0, D);
            for (int a = 0; a < nh; a++) { const int j = (head - 1 - a + MEM) % MEM; alh[j] = rhoh[j] * dotp(Sv + (size_t)j * cap, D, N); axpby(N, -alh[j], Yv + (size_t)j * cap, 1.0, D); }
            if (nh) { const int j = (head - 1 + MEM) % MEM; const double gam = dotp(Sv + (size_t)j * cap, Yv + (size_t)j * cap, N) / dotp(Yv + (size_t)j * cap, Yv + (size_t)j * cap, N); axpby(N, 0.0, D, gam, D); }
            for (int a = nh - 1; a >= 0; a--) { const int j = (head - 1 - a + MEM) % MEM; const double be = rhoh[j] * dotp(Yv + (size_t)j * cap, D, N); axpby(N, alh[j] - be, Sv + (size_t)j * cap, 1.0, D); }
            if (dotp(D, Gr, N) >= 0) { axpby(N, -1.0, Gr, 0.0, D); nh = 0; }
            double ph0, dph; const double a = gm_step(G, z, D, 1e6, &ph0, &dph);
            if (a <= 0) { if (nh == 0) break; nh = 0; continue; }
            memcpy(Rn, z, sizeof(double) * N); axpby(N, a, D, 1.0, Rn);
            /* the residual and the objective at R + a D from the line search's terms: t + a t1 + a^2 t2 and f + a o1 + a^2 o2
             * hold exactly (every residual is a quadratic in a); a full pass every 20 steps keeps the rounding from drifting */
            if ((it + 1) % 20 == 0) f = gm_resid(G, Rn);
            else { for (int i = 0; i < m; i++) G->t[i] += a * G->t1[i] + a * a * G->t2[i]; f += a * G->so1 + a * a * G->so2; }
            L = gm_lag(G, f, Rn, Gn);
            {   const int j = head; double *sj = Sv + (size_t)j * cap, *yj = Yv + (size_t)j * cap;
                axpby(N, a, D, 0.0, sj); memcpy(yj, Gn, sizeof(double) * N); axpby(N, -1.0, Gr, 1.0, yj);
                const double sy = dotp(sj, yj, N);
                if (sy > 1e-300) { rhoh[j] = 1 / sy; head = (head + 1) % MEM; if (nh < MEM) nh++; } }
            memcpy(z, Rn, sizeof(double) * N); memcpy(Gr, Gn, sizeof(double) * N);
            nit++;
            if (tlim > 0 && wtime() - t0 > tlim) break;
        }
        (void)L;
        double tn = 0; for (int i = 0; i < m; i++) tn += G->t[i] * G->t[i];
        const double pinf = sqrt(tn) / (1 + b1);
        for (int i = 0; i < m; i++) G->y[i] -= G->sigma * G->t[i];
        double dobj = 0; for (int i = 0; i < m; i++) dobj += P->b[i] * G->y[i];
        const int check = pinf <= 10 * tol || outer % 5 == 4;
        double lm = 0;
        for (int k = 0; k < nbk; k++) {
            pat_slack(G->S[k], G->y);
            { const double tl_ = wtime(); lmk[k] = check ? lmin_est(G->S[k], G->S[k]->ws, vk + (size_t)k * nmax, c1 / (G->S[k]->np ? G->S[k]->np : 1)) : 0; g_gmt[3] += wtime() - tl_; }
            if (lmk[k] < lm) lm = lmk[k];
        }
        double smin = 0, rf2 = 0;   /* the LP part of the dual: s_j >= 0 (squared), s_j = 0 (free) */
        for (int j = 0; j < G->nl; j++) {
            if (G->lp_kind[j] == 2) continue;
            double sj = G->lp_c[j]; for (int e = G->lp_ptr[j]; e < G->lp_ptr[j + 1]; e++) sj -= G->y[G->lp_con[e]] * G->lp_val[e];
            if (G->lp_kind[j] == 1) rf2 += sj * sj; else if (sj < smin) smin = sj;
        }
        if (check && smin < lm) lm = smin;
        M->pobj = f; M->dobj = dobj; M->e1 = pinf; M->lmin = lm;
        M->e2 = fmax(fmax(0, -lm) / (1 + c1), sqrt(rf2) / (1 + c1));
        M->e3 = fabs(M->pobj - M->dobj) / (1 + fabs(M->pobj) + fabs(M->dobj));
        M->lb = M->dobj; M->cgap = M->e3;
        int rsum = 0; for (int k = 0; k < nbk; k++) rsum += G->S[k]->r;
        if (verbose >= 0 && g_log_brief) {
            if (outer == 0) log_user(" ITE  OBJECTIVE          RANKS  PFEAS    GAP      MIN EIG(Z)  TIME\n");
            log_user("%4d  %+.10e  %5d  %.1e  %.1e  %+.2e %6.2f\n", outer, M->pobj, rsum, pinf, M->e3, M->lmin, wtime() - t0);
        } else if (verbose >= 0) printf("bm: alm %3d  its %6ld  ranks %4d  sigma %.1e  obj %.10e  pinf %.1e  gap %.1e  dual %s%.1e  %.2fs\n", outer, nit, rsum, G->sigma, M->pobj, pinf, M->e3, check ? "" : "~", M->e2, wtime() - t0);
        if (check && pinf <= tol && M->e2 <= tol && M->e3 <= tol) { rc = 0; break; }
        if (tlim > 0 && wtime() - t0 > tlim) break;
        /* 16 November 2026: a block also grows when the gap misses the tolerance and its negative eigenvalue explains a tenth of
         * the gap allowed: with A(X) = b the gap is <Z, X> >= sum_k lambda_min(Z_k) tr X_k, and lambda_min / (1 + |C|_1) can
         * be far below the tolerance while -lambda_min tr X_k is not (H2O, SDP: the dual measure 4e-6, the gap stalled at
         * 1e-4 for 60 outer iterations with no column added, ranks 708 throughout). BRISK_BMGAPGROW=0: the dual measure alone */
        static int gapgrow = -1; if (gapgrow < 0) { const char *e = getenv("BRISK_BMGAPGROW"); gapgrow = !(e && atoi(e) == 0); }
        const double gap_abs = fabs(M->pobj - M->dobj);
        double eterm = 0;   /* sum_k -lambda_min(Z_k) tr X_k over the blocks with a negative eigenvalue */
        int want = 0;
        for (int k = 0; k < nbk; k++) {
            const BB *s = G->S[k]; gw[k] = 0; lt[k] = 0;
            if (!(check && lmk[k] < 0 && s->r < s->n)) continue;
            if (fmax(0, -lmk[k]) / (1 + c1) > 10 * tol) { gw[k] = 1; continue; }
            if (gapgrow && M->e3 > tol) { const double *Rk = z + G->off[k]; double tr = 0; for (size_t q = 0; q < (size_t)s->n * s->r; q++) tr += Rk[q] * Rk[q]; lt[k] = -lmk[k] * tr; eterm += lt[k]; }
        }
        /* (the eigenvalues explain the gap: at least half of it, each grown block a tenth) */
        if (eterm >= 0.5 * gap_abs) for (int k = 0; k < nbk; k++) if (lt[k] >= 0.1 * gap_abs) gw[k] = 1;
        for (int k = 0; k < nbk; k++) want += gw[k];
        if (check && pinf <= 10 * tol && want > 0 && nesc < 40) {
            /* the rank of some blocks is too small: a column along the eigenvector of each block marked; z laid out again */
            int grown = 0;
            size_t Nn = 0; for (int k = 0; k < nbk; k++) Nn += (size_t)G->S[k]->n * (G->S[k]->r + gw[k]);
            double *zx = bx(sizeof(double) * (Nn + G->nl + (size_t)nmax * 8 * nbk + 1));
            size_t o = 0;
            for (int k = 0; k < nbk; k++) {
                BB *s = G->S[k]; const int n = s->n, r = s->r;
                const int add = gw[k];
                lmk[k] = add ? 1.0 : 0.0;   /* (from here: 1 when the block grows) */
                const double *Rk = z + G->off[k];
                if (!add) { memcpy(zx + o, Rk, sizeof(double) * (size_t)n * r); o += (size_t)n * r; continue; }
                const int r1 = r + 1;
                for (int i = 0; i < n; i++) { memcpy(zx + o + (size_t)i * r1, Rk + (size_t)i * r, sizeof(double) * r); zx[o + (size_t)i * r1 + r] = 0.0; }
                o += (size_t)n * r1; s->r = r1; grown++;
            }
            memcpy(zx + o, z + G->N, sizeof(double) * G->nl);
            free(z); z = zx; gm_layout(G); N = G->N + G->nl; nesc++;
            if (N > cap) { cap = N + (size_t)nmax * 8 * nbk; Gr = realloc(Gr, sizeof(double) * cap); Gn = realloc(Gn, sizeof(double) * cap); D = realloc(D, sizeof(double) * cap); Rn = realloc(Rn, sizeof(double) * cap); free(Sv); free(Yv); Sv = bx(sizeof(double) * cap * MEM); Yv = bx(sizeof(double) * cap * MEM); }
            /* z itself must keep room for a later step (Rn is copied into it): reallocate to cap */
            { double *zz = bx(sizeof(double) * cap); memcpy(zz, z, sizeof(double) * N); free(z); z = zz; }
            {   /* the new columns start at 0 and move along their eigenvectors by the exact line search of the Lagrangian (a quartic
                 * along any direction): v_k is a direction of negative curvature (v_k' Z_k v_k < 0 at the multipliers), so the
                 * step decreases L; a column added at a fixed size (0.1 |R| sqrt(n) v, the one-block path) knocked a near-optimal
                 * iterate away on NH2- (SDP): 62.70661 at the 17th outer iteration, then 63.6 after the escapes */
                memset(D, 0, sizeof(double) * N);
                for (int k = 0; k < nbk; k++) {
                    const BB *s = G->S[k]; const int r = s->r;
                    if (lmk[k] != 1.0) continue;
                    const double *vv = vk + (size_t)k * nmax; double *Dk = D + G->off[k];
                    for (int i = 0; i < s->n; i++) Dk[(size_t)i * r + r - 1] = vv[i];
                }
                (void)gm_resid(G, z);
                double ph0, dph; const double a = gm_step(G, z, D, 1e6, &ph0, &dph);
                if (a > 0) axpby(N, a, D, 1.0, z);
                else { const double eps = 1e-6; axpby(N, eps, D, 1.0, z); }   /* (no decrease along it: a small start, not an exact 0, which the gradient never leaves) */
                if (verbose >= 0 && !g_log_brief) printf("bm: the dual slack of %d block(s) has a negative eigenvalue: a column added to each (ranks %d in all), step %.2e along the eigenvectors\n", grown, rsum + grown, a);
            }
            continue;
        }
        if (pinf > 0.25 * pinf_prev && G->sigma < 1e8 * sigma0) G->sigma *= g_bm_sigfac;
        if (nit == nit_prev && ++nstall >= 3) break;
        if (nit != nit_prev) nstall = 0;
        nit_prev = nit;
        pinf_prev = pinf;
    }
    if (getenv("BRISK_BMPROF")) printf("bm: profile (s): residual %.2f, gradient %.2f, line search %.2f, lambda_min %.2f; total %.2f (the dual slack in the gradient %.2f)\n", g_gmt[0], g_gmt[1], g_gmt[2], g_gmt[3], wtime() - t0, g_gmt[5]);
    for (int k = 0; k < nbk; k++) { BB *s = G->S[k]; free(s->R); s->R = bx(sizeof(double) * ((size_t)s->n * s->r + 1)); memcpy(s->R, z + G->off[k], sizeof(double) * (size_t)s->n * s->r); }
    *vout = bx(sizeof(double) * (G->nl + 1)); memcpy(*vout, z + G->N, sizeof(double) * G->nl);
    memcpy(yout, G->y, sizeof(double) * m);
    for (int k = 0; k < nbk; k++) free(G->xs2[k]);
    free(G->xs2);
    for (int k = 0; k < nbk; k++) gu_free(G->U[k]);
    free(G->U); G->U = NULL;
    free(G->y); free(G->t); free(G->t1); free(G->t2); free(G->w); free(vk); free(lmk); free(gw); free(lt); free(z); free(Gr); free(Gn); free(D); free(Rn); free(Sv); free(Yv);
    return rc;
}

/* -bm 1 on several semidefinite blocks (and LP variables): the setup, gen_alm_multi, the result and the files */
/* Rp != NULL (16 November 2026, ISSUES 898): a solve inside the presolve pipeline (main.c): the solution of P (the presolved,
 * scaled problem) into yp and Xp (the layout of the pipeline: an SDP block n x n, an LP block n), the result into Rp; no log
 * of the problem, no files, P kept. The pipeline maps the solution back and measures it on the problem as read. */
static int bm_multi_x(Problem *P, Params *par, const char *yfile, const char *xfile, const char *zfile, BriskResult *res, double t0, double tread,
                      Result *Rp, double *yp, double **Xp) {
    const int m = P->m, verbose = par->verbose; const double tol = par->tol;
    int nsdp = 0, nlp = 0; for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) nsdp++; else if (P->blk[k].type == BLK_LP) nlp += P->blk[k].n; }
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type != BLK_SDP && P->blk[k].type != BLK_LP) { fprintf(stderr, "brisk: -bm 1 takes semidefinite and LP blocks only\n"); return 2; }
    GM G; memset(&G, 0, sizeof G); G.P = P; G.m = m; G.nbk = nsdp;
    /* LP variables by variable (the order of the LP blocks), their kinds: free pairs as one free variable */
    int *lpoff = bz(sizeof(int) * (P->nblk + 1));
    FreePair *fp = NULL; int nfp = 0;   /* free pairs (LP block, '+' slot, '-' slot) */
    if (nlp) {
        G.nl = nlp; G.lp_ptr = bz(sizeof(int) * (nlp + 1)); G.lp_c = bz(sizeof(double) * (nlp + 1)); G.lp_kind = bz(nlp + 1);
        int off = 0; size_t ne = 0;
        for (int k = 0; k < P->nblk; k++) { const Block *Bk = &P->blk[k]; lpoff[k] = off; if (Bk->type != BLK_LP) continue;
            for (int t = 0; t < Bk->ncon; t++) for (int q = 0; q < Bk->A[t].nnz; q++) { G.lp_ptr[off + Bk->A[t].row[q] + 1]++; ne++; }
            for (int q = 0; q < Bk->C.nnz; q++) G.lp_c[off + Bk->C.row[q]] += Bk->C.val[q];
            off += Bk->n; }
        for (int j = 0; j < nlp; j++) G.lp_ptr[j + 1] += G.lp_ptr[j];
        G.lp_con = bx(sizeof(int) * (ne + 1)); G.lp_val = bx(sizeof(double) * (ne + 1));
        int *pos = bx(sizeof(int) * (nlp + 1)); memcpy(pos, G.lp_ptr, sizeof(int) * nlp);
        for (int k = 0; k < P->nblk; k++) { const Block *Bk = &P->blk[k]; if (Bk->type != BLK_LP) continue;
            for (int t = 0; t < Bk->ncon; t++) for (int q = 0; q < Bk->A[t].nnz; q++) { const int j = lpoff[k] + Bk->A[t].row[q]; G.lp_con[pos[j]] = Bk->con[t]; G.lp_val[pos[j]] = Bk->A[t].val[q]; pos[j]++; } }
        free(pos);
        nfp = getenv("BRISK_BMNOFREE") ? 0 : free_pairs_detect(P, &fp);
        for (int f = 0; f < nfp; f++) { G.lp_kind[lpoff[fp[f].blk] + fp[f].ip] = 1; G.lp_kind[lpoff[fp[f].blk] + fp[f].im] = 2; }
        const int nf = nfp;
        if (verbose >= 0 && nf > 0 && !g_log_brief) printf("bm: %d free variable(s) given as split pairs: one unsquared variable each\n", nf);
    }
    /* the blocks, their first ranks and factors */
    BB *bb = bz(sizeof(BB) * (nsdp + 1)); G.S = bz(sizeof(BB *) * (nsdp + 1)); G.off = bz(sizeof(size_t) * (nsdp + 1));
    const int renv = getenv("BRISK_BMRANK") ? atoi(getenv("BRISK_BMRANK")) : 0;
    unsigned h = 987654321u; int q = 0, rsum = 0;
    for (int k = 0; k < P->nblk; k++) {
        if (P->blk[k].type != BLK_SDP) continue;
        BB *s = &bb[q]; s->jt = -1; s->jcon = -1;
        bb_setup(s, &P->blk[k], k);
        s->r = (int)fmin(s->n, renv > 0 ? renv : ceil(2.0 * log((double)s->ncon + 1)) + 2);
        if (s->r < 1) s->r = 1;
        s->R = bx(sizeof(double) * ((size_t)s->n * s->r + 1));
        for (int i = 0; i < s->n; i++) {
            double *ri = s->R + (size_t)i * s->r, nr = 0;
            for (int l = 0; l < s->r; l++) { h = h * 1103515245u + 12345u; ri[l] = ((h >> 8) & 0xffff) / 65535.0 - 0.5; nr += ri[l] * ri[l]; }
            nr = sqrt(nr); for (int l = 0; l < s->r; l++) ri[l] /= nr;
        }
        G.S[q] = s; rsum += s->r; q++;
    }
    if (verbose >= 0 && g_log_brief) {
        if (!Rp) { char pz[200]; brisk_log_problem(pz, sizeof pz, P); log_user("Problem:  %s\n", pz); }
        brisk_log_method("Burer-Monteiro: X_k = R_k R_k' on %d blocks (%d columns in all), by an augmented Lagrangian method", nsdp, rsum);
    } else if (verbose >= 0) printf("bm: general constraints (augmented Lagrangian), %d semidefinite blocks, %d LP variables, m = %d, ranks %d in all\n", nsdp, nlp, m, rsum);
    BMMeas M; memset(&M, 0, sizeof M);
    double *yv = bz(sizeof(double) * (m + 1)), *v = NULL;
    const int rc = gen_alm_multi(&G, &v, tol, verbose, par->timelimit, t0, &M, yv);
    const double tt = wtime() - t0;
    double err[7] = { 0, M.e1, 0, 0, M.e2, M.e3, M.e3 };
    const double acc = brisk_nmax(brisk_nmax(err[1], err[4]), err[5]);
    const int st = rc == 0 && acc <= tol ? ST_OPTIMAL : acc <= par->red_acc ? ST_REDUCED : (par->timelimit > 0 && tt >= par->timelimit) ? ST_TIME : ST_MAXIT;
    const char *sn = st == ST_OPTIMAL ? "OPTIMAL" : st == ST_REDUCED ? "SOLVED TO REDUCED ACCURACY" : st == ST_TIME ? "TIME LIMIT" : "ITERATION LIMIT";
    rsum = 0; for (int k = 0; k < nsdp; k++) rsum += G.S[k]->r;
    if (Rp) {
        /* the pipeline's X and y */
        int qq = 0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *Bk = &P->blk[k];
            if (Bk->type == BLK_LP) {
                for (int i = 0; i < Bk->n; i++) { const int j = lpoff[k] + i; const unsigned char kd = G.lp_kind[j]; Xp[k][i] = kd == 0 ? v[j] * v[j] : kd == 1 ? (v[j] > 0 ? v[j] : 0) : 0; }
                continue;
            }
            const BB *s = G.S[qq++]; const int n = s->n, r = s->r;
            double *Rc = bx(sizeof(double) * ((size_t)n * r + 1)), *X = Xp[k];
            for (int i = 0; i < n; i++) for (int l = 0; l < r; l++) Rc[i + (size_t)l * n] = s->R[(size_t)i * r + l];
            const double one = 1.0, zero = 0.0;
            BL(dsyrk_)("L", "N", &n, &r, &one, Rc, &n, &zero, X, &n);
            for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) X[j + (size_t)i * n] = X[i + (size_t)j * n];
            free(Rc);
        }
        for (int f = 0; f < nfp; f++) { const double u = v[lpoff[fp[f].blk] + fp[f].ip]; Xp[fp[f].blk][fp[f].im] = u < 0 ? -u : 0; }
        memcpy(yp, yv, sizeof(double) * (m > 0 ? m : 1));
        memset(Rp, 0, sizeof(*Rp));
        Rp->status = st; Rp->iters = 0; Rp->pobj = M.pobj; Rp->dobj = M.dobj;
        Rp->pinf = M.e1; Rp->dinf = M.e2; Rp->relgap = M.e3; Rp->relcomp = M.e3;
        for (int e = 1; e <= 6; e++) Rp->err[e] = err[e];
        Rp->t_total = tt;
        if (verbose >= 0 && !g_log_brief) { printf("bm: status %s (max rel error %.1e, on the presolved problem)   ranks %d in all:", sn, acc, rsum); for (int k = 0; k < nsdp; k++) printf(" %d", G.S[k]->r); printf("\n"); }
        for (int k = 0; k < nsdp; k++) bb_free(G.S[k]);
        free(bb); free(G.S); free(G.off); free(G.lp_ptr); free(G.lp_con); free(G.lp_val); free(G.lp_c); free(G.lp_kind); free(lpoff); free(fp); free(v); free(yv);
        return 0;
    }
    /* X and Z per block (dense SDP blocks up to the order 20,000; LP values: v^2, or a free value split into its two slots) */
    int dense_ok = 1; for (int k = 0; k < nsdp; k++) if (G.S[k]->n > 20000) dense_ok = 0;
    double **Xb = NULL, **Zb = NULL;
    if ((res || xfile || zfile) && dense_ok) {
        Xb = bz(sizeof(double *) * (P->nblk + 1)); Zb = bz(sizeof(double *) * (P->nblk + 1));
        int qq = 0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *Bk = &P->blk[k];
            if (Bk->type == BLK_LP) {
                Xb[k] = bz(sizeof(double) * (Bk->n + 1)); Zb[k] = bz(sizeof(double) * (Bk->n + 1));
                for (int i = 0; i < Bk->n; i++) {
                    const int j = lpoff[k] + i; const unsigned char kd = G.lp_kind[j];
                    if (kd == 0) Xb[k][i] = v[j] * v[j];
                    else if (kd == 1) Xb[k][i] = v[j] > 0 ? v[j] : 0;
                    double sj = G.lp_c[j]; for (int e = G.lp_ptr[j]; e < G.lp_ptr[j + 1]; e++) sj -= yv[G.lp_con[e]] * G.lp_val[e];
                    Zb[k][i] = sj;
                }
                continue;
            }
            const BB *s = G.S[qq++]; const int n = s->n, r = s->r; const size_t nn = (size_t)n * n;
            double *X = bx(sizeof(double) * (nn + 1)), *Z = bz(sizeof(double) * (nn + 1)), *Rc = bx(sizeof(double) * ((size_t)n * r + 1));
            for (int i = 0; i < n; i++) for (int l = 0; l < r; l++) Rc[i + (size_t)l * n] = s->R[(size_t)i * r + l];
            const double one = 1.0, zero = 0.0;
            BL(dsyrk_)("L", "N", &n, &r, &one, Rc, &n, &zero, X, &n);
            for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) X[j + (size_t)i * n] = X[i + (size_t)j * n];
            for (int q2 = 0; q2 < Bk->C.ef; q2++) Z[Bk->C.fr[q2] + (size_t)Bk->C.fc[q2] * n] += Bk->C.fv[q2];
            for (int t = 0; t < Bk->ncon; t++) { const SpSym *A = &Bk->A[t]; const double yt = yv[Bk->con[t]]; for (int q2 = 0; q2 < A->ef; q2++) Z[A->fr[q2] + (size_t)A->fc[q2] * n] -= yt * A->fv[q2]; }
            free(Rc); Xb[k] = X; Zb[k] = Z;
        }
        /* the free variables' '-' slots: the negative part of the value */
        for (int f = 0; f < nfp; f++) { const double u = v[lpoff[fp[f].blk] + fp[f].ip]; Xb[fp[f].blk][fp[f].im] = u < 0 ? -u : 0; }
        for (int w = 0; w < 2; w++) {
            const char *fn = w == 0 ? xfile : zfile; double **Mb = w == 0 ? Xb : Zb;
            if (!fn) continue;
            FILE *f = brisk_out_open(fn); if (!f) continue;
            for (int k = 0; k < P->nblk; k++) {
                const Block *Bk = &P->blk[k]; const int n = Bk->n;
                if (Bk->type == BLK_LP) { for (int i = 0; i < n; i++) if (Mb[k][i] != 0) fprintf(f, "%d %d %d %.17g\n", k + 1, i + 1, i + 1, Mb[k][i]); continue; }
                for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { const double val = Mb[k][i + (size_t)j * n]; if (val != 0) fprintf(f, "%d %d %d %.17g\n", k + 1, i + 1, j + 1, val); }
            }
            brisk_out_close(f, fn);
        }
    }
    if (yfile) { FILE *f = brisk_out_open(yfile); if (f) { for (int i = 0; i < m; i++) fprintf(f, "%.17g\n", yv[i]); brisk_out_close(f, yfile); } }
    if (res) {
        static const int ecode[7] = { 0, 11, 12, 13, 14, 10, 15 };
        res->status = st; res->exit_code = ecode[st];
        snprintf(res->status_str, sizeof(res->status_str), "%s", sn);
        res->iters = 0; res->pobj = M.pobj; res->dobj = M.dobj;
        for (int e = 1; e <= 6; e++) res->err[e] = err[e];
        res->time = tt; res->m = m; res->nblk = P->nblk;
        res->bs = malloc(sizeof(int) * (P->nblk + 1)); for (int k = 0; k < P->nblk; k++) res->bs[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
        res->y = malloc(sizeof(double) * (m + 1)); memcpy(res->y, yv, sizeof(double) * m);
        res->have_x = Xb != NULL; res->X = Xb; res->Z = Zb; Xb = Zb = NULL;
        res->bound_side = 0; res->bound_rigorous = NAN;
        snprintf(res->cause, sizeof(res->cause), "%s", st == ST_OPTIMAL ? "" : "Burer-Monteiro (-bm): the target accuracy was not reached");
    }
    if (Xb) { for (int k = 0; k < P->nblk; k++) { free(Xb[k]); free(Zb[k]); } free(Xb); free(Zb); }
    if (verbose >= 0 && g_log_brief) {
        char pz[200], sv[300]; brisk_log_problem(pz, sizeof pz, P);
        snprintf(sv, sizeof sv, "Burer-Monteiro (X_k = R_k R_k' on %d blocks, %d columns in all), augmented Lagrangian method", nsdp, rsum);
        brisk_log_summary(pz, sv, tt, sn, acc, "", M.pobj, M.dobj, err, "");
    } else if (verbose >= 0) {
        printf("status: %s (max rel error %.1e)   ranks %d in all:", sn, acc, rsum);
        for (int k = 0; k < nsdp; k++) printf(" %d", G.S[k]->r);
        printf("\noptimal value (SDPA/SDPLIB convention, max <F0,Y>): %.10e\n", -M.pobj);
        printf("primal objective %.12e  dual objective %.12e\n", M.pobj, M.dobj);
        printf("errors: primal %.1e  dual (lambda_min, free variables) %.1e  gap %.1e\n", M.e1, M.e2, M.e3);
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n", err[1], err[2], err[3], err[4], err[5], err[6]);
        printf("  Time:              %.2f s (reading %.2f s)\n", tt, tread);
    }
    for (int k = 0; k < nsdp; k++) bb_free(G.S[k]);
    free(bb); free(G.S); free(G.off); free(G.lp_ptr); free(G.lp_con); free(G.lp_val); free(G.lp_c); free(G.lp_kind); free(lpoff); free(fp); free(v); free(yv);
    problem_free(P);
    return st == ST_OPTIMAL ? 0 : st == ST_REDUCED ? 10 : st == ST_TIME ? 15 : 13;
}
static int bm_multi(Problem *P, Params *par, const char *yfile, const char *xfile, const char *zfile, BriskResult *res, double t0, double tread) {
    return bm_multi_x(P, par, yfile, xfile, zfile, res, t0, tread, NULL, NULL, NULL);
}
int bm_multi_solve(Problem *P, Params *par, Result *R, double *yout, double **Xout) {
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type != BLK_SDP && P->blk[k].type != BLK_LP) return -1;
    /* the method sees the presolved problem in its own units (the pipeline's scaling A~_i = d_i A_i, b~ = d b / bs,
     * C~ = C / cs undone on its values, which are restored bit for bit afterwards): its rules and its measures are made
     * for the data as given (NH2-, SDP: 25.6 s as read, 35.8 s on the scaled problem, its tolerance missed there) */
    const int sc = P->scaled && !getenv("BRISK_BMPIPESCALED");
    double *sb = NULL, **sv = NULL; size_t nsv = 0;
    if (sc) {
        size_t na = 0; for (int k = 0; k < P->nblk; k++) na += 2 + 2 * (size_t)P->blk[k].ncon;
        sv = malloc(sizeof(double *) * (na + 1));
        sb = malloc(sizeof(double) * ((size_t)P->m + 1)); memcpy(sb, P->b, sizeof(double) * (size_t)P->m);
        for (int k = 0; k < P->nblk; k++) {
            Block *B = &P->blk[k];
            SpSym *S = &B->C;
            sv[nsv] = malloc(sizeof(double) * ((size_t)S->nnz + 1)); memcpy(sv[nsv++], S->val, sizeof(double) * (size_t)S->nnz);
            sv[nsv] = malloc(sizeof(double) * ((size_t)S->ef + 1)); memcpy(sv[nsv++], S->fv, sizeof(double) * (size_t)S->ef);
            for (int q = 0; q < S->nnz; q++) S->val[q] *= P->cs;
            for (int q = 0; q < S->ef; q++) S->fv[q] *= P->cs;
            for (int t = 0; t < B->ncon; t++) {
                SpSym *A = &B->A[t]; const double f = P->d ? 1.0 / P->d[B->con[t]] : 1.0;
                sv[nsv] = malloc(sizeof(double) * ((size_t)A->nnz + 1)); memcpy(sv[nsv++], A->val, sizeof(double) * (size_t)A->nnz);
                sv[nsv] = malloc(sizeof(double) * ((size_t)A->ef + 1)); memcpy(sv[nsv++], A->fv, sizeof(double) * (size_t)A->ef);
                for (int q = 0; q < A->nnz; q++) A->val[q] *= f;
                for (int q = 0; q < A->ef; q++) A->fv[q] *= f;
            }
        }
        for (int i = 0; i < P->m; i++) P->b[i] *= P->bs / (P->d ? P->d[i] : 1.0);
    }
    const double ts = wtime();
    const int rc = bm_multi_x(P, par, NULL, NULL, NULL, NULL, par->t_start > 0 ? par->t_start : ts, 0, R, yout, Xout);   /* (the time limit from the run's start) */
    R->t_total = wtime() - ts;
    if (sc) {
        nsv = 0;
        memcpy(P->b, sb, sizeof(double) * (size_t)P->m); free(sb);
        for (int k = 0; k < P->nblk; k++) {
            Block *B = &P->blk[k];
            memcpy(B->C.val, sv[nsv], sizeof(double) * (size_t)B->C.nnz); free(sv[nsv++]);
            memcpy(B->C.fv, sv[nsv], sizeof(double) * (size_t)B->C.ef); free(sv[nsv++]);
            for (int t = 0; t < B->ncon; t++) {
                memcpy(B->A[t].val, sv[nsv], sizeof(double) * (size_t)B->A[t].nnz); free(sv[nsv++]);
                memcpy(B->A[t].fv, sv[nsv], sizeof(double) * (size_t)B->A[t].ef); free(sv[nsv++]);
            }
        }
        free(sv);
    } else if (rc == 0 && P->scaled) {   /* (BRISK_BMPIPESCALED=1: solved scaled; the pipeline's solvers return the unscaled solution) */
        for (int k = 0; k < P->nblk; k++) { const size_t len = P->blk[k].type == BLK_SDP ? (size_t)P->blk[k].n * P->blk[k].n : (size_t)P->blk[k].n; for (size_t i = 0; i < len; i++) Xout[k][i] *= P->bs; }
        for (int i = 0; i < P->m; i++) yout[i] *= P->cs * (P->d ? P->d[i] : 1.0);
        R->pobj *= P->bs * P->cs; R->dobj *= P->bs * P->cs;
    }
    if (getenv("BRISK_BMPIPEDBG") && rc == 0) {   /* (the primal residual of the returned X on P itself) */
        double *r = calloc((size_t)P->m + 1, sizeof(double)), mx = 0, obj = 0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k]; const int n = B->n;
            for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; double sacc = 0;
                for (int q = 0; q < A->nnz; q++) { const int i = A->row[q], j = A->col[q];
                    sacc += B->type == BLK_LP ? A->val[q] * Xout[k][i] : (i == j ? 1.0 : 2.0) * A->val[q] * Xout[k][i + (size_t)j * n]; }
                r[B->con[t]] += sacc; }
            for (int q = 0; q < B->C.nnz; q++) { const int i = B->C.row[q], j = B->C.col[q];
                obj += B->type == BLK_LP ? B->C.val[q] * Xout[k][i] : (i == j ? 1.0 : 2.0) * B->C.val[q] * Xout[k][i + (size_t)j * n]; }
        }
        for (int i = 0; i < P->m; i++) mx = fmax(mx, fabs(r[i] - P->b[i]));
        fprintf(stderr, "bm pipe: (X unscaled) max |A X - b| %.3e, <C,X> %.10e (scaled %d)\n", mx, obj, P->scaled);
        free(r);
    }
    return rc;
}
/* the semidefinite blocks in the file's header (or the data in memory): the multi-block method goes through the presolve */
int bm_header_nsdp(const char *fname, const BriskData *data) {
    if (data) { int c = 0; for (int k = 0; k < data->nblk; k++) if (data->bs[k] > 0) c++; return c; }
    FILE *f = fopen(fname, "r"); if (!f) return 0;
    char line[1 << 16]; int got = 0, nb = 0, c = 0, seen = 0;
    while (got < 3 && fgets(line, sizeof line, f)) {
        if (line[0] == '*' || line[0] == '"') continue;
        for (char *q = line; *q; q++) if (*q == ',' || *q == '{' || *q == '}' || *q == '(' || *q == ')') *q = ' ';
        if (got == 0) { got++; continue; }
        if (got == 1) { nb = atoi(line); got++; continue; }
        char *p = line, *e;
        while (seen < nb) { const long v = strtol(p, &e, 10); if (e == p) break; if (v > 0) c++; seen++; p = e; }
        if (seen >= nb) got++;
    }
    fclose(f);
    return c;
}


/* 5.14: the automatic route: one semidefinite block of order n >= BRISK_BMAUTON (500) with m = n
 * constraints, from the file's header (or the data in memory); bm_run then checks that every
 * constraint fixes a diagonal entry */
/* (13 November 2026) a constraint a <J, X> (J = e e') with right-hand side 0: every entry of the upper triangle stored,
 * all of one value a; returns a, else 0 */
static double bm_jrow(const SpSym *A, int n, double b) {
    if (b != 0 && b != b) return 0;   /* (any right-hand side: the exact penalty checks b = 0 itself) */
    if ((size_t)A->nnz != (size_t)n * (n + 1) / 2 || A->nnz < 1) return 0;
    const double a = A->val[0];
    if (a == 0) return 0;
    for (int q = 1; q < A->nnz; q++) if (A->val[q] != a) return 0;
    return a;
}

int bm_auto_candidate(const char *fname, const BriskData *data, const Params *par, int need_dense) {
    if (par->bm != -1 || par->bound_side || par->fom > 0 || par->mfipm || par->lralm > 0 || par->dual || par->crossover) return 0;
    const int nmin = getenv("BRISK_BMAUTON") ? atoi(getenv("BRISK_BMAUTON")) : 500;
    int m = 0, nb = 0, n0 = 0;
    if (data) { m = data->m; nb = data->nblk; n0 = nb == 1 ? data->bs[0] : 0; }
    else {
        FILE *f = fopen(fname, "r"); if (!f) return 0;
        char line[4096]; int got = 0;
        while (got < 3 && fgets(line, sizeof line, f)) {
            if (line[0] == '*' || line[0] == '"') continue;
            for (char *c = line; *c; c++) if (*c == ',' || *c == '{' || *c == '}' || *c == '(' || *c == ')') *c = ' ';
            if (got == 0) { m = atoi(line); got++; }
            else if (got == 1) { nb = atoi(line); got++; }
            else { n0 = nb == 1 ? atoi(line) : 0; got++; }
        }
        fclose(f);
    }
    if (nb != 1 || n0 < 2) return 0;
    if (need_dense && n0 > 20000) return 0;   /* (X would be returned dense) */
    /* (1 November 2026: m > n0 too, a fixed diagonal with dependent rows; bm_run_x removes them first when every row
     * touches diagonal entries only. The price on a problem of this shape that is no fixed diagonal: one more reading of
     * the file before the interior-point method) */
    const int depok = !(getenv("BRISK_BMDEP") && atoi(getenv("BRISK_BMDEP")) == 0);   /* (BRISK_BMDEP=0: m = n0 only, as before) */
    if (n0 >= nmin && (m == n0 || (depok && m > n0))) return 1;            /* a fixed diagonal, perhaps */
    if (8.0 * (double)m * (double)m > 0.5 * brisk_mem_limit()) return 2;  /* the Schur complement does not fit: entry fixings, perhaps */
    return 0;
}

int bm_run(const char *fname, Params *par, const char *yfile, const BriskData *data, BriskResult *res) {
    return bm_run_x(fname, par, yfile, NULL, NULL, data, res, 0);
}

int bm_run_x(const char *fname, Params *par, const char *yfile, const char *xfile, const char *zfile, const BriskData *data, BriskResult *res, int autom) {
    if (par->bm == 2 && !autom) return bmip_run(fname, par, yfile, data, res);   /* 5.16: the interior-point method on the factored program */
    const double t0 = wtime();
    if (getenv("BRISK_BMJROW")) g_bm_jrow = atoi(getenv("BRISK_BMJROW"));
    Problem PP, *P = &PP;
    memset(P, 0, sizeof PP);
    const int maxn0 = g_read_maxn;
    g_read_maxn = 2000000000;
    if (data ? problem_from_sdpa_data(data, P) != 0 : read_sdpa(fname, P) != 0) { g_read_maxn = maxn0; return autom ? -1 : 2; }
    g_read_maxn = maxn0;
    if (!P->b0) { P->b0 = bx(sizeof(double) * (P->m + 1)); memcpy(P->b0, P->b, sizeof(double) * P->m); }
    const double tread = wtime() - t0;
    const int verbose = par->verbose;
    const double tol = par->tol;   /* (-lrtol is -lralm's) */
    int nsdp = 0, nlp = 0; for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) nsdp++; else nlp++; }
    double tlim_auto = 0;
    const int mread = P->m; int dep_removed = 0;   /* (the constraints as read, and those the check below removed) */
    if (autom == 1 && nsdp == 1 && !nlp && P->blk[0].ncon == mread && mread > P->blk[0].n && par->deprows > 0) {
        /* 1 November 2026 (the problems written differently, held-out set): a fixed diagonal written with more rows than
         * the order, rows repeated or added as sums of others (max-cut with 1,082 such rows: 4.1 s here, 56.9 s by the
         * interior-point method, because this test saw the rows as read). When every constraint touches diagonal entries
         * only, the presolve's removal of linearly dependent rows runs first, as on the interior-point path (the removed
         * rows' dual values are 0); the test below then sees the problem the presolve would have handed on */
        int dg = 1;
        for (int t = 0; dg && t < P->blk[0].ncon; t++) { const SpSym *A = &P->blk[0].A[t]; for (int q = 0; q < A->nnz; q++) if (A->row[q] != A->col[q]) { dg = 0; break; } }
        if (dg) {
            int ninc = 0;
            const double dtol = getenv("BRISK_DEPTOL") ? atof(getenv("BRISK_DEPTOL")) : 1e-8;
            dep_removed = remove_dependent_rows(P, verbose >= 1 ? verbose : -1, dtol, &ninc);
            if (ninc > 0) { problem_free(P); return -1; }   /* (inconsistent rows: the main path reports them) */
        }
    }
    const int m = P->m;
    if (autom) {   /* 1: every constraint fixes a diagonal entry of the one block; 2: every constraint fixes one entry */
        int fx = nsdp == 1 && !nlp && P->blk[0].ncon == m, nj = 0;
        for (int t = 0; fx && t < P->blk[0].ncon; t++) { const SpSym *A = &P->blk[0].A[t];
            const int dg1 = A->nnz == 1 && A->row[0] == A->col[0] && A->val[0] > 0 && P->b[P->blk[0].con[t]] > 0;
            if (autom == 1 && g_bm_jrow && !dg1 && nj < 20) { nj++; continue; }   /* (up to 20 other rows: the augmented Lagrangian on them below; bm_run_x checks the diagonal) */
            if (A->nnz != 1 || (autom == 1 && !dg1)) fx = 0; }
        if (!fx) { problem_free(P); return -1; }
        if (getenv("BRISK_BMPROBE")) { printf("bm-probe: accepted, %d rows beside the fixed diagonal\n", nj); fflush(stdout); problem_free(P); return -1; }   /* (test: the detection only) */
        /* the attempt's budget: half the time limit, or 900 s */
        tlim_auto = par->timelimit > 0 ? 0.5 * par->timelimit : 900.0;
    }
    if (verbose >= 0) {
        if (autom == 1) printf("every constraint fixes a diagonal entry (n = %d): the Burer-Monteiro method first (-bm 0 turns this off)\n", P->blk[0].n);
        if (autom == 2) printf("the Schur complement does not fit (m = %d) and every constraint fixes one entry: the Burer-Monteiro method first (-bm 0 turns this off)\n", m);
        printf("BRISK %s Burer-Monteiro (-bm)\n", BRISK_VERSION);
        brisk_log_threads(1);
        printf("problem %s: m = %d, %d SDP blocks, %d LP blocks, read %.2fs\n", fname, m, nsdp, nlp, tread);
    }
    /* 15 November 2026: several blocks by gen_alm_multi (the explicit -bm 1 only: the automatic route takes one block, above) */
    if (nsdp >= 2 && !autom) return bm_multi(P, par, yfile, xfile, zfile, res, t0, tread);
    if (nsdp != 1) { fprintf(stderr, "brisk: -bm 1 needs at least one semidefinite block\n"); return 2; }
    {   /* not of the fixed-diagonal kind: equilibrate first (the setups below copy the data) */
        int fx = !nlp;
        for (int k = 0; k < P->nblk && fx; k++) { const Block *Bk = &P->blk[k]; if (Bk->type != BLK_SDP) continue;
            if (Bk->ncon != m) fx = 0;
            for (int t = 0; t < Bk->ncon && fx; t++) { const SpSym *A = &Bk->A[t]; if (A->nnz != 1 || A->row[0] != A->col[0] || A->val[0] <= 0 || P->b[Bk->con[t]] <= 0) fx = 0; } }
        if (!fx && getenv("BRISK_BMSCALE") && atoi(getenv("BRISK_BMSCALE"))) bm_scale(P);
    }
    GA LPd; memset(&LPd, 0, sizeof LPd);
    if (nlp) {   /* the LP variables, by variable: their constraints and costs */
        int nv = 0; for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_LP) nv += P->blk[k].n;
        LPd.nl = nv; LPd.lp_ptr = bz(sizeof(int) * (nv + 1)); LPd.lp_c = bz(sizeof(double) * (nv + 1));
        int off = 0; size_t ne = 0;
        for (int k = 0; k < P->nblk; k++) { const Block *Bk = &P->blk[k]; if (Bk->type != BLK_LP) continue;
            for (int t = 0; t < Bk->ncon; t++) for (int q = 0; q < Bk->A[t].nnz; q++) { LPd.lp_ptr[off + Bk->A[t].row[q] + 1]++; ne++; }
            for (int q = 0; q < Bk->C.nnz; q++) LPd.lp_c[off + Bk->C.row[q]] += Bk->C.val[q];
            off += Bk->n; }
        for (int j = 0; j < nv; j++) LPd.lp_ptr[j + 1] += LPd.lp_ptr[j];
        LPd.lp_con = bx(sizeof(int) * (ne + 1)); LPd.lp_val = bx(sizeof(double) * (ne + 1));
        int *pos = bx(sizeof(int) * (nv + 1)); memcpy(pos, LPd.lp_ptr, sizeof(int) * nv);
        off = 0;
        for (int k = 0; k < P->nblk; k++) { const Block *Bk = &P->blk[k]; if (Bk->type != BLK_LP) continue;
            for (int t = 0; t < Bk->ncon; t++) for (int q = 0; q < Bk->A[t].nnz; q++) { const int j = off + Bk->A[t].row[q]; LPd.lp_con[pos[j]] = Bk->con[t]; LPd.lp_val[pos[j]] = Bk->A[t].val[q]; pos[j]++; }
            off += Bk->n; }
        free(pos);
    }
    BB s; memset(&s, 0, sizeof s);
    int kb = 0; while (P->blk[kb].type != BLK_SDP) kb++;
    const Block *B = &P->blk[kb];
    s.jt = -1; s.jcon = -1;
    /* the rows that fix no diagonal entry: up to 20 (the augmented Lagrangian on them), at most one of them a J row */
    int ext[20], next = 0, xjt = -1; double xja = 0;
    {   int okx = g_bm_jrow && B->ncon == m && !nlp;
        for (int t = 0; okx && t < B->ncon; t++) {
            const SpSym *A = &B->A[t];
            if (A->nnz == 1 && A->row[0] == A->col[0] && A->val[0] > 0 && P->b[B->con[t]] > 0) continue;
            if (next >= 20) { okx = 0; break; }
            ext[next++] = t;
            const double a = bm_jrow(A, B->n, P->b[B->con[t]]);
            if (a != 0 && xjt < 0) { xjt = t; xja = a; }
            else if ((size_t)A->ef > 20 * (size_t)B->n) okx = 0;   /* (a dense row other than J: not this method) */
        }
        if (!okx) { next = 0; xjt = -1; }
    }
    g_bb_skip = xjt;
    bb_setup(&s, B, kb);
    g_bb_skip = -1;
    /* fixed diagonal? every constraint one diagonal entry, each row at most one */
    int fixed = B->ncon == m && !nlp;
    if (fixed) {
        s.rho = bx(sizeof(double) * s.n); s.adiag = bx(sizeof(double) * s.n); s.crow = bx(sizeof(int) * s.n);
        for (int i = 0; i < s.n; i++) s.crow[i] = -1;
        for (int t = 0; t < B->ncon && fixed; t++) {
            int isx = 0; for (int q = 0; q < next; q++) isx |= ext[q] == t;
            if (isx) continue;
            const SpSym *A = &B->A[t];
            if (A->nnz != 1 || A->row[0] != A->col[0] || A->val[0] <= 0 || P->b[B->con[t]] <= 0 || s.crow[A->row[0]] >= 0) { fixed = 0; break; }
            const int p = A->row[0];
            s.crow[p] = B->con[t]; s.adiag[p] = A->val[0]; s.rho[p] = P->b[B->con[t]] / A->val[0];
        }
        for (int i = 0; i < s.n && fixed; i++) if (s.crow[i] < 0) fixed = 0;
    }
    if (xjt >= 0 && !fixed) {   /* a J row without a fixed diagonal: the general method, on the full pattern */
        bb_free(&s); memset(&s, 0, sizeof s); s.jt = -1; s.jcon = -1;
        bb_setup(&s, B, kb);
        next = 0; xjt = -1;
    }
    if (!fixed) next = 0;
    if (autom == 1 && !fixed) {
        /* (13 November 2026) the automatic attempt takes rows beside the fixed diagonal only when the fixed-diagonal path does
         * (a J row or sparse rows, each row's diagonal fixed); anything else goes to the interior-point method at once, as
         * before the extension (equalG11, equalG51, SDP: a dense row other than J sent them to the general augmented
         * Lagrangian, 105 and 120 s against 3.2 and 8.4 s by the interior-point method) */
        bb_free(&s); problem_free(P);
        return -1;
    }
    /* one J row with right-hand side 0 and nothing else: the exact penalty (BRISK_BMALMJ=1: the augmented Lagrangian) */
    if (next == 1 && xjt >= 0 && P->b[B->con[xjt]] == 0 && !(getenv("BRISK_BMALMJ") && atoi(getenv("BRISK_BMALMJ")))) {
        s.jt = xjt; s.jcon = B->con[xjt]; s.jcoef = xja; next = 0;
    }
    int lift_n0 = 0;
    if (next == 1 && xjt >= 0 && P->b[B->con[xjt]] / xja > 0 && !(getenv("BRISK_BMLIFT") && atoi(getenv("BRISK_BMLIFT")) == 0)
        && !(getenv("BRISK_BMALMJ") && atoi(getenv("BRISK_BMALMJ")))) {
        /* (13 November 2026) a <J, X> = b with b / a = beta > 0: lifted to the order n + 1 with the added diagonal entry
         * X_nn = beta and the row w w' with w = (e, -1) at 0 (w'X^w = 0 makes the added row the sum of the others, so
         * e'Xe = beta; conversely X^ = [I e]' X [I e] for any feasible X): the exact penalty then applies. Near the
         * solution the row as read has a vanishing gradient (2 e'R, |e'R| = sqrt(beta)) and an augmented Lagrangian on it
         * crept (rendl1_2000_1e-6, SDP: 17 outer steps and |r| 2.8e-5 after 110 s). The dual as read from the lifted one:
         * Z = C - Diag(y) - y_J a J is the Schur complement of the added row in Z^ = C^ + jP w w' - Diag(y^), positive
         * semidefinite with it, y_J = jP y_n / (a (jP - y_n)). BRISK_BMLIFT=0: the augmented Lagrangian */
        const int n0 = s.n; lift_n0 = n0;
        s.jt = xjt; s.jcon = B->con[xjt]; s.jcoef = xja; s.jlast = 1; s.jb = P->b[B->con[xjt]] / xja; next = 0;
        s.sp = realloc(s.sp, sizeof(int) * (n0 + 2)); s.sj = realloc(s.sj, sizeof(int) * (s.np + 2));
        s.cv = realloc(s.cv, sizeof(double) * (s.np + 2)); s.ws = realloc(s.ws, sizeof(double) * (s.np + 2)); s.xs = realloc(s.xs, sizeof(double) * (s.np + 2));
        s.sj[s.np] = n0; s.cv[s.np] = 0; s.ws[s.np] = 0; s.xs[s.np] = 0; s.np++; s.sp[n0 + 1] = s.np;
        s.rho = realloc(s.rho, sizeof(double) * (n0 + 1)); s.adiag = realloc(s.adiag, sizeof(double) * (n0 + 1)); s.crow = realloc(s.crow, sizeof(int) * (n0 + 1));
        s.rho[n0] = s.jb; s.adiag[n0] = 1; s.crow[n0] = m;
        P->b = realloc(P->b, sizeof(double) * (m + 2)); P->b[m] = s.jb;
        s.n = n0 + 1;
    }
    if (next > 0) {
        s.kx = next;
        s.xt = bz(sizeof(int) * next); s.xcon = bz(sizeof(int) * next); s.xn = bz(sizeof(int) * next);
        s.xa = bz(sizeof(int *) * next); s.xb = bz(sizeof(int *) * next); s.xv = bz(sizeof(double *) * next);
        s.xJ = bz(sizeof(int) * next); s.xcoef = bz(sizeof(double) * next);
        s.xrhs = bz(sizeof(double) * next); s.xlam = bz(sizeof(double) * next); s.xres = bz(sizeof(double) * next); s.xw = bz(sizeof(double) * next);
        double cF = 0, aF = 0;
        for (int p = 0; p < s.np; p++) cF += s.cv[p] * s.cv[p];
        for (int q = 0; q < next; q++) {
            const int t = ext[q]; const SpSym *A = &B->A[t];
            s.xt[q] = t; s.xcon[q] = B->con[t]; s.xrhs[q] = P->b[B->con[t]];
            if (t == xjt) { s.xJ[q] = 1; s.xcoef[q] = xja; aF += xja * xja * (double)B->n * B->n; continue; }
            s.xn[q] = A->ef; s.xa[q] = bx(sizeof(int) * (A->ef + 1)); s.xb[q] = bx(sizeof(int) * (A->ef + 1)); s.xv[q] = bx(sizeof(double) * (A->ef + 1));
            for (int e = 0; e < A->ef; e++) { s.xa[q][e] = A->fr[e]; s.xb[q][e] = A->fc[e]; s.xv[q][e] = A->fv[e]; aF += A->fv[e] * A->fv[e]; }
        }
        s.cv0 = bx(sizeof(double) * (s.np + 1)); memcpy(s.cv0, s.cv, sizeof(double) * s.np);
        s.xsig = getenv("BRISK_BMXSIG") ? atof(getenv("BRISK_BMXSIG")) : sqrt(cF) / fmax(aF, 1e-300);
    }
    if (getenv("BRISK_BMSWEEPTOL")) g_bm_sweeptol = atof(getenv("BRISK_BMSWEEPTOL"));
    if (getenv("BRISK_BMNEWTON")) g_bm_newton = atoi(getenv("BRISK_BMNEWTON"));
    if (getenv("BRISK_BMCERT")) g_bm_cert = atoi(getenv("BRISK_BMCERT"));
    if (getenv("BRISK_BMSIGMA")) g_bm_sigma0 = atof(getenv("BRISK_BMSIGMA"));
    if (getenv("BRISK_BMSIGFAC")) g_bm_sigfac = atof(getenv("BRISK_BMSIGFAC"));
    if (getenv("BRISK_BMOMEGA")) g_bm_omega = atof(getenv("BRISK_BMOMEGA"));
    if (getenv("BRISK_BMPREC")) g_bm_prec = atoi(getenv("BRISK_BMPREC"));
    if (getenv("BRISK_BMPDELTA")) g_bm_pdelta = atof(getenv("BRISK_BMPDELTA"));
    if (getenv("BRISK_BMPMU")) g_bm_pmu = atof(getenv("BRISK_BMPMU"));
    if (getenv("BRISK_BMESCP")) g_bm_escp = atoi(getenv("BRISK_BMESCP"));
    if (getenv("BRISK_BMFUSE")) g_bm_fuse = atoi(getenv("BRISK_BMFUSE"));
    if (getenv("BRISK_BMLZK")) g_bm_lzk = atoi(getenv("BRISK_BMLZK"));
    if (getenv("BRISK_BMVERIFY")) g_bm_verify = atoi(getenv("BRISK_BMVERIFY"));
    if (getenv("BRISK_BMWARM")) g_bm_warm = atoi(getenv("BRISK_BMWARM"));
    if (getenv("BRISK_BMRNDFAC")) g_bm_rndfac = atof(getenv("BRISK_BMRNDFAC"));
    if (getenv("BRISK_BMESCSWEEP")) g_bm_escsweep = atoi(getenv("BRISK_BMESCSWEEP"));
    const int r0 = getenv("BRISK_BMRANK") ? atoi(getenv("BRISK_BMRANK")) : (int)fmin(s.n, fmax(8, ceil(2.0 * log((double)m + 1)) + 2));
    s.r = r0; s.R = bx(sizeof(double) * (size_t)s.n * s.r);
    {   unsigned h = 987654321u;
        for (int i = 0; i < s.n; i++) {
            double *ri = s.R + (size_t)i * s.r, nr = 0;
            for (int l = 0; l < s.r; l++) { h = h * 1103515245u + 12345u; ri[l] = ((h >> 8) & 0xffff) / 65535.0 - 0.5; nr += ri[l] * ri[l]; }
            nr = sqrt(nr); const double f = (fixed ? sqrt(s.rho[i]) : 1.0) / nr; for (int l = 0; l < s.r; l++) ri[l] *= f;
        }
    }
    BMMeas M; memset(&M, 0, sizeof M);
    double *yv = bz(sizeof(double) * (m + 1));
    int *bs0 = bx(sizeof(int) * (P->nblk + 1));
    for (int k = 0; k < P->nblk; k++) bs0[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    if (verbose >= 0 && g_log_brief) {
        char pz[200]; brisk_log_problem(pz, sizeof pz, P); log_user("Problem:  %s\n", pz);
        brisk_log_method("Burer-Monteiro: X = R R' with R of %d columns, %s", s.r, fixed ? "for a fixed diagonal" : "by an augmented Lagrangian method");
    } else if (verbose >= 0) printf("bm: %s, n = %d, m = %d, rank %d\n", fixed ? "fixed diagonal" : "general constraints (augmented Lagrangian)", s.n, m, s.r);
    g_bm_stall_stop = autom;
    int rc;
    if (fixed && s.kx) {
        /* (13 November 2026) the augmented Lagrangian on the kx extra rows: each inner solve is the fixed-diagonal method on
         * L(R) = <C, RR'> - lambda'r + sigma/2 |r|^2 (r_j = <A_j, RR'> - b_j; the gradient's cost C - sum (lambda_j - sigma
         * r_j) A_j, the Hessian's extra term 2 sigma sum <A_j R, P> A_j R); then lambda <- lambda - sigma r, and sigma four
         * times larger when |r| fell by less than four. sigma starts at |C|_F / sum |A_j|_F^2 */
        if (verbose >= 0) printf("bm: %d constraint(s) besides the fixed diagonal: an augmented Lagrangian on them (sigma %.3e)\n", s.kx, s.xsig);
        double rprev = INFINITY;
        for (int q = 0; ; q++) {
            rc = fixed_diag(&s, P, tol, verbose, autom ? tlim_auto : par->timelimit, t0, &M, yv);
            if (rc != 2 || q >= 80) break;
            x_resid(&s, s.R, s.r);
            double rn = 0; for (int j = 0; j < s.kx; j++) rn += s.xres[j] * s.xres[j]; rn = sqrt(rn);
            for (int j = 0; j < s.kx; j++) s.xlam[j] -= s.xsig * s.xres[j];
            if (rn > 0.25 * rprev) s.xsig *= 4;
            rprev = rn;
            if (verbose >= 0) printf("bm: outer %d: |A(X) - b| of the extra rows %.2e; multipliers updated, sigma %.3e\n", q + 1, rn, s.xsig);
        }
        if (rc == 2) rc = 1;
        memcpy(s.cv, s.cv0, sizeof(double) * s.np); s.jP = 0;   /* (the cost as read, for the matrices written below) */
    } else if (fixed && s.jt >= 0) {
        /* (13 November 2026) the exact penalty: min <C + jP J, X> over the fixed diagonal has the value of the problem with
         * a <J, X> = 0 once jP is at least the optimal multiplier of that row, -y_J a (C - Diag(y) - y_J a J is then
         * positive semidefinite for the penalized problem's y), and its solutions satisfy e'Xe = 0 when jP is above it.
         * jP starts at 4 sum|C_ij| / n^2 (the scale of e'(C - Diag(y))e / n^2) and is raised four times while the
         * method meets its tolerance with e'Xe still above 0 (fixed_diag returns 2). The certificate is y with
         * y_J = -jP / a: Z = C + jP J - Diag(a y), checked by a dense factorization (fd_verify) */
        double sc = 0; for (int p = 0; p < s.np; p++) sc += fabs(s.cv[p]);
        s.jP = fmax(4.0 * sc / ((double)s.n * s.n), 1e-12);
        if (verbose >= 0) { if (s.jlast) printf("bm: one constraint %g <J, X> = %g next to the fixed diagonal: lifted to the order %d (an added row with X_nn = %g and w'Xw = 0, w = (e, -1)), the exact penalty %.3e w w' on the cost\n", s.jcoef, s.jcoef * s.jb, s.n, s.jb, s.jP);
            else printf("bm: one constraint %g <J, X> = 0 next to the fixed diagonal: the exact penalty %.3e J on the cost\n", s.jcoef, s.jP); }
        for (int q = 0; ; q++) {
            rc = fixed_diag(&s, P, tol, verbose, autom ? tlim_auto : par->timelimit, t0, &M, yv);
            if (rc != 2 || q >= 20) break;
            s.jP *= 4;
            if (verbose >= 0) printf("bm: e'Xe = %.2e above 0 at the tolerance: the penalty raised to %.3e\n", jsq(&s, s.R, s.r), s.jP);
        }
        if (rc == 2) rc = 1;
    } else rc = fixed ? fixed_diag(&s, P, tol, verbose, autom ? tlim_auto : par->timelimit, t0, &M, yv) : gen_alm(&s, P, nlp ? &LPd : NULL, tol, verbose, autom ? tlim_auto : par->timelimit, t0, &M, yv);
    if (lift_n0) s.n = lift_n0;   /* (the added row dropped: X = R R' on the rows as read) */
    const double tt = wtime() - t0;
    /* y in the numbering of the rows as read (the removed dependent rows: 0, as the presolve's postsolve gives them) */
    double *yo = yv;
    if (dep_removed > 0 && P->orig) {
        yo = bz(sizeof(double) * (mread + 1));
        for (int i = 0; i < m; i++) if (P->orig[i] >= 0 && P->orig[i] < mread) yo[P->orig[i]] = yv[i];
    }
    /* the DIMACS errors: X = R R' is positive semidefinite (err2 = 0), Z = C - A'y exactly (err3 = 0) */
    double err[7] = { 0, M.e1, 0, 0, M.e2, M.e3, M.e3 };
    const double acc = brisk_nmax(brisk_nmax(err[1], err[4]), err[5]);
    /* (31 October 2026, the owner's external SDP benchmark, audit A8) on the fixed-diagonal path the dual cone error is a Lanczos
     * estimate unless a factorization of Z + delta I checked it (M.cert); without that check (no factor affordable) the
     * result is not called OPTIMAL */
    const int uncert = fixed && !M.cert;
    const int st = rc == 0 && !uncert && acc <= tol ? ST_OPTIMAL : acc <= par->red_acc ? ST_REDUCED : (par->timelimit > 0 && tt >= par->timelimit) ? ST_TIME : ST_MAXIT;
    const char *sn = st == ST_OPTIMAL ? "OPTIMAL" : st == ST_REDUCED ? "SOLVED TO REDUCED ACCURACY" : st == ST_TIME ? "TIME LIMIT" : "ITERATION LIMIT";
    if (res) {   /* the library interface: status, objectives, errors and y; no X (X = R R' is not formed) */
        static const int ecode[7] = { 0, 11, 12, 13, 14, 10, 15 };
        res->status = st; res->exit_code = ecode[st];
        snprintf(res->status_str, sizeof(res->status_str), "%s", sn);
        res->iters = 0; res->pobj = M.pobj; res->dobj = M.dobj;
        for (int e = 1; e <= 6; e++) res->err[e] = err[e];
        res->time = tt; res->m = mread; res->nblk = P->nblk;
        res->bs = malloc(sizeof(int) * (P->nblk + 1)); memcpy(res->bs, bs0, sizeof(int) * P->nblk);
        res->y = malloc(sizeof(double) * (mread + 1)); memcpy(res->y, yo, sizeof(double) * mread);
        res->have_x = 0; res->X = NULL; res->Z = NULL;
        res->bound_side = 0; res->bound_rigorous = NAN;
        snprintf(res->cause, sizeof(res->cause), "%s", st == ST_OPTIMAL ? "" : rc == 0 && uncert ? "Burer-Monteiro (-bm): the dual cone error is an estimate (no factorization of Z was affordable to check it)" : "Burer-Monteiro (-bm): the target accuracy was not reached");
    }
    if (yfile) { FILE *f = brisk_out_open(yfile); if (f) { for (int i = 0; i < mread; i++) fprintf(f, "%.17g\n", yo[i]); brisk_out_close(f, yfile); } }
    if ((res || xfile || zfile) && P->nblk == 1 && s.n <= 20000) {
        /* X = R R' and Z = C - A'y dense, as the interior-point path returns them */
        const int n = s.n, r = s.r; const size_t nn = (size_t)n * n;
        double *X = malloc(sizeof(double) * (nn + 1)), *Z = calloc(nn + 1, sizeof(double)), *Rc = malloc(sizeof(double) * ((size_t)n * r + 1));
        if (X && Z && Rc) {
            for (int i = 0; i < n; i++) for (int l = 0; l < r; l++) Rc[i + (size_t)l * n] = s.R[(size_t)i * r + l];
            const double one = 1.0, zero = 0.0;
            BL(dsyrk_)("L", "N", &n, &r, &one, Rc, &n, &zero, X, &n);
            for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) X[j + (size_t)i * n] = X[i + (size_t)j * n];
            for (int i = 0; i < n; i++) for (int p = s.sp[i]; p < s.sp[i + 1]; p++) Z[i + (size_t)s.sj[p] * n] = s.cv[p];
            const Block *B0 = &P->blk[0];
            for (int t = 0; t < B0->ncon; t++) { const SpSym *A = &B0->A[t]; const double yt = yv[B0->con[t]]; for (int q = 0; q < A->ef; q++) Z[A->fr[q] + (size_t)A->fc[q] * n] -= yt * A->fv[q]; }
            for (int w = 0; w < 2; w++) {
                const char *fn = w == 0 ? xfile : zfile; const double *M = w == 0 ? X : Z;
                if (!fn) continue;
                FILE *f = brisk_out_open(fn); if (!f) continue;
                for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { const double v = M[i + (size_t)j * n]; if (v != 0) fprintf(f, "1 %d %d %.17g\n", i + 1, j + 1, v); }
                brisk_out_close(f, fn);
            }
            if (res) { res->X = malloc(sizeof(double *) * 2); res->X[0] = X; X = NULL; res->Z = malloc(sizeof(double *) * 2); res->Z[0] = Z; Z = NULL; res->have_x = 1; }
        }
        free(X); free(Z); free(Rc);
    }
    /* (5 November 2026, ISSUES 820) the Summary after the files are written: a file not written completely is said in it */
    if (verbose >= 0 && g_log_brief) {
        char pz[200], sv[300], ex[300] = ""; brisk_log_problem(pz, sizeof pz, P);
        snprintf(sv, sizeof sv, "Burer-Monteiro (X = R R' with R of %d columns), %s", s.r, fixed ? "fixed diagonal" : "augmented Lagrangian method");
        if (dep_removed > 0) snprintf(pz + strlen(pz), sizeof pz - strlen(pz), " (%d as read; %d linearly dependent ones removed, dual values 0)", mread, dep_removed);
        if (fixed && M.cert) snprintf(ex, sizeof ex, "certified lower bound on the optimal value: %.12e", M.lb);
        brisk_log_summary(pz, sv, tt, sn, acc, "", M.pobj, M.dobj, err, ex);
    } else if (verbose >= 0) {
        printf("status: %s (max rel error %.1e)   rank %d\n", sn, acc, s.r);
        printf("optimal value (SDPA/SDPLIB convention, max <F0,Y>): %.10e\n", -M.pobj);
        printf("primal objective %.12e  dual objective %.12e\n", M.pobj, M.dobj);
        printf("errors: primal %.1e  dual (lambda_min) %.1e  gap %.1e\n", M.e1, M.e2, M.e3);
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n", err[1], err[2], err[3], err[4], err[5], err[6]);
        if (fixed && M.cert) printf("certified lower bound %.12e (Z + %.2e I factored): gap to the objective %.1e\n", M.lb, M.vdelta, M.cgap);
        else if (fixed) printf("lower bound %.12e with lambda_min(Z) %.2e estimated by Lanczos (not checked): gap to the objective %.1e\n", M.lb, M.lmin, M.cgap);
        printf("  Time:              %.2f s (reading %.2f s)\n", tt, tread);
    }
    bb_free(&s);
    free(LPd.lp_ptr); free(LPd.lp_con); free(LPd.lp_val); free(LPd.lp_c); if (yo != yv) free(yo); free(yv); free(bs0);
    problem_free(P);
    return st == ST_OPTIMAL ? 0 : st == ST_REDUCED ? 10 : st == ST_TIME ? 15 : 13;
}
