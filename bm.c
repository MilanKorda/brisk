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
} BB;

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
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) { pr[np][0] = B->A[t].fr[q]; pr[np][1] = B->A[t].fc[q]; np++; }
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
    for (int t = 0; t < B->ncon; t++) s->cp[t + 1] = s->cp[t] + B->A[t].ef;
    s->ep = bx(sizeof(int) * ((size_t)s->cp[B->ncon] + 1)); s->ev = bx(sizeof(double) * ((size_t)s->cp[B->ncon] + 1));
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) { s->ep[s->cp[t] + q] = ppos(s, B->A[t].fr[q], B->A[t].fc[q]); s->ev[s->cp[t] + q] = B->A[t].fv[q]; }
}

static void bb_free(BB *s) {
    free(s->sp); free(s->sj); free(s->cv); free(s->ws); free(s->xs); free(s->cp); free(s->ep); free(s->ev); free(s->R);
    free(s->rho); free(s->adiag); free(s->crow); free(s->cptr); free(s->cidx);
    if (s->F) schol_free(s->F);
    if (s->Fv) schol_free(s->Fv);
    free(s->pw); free(s->pb); free(s->pW); free(s->pS);
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
}

/* lambda_min of W (on the pattern) by Lanczos with full reorthogonalization, k steps; v: its vector */
static double g_bm_tsw = 0, g_bm_ttr = 0, g_bm_tlz = 0, g_bm_tvf = 0;   /* time in the sweeps, the trust region, Lanczos (verbose) */

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
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { double v = 0; for (int p = s->sp[i]; p < s->sp[i + 1]; p++) v += W[p] * qj[s->sj[p]]; w[i] = v; }
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
    pat_mul(s, s->cv, R, G, 1.0);
    double f = 0;
    #pragma omp parallel for schedule(static) reduction(+:f)
    for (int i = 0; i < n; i++) {
        const double *ri = R + (size_t)i * r; double *gi = G + (size_t)i * r;
        double d = 0; for (int l = 0; l < r; l++) d += ri[l] * gi[l];
        f += d; mu[i] = d / s->rho[i];
        for (int l = 0; l < r; l++) gi[l] -= mu[i] * ri[l];
    }
    return f;
}
/* Q = Proj_T(Z P): Z P = C P - Diag(mu) P, then each row projected off R_p */
static void fd_hess(const BB *s, const double *R, const double *mu, const double *Pd, double *Q) {
    const int n = s->n, r = s->r;
    pat_mul(s, s->cv, Pd, Q, 1.0);
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
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *qi = Q + (size_t)i * r; const double *ri = R + (size_t)i * r, *pi = Pd + (size_t)i * r;
        for (int l = 0; l < r; l++) qi[l] = -mu[i] * pi[l];
        for (int p = s->sp[i]; p < s->sp[i + 1]; p++) { const double a = s->cv[p]; if (a == 0) continue; const double *uj = Pd + (size_t)s->sj[p] * r; for (int l = 0; l < r; l++) qi[l] += a * uj[l]; }
        double d = 0; for (int l = 0; l < r; l++) d += qi[l] * ri[l];
        d /= s->rho[i];
        double v = 0; for (int l = 0; l < r; l++) { qi[l] -= d * ri[l]; v += qi[l] * pi[l]; }
        rowv[i] = v;
    }
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
        if (!pc && g_bm_fuse) {   /* tCG (Steihaug-Toint) on Proj(Z e) = -G in fused passes; H E is Rr - G */
            double *rowv = Zr;
            #pragma omp parallel for schedule(static) if (N > 65536)
            for (size_t q = 0; q < N; q++) { E[q] = 0; Rr[q] = G[q]; Pd[q] = -G[q]; }
            double rz = dotp(G, G, N); const double r0 = sqrt(rz);
            if (D < 0) D = dmax / 8;
            double ee = 0, ep = 0, pp = rz;
            for (int k = 0; k < 1000; k++) {
                const double pq = fd_hess_dot(s, s->R, mu, Pd, Q, rowv); ncg++;
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
            fd_hess(s, s->R, mu, Pd, Q); ncg++;
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
    double dobj = 0; for (int i = 0; i < P->m; i++) dobj += P->b[i] * y[i];
    memcpy(s->ws, s->cv, sizeof(double) * s->np);
    for (int i = 0; i < n; i++) s->ws[ppos(s, i, i)] -= s->adiag[i] * y[s->crow[i]];
    const double lm = lmin_est(s, s->ws, v, c1 / (s->np > 0 ? s->np : 1));
    s->vwarm = 1;
    M->cert = 0; M->pobj = f; M->dobj = dobj; M->lmin = lm;
    M->e1 = 0; M->e2 = fmax(0, -lm) / (1 + c1); M->e3 = fabs(f - dobj) / (1 + fabs(f) + fabs(dobj));
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
static int fd_verify(BB *s, double delta) { const double t = wtime(); const int v = fd_verify_(s, delta); g_bm_tvf += wtime() - t; return v; }
static int fd_verify_(BB *s, double delta) {
    const int n = s->n;
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
    double f = sweep(s), fprev = f;
    int it = 0, nesc = 0, rc = 1, ntr = 0, nver = 0;
    if (verbose >= 0) printf("bm: fixed diagonal, n = %d, rank %d, %d colours\n", n, s->r, s->ncol);
    int dosweep = 1, fresh = 0;
    for (;;) {
        /* sweeps until the objective settles */
        const double ts = wtime();
        for (int ns = 0; dosweep && ns < 100000; ns++) {
            fprev = f; f = sweep(s); it++;
            if (fabs(fprev - f) <= g_bm_sweeptol * (1 + fabs(f)) && ns >= 3) break;
            if (tlim > 0 && wtime() - t0 > tlim) break;
        }
        g_bm_tsw += wtime() - ts;
        int escape = 0;
        for (int round = 0; ; round++) {
            const double gn = fd_cert(s, P, y, v, c1, M);
            if (verbose >= 0) printf("bm: %6d sweeps %4d TR  rank %3d  obj %.10e  lambda_min(Z) %.2e  |ZR| %.1e  err2 %.1e  %.2fs\n", it, ntr, s->r, M->pobj, M->lmin, gn, M->e2, wtime() - t0);
            if (M->e2 <= tol && M->e3 <= tol && (!g_bm_cert || M->cgap <= tol)) {
                /* check the bound: the smallest delta = max(-lambda_min, 0) + eta 4^q for which Z + delta I
                 * factors (eta: a twentieth of the tolerance's room); a Lanczos estimate can be too high */
                const double eta = 0.05 * tol * (1 + fabs(M->pobj) + fabs(M->lb)) / rs;
                double delta = fmax(-M->lmin, 0) + eta; int vr = -1;
                const double tv0 = wtime(); double dlo = -1;
                if (g_bm_cert && g_bm_verify) for (int q = 0; q < 12; q++, delta = fmax(-M->lmin, 0) + eta * ldexp(1.0, 2 * q)) { vr = fd_verify(s, delta); if (vr != 0) break; dlo = delta; }
                if (vr == 1 && dlo > 0) {   /* bisect between the last shift that failed and the first that factored, while cheap */
                    const double tf = wtime() - tv0;
                    for (int b = 0; b < 6 && (wtime() - tv0) + tf < 0.1 * (wtime() - t0); b++) {
                        const double dm = sqrt(dlo * delta);
                        if (fd_verify(s, dm) == 1) delta = dm; else dlo = dm;
                    }
                }
                if (vr == 1) {
                    M->cert = 1; M->vdelta = delta; M->lmin = fmin(M->lmin, -delta + eta);
                    M->lb = M->dobj - delta * rs; M->cgap = (M->pobj - M->lb) / (1 + fabs(M->pobj) + fabs(M->lb));
                    M->e2 = fmax(0, delta - eta) / (1 + c1);
                    if (verbose > 0) printf("bm: the bound checked: Z + %.2e I factors\n", delta);
                }
                if (vr == 0 && verbose >= 0) printf("bm: Z + %.2e I does not factor: lambda_min is below the Lanczos estimate %.2e\n", delta, M->lmin);
                if (vr != 0 && M->e2 <= tol && M->cgap <= tol) { rc = 0; break; }
                if (vr == 0) { M->cgap = 1; nver++; if (nver >= 3) break; }
            }
            if (tlim > 0 && wtime() - t0 > tlim) break;
            /* a saddle (negative curvature dominates the gradient) or a rank too small: escape */
            if (!(fresh && round == 0) && M->lmin < 0 && -M->lmin * rnorm > 10 * gn && nesc < 40) { escape = 1; break; }
            if (!g_bm_newton || round >= (g_bm_rndfac > 0.05 ? 30 : 12)) { escape = M->lmin < 0 && nesc < 40; break; }
            const double tt = wtime();
            ntr += fd_newton(s, g_bm_rndfac * 2 * gn, 200, tlim, t0, verbose);
            g_bm_ttr += wtime() - tt;
        }
        if (rc == 0 || !escape || (tlim > 0 && wtime() - t0 > tlim)) break;
        if (g_bm_escp > 0) fd_escape(s, v, rnorm, M->pobj, verbose);
        else { add_col(s, v, 0.3 * rnorm); }
        nesc++; fresh = 1;   /* a trust-region round before the next escape (back-to-back escapes looped on amazon0505) */
        dosweep = g_bm_escsweep || g_bm_escp <= 0;
        if (dosweep) f = sweep(s);
    }
    if (verbose > 0) printf("bm: time in the sweeps %.2fs, the trust region %.2fs, Lanczos %.2fs, the bound's factorizations %.2fs; %d escapes\n", g_bm_tsw, g_bm_ttr, g_bm_tlz, g_bm_tvf, nesc);
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
        if (verbose >= 0) printf("bm: alm %3d  its %6ld  rank %3d  sigma %.1e  obj %.10e  pinf %.1e  gap %.1e  lmin %s%.2e  %.2fs\n", outer, nit, s->r, G.sigma, M->pobj, pinf, M->e3, check ? "" : "~", M->lmin, wtime() - t0);
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

/* 5.14: the automatic route: one semidefinite block of order n >= BRISK_BMAUTON (500) with m = n
 * constraints, from the file's header (or the data in memory); bm_run then checks that every
 * constraint fixes a diagonal entry */
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
    if (n0 >= nmin && m == n0) return 1;                                   /* a fixed diagonal, perhaps */
    if (8.0 * (double)m * (double)m > 0.5 * brisk_mem_limit()) return 2;  /* the Schur complement does not fit: entry fixings, perhaps */
    return 0;
}

int bm_run(const char *fname, Params *par, const char *yfile, const BriskData *data, BriskResult *res) {
    return bm_run_x(fname, par, yfile, NULL, NULL, data, res, 0);
}

int bm_run_x(const char *fname, Params *par, const char *yfile, const char *xfile, const char *zfile, const BriskData *data, BriskResult *res, int autom) {
    if (par->bm == 2 && !autom) return bmip_run(fname, par, yfile, data, res);   /* 5.16: the interior-point method on the factored program */
    const double t0 = wtime();
    Problem PP, *P = &PP;
    memset(P, 0, sizeof PP);
    const int maxn0 = g_read_maxn;
    g_read_maxn = 2000000000;
    if (data ? problem_from_sdpa_data(data, P) != 0 : read_sdpa(fname, P) != 0) { g_read_maxn = maxn0; return autom ? -1 : 2; }
    g_read_maxn = maxn0;
    if (!P->b0) { P->b0 = bx(sizeof(double) * (P->m + 1)); memcpy(P->b0, P->b, sizeof(double) * P->m); }
    const double tread = wtime() - t0;
    const int m = P->m, verbose = par->verbose;
    const double tol = par->tol;   /* (-lrtol is -lralm's) */
    int nsdp = 0, nlp = 0; for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) nsdp++; else nlp++; }
    double tlim_auto = 0;
    if (autom) {   /* 1: every constraint fixes a diagonal entry of the one block; 2: every constraint fixes one entry */
        int fx = nsdp == 1 && !nlp && P->blk[0].ncon == m;
        for (int t = 0; fx && t < P->blk[0].ncon; t++) { const SpSym *A = &P->blk[0].A[t]; if (A->nnz != 1 || (autom == 1 && (A->row[0] != A->col[0] || A->val[0] <= 0 || P->b[P->blk[0].con[t]] <= 0))) fx = 0; }
        if (!fx) { problem_free(P); return -1; }
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
    if (nsdp != 1) { fprintf(stderr, "brisk: -bm 1 needs exactly one semidefinite block for now\n"); return 2; }
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
    bb_setup(&s, B, kb);
    /* fixed diagonal? every constraint one diagonal entry, each row at most one */
    int fixed = B->ncon == m && !nlp;
    if (fixed) {
        s.rho = bx(sizeof(double) * s.n); s.adiag = bx(sizeof(double) * s.n); s.crow = bx(sizeof(int) * s.n);
        for (int i = 0; i < s.n; i++) s.crow[i] = -1;
        for (int t = 0; t < B->ncon && fixed; t++) {
            const SpSym *A = &B->A[t];
            if (A->nnz != 1 || A->row[0] != A->col[0] || A->val[0] <= 0 || P->b[B->con[t]] <= 0 || s.crow[A->row[0]] >= 0) { fixed = 0; break; }
            const int p = A->row[0];
            s.crow[p] = B->con[t]; s.adiag[p] = A->val[0]; s.rho[p] = P->b[B->con[t]] / A->val[0];
        }
        for (int i = 0; i < s.n && fixed; i++) if (s.crow[i] < 0) fixed = 0;
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
    if (verbose >= 0) printf("bm: %s, n = %d, m = %d, rank %d\n", fixed ? "fixed diagonal" : "general constraints (augmented Lagrangian)", s.n, m, s.r);
    const int rc = fixed ? fixed_diag(&s, P, tol, verbose, autom ? tlim_auto : par->timelimit, t0, &M, yv) : gen_alm(&s, P, nlp ? &LPd : NULL, tol, verbose, autom ? tlim_auto : par->timelimit, t0, &M, yv);
    const double tt = wtime() - t0;
    /* the DIMACS errors: X = R R' is positive semidefinite (err2 = 0), Z = C - A'y exactly (err3 = 0) */
    double err[7] = { 0, M.e1, 0, 0, M.e2, M.e3, M.e3 };
    const double acc = fmax(fmax(err[1], err[4]), err[5]);
    const int st = rc == 0 ? ST_OPTIMAL : acc <= par->red_acc ? ST_REDUCED : (par->timelimit > 0 && tt >= par->timelimit) ? ST_TIME : ST_MAXIT;
    const char *sn = st == ST_OPTIMAL ? "OPTIMAL" : st == ST_REDUCED ? "SOLVED TO REDUCED ACCURACY" : st == ST_TIME ? "TIME LIMIT" : "ITERATION LIMIT";
    if (verbose >= 0) {
        printf("status: %s (max rel error %.1e)   rank %d\n", sn, acc, s.r);
        printf("optimal value (SDPA/SDPLIB convention, max <F0,Y>): %.10e\n", -M.pobj);
        printf("primal objective %.12e  dual objective %.12e\n", M.pobj, M.dobj);
        printf("errors: primal %.1e  dual (lambda_min) %.1e  gap %.1e\n", M.e1, M.e2, M.e3);
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n", err[1], err[2], err[3], err[4], err[5], err[6]);
        if (fixed && M.cert) printf("certified lower bound %.12e (Z + %.2e I factored): gap to the objective %.1e\n", M.lb, M.vdelta, M.cgap);
        else if (fixed) printf("lower bound %.12e with lambda_min(Z) %.2e estimated by Lanczos (not checked): gap to the objective %.1e\n", M.lb, M.lmin, M.cgap);
        printf("  Time:              %.2f s (reading %.2f s)\n", tt, tread);
    }
    if (res) {   /* the library interface: status, objectives, errors and y; no X (X = R R' is not formed) */
        static const int ecode[7] = { 0, 11, 12, 13, 14, 10, 15 };
        res->status = st; res->exit_code = ecode[st];
        snprintf(res->status_str, sizeof(res->status_str), "%s", sn);
        res->iters = 0; res->pobj = M.pobj; res->dobj = M.dobj;
        for (int e = 1; e <= 6; e++) res->err[e] = err[e];
        res->time = tt; res->m = m; res->nblk = P->nblk;
        res->bs = malloc(sizeof(int) * (P->nblk + 1)); memcpy(res->bs, bs0, sizeof(int) * P->nblk);
        res->y = malloc(sizeof(double) * (m + 1)); memcpy(res->y, yv, sizeof(double) * m);
        res->have_x = 0; res->X = NULL; res->Z = NULL;
        res->bound_side = 0; res->bound_rigorous = NAN;
        snprintf(res->cause, sizeof(res->cause), "%s", st == ST_OPTIMAL ? "" : "Burer-Monteiro (-bm): the target accuracy was not reached");
    }
    if (yfile) { FILE *f = fopen(yfile, "w"); if (f) { for (int i = 0; i < m; i++) fprintf(f, "%.17g\n", yv[i]); fclose(f); } }
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
                FILE *f = fopen(fn, "w"); if (!f) { fprintf(stderr, "brisk: cannot write %s\n", fn); continue; }
                for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { const double v = M[i + (size_t)j * n]; if (v != 0) fprintf(f, "1 %d %d %.17g\n", i + 1, j + 1, v); }
                fclose(f);
            }
            if (res) { res->X = malloc(sizeof(double *) * 2); res->X[0] = X; X = NULL; res->Z = malloc(sizeof(double *) * 2); res->Z[0] = Z; Z = NULL; res->have_x = 1; }
        }
        free(X); free(Z); free(Rc);
    }
    bb_free(&s);
    free(LPd.lp_ptr); free(LPd.lp_con); free(LPd.lp_val); free(LPd.lp_c); free(yv); free(bs0);
    problem_free(P);
    return st == ST_OPTIMAL ? 0 : st == ST_REDUCED ? 10 : st == ST_TIME ? 15 : 13;
}
