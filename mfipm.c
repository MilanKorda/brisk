/* mfipm.c - BRISK 4.34: matrix-free primal-dual interior-point method (-mfipm 1).
 *
 * The NT direction of an infeasible primal-dual path-following method with Mehrotra's
 * predictor-corrector, where the Schur complement M = A (W (x) W) A* is never formed: the
 * Newton systems M dy = h are solved by preconditioned CG with
 *     M v = sum_b A_b( W_b A_b*(v) W_b )  (+ LP: A diag(x/z) A')    two symmetric GEMMs a block,
 * and the "scaled tangent" preconditioner of the handoff note (HANDOFF.md, note.pdf):
 *     P = A K A*,   K_b = D^1/2 [ Pi_T Wt(x)Wt Pi_T + tau_b^2 Pi_T^perp ] D^1/2,
 * D = diag(W_b), Wt = D^-1/2 W_b D^-1/2, T the tangent space of the outlier eigenvectors
 * of Wt (the few eigenvalues outside a bulk of spread <= rho), tau_b^2 the geometric centre
 * of the bulk. cond(P^-1 M) <= (bulk spread)^2, independent of mu when one side of the
 * optimal face has low rank. P = Dg + B diag(lam') B' is applied by Sherman-Morrison-
 * Woodbury: Dg = sum_b tau_b^2 diag(A_b (D (x) D) A_b*) (exact when every matrix entry
 * belongs to one constraint, "partial orthogonality", as in SOS/moment relaxations; the
 * diagonal part otherwise), B the images of the tangent pairs phi_ij = c_ij(q_i q_j' +
 * q_j q_i'), assembled through the sparse matrices Sq_i = [ (A_c q_i)' ]_c (m x n, never an
 * m x k array): B' Dg^-1 B has blocks Q' (Sq_i' Dg^-1 Sq_j) Q. The capacitance matrix
 * diag(1/lam') + B' Dg^-1 B is symmetric indefinite (dsytrf).
 *
 * Inexact Newton: the CG residual is moved out of the primal equation by the exact
 * projection dX += A*((A A*)^-1 (Rp - A dX)) (A A* factored once, fom.c's Gram matrix),
 * CG stops on the P^-1-norm residual relative to the right-hand side's, at
 * clamp(1e-2 mu/mu0, 1e-10, 1e-3). Memory O(sum n^2 + k^2 + nnz), no m x m object.
 *
 * Operates on the scaled problem (after problem_prepare), like fom.c; the result (y, X)
 * goes through the usual postsolve and is measured on the file data.                  */
#include "brisk.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
double wtime(void);

void BL(dsytrf_)(const char *, const int *, double *, const int *, int *, double *, const int *, int *);
void BL(dsytrs_)(const char *, const int *, const int *, const double *, const int *, const int *, double *, const int *, int *);
void BL(dgemv_)(const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);

static const int IONE = 1;
static const double DONE = 1.0, DZERO = 0.0, DMONE = -1.0;
static void *mx(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (matrix-free IPM, %.2f GB)\n", n / 1e9); exit(1); } return p; }
static void *mz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (matrix-free IPM, %.2f GB)\n", n / 1e9); exit(1); } return p; }

/* ---- per-block data ------------------------------------------------------------------ */
typedef struct {
    int type, n, ncon, *con;
    const Block *B;
    /* A* by columns (full expansion): colp[n+1], row, constraint, value */
    int *colp, *crow, *ccon; double *cval;
    double *C;                      /* dense n x n (SDP) or n (LP) */
    double *X, *Z, *dX, *dZ, *dXa, *dZa, *Rd, *E0, *Rc;   /* iterate, directions, residual, W Rd W, complementarity rhs */
    double *L, *G, *Y, *W, *ev, *dv;                  /* NT: X = L L', G = L U ev^-1/4, Y = Gi' (dXt = Y' dX Y), W = G G' */
    double *T1, *T2, *T3;                             /* work */
    /* preconditioner */
    double *dg, *Q, *wt;           /* diag(W), D^1/2 (eigvecs of Wt), eigenvalues of Wt (ascending) */
    double tau2, kbulk;
    int rlo, rhi;
    double *work; int lwork, *iwork, liwork;
} MB;

typedef struct {                   /* one outlier eigenvector: its tangent pairs */
    int b, i, k;                   /* block, eigen-index, number of pairs */
    int off;                       /* offset in the capacitance ordering */
    int *sp, *sj; double *sv;      /* Sq (local constraint rows): CSR */
    int *tp, *ti; double *tv;      /* Sq by columns (CSC): column p -> (local row, value) */
    double *QJ;                    /* n x k: Q[:, J] with the pair scale (1 or sqrt 2) */
    double *u;                     /* n work */
} Outl;

typedef struct {
    int m, nb, N;                  /* constraints, blocks, sum of block orders (LP counts n) */
    MB *b;
    Gram *G;                       /* A A* (the exact primal projection) */
    double *Dg, *Dgi;              /* the diagonal part of the preconditioner */
    int no; Outl *o;               /* outliers */
    int k; double *cap; int *ipiv; double *capwork; int lcw;
    double *tk, *tk2;              /* k work */
    double *lamp;                  /* pair eigenvalues (kept for checks) */
    int **pos;                     /* per block: global constraint -> local index, -1 */
    long ncg, nmv;
    double t_mv, t_prec, t_setup, t_nt;
} MF;

static size_t blen(const MB *b) { return b->type == BLK_LP ? (size_t)b->n : (size_t)b->n * b->n; }

static void mb_init(MF *S, const Problem *P) {
    S->m = P->m; S->nb = P->nblk; S->b = mz(sizeof(MB) * P->nblk); S->N = 0;
    S->pos = mz(sizeof(int *) * P->nblk);
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; MB *b = &S->b[k];
        b->B = B; b->type = B->type; b->n = B->n; b->ncon = B->ncon; b->con = B->con;
        const int n = b->n; const size_t len = blen(b);
        S->N += n;
        b->C = mz(sizeof(double) * len);
        if (b->type == BLK_LP) { for (int q = 0; q < B->C.nnz; q++) b->C[B->C.row[q]] += B->C.val[q]; }
        else for (int q = 0; q < B->C.ef; q++) b->C[B->C.fr[q] + (size_t)B->C.fc[q] * n] += B->C.fv[q];
        b->X = mz(sizeof(double) * len); b->Z = mz(sizeof(double) * len);
        b->dX = mz(sizeof(double) * len); b->dZ = mz(sizeof(double) * len);
        b->dXa = mz(sizeof(double) * len); b->dZa = mz(sizeof(double) * len);
        b->Rd = mz(sizeof(double) * len); b->E0 = mz(sizeof(double) * len); b->Rc = mz(sizeof(double) * len);
        b->W = mz(sizeof(double) * len); b->dg = mz(sizeof(double) * n);
        b->ev = mz(sizeof(double) * n); b->dv = mz(sizeof(double) * n);
        S->pos[k] = mx(sizeof(int) * (S->m + 1));
        for (int c = 0; c < S->m; c++) S->pos[k][c] = -1;
        for (int t = 0; t < b->ncon; t++) S->pos[k][b->con[t]] = t;
        b->T1 = mz(sizeof(double) * len); b->T2 = mz(sizeof(double) * len); b->T3 = mz(sizeof(double) * len);
        if (b->type == BLK_LP) continue;
        b->L = mx(sizeof(double) * len); b->G = mx(sizeof(double) * len); b->Y = mx(sizeof(double) * len);
        b->Q = mx(sizeof(double) * len); b->wt = mz(sizeof(double) * n);
        /* A* by columns */
        size_t ef = 0;
        for (int t = 0; t < B->ncon; t++) ef += B->A[t].ef;
        b->colp = mz(sizeof(int) * (n + 2));
        for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) b->colp[B->A[t].fc[q] + 1]++;
        for (int j = 0; j < n; j++) b->colp[j + 1] += b->colp[j];
        b->crow = mx(sizeof(int) * (ef + 1)); b->ccon = mx(sizeof(int) * (ef + 1)); b->cval = mx(sizeof(double) * (ef + 1));
        int *fill = mx(sizeof(int) * (n + 1)); memcpy(fill, b->colp, sizeof(int) * n);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t];
            for (int q = 0; q < A->ef; q++) { const int e = fill[A->fc[q]]++; b->crow[e] = A->fr[q]; b->ccon[e] = B->con[t]; b->cval[e] = A->fv[q]; }
        }
        free(fill);
        double wq; int iq, info, lm1 = -1;
        BL(dsyevd_)("V", "L", &n, b->T1, &n, b->ev, &wq, &lm1, &iq, &lm1, &info);
        b->lwork = (int)wq + 1; b->liwork = iq;
        { double wq2; int iq2, mf, il = 1, iu = 1; const double at = 0; int *isz = mx(sizeof(int) * (2 * n + 2));
          BL(dsyevr_)("N", "I", "L", &n, b->T1, &n, &DZERO, &DZERO, &il, &iu, &at, &mf, b->ev, b->T2, &n, isz, &wq2, &lm1, &iq2, &lm1, &info);
          free(isz);
          if ((int)wq2 + 1 > b->lwork) b->lwork = (int)wq2 + 1;
          if (iq2 > b->liwork) b->liwork = iq2; }
        b->work = mx(sizeof(double) * b->lwork); b->iwork = mx(sizeof(int) * (b->liwork + 2 * n));
    }
}

/* out = A(X) over all blocks (Xs: block arrays) */
static void op_A(MF *S, double **Xs, double *out) {
    memset(out, 0, sizeof(double) * S->m);
    for (int k = 0; k < S->nb; k++) {
        const MB *b = &S->b[k]; const Block *B = b->B; const double *X = Xs[k]; const int n = b->n;
        if (b->type == BLK_LP) {
            for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; double s = 0; for (int q = 0; q < A->nnz; q++) s += A->val[q] * X[A->row[q]]; out[B->con[t]] += s; }
            continue;
        }
        #pragma omp parallel for schedule(dynamic, 64) if (B->ncon > 4096)
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t]; double s = 0;
            for (int q = 0; q < A->ef; q++) s += A->fv[q] * X[A->fr[q] + (size_t)A->fc[q] * n];
            out[B->con[t]] += s;
        }
    }
}
/* T = A_b*(y) (dense full, or LP vector) */
static void op_At(const MB *b, const double *y, double *T) {
    const int n = b->n;
    if (b->type == BLK_LP) {
        memset(T, 0, sizeof(double) * n);
        const Block *B = b->B;
        for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; const double yc = y[B->con[t]]; for (int q = 0; q < A->nnz; q++) T[A->row[q]] += yc * A->val[q]; }
        return;
    }
    #pragma omp parallel for schedule(static) if (n > 256)
    for (int j = 0; j < n; j++) {
        double *col = T + (size_t)j * n;
        memset(col, 0, sizeof(double) * n);
        for (int e = b->colp[j]; e < b->colp[j + 1]; e++) col[b->crow[e]] += y[b->ccon[e]] * b->cval[e];
    }
}
static double dotn(size_t n, const double *a, const double *b) { double s = 0; for (size_t i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static void symmetrize(int n, double *A) { for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) { const double v = 0.5 * (A[i + (size_t)j * n] + A[j + (size_t)i * n]); A[i + (size_t)j * n] = A[j + (size_t)i * n] = v; } }
/* O = W V W (V symmetric), via T scratch: 4 n^3 */
static void wvw(const MB *b, const double *V, double *O, double *T) {
    const int n = b->n;
    if (b->type == BLK_LP) { for (int i = 0; i < n; i++) O[i] = b->W[i] * b->W[i] * V[i]; return; }
    BL(dsymm_)("L", "L", &n, &n, &DONE, b->W, &n, V, &n, &DZERO, T, &n);     /* T = W V */
    BL(dsymm_)("R", "L", &n, &n, &DONE, b->W, &n, T, &n, &DZERO, O, &n);     /* O = T W */
}

/* M v */
static void matvec(MF *S, const double *v, double *out) {
    const double t0 = wtime();
    memset(out, 0, sizeof(double) * S->m);
    for (int k = 0; k < S->nb; k++) {
        MB *b = &S->b[k]; const Block *B = b->B; const int n = b->n;
        if (b->type == BLK_LP) {
            double *u = b->T1;
            op_At(b, v, u);
            for (int i = 0; i < n; i++) u[i] *= b->W[i] * b->W[i];
            for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; double s = 0; for (int q = 0; q < A->nnz; q++) s += A->val[q] * u[A->row[q]]; out[B->con[t]] += s; }
            continue;
        }
        op_At(b, v, b->T1);
        wvw(b, b->T1, b->T3, b->T2);
        #pragma omp parallel for schedule(dynamic, 64) if (B->ncon > 4096)
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t]; double s = 0;
            for (int q = 0; q < A->ef; q++) s += A->fv[q] * b->T3[A->fr[q] + (size_t)A->fc[q] * n];
            out[B->con[t]] += s;
        }
    }
    S->nmv++; S->t_mv += wtime() - t0;
}

/* ---- NT scaling -------------------------------------------------------------------------- */
static int nt_block(MB *b) {
    const int n = b->n; int info = 0;
    if (b->type == BLK_LP) {
        for (int i = 0; i < n; i++) { const double x = b->X[i], z = b->Z[i]; b->W[i] = sqrt(x / z); b->dv[i] = sqrt(x * z); }
        return 0;
    }
    const size_t len = (size_t)n * n;
    memcpy(b->L, b->X, sizeof(double) * len);
    BL(dpotrf_)("L", &n, b->L, &n, &info); if (info) return 1;
    memcpy(b->T2, b->Z, sizeof(double) * len);
    BL(dpotrf_)("L", &n, b->T2, &n, &info); if (info) return 2;
    /* K = R' L (R = chol Z lower), T = K' K = L' Z L */
    for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) { b->L[i + (size_t)j * n] = 0; b->T2[i + (size_t)j * n] = 0; }
    memcpy(b->T1, b->L, sizeof(double) * len);
    BL(dtrmm_)("L", "L", "T", "N", &n, &n, &DONE, b->T2, &n, b->T1, &n);
    BL(dsyrk_)("L", "T", &n, &n, &DONE, b->T1, &n, &DZERO, b->T3, &n);
    int lw = b->lwork, liw = b->liwork;
    BL(dsyevd_)("V", "L", &n, b->T3, &n, b->ev, b->work, &lw, b->iwork, &liw, &info); if (info) return 3;
    for (int i = 0; i < n; i++) { if (!(b->ev[i] > 1e-300)) b->ev[i] = 1e-300; b->dv[i] = sqrt(b->ev[i]); }
    /* G = L U ev^-1/4 ; Y = L^-T U ev^1/4 (= Gi') */
    for (int j = 0; j < n; j++) {
        const double s = pow(b->ev[j], -0.25), s2 = pow(b->ev[j], 0.25);
        for (int i = 0; i < n; i++) { b->G[i + (size_t)j * n] = b->T3[i + (size_t)j * n] * s; b->Y[i + (size_t)j * n] = b->T3[i + (size_t)j * n] * s2; }
    }
    BL(dtrmm_)("L", "L", "N", "N", &n, &n, &DONE, b->L, &n, b->G, &n);
    BL(dtrsm_)("L", "L", "T", "N", &n, &n, &DONE, b->L, &n, b->Y, &n);
    BL(dsyrk_)("L", "N", &n, &n, &DONE, b->G, &n, &DZERO, b->W, &n);
    for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) b->W[j + (size_t)i * n] = b->W[i + (size_t)j * n];
    return 0;
}
/* scaled directions: Xt = Y' dX Y (= Gi dX Gi'), Zt = G' dZ G */
static void scaled_dirs(MB *b, const double *dX, const double *dZ, double *Xt, double *Zt) {
    const int n = b->n;
    if (b->type == BLK_LP) { for (int i = 0; i < n; i++) { Xt[i] = dX[i] / b->W[i]; Zt[i] = dZ[i] * b->W[i]; } return; }
    BL(dsymm_)("L", "L", &n, &n, &DONE, dX, &n, b->Y, &n, &DZERO, b->T1, &n);
    BL(dgemm_)("T", "N", &n, &n, &n, &DONE, b->Y, &n, b->T1, &n, &DZERO, Xt, &n);
    BL(dsymm_)("L", "L", &n, &n, &DONE, dZ, &n, b->G, &n, &DZERO, b->T1, &n);
    BL(dgemm_)("T", "N", &n, &n, &n, &DONE, b->G, &n, b->T1, &n, &DZERO, Zt, &n);
}
/* largest a with diag(dv) + a Mt >= 0 (Mt scaled direction; destroyed) */
static double max_step(MB *b, double *Mt) {
    const int n = b->n;
    double lmin;
    if (b->type == BLK_LP) { lmin = 0; for (int i = 0; i < n; i++) { const double r = Mt[i] / b->dv[i]; if (r < lmin) lmin = r; } }
    else {
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Mt[i + (size_t)j * n] /= sqrt(b->dv[i] * b->dv[j]);
        int il = 1, iu = 1, mf = 0, info = 0, lw = b->lwork, liw = b->liwork; const double at = 0; double w1[2];
        BL(dsyevr_)("N", "I", "L", &n, Mt, &n, &DZERO, &DZERO, &il, &iu, &at, &mf, w1, NULL, &n, b->iwork + b->liwork, b->work, &lw, b->iwork, &liw, &info);
        lmin = info ? -1e300 : w1[0];
    }
    return lmin < 0 ? -1.0 / lmin : 1e300;
}

/* ---- preconditioner ------------------------------------------------------------------ */
static void split_spectrum(const double *w, int n, double rho, int rmax, int *rlo, int *rhi) {
    double bw = 1e300; int blo = 0, bhi = 0;
    for (int r = 0; r <= rmax && r < n - 1; r++) {
        double cw = 1e300; int clo = 0, chi = 0;
        for (int lo = 0; lo <= r; lo++) { const int hi = r - lo; const double width = log(w[n - 1 - hi]) - log(w[lo]); if (width < cw) { cw = width; clo = lo; chi = hi; } }
        bw = cw; blo = clo; bhi = chi;
        if (cw <= log(rho)) break;
    }
    (void)bw; *rlo = blo; *rhi = bhi;
}
typedef struct { double key; int o, j; } PairKey;
static int cmp_pk(const void *a, const void *b) { const double x = ((const PairKey *)a)->key, y = ((const PairKey *)b)->key; return x < y ? 1 : x > y ? -1 : 0; }

static void prec_free_outl(MF *S) {
    for (int a = 0; a < S->no; a++) { Outl *o = &S->o[a]; free(o->sp); free(o->sj); free(o->sv); free(o->tp); free(o->ti); free(o->tv); free(o->QJ); free(o->u); }
    free(S->o); S->o = NULL; S->no = 0;
}

static void prec_setup(MF *S, const Params *par, double *lamp_out_kmax) {
    const double t0 = wtime();
    const double rho = par->mf_rho, drop = par->mf_drop;
    const int rmax = par->mf_rmax, kmax = par->mf_kmax;
    const int m = S->m;
    prec_free_outl(S);
    /* per block: Jacobi-scaled W, its spectrum, the bulk */
    int nout = 0;
    for (int k = 0; k < S->nb; k++) {
        MB *b = &S->b[k]; const int n = b->n;
        if (b->type == BLK_LP) { for (int i = 0; i < n; i++) b->dg[i] = b->W[i] * b->W[i]; b->tau2 = 1; b->rlo = b->rhi = 0; continue; }
        for (int i = 0; i < n; i++) { double d = b->W[i + (size_t)i * n]; b->dg[i] = d > 1e-300 ? d : 1e-300; }
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) b->Q[i + (size_t)j * n] = b->W[i + (size_t)j * n] / sqrt(b->dg[i] * b->dg[j]);
        int info = 0, lw = b->lwork, liw = b->liwork;
        if (rmax > 0) BL(dsyevd_)("V", "L", &n, b->Q, &n, b->wt, b->work, &lw, b->iwork, &liw, &info);
        else { BL(dsyevd_)("N", "L", &n, b->Q, &n, b->wt, b->work, &lw, b->iwork, &liw, &info); }
        for (int i = 0; i < n; i++) if (!(b->wt[i] > 1e-300)) b->wt[i] = 1e-300;
        split_spectrum(b->wt, n, rho, rmax, &b->rlo, &b->rhi);
        const double wlo = b->wt[b->rlo], whi = b->wt[n - 1 - b->rhi];
        b->tau2 = wlo * whi; b->kbulk = (whi / wlo) * (whi / wlo);
        if (rmax > 0) for (int j = 0; j < n; j++) { for (int i = 0; i < n; i++) b->Q[i + (size_t)j * n] *= sqrt(b->dg[i]); }
        nout += b->rlo + b->rhi;
    }
    /* Dg = sum_b tau_b^2 diag(A_b (D (x) D) A_b*) */
    memset(S->Dg, 0, sizeof(double) * m);
    for (int k = 0; k < S->nb; k++) {
        const MB *b = &S->b[k]; const Block *B = b->B;
        if (b->type == BLK_LP) {
            for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; double s = 0; for (int q = 0; q < A->nnz; q++) s += A->val[q] * A->val[q] * b->dg[A->row[q]]; S->Dg[B->con[t]] += s; }
            continue;
        }
        #pragma omp parallel for schedule(dynamic, 64)
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t]; double s = 0;
            for (int q = 0; q < A->ef; q++) s += A->fv[q] * A->fv[q] * b->dg[A->fr[q]] * b->dg[A->fc[q]];
            S->Dg[B->con[t]] += b->tau2 * s;
        }
    }
    if (par->mf_diag) {
        /* 4.35: the exact diagonal of M, M_cc = sum_b <A_c, W_b A_c W_b> = sum_{e,e'} a_e a_e' W[r_e, r_e'] W[c_e, c_e']
         * over the full-expansion entries (O(nnz_c^2) a row, cheaper than one matvec), minus the
         * diagonal of the outlier correction B lam' B' (so that diag(P) = diag(M)): for outlier i
         * and v = A_c q_i, sum_j (w_i w_j - tau^2) c_ij^2 (v'q_j)^2 = 2 (w_i v'Wv - tau^2 v'Dv)
         * - (w_i^2 - tau^2) (v'q_i)^2 up to the pairs dropped or capped. The base is then right
         * for any W on the diagonal; the correction models the outlier pairs off it. */
        memset(S->Dg, 0, sizeof(double) * m);
        for (int k = 0; k < S->nb; k++) {
            const MB *b = &S->b[k]; const Block *B = b->B; const int n = b->n;
            if (b->type == BLK_LP) {
                for (int t = 0; t < B->ncon; t++) { const SpSym *A = &B->A[t]; double s = 0; for (int q = 0; q < A->nnz; q++) s += A->val[q] * A->val[q] * b->dg[A->row[q]]; S->Dg[B->con[t]] += s; }
                continue;
            }
            const double *W = b->W; const int nout_b = rmax > 0 ? b->rlo + b->rhi : 0;
            #pragma omp parallel
            {
                double *acc = mz(sizeof(double) * n); int *mark = mx(sizeof(int) * n), *lst = mx(sizeof(int) * n);
                for (int i = 0; i < n; i++) mark[i] = -1;
                #pragma omp for schedule(dynamic, 64)
                for (int t = 0; t < B->ncon; t++) {
                    const SpSym *A = &B->A[t]; double s = 0;
                    for (int q = 0; q < A->ef; q++) {
                        const int r = A->fr[q], c = A->fc[q]; const double a = A->fv[q];
                        const double *Wr = W + (size_t)r * n, *Wc = W + (size_t)c * n;
                        double sq = 0;
                        for (int u = 0; u < A->ef; u++) sq += A->fv[u] * Wr[A->fr[u]] * Wc[A->fc[u]];
                        s += a * sq;
                    }
                    for (int u = 0; u < nout_b; u++) {
                        const int i = u < b->rlo ? u : n - 1 - (u - b->rlo);
                        const double *qi = b->Q + (size_t)i * n;
                        int nl = 0;
                        for (int e = 0; e < A->ef; e++) { const int r = A->fr[e]; if (mark[r] != t) { mark[r] = t; lst[nl++] = r; acc[r] = 0; } acc[r] += A->fv[e] * qi[A->fc[e]]; }
                        double vWv = 0, vDv = 0, vq = 0;
                        for (int l = 0; l < nl; l++) {
                            const int r = lst[l]; const double vl = acc[r];
                            vDv += vl * vl * b->dg[r]; vq += vl * qi[r];
                            const double *Wr = W + (size_t)r * n; double row = 0;
                            for (int l2 = 0; l2 < nl; l2++) row += acc[lst[l2]] * Wr[lst[l2]];
                            vWv += vl * row;
                        }
                        const double wi = b->wt[i];
                        s -= 2.0 * (wi * vWv - b->tau2 * vDv) - (wi * wi - b->tau2) * vq * vq;
                        mark[lst[0]] = -1;   /* the marks are per (t, i): reset by using a fresh tag below */
                        for (int l = 0; l < nl; l++) mark[lst[l]] = -1;
                    }
                    S->Dg[B->con[t]] += s > 0 ? s : 0;
                }
                free(acc); free(mark); free(lst);
            }
        }
    }
    if (getenv("BRISK_MFDIAGCHK") && S->k == 0) {
        /* check of the base against the true diagonal of M (unit-vector products), sample rows */
        double *e = mz(sizeof(double) * m), *q = mx(sizeof(double) * m);
        double emax = 0, rmax_ = 0, rmin_ = 1e300;
        for (int c = 0; c < m; c += (m > 200 ? m / 100 : 1)) {
            e[c] = 1; matvec(S, e, q); e[c] = 0;
            const double ratio = S->Dg[c] / q[c]; emax = fmax(emax, fabs(ratio - 1)); rmax_ = fmax(rmax_, ratio); rmin_ = fmin(rmin_, ratio);
        }
        printf("   [diag check (no outliers): base / true diagonal in [%.3e, %.3e]]\n", rmin_, rmax_);
        free(e); free(q);
    }
    double dmax = 0; for (int c = 0; c < m; c++) dmax = fmax(dmax, S->Dg[c]);
    for (int c = 0; c < m; c++) S->Dgi[c] = S->Dg[c] > 1e-300 * dmax && S->Dg[c] > 0 ? 1.0 / S->Dg[c] : (dmax > 0 ? 1.0 / dmax : 1.0);
    S->k = 0;
    if (nout == 0) { S->t_prec += wtime() - t0; if (lamp_out_kmax) *lamp_out_kmax = 0; return; }
    /* candidate pairs: (outlier i of block b, any j not an earlier outlier), ranked by |w_i w_j / tau^2 - 1| */
    S->o = mz(sizeof(Outl) * nout); S->no = nout;
    size_t npk = 0;
    for (int k = 0, a = 0; k < S->nb; k++) {
        const MB *b = &S->b[k]; const int n = b->n;
        for (int u = 0; u < b->rlo + b->rhi; u++, a++) { S->o[a].b = k; S->o[a].i = u < b->rlo ? u : n - 1 - (u - b->rlo); npk += n; }
    }
    PairKey *pk = mx(sizeof(PairKey) * (npk + 1)); size_t q = 0;
    for (int a = 0; a < nout; a++) {
        const MB *b = &S->b[S->o[a].b]; const int n = b->n, i = S->o[a].i;
        for (int j = 0; j < n; j++) {
            const int jout = j < b->rlo || j >= n - b->rhi;
            if (jout && j < i) continue;                  /* unordered pairs of outliers once */
            const double r = fabs(b->wt[i] * b->wt[j] / b->tau2 - 1.0);
            if (r < drop) continue;
            pk[q].key = r; pk[q].o = a; pk[q].j = j; q++;
        }
    }
    npk = q;
    if ((long)npk > kmax) { qsort(pk, npk, sizeof(PairKey), cmp_pk); npk = kmax; }   /* the pairs farthest from the bulk model */
    S->k = (int)npk;
    for (size_t p = 0; p < npk; p++) S->o[pk[p].o].k++;
    { int off = 0; for (int a = 0; a < nout; a++) { S->o[a].off = off; off += S->o[a].k; } }
    /* QJ, the pair eigenvalues lam' = w_i w_j - tau^2 */
    double *lamp = mx(sizeof(double) * (S->k + 1));
    { int *fill = mz(sizeof(int) * nout);
      for (int a = 0; a < nout; a++) S->o[a].QJ = mx(sizeof(double) * (size_t)S->b[S->o[a].b].n * (S->o[a].k + 1));
      for (size_t p = 0; p < npk; p++) {
          Outl *o = &S->o[pk[p].o]; const MB *b = &S->b[o->b]; const int n = b->n, j = pk[p].j, c = fill[pk[p].o]++;
          const double sc = j == o->i ? 1.0 : sqrt(2.0);
          for (int r = 0; r < n; r++) o->QJ[r + (size_t)c * n] = sc * b->Q[r + (size_t)j * n];
          lamp[o->off + c] = b->wt[o->i] * b->wt[j] - b->tau2;
      }
      free(fill); }
    free(pk);
    /* Sq_a: row t (local constraint) = (A_t q_i)', merged by column; and its transpose */
    for (int a = 0; a < nout; a++) {
        Outl *o = &S->o[a]; const MB *b = &S->b[o->b]; const Block *B = b->B; const int n = b->n;
        const double *qv = b->Q + (size_t)o->i * n;
        o->u = mx(sizeof(double) * (n + 1));
        int *cnt = mz(sizeof(int) * (b->ncon + 1));
        size_t tot = 0;
        for (int t = 0; t < b->ncon; t++) tot += B->A[t].ef;
        int *sj = mx(sizeof(int) * (tot + 1)); double *sv = mx(sizeof(double) * (tot + 1));
        int *sp = mz(sizeof(int) * (b->ncon + 1));
        { size_t o0 = 0; for (int t = 0; t < b->ncon; t++) { sp[t] = (int)o0; o0 += B->A[t].ef; } sp[b->ncon] = (int)o0; }
        #pragma omp parallel
        {
            double *acc = mz(sizeof(double) * n); int *mark = mx(sizeof(int) * n), *lst = mx(sizeof(int) * n);
            for (int i = 0; i < n; i++) mark[i] = -1;
            #pragma omp for schedule(dynamic, 64)
            for (int t = 0; t < b->ncon; t++) {
                const SpSym *A = &B->A[t]; int nl = 0;
                for (int e = 0; e < A->ef; e++) { const int r = A->fr[e]; if (mark[r] != t) { mark[r] = t; lst[nl++] = r; acc[r] = 0; } acc[r] += A->fv[e] * qv[A->fc[e]]; }
                size_t w = sp[t];
                for (int l = 0; l < nl; l++) { sj[w] = lst[l]; sv[w] = acc[lst[l]]; w++; }
                cnt[t] = nl;
            }
            free(acc); free(mark); free(lst);
        }
        /* compact */
        o->sp = mx(sizeof(int) * (b->ncon + 1)); size_t w = 0;
        for (int t = 0; t < b->ncon; t++) { o->sp[t] = (int)w; for (int e = 0; e < cnt[t]; e++) { sj[w] = sj[sp[t] + e]; sv[w] = sv[sp[t] + e]; w++; } }
        o->sp[b->ncon] = (int)w;
        o->sj = realloc(sj, sizeof(int) * (w + 1)); o->sv = realloc(sv, sizeof(double) * (w + 1));
        free(sp); free(cnt);
        /* CSC */
        o->tp = mz(sizeof(int) * (n + 2)); o->ti = mx(sizeof(int) * (w + 1)); o->tv = mx(sizeof(double) * (w + 1));
        for (size_t e = 0; e < w; e++) o->tp[o->sj[e] + 1]++;
        for (int j = 0; j < n; j++) o->tp[j + 1] += o->tp[j];
        int *f2 = mx(sizeof(int) * (n + 1)); memcpy(f2, o->tp, sizeof(int) * n);
        for (int t = 0; t < b->ncon; t++) for (int e = o->sp[t]; e < o->sp[t + 1]; e++) { const int f = f2[o->sj[e]]++; o->ti[f] = t; o->tv[f] = o->sv[e]; }
        free(f2);
    }
    /* capacitance: diag(1/lam') + B' Dg^-1 B, blocks QJ_a' (Sq_a' Dg^-1 Sq_c) QJ_c */
    const int kk = S->k;
    free(S->cap); S->cap = mx(sizeof(double) * (size_t)kk * kk);
    memset(S->cap, 0, sizeof(double) * (size_t)kk * kk);
    int maxn = 0; for (int k = 0; k < S->nb; k++) if (S->b[k].n > maxn) maxn = S->b[k].n;
    double *Sm = mx(sizeof(double) * (size_t)maxn * maxn), *Ym = mx(sizeof(double) * (size_t)maxn * (kk + 1));
    for (int a = 0; a < nout; a++) {
        const Outl *oa = &S->o[a]; const MB *ba = &S->b[oa->b]; const int na = ba->n;
        if (oa->k == 0) continue;
        for (int c = a; c < nout; c++) {
            const Outl *oc = &S->o[c]; const MB *bc = &S->b[oc->b]; const int nc = bc->n;
            if (oc->k == 0) continue;
            const int *posc = S->pos[oc->b];
            /* Sm (na x nc) = Sq_a' Dg^-1 Sq_c, rows p in parallel */
            #pragma omp parallel for schedule(dynamic, 16)
            for (int p = 0; p < na; p++) {
                for (int qq = 0; qq < nc; qq++) Sm[p + (size_t)qq * na] = 0;
                for (int e = oa->tp[p]; e < oa->tp[p + 1]; e++) {
                    const int gc = ba->con[oa->ti[e]], tc = posc[gc];
                    if (tc < 0) continue;
                    const double wv = oa->tv[e] * S->Dgi[gc];
                    for (int f = oc->sp[tc]; f < oc->sp[tc + 1]; f++) Sm[p + (size_t)oc->sj[f] * na] += wv * oc->sv[f];
                }
            }
            /* Ym = Sm QJ_c (na x kc), cap block = QJ_a' Ym */
            BL(dgemm_)("N", "N", &na, &oc->k, &nc, &DONE, Sm, &na, oc->QJ, &nc, &DZERO, Ym, &na);
            BL(dgemm_)("T", "N", &oa->k, &oc->k, &na, &DONE, oa->QJ, &na, Ym, &na, &DZERO, Sm, &oa->k);
            for (int jj = 0; jj < oc->k; jj++) for (int ii = 0; ii < oa->k; ii++) {
                const double v = Sm[ii + (size_t)jj * oa->k];
                S->cap[(oa->off + ii) + (size_t)(oc->off + jj) * kk] = v;
                S->cap[(oc->off + jj) + (size_t)(oa->off + ii) * kk] = v;
            }
        }
    }
    free(Sm); free(Ym);
    for (int i = 0; i < kk; i++) S->cap[i + (size_t)i * kk] += 1.0 / lamp[i];
    free(S->lamp); S->lamp = lamp;
    if (getenv("BRISK_MFDIAGCHK")) {
        /* diag(P) = Dg + sum_pairs lam' b_pair(c)^2 against the true diagonal of M */
        double *e = mz(sizeof(double) * m), *q = mx(sizeof(double) * m);
        double rmax_ = 0, rmin_ = 1e300, bmax = 0, bmin = 1e300;
        for (int c = 0; c < m; c += (m > 200 ? m / 100 : 1)) {
            e[c] = 1; matvec(S, e, q); e[c] = 0;
            double dp = S->Dg[c];
            for (int a = 0; a < nout; a++) {
                const Outl *o = &S->o[a]; const MB *b = &S->b[o->b]; const int n = b->n; const int t = S->pos[o->b][c];
                if (t < 0 || o->k == 0) continue;
                for (int j = 0; j < o->k; j++) {
                    double v = 0; for (int f = o->sp[t]; f < o->sp[t + 1]; f++) v += o->sv[f] * o->QJ[o->sj[f] + (size_t)j * n];
                    dp += lamp[o->off + j] * v * v;
                }
            }
            const double ratio = dp / q[c], rb = S->Dg[c] / q[c];
            rmax_ = fmax(rmax_, ratio); rmin_ = fmin(rmin_, ratio); bmax = fmax(bmax, rb); bmin = fmin(bmin, rb);
        }
        printf("   [diag check (%d pairs): diag(P) / diag(M) in [%.3e, %.3e], base / diag(M) in [%.3e, %.3e]]\n", kk, rmin_, rmax_, bmin, bmax);
        free(e); free(q);
    }
    free(S->ipiv); S->ipiv = mx(sizeof(int) * (kk + 1));
    { int lw = -1, info = 0; double wq; BL(dsytrf_)("L", &kk, S->cap, &kk, S->ipiv, &wq, &lw, &info);
      lw = (int)wq + 1; free(S->capwork); S->capwork = mx(sizeof(double) * lw); S->lcw = lw;
      BL(dsytrf_)("L", &kk, S->cap, &kk, S->ipiv, S->capwork, &lw, &info);
      if (info < 0) { fprintf(stderr, "brisk: matrix-free IPM: capacitance factorization failed (%d)\n", info); exit(1); } }
    free(S->tk); free(S->tk2); S->tk = mx(sizeof(double) * (kk + 1)); S->tk2 = mx(sizeof(double) * (kk + 1));
    S->t_prec += wtime() - t0;
    if (lamp_out_kmax) *lamp_out_kmax = kk;
}

/* z = P^-1 r */
static void prec_apply(MF *S, const double *r, double *z) {
    const double t0 = wtime();
    const int m = S->m;
    for (int c = 0; c < m; c++) z[c] = S->Dgi[c] * r[c];
    if (S->k > 0) {
        /* tk = B' z */
        for (int a = 0; a < S->no; a++) {
            Outl *o = &S->o[a]; const MB *b = &S->b[o->b]; const int n = b->n;
            if (o->k == 0) continue;
            #pragma omp parallel for schedule(static) if (n > 512)
            for (int p = 0; p < n; p++) { double s = 0; for (int e = o->tp[p]; e < o->tp[p + 1]; e++) s += o->tv[e] * z[b->con[o->ti[e]]]; o->u[p] = s; }
            BL(dgemv_)("T", &n, &o->k, &DONE, o->QJ, &n, o->u, &IONE, &DZERO, S->tk + o->off, &IONE);
        }
        int info = 0;
        BL(dsytrs_)("L", &S->k, &IONE, S->cap, &S->k, S->ipiv, S->tk, &S->k, &info);
        /* z -= Dg^-1 B tk */
        for (int a = 0; a < S->no; a++) {
            Outl *o = &S->o[a]; const MB *b = &S->b[o->b]; const int n = b->n;
            if (o->k == 0) continue;
            BL(dgemv_)("N", &n, &o->k, &DONE, o->QJ, &n, S->tk + o->off, &IONE, &DZERO, o->u, &IONE);
            #pragma omp parallel for schedule(static) if (b->ncon > 8192)
            for (int t = 0; t < b->ncon; t++) {
                double s = 0; for (int e = o->sp[t]; e < o->sp[t + 1]; e++) s += o->sv[e] * o->u[o->sj[e]];
                const int c = b->con[t]; z[c] -= S->Dgi[c] * s;
            }
        }
    }
    S->t_prec += wtime() - t0;
}

/* PCG on M x = h; stop on the P^-1-norm residual relative to that of h */
static int pcg(MF *S, const double *h, double *x, double tol, int maxit, double tmax, double *res_out, int warm) {
    const int m = S->m;
    double *r = mx(sizeof(double) * m), *z = mx(sizeof(double) * m), *p = mx(sizeof(double) * m), *q = mx(sizeof(double) * m);
    memcpy(r, h, sizeof(double) * m);
    prec_apply(S, r, z);
    double rz = dotn(m, r, z);
    const double nb = sqrt(fabs(rz));     /* the tolerance is relative to |h|_{P^-1} whatever the start */
    int k = 0; double res = 1.0;
    if (nb == 0) { memset(x, 0, sizeof(double) * m); free(r); free(z); free(p); free(q); *res_out = 0; return 0; }
    if (warm) {                            /* 4.35: start from the vector in x (the predictor's solution for the corrector) */
        matvec(S, x, q);
        for (int i = 0; i < m; i++) r[i] = h[i] - q[i];
        prec_apply(S, r, z); rz = dotn(m, r, z);
        res = sqrt(fabs(rz)) / nb;
        if (res >= 1.0) { memset(x, 0, sizeof(double) * m); memcpy(r, h, sizeof(double) * m); prec_apply(S, r, z); rz = dotn(m, r, z); res = 1.0; }
        else if (res < tol) { S->ncg += 0; free(r); free(z); free(p); free(q); *res_out = res; return 0; }
    } else memset(x, 0, sizeof(double) * m);
    memcpy(p, z, sizeof(double) * m);
    const double t0 = wtime();
    for (k = 1; k <= maxit; k++) {
        matvec(S, p, q);
        const double pq = dotn(m, p, q);
        if (!(pq > 0)) break;
        const double al = rz / pq;
        for (int i = 0; i < m; i++) { x[i] += al * p[i]; r[i] -= al * q[i]; }
        prec_apply(S, r, z);
        const double rz2 = dotn(m, r, z);
        res = sqrt(fabs(rz2)) / nb;
        if (res < tol) break;
        if (tmax > 0 && wtime() - t0 > tmax) break;
        const double be = rz2 / rz; rz = rz2;
        for (int i = 0; i < m; i++) p[i] = z[i] + be * p[i];
    }
    if (k > maxit) k = maxit;
    S->ncg += k;
    free(r); free(z); free(p); free(q);
    *res_out = res;
    return k;
}

/* ---- the method ------------------------------------------------------------------------- */
static double **blk_arrays(MF *S, int which) {
    double **a = mx(sizeof(double *) * S->nb);
    for (int k = 0; k < S->nb; k++) {
        MB *b = &S->b[k];
        a[k] = which == 0 ? b->X : which == 1 ? b->dX : which == 2 ? b->Rc : b->dXa;
    }
    return a;
}

int mfipm_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout) {
    const double t0 = wtime();
    memset(R, 0, sizeof(*R));
    const int m = P->m, nb = P->nblk, verbose = par->verbose;
    const double sc = P->bs * P->cs;
    MF S; memset(&S, 0, sizeof(S));
    mb_init(&S, P);
    S.Dg = mx(sizeof(double) * m); S.Dgi = mx(sizeof(double) * m);
    double tg = 0;
    S.G = fom_gram_build(P, verbose > 1, &tg);
    double *y = mz(sizeof(double) * m), *dy = mz(sizeof(double) * m), *dya = mz(sizeof(double) * m);
    double *Rp = mx(sizeof(double) * m), *h = mx(sizeof(double) * m), *tmp = mx(sizeof(double) * m), *tmp2 = mx(sizeof(double) * m);
    double **Xs = blk_arrays(&S, 0), **dXs = blk_arrays(&S, 1), **Ts = blk_arrays(&S, 2), **dXas = blk_arrays(&S, 3);
    /* starting point (SDPT3's rule on the scaled data) */
    double nC = 0, nbv = 0;
    for (int c = 0; c < m; c++) nbv += P->b[c] * P->b[c];
    nbv = sqrt(nbv);
    for (int k = 0; k < nb; k++) nC += dotn(blen(&S.b[k]), S.b[k].C, S.b[k].C);
    nC = sqrt(nC);
    for (int k = 0; k < nb; k++) {
        MB *b = &S.b[k]; const int n = b->n;
        double amax = 0; for (int t = 0; t < b->ncon; t++) { double s = 0; const SpSym *A = &b->B->A[t]; if (b->type == BLK_LP) { for (int q = 0; q < A->nnz; q++) s += A->val[q] * A->val[q]; } else for (int q = 0; q < A->ef; q++) s += A->fv[q] * A->fv[q]; amax = fmax(amax, sqrt(s)); }
        double xb = 0; for (int t = 0; t < b->ncon; t++) xb = fmax(xb, (1.0 + fabs(P->b[b->con[t]])) / (1.0 + amax));
        const double cn = sqrt(dotn(blen(b), b->C, b->C));
        const double xi = fmax(10.0, fmax(sqrt((double)n), n * xb)), zi = fmax(10.0, fmax(sqrt((double)n), fmax(amax, cn)));
        if (b->type == BLK_LP) for (int i = 0; i < n; i++) { b->X[i] = xi; b->Z[i] = zi; }
        else for (int i = 0; i < n; i++) { b->X[i + (size_t)i * n] = xi; b->Z[i + (size_t)i * n] = zi; }
    }
    if (verbose > 0) {
        size_t nnz = 0; double s3 = 0; for (int k = 0; k < nb; k++) { for (int t = 0; t < S.b[k].ncon; t++) nnz += P->blk[k].A[t].nnz; if (S.b[k].type == BLK_SDP) s3 += pow(S.b[k].n, 3); }
        printf("matrix-free IPM: m = %d, %d block(s), sum n^3 %.2e (M v %.2e flops), %zu nonzeros; A A' %s (%.2fs); rho %g, rmax %d, kmax %d\n",
               m, nb, s3, 4 * s3, nnz, S.G->dense ? "dense" : "sparse", tg, par->mf_rho, par->mf_rmax, par->mf_kmax);
        printf(" it   pobj            dobj            pinf     dinf     gap      mu       alp   ald   sigma   cg(pred,corr) k     kbulk    time\n");
    }
    double mu0 = -1, pobj = 0, dobj = 0, pinf = 1, dinf = 1, gap = 1;
    int it, status = ST_MAXIT, tot_cg = 0;
    const double tol = 0.5 * par->tol;   /* the file-data measure is typically 1.5x the scaled one */
    /* 4.35: graceful degradation. When CG cannot reach its tolerance within the budget the
     * directions are inexact and the iteration may stall or drift; the best iterate (by
     * max(gap, pinf, dinf)) is kept and returned, and the run stops after mf_stall iterations
     * without a 20% improvement once the CG budget has been hit. */
    double best_merit = 1e300, best_pobj = 0, best_dobj = 0, best_pinf = 0, best_dinf = 0, best_gap = 0;
    double *best_y = mx(sizeof(double) * m), **best_X = mx(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) best_X[k] = mx(sizeof(double) * blen(&S.b[k]));
    int nstall = 0, cg_capped = 0, best_it = 0;
    for (it = 0; it < par->maxit; it++) {
        /* residuals */
        op_A(&S, Xs, Rp);
        for (int c = 0; c < m; c++) Rp[c] = P->b[c] - Rp[c];
        double rdn = 0, xz = 0;
        pobj = 0; dobj = dotn(m, P->b, y);
        for (int k = 0; k < nb; k++) {
            MB *b = &S.b[k]; const size_t len = blen(b);
            op_At(b, y, b->Rd);
            for (size_t i = 0; i < len; i++) b->Rd[i] = b->C[i] - b->Rd[i] - b->Z[i];
            rdn += dotn(len, b->Rd, b->Rd); xz += dotn(len, b->X, b->Z); pobj += dotn(len, b->C, b->X);
        }
        const double mu = xz / S.N;
        if (mu0 < 0) mu0 = mu;
        pinf = sqrt(dotn(m, Rp, Rp)) / (1.0 + nbv); dinf = sqrt(rdn) / (1.0 + nC);
        gap = fabs(sc * (pobj - dobj)) / (1.0 + fabs(sc * pobj) + fabs(sc * dobj));   /* in the file's units (DIMACS 5) */
        {   const double merit = fmax(gap, fmax(pinf, dinf));
            if (merit < 0.8 * best_merit) nstall = 0; else if (cg_capped) nstall++;
            if (merit < best_merit) {
                best_merit = merit; best_it = it; best_pobj = pobj; best_dobj = dobj; best_pinf = pinf; best_dinf = dinf; best_gap = gap;
                memcpy(best_y, y, sizeof(double) * m);
                for (int k = 0; k < nb; k++) memcpy(best_X[k], S.b[k].X, sizeof(double) * blen(&S.b[k]));
            }
            if (merit < tol) { status = ST_OPTIMAL; break; }
            if (brisk_time_up(par)) { status = ST_TIME; break; }
            if (nstall >= par->mf_stall) {
                if (verbose > 0) printf("matrix-free IPM: no progress in %d iterations with the CG budget exhausted (best %.1e at iteration %d): stopping\n", nstall, best_merit, best_it);
                status = ST_MAXIT; break;
            }
        }
        /* NT scaling, W Rd W, preconditioner */
        double tn = wtime();
        for (int k = 0; k < nb; k++) {
            MB *b = &S.b[k];
            const int e = nt_block(b);
            if (e) { if (verbose >= 0) printf("matrix-free IPM: block %d: %s not positive definite\n", k + 1, e == 1 ? "X" : e == 2 ? "Z" : "L'ZL"); status = ST_NUMERIC; goto done; }
            wvw(b, b->Rd, b->E0, b->T2);
        }
        S.t_nt += wtime() - tn;
        double kk = 0; prec_setup(&S, par, &kk);
        double kbulk = 0; for (int k = 0; k < nb; k++) if (S.b[k].type == BLK_SDP) kbulk = fmax(kbulk, S.b[k].kbulk);
        const double cgtol = fmin(par->mf_cgtol_max, fmax(par->mf_cgtol_min, 1e-2 * mu / mu0));
        double tleft = par->timelimit > 0 ? par->timelimit - (wtime() - par->t_start) : 0;
        if (par->mf_cgtime > 0 && (tleft <= 0 || par->mf_cgtime < tleft)) tleft = par->mf_cgtime;   /* 4.35: a solve may not take longer than this */
        int cgk[2] = { 0, 0 }; double cgres[2] = { 0, 0 };
        double ap = 0, ad = 0, sigma = 0;
        for (int stage = 0; stage < 2; stage++) {
            /* Rc: predictor -X; corrector from Mehrotra's NT-scaled formula */
            for (int k = 0; k < nb; k++) {
                MB *b = &S.b[k]; const int n = b->n; const size_t len = blen(b);
                double *Rc = b->Rc;
                if (stage == 0) { for (size_t i = 0; i < len; i++) Rc[i] = -b->X[i]; continue; }
                double *Xt = b->dX, *Zt = b->dZ;    /* scratch: scaled predictor directions */
                scaled_dirs(b, b->dXa, b->dZa, Xt, Zt);
                const double smu = sigma * mu;
                if (b->type == BLK_LP) {
                    for (int i = 0; i < n; i++) { const double rhs = 2 * smu - 2 * b->dv[i] * b->dv[i] - 2 * Xt[i] * Zt[i]; Rc[i] = b->W[i] * rhs / (2 * b->dv[i]); }   /* G Zc G = W Zc (G^2 = W) */
                    continue;
                }
                /* rhs = 2 smu I - 2 dv^2 - (Xt Zt + Zt Xt) ; Zc = rhs ./ (dv_i + dv_j) ; Rc = G Zc G' */
                BL(dgemm_)("N", "N", &n, &n, &n, &DONE, Xt, &n, Zt, &n, &DZERO, b->T1, &n);
                for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
                    double v = -(b->T1[i + (size_t)j * n] + b->T1[j + (size_t)i * n]);
                    if (i == j) v += 2 * smu - 2 * b->dv[i] * b->dv[i];
                    b->T2[i + (size_t)j * n] = v / (b->dv[i] + b->dv[j]);
                }
                BL(dsymm_)("R", "L", &n, &n, &DONE, b->T2, &n, b->G, &n, &DZERO, b->T1, &n);   /* G Zc */
                BL(dgemm_)("N", "T", &n, &n, &n, &DONE, b->T1, &n, b->G, &n, &DZERO, Rc, &n);  /* (G Zc) G' */
                symmetrize(n, Rc);
            }
            /* h = Rp - A(Rc - W Rd W) */
            for (int k = 0; k < nb; k++) { MB *b = &S.b[k]; const size_t len = blen(b); for (size_t i = 0; i < len; i++) b->Rc[i] -= b->E0[i]; }
            op_A(&S, Ts, h);
            for (int c = 0; c < m; c++) h[c] = Rp[c] - h[c];
            for (int k = 0; k < nb; k++) { MB *b = &S.b[k]; const size_t len = blen(b); for (size_t i = 0; i < len; i++) b->Rc[i] += b->E0[i]; }
            double *dyv = stage == 0 ? dya : dy;
            if (stage == 1) memcpy(dy, dya, sizeof(double) * m);
            cgk[stage] = pcg(&S, h, dyv, cgtol, par->mf_cgmax, tleft > 0 ? tleft : 0, &cgres[stage], stage == 1 && par->mf_warm);
            if (cgres[stage] > cgtol) cg_capped = 1;
            tot_cg += cgk[stage];
            /* dZ = Rd - A*(dy); dX = Rc - W dZ W; exact projection */
            for (int k = 0; k < nb; k++) {
                MB *b = &S.b[k]; const size_t len = blen(b);
                double *dZ = stage == 0 ? b->dZa : b->dZ, *dX = stage == 0 ? b->dXa : b->dX;
                op_At(b, dyv, dZ);
                for (size_t i = 0; i < len; i++) dZ[i] = b->Rd[i] - dZ[i];
                wvw(b, dZ, dX, b->T2);
                for (size_t i = 0; i < len; i++) dX[i] = b->Rc[i] - dX[i];
                if (b->type == BLK_SDP) symmetrize(b->n, dX);
            }
            op_A(&S, stage == 0 ? dXas : dXs, tmp);
            for (int c = 0; c < m; c++) tmp[c] = Rp[c] - tmp[c];
            fom_gram_solve(S.G, tmp);
            for (int k = 0; k < nb; k++) {
                MB *b = &S.b[k]; const size_t len = blen(b);
                double *dX = stage == 0 ? b->dXa : b->dX;
                op_At(b, tmp, b->T1);
                for (size_t i = 0; i < len; i++) dX[i] += b->T1[i];
            }
            if (getenv("BRISK_MFDBG")) {
                /* checks: W Z W = X, A(dX) = Rp, dX + W dZ W = Rc */
                double e1 = 0, e2 = 0, e3 = 0, nx = 0;
                for (int k = 0; k < nb; k++) {
                    MB *b = &S.b[k]; const size_t len = blen(b);
                    double *dX = stage == 0 ? b->dXa : b->dX, *dZ = stage == 0 ? b->dZa : b->dZ;
                    double *o1 = mx(sizeof(double) * len), *o2 = mx(sizeof(double) * len);
                    wvw(b, b->Z, o1, o2); for (size_t i = 0; i < len; i++) { e1 = fmax(e1, fabs(o1[i] - b->X[i])); nx = fmax(nx, fabs(b->X[i])); }
                    wvw(b, dZ, o1, o2); for (size_t i = 0; i < len; i++) e3 = fmax(e3, fabs(dX[i] + o1[i] - b->Rc[i]));
                    free(o1); free(o2);
                }
                op_A(&S, stage == 0 ? dXas : dXs, tmp2); for (int c = 0; c < m; c++) e2 = fmax(e2, fabs(tmp2[c] - Rp[c]));
                printf("   [dbg stage %d: |WZW-X| %.2e (|X| %.2e), |A dX - Rp| %.2e, |dX + W dZ W - Rc| %.2e, cg res %.2e]\n", stage, e1, nx, e2, e3, cgres[stage]);
            }
            /* step lengths */
            ap = 1e300; ad = 1e300;
            for (int k = 0; k < nb; k++) {
                MB *b = &S.b[k];
                double *dX = stage == 0 ? b->dXa : b->dX, *dZ = stage == 0 ? b->dZa : b->dZ;
                double *Xt = b->T2, *Zt = b->T3;
                scaled_dirs(b, dX, dZ, Xt, Zt);
                ap = fmin(ap, max_step(b, Xt)); ad = fmin(ad, max_step(b, Zt));
            }
            if (stage == 0) {
                const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
                double xz2 = 0;
                for (int k = 0; k < nb; k++) {
                    MB *b = &S.b[k]; const size_t len = blen(b);
                    xz2 += dotn(len, b->X, b->Z) + a1 * dotn(len, b->dXa, b->Z) + a2 * dotn(len, b->X, b->dZa) + a1 * a2 * dotn(len, b->dXa, b->dZa);
                }
                const double muaff = xz2 / S.N;
                sigma = fmin(1.0, fmax(0.0, pow(muaff / mu, 3)));
                if (fmin(a1, a2) < 0.3) sigma = fmax(sigma, fmin(a1, a2) < 0.1 ? 0.5 : 0.2);     /* short affine step: recentre */
            }
        }
        const double gam = 0.9 + 0.09 * fmin(fmin(1.0, ap), fmin(1.0, ad));
        const double alp = fmin(1.0, gam * ap), ald = fmin(1.0, gam * ad);
        for (int k = 0; k < nb; k++) {
            MB *b = &S.b[k]; const size_t len = blen(b);
            for (size_t i = 0; i < len; i++) { b->X[i] += alp * b->dX[i]; b->Z[i] += ald * b->dZ[i]; }
            if (b->type == BLK_SDP) { symmetrize(b->n, b->X); symmetrize(b->n, b->Z); }
        }
        for (int c = 0; c < m; c++) y[c] += ald * dy[c];
        if (verbose > 0)
            printf("%3d %+.8e %+.8e %.2e %.2e %.2e %.2e %.3f %.3f %.3f %5d %5d %5d %.1e %8.1f\n",
                   it, sc * pobj, sc * dobj, pinf, dinf, gap, mu, alp, ald, sigma, cgk[0], cgk[1], S.k, kbulk, wtime() - t0);
        if (verbose > 0) fflush(stdout);
        if (alp < 1e-6 && ald < 1e-6) { status = ST_NUMERIC; it++; break; }
    }
done:
    if (status != ST_OPTIMAL && best_merit < fmax(gap, fmax(pinf, dinf))) {
        /* return the best iterate, not the last */
        if (verbose > 0) printf("matrix-free IPM: returning iteration %d (%.1e) rather than the last (%.1e)\n", best_it, best_merit, fmax(gap, fmax(pinf, dinf)));
        pobj = best_pobj; dobj = best_dobj; pinf = best_pinf; dinf = best_dinf; gap = best_gap;
        memcpy(y, best_y, sizeof(double) * m);
        for (int k = 0; k < nb; k++) memcpy(S.b[k].X, best_X[k], sizeof(double) * blen(&S.b[k]));
    }
    free(best_y); for (int k = 0; k < nb; k++) free(best_X[k]); free(best_X);
    if (status == ST_MAXIT && fmax(gap, fmax(pinf, dinf)) < par->red_acc) status = ST_REDUCED;
    R->status = status; R->iters = it;
    R->pobj = sc * pobj; R->dobj = sc * dobj; R->pinf = pinf; R->dinf = dinf; R->relgap = gap; R->relcomp = gap;
    R->direction = 1;
    R->t_total = wtime() - t0; R->t_chol = S.t_prec; R->t_dense = S.t_nt; R->t_schur = S.t_mv;
    if (verbose > 0)
        printf("matrix-free IPM: %d iterations, %ld CG steps (%.1f per solve), %ld products M v (%.1fs), preconditioner %.1fs, NT %.1fs, total %.1fs\n",
               it, S.ncg, it > 0 ? S.ncg / (2.0 * it) : 0.0, S.nmv, S.t_mv, S.t_prec, S.t_nt, R->t_total);
    if (yout) for (int i = 0; i < m; i++) yout[i] = P->cs * P->d[i] * y[i];
    if (Xout) for (int k = 0; k < nb; k++) { const size_t len = blen(&S.b[k]); for (size_t i = 0; i < len; i++) Xout[k][i] = P->bs * S.b[k].X[i]; }
    /* free */
    for (int k = 0; k < nb; k++) {
        MB *b = &S.b[k];
        free(b->C); free(b->X); free(b->Z); free(b->dX); free(b->dZ); free(b->dXa); free(b->dZa); free(b->Rd); free(b->E0); free(b->Rc);
        free(b->W); free(b->dg); free(b->ev); free(b->dv); free(b->L); free(b->G); free(b->Y); free(b->T1); free(b->T2); free(b->T3);
        free(b->Q); free(b->wt); free(b->colp); free(b->crow); free(b->ccon); free(b->cval); free(b->work); free(b->iwork);
        free(S.pos[k]);
    }
    prec_free_outl(&S);
    free(S.b); free(S.pos); free(S.Dg); free(S.Dgi); free(S.cap); free(S.lamp); free(S.ipiv); free(S.capwork); free(S.tk); free(S.tk2);
    fom_gram_free(S.G);
    free(y); free(dy); free(dya); free(Rp); free(h); free(tmp); free(tmp2); free(Xs); free(dXs); free(Ts); free(dXas);
    (void)tot_cg; (void)DMONE;
    return 0;
}
