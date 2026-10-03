/* fom.c - BRISK 4.31: first-order engine for problems too large for the interior-point method.
 *
 * Dual augmented-Lagrangian splitting (the ADMM of Wen-Goldfarb-Yin / the ADMM+ phase of
 * SDPNAL+) with the Halpern-anchored Peaceman-Rachford acceleration of Sun et al. (HPR):
 *
 *   min <C,X> s.t. A(X) = b, X in K          max b'y s.t. A*y + Z = C, Z in K
 *
 *   L_sigma(y, Z; X) = -b'y + <X, A*y + Z - C> + sigma/2 ||A*y + Z - C||^2
 *
 *   y      = (A A*)^-1 [ b/sigma - A(X/sigma + Z - C) ]            (A A* factored once)
 *   W      = X/sigma + A*y - C          (ADMM)      or      X/sigma + Z + 2 (A*y - C)  (PR)
 *   Z      = Pi_K(-W),  X+ = sigma Pi_K(W)                          (one eigendecomposition)
 *
 * Nothing of size m x m beyond the (sparse) Gram matrix A A* is formed; the memory is the
 * data plus a few n x n matrices per block. The result is the solution in the solver's
 * scaling, as dsdp_solve returns it; the pipeline's postsolve measures it on the file data.
 *
 * Free variables (split pairs) are handled as free: the projection is the identity on the
 * pair, which is exact because the pair's columns and costs are opposite.                */
#include "brisk.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const int IONE = 1;
static const double DONE = 1.0, DZERO = 0.0;
#define AA_MAX 32
static int g_fom_partial = 2;   /* 0 full eig, 1 + dsyevr ranges, 2 + Lanczos */
double wtime(void);

typedef struct Lz Lz;
typedef struct {
    int n, type;
    double *X, *Z, *W, *V, *ev, *work, *Xa, *Za, *T, *P;  /* T: A*y - C, P: Pi(W); Xa, Za: Halpern anchor */
    int lwork, liwork, *iwork;
    unsigned char *freev;                          /* LP: 1 = member of a split free pair */
    int npos, nneg;                                /* last eigen counts */
    double *Vp; int *isup; long n_partial, n_full, n_lanczos, n_lzfail;  /* partial eigendecomposition (dsyevr / Lanczos) */
    Lz *lz; int kwarm;
    double t_eig, t_core, t_syrk;   /* 4.39: the LAPACK call and the rank-k product within t_eig */
    float *Vf, *evf, *workf; int lworkf, liworkf, *iworkf; long n_single;   /* 4.37: single-precision projections */
    double *rrQ, *rrT, *rrH, *rrE; int rr_cap; long n_rr, n_rrfail; int rr_since;   /* 4.39: warm Rayleigh-Ritz projection */
    float *srZ, *srwork; int *srsup, *sriwork, srlw, srliw; long n_srange;          /* 4.39: single-precision range projection */
    /* 4.39: the decomposition split into tridiagonalization, divide and conquer on T and the
     * back-transformation of the small side's vectors only (xsyevd transforms all n) */
    float *dcd, *dce, *dctau, *dcw; int dclw, *dciw, dcliw; long n_dcpart;
    double *dcdd, *dcde, *dcdtau, *dcdw; int dcdlw, *dcdiw, dcdliw;
} FB;
static int g_fom_dcpart = 1;         /* 4.39: BRISK_FOMDCPART=0 restores xsyevd */
static int g_fom_single_now = 0;     /* 4.37: the projections of this iteration in single precision */
static int g_fom_rr = 0;             /* 4.39: warm-started block Rayleigh-Ritz projections (BRISK_FOMRR=1; off: the clustered spectra near 0 of the moment relaxations leave Ritz residuals ~1e-2, REPORT_4.39) */
static int g_fom_srange = 1;         /* 4.39: single-precision projections by ssyevr on the small side's value range */
static double g_rr_tol = 1e-6;       /* 4.39: their relative accuracy (follows the residual) */
static int g_rr_refresh = 40;        /* 4.39: a full decomposition at least every this many projections */
static FILE *g_ftrace = NULL;        /* 4.37: BRISK_FOMTRACE (both phases) */
static double g_split_rate = 0;      /* 4.37: the splitting's recent rate (log residual per second) at phase II entry */

static void *fo_malloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (first-order engine)\n"); exit(1); } return p; }
static size_t fbsz(const Block *B) { return B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n; }

/* F = sum_i y_i A_i (dense full / vector) */
static void fb_ATy(const Block *B, const double *y, double *F) {
    const int n = B->n;
    if (B->type == BLK_LP) {
        memset(F, 0, sizeof(double) * n);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t]; const double yi = y[B->con[t]];
            for (int q = 0; q < S->nnz; q++) F[S->row[q]] += yi * S->val[q];
        }
        return;
    }
    memset(F, 0, sizeof(double) * (size_t)n * n);
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *S = &B->A[t]; const double yi = y[B->con[t]];
        if (yi == 0) continue;
        for (int q = 0; q < S->ef; q++) F[S->fr[q] + (size_t)S->fc[q] * n] += yi * S->fv[q];
    }
}
/* out[con] += scale <A_con, D> (D symmetric, full) */
static void fb_Aop(const Block *B, const double *D, double *out, double scale) {
    const int n = B->n;
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *S = &B->A[t]; double v = 0;
        if (B->type == BLK_LP) for (int q = 0; q < S->nnz; q++) v += S->val[q] * D[S->row[q]];
        else for (int q = 0; q < S->nnz; q++) { const int p = S->row[q], c = S->col[q]; v += (p == c) ? S->val[q] * D[p + (size_t)p * n] : 2.0 * S->val[q] * D[p + (size_t)c * n]; }
        out[B->con[t]] += scale * v;
    }
}
static double fb_Cdot(const Block *B, const double *D) {
    const int n = B->n; const SpSym *S = &B->C; double v = 0;
    if (B->type == BLK_LP) for (int q = 0; q < S->nnz; q++) v += S->val[q] * D[S->row[q]];
    else for (int q = 0; q < S->nnz; q++) { const int p = S->row[q], c = S->col[q]; v += (p == c) ? S->val[q] * D[p + (size_t)p * n] : 2.0 * S->val[q] * D[p + (size_t)c * n]; }
    return v;
}
static void fb_addC(const Block *B, double a, double *D) {
    const int n = B->n; const SpSym *S = &B->C;
    if (B->type == BLK_LP) for (int q = 0; q < S->nnz; q++) D[S->row[q]] += a * S->val[q];
    else for (int q = 0; q < S->ef; q++) D[S->fr[q] + (size_t)S->fc[q] * n] += a * S->fv[q];
}

/* ---- 4.37: the constraint operators by position -------------------------------------
 * A* y and A(D) over all blocks at once, position-major: every stored position (i <= j) of every
 * block with the list of its (constraint, value) entries. A* y writes each position once
 * (streaming, both triangles) and gathers y; A(D) reads D once in storage order and scatters
 * into per-thread accumulators. The constraint-major loops (fb_ATy / fb_Aop) scattered into the
 * n x n arrays at random and ran on one thread: at Example 8.1.3 d = 5 (116 k rows, blocks up
 * to 1 051) three of them cost 0.5 s an iteration, more than the eigendecompositions. */
typedef struct {
    int nb; size_t np;
    int *pb, *pa, *pt;        /* block, dense offset i + j n, transposed offset j + i n (-1: diagonal / LP) */
    size_t *pp; int *pc; double *pv;
    int nth; double *acc, *acc2;   /* nth x m accumulators (two: fop_A2) */
    int m;
} FOp;
static FOp *g_fop = NULL;
static FOp *fop_build(const Problem *P) {
    FOp *O = calloc(1, sizeof(FOp)); O->nb = P->nblk; O->m = P->m;
    size_t np = 0, ne = 0;
    int **cnt = calloc(P->nblk, sizeof(int *));
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; const int n = B->n, lp = B->type == BLK_LP;
        const size_t len = lp ? (size_t)n : (size_t)n * n;
        cnt[k] = calloc(len + 1, sizeof(int));
        for (int t = 0; t < B->ncon; t++) { const SpSym *S = &B->A[t]; for (int q = 0; q < S->nnz; q++) { const int r = S->row[q], c = lp ? r : S->col[q]; const int lo = r < c ? r : c, hi = r < c ? c : r; cnt[k][lo + (size_t)hi * (lp ? 0 : n)]++; ne++; } }
        for (size_t p = 0; p < len; p++) if (cnt[k][p]) np++;
    }
    O->np = np; O->pb = fo_malloc(sizeof(int) * (np + 1)); O->pa = fo_malloc(sizeof(int) * (np + 1)); O->pt = fo_malloc(sizeof(int) * (np + 1));
    O->pp = fo_malloc(sizeof(size_t) * (np + 2)); O->pc = fo_malloc(sizeof(int) * (ne + 1)); O->pv = fo_malloc(sizeof(double) * (ne + 1));
    size_t p = 0, e = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; const int n = B->n, lp = B->type == BLK_LP;
        const size_t len = lp ? (size_t)n : (size_t)n * n;
        /* positions in storage order (column-major upper triangle); cnt becomes the fill pointer */
        int *pos = fo_malloc(sizeof(int) * (len + 1));
        for (size_t q = 0; q < len; q++) {
            pos[q] = -1;
            if (!cnt[k][q]) continue;
            const int i = lp ? (int)q : (int)(q % n), j = lp ? (int)q : (int)(q / n);
            O->pb[p] = k; O->pa[p] = lp ? i : i + j * n; O->pt[p] = (lp || i == j) ? -1 : j + i * n;
            O->pp[p] = e; e += cnt[k][q]; pos[q] = (int)p; p++;
        }
        size_t *fill = fo_malloc(sizeof(size_t) * (p + 1));
        for (size_t q = 0; q < len; q++) if (pos[q] >= 0) fill[pos[q]] = O->pp[pos[q]];
        for (int t = 0; t < B->ncon; t++) { const SpSym *S = &B->A[t]; for (int q = 0; q < S->nnz; q++) { const int r = S->row[q], c = lp ? r : S->col[q]; const int lo = r < c ? r : c, hi = r < c ? c : r;
            const int pp = pos[lo + (size_t)hi * (lp ? 0 : n)]; const size_t w = fill[pp]++; O->pc[w] = B->con[t]; O->pv[w] = S->val[q]; } }
        free(fill); free(pos); free(cnt[k]);
    }
    O->pp[np] = e;
    free(cnt);
#ifdef _OPENMP
    O->nth = omp_get_max_threads();
#else
    O->nth = 1;
#endif
    if (O->nth < 1) O->nth = 1;
    O->acc = O->nth > 1 ? fo_malloc(sizeof(double) * (size_t)O->nth * (O->m + 1)) : NULL;
    O->acc2 = O->nth > 1 ? fo_malloc(sizeof(double) * (size_t)O->nth * (O->m + 1)) : NULL;
    return O;
}
static void fop_free(FOp *O) { if (!O) return; free(O->pb); free(O->pa); free(O->pt); free(O->pp); free(O->pc); free(O->pv); free(O->acc); free(O->acc2); free(O); }
/* F_k = sum_i y_i A_i for every block (F[k] full n x n, or n for LP) */
static void fop_At(const FOp *O, const Problem *P, const double *y, double **F) {
    for (int k = 0; k < O->nb; k++) memset(F[k], 0, sizeof(double) * (P->blk[k].type == BLK_LP ? (size_t)P->blk[k].n : (size_t)P->blk[k].n * P->blk[k].n));
    const long np = (long)O->np;
    #pragma omp parallel for schedule(static) if (np > 20000)
    for (long p = 0; p < np; p++) {
        double s = 0; for (size_t q = O->pp[p]; q < O->pp[p + 1]; q++) s += y[O->pc[q]] * O->pv[q];
        double *Fk = F[O->pb[p]]; Fk[O->pa[p]] = s; if (O->pt[p] >= 0) Fk[O->pt[p]] = s;
    }
}
/* out[c] += scale <A_c, D> (D symmetric, full); with D2: out2[c] += scale2 <A_c, D2> in the same pass */
static void fop_A2(const FOp *O, double **D, double *out, double scale, double **D2, double *out2, double scale2) {
    const long np = (long)O->np; const int m = O->m;
    if (O->nth <= 1 || np <= 20000) {
        for (long p = 0; p < np; p++) {
            const double w = O->pt[p] >= 0 ? 2.0 : 1.0;
            const double v = scale * w * D[O->pb[p]][O->pa[p]], v2 = D2 ? scale2 * w * D2[O->pb[p]][O->pa[p]] : 0.0;
            for (size_t q = O->pp[p]; q < O->pp[p + 1]; q++) { out[O->pc[q]] += O->pv[q] * v; if (D2) out2[O->pc[q]] += O->pv[q] * v2; }
        }
        return;
    }
    /* 4.39: both matrices in one pass over the operator (two accumulators per thread) */
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const int tid = 0, nt = 1;
#endif
        double *a = O->acc + (size_t)tid * (m + 1), *a2 = D2 ? O->acc2 + (size_t)tid * (m + 1) : NULL;
        memset(a, 0, sizeof(double) * m); if (a2) memset(a2, 0, sizeof(double) * m);
        if (!D2) {
            #pragma omp for schedule(static)
            for (long p = 0; p < np; p++) {
                const double v = (O->pt[p] >= 0 ? 2.0 : 1.0) * D[O->pb[p]][O->pa[p]];
                if (v == 0) continue;
                for (size_t q = O->pp[p]; q < O->pp[p + 1]; q++) a[O->pc[q]] += O->pv[q] * v;
            }
        } else {
            #pragma omp for schedule(static)
            for (long p = 0; p < np; p++) {
                const double w = O->pt[p] >= 0 ? 2.0 : 1.0, v = w * D[O->pb[p]][O->pa[p]], v2 = w * D2[O->pb[p]][O->pa[p]];
                for (size_t q = O->pp[p]; q < O->pp[p + 1]; q++) { const int c = O->pc[q]; a[c] += O->pv[q] * v; a2[c] += O->pv[q] * v2; }
            }
        }
        #pragma omp for schedule(static)
        for (int c = 0; c < m; c++) {
            double s = 0; for (int t = 0; t < nt; t++) s += O->acc[(size_t)t * (m + 1) + c]; out[c] += scale * s;
            if (D2) { double s2 = 0; for (int t = 0; t < nt; t++) s2 += O->acc2[(size_t)t * (m + 1) + c]; out2[c] += scale2 * s2; }
        }
    }
}
static void fop_A(const FOp *O, double **D, double *out, double scale) { fop_A2(O, D, out, scale, NULL, NULL, 0.0); }

/* ---- Gram matrix A A* : sparse (BRISK's supernodal Cholesky) or dense --------------- */

typedef struct { int64_t key; int con; double val; } Ent;
static int cmp_ent(const void *a, const void *b) {
    const Ent *x = a, *y = b;
    return x->key < y->key ? -1 : x->key > y->key ? 1 : x->con - y->con;
}
typedef struct { int i, j; double v; } Pair;
static int cmp_pair(const void *a, const void *b) {
    const Pair *x = a, *y = b;
    return x->i != y->i ? x->i - y->i : x->j - y->j;
}

/* 4.37: Dw (per block, full n x n / n for LP) weights the entries: A diag(Dw) A* + shift I (the
 * phase II preconditioner from the diagonal of the projection's Jacobian); NULL: A A* */
static Gram *gram_build(const Problem *P, double *const *Dw, double shift, int verbose, double *t_gram);
Gram *fom_gram_build(const Problem *P, int verbose, double *t_gram) { return gram_build(P, NULL, 0.0, verbose, t_gram); }
static Gram *gram_build(const Problem *P, double *const *Dw, double shift, int verbose, double *t_gram) {
    const double t0 = wtime();
    const int m = P->m;
    Gram *G = calloc(1, sizeof(Gram)); G->m = m;
    /* pairs (i < j) with weights, diagonal separately */
    double *diag = calloc(m, sizeof(double));
    size_t np = 0, cp = 1 << 16;
    Pair *pr = fo_malloc(sizeof(Pair) * cp);
    /* 4.42: when many constraints share positions, the list of pairs with multiplicity is far
     * larger than the matrix (nn_n10_m50_K, m = 3 485: about 1e8 pairs, 1.5 GB and 84 s, almost
     * all of it sorting them). Then the pairs are summed in a dense m x m array instead. The
     * count is that of the loop below. */
    double *acc = NULL;
    {
        double tot = 0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            const size_t npos = B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
            int *cnt = calloc(npos ? npos : 1, sizeof(int));
            if (!cnt) { tot = 0; break; }
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *S = &B->A[t];
                for (int q = 0; q < S->nnz; q++) {
                    const size_t id = B->type == BLK_LP ? (size_t)S->row[q] : (size_t)S->row[q] * B->n + S->col[q];
                    if (id < npos) tot += cnt[id]++;
                }
            }
            free(cnt);
        }
        if (tot > 1e6 && 16.0 * tot > 8.0 * (double)m * m && 8.0 * (double)m * m <= 0.25 * brisk_mem_limit())
            acc = calloc((size_t)m * m, sizeof(double));
    }
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        const int lp = B->type == BLK_LP;
        size_t ne = 0;
        for (int t = 0; t < B->ncon; t++) ne += B->A[t].nnz;
        Ent *E = fo_malloc(sizeof(Ent) * (ne ? ne : 1));
        size_t q0 = 0;
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            for (int q = 0; q < S->nnz; q++) {
                const int r = S->row[q], c = lp ? r : S->col[q];
                E[q0].key = (int64_t)r * B->n + c; E[q0].con = B->con[t]; E[q0].val = S->val[q]; q0++;
            }
        }
        qsort(E, ne, sizeof(Ent), cmp_ent);
        for (size_t a = 0; a < ne;) {
            size_t b = a; while (b < ne && E[b].key == E[a].key) b++;
            const int r = (int)(E[a].key / B->n), c = (int)(E[a].key % B->n);
            const double w = ((lp || r == c) ? 1.0 : 2.0) * (Dw ? (lp ? Dw[k][r] : Dw[k][r + (size_t)c * B->n]) : 1.0);
            for (size_t p = a; p < b; p++) {
                diag[E[p].con] += w * E[p].val * E[p].val;
                if (acc) {     /* E is sorted by constraint within a position: con[p] < con[q] */
                    double *col = acc + (size_t)E[p].con * m; const double wp = w * E[p].val;
                    for (size_t q = p + 1; q < b; q++) col[E[q].con] += wp * E[q].val;
                    continue;
                }
                for (size_t q = p + 1; q < b; q++) {
                    if (np == cp) { cp *= 2; pr = realloc(pr, sizeof(Pair) * cp); if (!pr) { fprintf(stderr, "brisk: out of memory (Gram matrix)\n"); exit(1); } }
                    pr[np].i = E[p].con; pr[np].j = E[q].con; pr[np].v = w * E[p].val * E[q].val; np++;
                }
            }
            a = b;
        }
        free(E);
    }
    if (acc) {         /* the list of distinct pairs (i < j), in the order the sort below gives */
        size_t nz = 0;
        for (int i = 0; i < m; i++) { const double *col = acc + (size_t)i * m; for (int j = i + 1; j < m; j++) nz += col[j] != 0.0; }
        free(pr); pr = fo_malloc(sizeof(Pair) * (nz ? nz : 1)); np = 0;
        for (int i = 0; i < m; i++) { const double *col = acc + (size_t)i * m; for (int j = i + 1; j < m; j++) if (col[j] != 0.0) { pr[np].i = i; pr[np].j = j; pr[np].v = col[j]; np++; } }
        free(acc); acc = NULL;
    } else qsort(pr, np, sizeof(Pair), cmp_pair);
    size_t nu = 0;
    for (size_t p = 0; p < np; p++) {
        if (nu && pr[nu - 1].i == pr[p].i && pr[nu - 1].j == pr[p].j) pr[nu - 1].v += pr[p].v;
        else pr[nu++] = pr[p];
    }
    np = nu;
    const double dens = m ? (2.0 * np + m) / ((double)m * m) : 1.0;
    G->dense = (m <= 2500) || dens > 0.25 || getenv("BRISK_FOMDENSE");
    /* regularization: the rows may be dependent (a tiny multiple of the diagonal) */
    double dmax = 0; for (int i = 0; i < m; i++) dmax = fmax(dmax, diag[i]);
    G->reg = 1e-12 * (dmax > 0 ? dmax : 1.0) + shift;
    if (G->dense) {
        G->M = fo_malloc(sizeof(double) * (size_t)m * m);
        memset(G->M, 0, sizeof(double) * (size_t)m * m);
        for (int i = 0; i < m; i++) G->M[i + (size_t)i * m] = diag[i] + G->reg;
        for (size_t p = 0; p < np; p++) G->M[pr[p].j + (size_t)pr[p].i * m] = pr[p].v;   /* lower triangle (j > i) */
        int info = 0;
        BL(dpotrf_)("L", &m, G->M, &m, &info);
        for (double rg = 1e-8; info != 0 && rg <= 1e-2; rg *= 100) {
            /* dependent rows: stronger regularization (the y-step is then a proximal one; the
             * bias is rg * |y| on the primal residual, reported through the measures) */
            for (int i = 0; i < m; i++) G->M[i + (size_t)i * m] = diag[i] + rg * dmax + shift;
            for (int j = 0; j < m; j++) for (int i = j + 1; i < m; i++) G->M[i + (size_t)j * m] = 0;
            for (size_t p = 0; p < np; p++) G->M[pr[p].j + (size_t)pr[p].i * m] = pr[p].v;
            BL(dpotrf_)("L", &m, G->M, &m, &info);
            if (info == 0) { G->reg = rg * dmax + shift; if (verbose && !Dw) printf("first-order: dependent rows: Gram matrix regularized by %.0e\n", rg); }
        }
        if (info != 0 && Dw) { free(G->M); free(G); free(pr); free(diag); return NULL; }
        if (info != 0) { double dmin = 1e300; int nz = 0; for (int i = 0; i < m; i++) { if (diag[i] < dmin) dmin = diag[i]; if (diag[i] == 0) nz++; }
            fprintf(stderr, "brisk: first-order engine: the Gram matrix A A' is not positive definite (dpotrf info %d, diag min %.3e max %.3e, %d zero rows, %zu pairs)\n", info, dmin, dmax, nz, np); exit(1); }
    } else {
        int *deg = calloc(m, sizeof(int));
        for (size_t p = 0; p < np; p++) { deg[pr[p].i]++; deg[pr[p].j]++; }
        int **nbr = fo_malloc(sizeof(int *) * m); int *pool = fo_malloc(sizeof(int) * (2 * np + 1)); int *cnt = calloc(m, sizeof(int));
        size_t off = 0; for (int i = 0; i < m; i++) { nbr[i] = pool + off; off += deg[i]; }
        for (size_t p = 0; p < np; p++) { nbr[pr[p].i][cnt[pr[p].i]++] = pr[p].j; nbr[pr[p].j][cnt[pr[p].j]++] = pr[p].i; }
        G->S = schol_analyze_adj(m, deg, nbr, (size_t)4e9);
        free(nbr); free(pool); free(cnt); free(deg);
        if (!G->S) { fprintf(stderr, "brisk: first-order engine: the Gram matrix A A' is too dense to factor\n"); exit(1); }
        schol_zero(G->S);
        for (int i = 0; i < m; i++) schol_add(G->S, i, i, diag[i] + G->reg);
        for (size_t p = 0; p < np; p++) schol_add(G->S, pr[p].i, pr[p].j, pr[p].v);   /* one slot per unordered pair */
        int ok = schol_factor(G->S, 0.0) == 0;
        for (double rg = 1e-8; !ok && rg <= 1e-2; rg *= 100) {
            schol_zero(G->S);
            for (int i = 0; i < m; i++) schol_add(G->S, i, i, diag[i] + rg * dmax + shift);
            for (size_t p = 0; p < np; p++) schol_add(G->S, pr[p].i, pr[p].j, pr[p].v);   /* one slot per unordered pair */
            ok = schol_factor(G->S, 0.0) == 0;
            if (ok) { G->reg = rg * dmax + shift; if (verbose && !Dw) printf("first-order: dependent rows: Gram matrix regularized by %.0e\n", rg); }
        }
        if (!ok && Dw) { schol_free(G->S); free(G); free(pr); free(diag); return NULL; }
        if (!ok) { fprintf(stderr, "brisk: first-order engine: the Gram matrix A A' is not positive definite\n"); exit(1); }
        G->work = fo_malloc(sizeof(double) * (size_t)(m + 1) * 2);
    }
    *t_gram = wtime() - t0;
    if (verbose && !Dw)
        printf("first-order: Gram matrix A A' %d x %d, %zu off-diagonal pairs (%.2f%% dense), %s factor%s, %.2fs\n",
               m, m, np, 100.0 * dens, G->dense ? "dense" : "sparse", G->S ? "" : "", *t_gram);
    free(pr); free(diag);
    return G;
}
void fom_gram_solve(const Gram *G, double *x) {
    if (G->dense) { int info = 0; BL(dpotrs_)("L", &G->m, &IONE, G->M, &G->m, x, &G->m, &info); }
    else schol_solve(G->S, x, G->work);
}
void fom_gram_free(Gram *G) { if (!G) return; free(G->M); if (G->S) schol_free(G->S); free(G->work); free(G); }


/* ---- Lanczos (full reorthogonalization, thick restart) for the positive (or negative)
 * part of W when that side is small: the top eigenpairs down to the first non-positive
 * one, warm-started from the previous projection's eigenvectors.
 * Returns the number of eigenpairs of the wanted side (in ev ascending order semantics of
 * the caller: values in lam[0..k), vectors in Vout n x k), or -1 when not converged within
 * the basis budget. sign = +1: largest eigenvalues (positive part); -1: smallest (W is
 * used as -W).                                                                            */
struct Lz { int mdim; double *Q, *Tm, *Y, *th, *wk, *r; int lwk; };
static void lz_alloc(Lz *L, int n, int mdim) {
    L->mdim = mdim;
    L->Q = fo_malloc(sizeof(double) * (size_t)n * (mdim + 1)); L->Tm = fo_malloc(sizeof(double) * (size_t)mdim * mdim);
    L->Y = fo_malloc(sizeof(double) * (size_t)mdim * mdim); L->th = fo_malloc(sizeof(double) * mdim);
    L->r = fo_malloc(sizeof(double) * n); L->lwk = 8 * mdim + 64; L->wk = fo_malloc(sizeof(double) * L->lwk);
}
static void lz_free(Lz *L) { free(L->Q); free(L->Tm); free(L->Y); free(L->th); free(L->wk); free(L->r); }
static long g_lz_prod = 0;
static int lanczos_side(int n, const double *W, double sign, int kguess, const double *Vwarm, int kwarm,
                        Lz *L, double *lam, double *Vout, int maxk) {
    const int mdim = L->mdim;
    double *Q = L->Q, *Tm = L->Tm;
    /* start vector: the warm eigenvectors summed, plus a little noise (a new direction must
     * not be invisible), or a fixed pseudo-random vector */
    unsigned s = 12345u + (unsigned)n;
    for (int i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; Q[i] = ((s >> 9) & 0xFFFF) / 65536.0 - 0.5; }
    if (kwarm > 0) {
        double nq = 0; for (int i = 0; i < n; i++) nq += Q[i] * Q[i]; nq = sqrt(nq);
        for (int i = 0; i < n; i++) Q[i] *= 1e-3 / nq;
        for (int j = 0; j < kwarm; j++) { const double *v = Vwarm + (size_t)j * n; double nv = 0; for (int i = 0; i < n; i++) nv += v[i] * v[i]; nv = sqrt(nv); if (nv > 0) for (int i = 0; i < n; i++) Q[i] += v[i] / nv; }
    }
    { double nq = 0; for (int i = 0; i < n; i++) nq += Q[i] * Q[i]; nq = sqrt(nq); if (nq == 0) return -1; for (int i = 0; i < n; i++) Q[i] /= nq; }
    int j0 = 0;              /* kept (thick) vectors */
    double beta = 0;
    memset(Tm, 0, sizeof(double) * (size_t)mdim * mdim);
    for (int restart = 0; restart < 8; restart++) {
        /* Lanczos steps j0 .. mdim-1 with full reorthogonalization: Tm = Q' W Q */
        int mlast = mdim;
        for (int i = j0; i < mdim; i++) {
            double *qi = Q + (size_t)i * n, *r = L->r;
            BL(dsymv_)("L", &n, &sign, W, &n, qi, &IONE, &DZERO, r, &IONE); g_lz_prod++;
            /* the thick part: W q_j for kept j has a component beta * Y_last along q_{j0}; the
             * coefficients are already in Tm (arrowhead); here only the new column i */
            for (int pass = 0; pass < 2; pass++)
                for (int l = 0; l <= i; l++) {
                    const double *ql = Q + (size_t)l * n;
                    double d = 0; for (int p = 0; p < n; p++) d += ql[p] * r[p];
                    if (pass == 0) Tm[l + (size_t)i * mdim] = d; else Tm[l + (size_t)i * mdim] += d;
                    for (int p = 0; p < n; p++) r[p] -= d * ql[p];
                }
            double nr = 0; for (int p = 0; p < n; p++) nr += r[p] * r[p]; nr = sqrt(nr);
            beta = nr;
            if (i + 1 < mdim) { Tm[i + 1 + (size_t)i * mdim] = nr; }
            if (nr < 1e-14) { mlast = i + 1; beta = 0; break; }
            double *qn = Q + (size_t)(i + 1) * n;
            for (int p = 0; p < n; p++) qn[p] = r[p] / nr;
        }
        /* symmetric small eigenproblem on the leading mlast x mlast part */
        for (int c = 0; c < mlast; c++) for (int l = 0; l < c; l++) Tm[c + (size_t)l * mdim] = Tm[l + (size_t)c * mdim];
        for (int c = 0; c < mlast; c++) for (int l = 0; l < mlast; l++) L->Y[l + (size_t)c * mlast] = Tm[l + (size_t)c * mdim];
        int info = 0;
        BL(dsyev_)("V", "L", &mlast, L->Y, &mlast, L->th, L->wk, &L->lwk, &info);
        if (info != 0) return -1;
        /* Ritz pairs descending: converged while residual beta |Y[mlast-1, j]| <= tol */
        const double tol = 1e-9 * fmax(fabs(L->th[mlast - 1]), 1e-300);
        int kconv = 0, done = 0;
        for (int j = mlast - 1; j >= 0; j--) {
            const double res = beta * fabs(L->Y[mlast - 1 + (size_t)j * mlast]);
            if (res > tol) break;
            kconv++;
            if (L->th[j] <= 0) { done = 1; break; }
            if (kconv > maxk) break;
        }
        if (done || (mlast < mdim && kconv == mlast)) {
            /* all pairs with theta > 0 among the converged top ones (the last converged is <= 0 or the space is exhausted) */
            int k = 0;
            for (int j = mlast - 1; j >= 0 && k < kconv; j--) {
                if (L->th[j] <= 0) break;
                lam[k] = sign * L->th[j];
                double *v = Vout + (size_t)k * n; memset(v, 0, sizeof(double) * n);
                for (int l = 0; l < mlast; l++) { const double y = L->Y[l + (size_t)j * mlast]; if (y != 0) { const double *ql = Q + (size_t)l * n; for (int p = 0; p < n; p++) v[p] += y * ql[p]; } }
                k++;
            }
            return k;
        }
        if (mlast < mdim) return -1;             /* invariant subspace found but not converged: give up */
        /* thick restart: keep the top kk Ritz vectors, arrowhead coupling to q_m */
        int kk = kguess + 8; if (kk > mdim - 8) kk = mdim - 8; if (kk < 1) kk = 1;
        double *Qn = L->Q;                        /* rotate in place using Y: new q_l = Q Y[:, top l] */
        /* compute into the upper part of Q (columns mdim.. would be needed); use r as scratch per column via a temporary block */
        double *tmp = fo_malloc(sizeof(double) * (size_t)n * kk);
        for (int l = 0; l < kk; l++) {
            const int j = mlast - 1 - l; double *v = tmp + (size_t)l * n; memset(v, 0, sizeof(double) * n);
            for (int q = 0; q < mlast; q++) { const double y = L->Y[q + (size_t)j * mlast]; if (y != 0) { const double *ql = Q + (size_t)q * n; for (int p = 0; p < n; p++) v[p] += y * ql[p]; } }
        }
        double *qm = Q + (size_t)mdim * n;        /* the last residual vector q_{mdim} */
        memcpy(Q + (size_t)kk * n, qm, sizeof(double) * n);
        memcpy(Qn, tmp, sizeof(double) * (size_t)n * kk);
        free(tmp);
        memset(Tm, 0, sizeof(double) * (size_t)mdim * mdim);
        for (int l = 0; l < kk; l++) { const int j = mlast - 1 - l; Tm[l + (size_t)l * mdim] = L->th[j]; Tm[kk + (size_t)l * mdim] = beta * L->Y[mlast - 1 + (size_t)j * mlast]; Tm[l + (size_t)kk * mdim] = Tm[kk + (size_t)l * mdim]; }
        j0 = kk;
    }
    return -1;
}

/* ---- 4.39: warm-started block Rayleigh-Ritz projection --------------------------------
 * W changes little from one splitting step to the next, so the invariant subspace of its
 * small side (rank k, here ~n/10 on the moment relaxations) is nearly that of the previous
 * step. With V the previous small-side eigenvectors plus a margin (orthonormal, n x k0):
 *     Q = orth([V, S V]),  S = +-W (the small side positive),  H = Q' S Q,
 * the Ritz pairs of H with theta > 0 give the small side; accepted when every Ritz residual
 * |S x - theta x| is below tol * theta_max, when there is room left in the basis, and when
 * a few Lanczos steps on (I - X X') S (I - X X') find no positive eigenvalue above tol (a
 * missed direction). Cost: two BLAS-3 products n^2 k0 and the small eigenproblem, against
 * ~4 n^3 for a full decomposition. Up to two refinement passes (the next V the top Ritz
 * vectors); otherwise the caller does the full decomposition, which refreshes V. */
static int rr_orth(int n, int k, double *Q, double *G) {
    /* Cholesky QR, twice: Q <- Q R^-1 */
    for (int pass = 0; pass < 2; pass++) {
        const double one = 1.0, zero = 0.0; int info;
        BL(dsyrk_)("U", "T", &k, &n, &one, Q, &n, &zero, G, &k);
        BL(dpotrf_)("U", &k, G, &k, &info);
        if (info != 0) return 0;
        BL(dtrsm_)("R", "U", "N", "N", &n, &k, &one, G, &k, Q, &n);
    }
    return 1;
}
static int rr_project(FB *f, const double *W, int small_pos, double *T) {
    const int n = f->n;
    int k0 = f->kwarm;
    if (k0 < 1 || !f->Vp) return 0;
    const int cap = n / 2;
    if (2 * k0 > cap) return 0;
    if (!f->rrQ || f->rr_cap < cap) {
        free(f->rrQ); free(f->rrT); free(f->rrH); free(f->rrE);
        f->rrQ = fo_malloc(sizeof(double) * (size_t)n * cap); f->rrT = fo_malloc(sizeof(double) * (size_t)n * cap);
        f->rrH = fo_malloc(sizeof(double) * (size_t)cap * cap); f->rrE = fo_malloc(sizeof(double) * (cap + 1));
        f->rr_cap = cap;
    }
    const double s = small_pos ? 1.0 : -1.0, zero = 0.0, one = 1.0;
    double *Q = f->rrQ, *Tq = f->rrT, *H = f->rrH, *th = f->rrE;
    for (int pass = 0; pass < 3; pass++) {
        const int nbq = 2 * k0;
        memcpy(Q, f->Vp, sizeof(double) * (size_t)n * k0);
        BL(dsymm_)("L", "L", &n, &k0, &s, W, &n, f->Vp, &n, &zero, Q + (size_t)n * k0, &n);
        if (!rr_orth(n, nbq, Q, H)) return 0;
        BL(dsymm_)("L", "L", &n, &nbq, &s, W, &n, Q, &n, &zero, Tq, &n);
        BL(dgemm_)("T", "N", &nbq, &nbq, &n, &one, Q, &n, Tq, &n, &zero, H, &nbq);
        for (int j = 0; j < nbq; j++) for (int i = 0; i < j; i++) { const double a = 0.5 * (H[i + (size_t)j * nbq] + H[j + (size_t)i * nbq]); H[i + (size_t)j * nbq] = H[j + (size_t)i * nbq] = a; }
        int lw = -1, liw = -1, iwq = 0, info = 0; double wq = 0;
        BL(dsyevd_)("V", "L", &nbq, H, &nbq, th, &wq, &lw, &iwq, &liw, &info);
        lw = (int)wq + 1; liw = iwq + 1;
        double *wk = fo_malloc(sizeof(double) * lw); int *iw = fo_malloc(sizeof(int) * liw);
        BL(dsyevd_)("V", "L", &nbq, H, &nbq, th, wk, &lw, iw, &liw, &info);
        free(wk); free(iw);
        if (info != 0) return 0;
        int kp = 0; for (int j = 0; j < nbq; j++) if (th[j] > 0) kp++;
        const double tmax = fmax(fabs(th[nbq - 1]), fabs(th[0]));
        const double thr = g_rr_tol * fmax(tmax, 1e-300);
        const int margin = kp / 8 + 8;
        if (kp + margin > nbq - 2) { if (ENV_ON("BRISK_RRDBG")) printf("   [rr n %d k0 %d: no room (kp %d)]\n", n, k0, kp); return 0; }   /* no room */
        /* Ritz vectors and residuals of the positive pairs (the top kp columns of H) */
        const int c0 = nbq - kp;
        double *X = f->V, *R = T;                          /* scratch: n x kp each (T is n x n) */
        if (kp > 0) {
            BL(dgemm_)("N", "N", &n, &kp, &nbq, &one, Q, &n, H + (size_t)c0 * nbq, &nbq, &zero, X, &n);
            BL(dgemm_)("N", "N", &n, &kp, &nbq, &one, Tq, &n, H + (size_t)c0 * nbq, &nbq, &zero, R, &n);
        }
        double rmax = 0;
        for (int j = 0; j < kp; j++) {
            const double tj = th[c0 + j]; double r2 = 0;
            const double *x = X + (size_t)j * n, *r = R + (size_t)j * n;
            for (int i = 0; i < n; i++) { const double d = r[i] - tj * x[i]; r2 += d * d; }
            if (r2 > rmax) rmax = r2;
        }
        rmax = sqrt(rmax);
        /* the next warm vectors: the top kp + margin Ritz vectors */
        int kn = kp + margin; if (kn > nbq) kn = nbq; if (2 * kn > cap) kn = cap / 2;
        const int cn = nbq - kn;
        if (ENV_ON("BRISK_RRDBG")) printf("   [rr n %d k0 %d pass %d: kp %d rmax %.2e thr %.2e tmax %.2e]\n", n, k0, pass, kp, rmax, thr, tmax);
        if (rmax > thr) {
            if (kn < 1) return 0;
            BL(dgemm_)("N", "N", &n, &kn, &nbq, &one, Q, &n, H + (size_t)cn * nbq, &nbq, &zero, f->Vp, &n);
            k0 = kn; f->kwarm = kn;
            continue;                                      /* refine */
        }
        /* a missed positive direction: Lanczos steps on P S P, P = I - X X' (X orthonormal) */
        {
            double *v = R, *wv = R + n, *vold = R + 2 * (size_t)n;
            unsigned sd = 2463534242u + (unsigned)n;
            for (int i = 0; i < n; i++) { sd ^= sd << 13; sd ^= sd >> 17; sd ^= sd << 5; v[i] = (double)(sd & 0xFFFFFF) / 16777216.0 - 0.5; }
            double *cf = fo_malloc(sizeof(double) * (kp + 1));
            #define RR_DEFLATE(vec) do { if (kp > 0) { BL(dgemv_)("T", &n, &kp, &one, X, &n, vec, &IONE, &zero, cf, &IONE); const double mone = -1.0; BL(dgemv_)("N", &n, &kp, &mone, X, &n, cf, &IONE, &one, vec, &IONE); } } while (0)
            RR_DEFLATE(v);
            double nv = 0; for (int i = 0; i < n; i++) nv += v[i] * v[i]; nv = sqrt(nv);
            double alpha[16], beta[16]; int ml = 0;
            if (nv > 0) {
                for (int i = 0; i < n; i++) v[i] /= nv;
                memset(vold, 0, sizeof(double) * n);
                double bprev = 0;
                for (int l = 0; l < 12; l++) {
                    BL(dsymv_)("L", &n, &s, W, &n, v, &IONE, &zero, wv, &IONE);
                    RR_DEFLATE(wv);
                    double a = 0; for (int i = 0; i < n; i++) a += v[i] * wv[i];
                    for (int i = 0; i < n; i++) wv[i] -= a * v[i] + bprev * vold[i];
                    RR_DEFLATE(wv);
                    double b = 0; for (int i = 0; i < n; i++) b += wv[i] * wv[i]; b = sqrt(b);
                    alpha[ml] = a; beta[ml] = b; ml++;
                    if (b < 1e-14 * fmax(tmax, 1e-300)) break;
                    for (int i = 0; i < n; i++) { vold[i] = v[i]; v[i] = wv[i] / b; }
                    bprev = b;
                }
            }
            #undef RR_DEFLATE
            free(cf);
            /* the largest eigenvalue of the ml x ml tridiagonal matrix */
            double lmax = -INFINITY;
            if (ml > 0) {
                double d[16], e[16], z1 = 0; int info2 = 0, mm = ml;
                for (int i = 0; i < ml; i++) { d[i] = alpha[i]; e[i] = beta[i]; }
                (void)z1;
                BL(dsterf_)(&mm, d, e, &info2);
                if (info2 == 0) lmax = d[ml - 1];
            }
            if (lmax > thr) { if (ENV_ON("BRISK_RRDBG")) printf("   [rr n %d: missed direction %.2e > %.2e]\n", n, lmax, thr); return 0; }   /* a positive direction outside the basis */
        }
        /* accepted: T = X diag(theta) X' on the small side */
        memset(T, 0, sizeof(double) * (size_t)n * n);
        if (kp > 0) {
            for (int j = 0; j < kp; j++) { const double sq = sqrt(th[c0 + j]); double *x = X + (size_t)j * n; for (int i = 0; i < n; i++) x[i] *= sq; }
            BL(dsyrk_)("L", "N", &n, &kp, &one, X, &n, &zero, T, &n);
            for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) T[i + (size_t)j * n] = T[j + (size_t)i * n];
        }
        if (kn >= 1) {
            BL(dgemm_)("N", "N", &n, &kn, &nbq, &one, Q, &n, H + (size_t)cn * nbq, &nbq, &zero, f->Vp, &n);
            f->kwarm = kn;
        }
        if (small_pos) { f->npos = kp; f->nneg = n - kp; } else { f->nneg = kp; f->npos = n - kp; }
        return 1;
    }
    return 0;
}

/* ---- projection onto the cone: Pos = Pi_K(W), Neg = Pi_K(-W) = Pos - W ------------- */
static void fb_project(FB *f, const Block *B, const double *W, double *Pos, double *Neg) {
    const int n = f->n;
    if (B->type == BLK_LP) {
        for (int i = 0; i < n; i++) {
            if (f->freev && f->freev[i]) { Pos[i] = W[i]; Neg[i] = 0; }
            else { Pos[i] = W[i] > 0 ? W[i] : 0; Neg[i] = W[i] < 0 ? -W[i] : 0; }
        }
        return;
    }
    const double t0 = wtime();
    int info = 0, k = 0, small_pos = 1;
    double *Vk = NULL;                       /* k eigenvectors (scaled by sqrt|lambda| below) */
    if (g_fom_rr && n >= 120 && f->kwarm > 0 && f->npos + f->nneg == n && f->rr_since < g_rr_refresh) {
        const int sp = f->npos <= f->nneg;
        double *T = sp ? Pos : Neg;
        if (rr_project(f, W, sp, T)) {
            const size_t len = (size_t)n * n;
            if (sp) for (size_t i = 0; i < len; i++) Neg[i] = Pos[i] - W[i];
            else for (size_t i = 0; i < len; i++) Pos[i] = W[i] + Neg[i];
            f->n_rr++; f->rr_since++;
            f->t_eig += wtime() - t0;
            return;
        }
        f->n_rrfail++;
    }
    f->rr_since = 0;
    if (g_fom_single_now && f->Vf && g_fom_srange && f->npos + f->nneg == n && (f->npos <= n / 5 || f->nneg <= n / 5) && f->srZ && f->srwork) {
        /* 4.39: only the small side (<= n/5: the crossover with the split solver below, measured at n = 1 051), in single precision: ssyevr on the value range (0, +inf)
         * (or (-inf, 0)): the tridiagonalization, the eigenvalues in the range by MRRR and
         * k back-transformed vectors, against ssyevd's divide and conquer and n vectors. The
         * range is exact whatever the clustering near zero; the side's size is re-counted. */
        const size_t len = (size_t)n * n;
        for (size_t i = 0; i < len; i++) f->Vf[i] = (float)W[i];
        const int sp = f->npos <= f->nneg;
        const float big = 3.0e38f, vl = sp ? 0.0f : -big, vu = sp ? big : 0.0f, abst = 0.0f;
        int mf = 0, il = 1, iu = n;
        if (getenv("BRISK_DUMPW") && f->n_single >= atoi(getenv("BRISK_DUMPW")) && f->n_single < atoi(getenv("BRISK_DUMPW")) + 3 && n >= 1000) { char fn[128]; snprintf(fn, sizeof fn, "/tmp/claude-0/wdump_%ld.bin", f->n_single); FILE *fw = fopen(fn, "wb"); if (fw) { fwrite(&n, sizeof(int), 1, fw); fwrite(f->Vf, sizeof(float), len, fw); fclose(fw); } }
        const double tc = wtime();
        BL(ssyevr_)("V", "V", "L", &n, f->Vf, &n, &vl, &vu, &il, &iu, &abst, &mf, f->evf, f->srZ, &n, f->srsup, f->srwork, &f->srlw, f->sriwork, &f->srliw, &info);
        f->t_core += wtime() - tc;
        if (info == 0 && mf <= n / 3) {
            small_pos = sp; k = mf;
            for (int j = 0; j < k; j++) { const double sq = sqrt(fabs((double)f->evf[j])); const float *src = f->srZ + (size_t)j * n; double *dst = f->V + (size_t)j * n; for (int i = 0; i < n; i++) dst[i] = sq * (double)src[i]; }
            if (sp) { f->npos = k; f->nneg = n - k; } else { f->nneg = k; f->npos = n - k; }
            Vk = f->V; f->n_single++; f->n_srange++; f->kwarm = 0;
        }
    }
    if (!Vk && g_fom_single_now && f->Vf && f->dcw && !g_fom_rr) {
        /* 4.39: ssytrd, sstedc on T, the small side's vectors back-transformed (ssyevd: all n;
         * -28% at a side of 30% of n = 1 051) */
        const size_t len = (size_t)n * n;
        for (size_t i = 0; i < len; i++) f->Vf[i] = (float)W[i];
        int i1 = 0, i2 = 0, i3 = 0;
        const double tc = wtime();
        BL(ssytrd_)("L", &n, f->Vf, &n, f->dcd, f->dce, f->dctau, f->dcw, &f->dclw, &i1);
        if (i1 == 0) BL(sstedc_)("I", &n, f->dcd, f->dce, f->srZ, &n, f->dcw, &f->dclw, f->dciw, &f->dcliw, &i2);
        if (i1 == 0 && i2 == 0) {
            int npos = 0; for (int i = 0; i < n; i++) if (f->dcd[i] > 0) npos++;
            const int nneg = n - npos, sp = npos <= nneg, kk = sp ? npos : nneg, c0 = sp ? nneg : 0;
            if (kk > 0) BL(sormtr_)("L", "L", "N", &n, &kk, f->Vf, &n, f->dctau, f->srZ + (size_t)c0 * n, &n, f->dcw, &f->dclw, &i3);
            f->t_core += wtime() - tc;
            if (i3 == 0) {
                f->npos = npos; f->nneg = nneg; small_pos = sp; k = kk;
                for (int j = 0; j < k; j++) { const double sq = sqrt(fabs((double)f->dcd[c0 + j])); const float *src = f->srZ + (size_t)(c0 + j) * n; double *dst = f->V + (size_t)j * n; for (int i = 0; i < n; i++) dst[i] = sq * (double)src[i]; }
                Vk = f->V; f->n_single++; f->n_dcpart++; f->kwarm = 0;
            }
        } else f->t_core += wtime() - tc;
    }
    if (!Vk && g_fom_single_now && f->Vf) {
        /* 4.37: the eigendecomposition in single precision (ssyevd, 2.5-2.8x faster than dsyevd at
         * n = 1000-2000 here), the small side's scaled vectors back in double for the dsyrk; the
         * pair still satisfies Pos - Neg = W exactly. Used while the residuals are far above the
         * single-precision error of the projection (fom_single) */
        const size_t len = (size_t)n * n;
        for (size_t i = 0; i < len; i++) f->Vf[i] = (float)W[i];
        const double tc = wtime();
        BL(ssyevd_)("V", "L", &n, f->Vf, &n, f->evf, f->workf, &f->lworkf, f->iworkf, &f->liworkf, &info);
        f->t_core += wtime() - tc;
        if (info == 0) {
            int npos = 0; for (int i = 0; i < n; i++) if (f->evf[i] > 0) npos++;
            const int nneg = n - npos;
            f->npos = npos; f->nneg = nneg; small_pos = npos <= nneg; k = small_pos ? npos : nneg;
            const int c0 = small_pos ? nneg : 0;
            for (int j = 0; j < k; j++) { const double sq = sqrt(fabs((double)f->evf[c0 + j])); const float *src = f->Vf + (size_t)(c0 + j) * n; double *dst = f->V + (size_t)j * n; for (int i = 0; i < n; i++) dst[i] = sq * (double)src[i]; }
            Vk = f->V; f->n_single++; f->kwarm = 0;
            /* 4.39: the warm vectors of the Rayleigh-Ritz projection: the small side and a margin */
            if (g_fom_rr && f->Vp && n >= 120) {
                int kw = k + k / 8 + 8; if (kw > n / 4) kw = 0;
                if (kw > 0) {
                    const int w0 = small_pos ? n - kw : 0;
                    for (int j = 0; j < kw; j++) { const float *src = f->Vf + (size_t)(w0 + j) * n; double *dst = f->Vp + (size_t)j * n; for (int i = 0; i < n; i++) dst[i] = (double)src[i]; }
                    f->kwarm = kw;
                }
            }
        }
    }
    /* partial eigendecomposition (dsyevr on an index range) when the last iteration showed the
     * side of the spectrum to be computed is small: the top (or bottom) k_prev + margin pairs;
     * if the range does not reach the sign change, the full decomposition is done instead.
     * The tridiagonalization (4/3 n^3) remains, the back-transformation of the vectors
     * (2 n^3 for all) shrinks to 2 n^2 k. */
    const int kp = f->npos, kn = f->nneg;
    const int try_partial = !Vk && g_fom_partial && f->npos + f->nneg == n && (kp <= n / 5 || kn <= n / 5);   /* 4.39: n/5 (was n/4), the crossover with the split solver */
    /* Lanczos when the side is very small (rank r << n): O(n^2 (r + 30)) instead of O(n^3) */
    if (try_partial && g_fom_partial >= 2 && n >= 200 && (kp <= n / 12 || kn <= n / 12)) {
        small_pos = kp <= kn;
        const int kg = small_pos ? kp : kn;
        if (!f->lz) { f->lz = calloc(1, sizeof(Lz)); lz_alloc(f->lz, n, (kg + 40 > n / 3) ? n / 3 : kg + 40); }
        else if (f->lz->mdim < kg + 30) { lz_free(f->lz); lz_alloc(f->lz, n, (kg + 40 > n / 3) ? n / 3 : kg + 40); }
        const int kw = f->kwarm;
        int kf = lanczos_side(n, W, small_pos ? 1.0 : -1.0, kg, f->Vp, kw, f->lz, f->ev, f->V, n / 6);
        if (kf >= 0) {
            k = kf; Vk = f->V;
            memcpy(f->Vp, f->V, sizeof(double) * (size_t)n * k); f->kwarm = k;     /* warm start of the next call (unscaled) */
            for (int j = 0; j < k; j++) { const double sq = sqrt(fabs(f->ev[j])); double *col = Vk + (size_t)j * n; for (int i = 0; i < n; i++) col[i] *= sq; }
            if (small_pos) { f->npos = k; f->nneg = n - k; } else { f->nneg = k; f->npos = n - k; }
            f->n_lanczos++;
        } else f->n_lzfail++;
    }
    if (!Vk && try_partial) {
        small_pos = kp <= kn;
        int kt = (small_pos ? kp : kn) + 4 + (small_pos ? kp : kn) / 8;
        if (kt > n / 3) kt = n / 3;
        if (kt < 1) kt = 1;
        memcpy(f->V, W, sizeof(double) * (size_t)n * n);
        const int il = small_pos ? n - kt + 1 : 1, iu = small_pos ? n : kt;
        int mfound = 0; const double abstol = 0;
        BL(dsyevr_)("V", "I", "L", &n, f->V, &n, &DZERO, &DZERO, &il, &iu, &abstol, &mfound, f->ev, f->Vp, &n, f->isup, f->work, &f->lwork, f->iwork, &f->liwork, &info);
        if (info == 0 && mfound == kt) {
            /* eigenvalues ascending in ev[0..kt); the side is complete when the range crosses zero */
            int cnt = 0;
            if (small_pos) { for (int j = 0; j < kt; j++) if (f->ev[j] > 0) cnt++; if (f->ev[0] <= 0) { k = cnt; Vk = f->Vp + (size_t)(kt - cnt) * n; f->npos = cnt; f->nneg = n - cnt; } }
            else { for (int j = 0; j < kt; j++) if (f->ev[j] < 0) cnt++; if (f->ev[kt - 1] >= 0) { k = cnt; Vk = f->Vp; f->nneg = cnt; f->npos = n - cnt; } }
            if (Vk) for (int j = 0; j < k; j++) { const double s = sqrt(fabs(f->ev[small_pos ? kt - cnt + j : j])); double *col = Vk + (size_t)j * n; for (int i = 0; i < n; i++) col[i] *= s; }
        }
        if (Vk) f->n_partial++; else f->n_full++;
        f->kwarm = 0;
    }
    if (!Vk && f->dcdw && !g_fom_rr) {
        /* 4.39: dsytrd, dstedc on T, the small side's vectors back-transformed into the output
         * that is not the small side (free until the end) */
        memcpy(f->V, W, sizeof(double) * (size_t)n * n);
        int i1 = 0, i2 = 0, i3 = 0;
        double *Zt = NULL;
        BL(dsytrd_)("L", &n, f->V, &n, f->dcdd, f->dcde, f->dcdtau, f->dcdw, &f->dcdlw, &i1);
        /* the eigenvector matrix of T: into Pos (then the small side must be the negative one)
         * or Neg; decided after the eigenvalues, so D&C writes into Neg and the vectors are
         * moved if the small side is Neg */
        Zt = Neg;
        if (i1 == 0) BL(dstedc_)("I", &n, f->dcdd, f->dcde, Zt, &n, f->dcdw, &f->dcdlw, f->dcdiw, &f->dcdliw, &i2);
        if (i1 == 0 && i2 == 0) {
            int npos = 0; for (int i = 0; i < n; i++) if (f->dcdd[i] > 0) npos++;
            const int nneg = n - npos, sp = npos <= nneg, kk = sp ? npos : nneg, c0 = sp ? nneg : 0;
            if (kk > 0) BL(dormtr_)("L", "L", "N", &n, &kk, f->V, &n, f->dcdtau, Zt + (size_t)c0 * n, &n, f->dcdw, &f->dcdlw, &i3);
            if (i3 == 0) {
                f->npos = npos; f->nneg = nneg; small_pos = sp; k = kk;
                /* the vectors (unscaled) into Vp for the Lanczos warm start when the side is small */
                if (k > 0 && k <= n / 12 && f->Vp) { memcpy(f->Vp, Zt + (size_t)c0 * n, sizeof(double) * (size_t)n * k); f->kwarm = k; } else f->kwarm = 0;
                for (int j = 0; j < k; j++) { const double sq = sqrt(fabs(f->dcdd[c0 + j])); const double *src = Zt + (size_t)(c0 + j) * n; double *dst = f->V + (size_t)j * n; for (int i = 0; i < n; i++) dst[i] = sq * src[i]; }
                Vk = f->V; f->n_dcpart++;
                if (!try_partial) f->n_full++;
            }
        }
    }
    if (!Vk) {
        memcpy(f->V, W, sizeof(double) * (size_t)n * n);
        BL(dsyevd_)("V", "L", &n, f->V, &n, f->ev, f->work, &f->lwork, f->iwork, &f->liwork, &info);
        if (info != 0) { fprintf(stderr, "brisk: first-order engine: eigendecomposition failed (info %d)\n", info); exit(1); }
        int npos = 0; for (int i = 0; i < n; i++) if (f->ev[i] > 0) npos++;
        const int nneg = n - npos;
        f->npos = npos; f->nneg = nneg;
        small_pos = npos <= nneg;
        k = small_pos ? npos : nneg;
        Vk = f->V + (small_pos ? (size_t)nneg * n : 0);
        int kw_rr = 0;
        if (g_fom_rr && f->Vp && n >= 120) {    /* 4.39: warm vectors (unscaled), small side + margin */
            kw_rr = k + k / 8 + 8; if (kw_rr > n / 4) kw_rr = 0;
            if (kw_rr > 0) memcpy(f->Vp, f->V + (small_pos ? (size_t)(n - kw_rr) * n : 0), sizeof(double) * (size_t)n * kw_rr);
        }
        for (int j = 0; j < k; j++) {
            const double s = sqrt(fabs(f->ev[small_pos ? nneg + j : j]));
            double *col = Vk + (size_t)j * n;
            for (int i = 0; i < n; i++) col[i] *= s;
        }
        if (!try_partial) f->n_full++;
        /* warm start for Lanczos: the (unscaled) eigenvectors of the small side */
        if (kw_rr > 0) f->kwarm = kw_rr;
        else if (k > 0 && k <= n / 12 && f->Vp) { memcpy(f->Vp, f->V + (small_pos ? (size_t)nneg * n : 0), sizeof(double) * (size_t)n * k); f->kwarm = k; }
        else f->kwarm = 0;
    }
    /* the smaller side by a rank-k product, the other as the difference */
    double *T = small_pos ? Pos : Neg;
    memset(T, 0, sizeof(double) * (size_t)n * n);
    const double tk = wtime();
    if (k > 0) {
        BL(dsyrk_)("L", "N", &n, &k, &DONE, Vk, &n, &DZERO, T, &n);
        for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) T[i + (size_t)j * n] = T[j + (size_t)i * n];
    }
    const size_t len = (size_t)n * n;
    f->t_syrk += wtime() - tk;
    if (small_pos) for (size_t i = 0; i < len; i++) Neg[i] = Pos[i] - W[i];
    else for (size_t i = 0; i < len; i++) Pos[i] = W[i] + Neg[i];
    f->t_eig += wtime() - t0;
}


/* ======================================================================================
 * Phase II: augmented Lagrangian on the dual with a semismooth Newton-CG inner solver
 * (Zhao-Sun-Toh 2010 / SDPNAL+). With Z eliminated,
 *     psi(y) = -b'y + (1/(2 sigma)) ( ||Pi_K(X + sigma (A*y - C))||^2 - ||X||^2 ),
 *     grad   = -b + A(Pi_K(W)),   W = X + sigma (A*y - C),
 *     V d    = sigma A( dPi(W)[A* d] )    (generalized Jacobian, in the eigenbasis of W)
 * and the outer step is X <- Pi_K(W). dPi(W)[H] = P (Omega o (P'HP)) P' with Omega = 1 on
 * the positive-positive block, lambda_i/(lambda_i - lambda_j) between the signs and 0 on
 * the rest; computed through the smaller side in O(n^2 r).
 * ====================================================================================== */
typedef struct {
    int r, small_pos;          /* size of the small side; 1 = the positive side is the small one */
    double *V, *lam;           /* eigenvectors (ascending) and values of W */
    double *Pos;               /* Pi(W) */
    double *H, *U, *Y, *G;     /* scratch: n x n, r x n, r x n, n x n */
    double *wsmall;            /* the small side's eigenvalues / omega row factors */
} SB;

/* eigendecomposition of W into f->V / f->ev, Pos = Pi(W) */
static void ssn_eig(FB *f, const Block *B, const double *W, SB *sb) {
    const int n = f->n;
    if (B->type == BLK_LP) {
        for (int i = 0; i < n; i++) sb->Pos[i] = (f->freev && f->freev[i]) ? W[i] : (W[i] > 0 ? W[i] : 0);
        return;
    }
    const double t0 = wtime();
    memcpy(f->V, W, sizeof(double) * (size_t)n * n);
    int info = 0;
    BL(dsyevd_)("V", "L", &n, f->V, &n, f->ev, f->work, &f->lwork, f->iwork, &f->liwork, &info);
    if (info != 0) { fprintf(stderr, "brisk: first-order engine: eigendecomposition failed (info %d)\n", info); exit(1); }
    int npos = 0; for (int i = 0; i < n; i++) if (f->ev[i] > 0) npos++;
    f->npos = npos; f->nneg = n - npos;
    sb->small_pos = npos <= n - npos; sb->r = sb->small_pos ? npos : n - npos;
    /* Pos through the smaller side */
    const int k = sb->r; double *T = sb->small_pos ? sb->Pos : sb->G;
    memset(T, 0, sizeof(double) * (size_t)n * n);
    if (k > 0) {
        double *Vk = sb->U;      /* n x k scaled copy (U has n*n room) */
        const int off = sb->small_pos ? n - k : 0;
        for (int j = 0; j < k; j++) { const double sq = sqrt(fabs(f->ev[off + j])); const double *src = f->V + (size_t)(off + j) * n; double *dst = Vk + (size_t)j * n; for (int i = 0; i < n; i++) dst[i] = sq * src[i]; }
        BL(dsyrk_)("L", "N", &n, &k, &DONE, Vk, &n, &DZERO, T, &n);
        for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) T[i + (size_t)j * n] = T[j + (size_t)i * n];
    }
    const size_t len = (size_t)n * n;
    if (!sb->small_pos) for (size_t i = 0; i < len; i++) sb->Pos[i] = W[i] + sb->G[i];
    f->t_eig += wtime() - t0; f->n_full++;
}

/* G = dPi(W)[H] for the block (H symmetric dense); uses the small side */
static void ssn_jac(FB *f, const Block *B, SB *sb, const double *H, double *G) {
    const int n = f->n;
    if (B->type == BLK_LP) {
        /* Pos computed from W: derivative 1 where W > 0 (or free) */
        for (int i = 0; i < n; i++) G[i] = ((f->freev && f->freev[i]) || sb->Pos[i] > 0) ? H[i] : 0.0;
        return;
    }
    const int r = sb->r;
    const size_t len = (size_t)n * n;
    if (r == 0) { if (sb->small_pos) memset(G, 0, sizeof(double) * len); else memcpy(G, H, sizeof(double) * len); return; }
    /* small side S (r columns of V: the last r if positive, the first r if negative), other side O */
    const int off = sb->small_pos ? n - r : 0;
    const double *Vs = f->V + (size_t)off * n;
    /* U = Vs' H  (r x n) */
    BL(dgemm_)("T", "N", &r, &n, &n, &DONE, Vs, &n, H, &n, &DZERO, sb->U, &r);
    /* M = U V  (r x n, in V's column order) */
    BL(dgemm_)("N", "N", &r, &n, &n, &DONE, sb->U, &r, f->V, &n, &DZERO, sb->Y, &r);
    /* scale the columns of the other side by omega_ij = |lam_i| / (|lam_i| + |lam_j|) (i in S, j in O)
     * (for the positive small side: lam_i/(lam_i - lam_j); the sign-swapped case is the same formula
     * on -W, i.e. |.|); keep the S columns */
    for (int j = 0; j < n; j++) {
        const int inS = sb->small_pos ? (j >= n - r) : (j < r);
        if (inS) continue;
        const double lj = fabs(f->ev[j]);
        double *col = sb->Y + (size_t)j * r;
        for (int i = 0; i < r; i++) { const double li = fabs(f->ev[off + i]); col[i] *= li / (li + lj); }
    }
    /* Y2 = M V'  (r x n) : Y2 = sb->Y * V'  -> into U */
    BL(dgemm_)("N", "T", &r, &n, &n, &DONE, sb->Y, &r, f->V, &n, &DZERO, sb->U, &r);
    /* G1 = Vs U (n x n) */
    BL(dgemm_)("N", "N", &n, &n, &r, &DONE, Vs, &n, sb->U, &r, &DZERO, G, &n);
    /* G = G1 + G1' - Vs Mss Vs'  where Mss = the S x S block of M (already in sb->Y's S columns, unscaled) */
    for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) { const double v = G[i + (size_t)j * n] + G[j + (size_t)i * n]; G[i + (size_t)j * n] = v; G[j + (size_t)i * n] = v; }
    for (int i = 0; i < n; i++) G[i + (size_t)i * n] *= 2.0;
    {   /* Vs Mss Vs': K = Vs Mss (n x r), then G -= K Vs' */
        const double *Mss = sb->Y + (size_t)off * r;    /* r x r, columns off..off+r-1 of Y (leading dim r) */
        double *K = sb->U;                              /* n x r scratch (U's contents were consumed above) */
        BL(dgemm_)("N", "N", &n, &r, &r, &DONE, Vs, &n, Mss, &r, &DZERO, K, &n);
        const double mone = -1.0;
        BL(dgemm_)("N", "T", &n, &n, &r, &mone, K, &n, Vs, &n, &DONE, G, &n);
    }
    if (!sb->small_pos) for (size_t i = 0; i < len; i++) G[i] = H[i] - G[i];
}

/* 4.37: the diagonal of the projection's generalized Jacobian in the entry basis, approximately:
 * D_ij = sum_ab Omega_ab V_ia^2 V_jb^2 (the cross term sum_ab Omega_ab V_ia V_ib V_ja V_jb of the
 * off-diagonal entries left out), i.e. D = Q Omega Q' with Q = V o V: two GEMMs. Omega is 1 on the
 * positive-positive pairs, |l_a| / (|l_a| + |l_b|) between the signs, 0 on the negative ones.
 * LP: 1 where the projection is the identity. Q and T are n x n scratch. */
static void ssn_jacdiag(FB *f, const Block *B, SB *sb, double *D, double *Q, double *T) {
    const int n = f->n;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) D[i] = ((f->freev && f->freev[i]) || sb->Pos[i] > 0) ? 1.0 : 0.0; return; }
    const size_t len = (size_t)n * n;
    for (size_t i = 0; i < len; i++) Q[i] = f->V[i] * f->V[i];
    double *Om = D;       /* Omega first, then overwritten by the result */
    for (int b = 0; b < n; b++) for (int a = 0; a < n; a++) {
        const double la = f->ev[a], lb = f->ev[b];
        Om[a + (size_t)b * n] = (la > 0 && lb > 0) ? 1.0 : (la <= 0 && lb <= 0) ? 0.0 : (la > 0 ? la / (la - lb) : lb / (lb - la));
    }
    BL(dgemm_)("N", "N", &n, &n, &n, &DONE, Q, &n, Om, &n, &DZERO, T, &n);     /* T = Q Omega */
    BL(dgemm_)("N", "T", &n, &n, &n, &DONE, T, &n, Q, &n, &DZERO, D, &n);      /* D = T Q' */
}

/* psi, grad and the projections at y: W_k = X_k + sigma (A*y - C)_k; returns psi */
static double ssn_eval(Problem *P, FB *F, SB *S, int nb, int m, const double *y, double sigma, double *grad, double x2sum) {
    double psi = 0, pos2 = 0;
    for (int i = 0; i < m; i++) grad[i] = -P->b[i];
    double **Wp = fo_malloc(sizeof(double *) * (nb + 1)), **Pp = fo_malloc(sizeof(double *) * (nb + 1));
    for (int k = 0; k < nb; k++) { Wp[k] = F[k].W; Pp[k] = S[k].Pos; }
    fop_At(g_fop, P, y, Wp);
    #pragma omp parallel for schedule(dynamic) reduction(+:pos2) if (nb > 1)
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
        fb_addC(B, -1.0, f->W);
        for (size_t i = 0; i < len; i++) f->W[i] = f->X[i] + sigma * f->W[i];
        ssn_eig(f, B, f->W, &S[k]);
        double p2 = 0; for (size_t i = 0; i < len; i++) p2 += S[k].Pos[i] * S[k].Pos[i];
        pos2 += p2;
    }
    fop_A(g_fop, Pp, grad, 1.0);
    free(Wp); free(Pp);
    double by = 0; for (int i = 0; i < m; i++) by += P->b[i] * y[i];
    psi = -by + (pos2 - x2sum) / (2.0 * sigma);
    return psi;
}

/* v = (sigma A dPi A* + eps I) d */
static void ssn_matvec(Problem *P, FB *F, SB *S, int nb, int m, double sigma, double eps, const double *d, double *v) {
    for (int i = 0; i < m; i++) v[i] = eps * d[i];
    double **Hp = fo_malloc(sizeof(double *) * (nb + 1)), **Gp = fo_malloc(sizeof(double *) * (nb + 1));
    for (int k = 0; k < nb; k++) { Hp[k] = S[k].H; Gp[k] = S[k].G; }
    fop_At(g_fop, P, d, Hp);
    #pragma omp parallel for schedule(dynamic) if (nb > 1)
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; FB *f = &F[k];
        ssn_jac(f, B, &S[k], S[k].H, S[k].G);
    }
    fop_A(g_fop, Gp, v, sigma);
    free(Hp); free(Gp);
}

typedef struct { int newton, cg, ls, outer; double t; } SsnStat;

/* the phase: returns the status; y, X updated in place; residuals in *pinf, *dinf, *gap */
static int ssn_phase(Problem *P, const Params *par, FB *F, int nb, int m, double *y, double *sigma_io, const Gram *G,
                     double tol, double t0, int verbose, double *pinf_o, double *dinf_o, double *gap_o,
                     double *pobj_o, double *dobj_o, SsnStat *st, int *iters) {
    const double sc = P->bs * P->cs;
    SB *S = calloc(nb, sizeof(SB));
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; const size_t len = fbsz(B); const int n = B->n;
        S[k].Pos = fo_malloc(sizeof(double) * len); S[k].H = fo_malloc(sizeof(double) * len); S[k].G = fo_malloc(sizeof(double) * len);
        S[k].U = fo_malloc(sizeof(double) * (size_t)n * (B->type == BLK_LP ? 1 : n)); S[k].Y = fo_malloc(sizeof(double) * (size_t)n * (B->type == BLK_LP ? 1 : n));
    }
    double *grad = fo_malloc(sizeof(double) * m), *d = fo_malloc(sizeof(double) * m), *yt = fo_malloc(sizeof(double) * m);
    double **Dj = NULL; Gram *Gw = NULL; long n_wprec = 0, n_wfail = 0; double t_wprec = 0;
    if (par->fom_ssn_prec == 2) { Dj = fo_malloc(sizeof(double *) * nb); for (int k = 0; k < nb; k++) Dj[k] = fo_malloc(sizeof(double) * fbsz(&P->blk[k])); }
    double *cr = fo_malloc(sizeof(double) * m), *cp = fo_malloc(sizeof(double) * m), *cq = fo_malloc(sizeof(double) * m), *cz = fo_malloc(sizeof(double) * m);
    double *dprev = fo_malloc(sizeof(double) * m); int have_dprev = 0;
    double sigma = *sigma_io * par->fom_ssn_sig0;
    int status = ST_MAXIT;
    double pinf = 1, dinf = 1, gap = 1, pobj = 0, dobj = 0;
    double conv = P->bs / (1 + P->normb2);     /* reported primal residual per unit of |grad| (refined below) */
    { double dmax = 0; for (int i = 0; i < m; i++) dmax = fmax(dmax, P->du[i]); conv = dmax / (1 + P->normb2); }
    const int max_outer = par->fom_ssn_outer, max_newton = par->fom_ssn_newton, max_cg = par->fom_ssn_cg;
    const double ssn_c2 = getenv("BRISK_SSNC2") ? atof(getenv("BRISK_SSNC2")) : 0.0;   /* 4.37: curvature condition of the line search (0: Armijo only, the 4.31 rule) */
    const int ssn_eps0 = getenv("BRISK_SSNEPS0") ? atoi(getenv("BRISK_SSNEPS0")) : 0;
    const double ssn_tau = getenv("BRISK_SSNTAU") ? atof(getenv("BRISK_SSNTAU")) : 0.0;
    /* 4.39 (research, BRISK_SSNPROX=t): the proximal ALM of ISSUES 24(a): psi + (t / 2 sigma) |y - y_c|^2
     * with the prox centre y_c the outer step's starting y, in psi, its gradient, the Newton matrix
     * (V + (t / sigma) I) and the line search; the reported primal residual excludes the prox term */
    const double ssn_prox = getenv("BRISK_SSNPROX") ? atof(getenv("BRISK_SSNPROX")) : 0.0;
    double *y_c = ssn_prox > 0 ? fo_malloc(sizeof(double) * (m + 1)) : NULL;
    #define SSN_PROX(PSI, YY, GR) do { if (ssn_prox > 0) { const double pw = ssn_prox / sigma; double q2 = 0; \
        for (int i_ = 0; i_ < m; i_++) { const double dy_ = (YY)[i_] - y_c[i_]; q2 += dy_ * dy_; (GR)[i_] += pw * dy_; } (PSI) += 0.5 * pw * q2; } } while (0)
    /* 4.35: the best point (by max residual) is kept and restored on exit, and the phase gives
     * up (status -3, the caller resumes the splitting) after fom_ssn_stall outer steps
     * without a 20% improvement - on degenerate relaxations the Newton model can lose ground */
    double best_res = 1e300, best_pinf = 0, best_dinf = 0, best_gap = 0, best_pobj = 0, best_dobj = 0; int nstall = 0;
    const double res_entry = fmax(fmax(*pinf_o, *dinf_o), *gap_o), t_entry = wtime(), t_split = t_entry - t0;
    double *best_y = fo_malloc(sizeof(double) * m), **best_X = fo_malloc(sizeof(double *) * nb);
    /* 4.39: the time cap inside the Newton loop: a phase II attempt that has not beaten the entry
     * point by 20% after max(30 s, a quarter of the splitting's time) ends (the checks per outer
     * step come too late where one outer step takes many minutes: Example 8.1.3 d = 5) */
    double *y_entry = fo_malloc(sizeof(double) * m); memcpy(y_entry, y, sizeof(double) * m);
    const double t_cap = fmax(30.0, 0.25 * t_split); int capped = 0;
    for (int k = 0; k < nb; k++) best_X[k] = fo_malloc(sizeof(double) * fbsz(&P->blk[k]));
    int outer;
    for (outer = 1; outer <= max_outer; outer++) {
        double x2 = 0; for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); for (size_t i = 0; i < len; i++) x2 += F[k].X[i] * F[k].X[i]; }
        if (y_c) memcpy(y_c, y, sizeof(double) * m);
        double psi = ssn_eval(P, F, S, nb, m, y, sigma, grad, x2);
        SSN_PROX(psi, y, grad);
        double gn = 0; for (int i = 0; i < m; i++) gn += grad[i] * grad[i]; gn = sqrt(gn);
        /* inner tolerance: the primal residual (= |grad| in the scaled units, conv converts to the
         * reported measure) is driven to a fraction of the dual residual, never below tol/5 */
        const double eps_in = fmax(0.2 * tol, 0.2 * dinf) / conv;
        int nt, timeout = 0;
        const double gn_in = gn;
        for (nt = 0; nt < max_newton && gn > eps_in; nt++) {
            if (brisk_time_up(par)) { timeout = 1; break; }
            if (par->fom_ssn_stall > 0 && wtime() - t_entry > t_cap && best_res > 0.8 * res_entry) { capped = 1; break; }
            /* PCG on (V + eps I) d = -grad, preconditioner sigma A A* (the Gram factor): exact
             * where the projection is the identity, a rough scaling elsewhere */
            /* regularization of the Newton system: 4.31 min(1e-4, 0.1 |grad|); 4.37 options (research):
             * BRISK_SSNEPS0=1 freezes |grad| at the inner loop's start (the coupling with |grad| made
             * the steps alternate), BRISK_SSNTAU=t adds a floor t sigma (a proximal-like damping) */
            const double eps = fmax(fmin(1e-4, 0.1 * (ssn_eps0 ? gn_in : gn)), ssn_tau * sigma) + 1e-12;
            int prec = par->fom_ssn_prec;
            if (prec == 2) {
                /* 4.37: preconditioner sigma A diag(D) A* + eps I, D the diagonal of the Jacobian at y */
                const double tw = wtime();
                if (Gw) { fom_gram_free(Gw); Gw = NULL; }
                #pragma omp parallel for schedule(dynamic) if (nb > 1)
                for (int k = 0; k < nb; k++) { ssn_jacdiag(&F[k], &P->blk[k], &S[k], Dj[k], S[k].H, S[k].G); const size_t len = fbsz(&P->blk[k]); for (size_t i = 0; i < len; i++) Dj[k][i] *= sigma; }
                double tg = 0; Gw = gram_build(P, Dj, eps, 0, &tg);
                if (Gw) n_wprec++; else { n_wfail++; prec = 1; }
                t_wprec += wtime() - tw;
            }
            memset(d, 0, sizeof(double) * m);
            for (int i = 0; i < m; i++) cr[i] = -grad[i];
            if (par->fom_ssn_warm && have_dprev) {
                /* warm start along the previous Newton direction: x0 = s d_prev with the
                 * least-squares s (one matvec); kept when it reduces the residual */
                ssn_matvec(P, F, S, nb, m, sigma, eps + (ssn_prox > 0 ? ssn_prox / sigma : 0.0), dprev, cq);
                double den = 0; for (int i = 0; i < m; i++) den += cq[i] * cq[i];
                if (den > 0) {
                    /* minimize |cr - s cq| : s = <cq, cr> / <cq, cq> */
                    double cq_r = 0; for (int i = 0; i < m; i++) cq_r += cq[i] * cr[i];
                    const double sw = cq_r / den;
                    double r2 = 0; for (int i = 0; i < m; i++) { const double v = cr[i] - sw * cq[i]; r2 += v * v; }
                    if (r2 < 0.9 * gn * gn) { for (int i = 0; i < m; i++) { d[i] = sw * dprev[i]; cr[i] -= sw * cq[i]; } st->cg++; }
                }
            }
            memcpy(cz, cr, sizeof(double) * m);
            if (prec == 2) fom_gram_solve(Gw, cz);
            else if (prec) { fom_gram_solve(G, cz); for (int i = 0; i < m; i++) cz[i] /= sigma; }
            memcpy(cp, cz, sizeof(double) * m);
            double rz = 0; for (int i = 0; i < m; i++) rz += cr[i] * cz[i];
            double rr = 0; for (int i = 0; i < m; i++) rr += cr[i] * cr[i];
            const double cgtol = fmax(1e-14, fmin(par->fom_ssn_eta, sqrt(gn))) * gn;
            int cgit;
            for (cgit = 0; cgit < max_cg && sqrt(rr) > cgtol; cgit++) {
                if ((cgit & 15) == 15 && brisk_time_up(par)) break;
                ssn_matvec(P, F, S, nb, m, sigma, eps + (ssn_prox > 0 ? ssn_prox / sigma : 0.0), cp, cq);
                double pq = 0; for (int i = 0; i < m; i++) pq += cp[i] * cq[i];
                if (pq <= 0) break;
                const double al = rz / pq;
                for (int i = 0; i < m; i++) { d[i] += al * cp[i]; cr[i] -= al * cq[i]; }
                rr = 0; for (int i = 0; i < m; i++) rr += cr[i] * cr[i];
                memcpy(cz, cr, sizeof(double) * m);
                if (prec == 2) fom_gram_solve(Gw, cz);
                else if (prec) { fom_gram_solve(G, cz); for (int i = 0; i < m; i++) cz[i] /= sigma; }
                double rz2 = 0; for (int i = 0; i < m; i++) rz2 += cr[i] * cz[i];
                const double be = rz2 / rz; rz = rz2;
                for (int i = 0; i < m; i++) cp[i] = cz[i] + be * cp[i];
            }
            st->cg += cgit;
            memcpy(dprev, d, sizeof(double) * m); have_dprev = 1;
            if (ENV_ON("BRISK_SSNDBG") && nt == 0 && outer <= 3) {
                /* finite-difference check of the Jacobian along d: V d vs (grad(y + h d) - grad(y)) / h */
                double dn = 0; for (int i = 0; i < m; i++) dn += d[i] * d[i]; dn = sqrt(dn);
                const double h = 1e-6 / fmax(dn, 1e-300);
                ssn_matvec(P, F, S, nb, m, sigma, 0.0, d, cq);
                double *g0 = fo_malloc(sizeof(double) * m), *g1 = fo_malloc(sizeof(double) * m);
                memcpy(g0, grad, sizeof(double) * m);
                for (int i = 0; i < m; i++) yt[i] = y[i] + h * d[i];
                ssn_eval(P, F, S, nb, m, yt, sigma, g1, x2);
                double e2 = 0, v2 = 0; for (int i = 0; i < m; i++) { const double fd = (g1[i] - g0[i]) / h; e2 += (fd - cq[i]) * (fd - cq[i]); v2 += cq[i] * cq[i]; }
                printf("      [jacobian check: |V d - FD| / |V d| = %.2e]\n", sqrt(e2) / fmax(sqrt(v2), 1e-300));
                ssn_eval(P, F, S, nb, m, y, sigma, grad, x2);      /* restore */
                free(g0); free(g1);
            }
            double gd = 0; for (int i = 0; i < m; i++) gd += grad[i] * d[i];
            if (gd >= 0) { for (int i = 0; i < m; i++) d[i] = -grad[i]; gd = -gn * gn; }
            /* line search (each trial: one projection per block). 4.37: psi is convex along d, so
             * phi'(a) = grad(y + a d)'d is monotone: a trial is accepted when it decreases psi
             * (Armijo) and does not overshoot the minimum along d by much (phi'(a) <= c2 |phi'(0)|);
             * otherwise the next trial is the safeguarded secant root of phi' between the last
             * point below the minimum and the first one past it. The plain Armijo rule accepted
             * the full step on moment relaxations while |grad| jumped 15x and fell back the next
             * step, for dozens of Newton steps (roa_vdp_d10_I: psi down 1e-4 a step). */
            double al = 1.0, psi_t = 0; int ok = 0;
            double a_lo = 0, g_lo = gd, a_hi = -1, g_hi = 0;
            for (int ls = 0; ls < 12; ls++) {
                for (int i = 0; i < m; i++) yt[i] = y[i] + al * d[i];
                psi_t = ssn_eval(P, F, S, nb, m, yt, sigma, grad, x2);
                SSN_PROX(psi_t, yt, grad);
                st->ls++;
                const int armijo = psi_t <= psi + 1e-4 * al * gd;
                if (ssn_c2 <= 0) { if (armijo) { ok = 1; break; } al *= 0.5; continue; }       /* the 4.31 rule */
                double gt = 0; for (int i = 0; i < m; i++) gt += grad[i] * d[i];
                if (armijo && gt <= ssn_c2 * fabs(gd)) { ok = 1; break; }
                if (armijo && gt < 0) { a_lo = al; g_lo = gt; if (a_hi < 0) { ok = 1; break; } }   /* still descending at the longest step: take it */
                else { a_hi = al; g_hi = gt; }
                double an = (a_hi > 0 && g_hi > g_lo) ? a_lo - g_lo * (a_hi - a_lo) / (g_hi - g_lo) : 0.5 * (a_lo + (a_hi > 0 ? a_hi : 2 * al));
                const double w = a_hi > 0 ? a_hi - a_lo : al;
                if (an < a_lo + 0.1 * w) an = a_lo + 0.1 * w;
                if (a_hi > 0 && an > a_hi - 0.1 * w) an = a_hi - 0.1 * w;
                if (!armijo && gt <= 0) an = 0.5 * al;       /* (not convex along d numerically: halve) */
                al = an;
            }
            if (!ok) { double pq_ = ssn_eval(P, F, S, nb, m, y, sigma, grad, x2); SSN_PROX(pq_, y, grad); (void)pq_; break; }   /* restore the state at y */
            memcpy(y, yt, sizeof(double) * m); psi = psi_t;
            const double gn_old = gn;
            gn = 0; for (int i = 0; i < m; i++) gn += grad[i] * grad[i]; gn = sqrt(gn);
            st->newton++;
            if (ENV_ON("BRISK_SSNDBG")) { printf("      newton %2d: |grad| %.2e -> %.2e, cg %d (res %.1e, tol %.1e), step %.3g, eps %.1e, psi %.10e\n", nt + 1, gn_old, gn, cgit, sqrt(rr), cgtol, al, eps, psi); fflush(stdout); }
        }
        if (capped) {
            if (verbose > 1) printf("ALM: no gain after %.0fs (cap %.0fs): phase II ends\n", wtime() - t_entry, t_cap);
            if (best_res >= 1e300) { memcpy(y, y_entry, sizeof(double) * m); pinf = *pinf_o; dinf = *dinf_o; gap = *gap_o; pobj = *pobj_o; dobj = *dobj_o; }
            status = -3; break;
        }
        /* outer update X <- Pos, residuals */
        double rd2 = 0, po = 0; pobj = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
            for (size_t i = 0; i < len; i++) { const double v = S[k].Pos[i] - f->X[i]; rd2 += v * v; f->X[i] = S[k].Pos[i]; }
            pobj += fb_Cdot(B, f->X);
        }
        rd2 /= sigma * sigma;
        const double gn_inner = gn;   /* the inner problem's gradient (with the prox term): decides inner_ok */
        if (ssn_prox > 0) { const double pw = ssn_prox / sigma; for (int i = 0; i < m; i++) grad[i] -= pw * (y[i] - y_c[i]); gn = 0; for (int i = 0; i < m; i++) gn += grad[i] * grad[i]; gn = sqrt(gn); }
        double rpo = 0; for (int i = 0; i < m; i++) { const double v = grad[i] * P->du[i]; rpo += v * v; } rpo = sqrt(rpo);
        if (gn > 0 && rpo > 0) conv = rpo / gn / (1 + P->normb2);
        dobj = 0; for (int i = 0; i < m; i++) dobj += P->b[i] * y[i];
        po = sc * pobj; const double dob = sc * dobj, den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        pinf = rpo / (1 + P->normb2); dinf = P->cs * sqrt(rd2) / (1 + P->normC2); gap = fabs(po - dob) / den;
        const double res = fmax(fmax(pinf, dinf), gap);
        st->outer++; (*iters)++;
        if (g_ftrace) { fprintf(g_ftrace, "%d %.3f %.3e %.3e %.3e %.3e %.10e %.10e\n", *iters, wtime() - t0, pinf, dinf, gap, sigma, po + P->obj_off, dob + P->obj_off); fflush(g_ftrace); }
        if (verbose > 1) { printf("ALM%2d %14.7e %14.7e %8.1e %8.1e %8.1e %8.2g %7.1fs  (%d Newton, %d CG, |grad| %.1e)\n", outer, po + P->obj_off, dob + P->obj_off, pinf, dinf, gap, sigma, wtime() - t0, nt, st->cg, gn); fflush(stdout); }
        if (res < 0.8 * best_res) nstall = 0; else nstall++;
        if (res < best_res) {
            best_res = res; best_pinf = pinf; best_dinf = dinf; best_gap = gap; best_pobj = pobj; best_dobj = dobj;
            memcpy(best_y, y, sizeof(double) * m);
            for (int k = 0; k < nb; k++) memcpy(best_X[k], F[k].X, sizeof(double) * fbsz(&P->blk[k]));
        }
        if (res <= tol) { status = ST_OPTIMAL; break; }
        if (timeout || (brisk_time_up(par))) { status = ST_TIME; break; }
        /* stalled: fom_ssn_stall outer steps without a 20% gain on the best, or as much time as
         * the splitting took to get here without beating the entry point by 20% */
        if (par->fom_ssn_stall > 0 && (nstall >= par->fom_ssn_stall || (wtime() - t_entry > t_split && best_res > 0.8 * res_entry))) { status = -3; break; }
        /* 4.37: the rate race: phase II is worth its time only while it reduces the residual faster
         * (in log per second) than the splitting did just before it; after two outer steps and a
         * second, at half the splitting's rate it gives up (Example 8.1.3 d = 3: four attempts took
         * 100 of 120 s where the splitting alone gains a decade in 10 s) */
        if (par->fom_ssn_stall > 0 && g_split_rate > 0 && outer >= 2 && wtime() - t_entry > 1.0) {
            const double ph_rate = log(res_entry / fmin(best_res, res_entry)) / (wtime() - t_entry);
            if (ph_rate < 0.5 * g_split_rate) { if (verbose > 1) printf("ALM: slower than the splitting (%.2e against %.2e decades/s)\n", ph_rate / log(10.0), g_split_rate / log(10.0)); status = -3; break; }
        }
        /* penalty: the inner solve fixes the primal residual, the outer steps X <- Pi(W) drive
         * the dual one; a larger sigma makes the outer steps longer (fewer outer iterations)
         * and the inner problems harder */
        const int inner_ok = gn_inner <= eps_in;
        if (!inner_ok) sigma = fmax(sigma / 2.0, 1e-8);                 /* the inner problem was too hard */
        else if (pinf < dinf) sigma = fmin(sigma * par->fom_ssn_rho, 1e8);
        else if (pinf > 10 * dinf) sigma = fmax(sigma / 2.0, 1e-8);
    }
    if (best_res < 1e300 && fmax(fmax(pinf, dinf), gap) > best_res) {
        if (verbose > 1) printf("ALM: back to the best point (%.1e, from %.1e)\n", best_res, fmax(fmax(pinf, dinf), gap));
        pinf = best_pinf; dinf = best_dinf; gap = best_gap; pobj = best_pobj; dobj = best_dobj;
        memcpy(y, best_y, sizeof(double) * m);
        for (int k = 0; k < nb; k++) memcpy(F[k].X, best_X[k], sizeof(double) * fbsz(&P->blk[k]));
    }
    free(best_y); free(y_entry); free(y_c);
    for (int k = 0; k < nb; k++) free(best_X[k]);
    free(best_X);
    #undef SSN_PROX
    if (Gw) fom_gram_free(Gw);
    if (Dj) { for (int k = 0; k < nb; k++) free(Dj[k]); free(Dj); }
    if (verbose > 1 && (n_wprec || n_wfail)) printf("ALM: Jacobian-diagonal preconditioner built %ld times (%ld failed), %.1fs\n", n_wprec, n_wfail, t_wprec);
    *sigma_io = sigma; *pinf_o = pinf; *dinf_o = dinf; *gap_o = gap; *pobj_o = pobj; *dobj_o = dobj;
    for (int k = 0; k < nb; k++) { free(S[k].Pos); free(S[k].H); free(S[k].G); free(S[k].U); free(S[k].Y); }
    free(S); free(grad); free(d); free(yt); free(cr); free(cp); free(cq); free(cz); free(dprev);
    return status;
}


/* ======================================================================================
 * Kernel B: low-rank augmented Lagrangian (Burer-Monteiro), X_k = R_k R_k', LP x = v^2 (free
 * x = v). Inner solver L-BFGS on the stacked R; the multipliers y and the penalty rho as in
 * ManiSDP/SDPLR; the rank of each block grows by the eigenvectors of the most negative
 * eigenvalues of Z_k = (C - A*y)_k and shrinks on rank deficiency. Cost per gradient:
 * O(nnz(A) r + nnz(S) r) - no n^3, no m^2.
 *     L(R; y, rho) = <C,X> - y'(A(X) - b) + rho/2 ||A(X) - b||^2,  grad_R = 2 S R,
 *     S = C - A*(y - rho (A(X) - b))
 * ====================================================================================== */
typedef struct {
    int n, type, r, rmax;
    double *R;            /* n x rmax (LP: n) */
    /* union pattern (SDP): CSR over the full symmetric expansion of C and all A_t */
    int *sp, *sj; double *sv;     /* S values */
    int nnzS;
    int *cpos;            /* per constraint t: offsets into a flat (entry -> S position) list */
    int *epos; double *eval;      /* flattened entries of A_t: S position + value (full expansion) */
    int *c_epos; double *c_eval; int c_ne;   /* C */
    double *SR, *G;       /* n x rmax scratch */
    const unsigned char *freev;   /* LP: free slots (x = v, not v^2) */
} BB;

static int cmp_int2(const void *a, const void *b) { const int *x = a, *y = b; return x[0] != y[0] ? x[0] - y[0] : x[1] - y[1]; }

static void bb_setup(BB *q, const Block *B, int rmax) {
    const int n = B->n;
    q->n = n; q->type = B->type; q->rmax = rmax;
    if (B->type == BLK_LP) { q->R = calloc(n, sizeof(double)); q->G = fo_malloc(sizeof(double) * n); return; }
    q->R = calloc((size_t)n * rmax, sizeof(double));
    q->SR = fo_malloc(sizeof(double) * (size_t)n * rmax); q->G = fo_malloc(sizeof(double) * (size_t)n * rmax);
    /* union pattern */
    size_t tot = B->C.ef; for (int t = 0; t < B->ncon; t++) tot += B->A[t].ef;
    int (*pairs)[2] = fo_malloc(sizeof(int[2]) * (tot ? tot : 1));
    size_t np = 0;
    for (int qq = 0; qq < B->C.ef; qq++) { pairs[np][0] = B->C.fr[qq]; pairs[np][1] = B->C.fc[qq]; np++; }
    for (int t = 0; t < B->ncon; t++) for (int qq = 0; qq < B->A[t].ef; qq++) { pairs[np][0] = B->A[t].fr[qq]; pairs[np][1] = B->A[t].fc[qq]; np++; }
    qsort(pairs, np, sizeof(int[2]), cmp_int2);
    size_t nu = 0;
    for (size_t p = 0; p < np; p++) if (!nu || pairs[nu - 1][0] != pairs[p][0] || pairs[nu - 1][1] != pairs[p][1]) { pairs[nu][0] = pairs[p][0]; pairs[nu][1] = pairs[p][1]; nu++; }
    q->nnzS = (int)nu;
    q->sp = calloc(n + 1, sizeof(int)); q->sj = fo_malloc(sizeof(int) * (nu ? nu : 1)); q->sv = fo_malloc(sizeof(double) * (nu ? nu : 1));
    for (size_t p = 0; p < nu; p++) q->sp[pairs[p][0] + 1]++;
    for (int i = 0; i < n; i++) q->sp[i + 1] += q->sp[i];
    for (size_t p = 0; p < nu; p++) q->sj[p] = pairs[p][1];
    /* position lookup: binary search in the row's column list */
    #define SPOS(i, j) ({ int lo_ = q->sp[i], hi_ = q->sp[(i) + 1] - 1, r_ = -1; while (lo_ <= hi_) { int mid_ = (lo_ + hi_) / 2; if (q->sj[mid_] == (j)) { r_ = mid_; break; } if (q->sj[mid_] < (j)) lo_ = mid_ + 1; else hi_ = mid_ - 1; } r_; })
    q->cpos = fo_malloc(sizeof(int) * (B->ncon + 1)); q->cpos[0] = 0;
    for (int t = 0; t < B->ncon; t++) q->cpos[t + 1] = q->cpos[t] + B->A[t].ef;
    q->epos = fo_malloc(sizeof(int) * (q->cpos[B->ncon] + 1)); q->eval = fo_malloc(sizeof(double) * (q->cpos[B->ncon] + 1));
    for (int t = 0; t < B->ncon; t++) for (int qq = 0; qq < B->A[t].ef; qq++) { q->epos[q->cpos[t] + qq] = SPOS(B->A[t].fr[qq], B->A[t].fc[qq]); q->eval[q->cpos[t] + qq] = B->A[t].fv[qq]; }
    q->c_ne = B->C.ef; q->c_epos = fo_malloc(sizeof(int) * (q->c_ne + 1)); q->c_eval = fo_malloc(sizeof(double) * (q->c_ne + 1));
    for (int qq = 0; qq < B->C.ef; qq++) { q->c_epos[qq] = SPOS(B->C.fr[qq], B->C.fc[qq]); q->c_eval[qq] = B->C.fv[qq]; }
    #undef SPOS
    free(pairs);
}
static void bb_free(BB *q) { free(q->R); free(q->sp); free(q->sj); free(q->sv); free(q->cpos); free(q->epos); free(q->eval); free(q->c_epos); free(q->c_eval); free(q->SR); free(q->G); }

/* X entries on the union pattern: xs[p] = (R R')_{ij}; then A(X) and <C,X> */
static void bb_AX(const Block *B, BB *q, double *xs, double *ax, double *cx) {
    const int n = q->n, r = q->r;
    if (B->type == BLK_LP) {
        #define XV(i) ((q->freev && q->freev[i]) ? q->R[i] : q->R[i] * q->R[i])
        for (int t = 0; t < B->ncon; t++) { const SpSym *S = &B->A[t]; double v = 0; for (int qq = 0; qq < S->nnz; qq++) { const int i = S->row[qq]; v += S->val[qq] * XV(i); } ax[B->con[t]] += v; }
        double c = 0; for (int qq = 0; qq < B->C.nnz; qq++) { const int i = B->C.row[qq]; c += B->C.val[qq] * XV(i); }
        #undef XV
        *cx += c; return;
    }
    for (int i = 0; i < n; i++) for (int p = q->sp[i]; p < q->sp[i + 1]; p++) {
        const int j = q->sj[p]; double v = 0;
        for (int l = 0; l < r; l++) v += q->R[i + (size_t)l * n] * q->R[j + (size_t)l * n];
        xs[p] = v;
    }
    for (int t = 0; t < B->ncon; t++) { double v = 0; for (int e = q->cpos[t]; e < q->cpos[t + 1]; e++) v += q->eval[e] * xs[q->epos[e]]; ax[B->con[t]] += v; }
    double c = 0; for (int e = 0; e < q->c_ne; e++) c += q->c_eval[e] * xs[q->c_epos[e]];
    *cx += c;
}
/* S = C - A*(yh) on the pattern, G = 2 S R (the gradient) */
static void bb_grad(const Block *B, BB *q, const double *yh, double *G) {
    const int n = q->n, r = q->r;
    if (B->type == BLK_LP) {
        /* s_i = c_i - sum_t yh_t a_ti ; grad_v = 2 s_i v_i */
        double *sv = fo_malloc(sizeof(double) * n); memset(sv, 0, sizeof(double) * n);
        for (int qq = 0; qq < B->C.nnz; qq++) sv[B->C.row[qq]] += B->C.val[qq];
        for (int t = 0; t < B->ncon; t++) { const SpSym *S = &B->A[t]; const double yt = yh[B->con[t]]; for (int qq = 0; qq < S->nnz; qq++) sv[S->row[qq]] -= yt * S->val[qq]; }
        for (int i = 0; i < n; i++) G[i] = (q->freev && q->freev[i]) ? sv[i] : 2.0 * sv[i] * q->R[i];
        free(sv); return;
    }
    memset(q->sv, 0, sizeof(double) * q->nnzS);
    for (int e = 0; e < q->c_ne; e++) q->sv[q->c_epos[e]] += q->c_eval[e];
    for (int t = 0; t < B->ncon; t++) { const double yt = yh[B->con[t]]; if (yt == 0) continue; for (int e = q->cpos[t]; e < q->cpos[t + 1]; e++) q->sv[q->epos[e]] -= yt * q->eval[e]; }
    for (int l = 0; l < r; l++) {
        const double *Rl = q->R + (size_t)l * n; double *Gl = G + (size_t)l * n;
        for (int i = 0; i < n; i++) { double v = 0; for (int p = q->sp[i]; p < q->sp[i + 1]; p++) v += q->sv[p] * Rl[q->sj[p]]; Gl[i] = 2.0 * v; }
    }
}


typedef struct { int outer, inner, ls, rankup; double t; } BmStat;

/* the stacked vector: per block n_k r_k (LP: n_k) */
static size_t bb_len(const BB *Q, int nb) { size_t N = 0; for (int k = 0; k < nb; k++) N += Q[k].type == BLK_LP ? (size_t)Q[k].n : (size_t)Q[k].n * Q[k].r; return N; }
static void bb_gather(const BB *Q, int nb, double *v, int from_G) { size_t o = 0; for (int k = 0; k < nb; k++) { const size_t len = Q[k].type == BLK_LP ? (size_t)Q[k].n : (size_t)Q[k].n * Q[k].r; memcpy(v + o, from_G ? Q[k].G : Q[k].R, sizeof(double) * len); o += len; } }
static void bb_scatter(BB *Q, int nb, const double *v) { size_t o = 0; for (int k = 0; k < nb; k++) { const size_t len = Q[k].type == BLK_LP ? (size_t)Q[k].n : (size_t)Q[k].n * Q[k].r; memcpy(Q[k].R, v + o, sizeof(double) * len); o += len; } }

/* f(R) and its gradient (into Q[k].G); ax = A(X) - b returned in r */
static double bm_fg(Problem *P, BB *Q, int nb, int m, const double *y, double rho, double *r, double *yh, double **xs, double *cx_out) {
    double cx = 0;
    for (int i = 0; i < m; i++) r[i] = -P->b[i];
    for (int k = 0; k < nb; k++) bb_AX(&P->blk[k], &Q[k], xs[k], r, &cx);
    double f = cx, rr = 0; for (int i = 0; i < m; i++) { f -= y[i] * r[i]; rr += r[i] * r[i]; yh[i] = y[i] - rho * r[i]; }
    f += 0.5 * rho * rr;
    #pragma omp parallel for schedule(dynamic) if (nb > 1)
    for (int k = 0; k < nb; k++) bb_grad(&P->blk[k], &Q[k], yh, Q[k].G);
    *cx_out = cx;
    return f;
}

static int bm_phase(Problem *P, const Params *par, FB *F, int nb, int m, double *y, double tol, double t0, int verbose,
                    double *pinf_o, double *dinf_o, double *gap_o, double *pobj_o, double *dobj_o, BmStat *st, int *iters) {
    const double sc = P->bs * P->cs;
    BB *Q = calloc(nb, sizeof(BB));
    double **xs = fo_malloc(sizeof(double *) * nb);
    /* start: R from the eigenvectors of the splitting's X (rank by a relative threshold) */
    int rtot = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; FB *f = &F[k]; const int n = B->n;
        int r0 = 1;
        if (B->type == BLK_SDP) {
            memcpy(f->V, f->X, sizeof(double) * (size_t)n * n);
            int info = 0; BL(dsyevd_)("V", "L", &n, f->V, &n, f->ev, f->work, &f->lwork, f->iwork, &f->liwork, &info);
            const double lmax = fmax(f->ev[n - 1], 0.0);
            r0 = 0; for (int i = 0; i < n; i++) if (f->ev[i] > 1e-6 * lmax && f->ev[i] > 0) r0++;
            if (r0 < 1) r0 = 1;
            if (par->fom_bm_rank0 > 0 && r0 > par->fom_bm_rank0) r0 = par->fom_bm_rank0;   /* start low; the escape rule grows it */
            const int rmax = (r0 + 20 > n) ? n : r0 + 20;
            bb_setup(&Q[k], B, rmax);
            Q[k].r = r0;
            for (int l = 0; l < r0; l++) { const int c = n - 1 - l; const double sq = sqrt(fmax(f->ev[c], 1e-12)); for (int i = 0; i < n; i++) Q[k].R[i + (size_t)l * n] = sq * f->V[i + (size_t)c * n]; }
            xs[k] = fo_malloc(sizeof(double) * (Q[k].nnzS ? Q[k].nnzS : 1));
        } else {
            bb_setup(&Q[k], B, 1); Q[k].freev = f->freev; Q[k].r = 1;
            for (int i = 0; i < n; i++) Q[k].R[i] = (f->freev && f->freev[i]) ? f->X[i] : sqrt(fmax(f->X[i], 0.0));
            xs[k] = NULL;
        }
        rtot += Q[k].r;
    }
    if (verbose) printf("first-order: phase B (low-rank augmented Lagrangian): initial ranks total %d\n", rtot);
    double *r = fo_malloc(sizeof(double) * (m + 1)), *yh = fo_malloc(sizeof(double) * (m + 1));
    double rho = par->fom_bm_rho > 0 ? par->fom_bm_rho : 1.0;
    const int mem = 10;
    int status = ST_MAXIT, outer;
    double pinf = 1, dinf = 1, gap = 1, pobj = 0, dobj = 0, rn_prev = 1e300;
    for (outer = 1; outer <= par->fom_bm_outer; outer++) {
        /* ---- inner: L-BFGS on f(R) ---- */
        const size_t N = bb_len(Q, nb);
        double *x = fo_malloc(sizeof(double) * N), *g = fo_malloc(sizeof(double) * N), *xn = fo_malloc(sizeof(double) * N), *gn = fo_malloc(sizeof(double) * N), *d = fo_malloc(sizeof(double) * N);
        double *Sm = fo_malloc(sizeof(double) * N * mem), *Ym = fo_malloc(sizeof(double) * N * mem), rhoM[16], al[16];
        int nm = 0, head = 0;
        double cx = 0;
        double f = bm_fg(P, Q, nb, m, y, rho, r, yh, xs, &cx);
        bb_gather(Q, nb, x, 0); bb_gather(Q, nb, g, 1);
        double gnorm = 0; for (size_t i = 0; i < N; i++) gnorm += g[i] * g[i]; gnorm = sqrt(gnorm);
        const double eps_in = fmax(par->fom_bm_gtol, 1e-2 * tol) * (1.0 + gnorm * 0 );
        int inner;
        for (inner = 0; inner < par->fom_bm_inner && gnorm > eps_in; inner++) {
            if ((inner & 7) == 7 && brisk_time_up(par)) break;
            /* two-loop recursion */
            memcpy(d, g, sizeof(double) * N);
            for (int j = 0; j < nm; j++) { const int idx = (head - 1 - j + mem) % mem; const double *sj = Sm + (size_t)idx * N, *yj = Ym + (size_t)idx * N; double dot = 0; for (size_t i = 0; i < N; i++) dot += sj[i] * d[i]; al[idx] = rhoM[idx] * dot; for (size_t i = 0; i < N; i++) d[i] -= al[idx] * yj[i]; }
            if (nm > 0) { const int idx = (head - 1 + mem) % mem; const double *sj = Sm + (size_t)idx * N, *yj = Ym + (size_t)idx * N; double sy = 0, yy = 0; for (size_t i = 0; i < N; i++) { sy += sj[i] * yj[i]; yy += yj[i] * yj[i]; } const double gam = sy / fmax(yy, 1e-300); for (size_t i = 0; i < N; i++) d[i] *= gam; }
            for (int j = nm - 1; j >= 0; j--) { const int idx = (head - 1 - j + mem) % mem; const double *sj = Sm + (size_t)idx * N, *yj = Ym + (size_t)idx * N; double dot = 0; for (size_t i = 0; i < N; i++) dot += yj[i] * d[i]; const double be = rhoM[idx] * dot; for (size_t i = 0; i < N; i++) d[i] += (al[idx] - be) * sj[i]; }
            for (size_t i = 0; i < N; i++) d[i] = -d[i];
            double gd = 0; for (size_t i = 0; i < N; i++) gd += g[i] * d[i];
            if (gd >= 0) { for (size_t i = 0; i < N; i++) d[i] = -g[i]; gd = -gnorm * gnorm; nm = 0; }
            /* backtracking Armijo */
            double step = nm == 0 ? fmin(1.0, 1.0 / gnorm) : 1.0, fn = f; int ok = 0;
            for (int ls = 0; ls < 30; ls++) {
                for (size_t i = 0; i < N; i++) xn[i] = x[i] + step * d[i];
                bb_scatter(Q, nb, xn);
                fn = bm_fg(P, Q, nb, m, y, rho, r, yh, xs, &cx); st->ls++;
                if (fn <= f + 1e-4 * step * gd) { ok = 1; break; }
                step *= 0.5;
            }
            if (!ok) { bb_scatter(Q, nb, x); bm_fg(P, Q, nb, m, y, rho, r, yh, xs, &cx); break; }
            bb_gather(Q, nb, gn, 1);
            /* update memory */
            { double *sj = Sm + (size_t)head * N, *yj = Ym + (size_t)head * N; double sy = 0; for (size_t i = 0; i < N; i++) { sj[i] = xn[i] - x[i]; yj[i] = gn[i] - g[i]; sy += sj[i] * yj[i]; } if (sy > 1e-14) { rhoM[head] = 1.0 / sy; head = (head + 1) % mem; if (nm < mem) nm++; } }
            memcpy(x, xn, sizeof(double) * N); memcpy(g, gn, sizeof(double) * N); f = fn;
            gnorm = 0; for (size_t i = 0; i < N; i++) gnorm += g[i] * g[i]; gnorm = sqrt(gnorm);
            st->inner++;
        }
        free(x); free(g); free(xn); free(gn); free(d); free(Sm); free(Ym);
        /* ---- multiplier and penalty ---- */
        double rn = 0; for (int i = 0; i < m; i++) rn += r[i] * r[i]; rn = sqrt(rn);
        for (int i = 0; i < m; i++) y[i] -= rho * r[i];
        /* ---- KKT: Z = C - A*y, its most negative eigenvalues per block; rank update ---- */
        double lmin_all = 0, rd2 = 0; int rankup = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k]; FB *f2 = &F[k]; BB *q = &Q[k]; const int n = B->n;
            if (B->type == BLK_LP) {
                fb_ATy(B, y, f2->W); for (int i = 0; i < n; i++) f2->W[i] = -f2->W[i]; fb_addC(B, 1.0, f2->W);   /* Z */
                for (int i = 0; i < n; i++) if (!(f2->freev && f2->freev[i]) && f2->W[i] < lmin_all) lmin_all = f2->W[i];
                for (int i = 0; i < n; i++) if (f2->freev && f2->freev[i]) rd2 += f2->W[i] * f2->W[i];
                continue;
            }
            fb_ATy(B, y, f2->V); for (size_t i = 0; i < (size_t)n * n; i++) f2->V[i] = -f2->V[i]; fb_addC(B, 1.0, f2->V);
            int info = 0; BL(dsyevd_)("V", "L", &n, f2->V, &n, f2->ev, f2->work, &f2->lwork, f2->iwork, &f2->liwork, &info);
            f2->t_eig += 0; f2->n_full++;
            if (f2->ev[0] < lmin_all) lmin_all = f2->ev[0];
            /* escape directions: eigenvectors of the negative eigenvalues (up to 4), appended with a small scale */
            int add = 0; for (int i = 0; i < n && add < 4; i++) if (f2->ev[i] < -par->fom_bm_negtol) add++;
            if (add > 0 && q->r + add <= n) {
                if (q->r + add > q->rmax) { const int nr = q->r + add + 10 > n ? n : q->r + add + 10; q->R = realloc(q->R, sizeof(double) * (size_t)n * nr); q->SR = realloc(q->SR, sizeof(double) * (size_t)n * nr); q->G = realloc(q->G, sizeof(double) * (size_t)n * nr); q->rmax = nr; }
                double rnorm = 0; for (size_t i = 0; i < (size_t)n * q->r; i++) rnorm += q->R[i] * q->R[i]; rnorm = sqrt(rnorm / q->r);
                for (int a = 0; a < add; a++) { const double sq = fmax(1e-3 * rnorm, 1e-8); for (int i = 0; i < n; i++) q->R[i + (size_t)(q->r + a) * n] = sq * f2->V[i + (size_t)a * n]; }
                q->r += add; rankup += add;
            }
        }
        st->rankup += rankup;
        /* ---- residuals in the original scaling ---- */
        double rpo = 0; for (int i = 0; i < m; i++) { const double v = r[i] * P->du[i]; rpo += v * v; } rpo = sqrt(rpo);
        pinf = rpo / (1 + P->normb2);
        dinf = P->cs * sqrt(rd2 + lmin_all * lmin_all) / (1 + P->normC2);
        pobj = cx; dobj = 0; for (int i = 0; i < m; i++) dobj += P->b[i] * y[i];
        const double po = sc * pobj, dob = sc * dobj, den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        gap = fabs(po - dob) / den;
        const double res = fmax(fmax(pinf, dinf), gap);
        st->outer++; (*iters)++;
        if (g_ftrace) { fprintf(g_ftrace, "%d %.3f %.3e %.3e %.3e %.3e %.10e %.10e\n", *iters, wtime() - t0, pinf, dinf, gap, rho, po + P->obj_off, dob + P->obj_off); fflush(g_ftrace); }
        if (verbose > 1) { int rt = 0; for (int k = 0; k < nb; k++) rt += Q[k].r; printf("BM%3d %14.7e %14.7e %8.1e %8.1e %8.1e %8.2g %7.1fs  (%d L-BFGS, |grad| %.1e, ranks %d, +%d)\n", outer, po + P->obj_off, dob + P->obj_off, pinf, dinf, gap, rho, wtime() - t0, inner, gnorm, rt, rankup); fflush(stdout); }
        if (res <= tol) { status = ST_OPTIMAL; break; }
        if (brisk_time_up(par)) { status = ST_TIME; break; }
        if (gnorm <= eps_in && rn > 0.25 * rn_prev) rho = fmin(rho * 2.0, 1e8);   /* only when the inner problem was solved */
        rn_prev = rn;
    }
    /* X = R R' back into F */
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; FB *f2 = &F[k]; BB *q = &Q[k]; const int n = B->n;
        if (B->type == BLK_LP) { for (int i = 0; i < n; i++) f2->X[i] = (f2->freev && f2->freev[i]) ? q->R[i] : q->R[i] * q->R[i]; continue; }
        memset(f2->X, 0, sizeof(double) * (size_t)n * n);
        BL(dsyrk_)("L", "N", &n, &q->r, &DONE, q->R, &n, &DZERO, f2->X, &n);
        for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) f2->X[i + (size_t)j * n] = f2->X[j + (size_t)i * n];
    }
    *pinf_o = pinf; *dinf_o = dinf; *gap_o = gap; *pobj_o = pobj; *dobj_o = dobj;
    for (int k = 0; k < nb; k++) { bb_free(&Q[k]); free(xs[k]); }
    free(Q); free(xs); free(r); free(yh);
    return status;
}

/* ---- the engine ---------------------------------------------------------------------- */

/* ---- Anderson acceleration (type II) on a generic fixed-point sequence (4.35) -------------
 * Used by the Douglas-Rachford form of the splitting: the state is w = X/sigma + A*y - C (one
 * n x n array a block, half of (X, Z)), the map T is one iteration y-step -> W'. aa_apply gets
 * g = T(w_k) and returns the next w in place; a step whose fixed-point residual exceeds safe
 * times the last one is rejected: the previous plain step is restored and the memory cleared. */
typedef struct {
    size_t N; int mem, n, head, used, naccel, nrej; double safe;
    double *w, *g, *r, *rlast, *dR, *dG, *Gm, rprev, rsafe;
    double rv[AA_MAX];        /* 4.37: dR_c . r_last for the columns in memory (updated incrementally) */
    double reg;               /* relative Tikhonov regularization of the normal equations (1e-10) */
    int type;                 /* 4.39 (research, BRISK_AATYPE=1): type I (Broyden's first; SCS's default) */
    double *GR, gv[AA_MAX];   /* type I: GR[c mem + j] = dG_c . dR_j, gv[c] = dG_c . r_last */
    int n1, n1fail;
} AAcc;
static void aa_init(AAcc *A, size_t N, int mem, double safe) {
    memset(A, 0, sizeof *A); A->N = N; A->mem = mem; A->safe = safe; A->rprev = -1; A->rsafe = -1; A->reg = 1e-10;
    A->w = fo_malloc(sizeof(double) * N); A->g = fo_malloc(sizeof(double) * N); A->r = fo_malloc(sizeof(double) * N); A->rlast = fo_malloc(sizeof(double) * N);
    A->dR = fo_malloc(sizeof(double) * N * mem); A->dG = fo_malloc(sizeof(double) * N * mem); A->Gm = calloc((size_t)mem * mem, sizeof(double));
    A->type = getenv("BRISK_AATYPE") ? atoi(getenv("BRISK_AATYPE")) : 2; A->GR = calloc((size_t)mem * mem, sizeof(double));
}
static void aa_free(AAcc *A) { free(A->w); free(A->g); free(A->r); free(A->rlast); free(A->dR); free(A->dG); free(A->Gm); free(A->GR); }
static void aa_reset(AAcc *A) { A->n = 0; A->head = 0; A->used = 0; A->rprev = -1; A->rsafe = -1; }
/* gw: in = g_k = T(w_k) (w_k is A->w when A->rprev >= 0, else the first point); out = w_{k+1}.
 * returns 1 when the previous accelerated step was rejected (gw then holds the restored plain step).
 * 4.37: two cache-blocked parallel passes over the history instead of 3 + 2 mem: pass 1 forms
 * r = g - w, the new columns dG_h, dR_h and their dots with every column and with r (the other
 * dots dR_c . r_k = dR_c . r_{k-1} + dR_c . dR_h come from the Gram row, as r_k = r_{k-1} +
 * dR_h); pass 2 forms w_{k+1} = g - dG gamma chunk by chunk (each history column read once). The
 * history is memory-bound (Example 8.1.3 d = 5, memory 25: 2 GB read an iteration): 0.52 -> 0.2 s. */
static int aa_apply(AAcc *A, double *gw, int it) {
    const size_t N = A->N; const int mem = A->mem;
    if (A->rprev < 0 && A->used == 0 && A->n == 0 && it <= 1) { memcpy(A->w, gw, sizeof(double) * N); return 0; }   /* first point: w_1 = g_0 */
    const int push = A->rprev >= 0, h = A->head, na = push ? (A->n < mem ? A->n + 1 : mem) : 0;
    double *dRh = A->dR + (size_t)h * N, *dGh = A->dG + (size_t)h * N;
    double rn2 = 0, dots[AA_MAX + 1]; for (int c = 0; c <= AA_MAX; c++) dots[c] = 0;
    const int t1 = A->type == 1;
    double d1[AA_MAX + 1], d2[AA_MAX + 1]; for (int c = 0; c <= AA_MAX; c++) d1[c] = d2[c] = 0;   /* type I: dG_c . dR_h, dR_c . dG_h (+ dG_h . r in d1[na]) */
    /* the history as an N x mem column-major matrix: the dots are one dgemv per thread on its rows
     * (BLAS streams the columns at the memory bandwidth; the loops reached a third of it) */
    #pragma omp parallel if (N > 200000)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const int tid = 0, nt = 1;
#endif
        const size_t r0 = N * tid / nt, r1 = N * (tid + 1) / nt; const int rows = (int)(r1 - r0);
        double lr = 0, ld[AA_MAX + 1], l1[AA_MAX + 1], l2[AA_MAX + 1];
        for (size_t i = r0; i < r1; i++) { const double r = gw[i] - A->w[i]; A->r[i] = r; lr += r * r; }
        if (push && rows > 0) {
            for (size_t i = r0; i < r1; i++) { dGh[i] = gw[i] - A->g[i]; dRh[i] = A->r[i] - A->rlast[i]; }
            const int ldn = (int)N;
            BL(dgemv_)("T", &rows, &na, &DONE, A->dR + r0, &ldn, dRh + r0, &IONE, &DZERO, ld, &IONE);
            ld[na] = BL(ddot_)(&rows, dRh + r0, &IONE, A->r + r0, &IONE);
            if (t1) {
                BL(dgemv_)("T", &rows, &na, &DONE, A->dG + r0, &ldn, dRh + r0, &IONE, &DZERO, l1, &IONE);
                BL(dgemv_)("T", &rows, &na, &DONE, A->dR + r0, &ldn, dGh + r0, &IONE, &DZERO, l2, &IONE);
                l1[na] = BL(ddot_)(&rows, dGh + r0, &IONE, A->r + r0, &IONE);
            }
        } else for (int c = 0; c <= na; c++) ld[c] = 0;
        if (!(push && rows > 0 && t1)) for (int c = 0; c <= na; c++) l1[c] = l2[c] = 0;
        #pragma omp critical (aa_dots)
        { rn2 += lr; for (int c = 0; c <= na; c++) { dots[c] += ld[c]; d1[c] += l1[c]; d2[c] += l2[c]; } }
    }
    const double rn = sqrt(rn2);
    if (A->used && A->rsafe >= 0 && rn > A->rsafe) {
        A->nrej++; memcpy(gw, A->g, sizeof(double) * N); memcpy(A->w, gw, sizeof(double) * N); aa_reset(A);
        return 1;
    }
    if (push) {
        for (int c = 0; c < na; c++) { A->Gm[h * mem + c] = A->Gm[c * mem + h] = dots[c]; if (c != h) A->rv[c] += dots[c]; }
        A->rv[h] = dots[na];
        if (t1) { for (int c = 0; c < na; c++) { A->GR[c * mem + h] = d1[c]; A->GR[h * mem + c] = d2[c]; if (c != h) A->gv[c] += d1[c]; } A->gv[h] = d1[na]; }
        A->head = (A->head + 1) % mem; if (A->n < mem) A->n++;
    }
    { double *t = A->rlast; A->rlast = A->r; A->r = t; }          /* r_last = r_k (a swap, no copy) */
    A->rprev = rn; A->rsafe = A->safe * rn; A->used = 0;
    double gam[AA_MAX]; int solved = 0; const int n = A->n;
    if (n >= 1 && it >= 3) {
        double Mx[AA_MAX * AA_MAX];
        for (int a = 0; a < n; a++) for (int c = 0; c < n; c++) Mx[a * AA_MAX + c] = A->Gm[a * mem + c];
        double tr = 0; for (int a = 0; a < n; a++) tr += Mx[a * AA_MAX + a];
        for (int a = 0; a < n; a++) Mx[a * AA_MAX + a] += A->reg * tr + 1e-300;
        int ok = 1;
        for (int j = 0; j < n && ok; j++) {
            double d = Mx[j * AA_MAX + j]; for (int q = 0; q < j; q++) d -= Mx[j * AA_MAX + q] * Mx[j * AA_MAX + q];
            if (d <= 0) { ok = 0; break; }
            d = sqrt(d); Mx[j * AA_MAX + j] = d;
            for (int i = j + 1; i < n; i++) { double v = Mx[i * AA_MAX + j]; for (int q = 0; q < j; q++) v -= Mx[i * AA_MAX + q] * Mx[j * AA_MAX + q]; Mx[i * AA_MAX + j] = v / d; }
        }
        if (ok) {
            for (int i = 0; i < n; i++) { double v = A->rv[i]; for (int q = 0; q < i; q++) v -= Mx[i * AA_MAX + q] * gam[q]; gam[i] = v / Mx[i * AA_MAX + i]; }
            for (int i = n - 1; i >= 0; i--) { double v = gam[i]; for (int q = i + 1; q < n; q++) v -= Mx[q * AA_MAX + i] * gam[q]; gam[i] = v / Mx[i * AA_MAX + i]; }
            solved = 1; A->used = 1; A->naccel++;
        }
        if (ok && t1) {
            /* type I: (dW' dR) gamma = dW' r with dW = dG - dR (Gaussian elimination, partial
             * pivoting, the same relative regularization); the type II gamma when singular */
            double M1[AA_MAX * AA_MAX], g1[AA_MAX]; int piv_ok = 1;
            for (int a = 0; a < n; a++) { for (int c = 0; c < n; c++) M1[a * AA_MAX + c] = A->GR[a * mem + c] - A->Gm[a * mem + c]; g1[a] = A->gv[a] - A->rv[a]; }
            double trd = 0; for (int a = 0; a < n; a++) trd += fabs(M1[a * AA_MAX + a]);
            for (int a = 0; a < n; a++) M1[a * AA_MAX + a] += A->reg * trd;
            for (int j = 0; j < n && piv_ok; j++) {
                int pr = j; for (int i = j + 1; i < n; i++) if (fabs(M1[i * AA_MAX + j]) > fabs(M1[pr * AA_MAX + j])) pr = i;
                if (fabs(M1[pr * AA_MAX + j]) < 1e-14 * fmax(trd, 1e-300)) { piv_ok = 0; break; }
                if (pr != j) { for (int c = 0; c < n; c++) { const double t = M1[j * AA_MAX + c]; M1[j * AA_MAX + c] = M1[pr * AA_MAX + c]; M1[pr * AA_MAX + c] = t; } const double t = g1[j]; g1[j] = g1[pr]; g1[pr] = t; }
                for (int i = j + 1; i < n; i++) { const double f = M1[i * AA_MAX + j] / M1[j * AA_MAX + j]; for (int c = j; c < n; c++) M1[i * AA_MAX + c] -= f * M1[j * AA_MAX + c]; g1[i] -= f * g1[j]; }
            }
            if (piv_ok) { for (int i = n - 1; i >= 0; i--) { double v = g1[i]; for (int c = i + 1; c < n; c++) v -= M1[i * AA_MAX + c] * gam[c]; gam[i] = v / M1[i * AA_MAX + i]; } A->n1++; }
            else A->n1fail++;
        }
    }
    /* pass 2: g = g_k, w_{k+1} = g_k - dG gamma (one dgemv per thread on its rows) */
    #pragma omp parallel if (N > 200000)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const int tid = 0, nt = 1;
#endif
        const size_t r0 = N * tid / nt, r1 = N * (tid + 1) / nt; const int rows = (int)(r1 - r0);
        if (rows > 0) {
            memcpy(A->g + r0, gw + r0, sizeof(double) * rows);
            if (solved) { const int ldn = (int)N; const double mone = -1.0; BL(dgemv_)("N", &rows, &n, &mone, A->dG + r0, &ldn, gam, &IONE, &DONE, gw + r0, &IONE); }
            memcpy(A->w + r0, gw + r0, sizeof(double) * rows);
        }
    }
    return 0;
}

/* 4.39: the acceleration on the packed upper triangles of the symmetric blocks (diagonal, then
 * the strict upper part times sqrt 2, so that plain dots are the Frobenius products): the same
 * iteration in exact arithmetic with half the history (memory traffic and budget) */
static size_t pk_len(const Problem *P) { size_t N = 0; for (int k = 0; k < P->nblk; k++) { const int n = P->blk[k].n; N += P->blk[k].type == BLK_LP ? (size_t)n : (size_t)n * (n + 1) / 2; } return N; }
static void pk_pack(const Problem *P, double *const *W, const size_t *po, double *v) {
    const double r2 = sqrt(2.0);
    #pragma omp parallel for schedule(dynamic) if (P->nblk > 1)
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; const int n = B->n; double *o = v + po[k]; const double *w = W[k];
        if (B->type == BLK_LP) { memcpy(o, w, sizeof(double) * n); continue; }
        for (int j = 0; j < n; j++) o[j] = w[j + (size_t)j * n];
        o += n;
        for (int j = 1; j < n; j++) { const double *c = w + (size_t)j * n; for (int i = 0; i < j; i++) o[i] = r2 * c[i]; o += j; }
    }
}
static void pk_unpack(const Problem *P, double *const *W, const size_t *po, const double *v) {
    const double ir2 = 1.0 / sqrt(2.0);
    #pragma omp parallel for schedule(dynamic) if (P->nblk > 1)
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; const int n = B->n; const double *o = v + po[k]; double *w = W[k];
        if (B->type == BLK_LP) { memcpy(w, o, sizeof(double) * n); continue; }
        for (int j = 0; j < n; j++) w[j + (size_t)j * n] = o[j];
        o += n;
        for (int j = 1; j < n; j++) { double *c = w + (size_t)j * n; for (int i = 0; i < j; i++) { const double a = ir2 * o[i]; c[i] = a; w[j + (size_t)i * n] = a; } o += j; }
    }
}

/* 4.39: the dual residual with the best Z for the current y, min over Z in K of |A*y + Z - C|,
 * = the norm of the negative part of C - A*y (exact on free-pair slots: there Z = 0); T[k] holds
 * A*y - C of every block (the projection loop's AyC); V[k]: n x n scratch per SDP block */
static double fom_dinf_best(Problem *P, FB *F, const int *bord) {
    const int nb = P->nblk; double s2 = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:s2) if (nb > 1)
    for (int kk = 0; kk < nb; kk++) {
        const int k = bord[kk]; const Block *B = &P->blk[k]; FB *f = &F[k]; const int n = B->n;
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) { const double v = -f->T[i]; if (f->freev && f->freev[i]) s2 += v * v; else if (v < 0) s2 += v * v; }
            continue;
        }
        const size_t len = (size_t)n * n;
        for (size_t i = 0; i < len; i++) f->V[i] = -f->T[i];
        int info = 0;
        BL(dsyevd_)("N", "L", &n, f->V, &n, f->ev, f->work, &f->lwork, f->iwork, &f->liwork, &info);
        if (info != 0) { s2 += 1e300; continue; }
        for (int i = 0; i < n && f->ev[i] < 0; i++) s2 += f->ev[i] * f->ev[i];
    }
    return sqrt(s2);
}

/* 4.39: residuals of a point (X, Z, y) given as block arrays (X, Z full per block): the same
 * measures as the splitting loop; T: scratch block arrays, r: scratch m-vector */
static double fom_point_res(Problem *P, double **Xb, double **Zb, const double *y, double **T, double *r,
                            double *pinf_o, double *dinf_o, double *gap_o, double *pobj_o, double *dobj_o) {
    const int m = P->m, nb = P->nblk; const double sc = P->bs * P->cs;
    memcpy(r, P->b, sizeof(double) * m);
    fop_A(g_fop, Xb, r, -1.0);
    double rpo = 0; for (int i = 0; i < m; i++) { const double v = r[i] * P->du[i]; rpo += v * v; } rpo = sqrt(rpo);
    fop_At(g_fop, P, y, T);
    double rd2 = 0, pobj = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; const size_t len = fbsz(B);
        fb_addC(B, -1.0, T[k]);
        for (size_t i = 0; i < len; i++) { const double v = T[k][i] + Zb[k][i]; rd2 += v * v; }
        pobj += fb_Cdot(B, Xb[k]);
    }
    double dobj = 0; for (int i = 0; i < m; i++) dobj += P->b[i] * y[i];
    const double po = sc * pobj, dob = sc * dobj, den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
    *pinf_o = rpo / (1 + P->normb2); *dinf_o = P->cs * sqrt(rd2) / (1 + P->normC2); *gap_o = fabs(po - dob) / den;
    *pobj_o = pobj; *dobj_o = dobj;
    return fmax(fmax(*pinf_o, *dinf_o), *gap_o);
}

int fom_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout) {
    const double t0 = wtime();
    memset(R, 0, sizeof(*R));
    const int m = P->m, nb = P->nblk, verbose = par->verbose;
    const double sc = P->bs * P->cs;
    const int halpern = par->fom_halpern;
    { const char *e = getenv("BRISK_FOMPARTIAL"); if (e) g_fom_partial = atoi(e); }
    { const char *e = getenv("BRISK_FOMDCPART"); if (e) g_fom_dcpart = atoi(e); }
    { const char *e = getenv("BRISK_FOMRR"); if (e) g_fom_rr = atoi(e); e = getenv("BRISK_RRREFRESH"); if (e) g_rr_refresh = atoi(e); e = getenv("BRISK_FOMSRANGE"); if (e) g_fom_srange = atoi(e); }
    const double rr_c = getenv("BRISK_RRTOL") ? atof(getenv("BRISK_RRTOL")) : 1e-2;
    FB *F = calloc(nb, sizeof(FB));
    double sumn2 = 0; int maxn = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; FB *f = &F[k];
        f->n = B->n; f->type = B->type;
        const size_t len = fbsz(B);
        f->X = calloc(len, sizeof(double)); f->Z = calloc(len, sizeof(double)); f->W = fo_malloc(sizeof(double) * len);
        f->T = fo_malloc(sizeof(double) * len); f->P = fo_malloc(sizeof(double) * len);
        if (halpern) { f->Xa = calloc(len, sizeof(double)); f->Za = calloc(len, sizeof(double)); }
        if (B->type == BLK_SDP) {
            f->V = fo_malloc(sizeof(double) * len); f->ev = fo_malloc(sizeof(double) * B->n);
            const int n = B->n; double wq; int iq, info;
            const int lm1 = -1;
            BL(dsyevd_)("V", "L", &n, f->V, &n, f->ev, &wq, &lm1, &iq, &lm1, &info);
            f->lwork = (int)wq + 1; f->liwork = iq;
            { double wq2; int iq2, mf, il = 1, iu = 1; const double at = 0;
              f->Vp = fo_malloc(sizeof(double) * len); f->isup = fo_malloc(sizeof(int) * 2 * (size_t)n);
              BL(dsyevr_)("V", "I", "L", &n, f->V, &n, &DZERO, &DZERO, &il, &iu, &at, &mf, f->ev, f->Vp, &n, f->isup, &wq2, &lm1, &iq2, &lm1, &info);
              if ((int)wq2 + 1 > f->lwork) f->lwork = (int)wq2 + 1;
              if (iq2 > f->liwork) f->liwork = iq2; }
            f->work = fo_malloc(sizeof(double) * f->lwork); f->iwork = fo_malloc(sizeof(int) * f->liwork);
            if (g_fom_dcpart && n >= 64) {           /* 4.39: dsytrd + dstedc + dormtr on the small side */
                double q1, q2; int e1 = 0, kq = n; const int lm = -1;
                f->dcdd = fo_malloc(sizeof(double) * n); f->dcde = fo_malloc(sizeof(double) * n); f->dcdtau = fo_malloc(sizeof(double) * n);
                BL(dsytrd_)("L", &n, f->V, &n, f->dcdd, f->dcde, f->dcdtau, &q1, &lm, &e1);
                BL(dormtr_)("L", "L", "N", &n, &kq, f->V, &n, f->dcdtau, f->V, &n, &q2, &lm, &e1);
                int lw = 1 + 4 * n + n * n; if ((int)q1 + 1 > lw) lw = (int)q1 + 1; if ((int)q2 + 1 > lw) lw = (int)q2 + 1;
                f->dcdlw = lw; f->dcdw = fo_malloc(sizeof(double) * (size_t)lw); f->dcdliw = 3 + 5 * n; f->dcdiw = fo_malloc(sizeof(int) * f->dcdliw);
            }
            if (par->fom_single > 0 && n >= 64) {   /* 4.37: single-precision workspace */
                float wqf; int iqf; f->Vf = fo_malloc(sizeof(float) * len); f->evf = fo_malloc(sizeof(float) * n);
                BL(ssyevd_)("V", "L", &n, f->Vf, &n, f->evf, &wqf, &lm1, &iqf, &lm1, &info);
                f->lworkf = (int)wqf + 1; f->liworkf = iqf; f->workf = fo_malloc(sizeof(float) * f->lworkf); f->iworkf = fo_malloc(sizeof(int) * f->liworkf);
                if (g_fom_dcpart) {                  /* 4.39: ssytrd + sstedc + sormtr on the small side */
                    float q1, q2; int e1 = 0, kq = n; const int lm = -1;
                    f->dcd = fo_malloc(sizeof(float) * n); f->dce = fo_malloc(sizeof(float) * n); f->dctau = fo_malloc(sizeof(float) * n);
                    BL(ssytrd_)("L", &n, f->Vf, &n, f->dcd, f->dce, f->dctau, &q1, &lm, &e1);
                    BL(sormtr_)("L", "L", "N", &n, &kq, f->Vf, &n, f->dctau, f->Vf, &n, &q2, &lm, &e1);
                    int lw = 1 + 4 * n + n * n; if ((int)q1 + 1 > lw) lw = (int)q1 + 1; if ((int)q2 + 1 > lw) lw = (int)q2 + 1;
                    f->dclw = lw; f->dcw = fo_malloc(sizeof(float) * (size_t)lw); f->dcliw = 3 + 5 * n; f->dciw = fo_malloc(sizeof(int) * f->dcliw);
                    if (!f->srZ) f->srZ = fo_malloc(sizeof(float) * (size_t)n * n);
                }
                if (g_fom_srange && n >= 120) {      /* 4.39: ssyevr workspace, vectors up to n/3 */
                    float wr; int ir, mf2, il2 = 1, iu2 = 1; const float vl = 0, vu = 1, at = 0;
                    if (!f->srZ) { f->srZ = fo_malloc(sizeof(float) * (size_t)n * n); }
                    f->srsup = fo_malloc(sizeof(int) * 2 * (size_t)n);   /* RANGE = V: up to n vectors */
                    BL(ssyevr_)("V", "V", "L", &n, f->Vf, &n, &vl, &vu, &il2, &iu2, &at, &mf2, f->evf, f->srZ, &n, f->srsup, &wr, &lm1, &ir, &lm1, &info);
                    f->srlw = (int)wr + 1; f->srliw = ir; f->srwork = fo_malloc(sizeof(float) * f->srlw); f->sriwork = fo_malloc(sizeof(int) * f->srliw);
                }
            }
            sumn2 += (double)n * n; if (n > maxn) maxn = n;
        }
    }
    /* split free pairs: identity projection on both members */
    {
        FreePair *fp = NULL; const int nf = getenv("BRISK_FOMNOFREE") ? 0 : free_pairs_detect(P, &fp);
        for (int q = 0; q < nf; q++) {
            FB *f = &F[fp[q].blk];
            if (!f->freev) f->freev = calloc(f->n, 1);
            f->freev[fp[q].ip] = 1; f->freev[fp[q].im] = 1;
        }
        free(fp);
        if (verbose && nf) printf("first-order: %d split free pair(s) handled as free variables\n", nf);
    }
    double t_gram = 0;
    Gram *G = fom_gram_build(P, verbose, &t_gram);
    {   const double tf = wtime(); g_fop = fop_build(P);
        if (verbose) printf("first-order: operators by position: %zu positions, %zu entries, %.2fs\n", g_fop->np, g_fop->pp[g_fop->np], wtime() - tf); }
    double **Xa = fo_malloc(sizeof(double *) * (nb + 1)), **Wa = fo_malloc(sizeof(double *) * (nb + 1)), **Ta = fo_malloc(sizeof(double *) * (nb + 1));
    for (int k = 0; k < nb; k++) { Xa[k] = F[k].X; Wa[k] = F[k].W; Ta[k] = F[k].T; }
    double *Wall = NULL;          /* 4.37: contiguous W of all blocks (Douglas-Rachford acceleration) */
    if (getenv("BRISK_FOMGRAMCHK")) {
        /* accuracy of the Gram solve: x random, r = A A* x, solve, compare; and a few inverse
         * iterations for the smallest eigenvalue of A A* */
        double *x = fo_malloc(sizeof(double) * m), *r = fo_malloc(sizeof(double) * m);
        unsigned long long sd = 12345;
        for (int i = 0; i < m; i++) { sd = sd * 6364136223846793005ull + 1442695040888963407ull; x[i] = (double)(sd >> 11) / 9007199254740992.0 - 0.5; }
        for (int rep = 0; rep < 8; rep++) {
            memset(r, 0, sizeof(double) * m);
            for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; double *Fk = fo_malloc(sizeof(double) * fbsz(B)); fb_ATy(B, x, Fk); fb_Aop(B, Fk, r, 1.0); free(Fk); }
            double nx = 0, nr = 0; for (int i = 0; i < m; i++) { nx += x[i] * x[i]; nr += r[i] * r[i]; }
            double *z = fo_malloc(sizeof(double) * m); memcpy(z, r, sizeof(double) * m);
            fom_gram_solve(G, z);
            double e = 0; for (int i = 0; i < m; i++) e += (z[i] - x[i]) * (z[i] - x[i]);
            printf("   [Gram check %d: |x| %.3e, |AA*x|/|x| %.3e (Rayleigh), relative solve error %.2e]\n", rep, sqrt(nx), sqrt(nr / nx), sqrt(e / nx));
            /* inverse iteration step: x <- solve(x) normalised */
            memcpy(z, x, sizeof(double) * m); fom_gram_solve(G, z);
            double nz = 0; for (int i = 0; i < m; i++) nz += z[i] * z[i]; nz = sqrt(nz);
            for (int i = 0; i < m; i++) x[i] = z[i] / nz;
            free(z);
        }
        free(x); free(r);
    }
    double *y = calloc(m + 1, sizeof(double)), *rhs = fo_malloc(sizeof(double) * (m + 1)), *rp = fo_malloc(sizeof(double) * (m + 1));
    int warm = 0;
    if (par->fom_X0 || par->fom_y0) {
        /* 4.39: a starting point (-fomstart-x/-y, mapped by main.c): X as given, y as given, Z the
         * best one for y, Pi(C - A*y); the first y-step recomputes y from (X, Z). The penalty
         * starts at the ratio |X| / |Z| (rule 2's target) unless fixed */
        if (par->fom_X0) for (int k = 0; k < nb; k++) memcpy(F[k].X, par->fom_X0[k], sizeof(double) * fbsz(&P->blk[k]));
        if (par->fom_y0) memcpy(y, par->fom_y0, sizeof(double) * m);
        for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
            fb_ATy(B, y, f->T); fb_addC(B, -1.0, f->T);            /* A*y - C */
            for (size_t i = 0; i < len; i++) f->W[i] = f->T[i];    /* Z = Pi(-(A*y - C)) */
            fb_project(f, B, f->W, f->P, f->Z); }
        warm = 1;
    }
    double *ya = calloc(m + 1, sizeof(double));
    double nb2 = 0; for (int i = 0; i < m; i++) nb2 += P->b[i] * P->b[i]; nb2 = sqrt(nb2);
    double nC2 = 0; for (int k = 0; k < nb; k++) for (int q = 0; q < P->blk[k].C.ef; q++) nC2 += P->blk[k].C.fv[q] * P->blk[k].C.fv[q]; nC2 = sqrt(nC2);
    double sigma = par->fom_sigma > 0 ? par->fom_sigma : (par->fom_sigma0 > 0 ? par->fom_sigma0 : 1.0);
    if (warm && par->fom_sigma <= 0 && par->fom_sigma0 == 1.0) {   /* (an explicit -fomsigma0 is kept: e.g. the last sigma of the run continued) */
        double nx = 0, nz = 0; for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); for (size_t i = 0; i < len; i++) { nx += F[k].X[i] * F[k].X[i]; nz += F[k].Z[i] * F[k].Z[i]; } }
        if (nx > 0 && nz > 0) sigma = sqrt(nx / nz);
        if (verbose) printf("first-order: started from the given point, sigma %.3g\n", sigma);
    }
    const int maxit = par->fom_maxit > 0 ? par->fom_maxit : 100000;
    const double tol = par->fom_tol > 0 ? par->fom_tol : fmax(par->tol, 1e-6);
    int it, status = ST_MAXIT, kh = 0;         /* kh: Halpern counter since the last restart */
    double pinf = 1, dinf = 1, gap = 1, pobj = 0, dobj = 0, best = 1e300, res_anchor = 1e300, res_last = 1e300;
    double t_solve = 0;
    int nsig = 0, nrestart = 0;
    if (verbose) printf("first-order: %s, sigma %.3g, tol %.1e, %d blocks (max n %d), m = %d\n", halpern ? "Halpern Peaceman-Rachford" : "ADMM (step 1.618)", sigma, tol, nb, maxn, m);
    if (verbose > 1) printf("   it        pobj          dobj        pinf     dinf     gap      sigma   time\n");
    double tau = halpern ? 1.0 : 1.618;
    { const char *e = getenv("BRISK_FOMTAU"); if (e && !halpern) tau = atof(e); }
    /* Anderson acceleration (type II, memory aa_mem) on the fixed-point map (X, Z) -> T(X, Z)
     * of the splitting, with a safeguard: an accelerated point whose fixed-point residual
     * is larger than the last plain one is rejected and the plain iterate restored. */
    size_t N2 = 0; for (int k = 0; k < nb; k++) N2 += 2 * fbsz(&P->blk[k]);
    int aa_mem = halpern ? 0 : par->fom_aa;
    /* 4.35: the Douglas-Rachford form: acceleration on w = X/sigma + A*y - C (half the state of
     * (X, Z)), the relaxation step then 1 */
    const int aadr = par->fom_aadr && aa_mem > 0 && !halpern;
    size_t Nw = 0; for (int k = 0; k < nb; k++) Nw += fbsz(&P->blk[k]);
    AAcc AD; memset(&AD, 0, sizeof AD);
    const int aapack = aadr && !(getenv("BRISK_AAPACK") && atoi(getenv("BRISK_AAPACK")) == 0);   /* 4.39 */
    size_t Nv = aapack ? pk_len(P) : Nw; double *Wp = NULL; size_t *pko = NULL;
    if (aapack) { Wp = fo_malloc(sizeof(double) * (Nv + 1)); pko = fo_malloc(sizeof(size_t) * (nb + 1)); size_t o = 0; for (int k = 0; k < nb; k++) { pko[k] = o; o += P->blk[k].type == BLK_LP ? (size_t)P->blk[k].n : (size_t)P->blk[k].n * (P->blk[k].n + 1) / 2; } }
    if (aadr) {
        tau = 1.0;
        int mem = aa_mem > AA_MAX ? AA_MAX : aa_mem;
        { double budget = fmax(1.5e9, 0.3 * brisk_mem_limit());   /* 4.37: 30% of the memory limit (was 1.5 GB) */
          { const char *e = getenv("BRISK_AABUDGET"); if (e) budget = atof(e) * 1e9; } const int fit = (int)(budget / (8.0 * Nv) - 4) / 2; if (fit < mem) mem = fit; if (mem < 2) mem = 2; }
        aa_init(&AD, Nv, mem, par->fom_aasafe);
        { const char *e = getenv("BRISK_AAREG"); if (e) AD.reg = atof(e); }
        aa_mem = 0;                      /* the (X, Z) acceleration is off */
        /* 4.37: the blocks' W arrays as one contiguous vector (the acceleration works on it in place) */
        Wall = fo_malloc(sizeof(double) * (Nw + 1));
        { size_t o = 0; for (int k = 0; k < nb; k++) { free(F[k].W); F[k].W = Wall + o; Wa[k] = F[k].W; o += fbsz(&P->blk[k]); } }
    }
    if (aa_mem > AA_MAX) aa_mem = AA_MAX;
    if (aa_mem > 0) { double budget = 1.5e9; { const char *e = getenv("BRISK_AABUDGET"); if (e) budget = atof(e) * 1e9; } const int fit = (int)(budget / (8.0 * N2) - 4) / 2; if (fit < aa_mem) aa_mem = fit; if (aa_mem < 2) aa_mem = 0; }
    double *aa_w = NULL, *aa_g = NULL, *aa_r = NULL, *aa_rlast = NULL; double *aa_dR = NULL, *aa_dG = NULL;   /* 4.35: memory up to AA_MAX (single-precision history was tried: it changes the extrapolation and loses) */
    int aa_n = 0, aa_head = 0, aa_used = 0, aa_naccel = 0, aa_nrej = 0;
    double aa_Gm[AA_MAX * AA_MAX]; memset(aa_Gm, 0, sizeof aa_Gm);
    double aa_rprev = -1, aa_rsafe = -1;
    if (aa_mem > 0) {
        aa_w = fo_malloc(sizeof(double) * N2); aa_g = fo_malloc(sizeof(double) * N2); aa_r = fo_malloc(sizeof(double) * N2); aa_rlast = fo_malloc(sizeof(double) * N2);
        aa_dR = fo_malloc(sizeof(double) * N2 * aa_mem); aa_dG = fo_malloc(sizeof(double) * N2 * aa_mem);
    }
    int ssn_block_until = 0, nresume = 0;   /* 4.35: phase II is not re-entered before this iteration after a stall */
    it = 1;
    /* 4.37: the blocks in decreasing size for the parallel loops (largest first: the dynamic
     * schedule then balances the threads) */
    int *bord = fo_malloc(sizeof(int) * (nb + 1));
    { for (int k = 0; k < nb; k++) bord[k] = k;
      for (int a = 1; a < nb; a++) { const int t = bord[a]; int q = a - 1; while (q >= 0 && P->blk[bord[q]].n < P->blk[t].n) { bord[q + 1] = bord[q]; q--; } bord[q + 1] = t; } }
    int single_off = par->fom_single <= 0;
    double prof[8] = { 0 };
    int sig_cur = par->fom_sigint, sig_next = par->fom_sigint; const int sig_double = getenv("BRISK_SIGDOUBLE") ? atoi(getenv("BRISK_SIGDOUBLE")) : 1;   /* 4.37: default on */
    const int aa_keep = getenv("BRISK_AAKEEP") ? atoi(getenv("BRISK_AAKEEP")) : 0;
    double sig_res_last = 1e300, sb_lp = 0, sb_ld = 0; int sb_n = 0;
    const int sig_dir = getenv("BRISK_SIGDIR") ? atoi(getenv("BRISK_SIGDIR")) : 0;   /* 1 both directions, 2 decreases only */   /* 4.39 (research, off): the direction filter below; it held gpp250-1's useful increases (311 -> 676 iterations) */   /* 4.39: log residuals since the last penalty check */
    const int ssn_stallrule = getenv("BRISK_SSNSTALLRULE") ? atoi(getenv("BRISK_SSNSTALLRULE")) : 1;
    const int bhist_n = maxit + 2; double *bhist = ssn_stallrule ? fo_malloc(sizeof(double) * bhist_n) : NULL, *thist = ssn_stallrule ? fo_malloc(sizeof(double) * bhist_n) : NULL;
    int ssn_split0 = 1;           /* the iteration where the current splitting run began */
    double ssn_reentry = 1e300, ssn_entry_res = 1e300;
    double *sr_X = NULL, *sr_Z = NULL; size_t *sr_off = NULL;
    const double x_relax = getenv("BRISK_FOMRELAX") ? atof(getenv("BRISK_FOMRELAX")) : 1.0;
    const int x_mom = getenv("BRISK_FOMMOM") ? atoi(getenv("BRISK_FOMMOM")) : 0, x_every = getenv("BRISK_AAEVERY") ? atoi(getenv("BRISK_AAEVERY")) : 1;
    double mom_t = 1, mom_r = 1e300, mom_ra = 1e300; int mom_nrest = 0;
    FILE *ftrace = getenv("BRISK_FOMTRACE") ? fopen(getenv("BRISK_FOMTRACE"), "w") : NULL; g_ftrace = ftrace;
    /* 4.39: the next right-hand side from this iteration's residual pass: b/sigma - A(X/sigma + Z - C)
     * = r_p/sigma - A(Z) + A(C) with r_p = b - A(X), A(Z) accumulated in the same pass (one pass over
     * the operator and none over the blocks saved an iteration) */
    const int fuse = aadr && !halpern && aa_mem == 0 && !(getenv("BRISK_FOMFUSE") && atoi(getenv("BRISK_FOMFUSE")) == 0);
    double *AZ = NULL, *AC = NULL, **Za = NULL; int rhs_ok = 0;
    if (fuse) {
        AZ = fo_malloc(sizeof(double) * (m + 1)); AC = calloc(m + 1, sizeof(double)); Za = fo_malloc(sizeof(double *) * (nb + 1));
        for (int k = 0; k < nb; k++) { Za[k] = F[k].Z; memset(F[k].T, 0, sizeof(double) * fbsz(&P->blk[k])); fb_addC(&P->blk[k], 1.0, F[k].T); }
        fop_A(g_fop, Ta, AC, 1.0);
    }
    /* 4.39: a moving average of the iterates (X, Z, y), weight min(1, c/it) (a window of about it/c),
     * measured every 25 iterations; it stops the run when it meets the tolerance and replaces the
     * last iterate at the end when it is better. The iterates themselves are unchanged. The
     * splitting's iterates spiral in (Example 8.1.3: the gap oscillates with a period of ~100-200
     * iterations), and their average is closer to the centre. BRISK_FOMAVG=0 off, =c the window */
    const double avg_c = getenv("BRISK_FOMAVG") ? atof(getenv("BRISK_FOMAVG")) : 20.0;
    const double avg_rs = getenv("BRISK_FOMAVGRS") ? atof(getenv("BRISK_FOMAVGRS")) : 0.0;   /* research: restart from the average when its residual is below this times the iterate's */
    int avg_nrs = 0;
    const int avg_on = aadr && !halpern && aa_mem == 0 && avg_c > 0;
    double *AvX = NULL, *AvZ = NULL, *Avy = NULL, *Avr = NULL, **AvXa = NULL, **AvZa = NULL;
    double avg_res = 1e300; int avg_n = 0, avg_hits = 0;
    const int bestz_on = !(getenv("BRISK_FOMBESTZ") && atoi(getenv("BRISK_FOMBESTZ")) == 0); int bz_next = 0, n_bestz = 0;   /* 4.39 */
    if (avg_on) {
        AvX = fo_malloc(sizeof(double) * (Nw + 1)); AvZ = fo_malloc(sizeof(double) * (Nw + 1)); Avy = fo_malloc(sizeof(double) * (m + 1)); Avr = fo_malloc(sizeof(double) * (m + 1));
        AvXa = fo_malloc(sizeof(double *) * (nb + 1)); AvZa = fo_malloc(sizeof(double *) * (nb + 1));
        size_t o = 0; for (int k = 0; k < nb; k++) { AvXa[k] = AvX + o; AvZa[k] = AvZ + o; o += fbsz(&P->blk[k]); }
    }
resume_splitting:
    for (; it <= maxit; it++) {
        if (!single_off && best <= par->fom_single) single_off = 1;       /* double precision from here on */
        g_rr_tol = fmax(1e-11, fmin(1e-5, rr_c * fmin(best, 1.0)));      /* 4.39: Rayleigh-Ritz accuracy follows the residual */
        g_fom_single_now = !single_off;
        if (aa_mem > 0) {    /* w_k = (X, Z) */
            size_t o = 0;
            for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(aa_w + o, F[k].X, sizeof(double) * len); o += len; memcpy(aa_w + o, F[k].Z, sizeof(double) * len); o += len; }
        }
        /* y = (A A*)^-1 [ b/sigma - A(X/sigma + Z - C) ] */
        double tp0 = wtime();
        if (rhs_ok) { for (int i = 0; i < m; i++) rhs[i] = rp[i] / sigma - AZ[i] + AC[i]; }
        else {
        for (int i = 0; i < m; i++) rhs[i] = P->b[i] / sigma;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
            for (size_t i = 0; i < len; i++) f->W[i] = f->X[i] / sigma + f->Z[i];
            fb_addC(B, -1.0, f->W);
        }
        fop_A(g_fop, Wa, rhs, -1.0);
        }
        prof[0] += wtime() - tp0; tp0 = wtime();
        const double ts = wtime();
        if (ENV_ON("BRISK_FOMDBG") && it <= 2) {
            double *r0 = fo_malloc(sizeof(double) * m), *gy = fo_malloc(sizeof(double) * m);
            memcpy(r0, rhs, sizeof(double) * m);
            fom_gram_solve(G, rhs);
            /* G y through A A*: gy = A(A* y) */
            memset(gy, 0, sizeof(double) * m);
            for (int k = 0; k < nb; k++) { fb_ATy(&P->blk[k], rhs, F[k].T); fb_Aop(&P->blk[k], F[k].T, gy, 1.0); }
            double e2 = 0, b2 = 0, y2 = 0; for (int i = 0; i < m; i++) { e2 += (gy[i] - r0[i]) * (gy[i] - r0[i]); b2 += r0[i] * r0[i]; y2 += rhs[i] * rhs[i]; }
            printf("   [gram solve check it %d: |A A* y - rhs| / |rhs| = %.2e, |y| %.2e, |rhs| %.2e]\n", it, sqrt(e2 / fmax(b2, 1e-300)), sqrt(y2), sqrt(b2));
            free(r0); free(gy);
        } else
        fom_gram_solve(G, rhs);
        memcpy(y, rhs, sizeof(double) * m);
        t_solve += wtime() - ts;
        prof[1] += wtime() - tp0; tp0 = wtime();
        /* W and the projections: Pos = Pi(W) = X_new/sigma, Neg = Pi(-W) = Z_new; the dual
         * residual A*y + Z_new - C in both variants */
        double rd2 = 0;
        pobj = 0;
        if (aadr) {
            /* W' = X/sigma + A*y - C for all blocks, the acceleration on the concatenation, then the projections */
            fop_At(g_fop, P, y, Ta);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
                double *AyC = f->T; fb_addC(B, -1.0, AyC);
                for (size_t i = 0; i < len; i++) f->W[i] = f->X[i] / sigma + AyC[i];
            }
            /* the acceleration on the concatenation (contiguous since 4.37: in place) */
            prof[2] += wtime() - tp0; tp0 = wtime();
            /* 4.37 research knobs (environment): over-relaxation of the Douglas-Rachford map, Nesterov
             * momentum with adaptive restart or Halpern anchoring in place of Anderson, Anderson
             * applied every k-th step (on T^k) */
            double *Wv = aapack ? Wp : Wall;
            if (aapack) pk_pack(P, Wa, pko, Wp);
            if (x_relax != 1.0 && AD.rprev >= 0) { const double a1 = 1.0 - x_relax; for (size_t i = 0; i < Nv; i++) Wv[i] = a1 * AD.w[i] + x_relax * Wv[i]; }
            if (x_mom == 1) {
                /* Nesterov on the fixed-point map with the O'Donoghue-Candes restart (residual increase) */
                if (it <= 1) { memcpy(AD.w, Wv, sizeof(double) * Nv); memcpy(AD.g, Wv, sizeof(double) * Nv); mom_t = 1; mom_r = 1e300; }
                else {
                    double r2 = 0; for (size_t i = 0; i < Nv; i++) { const double d = Wv[i] - AD.w[i]; r2 += d * d; }
                    const double rr = sqrt(r2);
                    if (rr > mom_r) { mom_t = 1; mom_nrest++; }
                    mom_r = rr;
                    const double tn = 0.5 * (1 + sqrt(1 + 4 * mom_t * mom_t)), be = (mom_t - 1) / tn;
                    for (size_t i = 0; i < Nv; i++) { const double gi = Wv[i]; Wv[i] = gi + be * (gi - AD.g[i]); AD.g[i] = gi; }
                    memcpy(AD.w, Wv, sizeof(double) * Nv); mom_t = tn;
                }
            } else if (x_mom == 2) {
                /* Halpern anchoring with restarts (HPR's rules) on the Douglas-Rachford variable */
                if (it <= 1) { memcpy(AD.w, Wv, sizeof(double) * Nv); memcpy(AD.rlast, Wv, sizeof(double) * Nv); mom_t = 0; mom_r = 1e300; mom_ra = 1e300; }
                else {
                    double r2 = 0; for (size_t i = 0; i < Nv; i++) { const double d = Wv[i] - AD.w[i]; r2 += d * d; }
                    const double rr = sqrt(r2);
                    int rs = 0;
                    if (mom_t == 0) { mom_ra = rr; }
                    else if (rr <= 0.2 * mom_ra || (rr <= 0.8 * mom_ra && rr > mom_r) || (mom_t >= 0.2 * it && mom_t >= 100)) rs = 1;
                    mom_r = rr;
                    if (rs) { memcpy(AD.rlast, Wv, sizeof(double) * Nv); mom_t = 0; mom_ra = rr; mom_nrest++; memcpy(AD.w, Wv, sizeof(double) * Nv); }
                    else { const double a = 1.0 / (mom_t + 2.0), bq = (mom_t + 1.0) / (mom_t + 2.0); for (size_t i = 0; i < Nv; i++) Wv[i] = a * AD.rlast[i] + bq * Wv[i]; memcpy(AD.w, Wv, sizeof(double) * Nv); mom_t++; }
                }
            } else if (x_every <= 1 || it % x_every == 0 || it <= 1) aa_apply(&AD, Wv, it);
            if (aapack) pk_unpack(P, Wa, pko, Wp);
            prof[3] += wtime() - tp0; tp0 = wtime();
            #pragma omp parallel for schedule(dynamic) reduction(+:rd2,pobj) if (nb > 1)
            for (int kk = 0; kk < nb; kk++) {
                const int k = bord[kk];
                const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
                const double *AyC = f->T;
                fb_project(f, B, f->W, f->P, f->Z);
                double r2 = 0;
                for (size_t i = 0; i < len; i++) { const double v = AyC[i] + f->Z[i]; r2 += v * v; }
                for (size_t i = 0; i < len; i++) f->X[i] = sigma * f->P[i];
                rd2 += r2;
                pobj += fb_Cdot(B, f->X);
            }
        } else {
        fop_At(g_fop, P, y, Ta);
        #pragma omp parallel for schedule(dynamic) reduction(+:rd2,pobj) if (nb > 1)
        for (int kk = 0; kk < nb; kk++) {
            const int k = bord[kk];
            const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
            double *AyC = f->T;
            fb_addC(B, -1.0, AyC);
            if (halpern) for (size_t i = 0; i < len; i++) f->W[i] = f->X[i] / sigma + f->Z[i] + 2.0 * AyC[i];
            else for (size_t i = 0; i < len; i++) f->W[i] = f->X[i] / sigma + AyC[i];
            fb_project(f, B, f->W, f->P, f->Z);
            double r2 = 0;
            for (size_t i = 0; i < len; i++) { const double v = AyC[i] + f->Z[i]; r2 += v * v; }
            for (size_t i = 0; i < len; i++) f->X[i] = (1.0 - tau) * f->X[i] + tau * sigma * f->P[i];
            rd2 += r2;
            pobj += fb_Cdot(B, f->X);
        }
        }
        if (ENV_ON("BRISK_FOMDBG") && it <= 6) {
            /* per-block check of the projection: |Pos Neg|_F / (|Pos| |Neg|) (should be ~1e-15), norms of X, Z, y */
            double ny = 0; for (int i = 0; i < m; i++) ny += y[i] * y[i];
            printf("   [fomdbg it %d: |y| %.3e", it, sqrt(ny));
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
                double nx = 0, nz = 0, nw = 0; for (size_t i = 0; i < len; i++) { nx += f->X[i] * f->X[i]; nz += f->Z[i] * f->Z[i]; nw += f->W[i] * f->W[i]; }
                double cpl = -1;
                if (B->type == BLK_SDP) {
                    const int n = B->n; double *T = f->T;
                    BL(dgemm_)("N", "N", &n, &n, &n, &DONE, f->P, &n, f->Z, &n, &DZERO, T, &n);
                    double np2 = 0, c2 = 0; for (size_t i = 0; i < len; i++) { np2 += f->P[i] * f->P[i]; c2 += T[i] * T[i]; }
                    cpl = sqrt(c2) / fmax(sqrt(np2 * nz), 1e-300);
                }
                printf(" | blk %d |X| %.2e |Z| %.2e |W| %.2e PosNeg %.1e npos/nneg %d/%d", k, sqrt(nx), sqrt(nz), sqrt(nw), cpl, f->npos, f->nneg);
            }
            printf("]\n"); fflush(stdout);
        }
        /* residuals in the original scaling */
        prof[4] += wtime() - tp0; tp0 = wtime();
        memcpy(rp, P->b, sizeof(double) * m);
        if (fuse) { memset(AZ, 0, sizeof(double) * m); fop_A2(g_fop, Xa, rp, -1.0, Za, AZ, 1.0); rhs_ok = 1; }
        else fop_A(g_fop, Xa, rp, -1.0);
        prof[5] += wtime() - tp0;
        double rpo = 0; for (int i = 0; i < m; i++) { const double v = rp[i] * P->du[i]; rpo += v * v; } rpo = sqrt(rpo);
        dobj = 0; for (int i = 0; i < m; i++) dobj += P->b[i] * y[i];
        const double po = sc * pobj, dob = sc * dobj, den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        pinf = rpo / (1 + P->normb2); dinf = P->cs * sqrt(rd2) / (1 + P->normC2); gap = fabs(po - dob) / den;
        const double res = fmax(fmax(pinf, dinf), gap);
        if (res < best) best = res;
        if (pinf > 0 && dinf > 0) { sb_lp += log(pinf); sb_ld += log(dinf); sb_n++; }
        if (avg_on) {
            const double w = avg_n == 0 ? 1.0 : fmin(1.0, avg_c / it), w1 = 1.0 - w;
            #pragma omp parallel for schedule(dynamic) if (nb > 1)
            for (int kk = 0; kk < nb; kk++) { const int k = bord[kk]; const size_t len = fbsz(&P->blk[k]); double *ax = AvXa[k], *az = AvZa[k]; const double *x = F[k].X, *z = F[k].Z;
                for (size_t i = 0; i < len; i++) { ax[i] = w1 * ax[i] + w * x[i]; az[i] = w1 * az[i] + w * z[i]; } }
            for (int i = 0; i < m; i++) Avy[i] = w1 * Avy[i] + w * y[i];
            avg_n++;
            if (it % 25 == 0 && avg_n >= 50) {
                double ap, ad, ag, apo, ado;
                avg_res = fom_point_res(P, AvXa, AvZa, Avy, Ta, Avr, &ap, &ad, &ag, &apo, &ado);
                if (avg_res < res) avg_hits++;
                if (avg_rs > 0 && avg_res < avg_rs * res && avg_res > tol) {
                    /* restart the splitting from the average (PDLP's restart to the average): X, Z
                     * replaced, y recomputed by the next y-step, the acceleration reset */
                    for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, AvXa[k], sizeof(double) * len); memcpy(F[k].Z, AvZa[k], sizeof(double) * len); }
                    aa_reset(&AD); rhs_ok = 0; avg_n = 0; avg_nrs++;
                    if (verbose > 1) printf("      restart from the average (%.1e against %.1e)\n", avg_res, res);
                    continue;
                }
                if (verbose > 1 && it % 50 == 0) { printf("      average %8.1e %8.1e %8.1e  (objectives %.7e %.7e)\n", ap, ad, ag, sc * apo + P->obj_off, sc * ado + P->obj_off); fflush(stdout); }
                if (avg_res <= tol) {
                    for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, AvXa[k], sizeof(double) * len); memcpy(F[k].Z, AvZa[k], sizeof(double) * len); }
                    memcpy(y, Avy, sizeof(double) * m); pinf = ap; dinf = ad; gap = ag; pobj = apo; dobj = ado;
                    if (verbose) printf("first-order: the average of the iterates meets the tolerance at iteration %d\n", it);
                    status = ST_OPTIMAL; break;
                }
            }
        }
        if (ftrace) { fprintf(ftrace, "%d %.3f %.3e %.3e %.3e %.3e %.10e %.10e\n", it, wtime() - t0, pinf, dinf, gap, sigma, po + P->obj_off, dob + P->obj_off); if (it % 50 == 0) fflush(ftrace); }
        if (verbose > 1 && (it % (verbose > 2 ? 1 : 50) == 0 || it == 1 || res <= tol))
            { printf("%5d %14.7e %14.7e %8.1e %8.1e %8.1e %8.2g %7.1fs\n", it, po + P->obj_off, dob + P->obj_off, pinf, dinf, gap, sigma, wtime() - t0); fflush(stdout); }
        if (res <= tol) { status = ST_OPTIMAL; break; }
        /* 4.39: when only the dual residual is above the tolerance, the best Z for this y (the
         * negative part of C - A*y; the final measure on the file uses it too) may already meet it;
         * checked at most every 5 iterations (one eigenvalue computation per block) */
        if (bestz_on && aadr && fmax(pinf, gap) <= tol && dinf > tol && it >= bz_next) {
            bz_next = it + 5; n_bestz++;
            const double db = P->cs * fom_dinf_best(P, F, bord) / (1 + P->normC2);
            if (verbose > 1) printf("      dual residual with the best Z: %.1e (iterate's %.1e)\n", db, dinf);
            if (db <= tol) { dinf = db; status = ST_OPTIMAL; if (verbose) printf("first-order: the best Z for y meets the tolerance at iteration %d\n", it); break; }
        }
        if (brisk_time_up(par)) { status = ST_TIME; break; }
        if (par->fom_bm && (it >= par->fom_ssn_after || res <= par->fom_ssn_res)) { status = -2; break; }   /* to kernel B */
        /* to phase II: at a residual of fom_ssn_res, or after fom_ssn_after iterations once the
         * splitting stalls (4.37: its best residual improved by less than 1.3x since iteration
         * it/2; the 4.31-4.36 rule entered at iteration 300 regardless, which cut the splitting
         * short on the moment relaxations while it still made k^-0.8 progress); after a stalled
         * phase II not before the residual is 3x below that attempt's entry */
        if (bhist && it < bhist_n) { bhist[it] = best; thist[it] = wtime(); }
        {   const int stalled = !ssn_stallrule || (it >= 200 && bhist && it < bhist_n && best > bhist[it / 2] / 1.3);
            if (par->fom_ssn && it >= ssn_block_until && res <= ssn_reentry && ((it >= par->fom_ssn_after && stalled) || res <= par->fom_ssn_res)) {
                ssn_entry_res = res; status = -1;
                /* the splitting's rate over the second half of its run (or of the last resume) */
                g_split_rate = 0;
                if (bhist && it < bhist_n) { const int i0 = ssn_split0 + (it - ssn_split0) / 2; if (i0 >= 1 && it > i0 && thist[it] > thist[i0] && bhist[i0] > best) g_split_rate = log(bhist[i0] / best) / (thist[it] - thist[i0]); }
                break;
            }   /* to phase II */
        }
        /* Halpern anchoring with restarts (HPR): z_{k+1} = 1/(k+2) z_a + (k+1)/(k+2) T z_k */
        if (halpern) {
            int restart = 0;
            if (kh == 0) { res_anchor = res; res_last = res; }
            else {
                if (res <= 0.2 * res_anchor) restart = 1;                       /* sufficient decay */
                else if (res <= 0.8 * res_anchor && res > res_last) restart = 1; /* necessary decay + no progress */
                else if (kh >= 0.2 * it && kh >= 200) restart = 1;             /* long */
            }
            if (restart) {
                for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].Xa, F[k].X, sizeof(double) * len); memcpy(F[k].Za, F[k].Z, sizeof(double) * len); }
                kh = 0; nrestart++; res_anchor = res;
            } else {
                const double a = 1.0 / (kh + 2.0), bq = (kh + 1.0) / (kh + 2.0);
                for (int k = 0; k < nb; k++) {
                    const size_t len = fbsz(&P->blk[k]); FB *f = &F[k];
                    for (size_t i = 0; i < len; i++) { f->X[i] = a * f->Xa[i] + bq * f->X[i]; f->Z[i] = a * f->Za[i] + bq * f->Z[i]; }
                }
            }
            res_last = res;
            kh++;
        }
        /* Anderson step */
        if (aa_mem > 0) {
            /* g_k = T(w_k) and r_k = g_k - w_k */
            size_t o = 0; double rn2 = 0;
            for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]);
                for (size_t i = 0; i < len; i++) { const double g = F[k].X[i], r = g - aa_w[o + i]; aa_r[o + i] = r; rn2 += r * r; } o += len;
                for (size_t i = 0; i < len; i++) { const double g = F[k].Z[i], r = g - aa_w[o + i]; aa_r[o + i] = r; rn2 += r * r; } o += len; }
            const double rn = sqrt(rn2);
            int reject = 0;
            if (aa_used && aa_rsafe >= 0 && rn > aa_rsafe) {
                /* the accelerated w_k did worse than the plain step would have: back to g_{k-1} */
                reject = 1; aa_nrej++;
                o = 0;
                for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, aa_g + o, sizeof(double) * len); o += len; memcpy(F[k].Z, aa_g + o, sizeof(double) * len); o += len; }
                aa_n = 0; aa_head = 0; aa_used = 0; aa_rsafe = -1; aa_rprev = -1;
            }
            if (!reject) {
                if (aa_rprev >= 0) {          /* push differences (r_k - r_{k-1}, g_k - g_{k-1}); aa_g/aa_r hold k-1 until overwritten */
                    double *dR = aa_dR + (size_t)aa_head * N2, *dG = aa_dG + (size_t)aa_head * N2;
                    o = 0;
                    for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]);
                        for (size_t i = 0; i < len; i++) { dG[o + i] = F[k].X[i] - aa_g[o + i]; } o += len;
                        for (size_t i = 0; i < len; i++) { dG[o + i] = F[k].Z[i] - aa_g[o + i]; } o += len; }
                    for (size_t i = 0; i < N2; i++) dR[i] = aa_r[i] - aa_rlast[i];
                    {   /* the Gram matrix of the differences, one new row/column (O(mem N2), not O(mem^2 N2)) */
                        const int h = aa_head, na = aa_n < aa_mem ? aa_n + 1 : aa_mem;
                        #pragma omp parallel for schedule(static) if (N2 > 200000)
                        for (int c = 0; c < na; c++) { const double *dc = aa_dR + (size_t)c * N2; double v = 0; for (size_t i = 0; i < N2; i++) v += dR[i] * dc[i]; aa_Gm[h * AA_MAX + c] = aa_Gm[c * AA_MAX + h] = v; }
                    }
                    aa_head = (aa_head + 1) % aa_mem; if (aa_n < aa_mem) aa_n++;
                    if (aa_n == aa_mem && ENV_ON("BRISK_AARESTART")) { aa_n = 0; aa_head = 0; }   /* restarted memory (COSMO's) */
                }
                /* store g_k, r_k */
                o = 0;
                for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(aa_g + o, F[k].X, sizeof(double) * len); o += len; memcpy(aa_g + o, F[k].Z, sizeof(double) * len); o += len; }
                memcpy(aa_rlast, aa_r, sizeof(double) * N2);
                aa_rprev = rn;
                aa_rsafe = par->fom_aasafe * rn;   /* the next residual must not exceed this */
                aa_used = 0;
                if (aa_n >= 1 && it >= 3) {
                    /* gamma = argmin || r_k - dR gamma ||  (normal equations, regularized) */
                    double Mx[AA_MAX * AA_MAX], rv[AA_MAX], gam[AA_MAX];
                    #pragma omp parallel for schedule(static) if (N2 > 200000)
                    for (int a = 0; a < aa_n; a++) {
                        const double *da = aa_dR + (size_t)a * N2;
                        double s0 = 0; for (size_t i = 0; i < N2; i++) s0 += da[i] * aa_r[i]; rv[a] = s0;
                    }
                    for (int a = 0; a < aa_n; a++) for (int c = 0; c < aa_n; c++) Mx[a * AA_MAX + c] = aa_Gm[a * AA_MAX + c];
                    double tr = 0; for (int a = 0; a < aa_n; a++) tr += Mx[a * AA_MAX + a];
                    for (int a = 0; a < aa_n; a++) Mx[a * AA_MAX + a] += 1e-10 * tr + 1e-300;
                    /* Cholesky solve */
                    int ok = 1;
                    for (int j = 0; j < aa_n && ok; j++) {
                        double d = Mx[j * AA_MAX + j]; for (int q = 0; q < j; q++) d -= Mx[j * AA_MAX + q] * Mx[j * AA_MAX + q];
                        if (d <= 0) { ok = 0; break; }
                        d = sqrt(d); Mx[j * AA_MAX + j] = d;
                        for (int i = j + 1; i < aa_n; i++) { double v = Mx[i * AA_MAX + j]; for (int q = 0; q < j; q++) v -= Mx[i * AA_MAX + q] * Mx[j * AA_MAX + q]; Mx[i * AA_MAX + j] = v / d; }
                    }
                    if (ok) {
                        for (int i = 0; i < aa_n; i++) { double v = rv[i]; for (int q = 0; q < i; q++) v -= Mx[i * AA_MAX + q] * gam[q]; gam[i] = v / Mx[i * AA_MAX + i]; }
                        for (int i = aa_n - 1; i >= 0; i--) { double v = gam[i]; for (int q = i + 1; q < aa_n; q++) v -= Mx[q * AA_MAX + i] * gam[q]; gam[i] = v / Mx[i * AA_MAX + i]; }
                        /* w_{k+1} = g_k - dG gamma */
                        o = 0;
                        for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]);
                            for (int a = 0; a < aa_n; a++) { const double *dg = aa_dG + (size_t)a * N2; const double ga = gam[a];
                                for (size_t i = 0; i < len; i++) F[k].X[i] -= ga * dg[o + i];
                                for (size_t i = 0; i < len; i++) F[k].Z[i] -= ga * dg[o + len + i]; }
                            o += 2 * len; }
                        aa_used = 1; aa_naccel++;
                    }
                }
            }
        }
        /* sigma: balance the residuals (pinf ~ sigma, dinf ~ 1/sigma) */
        if (it >= sig_next && par->fom_sigma <= 0) {
            sig_next = it + sig_cur;
            const double r = pinf / fmax(dinf, 1e-300);
            double fac = 1.0;
            if (par->fom_sigrule == 0) {
                if (r > 5.0) fac = 1.0 / fmin(2.0, sqrt(r) / 1.5);
                else if (r < 0.2) fac = fmin(2.0, sqrt(1.0 / r) / 1.5);
            } else if (par->fom_sigrule >= 2) {
                /* 4.37 (research): the penalty as the ratio of the primal and dual iterates (2: norms of
                 * X and Z; 3: their movement since the last update, PDLP's primal weight), in the
                 * Douglas-Rachford variable's balance X/sigma ~ Z; a dead zone of 2, smoothed in log */
                double nx = 0, nz = 0;
                for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); const FB *f = &F[k];
                    if (par->fom_sigrule == 2 || par->fom_sigrule == 4) { for (size_t i = 0; i < len; i++) { nx += f->X[i] * f->X[i]; nz += f->Z[i] * f->Z[i]; } }
                    else if (sr_X) { const size_t o = sr_off[k]; for (size_t i = 0; i < len; i++) { const double a = f->X[i] - sr_X[o + i], b2 = f->Z[i] - sr_Z[o + i]; nx += a * a; nz += b2 * b2; } } }
                if (par->fom_sigrule == 3) { if (!sr_X) { sr_X = fo_malloc(sizeof(double) * Nw); sr_Z = fo_malloc(sizeof(double) * Nw); sr_off = fo_malloc(sizeof(size_t) * (nb + 1)); size_t o = 0; for (int k = 0; k < nb; k++) { sr_off[k] = o; o += fbsz(&P->blk[k]); } }
                    for (int k = 0; k < nb; k++) { memcpy(sr_X + sr_off[k], F[k].X, sizeof(double) * fbsz(&P->blk[k])); memcpy(sr_Z + sr_off[k], F[k].Z, sizeof(double) * fbsz(&P->blk[k])); } }
                if (getenv("BRISK_SIGDBG") && atoi(getenv("BRISK_SIGDBG")) >= 2) {
                    printf("   [blocks it %d:", it);
                    for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); const FB *f = &F[k]; double a = 0, b2 = 0, dr = 0; for (size_t i = 0; i < len; i++) { a += f->X[i] * f->X[i]; b2 += f->Z[i] * f->Z[i]; const double v = f->T[i] + f->Z[i]; dr += v * v; }
                        printf(" %d:X%.2e Z%.2e r%.1e d%.1e", k, sqrt(a), sqrt(b2), b2 > 0 ? sqrt(a / b2) : 0.0, sqrt(dr)); }
                    printf("]\n");
                }
                if (getenv("BRISK_SIGDBG")) printf("   [sigma rule it %d: |X| %.3e |Z| %.3e target %.3e sigma %.3e, pinf/dinf %.2e]\n", it, sqrt(nx), sqrt(nz), nz > 0 ? sqrt(nx / nz) : 0.0, sigma, r);
                if (nx > 0 && nz > 0 && it >= 40) {
                    double target = sqrt(nx / nz);
                    /* 4.39 (rule 4): the norm ratio corrected by the residual balance over the interval,
                     * sqrt(dinf / pinf) (geometric means; the factor within [1/4, 4]) */
                    if (par->fom_sigrule == 4 && sb_n > 0) { const double cf = exp(0.5 * (sb_ld - sb_lp) / sb_n); target *= fmin(4.0, fmax(0.25, cf)); }
                    const double lf = 0.5 * log(target / sigma);      /* half-way in log */
                    fac = exp(fmin(log(par->fom_sigmax * 2), fmax(-log(par->fom_sigmax * 2), lf)));
                    if (fabs(log(target / sigma)) < log(2.0)) fac = 1.0;
                }
                /* the iterates' norms mean something only near a solution: while the residuals are
                 * lopsided (ratio outside [0.1, 10]) and not falling (less than 30% since the last
                 * check), residual balancing (ss30: both norms grow with sigma = 1; balancing takes
                 * sigma to 0.13, where the splitting converges) */
                if ((r > 10 || r < 0.1) && res > 0.7 * sig_res_last) {
                    fac = 1.0;
                    if (r > 5.0) fac = 1.0 / fmin(2.0, sqrt(r) / 1.5);
                    else if (r < 0.2) fac = fmin(2.0, sqrt(1.0 / r) / 1.5);
                }
                /* 4.39: never against the residual balance over the interval (geometric means):
                 * a smaller sigma lowers pinf and raises dinf, so it waits while dinf is the larger,
                 * and a larger one while pinf is (Example 8.1.3 d = 5: at iteration ~3 800 the norm
                 * rule took sigma 0.037 -> 0.026 with pinf = dinf = 3e-3, and dinf went to 5e-3 ... 2e-2) */
                if (sig_dir && sb_n > 0 && fac != 1.0) {
                    const double lr = (sb_lp - sb_ld) / sb_n;    /* log(pinf / dinf) */
                    if ((fac < 1.0 && lr < 0) || (fac > 1.0 && lr > 0 && sig_dir == 1)) { if (getenv("BRISK_SIGDBG")) printf("   [sigma change %.3g held: pinf/dinf %.2f over the interval]\n", fac, exp(lr)); fac = 1.0; }
                }
                sig_res_last = res; sb_lp = sb_ld = 0; sb_n = 0;
            } else {
                fac = fmin(par->fom_sigmax, fmax(1.0 / par->fom_sigmax, sqrt(1.0 / r)));
                if (fabs(log(fac)) < log(1.2)) fac = 1.0;
            }
            if (fac != 1.0) {
                sigma *= fac; nsig++;
                if (sig_double) { sig_cur *= 2; sig_next = it + sig_cur; }     /* 4.37: rarer changes (each resets the acceleration) */
                if (aadr && !aa_keep) aa_reset(&AD);
                if (aa_mem > 0) { aa_n = 0; aa_head = 0; aa_rprev = -1; aa_rsafe = -1; if (aa_used) { size_t o = 0; for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, aa_g + o, sizeof(double) * len); o += len; memcpy(F[k].Z, aa_g + o, sizeof(double) * len); o += len; } aa_used = 0; } }
                if (halpern) { for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].Xa, F[k].X, sizeof(double) * len); memcpy(F[k].Za, F[k].Z, sizeof(double) * len); } kh = 0; nrestart++; }
            }
        }
    }
    if (it > maxit) it = maxit;
    if (avg_on && (status == ST_TIME || status == ST_MAXIT) && avg_n >= 50) {
        double ap, ad, ag, apo, ado;
        const double ra = fom_point_res(P, AvXa, AvZa, Avy, Ta, Avr, &ap, &ad, &ag, &apo, &ado);
        if (ra < fmax(fmax(pinf, dinf), gap)) {
            if (verbose) printf("first-order: the average of the iterates is returned (%.1e %.1e %.1e against %.1e %.1e %.1e)\n", ap, ad, ag, pinf, dinf, gap);
            for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, AvXa[k], sizeof(double) * len); memcpy(F[k].Z, AvZa[k], sizeof(double) * len); }
            memcpy(y, Avy, sizeof(double) * m); pinf = ap; dinf = ad; gap = ag; pobj = apo; dobj = ado;
        }
    }
    if (verbose && n_bestz) printf("first-order: %d checks of the dual residual with the best Z\n", n_bestz);
    if (avg_on && verbose) printf("first-order: iterate average: better than the iterate at %d of the checks, %d restarts from it\n", avg_hits, avg_nrs);
    g_fom_single_now = 0;
    SsnStat st = { 0, 0, 0, 0, 0 };
    BmStat bst = { 0, 0, 0, 0, 0 };
    if (status == -2) {
        if (aa_used) { size_t o = 0; for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, aa_g + o, sizeof(double) * len); o += len; memcpy(F[k].Z, aa_g + o, sizeof(double) * len); o += len; } }
        if (verbose) printf("first-order: kernel B (low-rank augmented Lagrangian) from iteration %d, residuals %.1e %.1e %.1e\n", it, pinf, dinf, gap);
        const double ts = wtime();
        status = bm_phase(P, par, F, nb, m, y, tol, t0, verbose, &pinf, &dinf, &gap, &pobj, &dobj, &bst, &it);
        bst.t = wtime() - ts;
        if (verbose) printf("first-order: kernel B %d outer, %d L-BFGS steps, %d function evaluations, %d rank increases, %.1fs\n", bst.outer, bst.inner, bst.ls, bst.rankup, bst.t);
        if (status != ST_OPTIMAL && par->fom_ssn) status = -1;      /* not converged: phase II from here */
    }
    if (status == -1) {
        if (aa_used) { size_t o = 0; for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); memcpy(F[k].X, aa_g + o, sizeof(double) * len); o += len; memcpy(F[k].Z, aa_g + o, sizeof(double) * len); o += len; } }
        if (verbose) printf("first-order: phase II (augmented Lagrangian, semismooth Newton-CG) from iteration %d, residuals %.1e %.1e %.1e\n", it, pinf, dinf, gap);
        const double ts = wtime();
        double sig_ssn = sigma;
        status = ssn_phase(P, par, F, nb, m, y, &sig_ssn, G, tol, t0, verbose, &pinf, &dinf, &gap, &pobj, &dobj, &st, &it);
        st.t += wtime() - ts;
        if (status == -3 && it < maxit && !(brisk_time_up(par))) {
            /* 4.37: after the third stalled attempt the splitting goes on without phase II (it used
             * to end the run: with the rate race the attempts end sooner, and d = 3 stopped at 54 s) */
            /* phase II stalled: back to the splitting from its best point. The state (X, Z) is
             * rebuilt from (X, y): W = X/sigma + A*y - C, X = sigma Pi(W), Z = Pi(-W); the
             * acceleration starts afresh; phase II is tried again after fom_ssn_after more iterations */
            nresume++;
            if (verbose) printf("first-order: phase II stalled at %.1e %.1e %.1e: back to the splitting from iteration %d\n", pinf, dinf, gap, it);
            fop_At(g_fop, P, y, Ta);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; FB *f = &F[k]; const size_t len = fbsz(B);
                fb_addC(B, -1.0, f->T);
                for (size_t i = 0; i < len; i++) f->W[i] = f->X[i] / sigma + f->T[i];
                fb_project(f, B, f->W, f->P, f->Z);
                for (size_t i = 0; i < len; i++) f->X[i] = sigma * f->P[i];
            }
            if (aadr) aa_reset(&AD);
            if (aa_mem > 0) { aa_n = 0; aa_head = 0; aa_rprev = -1; aa_rsafe = -1; aa_used = 0; }
            ssn_block_until = nresume >= 3 ? maxit + 1 : it + par->fom_ssn_after; it++;
            ssn_reentry = 0.3 * ssn_entry_res;              /* 4.37: back in only 3x below this attempt's entry */
            ssn_split0 = it;
            status = ST_MAXIT; rhs_ok = 0;
            goto resume_splitting;
        }
        if (status == -3) status = (brisk_time_up(par)) ? ST_TIME : ST_MAXIT;
        if (verbose) printf("first-order: phase II %d outer, %d Newton steps, %d CG steps, %d line-search projections, %.1fs\n", st.outer, st.newton, st.cg, st.ls, st.t);
    }
    R->status = status; R->iters = it;
    R->pobj = sc * pobj; R->dobj = sc * dobj; R->pinf = pinf; R->dinf = dinf; R->relgap = gap; R->relcomp = gap;
    R->methods = 4;
    double t_eig = 0; for (int k = 0; k < nb; k++) t_eig += F[k].t_eig;
    R->t_total = wtime() - t0; R->t_chol = t_solve; R->t_dense = t_eig; R->t_setup = t_gram;
    if (verbose) { printf("first-order: eigenvalue signs of the last W per SDP block (positive = rank of X, negative = rank of Z):"); for (int k = 0; k < nb; k++) if (F[k].type == BLK_SDP) printf(" %d/%d", F[k].npos, F[k].nneg); printf("\n"); }
    long npart = 0, nfull = 0, nlz = 0, nlzf = 0, nsgl = 0; for (int k = 0; k < nb; k++) { npart += F[k].n_partial; nfull += F[k].n_full; nlz += F[k].n_lanczos; nlzf += F[k].n_lzfail; nsgl += F[k].n_single; }
    if (verbose && nsgl) printf("first-order: %ld projections in single precision\n", nsgl);
    if (ftrace) fclose(ftrace);
    g_ftrace = NULL;
    free(sr_X); free(sr_Z); free(sr_off); free(bhist); free(thist);
    if (verbose && x_mom) printf("first-order: %s, %d restarts\n", x_mom == 1 ? "Nesterov momentum" : "Halpern anchoring", mom_nrest);
    { long nrr = 0, nrrf = 0; for (int k = 0; k < nb; k++) { nrr += F[k].n_rr; nrrf += F[k].n_rrfail; }
      long nsr = 0; for (int k = 0; k < nb; k++) nsr += F[k].n_srange;
      if (verbose && (nrr || nrrf)) printf("first-order: %ld Rayleigh-Ritz projections (%ld fell back to a full decomposition)\n", nrr, nrrf);
      if (verbose && nsr) printf("first-order: %ld single-precision projections by the small side's value range (ssyevr)\n", nsr);
      long ndc = 0; for (int k = 0; k < nb; k++) ndc += F[k].n_dcpart;
      if (verbose && ndc) printf("first-order: %ld projections by tridiagonalization, divide and conquer and the small side's back-transformation\n", ndc); }
    if (getenv("BRISK_FOMPROF")) { double tc = 0, tk = 0, te = 0; for (int k = 0; k < nb; k++) { tc += F[k].t_core; tk += F[k].t_syrk; te += F[k].t_eig; } printf("first-order projections, summed over the blocks (s): %.2f, of which single-precision LAPACK %.2f, rank-k products %.2f\n", te, tc, tk); }
    if (getenv("BRISK_FOMPROF")) printf("first-order profile (s): rhs %.2f, Gram solve %.2f, A*y + W %.2f, acceleration %.2f, projections %.2f, residual %.2f\n", prof[0], prof[1], prof[2], prof[3], prof[4], prof[5]);
    free(bord);
    if (verbose) printf("first-order: %d iterations, %d sigma changes, %d restarts, Anderson(%d) %d accelerated %d rejected; eig %.1fs (%ld Lanczos [%ld failed, %ld products], %ld partial, %ld full), Gram solves %.1fs, total %.1fs\n", it, nsig, nrestart, aa_mem, aa_naccel, aa_nrej, t_eig, nlz, nlzf, g_lz_prod, npart, nfull, t_solve, R->t_total);
    free(aa_w); free(aa_g); free(aa_r); free(aa_rlast); free(aa_dR); free(aa_dG);
    if (aadr) { if (verbose) printf("first-order: Douglas-Rachford acceleration (memory %d%s): %d accelerated, %d rejected\n", AD.mem, aapack ? ", packed triangles" : "", AD.naccel, AD.nrej);
        if (verbose && AD.type == 1) printf("first-order: type I steps %d (%d singular, type II used)\n", AD.n1, AD.n1fail);
        aa_free(&AD); }
    if (yout) for (int i = 0; i < m; i++) yout[i] = P->cs * P->d[i] * y[i];
    /* split free pairs: the engine keeps the free value spread over the two slots (X+ = -X-
     * in the splitting phase); the file's variables are nonnegative, so the value f = X+ - X-
     * goes back as (max(f, 0), max(-f, 0)) - A(X) and <C, X> are unchanged */
    { FreePair *fp = NULL; const int nf = getenv("BRISK_FOMNOFREE") ? 0 : free_pairs_detect(P, &fp);
      for (int q = 0; q < nf; q++) { double *X = F[fp[q].blk].X; const double f = X[fp[q].ip] - X[fp[q].im]; X[fp[q].ip] = f > 0 ? f : 0; X[fp[q].im] = f < 0 ? -f : 0; }
      free(fp); }
    if (Xout) for (int k = 0; k < nb; k++) { const size_t len = fbsz(&P->blk[k]); for (size_t i = 0; i < len; i++) Xout[k][i] = P->bs * F[k].X[i]; }
    for (int k = 0; k < nb; k++) { FB *f = &F[k]; free(f->Vf); free(f->evf); free(f->workf); free(f->iworkf); free(f->X); free(f->Z); if (!Wall) free(f->W); free(f->V); free(f->ev); free(f->work); free(f->iwork); free(f->Xa); free(f->Za); free(f->T); free(f->P); free(f->freev); free(f->Vp); free(f->rrQ); free(f->rrT); free(f->rrH); free(f->rrE); free(f->srZ); free(f->dcd); free(f->dce); free(f->dctau); free(f->dcw); free(f->dciw); free(f->dcdd); free(f->dcde); free(f->dcdtau); free(f->dcdw); free(f->dcdiw); free(f->srsup); free(f->srwork); free(f->sriwork); free(f->isup); if (f->lz) { lz_free(f->lz); free(f->lz); } }
    free(F); free(y); free(ya); free(rhs); free(rp); fom_gram_free(G);
    free(Xa); free(Wa); free(Ta); free(Wall); free(Wp); free(pko); free(AZ); free(AC); free(Za); free(AvX); free(AvZ); free(Avy); free(Avr); free(AvXa); free(AvZa); fop_free(g_fop); g_fop = NULL;
    return 0;
}
