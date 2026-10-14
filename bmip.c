/* bmip.c - 5.16: the Burer-Monteiro program solved by a primal-dual interior-point method for nonlinear
 * programs (`-bm 2`; IMPLEMENTATION 15.5).
 *
 *   min sum_k <C_k, R_k R_k'> + c'x   s.t.  sum_k <A_ik, R_k R_k'> + a_i'x = b_i,  x >= 0,
 *
 * one factor R_k (n_k x r_k) per semidefinite block, x the LP variables. The constraints are quadratic in
 * R, so the program is a smooth nonlinear program; its KKT system has the sparsity of the data (the pattern
 * of S_k = C_k + sum_i lam_i A_ik for every column of R_k, and the rows of the Jacobian, 2 A_ik R_k) and is
 * factored by the sparse LDL' of sparsechol.c with prescribed pivot signs (positive for the variables,
 * negative for the constraints): a factorization that succeeds has the right inertia, one that fails asks
 * for a larger regularization of the Hessian (the inertia correction of Waechter and Biegler). The step is
 * accepted by their filter line search; the barrier parameter decreases monotonically.
 *
 * Before: the sign symmetries of the data (symred.c) split every block into its sign classes and drop the
 * constraints that change sign (the angle reference of an AC power flow relaxation is one: degenerate in the
 * factored form, its multiplier diverges). A row of R_k whose diagonal X_ii is fixed by a constraint of its
 * own is fixed (R_i = (sqrt(X_ii), 0, ..)), else one row's columns 1.. are fixed at 0 (the rotation R -> R Q).
 *
 * After: S_k = C_k - sum y_i A_ik (y = -lam) is positive semidefinite exactly when the point solves the SDP;
 * lambda_min(S_k) is found by factorizations of S_k - s I (bisection) and inverse iteration. A block with a
 * clearly negative lambda_min gets one more column along its eigenvector and the program is solved again
 * from the current point (warm start, small barrier parameter).
 *
 * Start: rank one with all rows parallel, each row's square chosen so that the constraints on diagonal
 * entries alone sit in the middle of their range (for a power flow relaxation the flat start, rotated by
 * the global phase), the other columns small.                                                              */
#include "brisk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double wtime(void);
#define DBL_EPSILON_ 2.220446049250313e-16

static void *ix(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (-bm 2, %.2f GB)\n", n / 1e9); exit(1); } return p; }
static void *iz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (-bm 2, %.2f GB)\n", n / 1e9); exit(1); } return p; }

typedef struct {
    int n, r;
    int np;                  /* positions (j <= k) of the union pattern of C and the A_i, diagonal included */
    int *pj, *pk;
    int *sp, *sj, *spos;     /* the pattern by rows, both triangles: row j, column sj[q], position spos[q] */
    int fixrow; double fixval; int fixcon;   /* the fixed row, its value and its constraint */
    int grow;                /* gauge row (columns 1.. fixed at 0) when there is no fixed row */
    char *zr;                /* rows forced to zero (a constraint X_jj = 0: X >= 0 makes row and column j vanish) */
    int nz;                  /* their number */
    double *R;               /* n x r, by rows */
    int *vid;                /* variable of R[j][c] (by rows), -1 when fixed */
    char *fz;                /* n x r, by rows: entries fixed at 0 (a column kept on its support; NULL: none) */
} IBlk;

typedef struct {
    int m, K, nl, nR, nx;    /* constraints, blocks, LP variables, R variables, all variables */
    IBlk *bk;
    int *poff, E;            /* global positions: poff[k] + p */
    int *pb;                 /* the block of a global position */
    double *cE;              /* objective per position (the factor 2 of an off-diagonal position included) */
    int *arp, *api; double *av;   /* constraints by row over positions */
    int *acp, *aci; double *acv;  /* by position over rows */
    int *lrp, *lri; double *lrv;  /* LP part by row: variable, value */
    int *lcp, *lci; double *lcv;  /* by variable */
    double *cl;              /* LP costs */
    double *b;
    double *rs, cs;          /* row scaling (A_i, b_i times rs_i) and objective scaling (C, c divided by cs) */
    char *dead;              /* constraints X_jj = 0 of a zero row: satisfied identically, set aside in the solve */
    int deepmu;              /* the barrier down to the duality gap (a last round once the certificate holds) */
} IP;

/* ---------------------------------------------------------------------------------------------------- */
typedef struct { int j, k, b; double v; int row; } PTrip;
static int cmp_ptrip(const void *x, const void *y) {
    const PTrip *a = x, *c = y;
    if (a->b != c->b) return a->b - c->b;
    if (a->k != c->k) return a->k - c->k;
    if (a->j != c->j) return a->j - c->j;
    return a->row - c->row;
}
static int cmp_int1(const void *x, const void *y) { const int a = *(const int *)x, c = *(const int *)y; return (a > c) - (a < c); }

static void ip_build(IP *Q, const Problem *P) {
    memset(Q, 0, sizeof *Q);
    const int m = P->m; Q->m = m;
    int K = 0, nl = 0;
    for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_SDP) K++; else nl += P->blk[k].n; }
    Q->K = K; Q->nl = nl;
    Q->bk = iz(sizeof(IBlk) * (K + 1));
    Q->poff = iz(sizeof(int) * (K + 1));
    /* the SDP entries as (block, j <= k, row (-1 objective), value) */
    size_t nt = 0;
    for (int k = 0; k < P->nblk; k++) { const Block *B = &P->blk[k]; if (B->type != BLK_SDP) continue; nt += B->C.ef; for (int t = 0; t < B->ncon; t++) nt += B->A[t].ef; }
    PTrip *T = ix(sizeof(PTrip) * (nt + 1)); size_t q = 0;
    int kb = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; if (B->type != BLK_SDP) continue;
        Q->bk[kb].n = B->n;
        for (int t = -1; t < B->ncon; t++) {
            const SpSym *A = t < 0 ? &B->C : &B->A[t];
            for (int e = 0; e < A->ef; e++) {
                const int i = A->fr[e], j = A->fc[e];
                T[q].b = kb; T[q].j = i < j ? i : j; T[q].k = i < j ? j : i; T[q].v = A->fv[e]; T[q].row = t < 0 ? -1 : B->con[t]; q++;
            }
        }
        kb++;
    }
    nt = q;
    qsort(T, nt, sizeof(PTrip), cmp_ptrip);
    /* positions: distinct (b, j, k), plus every diagonal (the start and the certificate use them) */
    int *cnt = iz(sizeof(int) * (K + 1));
    for (size_t a = 0; a < nt; a++) if (a == 0 || T[a].b != T[a - 1].b || T[a].j != T[a - 1].j || T[a].k != T[a - 1].k) cnt[T[a].b]++;
    /* diagonals not in the data */
    char **hasd = ix(sizeof(char *) * (K + 1));
    for (int k = 0; k < K; k++) hasd[k] = iz(Q->bk[k].n + 1);
    for (size_t a = 0; a < nt; a++) if (T[a].j == T[a].k) hasd[T[a].b][T[a].j] = 1;
    for (int k = 0; k < K; k++) for (int i = 0; i < Q->bk[k].n; i++) if (!hasd[k][i]) cnt[k]++;
    int E = 0;
    for (int k = 0; k < K; k++) { Q->poff[k] = E; E += cnt[k]; IBlk *Bk = &Q->bk[k]; Bk->np = cnt[k]; Bk->pj = ix(sizeof(int) * (cnt[k] + 1)); Bk->pk = ix(sizeof(int) * (cnt[k] + 1)); Bk->np = 0; }
    Q->E = E;
    Q->cE = iz(sizeof(double) * (E + 1));
    Q->pb = ix(sizeof(int) * (E + 1));
    for (int k = 0; k < K; k++) for (int p = 0; p < cnt[k]; p++) Q->pb[Q->poff[k] + p] = k;
    /* entries per position and row: aggregated */
    int *pos = ix(sizeof(int) * (nt + 1));
    for (size_t a = 0; a < nt; a++) {
        IBlk *Bk = &Q->bk[T[a].b];
        if (a == 0 || T[a].b != T[a - 1].b || T[a].j != T[a - 1].j || T[a].k != T[a - 1].k) { Bk->pj[Bk->np] = T[a].j; Bk->pk[Bk->np] = T[a].k; Bk->np++; }
        pos[a] = Q->poff[T[a].b] + Bk->np - 1;
    }
    for (int k = 0; k < K; k++) { IBlk *Bk = &Q->bk[k]; for (int i = 0; i < Bk->n; i++) if (!hasd[k][i]) { Bk->pj[Bk->np] = i; Bk->pk[Bk->np] = i; Bk->np++; } free(hasd[k]); }
    free(hasd); free(cnt);
    /* constraint matrix over positions (by row): sum of the entries of one row at one position */
    int *rc = iz(sizeof(int) * (m + 1)); size_t na = 0;
    for (size_t a = 0; a < nt; a++) {
        if (T[a].row < 0) { Q->cE[pos[a]] += T[a].v; continue; }
        if (a > 0 && T[a - 1].row == T[a].row && pos[a - 1] == pos[a]) continue;
        rc[T[a].row]++; na++;
    }
    Q->arp = ix(sizeof(int) * (m + 1)); Q->arp[0] = 0; for (int i = 0; i < m; i++) Q->arp[i + 1] = Q->arp[i] + rc[i];
    Q->api = ix(sizeof(int) * (na + 1)); Q->av = iz(sizeof(double) * (na + 1));
    int *fill = ix(sizeof(int) * (m + 1)); memcpy(fill, Q->arp, sizeof(int) * m);
    for (size_t a = 0; a < nt; a++) {
        if (T[a].row < 0) continue;
        if (a > 0 && T[a - 1].row == T[a].row && pos[a - 1] == pos[a]) { Q->av[fill[T[a].row] - 1] += T[a].v; continue; }
        Q->api[fill[T[a].row]] = pos[a]; Q->av[fill[T[a].row]] = T[a].v; fill[T[a].row]++;
    }
    free(fill); free(rc); free(pos); free(T);
    /* by position */
    Q->acp = iz(sizeof(int) * (E + 2));
    for (int i = 0; i < m; i++) for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) Q->acp[Q->api[a] + 1]++;
    for (int e = 0; e < E; e++) Q->acp[e + 1] += Q->acp[e];
    Q->aci = ix(sizeof(int) * (na + 1)); Q->acv = ix(sizeof(double) * (na + 1));
    { int *f2 = ix(sizeof(int) * (E + 1)); memcpy(f2, Q->acp, sizeof(int) * E);
      for (int i = 0; i < m; i++) for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) { const int e = Q->api[a]; Q->aci[f2[e]] = i; Q->acv[f2[e]] = Q->av[a]; f2[e]++; }
      free(f2); }
    /* the pattern by rows, both triangles */
    for (int k = 0; k < K; k++) {
        IBlk *Bk = &Q->bk[k]; const int n = Bk->n;
        Bk->sp = iz(sizeof(int) * (n + 1));
        for (int p = 0; p < Bk->np; p++) { Bk->sp[Bk->pj[p] + 1]++; if (Bk->pj[p] != Bk->pk[p]) Bk->sp[Bk->pk[p] + 1]++; }
        for (int i = 0; i < n; i++) Bk->sp[i + 1] += Bk->sp[i];
        Bk->sj = ix(sizeof(int) * (Bk->sp[n] + 1)); Bk->spos = ix(sizeof(int) * (Bk->sp[n] + 1));
        int *f3 = ix(sizeof(int) * (n + 1)); memcpy(f3, Bk->sp, sizeof(int) * n);
        for (int p = 0; p < Bk->np; p++) {
            const int j = Bk->pj[p], kk = Bk->pk[p];
            Bk->sj[f3[j]] = kk; Bk->spos[f3[j]] = p; f3[j]++;
            if (j != kk) { Bk->sj[f3[kk]] = j; Bk->spos[f3[kk]] = p; f3[kk]++; }
        }
        free(f3);
        Bk->fixrow = -1; Bk->grow = -1; Bk->zr = iz(n + 1); Bk->nz = 0;
    }
    /* LP variables */
    Q->cl = iz(sizeof(double) * (nl + 1));
    Q->lcp = iz(sizeof(int) * (nl + 2));
    { int off = 0; size_t ne = 0;
      for (int k = 0; k < P->nblk; k++) { const Block *B = &P->blk[k]; if (B->type == BLK_SDP) continue;
          for (int t = 0; t < B->ncon; t++) for (int e = 0; e < B->A[t].nnz; e++) { Q->lcp[off + B->A[t].row[e] + 1]++; ne++; }
          for (int e = 0; e < B->C.nnz; e++) Q->cl[off + B->C.row[e]] += B->C.val[e];
          off += B->n; }
      for (int j = 0; j < nl; j++) Q->lcp[j + 1] += Q->lcp[j];
      Q->lci = ix(sizeof(int) * (ne + 1)); Q->lcv = ix(sizeof(double) * (ne + 1));
      int *f4 = ix(sizeof(int) * (nl + 1)); memcpy(f4, Q->lcp, sizeof(int) * nl);
      off = 0;
      for (int k = 0; k < P->nblk; k++) { const Block *B = &P->blk[k]; if (B->type == BLK_SDP) continue;
          for (int t = 0; t < B->ncon; t++) for (int e = 0; e < B->A[t].nnz; e++) { const int j = off + B->A[t].row[e]; Q->lci[f4[j]] = B->con[t]; Q->lcv[f4[j]] = B->A[t].val[e]; f4[j]++; }
          off += B->n; }
      free(f4);
      Q->lrp = iz(sizeof(int) * (m + 2));
      for (int j = 0; j < nl; j++) for (int a = Q->lcp[j]; a < Q->lcp[j + 1]; a++) Q->lrp[Q->lci[a] + 1]++;
      for (int i = 0; i < m; i++) Q->lrp[i + 1] += Q->lrp[i];
      Q->lri = ix(sizeof(int) * (ne + 1)); Q->lrv = ix(sizeof(double) * (ne + 1));
      int *f5 = ix(sizeof(int) * (m + 1)); memcpy(f5, Q->lrp, sizeof(int) * m);
      for (int j = 0; j < nl; j++) for (int a = Q->lcp[j]; a < Q->lcp[j + 1]; a++) { const int i = Q->lci[a]; Q->lri[f5[i]] = j; Q->lrv[f5[i]] = Q->lcv[a]; f5[i]++; }
      free(f5); }
    Q->b = ix(sizeof(double) * (m + 1)); memcpy(Q->b, P->b, sizeof(double) * m);
    Q->rs = ix(sizeof(double) * (m + 1)); for (int i = 0; i < m; i++) Q->rs[i] = 1.0;
    Q->cs = 1.0;
    Q->dead = iz(m + 1);
}

static void ip_free(IP *Q) {
    for (int k = 0; k < Q->K; k++) { IBlk *B = &Q->bk[k]; free(B->pj); free(B->pk); free(B->sp); free(B->sj); free(B->spos); free(B->R); free(B->vid); free(B->zr); free(B->fz); }
    free(Q->bk); free(Q->poff); free(Q->pb); free(Q->cE); free(Q->arp); free(Q->api); free(Q->av); free(Q->acp); free(Q->aci); free(Q->acv);
    free(Q->lrp); free(Q->lri); free(Q->lrv); free(Q->lcp); free(Q->lci); free(Q->lcv); free(Q->cl); free(Q->b); free(Q->rs); free(Q->dead);
}

/* rows to unit 2-norm (positions weighted as the entries they stand for), C and c to unit largest entry */
static void ip_equilibrate(IP *Q) {
    for (int i = 0; i < Q->m; i++) {
        double s = 0;   /* the Frobenius norm of the row's matrix: an off-diagonal position holds 2a and stands for a twice */
        for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) { const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; s += Q->av[a] * Q->av[a] * (Q->bk[k].pj[p] == Q->bk[k].pk[p] ? 1.0 : 0.5); }
        for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) s += Q->lrv[a] * Q->lrv[a];
        const double d = s > 0 ? 1.0 / sqrt(s) : 1.0;
        Q->rs[i] = d; Q->b[i] *= d;
        for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) Q->av[a] *= d;
        for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) Q->lrv[a] *= d;
    }
    for (int e = 0; e < Q->E; e++) for (int a = Q->acp[e]; a < Q->acp[e + 1]; a++) Q->acv[a] *= Q->rs[Q->aci[a]];
    for (int j = 0; j < Q->nl; j++) for (int a = Q->lcp[j]; a < Q->lcp[j + 1]; a++) Q->lcv[a] *= Q->rs[Q->lci[a]];
    double cm = 0;
    for (int e = 0; e < Q->E; e++) cm = fmax(cm, fabs(Q->cE[e]));
    for (int j = 0; j < Q->nl; j++) cm = fmax(cm, fabs(Q->cl[j]));
    Q->cs = cm > 0 ? cm : 1.0;
    for (int e = 0; e < Q->E; e++) Q->cE[e] /= Q->cs;
    for (int j = 0; j < Q->nl; j++) Q->cl[j] /= Q->cs;
}

/* ---------------------------------------------------------------------------------------------------- */
/* u (positions) from R; g = A u + A_l x - b; f = cE'u + cl'x */
static void ip_eval(const IP *Q, const double *xl, double *u, double *g, double *f) {
    for (int k = 0; k < Q->K; k++) {
        const IBlk *B = &Q->bk[k]; const int r = B->r; double *uk = u + Q->poff[k];
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < B->np; p++) { const double *a = B->R + (size_t)B->pj[p] * r, *c = B->R + (size_t)B->pk[p] * r; double s = 0; for (int l = 0; l < r; l++) s += a[l] * c[l]; uk[p] = s; }
    }
    if (g) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Q->m; i++) {
            double s = -Q->b[i];
            for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) s += Q->av[a] * u[Q->api[a]];
            for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) s += Q->lrv[a] * xl[Q->lri[a]];
            g[i] = s;
        }
    }
    if (f) { double s = 0; for (int e = 0; e < Q->E; e++) s += Q->cE[e] * u[e]; for (int j = 0; j < Q->nl; j++) s += Q->cl[j] * xl[j]; *f = s; }
}

/* vE = cE + A' lam (positions): S_k = C_k + sum lam_i A_ik with S_jk = vE / 2 off the diagonal */
static void ip_svals(const IP *Q, const double *lam, double *vE) {
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < Q->E; e++) { double s = Q->cE[e]; for (int a = Q->acp[e]; a < Q->acp[e + 1]; a++) s += Q->acv[a] * lam[Q->aci[a]]; vE[e] = s; }
}
/* G = S R for block k (n x r by rows) */
static void ip_sr(const IP *Q, int k, const double *vE, const double *R, double *G) {
    const IBlk *B = &Q->bk[k]; const int r = B->r; const double *vk = vE + Q->poff[k];
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < B->n; j++) {
        double *gj = G + (size_t)j * r; for (int l = 0; l < r; l++) gj[l] = 0;
        for (int q = B->sp[j]; q < B->sp[j + 1]; q++) {
            const int c = B->sj[q], p = B->spos[q]; const double s = c == j ? vk[p] : 0.5 * vk[p];
            const double *rc = R + (size_t)c * r; for (int l = 0; l < r; l++) gj[l] += s * rc[l];
        }
    }
}

/* ---------------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------------- */
/* LDL' without pivoting of a symmetric matrix in upper-triangular CSC (the up-looking algorithm of QDLDL,
 * Stellato et al., OSQP), after an AMD ordering. Pivots of either sign: the count of negative ones is the
 * inertia that the inertia correction tests (the static-sign factorization of sparsechol.c would fail on any
 * wrong-sign pivot, which forces the Hessian block itself to be positive definite) */
#include "amd/amd.h"
typedef struct {
    int n; int *perm, *iperm;          /* perm[k] = original index eliminated k-th */
    int *Ap, *Ai; double *Ax;          /* permuted upper triangle (diagonal included), by columns */
    int *etree, *Lnz, *Lp, *Li; double *Lx, *D, *Dinv;
    int *iw; double *fw; unsigned char *bw;
    int npos;
} LDL;
static int ldl_analyze(LDL *F, int n, const int *deg, int **nbr) {
    memset(F, 0, sizeof *F); F->n = n;
    size_t nz = 0; for (int i = 0; i < n; i++) nz += deg[i];
    int *Cp = ix(sizeof(int) * (n + 1)), *Ci = ix(sizeof(int) * (nz + 1));
    Cp[0] = 0; for (int i = 0; i < n; i++) { memcpy(Ci + Cp[i], nbr[i], sizeof(int) * deg[i]); Cp[i + 1] = Cp[i] + deg[i]; }
    F->perm = ix(sizeof(int) * (n + 1)); F->iperm = ix(sizeof(int) * (n + 1));
    double Control[AMD_CONTROL], Info[AMD_INFO]; amd_defaults(Control); Control[AMD_AGGRESSIVE] = 1;
    const int st = amd_order(n, Cp, Ci, F->perm, Control, Info);
    free(Cp); free(Ci);
    if (st != AMD_OK && st != AMD_OK_BUT_JUMBLED) return 1;
    for (int k = 0; k < n; k++) F->iperm[F->perm[k]] = k;
    /* the permuted upper triangle: column c holds rows r <= c (and the diagonal) */
    F->Ap = iz(sizeof(int) * (n + 2));
    for (int i = 0; i < n; i++) { const int pi = F->iperm[i]; F->Ap[pi + 1]++; for (int q = 0; q < deg[i]; q++) { const int pj = F->iperm[nbr[i][q]]; if (pj < pi) F->Ap[pi + 1]++; } }
    for (int c = 0; c < n; c++) F->Ap[c + 1] += F->Ap[c];
    F->Ai = ix(sizeof(int) * (F->Ap[n] + 1)); F->Ax = iz(sizeof(double) * (F->Ap[n] + 1));
    int *fill = ix(sizeof(int) * (n + 1)); memcpy(fill, F->Ap, sizeof(int) * n);
    for (int i = 0; i < n; i++) { const int pi = F->iperm[i]; F->Ai[fill[pi]++] = pi; for (int q = 0; q < deg[i]; q++) { const int pj = F->iperm[nbr[i][q]]; if (pj < pi) F->Ai[fill[pi]++] = pj; } }
    free(fill);
    /* rows sorted within each column (the diagonal last) */
    for (int c = 0; c < n; c++) qsort(F->Ai + F->Ap[c], F->Ap[c + 1] - F->Ap[c], sizeof(int), cmp_int1);
    /* elimination tree and column counts */
    F->etree = ix(sizeof(int) * (n + 1)); F->Lnz = iz(sizeof(int) * (n + 1)); int *work = ix(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) { work[i] = 0; F->etree[i] = -1; }
    for (int j = 0; j < n; j++) {
        work[j] = j;
        for (int p = F->Ap[j]; p < F->Ap[j + 1]; p++) {
            int i = F->Ai[p];
            while (work[i] != j) { if (F->etree[i] == -1) F->etree[i] = j; F->Lnz[i]++; work[i] = j; i = F->etree[i]; }
        }
    }
    free(work);
    F->Lp = ix(sizeof(int) * (n + 1)); F->Lp[0] = 0;
    for (int i = 0; i < n; i++) F->Lp[i + 1] = F->Lp[i] + F->Lnz[i];
    F->Li = ix(sizeof(int) * ((size_t)F->Lp[n] + 1)); F->Lx = ix(sizeof(double) * ((size_t)F->Lp[n] + 1));
    F->D = ix(sizeof(double) * (n + 1)); F->Dinv = ix(sizeof(double) * (n + 1));
    F->iw = ix(sizeof(int) * (3 * (size_t)n + 1)); F->fw = ix(sizeof(double) * (n + 1)); F->bw = ix(n + 1);
    return 0;
}
static size_t ldl_offset(const LDL *F, int r, int c) {   /* original indices */
    int pr = F->iperm[r], pc = F->iperm[c]; if (pr > pc) { const int t = pr; pr = pc; pc = t; }
    int lo = F->Ap[pc], hi = F->Ap[pc + 1] - 1;
    while (lo <= hi) { const int mid = (lo + hi) >> 1; if (F->Ai[mid] == pr) return (size_t)mid; if (F->Ai[mid] < pr) lo = mid + 1; else hi = mid - 1; }
    return SIZE_MAX;
}
/* returns the number of positive pivots, -1 on a zero (or tiny, relative to tinyrel times the largest diagonal) pivot */
static int ldl_factor(LDL *F, double tinyrel) {
    const int n = F->n;
    int *yIdx = F->iw, *elim = F->iw + n, *nextc = F->iw + 2 * (size_t)n;
    double *yv = F->fw; unsigned char *mk = F->bw;
    const double tiny = tinyrel;   /* absolute: a zero pivot (the regularizations keep the others away from it) */
    for (int i = 0; i < n; i++) { mk[i] = 0; yv[i] = 0; F->D[i] = 0; nextc[i] = F->Lp[i]; }
    int npos = 0;
    for (int k = 0; k < n; k++) {
        int nnzY = 0;
        for (int p = F->Ap[k]; p < F->Ap[k + 1]; p++) {
            const int b = F->Ai[p];
            if (b == k) { F->D[k] = F->Ax[p]; continue; }
            yv[b] = F->Ax[p];
            int nx = b;
            if (!mk[nx]) {
                mk[nx] = 1; elim[0] = nx; int nE = 1;
                nx = F->etree[b];
                while (nx != -1 && nx < k) { if (mk[nx]) break; mk[nx] = 1; elim[nE++] = nx; nx = F->etree[nx]; }
                while (nE) yIdx[nnzY++] = elim[--nE];
            }
        }
        for (int i = nnzY - 1; i >= 0; i--) {
            const int c = yIdx[i]; const int t = nextc[c]; const double yc = yv[c];
            for (int j = F->Lp[c]; j < t; j++) yv[F->Li[j]] -= F->Lx[j] * yc;
            F->Li[t] = k; F->Lx[t] = yc * F->Dinv[c];
            F->D[k] -= yc * F->Lx[t];
            nextc[c]++; yv[c] = 0; mk[c] = 0;
        }
        if (!(fabs(F->D[k]) > tiny) || !isfinite(F->D[k])) return -1;
        if (F->D[k] > 0) npos++;
        F->Dinv[k] = 1.0 / F->D[k];
    }
    F->npos = npos;
    return npos;
}
static void ldl_solve(const LDL *F, double *b, double *x) {   /* b, x original order (b kept) */
    const int n = F->n;
    for (int k = 0; k < n; k++) x[k] = b[F->perm[k]];
    for (int i = 0; i < n; i++) { const double xi = x[i]; for (int j = F->Lp[i]; j < F->Lp[i + 1]; j++) x[F->Li[j]] -= F->Lx[j] * xi; }
    for (int i = 0; i < n; i++) x[i] *= F->Dinv[i];
    for (int i = n - 1; i >= 0; i--) { double s = x[i]; for (int j = F->Lp[i]; j < F->Lp[i + 1]; j++) s -= F->Lx[j] * x[F->Li[j]]; x[i] = s; }
    for (int k = 0; k < n; k++) b[F->perm[k]] = x[k];
}
static void ldl_mv(const LDL *F, const double *x, double *y) {   /* y = A x, original order */
    const int n = F->n;
    for (int i = 0; i < n; i++) y[i] = 0;
    for (int c = 0; c < n; c++) { const int oc = F->perm[c];
        for (int p = F->Ap[c]; p < F->Ap[c + 1]; p++) { const int r = F->Ai[p], orr = F->perm[r]; const double a = F->Ax[p];
            y[orr] += a * x[oc]; if (r != c) y[oc] += a * x[orr]; } }
}
static void ldl_free(LDL *F) {
    free(F->perm); free(F->iperm); free(F->Ap); free(F->Ai); free(F->Ax); free(F->etree); free(F->Lnz); free(F->Lp); free(F->Li); free(F->Lx);
    free(F->D); free(F->Dinv); free(F->iw); free(F->fw); free(F->bw); memset(F, 0, sizeof *F);
}


/* the KKT system: variables (R free entries, LP), then rows; pattern analysed once per rank */
typedef struct {
    int N, nx, m;
    LDL L;
    size_t *offH;  int nH;   /* Hessian entries: per block, column, pattern position (j <= k, both free) */
    int *hk, *hp;            /* their block and position (the value is 2 S_jk) */
    size_t *offJ; int nJ;    /* Jacobian entries: (row, variable) pairs in a fixed order of generation */
    size_t *offL; int nL;    /* LP entries (row, variable) */
    size_t *offD;            /* the diagonal of every unknown */
    double *work, *rhs, *sol, *res;
} KKT;

typedef struct { int *a; int n, cap; } IVec;
static void iv_push(IVec *v, int x) { if (v->n == v->cap) { v->cap = v->cap ? 2 * v->cap : 8; v->a = realloc(v->a, sizeof(int) * v->cap); if (!v->a) { fprintf(stderr, "brisk: out of memory (-bm 2)\n"); exit(1); } } v->a[v->n++] = x; }


/* the Jacobian rows: for row i, position (j, k), column l: d/dR[j][l] = a R[k][l], d/dR[k][l] = a R[j][l] (2 a R[j][l] when j = k) */
static void kkt_setup(KKT *Kt, IP *Q) {
    memset(Kt, 0, sizeof *Kt);
    const int m = Q->m, nx = Q->nx, N = nx + m;
    Kt->N = N; Kt->nx = nx; Kt->m = m;
    const int *pb = Q->pb;
    IVec *adj = iz(sizeof(IVec) * (N + 1));
    /* Hessian */
    int nH = 0;
    for (int k = 0; k < Q->K; k++) { const IBlk *B = &Q->bk[k];
        for (int p = 0; p < B->np; p++) for (int l = 0; l < B->r; l++) {
            const int va = B->vid[(size_t)B->pj[p] * B->r + l], vb = B->vid[(size_t)B->pk[p] * B->r + l];
            if (va < 0 || vb < 0) continue;
            nH++;
            if (va != vb) { iv_push(&adj[va], vb); iv_push(&adj[vb], va); }
        } }
    Kt->nH = nH; Kt->hk = ix(sizeof(int) * (nH + 1)); Kt->hp = ix(sizeof(int) * (nH + 1)); Kt->offH = ix(sizeof(size_t) * (nH + 1));
    /* Jacobian */
    int nJ = 0;
    for (int i = 0; i < m; i++) for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
        const int e = Q->api[a], k = pb[e]; const IBlk *B = &Q->bk[k]; const int p = e - Q->poff[k];
        for (int l = 0; l < B->r; l++) {
            const int va = B->vid[(size_t)B->pj[p] * B->r + l], vb = B->vid[(size_t)B->pk[p] * B->r + l];
            if (va >= 0) { iv_push(&adj[nx + i], va); iv_push(&adj[va], nx + i); nJ++; }
            if (B->pj[p] != B->pk[p] && vb >= 0) { iv_push(&adj[nx + i], vb); iv_push(&adj[vb], nx + i); nJ++; }
        }
    }
    Kt->nJ = nJ; Kt->offJ = ix(sizeof(size_t) * (nJ + 1));
    int nL = 0;
    for (int i = 0; i < m; i++) for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) { const int v = Q->nR + Q->lri[a]; iv_push(&adj[nx + i], v); iv_push(&adj[v], nx + i); nL++; }
    Kt->nL = nL; Kt->offL = ix(sizeof(size_t) * (nL + 1));
    /* unique neighbours */
    int *deg = ix(sizeof(int) * (N + 1)); int **nbr = ix(sizeof(int *) * (N + 1));
    size_t tot = 0;
    for (int v = 0; v < N; v++) {
        IVec *A = &adj[v];
        if (A->n > 1) {
            /* sort and unique */
            int *t = A->a; const int n0 = A->n;
            /* insertion sort for short lists, qsort otherwise */
            if (n0 < 32) { for (int x = 1; x < n0; x++) { const int y = t[x]; int z = x - 1; while (z >= 0 && t[z] > y) { t[z + 1] = t[z]; z--; } t[z + 1] = y; } }
            else qsort(t, n0, sizeof(int), cmp_int1);
            int w = 0; for (int x = 0; x < n0; x++) if (w == 0 || t[w - 1] != t[x]) t[w++] = t[x];
            A->n = w;
        }
        deg[v] = A->n; nbr[v] = A->a ? A->a : (int *)ix(sizeof(int)); tot += A->n;
    }
    if (ldl_analyze(&Kt->L, N, deg, nbr) != 0) { fprintf(stderr, "brisk: -bm 2: the KKT system's ordering failed\n"); exit(1); }
    for (int v = 0; v < N; v++) free(nbr[v]);
    free(nbr); free(deg); free(adj);
    (void)tot;
    /* offsets */
    int h = 0;
    for (int k = 0; k < Q->K; k++) { const IBlk *B = &Q->bk[k];
        for (int p = 0; p < B->np; p++) for (int l = 0; l < B->r; l++) {
            const int va = B->vid[(size_t)B->pj[p] * B->r + l], vb = B->vid[(size_t)B->pk[p] * B->r + l];
            if (va < 0 || vb < 0) continue;
            Kt->hk[h] = k; Kt->hp[h] = p; Kt->offH[h] = ldl_offset(&Kt->L, va, vb); h++;
        } }
    int jq = 0;
    for (int i = 0; i < m; i++) for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
        const int e = Q->api[a], k = pb[e]; const IBlk *B = &Q->bk[k]; const int p = e - Q->poff[k];
        for (int l = 0; l < B->r; l++) {
            const int va = B->vid[(size_t)B->pj[p] * B->r + l], vb = B->vid[(size_t)B->pk[p] * B->r + l];
            if (va >= 0) Kt->offJ[jq++] = ldl_offset(&Kt->L, nx + i, va);
            if (B->pj[p] != B->pk[p] && vb >= 0) Kt->offJ[jq++] = ldl_offset(&Kt->L, nx + i, vb);
        }
    }
    int lq = 0;
    for (int i = 0; i < m; i++) for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) Kt->offL[lq++] = ldl_offset(&Kt->L, nx + i, Q->nR + Q->lri[a]);
    Kt->offD = ix(sizeof(size_t) * (N + 1));
    for (int v = 0; v < N; v++) Kt->offD[v] = ldl_offset(&Kt->L, v, v);
    Kt->work = ix(sizeof(double) * (N + 1)); Kt->rhs = ix(sizeof(double) * (N + 1)); Kt->sol = ix(sizeof(double) * (N + 1)); Kt->res = ix(sizeof(double) * (N + 1));
}
static void kkt_free(KKT *Kt) {
    ldl_free(&Kt->L);
    free(Kt->offH); free(Kt->hk); free(Kt->hp); free(Kt->offJ); free(Kt->offL); free(Kt->offD); free(Kt->work); free(Kt->rhs); free(Kt->sol); free(Kt->res);
    memset(Kt, 0, sizeof *Kt);
}

/* values: H = 2 S (W = 0 when hess = 0: the restoration's least-norm steps), J, A_l, diagonals.
 * dR: the regularization of the R variables; dl[j]: the LP diagonal (Sigma + dw); dc: the rows' -dc */
static void kkt_assemble_d(KKT *Kt, const IP *Q, const double *vE, int hess, double dR, const double *dl, double dc, const double *drow);
static void kkt_assemble(KKT *Kt, const IP *Q, const double *vE, int hess, double dR, const double *dl, double dc) { kkt_assemble_d(Kt, Q, vE, hess, dR, dl, dc, NULL); }
static void kkt_assemble_d_(KKT *Kt, const IP *Q, const double *vE, int hess, double dR, const double *dl, double dc, const double *drow);
static double g_t_asm2 = 0;
static void kkt_assemble_d(KKT *Kt, const IP *Q, const double *vE, int hess, double dR, const double *dl, double dc, const double *drow) {
    const double t = wtime(); kkt_assemble_d_(Kt, Q, vE, hess, dR, dl, dc, drow); g_t_asm2 += wtime() - t;
}
static void kkt_assemble_d_(KKT *Kt, const IP *Q, const double *vE, int hess, double dR, const double *dl, double dc, const double *drow) {
    double *V = Kt->L.Ax;
    memset(V, 0, sizeof(double) * Kt->L.Ap[Kt->L.n]);
    if (hess) for (int h = 0; h < Kt->nH; h++) {
        const IBlk *B = &Q->bk[Kt->hk[h]]; const int p = Kt->hp[h];
        const double s = vE[Q->poff[Kt->hk[h]] + p];
        V[Kt->offH[h]] += B->pj[p] == B->pk[p] ? 2.0 * s : s;   /* 2 S_jk; S_jk = vE / 2 off the diagonal */
    }
    /* Jacobian: same generation order as in kkt_setup */
    int jq = 0;
    for (int i = 0; i < Q->m; i++) for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
        const int e = Q->api[a], k = Q->pb[e];
        const IBlk *B = &Q->bk[k]; const int p = e - Q->poff[k], r = B->r;
        const int j = B->pj[p], kk = B->pk[p];
        const double *Rj = B->R + (size_t)j * r, *Rk = B->R + (size_t)kk * r;
        for (int l = 0; l < r; l++) {
            const int va = B->vid[(size_t)j * r + l], vb = B->vid[(size_t)kk * r + l];
            if (j == kk) { if (va >= 0) V[Kt->offJ[jq++]] += 2.0 * Q->av[a] * Rj[l]; }
            else {
                if (va >= 0) V[Kt->offJ[jq++]] += Q->av[a] * Rk[l];
                if (vb >= 0) V[Kt->offJ[jq++]] += Q->av[a] * Rj[l];
            }
        }
    }
    int lq = 0;
    for (int i = 0; i < Q->m; i++) for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) V[Kt->offL[lq++]] += Q->lrv[a];
    for (int v = 0; v < Q->nR; v++) V[Kt->offD[v]] += dR;
    for (int j = 0; j < Q->nl; j++) V[Kt->offD[Q->nR + j]] += dl[j];
    for (int i = 0; i < Q->m; i++) V[Kt->offD[Kt->nx + i]] -= Q->dead[i] ? 1.0 : drow ? drow[i] + dc : dc;
}

/* solve with two steps of iterative refinement against the assembled matrix */
static int g_kkt_dbg = -1;
static void kkt_solve_(KKT *Kt, const double *rhs, double *sol);
static double g_t_sol2 = 0;
static void kkt_solve(KKT *Kt, const double *rhs, double *sol) { const double t = wtime(); kkt_solve_(Kt, rhs, sol); g_t_sol2 += wtime() - t; }
static void kkt_solve_(KKT *Kt, const double *rhs, double *sol) {
    if (g_kkt_dbg < 0) g_kkt_dbg = getenv("BRISK_BM2KKTDBG") != NULL;
    const int N = Kt->N;
    memcpy(sol, rhs, sizeof(double) * N);
    ldl_solve(&Kt->L, sol, Kt->work);
    for (int it = 0; it < 3; it++) {
        ldl_mv(&Kt->L, sol, Kt->res);
        double rn = 0, bn = 0;
        for (int v = 0; v < N; v++) { Kt->res[v] = rhs[v] - Kt->res[v]; rn = fmax(rn, fabs(Kt->res[v])); bn = fmax(bn, fabs(rhs[v])); }
        if (rn <= 1e-14 * fmax(bn, 1e-300)) break;
        ldl_solve(&Kt->L, Kt->res, Kt->work);
        for (int v = 0; v < N; v++) sol[v] += Kt->res[v];
    }
    if (g_kkt_dbg) {
        ldl_mv(&Kt->L, sol, Kt->res); double rn = 0, bn = 0, sn = 0;
        for (int v = 0; v < N; v++) { rn = fmax(rn, fabs(rhs[v] - Kt->res[v])); bn = fmax(bn, fabs(rhs[v])); sn = fmax(sn, fabs(sol[v])); }
        double lmax = 0; for (size_t q = 0; q < (size_t)Kt->L.Lp[Kt->L.n]; q++) lmax = fmax(lmax, fabs(Kt->L.Lx[q]));
        printf("        [kkt solve: residual %.1e of rhs %.1e, |sol| %.1e, max |L| %.1e]\n", rn, bn, sn, lmax);
    }
}
/* 0 when the factorization has the inertia (nx positive, m negative pivots) */
static double g_t_fac = 0;   /* time per part (BRISK_BM2PROF) */
static int kkt_factor(KKT *Kt) { const double t = wtime(); const int np = ldl_factor(&Kt->L, 0.0); g_t_fac += wtime() - t; return np < 0 ? 1000000000 : np - Kt->nx; }   /* 0: the inertia is right */

/* ---------------------------------------------------------------------------------------------------- */
/* the free entries of R and the LP variables <-> the vector x */
static void x_get(const IP *Q, const double *xl, double *x) {
    for (int k = 0; k < Q->K; k++) { const IBlk *B = &Q->bk[k]; const size_t nr = (size_t)B->n * B->r; for (size_t q = 0; q < nr; q++) if (B->vid[q] >= 0) x[B->vid[q]] = B->R[q]; }
    memcpy(x + Q->nR, xl, sizeof(double) * Q->nl);
}
static void x_set(IP *Q, const double *x, double *xl) {
    for (int k = 0; k < Q->K; k++) { IBlk *B = &Q->bk[k]; const size_t nr = (size_t)B->n * B->r; for (size_t q = 0; q < nr; q++) if (B->vid[q] >= 0) B->R[q] = x[B->vid[q]]; }
    memcpy(xl, x + Q->nR, sizeof(double) * Q->nl);
}
/* the variables of R: free entries numbered by blocks, rows, columns; the fixed row, else the gauge row's columns 1.. */
static void ip_number(IP *Q) {
    int v = 0;
    for (int k = 0; k < Q->K; k++) {
        IBlk *B = &Q->bk[k]; const int r = B->r;
        free(B->vid); B->vid = ix(sizeof(int) * ((size_t)B->n * r + 1));
        for (int j = 0; j < B->n; j++) for (int l = 0; l < r; l++) {
            const int fixed = j == B->fixrow || (j == B->grow && l >= 1) || B->zr[j] || (B->fz && B->fz[(size_t)j * r + l]);
            B->vid[(size_t)j * r + l] = fixed ? -1 : v++;
        }
    }
    Q->nR = v; Q->nx = v + Q->nl;
}

/* gradient of the Lagrangian f + lam'g - z'x at the current point, and of f alone (gf) */
static void ip_grad(IP *Q, const double *vE, const double *vC, const double *lam, const double *z, double *gL, double *gf, double **Gk) {
    for (int k = 0; k < Q->K; k++) {
        IBlk *B = &Q->bk[k]; const int r = B->r; const size_t nr = (size_t)B->n * r;
        ip_sr(Q, k, vE, B->R, Gk[k]);
        for (size_t q = 0; q < nr; q++) if (B->vid[q] >= 0) gL[B->vid[q]] = 2.0 * Gk[k][q];
        if (gf) { ip_sr(Q, k, vC, B->R, Gk[k]); for (size_t q = 0; q < nr; q++) if (B->vid[q] >= 0) gf[B->vid[q]] = 2.0 * Gk[k][q]; }
    }
    for (int j = 0; j < Q->nl; j++) {
        double s = Q->cl[j]; for (int a = Q->lcp[j]; a < Q->lcp[j + 1]; a++) s += Q->lcv[a] * lam[Q->lci[a]];
        gL[Q->nR + j] = s - z[j];
        if (gf) gf[Q->nR + j] = Q->cl[j];
    }
}

typedef struct { double th, ph; } FEnt;
typedef struct { int its, status, nfac, nres; double f, E0; } IStat;

static double c0avg(const double *x, const double *z, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i] * z[i]; return n > 0 ? s / n : 0; }
static double nrm1(const double *a, int n) { double s = 0; for (int i = 0; i < n; i++) s += fabs(a[i]); return s; }
static double nrmi(const double *a, int n) { double s = 0; for (int i = 0; i < n; i++) s = fmax(s, fabs(a[i])); return s; }

static int resto_pen(IP *Q, KKT *Kt, double *x, double *xl, double *lam, double *z, double mu, double kap_d, double th_R,
                     const double *fth, const double *fph, int nf, double th_max, int verbose, double t0, int *nfac);
/* the primal-dual interior-point method (Waechter-Biegler, monotone barrier, filter line search with second-order
 * corrections, a least-squares restoration). Returns 0 converged, 1 iteration limit, 2 restoration failed, 3 time */
static int ipm(IP *Q, KKT *Kt, double *xl, double *lam, double *z, double *mu_io, double tol, int maxit, int verbose, double tlim, double t0, IStat *st) {
    const int m = Q->m, nx = Q->nx, nR = Q->nR, nl = Q->nl, N = Kt->N;
    double mu = *mu_io;
    const double kap_eps = 10, kap_mu = 0.2, th_mu = 1.5, tau_min = 0.99, s_th = 1.1, s_ph = 2.3, dlt = 1.0, eta = 1e-8, g_th = 1e-5, g_ph = 1e-8, g_al = 0.05, kap_sig = 1e10, smax = 100;
    const double dwmin = getenv("BRISK_BM2DWMIN") ? atof(getenv("BRISK_BM2DWMIN")) : 1e-8;
    const double dcfloor = getenv("BRISK_BM2DC") ? atof(getenv("BRISK_BM2DC")) : 0.0;
    const double kap_d = getenv("BRISK_BM2KD") ? atof(getenv("BRISK_BM2KD")) : 1e-5;   /* the damping of one-sided bounds (Waechter-Biegler 3.7) */
    double *u = ix(sizeof(double) * (Q->E + 1)), *g = ix(sizeof(double) * (m + 1)), *vE = ix(sizeof(double) * (Q->E + 1));
    double *x = ix(sizeof(double) * (nx + 1)), *xt = ix(sizeof(double) * (nx + 1)), *xlt = ix(sizeof(double) * (nl + 1));
    double *gL = ix(sizeof(double) * (nx + 1)), *gf = ix(sizeof(double) * (nx + 1)), *dl = ix(sizeof(double) * (nl + 1));
    double *ut = ix(sizeof(double) * (Q->E + 1)), *gt = ix(sizeof(double) * (m + 1)), *dz = ix(sizeof(double) * (nl + 1));
    double *rhs = ix(sizeof(double) * (N + 1)), *sol = ix(sizeof(double) * (N + 1)), *sol2 = ix(sizeof(double) * (N + 1)), *csoc = ix(sizeof(double) * (m + 1));
    double **Gk = ix(sizeof(double *) * (Q->K + 1));
    for (int k = 0; k < Q->K; k++) Gk[k] = ix(sizeof(double) * ((size_t)Q->bk[k].n * Q->bk[k].r + 1));
    FEnt *filt = NULL; int nf = 0, capf = 0;
    double dw_last = 0, f = 0, fT0 = 0, Eref = 0; int nstall = 0, freemode = getenv("BRISK_BM2MU") ? atoi(getenv("BRISK_BM2MU")) : 1;
    int status = 1, it, nfac = 0, nres = 0;
    x_get(Q, xl, x);
    ip_eval(Q, xl, u, g, &f);
    const double th0 = nrm1(g, m), th_max = 1e4 * fmax(1.0, th0), th_min = 1e-4 * fmax(1.0, th0);
    const double mu_max = 1e3 * fmax(c0avg(xl, z, nl), mu);   /* Ipopt's mu_max_fact */
    if (getenv("BRISK_BM2DUMPG0")) { FILE *fg = fopen(getenv("BRISK_BM2DUMPG0"), "w"); if (fg) { for (int i = 0; i < m; i++) fprintf(fg, "%.17g\n", g[i]); fclose(fg); } }
    if (getenv("BRISK_BM2G0")) { /* the largest violations at the start */
        for (int t = 0; t < 4; t++) { int am = -1; double mx = -1; for (int i = 0; i < m; i++) if (fabs(g[i]) > mx) { int seen = 0; (void)seen; mx = fabs(g[i]); am = i; } if (am < 0) break;
            printf("   [g0: row %d value %.6e]\n", am, g[am]); g[am] = -g[am] * 0 + (g[am] > 0 ? -0.0 : 0.0) + g[am] * 1e-300; }
        ip_eval(Q, xl, u, g, &f);
    }
    int nacc = 0;   /* consecutive iterations at the acceptable level */
    double Ewin = INFINITY, Eprev = INFINITY, fwin = INFINITY; int itchk = -1;
    double Ebest = INFINITY, mub = mu, *xb = ix(sizeof(double) * (nx + 1)), *lb_ = ix(sizeof(double) * (m + 1)), *zb = ix(sizeof(double) * (nl + 1));
    double dwforce = 0, dc_last = 0; int nretry = 0;
    const int dumpkkt = getenv("BRISK_BM2DUMPKKT") ? atoi(getenv("BRISK_BM2DUMPKKT")) : -1;
    const int lsm = getenv("BRISK_BM2LSM") != NULL; int lsm_it = -1, lsm_last = -100, nlsm = 0;
    const double sigmax = getenv("BRISK_BM2SIGMAX") ? atof(getenv("BRISK_BM2SIGMAX")) : 100.0;
    if (nrmi(lam, m) == 0 && th0 > 1e-6 && getenv("BRISK_BM2PH1")) {
        /* phase 1 (cold start): least-norm Gauss-Newton steps on g(x) = 0, x >= 0 kept by the fraction to the boundary,
         * down to a hundredth of the start's infeasibility: the method then starts nearly feasible */
        double rho = 1e-6, thc = th0;
        for (int rit = 0; rit < 50; rit++) {
            for (int j = 0; j < nl; j++) dl[j] = rho + 1.0 / (xl[j] * xl[j]) * 1e-6;
            kkt_assemble(Kt, Q, Q->cE, 0, rho, dl, 1.0);
            nfac++;
            if (kkt_factor(Kt) != 0) { rho *= 10; continue; }
            for (int v = 0; v < nx; v++) rhs[v] = 0;
            for (int i = 0; i < m; i++) rhs[nx + i] = -g[i];
            kkt_solve(Kt, rhs, sol);
            double a2 = 1.0; for (int j = 0; j < nl; j++) { const double d = sol[nR + j]; if (d < 0) a2 = fmin(a2, -0.99 * xl[j] / d); }
            int took = 0; double thn = thc;
            for (int b2 = 0; b2 < 30; b2++, a2 *= 0.5) {
                for (int v = 0; v < nx; v++) xt[v] = x[v] + a2 * sol[v];
                x_set(Q, xt, xlt); ip_eval(Q, xlt, ut, gt, &fT0);
                thn = nrm1(gt, m);
                if (thn < (1 - 1e-4 * a2) * thc) { took = 1; break; }
            }
            if (!took) { x_set(Q, x, xl); rho *= 10; if (rho > 1e6) break; continue; }
            memcpy(x, xt, sizeof(double) * nx); x_set(Q, x, xl);
            ip_eval(Q, xl, u, g, &f); thc = thn; rho = fmax(1e-10, rho / 10);
            if (verbose >= 1) printf("   phase 1: %2d theta %.2e (step %.2e)\n", rit, thc, a2);
            if (thc <= 1e-2 * th0 || thc <= 1e-8) break;
        }
        for (int j = 0; j < nl; j++) z[j] = fmax(z[j], 1e-2);
    }
    if (nrmi(lam, m) == 0 && !getenv("BRISK_BM2LAM0")) {
        /* the least-squares multipliers (Ipopt's start, kept when they are at most 1000): min |grad f - z + J'lam| */
        ip_svals(Q, lam, vE);
        ip_grad(Q, vE, Q->cE, lam, z, gL, NULL, Gk);
        for (int j = 0; j < nl; j++) dl[j] = 1.0;
        kkt_assemble(Kt, Q, vE, 0, 1.0, dl, 1e-10);
        if (kkt_factor(Kt) == 0) {
            for (int v = 0; v < nx; v++) rhs[v] = -gL[v];
            for (int i = 0; i < m; i++) rhs[nx + i] = 0;
            kkt_solve(Kt, rhs, sol);
            if (nrmi(sol + nx, m) <= 1000) for (int i = 0; i < m; i++) lam[i] = sol[nx + i];
        }
        nfac++;
    }
    for (it = 0; it < maxit; it++) {
        ip_svals(Q, lam, vE);
        ip_grad(Q, vE, Q->cE, lam, z, gL, gf, Gk);
        /* optimality error (scaled as Ipopt's) */
        const double sd = fmax(smax, (nrm1(lam, m) + nrm1(z, nl)) / (m + nl)) / smax, sc = fmax(smax, nrm1(z, nl) / (nl > 0 ? nl : 1)) / smax;
        double dinf = nrmi(gL, nx), pinf = nrmi(g, m), c0 = 0, cmu = 0;
        double sxz = 0;
        for (int j = 0; j < nl; j++) { c0 = fmax(c0, fabs(xl[j] * z[j])); cmu = fmax(cmu, fabs(xl[j] * z[j] - mu)); sxz += xl[j] * z[j]; }
        const double E0 = fmax(fmax(dinf / sd, pinf), c0 / sc);
        /* the barrier's duality gap is sum x_j z_j, about nl mu: the floor of mu and the stop keep it within tol of the
         * objective (relative), as the gap to the lower bound is measured; tol / 10 per pair alone left 1e-4 at
         * tol 1e-5 with 1e4 inequalities */
        const double gtol = tol * (1 + fabs(f * Q->cs)) / Q->cs, mumin = !Q->deepmu || nl == 0 ? tol / 10 : fmin(tol / 10, 0.1 * gtol / nl);
        double logb = 0; for (int j = 0; j < nl; j++) logb += log(xl[j]);
        const double th = nrm1(g, m);
        if (verbose >= 1) printf("  %3d f %.10e pinf %.1e dinf %.1e comp %.1e mu %.1e dw %.1e  %.1fs\n", it, f * Q->cs, pinf, dinf, c0, mu, dw_last, wtime() - t0);
        st->E0 = E0;
        if (E0 < Ebest) {   /* the best iterate so far: what a failure returns */
            Ebest = E0; memcpy(xb, x, sizeof(double) * nx); memcpy(lb_, lam, sizeof(double) * m); memcpy(zb, z, sizeof(double) * nl); mub = mu;
        }
        if (it == 0 && getenv("BRISK_BM2CHECK")) {   /* finite-difference check of J and W = 2 (S (x) I) */
            double *w = ix(sizeof(double) * (N + 1)), *Kw = ix(sizeof(double) * (N + 1)), *gL2 = ix(sizeof(double) * (nx + 1)), *zz = ix(sizeof(double) * (nl + 1));
            unsigned hh = 12345u; for (int v = 0; v < nx; v++) { hh = hh * 1103515245u + 12345u; w[v] = ((hh >> 8) & 0xffff) / 65535.0 - 0.5; }
            for (int i = 0; i < m; i++) w[nx + i] = 0;
            for (int j = 0; j < nl; j++) dl[j] = 0;
            kkt_assemble(Kt, Q, vE, 1, 0.0, dl, 0.0);
            ldl_mv(&Kt->L, w, Kw);
            const double hstep = 1e-6;
            for (int v = 0; v < nx; v++) xt[v] = x[v] + hstep * w[v];
            x_set(Q, xt, xlt); ip_eval(Q, xlt, ut, gt, &fT0);
            ip_grad(Q, vE, Q->cE, lam, z, gL2, NULL, Gk);
            double eJ = 0, nJ = 0, eW = 0, nW = 0;
            for (int i = 0; i < m; i++) { const double fd = (gt[i] - g[i]) / hstep; eJ = fmax(eJ, fabs(fd - Kw[nx + i])); nJ = fmax(nJ, fabs(fd)); }
            for (int v = 0; v < nR; v++) { const double fd = (gL2[v] - gL[v]) / hstep; eW = fmax(eW, fabs(fd - Kw[v])); nW = fmax(nW, fabs(fd)); }
            printf("   [check: J error %.2e of %.2e, W error %.2e of %.2e]\n", eJ, nJ, eW, nW);
            x_set(Q, x, xl);
            free(w); free(Kw); free(gL2); free(zz);
        }
        if (E0 <= tol && (!Q->deepmu || sxz <= gtol)) { status = 0; break; }
        if (E0 <= fmax(1e2 * tol, 1e-6)) { if (++nacc >= 10) { status = 0; break; } } else nacc = 0;
        /* a stall: the best optimality error of 200 iterations not a tenth below that of the 200 before while the
         * objective stays put (1e-6); the round ends and the certificate, the lower bound and the rank decide */
        Ewin = fmin(Ewin, E0);
        if (it % 200 == 199 && it != itchk) {   /* once per iteration number (a retried line search repeats it) */
            itchk = it;
            if (Ewin > 0.9 * Eprev && fabs(f - fwin) <= 1e-6 * (1 + fabs(f)) && !getenv("BRISK_BM2NOSTALL")) {
                if (verbose >= 1) printf("      stalled (best error %.2e, before %.2e; objective unchanged)\n", Ewin, Eprev);
                status = 1; break;
            }
            Eprev = Ewin; Ewin = INFINITY; fwin = f;
        }
        if (tlim > 0 && wtime() - t0 > tlim) { status = 3; break; }
        /* least-squares multipliers in a crawl: when the last iteration needed the inertia correction and the dual
         * infeasibility dominates, the multipliers (moved by the same short steps as x) keep S = C + A*lam indefinite
         * and the steps short (case4661: 1,900 iterations at one mu with corrections of 10-500); their least-squares
         * estimate at the current point, min |grad f - z + J'lam| (Ipopt's start), is taken when it halves the dual
         * infeasibility; at most every fifth iteration (an experiment, BRISK_BM2LSM=1: case588 to OPTIMAL in
         * 13 s from the iteration limit, but case1888 fails) */
        if (lsm && it > 0 && it != lsm_it && it - lsm_last >= 5 && dw_last > 1e-6 && dinf / sd > 10 * fmax(pinf, c0 / sc)) {
            lsm_it = it; lsm_last = it;
            for (int j = 0; j < nl; j++) dl[j] = 1.0;
            kkt_assemble(Kt, Q, vE, 0, 1.0, dl, 1e-10);
            nfac++;
            if (kkt_factor(Kt) == 0) {
                for (int v = 0; v < nx; v++) rhs[v] = -gL[v];
                for (int i = 0; i < m; i++) rhs[nx + i] = 0;
                kkt_solve(Kt, rhs, sol);
                const double wn = nrmi(sol, nx);   /* the new dual infeasibility: gL + J'dlam = -w */
                if (wn < 0.5 * dinf) {
                    for (int i = 0; i < m; i++) lam[i] += sol[nx + i];
                    nlsm++;
                    if (verbose >= 2) printf("      least-squares multipliers: dual infeasibility %.1e -> %.1e\n", dinf, wn);
                    it--; continue;
                }
            }
        }
        /* barrier parameter: monotone (Fiacco-McCormick) or, in the free mode, by a probing step */
        if (!freemode) {
            double Emu = fmax(fmax(dinf / sd, pinf), cmu / sc);
            while (Emu <= kap_eps * mu && mu > mumin) {
                mu = fmax(mumin, fmin(kap_mu * mu, pow(mu, th_mu)));
                nf = 0;
                cmu = 0; for (int j = 0; j < nl; j++) cmu = fmax(cmu, fabs(xl[j] * z[j] - mu));
                Emu = fmax(fmax(dinf / sd, pinf), cmu / sc);
            }
        } else {
            /* progress check: the optimality error must fall by a factor within a few iterations */
            if (it == 0 || E0 < 0.9999 * Eref) { Eref = E0; nstall = 0; } else nstall++;
            if (nstall >= 8) { freemode = 0; mu = fmax(mumin, 0.8 * c0avg(xl, z, nl)); nf = 0; if (verbose >= 1) printf("      monotone barrier from here (mu %.1e)\n", mu); }
        }
        double tau = fmax(tau_min, 1 - mu);
        /* the Newton system with the inertia correction */
        for (int j = 0; j < nl; j++) dl[j] = z[j] / xl[j];
        for (int v = 0; v < nR; v++) rhs[v] = -gL[v];
        for (int j = 0; j < nl; j++) rhs[nR + j] = -(gL[nR + j] + z[j] - mu / xl[j] + kap_d * mu);
        for (int i = 0; i < m; i++) rhs[nx + i] = -g[i];
        double dc = fmax(fmax(getenv("BRISK_BM2DC0") ? 0.0 : 1e-8 * pow(mu, 0.25), dc_last > 0 ? dc_last / 10 : 0.0), dcfloor);
        double dw = fmax(dwforce, dwmin);   /* a floor: the Hessian is singular along the symmetries the gauge does not fix (a global phase, rotations among columns) */
        for (int tries = 0; ; tries++) {
            double *dd = ix(sizeof(double) * (nl + 1)); for (int j = 0; j < nl; j++) dd[j] = dl[j] + dw;
            kkt_assemble(Kt, Q, vE, 1, dw, dd, dc); free(dd);
            nfac++;
            if (dumpkkt == it) {   /* diagnostic: the matrix and its inertia as read, for an outside check */
                char fn[64]; snprintf(fn, sizeof fn, "kkt_%d_%d.txt", it, tries);
                FILE *fo = fopen(fn, "w");
                if (fo) { const LDL *F = &Kt->L; fprintf(fo, "%d %d %d %.17g %.17g\n", F->n, nx, F->Ap[F->n], dw, dc);
                          for (int c = 0; c < F->n; c++) for (int q = F->Ap[c]; q < F->Ap[c + 1]; q++) fprintf(fo, "%d %d %.17g\n", F->Ai[q], c, F->Ax[q]);
                          fclose(fo); }
            }
            const int inr = kkt_factor(Kt);
            if (dumpkkt == it) printf("      [dump: try %d, dw %.3e, dc %.3e, positive pivots minus nx: %d]\n", tries, dw, dc, inr);
            if (inr == 0) break;
            if (inr > 0 && dc < 1e-4) { dc = fmax(10 * dc, 1e-8); continue; }   /* a constraint pivot came out positive or zero: dependent rows */
            if (dw <= dwmin) dw = dw_last <= dwmin ? 1e-4 : fmax(dwmin, dw_last / 3);
            else dw *= dw_last <= dwmin ? 100 : 8;
            if (dw > 1e40 || tries > 60) { if (verbose >= 1) printf("      inertia correction failed (dw %.1e)\n", dw); status = 2; goto done; }
        }
        dw_last = dw; dc_last = dc > 1e-8 * pow(mu, 0.25) * 1.0001 ? dc : 0.0;
        if (freemode && nl > 0) {
            /* Ipopt's quality function: the direction is affine in mu = sigma mu_avg; sigma in [1e-6, 100] minimizes
             * the predicted errors after the step (dual and primal infeasibility shrink by 1 - alpha, complementarity
             * at the new point), by golden section on log sigma */
            const double mavg = c0avg(xl, z, nl);
            for (int j = 0; j < nl; j++) rhs[nR + j] = -(gL[nR + j] + z[j]);
            kkt_solve(Kt, rhs, sol);                     /* sigma = 0 */
            for (int j = 0; j < nl; j++) rhs[nR + j] = -(gL[nR + j] + z[j] - mavg / xl[j] + kap_d * mavg);
            kkt_solve(Kt, rhs, sol2);                    /* sigma = 1 */
            double nd2 = 0, np2 = 0; for (int v = 0; v < nx; v++) nd2 += gL[v] * gL[v]; for (int i = 0; i < m; i++) np2 += g[i] * g[i];
            nd2 /= nx; np2 /= m;   /* mean squares, as Ipopt's quality function (2-norm-squared, by the number of entries) */
            #define QF(sg, out) do { double ap_ = 1, ad_ = 1, cq_ = 0; const double mu_ = (sg) * mavg; \
                for (int j = 0; j < nl; j++) { const double dx_ = sol[nR + j] + (sg) * (sol2[nR + j] - sol[nR + j]); if (dx_ < 0) ap_ = fmin(ap_, -tau_min * xl[j] / dx_); \
                    const double dz_ = mu_ / xl[j] - z[j] - dl[j] * dx_; if (dz_ < 0) ad_ = fmin(ad_, -tau_min * z[j] / dz_); } \
                for (int j = 0; j < nl; j++) { const double dx_ = sol[nR + j] + (sg) * (sol2[nR + j] - sol[nR + j]); const double dz_ = mu_ / xl[j] - z[j] - dl[j] * dx_; \
                    const double c_ = (xl[j] + ap_ * dx_) * (z[j] + ad_ * dz_); cq_ += c_ * c_; } \
                out = (1 - ad_) * (1 - ad_) * nd2 + (1 - ap_) * (1 - ap_) * np2 + cq_ / nl; } while (0)
            if (getenv("BRISK_BM2QFDBG") && it < 3) {
                printf("      [qf: mavg %.2e nd2 %.2e np2 %.2e]", mavg, nd2, np2);
                for (double lsg = -6; lsg <= log10(sigmax) + 1e-9; lsg += 1) { double qv; QF(pow(10, lsg), qv); printf(" s=1e%+.0f:%.3e", lsg, qv); }
                printf("\n");
            }
            double a = -6, c = log10(sigmax), gr = 0.5 * (sqrt(5.0) - 1), x1 = c - gr * (c - a), x2 = a + gr * (c - a), q1, q2;
            QF(pow(10, x1), q1); QF(pow(10, x2), q2);
            for (int k = 0; k < 24; k++) {
                if (q1 <= q2) { c = x2; x2 = x1; q2 = q1; x1 = c - gr * (c - a); QF(pow(10, x1), q1); }
                else { a = x1; x1 = x2; q1 = q2; x2 = a + gr * (c - a); QF(pow(10, x2), q2); }
            }
            #undef QF
            const double sg = pow(10, 0.5 * (a + c));
            const double munew = fmax(mumin, fmin(sg * mavg, mu_max));
            if (munew != mu && !getenv("BRISK_BM2OLDLS")) nf = 0;   /* the filter's entries belong to the previous barrier problem */
            else if (munew < mu) nf = 0;
            mu = munew;
            for (int v = 0; v < N; v++) sol[v] += sg * (sol2[v] - sol[v]);
            for (int j = 0; j < nl; j++) rhs[nR + j] = -(gL[nR + j] + z[j] - mu / xl[j] + kap_d * mu);
        } else kkt_solve(Kt, rhs, sol);
        tau = fmax(tau_min, 1 - mu);
        if (getenv("BRISK_BM2STEPCHK") && it < 3) {
            /* the step in the linearized equations: W dx + J'dlam - [0; dz] = -grad L, J dx = -g, z dx_l + x dz = mu - x z */
            double *t1 = ix(sizeof(double) * (N + 1)), *t2 = ix(sizeof(double) * (N + 1));
            for (int v = 0; v < nx; v++) t1[v] = sol[v];
            for (int i = 0; i < m; i++) t1[nx + i] = sol[nx + i];
            ldl_mv(&Kt->L, t1, t2);   /* K [dx; dlam] with the regularizations in K */
            double e1 = 0, e2 = 0, e3 = 0, n1 = 0;
            { double rr = 0, bb = 0; for (int v = 0; v < N; v++) { rr = fmax(rr, fabs(t2[v] - rhs[v])); bb = fmax(bb, fabs(rhs[v])); } printf("      [K sol - rhs: %.1e of %.1e]\n", rr, bb); }
            double eR = 0; int vR = -1;
            for (int v = 0; v < nR; v++) { const double r = t2[v] - dw_last * sol[v] + gL[v]; if (fabs(r) > eR) { eR = fabs(r); vR = v; } n1 = fmax(n1, fabs(gL[v])); }
            printf("      [R part: residual %.1e at variable %d (step %.2e, grad %.2e)]\n", eR, vR, vR >= 0 ? sol[vR] : 0, vR >= 0 ? gL[vR] : 0);
            e1 = eR;
            for (int j = 0; j < nl; j++) {
                const double dzj = mu / xl[j] - z[j] - dl[j] * sol[nR + j];
                /* row of x_l in K: (Sigma + dw) dx_l + A_l' dlam; the true equation: A_l' dlam - dz = -(grad L_l) */
                const double lhs = t2[nR + j] - (dl[j] + dw_last) * sol[nR + j] - dzj;
                e1 = fmax(e1, fabs(lhs + gL[nR + j]));
                e3 = fmax(e3, fabs(z[j] * sol[nR + j] + xl[j] * dzj - (mu - xl[j] * z[j])));
            }
            for (int i = 0; i < m; i++) { const double r = t2[nx + i] + (1e-8 * pow(mu, 0.25)) * sol[nx + i] + g[i]; e2 = fmax(e2, fabs(r)); }
            printf("      [step check: dual %.1e (of %.1e), primal %.1e, complementarity %.1e; dw %.1e]\n", e1, n1, e2, e3, dw_last);
            free(t1); free(t2);
        }
        if (it == 0 && getenv("BRISK_BM2DIR0")) {   /* diagnostic: the first direction (LP variables, then block 0's rows) */
            FILE *fo = fopen(getenv("BRISK_BM2DIR0"), "w");
            if (fo) { fprintf(fo, "%d %d %.17g\n", nl, Q->bk[0].n, mu);
                      for (int j = 0; j < nl; j++) fprintf(fo, "x %d %.17g %.17g\n", j, xl[j], sol[nR + j]);
                      for (int j = 0; j < Q->bk[0].n; j++) { const int v = Q->bk[0].vid[(size_t)j * Q->bk[0].r]; fprintf(fo, "r %d %.17g %.17g\n", j, Q->bk[0].R[(size_t)j * Q->bk[0].r], v >= 0 ? sol[v] : 0.0); }
                      fclose(fo); }
        }
        /* step lengths */
        double amax = 1.0, az = 1.0;
        for (int j = 0; j < nl; j++) {
            const double d = sol[nR + j]; if (d < 0) amax = fmin(amax, -tau * xl[j] / d);
            dz[j] = mu / xl[j] - z[j] - dl[j] * d; if (dz[j] < 0) az = fmin(az, -tau * z[j] / dz[j]);
        }
        /* a tiny step (relative to the point, Ipopt's tiny_step_tol): take it; in the monotone mode lower mu */
        { double tn = 0; for (int v = 0; v < nx; v++) tn = fmax(tn, fabs(sol[v]) / (1 + fabs(x[v])));
          if (tn < 10 * DBL_EPSILON_ && pinf < 1e-4) {
              for (int v = 0; v < nx; v++) x[v] += sol[v];
              x_set(Q, x, xl); for (int i = 0; i < m; i++) lam[i] += sol[nx + i];
              for (int j = 0; j < nl; j++) { z[j] += dz[j]; z[j] = fmin(fmax(z[j], mu / (kap_sig * xl[j])), kap_sig * mu / xl[j]); }
              ip_eval(Q, xl, u, g, &f);
              if (!freemode) { mu = fmax(mumin, fmin(kap_mu * mu, pow(mu, th_mu))); nf = 0; }
              continue;
          } }
        /* filter line search */
        const double bw = freemode && getenv("BRISK_BM2OLDLS") ? 0.0 : mu;   /* the barrier function of the current mu (in the free mode too) */
        double sxl = 0; for (int j = 0; j < nl; j++) sxl += xl[j];
        const double ph = f - bw * logb + kap_d * bw * sxl;
        double gph = 0; for (int v = 0; v < nx; v++) gph += gf[v] * sol[v]; for (int j = 0; j < nl; j++) gph -= bw * sol[nR + j] / xl[j] - kap_d * bw * sol[nR + j];
        const double amin = gph < 0 ? g_al * fmin(g_th, fmin(g_ph * th / -gph, dlt * pow(th, s_th) / pow(-gph, s_ph))) : g_al * g_th;
        double al = amax; int acc = 0, ftype = 0; double fT = 0, thT = 0, phT = 0;
        for (int ls = 0; ls < 60 && al >= amin; ls++) {
            for (int v = 0; v < nx; v++) xt[v] = x[v] + al * sol[v];
            x_set(Q, xt, xlt);
            ip_eval(Q, xlt, ut, gt, &fT);
            double lb2 = 0; for (int j = 0; j < nl; j++) lb2 += log(xlt[j]);
            double sx2 = 0; for (int j = 0; j < nl; j++) sx2 += xlt[j];
            thT = nrm1(gt, m); phT = fT - bw * lb2 + kap_d * bw * sx2;
            int ok = thT <= th_max;
            for (int q = 0; q < nf && ok; q++) if (thT >= filt[q].th && phT >= filt[q].ph) ok = 0;
            if (ok) {
                const int sw = gph < 0 && al * pow(-gph, s_ph) > dlt * pow(th, s_th);
                if (th <= th_min && sw) { ok = phT <= ph + eta * al * gph; ftype = ok; }
                else ok = thT <= (1 - g_th) * th || phT <= ph - g_ph * th;
            }
            if (ok) { acc = 1; break; }
            /* second-order corrections at the first trial point */
            if (ls == 0 && thT >= th) {
                double thold = th; for (int i = 0; i < m; i++) csoc[i] = al * g[i] + gt[i];
                for (int p = 0; p < 4; p++) {
                    for (int v = 0; v < nx; v++) rhs[v] = v < nR ? -gL[v] : -(gL[v] + z[v - nR] - mu / xl[v - nR]);
                    for (int i = 0; i < m; i++) rhs[nx + i] = -csoc[i];
                    kkt_solve(Kt, rhs, sol2);
                    double asoc = 1.0; for (int j = 0; j < nl; j++) { const double d = sol2[nR + j]; if (d < 0) asoc = fmin(asoc, -tau * xl[j] / d); }
                    for (int v = 0; v < nx; v++) xt[v] = x[v] + asoc * sol2[v];
                    x_set(Q, xt, xlt);
                    ip_eval(Q, xlt, ut, gt, &fT);
                    double lb3 = 0; for (int j = 0; j < nl; j++) lb3 += log(xlt[j]);
                    double sx3 = 0; for (int j = 0; j < nl; j++) sx3 += xlt[j];
                    const double thS = nrm1(gt, m), phS = fT - bw * lb3 + kap_d * bw * sx3;
                    int okS = thS <= th_max;
                    for (int q = 0; q < nf && okS; q++) if (thS >= filt[q].th && phS >= filt[q].ph) okS = 0;
                    if (okS) {
                        const int sw = gph < 0 && al * pow(-gph, s_ph) > dlt * pow(th, s_th);
                        if (th <= th_min && sw) { okS = phS <= ph + eta * al * gph; ftype = okS; }
                        else okS = thS <= (1 - g_th) * th || phS <= ph - g_ph * th;
                    }
                    if (okS) { acc = 2; thT = thS; phT = phS; al = asoc; memcpy(sol, sol2, sizeof(double) * nx); break; }
                    if (thS > 0.99 * thold) break;
                    thold = thS; for (int i = 0; i < m; i++) csoc[i] = asoc * csoc[i] + gt[i];
                }
                if (acc) break;
            }
            al *= 0.5;
        }
        if (!acc && nretry < 6) {
            /* the line search failed: the same iteration again with a larger regularization of the Hessian (shorter,
             * more gradient-like steps) before the restoration */
            dwforce = fmax(10 * fmax(dw_last, dwforce), 1e-4); nretry++;
            if (verbose >= 2) printf("      line search failed: again with dw %.1e\n", dwforce);
            x_set(Q, x, xl); ip_eval(Q, xl, u, g, &f);
            it--; continue;
        }
        if (!acc) {
            /* restoration: min 1/2 |g(x)|^2 + zeta/2 |D (x - x_R)|^2 - mu sum log x_l by damped Gauss-Newton steps,
             * [[zeta D^2 + mu/x^2 + nu I, J'], [J, -I]] (quasi-definite), Armijo on that smooth function; until theta
             * falls to 0.9 of its value and the filter accepts the point */
            nres++;
            if (verbose >= 1) printf("      restoration (theta %.2e)\n", th);
            if (nf == capf) { capf = capf ? 2 * capf : 16; filt = realloc(filt, sizeof(FEnt) * capf); }
            filt[nf].th = (1 - g_th) * th; filt[nf].ph = ph - g_ph * th; nf++;
            x_set(Q, x, xl);
            if (!getenv("BRISK_BM2GNRES")) {
                /* Ipopt's restoration: the l1-elastic problem by Newton steps (resto_pen) */
                double *fth = ix(sizeof(double) * (nf + 1)), *fph = ix(sizeof(double) * (nf + 1));
                for (int q = 0; q < nf; q++) { fth[q] = filt[q].th; fph[q] = filt[q].ph; }
                const int okr = resto_pen(Q, Kt, x, xl, lam, z, mu, kap_d, th, fth, fph, nf, th_max, verbose, t0, &nfac);
                free(fth); free(fph);
                if (!okr) { status = 2; goto done; }
                ip_eval(Q, xl, u, g, &f);
                for (int j = 0; j < nl; j++) z[j] = fmin(fmax(z[j], mu / (kap_sig * xl[j])), kap_sig * mu / xl[j]);
                continue;
            }
            double *xR = ix(sizeof(double) * (nx + 1)), *qd = ix(sizeof(double) * (nx + 1)); memcpy(xR, x, sizeof(double) * nx);
            const double zeta = sqrt(fmax(mu, 1e-12)), mur = fmax(mu, 1e-8);
            for (int v = 0; v < nx; v++) qd[v] = zeta * fmin(1.0, 1.0 / fmax(fabs(xR[v]), 1e-300)) * fmin(1.0, 1.0 / fmax(fabs(xR[v]), 1e-300));
            double nu = 0, thc = th; int okr = 0;
            #define RPHI(xv, xlv, gv, out) do { double s_ = 0; for (int i = 0; i < m; i++) s_ += 0.5 * gv[i] * gv[i]; \
                for (int v = 0; v < nx; v++) s_ += 0.5 * qd[v] * (xv[v] - xR[v]) * (xv[v] - xR[v]); \
                for (int j = 0; j < nl; j++) s_ -= mur * log(xlv[j]); \
                out = s_; } while (0)
            ip_eval(Q, xl, u, g, &f);
            double rp; RPHI(x, xl, g, rp);
            for (int rit = 0; rit < 200 && !okr; rit++) {
                /* the gradient J'g + qd (x - x_R) - mur / x_l, through the KKT: rhs [-(qd (x - x_R) - mur/x); -g] gives d with J d ~ -g */
                for (int j = 0; j < nl; j++) dl[j] = qd[nR + j] + mur / (xl[j] * xl[j]) + nu;
                {   double *qq = ix(sizeof(double) * (nx + 1)); for (int v = 0; v < nx; v++) qq[v] = qd[v] + nu;
                    /* the R diagonal varies per variable: assemble with dR = 0 and add qq afterwards */
                    kkt_assemble(Kt, Q, vE, 0, 0.0, dl, 1.0);
                    for (int v = 0; v < nR; v++) Kt->L.Ax[Kt->offD[v]] += qq[v];
                    free(qq); }
                nfac++;
                if (kkt_factor(Kt) != 0) { nu = fmax(10 * nu, 1e-8); continue; }
                for (int v = 0; v < nR; v++) rhs[v] = -qd[v] * (x[v] - xR[v]);
                for (int j = 0; j < nl; j++) rhs[nR + j] = -(qd[nR + j] * (x[nR + j] - xR[nR + j]) - mur / xl[j]);
                for (int i = 0; i < m; i++) rhs[nx + i] = -g[i];
                kkt_solve(Kt, rhs, sol);
                double a2 = 1.0; for (int j = 0; j < nl; j++) { const double d = sol[nR + j]; if (d < 0) a2 = fmin(a2, -0.995 * xl[j] / d); }
                /* directional derivative of the restoration function: (J'g + qd (x - x_R) - mur/x)'d */
                double gd = 0;
                {   double *t1 = ix(sizeof(double) * (N + 1)), *t2 = ix(sizeof(double) * (N + 1));
                    for (int v = 0; v < nx; v++) t1[v] = 0;
                    for (int i = 0; i < m; i++) t1[nx + i] = g[i];
                    ldl_mv(&Kt->L, t1, t2);                    /* top: J'g */
                    for (int v = 0; v < nx; v++) gd += (t2[v] + qd[v] * (x[v] - xR[v]) - (v >= nR ? mur / xl[v - nR] : 0.0)) * sol[v];
                    free(t1); free(t2); }
                int took = 0; double rpt = rp;
                for (int b2 = 0; b2 < 40; b2++, a2 *= 0.5) {
                    for (int v = 0; v < nx; v++) xt[v] = x[v] + a2 * sol[v];
                    x_set(Q, xt, xlt); ip_eval(Q, xlt, ut, gt, &fT);
                    RPHI(xt, xlt, gt, rpt);
                    if (isfinite(rpt) && rpt <= rp + 1e-4 * a2 * fmin(gd, 0.0)) { took = 1; break; }
                }
                if (!took) { x_set(Q, x, xl); nu = fmax(10 * nu, 1e-6); if (verbose >= 2) printf("        resto %d: no decrease, nu %.1e\n", rit, nu); if (nu > 1e10) break; continue; }
                memcpy(x, xt, sizeof(double) * nx); x_set(Q, x, xl); memcpy(g, gt, sizeof(double) * m); f = fT; rp = rpt;
                nu = nu / 10 < 1e-10 ? 0 : nu / 10;
                thc = nrm1(g, m);
                double lb4 = 0, sx4 = 0; for (int j = 0; j < nl; j++) { lb4 += log(xl[j]); sx4 += xl[j]; }
                const double phn = f - bw * lb4 + kap_d * bw * sx4;
                if (verbose >= 2) printf("        resto %d: theta %.3e (step %.1e)\n", rit, thc, a2);
                int okf = thc <= 0.9 * th;
                for (int q = 0; q < nf && okf; q++) if (thc >= filt[q].th && phn >= filt[q].ph) okf = 0;
                if (okf) okr = 1;
            }
            #undef RPHI
            free(xR); free(qd);
            if (!okr) { if (verbose >= 1) printf("      restoration failed (theta %.2e)\n", thc); status = 2; goto done; }
            x_get(Q, xl, x);
            ip_eval(Q, xl, u, g, &f);
            for (int j = 0; j < nl; j++) z[j] = fmin(fmax(z[j], mu / (kap_sig * xl[j])), kap_sig * mu / xl[j]);
            continue;
        }
        if (verbose >= 2) { /* the step's size per block and on the LP variables, and the multipliers' */
            for (int k = 0; k < Q->K; k++) { const IBlk *B = &Q->bk[k]; double mx = 0; int am = -1; for (size_t q = 0; q < (size_t)B->n * B->r; q++) if (B->vid[q] >= 0 && fabs(sol[B->vid[q]]) > mx) { mx = fabs(sol[B->vid[q]]); am = (int)(q / B->r); } printf("      |dR_%d| %.2e (row %d)", k, mx, am); }
            { double mx = 0, mz = 0; int am = -1; for (int j = 0; j < nl; j++) { if (fabs(sol[nR + j]) > mx) { mx = fabs(sol[nR + j]); am = j; } mz = fmax(mz, fabs(dz[j])); } printf("  |dx_l| %.2e (var %d, x %.2e)  |dz| %.2e", mx, am, am >= 0 ? xl[am] : 0, mz); }
            { double mx = 0; for (int i = 0; i < m; i++) mx = fmax(mx, fabs(sol[nx + i])); printf("  |dlam| %.2e\n", mx); } }
        if (verbose >= 2) { double dn = 0; for (int v = 0; v < nx; v++) dn = fmax(dn, fabs(sol[v])); printf("      step |d| %.2e  alpha %.3e (max %.3e)  alpha_z %.3e  acc %d ftype %d\n", dn, al, amax, az, acc, ftype); }
        if (!ftype) {
            if (nf == capf) { capf = capf ? 2 * capf : 16; filt = realloc(filt, sizeof(FEnt) * capf); }
            filt[nf].th = (1 - g_th) * th; filt[nf].ph = ph - g_ph * th; nf++;
        }
        /* the accepted point: x, lam with al, z with az */
        dwforce = 0; nretry = 0;
        for (int v = 0; v < nx; v++) x[v] += al * sol[v];
        x_set(Q, x, xl);
        if (acc == 1) for (int i = 0; i < m; i++) lam[i] += al * sol[nx + i];
        else for (int i = 0; i < m; i++) lam[i] += sol[nx + i] * al;
        for (int j = 0; j < nl; j++) { z[j] += az * dz[j]; z[j] = fmin(fmax(z[j], mu / (kap_sig * xl[j])), kap_sig * mu / xl[j]); }
        ip_eval(Q, xl, u, g, &f);
        (void)thT; (void)phT;
    }
done:
    if (status != 0 && isfinite(Ebest) && st->E0 > Ebest) {
        /* a failure or a stop away from the best iterate: back to the best one (a failed restoration resets the
         * multipliers; a converged point must not be lost to a line search that fails on rounding) */
        memcpy(x, xb, sizeof(double) * nx); x_set(Q, x, xl); memcpy(lam, lb_, sizeof(double) * m); memcpy(z, zb, sizeof(double) * nl);
        mu = mub; ip_eval(Q, xl, u, g, &f); st->E0 = Ebest;
        if (verbose >= 1) printf("      back to the best iterate (error %.2e)\n", Ebest);
        if (Ebest <= fmax(1e2 * tol, 1e-6) && status == 2) status = 1;
    }
    free(xb); free(lb_); free(zb);
    if (verbose >= 1 && nlsm) printf("      least-squares multipliers taken %d time%s\n", nlsm, nlsm > 1 ? "s" : "");
    st->its = it; st->status = status; st->nfac = nfac; st->nres = nres; st->f = f;
    *mu_io = mu;
    free(u); free(g); free(vE); free(x); free(xt); free(xlt); free(gL); free(gf); free(dl); free(ut); free(gt); free(dz); free(rhs); free(sol); free(sol2); free(csoc);
    for (int k = 0; k < Q->K; k++) free(Gk[k]);
    free(Gk); free(filt);
    return status;
}


/* ---------------------------------------------------------------------------------------------------- */
/* The penalty-barrier method (the default): every constraint gets elastic variables, c(x) - p + n = 0, p, n >= 0
 * at the cost rho (p + n) (the l1 penalty of Ipopt's restoration phase, here the method itself). The program is
 * always feasible (no restoration phase); with p, n eliminated the KKT matrix has the strictly negative diagonal
 * -(1/Sp + 1/Sn) in the constraint block (dependent rows are harmless). After every step p and n are reset to
 * the minimizers of rho (p + n) - mu (log p + log n) on p - n = c(x) (closed form), which makes the merit
 *     Phi(x) = f(x) + sum_i psi(c_i(x)) - mu sum log x_l,  psi(c) = min_{p - n = c} rho (p + n) - mu (log p + log n),
 * a smooth exact penalty (psi -> rho |c|); Armijo backtracking on Phi. rho grows tenfold when a barrier problem
 * is solved while c(x) is not small. Barrier parameter: monotone (Fiacco-McCormick). */
static void el_reset(double c, double rho, double mu, double *p, double *n) {
    const double a = 2 * rho * c + 2 * mu, disc = a * a - 8 * rho * mu * c;
    double pp = (a + sqrt(fmax(disc, 0.0))) / (4 * rho);
    if (pp <= fmax(c, 0.0)) pp = fmax(c, 0.0) + 1e-300;
    *p = pp; *n = pp - c; if (*n <= 0) *n = 1e-300;
}
static double el_psi(double c, double rho, double mu, double *pp, double *nn) {
    double p, n; el_reset(c, rho, mu, &p, &n); if (pp) *pp = p; if (nn) *nn = n;
    return rho * (p + n) - mu * (log(p) + log(n));
}

/* Ipopt's restoration phase (Waechter-Biegler 2006, section 3.3): from x_R,
 *     min rho sum (p + n) + zeta/2 |D (x - x_R)|^2   s.t.  g(x) - p + n = 0,  x_l, p, n >= 0,
 * rho = 1000, D = diag(min(1, 1/|x_R|)), barrier mu_R = max(mu, |g(x_R)|_inf), zeta = sqrt(mu_R); p and n
 * eliminated in closed form as in ipm_pen (the merit sum_i psi(g_i) is a smooth exact penalty). Newton steps on
 * [[W + zeta D^2 + Sigma, J'], [J, -(1/Sp + 1/Sn)]] (W = 2 (A*lam (x) I), no objective), Armijo on the merit;
 * mu_R decreases when its barrier problem is solved. Returns 1 as soon as theta = |g|_1 <= 0.9 theta_R and the
 * original filter accepts the point (theta, f - mu sum log x_l + kap_d mu sum x_l); x, xl, z are then the new
 * point, lam is set to 0 (Ipopt resets the constraint multipliers after a restoration). 0: failed. */
static int resto_pen(IP *Q, KKT *Kt, double *x, double *xl, double *lam, double *z, double mu, double kap_d, double th_R,
                     const double *fth, const double *fph, int nf, double th_max, int verbose, double t0, int *nfac) {
    const int m = Q->m, nx = Q->nx, nR = Q->nR, nl = Q->nl, N = Kt->N;
    const double rho = 1000, kap_sig = 1e10, eta = 1e-4;
    double *u = ix(sizeof(double) * (Q->E + 1)), *g = ix(sizeof(double) * (m + 1)), *vE = ix(sizeof(double) * (Q->E + 1));
    double *xR = ix(sizeof(double) * (nx + 1)), *D2 = ix(sizeof(double) * (nx + 1)), *xt = ix(sizeof(double) * (nx + 1)), *xlt = ix(sizeof(double) * (nl + 1));
    double *gL = ix(sizeof(double) * (nx + 1)), *gp = ix(sizeof(double) * (nx + 1)), *dl = ix(sizeof(double) * (nl + 1)), *dd = ix(sizeof(double) * (nl + 1));
    double *ut = ix(sizeof(double) * (Q->E + 1)), *gt = ix(sizeof(double) * (m + 1)), *dz = ix(sizeof(double) * (nl + 1));
    double *rhs = ix(sizeof(double) * (N + 1)), *sol = ix(sizeof(double) * (N + 1)), *drow = ix(sizeof(double) * (m + 1)), *Jd = ix(sizeof(double) * (N + 1)), *tmp = ix(sizeof(double) * (N + 1));
    double *p = ix(sizeof(double) * (m + 1)), *n = ix(sizeof(double) * (m + 1)), *zp = ix(sizeof(double) * (m + 1)), *zn = ix(sizeof(double) * (m + 1));
    double *dzp = ix(sizeof(double) * (m + 1)), *dzn = ix(sizeof(double) * (m + 1));
    double **Gk = ix(sizeof(double *) * (Q->K + 1));
    for (int k = 0; k < Q->K; k++) Gk[k] = ix(sizeof(double) * ((size_t)Q->bk[k].n * Q->bk[k].r + 1));
    memcpy(xR, x, sizeof(double) * nx);
    for (int v = 0; v < nx; v++) { const double d = fmin(1.0, 1.0 / fmax(fabs(xR[v]), 1e-300)); D2[v] = d * d; }
    double f = 0; ip_eval(Q, xl, u, g, &f);
    double muR = fmax(mu, nrmi(g, m)), dw_last = 0;
    for (int i = 0; i < m; i++) { el_reset(g[i], rho, muR, &p[i], &n[i]); zp[i] = muR / p[i]; zn[i] = muR / n[i]; lam[i] = rho - zp[i]; }
    for (int j = 0; j < nl; j++) z[j] = muR / xl[j];
    int ok = 0, it;
    for (it = 0; it < 500; it++) {
        const double zeta = sqrt(muR);
        /* the restoration Lagrangian's gradient: no objective, the proximal term */
        ip_svals(Q, lam, vE); for (int e = 0; e < Q->E; e++) vE[e] -= Q->cE[e];
        ip_grad(Q, vE, NULL, lam, z, gL, NULL, Gk);
        for (int j = 0; j < nl; j++) gL[nR + j] -= Q->cl[j];
        for (int v = 0; v < nx; v++) { gp[v] = zeta * D2[v] * (x[v] - xR[v]); gL[v] += gp[v]; }
        /* the original filter */
        const double th = nrm1(g, m);
        if (it > 0 && th <= 0.9 * th_R && th <= th_max) {
            double lb = 0, sx = 0; for (int j = 0; j < nl; j++) { lb += log(xl[j]); sx += xl[j]; }
            const double ph = f - mu * lb + kap_d * mu * sx;
            int acc = 1; for (int q = 0; q < nf && acc; q++) if (th >= fth[q] && ph >= fph[q]) acc = 0;
            if (acc) { ok = 1; break; }
        }
        /* its own barrier problem */
        double dinf = nrmi(gL, nx), cmu = 0, rc = 0, einf = 0;
        for (int i = 0; i < m; i++) { einf = fmax(einf, fmax(fabs(rho - lam[i] - zp[i]), fabs(rho + lam[i] - zn[i]))); cmu = fmax(cmu, fmax(fabs(p[i] * zp[i] - muR), fabs(n[i] * zn[i] - muR))); rc = fmax(rc, fabs(g[i] - p[i] + n[i])); }
        for (int j = 0; j < nl; j++) cmu = fmax(cmu, fabs(xl[j] * z[j] - muR));
        const double sd = fmax(100.0, (nrm1(lam, m) + nrm1(z, nl)) / (m + nl)) / 100.0;
        if (verbose >= 2) printf("        resto %3d theta %.3e mu_R %.1e dinf %.1e comp %.1e\n", it, th, muR, dinf, cmu);
        if (fmax(fmax(dinf, einf) / sd, fmax(rc, cmu)) <= 10 * muR && muR > 1e-10) {
            muR = fmax(1e-10, fmin(0.2 * muR, pow(muR, 1.5)));
            for (int i = 0; i < m; i++) el_reset(g[i], rho, muR, &p[i], &n[i]);
            continue;
        }
        const double tau = fmax(0.99, 1 - muR);
        for (int j = 0; j < nl; j++) dl[j] = z[j] / xl[j];
        for (int i = 0; i < m; i++) {
            const double Sp = zp[i] / p[i], Sn = zn[i] / n[i];
            drow[i] = 1.0 / Sp + 1.0 / Sn;
            const double rp = rho - lam[i] - zp[i], rn = rho + lam[i] - zn[i];
            rhs[nx + i] = -(g[i] - p[i] + n[i]) + (-rp + muR / p[i] - zp[i]) / Sp - (-rn + muR / n[i] - zn[i]) / Sn;
        }
        for (int v = 0; v < nR; v++) rhs[v] = -gL[v];
        for (int j = 0; j < nl; j++) rhs[nR + j] = -(gL[nR + j] + z[j] - muR / xl[j]);
        double dw = 0, gphi = 0; int good = 0;
        for (int tries = 0; tries < 40; tries++) {
            for (int j = 0; j < nl; j++) dd[j] = dl[j] + dw;
            kkt_assemble_d(Kt, Q, vE, 1, dw, dd, 0.0, drow);
            for (int v = 0; v < nx; v++) Kt->L.Ax[Kt->offD[v]] += zeta * D2[v];
            (*nfac)++;
            if (kkt_factor(Kt) == 0) {
                kkt_solve(Kt, rhs, sol);
                for (int v = 0; v < nx; v++) tmp[v] = sol[v];
                for (int i = 0; i < m; i++) tmp[nx + i] = 0;
                ldl_mv(&Kt->L, tmp, Jd);                      /* rows: J dx */
                gphi = 0; for (int v = 0; v < nx; v++) gphi += gp[v] * sol[v];
                for (int i = 0; i < m; i++) gphi += (rho - muR / p[i]) * Jd[nx + i];
                for (int j = 0; j < nl; j++) gphi -= muR * sol[nR + j] / xl[j];
                if (gphi < 0) { good = 1; break; }
            }
            if (dw == 0) dw = dw_last == 0 ? 1e-4 : fmax(1e-20, dw_last / 3);
            else dw *= dw_last == 0 ? 100 : 8;
            if (dw > 1e40) break;
        }
        if (!good) break;
        dw_last = dw;
        double amax = 1.0, az = 1.0;
        for (int j = 0; j < nl; j++) {
            const double d = sol[nR + j]; if (d < 0) amax = fmin(amax, -tau * xl[j] / d);
            dz[j] = muR / xl[j] - z[j] - dl[j] * d; if (dz[j] < 0) az = fmin(az, -tau * z[j] / dz[j]);
        }
        for (int i = 0; i < m; i++) {
            const double dlam = sol[nx + i], Sp = zp[i] / p[i], Sn = zn[i] / n[i];
            const double rp = rho - lam[i] - zp[i], rn = rho + lam[i] - zn[i];
            const double dp = (-rp + dlam + muR / p[i] - zp[i]) / Sp, dn = (-rn - dlam + muR / n[i] - zn[i]) / Sn;
            dzp[i] = (muR - p[i] * zp[i] - zp[i] * dp) / p[i]; dzn[i] = (muR - n[i] * zn[i] - zn[i] * dn) / n[i];
            if (dzp[i] < 0) az = fmin(az, -tau * zp[i] / dzp[i]);
            if (dzn[i] < 0) az = fmin(az, -tau * zn[i] / dzn[i]);
        }
        double ph = 0; for (int v = 0; v < nx; v++) ph += 0.5 * zeta * D2[v] * (x[v] - xR[v]) * (x[v] - xR[v]);
        for (int i = 0; i < m; i++) ph += el_psi(g[i], rho, muR, NULL, NULL);
        for (int j = 0; j < nl; j++) ph -= muR * log(xl[j]);
        double al = amax, fT = 0; int acc = 0;
        for (int ls = 0; ls < 50; ls++) {
            for (int v = 0; v < nx; v++) xt[v] = x[v] + al * sol[v];
            x_set(Q, xt, xlt);
            ip_eval(Q, xlt, ut, gt, &fT);
            double phT = 0; for (int v = 0; v < nx; v++) phT += 0.5 * zeta * D2[v] * (xt[v] - xR[v]) * (xt[v] - xR[v]);
            for (int i = 0; i < m; i++) phT += el_psi(gt[i], rho, muR, NULL, NULL);
            for (int j = 0; j < nl; j++) phT -= muR * log(xlt[j]);
            if (isfinite(phT) && phT <= ph + eta * al * gphi) { acc = 1; break; }
            al *= 0.5;
        }
        if (!acc) { x_set(Q, x, xl); dw_last = fmax(10 * dw_last, 1e-4); continue; }
        memcpy(x, xt, sizeof(double) * nx); x_set(Q, x, xl);
        memcpy(g, gt, sizeof(double) * m); memcpy(u, ut, sizeof(double) * Q->E); f = fT;
        for (int i = 0; i < m; i++) {
            lam[i] += al * sol[nx + i];
            el_reset(g[i], rho, muR, &p[i], &n[i]);
            zp[i] += az * dzp[i]; zn[i] += az * dzn[i];
            zp[i] = fmin(fmax(zp[i], muR / (kap_sig * p[i])), kap_sig * muR / p[i]);
            zn[i] = fmin(fmax(zn[i], muR / (kap_sig * n[i])), kap_sig * muR / n[i]);
        }
        for (int j = 0; j < nl; j++) { z[j] += az * dz[j]; z[j] = fmin(fmax(z[j], muR / (kap_sig * xl[j])), kap_sig * muR / xl[j]); }
        if (t0 > 0 && 0) break;
    }
    if (verbose >= 1) printf("      restoration %s after %d iterations (theta %.2e -> %.2e)\n", ok ? "done" : "failed", it, th_R, nrm1(g, m));
    for (int i = 0; i < m; i++) lam[i] = 0;
    { double zm = nrmi(z, nl); if (zm > 1000) for (int j = 0; j < nl; j++) z[j] = 1.0; }
    free(u); free(g); free(vE); free(xR); free(D2); free(xt); free(xlt); free(gL); free(gp); free(dl); free(dd); free(ut); free(gt); free(dz);
    free(rhs); free(sol); free(drow); free(Jd); free(tmp); free(p); free(n); free(zp); free(zn); free(dzp); free(dzn);
    for (int k = 0; k < Q->K; k++) free(Gk[k]);
    free(Gk);
    (void)t0;
    return ok;
}
static int ipm_pen(IP *Q, KKT *Kt, double *xl, double *lam, double *z, double *mu_io, double *rho_io, double tol, int maxit, int verbose, double tlim, double t0, IStat *st) {
    const int m = Q->m, nx = Q->nx, nR = Q->nR, nl = Q->nl, N = Kt->N;
    double mu = *mu_io, rho = *rho_io;
    const double kap_eps = 10, kap_mu = 0.2, th_mu = 1.5, tau_min = 0.99, eta = 1e-4, kap_sig = 1e10;
    double *u = ix(sizeof(double) * (Q->E + 1)), *g = ix(sizeof(double) * (m + 1)), *vE = ix(sizeof(double) * (Q->E + 1));
    double *x = ix(sizeof(double) * (nx + 1)), *xt = ix(sizeof(double) * (nx + 1)), *xlt = ix(sizeof(double) * (nl + 1));
    double *gL = ix(sizeof(double) * (nx + 1)), *gf = ix(sizeof(double) * (nx + 1)), *dl = ix(sizeof(double) * (nl + 1));
    double *ut = ix(sizeof(double) * (Q->E + 1)), *gt = ix(sizeof(double) * (m + 1)), *dz = ix(sizeof(double) * (nl + 1));
    double *rhs = ix(sizeof(double) * (N + 1)), *sol = ix(sizeof(double) * (N + 1)), *drow = ix(sizeof(double) * (m + 1));
    double *p = ix(sizeof(double) * (m + 1)), *n = ix(sizeof(double) * (m + 1)), *zp = ix(sizeof(double) * (m + 1)), *zn = ix(sizeof(double) * (m + 1));
    double *dzp = ix(sizeof(double) * (m + 1)), *dzn = ix(sizeof(double) * (m + 1)), *psd = ix(sizeof(double) * (m + 1));
    double **Gk = ix(sizeof(double *) * (Q->K + 1));
    for (int k = 0; k < Q->K; k++) Gk[k] = ix(sizeof(double) * ((size_t)Q->bk[k].n * Q->bk[k].r + 1));
    double f = 0, dw_last = 0;
    int status = 1, it, nfac = 0, nres = 0;
    x_get(Q, xl, x);
    ip_eval(Q, xl, u, g, &f);
    /* the elastic variables and their multipliers, consistent with lam where possible */
    for (int i = 0; i < m; i++) {
        el_reset(g[i], rho, mu, &p[i], &n[i]);
        if (fabs(lam[i]) < 0.9 * rho) { zp[i] = rho - lam[i]; zn[i] = rho + lam[i]; }
        else { zp[i] = mu / p[i]; zn[i] = mu / n[i]; lam[i] = rho - zp[i]; }
    }
    for (it = 0; it < maxit; it++) {
        ip_svals(Q, lam, vE);
        ip_grad(Q, vE, Q->cE, lam, z, gL, gf, Gk);
        double dinf = nrmi(gL, nx), einf = 0, c0 = 0, cmu = 0, th = 0, rc = 0;
        for (int i = 0; i < m; i++) {
            einf = fmax(einf, fmax(fabs(rho - lam[i] - zp[i]), fabs(rho + lam[i] - zn[i])));
            c0 = fmax(c0, fmax(p[i] * zp[i], n[i] * zn[i])); cmu = fmax(cmu, fmax(fabs(p[i] * zp[i] - mu), fabs(n[i] * zn[i] - mu)));
            th = fmax(th, fabs(g[i])); rc = fmax(rc, fabs(g[i] - p[i] + n[i]));
        }
        for (int j = 0; j < nl; j++) { c0 = fmax(c0, xl[j] * z[j]); cmu = fmax(cmu, fabs(xl[j] * z[j] - mu)); }
        const double sd = fmax(100.0, (nrm1(lam, m) + nrm1(z, nl)) / (m + nl)) / 100.0;
        const double E0 = fmax(fmax(dinf / sd, fmax(einf / sd, rc)), fmax(c0, th));   /* th: the original problem's infeasibility */
        const double Emu = fmax(fmax(dinf / sd, fmax(einf / sd, rc)), cmu);
        if (verbose >= 1) printf("  %3d f %.10e pinf %.1e dinf %.1e comp %.1e mu %.1e rho %.0e dw %.1e |lam| %.1e  %.1fs\n", it, f * Q->cs, th, dinf, c0, mu, rho, dw_last, nrmi(lam, m), wtime() - t0);
        st->E0 = E0;
        if (E0 <= tol) { status = 0; break; }
        if (tlim > 0 && wtime() - t0 > tlim) { status = 3; break; }
        /* barrier parameter and penalty */
        int upd = 0;
        while (Emu <= kap_eps * mu && mu > tol / 10) {
            /* the barrier problem is solved: if c(x) is not small while the multipliers press on rho, raise rho */
            if (th > fmax(1e2 * mu, tol) && nrmi(lam, m) > 0.5 * rho && rho < 1e12) { rho *= 10; for (int i = 0; i < m; i++) { zp[i] = rho - lam[i]; zn[i] = rho + lam[i]; } upd = 1; break; }
            mu = fmax(tol / 10, fmin(kap_mu * mu, pow(mu, th_mu))); upd = 1;
            break;
        }
        if (upd) { for (int i = 0; i < m; i++) { el_reset(g[i], rho, mu, &p[i], &n[i]); } }
        const double tau = fmax(tau_min, 1 - mu);
        /* the Newton system: R, x_l, rows; p, n eliminated */
        for (int j = 0; j < nl; j++) dl[j] = z[j] / xl[j];
        for (int i = 0; i < m; i++) {
            const double Sp = zp[i] / p[i], Sn = zn[i] / n[i];
            drow[i] = 1.0 / Sp + 1.0 / Sn;
            const double rp = rho - lam[i] - zp[i], rn = rho + lam[i] - zn[i];
            rhs[nx + i] = -(g[i] - p[i] + n[i]) + (-rp + mu / p[i] - zp[i]) / Sp - (-rn + mu / n[i] - zn[i]) / Sn;
        }
        for (int v = 0; v < nR; v++) rhs[v] = -gL[v];
        for (int j = 0; j < nl; j++) rhs[nR + j] = -(gL[nR + j] + z[j] - mu / xl[j]);
        double dw = 0, gphi = 0; int ok = 0;
        for (int tries = 0; tries < 40; tries++) {
            double *dd = ix(sizeof(double) * (nl + 1)); for (int j = 0; j < nl; j++) dd[j] = dl[j] + dw;
            kkt_assemble_d(Kt, Q, vE, 1, dw, dd, 0.0, drow); free(dd);
            nfac++;
            if (kkt_factor(Kt) == 0) {
                kkt_solve(Kt, rhs, sol);
                /* the merit's derivative along the step: grad f + psi'(c) J (psi' = rho - mu / p) - mu / x_l */
                ip_svals(Q, lam, vE);   /* (unchanged) */
                for (int i = 0; i < m; i++) psd[i] = rho - mu / p[i];
                /* J dx via the assembled KKT: the rows' part of K [dx; 0] */
                for (int v = 0; v < N; v++) rhs[v + 0] = rhs[v];   /* (keep) */
                double *w = Kt->res; double *tmp = Kt->work;
                for (int v = 0; v < nx; v++) tmp[v] = sol[v];
                for (int i = 0; i < m; i++) tmp[nx + i] = 0;
                ldl_mv(&Kt->L, tmp, w);
                gphi = 0; for (int v = 0; v < nx; v++) gphi += gf[v] * sol[v];
                for (int i = 0; i < m; i++) gphi += psd[i] * w[nx + i];
                for (int j = 0; j < nl; j++) gphi -= mu * sol[nR + j] / xl[j];
                if (gphi < 0) { ok = 1; break; }
                /* not a descent direction for the merit: more regularization */
            }
            if (dw == 0) dw = dw_last == 0 ? 1e-4 : fmax(1e-20, dw_last / 3);
            else dw *= dw_last == 0 ? 100 : 8;
            if (dw > 1e40) break;
            /* the right-hand side was not changed */
        }
        if (!ok) { if (verbose >= 1) printf("      no descent direction (dw %.1e)\n", dw); status = 2; break; }
        dw_last = dw;
        /* the elastic and dual parts of the step */
        double amax = 1.0, az = 1.0;
        for (int j = 0; j < nl; j++) {
            const double d = sol[nR + j]; if (d < 0) amax = fmin(amax, -tau * xl[j] / d);
            dz[j] = mu / xl[j] - z[j] - dl[j] * d; if (dz[j] < 0) az = fmin(az, -tau * z[j] / dz[j]);
        }
        for (int i = 0; i < m; i++) {
            const double dlam = sol[nx + i], Sp = zp[i] / p[i], Sn = zn[i] / n[i];
            const double rp = rho - lam[i] - zp[i], rn = rho + lam[i] - zn[i];
            const double dp = (-rp + dlam + mu / p[i] - zp[i]) / Sp, dn = (-rn - dlam + mu / n[i] - zn[i]) / Sn;
            dzp[i] = (mu - p[i] * zp[i] - zp[i] * dp) / p[i]; dzn[i] = (mu - n[i] * zn[i] - zn[i] * dn) / n[i];
            if (dzp[i] < 0) az = fmin(az, -tau * zp[i] / dzp[i]);
            if (dzn[i] < 0) az = fmin(az, -tau * zn[i] / dzn[i]);
        }
        /* Armijo on the merit (p, n reset at the trial point) */
        double ph = f; for (int i = 0; i < m; i++) ph += el_psi(g[i], rho, mu, NULL, NULL); for (int j = 0; j < nl; j++) ph -= mu * log(xl[j]);
        double al = amax, fT = 0; int acc = 0;
        for (int ls = 0; ls < 50; ls++) {
            for (int v = 0; v < nx; v++) xt[v] = x[v] + al * sol[v];
            x_set(Q, xt, xlt);
            ip_eval(Q, xlt, ut, gt, &fT);
            double phT = fT; for (int i = 0; i < m; i++) phT += el_psi(gt[i], rho, mu, NULL, NULL); for (int j = 0; j < nl; j++) phT -= mu * log(xlt[j]);
            if (phT <= ph + eta * al * gphi) { acc = 1; break; }
            al *= 0.5;
        }
        if (!acc) {
            /* no decrease along a descent direction (rounding): take the smallest trial anyway once, then stop */
            nres++;
            if (nres > 20) { if (verbose >= 1) printf("      line search failed\n"); x_set(Q, x, xl); status = 2; break; }
            dw_last = fmax(10 * dw_last, 1e-4);
            x_set(Q, x, xl);
            continue;
        }
        memcpy(x, xt, sizeof(double) * nx); x_set(Q, x, xl);
        memcpy(g, gt, sizeof(double) * m); memcpy(u, ut, sizeof(double) * Q->E); f = fT;
        for (int i = 0; i < m; i++) {
            lam[i] += al * sol[nx + i];
            el_reset(g[i], rho, mu, &p[i], &n[i]);
            zp[i] += az * dzp[i]; zn[i] += az * dzn[i];
            zp[i] = fmin(fmax(zp[i], mu / (kap_sig * p[i])), kap_sig * mu / p[i]);
            zn[i] = fmin(fmax(zn[i], mu / (kap_sig * n[i])), kap_sig * mu / n[i]);
        }
        for (int j = 0; j < nl; j++) { z[j] += az * dz[j]; z[j] = fmin(fmax(z[j], mu / (kap_sig * xl[j])), kap_sig * mu / xl[j]); }
    }
    st->its = it; st->status = status; st->nfac = nfac; st->nres = nres; st->f = f;
    *mu_io = mu; *rho_io = rho;
    free(u); free(g); free(vE); free(x); free(xt); free(xlt); free(gL); free(gf); free(dl); free(ut); free(gt); free(dz); free(rhs); free(sol); free(drow);
    free(p); free(n); free(zp); free(zn); free(dzp); free(dzn); free(psd);
    for (int k = 0; k < Q->K; k++) free(Gk[k]);
    free(Gk);
    return status;
}


/* ---------------------------------------------------------------------------------------------------- */
/* The augmented Lagrangian with a barrier and trust-region Newton steps (BRISK_BM2ALM=1): for fixed lam,
 * sigma, mu the subproblem
 *     min Phi(R, x) = f + lam'c + sigma/2 |c|^2 - mu sum log x
 * is smooth; its Hessian is 2 (S~ (x) I) (+ mu/x^2 on x) + sigma J'J with S~ = C + sum (lam_i + sigma c_i) A_i,
 * and (H + t I) d = -grad Phi is solved through [[2 S~ (x) I + t I, J'], [J, -I/sigma]] (the constraint block
 * -1/sigma keeps the factorization without pivoting stable; its inertia is right exactly when H + t I is
 * positive definite). t >= 0 from the trust region (|d| <= Delta), the step judged by the ratio of actual to
 * predicted decrease. Outer: lam += sigma c; sigma x10 when |c| did not fall to a quarter; mu x0.2. */
static int ipm_alm(IP *Q, KKT *Kt, double *xl, double *lam, double *z, double *mu_io, double tol, int maxit, int verbose, double tlim, double t0, IStat *st) {
    const int m = Q->m, nx = Q->nx, nR = Q->nR, nl = Q->nl, N = Kt->N;
    double mu = fmax(*mu_io, 1e-8), sig = getenv("BRISK_BM2SIG") ? atof(getenv("BRISK_BM2SIG")) : 1e2;
    double *u = ix(sizeof(double) * (Q->E + 1)), *g = ix(sizeof(double) * (m + 1)), *vE = ix(sizeof(double) * (Q->E + 1));
    double *x = ix(sizeof(double) * (nx + 1)), *xt = ix(sizeof(double) * (nx + 1)), *xlt = ix(sizeof(double) * (nl + 1));
    double *gr = ix(sizeof(double) * (nx + 1)), *dl = ix(sizeof(double) * (nl + 1)), *lt = ix(sizeof(double) * (m + 1));
    double *ut = ix(sizeof(double) * (Q->E + 1)), *gt = ix(sizeof(double) * (m + 1));
    double *rhs = ix(sizeof(double) * (N + 1)), *sol = ix(sizeof(double) * (N + 1)), *tmp = ix(sizeof(double) * (N + 1)), *Kd = ix(sizeof(double) * (N + 1));
    double *zz = iz(sizeof(double) * (nl + 1));
    double **Gk = ix(sizeof(double *) * (Q->K + 1));
    for (int k = 0; k < Q->K; k++) Gk[k] = ix(sizeof(double) * ((size_t)Q->bk[k].n * Q->bk[k].r + 1));
    double f = 0, Delta = 1.0, cprev = INFINITY, omega = 1e-1;
    int status = 1, it = 0, nfac = 0, outer;
    x_get(Q, xl, x);
    #define PHI(xlv, gv, fv, out) do { double s_ = fv; for (int i = 0; i < m; i++) s_ += lam[i] * gv[i] + 0.5 * sig * gv[i] * gv[i]; for (int j = 0; j < nl; j++) s_ -= mu * log(xlv[j]); out = s_; } while (0)
    for (outer = 0; outer < 200 && it < maxit; outer++) {
        /* the subproblem by trust-region Newton */
        double ph;
        ip_eval(Q, xl, u, g, &f); PHI(xl, g, f, ph);
        int inner;
        for (inner = 0; inner < 200 && it < maxit; inner++, it++) {
            for (int i = 0; i < m; i++) lt[i] = lam[i] + sig * g[i];
            ip_svals(Q, lt, vE);
            ip_grad(Q, vE, Q->cE, lt, zz, gr, NULL, Gk);           /* z = 0: the barrier's gradient added next */
            for (int j = 0; j < nl; j++) gr[nR + j] -= mu / xl[j];
            const double gn = nrmi(gr, nx);
            if (verbose >= 2) printf("      in %3d phi %.10e |grad| %.1e |c| %.1e Delta %.1e\n", inner, ph, gn, nrmi(g, m), Delta);
            if (gn <= omega) break;
            if (tlim > 0 && wtime() - t0 > tlim) { status = 3; goto done; }
            /* (H + t I) d = -grad, t >= 0 with |d| <= Delta */
            for (int j = 0; j < nl; j++) dl[j] = mu / (xl[j] * xl[j]);
            for (int v = 0; v < nx; v++) rhs[v] = -gr[v];
            for (int i = 0; i < m; i++) rhs[nx + i] = 0;
            double tr = 0; int ok = 0;
            for (int tries = 0; tries < 60; tries++) {
                double *dd = ix(sizeof(double) * (nl + 1)); for (int j = 0; j < nl; j++) dd[j] = dl[j] + tr;
                kkt_assemble(Kt, Q, vE, 1, tr, dd, 1.0 / sig); free(dd);
                nfac++;
                if (kkt_factor(Kt) == 0) {
                    kkt_solve(Kt, rhs, sol);
                    double dn = 0; for (int v = 0; v < nx; v++) dn += sol[v] * sol[v]; dn = sqrt(dn);
                    if (dn <= Delta * 1.0001) { ok = 1; break; }
                }
                tr = tr == 0 ? fmax(1e-8, 1e-4 * gn) : tr * 4;
            }
            if (!ok) { if (verbose >= 1) printf("      no trust-region step\n"); status = 2; goto done; }
            /* fraction to the boundary on x */
            double amax = 1.0; for (int j = 0; j < nl; j++) { const double d = sol[nR + j]; if (d < 0) amax = fmin(amax, -0.995 * xl[j] / d); }
            for (int v = 0; v < nx; v++) sol[v] *= amax;
            /* predicted decrease: grad'd + d'(H d)/2, H d = (2 S~ (x) I) d + mu/x^2 d_x + sigma J'(J d) */
            for (int v = 0; v < nx; v++) tmp[v] = sol[v];
            for (int i = 0; i < m; i++) tmp[nx + i] = 0;
            ldl_mv(&Kt->L, tmp, Kd);                                 /* [ (W + t I) d ; J d ] */
            double pred = 0, dd2 = 0;
            for (int v = 0; v < nx; v++) { pred += gr[v] * sol[v]; dd2 += sol[v] * (Kd[v] - tr * sol[v]); }
            { double jj = 0; for (int i = 0; i < m; i++) jj += Kd[nx + i] * Kd[nx + i]; dd2 += sig * jj; }
            pred += 0.5 * dd2;
            for (int v = 0; v < nx; v++) xt[v] = x[v] + sol[v];
            x_set(Q, xt, xlt);
            double fT; ip_eval(Q, xlt, ut, gt, &fT);
            double phT; PHI(xlt, gt, fT, phT);
            const double rho = pred < 0 ? (phT - ph) / pred : -1;
            double dn = 0; for (int v = 0; v < nx; v++) dn += sol[v] * sol[v]; dn = sqrt(dn);
            if (rho > 1e-4 && isfinite(phT)) {
                memcpy(x, xt, sizeof(double) * nx); memcpy(xl, xlt, sizeof(double) * nl); memcpy(u, ut, sizeof(double) * Q->E); memcpy(g, gt, sizeof(double) * m);
                f = fT; ph = phT;
                if (rho > 0.75 && dn > 0.8 * Delta) Delta = fmin(2 * Delta, 1e6);
            } else x_set(Q, x, xl);
            if (rho < 0.25) Delta = fmax(dn / 4, 1e-12);
            if (Delta < 1e-12) break;
        }
        /* outer update */
        const double cn = nrmi(g, m);
        for (int i = 0; i < m; i++) lam[i] += sig * g[i];
        for (int j = 0; j < nl; j++) z[j] = mu / xl[j];
        if (verbose >= 1) printf("  out %3d f %.10e |c| %.1e |grad| <= %.1e sig %.0e mu %.1e |lam| %.1e  inner %d  %.1fs\n", outer, f * Q->cs, cn, omega, sig, mu, nrmi(lam, m), inner, wtime() - t0);
        if (cn <= tol && omega <= tol && mu <= tol) { status = 0; break; }
        if (cn > 0.25 * cprev && sig < 1e8) sig *= 10;
        cprev = cn;
        mu = fmax(tol / 10, mu * 0.2);
        omega = fmax(tol, omega * 0.2);
    }
done:
    #undef PHI
    st->its = it; st->status = status; st->nfac = nfac; st->nres = 0; st->f = f;
    *mu_io = mu;
    free(u); free(g); free(vE); free(x); free(xt); free(xlt); free(gr); free(dl); free(lt); free(ut); free(gt); free(rhs); free(sol); free(tmp); free(Kd); free(zz);
    for (int k = 0; k < Q->K; k++) free(Gk[k]);
    free(Gk);
    return status;
}

/* ---------------------------------------------------------------------------------------------------- */
/* lambda_min of S_k (vE on the block's positions) by factorizations of S + s I (bisection on s) and the
 * eigenvector by inverse iteration with the last shift that factored; returns lambda_min */
typedef struct { SChol *F; size_t *offP; size_t *offD; int n; } SFac;
static void sfac_setup(SFac *Sf, const IBlk *B) {
    memset(Sf, 0, sizeof *Sf);
    const int n = B->n; Sf->n = n;
    int *deg = ix(sizeof(int) * (n + 1)); int **nbr = ix(sizeof(int *) * (n + 1));
    for (int j = 0; j < n; j++) {
        int c = 0; nbr[j] = ix(sizeof(int) * (B->sp[j + 1] - B->sp[j] + 1));
        for (int q = B->sp[j]; q < B->sp[j + 1]; q++) if (B->sj[q] != j) nbr[j][c++] = B->sj[q];
        deg[j] = c;
    }
    Sf->F = schol_analyze_adj(n, deg, nbr, (size_t)40 * B->sp[n] + (size_t)8 * n + 100000);
    for (int j = 0; j < n; j++) free(nbr[j]);
    free(nbr); free(deg);
    Sf->offP = ix(sizeof(size_t) * (B->np + 1)); Sf->offD = ix(sizeof(size_t) * (n + 1));
    for (int p = 0; p < B->np; p++) Sf->offP[p] = schol_offset(Sf->F, B->pj[p], B->pk[p]);
    for (int j = 0; j < n; j++) Sf->offD[j] = schol_offset(Sf->F, j, j);
}
static void sfac_free(SFac *Sf) { if (Sf->F) schol_free(Sf->F); free(Sf->offP); free(Sf->offD); memset(Sf, 0, sizeof *Sf); }
static int sfac_try(SFac *Sf, const IBlk *B, const double *vk, double s) {
    schol_zero(Sf->F); double *V = schol_values(Sf->F);
    for (int p = 0; p < B->np; p++) V[Sf->offP[p]] += B->pj[p] == B->pk[p] ? vk[p] : 0.5 * vk[p];
    for (int j = 0; j < Sf->n; j++) V[Sf->offD[j]] += s;
    return schol_factor(Sf->F, 0.0) == 0;
}
static double block_lmin(const IBlk *B, const double *vk, double eps, double *vout, int *certified) {
    SFac Sf; sfac_setup(&Sf, B);
    const int n = B->n;
    /* Gershgorin: lambda_min >= min_j (S_jj - sum_{k != j} |S_jk|) */
    double smax = 0;
    { double *gd = iz(sizeof(double) * (n + 1));
      for (int p = 0; p < B->np; p++) { if (B->pj[p] == B->pk[p]) gd[B->pj[p]] += vk[p]; else { gd[B->pj[p]] -= 0.5 * fabs(vk[p]); gd[B->pk[p]] -= 0.5 * fabs(vk[p]); } }
      for (int j = 0; j < n; j++) smax = fmax(smax, -gd[j]);
      free(gd); }
    smax = fmax(smax, 1e-300);
    double lo = 0, hi;
    *certified = 0;
    double lm;
    if (sfac_try(&Sf, B, vk, eps)) { *certified = 1; hi = eps; lo = -smax; }
    else { hi = eps; lo = eps; while (!sfac_try(&Sf, B, vk, hi)) { lo = hi; hi = hi * 4 + eps; if (hi > 1e300) break; } }
    if (*certified) {
        /* lambda_min >= -eps: bisect downwards for the value itself (a few steps) */
        double a = -smax, c = eps;
        for (int it = 0; it < 60 && c - a > 1e-6 * fmax(1.0, fabs(c)) ; it++) { const double mid = 0.5 * (a + c); if (sfac_try(&Sf, B, vk, mid)) c = mid; else a = mid; }
        lm = -c;
        if (!vout) { sfac_free(&Sf); return lm; }
        hi = c; lo = a;
    } else {
        for (int it = 0; it < 80 && hi - lo > 1e-4 * hi; it++) { const double mid = 0.5 * (lo + hi); if (sfac_try(&Sf, B, vk, mid)) hi = mid; else lo = mid; }
        lm = -hi;
    }
    if (vout) {
        /* inverse iteration with S + hi (1 + 1e-3) I (positive definite, its smallest eigenvalue near 0) */
        const double s = hi + 1e-3 * fabs(hi) + 1e-12;
        if (!sfac_try(&Sf, B, vk, s)) sfac_try(&Sf, B, vk, 2 * s + 1e-9);
        double *w = ix(sizeof(double) * (n + 1)), *work = ix(sizeof(double) * (n + 1));
        unsigned h = 2463534242u;
        for (int j = 0; j < n; j++) { h ^= h << 13; h ^= h >> 17; h ^= h << 5; vout[j] = (h & 0xffff) / 65535.0 - 0.5; }
        for (int it = 0; it < 30; it++) {
            double nv = 0; for (int j = 0; j < n; j++) nv += vout[j] * vout[j]; nv = sqrt(nv);
            for (int j = 0; j < n; j++) w[j] = vout[j] / nv;
            memcpy(vout, w, sizeof(double) * n);
            schol_solve(Sf.F, vout, work);
        }
        double nv = 0; for (int j = 0; j < n; j++) nv += vout[j] * vout[j]; nv = sqrt(nv);
        for (int j = 0; j < n; j++) vout[j] /= nv;
        free(w); free(work);
    }
    sfac_free(&Sf);
    return lm;
}

/* the k smallest eigenpairs of S_k (k <= 3) by subspace inverse iteration with S + s I, s just above -lambda_min
 * (positive definite), and Rayleigh-Ritz; returns how many came out below -eps (vals ascending, vecs n x k) */
static void smv(const IBlk *B, const double *vk, const double *x, double *y) {   /* y = S x */
    for (int j = 0; j < B->n; j++) y[j] = 0;
    for (int p = 0; p < B->np; p++) { const int j = B->pj[p], l = B->pk[p]; if (j == l) y[j] += vk[p] * x[j]; else { y[j] += 0.5 * vk[p] * x[l]; y[l] += 0.5 * vk[p] * x[j]; } }
}
static int block_negvecs(const IBlk *B, const double *vk, double lm, double eps, int k, double *vals, double *vecs) {
    const int n = B->n; if (k > n) k = n;
    SFac Sf; sfac_setup(&Sf, B);
    double s = -lm + 1e-3 * fabs(lm) + 1e-12;
    if (!sfac_try(&Sf, B, vk, s)) { s = 2 * s + 1e-9; if (!sfac_try(&Sf, B, vk, s)) { sfac_free(&Sf); return 0; } }
    double *V = ix(sizeof(double) * ((size_t)n * k + 1)), *work = ix(sizeof(double) * (n + 1)), *w = ix(sizeof(double) * (n + 1));
    unsigned h = 2463534242u;
    for (size_t q = 0; q < (size_t)n * k; q++) { h ^= h << 13; h ^= h >> 17; h ^= h << 5; V[q] = (h & 0xffff) / 65535.0 - 0.5; }
    for (int it = 0; it < 40; it++) {
        for (int c = 0; c < k; c++) schol_solve(Sf.F, V + (size_t)c * n, work);
        for (int c = 0; c < k; c++) {   /* Gram-Schmidt, twice */
            double *v = V + (size_t)c * n;
            for (int pass = 0; pass < 2; pass++) for (int d = 0; d < c; d++) { const double *u = V + (size_t)d * n; double t = 0; for (int j = 0; j < n; j++) t += u[j] * v[j]; for (int j = 0; j < n; j++) v[j] -= t * u[j]; }
            double nv = 0; for (int j = 0; j < n; j++) nv += v[j] * v[j]; nv = sqrt(nv); if (nv == 0) nv = 1; for (int j = 0; j < n; j++) v[j] /= nv;
        }
    }
    /* Rayleigh-Ritz: T = V'SV (k x k), Jacobi */
    double T[9] = { 0 }, U[9] = { 0 };
    for (int c = 0; c < k; c++) { smv(B, vk, V + (size_t)c * n, w); for (int d = 0; d < k; d++) { double t = 0; const double *u = V + (size_t)d * n; for (int j = 0; j < n; j++) t += u[j] * w[j]; T[d * 3 + c] = t; } }
    for (int c = 0; c < 3; c++) U[c * 3 + c] = 1;
    for (int sweep = 0; sweep < 30; sweep++) for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) {
        const double apq = 0.5 * (T[a * 3 + b] + T[b * 3 + a]); if (fabs(apq) < 1e-300) continue;
        const double th = 0.5 * atan2(2 * apq, T[b * 3 + b] - T[a * 3 + a]), cs = cos(th), sn = sin(th);
        for (int r = 0; r < k; r++) { const double x = T[r * 3 + a], y = T[r * 3 + b]; T[r * 3 + a] = cs * x - sn * y; T[r * 3 + b] = sn * x + cs * y; }
        for (int r = 0; r < k; r++) { const double x = T[a * 3 + r], y = T[b * 3 + r]; T[a * 3 + r] = cs * x - sn * y; T[b * 3 + r] = sn * x + cs * y; }
        for (int r = 0; r < k; r++) { const double x = U[r * 3 + a], y = U[r * 3 + b]; U[r * 3 + a] = cs * x - sn * y; U[r * 3 + b] = sn * x + cs * y; }
    }
    int ord[3] = { 0, 1, 2 };
    for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) if (T[ord[b] * 3 + ord[b]] < T[ord[a] * 3 + ord[a]]) { const int t = ord[a]; ord[a] = ord[b]; ord[b] = t; }
    int nneg = 0;
    for (int a = 0; a < k; a++) {
        const int c = ord[a]; vals[a] = T[c * 3 + c];
        double *out = vecs + (size_t)a * n; for (int j = 0; j < n; j++) { double t = 0; for (int d = 0; d < k; d++) t += V[(size_t)d * n + j] * U[d * 3 + c]; out[j] = t; }
        if (vals[a] < -eps) nneg++;
    }
    free(V); free(work); free(w); sfac_free(&Sf);
    return nneg;
}

/* the start: rank one, all rows parallel; a row's square from the constraints on diagonal entries alone,
 * in the middle of their range (geometric mean of the ratios), other rows at 1. In a block with a fixed row
 * (X_00 = c) the other rows are moments of order one, R_j = X_0j / sqrt(c): a bound X_jj <= U says only
 * |R_j| <= sqrt(U), whose centre is 0, so there only constraints that bound X_jj from below (or fix it)
 * move a row away from 0 */
static void ip_start(IP *Q, double *xl) {
    const int m = Q->m, K = Q->K;
    double **t = ix(sizeof(double *) * (K + 1)); double **ls = ix(sizeof(double *) * (K + 1)); int **nc = ix(sizeof(int *) * (K + 1));
    double **lsl = ix(sizeof(double *) * (K + 1)); int **ncl = ix(sizeof(int *) * (K + 1));   /* the same over the lower bounds */
    for (int k = 0; k < K; k++) {
        IBlk *B = &Q->bk[k];
        t[k] = ix(sizeof(double) * (B->n + 1)); ls[k] = iz(sizeof(double) * (B->n + 1)); nc[k] = iz(sizeof(int) * (B->n + 1));
        lsl[k] = iz(sizeof(double) * (B->n + 1)); ncl[k] = iz(sizeof(int) * (B->n + 1));
        for (int j = 0; j < B->n; j++) t[k][j] = B->zr[j] ? 0.0 : 1.0;
        if (B->fixrow >= 0) t[k][B->fixrow] = B->fixval * B->fixval;
    }
    char *diagonly = ix(m + 1);
    for (int i = 0; i < m; i++) {
        int ok = Q->arp[i + 1] > Q->arp[i];
        for (int a = Q->arp[i]; a < Q->arp[i + 1] && ok; a++) { const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; if (Q->bk[k].pj[p] != Q->bk[k].pk[p]) ok = 0; }
        diagonly[i] = ok;
    }
    for (int i = 0; i < m; i++) {
        if (!diagonly[i]) continue;
        double var = 0, cst = 0, sl = 0; int nsl = 0;
        for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
            const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k], j = Q->bk[k].pj[p];
            if (j == Q->bk[k].fixrow) cst += Q->av[a] * t[k][j]; else var += Q->av[a] * t[k][j];
        }
        for (int a = Q->lrp[i]; a < Q->lrp[i + 1]; a++) { sl = Q->lrv[a]; nsl++; }
        const double rhs = Q->b[i] - cst;
        if (nsl > 1 || var == 0 || rhs / var <= 0) continue;
        const int lower = nsl == 0 || sl * var < 0;   /* var(X) = rhs - sl s with s >= 0: a lower bound when sl var < 0 */
        const double lr = log(rhs / var);
        for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
            const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k], j = Q->bk[k].pj[p];
            if (j == Q->bk[k].fixrow || Q->bk[k].zr[j]) continue;
            ls[k][j] += lr; nc[k][j]++;
            if (lower) { lsl[k][j] += lr; ncl[k][j]++; }
        }
    }
    for (int k = 0; k < K; k++) {
        IBlk *B = &Q->bk[k]; const int r = B->r;
        free(B->R); B->R = iz(sizeof(double) * ((size_t)B->n * r + 1));
        for (int j = 0; j < B->n; j++) {
            double tj = t[k][j];
            if (B->zr[j]) continue;
            if (j != B->fixrow) {
                if (B->fixrow >= 0) tj = ncl[k][j] ? tj * exp(lsl[k][j] / ncl[k][j]) : 0;
                else if (nc[k][j]) tj *= exp(ls[k][j] / nc[k][j]);
            }
            B->R[(size_t)j * r] = j == B->fixrow ? B->fixval : (getenv("BRISK_BM2HALF") && B->fixrow < 0 ? sqrt(0.5) : sqrt(tj));
        }
        free(t[k]); free(ls[k]); free(nc[k]); free(lsl[k]); free(ncl[k]);
    }
    free(t); free(ls); free(nc); free(lsl); free(ncl); free(diagonly);
    /* rows of a block with a fixed row that a localizing constraint bounds, alpha X_jj + beta X_0j <= b' (alpha > 0;
     * (u - x)(x - l) >= 0 of a variable x = X_0j / sqrt(c)), start at the centre of that interval,
     * R_j = -beta sqrt(c) / (2 alpha): the middle of [l, u] */
    if (!getenv("BRISK_BM2NOCENTRE")) for (int i = 0; i < m; i++) {
        int ok = Q->arp[i + 1] > Q->arp[i], kd = -1, jd = -1, nd = 0; double sg = 0, alpha = 0, beta = 0;
        for (int a = Q->arp[i]; a < Q->arp[i + 1] && ok; a++) {
            const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; const IBlk *B = &Q->bk[k];
            const int j = B->pj[p], l = B->pk[p];
            if (B->fixrow < 0) { ok = 0; break; }
            if (j == l) { if (j == B->fixrow) continue; if (nd && (kd != k || jd != j)) { ok = 0; break; } nd = 1; kd = k; jd = j; }
        }
        if (!ok || !nd) continue;
        const IBlk *B = &Q->bk[kd];
        for (int a = Q->arp[i]; a < Q->arp[i + 1] && ok; a++) {
            const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; const int j = Q->bk[k].pj[p], l = Q->bk[k].pk[p];
            if (j == l) { if (j != B->fixrow) alpha += Q->av[a]; }
            else if (k == kd && ((j == B->fixrow && l == jd) || (l == B->fixrow && j == jd))) beta += Q->av[a];
            else ok = 0;
        }
        const int nlp = Q->lrp[i + 1] - Q->lrp[i];
        if (!ok || alpha == 0 || beta == 0 || nlp > 1) continue;
        sg = alpha > 0 ? 1 : -1;
        if (nlp == 1 && sg * Q->lrv[Q->lrp[i]] <= 0) continue;   /* oriented as alpha X_jj + beta X_0j <= b' with alpha > 0 */
        Q->bk[kd].R[(size_t)jd * Q->bk[kd].r] = -(sg * beta) * B->fixval / (2 * sg * alpha);
    }
    /* LP variables: the value that satisfies a row of their own, at least 1e-2; 1 otherwise */
    double *u = ix(sizeof(double) * (Q->E + 1)), *g = ix(sizeof(double) * (m + 1));
    for (int j = 0; j < Q->nl; j++) xl[j] = 0;
    ip_eval(Q, xl, u, g, NULL);
    for (int j = 0; j < Q->nl; j++) {
        double v = 1.0;
        if (Q->lcp[j + 1] - Q->lcp[j] == 1) { const int i = Q->lci[Q->lcp[j]]; const double a = Q->lcv[Q->lcp[j]]; v = -g[i] / a; }
        xl[j] = getenv("BRISK_BM2SL") ? atof(getenv("BRISK_BM2SL")) : fmax(v, 1e-2);
    }
    free(u); free(g);
}

/* constraints with one diagonal entry of a block and nothing else: b = 0 makes the row a zero row (X >= 0 and
 * X_jj = 0 force row and column j to zero: the facial reduction of a zero diagonal; the constraint is satisfied
 * identically and set aside), b > 0 the block's fixed row (one per block). The gauge row of a block without a
 * fixed row: its first row that is not a zero row */
static void ip_fixrows(IP *Q) {
    for (int i = 0; i < Q->m; i++) {
        if (Q->arp[i + 1] - Q->arp[i] != 1 || Q->lrp[i + 1] != Q->lrp[i]) continue;
        const int a = Q->arp[i], e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k];
        IBlk *B = &Q->bk[k];
        if (B->pj[p] != B->pk[p] || Q->av[a] == 0) continue;
        if (Q->b[i] == 0) { if (!B->zr[B->pj[p]]) { B->zr[B->pj[p]] = 1; B->nz++; } Q->dead[i] = 1; continue; }
        if (B->fixrow >= 0 || Q->b[i] / Q->av[a] <= 0) continue;
        B->fixrow = B->pj[p]; B->fixval = sqrt(Q->b[i] / Q->av[a]); B->fixcon = i;
    }
    for (int k = 0; k < Q->K; k++) {
        IBlk *B = &Q->bk[k];
        if (B->fixrow >= 0 && B->zr[B->fixrow]) { fprintf(stderr, "brisk: -bm 2: block %d: row %d is fixed both at zero and at a positive value (infeasible)\n", k, B->fixrow); B->fixrow = -1; }
        if (B->fixrow < 0) { B->grow = 0; while (B->grow < B->n - 1 && B->zr[B->grow]) B->grow++; }
    }
}

/* the multiplier of a fixed row's constraint: row fixrow of S R vanishes (its variables are fixed, so the
 * method leaves it at its start) */
static void ip_fixmult(IP *Q, double *lam, double *vE, double eps) {
    for (int k = 0; k < Q->K; k++) {
        IBlk *B = &Q->bk[k]; const int j = B->fixrow; if (j < 0) continue;
        const double *vk = vE + Q->poff[k]; double s = 0;
        for (int q = B->sp[j]; q < B->sp[j + 1]; q++) { const int c = B->sj[q], p = B->spos[q]; s += (c == j ? vk[p] : 0.5 * vk[p]) * B->R[(size_t)c * B->r]; }
        const int i = B->fixcon; const double a = Q->av[Q->arp[i]];
        lam[i] -= s / (a * B->R[(size_t)j * B->r]);
    }
    ip_svals(Q, lam, vE);
    /* the multiplier of X_jj = 0 is free (b = 0): raise S_jj to |s|^2 / delta, s the rest of row j of S, so that
     * the Schur complement S' - s s' / S_jj is within delta of S' (S without row and column j), the matrix the
     * reduced program is certified on; delta = eps / 10 shared by the block's zero rows */
    for (int i = 0; i < Q->m; i++) {
        if (!Q->dead[i]) continue;
        const int a = Q->arp[i], e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; const IBlk *B = &Q->bk[k]; const int j = B->pj[p];
        double s2 = 0; for (int q = B->sp[j]; q < B->sp[j + 1]; q++) if (B->sj[q] != j) { const double v = 0.5 * vE[Q->poff[k] + B->spos[q]]; s2 += v * v; }
        const double delta = 0.1 * eps / B->nz, want = s2 / delta + delta;
        if (vE[e] < want) lam[i] += (want - vE[e]) / Q->av[a];
    }
    ip_svals(Q, lam, vE);
}

/* add one column to block k along v (the gauge entry zeroed), scaled to eps times the largest entry of column 0 */
static int dcmp_desc(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? 1 : x > y ? -1 : 0; }

/* a new column along v, scaled to rel times the largest entry of the first column; keep (NULL: all rows) marks the
 * rows where it may be nonzero, the others are fixed at 0 */
static void ip_addcol_on(IBlk *B, const double *v, double rel, const char *keep) {
    const int n = B->n, r = B->r;
    double *Rn = iz(sizeof(double) * ((size_t)n * (r + 1) + 1));
    double c0 = 0, vm = 0;
    for (int j = 0; j < n; j++) { c0 = fmax(c0, fabs(B->R[(size_t)j * r])); if (j != B->fixrow && j != B->grow && !B->zr[j] && (!keep || keep[j])) vm = fmax(vm, fabs(v[j])); }
    const double sc = vm > 0 ? rel * c0 / vm : 0;
    for (int j = 0; j < n; j++) {
        memcpy(Rn + (size_t)j * (r + 1), B->R + (size_t)j * r, sizeof(double) * r);
        Rn[(size_t)j * (r + 1) + r] = (j == B->fixrow || j == B->grow || B->zr[j] || (keep && !keep[j])) ? 0.0 : sc * v[j];
    }
    if (keep || B->fz) {
        char *fz = iz((size_t)n * (r + 1) + 1);
        for (int j = 0; j < n; j++) {
            if (B->fz) memcpy(fz + (size_t)j * (r + 1), B->fz + (size_t)j * r, r);
            fz[(size_t)j * (r + 1) + r] = keep && !keep[j];
        }
        free(B->fz); B->fz = fz;
    }
    free(B->R); B->R = Rn; B->r = r + 1;
}
static void ip_addcol(IBlk *B, const double *v, double rel) { ip_addcol_on(B, v, rel, NULL); }

/* a block's part of the lower bound: T min(0, lambda_min(S)) with the trace bound T, or, with indices that have no
 * bound, -delta TB for the smallest delta with S + delta P_B >= 0 (P_B the bounded indices, TB their trace bound);
 * -inf when there is none. vk: S by positions */
static double block_bound(const IBlk *B, double *vk, double T, double TB, const char *bd, double eps, double big, double cs, int k, int verbose) {
    double cb = 0;
    int cert; double lm = block_lmin(B, vk, eps, NULL, &cert);
    if (lm < 0 && isfinite(T)) {
        /* the bound needs lambda_min to the rounding of the factorization: bisect the shift further, between
         * the Gershgorin bound and the last shift that factored */
        SFac Sf; sfac_setup(&Sf, B);
        double hi = -lm, lo = hi;
        { double *gd = iz(sizeof(double) * (B->n + 1));
          for (int p = 0; p < B->np; p++) { if (B->pj[p] == B->pk[p]) gd[B->pj[p]] += vk[p]; else { gd[B->pj[p]] -= 0.5 * fabs(vk[p]); gd[B->pk[p]] -= 0.5 * fabs(vk[p]); } }
          lo = INFINITY; for (int j = 0; j < B->n; j++) lo = fmin(lo, gd[j]); lo = -lo; free(gd); }
        if (lo >= hi) lo = hi - fabs(hi) - 1e-300;
        for (int it = 0; it < 100 && hi - lo > 1e-3 * fabs(hi) && hi > 1e-15 * big; it++) {
            const double mid = 0.5 * (lo + hi);
            if (sfac_try(&Sf, B, vk, mid)) hi = mid; else lo = mid;
        }
        sfac_free(&Sf);
        lm = -hi;
    }
    if (lm < 0 && isinf(T)) {
        /* indices without a bound: the smallest delta with S + delta P_B positive semidefinite (P_B the diagonal
         * of the bounded indices; S along the unbounded ones must be positive semidefinite by itself), then
         * <S, X> >= -delta sum_B U_j; by factorizations, bisected */
        SFac Sf; sfac_setup(&Sf, B);
        #define TRYB(dl_, ok_) do { schol_zero(Sf.F); double *V_ = schol_values(Sf.F); \
            for (int p_ = 0; p_ < B->np; p_++) V_[Sf.offP[p_]] += B->pj[p_] == B->pk[p_] ? vk[p_] : 0.5 * vk[p_]; \
            for (int j_ = 0; j_ < B->n; j_++) if (bd[j_]) V_[Sf.offD[j_]] += (dl_); \
            ok_ = schol_factor(Sf.F, 0.0) == 0; } while (0)
        double hi = fmax(-2 * lm, eps), lo = 0; int okb = 0;
        for (int t = 0; t < 60 && !okb; t++) { TRYB(hi, okb); if (!okb) { lo = hi; hi *= 4; } }
        if (okb) {
            for (int it = 0; it < 100 && hi - lo > 1e-3 * hi && hi > 1e-15 * big; it++) { const double mid = 0.5 * (lo + hi); int o2; TRYB(mid, o2); if (o2) hi = mid; else lo = mid; }
            cb = -hi * TB;
            if (verbose >= 0) printf("; block %d: lambda_min %.2e, unbounded indices: S + %.2e P_B >= 0, trace bound on the rest %.4g", k, lm * cs, hi * cs, TB);
        } else {
            cb = -INFINITY;
            if (verbose >= 0) printf("; block %d: lambda_min %.2e, S is not positive semidefinite along indices without a bound", k, lm * cs);
        }
        #undef TRYB
        sfac_free(&Sf);
    } else {
        if (lm < 0) cb = T * lm;
        if (verbose >= 0) printf("; block %d: lambda_min %.2e, trace bound %.4g", k, lm * cs, T);
    }
    return cb;
}

/* a lower bound on the optimal value from any multipliers, whatever the accuracy of the solve (weak duality with
 * bounds on the primal variables). For X, x feasible and y = -lam: <C,X> + c'x - b'y = sum_k <S_k, X_k> + z'x,
 * S_k = C_k - A_k*(y), z = c - A_l'y, so the optimum is at least
 *     b'y + sum_k T_k min(0, lambda_min(S_k)) + sum_j u_j min(0, z_j)
 * with tr X_k <= T_k and x_j <= u_j on the feasible set.
 *  - The sign of z is restored where one row decides it (an LP variable in a single row: that row's multiplier
 *    is moved to z_j = 0).
 *  - A free index j (X_jj in no data: S_jj = 0 for every y) needs S_jl = 0: a position (j, l) that one row
 *    alone touches (e.g. X_0j = 0) gets that row's multiplier so; X_jj is then unbounded but multiplies
 *    nothing, and the index leaves T_k and lambda_min.
 *  - T_k sums bounds U_j >= X_jj: a row on diagonal entries alone with coefficients of one sign and no LP
 *    variable, or one of the opposite sign (a'X <= b), gives X_jj <= b / a_jj; a row on one diagonal entry X_jj
 *    and its moment X_0j in the fixed row (X_00 = c: a localizing constraint such as (u - x)(x - l) >= 0)
 *    reads alpha X_jj + beta X_0j <= b' and, with |X_0j| <= sqrt(c X_jj), gives sqrt(X_jj) <= t, the root of
 *    alpha t^2 - |beta| sqrt(c) t - b'; entries on the fixed diagonal are constants.
 *  - lambda_min(S_k) is taken at a shift where S_k - lambda I factored.
 *  - u_j: a row with x_j as its only LP variable gives |a_ij| x_j <= |b_i| + sum_k ||A_ik||_F T_k.
 * On the problem as read after the sign reduction (which keeps the optimal value); exact up to the rounding of
 * the factorizations. Returns -inf when a needed bound is missing. lam is not changed. */
static double ip_lower_bound(const IP *Q, const double *lam0, double tol, double nC1, int verbose) {
    const int m = Q->m, K = Q->K;
    double *lam = ix(sizeof(double) * (m + 1)); memcpy(lam, lam0, sizeof(double) * m);
    for (int j = 0; j < Q->nl; j++) {
        if (Q->lcp[j + 1] - Q->lcp[j] != 1) continue;
        const int a = Q->lcp[j], i = Q->lci[a];
        const double zj = Q->cl[j] + Q->lcv[a] * lam[i];
        if (zj < 0) lam[i] -= zj / Q->lcv[a];
    }
    double *vE = ix(sizeof(double) * (Q->E + 1)); ip_svals(Q, lam, vE);
    /* free indices */
    char **fr = ix(sizeof(char *) * (K + 1)); int nfree = 0, freefail = 0;
    for (int k = 0; k < K; k++) {
        const IBlk *B = &Q->bk[k]; fr[k] = iz(B->n + 1);
        for (int p = 0; p < B->np; p++) {
            const int j = B->pj[p], e = Q->poff[k] + p;
            if (j != B->pk[p] || j == B->fixrow || B->zr[j]) continue;
            if (Q->cE[e] == 0 && Q->acp[e + 1] == Q->acp[e]) { fr[k][j] = 1; nfree++; }
        }
        for (int j = 0; j < B->n; j++) {
            if (!fr[k][j]) continue;
            for (int q = B->sp[j]; q < B->sp[j + 1]; q++) {
                const int l = B->sj[q], e = Q->poff[k] + B->spos[q];
                if (l == j || vE[e] == 0) continue;
                int fixed = 0;
                for (int a = Q->acp[e]; a < Q->acp[e + 1] && !fixed; a++) {   /* a row on this position alone */
                    const int i = Q->aci[a];
                    if (Q->arp[i + 1] - Q->arp[i] == 1 && Q->lrp[i + 1] == Q->lrp[i]) { lam[i] -= vE[e] / Q->acv[a]; vE[e] = 0; fixed = 1; }
                }
                if (!fixed) freefail = 1;
            }
        }
    }
    if (nfree) ip_svals(Q, lam, vE);
    /* trace bounds over the indices that are not free */
    double **U = ix(sizeof(double *) * (K + 1)), *T = ix(sizeof(double) * (K + 1));
    for (int k = 0; k < K; k++) { const IBlk *B = &Q->bk[k]; U[k] = ix(sizeof(double) * (B->n + 1)); for (int j = 0; j < B->n; j++) U[k][j] = B->zr[j] || fr[k][j] ? 0.0 : j == B->fixrow ? B->fixval * B->fixval : INFINITY; }
    for (int i = 0; i < m; i++) {
        int ok = Q->arp[i + 1] > Q->arp[i], kd = -1, jd = -1, nd = 0, noff = 0; double sg = 0, cst = 0, beta = 0;
        for (int a = Q->arp[i]; a < Q->arp[i + 1] && ok; a++) {
            const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; const IBlk *B = &Q->bk[k];
            const int j = B->pj[p], l = B->pk[p];
            if (j == l) {
                if (j == B->fixrow) { cst += Q->av[a] * B->fixval * B->fixval; continue; }
                if (Q->av[a] == 0) continue;
                const double t = Q->av[a] > 0 ? 1 : -1; if (sg == 0) sg = t; else if (t != sg) ok = 0;
                if (nd == 0 || kd != k || jd != j) nd++;
                kd = k; jd = j;
            } else noff++;
        }
        if (!ok || sg == 0) continue;
        const int nlp = Q->lrp[i + 1] - Q->lrp[i];
        if (nlp > 1 || (nlp == 1 && sg * Q->lrv[Q->lrp[i]] <= 0)) continue;
        if (noff == 0) {
            for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
                const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k], j = Q->bk[k].pj[p];
                if (Q->av[a] == 0 || j == Q->bk[k].fixrow) continue;
                U[k][j] = fmin(U[k][j], fmax(sg * (Q->b[i] - cst), 0.0) / (sg * Q->av[a]));
            }
            continue;
        }
        /* one diagonal index and its moment in the fixed row */
        if (nd != 1) continue;
        const IBlk *B = &Q->bk[kd]; double alpha = 0;
        for (int a = Q->arp[i]; a < Q->arp[i + 1] && ok; a++) {
            const int e = Q->api[a], k = Q->pb[e], p = e - Q->poff[k]; const int j = Q->bk[k].pj[p], l = Q->bk[k].pk[p];
            if (j == l) { if (j != Q->bk[k].fixrow) alpha += sg * Q->av[a]; }
            else if (k == kd && ((j == B->fixrow && l == jd) || (l == B->fixrow && j == jd))) beta += sg * Q->av[a];
            else ok = 0;
        }
        if (!ok || alpha <= 0 || B->fixrow < 0) continue;
        const double c = B->fixval * B->fixval, bp = sg * (Q->b[i] - cst), disc = beta * beta * c + 4 * alpha * bp;
        if (disc < 0) continue;
        const double t = (fabs(beta) * sqrt(c) + sqrt(disc)) / (2 * alpha);
        U[kd][jd] = fmin(U[kd][jd], t * t);
    }
    /* T[k] over all indices (inf if one is unbounded) and TB[k] over the bounded ones; bd[k][j] marks them */
    double *TB = ix(sizeof(double) * (K + 1)); char **bd = ix(sizeof(char *) * (K + 1));
    for (int k = 0; k < K; k++) {
        double t = 0, tb = 0; bd[k] = iz(Q->bk[k].n + 1);
        for (int j = 0; j < Q->bk[k].n; j++) { t += U[k][j]; if (isfinite(U[k][j])) { tb += U[k][j]; bd[k][j] = !fr[k][j]; } }
        T[k] = t; TB[k] = tb;
    }
    double L = 0; for (int i = 0; i < m; i++) L -= Q->b[i] * lam[i];
    const double eps = tol * (1 + nC1) / Q->cs;
    if (verbose >= 0) printf("bm2: lower bound from y: b'y %.10e%s", L * Q->cs, freefail ? " (free indices with data left)" : "");
    if (freefail) L = -INFINITY;
    for (int k = 0; k < K; k++) {
        const IBlk *B = &Q->bk[k];
        double *vk = ix(sizeof(double) * (B->np + 1)); memcpy(vk, vE + Q->poff[k], sizeof(double) * B->np);
        double big = 1; for (int p = 0; p < B->np; p++) big = fmax(big, fabs(vk[p]));
        for (int p = 0; p < B->np; p++) if (B->pj[p] == B->pk[p] && fr[k][B->pj[p]]) vk[p] = big;   /* rows of S that are zero: out of lambda_min */
        double cb = block_bound(B, vk, T[k], TB[k], bd[k], eps, big, Q->cs, k, verbose);
        if (cb < 0 && !getenv("BRISK_BM2NODIAG")) {
            /* per-index shifts: <S, X> >= <S + Diag(d), X> - sum_j d_j U_j for d >= 0, which uses the bounds U_j index
             * by index where the trace bound uses only their sum. d is built from the most negative eigenpairs of
             * S + Diag(d): (lambda, u) is covered by d_j += |lambda| |u_j| a / sqrt(U_j), a = sum_l |u_l| sqrt(U_l)
             * (Diag of it minus |lambda| u u' is positive semidefinite: its Schur complement is 1 - sum_j u_j^2
             * |lambda| / d_j = 0), at the cost |lambda| a^2, at most |lambda| sum_j U_j (Cauchy-Schwarz); localized
             * vectors cost far less. The rest is bounded as before, and the better of the two bounds is kept */
            const int n = B->n;
            double *d = iz(sizeof(double) * (n + 1)), *u = ix(sizeof(double) * (n + 1)), *vk2 = ix(sizeof(double) * (B->np + 1));
            int *dpos = ix(sizeof(int) * (n + 1));
            for (int j = 0; j < n; j++) dpos[j] = -1;
            for (int p = 0; p < B->np; p++) if (B->pj[p] == B->pk[p]) dpos[B->pj[p]] = p;
            memcpy(vk2, vk, sizeof(double) * B->np);
            double cost = 0; int steps = 0;
            const int maxsteps = getenv("BRISK_BM2DIAGSTEPS") ? atoi(getenv("BRISK_BM2DIAGSTEPS")) : 8;
            for (; steps < maxsteps; steps++) {
                int c2; const double l2 = block_lmin(B, vk2, eps, u, &c2);
                if (l2 >= -eps) break;
                double a = 0;
                for (int j = 0; j < n; j++) if (bd[k][j] && U[k][j] > 0) a += fabs(u[j]) * sqrt(U[k][j]);
                if (!(a > 0)) break;
                for (int j = 0; j < n; j++) {
                    if (!bd[k][j] || u[j] == 0 || dpos[j] < 0) continue;
                    const double dd = U[k][j] > 0 ? -l2 * fabs(u[j]) * a / sqrt(U[k][j]) : 10 * fmax(big, -l2);   /* X_jj = 0: free of cost */
                    vk2[dpos[j]] += dd; d[j] += dd;
                    if (U[k][j] > 0) cost += dd * U[k][j];
                }
            }
            if (steps > 0) {
                const double cb2 = block_bound(B, vk2, T[k], TB[k], bd[k], eps, big, Q->cs, k, -1) - cost;
                if (verbose >= 0) printf("; with %d per-index shifts: their cost %.3e, the block's bound %.3e (against %.3e)", steps, cost * Q->cs, cb2 * Q->cs, cb * Q->cs);
                cb = fmax(cb, cb2);
            }
            free(d); free(u); free(vk2); free(dpos);
        }
        L += cb;
        free(vk); free(fr[k]); free(bd[k]); free(U[k]);
    }
    free(fr); free(bd); free(TB); free(U);
    int nmiss = 0;
    for (int j = 0; j < Q->nl; j++) {
        double zj = Q->cl[j]; for (int a = Q->lcp[j]; a < Q->lcp[j + 1]; a++) zj += Q->lcv[a] * lam[Q->lci[a]];
        if (zj >= 0) continue;
        double u = INFINITY;
        for (int a = Q->lcp[j]; a < Q->lcp[j + 1]; a++) {
            const int i = Q->lci[a];
            if (Q->lrp[i + 1] - Q->lrp[i] != 1) continue;
            double s2 = fabs(Q->b[i]);
            double *fk = iz(sizeof(double) * (K + 1));
            for (int q = Q->arp[i]; q < Q->arp[i + 1]; q++) { const int e = Q->api[q], k = Q->pb[e], p = e - Q->poff[k]; fk[k] += (Q->bk[k].pj[p] == Q->bk[k].pk[p] ? 1.0 : 0.5) * Q->av[q] * Q->av[q]; }
            for (int k = 0; k < K; k++) if (fk[k] > 0) s2 += sqrt(fk[k]) * T[k];
            free(fk);
            u = fmin(u, s2 / fabs(Q->lcv[a]));
        }
        if (isinf(u)) nmiss++;
        L += u * zj;
    }
    if (verbose >= 0) printf("%s\nbm2: lower bound %.10e\n", nmiss ? "; LP variables without a bound" : "", L * Q->cs);
    free(lam); free(vE); free(T);
    return L * Q->cs;
}

/* The escape from the saddle at a rank increase (the second-order curve of the Burer-Monteiro method): with V_k the
 * new directions (unit vectors along negative Ritz vectors of S_k, n_k x q_k by columns, zero on the fixed, gauge and
 * zero rows),
 *     R_k(tau) = [R_k + tau W_k, sqrt(tau) V_k],   x_l(tau) = x_l + tau w,
 * with (W, w) the least-norm solution (w weighted by 1/x_l) of J_R W + A_l w = -A(V V'): the constraints hold to
 * second order in tau, and the Lagrangian falls by tau sum v'S v. A small new column instead (1e-2 of the first)
 * is a KKT point again for the interior-point method, which pulls it back to the saddle (4661-bus case: the round
 * converged in 25 iterations, objective up, lambda_min unchanged). tau from a line search on f + nu |g|_1 (nu the
 * largest multiplier) over a geometric grid. Returns 1 with R and x_l set to the best point, 0 if no tau decreases
 * the merit (the caller then adds small columns). */
static int ip_escape(IP *Q, double *xl, const double *lam, double **V, const int *q, int verbose) {
    for (int k = 0; k < Q->K; k++) if (Q->bk[k].fz) return 0;   /* (columns kept on supports: not handled here) */
    const int K = Q->K, m = Q->m;
    KKT Kt; kkt_setup(&Kt, Q);
    const int nx = Q->nx, nR = Q->nR, nl = Q->nl, N = Kt.N;
    double *u = ix(sizeof(double) * (Q->E + 1)), *g = ix(sizeof(double) * (m + 1)), *r = iz(sizeof(double) * (m + 1));
    double *rhs = iz(sizeof(double) * (N + 1)), *d = ix(sizeof(double) * (N + 1)), *dl = ix(sizeof(double) * (nl + 1)), *xlt = ix(sizeof(double) * (nl + 1));
    /* r = -A(V V') */
    for (int k = 0; k < K; k++) {
        const IBlk *B = &Q->bk[k]; if (!q[k]) continue;
        for (int p = 0; p < B->np; p++) { double t = 0; for (int c = 0; c < q[k]; c++) t += V[k][(size_t)c * B->n + B->pj[p]] * V[k][(size_t)c * B->n + B->pk[p]]; u[Q->poff[k] + p] = t; }
    }
    for (int i = 0; i < m; i++) { double t = 0; for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) { const int e = Q->api[a], k = Q->pb[e]; if (q[k]) t += Q->av[a] * u[e]; } r[i] = -t; }
    /* least-norm (W, w): [[D, J'], [J, -delta]] [d; z] = [0; r], D = I on R, 1/x_l^2 on x_l */
    for (int j = 0; j < nl; j++) dl[j] = 1.0 / (xl[j] * xl[j]);
    kkt_assemble(&Kt, Q, u, 0, 1.0, dl, 1e-12);
    int ok = kkt_factor(&Kt) == 0;
    if (ok) {
        for (int i = 0; i < m; i++) rhs[nx + i] = r[i];
        kkt_solve(&Kt, rhs, d);
        if (verbose >= 2) {   /* the residual of J d = r */
            double *t1 = iz(sizeof(double) * (N + 1)), *t2 = ix(sizeof(double) * (N + 1)); double e = 0, rn = 0;
            for (int v = 0; v < nx; v++) t1[v] = d[v];
            ldl_mv(&Kt.L, t1, t2);
            for (int i = 0; i < m; i++) { e = fmax(e, fabs(t2[nx + i] - r[i])); rn = fmax(rn, fabs(r[i])); }
            printf("        escape: |J d - r| %.2e of |r| %.2e\n", e, rn);
            if (getenv("BRISK_BM2ESCROW")) {
                const int i = atoi(getenv("BRISK_BM2ESCROW"));
                printf("        row %d: r %.3e, (J d) %.3e, dead %d, LP entries %d\n", i, r[i], t2[nx + i], Q->dead[i], Q->lrp[i + 1] - Q->lrp[i]);
                for (int a = Q->arp[i]; a < Q->arp[i + 1]; a++) {
                    const int e2 = Q->api[a], k = Q->pb[e2], p2 = e2 - Q->poff[k]; const IBlk *B = &Q->bk[k];
                    const int j = B->pj[p2], l = B->pk[p2];
                    printf("          block %d (%d, %d) av %.3e  vids", k, j, l, Q->av[a]);
                    for (int c = 0; c < B->r; c++) printf(" %d/%d", B->vid[(size_t)j * B->r + c], B->vid[(size_t)l * B->r + c]);
                    printf("  zr %d %d fix %d grow %d\n", B->zr[j], B->zr[l], B->fixrow, B->grow);
                }
            }
            free(t1); free(t2);
        }
    }
    kkt_free(&Kt);
    int done = 0;
    if (ok) {
        double f0; ip_eval(Q, xl, u, g, &f0);
        double *g0 = ix(sizeof(double) * (m + 1)); memcpy(g0, g, sizeof(double) * m);
        double nu = 0; for (int i = 0; i < m; i++) if (!Q->dead[i]) nu = fmax(nu, fabs(lam[i]));   /* (the zero rows' multipliers are the certificate's) */
        const double ph0 = f0 + nu * nrm1(g, m);
        double trR = 0; for (int k = 0; k < K; k++) for (size_t t = 0; t < (size_t)Q->bk[k].n * Q->bk[k].r; t++) trR += Q->bk[k].R[t] * Q->bk[k].R[t];
        /* the trial factors */
        double **Rold = ix(sizeof(double *) * (K + 1)), **Rt = ix(sizeof(double *) * (K + 1)), **Rb = ix(sizeof(double *) * (K + 1)); int *rold = ix(sizeof(int) * (K + 1));
        for (int k = 0; k < K; k++) { Rold[k] = Q->bk[k].R; rold[k] = Q->bk[k].r; Rt[k] = ix(sizeof(double) * ((size_t)Q->bk[k].n * (rold[k] + q[k]) + 1)); Rb[k] = NULL; }
        double best = ph0, tbest = 0, prev = INFINITY; int worse = 0;
        double *xlb = ix(sizeof(double) * (nl + 1));
        for (int s = 0; s < 60 && worse < 3; s++) {
            const double tau = 1e-10 * fmax(trR, 1.0) * pow(2.0, s);
            int pos = 1; for (int j = 0; j < nl; j++) { xlt[j] = xl[j] + tau * d[nR + j]; if (xlt[j] <= 0) pos = 0; }
            if (!pos) break;
            for (int k = 0; k < K; k++) {   /* every block moves by tau W; the new columns only where there are directions */
                IBlk *B = &Q->bk[k]; const int r0 = rold[k], r1 = r0 + q[k], n = B->n;
                for (int j = 0; j < n; j++) {
                    for (int l = 0; l < r0; l++) { const int v = B->vid[(size_t)j * r0 + l]; Rt[k][(size_t)j * r1 + l] = Rold[k][(size_t)j * r0 + l] + (v >= 0 ? tau * d[v] : 0.0); }
                    for (int c = 0; c < q[k]; c++) Rt[k][(size_t)j * r1 + r0 + c] = sqrt(tau) * V[k][(size_t)c * n + j];
                }
                B->R = Rt[k]; B->r = r1;
            }
            double ft; ip_eval(Q, xlt, u, g, &ft);
            for (int k = 0; k < K; k++) { Q->bk[k].R = Rold[k]; Q->bk[k].r = rold[k]; }
            const double ph = ft + nu * nrm1(g, m);
            if (verbose >= 2) {
                double dg = 0; int im = -1; double dm = 0; for (int i = 0; i < m; i++) { const double t = fabs(g[i] - g0[i]); dg += t; if (t > dm) { dm = t; im = i; } }
                printf("        escape tau %.2e: f %.10e |g|_1 %.2e merit %.10e  |g - g0|_1 / tau %.2e (largest row %d%s)\n", tau, ft * Q->cs, nrm1(g, m), ph * Q->cs, dg / tau, im, im >= 0 && Q->dead[im] ? ", dead" : "");
            }
            if (ph < best) {
                best = ph; tbest = tau; memcpy(xlb, xlt, sizeof(double) * nl);
                for (int k = 0; k < K; k++) { const size_t sz = (size_t)Q->bk[k].n * (rold[k] + q[k]); free(Rb[k]); Rb[k] = ix(sizeof(double) * (sz + 1)); memcpy(Rb[k], Rt[k], sizeof(double) * sz); }
            }
            worse = ph > prev ? worse + 1 : 0; prev = ph;
        }
        if (tbest > 0 && best < ph0 - 1e-12 * fabs(ph0)) {
            for (int k = 0; k < K; k++) { free(Q->bk[k].R); Q->bk[k].R = Rb[k]; Q->bk[k].r = rold[k] + q[k]; Rb[k] = NULL; }
            memcpy(xl, xlb, sizeof(double) * nl);
            done = 1;
            if (verbose >= 0) { int nq = 0; for (int k = 0; k < K; k++) nq += q[k]; printf("bm2: escape along %d direction(s): tau %.2e, merit %.10e -> %.10e\n", nq, tbest, ph0 * Q->cs, best * Q->cs); }
        }
        for (int k = 0; k < K; k++) { free(Rt[k]); free(Rb[k]); }
        free(Rold); free(Rt); free(Rb); free(rold); free(xlb); free(g0);
    }
    free(u); free(g); free(r); free(rhs); free(d); free(dl); free(xlt);
    return done;
}

int bmip_run(const char *fname, Params *par, const char *yfile, const BriskData *data, BriskResult *res) {
    const double t0 = wtime();
    const int verbose = par->verbose;
    Problem PP, *P = &PP; memset(P, 0, sizeof PP);
    extern int g_read_maxn;
    const int maxn0 = g_read_maxn; g_read_maxn = 2000000000;
    if (data ? problem_from_sdpa_data(data, P) != 0 : read_sdpa(fname, P) != 0) { g_read_maxn = maxn0; return 2; }
    g_read_maxn = maxn0;
    const int m0 = P->m;
    int *bs0 = ix(sizeof(int) * (P->nblk + 1)); const int nb0 = P->nblk;
    for (int k = 0; k < P->nblk; k++) bs0[k] = P->blk[k].type == BLK_LP ? -P->blk[k].n : P->blk[k].n;
    double nb1 = 0, nC1 = 0;   /* |b|_1 and |C|_1 of the file */
    for (int i = 0; i < m0; i++) nb1 += fabs(P->b[i]);
    for (int k = 0; k < P->nblk; k++) for (int e = 0; e < P->blk[k].C.ef; e++) nC1 += fabs(P->blk[k].C.fv[e]);
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SOC) { fprintf(stderr, "brisk: -bm 2: second-order cone blocks are not supported\n"); return 2; }
    if (verbose >= 0) {
        printf("BRISK %s Burer-Monteiro, interior-point method for the factored program (-bm 2)\n", BRISK_VERSION);
        brisk_log_threads(1);
        printf("problem %s: m = %d, %d blocks, read %.2fs\n", fname, m0, P->nblk, wtime() - t0);
    }
    PSSign *SG = sign_reduce(P, verbose, 0, 0.0, 1 << 30, 1.0);
    IP Q; ip_build(&Q, P);
    if (Q.K == 0) { fprintf(stderr, "brisk: -bm 2: no semidefinite block\n"); ip_free(&Q); problem_free(P); sign_free(SG); free(bs0); return 2; }
    ip_equilibrate(&Q);
    ip_fixrows(&Q);
    for (int k = 0; k < Q.K; k++) Q.bk[k].r = 1;
    ip_number(&Q);
    const int m = Q.m;
    double *xl = iz(sizeof(double) * (Q.nl + 1)), *lam = iz(sizeof(double) * (m + 1)), *z = ix(sizeof(double) * (Q.nl + 1));
    ip_start(&Q, xl);
    if (getenv("BRISK_BM2R0")) {   /* experiment: start the blocks without a fixed row at rank r0, the extra columns small and random */
        const int r0 = atoi(getenv("BRISK_BM2R0")); const double sc = getenv("BRISK_BM2R0S") ? atof(getenv("BRISK_BM2R0S")) : 1e-1;
        unsigned hh = 2463534242u;
        for (int k = 0; k < Q.K; k++) {
            IBlk *B = &Q.bk[k]; if (B->fixrow >= 0) continue;
            while (B->r < r0) {
                double *v = ix(sizeof(double) * (B->n + 1));
                for (int j = 0; j < B->n; j++) { hh ^= hh << 13; hh ^= hh >> 17; hh ^= hh << 5; v[j] = (hh & 0xffff) / 65535.0 - 0.5; }
                ip_addcol(B, v, sc); free(v);
            }
        }
        ip_number(&Q);
    }
    if (getenv("BRISK_BM2STARTDBG")) for (int k = 0; k < Q.K; k++) { IBlk *B = &Q.bk[k]; int nzr = 0; for (int j = 0; j < B->n; j++) if (B->R[(size_t)j * B->r] != 0 && j != B->fixrow) { if (nzr < 8) printf("   [start: block %d row %d value %.4e]\n", k, j, B->R[(size_t)j * B->r]); nzr++; } printf("   [start: block %d: %d nonzero rows of %d]\n", k, nzr, B->n); }
    double mu = 0.1, rho = getenv("BRISK_BM2RHO") ? atof(getenv("BRISK_BM2RHO")) : 1e3;
    for (int j = 0; j < Q.nl; j++) z[j] = getenv("BRISK_BM2Z0") ? mu / xl[j] : 1.0;   /* Ipopt's bound multipliers at the start */
    if (verbose >= 0) {
        printf("bm2: %d semidefinite block%s (orders", Q.K, Q.K > 1 ? "s" : ""); for (int k = 0; k < Q.K; k++) printf(" %d", Q.bk[k].n);
        printf("), %d LP variables, m = %d, %d positions; fixed rows:", Q.nl, m, Q.E); for (int k = 0; k < Q.K; k++) printf(" %d", Q.bk[k].fixrow);
        printf("; zero rows:"); for (int k = 0; k < Q.K; k++) printf(" %d", Q.bk[k].nz); printf("\n");
    }
    const double tol = par->tol > 0 ? par->tol : 1e-8;
    const int rmax = getenv("BRISK_BM2RMAX") ? atoi(getenv("BRISK_BM2RMAX")) : 16;
    const double tlim = par->timelimit;
    double *vE = ix(sizeof(double) * (Q.E + 1)), *lmin = ix(sizeof(double) * (Q.K + 1));
    double Lbest = -INFINITY;   /* every round's lower bound is valid: the best of them is the run's */
    int status = 1, totits = 0, polish = 0;
    IStat st; memset(&st, 0, sizeof st);
    for (int round = 0; ; round++) {
        for (int i = 0; i < m; i++) if (Q.dead[i]) lam[i] = 0;
        KKT Kt; const double ta = wtime();
        kkt_setup(&Kt, &Q);
        const double tan = wtime() - ta;
        const int maxit2 = getenv("BRISK_BM2MAXIT") ? atoi(getenv("BRISK_BM2MAXIT")) : 3000;
        if (round > 0) mu = fmax(mu, tol / 10);   /* a new round from a barrier no deeper than tol / 10 per pair: from the
                                                    * final one (down to the duality gap) the iterate stayed at the old point */
        const int rc = getenv("BRISK_BM2ALM") ? ipm_alm(&Q, &Kt, xl, lam, z, &mu, tol, maxit2, verbose >= 1 ? (getenv("BRISK_BM2V") ? 2 : 1) : 0, tlim, t0, &st) :
                       !getenv("BRISK_BM2PEN") ? ipm(&Q, &Kt, xl, lam, z, &mu, tol, maxit2, verbose >= 1 ? (getenv("BRISK_BM2V") ? 2 : 1) : 0, tlim, t0, &st)
                                                : ipm_pen(&Q, &Kt, xl, lam, z, &mu, &rho, tol, maxit2, verbose >= 1 ? 1 : 0, tlim, t0, &st);
        totits += st.its;
        if (verbose >= 0) {
            printf("bm2: ranks"); for (int k = 0; k < Q.K; k++) printf(" %d", Q.bk[k].r);
            printf(": %s, %d iterations (%d factorizations, %d restorations), KKT %d (analysis %.2fs), objective %.10e, %.1fs\n",
                   rc == 0 ? "converged" : rc == 1 ? "iteration limit" : rc == 2 ? "failed" : "time limit", st.its, st.nfac, st.nres, Kt.N, tan, st.f * Q.cs, wtime() - t0);
        }
        kkt_free(&Kt);
        if (verbose >= 0) for (int k = 0; k < Q.K; k++) if (Q.bk[k].r > 1) {   /* the columns' weights: eigenvalues of R'R */
            const IBlk *B = &Q.bk[k]; const int r = B->r; double *G = iz(sizeof(double) * (size_t)r * r);
            for (int j = 0; j < B->n; j++) for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) G[a * r + b] += B->R[(size_t)j * r + a] * B->R[(size_t)j * r + b];
            for (int sweep = 0; sweep < 30; sweep++) for (int a = 0; a < r; a++) for (int b = a + 1; b < r; b++) {
                if (fabs(G[a * r + b]) < 1e-300) continue;
                const double th = 0.5 * atan2(2 * G[a * r + b], G[b * r + b] - G[a * r + a]), cs = cos(th), sn = sin(th);
                for (int c = 0; c < r; c++) { const double x = G[c * r + a], y = G[c * r + b]; G[c * r + a] = cs * x - sn * y; G[c * r + b] = sn * x + cs * y; }
                for (int c = 0; c < r; c++) { const double x = G[a * r + c], y = G[b * r + c]; G[a * r + c] = cs * x - sn * y; G[b * r + c] = sn * x + cs * y; }
            }
            double ev[16]; for (int a = 0; a < r && a < 16; a++) ev[a] = G[a * r + a];
            for (int a = 0; a < r && a < 16; a++) for (int b = a + 1; b < r && b < 16; b++) if (ev[b] > ev[a]) { const double t = ev[a]; ev[a] = ev[b]; ev[b] = t; }
            printf("bm2: block %d: eigenvalues of X:", k); for (int a = 0; a < r && a < 16; a++) printf(" %.2e", ev[a]); printf("\n");
            free(G);
        }
        ip_svals(&Q, lam, vE);
        /* the certificate per block */
        const double eps = tol * (1 + nC1) / Q.cs;
        ip_fixmult(&Q, lam, vE, eps);
        int all = 1, grow = 0;
        double **vv = ix(sizeof(double *) * (Q.K + 1));
        for (int k = 0; k < Q.K; k++) {
            int cert = 0; vv[k] = ix(sizeof(double) * (Q.bk[k].n + 1));
            lmin[k] = block_lmin(&Q.bk[k], vE + Q.poff[k], eps, vv[k], &cert);
            if (!cert) all = 0;
            if (verbose >= 0) printf("bm2: block %d (order %d, rank %d): lambda_min(S) = %.3e%s\n", k, Q.bk[k].n, Q.bk[k].r, lmin[k] * Q.cs, cert ? " (certified)" : "");
        }
        if (rc == 0 && all) {
            status = 0;
            /* certified: what is left of the gap is the barrier's, sum x_j z_j (about nl mu, mu = tol / 10 per pair);
             * when it exceeds tol, one more round at this rank with mu down to the gap. Not in every round: from such
             * a point the next rank's warm start stays jammed at the bounds */
            if (!Q.deepmu && Q.nl > 0 && !getenv("BRISK_BM2NOPOLISH")) {
                Lbest = fmax(Lbest, ip_lower_bound(&Q, lam, tol, nC1, -1));
                const double pr = st.f * Q.cs, gp = (pr - Lbest) / (1 + fabs(pr));
                if (gp > tol) {
                    Q.deepmu = 1; status = 1; polish = 1;
                    if (verbose >= 0) printf("bm2: certified, gap %.1e (the barrier's): once more with mu down to the gap\n", gp);
                }
            }
        }
        else if (rc <= 1 || rc == 3) {
            /* the lower bound of the multipliers: stop when the primal value is within -tol of it */
            const double Lr0 = ip_lower_bound(&Q, lam, tol, nC1, getenv("BRISK_BM2LBV") ? 0 : -1), pr = st.f * Q.cs;
            Lbest = fmax(Lbest, Lr0);
            const double Lr = Lbest, gr = (pr - Lr) / (1 + fabs(pr));
            if (verbose >= 0) printf("bm2: lower bound %.10e, primal value %.10e, gap %.1e\n", Lr, pr, gr);
            if (gr <= tol && !getenv("BRISK_BM2NOGAPSTOP")) { status = 4; if (verbose >= 0) printf("bm2: the gap to the lower bound is within the tolerance\n"); }
        }
        if (status == 4) ;
        else if (rc == 0 || rc == 1) {
            /* a column for every clearly negative eigenvalue (at most three a round; a tenth of the most negative one or
             * below), by subspace inverse iteration (one along lambda_min's vector if that finds none); the new columns
             * enter along the escape curve (ip_escape), else small */
            double **Vn = iz(sizeof(double *) * (Q.K + 1)); int *qn = iz(sizeof(int) * (Q.K + 1));
            for (int k = 0; k < Q.K; k++) if (lmin[k] < -eps && Q.bk[k].r < rmax && Q.bk[k].r < Q.bk[k].n) {
                IBlk *B = &Q.bk[k]; const int kk = getenv("BRISK_BM2ONECOL") ? 1 : 3, n = B->n;
                double vals[3]; double *vecs = ix(sizeof(double) * ((size_t)n * 3 + 1));
                const int nneg = kk > 1 ? block_negvecs(B, vE + Q.poff[k], lmin[k], eps, kk, vals, vecs) : 0;
                Vn[k] = ix(sizeof(double) * ((size_t)n * 3 + 1));
                for (int a = 0; a < nneg && B->r + qn[k] < rmax && B->r + qn[k] < n; a++) if (vals[a] <= 0.1 * vals[0]) { memcpy(Vn[k] + (size_t)qn[k] * n, vecs + (size_t)a * n, sizeof(double) * n); qn[k]++; }
                if (!qn[k]) { memcpy(Vn[k], vv[k], sizeof(double) * n); qn[k] = 1; }
                for (int c = 0; c < qn[k]; c++) {   /* zero on the fixed, gauge and zero rows; unit length */
                    double *v = Vn[k] + (size_t)c * n, nv = 0;
                    for (int j = 0; j < n; j++) { if (j == B->fixrow || j == B->grow || B->zr[j]) v[j] = 0; nv += v[j] * v[j]; }
                    nv = sqrt(nv); if (nv > 0) for (int j = 0; j < n; j++) v[j] /= nv;
                }
                if (verbose >= 0 && qn[k] > 1) printf("bm2: block %d: %d new directions (Ritz values %.2e .. %.2e)\n", k, qn[k], vals[0] * Q.cs, vals[qn[k] - 1] * Q.cs);
                if (verbose >= 0 && getenv("BRISK_BM2SUPP")) for (int c = 0; c < qn[k]; c++) {   /* diagnostic: rows carrying 1 - 1e-2, 1e-4, 1e-6 of |v|^2 */
                    double *a2 = ix(sizeof(double) * (n + 1)); const double *v = Vn[k] + (size_t)c * n;
                    for (int j = 0; j < n; j++) a2[j] = v[j] * v[j];
                    qsort(a2, n, sizeof(double), dcmp_desc);
                    int s2 = 0, s4 = 0, s6 = 0; double acc = 0;
                    for (int j = 0; j < n; j++) { acc += a2[j]; if (!s2 && acc >= 1 - 1e-2) s2 = j + 1; if (!s4 && acc >= 1 - 1e-4) s4 = j + 1; if (!s6 && acc >= 1 - 1e-6) { s6 = j + 1; break; } }
                    printf("bm2: block %d: direction %d: %d, %d, %d rows carry all but 1e-2, 1e-4, 1e-6 of it (of %d)\n", k, c, s2, s4, s6, n);
                    free(a2);
                }
                free(vecs); grow = 1;
            }
            if (grow) {
                const int esc = getenv("BRISK_BM2ESC") && ip_escape(&Q, xl, lam, Vn, qn, verbose >= 1 ? (getenv("BRISK_BM2V") ? 2 : 1) : 0);   /* (experiment: off) */
                const double ncs = getenv("BRISK_BM2NEWCOL") ? atof(getenv("BRISK_BM2NEWCOL")) : 1e-2;
                const double spe = getenv("BRISK_BM2SPARSECOL") ? atof(getenv("BRISK_BM2SPARSECOL")) : 0;   /* experiment */
                if (!esc) for (int k = 0; k < Q.K; k++) for (int c = 0; c < qn[k]; c++) {
                    const int n = Q.bk[k].n; const double *v = Vn[k] + (size_t)c * n;
                    char *keep = NULL;
                    if (spe > 0) {   /* the column on the rows that carry all but spe of |v|^2 */
                        double *a2 = ix(sizeof(double) * (n + 1)); for (int j = 0; j < n; j++) a2[j] = v[j] * v[j];
                        qsort(a2, n, sizeof(double), dcmp_desc);
                        double acc = 0, thr = 0; for (int j = 0; j < n; j++) { acc += a2[j]; if (acc >= 1 - spe) { thr = a2[j]; break; } }
                        keep = iz(n + 1); int nk = 0; for (int j = 0; j < n; j++) if (v[j] * v[j] >= thr && thr > 0) { keep[j] = 1; nk++; }
                        if (verbose >= 0) printf("bm2: block %d: new column on %d of %d rows\n", k, nk, n);
                        free(a2);
                    }
                    ip_addcol_on(&Q.bk[k], v, ncs, keep); free(keep);
                }
            }
            for (int k = 0; k < Q.K; k++) free(Vn[k]);
            free(Vn); free(qn);
        }
        for (int k = 0; k < Q.K; k++) free(vv[k]);
        free(vv);
        if (polish == 1) { polish = 2; ip_number(&Q); continue; }
        if (grow) Q.deepmu = 0;
        if (status == 0 || status == 4 || !grow || rc >= 2 || (tlim > 0 && wtime() - t0 > tlim)) { if (status == 4) status = 1; else if (status != 0) status = rc == 3 ? 3 : 1; break; }
        ip_number(&Q);
        mu = fmin(mu, 1e-6);
        for (int j = 0; j < Q.nl; j++) { xl[j] = fmax(xl[j], 1e-9); z[j] = fmax(z[j], 1e-9); }
    }
    /* measures on the problem as read (the reduced problem has the file's rows minus the dropped ones) */
    double *u = ix(sizeof(double) * (Q.E + 1)), *g = ix(sizeof(double) * (m + 1)); double f;
    ip_eval(&Q, xl, u, g, &f);
    double pinf2 = 0, dobj = 0;
    double *yr = ix(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) { const double gi = g[i] / Q.rs[i]; pinf2 += gi * gi; yr[i] = -Q.cs * Q.rs[i] * lam[i]; dobj += (Q.b[i] / Q.rs[i]) * yr[i]; }
    const double pobj = f * Q.cs;
    double lmn = 0; for (int k = 0; k < Q.K; k++) lmn = fmin(lmn, lmin[k] * Q.cs);
    for (int j = 0; j < Q.nl; j++) { double s = Q.cl[j]; for (int a = Q.lcp[j]; a < Q.lcp[j + 1]; a++) s += Q.lcv[a] * lam[Q.lci[a]]; lmn = fmin(lmn, s * Q.cs); }
    const double e1 = sqrt(pinf2) / (1 + nb1), e2 = -lmn / (1 + nC1), e3 = (pobj - dobj) / (1 + fabs(pobj) + fabs(dobj));
    const double Lb = fmax(ip_lower_bound(&Q, lam, tol, nC1, verbose), Lbest);
    if (verbose >= 0 && Lb > -INFINITY) printf("bm2: best lower bound of the run %.10e\n", Lb);
    const double acc = fmax(e1, fmax(e2, fabs(e3)));
    const double tt = wtime() - t0;
    const int stc = status == 0 && acc <= fmax(tol, 1e-12) * 10 ? ST_OPTIMAL : acc <= par->red_acc ? ST_REDUCED : status == 3 ? ST_TIME : ST_MAXIT;
    const char *sn = stc == ST_OPTIMAL ? "OPTIMAL" : stc == ST_REDUCED ? "SOLVED TO REDUCED ACCURACY" : stc == ST_TIME ? "TIME LIMIT" : "ITERATION LIMIT";
    double *yo = iz(sizeof(double) * (m0 + 1));
    if (SG) sign_unmap(SG, NULL, yr, NULL, yo); else memcpy(yo, yr, sizeof(double) * m);
    if (getenv("BRISK_BM2PROF")) printf("bm2: time: factorizations %.1fs, solves %.1fs, assembly %.1fs\n", g_t_fac, g_t_sol2, g_t_asm2);
    if (verbose >= 0) {
        printf("status: %s (max rel error %.1e)   ranks", sn, acc); for (int k = 0; k < Q.K; k++) printf(" %d", Q.bk[k].r); printf(", %d iterations\n", totits);
        printf("optimal value (SDPA/SDPLIB convention, max <F0,Y>): %.10e\n", -pobj);
        printf("primal objective %.12e  dual objective %.12e\n", pobj, dobj);
        printf("errors: primal %.1e  dual (lambda_min) %.1e  gap %.1e\n", e1, e2, e3);
        printf("  DIMACS errors: %.1e %.1e %.1e %.1e %.1e %.1e\n", e1, 0.0, 0.0, e2, e3, e3);
        printf("  Time:              %.2f s\n", tt);
    }
    if (res) {
        static const int ecode[7] = { 0, 11, 12, 13, 14, 10, 15 };
        res->status = stc; res->exit_code = ecode[stc];
        snprintf(res->status_str, sizeof(res->status_str), "%s", sn);
        res->iters = totits; res->pobj = pobj; res->dobj = dobj;
        res->err[1] = e1; res->err[2] = 0; res->err[3] = 0; res->err[4] = e2; res->err[5] = e3; res->err[6] = e3;
        res->time = tt; res->m = m0; res->nblk = nb0;
        res->bs = malloc(sizeof(int) * (nb0 + 1)); memcpy(res->bs, bs0, sizeof(int) * nb0);
        res->y = malloc(sizeof(double) * (m0 + 1)); memcpy(res->y, yo, sizeof(double) * m0);
        res->have_x = 0; res->X = NULL; res->Z = NULL;
        res->bound_side = isfinite(Lb) ? 2 : 0; res->bound_value = Lb; res->bound_lammin = lmn; res->bound_valid = isfinite(Lb) ? 1 : 0; res->bound_certified = 0;
        res->bound_rigorous = NAN;
        snprintf(res->cause, sizeof(res->cause), "%s", stc == ST_OPTIMAL ? "" : "Burer-Monteiro (-bm 2): the target accuracy was not reached");
    }
    if (yfile) { FILE *fy = fopen(yfile, "w"); if (fy) { for (int i = 0; i < m0; i++) fprintf(fy, "%.17g\n", yo[i]); fclose(fy); } }
    free(u); free(g); free(yr); free(yo); free(vE); free(lmin); free(xl); free(lam); free(z); free(bs0);
    ip_free(&Q); problem_free(P); sign_free(SG);
    return stc == ST_OPTIMAL ? 0 : stc == ST_REDUCED ? 10 : stc == ST_TIME ? 15 : 13;
}
