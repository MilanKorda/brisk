/* lpsolve.c - a primal-dual interior-point method for linear programs in bounded standard form
 *
 *     min c'x,   A x = b,   0 <= x_j <= u_j   (u_j = +inf: no upper bound; the first nfree
 *                                              columns are free: a proximal term)
 *
 * Mehrotra's predictor-corrector from an infeasible start, with
 *   - upper bounds kept as bounds (x + s = u, multipliers w): the normal equations are
 *     A Theta A' dy = r with Theta_j = 1 / (z_j / x_j + w_j / s_j): no rows for the bounds;
 *   - the normal equations factored by the supernodal sparse Cholesky code (sparsechol.c: AMD
 *     order, BLAS panels, dependent rows by tiny-pivot replacement), assembled row by row into
 *     the pattern of A A' of the sparse columns;
 *   - dense columns (long against the others) left out of the factored matrix and brought back
 *     by the Schur complement of the Sherman-Morrison-Woodbury formula;
 *   - every solve refined by preconditioned CG on the true matrix A Theta A' (one or two steps
 *     in the regular case, more when pivots were replaced or dense columns are ill-conditioned);
 *   - separate primal and dual step lengths, Gondzio's multiple centrality correctors (their
 *     number from the cost of a factorization against a solve, by flop counts);
 *   - Ruiz equilibration of A; termination on the residuals and the gap of the problem as given.
 * The method has no certificate of infeasibility: when the iteration diverges or stalls the
 * caller hands the problem to the SOCP solver (homogeneous self-dual model).                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include "brisk.h"
#include "lpsolve.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct SChol SChol;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
int    schol_amd_estimate(int m, const int *deg, int *const *nbr, double *flops_out, double *nnz_out);
int    schol_amd_estimate_cap(int m, const int *deg, int *const *nbr, const volatile double *cap, double *flops_out, double *nnz_out);
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_tinypiv(SChol *S, double t);
void   schol_set_tinydag(SChol *S, int on);
void   schol_set_lpsmall(SChol *S, int on);
int    schol_factor(SChol *S, double shift);
void   schol_set_stop(SChol *S, const volatile int *f, double deadline);
int    schol_stopped(const SChol *S);
void   schol_set_analysis_stop(const volatile int *f, double deadline);
size_t schol_lsolve_block_scratch(const SChol *S, int nrhs);
double schol_lsolve_flops(const SChol *S);
double schol_sn_solve_flops(const SChol *S, int k);
const int *schol_iperm(const SChol *S);
void   schol_lsolve_block(const SChol *S, double *Xt, int nrhs, double *acc);
void   schol_ltsolve_block(const SChol *S, double *Xt, int nrhs, double *acc);
void   schol_solve(const SChol *S, double *b, double *work);
double schol_lsolve_sparse_flops(const SChol *S, const int *bi, int nb, int *sn, char *mark);
size_t schol_nnz(const SChol *S);
int    schol_set_amd(int mode);
void   schol_zero(SChol *S);
double schol_flops(const SChol *S);
int    schol_ntiny(const SChol *S);
void   schol_set_solve_seq(int on);
void   schol_set_perm(const int *perm);
void   schol_set_nd(int on);
double schol_set_amalg(double f);
int schol_set_amalg_siblings(int on);

static double lp_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* (26 October 2026) the solve's deadline (lp_time seconds, 0: none) and stop flag, for the analysis before the iterations:
 * the estimates of the forms and of the dense columns' choices (AMD's counts) looked at neither, and the estimate of a
 * large LP's primal form took 23 s past a time limit of 10 s (opm2-z12-s8, the mixed-integer developer's first LP).
 * Set by lpipm_solve on its thread, taken over by the threads of the dense-column choice, handed to AMD (amd_2.c) */
static __thread double ne_dl = 0;
static __thread const volatile int *ne_stopf = NULL;
#ifdef HAVE_AMD
extern __thread double brisk_amd_deadline;
extern __thread const volatile int *brisk_amd_stop;
#endif
static int ne_expired(void) { return (ne_stopf && *ne_stopf) || (ne_dl > 0 && lp_time() > ne_dl); }
static void *xm(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (LP solver)\n"); exit(1); } return p; }
static void *xz(size_t n) { void *p = calloc(1, n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (LP solver)\n"); exit(1); } return p; }
static double dot(int n, const double *a, const double *b) { double s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static double nrminf(int n, const double *a) { double s = 0; for (int i = 0; i < n; i++) { const double v = fabs(a[i]); if (v > s) s = v; } return s; }

typedef struct {
    int m, n;
    int *Ap, *Ai; double *Ax;          /* scaled A by columns (rows ascending in a column) */
    int *Rp, *Rj, *Rq; double *Rx;     /* the sparse columns by rows; Rq: the position of the entry in its column */
    double *b, *c, *u; char *bd, *fr;  /* scaled data; bd[j]: x_j has an upper bound; fr[j]: x_j is free */
    double *rs, *cs;                   /* row and column scales: A_scaled = diag(rs) A diag(cs) */
    int nd, *dcol; char *isd;          /* dense columns */
    int bord, noborder, nbw, bwtried, *bwk; size_t *Bo; double *bw;   /* (nbw, bwk: the weak rows ordered after the border) */  /* (2 November 2026, BRISK_LPBORDER, an experiment) the dense columns as border rows of a
                                          quasidefinite matrix [M_s A_d; A_d' -Theta_d^-1] ordered last: their count, the offsets of their
                                          entries (per dense column: its rows, then its diagonal), the solves' work (2 (m + nd) + 2) */
    char *djoin; int njoin;            /* (1 November 2026) long columns kept in the factored part after a loss of the correction's accuracy */
    int nfree, qd, qdloop, qdns;       /* (4 November 2026, BRISK_LPFREEQD) the free columns (the first nfree) as unknowns of the factored
                                        * quasidefinite matrix; qdloop: the iterations under way (the start uses the free columns' weight 1);
                                        * qdns: the border vertices that are free columns placed among the rows */
    double qrho, qtgt;                 /* (the free columns' regularization in the factor, and in the system the refinement solves) */
    double *qv;                        /* (work: m + n + nd values) */
    SChol *sc; int *Mp, *Mi; size_t *Mo;   /* the lower pattern of A_s A_s' by columns: rows Mi >= the column, offsets Mo */
    double *V, *G, *gw;                /* dense columns: V = M_s^-1 A_d (m x nd), G = Theta_d^-1 + A_d' V = L L' */
    /* (22 October 2026) the rows the dense columns alone cover: their diagonal (from the dense columns) put into the
     * factored matrix and taken out again as the last nE columns of the correction, which is then indefinite: V has
     * nd + nE columns, G = C^-1 + U' V with U = [A_d, e_E], C = diag(Theta_d, -Edel), factored by Bunch-Kaufman */
    int nE, *Erow, *Gpiv; double *Edel, *Gwork, *Glam; int lGwork; int dbg_prec;
    int ntcap, Ecap;                 /* (1 November 2026) the capacities of G, gw, Wp, Gpiv, Gwork, Glam (nd + nE) and of Erow, Edel */
    int andon, nweak;                /* (1 November 2026) the weak rows taken out (after a loss of the correction); their number at the last factorization */
    /* (22 October 2026) W = L^-1 P U, sparse (the forward solves of U's columns reach few supernodes), by columns:
     * G = C^-1 + W'W, and a solve P' L^-T (L^-1 P r - W G^-1 W' L^-1 P r): one solve and two passes over W. xw (m,
     * zeros), wmark (supernodes, zeros), wsn: the sparse solves' work */
    int *Wp, *Wi, Wcap; double *Wx, *xw; char *wmark; int *wsn;
    size_t *Wrp; int *Wrc; double *Wrv; int Wrok;   /* (1 November 2026) W by rows (permuted rows; columns increasing), for the solves' correction on the threads */
    int nthw; double **xw_t; char **wmark_t; int **wsn_t;   /* (23 October 2026) per-thread work of the dense columns' solves */
    double *Vw;                        /* (22 October 2026) the work of the solve of V's columns at once (m x nd) */
    double *theta;
    double *w1, *w2, *w3, *w4, *wn, *acc; long nref; double worst;   /* work: m, m, m, n, m */
    double *pr, *pz, *pp, *pq;         /* CG work (m) */
    double reg;
    double pdreg;                      /* (22 October 2026, BRISK_LPPDREG, an experiment, 0 off) the primal-dual regularization
                                          delta: Theta = 1 / (z/x + w/s + delta), A Theta A' + delta I, the proximal centres at
                                          the iterate (the right-hand sides unchanged) */
    double pdd;                        /* (the same night) the dual part when on (BRISK_LPPDDUAL; by default the same as pdreg) */
    double pdrel, *dreg;               /* (the same night, BRISK_LPPDREL=eps, an experiment) with the regularization on, the dual
                                          part also eps M_ii per row, scaled to the matrix: dreg[i] at each factorization */
    double tinyth;                   /* the tiny-pivot threshold of the factorizations */
    int tinyup_tried;                /* (4 November 2026) the threshold raised once after failed solves (tried; kept or undone) */
    long nfact, nsolve, ncg, nprec; int ntiny;
    double t_fact, t_solve, t_asm, t_dense;
    double w_sp_flops;              /* the sparse forward solves of W's columns: their counted multiplications (0: not yet) */
    double *Z; size_t Zcap; int zok;  /* Z = M_s^-1 U with the blocks (chunks of 32 columns, each m rows of 32 in the permuted ordering): the correction around the full solve */
    int znot;                         /* (31 October 2026) the correction around the full solve lost its accuracy: the two sweeps with W from here */
    int fstop;                      /* the last factorization was stopped (the stop flag or the deadline), not failed */
    double **acc_t; int nacc_t;          /* 17 October 2026: per-thread accumulators of the assembly */
    int *Fp, *Fj; double *Fx;          /* 18 October 2026: all of A by rows (columns ascending), for A x on the threads */
    int dasm, nsc, *scol; double *Md, *Wd; int wdc;   /* 18 October 2026 (LP/SOCP): A_s Theta A_s' by dsyrk (dense; scol: the sparse columns, Wd: m x wdc) */
    int est;                           /* ne_setup for a cost estimate only (AMD, no offsets) */
    int est_nd;                        /* (2 November 2026) an estimate with the ordering the factorization uses (the nested dissection), not AMD's count */
    double dthr;                       /* > 0: the length above which a column is dense (the cost-based choice, ne_dense_choice) */
    double est_budget, est_lb; long est_w;   /* the estimate's budget; its lower bound on the flops when it is exceeded */
    int est_pat;                            /* (the lower bound from the sparse columns' pattern alone: it holds for any choice that keeps more of them) */
    const volatile double *est_cap;         /* (may be NULL) a cap on the estimate's AMD count, lowered while it runs */
    double *vpart;                     /* (the partial sums of the O(n) passes, one per chunk of LPV_CH columns) */
    double *wa;                        /* the affine direction while the weighted corrector is chosen (4 n + m) */
} LpS;

/* 18 October 2026: with four threads and A of 1e5 entries or more the products run on the threads: A'y by columns
 * (each x_j its own sum), A x as a gather by rows from a row copy whose entries are in ascending column order - the
 * order in which the scatter by columns adds into y_i, so the same sums (BRISK_LPAPAR=0: the sequential products) */
static int lp_apar(const LpS *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_LPAPAR"); on = e ? atoi(e) : 1; }
    return on && S->Fp && S->Ap[S->n] >= 100000 && omp_get_max_threads() > 1 && !omp_in_parallel();
}
static void A_mul(const LpS *S, const double *x, double *y) {           /* y = A x */
    if (lp_apar(S)) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < S->m; i++) { double v = 0; for (int q = S->Fp[i]; q < S->Fp[i + 1]; q++) { const double xj = x[S->Fj[q]]; if (xj != 0.0) v += S->Fx[q] * xj; } y[i] = v; }
        return;
    }
    for (int i = 0; i < S->m; i++) y[i] = 0;
    for (int j = 0; j < S->n; j++) { const double xj = x[j]; if (xj != 0.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) y[S->Ai[p]] += S->Ax[p] * xj; }
}
static void At_mul(const LpS *S, const double *y, double *x) {          /* x = A' y */
    #pragma omp parallel for schedule(static) if (lp_apar(S))
    for (int j = 0; j < S->n; j++) { double s = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * y[S->Ai[p]]; x[j] = s; }
}
/* 21 October 2026 (LP/SOCP): the O(n) passes of the iteration on the threads for LPs of at least LPV_MIN columns (the
 * residuals, mu, Theta, the right-hand sides, the step lengths, the updates; transport_1000, a million columns: 8.2 of the
 * 18.6 s of its iterations were these passes, sequential). The elementwise passes and the minima and maxima give the same
 * bits at any thread count; the sums run over fixed chunks of LPV_CH columns whose partial sums are added in order, so
 * they too are the same at any thread count (and below LPV_MIN columns everything is as before). BRISK_LPVPAR=0: one
 * thread (the chunked sums kept). */
#define LP_NDMAX 2048       /* the most columns kept apart from the factor when the cost model chooses them */
/* (27 October 2026) from 20,000 columns, before 100,000: the interior-point method alone 4-10 % faster on LPs of 20,000 to
 * 100,000 columns (pds-20, LP: 2.87 -> 2.57 s; pds-10, ken-18, cre-b, pilot87 alike), the race of the methods unchanged
 * (Kennington 0.98 and 0.99 in two runs; 1.004 on the 237 MIPLIB relaxations, 0.999 on the 26 it touches).
 * BRISK_LPVMIN=k: from k columns */
static int lpv_min(void) { static int v = -1; if (v < 0) { const char *e = getenv("BRISK_LPVMIN"); v = e ? atoi(e) : 20000; } return v; }
#define LPV_MIN lpv_min()
#define LPV_CH 16384
static int lpv_par(const LpS *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_LPVPAR"); on = e ? atoi(e) : 1; }
    return on && S->n >= LPV_MIN && omp_get_max_threads() > 1 && !omp_in_parallel();
}
#define LPV_FOR _Pragma("omp parallel for schedule(static) if (lpv_par(S))")
/* res += the sum over the columns j of what the body adds to v */
#define LPV_SUM(res, ...) do { const int n_ = S->n; \
    if (n_ >= LPV_MIN) { const int nch_ = (n_ + LPV_CH - 1) / LPV_CH; double *pt_ = S->vpart; \
        _Pragma("omp parallel for schedule(static) if (lpv_par(S))") \
        for (int c_ = 0; c_ < nch_; c_++) { double v = 0; const int j1_ = (c_ + 1) * LPV_CH < n_ ? (c_ + 1) * LPV_CH : n_; \
            for (int j = c_ * LPV_CH; j < j1_; j++) { __VA_ARGS__; } pt_[c_] = v; } \
        for (int c_ = 0; c_ < nch_; c_++) (res) += pt_[c_]; \
    } else { double v = 0; for (int j = 0; j < n_; j++) { __VA_ARGS__; } (res) += v; } } while (0)
/* q = A Theta A' v (the true matrix: the regularization is in the factor, the preconditioner, only) */
static void M_mul(LpS *S, const double *v, double *q) {
    At_mul(S, v, S->wn);
    LPV_FOR for (int j = 0; j < S->n; j++) S->wn[j] *= S->theta[j];
    A_mul(S, S->wn, q);
    if (S->pdreg > 0) for (int i = 0; i < S->m; i++) q[i] += (S->pdd + (S->dreg ? S->dreg[i] : 0.0)) * v[i];
}
static void ne_prec(LpS *S, const double *r, double *z);
/* (31 October 2026) the width of the blocks of right-hand sides of the dense columns' solves (W, Z): fixed, so that each
 * column's arithmetic is the same at any thread count */
#ifndef LP_ZCB
#define LP_ZCB 64     /* (energy Q017, LP: 64 columns 0.37 s of blocked solves a factorization, 32 columns 0.53 s) */
#endif
/* the columns a0..a1-1 of a block held transposed (Xt: m rows of cb, row i the permuted index i) as sparse columns, rows
 * increasing: two passes along the rows (the block read in its own order, not column by column at a stride of cb) */
static void lp_chunk_extract(const double *Xt, int m, int cb, int a0, int a1, int **ci, double **cv, int *cn) {
    const int w = a1 - a0;
    int cnt[LP_ZCB]; for (int c = 0; c < w; c++) cnt[c] = 0;
    for (int i = 0; i < m; i++) { const double *r = Xt + (size_t)i * cb; for (int c = 0; c < w; c++) cnt[c] += r[c] != 0.0; }
    int pos[LP_ZCB];
    for (int c = 0; c < w; c++) { ci[a0 + c] = xm(sizeof(int) * ((size_t)cnt[c] + 1)); cv[a0 + c] = xm(sizeof(double) * ((size_t)cnt[c] + 1)); cn[a0 + c] = cnt[c]; pos[c] = 0; }
    for (int i = 0; i < m; i++) {
        const double *r = Xt + (size_t)i * cb;
        for (int c = 0; c < w; c++) { const double v = r[c]; if (v != 0.0) { const int q = pos[c]++; ci[a0 + c][q] = i; cv[a0 + c][q] = v; } }
    }
}
/* the factorization of A_s Theta A_s' + reg (and the Schur complement of the dense columns); 0: done */
static int ne_factor(LpS *S) {
    const double t0 = lp_time();
    const int m = S->m;
    /* (31 October 2026) past the deadline or stopped: no factorization (zeroing a large factor is itself seconds of page
     * faults: 1.7 GB on degenerate_cover scale 4, LP) */
    S->fstop = 0;
    if (ne_expired()) { S->fstop = 1; return 1; }
    schol_zero(S->sc);
    double *pm = schol_values(S->sc);
    /* 17 October 2026: the columns of M on the threads (each its own accumulator; a column's entries are summed in
     * the same order, so the same results): the assembly was two thirds of the factorizations' time on osa-30.
     * BRISK_LPASMPAR=0 off. */
    static int asmpar = -1; if (asmpar < 0) { const char *e = getenv("BRISK_LPASMPAR"); asmpar = !(e && atoi(e) == 0); }
    int nt = 1;
#ifdef _OPENMP
    if (asmpar && S->Mp[m] >= 100000 && !omp_in_parallel()) nt = omp_get_max_threads();
#endif
    if (nt > 1 && !S->acc_t) { S->acc_t = xm(sizeof(double *) * (size_t)nt); S->nacc_t = nt; for (int t = 0; t < nt; t++) S->acc_t[t] = xm(sizeof(double) * (size_t)(m + 1)); }
    if (nt > S->nacc_t) nt = S->nacc_t > 0 ? S->nacc_t : 1;
    if (S->dasm) {
        /* 18 October 2026 (LP/SOCP): A_s Theta A_s' as dsyrk products of blocks of columns of A_s Theta^1/2, made
         * dense (m x wdc), then the pattern's entries copied into the factor (see ne_setup for the choice) */
        const double one = 1.0; double beta = 0.0;
        for (int c0 = 0; c0 < S->nsc; c0 += S->wdc) {
            const int kk = S->nsc - c0 < S->wdc ? S->nsc - c0 : S->wdc;
            memset(S->Wd, 0, sizeof(double) * (size_t)m * (size_t)kk);
            for (int t = 0; t < kk; t++) {
                const int j = S->scol[c0 + t]; const double f = sqrt(S->theta[j]); double *wc = S->Wd + (size_t)t * m;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) wc[S->Ai[p]] = f * S->Ax[p];
            }
            /* (28 October 2026) the product by strips of 128 output columns on the threads: one BLAS call ran on one
             * core (BLAS is held to one thread during a solve), most of the factorizations' time on dense normal
             * equations (a_near_parallel_05, LP: 0.99 of 1.08 s). Each strip: dsyrk on its diagonal block, dgemm on the
             * rows below. The strips depend on m only, so the arithmetic is the same at any thread count */
            const int TB = 128, nb = (m + TB - 1) / TB;
            const double bt = beta;
            int ntd = 1;     /* (the threads by the product's own work, about m^2 kk / 2 multiplications) */
#ifdef _OPENMP
            if (asmpar && 0.5 * (double)m * m * kk >= 2e7 && !omp_in_parallel()) ntd = omp_get_max_threads();
#endif
            #pragma omp parallel for schedule(dynamic, 1) num_threads(ntd) if (ntd > 1)
            for (int b = 0; b < nb; b++) {
                const int r0 = b * TB, rb = m - r0 < TB ? m - r0 : TB, rest = m - r0 - rb;
                double *Wr = S->Wd + r0, *Mb = S->Md + (size_t)r0 * m + r0;
                BL(dsyrk_)("L", "N", &rb, &kk, &one, Wr, &m, &bt, Mb, &m);
                if (rest > 0) BL(dgemm_)("N", "T", &rest, &rb, &kk, &one, Wr + rb, &m, Wr, &m, &bt, Mb + rb, &m);
            }
            beta = 1.0;
        }
        if (beta == 0.0) for (int i = 0; i < m; i++) memset(S->Md + (size_t)i * m + i, 0, sizeof(double) * (size_t)(m - i));
        for (int i = 0; i < m; i++) {
            const double *col = S->Md + (size_t)i * m;
            for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) pm[S->Mo[e]] = col[S->Mi[e]];
            if (S->dreg && S->pdreg > 0) { S->dreg[i] = S->pdrel * pm[S->Mo[S->Mp[i]]]; pm[S->Mo[S->Mp[i]]] += S->dreg[i]; }
            pm[S->Mo[S->Mp[i]]] += S->reg;
        }
    } else {
    #pragma omp parallel num_threads(nt) if (nt > 1)
    {
#ifdef _OPENMP
        double *acc = nt > 1 ? S->acc_t[omp_get_thread_num()] : S->acc;
#else
        double *acc = S->acc;
#endif
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < m; i++) {
            for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) acc[S->Mi[e]] = 0.0;
            for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) {
                const int j = S->Rj[q]; const double t = S->Rx[q] * S->theta[j];
                for (int p = S->Rq[q]; p < S->Ap[j + 1]; p++) acc[S->Ai[p]] += t * S->Ax[p];
            }
            for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) pm[S->Mo[e]] = acc[S->Mi[e]];
            if (S->dreg && S->pdreg > 0) { S->dreg[i] = S->pdrel * pm[S->Mo[S->Mp[i]]]; pm[S->Mo[S->Mp[i]]] += S->dreg[i]; }
            pm[S->Mo[S->Mp[i]]] += S->reg;                    /* (the diagonal is the first entry of the column) */
        }
    }
    }
    /* (20 October 2026, LP/SOCP: 1e-16 instead of 1e-20. qap15's normal equations (folded or not) have 240-344 dependent
     * rows of 2,121; the pivots between 1e-20 and 1e-16 were kept as information and the direction was garbage: the gap
     * reached 5e-6, then the primal residual grew and the method gave up after 40 iterations. At 1e-16 (the rounding
     * level of a pivot of order one) 17 iterations, 0.9 s; the other 141 LPs of the sets the same status lines and
     * iterations within one. ISSUES 226.) */
    /* the rows the dense columns alone cover: their diagonal in A Theta A' (sum over the dense columns of theta a^2), or 1
     * when it is zero, put into the factored matrix (taken out again by the correction) */
    /* (1 November 2026, ISSUES 813) the rows where the sparse part is weak against the dense columns join the rows they
     * alone cover: when the sparse diagonal of a row covered by the dense columns falls below eps times theirs, the sparse
     * factor is nearly singular there and the correction cancels large terms (k1mushroom, LP: 80 dense columns over all
     * 12,982 rows, 392 rows below 1e-10 at the iteration where the correction was lost). Their diagonal from the dense
     * columns goes into the factored matrix and comes out again as further unit columns of the correction (Andersen's
     * modified Schur complement). BRISK_LPANDEPS: eps (0: off) */
    static double andeps = -1; if (andeps < 0) { const char *e_ = getenv("BRISK_LPANDEPS"); andeps = e_ ? atof(e_) : 1e-10; }
    S->nweak = 0;
    if (S->bord) {      /* (BRISK_LPBORDER) the border: the dense columns' entries and -1/theta on their diagonal */
        size_t q = 0;
        for (int k = 0; k < S->bord; k++) { const int j = S->dcol[k];
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) pm[S->Bo[q++]] = S->Ax[p];
            pm[S->Bo[q++]] = S->qd && j < S->nfree && S->qdloop ? -S->qrho : S->qd && !S->isd[j] ? -1e30 : -1.0 / S->theta[j]; }
    }
    if (S->nd > 0 && !S->bord && !S->est && andeps > 0 && S->andon && S->Rp) {
        double *dd = xz(sizeof(double) * (size_t)m);
        for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) dd[S->Ai[p]] += S->theta[j] * S->Ax[p] * S->Ax[p]; }
        int ne = 0, ne0 = 0;
        for (int i = 0; i < m; i++) { if (S->Rp[i + 1] == S->Rp[i]) ne0++; else if (dd[i] > 0 && pm[S->Mo[S->Mp[i]]] <= andeps * dd[i]) ne++; }
        /* (the cost of the correction with them against the budget of the rule that joins the dense columns, ten times the
         * factorization with them apart: a sparse solve with each further column, G's entries and its eigen-decomposition with
         * its vectors, about 9 nt^3; above it only the rows of no sparse entry, as before, and the rules after a loss decide) */
        {   const double nt_ = (double)S->nd + ne0 + ne, nnzl = (double)schol_nnz(S->sc);
            const double fs = schol_flops(S->sc) + S->nd * (4.0 * nnzl + 2.0 * S->nd * m);
            const double fe = ne * 4.0 * nnzl + 2.0 * nt_ * nt_ * fmin((double)m, nnzl / fmax(1.0, (double)m) * 8.0) + 9.0 * nt_ * nt_ * nt_;
            if (ne > 0 && fe > 10.0 * fs) { if (getenv("BRISK_LPDBG")) printf("     [dense columns: %d weak rows left in, their correction %.1e flops against %.1e]\n", ne, fe, 10.0 * fs); ne = 0; }
        }
        const int weak = ne > 0; S->nweak = ne; ne += ne0;
        if (ne > S->Ecap) { free(S->Erow); free(S->Edel); S->Ecap = ne + ne / 2 + 16 < m ? ne + ne / 2 + 16 : m; S->Erow = xm(sizeof(int) * (size_t)S->Ecap); S->Edel = xm(sizeof(double) * (size_t)S->Ecap); }
        int k = 0; for (int i = 0; i < m; i++) if (S->Rp[i + 1] == S->Rp[i] || (weak && dd[i] > 0 && pm[S->Mo[S->Mp[i]]] <= andeps * dd[i])) S->Erow[k++] = i;
        free(dd);
        if (getenv("BRISK_LPDBG") && ne != S->nE) printf("     [dense columns: %d rows taken out by unit columns (were %d)]\n", ne, S->nE);
        S->nE = ne;
        const int nt = S->nd + ne;
        if (ne > 0 && nt > S->ntcap) {
            const int c = nt + nt / 4 + 8;
            free(S->G); free(S->gw); free(S->Gpiv); free(S->Gwork); free(S->Glam);
            S->G = xm(sizeof(double) * (size_t)c * c); S->gw = xm(sizeof(double) * ((size_t)c + 1));
            S->Wp = realloc(S->Wp, sizeof(int) * ((size_t)c + 1));
            S->Gpiv = xm(sizeof(int) * ((size_t)c + 1)); S->lGwork = 64 * c + 64; S->Gwork = xm(sizeof(double) * (size_t)S->lGwork); S->Glam = xm(sizeof(double) * (2 * (size_t)c + 1));
            S->ntcap = c;
        }
    }
    if (S->nE > 0) {
        for (int k = 0; k < S->nE; k++) S->Edel[k] = 0.0;
        for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d];
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p];
                /* (Erow ascending: a binary search) */
                int lo = 0, hi = S->nE - 1; while (lo < hi) { const int mid = (lo + hi) / 2; if (S->Erow[mid] < i) lo = mid + 1; else hi = mid; }
                if (S->Erow[lo] == i) S->Edel[lo] += S->theta[j] * S->Ax[p] * S->Ax[p]; } }
        for (int k = 0; k < S->nE; k++) { if (!(S->Edel[k] > 0)) S->Edel[k] = 1.0; pm[S->Mo[S->Mp[S->Erow[k]]]] += S->Edel[k]; }
    }
    if (getenv("BRISK_ANDDBG") && S->nd > 0) {      /* (debug) rows where the sparse part's diagonal is weak against the dense columns' */
        double *dd = xz(sizeof(double) * (size_t)m);
        for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) dd[S->Ai[p]] += S->theta[j] * S->Ax[p] * S->Ax[p]; }
        int c6 = 0, c10 = 0, c14 = 0, cov = 0; double mn = 1e300;
        for (int i = 0; i < m; i++) { if (dd[i] <= 0) continue; cov++; const double r = pm[S->Mo[S->Mp[i]]] / dd[i]; if (r < mn) mn = r; if (r < 1e-6) c6++; if (r < 1e-10) c10++; if (r < 1e-14) c14++; }
        printf("     [sparse/dense diagonal: %d rows covered, min ratio %.1e, below 1e-6 %d, 1e-10 %d, 1e-14 %d; nE %d]\n", cov, mn, c6, c10, c14, S->nE);
        free(dd);
    }
    S->t_asm += lp_time() - t0;
    schol_set_tinypiv(S->sc, S->tinyth > 0 ? S->tinyth : 1e-16);
    /* (BRISK_LPFREEQD) a free column's pivot that has lost its sign (the column dependent on the others at this point) is
     * replaced by -1e128: its column of the factor ~ 0, its step 0, as a dependent row's in the normal equations */
    if (S->qd && S->bord) { extern void schol_set_dynpiv(SChol *, double); schol_set_dynpiv(S->sc, 1e128); }
    /* 18 October 2026 (LP/SOCP): the large nodes by the tile task graph of sparsechol.c (fnode_dag), the tiny-pivot rule
     * in its diagonal tiles; it was skipped with tiny pivots (BRISK_LPTDAG=0: as before) */
    { static int tdag = -1; if (tdag < 0) { const char *e = getenv("BRISK_LPTDAG"); tdag = !(e && atoi(e) == 0); } schol_set_tinydag(S->sc, tdag); }
    /* (26 October 2026, the merge for 1.6) the LP method's small sources straight into the target (sparsechol.c,
     * BRISK_LPSMALL): a mode of its own, since the tiny-pivot rule is also used by the SDP solver and the QP method,
     * whose factorizations must keep their rounding */
    schol_set_lpsmall(S->sc, 1);
    /* 17 October 2026: the factorization by the tree code of sparsechol.c on the threads, the solves
     * sequential (23 October 2026: on the threads too for large factors, below). The left-looking code (schol_set_solve_seq) was 25-30 % faster at one thread in 5.7; with
     * the kernels of since the two are even at one thread (Kennington 13.3 against 13.0 s) and the tree code
     * uses the threads (four: 11.8 s; pds-20 5.1 -> 3.6 s), with the same results at any thread count; its
     * solves stay slower (osa-60: 1.2 -> 3.3 s if they go with it). BRISK_LPFLL=1: the left-looking code. */
    static int fll = -1; if (fll < 0) fll = getenv("BRISK_LPFLL") != NULL;
    if (!fll) schol_set_solve_seq(0);
    /* (31 October 2026) the solve's stop flag and deadline handed to the factorization, which checks them at every
     * supernode on every thread: the first factorization of a large LP's normal equations took 29-40 s past a time limit
     * of 2.6 s (degenerate_cover scale 4, LP), and the simplex method beside it could not stop it once it had won.
     * omp_get_wtime's clock: the deadline moved over from lp_time's by the difference of the two clocks now */
    schol_set_stop(S->sc, ne_stopf, ne_dl > 0 ? omp_get_wtime() + (ne_dl - lp_time()) : 0.0);
    /* (BRISK_LPFREEQD) a row pivot of the factored matrix that vanishes (the sparse part alone singular there, the free columns'
     * entries not yet in it: they come after the row) is replaced by a small multiple of its diagonal and its column kept, not
     * dropped as a dependent row of the normal equations: the free columns may determine that row. BRISK_LPFREEQDTINY=eps (0: drop) */
    extern void schol_set_tinyreg(double eps);
    static double qtiny = -1; if (qtiny < 0) { const char *e_ = getenv("BRISK_LPFREEQDTINY"); qtiny = e_ ? atof(e_) : 0.0; }
    if (S->qd && S->bord && qtiny > 0) schol_set_tinyreg(qtiny);
    const int bad = schol_factor(S->sc, 0.0);
    if (S->qd && S->bord) schol_set_tinyreg(0.0);
    S->fstop = schol_stopped(S->sc);
    {   /* (23 October 2026) the solves on the threads too (the tree code's) for a factor of at least 500,000 entries: each
         * solve is a pass through the factor, 3.1 of opm2-z10-s4's 10 s with 17.8 million entries. Four threads, total
         * time (LPs): osa-60 1.37 -> 1.21 s, ken-18 2.03 -> 1.76, s100 22.1 -> 19.7, physiciansched3-3 71.1 -> 60.8,
         * opm2-z10-s4 19.3 -> 18.3; rail01 23.1 -> 24.4, supportcase10 6.7 -> 7.3, pds-20 2.85 -> 2.91 (the regression
         * that made them sequential, osa-60 1.2 -> 3.3 s, is gone with the tree code's present solves). BRISK_LPSOLVEPAR:
         * the threshold (0: sequential always) */
        static double spar = -1; if (spar < 0) { const char *e = getenv("BRISK_LPSOLVEPAR"); spar = e ? atof(e) : 5e5; }
        if (!fll) schol_set_solve_seq(!(spar > 0 && (double)schol_nnz(S->sc) >= spar));
    }
    S->ntiny += schol_ntiny(S->sc);
    S->nfact++;
    if (bad) { S->t_fact += lp_time() - t0; return 1; }
    if (S->nd > 0 && getenv("BRISK_LPTHDBG")) {
        /* (debug) the dense columns' weights against the sparse part's diagonal scale: how many matter */
        double dmax = 0; int c0 = 0, c2 = 0, c4 = 0, c6 = 0;
        for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; double a2 = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) a2 += S->Ax[p] * S->Ax[p]; const double v = S->theta[j] * a2; if (v > dmax) dmax = v; }
        for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; double a2 = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) a2 += S->Ax[p] * S->Ax[p]; const double v = S->theta[j] * a2 / dmax; if (v > 1e-2) c0++; if (v > 1e-4) c2++; if (v > 1e-6) c4++; if (v > 1e-8) c6++; }
        fprintf(stderr, "   [dense weights, factorization %ld: %d columns; relative to the largest, above 1e-2: %d, 1e-4: %d, 1e-6: %d, 1e-8: %d]\n", S->nfact, S->nd, c0, c2, c4, c6);
    }
    if (S->nd > 0 && !S->bord) {
        /* the correction (the Sherman-Morrison-Woodbury formula) over U = [A_d, e_E], C = diag(Theta_d, -Edel):
         * G = C^-1 + U' M~^-1 U, its column a from v = M~^-1 u_a (kept in V, or one work vector), its lower part */
        const int nd = S->nd, nE = S->nE, nt = nd + nE;
        const double tg0 = lp_time();
        /* W's columns: the sparse forward solves of U's columns. (23 October 2026) On the threads when there are several
         * columns: each column's solve into a buffer of its own (the same arithmetic on any thread: the same W at any
         * thread count), then the buffers in column order; on ex9 (LP; 94 dense columns, W 497,151 entries) the solves
         * took 0.5-0.75 s of each factorization one after another */
        int nw = 0; S->Wp[0] = 0;
        S->Wrok = 0;
        int one_i = 0; double one_v = 1.0;
        int nthw = 1;
#ifdef _OPENMP
        if (nt >= 8 && !omp_in_parallel()) nthw = omp_get_max_threads();
        if (nthw > nt) nthw = nt;
#endif
        /* (31 October 2026) W by blocks of right-hand sides (schol_lsolve_block: a dtrsm and a dgemm per supernode, 32
         * columns of W at a time, the chunks on the threads; the width fixed, so the same bits at any thread count) when it
         * costs less than the sparse solves: their multiplications are counted at the first factorization (W's pattern is
         * the same at every one), the blocks' are nt times those of a forward solve, at a rate WRATE times higher (dtrsm and
         * dgemm on 32 columns against the scalar sweep; measured on energy Q017, LP: below). BRISK_LPWBLOCK=0 never, 1
         * always */
        static int wblock = -2; if (wblock == -2) { const char *e = getenv("BRISK_LPWBLOCK"); wblock = e ? atoi(e) : -1; }
        const double WRATE = 2.5;     /* (measured: 2.8 on energy Q017, LP: blocks 3.05e9 multiplications in 0.287 s, sparse solves 2.43e9 in 0.649 s, 3 threads each) */
        /* (1 November 2026) at the first factorization the sparse solves' multiplications are counted from the structure
         * (the supernodes each column's solve reaches), so the blocks are taken from the start when they cost less: before,
         * the first factorization made the sparse solves to count them (Q006_energy_a0, LP: 10.4 s against 1.1 s by blocks) */
        if (S->w_sp_flops <= 0 && nt >= 8 && wblock != 0) {
            double f = 0; int oi = 0;
            for (int a = 0; a < nt; a++) {
                if (a < nd) { const int j = S->dcol[a]; f += schol_lsolve_sparse_flops(S->sc, S->Ai + S->Ap[j], S->Ap[j + 1] - S->Ap[j], S->wsn, S->wmark); }
                else { oi = S->Erow[a - nd]; f += schol_lsolve_sparse_flops(S->sc, &oi, 1, S->wsn, S->wmark); }
            }
            S->w_sp_flops = f;
        }
        const int useblk = nt >= 8 && (wblock == 1 || (wblock != 0 && S->w_sp_flops > 0 && (double)nt * schol_lsolve_flops(S->sc) < WRATE * S->w_sp_flops));
        S->zok = 0;
        if (useblk) {
            const int CB = LP_ZCB, nch = (nt + CB - 1) / CB;
            const int *ip = schol_iperm(S->sc);
            /* (31 October 2026) and Z = M_s^-1 U = L^-T W by the blocked backward solve on the same chunks, when it fits in
             * 1 GB (BRISK_LPZMEM, MB; 0: never): each solve is then the factor's full solve on the threads, U' y (U sparse)
             * and y - Z g, instead of the two sweeps one after the other with W between them. energy Q017 (LP): the full
             * solve 7.2 ms, the sweeps 40.5 ms, the corrected solve 55.2 ms on its factor of 12 million entries */
            static double zmem = -1; if (zmem < 0) { const char *e = getenv("BRISK_LPZMEM"); zmem = e ? atof(e) : 0.0; }     /* (1 November 2026: off by default; the correction between the halves of the solve, ne_mid, costs less) */
            const size_t zsz = (size_t)nch * CB * (size_t)m;
            /* (only when the factor's solves run on the threads, from BRISK_LPSOLVEPAR's 500,000 entries: on a smaller factor
             * the full solve is the same sweeps one after the other, and y - Z g, dense, only adds to them: umts, LP, 0.08 ->
             * 0.12 s) */
            static double zpar = -1; if (zpar < 0) { const char *e = getenv("BRISK_LPSOLVEPAR"); zpar = e ? atof(e) : 5e5; }
            const int wantz = !S->znot && zmem > 0 && 8.0 * (double)zsz <= zmem * 1048576.0 && zpar > 0 && (double)schol_nnz(S->sc) >= zpar;
            if (wantz && S->Zcap < zsz) { free(S->Z); S->Z = xm(sizeof(double) * zsz); S->Zcap = zsz; }
            if (wantz) {
                /* the chunks in Z's own memory: the forward and backward solves in place; W itself is not needed (G = C^-1 +
                 * U'Z below, U sparse) */
                const size_t scr2 = schol_lsolve_block_scratch(S->sc, CB);
                int nthz = 1;
#ifdef _OPENMP
                if (!omp_in_parallel()) nthz = omp_get_max_threads();
                if (nthz > nch) nthz = nch;
#endif
                double **Az = xm(sizeof(double *) * nthz);
                for (int t = 0; t < nthz; t++) Az[t] = xm(sizeof(double) * scr2);
                /* (W's columns kept too, from the chunk between the two solves: G = C^-1 + W'W, the Gram form, stays positive
                 * semidefinite in floating point; formed as U'Z it lost that on neos-829552, LP, and its factorization failed) */
                int **zci = xm(sizeof(int *) * nt); double **zcv = xm(sizeof(double *) * nt); int *zcn = xz(sizeof(int) * (nt + 1));
                #pragma omp parallel for num_threads(nthz) schedule(dynamic, 1) if (nthz > 1)
                for (int ch = 0; ch < nch; ch++) {
#ifdef _OPENMP
                    const int tid = omp_get_thread_num();
#else
                    const int tid = 0;
#endif
                    double *Xt = S->Z + (size_t)ch * CB * m;
                    memset(Xt, 0, sizeof(double) * (size_t)m * CB);
                    const int a0 = ch * CB, a1 = a0 + CB < nt ? a0 + CB : nt;
                    for (int a = a0; a < a1; a++) {
                        if (a < nd) { const int j = S->dcol[a]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) Xt[(size_t)ip[S->Ai[p]] * CB + (a - a0)] += S->Ax[p]; }
                        else Xt[(size_t)ip[S->Erow[a - nd]] * CB + (a - a0)] += 1.0;
                    }
                    schol_lsolve_block(S->sc, Xt, CB, Az[tid]);
                    lp_chunk_extract(Xt, m, CB, a0, a1, zci, zcv, zcn);
                    schol_ltsolve_block(S->sc, Xt, CB, Az[tid]);
                }
                for (int t = 0; t < nthz; t++) free(Az[t]);
                free(Az);
                long tot = 0; for (int a = 0; a < nt; a++) tot += zcn[a];
                if (tot > S->Wcap) { S->Wcap = (int)(tot + 1024); S->Wi = realloc(S->Wi, sizeof(int) * (size_t)S->Wcap); S->Wx = realloc(S->Wx, sizeof(double) * (size_t)S->Wcap); }
                for (int a = 0; a < nt; a++) {
                    memcpy(S->Wi + nw, zci[a], sizeof(int) * (size_t)zcn[a]); memcpy(S->Wx + nw, zcv[a], sizeof(double) * (size_t)zcn[a]);
                    nw += zcn[a]; S->Wp[a + 1] = nw; free(zci[a]); free(zcv[a]);
                }
                free(zci); free(zcv); free(zcn);
                S->zok = 1;
            }
        }
        if (useblk && !S->zok) {
            const int CB = LP_ZCB, nch = (nt + CB - 1) / CB;
            const int *ip = schol_iperm(S->sc);
            const size_t scr = schol_lsolve_block_scratch(S->sc, CB);
            int nthb = 1;
#ifdef _OPENMP
            if (!omp_in_parallel()) nthb = omp_get_max_threads();
            if (nthb > nch) nthb = nch;
#endif
            double **Xb = xm(sizeof(double *) * nthb), **Ab = xm(sizeof(double *) * nthb);
            for (int t = 0; t < nthb; t++) { Xb[t] = xm(sizeof(double) * (size_t)m * CB); Ab[t] = xm(sizeof(double) * scr); }
            int **ci = xm(sizeof(int *) * nt); double **cv = xm(sizeof(double *) * nt); int *cn = xz(sizeof(int) * (nt + 1));
            #pragma omp parallel for num_threads(nthb) schedule(dynamic, 1) if (nthb > 1)
            for (int ch = 0; ch < nch; ch++) {
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                double *Xt = Xb[tid];
                memset(Xt, 0, sizeof(double) * (size_t)m * CB);
                const int a0 = ch * CB, a1 = a0 + CB < nt ? a0 + CB : nt;
                for (int a = a0; a < a1; a++) {
                    if (a < nd) { const int j = S->dcol[a]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) Xt[(size_t)ip[S->Ai[p]] * CB + (a - a0)] += S->Ax[p]; }
                    else Xt[(size_t)ip[S->Erow[a - nd]] * CB + (a - a0)] += 1.0;
                }
                schol_lsolve_block(S->sc, Xt, CB, Ab[tid]);
                lp_chunk_extract(Xt, m, CB, a0, a1, ci, cv, cn);
            }
            for (int t = 0; t < nthb; t++) { free(Xb[t]); free(Ab[t]); }
            free(Xb); free(Ab);
            long tot = 0; for (int a = 0; a < nt; a++) tot += cn[a];
            if (tot > S->Wcap) { S->Wcap = (int)(tot + 1024); S->Wi = realloc(S->Wi, sizeof(int) * (size_t)S->Wcap); S->Wx = realloc(S->Wx, sizeof(double) * (size_t)S->Wcap); }
            for (int a = 0; a < nt; a++) {
                memcpy(S->Wi + nw, ci[a], sizeof(int) * (size_t)cn[a]); memcpy(S->Wx + nw, cv[a], sizeof(double) * (size_t)cn[a]);
                nw += cn[a]; S->Wp[a + 1] = nw; free(ci[a]); free(cv[a]);
            }
            free(ci); free(cv); free(cn);
        } else if (S->zok) {
        } else if (nthw > 1) {
            const int nsup = schol_nsuper(S->sc);
            if (S->nthw < nthw) {
                for (int t = 0; t < S->nthw; t++) { free(S->xw_t[t]); free(S->wmark_t[t]); free(S->wsn_t[t]); }
                free(S->xw_t); free(S->wmark_t); free(S->wsn_t);
                S->xw_t = xm(sizeof(double *) * nthw); S->wmark_t = xm(sizeof(char *) * nthw); S->wsn_t = xm(sizeof(int *) * nthw);
                for (int t = 0; t < nthw; t++) { S->xw_t[t] = xz(sizeof(double) * ((size_t)m + 1)); S->wmark_t[t] = xz((size_t)nsup + 1); S->wsn_t[t] = xm(sizeof(int) * ((size_t)nsup + 1)); }
                S->nthw = nthw;
            }
            int **ci = xm(sizeof(int *) * nt); double **cv = xm(sizeof(double *) * nt); int *cn = xz(sizeof(int) * (nt + 1));
            double *cfl = xz(sizeof(double) * (nt + 1));
            #pragma omp parallel for num_threads(nthw) schedule(dynamic, 1)
            for (int a = 0; a < nt; a++) {
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                double *xw = S->xw_t[tid]; char *wmk = S->wmark_t[tid]; int *wsn = S->wsn_t[tid];
                int oi = 0; const double ov = 1.0; int ns_r;
                if (a < nd) { const int j = S->dcol[a]; ns_r = schol_lsolve_sparse(S->sc, S->Ai + S->Ap[j], S->Ax + S->Ap[j], S->Ap[j + 1] - S->Ap[j], xw, wsn, wmk); }
                else { oi = S->Erow[a - nd]; ns_r = schol_lsolve_sparse(S->sc, &oi, &ov, 1, xw, wsn, wmk); }
                long cap = 0; double fl = 0; for (int t = 0; t < ns_r; t++) { int j0, w; schol_sn_cols(S->sc, wsn[t], &j0, &w); cap += w; fl += schol_sn_solve_flops(S->sc, wsn[t]); }
                cfl[a] = fl;
                int *ii = xm(sizeof(int) * ((size_t)cap + 1)); double *vv = xm(sizeof(double) * ((size_t)cap + 1)); int c = 0;
                for (int t = 0; t < ns_r; t++) {
                    int j0, w; schol_sn_cols(S->sc, wsn[t], &j0, &w);
                    for (int q = j0; q < j0 + w; q++) { if (xw[q] != 0.0) { ii[c] = q; vv[c] = xw[q]; c++; } xw[q] = 0.0; }
                    wmk[wsn[t]] = 0;
                }
                ci[a] = ii; cv[a] = vv; cn[a] = c;
            }
            long tot = 0; for (int a = 0; a < nt; a++) tot += cn[a];
            if (tot > S->Wcap) { S->Wcap = (int)(tot + 1024); S->Wi = realloc(S->Wi, sizeof(int) * (size_t)S->Wcap); S->Wx = realloc(S->Wx, sizeof(double) * (size_t)S->Wcap); }
            for (int a = 0; a < nt; a++) {
                memcpy(S->Wi + nw, ci[a], sizeof(int) * (size_t)cn[a]); memcpy(S->Wx + nw, cv[a], sizeof(double) * (size_t)cn[a]);
                nw += cn[a]; S->Wp[a + 1] = nw; free(ci[a]); free(cv[a]);
            }
            { double f = 0; for (int a = 0; a < nt; a++) f += cfl[a]; S->w_sp_flops = f; }     /* (in column order: the same sum at any thread count) */
            free(ci); free(cv); free(cn); free(cfl);
        } else {
        double fl_seq = 0;
        for (int a = 0; a < nt; a++) {
            int ns_r;
            if (a < nd) { const int j = S->dcol[a]; ns_r = schol_lsolve_sparse(S->sc, S->Ai + S->Ap[j], S->Ax + S->Ap[j], S->Ap[j + 1] - S->Ap[j], S->xw, S->wsn, S->wmark); }
            else { one_i = S->Erow[a - nd]; ns_r = schol_lsolve_sparse(S->sc, &one_i, &one_v, 1, S->xw, S->wsn, S->wmark); }
            for (int t = 0; t < ns_r; t++) {
                int j0, w; schol_sn_cols(S->sc, S->wsn[t], &j0, &w);
                fl_seq += schol_sn_solve_flops(S->sc, S->wsn[t]);
                if (nw + w > S->Wcap) { S->Wcap = 2 * S->Wcap + w; S->Wi = realloc(S->Wi, sizeof(int) * (size_t)S->Wcap); S->Wx = realloc(S->Wx, sizeof(double) * (size_t)S->Wcap); }
                for (int c = j0; c < j0 + w; c++) { if (S->xw[c] != 0.0) { S->Wi[nw] = c; S->Wx[nw] = S->xw[c]; nw++; } S->xw[c] = 0.0; }
                S->wmark[S->wsn[t]] = 0;
            }
            S->Wp[a + 1] = nw;
        }
        S->w_sp_flops = fl_seq;
        }
        const double tgW = lp_time();
        static double t_wpart = 0, t_gpart = 0; const double tgw = lp_time(); t_wpart += tgw - tg0;
        if (getenv("BRISK_LPDBG") && (S->nfact % 10 == 0)) fprintf(stderr, "   [dense correction so far: W %.3fs, G and its factorization %.3fs]\n", t_wpart, t_gpart);
        /* G = C^-1 + W'W (lower part). (31 October 2026) W'W by the cheaper of two forms, by their counted products: the
         * column form (column a of W scattered, the dots with the later columns: sum over b of (b + 1) nnz(W_b) products) or
         * the row form (each row of W with r entries adds its outer product, r (r + 1) / 2 products), the rows in strips
         * fixed by m and nt on the threads, each strip's sum its own, the strips added in their order: the same bits at
         * any thread count. energy Q017 (LP; 242 dense columns of 160 entries, W 4.5 million entries over 116,160 rows):
         * the column form 0.46-0.64 s of each factorization's 1.0-1.2 s of correction */
        double fl_col = 0; for (int b = 0; b < nt; b++) fl_col += (double)(b + 1) * (S->Wp[b + 1] - S->Wp[b]);
        int *rc = xz(sizeof(int) * ((size_t)m + 1));
        /* (1 November 2026) W's transposition into rows on the threads: each thread counts, then places, the entries of its
         * own range of columns (ranges of equal entries), the ranges in column order, so each row lists its columns in
         * increasing order whatever the thread count; serial, it took 0.19-0.27 s of G's 0.5 s on genus-sym-g62-2 (LP; W
         * 10 million entries over 78,095 rows) */
        int wT = 1;
#ifdef _OPENMP
        if (nw >= 200000 && !omp_in_parallel()) wT = omp_get_max_threads();
#endif
        if (wT > nt) wT = nt;
        int *wa = xm(sizeof(int) * ((size_t)wT + 1));
        { int a = 0; wa[0] = 0; for (int t = 1; t < wT; t++) { const long tgt = (long)nw * t / wT; while (a < nt && S->Wp[a] < tgt) a++; wa[t] = a; } wa[wT] = nt; }
        int *wcnt = xz(sizeof(int) * (size_t)wT * ((size_t)m + 1));
        #pragma omp parallel for schedule(static, 1) num_threads(wT) if (wT > 1)
        for (int t = 0; t < wT; t++) { int *c = wcnt + (size_t)t * m; for (int p = S->Wp[wa[t]]; p < S->Wp[wa[t + 1]]; p++) c[S->Wi[p]]++; }
        #pragma omp parallel for schedule(static) num_threads(wT) if (wT > 1)
        for (int i = 0; i < m; i++) { int c = 0; for (int t = 0; t < wT; t++) c += wcnt[(size_t)t * m + i]; rc[i] = c; }
        double fl_row = 0; for (int i = 0; i < m; i++) fl_row += 0.5 * (double)rc[i] * (rc[i] + 1);
        /* (1 November 2026) the rows of at least rdmin entries go by dsyrk in the row form (below): their products counted
         * apart (fl_rd), at an eighth of a scattered update's cost */
        static int gdense = -1; if (gdense < 0) { const char *e = getenv("BRISK_LPGDENSE"); gdense = e ? atoi(e) : 1; }
        const int rdmin = gdense ? (nt / 4 > 8 ? nt / 4 : 8) : nt + 1;
        double fl_rd = 0; int nrd = 0; for (int i = 0; i < m; i++) if (rc[i] >= rdmin) { fl_rd += 0.5 * (double)rc[i] * (rc[i] + 1); nrd++; }
        /* (the strips' partial sums, nt (nt + 1) / 2 each, zeroed and added: as many strips as keep them within an eighth of
         * the products, at most 64 and within 64 MB; counted with the row form. Without that count the row form was taken
         * where 64 strips of partial sums cost more than the products: r4l4-02-tree-bounds-50, LP, 0.46 -> 1.1 s) */
        const double half = 0.5 * (double)nt * (nt + 1);
        int gns = (int)fmin(64.0, fmax(1.0, floor(fl_row / (8.0 * half))));
        { const int cap = (int)fmax(1.0, floor((64.0 * 1048576.0) / (8.0 * (double)nt * nt))); if (gns > cap) gns = cap; if (gns > m) gns = m > 0 ? m : 1; }
        /* (1 November 2026: when the long rows hold most of the products, at most as many strips as give each 128 long rows,
         * so that each dsyrk has a block of 64 rows: r4l4-02-tree-bounds-50, LP, 22 strips of about 21 long rows each took
         * G 0.023 -> 0.055 s) */
        if (fl_rd > 0.5 * fl_row && gns > nrd / 128) gns = nrd / 128 > 1 ? nrd / 128 : 1;
        static int gform = -1; if (gform < 0) { const char *e = getenv("BRISK_LPGFORM"); gform = e ? atoi(e) : -1; }     /* (1 column, 2 row: forced) */
        /* (and only while a strip's partial sum, nt^2 doubles, fits in a core's cache, 512 KB: its updates go to scattered
         * places of it; with 491 dense columns, 1.9 MB, r4l4-02-tree-bounds-50, LP, ran at 66 million updates a second and
         * its counted 8-fold saving was a loss, 0.46 -> 0.89 s; energy Q017, LP, 242 columns: 0.46 -> 0.18 s) */
        /* (1 November 2026) the scattered updates of the short rows at 16 times their count when a strip's partial sum does
         * not fit in 512 KB (r4l4-02-tree-bounds-50: a counted 8-fold saving, a 2-fold loss), the dense rows' dsyrk at an
         * eighth: genus-sym-g62-2 (LP; 800 dense columns, W 10 million entries over 78,095 rows) G 4.8 -> 0.48 s a
         * factorization in the row form, 38 -> 13 s in all */
        const double gpen = 8.0 * (double)nt * nt <= 524288.0 ? 1.0 : 16.0;
        const double fl_rowc = gpen * (fl_row - fl_rd) + fl_rd / 8.0 + 2.0 * gns * half;
        const int byrow = gform == 2 || (gform != 1 && nt >= 8 && fl_rowc < fl_col);
        /* rows of W (permuted indices) as lists of (column, value), columns increasing; kept for the solves (ne_mid) */
        S->Wrp = realloc(S->Wrp, sizeof(size_t) * ((size_t)m + 1)); S->Wrc = realloc(S->Wrc, sizeof(int) * ((size_t)nw + 1)); S->Wrv = realloc(S->Wrv, sizeof(double) * ((size_t)nw + 1));
        {
            size_t *rp = S->Wrp; int *rcol = S->Wrc; double *rval = S->Wrv;
            rp[0] = 0;
            for (int i = 0; i < m; i++) rp[i + 1] = rp[i] + (size_t)rc[i];
            #pragma omp parallel for schedule(static) num_threads(wT) if (wT > 1)
            for (int i = 0; i < m; i++) { int o = (int)rp[i]; for (int t = 0; t < wT; t++) { const int c = wcnt[(size_t)t * m + i]; wcnt[(size_t)t * m + i] = o; o += c; } }
            #pragma omp parallel for schedule(static, 1) num_threads(wT) if (wT > 1)
            for (int t = 0; t < wT; t++) { int *o = wcnt + (size_t)t * m;
                for (int a = wa[t]; a < wa[t + 1]; a++) for (int p = S->Wp[a]; p < S->Wp[a + 1]; p++) { const int q = o[S->Wi[p]]++; rcol[q] = a; rval[q] = S->Wx[p]; } }
        }
        free(wcnt); free(wa);
        size_t *rp = S->Wrp; int *rcol = S->Wrc; double *rval = S->Wrv;
        S->Wrok = 1;
        const double tgr = lp_time();
        if (byrow) {
            const int ns = gns;
            /* (1 November 2026) the rows of at least nt / 4 entries of a strip by blocks of GDB rows gathered dense and one
             * dsyrk each (one thread per call, the strips fixed: the same bits at any thread count); the scattered updates
             * of the outer products only for the shorter rows. Q083_energy (LP; 242 dense columns, W 4.3 million entries
             * over 113,256 rows): below */
            const int GDB = 64;     /* (256 measured the same on genus-sym-g62-2, LP) */
            double *Gs = xz(sizeof(double) * (size_t)ns * nt * nt);
            int nth = 1;
#ifdef _OPENMP
            if (!omp_in_parallel() && fl_row >= 1e6) nth = omp_get_max_threads();
#endif
            #pragma omp parallel for schedule(dynamic, 1) num_threads(nth) if (nth > 1)
            for (int st = 0; st < ns; st++) {
                double *g = Gs + (size_t)st * nt * nt;
                const int i0 = (int)((long long)m * st / ns), i1 = (int)((long long)m * (st + 1) / ns);
                double *D = NULL; int kb = 0;
                for (int i = i0; i < i1; i++) {
                    const int *cc = rcol + rp[i]; const double *vv = rval + rp[i]; const int r = (int)(rp[i + 1] - rp[i]);
                    if (r >= rdmin) {
                        if (!D) D = xz(sizeof(double) * (size_t)GDB * nt);
                        for (int u = 0; u < r; u++) D[(size_t)cc[u] * GDB + kb] = vv[u];
                        if (++kb == GDB) {
                            const double one_ = 1.0; const int ldd = GDB;
                            BL(dsyrk_)("L", "T", &nt, &kb, &one_, D, &ldd, &one_, g, &nt);
                            memset(D, 0, sizeof(double) * (size_t)GDB * nt); kb = 0;
                        }
                        continue;
                    }
                    for (int u = 0; u < r; u++) { const double vu = vv[u]; double *gc = g + (size_t)cc[u] * nt; for (int v = u; v < r; v++) gc[cc[v]] += vu * vv[v]; }
                }
                if (kb > 0) { const double one_ = 1.0; const int ldd = GDB; BL(dsyrk_)("L", "T", &nt, &kb, &one_, D, &ldd, &one_, g, &nt); }
                free(D);
            }
            const double tgs = lp_time();
            #pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
            for (int a = 0; a < nt; a++)
                for (int b = a; b < nt; b++) { double t = 0; for (int st = 0; st < ns; st++) t += Gs[(size_t)st * nt * nt + (size_t)a * nt + b]; S->G[b + (size_t)a * nt] = t; }
            if (getenv("BRISK_LPDBG") && S->nfact <= 3) printf("     [G by rows: W by rows %.3fs, %d strips on %d threads %.3fs, their sum %.3fs]\n", tgr - tgW, ns, nth, tgs - tgr, lp_time() - tgs);
            free(Gs);
        } else
            for (int a = 0; a < nt; a++) {
                for (int p = S->Wp[a]; p < S->Wp[a + 1]; p++) S->xw[S->Wi[p]] = S->Wx[p];
                for (int b = a; b < nt; b++) { double s = 0; for (int p = S->Wp[b]; p < S->Wp[b + 1]; p++) s += S->Wx[p] * S->xw[S->Wi[p]]; S->G[b + (size_t)a * nt] = s; }
                for (int p = S->Wp[a]; p < S->Wp[a + 1]; p++) S->xw[S->Wi[p]] = 0.0;
            }
        free(rc);
        for (int a = 0; a < nt; a++) {
            const double ci = a < nd ? 1.0 / S->theta[S->dcol[a]] : -1.0 / S->Edel[a - nd];
            if (nE) S->Glam[nt + a] = 1.0 / sqrt(fabs(ci) + S->G[a + (size_t)a * nt]);    /* (the scaling of G for its eigenvalues) */
            S->G[a + (size_t)a * nt] += ci;
        }
        if (nE) for (int a = 0; a < nt; a++) for (int b = a; b < nt; b++) S->G[b + (size_t)a * nt] *= S->Glam[nt + a] * S->Glam[nt + b];
        if (getenv("BRISK_LPDBG") && S->nfact <= 3) printf("     [dense correction: %d columns, W %d entries (%.1f per column of %d rows), %.3fs (W %.3fs by %s, G %.3fs by %s: products by columns %.2e, by rows %.2e of which %.2e in long rows); sparse solves %.2e, blocks %.2e multiplications]\n", nt, nw, (double)nw / nt, m, lp_time() - tg0, tgW - tg0, useblk ? "blocks" : "sparse solves", lp_time() - tgW, byrow ? "rows" : "columns", fl_col, fl_row, fl_rd, S->w_sp_flops, (double)nt * schol_lsolve_flops(S->sc));
        int info = 0;
        /* with rows of no sparse entry the dense columns alone cover them, and when they are more than those columns the
         * normal equations are singular (lp_densecols: 139 such rows, 20 columns; the joined factorization replaces the
         * tiny pivots, here G is singular): G by its eigenvalues, the directions of eigenvalues below 1e-12 of the
         * largest dropped. For consistent equations (b in the range of A) any solution z of G z = h gives a solution of
         * the normal equations: M (M_s^-1 r - M_s^-1 U z) = r + U C (h - G z) with M = M_s + U C U' (22 October 2026) */
        double gd0 = 0, gd1 = 0;
        if (getenv("BRISK_GDBG")) { gd0 = 1e300; for (int a = 0; a < nt; a++) { const double v = S->G[a + (size_t)a * nt]; gd0 = fmin(gd0, v); gd1 = fmax(gd1, v); } }
        if (nE == 0) BL(dpotrf_)("L", &nt, S->G, &nt, &info);
        else BL(dsyev_)("V", "L", &nt, S->G, &nt, S->Glam, S->Gwork, &S->lGwork, &info);
        if (getenv("BRISK_GDBG") && nE == 0 && !info) {     /* (debug) G's conditioning by its Cholesky pivots, the dense columns' weights */
            double p0 = 1e300, p1 = 0, t0 = 1e300, t1 = 0; int nbig = 0;
            for (int a = 0; a < nt; a++) { const double l = S->G[a + (size_t)a * nt]; p0 = fmin(p0, l * l); p1 = fmax(p1, l * l); }
            for (int d = 0; d < nd; d++) { const double t = S->theta[S->dcol[d]]; t0 = fmin(t0, t); t1 = fmax(t1, t); if (t > 1e6) nbig++; }
            printf("     [G: diagonal %.1e .. %.1e, Cholesky pivots %.1e .. %.1e (ratio %.1e); dense weights %.1e .. %.1e, %d above 1e6]\n", gd0, gd1, p0, p1, p1 / p0, t0, t1, nbig);
        }
        if (info) { S->t_fact += lp_time() - t0; return 1; }
        if (getenv("BRISK_LPDBG") && S->nfact <= 3) S->dbg_prec = 1;
        if (nE && getenv("BRISK_LPDBG") && S->nfact <= 3) {
            double lmax = 0; for (int a = 0; a < nt; a++) lmax = fmax(lmax, fabs(S->Glam[a]));
            int nz = 0, nneg = 0; for (int a = 0; a < nt; a++) { if (fabs(S->Glam[a]) <= 1e-12 * lmax) nz++; else if (S->Glam[a] < 0) nneg++; }
            printf("     [G: eigenvalues %.1e .. %.1e (largest |.| %.1e), %d dropped, %d negative; sparse factor: %d tiny pivots]\n", S->Glam[0], S->Glam[nt - 1], lmax, nz, nneg, schol_ntiny(S->sc));
        }

        S->t_dense += lp_time() - tg0; t_gpart += lp_time() - tgw;
    }
    S->t_fact += lp_time() - t0;
    return 0;
}
/* gw <- G^-1 gw (the dense columns' small matrix, factored by ne_factor) */
static void ne_gsolve(LpS *S) {
    const int nt = S->nd + S->nE, one = 1; int info = 0;
    if (S->nE == 0) { BL(dpotrs_)("L", &nt, &one, S->G, &nt, S->gw, &nt, &info); return; }
    /* G = D^-1 Gs D^-1 (D the scaling, Glam[nt..]), Gs = V diag(lambda) V': z = D V diag(1 / lambda, 0 below 1e-12 of the largest) V' D h */
    const double *sc = S->Glam + nt;
    double lmax = 0; for (int a = 0; a < nt; a++) lmax = fmax(lmax, fabs(S->Glam[a]));
    double *t = S->Gwork;
    for (int b = 0; b < nt; b++) S->gw[b] *= sc[b];
    for (int a = 0; a < nt; a++) { double v = 0; const double *va = S->G + (size_t)a * nt; for (int b = 0; b < nt; b++) v += va[b] * S->gw[b]; t[a] = fabs(S->Glam[a]) > 1e-12 * lmax ? v / S->Glam[a] : 0.0; }
    for (int b = 0; b < nt; b++) S->gw[b] = 0;
    for (int a = 0; a < nt; a++) { const double ta = t[a]; if (ta != 0.0) { const double *va = S->G + (size_t)a * nt; for (int b = 0; b < nt; b++) S->gw[b] += ta * va[b]; } }
    for (int b = 0; b < nt; b++) S->gw[b] *= sc[b];
}
/* the correction between the halves of the solve: x <- x - W G^-1 W' x on x = L^-1 P r (permuted ordering) */
static void ne_mid(void *ctx, double *x) {
    LpS *S = ctx;
    const int nt = S->nd + S->nE, m = S->m;
    /* (W' x by columns, W g by rows: each sum in a fixed order, on the threads from 200,000 entries of W) */
    int nth = 1;
#ifdef _OPENMP
    if (!omp_in_parallel() && S->Wp[nt] >= 200000) nth = omp_get_max_threads();
#endif
    #pragma omp parallel for schedule(dynamic, 4) num_threads(nth) if (nth > 1)
    for (int a = 0; a < nt; a++) { double s = 0; for (int p = S->Wp[a]; p < S->Wp[a + 1]; p++) s += S->Wx[p] * x[S->Wi[p]]; S->gw[a] = s; }
    ne_gsolve(S);
    if (S->Wrok) {
        const size_t *rp = S->Wrp; const int *rc = S->Wrc; const double *rv = S->Wrv; const double *g = S->gw;
        #pragma omp parallel for schedule(static, 4096) num_threads(nth) if (nth > 1)
        for (int i = 0; i < m; i++) { double t = 0; for (size_t q = rp[i]; q < rp[i + 1]; q++) t += rv[q] * g[rc[q]]; x[i] -= t; }
    } else
        for (int a = 0; a < nt; a++) { const double g = S->gw[a]; if (g != 0.0) for (int p = S->Wp[a]; p < S->Wp[a + 1]; p++) x[S->Wi[p]] -= g * S->Wx[p]; }
}
int schol_solve_mid(const SChol *S, double *b, void (*mid)(void *, double *), void *ctx);
/* z = P^-1 r: the factored sparse part and the dense columns by the Woodbury formula */
static void ne_prec(LpS *S, const double *r, double *z) {
    const int m = S->m;
    S->nprec++;
    if (S->nd == 0) { memcpy(z, r, sizeof(double) * (size_t)m); schol_solve(S->sc, z, S->w3); return; }
    if (S->bord) {      /* (BRISK_LPBORDER) [M_s A_d; A_d' -Theta_d^-1] [z; t] = [r; 0] */
        double *zb = S->bw, *wk = S->bw + m + S->bord + 1;
        memcpy(zb, r, sizeof(double) * (size_t)m); memset(zb + m, 0, sizeof(double) * (size_t)S->bord);
        schol_solve(S->sc, zb, wk); memcpy(z, zb, sizeof(double) * (size_t)m); return;
    }
    /* the dense columns: z = P' L^-T (x - W G^-1 W' x), x = L^-1 P r */
    const int nd = S->nd, nt = nd + S->nE;
    double *x = S->w3;
    const int withz = S->zok;     /* (31 October 2026: y = M_s^-1 r by the full solve, then y - Z G^-1 U'y) */
    if (withz) {
        memcpy(z, r, sizeof(double) * (size_t)m);
        schol_solve(S->sc, z, x);
        for (int a = 0; a < nt; a++) {
            double s = 0;
            if (a < nd) { const int j = S->dcol[a]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * z[S->Ai[p]]; }
            else s = z[S->Erow[a - nd]];
            S->gw[a] = s;
        }
    } else {
        /* (1 November 2026) the factor's solve on the threads with the correction between its halves (ne_mid): the
         * two-sweep form at the cost of the full solve, without Z */
        memcpy(z, r, sizeof(double) * (size_t)m);
        if (schol_solve_mid(S->sc, z, ne_mid, S)) return;
        schol_perm_vec(S->sc, r, x);
        schol_lsolve(S->sc, x);
        ne_mid(S, x);
        schol_ltsolve(S->sc, x);
        schol_unperm_vec(S->sc, x, z);
        return;
    }
    ne_gsolve(S);
    {
        /* z -= Z g, Z in chunks of 32 columns, each m rows of 32 in the permuted ordering: row i of the permuted vector
         * takes its sum over the columns in their order, the rows in strips on the threads (any thread count) */
        const int CB = LP_ZCB, nch = (nt + CB - 1) / CB; const int *ip = schol_iperm(S->sc);
        double *yp = x;                                       /* (w3: 2m + 2 doubles) */
        for (int i = 0; i < m; i++) yp[ip[i]] = z[i];
        double *gc = xm(sizeof(double) * (size_t)nch * CB);
        for (int a = 0; a < nch * CB; a++) gc[a] = a < nt ? S->gw[a] : 0.0;
        const int nst = (m + 4095) / 4096;
        int nth = 1;
#ifdef _OPENMP
        if (!omp_in_parallel() && (double)m * nt >= 2e6) nth = omp_get_max_threads();
#endif
        #pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (int st = 0; st < nst; st++) {
            const int r0 = st * 4096, r1 = r0 + 4096 < m ? r0 + 4096 : m;
            for (int ch = 0; ch < nch; ch++) {
                const double *Zc = S->Z + (size_t)ch * CB * m; const double *g = gc + (size_t)ch * CB;
                for (int i = r0; i < r1; i++) { const double *zr = Zc + (size_t)i * CB; double t = 0; for (int c = 0; c < CB; c++) t += zr[c] * g[c]; yp[i] -= t; }
            }
        }
        free(gc);
        for (int i = 0; i < m; i++) z[i] = yp[ip[i]];
    }
}
/* (A Theta A') x = r: the preconditioner, then CG on the true matrix until the residual is below
 * rtol |r| (at most maxcg steps); returns the relative residual */
static double ne_cg(LpS *S, double *x, double nr, double rn, double rtol, int maxcg);
static void M_mul(LpS *S, const double *x, double *y);
static double ne_solve(LpS *S, const double *r, double *x, double rtol, int maxcg) {
    const double t0 = lp_time();
    if (S->dbg_prec) {      /* (debugging) the preconditioner on a consistent right-hand side M y, y random */
        S->dbg_prec = 0;
        const int mm = S->m; double *yy = xm(sizeof(double) * mm), *rr = xm(sizeof(double) * mm), *zz = xm(sizeof(double) * mm), *qq = xm(sizeof(double) * mm);
        unsigned sd = 7; for (int i = 0; i < mm; i++) { sd = sd * 1103515245u + 12345u; yy[i] = ((sd >> 16) & 1023) / 512.0 - 1; }
        M_mul(S, yy, rr); ne_prec(S, rr, zz); M_mul(S, zz, qq);
        double e = 0, nn = 0, eE = 0; for (int i = 0; i < mm; i++) { e += (qq[i] - rr[i]) * (qq[i] - rr[i]); nn += rr[i] * rr[i]; }
        for (int k = 0; k < S->nE; k++) { const int i = S->Erow[k]; eE += (qq[i] - rr[i]) * (qq[i] - rr[i]); }
        printf("     [preconditioner on M y: relative residual %.1e (in the rows of no sparse entry %.1e)]\n", sqrt(e / nn), sqrt(eE / nn));
        {   /* the sparse part alone: M_s y = A_s Theta A_s' y + Edel y_E + reg y, solved by the factor */
            memset(rr, 0, sizeof(double) * mm);
            for (int j = 0; j < S->n; j++) { if (S->isd && S->isd[j]) continue; double t = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) t += S->Ax[p] * yy[S->Ai[p]]; t *= S->theta[j]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) rr[S->Ai[p]] += t * S->Ax[p]; }
            for (int k = 0; k < S->nE; k++) rr[S->Erow[k]] += S->Edel[k] * yy[S->Erow[k]];
            for (int i = 0; i < mm; i++) rr[i] += S->reg * yy[i];
            memcpy(zz, rr, sizeof(double) * mm); schol_solve(S->sc, zz, S->w3);
            double e2 = 0, n2 = 0; for (int i = 0; i < mm; i++) { e2 += (zz[i] - yy[i]) * (zz[i] - yy[i]); n2 += yy[i] * yy[i]; }
            printf("     [the sparse factor on M_s y: relative error %.1e; %d dense columns flagged]\n", sqrt(e2 / n2), S->isd ? 1 : 0);
        }
        free(yy); free(rr); free(zz); free(qq);
    }
    const int m = S->m;
    double *res = S->pr, *q = S->pq;
    const double nr = sqrt(dot(m, r, r));
    S->nsolve++;
    if (nr == 0) { memset(x, 0, sizeof(double) * (size_t)m); S->t_solve += lp_time() - t0; return 0; }
    ne_prec(S, r, x);
    M_mul(S, x, q);
    for (int i = 0; i < m; i++) res[i] = r[i] - q[i];
    const double rn0 = sqrt(dot(m, res, res));
    if (rn0 <= rtol * nr || maxcg <= 0) { S->t_solve += lp_time() - t0; return rn0 / nr; }
    const double out = ne_cg(S, x, nr, rn0, rtol, maxcg);
    S->t_solve += lp_time() - t0;
    return out;
}
/* CG on A Theta A' from x with the residual in S->pr (norm rn; the right-hand side's norm nr):
 * the best iterate is left in x; returns its relative residual */
static double ne_cg(LpS *S, double *x, double nr, double rn, double rtol, int maxcg) {
    const int m = S->m;
    double *res = S->pr, *z = S->pz, *p = S->pp, *q = S->pq;
    ne_prec(S, res, z);
    memcpy(p, z, sizeof(double) * (size_t)m);
    double rz = dot(m, res, z), best = rn;
    double rprev2 = rn, rprev1 = rn;
    double *xb = S->w2; memcpy(xb, x, sizeof(double) * (size_t)m);
    for (int k = 0; k < maxcg; k++) {
        M_mul(S, p, q);
        const double pq = dot(m, p, q);
        if (!(pq > 0) || !isfinite(pq)) break;
        const double al = rz / pq;
        for (int i = 0; i < m; i++) { x[i] += al * p[i]; res[i] -= al * q[i]; }
        S->ncg++;
        rn = sqrt(dot(m, res, res));
        if (rn < best) { best = rn; memcpy(xb, x, sizeof(double) * (size_t)m); }
        if (rn <= rtol * nr) break;
        if (rn > 1e3 * best) break;
        /* no progress in two steps (replaced pivots: the preconditioner is singular there): the direct solve stands */
        if (k >= 1 && rn > 0.5 * rprev2) break;
        rprev2 = rprev1; rprev1 = rn;
        ne_prec(S, res, z);
        const double rz2 = dot(m, res, z), be = rz2 / rz; rz = rz2;
        for (int i = 0; i < m; i++) p[i] = z[i] + be * p[i];
    }
    memcpy(x, xb, sizeof(double) * (size_t)m);
    if (best / nr > S->worst) S->worst = best / nr;
    return best / nr;
}

static void lps_free(LpS *S) {
    free(S->Fp); free(S->Fj); free(S->Fx);
    free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Rj); free(S->Rq); free(S->Rx); free(S->b); free(S->c); free(S->u); free(S->bd); free(S->fr);
    free(S->rs); free(S->cs); free(S->dcol); free(S->isd); free(S->djoin); free(S->qv); free(S->Bo); free(S->bw); free(S->bwk); if (S->sc) schol_free(S->sc); free(S->Mp); free(S->Mi); free(S->Mo);
    free(S->Z); free(S->V); free(S->Vw); free(S->G); free(S->gw); free(S->Wp); free(S->Wi); free(S->Wx); free(S->Wrp); free(S->Wrc); free(S->Wrv); S->Wrp = NULL; S->Wrc = NULL; S->Wrv = NULL; S->Wrok = 0; free(S->xw); free(S->wmark); free(S->wsn); free(S->Erow); free(S->Edel); free(S->Gpiv); free(S->Gwork); free(S->Glam); free(S->dreg); free(S->theta); free(S->w1); free(S->w2); free(S->w4); free(S->w3); free(S->wn); free(S->acc); if (S->acc_t) { for (int t = 0; t < S->nacc_t; t++) free(S->acc_t[t]); free(S->acc_t); }
    free(S->pr); free(S->pz); free(S->pp); free(S->pq);
    free(S->Md); free(S->Wd); free(S->scol); free(S->wa); free(S->vpart);
}

/* 18 October 2026 (LP/SOCP): the largest factor (entries) that fits: half of the memory the process may use (the
 * physical memory, or the address-space limit when one is set), at 8 bytes an entry and twice that for the
 * analysis and the panels. Before, 2^31 entries (17 GB): a staircase LP whose factor needs 1.96e8 entries more than
 * the 7 GB of the host was killed instead of being handed to the SOCP solver. */
static size_t lp_fillcap(void) {
    double mem = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE);
    struct rlimit rl; if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && (double)rl.rlim_cur < mem) mem = (double)rl.rlim_cur;
    if (!(mem > 0)) mem = 8e9;
    double cap = 0.5 * mem / 16.0;
    if (cap > (double)((size_t)1 << 31)) cap = (double)((size_t)1 << 31);
    return (size_t)cap;
}
/* the dense columns, the sparse columns by rows, the pattern of A_s A_s' and its analysis
 * (again, without dense columns, when the Woodbury correction loses its accuracy); 0: done */
typedef struct { double flops, lnz; int nd, ok; double dthr, dnz; int pat; } LpNeCost;
static void ne_cost_thr(int m, int n, const int *Ap, const int *Ai, const double *Ax, LpNeCost *C, double budget, double thr, const volatile double *cap);
static double ne_dense_choice(int m, int n, const int *Ap, const int *Ai, const double *Ax, int verbose, double budget, LpNeCost *out);
static double ne_iter_cost_of(const LpS *S);   /* (an iteration's cost by ne_iter_cost on S's analysis) */
static int ne_setup_in(LpS *S, int usedense, const int *rpos);
void schol_set_snsign(const signed char *sg);
int schol_ndyn(const SChol *S);
/* (4 November 2026, BRISK_LPFREEQD=1) the free columns as unknowns of the factored matrix
 *     K = [A_s Theta_s A_s' + delta I, A_b; A_b', -D],
 * A_b the free columns (D = rho I, rho = BRISK_LPFREEQDRHO, 1e-8) and the long columns (D = Theta_d^-1), signs -1 on their
 * vertices, no supernode holding both signs. The order: the rows by the ordering of the normal equations with the free columns in
 * them (a first analysis), each short free column placed right after the first of its rows that has a sparse entry
 * (BRISK_LPFREEQDPOS=0: after the last), then the long columns, then the rows with no sparse entry. After a row, the free
 * column's pivot is -rho minus that row's contribution, never -rho alone (which would put 1/rho into the factor, the
 * proximal method's weight); after the first row rather than the last, its column of the factor is that row's, and the factor
 * costs about what the normal equations do (after the last: stockholm, LP, 3.66e9 flops against 2.47e8; after the first,
 * 2.81e8) */
static int ne_setup(LpS *S, int usedense) {
    if (!S->qd || S->nfree <= 0 || S->est) return ne_setup_in(S, usedense, NULL);
    const int m = S->m;
    S->qd = 0;                               /* (pass 1: the normal equations' order with the free columns in them) */
    int st = ne_setup_in(S, usedense, NULL);
    S->qd = 1;
    if (st || !S->sc) return st ? st : 1;
    int *rpos = xm(sizeof(int) * ((size_t)m + 1)); memcpy(rpos, schol_iperm(S->sc), sizeof(int) * (size_t)m);
    const double fl1 = schol_flops(S->sc);
    st = ne_setup_in(S, usedense, rpos);
    free(rpos);
    if (getenv("BRISK_LPDBG") && S->sc) printf("     [free columns in the factored matrix: %d border vertices (%d free columns among the rows); factor %.3g flops against %.3g for the normal equations with them]\n", S->bord, S->qdns, schol_flops(S->sc), fl1);
    return st;
}
static int ne_setup_in(LpS *S, int usedense, const int *rpos) {
    const int m = S->m, n = S->n, nz = S->Ap[n];
    free(S->qv); S->qv = NULL;
    if (ne_expired()) return 1;        /* (26 October 2026: past the time limit, or stopped: no analysis) */
    /* (25 October 2026) an estimate under a shared cap (the dense-column choice): its value now as the budget of the checks
     * on the pattern below (the bounds w^2 / m and the pattern's growth), which come before AMD's own stop at the cap:
     * savsched1's dual form (LP), the default rule's pattern alone took most of 46 s */
    if (S->est && S->est_cap && *S->est_cap > 0 && (S->est_budget <= 0 || *S->est_cap < S->est_budget)) S->est_budget = *S->est_cap;
    free(S->Rp); free(S->Rj); free(S->Rq); free(S->Rx); free(S->Mp); free(S->Mi); free(S->Mo); free(S->V); free(S->Vw); free(S->G); free(S->gw);
    free(S->Erow); free(S->Edel); free(S->Gpiv); free(S->Gwork); free(S->Glam); S->Erow = S->Gpiv = NULL; S->Edel = S->Gwork = S->Glam = NULL; S->nE = 0; S->lGwork = 0; S->ntcap = 0; S->Ecap = 0;
    for (int t = 0; t < S->nthw; t++) { free(S->xw_t[t]); free(S->wmark_t[t]); free(S->wsn_t[t]); }
    free(S->xw_t); free(S->wmark_t); free(S->wsn_t); S->xw_t = NULL; S->wmark_t = NULL; S->wsn_t = NULL; S->nthw = 0;
    free(S->Wp); free(S->Wi); free(S->Wx); free(S->Wrp); free(S->Wrc); free(S->Wrv); S->Wrp = NULL; S->Wrc = NULL; S->Wrv = NULL; S->Wrok = 0; free(S->xw); free(S->wmark); free(S->wsn); S->Wp = S->Wi = S->wsn = NULL; S->Wx = S->xw = NULL; S->wmark = NULL; S->Wcap = 0;
    S->Rp = S->Rj = S->Rq = S->Mp = S->Mi = NULL; S->Rx = S->V = S->Vw = S->G = S->gw = NULL; S->Mo = NULL;
    if (S->sc) { schol_free(S->sc); S->sc = NULL; }
    if (!S->isd) { S->isd = xz(n + 1); S->dcol = xm(sizeof(int) * (n + 1)); }
    memset(S->isd, 0, n + 1); S->nd = 0;
    /* dense columns: long against the average and against the rows, and few */
    if (usedense && m >= 500) {
        double avg = (double)nz / fmax(n, 1);
        const double thr = S->dthr > 0 ? S->dthr : fmax(fmax(50.0, 10.0 * avg), 0.1 * m);
        int nd = 0;
        for (int j = 0; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] > thr && !(S->djoin && S->djoin[j])) nd++;
        if (nd > 0 && nd <= (S->dthr > 0 ? LP_NDMAX : 400) && nd < n / 4) { for (int j = 0; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] > thr && !(S->djoin && S->djoin[j])) { S->isd[j] = 1; S->dcol[S->nd++] = j; } }
    }
    S->qdns = 0;
    if (rpos) {      /* (BRISK_LPFREEQD) the free columns join the border: the short ones first (placed among the rows), then the long ones */
        int *dl = xm(sizeof(int) * ((size_t)S->nd + 1)); const int nd0 = S->nd; memcpy(dl, S->dcol, sizeof(int) * (size_t)nd0);
        S->nd = 0;
        for (int j = 0; j < S->nfree; j++) if (!S->isd[j]) S->dcol[S->nd++] = j;     /* (they stay in the sparse part too, at the weight gamma) */
        S->qdns = S->nd;
        for (int d = 0; d < nd0; d++) S->dcol[S->nd++] = dl[d];
        free(dl);
    }
    /* (22 October 2026, the night) an estimate against a budget: the dense columns' part of an iteration alone, at least
     * nd (2 m) for their solves (ne_iter_cost; 2 nd^2 m before the afternoon),
     * above it: no pattern, no analysis (neos-4532248-waihi's primal form, 290 columns over 42,907 rows) */
    if (S->est && S->est_budget > 0 && 2.0 * S->nd * (double)m > S->est_budget) { S->est_lb = 2.0 * S->nd * (double)m; S->est_w = 0; return 0; }
    /* ... and the clique of the longest sparse column: its L rows are a clique of A_s A_s', whose factorization in any
     * order costs at least 1^2 + ... + (L-1)^2 flops */
    if (S->est && S->est_budget > 0) {
        int lm = 0; for (int j = 0; j < n; j++) if (!S->isd[j] && S->Ap[j + 1] - S->Ap[j] > lm) lm = S->Ap[j + 1] - S->Ap[j];
        const double lb = (lm - 1.0) * lm * (2.0 * lm - 1.0) / 6.0;
        if (lb > S->est_budget) { S->est_lb = lb; S->est_w = (long)lm * (lm + 1) / 2; S->est_pat = 1; return 0; }
    }
    /* the sparse columns by rows */
    S->Rp = xz(sizeof(int) * (m + 2));
    int nzs = 0;
    for (int j = 0; j < n; j++) if (!S->isd[j]) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { S->Rp[S->Ai[p] + 2]++; nzs++; }
    for (int i = 0; i < m; i++) S->Rp[i + 2] += S->Rp[i + 1];
    /* the rows the dense columns alone cover: no entry of the factored matrix A_s Theta A_s' (a zero pivot, which the
     * factorization would replace and the correction could not undo: neos-4763324-toguru, the row of the 231 dense
     * columns y, residual 1.7 at the first iteration) */
    if (S->nd > 0 && !S->est) {
        int ne0 = 0; for (int i = 0; i < m; i++) ne0 += S->Rp[i + 2] == S->Rp[i + 1];
        if (ne0 > 0) { S->Erow = xm(sizeof(int) * (size_t)ne0); S->Edel = xm(sizeof(double) * (size_t)ne0);
            for (int i = 0; i < m; i++) if (S->Rp[i + 2] == S->Rp[i + 1]) S->Erow[S->nE++] = i; }
        if (getenv("BRISK_LPDBG")) printf("   [dense columns apart: %d rows with no entry left]\n", ne0);
    }
    S->Rj = xm(sizeof(int) * (nzs + 1)); S->Rq = xm(sizeof(int) * (nzs + 1)); S->Rx = xm(sizeof(double) * (nzs + 1));
    for (int j = 0; j < n; j++) if (!S->isd[j]) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = S->Rp[S->Ai[p] + 1]++; S->Rj[q] = j; S->Rq[q] = p; S->Rx[q] = S->Ax[p]; }
    /* 18 October 2026 (LP/SOCP): the assembly by dsyrk when it is cheaper: the scattered assembly costs
     * sum_j len_j (len_j + 1) / 2 updates of one entry each, the dense one nsc m (m + 1) / 2 flops at BLAS-3 speed,
     * about 20 times the rate of a scattered update (a dense A: 300 rows and 10,000 columns of 300 entries,
     * 0.5 s an assembly by rows). The buffer for M (m^2 entries) is bounded by 512 MB. Columns that long (a
     * fifth of the rows on average) make A A' full: its pattern is then taken as full (building it entry by entry
     * cost as much as the scattered assembly). BRISK_LPDASM=0/1: off/forced. */
    free(S->Md); free(S->Wd); free(S->scol); S->Md = S->Wd = NULL; S->scol = NULL; S->dasm = 0;
    {
        double ws = 0; int nsc = 0;
        for (int j = 0; j < n; j++) if (!S->isd[j]) { const double l = S->Ap[j + 1] - S->Ap[j]; ws += 0.5 * l * (l + 1); nsc++; }
        const double wd = 0.5 * (double)nsc * (double)m * (m + 1.0);
        const char *e = getenv("BRISK_LPDASM");
        int on = m <= 8192 && wd < 20.0 * ws;
        if (e) on = atoi(e) != 0 && m <= 8192;
        S->dasm = on; S->nsc = nsc;
    }
    const double tsu0 = lp_time();
    /* the pattern of A_s A_s' (lower, by columns; the diagonal first) */
    int *mark = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) mark[i] = -1;
    long cap = 4L * nzs + m + 16, w = 0;
    int *Mi = xm(sizeof(int) * cap); S->Mp = xm(sizeof(int) * (m + 1));
    /* (22 October 2026, the night) an estimate against a budget stops as soon as the pattern alone exceeds it (the bound
     * below): sorrell3 folded, priced with its 271 columns of 50 to 2,700 entries in the factor, built a pattern of 2.7e8
     * entries in 13 s to find 2.65e12 flops against the other choice's 4.3e9 */
    const double wcap = S->est && S->est_budget > 0 ? sqrt(S->est_budget * m) : 1e300;
    if (S->dasm) { cap = (long)m * (m + 1) / 2 + 1; Mi = realloc(Mi, sizeof(int) * cap); for (int i = 0; i < m; i++) { S->Mp[i] = (int)w; for (int k = i; k < m; k++) Mi[w++] = k; } }
    else for (int i = 0; i < m; i++) {
        S->Mp[i] = (int)w;
        if (w + m + 1 > cap) { cap = 2 * cap + m; Mi = realloc(Mi, sizeof(int) * cap); }
        Mi[w++] = i; mark[i] = i;
        for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) {
            const int j = S->Rj[q];
            for (int p = S->Rq[q] + 1; p < S->Ap[j + 1]; p++) { const int k = S->Ai[p]; if (mark[k] != i) { mark[k] = i; Mi[w++] = k; } }
        }
        if ((double)w > wcap) { for (int r = i + 1; r <= m; r++) S->Mp[r] = (int)w; S->Mi = Mi; free(mark); S->est_lb = (double)w * (double)w / m; S->est_w = w; S->est_pat = 1; return 0; }
        /* (28 October 2026, the night) an estimate stopped by the caller's flag or the deadline, here too (as AMD's count is,
         * est_st 3 below): neos-5106984-jizera (LP), the dual form's pattern took seconds after the dual simplex method
         * beside it had its result, and the run waited 5.8 s for it */
        if (S->est && (i & 4095) == 4095 && ne_expired()) { for (int r = i + 1; r <= m; r++) S->Mp[r] = (int)w; S->Mi = Mi; free(mark); S->est_lb = 1e300; S->est_w = (long)1 << 62; return 0; }
        /* (3 November 2026, ISSUES 825) the method's own set-up stops there too (the caller reports "stopped during the
         * analysis"): neos-2974461-ibar (LP), its second attempt with 34 s left built a pattern of 343 million entries for 52 s,
         * and the run with -timelimit 120 ended at 140 s */
        if (!S->est && (i & 4095) == 4095 && ne_expired()) { for (int r = i + 1; r <= m; r++) S->Mp[r] = (int)w; S->Mi = Mi; free(mark); return 1; }
    }
    S->Mp[m] = (int)w; S->Mi = Mi;
    free(mark);
    /* (a cost estimate against a budget: the factor has at least the w entries of the pattern, so by Cauchy-Schwarz
     * its factorization costs at least w^2 / m flops; above the budget the ordering is not made) */
    if (S->est && S->est_budget > 0 && (double)w * (double)w / m > S->est_budget) { S->est_lb = (double)w * (double)w / m; S->est_w = w; S->est_pat = 1; return 0; }
    /* (28 October 2026, the night) stopped meanwhile: no adjacency, no ordering (neos-5106984-jizera, LP, above) */
    if (S->est && ne_expired()) { S->est_lb = 1e300; S->est_w = (long)1 << 62; return 0; }
    int *deg = xz(sizeof(int) * (m + 1)), **nbr = xm(sizeof(int *) * (m + 1));
    for (int i = 0; i < m; i++) for (int e = S->Mp[i] + 1; e < S->Mp[i + 1]; e++) { deg[i]++; deg[Mi[e]]++; }
    int *pool = xm(sizeof(int) * (2 * (w - m) + 1));
    { long o = 0; for (int i = 0; i < m; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; } }
    for (int i = 0; i < m; i++) for (int e = S->Mp[i] + 1; e < S->Mp[i + 1]; e++) { const int k = Mi[e]; nbr[i][deg[i]++] = k; nbr[k][deg[k]++] = i; }
    /* (23 October 2026) an estimate: AMD's own counts of the factor's entries and flops in its order (schol_amd_estimate),
     * without the symbolic factorization, the supernodes and the pattern's offsets, which the estimate does not use; no
     * global setting is touched, so the dense-column choice runs its candidates' estimates on the threads. Before, each
     * estimate ran the analysis of a factorization: the candidates took 11.6 s of physiciansched3-3 (LP), 2.1 s of rail01,
     * 1.1 s of sing44, whose interior-point method takes 1.3 s. BRISK_LPESTAMD=0: the analysis as before */
    {
        static int estamd = -1; if (estamd < 0) { const char *e = getenv("BRISK_LPESTAMD"); estamd = !(e && atoi(e) == 0); }
        double efl = 0, enz = 0;
        /* (25 October 2026) AMD stops once its count passes the cap (the budget, or the dense-column choice's shared cap):
         * the count so far is then a lower bound, as the budget's other exits give one */
        volatile double capb = S->est_budget > 0 ? S->est_budget : 0;
        const volatile double *cap = S->est_cap ? S->est_cap : S->est_budget > 0 ? &capb : NULL;
#ifdef HAVE_AMD
        brisk_amd_deadline = ne_dl; brisk_amd_stop = ne_stopf;
#endif
        const double tdbg0 = lp_time();
        const int est_st = S->est && estamd && !S->est_nd ? schol_amd_estimate_cap(m, deg, nbr, cap, &efl, &enz) : 1;
#ifdef HAVE_AMD
        brisk_amd_deadline = 0; brisk_amd_stop = NULL;
#endif
        if (getenv("BRISK_LPDBG")) printf("     [ne_setup %d: the pattern %ld entries; AMD's count %.3fs, status %d]\n", m, w, lp_time() - tdbg0, est_st);
        if (est_st != 1) {
            free(deg); free(nbr); free(pool);
            /* (26 October 2026) stopped by the clock or the stop flag (3): its count is no lower bound worth using (an
             * estimate stopped early counts few flops and was chosen: opm2-z12-s8's primal form at 7.85e8 against the
             * dual's 7.85e10, whose real analysis then took 147 s); the form or choice costs "too much" */
            S->est_lb = est_st == 3 ? 1e300 : efl; S->est_w = est_st == 3 ? (long)1 << 62 : (long)enz;
            return 0;
        }
    }
    if (getenv("BRISK_LPDUMP")) {          /* (experiments: the pattern of the normal equations, one row "i: neighbours") */
        FILE *fd = fopen(getenv("BRISK_LPDUMP"), "w");
        if (fd) { fprintf(fd, "%d\n", m); for (int i = 0; i < m; i++) { for (int e = 0; e < deg[i]; e++) fprintf(fd, "%d ", nbr[i][e]); fprintf(fd, "\n"); } fclose(fd); }
    }
    int *fperm = NULL;
    if (getenv("BRISK_LPPERM")) {          /* (experiments: an ordering from a file, perm[new] = old) */
        FILE *fd = fopen(getenv("BRISK_LPPERM"), "r");
        if (fd) { fperm = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) if (fscanf(fd, "%d", &fperm[i]) != 1) { free(fperm); fperm = NULL; break; } fclose(fd); }
        if (fperm) schol_set_perm(fperm);
    }
    /* (an estimate under a budget: AMD alone, without the own orderings that follow when it stops at the budget, whose work
     * grows with the flops up to the same budget: neos-1122047's primal form, 1.9 s of its 3.3 s, 22 October 2026) */
    const int om = schol_set_amd(getenv("BRISK_LPAMD") ? atoi(getenv("BRISK_LPAMD")) : S->est && S->est_budget > 0 ? 3 : 2);
    /* 18 October 2026 (LP/SOCP): no nested dissection on a pattern with more than a tenth of its entries: a separator
     * of a graph that dense is most of it, and the bisection's matching and refinement cost O(edges) a level (L1
     * regression with its free columns joined: 35 s in the bisection for a factor that AMD orders at once) */
    const int dense_pat = (double)(w - m) > 0.05 * (double)m * (m - 1.0);
    schol_set_nd((S->est && !S->est_nd) || dense_pat ? 0 : getenv("BRISK_LPND") ? atoi(getenv("BRISK_LPND")) : 1);
    const double oa = schol_set_amalg(0.3);      /* 5.7: relaxed supernodes (30 % explicit zeros at most, 64 columns) */
    const int osib = schol_set_amalg_siblings(1);   /* (21 October 2026: and siblings with the same pattern) */
    /* (22 October 2026, the night) an estimate against a budget: the analysis stops once the factorization's flops pass
     * it (the estimate is then "above the budget", as from the pattern above). sorrell3 folded: its dense-column choice
     * priced the factor with 271 columns of 50 to 2,700 entries in it, a sparse pattern whose factor is nearly full, to
     * find 2.65e12 flops against the other choice's 4.3e9: 13 s of its 14 s, the solve 0.03 s */
    if (S->est && S->est_budget > 0) schol_set_flopcap(S->est_budget);
    /* (28 October 2026) the stop flag to the dissection (nd.c): stopped, it gives way to AMD's order at once (satellites2-25,
     * LP: the dual simplex method beside it had won after 2.1 s, and this analysis ran on for 11 s) */
    /* (31 October 2026) and the deadline, to the dissection and to the symbolic analysis (sparsechol.c), which writes every
     * row index of the factor: multicommodity scale 4 (LP) spent 9 s in this call with a time limit of 2.6 s, degenerate_cover
     * scale 4 (LP) 5.7 s. Both clocks are CLOCK_MONOTONIC, as lp_time */
    extern void nd_set_stop(const volatile int *f); extern void nd_set_stop_deadline(double t);
    nd_set_stop(ne_stopf); nd_set_stop_deadline(ne_dl); schol_set_analysis_stop(ne_stopf, ne_dl);
    const double tsu1 = lp_time();
    S->bord = 0;
    if (rpos) {
        /* (BRISK_LPFREEQD) the order: the rows with a sparse entry by their positions in the first analysis, each short free column
         * right after the first of its rows, then the long columns, then the rows with no sparse entry */
        const int nb = S->nd, mb = m + nb;
        int *ord = xm(sizeof(int) * ((size_t)m + 1)), *last = xm(sizeof(int) * ((size_t)nb + 1)), *bhead = xm(sizeof(int) * ((size_t)m + 1)), *bnext = xm(sizeof(int) * ((size_t)nb + 1));
        char *bare = xz((size_t)m + 1);
        for (int i = 0; i < m; i++) { ord[rpos[i]] = i; bhead[i] = -1; bare[i] = S->Rp[i + 1] == S->Rp[i]; }
        for (int k = 0; k < nb; k++) {
            last[k] = -1;
            /* (BRISK_LPFREEQDPOS=1: after the first of its rows instead, an experiment: its pivot then holds that row's
             * contribution, and its column of the factor is the first row's, shorter than the last one's) */
            static int qpos = -1; if (qpos < 0) { const char *e_ = getenv("BRISK_LPFREEQDPOS"); qpos = e_ ? atoi(e_) : 1; }
            if (k < S->qdns) { const int j = S->dcol[k]; int fst = m; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; if (!bare[i] && rpos[i] > last[k]) last[k] = rpos[i]; if (!bare[i] && rpos[i] < fst) fst = rpos[i]; }
                if (qpos == 1 && fst < m) last[k] = fst; }
            if (last[k] >= 0) { bnext[k] = bhead[last[k]]; bhead[last[k]] = k; }
        }
        int *bperm = xm(sizeof(int) * ((size_t)mb + 1)); int q = 0;
        for (int t = 0; t < m; t++) { const int i = ord[t]; if (bare[i]) continue; bperm[q++] = i; for (int k = bhead[t]; k >= 0; k = bnext[k]) bperm[q++] = m + k; }
        for (int k = 0; k < nb; k++) if (last[k] < 0) bperm[q++] = m + k;
        for (int t = 0; t < m; t++) if (bare[ord[t]]) bperm[q++] = ord[t];
        int *deg2 = xz(sizeof(int) * ((size_t)mb + 1)), **nbr2 = xm(sizeof(int *) * ((size_t)mb + 1));
        long tot = 0;
        for (int i = 0; i < m; i++) deg2[i] = deg[i];
        for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { deg2[S->Ai[p]]++; deg2[m + k]++; } }
        for (int i = 0; i < mb; i++) tot += deg2[i];
        int *pool2 = xm(sizeof(int) * ((size_t)tot + 1));
        { long o = 0; for (int i = 0; i < mb; i++) { nbr2[i] = pool2 + o; o += deg2[i]; } }
        for (int i = 0; i < m; i++) { memcpy(nbr2[i], nbr[i], sizeof(int) * (size_t)deg[i]); deg2[i] = deg[i]; }
        for (int k = 0; k < nb; k++) deg2[m + k] = 0;
        for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; nbr2[i][deg2[i]++] = m + k; nbr2[m + k][deg2[m + k]++] = i; } }
        signed char *sg = xm((size_t)mb + 1); for (int i = 0; i < mb; i++) sg[i] = i < m ? 1 : -1;
        static int qdord = -1; if (qdord < 0) { const char *e_ = getenv("BRISK_LPFREEQDORD"); qdord = e_ ? atoi(e_) : 1; }   /* (0: the analysis' own order, unconstrained) */
        schol_set_perm(qdord ? bperm : NULL); schol_set_snsign(sg);
        S->sc = schol_analyze_adj(mb, deg2, nbr2, lp_fillcap());
        schol_set_perm(fperm); schol_set_snsign(NULL);
        if (S->sc) { schol_set_signs(S->sc, sg); S->bord = nb; }
        if (S->sc && S->nE > 0) { free(S->Erow); free(S->Edel); S->Erow = NULL; S->Edel = NULL; S->nE = 0; }
        free(sg); free(bperm); free(deg2); free(nbr2); free(pool2); free(ord); free(last); free(bhead); free(bnext); free(bare);
    } else
    S->sc = schol_analyze_adj(m, deg, nbr, lp_fillcap());
    const double tsu2 = lp_time();
    {   static int lpborder = -1; if (lpborder < 0) { const char *e = getenv("BRISK_LPBORDER"); lpborder = e ? atoi(e) : 2; }
        if (!rpos && S->sc && lpborder > 0 && !S->noborder && S->nd > 0 && !S->est && !S->dasm) {
            /* (the rows that only the dense columns cover, no entry of M_s: ordered after the border from the start, as the weak
             * rows after a loss, so that their pivots hold the dense columns' contribution, their only one; genus-sym-g62-2, LP:
             * 256 such rows beside 800 dense columns kept the Woodbury correction, 9.4 s of its 13 s) */
            int own_bwk = 0;
            if (S->nE > 0 && S->nbw == 0) { free(S->bwk); S->bwk = xm(sizeof(int) * (size_t)S->nE); memcpy(S->bwk, S->Erow, sizeof(int) * (size_t)S->nE); S->nbw = S->nE; own_bwk = 1; }
            /* the bordered matrix: the dense columns as nd further vertices, each adjacent to its rows, ordered after the
             * sparse part's own ordering (the Schur complement on them is then the Woodbury matrix G, computed by the
             * factorization's block updates) */
            const int nb = S->nd, mb = m + nb; const int *ip = schol_iperm(S->sc);
            int *bperm = xm(sizeof(int) * ((size_t)mb + 1));
            /* (the weak rows, after a loss: rows where the sparse part's diagonal is negligible against the dense columns'
             * contribution, ordered after the border, so that their pivots include that contribution) */
            {   int *ord = xm(sizeof(int) * ((size_t)m + 1)); char *wk = xz((size_t)m + 1);
                for (int t = 0; t < S->nbw; t++) wk[S->bwk[t]] = 1;
                for (int i = 0; i < m; i++) ord[ip[i]] = i;
                int q = 0; for (int t = 0; t < m; t++) if (!wk[ord[t]]) bperm[q++] = ord[t];
                for (int k = 0; k < nb; k++) bperm[q++] = m + k;
                for (int t = 0; t < m; t++) if (wk[ord[t]]) bperm[q++] = ord[t];
                free(ord); free(wk); }
            int *deg2 = xz(sizeof(int) * ((size_t)mb + 1)), **nbr2 = xm(sizeof(int *) * ((size_t)mb + 1));
            long tot = 0;
            for (int i = 0; i < m; i++) deg2[i] = deg[i];
            for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { deg2[S->Ai[p]]++; deg2[m + k]++; } }
            for (int i = 0; i < mb; i++) tot += deg2[i];
            int *pool2 = xm(sizeof(int) * ((size_t)tot + 1));
            { long o = 0; for (int i = 0; i < mb; i++) { nbr2[i] = pool2 + o; o += deg2[i]; } }
            for (int i = 0; i < m; i++) { memcpy(nbr2[i], nbr[i], sizeof(int) * (size_t)deg[i]); deg2[i] = deg[i]; }
            for (int k = 0; k < nb; k++) deg2[m + k] = 0;
            for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; nbr2[i][deg2[i]++] = m + k; nbr2[m + k][deg2[m + k]++] = i; } }
            const double fl0 = schol_flops(S->sc);
            const double tb0 = lp_time();
            schol_set_perm(bperm); { extern void schol_set_snbreak(int, int); schol_set_snbreak(m - S->nbw, S->nbw > 0 ? m - S->nbw + nb : -1); }
            SChol *B = schol_analyze_adj(mb, deg2, nbr2, lp_fillcap());
            schol_set_perm(fperm); { extern void schol_set_snbreak(int, int); schol_set_snbreak(-1, -1); }
            if (B) {
                signed char *sg = xm((size_t)mb + 1); for (int i = 0; i < mb; i++) sg[i] = i < m ? 1 : -1;
                schol_set_signs(B, sg); free(sg);
                if (getenv("BRISK_LPDBG")) fprintf(stderr, "   [bordered: %d dense columns as rows; factor flops %.3g (the sparse part alone %.3g), entries %.3g, %d supernodes; analysis %.2fs]\n", nb, schol_flops(B), fl0, (double)schol_nnz(B), schol_nsuper(B), lp_time() - tb0);
                schol_free(S->sc); S->sc = B; S->bord = nb;
                if (S->nE > 0) { free(S->Erow); free(S->Edel); S->Erow = NULL; S->Edel = NULL; S->nE = 0; }
            } else if (own_bwk) { free(S->bwk); S->bwk = NULL; S->nbw = 0; }
            free(bperm); free(deg2); free(nbr2); free(pool2);
        }
    }
    nd_set_stop(NULL); nd_set_stop_deadline(0); schol_set_analysis_stop(NULL, 0);
    if (S->est && S->est_budget > 0) schol_set_flopcap(0);
    schol_set_amalg(oa); schol_set_amalg_siblings(osib);
    schol_set_nd(0);
    if (fperm) { schol_set_perm(NULL); free(fperm); }
    schol_set_amd(om);
    free(deg); free(nbr); free(pool);
    if (!S->sc) return 1;
    if (S->est) return 0;
    const double tsu3 = lp_time();
    S->Mo = xm(sizeof(size_t) * (w + 1));
    for (int i = 0; i < m; i++) for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) {
        S->Mo[e] = schol_offset(S->sc, (size_t)Mi[e], (size_t)i);
        if (S->Mo[e] == (size_t)-1) return 1;
    }
    free(S->Bo); S->Bo = NULL; free(S->bw); S->bw = NULL;
    if (S->bord) {
        size_t nb = 0; for (int k = 0; k < S->bord; k++) nb += (size_t)(S->Ap[S->dcol[k] + 1] - S->Ap[S->dcol[k]]) + 1;
        S->Bo = xm(sizeof(size_t) * (nb + 1)); S->bw = xm(sizeof(double) * (2 * ((size_t)m + S->bord) + 2));
        size_t q = 0;
        for (int k = 0; k < S->bord; k++) { const int j = S->dcol[k];
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { S->Bo[q] = schol_offset(S->sc, (size_t)(m + k), (size_t)S->Ai[p]); if (S->Bo[q++] == (size_t)-1) return 1; }
            S->Bo[q] = schol_offset(S->sc, (size_t)(m + k), (size_t)(m + k)); if (S->Bo[q++] == (size_t)-1) return 1; }
    }
    if (getenv("BRISK_LPDBG")) fprintf(stderr, "   [setup: pattern and adjacency %.3fs, analysis %.3fs, border %.3fs, offsets %.3fs]\n", tsu1 - tsu0, tsu2 - tsu1, tsu3 - tsu2, lp_time() - tsu3);
    if (S->nd && !S->bord) { const size_t nt = (size_t)S->nd + S->nE;
        S->G = xm(sizeof(double) * nt * nt); S->gw = xm(sizeof(double) * (nt + 1));
        S->Wp = xm(sizeof(int) * (nt + 1)); S->Wcap = (int)(4 * nt + 1024); S->Wi = xm(sizeof(int) * (size_t)S->Wcap); S->Wx = xm(sizeof(double) * (size_t)S->Wcap);
        S->xw = xz(sizeof(double) * ((size_t)m + 1)); S->wmark = xz((size_t)schol_nsuper(S->sc) + 1); S->wsn = xm(sizeof(int) * ((size_t)schol_nsuper(S->sc) + 1));
        if (S->nE) { S->Gpiv = xm(sizeof(int) * (nt + 1)); S->lGwork = (int)(64 * nt + 64); S->Gwork = xm(sizeof(double) * (size_t)S->lGwork); S->Glam = xm(sizeof(double) * (2 * nt + 1)); }
        S->ntcap = S->nE ? (int)nt : 0; S->Ecap = S->nE; }
    if (S->dasm && !S->est) {
        S->scol = xm(sizeof(int) * (S->nsc + 1));
        int k = 0; for (int j = 0; j < n; j++) if (!S->isd[j]) S->scol[k++] = j;
        S->wdc = (int)fmin((double)S->nsc, fmax(1.0, (double)(64 << 20) / (8.0 * m)));
        S->Md = xm(sizeof(double) * (size_t)m * m); S->Wd = xm(sizeof(double) * (size_t)m * S->wdc);
    }
    return 0;
}

static void kkt_solve(LpS *S, const double *g, const double *rb, double *dx, double *dy, double rtol, int maxcg);
/* the direction for the right-hand sides (rb, ru, rc, rxz, rsw):
 *   A dx = rb, dx + ds = ru, A'dy + dz - dw = rc, z dx + x dz = rxz, w ds + s dw = rsw */
static void direction(LpS *S, const double *x, const double *s, const double *z, const double *w,
                      const double *rb, const double *ru, const double *rc, const double *rxz, const double *rsw,
                      double *dx, double *dy, double *dz, double *ds, double *dw, double *g, double rtol, int maxcg) {
    const int n = S->n;
    /* g = rc - rxz / x + (rsw - w ru) / s;  dx = Theta (A'dy - g);  A Theta A' dy = rb + A Theta g */
    LPV_FOR for (int j = 0; j < n; j++) {
        double v = rc[j];
        if (!S->fr[j]) v -= rxz[j] / x[j];
        if (S->bd[j]) v += (rsw[j] - w[j] * ru[j]) / s[j];
        g[j] = v;
    }
    kkt_solve(S, g, rb, dx, dy, rtol, maxcg);
    LPV_FOR for (int j = 0; j < n; j++) {
        dz[j] = S->fr[j] ? 0.0 : (rxz[j] - z[j] * dx[j]) / x[j];
        if (S->bd[j]) { ds[j] = ru[j] - dx[j]; dw[j] = (rsw[j] - w[j] * ds[j]) / s[j]; } else { ds[j] = 0; dw[j] = 0; }
    }
}
/* dx = Theta (A'dy - g), A dx = rb, by the normal equations */
/* (4 November 2026, BRISK_LPFREEQD) the step with the free columns (and the long columns) as unknowns t of the factored matrix.
 * The system solved, with dx_s = Theta_s (A_s'dy - g_s) for the other columns and dx_b = t:
 *     [A_s Theta_s A_s' (+ delta I)  A_f  A_d      ] [dy ]   [rb + A_s Theta_s g_s]
 *     [A_f'                          0    0        ] [t_f] = [g_f                ]
 *     [A_d'                          0    -Theta_d^-1] [t_d]   [g_d                ]
 * the free columns' equations exact (no proximal term). It is factored in the equivalent form where gamma A_f times the second
 * block row is added to the first (an augmented Lagrangian: the rows get gamma A_f A_f', gamma the free columns' weight of the
 * normal equations, so that no row pivot vanishes where only free columns cover a row), with -rho on the free columns' diagonal
 * in the factor only; iterative refinement on the residuals of the system above removes rho */
/* (BRISK_LPFREEQD) the operator of the system solved: v = [vy; vt] -> [A_s Theta A_s' vy + A_b vt (+ delta vy); A_b' vy - D vt] */
static void qd_op(LpS *S, const double *v, double *out) {
    const int m = S->m, n = S->n, nb = S->bord, ex = S->qdloop;
    At_mul(S, v, S->wn);
    for (int k = 0; k < nb; k++) { const int j = S->dcol[k];
        if (!S->isd[j] && !ex) { out[m + k] = -1e30 * v[m + k]; continue; }
        const double tg = j < S->nfree && ex ? S->qtgt : 1.0 / S->theta[j];
        out[m + k] = S->wn[j] - tg * v[m + k]; }
    LPV_FOR for (int j = 0; j < n; j++) S->wn[j] = S->isd[j] ? 0.0 : S->theta[j] * S->wn[j];
    for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; if (S->isd[j] || ex) S->wn[j] += v[m + k]; }
    A_mul(S, S->wn, out);
    if (S->pdreg > 0) for (int i = 0; i < m; i++) out[i] += (S->pdd + (S->dreg ? S->dreg[i] : 0.0)) * v[i];
}
/* (4 November 2026, BRISK_LPFREEQD) the step with the free columns (and the long columns) as unknowns t of the factored matrix.
 * The system solved, with dx_s = Theta_s (A_s'dy - g_s) for the other columns and dx_b = t:
 *     [A_s Theta_s A_s' (+ delta I)  A_f  A_d        ] [dy ]   [rb + A_s Theta_s g_s]
 *     [A_f'                          0    0          ] [t_f] = [g_f                ]
 *     [A_d'                          0    -Theta_d^-1] [t_d]   [g_d                ]
 * the free columns' equations exact (no proximal term; rho on their diagonal in the factor only). Solved by the factor and
 * iterative refinement, then, when the refinement stops short, by GMRES with the factor as the preconditioner (the system is
 * indefinite: the conjugate gradients of the normal equations do not apply), to A dx = rb within 1e-3 of rb and the free
 * columns' equations within 1e-3 of their right-hand side, as the normal equations' solves. BRISK_LPFREEQDGM=0: refinement
 * alone, 1: GMRES alone; BRISK_LPFREEQDREF=k: at most k steps (30) */
static void kkt_solve_qd(LpS *S, const double *g, const double *rb, double *dx, double *dy, double rtol) {
    const int m = S->m, n = S->n, nb = S->bord, mb = m + nb;
    const double t0 = lp_time();
    static int kmax = -1; if (kmax < 0) { const char *e_ = getenv("BRISK_LPFREEQDREF"); kmax = e_ ? atoi(e_) : 30; }
    if (kmax < 2) kmax = 2;
    if (!S->qv) S->qv = xm(sizeof(double) * ((size_t)(kmax + 4) * (size_t)(mb + 1) + (size_t)(kmax + 2) * (kmax + 2) + 4 * (size_t)kmax + 16));
    double *sol = S->bw, *wk = S->bw + mb + 1;
    double *rhs = S->qv, *r = rhs + (mb + 1), *z = r + (mb + 1), *V = z + (mb + 1);
    double *H = V + (size_t)(kmax + 1) * (mb + 1), *cs = H + (size_t)(kmax + 2) * (kmax + 2), *sn = cs + kmax + 1, *e = sn + kmax + 1, *yk = e + kmax + 2;
    /* the right-hand side */
    LPV_FOR for (int j = 0; j < n; j++) S->wn[j] = S->isd[j] ? 0.0 : S->theta[j] * g[j];
    A_mul(S, S->wn, rhs);
    for (int i = 0; i < m; i++) rhs[i] += rb[i];
    for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; rhs[m + k] = !S->isd[j] && !S->qdloop ? 0.0 : g[j]; }
    const double nrb = nrminf(m, rb), ngb = nrminf(nb, rhs + m), nrhs = fmax(nrminf(mb, rhs), 1e-300);
    memcpy(sol, rhs, sizeof(double) * (size_t)mb);
    schol_solve(S->sc, sol, wk); S->nsolve++; S->nprec++;
    /* (BRISK_LPFREEQDGM=0, the default: iterative refinement with the factor, at most kmax passes while the residual halves,
     * to A dx = rb within 1e-3 of rb and the free columns' equations within 1e-3 of their right-hand side; 1: GMRES with the
     * factor as the preconditioner) */
    static int gm = -1; if (gm < 0) { const char *e_ = getenv("BRISK_LPFREEQDGM"); gm = e_ ? atoi(e_) : 2; }
    qd_op(S, sol, r);
    for (int i = 0; i < mb; i++) r[i] = rhs[i] - r[i];
    /* (GMRES only when the whole residual is above 1e-3 of the whole right-hand side; scaling the free columns' rows so that
     * one target holds each part apart was tried: cdma (LP) then ended with a row violation of 3e-5 on the problem as read) */
    /* (and not below the normal equations' own tolerance for their conjugate gradients, rtol relative to the whole right-hand
     * side: the corrector's directions, rb = 0, ran GMRES to its step limit for a target of 1e-13) */
    static int gr = -1; if (gr < 0) { const char *e_ = getenv("BRISK_LPFREEQDGRT"); gr = e_ ? atoi(e_) : 0; }   /* (BRISK_LPFREEQDGRT=1, off: with it hgms62 (LP) took 95 iterations instead of 53, atlanta-ip ended in the crossover) */
    const double tol2 = fmax(fmax(1e-3 * sqrt(nrb * nrb + ngb * ngb), 1e-13), gr ? rtol * sqrt(dot(mb, rhs, rhs)) : 0.0);
    double beta = sqrt(dot(mb, r, r));
    int kk = 0;
    double rq = 0;
    if (gm != 1) {
        rq = fmax(nrminf(m, r) / fmax(1e-3 * nrb, 1e-13), nrminf(nb, r + m) / fmax(1e-3 * ngb, 1e-13));
        for (kk = 0; kk < kmax && rq > 1.0 && isfinite(rq); kk++) {
            memcpy(z, r, sizeof(double) * (size_t)mb); schol_solve(S->sc, z, wk); S->nprec++;
            double *trial = V; for (int i = 0; i < mb; i++) trial[i] = sol[i] + z[i];
            double *r2 = V + (mb + 1); qd_op(S, trial, r2); for (int i = 0; i < mb; i++) r2[i] = rhs[i] - r2[i];
            const double rq2 = fmax(nrminf(m, r2) / fmax(1e-3 * nrb, 1e-13), nrminf(nb, r2 + m) / fmax(1e-3 * ngb, 1e-13));
            if (!(rq2 < 0.5 * rq)) break;
            memcpy(sol, trial, sizeof(double) * (size_t)mb); memcpy(r, r2, sizeof(double) * (size_t)mb); rq = rq2;
        }
        beta = sqrt(dot(mb, r, r));
    }
    /* (2: GMRES from the refined point when the refinement stopped short) */
    if (gm != 0 && (gm == 1 || rq > 1.0) && beta > tol2 && isfinite(beta)) {
        kk = 0;
        for (int i = 0; i < mb; i++) V[i] = r[i] / beta;
        for (int i = 0; i < kmax + 2; i++) e[i] = 0;
        e[0] = beta;
        for (kk = 0; kk < kmax; kk++) {
            double *vk = V + (size_t)kk * (mb + 1), *w = V + (size_t)(kk + 1) * (mb + 1);
            memcpy(z, vk, sizeof(double) * (size_t)mb); schol_solve(S->sc, z, wk); S->nprec++; S->ncg++;
            qd_op(S, z, w);
            for (int q = 0; q <= kk; q++) { const double *vq = V + (size_t)q * (mb + 1); const double h = dot(mb, w, vq); H[q + (size_t)kk * (kmax + 2)] = h; for (int i = 0; i < mb; i++) w[i] -= h * vq[i]; }
            const double hn = sqrt(dot(mb, w, w)); H[kk + 1 + (size_t)kk * (kmax + 2)] = hn;
            if (hn > 0) for (int i = 0; i < mb; i++) w[i] /= hn;
            for (int q = 0; q < kk; q++) { double *hc = H + (size_t)kk * (kmax + 2); const double a = hc[q], b = hc[q + 1]; hc[q] = cs[q] * a + sn[q] * b; hc[q + 1] = -sn[q] * a + cs[q] * b; }
            { double *hc = H + (size_t)kk * (kmax + 2); const double a = hc[kk], b = hc[kk + 1], rr = hypot(a, b);
              cs[kk] = rr > 0 ? a / rr : 1.0; sn[kk] = rr > 0 ? b / rr : 0.0; hc[kk] = rr; hc[kk + 1] = 0;
              e[kk + 1] = -sn[kk] * e[kk]; e[kk] = cs[kk] * e[kk]; }
            if (!(fabs(e[kk + 1]) > tol2) || !(hn > 0)) { kk++; break; }
        }
        /* y from the triangle, x += M^-1 V y */
        for (int q = kk - 1; q >= 0; q--) { double t = e[q]; for (int c = q + 1; c < kk; c++) t -= H[q + (size_t)c * (kmax + 2)] * yk[c]; yk[q] = H[q + (size_t)q * (kmax + 2)] != 0 ? t / H[q + (size_t)q * (kmax + 2)] : 0.0; }
        memset(z, 0, sizeof(double) * (size_t)mb);
        for (int q = 0; q < kk; q++) { const double *vq = V + (size_t)q * (mb + 1); for (int i = 0; i < mb; i++) z[i] += yk[q] * vq[i]; }
        schol_solve(S->sc, z, wk); S->nprec++;
        int ok = 1; for (int i = 0; i < mb; i++) if (!isfinite(z[i])) { ok = 0; break; }
        if (ok) for (int i = 0; i < mb; i++) sol[i] += z[i];
    }
    /* dx, and the residuals for the record */
    memcpy(dy, sol, sizeof(double) * (size_t)m);
    At_mul(S, dy, S->wn);
    LPV_FOR for (int j = 0; j < n; j++) dx[j] = S->isd[j] ? 0.0 : S->theta[j] * (S->wn[j] - g[j]);
    for (int k = 0; k < nb; k++) { const int j = S->dcol[k]; if (S->isd[j] || S->qdloop) dx[j] = sol[m + k]; }
    qd_op(S, sol, r);
    for (int i = 0; i < mb; i++) r[i] = rhs[i] - r[i];
    S->worst = nrminf(mb, r) / nrhs; S->nref += kk;
    if (getenv("BRISK_LPFREEQDDBG")) printf("       [qd solve: GMRES %d steps (from %.2e to %.2e, target %.2e); ry %.2e (rb %.2e) rt %.2e (g_b %.2e)]\n", kk, beta, sqrt(dot(mb, r, r)), tol2, nrminf(m, r), nrb, nrminf(nb, r + m), ngb);
    S->t_solve += lp_time() - t0;
}
static void kkt_solve(LpS *S, const double *g, const double *rb, double *dx, double *dy, double rtol, int maxcg) {
    if (S->qd && S->bord) { kkt_solve_qd(S, g, rb, dx, dy, rtol); return; }
    const int m = S->m, n = S->n;
    LPV_FOR for (int j = 0; j < n; j++) dx[j] = S->theta[j] * g[j];
    A_mul(S, dx, S->w1);
    for (int i = 0; i < m; i++) S->w1[i] += rb[i];
    /* the factor's solve; its residual is rb - A dx (the same vector as rhs - A Theta A'dy, and
     * dx is needed anyway: one product with A instead of two more with A and A') */
    int have_res = 0;
    {
        const double t0 = lp_time();
        const double nr = sqrt(dot(m, S->w1, S->w1));
        S->nsolve++;
        if (nr == 0) memset(dy, 0, sizeof(double) * (size_t)m); else ne_prec(S, S->w1, dy);
        At_mul(S, dy, dx);
        LPV_FOR for (int j = 0; j < n; j++) dx[j] = S->theta[j] * (dx[j] - g[j]);
        A_mul(S, dx, S->pr);
        for (int i = 0; i < m; i++) S->pr[i] = rb[i] - S->pr[i];
        if (S->pdreg > 0) for (int i = 0; i < m; i++) S->pr[i] -= (S->pdd + (S->dreg ? S->dreg[i] : 0.0)) * dy[i];
        const double rn = sqrt(dot(m, S->pr, S->pr));
        have_res = 1;
        if (nr > 0 && rn > rtol * nr && maxcg > 0) {
            ne_cg(S, dy, nr, rn, rtol, maxcg);
            At_mul(S, dy, dx);
            LPV_FOR for (int j = 0; j < n; j++) dx[j] = S->theta[j] * (dx[j] - g[j]);
            have_res = 0;
        }
        S->t_solve += lp_time() - t0;
    }
    /* A dx = rb to the accuracy the step needs (the right-hand side rb + A Theta g loses it by
     * cancellation when Theta spreads): correct dy on the residual */
    { const double nrb = nrminf(m, rb);
      for (int pass = 0; pass < 3; pass++) {
          if (have_res) { memcpy(S->w1, S->pr, sizeof(double) * (size_t)m); have_res = 0; }
          else { A_mul(S, dx, S->w1); for (int i = 0; i < m; i++) S->w1[i] = rb[i] - S->w1[i]; if (S->pdreg > 0) for (int i = 0; i < m; i++) S->w1[i] -= (S->pdd + (S->dreg ? S->dreg[i] : 0.0)) * dy[i]; }
          const double nr = nrminf(m, S->w1);
          static double reffl = -1; if (reffl < 0) { const char *e_ = getenv("BRISK_LPREFFL"); reffl = e_ ? atof(e_) : 1e-13; }   /* (2 November 2026, an experiment: the floor of the refinement) */
          if (!(nr > fmax(1e-3 * nrb, reffl)) || !isfinite(nr)) break;
          double *ddy = S->w4;
          ne_solve(S, S->w1, ddy, rtol, maxcg);
          At_mul(S, ddy, S->wn);
          double nn = 0;
          LPV_FOR for (int j = 0; j < n; j++) { S->wn[j] *= S->theta[j]; }
          A_mul(S, S->wn, S->acc);
          if (S->pdreg > 0) for (int i = 0; i < m; i++) S->acc[i] += (S->pdd + (S->dreg ? S->dreg[i] : 0.0)) * ddy[i];
          for (int i = 0; i < m; i++) { const double v = fabs(S->w1[i] - S->acc[i]); if (v > nn) nn = v; S->acc[i] = 0; }
          if (!(nn < 0.5 * nr)) break;                    /* (no gain: the factorization is at its limit) */
          LPV_FOR for (int j = 0; j < n; j++) dx[j] += S->wn[j];
          for (int i = 0; i < m; i++) dy[i] += ddy[i];
          S->nref++;
      } }
}
static void step_max(const LpS *S, const double *x, const double *s, const double *z, const double *w,
                     const double *dx, const double *ds, const double *dz, const double *dw, double *ap, double *ad) {
    double a = 1e300, d = 1e300;
    #pragma omp parallel for schedule(static) reduction(min:a, d) if (lpv_par(S))
    for (int j = 0; j < S->n; j++) {
        if (S->fr[j]) continue;
        if (dx[j] < 0) { const double t = -x[j] / dx[j]; if (t < a) a = t; }
        if (dz[j] < 0) { const double t = -z[j] / dz[j]; if (t < d) d = t; }
        if (S->bd[j]) {
            if (ds[j] < 0) { const double t = -s[j] / ds[j]; if (t < a) a = t; }
            if (dw[j] < 0) { const double t = -w[j] / dw[j]; if (t < d) d = t; }
        }
    }
    *ap = a; *ad = d;
}

/* 19 October 2026 (LP/SOCP): the arc search's step. The largest a in (0, pi/2] with v + sin(a) d1 + (1 - cos(a)) d2 > 0:
 * the first root of v + d2 + R sin(a - phi), R sin(phi) = d2, R cos(phi) = d1; 10 when the arc stays inside */
static double arc_max1(double v, double d1, double d2) {
    if (v + fmin(d1, 0.0) + fmin(d2, 0.0) > 0) return 10.0;      /* (sin a and 1 - cos a lie in [0, 1]) */
    const double R = hypot(d1, d2);
    if (!(R > 0)) return 10.0;
    const double c = -(v + d2) / R;
    if (c < -1.0 || c > 1.0) return 10.0;
    const double ph = atan2(d2, d1), as = asin(c), tw = 2.0 * M_PI;
    double t1 = fmod(ph + as, tw), t2 = fmod(ph + M_PI - as, tw), r = 10.0;
    if (t1 < 0) t1 += tw;
    if (t2 < 0) t2 += tw;
    if (t1 > 0 && t1 < r) r = t1;
    if (t2 > 0 && t2 < r) r = t2;
    return r;
}
static void arc_max(const LpS *S, const double *x, const double *s, const double *z, const double *w,
                    const double *dx1, const double *ds1, const double *dz1, const double *dw1,
                    const double *dx2, const double *ds2, const double *dz2, const double *dw2, double *ap, double *ad) {
    /* a component that stays positive up to the smallest angle found so far cannot lower it (sin a and 1 - cos a grow
     * on [0, pi/2]): only the others are solved for their root */
    double a = 10.0, d = 10.0, sa = 1.0, ca = 1.0, sd = 1.0, cd = 1.0;
#define ARC_P(v, d1, d2) if ((v) + fmin(d1, 0.0) * sa + fmin(d2, 0.0) * ca <= 0) { const double r_ = arc_max1(v, d1, d2); if (r_ < a) { a = r_; sa = sin(fmin(a, M_PI / 2)); ca = 1.0 - cos(fmin(a, M_PI / 2)); } }
#define ARC_D(v, d1, d2) if ((v) + fmin(d1, 0.0) * sd + fmin(d2, 0.0) * cd <= 0) { const double r_ = arc_max1(v, d1, d2); if (r_ < d) { d = r_; sd = sin(fmin(d, M_PI / 2)); cd = 1.0 - cos(fmin(d, M_PI / 2)); } }
    for (int j = 0; j < S->n; j++) {
        if (S->fr[j]) continue;
        ARC_P(x[j], dx1[j], dx2[j]); ARC_D(z[j], dz1[j], dz2[j]);
        if (S->bd[j]) { ARC_P(s[j], ds1[j], ds2[j]); ARC_D(w[j], dw1[j], dw2[j]); }
    }
#undef ARC_P
#undef ARC_D
    *ap = fmin(a, M_PI / 2); *ad = fmin(d, M_PI / 2);
}

/* (3 November 2026) a diverging iterate as a certificate of dual infeasibility. When x grows without bound its direction
 * d = x / |x| (in the scaled problem: min c'x, A x = b, 0 <= x <= u, the first nfree columns free) is checked as a ray:
 * d = 0 on the bounded columns, d >= 0 on the others but the free ones, A d = 0 to 1e-9 of each row's terms, and c'd < 0.
 * Such a d proves that no y has c - A'y >= 0 on the columns' signs (y'A d = 0 > c'd), so the problem is dual infeasible
 * (unbounded when it has a feasible point). Before, the run went on to the SOCP solver's homogeneous model for the same
 * verdict: neos-3230376-yser (LP) diverged at iteration 4 (9.5 s) and the SOCP solver took 9 s more, agout 1.3 s.
 * Returns 1 with the ray in d (the scaled problem's units), 0 when the direction is no ray. BRISK_LPRAY=0: off */
static int lp_ray_check(const LpS *S, int n, int m, const double *x, double *d) {
    { static int on = -1; if (on < 0) { const char *e = getenv("BRISK_LPRAY"); on = !(e && atoi(e) == 0); } if (!on) return 0; }
    double xm = 0;
    for (int j = 0; j < n; j++) if (!S->bd[j] && fabs(x[j]) > xm) xm = fabs(x[j]);
    if (!(xm > 0) || !isfinite(xm)) return 0;
    for (int j = 0; j < n; j++) {
        double v = S->bd[j] ? 0.0 : x[j] / xm;
        if (!S->fr[j] && v < 0) v = 0;
        if (fabs(v) < 1e-12) v = 0;
        d[j] = v;
    }
    double cd = 0, cab = 0;
    for (int j = 0; j < n; j++) { cd += S->c[j] * d[j]; cab += fabs(S->c[j] * d[j]); }
    if (!(cd < -1e-9) || !(cd < -1e-6 * cab)) return 0;
    double *r = calloc((size_t)m + 1, sizeof(double)), *ra = calloc((size_t)m + 1, sizeof(double));
    if (!r || !ra) { free(r); free(ra); return 0; }
    for (int j = 0; j < n; j++) if (d[j] != 0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { r[S->Ai[p]] += S->Ax[p] * d[j]; ra[S->Ai[p]] += fabs(S->Ax[p] * d[j]); }
    int ok = 1;
    for (int i = 0; i < m && ok; i++) if (fabs(r[i]) > 1e-9 * (1.0 + ra[i])) ok = 0;
    free(r); free(ra);
    return ok;
}
/* ---- the scaling of A: Ruiz equilibration (passes bringing the largest |entry| of every column and row to 1), alone or after
 * geometric-mean scaling (passes dividing each column, then each row, by the geometric mean of its largest and smallest
 * |entry|, which narrows the spread of the magnitudes; the equilibration alone leaves the small entries where they are).
 * rs, cs: the factors, multiplied in (A_scaled = diag(rs) A diag(cs)). */
static void lp_scale_ruiz(int m, int n, const int *Ap, const int *Ai, double *Ax, double *rs, double *cs) {
    double *rmax = xm(sizeof(double) * (m + 1));
    for (int pass = 0; pass < 10; pass++) {
        for (int i = 0; i < m; i++) rmax[i] = 0;
        double dev = 0;
        for (int j = 0; j < n; j++) {
            double cmx = 0;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const double v = fabs(Ax[p]); if (v > cmx) cmx = v; if (v > rmax[Ai[p]]) rmax[Ai[p]] = v; }
            if (cmx > 0) { const double f = 1.0 / sqrt(cmx); dev = fmax(dev, fabs(1.0 - cmx)); cs[j] *= f; for (int p = Ap[j]; p < Ap[j + 1]; p++) Ax[p] *= f; }
        }
        for (int i = 0; i < m; i++) rmax[i] = 0;
        for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const double v = fabs(Ax[p]); if (v > rmax[Ai[p]]) rmax[Ai[p]] = v; }
        for (int i = 0; i < m; i++) if (rmax[i] > 0) { dev = fmax(dev, fabs(1.0 - rmax[i])); rmax[i] = 1.0 / sqrt(rmax[i]); rs[i] *= rmax[i]; } else rmax[i] = 1.0;
        for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) Ax[p] *= rmax[Ai[p]];
        if (dev < 1e-2) break;
    }
    free(rmax);
}
static void lp_scale_geo(int m, int n, const int *Ap, const int *Ai, double *Ax, double *rs, double *cs) {
    double *rmx = xm(sizeof(double) * (m + 1)), *rmn = xm(sizeof(double) * (m + 1));
    double prev = HUGE_VAL;
    for (int pass = 0; pass < 8; pass++) {
        double spread = 1.0;
        for (int j = 0; j < n; j++) {
            double mx = 0, mn = HUGE_VAL;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const double v = fabs(Ax[p]); if (v == 0) continue; if (v > mx) mx = v; if (v < mn) mn = v; }
            if (mx == 0) continue;
            if (mx / mn > spread) spread = mx / mn;
            const double f = 1.0 / sqrt(mx * mn);
            cs[j] *= f; for (int p = Ap[j]; p < Ap[j + 1]; p++) Ax[p] *= f;
        }
        if (pass > 0 && !(spread < 0.9 * prev)) break;     /* (the columns' spread no longer falls by a tenth a pass) */
        prev = spread;
        for (int i = 0; i < m; i++) { rmx[i] = 0; rmn[i] = HUGE_VAL; }
        for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const double v = fabs(Ax[p]); if (v == 0) continue; const int i = Ai[p]; if (v > rmx[i]) rmx[i] = v; if (v < rmn[i]) rmn[i] = v; }
        for (int i = 0; i < m; i++) rmx[i] = rmx[i] > 0 ? 1.0 / sqrt(rmx[i] * rmn[i]) : 1.0;
        for (int i = 0; i < m; i++) rs[i] *= rmx[i];
        for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) Ax[p] *= rmx[Ai[p]];
    }
    free(rmx); free(rmn);
}
static double lp_range(int nz, const double *Ax) {    /* largest over smallest nonzero |entry| */
    double mx = 0, mn = HUGE_VAL;
    for (int p = 0; p < nz; p++) { const double v = fabs(Ax[p]); if (v == 0) continue; if (v > mx) mx = v; if (v < mn) mn = v; }
    return mx > 0 ? mx / mn : 1.0;
}
/* (4 November 2026) the range of A after the equilibration alone over the range after the geometric-mean scaling and it (on
 * copies): whether the other scaling would change the problem the method sees */
static double lp_geo_gain(int m, int n, const int *Ap, const int *Ai, const double *Ax) {
    const int nz = Ap[n]; if (nz == 0) return 1.0;
    double *A1 = xm(sizeof(double) * ((size_t)nz + 1)), *A2 = xm(sizeof(double) * ((size_t)nz + 1));
    double *r = xm(sizeof(double) * (m + 1)), *c = xm(sizeof(double) * (n + 1));
    memcpy(A1, Ax, sizeof(double) * (size_t)nz); memcpy(A2, Ax, sizeof(double) * (size_t)nz);
    for (int i = 0; i < m; i++) r[i] = 1.0;
    for (int j = 0; j < n; j++) c[j] = 1.0;
    lp_scale_ruiz(m, n, Ap, Ai, A1, r, c);
    const double R1 = lp_range(nz, A1);
    for (int i = 0; i < m; i++) r[i] = 1.0;
    for (int j = 0; j < n; j++) c[j] = 1.0;
    lp_scale_geo(m, n, Ap, Ai, A2, r, c); lp_scale_ruiz(m, n, Ap, Ai, A2, r, c);
    const double R2 = lp_range(nz, A2);
    free(A1); free(A2); free(r); free(c);
    return R2 > 0 ? R1 / R2 : 1.0;
}
/* (3 November 2026) the equilibration alone, unless the geometric-mean scaling before it leaves a scaled matrix whose range
 * (largest over smallest |entry|) is at least ten times smaller. On badly scaled data the equilibration alone keeps a wide
 * range and the linear solves lose their accuracy near the optimum: ofi (LP, entries from 1e-8 to 1e11) has a range of
 * 7.8e7 after it and 77 after both; the method needed 151 iterations on Gurobi's presolved ofi, 32 now (Gurobi's barrier:
 * 240 iterations with its scaling off, 245 with its equilibration, 33 with its geometric scaling, ScaleFlag 2). Where the
 * data are evenly scaled both give about the same matrix, yet the other factors move the dense columns' accuracy: always
 * applied, the geometric scaling cost Q083_energy_a0 (LP, range 1.2) 14.9 -> 20.6 s, zeil 26.6 -> 31.1 s, rmine13 37.7 ->
 * 42.8 s (ranges 1.1e3 -> 490 and 87 -> 75 after the two). With the tenfold test alone it was taken on LPs that the
 * equilibration leaves at ranges of 2e2 to 4e3, and lost on five of them in the full pass of 4 November (gasprod2-2 0.9 ->
 * 21.9 s, its dense columns' correction then lost; ns1760995 8.7 -> 22.1 s; satellites2-60-fs, triptim7, s250r10), while
 * hgms62 (1.2e3) gained 11.5 -> 7.8 s: at such ranges a change of scaling is a perturbation, good or bad. So it is taken
 * only beyond a range of 1e6 after the equilibration: the normal equations hold products of two entries of A, so the range
 * of A squared, and from 1e12 on the Cholesky factor in double precision has no digits to spare once the iterates spread
 * their weights (ofi: 7.8e7). BRISK_LPGEOSCALE=0: never, 2: always */
static _Thread_local int g_lp_geo_force = 0;     /* (4 November 2026) the retry of lpipm_core: the geometric-mean scaling forced */
static int lp_scale_choose(int m, int n, const int *Ap, const int *Ai, double *Ax, double *rs, double *cs, int verbose) {
    static int geo_env = -1; if (geo_env < 0) { const char *e_ = getenv("BRISK_LPGEOSCALE"); geo_env = e_ ? atoi(e_) : 1; }
    const int geo = g_lp_geo_force ? 2 : geo_env;
    const int nz = Ap[n];
    for (int i = 0; i < m; i++) rs[i] = 1.0;
    for (int j = 0; j < n; j++) cs[j] = 1.0;
    if (geo <= 0 || nz == 0) { lp_scale_ruiz(m, n, Ap, Ai, Ax, rs, cs); return 0; }
    if (geo >= 2) { lp_scale_geo(m, n, Ap, Ai, Ax, rs, cs); lp_scale_ruiz(m, n, Ap, Ai, Ax, rs, cs); return 1; }
    int taken = 0;
    double *A2 = xm(sizeof(double) * ((size_t)nz + 1)), *r2 = xm(sizeof(double) * (m + 1)), *c2 = xm(sizeof(double) * (n + 1));
    memcpy(A2, Ax, sizeof(double) * (size_t)nz);
    for (int i = 0; i < m; i++) r2[i] = 1.0;
    for (int j = 0; j < n; j++) c2[j] = 1.0;
    lp_scale_ruiz(m, n, Ap, Ai, Ax, rs, cs);
    const double R1 = lp_range(nz, Ax);
    if (R1 > 1e6) {
        lp_scale_geo(m, n, Ap, Ai, A2, r2, c2); lp_scale_ruiz(m, n, Ap, Ai, A2, r2, c2);
        const double R2 = lp_range(nz, A2);
        if (verbose > 1) printf("     (scaling: the range of the scaled matrix %.1e with the equilibration alone, %.1e after the geometric-mean scaling: %s)\n",
                                R1, R2, R2 * 10.0 <= R1 ? "the latter taken" : "the former kept");
        if (R2 * 10.0 <= R1) { memcpy(Ax, A2, sizeof(double) * (size_t)nz); memcpy(rs, r2, sizeof(double) * m); memcpy(cs, c2, sizeof(double) * n); taken = 1; }
    }
    free(A2); free(r2); free(c2);
    return taken;
}
static int lpipm_core0(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *zl, double *wu);
/* BRISK_LPFREESPLIT: 0 off, 1 the free columns split from the start (an experiment), 2 (the default) after a stall */
static int fsplit_mode(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_LPFREESPLIT"); v = e_ ? atoi(e_) : 2; } return v; }
static int fsplit_on(void) { return fsplit_mode() == 2; }
/* (2 November 2026) the free columns split, x_j = x_j+ - x_j-, both nonnegative (the negated copies appended), instead of the
 * proximal term; the multipliers of a free column come back as the mean of its two parts' */
static int lpipm_fsplit(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *zl, double *wu) {
        const int n2 = n + nfree; const long nz = Ap0[n], nzf = Ap0[nfree];
        int *Ap2 = xm(sizeof(int) * ((size_t)n2 + 1)), *Ai2 = xm(sizeof(int) * ((size_t)(nz + nzf) + 1)); double *Ax2 = xm(sizeof(double) * ((size_t)(nz + nzf) + 1));
        double *c2 = xm(sizeof(double) * ((size_t)n2 + 1)), *u2 = xm(sizeof(double) * ((size_t)n2 + 1));
        memcpy(Ap2, Ap0, sizeof(int) * ((size_t)n + 1)); memcpy(Ai2, Ai0, sizeof(int) * (size_t)nz); memcpy(Ax2, Ax0, sizeof(double) * (size_t)nz);
        for (int k = 0; k < nfree; k++) { Ap2[n + k + 1] = Ap2[n + k] + (Ap0[k + 1] - Ap0[k]);
            for (int p = Ap0[k], q = Ap2[n + k]; p < Ap0[k + 1]; p++, q++) { Ai2[q] = Ai0[p]; Ax2[q] = -Ax0[p]; } }
        memcpy(c2, c0, sizeof(double) * (size_t)n); for (int k = 0; k < nfree; k++) c2[n + k] = -c0[k];
        for (int j = 0; j < n2; j++) u2[j] = j < nfree || j >= n ? 1e30 : (u0 ? u0[j] : 1e30);
        double *x2 = xz(sizeof(double) * ((size_t)n2 + 1)), *z2 = xz(sizeof(double) * ((size_t)n2 + 1));
        double *zl2 = zl ? xz(sizeof(double) * ((size_t)n2 + 1)) : NULL, *wu2 = wu ? xz(sizeof(double) * ((size_t)n2 + 1)) : NULL;
        const int st = lpipm_core0(m, n2, 0, Ap2, Ai2, Ax2, b0, c2, u2, opt, x2, yout, z2, info, zl2, wu2);
        for (int j = 0; j < n; j++) {
            xout[j] = j < nfree ? x2[j] - x2[n + j] : x2[j];
            if (zout) zout[j] = j < nfree ? 0.5 * (z2[j] - z2[n + j]) : z2[j];
            if (zl) zl[j] = j < nfree ? 0.0 : zl2[j];
            if (wu) wu[j] = j < nfree ? 0.0 : wu2[j];
        }
        free(Ap2); free(Ai2); free(Ax2); free(c2); free(u2); free(x2); free(z2); free(zl2); free(wu2);
        return st;
}
/* zl, wu (may be NULL): the multipliers of the lower and the upper bounds apart (zout is their difference) */
static int lpipm_core(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *zl, double *wu) {
    /* (2 November 2026) the free columns split when the proximal term has failed: a stall with the free columns' weight at its
     * largest, 1e12. The proximal term leaves each step a dual residual of about rho |dx_free|; where the free columns keep
     * moving it floors the residual whatever the weight (hgms62, LP, its dual form with 5,749 free y: 73 iterations to
     * "no progress", a second attempt the same, then the SOCP solver, 60.7 s; split, 87 iterations, 9.6 s in all). Split from
     * the start it cost where the method converges (55 tall LPs 354 -> 472 s: ns1760995 12 -> 55 s, the two parts growing
     * together), so only after that failure. The split's result is kept when it is optimal or more accurate than the
     * stalled one. BRISK_LPFREESPLIT=0: off; 1: split from the start (the experiment) */
    const int fsplit = fsplit_mode();
    if (fsplit == 1 && nfree > 0) return lpipm_fsplit(m, n, nfree, Ap0, Ai0, Ax0, b0, c0, u0, opt, xout, yout, zout, info, zl, wu);
    const double t0 = lp_time();
    int st = lpipm_core0(m, n, nfree, Ap0, Ai0, Ax0, b0, c0, u0, opt, xout, yout, zout, info, zl, wu);
    if (fsplit == 2 && st == 4 && info->free_stalled && !(opt->stop && *opt->stop)) {
        LpIpmOpts o2 = *opt;
        if (opt->timelimit > 0) { o2.timelimit = opt->timelimit - (lp_time() - t0); if (o2.timelimit <= 0) return st; }
        if (opt->verbose > 1) printf("     (no progress with the free columns' weight at its largest: the %d free columns split into two nonnegative parts, solving again)\n", nfree);
        double *x2 = xz(sizeof(double) * ((size_t)n + 1)), *y2 = xz(sizeof(double) * ((size_t)m + 1)), *z2 = zout ? xz(sizeof(double) * ((size_t)n + 1)) : NULL;
        double *zl2 = zl ? xz(sizeof(double) * ((size_t)n + 1)) : NULL, *wu2 = wu ? xz(sizeof(double) * ((size_t)n + 1)) : NULL;
        LpIpmInfo i2;
        const int s2 = lpipm_fsplit(m, n, nfree, Ap0, Ai0, Ax0, b0, c0, u0, &o2, x2, y2, z2, &i2, zl2, wu2);
        const double e1 = fmax(info->pinf, fmax(info->dinf, info->gap)), e2 = fmax(i2.pinf, fmax(i2.dinf, i2.gap));
        if (s2 == 0 || (isfinite(e2) && e2 < e1)) {
            const int it1 = info->iters; *info = i2; info->iters += it1; st = s2; info->free_stalled = 0;
            memcpy(xout, x2, sizeof(double) * (size_t)n); memcpy(yout, y2, sizeof(double) * (size_t)m);
            if (zout) memcpy(zout, z2, sizeof(double) * (size_t)n);
            if (zl) memcpy(zl, zl2, sizeof(double) * (size_t)n);
            if (wu) memcpy(wu, wu2, sizeof(double) * (size_t)n);
        }
        free(x2); free(y2); free(z2); free(zl2); free(wu2);
        /* (the split stalled as well: 2, and the caller's second attempt, after the standard presolve, is skipped: its free
         * columns meet the same end, as in hgms62 and momentum2, LPs, whose second attempts stalled with the weight at 1e12) */
        if (st == 4) info->free_stalled = 2;
    }
    /* (4 November 2026) a failed attempt far from the optimum (errors above a hundred times the tolerance; near it the
     * caller's crossover takes the point) solved once more with the geometric-mean scaling before the equilibration, when
     * the equilibration alone was used and the other scaling changes the matrix (its range at least halves), within twice
     * the first attempt's time (at least a second). The scaling decides some of these failures in either direction, and no
     * test on the data alone tells which (at moderate ranges it helped highschool1-aigio, LP: the interior-point method
     * stalled after 35 iterations and the SOCP solver took over, 74.7 s; with the geometric scaling 15 iterations, 12.3 s;
     * and cost gasprod2-2 and ns1760995): the retry costs only where the method has failed. The retry's result is kept when
     * it is optimal or more accurate. BRISK_LPGEORETRY=0: off */
    {   static int gr = -1; if (gr < 0) { const char *e_ = getenv("BRISK_LPGEORETRY"); gr = !(e_ && atoi(e_) == 0); }
        const double tol = opt->tol > 0 ? opt->tol : 1e-8, e1 = fmax(info->pinf, fmax(info->dinf, info->gap));
        /* (only after a stall, "no progress": a failed factorization is the form's, which the caller's second attempt after
         * the other presolve repairs: satellites2-60-fs, LP, failed so, and its second attempt takes 0.8 s, where the retry
         * here lost to the dual simplex method beside it, 1.7 -> 5.3 s) */
        if (gr && st == 4 && !g_lp_geo_force && !info->geo_used && !(opt->stop && *opt->stop) && !(e1 <= 100.0 * tol) && !strncmp(info->why, "no progress", 11)) {
            const double tspent = lp_time() - t0;
            LpIpmOpts o2 = *opt;
            double tl = fmax(2.0 * tspent, 1.0);
            if (opt->timelimit > 0) { const double left = opt->timelimit - tspent; if (left <= 0) return st; tl = fmin(tl, left); }
            o2.timelimit = tl;
            const double gain = lp_geo_gain(m, n, Ap0, Ai0, Ax0);
            if (gain >= 2.0) {
                if (opt->verbose > 1) printf("     (the attempt failed (%s): solving again with the geometric-mean scaling, which narrows the scaled range %.1f-fold, within %.1f s)\n", info->why, gain, tl);
                double *x2 = xz(sizeof(double) * ((size_t)n + 1)), *y2 = xz(sizeof(double) * ((size_t)m + 1)), *z2 = zout ? xz(sizeof(double) * ((size_t)n + 1)) : NULL;
                double *zl2 = zl ? xz(sizeof(double) * ((size_t)n + 1)) : NULL, *wu2 = wu ? xz(sizeof(double) * ((size_t)n + 1)) : NULL;
                LpIpmInfo i2;
                g_lp_geo_force = 1;
                const int s2 = lpipm_core(m, n, nfree, Ap0, Ai0, Ax0, b0, c0, u0, &o2, x2, y2, z2, &i2, zl2, wu2);
                g_lp_geo_force = 0;
                const double e2 = fmax(i2.pinf, fmax(i2.dinf, i2.gap));
                if (s2 == 0 || (s2 != 6 && isfinite(e2) && e2 < e1)) {
                    const int it1 = info->iters; *info = i2; info->iters += it1; st = s2;
                    memcpy(xout, x2, sizeof(double) * (size_t)n); memcpy(yout, y2, sizeof(double) * (size_t)m);
                    if (zout) memcpy(zout, z2, sizeof(double) * (size_t)n);
                    if (zl) memcpy(zl, zl2, sizeof(double) * (size_t)n);
                    if (wu) memcpy(wu, wu2, sizeof(double) * (size_t)n);
                } else if (opt->verbose > 1) printf("     (the retry with the geometric-mean scaling ended with status %d: the first attempt's result stands)\n", s2);
                free(x2); free(y2); free(z2); free(zl2); free(wu2);
            }
        } }
    return st;
}
static int lpipm_core0(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *zl, double *wu) {
    const double t0 = lp_time();
    const double tol = opt->tol > 0 ? opt->tol : 1e-8;
    const int maxit = opt->maxit > 0 ? opt->maxit : 200, verbose = opt->verbose;
    memset(info, 0, sizeof(*info));
    /* (31 October 2026, ISSUES 637) the errors are unknown until an iterate is measured: a method that stopped before its
     * first iteration (the factor does not fit, the analysis stopped, a first factorization that failed) reported errors
     * of 0, and the route's rule that keeps a result within 100 times the tolerance returned the starting point as
     * "solved to reduced accuracy" (Q003_lattice_equalities, LP: a row violation of 0.69 on the problem as read; the
     * owner's benchmark of 29 October); the simplex method's interior-point route read the same zeros as "near" */
    info->pinf = info->dinf = info->gap = HUGE_VAL;
    LpS S_; LpS *S = &S_; memset(S, 0, sizeof(*S));
    S->m = m; S->n = n;
    const int nz = Ap0[n];
    S->Ap = xm(sizeof(int) * (n + 1)); S->Ai = xm(sizeof(int) * (nz + 1)); S->Ax = xm(sizeof(double) * (nz + 1));
    memcpy(S->Ap, Ap0, sizeof(int) * (n + 1)); memcpy(S->Ai, Ai0, sizeof(int) * nz); memcpy(S->Ax, Ax0, sizeof(double) * nz);
    /* rows ascending within a column */
    for (int j = 0; j < n; j++) {
        const int a = S->Ap[j], e = S->Ap[j + 1];
        for (int p = a + 1; p < e; p++) { const int r = S->Ai[p]; const double v = S->Ax[p]; int q = p - 1; while (q >= a && S->Ai[q] > r) { S->Ai[q + 1] = S->Ai[q]; S->Ax[q + 1] = S->Ax[q]; q--; } S->Ai[q + 1] = r; S->Ax[q + 1] = v; }
    }
    const double tph0 = lp_time();
    /* ---- scaling: Ruiz equilibration, or geometric-mean scaling followed by it (lp_scale_choose) */
    S->rs = xm(sizeof(double) * (m + 1)); S->cs = xm(sizeof(double) * (n + 1));
    info->geo_used = lp_scale_choose(m, n, S->Ap, S->Ai, S->Ax, S->rs, S->cs, opt->verbose);
    S->b = xm(sizeof(double) * (m + 1)); S->c = xm(sizeof(double) * (n + 1)); S->u = xm(sizeof(double) * (n + 1)); S->bd = xz(n + 1); S->fr = xz(n + 1);
    for (int j = 0; j < nfree && j < n; j++) S->fr[j] = 1;
    S->nfree = nfree < n ? nfree : n;
    {   static int qd = -1; static double qr = -1, qt = -1;
        if (qd < 0) { const char *e_ = getenv("BRISK_LPFREEQD"); qd = e_ ? atoi(e_) : 0; e_ = getenv("BRISK_LPFREEQDRHO"); qr = e_ ? atof(e_) : 1e-8; e_ = getenv("BRISK_LPFREEQDTGT"); qt = e_ ? atof(e_) : 0.0; }
        S->qd = qd == 1 && S->nfree > 0; S->qrho = qr; S->qtgt = qt; }
    for (int i = 0; i < m; i++) S->b[i] = b0[i] * S->rs[i];
    int nbd = 0;
    for (int j = 0; j < n; j++) { S->c[j] = c0[j] * S->cs[j]; S->bd[j] = !S->fr[j] && u0 && u0[j] < 1e29; S->u[j] = S->bd[j] ? u0[j] / S->cs[j] : 0.0; nbd += S->bd[j]; }
    /* scalar scaling of b (with u) and c: the iterates are then O(1) */
    double bsc = fmax(nrminf(m, S->b), 1e-300), csc = fmax(nrminf(n, S->c), 1e-300);
    { double um = 0; for (int j = 0; j < n; j++) if (S->bd[j] && S->u[j] > um) um = S->u[j]; if (um > bsc) bsc = um; }
    if (!(bsc > 1e-12)) bsc = 1.0;
    if (!(csc > 1e-12)) csc = 1.0;
    for (int i = 0; i < m; i++) S->b[i] /= bsc;
    for (int j = 0; j < n; j++) { S->c[j] /= csc; S->u[j] /= bsc; }
    const double tph1 = lp_time();
    {   /* the row copy of A for the threaded A x (entries of a row in ascending column order) */
        const int nzz = S->Ap[n];
        if (nzz >= 100000) {
            S->Fp = xz(sizeof(int) * (m + 2)); S->Fj = xm(sizeof(int) * (nzz + 1)); S->Fx = xm(sizeof(double) * (nzz + 1));
            for (int p = 0; p < nzz; p++) S->Fp[S->Ai[p] + 2]++;
            for (int i = 0; i < m; i++) S->Fp[i + 2] += S->Fp[i + 1];
            for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = S->Fp[S->Ai[p] + 1]++; S->Fj[q] = j; S->Fx[q] = S->Ax[p]; }
        }
    }
    /* the normal equations (free variables enter with the weight of a proximal term) */
    if (!getenv("BRISK_LPNODENSE")) S->dthr = opt->dthr > 0 ? opt->dthr : opt->dthr < 0 ? 0.0 : ne_dense_choice(m, n, S->Ap, S->Ai, S->Ax, verbose, 0.0, NULL);
    if (ne_setup(S, getenv("BRISK_LPNODENSE") ? 0 : 1)) {
        const int st = ne_expired() ? 6 : 4;          /* (31 October 2026: stopped or past the deadline during the analysis) */
        lps_free(S); info->status = st; snprintf(info->why, sizeof info->why, st == 6 ? "stopped during the analysis" : "the factor of the normal equations does not fit"); return st;
    }
    SChol *F = S->sc;
    info->lnz = (long)schol_nnz(F); info->ndense = S->nd;
    /* (28 October 2026) stopped during the analysis (the simplex method beside it has won): no starting point. (31 October
     * 2026: also past the deadline; the analysis of degenerate_cover scale 4 (LP) ended at 5 s with a limit of 2.6 s, and
     * the first factorization began) */
    if (ne_expired()) {
        if (getenv("BRISK_LPDBG")) printf("   [LP: stopped after the analysis: copy %.3fs, scaling %.3fs, pattern and analysis %.3fs]\n", tph0 - t0, tph1 - tph0, lp_time() - tph1);
        lps_free(S); info->status = 6; snprintf(info->why, sizeof info->why, "stopped"); return 6;
    }
    if (opt->analyzed) {
        if (opt->analyzed(opt->analyzed_ctx, ne_iter_cost_of(S))) { lps_free(S); info->status = 6; snprintf(info->why, sizeof info->why, "declined"); return 6; }
    }
    S->theta = xm(sizeof(double) * (n + 1));
    S->w1 = xm(sizeof(double) * (m + 1)); S->w2 = xm(sizeof(double) * (m + 1)); S->w4 = xm(sizeof(double) * (m + 1)); S->w3 = xm(sizeof(double) * (2 * (size_t)m + 2)); S->wn = xm(sizeof(double) * (n + 1)); S->acc = xz(sizeof(double) * (m + 1));
    S->pr = xm(sizeof(double) * (m + 1)); S->pz = xm(sizeof(double) * (m + 1)); S->pp = xm(sizeof(double) * (m + 1)); S->pq = xm(sizeof(double) * (m + 1));
    S->reg = getenv("BRISK_LPREG") ? atof(getenv("BRISK_LPREG")) : 1e-14;
    S->pdreg = getenv("BRISK_LPPDREG") ? atof(getenv("BRISK_LPPDREG")) : 0.0;
    /* (2 November 2026, BRISK_LPBORDER) the bordered quasidefinite matrix with the primal-dual regularization from the start
     * (1e-12): its factorization in a fixed order is stable for a regularized matrix (Saunders; Altman and Gondzio). Without
     * it Q083_energy_a0 (LP) lost the accuracy at its last iteration and joined 37 columns (33.4 against 22.3 s) */
    {   static int lpb = -1; if (lpb < 0) { const char *e = getenv("BRISK_LPBORDER"); lpb = e ? atoi(e) : 2; }
        if (S->bord && (lpb == 1 || lpb == 3) && !getenv("BRISK_LPPDREG") && !opt->nopdreg) S->pdreg = 1e-12;
        if (S->pdreg > 0) S->reg = S->pdd = S->pdreg;
        /* (BRISK_LPBORDER=3, an experiment) the values of the regularization switched on after a loss (rho 1e-12 in Theta,
         * delta 1e-8 in A Theta A' + delta I), from the first iteration */
        /* (BRISK_LPBORDER=4, an experiment) rho = delta = 1e-8 from the first iteration: Theta capped at 1e8, the bordered
         * matrix kept well conditioned (Q083_energy_a0, LP: 22.8 -> 19.0 s, CG steps 282 -> 44) */
        if (S->bord && lpb == 4 && !getenv("BRISK_LPPDREG") && !opt->nopdreg) { const char *e_ = getenv("BRISK_LPBREG"); S->pdreg = S->pdd = S->reg = e_ ? atof(e_) : 1e-8; }
        if (S->bord && lpb == 3 && !getenv("BRISK_LPPDREG") && !opt->nopdreg) { const char *e_ = getenv("BRISK_LPPDDUAL"); S->pdd = e_ ? atof(e_) : 1e-8; S->reg = fmax(S->pdd, 1e-14); } }
    S->pdrel = getenv("BRISK_LPPDREL") ? atof(getenv("BRISK_LPPDREL")) : 0.0;
    if (S->pdrel > 0) S->dreg = xz(sizeof(double) * (size_t)(m + 1));
    /* (22 October 2026, the night; BRISK_LPPDAUTO=delta, an experiment, off) the regularization switched on when the
     * normal equations show that they have lost rank: the first factorization with replaced pivots, a failed one, or
     * the dense columns' correction losing its accuracy; the solves that never break down are as before */
    /* (22 October 2026, 07:00: on by default, rho 1e-12, delta 1e-8, the window up to a gap of 1e-2; judged on Netlib,
     * Kennington, the adversarial LPs, the 174 MIPLIB relaxations that Gurobi's barrier solves in under a second and
     * eleven with breakdowns: the same statuses and objectives, rail01 97.7 -> 30.6 s, splice1k1 90.2 -> 42.6 s, uccase9
     * 21.1 -> 15.1 s, momentum1, rocII-5-11 and neos-4722843-widden without the SOCP solver; BRISK_LPPDAUTO=0: off) */
    const double pdauto = opt->nopdreg ? 0.0 : getenv("BRISK_LPPDAUTO") ? atof(getenv("BRISK_LPPDAUTO")) : 1e-12;
    const double pddual = getenv("BRISK_LPPDDUAL") ? atof(getenv("BRISK_LPPDDUAL")) : 1e-8;
    const double pdgap = getenv("BRISK_LPPDGAP") ? atof(getenv("BRISK_LPPDGAP")) : 1e-2;
    const double pdres = getenv("BRISK_LPPDRES") ? atof(getenv("BRISK_LPPDRES")) : 0.0;
    const int pdfb = getenv("BRISK_LPPDFB") && atoi(getenv("BRISK_LPPDFB")) == 1;
    S->tinyth = getenv("BRISK_LPTINY") ? atof(getenv("BRISK_LPTINY")) : 1e-16;
    schol_set_solve_seq(1);
    /* the number of centrality correctors: from the cost of a factorization against a solve */
    int kcorr;
    { const double ff = schol_flops(F) + (double)S->nd * 2.0 * (double)schol_nnz(F), fs = 4.0 * (double)schol_nnz(F) + 4.0 * nz;
      const double ratio = ff / fmax(fs, 1.0);
      /* (24 October 2026, the night: 3 above a flop ratio of 150 and none below was tried: dfl001 and qap15 gained a fifth in
       * the method, but s250r10 (LP) 9.2 -> 47.1 s and irish-electricity 14.4 -> 20.2 s in all, their points worse for the
       * crossover; TRICKS 8) */
      double t3 = 50, t2 = 30, t1 = 10;
      if (getenv("BRISK_LPCORRT")) sscanf(getenv("BRISK_LPCORRT"), "%lf,%lf,%lf", &t3, &t2, &t1);
      kcorr = ratio > t3 ? 3 : ratio > t2 ? 2 : ratio > t1 ? 1 : 0;
      if (opt->kcorr >= 0) kcorr = opt->kcorr; }
    if (getenv("BRISK_LPDBG")) fprintf(stderr, "   [dense correction share: %d dense columns, their solves %.2e flops against the factorization's %.2e (ratio %.2f), m %d]\n",
                                       S->nd, (double)(S->nd + S->nE) * 2.0 * schol_lsolve_flops(F), schol_flops(F), schol_flops(F) > 0 ? (double)(S->nd + S->nE) * 2.0 * schol_lsolve_flops(F) / schol_flops(F) : 0.0, m);
    if (verbose > 0)
        printf("BRISK LP interior-point solver: %d rows, %d columns (%d with upper bounds), %d nonzeros; normal equations: nnz(L) = %ld, %.2e flops, %d dense column%s; %d corrector%s\n",
               m, n, nbd, nz, info->lnz, schol_flops(F), S->nd, S->nd == 1 ? "" : "s", kcorr, kcorr == 1 ? "" : "s");
#ifdef _OPENMP
    if (verbose > 0) { const int nt = omp_in_parallel() ? 1 : omp_get_max_threads(); printf("Number of threads: %d%s\n", nt, brisk_threads_label(nt)); }   /* (31 October 2026, ISSUES 727: the owner's wording, the reason only when the user forced the count) */
#else
    if (verbose > 0) printf("Number of threads: 1%s\n", brisk_threads_label(1));
#endif
    double *x = xz(sizeof(double) * (n + 1)), *s = xz(sizeof(double) * (n + 1)), *z = xz(sizeof(double) * (n + 1)), *w = xz(sizeof(double) * (n + 1)), *y = xz(sizeof(double) * (m + 1));
    double *dx = xz(sizeof(double) * (n + 1)), *ds = xz(sizeof(double) * (n + 1)), *dz = xz(sizeof(double) * (n + 1)), *dw = xz(sizeof(double) * (n + 1)), *dy = xz(sizeof(double) * (m + 1));
    double *dx2 = xz(sizeof(double) * (n + 1)), *ds2 = xz(sizeof(double) * (n + 1)), *dz2 = xz(sizeof(double) * (n + 1)), *dw2 = xz(sizeof(double) * (n + 1)), *dy2 = xz(sizeof(double) * (m + 1));
    double *rb = xm(sizeof(double) * (m + 1)), *ru = xz(sizeof(double) * (n + 1)), *rc = xm(sizeof(double) * (n + 1)), *rxz = xm(sizeof(double) * (n + 1)), *rsw = xz(sizeof(double) * (n + 1));
    double *g = xm(sizeof(double) * (n + 1)), *zr = xz(sizeof(double) * (n + 1)), *zm = xz(sizeof(double) * (m + 1));
    int status = 3, it = 0;
    const double tph2 = lp_time(); double tph3 = tph2;
    /* ---- starting point (Mehrotra's, with the bounds): least-norm x and least-squares y */
    for (int j = 0; j < n; j++) S->theta[j] = S->bd[j] ? 0.5 : 1.0;      /* (free columns: 1, the least-norm start) */
    /* (23 October 2026) the best iterate (the least of the three errors), returned when the method stops without reaching
     * its tolerance: on cygnet (LP) the normal equations broke down at errors of 4e-8, the steps went to zero and 23 more
     * iterations left a primal infeasibility of 3.3e-4, which is what the method returned */
    double bsave = 1e300, blast = 1e300, bv[6] = { 0 }; double *xb = NULL, *sb = NULL, *zb = NULL, *wb = NULL, *yb = NULL;
    if (ne_factor(S)) {
        if (S->fstop) { status = 6; snprintf(info->why, sizeof info->why, "stopped during the first factorization"); goto done; }
        status = 4; snprintf(info->why, sizeof info->why, "the first factorization failed"); goto done;
    }
    {
        for (int j = 0; j < n; j++) x[j] = S->bd[j] ? 0.5 * S->u[j] : 0.0;
        A_mul(S, x, rb);
        for (int i = 0; i < m; i++) rb[i] = S->b[i] - rb[i];
        kkt_solve(S, zr, rb, dx, dy, 1e-10, 20);                 /* the least-norm correction: A dx = b - A x */
        for (int j = 0; j < n; j++) { x[j] += dx[j]; s[j] = S->bd[j] ? S->u[j] - x[j] : 0.0; }
        kkt_solve(S, S->c, zm, dx, y, 1e-10, 20);                /* A Theta (A'y - c) = 0 */
        for (int j = 0; j < n; j++) { const double r = -dx[j] / S->theta[j]; if (S->fr[j]) { z[j] = 0; w[j] = 0; } else if (S->bd[j]) { z[j] = 0.5 * r; w[j] = -0.5 * r; } else { z[j] = r; w[j] = 0; } }
        if (getenv("BRISK_LPSTARTY0")) {     /* (test, 21 October 2026: y = 0, z = c before the shifts) */
            for (int i = 0; i < m; i++) y[i] = 0;
            for (int j = 0; j < n; j++) { const double r = S->c[j]; if (S->fr[j]) { z[j] = 0; w[j] = 0; } else if (S->bd[j]) { z[j] = r > 0 ? r : 0; w[j] = r < 0 ? -r : 0; } else { z[j] = r; w[j] = 0; } }
        }
        /* (test, 21 October 2026, the night: BRISK_LPSTARTBOX=theta, a bounded column kept in [theta u, (1 - theta) u] with
         * s = u - x, and left out of the shifts of x, which a single far-off column otherwise makes large for all) */
        static double tbox = -1; if (tbox < 0) { const char *e_ = getenv("BRISK_LPSTARTBOX"); tbox = e_ ? atof(e_) : 0.0; }
        if (tbox > 0) for (int j = 0; j < n; j++) if (S->bd[j] && !S->fr[j]) { x[j] = fmin(fmax(x[j], tbox * S->u[j]), (1.0 - tbox) * S->u[j]); s[j] = S->u[j] - x[j]; }
        double xmin = 1e300, zmin = 1e300;
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; if (!(tbox > 0 && S->bd[j])) xmin = fmin(xmin, x[j]); zmin = fmin(zmin, z[j]); if (S->bd[j]) { if (!(tbox > 0)) xmin = fmin(xmin, s[j]); zmin = fmin(zmin, w[j]); } }
        if (xmin > 1e299) xmin = 0;
        const double dxs = fmax(-1.5 * xmin, 0.0), dzs = fmax(-1.5 * zmin, 0.0);
        double xz_ = 0, sx = 0, sz = 0;
        for (int j = 0; j < n; j++) {
            if (S->fr[j]) continue;
            const double dxj = tbox > 0 && S->bd[j] ? 0.0 : dxs;
            x[j] += dxj; z[j] += dzs; xz_ += x[j] * z[j]; sx += x[j]; sz += z[j];
            if (S->bd[j]) { s[j] += dxj; w[j] += dzs; xz_ += s[j] * w[j]; sx += s[j]; sz += w[j]; }
        }
        double dx3 = sz > 0 ? 0.5 * xz_ / sz : 0.0, dz3 = sx > 0 ? 0.5 * xz_ / sx : 0.0;
        if (!(dx3 > 1e-8)) dx3 = fmax(dx3, 1e-2);         /* (a start at zero: push inside) */
        if (!(dz3 > 1e-8)) dz3 = fmax(dz3, 1e-2);
        if (getenv("BRISK_LPSTART")) { const double f = atof(getenv("BRISK_LPSTART")); dx3 *= f; dz3 *= f; }
        if (getenv("BRISK_LPSTARTXZ")) { double fx = 1, fz = 1; sscanf(getenv("BRISK_LPSTARTXZ"), "%lf,%lf", &fx, &fz); dx3 *= fx; dz3 *= fz; }   /* (test, 21 October 2026: the shifts of x and z apart) */
        if (getenv("BRISK_LPDBG")) printf("   [start: shifts %.3e %.3e then %.3e %.3e; mean x %.3e, mean z %.3e]\n", dxs, dzs, dx3, dz3, sx / fmax(1, n), sz / fmax(1, n));
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; const double d3 = tbox > 0 && S->bd[j] ? 0.0 : dx3; x[j] += d3; z[j] += dz3; if (S->bd[j]) { s[j] += d3; w[j] += dz3; } }
        if (getenv("BRISK_LPSTART0")) {      /* (test: a flat start, x = z = 1 (bounded: x = s = u/2 at most 1), y = 0) */
            const double f = atof(getenv("BRISK_LPSTART0"));
            for (int i = 0; i < m; i++) y[i] = 0;
            for (int j = 0; j < n; j++) { if (S->fr[j]) { x[j] = 0; continue; } x[j] = S->bd[j] ? fmin(f, 0.5 * S->u[j]) : f; z[j] = f; if (S->bd[j]) { s[j] = S->u[j] - x[j]; w[j] = f; } }
        }
    }
    const double nb0 = nrminf(m, b0), nc0 = nrminf(n, c0);
    double mu0 = -1, best = 1e300, err0 = -1; int nstall = 0, nsafe = 0;
    const int steprule = getenv("BRISK_LPSTEP") ? atoi(getenv("BRISK_LPSTEP")) : 1;
    const double frth = getenv("BRISK_LPFREETH") ? atof(getenv("BRISK_LPFREETH")) : 1e6;
    double frw = frth;     /* (the free columns' weight now: raised at a stall, below) */
    if (verbose > 1) printf("  it      pobj             dobj         pinf     dinf     gap      mu      alp_p alp_d sigma  cg  solres tiny\n");
    double tab_step = 0;   /* (3 November 2026: the default log's iteration table, brisk_itab_row: the last step's length) */
    tph3 = lp_time();
    int ncomp0 = 0; for (int j = 0; j < n; j++) if (!S->fr[j]) ncomp0 += 1 + (S->bd[j] != 0);
    if (!S->vpart) S->vpart = xm(sizeof(double) * (4 * ((size_t)n / LPV_CH + 2)));     /* (2 doubles and 4 ints a chunk for the blocking search) */
    S->qdloop = 1;
    for (it = 0; ; it++) {
        /* ---- residuals (scaled problem) and the errors of the problem as given */
        A_mul(S, x, rb);
        for (int i = 0; i < m; i++) rb[i] = S->b[i] - rb[i];
        At_mul(S, y, rc);
        double mu = 0; const int ncomp = ncomp0;
        LPV_SUM(mu, rc[j] = S->c[j] - rc[j] - z[j] + w[j];
            if (S->fr[j]) continue;
            v += x[j] * z[j];
            if (S->bd[j]) { ru[j] = S->u[j] - x[j] - s[j]; v += s[j] * w[j]; });
        mu /= fmax(ncomp, 1);
        if (mu0 < 0) mu0 = mu;
        double ep = 0, ed = 0, pobj = 0, dobj = 0;
        for (int i = 0; i < m; i++) { const double v = fabs(rb[i]) * bsc / S->rs[i]; if (v > ep) ep = v; dobj += S->b[i] * y[i]; }
        if (n < LPV_MIN)
        for (int j = 0; j < n; j++) {
            const double v = fabs(rc[j]) * csc / S->cs[j]; if (v > ed) ed = v;
            pobj += S->c[j] * x[j];
            if (S->bd[j]) { const double q = fabs(ru[j]) * bsc * S->cs[j]; if (q > ep) ep = q; dobj -= S->u[j] * w[j]; }
        }
        else {
            #pragma omp parallel for schedule(static) reduction(max:ed, ep) if (lpv_par(S))
            for (int j = 0; j < n; j++) {
                const double v = fabs(rc[j]) * csc / S->cs[j]; if (v > ed) ed = v;
                if (S->bd[j]) { const double q = fabs(ru[j]) * bsc * S->cs[j]; if (q > ep) ep = q; }
            }
            LPV_SUM(pobj, v += S->c[j] * x[j]);
            double du = 0; LPV_SUM(du, if (S->bd[j]) v += S->u[j] * w[j]);
            dobj -= du;
        }
        pobj *= bsc * csc; dobj *= bsc * csc;
        /* (22 October 2026, the night) the gap against the objective as given, c'x + objoff: the standard form drops the
         * constant of the columns shifted to their bounds, and relative to the rest the gap test could be 35,000 times
         * too lenient (supportcase42 (LP): an objective of 7.34 and a standard form's of -2.6e5; its result 1e-4 off) */
        const double e_p = ep / (1.0 + nb0), e_d = ed / (1.0 + nc0), e_g = fabs(pobj - dobj) / (1.0 + fabs(pobj + opt->objoff));
        const double err = fmax(fmax(e_p, e_d), e_g);
        info->pobj = pobj; info->dobj = dobj; info->pinf = e_p; info->dinf = e_d; info->gap = e_g; info->iters = it;
        brisk_itab_row(it, pobj + opt->objoff, dobj + opt->objoff, e_p, e_d, e_g, mu, tab_step);
        if (verbose > 1 && it == 0) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e\n", it, pobj, dobj, e_p, e_d, e_g, mu);
        if (opt && opt->iter_cb) opt->iter_cb(opt->iter_ctx, it, pobj, fmax(e_p, fmax(e_d, e_g)));   /* (the race's log) */
        if (getenv("BRISK_LPDBG")) {
            int im = 0; double vm = 0; for (int i = 0; i < m; i++) { const double v = fabs(rb[i]) / S->rs[i]; if (v > vm) { vm = v; im = i; } }
            double sa = 0, xm_ = 0, ym = nrminf(m, y); for (int j = 0; j < n; j++) { xm_ = fmax(xm_, fabs(x[j])); for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (S->Ai[p] == im) sa += fabs(S->Ax[p] * x[j]); }
            printf("     [dbg: worst row %d: rb %.2e (scaled; max scaled %.2e), rs %.2e, sum|a x| %.2e, b %.2e; max|x| %.2e max|y| %.2e]\n", im, rb[im], nrminf(m, rb), S->rs[im], sa, S->b[im], xm_, ym);
        }
        if (!isfinite(pobj) || !isfinite(dobj) || !isfinite(mu) || !isfinite(ep) || !isfinite(ed)) { status = 4; snprintf(info->why, sizeof info->why, "not finite"); break; }
        if (err <= tol) { status = 0; break; }
        blast = err;
        if (err < bsave && isfinite(err)) {
            if (!xb) { xb = xm(sizeof(double) * (n + 1)); sb = xm(sizeof(double) * (n + 1)); zb = xm(sizeof(double) * (n + 1)); wb = xm(sizeof(double) * (n + 1)); yb = xm(sizeof(double) * (m + 1)); }
            memcpy(xb, x, sizeof(double) * n); memcpy(sb, s, sizeof(double) * n); memcpy(zb, z, sizeof(double) * n); memcpy(wb, w, sizeof(double) * n); memcpy(yb, y, sizeof(double) * m);
            bsave = err; bv[0] = pobj; bv[1] = dobj; bv[2] = e_p; bv[3] = e_d; bv[4] = e_g; bv[5] = it;
        }
        if (it >= maxit) { status = 3; snprintf(info->why, sizeof info->why, "iteration limit"); break; }
        if (opt->timelimit > 0 && lp_time() - t0 > opt->timelimit) { status = 6; break; }
        /* 18 October 2026 (LP/SOCP): a factorization that would end past the time limit is not started (its time from
         * the last ones, the first one's from its flops at 5e10 a second): the factorization does not look at the
         * clock, and a run with -timelimit 400 ended at 639 s */
        if (opt->timelimit > 0) {
            const double tf = S->nfact > 0 ? S->t_fact / S->nfact : schol_flops(S->sc) / 5e10;
            if (lp_time() - t0 + tf > opt->timelimit) { status = 6; snprintf(info->why, sizeof info->why, "the next factorization would end past the time limit"); break; }
        }
        if (opt->stop && *opt->stop) { status = 6; break; }
        /* divergence or stall: the caller's homogeneous model decides (infeasible or unbounded problems) */
        if (err0 < 0) err0 = err;
        if (err < 0.9 * best) { best = err; nstall = 0; } else nstall++;
        /* (24 October 2026) a stall with free columns: their weight (the proximal term's 1 / rho) a hundred times larger,
         * up to 1e12, and the stall's count restarted. The proximal term leaves each step a dual residual of about
         * rho |dx_free| on the free columns; where those keep moving it floors the residual (atlanta-ip, LP, on its dual
         * form with 6,126 free y: the gap at 6e-3 and the dual residual at 3e-8 for 25 iterations, then "no progress";
         * with the weight 1e8 from the start 39 iterations). A larger weight from the start, or one following the other
         * columns', cost elsewhere (opm2-z10-s4 13.8 -> 22.2 s, physiciansched3-3 37 -> 243 s): runs that progress keep
         * 1e6. BRISK_LPFREESTALL=0: off */
        { static int fst = -1; if (fst < 0) { const char *e_ = getenv("BRISK_LPFREESTALL"); fst = e_ ? atoi(e_) : 2; }
          /* (the same afternoon) only a stall after the error has fallen a hundredfold from its first value: in the
           * opening iterations the error can stand still while the gap grows, nothing floored (rmatr200-p5, LP: the rule
           * fired at iterations 4 and 9, 53 -> 71 iterations, the race 4.2 -> 5.6 s; with the condition 4.45 s; the 28 LPs
           * the interior-point method wins 4.397 -> 4.352 s, the same objectives); and, as a last try, a stall about to
           * end the run (20 iterations of the 25 that stop it: atlanta-ip, LP, on its dual form stopped at iteration 29
           * without it). BRISK_LPFREESTALL=1: at any stall, as before; 0: off */
          if (fst && nfree > 0 && !S->qd && nstall >= 4 && frw < 1e12 && (fst != 2 || best <= 1e-2 * err0 || nstall >= 20)) { frw = fmin(100.0 * frw, 1e12); nstall = 0;
              if (verbose > 1) printf("     (no progress with free columns: their weight %.0e from here)\n", frw); } }
        if (!isfinite(err) || !isfinite(mu)) { status = 4; snprintf(info->why, sizeof info->why, "not finite"); break; }
        /* (23 October 2026) within a hundred times the tolerance five iterations without progress are a stall: what is left
         * there is a step or two, and a breakdown of the normal equations there (cygnet: steps of zero from iteration 20)
         * does not mend itself */
        const int nst = best <= 100 * tol ? 5 : 25;
        /* (2 November 2026, BRISK_LPFREESPLITN=k) with the free columns' weight at its largest, five iterations without progress end
         * the attempt for the split (lpipm_core) instead of 25: the weight cannot be raised again, as four such iterations would
         * have raised it (hgms62, LP: 20.2 -> 16.0 s; the 55 tall LPs otherwise the same). 0: the 25 */
        { static int fsn = -1; if (fsn < 0) { const char *e_ = getenv("BRISK_LPFREESPLITN"); fsn = e_ ? atoi(e_) : 5; }
          if (fsn > 0 && nfree > 0 && !S->qd && frw >= 1e12 && fsplit_on() && nstall >= fsn && nstall < nst) { status = 4; snprintf(info->why, sizeof info->why, "no progress in %d iterations with the free columns' weight at its largest", fsn); info->free_stalled = 1; break; } }
        if (nstall >= nst || mu > 1e12 * fmax(mu0, 1.0) || nrminf(n, x) > 1e14 || nrminf(m, y) > 1e14) { status = 4; snprintf(info->why, sizeof info->why, nstall >= nst ? "no progress in %d iterations" : "the iterates diverge", nst);
            info->free_stalled = nstall >= nst && nfree > 0 && frw >= 1e12 && !S->qd;
            if (nstall < nst && nrminf(n, x) > 1e14) {
                double *d = xm(sizeof(double) * (n + 1));
                if (lp_ray_check(S, n, m, x, d)) {
                    status = 2; snprintf(info->why, sizeof info->why, "the iterates diverge along a ray: the problem is dual infeasible");
                    memcpy(x, d, sizeof(double) * n);
                }
                free(d);
            }
            break; }
        /* ---- factorization */
        double thmax = 0;
        #pragma omp parallel for schedule(static) reduction(max:thmax) if (lpv_par(S))
        for (int j = 0; j < n; j++) if (!S->fr[j]) { S->theta[j] = 1.0 / (z[j] / x[j] + (S->bd[j] ? w[j] / s[j] : 0.0) + S->pdreg); if (S->theta[j] > thmax) thmax = S->theta[j]; }
        /* free variables: a proximal term rho |dx|^2 / 2, Theta = 1 / rho (the Newton step of the regularized problem) */
        /* (4 November 2026) the free columns' weight at least a ten-thousandth of the largest weight of the other columns
         * (which grows as mu falls): the proximal term fades as the method converges, as in regularized interior-point
         * methods, instead of flooring the steps of the free columns. atlanta-ip (LP, its dual form with 6,105 free y): 103
         * -> 41 iterations of the primal form's kind, 6.07 -> 3.17 s; stockholm 2.65 -> 1.62 s; on the 123 development LPs
         * with free columns 0.976 (geometric mean), on 32 held-out ones 0.929 (cdma, LP, 2.98 -> 0.15 s). The fraction is
         * measured: the whole largest weight (1) and a hundredth cost ns1760995 (LP) 2.6 and 2.3 times, its free columns
         * then too free. BRISK_LPFREEREL=r: the fraction r; 0: the weight alone, as before */
        { static double frr = -1; if (frr < 0) { const char *e_ = getenv("BRISK_LPFREEREL"); frr = e_ ? atof(e_) : 1e-4; }
          double thf = frr > 0 ? fmin(fmax(frw, frr * thmax), 1e12) : frw;
          /* (BRISK_LPFREEQD) the free columns' weight gamma in the rows of the factored form (BRISK_LPFREEQDGAM=g; negative: the
           * weight of the proximal method above) */
          {   static double qg = -2; if (qg < -1) { const char *e_ = getenv("BRISK_LPFREEQDGAM"); qg = e_ ? atof(e_) : 0.0; }
              if (S->qd && S->bord && qg >= 0) thf = qg; }
          for (int j = 0; j < nfree; j++) S->theta[j] = thf; }
        {   int fb = ne_factor(S);
            if (fb && S->fstop) { status = 6; snprintf(info->why, sizeof info->why, "stopped during a factorization"); break; }
            /* (replaced pivots count only in the end phase, the relative gap below pdgap (BRISK_LPPDGAP, 0: always): before
             * it they come from dependent rows, which the replacement handles (trento1 from iteration 4, the gap at 1);
             * a failed factorization always) */
            /* (and, BRISK_LPPDRES, only when the previous iteration's solves were inexact, the residual above pdres after the
             * conjugate gradients: replaced pivots with accurate directions are no breakdown, trento1's 23 at its last
             * iteration) */
            /* (and not within a hundred times the tolerance, where a step or two remains and the regularized step cannot
             * close the last residual: trento1's pivots are replaced at a gap of 2.4e-8, its last iteration) */
            /* (a failed factorization switches it on only with BRISK_LPPDFB=1: var-smallemery-m6j6's first one, the dense
             * columns' Schur complement indefinite, went on regularized for 19 iterations to the same failure, 9.6 s
             * against 4.7 s through the SOCP solver at once) */
            if (pdauto > 0 && S->pdreg == 0 && ((fb && pdfb) || (!fb && schol_ntiny(S->sc) > 0 && (pdgap <= 0 || (e_g < pdgap && e_g > 100.0 * tol)) && (pdres <= 0 || S->worst > pdres)))) {
                if (verbose > 1) printf("     (the normal equations have lost rank (%s): regularized by %.0e from here)\n", fb ? "a failed factorization" : "replaced pivots", pdauto);
                S->pdreg = pdauto; S->pdd = pddual >= 0 ? pddual : pdauto; S->reg = fmax(S->pdd, 1e-14);
                for (int j = 0; j < n; j++) if (!S->fr[j]) S->theta[j] = 1.0 / (z[j] / x[j] + (S->bd[j] ? w[j] / s[j] : 0.0) + S->pdreg);
                fb = ne_factor(S);
                if (fb && S->fstop) { status = 6; snprintf(info->why, sizeof info->why, "stopped during a factorization"); break; }
            }
            if (fb) { status = 4; snprintf(info->why, sizeof info->why, "factorization failed"); break; }
        }
        /* (23 October 2026) the solves' relative residual 1e-2 times the relative gap, between 1e-10 and 1e-6: early, where the
         * gap is large, an inexact direction costs nothing (each iteration's residuals are recomputed) and saves the
         * refinement's solves (sing44, LP: 10 solves an iteration, 0.85 s of solves against 0.42 of factorizations). LPs,
         * four threads, total time: s100 24.2 -> 19.2 s, triptim1 19.4 -> 16.9, opm2-z10-s4 23.1 -> 20.4, sing44 3.2 -> 2.8,
         * dfl001 2.7 -> 2.2; irish-electricity 13.3 -> 15.2 (its crossover: 3,072 simplex iterations instead of 363), qap15
         * 6.3 -> 6.7. With 1 times the gap qap15's method ended short. BRISK_LPRTOLF=f: the factor (0: 1e-10 always) */
        static double rtolf = -1; if (rtolf < 0) { const char *e_ = getenv("BRISK_LPRTOLF"); rtolf = e_ ? atof(e_) : 1e-2; }
        const double rtol = rtolf > 0 ? fmax(1e-10, fmin(1e-6, rtolf * info->gap)) : 1e-10;
        const int maxcg = getenv("BRISK_LPCG") ? atoi(getenv("BRISK_LPCG")) : 30;
        const long cg0 = S->ncg;
        /* ---- predictor */
        LPV_FOR for (int j = 0; j < n; j++) { rxz[j] = S->fr[j] ? 0.0 : -x[j] * z[j]; rsw[j] = S->bd[j] ? -s[j] * w[j] : 0.0; }
        S->worst = 0;
        direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        /* (4 November 2026) pivots replaced as tiny and a direction whose solves fail (the residual after the conjugate
         * gradients of the order of the right-hand side or more): the pivots between 1e-16 and 1e-12 of their diagonal that
         * the factor kept carry a relative error of 1e-4 or more (a pivot's rounding is about the unit roundoff times its
         * original diagonal), so they are taken as dependent too, for the rest of the run, and the direction computed again
         * before any step. rail03 (LP, set covering, no dense columns): 47 pivots replaced at a gap of 6.2e-7, the solves'
         * residual 2e2, then 4e7 and a step that spoiled the point; the method stopped and the crossover took 55 s (260,000
         * pushes), 110 s in all; with 1e-12 it converges in 67 iterations, 46 s (Gurobi 46 s). A global 1e-12 trades classes
         * (TRICKS 8: zeil, Q083_energy_a0 slower, stockholm less accurate): here it applies only where 1e-16 failed.
         * BRISK_LPTINYUP=0: off */
        /* (only in the last stretch, the relative gap within a hundred times the tolerance: there the regularization that
         * replaced pivots switch on from a gap of 1e-2 is not used (a step or two remains), and this takes its place.
         * stockholm (LP) failed at a gap of 7e-6, inside the regularization's window, and with 1e-12 it ended at 2.7e-6
         * instead of 3.6e-7; the residual after the switch tells nothing, since the replaced directions are not solved for:
         * rail03's was 1.0 and it converged, stockholm's 0.5. Nor with free columns: their proximal weights (1e6 and more)
         * inflate the diagonals the test is relative to, so legitimate pivots look like noise; stockholm, 18,720 free
         * columns, failed at a gap of 7.3e-7 as well and ended with a row violation of 2.6e-6 instead of 3.6e-7) */
        {   static int tup = -1; if (tup < 0) { const char *e_ = getenv("BRISK_LPTINYUP"); tup = !(e_ && atoi(e_) == 0); }
            if (tup && !S->tinyup_tried && S->nd == 0 && nfree == 0 && S->tinyth < 1e-12 && e_g <= 100.0 * tol && schol_ntiny(S->sc) > 0 && S->worst > 1.0) {
                if (verbose > 1) printf("     (%d pivots replaced and the solves failed (residual %.1e) at a gap of %.1e: pivots below 1e-12 of their diagonal taken as dependent from here)\n", schol_ntiny(S->sc), S->worst, e_g);
                S->tinyup_tried = 1; S->tinyth = 1e-12;
                if (ne_factor(S)) { status = S->fstop ? 6 : 4; snprintf(info->why, sizeof info->why, status == 6 ? "stopped during a factorization" : "factorization failed"); break; }
                S->worst = 0;
                direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
            } }
        /* (21 October 2026, the night: the limit 1e-4 instead of 1e-6, BRISK_LPDENSETOL. supportcase33 stopped at iteration
         * 19 of 20 with a residual of 3.6e-6, a gap of 1.5e-4 and mu 1e-10, and the SOCP solver started again from scratch
         * (2.2 s in all); its last step from that direction ends with errors of 1e-14 (1.1 s). Each iteration's residuals
         * are recomputed, so an inexact direction costs progress, not accuracy; a correction that is lost has residuals of
         * order one (the dense-column test LPs fail at once, either limit).) */
        static double dtol = -1; if (dtol < 0) { const char *e_ = getenv("BRISK_LPDENSETOL"); dtol = e_ ? atof(e_) : 1e-4; }
        /* (... and only once the method is under way, the gap below ten times pdgap: a correction lost at the start is the
         * structure's, and regularized the method only ran on to the same failure, lp_densecols_small 2.2 -> 7.1 s;
         * rocII-5-11's came at a gap of 3.7e-2) */
        /* (31 October 2026, the night) the correction applied after the full solve, y - Z G^-1 U'y, subtracts two vectors of the
         * size of M_s^-1 r: as the sparse part becomes ill-conditioned near the end they grow and the difference loses its
         * digits, where the two sweeps with W between them (the correction in the factor's own space) keep their accuracy.
         * k1mushroom and hypothyroid-k1 (LPs, the MIPLIB 2017 relaxations): lost at iteration 21, the SOCP solver after it
         * (5.7 -> 16.5 s, 2.3 -> 3.7 s by default); with the sweeps, OPTIMAL at iteration 22. So the sweeps from here, the
         * direction again, before the regularization and the rest below */
        /* (2 November 2026, BRISK_LPBORDER) the bordered matrix has lost its accuracy: the Woodbury correction from here,
         * whose rules for a loss follow (the regularization, the weak rows taken out, the columns joined) */
        if (S->nd > 0 && S->worst > dtol && S->zok && !S->qd) {
            if (verbose > 1) printf("     (dense columns: residual %.1e with the correction after the full solve: the two sweeps from here)\n", S->worst);
            S->znot = 1;
            if (ne_factor(S)) { status = S->fstop ? 6 : 4; snprintf(info->why, sizeof info->why, status == 6 ? "stopped during a factorization" : "factorization failed"); break; }
            S->worst = 0;
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        if (S->nd > 0 && S->worst > dtol && pdauto > 0 && S->pdreg == 0 && (pdgap <= 0 || e_g < 10.0 * pdgap)) {
            if (verbose > 1) printf("     (dense columns: residual %.1e: regularized by %.0e from here)\n", S->worst, pdauto);
            S->pdreg = pdauto; S->pdd = pddual >= 0 ? pddual : pdauto; S->reg = fmax(S->pdd, 1e-14);
            for (int j = 0; j < n; j++) if (!S->fr[j]) S->theta[j] = 1.0 / (z[j] / x[j] + (S->bd[j] ? w[j] / s[j] : 0.0) + S->pdreg);
            if (ne_factor(S)) { status = S->fstop ? 6 : 4; snprintf(info->why, sizeof info->why, status == 6 ? "stopped during a factorization" : "factorization failed"); break; }
            S->worst = 0;
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        if (S->bord && S->worst > dtol && !S->bwtried && !S->qd) {
            /* first the weak rows after the border: rows where the sparse part's diagonal is below eps times the dense
             * columns' (eps 1e-6; at most 4,000 rows, whose dense block costs about 2e10 flops) */
            S->bwtried = 1;
            double *dd = xz(sizeof(double) * (size_t)m), *dsp = xz(sizeof(double) * (size_t)m);
            for (int j = 0; j < n; j++) { double *t = S->isd[j] ? dd : dsp; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) t[S->Ai[p]] += S->theta[j] * S->Ax[p] * S->Ax[p]; }
            static double bweps = -1; if (bweps < 0) { const char *e_ = getenv("BRISK_LPBWEPS"); bweps = e_ ? atof(e_) : 1e-6; }
            int c6 = 0, c8 = 0, c10 = 0, cw = 0;
            for (int i = 0; i < m; i++) { const double d = dsp[i] + S->reg; if (dd[i] <= 0) continue; if (d <= 1e-6 * dd[i]) c6++; if (d <= 1e-8 * dd[i]) c8++; if (d <= 1e-10 * dd[i]) c10++; if (d <= bweps * dd[i]) cw++; }
            if (getenv("BRISK_LPDBG")) fprintf(stderr, "   [bordered, lost: weak rows below 1e-6: %d, 1e-8: %d, 1e-10: %d; taken %d]\n", c6, c8, c10, cw);
            if (cw > 0 && cw <= 4000) {
                free(S->bwk); S->bwk = xm(sizeof(int) * (size_t)cw); S->nbw = 0;
                for (int i = 0; i < m; i++) if (dd[i] > 0 && dsp[i] + S->reg <= bweps * dd[i]) S->bwk[S->nbw++] = i;
                if (verbose > 1) printf("     (dense columns: residual %.1e with the bordered matrix: %d weak rows ordered after the dense columns)\n", S->worst, S->nbw);
                free(dd); free(dsp);
                if (ne_setup(S, 1) || !S->sc) { status = 4; snprintf(info->why, sizeof info->why, "factorization failed"); break; }
                if (ne_factor(S)) { status = S->fstop ? 6 : 4; snprintf(info->why, sizeof info->why, status == 6 ? "stopped during a factorization" : "factorization failed"); break; }
                S->worst = 0;
                direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
            } else { free(dd); free(dsp); }
        }
        if (S->bord && S->worst > dtol && !S->qd) {
            if (verbose > 1) printf("     (dense columns: residual %.1e with the bordered matrix: the Woodbury correction from here)\n", S->worst);
            S->noborder = 1;
            if (ne_setup(S, 1) || !S->sc) { status = 4; snprintf(info->why, sizeof info->why, "factorization failed"); break; }
            if (ne_factor(S)) { status = S->fstop ? 6 : 4; snprintf(info->why, sizeof info->why, status == 6 ? "stopped during a factorization" : "factorization failed"); break; }
            S->worst = 0;
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        /* (1 November 2026, ISSUES 813) after the regularization, the rows where the sparse part has become weak against the dense columns taken
         * out by unit columns of the correction (Andersen's modified Schur complement; ne_factor), for the rest of the run.
         * Only after a loss, and after the regularization (which costs nothing an iteration): taken at every factorization
         * they changed the paths of LPs that never lost the correction (shipsched 0.33 -> 2.99 s, genus-sym-g62-2 31.9 ->
         * 64.7 s), and before the regularization they cost shipsched 0.35 -> 1.74 s, where the regularization alone
         * recovers; when no row is weak the factor is the same and the direction stands */
        if (S->nd > 0 && S->worst > dtol && !S->andon && !S->qd) {
            S->andon = 1;
            if (ne_factor(S)) { status = S->fstop ? 6 : 4; snprintf(info->why, sizeof info->why, status == 6 ? "stopped during a factorization" : "factorization failed"); break; }
            if (S->nweak > 0) {
                if (verbose > 1) printf("     (dense columns: residual %.1e: %d rows where the sparse part is weak taken out of the factor from here)\n", S->worst, S->nweak);
                S->worst = 0;
                direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
            }
        }
        if (S->nd > 0 && S->worst > dtol && !S->qd) {
            /* the Woodbury correction of the dense columns has lost its accuracy: all columns in the factor, unless
             * the clique that they make over their rows costs more than ten times the factorization with them apart
             * (18 October 2026, LP/SOCP: c^3/3 flops for a clique of c rows; 20 dense columns over 10,000 rows made
             * a dense factor of order 10,000, and over 50,000 rows ran out of memory). The SOCP solver then takes the
             * problem: its augmented system keeps the dense columns apart. */
            long c = 0; { char *mk = xz(m + 1); for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (!mk[S->Ai[p]]) { mk[S->Ai[p]] = 1; c++; } } free(mk); }
            const double fs = schol_flops(S->sc) + S->nd * (4.0 * (double)schol_nnz(S->sc) + 2.0 * S->nd * m);
            double fj = (double)c * c * c / 3.0;
            /* (31 October 2026) the clique over the union of their rows is an upper bound: when it refuses, the factor with
             * every column in it is estimated (AMD's order, exact counts, against the same ten times as budget). ab71-20-100
             * (LP, folded): 40 columns over 3,197 rows, a clique of 1.1e10 flops, while the factor with them in it costs 2.1e7
             * (the choice before the exact counts took it from the start); refused, the SOCP solver started again after 33
             * iterations, 0.39 -> 0.80 s */
            /* (2 November 2026) or what the factorizations with them joined cost for ten more iterations against the time the
             * method has spent (about what the crossover or the SOCP solver's new start that follows a refusal costs), at
             * the rate the factorizations have run: zeil (LP) was refused at 1.12e10 flops against the cap's 1.12e10
             * (0.25 s a factorization) at a gap of 2e-7, and its crossover then decided the run (8 s; on a permuted copy it
             * failed). BRISK_LPJOINTIME=0: the ten times alone */
            double jcap = 10.0 * fs;
            {   static int jtime = -1; if (jtime < 0) { const char *e_ = getenv("BRISK_LPJOINTIME"); jtime = !(e_ && atoi(e_) == 0); }
                const double rate = S->t_fact > 0 && S->nfact > 0 ? (double)S->nfact * fs / S->t_fact : 0.0;
                const double tspent = lp_time() - tph0;
                if (jtime && rate > 0 && tspent > 0) jcap = fmax(jcap, rate * tspent / 10.0); }
            const double tj0 = lp_time();
            /* (4 November 2026) the exact count within a quarter of the time the method has spent (at least 0.5 s), as the
             * estimate of the form's choice: it builds the joined pattern, whose cliques can hold hundreds of millions of
             * entries, and when it does not finish in that time the join counts as too costly, which the clique over the
             * union already suggested. rmine13 (LP, dual form, 26 dense columns of up to 4,394 entries over all 23,892 rows):
             * the count took 11.7 s to find 4.18e12 flops against the cap's 1.85e11 (the clique bound 4.55e12), and refused.
             * A tenth was too little for zeil's permuted copy (LP, the case the join exists for: its count took 4.1 s after
             * about 20 s of the method, and refused, its crossover failed: 34 s -> past 150 s). BRISK_LPJOINEST=0: no time limit */
            if (fj > jcap) {
                static int jest = -1; if (jest < 0) { const char *e_ = getenv("BRISK_LPJOINEST"); jest = !(e_ && atoi(e_) == 0); }
                const double dl0 = ne_dl;
                if (jest) { const double tb = fmax(0.5, 0.25 * (tj0 - tph0)); ne_dl = ne_dl > 0 ? fmin(ne_dl, tj0 + tb) : tj0 + tb; }
                LpNeCost J; ne_cost_thr(m, S->n, S->Ap, S->Ai, S->Ax, &J, jcap, 1e300, NULL);
                const int cut = jest && !(ne_stopf && *ne_stopf) && (dl0 <= 0 || lp_time() < dl0) && lp_time() > ne_dl;
                ne_dl = dl0;
                fj = J.ok && !cut ? J.flops : 1e300;
                if (getenv("BRISK_LPDBG") && cut) printf("   [the joined form's count cut at its time limit: the join refused]\n");
            }
            if (getenv("BRISK_LPDBG")) printf("   [dense columns lost: iteration %d, gap %.1e, m %d, nd %d, fs %.2e, joined %.2e (cap %.2e), the estimate %.3f s]\n", it, e_g, m, S->nd, fs, fj, jcap, lp_time() - tj0);
            /* (1 November 2026) only the dense columns whose weight has grown join the factored part: the loss comes from
             * them (G = C^-1 + W'W grows ill-conditioned in their span: on Q083_energy, LP, 37 of 242 columns basic, weights
             * up to 2.6e11), while the others' weights fall and their correction stays well conditioned. Taken when the
             * factorization with them in it costs at most the same ten times (the analysis made to know it; when it costs
             * more, the former setup again). BRISK_LPPJOIN=0: not */
            static int pjoin = -1; if (pjoin < 0) { const char *e_ = getenv("BRISK_LPPJOIN"); pjoin = !(e_ && atoi(e_) == 0); }
            int pj_done = 0;
            if (fj > jcap && pjoin && !getenv("BRISK_LPJOIN")) {
                int k = 0; for (int d = 0; d < S->nd; d++) k += S->theta[S->dcol[d]] > 1.0;
                /* (the clique over the union of their rows first, without an analysis: a new analysis refused afterwards costs
                 * two (neos-4306827-ravan, LP: 23 of 24 columns over 15,994 rows, the attempt 2 -> 9.7 s)) */
                if (k > 0 && k < S->nd) {
                    long cs = 0; char *mk = xz((size_t)m + 1);
                    for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; if (S->theta[j] > 1.0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (!mk[S->Ai[p]]) { mk[S->Ai[p]] = 1; cs++; } }
                    free(mk);
                    if (getenv("BRISK_LPDBG")) printf("   [dense columns with weight above 1: %d of %d over %ld rows: a clique of %.2e flops against the cap %.2e]\n", k, S->nd, cs, (double)cs * cs * cs / 3.0, jcap);
                    if ((double)cs * cs * cs / 3.0 > jcap) k = 0;
                }
                if (k > 0 && k < S->nd) {
                    if (!S->djoin) S->djoin = xz((size_t)S->n + 1);
                    int *jl = xm(sizeof(int) * (size_t)k); int nj = 0;
                    for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; if (S->theta[j] > 1.0) { S->djoin[j] = 1; jl[nj++] = j; } }
                    const int nd0 = S->nd;
                    if (!ne_setup(S, 1) && S->sc) {
                        const double fn = schol_flops(S->sc) + S->nd * (4.0 * (double)schol_nnz(S->sc) + 2.0 * S->nd * m);
                        if (getenv("BRISK_LPDBG")) printf("   [dense columns: %d of %d with weight above 1 joined: %.2e flops a factorization against the cap %.2e]\n", nj, nd0, fn, jcap);
                        if (fn <= jcap && !ne_factor(S)) {
                            pj_done = 1; S->njoin += nj;
                            if (verbose > 1) printf("     (dense columns: residual %.1e; %d of the %d with weights above 1 join the factored matrix)\n", S->worst, nj, nd0);
                        }
                    }
                    if (!pj_done) {
                        for (int t = 0; t < nj; t++) S->djoin[jl[t]] = 0;
                        if (ne_expired() || S->fstop) { free(jl); status = 6; snprintf(info->why, sizeof info->why, "stopped"); break; }
                        if (ne_setup(S, 1) || ne_factor(S)) { free(jl); status = 4; snprintf(info->why, sizeof info->why, "factorization failed"); break; }
                    }
                    free(jl);
                }
            }
            if (pj_done) {
                info->ndense = S->nd; info->lnz = (long)schol_nnz(S->sc); S->worst = 0;
                direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
            } else
            if (fj > jcap && !getenv("BRISK_LPJOIN")) {
                if (verbose > 1) printf("     (dense columns: residual %.1e; joined they would make a clique of %ld rows, %.1e flops a factorization)\n", S->worst, c, fj);
                status = 4; info->dense_lost = 1; snprintf(info->why, sizeof info->why, "the dense columns lost their accuracy; joined, %ld rows would be dense", c); break;
            }
            if (!pj_done) {
            if (verbose > 1) printf("     (dense columns: residual %.1e, they join the factored matrix)\n", S->worst);
            if (ne_setup(S, 0) || ne_factor(S)) {
                if (ne_expired() || S->fstop) { status = 6; snprintf(info->why, sizeof info->why, "stopped"); break; }
                status = 4; snprintf(info->why, sizeof info->why, "the factor with the dense columns does not fit"); break;
            }
            info->ndense = 0; info->lnz = (long)schol_nnz(S->sc); S->worst = 0;
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
            }
        }
        /* (21 October 2026, the night, tried and dropped: stopping when a solve's residual exceeds its right-hand side a
         * hundredfold. neos-5052403-cygnet after the simplex method's presolve breaks down that way near its end, but on
         * the problem as read rail01 and cygnet go on through residuals of 4e8 and 5e5 and converge (42 and 27
         * iterations); the stall test below decides, TRICKS 8.) */
        double ap, ad;
        step_max(S, x, s, z, w, dx, ds, dz, dw, &ap, &ad);
        ap = fmin(1.0, ap); ad = fmin(1.0, ad);
        double mua = 0;
        LPV_SUM(mua, if (S->fr[j]) continue; v += (x[j] + ap * dx[j]) * (z[j] + ad * dz[j]); if (S->bd[j]) v += (s[j] + ap * ds[j]) * (w[j] + ad * dw[j]););
        mua /= fmax(ncomp, 1);
        if (getenv("BRISK_LPDBG")) {      /* (diagnostics: the affine steps and the spread of the products around mu) */
            double pmn = 1e300, pmx = 0; int nlo = 0, nhi = 0;
            for (int j = 0; j < n; j++) { if (S->fr[j]) continue; const double p = x[j] * z[j]; pmn = fmin(pmn, p); pmx = fmax(pmx, p); nlo += p < 1e-2 * mu; nhi += p > 1e2 * mu;
                if (S->bd[j]) { const double q = s[j] * w[j]; pmn = fmin(pmn, q); pmx = fmax(pmx, q); nlo += q < 1e-2 * mu; nhi += q > 1e2 * mu; } }
            printf("     [affine steps %.3f %.3f, mu_aff/mu %.3f; products / mu %.1e .. %.1e, %d below mu/100, %d above 100 mu]\n", ap, ad, mua / mu, pmn / mu, pmx / mu, nlo, nhi);
            int kd = -1, nblk = 0; double bd = 1e300;
            for (int j = 0; j < n; j++) { if (S->fr[j]) continue; if (dz[j] < 0) { const double t = -z[j] / dz[j]; if (t < bd) { bd = t; kd = j; } if (t < 2 * ad + 1e-12) nblk++; } }
            if (kd >= 0) printf("     [dual blocking: column %d (%d entries, cost %.2e, ub %d): x %.2e z %.2e dz %.2e dx %.2e x z / mu %.1e; %d components within twice the step]\n",
                                kd, S->Ap[kd + 1] - S->Ap[kd], S->c[kd], S->bd[kd], x[kd], z[kd], dz[kd], dx[kd], x[kd] * z[kd] / mu, nblk);
            int kp = -1; double bp = 1e300;
            for (int j = 0; j < n; j++) { if (S->fr[j]) continue; if (dx[j] < 0) { const double t = -x[j] / dx[j]; if (t < bp) { bp = t; kp = j; } } }
            if (kp >= 0) printf("     [primal blocking: column %d (%d entries): x %.2e z %.2e dx %.2e x z / mu %.1e]\n", kp, S->Ap[kp + 1] - S->Ap[kp], x[kp], z[kp], dx[kp], x[kp] * z[kp] / mu);
        }
        double sigma = pow(mua / mu, 3.0);
        if (sigma > 1.0) sigma = 1.0;
        if (sigma < 1e-6) sigma = 1e-6;
        /* ---- 20 October 2026 (LP/SOCP): Mehrotra's sigma bounded by 0.3 (BRISK_LPSIGMAX=s; 1 the plain heuristic). On
         * the transportation LPs the affine step is blocked by the columns that enter the basis (dz / z = -1 - dx / x)
         * at a point whose products lie within [0.3, 3] mu: the heuristic reads the short step as a lack of centrality
         * and centres (sigma 0.9, mu falling by 5-10 % an iteration) where a long step is safe. The 142 LPs of the sets:
         * no status changed, iterations 2,871 -> 2,810, time 0.966 (0.906 over a second); transport_300 38 -> 24,
         * transport_1000 95 -> 86, cre-b 46 -> 37, pds-20 53 -> 47; pilot4 38 -> 47, perold 38 -> 44. A cap of 0.1
         * gained the same time with more losses (pilot4 56; perold's LP method failed, the fallback solved it). A test,
         * off: the cap only at a centred point (BRISK_LPSIGCEN=gamma: every product at least gamma mu) never fires. */
        { static double smax = -1, scen = -1;
          if (smax < 0) { const char *e_ = getenv("BRISK_LPSIGMAX"); smax = e_ ? atof(e_) : 0.3; const char *g_ = getenv("BRISK_LPSIGCEN"); scen = g_ ? atof(g_) : 0.0; }
          if (sigma > smax) {
              int cen = 1;
              if (scen > 0) for (int j = 0; j < n && cen; j++) { if (S->fr[j]) continue; if (x[j] * z[j] < scen * mu || (S->bd[j] && s[j] * w[j] < scen * mu)) cen = 0; }
              if (cen) sigma = smax;
          } }
        /* ---- 19 October 2026 (LP/SOCP), a test, off (BRISK_LPARC=1): an arc search (Yang's arc-search interior-point
         * methods). The point moves on the ellipse x(a) = x + sin(a) d1 + (1 - cos(a)) d2 through the path's first and
         * second derivatives: d1 = d_aff + c, c the centring direction (sigma mu, no residuals), and d2 from
         * -2 d_aff d_aff (no residuals); Mehrotra's line is d1 + d2 / 2, so one more solve gives both. The residuals fall
         * by 1 - sin(a) on the arc and by 1 - alpha on the line: the one that removes more of them is taken; Gondzio's
         * correctors at a trial point of the arc are added to d2. On transportation problems the point stays centred
         * while the line's affine step is 0.01-0.07 (the path curves: the flow moves from 10^6 arcs onto 2,000):
         * lp_transport_300/1000 38/95 -> 29/66 iterations (65.6 -> 52.0 s), mcf 21 -> 17 (45.6 -> 35.7 s); Netlib
         * 1,700 -> 1,694, Kennington 455 -> 440, the benchmark's LPs 397 -> 357, all status lines kept (ISSUES 208).
         * Not the default: with the iterations equal it is slower (the extra solves; osa-60 1.79 -> 2.40 s, ken-18
         * 2.71 -> 3.92 s, beckmann 5.1 -> 6.1 s). The centring belongs in d1: with sigma mu in d2 a short arc loses it
         * ((1 - cos a) is of second order) and Netlib needed 1,832 iterations. */
        int arc_line = 0;
        static int arcon = -1; if (arcon < 0) { const char *e_ = getenv("BRISK_LPARC"); arcon = e_ ? atoi(e_) : 0; }
        if (arcon == 1) {
            if (!S->wa) { S->wa = xm(sizeof(double) * (4 * (size_t)n + m + 4)); }
            double *dxa = S->wa, *dza = dxa + n, *dsa = dza + n, *dwa = dsa + n, *dya = dwa + n;
            memcpy(dxa, dx, sizeof(double) * n); memcpy(dza, dz, sizeof(double) * n); memcpy(dsa, ds, sizeof(double) * n); memcpy(dwa, dw, sizeof(double) * n); memcpy(dya, dy, sizeof(double) * m);
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu; rsw[j] = S->bd[j] ? sigma * mu : 0.0; }
            direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = -2.0 * dxa[j] * dza[j]; rsw[j] = S->bd[j] ? -2.0 * dsa[j] * dwa[j] : 0.0; }
            direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
            for (int j = 0; j < n; j++) { dxa[j] += dx2[j]; dza[j] += dz2[j]; dsa[j] += ds2[j]; dwa[j] += dw2[j]; dx2[j] = dxa[j] + 0.5 * dx[j]; dz2[j] = dza[j] + 0.5 * dz[j]; ds2[j] = dsa[j] + 0.5 * ds[j]; dw2[j] = dwa[j] + 0.5 * dw[j]; }
            for (int i = 0; i < m; i++) { dya[i] += dy2[i]; dy2[i] = dya[i] + 0.5 * dy[i]; }
            double tp, td; arc_max(S, x, s, z, w, dxa, dsa, dza, dwa, dx, ds, dz, dw, &tp, &td);
            double lp_, ld_; step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &lp_, &ld_);
            if (fmin(1.0, lp_) + fmin(1.0, ld_) > sin(tp) + sin(td)) {
                memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                arc_line = 1;
            } else {
                for (int k = 0; k < kcorr; k++) {     /* Gondzio's correctors at a trial point of the arc */
                    if (tp >= 0.999 * M_PI / 2 && td >= 0.999 * M_PI / 2) break;
                    const double ttp = fmin(M_PI / 2, 1.08 * tp + 0.08), ttd = fmin(M_PI / 2, 1.08 * td + 0.08);
                    const double s1 = sin(ttp), c1 = 1.0 - cos(ttp), s2 = sin(ttd), c2 = 1.0 - cos(ttd);
                    const double mt = sigma * mu, lo = 0.1 * mt, hi = 10.0 * mt;
                    for (int j = 0; j < n; j++) {
                        if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; }
                        double v = (x[j] + s1 * dxa[j] + c1 * dx[j]) * (z[j] + s2 * dza[j] + c2 * dz[j]);
                        double t = v < lo ? lo - v : v > hi ? hi - v : 0.0; if (t < -hi) t = -hi; rxz[j] = t;
                        if (S->bd[j]) { v = (s[j] + s1 * dsa[j] + c1 * ds[j]) * (w[j] + s2 * dwa[j] + c2 * dw[j]); t = v < lo ? lo - v : v > hi ? hi - v : 0.0; if (t < -hi) t = -hi; rsw[j] = t; } else rsw[j] = 0;
                    }
                    direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
                    for (int j = 0; j < n; j++) { dx2[j] = dx[j] + dx2[j] / c1; dz2[j] = dz[j] + dz2[j] / c2; ds2[j] = ds[j] + ds2[j] / c1; dw2[j] = dw[j] + dw2[j] / c2; }
                    for (int i = 0; i < m; i++) dy2[i] = dy[i] + dy2[i] / c2;
                    double bp, bdl; arc_max(S, x, s, z, w, dxa, dsa, dza, dwa, dx2, ds2, dz2, dw2, &bp, &bdl);
                    if (!(bp + bdl > tp + td + 1e-3)) break;
                    memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                    tp = bp; td = bdl; info->ncorr++;
                }
                const double eta = fmax(0.9, fmin(0.99995, 1.0 - 0.1 * mu / fmax(mu0, 1e-300))), etaf = mu < 1e-6 * mu0 ? 0.9999 : fmax(eta, 0.995);
                if (tp < M_PI / 2) tp *= etaf;
                if (td < M_PI / 2) td *= etaf;
                if (nfree == n) { tp = td = M_PI / 2; }
                const double sp = sin(tp), cp = 1.0 - cos(tp), sd = sin(td), cd = 1.0 - cos(td);
                for (int j = 0; j < n; j++) { x[j] += sp * dxa[j] + cp * dx[j]; z[j] += sd * dza[j] + cd * dz[j]; if (S->bd[j]) { s[j] += sp * dsa[j] + cp * ds[j]; w[j] += sd * dwa[j] + cd * dw[j]; } }
                for (int i = 0; i < m; i++) y[i] += sd * dya[i] + cd * dy[i];
                if (verbose > 1) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %5.3f %5.3f %6.0e %3ld %7.0e %d arc\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, sp, sd, sigma, S->ncg - cg0, S->worst, schol_ntiny(S->sc));
                continue;
            }
        }
        /* ---- corrector. 18 October 2026 (LP/SOCP), a test, off: weighted (Colombo and Gondzio 2008): the corrector's own
         * direction (no residuals; sigma mu - dx_a dz_a) solved apart and added to the affine one with the weight omega
         * in [alpha_p alpha_d, 1] (the affine steps) that gives the longest steps (BRISK_LPWCORR=1; 2: the weight with
         * the least complementarity after the step). The same number of solves. With weight 1 the second-order term
         * of a short affine step (0.02-0.05 on a degenerate transportation problem) is too large. Measured (ISSUES
         * 208): lp_transport_1000 95 -> 63 iterations, Netlib 1702 -> 1680, but Kennington 455 -> 488 (osa-14/30/60
         * 22 -> 38-41: the greedy weight leads into badly centred regions); rule 2 worse everywhere. */
        static int wcorr = -1; if (wcorr < 0) { const char *e_ = getenv("BRISK_LPWCORR"); wcorr = e_ ? atoi(e_) : 0; }
        if (wcorr && !arcon) {
            if (!S->wa) { S->wa = xm(sizeof(double) * (4 * (size_t)n + m + 4)); }
            double *dxa = S->wa, *dza = dxa + n, *dsa = dza + n, *dwa = dsa + n, *dya = dwa + n;
            memcpy(dxa, dx, sizeof(double) * n); memcpy(dza, dz, sizeof(double) * n); memcpy(dsa, ds, sizeof(double) * n); memcpy(dwa, dw, sizeof(double) * n); memcpy(dya, dy, sizeof(double) * m);
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu - dxa[j] * dza[j]; rsw[j] = S->bd[j] ? sigma * mu - dsa[j] * dwa[j] : 0.0; }
            direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);      /* (dx..: the corrector part) */
            const double om0 = fmin(1.0, fmax(ap, 0.0) * fmax(ad, 0.0));
            double bom = 1.0, bsum = -1e300;
            for (int k = 0; k <= 8; k++) {
                const double om = om0 + (1.0 - om0) * k / 8.0;
                for (int j = 0; j < n; j++) { dx2[j] = dxa[j] + om * dx[j]; dz2[j] = dza[j] + om * dz[j]; ds2[j] = dsa[j] + om * ds[j]; dw2[j] = dwa[j] + om * dw[j]; }
                double bp, bdl; step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &bp, &bdl);
                double sm = fmin(1.0, bp) + fmin(1.0, bdl);
                if (wcorr == 2) {          /* (test: the weight with the least complementarity after 0.95 of the steps) */
                    const double a1 = 0.95 * fmin(1.0, bp), a2 = 0.95 * fmin(1.0, bdl); double mf = 0;
                    for (int j = 0; j < n; j++) { if (S->fr[j]) continue; mf += (x[j] + a1 * dx2[j]) * (z[j] + a2 * dz2[j]); if (S->bd[j]) mf += (s[j] + a1 * ds2[j]) * (w[j] + a2 * dw2[j]); }
                    sm = -mf;
                }
                if (sm > bsum + 1e-12 || k == 0) { bsum = sm; bom = om; }
            }
            for (int j = 0; j < n; j++) { dx[j] = dxa[j] + bom * dx[j]; dz[j] = dza[j] + bom * dz[j]; ds[j] = dsa[j] + bom * ds[j]; dw[j] = dwa[j] + bom * dw[j]; }
            for (int i = 0; i < m; i++) dy[i] = dya[i] + bom * dy[i];
            sigma *= bom;          /* (the centring target that the direction carries, for the correctors below) */
            if (getenv("BRISK_LPDBG")) printf("     [corrector weight %.3f (from %.3f)]\n", bom, om0);
        } else if (!arc_line) {
            if (arcon == 2) {           /* (test: the affine direction kept for the arc after the line) */
                if (!S->wa) { S->wa = xm(sizeof(double) * (4 * (size_t)n + m + 4)); }
                double *dxa = S->wa, *dza = dxa + n, *dsa = dza + n, *dwa = dsa + n;
                memcpy(dxa, dx, sizeof(double) * n); memcpy(dza, dz, sizeof(double) * n); memcpy(dsa, ds, sizeof(double) * n); memcpy(dwa, dw, sizeof(double) * n);
            }
            LPV_FOR for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu - x[j] * z[j] - dx[j] * dz[j]; rsw[j] = S->bd[j] ? sigma * mu - s[j] * w[j] - ds[j] * dw[j] : 0.0; }
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        step_max(S, x, s, z, w, dx, ds, dz, dw, &ap, &ad);
        /* ---- 20 October 2026 (LP/SOCP), a test, off (BRISK_LPSIGSEL=1): sigma chosen by the progress it gives. The
         * corrected direction is affine in sigma, d(sigma) = d(sigma_M) + (sigma - sigma_M) d_c with d_c the centring
         * direction (right-hand side mu e, no residuals), so one more solve gives every candidate; the candidate whose
         * step leaves the smallest mean product is taken. Tried when Mehrotra's sigma is at least BRISK_LPSIGSELT (0.2):
         * on the transportation LPs the affine step is blocked by the entering columns (dz / z = -1 - dx / x) at a
         * centred point, and the heuristic centres (sigma 0.9, mu falling 5-10 % an iteration) where a long step is safe. */
        { static int ssel = -1; static double sselt = 0.2;
          if (ssel < 0) { const char *e_ = getenv("BRISK_LPSIGSEL"); ssel = e_ ? atoi(e_) : 0; const char *t_ = getenv("BRISK_LPSIGSELT"); if (t_) sselt = atof(t_); }
          if (ssel && sigma >= sselt && !arc_line) {
              if (!S->wa) { S->wa = xm(sizeof(double) * (4 * (size_t)n + m + 4)); }
              double *tx = S->wa, *tz = tx + n, *ts = tz + n, *tw = ts + n;
              for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = mu; rsw[j] = S->bd[j] ? mu : 0.0; }
              direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
              const double cand[7] = { 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7 };
              double best = 0, bsig = sigma, bap = ap, bad_ = ad;
              { const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
                for (int j = 0; j < n; j++) { if (S->fr[j]) continue; best += (x[j] + a1 * dx[j]) * (z[j] + a2 * dz[j]); if (S->bd[j]) best += (s[j] + a1 * ds[j]) * (w[j] + a2 * dw[j]); } }
              for (int c = 0; c < 7; c++) {
                  const double ds_ = cand[c] - sigma; if (fabs(ds_) < 1e-3) continue;
                  for (int j = 0; j < n; j++) { tx[j] = dx[j] + ds_ * dx2[j]; tz[j] = dz[j] + ds_ * dz2[j]; ts[j] = ds[j] + ds_ * ds2[j]; tw[j] = dw[j] + ds_ * dw2[j]; }
                  double cp, cd; step_max(S, x, s, z, w, tx, ts, tz, tw, &cp, &cd);
                  const double a1 = fmin(1.0, cp), a2 = fmin(1.0, cd); double v = 0;
                  for (int j = 0; j < n; j++) { if (S->fr[j]) continue; v += (x[j] + a1 * tx[j]) * (z[j] + a2 * tz[j]); if (S->bd[j]) v += (s[j] + a1 * ts[j]) * (w[j] + a2 * tw[j]); }
                  if (v < best) { best = v; bsig = cand[c]; bap = cp; bad_ = cd; }
              }
              if (bsig != sigma) {
                  const double ds_ = bsig - sigma;
                  for (int j = 0; j < n; j++) { dx[j] += ds_ * dx2[j]; dz[j] += ds_ * dz2[j]; ds[j] += ds_ * ds2[j]; dw[j] += ds_ * dw2[j]; }
                  for (int i = 0; i < m; i++) dy[i] += ds_ * dy2[i];
                  if (getenv("BRISK_LPDBG")) printf("     [sigma %.2f -> %.2f by the progress: steps %.3f %.3f -> %.3f %.3f]\n", sigma, bsig, ap, ad, bap, bad_);
                  sigma = bsig; ap = bap; ad = bad_;
              }
          } }
        /* (20 October 2026, LP/SOCP, a test, BRISK_LPARC=2: the arc only after a short line. Mehrotra's direction is
         * d1 + d2/2 of the arc's d1 = d_aff + c and d2 (from -2 d_aff d_aff, no residuals), so one more solve gives
         * both; taken when the line's shorter step is below BRISK_LPARCT (0.5) and the arc removes more) */
        if (arcon == 2 && !arc_line) {
            static double arct = -1; if (arct < 0) { const char *e_ = getenv("BRISK_LPARCT"); arct = e_ ? atof(e_) : 0.5; }
            if (fmin(ap, ad) < arct) {
                double *dxa = S->wa, *dza = dxa + n, *dsa = dza + n, *dwa = dsa + n;
                for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = -2.0 * dxa[j] * dza[j]; rsw[j] = S->bd[j] ? -2.0 * dsa[j] * dwa[j] : 0.0; }
                direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
                for (int j = 0; j < n; j++) { dxa[j] = dx[j] - 0.5 * dx2[j]; dza[j] = dz[j] - 0.5 * dz2[j]; dsa[j] = ds[j] - 0.5 * ds2[j]; dwa[j] = dw[j] - 0.5 * dw2[j]; }
                double tp, td; arc_max(S, x, s, z, w, dxa, dsa, dza, dwa, dx2, ds2, dz2, dw2, &tp, &td);
                if (sin(fmin(tp, M_PI / 2)) + sin(fmin(td, M_PI / 2)) > fmin(1.0, ap) + fmin(1.0, ad)) {
                    const double eta = fmax(0.9, fmin(0.99995, 1.0 - 0.1 * mu / fmax(mu0, 1e-300))), etaf = mu < 1e-6 * mu0 ? 0.9999 : fmax(eta, 0.995);
                    if (tp < M_PI / 2) tp *= etaf;
                    if (td < M_PI / 2) td *= etaf;
                    const double sp = sin(tp), cp = 1.0 - cos(tp), sd = sin(td), cd = 1.0 - cos(td);
                    for (int j = 0; j < n; j++) { x[j] += sp * dxa[j] + cp * dx2[j]; z[j] += sd * dza[j] + cd * dz2[j]; if (S->bd[j]) { s[j] += sp * dsa[j] + cp * ds2[j]; w[j] += sd * dwa[j] + cd * dw2[j]; } }
                    for (int i = 0; i < m; i++) y[i] += sd * (dy[i] - 0.5 * dy2[i]) + cd * dy2[i];
                    if (verbose > 1) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %5.3f %5.3f %6.0e %3ld %7.0e %d arc\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, sp, sd, sigma, S->ncg - cg0, S->worst, schol_ntiny(S->sc));
                    continue;
                }
            }
        }
        /* a blocked step: the second-order term was wrong for this point - a centering step instead */
        { static int safe = -1; if (safe < 0) { const char *e_ = getenv("BRISK_LPSAFE"); safe = getenv("BRISK_LPNOSAFE") ? 0 : e_ ? atoi(e_) : 4; }   /* (21 October 2026, LP/SOCP: 4 by default; tests: 0 off, 1 as before, 2 sigma kept when the centering step is rejected, 3 kept by the predicted progress) */
        /* (21 October 2026, LP/SOCP) only when this iteration's solves needed refinement (safe = 4): the direction is then
         * inaccurate and its second-order term suspect, the case the centering step rescues (the multicommodity flow's end
         * phase: steps 0.06, 0.01, 0.00 with conjugate-gradient steps in every solve; 17 iterations with it, 24 without).
         * A step that is short with an accurate direction is the path's curvature (the transportation LPs: one or two
         * entering columns block it); the longer centering step made little progress there and kept the iterate in that
         * regime (transport_1000 86 iterations with the rule, 40 without; a family of eight transportation LPs 22-98 with,
         * 22-40 without). */
        if (safe && fmin(ap, ad) < 0.1 && (safe != 4 || S->ncg > cg0)) {
            const double sigc = fmax(sigma, 0.5);
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigc * mu - x[j] * z[j]; rsw[j] = S->bd[j] ? sigc * mu - s[j] * w[j] : 0.0; }
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            double bp, bdl;
            step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &bp, &bdl);
            int take = fmin(bp, bdl) > fmin(ap, ad);
            if (safe == 3) {          /* the larger of the infeasibilities' and the complementarity's remaining fractions after each step */
                double m1 = 0, m2 = 0; const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad), b1 = fmin(1.0, bp), b2 = fmin(1.0, bdl);
                for (int j = 0; j < n; j++) { if (S->fr[j]) continue; m1 += (x[j] + a1 * dx[j]) * (z[j] + a2 * dz[j]); m2 += (x[j] + b1 * dx2[j]) * (z[j] + b2 * dz2[j]);
                    if (S->bd[j]) { m1 += (s[j] + a1 * ds[j]) * (w[j] + a2 * dw[j]); m2 += (s[j] + b1 * ds2[j]) * (w[j] + b2 * dw2[j]); } }
                m1 /= fmax(ncomp, 1) * mu; m2 /= fmax(ncomp, 1) * mu;
                const double inf = fmax(e_p, e_d) > 1e-3 * fmax(e_g, 1e-300) ? 1.0 : 0.0;   /* (the infeasibilities count while they are not negligible against the gap) */
                const double p1 = fmax(m1, inf * (1.0 - fmin(a1, a2))), p2 = fmax(m2, inf * (1.0 - fmin(b1, b2)));
                take = p2 < p1;
            }
            if (take) {
                memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                ap = bp; ad = bdl; nsafe++; sigma = sigc;
            } else if (safe == 1) sigma = sigc;          /* (as before: sigma raised even when the centering step was not taken) */
        } }
        /* ---- Gondzio's centrality correctors. (24 October 2026) where the cost ratio allows none, a blocked step (below 0.1)
         * still gets them: the blocking component sits at gamma_f times the mean product (Mehrotra's step rule puts it
         * there), the next direction drives it to its bound again, and the steps stay near zero for several iterations;
         * the correctors lift such outliers (buildingenergy, LP: 12 of 55 iterations with steps of 0.001-0.03, no
         * correctors by the cost ratio). BRISK_LPCORRBLK=k: their number there (0: none) */
        static int corrblk = -1; if (corrblk < 0) { const char *e_ = getenv("BRISK_LPCORRBLK"); corrblk = e_ ? atoi(e_) : 2; }
        const int kc_it = kcorr == 0 && fmin(ap, ad) < 0.1 ? corrblk : kcorr;
        for (int k = 0; k < kc_it; k++) {
            const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
            if (a1 >= 0.999 && a2 >= 0.999) break;
            const double tp = fmin(1.0, 1.08 * a1 + 0.08), td = fmin(1.0, 1.08 * a2 + 0.08);
            const double mt = sigma * mu, lo = 0.1 * mt, hi = 10.0 * mt;
            LPV_FOR for (int j = 0; j < n; j++) {
                if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; }
                double v = (x[j] + tp * dx[j]) * (z[j] + td * dz[j]);
                double t = v < lo ? lo - v : v > hi ? hi - v : 0.0;
                if (t < -hi) t = -hi;
                rxz[j] = t;
                if (S->bd[j]) { v = (s[j] + tp * ds[j]) * (w[j] + td * dw[j]); t = v < lo ? lo - v : v > hi ? hi - v : 0.0; if (t < -hi) t = -hi; rsw[j] = t; } else rsw[j] = 0;
            }
            direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            LPV_FOR for (int j = 0; j < n; j++) { dx2[j] += dx[j]; dz2[j] += dz[j]; ds2[j] += ds[j]; dw2[j] += dw[j]; }
            for (int i = 0; i < m; i++) dy2[i] += dy[i];
            double bp, bdl;
            step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &bp, &bdl);
            if (fmin(1.0, bp) >= a1 + 0.01 * 0.08 || fmin(1.0, bdl) >= a2 + 0.01 * 0.08) {
                if (!(fmin(1.0, bp) + fmin(1.0, bdl) > a1 + a2)) break;
                memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                ap = bp; ad = bdl; info->ncorr++;
            } else break;
        }
        /* ---- step lengths. Mehrotra's rule: the blocking component keeps a complementarity product of
         * gamma_f times the mean after the full step, and the step is at least 0.9 of the way to the boundary */
        if (steprule == 1) {
            const double gf = getenv("BRISK_LPGF") ? atof(getenv("BRISK_LPGF")) : 0.01, ga = 0.1;
            const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
            double muf = 0;
            LPV_SUM(muf, if (S->fr[j]) continue; v += (x[j] + a1 * dx[j]) * (z[j] + a2 * dz[j]); if (S->bd[j]) v += (s[j] + a1 * ds[j]) * (w[j] + a2 * dw[j]););
            muf = fmax(muf, 0.0) / fmax(ncomp, 1);
            /* the blocking components */
            int kp = -1, kd = -1, tp_ = 0, td_ = 0; double bp_ = 1e300, bd_ = 1e300;
            {   /* (by chunks on the threads, the first strict minimum of each chunk, the chunks combined in order: the
                 * sequential pass's choice) */
                const int nch = n >= LPV_MIN ? (n + LPV_CH - 1) / LPV_CH : 1, ch = n >= LPV_MIN ? LPV_CH : n;
                double *cb = S->vpart; int *ci = (int *)(S->vpart + 2 * (size_t)nch);
                #pragma omp parallel for schedule(static) if (lpv_par(S))
                for (int c = 0; c < nch; c++) {
                    int kp1 = -1, kd1 = -1, tp1 = 0, td1 = 0; double bp1 = 1e300, bd1 = 1e300; const int j1 = (c + 1) * ch < n ? (c + 1) * ch : n;
                    for (int j = c * ch; j < j1; j++) {
                        if (S->fr[j]) continue;
                        if (dx[j] < 0) { const double t = -x[j] / dx[j]; if (t < bp1) { bp1 = t; kp1 = j; tp1 = 0; } }
                        if (dz[j] < 0) { const double t = -z[j] / dz[j]; if (t < bd1) { bd1 = t; kd1 = j; td1 = 0; } }
                        if (S->bd[j]) {
                            if (ds[j] < 0) { const double t = -s[j] / ds[j]; if (t < bp1) { bp1 = t; kp1 = j; tp1 = 1; } }
                            if (dw[j] < 0) { const double t = -w[j] / dw[j]; if (t < bd1) { bd1 = t; kd1 = j; td1 = 1; } }
                        }
                    }
                    cb[2 * c] = bp1; cb[2 * c + 1] = bd1; ci[4 * c] = kp1; ci[4 * c + 1] = tp1; ci[4 * c + 2] = kd1; ci[4 * c + 3] = td1;
                }
                for (int c = 0; c < nch; c++) {
                    if (cb[2 * c] < bp_) { bp_ = cb[2 * c]; kp = ci[4 * c]; tp_ = ci[4 * c + 1]; }
                    if (cb[2 * c + 1] < bd_) { bd_ = cb[2 * c + 1]; kd = ci[4 * c + 2]; td_ = ci[4 * c + 3]; }
                }
            }
            double fp = 1.0, fd = 1.0;
            if (kp >= 0 && ap < 1.0) {
                const double xk = tp_ ? s[kp] : x[kp], dxk = tp_ ? ds[kp] : dx[kp], zk = tp_ ? w[kp] + a2 * dw[kp] : z[kp] + a2 * dz[kp];
                fp = zk > 0 ? (gf * muf / zk - xk) / (ap * dxk) : 1.0 - ga;
                fp = fmin(fmax(fp, 1.0 - ga), 0.999999);
            }
            if (kd >= 0 && ad < 1.0) {
                const double zk = td_ ? w[kd] : z[kd], dzk = td_ ? dw[kd] : dz[kd], xk = td_ ? s[kd] + a1 * ds[kd] : x[kd] + a1 * dx[kd];
                fd = xk > 0 ? (gf * muf / xk - zk) / (ad * dzk) : 1.0 - ga;
                fd = fmin(fmax(fd, 1.0 - ga), 0.999999);
            }
            ap = fmin(1.0, fp * ap); ad = fmin(1.0, fd * ad);
        } else {
            const double eta = fmax(0.9, fmin(0.99995, 1.0 - 0.1 * mu / fmax(mu0, 1e-300)));
            const double etaf = steprule == 3 ? 0.99 : steprule == 2 ? eta : mu < 1e-6 * mu0 ? 0.9999 : fmax(eta, 0.995);
            ap = fmin(1.0, etaf * ap); ad = fmin(1.0, etaf * ad);
        }
        if (steprule == 3) { ap = ad = fmin(ap, ad); }
        if (nfree == n) { ap = 1.0; ad = 1.0; }
        LPV_FOR for (int j = 0; j < n; j++) { x[j] += ap * dx[j]; z[j] += ad * dz[j]; if (S->bd[j]) { s[j] += ap * ds[j]; w[j] += ad * dw[j]; } }
        for (int i = 0; i < m; i++) y[i] += ad * dy[i];
        tab_step = fmin(ap, ad);
        if (verbose > 1) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %5.3f %5.3f %6.0e %3ld %7.0e %d\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, ap, ad, sigma, S->ncg - cg0, S->worst, schol_ntiny(S->sc));
    }
    (void)nsafe;
done:
    brisk_itab_end();
    if (getenv("BRISK_LPDBG")) printf("   [LP: the assembly of the normal equations %.3fs of the factorizations' time]\n", S->t_asm);
    if (getenv("BRISK_LPDBG")) fprintf(stderr, "   [LP phases: copy %.3fs, scaling %.3fs, pattern and analysis %.3fs, start %.3fs, iterations %.3fs (factorizations %.3fs, of which the dense columns' correction %.3fs, solves %.3fs; %ld factorizations, %ld preconditioner solves, %ld CG steps, %ld solve calls, %ld refinements)]\n", tph0 - t0, tph1 - tph0, tph2 - tph1, tph3 - tph2, lp_time() - tph3, S->t_fact, S->t_dense, S->t_solve, S->nfact, S->nprec, S->ncg, S->nsolve, S->nref);
    if (getenv("BRISK_LPDBG")) printf("   [LP solves: %ld factorizations, %ld directions, %ld solves with the factor (%ld in CG, %ld refinements), %d correctors taken]\n", S->nfact, S->nsolve, S->nprec, S->ncg, S->nref, info->ncorr);
    if ((status == 3 || status == 4) && xb && bsave < blast) {
        memcpy(x, xb, sizeof(double) * n); memcpy(s, sb, sizeof(double) * n); memcpy(z, zb, sizeof(double) * n); memcpy(w, wb, sizeof(double) * n); memcpy(y, yb, sizeof(double) * m);
        info->pobj = bv[0]; info->dobj = bv[1]; info->pinf = bv[2]; info->dinf = bv[3]; info->gap = bv[4];
        if (verbose > 1) printf("  the best iterate returned (iteration %d, errors %.1e)\n", (int)bv[5], bsave);
    }
    free(xb); free(sb); free(zb); free(wb); free(yb);
    info->status = status; info->iters = it; info->time = lp_time() - t0;
    info->nfact = S->nfact; info->nsolve = S->nsolve; info->ncg = S->ncg; info->ntiny = S->ntiny; info->t_fact = S->t_fact; info->t_solve = S->t_solve;
    /* the solution of the problem as given */
    for (int j = 0; j < n; j++) { xout[j] = x[j] * S->cs[j] * bsc; if (zout) zout[j] = (z[j] - w[j]) * csc / S->cs[j]; if (zl) zl[j] = z[j] * csc / S->cs[j]; if (wu) wu[j] = w[j] * csc / S->cs[j]; }
    for (int i = 0; i < m; i++) yout[i] = y[i] * S->rs[i] * csc;
    schol_set_solve_seq(0);
    free(x); free(s); free(z); free(w); free(y); free(dx); free(ds); free(dz); free(dw); free(dy); free(dx2); free(ds2); free(dz2); free(dw2); free(dy2);
    free(rb); free(ru); free(rc); free(rxz); free(rsw); free(g); free(zr); free(zm);
    lps_free(S);
    return status;
}

/* ---- 18 October 2026 (LP/SOCP): the dual form ----
 * The bounded standard form  min c'x, A x = b, x_F free, 0 <= x_B <= u_B, x_L >= 0  has the dual
 *     max b'y - u_B'w   s.t.   A_j'y = c_j (j in F),  A_j'y + z_j - w_j = c_j (j in B),  A_j'y + z_j = c_j (j in L),  z, w >= 0.
 * A column of L with one entry a_ij (a slack, the positive or negative part of a residual) makes its dual row
 * a bound on y_i, a_ij y_i <= c_j; the other columns (the set R) stay rows. In BRISK's form the dual is then
 *     min -b'y + u'w   s.t.   A_R'y + z - w = c_R,   lo <= y <= up   (y_i shifted to a bound, reflected or free),
 * with |R| rows instead of m: an L1, Chebyshev or quantile regression with N residuals and p coefficients has
 * p rows instead of N (its p free columns, which the primal's normal equations can only take as dense columns,
 * are equality rows of the dual). The primal point is recovered from the dual's multipliers: x_j = -ytilde_j
 * for j in R, and the singleton columns from the multipliers of the bounds of y_i, a_ij x_j = -(zl_i - wu_i)
 * (sign +1) or +zl_i (y_i reflected); a singleton column whose bound is not the tightest one on its y_i is 0. */
static double ne_dense_nnz(const LpS *T) { double s = 0; for (int d = 0; d < T->nd; d++) s += T->Ap[T->dcol[d] + 1] - T->Ap[T->dcol[d]]; return s; }
static void ne_cost(int m, int n, const int *Ap, const int *Ai, const double *Ax, LpNeCost *C, double budget) {
    /* (24 October 2026) the dense-column choice's own estimate of the form it chose, when it made one: the same analysis
     * (the same threshold and budget; a chosen candidate was below its budget, so its estimate is exact) was made
     * again here (irish-electricity, LP: 0.38 s of the default rule's analysis repeated). BRISK_LPNECHOICE=0: again */
    LpNeCost Ch; Ch.ok = -1;
    const double tq0 = lp_time();
    const double thr = ne_dense_choice(m, n, Ap, Ai, Ax, 0, budget, &Ch);
    if (getenv("BRISK_LPDBG")) printf("     [ne_cost %d x %d: the dense-column choice %.3fs (its estimate %s)]\n", m, n, lp_time() - tq0, Ch.ok >= 0 ? "taken" : "none");
    if (Ch.ok >= 0 && !(getenv("BRISK_LPNECHOICE") && atoi(getenv("BRISK_LPNECHOICE")) == 0)) { *C = Ch; C->dthr = thr; return; }
    LpS T; memset(&T, 0, sizeof T); T.m = m; T.n = n; T.est = 1; T.est_budget = budget; T.dthr = thr;
    C->dthr = T.dthr;
    T.Ap = (int *)Ap; T.Ai = (int *)Ai; T.Ax = (double *)Ax;
    const double tq1 = lp_time();
    C->ok = ne_setup(&T, 1) == 0;
    if (getenv("BRISK_LPDBG")) printf("     [ne_cost %d x %d: the analysis %.3fs (ok %d, a lower bound %.3g)]\n", m, n, lp_time() - tq1, C->ok, T.est_lb);
    C->dnz = ne_dense_nnz(&T);
    if (C->ok && T.est_lb > 0) { C->flops = T.est_lb; C->lnz = (double)T.est_w; C->nd = T.nd; T.Ap = NULL; T.Ai = NULL; T.Ax = NULL; lps_free(&T); return; }
    C->flops = C->ok ? schol_flops(T.sc) : 1e300; C->lnz = C->ok ? (double)schol_nnz(T.sc) : 1e300; C->nd = T.nd;
    T.Ap = NULL; T.Ai = NULL; T.Ax = NULL;
    lps_free(&T);
}
/* (2 November 2026) the cost of a form with the ordering its factorization would use (the nested dissection when it pays,
 * as ne_setup decides), at the dense-column threshold dthr already chosen: AMD's count, which the estimates use for speed,
 * depends on the order of the input and can be far above the dissection's (the dual form of genus-sym-g62-2, LP: 2.8e10 or
 * 1.2e10 by AMD depending on the order of the rows, 9e8 by the dissection) */
static void ne_cost_nd(int m, int n, const int *Ap, const int *Ai, const double *Ax, double dthr, double budget, LpNeCost *C) {
    LpS T; memset(&T, 0, sizeof T); T.m = m; T.n = n; T.est = 1; T.est_nd = 1; T.est_budget = budget; T.dthr = dthr;
    T.Ap = (int *)Ap; T.Ai = (int *)Ai; T.Ax = (double *)Ax;
    C->dthr = dthr; C->ok = ne_setup(&T, 1) == 0 && T.sc;
    C->dnz = ne_dense_nnz(&T); C->nd = T.nd;
    C->flops = C->ok ? schol_flops(T.sc) : 1e300; C->lnz = C->ok ? (double)schol_nnz(T.sc) : 1e300;
    if (T.est_lb > 0) { C->ok = 0; C->flops = 1e300; }        /* (above the budget, or stopped: it cannot win) */
    T.Ap = NULL; T.Ai = NULL; T.Ax = NULL;
    lps_free(&T);
}
/* the cost of one iteration in flops: a factorization (with the dense columns' solves and Schur complement)
 * and about ten solves and products with A (the predictor, the corrector, the correctors, the refinement).
 * (22 October 2026: the dense columns as ne_factor and ne_prec treat them: V = M_s^-1 A_d, nd solves of 4 lnz + 2 m;
 * G = Theta_d^-1 + A_d' V, nd dot products over the dense columns' entries, 2 nd dnz; its Cholesky nd^3 / 3; each solve
 * V' r and V g, 4 nd m. The term was 2 nd^2 m, the cost of A_d' V as if A_d were dense: neos-4763324-toguru's dual
 * form (53,824 rows, 232 columns of 231 entries apart) was priced at 2.4e10 flops an iteration against the primal's
 * 3.4e9, where Gurobi's barrier takes that dual form with a factor of 3e7 operations) */
static double ne_iter_cost(const LpNeCost *C, int m, long nnz) {
    /* (28 October 2026, the night) an estimate that did not complete costs "too much": its other fields may never have
     * been set (the dense-column choice's estimates stopped by the deadline before they began), and they were read: a form
     * priced at -8.55e20 flops, z26 (LP) with a time limit that expired in the analysis */
    if (!C->ok) return 1e300;
    const double nd = C->nd;
    /* (each solve's correction: a pass over V or a second solve, whichever is cheaper, as ne_setup chooses) */
    const double corr = nd > 0 ? fmin(4.0 * nd * m, 4.0 * C->lnz + 2.0 * C->dnz) : 0.0;
    return C->flops + nd * (4.0 * C->lnz + 2.0 * m) + 2.0 * nd * C->dnz + nd * nd * nd / 3.0 + 10.0 * (4.0 * C->lnz + 4.0 * (double)nnz + corr);
}
static double ne_iter_cost_of(const LpS *S) {
    LpNeCost C; C.dthr = S->dthr; C.dnz = ne_dense_nnz(S); C.flops = schol_flops(S->sc); C.lnz = (double)schol_nnz(S->sc); C.nd = S->qd && S->bord ? 0 : S->nd; C.ok = 1;   /* (BRISK_LPFREEQD: the border is in the factor's flops) */
    return ne_iter_cost(&C, S->m, S->Ap[S->n]);
}
/* 21 October 2026 (the night): which columns are dense, by cost. The default rule takes a column as dense when it is
 * longer than 10 times the mean, 50 entries and a tenth of the rows; the last bound keeps semi-dense columns in the
 * factor, where on a problem of many rows each one's clique can fill it: ex9 in its dual form (8,479 rows), 94 columns of
 * 100 to 196 entries (the mean 13) made 2.87e7 entries and 1.30e11 flops a factorization; apart (the Woodbury formula),
 * 1.30e7 and 2.5e10, Gurobi's own counts on that form. When the rule without the tenth of the rows finds 1 to 400 columns
 * more than the default, the iteration cost of both choices (ne_iter_cost: the factorization by AMD's count, the dense
 * columns' solves and Schur complement, ten solves) is computed and the looser rule kept when it costs at most 0.7 of
 * the other (a margin for the formula's accuracy, which the method checks and falls back from). Returns the threshold
 * to use, 0 for the default rule. BRISK_LPDENSE2=0: the default rule only. With a budget (the cost model's estimate of
 * a form against the other form's cost; 0: none) both analyses stop at it, a choice above it costing "more than the
 * budget" (22 October 2026, the night: sorrell3 folded, the analysis of its primal form with the 271 columns apart, 4.3e9
 * flops against the dual form's 9e6, took 3.8 s of its 4.5 s). */
static void ne_cost_thr(int m, int n, const int *Ap, const int *Ai, const double *Ax, LpNeCost *C, double budget, double thr, const volatile double *cap);
static int cmp_int_desc(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return x > y ? -1 : x < y; }
static double ne_dense_choice(int m, int n, const int *Ap, const int *Ai, const double *Ax, int verbose, double budget, LpNeCost *out) {
    const char *e = getenv("BRISK_LPDENSE2"); if ((e && atoi(e) == 0) || m < 500 || n < 1) return 0;
    const long nz = Ap[n];
    const double avg = (double)nz / n, thr0 = fmax(fmax(50.0, 10.0 * avg), 0.1 * m), thr1 = fmax(50.0, 10.0 * avg);
    int n0 = 0;
    for (int j = 0; j < n; j++) if (Ap[j + 1] - Ap[j] > thr0) n0++;
    /* (22 October 2026) the candidates: the columns longer than thr1 (as before), and the K longest for K = 32, 64, ...
     * up to LP_NDMAX and a quarter of the columns, each as the threshold "longer than the (K+1)-th longest", columns of
     * at least 3 times the mean length and 10 entries; the cheapest by ne_iter_cost (analyses against a budget: the
     * best so far), taken when it costs at most 0.7 of the default rule's. The cap of 400 dense columns and the single
     * looser threshold missed atlanta-ip (LP): 627 columns of 51 to 1,071 entries, none apart, a factor of 3.1e10 flops
     * against Gurobi's 2.2e9 with 237 apart */
    int *len = xm(sizeof(int) * (size_t)(n + 1));
    for (int j = 0; j < n; j++) len[j] = Ap[j + 1] - Ap[j];
    qsort(len, (size_t)n, sizeof(int), cmp_int_desc);
    double cand[16]; int nc = 0;
    { int n1 = 0; for (int j = 0; j < n; j++) if (len[j] > thr1) n1++;
      if (n1 > n0 && n1 <= LP_NDMAX && n1 < n / 4) cand[nc++] = thr1; }
    for (int K = 32; K <= LP_NDMAX && K < n / 4 && nc < 16; K *= 2) {
        const double t = len[K];                     /* columns longer than the (K+1)-th: at most K of them */
        if (t < fmax(10.0, 3.0 * avg) || len[0] <= t) break;
        int cnt = 0; while (cnt < n && len[cnt] > t) cnt++;
        if (cnt <= n0) continue;
        int dup = 0; for (int c = 0; c < nc; c++) dup |= cand[c] == t;
        if (!dup) cand[nc++] = t;
    }
    free(len);
    if (nc == 0) return 0;
    const double tdc0 = lp_time();
    LpNeCost C0;
    /* (25 October 2026) the default rule's estimate on the threads together with the candidates', each against the outer
     * budget only, when the threads are there: each candidate keeps more columns apart than the default rule, so its
     * pattern is part of the default rule's and its ordering costs no more; the choice is the same (a candidate taken was
     * below 0.7 of the default rule's cost, so its estimate was exact either way). Before, the default rule's estimate
     * came first, alone, for the candidates' budget: rmatr200-p5's dual form (LP), 0.41 s for one candidate. BRISK_LPDCPAR=0:
     * first */
    static int dcpar = -1; if (dcpar < 0) { const char *e_ = getenv("BRISK_LPDCPAR"); dcpar = !(e_ && atoi(e_) == 0); }
    int c0par = 0;
#ifdef _OPENMP
    if (dcpar && !omp_in_parallel() && omp_get_max_threads() > 1) c0par = 1;
#endif
    /* (one thread: the default rule's estimate first, for the candidates' budget; with the shared caps below, last, under
     * the best candidate's cost / 0.7: the candidates keep more columns apart and cost less to estimate) */
    static int dccap0 = -1; if (dccap0 < 0) { const char *e_ = getenv("BRISK_LPDCCAP"); dccap0 = !(e_ && atoi(e_) == 0); }
    if (dcpar && dccap0) c0par = 1;
    if (!c0par) ne_cost_thr(m, n, Ap, Ai, Ax, &C0, budget, 0.0, NULL);
    double c0 = !c0par ? (C0.ok ? ne_iter_cost(&C0, m, nz) : 1e300) : 1e300;
    double best = 1e300, bthr = 0; int bnd = 0, bi = -1;
    /* (23 October 2026) the candidates' estimates on the threads, each against the same budget (0.7 of the default rule's
     * cost; before, also the best so far, which only cut short estimates that lose anyway), the cheapest taken in the
     * candidates' order: the same choice at any thread count. Not when the pattern of A A' may be large (a bound of
     * 4e7 entries over the threads: each estimate holds its pattern and AMD's work) */
    LpNeCost Cc[17]; double cc[17];
    int npar = 1, ntask = nc + c0par;      /* (the default rule's estimate as the last task) */
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        double pb = m; for (int j = 0; j < n; j++) { const double l = Ap[j + 1] - Ap[j]; pb += 0.5 * l * (l - 1); }
        npar = omp_get_max_threads(); if (npar > ntask) npar = ntask;
        while (npar > 1 && (double)npar * pb > 4e7) npar--;
    }
#endif
    if (c0par && npar <= 1 && !dccap0) {    /* (one thread, no shared caps: the default rule's estimate first) */
        c0par = 0; ntask = nc;
        ne_cost_thr(m, n, Ap, Ai, Ax, &C0, budget, 0.0, NULL);
        c0 = C0.ok ? ne_iter_cost(&C0, m, nz) : 1e300;
    }
    double bud = c0par ? (budget > 0 ? budget : 0) : 0.7 * c0; if (!c0par && budget > 0) bud = fmin(bud, budget); if (bud >= 1e299) bud = 0;
    /* (25 October 2026) shared caps, lowered as the estimates end: a candidate is chosen only below every other candidate
     * and 0.7 of the default rule's cost, the default rule only below the best candidate / 0.7, so an AMD count that
     * passes its cap stops (an estimate's flops are at most its iteration's cost; a stopped estimate's lower bound is
     * above the cap, so it is not chosen, as its exact value would not be). The choice is the same, whatever ends first;
     * only the costs printed for the losers are then lower bounds. savsched1's dual form (LP): the default rule's
     * estimate counted 4.8e14 flops in 48 s after the best candidate had shown 5.6e10. BRISK_LPDCCAP=0: no shared caps */
    static int dccap = -1; if (dccap < 0) { const char *e_ = getenv("BRISK_LPDCCAP"); dccap = !(e_ && atoi(e_) == 0); }
    volatile double cap0 = budget > 0 ? budget : 0, capc = bud;
    const double dl0 = ne_dl; const volatile int *sf0 = ne_stopf;
    /* (31 October 2026) on one thread, the candidate that keeps the most columns apart first: the others and the default
     * rule keep more columns in the factor, so their patterns contain its pattern. When its estimate stops on a bound from
     * the pattern alone (the clique of its longest column, or the pattern's entries w with w^2 / m above the budget), the
     * same bound holds for every other choice, which costs more than the budget too: they are not estimated (each would
     * build its larger pattern up to the same bound). The primal form of an SVM LP (19,600 rows, the dual form taken):
     * 7 candidates built their patterns for 3.9 s; the choice is the same (every one above the budget). BRISK_LPDCNEST=0:
     * all estimated */
    int skip_rest = 0, cfirst = -1;
    static int dcnest = -1; if (dcnest < 0) { const char *e_ = getenv("BRISK_LPDCNEST"); dcnest = !(e_ && atoi(e_) == 0); }
    if (npar <= 1 && dcnest && bud > 0 && nc > 1) {
        cfirst = 0; for (int c = 1; c < nc; c++) if (cand[c] < cand[cfirst]) cfirst = c;
        if (!ne_expired()) {
            ne_cost_thr(m, n, Ap, Ai, Ax, &Cc[cfirst], bud, cand[cfirst], dccap ? &capc : NULL);
            cc[cfirst] = Cc[cfirst].ok ? ne_iter_cost(&Cc[cfirst], m, nz) : 1e300;
            if (dccap && cc[cfirst] < 1e299 && (capc <= 0 || cc[cfirst] < capc)) capc = cc[cfirst];
            if (dccap && cc[cfirst] < 1e299 && (cap0 <= 0 || cc[cfirst] / 0.7 < cap0)) cap0 = cc[cfirst] / 0.7;
            skip_rest = Cc[cfirst].ok && Cc[cfirst].pat && Cc[cfirst].flops > bud && (!c0par || budget <= bud);
            if (skip_rest && getenv("BRISK_LPDBG")) printf("   [dense columns: the candidate keeping the most apart is above the budget by its pattern (%.3g > %.3g): the other choices not estimated]\n", Cc[cfirst].flops, bud);
        } else { Cc[cfirst].ok = 0; Cc[cfirst].flops = Cc[cfirst].lnz = 1e300; Cc[cfirst].nd = 0; Cc[cfirst].dnz = 0; Cc[cfirst].dthr = 0; cc[cfirst] = 1e300; }
    }
    #pragma omp parallel for num_threads(npar) schedule(dynamic, 1) if (npar > 1)
    for (int k = 0; k < ntask; k++) {
        const int c = npar > 1 ? ntask - 1 - k : k;   /* (on the threads the default rule's first: it may take the longest) */
        if (c == cfirst) continue;
        if (skip_rest) { LpNeCost *E = c == nc ? &C0 : &Cc[c]; E->ok = 0; E->flops = E->lnz = 1e300; E->nd = 0; E->dnz = 0; E->dthr = 0; E->pat = 1; if (c != nc) cc[c] = 1e300; continue; }
        ne_dl = dl0; ne_stopf = sf0;
        if (ne_expired()) { LpNeCost *E = c == nc ? &C0 : &Cc[c]; E->ok = 0; E->flops = E->lnz = 1e300; E->nd = 0; E->dnz = 0; E->dthr = 0; if (c != nc) cc[c] = 1e300; continue; }
        if (c == nc) {
            ne_cost_thr(m, n, Ap, Ai, Ax, &C0, budget, 0.0, dccap ? &cap0 : NULL);
            if (dccap && C0.ok) {
                const double v = 0.7 * ne_iter_cost(&C0, m, nz);
                #pragma omp critical(lp_dccap)
                { if (capc <= 0 || v < capc) capc = v; }
            }
            continue;
        }
        ne_cost_thr(m, n, Ap, Ai, Ax, &Cc[c], bud, cand[c], dccap ? &capc : NULL);
        cc[c] = Cc[c].ok ? ne_iter_cost(&Cc[c], m, nz) : 1e300;
        if (dccap && cc[c] < 1e299) {
            #pragma omp critical(lp_dccap)
            { if (capc <= 0 || cc[c] < capc) capc = cc[c];
              const double v0 = cc[c] / 0.7; if (cap0 <= 0 || v0 < cap0) cap0 = v0; }
        }
    }
    if (c0par) c0 = C0.ok ? ne_iter_cost(&C0, m, nz) : 1e300;
    for (int c = 0; c < nc; c++) {
        if (getenv("BRISK_LPDBG") && atoi(getenv("BRISK_LPDBG")) > 1) printf("   [dense candidate: longer than %.0f, %d columns, an iteration %.2e flops (budget %.2e)]\n", cand[c], Cc[c].nd, cc[c], bud);
        if (cc[c] < best) { best = cc[c]; bthr = cand[c]; bnd = Cc[c].nd; bi = c; }
    }
    if (verbose > 0 || getenv("BRISK_LPDBG")) printf("LP dense columns: %d candidates, the best %d columns longer than %.0f entries apart: an iteration %.2e flops; %d by the default rule: %.2e%s (%.2f s)\n", nc, bnd, bthr, best, n0, c0, best <= 0.7 * c0 ? " (the former taken)" : "", lp_time() - tdc0);
    if (out) *out = best <= 0.7 * c0 ? Cc[bi] : C0;     /* (the estimate of the choice, for ne_cost) */
    return best <= 0.7 * c0 ? bthr : 0;
}
static void ne_cost_thr(int m, int n, const int *Ap, const int *Ai, const double *Ax, LpNeCost *C, double budget, double thr, const volatile double *cap) {
    LpS T; memset(&T, 0, sizeof T); T.m = m; T.n = n; T.est = 1; T.est_budget = budget; T.dthr = thr; T.est_cap = cap;
    T.Ap = (int *)Ap; T.Ai = (int *)Ai; T.Ax = (double *)Ax;
    C->ok = ne_setup(&T, 1) == 0;
    C->dnz = ne_dense_nnz(&T); C->pat = T.est_pat;
    if (C->ok && T.est_lb > 0) { C->flops = T.est_lb; C->lnz = (double)T.est_w; C->nd = T.nd; T.Ap = NULL; T.Ai = NULL; T.Ax = NULL; lps_free(&T); return; }
    C->flops = C->ok ? schol_flops(T.sc) : 1e300; C->lnz = C->ok ? (double)schol_nnz(T.sc) : 1e300; C->nd = T.nd;
    T.Ap = NULL; T.Ai = NULL; T.Ax = NULL;
    lps_free(&T);
}
static void sort_cols(int n, const int *Ap, int *Ai, double *Ax) {
    for (int j = 0; j < n; j++) {
        const int a = Ap[j], e = Ap[j + 1];
        for (int p = a + 1; p < e; p++) { const int r = Ai[p]; const double v = Ax[p]; int q = p - 1; while (q >= a && Ai[q] > r) { Ai[q + 1] = Ai[q]; Ax[q + 1] = Ax[q]; q--; } Ai[q + 1] = r; Ax[q + 1] = v; }
    }
}
static int g_iterflops_ns = 0, g_iterflops_nd = 0;
/* (2 November 2026) the rows' positions in the fill-reducing ordering of the normal equations with the dense columns
 * apart (the analysis lpipm_iter_flops makes); 0 when it fails */
static int lp_rows_position(int m, int n, const int *Ap, const int *Ai, const double *Ax, int *pos) {
    if (m <= 0 || n <= 0) return 0;
    const long nz = Ap[n];
    int *Ai1 = xm(sizeof(int) * (nz + 1)); double *Ax1 = xm(sizeof(double) * (nz + 1));
    memcpy(Ai1, Ai, sizeof(int) * nz); memcpy(Ax1, Ax, sizeof(double) * nz); sort_cols(n, Ap, Ai1, Ax1);
    LpS T; memset(&T, 0, sizeof T); T.m = m; T.n = n; T.Ap = (int *)Ap; T.Ai = Ai1; T.Ax = Ax1;
    T.dthr = ne_dense_choice(m, n, Ap, Ai1, Ax1, 0, 0.0, NULL);
    const int ok = ne_setup(&T, 1) == 0 && T.sc;
    if (ok) { const int *ip = schol_iperm(T.sc); for (int i = 0; i < m; i++) pos[i] = ip[i]; }
    T.Ap = NULL; T.Ai = NULL; T.Ax = NULL; lps_free(&T); free(Ai1); free(Ax1);
    return ok;
}     /* (the last estimate's supernodes and dense columns, for the column splitting's choice) */
double lpipm_iter_flops(int m, int n, const int *Ap, const int *Ai, const double *Ax, double budget) {
    if (m <= 0 || n <= 0) return 0;
    const long nz = Ap[n];
    int *Ai1 = xm(sizeof(int) * (nz + 1)); double *Ax1 = xm(sizeof(double) * (nz + 1));
    memcpy(Ai1, Ai, sizeof(int) * nz); memcpy(Ax1, Ax, sizeof(double) * nz); sort_cols(n, Ap, Ai1, Ax1);
    LpS T; memset(&T, 0, sizeof T); T.m = m; T.n = n;
    T.Ap = (int *)Ap; T.Ai = Ai1; T.Ax = Ax1;
    T.dthr = ne_dense_choice(m, n, Ap, Ai1, Ax1, 0, budget > 0 ? budget : 0.0, NULL);
    double c = 1e300;
    /* (with a budget: the analysis stops once the factorization's flops pass it; the form then costs "more than it") */
    if (budget > 0) schol_set_flopcap(budget);
    const int ok = ne_setup(&T, 1) == 0 && T.sc;
    if (budget > 0) schol_set_flopcap(0);
    if (ok) c = ne_iter_cost_of(&T);
    g_iterflops_ns = ok ? schol_nsuper(T.sc) : 0; g_iterflops_nd = ok ? T.nd : 0;
    T.Ap = NULL; T.Ai = NULL; T.Ax = NULL; lps_free(&T); free(Ai1); free(Ax1);
    return c;
}
/* (22 October 2026, the night) *dthr_p: when the primal form is kept by the cost model, its dense-column choice (> 0 the
 * threshold, < 0 the default rule; 0 not made), for the caller's solve of the primal (LpIpmOpts.dthr): the primal is
 * kept only below its budget, where its estimate is exact */
static int lp_dual_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                         const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *dthr_p) {
    if (dthr_p) *dthr_p = 0;
    const double tds0 = lp_time();
    const char *e = getenv("BRISK_LPDUAL"); const int mode = e ? atoi(e) : 2;      /* 0: never, 1: always, 2: by the cost model */
    if (mode == 0 || m < 1 || n < 1) return -1;
    const double BIG = 1e29;
    int *rpos = xm(sizeof(int) * (n + 1)), nr = 0;
    double *lo = xm(sizeof(double) * (m + 1)), *up = xm(sizeof(double) * (m + 1)); int *jlo = xm(sizeof(int) * (m + 1)), *jup = xm(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) { lo[i] = -1e300; up[i] = 1e300; jlo[i] = jup[i] = -1; }
    for (int j = 0; j < n; j++) {
        const int len = Ap[j + 1] - Ap[j], bnd = u && u[j] < BIG;
        if (len == 1 && j >= nfree && !bnd && Ax[Ap[j]] != 0.0) {
            const int i = Ai[Ap[j]]; const double a = Ax[Ap[j]], v = c[j] / a;
            if (a > 0) { if (v < up[i]) { up[i] = v; jup[i] = j; } } else { if (v > lo[i]) { lo[i] = v; jlo[i] = j; } }
            rpos[j] = -1;
        } else rpos[j] = nr++;
    }
    int bad = 0; for (int i = 0; i < m; i++) if (lo[i] > up[i]) bad = 1;
    const long nnz = Ap[n];
    if (bad || (mode == 2 && nr >= m)) { free(rpos); free(lo); free(up); free(jlo); free(jup); return -1; }
    /* the dual's columns: y (free ones first), z_j (j in R, not free), w_j (j in R, bounded) */
    int nyf = 0; for (int i = 0; i < m; i++) if (lo[i] < -1e299 && up[i] > 1e299) nyf++;
    int nz_ = 0, nw = 0; for (int j = 0; j < n; j++) if (rpos[j] >= 0) { if (j >= nfree) nz_++; if (u && u[j] < BIG) nw++; }
    const int n2 = m + nz_ + nw, m2 = nr;
    int *ycol = xm(sizeof(int) * (m + 1)); double *ysgn = xm(sizeof(double) * (m + 1)), *ys0 = xm(sizeof(double) * (m + 1));
    { int kf = 0, kb = nyf; for (int i = 0; i < m; i++) {
        const int fr = lo[i] < -1e299 && up[i] > 1e299;
        ycol[i] = fr ? kf++ : kb++;
        if (fr) { ysgn[i] = 1; ys0[i] = 0; } else if (lo[i] > -1e299) { ysgn[i] = 1; ys0[i] = lo[i]; } else { ysgn[i] = -1; ys0[i] = up[i]; }
    } }
    /* A by rows (the y columns of the dual), restricted to R */
    int *Rp = xz(sizeof(int) * (m + 2));
    for (int j = 0; j < n; j++) if (rpos[j] >= 0) for (int p = Ap[j]; p < Ap[j + 1]; p++) Rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    const long nzr = Rp[m + 1];
    const long nz2 = nzr + nz_ + nw;
    int *Bp = xm(sizeof(int) * (n2 + 1)), *Bi = xm(sizeof(int) * (nz2 + 1)); double *Bx = xm(sizeof(double) * (nz2 + 1));
    double *b2 = xm(sizeof(double) * (m2 + 1)), *c2 = xm(sizeof(double) * (n2 + 1)), *u2 = xm(sizeof(double) * (n2 + 1));
    {   /* the y columns: entries in row order (ascending r: j ascending) */
        int *cnt = xz(sizeof(int) * (m + 1));
        for (int j = 0; j < n; j++) if (rpos[j] >= 0) for (int p = Ap[j]; p < Ap[j + 1]; p++) cnt[Ai[p]]++;
        int *ord = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) ord[ycol[i]] = i;
        long w = 0;
        int *fill = xm(sizeof(int) * (m + 1));
        for (int k = 0; k < m; k++) { const int i = ord[k]; Bp[k] = (int)w; fill[i] = (int)w; w += cnt[i]; }
        for (int j = 0; j < n; j++) if (rpos[j] >= 0) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int i = Ai[p]; Bi[fill[i]] = rpos[j]; Bx[fill[i]++] = ysgn[i] * Ax[p]; }
        for (int k = 0; k < m; k++) { const int i = ord[k]; c2[k] = -ysgn[i] * b[i]; u2[k] = lo[i] > -1e299 && up[i] < 1e299 ? up[i] - lo[i] : 1e30; }
        int k = m;
        for (int j = 0; j < n; j++) if (rpos[j] >= 0 && j >= nfree) { Bp[k] = (int)w; Bi[w] = rpos[j]; Bx[w++] = 1.0; c2[k] = 0; u2[k] = 1e30; k++; }
        for (int j = 0; j < n; j++) if (rpos[j] >= 0 && u && u[j] < BIG) { Bp[k] = (int)w; Bi[w] = rpos[j]; Bx[w++] = -1.0; c2[k] = u[j]; u2[k] = 1e30; k++; }
        Bp[n2] = (int)w;
        free(cnt); free(ord); free(fill);
        for (int j = 0; j < n; j++) if (rpos[j] >= 0) { double v = c[j]; for (int p = Ap[j]; p < Ap[j + 1]; p++) v -= Ax[p] * ys0[Ai[p]]; b2[rpos[j]] = v; }
    }
    free(Rp);
    int choose = mode == 1;
    double cp = 0, cd = 0, dthr_d = 0;
    if (mode == 2) {
        /* the cost model: one iteration of each form, from the symbolic factorizations (AMD). The dual's first: the
         * primal's analysis is skipped when the assembly of its normal equations alone, sum over its sparse columns of
         * len (len + 1) / 2 multiply-adds (a lower bound on its iteration), costs more than twice the dual's iteration
         * (a sparse quantile regression 30,000 x 501: the primal's analysis took 22-55 s for a solve of 0.27 s); and its
         * ordering is not made when its pattern alone shows that it costs more (ne_setup's budget) */
        LpNeCost P, D;
        /* the assembly of each form's normal equations, a lower bound on its iteration (its sparse columns' len (len + 1) / 2
         * multiply-adds) and on its estimate's work (the pattern) */
        double lbp = 0, lbd = 0;
        {   const double avg = (double)nnz / fmax(n, 1), thr = fmax(fmax(50.0, 10.0 * avg), 0.1 * m);
            int ndn = 0; for (int j = 0; j < n; j++) if (m >= 500 && Ap[j + 1] - Ap[j] > thr) ndn++;
            const int usedn = ndn > 0 && ndn <= 400 && ndn < n / 4;
            for (int j = 0; j < n; j++) { const double l = Ap[j + 1] - Ap[j]; if (!(usedn && l > thr)) lbp += 0.5 * l * (l + 1); } }
        {   const double avg = (double)nz2 / fmax(n2, 1), thr = fmax(fmax(50.0, 10.0 * avg), 0.1 * m2);
            int ndn = 0; for (int j = 0; j < n2; j++) if (m2 >= 500 && Bp[j + 1] - Bp[j] > thr) ndn++;
            const int usedn = ndn > 0 && ndn <= 400 && ndn < n2 / 4;
            for (int j = 0; j < n2; j++) { const double l = Bp[j + 1] - Bp[j]; if (!(usedn && l > thr)) lbd += 0.5 * l * (l + 1); } }
        /* (28 October 2026, the night; ISSUES 621) the form whose bound is smaller first, the other under the budget the rule
         * below sets from it (the dual is chosen at most half the primal's cost: the primal under twice the dual's, the dual
         * under half the primal's); the choice is the same in either order, only the work of the losing estimate differs.
         * The dual's first, unbounded, took 5.4 s of neos-5106984-jizera (LP) and most of z26's 27.5 s, whose primal estimates
         * were cheap. On the 17 LPs where the choice of form has mattered (formorder2, the interior-point method alone, no
         * time limit): the same choice on every one, the estimates of savsched1 68.9 -> 7.6 s and of jizera 32.4 -> 27.2 s, the
         * others within 0.6 s. BRISK_LPFORMORDER=0: the dual's first, as before */
        static int ford = -1; if (ford < 0) { const char *e_ = getenv("BRISK_LPFORMORDER"); ford = !(e_ && atoi(e_) == 0); }
        const int pfirst = ford && lbp < lbd;
        int *Ai1 = NULL; double *Ax1 = NULL;
        const double tds1 = lp_time();
        P.dthr = 0; P.ok = 0; D.ok = 0; D.dthr = 0;
        if (pfirst) {
            Ai1 = xm(sizeof(int) * (nnz + 1)); Ax1 = xm(sizeof(double) * (nnz + 1));
            memcpy(Ai1, Ai, sizeof(int) * nnz); memcpy(Ax1, Ax, sizeof(double) * nnz); sort_cols(n, Ap, Ai1, Ax1);
            ne_cost(m, n, Ap, Ai1, Ax1, &P, 0.0); free(Ai1); free(Ax1);
            cp = ne_iter_cost(&P, m, nnz);
            if (P.ok && lbd > 0.5 * cp) { D.ok = 0; cd = lbd; }
            else { ne_cost(m2, n2, Bp, Bi, Bx, &D, P.ok ? 0.5 * cp : 0.0); cd = ne_iter_cost(&D, m2, nz2); }
            dthr_d = D.ok && D.dthr > 0 ? D.dthr : -1.0;
        } else {
            ne_cost(m2, n2, Bp, Bi, Bx, &D, 0.0);
            cd = ne_iter_cost(&D, m2, nz2); dthr_d = D.dthr > 0 ? D.dthr : -1.0;
            if (D.ok && lbp > 2.0 * cd) { P.ok = 0; cp = lbp; }
            else {
                Ai1 = xm(sizeof(int) * (nnz + 1)); Ax1 = xm(sizeof(double) * (nnz + 1));
                memcpy(Ai1, Ai, sizeof(int) * nnz); memcpy(Ax1, Ax, sizeof(double) * nnz); sort_cols(n, Ap, Ai1, Ax1);
                ne_cost(m, n, Ap, Ai1, Ax1, &P, D.ok ? 2.0 * cd : 0.0); free(Ai1); free(Ax1);
                cp = ne_iter_cost(&P, m, nnz);
            }
        }
        const double tds2 = lp_time();
        if (getenv("BRISK_LPDBG")) printf("   [the forms' costs: the dual built %.3fs, the estimates %.3fs (the %s first; bounds %.3g and %.3g); dual: ok %d flops %.3g lnz %.3g nd %d dnz %.3g; primal: ok %d flops %.3g lnz %.3g nd %d dnz %.3g]\n",
                                          tds1 - tds0, tds2 - tds1, pfirst ? "primal's" : "dual's", lbp, lbd, D.ok, D.ok ? D.flops : cd, D.ok ? D.lnz : 0.0, D.ok ? D.nd : 0, D.ok ? D.dnz : 0.0, P.ok, P.ok ? P.flops : cp, P.ok ? P.lnz : 0.0, P.ok ? P.nd : 0, P.ok ? P.dnz : 0.0);
        /* the dual when it costs at most half: the primal is the form the method is tuned on, and its free
         * variables (the dual's free y) enter both as a proximal term */
        choose = D.ok && cd <= 0.5 * cp;
        /* (2 November 2026) a close choice between large forms checked with the ordering the factorization uses: both forms
         * again by the nested dissection at their chosen dense-column thresholds, when the losing form's estimate is within
         * 16 times the winner's and one of them is above 1e9 flops (where the dissection replaces AMD). genus-sym-g62-2 (LP):
         * the primal form chosen at 3.7e10 against the dual's 4.2e10 (AMD), the dual's real factor 1.7e9 bordered; a permuted
         * copy chose the dual by AMD's other count (2.85 against 13.2 s). BRISK_LPFORMND=0: AMD's choice */
        {   static int fnd = -1; if (fnd < 0) { const char *e_ = getenv("BRISK_LPFORMND"); fnd = !(e_ && atoi(e_) == 0); }
            const double lo = fmin(cp, cd), hi = fmax(cp, cd);
            /* only when the dual form lost: it is the smaller form, its dissection cheap; the primal again only when the dual then
             * looks at most half as costly, under twice the dual's cost (a primal above it stops at its pattern's bound). When
             * the dual won, a check could only bring back the larger form, whose estimate is the costly one (a permuted copy of
             * genus-sym-g62-2: 1.7 s for nothing) */
            if (fnd && !choose && P.ok && D.ok && hi < 16.0 * lo && fmax(P.flops, D.flops) >= 1e9 && m >= 1000 && m2 >= 1000 && !ne_expired()) {
                /* (its time: a tenth of the estimates', at least 0.5 s; with the estimates' whole time the dual form of
                 * neos-2075418-temuka, LP, 122,304 rows, took 30 s to fail: 101.8 -> 125.6 s) */
                const double tn0 = lp_time(), dl0 = ne_dl, tb = fmax(0.5, 0.1 * (tds2 - tds1));
                ne_dl = ne_dl > 0 ? fmin(ne_dl, tn0 + tb) : tn0 + tb;
                LpNeCost D2, P2; double cd2 = 1e300, cp2 = cp;
                ne_cost_nd(m2, n2, Bp, Bi, Bx, D.dthr, cp, &D2); if (D2.ok) cd2 = ne_iter_cost(&D2, m2, nz2);
                if (cd2 <= 0.5 * cp && !ne_expired()) {
                    int *Ai2 = xm(sizeof(int) * (nnz + 1)); double *Ax2 = xm(sizeof(double) * (nnz + 1));
                    memcpy(Ai2, Ai, sizeof(int) * nnz); memcpy(Ax2, Ax, sizeof(double) * nnz); sort_cols(n, Ap, Ai2, Ax2);
                    ne_cost_nd(m, n, Ap, Ai2, Ax2, P.dthr, 2.0 * cd2, &P2); free(Ai2); free(Ax2);
                    cp2 = P2.ok ? ne_iter_cost(&P2, m, nnz) : fmax(cp, 2.0 * cd2);      /* (above its budget: at least twice the dual's) */
                    cp = cp2; cd = cd2; choose = cd <= 0.5 * cp;
                }
                if (getenv("BRISK_LPDBG")) printf("   [the forms by the dissection: dual %.3g, primal %.3g (AMD: dual %.3g); %.3fs]\n", cd2, cp2, D.ok ? ne_iter_cost(&D, m2, nz2) : 0.0, lp_time() - tn0);
                ne_dl = dl0;
            }
        }
        if (!choose && P.ok && dthr_p) *dthr_p = P.dthr > 0 ? P.dthr : -1.0;
    }
    if (opt->verbose > 0 && (choose || getenv("BRISK_LPDBG")))
        printf("LP dual form %s: %d rows instead of %d (%d bounds on y from singleton columns, %d free y); cost of an iteration %.2e against %.2e flops\n",
               choose ? "chosen" : "not chosen", m2, m, m - nyf, nyf, cd, cp);
    int st = -1;
    if (choose) {
        double *x2 = xz(sizeof(double) * (n2 + 1)), *y2 = xz(sizeof(double) * (m2 + 1)), *zl2 = xz(sizeof(double) * (n2 + 1)), *wu2 = xz(sizeof(double) * (n2 + 1));
        LpIpmInfo inf2;
        /* (the dual's objective is -(the primal's) + b'ys0: the objective as given is objoff + b'ys0 - (the dual's)) */
        LpIpmOpts od = *opt; { double bys = 0; for (int i = 0; i < m; i++) bys += b[i] * ys0[i]; od.objoff = -(bys + opt->objoff); }
        od.dthr = dthr_d;    /* (the dense columns as the cost model chose them on this matrix: not again) */
        st = lpipm_core(m2, n2, nyf, Bp, Bi, Bx, b2, c2, u2, &od, x2, y2, NULL, &inf2, zl2, wu2);
        if (st == 2) { st = 4; inf2.status = 4; }     /* (a ray of the dual form: the primal infeasible; the homogeneous model decides) */
        /* the primal point */
        for (int i = 0; i < m; i++) yout[i] = ys0[i] + ysgn[i] * x2[ycol[i]];
        for (int j = 0; j < n; j++) xout[j] = rpos[j] >= 0 ? -y2[rpos[j]] : 0.0;
        for (int i = 0; i < m; i++) {
            const int k = ycol[i];
            const double zt = zl2[k], wt = wu2[k];
            if (ysgn[i] > 0) {
                if (jlo[i] >= 0) xout[jlo[i]] = zt / -Ax[Ap[jlo[i]]];
                if (jup[i] >= 0) xout[jup[i]] = wt / Ax[Ap[jup[i]]];
            } else if (jup[i] >= 0) xout[jup[i]] = zt / Ax[Ap[jup[i]]];
        }
        /* the errors of the primal form (as the primal method measures them) */
        double *r = xz(sizeof(double) * (m + 1));
        for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) r[Ai[p]] += Ax[p] * xout[j];
        double ep = 0, ed = 0, pobj = 0, dobj = 0;
        for (int i = 0; i < m; i++) { ep = fmax(ep, fabs(b[i] - r[i])); dobj += b[i] * yout[i]; }
        for (int j = 0; j < n; j++) {
            double s = c[j]; for (int p = Ap[j]; p < Ap[j + 1]; p++) s -= Ax[p] * yout[Ai[p]];
            if (zout) zout[j] = s;
            pobj += c[j] * xout[j];
            if (j < nfree) ed = fmax(ed, fabs(s));
            else if (u && u[j] < BIG) { if (s < 0) dobj += u[j] * s; ep = fmax(ep, fmax(0.0, xout[j] - u[j])); ep = fmax(ep, -xout[j]); }
            else { ed = fmax(ed, -s); ep = fmax(ep, -xout[j]); }
        }
        free(r);
        *info = inf2;
        info->pobj = pobj; info->dobj = dobj;
        info->pinf = ep / (1.0 + nrminf(m, b)); info->dinf = ed / (1.0 + nrminf(n, c)); info->gap = fabs(pobj - dobj) / (1.0 + fabs(pobj + opt->objoff));
        free(x2); free(y2); free(zl2); free(wu2);
    }
    free(rpos); free(lo); free(up); free(jlo); free(jup); free(ycol); free(ysgn); free(ys0);
    free(Bp); free(Bi); free(Bx); free(b2); free(c2); free(u2);
    return st;
}
/* ---- 20 October 2026 (LP/SOCP): folding by the coarsest equitable partition ----
 * Rows and columns are coloured (rows by b_i, columns by c_j, u_j and whether they are free) and the colouring is
 * refined until it is stable: a row's new colour is its colour with the multiset of (a_ij, colour of j) over its
 * entries, a column's likewise. The stable colouring is an equitable partition of the weighted bipartite graph of
 * A: for every row class I_l and column class J_k the sums Ahat_lk = sum_{j in J_k} a_ij are the same for every
 * i in I_l, and B_kl = sum_{i in I_l} a_ij the same for every j in J_k, with n_k B_kl = m_l Ahat_lk (both are the
 * sum over the block). Averaging a feasible x over each column class keeps it feasible with the same objective
 * (sum the rows of a class), so the LP in the class averages X,
 *     min sum_k n_k c_k X_k,   sum_k Ahat_lk X_k = b_l,   X free / 0 <= X <= u / X >= 0 by class,
 * has the same optimal value, and x_j = X_k is optimal and interior when X is; the duals come back as
 * y_i = Y_l / m_l and z_j = Z_k / n_k. (The quotient is what MOSEK's presolve calls folding; a QAP linearization
 * folds to 0.28 of its size and its factor to a few percent of the work.) The hashes of the refinement are
 * only a guide: the sums are verified before the fold is taken. BRISK_LPNOFOLD=1 turns it off. */
static uint64_t fold_mix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static uint64_t fold_dbl(double v) { uint64_t u; if (v == 0.0) v = 0.0; memcpy(&u, &v, 8); return fold_mix(u); }
static _Thread_local const uint64_t *fold_keys_g;   /* (per thread: the comparator's keys, ISSUES 454) */
static int fold_cmp(const void *a, const void *b) { const uint64_t ka = fold_keys_g[*(const int *)a], kb = fold_keys_g[*(const int *)b]; return ka < kb ? -1 : ka > kb ? 1 : (*(const int *)a < *(const int *)b ? -1 : 1); }
/* classes 0..k-1 by increasing key; returns k */
static int fold_relabel(int n, const uint64_t *keys, int *cls, int *idx) {
    for (int i = 0; i < n; i++) idx[i] = i;
    fold_keys_g = keys; qsort(idx, (size_t)n, sizeof(int), fold_cmp);
    int k = 0;
    for (int t = 0; t < n; t++) { if (t > 0 && keys[idx[t]] != keys[idx[t - 1]]) k++; cls[idx[t]] = k; }
    return n > 0 ? k + 1 : 0;
}
/* ---- 22 October 2026 (the night): the same partition by incremental refinement (lp_fold_partition_inc). Each pass of
 * lp_fold_partition hashes every row and column and sorts all the keys; a refinement that splits a little at a time takes
 * hundreds of passes (neos-631710 (LP): 280 passes, 12.6 s on the problem as read and 19.1 s on its standard form, for a
 * solve of 0.06 s; chromaticindex512-7 (LP): 1,029 passes, 9.2 and 12.4 s). Here a row's key is the sum of mix(a_ij,
 * class of column j) over its entries, kept up to date: when a column changes class only its rows' keys change, and only
 * the classes of the touched rows are split (the touched members by their keys; the untouched ones keep the class's old
 * key and its number). Alternating over rows and columns until nothing splits gives the coarsest equitable refinement of
 * the start (rows by b; columns by cost, bound and freedom), the partition of lp_fold_partition; the numbering of the
 * classes differs. The work of a pass is that of the entries of the elements that changed. */
typedef struct { uint64_t key; int cls, el; } FoldEnt;
static int fold_ent_cmp(const void *a, const void *b) {
    const FoldEnt *x = a, *y = b;
    if (x->cls != y->cls) return x->cls < y->cls ? -1 : 1;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    return x->el < y->el ? -1 : x->el > y->el;
}
static inline uint64_t fold_term(double v, int k, uint64_t salt) { return fold_mix(fold_dbl(v) ^ fold_mix((uint64_t)k + salt)); }
/* split the classes of the touched elements T (nt of them) by their keys H; cls, sz, ckey per class; the elements whose
 * class changed go to D with their old class in Dold; returns the new number of classes */
static int fold_split(int nt, const int *T, const uint64_t *H, int *cls, int *sz, uint64_t *ckey, int ncls, FoldEnt *buf, int *D, int *Dold, int *nd) {
    for (int t = 0; t < nt; t++) { buf[t].key = H[T[t]]; buf[t].cls = cls[T[t]]; buf[t].el = T[t]; }
    qsort(buf, (size_t)nt, sizeof(FoldEnt), fold_ent_cmp);
    int t = 0;
    while (t < nt) {
        const int k = buf[t].cls; int e = t; while (e < nt && buf[e].cls == k) e++;      /* the touched members of class k: [t, e) */
        const int untouched = sz[k] - (e - t);
        /* the runs of equal keys; the one that keeps the class's number: the untouched members' (key ckey[k]), else the longest */
        int keep = -1, best = -1;
        for (int a = t; a < e; ) { int b = a; while (b < e && buf[b].key == buf[a].key) b++;
            if (untouched > 0) { if (buf[a].key == ckey[k]) keep = a; }
            else if (b - a > best) { best = b - a; keep = a; }
            a = b; }
        int kept = untouched;
        for (int a = t; a < e; ) { int b = a; while (b < e && buf[b].key == buf[a].key) b++;
            if (a == keep) { kept += b - a; if (untouched == 0) ckey[k] = buf[a].key; }
            else { const int knew = ncls++; sz[knew] = b - a; ckey[knew] = buf[a].key;
                   for (int q = a; q < b; q++) { const int el = buf[q].el; cls[el] = knew; D[*nd] = el; Dold[*nd] = k; (*nd)++; } }
            a = b; }
        sz[k] = kept;
        t = e;
    }
    return ncls;
}
int lp_fold_partition_inc(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                          int *rcl, int *ccl, int *nrc, int *ncc) {
    return lp_fold_partition_stop(m, n, nfree, Ap, Ai, Ax, b, c, u, rcl, ccl, nrc, ncc, 0.0);
}
const volatile int *lp_get_stop(void);      /* (lpio.c: the stop flag of the calling thread) */
int lp_fold_partition_stop(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                           int *rcl, int *ccl, int *nrc, int *ncc, double stop) {
    /* (4 November 2026) the calling thread's stop flag read at every pass: the default route's main thread folds the problem
     * as read and the interior-point method's problems while the dual simplex method beside it may have its result already;
     * a stopped partition is "no fold". ns1111636 (LP): the dual simplex method done at 0.52 s, the run returned at 0.82 s
     * after three foldings. BRISK_LPFOLDNOSTOP=1: not read */
    static int fns = -1; if (fns < 0) fns = getenv("BRISK_LPFOLDNOSTOP") != NULL;
    const volatile int *sflag = fns ? NULL : lp_get_stop(), *sflag2 = fns ? NULL : ne_stopf;   /* (ne_stopf: inside lpipm_solve, its caller's stop) */
    const int nz = Ap[n];
    int *Rp = NULL, *Rj = NULL; double *Rx = NULL;
    const int mn = m > n ? m : n;
    uint64_t *kr = xz(sizeof(uint64_t) * (m + 1)), *kc = xz(sizeof(uint64_t) * (n + 1)), *ckr = xz(sizeof(uint64_t) * (m + 1)), *ckc = xz(sizeof(uint64_t) * (n + 1));
    int *idx = xz(sizeof(int) * mn + 4), *szr = xz(sizeof(int) * (m + 1)), *szc = xz(sizeof(int) * (n + 1));
    int *Dr = xm(sizeof(int) * (m + 1)), *Dro = xm(sizeof(int) * (m + 1)), *Dc = xm(sizeof(int) * (n + 1)), *Dco = xm(sizeof(int) * (n + 1));
    int *T = xz(sizeof(int) * mn + 4); char *mk = xz((size_t)mn + 1);
    FoldEnt *buf = xm(sizeof(FoldEnt) * ((size_t)mn + 1));
    const uint64_t SR = 0x1234567ull, SC = 0x7654321ull;
    /* the start: rows by b, columns by cost, bound and freedom (as lp_fold_partition) */
    for (int i = 0; i < m; i++) kr[i] = fold_dbl(b[i]);
    for (int j = 0; j < n; j++) kc[j] = fold_mix(fold_dbl(c[j]) ^ fold_mix(fold_dbl(u && u[j] < 1e30 ? u[j] : 1e300)) ^ (j < nfree ? 0x5bd1e995ull : 0));
    int nr = fold_relabel(m, kr, rcl, idx), nc = fold_relabel(n, kc, ccl, idx);
    /* (22 October 2026) the stop before any pass over the matrix: lp_cheb_10000x300 (6.04 million nonzeros) has 20,000
     * distinct right-hand sides, and the command line's check cost 0.35 s for that answer (ISSUES 421) */
    int stopped = (stop > 0 && (double)nr + nc > stop * ((double)m + n)) || (sflag && *sflag) || (sflag2 && *sflag2);
    int pass = 0, ndr = 0, ndc = 0; double tr0 = lp_time();
    if (stopped) goto done;
    Rp = xz(sizeof(int) * (m + 2)); Rj = xm(sizeof(int) * (nz + 1)); Rx = xm(sizeof(double) * (nz + 1));
    for (int p = 0; p < nz; p++) Rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int q = Rp[Ai[p] + 1]++; Rj[q] = j; Rx[q] = Ax[p]; }
    for (int i = 0; i < m; i++) szr[rcl[i]]++;
    for (int j = 0; j < n; j++) szc[ccl[j]]++;
    /* the first passes over everything: the keys from scratch, every element touched */
    for (int i = 0; i < m; i++) { uint64_t h = 0; for (int p = Rp[i]; p < Rp[i + 1]; p++) h += fold_term(Rx[p], ccl[Rj[p]], SR); kr[i] = h; T[i] = i; }
    for (int k = 0; k < nr; k++) ckr[k] = 0;
    /* (all touched: every class's untouched part is empty, so the longest run keeps its number) */
    nr = fold_split(m, T, kr, rcl, szr, ckr, nr, buf, Dr, Dro, &ndr);
    if ((stop > 0 && (double)nr + nc > stop * ((double)m + n)) || (sflag && *sflag) || (sflag2 && *sflag2)) { stopped = 1; goto done; }
    for (int j = 0; j < n; j++) { uint64_t h = 0; for (int p = Ap[j]; p < Ap[j + 1]; p++) h += fold_term(Ax[p], rcl[Ai[p]], SC); kc[j] = h; T[j] = j; }
    nc = fold_split(n, T, kc, ccl, szc, ckc, nc, buf, Dc, Dco, &ndc);
    ndr = 0;      /* (the rows' changes are in the columns' keys already, computed from the rows' current classes) */
    tr0 = lp_time(); pass = 2;
    while (ndc > 0 || ndr > 0) {
        if ((double)nr + nc > 0.8 * ((double)m + n)) { nr = m; nc = n; break; }     /* (as lp_fold_partition: no fold) */
        /* the rows of the columns that changed class: their keys, then their classes */
        int nt = 0;
        for (int d = 0; d < ndc; d++) { const int j = Dc[d], ko = Dco[d], kn = ccl[j];
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int i = Ai[p]; kr[i] += fold_term(Ax[p], kn, SR) - fold_term(Ax[p], ko, SR); if (!mk[i]) { mk[i] = 1; T[nt++] = i; } } }
        for (int t = 0; t < nt; t++) mk[T[t]] = 0;
        ndc = 0; ndr = 0;
        if (nt) nr = fold_split(nt, T, kr, rcl, szr, ckr, nr, buf, Dr, Dro, &ndr);
        pass++;
        if (ndr == 0) break;
        if ((double)nr + nc > 0.8 * ((double)m + n)) { nr = m; nc = n; break; }
        /* the columns of the rows that changed class */
        nt = 0;
        for (int d = 0; d < ndr; d++) { const int i = Dr[d], ko = Dro[d], kn = rcl[i];
            for (int p = Rp[i]; p < Rp[i + 1]; p++) { const int j = Rj[p]; kc[j] += fold_term(Rx[p], kn, SC) - fold_term(Rx[p], ko, SC); if (!mk[j]) { mk[j] = 1; T[nt++] = j; } } }
        for (int t = 0; t < nt; t++) mk[T[t]] = 0;
        ndr = 0;
        if (nt) nc = fold_split(nt, T, kc, ccl, szc, ckc, nc, buf, Dc, Dco, &ndc);
        pass++;
        if ((stop > 0 && (double)nr + nc > stop * ((double)m + n)) || (sflag && *sflag) || (sflag2 && *sflag2)) { stopped = 1; break; }
    }
done:
    if (stopped) { nr = m; nc = n; }
    if (getenv("BRISK_LPDBG")) printf("[folding (incremental): %d passes, %d row and %d column classes%s, %.3fs]\n", pass, nr, nc, stopped ? " (stopped: no fold to the fraction asked)" : "", lp_time() - tr0);
    free(Rp); free(Rj); free(Rx); free(kr); free(kc); free(ckr); free(ckc); free(idx); free(szr); free(szc); free(Dr); free(Dro); free(Dc); free(Dco); free(T); free(mk); free(buf);
    *nrc = nr; *ncc = nc;
    return nr < m || nc < n;
}
/* the partition; returns the number of row classes (in *nrc) and column classes (*ncc), or 0 when there is nothing to fold */
int lp_fold_partition(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                             int *rcl, int *ccl, int *nrc, int *ncc) {
    const int nz = Ap[n];
    int *Rp = xz(sizeof(int) * (m + 2)), *Rj = xm(sizeof(int) * (nz + 1)); double *Rx = xm(sizeof(double) * (nz + 1));
    for (int p = 0; p < nz; p++) Rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int q = Rp[Ai[p] + 1]++; Rj[q] = j; Rx[q] = Ax[p]; }
    uint64_t *kr = xm(sizeof(uint64_t) * (m + 1)), *kc = xm(sizeof(uint64_t) * (n + 1));
    int *idx = xm(sizeof(int) * (m > n ? m : n) + 4);
    for (int i = 0; i < m; i++) kr[i] = fold_dbl(b[i]);
    for (int j = 0; j < n; j++) kc[j] = fold_mix(fold_dbl(c[j]) ^ fold_mix(fold_dbl(u && u[j] < 1e30 ? u[j] : 1e300)) ^ (j < nfree ? 0x5bd1e995ull : 0));
    int nr = fold_relabel(m, kr, rcl, idx), nc = fold_relabel(n, kc, ccl, idx);
    const double tr0 = lp_time(); int round;
    for (round = 0; round < m + n + 2; round++) {
        for (int i = 0; i < m; i++) { uint64_t h = fold_mix((uint64_t)rcl[i] * 0x9e3779b97f4a7c15ull); for (int p = Rp[i]; p < Rp[i + 1]; p++) h += fold_mix(fold_dbl(Rx[p]) ^ fold_mix((uint64_t)ccl[Rj[p]] + 0x1234567ull)); kr[i] = h; }
        const int nr2 = fold_relabel(m, kr, rcl, idx);
        for (int j = 0; j < n; j++) { uint64_t h = fold_mix((uint64_t)ccl[j] * 0x94d049bb133111ebull); for (int p = Ap[j]; p < Ap[j + 1]; p++) h += fold_mix(fold_dbl(Ax[p]) ^ fold_mix((uint64_t)rcl[Ai[p]] + 0x7654321ull)); kc[j] = h; }
        const int nc2 = fold_relabel(n, kc, ccl, idx);
        const int done = nr2 == nr && nc2 == nc;
        nr = nr2; nc = nc2;
        if (done) break;
        /* 21 October 2026 (the integrator, the owner's decision): the refinement only splits classes, so once they number
         * more than 0.8 (m + n) the quotient would keep more than 80 % of the problem and save less than its own
         * construction costs: stop and do not fold. On ken-18 (LP) the full refinement took 3.5 s to remove 10 of 42,836
         * rows, on pds-20 (LP) 2.5 s for 0.5 %, against solves of 2.6 and 3.7 s. */
        if ((double)nr + nc > 0.8 * ((double)m + n)) { nr = m; nc = n; break; }
    }
    if (getenv("BRISK_LPDBG")) printf("[folding: the refinement %d rounds, %d row and %d column classes, %.3fs]\n", round + 1, nr, nc, lp_time() - tr0);
    free(Rp); free(Rj); free(Rx); free(kr); free(kc); free(idx);
    *nrc = nr; *ncc = nc;
    return nr < m || nc < n;
}
/* (20 October 2026, LP/SOCP, a measurement) the partition with the cones kept together: the columns start with the colour
 * of (kind, cone dimension, position in the cone, c_j), the refinement as above, and after each round a cone's colour (the
 * hash of its entries' colours in order) is folded into its entries, so that the cones of a class have the same colour
 * pattern position by position. Returns the number of row classes, column classes and cone classes. */
int lp_fold_partition_cones(int m, int n, int nfree, int nlin, int nq, const int *q, int nr, const int *r,
                            const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c,
                            int *rcl, int *ccl, int *nrc, int *ncc, int *nkc) {
    const int nz = Ap[n], ncone = nq + nr;
    int *Rp = xz(sizeof(int) * (m + 2)), *Rj = xm(sizeof(int) * (nz + 1)); double *Rx = xm(sizeof(double) * (nz + 1));
    for (int p = 0; p < nz; p++) Rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int q_ = Rp[Ai[p] + 1]++; Rj[q_] = j; Rx[q_] = Ax[p]; }
    int *cstart = xm(sizeof(int) * (ncone + 2)), *cdim = xm(sizeof(int) * (ncone + 2)), *colk = xm(sizeof(int) * (n + 1)), *pos = xm(sizeof(int) * (n + 1));
    { int s0 = nfree + nlin; for (int k = 0; k < ncone; k++) { cstart[k] = s0; cdim[k] = k < nq ? q[k] : r[k - nq]; s0 += cdim[k]; } }
    for (int j = 0; j < n; j++) { colk[j] = -1; pos[j] = 0; }
    for (int k = 0; k < ncone; k++) for (int t = 0; t < cdim[k]; t++) { colk[cstart[k] + t] = k; pos[cstart[k] + t] = t; }
    uint64_t *kr = xm(sizeof(uint64_t) * (m + 1)), *kc = xm(sizeof(uint64_t) * (n + 1)), *kk = xm(sizeof(uint64_t) * (ncone + 1));
    int *idx = xm(sizeof(int) * (m > n ? m : n) + 4), *kcl = xm(sizeof(int) * (ncone + 1));
    for (int i = 0; i < m; i++) kr[i] = fold_dbl(b[i]);
    for (int j = 0; j < n; j++) {
        const int k = colk[j];
        const uint64_t kind = j < nfree ? 1 : j < nfree + nlin ? 2 : k < nq ? 3 : 4;
        kc[j] = fold_mix(fold_dbl(c[j]) ^ fold_mix(kind * 0x9e3779b97f4a7c15ull + (k >= 0 ? (uint64_t)cdim[k] * 0x100000001b3ull + (uint64_t)pos[j] * 0xff51afd7ed558ccdull : 0)));
    }
    int nrc_ = fold_relabel(m, kr, rcl, idx), ncc_ = fold_relabel(n, kc, ccl, idx), nkc_ = 0;
    for (int round = 0; round < m + n + 2; round++) {
        for (int i = 0; i < m; i++) { uint64_t h = fold_mix((uint64_t)rcl[i] * 0x9e3779b97f4a7c15ull); for (int p = Rp[i]; p < Rp[i + 1]; p++) h += fold_mix(fold_dbl(Rx[p]) ^ fold_mix((uint64_t)ccl[Rj[p]] + 0x1234567ull)); kr[i] = h; }
        const int nr2 = fold_relabel(m, kr, rcl, idx);
        for (int j = 0; j < n; j++) { uint64_t h = fold_mix((uint64_t)ccl[j] * 0x94d049bb133111ebull); for (int p = Ap[j]; p < Ap[j + 1]; p++) h += fold_mix(fold_dbl(Ax[p]) ^ fold_mix((uint64_t)rcl[Ai[p]] + 0x7654321ull)); kc[j] = h; }
        /* the cones: a cone's colour from its entries' colours in order, folded back into the entries */
        for (int k = 0; k < ncone; k++) { uint64_t h = 0x2545f4914f6cdd1dull; for (int t = 0; t < cdim[k]; t++) h = fold_mix(h ^ kc[cstart[k] + t]); kk[k] = h; }
        for (int k = 0; k < ncone; k++) for (int t = 0; t < cdim[k]; t++) kc[cstart[k] + t] = fold_mix(kc[cstart[k] + t] ^ kk[k]);
        const int nc2 = fold_relabel(n, kc, ccl, idx);
        const int nk2 = ncone > 0 ? fold_relabel(ncone, kk, kcl, idx) : 0;
        const int done = nr2 == nrc_ && nc2 == ncc_ && nk2 == nkc_;
        nrc_ = nr2; ncc_ = nc2; nkc_ = nk2;
        if (done) break;
    }
    free(Rp); free(Rj); free(Rx); free(cstart); free(cdim); free(colk); free(pos); free(kr); free(kc); free(kk); free(idx); free(kcl);
    *nrc = nrc_; *ncc = ncc_; *nkc = nkc_;
    return nrc_ < m || ncc_ < n;
}
static int fold_icmp(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return x < y ? -1 : x > y; }
/* the quotient built and verified (the sums over the blocks the same for every row of a class and every column of a
 * class, to 1e-9 relative); the free classes first. Returns 0 when the partition is not equitable. */
int lp_fold_quotient(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                            int *rcl, int *ccl, int nr, int nc, int *nfree_h, int **Ap_h, int **Ai_h, double **Ax_h, double **b_h, double **c_h, double **u_h,
                            int *rcnt, int *ccnt) {
    /* the free classes first */
    { int *map = xm(sizeof(int) * (nc + 1)), *isf = xz(sizeof(int) * (nc + 1));
      for (int j = 0; j < nfree; j++) isf[ccl[j]] = 1;
      int w = 0; for (int k = 0; k < nc; k++) if (isf[k]) map[k] = w++;
      *nfree_h = w; for (int k = 0; k < nc; k++) if (!isf[k]) map[k] = w++;
      for (int j = 0; j < n; j++) ccl[j] = map[ccl[j]];
      free(map); free(isf); }
    for (int l = 0; l < nr; l++) rcnt[l] = 0;
    for (int k = 0; k < nc; k++) ccnt[k] = 0;
    for (int i = 0; i < m; i++) rcnt[rcl[i]]++;
    for (int j = 0; j < n; j++) ccnt[ccl[j]]++;
    /* the column sums B_kl, by the columns (the first column of a class defines them, the others are checked) */
    int *first_c = xm(sizeof(int) * (nc + 1)); for (int k = 0; k < nc; k++) first_c[k] = -1;
    for (int j = 0; j < n; j++) if (first_c[ccl[j]] < 0) first_c[ccl[j]] = j;
    /* (21 October 2026, LP/SOCP, ISSUES 227: the members of each class linked once; the search of the later columns for a
     * class's members, a scan of all of them per class, was quadratic: 5.2 s of osa-60's (LP) 6.8 s) */
    int *nxt_c = xm(sizeof(int) * (n + 1));
    { int *last = xm(sizeof(int) * (nc + 1)); for (int k = 0; k < nc; k++) last[k] = -1;
      for (int j = n - 1; j >= 0; j--) { nxt_c[j] = last[ccl[j]]; last[ccl[j]] = j; }
      free(last); }
    double *acc = xz(sizeof(double) * (nr + 1)), *refa = xz(sizeof(double) * (nr + 1)); int *touch = xm(sizeof(int) * (2 * nr + 2)); char *seen = xz(nr + 1);
    long nzh = 0; for (int k = 0; k < nc; k++) { const int j = first_c[k]; nzh += Ap[j + 1] - Ap[j]; }
    int *Bp = xz(sizeof(int) * (nc + 2)), *Bi = xm(sizeof(int) * (nzh + 1)); double *Bx = xm(sizeof(double) * (nzh + 1));
    int ok = 1; long w = 0;
    for (int k = 0; k < nc && ok; k++) {
        const int j0 = first_c[k]; int nt = 0;
        for (int p = Ap[j0]; p < Ap[j0 + 1]; p++) { const int l = rcl[Ai[p]]; if (!seen[l]) { seen[l] = 1; touch[nt++] = l; } acc[l] += Ax[p]; }
        Bp[k] = (int)w;
        /* (22 October 2026, the night: the row classes sorted here and the reference sums kept by row class; the search of
         * the reference for each class a member touches and the insertion sort below were quadratic in the column's
         * length: 1.9 s of rd-rplusc-21's (LP) 2.0 s of folding, columns of 1,370 row classes) */
        if (nt > 1) qsort(touch, (size_t)nt, sizeof(int), fold_icmp);
        for (int t = 0; t < nt; t++) { Bi[w] = touch[t]; Bx[w] = acc[touch[t]]; refa[touch[t]] = acc[touch[t]]; w++; }
        for (int t = 0; t < nt; t++) seen[touch[t]] = 0;      /* (acc kept for the check) */
        /* the other columns of the class */
        for (int j = nxt_c[j0]; j >= 0 && ok; j = nxt_c[j]) {
            int nt2 = 0;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int l = rcl[Ai[p]]; if (!seen[l]) { seen[l] = 1; touch[nt + nt2++] = l; } acc[l] -= Ax[p]; }
            for (int t = 0; t < nt2; t++) { const int l = touch[nt + t]; const double ref = refa[l]; if (fabs(acc[l]) > 1e-9 * (1.0 + fabs(ref))) ok = 0; seen[l] = 0; acc[l] = ref; }   /* (the column's sums subtracted from the reference: zero when equal) */
            if (nt2 != nt) ok = 0;
        }
        for (int t = 0; t < nt; t++) { acc[touch[t]] = 0.0; refa[touch[t]] = 0.0; }
    }
    Bp[nc] = (int)w; free(refa);
    free(first_c); free(nxt_c);
    if (!ok) { free(acc); free(touch); free(seen); free(Bp); free(Bi); free(Bx); return 0; }
    /* the rows checked the same way: every row of a class has the same sums over the column classes (needed for Ahat) */
    { double *acc2 = xz(sizeof(double) * (nc + 1)); int *touch2 = xm(sizeof(int) * (nc + 1)); char *seen2 = xz(nc + 1);
      int *Rp = xz(sizeof(int) * (m + 2)), *Rj = xm(sizeof(int) * (Ap[n] + 1)); double *Rx = xm(sizeof(double) * (Ap[n] + 1));
      for (int p = 0; p < Ap[n]; p++) Rp[Ai[p] + 2]++;
      for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
      for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) { const int q = Rp[Ai[p] + 1]++; Rj[q] = j; Rx[q] = Ax[p]; }
      int *first_r = xm(sizeof(int) * (nr + 1)); for (int l = 0; l < nr; l++) first_r[l] = -1;
      for (int i = 0; i < m; i++) if (first_r[rcl[i]] < 0) first_r[rcl[i]] = i;
      double *ref = xm(sizeof(double) * (nc + 1)); int *rk = xm(sizeof(int) * (nc + 1));
      for (int i = 0; i < m && ok; i++) {
          const int l = rcl[i], i0 = first_r[l]; if (i == i0) continue;
          int nt = 0; for (int p = Rp[i0]; p < Rp[i0 + 1]; p++) { const int k = ccl[Rj[p]]; if (!seen2[k]) { seen2[k] = 1; touch2[nt++] = k; } acc2[k] += Rx[p]; }
          for (int t = 0; t < nt; t++) { ref[t] = acc2[touch2[t]]; rk[t] = touch2[t]; acc2[touch2[t]] = 0.0; }
          int nt2 = 0; for (int p = Rp[i]; p < Rp[i + 1]; p++) { const int k = ccl[Rj[p]]; if (seen2[k] == 1) { seen2[k] = 2; nt2++; } acc2[k] += Rx[p]; }
          if (nt2 != nt) ok = 0;
          for (int p = Rp[i]; p < Rp[i + 1]; p++) if (seen2[ccl[Rj[p]]] == 0) ok = 0;    /* (an entry in a class the first row does not touch) */
          for (int t = 0; t < nt; t++) { const int k = rk[t]; if (fabs(acc2[k] - ref[t]) > 1e-9 * (1.0 + fabs(ref[t]))) ok = 0; acc2[k] = 0.0; seen2[k] = 0; }
      }
      free(acc2); free(touch2); free(seen2); free(Rp); free(Rj); free(Rx); free(first_r); free(ref); free(rk); }
    free(acc); free(touch); free(seen);
    if (!ok) { free(Bp); free(Bi); free(Bx); return 0; }
    /* Ahat_lk = n_k B_kl / m_l, by columns k (the rows l ascending within a column: sorted) */
    for (int k = 0; k < nc; k++) {
        for (long q = Bp[k]; q < Bp[k + 1]; q++) Bx[q] = Bx[q] * (double)ccnt[k] / (double)rcnt[Bi[q]];
        /* insertion sort by row (the column of a class is as long as one column of A) */
        for (long q = Bp[k] + 1; q < Bp[k + 1]; q++) { const int bi = Bi[q]; const double bx = Bx[q]; long r = q - 1; while (r >= Bp[k] && Bi[r] > bi) { Bi[r + 1] = Bi[r]; Bx[r + 1] = Bx[r]; r--; } Bi[r + 1] = bi; Bx[r + 1] = bx; }
    }
    *Ap_h = Bp; *Ai_h = Bi; *Ax_h = Bx;
    *b_h = xm(sizeof(double) * (nr + 1)); *c_h = xm(sizeof(double) * (nc + 1)); *u_h = xm(sizeof(double) * (nc + 1));
    for (int i = 0; i < m; i++) (*b_h)[rcl[i]] = b[i];
    for (int j = 0; j < n; j++) { (*c_h)[ccl[j]] = c[j] * (double)ccnt[ccl[j]]; (*u_h)[ccl[j]] = u && u[j] < 1e30 ? u[j] : 1e300; }
    return 1;
}

/* ---- 21 October 2026 (LP/SOCP): sifting, for LPs with many more columns than rows ----
 * (Bixby, Gregory, Lustig, Marsten and Shanno 1992, "Very large-scale linear programming: a case study in combining
 * interior point and simplex methods"; the sifting of CPLEX and Gurobi.) The LP is solved on a working set W of its
 * columns; the others are priced by their reduced costs c_j - a_j'y at the dual point y of that solve; the most negative
 * ones join W, and the loop ends when none is negative. Then x (W's solution, zero elsewhere) and y satisfy the
 * optimality conditions of the whole LP, which are measured on it. W starts with the free columns and the k cheapest
 * columns of each row. Every restricted LP is made feasible by two elastic columns per row (+e_i and -e_i at the cost
 * M = 1e3 (1 + max |c|)); they are zero at the end unless the LP is infeasible or its duals exceed M, and then the
 * whole LP is solved. Returns -1 when not taken or given up (the caller solves the whole LP). */
static _Thread_local const double *g_sift_rc;   /* (per thread, ISSUES 454) */
static int sift_cmp(const void *a, const void *b) { const double x = g_sift_rc[*(const int *)a], y = g_sift_rc[*(const int *)b]; return x < y ? -1 : x > y ? 1 : (*(const int *)a < *(const int *)b ? -1 : 1); }
/* (2 November 2026, on trial: BRISK_LPSPLIT=p) the splitting of long columns (Vanderbei 1991): a bounded column with more
 * than 2p entries becomes copies of it over consecutive slices of p of its rows (in the rows' order), tied by equality
 * rows x_t - x_(t+1) = 0; the copies carry the column's bounds and no cost. In the normal equations a column of length L
 * makes a clique of its L rows, which the sparse factor cannot take when the rows lie far apart in its ordering, so it
 * is kept apart and corrected (the Sherman-Morrison-Woodbury formula, its W = L^-1 U recomputed at every factorization:
 * 15 of Q083_energy's 22 s of factorizations); split, each slice makes a clique of p rows and a link, in the factor.
 * x and y of the original rows come back as they are; the reduced cost of a split column is c_j - A_j' y */
static int lpipm_core(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *zl, double *wu);
static int lp_split_ipm(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                        const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info) {
    static int psz = -1, sauto = -1;
    if (psz < 0) { const char *e = getenv("BRISK_LPSPLIT"); psz = e ? atoi(e) : 0; const char *a = getenv("BRISK_LPSPLITAUTO"); sauto = a ? atoi(a) : 0; if (sauto > 0 && psz == 0) psz = 3; }
    int nsplit = 0, ncopy = 0; long extra = 0;
    static int smin = -1; if (smin < 0) { const char *e = getenv("BRISK_LPSPLITMIN"); smin = e ? atoi(e) : 0; }
    const int lmin = smin > 2 * psz ? smin : 2 * psz + 1;     /* (the shortest column split) */
    if (psz > 0) for (int j = nfree; j < n; j++) { const int L = Ap[j + 1] - Ap[j]; if (L >= lmin) { const int k = (L + psz - 1) / psz; nsplit++; ncopy += k - 1; extra += 2L * (k - 1); } }
    if (nsplit == 0) return lpipm_core(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info, NULL, NULL);
    /* (the automatic choice, BRISK_LPSPLITAUTO=1, on trial) only where an iteration is worth two more analyses: from 10,000
     * rows and 50,000 nonzeros */
    if (sauto > 0 && (m < 10000 || Ap[n] < 50000)) return lpipm_core(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info, NULL, NULL);
    /* (BRISK_LPSPLITORD=1, on trial) each split column's entries in the order of their rows' positions in the sparse
     * part's ordering, so that a slice holds rows the elimination keeps together */
    static int sord = -1; if (sord < 0) { const char *e = getenv("BRISK_LPSPLITORD"); sord = e ? atoi(e) : 0; }
    int *eord = xm(sizeof(int) * ((size_t)Ap[n] + 1));      /* (eord[p]: the entry of column j at its p-th place) */
    for (long p = 0; p < Ap[n]; p++) eord[p] = (int)p;
    if (sord > 0) {
        int *pos = xm(sizeof(int) * (m + 1));
        if (lp_rows_position(m, n, Ap, Ai, Ax, pos)) {
            for (int j = nfree; j < n; j++) {
                const int L = Ap[j + 1] - Ap[j]; if (L < lmin) continue;
                int *e0 = eord + Ap[j];
                for (int a = 1; a < L; a++) { const int v = e0[a]; const int pv = pos[Ai[v]]; int b = a - 1; while (b >= 0 && pos[Ai[e0[b]]] > pv) { e0[b + 1] = e0[b]; b--; } e0[b + 1] = v; }
            }
        }
        free(pos);
    }
    const int m2 = m + ncopy, n2 = n + ncopy;
    const long nz2 = Ap[n] + extra;
    int *Ap2 = xm(sizeof(int) * (n2 + 1)), *Ai2 = xm(sizeof(int) * (nz2 + 1)); double *Ax2 = xm(sizeof(double) * (nz2 + 1));
    double *b2 = xz(sizeof(double) * (m2 + 1)), *c2 = xz(sizeof(double) * (n2 + 1)), *u2 = xm(sizeof(double) * (n2 + 1));
    memcpy(b2, b, sizeof(double) * m); memcpy(c2, c, sizeof(double) * n);
    for (int j = 0; j < n; j++) u2[j] = u ? u[j] : 1e30;
    /* the copies' columns are built after the originals: first the originals (a split one keeps its first slice and the
     * link to its first copy), then the copies in order */
    int *cfirst = xm(sizeof(int) * (n + 1));     /* (the first copy's index of a split column, or -1) */
    int nx = n, rx = m;
    for (int j = 0; j < n; j++) { cfirst[j] = -1; const int L = Ap[j + 1] - Ap[j]; if (j >= nfree && psz > 0 && L >= lmin) { cfirst[j] = nx; nx += (L + psz - 1) / psz - 1; } }
    long q = 0;
    for (int j = 0; j < n; j++) {
        Ap2[j] = (int)q;
        const int L = Ap[j + 1] - Ap[j];
        if (cfirst[j] < 0) { for (int p = Ap[j]; p < Ap[j + 1]; p++) { Ai2[q] = Ai[p]; Ax2[q] = Ax[p]; q++; } continue; }
        for (int p = Ap[j]; p < Ap[j] + psz; p++) { const int e = eord[p]; Ai2[q] = Ai[e]; Ax2[q] = Ax[e]; q++; }
        (void)L;
    }
    /* the link rows: column j's own link first (row rx for the pair (j, first copy)), so each split column's links are
     * consecutive rows: link t between slice t and slice t+1 */
    int *lrow0 = xm(sizeof(int) * (n + 1));
    for (int j = 0; j < n; j++) { lrow0[j] = -1; if (cfirst[j] >= 0) { const int k = (Ap[j + 1] - Ap[j] + psz - 1) / psz; lrow0[j] = rx; rx += k - 1; } }
    /* the original columns' link entries go at the end of their columns: rebuild with them (simplest: a second pass) */
    {
        long q2 = 0; int *Ap3 = xm(sizeof(int) * (n2 + 1)), *Ai3 = xm(sizeof(int) * (nz2 + 1)); double *Ax3 = xm(sizeof(double) * (nz2 + 1));
        for (int j = 0; j < n; j++) {
            Ap3[j] = (int)q2;
            for (long p = Ap2[j]; p < (j + 1 < n ? Ap2[j + 1] : q); p++) { Ai3[q2] = Ai2[p]; Ax3[q2] = Ax2[p]; q2++; }
            if (cfirst[j] >= 0) { Ai3[q2] = lrow0[j]; Ax3[q2] = 1.0; q2++; }
        }
        for (int j = 0; j < n; j++) {
            if (cfirst[j] < 0) continue;
            const int L = Ap[j + 1] - Ap[j], k = (L + psz - 1) / psz;
            for (int t = 1; t < k; t++) {
                const int col = cfirst[j] + t - 1;
                Ap3[col] = (int)q2;
                const int p0 = Ap[j] + t * psz, p1 = Ap[j] + (t + 1) * psz < Ap[j + 1] ? Ap[j] + (t + 1) * psz : Ap[j + 1];
                for (int p = p0; p < p1; p++) { const int e = eord[p]; Ai3[q2] = Ai[e]; Ax3[q2] = Ax[e]; q2++; }
                Ai3[q2] = lrow0[j] + t - 1; Ax3[q2] = -1.0; q2++;          /* x_(t-1) - x_t = 0 */
                if (t + 1 < k) { Ai3[q2] = lrow0[j] + t; Ax3[q2] = 1.0; q2++; }
                c2[col] = 0.0; u2[col] = u ? u[j] : 1e30;
            }
        }
        Ap3[n2] = (int)q2;
        sort_cols(n2, Ap3, Ai3, Ax3);     /* (each column's rows ascending, as the method's input has them) */
        free(Ap2); free(Ai2); free(Ax2); Ap2 = Ap3; Ai2 = Ai3; Ax2 = Ax3;
        /* (the rows within each column ascending: the originals' slices are, the link rows come after every original row) */
    }
    if (sauto > 0 || getenv("BRISK_LPDBG")) {
        /* the choice: an iteration's flops over the mean supernode width, for each form. With supernodes of a few columns
         * (1 to 10 here) the factorization's and the solves' rate grows about as their width: the per-supernode work of
         * the kernels is paid per column. Measured on 54 LPs with long columns (bench/lp_24h): the split form taken where
         * this ratio is below 1 gave every gain (the energy LPs 0.36-0.81, neos-2974461-ibar 0.46) and none of the
         * losses but umts (0.12 -> 0.17 s) */
        const double c0 = lpipm_iter_flops(m, n, Ap, Ai, Ax, 0.0); const int ns0 = g_iterflops_ns, nd0 = g_iterflops_nd;
        const double c1 = lpipm_iter_flops(m2, n2, Ap2, Ai2, Ax2, 3.0 * c0); const int ns1 = g_iterflops_ns, nd1 = g_iterflops_nd;
        const double w0 = ns0 > 0 ? (double)m / ns0 : 1.0, w1 = ns1 > 0 ? (double)m2 / ns1 : 1.0;
        const double r = (c1 < 1e299 && c0 > 0 && c0 < 1e299) ? (c1 / w1) / (c0 / w0) : 1e300;
        if (getenv("BRISK_LPDBG"))
            fprintf(stderr, "   [split estimate: original %.3e flops (%d supernodes, mean width %.2f, %d dense), split %.3e (%d supernodes, mean width %.2f, %d dense): ratio %.2f]\n",
                    c0, ns0, w0, nd0, c1, ns1, w1, nd1, r);
        if (sauto > 0 && !(r < 1.0)) {
            free(Ap2); free(Ai2); free(Ax2); free(b2); free(c2); free(u2); free(cfirst); free(lrow0); free(eord);
            return lpipm_core(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info, NULL, NULL);
        }
    }
    if (opt->verbose > 0) printf("LP long columns split: %d columns into slices of %d rows (%d copies, %d link rows)\n", nsplit, psz, ncopy, ncopy);
    double *x2 = xz(sizeof(double) * (n2 + 1)), *y2 = xz(sizeof(double) * (m2 + 1)), *z2 = z ? xz(sizeof(double) * (n2 + 1)) : NULL;
    const int st = lpipm_core(m2, n2, nfree, Ap2, Ai2, Ax2, b2, c2, u2, opt, x2, y2, z2, info, NULL, NULL);
    memcpy(x, x2, sizeof(double) * n); memcpy(y, y2, sizeof(double) * m);
    if (z) for (int j = 0; j < n; j++) {
        if (cfirst[j] < 0) { z[j] = z2[j]; continue; }
        double s2 = c[j]; for (int p = Ap[j]; p < Ap[j + 1]; p++) s2 -= Ax[p] * y[Ai[p]];
        z[j] = s2;
    }
    free(x2); free(y2); free(z2); free(Ap2); free(Ai2); free(Ax2); free(b2); free(c2); free(u2); free(cfirst); free(lrow0); free(eord);
    return st;
}
static int lp_sift_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                         const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info) {
    static int mode = -1, kset = 0; static double ratio = 20.0, addf = 1.0;
    if (mode < 0) { const char *e_ = getenv("BRISK_LPSIFT"); mode = e_ ? atoi(e_) : 2;      /* 0 off, 1 whenever n >= 2 m, 2 (the default) by the shape: n - nfree >= 20 m and n >= 100,000 */
        if ((e_ = getenv("BRISK_LPSIFTK"))) kset = atoi(e_);
        if ((e_ = getenv("BRISK_LPSIFTRATIO"))) ratio = atof(e_);
        if ((e_ = getenv("BRISK_LPSIFTADD"))) addf = atof(e_); }
    if (mode == 0 || m < 1) return -1;
    if (!((double)(n - nfree) >= (mode == 1 ? 2.0 : ratio) * m && (mode == 1 || n >= 100000))) return -1;
    const double t0 = lp_time(), tol = opt->tol > 0 ? opt->tol : 1e-8, BIG = 1e29;
    const double cmax = nrminf(n, c), bmax = nrminf(m, b), M = 1e3 * (1.0 + cmax);
    /* the first working set of about 10 m columns: a column lies in avglen rows on average, so the k cheapest of each
     * row give about k m / avglen of them */
    const int kfirst = kset > 0 ? kset : (int)fmin(64.0, fmax(4.0, ceil(10.0 * (double)Ap[n] / fmax(1.0, (double)n))));
    /* A by rows, for the first working set */
    int *Rp = xz(sizeof(int) * (m + 2)), *Rj = xm(sizeof(int) * ((size_t)Ap[n] + 1));
    for (long p = 0; p < Ap[n]; p++) Rp[Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) Rj[Rp[Ai[p] + 1]++] = j;
    unsigned char *in = xz((size_t)n + 1);
    int nw = 0;
    for (int j = 0; j < nfree; j++) { in[j] = 1; nw++; }
    {   int *hp = xm(sizeof(int) * ((size_t)kfirst + 1));      /* a max-heap by cost of the k cheapest of the row so far */
        for (int i = 0; i < m; i++) {
            int h = 0;
            for (int q = Rp[i]; q < Rp[i + 1]; q++) {
                const int j = Rj[q]; if (j < nfree) continue;
                if (h < kfirst) { int s = h++; while (s > 0 && c[hp[(s - 1) / 2]] < c[j]) { hp[s] = hp[(s - 1) / 2]; s = (s - 1) / 2; } hp[s] = j; }
                else if (c[j] < c[hp[0]]) {
                    int s = 0;
                    for (;;) { const int l = 2 * s + 1, r = l + 1; int g = s; double cg = c[j];
                        if (l < h && c[hp[l]] > cg) { g = l; cg = c[hp[l]]; }
                        if (r < h && c[hp[r]] > cg) g = r;
                        if (g == s) break;
                        hp[s] = hp[g]; s = g; }
                    hp[s] = j;
                }
            }
            for (int t = 0; t < h; t++) if (!in[hp[t]]) { in[hp[t]] = 1; nw++; }
        }
        free(hp);
    }
    free(Rp); free(Rj);
    /* (31 October 2026) sifting pays only when an iteration on a working set costs much less than one on the whole LP: each
     * round is a solve of its own, and the rounds are several. With few rows the normal equations are dense whatever the
     * columns, and the factorization costs the same for both: cutstock (LPs of an independent benchmark, 7,899-10,000 rows,
     * 0.7-1 million columns) sifted 6 rounds for 401 s, then solved the whole LP in 64.5 s. The cost of an iteration on the
     * first working set (c_r) and on the whole LP (c_w, under the budget SIFT_ROUNDS c_r, so it stops early) by the method's
     * own estimate (lpipm_iter_flops); no sifting when c_w < SIFT_ROUNDS c_r. SIFT_ROUNDS = 4: the rounds sifting needs
     * when it succeeds (2-6 on the sets). BRISK_LPSIFTCOST=0: as before */
    {   static int sc = -1; if (sc < 0) { const char *e_ = getenv("BRISK_LPSIFTCOST"); sc = !(e_ && atoi(e_) == 0); }
        if (sc) {
            const double SIFT_ROUNDS = 4.0;
            int nwc = 0; long nzw = 0; for (int j = 0; j < n; j++) if (in[j]) { nwc++; nzw += Ap[j + 1] - Ap[j]; }
            int *Wp = xm(sizeof(int) * ((size_t)nwc + 1)), *Wi = xm(sizeof(int) * ((size_t)nzw + 1)); double *Wx = xm(sizeof(double) * ((size_t)nzw + 1));
            long w = 0; int k = 0;
            for (int j = 0; j < n; j++) if (in[j]) { Wp[k++] = (int)w; for (int p = Ap[j]; p < Ap[j + 1]; p++) { Wi[w] = Ai[p]; Wx[w++] = Ax[p]; } }
            Wp[nwc] = (int)w;
            const double cr = lpipm_iter_flops(m, nwc, Wp, Wi, Wx, 0);
            free(Wp); free(Wi); free(Wx);
            const double cw = cr < 1e299 ? lpipm_iter_flops(m, n, Ap, Ai, Ax, SIFT_ROUNDS * cr) : 1e300;
            if (opt->verbose > 1) printf("sifting: an iteration on the first working set %.2e flops, on the whole LP %s%.2e%s\n", cr, cw >= SIFT_ROUNDS * cr ? "above " : "", cw >= SIFT_ROUNDS * cr ? SIFT_ROUNDS * cr : cw, cw >= SIFT_ROUNDS * cr ? "" : ": the whole LP");
            if (cw < SIFT_ROUNDS * cr) { free(in); return -1; }
        }
    }
    double *rc = xm(sizeof(double) * ((size_t)n + 1)), *ys = xz(sizeof(double) * ((size_t)m + 1));
    int *neg = xm(sizeof(int) * ((size_t)n + 1)), *col = xm(sizeof(int) * ((size_t)n + 1));
    double *xs = NULL;
    int st = -1, rounds = 0, its = 0, done = 0, prev_nneg = 0; long nfact = 0, nsolve = 0, ncg = 0; double tfact = 0, tsolve = 0;
    while (!done) {
        rounds++;
        if (nw > n / 3 || rounds > 12) break;           /* (the working set no longer small: the whole LP) */
        const int ns = nw + 2 * m;
        long nzs = 2 * (long)m; for (int j = 0; j < n; j++) if (in[j]) nzs += Ap[j + 1] - Ap[j];
        int *Bp = xm(sizeof(int) * ((size_t)ns + 1)), *Bi = xm(sizeof(int) * ((size_t)nzs + 1));
        double *Bx = xm(sizeof(double) * ((size_t)nzs + 1)), *cs = xm(sizeof(double) * ((size_t)ns + 1)), *us = xm(sizeof(double) * ((size_t)ns + 1));
        long w = 0; int k = 0;
        for (int j = 0; j < n; j++) if (in[j]) {
            Bp[k] = (int)w; col[k] = j; cs[k] = c[j]; us[k] = u ? u[j] : 1e300;
            for (int p = Ap[j]; p < Ap[j + 1]; p++) { Bi[w] = Ai[p]; Bx[w++] = Ax[p]; }
            k++;
        }
        for (int i = 0; i < m; i++) for (int s = 0; s < 2; s++) { Bp[k] = (int)w; Bi[w] = i; Bx[w++] = s ? -1.0 : 1.0; cs[k] = M; us[k] = 1e300; k++; }
        Bp[ns] = (int)w;
        free(xs); xs = xz(sizeof(double) * ((size_t)ns + 1));
        double *zs = xz(sizeof(double) * ((size_t)ns + 1));
        LpIpmOpts o2 = *opt; o2.verbose = opt->verbose > 1 ? 1 : 0; o2.tol = 0.1 * tol;
        o2.analyzed = NULL;     /* (24 October 2026: a restricted problem's iteration is no measure for the caller's choice of form) */
        int s2 = 1;
        if (opt->timelimit > 0) o2.timelimit = opt->timelimit - (lp_time() - t0);
        LpIpmInfo i2; memset(&i2, 0, sizeof i2);
        if (opt->timelimit <= 0 || o2.timelimit > 0) {
            double dp = 0;
            s2 = lp_dual_solve(m, ns, nfree, Bp, Bi, Bx, b, cs, us, &o2, xs, ys, zs, &i2, &dp);
            if (s2 < 0) { LpIpmOpts op = o2; op.dthr = dp; s2 = lpipm_core(m, ns, nfree, Bp, Bi, Bx, b, cs, us, &op, xs, ys, zs, &i2, NULL, NULL); }
        }
        its += i2.iters; nfact += i2.nfact; nsolve += i2.nsolve; ncg += i2.ncg; tfact += i2.t_fact; tsolve += i2.t_solve;
        free(Bp); free(Bi); free(Bx); free(cs); free(us); free(zs);
        if (s2 != 0) break;
        double el = 0; for (int t = nw; t < ns; t++) el = fmax(el, xs[t]);
        int nneg = 0; double rmin = 0;
        const double thr = -0.1 * tol * (1.0 + cmax);
        for (int j = 0; j < n; j++) {
            if (in[j]) continue;
            double s = c[j]; for (int p = Ap[j]; p < Ap[j + 1]; p++) s -= Ax[p] * ys[Ai[p]];
            rc[j] = s;
            if (s < thr) { neg[nneg++] = j; if (s < rmin) rmin = s; }
        }
        if (opt->verbose > 1)
            printf("sifting: round %d, %d of %d columns, %d iterations, objective %.10e, elastic part %.1e, %d reduced costs negative (least %.1e), %.2fs\n",
                   rounds, nw, n, i2.iters, i2.pobj, el, nneg, rmin, lp_time() - t0);
        /* (24 October 2026) a round that prices out at least as many columns as the one before has not converged and will
         * not: the whole LP (supportcase19, LP: 37,219, 173,249, 162,598 columns priced out with the elastic part above zero,
         * four rounds and 23.6 s before the whole LP was solved; the transportation LPs 720, 6, 0). BRISK_LPSIFTCONV=0: as
         * before (the working set past a third of the columns, or 12 rounds) */
        { static int conv = -1; if (conv < 0) { const char *e_ = getenv("BRISK_LPSIFTCONV"); conv = !(e_ && atoi(e_) == 0); }
          if (conv && nneg > 0 && rounds >= 2 && nneg >= prev_nneg) break;
          prev_nneg = nneg; }
        if (nneg == 0) {
            if (el > 0.1 * tol * (1.0 + bmax)) break;       /* (infeasible, or duals beyond M: the whole LP) */
            for (int j = 0; j < n; j++) xout[j] = 0.0;
            for (int t = 0; t < nw; t++) xout[col[t]] = xs[t];
            memcpy(yout, ys, sizeof(double) * (size_t)m);
            st = 0; done = 1;
        } else {
            int add = (int)fmax((double)m, addf * nw); if (add > nneg) add = nneg;
            if (add < nneg) { g_sift_rc = rc; qsort(neg, (size_t)nneg, sizeof(int), sift_cmp); g_sift_rc = NULL; }
            for (int t = 0; t < add; t++) { in[neg[t]] = 1; nw++; }
        }
    }
    if (st == 0) {      /* the errors of the whole LP, as the method measures them */
        double *r = xz(sizeof(double) * ((size_t)m + 1));
        for (int j = 0; j < n; j++) if (xout[j] != 0.0) for (int p = Ap[j]; p < Ap[j + 1]; p++) r[Ai[p]] += Ax[p] * xout[j];
        double ep = 0, ed = 0, pobj = 0, dobj = 0;
        for (int i = 0; i < m; i++) { ep = fmax(ep, fabs(b[i] - r[i])); dobj += b[i] * yout[i]; }
        for (int j = 0; j < n; j++) {
            double s = c[j]; for (int p = Ap[j]; p < Ap[j + 1]; p++) s -= Ax[p] * yout[Ai[p]];
            if (zout) zout[j] = s;
            pobj += c[j] * xout[j];
            if (j < nfree) ed = fmax(ed, fabs(s));
            else if (u && u[j] < BIG) { if (s < 0) dobj += u[j] * s; ep = fmax(ep, fmax(0.0, xout[j] - u[j])); ep = fmax(ep, -xout[j]); }
            else { ed = fmax(ed, -s); ep = fmax(ep, -xout[j]); }
        }
        free(r);
        memset(info, 0, sizeof(*info));
        info->status = 0; info->iters = its; info->nfact = nfact; info->nsolve = nsolve; info->ncg = ncg; info->t_fact = tfact; info->t_solve = tsolve;
        info->pobj = pobj; info->dobj = dobj;
        info->pinf = ep / (1.0 + bmax); info->dinf = ed / (1.0 + cmax); info->gap = fabs(pobj - dobj) / (1.0 + fabs(pobj + opt->objoff));
        info->time = lp_time() - t0;
        if (opt->verbose > 0) printf("sifting: %d rounds, the LP solved on %d of its %d columns (%d iterations in all, %.2fs); on the whole LP: pinf %.1e dinf %.1e gap %.1e\n",
                                     rounds, nw, n, its, info->time, info->pinf, info->dinf, info->gap);
    } else if (opt->verbose > 0) printf("sifting: given up after %d rounds (%d of %d columns, %.2fs): the whole LP is solved\n", rounds, nw, n, lp_time() - t0);
    free(rc); free(ys); free(neg); free(col); free(xs); free(in);
    return st;
}

static int lpipm_solve_in(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                          const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info);
int lpipm_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info) {
    const double dl = ne_dl; const volatile int *sf = ne_stopf;
    ne_dl = opt && opt->timelimit > 0 ? lp_time() + opt->timelimit : 0; ne_stopf = opt ? opt->stop : NULL;
    const int st = lpipm_solve_in(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info);
    ne_dl = dl; ne_stopf = sf;
    return st;
}
static int lpipm_solve_in(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                          const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info) {
    /* (18 October 2026: the busy-machine rule of the other engines, solver.c: only the free cores; restored on return) */
#ifdef _OPENMP
    const int ntb = omp_in_parallel() ? -1 : brisk_threads_busy(0);
#endif
    int st = -1, folded = 0;
    if (getenv("BRISK_LPWRITE")) {      /* (experiments: the standard form as the method gets it, as a free MPS file:
                                           min c'x, A x = b, 0 <= x <= u, the first nfree columns free) */
        FILE *fw = fopen(getenv("BRISK_LPWRITE"), "w");
        if (fw) {
            fprintf(fw, "NAME stdform\nROWS\n N obj\n");
            for (int i = 0; i < m; i++) fprintf(fw, " E r%d\n", i);
            fprintf(fw, "COLUMNS\n");
            for (int j = 0; j < n; j++) {
                if (c[j] != 0) fprintf(fw, " x%d obj %.17g\n", j, c[j]);
                for (int p = Ap[j]; p < Ap[j + 1]; p++) fprintf(fw, " x%d r%d %.17g\n", j, Ai[p], Ax[p]);
            }
            fprintf(fw, "RHS\n");
            for (int i = 0; i < m; i++) if (b[i] != 0) fprintf(fw, " rhs r%d %.17g\n", i, b[i]);
            fprintf(fw, "BOUNDS\n");
            for (int j = 0; j < n; j++) {
                if (j < nfree) fprintf(fw, " FR bnd x%d\n", j);
                else if (u && u[j] < 1e29) fprintf(fw, " UP bnd x%d %.17g\n", j, u[j]);
            }
            fprintf(fw, "ENDATA\n"); fclose(fw);
        }
    }
    if (!getenv("BRISK_LPNOFOLD") && m > 0 && n > 0) {       /* (20 October 2026: the coarsest equitable partition, when it is coarser than the trivial one) */
        int *rcl = xm(sizeof(int) * (m + 1)), *ccl = xm(sizeof(int) * (n + 1)); int nr = m, nc = n;
        const double tf = lp_time();
        if ((getenv("BRISK_LPFOLDOLD") ? lp_fold_partition : lp_fold_partition_inc)(m, n, nfree, Ap, Ai, Ax, b, c, u, rcl, ccl, &nr, &nc)) {
            int nfh = 0, *Aph = NULL, *Aih = NULL; double *Axh = NULL, *bh = NULL, *ch = NULL, *uh = NULL;
            int *rcnt = xm(sizeof(int) * (nr + 1)), *ccnt = xm(sizeof(int) * (nc + 1));
            if (lp_fold_quotient(m, n, nfree, Ap, Ai, Ax, b, c, u, rcl, ccl, nr, nc, &nfh, &Aph, &Aih, &Axh, &bh, &ch, &uh, rcnt, ccnt)) {
                if (opt->verbose > 0) printf("folding: %d rows -> %d, %d columns -> %d (the coarsest equitable partition of the constraint graph; %.2fs)\n", m, nr, n, nc, lp_time() - tf);
                double *xh = xz(sizeof(double) * (nc + 1)), *yh = xz(sizeof(double) * (nr + 1)), *zh = xz(sizeof(double) * (nc + 1));
                double dp = 0;
                st = lp_dual_solve(nr, nc, nfh, Aph, Aih, Axh, bh, ch, uh, opt, xh, yh, zh, info, &dp);
                if (st < 0) { LpIpmOpts op = *opt; op.dthr = dp; st = lpipm_core(nr, nc, nfh, Aph, Aih, Axh, bh, ch, uh, &op, xh, yh, zh, info, NULL, NULL); }
                for (int j = 0; j < n; j++) x[j] = xh[ccl[j]];
                for (int i = 0; i < m; i++) y[i] = yh[rcl[i]] / (double)rcnt[rcl[i]];
                if (z) for (int j = 0; j < n; j++) z[j] = zh[ccl[j]] / (double)ccnt[ccl[j]];
                free(xh); free(yh); free(zh); folded = 1;
            } else if (opt->verbose > 0) printf("folding: the partition of the hashes is not equitable (%d rows, %d columns): not taken\n", nr, nc);
            free(Aph); free(Aih); free(Axh); free(bh); free(ch); free(uh); free(rcnt); free(ccnt);
        }
        free(rcl); free(ccl);
    }
    if (!folded) {
        st = lp_sift_solve(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info);
        double dp = 0;
        if (st < 0) st = lp_dual_solve(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info, &dp);
        if (st < 0) { LpIpmOpts op = *opt; op.dthr = dp; st = lp_split_ipm(m, n, nfree, Ap, Ai, Ax, b, c, u, &op, x, y, z, info); }
    }
#ifdef _OPENMP
    if (ntb > 0) omp_set_num_threads(ntb);
#endif
    return st;
}
