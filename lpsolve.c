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
 * caller hands the problem to the cone solver (homogeneous self-dual model).                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
void   schol_free(SChol *S);
double *schol_values(SChol *S);
size_t schol_offset(const SChol *S, size_t r, size_t c);
void   schol_set_tinypiv(SChol *S, double t);
void   schol_set_tinydag(SChol *S, int on);
int    schol_factor(SChol *S, double shift);
void   schol_solve(const SChol *S, double *b, double *work);
size_t schol_nnz(const SChol *S);
int    schol_set_amd(int mode);
void   schol_zero(SChol *S);
double schol_flops(const SChol *S);
int    schol_ntiny(const SChol *S);
void   schol_set_solve_seq(int on);
void   schol_set_perm(const int *perm);
void   schol_set_nd(int on);
double schol_set_amalg(double f);

static double lp_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
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
    SChol *sc; int *Mp, *Mi; size_t *Mo;   /* the lower pattern of A_s A_s' by columns: rows Mi >= the column, offsets Mo */
    double *V, *G, *gw;                /* dense columns: V = M_s^-1 A_d (m x nd), G = Theta_d^-1 + A_d' V = L L' */
    double *theta;
    double *w1, *w2, *w3, *w4, *wn, *acc; long nref; double worst;   /* work: m, m, m, n, m */
    double *pr, *pz, *pp, *pq;         /* CG work (m) */
    double reg;
    long nfact, nsolve, ncg; int ntiny;
    double t_fact, t_solve;
    double **acc_t; int nacc_t;          /* 17 October 2026: per-thread accumulators of the assembly */
    int *Fp, *Fj; double *Fx;          /* 18 October 2026: all of A by rows (columns ascending), for A x on the threads */
    int dasm, nsc, *scol; double *Md, *Wd; int wdc;   /* 18 October 2026 (LP/SOCP): A_s Theta A_s' by dsyrk (dense; scol: the sparse columns, Wd: m x wdc) */
    int est;                           /* ne_setup for a cost estimate only (AMD, no offsets) */
    double est_budget, est_lb; long est_w;   /* the estimate's budget; its lower bound on the flops when it is exceeded */
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
/* q = A Theta A' v (the true matrix: the regularization is in the factor, the preconditioner, only) */
static void M_mul(LpS *S, const double *v, double *q) {
    At_mul(S, v, S->wn);
    for (int j = 0; j < S->n; j++) S->wn[j] *= S->theta[j];
    A_mul(S, S->wn, q);
}
/* the factorization of A_s Theta A_s' + reg (and the Schur complement of the dense columns); 0: done */
static int ne_factor(LpS *S) {
    const double t0 = lp_time();
    const int m = S->m;
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
            BL(dsyrk_)("L", "N", &m, &kk, &one, S->Wd, &m, &beta, S->Md, &m);
            beta = 1.0;
        }
        if (beta == 0.0) for (int i = 0; i < m; i++) memset(S->Md + (size_t)i * m + i, 0, sizeof(double) * (size_t)(m - i));
        for (int i = 0; i < m; i++) {
            const double *col = S->Md + (size_t)i * m;
            for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) pm[S->Mo[e]] = col[S->Mi[e]];
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
            pm[S->Mo[S->Mp[i]]] += S->reg;                    /* (the diagonal is the first entry of the column) */
        }
    }
    }
    schol_set_tinypiv(S->sc, getenv("BRISK_LPTINY") ? atof(getenv("BRISK_LPTINY")) : 1e-20);
    /* 18 October 2026 (LP/SOCP): the large nodes by the tile task graph of sparsechol.c (fnode_dag), the tiny-pivot rule
     * in its diagonal tiles; it was skipped with tiny pivots (BRISK_LPTDAG=0: as before) */
    { static int tdag = -1; if (tdag < 0) { const char *e = getenv("BRISK_LPTDAG"); tdag = !(e && atoi(e) == 0); } schol_set_tinydag(S->sc, tdag); }
    /* 17 October 2026: the factorization by the tree code of sparsechol.c on the threads, the solves
     * sequential. The left-looking code (schol_set_solve_seq) was 25-30 % faster at one thread in 5.7; with
     * the kernels of since the two are even at one thread (Kennington 13.3 against 13.0 s) and the tree code
     * uses the threads (four: 11.8 s; pds-20 5.1 -> 3.6 s), with the same results at any thread count; its
     * solves stay slower (osa-60: 1.2 -> 3.3 s if they go with it). BRISK_LPFLL=1: the left-looking code. */
    static int fll = -1; if (fll < 0) fll = getenv("BRISK_LPFLL") != NULL;
    if (!fll) schol_set_solve_seq(0);
    const int bad = schol_factor(S->sc, 0.0);
    if (!fll) schol_set_solve_seq(1);
    S->ntiny += schol_ntiny(S->sc);
    S->nfact++;
    if (bad) { S->t_fact += lp_time() - t0; return 1; }
    if (S->nd > 0) {
        const int nd = S->nd;
        for (int d = 0; d < nd; d++) {
            const int j = S->dcol[d]; double *v = S->V + (size_t)d * m;
            for (int i = 0; i < m; i++) v[i] = 0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v[S->Ai[p]] = S->Ax[p];
            schol_solve(S->sc, v, S->w3);
        }
        for (int d = 0; d < nd; d++) {
            const double *v = S->V + (size_t)d * m;
            for (int e = 0; e <= d; e++) {
                const int j = S->dcol[e]; double s = 0;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * v[S->Ai[p]];
                S->G[d + (size_t)e * nd] = s; S->G[e + (size_t)d * nd] = s;
            }
            S->G[d + (size_t)d * nd] += 1.0 / S->theta[S->dcol[d]];
        }
        int info = 0;
        BL(dpotrf_)("L", &nd, S->G, &nd, &info);
        if (info) { S->t_fact += lp_time() - t0; return 1; }
    }
    S->t_fact += lp_time() - t0;
    return 0;
}
/* z = P^-1 r: the factored sparse part and the dense columns by the Woodbury formula */
static void ne_prec(LpS *S, const double *r, double *z) {
    const int m = S->m;
    memcpy(z, r, sizeof(double) * (size_t)m);
    schol_solve(S->sc, z, S->w3);
    if (S->nd > 0) {
        const int nd = S->nd, one = 1; int info = 0;
        for (int d = 0; d < nd; d++) {
            const int j = S->dcol[d]; double s = 0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * z[S->Ai[p]];
            S->gw[d] = s;
        }
        BL(dpotrs_)("L", &nd, &one, S->G, &nd, S->gw, &nd, &info);
        for (int d = 0; d < nd; d++) { const double g = S->gw[d]; const double *v = S->V + (size_t)d * m; if (g != 0.0) for (int i = 0; i < m; i++) z[i] -= g * v[i]; }
    }
}
/* (A Theta A') x = r: the preconditioner, then CG on the true matrix until the residual is below
 * rtol |r| (at most maxcg steps); returns the relative residual */
static double ne_cg(LpS *S, double *x, double nr, double rn, double rtol, int maxcg);
static double ne_solve(LpS *S, const double *r, double *x, double rtol, int maxcg) {
    const double t0 = lp_time();
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
    free(S->rs); free(S->cs); free(S->dcol); free(S->isd); if (S->sc) schol_free(S->sc); free(S->Mp); free(S->Mi); free(S->Mo);
    free(S->V); free(S->G); free(S->gw); free(S->theta); free(S->w1); free(S->w2); free(S->w4); free(S->w3); free(S->wn); free(S->acc); if (S->acc_t) { for (int t = 0; t < S->nacc_t; t++) free(S->acc_t[t]); free(S->acc_t); }
    free(S->pr); free(S->pz); free(S->pp); free(S->pq);
    free(S->Md); free(S->Wd); free(S->scol); free(S->wa);
}

/* 18 October 2026 (LP/SOCP): the largest factor (entries) that fits: half of the memory the process may use (the
 * physical memory, or the address-space limit when one is set), at 8 bytes an entry and twice that for the
 * analysis and the panels. Before, 2^31 entries (17 GB): a staircase LP whose factor needs 1.96e8 entries more than
 * the 7 GB of the host was killed instead of being handed to the cone solver. */
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
static int ne_setup(LpS *S, int usedense) {
    const int m = S->m, n = S->n, nz = S->Ap[n];
    free(S->Rp); free(S->Rj); free(S->Rq); free(S->Rx); free(S->Mp); free(S->Mi); free(S->Mo); free(S->V); free(S->G); free(S->gw);
    S->Rp = S->Rj = S->Rq = S->Mp = S->Mi = NULL; S->Rx = S->V = S->G = S->gw = NULL; S->Mo = NULL;
    if (S->sc) { schol_free(S->sc); S->sc = NULL; }
    if (!S->isd) { S->isd = xz(n + 1); S->dcol = xm(sizeof(int) * (n + 1)); }
    memset(S->isd, 0, n + 1); S->nd = 0;
    /* dense columns: long against the average and against the rows, and few */
    if (usedense && m >= 500) {
        double avg = (double)nz / fmax(n, 1);
        const double thr = fmax(fmax(50.0, 10.0 * avg), 0.1 * m);
        int nd = 0;
        for (int j = 0; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] > thr) nd++;
        if (nd > 0 && nd <= 400 && nd < n / 4) { for (int j = 0; j < n; j++) if (S->Ap[j + 1] - S->Ap[j] > thr) { S->isd[j] = 1; S->dcol[S->nd++] = j; } }
    }
    /* the sparse columns by rows */
    S->Rp = xz(sizeof(int) * (m + 2));
    int nzs = 0;
    for (int j = 0; j < n; j++) if (!S->isd[j]) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { S->Rp[S->Ai[p] + 2]++; nzs++; }
    for (int i = 0; i < m; i++) S->Rp[i + 2] += S->Rp[i + 1];
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
    /* the pattern of A_s A_s' (lower, by columns; the diagonal first) */
    int *mark = xm(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) mark[i] = -1;
    long cap = 4L * nzs + m + 16, w = 0;
    int *Mi = xm(sizeof(int) * cap); S->Mp = xm(sizeof(int) * (m + 1));
    if (S->dasm) { cap = (long)m * (m + 1) / 2 + 1; Mi = realloc(Mi, sizeof(int) * cap); for (int i = 0; i < m; i++) { S->Mp[i] = (int)w; for (int k = i; k < m; k++) Mi[w++] = k; } }
    else for (int i = 0; i < m; i++) {
        S->Mp[i] = (int)w;
        if (w + m + 1 > cap) { cap = 2 * cap + m; Mi = realloc(Mi, sizeof(int) * cap); }
        Mi[w++] = i; mark[i] = i;
        for (int q = S->Rp[i]; q < S->Rp[i + 1]; q++) {
            const int j = S->Rj[q];
            for (int p = S->Rq[q] + 1; p < S->Ap[j + 1]; p++) { const int k = S->Ai[p]; if (mark[k] != i) { mark[k] = i; Mi[w++] = k; } }
        }
    }
    S->Mp[m] = (int)w; S->Mi = Mi;
    free(mark);
    /* (a cost estimate against a budget: the factor has at least the w entries of the pattern, so by Cauchy-Schwarz
     * its factorization costs at least w^2 / m flops; above the budget the ordering is not made) */
    if (S->est && S->est_budget > 0 && (double)w * (double)w / m > S->est_budget) { S->est_lb = (double)w * (double)w / m; S->est_w = w; return 0; }
    int *deg = xz(sizeof(int) * (m + 1)), **nbr = xm(sizeof(int *) * (m + 1));
    for (int i = 0; i < m; i++) for (int e = S->Mp[i] + 1; e < S->Mp[i + 1]; e++) { deg[i]++; deg[Mi[e]]++; }
    int *pool = xm(sizeof(int) * (2 * (w - m) + 1));
    { long o = 0; for (int i = 0; i < m; i++) { nbr[i] = pool + o; o += deg[i]; deg[i] = 0; } }
    for (int i = 0; i < m; i++) for (int e = S->Mp[i] + 1; e < S->Mp[i + 1]; e++) { const int k = Mi[e]; nbr[i][deg[i]++] = k; nbr[k][deg[k]++] = i; }
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
    const int om = schol_set_amd(getenv("BRISK_LPAMD") ? atoi(getenv("BRISK_LPAMD")) : 2);
    /* 18 October 2026 (LP/SOCP): no nested dissection on a pattern with more than a tenth of its entries: a separator
     * of a graph that dense is most of it, and the bisection's matching and refinement cost O(edges) a level (L1
     * regression with its free columns joined: 35 s in the bisection for a factor that AMD orders at once) */
    const int dense_pat = (double)(w - m) > 0.05 * (double)m * (m - 1.0);
    schol_set_nd(S->est || dense_pat ? 0 : getenv("BRISK_LPND") ? atoi(getenv("BRISK_LPND")) : 1);
    const double oa = schol_set_amalg(0.3);      /* 5.7: relaxed supernodes (30 % explicit zeros at most, 64 columns) */
    S->sc = schol_analyze_adj(m, deg, nbr, lp_fillcap());
    schol_set_amalg(oa);
    schol_set_nd(0);
    if (fperm) { schol_set_perm(NULL); free(fperm); }
    schol_set_amd(om);
    free(deg); free(nbr); free(pool);
    if (!S->sc) return 1;
    if (S->est) return 0;
    S->Mo = xm(sizeof(size_t) * (w + 1));
    for (int i = 0; i < m; i++) for (int e = S->Mp[i]; e < S->Mp[i + 1]; e++) {
        S->Mo[e] = schol_offset(S->sc, (size_t)Mi[e], (size_t)i);
        if (S->Mo[e] == (size_t)-1) return 1;
    }
    if (S->nd) { S->V = xm(sizeof(double) * (size_t)m * S->nd); S->G = xm(sizeof(double) * (size_t)S->nd * S->nd); S->gw = xm(sizeof(double) * (S->nd + 1)); }
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
    for (int j = 0; j < n; j++) {
        double v = rc[j];
        if (!S->fr[j]) v -= rxz[j] / x[j];
        if (S->bd[j]) v += (rsw[j] - w[j] * ru[j]) / s[j];
        g[j] = v;
    }
    kkt_solve(S, g, rb, dx, dy, rtol, maxcg);
    for (int j = 0; j < n; j++) {
        dz[j] = S->fr[j] ? 0.0 : (rxz[j] - z[j] * dx[j]) / x[j];
        if (S->bd[j]) { ds[j] = ru[j] - dx[j]; dw[j] = (rsw[j] - w[j] * ds[j]) / s[j]; } else { ds[j] = 0; dw[j] = 0; }
    }
}
/* dx = Theta (A'dy - g), A dx = rb, by the normal equations */
static void kkt_solve(LpS *S, const double *g, const double *rb, double *dx, double *dy, double rtol, int maxcg) {
    const int m = S->m, n = S->n;
    for (int j = 0; j < n; j++) dx[j] = S->theta[j] * g[j];
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
        for (int j = 0; j < n; j++) dx[j] = S->theta[j] * (dx[j] - g[j]);
        A_mul(S, dx, S->pr);
        for (int i = 0; i < m; i++) S->pr[i] = rb[i] - S->pr[i];
        const double rn = sqrt(dot(m, S->pr, S->pr));
        have_res = 1;
        if (nr > 0 && rn > rtol * nr && maxcg > 0) {
            ne_cg(S, dy, nr, rn, rtol, maxcg);
            At_mul(S, dy, dx);
            for (int j = 0; j < n; j++) dx[j] = S->theta[j] * (dx[j] - g[j]);
            have_res = 0;
        }
        S->t_solve += lp_time() - t0;
    }
    /* A dx = rb to the accuracy the step needs (the right-hand side rb + A Theta g loses it by
     * cancellation when Theta spreads): correct dy on the residual */
    { const double nrb = nrminf(m, rb);
      for (int pass = 0; pass < 3; pass++) {
          if (have_res) { memcpy(S->w1, S->pr, sizeof(double) * (size_t)m); have_res = 0; }
          else { A_mul(S, dx, S->w1); for (int i = 0; i < m; i++) S->w1[i] = rb[i] - S->w1[i]; }
          const double nr = nrminf(m, S->w1);
          if (!(nr > fmax(1e-3 * nrb, 1e-13)) || !isfinite(nr)) break;
          double *ddy = S->w4;
          ne_solve(S, S->w1, ddy, rtol, maxcg);
          At_mul(S, ddy, S->wn);
          double nn = 0;
          for (int j = 0; j < n; j++) { S->wn[j] *= S->theta[j]; }
          A_mul(S, S->wn, S->acc);
          for (int i = 0; i < m; i++) { const double v = fabs(S->w1[i] - S->acc[i]); if (v > nn) nn = v; S->acc[i] = 0; }
          if (!(nn < 0.5 * nr)) break;                    /* (no gain: the factorization is at its limit) */
          for (int j = 0; j < n; j++) dx[j] += S->wn[j];
          for (int i = 0; i < m; i++) dy[i] += ddy[i];
          S->nref++;
      } }
}
static void step_max(const LpS *S, const double *x, const double *s, const double *z, const double *w,
                     const double *dx, const double *ds, const double *dz, const double *dw, double *ap, double *ad) {
    double a = 1e300, d = 1e300;
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

/* zl, wu (may be NULL): the multipliers of the lower and the upper bounds apart (zout is their difference) */
static int lpipm_core(int m, int n, int nfree, const int *Ap0, const int *Ai0, const double *Ax0, const double *b0, const double *c0, const double *u0,
                const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info, double *zl, double *wu) {
    const double t0 = lp_time();
    const double tol = opt->tol > 0 ? opt->tol : 1e-8;
    const int maxit = opt->maxit > 0 ? opt->maxit : 200, verbose = opt->verbose;
    memset(info, 0, sizeof(*info));
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
    /* ---- Ruiz equilibration */
    S->rs = xm(sizeof(double) * (m + 1)); S->cs = xm(sizeof(double) * (n + 1));
    for (int i = 0; i < m; i++) S->rs[i] = 1.0;
    for (int j = 0; j < n; j++) S->cs[j] = 1.0;
    { double *rmax = xm(sizeof(double) * (m + 1));
      for (int pass = 0; pass < 10; pass++) {
          for (int i = 0; i < m; i++) rmax[i] = 0;
          double dev = 0;
          for (int j = 0; j < n; j++) {
              double cmx = 0;
              for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > cmx) cmx = v; if (v > rmax[S->Ai[p]]) rmax[S->Ai[p]] = v; }
              if (cmx > 0) { const double f = 1.0 / sqrt(cmx); dev = fmax(dev, fabs(1.0 - cmx)); S->cs[j] *= f; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= f; }
          }
          for (int i = 0; i < m; i++) rmax[i] = 0;
          for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double v = fabs(S->Ax[p]); if (v > rmax[S->Ai[p]]) rmax[S->Ai[p]] = v; }
          for (int i = 0; i < m; i++) if (rmax[i] > 0) { dev = fmax(dev, fabs(1.0 - rmax[i])); rmax[i] = 1.0 / sqrt(rmax[i]); S->rs[i] *= rmax[i]; } else rmax[i] = 1.0;
          for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= rmax[S->Ai[p]];
          if (dev < 1e-2) break;
      }
      free(rmax); }
    S->b = xm(sizeof(double) * (m + 1)); S->c = xm(sizeof(double) * (n + 1)); S->u = xm(sizeof(double) * (n + 1)); S->bd = xz(n + 1); S->fr = xz(n + 1);
    for (int j = 0; j < nfree && j < n; j++) S->fr[j] = 1;
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
    if (ne_setup(S, getenv("BRISK_LPNODENSE") ? 0 : 1)) { lps_free(S); info->status = 4; snprintf(info->why, sizeof info->why, "the factor of the normal equations does not fit"); return 4; }
    SChol *F = S->sc;
    info->lnz = (long)schol_nnz(F); info->ndense = S->nd;
    S->theta = xm(sizeof(double) * (n + 1));
    S->w1 = xm(sizeof(double) * (m + 1)); S->w2 = xm(sizeof(double) * (m + 1)); S->w4 = xm(sizeof(double) * (m + 1)); S->w3 = xm(sizeof(double) * (2 * (size_t)m + 2)); S->wn = xm(sizeof(double) * (n + 1)); S->acc = xz(sizeof(double) * (m + 1));
    S->pr = xm(sizeof(double) * (m + 1)); S->pz = xm(sizeof(double) * (m + 1)); S->pp = xm(sizeof(double) * (m + 1)); S->pq = xm(sizeof(double) * (m + 1));
    S->reg = getenv("BRISK_LPREG") ? atof(getenv("BRISK_LPREG")) : 1e-14;
    schol_set_solve_seq(1);
    /* the number of centrality correctors: from the cost of a factorization against a solve */
    int kcorr;
    { const double ff = schol_flops(F) + (double)S->nd * 2.0 * (double)schol_nnz(F), fs = 4.0 * (double)schol_nnz(F) + 4.0 * nz;
      const double ratio = ff / fmax(fs, 1.0);
      double t3 = 50, t2 = 30, t1 = 10;
      if (getenv("BRISK_LPCORRT")) sscanf(getenv("BRISK_LPCORRT"), "%lf,%lf,%lf", &t3, &t2, &t1);
      kcorr = ratio > t3 ? 3 : ratio > t2 ? 2 : ratio > t1 ? 1 : 0;
      if (opt->kcorr >= 0) kcorr = opt->kcorr; }
    if (verbose > 0)
        printf("BRISK LP interior-point solver: %d rows, %d columns (%d with upper bounds), %d nonzeros; normal equations: nnz(L) = %ld, %.2e flops, %d dense column%s; %d corrector%s\n",
               m, n, nbd, nz, info->lnz, schol_flops(F), S->nd, S->nd == 1 ? "" : "s", kcorr, kcorr == 1 ? "" : "s");
#ifdef _OPENMP
    if (verbose > 0) printf("Number of threads: %d (the factorizations and, on large problems, the products with A; the rest is sequential)\n", omp_in_parallel() ? 1 : omp_get_max_threads());
#else
    if (verbose > 0) printf("Number of threads: 1\n");
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
    if (ne_factor(S)) { status = 4; snprintf(info->why, sizeof info->why, "the first factorization failed"); goto done; }
    {
        for (int j = 0; j < n; j++) x[j] = S->bd[j] ? 0.5 * S->u[j] : 0.0;
        A_mul(S, x, rb);
        for (int i = 0; i < m; i++) rb[i] = S->b[i] - rb[i];
        kkt_solve(S, zr, rb, dx, dy, 1e-10, 20);                 /* the least-norm correction: A dx = b - A x */
        for (int j = 0; j < n; j++) { x[j] += dx[j]; s[j] = S->bd[j] ? S->u[j] - x[j] : 0.0; }
        kkt_solve(S, S->c, zm, dx, y, 1e-10, 20);                /* A Theta (A'y - c) = 0 */
        for (int j = 0; j < n; j++) { const double r = -dx[j] / S->theta[j]; if (S->fr[j]) { z[j] = 0; w[j] = 0; } else if (S->bd[j]) { z[j] = 0.5 * r; w[j] = -0.5 * r; } else { z[j] = r; w[j] = 0; } }
        double xmin = 1e300, zmin = 1e300;
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; xmin = fmin(xmin, x[j]); zmin = fmin(zmin, z[j]); if (S->bd[j]) { xmin = fmin(xmin, s[j]); zmin = fmin(zmin, w[j]); } }
        const double dxs = fmax(-1.5 * xmin, 0.0), dzs = fmax(-1.5 * zmin, 0.0);
        double xz_ = 0, sx = 0, sz = 0;
        for (int j = 0; j < n; j++) {
            if (S->fr[j]) continue;
            x[j] += dxs; z[j] += dzs; xz_ += x[j] * z[j]; sx += x[j]; sz += z[j];
            if (S->bd[j]) { s[j] += dxs; w[j] += dzs; xz_ += s[j] * w[j]; sx += s[j]; sz += w[j]; }
        }
        double dx3 = sz > 0 ? 0.5 * xz_ / sz : 0.0, dz3 = sx > 0 ? 0.5 * xz_ / sx : 0.0;
        if (!(dx3 > 1e-8)) dx3 = fmax(dx3, 1e-2);         /* (a start at zero: push inside) */
        if (!(dz3 > 1e-8)) dz3 = fmax(dz3, 1e-2);
        if (getenv("BRISK_LPSTART")) { const double f = atof(getenv("BRISK_LPSTART")); dx3 *= f; dz3 *= f; }
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; x[j] += dx3; z[j] += dz3; if (S->bd[j]) { s[j] += dx3; w[j] += dz3; } }
        if (getenv("BRISK_LPSTART0")) {      /* (test: a flat start, x = z = 1 (bounded: x = s = u/2 at most 1), y = 0) */
            const double f = atof(getenv("BRISK_LPSTART0"));
            for (int i = 0; i < m; i++) y[i] = 0;
            for (int j = 0; j < n; j++) { if (S->fr[j]) { x[j] = 0; continue; } x[j] = S->bd[j] ? fmin(f, 0.5 * S->u[j]) : f; z[j] = f; if (S->bd[j]) { s[j] = S->u[j] - x[j]; w[j] = f; } }
        }
    }
    const double nb0 = nrminf(m, b0), nc0 = nrminf(n, c0);
    double mu0 = -1, best = 1e300; int nstall = 0, nsafe = 0;
    const int steprule = getenv("BRISK_LPSTEP") ? atoi(getenv("BRISK_LPSTEP")) : 1;
    const double frth = getenv("BRISK_LPFREETH") ? atof(getenv("BRISK_LPFREETH")) : 1e6;
    if (verbose > 1) printf("  it      pobj             dobj         pinf     dinf     gap      mu      alp_p alp_d sigma  cg  solres tiny\n");
    tph3 = lp_time();
    for (it = 0; ; it++) {
        /* ---- residuals (scaled problem) and the errors of the problem as given */
        A_mul(S, x, rb);
        for (int i = 0; i < m; i++) rb[i] = S->b[i] - rb[i];
        At_mul(S, y, rc);
        double mu = 0; int ncomp = 0;
        for (int j = 0; j < n; j++) {
            rc[j] = S->c[j] - rc[j] - z[j] + w[j];
            if (S->fr[j]) continue;
            mu += x[j] * z[j]; ncomp++;
            if (S->bd[j]) { ru[j] = S->u[j] - x[j] - s[j]; mu += s[j] * w[j]; ncomp++; }
        }
        mu /= fmax(ncomp, 1);
        if (mu0 < 0) mu0 = mu;
        double ep = 0, ed = 0, pobj = 0, dobj = 0;
        for (int i = 0; i < m; i++) { const double v = fabs(rb[i]) * bsc / S->rs[i]; if (v > ep) ep = v; dobj += S->b[i] * y[i]; }
        for (int j = 0; j < n; j++) {
            const double v = fabs(rc[j]) * csc / S->cs[j]; if (v > ed) ed = v;
            pobj += S->c[j] * x[j];
            if (S->bd[j]) { const double q = fabs(ru[j]) * bsc * S->cs[j]; if (q > ep) ep = q; dobj -= S->u[j] * w[j]; }
        }
        pobj *= bsc * csc; dobj *= bsc * csc;
        const double e_p = ep / (1.0 + nb0), e_d = ed / (1.0 + nc0), e_g = fabs(pobj - dobj) / (1.0 + fabs(pobj));
        const double err = fmax(fmax(e_p, e_d), e_g);
        info->pobj = pobj; info->dobj = dobj; info->pinf = e_p; info->dinf = e_d; info->gap = e_g; info->iters = it;
        if (verbose > 1 && it == 0) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e\n", it, pobj, dobj, e_p, e_d, e_g, mu);
        if (getenv("BRISK_LPDBG")) {
            int im = 0; double vm = 0; for (int i = 0; i < m; i++) { const double v = fabs(rb[i]) / S->rs[i]; if (v > vm) { vm = v; im = i; } }
            double sa = 0, xm_ = 0, ym = nrminf(m, y); for (int j = 0; j < n; j++) { xm_ = fmax(xm_, fabs(x[j])); for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (S->Ai[p] == im) sa += fabs(S->Ax[p] * x[j]); }
            printf("     [dbg: worst row %d: rb %.2e (scaled; max scaled %.2e), rs %.2e, sum|a x| %.2e, b %.2e; max|x| %.2e max|y| %.2e]\n", im, rb[im], nrminf(m, rb), S->rs[im], sa, S->b[im], xm_, ym);
        }
        if (!isfinite(pobj) || !isfinite(dobj) || !isfinite(mu) || !isfinite(ep) || !isfinite(ed)) { status = 4; snprintf(info->why, sizeof info->why, "not finite"); break; }
        if (err <= tol) { status = 0; break; }
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
        if (err < 0.9 * best) { best = err; nstall = 0; } else nstall++;
        if (!isfinite(err) || !isfinite(mu)) { status = 4; snprintf(info->why, sizeof info->why, "not finite"); break; }
        if (nstall >= 25 || mu > 1e12 * fmax(mu0, 1.0) || nrminf(n, x) > 1e14 || nrminf(m, y) > 1e14) { status = 4; snprintf(info->why, sizeof info->why, "%s", nstall >= 25 ? "no progress in 25 iterations" : "the iterates diverge"); break; }
        /* ---- factorization */
        double thmax = 0;
        for (int j = 0; j < n; j++) if (!S->fr[j]) { S->theta[j] = 1.0 / (z[j] / x[j] + (S->bd[j] ? w[j] / s[j] : 0.0)); if (S->theta[j] > thmax) thmax = S->theta[j]; }
        /* free variables: a proximal term rho |dx|^2 / 2, Theta = 1 / rho (the Newton step of the regularized problem) */
        { const double thf = getenv("BRISK_LPFREEREL") ? fmin(fmax(frth, atof(getenv("BRISK_LPFREEREL")) * thmax), 1e12) : frth; for (int j = 0; j < nfree; j++) S->theta[j] = thf; }
        if (ne_factor(S)) { status = 4; snprintf(info->why, sizeof info->why, "factorization failed"); break; }
        const double rtol = 1e-10; const int maxcg = getenv("BRISK_LPCG") ? atoi(getenv("BRISK_LPCG")) : 30;
        const long cg0 = S->ncg;
        /* ---- predictor */
        for (int j = 0; j < n; j++) { rxz[j] = S->fr[j] ? 0.0 : -x[j] * z[j]; rsw[j] = S->bd[j] ? -s[j] * w[j] : 0.0; }
        S->worst = 0;
        direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        if (S->nd > 0 && S->worst > 1e-6) {
            /* the Woodbury correction of the dense columns has lost its accuracy: all columns in the factor, unless
             * the clique that they make over their rows costs more than ten times the factorization with them apart
             * (18 October 2026, LP/SOCP: c^3/3 flops for a clique of c rows; 20 dense columns over 10,000 rows made
             * a dense factor of order 10,000, and over 50,000 rows ran out of memory). The cone solver then takes the
             * problem: its augmented system keeps the dense columns apart. */
            long c = 0; { char *mk = xz(m + 1); for (int d = 0; d < S->nd; d++) { const int j = S->dcol[d]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (!mk[S->Ai[p]]) { mk[S->Ai[p]] = 1; c++; } } free(mk); }
            const double fj = (double)c * c * c / 3.0, fs = schol_flops(S->sc) + S->nd * (4.0 * (double)schol_nnz(S->sc) + 2.0 * S->nd * m);
            if (fj > 10.0 * fs && !getenv("BRISK_LPJOIN")) {
                if (verbose > 1) printf("     (dense columns: residual %.1e; joined they would make a clique of %ld rows, %.1e flops a factorization)\n", S->worst, c, fj);
                status = 4; snprintf(info->why, sizeof info->why, "the dense columns lost their accuracy; joined, %ld rows would be dense", c); break;
            }
            if (verbose > 1) printf("     (dense columns: residual %.1e, they join the factored matrix)\n", S->worst);
            if (ne_setup(S, 0) || ne_factor(S)) { status = 4; snprintf(info->why, sizeof info->why, "the factor with the dense columns does not fit"); break; }
            info->ndense = 0; info->lnz = (long)schol_nnz(S->sc); S->worst = 0;
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        double ap, ad;
        step_max(S, x, s, z, w, dx, ds, dz, dw, &ap, &ad);
        ap = fmin(1.0, ap); ad = fmin(1.0, ad);
        double mua = 0;
        for (int j = 0; j < n; j++) { if (S->fr[j]) continue; mua += (x[j] + ap * dx[j]) * (z[j] + ad * dz[j]); if (S->bd[j]) mua += (s[j] + ap * ds[j]) * (w[j] + ad * dw[j]); }
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
        if (arcon) {
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
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu - x[j] * z[j] - dx[j] * dz[j]; rsw[j] = S->bd[j] ? sigma * mu - s[j] * w[j] - ds[j] * dw[j] : 0.0; }
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx, dy, dz, ds, dw, g, rtol, maxcg);
        }
        step_max(S, x, s, z, w, dx, ds, dz, dw, &ap, &ad);
        /* a blocked step: the second-order term was wrong for this point - a centering step instead */
        if (fmin(ap, ad) < 0.1 && !getenv("BRISK_LPNOSAFE")) {
            sigma = fmax(sigma, 0.5);
            for (int j = 0; j < n; j++) { if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; } rxz[j] = sigma * mu - x[j] * z[j]; rsw[j] = S->bd[j] ? sigma * mu - s[j] * w[j] : 0.0; }
            direction(S, x, s, z, w, rb, ru, rc, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            double bp, bdl;
            step_max(S, x, s, z, w, dx2, ds2, dz2, dw2, &bp, &bdl);
            if (fmin(bp, bdl) > fmin(ap, ad)) {
                memcpy(dx, dx2, sizeof(double) * n); memcpy(dz, dz2, sizeof(double) * n); memcpy(ds, ds2, sizeof(double) * n); memcpy(dw, dw2, sizeof(double) * n); memcpy(dy, dy2, sizeof(double) * m);
                ap = bp; ad = bdl; nsafe++;
            }
        }
        /* ---- Gondzio's centrality correctors */
        for (int k = 0; k < kcorr; k++) {
            const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
            if (a1 >= 0.999 && a2 >= 0.999) break;
            const double tp = fmin(1.0, 1.08 * a1 + 0.08), td = fmin(1.0, 1.08 * a2 + 0.08);
            const double mt = sigma * mu, lo = 0.1 * mt, hi = 10.0 * mt;
            for (int j = 0; j < n; j++) {
                if (S->fr[j]) { rxz[j] = 0; rsw[j] = 0; continue; }
                double v = (x[j] + tp * dx[j]) * (z[j] + td * dz[j]);
                double t = v < lo ? lo - v : v > hi ? hi - v : 0.0;
                if (t < -hi) t = -hi;
                rxz[j] = t;
                if (S->bd[j]) { v = (s[j] + tp * ds[j]) * (w[j] + td * dw[j]); t = v < lo ? lo - v : v > hi ? hi - v : 0.0; if (t < -hi) t = -hi; rsw[j] = t; } else rsw[j] = 0;
            }
            direction(S, x, s, z, w, zm, zr, zr, rxz, rsw, dx2, dy2, dz2, ds2, dw2, g, rtol, maxcg);
            for (int j = 0; j < n; j++) { dx2[j] += dx[j]; dz2[j] += dz[j]; ds2[j] += ds[j]; dw2[j] += dw[j]; }
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
            for (int j = 0; j < n; j++) { if (S->fr[j]) continue; muf += (x[j] + a1 * dx[j]) * (z[j] + a2 * dz[j]); if (S->bd[j]) muf += (s[j] + a1 * ds[j]) * (w[j] + a2 * dw[j]); }
            muf = fmax(muf, 0.0) / fmax(ncomp, 1);
            /* the blocking components */
            int kp = -1, kd = -1, tp_ = 0, td_ = 0; double bp_ = 1e300, bd_ = 1e300;
            for (int j = 0; j < n; j++) {
                if (S->fr[j]) continue;
                if (dx[j] < 0) { const double t = -x[j] / dx[j]; if (t < bp_) { bp_ = t; kp = j; tp_ = 0; } }
                if (dz[j] < 0) { const double t = -z[j] / dz[j]; if (t < bd_) { bd_ = t; kd = j; td_ = 0; } }
                if (S->bd[j]) {
                    if (ds[j] < 0) { const double t = -s[j] / ds[j]; if (t < bp_) { bp_ = t; kp = j; tp_ = 1; } }
                    if (dw[j] < 0) { const double t = -w[j] / dw[j]; if (t < bd_) { bd_ = t; kd = j; td_ = 1; } }
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
        for (int j = 0; j < n; j++) { x[j] += ap * dx[j]; z[j] += ad * dz[j]; if (S->bd[j]) { s[j] += ap * ds[j]; w[j] += ad * dw[j]; } }
        for (int i = 0; i < m; i++) y[i] += ad * dy[i];
        if (verbose > 1) printf("%4d %+.9e %+.9e %8.1e %8.1e %8.1e %8.1e %5.3f %5.3f %6.0e %3ld %7.0e %d\n", it + 1, pobj, dobj, e_p, e_d, e_g, mu, ap, ad, sigma, S->ncg - cg0, S->worst, schol_ntiny(S->sc));
    }
    (void)nsafe;
done:
    if (getenv("BRISK_LPDBG")) printf("   [LP phases: copy %.3fs, scaling %.3fs, pattern and analysis %.3fs, start %.3fs, iterations %.3fs (factorizations %.3fs, solves %.3fs)]\n", tph0 - t0, tph1 - tph0, tph2 - tph1, tph3 - tph2, lp_time() - tph3, S->t_fact, S->t_solve);
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
typedef struct { double flops, lnz; int nd, ok; } LpNeCost;
static void ne_cost(int m, int n, const int *Ap, const int *Ai, const double *Ax, LpNeCost *C, double budget) {
    LpS T; memset(&T, 0, sizeof T); T.m = m; T.n = n; T.est = 1; T.est_budget = budget;
    T.Ap = (int *)Ap; T.Ai = (int *)Ai; T.Ax = (double *)Ax;
    C->ok = ne_setup(&T, 1) == 0;
    if (C->ok && T.est_lb > 0) { C->flops = T.est_lb; C->lnz = (double)T.est_w; C->nd = T.nd; T.Ap = NULL; T.Ai = NULL; T.Ax = NULL; lps_free(&T); return; }
    C->flops = C->ok ? schol_flops(T.sc) : 1e300; C->lnz = C->ok ? (double)schol_nnz(T.sc) : 1e300; C->nd = T.nd;
    T.Ap = NULL; T.Ai = NULL; T.Ax = NULL;
    lps_free(&T);
}
/* the cost of one iteration in flops: a factorization (with the dense columns' solves and Schur complement)
 * and about ten solves and products with A (the predictor, the corrector, the correctors, the refinement) */
static double ne_iter_cost(const LpNeCost *C, int m, long nnz) {
    return C->flops + C->nd * (4.0 * C->lnz + 2.0 * C->nd * m) + 10.0 * (4.0 * C->lnz + 4.0 * (double)nnz + 4.0 * C->nd * m);
}
static void sort_cols(int n, const int *Ap, int *Ai, double *Ax) {
    for (int j = 0; j < n; j++) {
        const int a = Ap[j], e = Ap[j + 1];
        for (int p = a + 1; p < e; p++) { const int r = Ai[p]; const double v = Ax[p]; int q = p - 1; while (q >= a && Ai[q] > r) { Ai[q + 1] = Ai[q]; Ax[q + 1] = Ax[q]; q--; } Ai[q + 1] = r; Ax[q + 1] = v; }
    }
}
static int lp_dual_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                         const LpIpmOpts *opt, double *xout, double *yout, double *zout, LpIpmInfo *info) {
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
    double cp = 0, cd = 0;
    if (mode == 2) {
        /* the cost model: one iteration of each form, from the symbolic factorizations (AMD). The dual's first: the
         * primal's analysis is skipped when the assembly of its normal equations alone, sum over its sparse columns of
         * len (len + 1) / 2 multiply-adds (a lower bound on its iteration), costs more than twice the dual's iteration
         * (a sparse quantile regression 30,000 x 501: the primal's analysis took 22-55 s for a solve of 0.27 s); and its
         * ordering is not made when its pattern alone shows that it costs more (ne_setup's budget) */
        LpNeCost P, D;
        ne_cost(m2, n2, Bp, Bi, Bx, &D, 0.0);
        cd = ne_iter_cost(&D, m2, nz2);
        double lbp = 0;
        {   const double avg = (double)nnz / fmax(n, 1), thr = fmax(fmax(50.0, 10.0 * avg), 0.1 * m);
            int ndn = 0; for (int j = 0; j < n; j++) if (m >= 500 && Ap[j + 1] - Ap[j] > thr) ndn++;
            const int usedn = ndn > 0 && ndn <= 400 && ndn < n / 4;
            for (int j = 0; j < n; j++) { const double l = Ap[j + 1] - Ap[j]; if (!(usedn && l > thr)) lbp += 0.5 * l * (l + 1); } }
        if (D.ok && lbp > 2.0 * cd) { P.ok = 0; cp = lbp; }
        else {
            int *Ai1 = xm(sizeof(int) * (nnz + 1)); double *Ax1 = xm(sizeof(double) * (nnz + 1));
            memcpy(Ai1, Ai, sizeof(int) * nnz); memcpy(Ax1, Ax, sizeof(double) * nnz); sort_cols(n, Ap, Ai1, Ax1);
            ne_cost(m, n, Ap, Ai1, Ax1, &P, D.ok ? 2.0 * cd : 0.0); free(Ai1); free(Ax1);
            cp = ne_iter_cost(&P, m, nnz);
        }
        /* the dual when it costs at most half: the primal is the form the method is tuned on, and its free
         * variables (the dual's free y) enter both as a proximal term */
        choose = D.ok && cd <= 0.5 * cp;
    }
    if (opt->verbose > 0 && (choose || getenv("BRISK_LPDBG")))
        printf("LP dual form %s: %d rows instead of %d (%d bounds on y from singleton columns, %d free y); cost of an iteration %.2e against %.2e flops\n",
               choose ? "chosen" : "not chosen", m2, m, m - nyf, nyf, cd, cp);
    int st = -1;
    if (choose) {
        double *x2 = xz(sizeof(double) * (n2 + 1)), *y2 = xz(sizeof(double) * (m2 + 1)), *zl2 = xz(sizeof(double) * (n2 + 1)), *wu2 = xz(sizeof(double) * (n2 + 1));
        LpIpmInfo inf2;
        st = lpipm_core(m2, n2, nyf, Bp, Bi, Bx, b2, c2, u2, opt, x2, y2, NULL, &inf2, zl2, wu2);
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
        info->pinf = ep / (1.0 + nrminf(m, b)); info->dinf = ed / (1.0 + nrminf(n, c)); info->gap = fabs(pobj - dobj) / (1.0 + fabs(pobj));
        free(x2); free(y2); free(zl2); free(wu2);
    }
    free(rpos); free(lo); free(up); free(jlo); free(jup); free(ycol); free(ysgn); free(ys0);
    free(Bp); free(Bi); free(Bx); free(b2); free(c2); free(u2);
    return st;
}
int lpipm_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info) {
    /* (18 October 2026: the busy-machine rule of the other engines, solver.c: only the free cores; restored on return) */
#ifdef _OPENMP
    const int ntb = omp_in_parallel() ? -1 : brisk_threads_busy(0);
#endif
    int st = lp_dual_solve(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info);
    if (st < 0) st = lpipm_core(m, n, nfree, Ap, Ai, Ax, b, c, u, opt, x, y, z, info, NULL, NULL);
#ifdef _OPENMP
    if (ntb > 0) omp_set_num_threads(ntb);
#endif
    return st;
}
