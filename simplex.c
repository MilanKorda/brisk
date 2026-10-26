/* simplex.c - the dual and the primal simplex method for linear programs in general form (simplex.h).
 *
 * Computational form: the n columns of A and one logical per row, s = -A x, so that [A I] (x, s) = 0 and every one of
 * the n + m variables has bounds (the logical of row i: -ru_i <= s_i <= -rl_i). The starting basis is that of the
 * logicals (B = I). The problem is scaled (geometric means, then equilibration; powers of two).
 *
 * Dual simplex (the default): the leaving row by dual steepest edge (Forrest and Goldfarb 1992), the entering column by
 * the bound-flipping ratio test with Harris' tolerances (Fourer 1994, Maros 2003, Koberstein 2005), costs perturbed at
 * the start against dual degeneracy and shifted where a tolerance step leaves a small dual infeasibility; dual phase 1
 * by the auxiliary problem with boxes (each variable's bounds replaced by [-1000, 1000] if free, [0, 1] or [-1, 0] if
 * bounded on one side, [0, 0] if boxed), whose optimal basis is dual feasible for the problem when its objective is 0.
 * Primal simplex: Devex pricing (Harris 1973), Harris' two-pass ratio test with bound shifting, phase 1 on the sum of
 * infeasibilities; it also removes the dual infeasibilities left when the perturbation of the costs is taken away.
 *
 * The basis factorization and its updates: simplex_lu.c. */
#include "simplex.h"
#include "simplex_int.h"
#include "simplex_pre.h"
#include "lpsolve.h"
#include "socp.h"
#include "amd/amd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif
#include <unistd.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_OPENMP) && !defined(BRISK_OMP_BUNDLED) && !defined(_MSC_VER)
#include <pthread.h>
#define SX_RACE 1             /* the automatic choice runs the interior-point method on a thread of its own (auto_route) */
int brisk_threads_busy(int verbose);   /* (solver.c: lowers the thread count to the free cores; returns the count to restore, or -1) */
#endif
#define SX_PAR_FTRAN 2000     /* expected entries of the two solves of an iteration from which they run in parallel */

#define SX_INF 1e30
#define HUGEB 1e29     /* a bound beyond this is infinite */
#define SX_TARGET_DEF 1e-7  /* largest violation accepted on the problem as read (relative: unscaled_violations); the
                               method's own tolerances (1e-7). 1e-9 until 21 October 2026: on 81 MIPLIB relaxations the
                               same results (4 differ by 4-11 iterations, largest dual violation 1.4e-8 -> 8.3e-8), and
                               supportcase19 from an optimal crossover basis no longer runs the primal method for 11,700
                               iterations to tighten 2.3e-9 (unsolved in 300 s -> 47 s at four threads) */
/* (BRISK_SXTARGET: an experiment; read by sx_setup into env_target) */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* an indexed max-heap over 0..cap-1: keys > 0 are in it; ties go to the lower index (the choice is the same as a
 * scan's that takes the first largest) */
typedef struct { int *h, *pos, n; double *key; } SxHeap;
#define SX_PC_K 128          /* the primal method's candidate list: its length after a scan (pheap_rebuild) */
static void hp_init(SxHeap *H, int cap) { H->h = malloc(sizeof(int) * (cap + 1)); H->pos = malloc(sizeof(int) * (cap + 1)); H->key = malloc(sizeof(double) * (cap + 1)); H->n = 0; for (int i = 0; i < cap; i++) H->pos[i] = -1; }
static void hp_free(SxHeap *H) { free(H->h); free(H->pos); free(H->key); }
static int hp_better(const SxHeap *H, int a, int b) { return H->key[a] > H->key[b] || (H->key[a] == H->key[b] && a < b); }
static void hp_swap(SxHeap *H, int p, int q) { const int a = H->h[p], b = H->h[q]; H->h[p] = b; H->h[q] = a; H->pos[b] = p; H->pos[a] = q; }
static void hp_up(SxHeap *H, int p) { while (p > 0) { const int q = (p - 1) / 2; if (!hp_better(H, H->h[p], H->h[q])) break; hp_swap(H, p, q); p = q; } }
static void hp_down(SxHeap *H, int p) {
    for (;;) {
        const int l = 2 * p + 1, r = l + 1; int b = p;
        if (l < H->n && hp_better(H, H->h[l], H->h[b])) b = l;
        if (r < H->n && hp_better(H, H->h[r], H->h[b])) b = r;
        if (b == p) break;
        hp_swap(H, p, b); p = b;
    }
}
/* set the key of i (k <= 0: out of the heap) */
static void hp_set(SxHeap *H, int i, double k) {
    const int p = H->pos[i];
    if (!(k > 0)) {
        if (p < 0) return;
        const int last = H->n - 1;
        if (p != last) { hp_swap(H, p, last); H->n--; H->pos[i] = -1; hp_down(H, p); hp_up(H, p); }
        else { H->n--; H->pos[i] = -1; }
        return;
    }
    if (p < 0) { H->key[i] = k; H->pos[i] = H->n; H->h[H->n++] = i; hp_up(H, H->n - 1); return; }
    const double old = H->key[i];
    H->key[i] = k;
    if (k > old) hp_up(H, p); else if (k < old) hp_down(H, p);
}
static void hp_clear(SxHeap *H) { for (int k = 0; k < H->n; k++) H->pos[H->h[k]] = -1; H->n = 0; }

typedef struct {
    /* problem (scaled), n structurals and m logicals */
    int m, n, N;
    int *Ap, *Ai; double *Ax;          /* A by columns */
    int *Rp, *Ri; double *Rx;          /* A by rows; in each row the entries of the nonbasic columns first */
    int *Rnb;                          /* the number of nonbasic entries of each row */
    int *Rc, *Cr;                      /* the entry's position by columns (of each by-rows entry) and by rows (of each
                                          by-columns entry): an entering or leaving column moves in its rows directly */
    double *cscale, *rscale;           /* x = cscale * x_s; row i scaled by rscale_i */
    double *lo0, *up0, *cost0;         /* the scaled problem's bounds and costs (N) */
    double *lo, *up, *cost;            /* working bounds and costs (phase 1 boxes, perturbation, shifts) */
    double *shift;                     /* the cost shift of each variable (removed when it leaves the basis) */
    double costscale;
    int maxim;                         /* the problem as read maximizes (the costs here are negated) */
    /* state */
    int *basic;                        /* basic[i]: variable in position (row) i */
    int *pos;                          /* pos[j]: position of a basic variable, -1 nonbasic */
    signed char *move;                 /* nonbasic: +1 at lower (may increase), -1 at upper, 0 fixed or free */
    double *x;                         /* values of all variables */
    double *d;                         /* reduced costs (0 for basic) */
    double *w;                         /* dual steepest-edge weights (by position) */
    double *dvx;                       /* Devex weights (by variable) */
    double tolp, told;
    double cpiv;                /* the dual ratio test's pivot tolerance (0: 1e-7; 1e-9 for the second test of an infeasibility claim) */
    int nfkretry, nfkfail;      /* infeasibility claims not proved: the second ratio test found a column / none (status 4) */
    /* factorization and work vectors */
    SxLU F;
    SxVec rho, col, tau, bfrt, spike, upart, row;   /* row: the pivotal row over all N variables */
    double dens_col, dens_rho, dens_tau, dens_row;
    double sw_ew, sw_base; int sw_n, refac_meas;   /* the solves' work since the factorization (need_refactor; 24 October 2026) */
    double dens_taulr;                 /* density of the weights' solve after L and R (when restricted to the column) */
    int taupart;                       /* the weights' solve may be restricted to the entering column's nonzeros (ftran2; 2: always) */
    double tw_part, tw_full;           /* the work of the weights' restricted and full solves lately (ftran2) */
    long tw_last, n_taupart, n_taufull; /* the last trial of the solve not chosen; the counts of each (ftran2) */
    int price_dense_div;               /* the pricing by rows accumulates densely past N / this many entries (8; BRISK_SXPRICEDIV: an experiment) */
    int *out;
    /* the leaving-row candidates: an indexed max-heap of the rows with a primal infeasibility, keyed by
     * infeasibility^2 / weight (ties: the lower row index), updated for the rows an iteration changes */
    int *heap, *hpos, nheap; double *hkey;
    int heap_ok;
    char *inI; int *Ilist, nI, Ifull;  /* the rows that may be primal infeasible (a superset; Ifull: unknown, all rows) */
    /* candidates of the ratio test */
    int *cand; double *ca, *cr; int ncand;
    double *ck, *chm; int *cp, *cti; double *ctd;   /* (the ratio test's sorted groups: ratios, Harris suffix minima, order, work) */
    /* bookkeeping */
    long iter, it_dual, it_primal, it_ph1;
    long nprice_row, nprice_col, nprice_dense; double sum_rho, sum_col, sum_rowc; long ncolstat;
    int nbshift, ndvxreset, it_given;
    SxHeap ch;                         /* the primal method's entering candidates: d_j^2 / Devex weight */
    int pc_heap, pc_n, pc_cap, *pc_set, *pc_at; double *pc_kv, *pc_mb, pc_tau;   /* (the candidate list instead of the heap: pc_set_key) */
    int ch_ok;
    signed char *c1;                   /* phase 1 of the primal method: the cost (-1, 0, +1) of the basic variable of each row */
    double *rayout; int rayrow, have_ray; double raymo;
    int nfactor, nflips, nshift, nsh_free, nsh_reb, nsh_it, nflip_reb, nflip_it, nshiftback; double shmax;
    int updlim;
    double t0, tlim, t_factor, t_ftran, t_btran, t_price;
    long maxit;
    int verbose, dbg;
    double work, workbudget;           /* entries touched by solves, pricing and factorizations (deterministic); a limit */
    double (*budget_cb)(void *, double); void *budget_ctx;   /* called when the work passes the budget: a new one */
    int nthreads; long npar_ftran; SxWs ws2;   /* the threads; the solves run in parallel; the second solve's workspace */
    const volatile int *threads; int cur_threads;   /* (the race) the threads allowed now, and those set */
    int keepx;                         /* the crossover: a factorization keeps the values of the variables it replaces */
    const volatile int *stop;
    unsigned long long rng;
    int phase;                         /* 1 dual phase 1 (boxes), 2 phase 2 */
    int perturbed, shifted;
    double pert_base; int pert_level, pert_pending; long zw_it, zw_zero, nrepert;   /* the perturbation's base; its escalations on stalling */
    int lazy_pert;                     /* (the re-solves: the costs perturbed at the first stall, not at the start) */
    long nchk;                         /* passes through check_stop (the clock read every 16th) */
    long xo_dnoblock;                  /* the crossover's dual pushes that changed no basis (a measurement) */
    /* (22 October 2026) the persistent state for re-solves (sxm_*, lpmodel.c) */
    double env_slackpref;
    int env_stallobj;                  /* (experiment, SXSTALLOBJ=1) the perturbation renewed when the objective does not move */
    double stall_obj;                  /* the objective at the last window's end */
    int env_ddevex;                    /* (experiment, SXDUALDEVEX=1) the dual method's leaving row by Devex weights: no solve for
                                          the weights (B^-1 rho_r), the update from the entering column alone */       /* (experiment, SXSLACKPREF) see chuzc_dual */
    int fresh;                  /* (sxm_solve) the factorization, the values and the reduced costs are current at the next
                                   loop's entry: no factorization there */
    int noverify;               /* (sxm_solve) no new factorization to confirm optimality at the loop's end: sxm_solve
                                   checks the violations on the problem as given and factorizes only when they are above
                                   its target (22 October 2026, the mixed-integer developer's request: two factorizations
                                   in every re-solve, 2.0 per re-solve on binkar10_1, 85% of a dive's time on cod105) */
    int cut_on; double cut_val, cut_bound;   /* the dual method stops (status 7) once its dual bound (scaled, minimized)
                                                passes cut_val; the bound then */
    double *prayout; int have_pray;    /* an unbounded primal: the direction of all N variables (scaled) */
    const double *cs_in, *rs_in; int m_in; double costscale_in;   /* sx_setup: a scaling given (the first m_in rows') */
    /* experiments' settings from the environment, read by sx_setup (not into static variables: objects of the re-solve
     * interface start on several threads at once, ISSUES 422) */
    double env_refac_w, env_target; int env_noshiftback, env_nostall;
    long ntiny;                        /* entries dropped by sx_setup as too small to matter */
} Sx;

static double urand(Sx *S) { S->rng ^= S->rng << 13; S->rng ^= S->rng >> 7; S->rng ^= S->rng << 17; return (S->rng >> 11) * (1.0 / 9007199254740992.0); }

/* ---------------------------------------------------------------- setup and scaling */
static double pow2round(double s) { int e; frexp(s, &e); return ldexp(1.0, e - 1); }   /* the power of two <= s (within a factor 2) */

static void build_rows(Sx *S) {
    const int m = S->m, n = S->n, nz = S->Ap[n];
    S->Rp = xcalloc(m + 1, sizeof(int)); S->Ri = xmalloc(sizeof(int) * (nz + 1)); S->Rx = xmalloc(sizeof(double) * (nz + 1));
    for (int p = 0; p < nz; p++) S->Rp[S->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) S->Rp[i + 1] += S->Rp[i];
    int *c = xmalloc(sizeof(int) * (m + 1));
    memcpy(c, S->Rp, sizeof(int) * m);
    S->Rc = xmalloc(sizeof(int) * (nz + 1)); S->Cr = xmalloc(sizeof(int) * (nz + 1));
    for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int q = c[S->Ai[p]]++; S->Ri[q] = j; S->Rx[q] = S->Ax[p]; S->Rc[q] = p; S->Cr[p] = q; }
    free(c);
    S->Rnb = xmalloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) S->Rnb[i] = S->Rp[i + 1] - S->Rp[i];
}

/* geometric scaling passes (rows then columns: 1/sqrt(min max) of the entries), then equilibration of the columns
 * (largest entry 1); all factors powers of two, so that scaling is exact */
static void scale(Sx *S, int on) {
    const int m = S->m, n = S->n;
    double *r = S->rscale, *c = S->cscale;
    for (int i = 0; i < m; i++) r[i] = 1;
    for (int j = 0; j < n; j++) c[j] = 1;
    if (!on) return;
    double *rmin = xmalloc(sizeof(double) * m), *rmax = xmalloc(sizeof(double) * m);
    double prev = SX_INF;
    for (int pass = 0; pass < 8; pass++) {
        for (int i = 0; i < m; i++) { rmin[i] = SX_INF; rmax[i] = 0; }
        double gmin = SX_INF, gmax = 0;
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j];
            if (a == 0) continue;
            if (a < rmin[S->Ai[p]]) rmin[S->Ai[p]] = a;
            if (a > rmax[S->Ai[p]]) rmax[S->Ai[p]] = a;
            if (a < gmin) gmin = a;
            if (a > gmax) gmax = a;
        }
        const double ratio = gmax > 0 ? gmax / gmin : 1;
        if (pass > 0 && ratio > 0.9 * prev) break;
        prev = ratio;
        for (int i = 0; i < m; i++) if (rmax[i] > 0) r[i] /= sqrt(rmin[i] * rmax[i]);
        for (int j = 0; j < n; j++) {
            double mn = SX_INF, mx = 0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a == 0) continue; if (a < mn) mn = a; if (a > mx) mx = a; }
            if (mx > 0) c[j] /= sqrt(mn * mx);
        }
    }
    /* equilibrate columns to a largest entry of about 1 */
    for (int j = 0; j < n; j++) {
        double mx = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > mx) mx = a; }
        if (mx > 0) c[j] /= mx;
    }
    for (int i = 0; i < m; i++) r[i] = pow2round(r[i]);
    for (int j = 0; j < n; j++) c[j] = pow2round(c[j]);
    free(rmin); free(rmax);
}

/* (25 October 2026) the entries of A worth keeping (keep[p] = 1): not the explicit zeros, nor the entries too small to
 * matter: an entry below 1e-9 of its row's largest, of a column with finite bounds, while the row's dropped entries can
 * move its activity by at most sum |a_ij| max(|l_j|, |u_j|) <= 1e-10 (a tenth of the feasibility tolerance; SCIP's
 * presolve allows a hundredth of its tolerance 1e-6, and SCIP and HiGHS take any entry up to 1e-9 as zero). Such entries
 * steer the geometric scaling (a row's factor from its smallest and largest entries: 1e-20 beside 1 scaled the row's
 * other entries by 1e10) and give the factorization near-singular bases. momentum2 (MIPLIB 2017) has 95,000 entries
 * below 1e-12, down to 3e-21: the mixed-integer developer's first LP of it, on its own form, ran over 60 s with the
 * primal infeasibility at 1e36 (request 8); without them 3.1 s. Applied where the bounds are those given (simplex_solve,
 * before the presolve drops the bounds the rows imply, and sx_setup). The number dropped returned. BRISK_SXTINY=0: off */
static long tiny_keep(const LpProb *L, unsigned char *keep) {
    const int m = L->m, n = L->n, nz = L->Ap[n];
    for (int p = 0; p < nz; p++) keep[p] = L->Ax[p] != 0;
    static int off = -1; if (off < 0) { const char *e = getenv("BRISK_SXTINY"); off = e && atoi(e) == 0; }
    if (off) return 0;
    double *rmx = xcalloc(m + 1, sizeof(double)), *rdrop = xcalloc(m + 1, sizeof(double));
    for (int p = 0; p < nz; p++) { const double a = fabs(L->Ax[p]); if (a > rmx[L->Ai[p]]) rmx[L->Ai[p]] = a; }
    long nd = 0;
    for (int j = 0; j < n; j++) {
        if (!(L->l[j] > -SX_INF && L->u[j] < SX_INF)) continue;
        const double bj = fmax(fabs(L->l[j]), fabs(L->u[j]));
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) {
            const int i = L->Ai[p]; const double a = fabs(L->Ax[p]);
            if (keep[p] && a < 1e-9 * rmx[i] && rdrop[i] + a * bj <= 1e-10) { rdrop[i] += a * bj; keep[p] = 0; nd++; }
        }
    }
    free(rmx); free(rdrop);
    return nd;
}

static void sx_setup(Sx *S, const LpProb *L, const SimplexOpts *o) {
    const int m = L->m, n = L->n, N = n + m;
    S->m = m; S->n = n; S->N = N;
    const int nz = L->Ap[n];
    S->Ap = xmalloc(sizeof(int) * (n + 1)); S->Ai = xmalloc(sizeof(int) * (nz + 1)); S->Ax = xmalloc(sizeof(double) * (nz + 1));
    /* copy A dropping explicit zeros and the entries too small to matter (tiny_keep) */
    unsigned char *keep = xmalloc((size_t)nz + 1);
    S->ntiny = tiny_keep(L, keep);
    int e = 0;
    for (int j = 0; j < n; j++) {
        S->Ap[j] = e;
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (keep[p]) { S->Ai[e] = L->Ai[p]; S->Ax[e] = L->Ax[p]; e++; }
    }
    S->Ap[n] = e;
    free(keep);
    S->rscale = xmalloc(sizeof(double) * (m + 1)); S->cscale = xmalloc(sizeof(double) * (n + 1));
    if (S->cs_in) {
        /* (22 October 2026, sxm_new) the scaling given: the columns' and the first m_in rows' factors kept, so that a
         * re-solve after rows were added or deleted works on the same scaled problem; a new row by one geometric pass
         * over its entries as the columns scale them */
        memcpy(S->cscale, S->cs_in, sizeof(double) * n);
        double *rmin = xmalloc(sizeof(double) * (m + 1)), *rmax = xcalloc(m + 1, sizeof(double));
        for (int i = 0; i < m; i++) rmin[i] = SX_INF;
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p]; if (i < S->m_in) continue;
            const double a = fabs(S->Ax[p]) * S->cscale[j];
            if (a < rmin[i]) rmin[i] = a;
            if (a > rmax[i]) rmax[i] = a;
        }
        for (int i = 0; i < m; i++) S->rscale[i] = i < S->m_in ? S->rs_in[i] : rmax[i] > 0 ? pow2round(1.0 / sqrt(rmin[i] * rmax[i])) : 1.0;
        free(rmin); free(rmax);
    } else scale(S, !(o && o->noscale));
    for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= S->rscale[S->Ai[p]] * S->cscale[j];
    build_rows(S);
    S->lo0 = xmalloc(sizeof(double) * N); S->up0 = xmalloc(sizeof(double) * N); S->cost0 = xmalloc(sizeof(double) * N);
    const double sg = L->maxim ? -1.0 : 1.0;
    S->maxim = L->maxim;
    double cmax = 0;
    for (int j = 0; j < n; j++) {
        const double cs = S->cscale[j];
        S->lo0[j] = L->l[j] <= -HUGEB ? -SX_INF : L->l[j] / cs;
        S->up0[j] = L->u[j] >= HUGEB ? SX_INF : L->u[j] / cs;
        S->cost0[j] = sg * L->c[j] * cs;
        if (fabs(S->cost0[j]) > cmax) cmax = fabs(S->cost0[j]);
    }
    for (int i = 0; i < m; i++) {
        const double rs = S->rscale[i];
        S->lo0[n + i] = L->ru[i] >= HUGEB ? -SX_INF : -L->ru[i] * rs;
        S->up0[n + i] = L->rl[i] <= -HUGEB ? SX_INF : -L->rl[i] * rs;
        S->cost0[n + i] = 0;
    }
    /* the costs as given, unless the largest exceeds 1e4: then divided by a power of two that brings it to about 1e4,
     * so that the absolute dual tolerance (1e-7) stays above the rounding errors of the reduced costs (about 1e-16 times
     * the largest cost). A scaling of all costs by the largest made the tolerance and the cost perturbation negligible
     * for the many small costs of problems with a few large ones: stalling, ten times the iterations on some */
    {
        const double cbig = getenv("SXCOSTBIG") ? atof(getenv("SXCOSTBIG")) : 1e4;   /* (experiments) */
        S->costscale = cmax > cbig ? pow2round(cmax / cbig) : 1.0;
    }
    if (getenv("SXNOCOSTSCALE")) S->costscale = 1.0;
    if (S->costscale_in > 0) S->costscale = S->costscale_in;
    if (getenv("SXDBG")) {
        double a = 1e300, b = 0, c = 1e300, d = 0;
        for (int j = 0; j < n; j++) { a = fmin(a, S->cscale[j]); b = fmax(b, S->cscale[j]); }
        for (int i = 0; i < m; i++) { c = fmin(c, S->rscale[i]); d = fmax(d, S->rscale[i]); }
        printf("scaling: columns %.1e .. %.1e, rows %.1e .. %.1e, costs / %.1e\n", a, b, c, d, S->costscale);
    }
    for (int j = 0; j < n; j++) S->cost0[j] /= S->costscale;
    S->lo = xmalloc(sizeof(double) * N); S->up = xmalloc(sizeof(double) * N); S->cost = xmalloc(sizeof(double) * N); S->shift = xcalloc(N, sizeof(double));
    memcpy(S->lo, S->lo0, sizeof(double) * N); memcpy(S->up, S->up0, sizeof(double) * N); memcpy(S->cost, S->cost0, sizeof(double) * N);
    S->basic = xmalloc(sizeof(int) * (m + 1)); S->pos = xmalloc(sizeof(int) * N); S->move = xmalloc(N + 8);
    S->x = xcalloc(N, sizeof(double)); S->d = xcalloc(N, sizeof(double)); S->w = xmalloc(sizeof(double) * (m + 1)); S->dvx = xmalloc(sizeof(double) * N);
    sxv_init(&S->rho, m); sxv_init(&S->col, m); sxv_init(&S->tau, m); sxv_init(&S->bfrt, m); sxv_init(&S->spike, m); sxv_init(&S->upart, m);
    sxv_init(&S->row, N);
    S->out = xmalloc(sizeof(int) * (m + 1));
    S->heap = xmalloc(sizeof(int) * (m + 1)); S->hpos = xmalloc(sizeof(int) * (m + 1)); S->hkey = xmalloc(sizeof(double) * (m + 1));
    S->inI = xcalloc(m + 1, 1); S->Ilist = xmalloc(sizeof(int) * (m + 1)); S->nI = 0; S->Ifull = 1;
    S->nthreads = 1;
#ifdef _OPENMP
    S->nthreads = omp_get_max_threads();
    if (o && o->threads && *o->threads > S->nthreads) S->nthreads = *o->threads;
#endif
    sxws_init(&S->ws2, m);
    hp_init(&S->ch, N); S->c1 = xcalloc(m + 1, 1);
    S->pc_heap = getenv("BRISK_SXPHEAP") && atoi(getenv("BRISK_SXPHEAP")) == 1;
    S->pc_cap = 2 * SX_PC_K; S->pc_n = 0; S->pc_tau = 0;
    S->pc_set = xmalloc(sizeof(int) * (S->pc_cap + 1)); S->pc_kv = xmalloc(sizeof(double) * (S->pc_cap + 1));
    S->pc_at = xmalloc(sizeof(int) * (N + 1)); for (int j = 0; j < N; j++) S->pc_at[j] = -1;
    S->pc_mb = NULL;
    S->cand = xmalloc(sizeof(int) * N); S->ca = xmalloc(sizeof(double) * N); S->cr = xmalloc(sizeof(double) * N);
    S->ck = xmalloc(sizeof(double) * (N + 1)); S->chm = xmalloc(sizeof(double) * (N + 1)); S->cp = xmalloc(sizeof(int) * (N + 1));
    S->cti = xmalloc(sizeof(int) * (N + 1)); S->ctd = xmalloc(sizeof(double) * (N + 1));
    S->tolp = o && o->tol_primal > 0 ? o->tol_primal : 1e-7;
    S->told = o && o->tol_dual > 0 ? o->tol_dual : 1e-7;
    S->dens_col = S->dens_rho = S->dens_tau = S->dens_row = S->dens_taulr = 0;
    S->taupart = getenv("BRISK_SXTAUFULL") ? 0 : getenv("BRISK_SXTAUPART") ? 2 : 1;   /* (switches: the full solve always, the restricted one always) */
    S->price_dense_div = getenv("BRISK_SXPRICEDIV") ? atoi(getenv("BRISK_SXPRICEDIV")) : 8; if (S->price_dense_div < 1) S->price_dense_div = 8;
    S->updlim = getenv("BRISK_SXUPDLIM") ? atoi(getenv("BRISK_SXUPDLIM")) : 1000;   /* (the switch: an experiment, 24 October 2026) */
    S->refac_meas = !(getenv("BRISK_SXREFACMEAS") && atoi(getenv("BRISK_SXREFACMEAS")) == 0);
    S->rng = 0x9E3779B97F4A7C15ULL;
    { const char *e = getenv("SXREFAC_W"); S->env_refac_w = e ? atof(e) : 10.0; }
    { const char *e = getenv("SXSLACKPREF"); S->env_slackpref = e ? atof(e) : 0; }
    { const char *e = getenv("SXDUALDEVEX"); S->env_ddevex = e ? atoi(e) : 0; }
    { const char *e = getenv("SXSTALLOBJ"); S->env_stallobj = e ? atoi(e) : 0; S->stall_obj = NAN; }
    { const char *e = getenv("BRISK_SXTARGET"); S->env_target = e ? atof(e) : SX_TARGET_DEF; if (!(S->env_target > 0)) S->env_target = SX_TARGET_DEF; }
    S->env_noshiftback = getenv("SXNOSHIFTBACK") != NULL;
    S->env_nostall = getenv("BRISK_SXSTALL") == NULL;
}
static void sx_free(Sx *S) {
    free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Ri); free(S->Rx); free(S->Rnb); free(S->Rc); free(S->Cr); free(S->cscale); free(S->rscale);
    free(S->lo0); free(S->up0); free(S->cost0); free(S->lo); free(S->up); free(S->cost); free(S->shift);
    free(S->basic); free(S->pos); free(S->move); free(S->x); free(S->d); free(S->w); free(S->dvx);
    sxv_free(&S->rho); sxv_free(&S->col); sxv_free(&S->tau); sxv_free(&S->bfrt); sxv_free(&S->spike); sxv_free(&S->upart); sxv_free(&S->row);
    free(S->out); free(S->heap); free(S->hpos); free(S->hkey); free(S->inI); free(S->Ilist); if (S->ws2.stack) sxws_free(&S->ws2); free(S->cand); hp_free(&S->ch); free(S->pc_set); free(S->pc_kv); free(S->pc_at); free(S->pc_mb); free(S->c1); free(S->ca); free(S->cr); free(S->ck); free(S->chm); free(S->cp); free(S->cti); free(S->ctd);
    sxlu_free(&S->F);
}

/* ---------------------------------------------------------------- basis, values, duals */
static int is_boxed(const Sx *S, int j) { return S->lo[j] > -SX_INF && S->up[j] < SX_INF; }

/* a nonbasic variable at the bound its reduced cost prefers (free: at zero) */
static void set_nonbasic(Sx *S, int j, double dj) {
    const double l = S->lo[j], u = S->up[j];
    if (l == u) { S->move[j] = 0; S->x[j] = l; }
    else if (l > -SX_INF && u < SX_INF) { if (dj >= 0) { S->move[j] = 1; S->x[j] = l; } else { S->move[j] = -1; S->x[j] = u; } }
    else if (l > -SX_INF) { S->move[j] = 1; S->x[j] = l; }
    else if (u < SX_INF) { S->move[j] = -1; S->x[j] = u; }
    else { S->move[j] = 0; S->x[j] = 0; }
}
/* after the working bounds changed (restored, perturbed): each nonbasic variable at one of its bounds again, with the
 * status that goes with them (fixed: no move; a bound that is gone: the other one; free: at zero) */
static void normalize_nonbasic(Sx *S) {
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0) continue;
        const double l = S->lo[j], u = S->up[j];
        if (l == u) { S->move[j] = 0; S->x[j] = l; continue; }
        int mv = S->move[j];
        if (mv > 0 && l <= -SX_INF) mv = u < SX_INF ? -1 : 0;
        else if (mv < 0 && u >= SX_INF) mv = l > -SX_INF ? 1 : 0;
        else if (mv == 0 && (l > -SX_INF || u < SX_INF)) mv = l > -SX_INF ? 1 : -1;
        S->move[j] = mv;
        S->x[j] = mv > 0 ? l : mv < 0 ? u : 0;
    }
}
static void slack_basis(Sx *S) {
    for (int i = 0; i < S->m; i++) { S->basic[i] = S->n + i; S->pos[S->n + i] = i; S->move[S->n + i] = 0; }
    for (int j = 0; j < S->n; j++) { S->pos[j] = -1; set_nonbasic(S, j, S->cost[j]); }
    for (int i = 0; i < S->m; i++) S->w[i] = 1.0;
    for (int j = 0; j < S->N; j++) S->dvx[j] = 1.0;
}
static void rows_partition(Sx *S);
/* a given starting basis (statuses of the columns and rows, simplex.h): the basic variables, the nonbasic ones at the
 * bounds their status names; more basic variables than rows: the last ones become nonbasic; fewer: logicals are
 * added (a singular basis is repaired by the factorization) */
static void given_basis(Sx *S, const int *cst, const int *rst) {
    const int n = S->n, m = S->m;
    int nb = 0;
    for (int j = 0; j < S->N; j++) {
        const int st = j < n ? cst[j] : rst[j - n];
        S->pos[j] = -1;
        if (st == SX_BASIC && nb < m) { S->basic[nb] = j; S->pos[j] = nb; S->move[j] = 0; nb++; continue; }
        /* rows: the status names the row's bound; the logical s = -Ax has its bounds reversed */
        int want = st == SX_BASIC ? 0 : st;
        if (j >= n && (want == SX_LOWER || want == SX_UPPER)) want = want == SX_LOWER ? SX_UPPER : SX_LOWER;
        const double l = S->lo[j], u = S->up[j];
        if (l == u) { S->move[j] = 0; S->x[j] = l; }
        else if (want == SX_UPPER && u < SX_INF) { S->move[j] = -1; S->x[j] = u; }
        else if (want == SX_LOWER && l > -SX_INF) { S->move[j] = 1; S->x[j] = l; }
        else if (l > -SX_INF) { S->move[j] = 1; S->x[j] = l; }
        else if (u < SX_INF) { S->move[j] = -1; S->x[j] = u; }
        else { S->move[j] = 0; S->x[j] = 0; }
    }
    for (int i = 0; i < m && nb < m; i++) {
        const int j = n + i;
        if (S->pos[j] >= 0) continue;
        S->basic[nb] = j; S->pos[j] = nb; S->move[j] = 0; nb++;
    }
    for (int i = 0; i < m; i++) S->w[i] = 1.0;
    for (int j = 0; j < S->N; j++) S->dvx[j] = 1.0;
    rows_partition(S);
}
static void col_scatter(const Sx *S, int j, SxVec *v, double mult) {   /* v += mult * a_j (column of [A I]) */
    if (j >= S->n) { const int i = j - S->n; if (v->val[i] == 0) v->idx[v->cnt++] = i; v->val[i] += mult; if (v->val[i] == 0) v->val[i] = 1e-300; return; }
    for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
        const int i = S->Ai[p];
        if (v->val[i] == 0) v->idx[v->cnt++] = i;
        v->val[i] += mult * S->Ax[p];
        if (v->val[i] == 0) v->val[i] = 1e-300;
    }
}
/* (24 October 2026, the night) a solve's counted work (entries and search edges), for the refactorization rule: the mean of
 * the first 8 solves after a factorization, then a running mean */
static inline void sw_note(Sx *S, double w, int nsolves) {
    if (nsolves > 1) w /= nsolves;
    for (int k = 0; k < nsolves; k++) {
        if (S->sw_n < 8) { S->sw_base += w / 8.0; S->sw_n++; S->sw_ew = S->sw_base * 8.0 / S->sw_n; }
        else S->sw_ew = 0.9 * S->sw_ew + 0.1 * w;
    }
}
static void ftran(Sx *S, SxVec *v, double *dens, SxVec *spike) {
    const double t = now();
    const long f0 = S->F.nflops, e0 = S->F.nedges;
    sxlu_ftran(&S->F, v, *dens, spike);
    if (*dens < 1.0) sw_note(S, (double)(S->F.nflops - f0 + S->F.nedges - e0), 1);   /* (not the recomputations of all values, density 1) */
    S->work += (double)(S->F.nflops - f0) + (v->cnt >= 0 ? v->cnt : S->m);
    if (v->cnt >= 0) *dens = 0.95 * *dens + 0.05 * (double)v->cnt / S->m;
    S->t_ftran += now() - t;
}
/* the entering column's solve (with its spike) and the steepest-edge weights' solve. The weights' update reads B^{-1} rho_r
 * only where the entering column is nonzero: with a sparse column it is solved for after the column, at those positions
 * only (sxlu_ftran_part); otherwise in full, at the same time as the column when threads are there and the work pays for
 * them (each solve by one thread: the same bits as one after the other) */
#define SX_TAUPART_DENS 0.1   /* the entering column's density below which the weights' solve may be restricted to it */
#define SX_TAUPROBE 64        /* the iterations between trials of the weights' solve not chosen (ftran2) */
static void ftran2(Sx *S, SxVec *col, SxVec *tau) {
    const double t = now();
    const long f0 = S->F.nflops, e0f = S->F.nedges;
    const double work = (S->dens_col + S->dens_tau) * S->m + 0.02 * S->m;
    if (S->taupart && S->dens_col < SX_TAUPART_DENS) {
        sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
        /* (23 October 2026) the restricted solve or the full one, whichever cost less work lately (entries touched and
         * the searches' edges, counted: the same choice on every run); the other is tried again every SX_TAUPROBE
         * iterations. The restricted solve's two searches cost more than the whole full solve on some problems (sing44,
         * LP: 18.3 s against 12.8 s with the full solve) and far less on others (buildingenergy, LP: 19.7 against 193 s) */
        int part = S->tw_part <= S->tw_full;
        {   /* (the trials of the other solve the rarer the more it cost: every SX_TAUPROBE iterations times the ratio of the
             * two costs, at most 16 times that; buildingenergy's full solve costs ten times the restricted one) */
            const double lo = fmin(S->tw_part, S->tw_full), hi = fmax(S->tw_part, S->tw_full);
            const long gap = lo > 0 ? (long)(SX_TAUPROBE * fmin(hi / lo, 16.0)) : SX_TAUPROBE;
            if (S->iter - S->tw_last >= gap) { part = !part; S->tw_last = S->iter; }
        }
        if (S->taupart == 2) part = 1;
        if (part) S->n_taupart++; else S->n_taufull++;
        if (col->cnt >= 0 && col->cnt < SX_TAUPART_DENS * S->m && part) {
            const long w0 = S->F.nflops + S->F.nedges;
            const int nlr = sxlu_ftran_part(&S->F, &S->ws2, tau, S->dens_taulr, col->idx, col->cnt);
            if (nlr >= 0) S->dens_taulr = 0.95 * S->dens_taulr + 0.05 * (double)nlr / S->m;
            const double w = (double)(S->F.nflops + S->F.nedges - w0);
            S->tw_part = S->tw_part > 0 ? 0.9 * S->tw_part + 0.1 * w : w;
        } else if (col->cnt >= 0 && col->cnt < SX_TAUPART_DENS * S->m) {
            const long w0 = S->F.nflops + S->F.nedges;
            sxlu_ftran(&S->F, tau, S->dens_tau, NULL);
            if (tau->cnt >= 0) S->dens_tau = 0.95 * S->dens_tau + 0.05 * (double)tau->cnt / S->m;
            const double w = (double)(S->F.nflops + S->F.nedges - w0);
            S->tw_full = S->tw_full > 0 ? 0.9 * S->tw_full + 0.1 * w : w;
        } else {
            sxlu_ftran(&S->F, tau, S->dens_tau, NULL);
            if (tau->cnt >= 0) S->dens_tau = 0.95 * S->dens_tau + 0.05 * (double)tau->cnt / S->m;
        }
        if (col->cnt >= 0) S->dens_col = 0.95 * S->dens_col + 0.05 * (double)col->cnt / S->m;
        S->work += (double)(S->F.nflops - f0) + (col->cnt >= 0 ? col->cnt : S->m) + (tau->cnt >= 0 ? tau->cnt : S->m);
        sw_note(S, (double)(S->F.nflops - f0 + S->F.nedges - e0f), 2);
        S->t_ftran += now() - t;
        return;
    }
#ifdef _OPENMP
    if (S->nthreads > 1 && work >= SX_PAR_FTRAN && omp_get_max_threads() > 1) {
        SxWs *W2 = &S->ws2;
        W2->nflops = 0; W2->nedges = 0;
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
            #pragma omp section
            sxlu_ftran_ws(&S->F, W2, tau, S->dens_tau, NULL);
        }
        S->F.nflops += W2->nflops; S->F.nedges += W2->nedges;
        S->npar_ftran++;
    } else
#endif
    {
        sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
        sxlu_ftran(&S->F, tau, S->dens_tau, NULL);
    }
    if (col->cnt >= 0) S->dens_col = 0.95 * S->dens_col + 0.05 * (double)col->cnt / S->m;
    if (tau->cnt >= 0) S->dens_tau = 0.95 * S->dens_tau + 0.05 * (double)tau->cnt / S->m;
    S->work += (double)(S->F.nflops - f0) + (col->cnt >= 0 ? col->cnt : S->m) + (tau->cnt >= 0 ? tau->cnt : S->m);
    sw_note(S, (double)(S->F.nflops - f0 + S->F.nedges - e0f), 2);
    S->t_ftran += now() - t;
}
static void btran(Sx *S, SxVec *v, double *dens, SxVec *upart) {
    const double t = now();
    const long f0 = S->F.nflops, e0 = S->F.nedges;
    sxlu_btran(&S->F, v, *dens, upart);
    if (*dens < 1.0) sw_note(S, (double)(S->F.nflops - f0 + S->F.nedges - e0), 1);
    S->work += (double)(S->F.nflops - f0) + (v->cnt >= 0 ? v->cnt : S->m);
    if (v->cnt >= 0) *dens = 0.95 * *dens + 0.05 * (double)v->cnt / S->m;
    S->t_btran += now() - t;
}
/* x_B = -B^{-1} N x_N */
static void compute_primal(Sx *S) {
    SxVec *v = &S->col; sxv_clear(v);
    for (int j = 0; j < S->N; j++) if (S->pos[j] < 0 && S->x[j] != 0) col_scatter(S, j, v, -S->x[j]);
    double dd = 1; ftran(S, v, &dd, NULL);
    for (int i = 0; i < S->m; i++) S->x[S->basic[i]] = v->val[i];
    sxv_clear(v);
}
/* y = B^{-T} c_B; d_j = c_j - a_j' y (nonbasic), 0 (basic) */
static void compute_dual(Sx *S) {
    SxVec *v = &S->rho; sxv_clear(v);
    for (int i = 0; i < S->m; i++) v->val[i] = S->cost[S->basic[i]];
    sxv_reindex(v);
    double dd = 1; btran(S, v, &dd, NULL);
    const double *y = v->val;
    for (int j = 0; j < S->n; j++) {
        if (S->pos[j] >= 0) { S->d[j] = 0; continue; }
        double s = S->cost[j];
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s -= S->Ax[p] * y[S->Ai[p]];
        S->d[j] = s;
    }
    for (int i = 0; i < S->m; i++) { const int j = S->n + i; S->d[j] = S->pos[j] >= 0 ? 0 : S->cost[j] - y[i]; }
    sxv_clear(v);
}
/* factorize the current basis; replaced columns become nonbasic at a bound */
static int refactor(Sx *S) {
    const double t = now();
    /* the steepest-edge weights are kept by position: carried by variable through the reordering of the factorization;
     * a logical that comes in for a column the factorization could not pivot starts with weight 1 */
    double *wvar = xcalloc(S->N + 1, sizeof(double));
    for (int i = 0; i < S->m; i++) wvar[S->basic[i]] = S->w[i];
    const int nout = sxlu_factor(&S->F, S->m, S->n, S->Ap, S->Ai, S->Ax, S->basic, S->out);
    for (int k = 0; k < nout; k++) {
        const int j = S->out[k]; S->pos[j] = -1;
        if (!S->keepx) { set_nonbasic(S, j, S->d[j]); continue; }
        /* (the crossover: the point stays as it is; a replaced variable not at a bound is pushed again) */
        S->move[j] = S->lo[j] == S->up[j] ? 0 : S->x[j] == S->lo[j] ? 1 : S->x[j] == S->up[j] ? -1 : 0;
    }
    for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; S->pos[j] = i; S->move[j] = 0; S->w[i] = wvar[j] > 0 ? wvar[j] : 1.0; }
    free(wvar);
    if (nout) rows_partition(S);
    S->nfactor++;
    S->sw_n = 0; S->sw_base = 0; S->sw_ew = 0;
    S->work += S->F.factor_work;
    S->t_factor += now() - t;
    if (S->dbg) printf("  factor %d: m %d, L %ld U %ld, %.3f s; densities: column %.4f, row of B^-1 %.4f, weights %.4f, pivotal row %.4f\n", S->nfactor, S->m, (long)S->F.nnz_l, (long)S->F.nnz_u, now() - t, S->dens_col, S->dens_rho, S->dens_tau, S->dens_row);
    return nout;
}

/* ---------------------------------------------------------------- the dual simplex method */
static double infeas(const Sx *S, int i) {
    const int j = S->basic[i];
    const double v = S->x[j];
    if (v < S->lo[j] - S->tolp) return S->lo[j] - v;
    if (v > S->up[j] + S->tolp) return v - S->up[j];
    return 0;
}
/* the heap of the leaving-row candidates */
static int hbetter(const Sx *S, int a, int b) { return S->hkey[a] > S->hkey[b] || (S->hkey[a] == S->hkey[b] && a < b); }
static void hswap(Sx *S, int p, int q) { const int a = S->heap[p], b = S->heap[q]; S->heap[p] = b; S->heap[q] = a; S->hpos[b] = p; S->hpos[a] = q; }
static void hup(Sx *S, int p) { while (p > 0) { const int q = (p - 1) / 2; if (!hbetter(S, S->heap[p], S->heap[q])) break; hswap(S, p, q); p = q; } }
static void hdown(Sx *S, int p) {
    for (;;) {
        const int l = 2 * p + 1, r = l + 1; int b = p;
        if (l < S->nheap && hbetter(S, S->heap[l], S->heap[b])) b = l;
        if (r < S->nheap && hbetter(S, S->heap[r], S->heap[b])) b = r;
        if (b == p) break;
        hswap(S, p, b); p = b;
    }
}
/* the heap from the rows that may be infeasible (all rows when that set is not known: after a recomputation) */
static void heap_rebuild(Sx *S) {
    S->nheap = 0;
    if (S->Ifull) {
        S->nI = 0;
        for (int i = 0; i < S->m; i++) { S->hpos[i] = -1; S->inI[i] = 0; if (infeas(S, i) > 0) { S->inI[i] = 1; S->Ilist[S->nI++] = i; } }
        S->Ifull = 0;
    } else for (int k = 0; k < S->nI; k++) S->hpos[S->Ilist[k]] = -1;
    int c = 0;
    for (int k = 0; k < S->nI; k++) {
        const int i = S->Ilist[k];
        const double f = infeas(S, i);
        if (f == 0) { S->inI[i] = 0; continue; }
        S->Ilist[c++] = i;
        S->hkey[i] = f * f / S->w[i];
        S->hpos[i] = S->nheap; S->heap[S->nheap++] = i;
    }
    S->nI = c;
    for (int p = S->nheap / 2 - 1; p >= 0; p--) hdown(S, p);
    S->heap_ok = 1;
}
static void heap_touch(Sx *S, int i) {
    const double f = infeas(S, i);
    if (f > 0 && !S->inI[i]) { S->inI[i] = 1; S->Ilist[S->nI++] = i; }
    const int p = S->hpos[i];
    if (f == 0) {
        if (p < 0) return;
        const int last = S->nheap - 1;
        if (p != last) { hswap(S, p, last); S->nheap--; S->hpos[i] = -1; hdown(S, p); hup(S, p); }
        else { S->nheap--; S->hpos[i] = -1; }
        return;
    }
    const double k = f * f / S->w[i];
    if (p < 0) { S->hkey[i] = k; S->hpos[i] = S->nheap; S->heap[S->nheap++] = i; hup(S, S->nheap - 1); return; }
    const double old = S->hkey[i];
    S->hkey[i] = k;
    if (k > old) hup(S, p); else if (k < old) hdown(S, p);
}
/* the rows of a vector changed by an iteration: with a dense vector the heap is rebuilt at the next choice */
static void heap_touch_vec(Sx *S, const SxVec *v) {
    if (v->cnt < 0) { S->heap_ok = 0; S->Ifull = 1; return; }
    if (S->heap_ok && v->cnt <= S->m / 8) { for (int k = 0; k < v->cnt; k++) heap_touch(S, v->idx[k]); return; }
    S->heap_ok = 0;
    if (S->Ifull) return;
    for (int k = 0; k < v->cnt; k++) { const int i = v->idx[k]; if (!S->inI[i] && infeas(S, i) > 0) { S->inI[i] = 1; S->Ilist[S->nI++] = i; } }
}
/* the leaving row: largest infeasibility^2 / weight (the lowest index among equals) */
static int chuzr(Sx *S) {
    if (!S->heap_ok) heap_rebuild(S);
    return S->nheap ? S->heap[0] : -1;
}
/* the row-wise copy keeps the entries of the nonbasic columns at the front of each row (HiGHS does the same): the
 * pricing by rows touches only them. A column that enters the basis moves its entries behind that part, one that
 * leaves moves them back. */
static void rows_swap(Sx *S, int p, int q) {
    if (p == q) return;
    const int tj = S->Ri[p], tc = S->Rc[p]; const double tx = S->Rx[p];
    S->Ri[p] = S->Ri[q]; S->Rx[p] = S->Rx[q]; S->Rc[p] = S->Rc[q]; S->Ri[q] = tj; S->Rx[q] = tx; S->Rc[q] = tc;
    S->Cr[S->Rc[p]] = p; S->Cr[S->Rc[q]] = q;
}
static void rows_enter(Sx *S, int j) {
    if (j >= S->n) return;
    for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
        const int i = S->Ai[p], k = S->Cr[p], e = S->Rp[i] + S->Rnb[i];
        if (k < e) { rows_swap(S, k, e - 1); S->Rnb[i]--; }
    }
}
static void rows_leave(Sx *S, int j) {
    if (j >= S->n) return;
    for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
        const int i = S->Ai[p], k = S->Cr[p], e = S->Rp[i] + S->Rnb[i];
        if (k >= e) { rows_swap(S, k, e); S->Rnb[i]++; }
    }
}
static void rows_partition(Sx *S) {
    for (int i = 0; i < S->m; i++) {
        int e = S->Rp[i];
        for (int k = S->Rp[i]; k < S->Rp[i + 1]; k++) if (S->pos[S->Ri[k]] < 0) { rows_swap(S, k, e); e++; }
        S->Rnb[i] = e - S->Rp[i];
    }
}

/* the list of rv's entries above 1e-14 in absolute value among the first N, in index order; the entries at most 1e-14 set
 * to zero. (24 October 2026) by eight entries at once where the processor has AVX-512 (a mask of the entries above the
 * threshold, their indices stored compressed): the pass over all variables was most of the dense pricing (cre-d, LP:
 * 31,000 variables an iteration, 15 of its 145 microseconds); the same list in the same order without it.
 * BRISK_SXVECLIST=0: the scalar pass */
static int dense_list(double *restrict rv, int *restrict ix, int N) {
    int c = 0, j = 0;
#if defined(__AVX512F__) && defined(__AVX512VL__)
    static int vec = -1; if (vec < 0) { const char *e = getenv("BRISK_SXVECLIST"); vec = !(e && atoi(e) == 0); }
    if (vec) {
        const __m512d thr = _mm512_set1_pd(1e-14), zero = _mm512_setzero_pd();
        const __m512i absm = _mm512_set1_epi64(0x7fffffffffffffffLL);
        __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i eight = _mm256_set1_epi32(8);
        for (; j + 8 <= N; j += 8) {
            const __m512d v = _mm512_loadu_pd(rv + j);
            const __m512d a = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v), absm));
            const __mmask8 big = _mm512_cmp_pd_mask(a, thr, _CMP_GT_OQ);
            const __mmask8 nz = _mm512_cmp_pd_mask(v, zero, _CMP_NEQ_UQ);
            if (nz & ~big) _mm512_mask_storeu_pd(rv + j, (__mmask8)(nz & ~big), zero);
            _mm256_mask_compressstoreu_epi32(ix + c, big, idx);
            c += __builtin_popcount((unsigned)big);
            idx = _mm256_add_epi32(idx, eight);
        }
    }
#endif
    int c0 = c;
    for (int q = j; q < N; q++) { ix[c] = q; c += rv[q] != 0; }
    int c2 = c0;
    for (int k = c0; k < c; k++) { const int q = ix[k]; if (fabs(rv[q]) > 1e-14) ix[c2++] = q; else rv[q] = 0; }
    return c2;
}

/* row: alpha_j = rho' a_j for all variables (logicals: rho_i); its list holds the nonzeros among the nonbasic */
#define SX_PRICE_BLK 256      /* columns per block of the parallel pricing */
#define SX_PAR_MIN 50000      /* entries of the matrix from which the pricing by columns uses the threads */
static void price(Sx *S, const SxVec *rho) {
    const double t = now();
    SxVec *row = &S->row;
    sxv_clear(row);
    const int n = S->n;
    double *rv = row->val;
    /* by rows when the nonbasic entries of the rows of rho's nonzeros are fewer than the entries of the matrix
     * (pricing by columns touches every nonbasic column) */
    long rowwork = -1;
    if (rho->cnt >= 0) {
        rowwork = 0;
        const long lim = S->Ap[n] / 2 + n;
        for (int k = 0; k < rho->cnt && rowwork < lim; k++) rowwork += S->Rnb[rho->idx[k]] + 1;
        if (rowwork >= lim) rowwork = -1;
    }
    if (rowwork >= 0 && rowwork > S->N / S->price_dense_div) {
        /* a dense result: accumulated without marks, its list built by one pass in memory order (the later passes over
         * the row then run in memory order too) */
        S->nprice_row++; S->sum_rho += rho->cnt; S->nprice_dense++;
        for (int k = 0; k < rho->cnt; k++) {
            const int i = rho->idx[k];
            const double ri = rho->val[i];
            if (ri == 0) continue;
            const int e = S->Rp[i] + S->Rnb[i];
            for (int p = S->Rp[i]; p < e; p++) rv[S->Ri[p]] += ri * S->Rx[p];
            if (S->pos[n + i] < 0) rv[n + i] = ri;
        }
        /* (24 October 2026, the night) the list without a branch per column (each index written, the count advanced by
         * whether its entry is nonzero), the tiny entries then dropped from the list: the same list in the same order. At a
         * tenth of the columns nonzero the branch was mispredicted often: the pass was three quarters of the pricing on
         * cre-d (LP), 31,000 columns for rows of 3,600 entries */
        row->cnt = dense_list(rv, row->idx, S->N);
    } else if (rowwork >= 0) {
        S->nprice_row++; S->sum_rho += rho->cnt;
        for (int k = 0; k < rho->cnt; k++) {
            const int i = rho->idx[k];
            const double ri = rho->val[i];
            if (ri == 0) continue;
            const int e = S->Rp[i] + S->Rnb[i];
            for (int p = S->Rp[i]; p < e; p++) {
                const int j = S->Ri[p];
                if (!row->mark[j]) { row->mark[j] = 1; row->idx[row->cnt++] = j; }
                rv[j] += ri * S->Rx[p];
            }
            const int jl = n + i;
            if (S->pos[jl] < 0) { row->mark[jl] = 1; row->idx[row->cnt++] = jl; rv[jl] = ri; }
        }
        int c = 0;
        for (int k = 0; k < row->cnt; k++) { const int j = row->idx[k]; row->mark[j] = 0; if (fabs(rv[j]) > 1e-14) row->idx[c++] = j; else rv[j] = 0; }
        row->cnt = c;
    } else {
        S->nprice_col++;
        const double *y = rho->val;
        int c = 0;
        /* each column's product by one thread in a fixed order, so the bits do not depend on the number of threads;
         * blocks of a fixed size handed out as threads become free (busy or slower cores take fewer); the list is
         * then collected in column order */
        const int nblk = (n + SX_PRICE_BLK - 1) / SX_PRICE_BLK;
        #pragma omp parallel for schedule(dynamic, 1) if (S->Ap[n] >= SX_PAR_MIN && S->nthreads > 1)
        for (int b = 0; b < nblk; b++) {
            const int j1 = b * SX_PRICE_BLK, j2 = j1 + SX_PRICE_BLK < n ? j1 + SX_PRICE_BLK : n;
            for (int j = j1; j < j2; j++) {
                if (S->pos[j] >= 0) continue;
                double s = 0;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * y[S->Ai[p]];
                if (fabs(s) > 1e-14) rv[j] = s;
            }
        }
        c = dense_list(rv, row->idx, n);      /* (the entries are 0 or above 1e-14 here) */
        for (int i = 0; i < S->m; i++) {
            const int j = n + i;
            if (S->pos[j] >= 0 || fabs(y[i]) <= 1e-14) continue;
            rv[j] = y[i]; row->idx[c++] = j;
        }
        row->cnt = c;
    }
    S->dens_row = 0.95 * S->dens_row + 0.05 * (double)row->cnt / S->N;
    S->work += rowwork >= 0 ? (double)rowwork + row->cnt : (double)S->Ap[n] + S->m;
    S->t_price += now() - t;
}

/* the bound-flipping ratio test with Harris' tolerances. delta: the primal infeasibility of the leaving row (< 0 below its
 * lower bound). Returns the entering variable (-1: none: the leaving row cannot be made feasible: primal infeasible);
 * the variables to flip are returned in S->cand[0..*nflip-1] */
static int chuzc_dual(Sx *S, double delta, int *nflip) {
    const SxVec *row = &S->row;
    const double mo = delta < 0 ? -1.0 : 1.0;
    const double pivtol = S->cpiv > 0 ? S->cpiv : 1e-7, told = S->told;
    int nc = 0, k0 = 0;
    /* (24 October 2026) eight entries at once where the processor has AVX-512 (the indices, |alpha|, the moves and the
     * reduced costs gathered, the candidates stored compressed; a block holding a fixed or free variable by the scalar
     * loop): the same candidates with the same values in the same order. On wide problems the row has thousands of
     * entries (fast0507, LP: 62,159 columns of 482 rows; the ratio test was 240 of 600 microseconds an iteration).
     * BRISK_SXVECCHUZC=0: the scalar loop */
#if defined(__AVX512F__) && defined(__AVX512VL__)
    static int vecc = -1; if (vecc < 0) { const char *e = getenv("BRISK_SXVECCHUZC"); vecc = !(e && atoi(e) == 0); }
    if (vecc) {
        const __m512d vmo = _mm512_set1_pd(mo), vpt = _mm512_set1_pd(pivtol), zero = _mm512_setzero_pd();
        const __m256i z256 = _mm256_setzero_si256();
        for (; k0 + 8 <= row->cnt; k0 += 8) {
            const __m256i jj = _mm256_loadu_si256((const __m256i *)(row->idx + k0));
            __m256i mw = _mm256_i32gather_epi32((const int *)S->move, jj, 1);
            mw = _mm256_srai_epi32(_mm256_slli_epi32(mw, 24), 24);
            if (_mm256_cmpeq_epi32_mask(mw, z256)) {        /* (a fixed or free variable: this block by the scalar code) */
                for (int k = k0; k < k0 + 8; k++) {
                    const int j = row->idx[k]; const double al = row->val[j]; int mv = S->move[j];
                    if (mv == 0) {
                        if (S->lo[j] == S->up[j]) continue;
                        mv = al * mo > 0 ? 1 : -1;
                        if (fabs(al) < pivtol) continue;
                        S->cand[nc] = j; S->ca[nc] = fabs(al); S->cr[nc] = 0; nc++;
                        continue;
                    }
                    const double a = al * mo * mv;
                    if (a <= pivtol) continue;
                    S->cand[nc] = j; S->ca[nc] = a; S->cr[nc] = mv * S->d[j]; nc++;
                }
                continue;
            }
            const __m512d al = _mm512_i32gather_pd(jj, row->val, 8);
            const __m512d mvd = _mm512_cvtepi32_pd(mw);
            const __m512d a = _mm512_mul_pd(_mm512_mul_pd(al, vmo), mvd);
            const __mmask8 ok = _mm512_cmp_pd_mask(a, vpt, _CMP_GT_OQ);
            if (!ok) continue;
            const __m512d dd = _mm512_mask_i32gather_pd(zero, ok, jj, S->d, 8);
            const __m512d td = _mm512_mul_pd(mvd, dd);
            _mm256_mask_compressstoreu_epi32(S->cand + nc, ok, jj);
            _mm512_mask_compressstoreu_pd(S->ca + nc, ok, a);
            _mm512_mask_compressstoreu_pd(S->cr + nc, ok, td);
            nc += __builtin_popcount((unsigned)ok);
        }
    }
#endif
    for (int k = k0; k < row->cnt; k++) {
        const int j = row->idx[k];
        const double al = row->val[j];
        int mv = S->move[j];
        if (mv == 0) {
            if (S->lo[j] == S->up[j]) continue;              /* fixed: never enters */
            mv = al * mo > 0 ? 1 : -1;                       /* free: either direction */
            if (fabs(al) < pivtol) continue;
            S->cand[nc] = j; S->ca[nc] = fabs(al); S->cr[nc] = 0; nc++;   /* a free variable enters at once */
            continue;
        }
        const double a = al * mo * mv;
        if (a <= pivtol) continue;
        double td = mv * S->d[j];
        S->cand[nc] = j; S->ca[nc] = a; S->cr[nc] = td; nc++;
    }
    /* (22 October 2026) cr holds the reduced cost td, not the ratio td / a: every test below compares by products
     * (td <= s a for a > 0; the least (td + told) / a by cross products), and one division gives each bound. Three
     * divisions per candidate were most of the ratio test's time (lp_setcover_5k, LP: rows of 16,000 entries, 0.14 ms an
     * iteration) */
    *nflip = 0;
    if (nc == 0) return -1;
    double slope = fabs(delta);
    /* preselection (HiGHS): the candidates up to a step where the flips' total change exceeds the infeasibility, the
     * step growing tenfold from ten times the first Harris bound; the groups are then formed within them */
    {
        double hn = SX_INF, hd = 1;      /* the least (td + told) / a, as hn / hd */
        /* (24 October 2026) in four independent lanes (each comparison waited for the last: fast0507, LP, 94 microseconds
         * an iteration), combined by the same comparison, a tie to the earlier candidate: the first least as before */
        {   double ln[4] = { SX_INF, SX_INF, SX_INF, SX_INF }, ld[4] = { 1, 1, 1, 1 }; int lk[4] = { -1, -1, -1, -1 };
            int k = 0;
            for (; k + 4 <= nc; k += 4)
                for (int l = 0; l < 4; l++) { const double nk = S->cr[k + l] + told; if (nk * ld[l] < ln[l] * S->ca[k + l]) { ln[l] = nk; ld[l] = S->ca[k + l]; lk[l] = k + l; } }
            for (; k < nc; k++) { const int l = k & 3; const double nk = S->cr[k] + told; if (nk * ld[l] < ln[l] * S->ca[k]) { ln[l] = nk; ld[l] = S->ca[k]; lk[l] = k; } }
            int bk = -1;
            for (int l = 0; l < 4; l++) {
                if (lk[l] < 0) continue;
                if (bk < 0 || ln[l] * hd < hn * ld[l] || (!(hn * ld[l] < ln[l] * hd) && lk[l] < bk)) { hn = ln[l]; hd = ld[l]; bk = lk[l]; }
            }
        }
        double th0 = hn < SX_INF ? hn / hd : SX_INF;
        if (!(th0 >= 0)) th0 = 0;       /* (a reduced cost of the wrong sign beyond the tolerance: a step of zero; a negative
                                            start grew tenfold away from the candidates and never ended) */
        double sel = 10 * th0 + 1e-7, tot = 0;
        int w = 0;
        for (;;) {
            for (int k = w; k < nc; k++) {
                if (S->cr[k] <= sel * S->ca[k]) {
                    const int j = S->cand[k];
                    const double t1 = S->ca[k], t2 = S->cr[k];
                    S->ca[k] = S->ca[w]; S->cr[k] = S->cr[w]; S->cand[k] = S->cand[w];
                    S->ca[w] = t1; S->cr[w] = t2; S->cand[w] = j; w++;
                    const double rg = S->up[j] - S->lo[j];
                    tot += (rg < SX_INF && S->move[j] != 0) ? t1 * rg : SX_INF;
                }
            }
            if (tot >= slope || w == nc || !(sel < 1e300)) break;
            sel *= 10;
        }
        nc = w;
    }
    /* (23 October 2026) the groups of the bound-flipping ratio test from the candidates sorted by their ratio td / a (ties
     * by the variable's index): with the Harris bounds' suffix minima computed once, each group is the next run of the
     * sorted candidates, so the groups cost one sweep. Before, each group took the Harris bound over all remaining
     * candidates and partitioned them again: the groups times the candidates (neos-933966's re-solves, LP: 0.55 ms of a
     * 1.28 ms iteration in the ratio test). The groups are the same; the entering variable is the largest |alpha| of the
     * final group, the first in the sorted order among equal ones. Not taken (an experiment, BRISK_SXCHUZCSORT=1): the
     * sort costs more than the groups it saves (usually few), and the order among equal |alpha| changes the path:
     * neos-933966's re-solves 1.24 -> 1.65 ms an iteration, 107 -> 136 iterations; Netlib and Kennington at one thread
     * the same (TRICKS 8) */
    static int csort = -1; if (csort < 0) { const char *e = getenv("BRISK_SXCHUZCSORT"); csort = e && atoi(e) == 1; }
    if (csort && nc > 1) {
        double *key = S->ck, *hm = S->chm; int *ord = S->cp;
        for (int k = 0; k < nc; k++) { key[k] = S->cr[k] / S->ca[k]; ord[k] = k; }
        /* sort ord by (key, cand): an insertion sort for few, else a heap sort (no qsort: its comparator would need the
         * arrays through a global) */
        if (nc <= 32) {
            for (int a = 1; a < nc; a++) { const int t = ord[a]; int b = a - 1;
                while (b >= 0 && (key[ord[b]] > key[t] || (key[ord[b]] == key[t] && S->cand[ord[b]] > S->cand[t]))) { ord[b + 1] = ord[b]; b--; }
                ord[b + 1] = t; }
        } else {
            #define SXK_LT(x, y) (key[x] < key[y] || (key[x] == key[y] && S->cand[x] < S->cand[y]))
            for (int r = nc / 2 - 1; r >= 0; r--) { int i = r; for (;;) { int c = 2 * i + 1; if (c >= nc) break; if (c + 1 < nc && SXK_LT(ord[c], ord[c + 1])) c++; if (!SXK_LT(ord[i], ord[c])) break; const int t = ord[i]; ord[i] = ord[c]; ord[c] = t; i = c; } }
            for (int e = nc - 1; e > 0; e--) { int t = ord[0]; ord[0] = ord[e]; ord[e] = t;
                int i = 0; for (;;) { int c = 2 * i + 1; if (c >= e) break; if (c + 1 < e && SXK_LT(ord[c], ord[c + 1])) c++; if (!SXK_LT(ord[i], ord[c])) break; t = ord[i]; ord[i] = ord[c]; ord[c] = t; i = c; } }
            #undef SXK_LT
        }
        /* the candidates in that order */
        { int *ti = S->cti; double *td = S->ctd;
          for (int k = 0; k < nc; k++) ti[k] = S->cand[ord[k]];
          memcpy(S->cand, ti, sizeof(int) * (size_t)nc);
          for (int k = 0; k < nc; k++) td[k] = S->ca[ord[k]];
          memcpy(S->ca, td, sizeof(double) * (size_t)nc);
          for (int k = 0; k < nc; k++) td[k] = S->cr[ord[k]];
          memcpy(S->cr, td, sizeof(double) * (size_t)nc); }
        hm[nc] = SX_INF;
        for (int k = nc - 1; k >= 0; k--) { const double h = (S->cr[k] + told) / S->ca[k]; hm[k] = h < hm[k + 1] ? h : hm[k + 1]; }
        int st = 0;
        for (;;) {
            const double th = hm[st];
            int g = st; double dsl = 0; int inf = 0;
            while (g < nc && S->cr[g] <= th * S->ca[g]) {
                const int j = S->cand[g];
                const double rg = S->up[j] - S->lo[j];
                if (!(rg < SX_INF) || S->move[j] == 0) inf = 1; else dsl += S->ca[g] * rg;
                g++;
            }
            if (g == st) g = st + 1;   /* (rounding at the boundary: the candidate of the bound itself belongs to its group) */
            if (inf || slope - dsl <= 0 || (g == nc && slope - dsl <= S->tolp)) {
                int q = -1; double best = 0;
                for (int k = st; k < g; k++) if (S->ca[k] > best) { best = S->ca[k]; q = S->cand[k]; }
                *nflip = st;
                return q;
            }
            slope -= dsl;
            st = g;
            if (st == nc) { *nflip = 0; return -1; }
        }
    }
    int start = 0;
    for (;;) {
        /* the Harris bound over the remaining candidates, and the group within it */
        double hn = SX_INF, hd = 1;
        for (int k = start; k < nc; k++) { const double nk = S->cr[k] + told; if (nk * hd < hn * S->ca[k]) { hn = nk; hd = S->ca[k]; } }
        const double th = hn < SX_INF ? hn / hd : SX_INF;
        /* move the group (ratio <= th) to the front of the remaining part */
        int g = start; double dsl = 0; int inf = 0;
        for (int k = start; k < nc; k++) {
            if (S->cr[k] <= th * S->ca[k]) {
                const int j = S->cand[k];
                double t1 = S->ca[k], t2 = S->cr[k]; int t3 = S->cand[k];
                S->ca[k] = S->ca[g]; S->cr[k] = S->cr[g]; S->cand[k] = S->cand[g];
                S->ca[g] = t1; S->cr[g] = t2; S->cand[g] = t3; g++;
                const double rg = S->up[j] - S->lo[j];
                if (!(rg < SX_INF) || S->move[j] == 0) inf = 1; else dsl += t1 * rg;
            }
        }
        if (g == start) {   /* numerical corner: nothing within the bound (cannot happen with th finite) */
            return -1;
        }
        /* the final group: the slope turns negative, an unbounded candidate, or the last candidates (when their flips
         * leave the row feasible within the tolerance; otherwise the row cannot be made feasible) */
        if (inf || slope - dsl <= 0 || (g == nc && slope - dsl <= S->tolp)) {
            /* the final group: the largest |alpha| enters */
            int q = -1; double best = 0;
            for (int k = start; k < g; k++) if (S->ca[k] > best) { best = S->ca[k]; q = S->cand[k]; }
            if (S->env_slackpref > 0 && q >= 0 && q < S->n) {
                /* (experiment) among candidates with |alpha| >= env_slackpref of the largest, a logical (or the shortest
                 * column) enters: the basis stays sparser */
                int q2 = -1; long l2 = 1L << 40; double b2 = 0;
                for (int k = start; k < g; k++) {
                    if (S->ca[k] < S->env_slackpref * best) continue;
                    const int j = S->cand[k]; const long len = j >= S->n ? 1 : S->Ap[j + 1] - S->Ap[j];
                    if (len < l2 || (len == l2 && S->ca[k] > b2)) { l2 = len; b2 = S->ca[k]; q2 = j; }
                }
                if (q2 >= 0) q = q2;
            }
            *nflip = start;
            return q;
        }
        slope -= dsl;
        start = g;
        if (start == nc) { *nflip = 0; return -1; }   /* all flipped and still infeasible */
    }
}

/* (experiment) the dual Devex weights (Forrest and Goldfarb 1992): w_i = max(w_i, (alpha_i / alpha_r)^2 w_r) on the
 * entering column's pattern, w_r = max(w_r / alpha_r^2, 1) */
static void update_ddevex(Sx *S, int r, const SxVec *alpha) {
    const double ar = alpha->val[r], wr = S->w[r];
    if (alpha->cnt >= 0) {
        for (int k = 0; k < alpha->cnt; k++) {
            const int i = alpha->idx[k];
            if (i == r) continue;
            const double ratio = alpha->val[i] / ar, wi = ratio * ratio * wr;
            if (wi > S->w[i]) S->w[i] = wi;
        }
    } else {
        for (int i = 0; i < S->m; i++) {
            if (i == r || alpha->val[i] == 0) continue;
            const double ratio = alpha->val[i] / ar, wi = ratio * ratio * wr;
            if (wi > S->w[i]) S->w[i] = wi;
        }
    }
    const double nw = wr / (ar * ar);
    S->w[r] = nw > 1.0 ? nw : 1.0;
}
/* update the dual steepest-edge weights; alpha: B^{-1} a_q (by position), tau = B^{-1} rho_r, wr = |rho_r|^2 */
static void update_dse(Sx *S, int r, const SxVec *alpha, const SxVec *tau, double wr) {
    const double ar = alpha->val[r];
    const double nw = wr / (ar * ar);
    if (alpha->cnt >= 0) {
        for (int k = 0; k < alpha->cnt; k++) {
            const int i = alpha->idx[k];
            if (i == r) continue;
            const double ai = alpha->val[i];
            const double ratio = ai / ar;
            double wi = S->w[i] + ratio * (ratio * wr - 2 * tau->val[i]);
            S->w[i] = wi > 1e-4 ? wi : 1e-4;
        }
    } else {
        for (int i = 0; i < S->m; i++) {
            if (i == r || alpha->val[i] == 0) continue;
            const double ratio = alpha->val[i] / ar;
            double wi = S->w[i] + ratio * (ratio * wr - 2 * tau->val[i]);
            S->w[i] = wi > 1e-4 ? wi : 1e-4;
        }
    }
    S->w[r] = nw > 1e-4 ? nw : 1e-4;
}

/* the dual method's starting bounds: with the basis of the logicals, a boxed column of cost zero is dual feasible at
 * either bound, so it is put at the one that leaves the smaller primal infeasibility (the sum over its rows of the
 * logicals' bound violations), one pass over the columns; the perturbation then keeps that bound. Returns the flips. */
static long dual_start_bounds(Sx *S) {
    const int n = S->n, m = S->m;
    double *v = xcalloc((size_t)m, sizeof(double));      /* the logicals' values s = -A x */
    for (int j = 0; j < n; j++) {
        const double xj = S->x[j];
        if (xj != 0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v[S->Ai[p]] -= S->Ax[p] * xj;
    }
#define SX_VIOL(i, s) ((s) < S->lo[n + (i)] ? S->lo[n + (i)] - (s) : (s) > S->up[n + (i)] ? (s) - S->up[n + (i)] : 0.0)
    long nfl = 0;
    for (int j = 0; j < n; j++) {
        if (S->pos[j] >= 0 || S->cost[j] != 0 || !is_boxed(S, j) || S->lo[j] == S->up[j]) continue;
        const double to = S->move[j] > 0 ? S->up[j] : S->lo[j], dx = to - S->x[j];
        double g = 0, base = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p];
            const double e0 = SX_VIOL(i, v[i]), e1 = SX_VIOL(i, v[i] - S->Ax[p] * dx);
            g += e1 - e0; base += e0;
        }
        if (!(g < -1e-9 * (1 + base))) continue;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v[S->Ai[p]] -= S->Ax[p] * dx;
        S->x[j] = to; S->move[j] = (signed char)-S->move[j]; nfl++;
    }
#undef SX_VIOL
    free(v);
    return nfl;
}
/* cost perturbation (against dual degeneracy): structurals by a relative random amount in the direction that keeps the
 * starting basis' reduced costs feasible */
/* (the stall rule on trial: BRISK_SXSTALL=1; off until measured on the sets; read by sx_setup into env_nostall) */
static void perturb_costs(Sx *S) {
    const int n = S->n;
    double bigc = 0;
    for (int j = 0; j < n; j++) if (fabs(S->cost0[j]) > bigc) bigc = fabs(S->cost0[j]);
    if (bigc > 100) bigc = sqrt(sqrt(bigc));
    int nbox = 0; for (int j = 0; j < S->N; j++) nbox += is_boxed(S, j);
    if ((double)nbox / S->N < 0.01 && bigc > 1) bigc = 1;
    double base = 5e-7 * bigc, one = 1;
    if (getenv("SXPERT_BASE")) base *= atof(getenv("SXPERT_BASE"));
    const int mode = getenv("SXPERT_MODE") ? atoi(getenv("SXPERT_MODE")) : 1;
    if (mode == 1) {   /* the rule in the units of the costs as given (HiGHS): bigc and the floor 1 unscaled */
        double bc = bigc * S->costscale; if (bc > 100) bc = sqrt(sqrt(bc));
        if ((double)nbox / S->N < 0.01 && bc > 1) bc = 1;
        base = 5e-7 * bc / S->costscale; one = 1.0;
        if (getenv("SXPERT_BASE")) base *= atof(getenv("SXPERT_BASE"));
        one = 1.0;   /* xp = (|c_j| S + 1) base_u / S in scaled units = (|c_j| + 1/S) base_u */
    }
    for (int j = 0; j < n; j++) {
        const double l = S->lo0[j], u = S->up0[j];
        const double xp = mode == 1 ? (fabs(S->cost0[j]) * S->costscale + one) * base * (1 + urand(S))
                                    : (fabs(S->cost0[j]) + one) * base * (1 + urand(S));
        if (l <= -SX_INF && u >= SX_INF) continue;
        if (l == u) continue;
        /* (23 October 2026) the nonbasic columns only: from the slack basis that is every column; from a given basis
         * (the rounds of the row sifting) a basic column's perturbation moves the duals and leaves nonbasic reduced
         * costs infeasible by its size, and the round began with a dual phase 1 (comp21-2idx: 1,705 iterations) */
        if (S->pos[j] >= 0) continue;
        if (u >= SX_INF) S->cost[j] += xp;
        else if (l <= -SX_INF) S->cost[j] -= xp;
        else if (S->cost[j] == 0 && S->pos[j] < 0) S->cost[j] += S->move[j] < 0 ? -xp : xp;   /* (the bound it is at) */
        else S->cost[j] += S->cost[j] >= 0 ? xp : -xp;
    }
    for (int i = 0; i < S->m; i++) S->cost[n + i] += (0.5 - urand(S)) * 1e-12;
    S->perturbed = 1; S->pert_base = base; S->pert_level = 0; S->zw_it = S->zw_zero = 0;
}
/* the dual method stalls (half of the steps of a window are of zero length: the perturbation is too small for the
 * problem's dual degeneracy): the nonbasic structurals' costs perturbed again, ten times more each time (at most three
 * times), each in the direction that keeps its reduced cost feasible, so the basis stays dual feasible (Maros 2003,
 * Koberstein 2005: perturb on stalling). savsched1: 252,731 columns of cost zero, the objective at -20,807.47 for 25,000
 * iterations. The perturbation is removed at the end as the first one (the primal method). */
#define SX_STALL_WIN 1000
static void perturb_more(Sx *S) {
    S->pert_level++; S->nrepert++;
    double sc = 1; for (int k = 0; k < S->pert_level; k++) sc *= 10;
    for (int j = 0; j < S->n; j++) {
        if (S->pos[j] >= 0 || S->lo0[j] == S->up0[j] || S->move[j] == 0) continue;
        const double xp = (fabs(S->cost0[j]) * S->costscale + 1) * S->pert_base * sc * (1 + urand(S));
        if (S->move[j] > 0) { S->cost[j] += xp; S->d[j] += xp; } else { S->cost[j] -= xp; S->d[j] -= xp; }
    }
}

/* (23 October 2026) the first perturbation during the loop, for the re-solves (sxm_solve), which start unperturbed: the
 * base of perturb_costs, applied as perturb_more applies its escalations (the nonbasic structurals at a bound, in the
 * direction that keeps their reduced costs feasible). Without it a degenerate re-solve had no remedy for a stall: the
 * stall rule applied to perturbed costs only (the MIP developer's csched008: 365,695 iterations in one node's re-solve). */
static void perturb_first(Sx *S) {
    double bigc = 0;
    for (int j = 0; j < S->n; j++) if (fabs(S->cost0[j]) > bigc) bigc = fabs(S->cost0[j]);
    int nbox = 0; for (int j = 0; j < S->N; j++) nbox += is_boxed(S, j);
    double bc = bigc * S->costscale; if (bc > 100) bc = sqrt(sqrt(bc));
    if ((double)nbox / S->N < 0.01 && bc > 1) bc = 1;
    S->pert_base = 5e-7 * bc / S->costscale; S->pert_level = -1;
    perturb_more(S);
    S->perturbed = 1; S->zw_it = S->zw_zero = 0;
}

/* a cost shift that makes the reduced cost of nonbasic j zero; recorded so that it is removed when j leaves the basis */
static void shift_cost(Sx *S, int j) {
    const double dj = S->d[j];
    S->cost[j] -= dj; S->shift[j] -= dj; S->d[j] = 0;
    S->nshift++; S->shifted = 1;
    if (fabs(dj) > S->shmax) S->shmax = fabs(dj);
}
/* remove the shift of a variable that leaves the basis (its reduced cost takes the change) */
static void shift_back(Sx *S, int j) {
    if (S->shift[j] == 0 || S->env_noshiftback) return;
    S->cost[j] -= S->shift[j]; S->d[j] -= S->shift[j]; S->shift[j] = 0;
    S->nshiftback++;
}
/* after a recomputation of the duals: nonbasic variables whose reduced cost has the wrong sign beyond the tolerance are
 * flipped to their other bound (boxed) or get their cost shifted (otherwise); returns 1 if any was flipped (the primal
 * values must then be recomputed) */
static int correct_duals(Sx *S, int *nshift) {
    int flipped = 0;
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0) continue;
        const double dj = S->d[j];
        if (S->move[j] == 0) {
            if (S->lo[j] == S->up[j] || fabs(dj) <= S->told) continue;
            /* free nonbasic with a nonzero reduced cost: shift */
            shift_cost(S, j); (*nshift)++; S->nsh_free++; continue;
        }
        if (S->move[j] * dj >= -S->told) continue;
        if (is_boxed(S, j)) { S->move[j] = -S->move[j]; S->x[j] = S->move[j] > 0 ? S->lo[j] : S->up[j]; flipped = 1; S->nflip_reb++; }
        else { shift_cost(S, j); (*nshift)++; S->nsh_reb++; }
    }
    return flipped;
}

/* refactorize when the average work of an iteration since the last factorization, the factorization included, is
 * least: the work of an iteration grows by about three solves' worth of the entries each update adds (R etas and the
 * spikes in U), so with factorization work W and growth a per update the average W/k + c + a k/2 is least at
 * k = sqrt(2 W / a). Counted in entries, not seconds: the same decisions on every run. An entry touched by the
 * factorization costs about ten touched by a solve (scattered writes, the setup of its structures): the factor 10 was
 * measured on Netlib and Kennington (1, 3, 10, 30, 100: shifted geometric means 0.098, 0.095, 0.092, 0.095, 0.101 s). */
static int need_refactor(const Sx *S) {
    const SxLU *F = &S->F;
    if (F->nupd >= S->updlim) return 1;
    if (F->nupd < 20) return 0;
    double grow = (double)(F->nnz_r + (F->nnz_u - F->nnz_u0)) / F->nupd;
    /* (24 October 2026, the night) the growth of the solves' work measured (the counted work of the solves now against the
     * first ones after the factorization), not the entries the updates added: the hypersparse solves skip most of those
     * (lp_beckmann_300's crossover, LP: a factorization every 146 updates, 4.8 s of factorizations). BRISK_SXREFACMEAS=0:
     * the entries as before */
    if (S->refac_meas && S->sw_n >= 8) grow = (S->sw_ew - S->sw_base) / F->nupd;
    const double a = 3.0 * (grow > 1 ? grow : 1);
    return F->nupd >= sqrt(2.0 * S->env_refac_w * F->factor_work / a);
}
static int check_stop(Sx *S) {
    if (S->stop && *S->stop) return 6;
#ifdef _OPENMP
    if (S->threads) { const int t = *S->threads; if (t > 0 && t != S->cur_threads) { omp_set_num_threads(t); S->cur_threads = t; } }
#endif
    if (S->maxit > 0 && S->iter >= S->maxit) return 3;
    if (S->tlim > 0 && (++S->nchk & 15) == 0 && now() - S->t0 > S->tlim) return 6;
    if (S->workbudget > 0 && S->work > S->workbudget) {
        /* the budget's owner may raise it (the automatic choice of the method: a second budget once the first is used) */
        if (S->budget_cb) { const double nb = S->budget_cb(S->budget_ctx, S->work); if (nb > S->work) { S->workbudget = nb; return 0; } }
        return 6;
    }
    return 0;
}

static double objective(const Sx *S) { double s = 0; for (int j = 0; j < S->N; j++) s += S->cost[j] * S->x[j]; return s; }
/* (22 October 2026) the dual bound of the current duals y for the problem's own costs and bounds: for any y,
 * min { cost0'v : [A I] v = 0, lo0 <= v <= up0 } >= sum_j min { dbar_j v_j : lo0_j <= v_j <= up0_j }, dbar = cost0 - [A I]'y
 * = d - (cost - cost0) (the perturbation and the shifts taken out). A reduced cost within the tolerance towards an
 * infinite bound counts as zero; beyond it the bound is -infinity. With a dual feasible basis and the costs as they are,
 * this is the objective of the basic solution; it stays a valid bound while costs are perturbed or shifted */
static double dual_bound(const Sx *S) {
    double s = 0;
    for (int j = 0; j < S->N; j++) {
        const double dj = S->d[j] - (S->cost[j] - S->cost0[j]);
        if (dj == 0) continue;
        const double b = dj > 0 ? S->lo0[j] : S->up0[j];
        if (fabs(b) >= SX_INF) { if (fabs(dj) > S->told) return -SX_INF; continue; }
        s += dj * b;
    }
    return s;
}
/* the same bound for the working costs (perturbed and shifted) and the working bounds' kind: the dual objective the
 * method itself sees */
static double dual_bound_w(const Sx *S) {
    double s = 0;
    for (int j = 0; j < S->N; j++) {
        const double dj = S->d[j];
        if (dj == 0) continue;
        const double b = dj > 0 ? S->lo0[j] : S->up0[j];
        if (fabs(b) >= SX_INF) { if (fabs(dj) > S->told) return -SX_INF; continue; }
        s += dj * b;
    }
    return s;
}
static double sum_pinf(const Sx *S, int *cnt) {
    double s = 0; int c = 0;
    for (int i = 0; i < S->m; i++) { const double f = infeas(S, i); if (f > 0) { s += f; c++; } }
    if (cnt) *cnt = c;
    return s;
}
static void log_line(Sx *S, const char *what) {
    if (S->verbose < 2) return;
    int c; const double s = sum_pinf(S, &c);
    printf("  %-8s %8ld  obj %+.10e  pinf %.2e (%d)  %.2fs\n", what, S->iter, objective(S) * S->costscale, s, c, now() - S->t0);
}

/* rebuild: factorization, primal and dual values; corrections of the duals */
static int rebuild(Sx *S) {
    refactor(S);
    compute_dual(S);
    int ns = 0;
    correct_duals(S, &ns);
    S->nshift += ns;
    compute_primal(S);
    return 0;
}

/* dual simplex iterations until no row is infeasible (returns 0), the problem is primal infeasible (1), or a limit (3, 6) */
static int dual_loop(Sx *S) {
    const int m = S->m;
    SxVec *rho = &S->rho, *col = &S->col, *tau = &S->tau, *bf = &S->bfrt;
    int need_rebuild = 1, verified = 0;
    long last_log = -1000;
    S->heap_ok = 0; S->Ifull = 1;
    for (;;) {
        int st = check_stop(S);
        if (st) return st;
        if (S->fresh) { S->fresh = 0; need_rebuild = 0; }
        if (need_rebuild || need_refactor(S)) { rebuild(S); need_rebuild = 0; S->heap_ok = 0; S->Ifull = 1; }
        if (S->pert_pending) { if (S->perturbed) perturb_more(S); else perturb_first(S); S->pert_pending = 0; }
        if (S->iter - last_log >= 1000) { log_line(S, S->phase == 1 ? "dual ph1" : "dual"); last_log = S->iter; }
        /* (22 October 2026) the objective cutoff (sxm_solve): the dual bound of the current duals with the problem's own
         * costs and bounds passes the cutoff: the optimum lies beyond it, and the node can be pruned */
        if (S->cut_on && S->phase == 2 && (S->N <= 20000 || S->iter % 10 == 0)) {
            const double b = dual_bound(S);
            const double cv = S->cut_val + 1e-9 * (1 + fabs(S->cut_val));
            if (b > cv) { S->cut_bound = b; return 7; }
            /* (23 October 2026) with perturbed costs the bound above can be minus infinity (a column without the bound
             * its true reduced cost asks for); when the working costs' own bound passes the cutoff, the perturbation is
             * removed and the duals recomputed (as HiGHS does): the costs as they are then decide, and the loop goes on
             * unperturbed when they do not prune */
            if (S->perturbed && dual_bound_w(S) > cv) {
                for (int j = 0; j < S->N; j++) S->cost[j] = S->cost0[j] + S->shift[j];
                S->perturbed = 0; S->zw_it = S->zw_zero = 0;
                compute_dual(S);
                int ns = 0; if (correct_duals(S, &ns)) compute_primal(S);
                S->nshift += ns; S->heap_ok = 0; S->Ifull = 1;
                const double b2 = dual_bound(S);
                if (b2 > cv) { S->cut_bound = b2; return 7; }
            }
        }
        const int r = chuzr(S);
        if (r < 0) {
            if (S->F.nupd > 0 && !verified && !S->noverify) { need_rebuild = 1; verified = 1; continue; }   /* confirm on fresh values */
            return 0;
        }
        verified = 0;
        const int jl = S->basic[r];
        const double xr = S->x[jl];
        const double delta = xr < S->lo[jl] ? xr - S->lo[jl] : xr - S->up[jl];
        /* BTRAN e_r */
        sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
        btran(S, rho, &S->dens_rho, &S->upart);
        if (rho->cnt < 0) sxv_reindex(rho);
        double wr = 0; for (int k = 0; k < rho->cnt; k++) wr += rho->val[rho->idx[k]] * rho->val[rho->idx[k]];
        price(S, rho);
        int nflip;
        int q = chuzc_dual(S, delta, &nflip);
        if (q < 0) {
            if (S->F.nupd > 0) { need_rebuild = 1; S->w[r] = wr; S->heap_ok = 0; continue; }
            /* a tiny infeasibility that no column can remove: the bound is moved to the value (restored at the end) */
            const double bd = delta < 0 ? S->lo[jl] : S->up[jl];
            if (fabs(delta) <= 1e-6 * (1 + fabs(bd))) {
                if (delta < 0) S->lo[jl] = xr; else S->up[jl] = xr;
                S->nbshift++; S->heap_ok = 0; continue;
            }
            /* (25 October 2026, the MIP developer's request) the claim checked as the proof it stands for: row r of B^-1
             * [A I] gives x_jl = -sum alpha_j x_j over the nonbasic variables, and the claim holds when their boxes cannot
             * move x_jl by |delta|. The ratio test passes over entries |alpha_j| up to its pivot tolerance; with wide
             * boxes they can together remove the infeasibility (bc1's strong branching: entries about 1e-8 of columns
             * bounded at 7e6, a feasible LP claimed infeasible, the ray failing the MIP solver's check by 0.5). Then
             * the ratio test once more with those entries admitted down to 1e-9 (the bound-flipping test spreads the
             * step over their boxes); when no column enters either, the claim is not made: status 4. Entries of at
             * most 1e-9 count only with a finite box (with an infinite one they are rounding, as in the MIP solver's
             * check of the certificate). HiGHS checks its proofs of infeasibility the same way. (Fixed variables enter
             * at their value; the basic variables' entries are zero up to rounding.) */
            {
                /* the value of x_jl nearest its violated bound over the boxes: sum -alpha_j b_j with b_j the bound that
                 * moves x_jl toward it (from the row, not from x_jl, whose value can be off by the basis' rounding) */
                const double mo = delta < 0 ? -1.0 : 1.0;
                long double best = 0; int open = 0;
                for (int k = 0; k < S->row.cnt; k++) {
                    const int j = S->row.idx[k]; const double al = S->row.val[j];
                    if (al == 0) continue;
                    const double b = S->lo[j] == S->up[j] ? S->lo[j] : al * mo > 0 ? S->up[j] : S->lo[j];
                    if (fabs(b) >= SX_INF / 2) { if (fabs(al) > 1e-9) open++; continue; }
                    best -= (long double)al * b;
                }
                const int proved = !open && (delta < 0 ? best < S->lo[jl] - S->tolp : best > S->up[jl] + S->tolp);
                if (!proved) {
                    S->cpiv = 1e-9; q = chuzc_dual(S, delta, &nflip); S->cpiv = 0;
                    if (q < 0) { S->nfkfail++; return 4; }
                    S->nfkretry++;
                }
            }
        }
        if (q < 0) {
            S->rayrow = r; S->raymo = delta < 0 ? -1 : 1;
            if (S->rayout) { for (int i = 0; i < S->m; i++) S->rayout[i] = 0; for (int k = 0; k < rho->cnt; k++) { const int i = rho->idx[k]; S->rayout[i] = rho->val[i] * S->rscale[i]; } S->have_ray = 1; }
            if (getenv("SXDBG")) {
                printf("infeasible row %d (var %d, x %g lo %g up %g delta %g), row nnz %d\n", r, jl, S->x[jl], S->lo[jl], S->up[jl], delta, S->row.cnt);
                for (int k = 0; k < S->row.cnt && k < 40; k++) { const int j = S->row.idx[k]; printf("   j %d alpha %g move %d lo %g up %g x %g d %g\n", j, S->row.val[j], S->move[j], S->lo[j], S->up[j], S->x[j], S->d[j]); }
            }
            return 1;
        }
        const double arow = S->row.val[q];
        S->sum_rowc += S->row.cnt;
        /* FTRAN the entering column, and rho for the weights */
        sxv_clear(col); col_scatter(S, q, col, 1.0);
        if (S->env_ddevex) {
            const double t = now(); const long f0 = S->F.nflops;
            sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
            if (col->cnt >= 0) S->dens_col = 0.95 * S->dens_col + 0.05 * (double)col->cnt / S->m;
            S->work += (double)(S->F.nflops - f0) + (col->cnt >= 0 ? col->cnt : S->m);
            S->t_ftran += now() - t;
        } else { sxv_copy(tau, rho); ftran2(S, col, tau); }
        const double acol = col->val[r];
        S->sum_col += col->cnt >= 0 ? col->cnt : m;
        if (fabs(acol - arow) > 1e-7 * (1 + fabs(acol)) || fabs(acol) < 1e-9) {
            /* the row and the column disagree: refactorize and try again */
            if (S->F.nupd > 0) { need_rebuild = 1; continue; }
            if (fabs(acol) < 1e-9) { S->w[r] *= 10; S->heap_ok = 0; continue; }
        }
        /* the dual step: d_q / alpha_q when the entering reduced cost has the sign of its move; a step of zero when it
         * is zero or of the wrong sign within the tolerance (its cost shifted to make it zero; no flips then) */
        const int mvq = S->move[q] ? S->move[q] : (arow * (delta < 0 ? -1.0 : 1.0) > 0 ? 1 : -1);
        double thd = 0;
        if ((S->perturbed || S->lazy_pert) && S->phase == 2 && !S->env_nostall) {
            /* (24 October 2026, the night) a step is degenerate when the entering reduced cost is within twice the
             * perturbation that column received (it moves only by the perturbation), not only when it is zero: mzzv11 (LP)
             * moved its objective by 5e-4 in 6,000 iterations of such steps, none of zero length, and the rule never fired */
            double zq = S->told;
            if (S->perturbed && q < S->n) { double sc = 1; for (int k = 0; k < S->pert_level; k++) sc *= 10; const double pq = 2.0 * (fabs(S->cost0[q]) * S->costscale + 1) * S->pert_base * sc; if (pq > zq) zq = pq; }
            S->zw_it++; if (!(mvq * S->d[q] > zq)) S->zw_zero++;
            if (S->zw_it >= SX_STALL_WIN) {
                /* (experiment) a window in which the objective moved by less than 1e-7 of its size: a stall as well,
                 * whether the steps were zero or only tiny (savsched1: -20,657.078 over 20,000 iterations) */
                int objstall = 0;
                if (S->env_stallobj) {
                    const double ob = objective(S);
                    objstall = isfinite(S->stall_obj) && fabs(ob - S->stall_obj) <= 1e-7 * (1 + fabs(ob));
                    S->stall_obj = ob;
                }
                static int smax = -1; if (smax < 0) { const char *e = getenv("SXSTALLMAX"); smax = e ? atoi(e) : 3; }   /* (experiment) the escalations allowed */
                if ((2 * S->zw_zero >= S->zw_it || objstall) && (!S->perturbed || S->pert_level < smax)) {
                    S->pert_pending = 1;     /* (at the next iteration's start: this one's ratio test used the costs as they are) */
                    if (S->verbose >= 2) printf("  %ld of the last %ld steps of zero length: the costs perturbed %s (x%g)\n", S->zw_zero, S->zw_it, S->perturbed ? "again" : "", !S->perturbed ? 1.0 : S->pert_level == 0 ? 10.0 : S->pert_level == 1 ? 100.0 : 1000.0);
                }
                S->zw_it = S->zw_zero = 0;
            }
        }
        if (mvq * S->d[q] > 0) thd = S->d[q] / acol;
        else { shift_cost(S, q); nflip = 0; }
        /* flips: the candidates the ratio test passed, and the boxed variables of the row whose reduced cost has the
         * wrong sign beyond the tolerance after the step (entries below the pivot tolerance, which the ratio test does
         * not consider); the others with such a reduced cost get their cost shifted. One solve for all the flips. */
        int nfl = 0;
        sxv_clear(bf);
        for (int k = 0; k < nflip; k++) {
            const int j = S->cand[k];
            if (S->move[j] > 0) { col_scatter(S, j, bf, S->up[j] - S->lo[j]); S->x[j] = S->up[j]; S->move[j] = -1; }
            else { col_scatter(S, j, bf, S->lo[j] - S->up[j]); S->x[j] = S->lo[j]; S->move[j] = 1; }
            nfl++;
        }
        /* the dual update, and in the same pass the entries whose reduced cost it leaves of the wrong sign */
        int nbad = 0;
        if (thd != 0) {
            const double mtd = -S->told;
            for (int k = 0; k < S->row.cnt; k++) {
                const int j = S->row.idx[k];
                const double dj = S->d[j] - thd * S->row.val[j];
                S->d[j] = dj;
                if (S->move[j] * dj < mtd) S->cand[nbad++] = j;     /* (cand's flips were taken above) */
            }
        }
        S->d[q] = 0;
        for (int t = 0; t < nbad; t++) {
            const int j = S->cand[t];
            if (j == q || S->move[j] == 0 || S->move[j] * S->d[j] >= -S->told) continue;
            if (is_boxed(S, j)) {
                if (S->move[j] > 0) { col_scatter(S, j, bf, S->up[j] - S->lo[j]); S->x[j] = S->up[j]; S->move[j] = -1; }
                else { col_scatter(S, j, bf, S->lo[j] - S->up[j]); S->x[j] = S->lo[j]; S->move[j] = 1; }
                nfl++; S->nflip_it++;
            } else { shift_cost(S, j); S->nsh_it++; }
        }
        if (nfl) {
            S->nflips += nfl;
            double dd = S->dens_col; ftran(S, bf, &dd, NULL);
            if (bf->cnt >= 0) { for (int k = 0; k < bf->cnt; k++) { const int i = bf->idx[k]; S->x[S->basic[i]] -= bf->val[i]; } }
            else for (int i = 0; i < m; i++) S->x[S->basic[i]] -= bf->val[i];
        }
        /* the primal step */
        const double xr2 = S->x[jl];
        const double bound = delta < 0 ? S->lo[jl] : S->up[jl];
        const double thp = (xr2 - bound) / acol;
        if (col->cnt >= 0) { for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= thp * col->val[i]; } }
        else for (int i = 0; i < m; i++) if (col->val[i] != 0) S->x[S->basic[i]] -= thp * col->val[i];
        S->x[q] += thp;
        S->x[jl] = bound;
        /* weights */
        if (S->env_ddevex) update_ddevex(S, r, col); else update_dse(S, r, col, tau, wr);
        /* the basis change */
        S->basic[r] = q; S->pos[q] = r; S->move[q] = 0;
        S->pos[jl] = -1;
        rows_enter(S, q); rows_leave(S, jl);
        S->d[jl] = -thd;
        if (S->lo[jl] == S->up[jl]) S->move[jl] = 0; else S->move[jl] = delta < 0 ? 1 : -1;
        shift_back(S, jl);
        const int bad = sxlu_update(&S->F, r, &S->spike, &S->upart, acol);
        if (bad) need_rebuild = 1;
        /* the leaving-row candidates of the rows that changed */
        heap_touch_vec(S, col);
        if (nfl) heap_touch_vec(S, bf);
        if (S->heap_ok) heap_touch(S, r);
        else if (!S->Ifull && !S->inI[r] && infeas(S, r) > 0) { S->inI[r] = 1; S->Ilist[S->nI++] = r; }
        S->iter++; S->it_dual++; if (S->phase == 1) S->it_ph1++;
    }
}

/* ---------------------------------------------------------------- the primal simplex method */
/* the entering variable by Devex pricing: largest d_j^2 / w_j among the dual infeasible */
/* the merit of nonbasic j as an entering candidate (0: none) */
static double pmerit(const Sx *S, int j) {
    if (S->pos[j] >= 0) return 0;
    const double dj = S->d[j];
    double inf;
    if (S->move[j] == 0) { if (S->lo[j] == S->up[j]) return 0; inf = fabs(dj); }
    else inf = -S->move[j] * dj;
    if (inf <= S->told) return 0;
    const double mr = inf * inf / S->dvx[j];
    return mr > 1e-300 ? mr : 1e-300;
}
/* (23 October 2026) the candidates as a short list instead of a heap of all of them (Hall and McKinnon's hypersparse
 * pricing): the list holds the best candidates with their merits, and pc_tau bounds the merit of every candidate outside
 * it. Only the pivotal row's entries (and the entering and leaving variables) change merit in an iteration, so the
 * bound stays valid; while the list's best is at least pc_tau it is the best of all, and when it falls below, all are
 * scanned again. The heap updated its key for every entry of the pivotal row (neos-631710, LP: rows of 21,000 entries,
 * 26,000 heap updates an iteration, more than half of the primal method's time). The same choice as the heap's but among
 * equal merits; BRISK_SXPHEAP=1: the heap. */
static int cmp_pckey(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? 1 : x > y ? -1 : 0; }
static void pc_drop(Sx *S, int p) {
    const int j = S->pc_set[p], last = --S->pc_n;
    S->pc_at[j] = -1;
    if (p != last) { S->pc_set[p] = S->pc_set[last]; S->pc_kv[p] = S->pc_kv[last]; S->pc_at[S->pc_set[p]] = p; }
}
static void pc_put(Sx *S, int j, double k) { const int p = S->pc_n++; S->pc_set[p] = j; S->pc_kv[p] = k; S->pc_at[j] = p; }
/* the list full: the best SX_PC_K kept, the bound raised to the best of those dropped */
static void pc_shrink(Sx *S) {
    double *t = xmalloc(sizeof(double) * (S->pc_n + 1));
    memcpy(t, S->pc_kv, sizeof(double) * S->pc_n);
    qsort(t, S->pc_n, sizeof(double), cmp_pckey);
    const double cut = t[SX_PC_K];      /* (the best of those beyond the first SX_PC_K) */
    free(t);
    if (cut > S->pc_tau) S->pc_tau = cut;
    for (int p = S->pc_n - 1; p >= 0; p--) if (S->pc_kv[p] <= cut) pc_drop(S, p);
}
static void pc_set_key(Sx *S, int j, double k) {
    if (S->pc_heap) { hp_set(&S->ch, j, k); return; }
    const int p = S->pc_at[j];
    if (!(k > 0)) { if (p >= 0) pc_drop(S, p); return; }
    if (p >= 0) { S->pc_kv[p] = k; return; }
    if (k <= S->pc_tau) return;
    if (S->pc_n >= S->pc_cap) { pc_shrink(S); if (k <= S->pc_tau) return; }
    pc_put(S, j, k);
}
static void pheap_rebuild(Sx *S) {
    if (S->pc_heap) {
        hp_clear(&S->ch);
        for (int j = 0; j < S->N; j++) { const double k = pmerit(S, j); if (k > 0) hp_set(&S->ch, j, k); }
        S->ch_ok = 1;
        return;
    }
    while (S->pc_n > 0) pc_drop(S, S->pc_n - 1);
    if (!S->pc_mb) S->pc_mb = xmalloc(sizeof(double) * (S->N + 1));
    double *mb = S->pc_mb; int np = 0;
    for (int j = 0; j < S->N; j++) { const double k = pmerit(S, j); if (k > 0) mb[np++] = k; }
    S->pc_tau = 0;
    if (np > SX_PC_K) {     /* the SX_PC_K-th largest merit by selection (quickselect on a copy) */
        int lo = 0, hi = np - 1; const int kth = SX_PC_K - 1;
        while (lo < hi) {
            const double piv = mb[(lo + hi) / 2]; int i = lo, e = hi;
            while (i <= e) { while (mb[i] > piv) i++; while (mb[e] < piv) e--; if (i <= e) { const double t = mb[i]; mb[i] = mb[e]; mb[e] = t; i++; e--; } }
            if (kth <= e) hi = e; else if (kth >= i) lo = i; else break;
        }
        S->pc_tau = mb[kth];
    }
    /* the merits above the bound, then those equal to it while there is room (the lower indices first) */
    for (int j = 0; j < S->N; j++) { const double k = pmerit(S, j); if (k > S->pc_tau) pc_put(S, j, k); }
    if (S->pc_tau > 0) for (int j = 0; j < S->N && S->pc_n < S->pc_cap; j++) if (S->pc_at[j] < 0 && pmerit(S, j) == S->pc_tau) pc_put(S, j, S->pc_tau);
    S->ch_ok = 1;
}
/* the entering variable: the largest d_j^2 / w_j among the dual infeasible (Devex), from the heap or the list */
static int chuzc_primal(Sx *S) {
    if (!S->ch_ok) pheap_rebuild(S);
    if (S->pc_heap) return S->ch.n ? S->ch.h[0] : -1;
    for (int pass = 0; pass < 2; pass++) {
        int bj = -1; double bk = 0;
        for (int p = 0; p < S->pc_n; p++) { const int j = S->pc_set[p]; const double k = S->pc_kv[p]; if (k > bk || (k == bk && bj >= 0 && j < bj)) { bk = k; bj = j; } }
        if (bj >= 0 && bk >= S->pc_tau) return bj;
        if (S->pc_tau == 0) return bj;
        pheap_rebuild(S);
    }
    int bj = -1; double bk = 0;
    for (int p = 0; p < S->pc_n; p++) { const int j = S->pc_set[p]; const double k = S->pc_kv[p]; if (k > bk || (k == bk && bj >= 0 && j < bj)) { bk = k; bj = j; } }
    return bj;
}
static void pheap_touch_row(Sx *S) {
    if (!S->ch_ok) return;
    for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; pc_set_key(S, j, pmerit(S, j)); }
}
/* the Devex reference framework is reset (all weights 1) when the weights have grown too far from it: their growth
 * makes the pricing ignore columns (Forrest and Goldfarb 1992) */
static void devex_check(Sx *S) {
    double mx = 0;
    for (int j = 0; j < S->N; j++) if (S->pos[j] < 0 && S->dvx[j] > mx) mx = S->dvx[j];
    if (mx > 1e6) { for (int j = 0; j < S->N; j++) S->dvx[j] = 1.0; S->ndvxreset++; S->ch_ok = 0; }
}
/* primal values infeasible beyond the tolerance (phase 1 needed) */
static int primal_infeasible(const Sx *S) { for (int i = 0; i < S->m; i++) if (infeas(S, i) > 0) return 1; return 0; }

/* phase-1 duals: the costs -1 / +1 of the basic variables below / above their bounds, 0 elsewhere */
static void compute_dual_ph1(Sx *S) {
    SxVec *v = &S->rho; sxv_clear(v);
    for (int i = 0; i < S->m; i++) {
        const int j = S->basic[i];
        S->c1[i] = S->x[j] < S->lo[j] - S->tolp ? -1 : S->x[j] > S->up[j] + S->tolp ? 1 : 0;
        v->val[i] = S->c1[i];
    }
    sxv_reindex(v);
    double dd = 1; btran(S, v, &dd, NULL);
    const double *y = v->val;
    for (int j = 0; j < S->n; j++) {
        if (S->pos[j] >= 0) { S->d[j] = 0; continue; }
        double s = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s -= S->Ax[p] * y[S->Ai[p]];
        S->d[j] = s;
    }
    for (int i = 0; i < S->m; i++) { const int j = S->n + i; S->d[j] = S->pos[j] >= 0 ? 0 : -y[i]; }
    sxv_clear(v);
}

/* phase 2 of the primal method keeps the basis feasible: a basic variable that drifts beyond a bound by more than the
 * tolerance (the Harris steps allow it, rounding adds to it) gets that bound moved to (1 + random) tolerances beyond
 * its value (HiGHS); the moves are undone with the other bound changes at the end. Returns 1 if a violation was too
 * large for that (relative 1e-5: phase 1 then). */
static int shift_bounds_primal(Sx *S, const SxVec *v) {
    int big = 0;
    const int cnt = v ? v->cnt : S->m;
    for (int k = 0; k < cnt; k++) {
        const int i = v ? v->idx[k] : k;
        const int j = S->basic[i];
        const double x = S->x[j];
        if (x < S->lo[j] - S->tolp) {
            if (S->lo[j] - x > 1e-5 * (1 + fabs(S->lo[j]))) { big = 1; continue; }
            S->lo[j] = x - (1 + urand(S)) * S->tolp; S->nbshift++;
        } else if (x > S->up[j] + S->tolp) {
            if (x - S->up[j] > 1e-5 * (1 + fabs(S->up[j]))) { big = 1; continue; }
            S->up[j] = x + (1 + urand(S)) * S->tolp; S->nbshift++;
        }
    }
    return big;
}

/* primal simplex iterations (phase 1 when the basis is infeasible, then phase 2): 0 optimal, 2 unbounded, 1 infeasible.
 * Phase 1 minimizes the sum of infeasibilities with the costs -1 / +1 of the basic variables below / above their bounds;
 * its duals are recomputed after every iteration (the costs change with the set of infeasible variables). Phase 2 updates
 * the duals by the pivotal row and recomputes them at each refactorization. */
static int primal_loop(Sx *S) {
    SxVec *rho = &S->rho, *col = &S->col;
    int need_rebuild = 1, verified = 0, ph1 = 0;
    long last_log = -1000;
    S->ch_ok = 0;
    for (;;) {
        int st = check_stop(S);
        if (st) return st;
        if (need_rebuild || need_refactor(S)) {
            const int was2 = !ph1 && S->it_primal > 0;
            if (S->fresh && !need_refactor(S)) S->fresh = 0;      /* (the factorization and the values as sxm_solve left them) */
            else { S->fresh = 0; refactor(S); compute_primal(S); devex_check(S); }
            if (was2) shift_bounds_primal(S, NULL);      /* the drift of the recomputation absorbed in phase 2 */
            ph1 = primal_infeasible(S);
            if (ph1) compute_dual_ph1(S); else compute_dual(S);
            S->ch_ok = 0;
            need_rebuild = 0;
        }
        if (S->iter - last_log >= 1000) { log_line(S, ph1 ? "prim ph1" : "primal"); last_log = S->iter; }
        const int q = chuzc_primal(S);
        if (q < 0) {
            if (S->F.nupd > 0 && !verified && !(S->noverify && !ph1)) { need_rebuild = 1; verified = 1; continue; }
            if (!ph1) return 0;
            /* phase 1 ends with infeasibilities: the problem is infeasible only if they are not tiny (below the level
             * where the phase-1 reduced costs carry information); tiny ones get the bound moved to the value (removed
             * with the other bound changes at the end) */
            double mx = 0;
            for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; const double v = S->x[j];
                const double e = v < S->lo[j] ? (S->lo[j] - v) / (1 + fabs(S->lo[j])) : v > S->up[j] ? (v - S->up[j]) / (1 + fabs(S->up[j])) : 0; if (e > mx) mx = e; }
            if (mx > 1e-6) return 1;
            for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; if (S->x[j] < S->lo[j]) S->lo[j] = S->x[j]; else if (S->x[j] > S->up[j]) S->up[j] = S->x[j]; }
            S->nbshift++;
            ph1 = 0; compute_dual(S); S->ch_ok = 0; verified = 0;
            continue;
        }
        verified = 0;
        const double dir = (S->move[q] == 0) ? (S->d[q] > 0 ? -1.0 : 1.0) : (double)S->move[q];
        sxv_clear(col); col_scatter(S, q, col, 1.0);
        ftran(S, col, &S->dens_col, &S->spike);
        if (col->cnt < 0) sxv_reindex(col);
        /* ratio test (Harris, two passes): x_B(t) = x_B - t dir alpha */
        const double ptol = S->tolp;
        double thmax = SX_INF;
        for (int k = 0; k < col->cnt; k++) {
            const int i = col->idx[k];
            const double a = dir * col->val[i];
            if (fabs(a) < 1e-9) continue;
            const int j = S->basic[i];
            const double v = S->x[j];
            double lo = S->lo[j], up = S->up[j];
            if (ph1) {   /* an infeasible variable may move up to its violated bound, and not beyond */
                if (v < lo - ptol) { if (a < 0) { up = lo; lo = -SX_INF; } else continue; }
                else if (v > up + ptol) { if (a > 0) { lo = up; up = SX_INF; } else continue; }
            }
            if (a > 0 ? lo <= -SX_INF : up >= SX_INF) continue;
            const double t = a > 0 ? (v - lo + ptol) / a : (up - v + ptol) / -a;
            if (t < thmax) thmax = t;
        }
        int r = -1; double best = 0, thp = 0;
        if (thmax < SX_INF) {
            for (int k = 0; k < col->cnt; k++) {
                const int i = col->idx[k];
                const double a = dir * col->val[i];
                if (fabs(a) < 1e-9) continue;
                const int j = S->basic[i];
                const double v = S->x[j];
                double lo = S->lo[j], up = S->up[j];
                if (ph1) {
                    if (v < lo - ptol) { if (a < 0) { up = lo; lo = -SX_INF; } else continue; }
                    else if (v > up + ptol) { if (a > 0) { lo = up; up = SX_INF; } else continue; }
                }
                if (a > 0 ? lo <= -SX_INF : up >= SX_INF) continue;
                const double t = a > 0 ? (v - lo) / a : (up - v) / -a;
                if (t <= thmax && fabs(a) > best) { best = fabs(a); r = i; thp = t; }
            }
        }
        const double rangeq = S->up[q] - S->lo[q];
        int flip = rangeq < SX_INF && rangeq <= thmax && (r < 0 || rangeq <= thp);
        if (!flip && r < 0) {
            if (ph1) { S->dvx[q] = fmin(S->dvx[q] * 1e3, 1e30); pc_set_key(S, q, pmerit(S, q)); continue; }
            if (S->F.nupd > 0) { need_rebuild = 1; continue; }
            if (S->prayout) {     /* the direction: the entering variable by dir, the basic ones by -dir B^{-1} a_q */
                memset(S->prayout, 0, sizeof(double) * S->N);
                for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->prayout[S->basic[i]] = -dir * col->val[i]; }
                S->prayout[q] = dir; S->have_pray = 1;
            }
            return 2;
        }
        int jl = -1; double ar = 0, bound = 0;
        if (flip) {
            /* a bound flip of the entering variable (no basis change) */
            const double dx = dir * rangeq;
            for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= dx * col->val[i]; }
            S->x[q] = S->move[q] > 0 ? S->up[q] : S->lo[q];
            S->move[q] = -S->move[q];
            S->nflips++;
            pc_set_key(S, q, pmerit(S, q));
        } else {
            if (thp < 0) thp = 0;          /* a variable slightly beyond its bound: a step of zero */
            jl = S->basic[r];
            ar = col->val[r];
            const double a = dir * ar;
            {
                const double v = S->x[jl];
                if (ph1 && v < S->lo[jl] - ptol) bound = S->lo[jl];
                else if (ph1 && v > S->up[jl] + ptol) bound = S->up[jl];
                else bound = a > 0 ? S->lo[jl] : S->up[jl];
            }
            /* BTRAN e_r and the pivotal row (the dual update and the Devex weights) */
            sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
            btran(S, rho, &S->dens_rho, &S->upart);
            if (rho->cnt < 0) sxv_reindex(rho);
            price(S, rho);
            /* primal update */
            const double step = thp * dir;
            for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= step * col->val[i]; }
            S->x[q] += step;
            S->x[jl] = bound;
            if (!ph1) shift_bounds_primal(S, col);
            /* dual update (with the costs as they are: phase 1's change below) */
            const double thd = S->d[q] / ar;
            for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; S->d[j] -= thd * S->row.val[j]; }
            S->d[q] = 0; S->d[jl] = -thd;
            /* Devex weights */
            {
                const double wq = S->dvx[q];
                for (int k = 0; k < S->row.cnt; k++) {
                    const int j = S->row.idx[k];
                    const double rr = S->row.val[j] / ar;
                    const double nw = rr * rr * wq;
                    if (nw > S->dvx[j]) S->dvx[j] = nw < 1e30 ? nw : 1e30;
                }
                const double wl = wq / (ar * ar);
                S->dvx[jl] = wl > 1 ? (wl < 1e30 ? wl : 1e30) : 1;
            }
            S->basic[r] = q; S->pos[q] = r; S->move[q] = 0;
            S->pos[jl] = -1;
            rows_enter(S, q); rows_leave(S, jl);
            if (S->lo[jl] == S->up[jl]) S->move[jl] = 0;
            else if (bound == S->lo[jl]) S->move[jl] = 1;
            else S->move[jl] = -1;
            const int bad = sxlu_update(&S->F, r, &S->spike, &S->upart, ar);
            if (bad) need_rebuild = 1;
            /* the candidates: the row's entries, the entering one out, the leaving one in */
            pheap_touch_row(S);
            pc_set_key(S, q, 0);
        }
        /* phase 1: the costs of the basic variables whose feasibility changed (and of the leaving variable, nonbasic
         * now with cost 0): the duals follow by one solve with the change (y += B^{-T} delta) */
        if (ph1 && !need_rebuild) {
            SxVec *dl = &S->tau; sxv_clear(dl);
            int nch = 0, ninf = 0;
            if (!flip) {
                const int cl = S->c1[r];
                if (cl) { S->d[jl] -= cl; }
                S->c1[r] = 0;   /* q enters at a value within its bounds */
            }
            for (int k = 0; k < col->cnt; k++) {
                const int i = col->idx[k];
                const int j = S->basic[i];
                const signed char nc = S->x[j] < S->lo[j] - S->tolp ? -1 : S->x[j] > S->up[j] + S->tolp ? 1 : 0;
                if (nc != S->c1[i]) { dl->val[i] = nc - S->c1[i]; dl->idx[dl->cnt++] = i; S->c1[i] = nc; nch++; }
            }
            if (!flip) { const int j = S->basic[r]; const signed char nc = S->x[j] < S->lo[j] - S->tolp ? -1 : S->x[j] > S->up[j] + S->tolp ? 1 : 0;
                if (nc != S->c1[r]) { if (dl->val[r] == 0) dl->idx[dl->cnt++] = r; dl->val[r] += nc - S->c1[r]; S->c1[r] = nc; nch++; } }
            if (!flip) pc_set_key(S, jl, pmerit(S, jl));
            for (int i = 0; i < S->m && !ninf; i++) ninf += S->c1[i] != 0;
            if (!ninf) {
                /* feasible: phase 2 with the costs of the problem */
                ph1 = 0; compute_dual(S); S->ch_ok = 0;
            } else if (nch) {
                double dd = S->dens_tau;
                btran(S, dl, &dd, NULL);
                if (dl->cnt < 0) sxv_reindex(dl);
                price(S, dl);
                for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; S->d[j] -= S->row.val[j]; }
                pheap_touch_row(S);
            }
            sxv_clear(dl);
        } else if (!flip) pc_set_key(S, jl, pmerit(S, jl));
        S->iter++; S->it_primal++; if (ph1) S->it_ph1++;
    }
}

/* the primal method with its bounds perturbed against primal degeneracy (each finite bound moved outwards by a random
 * relative amount of at most 5e-7), then the bounds restored: the nonbasic variables go back to their bounds, and the
 * primal infeasibilities this leaves are removed by the dual method (the basis is dual feasible) */
static int dual_loop(Sx *S);
static int primal_simplex(Sx *S, int perturb) {
    if (perturb) {
        const double base = 5e-7;
        /* the basic variables' bounds only: the basic solution stays as it is (and feasible if it was) */
        for (int i = 0; i < S->m; i++) {
            const int j = S->basic[i];
            const double l = S->lo[j], u = S->up[j];
            const double r = urand(S);
            if (l > -SX_INF) S->lo[j] = l - r * base * (fabs(l) > 1 ? fabs(l) : 1);
            if (u < SX_INF) S->up[j] = u + r * base * (fabs(u) > 1 ? fabs(u) : 1);
        }
    }
    int st = primal_loop(S);
    if (st != 0) return st;
    for (int round = 0; round < 3; round++) {
        memcpy(S->lo, S->lo0, sizeof(double) * S->N); memcpy(S->up, S->up0, sizeof(double) * S->N);
        normalize_nonbasic(S);
        compute_primal(S);
        if (!primal_infeasible(S)) return 0;
        if (S->verbose >= 2) printf("  primal infeasibilities after the bounds are restored: dual simplex\n");
        st = dual_loop(S);
        if (st != 0) return st;
        compute_dual(S);
        int bad = 0;
        for (int j = 0; j < S->N; j++) {
            if (S->pos[j] >= 0) continue;
            if (S->move[j] == 0) { if (S->lo[j] != S->up[j] && fabs(S->d[j]) > S->told) bad++; }
            else if (S->move[j] * S->d[j] < -S->told) bad++;
        }
        if (!bad) return 0;
        st = primal_loop(S);
        if (st != 0) return st;
    }
    return st;
}

/* ---------------------------------------------------------------- the driver */
/* (SX_TARGET_DEF, the largest violation accepted on the problem as read: at the top of the file) */

/* the largest violations of the current basic solution on the unscaled problem: the bounds of the basic variables,
 * relative to 1 + |bound|, and the signs of the reduced costs of the nonbasic variables, relative to the largest cost
 * when it exceeds 1 (an absolute dual tolerance on costs of 1e7 would ask for 1e-14 relative accuracy) */
static void unscaled_violations(const Sx *S, double *pv, double *dv) {
    double p = 0, d = 0;
    const double cs = S->costscale > 1 ? 1.0 : S->costscale;    /* d_j * costscale / max(1, costscale) */
    for (int i = 0; i < S->m; i++) {
        const int j = S->basic[i];
        const double sc = j < S->n ? S->cscale[j] : 1.0 / S->rscale[j - S->n];
        const double v = S->x[j];
        if (v < S->lo0[j]) p = fmax(p, (S->lo0[j] - v) * sc / (1 + fabs(S->lo0[j] * sc)));
        if (v > S->up0[j]) p = fmax(p, (v - S->up0[j]) * sc / (1 + fabs(S->up0[j] * sc)));
    }
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0) continue;
        const double sc = (j < S->n ? 1.0 / S->cscale[j] : S->rscale[j - S->n]) * cs;
        const double dj = S->d[j] * sc;
        if (S->lo0[j] == S->up0[j]) continue;
        const double dold = d;
        if (S->move[j] == 0) d = fmax(d, fabs(dj));
        else if (S->move[j] * dj < 0) d = fmax(d, -S->move[j] * dj);
        if (d > dold && d > 1e-3 && getenv("SXDBG")) printf("  dual violation %g at %d (n %d): move %d lo0 %g up0 %g lo %g up %g x %g d %g\n", d, j, S->n, S->move[j], S->lo0[j], S->up0[j], S->lo[j], S->up[j], S->x[j], S->d[j]);
    }
    *pv = p; *dv = d;
}

static int dual_simplex(Sx *S) {
    /* dual phase 1 when the starting basis has dual infeasibilities that a bound flip cannot remove */
    compute_dual(S);
    int need1 = 0;
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0) continue;
        const double dj = S->d[j];
        if (is_boxed(S, j) || S->lo[j] == S->up[j]) continue;
        if (S->lo[j] <= -SX_INF && S->up[j] >= SX_INF) { if (fabs(dj) > S->told) need1 = 1; }
        else if (S->lo[j] > -SX_INF) { if (dj < -S->told) need1 = 1; }
        else if (dj > S->told) need1 = 1;
    }
    if (need1) {
        S->phase = 1;
        for (int j = 0; j < S->N; j++) {
            const double l = S->lo0[j], u = S->up0[j];
            if (l <= -SX_INF && u >= SX_INF) { S->lo[j] = -1000; S->up[j] = 1000; }
            else if (l <= -SX_INF) { S->lo[j] = -1; S->up[j] = 0; }
            else if (u >= SX_INF) { S->lo[j] = 0; S->up[j] = 1; }
            else { S->lo[j] = 0; S->up[j] = 0; }
        }
        for (int j = 0; j < S->N; j++) if (S->pos[j] < 0) set_nonbasic(S, j, S->d[j]);
        int st = dual_loop(S);
        if (st == 3 || st == 6) return st;
        /* the auxiliary problem's objective: 0 means a dual feasible basis */
        memcpy(S->lo, S->lo0, sizeof(double) * S->N); memcpy(S->up, S->up0, sizeof(double) * S->N);
        compute_dual(S);
        int bad = 0;
        for (int j = 0; j < S->N; j++) {
            if (S->pos[j] >= 0) continue;
            set_nonbasic(S, j, S->d[j]);
            const double dj = S->d[j];
            if (S->move[j] == 0) { if (S->lo[j] != S->up[j] && fabs(dj) > S->told) bad++; }
            else if (S->move[j] * dj < -S->told) bad++;
        }
        S->phase = 2;
        if (bad) return 2;      /* dual infeasible: settled by the primal method */
    } else {
        for (int j = 0; j < S->N; j++) if (S->pos[j] < 0) set_nonbasic(S, j, S->d[j]);
    }
    S->phase = 2;
    return dual_loop(S);
}

/* ---------------------------------------------------------------- crossover
 * From an optimal point that is not a basic solution (the solution of the problem reduced by its symmetry, or of an
 * interior-point method) to an optimal basis (the pushes of Bixby and Saltzman). The start: the variables strictly
 * between their bounds in the basis (as many as are independent: the factorization replaces the others by logicals),
 * the logicals of the other rows filling it, every other variable nonbasic at its value. The dual push: each basic
 * variable whose reduced cost at the given duals is not zero (it is at a bound, by complementarity) leaves the basis,
 * the duals moving along its row of B^{-1}; then every basic variable has reduced cost zero, so the given duals are
 * those of the basis. The primal push: each nonbasic variable not at a bound (superbasic; its reduced cost is zero)
 * moves to its nearer bound with the basic variables following, or enters the basis when a basic variable reaches a
 * bound first; these moves keep the point feasible and the objective as it is, and the duals stay those of the basis.
 * With an exact optimal point the basis is then optimal; the primal method removes what the tolerances leave. x0: the
 * structural values of the problem as read, y0 its row duals (may be NULL: no dual push); returns the primal pushes. */
static void xo_dbg(Sx *S, const char *when, const double *xs) {
    if (!S->dbg) return;
    double mi = 0, md = 0; int ni = 0;
    for (int i = 0; i < S->m; i++) { const double e = infeas(S, i); if (e > S->tolp) { ni++; if (e > mi) mi = e; } }
    if (xs) for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; const double d = fabs(S->x[j] - xs[j]); if (d > md) md = d; }
    printf("  crossover %s: %d basic infeasible (largest %.1e), largest change of a basic value %.1e\n", when, ni, mi, md);
}
#define SX_XO_PIVTOL 1e-7     /* the pushes' pivot tolerance (the scaled problem) */
#define SX_XO_RELPIV 1e-4     /* the primal push's pivot tolerance relative to the largest entry of the column */
#define SX_XO_CANC 0.1        /* the start basis: a pivot below this times the largest entry of its column is refused */
/* an estimate of the condition number of the basis in the 1-norm, ||B||_1 ||B^{-1}||_1, the second factor by Hager's
 * method (Higham 1988: a few solves with B and B^T, a lower bound that is rarely far off) */
static double basis_cond(Sx *S) {
    const int m = S->m;
    double nb = 0;
    for (int i = 0; i < m; i++) {
        const int j = S->basic[i]; double c = 0;
        if (j >= S->n) c = 1; else for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) c += fabs(S->Ax[p]);
        if (c > nb) nb = c;
    }
    SxVec V; sxv_init(&V, m); SxVec *v = &V;
    double dd = 1, est = 0, *xv = xmalloc(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) v->val[i] = 1.0 / m;
    sxv_reindex(v);
    for (int it = 0; it < 5; it++) {
        ftran(S, v, &dd, NULL);
        if (v->cnt < 0) sxv_reindex(v);
        double e = 0; for (int i = 0; i < m; i++) { e += fabs(v->val[i]); xv[i] = v->val[i] >= 0 ? 1.0 : -1.0; }
        if (it > 0 && e <= est) { est = fmax(est, e); break; }
        est = e;
        sxv_clear(v); for (int i = 0; i < m; i++) v->val[i] = xv[i];
        sxv_reindex(v);
        btran(S, v, &dd, NULL);
        if (v->cnt < 0) sxv_reindex(v);
        int jm = 0; double zm = -1; for (int i = 0; i < m; i++) if (fabs(v->val[i]) > zm) { zm = fabs(v->val[i]); jm = i; }
        sxv_clear(v); v->val[jm] = 1; v->idx[0] = jm; v->cnt = 1;
    }
    sxv_free(v); free(xv);
    return nb * est;
}
static long crossover_push(Sx *S, const double *x0, const double *y0, long *ndpush) {
    const int n = S->n, m = S->m, N = S->N;
    double *xs = xmalloc(sizeof(double) * (N + 1));
    for (int j = 0; j < n; j++) xs[j] = x0[j] / S->cscale[j];
    for (int i = 0; i < m; i++) xs[n + i] = 0;
    for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) xs[n + S->Ai[p]] -= S->Ax[p] * xs[j];
    if (S->dbg) {
        double mv = 0; int jm = -1;
        for (int j = 0; j < N; j++) { const double v = fmax(S->lo[j] - xs[j], xs[j] - S->up[j]); if (v > mv) { mv = v; jm = j; } }
        printf("  crossover: the given point's largest violation %.1e (variable %d of %d structurals)\n", mv, jm, n);
    }
    for (int j = 0; j < N; j++) { if (xs[j] < S->lo[j]) xs[j] = S->lo[j]; if (xs[j] > S->up[j]) xs[j] = S->up[j]; }
    /* the reduced costs at the given duals (scaled), for the dual push and to read which bound a variable of an
     * interior point belongs to */
    double *dz = NULL;
    if (y0) {
        dz = xmalloc(sizeof(double) * (N + 1));
        const double sg = S->maxim ? -1.0 : 1.0;
        for (int i = 0; i < m; i++) dz[n + i] = -sg * y0[i] / (S->rscale[i] * S->costscale);
        for (int j = 0; j < n; j++) { double v = S->cost0[j]; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v += S->Ax[p] * dz[n + S->Ai[p]]; dz[j] = v; }
    }
    /* the variables strictly inside their bounds (relative margin 1e-9); with duals, a variable whose distance to a
     * bound is below its reduced cost of the sign of that bound is at the bound (an interior point's complementarity;
     * the basic variables absorb the move, slightly infeasible at the start: by up to 1e-3 on rmatr200-p5, which the
     * pushes' ratio test, below, keeps from growing) */
    char *in = xcalloc(N + 1, 1), *at = xcalloc(N + 1, 1);     /* at: 1 at the lower, 2 at the upper bound by the rule above */
    /* (an experiment, 23 October 2026: BRISK_SXXOATTOL, the relative distance below which a variable is at its bound) */
    static double attol = -1; if (attol < 0) { const char *e = getenv("BRISK_SXXOATTOL"); attol = e ? atof(e) : 1e-9; }
    for (int j = 0; j < N; j++) {
        const double l = S->lo[j], u = S->up[j], v = xs[j];
        int al = l > -SX_INF && v - l <= attol * (1 + fabs(l)), au = u < SX_INF && u - v <= attol * (1 + fabs(u));
        if (dz && !al && !au) {
            if (l > -SX_INF && dz[j] > 0 && v - l < dz[j]) al = 1;
            else if (u < SX_INF && dz[j] < 0 && u - v < -dz[j]) au = 1;
        }
        if (al) xs[j] = l; else if (au) xs[j] = u;
        in[j] = !al && !au && !(l <= -SX_INF && u >= SX_INF && v == 0);
        at[j] = al ? 1 : au ? 2 : 0;
    }
    if (S->dbg) {   /* (24 October 2026) the inside variables' distances to their nearer bound by decade, and their reduced costs */
        int hx[20] = { 0 }, hz[20] = { 0 };
        for (int j = 0; j < N; j++) {
            if (!in[j]) continue;
            const double d = fmin(S->lo[j] > -SX_INF ? xs[j] - S->lo[j] : SX_INF, S->up[j] < SX_INF ? S->up[j] - xs[j] : SX_INF);
            int k = d > 0 && d < SX_INF ? (int)floor(log10(d)) + 16 : 19; if (k < 0) k = 0; if (k > 19) k = 19; hx[k]++;
            if (dz) { const double a = fabs(dz[j]); int q = a > 0 ? (int)floor(log10(a)) + 16 : 0; if (q < 0) q = 0; if (q > 19) q = 19; hz[q]++; }
        }
        printf("  inside variables: distance to the nearer bound by decade (1e-16 .. 1e3, last: no bound):"); for (int k = 0; k < 20; k++) printf(" %d", hx[k]); printf("\n");
        if (dz) { printf("  inside variables: reduced cost by decade:"); for (int k = 0; k < 20; k++) printf(" %d", hz[k]); printf("\n"); }
    }
    /* the basis: the inside logicals, the inside structurals, then the logicals of the other rows */
    int nb = 0;
    for (int j = 0; j < N; j++) S->pos[j] = -1;
    for (int i = 0; i < m && nb < m; i++) if (in[n + i]) { S->basic[nb] = n + i; S->pos[n + i] = nb++; }
    for (int j = 0; j < n && nb < m; j++) if (in[j]) { S->basic[nb] = j; S->pos[j] = nb++; }
    /* (24 October 2026) then the variables at a bound whose reduced cost at the given duals is zero, logicals first: a
     * basic variable with a nonzero reduced cost costs a dual push, and the logical of a row with a nonzero dual is
     * one. On the unfolding of a symmetric solution (qap15, LP: 5,726 inside of 6,330 rows) the logicals that completed
     * the basis cost 1,196 dual pushes; Gurobi's start from the same kind of point needs none. BRISK_SXXOZDZ=0: off. */
    static int zdz = -1; if (zdz < 0) { const char *e = getenv("BRISK_SXXOZDZ"); zdz = e ? atoi(e) : 1; }
    if (dz && zdz) {
        for (int i = 0; i < m && nb < m; i++) if (S->pos[n + i] < 0 && fabs(dz[n + i]) <= S->told) { S->basic[nb] = n + i; S->pos[n + i] = nb++; }
        for (int j = 0; j < n && nb < m; j++) if (S->pos[j] < 0 && fabs(dz[j]) <= S->told) { S->basic[nb] = j; S->pos[j] = nb++; }
    }
    for (int i = 0; i < m && nb < m; i++) if (S->pos[n + i] < 0) { S->basic[nb] = n + i; S->pos[n + i] = nb++; }
    for (int i = 0; i < m; i++) S->w[i] = 1.0;
    for (int j = 0; j < N; j++) S->dvx[j] = 1.0;
    rows_partition(S);
    for (int j = 0; j < N; j++) S->x[j] = xs[j];
    S->keepx = 1;
    /* the start basis keeps only columns that do not nearly depend on those before them: a pivot below SX_XO_CANC times
     * the largest entry of its column refused (the column then gives way to a logical and stays superbasic for the
     * primal push). Without it, on pilot87 the 2,550 variables inside their bounds (1,902 rows) gave a basis whose
     * condition estimate was 7.6e37 (the factorization's own tests are relative to the active submatrix and cannot see
     * a column that lost all but a part in 1e10 of its size to cancellation): basic values of 1.7e28, 718
     * infeasibilities after the pushes, 14,392 iterations of the primal method, 9.7 s; with it 7.5e5, 6 iterations,
     * 1.4 s (22 October 2026) */
    S->F.pivcanc = SX_XO_CANC;
    refactor(S);
    S->F.pivcanc = 0;
    /* every nonbasic variable at its value (the replaced ones too), its status by where it is */
    for (int j = 0; j < N; j++) {
        if (S->pos[j] >= 0) { S->move[j] = 0; continue; }
        S->x[j] = xs[j];
        S->move[j] = S->lo[j] == S->up[j] ? 0 : xs[j] == S->lo[j] ? 1 : xs[j] == S->up[j] ? -1 : 0;
    }
    compute_primal(S);
    if (S->dbg) printf("  crossover start: condition estimate %.1e\n", basis_cond(S));
    if (S->dbg) {
        double mv = 0; for (int j = 0; j < N; j++) { const double v = S->lo[j] - xs[j] > 0 ? S->lo[j] - xs[j] : xs[j] - S->up[j]; if (v > mv) mv = v; }
        int nin = 0; for (int j = 0; j < N; j++) nin += in[j];
        printf("  crossover start: %d inside, %d replaced by the factorization\n", nin, S->F.rank_deficient);
    }
    xo_dbg(S, "start", xs);
    SxVec *col = &S->col, *rho = &S->rho;
    long npush = 0;
    /* the dual push: a basic variable at a bound whose reduced cost at the given duals is not zero leaves the basis.
     * The duals move by t times row r of B^{-1} (y + t rho_r), which changes that reduced cost by -t and those of the
     * nonbasic variables by -t alpha_rk, the other basic ones not at all; t runs to the reduced cost unless a nonbasic
     * reduced cost would get the wrong sign first: then that variable enters (a step of zero in the primal, since the
     * leaving one is at its bound). At a bound means: put there by the rule above (its distance below its reduced cost),
     * or within the feasibility tolerance of it (21 October 2026: before, only the latter, on the values the basis gives;
     * on irish-electricity the logicals of 19,638 rows that the interior point leaves 1e-7 to 1e-5 from their bound with
     * duals up to 6.9e3 were taken as inside and left to the primal method, 4,512 infeasibilities after the pushes). */
    *ndpush = 0;
    if (dz) {
        const double told = S->told;
        if (S->dbg) {
            int bad = 0, skip = 0, basicnz = 0; double mx = 0;
            for (int j = 0; j < N; j++) {
                const double xv = S->x[j];
                const int fx = S->lo[j] == S->up[j];     /* (fixed: at its bound whatever the residual; a logical of an equation basic at 1e-7 on supportcase19) */
                const int atl = fx || at[j] == 1 || (S->lo[j] > -SX_INF && fabs(xv - S->lo[j]) <= S->tolp), atu = fx || at[j] == 2 || (S->up[j] < SX_INF && fabs(xv - S->up[j]) <= S->tolp);
                double e = 0;
                if (atl && atu) e = 0; else if (atl) e = -dz[j]; else if (atu) e = dz[j]; else e = fabs(dz[j]);
                if (e > told) { bad++; mx = fmax(mx, e); if (S->pos[j] >= 0 && !atl && !atu) skip++;
                    static int nbig = 0;
                    if (bad <= 10 || (S->dbg > 1 && e > 1e-2 && nbig++ < 15)) printf("    variable %d (%s %d): x %.6g in [%.6g, %.6g], reduced cost %.3g, %s\n", j, j < n ? "column" : "row", j < n ? j : j - n,
                                          xv, S->lo[j], S->up[j], dz[j], S->pos[j] >= 0 ? "basic" : "nonbasic"); }
                if (S->pos[j] >= 0 && fabs(dz[j]) > told) basicnz++;
            }
            printf("  given duals: %d violations of complementarity or sign (largest %.1e; %d basic inside), %d basic with nonzero reduced cost\n", bad, mx, skip, basicnz);
            {   /* (24 October 2026) the basic variables' reduced costs by decade, against the tolerance */
                int hist[20] = { 0 };
                for (int j = 0; j < N; j++) { if (S->pos[j] < 0) continue; const double a = fabs(dz[j]); int d = a > 0 ? (int)floor(log10(a)) + 16 : 0; if (d < 0) d = 0; if (d > 19) d = 19; hist[d]++; }
                printf("  basic reduced costs by decade (1e-16 .. 1e3; told %.0e):", told);
                for (int d = 0; d < 20; d++) printf(" %d", hist[d]);
                printf("\n");
            }
        }
        for (int j = 0; j < N; j++) {
            if (S->pos[j] < 0 || fabs(dz[j]) <= told) continue;
            const double xv = S->x[j];
            const int fx = S->lo[j] == S->up[j];
            const int atl = fx || at[j] == 1 || (S->lo[j] > -SX_INF && fabs(xv - S->lo[j]) <= S->tolp), atu = fx || at[j] == 2 || (S->up[j] < SX_INF && fabs(xv - S->up[j]) <= S->tolp);
            if (!atl && !atu) continue;              /* (not complementary: left to the primal method) */
            if ((atl && !atu && dz[j] < 0) || (atu && !atl && dz[j] > 0)) continue;   /* (the wrong sign: likewise) */
            if (need_refactor(S)) { refactor(S); compute_primal(S); }
            const int r = S->pos[j];
            const double sgt = dz[j] > 0 ? 1.0 : -1.0, smax = fabs(dz[j]);
            sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
            btran(S, rho, &S->dens_rho, &S->upart);
            if (rho->cnt < 0) sxv_reindex(rho);
            price(S, rho);
            const SxVec *row = &S->row;
            double hmax = smax;
            for (int k = 0; k < row->cnt; k++) {
                const int q = row->idx[k]; const double b = sgt * row->val[q];
                if (fabs(b) < SX_XO_PIVTOL || S->lo[q] == S->up[q]) continue;
                double t;
                if (S->move[q] > 0) { if (b <= 0) continue; t = (dz[q] + told) / b; }
                else if (S->move[q] < 0) { if (b >= 0) continue; t = (dz[q] - told) / b; }
                else t = told / fabs(b);
                if (t < hmax) hmax = t < 0 ? 0 : t;
            }
            int q = -1; double best = 0, sstep = smax;
            if (hmax < smax) {
                for (int k = 0; k < row->cnt; k++) {
                    const int c = row->idx[k]; const double b = sgt * row->val[c];
                    if (fabs(b) < SX_XO_PIVTOL || S->lo[c] == S->up[c]) continue;
                    double t;
                    if (S->move[c] > 0) { if (b <= 0) continue; t = dz[c] / b; }
                    else if (S->move[c] < 0) { if (b >= 0) continue; t = dz[c] / b; }
                    else t = 0;
                    if (t <= hmax && fabs(b) > best) { best = fabs(b); q = c; sstep = t < 0 ? 0 : t; }
                }
            }
            for (int k = 0; k < row->cnt; k++) { const int c = row->idx[k]; dz[c] -= sstep * sgt * row->val[c]; }
            dz[j] -= sstep * sgt;
            (*ndpush)++;
            if (q < 0) { dz[j] = 0; S->xo_dnoblock++; continue; }
            /* q enters at row r, j leaves at its bound */
            sxv_clear(col); col_scatter(S, q, col, 1.0);
            ftran(S, col, &S->dens_col, &S->spike);
            if (col->cnt < 0) sxv_reindex(col);
            const double ar = col->val[r];
            if (fabs(ar) < 1e-9) { sxv_clear(col); continue; }
            dz[q] = 0;
            S->x[j] = atl ? S->lo[j] : S->up[j];
            S->basic[r] = q; S->pos[q] = r; S->move[q] = 0; S->pos[j] = -1;
            S->move[j] = S->lo[j] == S->up[j] ? 0 : atl ? 1 : -1;
            rows_enter(S, q); rows_leave(S, j);
            if (sxlu_update(&S->F, r, &S->spike, &S->upart, ar)) { refactor(S); compute_primal(S); }
            sxv_clear(col);
        }
    }
    free(dz);
    xo_dbg(S, "after the dual push", xs);
    if (S->dbg) printf("  crossover: %ld dual pushes, %ld without a basis change\n", *ndpush, S->xo_dnoblock);
    /* the primal push (passes until no superbasic variable is left: a refactorization that replaces a column leaves it
     * superbasic) */
    /* (23 October 2026) pushes in batches: the variables of a batch moved to their targets at once, x_B -= B^-1 sum_q
     * (target_q - x_q) a_q, one solve; when every basic variable stays within its bounds (the feasibility tolerance) the
     * batch is taken (the state is that of pushing them one by one without a basis change), otherwise its variables are
     * pushed one by one as before. The batch doubles on success (up to 4096) and shrinks fourfold on a failure; at 2,
     * a failure turns batches off for the next 64 variables. supportcase12 (LP): 136,070 pushes of which about 100,000
     * change no basis, each with a solve. BRISK_SXXOBATCH=0: one by one */
    static int xbatch = -1; if (xbatch < 0) { const char *e = getenv("BRISK_SXXOBATCH"); xbatch = !(e && atoi(e) == 0); }
    int bsz = 64, single_until = -1, nbat = 0, nbatq = 0;
    double dupart = 0;                  /* (the update's solves' density, below) */
    int *bq = xbatch ? xmalloc(sizeof(int) * 4096) : NULL; double *bt = xbatch ? xmalloc(sizeof(double) * 4096) : NULL;
    /* (25 October 2026) the basic values recomputed before the primal pushes: a dual push sets the variable that leaves to
     * its bound (at a bound means within the tolerance, or put there by the start's rule: its distance below its reduced
     * cost) without moving the basic variables with it, so the values the pushes start from were not those of the basis
     * (neos-3216931-puriri, LP: largest infeasibility 0.0074 updated against 0.014 recomputed); the batches of pushes,
     * accepted on those values, then left 469 basic variables infeasible by up to 5.6 and 3,641 iterations of the dual
     * method after the pushes; with the values recomputed 51 (largest 0.0078) and 823 iterations, 2.29 -> 1.23 s in all.
     * One solve. BRISK_SXXOFIX0=0: not */
    { static int fix0 = -1; if (fix0 < 0) { const char *e = getenv("BRISK_SXXOFIX0"); fix0 = !(e && atoi(e) == 0); }
      if (fix0) { if (S->dbg) { double mi = 0; for (int i = 0; i < m; i++) mi = fmax(mi, infeas(S, i)); printf("  crossover: before the primal push, largest basic infeasibility %.1e (updated)", mi); }
                  compute_primal(S);
                  if (S->dbg) { double mi = 0; for (int i = 0; i < m; i++) mi = fmax(mi, infeas(S, i)); printf(", %.1e (recomputed)\n", mi); } } }
    for (int pass = 0, left = 1; pass < 5 && left; pass++)
    for (int q = (left = 0); q < N; q++) {
        if (S->pos[q] >= 0) continue;
        const double l = S->lo[q], u = S->up[q], v = S->x[q];
        if (v == l || v == u || (l <= -SX_INF && u >= SX_INF && v == 0)) continue;
        left = 1;
        if (need_refactor(S)) {
            double mi0 = 0; if (S->dbg > 1) for (int i = 0; i < m; i++) mi0 = fmax(mi0, infeas(S, i));
            const int nout = refactor(S); compute_primal(S);
            if (S->dbg > 1) { double mi = 0; for (int i = 0; i < m; i++) mi = fmax(mi, infeas(S, i)); if (mi > 10 * mi0 + 1e-6) printf("  refactorization in the primal push: %d replaced, basic infeasibility %.1e -> %.1e\n", nout, mi0, mi); }
        }
        if (xbatch && q > single_until) {
            int nbq = 0;
            sxv_clear(col);
            for (int q2 = q; q2 < N && nbq < bsz; q2++) {
                if (S->pos[q2] >= 0) continue;
                const double l2 = S->lo[q2], u2 = S->up[q2], v2 = S->x[q2];
                if (v2 == l2 || v2 == u2 || (l2 <= -SX_INF && u2 >= SX_INF && v2 == 0)) continue;
                const double t2 = l2 <= -SX_INF && u2 >= SX_INF ? 0 : l2 <= -SX_INF ? u2 : u2 >= SX_INF ? l2 : (v2 - l2 <= u2 - v2 ? l2 : u2);
                col_scatter(S, q2, col, t2 - v2);
                bq[nbq] = q2; bt[nbq] = t2; nbq++;
            }
            if (nbq >= 2) {
                double dd = fmin(1.0, S->dens_col * nbq);   /* (its own density estimate: the single pushes' stays theirs) */
                ftran(S, col, &dd, NULL);
                if (col->cnt < 0) sxv_reindex(col);
                const double ptol = S->tolp;
                int ok = 1;
                for (int k = 0; k < col->cnt && ok; k++) {
                    const int i = col->idx[k], j = S->basic[i]; const double xn = S->x[j] - col->val[i];
                    if (xn < S->lo[j] - ptol || xn > S->up[j] + ptol) ok = 0;
                }
                if (ok) {
                    for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= col->val[i]; }
                    for (int k = 0; k < nbq; k++) {
                        const int j = bq[k]; const double t2 = bt[k], l2 = S->lo[j], u2 = S->up[j];
                        S->x[j] = t2;
                        S->move[j] = l2 == u2 ? 0 : t2 == l2 && l2 > -SX_INF ? 1 : t2 == u2 && u2 < SX_INF ? -1 : 0;
                    }
                    npush += nbq; nbat++; nbatq += nbq;
                    if (bsz < 4096) bsz *= 2;
                    continue;
                }
                /* (25 October 2026) a failed batch is tried again from the same variable at a quarter of its size, down to two,
                 * before its variables are pushed one by one: the batches grow to 4,096 on success, and a failure sent all of
                 * them to single pushes, most of which then went to their bounds without a pivot (supportcase12, LP: 41,774
                 * of 57,462). The crossover of supportcase12 2.07 -> 1.68 s, neos-960392's unfolding 0.47 -> 0.42 s, four
                 * other crossover-heavy LPs the same. BRISK_SXXORETRY=0: as before */
                { static int xretry = -1; if (xretry < 0) { const char *e = getenv("BRISK_SXXORETRY"); xretry = !(e && atoi(e) == 0); }
                  if (xretry && bsz > 2) { bsz = bsz / 4 < 2 ? 2 : bsz / 4; q--; continue; } }
                single_until = bq[nbq - 1];
                if (bsz <= 2) { int c = 0; for (int q2 = q; q2 < N && c < 64; q2++) if (S->pos[q2] < 0) { single_until = q2; c++; } }
                else bsz = bsz / 4 < 2 ? 2 : bsz / 4;
            }
        }
        /* the target: the nearer bound (zero for a free variable) */
        double target = l <= -SX_INF && u >= SX_INF ? 0 : l <= -SX_INF ? u : u >= SX_INF ? l : (v - l <= u - v ? l : u);
        const double dist = fabs(target - v), dir = target > v ? 1.0 : -1.0;
        sxv_clear(col); col_scatter(S, q, col, 1.0);
        ftran(S, col, &S->dens_col, &S->spike);
        if (col->cnt < 0) sxv_reindex(col);
        /* ratio test (Harris, two passes): x_B(t) = x_B - t dir alpha, 0 <= t <= dist; entries below the pivot
         * tolerance do not limit the step, and a basic variable already beyond its bound limits it to zero (its ratio
         * is not negative: a negative one admits only entries smaller still; on rmatr200-p5, pivots of 1e-9 in columns
         * of 1e4 made the basis nearly singular, and the pushes left 6857 violations of up to 2.7e3, 103 s of the
         * primal method after them instead of 76 iterations) */
        const double ptol = S->tolp;
        double thmax = dist;
        /* the pivot tolerance relative to the column's largest entry as well (22 October 2026): with the absolute one alone
         * the pushes took pivots of 1e-6 to 1e-10 of their column (on pilot87 from the slack basis, a third of the
         * pivots below 1e-3, a tenth below 1e-6), the basis grew so ill-conditioned that every update failed its check
         * and every new factorization recomputed basic values off by up to 1e13. An entry below it does not limit the step
         * (its basic variable may leave its bound by at most step * SX_XO_RELPIV of the column; the primal method cleans
         * up) */
        double amax = 0; for (int k = 0; k < col->cnt; k++) amax = fmax(amax, fabs(col->val[col->idx[k]]));
        const double atol = fmax(SX_XO_PIVTOL, SX_XO_RELPIV * amax);
        for (int k = 0; k < col->cnt; k++) {
            const int i = col->idx[k]; const double a = dir * col->val[i];
            if (fabs(a) < atol) continue;
            const int j = S->basic[i]; const double xv = S->x[j];
            if (a > 0 ? S->lo[j] <= -SX_INF : S->up[j] >= SX_INF) continue;
            const double t = a > 0 ? (xv - S->lo[j] + ptol) / a : (S->up[j] - xv + ptol) / -a;
            if (t < thmax) thmax = t < 0 ? 0 : t;
        }
        int r = -1; double best = 0, thp = dist;
        if (thmax < dist) {
            for (int k = 0; k < col->cnt; k++) {
                const int i = col->idx[k]; const double a = dir * col->val[i];
                if (fabs(a) < atol) continue;
                const int j = S->basic[i]; const double xv = S->x[j];
                if (a > 0 ? S->lo[j] <= -SX_INF : S->up[j] >= SX_INF) continue;
                const double t = a > 0 ? (xv - S->lo[j]) / a : (S->up[j] - xv) / -a;
                if (t <= thmax && fabs(a) > best) { best = fabs(a); r = i; thp = t < 0 ? 0 : t; }
            }
        }
        const double step = (r < 0 ? dist : thp) * dir;
        for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= step * col->val[i]; }
        npush++;
        if (S->dbg > 1) {      /* (the pushes after which the largest basic infeasibility grew tenfold) */
            static double mimax = 1e-6;
            double mi = 0; for (int i = 0; i < m; i++) { const double e = infeas(S, i); if (e > mi) mi = e; }
            if (mi > 10 * mimax) { printf("  push %ld (variable %d, %s, |alpha_r| %.1e, step %.1e, dist %.1e, nupd %d, |col| %d): basic infeasibility %.1e\n", npush, q, r < 0 ? "to its bound" : "pivot", r < 0 ? 0 : fabs(col->val[r]), step, dist, S->F.nupd, col->cnt, mi); mimax = mi; }
        }
        if (r < 0) {
            S->x[q] = target;
            S->move[q] = l == u ? 0 : target == l && l > -SX_INF ? 1 : target == u && u < SX_INF ? -1 : 0;
            continue;
        }
        /* the basic variable of row r reaches its bound first: it leaves there, q enters */
        const int jl = S->basic[r]; const double ar = col->val[r];
        const double bound = dir * ar > 0 ? S->lo[jl] : S->up[jl];
        S->x[q] += step; S->x[jl] = bound;
        sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
        {   /* (23 October 2026) only the update's part of the solve: no pricing follows (qap15, LP: a whole solve of 0.47 ms
             * per pivoting push). (24 October 2026, the night) with its own density estimate: the dual pushes' full solves
             * of rho had left theirs high, so this one swept all of U (lp_beckmann_300, LP: 62,507 such solves of 120
             * microseconds on a basis of 90,000 rows, 7.5 of its 27 s) */
            const double t = now(); const long f0 = S->F.nflops;
            sxlu_btran_upart(&S->F, rho, dupart, &S->upart);
            if (rho->cnt >= 0) dupart = 0.95 * dupart + 0.05 * (double)rho->cnt / m;
            S->work += (double)(S->F.nflops - f0); S->t_btran += now() - t;
        }
        S->basic[r] = q; S->pos[q] = r; S->move[q] = 0; S->pos[jl] = -1;
        S->move[jl] = S->lo[jl] == S->up[jl] ? 0 : bound == S->lo[jl] ? 1 : -1;
        rows_enter(S, q); rows_leave(S, jl);
        if (sxlu_update(&S->F, r, &S->spike, &S->upart, ar)) { refactor(S); compute_primal(S); }
        sxv_clear(rho);
    }
    sxv_clear(col); sxv_clear(rho);
    free(bq); free(bt);
    if (S->dbg && xbatch) printf("  crossover: %d batches of pushes took %d variables\n", nbat, nbatq);
    xo_dbg(S, "after the primal push", xs);
    if (S->dbg) { int ns = 0; for (int q = 0; q < N; q++) if (S->pos[q] < 0 && S->x[q] != S->lo[q] && S->x[q] != S->up[q] && !(S->lo[q] <= -SX_INF && S->up[q] >= SX_INF && S->x[q] == 0)) ns++; printf("  crossover: %d superbasic left, %d replaced by the factorizations\n", ns, S->F.rank_deficient); }
    S->keepx = 0;
    free(xs); free(in); free(at);
    return npush;
}

static int sx_solve_core(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                         SimplexInfo *info) {
    Sx S; memset(&S, 0, sizeof S);
    SimplexInfo dummy; if (!info) info = &dummy;
    memset(info, 0, sizeof *info);
    S.t0 = now();
    S.rayout = o ? o->ray : NULL;
    S.tlim = o ? o->timelimit : 0; S.maxit = o ? o->maxit : 0; S.verbose = o ? o->verbose : 0; S.stop = o ? o->stop : NULL; S.threads = o ? o->threads : NULL; S.dbg = getenv("SXDBG") ? atoi(getenv("SXDBG")) : 0; if (S.dbg < 0) S.dbg = 1;
    S.workbudget = o ? o->workbudget : 0; S.budget_cb = o ? o->budget_cb : NULL; S.budget_ctx = o ? o->budget_ctx : NULL;
    const int method = o ? o->method : 0;
    sx_setup(&S, L, o);
    if (S.verbose >= 2 && S.ntiny) printf("  %ld matrix entries too small to matter dropped (below 1e-9 of their row's largest; each row's activity moved by at most 1e-10)\n", S.ntiny);
    for (int j = 0; j < S.N; j++) if (S.lo0[j] > S.up0[j] + 1e-9 * (1 + fabs(S.lo0[j]))) { info->status = 1; snprintf(info->why, sizeof info->why, "a variable or row has lower bound above upper bound"); sx_free(&S); return 1; }
    int st;
    if (o && o->cstat0 && o->rstat0) {
        /* a starting basis: if it is optimal (or nearly), the method that keeps its feasible side finishes */
        given_basis(&S, o->cstat0, o->rstat0);
        refactor(&S); compute_primal(&S); compute_dual(&S);
        S.it_given = 1;
    } else slack_basis(&S);
    const int pert = !(o && o->noperturb);
    int given_route = 0;     /* with a starting basis: 1 optimal as it is, 2 primal feasible, 3 dual feasible */
    if (S.it_given) {
        int pinf = primal_infeasible(&S), dinf = 0;
        for (int j = 0; j < S.N && !dinf; j++) {
            if (S.pos[j] >= 0) continue;
            if (S.move[j] == 0) { if (S.lo[j] != S.up[j] && fabs(S.d[j]) > S.told) dinf = 1; }
            else if (S.move[j] * S.d[j] < -S.told) dinf = 1;
        }
        given_route = !pinf && !dinf ? 1 : !pinf ? 2 : !dinf ? 3 : 0;
        if (S.dbg) {    /* (the given basis' defects: counts, the largest, the first few variables) */
            int np = 0, nd = 0, shown = 0; double mp = 0, md = 0;
            for (int i = 0; i < S.m; i++) { const double f = infeas(&S, i); if (f > 0) { np++; mp = fmax(mp, f); if (shown < 8) { shown++; printf("    primal infeasible: row %d, basic variable %d (%s), x %g in [%g, %g]\n", i, S.basic[i], S.basic[i] < S.n ? "column" : "logical", S.x[S.basic[i]], S.lo[S.basic[i]], S.up[S.basic[i]]); } } }
            shown = 0;
            for (int j = 0; j < S.N; j++) {
                if (S.pos[j] >= 0) continue;
                const double e = S.move[j] == 0 ? (S.lo[j] != S.up[j] ? fabs(S.d[j]) : 0) : -S.move[j] * S.d[j];
                if (e > S.told) { nd++; md = fmax(md, e); if (shown < 8) { shown++; printf("    dual infeasible: %s %d, move %d, d %g, x %g in [%g, %g]\n", j < S.n ? "column" : "logical", j < S.n ? j : j - S.n, S.move[j], S.d[j], S.x[j], S.lo[j], S.up[j]); } }
            }
            printf("  starting basis: %d primal infeasible (largest %.1e), %d dual infeasible (largest %.1e), %d replaced by the factorization\n", np, mp, nd, md, S.F.rank_deficient);
        }
        if (S.verbose >= 2) printf("  starting basis: %s\n", given_route == 1 ? "optimal" : given_route == 2 ? "primal feasible" : given_route == 3 ? "dual feasible" : "neither primal nor dual feasible");
    }
    const int xo = o && o->xstart && !S.it_given;
    int xo_dual = 0;
    if (xo) {
        long nd = 0;
        const long np = crossover_push(&S, o->xstart, o->ystart, &nd);
        if (S.verbose >= 1) printf("  crossover: %ld primal and %ld dual pushes (%.2f s)\n", np, nd, now() - S.t0);
        info->pushes = np + nd;
        refactor(&S); compute_primal(&S); compute_dual(&S);
        int np1 = 0, nd1 = 0; double mp = 0, md = 0;
        for (int i = 0; i < S.m; i++) { const double e = infeas(&S, i); if (e > S.tolp) { np1++; mp = fmax(mp, e); } }
        for (int j = 0; j < S.N; j++) { if (S.pos[j] >= 0 || S.lo[j] == S.up[j]) continue; const double e = S.move[j] > 0 ? -S.d[j] : S.move[j] < 0 ? S.d[j] : fabs(S.d[j]); if (e > S.told) { nd1++; md = fmax(md, e); } }
        /* 21 October 2026: a basis without dual infeasibilities (the given duals optimal, made the basis's by the dual
         * push) is finished by the dual method, which keeps them; the primal method's phase 1 gave up the objective
         * (irish-electricity: 263 primal infeasibilities after the pushes, then over 13,000 primal iterations that
         * lost feasibility again). Otherwise the primal method. */
        /* (24 October 2026) and also when the dual infeasibilities are fewer than the primal ones: the dual method's
         * recomputation of the duals flips them (boxed) or shifts their costs, the shifts removed at the end, where the
         * primal method would start a phase 1 from all the primal ones. On irish-electricity (LP) one dual infeasibility
         * of 1.7e-5 beside 161 primal ones sent the crossover to the primal method: 5,050 iterations and 11 s against
         * 339 iterations when rounding left none (the factorization's lazy row removal moved it). BRISK_SXXODUALMIN=0:
         * the dual method only without dual infeasibilities */
        static int xdm = -1; if (xdm < 0) { const char *e = getenv("BRISK_SXXODUALMIN"); xdm = !(e && atoi(e) == 0); }
        xo_dual = np1 > 0 && (nd1 == 0 || (xdm && nd1 < np1)) && !getenv("BRISK_SXXOPRIMAL");
        if (S.verbose >= 1) printf("  after the pushes: %d primal infeasibilities (largest %.1e), %d dual (largest %.1e): the %s method\n", np1, mp, nd1, md, xo_dual ? "dual" : "primal");
        if (!xo_dual) st = primal_simplex(&S, pert);
    }
    if (xo && !xo_dual) ;
    /* (22 October 2026) a dual feasible basis after the pushes: the dual method's phase 2 from it, as for a given basis that
     * is dual feasible; the full dual simplex method perturbed the costs and began with its phase 1, which gave up the
     * crossover's basis (lp_beckmann_300: 26 primal infeasibilities of at most 1.2e-6 after the pushes, then a phase 1
     * from 70,105 infeasibilities and 88,000 iterations, 19 s) */
    else if (xo_dual) { S.phase = 2; st = dual_loop(&S); }
    else if (given_route == 1) st = 0;
    else if (given_route == 2) st = primal_simplex(&S, pert);
    else if (given_route == 3 && !(o && o->given_perturb)) { S.phase = 2; st = dual_loop(&S); }
    else if (method == 2 && !xo_dual) {
        st = primal_simplex(&S, pert);
    } else {
        if (pert && !xo_dual && getenv("BRISK_SXSTARTFLIP")) {     /* (on trial: off until measured on the sets) */
            const long nfl = dual_start_bounds(&S);
            if (S.verbose >= 2 && nfl) printf("  %ld columns of cost zero start at their other bound (less primal infeasibility)\n", nfl);
        }
        if (pert) perturb_costs(&S);
        st = dual_simplex(&S);
        /* (25 October 2026) an infeasibility the dual method could not prove (dual_loop's status 4): the primal method
         * decides */
        if (st == 4) st = primal_simplex(&S, pert);
        if (st == 0 || st == 2) {
            /* the perturbation and the shifts removed: the primal method removes what dual infeasibility is left */
            memcpy(S.cost, S.cost0, sizeof(double) * S.N); memset(S.shift, 0, sizeof(double) * S.N);
            S.perturbed = S.shifted = 0;
            if (st == 2) st = primal_simplex(&S, pert);
            else {
                compute_dual(&S);
                int bad = 0;
                for (int j = 0; j < S.N; j++) {
                    if (S.pos[j] >= 0) continue;
                    if (S.move[j] == 0) { if (S.lo[j] != S.up[j] && fabs(S.d[j]) > S.told) bad++; }
                    else if (S.move[j] * S.d[j] < -S.told) bad++;
                }
                if (S.verbose >= 2) { int ns = 0; for (int j = 0; j < S.N; j++) ns += S.shift[j] != 0; printf("  cost shifts: %d in iterations, %d at rebuilds, %d free, %d removed when leaving, %d left; largest %.1e; flips for dual feasibility: %d in iterations, %d at rebuilds\n", S.nsh_it, S.nsh_reb, S.nsh_free, S.nshiftback, ns, S.shmax, S.nflip_it, S.nflip_reb); }
                if (bad) { if (S.verbose >= 2) printf("  %d dual infeasibilities after the perturbation is removed: primal simplex\n", bad); st = primal_simplex(&S, pert); }
            }
        }
    }
    /* final values with the original costs and bounds; the violations on the unscaled problem: while they exceed the
     * tolerance, the scaled tolerances are tightened and the primal (dual violations) or the dual method (primal
     * violations) continues from the current basis */
    for (int round = 0; round < 5 && st == 0; round++) {
        /* the costs and bounds of the problem (shifts and perturbations gone), the nonbasic variables at them */
        memcpy(S.cost, S.cost0, sizeof(double) * S.N); memset(S.shift, 0, sizeof(double) * S.N);
        memcpy(S.lo, S.lo0, sizeof(double) * S.N); memcpy(S.up, S.up0, sizeof(double) * S.N);
        normalize_nonbasic(&S);
        if (S.F.nupd > 0 || S.nfactor == 0) refactor(&S);
        compute_primal(&S); compute_dual(&S);
        double pv, dv;
        unscaled_violations(&S, &pv, &dv);
        if (S.verbose >= 2) printf("  violations of the unscaled problem: primal %.1e, dual %.1e\n", pv, dv);
        if (pv <= S.env_target && dv <= S.env_target) break;
        if (round == 4) break;
        if (dv > S.env_target) {
            S.told = fmax(S.told * 0.5 * S.env_target / dv, 1e-12);
            st = primal_simplex(&S, pert);
        } else {
            if (pv > S.env_target) S.tolp = fmax(S.tolp * 0.5 * S.env_target / pv, 1e-12);
            st = dual_loop(&S);
        }
    }
    /* the solution on the problem as read */
    const int n = S.n, m = S.m;
    const double sg = L->maxim ? -1.0 : 1.0;
    double obj = L->c0;
    for (int j = 0; j < n; j++) { const double xj = S.x[j] * S.cscale[j]; if (xout) xout[j] = xj; obj += L->c[j] * xj; }
    double *ax = xcalloc(m + 1, sizeof(double));
    for (int j = 0; j < n; j++) { const double xj = S.x[j] * S.cscale[j]; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) ax[L->Ai[p]] += L->Ax[p] * xj; }
    if (axout) memcpy(axout, ax, sizeof(double) * m);
    /* duals: y_i = -d_{logical i} (scaled back), z_j = d_j / cscale_j; costs scaled by costscale; the sense */
    double *y = xmalloc(sizeof(double) * (m + 1)), *z = xmalloc(sizeof(double) * (n + 1));
    for (int i = 0; i < m; i++) y[i] = sg * (-S.d[n + i]) * S.rscale[i] * S.costscale;
    for (int j = 0; j < n; j++) z[j] = sg * S.d[j] / S.cscale[j] * S.costscale;
    if (yout) memcpy(yout, y, sizeof(double) * m);
    if (zout) memcpy(zout, z, sizeof(double) * n);
    /* violations on the problem as read */
    double pinf = 0, dinf = 0;
    for (int j = 0; j < n; j++) {
        const double xj = S.x[j] * S.cscale[j];
        if (L->l[j] > -HUGEB && xj < L->l[j]) pinf = fmax(pinf, L->l[j] - xj);
        if (L->u[j] < HUGEB && xj > L->u[j]) pinf = fmax(pinf, xj - L->u[j]);
    }
    for (int i = 0; i < m; i++) {
        if (L->rl[i] > -HUGEB && ax[i] < L->rl[i]) pinf = fmax(pinf, L->rl[i] - ax[i]);
        if (L->ru[i] < HUGEB && ax[i] > L->ru[i]) pinf = fmax(pinf, ax[i] - L->ru[i]);
    }
    for (int j = 0; j < n + m; j++) {
        if (S.pos[j] >= 0) continue;
        const double dj = j < n ? sg * z[j] : -sg * y[j - n];   /* for the minimization */
        if (S.move[j] == 0) { if (S.lo0[j] != S.up0[j]) dinf = fmax(dinf, fabs(dj)); }
        else if (S.move[j] * dj < 0) dinf = fmax(dinf, -S.move[j] * dj);
    }
    if (cstat) for (int j = 0; j < n; j++) cstat[j] = S.pos[j] >= 0 ? SX_BASIC : S.lo0[j] == S.up0[j] ? SX_FIXED : S.move[j] > 0 ? SX_LOWER : S.move[j] < 0 ? SX_UPPER : SX_ZERO;
    if (rstat) for (int i = 0; i < m; i++) { const int j = n + i; rstat[i] = S.pos[j] >= 0 ? SX_BASIC : S.lo0[j] == S.up0[j] ? SX_FIXED : S.move[j] > 0 ? SX_UPPER : S.move[j] < 0 ? SX_LOWER : SX_ZERO; }
    if (S.verbose >= 2) printf("  pricing: %ld by rows (%ld dense; average |rho| %.1f of %d rows), %ld by columns; average |row| %.1f; average |alpha_q| %.1f\n",
                               S.nprice_row, S.nprice_dense, S.nprice_row ? S.sum_rho / S.nprice_row : 0.0, S.m, S.nprice_col, S.iter ? S.sum_rowc / S.iter : 0.0, S.iter ? S.sum_col / S.iter : 0.0);
    if (S.verbose >= 2 && S.n_taupart + S.n_taufull > 0) printf("  the weights' solve: %ld restricted, %ld full (work lately %.3g and %.3g)\n", S.n_taupart, S.n_taufull, S.tw_part, S.tw_full);
    info->status = st;
    info->have_ray = st == 1 && S.have_ray; info->ray_sign = (int)S.raymo;
    info->iters = S.iter; info->iters_dual = S.it_dual; info->iters_primal = S.it_primal; info->iters_phase1 = S.it_ph1;
    info->nfactor = S.nfactor; info->nflips = S.nflips; info->work = S.work;
    info->obj = obj; info->pinf = pinf; info->dinf = dinf;
    info->time = now() - S.t0; info->t_factor = S.t_factor; info->t_ftran = S.t_ftran; info->t_btran = S.t_btran; info->t_price = S.t_price;
    free(ax); free(y); free(z);
    sx_free(&S);
    return st;
}

/* ---------------------------------------------------------------- sifting (wide problems)
 * When the columns far outnumber the rows, most of them are nonbasic at the optimum, and pricing every one in every
 * iteration costs more than the rest of the iteration. Sifting (column generation on the problem's own columns) solves
 * the problem restricted to a working set of columns (the others fixed at their bounds), prices all columns once
 * against the restricted optimum's duals, adds those that would improve it, and solves again from the previous basis
 * (the primal method: the added columns are nonbasic at their bounds, so the basis stays primal feasible); it ends
 * when no column prices out, which is optimality for the whole problem. */
typedef struct { double s; int j; } SiftKey;
static int cmp_siftkey(const void *a, const void *b) {
    const SiftKey *x = a, *y = b;
    if (x->s != y->s) return x->s < y->s ? -1 : 1;
    return x->j < y->j ? -1 : x->j > y->j;
}
/* ord[0..k) by score[ord[.]], ties by column: the same order on every run */
static void sift_sort(int *ord, int k, const double *score) {
    SiftKey *key = malloc(sizeof(SiftKey) * (k + 1));
    for (int a = 0; a < k; a++) { key[a].s = score[ord[a]]; key[a].j = ord[a]; }
    qsort(key, k, sizeof(SiftKey), cmp_siftkey);
    for (int a = 0; a < k; a++) ord[a] = key[a].j;
    free(key);
}
static int sift_solve(const LpProb *L, const SimplexOpts *o, double *xo, double *yo, int *cso, int *rso, SimplexInfo *info) {
    const int m = L->m, n = L->n;
    const double t0 = now();
    char *inW = calloc(n + 1, 1); int *W = malloc(sizeof(int) * (n + 1)); int nw = 0;
    double *score = calloc(n + 1, sizeof(double)); int *ord = malloc(sizeof(int) * (n + 1));
    double *cn = malloc(sizeof(double) * (n + 1));
    double cmax = 0;
    for (int j = 0; j < n; j++) {
        double s1 = 0; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) s1 += fabs(L->Ax[p]);
        cn[j] = s1 > 0 ? s1 : 1;
        if (fabs(L->c[j]) > cmax) cmax = fabs(L->c[j]);
    }
    /* the starting set: free columns, columns whose bound is not the one their cost prefers, then the cheapest per unit
     * of entries (the cost of a column against its entries' size), 2m columns in all */
    int nord = 0;
    for (int j = 0; j < n; j++) {
        const int lf = L->l[j] > -HUGEB, uf = L->u[j] < HUGEB;
        const double c = L->c[j];
        if ((!lf && !uf) || (lf && c < 0 && uf) || (!lf && uf && c > 0) || (lf && !uf && c < 0) || L->Ap[j + 1] == L->Ap[j]) { inW[j] = 1; W[nw++] = j; continue; }
        score[j] = fabs(c) / cn[j]; ord[nord++] = j;
    }
    /* the cheapest: the candidates sorted by score, ties by column (deterministic); an insertion sort here took 40 s of
     * the 44 on neos-848589 (550,539 columns) */
    {
        sift_sort(ord, nord, score);
    }
    const int target = 2 * m + 100;
    for (int k = 0; k < nord && nw < target; k++) { inW[ord[k]] = 1; W[nw++] = ord[k]; }
    /* the outside columns' values: at the bound their cost prefers */
    double *xv = malloc(sizeof(double) * (n + 1));
    for (int j = 0; j < n; j++) {
        const int lf = L->l[j] > -HUGEB, uf = L->u[j] < HUGEB;
        xv[j] = lf && (L->c[j] >= 0 || !uf) ? L->l[j] : uf ? L->u[j] : 0;
    }
    /* rows that the outside columns' values leave short of a bound: their 20 cheapest columns (cost per unit of
     * coefficient) that can move the row towards it join the set (the restricted problem is then usually feasible) */
    {
        double *act = calloc(m + 1, sizeof(double));
        for (int j = 0; j < n; j++) if (!inW[j] && xv[j] != 0) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) act[L->Ai[p]] += L->Ax[p] * xv[j];
        for (int j = 0; j < n; j++) if (inW[j]) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) act[L->Ai[p]] = act[L->Ai[p]];
        int *rows = malloc(sizeof(int) * (m + 1)); signed char *need = calloc(m + 1, 1); int nr = 0;
        for (int i = 0; i < m; i++) {
            if (L->rl[i] > -HUGEB && act[i] < L->rl[i] - 1e-9 * (1 + fabs(L->rl[i]))) { need[i] = 1; rows[nr++] = i; }
            else if (L->ru[i] < HUGEB && act[i] > L->ru[i] + 1e-9 * (1 + fabs(L->ru[i]))) { need[i] = -1; rows[nr++] = i; }
        }
        if (nr) {
            /* by rows: the candidates of each needy row, the 20 best by cost per unit */
            int *cnt = calloc(m + 1, sizeof(int)); int *best = malloc(sizeof(int) * 20 * (size_t)(m + 1)); double *bs = malloc(sizeof(double) * 20 * (size_t)(m + 1));
            for (int j = 0; j < n; j++) {
                if (inW[j]) continue;
                const int canup = L->u[j] >= HUGEB || xv[j] < L->u[j], candn = L->l[j] <= -HUGEB || xv[j] > L->l[j];
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) {
                    const int i = L->Ai[p];
                    if (!need[i]) continue;
                    const double a = L->Ax[p] * need[i];       /* > 0: increasing x_j moves the row the right way */
                    if (!((a > 0 && canup) || (a < 0 && candn))) continue;
                    const double sc = fabs(L->c[j]) / fabs(a);
                    /* insert into the row's list of 20 best */
                    int k = cnt[i] < 20 ? cnt[i]++ : 20;
                    if (k == 20) { if (sc >= bs[20 * (size_t)i + 19]) continue; k = 19; }
                    while (k > 0 && bs[20 * (size_t)i + k - 1] > sc) { best[20 * (size_t)i + k] = best[20 * (size_t)i + k - 1]; bs[20 * (size_t)i + k] = bs[20 * (size_t)i + k - 1]; k--; }
                    best[20 * (size_t)i + k] = j; bs[20 * (size_t)i + k] = sc;
                }
            }
            for (int t = 0; t < nr; t++) { const int i = rows[t]; for (int k = 0; k < cnt[i]; k++) { const int j = best[20 * (size_t)i + k]; if (!inW[j]) { inW[j] = 1; W[nw++] = j; } } }
            free(cnt); free(best); free(bs);
        }
        free(act); free(rows); free(need);
    }
    int *cs = malloc(sizeof(int) * (n + 1)), *rs = malloc(sizeof(int) * (m + 1));
    for (int j = 0; j < n; j++) cs[j] = L->l[j] > -HUGEB && xv[j] == L->l[j] ? SX_LOWER : L->u[j] < HUGEB && xv[j] == L->u[j] ? SX_UPPER : SX_ZERO;
    double *y = calloc(m + 1, sizeof(double)), *ray = calloc(m + 1, sizeof(double));
    int st = 0, round = 0; long its = 0; int havebasis = 0;
    LpProb S; memset(&S, 0, sizeof S);
    for (;; round++) {
        /* the restricted problem */
        long nz = 0; for (int k = 0; k < nw; k++) nz += L->Ap[W[k] + 1] - L->Ap[W[k]];
        S.m = m; S.n = nw;
        S.Ap = realloc(S.Ap, sizeof(int) * (nw + 1)); S.Ai = realloc(S.Ai, sizeof(int) * (nz + 1)); S.Ax = realloc(S.Ax, sizeof(double) * (nz + 1));
        S.c = realloc(S.c, sizeof(double) * (nw + 1)); S.l = realloc(S.l, sizeof(double) * (nw + 1)); S.u = realloc(S.u, sizeof(double) * (nw + 1));
        if (!S.rl) { S.rl = malloc(sizeof(double) * (m + 1)); S.ru = malloc(sizeof(double) * (m + 1)); }
        int q = 0;
        for (int k = 0; k < nw; k++) {
            const int j = W[k];
            S.Ap[k] = q;
            for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { S.Ai[q] = L->Ai[p]; S.Ax[q] = L->Ax[p]; q++; }
            S.c[k] = L->c[j]; S.l[k] = L->l[j]; S.u[k] = L->u[j];
        }
        S.Ap[nw] = q;
        for (int i = 0; i < m; i++) { S.rl[i] = L->rl[i]; S.ru[i] = L->ru[i]; }
        for (int j = 0; j < n; j++) {
            if (inW[j] || xv[j] == 0) continue;
            for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) {
                const int i = L->Ai[p];
                if (S.rl[i] > -HUGEB) S.rl[i] -= L->Ax[p] * xv[j];
                if (S.ru[i] < HUGEB) S.ru[i] -= L->Ax[p] * xv[j];
            }
        }
        S.c0 = 0; S.maxim = 0;
        SimplexOpts o2; if (o) o2 = *o; else memset(&o2, 0, sizeof o2);
        o2.nopresolve = 1; o2.ray = ray;
        int *css = malloc(sizeof(int) * (nw + 1));
        for (int k = 0; k < nw; k++) css[k] = cs[W[k]];
        if (havebasis) { o2.cstat0 = css; o2.rstat0 = rs; }
        double *xs = malloc(sizeof(double) * (nw + 1));
        SimplexInfo is;
        st = sx_solve_core(&S, &o2, xs, NULL, y, NULL, css, rs, &is);
        its += is.iters;
        info->iters_dual += is.iters_dual; info->iters_primal += is.iters_primal; info->iters_phase1 += is.iters_phase1;
        info->nfactor += is.nfactor; info->t_factor += is.t_factor; info->t_ftran += is.t_ftran; info->t_btran += is.t_btran; info->t_price += is.t_price;
        for (int k = 0; k < nw; k++) { cs[W[k]] = css[k]; xv[W[k]] = xs[k]; }
        free(css); free(xs);
        havebasis = 1;
        if (st != 0) {
            if (st == 1 && nw < n && is.have_ray) {
                /* the restricted problem infeasible: the outside columns that can help the row of the proof */
                int nv = 0;
                for (int j = 0; j < n; j++) {
                    if (inW[j]) continue;
                    double al = 0;
                    for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) al += ray[L->Ai[p]] * L->Ax[p];
                    const int lf = L->l[j] > -HUGEB, uf = L->u[j] < HUGEB;
                    const double up = (cs[j] == SX_LOWER || !lf) && (!uf || cs[j] != SX_UPPER) ? 1 : 0, dn = (cs[j] == SX_UPPER || !uf) ? 1 : 0;
                    const double v = is.ray_sign * al;
                    if ((up && v > 1e-9) || (dn && -v > 1e-9)) { score[j] = -fabs(al) / cn[j]; ord[nv++] = j; }
                }
                if (o && o->verbose >= 2) printf("  sifting round %d: restricted problem infeasible; %d columns can help\n", round, nv);
                if (nv == 0) break;      /* infeasible as a whole */
                const int kadd = m > 1000 ? m : 1000;
                if (nv > kadd) {
                    sift_sort(ord, nv, score);
                    nv = kadd;
                }
                for (int k = 0; k < nv; k++) { const int j = ord[k]; inW[j] = 1; W[nw++] = j; }
                havebasis = 0;     /* (the dual method from scratch: the basis of an infeasible solve is not a good start) */
                continue;
            }
            if (st == 1 && nw < n) { for (int j = 0; j < n; j++) if (!inW[j]) { inW[j] = 1; W[nw++] = j; } continue; }
            break;
        }
        if (nw == n) break;
        /* price the others against the duals */
        int nv = 0; const double tol = 1e-9 * (cmax > 1 ? cmax : 1);
        for (int j = 0; j < n; j++) {
            if (inW[j]) continue;
            double dj = L->c[j];
            for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) dj -= L->Ax[p] * y[L->Ai[p]];
            const int atl = cs[j] == SX_LOWER, atu = cs[j] == SX_UPPER;
            const double viol = atl ? -dj : atu ? dj : fabs(dj);
            if (viol > tol) { score[j] = -viol / cn[j]; ord[nv++] = j; }
        }
        if (o && o->verbose >= 2) printf("  sifting round %d: %d columns of %d, %ld iterations; %d price out\n", round, nw, n, is.iters, nv);
        if (nv == 0) break;
        /* the most violated (relative to the column's size) first: at most max(m, 1000) of them */
        const int kadd = m > 1000 ? m : 1000;
        if (nv > kadd) {
            sift_sort(ord, nv, score);
            nv = kadd;
        }
        for (int k = 0; k < nv; k++) { const int j = ord[k]; inW[j] = 1; W[nw++] = j; }
    }
    if (o && o->verbose >= 1) printf("sifting: %d rounds, %d of %d columns in the last, %ld iterations, %.2f s\n", round + 1, nw, n, its, now() - t0);
    info->iters = its; info->status = st;
    for (int j = 0; j < n; j++) { xo[j] = xv[j]; cso[j] = cs[j]; }
    for (int i = 0; i < m; i++) { yo[i] = y[i]; rso[i] = rs[i]; }
    free(S.Ap); free(S.Ai); free(S.Ax); free(S.c); free(S.l); free(S.u); free(S.rl); free(S.ru);
    free(inW); free(W); free(score); free(ord); free(cn); free(xv); free(cs); free(rs); free(y); free(ray);
    return st;
}

/* the presolved problem: by sifting when it is wide, else directly */
/* sifting on the dual's columns, the rows (23 October 2026; dual-form sifting, the owner's idea of 22 October): the dual
 * simplex method on a working set of rows, the others left out; the rows the solution violates join the set, their
 * logicals basic (the basis stays dual feasible) and the method continues from that basis; when no row outside the set is
 * violated, the solution and the basis (with the logicals of the rows outside basic) are optimal for the whole problem.
 * The set starts with the equality rows and the rows violated where the dual simplex method starts (each column at the
 * bound its cost points to). It applies when that start set holds at most SX_RSIFT_START of the rows and there are at
 * least SX_RSIFT_MINROWS: the work it can save lies in the rows it leaves out (uccase9: 12,047 of 49,565 at the start,
 * 22,134 at the end; the satellites problems, 65 and 83 % at the start, lost three to five times). After SX_RSIFT_CAP
 * iterations per row of the problem in all, a round of more than SX_RSIFT_RCAP iterations per row of its set (a problem
 * on the set harder than the whole: netdiversion's first round passed 60 s, the whole problem 5 s), or a restricted
 * solve that does not end optimal, every row left joins and the method finishes on the whole problem from the basis it
 * has. The rounds after the first perturb the costs as a solve from the slack basis does (given_perturb). */
#define SX_RSIFT_START 0.5
#define SX_RSIFT_MINROWS 10000
#define SX_RSIFT_CAP 1.0           /* iterations in all, per row of the problem */
#define SX_RSIFT_RCAP 2.0          /* iterations of a round, per row of its set (and 1,000 more) */
#define SX_RSIFT_FULL 0.6          /* a set with more of the rows: the whole problem */
static int rowsift_applies(const LpProb *L, char *in, int *nin_out) {
    const int m = L->m, n = L->n;
    const double minrows = getenv("BRISK_SXROWSIFTMIN") ? atof(getenv("BRISK_SXROWSIFTMIN")) : SX_RSIFT_MINROWS;   /* (experiments) */
    if (m < minrows || getenv("BRISK_SXNOROWSIFT")) return 0;
    double *x0 = malloc(sizeof(double) * (n + 1)), *r = calloc(m + 1, sizeof(double));
    for (int j = 0; j < n; j++) {
        const double c = L->maxim ? -L->c[j] : L->c[j];
        const int lf = L->l[j] > -HUGEB, uf = L->u[j] < HUGEB;
        x0[j] = c >= 0 ? (lf ? L->l[j] : uf ? L->u[j] : 0) : (uf ? L->u[j] : lf ? L->l[j] : 0);
        if (x0[j] != 0) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) r[L->Ai[p]] += L->Ax[p] * x0[j];
    }
    int nin = 0;
    for (int i = 0; i < m; i++) {
        const double lo = L->rl[i], hi = L->ru[i];
        in[i] = lo == hi || (lo > -HUGEB && r[i] < lo - 1e-9 * (1 + fabs(lo))) || (hi < HUGEB && r[i] > hi + 1e-9 * (1 + fabs(hi)));
    }
    /* the rows of a column whose cost points to a missing bound: only rows stop it, and without them the problem on the
     * set is unbounded (uccase12: the first round unbounded, then the whole problem from scratch, 1.2 -> 2.5 s) */
    for (int j = 0; j < n; j++) {
        const double c = L->maxim ? -L->c[j] : L->c[j];
        if ((c < 0 && L->u[j] >= HUGEB) || (c > 0 && L->l[j] <= -HUGEB) || (c == 0 && L->l[j] <= -HUGEB && L->u[j] >= HUGEB))
            for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) in[L->Ai[p]] = 1;
    }
    for (int i = 0; i < m; i++) nin += in[i];
    free(x0); free(r);
    *nin_out = nin;
    return nin <= SX_RSIFT_START * m;
}
/* (23 October 2026) the work budget across the rounds of row sifting: each round's solve counts its work from zero, so
 * the automatic choice's callback is given the work of the rounds before it as well, and a budget it extends holds for
 * the later rounds. Before, a round passed its own work (the interior-point method started after a fraction of the
 * intended work) and the next round's budget was the first one less the work so far, which the callback, having already
 * predicted, declined at once: the deterministic automatic choice then took the interior-point method's result early
 * (the MIP developer's first LPs, one thread: netdiversion 6.8 -> 35 s, mzzv11 11.3 -> 29.7 s) */
typedef struct { double (*cb)(void *, double); void *ctx; double off, babs; } SiftBudget;
static double sift_budget(void *v, double w) {
    SiftBudget *B = v;
    const double nb = B->cb(B->ctx, B->off + w);
    if (!(nb > 0)) return nb;
    B->babs = nb;
    return nb >= 1e299 ? nb : nb - B->off;
}
static int rowsift_solve(const LpProb *L, const SimplexOpts *o, char *in, int nin, double *xo, double *yo, int *cso, int *rso, SimplexInfo *info) {
    const int m = L->m, n = L->n, mstart = nin;
    const double t0 = now();
    /* the rows by rows, for the test of the rows outside the set */
    int *Rp = calloc(m + 2, sizeof(int)), *Rj = malloc(sizeof(int) * ((size_t)L->Ap[n] + 1)); double *Rx = malloc(sizeof(double) * ((size_t)L->Ap[n] + 1));
    for (int p = 0; p < L->Ap[n]; p++) Rp[L->Ai[p] + 2]++;
    for (int i = 0; i < m; i++) Rp[i + 2] += Rp[i + 1];
    for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int q = Rp[L->Ai[p] + 1]++; Rj[q] = j; Rx[q] = L->Ax[p]; }
    int *map = malloc(sizeof(int) * (m + 1)), *rows = malloc(sizeof(int) * (m + 1));
    int *cs = malloc(sizeof(int) * (n + 1)), *rs = malloc(sizeof(int) * (m + 1)), *rss = malloc(sizeof(int) * (m + 1));
    int *cs0 = malloc(sizeof(int) * (n + 1)), *rs0 = malloc(sizeof(int) * (m + 1));
    double *x = calloc(n + 1, sizeof(double)), *ys = calloc(m + 1, sizeof(double));
    double *vio = malloc(sizeof(double) * (m + 1)); int *ord = malloc(sizeof(int) * (m + 1));
    for (int i = 0; i < m; i++) rs[i] = SX_BASIC;
    LpProb S; memset(&S, 0, sizeof S);
    int st = 0, round = 0, havebasis = 0, all = nin == m, nvh[64], scratch = 0; long its = 0; double work = 0;
    SiftBudget sbud; memset(&sbud, 0, sizeof sbud);
    for (;; round++) {
        /* the restricted problem: the rows of the set, in their order */
        int ms = 0;
        for (int i = 0; i < m; i++) { map[i] = in[i] ? ms : -1; if (in[i]) rows[ms++] = i; }
        long nz = 0; for (int p = 0; p < L->Ap[n]; p++) nz += in[L->Ai[p]];
        S.m = ms; S.n = n;
        S.Ap = realloc(S.Ap, sizeof(int) * (n + 1)); S.Ai = realloc(S.Ai, sizeof(int) * (nz + 1)); S.Ax = realloc(S.Ax, sizeof(double) * (nz + 1));
        S.rl = realloc(S.rl, sizeof(double) * (ms + 1)); S.ru = realloc(S.ru, sizeof(double) * (ms + 1));
        S.c = L->c; S.l = L->l; S.u = L->u; S.c0 = L->c0; S.maxim = L->maxim;
        int q = 0;
        for (int j = 0; j < n; j++) { S.Ap[j] = q; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (in[L->Ai[p]]) { S.Ai[q] = map[L->Ai[p]]; S.Ax[q] = L->Ax[p]; q++; } }
        S.Ap[n] = q;
        for (int k = 0; k < ms; k++) { S.rl[k] = L->rl[rows[k]]; S.ru[k] = L->ru[rows[k]]; rss[k] = rs[rows[k]]; }
        SimplexOpts o2; if (o) o2 = *o; else memset(&o2, 0, sizeof o2);
        o2.nopresolve = 1; o2.ray = NULL;
        if (havebasis) { memcpy(cs0, cs, sizeof(int) * n); memcpy(rs0, rss, sizeof(int) * ms); o2.cstat0 = cs0; o2.rstat0 = rs0; o2.given_perturb = 1; }
        if (o2.workbudget > 0) {
            if (o2.budget_cb) {
                if (!sbud.cb) { sbud.cb = o2.budget_cb; sbud.ctx = o2.budget_ctx; sbud.babs = o2.workbudget; }
                sbud.off = work;
                o2.budget_cb = sift_budget; o2.budget_ctx = &sbud;
                o2.workbudget = sbud.babs >= 1e299 ? sbud.babs : fmax(sbud.babs - work, 1.0);
            } else o2.workbudget = fmax(o2.workbudget - work, 1.0);
        }
        const long cap = (long)(SX_RSIFT_CAP * m), rcap = (long)(SX_RSIFT_RCAP * ms) + 1000;
        if (!all) { o2.maxit = cap - its > 1 ? cap - its : 1; if (rcap < o2.maxit) o2.maxit = rcap; if (o && o->maxit > 0 && o->maxit - its < o2.maxit) o2.maxit = o->maxit - its > 1 ? o->maxit - its : 1; }
        else if (o && o->maxit > 0) o2.maxit = o->maxit - its > 1 ? o->maxit - its : 1;
        SimplexInfo is;
        st = sx_solve_core(&S, &o2, x, NULL, ys, NULL, cs, rss, &is);
        its += is.iters; work += is.work;
        info->iters_dual += is.iters_dual; info->iters_primal += is.iters_primal; info->iters_phase1 += is.iters_phase1;
        info->nfactor += is.nfactor; info->nflips += is.nflips; info->t_factor += is.t_factor; info->t_ftran += is.t_ftran; info->t_btran += is.t_btran; info->t_price += is.t_price;
        for (int k = 0; k < ms; k++) rs[rows[k]] = rss[k];
        havebasis = 1;
        if (o && o->verbose >= 2) printf("  row sifting round %d: %d of %d rows, status %d, %ld iterations, %.2f s\n", round, ms, m, st, is.iters, now() - t0);
        if (all) break;
        /* a stop, a time limit or the work budget (status 6: the automatic choice may take the interior-point method now),
         * or the caller's iteration limit: returned as they are */
        if (st == 6 || (st == 3 && o && o->maxit > 0 && its >= o->maxit)) break;
        int nv = 0;
        if (st == 0) {
            for (int i = 0; i < m; i++) {
                if (in[i]) continue;
                double r = 0; for (int t = Rp[i]; t < Rp[i + 1]; t++) r += Rx[t] * x[Rj[t]];
                const double lo = L->rl[i], hi = L->ru[i];
                const double v = fmax(lo > -HUGEB ? lo - r : 0, hi < HUGEB ? r - hi : 0) / (1 + fmax(lo > -HUGEB ? fabs(lo) : 0, hi < HUGEB ? fabs(hi) : 0));
                if (v > 1e-9) { vio[i] = -v; ord[nv++] = i; }
            }
            if (o && o->verbose >= 2) printf("    %d rows violated\n", nv);
            if (nv == 0) break;
            /* the violated rows not falling (below 0.7 of their number three rounds before, from the fifth round): the
             * set does not settle, and every round costs a new start (netdiversion: 164 to 188 a round, 107 rounds,
             * 14.6 s against 5.3 s for the whole problem); the whole problem from the basis reached */
            if (round < 64) nvh[round] = nv;
            if (round >= 5 && round < 64 && nv >= 0.7 * nvh[round - 3]) { st = 3; nv = 0; }
            const double addf = getenv("BRISK_SXROWSIFTADD") ? atof(getenv("BRISK_SXROWSIFTADD")) : 0.5;   /* (experiments; 0: all) */
            const int kadd = addf <= 0 ? m : (int)fmax(1000.0, addf * nin);
            if (nv > kadd) { sift_sort(ord, nv, vio); nv = kadd; }
            for (int k = 0; k < nv; k++) { in[ord[k]] = 1; nin++; }
            /* most rows in the set: little left to save, and the solves on it cost what the whole problem's do (atlanta-ip:
             * 63 % after two rounds, then 12.5 s against 8.0 s); from the slack basis, since the basis of a set that far from
             * the whole cost more iterations than none (atlanta-ip from it 33,167, from the slack basis 15,718) */
            if (nin > SX_RSIFT_FULL * m) { st = 3; scratch = 1; }
        }
        /* the work cap, or a restricted solve that did not end optimal (infeasible, unbounded, numerical difficulties):
         * the whole problem from this basis decides */
        if (st != 0 || its >= cap) { for (int i = 0; i < m; i++) in[i] = 1; nin = m; all = 1; if (scratch) havebasis = 0; if (o && o->verbose >= 1) printf("  row sifting: %s after %d rounds: the whole problem %s\n", scratch ? "most rows in the set" : st == 3 ? "a cap, or the violated rows not falling," : st != 0 ? "a solve on the set not optimal" : "the cap on the iterations", round + 1, scratch ? "from the start" : "from the basis reached"); }
    }
    if (o && o->verbose >= 1) printf("row sifting: %d rounds, %d of %d rows at the start, %d in the last, %ld iterations, %.2f s\n", round + 1, mstart, m, nin, its, now() - t0);
    info->iters = its; info->status = st; info->work = work; info->time = now() - t0;
    for (int j = 0; j < n; j++) { xo[j] = x[j]; cso[j] = cs[j]; }
    for (int i = 0; i < m; i++) { yo[i] = map[i] >= 0 ? ys[map[i]] : 0; rso[i] = in[i] ? rs[i] : SX_BASIC; }
    free(S.Ap); free(S.Ai); free(S.Ax); free(S.rl); free(S.ru);
    free(Rp); free(Rj); free(Rx); free(map); free(rows); free(cs); free(rs); free(rss); free(cs0); free(rs0); free(x); free(ys); free(vio); free(ord);
    return st;
}
static int reduced_solve(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir) {
    const double ratio = getenv("BRISK_SXSIFT") ? atof(getenv("BRISK_SXSIFT")) : 100;   /* (experiments) */
    const int wide = !getenv("BRISK_SXNOSIFT") && R->n >= ratio * R->m && R->n >= (getenv("BRISK_SXSIFT") ? 1000 : 20000) && o2->method != 2;
    if (wide) return sift_solve(R, o2, xr, yr, csr, rsr, ir);
    if (o2->method != 2 && !o2->cstat0 && !o2->xstart) {
        char *in = malloc(R->m + 1); int nin = 0, st = -1;
        if (rowsift_applies(R, in, &nin)) st = rowsift_solve(R, o2, in, nin, xr, yr, csr, rsr, ir);
        free(in);
        if (st >= 0) return st;
    }
    return sx_solve_core(R, o2, xr, NULL, yr, NULL, csr, rsr, ir);
}

/* a tall problem (many more rows than columns that are not column singletons) through its dual: the basis of the dual
 * has one row per column of the problem, so its factorization and solves cost a fraction of those of the problem's
 * own basis (one row per row). The dual is built (sxpre_dualize), presolved and solved; its basis is carried back by
 * complementarity, and the problem solved from that basis (optimal as it is, or finished by a few iterations). Used
 * when the presolved dual has at most half the problem's rows; if the dual's solve does not end optimal, the problem
 * is solved directly (*done stays 0). */
static int ipm_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done);
static int dual_route_applies(const LpProb *R);
static int dual_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    if (getenv("BRISK_SXNODUAL")) return 0;
    int est = 0;
    for (int j = 0; j < R->n; j++) {
        if (R->l[j] == R->u[j]) continue;
        const int boxed = R->l[j] > -HUGEB && R->u[j] < HUGEB;
        if (R->Ap[j + 1] - R->Ap[j] > 1 || boxed) est++;
    }
    const int force = getenv("BRISK_SXFORCEDUAL") != NULL;      /* (an experiment, 23 October 2026) */
    /* the factor of the two tests: 2 alone, 1 as one of the race's two simplex routes (the direct one beside it on a
     * thread of its own bounds the loss when the dual is the worse: 23 October 2026, the MIPLIB relaxations at four threads
     * 1.219 -> 1.193 s, savsched1 276 -> 27.7 s, ns1644855 78.5 -> 23.5 s; alone the dual loses badly on some problems
     * with fewer rows, uccase9 1.5 -> 56 s at one thread). BRISK_SXDUALF: the factor (an experiment) */
    const double df = getenv("BRISK_SXDUALF") ? atof(getenv("BRISK_SXDUALF")) : o2->dual_race ? 1.0 : 2.0;
    if (df * est > R->m && !force) return 0;
    const double t0 = now();
    const int verbose = o2->verbose;
    LpProb D, DR; SxDual *M = NULL; SxPre *PD = NULL; char why[160] = "";
    sxpre_dualize(R, &D, &M);
    const int ps = sxpre_presolve(&D, &DR, &PD, verbose >= 1, why, sizeof why);
    /* (the interior-point route: the dual at most a quarter of the rows, 23 October 2026; the method's iterations cost
     * about the same on either form, since it chooses its normal equations itself, and the crossover's basis must be far
     * smaller to pay for the change: opm2-z10-s4, supportcase10, supportcase22 (LPs; 11 to 40 times fewer rows) 41.5 ->
     * 18.5, 15.9 -> 7.1, 12.5 -> 3.9 s, physiciansched3-3 (LP; 3.6 times) 71 -> 187 s, 63 iterations against 37) */
    /* (24 October 2026, the night: at most an eighth. The simplex presolve's rows made equations by the cost made
     * physiciansched3-3 66,090 x 14,488, its dual 4.7 times smaller: on the dual the method stalled, 111 s, against 36.4 s
     * on the problem itself (Gurobi's time, 36.4 s); the duals that paid were 11 to 40 times smaller. The crossover's saving
     * grows with the ratio, the risk of the method on the other form does not) */
    const double dfi = o2->in_dual ? 8.0 : df;
    if (ps || (dfi * DR.m > R->m && !force) || DR.m == 0 || DR.n == 0) {
        if (verbose >= 1) printf("  the dual (%d x %d, presolved %d x %d) is not used\n", D.m, D.n, ps ? -1 : DR.m, ps ? -1 : DR.n);
        if (!ps) lp_free(&DR);
        lp_free(&D); sxpre_free(PD); sxpre_dual_free(M);
        return 0;
    }
    if (verbose >= 1) printf("  the dual problem: %d x %d, presolved %d x %d (%.2f s)\n", D.m, D.n, DR.m, DR.n, now() - t0);
    const int md = D.m, nd = D.n, mdr = DR.m, ndr = DR.n;
    int *csdr = malloc(sizeof(int) * (ndr + 1)), *rsdr = malloc(sizeof(int) * (mdr + 1));
    double *xdr = malloc(sizeof(double) * (ndr + 1)), *ydr = malloc(sizeof(double) * (mdr + 1));
    SimplexInfo id; memset(&id, 0, sizeof id);
    SimplexOpts od = *o2; if (getenv("BRISK_SXDUALMETHOD")) od.method = atoi(getenv("BRISK_SXDUALMETHOD"));   /* (experiments) */
    int st, dn2 = 0;
    if (o2->in_dual) {
        /* (23 October 2026) the interior-point route on a tall problem: the interior-point method and the crossover on its
         * dual, whose basis has as many rows as the problem has columns (opm2-z10-s4, LP: 146,325 rows, 5,958 columns;
         * the crossover on the problem itself made 128,391 dual pushes in 28 s, Gurobi's on the dual 0.07 s) */
        od.method = 3; od.orig = NULL;
        st = ipm_route(&DR, &od, xdr, ydr, csdr, rsdr, &id, &dn2);
    }
    if (!dn2) { od.method = o2->in_dual ? 0 : od.method; st = reduced_solve(&DR, &od, xdr, ydr, csdr, rsdr, &id); }
    lp_free(&DR);
    if (st != 0) {
        if (verbose >= 1) printf("  the dual ended with status %d: the problem itself is solved\n", st);
        /* (24 October 2026) its iterations and time kept for the caller's summary: stopped by the interior-point method
         * beside it, the route's work was reported as 0 iterations (irish-electricity, LP: 23,000 in 20 s) */
        ir->iters += id.iters; ir->time += id.time; ir->iters_dual += id.iters_dual; ir->iters_primal += id.iters_primal;
        free(csdr); free(rsdr); free(xdr); free(ydr); lp_free(&D); sxpre_free(PD); sxpre_dual_free(M);
        return 0;
    }
    int *csd = malloc(sizeof(int) * (nd + 1)), *rsd = malloc(sizeof(int) * (md + 1));
    double *xd = malloc(sizeof(double) * (nd + 1)), *yd = malloc(sizeof(double) * (md + 1));
    sxpre_postsolve(PD, xdr, ydr, csdr, rsdr, xd, yd, csd, rsd);
    int *cs0 = malloc(sizeof(int) * (R->n + 1)), *rs0 = malloc(sizeof(int) * (R->m + 1));
    sxpre_dual_basis(M, R, csd, rsd, cs0, rs0);
    free(csdr); free(rsdr); free(xdr); free(ydr); free(csd); free(rsd); free(xd); free(yd);
    lp_free(&D); sxpre_free(PD); sxpre_dual_free(M);
    SimplexOpts o3 = *o2; o3.cstat0 = cs0; o3.rstat0 = rs0; o3.nopresolve = 1;
    SimplexInfo i3;
    st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    if (verbose >= 1) printf("  the problem from the dual's basis: status %d, %ld iterations (%.2f s)\n", st, i3.iters, i3.time);
    free(cs0); free(rs0);
    *ir = i3;
    ir->iters = id.iters + i3.iters; ir->iters_dual = id.iters_dual + i3.iters_dual; ir->iters_primal = id.iters_primal + i3.iters_primal;
    ir->iters_phase1 = id.iters_phase1 + i3.iters_phase1; ir->nfactor = id.nfactor + i3.nfactor; ir->nflips = id.nflips + i3.nflips;
    ir->t_factor = id.t_factor + i3.t_factor; ir->t_ftran = id.t_ftran + i3.t_ftran; ir->t_btran = id.t_btran + i3.t_btran; ir->t_price = id.t_price + i3.t_price;
    ir->time = now() - t0; ir->dualized = 1; ir->work = id.work + i3.work;
    ir->ipm_iters = id.ipm_iters; ir->t_ipm = id.t_ipm; ir->pushes = id.pushes + i3.pushes;
    *done = 1;
    return st;
}

/* (23 October 2026) the simplex method of the race (and of the automatic method) on the problem presolved once more with
 * the bounded substitution of columns without cost (a column with bounds the rows do not imply, no cost, substituted
 * through an equality row of at most 4 new entries; the row keeps its bounds as a range: Gurobi's aggregation). It helps
 * the simplex method (satellites2-40, LP: 14.9 -> 2.3 s, mzzv11 3.3 -> 0.7 s) and hurts the interior-point method on the
 * same problem (s250r10 13.9 -> 57.8 s), which therefore keeps the problem without it (ISSUES 428). Its basis is carried
 * back by the postsolve and the problem solved from it. Only when the substitution removes at least 5 % of the
 * columns (the satellites lose 31 %; pds-20, LP, 1.5 %, its fill costing more than it saves: 1.74 -> 2.00 s with 0.5 %);
 * if the solve does not end optimal, the problem is solved directly (*done stays 0). BRISK_SXNOAGGBR=1: off */
static int aggb_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    if (getenv("BRISK_SXNOAGGBR") || o2->cstat0 || o2->xstart || R->m < 100) return 0;
    {   /* (23 October 2026, the evening) a scan before the presolve: an upper bound on the substitutions, the columns without
         * cost, not fixed, with two entries or more, one of them in an equality row short enough for at most 4 new entries;
         * the presolve only when they can reach the threshold (its run was the route's cost on the problems it skipped:
         * lp_transport_1000, LP, 2.89 -> 3.69 s) */
        int *rlen = calloc((size_t)R->m + 1, sizeof(int)); long cand = 0;
        for (int j = 0; j < R->n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) rlen[R->Ai[p]]++;
        for (int j = 0; j < R->n; j++) {
            const int cl = R->Ap[j + 1] - R->Ap[j];
            if (R->c[j] != 0.0 || cl < 2 || R->l[j] == R->u[j]) continue;
            for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int i = R->Ai[p];
                if (R->rl[i] == R->ru[i] && (long)(cl - 1) * (rlen[i] - 2) <= 4) { cand++; break; } }
        }
        free(rlen);
        if (cand < fmax(10.0, 0.05 * R->n)) return 0;
    }
    const double t0 = now();
    LpProb R2; SxPre *P2 = NULL; char why[160] = "";
    if (sxpre_presolve_ex(R, &R2, &P2, 0, 1, why, sizeof why)) return 0;
    if (R->n - R2.n < fmax(10.0, 0.05 * R->n)) { lp_free(&R2); sxpre_free(P2); return 0; }
    const int verbose = o2->verbose;
    if (verbose >= 1) printf("  the bounded substitution for the simplex method: %d x %d -> %d x %d (%.2f s)\n", R->m, R->n, R2.m, R2.n, now() - t0);
    const int m2 = R2.m, n2 = R2.n;
    int *cs2 = malloc(sizeof(int) * (n2 + 1)), *rs2 = malloc(sizeof(int) * (m2 + 1));
    double *x2 = malloc(sizeof(double) * (n2 + 1)), *y2 = malloc(sizeof(double) * (m2 + 1));
    SimplexInfo i2; memset(&i2, 0, sizeof i2);
    int st = reduced_solve(&R2, o2, x2, y2, cs2, rs2, &i2);
    lp_free(&R2);
    if (st != 0) {
        if (verbose >= 1 && !(o2->stop && *o2->stop)) printf("  the problem with the substitution ended with status %d: the problem itself is solved\n", st);
        free(cs2); free(rs2); free(x2); free(y2); sxpre_free(P2);
        return 0;
    }
    int *cs0 = malloc(sizeof(int) * (R->n + 1)), *rs0 = malloc(sizeof(int) * (R->m + 1));
    double *x0 = malloc(sizeof(double) * (R->n + 1)), *y0 = malloc(sizeof(double) * (R->m + 1));
    sxpre_postsolve(P2, x2, y2, cs2, rs2, x0, y0, cs0, rs0);
    free(cs2); free(rs2); free(x2); free(y2); free(x0); free(y0); sxpre_free(P2);
    SimplexOpts o3 = *o2; o3.cstat0 = cs0; o3.rstat0 = rs0; o3.nopresolve = 1;
    SimplexInfo i3;
    st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    if (verbose >= 1) printf("  the problem from the substituted problem's basis: status %d, %ld iterations (%.2f s)\n", st, i3.iters, i3.time);
    free(cs0); free(rs0);
    *ir = i3;
    ir->iters = i2.iters + i3.iters; ir->iters_dual = i2.iters_dual + i3.iters_dual; ir->iters_primal = i2.iters_primal + i3.iters_primal;
    ir->iters_phase1 = i2.iters_phase1 + i3.iters_phase1; ir->nfactor = i2.nfactor + i3.nfactor; ir->nflips = i2.nflips + i3.nflips;
    ir->t_factor = i2.t_factor + i3.t_factor; ir->t_ftran = i2.t_ftran + i3.t_ftran; ir->t_btran = i2.t_btran + i3.t_btran; ir->t_price = i2.t_price + i3.t_price;
    ir->time = now() - t0; ir->work = i2.work + i3.work;
    *done = 1;
    return st;
}

/* a problem with symmetry: solved on the classes of its coarsest equitable partition (sxpre_symmetry; the optimal value
 * is the same), the solution carried back (each column at its class' value: an optimal point, in general not basic) and
 * turned into an optimal basis by the crossover and the primal method. If the problem on the classes does not end
 * optimal, the problem itself is solved (*done stays 0). */
static int sym_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    const double t0 = now();
    LpProb Q; SxSym *M = NULL;
    if (!sxpre_symmetry(R, &Q, &M, o2->verbose >= 1, 0)) return 0;
    const double tsym = now() - t0;
    SimplexOpts oq = *o2; oq.nosymmetry = 1; oq.nopresolve = 0;
    double *xq = malloc(sizeof(double) * (Q.n + 1)), *yq = malloc(sizeof(double) * (Q.m + 1));
    SimplexInfo iq;
    const int mq = Q.m, nq = Q.n;
    int st = simplex_solve(&Q, &oq, xq, NULL, yq, NULL, NULL, NULL, &iq);
    lp_free(&Q);
    if (o2->verbose >= 1) printf("  the problem on the classes (%d x %d, %.2f s to find them): status %d, %ld iterations (%.2f s)\n", mq, nq, tsym, st, iq.iters, iq.time);
    if (st != 0) { free(xq); free(yq); sxpre_sym_free(M); return 0; }
    double *x = malloc(sizeof(double) * (R->n + 1)), *y = malloc(sizeof(double) * (R->m + 1));
    sxpre_sym_lift(M, xq, yq, x, y);
    free(xq); free(yq); sxpre_sym_free(M);
    SimplexOpts o3 = *o2; o3.xstart = x; o3.ystart = y; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL;
    SimplexInfo i3;
    st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    free(x); free(y);
    if (o2->verbose >= 1) printf("  the problem from the classes' solution: status %d, %ld pushes, %ld iterations (%.2f s)\n", st, i3.pushes, i3.iters, i3.time);
    *ir = i3;
    ir->iters = iq.iters + i3.iters; ir->iters_dual = iq.iters_dual + i3.iters_dual; ir->iters_primal = iq.iters_primal + i3.iters_primal;
    ir->iters_phase1 = iq.iters_phase1 + i3.iters_phase1; ir->nfactor = iq.nfactor + i3.nfactor; ir->nflips = iq.nflips + i3.nflips;
    ir->t_factor = iq.t_factor + i3.t_factor; ir->t_ftran = iq.t_ftran + i3.t_ftran; ir->t_btran = iq.t_btran + i3.t_btran; ir->t_price = iq.t_price + i3.t_price;
    ir->time = now() - t0; ir->sym_m = mq; ir->sym_n = nq; ir->work = iq.work + i3.work;
    *done = 1;
    return st;
}

#define SX_IPM_NEAR_INF 1e-6        /* a stopped interior point the crossover takes: relative infeasibilities and gap (as */
#define SX_IPM_NEAR_GAP 1e-6        /* lpsolve.c reports them) within a hundred times its tolerance 1e-8 (triptim1 stalled at
                                       2e-7: 9,173 pushes and 7,483 iterations, 7.7 s; irish-electricity at a gap of 5e-4:
                                       6,365 rows infeasible after the pushes, the primal method over 250 s) */
/* the interior-point method (lpsolve.c, called as it is) on the presolved problem, then the crossover to an optimal
 * basis (27.10): what the large problems the simplex method iterates long on need. If the interior-point method does
 * not end optimal, the simplex method solves the problem (*done stays 0). */
/* the problem as read and the presolve's maps (SimplexOpts.orig) */
/* the problem as read, for the crossover route's second interior-point attempt (simplex_solve sets it): that attempt's
 * point is carried to a basis on the problem as read itself (23 October 2026: restricted to the presolved problem it was
 * not optimal there: the presolve's reductions by the costs (columns fixed by them, dominated columns) choose one optimal
 * point among several, an interior point is not that point, and the duals of rows the presolve merged or moved costs into
 * do not carry over; an optimal basic point of 13 Netlib LPs restricted so had dual violations up to 3e4, rail02's interior
 * point 23,349 primal infeasibilities after the pushes). out receives that basis: simplex_solve then skips the postsolve. */
struct SxOut { double *x, *y; int *cs, *rs; int have; };
struct SxOrig { const LpProb *L; struct SxOut *out; };
static void sxout_init(struct SxOut *O, const LpProb *L) {
    O->x = malloc(sizeof(double) * ((size_t)L->n + 1)); O->y = malloc(sizeof(double) * ((size_t)L->m + 1));
    O->cs = malloc(sizeof(int) * ((size_t)L->n + 1)); O->rs = malloc(sizeof(int) * ((size_t)L->m + 1)); O->have = 0;
    if (!O->x || !O->y || !O->cs || !O->rs) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); }
}
static void sxout_free(struct SxOut *O) { free(O->x); free(O->y); free(O->cs); free(O->rs); memset(O, 0, sizeof *O); }
static void sxout_copy(struct SxOut *D, const struct SxOut *S, const LpProb *L) {
    memcpy(D->x, S->x, sizeof(double) * L->n); memcpy(D->y, S->y, sizeof(double) * L->m);
    memcpy(D->cs, S->cs, sizeof(int) * L->n); memcpy(D->rs, S->rs, sizeof(int) * L->m); D->have = S->have;
}

/* the LP interior-point method on L (after lp_to_std's presolve level pre): its status; x (L->n) and y (L->m, NULL
 * allowed) get the point when it ended optimal or stopped near the optimum (*near) */
/* the method's cost of an iteration on L's standard form (lpipm_iter_flops: its own analysis) */
static double ipm_std_flops(const LpProb *L, double budget) {
    LpProb Lc = *L; SedumiProb P; LpMap M; double *ub = NULL;
    lp_to_std(&Lc, 0, 0, &P, &M, &ub);
    const double f = M.status || P.n == 0 ? 1e300 : lpipm_iter_flops(P.m, P.n, P.Ap, P.Ai, P.Ax, budget);
    sedumi_free(&P); lp_map_free(&M); free(ub);
    return f;
}
#define SX_ASREAD_FLOPS 1e10    /* (the presolved problem's iteration above which the problem as read is analyzed too) */
typedef struct { const LpProb *L; int declined; double fr, fl; int verbose; double lbr, lbl; int mr; } IpmAsRead;
static double ipm_form_lb(const LpProb *P);
static int ipm_asread_cb(void *v, double flops) {
    IpmAsRead *A = v;
    A->fr = flops;
    if (!(flops > SX_ASREAD_FLOPS)) return 0;
    /* (the analysis of the problem as read only when the presolve took out at least a quarter of its rows (fill from its
     * substitutions needs eliminations; rail02: 95,791 rows as read, 50,160 presolved) and the cheap bound does not make
     * the problem as read the larger by half (physiciansched3-3, LP, with its long columns: 7.4e7 against 2.9e7; that
     * analysis took 73 s, for 4.08e11 flops against 6.19e10; rail02 3.5e7 against 2.9e7)) */
    if (!(A->L->m >= 1.3 * A->mr && A->lbl <= 1.5 * A->lbr)) { if (A->verbose >= 1) printf("  the interior-point method's iteration: %.3g flops on the presolved problem; the problem as read not analyzed (bounds %.3g against %.3g)\n", flops, A->lbl, A->lbr); return 0; }
    A->fl = ipm_std_flops(A->L, 0);                /* (no cap: with one, at half or at all of the presolved iteration's cost, rail02's analysis
                                                      * of 3.35e10 flops stopped as if dearer; the two conditions above bound the cases) */
    A->declined = A->fl <= 0.5 * flops;
    if (A->verbose >= 1) printf("  the interior-point method's iteration: %.3g flops on the presolved problem, %.3g as read%s\n", flops, A->fl, A->declined ? ": the problem as read" : "");
    return A->declined;
}
void *sx_asread_new(const LpProb *asread, const LpProb *presolved, int verbose) {
    IpmAsRead *A = calloc(1, sizeof *A);
    if (!A) return NULL;
    A->L = asread; A->verbose = verbose; A->mr = presolved->m;
    A->lbr = ipm_form_lb(presolved); A->lbl = ipm_form_lb(asread);
    return A;
}
int sx_asread_cb(void *ctx, double flops) { return ipm_asread_cb(ctx, flops); }
int sx_asread_declined(const void *ctx) { return ctx && ((const IpmAsRead *)ctx)->declined; }
void sx_asread_free(void *ctx) { free(ctx); }
static int ipm_point(const LpProb *L, int pre, int pdreg, const SimplexOpts *o2, double t0, double *x, double *y, LpIpmInfo *ii, int *near,
                     int (*analyzed)(void *, double), void *actx) {
    LpProb Rc = *L;             /* (lp_to_std reads it only) */
    SedumiProb P; LpMap M; double *ub = NULL;
    *near = 0; memset(ii, 0, sizeof *ii);
    lp_to_std(&Rc, pre, 0, &P, &M, &ub);
    if (M.status) { sedumi_free(&P); lp_map_free(&M); free(ub); snprintf(ii->why, sizeof ii->why, "the standard form failed"); return 4; }
    LpIpmOpts io; memset(&io, 0, sizeof io);
    io.nopdreg = !pdreg;      /* (22 October 2026: the crossover wants the unregularized end point, LpIpmOpts; the second attempt allows it) */
    if (!getenv("BRISK_LPNOOBJOFF") && P.n > 0) {     /* (22 October 2026: the objective's constant, for the method's gap; lpsolve.h) */
        double *z0 = calloc((size_t)P.n + 1, sizeof(double)), *x0 = calloc((size_t)Rc.n + 1, sizeof(double));
        const double v = lp_recover(&Rc, &M, z0, x0); io.objoff = Rc.maxim ? -v : v; free(z0); free(x0);
    }
    io.tol = getenv("BRISK_SXIPMTOL") ? atof(getenv("BRISK_SXIPMTOL")) : 1e-8; io.kcorr = o2->ipm_kcorr > 0 ? o2->ipm_kcorr - 1 : -1; io.verbose = o2->verbose >= 2 ? 1 : 0;   /* (the tolerance's switch: a test, 21 October 2026) */
    if (o2->timelimit > 0) io.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3);
    io.stop = o2->stop;
    io.analyzed = analyzed; io.analyzed_ctx = actx;
    /* (an experiment, 23 October 2026: BRISK_SXXOPERT=eps) the costs of the nonnegative columns raised by eps times a
     * deterministic factor in [0.5, 1] times the mean cost: on a problem with many optimal points the method converges
     * to the centre of the optimal face, where most variables are inside their bounds and each is pushed by the
     * crossover (neos-631710, LP: 166,200 pushes); the perturbed problem has a single optimal vertex near which the
     * point ends. The simplex method after the pushes works with the costs as they are. */
    static double xopert = -1; if (xopert < 0) { const char *e = getenv("BRISK_SXXOPERT"); xopert = e ? atof(e) : 0.0; }
    if (xopert > 0 && P.nl > 0) {
        double cs = 0; int nc = 0;
        for (int j = P.nf; j < P.nf + P.nl; j++) if (P.c[j] != 0) { cs += fabs(P.c[j]); nc++; }
        const double cm = nc ? cs / nc : 1.0;
        for (int j = P.nf; j < P.nf + P.nl; j++) {
            unsigned long long h = (unsigned long long)(j + 1) * 0x9e3779b97f4a7c15ULL; h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 29;
            P.c[j] += xopert * cm * (0.5 + 0.5 * (double)(h >> 11) / 9007199254740992.0);
        }
    }
    double *xs = calloc((size_t)P.n + 1, sizeof(double)), *ys = calloc((size_t)P.m + 1, sizeof(double)), *zs = calloc((size_t)P.n + 1, sizeof(double));
    const double t1 = now();
    const int ist = P.n > 0 ? lpipm_solve(P.m, P.n, P.nf, P.Ap, P.Ai, P.Ax, P.b, P.c, ub, &io, xs, ys, zs, ii) : 0;
    if (o2->verbose >= 1) printf("  the interior-point method%s: status %d, %d iterations, %.2f s (primal %.1e, dual %.1e, gap %.1e; %d x %d, %ld entries, factor %ld, %d dense columns)\n",
                                 o2->orig && L == o2->orig->L ? " on the problem as read" : "", ist, ii->iters, now() - t1, ii->pinf, ii->dinf, ii->gap, P.m, P.n, (long)P.Ap[P.n], ii->lnz, ii->ndense);
    /* a point it stopped at without reaching its tolerance (stalled, or out of iterations) but nearly optimal is still the
     * crossover's: what it lacks the simplex method after the pushes removes, from a basis next to the optimal one
     * (triptim1: stalled at 61 iterations with errors 1e-7, then 142,000 simplex iterations from the start) */
    *near = (ist == 3 || ist == 4) && ii->pinf <= SX_IPM_NEAR_INF && ii->dinf <= SX_IPM_NEAR_INF && ii->gap <= SX_IPM_NEAR_GAP;
    if (ist != 0 && *near && o2->verbose >= 1) printf("  the interior-point method stopped near the optimum (%s): its point goes to the crossover\n", ii->why);
    if (ist == 0 || *near) {
        lp_recover(L, &M, xs, x);
        if (y) for (int i = 0; i < L->m; i++) y[i] = M.rpos[i] >= 0 ? M.rsgn[i] * ys[M.rpos[i]] : 0;
    }
    free(xs); free(ys); free(zs); sedumi_free(&P); lp_map_free(&M); free(ub);
    return ist;
}

static int ipm_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    if (!o2->in_dual && !getenv("BRISK_SXNOIPMDUAL") && dual_route_applies(R)) {
        /* a tall problem: the interior-point method and the crossover on its dual (dual_route; in_dual keeps the dual's
         * own route from dualizing again) */
        SimplexOpts od = *o2; od.in_dual = 1;
        int dn = 0;
        const int st = dual_route(R, &od, xr, yr, csr, rsr, ir, &dn);
        if (dn) { *done = 1; return st; }
    }
    const double t0 = now();
    double *x = calloc((size_t)R->n + 1, sizeof(double)), *y = calloc((size_t)R->m + 1, sizeof(double));
    LpIpmInfo ii; int near;
    /* (23 October 2026) the problem as read first when its normal equations cost clearly less than the presolved
     * problem's: the presolve's substitutions can fill them (rail02, LP: 50,160 rows after it, a factor of 50.5 million
     * entries; 95,791 rows as read, 28.2 million; the method stalled on the first after 64 iterations, 136 s, and converged
     * on the second in 52, 50 s). The patterns' lower bounds first, the orderings when the presolved one is the larger by
     * a third or large; the problem as read when its cost is at most half. (see below: off by default since the evening) */
    /* (23 October 2026, the evening) the problem as read instead, when the method's own analysis of the presolved
     * problem finds an expensive iteration (above 1e10 flops: a factorization of seconds) and the same analysis of the
     * problem as read costs at most half: the presolve's substitutions can fill the normal equations. rail02 (LP): 1.03e11
     * flops a factorization presolved (50.5 million entries; the method stalled there after 65 iterations, 101 s), 3.35e10
     * as read (28.2 million; 54 iterations, 39 s). Estimates before the method (AMD's counts) did not separate the two and
     * cost 1-1.5 s on smaller problems; this costs one more analysis, on the expensive ones only. BRISK_SXIPMASREAD=0: off */
    IpmAsRead ar = { o2->orig && !o2->in_dual ? o2->orig->L : NULL, 0, 0, 0, o2->verbose, 0, 0, R->m };
    const int useas = ar.L && !(getenv("BRISK_SXIPMASREAD") && atoi(getenv("BRISK_SXIPMASREAD")) == 0);
    if (useas) { ar.lbr = ipm_form_lb(R); ar.lbl = ipm_form_lb(ar.L); }
    int ist = ipm_point(R, 0, getenv("BRISK_SXIPMPDREG") ? 1 : 0, o2, t0, x, y, &ii, &near, useas ? ipm_asread_cb : NULL, &ar);   /* (the switch: an experiment, 23 October 2026) */
    const int asread = ar.declined;
    if (asread) { ist = 4; ii.iters = 0; snprintf(ii.why, sizeof ii.why, "the problem as read is cheaper"); near = 0; }
    int iters = ii.iters;
    const int stopped = ist == 6 || (o2->stop && *o2->stop);
    if (ist != 0 && !near && !asread && o2->verbose >= 1) printf("  the interior-point method did not end optimal: %s\n", ii.why);
    /* a second attempt on the problem as read, without a presolve (22 October 2026: on triptim1 the method stalls after
     * 53 iterations on the simplex method's presolved problem, and after the standard presolve as well, and converges in
     * 26 on the problem as read; the standard presolve would also lose the duals of the rows it removes, which the
     * crossover needs); its point, restricted to the columns and rows the simplex method's presolve kept, goes to the
     * crossover on the problem as read (struct SxOrig). This attempt allows the regularization at a breakdown of the normal equations (23 October 2026: rail02
     * stalls at a gap of 4e-3 without it in both forms and converges in 52 iterations with it, as the LP route solves
     * it); the first attempt does without, for the cleaner end point (dfl001: 3 cleanup iterations against 17,075) */
    double *xo = NULL, *yo = NULL;      /* (the point of the second attempt: on the problem as read) */
    if (ist != 0 && !near && !stopped && o2->orig) {
        const LpProb *L = o2->orig->L;
        xo = calloc((size_t)L->n + 1, sizeof(double)); yo = calloc((size_t)L->m + 1, sizeof(double));
        LpIpmInfo i2; int near2;
        const int ist2 = ipm_point(L, 0, 1, o2, t0, xo, yo, &i2, &near2, NULL, NULL);
        iters += i2.iters;
        if (ist2 == 0 || near2) { ist = ist2; near = near2; }
        else {
            if (o2->verbose >= 1) printf("  the interior-point method on the problem as read did not end optimal: %s\n", i2.why);
            free(xo); free(yo); xo = yo = NULL;
        }
    }
    if (xo) {
        /* the crossover on the problem as read; simplex_solve finishes from the basis in out */
        free(x); free(y);
        const double tipm = now() - t0;
        SimplexOpts o3 = *o2; o3.xstart = xo; o3.ystart = yo; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.method = 0; o3.orig = NULL;
        SimplexInfo i3;
        struct SxOut *O = o2->orig->out;
        const int st = sx_solve_core(o2->orig->L, &o3, O->x, NULL, O->y, NULL, O->cs, O->rs, &i3);
        O->have = 1;
        free(xo); free(yo);
        if (o2->verbose >= 1) printf("  the crossover and the simplex method from the interior point (on the problem as read): status %d, %ld pushes, %ld iterations (%.2f s)\n", st, i3.pushes, i3.iters, i3.time);
        *ir = i3; ir->time = now() - t0; ir->ipm_iters = iters; ir->t_ipm = tipm;
        *done = 1;
        return st;
    }
    if (ist != 0 && !near) {
        free(x); free(y); x = y = NULL;
        /* the SOCP solver's interior point instead, as the LP route takes it when this method gives up (22 October
         * 2026: lp_densecols_small, whose dense columns' correction lost its accuracy, went to the dual simplex method
         * for 108 s against 0.7 s of the SOCP solver) */
        if (ist == 6 || (o2->stop && *o2->stop)) return 0;
        LpProb Rc = *R;
        SedumiProb P2; LpMap M2;
        lp_to_sedumi(&Rc, 0, 0, &P2, &M2);
        if (M2.status || P2.ns != 0 || P2.n == 0) { sedumi_free(&P2); lp_map_free(&M2); return 0; }
        SocpProb Q = { P2.m, P2.n, P2.Ap, P2.Ai, P2.Ax, P2.b, P2.c, P2.nf, P2.nl, P2.nq, P2.q, P2.nr, P2.r };
        SocpOpts so; socp_default_opts(&so); so.tol = 1e-8; so.verbose = 0;
        if (o2->timelimit > 0) so.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3);
        volatile int *stop0 = socp_stop; if (o2->stop) socp_stop = (volatile int *)o2->stop;      /* (never NULL: the solver reads it) */
        SocpRes SR; memset(&SR, 0, sizeof SR);
        const int sst = socp_solve(&Q, &so, &SR);
        socp_stop = stop0;
        if (o2->verbose >= 1) printf("  the SOCP solver: status %d, %d iterations (%.2f s)\n", sst, SR.iters, SR.time);
        if (sst == SOCP_OPTIMAL) {
            x = calloc((size_t)R->n + 1, sizeof(double)); y = calloc((size_t)R->m + 1, sizeof(double));
            lp_recover(R, &M2, SR.x, x);
            for (int i = 0; i < R->m; i++) y[i] = M2.rpos[i] >= 0 ? M2.rsgn[i] * SR.y[M2.rpos[i]] : 0;
        }
        socp_result_free(&SR); sedumi_free(&P2); lp_map_free(&M2);
        if (!x) return 0;
    }
    const double tipm = now() - t0;
    SimplexOpts o3 = *o2; o3.xstart = x; o3.ystart = y; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.method = 0;
    SimplexInfo i3;
    const int st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    free(x); free(y);
    if (o2->verbose >= 1) printf("  the crossover and the simplex method from the interior point: status %d, %ld pushes, %ld iterations (%.2f s)\n", st, i3.pushes, i3.iters, i3.time);
    *ir = i3; ir->time = now() - t0; ir->ipm_iters = iters; ir->t_ipm = tipm;
    *done = 1;
    return st;
}

/* the automatic choice (method 0): the simplex method first, with a budget of work; a problem it does not finish
 * within SX_AUTO_B0 entries touched per entry of the matrix gets a prediction of the interior-point method's work, and
 * the simplex method continues up to that work (the rule "rent until the rent paid equals the price": when the
 * prediction holds, the total is at most twice that of the better choice made in hindsight); if it does not finish
 * there, the interior-point method and the crossover solve the problem (from scratch; the simplex method from its last
 * basis if they fail). The prediction follows BRISK's interior-point method (lpsolve.c) as it is: the normal equations
 * of the standard form (the dense columns apart), or those of its dual form when they cost at most half (the dual's
 * rows are the columns that are not slacks: a tall problem has few); each ordered by minimum degree (AMD), whose
 * counts give one iteration's flops as that method counts them, times SX_AUTO_IT iterations, at SX_AUTO_FLOPS flops
 * for an entry touched by the simplex method. Before any ordering, a lower bound (the assembly alone, the sum of
 * len (len + 1) / 2 over each form's columns) is compared to the work. The decision depends on the work counts only:
 * the same at any thread count. */
#define SX_AUTO_B0 200.0
#define SX_AUTO_FLOPS 30.0          /* (fitted: the prediction at 15 was 1.9 times the time of the interior-point method and the crossover, geometric mean of 16 MIPLIB relaxations) */
#define SX_AUTO_IT 30.0
#define SX_AUTO_SETUP 2e8           /* the interior-point method's setup, in entries touched (about 1.2 s of the simplex
                                       method): its analyses run the sparse Cholesky's own orderings after AMD, 0.3-2 s on
                                       LPs whose factorization takes milliseconds (ISSUES 409); without this term the
                                       method switched on 21 of 168 small MIPLIB relaxations that the simplex method
                                       solves in 0.06-0.7 s, and lost (shifted geometric mean 0.550 -> 0.586 s) */
#define SX_AUTO_PATCAP 5e7          /* the largest pattern (entries of M M') ordered for the prediction */
/* one form's normal equations M M': M has nr rows and the columns Cp, Ci (nc), and ns singleton columns more (the
 * slacks: the diagonal only). The dense columns as lpsolve.c's ne_setup takes them: longer than max(50, 10 x the
 * average, nr / 10), when nr >= 500 and there are 1 to 400 of them and fewer than a quarter of the columns. */
typedef struct { int nr, nc, ns, nd; int *Cp, *Ci; char *isd; double lb, pb, lnz, cost; } NeForm;   /* (pb: the bound on the pattern's entries) */
static void neform_init(NeForm *F) {
    const int nc = F->nc, ntot = nc + F->ns; const double nz = (double)F->Cp[nc] + F->ns;
    F->isd = calloc((size_t)nc + 1, 1); F->nd = 0;
    if (F->nr >= 500) {
        const double thr = fmax(fmax(50.0, 10.0 * nz / fmax(ntot, 1)), 0.1 * F->nr);
        int nd = 0; for (int j = 0; j < nc; j++) if (F->Cp[j + 1] - F->Cp[j] > thr) nd++;
        if (nd > 0 && nd <= 400 && nd < ntot / 4) { for (int j = 0; j < nc; j++) if (F->Cp[j + 1] - F->Cp[j] > thr) F->isd[j] = 1; F->nd = nd; }
    }
    F->lb = 0;
    for (int j = 0; j < nc; j++) if (!F->isd[j]) { const double l = F->Cp[j + 1] - F->Cp[j]; F->lb += 0.5 * l * (l + 1); }
    F->pb = F->lb;
    F->lb += 40.0 * nz;             /* the products with A of the ten solves (below) */
}
/* one iteration's flops (lpsolve.c's ne_iter_cost): the factorization, the dense columns' solves and their Schur
 * complement, about ten solves and products with A. cost = 0: the pattern is too large to order. */
static void neform_cost(NeForm *F) {
    const int nr = F->nr, nc = F->nc; const int *Cp = F->Cp, *Ci = F->Ci;
    F->cost = 0;
    /* (23 October 2026) the cap on the pattern's own bound: before, the bound on the iteration's flops, whose term for
     * the solves' products (40 per entry of the matrix) alone passed it on rail02 (LP; 614,000 entries), so the
     * prediction and the interior-point route's choice of form gave up on it */
    static int oldcap = -1; if (oldcap < 0) oldcap = getenv("BRISK_SXPATCAPOLD") && atoi(getenv("BRISK_SXPATCAPOLD")) == 1;
    if (2.0 * (oldcap ? F->lb : F->pb) > SX_AUTO_PATCAP) return;
    int *tp = calloc((size_t)nr + 2, sizeof(int)), *tj = malloc(sizeof(int) * ((size_t)Cp[nc] + 1));
    for (int j = 0; j < nc; j++) if (!F->isd[j]) for (int p = Cp[j]; p < Cp[j + 1]; p++) tp[Ci[p] + 1]++;
    for (int i = 0; i < nr; i++) tp[i + 1] += tp[i];
    { int *f = malloc(sizeof(int) * ((size_t)nr + 1)); memcpy(f, tp, sizeof(int) * nr);
      for (int j = 0; j < nc; j++) if (!F->isd[j]) for (int p = Cp[j]; p < Cp[j + 1]; p++) tj[f[Ci[p]]++] = j;
      free(f); }
    /* each row's neighbours through its columns: the pattern of M M' (both triangles, the diagonal included) */
    int *mark = malloc(sizeof(int) * ((size_t)nr + 1)), *ps = malloc(sizeof(int) * ((size_t)nr + 1));
    int *pi = malloc(sizeof(int) * ((size_t)(2.0 * F->pb) + nr + 1));
    for (int i = 0; i < nr; i++) mark[i] = -1;
    long e = 0;
    for (int i = 0; i < nr; i++) {
        ps[i] = (int)e; mark[i] = i; pi[e++] = i;
        for (int t = tp[i]; t < tp[i + 1]; t++) { const int j = tj[t]; for (int p = Cp[j]; p < Cp[j + 1]; p++) { const int k = Ci[p]; if (mark[k] != i) { mark[k] = i; pi[e++] = k; } } }
    }
    ps[nr] = (int)e;
    int *perm = malloc(sizeof(int) * ((size_t)nr + 1));
    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_defaults(Control);
    if (amd_order(nr, ps, pi, perm, Control, Info) >= 0) {
        const double lnz = Info[AMD_LNZ] + nr, fl = Info[AMD_NDIV] + 2.0 * Info[AMD_NMULTSUBS_LDL], nd = F->nd;
        const double nnz = (double)Cp[nc] + F->ns;
        F->lnz = lnz;
        F->cost = fl + nd * (4.0 * lnz + 2.0 * nd * nr) + 10.0 * (4.0 * lnz + 4.0 * nnz + 4.0 * nd * nr);
    }
    free(tp); free(tj); free(mark); free(ps); free(pi); free(perm);
}
/* the interior-point method and the crossover on a thread of their own, while the simplex method continues (SX_RACE) */
typedef struct {
    const LpProb *R; SimplexOpts o; int nt;
    double *x, *y; int *cs, *rs; SimplexInfo info; int st, done;
    volatile int stop;
    /* the race by the clock: when the job ends optimal it sets the simplex method's stop flag (stop_other, in the caller's
     * frame: written under mu, never once the caller has left the job behind). A job left behind works on its own copy
     * of the problem and frees itself. */
    LpProb own; int race; volatile int stop_s;
    LpProb own_orig; struct SxOut own_out; struct SxOrig own_o; int has_orig;   /* (the race) copies of SimplexOpts.orig:
                                                                 an abandoned job may outlive the caller's problem */
    volatile int *stop_other;
    volatile int *threads_other; int nt_all;   /* the simplex method's thread count: all of them back when this job fails */
    volatile int *stop_other2;  /* (the race of the two simplex routes) the direct route's stop flag */
#ifdef SX_RACE
    pthread_mutex_t mu;
#endif
    int finished, abandoned;
} IpmJob;
static void lp_dup(const LpProb *A, LpProb *B) {
    *B = *A; B->rname = B->cname = NULL;
    const int m = A->m, n = A->n, nz = A->Ap[n];
    B->Ap = malloc(sizeof(int) * ((size_t)n + 1)); B->Ai = malloc(sizeof(int) * ((size_t)nz + 1)); B->Ax = malloc(sizeof(double) * ((size_t)nz + 1));
    B->c = malloc(sizeof(double) * ((size_t)n + 1)); B->l = malloc(sizeof(double) * ((size_t)n + 1)); B->u = malloc(sizeof(double) * ((size_t)n + 1));
    B->rl = malloc(sizeof(double) * ((size_t)m + 1)); B->ru = malloc(sizeof(double) * ((size_t)m + 1));
    if (!B->Ap || !B->Ai || !B->Ax || !B->c || !B->l || !B->u || !B->rl || !B->ru) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); }
    memcpy(B->Ap, A->Ap, sizeof(int) * (n + 1)); memcpy(B->Ai, A->Ai, sizeof(int) * nz); memcpy(B->Ax, A->Ax, sizeof(double) * nz);
    memcpy(B->c, A->c, sizeof(double) * n); memcpy(B->l, A->l, sizeof(double) * n); memcpy(B->u, A->u, sizeof(double) * n);
    memcpy(B->rl, A->rl, sizeof(double) * m); memcpy(B->ru, A->ru, sizeof(double) * m);
}
static void job_free(IpmJob *J) {
    if (J->race) {
        lp_free(&J->own);
        if (J->has_orig) { lp_free(&J->own_orig); sxout_free(&J->own_out); }
#ifdef SX_RACE
        pthread_mutex_destroy(&J->mu);
#endif
    }
    free(J->x); free(J->y); free(J->cs); free(J->rs); free(J);
}
#ifdef SX_RACE
/* jobs left behind and still running; a process that exits meanwhile would wait forever in the threaded BLAS's shutdown
 * (its workers serve such a job), so the exit then skips the libraries' shutdown: the output is flushed and the process
 * ends with its status (glibc's on_exit gives it; elsewhere the losing job is waited for instead) */
static pthread_mutex_t g_left_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_left, g_left_guard;
#if defined(__GLIBC__)
static void left_exit(int status, void *arg) {
    (void)arg;
    pthread_mutex_lock(&g_left_mu); const int n = g_left; pthread_mutex_unlock(&g_left_mu);
    if (n > 0) { fflush(NULL); _exit(status); }
}
#define SX_LEAVE 1
#endif
#endif
#ifdef SX_RACE
static void *ipm_job(void *p) {
    IpmJob *J = p;
    omp_set_num_threads(J->nt);           /* (a new thread starts from the default count) */
    SimplexOpts ox = J->o; ox.method = 3; ox.verbose = 0; ox.stop = &J->stop;
    J->st = ipm_route(J->R, &ox, J->x, J->y, J->cs, J->rs, &J->info, &J->done);
    if (!J->race) return NULL;
    pthread_mutex_lock(&J->mu);
    if (J->done && J->st == 0 && !J->abandoned) { J->stop_s = 1; if (J->stop_other) *J->stop_other = 1; if (J->stop_other2) *J->stop_other2 = 1; }
    else if (!J->abandoned && J->threads_other) *J->threads_other = J->nt_all;
    J->finished = 1;
    const int ab = J->abandoned;
    pthread_mutex_unlock(&J->mu);
    if (ab) {
        job_free(J);
        pthread_mutex_lock(&g_left_mu); g_left--; pthread_mutex_unlock(&g_left_mu);
    }
    return NULL;
}
#endif
typedef struct {
    const LpProb *R; int stage, predicted, declined; double wpred; int verbose; NeForm P, D;
    const SimplexOpts *o; int nt0, launched; IpmJob *job; double t0;
#ifdef SX_RACE
    pthread_t th;
#endif
} AutoCtx;
/* the two forms of R as lpsolve.c builds them from its standard form: fixed columns out; a one-sided row gets a slack
 * (a singleton, unbounded), a ranged row a bounded one. The dual form's rows are the columns that are not singletons
 * with one bound (those are bounds on a row's multiplier there) and the ranged rows' slacks; its columns the rows'
 * multipliers (the rows of A on those columns) and one singleton for each bound of a dual row. */
static void auto_forms(AutoCtx *A) {
    const LpProb *R = A->R; const int m = R->m, n = R->n;
    NeForm *P = &A->P, *D = &A->D;
    int nineq = 0, nrng = 0;
    int *rk = malloc(sizeof(int) * ((size_t)m + 1));        /* row kind: 0 equation, 1 one-sided, 2 ranged, 3 free */
    for (int i = 0; i < m; i++) {
        const int lo = R->rl[i] > -LP_INF, up = R->ru[i] < LP_INF;
        rk[i] = lo && up ? (R->rl[i] == R->ru[i] ? 0 : 2) : (lo || up ? 1 : 3);
        if (rk[i] == 1) nineq++; else if (rk[i] == 2) nrng++;
    }
    /* the standard form: the columns that are not fixed */
    P->nr = m; P->Cp = malloc(sizeof(int) * ((size_t)n + 1)); P->Ci = malloc(sizeof(int) * ((size_t)R->Ap[n] + 1));
    int nc = 0; long q = 0; P->Cp[0] = 0;
    int *drow = malloc(sizeof(int) * ((size_t)n + 1)); int nrow2 = 0, nsd = 0;
    for (int j = 0; j < n; j++) {
        const int lo = R->l[j] > -LP_INF, up = R->u[j] < LP_INF, len = R->Ap[j + 1] - R->Ap[j];
        drow[j] = -1;
        if (lo && up && R->l[j] == R->u[j]) continue;
        for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) P->Ci[q++] = R->Ai[p];
        P->Cp[++nc] = (int)q;
        const int free_ = !lo && !up, box = lo && up;
        if (!(len == 1 && !free_ && !box)) { drow[j] = nrow2++; nsd += !free_ + box; }
    }
    P->nc = nc; P->ns = nineq + nrng;
    /* the dual form: by the rows of A, the columns of the dual rows, then the ranged rows' slacks */
    D->nr = nrow2 + nrng; D->nc = m; D->ns = nsd + 2 * nrng;
    D->Cp = calloc((size_t)m + 1, sizeof(int));
    for (int j = 0; j < n; j++) if (drow[j] >= 0) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) D->Cp[R->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) if (rk[i] == 2) D->Cp[i + 1]++;
    for (int i = 0; i < m; i++) D->Cp[i + 1] += D->Cp[i];
    D->Ci = malloc(sizeof(int) * ((size_t)D->Cp[m] + 1));
    { int *f = malloc(sizeof(int) * ((size_t)m + 1)); memcpy(f, D->Cp, sizeof(int) * m);
      for (int j = 0; j < n; j++) if (drow[j] >= 0) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) D->Ci[f[R->Ai[p]]++] = drow[j];
      int k = nrow2; for (int i = 0; i < m; i++) if (rk[i] == 2) D->Ci[f[i]++] = k++;
      free(f); }
    free(rk); free(drow);
    neform_init(P); neform_init(D);
}
static void auto_free(AutoCtx *A) {
    NeForm *F[2] = { &A->P, &A->D };
    for (int k = 0; k < 2; k++) { free(F[k]->Cp); free(F[k]->Ci); free(F[k]->isd); }
}
/* a cheap lower bound on an iteration's flops on L's normal equations (the columns' l(l+1)/2, the dense ones apart, as the
 * automatic choice's patterns: neform_init), the cheaper of the two forms */
static double ipm_form_lb(const LpProb *P) {
    AutoCtx A; memset(&A, 0, sizeof A); A.R = P;
    auto_forms(&A);
    const double c = A.D.nr < A.P.nr ? fmin(A.P.lb, A.D.lb) : A.P.lb;
    auto_free(&A);
    return c;
}
static double auto_budget(void *vctx, double work) {
    AutoCtx *A = vctx;
    if (A->predicted) { A->declined = 1; return 0; }
    if (A->stage == 0) {
        /* a lower bound first, without an ordering */
        A->stage = 1;
        auto_forms(A);
        const double w0 = SX_AUTO_IT * (A->D.nr < A->P.nr ? fmin(A->P.lb, A->D.lb) : A->P.lb) / SX_AUTO_FLOPS + SX_AUTO_SETUP;
        if (w0 > work) return w0;
    }
    A->predicted = 1;
    /* the dual form first; the standard form's ordering only when its lower bound is below twice the dual's cost */
    NeForm *P = &A->P, *D = &A->D;
    if (D->nr < P->nr) neform_cost(D);      /* (the dual form only with fewer rows, as lpsolve.c takes it) */
    if (!(D->cost > 0 && P->lb > 2.0 * D->cost)) neform_cost(P);
    double c = 0; int dual = 0;
    if (D->cost > 0 && (P->cost == 0 || D->cost <= 0.5 * P->cost)) { c = D->cost; dual = 1; }
    else c = P->cost;
    const double wp = c > 0 ? SX_AUTO_IT * c / SX_AUTO_FLOPS + SX_AUTO_SETUP : 0;
    if (A->verbose >= 1) {
        if (c > 0) printf("  the work so far %.3g; the interior-point method's predicted %.3g (%s form: %d rows, factor %.3g, %d dense columns)\n",
                          work, wp, dual ? "dual" : "standard", dual ? D->nr : P->nr, dual ? D->lnz : P->lnz, dual ? D->nd : P->nd);
        else printf("  the work so far %.3g; the normal equations too large to predict: the simplex method continues\n", work);
    }
    A->wpred = wp;
    if (wp == 0) { A->declined = 0; return 1e300; }
    if (wp > work) {
#ifdef SX_RACE
        /* with two threads or more, the interior-point method starts now on half of them; the simplex method continues
         * on the others up to the same budget, so that the result is the one of the sequential order */
        int nf = A->nt0;
        if (nf >= 2) {          /* only the free cores (the busy-machine rule of the other engines, solver.c) */
            const int nb = brisk_threads_busy(0);
            nf = omp_get_max_threads();
            if (nb > 0) omp_set_num_threads(nb);
        }
        if (nf >= 2 && !getenv("BRISK_SXNORACE")) {
            const LpProb *R = A->R; IpmJob *J = calloc(1, sizeof *J);
            J->R = R; J->o = *A->o; J->nt = nf - nf / 2;
            if (J->o.timelimit > 0) J->o.timelimit = fmax(A->o->timelimit - (now() - A->t0), 1e-3);
            J->x = malloc(sizeof(double) * (R->n + 1)); J->y = malloc(sizeof(double) * (R->m + 1));
            J->cs = malloc(sizeof(int) * (R->n + 1)); J->rs = malloc(sizeof(int) * (R->m + 1));
            if (pthread_create(&A->th, NULL, ipm_job, J) == 0) {
                A->launched = 1; A->job = J; omp_set_num_threads(nf / 2);
                if (A->verbose >= 1) printf("  the interior-point method started on %d of %d threads, alongside the simplex method\n", J->nt, nf);
            } else { free(J->x); free(J->y); free(J->cs); free(J->rs); free(J); }
        }
#endif
        return wp;
    }
    A->declined = 1;
    return 0;
}
/* the test of dual_route's first step: a problem it would consider solving through its dual (outside the parallel
 * code: ipm_route uses it in every build; inside it, the builds without SX_RACE, the bundled OpenMP runtime of
 * macOS and OMP=0, did not link) */
static int dual_route_applies_f(const LpProb *R, double f) {
    if (getenv("BRISK_SXNODUAL")) return 0;
    if (getenv("BRISK_SXFORCEDUAL")) return 1;      /* (an experiment, 23 October 2026: savsched1, which Gurobi dualizes) */
    const double df = getenv("BRISK_SXDUALF") ? atof(getenv("BRISK_SXDUALF")) : f;
    int est = 0;
    for (int j = 0; j < R->n; j++) {
        if (R->l[j] == R->u[j]) continue;
        const int boxed = R->l[j] > -HUGEB && R->u[j] < HUGEB;
        if (R->Ap[j + 1] - R->Ap[j] > 1 || boxed) est++;
    }
    return df * est <= R->m;
}
static int dual_route_applies(const LpProb *R) { return dual_route_applies_f(R, 2.0); }
#ifdef SX_RACE
/* the race by the clock (the owner's decision of 21 October 2026: the default with two free threads or more). The simplex
 * method starts alone with the first budget of the automatic choice, so that a cheap problem never starts the
 * interior-point method; past it, the interior-point method and its crossover start on all free threads but one, and
 * the simplex method continues on that one. The first to end optimal gives the result: when it is the simplex method the
 * other is told to stop and left to end by itself on its copy of the problem (it looks at its flag only once per
 * iteration), when it is the
 * interior-point method the simplex method stops at its next iteration. If the interior-point method fails, the simplex
 * method simply continues. The result can differ from run to run when the two end close together. */
/* the race starts at the first budget of the automatic choice, but not before the simplex method has spent a quarter of
 * the interior-point method's fixed setup (SX_AUTO_SETUP, about 1.2 s of the simplex method: its analyses): it cannot end
 * before that setup is paid, and a problem the simplex method solves within it only pays for the start (bnatt400 0.07 ->
 * 0.15 s, csched008 0.03 -> 0.09 s at four threads when the race started at the first budget alone) */
/* when the interior-point method joins the race: after max(SX_RACE_B0 times the entries and rows, SX_RACE_START) of the
 * simplex method's work. (23 October 2026: 200 and 0.25 SX_AUTO_SETUP before; with 20 and 0.025 SX_AUTO_SETUP the MIPLIB
 * relaxations (LPs, four threads, total time) 1.219 -> 1.170 s, 41 files faster and 11 slower by more than a tenth: bab2
 * 10.9 -> 6.9 s, ns1760995 24.7 -> 18.6, cryptanalysiskb128n5obj16 4.8 -> 3.6; map10 4.9 -> 5.7, k1mushroom 7.7 -> 8.7.
 * BRISK_SXRACEB0, BRISK_SXRACESTART: the two values) */
#define SX_RACE_B0 20.0
#define SX_RACE_START (0.025 * SX_AUTO_SETUP)
/* a second simplex job beside the main one: for a tall problem the direct route, beside the route through the dual
 * problem (the race of the two simplex routes); for another problem the primal simplex method, beside the dual (Gurobi's
 * concurrent default for linear programs runs the primal and the dual simplex method and the barrier) */
typedef struct {
    const LpProb *R; SimplexOpts o; double *x, *y; int *cs, *rs; SimplexInfo info; int st;
    int primal;                 /* 1: the primal simplex method (the main thread runs the dual); 0: the direct route */
    volatile int stop, one, won;
    volatile int *stop_main;    /* the other simplex route's stop flag (the caller waits for this job: always valid) */
    struct RaceCtx_ *C;
} DirJob;
typedef struct RaceCtx_ { const LpProb *R; const SimplexOpts *o; int nf; IpmJob *J; pthread_t th; double t0; volatile int stop_ind, threads;
                          DirJob *DJ; pthread_t thd; pthread_mutex_t mu; } RaceCtx;
static void *dir_job(void *p) {
    DirJob *D = p;
    omp_set_num_threads(1);
    D->st = reduced_solve(D->R, &D->o, D->x, D->y, D->cs, D->rs, &D->info);
    if (D->st == 0 && !D->stop) {
        D->won = 1; *D->stop_main = 1;
        pthread_mutex_lock(&D->C->mu);
        if (D->C->J) D->C->J->stop = 1;
        pthread_mutex_unlock(&D->C->mu);
    }
    return NULL;
}
static double g_race_wait;   /* (diagnostic) the time spent waiting for a losing interior-point method to stop */
static double race_budget(void *vctx, double work) {
    RaceCtx *C = vctx;
    (void)work;
    if (C->J || (C->DJ && C->DJ->won)) return 1e300;
    const LpProb *R = C->R;
    IpmJob *J = calloc(1, sizeof *J);
    if (!J) return 1e300;
    lp_dup(R, &J->own); J->R = &J->own; J->race = 1; J->o = *C->o; J->o.stop = NULL; J->o.threads = NULL; J->stop_other = &C->stop_ind;
    if (C->o->orig) {
        lp_dup(C->o->orig->L, &J->own_orig); sxout_init(&J->own_out, &J->own_orig); J->has_orig = 1;
        J->own_o.L = &J->own_orig; J->own_o.out = &J->own_out; J->o.orig = &J->own_o;
    }
    J->nt = C->nf - 1 - (C->DJ != NULL); if (J->nt < 1) J->nt = 1;
    J->threads_other = &C->threads; J->nt_all = C->DJ ? 1 : C->nf;
    if (C->DJ) J->stop_other2 = &C->DJ->stop;
    if (J->o.timelimit > 0) J->o.timelimit = fmax(C->o->timelimit - (now() - C->t0), 1e-3);
    J->x = malloc(sizeof(double) * (R->n + 1)); J->y = malloc(sizeof(double) * (R->m + 1));
    J->cs = malloc(sizeof(int) * (R->n + 1)); J->rs = malloc(sizeof(int) * (R->m + 1));
    pthread_mutex_init(&J->mu, NULL);
    C->threads = 1;                 /* (before the job starts: the simplex method keeps one thread while it runs) */
    pthread_mutex_lock(&C->mu);
    const int ok = !(C->DJ && C->DJ->won) && pthread_create(&C->th, NULL, ipm_job, J) == 0;
    if (ok) C->J = J;
    pthread_mutex_unlock(&C->mu);
    if (!ok) { job_free(J); C->threads = C->DJ ? 1 : C->nf; return 1e300; }
    if (C->o->verbose >= 1) printf("  the interior-point method started on %d of %d threads, beside the simplex method (the first to end wins)\n", J->nt, C->nf);
    return 1e300;
}
/* the free threads (the busy-machine rule of the other engines, solver.c) */
static int race_threads(void) {
    int nf = omp_get_max_threads();
    if (nf >= 2) { const int nb = brisk_threads_busy(0); if (nb > 0 && nb < nf) nf = nb; }
    return nf;
}
static int race_route(const LpProb *R, const SimplexOpts *o2, int nf, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    const double t0 = now();
    const int nt0 = omp_get_max_threads();
    RaceCtx C; memset(&C, 0, sizeof C); C.R = R; C.o = o2; C.nf = nf; C.t0 = t0;
    SimplexOpts os = *o2; os.method = 1; os.stop = &C.stop_ind;
    {
        const double b0 = getenv("BRISK_SXRACEB0") ? atof(getenv("BRISK_SXRACEB0")) : SX_RACE_B0;
        const double bs = getenv("BRISK_SXRACESTART") ? atof(getenv("BRISK_SXRACESTART")) : SX_RACE_START;
        os.workbudget = fmax(b0 * (R->Ap[R->n] + R->m), bs);
    }
    os.budget_cb = race_budget; os.budget_ctx = &C;
    /* all the free threads until the race starts, one while the interior-point method runs, all again if it fails */
    C.threads = nf; os.threads = &C.threads;
    pthread_mutex_init(&C.mu, NULL);
    /* a tall problem: the direct route on a thread of its own beside the route through the dual problem (either can be
     * far the better: ex9 over 300 s through its dual and 38 s directly, gfd-schedulen180f7d50m30k18 the other way round) */
    DirJob *DJ = NULL;
    const int tall = nf >= 3 && dual_route_applies_f(R, 1.0) && !getenv("BRISK_SXNOROUTERACE");
    /* (BRISK_SXPRIMALRACE=1, an experiment: the primal simplex method as a third method on a thread of its own, as
     * Gurobi's concurrent default runs it; off: on the hardest MIPLIB relaxations at four threads it lost 17 of 18
     * measured cleanly, shifted geometric mean 13.1 -> 16.5 s, the thread it takes slowing the other two; TRICKS 8) */
    const int prim = !tall && nf >= 3 && getenv("BRISK_SXPRIMALRACE") && atoi(getenv("BRISK_SXPRIMALRACE")) == 1;
    if (tall || prim) {
        DJ = calloc(1, sizeof *DJ);
        DJ->R = R; DJ->o = os; DJ->o.stop = &DJ->stop; DJ->o.workbudget = 0; DJ->o.budget_cb = NULL; DJ->o.budget_ctx = NULL;
        DJ->primal = prim; if (prim) DJ->o.method = 2;
        DJ->one = 1; DJ->o.threads = &DJ->one; DJ->o.verbose = 0; DJ->stop_main = &C.stop_ind; DJ->C = &C;
        DJ->x = malloc(sizeof(double) * (R->n + 1)); DJ->y = malloc(sizeof(double) * (R->m + 1));
        DJ->cs = malloc(sizeof(int) * (R->n + 1)); DJ->rs = malloc(sizeof(int) * (R->m + 1));
        if (pthread_create(&C.thd, NULL, dir_job, DJ) == 0) {
            C.DJ = DJ; C.threads = 1;
            if (o2->verbose >= 1) printf(prim ? "  the primal simplex method on a thread of its own, beside the dual simplex method\n"
                                              : "  the direct route on a thread of its own, beside the route through the dual problem\n");
        }
        else { free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); DJ = NULL; }
    }
    omp_set_num_threads(C.threads);
    SimplexInfo is; memset(&is, 0, sizeof is);
    os.dual_race = DJ && !DJ->primal;      /* (the direct route beside it: the route through the dual at the factor 1) */
    int dn = 0, st = dual_route(R, &os, xr, yr, csr, rsr, &is, &dn);
    os.dual_race = 0;
    if (!dn && !C.stop_ind) {
        if (DJ && !DJ->primal) {   /* the dual problem was not used: the direct route's job is this route's solve */
            pthread_join(C.thd, NULL);
            st = DJ->st; is = DJ->info;
            memcpy(xr, DJ->x, sizeof(double) * R->n); memcpy(yr, DJ->y, sizeof(double) * R->m);
            memcpy(csr, DJ->cs, sizeof(int) * R->n); memcpy(rsr, DJ->rs, sizeof(int) * R->m);
            if (DJ->won) C.stop_ind = 0;     /* (its own result: not a stop from outside) */
            free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); DJ = C.DJ = NULL;
        } else {
            int da = 0; st = aggb_route(R, &os, xr, yr, csr, rsr, &is, &da);
            if (!da && !C.stop_ind) st = reduced_solve(R, &os, xr, yr, csr, rsr, &is);
        }
    }
    omp_set_num_threads(nt0);
    if (DJ) {
        /* the two simplex routes: the direct one won (it stopped the other), or it is told to stop and waited for (it reads
         * its flag every iteration) */
        DJ->stop = 1;
        pthread_join(C.thd, NULL);
        if (DJ->won && C.stop_ind) {
            st = DJ->st; is = DJ->info;
            memcpy(xr, DJ->x, sizeof(double) * R->n); memcpy(yr, DJ->y, sizeof(double) * R->m);
            memcpy(csr, DJ->cs, sizeof(int) * R->n); memcpy(rsr, DJ->rs, sizeof(int) * R->m);
            C.stop_ind = 0;
            if (o2->verbose >= 1) printf(DJ->primal ? "  the primal simplex method ended first\n" : "  the direct route ended first\n");
        }
        free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); DJ = C.DJ = NULL;
    }
    IpmJob *J = C.J;
    if (!J) { *ir = is; *done = 1; pthread_mutex_destroy(&C.mu); return st; }
    int use_ipm = 0;
    if (C.stop_ind) {       /* (the interior-point method ended optimal first; the simplex method stopped, possibly inside
                               the route through the dual problem, which then returns as not done) */
        pthread_join(C.th, NULL);          /* (it has ended: it set the flag) */
        use_ipm = J->done && J->st == 0;
        if (use_ipm) {
            memcpy(xr, J->x, sizeof(double) * R->n); memcpy(yr, J->y, sizeof(double) * R->m);
            memcpy(csr, J->cs, sizeof(int) * R->n); memcpy(rsr, J->rs, sizeof(int) * R->m);
            if (J->has_orig && J->own_out.have) sxout_copy(o2->orig->out, &J->own_out, o2->orig->L);
            *ir = J->info; ir->iters += is.iters; ir->work += is.work; ir->auto_ipm = 1; ir->iters_first = is.iters; ir->t_first = is.time;
            st = J->st;
            if (o2->verbose >= 1) printf("  the interior-point method and the crossover ended first (%d iterations, %.2f s; the simplex method stopped after %ld iterations)\n", J->info.ipm_iters, J->info.t_ipm, is.iters);
        }
        job_free(J);
    } else {
        /* the interior-point method is told to stop. It looks at its flag once per iteration, not during its analysis, so
         * it is left to end by itself on its copy of the problem (where on_exit is not available, it is waited for) */
        const double tw = now();
        J->stop = 1;
#ifdef SX_LEAVE
        pthread_mutex_lock(&g_left_mu);
        if (!g_left_guard) { on_exit(left_exit, NULL); g_left_guard = 1; }
        pthread_mutex_unlock(&g_left_mu);
        pthread_mutex_lock(&J->mu);
        const int fin = J->finished;
        if (!fin) { J->abandoned = 1; pthread_mutex_lock(&g_left_mu); g_left++; pthread_mutex_unlock(&g_left_mu); }
        pthread_mutex_unlock(&J->mu);
        if (fin) { pthread_join(C.th, NULL); job_free(J); } else pthread_detach(C.th);
#else
        pthread_join(C.th, NULL);
        job_free(J);
#endif
        if (o2->verbose >= 1) printf("  the simplex method ended first: the interior-point method told to stop\n");
        g_race_wait += now() - tw;
    }
    if (!use_ipm) *ir = is;
    ir->raced = 1; ir->time = now() - t0;
    *done = 1;
    pthread_mutex_destroy(&C.mu);
    return st;
}
#endif
static int auto_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    if (getenv("BRISK_SXNOAUTO")) return 0;
#ifdef SX_RACE
    if (!o2->deterministic && !getenv("BRISK_SXDETERMINISTIC") && !getenv("BRISK_SXNORACE")) {
        const int nf = race_threads();
        if (nf >= 2) return race_route(R, o2, nf, xr, yr, csr, rsr, ir, done);
    }
#endif
    const double t0 = now();
    AutoCtx A; memset(&A, 0, sizeof A); A.R = R; A.verbose = o2->verbose; A.o = o2; A.t0 = t0;
#ifdef _OPENMP
    A.nt0 = omp_get_max_threads();
#endif
    SimplexOpts os = *o2; os.method = 1;
    os.workbudget = SX_AUTO_B0 * (R->Ap[R->n] + R->m); os.budget_cb = auto_budget; os.budget_ctx = &A;
    SimplexInfo is; memset(&is, 0, sizeof is);
    int dn = 0, st = dual_route(R, &os, xr, yr, csr, rsr, &is, &dn);
    if (!dn) st = reduced_solve(R, &os, xr, yr, csr, rsr, &is);
    auto_free(&A);
#ifdef SX_RACE
    if (A.launched) {
        IpmJob *J = A.job;
        omp_set_num_threads(A.nt0);
        if (!(st == 6 && A.declined)) J->stop = 1;      /* the simplex method finished within the budget: its result */
        pthread_join(A.th, NULL);
        const int use = st == 6 && A.declined && J->done;
        if (use) {
            memcpy(xr, J->x, sizeof(double) * R->n); memcpy(yr, J->y, sizeof(double) * R->m);
            memcpy(csr, J->cs, sizeof(int) * R->n); memcpy(rsr, J->rs, sizeof(int) * R->m);
            if (J->has_orig && J->own_out.have) sxout_copy(o2->orig->out, &J->own_out, o2->orig->L);
            if (o2->verbose >= 1) printf("  the simplex method passed the predicted work of the interior-point method (%.3g): the interior-point method's result (%d iterations, %.2f s; the crossover %ld iterations)\n",
                                         A.wpred, J->info.ipm_iters, J->info.t_ipm, J->info.iters);
            *ir = J->info; ir->iters += is.iters; ir->work += is.work; ir->auto_ipm = 1; ir->iters_first = is.iters; ir->t_first = is.time;
            ir->time = now() - t0; *done = 1; st = J->st;
        }
        free(J->x); free(J->y); free(J->cs); free(J->rs); free(J);
        if (use) return st;
        if (!(st == 6 && A.declined)) { *ir = is; *done = 1; return st; }
        goto fallback;
    }
#endif
    if (!(st == 6 && A.declined)) { *ir = is; *done = 1; return st; }
    if (o2->verbose >= 1) printf("  the simplex method passed the predicted work of the interior-point method (%.3g): the interior-point method (%.2f s)\n", A.wpred, now() - t0);
    /* (the time limit is the whole solve's: the interior-point method and the fallback get what is left) */
    SimplexOpts ox = *o2; ox.method = 3;
    if (o2->timelimit > 0) { ox.timelimit = o2->timelimit - (now() - t0); if (ox.timelimit <= 0) { *ir = is; *done = 1; return 6; } }
    SimplexInfo ix; memset(&ix, 0, sizeof ix);
    int dx = 0;
    st = ipm_route(R, &ox, xr, yr, csr, rsr, &ix, &dx);
    if (dx) { *ir = ix; ir->iters += is.iters; ir->work += is.work; ir->auto_ipm = 1; ir->iters_first = is.iters; ir->t_first = is.time; ir->time = now() - t0; *done = 1; return st; }
#ifdef SX_RACE
fallback:
#endif
    /* the interior-point method failed: the simplex method from the basis it reached */
    if (o2->verbose >= 1) printf("  the interior-point method did not end optimal: the simplex method from its basis\n");
    SimplexOpts oc = *o2; oc.method = 1; oc.cstat0 = csr; oc.rstat0 = rsr; oc.nopresolve = 1;
    if (o2->timelimit > 0) { oc.timelimit = o2->timelimit - (now() - t0); if (oc.timelimit <= 0) { *ir = is; ir->time = now() - t0; *done = 1; return 6; } }
    int *cs0 = malloc(sizeof(int) * (R->n + 1)), *rs0 = malloc(sizeof(int) * (R->m + 1));
    memcpy(cs0, csr, sizeof(int) * R->n); memcpy(rs0, rsr, sizeof(int) * R->m); oc.cstat0 = cs0; oc.rstat0 = rs0;
    SimplexInfo ic; st = sx_solve_core(R, &oc, xr, NULL, yr, NULL, csr, rsr, &ic);
    free(cs0); free(rs0);
    *ir = ic; ir->iters += is.iters; ir->work += is.work; ir->time = now() - t0; *done = 1;
    return st;
}

/* the simplex method with the presolve: the reduced problem is solved, its basis carried back by the postsolve, and the
 * problem as read solved from that basis (optimal as it is, or finished by a few iterations) */
/* the symmetry of the problem as read, when the presolved problem has none of use (23 October 2026): the presolve can break it (ex9 (LP): 40,962
 * rows in 10,179 classes as read, none left after the presolve, whose reductions treat the rows of a class one at a time),
 * and the LP route folds the problem as read first for that reason (main.c). The problem on the classes is solved by
 * simplex_solve (its presolve and routes), its point lifted to the problem as read, and the crossover finishes there. */
static int sym_first(const LpProb *L, const SimplexOpts *o, double size_pre, double *xout, double *axout, double *yout, double *zout,
                     int *cstat, int *rstat, SimplexInfo *info, int *done) {
    *done = 0;
    const double t0 = now();
    LpProb Q; SxSym *M = NULL;
    /* (at most ten passes over the matrix: on problems whose columns share costs and bounds the classes by the data are
     * few and the rounds run long for nothing; ex9's classes take five rounds) */
    if (!sxpre_symmetry(L, &Q, &M, o && o->verbose >= 1, 10.0 * ((double)L->Ap[L->n] + L->m + L->n))) return 0;
    /* (used when the classes are at most 0.7 of the presolved problem's rows and columns, the ratio of the presolved
     * problem's own symmetry test: the presolve alone may reduce more) */
    if ((double)Q.m + Q.n > 0.7 * size_pre) { lp_free(&Q); sxpre_sym_free(M); return 0; }
    const double tsym = now() - t0;
    SimplexOpts oq; if (o) oq = *o; else memset(&oq, 0, sizeof oq);
    oq.nosymmetry = 1; oq.nopresolve = 0; oq.orig = NULL;
    double *xq = malloc(sizeof(double) * (Q.n + 1)), *yq = malloc(sizeof(double) * (Q.m + 1));
    SimplexInfo iq; memset(&iq, 0, sizeof iq);
    const int mq = Q.m, nq = Q.n;
    int st = simplex_solve(&Q, &oq, xq, NULL, yq, NULL, NULL, NULL, &iq);
    lp_free(&Q);
    if (o && o->verbose >= 1) printf("  the problem on the classes of the problem as read (%d x %d, %.2f s to find them): status %d, %ld iterations (%.2f s)\n", mq, nq, tsym, st, iq.iters, iq.time);
    if (st != 0) { free(xq); free(yq); sxpre_sym_free(M); return 0; }
    double *x = malloc(sizeof(double) * (L->n + 1)), *y = malloc(sizeof(double) * (L->m + 1));
    sxpre_sym_lift(M, xq, yq, x, y);
    free(xq); free(yq); sxpre_sym_free(M);
    SimplexOpts o3; if (o) o3 = *o; else memset(&o3, 0, sizeof o3);
    o3.xstart = x; o3.ystart = y; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.orig = NULL;
    if (o3.timelimit > 0) { o3.timelimit -= now() - t0; if (o3.timelimit <= 0) o3.timelimit = 1e-3; }
    SimplexInfo i3;
    st = sx_solve_core(L, &o3, xout, axout, yout, zout, cstat, rstat, &i3);
    free(x); free(y);
    if (o && o->verbose >= 1) printf("  the problem as read from the classes' solution: status %d, %ld pushes, %ld iterations (%.2f s)\n", st, i3.pushes, i3.iters, i3.time);
    *info = i3;
    info->iters = iq.iters + i3.iters; info->iters_dual = iq.iters_dual + i3.iters_dual; info->iters_primal = iq.iters_primal + i3.iters_primal;
    info->iters_phase1 = iq.iters_phase1 + i3.iters_phase1; info->nfactor = iq.nfactor + i3.nfactor; info->nflips = iq.nflips + i3.nflips;
    info->t_factor = iq.t_factor + i3.t_factor; info->t_ftran = iq.t_ftran + i3.t_ftran; info->t_btran = iq.t_btran + i3.t_btran; info->t_price = iq.t_price + i3.t_price;
    info->t_presolve = iq.t_presolve; info->m_reduced = iq.m_reduced; info->n_reduced = iq.n_reduced; info->ipm_iters = iq.ipm_iters; info->t_ipm = iq.t_ipm;
    info->auto_ipm = iq.auto_ipm; info->raced = iq.raced; info->dualized = iq.dualized;
    info->time = now() - t0; info->sym_m = mq; info->sym_n = nq; info->work = iq.work + i3.work; info->pushes = iq.pushes + i3.pushes;
    *done = 1;
    return st;
}

static int simplex_solve_in(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat,
                            int *rstat, SimplexInfo *info);
/* the problem without its entries too small to matter (tiny_keep), when it has any; the solution's row activities are
 * then those of that problem (within 1e-10 of the given one's) */
int simplex_solve(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                  SimplexInfo *info) {
    const int n = L->n, nz = L->Ap[n];
    unsigned char *keep = xmalloc((size_t)nz + 1);
    const long nd = tiny_keep(L, keep);
    if (!nd) { free(keep); return simplex_solve_in(L, o, xout, axout, yout, zout, cstat, rstat, info); }
    LpProb T = *L;
    T.Ap = xmalloc(sizeof(int) * ((size_t)n + 1)); T.Ai = xmalloc(sizeof(int) * ((size_t)(nz - nd) + 1)); T.Ax = xmalloc(sizeof(double) * ((size_t)(nz - nd) + 1));
    int e = 0;
    for (int j = 0; j < n; j++) { T.Ap[j] = e; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (keep[p]) { T.Ai[e] = L->Ai[p]; T.Ax[e] = L->Ax[p]; e++; } }
    T.Ap[n] = e;
    free(keep);
    if (o && o->verbose >= 1) printf("  %ld matrix entries too small to matter dropped (below 1e-9 of their row's largest; each row's activity moved by at most 1e-10)\n", nd);
    const int st = simplex_solve_in(&T, o, xout, axout, yout, zout, cstat, rstat, info);
    free(T.Ap); free(T.Ai); free(T.Ax);
    return st;
}
static int simplex_solve_in(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat,
                            int *rstat, SimplexInfo *info) {
    SimplexInfo dummy; if (!info) info = &dummy;
    if ((o && (o->nopresolve || o->cstat0)) || getenv("BRISK_SXNOPRESOLVE")) return sx_solve_core(L, o, xout, axout, yout, zout, cstat, rstat, info);
    const double t0 = now();
    LpProb R; SxPre *P = NULL; char why[160] = "";
    const int ps = sxpre_presolve(L, &R, &P, o ? o->verbose >= 1 : 0, why, sizeof why);
    const double tpre = now() - t0;
    if (ps) {
        memset(info, 0, sizeof *info);
        info->status = ps; snprintf(info->why, sizeof info->why, "presolve: %s", why);
        info->time = now() - t0; info->t_presolve = tpre;
        sxpre_free(P);
        return ps;
    }
    const int m = L->m, n = L->n, mr = R.m, nr = R.n;
    int *cs = malloc(sizeof(int) * (n + 1)), *rs = malloc(sizeof(int) * (m + 1));
    double *x = malloc(sizeof(double) * (n + 1)), *y = malloc(sizeof(double) * (m + 1));
    SimplexInfo ir; memset(&ir, 0, sizeof ir);
    int st = 0;
    if (mr > 0 && nr > 0) {
        int *csr = malloc(sizeof(int) * (nr + 1)), *rsr = malloc(sizeof(int) * (mr + 1));
        double *xr = malloc(sizeof(double) * (nr + 1)), *yr = malloc(sizeof(double) * (mr + 1));
        SimplexOpts o2; if (o) o2 = *o; else memset(&o2, 0, sizeof o2);
        o2.nopresolve = 1;
        struct SxOut out; sxout_init(&out, L);
        struct SxOrig orig = { L, &out }; o2.orig = &orig;
        int done = 0;
        /* (the time limit is the whole solve's: a route that gives up leaves the next one what is left) */
        #define SX_LEFT() (o && o->timelimit > 0 ? (o2.timelimit = o->timelimit - (now() - t0)) > 0 : 1)
        const int sym = !(o && (o->nosymmetry || o->xstart)) && !getenv("BRISK_SXNOSYM");
        if (sym) st = sym_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT() && sym && !getenv("BRISK_SXNOSYMREAD")) {
            /* the presolved problem has no symmetry of use: that of the problem as read, which the presolve may have
             * broken (sym_first); its result is final */
            int d2 = 0;
            const int st2 = sym_first(L, &o2, (double)mr + nr, xout, axout, yout, zout, cstat, rstat, info, &d2);
            if (d2) {
                free(csr); free(rsr); free(xr); free(yr); sxout_free(&out); lp_free(&R); sxpre_free(P);
                free(cs); free(rs); free(x); free(y);
                info->t_presolve = tpre; info->time = now() - t0;
                return st2;
            }
        }
        if (!done && SX_LEFT() && o && o->method == 3) st = ipm_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT() && (!o || o->method == 0)) st = auto_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT() && (!o || o->method != 2)) st = dual_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT()) st = reduced_solve(&R, &o2, xr, yr, csr, rsr, &ir);
        if (!done && o && o->timelimit > 0 && o2.timelimit <= 0) { st = 6; ir.status = 6; }
        #undef SX_LEFT
        if (out.have) { memcpy(cs, out.cs, sizeof(int) * n); memcpy(rs, out.rs, sizeof(int) * m); }   /* (a basis of the problem as read: no postsolve) */
        else if (st == 0) sxpre_postsolve(P, xr, yr, csr, rsr, x, y, cs, rs);
        free(csr); free(rsr); free(xr); free(yr); sxout_free(&out);
    } else {
        /* everything removed: the postsolve alone (the reduced problem may keep rows without columns: their logicals) */
        int *csr = malloc(sizeof(int) * (nr + 1)), *rsr = malloc(sizeof(int) * (mr + 1));
        double *xr = calloc(nr + 1, sizeof(double)), *yr = calloc(mr + 1, sizeof(double));
        for (int k = 0; k < nr; k++) { csr[k] = R.l[k] > -1e29 ? SX_LOWER : R.u[k] < 1e29 ? SX_UPPER : SX_ZERO; xr[k] = R.l[k] > -1e29 ? R.l[k] : R.u[k] < 1e29 ? R.u[k] : 0; }
        for (int k = 0; k < mr; k++) rsr[k] = SX_BASIC;
        sxpre_postsolve(P, xr, yr, csr, rsr, x, y, cs, rs);
        free(csr); free(rsr); free(xr); free(yr);
    }
    lp_free(&R); sxpre_free(P);
    if (st != 0) {
        *info = ir; info->t_presolve = tpre; info->time = now() - t0;
        free(cs); free(rs); free(x); free(y);
        return st;
    }
    /* the problem as read, from the postsolved basis */
    SimplexOpts o3; if (o) o3 = *o; else memset(&o3, 0, sizeof o3);
    o3.cstat0 = cs; o3.rstat0 = rs; o3.nopresolve = 1;
    SimplexInfo i3;
    st = sx_solve_core(L, &o3, xout, axout, yout, zout, cstat, rstat, &i3);
    if (o && o->verbose >= 2) printf("  the problem as read from the postsolved basis: status %d, %ld iterations (%.2f s)\n", st, i3.iters, i3.time);
    *info = i3;
    info->iters = ir.iters + i3.iters; info->iters_dual = ir.iters_dual + i3.iters_dual; info->iters_primal = ir.iters_primal + i3.iters_primal;
    info->iters_phase1 = ir.iters_phase1 + i3.iters_phase1; info->iters_post = i3.iters;
    info->nfactor = ir.nfactor + i3.nfactor; info->nflips = ir.nflips + i3.nflips;
    info->t_factor = ir.t_factor + i3.t_factor; info->t_ftran = ir.t_ftran + i3.t_ftran; info->t_btran = ir.t_btran + i3.t_btran; info->t_price = ir.t_price + i3.t_price;
    info->t_presolve = tpre; info->time = now() - t0; info->work = ir.work + i3.work;
    info->m_reduced = mr; info->n_reduced = nr; info->dualized = ir.dualized; info->sym_m = ir.sym_m; info->sym_n = ir.sym_n; info->pushes = ir.pushes; info->ipm_iters = ir.ipm_iters; info->t_ipm = ir.t_ipm; info->auto_ipm = ir.auto_ipm; info->raced = ir.raced;
    info->iters_first = ir.iters_first; info->t_first = ir.t_first;
    free(cs); free(rs); free(x); free(y);
    return st;
}

/* ---------------------------------------------------------------- the cone form of the interfaces */
int simplex_solve_sedumi(const SedumiProb *P, const SimplexOpts *o, double *x, double *y, double *z, double *pobj, double *dobj,
                         SimplexInfo *info) {
    SimplexInfo dummy; if (!info) info = &dummy;
    memset(info, 0, sizeof *info);
    if (P->nq || P->nr || P->ns || P->nf + P->nl != P->n || P->m <= 0 || P->n <= 0) return -1;
    const int m = P->m, n = P->n;
    LpProb L; memset(&L, 0, sizeof L);
    L.m = m; L.n = n; L.Ap = P->Ap; L.Ai = P->Ai; L.Ax = P->Ax; L.c = P->c; L.rl = P->b; L.ru = P->b;
    L.l = malloc(sizeof(double) * ((size_t)n + 1)); L.u = malloc(sizeof(double) * ((size_t)n + 1));
    double *ray = malloc(sizeof(double) * ((size_t)m + 1));
    if (!L.l || !L.u || !ray) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); }
    for (int j = 0; j < n; j++) { L.l[j] = j < P->nf ? -LP_INF : 0.0; L.u[j] = LP_INF; }
    snprintf(L.name, sizeof L.name, "cone form");
    SimplexOpts oo; if (o) oo = *o; else memset(&oo, 0, sizeof oo);
    oo.ray = ray;
    int st = simplex_solve(&L, &oo, x, NULL, y, z, NULL, NULL, info);
    if (st == 1 && info->have_ray) {
        /* the Farkas certificate: a column j can help only if ray_sign (ray'a_j) dir_j > 0, so y = ray_sign ray has
         * (A'y)_j <= 0 on the nonnegative columns and 0 on the free ones; then b'y > 0, normalised to 1 */
        double by = 0;
        for (int i = 0; i < m; i++) { y[i] = info->ray_sign * ray[i]; by += P->b[i] * y[i]; }
        int ok = by > 0;
        double ymax = 0; for (int i = 0; i < m; i++) if (fabs(y[i]) > ymax) ymax = fabs(y[i]);
        for (int j = 0; j < n && ok; j++) {
            double s = 0; for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) s += P->Ax[p] * y[P->Ai[p]];
            z[j] = -s / by;
            if (j < P->nf ? fabs(s) > 1e-7 * (1 + ymax) : s > 1e-7 * (1 + ymax)) ok = 0;
        }
        if (ok) {
            for (int i = 0; i < m; i++) y[i] /= by;
            for (int j = 0; j < n; j++) x[j] = 0;
            *pobj = 0; *dobj = 1;
        } else st = -1;
    } else if (st == 1 || st == 2) st = -1;
    else {
        double cx = 0, by = 0;
        for (int j = 0; j < n; j++) cx += P->c[j] * x[j];
        for (int i = 0; i < m; i++) by += P->b[i] * y[i];
        *pobj = cx; *dobj = by;
    }
    free(L.l); free(L.u); free(ray);
    return st;
}


/* ---------------------------------------------------------------- the state kept between solves (sxmodel.h)
 * (22 October 2026) For the re-solve interface (lpmodel.c): the problem as given, scaled once, with its basis, the
 * factorization and its updates, the values and the steepest-edge weights kept from one solve to the next, so that a
 * re-solve after a few changed bounds or added rows starts from the last basis with no setup. One thread: the solves of
 * the dual method do not run beside each other and the pricing by columns stays on the calling thread. */
#include "sxmodel.h"
struct SxModel {
    Sx S;
    int factored;                      /* the factorization is that of the current basis */
    int cost_changed;                  /* the costs changed since the last solve (the primal method then) */
    int xok, dok;                      /* the basic values and the reduced costs are those of the current basis, bounds and
                                          costs (an optimal solve leaves them so; a change of the bounds, the costs or the
                                          basis clears them): the next solve need not compute them again */
    double *ray, *pray;                /* the Farkas multipliers (m, as given) and the unbounded direction (N, scaled) */
    int have_farkas, have_pray, raysign;
    int cutoff;                        /* the last solve stopped at the cutoff (sxm_objective gives the bound then) */
};
SxModel *sxm_new(const LpProb *L, const double *cscale, const double *rscale, int mscaled, double costscale) {
    SxModel *M = xcalloc(1, sizeof *M);
    Sx *S = &M->S;
    if (cscale) { S->cs_in = cscale; S->rs_in = rscale; S->m_in = mscaled; S->costscale_in = costscale; }
    SimplexOpts o; memset(&o, 0, sizeof o);
    sx_setup(S, L, &o);
    S->cs_in = S->rs_in = NULL;
    S->nthreads = 1;
    M->ray = xcalloc((size_t)S->m + 1, sizeof(double)); M->pray = xcalloc((size_t)S->N + 1, sizeof(double));
    S->rayout = M->ray; S->prayout = M->pray;
    slack_basis(S);
    return M;
}
void sxm_free(SxModel *M) {
    if (!M) return;
    sx_free(&M->S); free(M->ray); free(M->pray); free(M);
}
void sxm_scaling(const SxModel *M, double *cscale, double *rscale, double *costscale) {
    const Sx *S = &M->S;
    if (cscale) memcpy(cscale, S->cscale, sizeof(double) * S->n);
    if (rscale) memcpy(rscale, S->rscale, sizeof(double) * S->m);
    if (costscale) *costscale = S->costscale;
}
void sxm_col_bounds(SxModel *M, int j, double l, double u) {
    Sx *S = &M->S; const double cs = S->cscale[j];
    S->lo0[j] = l <= -HUGEB ? -SX_INF : l / cs;
    S->up0[j] = u >= HUGEB ? SX_INF : u / cs;
    M->xok = 0;
}
void sxm_row_bounds(SxModel *M, int i, double rl, double ru) {
    Sx *S = &M->S; const int j = S->n + i; const double rs = S->rscale[i];
    S->lo0[j] = ru >= HUGEB ? -SX_INF : -ru * rs;
    S->up0[j] = rl <= -HUGEB ? SX_INF : -rl * rs;
    M->xok = 0;
}
void sxm_cost(SxModel *M, int j, double c) {
    Sx *S = &M->S;
    S->cost0[j] = (S->maxim ? -c : c) * S->cscale[j] / S->costscale;
    M->cost_changed = 1; M->dok = 0;
}
void sxm_get_basis(const SxModel *M, int *cstat, int *rstat) {
    const Sx *S = &M->S; const int n = S->n;
    if (cstat) for (int j = 0; j < n; j++) cstat[j] = S->pos[j] >= 0 ? SX_BASIC : S->lo0[j] == S->up0[j] ? SX_FIXED : S->move[j] > 0 ? SX_LOWER : S->move[j] < 0 ? SX_UPPER : SX_ZERO;
    if (rstat) for (int i = 0; i < S->m; i++) { const int j = n + i; rstat[i] = S->pos[j] >= 0 ? SX_BASIC : S->lo0[j] == S->up0[j] ? SX_FIXED : S->move[j] > 0 ? SX_UPPER : S->move[j] < 0 ? SX_LOWER : SX_ZERO; }
}
/* the working bounds and costs those of the problem (phase 1 boxes, perturbations, shifts and moved bounds gone) */
static void sxm_reset_work(Sx *S) {
    memcpy(S->cost, S->cost0, sizeof(double) * S->N); memset(S->shift, 0, sizeof(double) * S->N);
    memcpy(S->lo, S->lo0, sizeof(double) * S->N); memcpy(S->up, S->up0, sizeof(double) * S->N);
    S->perturbed = S->shifted = 0; S->pert_pending = 0; S->phase = 2;
}
int sxm_set_basis(SxModel *M, const int *cstat, const int *rstat) {
    Sx *S = &M->S;
    sxm_reset_work(S);
    given_basis(S, cstat, rstat);
    const int nrep = refactor(S);
    M->factored = 1; M->xok = M->dok = 0;
    return nrep;
}
int sxm_solve(SxModel *M, const SxmOpts *o, SxmRes *res) {
    Sx *S = &M->S;
    const int N = S->N;
    SxmRes dummy; if (!res) res = &dummy;
    memset(res, 0, sizeof *res);
    S->t0 = now(); S->tlim = o ? o->timelimit : 0; S->maxit = o ? o->maxit : 0; S->stop = o ? o->stop : NULL; S->nchk = 0;
    S->iter = S->it_dual = S->it_primal = S->it_ph1 = 0;
    const int nf0 = S->nfactor; const double w0 = S->work;
    /* (23 October 2026) the stall rule from the first window, and the same random numbers in every re-solve: a re-solve's
     * path depends on the basis, the bounds and the weights, not on the solves before it */
    S->lazy_pert = !(getenv("BRISK_SXMLAZYPERT") && atoi(getenv("BRISK_SXMLAZYPERT")) == 0);
    S->rng = 0x9E3779B97F4A7C15ULL; S->zw_it = S->zw_zero = 0; S->pert_level = 0;
    S->tolp = o && o->tol_primal > 0 ? o->tol_primal : 1e-7;
    S->told = o && o->tol_dual > 0 ? o->tol_dual : 1e-7;
    M->have_farkas = M->have_pray = 0; S->have_ray = 0; S->have_pray = 0; M->cutoff = 0;
    for (int j = 0; j < N; j++) if (S->lo0[j] > S->up0[j] + 1e-9 * (1 + fabs(S->lo0[j]))) { res->status = 1; return 1; }
    sxm_reset_work(S);
    /* the nonbasic variables at their bounds again: a bound that moved takes the variable with it; one that was fixed
     * or lost the bound it was at goes to the bound its reduced cost asks for */
    for (int j = 0; j < N; j++) {
        if (S->pos[j] >= 0) continue;
        const double l = S->lo[j], u = S->up[j]; const int mv = S->move[j];
        if (l == u) { S->move[j] = 0; S->x[j] = l; }
        else if (mv == 0 || (mv > 0 && l <= -SX_INF) || (mv < 0 && u >= SX_INF)) set_nonbasic(S, j, S->d[j]);
        else S->x[j] = mv > 0 ? l : u;
    }
    if (!M->factored) { refactor(S); M->factored = 1; M->xok = M->dok = 0; }
    /* (the values of the last optimal solve kept when nothing they depend on changed: a re-solve with nothing changed,
     * or after a cost change only, skips one or both passes over the matrix) */
    if (!M->xok) compute_primal(S);
    if (!M->dok || M->cost_changed) compute_dual(S);
    int fresh0 = 0;      /* (optimal as it stands: the values computed above are those to check at the end) */
    S->cut_on = o && o->cutoff_on; S->cut_val = S->cut_on ? (S->maxim ? -o->cutoff : o->cutoff) / S->costscale : 0;
    const int pinf = primal_infeasible(S);
    int dnb = 0, dbox = 0;      /* dual infeasibilities: of variables a bound flip cannot help, of boxed ones */
    for (int j = 0; j < N; j++) {
        if (S->pos[j] >= 0 || S->lo[j] == S->up[j]) continue;
        const double dj = S->d[j];
        const int bad = S->move[j] == 0 ? fabs(dj) > S->told : S->move[j] * dj < -S->told;
        if (bad) { if (is_boxed(S, j)) dbox++; else dnb++; }
    }
    int st;
    S->noverify = 1;
    /* (23 October 2026) the costs perturbed at the start of a solve by the dual method, as the first solve's are (the end
     * of the solve removes them; BRISK_SXMPERT=0: not). Unperturbed, the MIP developer's node LP of csched008 took 3,723
     * iterations from its basis and 6,551 from scratch, perturbed 930 and 1,564; long dives on neos-933966 (lpdive.c)
     * over 600 s -> 38 s; air05, binkar10_1, mas76, 30n20b8 about the same */
    static int sxmpert = -1; if (sxmpert < 0) sxmpert = !(getenv("BRISK_SXMPERT") && atoi(getenv("BRISK_SXMPERT")) == 0);
    /* (fresh: the loop entered directly uses the factorization and the values computed above) */
    if (!pinf && !dnb && !dbox) { st = 0; fresh0 = 1; }
    else if (M->cost_changed && !pinf) { S->fresh = 1; st = primal_simplex(S, 1); }
    else if (!dnb) {
        if (dbox) { for (int j = 0; j < N; j++) if (S->pos[j] < 0 && is_boxed(S, j)) set_nonbasic(S, j, S->d[j]); compute_primal(S); }
        S->phase = 2; S->fresh = 1;
        if (sxmpert) perturb_first(S);
        st = dual_loop(S);
    } else if (!pinf) { S->fresh = 1; st = primal_simplex(S, 1); }
    else { if (sxmpert) perturb_first(S); st = dual_simplex(S); if (st == 2) st = primal_simplex(S, 1); }
    /* (25 October 2026) an infeasibility the dual method could not prove (dual_loop's status 4): the primal method
     * decides */
    if (st == 4) { sxm_reset_work(S); normalize_nonbasic(S); compute_primal(S); compute_dual(S); S->fresh = 1; st = primal_simplex(S, 1); }
    /* an infeasibility found by the primal method's phase 1: the dual method from there, for the proof */
    if (st == 1 && !S->have_ray) { sxm_reset_work(S); normalize_nonbasic(S); compute_primal(S); const int s2 = dual_simplex(S); if (s2 == 1 || s2 == 3 || s2 == 6) st = s2; }
    /* the problem's own costs and bounds; the violations on the problem as given (as sx_solve_core), the updated
     * factorization first, a new one only when they exceed the target */
    for (int round = 0; round < 5 && st == 0; round++) {
        sxm_reset_work(S); normalize_nonbasic(S);
        if (!(round == 0 && fresh0)) { compute_primal(S); compute_dual(S); }
        double pv, dv; unscaled_violations(S, &pv, &dv);
        if ((pv > S->env_target || dv > S->env_target) && S->F.nupd > 0) { refactor(S); compute_primal(S); compute_dual(S); unscaled_violations(S, &pv, &dv); }
        if (pv <= S->env_target && dv <= S->env_target) break;
        if (round == 4) break;
        if (dv > S->env_target) { S->told = fmax(S->told * 0.5 * S->env_target / dv, 1e-12); st = primal_simplex(S, 1); }
        else { S->tolp = fmax(S->tolp * 0.5 * S->env_target / pv, 1e-12); st = dual_loop(S); }
    }
    S->fresh = 0; S->noverify = 0;
    M->cost_changed = 0;
    M->xok = M->dok = st == 0;
    M->cutoff = st == 7;
    M->have_farkas = st == 1 && S->have_ray; M->raysign = S->raymo < 0 ? -1 : 1;
    M->have_pray = st == 2 && S->have_pray;
    res->status = st; res->iters = S->iter; res->iters_dual = S->it_dual; res->iters_primal = S->it_primal;
    res->nfactor = S->nfactor - nf0; res->work = S->work - w0;
    return st;
}
void sxm_values(const SxModel *M, double *x, double *y, double *z) {
    const Sx *S = &M->S; const int n = S->n, m = S->m;
    const double sg = S->maxim ? -1.0 : 1.0;
    if (x) for (int j = 0; j < n; j++) x[j] = S->x[j] * S->cscale[j];
    if (y) for (int i = 0; i < m; i++) y[i] = sg * (-S->d[n + i]) * S->rscale[i] * S->costscale;
    if (z) for (int j = 0; j < n; j++) z[j] = sg * S->d[j] / S->cscale[j] * S->costscale;
}
double sxm_objective(const SxModel *M) {
    const Sx *S = &M->S; double s = 0;
    if (M->cutoff) return (S->maxim ? -S->cut_bound : S->cut_bound) * S->costscale;
    for (int j = 0; j < S->n; j++) s += S->cost0[j] * S->x[j];
    return (S->maxim ? -s : s) * S->costscale;
}
int sxm_farkas(const SxModel *M, double *y) {
    if (!M->have_farkas) return 0;
    for (int i = 0; i < M->S.m; i++) y[i] = M->raysign * M->ray[i];
    return 1;
}
int sxm_ray(const SxModel *M, double *d) {
    if (!M->have_pray) return 0;
    for (int j = 0; j < M->S.n; j++) d[j] = M->pray[j] * M->S.cscale[j];
    return 1;
}
void sxm_head(const SxModel *M, int *head) { memcpy(head, M->S.basic, sizeof(int) * M->S.m); }
/* row i of B_s^{-1} (scaled) in S->rho; returns the factor that takes it to the problem as given: the basic variable's
 * scale (a column's cscale, a logical's 1 / rscale) and the sign of the row activities as logicals (r = -s) */
static double sxm_rho(SxModel *M, int i) {
    Sx *S = &M->S;
    SxVec *rho = &S->rho; sxv_clear(rho);
    rho->val[i] = 1; rho->idx[0] = i; rho->cnt = 1;
    double dd = S->dens_rho; btran(S, rho, &dd, NULL);
    const int j = S->basic[i];
    return j < S->n ? S->cscale[j] : -1.0 / S->rscale[j - S->n];
}
void sxm_binv_row(SxModel *M, int i, double *out) {
    Sx *S = &M->S;
    const double f = sxm_rho(M, i);
    for (int k = 0; k < S->m; k++) out[k] = f * S->rho.val[k] * S->rscale[k];
    sxv_clear(&S->rho);
}
void sxm_binva_row(SxModel *M, int i, double *out) {
    Sx *S = &M->S;
    const double f = sxm_rho(M, i);
    SxVec *rho = &S->rho;
    if (rho->cnt < 0) sxv_reindex(rho);
    /* by the rows of A where row i of B^{-1} is not zero (as the pricing by rows) */
    for (int j = 0; j < S->n; j++) out[j] = 0;
    for (int t = 0; t < rho->cnt; t++) {
        const int k = rho->idx[t]; const double v = rho->val[k];
        if (v == 0) continue;
        for (int q = S->Rp[k]; q < S->Rp[k + 1]; q++) out[S->Ri[q]] += v * S->Rx[q];
    }
    for (int j = 0; j < S->n; j++) out[j] *= f / S->cscale[j];
    sxv_clear(rho);
}
struct SxmSaved {
    int m, N, factored;
    int *basic, *pos; signed char *move; double *x, *d, *w, *lo0, *up0;
    SxLU F;
};
SxmSaved *sxm_save(const SxModel *M) {
    const Sx *S = &M->S; const int m = S->m, N = S->N;
    SxmSaved *V = xcalloc(1, sizeof *V);
    V->m = m; V->N = N; V->factored = M->factored;
    V->basic = xmalloc(sizeof(int) * (m + 1)); memcpy(V->basic, S->basic, sizeof(int) * m);
    V->pos = xmalloc(sizeof(int) * N); memcpy(V->pos, S->pos, sizeof(int) * N);
    V->move = xmalloc(N + 8); memcpy(V->move, S->move, N);
    V->x = xmalloc(sizeof(double) * N); memcpy(V->x, S->x, sizeof(double) * N);
    V->d = xmalloc(sizeof(double) * N); memcpy(V->d, S->d, sizeof(double) * N);
    V->w = xmalloc(sizeof(double) * (m + 1)); memcpy(V->w, S->w, sizeof(double) * m);
    V->lo0 = xmalloc(sizeof(double) * N); memcpy(V->lo0, S->lo0, sizeof(double) * N);
    V->up0 = xmalloc(sizeof(double) * N); memcpy(V->up0, S->up0, sizeof(double) * N);
    if (M->factored) sxlu_copy(&V->F, &S->F);
    return V;
}
void sxm_restore(SxModel *M, const SxmSaved *V) {
    Sx *S = &M->S; const int m = S->m, N = S->N;
    if (V->m != m || V->N != N) return;
    memcpy(S->basic, V->basic, sizeof(int) * m); memcpy(S->pos, V->pos, sizeof(int) * N); memcpy(S->move, V->move, N);
    memcpy(S->x, V->x, sizeof(double) * N); memcpy(S->d, V->d, sizeof(double) * N); memcpy(S->w, V->w, sizeof(double) * m);
    memcpy(S->lo0, V->lo0, sizeof(double) * N); memcpy(S->up0, V->up0, sizeof(double) * N);
    if (V->factored) sxlu_copy(&S->F, &V->F);
    M->factored = V->factored; M->xok = M->dok = 0;
    rows_partition(S);
    sxm_reset_work(S);
}
void sxm_saved_free(SxmSaved *V) {
    if (!V) return;
    free(V->basic); free(V->pos); free(V->move); free(V->x); free(V->d); free(V->w); free(V->lo0); free(V->up0);
    sxlu_free(&V->F); free(V);
}
