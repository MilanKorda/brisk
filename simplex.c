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
#include "amd/amd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* an indexed max-heap over 0..cap-1: keys > 0 are in it; ties go to the lower index (the choice is the same as a
 * scan's that takes the first largest) */
typedef struct { int *h, *pos, n; double *key; } SxHeap;
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
    /* factorization and work vectors */
    SxLU F;
    SxVec rho, col, tau, bfrt, spike, upart, row;   /* row: the pivotal row over all N variables */
    double dens_col, dens_rho, dens_tau, dens_row;
    double dens_taulr;                 /* density of the weights' solve after L and R (when restricted to the column) */
    int taupart;                       /* the weights' solve restricted to the entering column's nonzeros (ftran2) */
    int *out;
    /* the leaving-row candidates: an indexed max-heap of the rows with a primal infeasibility, keyed by
     * infeasibility^2 / weight (ties: the lower row index), updated for the rows an iteration changes */
    int *heap, *hpos, nheap; double *hkey;
    int heap_ok;
    char *inI; int *Ilist, nI, Ifull;  /* the rows that may be primal infeasible (a superset; Ifull: unknown, all rows) */
    /* candidates of the ratio test */
    int *cand; double *ca, *cr; int ncand;
    /* bookkeeping */
    long iter, it_dual, it_primal, it_ph1;
    long nprice_row, nprice_col, nprice_dense; double sum_rho, sum_col, sum_rowc; long ncolstat;
    int nbshift, ndvxreset, it_given;
    SxHeap ch;                         /* the primal method's entering candidates: d_j^2 / Devex weight */
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
    long nchk;                         /* passes through check_stop (the clock read every 16th) */
    long xo_dnoblock;                  /* the crossover's dual pushes that changed no basis (a measurement) */
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

static void sx_setup(Sx *S, const LpProb *L, const SimplexOpts *o) {
    const int m = L->m, n = L->n, N = n + m;
    S->m = m; S->n = n; S->N = N;
    const int nz = L->Ap[n];
    S->Ap = xmalloc(sizeof(int) * (n + 1)); S->Ai = xmalloc(sizeof(int) * (nz + 1)); S->Ax = xmalloc(sizeof(double) * (nz + 1));
    /* copy A dropping explicit zeros */
    int e = 0;
    for (int j = 0; j < n; j++) {
        S->Ap[j] = e;
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) if (L->Ax[p] != 0) { S->Ai[e] = L->Ai[p]; S->Ax[e] = L->Ax[p]; e++; }
    }
    S->Ap[n] = e;
    S->rscale = xmalloc(sizeof(double) * (m + 1)); S->cscale = xmalloc(sizeof(double) * (n + 1));
    scale(S, !(o && o->noscale));
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
    if (getenv("SXDBG")) {
        double a = 1e300, b = 0, c = 1e300, d = 0;
        for (int j = 0; j < n; j++) { a = fmin(a, S->cscale[j]); b = fmax(b, S->cscale[j]); }
        for (int i = 0; i < m; i++) { c = fmin(c, S->rscale[i]); d = fmax(d, S->rscale[i]); }
        printf("scaling: columns %.1e .. %.1e, rows %.1e .. %.1e, costs / %.1e\n", a, b, c, d, S->costscale);
    }
    for (int j = 0; j < n; j++) S->cost0[j] /= S->costscale;
    S->lo = xmalloc(sizeof(double) * N); S->up = xmalloc(sizeof(double) * N); S->cost = xmalloc(sizeof(double) * N); S->shift = xcalloc(N, sizeof(double));
    memcpy(S->lo, S->lo0, sizeof(double) * N); memcpy(S->up, S->up0, sizeof(double) * N); memcpy(S->cost, S->cost0, sizeof(double) * N);
    S->basic = xmalloc(sizeof(int) * (m + 1)); S->pos = xmalloc(sizeof(int) * N); S->move = xmalloc(N);
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
    S->cand = xmalloc(sizeof(int) * N); S->ca = xmalloc(sizeof(double) * N); S->cr = xmalloc(sizeof(double) * N);
    S->tolp = o && o->tol_primal > 0 ? o->tol_primal : 1e-7;
    S->told = o && o->tol_dual > 0 ? o->tol_dual : 1e-7;
    S->dens_col = S->dens_rho = S->dens_tau = S->dens_row = S->dens_taulr = 0;
    S->taupart = !getenv("BRISK_SXTAUFULL");
    S->updlim = 1000;
    S->rng = 0x9E3779B97F4A7C15ULL;
}
static void sx_free(Sx *S) {
    free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Ri); free(S->Rx); free(S->Rnb); free(S->Rc); free(S->Cr); free(S->cscale); free(S->rscale);
    free(S->lo0); free(S->up0); free(S->cost0); free(S->lo); free(S->up); free(S->cost); free(S->shift);
    free(S->basic); free(S->pos); free(S->move); free(S->x); free(S->d); free(S->w); free(S->dvx);
    sxv_free(&S->rho); sxv_free(&S->col); sxv_free(&S->tau); sxv_free(&S->bfrt); sxv_free(&S->spike); sxv_free(&S->upart); sxv_free(&S->row);
    free(S->out); free(S->heap); free(S->hpos); free(S->hkey); free(S->inI); free(S->Ilist); if (S->ws2.stack) sxws_free(&S->ws2); free(S->cand); hp_free(&S->ch); free(S->c1); free(S->ca); free(S->cr);
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
static void ftran(Sx *S, SxVec *v, double *dens, SxVec *spike) {
    const double t = now();
    const long f0 = S->F.nflops;
    sxlu_ftran(&S->F, v, *dens, spike);
    S->work += (double)(S->F.nflops - f0) + (v->cnt >= 0 ? v->cnt : S->m);
    if (v->cnt >= 0) *dens = 0.95 * *dens + 0.05 * (double)v->cnt / S->m;
    S->t_ftran += now() - t;
}
/* the entering column's solve (with its spike) and the steepest-edge weights' solve. The weights' update reads B^{-1} rho_r
 * only where the entering column is nonzero: with a sparse column it is solved for after the column, at those positions
 * only (sxlu_ftran_part); otherwise in full, at the same time as the column when threads are there and the work pays for
 * them (each solve by one thread: the same bits as one after the other) */
#define SX_TAUPART_DENS 0.1   /* the entering column's density below which the weights' solve is restricted to it */
static void ftran2(Sx *S, SxVec *col, SxVec *tau) {
    const double t = now();
    const long f0 = S->F.nflops;
    const double work = (S->dens_col + S->dens_tau) * S->m + 0.02 * S->m;
    if (S->taupart && S->dens_col < SX_TAUPART_DENS) {
        sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
        if (col->cnt >= 0 && col->cnt < SX_TAUPART_DENS * S->m) {
            const int nlr = sxlu_ftran_part(&S->F, &S->ws2, tau, S->dens_taulr, col->idx, col->cnt);
            if (nlr >= 0) S->dens_taulr = 0.95 * S->dens_taulr + 0.05 * (double)nlr / S->m;
        } else {
            sxlu_ftran(&S->F, tau, S->dens_tau, NULL);
            if (tau->cnt >= 0) S->dens_tau = 0.95 * S->dens_tau + 0.05 * (double)tau->cnt / S->m;
        }
        if (col->cnt >= 0) S->dens_col = 0.95 * S->dens_col + 0.05 * (double)col->cnt / S->m;
        S->work += (double)(S->F.nflops - f0) + (col->cnt >= 0 ? col->cnt : S->m) + (tau->cnt >= 0 ? tau->cnt : S->m);
        S->t_ftran += now() - t;
        return;
    }
#ifdef _OPENMP
    if (S->nthreads > 1 && work >= SX_PAR_FTRAN && omp_get_max_threads() > 1) {
        SxWs *W2 = &S->ws2;
        W2->nflops = 0;
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
            #pragma omp section
            sxlu_ftran_ws(&S->F, W2, tau, S->dens_tau, NULL);
        }
        S->F.nflops += W2->nflops;
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
    S->t_ftran += now() - t;
}
static void btran(Sx *S, SxVec *v, double *dens, SxVec *upart) {
    const double t = now();
    const long f0 = S->F.nflops;
    sxlu_btran(&S->F, v, *dens, upart);
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
    S->work += S->F.factor_work;
    S->t_factor += now() - t;
    if (S->dbg) printf("  factor %d: m %d, L %ld U %ld, %.3f s\n", S->nfactor, S->m, (long)S->F.nnz_l, (long)S->F.nnz_u, now() - t);
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
    if (rowwork >= 0 && rowwork > S->N / 8) {
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
        int c = 0;
        const int N = S->N;
        for (int j = 0; j < N; j++) {
            const double v = rv[j];
            if (v == 0) continue;
            if (fabs(v) > 1e-14) row->idx[c++] = j; else rv[j] = 0;
        }
        row->cnt = c;
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
        #pragma omp parallel for schedule(dynamic, 1) if (S->Ap[n] >= SX_PAR_MIN)
        for (int b = 0; b < nblk; b++) {
            const int j1 = b * SX_PRICE_BLK, j2 = j1 + SX_PRICE_BLK < n ? j1 + SX_PRICE_BLK : n;
            for (int j = j1; j < j2; j++) {
                if (S->pos[j] >= 0) continue;
                double s = 0;
                for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s += S->Ax[p] * y[S->Ai[p]];
                if (fabs(s) > 1e-14) rv[j] = s;
            }
        }
        for (int j = 0; j < n; j++) if (rv[j] != 0) row->idx[c++] = j;
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
    const double pivtol = 1e-7, told = S->told;
    int nc = 0;
    for (int k = 0; k < row->cnt; k++) {
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
        S->cand[nc] = j; S->ca[nc] = a; S->cr[nc] = td / a; nc++;
    }
    *nflip = 0;
    if (nc == 0) return -1;
    double slope = fabs(delta);
    /* preselection (HiGHS): the candidates up to a step where the flips' total change exceeds the infeasibility, the
     * step growing tenfold from ten times the first Harris bound; the groups are then formed within them */
    {
        double th0 = SX_INF;
        for (int k = 0; k < nc; k++) { const double h = (S->cr[k] * S->ca[k] + told) / S->ca[k]; if (h < th0) th0 = h; }
        if (!(th0 >= 0)) th0 = 0;       /* (a reduced cost of the wrong sign beyond the tolerance: a step of zero; a negative
                                            start grew tenfold away from the candidates and never ended) */
        double sel = 10 * th0 + 1e-7, tot = 0;
        int w = 0;
        for (;;) {
            for (int k = w; k < nc; k++) {
                if (S->cr[k] <= sel) {
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
    int start = 0;
    for (;;) {
        /* the Harris bound over the remaining candidates, and the group within it */
        double th = SX_INF;
        for (int k = start; k < nc; k++) {
            const double h = (S->cr[k] * S->ca[k] + told) / S->ca[k];
            if (h < th) th = h;
        }
        /* move the group (ratio <= th) to the front of the remaining part */
        int g = start; double dsl = 0; int inf = 0;
        for (int k = start; k < nc; k++) {
            if (S->cr[k] <= th) {
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
            *nflip = start;
            return q;
        }
        slope -= dsl;
        start = g;
        if (start == nc) { *nflip = 0; return -1; }   /* all flipped and still infeasible */
    }
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
static int getenv_nostall(void) { static int v = -1; if (v < 0) v = getenv("BRISK_SXSTALL") == NULL; return v; }   /* (on trial: BRISK_SXSTALL=1; off until measured on the sets) */
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

/* a cost shift that makes the reduced cost of nonbasic j zero; recorded so that it is removed when j leaves the basis */
static void shift_cost(Sx *S, int j) {
    const double dj = S->d[j];
    S->cost[j] -= dj; S->shift[j] -= dj; S->d[j] = 0;
    S->nshift++; S->shifted = 1;
    if (fabs(dj) > S->shmax) S->shmax = fabs(dj);
}
/* remove the shift of a variable that leaves the basis (its reduced cost takes the change) */
static void shift_back(Sx *S, int j) {
    static int off = -1; if (off < 0) off = getenv("SXNOSHIFTBACK") != NULL;
    if (S->shift[j] == 0 || off) return;
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
    const double grow = (double)(F->nnz_r + (F->nnz_u - F->nnz_u0)) / F->nupd;
    const double a = 3.0 * (grow > 1 ? grow : 1);
    static double wmul = -1; if (wmul < 0) wmul = getenv("SXREFAC_W") ? atof(getenv("SXREFAC_W")) : 10.0;
    return F->nupd >= sqrt(2.0 * wmul * F->factor_work / a);
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
        if (need_rebuild || need_refactor(S)) { rebuild(S); need_rebuild = 0; S->heap_ok = 0; S->Ifull = 1; }
        if (S->pert_pending) { perturb_more(S); S->pert_pending = 0; }
        if (S->iter - last_log >= 1000) { log_line(S, S->phase == 1 ? "dual ph1" : "dual"); last_log = S->iter; }
        const int r = chuzr(S);
        if (r < 0) {
            if (S->F.nupd > 0 && !verified) { need_rebuild = 1; verified = 1; continue; }   /* confirm on fresh values */
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
        const int q = chuzc_dual(S, delta, &nflip);
        if (q < 0) {
            if (S->F.nupd > 0) { need_rebuild = 1; S->w[r] = wr; S->heap_ok = 0; continue; }
            /* a tiny infeasibility that no column can remove: the bound is moved to the value (restored at the end) */
            const double bd = delta < 0 ? S->lo[jl] : S->up[jl];
            if (fabs(delta) <= 1e-6 * (1 + fabs(bd))) {
                if (delta < 0) S->lo[jl] = xr; else S->up[jl] = xr;
                S->nbshift++; S->heap_ok = 0; continue;
            }
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
        sxv_copy(tau, rho);
        ftran2(S, col, tau);
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
        if (S->perturbed && S->phase == 2 && !getenv_nostall()) {
            S->zw_it++; if (!(mvq * S->d[q] > S->told)) S->zw_zero++;
            if (S->zw_it >= SX_STALL_WIN) {
                if (2 * S->zw_zero >= S->zw_it && S->pert_level < 3) {
                    S->pert_pending = 1;     /* (at the next iteration's start: this one's ratio test used the costs as they are) */
                    if (S->verbose >= 2) printf("  %ld of the last %ld steps of zero length: the costs perturbed again (x%g)\n", S->zw_zero, S->zw_it, S->pert_level == 0 ? 10.0 : S->pert_level == 1 ? 100.0 : 1000.0);
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
        update_dse(S, r, col, tau, wr);
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
static void pheap_rebuild(Sx *S) {
    hp_clear(&S->ch);
    for (int j = 0; j < S->N; j++) { const double k = pmerit(S, j); if (k > 0) hp_set(&S->ch, j, k); }
    S->ch_ok = 1;
}
/* the entering variable: the largest d_j^2 / w_j among the dual infeasible (Devex), from the heap */
static int chuzc_primal(Sx *S) {
    if (!S->ch_ok) pheap_rebuild(S);
    return S->ch.n ? S->ch.h[0] : -1;
}
static void pheap_touch_row(Sx *S) {
    if (!S->ch_ok) return;
    for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; hp_set(&S->ch, j, pmerit(S, j)); }
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
            refactor(S); compute_primal(S); devex_check(S);
            if (was2) shift_bounds_primal(S, NULL);      /* the drift of the recomputation absorbed in phase 2 */
            ph1 = primal_infeasible(S);
            if (ph1) compute_dual_ph1(S); else compute_dual(S);
            S->ch_ok = 0;
            need_rebuild = 0;
        }
        if (S->iter - last_log >= 1000) { log_line(S, ph1 ? "prim ph1" : "primal"); last_log = S->iter; }
        const int q = chuzc_primal(S);
        if (q < 0) {
            if (S->F.nupd > 0 && !verified) { need_rebuild = 1; verified = 1; continue; }
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
            if (ph1) { S->dvx[q] = fmin(S->dvx[q] * 1e3, 1e30); hp_set(&S->ch, q, pmerit(S, q)); continue; }
            if (S->F.nupd > 0) { need_rebuild = 1; continue; }
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
            hp_set(&S->ch, q, pmerit(S, q));
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
            hp_set(&S->ch, q, 0);
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
            if (!flip) hp_set(&S->ch, jl, pmerit(S, jl));
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
        } else if (!flip) hp_set(&S->ch, jl, pmerit(S, jl));
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
#define SX_TARGET_DEF 1e-7  /* largest violation accepted on the problem as read (relative: unscaled_violations); the
                               method's own tolerances (1e-7). 1e-9 until 21 October 2026: on 81 MIPLIB relaxations the
                               same results (4 differ by 4-11 iterations, largest dual violation 1.4e-8 -> 8.3e-8), and
                               supportcase19 from an optimal crossover basis no longer runs the primal method for 11,700
                               iterations to tighten 2.3e-9 (unsolved in 300 s -> 47 s at four threads) */
static double sx_target(void) { static double t = 0; if (t == 0) { const char *e = getenv("BRISK_SXTARGET"); t = e ? atof(e) : SX_TARGET_DEF; if (!(t > 0)) t = SX_TARGET_DEF; } return t; }
#define SX_TARGET sx_target()   /* (BRISK_SXTARGET: an experiment) */

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
    for (int j = 0; j < N; j++) {
        const double l = S->lo[j], u = S->up[j], v = xs[j];
        int al = l > -SX_INF && v - l <= 1e-9 * (1 + fabs(l)), au = u < SX_INF && u - v <= 1e-9 * (1 + fabs(u));
        if (dz && !al && !au) {
            if (l > -SX_INF && dz[j] > 0 && v - l < dz[j]) al = 1;
            else if (u < SX_INF && dz[j] < 0 && u - v < -dz[j]) au = 1;
        }
        if (al) xs[j] = l; else if (au) xs[j] = u;
        in[j] = !al && !au && !(l <= -SX_INF && u >= SX_INF && v == 0);
        at[j] = al ? 1 : au ? 2 : 0;
    }
    /* the basis: the inside logicals, the inside structurals, then the logicals of the other rows */
    int nb = 0;
    for (int j = 0; j < N; j++) S->pos[j] = -1;
    for (int i = 0; i < m && nb < m; i++) if (in[n + i]) { S->basic[nb] = n + i; S->pos[n + i] = nb++; }
    for (int j = 0; j < n && nb < m; j++) if (in[j]) { S->basic[nb] = j; S->pos[j] = nb++; }
    for (int i = 0; i < m && nb < m; i++) if (S->pos[n + i] < 0) { S->basic[nb] = n + i; S->pos[n + i] = nb++; }
    for (int i = 0; i < m; i++) S->w[i] = 1.0;
    for (int j = 0; j < N; j++) S->dvx[j] = 1.0;
    rows_partition(S);
    for (int j = 0; j < N; j++) S->x[j] = xs[j];
    S->keepx = 1;
    refactor(S);
    /* every nonbasic variable at its value (the replaced ones too), its status by where it is */
    for (int j = 0; j < N; j++) {
        if (S->pos[j] >= 0) { S->move[j] = 0; continue; }
        S->x[j] = xs[j];
        S->move[j] = S->lo[j] == S->up[j] ? 0 : xs[j] == S->lo[j] ? 1 : xs[j] == S->up[j] ? -1 : 0;
    }
    compute_primal(S);
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
    for (int pass = 0, left = 1; pass < 5 && left; pass++)
    for (int q = (left = 0); q < N; q++) {
        if (S->pos[q] >= 0) continue;
        const double l = S->lo[q], u = S->up[q], v = S->x[q];
        if (v == l || v == u || (l <= -SX_INF && u >= SX_INF && v == 0)) continue;
        left = 1;
        if (need_refactor(S)) { refactor(S); compute_primal(S); }
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
        for (int k = 0; k < col->cnt; k++) {
            const int i = col->idx[k]; const double a = dir * col->val[i];
            if (fabs(a) < SX_XO_PIVTOL) continue;
            const int j = S->basic[i]; const double xv = S->x[j];
            if (a > 0 ? S->lo[j] <= -SX_INF : S->up[j] >= SX_INF) continue;
            const double t = a > 0 ? (xv - S->lo[j] + ptol) / a : (S->up[j] - xv + ptol) / -a;
            if (t < thmax) thmax = t < 0 ? 0 : t;
        }
        int r = -1; double best = 0, thp = dist;
        if (thmax < dist) {
            for (int k = 0; k < col->cnt; k++) {
                const int i = col->idx[k]; const double a = dir * col->val[i];
                if (fabs(a) < SX_XO_PIVTOL) continue;
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
        btran(S, rho, &S->dens_rho, &S->upart);
        S->basic[r] = q; S->pos[q] = r; S->move[q] = 0; S->pos[jl] = -1;
        S->move[jl] = S->lo[jl] == S->up[jl] ? 0 : bound == S->lo[jl] ? 1 : -1;
        rows_enter(S, q); rows_leave(S, jl);
        if (sxlu_update(&S->F, r, &S->spike, &S->upart, ar)) { refactor(S); compute_primal(S); }
        sxv_clear(rho);
    }
    sxv_clear(col); sxv_clear(rho);
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
        xo_dual = nd1 == 0 && np1 > 0 && !getenv("BRISK_SXXOPRIMAL");
        if (S.verbose >= 1) printf("  after the pushes: %d primal infeasibilities (largest %.1e), %d dual (largest %.1e): the %s method\n", np1, mp, nd1, md, xo_dual ? "dual" : "primal");
        if (!xo_dual) st = primal_simplex(&S, pert);
    }
    if (xo && !xo_dual) ;
    else if (given_route == 1) st = 0;
    else if (given_route == 2) st = primal_simplex(&S, pert);
    else if (given_route == 3) { S.phase = 2; st = dual_loop(&S); }
    else if (method == 2 && !xo_dual) {
        st = primal_simplex(&S, pert);
    } else {
        if (pert && !xo_dual && getenv("BRISK_SXSTARTFLIP")) {     /* (on trial: off until measured on the sets) */
            const long nfl = dual_start_bounds(&S);
            if (S.verbose >= 2 && nfl) printf("  %ld columns of cost zero start at their other bound (less primal infeasibility)\n", nfl);
        }
        if (pert) perturb_costs(&S);
        st = dual_simplex(&S);
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
        if (pv <= SX_TARGET && dv <= SX_TARGET) break;
        if (round == 4) break;
        if (dv > SX_TARGET) {
            S.told = fmax(S.told * 0.5 * SX_TARGET / dv, 1e-12);
            st = primal_simplex(&S, pert);
        } else {
            if (pv > SX_TARGET) S.tolp = fmax(S.tolp * 0.5 * SX_TARGET / pv, 1e-12);
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
static int reduced_solve(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir) {
    const double ratio = getenv("BRISK_SXSIFT") ? atof(getenv("BRISK_SXSIFT")) : 100;   /* (experiments) */
    const int wide = !getenv("BRISK_SXNOSIFT") && R->n >= ratio * R->m && R->n >= (getenv("BRISK_SXSIFT") ? 1000 : 20000) && o2->method != 2;
    if (wide) return sift_solve(R, o2, xr, yr, csr, rsr, ir);
    return sx_solve_core(R, o2, xr, NULL, yr, NULL, csr, rsr, ir);
}

/* a tall problem (many more rows than columns that are not column singletons) through its dual: the basis of the dual
 * has one row per column of the problem, so its factorization and solves cost a fraction of those of the problem's
 * own basis (one row per row). The dual is built (sxpre_dualize), presolved and solved; its basis is carried back by
 * complementarity, and the problem solved from that basis (optimal as it is, or finished by a few iterations). Used
 * when the presolved dual has at most half the problem's rows; if the dual's solve does not end optimal, the problem
 * is solved directly (*done stays 0). */
static int dual_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    if (getenv("BRISK_SXNODUAL")) return 0;
    int est = 0;
    for (int j = 0; j < R->n; j++) {
        if (R->l[j] == R->u[j]) continue;
        const int boxed = R->l[j] > -HUGEB && R->u[j] < HUGEB;
        if (R->Ap[j + 1] - R->Ap[j] > 1 || boxed) est++;
    }
    if (2L * est > R->m) return 0;
    const double t0 = now();
    const int verbose = o2->verbose;
    LpProb D, DR; SxDual *M = NULL; SxPre *PD = NULL; char why[160] = "";
    sxpre_dualize(R, &D, &M);
    const int ps = sxpre_presolve(&D, &DR, &PD, verbose >= 1, why, sizeof why);
    if (ps || 2L * DR.m > R->m || DR.m == 0 || DR.n == 0) {
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
    int st = reduced_solve(&DR, &od, xdr, ydr, csdr, rsdr, &id);
    lp_free(&DR);
    if (st != 0) {
        if (verbose >= 1) printf("  the dual ended with status %d: the problem itself is solved\n", st);
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
    if (!sxpre_symmetry(R, &Q, &M, o2->verbose >= 1)) return 0;
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
static int ipm_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    const double t0 = now();
    LpProb Rc = *R;             /* (lp_to_std reads it only) */
    SedumiProb P; LpMap M; double *ub = NULL;
    lp_to_std(&Rc, 0, 0, &P, &M, &ub);
    if (M.status) { sedumi_free(&P); lp_map_free(&M); free(ub); return 0; }
    LpIpmOpts io; LpIpmInfo ii; memset(&io, 0, sizeof io); memset(&ii, 0, sizeof ii);
    io.nopdreg = 1;      /* (22 October 2026: the crossover wants the unregularized end point, LpIpmOpts) */
    if (!getenv("BRISK_LPNOOBJOFF") && P.n > 0) {     /* (22 October 2026: the objective's constant, for the method's gap; lpsolve.h) */
        double *z0 = calloc((size_t)P.n + 1, sizeof(double)), *x0 = calloc((size_t)Rc.n + 1, sizeof(double));
        const double v = lp_recover(&Rc, &M, z0, x0); io.objoff = Rc.maxim ? -v : v; free(z0); free(x0);
    }
    io.tol = getenv("BRISK_SXIPMTOL") ? atof(getenv("BRISK_SXIPMTOL")) : 1e-8; io.kcorr = -1; io.verbose = o2->verbose >= 2 ? 1 : 0;   /* (the tolerance's switch: a test, 21 October 2026) */
    if (o2->timelimit > 0) io.timelimit = o2->timelimit;
    io.stop = o2->stop;
    double *xs = calloc((size_t)P.n + 1, sizeof(double)), *ys = calloc((size_t)P.m + 1, sizeof(double)), *zs = calloc((size_t)P.n + 1, sizeof(double));
    const int ist = P.n > 0 ? lpipm_solve(P.m, P.n, P.nf, P.Ap, P.Ai, P.Ax, P.b, P.c, ub, &io, xs, ys, zs, &ii) : 0;
    const double tipm = now() - t0;
    if (o2->verbose >= 1) printf("  the interior-point method: status %d, %d iterations, %.2f s (primal %.1e, dual %.1e, gap %.1e; %d x %d, %ld entries, factor %ld, %d dense columns)\n", ist, ii.iters, tipm, ii.pinf, ii.dinf, ii.gap, P.m, P.n, (long)P.Ap[P.n], ii.lnz, ii.ndense);
    /* a point it stopped at without reaching its tolerance (stalled, or out of iterations) but nearly optimal is still the
     * crossover's: what it lacks the simplex method after the pushes removes, from a basis next to the optimal one
     * (triptim1: stalled at 61 iterations with errors 1e-7, then 142,000 simplex iterations from the start) */
    const int near = (ist == 3 || ist == 4) && ii.pinf <= SX_IPM_NEAR_INF && ii.dinf <= SX_IPM_NEAR_INF && ii.gap <= SX_IPM_NEAR_GAP;
    if (ist != 0 && near && o2->verbose >= 1) printf("  the interior-point method stopped near the optimum (%s): its point goes to the crossover\n", ii.why);
    if (ist != 0 && !near) {
        if (o2->verbose >= 1) printf("  the interior-point method did not end optimal: %s\n", ii.why);
        free(xs); free(ys); free(zs); sedumi_free(&P); lp_map_free(&M); free(ub); return 0;
    }
    double *x = calloc((size_t)R->n + 1, sizeof(double)), *y = calloc((size_t)R->m + 1, sizeof(double));
    lp_recover(R, &M, xs, x);
    for (int i = 0; i < R->m; i++) y[i] = M.rpos[i] >= 0 ? M.rsgn[i] * ys[M.rpos[i]] : 0;
    free(xs); free(ys); free(zs); sedumi_free(&P); lp_map_free(&M); free(ub);
    SimplexOpts o3 = *o2; o3.xstart = x; o3.ystart = y; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.method = 0;
    SimplexInfo i3;
    const int st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    free(x); free(y);
    if (o2->verbose >= 1) printf("  the crossover and the simplex method from the interior point: status %d, %ld pushes, %ld iterations (%.2f s)\n", st, i3.pushes, i3.iters, i3.time);
    *ir = i3; ir->time = now() - t0; ir->ipm_iters = ii.iters; ir->t_ipm = tipm;
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
typedef struct { int nr, nc, ns, nd; int *Cp, *Ci; char *isd; double lb, lnz, cost; } NeForm;
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
    F->lb += 40.0 * nz;             /* the products with A of the ten solves (below) */
}
/* one iteration's flops (lpsolve.c's ne_iter_cost): the factorization, the dense columns' solves and their Schur
 * complement, about ten solves and products with A. cost = 0: the pattern is too large to order. */
static void neform_cost(NeForm *F) {
    const int nr = F->nr, nc = F->nc; const int *Cp = F->Cp, *Ci = F->Ci;
    F->cost = 0;
    if (2.0 * F->lb > SX_AUTO_PATCAP) return;
    int *tp = calloc((size_t)nr + 2, sizeof(int)), *tj = malloc(sizeof(int) * ((size_t)Cp[nc] + 1));
    for (int j = 0; j < nc; j++) if (!F->isd[j]) for (int p = Cp[j]; p < Cp[j + 1]; p++) tp[Ci[p] + 1]++;
    for (int i = 0; i < nr; i++) tp[i + 1] += tp[i];
    { int *f = malloc(sizeof(int) * ((size_t)nr + 1)); memcpy(f, tp, sizeof(int) * nr);
      for (int j = 0; j < nc; j++) if (!F->isd[j]) for (int p = Cp[j]; p < Cp[j + 1]; p++) tj[f[Ci[p]]++] = j;
      free(f); }
    /* each row's neighbours through its columns: the pattern of M M' (both triangles, the diagonal included) */
    int *mark = malloc(sizeof(int) * ((size_t)nr + 1)), *ps = malloc(sizeof(int) * ((size_t)nr + 1));
    int *pi = malloc(sizeof(int) * ((size_t)(2.0 * F->lb) + nr + 1));
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
#define SX_RACE_START (0.25 * SX_AUTO_SETUP)
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
/* the test of dual_route's first step: a problem it would consider solving through its dual */
static int dual_route_applies(const LpProb *R) {
    if (getenv("BRISK_SXNODUAL")) return 0;
    int est = 0;
    for (int j = 0; j < R->n; j++) {
        if (R->l[j] == R->u[j]) continue;
        const int boxed = R->l[j] > -HUGEB && R->u[j] < HUGEB;
        if (R->Ap[j + 1] - R->Ap[j] > 1 || boxed) est++;
    }
    return 2L * est <= R->m;
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
    os.workbudget = fmax(SX_AUTO_B0 * (R->Ap[R->n] + R->m), SX_RACE_START); os.budget_cb = race_budget; os.budget_ctx = &C;
    /* all the free threads until the race starts, one while the interior-point method runs, all again if it fails */
    C.threads = nf; os.threads = &C.threads;
    pthread_mutex_init(&C.mu, NULL);
    /* a tall problem: the direct route on a thread of its own beside the route through the dual problem (either can be
     * far the better: ex9 over 300 s through its dual and 38 s directly, gfd-schedulen180f7d50m30k18 the other way round) */
    DirJob *DJ = NULL;
    const int tall = nf >= 3 && dual_route_applies(R) && !getenv("BRISK_SXNOROUTERACE");
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
    int dn = 0, st = dual_route(R, &os, xr, yr, csr, rsr, &is, &dn);
    if (!dn && !C.stop_ind) {
        if (DJ && !DJ->primal) {   /* the dual problem was not used: the direct route's job is this route's solve */
            pthread_join(C.thd, NULL);
            st = DJ->st; is = DJ->info;
            memcpy(xr, DJ->x, sizeof(double) * R->n); memcpy(yr, DJ->y, sizeof(double) * R->m);
            memcpy(csr, DJ->cs, sizeof(int) * R->n); memcpy(rsr, DJ->rs, sizeof(int) * R->m);
            if (DJ->won) C.stop_ind = 0;     /* (its own result: not a stop from outside) */
            free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); DJ = C.DJ = NULL;
        } else st = reduced_solve(R, &os, xr, yr, csr, rsr, &is);
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
int simplex_solve(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                  SimplexInfo *info) {
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
        int done = 0;
        /* (the time limit is the whole solve's: a route that gives up leaves the next one what is left) */
        #define SX_LEFT() (o && o->timelimit > 0 ? (o2.timelimit = o->timelimit - (now() - t0)) > 0 : 1)
        if (o && o->method == 3) st = ipm_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT() && !(o && o->nosymmetry) && !getenv("BRISK_SXNOSYM")) st = sym_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT() && (!o || o->method == 0)) st = auto_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT() && (!o || o->method != 2)) st = dual_route(&R, &o2, xr, yr, csr, rsr, &ir, &done);
        if (!done && SX_LEFT()) st = reduced_solve(&R, &o2, xr, yr, csr, rsr, &ir);
        if (!done && o && o->timelimit > 0 && o2.timelimit <= 0) { st = 6; ir.status = 6; }
        #undef SX_LEFT
        if (st == 0) sxpre_postsolve(P, xr, yr, csr, rsr, x, y, cs, rs);
        free(csr); free(rsr); free(xr); free(yr);
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

