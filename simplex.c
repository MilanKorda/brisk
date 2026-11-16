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
 * Primal simplex: from the start basis, steepest-edge pricing (Goldfarb and Reid 1977, Forrest and Goldfarb 1992; since
 * 4 November 2026, BRISK_SXPSE=0: Devex), from other bases Devex pricing (Harris 1973); Harris' two-pass ratio test with
 * bound shifting, phase 1 on the sum of infeasibilities; it also removes the dual infeasibilities left when the
 * perturbation of the costs is taken away.
 *
 * The basis factorization and its updates: simplex_lu.c. */
#include "simplex.h"
#include "simplex_int.h"
#include "simplex_pre.h"
#include "lpsolve.h"
#include "socp.h"
#include "amd/amd.h"
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static double g_sx_taboff = 0, g_sx_tabsign = 1;   /* (3 November 2026: the table's objective, c'x + c0 in the problem's sense) */
int brisk_stab_due(long it); void brisk_stab_row(long it, const char *what, double obj, double pinf, int ninf);   /* (main.c: the default log's table) */
void brisk_stab_end(long it, const char *what, double obj, double pinf, int ninf);
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
/* (27 October 2026) the settings through a table of the BRISK_ and SX entries of the environment (simplex_lu.c) */
const char *sx_getenv(const char *name);
void sx_env_refresh(void);
#define getenv(n) sx_getenv(n)
#define SX_PAR_FTRAN 2000     /* expected entries of the two solves of an iteration from which they run in parallel */

#define SX_INF 1e30
#define HUGEB 1e29     /* a bound beyond this is infinite */
#define SX_TARGET_DEF 1e-7  /* largest violation accepted on the problem as read (relative: unscaled_violations); the
                               method's own tolerances (1e-7). 1e-9 until 21 October 2026: on 81 MIPLIB relaxations the
                               same results (4 differ by 4-11 iterations, largest dual violation 1.4e-8 -> 8.3e-8), and
                               supportcase19 from an optimal crossover basis no longer runs the primal method for 11,700
                               iterations to tighten 2.3e-9 (unsolved in 300 s -> 47 s at four threads) */
/* (BRISK_SXTARGET: an experiment; read by sx_setup into env_target) */
/* (7 November 2026, the owner's decision; ISSUES 843) BRISK_LPGRBTOL=1 (lpsolve.h): Gurobi's stopping criteria. The primal
 * and dual tolerances 1e-6, absolute, on the scaled problem the method works on (FeasibilityTol, OptimalityTol); the final
 * rounds after the perturbations and shifts are removed check the same tolerances on the same scaled problem (they serve
 * the postsolve; no tighter target on the problem before the scaling); the postsolved basic solution is taken when it is
 * optimal on the problem as read within 1e-6 (sx_direct); one check on the problem as read keeps a plainly wrong result
 * from being reported optimal (SX_SANITY, simplex_solve) */
#define SX_GRB_TOL 1e-6
#define SX_SANITY 1e-3
#define SX_PSCALE_REL 1e-7  /* (11 November 2026) the primal simplex method's scaling: entries below this fraction of the largest of their row and of their column are left out of the geometric passes (the feasibility tolerance's level) */
#define SX_SCALE_REL 1e-9   /* (BRISK_LPGRBTOL bit 8: the scaling's geometric passes leave out entries below this fraction of their row's or column's largest; the level tiny_keep drops such entries at, where it may) */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "brisk: out of memory (simplex)\n"); exit(1); } return p; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* (27 October 2026) the time left of o's limit counted from t0, for a later phase given a copy of the options: 0 without a
 * limit, else at least 1e-3 (a phase started past the limit stops at once). Before, eight later phases were given the
 * whole limit again from their own start (the final solve after the route through the dual problem and after the bounded
 * substitution, the symmetry route's two solves, both crossovers of the interior-point route, the solve and the check on
 * the problem as read after the postsolve): savsched1 (LP) at one thread with 120 s ran its crossover past 330 s
 * (ISSUES 460) */
static double sx_tleft(const SimplexOpts *o, double t0) {
    if (!o || o->timelimit <= 0) return 0;
    const double l = o->timelimit - (now() - t0);
    return l > 1e-3 ? l : 1e-3;
}

/* an indexed max-heap over 0..cap-1: keys > 0 are in it; ties go to the lower index (the choice is the same as a
 * scan's that takes the first largest) */
/* (12 November 2026, a diagnostic: BRISK_SXENTDBG=1, log_line) */
static int g_entdbg = -1; static int g_priceunsc = -1; static int g_dsefull = -1; static int g_dserows = -1; static int g_wlb = -1;   /* (12-13 November 2026, BRISK_SXWLB: an experiment, the weights' update bounded below by (alpha_i / alpha_r)^2 w_r) */   /* (BRISK_SXPRICEUNSC, BRISK_SXDSEFULL: below) */   /* (BRISK_SXPRICEUNSC, below) */ static long g_ent_long, g_ent_long_d0, g_leave_long, g_ent_all;
typedef struct { int *h, *pos, n; double *key; } SxHeap;
/* (4 November 2026) the primal method from the start basis prices by steepest edge (BRISK_SXPSE=1) or by Devex (=0) */
struct SxBp { double t; int e; };     /* (a breakpoint of phase 1's ratio test: its step, its entry) */
static int cmp_bp(const void *a, const void *b);
#ifndef PH1LONG_DEFAULT
#define PH1LONG_DEFAULT 0
#endif
/* (5 November 2026, ISSUES 839) the primal method from the start basis: the ratio test's stability test (BRISK_SXPRT, the
 * threshold relative to the column's largest entry; 0: off; recommended 1e-6), the end of phase 2 at a basis seen at the
 * last verification (BRISK_SXPCYC; recommended 1), the step back to the last good basis when a factorization changes the
 * basic values by more than this many times their scale (BRISK_SXPBACKREL; 0: off; recommended 1) */
#ifndef PRT_DEFAULT
#define PRT_DEFAULT 1e-6   /* (5 November 2026: on, momentum2, LP, solved on 13 of 13 random sequences with either pricing; neutral in time on 40 LPs) */
#endif
#ifndef PCYC_DEFAULT
#define PCYC_DEFAULT 1
#endif
#ifndef PBACKREL_DEFAULT
#define PBACKREL_DEFAULT 1.0
#endif
#ifndef PSE_DEFAULT
#define SX_PSE_MAXROWS 10000
#define SX_PSE_MINROWNZ 5.0
#define PSE_DEFAULT 2   /* (10 November 2026) by the problem's size (BRISK_SXPSE=2, below; 0 Devex, 1 steepest edge always): the primal method alone, 4 threads, 67 LPs 0.798, the 53 where it acts 0.754 (the outside benchmark's production, network, workforce and diet LPs 0.61-0.79); slower: near-parallel LPs of 7,000 rows 1.30 and 1.53 (a longer path), co-100 1.21, neos-4647030-tutaki 1.09. Until then 0 (Devex): (4 November 2026) off: on 34 LPs 0.68 of the iterations but 1.05 of the time, minutedispatchstrategy 3.9 times slower, and momentum2 (LP) failing at the default random sequence (ISSUES 839); BRISK_SXPSE=1: on */
#endif
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
    double *icscale, *irscale;         /* (27 October 2026) their inverses (the factors are powers of two: the same bits as a division) */
    double *pert_u, pert_bigc; int pert_u_ok;     /* (27 October 2026, the re-solves) each column's perturbation factor, kept while the costs stay */
    int nbox_given, nbox_given_ok;     /* (11 November 2026) the boxed variables counted by sxm_solve's pass over the bounds, for its next perturb_first (ok: given) */
    double *lo0, *up0, *cost0;         /* the scaled problem's bounds and costs (N) */
    double *lo, *up, *cost;            /* working bounds and costs (phase 1 boxes, perturbation, shifts) */
    double *shift;                     /* the cost shift of each variable (removed when it leaves the basis) */
    double costscale;
    double cmax_read;           /* the largest cost as read (unscaled): the dual measure's reference (unscaled_violations) */
    int maxim;                         /* the problem as read maximizes (the costs here are negated) */
    /* state */
    int *basic;                        /* basic[i]: variable in position (row) i */
    int *pos;                          /* pos[j]: position of a basic variable, -1 nonbasic */
    signed char *move;                 /* nonbasic: +1 at lower (may increase), -1 at upper, 0 fixed or free */
    double *x;                         /* values of all variables */
    double *d;                         /* reduced costs (0 for basic) */
    double *w;                         /* dual steepest-edge weights (by position) */
    int *dsnz; long *dsst;            /* (13 November 2026, BRISK_SXDEGSPARSE=2) each variable's |B^-1 a_j| as last solved and the factorization it was solved on */
    int *ds_el, ds_nel, ds_elcap;      /* (13 November 2026) the qualifying members of the last group (for BRISK_SXDEGCHECK) */
    int ds_scratch;   /* (13 November 2026) the dual simplex method of a solve from the logicals' basis (not from a given basis, the crossover, the re-solves of sxm_solve, the clean-up rounds): where the sparsest-member rule acts */
    int ds_tq; double ds_tst; long ds_nsh; double ds_shmax;   /* (13 November 2026, BRISK_SXDEGSHIFT) the rule's member and the dual step it takes; the shifts made and the largest */
    long ds_grp, ds_chg, ds_solve, ds_hit, ds_memb, ds_oob, ds_oob0, ds_cr0, ds_stsm, ds_stlg, ds_inc, ds_shref; double ds_t, ds_ar, ds_strat;   /* (13 November 2026, the sparsest-member rule's cost at -v 2: groups examined, entering
                                        * choices changed, fresh solves, counts taken from the cache, eligible members, time) */
    long dc_n, dc_ag[8], dc_near[8], dc_sol[8]; double dc_lr[8]; int *dc_nz[3]; long *dc_it[3];   /* (13 November 2026, a diagnostic: BRISK_SXDEGCHECK=1, the estimates' agreement with the exact counts) */
    char *wk; int wlazy; SxVec wv;     /* (27 October 2026, the option BRISK_SXMLAZYW=1) whether a weight is known (exact up to
                                        * the updates); with wlazy, a candidate's unknown weight is computed when it comes
                                        * to the top of the leaving rows' heap (chuzr), in wv */
    double *dvx;                       /* Devex weights (by variable) */
    long cut_skip;                     /* (7 November 2026) after a cutoff claim the returned duals did not prove: no check before this iteration */
    int *gb_basic, *gb_pos; signed char *gb_move; double *gb_x, *gb_lo, *gb_up; int gb_ok, nback;   /* (4 November 2026) the primal method's last good basis (ISSUES 839) */
    double prt, prt_tol; long n_prt, n_prtfail, n_prtshift; int *prt_list, prt_n;   /* (5 November 2026, ISSUES 839) the primal ratio test's stability test (BRISK_SXPRT, BRISK_SXPRTTOL): its threshold and tolerance, its counts, phase 1's list of the rows it relaxed */
    double *px_old, pback_rel; int pcyc_on; unsigned long long vhash; long n_pcyc;   /* (5 November 2026) the basic values before a factorization, the change that counts as a basis that lost its conditioning (BRISK_SXPBACKREL); the end of phase 2 at a basis seen at the last verification (BRISK_SXPCYC) */
    double pv_minrel; int pv_nsmall, pv_n; double fl_pse, fl_prod, t_prod, n_alpha, n_row;   /* (statistics: the pivots' sizes relative to their column's largest (SXDBG=2), the steepest-edge pricing's work) */
    unsigned char *dvref; int dvx2, dvbad, pstab, nptrouble;   /* (27 October 2026; by default since 28 October, BRISK_SXDEVEX2=0: off) the reference framework of the Devex weights, the rule on its resets, the weights found too large since the last */
    int pse, pse_on, pse_route; long npar_pse;   /* (4 November 2026) the primal method's steepest-edge pricing: chosen (BRISK_SXPSE), in use, allowed in this call (the primal method from the start basis); its solves beside the row's */
    SxVec pv, row2; double dens_pv, t_pse;       /* (its solve B^{-T} alpha_q, the products a_j' v, the solve's density and time) */
    long iter_rdbg0;
    int dsefull_init;
    long n_dvxreset;                   /* (SXDUALDEVEX=2: the reference framework's resets) */                  /* (BRISK_SXDSEFULL: the starting weights set) */                   /* (a diagnostic, BRISK_SXROWDBG: the iteration at which to print, 0 at the set-up) */
    int ph1long, ph1flat; long n_ph1flat; double *bp_r, *bp_t; int *bp_i; struct SxBp *bp_o; long n_ph1pass;   /* (4 November 2026, BRISK_SXPH1LONG) phase 1's ratio test past the points where infeasible variables become feasible: its lists; the points passed */
    double tolp, told;
    double cpiv;                /* the dual ratio test's pivot tolerance (0: 1e-7; 1e-9 for the second test of an infeasibility claim) */
    int nfkretry, nfkfail;      /* infeasibility claims not proved: the second ratio test found a column / none (status 4) */
    unsigned char *tiny; int tiny_n, tiny_on;   /* (28 October 2026; ISSUES 629) during the clean-up rounds: the basic variables
                                * whose tiny violation no column could remove (the dual method moved their bound), which
                                * the rounds' measure then accepts */
    /* factorization and work vectors */
    SxLU F;
    SxVec rho, col, tau, bfrt, spike, upart, row;   /* row: the pivotal row over all N variables */
    double dens_col, dens_rho, dens_tau, dens_row;
    double sw_ew, sw_base; int sw_n, refac_meas;   /* the solves' work since the factorization (need_refactor; 24 October 2026) */
    double dens_taulr;                 /* density of the weights' solve after L and R (when restricted to the column) */
    int taupart;                       /* the weights' solve may be restricted to the entering column's nonzeros (ftran2; 2: always) */
    double tw_part, tw_full;           /* the work of the weights' restricted and full solves lately (ftran2) */
    long tw_last, n_taupart, n_taufull; /* the last trial of the solve not chosen; the counts of each (ftran2) */
    double dbg_wcol, dbg_wtau, dbg_ncol;    /* (a diagnostic, BRISK_SXFTDBG: the entering columns' solves' work and the weights' solves') */
    int price_dense_div;               /* the pricing by rows accumulates densely past N / this many entries (8; BRISK_SXPRICEDIV: an experiment) */
    int *out;
    /* the leaving-row candidates: an indexed max-heap of the rows with a primal infeasibility, keyed by
     * infeasibility^2 / weight (ties: the lower row index), updated for the rows an iteration changes */
    int *heap, *hpos, nheap; double *hkey;
    int *rrank;                        /* (14 November 2026, BRISK_SXRANKTIE=1) each basis position's rank in the rows' reverse Cuthill-McKee order: the pricing's ties by it */
    int heap_ok;
    char *inI; int *Ilist, nI, Ifull;  /* the rows that may be primal infeasible (a superset; Ifull: unknown, all rows) */
    /* candidates of the ratio test */
    int *cand; double *ca, *cr; int ncand;
    double *ck, *chm; int *cp, *cti; double *ctd;   /* (the ratio test's sorted groups: ratios, Harris suffix minima, order, work) */
    /* bookkeeping */
    long iter, it_dual, it_primal, it_ph1;
    long nprice_row, nprice_col, nprice_dense; double sum_rho, sum_col, sum_rowc; long ncolstat;
    int row_ord;                       /* the pivot row's list in increasing order, its values zero off the list (price) */
    int nbshift, ndvxreset, it_given;
    double cl_target;           /* (1 November 2026) > 0 in the clean-up rounds: the primal method stops at a refactorization once the
                                   unscaled violations are within it (ISSUES 815) */
    long n_pdegen, n_prescan;    /* (a diagnostic at -v 2) the primal method's steps of length zero and full rescans of the candidates */
    SxHeap ch;                         /* the primal method's entering candidates: d_j^2 / Devex weight */
    int pc_heap, pc_n, pc_cap, *pc_set, *pc_at, *pc_mj; double *pc_kv, *pc_mb, *pc_mk, pc_tau;   /* (the candidate list instead of the heap: pc_set_key) */
    int ch_ok;
    signed char *c1;                   /* phase 1 of the primal method: the cost (-1, 0, +1) of the basic variable of each row */
    double *rayout; int rayrow, have_ray; double raymo;
    int nfactor, nflips, nshift, nsh_free, nsh_reb, nsh_it, nflip_reb, nflip_it, nshiftback; double shmax;
    int updlim;
    double *wvar;                      /* (refactor) the weights by variable while the basis is reordered; zero outside it */
    double t0, tlim, t_factor, t_ftran, t_btran, t_price;
    int tdiag; double t_chuzr, t_chuzc, t_dupd, t_pupd, t_dse, t_heap, t_rebuild, t_loop, t_row;   /* (27 October 2026, a diagnostic at -v 2: the rest of the dual iteration's time; t_row the pivot row's computation) */
    long maxit;
    int verbose, dbg;
    double work, workbudget;           /* entries touched by solves, pricing and factorizations (deterministic); a limit */
    long pg_it; double pg_work; int pg_ninf, pg_ext;   /* (13 November 2026, BRISK_LPAUTOPROG) the last progress snapshot, the extensions made */
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
    int xo_pert_tol;                   /* (13 November 2026, BRISK_SXXOPERTTOL=1) the crossover's dual method: its perturbation below the dual tolerance */
    int lazy_main;                     /* (13 November 2026) lazy_pert set by BRISK_SXLAZYPERT for a whole solve, not by a re-solve */
    int xo_deg_only;                   /* (BRISK_SXXODEG=1: the rule nowhere else in that solve's process) */
    /* (14 November 2026, a test: BRISK_SXVFAC=P; bench/lp_r16/uccase7b) a factor on the leaving row's merit kept per basic variable: every P
     * iterations each structural column that became basic since the previous time and lies outside its bounds gets the factor
     * max(1, c sqrt(its entries)) (c BRISK_SXVFACC, 0.5), kept until it enters again; Gurobi's default dual pricing on uccase7 (LP) favours
     * such columns' rows so, and its long columns leave within a few dozen iterations (an agent's replay of its path) */
    double *vfac; long *vent; long vf_last; int vf_p; double vf_c;
    int *vden; int vf_d;               /* (BRISK_SXVFACD=1: only a column whose entering solve was denser than the running average) */
    unsigned char *vonce; int vf_once;  /* (BRISK_SXVFACONCE=1: the factor at most once a column in a solve) */
    int xo_deg;                        /* (14 November 2026, BRISK_SXXODEG=1) the crossover's dual method: the sparsest-member rule once its costs are perturbed */
    long n_tr_rowcol, n_tr_jump, n_tr_lost;   /* (11 November 2026) the primal method's numerical troubles by kind, for its log */
    int scale_prim;                    /* (11 November 2026) the primal simplex method's solve: the scaling leaves out the entries small in their row and their column */
    long nchk;                         /* passes through check_stop (the clock read every 16th) */
    long xo_dnoblock;                  /* the crossover's dual pushes that changed no basis (a measurement) */
    SimplexProgress *prog; long prog_base;   /* (4 November 2026) the race's log (simplex.h) */
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
    int fg_n;                          /* (chuzc_dual) the size of the final group of the last ratio test */
    double env_grpbf;   /* (final_group_bf: BRISK_SXGRPBF, the pivot threshold gamma; 0.1 by default; 0: off) */
    /* (7 November 2026, the MIP developer's idea; BRISK_SXINTPREF=gamma, off by default) the integer columns of a mixed-integer
     * program's relaxation (sxm_set_integer: n flags, NULL when none were given, as for an LP read from a file), and the
     * preference of the dual ratio test: among the members of its final group (the candidates within the Harris bound,
     * any of which keeps every reduced cost within the dual tolerance) whose |alpha| is at least gamma times the group's
     * largest, a row logical or a continuous column enters rather than an integer column (int_pref_pick) */
    const signed char *isint; double env_intpref; long n_intpref, n_ent_int, n_ent_oth, n_ip_small, n_ip_none;
    long env_stopit;                   /* (experiments, BRISK_SXSTOPIT) the dual simplex method stopped after this many iterations */
    FILE *trace;                       /* (experiments, BRISK_SXTRACE) one line per dual iteration */
    int trace_nf, trace_fl[512];        /* the iteration's bound flips (the first 16) */
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
    /* (8 November 2026; ISSUES 843, BRISK_LPGRBTOL bit 8) the geometric passes take a row's (a column's) smallest entry among
     * those at least SX_SCALE_REL of its largest: an entry below that is at the level of the rounding of the row's other
     * terms and says nothing of the row's scale, but 1 / sqrt(min max) follows it. ns1904248 (LP) holds cosines, 6.1e-17 for
     * cos(pi/2) in rows of entries near 1, in free columns (which tiny_keep cannot drop): its rows got factors near 1e8 and its
     * columns near 3e-8, and a reduced cost of 2.4e-8 on the scaled problem was 0.81 on the problem as read. Any factors give
     * an exact scaling (powers of two); only the choice changes */
    const double srel = brisk_lpgrbtol() & LPGRB_SXSCALE ? SX_SCALE_REL : 0.0;
    static double srel_env = -1; if (srel_env < 0) { const char *e_ = getenv("BRISK_SXSCALEREL"); srel_env = e_ ? atof(e_) : 0; }   /* (10 November 2026, a test: the level below which an entry is left out of the geometric passes, for every method) */
    /* (11 November 2026, by default; BRISK_SXPSCALEREL=0: off) the primal simplex method's solve: an entry below SX_PSCALE_REL of
     * the largest of its row and of its column is left out of the geometric passes. One such entry sets a row's factor
     * (1 / sqrt(min max)): on the near-parallel LPs (A stacked on A + 1e-8 I) each pair's second row was scaled up by about 1e4,
     * its slack's feasibility tolerance became 1e4 times finer than its twin's, and the method's bases with both rows active
     * were nearly singular: on Hnear_parallel_3500 (LP) 7,393 refactorizations forced by the check of the pivot from the row
     * against the pivot from the column, 186,135 iterations; with the rule 32 and 47,195. The dual simplex method shows no such
     * troubles on these LPs and its path only changes (better or worse), so the rule is the primal method's alone */
    static int psr = -1; if (psr < 0) { const char *e_ = getenv("BRISK_SXPSCALEREL"); psr = !(e_ && atoi(e_) == 0); }
    const double srel_p = srel_env > 0 ? srel_env : (psr && S->scale_prim ? SX_PSCALE_REL : 0);
    const double srelx_ = srel_p > 0 ? srel_p : srel;
    /* (8 November 2026, a test: BRISK_SXSCALEMODE bits) 1 no final equilibration of the columns, 2 every pass (no stop on a
     * pass that gains less than a tenth), 4 the rows equilibrated after the columns, 8 twenty passes */
    static int scmode = -1; if (scmode < 0) { const char *e_ = getenv("BRISK_SXSCALEMODE"); scmode = e_ ? atoi(e_) : 0; }
    /* (13 November 2026, an experiment: BRISK_SXSCALEMODE bit 128) Gurobi's scaling as found on satellites4-25's copy (LP;
     * bench/lp_r16/mystery/A): one geometric-mean pass (rows, then columns), then equilibration of the rows, then of the columns */
    const int npass = scmode & 128 ? 1 : scmode & 8 ? 20 : 8;
    /* (10 November 2026, a test: BRISK_SXSCALEMODE bit 16) the passes divide by the geometric mean of a row's (a column's) entries
     * instead of the geometric mean of its smallest and largest: one tiny entry among many does not set the factor */
    const int lmean = (scmode & 16) != 0;
    double *rls = lmean ? xmalloc(sizeof(double) * (m + 1)) : NULL; int *rcn = lmean ? xmalloc(sizeof(int) * (m + 1)) : NULL;
    /* (10 November 2026, BRISK_SXSCALEREL) an entry left out only when it is below the level in its row and in its column:
     * one small in its row but not in its column (a big-M row's ordinary coefficients, neos-4264598-oueme, LP) is brought
     * in line by its column's factor, while one small in both (1e-8 beside entries near 1: the near-parallel LPs) cannot be
     * by any factors and only drags them */
    const int both = srel_p > 0;
    double *cmx = both ? xmalloc(sizeof(double) * (n + 1)) : NULL, *rmx2 = both ? xmalloc(sizeof(double) * (m + 1)) : NULL;
    /* (13 November 2026, an experiment: BRISK_SXSCALEMODE bit 64) Ruiz's equilibration (2001) in place of the geometric passes and the
     * column equilibration: rows and columns divided at once by the square roots of their largest entries, repeated until every
     * row's and column's largest entry is within 1 +- 0.05 of 1 (at most 30 passes), then rounded to powers of two. Gurobi on
     * satellites4-25's copy (LP), its scaling off, on a copy so scaled: 29,673 iterations and 3.2-4.6 s, as with its own scaling
     * (31,879, 3.7 s); on a copy scaled as ours 35,541 and 8.5-16 s (bench/lp_r16/mystery/C) */
    if (scmode & 64) {
        double *cmxr = xmalloc(sizeof(double) * (n + 1));
        for (int pass = 0; pass < 30; pass++) {
            for (int i = 0; i < m; i++) rmax[i] = 0;
            double dev = 0;
            for (int j = 0; j < n; j++) { cmxr[j] = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > rmax[S->Ai[p]]) rmax[S->Ai[p]] = a; if (a > cmxr[j]) cmxr[j] = a; } }
            for (int i = 0; i < m; i++) if (rmax[i] > 0) dev = fmax(dev, fabs(1 - rmax[i]));
            for (int j = 0; j < n; j++) if (cmxr[j] > 0) dev = fmax(dev, fabs(1 - cmxr[j]));
            if (dev < 0.05) break;
            for (int i = 0; i < m; i++) if (rmax[i] > 0) r[i] /= sqrt(rmax[i]);
            for (int j = 0; j < n; j++) if (cmxr[j] > 0) c[j] /= sqrt(cmxr[j]);
        }
        free(cmxr);
        goto sc_round;
    }
    for (int pass = 0; pass < npass; pass++) {
        for (int i = 0; i < m; i++) { rmin[i] = SX_INF; rmax[i] = 0; }
        if (lmean) for (int i = 0; i < m; i++) { rls[i] = 0; rcn[i] = 0; }
        if (srelx_ > 0) for (int j = 0; j < n; j++) { if (both) cmx[j] = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > rmax[S->Ai[p]]) rmax[S->Ai[p]] = a; if (both && a > cmx[j]) cmx[j] = a; } }
        double gmin = SX_INF, gmax = 0;
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j];
            if (a == 0 || (a < srelx_ * rmax[S->Ai[p]] && (!both || a < srelx_ * cmx[j]))) continue;
            if (a < rmin[S->Ai[p]]) rmin[S->Ai[p]] = a;
            if (a > rmax[S->Ai[p]]) rmax[S->Ai[p]] = a;
            if (lmean) { rls[S->Ai[p]] += log(a); rcn[S->Ai[p]]++; }
            if (a < gmin) gmin = a;
            if (a > gmax) gmax = a;
        }
        const double ratio = gmax > 0 ? gmax / gmin : 1;
        if (pass > 0 && ratio > 0.9 * prev && !(scmode & 2)) break;
        prev = ratio;
        if (lmean) { for (int i = 0; i < m; i++) if (rcn[i] > 0) r[i] /= exp(rls[i] / rcn[i]); }
        else for (int i = 0; i < m; i++) if (rmax[i] > 0) r[i] /= sqrt(rmin[i] * rmax[i]);
        if (both) { for (int i = 0; i < m; i++) rmx2[i] = 0; for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > rmx2[S->Ai[p]]) rmx2[S->Ai[p]] = a; } }
        for (int j = 0; j < n; j++) {
            double mn = SX_INF, mx = 0;
            if (srelx_ > 0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > mx) mx = a; }
            double ls = 0; int cn = 0;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a == 0 || (a < srelx_ * mx && (!both || a < srelx_ * rmx2[S->Ai[p]]))) continue; if (a < mn) mn = a; if (a > mx) mx = a; if (lmean) { ls += log(a); cn++; } }
            if (lmean) { if (cn > 0) c[j] /= exp(ls / cn); }
            else if (mx > 0) c[j] /= sqrt(mn * mx);
        }
    }
    if (scmode & 128) {   /* (bit 128: the rows equilibrated before the columns) */
        for (int i = 0; i < m; i++) rmax[i] = 0;
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > rmax[S->Ai[p]]) rmax[S->Ai[p]] = a; }
        for (int i = 0; i < m; i++) if (rmax[i] > 0) r[i] /= rmax[i];
    }
    /* equilibrate columns to a largest entry of about 1 */
    if (!(scmode & 1)) for (int j = 0; j < n; j++) {
        double mx = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > mx) mx = a; }
        if (mx > 0) c[j] /= mx;
    }
    if (scmode & 4) {
        for (int i = 0; i < m; i++) rmax[i] = 0;
        for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double a = fabs(S->Ax[p]) * r[S->Ai[p]] * c[j]; if (a > rmax[S->Ai[p]]) rmax[S->Ai[p]] = a; }
        for (int i = 0; i < m; i++) if (rmax[i] > 0) r[i] /= rmax[i];
    }
sc_round:
    for (int i = 0; i < m; i++) r[i] = pow2round(r[i]);
    for (int j = 0; j < n; j++) c[j] = pow2round(c[j]);
    free(rmin); free(rmax); free(rls); free(rcn); free(cmx); free(rmx2);
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

static void cn_rcm_rows(const LpProb *L, int *q);
static void sx_setup(Sx *S, const LpProb *L, const SimplexOpts *o) {
    const int m = L->m, n = L->n, N = n + m;
    S->m = m; S->n = n; S->N = N;
    S->prog = o ? o->prog : NULL;
    if (S->prog) { S->prog_base = S->prog->it; S->prog->started = 1; }
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
    {   /* (14 November 2026, an experiment: BRISK_SXRANKTIE=1) the leaving row's ties broken by each position's rank in the rows' reverse
         * Cuthill-McKee order instead of its index: the problem kept in its order (the rows-only reordering, BRISK_SXRCM=2, changes what
         * the factorization and the row sifting see as well) */
        static int rt_ = -1; if (rt_ < 0) { const char *e_ = getenv("BRISK_SXRANKTIE"); rt_ = e_ && atoi(e_) != 0; }
        S->rrank = NULL;
        if (rt_ && m > 0) {
            LpProb T_; memset(&T_, 0, sizeof T_); T_.m = m; T_.n = n; T_.Ap = S->Ap; T_.Ai = S->Ai; T_.Ax = S->Ax;
            int *q_ = xmalloc(sizeof(int) * ((size_t)m + 1)); cn_rcm_rows(&T_, q_);
            S->rrank = xmalloc(sizeof(int) * ((size_t)m + 1)); for (int k = 0; k < m; k++) S->rrank[q_[k]] = k;
            free(q_);
        }
    }
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
    } else { S->scale_prim = o && o->method == 2; scale(S, !(o && o->noscale)); }
    for (int j = 0; j < n; j++) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->Ax[p] *= S->rscale[S->Ai[p]] * S->cscale[j];
    build_rows(S);
    S->icscale = xmalloc(sizeof(double) * (n + 1)); S->irscale = xmalloc(sizeof(double) * (m + 1));
    for (int j = 0; j < n; j++) S->icscale[j] = 1.0 / S->cscale[j];
    for (int i = 0; i < m; i++) S->irscale[i] = 1.0 / S->rscale[i];
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
        double c0 = 0; for (int j = 0; j < n; j++) c0 = fmax(c0, fabs(L->c[j]));
        S->cmax_read = c0;
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
    S->x = xcalloc(N, sizeof(double)); S->d = xcalloc(N, sizeof(double)); S->w = xmalloc(sizeof(double) * (m + 1)); S->dvx = xmalloc(sizeof(double) * N); S->dvref = xcalloc(N, 1);
    /* (28 October 2026) both rules of the primal method on by default (BRISK_SXDEVEX2=0, BRISK_SXPSTAB=0: off): over 211 LPs
     * (the tester's generator, Netlib, Kennington; one thread, 30 s) 0.910 of the time in the geometric mean, 33 faster
     * and 11 slower by more than 30 %, 4 more solved (a_near_parallel_05, a_dense_columns_05, n_multicommodity_07, dfl001) */
    { const char *e = getenv("BRISK_SXDEVEX2"); S->dvx2 = !(e && atoi(e) == 0); e = getenv("BRISK_SXPSTAB"); S->pstab = !(e && atoi(e) == 0); }
    S->wk = xmalloc((size_t)m + 1); memset(S->wk, 1, (size_t)m + 1); S->wlazy = 0; sxv_init(&S->wv, m);
    S->wvar = xcalloc(N + 1, sizeof(double));
    sxv_init(&S->rho, m); sxv_init(&S->col, m); sxv_init(&S->tau, m); sxv_init(&S->bfrt, m); sxv_init(&S->spike, m); sxv_init(&S->upart, m);
    sxv_init(&S->row, N);
    { const char *e = getenv("BRISK_SXPH1LONG"); S->ph1long = e ? atoi(e) : PH1LONG_DEFAULT;
      e = getenv("BRISK_SXPH1FLAT"); S->ph1flat = !(e && atoi(e) == 0);   /* (on by default since 7 November 2026: the default route, 4 threads, 23 problem LPs 1.004, 40 development LPs 1.012, 46 rescaled copies 1.005, the iterations the same on 108 of 109; lectsched-1's root LP, MILP, fewer fractional columns, ISSUES 842); BRISK_SXPH1FLAT=0: off */
      if (S->ph1long) { S->bp_r = xmalloc(sizeof(double) * (2 * (size_t)m + 2)); S->bp_t = xmalloc(sizeof(double) * (2 * (size_t)m + 2)); S->bp_i = xmalloc(sizeof(int) * (2 * (size_t)m + 2)); S->bp_o = xmalloc(sizeof(struct SxBp) * (2 * (size_t)m + 2)); } }
    /* (10 November 2026, BRISK_SXPSE=2, the default: by the problem's size) steepest edge where its extra solve per iteration can pay
     * for itself: at most 10,000 rows (on 39 LPs of the primal method's random set, one thread: 0.847 below, 1.180 above; its
     * losers minutedispatchstrategy, prosna, cryptanalysis, ken-13 have 10,800 rows or more) and at least 5 nonzeros a row
     * (where rows have 2-4, the iterations are so cheap that the extra solve doubles their work for the same path: the outside
     * benchmark's triangular, dispatch and inventory LPs 1.05-1.8 times slower) */
    { const char *e = getenv("BRISK_SXPSE"); const int v = e ? atoi(e) : PSE_DEFAULT;
      S->pse = v == 2 ? (m <= SX_PSE_MAXROWS && (double)S->Ap[n] >= SX_PSE_MINROWNZ * (double)m) : v != 0; }
    { const char *e = getenv("BRISK_SXPRT"); S->prt = e ? atof(e) : PRT_DEFAULT; e = getenv("BRISK_SXPRTTOL"); S->prt_tol = e ? atof(e) : 100; }
    { const char *e = getenv("BRISK_SXPCYC"); S->pcyc_on = e ? atoi(e) : PCYC_DEFAULT; }
    { const char *e = getenv("BRISK_SXPBACKREL"); S->pback_rel = e ? atof(e) : PBACKREL_DEFAULT; }
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
    S->pc_mb = NULL; S->pc_mj = NULL; S->pc_mk = NULL;
    /* (8 entries beyond N: the ratio test's vector stores write a full block at the end of the candidates) */
    S->cand = xmalloc(sizeof(int) * (N + 8)); S->ca = xmalloc(sizeof(double) * (N + 8)); S->cr = xmalloc(sizeof(double) * (N + 8));
    S->ck = xmalloc(sizeof(double) * (N + 1)); S->chm = xmalloc(sizeof(double) * (N + 1)); S->cp = xmalloc(sizeof(int) * (N + 1));
    S->cti = xmalloc(sizeof(int) * (N + 1)); S->ctd = xmalloc(sizeof(double) * (N + 1));
    S->tolp = o && o->tol_primal > 0 ? o->tol_primal : brisk_lpgrbtol() & LPGRB_SXTOL ? SX_GRB_TOL : 1e-7;
    S->told = o && o->tol_dual > 0 ? o->tol_dual : brisk_lpgrbtol() & LPGRB_SXTOL ? SX_GRB_TOL : 1e-7;
    { const char *e = getenv("BRISK_SXTOLP"); if (e) S->tolp = atof(e); e = getenv("BRISK_SXTOLD"); if (e) S->told = atof(e); }   /* (10 November 2026, an experiment: each tolerance alone) */
    S->dens_col = S->dens_rho = S->dens_tau = S->dens_row = S->dens_taulr = 0;
    S->taupart = getenv("BRISK_SXTAUFULL") ? 0 : getenv("BRISK_SXTAUPART") ? 2 : 1;   /* (switches: the full solve always, the restricted one always) */
    S->price_dense_div = getenv("BRISK_SXPRICEDIV") ? atoi(getenv("BRISK_SXPRICEDIV")) : 8; if (S->price_dense_div < 1) S->price_dense_div = 8;
    S->updlim = getenv("BRISK_SXUPDLIM") ? atoi(getenv("BRISK_SXUPDLIM")) : 1000;   /* (the switch: an experiment, 24 October 2026) */
    S->refac_meas = !(getenv("BRISK_SXREFACMEAS") && atoi(getenv("BRISK_SXREFACMEAS")) == 0);
    S->rng = 0x9E3779B97F4A7C15ULL;
    { const char *e = getenv("BRISK_SXSEED"); if (e && atoi(e) > 0) { S->rng ^= (unsigned long long)atoi(e) * 0xBF58476D1CE4E5B9ULL; for (int k_ = 0; k_ < 4; k_++) urand(S); } }   /* (experiments: another random sequence) */
    { const char *e = getenv("SXREFAC_W"); S->env_refac_w = e ? atof(e) : 10.0; }
    { const char *e = getenv("SXSLACKPREF"); S->env_slackpref = e ? atof(e) : 0; }
    if (g_entdbg < 0) g_entdbg = getenv("BRISK_SXENTDBG") != NULL;
    if (g_priceunsc < 0) { const char *e_ = getenv("BRISK_SXPRICEUNSC"); g_priceunsc = e_ && atoi(e_) != 0; }
    if (g_dsefull < 0) { const char *e_ = getenv("BRISK_SXDSEFULL"); g_dsefull = e_ && atoi(e_) != 0; }
    if (g_dserows < 0) { const char *e_ = getenv("BRISK_SXDSEROWS"); g_dserows = e_ && atoi(e_) != 0; }
    if (g_wlb < 0) { const char *e_ = getenv("BRISK_SXWLB"); g_wlb = e_ && atoi(e_) != 0; }
    { const char *e = getenv("SXDUALDEVEX"); S->env_ddevex = e ? atoi(e) : 0; }
    { const char *e = getenv("SXSTALLOBJ"); S->env_stallobj = e ? atoi(e) : 0; S->stall_obj = NAN; }
    { const char *e = getenv("BRISK_SXTARGET"); S->env_target = e ? atof(e) : SX_TARGET_DEF; if (!(S->env_target > 0)) S->env_target = SX_TARGET_DEF; }
    S->env_noshiftback = getenv("SXNOSHIFTBACK") != NULL;
    S->env_nostall = getenv("BRISK_SXSTALL") == NULL;
    { const char *e = getenv("BRISK_SXGRPBF"); S->env_grpbf = e ? atof(e) : 0.1; }   /* (0.1 by default since 7 November 2026: the default route, 23 problem LPs 0.983, the 47-LP dual simplex set 0.988, 40 fresh held-out LPs 0.983 (159.4 -> 112.7 s); BRISK_SXGRPBF=0: off) */
    { const char *e = getenv("BRISK_SXINTPREF"); S->env_intpref = e ? atof(e) : 0; if (!(S->env_intpref > 0)) S->env_intpref = 0; if (S->env_intpref > 1) S->env_intpref = 1; }
    { const char *e = getenv("BRISK_SXSTOPIT"); S->env_stopit = e ? atol(e) : 0; }
    /* (5 November 2026, experiments on ISSUES 835) BRISK_SXTRACE=<file>: a line per dual iteration (appended: the dual
     * simplex method of every solve of the run): the iteration, the leaving row and variable, the entering variable and its
     * column's entries, the leaving infeasibility, the row's steepest-edge weight, the pivot, the dual and primal steps, the
     * entries of the entering column's and the weights' solves, the final group's size, the bound flips and the first 16
     * flipped variables (variables: columns 0..n-1, then the logicals n + row) */
    S->trace = getenv("BRISK_SXTRACE") ? fopen(getenv("BRISK_SXTRACE"), "a") : NULL;
}
static void sx_free(Sx *S) {
    if (S->trace) { fclose(S->trace); S->trace = NULL; }
    free(S->dsnz); free(S->dsst); S->dsnz = NULL; S->dsst = NULL;
    free(S->ds_el); S->ds_el = NULL; S->ds_nel = S->ds_elcap = 0;
    for (int t_ = 0; t_ < 3; t_++) { free(S->dc_nz[t_]); free(S->dc_it[t_]); S->dc_nz[t_] = NULL; S->dc_it[t_] = NULL; }
    free(S->Ap); free(S->Ai); free(S->Ax); free(S->Rp); free(S->Ri); free(S->Rx); free(S->Rnb); free(S->Rc); free(S->Cr); free(S->cscale); free(S->rscale); free(S->icscale); free(S->irscale); free(S->pert_u);
    free(S->lo0); free(S->up0); free(S->cost0); free(S->lo); free(S->up); free(S->cost); free(S->shift);
    free(S->prt_list); free(S->px_old); free(S->gb_basic); free(S->gb_pos); free(S->gb_move); free(S->gb_x); free(S->gb_lo); free(S->gb_up);
    free(S->tiny); free(S->basic); free(S->pos); free(S->move); free(S->x); free(S->d); free(S->w); free(S->dvx); free(S->dvref); free(S->wk); free(S->wvar); sxv_free(&S->wv);
    free(S->rrank); S->rrank = NULL;
    free(S->vfac); free(S->vent); free(S->vden); free(S->vonce); S->vfac = NULL; S->vent = NULL; S->vden = NULL; S->vonce = NULL;   /* (BRISK_SXVFAC) */
    sxv_free(&S->rho); sxv_free(&S->col); sxv_free(&S->tau); sxv_free(&S->bfrt); sxv_free(&S->spike); sxv_free(&S->upart); sxv_free(&S->row); free(S->bp_r); free(S->bp_t); free(S->bp_i); free(S->bp_o); if (S->pv.val) { sxv_free(&S->pv); sxv_free(&S->row2); }
    free(S->out); free(S->heap); free(S->hpos); free(S->hkey); free(S->inI); free(S->Ilist); if (S->ws2.stack) sxws_free(&S->ws2); free(S->cand); hp_free(&S->ch); free(S->pc_set); free(S->pc_kv); free(S->pc_at); free(S->pc_mb); free(S->pc_mj); free(S->pc_mk); free(S->c1); free(S->ca); free(S->cr); free(S->ck); free(S->chm); free(S->cp); free(S->cti); free(S->ctd);
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
    memset(S->wk, 1, (size_t)S->m); S->wlazy = 0;
    for (int j = 0; j < S->N; j++) S->dvx[j] = 1.0;
}
static void rows_partition(Sx *S);
static int refactor(Sx *S);
static void compute_primal(Sx *S);
static void compute_dual(Sx *S);
static void compute_primal_dual(Sx *S);
static void sx_sort_by_key_desc(int *ord, int k, const double *key);
/* (4 November 2026) the free columns into the basis of logicals before the dual method starts (a crash, as Gurobi's "handle
 * free variables"): a nonbasic free column with a nonzero reduced cost is dual infeasible, and from the logicals' basis the
 * dual method's phase 1 brought them in one at a time inside boxes of [-1000, 1000]: neos-3754224-navua (LP, 65,010 free
 * columns in its standard form) spent 128,712 of 163,853 iterations in phase 1 within 40 s and did not finish in 150 s
 * (Gurobi's dual simplex method 95 s). Each free column, sparsest first, takes the row of its largest entry (at least a
 * tenth of the column's largest) whose logical is still basic and has a finite side (a free row's logical would itself be
 * dual infeasible nonbasic); that logical leaves at its bound. The factorization replaces the columns that turn out
 * dependent by logicals. Returns the columns brought in. Not on by default (BRISK_SXFREECRASH=1: on): navua's phase 1 128,712 ->
 * 70,111 iterations but still past 150 s, the iterations dearer; on the 47 LPs of the dual simplex comparison 1.013 in
 * geometric mean, hgms62 (LP) 36.7 -> 60.5 s, momentum1 and satellites2-25 6-13 % faster (TRICKS 8) */
static int free_crash(Sx *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_SXFREECRASH"); on = e && atoi(e) != 0; }   /* (off: measured worse, TRICKS 8) */
    if (!on) return 0;
    const int n = S->n, m = S->m;
    int nf = 0;
    for (int j = 0; j < n; j++) if (S->pos[j] < 0 && S->lo[j] <= -SX_INF && S->up[j] >= SX_INF && S->Ap[j + 1] > S->Ap[j]) nf++;
    if (nf == 0) return 0;
    int *cand = xmalloc(sizeof(int) * (size_t)nf); double *key = xmalloc(sizeof(double) * ((size_t)n + 1)); int k = 0;
    for (int j = 0; j < n; j++) if (S->pos[j] < 0 && S->lo[j] <= -SX_INF && S->up[j] >= SX_INF && S->Ap[j + 1] > S->Ap[j]) { cand[k++] = j; key[j] = -(double)(S->Ap[j + 1] - S->Ap[j]); }
    sx_sort_by_key_desc(cand, nf, key);      /* (the fewest entries first; ties by index) */
    unsigned char *used = xcalloc((size_t)m + 1, 1);
    int nc = 0;
    for (int t = 0; t < nf; t++) {
        const int j = cand[t];
        double cmax = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) cmax = fmax(cmax, fabs(S->Ax[p]));
        int bi = -1; double ba = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p]; const double a = fabs(S->Ax[p]);
            if (used[i] || S->basic[i] != n + i || a < 0.1 * cmax) continue;
            if (S->lo[n + i] <= -SX_INF && S->up[n + i] >= SX_INF) continue;
            if (a > ba) { ba = a; bi = i; }
        }
        if (bi < 0) continue;
        used[bi] = 1;
        S->basic[bi] = j; S->pos[j] = bi; S->move[j] = 0;
        S->pos[n + bi] = -1; set_nonbasic(S, n + bi, S->cost[n + bi]);
        nc++;
    }
    free(cand); free(key); free(used);
    if (nc) {
        for (int i = 0; i < m; i++) S->w[i] = 1.0;
        memset(S->wk, 1, (size_t)m); S->wlazy = 0;
        rows_partition(S);
        refactor(S); compute_primal_dual(S);
    }
    return nc;
}
/* (4 November 2026, on trial: BRISK_SXCRASH=1) the primal method's starting basis by Bixby's crash (Bixby 1992,
 * "Implementing the simplex method: the initial basis"): from the logicals' basis, structural columns replace logicals
 * along a nearly triangular pattern. The columns by preference: free ones first (a free column wants to be basic), then
 * those with one bound, then boxed ones, the sparser first within a class (ties by a small part of the cost); a column
 * takes the row of its largest entry (at least 0.99 of the column's largest) among the rows not taken whose logical is not
 * free, the rows of equations first (their fixed logicals make the logicals' basis infeasible whenever the row is not
 * met), and is accepted only when its entries in the rows already taken are at most 1 % of those rows' pivots (the basis
 * stays nearly triangular, so well conditioned). Returns the columns brought in. stockholm (LP): the primal method's
 * phase 1 took 43,974 of its 65,768 iterations (38 s; Gurobi's primal method 3,902 iterations in all) */
static int primal_crash(Sx *S) {
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_SXCRASH"); on = e && atoi(e) != 0; }
    if (!on) return 0;
    const int n = S->n, m = S->m;
    double cm = 0; for (int j = 0; j < n; j++) cm = fmax(cm, fabs(S->cost[j]));
    int *cand = xmalloc(sizeof(int) * ((size_t)n + 1)); double *key = xmalloc(sizeof(double) * ((size_t)n + 1)); int nc = 0;
    for (int j = 0; j < n; j++) {
        if (S->pos[j] >= 0 || S->Ap[j + 1] == S->Ap[j]) continue;
        const int lf = S->lo[j] > -SX_INF, uf = S->up[j] < SX_INF;
        if (lf && uf && S->lo[j] == S->up[j]) continue;
        const double cls = !lf && !uf ? 0.0 : (lf != uf) ? 1.0 : 2.0;
        key[j] = -(cls * 1e7 + (double)(S->Ap[j + 1] - S->Ap[j]) + (cm > 0 ? 0.5 * fabs(S->cost[j]) / cm : 0.0));
        cand[nc++] = j;
    }
    sx_sort_by_key_desc(cand, nc, key);       /* (the most preferred first) */
    double *rpiv = xcalloc((size_t)m + 1, sizeof(double));
    int done = 0;
    for (int t = 0; t < nc; t++) {
        const int j = cand[t];
        double cmax = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) cmax = fmax(cmax, fabs(S->Ax[p]));
        int bi = -1, beq = 0; double ba = 0, worst = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p]; const double a = fabs(S->Ax[p]);
            if (rpiv[i] > 0) { const double r = a / rpiv[i]; if (r > worst) worst = r; continue; }
            if (a < 0.99 * cmax || S->basic[i] != n + i) continue;
            const int L = n + i;
            if (S->lo[L] <= -SX_INF && S->up[L] >= SX_INF) continue;        /* (a free row's logical stays) */
            const int eq = S->lo[L] == S->up[L];
            if (bi < 0 || (eq && !beq) || (eq == beq && a > ba)) { bi = i; ba = a; beq = eq; }
        }
        if (bi < 0 || worst > 0.01) continue;
        rpiv[bi] = ba;
        S->basic[bi] = j; S->pos[j] = bi; S->move[j] = 0;
        S->pos[n + bi] = -1; set_nonbasic(S, n + bi, S->cost[n + bi]);
        done++;
    }
    free(cand); free(key); free(rpiv);
    if (done) {
        for (int i = 0; i < m; i++) S->w[i] = 1.0;
        memset(S->wk, 1, (size_t)m); S->wlazy = 0;
        for (int j = 0; j < S->N; j++) S->dvx[j] = 1.0;
        rows_partition(S);
        refactor(S); compute_primal_dual(S);
    }
    return done;
}
/* (8 November 2026, a test: BRISK_SXEQCRASH=1) the dual method's start on equations: from the logicals' basis an equation's
 * logical is basic and fixed, so it must leave, one iteration per such row at least (heldout2_dc_power_dispatch_2160, LP: 2,156
 * equations, 3,051 iterations; Gurobi's dual simplex method 562). A nonbasic fixed logical is dual feasible whatever its reduced
 * cost, and so is a nonbasic boxed column at the bound its reduced cost's sign names: structural columns that are free or boxed
 * may take the place of equations' logicals without making the start dual infeasible beyond what the bounds repair. Bixby's
 * rules (primal_crash): free columns first, then boxed ones, the sparser first; each in the row of an equation holding its
 * largest entry (at least 0.99 of it) not yet taken, and only when its entries in the rows taken are at most 1 % of their
 * pivots (a nearly triangular, well-conditioned basis) */
static int eq_crash(Sx *S) {
    /* (12 November 2026) on by default in its form with columns of cost zero (2), with the settings it was judged with (every
     * equation, the threshold 0.5 of the column's largest entry, entries in the rows taken up to 1 of their pivots, only a column
     * that absorbs its row's violation), and not on row sifting's sets (sx_solve_core): the 76 LPs it acts on and the 618
     * development LPs by the default route 0.993 and 1.002 (0.989 on the 59 whose iterations changed, three rounds), one status
     * gained (rail03), the held-out set unchanged; heldout2_dc_power_dispatch_2160 (LP) 0.47 -> 0.11 s with the dual simplex
     * method, supportcase34 1.29 -> 1.06 s, neos-4531126-vouga 2.52 -> 1.79 s by default. Losses: piperout-03 +26 ms, roll3000
     * +16 ms, rd-rplusc-21 +57 ms by default; ns1644855 with the dual simplex method alone past 60 s (the default route solves it
     * by the interior-point method). BRISK_SXEQCRASH=0: off; 1: the form of 8 November */
    static int on = -1; if (on < 0) { const char *e = getenv("BRISK_SXEQCRASH"); on = e ? atoi(e) : 2; }
    if (!on) return 0;
    const int eqzero = on == 2;
    static int eqabs = -1; if (eqabs < 0) { const char *e = getenv("BRISK_SXEQCRASHABS"); eqabs = e ? atoi(e) != 0 : 1; }
    if (!eqzero) eqabs = 0;
    /* (11 November 2026, with BRISK_SXEQCRASH=2) only from a dual feasible start: the columns of cost zero keep it so, which is
     * the point of the rule; a start with dual infeasibilities (columns with one bound and a cost of the wrong sign) goes to a
     * dual phase 1 whose kind depends on whether the start is primal feasible, and a crash that makes it so changes that kind
     * (ns1690781, LP: one column brought in, 1,126 -> 3,888 iterations by the auxiliary problem's phase 1) */
    if (eqzero) {
        compute_dual(S);
        for (int j = 0; j < S->N; j++) {
            if (S->pos[j] >= 0 || is_boxed(S, j) || S->lo[j] == S->up[j]) continue;
            const double dj = S->d[j];
            if (S->lo[j] <= -SX_INF && S->up[j] >= SX_INF ? fabs(dj) > S->told : S->lo[j] > -SX_INF ? dj < -S->told : dj > S->told) return 0;
        }
    }
    const int n = S->n, m = S->m;
    /* (the rows of equations the start violates: an equation it meets keeps its logical basic at its value, no iteration
     * needed; BRISK_SXEQCRASHALL=1: every equation) */
    static int eqall = -1; if (eqall < 0) { const char *e = getenv("BRISK_SXEQCRASHALL"); eqall = e ? atoi(e) != 0 : eqzero; }
    compute_primal(S);
    unsigned char *viol = xcalloc((size_t)m + 1, 1);
    int neq = 0;
    for (int i = 0; i < m; i++) {
        if (S->basic[i] != n + i || S->lo[n + i] != S->up[n + i]) continue;
        const double v = S->x[n + i], b = S->lo[n + i];
        if (eqall || fabs(v - b) > S->tolp) { viol[i] = 1; neq++; }
    }
    if (neq == 0) { free(viol); return 0; }
    static double eqrel = -1; if (eqrel < 0) { const char *e = getenv("BRISK_SXEQCRASHREL"); eqrel = e ? atof(e) : eqzero ? 0.5 : 0.99; }
    static double eqtol = -1; if (eqtol < 0) { const char *e = getenv("BRISK_SXEQCRASHTOL"); eqtol = e ? atof(e) : eqzero ? 1.0 : 0.01; }
    static double eqlen = -1; if (eqlen < 0) { const char *e = getenv("BRISK_SXEQCRASHLEN"); eqlen = e ? atof(e) : 0; }
    const double avglen_ = n > 0 ? fmax(1.0, (double)S->Ap[n] / n) : 1.0;
    int *cand = xmalloc(sizeof(int) * ((size_t)n + 1)); double *key = xmalloc(sizeof(double) * ((size_t)n + 1)); int nc = 0;
    for (int j = 0; j < n; j++) {
        if (S->pos[j] >= 0 || S->Ap[j + 1] == S->Ap[j]) continue;
        const int lf = S->lo[j] > -SX_INF, uf = S->up[j] < SX_INF;
        if (lf != uf || (lf && S->lo[j] == S->up[j])) continue;       /* (free or boxed only) */
        /* (11 November 2026, a test: BRISK_SXEQCRASH=2) only columns of cost zero: with the logicals' costs zero the duals
         * stay zero and every reduced cost as it was, so the start stays dual feasible exactly and only its primal
         * infeasibility changes */
        if (eqzero && S->cost[j] != 0) continue;
        if (eqlen > 0 && S->Ap[j + 1] - S->Ap[j] > eqlen * avglen_) continue;   /* (12 November 2026, an experiment: BRISK_SXEQCRASHLEN=r) */
        key[j] = -((lf ? 1e7 : 0.0) + (double)(S->Ap[j + 1] - S->Ap[j]));
        cand[nc++] = j;
    }
    sx_sort_by_key_desc(cand, nc, key);
    if (getenv("BRISK_SXEQCRASHDBG")) {   /* (13 November 2026, a diagnostic) the start's candidates and equations */
        int nlo = 0, nbox = 0, nfree = 0, ncz = 0;
        for (int j = 0; j < n; j++) { if (S->pos[j] >= 0) continue; const int lf = S->lo[j] > -SX_INF, uf = S->up[j] < SX_INF;
            if (lf && uf) nbox++; else if (!lf && !uf) nfree++; else nlo++; if (S->cost[j] == 0) ncz++; }
        int nrng = 0; for (int i = 0; i < m; i++) if (S->lo[n + i] > -SX_INF && S->up[n + i] < SX_INF && S->lo[n + i] < S->up[n + i]) nrng++;
        int neqt = 0, nlogb = 0; for (int i = 0; i < m; i++) { if (S->lo[n + i] == S->up[n + i]) neqt++; if (S->basic[i] == n + i) nlogb++; }
        printf("  [eq_crash: %d rows, %d equations, %d logicals basic]\n", m, neqt, nlogb);
        printf("  [eq_crash: %d equations to take, %d candidates; nonbasic columns: %d boxed, %d free, %d with one bound, %d of cost zero; %d ranged rows]\n", neq, nc, nbox, nfree, nlo, ncz, nrng);
    }
    double *rpiv = xcalloc((size_t)m + 1, sizeof(double));
    int done = 0;
    for (int t = 0; t < nc && done < neq; t++) {
        const int j = cand[t];
        double cmax = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) cmax = fmax(cmax, fabs(S->Ax[p]));
        int bi = -1; double ba = 0, worst = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p]; const double a = fabs(S->Ax[p]);
            if (rpiv[i] > 0) { const double r = a / rpiv[i]; if (r > worst) worst = r; continue; }
            if (a < eqrel * cmax || S->basic[i] != n + i || !viol[i]) continue;
            /* (11 November 2026, with BRISK_SXEQCRASH=2) among entries within the threshold the row of fewest entries, as
             * Bixby's crash (less fill; a flow column of a network's equations then takes its own definition row, not a node's
             * balance row), then the larger entry */
            if (eqzero) { const int rl_ = S->Rp[i + 1] - S->Rp[i], rb_ = bi >= 0 ? S->Rp[bi + 1] - S->Rp[bi] : INT_MAX;
                if (bi < 0 || rl_ < rb_ || (rl_ == rb_ && a > ba)) { bi = i; ba = a; } continue; }
            if (a > ba) { bi = i; ba = a; }
        }
        if (bi < 0 || worst > eqtol) continue;
        /* (11 November 2026, with BRISK_SXEQCRASH=2, BRISK_SXEQCRASHABS=1) only a column that absorbs its row's violation
         * within its own bounds: the move t = (s_i - b_i) / a_ij that satisfies the equation (s = -Ax its logical) leaves
         * x_j + t inside [l_j, u_j]; the logicals of the column's other rows follow the move, so later choices see the rows as
         * they are then. A column that cannot moves the infeasibility to itself instead of removing it */
        if (eqabs) {
            double aij = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) if (S->Ai[p] == bi) { aij = S->Ax[p]; break; }
            const double t = (S->x[n + bi] - S->lo[n + bi]) / aij, xn = S->x[j] + t;
            if (xn < S->lo[j] - S->tolp * (1 + fabs(S->lo[j])) || xn > S->up[j] + S->tolp * (1 + fabs(S->up[j]))) continue;
            S->x[j] = xn;
            for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) S->x[n + S->Ai[p]] -= S->Ax[p] * t;
        }
        rpiv[bi] = ba;
        S->basic[bi] = j; S->pos[j] = bi; S->move[j] = 0;
        S->pos[n + bi] = -1; set_nonbasic(S, n + bi, S->cost[n + bi]);
        done++;
    }
    { static int cdb = -1; if (cdb < 0) cdb = getenv("BRISK_SXEQCRASHDBG") != NULL;   /* (a diagnostic: the start's coverage) */
      if (cdb) fprintf(stderr, "[eqcrash] %d of %d equations (m %d)\n", done, neq, m); }
    free(cand); free(key); free(rpiv); free(viol);
    if (done) {
        for (int i = 0; i < m; i++) S->w[i] = 1.0;
        memset(S->wk, 1, (size_t)m); S->wlazy = 0;
        rows_partition(S);
        refactor(S); compute_primal_dual(S);
        /* (11 November 2026, a test: BRISK_SXEQCRASHW=1) the dual steepest-edge weights of the crashed basis computed lazily, a
         * row's the first time it is a candidate to leave (as for a basis given to the re-solves, sxm_set_basis_w's mode 1),
         * instead of the logicals' basis's 1 for every row: with structural columns in the basis the rows of B^-1 are not unit
         * vectors. The stand-in for an unknown weight is the lower bound 1 / |b_i|^2 of the row's basic column */
        static int crw = -1; if (crw < 0) { const char *e = getenv("BRISK_SXEQCRASHW"); crw = e && atoi(e) != 0; }
        if (crw) {
            memset(S->wk, 0, (size_t)m); S->wlazy = 1;
            for (int i = 0; i < m; i++) {
                const int j = S->basic[i];
                if (j >= n) continue;
                double c2 = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) c2 += S->Ax[p] * S->Ax[p];
                S->w[i] = c2 > 1 ? fmax(1.0 / c2, 1e-4) : 1.0;
            }
        }
    }
    return done;
}
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
    memset(S->wk, 1, (size_t)m); S->wlazy = 0;
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
        static int ftd_ = -1; if (ftd_ < 0) ftd_ = getenv("BRISK_SXFTDBG") != NULL;
        const long wc0 = S->F.nflops + S->F.nedges;
        sxlu_ftran(&S->F, col, S->dens_col, &S->spike);
        if (ftd_) { S->dbg_wcol += (double)(S->F.nflops + S->F.nedges - wc0); S->dbg_ncol += col->cnt >= 0 ? col->cnt : S->m; }
        const long wt0 = S->F.nflops + S->F.nedges;
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
        if (ftd_) S->dbg_wtau += (double)(S->F.nflops + S->F.nedges - wt0);
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
    {   /* (3 November 2026) in one pass over the factor when neither solve is hypersparse: the same bits (sxlu_ftran_pair) */
        sxlu_ftran_pair(&S->F, col, S->dens_col, &S->spike, tau, S->dens_tau);
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
/* (4 November 2026) the primal steepest-edge weights' solve v = B^{-T} alpha_q: by the factorization's routine with the
 * second workspace (sxlu_btran_ws), so that the same routine runs alone or beside the row's solve (the same bits at
 * every thread count), and not counted in the refactorization rule's solve work nor in the statistics of the row's solves */
static void btran_pse_note(Sx *S, const SxVec *v, double t) {
    SxWs *W = &S->ws2;
    S->F.nflops += W->nflops; S->F.nedges += W->nedges;
    S->work += (double)W->nflops + (v->cnt >= 0 ? v->cnt : S->m);
    S->dens_pv = 0.95 * S->dens_pv + 0.05 * (v->cnt >= 0 ? (double)v->cnt / S->m : 1.0);
    S->t_pse += t; S->fl_pse += (double)W->nflops + W->nedges;
    /* (11 November 2026, a test: BRISK_SXREFACPSE=1) the weights' solve counted in the refactorization rule's measure of the
     * solves' work as well: it runs through the same updates, and when it is dense (Hdiet_5000, LP: 0.99 of the rows) its cost
     * grows with every update while the row's sparser solve hides that growth. Noted here, after the parallel section and
     * after the row's solve in either case, so the order of the notes is the same at every thread count */
    static int rpse = -1; if (rpse < 0) { const char *e_ = getenv("BRISK_SXREFACPSE"); rpse = e_ ? atoi(e_) : 0; }
    if (rpse) sw_note(S, (double)W->nflops + W->nedges, 1);
}
static void btran_pse(Sx *S, SxVec *v) {
    const double t = now();
    S->ws2.nflops = 0; S->ws2.nedges = 0;
    sxlu_btran_ws(&S->F, &S->ws2, v, S->dens_pv);
    btran_pse_note(S, v, now() - t);
}
/* x_B = -B^{-1} N x_N */
static void compute_primal(Sx *S) {
    SxVec *v = &S->col; sxv_clear(v);
    /* (11 November 2026) accumulated as a dense vector and listed once (the forward solve lists it: cnt -1), instead of a
     * check for a new nonzero at every entry (col_scatter): this sum was 10 % of the time of a node LP of the mixed-integer
     * solver (a multiple knapsack LP, 10 rows, 300 columns, half of them at their upper bound). The same sums in the same order */
    double *rv = v->val; const int n = S->n;
    for (int j = 0; j < n; j++) {
        const double xj = S->x[j];
        if (S->pos[j] >= 0 || xj == 0) continue;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) rv[S->Ai[p]] -= xj * S->Ax[p];
    }
    for (int i = 0; i < S->m; i++) { const int j = n + i; if (S->pos[j] < 0 && S->x[j] != 0) rv[i] -= S->x[j]; }
    v->cnt = -1;
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
/* (12 November 2026) compute_primal and compute_dual with one pass over the columns: the nonbasic values' sum and the
 * reduced costs read each nonbasic column once instead of twice. The two computations are independent (the forward solve
 * of the sum and the backward solve of the basic costs use the factorization only, on separate vectors), and each sum is
 * formed in the same order as in the separate calls, so the results have the same bits. The final check of every re-solve
 * computes both (the mixed-integer solver's node LPs). BRISK_SXPDFUSE=0: the two calls */
static void compute_primal_dual(Sx *S) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_SXPDFUSE"); on = !(e_ && atoi(e_) == 0); }
    if (!on) { compute_primal(S); compute_dual(S); return; }
    SxVec *v = &S->col; sxv_clear(v);
    SxVec *w = &S->rho; sxv_clear(w);
    for (int i = 0; i < S->m; i++) w->val[i] = S->cost[S->basic[i]];
    sxv_reindex(w);
    double dd = 1; btran(S, w, &dd, NULL);
    const double *y = w->val;
    double *rv = v->val; const int n = S->n;
    for (int j = 0; j < n; j++) {
        if (S->pos[j] >= 0) { S->d[j] = 0; continue; }
        const double xj = S->x[j];
        double s = S->cost[j];
        if (xj != 0) { for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; rv[i] -= xj * S->Ax[p]; s -= S->Ax[p] * y[i]; } }
        else for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) s -= S->Ax[p] * y[S->Ai[p]];
        S->d[j] = s;
    }
    for (int i = 0; i < S->m; i++) {
        const int j = n + i;
        if (S->pos[j] >= 0) { S->d[j] = 0; continue; }
        if (S->x[j] != 0) rv[i] -= S->x[j];
        S->d[j] = S->cost[j] - y[i];
    }
    sxv_clear(w);
    v->cnt = -1;
    dd = 1; ftran(S, v, &dd, NULL);
    for (int i = 0; i < S->m; i++) S->x[S->basic[i]] = v->val[i];
    sxv_clear(v);
}
/* (1 November 2026) one step of iterative refinement of the basic values and of the duals with the current factorization
 * and its updates: x_B += B^-1 r, r the residual of [A I] x = 0; y += B^-T (c_B - B'y), then the reduced costs. The
 * re-solves' clean-up tries it before a new factorization (the MIP developer's lotsizing_15: most of the node LPs'
 * factorizations came from that clean-up after a few updates) */
static void refine_primal(Sx *S) {
    SxVec *v = &S->col; sxv_clear(v);
    for (int j = 0; j < S->N; j++) if (S->x[j] != 0) col_scatter(S, j, v, -S->x[j]);
    double dd = 1; ftran(S, v, &dd, NULL);
    for (int i = 0; i < S->m; i++) S->x[S->basic[i]] += v->val[i];
    sxv_clear(v);
}
static void refine_dual(Sx *S) {
    SxVec *v = &S->rho; sxv_clear(v);
    for (int i = 0; i < S->m; i++) v->val[i] = S->cost[S->basic[i]];
    sxv_reindex(v);
    double dd = 1; btran(S, v, &dd, NULL);
    const int m = S->m; double *y = malloc(sizeof(double) * ((size_t)m + 1));
    memcpy(y, v->val, sizeof(double) * (size_t)m);
    sxv_clear(v);
    /* the residual on the basic columns: c_j - a_j'y */
    for (int i = 0; i < m; i++) {
        const int j = S->basic[i]; double r = S->cost[j];
        if (j < S->n) { for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) r -= S->Ax[p] * y[S->Ai[p]]; }
        else r -= y[j - S->n];
        v->val[i] = r;
    }
    sxv_reindex(v);
    dd = 1; btran(S, v, &dd, NULL);
    for (int i = 0; i < m; i++) y[i] += v->val[i];
    sxv_clear(v);
    for (int j = 0; j < S->n; j++) {
        if (S->pos[j] >= 0) { S->d[j] = 0; continue; }
        double t = S->cost[j];
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) t -= S->Ax[p] * y[S->Ai[p]];
        S->d[j] = t;
    }
    for (int i = 0; i < m; i++) { const int j = S->n + i; S->d[j] = S->pos[j] >= 0 ? 0 : S->cost[j] - y[i]; }
    free(y);
}
/* factorize the current basis; replaced columns become nonbasic at a bound */
static int refactor(Sx *S) {
    const double t = now();
    /* the steepest-edge weights are kept by position: carried by variable through the reordering of the factorization;
     * a logical that comes in for a column the factorization could not pivot starts with weight 1 */
    /* (28 October 2026) one array kept by the method, zero outside this function: the entries of the old basis are set and,
     * after the reordering, those of the new basis and of the columns replaced are cleared again (the old basis lies in
     * these). It was allocated and zeroed for all variables at every factorization: 15.6 MB on the tester's largest coupled
     * network (LP, 575,040 rows and 1,376,640 columns), a pass over all variables where the basis has a third of them */
    double *wvar = S->wvar;
    for (int i = 0; i < S->m; i++) wvar[S->basic[i]] = S->wk[i] ? S->w[i] : -S->w[i];   /* (an unknown weight carried negative) */
    {   static int ftd = -1; if (ftd < 0) ftd = getenv("BRISK_SXFTDBG") != NULL;     /* (8 November 2026, a diagnostic) */
        if (ftd) fprintf(stderr, "[ftdbg] it %ld: before the factorization L %ld U %ld (at the factorization %ld) R %ld, %d updates; solves so far: forward %.3f s, entries %ld, edges %ld; restricted path: columns %.3g (their entries %.3g), weights %.3g; weights restricted %ld, full %ld\n",
                         (long)S->iter, S->F.nnz_l, S->F.nnz_u, S->F.nnz_u0, S->F.nnz_r, S->F.nupd, S->t_ftran, S->F.nflops, S->F.nedges, S->dbg_wcol, S->dbg_ncol, S->dbg_wtau, S->n_taupart, S->n_taufull);
        if (ftd) fprintf(stderr, "[ftdbg] the rule: factor_work %.3g, solve work at the start %.3g now %.3g (%d noted), growth an update %.3g; times: forward %.3f backward %.3f weights %.3f factorizations %.3f s\n",
                         S->F.factor_work, S->sw_base, S->sw_ew, S->sw_n, S->F.nupd > 0 ? (S->sw_ew - S->sw_base) / S->F.nupd : 0.0, S->t_ftran, S->t_btran, S->t_pse, S->t_factor);
        if (ftd) fprintf(stderr, "[ftdbg] the dual's parts so far: leaving row %.3f pivot row %.3f ratio test %.3f dual update %.3f primal step %.3f weights %.3f candidates %.3f rebuilds %.3f s\n",
                         S->t_chuzr, S->t_row, S->t_chuzc, S->t_dupd, S->t_pupd, S->t_dse, S->t_heap, S->t_rebuild); }
    const double tf_dbg = now();
    if (getenv("BRISK_SXLUSTAT") && S->F.m == S->m) fprintf(stderr, "[lu] before: %d updates, U %ld (from %ld), R %ld, L %ld\n", S->F.nupd, S->F.nnz_u, S->F.nnz_u0, S->F.nnz_r, S->F.nnz_l);
    const int nout = sxlu_factor(&S->F, S->m, S->n, S->Ap, S->Ai, S->Ax, S->basic, S->out);
    {   /* (12 November 2026, a diagnostic: BRISK_SXLUSTAT=1 prints each factorization's size; BRISK_SXLUDUMP=<file> with
         * BRISK_SXLUDUMPK=<k> writes the k-th factorization's basis as triplets (row, position, value; logicals as unit columns)) */
        static int lst = -1, nfac_ = 0; if (lst < 0) lst = getenv("BRISK_SXLUSTAT") != NULL;
        const char *df_ = getenv("BRISK_SXLUDUMP"); nfac_++;
        if (lst || df_) {
            long bnz = 0; int nst = 0;
            for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; if (j < S->n) { nst++; bnz += S->Ap[j + 1] - S->Ap[j]; } else bnz++; }
            if (lst) fprintf(stderr, "[lu] %d: m %d, structurals %d, basis %ld entries, L %ld, U %ld (fill %.2f)\n", nfac_, S->m, nst, bnz, S->F.nnz_l, S->F.nnz_u0, (double)(S->F.nnz_l + S->F.nnz_u0) / (double)bnz);
            if (df_ && getenv("BRISK_SXLUDUMPK") && atoi(getenv("BRISK_SXLUDUMPK")) == nfac_) {
                FILE *fd = fopen(df_, "w");
                if (fd) { fprintf(fd, "%d %ld\n", S->m, bnz);
                    for (int i = 0; i < S->m; i++) { const int j = S->basic[i];
                        if (j < S->n) { for (int k = S->Ap[j]; k < S->Ap[j + 1]; k++) fprintf(fd, "%d %d %.17g\n", S->Ai[k], i, S->Ax[k]); }
                        else fprintf(fd, "%d %d 1\n", j - S->n, i); }
                    fclose(fd); }
            }
        }
    }
    {   static int ftd2 = -1; if (ftd2 < 0) ftd2 = getenv("BRISK_SXFTDBG") != NULL;
        if (ftd2) fprintf(stderr, "[ftdbg] the factorization: %.3f s, L %ld U %ld\n", now() - tf_dbg, S->F.nnz_l, S->F.nnz_u); }
    for (int k = 0; k < nout; k++) {
        const int j = S->out[k]; S->pos[j] = -1;
        if (!S->keepx) { set_nonbasic(S, j, S->d[j]); continue; }
        /* (the crossover: the point stays as it is; a replaced variable not at a bound is pushed again) */
        S->move[j] = S->lo[j] == S->up[j] ? 0 : S->x[j] == S->lo[j] ? 1 : S->x[j] == S->up[j] ? -1 : 0;
    }
    for (int i = 0; i < S->m; i++) {
        const int j = S->basic[i]; S->pos[j] = i; S->move[j] = 0;
        S->w[i] = wvar[j] > 0 ? wvar[j] : wvar[j] < 0 ? -wvar[j] : 1.0;
        S->wk[i] = wvar[j] > 0 || (wvar[j] == 0 && !S->wlazy);
    }
    for (int i = 0; i < S->m; i++) wvar[S->basic[i]] = 0;
    for (int k = 0; k < nout; k++) wvar[S->out[k]] = 0;
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
/* (12 November 2026, an experiment: BRISK_SXPRICEUNSC=1) the leaving row's merit by the infeasibility in the problem's own units
 * (the scaled infeasibility divided by the row's scale for a logical, times the column's scale for a structural): Gurobi's choices on
 * uccase7 (LP) fit that measure, rows 105.8 y - x >= 0 violated by 105.8 chosen before columns 78 above their bound, which our scaled
 * measure ranks first (REPORT_lp_r16) */
static double col_rand(int j);
static inline double infeas_merit(const Sx *S, int i) {
    double f = infeas(S, i);
    if (g_priceunsc > 0 && f > 0 && S->cscale && S->rscale) { const int j = S->basic[i]; f = j < S->n ? f * S->cscale[j] : f / S->rscale[j - S->n]; }
    /* (12 November 2026, an experiment: BRISK_SXPRICERANGE=1) a basic variable with a finite range: its merit divided by
     * (1 + the range), squared into the score (f^2 / (w (1 + range))) */
    { static int pr_ = -1; if (pr_ < 0) { const char *e_ = getenv("BRISK_SXPRICERANGE"); pr_ = e_ && atoi(e_) != 0; }
      if (pr_ && f > 0) { const int j = S->basic[i]; const double rg = S->up[j] - S->lo[j]; if (rg < SX_INF) f /= sqrt(1 + rg); } }
    /* (13 November 2026, an experiment: BRISK_SXSTRUCTW=k) a row whose basic variable is a structural column: its score divided
     * by k (Gurobi's dual simplex method on satellites4-25's copy takes a row's logical as the leaving variable in 750 of 750
     * iterations, where 505 basic structural columns are infeasible and our scores rank them first) */
    { static double sw_ = -1; if (sw_ < 0) { const char *e_ = getenv("BRISK_SXSTRUCTW"); sw_ = e_ ? atof(e_) : 0; }
      if (sw_ > 0 && f > 0 && S->basic[i] < S->n) f /= sqrt(sw_); }
    /* (13 November 2026, an experiment: BRISK_SXINFCAP=c, BRISK_SXINFPOW=p) the merit capped, min(f, c), or raised to the power p
     * (0 < p < 1): the score f^2 / w then leans on the weight, so among rows violated by c or more the one with the sparsest row of
     * B^-1 leaves. Gurobi's leaving rows on satellites4-25's copy (LP) are violated by 1 where hundreds of rows are violated more,
     * with rows of B^-1 of 8-14 entries, ranked 150-237 by our exact score (bench/lp_r16/mystery/B) */
    { static double cap_ = -1, pw_ = -1; if (cap_ < 0) { const char *e_ = getenv("BRISK_SXINFCAP"); cap_ = e_ ? atof(e_) : 0; e_ = getenv("BRISK_SXINFPOW"); pw_ = e_ ? atof(e_) : 0; }
      if (f > 0) { if (pw_ > 0) f = pow(f, pw_); if (cap_ > 0 && f > cap_) f = cap_; } }
    return f;
}
/* the heap of the leaving-row candidates */
/* candidates of equal merit by the lowest index (a scrambled order was tried and dropped: ISSUES 835) */
/* (13 November 2026, an experiment: BRISK_SXTIEVAR=1) equal merits by the lowest index of the basic variable, not of the row's
 * position: Gurobi's first pivot from its basis on satellites4-25's copy with perturbed costs (LP) takes variable 17154 of two rows
 * tied at the infeasibility 480.023 (ours the first position, variable 71612) */
static int g_tievar = -1;
static long g_forcedit = -1;   /* (BRISK_SXFORCEPATH: the iteration forced last) */
static int hbetter(const Sx *S, int a, int b) {
    if (g_tievar < 0) { const char *e_ = getenv("BRISK_SXTIEVAR"); g_tievar = e_ && atoi(e_) != 0; }
    /* (the merits equal within 1e-12 relative: the two rows' infeasibilities differ in the last bit by rounding) */
    if (g_tievar) { const double ka = S->hkey[a], kb = S->hkey[b];
        if (fabs(ka - kb) > 1e-12 * fmax(ka, kb)) return ka > kb;
        return S->basic[a] < S->basic[b]; }
    if (S->hkey[a] != S->hkey[b]) return S->hkey[a] > S->hkey[b];
    if (S->rrank) return S->rrank[a] < S->rrank[b];
    return a < b;
}
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
#define VFAC(S_, i_) ((S_)->vfac ? (S_)->vfac[(S_)->basic[i_]] : 1.0)   /* (BRISK_SXVFAC: the merit's factor of the row's basic variable) */
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
        const double f = infeas_merit(S, i);
        if (f == 0) { S->inI[i] = 0; continue; }
        S->Ilist[c++] = i;
        S->hkey[i] = f * f / S->w[i] * VFAC(S, i);
        S->hpos[i] = S->nheap; S->heap[S->nheap++] = i;
    }
    S->nI = c;
    for (int p = S->nheap / 2 - 1; p >= 0; p--) hdown(S, p);
    S->heap_ok = 1;
}
static void heap_touch(Sx *S, int i) {
    const double f = infeas_merit(S, i);
    if (f > 0 && !S->inI[i]) { S->inI[i] = 1; S->Ilist[S->nI++] = i; }
    const int p = S->hpos[i];
    if (f == 0) {
        if (p < 0) return;
        const int last = S->nheap - 1;
        if (p != last) { hswap(S, p, last); S->nheap--; S->hpos[i] = -1; hdown(S, p); hup(S, p); }
        else { S->nheap--; S->hpos[i] = -1; }
        return;
    }
    const double k = f * f / S->w[i] * VFAC(S, i);
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
    /* (the option BRISK_SXMLAZYW=1) a candidate whose weight is not known gets its exact weight, |row of B^{-1}|^2, when
     * it comes to the top, and the heap is sifted again: a few backward solves instead of one a row */
    while (S->wlazy && S->nheap && !S->wk[S->heap[0]]) {
        /* (a cap on this work, the rest left at 1, was worse than none: csched008, LP, 240 -> 907 iterations, the
         * exact weights beside the ones left at 1 favouring the latter's rows) */
        const int i = S->heap[0];
        SxVec *v = &S->wv; sxv_clear(v); v->val[i] = 1; v->idx[0] = i; v->cnt = 1;
        double dd = S->dens_rho; btran(S, v, &dd, NULL);
        double w = 0;
        if (v->cnt >= 0) for (int t = 0; t < v->cnt; t++) { const double a = v->val[v->idx[t]]; w += a * a; }
        else for (int k = 0; k < S->m; k++) w += v->val[k] * v->val[k];
        sxv_clear(v);
        S->w[i] = w > 1e-4 ? w : 1e-4; S->wk[i] = 1;
        const double f = infeas_merit(S, i);
        S->hkey[i] = f * f / S->w[i] * VFAC(S, i);
        hdown(S, 0);
    }
    /* (13 November 2026, an experiment: BRISK_SXNEARTOL=e) among the rows whose score is within the factor 1 - e of the best (their
     * weight known), the one of the smallest steepest-edge weight, |row of B^-1|^2, leaves: a sparser row of B^-1 at almost the same
     * merit (a pass over the heap's candidates; satellites4-25's copy, LP, bench/lp_r16/mystery/B) */
    { static double ntol_ = -1; if (ntol_ < 0) { const char *e_ = getenv("BRISK_SXNEARTOL"); ntol_ = e_ ? atof(e_) : 0; }
      if (ntol_ > 0 && S->nheap > 1) { const int b0 = S->heap[0]; const double lim = (1 - ntol_) * S->hkey[b0]; int b = b0;
          for (int p = 1; p < S->nheap; p++) { const int i = S->heap[p]; if (S->hkey[i] < lim || (S->wlazy && !S->wk[i])) continue; if (S->w[i] < S->w[b]) b = i; }
          return b; } }
    {   /* (13 November 2026, a diagnostic: BRISK_SXTRACEHEAP=k with BRISK_SXTRACE) at iteration k, the leaving-row candidates
         * whose score is within 1e-6 of the best, and any row listed in BRISK_SXTRACEROWS (comma-separated positions), on an "H" line:
         * position basic-variable score infeasibility weight weight-known */
        static long th_ = -2; if (th_ == -2) { const char *e_ = getenv("BRISK_SXTRACEHEAP"); th_ = e_ ? atol(e_) : -1; }
        if (th_ >= 0 && S->iter == th_ && S->trace && S->nheap) {
            const double top_ = S->hkey[S->heap[0]]; fprintf(S->trace, "H %ld top %d :", S->iter, S->heap[0]);
            for (int p_ = 0; p_ < S->nheap; p_++) { const int i_ = S->heap[p_]; if (S->hkey[i_] >= top_ * (1 - 1e-6)) fprintf(S->trace, " %d %d %.17g %.17g %.10g %d", i_, S->basic[i_], S->hkey[i_], infeas(S, i_), S->w[i_], S->wk ? S->wk[i_] : 1); }
            const char *rl_ = getenv("BRISK_SXTRACEROWS");
            if (rl_) { fprintf(S->trace, " | listed:"); char buf_[512]; snprintf(buf_, sizeof buf_, "%s", rl_);
                for (char *t_ = strtok(buf_, ","); t_; t_ = strtok(NULL, ",")) { const int i_ = atoi(t_); if (i_ >= 0 && i_ < S->m)
                    fprintf(S->trace, " %d %d inheap %d key %.10g infeas %.10g w %.10g x %.10g lo %.10g up %.10g", i_, S->basic[i_], S->hpos ? S->hpos[i_] : -9, S->hkey[i_], infeas(S, i_), S->w[i_], S->x[S->basic[i_]], S->lo[S->basic[i_]], S->up[S->basic[i_]]); } }
            fputc('\n', S->trace);
            /* (13 November 2026, a diagnostic: BRISK_SXTRACEHEAPN=n) the n best leaving candidates by score on a "HN" line: position
             * basic-variable score infeasibility weight weight-known */
            static int thn_ = -2; if (thn_ == -2) { const char *e_ = getenv("BRISK_SXTRACEHEAPN"); thn_ = e_ ? atoi(e_) : 0; }
            if (thn_ > 0) { int *ix_ = xmalloc(sizeof(int) * (size_t)S->nheap); for (int p_ = 0; p_ < S->nheap; p_++) ix_[p_] = S->heap[p_];
                fprintf(S->trace, "HN %ld :", S->iter);
                for (int t_ = 0; t_ < thn_ && t_ < S->nheap; t_++) { int b_ = t_; for (int p_ = t_ + 1; p_ < S->nheap; p_++) if (S->hkey[ix_[p_]] > S->hkey[ix_[b_]]) b_ = p_;
                    const int tmp_ = ix_[t_]; ix_[t_] = ix_[b_]; ix_[b_] = tmp_; const int i_ = ix_[t_];
                    fprintf(S->trace, " %d %d %.6g %.6g %.6g %d", i_, S->basic[i_], S->hkey[i_], infeas(S, i_), S->w[i_], S->wk ? S->wk[i_] : 1); }
                fputc('\n', S->trace); free(ix_); }
            /* (BRISK_SXTRACEVARS=v1,v2,...: those basic variables' rows on a "HV" line: variable position key infeasibility weight known,
             * and the rank of the key among the heap) */
            const char *vl_ = getenv("BRISK_SXTRACEVARS");
            if (vl_) { fprintf(S->trace, "HV %ld :", S->iter); char buf_[2048]; snprintf(buf_, sizeof buf_, "%s", vl_);
                for (char *t_ = strtok(buf_, ","); t_; t_ = strtok(NULL, ",")) { const int v_ = atoi(t_); const int i_ = v_ >= 0 && v_ < S->N ? S->pos[v_] : -1;
                    if (i_ < 0) { fprintf(S->trace, " %d nonbasic", v_); continue; }
                    int rk_ = 0; for (int p_ = 0; p_ < S->nheap; p_++) if (S->hkey[S->heap[p_]] > S->hkey[i_]) rk_++;
                    fprintf(S->trace, " %d pos %d inheap %d key %.6g infeas %.6g w %.6g wk %d rank %d", v_, i_, S->hpos[i_], S->hkey[i_], infeas(S, i_), S->w[i_], S->wk ? S->wk[i_] : 1, rk_); }
                fputc('\n', S->trace); } } }
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
/* the row copy's partition (each row's nonbasic entries first) brought to the basis in pos. (11 November 2026) Only the
 * columns whose side changed are moved (rows_enter, rows_leave): every column's entries are always all on one side (the
 * row copy starts all nonbasic, and only these two routines move entries), so the side its first entry is on is the
 * column's. Before, every entry of every row was passed and swapped: on the mixed-integer solver's node LPs, where a new
 * basis differs from the last in a few columns, that pass was 12 % of the time of a node (a multiple knapsack LP of 10 rows
 * and 300 columns, about three iterations a node) */
static void rows_partition(Sx *S) {
    const int n = S->n;
    for (int j = 0; j < n; j++) {
        const int p0 = S->Ap[j];
        if (p0 == S->Ap[j + 1]) continue;
        const int i = S->Ai[p0];
        const int nb = S->Cr[p0] < S->Rp[i] + S->Rnb[i];
        if (S->pos[j] >= 0) { if (nb) rows_enter(S, j); }
        else if (!nb) rows_leave(S, j);
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
            /* (26 October 2026) compressed in a register and stored whole (the list has room for 8 beyond N): a compressed
             * store to memory costs several times as much on this host's processors */
            _mm256_storeu_si256((__m256i *)(ix + c), _mm256_maskz_compress_epi32(big, idx));
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
        S->row_ord = 1;
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
        S->row_ord = 0;
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
        S->row_ord = 1;
    }
    S->dens_row = 0.95 * S->dens_row + 0.05 * (double)row->cnt / S->N;
    S->work += rowwork >= 0 ? (double)rowwork + row->cnt : (double)S->Ap[n] + S->m;
    S->t_price += now() - t;
}

/* (7 November 2026, BRISK_SXINTPREF=gamma; only with integer flags given, sxm_set_integer) the entering variable of a final
 * group cand[k0 .. k1) of the dual ratio test: q, the largest |alpha| (best), when it is a row logical or a continuous
 * column; when it is an integer column, the member of largest |alpha| among the logicals and continuous columns whose
 * |alpha| is at least gamma times best (the first such in the group's order among equal ones), if there is one. Every
 * member of the group lies within the Harris bound, so the step of any of them keeps every reduced cost within the dual
 * tolerance: the choice between them is free, and on a dual-degenerate LP (the relaxation of a mixed-integer program with
 * a zero objective, where every ratio is zero) it decides which variable becomes basic. An integer column that enters
 * becomes basic, usually at a fraction; a logical or a continuous column that enters leaves the integer columns at their
 * bounds. A group of one member, or one with no integer column at its head, is not changed */
/* (experiment, SXSLACKPREF=gamma; 13 November 2026: in both branches of the ratio test, before only in the unsorted one) among
 * the final group's members with |alpha| >= gamma times the largest, a logical (length 1) or the shortest column enters, ties
 * by the larger |alpha|: the choice within the group is free (every member keeps the reduced costs within the tolerance), and
 * a short entering column keeps the basis sparse */
static int short_pref_pick(Sx *S, int k0, int k1, int q, double best) {
    if (!(S->env_slackpref > 0) || q < 0 || q >= S->n || k1 - k0 < 2) return q;
    int q2 = -1; long l2 = 1L << 40; double b2 = 0;
    for (int k = k0; k < k1; k++) {
        if (S->ca[k] < S->env_slackpref * best) continue;
        const int j = S->cand[k]; const long len = j >= S->n ? 1 : S->Ap[j + 1] - S->Ap[j];
        if (len < l2 || (len == l2 && S->ca[k] > b2)) { l2 = len; b2 = S->ca[k]; q2 = j; }
    }
    return q2 >= 0 ? q2 : q;
}
static int int_pref_pick(Sx *S, int k0, int k1, int q, double best) {
    if (!S->isint || !(S->env_intpref > 0) || q < 0 || q >= S->n || !S->isint[q] || k1 - k0 < 2) return q;
    const double lim = S->env_intpref * best;
    int q2 = -1, nalt = 0; double b2 = 0;
    for (int k = k0; k < k1; k++) {
        const int j = S->cand[k];
        if (j < S->n && S->isint[j]) continue;
        nalt++;
        if (S->ca[k] >= lim && S->ca[k] > b2) { b2 = S->ca[k]; q2 = j; }
    }
    if (q2 < 0) { if (nalt) S->n_ip_small++; else S->n_ip_none++; return q; }
    S->n_intpref++;
    return q2;
}

/* the bound-flipping ratio test with Harris' tolerances. delta: the primal infeasibility of the leaving row (< 0 below its
 * lower bound). Returns the entering variable (-1: none: the leaving row cannot be made feasible: primal infeasible);
 * the variables to flip are returned in S->cand[0..*nflip-1] */
/* (8 November 2026, a test: BRISK_SXNOBF=1) the ratio test without its long steps over bound flips: the first group only */
static int sx_nobf(void) { static int c = -1; if (c < 0) { const char *e = getenv("BRISK_SXNOBF"); c = e && atoi(e) != 0; } return c; }

/* (14 November 2026, an experiment in a scratch copy, bench/lp_r16/uccase7: BRISK_SXBFEARLY=gamma) the bound-flipping ratio test may
 * stop at any breakpoint it passes: the dual objective rises up to the textbook stop. Among the candidates of all groups up to the
 * final one whose |alpha| is at least gamma times the largest of them, the one with the fewest column entries enters (ties to the
 * last), if it is strictly shorter than the final group's choice; the candidates of the groups before its group are flipped, the
 * rest stay at their bounds. BRISK_SXBFEARLYM=1: the shortest only among the candidates of |alpha| at least gamma times the
 * final group's choice's |alpha| */
static _Thread_local int bfe_ng = 0, bfe_cap = 0; static _Thread_local int *bfe_gs = NULL;   /* (per thread: the ratio test runs on two threads in the race; the array is kept for the thread's life) */
static int bfe_on_ = -1;   /* (the switch read once; with it off nothing is recorded and nothing allocated: the default path neither shares nor leaks this state) */
static void bfe_push(int st) { if (bfe_on_ < 0) bfe_on_ = getenv("BRISK_SXBFEARLY") != NULL; if (!bfe_on_) return; if (bfe_ng >= bfe_cap) { bfe_cap = bfe_cap ? 2 * bfe_cap : 256; bfe_gs = realloc(bfe_gs, sizeof(int) * bfe_cap); } bfe_gs[bfe_ng++] = st; }
static _Thread_local long bfe_n = 0, bfe_chg = 0;
static int bfe_pick(Sx *S, int g, int q, int *nfl) {
    static double gm = -2; static int md = 0;
    if (gm == -2) { const char *e = getenv("BRISK_SXBFEARLY"); gm = e ? atof(e) : -1; const char *e2 = getenv("BRISK_SXBFEARLYM"); md = e2 ? atoi(e2) : 0; }
    if (gm < 0 || q < 0 || bfe_ng < 2) return q;
    bfe_n++;
    double amx = 0, aq = 0; for (int k = 0; k < g; k++) { if (S->ca[k] > amx) amx = S->ca[k]; if (S->cand[k] == q) aq = S->ca[k]; }
    const double thr = md == 1 ? gm * aq : gm * amx;
    #define BFE_LEN(j) ((j) < S->n ? S->Ap[(j) + 1] - S->Ap[(j)] : 1)
    const int lq = BFE_LEN(q); int bk = -1, bl = lq;
    const int g0 = bfe_gs[bfe_ng - 1];
    for (int k = 0; k < g0; k++) { const int j = S->cand[k]; if (S->move[j] == 0 || S->ca[k] < thr) continue; const int l = BFE_LEN(j); if (l < bl || (l == bl && bk >= 0)) { bl = l; bk = k; } }
    #undef BFE_LEN
    if (bk < 0 || bl >= lq) return q;
    int gi = 0; while (gi + 1 < bfe_ng && bfe_gs[gi + 1] <= bk) gi++;
    const int gs = bfe_gs[gi];
    { int t = S->cand[bk]; S->cand[bk] = S->cand[gs]; S->cand[gs] = t; double u = S->ca[bk]; S->ca[bk] = S->ca[gs]; S->ca[gs] = u; u = S->cr[bk]; S->cr[bk] = S->cr[gs]; S->cr[gs] = u; }
    *nfl = gs; S->fg_n = 1; bfe_chg++;
    return S->cand[gs];
}
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
    /* (26 October 2026) 2, the default: the blocks compressed in a register and stored whole (a compressed store to memory
     * costs several times as much on the processors of this host), and the two passes over the candidates below eight at
     * a time; 1: the code of 24 October */
    static int vecc = -1; if (vecc < 0) { const char *e = getenv("BRISK_SXVECCHUZC"); vecc = e ? atoi(e) : 2; }
    /* (26 October 2026) a row listed in increasing order and longer than a sixth of the columns: one pass over the dense
     * arrays in memory order (contiguous loads; the entries off the row are zero and are skipped), the same candidates in
     * the same order. The gathers of the list's entries cost about 14 cycles an entry on this host (L2 misses: the row,
     * the reduced costs and the moves of 62,000 columns), the dense pass about 2 a column: fast0507 (LP) without
     * sifting, rows of 38,700 of 62,600 entries, 590,000 -> about 140,000 cycles a call (a microbenchmark: 545,000 ->
     * 133,000). BRISK_SXVECCHUZC=1: the list's gathers always */
    if (vecc >= 2 && S->row_ord && 6L * row->cnt > S->N) {
        const int N = S->N;
        const __m512d vmo = _mm512_set1_pd(mo), vpt = _mm512_set1_pd(pivtol), zero = _mm512_setzero_pd();
        const __m256i z256 = _mm256_setzero_si256(), e8 = _mm256_set1_epi32(8);
        __m256i jj = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        int j0 = 0;
        for (; j0 + 8 <= N; j0 += 8, jj = _mm256_add_epi32(jj, e8)) {
            const __m512d al = _mm512_loadu_pd(row->val + j0);
            const __mmask8 nz = _mm512_cmp_pd_mask(al, zero, _CMP_NEQ_UQ);
            if (!nz) continue;
            const __m256i mw = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(S->move + j0)));
            if (_mm256_mask_cmpeq_epi32_mask(nz, mw, z256)) {      /* (a fixed or free variable on the row: the scalar code) */
                for (int j = j0; j < j0 + 8; j++) {
                    const double alj = row->val[j]; if (alj == 0) continue;
                    int mv = S->move[j];
                    if (mv == 0) {
                        if (S->lo[j] == S->up[j]) continue;
                        if (fabs(alj) < pivtol) continue;
                        S->cand[nc] = j; S->ca[nc] = fabs(alj); S->cr[nc] = 0; nc++;
                        continue;
                    }
                    const double a = alj * mo * mv;
                    if (a <= pivtol) continue;
                    S->cand[nc] = j; S->ca[nc] = a; S->cr[nc] = mv * S->d[j]; nc++;
                }
                continue;
            }
            const __m512d mvd = _mm512_cvtepi32_pd(mw);
            const __m512d a = _mm512_mul_pd(_mm512_mul_pd(al, vmo), mvd);
            const __mmask8 ok = _mm512_mask_cmp_pd_mask(nz, a, vpt, _CMP_GT_OQ);
            if (!ok) continue;
            const __m512d td = _mm512_mul_pd(mvd, _mm512_maskz_loadu_pd(ok, S->d + j0));
            _mm256_storeu_si256((__m256i *)(S->cand + nc), _mm256_maskz_compress_epi32(ok, jj));
            _mm512_storeu_pd(S->ca + nc, _mm512_maskz_compress_pd(ok, a));
            _mm512_storeu_pd(S->cr + nc, _mm512_maskz_compress_pd(ok, td));
            nc += __builtin_popcount((unsigned)ok);
        }
        for (int j = j0; j < N; j++) {
            const double alj = row->val[j]; if (alj == 0) continue;
            int mv = S->move[j];
            if (mv == 0) {
                if (S->lo[j] == S->up[j]) continue;
                if (fabs(alj) < pivtol) continue;
                S->cand[nc] = j; S->ca[nc] = fabs(alj); S->cr[nc] = 0; nc++;
                continue;
            }
            const double a = alj * mo * mv;
            if (a <= pivtol) continue;
            S->cand[nc] = j; S->ca[nc] = a; S->cr[nc] = mv * S->d[j]; nc++;
        }
        k0 = row->cnt;           /* (the list's loops below: nothing left) */
    } else if (vecc) {
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
            if (vecc >= 2) {
                _mm256_storeu_si256((__m256i *)(S->cand + nc), _mm256_maskz_compress_epi32(ok, jj));
                _mm512_storeu_pd(S->ca + nc, _mm512_maskz_compress_pd(ok, a));
                _mm512_storeu_pd(S->cr + nc, _mm512_maskz_compress_pd(ok, td));
            } else {
                _mm256_mask_compressstoreu_epi32(S->cand + nc, ok, jj);
                _mm512_mask_compressstoreu_pd(S->ca + nc, ok, a);
                _mm512_mask_compressstoreu_pd(S->cr + nc, ok, td);
            }
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
    *nflip = 0; bfe_ng = 0;
    if (nc == 0) return -1;
    double slope = fabs(delta);
    /* preselection (HiGHS): the candidates up to a step where the flips' total change exceeds the infeasibility, the
     * step growing tenfold from ten times the first Harris bound; the groups are then formed within them */
    {
        double hn = SX_INF, hd = 1;      /* the least (td + told) / a, as hn / hd */
        /* (24 October 2026) in four independent lanes (each comparison waited for the last: fast0507, LP, 94 microseconds
         * an iteration), combined by the same comparison, a tie to the earlier candidate: the first least as before */
        {   double ln[8] = { SX_INF, SX_INF, SX_INF, SX_INF, SX_INF, SX_INF, SX_INF, SX_INF }, ld[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
            int lk[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
            int k = 0, nl = 4;
#if defined(__AVX512F__) && defined(__AVX512VL__)
            /* (26 October 2026) eight lanes in one register: the same comparison in each, the same combination */
            if (vecc >= 2) {
                nl = 8;
                __m512d vln = _mm512_set1_pd(SX_INF), vld = _mm512_set1_pd(1.0), vlk = _mm512_set1_pd(-1.0);
                const __m512d vtd = _mm512_set1_pd(told), v8 = _mm512_set1_pd(8.0);
                __m512d vk = _mm512_setr_pd(0, 1, 2, 3, 4, 5, 6, 7);
                for (; k + 8 <= nc; k += 8) {
                    const __m512d ca = _mm512_loadu_pd(S->ca + k), nk = _mm512_add_pd(_mm512_loadu_pd(S->cr + k), vtd);
                    const __mmask8 lt = _mm512_cmp_pd_mask(_mm512_mul_pd(nk, vld), _mm512_mul_pd(vln, ca), _CMP_LT_OQ);
                    vln = _mm512_mask_blend_pd(lt, vln, nk); vld = _mm512_mask_blend_pd(lt, vld, ca); vlk = _mm512_mask_blend_pd(lt, vlk, vk);
                    vk = _mm512_add_pd(vk, v8);
                }
                double tk[8]; _mm512_storeu_pd(ln, vln); _mm512_storeu_pd(ld, vld); _mm512_storeu_pd(tk, vlk);
                for (int l = 0; l < 8; l++) lk[l] = (int)tk[l];
            }
#endif
            if (nl == 4)
                for (; k + 4 <= nc; k += 4)
                    for (int l = 0; l < 4; l++) { const double nk = S->cr[k + l] + told; if (nk * ld[l] < ln[l] * S->ca[k + l]) { ln[l] = nk; ld[l] = S->ca[k + l]; lk[l] = k + l; } }
            for (; k < nc; k++) { const int l = k & (nl - 1); const double nk = S->cr[k] + told; if (nk * ld[l] < ln[l] * S->ca[k]) { ln[l] = nk; ld[l] = S->ca[k]; lk[l] = k; } }
            int bk = -1;
            for (int l = 0; l < nl; l++) {
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
            #define SX_PRESEL_TAKE(kk) do { const int k_ = (kk); const int j = S->cand[k_]; \
                    const double t1 = S->ca[k_], t2 = S->cr[k_]; \
                    S->ca[k_] = S->ca[w]; S->cr[k_] = S->cr[w]; S->cand[k_] = S->cand[w]; \
                    S->ca[w] = t1; S->cr[w] = t2; S->cand[w] = j; w++; \
                    const double rg = S->up[j] - S->lo[j]; \
                    tot += (rg < SX_INF && S->move[j] != 0) ? t1 * rg : SX_INF; } while (0)
            int k = w;
#if defined(__AVX512F__) && defined(__AVX512VL__)
            /* (26 October 2026) the test eight at a time, the few taken one by one in order (a swap moves an entry
             * already passed, below the block's later ones: the block's mask stays valid) */
            if (vecc >= 2) {
                const __m512d vs = _mm512_set1_pd(sel);
                for (; k + 8 <= nc; k += 8) {
                    __mmask8 mk = _mm512_cmp_pd_mask(_mm512_loadu_pd(S->cr + k), _mm512_mul_pd(vs, _mm512_loadu_pd(S->ca + k)), _CMP_LE_OQ);
                    while (mk) { const int b = __builtin_ctz((unsigned)mk); mk &= (__mmask8)(mk - 1); SX_PRESEL_TAKE(k + b); }
                }
            }
#endif
            for (; k < nc; k++) if (S->cr[k] <= sel * S->ca[k]) SX_PRESEL_TAKE(k);
            #undef SX_PRESEL_TAKE
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
    /* (1 November 2026) and the two by their cost: the loop below recomputes the Harris bound over the remaining candidates
     * for each group (about nc a group), the sort costs about nc log2 nc once; after log2 of the remaining candidates'
     * groups the rest are sorted and their groups formed in one sweep. The LP relaxation of nw04 after the mixed-integer
     * presolve (36 rows, 86,652 binary columns, a set partitioning): thousands of groups an iteration, 35 ms in the
     * ratio test; 7.48 -> 0.49 s for the same 206 iterations. BRISK_SXCHUZCSORT=1: the sort always, 0: never */
    static int csort = -1; if (csort < 0) { const char *e = getenv("BRISK_SXCHUZCSORT"); csort = e ? atoi(e) : -1; }
    int s0 = 0;
    if (csort == 1 && nc > 1) {
      sorted: ;
        double *key = S->ck, *hm = S->chm; int *ord = S->cp;
        for (int k = s0; k < nc; k++) { key[k] = S->cr[k] / S->ca[k]; ord[k] = k; }
        /* sort ord by (key, cand): an insertion sort for few, else a heap sort (no qsort: its comparator would need the
         * arrays through a global) */
        {   int *od = ord + s0; const int nn = nc - s0;      /* (the candidates from s0 on: those before are flipped) */
        if (nn <= 32) {
            for (int a = 1; a < nn; a++) { const int t = od[a]; int b = a - 1;
                while (b >= 0 && (key[od[b]] > key[t] || (key[od[b]] == key[t] && S->cand[od[b]] > S->cand[t]))) { od[b + 1] = od[b]; b--; }
                od[b + 1] = t; }
        } else {
            #define SXK_LT(x, y) (key[x] < key[y] || (key[x] == key[y] && S->cand[x] < S->cand[y]))
            for (int r = nn / 2 - 1; r >= 0; r--) { int i = r; for (;;) { int c = 2 * i + 1; if (c >= nn) break; if (c + 1 < nn && SXK_LT(od[c], od[c + 1])) c++; if (!SXK_LT(od[i], od[c])) break; const int t = od[i]; od[i] = od[c]; od[c] = t; i = c; } }
            for (int e = nn - 1; e > 0; e--) { int t = od[0]; od[0] = od[e]; od[e] = t;
                int i = 0; for (;;) { int c = 2 * i + 1; if (c >= e) break; if (c + 1 < e && SXK_LT(od[c], od[c + 1])) c++; if (!SXK_LT(od[i], od[c])) break; t = od[i]; od[i] = od[c]; od[c] = t; i = c; } }
            #undef SXK_LT
        }
        }
        /* the candidates in that order */
        { int *ti = S->cti; double *td = S->ctd; const size_t nn = (size_t)(nc - s0);
          for (int k = s0; k < nc; k++) ti[k] = S->cand[ord[k]];
          memcpy(S->cand + s0, ti + s0, sizeof(int) * nn);
          for (int k = s0; k < nc; k++) td[k] = S->ca[ord[k]];
          memcpy(S->ca + s0, td + s0, sizeof(double) * nn);
          for (int k = s0; k < nc; k++) td[k] = S->cr[ord[k]];
          memcpy(S->cr + s0, td + s0, sizeof(double) * nn); }
        hm[nc] = SX_INF;
        for (int k = nc - 1; k >= s0; k--) { const double h = (S->cr[k] + told) / S->ca[k]; hm[k] = h < hm[k + 1] ? h : hm[k + 1]; }
        int st = s0;
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
            bfe_push(st);
            if (sx_nobf()) inf = 1;
            if (inf || slope - dsl <= 0 || (g == nc && slope - dsl <= S->tolp)) {
                int q = -1; double best = 0;
                for (int k = st; k < g; k++) if (S->ca[k] > best) { best = S->ca[k]; q = S->cand[k]; }
                q = int_pref_pick(S, st, g, q, best);
                q = short_pref_pick(S, st, g, q, best);
                S->fg_n = g - st;
                *nflip = st;
                return bfe_pick(S, g, q, nflip);
            }
            slope -= dsl;
            st = g;
            if (st == nc) { *nflip = 0; return -1; }
        }
    }
    int start = 0, ngr = 0;
    for (;;) {
        /* (the sort when the groups so far have cost what it costs: log2 of the remaining candidates) */
        if (csort != 0 && nc - start > 32 && ngr >= 2) {
            int lg = 0; for (int t = nc - start; t > 1; t >>= 1) lg++;
            if (ngr >= lg) { s0 = start; goto sorted; }
        }
        ngr++;
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
        bfe_push(start);
        /* the final group: the slope turns negative, an unbounded candidate, or the last candidates (when their flips
         * leave the row feasible within the tolerance; otherwise the row cannot be made feasible) */
        if (sx_nobf()) inf = 1;
        if (inf || slope - dsl <= 0 || (g == nc && slope - dsl <= S->tolp)) {
            /* the final group: the largest |alpha| enters */
            int q = -1; double best = 0;
            for (int k = start; k < g; k++) if (S->ca[k] > best) { best = S->ca[k]; q = S->cand[k]; }
            {   /* (13 November 2026, an experiment: BRISK_SXTIEPICK=1 the highest index among the members of the largest |alpha|,
                 * =2 one of them at random (a hash of the iteration and the index); by default the first met, the lowest index. Gurobi
                 * on satellites4-25's copy (LP) breaks such exact ties otherwise: 63325, the 29th of 30, where ours takes 157) */
                static int tp_ = -1; if (tp_ < 0) { const char *e_ = getenv("BRISK_SXTIEPICK"); tp_ = e_ ? atoi(e_) : 0; }
                if (tp_ && q >= 0) { double bh = -1;
                    for (int k = start; k < g; k++) if (S->ca[k] >= best * (1 - 1e-12)) {
                        const double h = tp_ == 1 ? (double)S->cand[k] : col_rand(S->cand[k] * 7919 + (int)(S->iter % 100003));
                        if (h > bh) { bh = h; q = S->cand[k]; } } } }
            {   /* (13 November 2026, a diagnostic: BRISK_SXFORCEQ=<file> of lines "iteration variable") the entering variable forced to
                 * the given one when it is in the final group (Gurobi's choices among exact ties on satellites4-25's copy, LP, so that
                 * the next difference that is not a tie shows) */
                static int nfq_ = -1; static long *fqi_ = NULL; static int *fqv_ = NULL;
                if (nfq_ < 0) { nfq_ = 0; const char *e_ = getenv("BRISK_SXFORCEQ"); FILE *ff_ = e_ ? fopen(e_, "r") : NULL;
                    if (ff_) { long a_; int b_; int cap_ = 0; while (fscanf(ff_, "%ld %d", &a_, &b_) == 2) { if (nfq_ >= cap_) { cap_ = cap_ ? 2 * cap_ : 1024; fqi_ = realloc(fqi_, sizeof(long) * cap_); fqv_ = realloc(fqv_, sizeof(int) * cap_); } fqi_[nfq_] = a_; fqv_[nfq_++] = b_; } fclose(ff_); } }
                if (nfq_ > 0) { int lo_ = 0, hi_ = nfq_ - 1, f_ = -1; while (lo_ <= hi_) { const int md_ = (lo_ + hi_) / 2; if (fqi_[md_] == S->iter) { f_ = md_; break; } if (fqi_[md_] < S->iter) lo_ = md_ + 1; else hi_ = md_ - 1; }
                    if (f_ >= 0) for (int k = start; k < g; k++) if (S->cand[k] == fqv_[f_]) { q = fqv_[f_]; best = S->ca[k]; break; } } }
            q = int_pref_pick(S, start, g, q, best);
            S->fg_n = g - start;
            q = short_pref_pick(S, start, g, q, best);
            {   /* (13 November 2026, a diagnostic: BRISK_SXTRACEGRP=1 with BRISK_SXTRACE) the final group's members on a line of their
                 * own before the iteration's line: "G iteration chosen : variable |alpha| ratio reduced-cost ..." (the tie-breaks
                 * against Gurobi's on satellites4-25's copy, LP) */
                static int tg_ = -1; if (tg_ < 0) { const char *e_ = getenv("BRISK_SXTRACEGRP"); tg_ = e_ && atoi(e_) == 1; }
                if (tg_ && S->trace) { fprintf(S->trace, "G %ld %d slope %.6g dsl %.6g inf %d nc %d start %d g %d :", S->iter, q, slope, dsl, inf, nc, start, g);
                    for (int k = start; k < g && k < start + 200; k++) fprintf(S->trace, " %d %.6g %.6g %.6g", S->cand[k], S->ca[k], S->cr[k], S->d[S->cand[k]]);
                    fputc('\n', S->trace);
                    /* (13 November 2026, a diagnostic: BRISK_SXTRACEALL=1) every candidate of the ratio test (after the preselection) on an
                     * "A" line: variable |alpha| reduced-cost-in-the-move's-sign range */
                    static int ta_ = -1; if (ta_ < 0) { const char *e2_ = getenv("BRISK_SXTRACEALL"); ta_ = e2_ && atoi(e2_) == 1; }
                    if (ta_) { fprintf(S->trace, "A %ld delta %.10g told %.3g :", S->iter, delta, told);
                        for (int k = 0; k < nc; k++) fprintf(S->trace, " %d %.10g %.10g %.6g", S->cand[k], S->ca[k], S->cr[k], S->up[S->cand[k]] - S->lo[S->cand[k]]);
                        fputc('\n', S->trace); } } }
            *nflip = start;
            return bfe_pick(S, g, q, nflip);
        }
        slope -= dsl;
        start = g;
        if (start == nc) { *nflip = 0; return -1; }   /* all flipped and still infeasible */
    }
}

/* (7 November 2026, ISSUES 831; BRISK_SXGRPBF=gamma, 0.1 by default, 0: off) the final group of the bound-
 * flipping ratio test passed in the order of its ratios, as the test passes the groups before it. The Harris pass lumps
 * the candidates whose ratios lie within the dual tolerance into one group, and the test lets the largest |alpha| of it
 * enter while the others stay at their bounds. When the infeasibility left exceeds what that member's box can take
 * (|alpha_q| (u_q - l_q)), the member becomes basic outside its box, a new primal infeasibility about as large as the one
 * removed; on LPs whose costs are mostly zero (brazil3, mzzv11, ns1644855) the perturbed reduced costs lie within a few
 * tolerances of each other, the groups hold tens of members, and this happened in up to two thirds of the iterations,
 * half of them steps of length zero. Here the members are taken in increasing ratio td / |alpha| (ties to the larger
 * |alpha|, then the earlier member): a boxed member is flipped while the infeasibility left exceeds its share
 * |alpha_j| (u_j - l_j); the first member that can take the rest (or has no finite box) enters if its reduced cost has
 * the right sign (a step of positive length) and its |alpha| is at least gamma times the group's largest (the pivot's
 * size; HiGHS accepts a final group whose largest |alpha| reaches 0.1 of the row's largest), else it is passed over and
 * stays at its bound. Every flipped member has a ratio at most the entering one's, so after the step its reduced cost
 * has the right sign at its new bound; the members passed over and those after the entering one keep reduced costs
 * within the tolerance (the Harris bound covers the whole group). A flip changes the objective by td (u_j - l_j): a
 * member of the wrong sign within the tolerance (td < 0) lowers it. Such flips are made only while they take at most
 * half of the iteration's rise (the step's td_q / |alpha_q| times the infeasibility left, and the other flips), so that
 * the dual objective rises at every step that shifts no cost and the method cannot cycle; without that bound they
 * cycled while this was built (grow22 and pilot87 of Netlib, LPs: 2.1 million and 126,000 iterations at 60 s), and
 * without any of them ns1644855 (LP) took 24,000-40,000 iterations instead of 17,000-22,000. A member of reduced cost
 * zero is not flipped (no change of the objective). When no member qualifies, the largest |alpha| enters as before. */
static int g_dsws = 1;   /* (13 November 2026, BRISK_SXDEGWS=0: the sparsest-member rule's solves through the factor's own workspace, as first built) */
/* (13 November 2026) the sparsest-member rule's count: the nonzeros of B^-1 a_j, solved in the second solve's workspace (wv) */
static long ds_count(Sx *S, int j) {
    SxVec *v = &S->wv; long nz = 0;
    sxv_clear(v); col_scatter(S, j, v, 1.0);
    /* (the factor's running statistics of the forward solve through U, which choose how the iteration's own solves run, saved and
     * restored: without that the rule changed the path where it changed no choice, qap12, LP: 11,939 -> 6,940 iterations) */
    const double hc = S->F.hy_canc; const int hp = S->F.hy_cp;
    if (g_dsws) { SxWs *W = &S->ws2; W->nflops = W->nedges = 0; sxlu_ftran_ws(&S->F, W, v, S->dens_col, NULL); } else sxlu_ftran(&S->F, v, S->dens_col, NULL);
    if (g_dsws) { S->F.hy_canc = hc; S->F.hy_cp = hp; }
    if (v->cnt >= 0) nz = v->cnt; else for (int i = 0; i < S->m; i++) nz += v->val[i] != 0;
    sxv_clear(v);
    S->ds_solve++;
    return nz;
}
/* (14 November 2026, BRISK_SXDEGBOUND=1; ISSUES 881) the count on the factor's patterns, stopped once past bound (sxlu_count_bounded):
 * a member denser than the best so far is known as such for the work of reaching bound + 1 rows */
static long ds_count_sym(Sx *S, int j, long bound) {
    SxVec *v = &S->wv;
    sxv_clear(v); col_scatter(S, j, v, 1.0); if (v->cnt < 0) sxv_reindex(v);
    SxWs *W = &S->ws2; W->nflops = W->nedges = 0;
    const long n = sxlu_count_bounded(&S->F, W, v->idx, v->cnt, bound);
    sxv_clear(v);
    S->ds_solve++;
    return n;
}
static _Thread_local const Sx *bf_S; static _Thread_local int bf_pass;   /* (final_group_bf's sort) */
static int bf_cmp(const void *pa, const void *pb) {
    const Sx *S = bf_S; const int x = *(const int *)pa, y = *(const int *)pb, pass = bf_pass;
    #define BF_LT2(x, y) (S->cr[x] * S->ca[y] < S->cr[y] * S->ca[x] || (S->cr[x] * S->ca[y] == S->cr[y] * S->ca[x] && (pass == 2 ? S->cand[x] < S->cand[y] : pass == 0 ? (S->ca[x] > S->ca[y] || (S->ca[x] == S->ca[y] && x < y)) : (S->ca[x] < S->ca[y] || (S->ca[x] == S->ca[y] && x < y)))))
    if (BF_LT2(x, y)) return -1; if (BF_LT2(y, x)) return 1; return 0;
    #undef BF_LT2
}
static int final_group_bf(Sx *S, int q, double delta, int *nflip) {
    const int g0 = *nflip, g1 = g0 + S->fg_n, ng = g1 - g0;
    if (ng < 2) return q;
    double sl = fabs(delta);   /* the infeasibility left after the earlier groups' flips */
    for (int k = 0; k < g0; k++) { const int j = S->cand[k]; sl -= S->ca[k] * (S->up[j] - S->lo[j]); }
    int *od = S->cp; double amax = 0;
    for (int k = 0; k < ng; k++) { od[k] = g0 + k; if (S->ca[g0 + k] > amax) amax = S->ca[g0 + k]; }
    /* (the members by increasing ratio cr / ca, compared by cross products; an insertion sort: groups are small) */
    /* (13 November 2026, an experiment: BRISK_SXGRPBFRETRY=1) when no member can enter in this order (the members left once the
     * flips have taken the infeasibility are all below gamma times the largest |alpha|), the group passed again with the ties of
     * ratio ordered by increasing |alpha|: the small members flipped first, a large one left to enter. Satellites4-25's copy (LP):
     * a group of 42 at one ratio, 30 of |alpha| 1 and 12 of 1/32 (each box taking 1 of the infeasibility 41): the 30 large were
     * flipped first, then 10 small, the two small left could not enter, and none was flipped (the largest |alpha| entered with a
     * step of 41 across its box of 1); Gurobi flips 40 and enters a member of |alpha| 1 (bench/lp_r16/slack) */
    static int bfretry = -1; if (bfretry < 0) { const char *e_ = getenv("BRISK_SXGRPBFRETRY"); bfretry = e_ ? atoi(e_) : 0; }
    const double lim = S->env_grpbf * amax;
    const double sl0 = sl;
    int ke = -1, nf = 0;
    /* (BRISK_SXGRPBFRETRY=2: one pass, the ties of ratio in the order of the variables' indices, as Gurobi's on that LP) */
    for (int pass = bfretry == 2 ? 2 : 0; pass < (bfretry == 1 ? 2 : bfretry == 2 ? 3 : 1) && ke < 0; pass++) {
    if (pass == 1) { sl = sl0; nf = 0; for (int k = 0; k < ng; k++) od[k] = g0 + k; }
    #define BF_LT(x, y) (S->cr[x] * S->ca[y] < S->cr[y] * S->ca[x] || (S->cr[x] * S->ca[y] == S->cr[y] * S->ca[x] && (pass == 2 ? S->cand[x] < S->cand[y] : pass == 0 ? (S->ca[x] > S->ca[y] || (S->ca[x] == S->ca[y] && x < y)) : (S->ca[x] < S->ca[y] || (S->ca[x] == S->ca[y] && x < y)))))
    /* (13 November 2026) groups of more than 32 members sorted by qsort with the same comparison (a strict total order: the same
     * result as the insertion sort, which costs the square of the group's size: with the costs unperturbed a group can hold
     * thousands of members; neos-578379, LP, all costs zero, 21 s of its 24 s here) */
    if (ng > 32) { bf_S = S; bf_pass = pass; qsort(od, (size_t)ng, sizeof(int), bf_cmp); }
    else
    for (int a = 1; a < ng; a++) { const int t = od[a]; int b = a - 1; while (b >= 0 && BF_LT(t, od[b])) { od[b + 1] = od[b]; b--; } od[b + 1] = t; }
    #undef BF_LT
    for (int a = 0; a < ng; a++) {
        const int k = od[a], j = S->cand[k];
        const int boxed = S->move[j] != 0 && S->up[j] - S->lo[j] < SX_INF;
        const double cap = boxed ? S->ca[k] * (S->up[j] - S->lo[j]) : SX_INF;
        if (sl - cap > 0) { if (S->cr[k] != 0) { od[nf++] = k; sl -= cap; } continue; }   /* flipped (od[0 .. nf)) */
        if (S->cr[k] > 0 && S->ca[k] >= lim) {
            ke = k;
            /* (BRISK_SXINTPREF, with integer flags) an integer column found to enter: the next member in the order of the
             * ratios that is a logical or a continuous column and could enter as well (its box takes the rest, a step of
             * positive length, |alpha| at least gamma times the group's largest and at least this test's own bound)
             * enters instead; the members between stay at their bounds (within the Harris bound, as those passed over) */
            if (S->isint && S->env_intpref > 0 && j < S->n && S->isint[j]) {
                const double lim2 = fmax(lim, S->env_intpref * amax);
                for (int a2 = a + 1; a2 < ng; a2++) {
                    const int k2 = od[a2], j2 = S->cand[k2];
                    if (j2 < S->n && S->isint[j2]) continue;
                    const int bx2 = S->move[j2] != 0 && S->up[j2] - S->lo[j2] < SX_INF;
                    const double cap2 = bx2 ? S->ca[k2] * (S->up[j2] - S->lo[j2]) : SX_INF;
                    if (sl - cap2 > 0 || !(S->cr[k2] > 0) || S->ca[k2] < lim2) continue;
                    ke = k2; S->n_intpref++; break;
                }
            }
            break;
        }
    }
    }
    if (ke < 0) return q;
    {   /* the objective's change: the flips at the reduced costs before the step, sum td_j (u_j - l_j), and the step on
         * the infeasibility left, (td_q / |alpha_q|) sl; the flips of members of the wrong sign (td_j < 0) lower it */
        double gain = S->cr[ke] / S->ca[ke] * sl, loss = 0;
        for (int a = 0; a < nf; a++) { const int k = od[a], j = S->cand[k]; const double v = S->cr[k] * (S->up[j] - S->lo[j]); if (v > 0) gain += v; else loss -= v; }
        /* (BRISK_SXGRPBFLOSS: the share of the rise they may take, 0.5 by default; above 1 the objective can fall and
         * the method cycle: grow22, pilot87 with no bound) */
        static double lfr = -1; if (lfr < 0) { const char *e_ = getenv("BRISK_SXGRPBFLOSS"); lfr = e_ ? atof(e_) : 0.5; }
        if (loss > lfr * gain) {   /* (they would take more than that share of the rise: left at their bounds) */
            int w2 = 0;
            for (int a = 0; a < nf; a++) if (S->cr[od[a]] > 0) od[w2++] = od[a];
            nf = w2;
        }
    }
    /* the flipped members to cand[g0 .. g0 + nf), the others after them in their order */
    const int qe = S->cand[ke];
    int *tc = S->cti; double *ta = S->ctd, *tr = S->ck, *isf = S->chm;
    for (int k = g0; k < g1; k++) isf[k] = 0;
    int w = 0;
    for (int a = 0; a < nf; a++) { const int k = od[a]; isf[k] = 1; tc[w] = S->cand[k]; ta[w] = S->ca[k]; tr[w] = S->cr[k]; w++; }
    for (int k = g0; k < g1; k++) if (!isf[k]) { tc[w] = S->cand[k]; ta[w] = S->ca[k]; tr[w] = S->cr[k]; w++; }
    for (int k = 0; k < ng; k++) { S->cand[g0 + k] = tc[k]; S->ca[g0 + k] = ta[k]; S->cr[g0 + k] = tr[k]; }
    *nflip = g0 + nf;
    return qe;
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
/* (13 November 2026, an experiment: BRISK_SXWMONO=1) a row's steepest-edge weight never falls in the update: Gurobi passes over
 * rows whose exact weight is small for tens of iterations (satellites4-25's copy, LP: a column infeasible by 1 with weight 1 left
 * for 55 iterations), as if its weights only grew */
static int g_wmono = -1;
static void update_dse(Sx *S, int r, const SxVec *alpha, const SxVec *tau, double wr) {
    if (g_wmono < 0) { const char *e_ = getenv("BRISK_SXWMONO"); g_wmono = e_ && atoi(e_) != 0; }
    const double ar = alpha->val[r];
    const double nw = wr / (ar * ar);
    if (alpha->cnt >= 0) {
        for (int k = 0; k < alpha->cnt; k++) {
            const int i = alpha->idx[k];
            if (i == r) continue;
            const double ai = alpha->val[i];
            const double ratio = ai / ar;
            double wi = S->w[i] + ratio * (ratio * wr - 2 * tau->val[i]);
            if (g_wlb > 0 && wi < ratio * ratio * wr) wi = ratio * ratio * wr;   /* (BRISK_SXWLB: Forrest and Goldfarb's lower bound) */
            if (g_wmono > 0 && wi < S->w[i]) wi = S->w[i];   /* (BRISK_SXWMONO: a weight never falls) */
            S->w[i] = wi > 1e-4 ? wi : 1e-4;
        }
    } else {
        for (int i = 0; i < S->m; i++) {
            if (i == r || alpha->val[i] == 0) continue;
            const double ratio = alpha->val[i] / ar;
            double wi = S->w[i] + ratio * (ratio * wr - 2 * tau->val[i]);
            if (g_wlb > 0 && wi < ratio * ratio * wr) wi = ratio * ratio * wr;
            if (g_wmono > 0 && wi < S->w[i]) wi = S->w[i];
            S->w[i] = wi > 1e-4 ? wi : 1e-4;
        }
    }
    S->w[r] = nw > 1e-4 ? nw : 1e-4; S->wk[r] = 1;     /* (from |rho_r|^2, computed in the iteration: known) */
    {   /* (12 November 2026, an experiment: BRISK_SXWLOG1=1) the rows whose basic variable is a logical keep the weight 1 (the
         * entering variable q at row r after the basis change: a logical there too) */
        static int wl1_ = -1; if (wl1_ < 0) { const char *e_ = getenv("BRISK_SXWLOG1"); wl1_ = e_ && atoi(e_) != 0; }
        if (wl1_) { if (alpha->cnt >= 0) { for (int k = 0; k < alpha->cnt; k++) { const int i = alpha->idx[k]; if (i != r && S->basic[i] >= S->n) S->w[i] = 1.0; } }
                    else for (int i = 0; i < S->m; i++) if (i != r && S->basic[i] >= S->n) S->w[i] = 1.0; }
    }
}

/* the dual method's starting bounds: with the basis of the logicals, a boxed column of cost zero is dual feasible at
 * either bound, so it is put at the one that leaves the smaller primal infeasibility (the sum over its rows of the
 * logicals' bound violations), one pass over the columns; the perturbation then keeps that bound. Returns the flips. */
static long dual_start_bounds_(Sx *S, int anycost) {
    const int n = S->n, m = S->m;
    double *v = xcalloc((size_t)m, sizeof(double));      /* the logicals' values s = -A x */
    for (int j = 0; j < n; j++) {
        const double xj = S->x[j];
        if (xj != 0) for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v[S->Ai[p]] -= S->Ax[p] * xj;
    }
#define SX_VIOL(i, s) ((anycost >= 4 && S->lo[n + (i)] == S->up[n + (i)]) ? 0.0 : (s) < S->lo[n + (i)] ? S->lo[n + (i)] - (s) : (s) > S->up[n + (i)] ? (s) - S->up[n + (i)] : 0.0)   /* (4: inequality rows only, a test) */
    long nfl = 0;
    /* (11 November 2026, a test: anycost 3) the columns in the order of the objective a flip loses per unit of infeasibility it
     * removes at the start, cheapest first (a greedy covering) */
    int *ord = NULL; int nord = n;
    if (anycost >= 3) {
        ord = xmalloc(sizeof(int) * ((size_t)n + 1)); double *key = xmalloc(sizeof(double) * ((size_t)n + 1)); nord = 0;
        for (int j = 0; j < n; j++) {
            if (S->pos[j] >= 0 || !is_boxed(S, j) || S->lo[j] == S->up[j]) continue;
            const double to = S->move[j] > 0 ? S->up[j] : S->lo[j], dx = to - S->x[j];
            double g = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const int i = S->Ai[p]; g += SX_VIOL(i, v[i] - S->Ax[p] * dx) - SX_VIOL(i, v[i]); }
            if (!(g < 0)) continue;
            key[j] = S->cost[j] * dx / -g; ord[nord++] = j;
        }
        {   /* (by the key, ties by the column's index: the same order as the insertion sort it replaces, in n log n) */
            struct SxBp *kb = xmalloc(sizeof(struct SxBp) * ((size_t)nord + 1));
            for (int k = 0; k < nord; k++) { kb[k].t = key[ord[k]]; kb[k].e = ord[k]; }
            qsort(kb, nord, sizeof *kb, cmp_bp);
            for (int k = 0; k < nord; k++) ord[k] = kb[k].e;
            free(kb);
        }
        free(key);
    }
    for (int jj = 0; jj < nord; jj++) {
        const int j = ord ? ord[jj] : jj;
        if (S->pos[j] >= 0 || (S->cost[j] != 0 && !anycost) || !is_boxed(S, j) || S->lo[j] == S->up[j]) continue;
        const double to = S->move[j] > 0 ? S->up[j] : S->lo[j], dx = to - S->x[j];
        double g = 0, base = 0; int worse = 0;
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) {
            const int i = S->Ai[p];
            const double e0 = SX_VIOL(i, v[i]), e1 = SX_VIOL(i, v[i] - S->Ax[p] * dx);
            g += e1 - e0; base += e0; if (e1 > e0) worse = 1;
        }
        if (!(g < -1e-9 * (1 + base))) continue;
        if (anycost >= 5 && worse) continue;   /* (11 November 2026, a test: 5, a flip that makes no row worse) */
        for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) v[S->Ai[p]] -= S->Ax[p] * dx;
        S->x[j] = to; S->move[j] = (signed char)-S->move[j]; nfl++;
        if (anycost >= 2 && nfl <= 12 && getenv("BRISK_SXPSTARTDIAG")) printf("  flip %d: column %d, [%g, %g] to %g, cost %.3g, %d entries\n", (int)nfl, j, S->lo[j], S->up[j], to, S->cost[j], S->Ap[j + 1] - S->Ap[j]);
    }
#undef SX_VIOL
    free(v); free(ord);
    return nfl;
}
static long dual_start_bounds(Sx *S) { return dual_start_bounds_(S, 0); }
/* cost perturbation (against dual degeneracy): structurals by a relative random amount in the direction that keeps the
 * starting basis' reduced costs feasible */
/* (the stall rule on trial: BRISK_SXSTALL=1; off until measured on the sets; read by sx_setup into env_nostall) */
/* (13 November 2026, the LP developer) BRISK_SXLAZYSTALL: the stall rule serves the perturbation at the first stall (dual_loop, dual_simplex) */
static int lazystall(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_SXLAZYSTALL"); v = e_ ? atoi(e_) : 0; if (v < 0) v = 0; } return v; }
static int cmp_dbl_asc(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static void perturb_costs(Sx *S) {
    const int n = S->n;
    double bigc = 0;
    for (int j = 0; j < n; j++) if (fabs(S->cost0[j]) > bigc) bigc = fabs(S->cost0[j]);
    if (bigc > 100) bigc = sqrt(sqrt(bigc));
    if (bigc == 0) bigc = 1;      /* (28 October 2026: no cost at all; see mode 1 below) */
    int nbox = 0; for (int j = 0; j < S->N; j++) nbox += is_boxed(S, j);
    if ((double)nbox / S->N < 0.01 && bigc > 1) bigc = 1;
    double base = 5e-7 * bigc, one = 1;
    if (getenv("SXPERT_BASE")) base *= atof(getenv("SXPERT_BASE"));
    int mode = getenv("SXPERT_MODE") ? atoi(getenv("SXPERT_MODE")) : 1;
    if (mode == 1) {   /* the rule in the units of the costs as given (HiGHS): bigc and the floor 1 unscaled */
        double bc = bigc * S->costscale; if (bc > 100) bc = sqrt(sqrt(bc));
        /* (28 October 2026) no cost at all (a feasibility problem): as if the largest cost were 1, as HiGHS does ("perturb
         * as if max cost was 1"); the base was 0, no perturbation at all, and every reduced cost tied: neos-578379 (LP,
         * held-out set, all costs zero) 22,492 iterations against HiGHS's 4,505 on a presolved problem of the same size */
        if (bc == 0) bc = 1;
        if ((double)nbox / S->N < 0.01 && bc > 1) bc = 1;
        base = 5e-7 * bc / S->costscale; one = 1.0;
        if (getenv("SXPERT_BASE")) base *= atof(getenv("SXPERT_BASE"));
        one = 1.0;   /* xp = (|c_j| S + 1) base_u / S in scaled units = (|c_j| + 1/S) base_u */
    }
    /* (26 October 2026, experiment SXPERT_MODE=2) Koberstein's rule (2005, 6.3.1; MOPS, and close to COIN's CLP): in the
     * units of the reduced costs and their tolerance eps_D, xi_j = 100 eps_D + 1e-5 |c_j|, weighted by the column's
     * count of entries (1e-2, 1e-1, 1, 2, 5, 10, 20, 30, 40 for 1 to 9, 100 from 10: a sparse column is preferred to
     * enter, which keeps the factor sparse), kept within [min(1e-2 eps_D, 1e-5), max(1e3 eps_D, 1e-4 mean |c|)] by
     * tenfold steps, then randomized to (0.5 .. 1) xi_j. The constant part is a hundred times the dual tolerance whatever
     * the costs: an LP whose costs are mostly zero (mzzv11, 98 % of its columns) gets reduced costs spread well above the
     * tolerance; the rule above (HiGHS) gave those columns about 5e-7 times the fourth root of the largest cost */
    double kxmin = 0, kxmax = 0; const double kpsi = 1e-5;
    /* (mode 3) Koberstein's condition as well: his rule when the structural costs take fewer distinct values than a
     * quarter of the columns (the problem is markedly dual degenerate by its costs), the rule above otherwise (pilot87,
     * LP, many distinct costs: 12.3 s with the rule above, 84 s with his on it) */
    if (mode == 3) {
        double *cv = malloc(sizeof(double) * (n + 1)); int nc = 0;
        for (int j = 0; j < n; j++) if (S->lo0[j] != S->up0[j]) cv[nc++] = S->cost0[j];
        qsort(cv, nc, sizeof(double), cmp_dbl_asc);
        int nd = nc > 0; for (int k = 1; k < nc; k++) nd += cv[k] != cv[k - 1];
        free(cv);
        mode = 4 * nd < nc ? 2 : 1;
        if (S->verbose >= 2) printf("  cost perturbation: %d distinct costs of %d columns: %s\n", nd, nc, mode == 2 ? "Koberstein's rule" : "the rule by the largest cost");
    }
    if (mode == 2) {
        double sc = 0; for (int j = 0; j < n; j++) sc += fabs(S->cost0[j]);
        kxmin = fmin(1e-2 * S->told, kpsi); kxmax = fmax(1e3 * S->told, kpsi * 10 * sc / (n > 0 ? n : 1));
    }
    static const double kw[10] = { 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 100.0 };
    /* (4 November 2026) a boxed nonbasic column is perturbed towards the bound it is at, whatever its cost's sign: from the
     * slack basis that is the sign of its cost (the same perturbation as before), from a given basis (the rounds of the row
     * sifting) the sign of its reduced cost. By the sign of the cost, a column at its upper bound with a positive cost and a
     * reduced cost near zero (an optimal face: satellites2-25, LP, 97 % of its columns of cost zero) got a reduced cost of
     * the wrong sign, the dual method flipped it to its other bound, and each round began with thousands of primal
     * infeasibilities where the rows that joined were 59 (round 1: 378 flips, 4,005 infeasible rows; the six rounds 1,500 to
     * 3,500 iterations each, now 435 to 780). BRISK_SXPERTMOVE=0: by the sign of the cost */
    static int pmove = -1; if (pmove < 0) { const char *e_ = getenv("BRISK_SXPERTMOVE"); pmove = !(e_ && atoi(e_) == 0); }
    /* (12 November 2026, an experiment: BRISK_SXPERTLEN=w, a column's perturbation times (its entries / the mean entries)^w when longer than the mean) */
    static double plen_w = -1; if (plen_w < 0) { const char *e_ = getenv("BRISK_SXPERTLEN"); plen_w = e_ ? atof(e_) : 0; }
    const double plen_avg = n > 0 ? fmax(1.0, (double)S->Ap[n] / n) : 1.0;
    /* (14 November 2026, an experiment: BRISK_SXPERTUNITS=1) the rule by the largest cost (mode 1) applied in the problem's own units,
     * column by column: xp = 5e-7 bcu (|c_j| + 1) (1 + u) with c_j and bcu (the largest |c_j|, its fourth root above 100) in the units
     * of the problem as given, converted to the scaled problem by cscale_j / costscale. The rule in the scaled problem gives a column
     * (|c_j| + 1 / cscale_j) times the base in the problem's units: a column scaled down (long columns of large entries) gets less
     * (satellites4-25's shuffled copies, LP: the 15-entry columns 0.64 times the 2-entry columns' perturbation, Gurobi's 2.6 times;
     * bench/lp_r16/satcopy/REPORT.md). With BRISK_SXPERTLEN the length factor follows */
    static int punits_ = -1; if (punits_ < 0) { const char *e_ = getenv("BRISK_SXPERTUNITS"); punits_ = e_ && atoi(e_) != 0; }
    double bcu_ = 0;
    if (punits_ && mode == 1) {
        for (int j = 0; j < n; j++) { const double cj = fabs(S->cost0[j]) * S->costscale / (S->cscale ? S->cscale[j] : 1.0); if (cj > bcu_) bcu_ = cj; }
        if (bcu_ > 100) bcu_ = sqrt(sqrt(bcu_));
        if (bcu_ == 0) bcu_ = 1;
        if ((double)nbox / S->N < 0.01 && bcu_ > 1) bcu_ = 1;
        if (S->verbose >= 2) printf("  cost perturbation in the problem's units: largest cost (fourth root above 100) %.3g\n", bcu_);
    }
    /* (14 November 2026, an experiment: BRISK_SXPERTNORM=1) each column's perturbation times its 2-norm in the scaled problem over the mean
     * of those norms. The ratio test compares d_j / |alpha_rj|, and alpha_rj = rho_r a_j sums more terms in a long column: an even
     * perturbation breaks ties among columns of equal true reduced cost in favour of long columns (satellites4-25's shuffled copies, LP,
     * bench/lp_r16/satcopy/REPORT.md); one proportional to the norm leaves them on an equal footing, whatever the optimum needs (qap15,
     * LP: its 90 long columns all basic at the optimum, held back by a factor that favours short columns) */
    static int pnorm_ = -1; if (pnorm_ < 0) { const char *e_ = getenv("BRISK_SXPERTNORM"); pnorm_ = e_ && atoi(e_) != 0; }
    double *cnorm_ = NULL;
    if (pnorm_ && n > 0) {
        cnorm_ = malloc(sizeof(double) * n); double sn_ = 0; int nn_ = 0;
        for (int j = 0; j < n; j++) { double q_ = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) q_ += S->Ax[p] * S->Ax[p]; cnorm_[j] = sqrt(q_); if (q_ > 0) { sn_ += cnorm_[j]; nn_++; } }
        sn_ = nn_ > 0 ? sn_ / nn_ : 1.0;
        for (int j = 0; j < n; j++) cnorm_[j] = cnorm_[j] > 0 ? cnorm_[j] / sn_ : 1.0;
    }
    for (int j = 0; j < n; j++) {
        const double l = S->lo0[j], u = S->up0[j];
        double xp;
        if (mode == 2) {
            const int nu = S->Ap[j + 1] - S->Ap[j];
            double xi = (100 * S->told + kpsi * fabs(S->cost0[j])) * (nu >= 1 ? kw[nu < 10 ? nu - 1 : 9] : 1.0);
            if (getenv("SXPERT_BASE")) xi *= atof(getenv("SXPERT_BASE"));
            for (int t = 0; t < 40 && xi > kxmax; t++) xi *= 0.1;
            for (int t = 0; t < 40 && xi < kxmin; t++) xi *= 10;
            xp = 0.5 * xi * (1 + urand(S));
        } else
            xp = mode == 1 ? (fabs(S->cost0[j]) * S->costscale + one) * base * (1 + urand(S))
                           : (fabs(S->cost0[j]) + one) * base * (1 + urand(S));
        {   /* (13 November 2026, an experiment: SXPERT_UNSC=x) the perturbation set in the problem's own units: x (|c_j| + 1) (1 + u),
             * converted to the scaled problem (times cscale_j / costscale); Gurobi's on sat4p1 (LP) is about 1e-6 in those units,
             * ours 4e-4 there and varying with the columns' scale factors */
            static double pun_ = -1; if (pun_ < 0) { const char *e_ = getenv("SXPERT_UNSC"); pun_ = e_ ? atof(e_) : 0; }
            if (pun_ > 0 && mode != 2) xp = pun_ * (fabs(S->cost0[j]) * S->costscale / (S->cscale ? S->cscale[j] : 1.0) + 1.0) * (1 + urand(S)) * (S->cscale ? S->cscale[j] : 1.0) / S->costscale; }
        if (punits_ && mode == 1) {   /* (the experiment BRISK_SXPERTUNITS above) */
            const double cs_ = S->cscale ? S->cscale[j] : 1.0;
            xp = 5e-7 * bcu_ * (getenv("SXPERT_BASE") ? atof(getenv("SXPERT_BASE")) : 1.0) * (fabs(S->cost0[j]) * S->costscale / cs_ + 1.0) * (1 + urand(S)) * cs_ / S->costscale;
        }
        {   /* (13 November 2026, an experiment: SXPERT_SPREAD=1; after SXPERT_UNSC, so that the two combine) the random factor uniform on (0, 2) instead of (1, 2): perturbations
             * spread from near zero to twice the base, as Gurobi's are (sat4p1, LP: its perturbed costs give our method bases 30
             * times sparser) */
            static int spr_ = -1; if (spr_ < 0) { const char *e_ = getenv("SXPERT_SPREAD"); spr_ = e_ ? atoi(e_) : 0; }
            if (spr_ == 1 && mode != 2) xp *= 2.0 * urand(S) / 1.5; }
        if (cnorm_ && mode != 2) xp *= cnorm_[j];   /* (the experiment BRISK_SXPERTNORM above) */
        if (plen_w > 0 && mode != 2) { const double r_ = (double)(S->Ap[j + 1] - S->Ap[j]) / plen_avg; if (r_ > 1) xp *= pow(r_, plen_w); }   /* (an experiment, BRISK_SXPERTLEN) */
        if (l <= -SX_INF && u >= SX_INF) continue;
        if (l == u) continue;
        /* (23 October 2026) the nonbasic columns only: from the slack basis that is every column; from a given basis
         * (the rounds of the row sifting) a basic column's perturbation moves the duals and leaves nonbasic reduced
         * costs infeasible by its size, and the round began with a dual phase 1 (comp21-2idx: 1,705 iterations) */
        /* (13 November 2026, an experiment: BRISK_SXPERTBASIC=1) the basic columns perturbed too, the reduced costs it leaves of
         * the wrong sign repaired by the flips and shifts that follow (sat4p1: a start with 15,000 structural columns basic left
         * them, the rows' duals and the logicals' reduced costs exactly zero, and the method pivoted through that unperturbed part) */
        static int pbas_ = -1; if (pbas_ < 0) { const char *e_ = getenv("BRISK_SXPERTBASIC"); pbas_ = e_ && atoi(e_) != 0; }
        if (S->pos[j] >= 0) { if (pbas_ && S->cost[j] == 0) S->cost[j] += (urand(S) < 0.5 ? -xp : xp); continue; }
        if (u >= SX_INF) S->cost[j] += xp;
        else if (l <= -SX_INF) S->cost[j] -= xp;
        else if ((S->cost[j] == 0 || pmove) && S->pos[j] < 0) S->cost[j] += S->move[j] < 0 ? -xp : xp;   /* (the bound it is at) */
        else S->cost[j] += S->cost[j] >= 0 ? xp : -xp;
    }
    {   /* (13 November 2026, an experiment: BRISK_SXPERTLOG=1) the rows' logicals perturbed as a structural column of cost zero is,
         * in the direction that keeps a nonbasic one at its bound, a basic one by a random sign; else the 1e-12 of HiGHS */
        static int plog_ = -1; if (plog_ < 0) { const char *e_ = getenv("BRISK_SXPERTLOG"); plog_ = e_ && atoi(e_) != 0; }
        for (int i = 0; i < S->m; i++) {
            const int j = n + i;
            if (!plog_ || S->lo0[j] == S->up0[j]) { S->cost[j] += (0.5 - urand(S)) * 1e-12; continue; }
            const double xp = mode == 2 ? 0.5 * (100 * S->told) * (1 + urand(S)) : one * base * (1 + urand(S));
            if (S->pos[j] >= 0) S->cost[j] += urand(S) < 0.5 ? -xp : xp;
            else if (S->up0[j] >= SX_INF) S->cost[j] += xp;
            else if (S->lo0[j] <= -SX_INF) S->cost[j] -= xp;
            else S->cost[j] += S->move[j] < 0 ? -xp : xp;
        }
    }
    {   const char *pd_ = getenv("BRISK_SXPERTDUMP");   /* (13 November 2026, a diagnostic: each column's perturbation in the problem's own units) */
        if (pd_) { FILE *fp_ = fopen(pd_, "w"); if (fp_) { for (int j = 0; j < n; j++) fprintf(fp_, "%d %d %.6e %d\n", j, S->Ap[j + 1] - S->Ap[j], (S->cost[j] - S->cost0[j]) * S->costscale / (S->cscale ? S->cscale[j] : 1.0), S->pos[j] >= 0); fclose(fp_); } } }
    free(cnorm_);
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
/* (27 October 2026) each column's factor of the first perturbation, (|c_j| + 1)(1 + r_j) with r_j in [0, 1) drawn from the
 * column's index (not from a sequence shared by the columns), kept while the costs stay: a re-solve of the mixed-integer
 * solver applies it by one plain pass, where it drew a random number per column every time (fiball: perturb_first 6.7 %
 * of the tree search, re-solves of about five iterations). The largest cost is kept with them */
static double col_rand(int j) {
    unsigned long long z = (unsigned long long)j * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
static void perturb_first(Sx *S) {
    static int oldpert = -1; if (oldpert < 0) oldpert = getenv("BRISK_SXMPERTRAND") != NULL;   /* (the random sequence as before) */
    double bigc = 0;
    if (!oldpert && S->pert_u_ok) bigc = S->pert_bigc;
    else for (int j = 0; j < S->n; j++) if (fabs(S->cost0[j]) > bigc) bigc = fabs(S->cost0[j]);
    int nbox = 0;
    if (S->nbox_given_ok) { nbox = S->nbox_given; S->nbox_given_ok = 0; }     /* (counted by sxm_solve on the same bounds) */
    else for (int j = 0; j < S->N; j++) nbox += is_boxed(S, j);
    double bc = bigc * S->costscale; if (bc > 100) bc = sqrt(sqrt(bc));
    if (bc == 0) bc = 1;     /* (28 October 2026: no cost at all, as in perturb_costs) */
    if ((double)nbox / S->N < 0.01 && bc > 1) bc = 1;
    S->pert_base = 5e-7 * bc / S->costscale; S->pert_level = -1;
    if (oldpert) perturb_more(S);
    else {
        if (!S->pert_u_ok) {
            if (!S->pert_u) S->pert_u = xmalloc(sizeof(double) * ((size_t)S->n + 1));
            for (int j = 0; j < S->n; j++) S->pert_u[j] = (fabs(S->cost0[j]) * S->costscale + 1) * (1 + col_rand(j));
            S->pert_bigc = bigc; S->pert_u_ok = 1;
        }
        S->pert_level = 0; S->nrepert++;
        /* (13 November 2026, an experiment: BRISK_SXXOPERTTOL=1, ISSUES 881) the crossover's dual method, near the optimum: the perturbation
         * a quarter of the dual tolerance times (|c_j| + 1)(1 + r_j), so that its removal leaves no dual infeasibility beyond the tolerance
         * to the clean-up (neos-3214367-sovi's root LP: the perturbation of the usual size left 0.49, and the primal clean-up and the dual
         * method from the restored bounds ended in numerical difficulties) */
        const double f = S->xo_pert_tol ? 0.25 * S->told / (S->costscale > 0 ? 1.0 : 1.0) : S->pert_base;
        for (int j = 0; j < S->n; j++) {
            if (S->pos[j] >= 0 || S->lo0[j] == S->up0[j] || S->move[j] == 0) continue;
            double xp = S->pert_u[j] * f;
            if (S->xo_pert_tol == 2) xp = f * (1 + col_rand(j));   /* (14 November 2026, a test: =2 without the cost's factor (|c_j| + 1), so that every removal stays below half the tolerance) */
            {   static double pun_ = -1; if (pun_ < 0) { const char *e_ = getenv("SXPERT_UNSC"); pun_ = e_ ? atof(e_) : 0; }   /* (the experiment SXPERT_UNSC here too) */
                if (pun_ > 0) xp = pun_ * S->pert_u[j] * (S->cscale ? S->cscale[j] : 1.0) / S->costscale; }
            if (S->move[j] > 0) { S->cost[j] += xp; S->d[j] += xp; } else { S->cost[j] -= xp; S->d[j] -= xp; }
        }
    }
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
    if (brisk_flag_get(S->stop)) return 6;
#ifdef _OPENMP
    /* (28 October 2026) S->nthreads follows the count too: the parallel parts test it (> 1 only), and the dual simplex
     * method beside the interior-point method, set up on one thread, ran on one after it was given all of them */
    if (S->threads) { const int t = *S->threads; if (t > 0 && t != S->cur_threads) { omp_set_num_threads(t); S->cur_threads = t; S->nthreads = t; } }
#endif
    if (S->maxit > 0 && S->iter >= S->maxit) return 3;
    if (S->env_stopit > 0 && S->iter >= S->env_stopit) return 3;   /* (BRISK_SXSTOPIT: experiments) */
    if (S->tlim > 0 && (++S->nchk & 15) == 0 && now() - S->t0 > S->tlim) return 6;
    {   /* (13 November 2026, an experiment: BRISK_LPAUTOPROG=1; ISSUES 880) the simplex method's progress weighed when its work budget
         * (the automatic choice's: the interior-point method's predicted work) runs out: every 500 iterations the work and the count of
         * primal infeasibilities are noted; at the budget, the remaining work projected from the count's fall since the last note; when it
         * is under half the budget, the budget is raised by twice the projection (at most twice). neos-3654993-kolva's root LP (LP
         * relaxation, all costs zero): stopped after 12,401 of the 13,428 iterations it needed, then the crossover failed */
        static int pg_ = -1; if (pg_ < 0) { const char *e_ = getenv("BRISK_LPAUTOPROG"); pg_ = e_ && atoi(e_) == 1; }
        if (pg_ && S->workbudget > 0) {
            if (S->iter - S->pg_it >= 500 || S->pg_it == 0) {
                if (S->work <= S->workbudget || S->pg_it == 0) { int n_ = 0; for (int i = 0; i < S->m; i++) n_ += infeas(S, i) > 0;
                    if (S->pg_it > 0 && S->work > S->workbudget * 0.0) { } S->pg_it = S->iter ? S->iter : 1; S->pg_work = S->work; S->pg_ninf = n_; } }
            if (S->work > S->workbudget && S->pg_ext < 2 && S->pg_it > 1) {
                int n_ = 0; for (int i = 0; i < S->m; i++) n_ += infeas(S, i) > 0;
                const double dw = S->work - S->pg_work; const int dn = S->pg_ninf - n_;
                if (dw > 0 && dn > 0) { const double rem = (double)n_ * dw / dn;
                    if (rem < 0.5 * S->workbudget) { S->workbudget = S->work + 2.0 * rem; S->pg_ext++;
                        if (S->verbose >= 2) printf("  the work budget raised by the progress: %d primal infeasibilities, %d fewer over %.3g work, the rest projected %.3g\n", n_, dn, dw, rem);
                        return 0; } } } }
    }
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
/* (7 November 2026, the MIP developer's finding on app1-1) the bound the duals that will be returned prove: y = B^-T c_B
 * with the working costs, then every column's reduced cost cost0_j - [A I]_j'y, basic columns included (dual_bound takes
 * theirs as zero; on an ill-conditioned basis they are not: app1-1's node LPs claimed -2.999994 where the returned duals
 * proved -3.00025 to -3.00005, and the mixed-integer solver, which checks every claim, lost 7,047 nodes in 60 s). The cutoff
 * stop is claimed only when this bound passes. BRISK_SXCUTEXACT=0: the earlier claim */
static double dual_bound_exact(Sx *S) {
    SxVec *v = &S->rho; sxv_clear(v);
    for (int i = 0; i < S->m; i++) v->val[i] = S->cost[S->basic[i]];
    sxv_reindex(v);
    double dd = 1; btran(S, v, &dd, NULL);
    const double *y = v->val;
    double s = 0; int inf = 0;
    for (int j = 0; j < S->N && !inf; j++) {
        double dj = S->cost0[j];
        if (j < S->n) { for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) dj -= S->Ax[p] * y[S->Ai[p]]; }
        else dj -= y[j - S->n];
        if (dj == 0) continue;
        const double b = dj > 0 ? S->lo0[j] : S->up0[j];
        if (fabs(b) >= SX_INF) { if (fabs(dj) > S->told) inf = 1; continue; }
        s += dj * b;
    }
    sxv_clear(v);
    return inf ? -SX_INF : s;
}
/* (12 November 2026) compute_dual and dual_bound_exact in one backward solve and one pass over the columns: both take the
 * same y = B^-T c_B (the working costs) and the same products a_j'y, subtracted in the same order from the working cost
 * (the reduced cost kept) and from the cost as given (the exact bound's term), so each result has the bits of the separate
 * calls. The cutoff's confirmation computed both one after the other; on a multiple knapsack MILP's node LPs half the
 * re-solves end at the cutoff, and the second solve and pass were 5 % of the instructions */
static double compute_dual_exact(Sx *S) {
    SxVec *v = &S->rho; sxv_clear(v);
    for (int i = 0; i < S->m; i++) v->val[i] = S->cost[S->basic[i]];
    sxv_reindex(v);
    double dd = 1; btran(S, v, &dd, NULL);
    const double *y = v->val;
    double s = 0; int inf = 0;
    for (int j = 0; j < S->N; j++) {
        const int bas = S->pos[j] >= 0;
        double dw = S->cost[j], dj = S->cost0[j];
        if (j < S->n) { for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) { const double t = S->Ax[p] * y[S->Ai[p]]; dw -= t; dj -= t; } }
        else { dw -= y[j - S->n]; dj -= y[j - S->n]; }
        S->d[j] = bas ? 0 : dw;
        if (inf || dj == 0) continue;
        const double b = dj > 0 ? S->lo0[j] : S->up0[j];
        if (fabs(b) >= SX_INF) { if (fabs(dj) > S->told) inf = 1; continue; }
        s += dj * b;
    }
    sxv_clear(v);
    return inf ? -SX_INF : s;
}
static int cutexact_on(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_SXCUTEXACT"); v = !(e_ && atoi(e_) == 0); } return v; }
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
/* (12 November 2026, a diagnostic: BRISK_SXENTDBG=1, the entering and leaving columns of 40 entries or more since the last log line) */
static void log_line(Sx *S, const char *what) {
    if (S->verbose < 2) return;
    int c; const double s = sum_pinf(S, &c);
    printf("  %-8s %8ld  obj %+.10e  pinf %.2e (%d)  %.2fs\n", what, S->iter, objective(S) * S->costscale, s, c, now() - S->t0);
    if (g_entdbg > 0) {   /* the infeasible rows whose basic column is long, and their pricing score (infeasibility^2 / weight) against the others' */
        long nl = 0, nb = 0, no = 0; double sl = 0, so = 0, mx_l = 0, mx_o = 0;
        for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; if (j < S->n && S->Ap[j + 1] - S->Ap[j] >= 40) nb++;
            const double f = infeas_merit(S, i); if (f <= 0) continue; const double sc = f * f / (S->w[i] > 0 ? S->w[i] : 1);
            if (j < S->n && S->Ap[j + 1] - S->Ap[j] >= 40) { nl++; sl += sc; if (sc > mx_l) mx_l = sc; } else { no++; so += sc; if (sc > mx_o) mx_o = sc; } }
        printf("           long basic %ld, of them infeasible %ld (mean score %.3g, largest %.3g); other infeasible rows %ld (mean score %.3g, largest %.3g)\n", nb, nl, nl ? sl / nl : 0, mx_l, no, no ? so / no : 0, mx_o); }
    if (g_entdbg > 0) { printf("           entering %ld, of them long %ld (reduced cost within the tolerance %ld); long leaving %ld\n", g_ent_all, g_ent_long, g_ent_long_d0, g_leave_long); g_ent_all = g_ent_long = g_ent_long_d0 = g_leave_long = 0; }
}

/* rebuild: factorization, primal and dual values; corrections of the duals */
static int rebuild(Sx *S) {
    refactor(S);
    {   /* (12 November 2026, an experiment: BRISK_SXWRESET=1) the steepest-edge weights reset to 1 at every rebuild */
        static int wrs_ = -1; if (wrs_ < 0) { const char *e_ = getenv("BRISK_SXWRESET"); wrs_ = e_ && atoi(e_) != 0; }
        if (wrs_) { for (int i = 0; i < S->m; i++) S->w[i] = 1.0; if (S->wk) memset(S->wk, 1, (size_t)S->m); S->wlazy = 0; }
    }
    {   /* (8 November 2026, a test: BRISK_SXEXACTWREF=1) the exact dual steepest-edge weights at every rebuild (one backward
         * solve a row): whether the updated weights' drift along the path costs iterations */
        static int exr = -1; if (exr < 0) { const char *e = getenv("BRISK_SXEXACTWREF"); exr = e && atoi(e) != 0; }
        if (exr) {
            SxVec *rho = &S->rho;
            for (int i = 0; i < S->m; i++) {
                sxv_clear(rho); rho->val[i] = 1; rho->idx[0] = i; rho->cnt = 1;
                double dd = S->dens_rho; btran(S, rho, &dd, NULL);
                double w = 0;
                if (rho->cnt >= 0) for (int t = 0; t < rho->cnt; t++) { const double v = rho->val[rho->idx[t]]; w += v * v; }
                else for (int k = 0; k < S->m; k++) w += rho->val[k] * rho->val[k];
                S->w[i] = w > 1e-4 ? w : 1e-4;
            }
            sxv_clear(rho);
            if (S->wk) memset(S->wk, 1, (size_t)S->m);
        }
    }
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
    {   static int vfp_ = -1; static double vfc_ = 0.5;   /* (BRISK_SXVFAC=P, BRISK_SXVFACC=c: the merit's factor, a test) */
        if (vfp_ < 0) { const char *e_ = getenv("BRISK_SXVFAC"); vfp_ = e_ ? atoi(e_) : 0; const char *c_ = getenv("BRISK_SXVFACC"); if (c_) vfc_ = atof(c_); }
        if (vfp_ > 0 && !S->vfac) {
            S->vfac = xmalloc(sizeof(double) * ((size_t)S->N + 1)); S->vent = xmalloc(sizeof(long) * ((size_t)S->N + 1));
            for (int j = 0; j <= S->N; j++) { S->vfac[j] = 1.0; S->vent[j] = -1; }
            S->vf_last = S->iter; S->vf_p = vfp_; S->vf_c = vfc_;
            { const char *d_ = getenv("BRISK_SXVFACD"); S->vf_d = d_ && atoi(d_) == 1; }
            S->vden = xcalloc((size_t)S->N + 1, sizeof(int));
            { const char *o_ = getenv("BRISK_SXVFACONCE"); S->vf_once = o_ && atoi(o_) == 1; } S->vonce = xcalloc((size_t)S->N + 1, 1);
        }
    }
    if (g_dserows > 0 && !S->dsefull_init && S->rscale) {   /* (BRISK_SXDSEROWS) the starting weights r_i^2, exact for the logicals' basis */
        S->dsefull_init = 1; S->wlazy = 0;
        for (int i = 0; i < m; i++) { const int j_ = S->basic[i]; S->w[i] = j_ >= S->n ? S->rscale[i] * S->rscale[i] : 1.0; if (S->wk) S->wk[i] = 1; }
    }
    if (g_dsefull > 0 && !S->dsefull_init) {   /* (BRISK_SXDSEFULL) the starting weights 1 + the row's nonbasic structural entries squared (exact for the logicals' basis) */
        S->dsefull_init = 1; S->wlazy = 0;
        for (int i = 0; i < m; i++) { double w_ = 1; const int e_ = S->Rp[i] + S->Rnb[i]; for (int p_ = S->Rp[i]; p_ < e_; p_++) w_ += S->Rx[p_] * S->Rx[p_]; S->w[i] = w_; if (S->wk) S->wk[i] = 1; }
    }
    const double tloop0 = S->tdiag ? now() : 0;
    for (;;) {
        int st = check_stop(S);
        if (st) { if (S->tdiag) S->t_loop += now() - tloop0; return st; }
        if (S->fresh) { S->fresh = 0; need_rebuild = 0; }
        if (need_rebuild || need_refactor(S)) { const double tr0 = S->tdiag ? now() : 0; rebuild(S); need_rebuild = 0; S->heap_ok = 0; S->Ifull = 1; if (S->tdiag) S->t_rebuild += now() - tr0; }
        {   /* (13 November 2026, an experiment: BRISK_SXPERTAT=k with BRISK_SXLAZYPERT=1) the first perturbation at iteration k of this
             * solve instead of at a stall: Gurobi's dual simplex method on sat4p1 (LP) perturbs between its iterations 1,000 and
             * 4,000, and which column gets which value matters (its values shuffled within each length class: dense again) */
            static long pat_ = -2; if (pat_ == -2) { const char *e_ = getenv("BRISK_SXPERTAT"); pat_ = e_ ? atol(e_) : -1; }
            if (pat_ >= 0 && S->lazy_pert && !S->perturbed && S->iter == pat_) S->pert_pending = 1; }
        if (S->pert_pending) { if (S->perturbed) perturb_more(S); else perturb_first(S); S->pert_pending = 0; }
        {   /* (12 November 2026, a diagnostic: BRISK_SXROWDBG=<row>) at the first iteration: the ten rows of the largest pricing score
             * (infeasibility squared over the steepest-edge weight) and the score and rank of the given row */
            static int rdbg_ = -2; if (rdbg_ == -2) { const char *e_ = getenv("BRISK_SXROWDBG"); rdbg_ = e_ ? atoi(e_) : -1; }
            static long rdit_ = -2; if (rdit_ == -2) { const char *e_ = getenv("BRISK_SXROWDBGIT"); rdit_ = e_ ? atol(e_) : 0; }
            if (rdbg_ >= 0 && S->iter_rdbg0 >= 0 && S->iter >= rdit_) { S->iter_rdbg0 = -1;
                double *sc_ = xmalloc(sizeof(double) * ((size_t)m + 1)); int ninf_ = 0;
                for (int i = 0; i < m; i++) { const double f = infeas(S, i); sc_[i] = f > 0 ? f * f / (S->w[i] > 0 ? S->w[i] : 1) : 0; ninf_ += f > 0; }
                int rank_ = 0; for (int i = 0; i < m; i++) if (sc_[i] > sc_[rdbg_ < m ? rdbg_ : 0]) rank_++;
                fprintf(stderr, "[rowdbg] infeasible rows %d; row %d: basic %d (%s, %d entries), infeasibility %.3g, weight %.3g, score %.3g, rank %d\n", ninf_, rdbg_, S->basic[rdbg_],
                        S->basic[rdbg_] < S->n ? "column" : "logical", S->basic[rdbg_] < S->n ? S->Ap[S->basic[rdbg_] + 1] - S->Ap[S->basic[rdbg_]] : 1, infeas(S, rdbg_), S->w[rdbg_], sc_[rdbg_], rank_);
                for (int t = 0; t < 10; t++) { int b = -1; for (int i = 0; i < m; i++) if (sc_[i] > 0 && (b < 0 || sc_[i] > sc_[b])) b = i; if (b < 0) break;
                    double we_ = 0, wf_ = 0; { SxVec *rh_ = &S->rho; sxv_clear(rh_); rh_->val[b] = 1; rh_->idx[0] = b; rh_->cnt = 1; double dd_ = S->dens_rho; btran(S, rh_, &dd_, NULL);
                        if (rh_->cnt >= 0) for (int u = 0; u < rh_->cnt; u++) we_ += rh_->val[rh_->idx[u]] * rh_->val[rh_->idx[u]]; else for (int k = 0; k < m; k++) we_ += rh_->val[k] * rh_->val[k];
                        /* (the tableau row's squared norm over the structural columns: the weight of the steepest edge in the space of all variables is we_ + this) */
                        double *rv_ = xcalloc((size_t)m + 1, sizeof(double)); if (rh_->cnt >= 0) for (int u = 0; u < rh_->cnt; u++) rv_[rh_->idx[u]] = rh_->val[rh_->idx[u]]; else for (int k = 0; k < m; k++) rv_[k] = rh_->val[k];
                        for (int jj = 0; jj < S->n; jj++) { if (S->pos[jj] >= 0) continue; double a_ = 0; for (int p_ = S->Ap[jj]; p_ < S->Ap[jj + 1]; p_++) a_ += rv_[S->Ai[p_]] * S->Ax[p_]; wf_ += a_ * a_; }
                        free(rv_); sxv_clear(rh_); }
                    fprintf(stderr, "[rowdbg]   top %d: row %d, basic %d (%s, %d entries), infeasibility %.3g, weight %.3g (exact %.3g, with the tableau row %.3g), score %.3g (with the tableau row %.3g)\n", t, b, S->basic[b], S->basic[b] < S->n ? "column" : "logical",
                            S->basic[b] < S->n ? S->Ap[S->basic[b] + 1] - S->Ap[S->basic[b]] : 1, infeas(S, b), S->w[b], we_, we_ + wf_, sc_[b], infeas(S, b) * infeas(S, b) / (we_ + wf_)); sc_[b] = -1; }
                /* (13 November 2026, BRISK_SXROWDBGVAR=<variable>: every infeasible row's exact weight, the ranking by the exact score
                 * (infeasibility squared over |row of B^-1|^2), the top ten and the rank of the row whose basic variable is the given
                 * one: a column index, or n + i for row i's logical) */
                { static long rvar_ = -2; if (rvar_ == -2) { const char *e_ = getenv("BRISK_SXROWDBGVAR"); rvar_ = e_ ? atol(e_) : -1; }
                  if (rvar_ >= 0) {
                    for (int i = 0; i < m; i++) { const double f = infeas(S, i); sc_[i] = 0; if (f <= 0) continue;
                        SxVec *rh_ = &S->rho; sxv_clear(rh_); rh_->val[i] = 1; rh_->idx[0] = i; rh_->cnt = 1; double dd_ = S->dens_rho; btran(S, rh_, &dd_, NULL);
                        double we_ = 0; if (rh_->cnt >= 0) for (int u = 0; u < rh_->cnt; u++) we_ += rh_->val[rh_->idx[u]] * rh_->val[rh_->idx[u]]; else for (int k = 0; k < m; k++) we_ += rh_->val[k] * rh_->val[k];
                        sxv_clear(rh_); sc_[i] = f * f / (we_ > 1e-12 ? we_ : 1e-12); }
                    const int pv_ = rvar_ < S->N ? S->pos[rvar_] : -1;
                    int rk_ = -1; if (pv_ >= 0) { rk_ = 0; for (int i = 0; i < m; i++) if (sc_[i] > sc_[pv_]) rk_++; }
                    fprintf(stderr, "[rowdbg exact] variable %ld at row %d: infeasibility %.3g, exact score %.3g, rank %d of %d infeasible\n", rvar_, pv_, pv_ >= 0 ? infeas(S, pv_) : 0.0, pv_ >= 0 ? sc_[pv_] : 0.0, rk_, ninf_);
                    for (int t = 0; t < 10; t++) { int b = -1; for (int i = 0; i < m; i++) if (sc_[i] > 0 && (b < 0 || sc_[i] > sc_[b])) b = i; if (b < 0) break;
                        fprintf(stderr, "[rowdbg exact]   top %d: row %d, basic %d (%s, %d entries), infeasibility %.3g, exact score %.3g\n", t, b, S->basic[b], S->basic[b] < S->n ? "column" : "logical",
                                S->basic[b] < S->n ? S->Ap[S->basic[b] + 1] - S->Ap[S->basic[b]] : 1, infeas(S, b), sc_[b]); sc_[b] = -1; } } }
                free(sc_); } }
        static long logint_ = -1; if (logint_ < 0) { const char *e_ = getenv("BRISK_SXLOGINT"); logint_ = e_ && atol(e_) > 0 ? atol(e_) : 1000; }   /* (12 November 2026, a diagnostic: the log's interval) */
        if (S->iter - last_log >= logint_) {
            log_line(S, S->phase == 1 ? "dual ph1" : "dual"); last_log = S->iter;
            /* (10 November 2026) the infeasible rows for the race's retirement rule (main.c, BRISK_LPRACERETIRE): a pass over
             * the rows every 1,000 iterations, in phase 2 only (phase 1's are of another problem) */
            if (S->prog && S->phase != 1) { int c_; (void)sum_pinf(S, &c_); const double t_ = now();
                if (S->prog->nsamp == 0) { S->prog->ninf0 = c_; S->prog->tpub0 = t_; }
                S->prog->ninf = c_; S->prog->tpub = t_; S->prog->nsamp++; }
        }
        if (brisk_stab_due(S->iter)) { int c_; const double s_ = sum_pinf(S, &c_); brisk_stab_row(S->iter, S->phase == 1 ? "dual, phase 1" : "dual", g_sx_tabsign * objective(S) * S->costscale + g_sx_taboff, s_, c_); }
        /* (22 October 2026) the objective cutoff (sxm_solve): the dual bound of the current duals with the problem's own
         * costs and bounds passes the cutoff: the optimum lies beyond it, and the node can be pruned */
        if (S->cut_on && S->phase == 2 && S->iter >= S->cut_skip && (S->N <= 20000 || S->iter % 10 == 0)) {
            const double b = dual_bound(S);
            const double cv = S->cut_val + 1e-9 * (1 + fabs(S->cut_val));
            /* (27 October 2026) confirmed on reduced costs computed afresh from the basis before the stop is claimed: the
             * updated ones drift, and the mixed-integer solver checks the claim with the duals returned (sxm_values gives,
             * at the cutoff, the duals of this bound). sp150x300d with substitutions: 23 claims its check could not prove.
             * When the fresh bound does not pass, the loop goes on from the fresh reduced costs */
            if (b > cv) {
                static int fuse = -1; if (fuse < 0) { const char *e_ = getenv("BRISK_SXCUTFUSE"); fuse = !(e_ && atoi(e_) == 0); }
                const int fz = fuse && cutexact_on();
                double bxf = 0;
                if (fz) bxf = compute_dual_exact(S); else compute_dual(S);
                const double bf = dual_bound(S);
                if (bf > cv && cutexact_on()) { const double be = fz ? bxf : dual_bound_exact(S); if (be > cv) { S->cut_bound = be; return 7; } S->cut_skip = S->iter + 50; }
                else if (bf > cv) { S->cut_bound = bf; return 7; }
                int ns = 0; if (correct_duals(S, &ns)) compute_primal(S);
                S->nshift += ns; S->heap_ok = 0; S->Ifull = 1;
            }
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
                if (b2 > cv && cutexact_on()) { const double be = dual_bound_exact(S); if (be > cv) { S->cut_bound = be; return 7; } S->cut_skip = S->iter + 50; }
                else if (b2 > cv) { S->cut_bound = b2; return 7; }
            }
        }
        double tdg = S->tdiag ? now() : 0;
        int r = chuzr(S);
        {   /* (13 November 2026, a diagnostic: BRISK_SXFORCEVARS=<file> of lines "iteration variable", variables numbered as the
             * trace numbers them, columns 0..n-1 and n + i for row i's logical) the leaving variable forced at those iterations of
             * this solve, when it is basic and infeasible: to compare the entering choices with another solver's from its basis */
            static int nfv_ = -1; static long *fit_ = NULL; static int *fvar_ = NULL;
            if (nfv_ < 0) { nfv_ = 0; const char *e_ = getenv("BRISK_SXFORCEVARS"); FILE *ff_ = e_ ? fopen(e_, "r") : NULL;
                if (ff_) { long a_; int b_; int cap_ = 0; while (fscanf(ff_, "%ld %d", &a_, &b_) == 2) { if (nfv_ >= cap_) { cap_ = cap_ ? 2 * cap_ : 256; fit_ = realloc(fit_, sizeof(long) * cap_); fvar_ = realloc(fvar_, sizeof(int) * cap_); } fit_[nfv_] = a_; fvar_[nfv_] = b_; nfv_++; } fclose(ff_); } }
            for (int t_ = 0; t_ < nfv_; t_++) if (fit_[t_] == S->iter) {
                const int v_ = fvar_[t_], p_ = v_ >= 0 && v_ < S->N ? S->pos[v_] : -1;
                {   /* (13 November 2026, a diagnostic: BRISK_SXTRACEFORCED=1 with BRISK_SXTRACE) the forced row against ours on an "F" line:
                     * iteration forced-variable its-rank key infeasibility weight | ours: variable key infeasibility weight */
                    static int tf_ = -1; if (tf_ < 0) { const char *e2_ = getenv("BRISK_SXTRACEFORCED"); tf_ = e2_ ? atoi(e2_) : 0; }
                    if (tf_ && S->trace && p_ >= 0 && r >= 0) { int rk_ = 0; const int inh_ = S->hpos[p_] >= 0;
                        if (inh_) for (int q_ = 0; q_ < S->nheap; q_++) if (S->hkey[S->heap[q_]] > S->hkey[p_]) rk_++;
                        fprintf(S->trace, "F %ld %d rank %d key %.6g inf %.6g w %.6g wk %d | %d key %.6g inf %.6g w %.6g wk %d nheap %d\n", S->iter, v_, inh_ ? rk_ : -1, inh_ ? S->hkey[p_] : 0.0, infeas(S, p_), S->w[p_], S->wk ? S->wk[p_] : 1,
                                S->basic[r], S->hkey[r], infeas(S, r), S->w[r], S->wk ? S->wk[r] : 1, S->nheap);
                        /* (BRISK_SXTRACEFORCED=2: and the forced row's rank by the exact steepest-edge score, every heap row's |row of B^-1|^2
                         * computed: an "E" line: iteration forced-variable exact-rank exact-weight | the best by the exact score: variable
                         * infeasibility exact-weight; ties within 1e-12 counted as ahead only if a smaller variable index) */
                        if (tf_ >= 2 && inh_) { SxVec *v2_ = &S->wv; double fw_ = 0, fs_ = 0, bs_ = -1, bw_ = 0; int bi_ = -1, erk_ = 0;
                            double *sc_ = xmalloc(sizeof(double) * (size_t)S->nheap);
                            for (int q_ = 0; q_ < S->nheap; q_++) { const int i_ = S->heap[q_]; sxv_clear(v2_); v2_->val[i_] = 1; v2_->idx[0] = i_; v2_->cnt = 1;
                                double dd_ = S->dens_rho; btran(S, v2_, &dd_, NULL); double w_ = 0;
                                if (v2_->cnt >= 0) for (int t_ = 0; t_ < v2_->cnt; t_++) { const double a_ = v2_->val[v2_->idx[t_]]; w_ += a_ * a_; } else for (int k_ = 0; k_ < S->m; k_++) w_ += v2_->val[k_] * v2_->val[k_];
                                sxv_clear(v2_); const double f_ = infeas_merit(S, i_); sc_[q_] = f_ * f_ / w_;
                                if (i_ == p_) { fw_ = w_; fs_ = sc_[q_]; }
                                if (sc_[q_] > bs_ * (1 + 1e-12) || (sc_[q_] >= bs_ * (1 - 1e-12) && bi_ >= 0 && S->basic[i_] < S->basic[bi_])) { bs_ = sc_[q_]; bi_ = i_; bw_ = w_; } }
                            for (int q_ = 0; q_ < S->nheap; q_++) { const int i_ = S->heap[q_]; if (sc_[q_] > fs_ * (1 + 1e-12) || (sc_[q_] >= fs_ * (1 - 1e-12) && S->basic[i_] < v_)) erk_++; }
                            free(sc_);
                            fprintf(S->trace, "E %ld %d erank %d ew %.6g inf %.6g | %d inf %.6g ew %.6g\n", S->iter, v_, erk_, fw_, infeas(S, p_), S->basic[bi_], infeas(S, bi_), bw_); } } }
                if (p_ >= 0 && infeas(S, p_) > 0) r = p_;
                else fprintf(stderr, "   [force: iteration %ld, variable %d %s]\n", S->iter, v_, p_ < 0 ? "not basic" : "feasible");
                break; }
        }
        if (S->tdiag) { const double t1 = now(); S->t_chuzr += t1 - tdg; tdg = t1; }
        if (r < 0) {
            if (S->F.nupd > 0 && !verified && !S->noverify) { need_rebuild = 1; verified = 1; continue; }   /* confirm on fresh values */
            if (S->tdiag) S->t_loop += now() - tloop0;
            return 0;
        }
        verified = 0;
        const double w_r = S->w[r];   /* (the trace) */
        const int jl = S->basic[r];
        const double xr = S->x[jl];
        const double delta = xr < S->lo[jl] ? xr - S->lo[jl] : xr - S->up[jl];
        /* BTRAN e_r */
        sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
        btran(S, rho, &S->dens_rho, &S->upart);
        if (rho->cnt < 0) sxv_reindex(rho);
        double wr = 0; for (int k = 0; k < rho->cnt; k++) wr += rho->val[rho->idx[k]] * rho->val[rho->idx[k]];
        if (S->tdiag) tdg = now();
        price(S, rho);
        if (S->tdiag) S->t_row += now() - tdg;
        /* (12 November 2026, an experiment: BRISK_SXDSEFULL=1) the steepest edge in the space of all variables (Forrest and
         * Goldfarb 1992): the weight of row r is the squared norm of its tableau row over every column, |rho_r|^2 (the
         * logicals) + the pivot row's structural entries squared + 1 when the leaving variable is a structural column; the
         * weights' solve then takes rho_r + A (A' rho_r) instead of rho_r. uccase7 (LP): at Gurobi's basis of iteration 800
         * these weights rank the rows as Gurobi chooses them, ours by |rho_r|^2 do not (REPORT_lp_r16) */
        /* (12 November 2026, an experiment: BRISK_SXDSEROWS=1) the steepest-edge weights measured in the problem's own units: the
         * row of the unscaled basis's inverse is c_j (row of B^-1) R, so with the infeasibility unscaled the column's scale cancels
         * and the weight is sum_k (rho_rk r_k)^2; the weights' solve takes r^2 rho_r. Gurobi's default pricing norms keep its bases
         * sparse on uccase7 (LP), its NormAdjust 0 and 1 give bases as dense as ours (REPORT_lp_r16) */
        if (g_dserows > 0 && S->rscale) { wr = 0; for (int k = 0; k < rho->cnt; k++) { const int i = rho->idx[k]; const double v = rho->val[i] * S->rscale[i]; wr += v * v; } }
        if (g_dsefull > 0) { for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; if (j < S->n) wr += S->row.val[j] * S->row.val[j]; } if (S->basic[r] < S->n) wr += 1; }
        int nflip;
        if (S->tdiag) tdg = now();
        int q = chuzc_dual(S, delta, &nflip);
        if (S->tdiag) S->t_chuzc += now() - tdg;
        if (q < 0) {
            if (S->F.nupd > 0) { need_rebuild = 1; S->w[r] = wr; S->heap_ok = 0; continue; }
            /* a tiny infeasibility that no column can remove: the bound is moved to the value (restored at the end) */
            const double bd = delta < 0 ? S->lo[jl] : S->up[jl];
            if (fabs(delta) <= 1e-6 * (1 + fabs(bd))) {
                if (delta < 0) S->lo[jl] = xr; else S->up[jl] = xr;
                if (S->tiny_on) {
                    if (S->tiny_n < S->N + 1) { free(S->tiny); S->tiny = xcalloc(S->N + 1, 1); S->tiny_n = S->N + 1; }
                    S->tiny[jl] = 1;
                }
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
        const int nflip_pre_ = nflip;
        if (S->env_grpbf > 0 && S->fg_n > 1) q = final_group_bf(S, q, delta, &nflip);
        {   /* (13 November 2026, an experiment: BRISK_SXDEGSPARSE=1) a final group whose members all have a step of zero (reduced
             * costs zero: any member keeps the duals' signs): the member whose column B^-1 a_q is the sparsest enters (its |alpha| at
             * least gamma times the group's largest), not the largest |alpha|. Gurobi on satellites4-25's copy (LP), from the same
             * bases: its entering variable is the sparsest member in 14 of 15 such groups (3 nonzeros against 226 to 2,056 for the
             * others; ours took the largest |alpha|, columns of 1,420-2,056); bench/lp_r16/slack/grpfeat.py. One solve a member */
            static int ds_ = -1; if (ds_ < 0) { const char *e_ = getenv("BRISK_SXDEGSPARSE"); ds_ = e_ ? atoi(e_) : 0; }   /* (=2: each member's count kept until the next factorization) */
            { static int dw_ = -1; if (dw_ < 0) { const char *e_ = getenv("BRISK_SXDEGWS"); dw_ = !(e_ && atoi(e_) == 0); g_dsws = dw_; } }
            if (ds_ >= 2 && !S->dsnz) { S->dsnz = xmalloc(sizeof(int) * ((size_t)S->N + 1)); S->dsst = xmalloc(sizeof(long) * ((size_t)S->N + 1)); for (int j = 0; j <= S->N; j++) S->dsst[j] = -1; }
            S->ds_tq = -1;
            /* (13 November 2026, agent X; BRISK_SXDEGALL=1: in every solve, as first built) the rule acts only in the dual simplex
             * method of a solve from the logicals' basis. Its case is the fill-in of the basis along a long path from a sparse start
             * (Gurobi's from the slack basis on the satellites LPs; the row sifting's rounds, solved as from that basis, included); a solve from a given basis (the crossover's clean-up, a re-solve
             * of sxm_solve, the clean-up rounds after the perturbation is removed) is short, its fill is that of the basis given, and
             * a member of larger dual step there gives up the dual feasibility of the duals given: neos-4531126-vouga (LP), the
             * crossover's clean-up 620 -> 3,846 iterations with the rule (agent C) */
            static int dal_ = -1; if (dal_ < 0) { const char *e_ = getenv("BRISK_SXDEGALL"); dal_ = e_ && atoi(e_) > 0; }
            /* (14 November 2026, a test: BRISK_SXXODEG=1, ISSUES 881) in the crossover's dual method once its costs are perturbed (at a
             * stall), and nowhere else: the perturbation spreads the ties of step zero and the ratio test's choice among them fills the
             * basis (ns1904248, LP: the entering columns' solves 4,073 -> 11,062 entries on average with the perturbation; with the rule
             * 2,165, 63,351 -> 25,789 iterations, 27.5 -> 8.4 s), while before the perturbation the path from the given basis is short and
             * the rule's larger dual steps give up the given duals (neos-4531126-vouga, LP: the crossover's clean-up 620 -> 3,846
             * iterations with the rule on throughout, agent C) */
            const int xog_ = S->xo_deg && (S->perturbed || S->xo_deg == 2);   /* (=2: throughout the crossover's dual method) */
            if (ds_ > 0 && S->fg_n > 1 && q >= 0 && ((S->ds_scratch && !S->xo_deg_only) || dal_ || xog_)) {
                const int g0_ = nflip, g1_ = nflip + S->fg_n; int allz_ = 1; double amx_ = 0;
                for (int k = g0_; k < g1_; k++) { if (S->cr[k] != 0) { allz_ = 0; break; } if (S->ca[k] > amx_) amx_ = S->ca[k]; }
                /* (BRISK_SXDEGSPARSE=4: any final group the pass over it left whole, no member flipped: its members all lie within the
                 * Harris bound, and the sparsest of those of the right sign and |alpha| at least gamma times the largest enters) */
                if (ds_ >= 4 && ds_ != 5 && !allz_ && nflip == nflip_pre_) { allz_ = 1; amx_ = 0; for (int k = g0_; k < g1_; k++) if (S->ca[k] > amx_) amx_ = S->ca[k]; }
                int qpos_ = -1; for (int k = g0_; k < g1_; k++) if (S->cand[k] == q) { qpos_ = k; break; }
                /* (BRISK_SXDEGSPARSE=5, a cheap test: the last member of the candidate list of the right sign and |alpha| at least gamma
                 * times the largest, no solve; Gurobi's choice was that member in 14 of 14 zero-step groups on satellites4-25's copy) */
                if (ds_ == 5 && qpos_ >= 0) { if (!allz_ && nflip == nflip_pre_) { allz_ = 1; amx_ = 0; for (int k = g0_; k < g1_; k++) if (S->ca[k] > amx_) amx_ = S->ca[k]; }
                    if (allz_) { for (int k = g1_ - 1; k >= g0_; k--) if (S->cr[k] >= 0 && S->ca[k] >= S->env_grpbf * amx_) { q = S->cand[k]; break; } }
                    allz_ = 0; }
                if (allz_ && qpos_ >= 0) {
                    static int dst_ = -1; if (dst_ < 0) { const char *e_ = getenv("BRISK_SXDEGSTAT"); dst_ = e_ && atoi(e_) > 0; }
                    const double dt0_ = S->tdiag || dst_ ? now() : 0; const int q0_ = q; const long ns0_ = S->ds_solve;
                    S->ds_grp++;
                    double sl_ = fabs(delta); for (int k = 0; k < g0_; k++) { const int j = S->cand[k]; sl_ -= S->ca[k] * (S->up[j] - S->lo[j]); }   /* (the infeasibility left after the earlier groups' flips) */
                    SxVec *v_ = &S->wv; long bestnz_ = -1; int bq_ = q;
                    const double sq0_ = fmax(S->cr[qpos_], 0) / S->ca[qpos_];   /* (the dual step of the member the ratio test chose) */
                    static int dov_ = -1; if (dov_ < 0) { const char *e_ = getenv("BRISK_SXDEGOVER"); dov_ = e_ && atoi(e_) > 0; }
                    double ov0_ = 0; { const int jj_ = q; const double rg_ = S->move[jj_] != 0 ? S->up[jj_] - S->lo[jj_] : SX_INF; if (rg_ < SX_INF) ov0_ = fmax(0, sl_ / S->ca[qpos_] - rg_); }   /* (how far the chosen member leaves its box) */
                    /* (BRISK_SXDEGSPARSEK=k: at most k fresh solves a group, the members whose count is kept counted for free; with
                     * the costs unperturbed the groups can hold thousands of members: roi5alpha10n8, LP, 10 ms an iteration) */
                    static int dk_ = -1; if (dk_ < 0) { const char *e_ = getenv("BRISK_SXDEGSPARSEK"); dk_ = e_ ? atoi(e_) : 0; }
                    int nsolve_ = 0;
                    /* (13 November 2026, BRISK_SXDEGEST=K, K >= 1: a cheaper count) each member's count kept for good, not only until the next
                     * factorization (a member never counted is solved); the K best members by their kept counts are solved afresh and the
                     * sparsest of them enters. The rule as first built solves every member whose count is older than the factorization */
                    static int de_ = -1; if (de_ < 0) { const char *e_ = getenv("BRISK_SXDEGEST"); de_ = e_ ? atoi(e_) : 0; if (de_ > 8) de_ = 8; }
                    int tk_[8]; long te_[8]; int nt_ = 0;
                    /* (13 November 2026) the members that qualify counted first: with one, it enters without a solve (it is the sparsest of
                     * one); with none, the ratio test's choice stays; only with two or more are counts needed (BRISK_SXDEGONE=0: every
                     * qualifying member solved, as first built) */
                    static int d1_ = -1; if (d1_ < 0) { const char *e_ = getenv("BRISK_SXDEGONE"); d1_ = !(e_ && atoi(e_) == 0); }
                    int nel_ = 0, kel_ = -1, q0q_ = 0; S->ds_nel = 0;
                    /* (BRISK_SXDEGBOUND=1, with the second workspace) the ratio test's member counted first on the patterns, unbounded: the
                     * members are then counted against it */
                    static int dsb0_ = -1; if (dsb0_ < 0) { const char *e_ = getenv("BRISK_SXDEGBOUND"); dsb0_ = e_ && atoi(e_) == 1; }
                    static int dinc0_ = -1; if (dinc0_ < 0) { const char *e_ = getenv("BRISK_SXDEGINC"); dinc0_ = e_ ? atoi(e_) : 0; }
                    const int dsb_ = dsb0_ && g_dsws && de_ == 0;
                    long n0b_ = -1, incb_ = -1;
                    if (dsb_ && dinc0_ > 0) { n0b_ = ds_count_sym(S, q0_, 0x7fffffffffffffffL); incb_ = n0b_ - 1; }
                    for (int ps_ = d1_ ? 0 : 1; ps_ < 2; ps_++) {
                    if (ps_ == 1 && d1_ && nel_ < 2) { if (nel_ == 1) { bq_ = S->cand[kel_]; qpos_ = kel_; S->ds_memb++; } break; }
                    for (int k = g0_; k < g1_; k++) {
                        {   static double dg_ = -1; if (dg_ < 0) { const char *e_ = getenv("BRISK_SXDEGSPARSEG"); dg_ = e_ ? atof(e_) : 0; }   /* (the members considered: |alpha| at least this share of the group's largest, by default gamma) */
                            if (S->ca[k] < fmax(dg_, S->env_grpbf) * amx_) continue; }
                        /* (BRISK_SXDEGSPARSE=3: only members whose step, the leaving infeasibility over |alpha|, keeps them within their
                         * box: no new primal infeasibility; the largest |alpha| as before when none) */
                        if (ds_ >= 4 && S->cr[k] < 0) continue;   /* (a member of the wrong sign within the tolerance: not taken) */
                        /* (BRISK_SXDEGSPARSE=7: mode 4 with only the members whose dual step, reduced cost over |alpha|, is at least the step of
                         * the member the ratio test chose: the sparser member is taken only where the dual objective rises as much; =8: and
                         * their box takes the infeasibility left, as mode 6) */
                        if ((ds_ == 7 || ds_ == 8) && fmax(S->cr[k], 0) / S->ca[k] < sq0_ * (1 - 1e-9)) continue;
                        /* (BRISK_SXDEGSPARSE=9: mode 4 with only the members whose dual step is at most the chosen member's: the ratio test's pass
                         * over the group lets the member of the smallest ratio enter, so that no member's reduced cost takes the wrong sign; a
                         * member of a larger ratio leaves those before it of the wrong sign within the tolerance, which the iterations after must
                         * repair (supportcase12, LP: the rule's member had the larger step in 10,662 of 11,743 changed choices). Ties of the
                         * step, the zero steps above all, are what the rule was found on (Gurobi on satellites4-25, LP); =10: and their box takes the
                         * infeasibility left, as mode 6) */
                        if ((ds_ == 9 || ds_ == 10 || ds_ == 11) && fmax(S->cr[k], 0) / S->ca[k] > sq0_ * (1 + 1e-9) + 1e-300) continue;
                        /* (BRISK_SXDEGSPARSE=11: mode 10 with only the exact ties of the step: a member of a smaller step, a zero step where the
                         * chosen member's is positive (vouga, LP: 29 such choices), gives up the dual objective's rise; the rule then acts only
                         * where the ratio test cannot tell the members apart, the zero steps of a degenerate vertex above all) */
                        if (ds_ == 11 && fmax(S->cr[k], 0) / S->ca[k] < sq0_ * (1 - 1e-9)) continue;
                        /* (BRISK_SXDEGOVER=1, with modes 4, 6-8: only the members that leave their box by no more than the member the ratio test
                         * chose, the step sl / |alpha| less the box: where no member's box takes the infeasibility left, pythago7825 (LP), a
                         * member of smaller |alpha| enters further outside its box, a larger new primal infeasibility) */
                        if (dov_) { const int jj_ = S->cand[k]; const double rg_ = S->move[jj_] != 0 ? S->up[jj_] - S->lo[jj_] : SX_INF;
                            const double ov_ = rg_ < SX_INF ? fmax(0, sl_ / S->ca[k] - rg_) : 0; if (ov_ > ov0_ * (1 + 1e-9) + 1e-12) continue; }
                        /* (BRISK_SXDEGSPARSE=6: mode 4 with only the members whose box takes the infeasibility left after the earlier groups'
                         * flips, as the pass over the final group asks of the member it lets enter: no member becomes basic outside its box.
                         * Mode 4 on pythago7825 (LP): in 17,622 of its 19,109 changed choices no member's box could take the rest, the ratio
                         * test's member (the largest |alpha|) left its box least, and the sparsest (|alpha| 0.46 of the largest on average)
                         * entered further outside it, a larger new primal infeasibility each time: 25,038 -> 91,231 iterations at the 60 s
                         * limit. With this condition the rule leaves such groups to the ratio test: 28,474 iterations, 15.0 s (13.4 s)) */
                        if (ds_ == 6 || ds_ == 8 || ds_ == 10 || ds_ == 11) { const int jj_ = S->cand[k]; if (S->move[jj_] != 0 && S->up[jj_] - S->lo[jj_] < SX_INF && S->ca[k] * (S->up[jj_] - S->lo[jj_]) < sl_ * (1 - 1e-9)) continue; }
                        if (ds_ == 3) { const int jj_ = S->cand[k]; const double rg_ = S->up[jj_] - S->lo[jj_];
                            if (rg_ < SX_INF && fabs(delta) / S->ca[k] > rg_ * (1 + 1e-9)) continue; }
                        if (ps_ == (d1_ ? 0 : 1) && S->cand[k] == q0_) q0q_ = 1;   /* (the ratio test's member qualifies itself) */
                        if (ps_ == (d1_ ? 0 : 1)) { if (S->ds_nel >= S->ds_elcap) { S->ds_elcap = S->ds_elcap ? 2 * S->ds_elcap : 256; S->ds_el = realloc(S->ds_el, sizeof(int) * (size_t)S->ds_elcap); } S->ds_el[S->ds_nel++] = k; }   /* (the qualifying members, for the check) */
                        if (ps_ == 0) { nel_++; kel_ = k; continue; }
                        const int jq_ = S->cand[k]; long nz_ = 0;
                        S->ds_memb++;
                        if (de_ > 0 && ds_ >= 2) {   /* (the kept count, or a fresh one; the K best kept in tk_ by count, then larger |alpha|) */
                            if (S->dsst[jq_] >= 0) { nz_ = S->dsnz[jq_]; S->ds_hit++; } else { nz_ = ds_count(S, jq_); S->dsnz[jq_] = (int)nz_; S->dsst[jq_] = S->iter; }
                            int a_ = nt_ < de_ ? nt_++ : de_;
                            if (a_ == de_) { if (nz_ > te_[de_ - 1] || (nz_ == te_[de_ - 1] && S->ca[k] <= S->ca[tk_[de_ - 1]])) continue; a_ = de_ - 1; }
                            while (a_ > 0 && (nz_ < te_[a_ - 1] || (nz_ == te_[a_ - 1] && S->ca[k] > S->ca[tk_[a_ - 1]]))) { tk_[a_] = tk_[a_ - 1]; te_[a_] = te_[a_ - 1]; a_--; }
                            tk_[a_] = k; te_[a_] = nz_;
                            continue;
                        }
                        if (dsb_) {   /* (BRISK_SXDEGBOUND=1: the count on the patterns, bounded by the best so far and, with BRISK_SXDEGINC,
                                       * by one less than the ratio test's member's count) */
                            long bd_ = incb_ >= 0 ? incb_ : 0x7fffffffffffffffL; if (bestnz_ >= 0 && bestnz_ < bd_) bd_ = bestnz_;
                            nsolve_++; nz_ = ds_count_sym(S, jq_, bd_);
                        }
                        else if (ds_ >= 2 && S->dsst[jq_] == S->nfactor) { nz_ = S->dsnz[jq_]; S->ds_hit++; }
                        else if (dk_ > 0 && nsolve_ >= dk_) continue;
                        else { nsolve_++; S->ds_solve++;
                            sxv_clear(v_); col_scatter(S, jq_, v_, 1.0);
                            /* (13 November 2026) in the second solve's workspace: the factor's own statistics of its hypersparse solves (which
                             * choose how the iteration's own solves run) are left as they are, so the rule changes the path only by its choice */
                            {   const double hc_ = S->F.hy_canc; const int hp_ = S->F.hy_cp;
                                if (g_dsws) { SxWs *W_ = &S->ws2; W_->nflops = W_->nedges = 0; sxlu_ftran_ws(&S->F, W_, v_, S->dens_col, NULL); S->F.hy_canc = hc_; S->F.hy_cp = hp_; } else sxlu_ftran(&S->F, v_, S->dens_col, NULL); }
                            if (v_->cnt >= 0) nz_ = v_->cnt; else for (int i = 0; i < S->m; i++) nz_ += v_->val[i] != 0;
                            sxv_clear(v_);
                            if (ds_ >= 2) { S->dsnz[jq_] = (int)nz_; S->dsst[jq_] = S->nfactor; }
                        }
                        if (bestnz_ < 0 || nz_ < bestnz_ || (nz_ == bestnz_ && S->ca[k] > S->ca[qpos_])) { bestnz_ = nz_; bq_ = S->cand[k]; qpos_ = k; }
                    }
                    }
                    for (int t_ = 0; t_ < nt_; t_++) {   /* (BRISK_SXDEGEST: the K best by kept counts solved afresh) */
                        const int k = tk_[t_], jq_ = S->cand[k]; long nz_;
                        if (S->dsst[jq_] == S->iter) nz_ = S->dsnz[jq_]; else { nz_ = ds_count(S, jq_); S->dsnz[jq_] = (int)nz_; S->dsst[jq_] = S->iter; }
                        if (bestnz_ < 0 || nz_ < bestnz_ || (nz_ == bestnz_ && S->ca[k] > S->ca[qpos_])) { bestnz_ = nz_; bq_ = jq_; qpos_ = k; }
                    }
                    {   /* (13 November 2026, BRISK_SXDEGINC=1) the ratio test's member is the incumbent: another member enters only when its
                         * B^-1 a_j is strictly sparser than the incumbent's (both counted afresh). Without it (agent X's diagnostic,
                         * BRISK_SXDEGDIAG) a changed choice was no sparser in 2,737 of 8,977 changes on supportcase12 and 248 of 447 on ex9
                         * (LPs): the best two by kept counts can be no better than the incumbent, which was not counted, and a member that
                         * qualifies alone entered uncounted; such a change only takes a larger dual step, leaving the members of smaller
                         * ratio with reduced costs of the wrong sign (within the tolerance) for no gain in sparsity */
                        static int di_ = -1; if (di_ < 0) { const char *e_ = getenv("BRISK_SXDEGINC"); di_ = e_ ? atoi(e_) : 0; }
                        /* (BRISK_SXDEGINC=2: only where the incumbent qualifies itself (its box takes the infeasibility left, a reduced cost of the
                         * right sign); where it does not, it would become basic outside its box, and the member that qualifies enters as in mode 6) */
                        if (di_ > 0 && bq_ != q0_ && (di_ != 2 || q0q_)) {
                            long n0_; if (n0b_ >= 0) n0_ = n0b_; else if (de_ > 0 && ds_ >= 2 && S->dsst[q0_] == S->iter) n0_ = S->dsnz[q0_]; else { n0_ = ds_count(S, q0_); if (de_ > 0 && ds_ >= 2) { S->dsnz[q0_] = (int)n0_; S->dsst[q0_] = S->iter; } }
                            long n1_ = bestnz_; if (n1_ < 0 && n0b_ >= 0) n1_ = ds_count_sym(S, bq_, incb_); if (n1_ < 0) { if (de_ > 0 && ds_ >= 2 && S->dsst[bq_] == S->iter) n1_ = S->dsnz[bq_]; else { n1_ = ds_count(S, bq_); if (de_ > 0 && ds_ >= 2) { S->dsnz[bq_] = (int)n1_; S->dsst[bq_] = S->iter; } } bestnz_ = n1_; }
                            if (!(n1_ < n0_)) { bq_ = q0_; bestnz_ = n0_; S->ds_inc++; }
                        }
                    }
                    {   /* (BRISK_SXDEGSHIFT=1) the shift the member needs to take the ratio test's dual step, td_q - s0 |alpha_q|, kept within
                         * the dual tolerance: a larger one would change the problem's costs, not only the choice among ties (where the ratio
                         * test's member has a reduced cost of the wrong sign, its step zero, or the tolerance was tightened in the rounds after
                         * the solve); the ratio test's member then enters */
                        static int dsh2_ = -1; if (dsh2_ < 0) { const char *e_ = getenv("BRISK_SXDEGSHIFT"); dsh2_ = e_ ? atoi(e_) : 0; }
                        if (dsh2_ > 0 && bq_ != q0_) { int ka_ = -1, kb_ = -1; for (int k = g0_; k < g1_; k++) { if (S->cand[k] == bq_) ka_ = k; if (S->cand[k] == q0_) kb_ = k; }
                            if (ka_ < 0 || kb_ < 0 || fabs(S->cr[ka_] - fmax(S->cr[kb_], 0) / S->ca[kb_] * S->ca[ka_]) > S->told) { bq_ = q0_; S->ds_shref++; } }
                    }
                    q = bq_;
                    if (q != q0_) {   /* (BRISK_SXDEGSHIFT=1: the dual step of the ratio test's member, sb; see the step below) */
                        int kb_ = -1; for (int k = g0_; k < g1_; k++) if (S->cand[k] == q0_) { kb_ = k; break; }
                        if (kb_ >= 0) { S->ds_tq = q; S->ds_tst = fmax(S->cr[kb_], 0) / S->ca[kb_]; } }
                    {   /* (13 November 2026, a diagnostic: BRISK_SXDEGDIAG=1) each changed choice on stderr: the iteration, the group's size,
                         * both members' counts of B^-1 a_j, |alpha| over the largest, reduced cost over the tolerance, the dual steps, and the
                         * reduced costs of the wrong sign each step leaves on the group's other members (how many, the largest and the sum,
                         * over the tolerance) */
                        static int dg2_ = -1; if (dg2_ < 0) { const char *e_ = getenv("BRISK_SXDEGDIAG"); dg2_ = e_ && atoi(e_) > 0; }
                        if (dg2_ && q != q0_) { int ka_ = -1, kb_ = -1; for (int k = g0_; k < g1_; k++) { if (S->cand[k] == q) ka_ = k; if (S->cand[k] == q0_) kb_ = k; }
                            if (ka_ >= 0 && kb_ >= 0) { const double sa_ = fmax(S->cr[ka_], 0) / S->ca[ka_], sb_ = fmax(S->cr[kb_], 0) / S->ca[kb_];
                                int na_ = 0, nb_ = 0; double ma_ = 0, mb_ = 0, ua_ = 0, ub_ = 0;
                                for (int k = g0_; k < g1_; k++) { const double va_ = sa_ * S->ca[k] - S->cr[k], vb_ = sb_ * S->ca[k] - S->cr[k];
                                    if (k != ka_ && va_ > 0) { na_++; ma_ = fmax(ma_, va_); ua_ += va_; } if (k != kb_ && vb_ > 0) { nb_++; mb_ = fmax(mb_, vb_); ub_ += vb_; } }
                                const long nb0_ = ds_count(S, q0_);
                                const long na0_ = bestnz_ >= 0 ? bestnz_ : ds_count(S, q);
                                fprintf(stderr, "[degdiag] %ld g %d nz %ld %ld ar %.3g %.3g cr %.3g %.3g st %.4g %.4g wa %d %.3g %.3g wb %d %.3g %.3g sl %.3g dcol %.4f pert %d nel %d nupd %d\n", S->iter, g1_ - g0_, na0_, nb0_,
                                        S->ca[ka_] / amx_, S->ca[kb_] / amx_, S->cr[ka_] / S->told, S->cr[kb_] / S->told, sa_, sb_, na_, ma_ / S->told, ua_ / S->told, nb_, mb_ / S->told, ub_ / S->told, sl_, S->dens_col, S->perturbed, nel_, S->F.nupd); } } }
                    if (q != q0_) { S->ds_chg++;
                        for (int t_ = 0; t_ < 2; t_++) { const int jj_ = t_ ? q0_ : q; int kk_ = -1; for (int k = g0_; k < g1_; k++) if (S->cand[k] == jj_) kk_ = k;
                            if (kk_ >= 0 && S->move[jj_] != 0 && S->up[jj_] - S->lo[jj_] < SX_INF && S->ca[kk_] * (S->up[jj_] - S->lo[jj_]) < sl_ * (1 - 1e-9)) { if (t_) S->ds_oob0++; else S->ds_oob++; }
                            if (!t_ && kk_ >= 0) { S->ds_ar += S->ca[kk_] / amx_; if (S->cr[kk_] == 0) S->ds_cr0++; } }
                        {   int ka_ = -1, kb_ = -1; for (int k = g0_; k < g1_; k++) { if (S->cand[k] == q) ka_ = k; if (S->cand[k] == q0_) kb_ = k; }
                            if (ka_ >= 0 && kb_ >= 0) { const double sa_ = fmax(S->cr[ka_], 0) / S->ca[ka_], sb_ = fmax(S->cr[kb_], 0) / S->ca[kb_];
                                if (sa_ < sb_ * (1 - 1e-9)) { S->ds_stsm++; S->ds_strat += sb_ > 0 ? sa_ / sb_ : 1; } else if (sa_ > sb_ * (1 + 1e-9)) S->ds_stlg++; } } }
                    if (S->tdiag || dst_) S->ds_t += now() - dt0_;
                    (void)ns0_;
                    {   /* (13 November 2026, a diagnostic: BRISK_SXDEGCHECK=1) every eligible member solved afresh (its exact |B^-1 a_j|), and the
                         * entering member that each estimate would choose compared with the sparsest: 0 the rule as it runs (counts kept until
                         * the next factorization), 1 counts kept for good (only members never counted solved), 2 the same counts ranking and
                         * the best two by them solved afresh, 3 only members of |alpha| at least half the largest, 4 the column's own length,
                         * 5 counts kept 100 iterations, 6 the last member of the candidate list; their agreement printed at -v 2 */
                        static int dch_ = -1; if (dch_ < 0) { const char *e_ = getenv("BRISK_SXDEGCHECK"); dch_ = e_ && atoi(e_) > 0; }
                        if (dch_ && ds_ >= 4 && ds_ != 5) {
                            for (int t_ = 0; t_ < 3; t_++) if (!S->dc_nz[t_]) { S->dc_nz[t_] = xmalloc(sizeof(int) * ((size_t)S->N + 1)); S->dc_it[t_] = xmalloc(sizeof(long) * ((size_t)S->N + 1)); for (int j = 0; j <= S->N; j++) S->dc_it[t_][j] = -1; }
                            const int ng_ = g1_ - g0_; long *ex_ = xmalloc(sizeof(long) * (size_t)ng_);
                            for (int k = g0_; k < g1_; k++) ex_[k - g0_] = -1;
                            int nel_ = 0;
                            for (int t_ = 0; t_ < S->ds_nel; t_++) { const int k = S->ds_el[t_];
                                const int jq_ = S->cand[k]; long nz_ = 0; nel_++;
                                sxv_clear(v_); col_scatter(S, jq_, v_, 1.0); { SxWs *W_ = &S->ws2; const double hc_ = S->F.hy_canc; const int hp_ = S->F.hy_cp; sxlu_ftran_ws(&S->F, W_, v_, S->dens_col, NULL); S->F.hy_canc = hc_; S->F.hy_cp = hp_; }
                                if (v_->cnt >= 0) nz_ = v_->cnt; else for (int i = 0; i < S->m; i++) nz_ += v_->val[i] != 0;
                                sxv_clear(v_); ex_[k - g0_] = nz_; }
                            if (nel_ >= 2) {
                                long mn_ = -1; for (int k = 0; k < ng_; k++) if (ex_[k] >= 0 && (mn_ < 0 || ex_[k] < mn_)) mn_ = ex_[k];
                                int pk_[8]; for (int e = 0; e < 8; e++) pk_[e] = -1;
                                #define DC_BETTER(a_, b_, ka_, kb_) ((b_) < 0 || (a_) < (b_) || ((a_) == (b_) && S->ca[ka_] > S->ca[kb_]))
                                {   long b_ = -1; for (int k = g0_; k < g1_; k++) if (S->cand[k] == q) pk_[0] = k; (void)b_; }
                                for (int e = 1; e <= 5; e++) { if (e == 2 || e == 3 || e == 4) continue;
                                    const int c_ = e == 1 ? 0 : 2; long bb_ = -1; int bk_ = -1;
                                    for (int k = g0_; k < g1_; k++) { if (ex_[k - g0_] < 0) continue; const int j = S->cand[k];
                                        long es_; if (S->dc_it[c_][j] >= 0 && (e == 1 || S->iter - S->dc_it[c_][j] < 100)) es_ = S->dc_nz[c_][j];
                                        else { es_ = ex_[k - g0_]; S->dc_nz[c_][j] = (int)es_; S->dc_it[c_][j] = S->iter; S->dc_sol[e]++; }
                                        if (bk_ < 0 || DC_BETTER(es_, bb_, k, bk_)) { bb_ = es_; bk_ = k; } }
                                    pk_[e] = bk_; }
                                {   int k1_ = -1, k2_ = -1; long e1_ = -1, e2_ = -1;   /* (2: the best two by kept counts, solved afresh) */
                                    for (int k = g0_; k < g1_; k++) { if (ex_[k - g0_] < 0) continue; const int j = S->cand[k]; long es_;
                                        if (S->dc_it[1][j] >= 0) es_ = S->dc_nz[1][j]; else { es_ = ex_[k - g0_]; S->dc_nz[1][j] = (int)es_; S->dc_it[1][j] = S->iter; S->dc_sol[2]++; }
                                        if (k1_ < 0 || DC_BETTER(es_, e1_, k, k1_)) { k2_ = k1_; e2_ = e1_; k1_ = k; e1_ = es_; } else if (k2_ < 0 || DC_BETTER(es_, e2_, k, k2_)) { k2_ = k; e2_ = es_; } }
                                    int bk_ = -1; long bb_ = -1;
                                    for (int t_ = 0; t_ < 2; t_++) { const int k = t_ ? k2_ : k1_; if (k < 0) continue; const int j = S->cand[k];
                                        if (S->dc_it[1][j] != S->iter) S->dc_sol[2]++; S->dc_nz[1][j] = (int)ex_[k - g0_]; S->dc_it[1][j] = S->iter;
                                        if (bk_ < 0 || DC_BETTER(ex_[k - g0_], bb_, k, bk_)) { bb_ = ex_[k - g0_]; bk_ = k; } }
                                    pk_[2] = bk_; }
                                {   long bb_ = -1; int bk_ = -1; for (int k = g0_; k < g1_; k++) { if (ex_[k - g0_] < 0 || S->ca[k] < 0.5 * amx_) continue; S->dc_sol[3]++;
                                        if (bk_ < 0 || DC_BETTER(ex_[k - g0_], bb_, k, bk_)) { bb_ = ex_[k - g0_]; bk_ = k; } } pk_[3] = bk_; }
                                {   long bb_ = -1; int bk_ = -1; for (int k = g0_; k < g1_; k++) { if (ex_[k - g0_] < 0) continue; const int j = S->cand[k]; const long ln_ = j < S->n ? S->Ap[j + 1] - S->Ap[j] : 1;
                                        if (bk_ < 0 || DC_BETTER(ln_, bb_, k, bk_)) { bb_ = ln_; bk_ = k; } } pk_[4] = bk_; }
                                for (int k = g1_ - 1; k >= g0_; k--) if (ex_[k - g0_] >= 0) { pk_[6] = k; break; }
                                #undef DC_BETTER
                                S->dc_n++;
                                for (int e = 0; e < 7; e++) { if (pk_[e] < 0) continue; const long xe_ = ex_[pk_[e] - g0_]; if (xe_ < 0) continue;
                                    if (xe_ == mn_) S->dc_ag[e]++; if (xe_ <= 1.25 * mn_ + 2) S->dc_near[e]++; S->dc_lr[e] += log((xe_ + 1.0) / (mn_ + 1.0)); }
                            }
                            free(ex_);
                        }
                    }
                }
            } }
        {   /* (13 November 2026, a diagnostic: BRISK_SXFORCEENT=<file> of "iteration variable": another solver's entering variable at
             * that iteration, printed beside ours: its pivot entry, reduced cost, ratio, bounds and position, as ours) */
            static int nfe_ = -1; static long *eit_ = NULL; static int *evar_ = NULL;
            if (nfe_ < 0) { nfe_ = 0; const char *e_ = getenv("BRISK_SXFORCEENT"); FILE *ff_ = e_ ? fopen(e_, "r") : NULL;
                if (ff_) { long a_; int b_; int cap_ = 0; while (fscanf(ff_, "%ld %d", &a_, &b_) == 2) { if (nfe_ >= cap_) { cap_ = cap_ ? 2 * cap_ : 256; eit_ = realloc(eit_, sizeof(long) * cap_); evar_ = realloc(evar_, sizeof(int) * cap_); } eit_[nfe_] = a_; evar_[nfe_] = b_; nfe_++; } fclose(ff_); } }
            for (int t_ = 0; t_ < nfe_; t_++) if (eit_[t_] == S->iter) {
                const int g_ = evar_[t_];
                for (int u_ = 0; u_ < 2; u_++) { const int v_ = u_ ? g_ : q; if (v_ < 0 || v_ >= S->N) continue;
                    const double a_ = v_ < S->n ? S->row.val[v_] : S->rho.val[v_ - S->n] * -1.0;
                    fprintf(stderr, "   [ent %ld %s var %d (%s, %d entries): alpha %.4g, d %.4g, ratio %.4g, bounds [%g, %g], move %d, basic %d]\n", S->iter, u_ ? "other" : "ours ", v_,
                            v_ < S->n ? "column" : "logical", v_ < S->n ? S->Ap[v_ + 1] - S->Ap[v_] : 1, a_, S->d[v_], a_ != 0 ? fabs(S->d[v_] / a_) : -1.0, S->lo[v_], S->up[v_], S->move[v_], S->pos[v_] >= 0); }
                fprintf(stderr, "   [ent %ld: leaving infeasibility %.4g, flips %d, final group %d]\n", S->iter, delta, nflip, S->fg_n);
                break; }
        }
        {   /* (13 November 2026, a diagnostic: BRISK_SXFORCEPATH=<file> of lines "iteration leaving entering flip flip ...", gpath.py's
             * lines) at those iterations the entering variable and the bound flips forced to the file's (with BRISK_SXFORCEVARS for the
             * leaving one): our state follows another solver's path, so that its choices can be compared with ours at its own states */
            static int nfp_ = -1; static long *fpi_ = NULL; static int *fpq_ = NULL, *fpo_ = NULL, *fpf_ = NULL;
            if (nfp_ < 0) { nfp_ = 0; const char *e_ = getenv("BRISK_SXFORCEPATH"); FILE *ff_ = e_ ? fopen(e_, "r") : NULL;
                if (ff_) { char ln_[1 << 16]; int cap_ = 0, capf_ = 0, nf_ = 0;
                    while (fgets(ln_, sizeof ln_, ff_)) { char *sp_ = ln_; long k_ = strtol(sp_, &sp_, 10); char *tk_ = strtok(sp_, " \n"); if (!tk_ || tk_[0] == '[') continue;
                        tk_ = strtok(NULL, " \n"); if (!tk_ || tk_[0] == '[') continue; const int q_ = atoi(tk_);
                        if (nfp_ + 1 >= cap_) { cap_ = cap_ ? 2 * cap_ : 1024; fpi_ = realloc(fpi_, sizeof(long) * cap_); fpq_ = realloc(fpq_, sizeof(int) * cap_); fpo_ = realloc(fpo_, sizeof(int) * (cap_ + 1)); }
                        fpi_[nfp_] = k_; fpq_[nfp_] = q_; fpo_[nfp_] = nf_;
                        while ((tk_ = strtok(NULL, " \n"))) { if (nf_ >= capf_) { capf_ = capf_ ? 2 * capf_ : 4096; fpf_ = realloc(fpf_, sizeof(int) * capf_); } fpf_[nf_++] = atoi(tk_); }
                        nfp_++; fpo_[nfp_] = nf_; }
                    fclose(ff_); } }
            if (nfp_ > 0) { int f_ = -1; for (int t_ = 0; t_ < nfp_; t_++) if (fpi_[t_] == S->iter) { f_ = t_; break; }
                if (f_ >= 0) { const int gq_ = fpq_[f_];
                    if (gq_ >= 0 && gq_ < S->N && S->pos[gq_] < 0 && fabs(S->row.val[gq_]) > 1e-9) {
                        int ok_ = 1; for (int u_ = fpo_[f_]; u_ < fpo_[f_ + 1]; u_++) { const int j_ = fpf_[u_]; if (j_ < 0 || j_ >= S->N || S->pos[j_] >= 0 || S->move[j_] == 0 || !is_boxed(S, j_)) ok_ = 0; }
                        if (ok_) { q = gq_; nflip = 0; for (int u_ = fpo_[f_]; u_ < fpo_[f_ + 1]; u_++) S->cand[nflip++] = fpf_[u_]; g_forcedit = S->iter; }
                        else fprintf(stderr, "   [forcepath %ld: a flip not possible]\n", S->iter);
                    } else fprintf(stderr, "   [forcepath %ld: entering %d not possible (basic %d, alpha %g)]\n", S->iter, gq_, gq_ >= 0 && gq_ < S->N ? S->pos[gq_] >= 0 : -1, gq_ >= 0 && gq_ < S->N ? S->row.val[gq_] : 0.0);
                } }
        }
        if (S->isint) { if (q < S->n && S->isint[q]) S->n_ent_int++; else S->n_ent_oth++; }
        if (g_entdbg) { const int lq = q < S->n ? S->Ap[q + 1] - S->Ap[q] : 1, jr = S->basic[r], lr = jr < S->n ? S->Ap[jr + 1] - S->Ap[jr] : 1;   /* (a diagnostic) */
            if (lq >= 40) { g_ent_long++; if (fabs(S->d[q]) <= S->told) g_ent_long_d0++; } if (lr >= 40) g_leave_long++; g_ent_all++; }
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
        } else { sxv_copy(tau, rho);
            if (g_dserows > 0 && S->rscale) for (int k = 0; k < tau->cnt; k++) { const int i = tau->idx[k]; tau->val[i] *= S->rscale[i] * S->rscale[i]; }
            if (g_dsefull > 0) { for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; if (j < S->n && S->row.val[j] != 0) col_scatter(S, j, tau, S->row.val[j]); }
                if (S->basic[r] < S->n) col_scatter(S, S->basic[r], tau, 1.0); }
            ftran2(S, col, tau); }
        const double acol = col->val[r];
        {   static long itdbg_ = -1; if (itdbg_ < 0) { const char *e_ = getenv("BRISK_SXITERDBG"); itdbg_ = e_ ? atol(e_) : 0; }   /* (12 November 2026, a diagnostic: the first k iterations, the entering and leaving columns) */
            if (itdbg_ > 0 && S->iter < itdbg_) { const int jl_ = S->basic[r];
                fprintf(stderr, "[it] %ld enter %d (%d entries) leave %d (%d entries) |alpha_q| %d |rho| %d\n", S->iter, q, q < S->n ? S->Ap[q + 1] - S->Ap[q] : 1,
                        jl_, jl_ < S->n ? S->Ap[jl_ + 1] - S->Ap[jl_] : 1, col->cnt >= 0 ? col->cnt : S->m, S->row.cnt); } }
        S->sum_col += col->cnt >= 0 ? col->cnt : m;
        if (fabs(acol - arow) > 1e-7 * (1 + fabs(acol)) || fabs(acol) < 1e-9) {
            {   static int dd_ = -1; if (dd_ < 0) { const char *e_ = getenv("BRISK_SXDISAGREEDBG"); dd_ = e_ && atoi(e_) == 1; }   /* (13 November 2026, a diagnostic) */
                if (dd_) { int pos_ = -1; for (int k_ = 0; k_ < S->ncand; k_++) if (S->cand[k_] == q) { pos_ = k_; break; } fprintf(stderr, "[disagree] it %ld r %d q %d acol %.6g arow %.6g nupd %d nflip %d ncand %d q at %d ca %.4g; flipped 0..nflip-1 includes q: %d\n", S->iter, r, q, acol, arow, S->F.nupd, nflip, S->ncand, pos_, pos_ >= 0 ? S->ca[pos_] : -1.0, pos_ >= 0 && pos_ < nflip); } }
            /* the row and the column disagree: refactorize and try again */
            if (S->F.nupd > 0) { need_rebuild = 1; continue; }
            if (fabs(acol) < 1e-9) { S->w[r] *= 10; S->heap_ok = 0; continue; }
        }
        /* the dual step: d_q / alpha_q when the entering reduced cost has the sign of its move; a step of zero when it
         * is zero or of the wrong sign within the tolerance (its cost shifted to make it zero; no flips then) */
        const int mvq = S->move[q] ? S->move[q] : (arow * (delta < 0 ? -1.0 : 1.0) > 0 ? 1 : -1);
        double thd = 0;
        /* (13 November 2026, the LP developer; BRISK_SXLAZYSTALL=1, off by default) a solve whose costs are perturbed at the first
         * stall (lazy_pert: BRISK_SXLAZYPERT=1, and the re-solves of sxm_solve) relies on this stall rule for that perturbation, but
         * the rule is gated by its own trial switch (BRISK_SXSTALL, off): without it the costs are never perturbed at all, and the
         * runs said to perturb at the first stall ran unperturbed to the end (mzzv11, LP: the objective at -26,078.000 for 18,000
         * iterations, 45,954 iterations against the default's 9,382). With the switch the rule arms the first perturbation of a
         * solve with lazy_pert whatever BRISK_SXSTALL says; its escalations after it stay with BRISK_SXSTALL (=2: they too) */
        const int lzst_ = lazystall();
        const int stall_on_ = !S->env_nostall || (lzst_ && S->lazy_pert && (!S->perturbed || lzst_ >= 2));
        if ((S->perturbed || S->lazy_pert) && S->phase == 2 && stall_on_) {
            /* (24 October 2026, the night) a step is degenerate when the entering reduced cost is within twice the
             * perturbation that column received (it moves only by the perturbation), not only when it is zero: mzzv11 (LP)
             * moved its objective by 5e-4 in 6,000 iterations of such steps, none of zero length, and the rule never fired */
            double zq = S->told;
            if (S->perturbed && q < S->n) { double sc = 1; for (int k = 0; k < S->pert_level; k++) sc *= 10; const double pq = 2.0 * (fabs(S->cost0[q]) * S->costscale + 1) * S->pert_base * sc; if (pq > zq) zq = pq; }
            S->zw_it++; if (!(mvq * S->d[q] > zq)) S->zw_zero++;
            static long swin_ = -1; if (swin_ < 0) { const char *e_ = getenv("BRISK_SXSTALLWIN"); swin_ = e_ ? atol(e_) : SX_STALL_WIN; if (swin_ < 10) swin_ = SX_STALL_WIN; }   /* (13 November 2026, an experiment: the stall rule's window) */
            if (S->zw_it >= swin_) {
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
        {   /* (13 November 2026, BRISK_SXDEGSHIFT=1, agent X) the sparsest-member rule's member enters with the dual step of the member
             * the ratio test chose: its cost is shifted so that its reduced cost over |alpha| is that step (the shift is removed when it
             * leaves the basis, shift_back, or at the end). The other members' reduced costs then change exactly as under the ratio
             * test's choice: a member of larger ratio no longer leaves those of smaller ratio with reduced costs of the wrong sign (the
             * losses on supportcase12, nb10tb, ex9 and neos-5051588-culgoa, LPs: the rule's member had the larger step in 85-98 % of the
             * changed choices). Only the entering column changes: the sparsity is kept, the dual path is the ratio test's. Within the
             * Harris bound the shift is at most the tolerance times |alpha_q| / |alpha_q0| */
            static int dsh_ = -1; if (dsh_ < 0) { const char *e_ = getenv("BRISK_SXDEGSHIFT"); dsh_ = e_ ? atoi(e_) : 0; }
            if (dsh_ > 0 && S->ds_tq == q && S->move[q] != 0) {
                const double tg_ = mvq * S->ds_tst * fabs(arow), df_ = S->d[q] - tg_;
                if (df_ != 0) { S->cost[q] -= df_; S->shift[q] -= df_; S->d[q] = tg_; S->shifted = 1; S->ds_nsh++; if (fabs(df_) > S->ds_shmax) S->ds_shmax = fabs(df_); }
            }
        }
        if (mvq * S->d[q] > 0) thd = S->d[q] / acol;
        else { shift_cost(S, q); nflip = 0; }
        /* flips: the candidates the ratio test passed, and the boxed variables of the row whose reduced cost has the
         * wrong sign beyond the tolerance after the step (entries below the pivot tolerance, which the ratio test does
         * not consider); the others with such a reduced cost get their cost shifted. One solve for all the flips. */
        int nfl = 0; S->trace_nf = 0;
        sxv_clear(bf);
        for (int k = 0; k < nflip; k++) {
            const int j = S->cand[k];
            if (S->trace && S->trace_nf < 512) S->trace_fl[S->trace_nf++] = j;
            if (S->move[j] > 0) { col_scatter(S, j, bf, S->up[j] - S->lo[j]); S->x[j] = S->up[j]; S->move[j] = -1; }
            else { col_scatter(S, j, bf, S->lo[j] - S->up[j]); S->x[j] = S->lo[j]; S->move[j] = 1; }
            nfl++;
        }
        /* the dual update, and in the same pass the entries whose reduced cost it leaves of the wrong sign */
        if (S->tdiag) tdg = now();
        int nbad = 0;
        if (thd != 0) {
            const double mtd = -S->told;
            int k0 = 0;
#if defined(__AVX512F__) && defined(__AVX512VL__)
            /* (26 October 2026) a row in increasing order and longer than a sixth of the columns (the ratio test's dense pass):
             * the dense arrays in memory order, the entries off the row skipped; the same values, the same list in the same
             * order */
            static int vecu = -1; if (vecu < 0) { const char *e = getenv("BRISK_SXVECCHUZC"); vecu = e ? atoi(e) : 2; }
            if (vecu >= 2 && S->row_ord && 6L * S->row.cnt > S->N) {
                const int N = S->N; const double *rv = S->row.val;
                const __m512d vth = _mm512_set1_pd(thd), vmt = _mm512_set1_pd(mtd), zero = _mm512_setzero_pd();
                const __m256i e8 = _mm256_set1_epi32(8);
                __m256i jj = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                int j0 = 0;
                for (; j0 + 8 <= N; j0 += 8, jj = _mm256_add_epi32(jj, e8)) {
                    const __m512d al = _mm512_loadu_pd(rv + j0);
                    const __mmask8 nz = _mm512_cmp_pd_mask(al, zero, _CMP_NEQ_UQ);
                    if (!nz) continue;
                    const __m512d dj = _mm512_sub_pd(_mm512_maskz_loadu_pd(nz, S->d + j0), _mm512_mul_pd(vth, al));
                    _mm512_mask_storeu_pd(S->d + j0, nz, dj);
                    const __m512d mvd = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(S->move + j0))));
                    const __mmask8 bad = _mm512_mask_cmp_pd_mask(nz, _mm512_mul_pd(mvd, dj), vmt, _CMP_LT_OQ);
                    if (bad) { _mm256_storeu_si256((__m256i *)(S->cand + nbad), _mm256_maskz_compress_epi32(bad, jj)); nbad += __builtin_popcount((unsigned)bad); }
                }
                for (int j = j0; j < N; j++) {
                    if (rv[j] == 0) continue;
                    const double dj = S->d[j] - thd * rv[j];
                    S->d[j] = dj;
                    if (S->move[j] * dj < mtd) S->cand[nbad++] = j;
                }
                k0 = S->row.cnt;
            }
#endif
            for (int k = k0; k < S->row.cnt; k++) {
                const int j = S->row.idx[k];
                const double dj = S->d[j] - thd * S->row.val[j];
                S->d[j] = dj;
                if (S->move[j] * dj < mtd) S->cand[nbad++] = j;     /* (cand's flips were taken above) */
            }
        }
        S->d[q] = 0;
        {   static int fps_ = -1; if (fps_ < 0) { const char *e_ = getenv("BRISK_SXFORCEPATHSHIFT"); fps_ = e_ && atoi(e_) != 0; }
            /* (a forced iteration: no flips but the other solver's; BRISK_SXFORCEPATHSHIFT=1: the reduced costs left of the wrong sign
             * beyond the tolerance shifted to zero, else left as they are) */
            if (g_forcedit == S->iter && fps_) for (int t = 0; t < nbad; t++) { const int j = S->cand[t];
                if (j != q && S->move[j] != 0 && S->move[j] * S->d[j] < -S->told) { shift_cost(S, j); S->nsh_it++; } } }
        for (int t = 0; t < nbad && g_forcedit != S->iter; t++) {   /* (a forced iteration: see above) */
            const int j = S->cand[t];
            if (j == q || S->move[j] == 0 || S->move[j] * S->d[j] >= -S->told) continue;
            /* (10 November 2026, ISSUES 862; on by default since 11 November, BRISK_SXSMALLSHIFT=0: off) an entry below the ratio test's pivot tolerance is
             * zero for the ratio test, so it is zero here too: its reduced cost of the wrong sign is the step's rounding
             * times it, and the cost is shifted (removed at the end) instead of the column flipped across its box, which
             * moves the basic values by its range times its column (neos-5041822-cockle, LP: entries 2e-8, ranges 5e7, a
             * cycle of four iterations) */
            static int smsh = -1; if (smsh < 0) { const char *e_ = getenv("BRISK_SXSMALLSHIFT"); smsh = e_ ? atoi(e_) : 1; }
            /* (13 November 2026, an experiment: BRISK_SXNOPOSTFLIP=1) such a reduced cost is shifted to zero, never flipped: Gurobi on
             * satellites4-25's copy with perturbed costs (LP) flips no variable outside the ratio test's own flips (1,000 iterations) */
            static int npf_ = -1; if (npf_ < 0) { const char *e_ = getenv("BRISK_SXNOPOSTFLIP"); npf_ = e_ && atoi(e_) != 0; }
            if (!npf_ && is_boxed(S, j) && !(smsh && fabs(S->row.val[j]) < (S->cpiv > 0 ? S->cpiv : 1e-7))) {
                if (S->move[j] > 0) { col_scatter(S, j, bf, S->up[j] - S->lo[j]); S->x[j] = S->up[j]; S->move[j] = -1; }
                else { col_scatter(S, j, bf, S->lo[j] - S->up[j]); S->x[j] = S->lo[j]; S->move[j] = 1; }
                nfl++; S->nflip_it++; if (S->trace && S->trace_nf < 512) S->trace_fl[S->trace_nf++] = j;
            } else { shift_cost(S, j); S->nsh_it++; }
        }
        if (nfl) {
            S->nflips += nfl;
            double dd = S->dens_col; ftran(S, bf, &dd, NULL);
            if (bf->cnt >= 0) { for (int k = 0; k < bf->cnt; k++) { const int i = bf->idx[k]; S->x[S->basic[i]] -= bf->val[i]; } }
            else for (int i = 0; i < m; i++) S->x[S->basic[i]] -= bf->val[i];
        }
        if (S->tdiag) { const double t1 = now(); S->t_dupd += t1 - tdg; tdg = t1; }
        /* the primal step */
        const double xr2 = S->x[jl];
        const double bound = delta < 0 ? S->lo[jl] : S->up[jl];
        const double thp = (xr2 - bound) / acol;
        if (col->cnt >= 0) { for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= thp * col->val[i]; } }
        else for (int i = 0; i < m; i++) if (col->val[i] != 0) S->x[S->basic[i]] -= thp * col->val[i];
        S->x[q] += thp;
        S->x[jl] = bound;
        if (S->tdiag) { const double t1 = now(); S->t_pupd += t1 - tdg; tdg = t1; }
        /* weights */
        if (S->env_ddevex) {
            /* (13 November 2026, an experiment: SXDUALDEVEX=2) the reference framework reset when the leaving row's Devex weight
             * drifts more than threefold from its exact value |rho_r|^2, known in the iteration */
            if (S->env_ddevex == 2 && (S->w[r] > 3 * wr || 3 * S->w[r] < wr)) { for (int i = 0; i < S->m; i++) S->w[i] = 1.0; S->heap_ok = 0; S->n_dvxreset++; }
            update_ddevex(S, r, col);
        } else update_dse(S, r, col, tau, wr);
        if (S->tdiag) S->t_dse += now() - tdg;
        /* the basis change */
        S->basic[r] = q; S->pos[q] = r; S->move[q] = 0;
        S->pos[jl] = -1;
        if (S->vfac) { S->vfac[q] = 1.0; S->vent[q] = S->iter; S->vden[q] = col->cnt >= 0 ? col->cnt : m; }   /* (BRISK_SXVFAC) */
        if (S->vfac && S->iter - S->vf_last >= S->vf_p) {   /* (BRISK_SXVFAC) the columns that became basic since and lie outside their bounds */
            for (int i = 0; i < m; i++) {
                const int j = S->basic[i];
                if (j >= S->n || S->vent[j] <= S->vf_last || S->vfac[j] != 1.0 || !(infeas(S, i) > 0)) continue;
                if (S->vf_d && !(S->vden[j] > S->dens_col * m)) continue;
                if (S->vf_once && S->vonce[j]) continue;
                const double F = fmax(1.0, S->vf_c * sqrt((double)(S->Ap[j + 1] - S->Ap[j])));
                if (F > 1.0) { S->vfac[j] = F; S->vonce[j] = 1; if (S->heap_ok) heap_touch(S, i); }
            }
            S->vf_last = S->iter;
        }
        rows_enter(S, q); rows_leave(S, jl);
        S->d[jl] = -thd;
        if (S->lo[jl] == S->up[jl]) S->move[jl] = 0; else S->move[jl] = delta < 0 ? 1 : -1;
        shift_back(S, jl);
        const int bad = sxlu_update(&S->F, r, &S->spike, &S->upart, acol);
        if (bad) need_rebuild = 1;
        /* the leaving-row candidates of the rows that changed */
        if (S->tdiag) tdg = now();
        heap_touch_vec(S, col);
        if (nfl) heap_touch_vec(S, bf);
        if (S->heap_ok) heap_touch(S, r);
        else if (!S->Ifull && !S->inI[r] && infeas(S, r) > 0) { S->inI[r] = 1; S->Ilist[S->nI++] = r; }
        if (S->tdiag) S->t_heap += now() - tdg;
        if (S->trace) {
            fprintf(S->trace, "%ld r %d jl %d q %d nzq %d delta %.4g w %.4g acol %.4g thd %.4g thp %.4g col %d tau %d grp %d nfl %d :", S->iter, r, jl, q,
                    q < S->n ? S->Ap[q + 1] - S->Ap[q] : 1, delta, w_r, acol, thd, thp, col->cnt, tau->cnt, S->fg_n, nfl);
            for (int k = 0; k < S->trace_nf; k++) fprintf(S->trace, " %d (%.3g)", S->trace_fl[k], S->row.val[S->trace_fl[k]]);
            fprintf(S->trace, " | jl [%.4g, %.4g] q [%.4g, %.4g] dq %.4g", S->lo[jl], S->up[jl], S->lo[q], S->up[q], S->d[q]);
            fputc('\n', S->trace);
        }
        S->iter++; S->it_dual++; if (S->phase == 1) S->it_ph1++;
        if (S->prog) S->prog->it = S->prog_base + S->iter;   /* (the race's log: one store, nothing computed for it) */
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
    S->n_prescan++;
    if (!S->pc_mb) S->pc_mb = xmalloc(sizeof(double) * (S->N + 1));
    /* (1 November 2026) the merits once, with their columns in increasing order: the selection on a copy, then the list from
     * the stored pairs (three passes of pmerit over every column before: facility, LP, a rebuild every five iterations) */
    if (!S->pc_mj) { S->pc_mj = xmalloc(sizeof(int) * (S->N + 1)); S->pc_mk = xmalloc(sizeof(double) * (S->N + 1)); }
    int *mj = S->pc_mj; double *mk = S->pc_mk;
    double *mb = S->pc_mb; int np = 0;
    for (int j = 0; j < S->N; j++) { const double k = pmerit(S, j); if (k > 0) { mj[np] = j; mk[np] = k; mb[np++] = k; } }
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
    for (int t = 0; t < np; t++) if (mk[t] > S->pc_tau) pc_put(S, mj[t], mk[t]);
    if (S->pc_tau > 0) for (int t = 0; t < np && S->pc_n < S->pc_cap; t++) if (mk[t] == S->pc_tau && S->pc_at[mj[t]] < 0) pc_put(S, mj[t], S->pc_tau);
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
/* (27 October 2026; by default since 28 October, BRISK_SXDEVEX2=0: off) a new reference framework: the nonbasic variables, every weight 1 */
static void devex_reset(Sx *S) {
    if (S->pse_on) return;
    for (int j = 0; j < S->N; j++) { S->dvx[j] = 1.0; S->dvref[j] = S->pos[j] < 0; }
    S->dvbad = 0; S->ndvxreset++; S->ch_ok = 0;
}
static void devex_check(Sx *S) {
    if (S->pse_on) return;
    if (S->dvx2) { double mx = 0; for (int j = 0; j < S->N; j++) if (S->pos[j] < 0 && S->dvx[j] > mx) mx = S->dvx[j]; if (mx > 1e6) devex_reset(S); return; }
    double mx = 0;
    for (int j = 0; j < S->N; j++) if (S->pos[j] < 0 && S->dvx[j] > mx) mx = S->dvx[j];
    if (mx > 1e6) { for (int j = 0; j < S->N; j++) S->dvx[j] = 1.0; S->ndvxreset++; S->ch_ok = 0; }
}
static int cmp_bp(const void *a, const void *b) { const struct SxBp *x = a, *y = b; return x->t < y->t ? -1 : x->t > y->t ? 1 : x->e - y->e; }
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
static void unscaled_violations(const Sx *S, double tp, double td, double *pv, double *dv, double *psv, double *dsv);
/* (4 November 2026, ISSUES 839) the primal method's last good basis (its bounds and nonbasic values with it): saved at each
 * refactorization whose recomputed values agree with the updated ones, for the step back when a factorization shows that
 * the basis has lost its conditioning (BRISK_SXPBACKREL, primal_loop) */
static void pgood_save(Sx *S) {
    const int m = S->m, N = S->N;
    if (!S->gb_basic) {
        S->gb_basic = malloc(sizeof(int) * ((size_t)m + 1)); S->gb_pos = malloc(sizeof(int) * ((size_t)N + 1)); S->gb_move = malloc((size_t)N + 1);
        S->gb_x = malloc(sizeof(double) * ((size_t)N + 1)); S->gb_lo = malloc(sizeof(double) * ((size_t)N + 1)); S->gb_up = malloc(sizeof(double) * ((size_t)N + 1));
    }
    memcpy(S->gb_basic, S->basic, sizeof(int) * (size_t)m); memcpy(S->gb_pos, S->pos, sizeof(int) * (size_t)N); memcpy(S->gb_move, S->move, (size_t)N);
    memcpy(S->gb_x, S->x, sizeof(double) * (size_t)N); memcpy(S->gb_lo, S->lo, sizeof(double) * (size_t)N); memcpy(S->gb_up, S->up, sizeof(double) * (size_t)N);
    S->gb_ok = 1;
}
static void pgood_restore(Sx *S) {
    const int m = S->m, N = S->N;
    memcpy(S->basic, S->gb_basic, sizeof(int) * (size_t)m); memcpy(S->pos, S->gb_pos, sizeof(int) * (size_t)N); memcpy(S->move, S->gb_move, (size_t)N);
    memcpy(S->x, S->gb_x, sizeof(double) * (size_t)N); memcpy(S->lo, S->gb_lo, sizeof(double) * (size_t)N); memcpy(S->up, S->gb_up, sizeof(double) * (size_t)N);
}
/* (BRISK_SXPRT) phase 1: a feasible variable of a small entry that the step took beyond its bound (by at most the larger
 * tolerance of the ratio test's stability test) has the bound moved, as in phase 2, and stays feasible */
static void prt_shift_ph1(Sx *S, int r) {
    const double ptol = S->tolp;
    for (int k = 0; k < S->prt_n; k++) {
        const int i = S->prt_list[k], j = S->basic[i]; if (i == r) continue;
        const double x_ = S->x[j];
        if (x_ < S->lo[j] - ptol) { S->lo[j] = x_ - (1 + urand(S)) * ptol; S->nbshift++; S->n_prtshift++; }
        else if (x_ > S->up[j] + ptol) { S->up[j] = x_ + (1 + urand(S)) * ptol; S->nbshift++; S->n_prtshift++; }
    }
}
static int primal_loop(Sx *S) {
    SxVec *rho = &S->rho, *col = &S->col;
    int need_rebuild = 1, verified = 0, ph1 = 0;
    long last_log = -1000;
    S->ch_ok = 0;
    S->vhash = 0;
    S->gb_ok = 0;     /* (the last good basis of this call only: the bounds of an earlier call's are not this call's) */
    for (;;) {
        int st = check_stop(S);
        if (st) return st;
        if (need_rebuild || need_refactor(S)) {
            const int was2 = !ph1 && S->it_primal > 0;
            if (S->fresh && !need_refactor(S)) S->fresh = 0;      /* (the factorization and the values as sxm_solve left them) */
            else {
                S->fresh = 0;
                double sb = 0; int nb = 0; if (S->dbg >= 2 || S->pstab) for (int i = 0; i < S->m; i++) { const double f = infeas(S, i); if (f > 0) { sb += f; nb++; } }
                /* (5 November 2026, BRISK_SXPBACKREL) the largest change of a basic value by the new factorization, relative to
                 * the values' scale (1 + the largest |x_B| before) */
                const int chk = (S->pback_rel > 0 && S->pse_route) || S->dbg >= 2;
                double xs_old = 0;
                if (chk) { if (!S->px_old) S->px_old = xmalloc(sizeof(double) * ((size_t)S->N + 1)); memcpy(S->px_old, S->x, sizeof(double) * (size_t)S->N);
                    for (int i = 0; i < S->m; i++) { const double a_ = fabs(S->x[S->basic[i]]); if (a_ > xs_old) xs_old = a_; } }
                refactor(S); compute_primal(S); devex_check(S);
                double dxrel = 0; if (chk) { double dm = 0; for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; const double a_ = fabs(S->x[j] - S->px_old[j]); if (a_ > dm) dm = a_; } dxrel = dm / (1 + xs_old); }
                if (S->dbg >= 2 || S->pstab) {
                    double sa = 0; int na = 0; for (int i = 0; i < S->m; i++) { const double f = infeas(S, i); if (f > 0) { sa += f; na++; } }
                    if (S->dbg >= 2) { double xm = 0; for (int i = 0; i < S->m; i++) { const double a_ = fabs(S->x[S->basic[i]]); if (a_ > xm) xm = a_; }
                        printf("  primal refactorization at %ld: infeasibility %.3e (%d) before, %.3e (%d) after; max|xB| %.2e; pivots %d, rel<1e-3: %d, min rel %.1e, thr %.2g; change %.1e\n", S->iter, sb, nb, sa, na, xm, S->pv_n, S->pv_nsmall, S->pv_minrel, S->F.pivthr, dxrel); }
                    S->pv_n = 0; S->pv_nsmall = 0; S->pv_minrel = 1;
                    /* (by default since 28 October 2026; BRISK_SXPSTAB=0: off) the values kept by the updates and those of the new factorization far
                     * apart (the infeasibility at least ten times, and of at least ten more rows): the threshold raised */
                    const int jump = S->pstab && sa > 10 * sb + 1e-6 && na > nb + 10;
                    /* (5 November 2026, BRISK_SXPBACKREL) the basis has lost its conditioning: the new factorization changes a basic
                     * value by more than pback_rel times the values' scale; the drift of the updates changes them by far less (at
                     * most 4e-3 on the 40 random LPs, while the infeasibility-jump rule above reacts to 1e-5 to 1e-2 over 15-80
                     * rows; 4.3e4 at momentum2's failure, LP). Back to the last good basis at once, with the threshold at 0.5:
                     * raising the threshold cannot repair the basis itself. A safety net only: without the stability test it does
                     * not rescue momentum2 (the same failure comes back) */
                    const int lost = S->pback_rel > 0 && S->pse_route && dxrel > S->pback_rel;
                    if (lost && S->gb_ok && S->nback < 5) {
                        S->nback++; S->nptrouble++; S->n_tr_lost++; S->F.pivthr = 0.5;
                        if (S->verbose >= 2) printf("  primal refactorization at %ld: the basic values changed by %.1e of their scale: the basis has lost its conditioning, back to the last good basis\n", S->iter, dxrel);
                        pgood_restore(S); refactor(S); compute_primal(S); S->pse_on = 0; devex_reset(S);
                    } else if (jump && S->F.pivthr < 0.5) { S->F.pivthr = 0.5; S->nptrouble++; S->n_tr_jump++; }
                    else if (!jump && !lost && S->pback_rel > 0 && S->pse_route) pgood_save(S);
                }
            }
            if (was2) shift_bounds_primal(S, NULL);      /* the drift of the recomputation absorbed in phase 2 */
            ph1 = primal_infeasible(S);
            if (ph1) compute_dual_ph1(S); else compute_dual(S);
            /* (1 November 2026, ISSUES 815) in the clean-up rounds: done once the problem as read is dual feasible within the target (the
             * tightened working tolerance can keep the method going at the level of its rounding: nb10tb, LP, 10,000
             * iterations after the target was met) */
            if (S->cl_target > 0 && !ph1 && S->it_primal > 0) {
                double pv, dv, psv, dsv; unscaled_violations(S, S->cl_target, S->cl_target, &pv, &dv, &psv, &dsv);
                /* (the dual violation alone: the primal violations here are the perturbed bounds', which the dual method
                 * removes after the bounds are restored, primal_simplex) */
                if (dv <= S->cl_target) return 0;
            }
            S->ch_ok = 0;
            need_rebuild = 0;
        }
        if (S->iter - last_log >= 1000) { log_line(S, ph1 ? "prim ph1" : "primal"); last_log = S->iter; }
        if (brisk_stab_due(S->iter)) { int c_; const double s_ = sum_pinf(S, &c_); brisk_stab_row(S->iter, ph1 ? "primal, phase 1" : "primal", g_sx_tabsign * objective(S) * S->costscale + g_sx_taboff, s_, c_); }
        const int q = chuzc_primal(S);
        if (q < 0) {
            if (S->F.nupd > 0 && !verified && !(S->noverify && !ph1)) {
                if (S->pcyc_on && !ph1 && S->pse_route) {
                    /* (5 November 2026, BRISK_SXPCYC) phase 2 ends at the same basis (a hash of the basic set) as at the last
                     * verification: the factorization's new values and duals (on a basis whose recomputed values move by more
                     * than the tolerances) sent the method round a cycle back to it (momentum2, LP, Devex pricing at random
                     * sequence 1: a verification every 17 iterations, 11 rows off by 1e-2 at values of 1e6, until the time
                     * limit). The loop ends; primal_simplex restores the bounds and its rounds (the dual method for the primal
                     * infeasibilities, the primal method for the dual ones) finish from this basis */
                    unsigned long long h = 0; for (int i = 0; i < S->m; i++) h += ((unsigned long long)S->basic[i] + 1) * 0x9E3779B97F4A7C15ULL ^ (((unsigned long long)S->basic[i] + 1) << 17);
                    if (h == S->vhash) { S->n_pcyc++; if (S->verbose >= 2) printf("  the primal method at %ld: the same basis as at the last verification: the loop ends\n", S->iter); return 0; }
                    S->vhash = h;
                }
                need_rebuild = 1; verified = 1; continue;
            }
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
        int dbg_c = 0; const double dbg_f0 = S->dbg >= 3 ? sum_pinf(S, &dbg_c) : -1, dbg_dq = S->d[q]; const int dbg_ph1 = ph1;
        const double dir = (S->move[q] == 0) ? (S->d[q] > 0 ? -1.0 : 1.0) : (double)S->move[q];
        sxv_clear(col); col_scatter(S, q, col, 1.0);
        ftran(S, col, &S->dens_col, &S->spike);
        if (col->cnt < 0) sxv_reindex(col);
        /* ratio test (Harris, two passes): x_B(t) = x_B - t dir alpha */
        const double ptol = S->tolp;
        const double atol = 1e-9;
        double thmax = SX_INF;
        /* (5 November 2026, ISSUES 839, BRISK_SXPRT) a stability test in the ratio test (Harris's tolerance by the pivot's
         * size, as SoPlex's fast ratio test does): when the pivot chosen is below prt times the column's largest entry,
         * the two passes are repeated with prt_tol times the feasibility tolerance for every entry that small in the first
         * pass, so that it blocks the step later and the second pass, which takes the
         * largest entry among those that block first, can choose a larger pivot. The pivot's size against its column's
         * largest bounds the growth of the factor's update (the eta column alpha / alpha_r). momentum2 (LP): 75 % of the
         * pivots below 1e-3 of their column's largest, down to 1e-13; the basic values grew from 3e2 to 2e6 and a
         * factorization then gave 9e10. The violation a relaxed entry may reach is at most prt_tol tolerances: phase 2's
         * bound shifts absorb it, and in phase 1 prt_shift_ph1 moves the bound of a variable that was feasible */
        double asmall = 0, amax = 0; S->prt_n = 0;
        const double ptolb = ptol * S->prt_tol;
    ratio_again:
        thmax = SX_INF;
        for (int k = 0; k < col->cnt; k++) {
            const int i = col->idx[k];
            const double a = dir * col->val[i];
            if (fabs(a) < atol) continue;
            if (fabs(a) > amax) amax = fabs(a);
            const int j = S->basic[i];
            const double v = S->x[j];
            double lo = S->lo[j], up = S->up[j];
            if (ph1) {   /* an infeasible variable may move up to its violated bound, and not beyond */
                if (v < lo - ptol) { if (a < 0) { up = lo; lo = -SX_INF; } else continue; }
                else if (v > up + ptol) { if (a > 0) { lo = up; up = SX_INF; } else continue; }
            }
            if (a > 0 ? lo <= -SX_INF : up >= SX_INF) continue;
            const double pt = fabs(a) < asmall ? ptolb : ptol;
            if (pt != ptol && ph1 && v >= S->lo[j] - ptol && v <= S->up[j] + ptol) { if (!S->prt_list) S->prt_list = xmalloc(sizeof(int) * ((size_t)S->m + 1)); S->prt_list[S->prt_n++] = i; }
            const double t = a > 0 ? (v - lo + pt) / a : (up - v + pt) / -a;
            if (t < thmax) thmax = t;
        }
        int r = -1; double best = 0, thp = 0;
        int lsel = 0; double lbound = 0;
        if (ph1 && S->ph1long) {
            /* (4 November 2026, BRISK_SXPH1LONG) phase 1's ratio test over the breakpoints of the sum of infeasibilities
             * (Wolfe 1965; Maros 2003, ch. 9): the step passes the points where an infeasible basic variable becomes
             * feasible while the slope stays negative, each lowering it from |d_q| by |alpha_i|. Mode 1: a feasible
             * variable's bound, and an infeasible one's other bound, stop the step (none becomes infeasible); mode 2: they
             * are breakpoints too, passed the same way (HiGHS's phase1ChooseRow, which stops one point short). The leaving
             * variable: among the points up to the step, the last whose |alpha_i| is at least a tenth of the largest */
            double *bR = S->bp_r, *bT = S->bp_t; int *bi = S->bp_i; struct SxBp *bo = S->bp_o; int nb = 0;
            for (int k = 0; k < col->cnt; k++) {
                const int i = col->idx[k];
                const double a = dir * col->val[i];
                if (fabs(a) < atol) continue;
                const int j = S->basic[i];
                const double v = S->x[j], lo = S->lo[j], up = S->up[j];
                /* bi: 4 i + 2 (hard) + 1 (at the upper bound) */
                if (a > 0) {
                    if (v > up + ptol) { bR[nb] = (v - up - ptol) / a; bT[nb] = (v - up) / a; bi[nb++] = 4 * i + 1; }
                    if (lo > -SX_INF && v > lo - ptol) { bR[nb] = (v - lo + ptol) / a; bT[nb] = (v - lo) / a; bi[nb++] = 4 * i + 2; }
                } else {
                    if (v < lo - ptol) { bR[nb] = (lo - ptol - v) / -a; bT[nb] = (lo - v) / -a; bi[nb++] = 4 * i; }
                    if (up < SX_INF && v < up + ptol) { bR[nb] = (up + ptol - v) / -a; bT[nb] = (up - v) / -a; bi[nb++] = 4 * i + 3; }
                }
            }
            if (nb > 0) {
                for (int k = 0; k < nb; k++) { bo[k].t = bR[k]; bo[k].e = k; }
                qsort(bo, nb, sizeof *bo, cmp_bp);
                double g = fabs(S->d[q]), dmax = bR[bo[0].e]; int kk;
                for (kk = 0; kk < nb; kk++) {
                    const int e = bo[kk].e;
                    if (S->ph1long == 2) { g -= fabs(col->val[bi[e] >> 2]); if (g <= 0) break; dmax = bR[e]; continue; }
                    dmax = bR[e];
                    if (bi[e] & 2) break;
                    g -= fabs(col->val[bi[e] >> 2]);
                    if (g <= 0) break;
                }
                S->n_ph1pass += kk < nb ? kk : nb - 1;
                /* (7 November 2026; BRISK_SXPH1FLAT, on by default since that day; 0: off) the sum of infeasibilities flat after the point where the step stops
                 * (its slope zero: the infeasible variables that still improve balance those that get worse, typically none
                 * of either), and the entering variable's other bound reached before the next breakpoint: the step goes to
                 * that bound, a bound flip, instead of a basis change at the same phase-1 objective. A boxed column of a
                 * big-M row (lectsched-1, MILP: -M b <= -2 - s with b in [0, 1]) otherwise enters the basis at the fraction
                 * that just makes its row feasible, and the vertex reached has one basic structural column more for each
                 * row repaired so */
                int flatflip = 0;
                if (S->ph1flat && S->ph1long == 1 && kk < nb && !(bi[bo[kk].e] & 2) && fabs(g) <= 1e-9 * (1 + fabs(S->d[q]))) {
                    const double rq = S->up[q] - S->lo[q], tnext = kk + 1 < nb ? bR[bo[kk + 1].e] : SX_INF;
                    if (rq < SX_INF && rq <= tnext) flatflip = 1;
                }
                double amax = 0;
                for (int k = 0; k < nb; k++) if (fmin(bR[k], bT[k]) <= dmax) { const double aa = fabs(col->val[bi[k] >> 2]); if (aa > amax) amax = aa; }
                double tbest = -SX_INF;
                for (int k = 0; k < nb; k++) {
                    if (fmin(bR[k], bT[k]) > dmax) continue;
                    const int i = bi[k] >> 2; const double aa = fabs(col->val[i]);
                    if (aa >= 0.1 * amax && (bT[k] > tbest || (bT[k] == tbest && aa > best))) { tbest = bT[k]; best = aa; r = i; lbound = (bi[k] & 1) ? S->up[S->basic[i]] : S->lo[S->basic[i]]; }
                }
                thmax = dmax; thp = tbest; lsel = 1;
                if (flatflip) { thmax = thp = S->up[q] - S->lo[q]; r = -1; S->n_ph1flat++; }
            }
        } else if (thmax < SX_INF) {
            for (int k = 0; k < col->cnt; k++) {
                const int i = col->idx[k];
                const double a = dir * col->val[i];
                if (fabs(a) < atol) continue;
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
            /* (the stability test: the passes again, with the larger tolerance for the small entries, only when the pivot
             * chosen is small; most columns keep the plain ratio test's choice and cost nothing more) */
            if (asmall == 0 && S->prt > 0 && S->pse_route && r >= 0 && best < S->prt * amax) { asmall = S->prt * amax; S->n_prt++; goto ratio_again; }
            if (asmall > 0 && r >= 0 && best < asmall) S->n_prtfail++;
            /* (7 November 2026; BRISK_SXPH1FLAT, on by default since that day; 0: off) the same rule as in the long ratio test above, for the plain one: the row
             * chosen is an infeasible variable that becomes feasible there, the sum of infeasibilities is flat beyond it
             * (|d_q| = |alpha_r|), and no other basic variable stops the step before the entering variable's other bound:
             * a bound flip instead of the basis change */
            if (S->ph1flat && ph1 && r >= 0) {
                const int jr = S->basic[r]; const double vr = S->x[jr];
                const int soft = vr < S->lo[jr] - ptol || vr > S->up[jr] + ptol;
                const double rq = S->up[q] - S->lo[q];
                if (soft && rq < SX_INF && fabs(fabs(S->d[q]) - fabs(col->val[r])) <= 1e-9 * (1 + fabs(S->d[q]))) {
                    double tmin = SX_INF;
                    for (int k = 0; k < col->cnt && tmin >= rq; k++) {
                        const int i = col->idx[k];
                        const double a = dir * col->val[i];
                        if (fabs(a) < atol) continue;
                        const int j = S->basic[i];
                        const double v = S->x[j];
                        double lo = S->lo[j], up = S->up[j];
                        if (i == r) {       /* (the row's own other bound: an equation is infeasible again at once) */
                            const double t = a < 0 ? (up < SX_INF ? (up - v + ptol) / -a : SX_INF) : (lo > -SX_INF ? (v - lo + ptol) / a : SX_INF);
                            if (t < tmin) tmin = t;
                            continue;
                        }
                        if (v < lo - ptol) { if (a < 0) { up = lo; lo = -SX_INF; } else continue; }
                        else if (v > up + ptol) { if (a > 0) { lo = up; up = SX_INF; } else continue; }
                        if (a > 0 ? lo <= -SX_INF : up >= SX_INF) continue;
                        const double t = a > 0 ? (v - lo + ptol) / a : (up - v + ptol) / -a;
                        if (t < tmin) tmin = t;
                    }
                    if (rq <= tmin) { thmax = thp = rq; r = -1; S->n_ph1flat++; }
                }
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
            if (ph1 && S->prt_n) prt_shift_ph1(S, -1);
            pc_set_key(S, q, pmerit(S, q));
        } else {
            if (thp < 0) thp = 0;          /* a variable slightly beyond its bound: a step of zero */
            if (thp == 0) S->n_pdegen++;
            jl = S->basic[r];
            ar = col->val[r];
            S->n_alpha += col->cnt;
            if (S->dbg >= 2) { double am = 0; for (int k = 0; k < col->cnt; k++) { const double a_ = fabs(col->val[col->idx[k]]); if (a_ > am) am = a_; }
                const double rel = fabs(ar) / am; S->pv_n++; if (rel < 1e-3) S->pv_nsmall++; if (rel < S->pv_minrel) S->pv_minrel = rel; }
            const double a = dir * ar;
            if (lsel) bound = lbound;
            else {
                const double v = S->x[jl];
                if (ph1 && v < S->lo[jl] - ptol) bound = S->lo[jl];
                else if (ph1 && v > S->up[jl] + ptol) bound = S->up[jl];
                else bound = a > 0 ? S->lo[jl] : S->up[jl];
            }
            /* BTRAN e_r and the pivotal row (the dual update and the Devex weights) */
            sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
            int vdone = 0;
            if (S->pse_on) {
                /* the steepest-edge weights' solve v = B^{-T} alpha_q, independent of rho's: beside it on a second thread
                 * when there is one and the work pays (each solve by one thread, the same routine either way) */
                SxVec *v = &S->pv; sxv_clear(v);
                for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; v->val[i] = col->val[i]; v->idx[v->cnt++] = i; }
#ifdef _OPENMP
                if (S->nthreads > 1 && omp_get_max_threads() > 1 && (S->dens_rho + S->dens_pv) * S->m >= SX_PAR_FTRAN) {
                    S->ws2.nflops = 0; S->ws2.nedges = 0; double tv = 0;
                    #pragma omp parallel sections num_threads(2)
                    {
                        #pragma omp section
                        btran(S, rho, &S->dens_rho, &S->upart);
                        #pragma omp section
                        { const double t1 = now(); sxlu_btran_ws(&S->F, &S->ws2, &S->pv, S->dens_pv); tv = now() - t1; }
                    }
                    btran_pse_note(S, &S->pv, tv); S->npar_pse++;
                    vdone = 1;
                }
#endif
            }
            if (!vdone) btran(S, rho, &S->dens_rho, &S->upart);
            if (rho->cnt < 0) sxv_reindex(rho);
            price(S, rho);
            S->n_row += S->row.cnt;
            if (S->pstab) {
                /* (27 October 2026; by default since 28 October, BRISK_SXPSTAB=0: off) the pivot from the row and from the column disagree:
                 * nothing is updated, the basis is factored again; when they disagree right after a factorization, the
                 * threshold of its partial pivoting is raised, 0.1 to 0.5 (HiGHS's kPivotThresholdChangeFactor and
                 * kMaxPivotThreshold); at 0.5 the pivot is taken */
                const double arow = S->row.val[q];
                if (fabs(ar - arow) > 1e-7 * (1 + fabs(ar)) && (S->F.nupd > 0 || S->F.pivthr < 0.5)) {
                    S->nptrouble++; S->n_tr_rowcol++;
                    if (S->F.nupd == 0) S->F.pivthr = 0.5;
                    need_rebuild = 1; continue;
                }
            }
            /* primal update */
            const double step = thp * dir;
            for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= step * col->val[i]; }
            S->x[q] += step;
            S->x[jl] = bound;
            if (!ph1) shift_bounds_primal(S, col);
            else if (S->prt_n) prt_shift_ph1(S, r);
            /* dual update (with the costs as they are: phase 1's change below) and Devex weights, (1 November 2026) with the
             * candidates' keys in one pass over the pivotal row (three passes before: facility, LP, rows of thousands of
             * entries; each entry's key reads only its own reduced cost and weight, so the same keys in the same order) */
            const double thd = S->d[q] / ar;
            {
                double wq = S->dvx[q];
                if (S->dvx2 && !S->pse_on) {
                    /* (by default) the entering column's weight computed from its column (the reference framework's part of
                     * B^{-1} a_q, and q itself), a stored weight more than 3 times its norm counted as bad, the framework
                     * reset after more than 3 (HiGHS, HEkkPrimal::updateDevex; Forrest and Goldfarb 1992) */
                    double we = S->dvref[q];
                    for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; if (S->dvref[S->basic[i]]) we += col->val[i] * col->val[i]; }
                    if (wq > 9 * we) S->dvbad++;
                    wq = we;
                }
                const int touch = S->ch_ok;
                if (S->pse_on) {
                    /* (4 November 2026, BRISK_SXPSE) the steepest-edge weights g_j = 1 + |B^{-1} a_j|^2 (Goldfarb and Reid
                     * 1977; Forrest and Goldfarb 1992): with r_j = alpha_rj / alpha_rq, g_j <- max(g_j - 2 r_j a_j' v
                     * + r_j^2 g_q, 1 + r_j^2), g_q = 1 + |alpha_q|^2 exactly, the leaving variable's max(g_q / alpha_rq^2, 1) */
                    double gq = 1; for (int k = 0; k < col->cnt; k++) { const double a = col->val[col->idx[k]]; gq += a * a; }
                    SxVec *v = &S->pv;
                    if (!vdone) btran_pse(S, v);
                    if (v->cnt < 0) sxv_reindex(v);
                    /* a_j' v for the columns of the pivotal row only: column by column when their entries are fewer than
                     * those of v's rows, else by a pricing of v */
                    const double tp0_ = now();
                    long wc = 0, wr = 0;
                    for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; wc += j < S->n ? S->Ap[j + 1] - S->Ap[j] : 1; }
                    for (int k = 0; k < v->cnt; k++) wr += S->Rnb[v->idx[k]] + 1;
                    if (wc <= wr) {
                        double *pv2 = S->row2.val;
                        for (int k = 0; k < S->row.cnt; k++) {
                            const int j = S->row.idx[k];
                            double t = 0;
                            if (j < S->n) { for (int pp = S->Ap[j]; pp < S->Ap[j + 1]; pp++) t += S->Ax[pp] * v->val[S->Ai[pp]]; }
                            else t = v->val[j - S->n];
                            pv2[j] = t; S->row2.idx[k] = j;
                        }
                        S->row2.cnt = S->row.cnt;
                        S->work += wc;
                    } else {
                        SxVec tr = S->row; S->row = S->row2; const int ro = S->row_ord; const double dr = S->dens_row;
                        price(S, v); S->row2 = S->row; S->row = tr; S->row_ord = ro; S->dens_row = dr;
                    }
                    for (int k = 0; k < S->row.cnt; k++) {
                        const int j = S->row.idx[k]; if (j == q) continue;
                        const double rv = S->row.val[j];
                        S->d[j] -= thd * rv;
                        const double rr = rv / ar;
                        double g = S->dvx[j] - 2 * rr * S->row2.val[j] + rr * rr * gq;
                        if (g < 1 + rr * rr) g = 1 + rr * rr;
                        S->dvx[j] = g;
                        if (touch) pc_set_key(S, j, pmerit(S, j));
                    }
                    sxv_clear(&S->row2); sxv_clear(v);
                    S->t_prod += now() - tp0_; S->fl_prod += wc < wr ? wc : wr;
                    S->d[q] = 0; S->d[jl] = -thd;
                    const double wl = gq / (ar * ar);
                    S->dvx[jl] = wl > 1 ? wl : 1;
                } else {
                for (int k = 0; k < S->row.cnt; k++) {
                    const int j = S->row.idx[k];
                    const double rv = S->row.val[j];
                    S->d[j] -= thd * rv;
                    const double rr = rv / ar;
                    const double nw = rr * rr * wq;
                    if (nw > S->dvx[j]) S->dvx[j] = nw < 1e30 ? nw : 1e30;
                    if (touch) pc_set_key(S, j, j == q ? 0.0 : pmerit(S, j));     /* (q: its reduced cost set to 0 below, as before) */
                }
                S->d[q] = 0; S->d[jl] = -thd;
                const double wl = wq / (ar * ar);
                S->dvx[jl] = wl > 1 ? (wl < 1e30 ? wl : 1e30) : 1;
                }
            }
            S->basic[r] = q; S->pos[q] = r; S->move[q] = 0;
            S->pos[jl] = -1;
            rows_enter(S, q); rows_leave(S, jl);
            if (S->lo[jl] == S->up[jl]) S->move[jl] = 0;
            else if (bound == S->lo[jl]) S->move[jl] = 1;
            else S->move[jl] = -1;
            const int bad = sxlu_update(&S->F, r, &S->spike, &S->upart, ar);
            if (bad) need_rebuild = 1;
            /* the candidates: the row's entries (above), the entering one out, the leaving one in */
            pc_set_key(S, q, 0);
            if (S->dvx2 && S->dvbad > 3) devex_reset(S);
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
        if (S->dbg >= 3 && dbg_f0 >= 0) { int c2; const double f1 = sum_pinf(S, &c2); if (f1 > dbg_f0 * (1 + 1e-6) + 1e-6) printf("  phase-%d increase at %ld: %.6e (%d) -> %.6e (%d); q %d (%s) dir %+g d_q %.3e, %s, step %.3e, leaving row %d (variable %d, %s), |alpha_r| %.2e, |col| %d\n", dbg_ph1 ? 1 : 2, S->iter, dbg_f0, dbg_c, f1, c2, q, q < S->n ? "column" : "logical", dir, dbg_dq, flip ? "flip" : "pivot", thp, r, jl, jl < 0 ? "-" : jl < S->n ? "column" : "logical", fabs(ar), col->cnt); }
        S->iter++; S->it_primal++; if (ph1) S->it_ph1++;
        if (S->prog) S->prog->it = S->prog_base + S->iter;
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
    if (S->dvx2) devex_reset(S);
    S->pse_on = S->pse && S->pse_route;
    if (S->pse_on) {     /* the steepest-edge weights 1 + |a_j|^2: exact for the logicals' basis (B = I), estimates otherwise */
        if (!S->pv.val) { sxv_init(&S->pv, S->m); sxv_init(&S->row2, S->N); }
        for (int j = 0; j < S->N; j++) {
            if (S->pos[j] >= 0) { S->dvx[j] = 1; continue; }
            double g = 2;
            if (j < S->n) { g = 1; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) g += S->Ax[p] * S->Ax[p]; }
            S->dvx[j] = g;
        }
        S->ch_ok = 0;
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
static void unscaled_violations(const Sx *S, double tp, double td, double *pv, double *dv, double *psv, double *dsv) {
    /* (28 October 2026) psv, dsv: the least violation, as the working method measures it (in the scaled problem), of
     * the entries whose violation here exceeds the target tp or td; the clean-up rounds set the working tolerance below
     * it, so that the method sees the entries it must repair */
    double p = 0, d = 0, ps = SX_INF, ds = SX_INF;
    /* d_j * costscale / max(1, min(costscale, cmax_read / 1e4)): (27 October 2026, the night) the divisor no larger than
     * the costs as read allow. The cost scale follows the costs after the column scaling, which can be far larger: on
     * bley_xs2 (LP, MIPLIB 2017 collection; entries of 1e20) column factors up to 4.5e15 made it 1.1e15, and a dual
     * violation of 7.1e6 against costs of 1e7 was measured as 6.3e-9 and called optimal (Gurobi's optimum 97,250; ours
     * 2.68e7). Where the column scaling does not inflate the costs (cost scale at most cmax_read / 1e4) nothing changes */
    const double dvs = fmax(1.0, fmin(S->costscale, S->cmax_read > 1e4 ? pow2round(S->cmax_read / 1e4) : 1.0));   /* (rounded as the cost scale is) */
    const double cs = S->costscale / dvs;
    for (int i = 0; i < S->m; i++) {
        const int j = S->basic[i];
        const double sc = j < S->n ? S->cscale[j] : S->irscale[j - S->n];
        const double v = S->x[j];
        /* (28 October 2026; ISSUES 629) a violation the dual method found no column to remove in these rounds and accepted
         * by its own rule (at most 1e-6 relative to the bound, in the scaled problem: dual_loop) is accepted here too: the
         * two rules had disagreed between 1e-7 and 1e-6, and the rounds then ended in numerical difficulties (glass4's
         * cut rows: 6e-6 above a bound of 8.5) */
        const int acc = S->tiny_on && S->tiny && j < S->tiny_n && S->tiny[j];
        if (v < S->lo0[j] && !(acc && S->lo0[j] - v <= 1e-6 * (1 + fabs(S->lo0[j])))) { const double e = (S->lo0[j] - v) * sc / (1 + fabs(S->lo0[j] * sc)); p = fmax(p, e); if (e > tp) ps = fmin(ps, S->lo0[j] - v); }
        if (v > S->up0[j] && !(acc && v - S->up0[j] <= 1e-6 * (1 + fabs(S->up0[j])))) { const double e = (v - S->up0[j]) * sc / (1 + fabs(S->up0[j] * sc)); p = fmax(p, e); if (e > tp) ps = fmin(ps, v - S->up0[j]); }
    }
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0) continue;
        /* (27 October 2026) the inverse scales kept: a division per column was most of this pass in the mixed-integer
         * solver's re-solves (fiball: 6.7 % of its tree search) */
        const double sc = (j < S->n ? S->icscale[j] : S->rscale[j - S->n]) * cs;
        const double dj = S->d[j] * sc;
        if (S->lo0[j] == S->up0[j]) continue;
        const double dold = d;
        const double e = S->move[j] == 0 ? fabs(dj) : S->move[j] * dj < 0 ? -S->move[j] * dj : 0;
        d = fmax(d, e);
        if (e > td) ds = fmin(ds, fabs(S->d[j]));
        {   static int vdb = -1; if (vdb < 0) vdb = getenv("BRISK_SXVIOLDBG") != NULL;   /* (a diagnostic: every dual violation over the target) */
            if (vdb && e > td) printf("  dual violation %.2e at %d (n %d, %s): move %d lo0 %g up0 %g lo %g up %g x %g d %.3e scaled, told %.1e, pos %d\n", e, j, S->n, j < S->n ? "column" : "logical", S->move[j], S->lo0[j], S->up0[j], S->lo[j], S->up[j], S->x[j], S->d[j], S->told, S->pos[j]); }
        if (d > dold && d > 1e-3 && getenv("SXDBG")) printf("  dual violation %g at %d (n %d): move %d lo0 %g up0 %g lo %g up %g x %g d %g\n", d, j, S->n, S->move[j], S->lo0[j], S->up0[j], S->lo[j], S->up[j], S->x[j], S->d[j]);
    }
    *pv = p; *dv = d; *psv = ps; *dsv = ds;
}

/* (7 November 2026; ISSUES 843, BRISK_LPGRBTOL=1) the largest violations of the current basic solution on the scaled problem
 * the method works on, with the costs and bounds of the problem (no shifts or perturbations): the basic variables beyond
 * their bounds, the reduced costs of the nonbasic ones of the wrong sign, absolute, as the method's tolerances measure them.
 * A violation the dual method accepted by its own rule (unscaled_violations, ISSUES 629) is accepted here too */
static void scaled_violations(const Sx *S, double *pv, double *dv) {
    double p = 0, d = 0;
    for (int i = 0; i < S->m; i++) {
        const int j = S->basic[i]; const double v = S->x[j];
        const int acc = S->tiny_on && S->tiny && j < S->tiny_n && S->tiny[j];
        if (v < S->lo0[j] && !(acc && S->lo0[j] - v <= 1e-6 * (1 + fabs(S->lo0[j])))) p = fmax(p, S->lo0[j] - v);
        if (v > S->up0[j] && !(acc && v - S->up0[j] <= 1e-6 * (1 + fabs(S->up0[j])))) p = fmax(p, v - S->up0[j]);
    }
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0 || S->lo0[j] == S->up0[j]) continue;
        const double dj = S->d[j];
        d = fmax(d, S->move[j] == 0 ? fabs(dj) : S->move[j] * dj < 0 ? -S->move[j] * dj : 0);
    }
    *pv = p; *dv = d;
}

/* (8 November 2026, ISSUES 847; BRISK_SXPH1BIGM=<M>, on by default with M = 1e5) the dual phase 1 by artificial bounds: each column
 * that leaves the starting basis dual infeasible (not boxed, its cost of the wrong sign for its one finite bound, or free
 * with a cost) gets the bound its reduced cost asks for at the distance M from its finite bound (from zero when free),
 * and the dual method's phase 2 runs at once on the problem so bounded; it is dual feasible from the start. At the end a
 * column at an artificial bound gets that bound a thousand times further and the loop goes on (twice at most); when none
 * is left at one, the artificial bounds are removed: the basis is optimal for the problem itself. Otherwise (or when the
 * bounded problem is infeasible), the bounds are removed and the phase 1 by the auxiliary problem starts from the basis
 * reached. Gurobi's dual simplex method starts so (its log's first line: objective and primal infeasibility of the order
 * of the artificial bounds); ns1690781 (LP) 1,359 iterations in one phase against our 1,071 in phase 1 and 2,650 after.
 * Returns the loop's status, or -1 for the phase 1 by the auxiliary problem. */
static int dual_bigm(Sx *S, double M0) {
    unsigned char *art = xcalloc((size_t)S->N, 1);
    long nart = 0;
    for (int j = 0; j < S->N; j++) {
        if (S->pos[j] >= 0 || is_boxed(S, j) || S->lo[j] == S->up[j]) continue;
        const double dj = S->d[j], l = S->lo[j], u = S->up[j];
        if (l <= -SX_INF && u >= SX_INF) { if (fabs(dj) <= S->told) continue; if (dj > 0) S->lo[j] = -M0; else S->up[j] = M0; }
        else if (l > -SX_INF) { if (dj >= -S->told) continue; S->up[j] = l + M0; }
        else { if (dj <= S->told) continue; S->lo[j] = u - M0; }
        art[j] = 1; nart++;
        set_nonbasic(S, j, dj);
    }
    if (S->verbose >= 2) printf("  dual phase 1 by artificial bounds: %ld columns bounded at %.1e\n", nart, M0);
    if (getenv("BRISK_SXDBGP1")) { long nf = 0, nl = 0, nu = 0, nlog = 0; for (int j = 0; j < S->N; j++) if (art[j]) { if (j >= S->n) nlog++; if (S->lo0[j] <= -SX_INF && S->up0[j] >= SX_INF) nf++; else if (S->lo0[j] > -SX_INF) nl++; else nu++; }
        fprintf(stderr, "[p1] artificial: %ld free, %ld with a lower bound only, %ld with an upper bound only (%ld logicals)\n", nf, nl, nu, nlog); }
    S->phase = 2;
    int st = -1;
    double M = M0;
    for (int round = 0; ; round++) {
        st = dual_loop(S);
        if (st != 0) break;
        long at = 0;
        for (int j = 0; j < S->N; j++) {
            if (!art[j] || S->pos[j] >= 0) continue;
            if ((S->move[j] > 0 && S->lo[j] != S->lo0[j]) || (S->move[j] < 0 && S->up[j] != S->up0[j])) at++;
        }
        if (S->verbose >= 2) printf("  artificial bounds, round %d: %ld columns at one\n", round, at);
        if (!at) break;
        /* (12 November 2026) widened until M passes 1e11 (three rounds from M = 1e5, as before; more from a smaller start) */
        if (M * 1e3 > 1e11 * (1 + 1e-12)) { st = -1; break; }
        M *= 1e3;
        for (int j = 0; j < S->N; j++) {
            if (!art[j] || S->pos[j] >= 0) continue;
            if (S->move[j] > 0 && S->lo[j] != S->lo0[j]) { S->lo[j] = S->up0[j] < SX_INF ? S->up0[j] - M : -M; S->x[j] = S->lo[j]; }
            else if (S->move[j] < 0 && S->up[j] != S->up0[j]) { S->up[j] = S->lo0[j] > -SX_INF ? S->lo0[j] + M : M; S->x[j] = S->up[j]; }
        }
    }
    for (int j = 0; j < S->N; j++) if (art[j]) { S->lo[j] = S->lo0[j]; S->up[j] = S->up0[j]; }
    free(art);
    normalize_nonbasic(S);
    if (st == 0 || st == 3 || st == 6 || st == 7) return st;
    compute_primal_dual(S);
    return -1;
}
/* (on by default since 8 November 2026, M = 1e5; BRISK_SXPH1BIGM=0: the phase 1 by the auxiliary problem only). Judged on the
 * default route, 4 threads, interleaved: 23 problem LPs 0.983, 40 development LPs 0.994, 55 tall LPs 0.984, 45 fresh LPs of the
 * outside benchmark's families 1.001, 49 held-out relaxations 0.999, its 42 reproduced cases 0.995 (momentum1 0.82 -> 0.55 s,
 * neos-3209462-rhin 1.07 -> 0.60 s; neos-799711 0.79 -> 1.18 s, neos-4984115-canche 1.59 -> 1.93 s); the dual simplex method alone,
 * one thread, 74 LPs: the 24 it changes 0.895 of the time (ns1690781 0.64 -> 0.17 s) */
static int ph1bigm(void) { static double v = -1; if (v < 0) { const char *e_ = getenv("BRISK_SXPH1BIGM"); v = e_ ? atof(e_) : 1e5; } return v > 0; }
static double ph1bigm_m(void) { const char *e_ = getenv("BRISK_SXPH1BIGM"); return e_ ? atof(e_) : 1e5; }
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
    if (getenv("BRISK_SXDBGP1")) { int c_ = 0; for (int j = 0; j < S->N; j++) { if (S->pos[j] >= 0 || is_boxed(S, j) || S->lo[j] == S->up[j]) continue; const double dj = S->d[j];
        const int bad_ = (S->lo[j] <= -SX_INF && S->up[j] >= SX_INF) ? fabs(dj) > S->told : S->lo[j] > -SX_INF ? dj < -S->told : dj > S->told;
        if (bad_ && c_++ < 5) fprintf(stderr, "[p1] j %d (%s) d %.3g cost %.3g lo %.3g up %.3g\n", j, j < S->n ? "column" : "logical", dj, S->cost[j], S->lo[j], S->up[j]); }
        fprintf(stderr, "[p1] need1 %d, %d dual infeasible\n", need1, c_); }
    /* (the artificial bounds put primal values of the order of M into the basis; when the start is primal feasible as it is,
     * they would make phase 2 remove what the phase 1 by the auxiliary problem leaves alone: gfd-schedulen180f7d50m30k18 (LP),
     * whose dual form has all right-hand sides zero, 5.0 -> 32.7 s. Then the auxiliary problem, as before) */
    int pfeas0 = 1;
    if (need1 && ph1bigm() && !S->it_given) {
        for (int i = 0; i < S->m && pfeas0; i++) { const int j = S->basic[i]; const double v = S->x[j];
            if (v < S->lo0[j] - S->tolp * (1 + fabs(S->lo0[j])) || v > S->up0[j] + S->tolp * (1 + fabs(S->up0[j]))) pfeas0 = 0; }
        if (S->verbose >= 2 && pfeas0) printf("  dual phase 1: the start is primal feasible; by the auxiliary problem\n");
    }
    if (need1 && ph1bigm() && !S->it_given && !pfeas0) {
        double M_ = ph1bigm_m();
        {   /* (12 November 2026, an experiment: BRISK_SXPH1BIGMAUTO=k) the artificial bounds at the scale of the problem's own
             * right-hand sides: k times the median of the rows' finite nonzero bounds (scaled), so that the infeasibilities they
             * put into the basic values are of the size of the real ones (genus-sym-grafo5708-48, LP: M = 1e5 3,063 iterations on
             * average, M = 1 2,576, M = 0.1 1,887; a fixed small M loses elsewhere, REPORT_lp_r16) */
            static double ba_ = -1; if (ba_ < 0) { const char *e_ = getenv("BRISK_SXPH1BIGMAUTO"); ba_ = e_ ? atof(e_) : 0; }
            if (ba_ > 0) {
                double *v_ = xmalloc(sizeof(double) * (2 * (size_t)S->m + 1)); long nv_ = 0;
                for (int i = 0; i < S->m; i++) { const int j = S->n + i; const double l = S->lo0[j], u = S->up0[j];
                    if (l > -SX_INF && l != 0) v_[nv_++] = fabs(l); if (u < SX_INF && u != 0 && u != l) v_[nv_++] = fabs(u); }
                if (nv_ > 0) { qsort(v_, (size_t)nv_, sizeof(double), cmp_dbl_asc); M_ = ba_ * v_[nv_ / 2]; }
                free(v_);
                if (S->verbose >= 2) printf("  dual phase 1: artificial bounds at %.3g (%ld finite nonzero row bounds)\n", M_, nv_);
            }
        }
        const int stb = dual_bigm(S, M_);
        if (stb >= 0) return stb;
        need1 = 1;
    }
    if (need1) {
        /* (13 November 2026, the LP developer; with BRISK_SXLAZYSTALL) the stall rule counts phase 2 only, so a solve whose costs
         * wait for the first stall ran the auxiliary problem unperturbed (ns1904248, LP: 83,718 iterations of it against the
         * default's 4,171 on perturbed costs); the rule extended to phase 1 perturbs in the middle of it and was worse (past 60 s).
         * The auxiliary problem gets the default's perturbation from its start instead, the late one being for phase 2 */
        if (S->lazy_main && lazystall() && !S->perturbed && getenv("BRISK_SXLAZYPH1") == NULL) { perturb_costs(S); compute_dual(S); }
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
        /* (4 November 2026) from a given basis, a boxed column whose reduced cost is of the right sign within the dual
         * tolerance keeps its bound: placed by the sign alone, a reduced cost of +1e-12 at the upper bound moved it to the
         * lower one, which bought nothing in dual feasibility and moved the basic variables (satellites2-25, LP: the whole
         * problem after the row sifting's rounds began with 462 infeasible rows where 66 rows were violated, 14 such flips;
         * then 4,597 iterations, now 2,974). BRISK_SXKEEPBOUND=0: by the sign alone */
        static int keepb = -1; if (keepb < 0) { const char *e_ = getenv("BRISK_SXKEEPBOUND"); keepb = !(e_ && atoi(e_) == 0); }
        for (int j = 0; j < S->N; j++) {
            if (S->pos[j] >= 0) continue;
            if (keepb && S->it_given && is_boxed(S, j) && S->lo[j] != S->up[j]
                && ((S->move[j] > 0 && S->d[j] >= -S->told) || (S->move[j] < 0 && S->d[j] <= S->told))) continue;
            set_nonbasic(S, j, S->d[j]);
        }
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
/* (2 November 2026) ord[0..k) sorted by key[ord[]] decreasing, ties by increasing index (a fixed result for a given input) */
static _Thread_local const double *g_sx_key;   /* (per thread: LP objects run on several threads at once) */
static int sx_cmp_key_desc(const void *a, const void *b) {
    const int i = *(const int *)a, j = *(const int *)b; const double ki = g_sx_key[i], kj = g_sx_key[j];
    return ki > kj ? -1 : ki < kj ? 1 : (i > j) - (i < j);
}
static void sx_sort_by_key_desc(int *ord, int k, const double *key) { g_sx_key = key; qsort(ord, (size_t)k, sizeof(int), sx_cmp_key_desc); g_sx_key = NULL; }
static double basis_cond(Sx *S);
static void xo_dbg(Sx *S, const char *when, const double *xs) {
    if (!S->dbg) return;
    double mi = 0, md = 0; int ni = 0;
    for (int i = 0; i < S->m; i++) { const double e = infeas(S, i); if (e > S->tolp) { ni++; if (e > mi) mi = e; } }
    if (xs) for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; const double d = fabs(S->x[j] - xs[j]); if (d > md) md = d; }
    printf("  crossover %s: %d basic infeasible (largest %.1e), largest change of a basic value %.1e\n", when, ni, mi, md);
    if (S->dbg > 1) printf("  crossover %s: condition estimate %.1e\n", when, basis_cond(S));
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
/* (27 October 2026) the crossover's pushes stop at the time limit or at a stop request (ISSUES 460): their loops looked at
 * neither, and savsched1's (LP, a basis of 288,983 rows) ran minutes past the limit; the simplex loop after them then
 * returns the limit's status */
static int xo_stopped(const Sx *S) { return (brisk_flag_get(S->stop)) || (S->tlim > 0 && now() - S->t0 > S->tlim); }
static long crossover_push(Sx *S, const double *x0, const double *y0, long *ndpush) {
    unsigned xo_chk = 0; int xo_stop = 0;
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
    /* (2 November 2026) the inside structurals by decreasing distance to their nearer bound over their reduced cost (the
     * measure of how clearly a variable is basic at an interior point: Megiddo; Andersen and Ye), not by index: when more
     * are inside than there are rows, the index decided which ones were left out (zeil, LP: 106,549 inside for 81,558 rows;
     * the start basis's condition estimate 4.8e3 as written, 9.9e6 on a permuted copy, whose pushes then drifted and failed).
     * Ties keep the index order. Measured and left off (BRISK_SXXOSORT=1: on): the start basis came out worse conditioned,
     * 4.8e3 -> 1.1e7 on zeil as written (43.7 -> 51.4 s); the factorization's pivots, not the candidates' order, decide it */
    static int xosort = -1; if (xosort < 0) { const char *e = getenv("BRISK_SXXOSORT"); xosort = e && atoi(e) != 0; }   /* (off: measured worse, TRICKS 8) */
    int nin_s = 0; for (int j = 0; j < n; j++) nin_s += in[j];
    if (xosort && nb + nin_s > m) {
        int *ord = xmalloc(sizeof(int) * ((size_t)nin_s + 1)); double *key = xmalloc(sizeof(double) * ((size_t)n + 1)); int k = 0;
        for (int j = 0; j < n; j++) if (in[j]) {
            const double d = fmin(S->lo[j] > -SX_INF ? (xs[j] - S->lo[j]) / (1.0 + fabs(S->lo[j])) : SX_INF,
                                  S->up[j] < SX_INF ? (S->up[j] - xs[j]) / (1.0 + fabs(S->up[j])) : SX_INF);
            key[j] = d / (fabs(dz ? dz[j] : 0.0) + 1e-12); ord[k++] = j;
        }
        sx_sort_by_key_desc(ord, k, key);
        for (int t = 0; t < k && nb < m; t++) { const int j = ord[t]; S->basic[nb] = j; S->pos[j] = nb++; }
        free(ord); free(key);
    } else
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
    if (S->dbg && dz) { int nzc = 0, nzl = 0; for (int j = 0; j < N; j++) if (!in[j] && fabs(dz[j]) <= S->told) { if (j < n) nzc++; else nzl++; }
        printf("  crossover start: %d columns and %d logicals at a bound with a zero reduced cost; %d in the basis list before the logicals\n", nzc, nzl, nb); }
    for (int i = 0; i < m && nb < m; i++) if (S->pos[n + i] < 0) { S->basic[nb] = n + i; S->pos[n + i] = nb++; }
    for (int i = 0; i < m; i++) S->w[i] = 1.0;
    memset(S->wk, 1, (size_t)m); S->wlazy = 0;
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
    /* (2 November 2026, an experiment: BRISK_SXXOPIVTHR, the pivot threshold of the start basis's factorization) */
    const double pthr0 = S->F.pivthr;
    { const char *e = getenv("BRISK_SXXOPIVTHR"); if (e) S->F.pivthr = atof(e); }
    refactor(S);
    S->F.pivthr = pthr0;
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
            if ((brisk_flag_get(S->stop)) || ((++xo_chk & 63) == 0 && xo_stopped(S))) { xo_stop = 1; break; }   /* (27 October 2026, the night: the caller's flag at every push; a push costs up to 0.1 ms, 64 of them delayed the default route's stop of the dual simplex method by up to 8 ms) */
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
    /* (2 November 2026, an experiment: BRISK_SXXOPUSHORD=1) the superbasic variables pushed in an order fixed by the problem,
     * by increasing relative distance to the bound they are pushed to, instead of by index (zeil, LP: a permuted copy's
     * pushes drifted where the original's did not) */
    static int pord = -1; if (pord < 0) { const char *e = getenv("BRISK_SXXOPUSHORD"); pord = e ? atoi(e) : 0; }
    static int snapfix = -1; if (snapfix < 0) { const char *e = getenv("BRISK_SXXOSNAP"); snapfix = e && atoi(e) != 0; }   /* (off: measured worse, TRICKS 8) */
    long nsnap = 0;
    /* (7 November 2026, BRISK_SXXOSHIFT, on by default since that day; 0: off) a leaving variable beyond its bound leaves at its value, not set onto the
     * bound: the basic values stay the solution of the equations, and the variable, now nonbasic off its bound, is pushed
     * onto it like any other superbasic one, through the ratio test (which pivots when a basic variable reaches a bound
     * first). Set onto its bound without moving the basic values, as before, the basic values were no longer
     * B^-1 (b - N x_N): the error, its distance beyond the bound, came back at the next factorization multiplied by the
     * new B^-1 a of the leaving column, whose entry in its row is one over the pivot (neos-4264598-oueme, LP, from the
     * classes' solution: one leaving variable 0.14 beyond its bound at a pivot of 2e-4; updated basic infeasibilities at
     * most 0.14, recomputed 3.7e2, then 6,677 iterations of the dual method; with it 0.14 and 1,370). The basis itself
     * was not ill-conditioned (condition estimate 4.8e6 in both, no pivot replaced). Those still off their bound after
     * the passes are set onto it, the caller recomputing the basic values. */
    static int xshift = -1; if (xshift < 0) { const char *e = getenv("BRISK_SXXOSHIFT"); xshift = !(e && atoi(e) == 0); }   /* (on by default since 7 November 2026: the default route, 4 threads, the three fixes of that day together: 23 problem LPs 0.848, 40 development LPs 0.943, 80 rescaled copies of 40 held-out LPs 0.972, no status changed); BRISK_SXXOSHIFT=0: off) */
    long nsnapd = 0, nshift = 0; double snapmax = 0, snapamp = 0;   /* (the leaving variables beyond their bound: counts for SXDBG) */
    char *shf = xcalloc((size_t)N + 1, 1);
    int *pperm = xmalloc(sizeof(int) * ((size_t)N + 1)); double *pkey = pord ? xmalloc(sizeof(double) * ((size_t)N + 1)) : NULL;
    /* (14 November 2026, a test: BRISK_SXXOSNAPALL=1; ISSUES 884) every superbasic variable with a finite bound set onto its nearer bound
     * at once, the basic values recomputed once, and the dual or the primal method after the pushes repairs what that leaves; only the
     * free superbasic variables are pushed. The primal push solves with each superbasic column: highschool1-aigio (LP) 70,517 of them
     * at about 1 ms (the forward solves on a basis of 72,004 rows) */
    {   static int sall = -1; if (sall < 0) { const char *e_ = getenv("BRISK_SXXOSNAPALL"); sall = e_ && atoi(e_) == 1; }
        if (sall) {
            long nset = 0;
            for (int q = 0; q < N; q++) {
                if (S->pos[q] >= 0) continue;
                const double l = S->lo[q], u = S->up[q], v = S->x[q];
                if (v == l || v == u || (l <= -SX_INF && u >= SX_INF)) continue;
                const double t = l <= -SX_INF ? u : u >= SX_INF ? l : (v - l <= u - v ? l : u);
                S->x[q] = t; S->move[q] = t == l ? 1 : -1; nset++;
            }
            compute_primal(S);
            if (S->verbose >= 2) { int ni = 0; double mi = 0; for (int i = 0; i < m; i++) { const double f = infeas(S, i); if (f > 0) { ni++; mi = fmax(mi, f); } }
                printf("  crossover: %ld superbasic variables set onto their nearer bound at once: %d basic infeasible (largest %.1e)\n", nset, ni, mi); }
        }
    }
    for (int pass = 0, left = 1; pass < 5 && left && !xo_stop; pass++) {
    for (int j = 0; j < N; j++) pperm[j] = j;
    if (pord) {
        for (int j = 0; j < N; j++) {
            const double l = S->lo[j], u = S->up[j], v = S->x[j];
            const double t = l <= -SX_INF && u >= SX_INF ? 0 : l <= -SX_INF ? u : u >= SX_INF ? l : (v - l <= u - v ? l : u);
            pkey[j] = -fabs(t - v) / (1.0 + fabs(t));       /* (sorted decreasing: the nearest first) */
        }
        sx_sort_by_key_desc(pperm, N, pkey);
    }
    for (int qi = (left = 0); qi < N; qi++) {
        const int q = pperm[qi];
        if (S->pos[q] >= 0) continue;
        const double l = S->lo[q], u = S->up[q], v = S->x[q];
        if (v == l || v == u || (l <= -SX_INF && u >= SX_INF && v == 0)) continue;
        left = 1;
        if ((brisk_flag_get(S->stop)) || ((++xo_chk & 63) == 0 && xo_stopped(S))) { xo_stop = 1; break; }
        if (need_refactor(S)) {
            double mi0 = 0; if (S->dbg > 1) for (int i = 0; i < m; i++) mi0 = fmax(mi0, infeas(S, i));
            const int nout = refactor(S); compute_primal(S);
            if (S->dbg > 1) { double mi = 0; for (int i = 0; i < m; i++) mi = fmax(mi, infeas(S, i)); if (mi > 10 * mi0 + 1e-6) printf("  refactorization in the primal push: %d replaced, basic infeasibility %.1e -> %.1e, condition estimate %.1e\n", nout, mi0, mi, basis_cond(S)); }
        }
        if (xbatch && qi > single_until) {
            int nbq = 0, bqi_last = qi;
            sxv_clear(col);
            for (int q2i = qi; q2i < N && nbq < bsz; q2i++) {
                const int q2 = pperm[q2i];
                if (S->pos[q2] >= 0) continue;
                const double l2 = S->lo[q2], u2 = S->up[q2], v2 = S->x[q2];
                if (v2 == l2 || v2 == u2 || (l2 <= -SX_INF && u2 >= SX_INF && v2 == 0)) continue;
                const double t2 = l2 <= -SX_INF && u2 >= SX_INF ? 0 : l2 <= -SX_INF ? u2 : u2 >= SX_INF ? l2 : (v2 - l2 <= u2 - v2 ? l2 : u2);
                col_scatter(S, q2, col, t2 - v2);
                bq[nbq] = q2; bt[nbq] = t2; bqi_last = q2i; nbq++;
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
                  if (xretry && bsz > 2) { bsz = bsz / 4 < 2 ? 2 : bsz / 4; qi--; continue; } }
                single_until = bqi_last;
                if (bsz <= 2) { int c = 0; for (int q2i = qi; q2i < N && c < 64; q2i++) if (S->pos[pperm[q2i]] < 0) { single_until = q2i; c++; } }
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
            if (mi > 10 * mimax) { printf("  push %ld (variable %d, %s, |alpha_r| %.1e of the column's largest %.1e, step %.1e, dist %.1e, nupd %d, |col| %d): basic infeasibility %.1e\n", npush, q, r < 0 ? "to its bound" : "pivot", r < 0 ? 0 : fabs(col->val[r]), amax, step, dist, S->F.nupd, col->cnt, mi); mimax = mi; }
        }
        if (r < 0) {
            S->x[q] = target;
            S->move[q] = l == u ? 0 : target == l && l > -SX_INF ? 1 : target == u && u < SX_INF ? -1 : 0;
            continue;
        }
        /* the basic variable of row r reaches its bound first: it leaves there, q enters */
        const int jl = S->basic[r]; const double ar = col->val[r];
        const double bound = dir * ar > 0 ? S->lo[jl] : S->up[jl];
        const double snap = bound - S->x[jl];      /* (the leaving variable's move onto its bound: nonzero when it was beyond it) */
        const int shifted = xshift && fabs(snap) > S->tolp;
        if (fabs(snap) > S->tolp) { nsnapd++; snapmax = fmax(snapmax, fabs(snap)); snapamp = fmax(snapamp, fabs(snap) / fabs(ar)); }
        S->x[q] += step; if (!shifted) S->x[jl] = bound; else nshift++;
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
        S->move[jl] = shifted ? 0 : S->lo[jl] == S->up[jl] ? 0 : bound == S->lo[jl] ? 1 : -1;
        if (shifted) shf[jl] = 1;
        rows_enter(S, q); rows_leave(S, jl);
        if (sxlu_update(&S->F, r, &S->spike, &S->upart, ar)) { refactor(S); compute_primal(S); }
        else if (!shifted && snapfix && fabs(snap) > S->tolp) {
            /* (2 November 2026) the leaving variable was beyond its bound and is now nonbasic on it: the basic values follow,
             * x_B -= snap B^-1 a_jl on the new basis, so that they stay the solution of the equations. Without it the
             * difference waited for the next refactorization, which recomputed it multiplied by the basis's condition (zeil's
             * permuted copy, LP: 2.3e-2 -> 1.1 -> 33 -> 370 -> 1.5e5, the pushes then failed). BRISK_SXXOSNAP=0: not */
            sxv_clear(col); col_scatter(S, jl, col, 1.0);
            double dd = S->dens_col; ftran(S, col, &dd, NULL);
            if (col->cnt < 0) sxv_reindex(col);
            for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= snap * col->val[i]; }
            nsnap++;
        }
        sxv_clear(rho);
    }
    }
    if (S->dbg && snapfix) printf("  crossover: %ld leaving variables beyond their bound corrected in the primal push\n", nsnap);
    long nshleft = 0;
    if (xshift) for (int j = 0; j < N; j++) {     /* (the ones still off their bound: onto the nearer one; the caller recomputes the basic values) */
        if (!shf[j] || S->pos[j] >= 0) continue;
        const double l = S->lo[j], u = S->up[j], v = S->x[j];
        if (v == l || v == u || (l <= -SX_INF && u >= SX_INF && v == 0)) continue;
        const double t = l <= -SX_INF && u >= SX_INF ? 0 : l <= -SX_INF ? u : u >= SX_INF ? l : (v - l <= u - v ? l : u);
        S->x[j] = t; S->move[j] = l == u ? 0 : t == l && l > -SX_INF ? 1 : t == u && u < SX_INF ? -1 : 0; nshleft++;
    }
    if (S->dbg) printf("  crossover: %ld leaving variables beyond their bound (largest %.1e, largest over the pivot %.1e); %ld left at their value and pushed, %ld of them set onto their bound after the passes\n", nsnapd, snapmax, snapamp, nshift, nshleft);
    free(shf);
    free(pperm); free(pkey);
    sxv_clear(col); sxv_clear(rho);
    free(bq); free(bt);
    if (S->dbg && xbatch) printf("  crossover: %d batches of pushes took %d variables\n", nbat, nbatq);
    xo_dbg(S, "after the primal push", xs);
    if (S->dbg) { int ns = 0; for (int q = 0; q < N; q++) if (S->pos[q] < 0 && S->x[q] != S->lo[q] && S->x[q] != S->up[q] && !(S->lo[q] <= -SX_INF && S->up[q] >= SX_INF && S->x[q] == 0)) ns++; printf("  crossover: %d superbasic left, %d replaced by the factorizations\n", ns, S->F.rank_deficient); }
    S->keepx = 0;
    free(xs); free(in); free(at);
    return npush;
}

/* (5 November 2026, a diagnostic: BRISK_SXDIAGEND=1) the density of the basis' inverse, by the entries of every seventh of
 * its rows (split by rows whose basic variable is a column or a logical), and the columns basic (those strictly inside
 * their bounds apart), at the upper and at the lower bound; printed at a given start basis, after the crossover's pushes and
 * at the end of the solve */
static void sx_diag_density(Sx *S, const char *lab) {
    SxVec *rho = &S->rho; double sr = 0, srl = 0, srs = 0; int nl = 0, ns = 0, nup = 0, nlo = 0, nbs = 0, nfr = 0;
    for (int i = 0; i < S->m; i += 7) {
        sxv_clear(rho); rho->val[i] = 1; rho->idx[0] = i; rho->cnt = 1;
        double dd = S->dens_rho; btran(S, rho, &dd, NULL);
        if (rho->cnt < 0) sxv_reindex(rho);
        const int c = rho->cnt; sr += c;
        if (S->basic[i] < S->n) { srs += c; ns++; } else { srl += c; nl++; }
    }
    sxv_clear(rho);
    for (int j = 0; j < S->n; j++) { if (S->pos[j] >= 0) { nbs++; if (S->x[j] > S->lo[j] + 1e-7 && S->x[j] < S->up[j] - 1e-7) nfr++; } else if (S->move[j] < 0) nup++; else nlo++; }
    printf("    diag %s: |rho| %.1f (rows with a basic column %.1f, %d; with a basic logical %.1f, %d); columns basic %d (strictly inside %d), at upper %d, at lower %d\n",
           lab, sr / (nl + ns), ns ? srs / ns : 0.0, ns, nl ? srl / nl : 0.0, nl, nbs, nfr, nup, nlo);
}
static int sx_solve_core(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                         SimplexInfo *info) {
    SimplexInfo dummy; if (!info) info = &dummy;
    memset(info, 0, sizeof *info);
    if (o && brisk_flag_get(o->stop)) { info->status = 6; return 6; }    /* (the caller's flag set: no set-up) */
    Sx S; memset(&S, 0, sizeof S);
    S.t0 = now();
    g_sx_taboff = L->c0; g_sx_tabsign = L->maxim ? -1.0 : 1.0;   /* (the default log's table: the objective as given) */
    S.rayout = o ? o->ray : NULL;
    S.tlim = o ? o->timelimit : 0; S.maxit = o ? o->maxit : 0; S.verbose = o ? o->verbose : 0; S.stop = o ? o->stop : NULL; S.threads = o ? o->threads : NULL; S.dbg = getenv("SXDBG") ? atoi(getenv("SXDBG")) : 0; S.xo_deg_only = getenv("BRISK_SXXODEG") && atoi(getenv("BRISK_SXXODEG")) >= 1; if (S.dbg < 0) S.dbg = 1; S.tdiag = S.verbose >= 2;
    S.workbudget = o ? o->workbudget : 0; S.budget_cb = o ? o->budget_cb : NULL; S.budget_ctx = o ? o->budget_ctx : NULL;
    const int method = o ? o->method : 0;
    sx_setup(&S, L, o);
    const double t_setup = now() - S.t0;
    if (S.verbose >= 2 && S.ntiny) printf("  %ld matrix entries too small to matter dropped (below 1e-9 of their row's largest; each row's activity moved by at most 1e-10)\n", S.ntiny);
    for (int j = 0; j < S.N; j++) if (S.lo0[j] > S.up0[j] + 1e-9 * (1 + fabs(S.lo0[j]))) { info->status = 1; snprintf(info->why, sizeof info->why, "a variable or row has lower bound above upper bound"); sx_free(&S); return 1; }
    int st;
    if (o && o->cstat0 && o->rstat0) {
        /* a starting basis: if it is optimal (or nearly), the method that keeps its feasible side finishes */
        given_basis(&S, o->cstat0, o->rstat0);
        refactor(&S); compute_primal(&S); compute_dual(&S);
        S.it_given = 1;
        /* (4 November 2026) the rounds of the row sifting: the dual steepest-edge weights of the given basis computed
         * lazily (a row's exact weight when it first comes to the top of the leaving rows), as the re-solves of the
         * mixed-integer solver do (sxm_set_basis), instead of the slack basis's 1. BRISK_SXRSLAZYW=0: the weights 1 */
        static int rslw = -1; if (rslw < 0) { const char *e_ = getenv("BRISK_SXRSLAZYW"); rslw = !(e_ && atoi(e_) == 0); }
        /* (13 November 2026, an experiment: BRISK_SXGIVENLAZYW=1) the same for any given basis (BRISK_SXBASISFILE included) */
        static int glw = -1; if (glw < 0) { const char *e_ = getenv("BRISK_SXGIVENLAZYW"); glw = e_ && atoi(e_) == 1; }
        if (rslw && (o->given_perturb || glw)) {
            memset(S.wk, 0, (size_t)S.m); S.wlazy = 1;
            for (int i = 0; i < S.m; i++) {
                const int j = S.basic[i];
                if (j >= S.n) continue;
                double c2 = 0; for (int p = S.Ap[j]; p < S.Ap[j + 1]; p++) c2 += S.Ax[p] * S.Ax[p];
                S.w[i] = c2 > 1 ? fmax(1.0 / c2, 1e-4) : 1.0;
            }
        }
        /* (13 November 2026, an experiment: BRISK_SXGIVENEXACTW=1) the exact dual steepest-edge weights of the given basis, one
         * backward solve a row, at once (=2: the weights 1 for every row) */
        {   static int gew = -1; if (gew < 0) { const char *e_ = getenv("BRISK_SXGIVENEXACTW"); gew = e_ ? atoi(e_) : 0; }
            if (gew == 1) { SxVec *rh_ = &S.rho;
                for (int i = 0; i < S.m; i++) { sxv_clear(rh_); rh_->val[i] = 1; rh_->idx[0] = i; rh_->cnt = 1;
                    double dd = S.dens_rho; btran(&S, rh_, &dd, NULL); double w = 0;
                    if (rh_->cnt >= 0) for (int t = 0; t < rh_->cnt; t++) { const double v = rh_->val[rh_->idx[t]]; w += v * v; } else for (int k = 0; k < S.m; k++) w += rh_->val[k] * rh_->val[k];
                    S.w[i] = w > 1e-4 ? w : 1e-4; }
                sxv_clear(rh_); memset(S.wk, 1, (size_t)S.m); S.wlazy = 0; }
            else if (gew == 2) { for (int i = 0; i < S.m; i++) S.w[i] = 1.0; memset(S.wk, 1, (size_t)S.m); S.wlazy = 0; } }
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
        /* (14 November 2026, an experiment: BRISK_SXGIVENSHIFT=k; ISSUES 883) a given basis whose dual infeasibilities are all within k
         * times the dual tolerance (a basis carried back by the postsolve from an optimal basis, off by its rounding): those costs
         * shifted (shift_cost; the final rounds remove the shifts) and the dual method's phase 2 from it, where it was treated as a
         * start neither primal nor dual feasible (the costs perturbed, the dual method's phase 1). neos-4264598-oueme's root LP (LP,
         * the crossover's perturbation of BRISK_SXXOPERTTOL=2): 16 dual infeasibilities of at most 1.1e-7, 14 primal of at most 5.9e-7,
         * then a phase 1 from 20,122 primal infeasibilities and 31,340 iterations (11.5 s) */
        {   static double gsh_ = -1; if (gsh_ < 0) { const char *e_ = getenv("BRISK_SXGIVENSHIFT"); gsh_ = e_ ? atof(e_) : 0; }
            if (gsh_ > 0 && given_route == 0 && !(o && o->given_perturb)) {
                int big_ = 0;
                for (int j = 0; j < S.N && !big_; j++) {
                    if (S.pos[j] >= 0) continue;
                    const double e_ = S.move[j] == 0 ? (S.lo[j] != S.up[j] ? fabs(S.d[j]) : 0) : -S.move[j] * S.d[j];
                    if (e_ > gsh_ * S.told) big_ = 1;
                }
                if (!big_) {
                    int nsh_ = 0;
                    for (int j = 0; j < S.N; j++) {
                        if (S.pos[j] >= 0) continue;
                        const double e_ = S.move[j] == 0 ? (S.lo[j] != S.up[j] ? fabs(S.d[j]) : 0) : -S.move[j] * S.d[j];
                        if (e_ > S.told) { shift_cost(&S, j); nsh_++; }
                    }
                    given_route = 3;
                    if (S.verbose >= 2) printf("  starting basis: %d dual infeasibilities within %g times the tolerance, their costs shifted: dual feasible\n", nsh_, gsh_);
                }
            }
        }
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
    if (getenv("BRISK_SXDIAGEND") && S.it_given) sx_diag_density(&S, "given start");
    const int xo = o && o->xstart && !S.it_given;
    int xo_dual = 0;
    if (xo) {
        long nd = 0;
        /* (14 November 2026, ISSUES 884) the crossover's factorizations stop at the time limit or at a stop request (the
         * columns left replaced by logicals): its pushes check them only between pushes, and one factorization of the start
         * basis outlasted the limit by over 70 s (highschool1-aigio, LP) */
        S.F.stopf = S.stop; S.F.deadline = S.tlim > 0 ? S.t0 + S.tlim : 0;
        if (getenv("BRISK_SXXOTLDBG")) fprintf(stderr, "[xo] the crossover: time limit %.3f s, stop flag %s\n", S.tlim, S.stop ? "set" : "none");
        const long np = crossover_push(&S, o->xstart, o->ystart, &nd);
        S.F.stopf = NULL; S.F.deadline = 0;
        if (S.verbose >= 1) printf("  crossover: %ld primal and %ld dual pushes (%.2f s; %d factorizations, %.2f s)\n", np, nd, now() - S.t0, S.nfactor, S.t_factor);
        info->pushes = np + nd;
        refactor(&S); compute_primal(&S); compute_dual(&S);
        if (getenv("BRISK_SXDIAGEND")) sx_diag_density(&S, "crossover start");
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
    else if (xo_dual && !(o && o->given_perturb)) { S.phase = 2; if (getenv("BRISK_SXXOLAZY")) S.lazy_pert = 1; if (getenv("BRISK_SXXOPERTTOL")) S.xo_pert_tol = atoi(getenv("BRISK_SXXOPERTTOL")); if (getenv("BRISK_SXXODEG") && atoi(getenv("BRISK_SXXODEG")) >= 1) S.xo_deg = atoi(getenv("BRISK_SXXODEG")); st = dual_loop(&S); S.xo_deg = 0; }   /* (10 November 2026, a test: BRISK_SXXOLAZY, the stall rule's perturbation for the crossover's dual method, ISSUES 862) */
    else if (given_route == 1) st = 0;
    else if (given_route == 2) st = primal_simplex(&S, pert);
    else if (given_route == 3 && !(o && o->given_perturb)) { S.phase = 2; st = dual_loop(&S); }
    else if (method == 2 && !xo_dual) {
        if (!S.it_given && !xo) {
            /* (4 November 2026) the primal method from the logicals' basis: a boxed column starts at its bound nearer zero
             * (the lower one on a tie), not at the bound its cost favours: that is the dual method's start (dual
             * feasibility), and for the primal method it makes rows infeasible that the other bound leaves feasible.
             * neos-3065804-namu (LP): 2,555 columns in [0,1] with negative costs put at 1 violated 1,458 rows, a phase 1
             * of 39,998 iterations (62,554 in all), where x = 0 is feasible (Gurobi's primal method: 123 iterations).
             * BRISK_SXPSTART=0: by the cost */
            static int pstart = -1; if (pstart < 0) { const char *e_ = getenv("BRISK_SXPSTART"); pstart = e_ ? atoi(e_) : 3; }
            if (pstart) {
                for (int j = 0; j < S.n; j++) {
                    if (S.pos[j] >= 0) continue;
                    const double l = S.lo[j], u = S.up[j];
                    if (l > -SX_INF && u < SX_INF && l != u) {
                        /* (11 November 2026; 3 by default since that day, BRISK_SXPSTART=1 the rule of 4 November, 2 a test) a tie
                         * (|l| = |u|, as for a column in [-1, 1]) at the bound the cost favours (as set before this loop): the
                         * reason for the bound nearer zero, that x = 0 is often feasible, says nothing between two bounds equally
                         * far from zero, and the lower one was taken blindly. Hnear_parallel_3500 (LP, x in [-1, 1]): phase 1 ended
                         * at an objective of about 0 and phase 2 took 39,000 iterations; by the cost it ends near -1,200 (Gurobi's
                         * -1,367), 47,195 -> 19,909 iterations; the 67 LPs 0.962 (scaled 0.78, dense coupling 0.88; thin polytope
                         * +5 %, triangular and degenerate +2-14 % iterations). 2: a bound at zero if there is one, else by the cost */
                        if (pstart == 2) { if (l == 0) { S.move[j] = 1; S.x[j] = l; } else if (u == 0) { S.move[j] = -1; S.x[j] = u; } continue; }
                        if (pstart == 3 && fabs(l) == fabs(u)) continue;
                        if (fabs(l) <= fabs(u)) { S.move[j] = 1; S.x[j] = l; } else { S.move[j] = -1; S.x[j] = u; }
                    }
                }
                compute_primal(&S);
            }
            /* (7 November 2026, on trial: BRISK_SXPSTARTFLIP=1) a boxed column of cost zero at the bound that leaves its rows
             * the smaller sum of violations (the dual method's start rule, one greedy pass): the primal method's phase 1 then
             * starts nearer feasibility with every structural column still at a bound */
            if (getenv("BRISK_SXPSTARTFLIP") && atoi(getenv("BRISK_SXPSTARTFLIP")) > 0) {
                const long nfl = dual_start_bounds_(&S, atoi(getenv("BRISK_SXPSTARTFLIP")) >= 2 ? atoi(getenv("BRISK_SXPSTARTFLIP")) : 0); compute_primal(&S);   /* (11 November 2026, a test: 2 any column, whatever its cost) */
                if (S.verbose >= 2 && nfl) printf("  %ld columns of cost zero start at their other bound (less primal infeasibility)\n", nfl);
            }
            if (S.verbose >= 2 && getenv("BRISK_SXPSTARTDIAG")) {   /* (11 November 2026, a diagnostic) the start's violated rows */
                int nv = 0, nr = 0; for (int i = 0; i < S.m; i++) { const int j = S.basic[i]; if (j < S.n) continue; nr++; if (infeas(&S, i) > 0) nv++; }
                printf("  the start: %d of %d rows violated (%.3f)\n", nv, nr, nr ? (double)nv / nr : 0.0);
            }
            const int ncr = primal_crash(&S);
            if (S.verbose >= 2 && ncr) printf("  %d columns brought into the starting basis by the crash\n", ncr);
        }
        S.pse_route = 1; st = primal_simplex(&S, pert); S.pse_route = 0; S.pse_on = 0;
    } else {
        if (pert && !xo_dual && getenv("BRISK_SXSTARTFLIP")) {     /* (on trial: off until measured on the sets) */
            const long nfl = dual_start_bounds(&S);
            if (S.verbose >= 2 && nfl) printf("  %ld columns of cost zero start at their other bound (less primal infeasibility)\n", nfl);
        }
        if (!S.it_given && !xo) {
            const int nfc = free_crash(&S);
            if (S.verbose >= 2 && nfc) printf("  %d free columns brought into the starting basis (each for the logical of a row of its largest entry)\n", nfc);
            /* (12 November 2026) the start by columns of cost zero not on row sifting's sets: a set leaves most rows out, and the
             * columns the start takes for its equations ignore the rows not yet in it (satellites2-60-fs, LP: its first set's basis
             * then cost the later rounds 3,096 -> 3,808 iterations; nb10tb 47,900 -> 53,998). BRISK_SXEQCRASHNOSIFT=0: on them too */
            static int ecns = -1; if (ecns < 0) { const char *e_ = getenv("BRISK_SXEQCRASHNOSIFT"); ecns = !(e_ && atoi(e_) == 0); }
            const int nec = (ecns && o && o->in_rowsift) ? 0 : eq_crash(&S);
            if (S.verbose >= 1 && nec) printf("  %d free or boxed columns brought into the starting basis for the logicals of equations\n", nec);
            /* (11 November 2026, a test: BRISK_SXDCRASH=1 with BRISK_SXCRASH=1) the primal method's crash before the dual method */
            if (getenv("BRISK_SXDCRASH")) { const int ncr = primal_crash(&S); if (S.verbose >= 1 && ncr) printf("  %d columns brought into the starting basis by the crash (before the dual method)\n", ncr); }
        }
        /* (13 November 2026, an experiment: BRISK_SXLAZYPERT=1) the costs perturbed at the first stall (half of a window's steps of
         * zero length), not at the start: Gurobi's dual simplex method on sat4p1 (LP) had no perturbed cost at its iteration
         * 1,000 and 58,701 by 4,000, and its perturbed costs given to ours keep our bases 30 times sparser */
        static int lzp_ = -1; if (lzp_ < 0) { const char *e_ = getenv("BRISK_SXLAZYPERT"); lzp_ = e_ && atoi(e_) != 0; }
        if (pert && lzp_) S.lazy_pert = S.lazy_main = 1;
        else if (pert) perturb_costs(&S);
        S.ds_scratch = (!S.it_given && !xo) || (o && o->given_perturb);   /* (and the rounds of the row sifting and the other starts solved as from the logicals' basis: given_perturb) */
        {   /* (14 November 2026, a diagnostic: BRISK_SXSTARTDUMP=<file>) the dual method's start, once a process: per row its scale factor
             * and side, per column its entries, its scale factor, its cost in the problem's units, its bounds and the bound it starts at
             * (satellites4-25's copies, LP: the paths part at the first iteration although the presolved problems are the same) */
            static int sd_done_ = 0; const char *sd_ = getenv("BRISK_SXSTARTDUMP");
            if (sd_ && !sd_done_) { sd_done_ = 1; FILE *fp_ = fopen(sd_, "w"); if (fp_) {
                for (int i = 0; i < S.m; i++) fprintf(fp_, "R %d %.17g %.17g %.17g\n", i, S.rscale ? S.rscale[i] : 1.0, S.lo[S.n + i], S.up[S.n + i]);
                for (int j = 0; j < S.n; j++) fprintf(fp_, "C %d %d %.17g %.17g %.17g %.17g %d %d\n", j, S.Ap[j + 1] - S.Ap[j], S.cscale ? S.cscale[j] : 1.0, S.cost0[j] * S.costscale / (S.cscale ? S.cscale[j] : 1.0), S.lo0[j], S.up0[j], S.move[j], S.pos[j] >= 0);
                fclose(fp_); } } }
        st = dual_simplex(&S);
        S.ds_scratch = 0;
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
    const double tfin0 = now();
    if (getenv("BRISK_SXDIAGEND") && S.m > 0) sx_diag_density(&S, "end");
    /* (27 October 2026, the night; ISSUES 603) the rounds are a clean-up: at most as many iterations again as the solve
     * took (10,000 at least), and a result whose violations stay above the target after them is not reported optimal
     * (status 4, numerical difficulties): bley_xs2's dual violation of 7.1e6 against costs of 1e7 had been reported
     * OPTIMAL, and the primal method's clean-up then wandered for millions of iterations */
    const long fin_maxit0 = S.maxit, fin_cap = S.iter + (S.iter > 10000 ? S.iter : 10000);
    if (!(S.maxit > 0 && S.maxit < fin_cap)) S.maxit = fin_cap;
    S.tiny_on = 1; if (S.tiny) memset(S.tiny, 0, (size_t)S.tiny_n);
    const int grbt = (brisk_lpgrbtol() & LPGRB_SXROUNDS) != 0;
    int fin_tgt = 0;     /* (status 4 from these rounds alone: SimplexInfo.fin_target) */
    for (int round = 0; round < 5 && st == 0; round++) {
        /* the costs and bounds of the problem (shifts and perturbations gone), the nonbasic variables at them */
        memcpy(S.cost, S.cost0, sizeof(double) * S.N); memset(S.shift, 0, sizeof(double) * S.N);
        memcpy(S.lo, S.lo0, sizeof(double) * S.N); memcpy(S.up, S.up0, sizeof(double) * S.N);
        normalize_nonbasic(&S);
        if (S.F.nupd > 0 || S.nfactor == 0) refactor(&S);
        compute_primal(&S); compute_dual(&S);
        double pv, dv, psv, dsv;
        if (grbt) {
            /* (7 November 2026; ISSUES 843, BRISK_LPGRBTOL=1) Gurobi's rule: the method's own tolerances on the scaled
             * problem, checked once the shifts and perturbations are gone; the method continues from the basis while they
             * are not met (the primal method for reduced costs of the wrong sign, the dual method for values beyond their
             * bounds), with the same tolerances */
            scaled_violations(&S, &pv, &dv);
            if (S.verbose >= 2) printf("  violations of the scaled problem: primal %.1e, dual %.1e (tolerances %.0e, %.0e)\n", pv, dv, S.tolp, S.told);
            if (pv <= S.tolp && dv <= S.told) break;
            if (round == 4) { st = 4; fin_tgt = 1; if (S.verbose >= 2) printf("  the violations of the scaled problem stay above the tolerances: numerical difficulties\n"); break; }
            st = dv > S.told ? primal_simplex(&S, pert) : dual_loop(&S);
            if (st == 3 && S.maxit == fin_cap && fin_maxit0 != fin_cap) st = 4;
            continue;
        }
        unscaled_violations(&S, S.env_target, S.env_target, &pv, &dv, &psv, &dsv);
        if (S.verbose >= 2 && getenv("BRISK_SXSCVIOL")) { double a_, b_; scaled_violations(&S, &a_, &b_); printf("  violations of the scaled problem: primal %.1e, dual %.1e (tolerances %.0e, %.0e)\n", a_, b_, S.tolp, S.told); }
        if (S.verbose >= 2) printf("  violations of the unscaled problem: primal %.1e, dual %.1e\n", pv, dv);
        { int c_; const double s_ = sum_pinf(&S, &c_); brisk_stab_end(S.iter, "final", g_sx_tabsign * objective(&S) * S.costscale + g_sx_taboff, s_, c_); }
        if (pv <= S.env_target && dv <= S.env_target) break;
        if (round == 4) { st = 4; fin_tgt = 1; if (S.verbose >= 2) printf("  the violations stay above the target %.0e: numerical difficulties\n", S.env_target); break; }
        if (dv > S.env_target) {
            S.told = fmax(fmin(S.told * 0.5 * S.env_target / dv, 0.5 * dsv), 1e-12);
            S.cl_target = S.env_target; st = primal_simplex(&S, pert); S.cl_target = 0;
        } else {
            if (pv > S.env_target) S.tolp = fmax(fmin(S.tolp * 0.5 * S.env_target / pv, 0.5 * psv), 1e-12);
            st = dual_loop(&S);
        }
        /* the clean-up's own limit reached (not the caller's): numerical difficulties, unless the basis it reached meets the
         * target (1 November 2026, ISSUES 815: the primal method's clean-up stops on its own tightened tolerance, not on the
         * target; nb10tb, LP, after the symmetry classes' lifted point: a dual violation of 1.4e-7 against 1e-7, the clean-up
         * to its limit of 10,000 iterations and status 4, where the basis it had reached met the target (the method started
         * again from it: optimal in 0 iterations)) */
        if (st == 3 && S.maxit == fin_cap && fin_maxit0 != fin_cap) {
            memcpy(S.cost, S.cost0, sizeof(double) * S.N); memset(S.shift, 0, sizeof(double) * S.N);
            memcpy(S.lo, S.lo0, sizeof(double) * S.N); memcpy(S.up, S.up0, sizeof(double) * S.N);
            normalize_nonbasic(&S);
            refactor(&S); compute_primal(&S); compute_dual(&S);
            double pv2, dv2, psv2, dsv2;
            unscaled_violations(&S, S.env_target, S.env_target, &pv2, &dv2, &psv2, &dsv2);
            if (pv2 <= S.env_target && dv2 <= S.env_target) { st = 0; if (S.verbose >= 2) printf("  the clean-up reached %ld iterations at a basis within the target (primal %.1e, dual %.1e)\n", fin_cap, pv2, dv2); break; }
            st = 4; if (S.verbose >= 2) printf("  the clean-up reached %ld iterations: numerical difficulties (primal %.1e, dual %.1e)\n", fin_cap, pv2, dv2);
        }
    }
    S.maxit = fin_maxit0; S.tiny_on = 0;
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
    if (S.tdiag) printf("  the solve's set-up %.3f s (scaling, the row copy, the start basis), the final rounds on the unscaled problem and the output %.3f s\n", t_setup, now() - tfin0);
    if (S.tdiag) printf("  the dual iterations' other parts: leaving row %.3f s, pivot row %.3f, ratio test %.3f, dual update and flips %.3f, primal step %.3f, weights %.3f, candidates %.3f, refactorizations with their recomputations %.3f; the dual loop %.3f s; forward solves %.3f, backward solves %.3f\n", S.t_chuzr, S.t_row, S.t_chuzc, S.t_dupd, S.t_pupd, S.t_dse, S.t_heap, S.t_rebuild, S.t_loop, S.t_ftran, S.t_btran);
    if (S.verbose >= 2 && bfe_n) printf("  [early stop of the long step] %ld ratio tests with earlier groups, %ld entering choices changed\n", bfe_n, bfe_chg);
    if (S.verbose >= 2) printf("  pricing: %ld by rows (%ld dense; average |rho| %.1f of %d rows), %ld by columns; average |row| %.1f; average |alpha_q| %.1f\n",
                               S.nprice_row, S.nprice_dense, S.nprice_row ? S.sum_rho / S.nprice_row : 0.0, S.m, S.nprice_col, S.iter ? S.sum_rowc / S.iter : 0.0, S.iter ? S.sum_col / S.iter : 0.0);
    if (S.verbose >= 2 && S.it_primal > 0) printf("  the primal method: %ld iterations (%ld in phase 1), %ld of length zero, %ld bound flips, %d resets of the Devex weights, %ld full scans of the candidates, %d numerical troubles (pivot threshold %.2g; the pivot from the row against the column %ld, the infeasibility's jump at a refactorization %ld, steps back %ld)\n", S.it_primal, S.it_ph1, S.n_pdegen, (long)S.nflips, S.ndvxreset, S.n_prescan, S.nptrouble, S.F.pivthr > 0 ? S.F.pivthr : 0.1, S.n_tr_rowcol, S.n_tr_jump, S.n_tr_lost);
    if (S.verbose >= 2 && S.ds_grp) printf("  the sparsest member among the final group's ties: %ld groups, %ld entering choices changed, %ld eligible members, %ld fresh solves (%.3f a dual iteration), %ld counts kept, %.3f s; of the changed choices %ld outside their box (the replaced %ld), %ld of reduced cost zero, |alpha| %.3f of the largest on average; dual step smaller than the replaced member's %ld times (%.3f of it on average), larger %ld\n",
                               S.ds_grp, S.ds_chg, S.ds_memb, S.ds_solve, S.iter ? (double)S.ds_solve / S.iter : 0.0, S.ds_hit, S.ds_t, S.ds_oob, S.ds_oob0, S.ds_cr0, S.ds_chg ? S.ds_ar / S.ds_chg : 0.0,
                               S.ds_stsm, S.ds_stsm ? S.ds_strat / S.ds_stsm : 0.0, S.ds_stlg);
    if (S.verbose >= 2 && S.ds_nsh) printf("  the sparsest-member rule: %ld entering costs shifted to the ratio test's dual step, the largest shift %.2e (tolerance %.1e); %ld changes refused, the shift beyond the tolerance\n", S.ds_nsh, S.ds_shmax, S.told, S.ds_shref);
    if (S.verbose >= 2 && S.ds_inc) printf("  the sparsest-member rule: %ld changes refused, the member no sparser than the ratio test's\n", S.ds_inc);
    if (S.verbose >= 2 && S.dc_n) { printf("  the estimates against the exact counts over %ld groups (same member, within 1.25, mean log ratio, solves):", S.dc_n);
        for (int e = 0; e < 7; e++) printf(" [%d] %.3f %.3f %.3f %ld", e, (double)S.dc_ag[e] / S.dc_n, (double)S.dc_near[e] / S.dc_n, S.dc_lr[e] / S.dc_n, e == 0 ? S.ds_solve : S.dc_sol[e]); printf("\n"); }
    if (S.verbose >= 2 && S.n_prt) printf("  the stability test of the ratio test: %ld ratio tests repeated, %ld left with a small pivot, %ld bounds moved in phase 1\n", S.n_prt, S.n_prtfail, S.n_prtshift);
    if (S.verbose >= 2 && S.n_ph1pass > 0) printf("  phase 1's ratio test passed %ld points where variables became feasible\n", S.n_ph1pass);
    if (S.verbose >= 2 && S.n_ph1flat > 0) printf("  phase 1: %ld bound flips where the sum of infeasibilities was flat up to the entering variable's other bound\n", S.n_ph1flat);
    if (S.verbose >= 2 && S.it_primal > 0) printf("  the primal method's vectors: average |alpha_q| %.1f, |row| %.1f of %d rows, %d columns; steepest-edge work: solves %.3g, products %.3g (%.3f s); all solves %.3g\n", S.n_alpha / (double)S.it_primal, S.n_row / (double)S.it_primal, S.m, S.N, S.fl_pse, S.fl_prod, S.t_prod, (double)S.F.nflops + S.F.nedges);
    if (S.verbose >= 2 && S.t_pse > 0) printf("  the primal method's steepest-edge pricing: its solves %.3f s (density %.2f; %ld beside the row's solve)\n", S.t_pse, S.dens_pv, S.npar_pse);
    if (S.verbose >= 2 && S.n_taupart + S.n_taufull > 0) printf("  the weights' solve: %ld restricted, %ld full (work lately %.3g and %.3g)\n", S.n_taupart, S.n_taufull, S.tw_part, S.tw_full);
    info->status = st; info->fin_target = st == 4 && fin_tgt;
    info->have_ray = st == 1 && S.have_ray; info->ray_sign = (int)S.raymo;
    info->iters = S.iter; info->iters_dual = S.it_dual; info->iters_primal = S.it_primal; info->iters_phase1 = S.it_ph1;
    info->nfactor = S.nfactor; info->nflips = S.nflips; info->work = S.work;
    info->obj = obj; info->pinf = pinf; info->dinf = dinf;
    if (S.prog) { if (st == 0) { S.prog->obj = obj; S.prog->pinf = pinf; } S.prog->it = S.prog_base + S.iter; }
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
        /* (26 October 2026) each round the time left, not the whole limit (as row sifting) */
        if (o && o->timelimit > 0) { o2.timelimit = o->timelimit - (now() - t0); if (o2.timelimit <= 0) o2.timelimit = 1e-3; }
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
        if (o && o->timelimit > 0 && now() - t0 >= o->timelimit) { st = 6; break; }    /* (the limit passed in this round) */
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
 * has. The rounds after the first perturb the costs as a solve from the slack basis does (given_perturb), each boxed column
 * towards the bound it is at, and compute the dual steepest-edge weights of their given basis lazily. When the set's
 * objective has not moved over two rounds and the violated rows have not fallen, the whole problem from the basis reached
 * (4 November 2026). */
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
    int st = 0, round = 0, havebasis = 0, all = nin == m, nvh[64], scratch = 0, unbset = 0, objstall = 0; double obh[64]; long its = 0; double work = 0;
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
        o2.nopresolve = 1; o2.ray = NULL; o2.in_rowsift = 1;
        /* (26 October 2026) each round the time left, not the whole limit: highschool1-aigio (LP) with 60 s ran 107 s, the
         * last round on the whole problem given the full limit after 47 s of the others */
        if (o && o->timelimit > 0) { o2.timelimit = o->timelimit - (now() - t0); if (o2.timelimit <= 0) o2.timelimit = 1e-3; }
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
        /* (12 November 2026) a set whose solve ended at a basis optimal by the method's own tolerances but was reported with
         * numerical difficulties by the final check on the problem as given (violations above the target that the scaled
         * method cannot see: neos-799716, LP, column scale factors of 6e-8, a scaled reduced cost of 1e-14 is 2e-7 as given)
         * is an intermediate problem: the rows it violates join and the rounds go on from its basis. Only when it violates
         * none, its result would be the final one, and the whole problem is solved from the start as before. Before, the
         * rounds were given up: neos-799716's sixth round, after the set had reached the optimal value in its third, and the
         * whole problem from the start, 1,091 iterations of the rounds lost. BRISK_SXROWSIFTFIN=0: as before */
        int fin4 = 0;
        {   static int rsf = -1; if (rsf < 0) { const char *e_ = getenv("BRISK_SXROWSIFTFIN"); rsf = !(e_ && atoi(e_) == 0); }
            if (rsf && !all && st == 4 && is.fin_target) { st = 0; fin4 = 1; } }
        info->iters_dual += is.iters_dual; info->iters_primal += is.iters_primal; info->iters_phase1 += is.iters_phase1;
        info->nfactor += is.nfactor; info->nflips += is.nflips; info->t_factor += is.t_factor; info->t_ftran += is.t_ftran; info->t_btran += is.t_btran; info->t_price += is.t_price;
        for (int k = 0; k < ms; k++) rs[rows[k]] = rss[k];
        havebasis = 1;
        double obj = 0; for (int j = 0; j < n; j++) obj += L->c[j] * x[j];     /* (the set's objective, without c0) */
        if (o && o->verbose >= 2) printf("  row sifting round %d: %d of %d rows, status %d, %ld iterations, objective %.10g, %.2f s\n", round, ms, m, st, is.iters, obj, now() - t0);
        if (all) break;
        /* a stop, a time limit or the work budget (status 6: the automatic choice may take the interior-point method now),
         * or the caller's iteration limit: returned as they are */
        if (st == 6 || (st == 3 && o && o->maxit > 0 && its >= o->maxit)) break;
        if (o && o->timelimit > 0 && now() - t0 >= o->timelimit) { st = 6; break; }    /* (the limit passed in this round) */
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
            if (nv == 0) { if (!fin4) break; st = 4; goto sift_decide; }   /* (fin4: the set's result would be final: as before, the whole problem from the start) */
            /* the violated rows not falling (below 0.7 of their number three rounds before, from the fifth round): the
             * set does not settle, and every round costs a new start (netdiversion: 164 to 188 a round, 107 rounds,
             * 14.6 s against 5.3 s for the whole problem); the whole problem from the basis reached */
            if (round < 64) { nvh[round] = nv; obh[round] = obj; }
            /* (4 November 2026) the objective of the set not moving over two rounds while the violated rows do not fall
             * (as many as two rounds before, or more): the rows that join cut off the solution, not the set's optimal
             * face, and each round moves to another vertex of that face, which violates as many rows outside as before.
             * satellites2-25 (LP, 97 % of its columns of cost zero): the set's optimal value -30 is the whole problem's from
             * the first round on; 59, 81, 73, 75, 71, 66 rows violated in rounds of 435 to 780 iterations; the whole problem
             * from the basis reached costs about 2,600 to 3,000 iterations after any of them. The whole problem from the basis
             * reached after the third round: 7,951 -> 5,761 iterations. A set whose objective moves (neos-5104907-jarama,
             * neos-4976951-bunnoo: constant two rounds, then the rows that join raise it) or whose violated rows fall
             * (neos-5049753-cuanza, neos-799716, all LPs) goes on. BRISK_SXROWSIFTOBJSTALL=0: not */
            {   static int rso = -1; if (rso < 0) { const char *e_ = getenv("BRISK_SXROWSIFTOBJSTALL"); rso = !(e_ && atoi(e_) == 0); }
                if (rso && round >= 2 && round < 64 && nv >= nvh[round - 2] && fabs(obj - obh[round - 2]) <= 1e-9 * (1 + fabs(obj))) { st = 3; nv = 0; objstall = 1; }
            }
            if (round >= 5 && round < 64 && nv >= 0.7 * nvh[round - 3]) { st = 3; nv = 0;
                static int rsscr = -1; if (rsscr < 0) { const char *e_ = getenv("BRISK_SXROWSIFTSCRATCH"); rsscr = e_ && atoi(e_) == 1; }   /* (experiment: the whole problem from the start then) */
                if (rsscr) scratch = 1; }
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
        /* (4 November 2026) a set found unbounded (status 2) or left with numerical difficulties (4): its last basis is not
         * dual feasible for the whole problem, and the dual simplex method from it pays for that; the whole problem from the
         * start then. ns1904248 (LP): its first set (2,786 of 38,262 rows) unbounded, the whole problem from that basis 20,483
         * iterations and 22.5 s, from the start 4,457 and 0.69 s. BRISK_SXROWSIFTUNB=0: from the basis reached */
    sift_decide:;
        {   static int rsu = -1; if (rsu < 0) { const char *e_ = getenv("BRISK_SXROWSIFTUNB"); rsu = !(e_ && atoi(e_) == 0); }
            if (rsu && (st == 2 || st == 4)) { scratch = 1; unbset = 1; } }
        if (st != 0 || its >= cap) { for (int i = 0; i < m; i++) in[i] = 1; nin = m; all = 1; if (scratch) havebasis = 0; if (o && o->verbose >= 1) printf("  row sifting: %s after %d rounds: the whole problem %s\n", unbset ? "a solve on the set unbounded or with numerical difficulties" : objstall ? "the objective not moving and the violated rows not falling over two rounds," : scratch ? "most rows in the set" : st == 3 ? "a cap, or the violated rows not falling," : st != 0 ? "a solve on the set not optimal" : "the cap on the iterations", round + 1, scratch ? "from the start" : "from the basis reached"); }
    }
    if (o && o->verbose >= 1) printf("row sifting: %d rounds, %d of %d rows at the start, %d in the last, %ld iterations, %.2f s\n", round + 1, mstart, m, nin, its, now() - t0);
    info->iters = its; info->status = st; info->work = work; info->time = now() - t0;
    for (int j = 0; j < n; j++) { xo[j] = x[j]; cso[j] = cs[j]; }
    for (int i = 0; i < m; i++) { yo[i] = map[i] >= 0 ? ys[map[i]] : 0; rso[i] = in[i] ? rs[i] : SX_BASIC; }
    free(S.Ap); free(S.Ai); free(S.Ax); free(S.rl); free(S.ru);
    free(Rp); free(Rj); free(Rx); free(map); free(rows); free(cs); free(rs); free(rss); free(cs0); free(rs0); free(x); free(ys); free(vio); free(ord);
    return st;
}
static int psift_on(void) { static int v = -1; if (v < 0) { const char *e_ = getenv("BRISK_SXPSIFT"); v = !(e_ && atoi(e_) == 0); } return v; }
static int reduced_solve(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir) {
    const double ratio = getenv("BRISK_SXSIFT") ? atof(getenv("BRISK_SXSIFT")) : 100;   /* (experiments) */
    const int wide = !getenv("BRISK_SXNOSIFT") && R->n >= ratio * R->m && R->n >= (getenv("BRISK_SXSIFT") ? 1000 : 20000) && (o2->method != 2 || psift_on());
    /* (4 November 2026) the primal method too (it had been left out): the 16 wide LPs of the development set, the primal
     * method alone on one thread: fast0507 16.7 -> 1.07 s, rail507 16.5 -> 1.41 s, eilA101-2 4.27 -> 0.39 s, eilC76-2 0.87 ->
     * 0.16 s, rvb-sub 1.69 -> 0.90 s, supportcase6 2.37 -> 2.83 s, nine unchanged (Gurobi's primal method 1.83, 3.46, 0.48,
     * 0.12, 1.00, 2.17 s). BRISK_SXPSIFT=0: not for the primal method */
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
/* (experiments) a problem as a free MPS file: rows r<i>, columns x<j>, the objective minimized */
static void write_free_mps(const LpProb *Rp, FILE *fw) {
    const LpProb R = *Rp;
    {
            fprintf(fw, "NAME sxpre\nROWS\n N obj\n");
            for (int i = 0; i < R.m; i++) fprintf(fw, " %c r%d\n", R.rl[i] == R.ru[i] ? 'E' : R.rl[i] <= -LP_INF / 2 ? 'L' : 'G', i);
            fprintf(fw, "COLUMNS\n");
            for (int j = 0; j < R.n; j++) {
                if (R.c[j] != 0) fprintf(fw, " x%d obj %.17g\n", j, R.maxim ? -R.c[j] : R.c[j]);
                for (int p = R.Ap[j]; p < R.Ap[j + 1]; p++) fprintf(fw, " x%d r%d %.17g\n", j, R.Ai[p], R.Ax[p]);
            }
            fprintf(fw, "RHS\n");
            for (int i = 0; i < R.m; i++) { const double b = R.rl[i] > -LP_INF / 2 ? R.rl[i] : R.ru[i]; if (b != 0) fprintf(fw, " rhs r%d %.17g\n", i, b); }
            fprintf(fw, "RANGES\n");
            for (int i = 0; i < R.m; i++) if (R.rl[i] != R.ru[i] && R.rl[i] > -LP_INF / 2 && R.ru[i] < LP_INF / 2) fprintf(fw, " rng r%d %.17g\n", i, R.ru[i] - R.rl[i]);
            fprintf(fw, "BOUNDS\n");
            for (int j = 0; j < R.n; j++) {
                const double l = R.l[j], u = R.u[j];
                if (l <= -LP_INF / 2 && u >= LP_INF / 2) { fprintf(fw, " FR bnd x%d\n", j); continue; }
                if (l <= -LP_INF / 2) fprintf(fw, " MI bnd x%d\n", j); else if (l != 0) fprintf(fw, " LO bnd x%d %.17g\n", j, l);
                if (u < LP_INF / 2) fprintf(fw, " UP bnd x%d %.17g\n", j, u);
            }
            fprintf(fw, "ENDATA\n"); fclose(fw);
        }
}
static int ipm_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done);
static int dual_route_applies(const LpProb *R);
/* (8 November 2026, ISSUES 847) the columns that leave the logicals' basis dual infeasible: a column not fixed and not
 * boxed whose cost has the wrong sign for the one bound it can rest at (a free column: any cost). The dual simplex
 * method starts from that basis, and each such column calls for its dual phase 1 */
static long slack_dual_infeas(const LpProb *P) {
    const double sg = P->maxim ? -1.0 : 1.0;
    double cmax = 0; for (int j = 0; j < P->n; j++) if (fabs(P->c[j]) > cmax) cmax = fabs(P->c[j]);
    const double tol = 1e-9 * (1 + cmax);
    long k = 0;
    for (int j = 0; j < P->n; j++) {
        const double l = P->l[j], u = P->u[j], c = sg * P->c[j];
        if (l == u) continue;
        const int lf = l > -HUGEB, uf = u < HUGEB;
        if (lf && uf) continue;
        if (!lf && !uf) { if (fabs(c) > tol) k++; }
        else if (lf) { if (c < -tol) k++; }
        else if (c > tol) k++;
    }
    return k;
}
/* (12 November 2026) a route's attempt that does not give the result: its counts added to the caller's (iterations, factorizations,
 * the solves' and the pricing's times, the work), so that the Summary reports the work done; the route that follows overwrites
 * the result's own counts */
static void sxinfo_add(SimplexInfo *a, const SimplexInfo *b) {
    a->iters += b->iters; a->iters_dual += b->iters_dual; a->iters_primal += b->iters_primal; a->iters_phase1 += b->iters_phase1;
    a->nfactor += b->nfactor; a->nflips += b->nflips; a->work += b->work;
    a->t_factor += b->t_factor; a->t_ftran += b->t_ftran; a->t_btran += b->t_btran; a->t_price += b->t_price;
}
/* (14 November 2026, ISSUES 880) the record of the method that gave a route's first solve (the automatic choice's switch to the
 * interior-point method, the race, the interior-point method's iterations and time), carried into the route's result when the
 * solve that follows (from the first solve's basis or point) has none of its own: the routes that solve a transformed problem
 * first (its dual, the problem on the symmetry classes, the problem with the substitution) kept the counts of both solves and
 * dropped this record, and the Summary named the dual simplex method where the interior-point method and the crossover gave the
 * result (neos-1367061's root LP, LP, by the automatic choice) */
static void sxinfo_method(SimplexInfo *a, const SimplexInfo *first) {
    if (!a->auto_ipm && first->auto_ipm) { a->auto_ipm = 1; a->iters_first = first->iters_first; a->t_first = first->t_first; }
    if (!a->raced && first->raced) a->raced = first->raced;
    if (!a->ipm_iters && first->ipm_iters) { a->ipm_iters = first->ipm_iters; a->t_ipm = first->t_ipm; }
}
static int dual_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done, SimplexInfo *ifail) {
    *done = 0;
    if (getenv("BRISK_SXNODUAL")) { if (o2->dr_declined) *o2->dr_declined = 1; return 0; }
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
    if (df * est > R->m && !force) { if (o2->dr_declined) *o2->dr_declined = 1; return 0; }
    const double t0 = now();
    const int verbose = o2->verbose;
    LpProb D, DR; SxDual *M = NULL; SxPre *PD = NULL; char why[160] = "";
    sxpre_dualize(R, &D, &M);
    const int sfs_ = sxpre_set_for_simplex(1);
    const int ps = sxpre_presolve(&D, &DR, &PD, verbose >= 1, why, sizeof why);
    sxpre_set_for_simplex(sfs_);
    /* (the interior-point route: the dual at most a quarter of the rows, 23 October 2026; the method's iterations cost
     * about the same on either form, since it chooses its normal equations itself, and the crossover's basis must be far
     * smaller to pay for the change: opm2-z10-s4, supportcase10, supportcase22 (LPs; 11 to 40 times fewer rows) 41.5 ->
     * 18.5, 15.9 -> 7.1, 12.5 -> 3.9 s, physiciansched3-3 (LP; 3.6 times) 71 -> 187 s, 63 iterations against 37) */
    /* (24 October 2026, the night: at most an eighth. The simplex presolve's rows made equations by the cost made
     * physiciansched3-3 66,090 x 14,488, its dual 4.7 times smaller: on the dual the method stalled, 111 s, against 36.4 s
     * on the problem itself (Gurobi's time, 36.4 s); the duals that paid were 11 to 40 times smaller. The crossover's saving
     * grows with the ratio, the risk of the method on the other form does not) */
    const double dfi = o2->in_dual ? 8.0 : df;
    /* (8 November 2026, ISSUES 847; BRISK_SXDRFEAS=1, off until judged) the simplex methods on the dual problem only when it
     * does not cost the start the problem itself has: the dual simplex method starts from the logicals' basis, and on the
     * problem itself that basis is dual feasible when every column that is not boxed has a cost of the sign of its one
     * bound (no dual phase 1: the method has one phase). On the dual problem the same basis is then primal feasible and
     * dual infeasible (the dual simplex method on the dual is the primal simplex method on the problem: a phase 1 to reach
     * dual feasibility, which the problem itself had at the start, and a phase 2 that undoes it). The dual problem is
     * declined when its logicals' basis has more dual infeasibilities than the problem's own. neos-1367061 (LP): 0 on the
     * problem, 1,304 on its dual; the dual simplex method alone 28,421-33,732 iterations (21-25 s) through the dual,
     * 6,268-7,003 (1.9-2.1 s) on the problem itself, Gurobi's dual simplex method 6,392. Not for the interior-point route
     * (in_dual: no simplex start) */
    long dr_nR = -1, dr_nD = -1;
    static int drfeas = -1; if (drfeas < 0) { const char *e_ = getenv("BRISK_SXDRFEAS"); drfeas = e_ && atoi(e_) > 0; }
    if (!ps && drfeas && !o2->in_dual && !force) { dr_nR = slack_dual_infeas(R); dr_nD = slack_dual_infeas(&DR); }
    const int dr_feas_decl = dr_nD > dr_nR;
    if (!ps && getenv("BRISK_SXDRDBG")) {   /* (8 November 2026, a diagnostic) each form's share of free (not fixed) columns without cost */
        long zR = 0, nR_ = 0, zD = 0, nD_ = 0;
        for (int j = 0; j < R->n; j++) if (R->l[j] < R->u[j]) { nR_++; if (R->c[j] == 0) zR++; }
        for (int j = 0; j < DR.n; j++) if (DR.l[j] < DR.u[j]) { nD_++; if (DR.c[j] == 0) zD++; }
        fprintf(stderr, "[drdbg] problem %d x %d zero-cost %.3f | dual %d x %d zero-cost %.3f | dinf %ld %ld\n", R->m, R->n,
                nR_ ? (double)zR / nR_ : 0, DR.m, DR.n, nD_ ? (double)zD / nD_ : 0, slack_dual_infeas(R), slack_dual_infeas(&DR));
    }
    if (dr_feas_decl && verbose >= 1) printf("  the dual problem's start has %ld dual infeasibilities, the problem's own %ld: the problem itself is solved\n", dr_nD, dr_nR);
    if (ps || (dfi * DR.m > R->m && !force) || DR.m == 0 || DR.n == 0 || dr_feas_decl) {
        if (o2->dr_declined) *o2->dr_declined = 1;
        if (verbose >= 1) printf("  the dual (%d x %d, presolved %d x %d) is not used\n", D.m, D.n, ps ? -1 : DR.m, ps ? -1 : DR.n);
        if (!ps) lp_free(&DR);
        lp_free(&D); sxpre_free(PD); sxpre_dual_free(M);
        return 0;
    }
    if (verbose >= 1) printf("  the dual problem: %d x %d, presolved %d x %d (%.2f s)\n", D.m, D.n, DR.m, DR.n, now() - t0);
    /* (5 November 2026, experiments on ISSUES 835) BRISK_SXWRITEDUAL=<file>: the presolved dual as a free MPS file */
    if (getenv("BRISK_SXWRITEDUAL")) { FILE *fw = fopen(getenv("BRISK_SXWRITEDUAL"), "w"); if (fw) write_free_mps(&DR, fw); }
    const int md = D.m, nd = D.n, mdr = DR.m, ndr = DR.n;
    int *csdr = malloc(sizeof(int) * (ndr + 1)), *rsdr = malloc(sizeof(int) * (mdr + 1));
    double *xdr = malloc(sizeof(double) * (ndr + 1)), *ydr = malloc(sizeof(double) * (mdr + 1));
    SimplexInfo id; memset(&id, 0, sizeof id);
    SimplexOpts od = *o2; od.dr_declined = NULL; if (getenv("BRISK_SXDUALMETHOD")) od.method = atoi(getenv("BRISK_SXDUALMETHOD"));   /* (experiments) */
    int st, dn2 = 0;
    if (o2->in_dual) {
        /* (23 October 2026) the interior-point route on a tall problem: the interior-point method and the crossover on its
         * dual, whose basis has as many rows as the problem has columns (opm2-z10-s4, LP: 146,325 rows, 5,958 columns;
         * the crossover on the problem itself made 128,391 dual pushes in 28 s, Gurobi's on the dual 0.07 s) */
        od.method = 3; od.orig = NULL;
        st = ipm_route(&DR, &od, xdr, ydr, csdr, rsdr, &id, &dn2);
    }
    /* (26 October 2026) the time limit is what is left after the interior-point attempt: od kept the whole of it, and the
     * simplex method on the dual ran 9 s past a limit of 10 s (opm2-z12-s8, -lpmethod 4) */
    if (!dn2 && o2->timelimit > 0) { od.timelimit = o2->timelimit - (now() - t0); if (od.timelimit <= 0) { st = 6; dn2 = 1; } }
    /* (5 November 2026, experiments on ISSUES 835) BRISK_SXDUALSTART=<file>: the presolved dual solved from the basis in the
     * file (lines "c <column> <status>" and "r <row> <status>", the statuses of simplex.h), the costs perturbed as from the
     * logicals' basis and the steepest-edge weights computed exactly when a row first comes to the top (given_perturb);
     * BRISK_SXDUALBASIS=<file>: the dual's basis at the end of its solve, in that format. With BRISK_SXSTOPIT=<k> (the dual
     * simplex method stopped after k iterations) the route ends there, status 3, instead of solving the problem itself */
    int *cs_start = NULL, *rs_start = NULL;
    if (getenv("BRISK_SXDUALSTART")) {
        FILE *fs = fopen(getenv("BRISK_SXDUALSTART"), "r");
        if (fs) {
            cs_start = calloc((size_t)ndr + 1, sizeof(int)); rs_start = calloc((size_t)mdr + 1, sizeof(int));
            char t[4]; int k, v;
            while (fscanf(fs, "%3s %d %d", t, &k, &v) == 3) { if (t[0] == 'c' && k >= 0 && k < ndr) cs_start[k] = v; else if (t[0] == 'r' && k >= 0 && k < mdr) rs_start[k] = v; }
            fclose(fs);
            od.cstat0 = cs_start; od.rstat0 = rs_start; od.given_perturb = 1;
        }
    }
    if (!dn2) { od.method = o2->in_dual ? 0 : od.method; st = reduced_solve(&DR, &od, xdr, ydr, csdr, rsdr, &id); }
    free(cs_start); free(rs_start);
    if (getenv("BRISK_SXDUALBASIS")) {
        FILE *fb = fopen(getenv("BRISK_SXDUALBASIS"), "w");
        if (fb) { for (int j = 0; j < ndr; j++) fprintf(fb, "c %d %d\n", j, csdr[j]); for (int i = 0; i < mdr; i++) fprintf(fb, "r %d %d\n", i, rsdr[i]); fclose(fb); }
    }
    lp_free(&DR);
    if (st == 3 && getenv("BRISK_SXSTOPIT")) {
        free(csdr); free(rsdr); free(xdr); free(ydr); lp_free(&D); sxpre_free(PD); sxpre_dual_free(M);
        *ir = id; ir->time = now() - t0; ir->dualized = 1; *done = 1;
        return 3;
    }
    if (st != 0) {
        if (verbose >= 1) printf("  the dual ended with status %d: the problem itself is solved\n", st);
        /* (24 October 2026) its iterations and time kept for the caller's summary: stopped by the interior-point method
         * beside it, the route's work was reported as 0 iterations (irish-electricity, LP: 23,000 in 20 s) */
        if (ifail) sxinfo_add(ifail, &id);
        else { ir->iters += id.iters; ir->time += id.time; ir->iters_dual += id.iters_dual; ir->iters_primal += id.iters_primal; }
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
    SimplexOpts o3 = *o2; o3.cstat0 = cs0; o3.rstat0 = rs0; o3.nopresolve = 1; o3.timelimit = sx_tleft(o2, t0);
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
    /* (13 November 2026, ISSUES 880) the dual's solve's automatic choice passed on too: when its dual simplex method stopped at the
     * interior-point method's predicted work, the Summary names the methods that ran (it read "dual simplex method, 2443 iterations
     * (0 dual, 0 primal)" on neos-1367061's root LP, where the interior-point method and the crossover gave the result) */
    sxinfo_method(ir, &id);
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
/* whether the bounded substitution can apply: (23 October 2026, the evening) a scan before the presolve, an upper bound on
 * the substitutions, the columns without cost, not fixed, with two entries or more, one of them in an equality row short
 * enough for at most 4 new entries; the presolve only when they can reach the threshold (its run was the route's cost on
 * the problems it skipped: lp_transport_1000, LP, 2.89 -> 3.69 s) */
static int aggb_applies(const LpProb *R, const SimplexOpts *o2) {
    if (getenv("BRISK_SXNOAGGBR") || o2->cstat0 || o2->xstart || R->m < 100) return 0;
    int *rlen = calloc((size_t)R->m + 1, sizeof(int)); long cand = 0;
    for (int j = 0; j < R->n; j++) for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) rlen[R->Ai[p]]++;
    for (int j = 0; j < R->n; j++) {
        const int cl = R->Ap[j + 1] - R->Ap[j];
        if (R->c[j] != 0.0 || cl < 2 || R->l[j] == R->u[j]) continue;
        for (int p = R->Ap[j]; p < R->Ap[j + 1]; p++) { const int i = R->Ai[p];
            if (R->rl[i] == R->ru[i] && (long)(cl - 1) * (rlen[i] - 2) <= 4) { cand++; break; } }
    }
    free(rlen);
    return cand >= fmax(10.0, 0.05 * R->n);
}
int simplex_aggb_possible(const LpProb *L) { SimplexOpts o; memset(&o, 0, sizeof o); return aggb_applies(L, &o); }
static int aggb_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done, SimplexInfo *ifail) {
    *done = 0;
    if (!aggb_applies(R, o2)) return 0;
    const double t0 = now();
    LpProb R2; SxPre *P2 = NULL; char why[160] = "";
    if (sxpre_presolve_ex(R, &R2, &P2, 0, 1, why, sizeof why)) return 0;
    if (R->n - R2.n < fmax(10.0, 0.05 * R->n)) {
        if (o2->verbose >= 2) printf("  the bounded substitution would remove %d of %d columns: not used (%.3f s)\n", R->n - R2.n, R->n, now() - t0);
        lp_free(&R2); sxpre_free(P2); return 0;
    }
    const int verbose = o2->verbose;
    if (verbose >= 1) printf("  the bounded substitution for the simplex method: %d x %d -> %d x %d (%.2f s)\n", R->m, R->n, R2.m, R2.n, now() - t0);
    if (verbose >= 2) {     /* (5 November 2026) the two problems' entries and rows */
        long lr = 0, lr2 = 0; int ranged = 0;
        for (int i = 0; i < R2.m; i++) ranged += R2.rl[i] > -1e29 && R2.ru[i] < 1e29 && R2.rl[i] != R2.ru[i];
        int *c1 = calloc(R->m + 1, sizeof(int)), *c2 = calloc(R2.m + 1, sizeof(int));
        for (int p = 0; p < R->Ap[R->n]; p++) c1[R->Ai[p]]++;
        for (int p = 0; p < R2.Ap[R2.n]; p++) c2[R2.Ai[p]]++;
        for (int i = 0; i < R->m; i++) if (c1[i] > lr) lr = c1[i];
        for (int i = 0; i < R2.m; i++) if (c2[i] > lr2) lr2 = c2[i];
        printf("  the substitution: entries %d -> %d, longest row %ld -> %ld, %d ranged rows\n", R->Ap[R->n], R2.Ap[R2.n], lr, lr2, ranged);
        free(c1); free(c2);
    }
    const int m2 = R2.m, n2 = R2.n;
    int *cs2 = malloc(sizeof(int) * (n2 + 1)), *rs2 = malloc(sizeof(int) * (m2 + 1));
    double *x2 = malloc(sizeof(double) * (n2 + 1)), *y2 = malloc(sizeof(double) * (m2 + 1));
    SimplexInfo i2; memset(&i2, 0, sizeof i2);
    int st = reduced_solve(&R2, o2, x2, y2, cs2, rs2, &i2);
    lp_free(&R2);
    if (st != 0) {
        if (verbose >= 1 && !(brisk_flag_get(o2->stop))) printf("  the problem with the substitution ended with status %d: the problem itself is solved\n", st);
        /* (12 November 2026) its counts kept for the Summary: stopped by a time limit or by the method beside it, ofi2 (LP) was
         * reported with 0 iterations after 21 s of them */
        if (ifail) sxinfo_add(ifail, &i2);
        free(cs2); free(rs2); free(x2); free(y2); sxpre_free(P2);
        return 0;
    }
    int *cs0 = malloc(sizeof(int) * (R->n + 1)), *rs0 = malloc(sizeof(int) * (R->m + 1));
    double *x0 = malloc(sizeof(double) * (R->n + 1)), *y0 = malloc(sizeof(double) * (R->m + 1));
    sxpre_postsolve(P2, x2, y2, cs2, rs2, x0, y0, cs0, rs0);
    free(cs2); free(rs2); free(x2); free(y2); free(x0); free(y0); sxpre_free(P2);
    SimplexOpts o3 = *o2; o3.cstat0 = cs0; o3.rstat0 = rs0; o3.nopresolve = 1; o3.timelimit = sx_tleft(o2, t0);
    SimplexInfo i3;
    st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    if (verbose >= 1) printf("  the problem from the substituted problem's basis: status %d, %ld iterations (%.2f s)\n", st, i3.iters, i3.time);
    free(cs0); free(rs0);
    *ir = i3;
    ir->iters = i2.iters + i3.iters; ir->iters_dual = i2.iters_dual + i3.iters_dual; ir->iters_primal = i2.iters_primal + i3.iters_primal;
    ir->iters_phase1 = i2.iters_phase1 + i3.iters_phase1; ir->nfactor = i2.nfactor + i3.nfactor; ir->nflips = i2.nflips + i3.nflips;
    ir->t_factor = i2.t_factor + i3.t_factor; ir->t_ftran = i2.t_ftran + i3.t_ftran; ir->t_btran = i2.t_btran + i3.t_btran; ir->t_price = i2.t_price + i3.t_price;
    ir->time = now() - t0; ir->work = i2.work + i3.work; sxinfo_method(ir, &i2);
    *done = 1;
    return st;
}

/* a problem with symmetry: solved on the classes of its coarsest equitable partition (sxpre_symmetry; the optimal value
 * is the same), the solution carried back (each column at its class' value: an optimal point, in general not basic) and
 * turned into an optimal basis by the crossover and the primal method. If the problem on the classes does not end
 * optimal, the problem itself is solved (*done stays 0). */
/* (1 November 2026) ifail: when the problem on the classes does not end optimal, its counts (iterations, factorizations),
 * which the caller adds to those of the route that follows: highschool1-aigio (LP) stopped by the time limit in the classes'
 * row sifting after 40,626 iterations and reported 0 */
static int sym_route(const LpProb *R, const SimplexOpts *o2, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done, SimplexInfo *ifail) {
    *done = 0;
    const double t0 = now();
    LpProb Q; SxSym *M = NULL;
    if (!sxpre_symmetry(R, &Q, &M, o2->verbose >= 1, 0)) return 0;
    const double tsym = now() - t0;
    SimplexOpts oq = *o2; oq.nosymmetry = 1; oq.nopresolve = 0; oq.timelimit = sx_tleft(o2, t0);
    double *xq = malloc(sizeof(double) * (Q.n + 1)), *yq = malloc(sizeof(double) * (Q.m + 1));
    SimplexInfo iq;
    const int mq = Q.m, nq = Q.n;
    int st = simplex_solve(&Q, &oq, xq, NULL, yq, NULL, NULL, NULL, &iq);
    lp_free(&Q);
    if (o2->verbose >= 1) printf("  the problem on the classes (%d x %d, %.2f s to find them): status %d, %ld iterations (%.2f s)\n", mq, nq, tsym, st, iq.iters, iq.time);
    if (st != 0) {
        ifail->iters = iq.iters; ifail->iters_dual = iq.iters_dual; ifail->iters_primal = iq.iters_primal; ifail->iters_phase1 = iq.iters_phase1;
        ifail->nfactor = iq.nfactor; ifail->nflips = iq.nflips; ifail->work = iq.work;
        free(xq); free(yq); sxpre_sym_free(M); return 0;
    }
    double *x = malloc(sizeof(double) * (R->n + 1)), *y = malloc(sizeof(double) * (R->m + 1));
    sxpre_sym_lift(M, xq, yq, x, y);
    free(xq); free(yq); sxpre_sym_free(M);
    SimplexOpts o3 = *o2; o3.xstart = x; o3.ystart = y; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.timelimit = sx_tleft(o2, t0);
    SimplexInfo i3;
    st = sx_solve_core(R, &o3, xr, NULL, yr, NULL, csr, rsr, &i3);
    free(x); free(y);
    if (o2->verbose >= 1) printf("  the problem from the classes' solution: status %d, %ld pushes, %ld iterations (%.2f s)\n", st, i3.pushes, i3.iters, i3.time);
    *ir = i3;
    ir->iters = iq.iters + i3.iters; ir->iters_dual = iq.iters_dual + i3.iters_dual; ir->iters_primal = iq.iters_primal + i3.iters_primal;
    ir->iters_phase1 = iq.iters_phase1 + i3.iters_phase1; ir->nfactor = iq.nfactor + i3.nfactor; ir->nflips = iq.nflips + i3.nflips;
    ir->t_factor = iq.t_factor + i3.t_factor; ir->t_ftran = iq.t_ftran + i3.t_ftran; ir->t_btran = iq.t_btran + i3.t_btran; ir->t_price = iq.t_price + i3.t_price;
    ir->time = now() - t0; ir->sym_m = mq; ir->sym_n = nq; ir->work = iq.work + i3.work; sxinfo_method(ir, &iq);
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
    /* (26 October 2026) no attempt once the time limit has passed: the standard form and the analysis do not look at it */
    if ((o2->timelimit > 0 && o2->timelimit - (now() - t0) <= 0) || (brisk_flag_get(o2->stop))) { snprintf(ii->why, sizeof ii->why, "the time limit"); ii->status = 6; return 6; }
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
    /* (26 October 2026) the clock from the route's start: taken after the attempt through the dual, it gave the next attempt
     * the whole time limit again (opm2-z12-s8, -lpmethod 4: 10 s of estimates past a limit of 10 s) */
    const double t0 = now();
    if (!o2->in_dual && !getenv("BRISK_SXNOIPMDUAL") && dual_route_applies(R)) {
        /* a tall problem: the interior-point method and the crossover on its dual (dual_route; in_dual keeps the dual's
         * own route from dualizing again) */
        SimplexOpts od = *o2; od.in_dual = 1;
        int dn = 0;
        const int st = dual_route(R, &od, xr, yr, csr, rsr, ir, &dn, NULL);
        if (dn) { *done = 1; return st; }
    }
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
    const int stopped = ist == 6 || (brisk_flag_get(o2->stop));
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
        SimplexOpts o3 = *o2; o3.xstart = xo; o3.ystart = yo; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.method = 0; o3.orig = NULL; o3.timelimit = sx_tleft(o2, t0);
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
        if (ist == 6 || (brisk_flag_get(o2->stop))) return 0;
        LpProb Rc = *R;
        SedumiProb P2; LpMap M2;
        lp_to_sedumi(&Rc, 0, 0, &P2, &M2);
        if (M2.status || P2.ns != 0 || P2.n == 0) { sedumi_free(&P2); lp_map_free(&M2); return 0; }
        SocpProb Q = { P2.m, P2.n, P2.Ap, P2.Ai, P2.Ax, P2.b, P2.c, P2.nf, P2.nl, P2.nq, P2.q, P2.nr, P2.r };
        SocpOpts so; socp_default_opts(&so); so.tol = 1e-8; so.verbose = 0;
        /* (26 October 2026) no second attempt without time left: 1 ms did not stop its analysis (opm2-z12-s8: 147 s past the
         * limit) */
        if (o2->timelimit > 0 && o2->timelimit - (now() - t0) <= 0) { if (o2->verbose >= 1) printf("  the interior-point method stopped at the time limit\n"); return 6; }
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
    SimplexOpts o3 = *o2; o3.xstart = x; o3.ystart = y; o3.nopresolve = 1; o3.cstat0 = o3.rstat0 = NULL; o3.method = 0; o3.timelimit = sx_tleft(o2, t0);
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
#define SX_AUTO_NDK 1000.0         /* (ISSUES 880) the count by the factorization's own ordering once the work reaches this many entries touched per AMD factor entry */
#define SX_AUTO_PATCAP 5e7          /* the largest pattern (entries of M M') ordered for the prediction */
/* one form's normal equations M M': M has nr rows and the columns Cp, Ci (nc), and ns singleton columns more (the
 * slacks: the diagonal only). The dense columns as lpsolve.c's ne_setup takes them: longer than max(50, 10 x the
 * average, nr / 10), when nr >= 500 and there are 1 to 400 of them and fewer than a quarter of the columns. */
typedef struct { int nr, nc, ns, nd; int *Cp, *Ci; char *isd; double lb, pb, lnz, cost, fl, nnz; } NeForm;   /* (pb: the bound on the pattern's entries; fl: the factorization's flops; nnz: the entries with the slacks) */
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
        F->lnz = lnz; F->fl = fl; F->nnz = nnz;
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
    BRISK_THREAD_GUARD(J->st = ipm_route(J->R, &ox, J->x, J->y, J->cs, J->rs, &J->info, &J->done), J->st = 4);   /* (a fatal error in this thread: the job without a result) */
    if (!J->race) return NULL;
    pthread_mutex_lock(&J->mu);
    if (J->done && J->st == 0 && !J->abandoned) { brisk_flag_set(&J->stop_s, 1); if (J->stop_other) brisk_flag_set(J->stop_other, 1); if (J->stop_other2) brisk_flag_set(J->stop_other2, 1); }
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
    int nd_pending, nd_dual; double nd_c, nd_wp0;   /* (ISSUES 880) the count by the factorization's own ordering deferred: the form and its AMD cost */
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
/* the interior-point method's predicted work (the forms' orderings), printed with the work so far; 0 when too large */
/* (14 November 2026, ISSUES 880) the chosen form's factorization counted with the ordering the interior-point method's factorization
 * uses (lpsolve.c's estimate, lp_ne_cost_nd: the nested dissection where it pays) instead of AMD's, which can be several times higher:
 * triptim4's root LP (LP) 5.3e10 flops an iteration by AMD against the factorization's 7.1e9, the switch after 59 s of the simplex
 * method for the interior-point method's 10 s. Returns the iteration's cost with that count (c when it is not lower). No time limit:
 * the decision depends on the work counts only, the same at any thread count and on a busy machine (the pattern is at most
 * SX_AUTO_PATCAP entries; METIS's order is the same at any thread count) */
/* (14 November 2026, ISSUES 880; the default, BRISK_SXAUTONDFAST=0: lpsolve.c's whole analysis) the count by the nested dissection alone: the pattern of M M' (its
 * dense columns left out), the order of nd_order (the dissection the factorization calls) and the factor's exact column counts by
 * the elimination tree (schol_exact_counts), without the factorization's whole symbolic analysis (lp_ne_cost_nd: 2.1-2.2 s on
 * neos-4763324-toguru and physiciansched3-3, LPs, where the count found the orderings equal; this count 1.42 and 1.28 s, triptim4
 * 0.72 -> 0.41 s, the same decisions: when the dissection is worse than AMD the prediction keeps AMD's count). 1: done */
int nd_order(int n, const int *xadj, const int *adjncy, int *perm);
int schol_exact_counts(int m, const int *deg, int *const *nbr, const int *P, double *flops_out, double *nnz_out);
static int auto_nd_fast(const NeForm *F, double *fl, double *lnz) {
    const int nr = F->nr, nc = F->nc; const int *Cp = F->Cp, *Ci = F->Ci;
    int *tp = calloc((size_t)nr + 2, sizeof(int)), *tj = malloc(sizeof(int) * ((size_t)Cp[nc] + 1));
    for (int j = 0; j < nc; j++) if (!F->isd[j]) for (int p = Cp[j]; p < Cp[j + 1]; p++) tp[Ci[p] + 1]++;
    for (int i = 0; i < nr; i++) tp[i + 1] += tp[i];
    { int *f = malloc(sizeof(int) * ((size_t)nr + 1)); memcpy(f, tp, sizeof(int) * nr);
      for (int j = 0; j < nc; j++) if (!F->isd[j]) for (int p = Cp[j]; p < Cp[j + 1]; p++) tj[f[Ci[p]]++] = j;
      free(f); }
    int *mark = malloc(sizeof(int) * ((size_t)nr + 1)), *xadj = malloc(sizeof(int) * ((size_t)nr + 1));
    int *adj = malloc(sizeof(int) * ((size_t)(2.0 * F->pb) + 1)), *P = malloc(sizeof(int) * ((size_t)nr + 1));
    for (int i = 0; i < nr; i++) mark[i] = -1;
    long e = 0; xadj[0] = 0;
    for (int i = 0; i < nr; i++) {
        mark[i] = i;
        for (int t = tp[i]; t < tp[i + 1]; t++) { const int j = tj[t]; for (int p = Cp[j]; p < Cp[j + 1]; p++) { const int k = Ci[p]; if (mark[k] != i) { mark[k] = i; adj[e++] = k; } } }
        xadj[i + 1] = (int)e;
    }
    int ok = 0;
    if (nd_order(nr, xadj, adj, P) == 0) {
        int *deg = malloc(sizeof(int) * ((size_t)nr + 1)); int **nbr = malloc(sizeof(int *) * ((size_t)nr + 1));
        for (int i = 0; i < nr; i++) { deg[i] = xadj[i + 1] - xadj[i]; nbr[i] = adj + xadj[i]; }
        ok = schol_exact_counts(nr, deg, nbr, P, fl, lnz) == 0;
        free(deg); free(nbr);
    }
    free(tp); free(tj); free(mark); free(xadj); free(adj); free(P);
    return ok;
}
static double auto_nd_cost(AutoCtx *A, NeForm *F, double c, double work) {
    int *mp = malloc(sizeof(int) * ((size_t)F->nc + 1)), *mi = malloc(sizeof(int) * ((size_t)F->Cp[F->nc] + 1)); int k = 0, q = 0;
    mp[0] = 0;
    for (int j = 0; j < F->nc; j++) { if (F->isd && F->isd[j]) continue; for (int p = F->Cp[j]; p < F->Cp[j + 1]; p++) mi[q++] = F->Ci[p]; mp[++k] = q; }
    double fl2 = 0, lnz2 = 0; const double tq = now();
    static int fast_ = -1; if (fast_ < 0) { const char *e_ = getenv("BRISK_SXAUTONDFAST"); fast_ = !(e_ && atoi(e_) == 0); }
    const int ok = fast_ ? auto_nd_fast(F, &fl2, &lnz2) : lp_ne_cost_nd(F->nr, k, mp, mi, F->fl, 1e30, &fl2, &lnz2);
    free(mp); free(mi);
    if (!(ok && fl2 < F->fl)) {
        if (A->verbose >= 2) printf("  the interior-point method's own ordering not counted (%d, %.3g flops against AMD's %.3g; %.2f s; work so far %.3g)\n", ok, fl2, F->fl, now() - tq, work);
        return c;
    }
    const double nd = F->nd, nr = F->nr;
    const double c2 = fl2 + nd * (4.0 * lnz2 + 2.0 * nd * nr) + 10.0 * (4.0 * lnz2 + 4.0 * F->nnz + 4.0 * nd * nr);
    if (A->verbose >= 1) printf("  the interior-point method's factorization by its own ordering: %.3g flops, factor %.3g (AMD's %.3g, %.3g; %.2f s; work so far %.3g)\n", fl2, lnz2, F->fl, F->lnz, now() - tq, work);
    F->lnz = lnz2; F->fl = fl2;
    return c2;
}
static double auto_wp_print(AutoCtx *A, double work, double c, int dual) {
    const NeForm *P = &A->P, *D = &A->D;
    const double wp = c > 0 ? SX_AUTO_IT * c / SX_AUTO_FLOPS + SX_AUTO_SETUP : 0;
    if (A->verbose >= 1) {
        if (c > 0) printf("  the work so far %.3g; the interior-point method's predicted %.3g (%s form: %d rows, factor %.3g, %d dense columns)\n",
                          work, wp, dual ? "dual" : "standard", dual ? D->nr : P->nr, dual ? D->lnz : P->lnz, dual ? D->nd : P->nd);
        else printf("  the work so far %.3g; the normal equations too large to predict: the simplex method continues\n", work);
    }
    A->wpred = wp;
    return wp;
}
/* defer: the count may wait for the work (auto_budget, which makes it at the next call); 0 for the callers that take the prediction as
 * final (wrace_budget, an experiment: the count at once) */
static double auto_wp(AutoCtx *A, double work, int defer) {
    A->predicted = 1;
    /* the dual form first; the standard form's ordering only when its lower bound is below twice the dual's cost */
    NeForm *P = &A->P, *D = &A->D;
    if (D->nr < P->nr) neform_cost(D);      /* (the dual form only with fewer rows, as lpsolve.c takes it) */
    if (!(D->cost > 0 && P->lb > 2.0 * D->cost)) neform_cost(P);
    double c = 0; int dual = 0;
    if (D->cost > 0 && (P->cost == 0 || D->cost <= 0.5 * P->cost)) { c = D->cost; dual = 1; }
    else c = P->cost;
    /* (14 November 2026, ISSUES 880) a large prediction (the factorization at least 1e9 flops by AMD, where the interior-point method
     * orders by the nested dissection) that would keep the simplex method running: the count by that ordering (auto_nd_cost), once the
     * simplex method's work reaches SX_AUTO_NDK times AMD's factor entries. The count's time follows AMD's factor: 16-175 ns an entry on
     * the 42 development LPs where it acts (2.2 s on neos-4763324-toguru, 4.1 s on rail03, 1.2 s on satellites2-25, whose simplex method
     * ends in a few seconds); at the simplex method's 2e8-1e9 entries touched a second, 1,000 entries a factor entry keep the count at
     * most about a sixth of the time spent before it (a few percent typically), and a problem the simplex method ends sooner never pays
     * for it. BRISK_SXAUTOND=0: AMD's count alone */
    static int and_ = -1; if (and_ < 0) { const char *e_ = getenv("BRISK_SXAUTOND"); and_ = !(e_ && atoi(e_) == 0); }
    static double ndk_ = -1; if (ndk_ < 0) { const char *e_ = getenv("BRISK_SXAUTONDK"); ndk_ = e_ ? atof(e_) : SX_AUTO_NDK; }
    NeForm *F = dual ? D : P;
    const double wp0 = c > 0 ? SX_AUTO_IT * c / SX_AUTO_FLOPS + SX_AUTO_SETUP : 0;
    if (and_ && c > 0 && F->fl >= 1e9 && wp0 > work) {
        const double wnd = ndk_ * F->lnz;
        if (work >= wnd) c = auto_nd_cost(A, F, c, work);
        else if (!defer) c = auto_nd_cost(A, F, c, work);
        else if (wnd < wp0) {
            A->nd_pending = 1; A->nd_dual = dual; A->nd_c = c; A->nd_wp0 = wp0;
            auto_wp_print(A, work, c, dual);
            if (A->verbose >= 2) printf("  the count by the interior-point method's own ordering deferred to the work %.3g\n", wnd);
            return wnd;
        }
    }
    return auto_wp_print(A, work, c, dual);
}
static double auto_budget(void *vctx, double work) {
    AutoCtx *A = vctx;
    if (A->predicted && A->nd_pending) {     /* (ISSUES 880) the deferred count, now that the work has reached it */
        A->nd_pending = 0;
        const double c = auto_nd_cost(A, A->nd_dual ? &A->D : &A->P, A->nd_c, work);
        const double wp = auto_wp_print(A, work, c, A->nd_dual);
        if (wp > work) return wp;
    }
    if (A->predicted) { A->declined = 1; return 0; }
    if (A->stage == 0) {
        /* a lower bound first, without an ordering */
        A->stage = 1;
        auto_forms(A);
        const double w0 = SX_AUTO_IT * (A->D.nr < A->P.nr ? fmin(A->P.lb, A->D.lb) : A->P.lb) / SX_AUTO_FLOPS + SX_AUTO_SETUP;
        if (w0 > work) return w0;
    }
    const double wp = auto_wp(A, work, 1);
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
int simplex_dual_route_possible(const LpProb *L) { return dual_route_applies_f(L, 2.0); }
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
    int aggb;                   /* 1: the problem with the bounded substitution (aggb_route), beside the problem itself */
    volatile int stop, one, won, fin;    /* (fin: the job has ended) */
    volatile int *stop_main;    /* the other simplex route's stop flag (the caller waits for this job: always valid) */
    struct RaceCtx_ *C;
} DirJob;
typedef struct RaceCtx_ { const LpProb *R; const SimplexOpts *o; int nf; IpmJob *J; pthread_t th; double t0; volatile int stop_ind, threads, main_declined;
                          DirJob *DJ; pthread_t thd; pthread_mutex_t mu;
                          int dj_want, in_dual; SimplexOpts dj_o; } RaceCtx;   /* (the direct route's job started at the budget) */
static void *dir_job(void *p) {
    DirJob *D = p;
    omp_set_num_threads(1);
    BRISK_THREAD_GUARD(D->st = reduced_solve(D->R, &D->o, D->x, D->y, D->cs, D->rs, &D->info), D->st = 4);
    if (D->st == 0 && !brisk_flag_get(&D->stop)) {
        brisk_flag_set(&D->won, 1); brisk_flag_set(D->stop_main, 1);
        pthread_mutex_lock(&D->C->mu);
        if (D->C->J) brisk_flag_set(&D->C->J->stop, 1);
        pthread_mutex_unlock(&D->C->mu);
    }
    return NULL;
}
static double g_race_wait;   /* (diagnostic) the time spent waiting for a losing interior-point method to stop */
/* the second simplex job (DirJob) started: the direct route beside the route through the dual problem, or the primal
 * simplex method (an experiment) */
static void race_start_dj(RaceCtx *C, int prim) {
    const LpProb *R = C->R;
    DirJob *DJ = calloc(1, sizeof *DJ);
    if (!DJ) return;
    DJ->R = R; DJ->o = C->dj_o; DJ->o.stop = &DJ->stop; DJ->o.workbudget = 0; DJ->o.budget_cb = NULL; DJ->o.budget_ctx = NULL;
    DJ->primal = prim; if (prim) DJ->o.method = 2;
    DJ->one = 1; DJ->o.threads = &DJ->one; DJ->o.verbose = 0; DJ->stop_main = &C->stop_ind; DJ->C = C;
    DJ->x = malloc(sizeof(double) * (R->n + 1)); DJ->y = malloc(sizeof(double) * (R->m + 1));
    DJ->cs = malloc(sizeof(int) * (R->n + 1)); DJ->rs = malloc(sizeof(int) * (R->m + 1));
    if (pthread_create(&C->thd, NULL, dir_job, DJ) == 0) {
        C->DJ = DJ; C->threads = 1;
        if (C->o->verbose >= 1) printf(prim ? "  the primal simplex method on a thread of its own, beside the dual simplex method\n"
                                            : "  the direct route on a thread of its own, beside the route through the dual problem\n");
    }
    else { free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); }
}
/* (27 October 2026) the bounded substitution's route as the second simplex job, beside the simplex method on the problem
 * itself: the route costs a second presolve and a solve of the problem from the basis it carries back, which on a small
 * problem is most of its time (pds-02, LP: 0.0145 -> 0.0237 s with the route first, ganges 0.0050 -> 0.0077, pilot4 0.024
 * -> 0.036), while on others it is everything (satellites2-40, LP: 12.1 -> 1.5 s, 18,776 -> 7,884 iterations). The first
 * to end wins; a route that declines after its presolve leaves the problem to the main solve. BRISK_SXAGGBFIRST=1: the
 * route first and alone, as before */
static void *aggb_job(void *p) {
    DirJob *D = p;
    omp_set_num_threads(1);
    int dn = 0;
    BRISK_THREAD_GUARD(D->st = aggb_route(D->R, &D->o, D->x, D->y, D->cs, D->rs, &D->info, &dn, &D->info), (D->st = 4, dn = 0));
    if (dn && D->st == 0 && !brisk_flag_get(&D->stop)) {
        brisk_flag_set(&D->won, 1); brisk_flag_set(D->stop_main, 1);
        pthread_mutex_lock(&D->C->mu);
        if (D->C->J) brisk_flag_set(&D->C->J->stop, 1);
        pthread_mutex_unlock(&D->C->mu);
    }
    brisk_flag_set(&D->fin, 1);
    return NULL;
}
static void race_start_aggb(RaceCtx *C) {
    const LpProb *R = C->R;
    DirJob *DJ = calloc(1, sizeof *DJ);
    if (!DJ) return;
    DJ->R = R; DJ->o = C->dj_o; DJ->o.stop = &DJ->stop; DJ->o.workbudget = 0; DJ->o.budget_cb = NULL; DJ->o.budget_ctx = NULL;
    DJ->aggb = 1; DJ->one = 1; DJ->o.threads = &DJ->one; DJ->o.verbose = 0; DJ->stop_main = &C->stop_ind; DJ->C = C;
    DJ->x = malloc(sizeof(double) * (R->n + 1)); DJ->y = malloc(sizeof(double) * (R->m + 1));
    DJ->cs = malloc(sizeof(int) * (R->n + 1)); DJ->rs = malloc(sizeof(int) * (R->m + 1));
    if (pthread_create(&C->thd, NULL, aggb_job, DJ) == 0) {
        C->DJ = DJ; C->threads = 1;
        if (C->o->verbose >= 1) printf("  the problem with the bounded substitution on a thread of its own, beside the problem itself\n");
    }
    else { free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); }
}
static double race_budget(void *vctx, double work) {
    RaceCtx *C = vctx;
    (void)work;
    /* (27 October 2026) the direct route's job of a tall problem starts here, with the interior-point method, while the
     * route through the dual problem still works: a problem cheaper than the budget is solved by one route alone (Netlib,
     * LP: 2 % of the time for the second route's presolve, set-up and thread on its small problems) */
    if (C->dj_want && C->in_dual && !C->DJ) race_start_dj(C, 0);
    C->dj_want = 0;
    if (C->J || (C->DJ && brisk_flag_get(&C->DJ->won))) return 1e300;
    const LpProb *R = C->R;
    IpmJob *J = calloc(1, sizeof *J);
    if (!J) return 1e300;
    lp_dup(R, &J->own); J->R = &J->own; J->race = 1; J->o = *C->o; J->o.stop = NULL; J->o.threads = NULL; J->stop_other = &C->stop_ind;
    if (C->o->orig) {
        lp_dup(C->o->orig->L, &J->own_orig); sxout_init(&J->own_out, &J->own_orig); J->has_orig = 1;
        J->own_o.L = &J->own_orig; J->own_o.out = &J->own_out; J->o.orig = &J->own_o;
    }
    /* (BRISK_SXRACESXT=k, an experiment of 26 October 2026: the simplex method keeps k threads beside the interior-point
     * method, which takes the rest; default 1) */
    static int sxt = -1; if (sxt < 0) { const char *e = getenv("BRISK_SXRACESXT"); sxt = e && atoi(e) >= 1 ? atoi(e) : 1; }
    const int ks = sxt < C->nf - 1 ? sxt : 1;
    /* (27 October 2026) beside the bounded substitution's job still running, the main simplex method yields its thread at
     * this budget and the interior-point method takes all but the job's: on the problems where it wins, it had 3 of 4
     * threads when the route ran first and alone, 2 when the main simplex method kept one beside the job (rail02, LP: 56 ->
     * 73 s, triptim1 6.4 -> 8.0 s); a small problem has ended before the budget */
    const int aggb_wait = C->DJ && C->DJ->aggb && !brisk_flag_get(&C->DJ->fin) && !brisk_flag_get(&C->DJ->won);
    J->nt = aggb_wait ? C->nf - 1 : C->nf - ks - (C->DJ && !brisk_flag_get(&C->DJ->fin)); if (J->nt < 1) J->nt = 1;
    J->threads_other = &C->threads; J->nt_all = C->DJ ? 1 : C->nf;
    if (C->DJ) J->stop_other2 = &C->DJ->stop;
    if (J->o.timelimit > 0) J->o.timelimit = fmax(C->o->timelimit - (now() - C->t0), 1e-3);
    J->x = malloc(sizeof(double) * (R->n + 1)); J->y = malloc(sizeof(double) * (R->m + 1));
    J->cs = malloc(sizeof(int) * (R->n + 1)); J->rs = malloc(sizeof(int) * (R->m + 1));
    pthread_mutex_init(&J->mu, NULL);
    C->threads = C->DJ ? 1 : ks;    /* (before the job starts: the simplex method keeps one thread while it runs) */
    pthread_mutex_lock(&C->mu);
    const int ok = !(C->DJ && brisk_flag_get(&C->DJ->won)) && pthread_create(&C->th, NULL, ipm_job, J) == 0;
    if (ok) C->J = J;
    pthread_mutex_unlock(&C->mu);
    if (!ok) { job_free(J); C->threads = C->DJ ? 1 : C->nf; return 1e300; }
    if (C->o->verbose >= 1) printf("  the interior-point method started on %d of %d threads, beside the simplex method (the first to end wins)\n", J->nt, C->nf);
    if (aggb_wait) {
        if (C->o->verbose >= 1) printf("  the simplex method on the problem itself gives its thread to the interior-point method; the problem with the bounded substitution goes on\n");
        brisk_flag_set(&C->main_declined, 1); return 0;
    }
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
    int small;     /* (the budget at its floor: a problem whose size does not set it) */
    {
        const double b0 = getenv("BRISK_SXRACEB0") ? atof(getenv("BRISK_SXRACEB0")) : SX_RACE_B0;
        const double bs = getenv("BRISK_SXRACESTART") ? atof(getenv("BRISK_SXRACESTART")) : SX_RACE_START;
        os.workbudget = fmax(b0 * (R->Ap[R->n] + R->m), bs);
        small = b0 * (R->Ap[R->n] + R->m) < bs;
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
    C.dj_o = os;
    /* (27 October 2026) the direct route's job of a small tall problem (the budget at its floor) starts at the budget
     * (race_budget), with the interior-point method, so that a problem cheaper than the budget is solved by one route; a
     * larger one starts it here as before: the route through the dual problem spends seconds on its own preparation
     * before its work counts (neos-2075418-temuka, LP: 24 s with the direct route at once, 31 s from the budget). The
     * primal racer (an experiment) at once. BRISK_SXDJNOW=1: the direct route at once whatever the size */
    static int djnow = -1; if (djnow < 0) djnow = getenv("BRISK_SXDJNOW") != NULL;
    if (prim || (tall && (djnow || !small))) race_start_dj(&C, prim);
    else if (tall) C.dj_want = 1;
    DJ = C.DJ;
    omp_set_num_threads(C.threads);
    SimplexInfo is; memset(&is, 0, sizeof is);
    /* (the direct route beside it from the start: the route through the dual problem at the factor 1; a small problem's
     * at the factor 2 as without the race, its direct route's job only from the budget) */
    os.dual_race = C.DJ && !C.DJ->primal;
    C.in_dual = 1;
    int dn = 0, st = dual_route(R, &os, xr, yr, csr, rsr, &is, &dn, NULL);
    C.in_dual = 0; C.dj_want = 0;
    DJ = C.DJ;
    os.dual_race = 0;
    if (!dn && !brisk_flag_get(&C.stop_ind)) {
        if (DJ && !DJ->primal) {   /* the dual problem was not used: the direct route's job is this route's solve */
            pthread_join(C.thd, NULL);
            st = DJ->st; is = DJ->info;
            memcpy(xr, DJ->x, sizeof(double) * R->n); memcpy(yr, DJ->y, sizeof(double) * R->m);
            memcpy(csr, DJ->cs, sizeof(int) * R->n); memcpy(rsr, DJ->rs, sizeof(int) * R->m);
            if (brisk_flag_get(&DJ->won)) brisk_flag_set(&C.stop_ind, 0);     /* (its own result: not a stop from outside) */
            free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); DJ = C.DJ = NULL;
        } else {
            /* (26 October 2026) the time left after the route through the dual problem, not the race's whole limit */
            if (o2->timelimit > 0) os.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3);
            int da = 0;
            static int aggbfirst = -1; if (aggbfirst < 0) aggbfirst = getenv("BRISK_SXAGGBFIRST") != NULL;
            if (!aggbfirst && nf >= 2 && !C.DJ && aggb_applies(R, &os)) {
                C.dj_o = os; race_start_aggb(&C); DJ = C.DJ;
                omp_set_num_threads(C.threads);
            }
            if (!DJ) { if (o2->timelimit > 0) os.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3); st = aggb_route(R, &os, xr, yr, csr, rsr, &is, &da, NULL); }
            if (!da && !brisk_flag_get(&C.stop_ind)) { if (o2->timelimit > 0) os.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3); st = reduced_solve(R, &os, xr, yr, csr, rsr, &is); }
        }
    }
    int dj_joined = 0, ipm_joined = 0;
    if (brisk_flag_get(&C.main_declined)) {
        /* (27 October 2026) the main simplex method yielded at the budget (race_budget): the first of the bounded
         * substitution's job and the interior-point method (each stops the other when it wins); if neither ends optimal,
         * the simplex method on the problem itself to the end */
        if (DJ) { pthread_join(C.thd, NULL); dj_joined = 1; }
        if (!(DJ && brisk_flag_get(&DJ->won)) && C.J && !brisk_flag_get(&C.stop_ind)) { pthread_join(C.th, NULL); ipm_joined = 1; }
        if (!(DJ && brisk_flag_get(&DJ->won)) && !(ipm_joined ? C.J->done && C.J->st == 0 : brisk_flag_get(&C.stop_ind))) {
            SimplexOpts oc = os; oc.budget_cb = NULL; oc.budget_ctx = NULL; oc.workbudget = 0; oc.stop = NULL;
            C.threads = nf; oc.threads = &C.threads; omp_set_num_threads(nf);
            if (o2->timelimit > 0) oc.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3);
            st = reduced_solve(R, &oc, xr, yr, csr, rsr, &is);
        } else if (ipm_joined && C.J->done && C.J->st == 0) brisk_flag_set(&C.stop_ind, 1);     /* (the interior-point method's result, below) */
    }
    omp_set_num_threads(nt0);
    if (DJ) {
        /* the two simplex routes: the direct one won (it stopped the other), or it is told to stop and waited for (it reads
         * its flag every iteration) */
        brisk_flag_set(&DJ->stop, 1);
        if (!dj_joined) pthread_join(C.thd, NULL);
        if (brisk_flag_get(&DJ->won) && brisk_flag_get(&C.stop_ind)) {
            st = DJ->st; is = DJ->info;
            memcpy(xr, DJ->x, sizeof(double) * R->n); memcpy(yr, DJ->y, sizeof(double) * R->m);
            memcpy(csr, DJ->cs, sizeof(int) * R->n); memcpy(rsr, DJ->rs, sizeof(int) * R->m);
            brisk_flag_set(&C.stop_ind, 0);
            if (o2->verbose >= 1) printf(DJ->primal ? "  the primal simplex method ended first\n" : DJ->aggb ? "  the problem with the bounded substitution ended first\n" : "  the direct route ended first\n");
        }
        free(DJ->x); free(DJ->y); free(DJ->cs); free(DJ->rs); free(DJ); DJ = C.DJ = NULL;
    }
    IpmJob *J = C.J;
    if (!J) { *ir = is; *done = 1; pthread_mutex_destroy(&C.mu); return st; }
    int use_ipm = 0;
    if (brisk_flag_get(&C.stop_ind)) {       /* (the interior-point method ended optimal first; the simplex method stopped, possibly inside
                               the route through the dual problem, which then returns as not done) */
        if (!ipm_joined) pthread_join(C.th, NULL);          /* (it has ended: it set the flag) */
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
        brisk_flag_set(&J->stop, 1);
        if (ipm_joined) job_free(J);
        else {
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
        }
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
#ifdef SX_RACE
/* (27 October 2026, an experiment: BRISK_SXDETRACE=1) the race's bound on the route through the dual problem, in the
 * deterministic automatic choice, by counted work instead of the clock. A tall problem went through its dual alone, with
 * no bound on the loss: ex9's root LP after the MIP presolve (LP) 155 s, its dual problem's basis inverse dense, against
 * 21.5 s directly (ISSUES 401). With three threads or more, the route through the dual problem and the direct route run at
 * the same time, each on a thread of its own and each with its own work count (each route's path depends on its own
 * count only); a route that ends gives its count, the other stops once its count passes it, and of the routes that end
 * the one with less work gives the result: the same every run. Both stop at the interior-point method's predicted work,
 * as the sequential order does, the method running beside them on the other threads. */
typedef struct {
    pthread_mutex_t mu; pthread_cond_t cv; AutoCtx *A; int nipm;
    int have_w0, have_wp; double w0, wp;
    int ended[2], budget[2]; double wend[2];
    double k; int busy;
    int go, skip;       /* the direct route starts when the route through the dual problem passes the interior-point
                         * method's lower bound (go), or not at all when that route ends before it (skip): a problem cheaper
                         * than that is solved by one route alone (neos-3402294-bobin's root LP: 0.23 s alone, 0.55 s with
                         * both from the first budget; glass-sc, LP: 0.068 s alone, 0.146 s) */
} WRace;
typedef struct {
    WRace *W; int me, one; const LpProb *R; SimplexOpts o;
    double *x, *y; int *cs, *rs; SimplexInfo info; int st, dn;
} WArm;
static void wrace_launch(AutoCtx *A, int nt) {
    const LpProb *R = A->R; IpmJob *J = calloc(1, sizeof *J);
    if (!J) return;
    J->R = R; J->o = *A->o; J->nt = nt;
    if (J->o.timelimit > 0) J->o.timelimit = fmax(A->o->timelimit - (now() - A->t0), 1e-3);
    J->x = malloc(sizeof(double) * (R->n + 1)); J->y = malloc(sizeof(double) * (R->m + 1));
    J->cs = malloc(sizeof(int) * (R->n + 1)); J->rs = malloc(sizeof(int) * (R->m + 1));
    if (pthread_create(&A->th, NULL, ipm_job, J) == 0) {
        A->launched = 1; A->job = J;
        if (A->verbose >= 1) printf("  the interior-point method started on %d threads, beside the two simplex routes\n", nt);
    } else { free(J->x); free(J->y); free(J->cs); free(J->rs); free(J); }
}
/* each route's budget: a function of its own count (the interior-point method's lower bound, then its prediction), and a
 * stop once the other route has ended with less work; checked every 5 % of the count, so that a route stops soon. The
 * prediction (an ordering) is computed outside the lock, by the first route that needs it, and not at all by a route
 * whose stop comes before the lower bound */
static double wrace_budget(void *v, double work) {
    WArm *a = v; WRace *W = a->W; AutoCtx *A = W->A;
    double nb;
    pthread_mutex_lock(&W->mu);
    /* the direct route wins only with less than 1/k of the other's work (k = BRISK_SXDETRACEK, 1 by default): it stops at
     * that fraction, and the route through the dual problem at k times the direct route's work */
    const double thr = !W->ended[1 - a->me] ? 1e300 : a->me ? W->wend[0] / W->k : W->k * W->wend[1] * (1 + 1e-12);
    if (work >= thr) nb = 0;
    else {
        if (!W->have_w0) {
            W->have_w0 = 1; A->stage = 1;
            auto_forms(A);
            W->w0 = SX_AUTO_IT * (A->D.nr < A->P.nr ? fmin(A->P.lb, A->D.lb) : A->P.lb) / SX_AUTO_FLOPS + SX_AUTO_SETUP;
        }
        if (a->me == 0 && !W->go && work >= W->w0) { W->go = 1; pthread_cond_broadcast(&W->cv); }
        if (work < W->w0) nb = W->w0;
        else {
            if (!W->have_wp) {
                if (!W->busy) {
                    W->busy = 1;
                    pthread_mutex_unlock(&W->mu);
                    const double wp = auto_wp(A, work, 0);
                    if (wp > 0 && W->nipm >= 1) wrace_launch(A, W->nipm);
                    pthread_mutex_lock(&W->mu);
                    W->wp = wp; W->have_wp = 1; W->busy = 0;
                    pthread_cond_broadcast(&W->cv);
                } else while (!W->have_wp) pthread_cond_wait(&W->cv, &W->mu);
            }
            if (W->wp == 0) nb = 1e300;
            else if (work < W->wp) nb = W->wp;
            else { W->budget[a->me] = 1; nb = 0; }
        }
        if (nb > 0) nb = fmin(fmin(nb, thr), work + fmax(1e6, 0.05 * work));
    }
    pthread_mutex_unlock(&W->mu);
    return nb;
}
static void *wrace_arm(void *p) {
    WArm *a = p; WRace *W = a->W;
    omp_set_num_threads(1);
    int ended;
    if (a->me == 0) { a->dn = 0; BRISK_THREAD_GUARD(a->st = dual_route(a->R, &a->o, a->x, a->y, a->cs, a->rs, &a->info, &a->dn, NULL), (a->st = 4, a->dn = 0)); ended = a->dn && a->st != 6; }
    else {
        pthread_mutex_lock(&W->mu);
        while (!W->go) pthread_cond_wait(&W->cv, &W->mu);
        const int skip = W->skip;
        pthread_mutex_unlock(&W->mu);
        if (skip) { a->st = 6; return NULL; }
        BRISK_THREAD_GUARD(a->st = reduced_solve(a->R, &a->o, a->x, a->y, a->cs, a->rs, &a->info), a->st = 4); ended = a->st != 6;
    }
    pthread_mutex_lock(&W->mu);
    if (ended) { W->ended[a->me] = 1; W->wend[a->me] = a->info.work; }
    if (a->me == 0 && !W->go) { W->go = 1; W->skip = ended; pthread_cond_broadcast(&W->cv); }
    pthread_mutex_unlock(&W->mu);
    return NULL;
}
static int wrace_on(const LpProb *R, const AutoCtx *A) {
    static int on = -1; if (on < 0) on = getenv("BRISK_SXDETRACE") && atoi(getenv("BRISK_SXDETRACE")) == 1;
    return on && A->nt0 >= 3 && !getenv("BRISK_SXNORACE") && dual_route_applies(R);
}
/* the two routes run; the result of the one that ended with less work in xr.. and *ir, or 6 (A->declined set when the
 * direct route reached the interior-point method's predicted work) */
static int wrace_run(const LpProb *R, const SimplexOpts *os, AutoCtx *A, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir) {
    WRace W; memset(&W, 0, sizeof W); pthread_mutex_init(&W.mu, NULL); pthread_cond_init(&W.cv, NULL); W.A = A; W.nipm = A->nt0 - 2;
    W.k = getenv("BRISK_SXDETRACEK") ? fmax(atof(getenv("BRISK_SXDETRACEK")), 1.0) : 1.0;
    WArm a[2]; memset(a, 0, sizeof a);
    pthread_t th[2]; int ok[2] = { 0, 0 };
    for (int k = 0; k < 2; k++) {
        a[k].W = &W; a[k].me = k; a[k].one = 1; a[k].R = R; a[k].o = *os;
        a[k].o.threads = &a[k].one; a[k].o.budget_cb = wrace_budget; a[k].o.budget_ctx = &a[k];
        if (k == 1) a[k].o.verbose = 0;
        a[k].x = malloc(sizeof(double) * (R->n + 1)); a[k].y = malloc(sizeof(double) * (R->m + 1));
        a[k].cs = malloc(sizeof(int) * (R->n + 1)); a[k].rs = malloc(sizeof(int) * (R->m + 1));
    }
    if (os->verbose >= 1) printf("  the direct route on a thread of its own from the interior-point method's lower bound, beside the route through the dual problem (the result by counted work)\n");
    for (int k = 0; k < 2; k++) ok[k] = pthread_create(&th[k], NULL, wrace_arm, &a[k]) == 0;
    for (int k = 0; k < 2; k++) if (!ok[k]) wrace_arm(&a[k]);      /* (no thread: run here; the other, on its thread, gives
                                                                      * the start the direct route waits for) */
    for (int k = 0; k < 2; k++) if (ok[k]) pthread_join(th[k], NULL);
    int win = -1;
    if (W.ended[0] && W.ended[1]) win = W.k * W.wend[1] < W.wend[0] ? 1 : 0;
    else if (W.ended[0]) win = 0;
    else if (W.ended[1]) win = 1;
    int st = 6;
    if (win >= 0) {
        memcpy(xr, a[win].x, sizeof(double) * R->n); memcpy(yr, a[win].y, sizeof(double) * R->m);
        memcpy(csr, a[win].cs, sizeof(int) * R->n); memcpy(rsr, a[win].rs, sizeof(int) * R->m);
        *ir = a[win].info; st = a[win].st;
        if (os->verbose >= 1) {
            if (W.ended[1 - win]) printf("  the %s ended with less counted work (%.3g against %.3g)\n", win ? "direct route" : "route through the dual problem", W.wend[win], W.wend[1 - win]);
            else printf("  the %s ended (counted work %.3g); the other did not\n", win ? "direct route" : "route through the dual problem", W.wend[win]);
        }
    } else {
        *ir = a[1].info; ir->iters += a[0].info.iters;
        A->declined = W.budget[1];
    }
    for (int k = 0; k < 2; k++) { free(a[k].x); free(a[k].y); free(a[k].cs); free(a[k].rs); }
    pthread_mutex_destroy(&W.mu); pthread_cond_destroy(&W.cv);
    return st;
}
/* (8 November 2026, a test: BRISK_SXFORMRACE=1) the dual simplex method alone on a tall problem, with two free threads or
 * more: the problem itself and its dual problem each on half the threads, from the start. The route through the dual problem
 * alone took 1.152 of the time of the problem itself on 26 tall LPs (each form far the better on some: ns1904248 0.74 against
 * 37.7 s, genus-sym-g62-2 over 60 s against 4.5 s; neither form's share of columns without cost nor its start's dual
 * infeasibilities separate them), the better of the two 0.705. The result by counted work, as wrace_run's: a route that ends
 * gives its count, the other stops once its own passes it, and of the routes that end the one with less work gives the
 * result: the same every run */
typedef struct { WRace *W; int me, nt; const LpProb *R; SimplexOpts o; double *x, *y; int *cs, *rs; SimplexInfo info; int st, dn; } FArm;
/* (on one thread the two take turns, W->go set: a route runs until its count passes the other's by a step, then waits
 * for its turn; a route that has finished (W->skip bit) no longer takes one) */
static double frace_budget(void *v, double work) {
    FArm *a = v; WRace *W = a->W; const int me = a->me, ot = 1 - me;
    pthread_mutex_lock(&W->mu);
    const double thr = !W->ended[ot] ? 1e300 : me ? W->wend[0] : W->wend[1] * (1 + 1e-12);
    double nb = work >= thr ? 0 : fmin(thr, work + fmax(1e6, 0.05 * work));
    if (W->go && nb > 0) {
        W->w0 = me ? W->w0 : work; W->wp = me ? work : W->wp;          /* the running counts: w0 the dual problem's, wp the problem's */
        const double other = me ? W->w0 : W->wp;
        nb = fmin(thr, fmax(work, other) + fmax(1e6, 0.05 * work));
        if (!(W->skip & (1 << ot))) {
            W->busy = ot; pthread_cond_broadcast(&W->cv);
            while (W->busy != me && !(W->skip & (1 << ot))) pthread_cond_wait(&W->cv, &W->mu);
        }
    }
    pthread_mutex_unlock(&W->mu);
    return nb;
}
static void *frace_arm(void *p) {
    FArm *a = p; WRace *W = a->W;
    omp_set_num_threads(a->nt);
    if (W->go) {     /* (one thread: the route through the dual problem first) */
        pthread_mutex_lock(&W->mu);
        while (W->busy != a->me && !(W->skip & (1 << (1 - a->me)))) pthread_cond_wait(&W->cv, &W->mu);
        pthread_mutex_unlock(&W->mu);
    }
    int ended;
    if (a->me == 0) { a->dn = 0; BRISK_THREAD_GUARD(a->st = dual_route(a->R, &a->o, a->x, a->y, a->cs, a->rs, &a->info, &a->dn, NULL), (a->st = 4, a->dn = 0)); ended = a->dn && a->st != 6; }
    else { BRISK_THREAD_GUARD(a->st = reduced_solve(a->R, &a->o, a->x, a->y, a->cs, a->rs, &a->info), a->st = 4); ended = a->st != 6; }
    pthread_mutex_lock(&W->mu);
    if (ended) { W->ended[a->me] = 1; W->wend[a->me] = a->info.work; }
    W->skip |= 1 << a->me; W->busy = 1 - a->me; pthread_cond_broadcast(&W->cv);
    pthread_mutex_unlock(&W->mu);
    return NULL;
}
static int frace_on(const LpProb *R, const SimplexOpts *o) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_SXFORMRACE"); on = e_ && atoi(e_) == 1; }
    /* (at every thread count, so that the result does not depend on it: on one thread the two forms share it) */
    return on && o && o->method == 1 && !o->direct_peer && !o->in_dual && !o->dual_race && dual_route_applies(R);
}
/* the two forms run; *done set when one ended (its result in xr.. and *ir) */
static int frace_run(const LpProb *R, const SimplexOpts *os, double *xr, double *yr, int *csr, int *rsr, SimplexInfo *ir, int *done) {
    *done = 0;
    const int nf = omp_get_max_threads();
    WRace W; memset(&W, 0, sizeof W); pthread_mutex_init(&W.mu, NULL); pthread_cond_init(&W.cv, NULL);
    W.go = nf < 2; W.busy = 0;
    FArm a[2]; memset(a, 0, sizeof a);
    pthread_t th[2]; int ok[2] = { 0, 0 };
    for (int k = 0; k < 2; k++) {
        a[k].W = &W; a[k].me = k; a[k].nt = nf < 2 ? 1 : k ? nf / 2 : nf - nf / 2; a[k].R = R; a[k].o = *os;
        a[k].o.threads = &a[k].nt; a[k].o.budget_cb = frace_budget; a[k].o.budget_ctx = &a[k];
        a[k].o.workbudget = 1e6; a[k].o.no_dual_route = 1; a[k].o.dr_declined = NULL;
        if (k == 1) a[k].o.verbose = 0;
        a[k].x = malloc(sizeof(double) * (R->n + 1)); a[k].y = malloc(sizeof(double) * (R->m + 1));
        a[k].cs = malloc(sizeof(int) * (R->n + 1)); a[k].rs = malloc(sizeof(int) * (R->m + 1));
        if (!a[k].x || !a[k].y || !a[k].cs || !a[k].rs) {
            for (int q = 0; q <= k; q++) { free(a[q].x); free(a[q].y); free(a[q].cs); free(a[q].rs); }
            pthread_mutex_destroy(&W.mu); return 0;
        }
    }
    if (os->verbose >= 1) printf("  the problem itself and its dual problem side by side, on %d and %d threads (the result by counted work)\n", a[1].nt, a[0].nt);
    for (int k = 0; k < 2; k++) ok[k] = pthread_create(&th[k], NULL, frace_arm, &a[k]) == 0;
    for (int k = 0; k < 2; k++) if (!ok[k]) frace_arm(&a[k]);     /* (no thread: run here, after or beside the other) */
    for (int k = 0; k < 2; k++) if (ok[k]) pthread_join(th[k], NULL);
    omp_set_num_threads(nf);
    int win = -1;
    if (W.ended[0] && W.ended[1]) win = W.wend[1] < W.wend[0] ? 1 : 0;
    else if (W.ended[0]) win = 0;
    else if (W.ended[1]) win = 1;
    int st = 0;
    if (win >= 0) {
        memcpy(xr, a[win].x, sizeof(double) * R->n); memcpy(yr, a[win].y, sizeof(double) * R->m);
        memcpy(csr, a[win].cs, sizeof(int) * R->n); memcpy(rsr, a[win].rs, sizeof(int) * R->m);
        *ir = a[win].info; st = a[win].st; *done = 1;
        if (os->verbose >= 1) {
            if (W.ended[1 - win]) printf("  the %s ended with less counted work (%.3g against %.3g)\n", win ? "problem itself" : "dual problem", W.wend[win], W.wend[1 - win]);
            else printf("  the %s ended (counted work %.3g); the other did not\n", win ? "problem itself" : "dual problem", W.wend[win]);
        }
    } else if (a[1].st == 6 || a[0].st == 6) { *ir = a[1].info; ir->iters += a[0].info.iters; st = 6; *done = 1; }
    for (int k = 0; k < 2; k++) { free(a[k].x); free(a[k].y); free(a[k].cs); free(a[k].rs); }
    pthread_mutex_destroy(&W.mu); pthread_cond_destroy(&W.cv);
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
    int dn = 0, st;
#ifdef SX_RACE
    if (wrace_on(R, &A)) st = wrace_run(R, &os, &A, xr, yr, csr, rsr, &is);
    else
#endif
    {
        st = dual_route(R, &os, xr, yr, csr, rsr, &is, &dn, NULL);
        /* (14 November 2026, ISSUES 880) the route through the dual stopped by the budget (the interior-point method's predicted
         * work reached, A.declined): the interior-point method follows at once. The problem itself was solved by the simplex
         * method again, up to the first budget, and that solve's record replaced the dual's (neos-1367061's root LP, LP: the
         * Summary's first solve 2,443 iterations in 0.14 s, where the dual simplex method had run about 1.4 s before it) */
        if (!dn && A.declined) st = 6;
        else if (!dn) {
            SimplexInfo i0_ = is;
            if (o2->timelimit > 0) os.timelimit = fmax(o2->timelimit - (now() - t0), 1e-3);   /* (the time left, 26 October 2026) */
            st = reduced_solve(R, &os, xr, yr, csr, rsr, &is);
            sxinfo_add(&is, &i0_); is.time += i0_.time;
        }
    }
    auto_free(&A);
#ifdef SX_RACE
    if (A.launched) {
        IpmJob *J = A.job;
        omp_set_num_threads(A.nt0);
        /* (26 October 2026) a stop at the time limit is not the budget's: the interior-point method is told to stop too, and
         * not waited for to its end (opm2-z12-s8, the mixed-integer developer's first LP with 60 s: over 180 s) */
        const int tl_hit = o2->timelimit > 0 && now() - t0 >= o2->timelimit;
        if (!(st == 6 && A.declined) || tl_hit) J->stop = 1;      /* the simplex method finished within the budget: its result */
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
    if (!sxpre_symmetry_cap(L, &Q, &M, o && o->verbose >= 1, 10.0 * ((double)L->Ap[L->n] + L->m + L->n), 0.7 * size_pre)
        && !sxpre_symmetry_canon(L, &Q, &M, o && o->verbose >= 1, 10.0 * ((double)L->Ap[L->n] + L->m + L->n), o ? o->canon : NULL)) return 0;
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
static int simplex_solve_body(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                  SimplexInfo *info) {
    sx_env_refresh();     /* (the settings' table of each thread is built again: the environment may have changed) */
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
/* (4 November 2026) the crossover of a tall problem through its dual (simplex.h). On a problem with many more rows than
 * columns the crossover's basis has as many members as rows, and every row whose dual is not zero at the interior point
 * costs a dual push (its logical leaves): rmine13 (LP, 197,155 rows, 23,980 columns, its interior-point method run on the
 * dual form) made 163,343 dual pushes, 14.5 s, where Gurobi's crossover, on its dual form, takes about 2.6 s. Its dual
 * problem D (sxpre_dualize) has a row per column: the point is mapped to it (D's variables: the row duals, as minimization,
 * split into their positive and negative parts for a ranged row, and for each boxed column the multiplier of its upper
 * bound, max(0, a_j'y - c_j); D's row duals: -(x - s), s the bound each column was shifted by), the crossover runs on D,
 * and its optimal basis gives one of L by complementarity, from which the simplex method ends on L. */
int simplex_crossover_dual(const LpProb *L, const SimplexOpts *o, const double *x0, const double *y0, double *x, SimplexInfo *info) {
    LpProb D; SxDual *M = NULL;
    if (sxpre_dualize(L, &D, &M)) return -1;
    const int m = L->m, n = L->n, md = D.m, nd = D.n;
    const double sg = L->maxim ? -1.0 : 1.0, big = 1e29;
    double *xd = xcalloc((size_t)nd + 1, sizeof(double)), *yd = xcalloc((size_t)md + 1, sizeof(double)), *ymin = xmalloc(sizeof(double) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) ymin[i] = sg * y0[i];
    {   /* D's columns in sxpre_dualize's order: the rows of L (one, or two for a ranged row), then the boxed columns */
        int k = 0;
        double *s = xcalloc((size_t)n + 1, sizeof(double)), *rl = xmalloc(sizeof(double) * ((size_t)m + 1)), *ru = xmalloc(sizeof(double) * ((size_t)m + 1));
        for (int i = 0; i < m; i++) { rl[i] = L->rl[i]; ru[i] = L->ru[i]; }
        for (int j = 0; j < n; j++) {
            const double l = L->l[j], u = L->u[j];
            s[j] = l > -big ? l : u < big ? u : 0;
            if (s[j] != 0) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int i = L->Ai[p]; if (rl[i] > -big) rl[i] -= L->Ax[p] * s[j]; if (ru[i] < big) ru[i] -= L->Ax[p] * s[j]; }
        }
        for (int i = 0; i < m && k < nd; i++) {
            const int lf = rl[i] > -big, uf = ru[i] < big;
            if (!lf && !uf) continue;
            const double yi = ymin[i];
            if (lf && uf && rl[i] != ru[i]) { xd[k++] = yi > 0 ? yi : 0; xd[k++] = yi < 0 ? -yi : 0; }
            else { double v = yi; if (lf && !uf && v < 0) v = 0; if (uf && !lf && v > 0) v = 0; xd[k++] = v; }
        }
        int r = 0;
        for (int j = 0; j < n; j++) {
            const double l = L->l[j], u = L->u[j];
            if (l == u) continue;
            yd[r++] = -(x0[j] - s[j]);
            if (l > -big && u < big) {
                double ay = 0; for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) ay += L->Ax[p] * ymin[L->Ai[p]];
                const double w = ay - sg * L->c[j];
                if (k < nd) xd[k++] = w > 0 ? w : 0;
            }
        }
        /* (the boxed columns' w come after all the rows' v in D: the loop above appends them in the columns' order, as
         * sxpre_dualize does; a count mismatch means the two disagree: not applied) */
        free(s); free(rl); free(ru);
        if (k != nd || r != md) { free(xd); free(yd); free(ymin); lp_free(&D); sxpre_dual_free(M); return -1; }
    }
    int *csd = xmalloc(sizeof(int) * ((size_t)nd + 1)), *rsd = xmalloc(sizeof(int) * ((size_t)md + 1));
    SimplexOpts od = o ? *o : (SimplexOpts){ 0 };
    od.xstart = xd; od.ystart = yd; od.cstat0 = NULL; od.rstat0 = NULL; od.nopresolve = 1; od.no_dual_route = 1; od.nosymmetry = 1;
    SimplexInfo idd; memset(&idd, 0, sizeof idd);
    double *xdo = xmalloc(sizeof(double) * ((size_t)nd + 1));
    int st = simplex_solve(&D, &od, xdo, NULL, NULL, NULL, csd, rsd, &idd);
    free(xdo);
    if (st == 0) {
        int *cs0 = xmalloc(sizeof(int) * ((size_t)n + 1)), *rs0 = xmalloc(sizeof(int) * ((size_t)m + 1));
        sxpre_dual_basis(M, L, csd, rsd, cs0, rs0);
        SimplexOpts op = o ? *o : (SimplexOpts){ 0 };
        op.xstart = NULL; op.ystart = NULL; op.cstat0 = cs0; op.rstat0 = rs0; op.nopresolve = 1; op.nosymmetry = 1;
        if (o && o->timelimit > 0) { op.timelimit = o->timelimit - idd.time; if (op.timelimit <= 0) op.timelimit = 1e-3; }
        st = simplex_solve(L, &op, x, NULL, NULL, NULL, NULL, NULL, info);
        info->pushes += idd.pushes; info->iters += idd.iters; info->time += idd.time; info->dualized = 1;
        free(cs0); free(rs0);
    } else {
        memset(info, 0, sizeof *info); info->status = st; info->pushes = idd.pushes; info->iters = idd.iters; info->time = idd.time;
    }
    free(csd); free(rsd); free(xd); free(yd); free(ymin); lp_free(&D); sxpre_dual_free(M);
    return st;
}
static int sx_direct(const LpProb *L, const double *x, const double *ymin, const int *cs, const int *rs, double tol,
                     double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat, double *objp, double *pinfp, double *dinfp);
/* (13 November 2026) the point with every column at the bound its cost prefers (minimization: the lower bound for a positive
 * cost, the upper for a negative one; a cost of zero: the lower bound if finite, else the upper, else zero) minimizes the cost
 * over the bounds alone; when it satisfies every row within 1e-9 relative it is optimal for the LP, and the basis of the rows'
 * logicals with y = 0 certifies it (every reduced cost is the cost, of the sign its bound needs). One pass over the matrix,
 * ending at the first row or column that fails. neos-3148108-pahi (LP, 154,774 rows; 9 columns of cost 1, every column in
 * [0, 1], x = 0 feasible): the presolve, the symmetry reduction, two more presolves and six factorizations took 0.13 s for 0
 * iterations, where Gurobi's presolve removes every row and column. Returns 1 and fills the outputs as sx_solve_core does
 * (status 0, no iteration, no factorization), or 0 (nothing written). BRISK_SXBOUNDPT=0: off */
int sx_bound_point(const LpProb *L, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat, SimplexInfo *info) {
    static int on = -1; if (on < 0) { const char *e_ = getenv("BRISK_SXBOUNDPT"); on = !(e_ && atoi(e_) == 0); }
    const int m = L->m, n = L->n;
    if (!on || m <= 0 || n <= 0) return 0;
    const double sg = L->maxim ? -1.0 : 1.0;
    double *x = xmalloc(sizeof(double) * ((size_t)n + 1)), *ymin = xcalloc((size_t)m + 1, sizeof(double));
    int *cs = xmalloc(sizeof(int) * ((size_t)n + 1)), *rs = xmalloc(sizeof(int) * ((size_t)m + 1)), ok = 1;
    for (int j = 0; j < n && ok; j++) {
        const double c = sg * L->c[j], lj = L->l[j], uj = L->u[j];
        const int lf = lj > -HUGEB, uf = uj < HUGEB;
        if (lf && uf && lj == uj) { x[j] = lj; cs[j] = SX_FIXED; }
        else if (c > 0 || (c == 0 && lf)) { if (!lf) ok = 0; else { x[j] = lj; cs[j] = SX_LOWER; } }
        else if (c < 0 || uf) { if (!uf) ok = 0; else { x[j] = uj; cs[j] = SX_UPPER; } }
        else { x[j] = 0; cs[j] = SX_ZERO; }
    }
    for (int i = 0; i < m; i++) rs[i] = SX_BASIC;
    double obj = 0, pinf = 0, dinf = 0;
    if (ok) ok = sx_direct(L, x, ymin, cs, rs, 1e-9, xout, axout, yout, zout, cstat, rstat, &obj, &pinf, &dinf);
    free(x); free(ymin); free(cs); free(rs);
    if (!ok) return 0;
    if (info) { memset(info, 0, sizeof *info); info->status = 0; info->obj = obj; info->pinf = pinf; info->dinf = dinf; }
    return 1;
}
/* (27 October 2026) the postsolved solution as the result, without the factorization of the problem as read, when it is
 * optimal there within 1e-9 relative (below the method's 1e-7 on the problem as read, 27.6): every value within its
 * bounds, every row activity within the row's, every nonbasic variable at the bound its status names, and every reduced
 * cost and row dual of the sign its status needs (zero for a basic one). x and ymin as the postsolve gives them (the row
 * duals as minimization), cs and rs the statuses. Returns 1 and fills the outputs as sx_solve_core does, or 0 (then
 * nothing is written). The full step (the basis factorized on the problem as read, x and y recomputed) cost 5 to 15 % of
 * the LPs Gurobi solves in 0.05 to 0.3 s (radiationm18-12-05, LP: 0.03 of 0.20 s) and only confirmed the basis. */
static int sx_direct(const LpProb *L, const double *x, const double *ymin, const int *cs, const int *rs, double tol,
                     double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat, double *objp, double *pinfp, double *dinfp) {
    const int m = L->m, n = L->n;
    const double sg = L->maxim ? -1.0 : 1.0;
    double cmax = 1; for (int j = 0; j < n; j++) cmax = fmax(cmax, fabs(L->c[j]));
    double *ax = xcalloc((size_t)m + 1, sizeof(double)), *d = xmalloc(sizeof(double) * ((size_t)n + 1));
    double pinf = 0, dinf = 0; int ok = 1;
    for (int j = 0; j < n && ok; j++) {
        const double xj = x[j]; double dj = sg * L->c[j];
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { ax[L->Ai[p]] += L->Ax[p] * xj; dj -= L->Ax[p] * ymin[L->Ai[p]]; }
        d[j] = dj;
        const double lj = L->l[j], uj = L->u[j], pt = tol * (1 + fabs(xj)), dt = tol * cmax;
        if (lj > -HUGEB && xj < lj) { pinf = fmax(pinf, lj - xj); if (lj - xj > pt) ok = 0; }
        if (uj < HUGEB && xj > uj) { pinf = fmax(pinf, xj - uj); if (xj - uj > pt) ok = 0; }
        switch (cs[j]) {
        case SX_BASIC: dinf = fmax(dinf, fabs(dj)); if (fabs(dj) > dt) ok = 0; break;
        case SX_LOWER: if (!(lj > -HUGEB) || fabs(xj - lj) > pt) ok = 0; if (dj < 0) { dinf = fmax(dinf, -dj); if (-dj > dt) ok = 0; } break;
        case SX_UPPER: if (!(uj < HUGEB) || fabs(xj - uj) > pt) ok = 0; if (dj > 0) { dinf = fmax(dinf, dj); if (dj > dt) ok = 0; } break;
        case SX_FIXED: if (!(lj > -HUGEB) || fabs(xj - lj) > pt) ok = 0; break;
        default: if (fabs(xj) > pt) ok = 0; dinf = fmax(dinf, fabs(dj)); if (fabs(dj) > dt) ok = 0; break;   /* (SX_ZERO: a free one at zero) */
        }
        if (!ok && getenv("BRISK_SXDIRECTDBG")) printf("   [direct: column %d status %d x %.6g [%.6g, %.6g] d %.6g]\n", j, cs[j], xj, lj, uj, dj);
    }
    for (int i = 0; i < m && ok; i++) {
        const double a = ax[i], rl = L->rl[i], ru = L->ru[i], yi = ymin[i], pt = tol * (1 + fabs(a)), dt = tol * cmax;
        if (rl > -HUGEB && a < rl) { pinf = fmax(pinf, rl - a); if (rl - a > pt) ok = 0; }
        if (ru < HUGEB && a > ru) { pinf = fmax(pinf, a - ru); if (a - ru > pt) ok = 0; }
        /* (a row at its lower bound has y >= 0 as minimization, at its upper bound y <= 0) */
        switch (rs[i]) {
        case SX_BASIC: dinf = fmax(dinf, fabs(yi)); if (fabs(yi) > dt) ok = 0; break;
        case SX_LOWER: if (!(rl > -HUGEB) || fabs(a - rl) > pt) ok = 0; if (yi < 0) { dinf = fmax(dinf, -yi); if (-yi > dt) ok = 0; } break;
        case SX_UPPER: if (!(ru < HUGEB) || fabs(a - ru) > pt) ok = 0; if (yi > 0) { dinf = fmax(dinf, yi); if (yi > dt) ok = 0; } break;
        case SX_FIXED: if (!(rl > -HUGEB) || fabs(a - rl) > pt) ok = 0; break;
        default: if (fabs(yi) > dt) ok = 0; break;
        }
        if (!ok && getenv("BRISK_SXDIRECTDBG")) printf("   [direct: row %d status %d activity %.6g [%.6g, %.6g] y %.6g]\n", i, rs[i], a, rl, ru, yi);
    }
    if (ok) {
        double obj = L->c0;
        for (int j = 0; j < n; j++) { obj += L->c[j] * x[j]; if (xout) xout[j] = x[j]; if (zout) zout[j] = sg * d[j]; if (cstat) cstat[j] = cs[j]; }
        for (int i = 0; i < m; i++) { if (axout) axout[i] = ax[i]; if (yout) yout[i] = sg * ymin[i]; if (rstat) rstat[i] = rs[i]; }
        *objp = obj;
    }
    *pinfp = pinf; *dinfp = dinf;     /* (when refused: up to the first entry that failed) */
    free(ax); free(d);
    return ok;
}
/* (4 November 2026) the postsolved basic solution (x, ymin, statuses: sxpre_postsolve's output) taken when it is optimal on L
 * within 1e-9, as the simplex methods' presolved route does (sx_direct): 1 and x in xout, info's objective and violations;
 * 0 when not (the caller solves L from the basis) */
int simplex_postsolved_check(const LpProb *L, const double *x, const double *ymin, const int *cs, const int *rs, double *xout, SimplexInfo *info) {
    double obj = 0, pinf = 0, dinf = 0;
    const int ok = sx_direct(L, x, ymin, cs, rs, 1e-9, xout, NULL, NULL, NULL, NULL, NULL, &obj, &pinf, &dinf);
    if (ok && info) { memset(info, 0, sizeof *info); info->status = 0; info->obj = obj; info->pinf = pinf; info->dinf = dinf; }
    return ok;
}
/* (27 October 2026, the night) an infeasible or unbounded verdict on the presolved problem is checked on the problem as read,
 * solved without the presolve, whose result is returned (-1: no check, the time limit spent or BRISK_SXNOASREADCHECK=1).
 * The presolve decides in exact arithmetic on data rounded to the digits of the file, and a verdict on its problem carries no
 * certificate for the problem as read. dell (LP, MIPLIB 2017 collection): 0.137848325 for 1/7.25435 makes a row
 * 0.137848325 x107 - x553 >= 0 with x107 = 7.25435 x553 read x553 <= 0 exactly, and the presolve found the LP infeasible;
 * it is feasible within 4e-9, and Gurobi, HiGHS and the simplex method on the problem as read find its optimum -2301.99 */
static int sx_asread_check(const LpProb *L, const SimplexOpts *o, double t0, int st0, double *xout, double *axout, double *yout,
                           double *zout, int *cstat, int *rstat, SimplexInfo *info) {
    static int off = -1; if (off < 0) off = getenv("BRISK_SXNOASREADCHECK") != NULL;
    if (off || (o && brisk_flag_get(o->stop))) return -1;
    /* (28 October 2026) an unbounded verdict stands when a column of the problem as read is a ray by its signs alone */
    if (st0 == 2 && lp_ray_column(L) >= 0) return -1;
    SimplexOpts o3; if (o) o3 = *o; else memset(&o3, 0, sizeof o3);
    o3.nopresolve = 1; o3.orig = NULL;
    const double t1 = now();
    if (o && o->timelimit > 0 && (o3.timelimit = o->timelimit - (t1 - t0)) <= 0) return -1;
    if (o && o->verbose >= 1)
        printf("  the presolved problem is %s: the problem as read is solved without the presolve, to confirm\n", st0 == 1 ? "infeasible" : "unbounded");
    const int st3 = sx_solve_core(L, &o3, xout, axout, yout, zout, cstat, rstat, info);
    info->asread_check = 1; info->st_presolved = st0; info->t_presolved = t1 - t0;
    info->time = now() - t0;
    return st3;
}
static int simplex_solve_in2(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat,
                             int *rstat, SimplexInfo *info);
/* (27 October 2026, the night) the stop flag also for the presolve and the symmetry search, which look at the thread's
 * own (sxpre_set_stop); the caller's flag is restored on return (simplex_solve is called within its own routes) */
static int simplex_solve_in(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat,
                            int *rstat, SimplexInfo *info) {
    const volatile int *prev = sxpre_get_stop();
    sxpre_set_stop(o ? o->stop : NULL);
    const int st = simplex_solve_in2(L, o, xout, axout, yout, zout, cstat, rstat, info);
    sxpre_set_stop(prev);
    return st;
}
static int simplex_solve_in2(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat,
                             int *rstat, SimplexInfo *info) {
    SimplexInfo dummy; if (!info) info = &dummy;
    if ((o && (o->nopresolve || o->cstat0)) || getenv("BRISK_SXNOPRESOLVE")) return sx_solve_core(L, o, xout, axout, yout, zout, cstat, rstat, info);
    const double t0 = now();
    /* (13 November 2026) every column at the bound its cost prefers: when that point satisfies every row it is optimal (it
     * minimizes the cost over the bounds alone), with every row's logical basic and y = 0 (sx_bound_point) */
    if (sx_bound_point(L, xout, axout, yout, zout, cstat, rstat, info)) {
        info->time = now() - t0;
        if (o && o->verbose >= 1) printf("  every column at the bound its cost prefers satisfies every row: optimal, with every row's logical basic\n");
        return 0;
    }
    LpProb R; SxPre *P = NULL; char why[160] = "";
    /* (5 November 2026, an experiment: BRISK_SXAGGBSX=1) the dual or the primal simplex method alone: their presolve makes
     * the bounded substitution of columns without cost itself (simplex_pre.c, g_sxpre_simplex 2), instead of the route
     * below. Better on the satellites (LP; satellites4-25 4,040-5,167 iterations against the route's 5,829-6,557) but it
     * breaks the symmetry the symmetry route uses (chromaticindex512-7, LP: 0 -> 55,661 iterations), makes neos-799716
     * (LP) tall enough for the route through the dual (1,830 -> 21,714) and costs msc98-ip (LP) 9,125 -> 30,495 */
    static int agsx = -1; if (agsx < 0) { const char *e_ = getenv("BRISK_SXAGGBSX"); agsx = e_ && atoi(e_) == 1; }
    const int sfs_ = sxpre_set_for_simplex(agsx && o && (o->method == 1 || o->method == 2) ? 2 : 1);
    const int ps = sxpre_presolve(L, &R, &P, o ? o->verbose >= 1 : 0, why, sizeof why);
    sxpre_set_for_simplex(sfs_);
    const double tpre = now() - t0;
    if (!ps && getenv("BRISK_SXWRITE")) {   /* (experiments: the simplex methods' presolved problem as a free MPS file) */
        FILE *fw = fopen(getenv("BRISK_SXWRITE"), "w");
        if (fw && getenv("BRISK_SXWRITEMAP")) {     /* (experiments) the names of the rows and columns kept, as read */
            FILE *fm = fopen(getenv("BRISK_SXWRITEMAP"), "w");
            const int *rm, *cm; sxpre_maps(P, &rm, &cm);
            if (fm) {
                for (int i = 0; i < R.m; i++) fprintf(fm, "r %d %s\n", i, L->rname && rm ? L->rname[rm[i]] : "?");
                for (int j = 0; j < R.n; j++) fprintf(fm, "c %d %s\n", j, L->cname && cm ? L->cname[cm[j]] : "?");
                fclose(fm);
            }
        }
        if (fw) write_free_mps(&R, fw);
    }
    if (ps) {
        sxpre_free(P);
        if (ps == 6) { memset(info, 0, sizeof *info); info->status = 6; snprintf(info->why, sizeof info->why, "stopped"); info->time = now() - t0; info->t_presolve = tpre; return 6; }
        const int st3 = sx_asread_check(L, o, t0, ps, xout, axout, yout, zout, cstat, rstat, info);
        if (st3 >= 0) { info->t_presolve = tpre; return st3; }
        memset(info, 0, sizeof *info);
        info->status = ps; snprintf(info->why, sizeof info->why, "presolve: %s", why);
        info->time = now() - t0; info->t_presolve = tpre;
        return ps;
    }
    const int m = L->m, n = L->n, mr = R.m, nr = R.n;
    int *cs = malloc(sizeof(int) * (n + 1)), *rs = malloc(sizeof(int) * (m + 1));
    double *x = malloc(sizeof(double) * (n + 1)), *y = malloc(sizeof(double) * (m + 1));
    int postsolved = 0;      /* (x and y hold the postsolve's values: the direct result may be tried, sx_direct) */
    SimplexInfo ir; memset(&ir, 0, sizeof ir);
    int st = 0;
    if (mr > 0 && nr > 0) {
        int *csr = malloc(sizeof(int) * (nr + 1)), *rsr = malloc(sizeof(int) * (mr + 1));
        double *xr = malloc(sizeof(double) * (nr + 1)), *yr = malloc(sizeof(double) * (mr + 1));
        SimplexOpts o2; if (o) o2 = *o; else memset(&o2, 0, sizeof o2);
        o2.nopresolve = 1; o2.direct_peer = NULL; o2.dr_declined = NULL;
        struct SxOut out; sxout_init(&out, L);
        struct SxOrig orig = { L, &out }; o2.orig = &orig;
        int done = 0;
        /* (the time limit is the whole solve's: a route that gives up leaves the next one what is left) */
        /* (27 October 2026, the night) and none once the caller's flag is set: a route stopped by it (sym_route's
         * solve on the classes) reported "not done", and the next one began with a whole set-up of the problem (the
         * default route's join of the dual simplex method waited 25-45 ms on n_quantile_07, LP) */
        #define SX_LEFT() (!(o && brisk_flag_get(o->stop)) && (o && o->timelimit > 0 ? (o2.timelimit = o->timelimit - (now() - t0)) > 0 : 1))
        const int sym = !(o && (o->nosymmetry || o->xstart || o->aggb_only)) && !getenv("BRISK_SXNOSYM");
        int aggb_declined = 0;
        SimplexInfo isymf; memset(&isymf, 0, sizeof isymf);   /* (the counts of a route's attempt that did not give the result) */
        if (o && o->aggb_only) {
            /* (28 October 2026) the bounded substitution's job beside other methods: its route alone, or nothing */
            if (aggb_applies(&R, &o2)) st = aggb_route(&R, &o2, xr, yr, csr, rsr, &ir, &done, &isymf);
            if (!done) aggb_declined = 1;
            done = 1;
        }
        if (sym) st = sym_route(&R, &o2, xr, yr, csr, rsr, &ir, &done, &isymf);
        /* (11 November 2026) not when the presolve changed nothing: the problem as read is then the presolved one, whose
         * symmetry sym_route has just found of no use, and the same test on the same data repeats it (Htransport_10000, LP:
         * two passes of the detection, about a tenth of the solve; the canonical form's test that could see more is off
         * unless BRISK_LPFOLDSCALE) */
        int same = R.m == L->m && R.n == L->n && R.Ap[R.n] == L->Ap[L->n] && R.c0 == L->c0 && R.maxim == L->maxim;
        if (same) {
            const size_t n_ = (size_t)L->n, m_ = (size_t)L->m, z_ = (size_t)L->Ap[L->n];
            same = !memcmp(R.Ap, L->Ap, sizeof(int) * (n_ + 1)) && !memcmp(R.Ai, L->Ai, sizeof(int) * z_) && !memcmp(R.Ax, L->Ax, sizeof(double) * z_)
                && !memcmp(R.c, L->c, sizeof(double) * n_) && !memcmp(R.l, L->l, sizeof(double) * n_) && !memcmp(R.u, L->u, sizeof(double) * n_)
                && !memcmp(R.rl, L->rl, sizeof(double) * m_) && !memcmp(R.ru, L->ru, sizeof(double) * m_);
        }
        if (same && getenv("BRISK_LPFOLDSCALE") && atoi(getenv("BRISK_LPFOLDSCALE")) > 0) same = 0;
        if (!done && SX_LEFT() && sym && !same && !getenv("BRISK_SXNOSYMREAD")) {
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
        if (!done && SX_LEFT() && (!o || o->method != 2) && !(o && o->no_dual_route)) {
            /* (7 November 2026) a declined route tells the job beside it that solves the problem itself (main.c's
             * second simplex job, started on a test of the problem as read): it would repeat this solve */
            int decl = 0; o2.dr_declined = &decl;
#ifdef SX_RACE
            if (frace_on(&R, o)) { o2.dr_declined = NULL; st = frace_run(&R, &o2, xr, yr, csr, rsr, &ir, &done); }
            else
#endif
            st = dual_route(&R, &o2, xr, yr, csr, rsr, &ir, &done, &isymf);
            o2.dr_declined = NULL;
            volatile int *const dp = o ? o->direct_peer : NULL;
            if (decl && dp) *dp = 1;
        }
        /* (5 November 2026) the dual or the primal simplex method alone: the route of the bounded substitution (aggb_route:
         * the problem presolved once more with columns without cost substituted through short equations whose bounds the rows
         * do not imply, solved, and the problem solved from the basis carried back), as the automatic method's second
         * simplex job does; after the symmetry route, which the substitution would break (chromaticindex512-7, LP: 0 ->
         * 55,661 iterations when it is part of the presolve). It is the reduction behind Gurobi's aggregation on the
         * satellites (Gurobi with Aggregate=0: 2,248 -> 15,741 iterations on satellites2-25, LP): one thread, iterations of
         * the four satellites (LP) over three random sequences 6,183-9,002 -> 1,896-2,447 (2-25), 6,226-8,880 -> 1,651-2,338
         * (2-40), 4,444-5,308 -> 2,408-2,834 (2-60-fs), 10,710-15,340 -> 5,829-6,557 (4-25, held out). The substitution's two
         * limits of 5 November 2026 (simplex_pre.c: one bound implied, no long row) took ofi (LP) from over 150 s back to its
         * time without the route (three random sequences: 82,017-82,156 iterations against 83,909-84,371, CPU 1.00-1.07);
         * on 71 other LPs the route touches, iterations 0.876 and CPU time 0.993 of before (geometric means; neos-4971100-bodrog
         * 0.06, modszk1, uct-subprob, b1c1s1 0.40-0.43 of the iterations; the largest loss neos-3572885-glina 38 -> 61
         * iterations, 0.22 -> 0.25 s). It also serves the default command line's dual simplex method beside the
         * interior-point method (satellites2-25 at two threads 1.95 -> 0.69 s). BRISK_SXAGGBDUAL=0: not */
        {   static int agd = -1; if (agd < 0) { const char *e_ = getenv("BRISK_SXAGGBDUAL"); agd = !(e_ && atoi(e_) == 0); }
            if (agd && !agsx && !done && SX_LEFT() && o && (o->method == 1 || o->method == 2) && aggb_applies(&R, &o2)) st = aggb_route(&R, &o2, xr, yr, csr, rsr, &ir, &done, &isymf); }
        if (!done && SX_LEFT()) st = reduced_solve(&R, &o2, xr, yr, csr, rsr, &ir);
        if (!done && o && o->timelimit > 0 && o2.timelimit <= 0) { st = 6; ir.status = 6; }
        if (!done && o && brisk_flag_get(o->stop)) { st = 6; ir.status = 6; }
        #undef SX_LEFT
        /* (the classes' solve that did not end optimal: its iterations counted with those that followed) */
        sxinfo_add(&ir, &isymf);   /* (12 November 2026: with the solves' and the pricing's times) */
        if (aggb_declined) {
            free(csr); free(rsr); free(xr); free(yr); sxout_free(&out); lp_free(&R); sxpre_free(P);
            free(cs); free(rs); free(x); free(y);
            memset(info, 0, sizeof *info); info->status = -1; snprintf(info->why, sizeof info->why, "the bounded substitution does not apply");
            info->time = now() - t0; info->t_presolve = tpre;
            return -1;
        }
        if (out.have) { memcpy(cs, out.cs, sizeof(int) * n); memcpy(rs, out.rs, sizeof(int) * m); }   /* (a basis of the problem as read: no postsolve) */
        else if (st == 0) { sxpre_postsolve(P, xr, yr, csr, rsr, x, y, cs, rs); postsolved = 1; }
        free(csr); free(rsr); free(xr); free(yr); sxout_free(&out);
    } else {
        /* everything removed: the postsolve alone (the reduced problem may keep rows without columns: their logicals) */
        int *csr = malloc(sizeof(int) * (nr + 1)), *rsr = malloc(sizeof(int) * (mr + 1));
        double *xr = calloc(nr + 1, sizeof(double)), *yr = calloc(mr + 1, sizeof(double));
        for (int k = 0; k < nr; k++) { csr[k] = R.l[k] > -1e29 ? SX_LOWER : R.u[k] < 1e29 ? SX_UPPER : SX_ZERO; xr[k] = R.l[k] > -1e29 ? R.l[k] : R.u[k] < 1e29 ? R.u[k] : 0; }
        for (int k = 0; k < mr; k++) rsr[k] = SX_BASIC;
        sxpre_postsolve(P, xr, yr, csr, rsr, x, y, cs, rs); postsolved = 1;
        free(csr); free(rsr); free(xr); free(yr);
    }
    lp_free(&R); sxpre_free(P);
    if (st == 1 || st == 2) {
        const int st3 = sx_asread_check(L, o, t0, st, xout, axout, yout, zout, cstat, rstat, info);
        if (st3 >= 0) { free(cs); free(rs); free(x); free(y); info->t_presolve = tpre; return st3; }
    }
    if (st != 0) {
        *info = ir; info->t_presolve = tpre; info->time = now() - t0;
        free(cs); free(rs); free(x); free(y);
        return st;
    }
    /* the postsolved solution as it is, when optimal on the problem as read within 1e-9 (sx_direct). BRISK_SXDIRECT=0:
     * always the full step; 2: both, and the differences printed (a check of the conventions) */
    static int sxdirect = -1; if (sxdirect < 0) { const char *e = getenv("BRISK_SXDIRECT"); sxdirect = e ? atoi(e) : 1; }
    if (sxdirect && postsolved && !(o && o->ray)) {
        double obj, pinf, dinf;
        double *dx = NULL, *dax = NULL, *dy = NULL, *dz = NULL; int *dcs = NULL, *drs = NULL;
        if (sxdirect == 2) { dx = xmalloc(sizeof(double) * (n + 1)); dax = xmalloc(sizeof(double) * (m + 1)); dy = xmalloc(sizeof(double) * (m + 1)); dz = xmalloc(sizeof(double) * (n + 1)); dcs = xmalloc(sizeof(int) * (n + 1)); drs = xmalloc(sizeof(int) * (m + 1)); }
        const int dok = sx_direct(L, x, y, cs, rs, brisk_lpgrbtol() & LPGRB_SXROUNDS ? SX_GRB_TOL : 1e-9, sxdirect == 2 ? dx : xout, sxdirect == 2 ? dax : axout, sxdirect == 2 ? dy : yout,
                                  sxdirect == 2 ? dz : zout, sxdirect == 2 ? dcs : cstat, sxdirect == 2 ? drs : rstat, &obj, &pinf, &dinf);
        if (o && o->verbose >= 2) printf("  the postsolved solution %s on the problem as read (violations %.1e, %.1e)\n", dok ? "optimal" : "not optimal within 1e-9", pinf, dinf);
        if (dok && sxdirect == 1) {
            *info = ir;
            info->status = 0; info->obj = obj; info->pinf = pinf; info->dinf = dinf; info->have_ray = 0; info->iters_post = 0;
            info->t_presolve = tpre; info->time = now() - t0;
            info->m_reduced = mr; info->n_reduced = nr;
            free(cs); free(rs); free(x); free(y);
            return 0;
        }
        if (sxdirect == 2) {
            SimplexOpts o4; if (o) o4 = *o; else memset(&o4, 0, sizeof o4);
            o4.cstat0 = cs; o4.rstat0 = rs; o4.nopresolve = 1; o4.verbose = 0; o4.timelimit = sx_tleft(o, t0);
            double *fx = xmalloc(sizeof(double) * (n + 1)), *fax = xmalloc(sizeof(double) * (m + 1)), *fy = xmalloc(sizeof(double) * (m + 1)), *fz = xmalloc(sizeof(double) * (n + 1));
            int *fcs = xmalloc(sizeof(int) * (n + 1)), *frs = xmalloc(sizeof(int) * (m + 1)); SimplexInfo i4;
            const int s4 = sx_solve_core(L, &o4, fx, fax, fy, fz, fcs, frs, &i4);
            double ex = 0, ey = 0, ez = 0, ea = 0; int es = 0;
            if (dok) {
                for (int j = 0; j < n; j++) { ex = fmax(ex, fabs(fx[j] - dx[j]) / (1 + fabs(fx[j]))); ez = fmax(ez, fabs(fz[j] - dz[j]) / (1 + fabs(fz[j]))); es += fcs[j] != dcs[j]; }
                for (int i = 0; i < m; i++) { ey = fmax(ey, fabs(fy[i] - dy[i]) / (1 + fabs(fy[i]))); ea = fmax(ea, fabs(fax[i] - dax[i]) / (1 + fabs(fax[i]))); es += frs[i] != drs[i]; }
            }
            printf("  [direct check: %s; full step status %d, %ld iterations, objective %.12e against %.12e; differences x %.1e, ax %.1e, y %.1e, z %.1e, statuses %d]\n",
                   dok ? "accepted" : "refused", s4, i4.iters, i4.obj, dok ? obj : 0.0, ex, ea, ey, ez, es);
            free(fx); free(fax); free(fy); free(fz); free(fcs); free(frs);
            free(dx); free(dax); free(dy); free(dz); free(dcs); free(drs);
        }
    }
    /* the problem as read, from the postsolved basis */
    SimplexOpts o3; if (o) o3 = *o; else memset(&o3, 0, sizeof o3);
    o3.cstat0 = cs; o3.rstat0 = rs; o3.nopresolve = 1; o3.timelimit = sx_tleft(o, t0);
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

/* (7 November 2026; ISSUES 843) with BRISK_LPGRBTOL=1 the methods stop by Gurobi's tolerances on the scaled problem, and
 * one check on the problem as read keeps a plainly wrong result from being reported optimal: a row violated by more than
 * SX_SANITY (1e-3) of its size (1 + |side| + sum_j |a_ij x_j|), a bound by more than SX_SANITY (1 + |bound|), or a reduced
 * cost of the wrong sign by more than SX_SANITY (1 + the largest |c_j|): numerical difficulties. Rounding cannot reach
 * these (the method's tolerances are 1e-6 on the scaled problem); a wrong basis or a lost factorization can */
static double sx_sanity(const LpProb *L, const double *x, double dinf) {
    const int m = L->m, n = L->n;
    double *ax = xcalloc((size_t)m + 1, sizeof(double)), *an = xcalloc((size_t)m + 1, sizeof(double)), e = 0, cmax = 0;
    for (int j = 0; j < n; j++) {
        cmax = fmax(cmax, fabs(L->c[j]));
        if (L->l[j] > -HUGEB) e = fmax(e, (L->l[j] - x[j]) / (1 + fabs(L->l[j])));
        if (L->u[j] < HUGEB) e = fmax(e, (x[j] - L->u[j]) / (1 + fabs(L->u[j])));
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { ax[L->Ai[p]] += L->Ax[p] * x[j]; an[L->Ai[p]] += fabs(L->Ax[p] * x[j]); }
    }
    for (int i = 0; i < m; i++) {
        if (L->rl[i] > -HUGEB) e = fmax(e, (L->rl[i] - ax[i]) / (1 + fabs(L->rl[i]) + an[i]));
        if (L->ru[i] < HUGEB) e = fmax(e, (ax[i] - L->ru[i]) / (1 + fabs(L->ru[i]) + an[i]));
    }
    free(ax); free(an);
    return fmax(e, dinf / (1 + cmax));
}
static double *xoutpresent(const double *out, int k) { return out ? xmalloc(sizeof(double) * ((size_t)k + 1)) : NULL; }
/* (14 November 2026, an experiment: BRISK_SXCANON=1) the problem solved in an order of its rows and columns that does not depend on
 * the order it was written in: the rows sorted by their sides, their length and their sorted coefficients, then the columns
 * lexicographically by the positions of their rows in that order (a stable sort: columns alike keep their order). Satellites4-25 (LP)
 * is highly symmetric, and the method's choices among equal candidates follow the index: in the file the interchangeable columns come
 * in a structured order and the bases stay sparse, in a shuffled copy they fill (the dual simplex method 37-41 s against 4 s; the copy
 * so ordered 9.3 s, with the norm perturbation 4.5 s, Gurobi 4.4 s; bench/lp_r16/pertunits/micro15.txt). Not when a basis, a point or
 * the presolve's maps are given (they are by the order as given) */
static _Thread_local const LpProb *cn_L; static _Thread_local const int *cn_rp, *cn_rlen; static _Thread_local const double *cn_rv;
static _Thread_local const int *cn_cp, *cn_cv;
static int cn_rcmp(const void *a_, const void *b_) {
    const int a = *(const int *)a_, b = *(const int *)b_; const LpProb *L = cn_L;
    if (L->rl[a] != L->rl[b]) return L->rl[a] < L->rl[b] ? -1 : 1;
    if (L->ru[a] != L->ru[b]) return L->ru[a] < L->ru[b] ? -1 : 1;
    if (cn_rlen[a] != cn_rlen[b]) return cn_rlen[a] < cn_rlen[b] ? -1 : 1;
    for (int k = 0; k < cn_rlen[a]; k++) { const double x = cn_rv[cn_rp[a] + k], y = cn_rv[cn_rp[b] + k]; if (x != y) return x < y ? -1 : 1; }
    return a < b ? -1 : a > b;
}
static int cn_ccmp(const void *a_, const void *b_) {
    const int a = *(const int *)a_, b = *(const int *)b_;
    const int la = cn_cp[a + 1] - cn_cp[a], lb = cn_cp[b + 1] - cn_cp[b];
    for (int k = 0; k < la && k < lb; k++) { const int x = cn_cv[cn_cp[a] + k], y = cn_cv[cn_cp[b] + k]; if (x != y) return x < y ? -1 : 1; }
    if (la != lb) return la < lb ? -1 : 1;
    return a < b ? -1 : a > b;
}
/* (the refinement rounds) rows by their signature's group, then lexicographically by their columns' positions, ties by the row's
 * position before; columns by their rows' positions, ties by the column's position before */
static _Thread_local const int *cn_gid, *cn_qpos, *cn_rcp, *cn_rcv, *cn_ppos;
static int cn_rcmp2(const void *a_, const void *b_) {
    const int a = *(const int *)a_, b = *(const int *)b_;
    if (cn_gid[a] != cn_gid[b]) return cn_gid[a] < cn_gid[b] ? -1 : 1;
    const int la = cn_rcp[a + 1] - cn_rcp[a], lb = cn_rcp[b + 1] - cn_rcp[b];
    for (int k = 0; k < la && k < lb; k++) { const int x = cn_rcv[cn_rcp[a] + k], y = cn_rcv[cn_rcp[b] + k]; if (x != y) return x < y ? -1 : 1; }
    if (la != lb) return la < lb ? -1 : 1;
    return cn_qpos[a] < cn_qpos[b] ? -1 : cn_qpos[a] > cn_qpos[b];
}
static int cn_ccmp2(const void *a_, const void *b_) {
    const int a = *(const int *)a_, b = *(const int *)b_;
    const int la = cn_cp[a + 1] - cn_cp[a], lb = cn_cp[b + 1] - cn_cp[b];
    for (int k = 0; k < la && k < lb; k++) { const int x = cn_cv[cn_cp[a] + k], y = cn_cv[cn_cp[b] + k]; if (x != y) return x < y ? -1 : 1; }
    if (la != lb) return la < lb ? -1 : 1;
    return cn_ppos[a] < cn_ppos[b] ? -1 : cn_ppos[a] > cn_ppos[b];
}
static int cn_icmp(const void *a_, const void *b_) { const int a = *(const int *)a_, b = *(const int *)b_; return a < b ? -1 : a > b; }
static int cn_dcmp(const void *a_, const void *b_) { const double a = *(const double *)a_, b = *(const double *)b_; return a < b ? -1 : a > b; }
/* (14 November 2026, an experiment: BRISK_SXRCM=1 with BRISK_SXCANON=1) the rows in reverse Cuthill-McKee order on the pattern of
 * A A' (rows that share a column are neighbours): a breadth-first search from a row of the lowest degree in each connected part, each
 * row's unvisited neighbours taken in increasing degree (ties by index), the order reversed at the end. The search scans each column
 * once (all its rows are reached then), so its cost is that of the entries whatever the dense columns. The degree of a row is the
 * count of entries of its columns less its own length (an upper bound of its degree in A A'). Satellites4-25 (LP) and its shuffled
 * copies so ordered by scipy, then the columns lexicographically by the rows: 0.9-4.5 s against 36-44 s as shuffled, 3.6 s as written
 * (bench/lp_r16/pertunits/micro18, micro19): rows that are near in the problem's structure come together, as in the file's own order */
static _Thread_local const long *cn_deg;
static int cn_degcmp(const void *a_, const void *b_) { const int a = *(const int *)a_, b = *(const int *)b_; if (cn_deg[a] != cn_deg[b]) return cn_deg[a] < cn_deg[b] ? -1 : 1; return a < b ? -1 : a > b; }
static void cn_rcm_rows(const LpProb *L, int *q) {
    const int m = L->m, n = L->n, nz = L->Ap[n];
    int *rp = xcalloc((size_t)m + 2, sizeof(int)), *rj = xmalloc(sizeof(int) * ((size_t)nz + 1)), *f = xmalloc(sizeof(int) * ((size_t)m + 1));
    for (int p = 0; p < nz; p++) rp[L->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    memcpy(f, rp, sizeof(int) * (size_t)m);
    for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) rj[f[L->Ai[p]]++] = j;   /* (each row's columns in increasing order) */
    long *deg = xmalloc(sizeof(long) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { long d = 0; for (int t = rp[i]; t < rp[i + 1]; t++) d += L->Ap[rj[t] + 1] - L->Ap[rj[t]]; deg[i] = d - (rp[i + 1] - rp[i]); }
    int *byd = xmalloc(sizeof(int) * ((size_t)m + 1)); for (int i = 0; i < m; i++) byd[i] = i;
    cn_deg = deg; qsort(byd, (size_t)m, sizeof(int), cn_degcmp);
    unsigned char *vis = xcalloc((size_t)m + 1, 1), *cscan = xcalloc((size_t)n + 1, 1);
    int nq = 0;
    for (int s0 = 0; s0 < m; s0++) {
        const int st = byd[s0]; if (vis[st]) continue;
        vis[st] = 1; int head = nq; q[nq++] = st;
        while (head < nq) {
            const int i = q[head++]; const int k0 = nq;
            for (int t = rp[i]; t < rp[i + 1]; t++) { const int j = rj[t]; if (cscan[j]) continue; cscan[j] = 1;
                for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { const int r = L->Ai[p]; if (!vis[r]) { vis[r] = 1; q[nq++] = r; } } }
            if (nq - k0 > 1) qsort(q + k0, (size_t)(nq - k0), sizeof(int), cn_degcmp);
        }
    }
    for (int a = 0, b = m - 1; a < b; a++, b--) { const int t = q[a]; q[a] = q[b]; q[b] = t; }
    free(rp); free(rj); free(f); free(deg); free(byd); free(vis); free(cscan);
}
static int simplex_solve_body(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                              SimplexInfo *info);
static int simplex_solve_canon(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                               SimplexInfo *info) {
    const int m = L->m, n = L->n, nz = L->Ap[n];
    const double t0_ = now();
    /* the rows' sorted coefficients */
    int *rp = xcalloc((size_t)m + 2, sizeof(int)); double *rv = xmalloc(sizeof(double) * ((size_t)nz + 1));
    for (int p = 0; p < nz; p++) rp[L->Ai[p] + 1]++;
    for (int i = 0; i < m; i++) rp[i + 1] += rp[i];
    int *fill = xmalloc(sizeof(int) * ((size_t)m + 1)); memcpy(fill, rp, sizeof(int) * (size_t)m);
    for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) rv[fill[L->Ai[p]]++] = L->Ax[p];
    int *rlen = xmalloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) { rlen[i] = rp[i + 1] - rp[i]; qsort(rv + rp[i], (size_t)rlen[i], sizeof(double), cn_dcmp); }
    int *q = xmalloc(sizeof(int) * ((size_t)m + 1)), *qi = xmalloc(sizeof(int) * ((size_t)m + 1));
    for (int i = 0; i < m; i++) q[i] = i;
    cn_L = L; cn_rp = rp; cn_rv = rv; cn_rlen = rlen;
    static int rcm_ = -1; if (rcm_ < 0) { const char *e_ = getenv("BRISK_SXRCM"); rcm_ = e_ ? atoi(e_) : 0; }
    if (rcm_) cn_rcm_rows(L, q);    /* (BRISK_SXRCM=1: the rows in reverse Cuthill-McKee order) */
    else qsort(q, (size_t)m, sizeof(int), cn_rcmp);
    for (int k = 0; k < m; k++) qi[q[k]] = k;
    /* the columns by their rows' new positions */
    int *cv = xmalloc(sizeof(int) * ((size_t)nz + 1));
    for (int j = 0; j < n; j++) { for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) cv[p] = qi[L->Ai[p]]; qsort(cv + L->Ap[j], (size_t)(L->Ap[j + 1] - L->Ap[j]), sizeof(int), cn_icmp); }
    int *pc = xmalloc(sizeof(int) * ((size_t)n + 1));
    for (int j = 0; j < n; j++) pc[j] = j;
    cn_cp = L->Ap; cn_cv = cv;
    if (rcm_ != 2) qsort(pc, (size_t)n, sizeof(int), cn_ccmp);   /* (BRISK_SXRCM=2: the rows reordered, the columns kept in their order) */
    {   /* (BRISK_SXCANON=k, k >= 2) k - 1 refinement rounds: the rows of one signature reordered by their columns' positions, then the
         * columns by the rows' new positions (as the symmetry detection refines its classes); what stays tied is interchangeable */
        static int rounds_ = -1; if (rounds_ < 0) { const char *e_ = getenv("BRISK_SXCANON"); rounds_ = e_ ? atoi(e_) : 1; }
        if (rounds_ >= 2) {
            int *gid = xmalloc(sizeof(int) * ((size_t)m + 1)), *ppos = xmalloc(sizeof(int) * ((size_t)n + 1));
            int g_ = 0;
            for (int k = 0; k < m; k++) {
                if (k > 0) { const int a = q[k - 1], b = q[k]; int same = L->rl[a] == L->rl[b] && L->ru[a] == L->ru[b] && rlen[a] == rlen[b];
                    for (int t = 0; same && t < rlen[a]; t++) same = rv[rp[a] + t] == rv[rp[b] + t];
                    if (!same) g_++; }
                gid[q[k]] = g_;
            }
            /* the rows' columns (row-wise pattern) */
            int *rcp = xcalloc((size_t)m + 2, sizeof(int)), *rcv = xmalloc(sizeof(int) * ((size_t)nz + 1)), *rf = xmalloc(sizeof(int) * ((size_t)m + 1));
            for (int p = 0; p < nz; p++) rcp[L->Ai[p] + 1]++;
            for (int i = 0; i < m; i++) rcp[i + 1] += rcp[i];
            for (int r_ = 1; r_ < rounds_; r_++) {
                for (int k = 0; k < n; k++) ppos[pc[k]] = k;
                memcpy(rf, rcp, sizeof(int) * (size_t)m);
                for (int j = 0; j < n; j++) for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) rcv[rf[L->Ai[p]]++] = ppos[j];
                for (int i = 0; i < m; i++) qsort(rcv + rcp[i], (size_t)(rcp[i + 1] - rcp[i]), sizeof(int), cn_icmp);
                cn_gid = gid; cn_qpos = qi; cn_rcp = rcp; cn_rcv = rcv;
                qsort(q, (size_t)m, sizeof(int), cn_rcmp2);
                for (int k = 0; k < m; k++) qi[q[k]] = k;
                for (int j = 0; j < n; j++) { for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) cv[p] = qi[L->Ai[p]]; qsort(cv + L->Ap[j], (size_t)(L->Ap[j + 1] - L->Ap[j]), sizeof(int), cn_icmp); }
                cn_ppos = ppos;
                qsort(pc, (size_t)n, sizeof(int), cn_ccmp2);
            }
            free(gid); free(ppos); free(rcp); free(rcv); free(rf);
        }
    }
    free(cv); free(fill); free(rlen); free(rv); free(rp);
    /* the problem in that order */
    LpProb P = *L;
    P.Ap = xmalloc(sizeof(int) * ((size_t)n + 1)); P.Ai = xmalloc(sizeof(int) * ((size_t)nz + 1)); P.Ax = xmalloc(sizeof(double) * ((size_t)nz + 1));
    P.c = xmalloc(sizeof(double) * ((size_t)n + 1)); P.l = xmalloc(sizeof(double) * ((size_t)n + 1)); P.u = xmalloc(sizeof(double) * ((size_t)n + 1));
    P.rl = xmalloc(sizeof(double) * ((size_t)m + 1)); P.ru = xmalloc(sizeof(double) * ((size_t)m + 1));
    P.rname = L->rname ? xmalloc(sizeof(char *) * ((size_t)m + 1)) : NULL; P.cname = L->cname ? xmalloc(sizeof(char *) * ((size_t)n + 1)) : NULL;
    int e = 0;
    for (int k = 0; k < n; k++) {
        const int j = pc[k]; P.Ap[k] = e;
        for (int p = L->Ap[j]; p < L->Ap[j + 1]; p++) { P.Ai[e] = qi[L->Ai[p]]; P.Ax[e] = L->Ax[p]; e++; }
        P.c[k] = L->c[j]; P.l[k] = L->l[j]; P.u[k] = L->u[j]; if (P.cname) P.cname[k] = L->cname[j];
    }
    P.Ap[n] = e;
    for (int k = 0; k < m; k++) { P.rl[k] = L->rl[q[k]]; P.ru[k] = L->ru[q[k]]; if (P.rname) P.rname[k] = L->rname[q[k]]; }
    /* the rows' entries in increasing order within each column (the solvers expect sorted columns) */
    for (int k = 0; k < n; k++) for (int a = P.Ap[k] + 1; a < P.Ap[k + 1]; a++) { const int ti = P.Ai[a]; const double tx = P.Ax[a]; int b = a - 1;
        while (b >= P.Ap[k] && P.Ai[b] > ti) { P.Ai[b + 1] = P.Ai[b]; P.Ax[b + 1] = P.Ax[b]; b--; } P.Ai[b + 1] = ti; P.Ax[b + 1] = tx; }
    const double tcanon_ = now() - t0_;
    SimplexOpts o2; if (o) o2 = *o; else memset(&o2, 0, sizeof o2);
    double *ray2 = o && o->ray ? xcalloc((size_t)m + 1, sizeof(double)) : NULL; o2.ray = ray2;
    o2.canon = NULL;
    double *x2 = xoutpresent(xout, n), *ax2 = xoutpresent(axout, m), *y2 = xoutpresent(yout, m), *z2 = xoutpresent(zout, n);
    int *cs2 = cstat ? xmalloc(sizeof(int) * ((size_t)n + 1)) : NULL, *rs2 = rstat ? xmalloc(sizeof(int) * ((size_t)m + 1)) : NULL;
    const int st = simplex_solve_body(&P, &o2, x2, ax2, y2, z2, cs2, rs2, info);
    for (int k = 0; k < n; k++) { const int j = pc[k]; if (xout) xout[j] = x2[k]; if (zout) zout[j] = z2[k]; if (cstat) cstat[j] = cs2[k]; }
    for (int k = 0; k < m; k++) { const int i = q[k]; if (axout) axout[i] = ax2[k]; if (yout) yout[i] = y2[k]; if (rstat) rstat[i] = rs2[k]; if (ray2) o->ray[i] = ray2[k]; }
    free(x2); free(ax2); free(y2); free(z2); free(cs2); free(rs2); free(ray2);
    free(P.Ap); free(P.Ai); free(P.Ax); free(P.c); free(P.l); free(P.u); free(P.rl); free(P.ru); free(P.rname); free(P.cname);
    free(q); free(qi); free(pc);
    if (o && o->verbose >= 1) printf("  the problem solved in the order of its rows and columns by their entries (BRISK_SXCANON; the order %.3f s)\n", tcanon_);
    return st;
}
int simplex_solve(const LpProb *L, const SimplexOpts *o, double *xout, double *axout, double *yout, double *zout, int *cstat, int *rstat,
                  SimplexInfo *info) {
    static int canon_ = -1; if (canon_ < 0) { const char *e_ = getenv("BRISK_SXCANON"); canon_ = e_ && atoi(e_) > 0; }
    const int use_canon = canon_ && !(o && (o->cstat0 || o->rstat0 || o->xstart || o->ystart || o->orig));
    const int st = use_canon ? simplex_solve_canon(L, o, xout, axout, yout, zout, cstat, rstat, info)
                             : simplex_solve_body(L, o, xout, axout, yout, zout, cstat, rstat, info);
    if (st != 0 || !xout || !(brisk_lpgrbtol() & LPGRB_SXROUNDS)) return st;
    const double e = sx_sanity(L, xout, info->dinf);
    if (!(e <= SX_SANITY)) {
        info->status = 4; snprintf(info->why, sizeof info->why, "the result violates the problem as read by %.1e", e);
        if (o && o->verbose >= 1) printf("  the result violates the problem as read by %.1e (beyond any rounding): numerical difficulties\n", e);
        return 4;
    }
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
    int xinc;                          /* (11 November 2026) the values in S->x are those of the last optimal solve's final check, for
                                          the current basis and factorization: a change of bounds alone leaves them so (the nonbasic
                                          values move only in the next solve, which then updates the basic values by the moves); a
                                          given basis, a new factorization outside a solve or a trial undone clears it */
    double *ray, *pray;                /* the Farkas multipliers (m, as given) and the unbounded direction (N, scaled) */
    int have_farkas, have_pray, raysign;
    int cutoff;                        /* the last solve stopped at the cutoff (sxm_objective gives the bound then) */
    signed char *isint;                /* (7 November 2026) the integer columns given by sxm_set_integer (n), NULL: none */
    signed char *hint;                 /* (27 October 2026) the side a basis given by sxm_set_basis put a variable at that was
                                          fixed then (+1 its lower bound, -1 its upper, 0 none): taken when a later bound
                                          change frees it (the MIP developer's ISSUES 542) */
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
    M->hint = xcalloc((size_t)S->N + 1, 1);
    S->rayout = M->ray; S->prayout = M->pray;
    slack_basis(S);
    return M;
}
void sxm_free(SxModel *M) {
    if (!M) return;
    sx_free(&M->S); free(M->ray); free(M->pray); free(M->hint); free(M->isint); free(M);
}
void sxm_set_integer(SxModel *M, const signed char *isint) {
    Sx *S = &M->S;
    free(M->isint); M->isint = NULL; S->isint = NULL;
    if (!isint) return;
    int any = 0; for (int j = 0; j < S->n; j++) if (isint[j]) { any = 1; break; }
    if (!any) return;
    M->isint = xmalloc((size_t)S->n + 1);
    for (int j = 0; j < S->n; j++) M->isint[j] = isint[j] ? 1 : 0;
    S->isint = M->isint;
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
    M->cost_changed = 1; M->dok = 0; S->pert_u_ok = 0;
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
int sxm_set_basis(SxModel *M, const int *cstat, const int *rstat) { return sxm_set_basis_w(M, cstat, rstat, 0); }
int sxm_set_basis_w(SxModel *M, const int *cstat, const int *rstat, int wmode) {
    Sx *S = &M->S;
    sxm_reset_work(S);
    given_basis(S, cstat, rstat);
    /* a nonbasic variable fixed now keeps the side the basis names (given_basis gives it none): a bound change before the
     * next solve that frees it puts it there, as lpmodel.h promises (the basis kept across changes). Before, the solve
     * placed it by the reduced cost of the basis before this one (the MIP developer's ISSUES 542: 871-4,287 iterations
     * from a basis optimal for the same bounds) */
    for (int j = 0; j < S->N; j++) {
        M->hint[j] = 0;
        if (S->pos[j] >= 0 || S->lo[j] != S->up[j]) continue;
        const int st = j < S->n ? cstat[j] : rstat[j - S->n];
        int want = st == SX_LOWER ? 1 : st == SX_UPPER ? -1 : 0;
        if (j >= S->n) want = -want;            /* (the logical s = -Ax: the row's bounds reversed) */
        M->hint[j] = (signed char)want;
    }
    const int nrep = refactor(S);
    M->factored = 1; M->xok = M->dok = 0; M->xinc = 0;
    /* (27 October 2026, an experiment: BRISK_SXMEXACTW=1) the exact dual steepest-edge weights of the given basis, the
     * squared norms of the rows of its inverse (one backward solve a row), instead of the slack basis's 1 that
     * given_basis sets: the MIP developer's node LP of csched008 (LP) took 550 iterations from its parent's basis in a new
     * LP object, Gurobi's dual simplex 111 from the same basis; with the exact weights 240 (set and solve 0.022 -> 0.009 s),
     * supportcase18 164 -> 85, roll3000 197 -> 139, neos-4722843-widden 402 -> 308 iterations but 0.32 -> 0.63 s (one
     * backward solve for each of its 111,067 rows). Off until the weights are computed lazily, a row's the first time it
     * is a candidate to leave (IDEAS) */
    /* (28 October 2026) the caller's choice (LmOpts.basis_weights, the MIP developer's request: lazy weights for the
     * mixed-integer solver's node LPs, the slack basis's 1 for plain LPs); 0: the environment's experiments as before */
    static int exw = -1; if (exw < 0) exw = getenv("BRISK_SXMEXACTW") && atoi(getenv("BRISK_SXMEXACTW")) == 1;
    static int lazyw = -1; if (lazyw < 0) lazyw = getenv("BRISK_SXMLAZYW") && atoi(getenv("BRISK_SXMLAZYW")) == 1;
    const int wm = wmode == 1 || wmode == 2 ? wmode : exw ? 2 : lazyw ? 1 : 0;
    if (wm == 2) {
        SxVec *rho = &S->rho;
        for (int i = 0; i < S->m; i++) {
            sxv_clear(rho); rho->val[i] = 1; rho->idx[0] = i; rho->cnt = 1;
            double dd = S->dens_rho; btran(S, rho, &dd, NULL);
            double w = 0;
            if (rho->cnt >= 0) for (int t = 0; t < rho->cnt; t++) { const double v = rho->val[rho->idx[t]]; w += v * v; }
            else for (int k = 0; k < S->m; k++) w += rho->val[k] * rho->val[k];
            S->w[i] = w > 1e-4 ? w : 1e-4;
        }
        sxv_clear(rho);
    }
    /* (27 October 2026, an experiment: BRISK_SXMLAZYW=1) the same weights computed lazily: each marked unknown, and a
     * candidate's computed when it comes to the top of the leaving rows' heap (chuzr). The same iterations (csched008 240,
     * supportcase18 85, neos-4722843-widden 308; roll3000 147) at the setting's usual cost, but widden 0.40-0.46 ->
     * 0.51-0.54 s: with the unknown weights at 1 most of its infeasible rows come to the top and are computed */
    if (wm == 1) {
        /* the stand-in for an unknown weight: a lower bound, so that the choice is the one of the exact weights (an unknown
         * row's key is then at least its true one, and it is computed before it is chosen). Row i of B^{-1} has inner
         * product 1 with the basic column b_i, so |row|^2 >= 1 / |b_i|^2: 1 for a row's logical, at most 1 here. The
         * update then changes the stand-ins too; keeping them out of it changed little (csched008, LP: 240 -> 258
         * iterations, roll3000 147 either way against 139 with all exact) */
        memset(S->wk, 0, (size_t)S->m); S->wlazy = 1;
        for (int i = 0; i < S->m; i++) {
            const int j = S->basic[i];
            if (j >= S->n) continue;
            double c2 = 0; for (int p = S->Ap[j]; p < S->Ap[j + 1]; p++) c2 += S->Ax[p] * S->Ax[p];
            S->w[i] = c2 > 1 ? fmax(1.0 / c2, 1e-4) : 1.0;
        }
    }
    return nrep;
}
/* (7 November 2026; BRISK_SXPOLISH=1, off by default; only with integer flags given, sxm_set_integer) the LP solution
 * polishing of SoPlex (Gleixner and others): at an optimal basis, pivots that keep it optimal and take integer columns out of
 * the basis. A nonbasic row logical or continuous column of reduced cost within the dual tolerance enters by a primal step
 * (an exact ratio test, the step of the least ratio; among exact ties an integer column leaves, then the largest |alpha|)
 * when the variable that leaves is an integer column (it goes to a bound, integral); a pivot that would put another
 * reduced cost beyond the tolerance, one whose pivot is below 1e-7 or a hundredth of its column's largest entry, a bound
 * flip of the entering variable and an unbounded step are not made. The objective does not change (the reduced cost of
 * the entering variable is zero up to the tolerance), every value stays within its bounds, and the basis has one integer
 * column fewer. Passes over the candidates in the order of their indices (BRISK_SXPOLISH=k: at most k passes, until one
 * makes no pivot); then a new factorization and the values. On
 * a relaxation with a zero objective every candidate has reduced cost zero. Returns the pivots made */
/* one polishing attempt: q enters by a primal step when the variable that leaves is an integer column and every reduced cost
 * stays within the tolerance (see polish_int); returns 1 when the pivot was made */
static int polish_try(Sx *S, int q) {
    const int n = S->n;
    SxVec *col = &S->col, *rho = &S->rho, *tau = &S->tau;
    const double dir = (double)S->move[q];
    sxv_clear(col); col_scatter(S, q, col, 1.0);
    ftran(S, col, &S->dens_col, &S->spike);
    if (col->cnt < 0) sxv_reindex(col);
    double tmin = SX_INF, amax = 0; int r = -1, rint = 0; double ar_ = 0, bnd = 0;
    for (int k = 0; k < col->cnt; k++) { const double a = fabs(col->val[col->idx[k]]); if (a > amax) amax = a; }
    for (int k = 0; k < col->cnt; k++) {
        const int i = col->idx[k];
        const double a = dir * col->val[i];
        if (fabs(a) < 1e-9) continue;
        const int j = S->basic[i];
        const double v = S->x[j];
        double t; double b;
        if (a > 0) { if (S->lo[j] <= -SX_INF) continue; t = (v - S->lo[j]) / a; b = S->lo[j]; }
        else { if (S->up[j] >= SX_INF) continue; t = (S->up[j] - v) / -a; b = S->up[j]; }
        if (t < 0) t = 0;
        const int isi = j < n && S->isint[j];
        const int better = t < tmin * (1 - 1e-12) - 1e-15 ||
            (!(t > tmin * (1 + 1e-12) + 1e-15) && (isi > rint || (isi == rint && fabs(a) > fabs(ar_))));
        if (r < 0 || better) { tmin = t; r = i; rint = isi; ar_ = a; bnd = b; }
    }
    if (r < 0 || !rint) { sxv_clear(col); return 0; }
    if (S->up[q] - S->lo[q] <= tmin) { sxv_clear(col); return 0; }            /* (a bound flip of q first) */
    const double ar = col->val[r];
    if (fabs(ar) < 1e-7 || fabs(ar) < 1e-2 * amax) { sxv_clear(col); return 0; }
    /* the pivotal row: the reduced costs after the pivot must stay within the tolerance */
    sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
    btran(S, rho, &S->dens_rho, &S->upart);
    if (rho->cnt < 0) sxv_reindex(rho);
    price(S, rho);
    const double thd = S->d[q] / ar;
    int ok = 1;
    if (thd != 0) for (int k = 0; k < S->row.cnt && ok; k++) {
        const int j = S->row.idx[k]; if (j == q || S->lo[j] == S->up[j]) continue;
        const double dn = S->d[j] - thd * S->row.val[j];
        if (S->move[j] == 0 ? fabs(dn) > S->told : S->move[j] * dn < -S->told) ok = 0;
    }
    const int jl = S->basic[r];
    if (ok && thd != 0) { const double dl = -thd; const int mvl = S->lo[jl] == S->up[jl] ? 0 : bnd == S->lo[jl] ? 1 : -1; if (mvl * dl < -S->told) ok = 0; }
    if (!ok) { sxv_clear(col); sxv_clear(rho); return 0; }
    double wr = 0; for (int k = 0; k < rho->cnt; k++) wr += rho->val[rho->idx[k]] * rho->val[rho->idx[k]];
    sxv_copy(tau, rho);
    { double dd = S->dens_tau; ftran(S, tau, &dd, NULL); if (tau->cnt < 0) sxv_reindex(tau); }
    update_dse(S, r, col, tau, wr);
    if (S->wlazy) S->wk[r] = 1;
    for (int k = 0; k < S->row.cnt; k++) { const int j = S->row.idx[k]; S->d[j] -= thd * S->row.val[j]; }
    S->d[q] = 0; S->d[jl] = -thd;
    const double step = tmin * dir;
    for (int k = 0; k < col->cnt; k++) { const int i = col->idx[k]; S->x[S->basic[i]] -= step * col->val[i]; }
    S->x[q] += step;
    S->x[jl] = bnd;
    S->basic[r] = q; S->pos[q] = r; S->move[q] = 0; S->pos[jl] = -1;
    rows_enter(S, q); rows_leave(S, jl);
    S->move[jl] = S->lo[jl] == S->up[jl] ? 0 : bnd == S->lo[jl] ? 1 : -1;
    const int bad = sxlu_update(&S->F, r, &S->spike, &S->upart, ar);
    sxv_clear(col); sxv_clear(rho); sxv_clear(tau); sxv_clear(&S->row);
    if (bad || need_refactor(S)) { refactor(S); compute_primal_dual(S); }
    return 1;
}
static long polish_int(Sx *S) {
    if (!S->isint) return 0;
    const int n = S->n, N = S->N;
    long npiv = 0, ntry = 0;
    static int npass = -1; if (npass < 0) { const char *e = getenv("BRISK_SXPOLISH"); npass = e ? atoi(e) : 1; if (npass < 1) npass = 1; }
    /* (8 November 2026, BRISK_SXPOLISHCOST=bits, off: 0; the MIP developer's measurement: polishing on every node LP doubled
     * a node's cost on neos-3734794-moppy, MILP, where most nonbasic variables have reduced cost zero and each costs a
     * forward solve while few pivots are made) 1: a budget, the attempts at most the solve's own iterations (and 50);
     * 2: from the integer side when the basic integer columns are fewer than the candidates: for each basic integer column
     * its row of the tableau (one backward solve and a priced row), and only the candidates with an entry in that row tried,
     * the largest entries first, at most four a row; either side makes the same kind of pivot */
    static int pcost = -1; if (pcost < 0) { const char *e = getenv("BRISK_SXPOLISHCOST"); pcost = e ? atoi(e) : 0; }
    const long budget = pcost & 1 ? (S->iter > 50 ? S->iter : 50) : LONG_MAX;
    #define POLISH_CAND(q) (S->pos[q] < 0 && S->lo[q] != S->up[q] && S->move[q] != 0 && !((q) < n && S->isint[q]) && fabs(S->d[q]) <= S->told)
    int byrow = 0;
    if (pcost & 2) {
        long nbint = 0, ncand = 0;
        for (int i = 0; i < S->m; i++) { const int j = S->basic[i]; if (j < n && S->isint[j] && (fabs(S->x[j] - S->lo[j]) <= S->tolp * (1 + fabs(S->lo[j])) || fabs(S->x[j] - S->up[j]) <= S->tolp * (1 + fabs(S->up[j])))) nbint++; }
        for (int q = 0; q < N; q++) if (POLISH_CAND(q)) ncand++;
        byrow = nbint < ncand;
    }
    if (byrow) {
        const int m = S->m;
        int *rows = xmalloc(sizeof(int) * ((size_t)m + 1)), *cq = xmalloc(sizeof(int) * ((size_t)N + 1)); double *ca = xmalloc(sizeof(double) * ((size_t)N + 1));
        /* (only the rows whose integer column sits at a bound: on a degenerate vertex the step is zero, and the integer row wins
         * the ties of the ratio test; a column inside its bounds leaves only when its ratio is the least of all, which a
         * degenerate vertex's other rows at zero steps prevent) */
        int nr = 0; for (int i = 0; i < m; i++) { const int j = S->basic[i];
            if (j < n && S->isint[j] && (fabs(S->x[j] - S->lo[j]) <= S->tolp * (1 + fabs(S->lo[j])) || fabs(S->x[j] - S->up[j]) <= S->tolp * (1 + fabs(S->up[j])))) rows[nr++] = i; }
        SxVec *rho = &S->rho;
        for (int t = 0; t < nr && ntry < budget; t++) {
            const int r = rows[t], jb = S->basic[r];
            if (!(jb < n && S->isint[jb])) continue;           /* (an earlier pivot changed this row's basic variable) */
            sxv_clear(rho); rho->val[r] = 1; rho->idx[0] = r; rho->cnt = 1;
            { double dd = S->dens_rho; btran(S, rho, &dd, NULL); }
            if (rho->cnt < 0) sxv_reindex(rho);
            price(S, rho);
            int nc = 0;
            for (int k = 0; k < S->row.cnt; k++) { const int q = S->row.idx[k]; const double a = fabs(S->row.val[q]); if (a >= 1e-7 && POLISH_CAND(q)) { cq[nc] = q; ca[nc] = a; nc++; } }
            sxv_clear(rho); sxv_clear(&S->row);
            ntry++;
            for (int z = 0; z < 4 && z < nc && ntry < budget; z++) {
                int b = z; for (int y = z + 1; y < nc; y++) if (ca[y] > ca[b] || (ca[y] == ca[b] && cq[y] < cq[b])) b = y;
                { const int tq = cq[z]; cq[z] = cq[b]; cq[b] = tq; const double ta = ca[z]; ca[z] = ca[b]; ca[b] = ta; }
                ntry++;
                if (polish_try(S, cq[z])) { npiv++; break; }
            }
        }
        free(rows); free(cq); free(ca);
    } else
    for (int pass = 0; pass < npass; pass++) {
    const long np0 = npiv;
    for (int q = 0; q < N && ntry < budget; q++) {
        if (!POLISH_CAND(q)) continue;
        ntry++;
        if (polish_try(S, q)) npiv++;
    }
    if (npiv == np0) break;
    }
    #undef POLISH_CAND
    if (npiv) { refactor(S); compute_primal_dual(S); S->heap_ok = 0; S->ch_ok = 0; }
    {   static int ipd = -1; if (ipd < 0) ipd = getenv("BRISK_SXINTPREFDBG") != NULL;
        if (ipd) fprintf(stderr, "polish: %ld candidates, %ld pivots\n", ntry, npiv); }
    return npiv;
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
    /* (11 November 2026) the same pass counts the boxed variables for the perturbation at the start (perturb_first's own pass
     * over all variables, on the working bounds, which are these after sxm_reset_work) */
    int nbox0 = 0;
    for (int j = 0; j < N; j++) {
        if (S->lo0[j] > S->up0[j] + 1e-9 * (1 + fabs(S->lo0[j]))) { res->status = 1; return 1; }
        nbox0 += S->lo0[j] > -SX_INF && S->up0[j] < SX_INF;
    }
    sxm_reset_work(S);
    if (!M->factored) { refactor(S); M->factored = 1; M->xok = M->dok = 0; M->xinc = 0; }
    /* (the values of the last optimal solve kept when nothing they depend on changed: a re-solve with nothing changed,
     * or after a cost change only, skips one or both passes over the matrix) */
    /* (27 October 2026) the reduced costs before the nonbasic variables are placed: they depend on the basis and the
     * costs only, and the placement below reads them (before, those of the basis before a newly given one) */
    if (!M->dok || M->cost_changed) compute_dual(S);
    /* the nonbasic variables at their bounds again: a bound that moved takes the variable with it; one that was fixed
     * goes to the side a given basis named for it (M->hint), else, like one that lost the bound it was at, to the bound
     * its reduced cost asks for */
    /* (11 November 2026) and in the same pass the dual infeasibilities after the placement (they depend on the reduced costs
     * and the sides only, which nothing below changes before they are used): of variables a bound flip cannot help, of boxed
     * ones. Two passes over all variables before */
    int dnb = 0, dbox = 0;
    /* (11 November 2026) the basic values updated by the moves of the nonbasic values when they are those of the last optimal
     * solve's final check (M->xinc): x_B -= B^-1 sum_j a_j (x_j new - x_j old), one forward solve, none when nothing moved (a
     * branch on a basic variable), instead of the whole computation (compute_primal: a pass over every nonbasic column and a
     * forward solve); 6 % of the mixed-integer solver's time on a multiple knapsack MILP. BRISK_SXMXINC=0: as before */
    static int xinc_on = -1; if (xinc_on < 0) { const char *e_ = getenv("BRISK_SXMXINC"); xinc_on = !(e_ && atoi(e_) == 0); }
    const int useinc = !M->xok && M->xinc && xinc_on;
    SxVec *dv = &S->col; int nmoved = 0;
    if (useinc) sxv_clear(dv);
    for (int j = 0; j < N; j++) {
        if (S->pos[j] >= 0) continue;
        const double l = S->lo[j], u = S->up[j]; const int mv = S->move[j];
        const double xo = S->x[j];
        if (l == u) { S->move[j] = 0; S->x[j] = l; if (useinc && S->x[j] != xo) { col_scatter(S, j, dv, -(S->x[j] - xo)); nmoved++; } continue; }
        else if (mv == 0 && M->hint[j] > 0 && l > -SX_INF) { S->move[j] = 1; S->x[j] = l; M->hint[j] = 0; }
        else if (mv == 0 && M->hint[j] < 0 && u < SX_INF) { S->move[j] = -1; S->x[j] = u; M->hint[j] = 0; }
        else if (mv == 0 || (mv > 0 && l <= -SX_INF) || (mv < 0 && u >= SX_INF)) set_nonbasic(S, j, S->d[j]);
        else S->x[j] = mv > 0 ? l : u;
        if (useinc && S->x[j] != xo) { col_scatter(S, j, dv, -(S->x[j] - xo)); nmoved++; }
        const double dj = S->d[j];
        const int bad = S->move[j] == 0 ? fabs(dj) > S->told : S->move[j] * dj < -S->told;
        if (bad) { if (l > -SX_INF && u < SX_INF) dbox++; else dnb++; }
    }
    if (useinc) {
        if (nmoved) {
            double dd = 0; ftran(S, dv, &dd, NULL);     /* (a local density: the entering columns' estimate stays theirs) */
            if (dv->cnt < 0) { for (int i = 0; i < S->m; i++) S->x[S->basic[i]] += dv->val[i]; }
            else for (int k = 0; k < dv->cnt; k++) { const int i = dv->idx[k]; S->x[S->basic[i]] += dv->val[i]; }
        }
        sxv_clear(dv);
    } else if (!M->xok) compute_primal(S);
    M->xinc = 0;
    int fresh0 = 0;      /* (optimal as it stands: the values computed above are those to check at the end) */
    S->cut_skip = 0;   /* (a pause from an earlier solve does not carry over) */
    S->cut_on = o && o->cutoff_on; S->cut_val = S->cut_on ? (S->maxim ? -o->cutoff : o->cutoff) / S->costscale : 0;
    const int pinf = primal_infeasible(S);
    int st;
    S->noverify = 1;
    /* (23 October 2026) the costs perturbed at the start of a solve by the dual method, as the first solve's are (the end
     * of the solve removes them; BRISK_SXMPERT=0: not). Unperturbed, the MIP developer's node LP of csched008 took 3,723
     * iterations from its basis and 6,551 from scratch, perturbed 930 and 1,564; long dives on neos-933966 (lpdive.c)
     * over 600 s -> 38 s; air05, binkar10_1, mas76, 30n20b8 about the same */
    static int sxmpert = -1; if (sxmpert < 0) sxmpert = !(getenv("BRISK_SXMPERT") && atoi(getenv("BRISK_SXMPERT")) == 0);
    /* (fresh: the loop entered directly uses the factorization and the values computed above) */
    if (!pinf && !dnb && !dbox) { st = 0; fresh0 = !useinc; }     /* (updated values are checked on a fresh computation) */
    else if (M->cost_changed && !pinf) { S->fresh = 1; st = primal_simplex(S, 1); }
    else if (!dnb) {
        if (dbox) { for (int j = 0; j < N; j++) if (S->pos[j] < 0 && is_boxed(S, j)) set_nonbasic(S, j, S->d[j]); compute_primal(S); }
        S->phase = 2; S->fresh = 1;
        if (sxmpert) { S->nbox_given = nbox0; S->nbox_given_ok = 1; perturb_first(S); }
        st = dual_loop(S);
    } else if (!pinf) { S->fresh = 1; st = primal_simplex(S, 1); }
    else { if (sxmpert) { S->nbox_given = nbox0; S->nbox_given_ok = 1; perturb_first(S); } st = dual_simplex(S); if (st == 2) st = primal_simplex(S, 1); }
    /* (25 October 2026) an infeasibility the dual method could not prove (dual_loop's status 4): the primal method
     * decides */
    if (st == 4) { sxm_reset_work(S); normalize_nonbasic(S); compute_primal_dual(S); S->fresh = 1; st = primal_simplex(S, 1); }
    /* an infeasibility found by the primal method's phase 1: the dual method from there, for the proof */
    if (st == 1 && !S->have_ray) { sxm_reset_work(S); normalize_nonbasic(S); compute_primal(S); const int s2 = dual_simplex(S); if (s2 == 1 || s2 == 3 || s2 == 6) st = s2; }
    /* the problem's own costs and bounds; the violations on the problem as given (as sx_solve_core), the updated
     * factorization first, a new one only when they exceed the target */
    /* (28 October 2026; ISSUES 625) the targets: the caller's tolerances when they are looser than the default target (the
     * mixed-integer solver's node LPs: beasleyC3's ended the rounds at violations of 1.3e-8 and 1.5e-7 and were reported
     * failed by the rule of ISSUES 603, against the caller's 1e-7 that it may raise to 1e-6 at nodes) */
    const double tgp = fmax(S->env_target, o && o->tol_primal > 0 ? o->tol_primal : 0);
    const double tgd = fmax(S->env_target, o && o->tol_dual > 0 ? o->tol_dual : 0);
    S->tiny_on = 1; if (S->tiny) memset(S->tiny, 0, (size_t)S->tiny_n);
    for (int round = 0; round < 5 && st == 0; round++) {
        sxm_reset_work(S); normalize_nonbasic(S);
        if (!(round == 0 && fresh0)) { compute_primal_dual(S); }
        double pv, dv, psv, dsv; unscaled_violations(S, tgp, tgd, &pv, &dv, &psv, &dsv);
        /* (1 November 2026) the violations over the target come from the updated factorization's error or are the basis's
         * own (basic values outside their bounds by about the tolerance, which the method's scaled tolerance accepted): one
         * step of refinement with the updated factorization tells them apart. Only when it changes the violations is the
         * error the factorization's and a new one is made (if they are still over the target); otherwise the round below
         * removes them with the tightened tolerance. The MIP developer's lotsizing_15 (mixed-integer LP, one thread, 20 s):
         * 1,477 factorizations in this clean-up, the refinement changed no violation in any, and the new factorization left
         * 1,365 of them as they were (the other 112 at the target's rounding level). BRISK_SXMREFINE=0: the new factorization
         * at once, as before */
        static int mref = -1; if (mref < 0) mref = !(getenv("BRISK_SXMREFINE") && atoi(getenv("BRISK_SXMREFINE")) == 0);
        int facterr = 1;
        if ((pv > tgp || dv > tgd) && S->F.nupd > 0 && mref) {
            const double pv0 = pv, dv0 = dv;
            if (pv > tgp) refine_primal(S);
            if (dv > tgd) refine_dual(S);
            unscaled_violations(S, tgp, tgd, &pv, &dv, &psv, &dsv);
            facterr = (pv0 > tgp && pv < 0.9 * pv0) || (dv0 > tgd && dv < 0.9 * dv0);
        }
        if ((pv > tgp || dv > tgd) && S->F.nupd > 0 && facterr) { refactor(S); compute_primal_dual(S); unscaled_violations(S, tgp, tgd, &pv, &dv, &psv, &dsv); }
        if (pv <= tgp && dv <= tgd) break;
        /* (27 October 2026, the night; ISSUES 603) violations above the target after the rounds: not optimal */
        if (round == 4) { st = 4; break; }
        /* (28 October 2026; ISSUES 629) the working tolerance below the least scaled violation of the entries over the
         * target: the target ratio alone left it above them when the scaling makes a violation of the problem as given
         * far larger than in the scaled problem (glass4, sp150x300d: the same primal violation of 1e-7 to 6e-7 through
         * the five rounds, the method seeing none, thousands of node LPs reported failed) */
        if (dv > tgd) { S->told = fmax(fmin(S->told * 0.5 * tgd / dv, 0.5 * dsv), 1e-12); st = primal_simplex(S, 1); }
        else { S->tolp = fmax(fmin(S->tolp * 0.5 * tgp / pv, 0.5 * psv), 1e-12); st = dual_loop(S); }
    }
    S->tiny_on = 0;
    /* (SxmOpts.polish = LmOpts.polish, or BRISK_SXPOLISH=1 everywhere, with integer flags) the optimal basis polished: integer columns out of it (polish_int) */
    {   static int pol = -1; if (pol < 0) pol = getenv("BRISK_SXPOLISH") && atoi(getenv("BRISK_SXPOLISH")) > 0;
        if ((pol || (o && o->polish)) && st == 0 && S->isint) {
            /* (8 November 2026, the MIP developer: up to 3.4e-7 left on csched008, MILP) when the pivots leave a violation
             * above the target, the same final rounds as above repair it from the polished basis (the dual method for
             * values beyond their bounds, the primal for reduced costs of the wrong sign), so that an optimal status still
             * means within the targets; at most three rounds, the status from the last */
            if (polish_int(S) > 0) {
                for (int rr = 0; rr < 3 && st == 0; rr++) {
                    double pv, dv, psv, dsv; unscaled_violations(S, tgp, tgd, &pv, &dv, &psv, &dsv);
                    if (pv <= tgp && dv <= tgd) break;
                    if (getenv("BRISK_SXINTPREFDBG")) fprintf(stderr, "polish: violations %.2e %.2e after the pivots, repaired\n", pv, dv);
                    st = dv > tgd ? primal_simplex(S, 1) : dual_loop(S);
                }
            }
        } }
    S->fresh = 0; S->noverify = 0;
    M->cost_changed = 0;
    M->xok = M->dok = st == 0;
    M->xinc = st == 0;
    M->cutoff = st == 7;
    M->have_farkas = st == 1 && S->have_ray; M->raysign = S->raymo < 0 ? -1 : 1;
    M->have_pray = st == 2 && S->have_pray;
    res->status = st; res->iters = S->iter; res->iters_dual = S->it_dual; res->iters_primal = S->it_primal;
    {   static int ipd = -1; if (ipd < 0) ipd = getenv("BRISK_SXINTPREFDBG") != NULL;   /* (a diagnostic: the choices the integer preference changed) */
        if (ipd) { fprintf(stderr, "intpref: %ld choices changed in %ld iterations; entering: %ld integer, %ld other; integer heads kept: %ld with an alternative below the |alpha| test, %ld with none\n", S->n_intpref, S->iter, S->n_ent_int, S->n_ent_oth, S->n_ip_small, S->n_ip_none); }
        S->n_intpref = S->n_ent_int = S->n_ent_oth = S->n_ip_small = S->n_ip_none = 0; }
    res->nfactor = S->nfactor - nf0; res->work = S->work - w0;
    return st;
}
void sxm_values(const SxModel *M, double *x, double *y, double *z) {
    const Sx *S = &M->S; const int n = S->n, m = S->m;
    const double sg = S->maxim ? -1.0 : 1.0;
    if (x) for (int j = 0; j < n; j++) x[j] = S->x[j] * S->cscale[j];
    /* (27 October 2026) at the cutoff, the duals of the bound that was claimed (dual_bound): y = cost_B B^{-1} with the
     * working costs, read from a row's logical as its working cost minus its reduced cost (minus the reduced cost alone
     * when the logical's cost is zero, as at an optimum), and the reduced costs of the costs as given, c - A'y; the
     * perturbation or a shift of a logical had made the duals returned differ from those of the bound */
    if (M->cutoff) {
        if (y) for (int i = 0; i < m; i++) y[i] = sg * (S->cost[n + i] - S->d[n + i]) * S->rscale[i] * S->costscale;
        if (z) for (int j = 0; j < n; j++) z[j] = sg * (S->d[j] - (S->cost[j] - S->cost0[j])) / S->cscale[j] * S->costscale;
        return;
    }
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
    int m, N, factored, xinc;
    int *basic, *pos; signed char *move; double *x, *d, *w, *lo0, *up0;
    char *wk; int wlazy;
    SxLU F;
};
SxmSaved *sxm_save(const SxModel *M) {
    const Sx *S = &M->S; const int m = S->m, N = S->N;
    SxmSaved *V = xcalloc(1, sizeof *V);
    V->m = m; V->N = N; V->factored = M->factored; V->xinc = M->xinc;
    V->basic = xmalloc(sizeof(int) * (m + 1)); memcpy(V->basic, S->basic, sizeof(int) * m);
    V->pos = xmalloc(sizeof(int) * N); memcpy(V->pos, S->pos, sizeof(int) * N);
    V->move = xmalloc(N + 8); memcpy(V->move, S->move, N);
    V->x = xmalloc(sizeof(double) * N); memcpy(V->x, S->x, sizeof(double) * N);
    V->d = xmalloc(sizeof(double) * N); memcpy(V->d, S->d, sizeof(double) * N);
    V->w = xmalloc(sizeof(double) * (m + 1)); memcpy(V->w, S->w, sizeof(double) * m);
    V->wk = xmalloc((size_t)m + 1); memcpy(V->wk, S->wk, (size_t)m); V->wlazy = S->wlazy;
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
    memcpy(S->wk, V->wk, (size_t)m); S->wlazy = V->wlazy;
    memcpy(S->lo0, V->lo0, sizeof(double) * N); memcpy(S->up0, V->up0, sizeof(double) * N);
    if (V->factored) sxlu_copy(&S->F, &V->F);
    M->factored = V->factored; M->xok = M->dok = 0; M->xinc = V->xinc;
    rows_partition(S);
    sxm_reset_work(S);
}
void sxm_saved_free(SxmSaved *V) {
    if (!V) return;
    free(V->basic); free(V->pos); free(V->move); free(V->x); free(V->d); free(V->w); free(V->wk); free(V->lo0); free(V->up0);
    sxlu_free(&V->F); free(V);
}
