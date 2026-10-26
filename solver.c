#define _GNU_SOURCE          /* sched_getaffinity, CPU_COUNT (the cores the solve may use) */
#include "brisk.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <fcntl.h>
#include <dirent.h>
#include <sched.h>
#endif

/* OpenBLAS thread control (weak: absent for other BLAS libraries) */
#if defined(__APPLE__) || defined(BRISK_NO_OPENBLAS)
/* Mach-O links no undefined weak references (and Accelerate has no thread control): null
 * function pointers, so every "if (openblas_...)" test is false */
static void (*const BL(openblas_set_num_threads))(int) = 0;
static int (*const BL(openblas_get_num_threads))(void) = 0;
#else
extern void BL(openblas_set_num_threads)(int) __attribute__((weak));
extern int BL(openblas_get_num_threads)(void) __attribute__((weak));
#endif
static int blas_threads_saved = -1;
static void blas_serial_begin(void) {
    if (BL(openblas_get_num_threads) && BL(openblas_set_num_threads)) {
        blas_threads_saved = BL(openblas_get_num_threads)();
        if (blas_threads_saved > 1) BL(openblas_set_num_threads)(1);
    }
}
static void blas_serial_end(void) {
    if (blas_threads_saved > 1 && BL(openblas_set_num_threads)) BL(openblas_set_num_threads)(blas_threads_saved);
    blas_threads_saved = -1;
}
static inline int in_par(void) {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return 0;
#endif
}
static inline int nthreads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
/* the threads the running solve may use at most (brisk_solve): the count itself follows the
 * free cores from iteration to iteration, so what is decided once per solve looks at this */
static int g_nt_cap = 0;
static inline int nthreads_cap(void) { return g_nt_cap > 0 ? g_nt_cap : nthreads(); }
void brisk_threads_adapt(int verbose);

static const int IONE = 1;
static const double DONE = 1.0, DZERO = 0.0;
#define LANCZOS_K 30

/* 4.37: an interrupt (brisk_interrupt, capi.c) acts as the time limit reached now */
volatile sig_atomic_t brisk_stop_flag = 0;
int brisk_time_up(const Params *par) {
    return brisk_stop_flag || (par->timelimit > 0 && wtime() - par->t_start > par->timelimit);
}
double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

void params_default(Params *p) {
    /* 23 October 2026: every field defined, those not named below 0 or NULL (xcone and mf_try_budget had been left
     * unset: a stack Params passed its garbage xcone to brisk_solve after the matrix-free method's hand-over, main.c) */
    memset(p, 0, sizeof(*p));
    p->fom_start_x = p->fom_start_y = NULL; p->bound_anchor = NULL; p->fom_X0 = NULL; p->fom_y0 = NULL;
    p->tol = 1e-8;
    p->maxit = 100;
    p->verbose = 1;
    p->c_sparse = 1.0;
    p->c_blas = 0.05;
    p->max_correctors = 0;   /* available via -corr; no measured gain on the test suite */
    p->dense_mem = 768.0 * 1024 * 1024;
    p->lanczos_k = 30;
    p->init = 1;
    p->mixed = -1;
    p->lowrank = -1;
    p->dict = -1;
    p->cls = -1;
    p->gamma_max = 0.99;
    p->stall_win = 5;
    p->direction = -1;      /* auto */
    p->nt_fast = 0;          /* measured: more iterations (cancellation in dX = G Rt G' - W dZ W) */
    p->par_blocks = -1;
    p->known_feasible = 0;
    p->hsd_pat = 1;
    p->nt_split = 0;        /* experimental (see IMPLEMENTATION.md) */
    p->nt_ls = 3;
    p->nt_ls_gap = 1e-4;
    p->sparse_schur = -1;
    p->balance = 0.0;      /* off: mixed evidence (CHANGES.md) */
    p->pivtol = 1e-14;
    p->feas_grow = 0.0;      /* off by default: costs iterations, see CHANGES */
    p->sigma_min = 0.0;      /* off: a blanket floor costs iterations (see CHANGES) */
    p->chordal = -1;     /* automatic: convert when it lowers the estimated cost */
    p->chordal_minn = 100;
    p->chordal_density = 0.3;       /* the cost estimate decides; this only skips hopeless blocks early */
    p->chordal_maxclique = 160;   /* 4.22: 80 rejected the large RTE/PEGASE networks (case6468: max clique 60 once past a transient) */
    p->chordal_form = 1;
    p->deprows = 1;          /* 20 October 2026: dependent equality constraints removed (presolve.c) */
    p->free_elim = -1;       /* 4.30: -1 auto (kernel-form SOS class), 0 off, 1 on */
    p->free_elim_fill = 1e4;
    p->chordal_sup = -1;     /* -1: chosen on the estimated cost among 8, 4, 12, 0 */
    p->chordal_merge = 0.6;
    p->chordal_gain = 0.2;
    p->chordal_gain2 = 0.7;
    p->chordal_maxextra = 20.0;
    p->dual = 0;
    p->dual_rho = 3.0;
    p->dual_cert = 12;
    p->dual_margin = 1.5;
    p->dual_giveup = 40;
    p->dual_tau = 0.9;
    p->dual_murule = 0;
    p->dual_ls = 1;
    p->dual_corr = 12;
    p->dual_mured = 0;
    p->dual_tau_nb = 2.0;
    p->dual_tau_ub = 1.0;
    p->dual_sparse_n = 300;
    p->zsparse_n = 200;
    p->zsparse_fill = 0.1;
    p->dual_corr_auto = 1;
    p->dual_gstart = 1;
    p->dual_mudrop = 0.1;
    p->dual_verify_n = 500;
    p->dual_tau_long = 1e9;
    p->dual_gamma = 0;       /* phase-1 objective eps b'y - r with eps = 1/gamma (0: off; tried, no gain) */
    p->hsd = -1;             /* automatic: retry a stalled run with the embedding */
    p->no_retry = 0;
    p->hsd_hoc = 4;          /* 5.7: second-order passes at the chosen direction (kept only when the predicted reduction is not smaller) */
    p->hsd_nb = 0.3;         /* 5.7: after kept passes the step may go to 0.9995 of the boundary while lambda_min(XZ) >= 0.3 mu */
    p->hsd_first = 1; p->hsd_first_c = 12.0; p->hsd_first_asm = 0.5; p->hsd_first_on = 0; p->std_slow = 10; p->crossover = 0; p->fr_gain = 4.0;
    p->soc = 1; p->soc_minn = 3;
    p->hsd_sig = 3;          /* candidates in the sigma search (0 = Mehrotra only) */
    p->hsd_beta = 1e-3;
    p->hsd_batch = 1;        /* embedding: paired Schur solves as one multi-rhs PCG */
    p->hsd_cgain = 0;        /* embedding: a corrector must gain cgain x its share of an iteration in step length */
    p->hsd_cfrac = 1.0;      /* embedding: corrector time budget as a fraction of assembly + factorization */
    p->hsd_soltol = 0;       /* embedding: relative accuracy of the main Schur solves (0 = full) */
    p->hsd_reghint = 1;      /* embedding: reuse the Schur regularization of the previous iteration */
    p->hsd_x0 = 0;           /* 23 October 2026: the embedding's primal start as computed (-hsdx0 <f>: scaled by f) */
    p->hsd_pshrink = 0;      /* split free pairs: shrink min(x+,x-) by this fraction after each step */
    p->hsd_stall = 12;       /* embedding: iterations without a better iterate before giving up */
    p->hsd_free = -1;        /* free variables (split LP pairs) in the embedding: -1 auto, 0 off, 1 augmented Lagrangian, 2 Schur of M, 3 Bunch-Kaufman */
    p->hsd_fomega = 1e2;     /* ... weight of the augmented-Lagrangian term (relative to diag M) */
    p->hsd_refine = 2;       /* passes enforcing A(dX) - b dtau = eta rp */
    p->hsd_cucap = 0;        /* corrector: cap of the downward push of large products (0: the box top) */
    p->hsd_qscale = 0;       /* embedding: second-order term scaled by 1, the affine step (1) or its square (2) */
    p->hsd_pivtol = 1e-20;   /* embedding: squared-pivot threshold of the Schur factor (tiny pivots are refined by PCG) */
    p->hsd_corr = 3;         /* Gondzio correctors per iteration inside the embedding */
    p->hsd_cbmin = 0.1; p->hsd_cbmax = 10.0; p->hsd_cacc = 1.02; p->hsd_ctrace = 1; p->hsd_ctarget = 0;
    p->hsd_dir = -1;         /* direction inside the embedding: 0 HKM, 1 NT, -1 auto (NT on the chordal moment form, HKM otherwise) */      /* neighbourhood: tau kappa and LP products >= beta mu */
    p->dd_end = 1;
    p->dd_iters = 40;
    p->dd_maxm = 1200;
    p->dd_maxn2 = 0;         /* 15 October 2026 (ISSUES 99): 0 = from the memory limit (dd_n2_cap); was 2e5, a silent cap at a block order of about 450 */
    p->dd_budget = 1.0e9;
    p->dd_time = 60.0;
    p->polish = 1;
    p->polish_x = 1;
    p->polish_r = 1;         /* the restoration of the primal feasibility in the square-root metric before the other corrections */
    p->polish_mem = 512.0 * 1024 * 1024;
    p->reg_ill = 1e-10;
    p->sigma_rule = -1;   /* auto: cubic for pure LP, adaptive otherwise */
    p->timelimit = 0;
    p->fr_fallback = 0.0;    /* 5.17 (the owner, 16 October 2026): no re-solve when the removed duals are not rebuilt; the user is told
                              * how to solve without the reduction (5.16: the re-solve within 4x the solve's time; -frfallback 4) */
    p->sym_fallback = 4.0;   /* 5.16: the solve without an automorphism reduction when the reduced one ends worse than reduced accuracy, within 4x */
    p->bound_side = 0; p->bound_tol = 1e-10; p->bound_margin = 1e-13; p->bound_track_tol = -1; p->btrack = NULL; p->btrack2 = NULL; p->bound_xfile = NULL; p->bound_yfile = NULL; p->bound_prior = 1; p->bound_prec = -1;
    p->t_start = 0;
    p->retry_acc = 1e-7;     /* results within 1e-7 are returned as they are (no second solve) */
    p->dd_factor = 1.0;      /* endgame may cost at most as much as the solve itself ... */
    p->dd_minwork = 1e11;    /* ... or this floor, whichever is more (5.16: was 2e9, one dd iteration where three are the minimum on qap7-qap10; +2 % time on SDPLIB) */
    p->dd_done = 0;
    p->work_done = 0;
    p->dd_time = 0;          /* no wall-clock cap by default: the work budget is deterministic */
    p->acc_level = 1; p->red_acc = 1e-6; p->retry_hsd1 = 1e-6; p->slow_gate = 1e-5;
    p->tracebound = -1;
    p->dualize = -1;
    p->symfile = "auto"; p->symtime = 20.0; p->symnodes = 500; p->symbd = 1; p->symmin = 1.5; p->symsign = 1; p->symsigned = 1; p->returnx = -1; p->fom_race = 0; p->fom_race_budget = 0; p->fom_bound = 1; p->fom_bound_maxit = 0; p->hp_kind = 0; p->hp_digits = 0; p->hp_tol = 0; p->hp_ext = 2; p->symalg = -1; p->symalgmax = 1000; p->blkscale = 0; p->bm = -1;
    p->chordal_need = 0;
    p->lralm = 0; p->lr_rank = 1; p->lr_rmax = 32; p->lr_outer = 500; p->lr_inner = 2000; p->lr_escape = 1; p->lr_prec = 1; p->lr_newton = 1; p->lr_sigma = 10.0; p->lr_tol = 1e-6; p->lr_trace = 0;
    p->mfipm = 0; p->mf_rho = 10; p->mf_rmax = 4; p->mf_drop = 0.5; p->mf_kmax = 8000; p->mf_cgmax = 3000; p->mf_cgtol_min = 1e-10; p->mf_cgtol_max = 1e-3; p->mf_diag = 0; p->mf_warm = 1; p->mf_stall = 5; p->mf_cgtime = 0; p->mf_hand = 1e-3; p->mf_handcg = 600; p->mf_try = 1; p->mf_trial = 0; p->mf_trial_total = 0; p->mf_try_tl = 0; p->mf_try_fits = 1; p->mf_proj = 0; p->mf_eta = 0.1; p->fom_sigma0 = 1.0; p->fom_aadr = 1; p->fom_single = 1e-4; p->fom_ssn_stall = 6; p->mf_recycle = 0;
    p->mixed_frac = 0.4;
    p->fom = -1; p->fom_halpern = 0; p->fom_maxit = 100000; p->fom_tol = 0; p->fom_sigma = 0; p->fom_sigrule = 2; p->fom_sigint = 20; p->fom_aa = 25; p->fom_bm = 0; p->fom_bm_rank0 = 10; p->fom_bm_outer = 100; p->fom_bm_inner = 300; p->fom_bm_rho = 1.0; p->fom_bm_gtol = 1e-7; p->fom_bm_negtol = 1e-6; p->fom_ssn = 1; p->fom_ssn_after = 300; p->fom_ssn_res = 1e-3; p->fom_ssn_rho = 3.0; p->fom_ssn_prec = 1; p->fom_ssn_sig0 = 10.0; p->fom_ssn_eta = 0.1; p->fom_ssn_warm = 0; p->fom_ssn_outer = 200; p->fom_ssn_newton = 30; p->fom_ssn_cg = 200; p->fom_aasafe = 2.0; p->fom_sigmax = 2.0;
    p->warm_lam = 0;
    { const char *e = getenv("BRISK_WARM"); if (e) p->warm_lam = atof(e); }
    p->warm_X = p->warm_Z = NULL; p->warm_y = NULL;
}

/* 4.29: accuracy levels. low: tol 1e-6 and the thresholds derived from it (re-solve if the
 * standard method ends above 10 tol, the embedding-first attempt above 100 tol; the
 * slow-progress stop of a standard run that will be re-solved only below 100 retry_acc;
 * "reduced accuracy" below 100 tol). high: tol 1e-10 with the default re-solve thresholds:
 * scaled down (1e-9 / 1e-8) they re-solved 13 of 109 small instances, and only one (ss30)
 * gained; qap10 and gpp500-1 ended worse and 3-7x slower.                              */
void params_accuracy(Params *p, int level) {
    p->acc_level = level;
    if (level == 0) {
        p->tol = 1e-6; p->retry_acc = 1e-5; p->red_acc = 1e-4; p->retry_hsd1 = 1e-4; p->slow_gate = 1e-3;
    } else if (level == 2) {
        p->tol = 1e-10; p->retry_acc = 1e-7; p->red_acc = 1e-6; p->retry_hsd1 = 1e-6; p->slow_gate = 1e-5;
    } else {
        p->tol = 1e-8; p->retry_acc = 1e-7; p->red_acc = 1e-6; p->retry_hsd1 = 1e-6; p->slow_gate = 1e-5;
    }
}

/* 4.42: a dense array that cannot fit ends the solve with a message. Without this the
 * allocation succeeds (overcommit) and the kernel kills the process when the memory is
 * touched, which from MATLAB, Python or Julia takes the user's session with it
 * (mcp250-1 with -dualize 1: a 7.9 GB Schur complement on a 6.3 GB machine). */
static double mem_rss(void) {
    double r = 0; FILE *f = fopen("/proc/self/statm", "r");
    if (f) { long a = 0, b2 = 0; if (fscanf(f, "%ld %ld", &a, &b2) == 2) r = (double)b2 * 4096.0; fclose(f); }
    return r;
}
/* 15 October 2026 (the owner's benchmark, ISSUES 98): does an allocation of this many bytes fit in what is left of
 * the limit and of the machine (the retry with the other method on diamond_patch, n = 5,477, was killed by the
 * system at 5.8 GB after the standard method had ended at 7e-2: the user got nothing) */
static int mem_fits(double bytes, double *avail_out) {
    extern double brisk_mem_limit(void);
    const double lim = brisk_mem_limit();
    double avail = lim - mem_rss();
    { FILE *f = fopen("/proc/meminfo", "r");
      if (f) { char ln[160]; while (fgets(ln, sizeof ln, f)) { double kb; if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) { if (kb * 1024.0 < avail) avail = kb * 1024.0; break; } } fclose(f); } }
    if (avail_out) *avail_out = avail;
    return getenv("BRISK_NOMEMCHECK") || !(lim > 0) || bytes <= 0.95 * avail;
}
static void mem_check(double bytes, const char *what, int m) {
    extern double brisk_mem_limit(void);
    const double lim = brisk_mem_limit();
    double avail = lim - mem_rss();
    { FILE *f = fopen("/proc/meminfo", "r");     /* what is free now, when the system says (other processes count) */
      if (f) { char ln[160]; while (fgets(ln, sizeof ln, f)) { double kb; if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) { if (kb * 1024.0 < avail) avail = kb * 1024.0; break; } } fclose(f); } }
    if (getenv("BRISK_NOMEMCHECK") || !(lim > 0) || bytes <= 0.95 * avail) return;
    fprintf(stderr, "brisk: out of memory: %s (m = %d) needs %.1f GB; %.1f GB are available.\n"
                    "       For problems this large use the first-order engine (-fom 1) or, for large sparse\n"
                    "       problems, the low-rank method (-lralm 1); see OPTIONS.md.\n", what, m, bytes / 1e9, avail / 1e9);
    exit(1);
}
static void *amalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) { fprintf(stderr, "brisk: out of memory\n"); exit(1); }
    return p;
}
/* 5.12: zeroed and checked (setup_residual_pattern dereferenced a NULL calloc under a memory cap) */
static void *ccalloc(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) { fprintf(stderr, "brisk: out of memory\n"); exit(1); } return p; }

/* ------------------------------------------------------------------------ */
/* sparsity pattern of a middle matrix F in products Zi * F * R              */
typedef struct {
    int nf;              /* entries (full symmetric expansion) */
    int *fr, *fc;
    int nr;              /* distinct rows */
    int *rows;
    int route;           /* 0 row-subset, 1 sparse-left, 2 dense */
    int owned;
} Pat;

/* per-block iterate and workspace                                          */
typedef struct BS {
    int n, type;
    double *X, *Z, *LX, *LZ, *Zi;
    double *dX, *dZ, *F, *G, *Q, *W;
    double *Xn, *Zn;     /* temporaries of the (optional) centrality corrector only */
    double stepped[2];   /* step currently applied in place by try_step (dual, primal) */
    int rcap;            /* rows allocated in the row-subset work arrays */
    double *Fr, *T1, *Zc;
    double *lq, *v1, *v2;   /* lq: the Lanczos basis (LANCZOS_K columns) and, after it, the warm-start vectors of X and Z (5.12) */
    int lzw_ok[2];          /* a warm-start vector is there (X, Z) */
    double *Ct, *Dxc;    /* centrality correctors: trial product/eigvecs, correction */
    double *ev;          /* eigenvalues */
    double *dtmp;        /* nd */
    Pat upat;            /* union pattern of the constraint matrices */
    struct BS *vbs;      /* dictionary route: state of the virtual q x q block (X = P, Zi = Q) */
    double *dW;          /* n x dnv workspace (X * V) */
    double *Xb, *Zb;     /* best iterate (for polishing), if memory allows */
    double *Xr, *Zr;     /* the iterate the primal feasibility is restored from (RCand), if memory allows */
    SChol *dzs, *dzt;    /* dual method: sparse Cholesky of Z (iterate, tests), or NULL */
    SChol *zf;           /* 5.12 HKM iteration: sparse Cholesky of Z (dense X, sparse Z), or NULL */
    int zon;             /* zf is the factor of the current Z and replaces the dense one (inside the iteration loop) */
    int zlz;             /* ... and the dense factor LZ is kept as well (the embedding's correctors use it) */
    int *zcp, *zri;      /* its pattern (CSC, both triangles and the diagonal): every dZ and F of the products lives on it */
    double *zt;          /* 2n: work vectors of the dual step length */
    int *ucp, *uri;      /* the constraints' union pattern (upat) by columns (CSC, both triangles) */
    double *zdp;         /* the predictor's dZ on the pattern of zcp/zri (for the corrector's product) */
    /* 5.12: Z, dZ and the dual residual live on that pattern (zero elsewhere): the loop's passes
     * over them run on its positions only (zpat) */
    int zpat, nzp, zbc, zrc;   /* on; positions; Zb / Zr copied in full once (their zeros) */
    size_t *zpos;        /* position r + c n of pattern entry q (CSC order of zcp/zri) */
    int *zamap;          /* pattern entry of each flattened constraint entry (B->foff order) */
    double *zfq, *zcq, *zr0q;   /* A'y, C and R0 on the pattern */
    /* NT direction: W = G G' is stored in Zi; lam = diag of the scaled point */
    double *ntG;         /* set only in the NT "view" (NULL in the block's own state) */
    double *ntGbuf, *ntVt, *ntT, *lam;
    double *ntGold;      /* 5.12: the previous iteration's G (the scaling of the Schur matrix still factored, for the end's correction) */
    float *ntGf, *ntLXf, *ntVtf, *ntTf, *ntT2f;   /* 5.12: single-precision copies for the frame changes (g_ntsp) */
    int ntf_ok;          /* the copies belong to the current scaling */
    int nt_einfo;        /* the eigen-decomposition's info (nt_eig) */
    double *dX1, *dS1, *dX1t, *dS1t, *dXt, *dZt, *Rt;
    double *lrV, *lrPQ, *lrPd, *lrQd;   /* low-rank route: n x R, R x R (P upper, Q lower), diagonals */
    double *lrQ;         /* separate R x R buffer for Q when memory allows (sequential access) */
    double lr_mem;       /* memory budget for the low-rank route */
    int *lr_owner;
    Pat cpat;            /* union pattern + pattern of the initial dual residual */
    double *R0;          /* initial dual residual (sparse, dense storage); Rd = theta * R0 */
    int r0n;             /* its nonzeros (full expansion) */
    int *r0r, *r0c;
    double *r0v;
    int h_restricted;    /* evaluate A(Zi Rd X) on the constraint pattern only */
    double *Gd, *Mdd;    /* dense Schur route: chunk*n*n, nd*chunk */
    int chunk;
    double part[8];      /* per-block partial sums (summed in block order) */
    char *fm;            /* LP: free-pair mask inside the embedding (1 '+', 2 '-'), or NULL */
    double *sv, sbeta;   /* 5.16 second-order cone: the NT scaling W = sbeta (2 sv sv' - J) (lam: W z = W^{-1} x) */
    double *stmp;        /* ... and 3n of work space */
} BS;

static inline size_t bsz(const Block *B) { return B->type == BLK_SDP ? (size_t)B->n * B->n : (size_t)B->n; }

/* ---- 5.16: second-order cone blocks (BLK_SOC), REPORT_5.16 §9. The cone Q = {x : x_0 >= ||xb||}
 * (xb = (x_1, ..., x_{n-1})) with the Jordan algebra x o z = (x'z, x_0 zb + z_0 xb), the identity
 * e = (1, 0, ..., 0), J = diag(1, -1, ..., -1), det x = x'Jx, x^{-1} = Jx / det x; a cone counts
 * once in mu (x o z = mu e on the central path, x'z = mu). The Nesterov-Todd scaling W is
 * symmetric, W z = W^{-1} x = lam, and W = beta (2 v v' - J) with v'Jv = 1 (all O(n)):
 *   xn = x / sqrt(det x), zn = z / sqrt(det z), g = sqrt((1 + xn'zn) / 2),
 *   wn = (xn + J zn) / (2 g), v = (wn + e) / sqrt(2 (wn_0 + 1)), beta = (det x / det z)^(1/4).
 * The data are those of an LP block (a vector variable; A_i restricted to the block is a_i). */
static inline double soc_nb(int n, const double *x) { double t = 0; for (int i = 1; i < n; i++) t += x[i] * x[i]; return sqrt(t); }
/* det x = (x_0 - ||xb||)(x_0 + ||xb||): no cancellation near the boundary */
static inline double soc_det(int n, const double *x) { const double r = soc_nb(n, x); return (x[0] - r) * (x[0] + r); }
static inline int soc_interior(int n, const double *x) { const double r = soc_nb(n, x); return x[0] > 0 && x[0] - r > 1e-15 * x[0]; }
/* out = J d */
static inline void soc_J(int n, const double *d, double *out) { out[0] = d[0]; for (int i = 1; i < n; i++) out[i] = -d[i]; }
/* out = W d (out may not alias d) */
static void soc_W(const BS *s, int n, const double *d, double *out) {
    const double *v = s->sv;
    double vd = 0; for (int i = 0; i < n; i++) vd += v[i] * d[i];
    const double b = s->sbeta;
    out[0] = b * (2.0 * v[0] * vd - d[0]);
    for (int i = 1; i < n; i++) out[i] = b * (2.0 * v[i] * vd + d[i]);
}
/* out = W^{-1} d = (2 J v v' J - J) d / beta */
static void soc_Winv(const BS *s, int n, const double *d, double *out) {
    const double *v = s->sv;
    double vJd = v[0] * d[0]; for (int i = 1; i < n; i++) vJd -= v[i] * d[i];
    const double ib = 1.0 / s->sbeta;
    out[0] = ib * (2.0 * v[0] * vJd - d[0]);
    for (int i = 1; i < n; i++) out[i] = ib * (-2.0 * v[i] * vJd + d[i]);
}
/* out = W^2 d (the block's operator of the Newton system: dX = -W^2 dZ + ...) */
static void soc_H(const BS *s, int n, const double *d, double *out, double *tmp) { soc_W(s, n, d, tmp); soc_W(s, n, tmp, out); }
/* out = u o w */
static void soc_jprod(int n, const double *u, const double *w, double *out) {
    double t = 0; for (int i = 0; i < n; i++) t += u[i] * w[i];
    for (int i = 1; i < n; i++) out[i] = u[0] * w[i] + w[0] * u[i];
    out[0] = t;
}
/* out = l \ r: the solution of l o out = r (l in the interior; out may alias r) */
static void soc_jdiv(int n, const double *l, const double *r, double *out) {
    double lr = 0; for (int i = 1; i < n; i++) lr += l[i] * r[i];
    const double o0 = (l[0] * r[0] - lr) / soc_det(n, l), il0 = 1.0 / l[0];
    for (int i = 1; i < n; i++) out[i] = (r[i] - o0 * l[i]) * il0;
    out[0] = o0;
}
/* out = z^{-1} = J z / det z */
static void soc_inv(int n, const double *z, double *out) { const double id = 1.0 / soc_det(n, z); out[0] = z[0] * id; for (int i = 1; i < n; i++) out[i] = -z[i] * id; }
/* the NT scaling of (x, z) into s->sv, s->sbeta and s->lam; 1 when x or z is not interior */
static int soc_nt(BS *s, int n, const double *x, const double *z) {
    if (!soc_interior(n, x) || !soc_interior(n, z)) return 1;
    const double a = sqrt(soc_det(n, x)), b = sqrt(soc_det(n, z));
    double xz = 0; for (int i = 0; i < n; i++) xz += x[i] * z[i];
    const double g = sqrt(0.5 * (1.0 + xz / (a * b)));
    double *v = s->sv;
    const double ia = 1.0 / (2.0 * g * a), ib = 1.0 / (2.0 * g * b);
    v[0] = x[0] * ia + z[0] * ib;                         /* wn = (xn + J zn) / (2 g) */
    for (int i = 1; i < n; i++) v[i] = x[i] * ia - z[i] * ib;
    const double f = 1.0 / sqrt(2.0 * (v[0] + 1.0));
    v[0] = (v[0] + 1.0) * f;
    for (int i = 1; i < n; i++) v[i] *= f;
    s->sbeta = sqrt(a / b);
    soc_W(s, n, z, s->lam);
    return 0;
}
/* the identity scaling (W = I): beta = 1, v = e */
static void soc_nt_identity(BS *s, int n) { s->sbeta = 1.0; s->sv[0] = 1.0; for (int i = 1; i < n; i++) s->sv[i] = 0.0; }
/* the largest a with u + a du in the cone (1e30: no bound); the quadratic det(u + a du) = 0 in
 * the units of u (det = 1 after scaling by 1/sqrt(det u)) */
static double soc_maxstep(int n, const double *u, const double *du) {
    for (int i = 0; i < n; i++) if (!isfinite(du[i])) return 0.0;
    const double r = soc_nb(n, u), d = (u[0] - r) * (u[0] + r);
    if (!(d > 0) || !(u[0] > 0)) return 0.0;
    const double sc = 1.0 / sqrt(d);
    double nd = 0, ud = 0;
    for (int i = 1; i < n; i++) { nd += du[i] * du[i]; ud += u[i] * du[i]; }
    const double d0 = du[0] * sc, ndn = nd * sc * sc;
    const double qa = d0 * d0 - ndn, qb = 2.0 * (u[0] * sc * d0 - ud * sc * sc), qc = 1.0;
    double a = 1e30;
    if (qa == 0) { if (qb < 0) a = -qc / qb; }
    else {
        const double disc = qb * qb - 4.0 * qa * qc;
        if (disc >= 0) {
            const double q = -0.5 * (qb + (qb >= 0 ? sqrt(disc) : -sqrt(disc)));
            const double r1 = q / qa, r2 = q != 0 ? qc / q : 1e30;
            if (r1 > 0 && r1 < a) a = r1;
            if (r2 > 0 && r2 < a) a = r2;
        }
    }
    return a;
}
/* the smallest spectral value of the NT-scaled product: (lam_0 - ||lamb||)^2 of the pair
 * (x + a dx, z + a dz) (the block's lambda_min(XZ)); -1 when not interior. tmp: 2n doubles */
static double soc_lmin_at(int n, const double *x, const double *dx, const double *z, const double *dz, double a, double *tmp) {
    double *xt = tmp, *zt = tmp + n;
    for (int i = 0; i < n; i++) { xt[i] = x[i] + a * (dx ? dx[i] : 0.0); zt[i] = z[i] + a * (dz ? dz[i] : 0.0); }
    if (!soc_interior(n, xt) || !soc_interior(n, zt)) return -1.0;
    /* lam_0 +- ||lamb|| from det lam = sqrt(det x det z) and lam'lam = x'z */
    const double dl = sqrt(soc_det(n, xt) * soc_det(n, zt));
    double xz = 0; for (int i = 0; i < n; i++) xz += xt[i] * zt[i];
    /* lam_0^2 + ||lamb||^2 = x'z, lam_0^2 - ||lamb||^2 = dl: lam_0 - ||lamb|| = dl / (lam_0 + ||lamb||) */
    const double l0 = sqrt(0.5 * (xz + dl)), lb = sqrt(fmax(0.0, 0.5 * (xz - dl)));
    const double lm = dl / (l0 + lb);
    return lm * lm;
}

/* 4.37 bound mode: offer the current iterate to the tracker (X/tau of (P) or y/tau of (D),
 * unscaled as the solver's output). Admissible when the tracked side's residual is within
 * the tracking tolerance; kept when its objective is better. A best replaced by a better
 * one joins the anchors (the more central partners of the certificate's margin, bound.c)
 * when it is at least 100x less central than the last anchor. */
static void btrack_copy(const Problem *P, struct BS *S, const double *y, double tau, double **X, double *yv) {
    for (int i = 0; i < P->m; i++) yv[i] = P->cs * P->d[i] * y[i] / tau;
    const double f = P->bs / tau;
    for (int k = 0; k < P->nblk; k++) {
        const double *Xk = S[k].X;
        if (P->blk[k].type != BLK_SDP) { for (int i = 0; i < P->blk[k].n; i++) X[k][i] = f * Xk[i]; }
        else {
            const int n = P->blk[k].n;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    X[k][i + (size_t)j * n] = 0.5 * f * (Xk[i + (size_t)j * n] + Xk[j + (size_t)i * n]);
        }
    }
}
/* the newest anchor slot for a point of centrality mu: it overwrites the previous newest
 * unless that one is at least 100x more central than the anchor before it (the older
 * anchors form a ladder); the oldest is dropped when the slots are full */
static int btrack_anchor_slot(BoundTrack *T) {
    if (T->na >= 2 && T->mu_a[T->na - 1] > 1e-2 * T->mu_a[T->na - 2]) T->na--;
    if (T->na == BOUND_NANCHOR) {
        double **tx = T->Xa[0]; double *ty = T->ya[0];
        for (int a = 0; a + 1 < BOUND_NANCHOR; a++) { T->Xa[a] = T->Xa[a + 1]; T->ya[a] = T->ya[a + 1]; T->mu_a[a] = T->mu_a[a + 1]; }
        T->Xa[BOUND_NANCHOR - 1] = tx; T->ya[BOUND_NANCHOR - 1] = ty;
        T->na--;
    }
    return T->na++;
}
static void btrack_offer1(BoundTrack *T, const Problem *P, struct BS *S, const double *y, double tau,
                          double mu, const Result *R, int it) {
    if (!T || T->disabled || !(tau > 0)) return;
    const double res = T->side == 1 ? R->pinf : R->dinf;
    const double sc = T->side == 1 ? R->pobj : R->dobj;
    if (!isfinite(sc) || !isfinite(mu) || !(res <= T->tol_a)) return;
    if (!(res <= T->tol)) {
        /* not admissible as a certificate, but central and nearly feasible: an anchor */
        if (T->na && !(mu <= 1e-2 * T->mu_a[T->na - 1]) && !(T->na >= 2 && T->mu_a[T->na - 1] > 1e-2 * T->mu_a[T->na - 2])) return;
        const int s = btrack_anchor_slot(T);
        btrack_copy(P, S, y, tau, T->Xa[s], T->ya[s]);
        T->mu_a[s] = mu;
        return;
    }
    T->offers++;
    if (T->have && !(T->side == 1 ? sc < T->score : sc > T->score)) return;
    if (T->have) {
        /* the replaced best becomes the newest anchor */
        const int s = btrack_anchor_slot(T);
        double **tx = T->Xa[s]; T->Xa[s] = T->X; T->X = tx;
        double *ty = T->ya[s]; T->ya[s] = T->y; T->y = ty;
        T->mu_a[s] = T->mu;
    }
    if (!T->have || fabs(sc - T->score) > 1e-9 * (1 + fabs(sc))) T->it_imp = it;
    btrack_copy(P, S, y, tau, T->X, T->y);
    T->have = 1; T->score = sc; T->mu = mu; T->it = it; T->res = res;
    if (getenv("BRISK_BOUNDDBG")) { printf("   [bound track it %d score %.10e mu %.2e res %.1e; anchors", it, sc, mu, res); for (int a = 0; a < T->na; a++) printf(" %.1e", T->mu_a[a]); printf("]\n"); }
}
/* every side asked for (-bound pd: two trackers) */
static void btrack_offer(const Params *par, const Problem *P, struct BS *S, const double *y, double tau,
                         double mu, const Result *R, int it) {
    btrack_offer1(par->btrack, P, S, y, tau, mu, R, it);
    btrack_offer1(par->btrack2, P, S, y, tau, mu, R, it);
}
/* the last iteration at which a tracked bound improved (-1: none) */
static int btrack_it_imp(const Params *par) {
    int r = -1;
    if (par->btrack && par->btrack->have) r = par->btrack->it_imp;
    if (par->btrack2 && par->btrack2->have && par->btrack2->it_imp > r) r = par->btrack2->it_imp;
    return r;
}

/* 4.23: the parallel region only for large n. An "omp parallel if (false)" still enters the
 * runtime (0.22 us per call against 0.006 us for the loop at n = 3; TSSOS case6468 calls
 * these ~10^5 times per iteration on its 10,421 blocks) */
static void symmetrize(int n, double *A) {
    if (n > 1500 || ((size_t)n * n >= 2 * (size_t)65536 && !omp_in_parallel())) {     /* 5.12: from n = 362 (element-wise: the same bits) */
        /* 5.12: in tiles of 64 x 64 and their mirror (column by column, the mirrored accesses
         * went down a row: a cache line and a page per entry; 7 % of maxG55's run) */
        const int TB = 64, nt = (n + TB - 1) / TB;
        #pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
        for (int tj = 0; tj < nt; tj++) {
            const int j0 = tj * TB, j1 = j0 + TB < n ? j0 + TB : n;
            for (int ti = 0; ti <= tj; ti++) {
                const int i0 = ti * TB, i1 = i0 + TB < n ? i0 + TB : n;
                for (int j = j0; j < j1; j++) {
                    const int ie = ti == tj ? j : i1;
                    for (int i = i0; i < ie; i++) {
                        double s = 0.5 * (A[i + (size_t)j * n] + A[j + (size_t)i * n]);
                        A[i + (size_t)j * n] = s; A[j + (size_t)i * n] = s;
                    }
                }
            }
        }
        return;
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i < j; i++) {
            double s = 0.5 * (A[i + j * n] + A[j + i * n]);
            A[i + j * n] = s; A[j + i * n] = s;
        }
}
static void lower_to_full(int n, double *A) {
    if (n > 1500) {
        const int TB = 64, nt = (n + TB - 1) / TB;          /* 5.12: in tiles (see symmetrize) */
        #pragma omp parallel for schedule(dynamic, 1)
        for (int tj = 0; tj < nt; tj++) {
            const int j0 = tj * TB, j1 = j0 + TB < n ? j0 + TB : n;
            for (int ti = 0; ti <= tj; ti++) {
                const int i0 = ti * TB, i1 = i0 + TB < n ? i0 + TB : n;
                for (int j = j0; j < j1; j++) {
                    const int ie = ti == tj ? j : i1;
                    for (int i = i0; i < ie; i++) A[i + (size_t)j * n] = A[j + (size_t)i * n];
                }
            }
        }
        return;
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i < j; i++) A[i + j * n] = A[j + i * n];
}
#define PROD_SMALL 16          /* inline dense kernels below this block size */
/* 4.24: below this order the per-block kernels avoid OpenBLAS level-3 and LAPACK calls:
 * those take a global buffer lock (2 threads each calling dtrmm at n = 5: 3x slower per
 * call, dpotrf 4.4x, dpotri 7x, dsytrd 6x, dsymv 9x; dgemm and dgemv scale) */
#define LOCKFREE_N 64
static int g_lockfree = 1;     /* BRISK_LOCKFREE=0: the 4.23 kernel choices */

/* inverse of a symmetric positive definite matrix from its lower Cholesky factor */
static void inv_from_chol_small(int n, double *A) {
    double inv[LOCKFREE_N * LOCKFREE_N];
    for (int j = 0; j < n; j++) {                      /* invert L into `inv` (lower) */
        inv[j + j * n] = 1.0 / A[j + j * n];
        for (int i = j + 1; i < n; i++) {
            double s = 0;
            for (int k = j; k < i; k++) s -= A[i + k * n] * inv[k + j * n];
            inv[i + j * n] = s / A[i + i * n];
        }
    }
    for (int j = 0; j < n; j++)                        /* A = inv' * inv (lower part) */
        for (int i = j; i < n; i++) {
            double s = 0;
            for (int k = i; k < n; k++) s += inv[k + i * n] * inv[k + j * n];
            A[i + j * n] = s;
        }
}

static int chol_lower(int n, double *A) {
    if (n <= (g_lockfree ? LOCKFREE_N : 12)) {   /* inline: no LAPACK call overhead or lock */
        for (int j = 0; j < n; j++) {
            double d = A[j + j * n];
            for (int k = 0; k < j; k++) d -= A[j + k * n] * A[j + k * n];
            if (!(d > 0)) return j + 1;
            d = sqrt(d);
            A[j + j * n] = d;
            for (int i = j + 1; i < n; i++) {
                double v = A[i + j * n];
                for (int k = 0; k < j; k++) v -= A[i + k * n] * A[j + k * n];
                A[i + j * n] = v / d;
            }
        }
        return 0;
    }
    return brisk_dpotrf(n, A, n);
}
/* 5.12: passes over a large block on the threads. Element-wise loops take PAR_LEN (each entry
 * computed as before, so the bits do not change); sums over 2 PN_CH entries or more go in
 * fixed chunks of PN_CH, each summed in eight fixed lanes, the chunk sums added in chunk
 * order: the same bits for any thread count, inside a parallel region or not. Blocks below
 * order 362 keep the serial sums. (The serial dot product was a chain of dependent adds, about
 * 8 ms per pass at n = 2000 against about 1 ms on four threads.) */
#define PN_CH 65536
#define DO_PRAGMA_(x) _Pragma(#x)
/* 5.12: an element-wise loop over one block on the threads when it is long, else the plain loop.
 * An OpenMP region whose if clause is false is still entered (a team of one is set up): with
 * thousands of small blocks (AC OPF relaxations after the chordal conversion) one region per
 * block and loop cost a third of the embedding's dense work (case500_goc 0.54 -> 0.80 s). */
#define PAR_LEN_FOR(len, ...) do { const size_t pl_n_ = (size_t)(len); \
    if (pl_n_ >= 2 * (size_t)PN_CH && !omp_in_parallel()) { _Pragma("omp parallel for schedule(static)") for (size_t i = 0; i < pl_n_; i++) { __VA_ARGS__ } } \
    else { for (size_t i = 0; i < pl_n_; i++) { __VA_ARGS__ } } } while (0)
static inline double dot_lanes(size_t len, const double *restrict a, const double *restrict b) {
    double l[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, t = 0;
    size_t i = 0;
    for (; i + 8 <= len; i += 8) for (int j = 0; j < 8; j++) l[j] += a[i + j] * b[i + j];
    for (; i < len; i++) t += a[i] * b[i];
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7])) + t;
}
static double ddot_n(size_t len, const double *a, const double *b) {
    if (len < 2 * (size_t)PN_CH) {
        double s = 0;
        for (size_t i = 0; i < len; i++) s += a[i] * b[i];
        return s;
    }
    const long nch = (long)((len + PN_CH - 1) / PN_CH);
    double *part = malloc(sizeof(double) * nch), s = 0;
    #pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (long c = 0; c < nch; c++) {
        const size_t i0 = (size_t)c * PN_CH, l = len - i0 < (size_t)PN_CH ? len - i0 : (size_t)PN_CH;
        part[c] = dot_lanes(l, a + i0, b + i0);
    }
    for (long c = 0; c < nch; c++) s += part[c];
    free(part);
    return s;
}
/* dst = src on the threads (large blocks) */
static void pcopy(size_t len, double *dst, const double *src) {
    if (len < 2 * (size_t)PN_CH) { memcpy(dst, src, sizeof(double) * len); return; }
    const long nch = (long)((len + PN_CH - 1) / PN_CH);
    #pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (long c = 0; c < nch; c++) {
        const size_t i0 = (size_t)c * PN_CH, l = len - i0 < (size_t)PN_CH ? len - i0 : (size_t)PN_CH;
        memcpy(dst + i0, src + i0, sizeof(double) * l);
    }
}
static void pzero(size_t len, double *dst) {
    if (len < 2 * (size_t)PN_CH) { memset(dst, 0, sizeof(double) * len); return; }
    const long nch = (long)((len + PN_CH - 1) / PN_CH);
    #pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (long c = 0; c < nch; c++) {
        const size_t i0 = (size_t)c * PN_CH, l = len - i0 < (size_t)PN_CH ? len - i0 : (size_t)PN_CH;
        memset(dst + i0, 0, sizeof(double) * l);
    }
}
/* sum_i (X_i + a dX_i)(Z_i + a dZ_i) and sum_i (a_i - b_i)^2 for large blocks (the embedding's
 * trial complementarity and its second-order passes): fixed chunks and lanes, as ddot_n */
#define PN_CHUNKED(len, EXPR) do {                                                                  \
        const long nch_ = (long)(((len) + PN_CH - 1) / PN_CH);                                        \
        double *part_ = malloc(sizeof(double) * nch_);                                                 \
        _Pragma("omp parallel for schedule(static) if (!omp_in_parallel())")                          \
        for (long c_ = 0; c_ < nch_; c_++) {                                                           \
            const size_t i0_ = (size_t)c_ * PN_CH, l_ = (len) - i0_ < (size_t)PN_CH ? (len) - i0_ : (size_t)PN_CH; \
            double lv_[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, t_ = 0;                                        \
            size_t j_ = 0;                                                                             \
            for (; j_ + 8 <= l_; j_ += 8) for (int q_ = 0; q_ < 8; q_++) { const size_t i = i0_ + j_ + q_; lv_[q_] += (EXPR); } \
            for (; j_ < l_; j_++) { const size_t i = i0_ + j_; t_ += (EXPR); }                        \
            part_[c_] = ((lv_[0] + lv_[1]) + (lv_[2] + lv_[3])) + ((lv_[4] + lv_[5]) + (lv_[6] + lv_[7])) + t_; \
        }                                                                                              \
        double s_ = 0; for (long c_ = 0; c_ < nch_; c_++) s_ += part_[c_];                            \
        free(part_); return s_;                                                                        \
    } while (0)
static double dot_step(size_t len, const double *X, const double *dX, const double *Z, const double *dZ, double a) {
    if (len < 2 * (size_t)PN_CH) {
        double t = 0;
        for (size_t i = 0; i < len; i++) t += (X[i] + a * dX[i]) * (Z[i] + a * dZ[i]);
        return t;
    }
    PN_CHUNKED(len, (X[i] + a * dX[i]) * (Z[i] + a * dZ[i]));
}
static double dist2(size_t len, const double *a, const double *b) {
    if (len < 2 * (size_t)PN_CH) {
        double t = 0;
        for (size_t i = 0; i < len; i++) t += (a[i] - b[i]) * (a[i] - b[i]);
        return t;
    }
    PN_CHUNKED(len, (a[i] - b[i]) * (a[i] - b[i]));
}
/* F <- F + Z, returning ||F + Z||^2 (the residual pass of the standard loop) */
static double add_norm2(size_t len, double *F, const double *Z) {
    if (len < 2 * (size_t)PN_CH) {
        double ncr = 0;
        for (size_t i = 0; i < len; i++) { double v = F[i] + Z[i]; ncr += v * v; F[i] = v; }
        return ncr;
    }
    const long nch = (long)((len + PN_CH - 1) / PN_CH);
    double *part = malloc(sizeof(double) * nch), s = 0;
    #pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (long c = 0; c < nch; c++) {
        const size_t i0 = (size_t)c * PN_CH, l = len - i0 < (size_t)PN_CH ? len - i0 : (size_t)PN_CH;
        double *restrict f = F + i0; const double *restrict z = Z + i0;
        double lv[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, t = 0;
        size_t i = 0;
        for (; i + 8 <= l; i += 8) for (int j = 0; j < 8; j++) { const double v = f[i + j] + z[i + j]; lv[j] += v * v; f[i + j] = v; }
        for (; i < l; i++) { const double v = f[i] + z[i]; t += v * v; f[i] = v; }
        part[c] = ((lv[0] + lv[1]) + (lv[2] + lv[3])) + ((lv[4] + lv[5]) + (lv[6] + lv[7])) + t;
    }
    for (long c = 0; c < nch; c++) s += part[c];
    free(part);
    return s;
}

static int g_free_hsd = 0;  /* 4.24: embedding chosen for split free pairs in this solve */
static int g_noflat = 0;
static int g_fl_maxpcg = 4;   /* 4.31: single precision is dropped once a PCG run needs more steps than this (was 8) */   /* 4.23: BRISK_NOFLAT: per-constraint SpSym loops in blk_Aop / blk_ATy */
/* 4.24: order-preserving parallel A(.): inside a parallel block loop, blk_Aop into a
 * registered output writes each block's values to its own slots; aop_end then adds them
 * per constraint in block order, the exact sequence of additions of the serial loop. */
#define AOP_NS 4
static struct { const Block *blk0; int nb, m; size_t *boff; int *cptr; size_t *cidx; } g_am;
static int g_aop_on = 0; static double *g_aop_out[AOP_NS]; static double *g_aop_vals[AOP_NS]; static size_t g_aop_cap = 0;
static void aop_map_build(const Problem *P) {
    if (g_am.blk0 == P->blk && g_am.nb == P->nblk && g_am.m == P->m && g_am.boff) return;
    free(g_am.boff); free(g_am.cptr); free(g_am.cidx);
    const int nb = P->nblk, m = P->m;
    g_am.blk0 = P->blk; g_am.nb = nb; g_am.m = m;
    g_am.boff = malloc(sizeof(size_t) * (nb + 1));
    g_am.cptr = calloc(m + 1, sizeof(int));
    size_t tot = 0;
    for (int k = 0; k < nb; k++) { g_am.boff[k] = tot; tot += P->blk[k].ncon; for (int t = 0; t < P->blk[k].ncon; t++) g_am.cptr[P->blk[k].con[t] + 1]++; }
    g_am.boff[nb] = tot;
    for (int i = 0; i < m; i++) g_am.cptr[i + 1] += g_am.cptr[i];
    g_am.cidx = malloc(sizeof(size_t) * (tot + 1));
    int *fill = malloc(sizeof(int) * (m + 1)); memcpy(fill, g_am.cptr, sizeof(int) * m);
    for (int k = 0; k < nb; k++) for (int t = 0; t < P->blk[k].ncon; t++) g_am.cidx[fill[P->blk[k].con[t]]++] = g_am.boff[k] + t;
    free(fill);
    if (tot + 1 > g_aop_cap) {
        for (int b = 0; b < AOP_NS; b++) { free(g_aop_vals[b]); g_aop_vals[b] = calloc(tot + 1, sizeof(double)); }
        g_aop_cap = tot + 1;
    }
}
static int g_hpar;
static void aop_begin(double *o0, double *o1, double *o2, double *o3) {
    if (!g_hpar) return;       /* serial loops add straight into the outputs: the same sums */
    g_aop_out[0] = o0; g_aop_out[1] = o1; g_aop_out[2] = o2; g_aop_out[3] = o3; g_aop_on = 1;
}
static void aop_end(void) {
    if (!g_aop_on) return;
    g_aop_on = 0;
    const int m = g_am.m;
    for (int b = 0; b < AOP_NS; b++) {
        double *out = g_aop_out[b];
        if (!out) continue;
        const double *v = g_aop_vals[b];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            double acc = out[i];
            for (int e = g_am.cptr[i]; e < g_am.cptr[i + 1]; e++) acc += v[g_am.cidx[e]];
            out[i] = acc;
        }
        g_aop_out[b] = NULL;
    }
}
static void blk_Aop_serial(const Block *B, BS *s, const double *D, double *out, double scale, const int *cpos);
/* out[con] += scale * <A_con, D>   (D may be nonsymmetric) */
static void blk_Aop(const Block *B, BS *s, const double *D, double *out, double scale) {
    if (g_aop_on && B >= g_am.blk0 && B < g_am.blk0 + g_am.nb) {
        for (int b = 0; b < AOP_NS; b++)
            if (g_aop_out[b] == out) { blk_Aop_serial(B, s, D, g_aop_vals[b] + g_am.boff[B - g_am.blk0], scale, NULL); return; }
    }
    blk_Aop_serial(B, s, D, out, scale, B->con);
}
/* cpos = B->con: out[con[t]] += ...; cpos = NULL: out[t] = ... (the block's own slots) */
#define AOUT(t, val) do { if (cpos) out[cpos[t]] += (val); else out[t] = (val); } while (0)
/* 23 October 2026: an LP or cone block with at least this many entries has its products on the threads (outside a
 * parallel region): <A_t, D> per constraint, each sum in its serial order; A'y by variable through B->ltr_* */
#define LP_PAR_NNZ (1L << 20)
static long blk_nnz(const Block *B) { long z = 0; for (int t = 0; t < B->ncon; t++) z += B->A[t].nnz; return z; }
/* the rows of A' of a large LP block, in constraint order (the order in which the serial A'y adds them) */
static void lp_rows_build(Block *B) {
    if (B->ltr_off || B->type != BLK_LP || blk_nnz(B) < LP_PAR_NNZ || getenv("BRISK_LPPAR0")) return;
    const int n = B->n; const long z = blk_nnz(B);
    int *off = calloc((size_t)n + 1, sizeof(int)), *tt = malloc(sizeof(int) * (size_t)z); double *vv = malloc(sizeof(double) * (size_t)z);
    if (!off || !tt || !vv || z > 2000000000L) { free(off); free(tt); free(vv); return; }
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) off[B->A[t].row[q] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    int *pos = malloc(sizeof(int) * (size_t)n); if (!pos) { free(off); free(tt); free(vv); return; }
    memcpy(pos, off, sizeof(int) * (size_t)n);
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].nnz; q++) { const int i = B->A[t].row[q]; tt[pos[i]] = t; vv[pos[i]++] = B->A[t].val[q]; }
    free(pos);
    B->ltr_off = off; B->ltr_t = tt; B->ltr_v = vv;
}
static void blk_Aop_serial(const Block *B, BS *s, const double *D, double *out, double scale, const int *cpos) {
    int n = B->n;
    if (B->type != BLK_SDP) {             /* LP and second-order cone blocks: vectors */
        const int par = B->ltr_off && nthreads() > 1 && !in_par();
        #pragma omp parallel for schedule(dynamic, 64) if (par)
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            double v = 0;
            for (int q = 0; q < S->nnz; q++) v += S->val[q] * D[S->row[q]];
            AOUT(t, scale * v);
        }
        return;
    }
    if (B->lfoff && !g_noflat) {
        /* 4.23: the block's flattened entries (contiguous, built by analyze_sdp_block) instead
         * of one SpSym per constraint: its row/col/val arrays are three scattered loads per
         * constraint (0.85 us per 3x3 block in blk_ATy on TSSOS case6468). Same entries in
         * the same order as the loop below: bitwise the same sums. */
        const int *lo = B->lfoff, *lr = B->lfr, *lc = B->lfc, *sl = B->slist;
        const double *lv = B->lfv;
        for (int a = 0; a < B->ns; a++) {
            double v = 0;
            for (int q = lo[a]; q < lo[a + 1]; q++) {
                const int p = lr[q], c = lc[q];
                v += (p == c) ? lv[q] * D[p + p * n] : lv[q] * (D[p + c * n] + D[c + p * n]);
            }
            AOUT(sl[a], scale * v);
        }
    } else
    for (int a = 0; a < B->ns; a++) {
        const SpSym *S = &B->A[B->slist[a]];
        double v = 0;
        for (int q = 0; q < S->nnz; q++) {
            int p = S->row[q], c = S->col[q];
            v += (p == c) ? S->val[q] * D[p + p * n]
                          : S->val[q] * (D[p + c * n] + D[c + p * n]);
        }
        AOUT(B->slist[a], scale * v);
    }
    if (B->nd && B->dfac) {
        /* 5.18: through the factors: <A_a, D> = sum_k s_k w_k' D w_k (D may be nonsymmetric: the same value) */
        const int nd = B->nd, Rd = B->dfoff[nd];
        double *T = malloc(sizeof(double) * (size_t)n * (Rd ? Rd : 1));
        BL(dgemm_)("N", "N", &n, &Rd, &n, &DONE, D, &n, B->dfW, &n, &DZERO, T, &n);
        for (int a = 0; a < nd; a++) {
            double v = 0;
            for (int k = B->dfoff[a]; k < B->dfoff[a + 1]; k++) v += B->dfS[k] * BL(ddot_)(&n, B->dfW + (size_t)k * n, &IONE, T + (size_t)k * n, &IONE);
            AOUT(B->dlist[a], scale * v);
        }
        free(T);
    } else if (B->nd && B->Adl) {
        /* 17 October 2026: <A_a, D> = sum_i A_ii D_ii + sum_{i>j} A_ij (D_ij + D_ji) over the packed lower triangles */
        const int nl = n * (n + 1) / 2, nd = B->nd;
        double Dl[ADL_MAXN * (ADL_MAXN + 1) / 2];
        int k = 0;
        for (int j = 0; j < n; j++) { Dl[k++] = D[j + (size_t)j * n]; for (int i = j + 1; i < n; i++) Dl[k++] = D[i + (size_t)j * n] + D[j + (size_t)i * n]; }
        for (int a = 0; a < nd; a++) {
            const double *c = B->Adl + (size_t)a * nl; double v = 0;
            for (int q = 0; q < nl; q++) v += c[q] * Dl[q];
            AOUT(B->dlist[a], scale * v);
        }
    } else if (B->nd) {
        int n2 = n * n, nd = B->nd;
        BL(dgemv_)("T", &n2, &nd, &scale, B->Ad, &n2, D, &IONE, &DZERO, s->dtmp, &IONE);
        for (int a = 0; a < nd; a++) AOUT(B->dlist[a], s->dtmp[a]);
    }
}

/* F = sum_i y_i A_i (dense full / vector) */
static void blk_ATy(const Block *B, BS *s, const double *y, double *F) {
    int n = B->n;
    if (B->type != BLK_SDP) {
        if (B->ltr_off && nthreads() > 1 && !in_par()) {
            /* (the same additions in the same order per variable as the loop below: bit for bit the same F) */
            const int *off = B->ltr_off, *tt = B->ltr_t, *con = B->con; const double *vv = B->ltr_v;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) { double f = 0; for (int e = off[i]; e < off[i + 1]; e++) f += y[con[tt[e]]] * vv[e]; F[i] = f; }
            return;
        }
        memset(F, 0, sizeof(double) * n);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            double yi = y[B->con[t]];
            for (int q = 0; q < S->nnz; q++) F[S->row[q]] += yi * S->val[q];
        }
        return;
    }
    if (B->nd && B->dfac) {
        /* 5.18: sum_a y_a A_a = W diag(s y) W' through the factors */
        const int nd = B->nd, Rd = B->dfoff[nd];
        double *Wy = malloc(sizeof(double) * (size_t)n * (Rd ? Rd : 1));
        for (int a = 0; a < nd; a++) {
            const double ya = y[B->con[B->dlist[a]]];
            for (int k = B->dfoff[a]; k < B->dfoff[a + 1]; k++) {
                const double c = ya * B->dfS[k];
                const double *w = B->dfW + (size_t)k * n; double *o = Wy + (size_t)k * n;
                for (int i = 0; i < n; i++) o[i] = c * w[i];
            }
        }
        BL(dgemm_)("N", "T", &n, &n, &Rd, &DONE, Wy, &n, B->dfW, &n, &DZERO, F, &n);
        free(Wy);
    } else if (B->nd && B->Adl) {
        /* 17 October 2026: the packed lower triangles (see blk_Aop_serial), then mirrored */
        const int nl = n * (n + 1) / 2, nd = B->nd;
        double Fl[ADL_MAXN * (ADL_MAXN + 1) / 2];
        for (int q = 0; q < nl; q++) Fl[q] = 0.0;
        for (int a = 0; a < nd; a++) {
            const double ya = y[B->con[B->dlist[a]]];
            if (ya == 0.0) continue;
            const double *c = B->Adl + (size_t)a * nl;
            for (int q = 0; q < nl; q++) Fl[q] += ya * c[q];
        }
        int k = 0;
        for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { const double v = Fl[k++]; F[i + (size_t)j * n] = v; F[j + (size_t)i * n] = v; }
    } else if (B->nd) {
        int n2 = n * n, nd = B->nd;
        for (int a = 0; a < nd; a++) s->dtmp[a] = y[B->con[B->dlist[a]]];
        BL(dgemv_)("N", &n2, &nd, &DONE, B->Ad, &n2, s->dtmp, &IONE, &DZERO, F, &IONE);
    } else pzero((size_t)n * n, F);
    if (B->foff && !g_noflat) {
        const int *foff = B->foff, *ffr = B->ffr, *ffc = B->ffc, *sl = B->slist, *con = B->con;
        const double *ffv = B->ffv;
        for (int a = 0; a < B->ns; a++) {
            const double yi = y[con[sl[a]]];
            if (yi == 0) continue;
            for (int k = foff[a]; k < foff[a + 1]; k++) F[ffr[k] + (size_t)ffc[k] * n] += yi * ffv[k];
        }
        return;
    }
    for (int a = 0; a < B->ns; a++) {
        const SpSym *S = &B->A[B->slist[a]];
        double yi = y[B->con[B->slist[a]]];
        if (yi == 0) continue;
        for (int q = 0; q < S->ef; q++) F[S->fr[q] + (size_t)S->fc[q] * n] += yi * S->fv[q];
    }
}

static void sp_add(const Block *B, const SpSym *S, double a, double *D) {
    int n = B->n;
    if (B->type != BLK_SDP)
        for (int q = 0; q < S->nnz; q++) D[S->row[q]] += a * S->val[q];
    else
        for (int q = 0; q < S->ef; q++) D[S->fr[q] + (size_t)S->fc[q] * n] += a * S->fv[q];
}

static double sp_inner(const Block *B, const SpSym *S, const double *D) {
    int n = B->n;
    double s = 0;
    if (B->type != BLK_SDP)
        for (int q = 0; q < S->nnz; q++) s += S->val[q] * D[S->row[q]];
    else
        for (int q = 0; q < S->ef; q++) s += S->fv[q] * D[S->fr[q] + (size_t)S->fc[q] * n];
    return s;
}

static long cnt_prod3 = 0, cnt_zfr = 0, cnt_refine_try = 0;
/* 5.12 debug (BRISK_ZTIME): time per phase of the standard loop */
static double g_zt[12]; static int g_ztime = -1;
static const char *g_ztn[12] = { "Zi", "residuals", "predictor", "pred steps X", "pred steps Z", "corrector rhs", "corrector dX", "refine", "corr steps X", "corr steps Z", "try X", "try Z" };
static double g_zpt[2];
static void ztime_print(void) { fprintf(stderr, "   [loop phases:"); for (int i = 0; i < 12; i++) fprintf(stderr, " %s %.3f", g_ztn[i], g_zt[i]); fprintf(stderr, "; sparse products: F R %.3f, solves %.3f]\n", g_zpt[0], g_zpt[1]); }
#define ZT_ON() (g_ztime < 0 ? (g_ztime = getenv("BRISK_ZTIME") != NULL, g_ztime ? atexit(ztime_print) : 0, g_ztime) : g_ztime)
/* out = A * B * C with A, C symmetric (full storage), B general. W workspace. */
static int g_p3chk = -1; static double g_p3asym = 0;
static void p3chk_print(void) { printf("   [prod3 symmetry check (exit): max relative asymmetry %.2e]\n", g_p3asym); }
static void prod3(int n, const double *A, const double *Bm, const double *C, double *W, double *out) {
    #pragma omp atomic
    cnt_prod3++;
    if (g_p3chk < 0) { g_p3chk = getenv("BRISK_P3CHK") != NULL; if (g_p3chk) atexit(p3chk_print); }
    if (g_p3chk) {
        for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) {
            double da = fabs(A[i + (size_t)j * n] - A[j + (size_t)i * n]) / (fabs(A[i + (size_t)j * n]) + 1e-300);
            double dc = fabs(C[i + (size_t)j * n] - C[j + (size_t)i * n]) / (fabs(C[i + (size_t)j * n]) + 1e-300);
            if (fabs(A[i + (size_t)j * n]) > 1e-300 && da > g_p3asym) g_p3asym = da;
            if (fabs(C[i + (size_t)j * n]) > 1e-300 && dc > g_p3asym) g_p3asym = dc;
        }
    }
    /* 4.23: A and C are kept in full symmetric storage (checked with BRISK_P3CHK on the dev
     * set: asymmetry exactly 0), so two dgemm (OpenBLAS small-matrix kernels) replace the
     * symmetric products: 0.05 against 0.14 us at n = 3, 2.1 against 13.7 us at n = 32 */
    if (g_xsymm_loops) {
        xsymm("L", "L", &n, &n, &DONE, A, &n, Bm, &n, &DZERO, W, &n);
        xsymm("R", "L", &n, &n, &DONE, C, &n, W, &n, &DZERO, out, &n);
        return;
    }
    pdgemm("N", "N", &n, &n, &n, &DONE, A, &n, Bm, &n, &DZERO, W, &n);
    pdgemm("N", "N", &n, &n, &n, &DONE, W, &n, C, &n, &DZERO, out, &n);
}

/* row-subset route work arrays: r x n (Fr, T1) and n x r (Zc) */
/* D += theta * R0 (R0 stored dense, or as a sparse list for SDP blocks) */
static void add_R0(const BS *s, size_t len, double theta, double *D) {
    if (theta == 0) return;
    if (s->R0) { for (size_t i = 0; i < len; i++) D[i] += theta * s->R0[i]; return; }
    for (int e = 0; e < s->r0n; e++) D[s->r0r[e] + (size_t)s->r0c[e] * s->n] += theta * s->r0v[e];
}

/* sum_i R0_i D_i (R0 dense or the sparse list of add_R0) */
static double dot_R0(const BS *s, size_t len, const double *D) {
    double t = 0;
    if (s->R0) { for (size_t i = 0; i < len; i++) t += s->R0[i] * D[i]; return t; }
    for (int e = 0; e < s->r0n; e++) t += s->r0v[e] * D[s->r0r[e] + (size_t)s->r0c[e] * s->n];
    return t;
}

/* sum_i (D_i + theta R0_i)^2 */
static double norm2_plus_R0(const BS *s, size_t len, const double *D, double theta) {
    if (s->type == BLK_SDP && len >= 2 * (size_t)PN_CH) {
        /* 5.12: in chunks of whole columns on the threads, each column formed in a buffer and
         * summed in eight lanes, the chunk sums in order (the same bits for any thread count) */
        const int n = s->n, cpc = PN_CH / n > 0 ? PN_CH / n : 1, nch = (n + cpc - 1) / cpc;
        double *part = malloc(sizeof(double) * nch), v = 0;
        #pragma omp parallel if (!omp_in_parallel())
        {
            double *col = malloc(sizeof(double) * n);
            #pragma omp for schedule(static)
            for (int c = 0; c < nch; c++) {
                const int j0 = c * cpc, j1 = j0 + cpc < n ? j0 + cpc : n;
                int e = 0;
                if (!s->R0) { int lo = 0, hi = s->r0n; while (lo < hi) { const int mid = (lo + hi) / 2; if (s->r0c[mid] < j0) lo = mid + 1; else hi = mid; } e = lo; }
                double acc = 0;
                for (int j = j0; j < j1; j++) {
                    const double *dj = D + (size_t)j * n;
                    if (s->R0) { const double *rj = s->R0 + (size_t)j * n; for (int i = 0; i < n; i++) col[i] = dj[i] + theta * rj[i]; }
                    else {
                        memcpy(col, dj, sizeof(double) * n);
                        for (; e < s->r0n && s->r0c[e] == j; e++) col[s->r0r[e]] += theta * s->r0v[e];
                    }
                    acc += dot_lanes(n, col, col);
                }
                part[c] = acc;
            }
            free(col);
        }
        for (int c = 0; c < nch; c++) v += part[c];
        free(part);
        return v;
    }
    if (s->R0) {
        double v = 0;
        for (size_t i = 0; i < len; i++) { double t = D[i] + theta * s->R0[i]; v += t * t; }
        return v;
    }
    /* one pass in column-major order (the order of the sparse list): computing
     * ||D||^2 + sum((d + theta r)^2 - d^2) instead cancels to sqrt(eps) ||D|| */
    double v = 0;
    const int n = s->n;
    int e = 0;
    if (s->type != BLK_SDP) {
        for (int i = 0; i < n; i++) {
            double t = D[i];
            if (e < s->r0n && s->r0r[e] == i) t += theta * s->r0v[e++];
            v += t * t;
        }
        return v;
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double t = D[i + (size_t)j * n];
            if (e < s->r0n && s->r0c[e] == j && s->r0r[e] == i) t += theta * s->r0v[e++];
            v += t * t;
        }
    return v;
}

static void rowbuf_reserve(BS *s, int r) {
    if (r <= s->rcap) return;
    free(s->Fr); free(s->T1); free(s->Zc);
    size_t len = sizeof(double) * (size_t)(r > 0 ? r : 1) * s->n;
    s->Fr = amalloc(len); s->T1 = amalloc(len); s->Zc = amalloc(len);
    s->rcap = r;
}

static int pat_route(int n, double nf, int nr, double cs, double cb) {
    double n2 = (double)n * n, n3 = n2 * n;
    double c_row = cb * 2.0 * nr * n2;
    double c_sl = 0.2 * cs * nf * n + cb * n3;
    double c_dn = cb * 2.0 * n3;
    if (c_row <= c_sl && c_row <= c_dn) return 0;
    return (c_sl <= c_dn) ? 1 : 2;
}

/* 5.12, dense X and sparse Z: out = (Zi F R)' through the sparse factor of Z. Column i of
 * T = (F R)' is sum_k F(i, k) R(:, k) over the pattern of Z (R symmetric), so T holds the rows
 * of F R as columns, the layout of schol_solve_rows, which applies Z^{-1} to them: 2 nnz(F) n
 * + 4 nnz(L) n flops against the 2 n^3 of a product with the dense Zi. Every caller uses the
 * result symmetrized or through A(.) of symmetric matrices, where the transpose is the same.
 * The columns of T are formed independently (any thread count gives the same bits).        */
static void prod_zsp(int n, BS *s, const double *F, const double *Rm, double *out) {
    double *T = (out == F || out == Rm) ? s->W : out;
    const double zt0_ = omp_get_wtime();
    #pragma omp parallel for schedule(static) if (n >= 256 && !omp_in_parallel())
    for (int i = 0; i < n; i++) {
        double *restrict ti = T + (size_t)i * n;
        memset(ti, 0, sizeof(double) * n);
        for (int q = s->zcp[i]; q < s->zcp[i + 1]; q++) {
            const int k = s->zri[q];
            const double f = F[i + (size_t)k * n];
            if (f == 0) continue;
            const double *restrict rk = Rm + (size_t)k * n;
            for (int r = 0; r < n; r++) ti[r] += f * rk[r];
        }
    }
    const double zt1_ = omp_get_wtime();
    schol_solve_rows(s->zf, T, n, n);
    if (T != out) memcpy(out, T, sizeof(double) * (size_t)n * n);
    g_zpt[0] += zt1_ - zt0_; g_zpt[1] += omp_get_wtime() - zt1_;
}

static int zmerge_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("BRISK_ZMERGE"); on = !(e && e[0] == '0'); } return on; }
/* 5.12: prod_zsp of F1 R1 + F2 R2, where F2 is given on the pattern only (f2[q] for zri[q] in
 * column i): the corrector's Zi (dZc X + dZp dXp) in one set of sparse solves instead of two. */
static void prod_zsp2(int n, BS *s, const double *F1, const double *R1, const double *f2, const double *R2, double *out) {
    double *T = (out == F1 || out == R1 || out == R2) ? s->W : out;
    const double zt0_ = omp_get_wtime();
    #pragma omp parallel for schedule(static) if (n >= 256 && !omp_in_parallel())
    for (int i = 0; i < n; i++) {
        double *restrict ti = T + (size_t)i * n;
        memset(ti, 0, sizeof(double) * n);
        for (int q = s->zcp[i]; q < s->zcp[i + 1]; q++) {
            const int k = s->zri[q];
            const double f = F1[i + (size_t)k * n], g = f2[q];
            const double *restrict rk = R1 + (size_t)k * n, *restrict qk = R2 + (size_t)k * n;
            if (f != 0 && g != 0) for (int r = 0; r < n; r++) ti[r] += f * rk[r] + g * qk[r];
            else if (f != 0) for (int r = 0; r < n; r++) ti[r] += f * rk[r];
            else if (g != 0) for (int r = 0; r < n; r++) ti[r] += g * qk[r];
        }
    }
    const double zt1_ = omp_get_wtime();
    schol_solve_rows(s->zf, T, n, n);
    if (T != out) memcpy(out, T, sizeof(double) * (size_t)n * n);
    g_zpt[0] += zt1_ - zt0_; g_zpt[1] += omp_get_wtime() - zt1_;
}
/* 5.12: F (symmetric, on the pattern of zcp/zri) as values in the pattern's order: fq[q] =
 * F(zri[q], i) for q in column i (= F(i, zri[q])) */
static void zsp_compact(int n, const BS *s, const double *F, double *fq) {
    #pragma omp parallel for schedule(static) if (n >= 256 && !omp_in_parallel())
    for (int i = 0; i < n; i++) {
        const double *fi = F + (size_t)i * n;
        for (int q = s->zcp[i]; q < s->zcp[i + 1]; q++) fq[q] = fi[s->zri[q]];
    }
}
/* 5.12: the entries of Zi F R (F given by zsp_compact, R symmetric) on the constraints'
 * pattern, written to Q there (A(.) reads nothing else on these blocks): (Zi F R)_pc =
 * R(:, c)' (F Zi)(:, p), with the dense Zi of the iteration. nnz(F) n + 2 nnz(upat) n flops
 * against the 4 nnz(L) n of the sparse solves: the corrector's right-hand side needs only
 * A(Zi dZp dXp), and the product itself goes into the corrector's (prod_zsp2). */
static void aop_zifr(int n, BS *s, const double *fq, const double *R, double *Q) {
    /* the nonzeros of F by columns */
    int *fcp = malloc(sizeof(int) * (n + 1)), nz = 0;
    for (int k = 0; k < n; k++) { fcp[k] = nz; for (int q = s->zcp[k]; q < s->zcp[k + 1]; q++) nz += fq[q] != 0; }
    fcp[n] = nz;
    int *fri = malloc(sizeof(int) * (nz + 1)); double *fv = malloc(sizeof(double) * (nz + 1));
    for (int k = 0, t = 0; k < n; k++) for (int q = s->zcp[k]; q < s->zcp[k + 1]; q++) if (fq[q] != 0) { fri[t] = s->zri[q]; fv[t] = fq[q]; t++; }
    double *u = s->W;                               /* n per thread */
    #pragma omp parallel if (n >= 256 && !omp_in_parallel())
    {
        double *up = u + (size_t)omp_get_thread_num() * n;
        #pragma omp for schedule(dynamic, 16)
        for (int p = 0; p < n; p++) {
            if (s->ucp[p] == s->ucp[p + 1]) continue;
            const double *zp = s->Zi + (size_t)p * n;
            memset(up, 0, sizeof(double) * n);
            for (int k = 0; k < n; k++) {             /* up = F Zi(:, p) = sum_k Zi(k, p) F(:, k) */
                const double z = zp[k];
                for (int q = fcp[k]; q < fcp[k + 1]; q++) up[fri[q]] += fv[q] * z;
            }
            for (int a = s->ucp[p]; a < s->ucp[p + 1]; a++) {
                const int c = s->uri[a];
                const double *restrict rc = R + (size_t)c * n;
                double t = 0;
                #pragma omp simd reduction(+:t)
                for (int l = 0; l < n; l++) t += rc[l] * up[l];
                Q[p + (size_t)c * n] = t;
            }
        }
    }
    free(fcp); free(fri); free(fv);
}

/* out = Zi * F * R, where F is nonzero only on pattern pt.
 * Route 0: F has few nonzero rows -> two reduced dgemm (2 r n^2).
 * Route 1: F sparse -> Zi*F by column axpys (nnz*n), then one dsymm.
 * Route 2: dense -> two dsymm (2 n^3).                                   */
static void prod_pat(const Pat *pt, int n, BS *s, const double *F, const double *Rm, double *out) {
    if (s->zon) {
        #pragma omp atomic
        cnt_zfr++;
        prod_zsp(n, s, F, Rm, out);
        return;
    }
    if (pt->route == 2) { prod3(n, s->Zi, F, Rm, s->W, out); return; }
    #pragma omp atomic
    cnt_zfr++;
    if (pt->route == 1) {
        memset(s->W, 0, sizeof(double) * (size_t)n * n);
        for (int k = 0; k < pt->nf; k++) {
            double v = F[pt->fr[k] + (size_t)pt->fc[k] * n];
            if (v != 0) BL(daxpy_)(&n, &v, s->Zi + (size_t)pt->fr[k] * n, &IONE, s->W + (size_t)pt->fc[k] * n, &IONE);
        }
        xsymm("R", "L", &n, &n, &DONE, Rm, &n, s->W, &n, &DZERO, out, &n);
        return;
    }
    int r = pt->nr;
    if (r == 0) { memset(out, 0, sizeof(double) * (size_t)n * n); return; }
    const int *U = pt->rows;
    for (int j = 0; j < n; j++) {
        const double *fj = F + (size_t)j * n;
        double *dst = s->Fr + (size_t)j * r;
        for (int k = 0; k < r; k++) dst[k] = fj[U[k]];
    }
    pdgemm("N", "N", &r, &n, &n, &DONE, s->Fr, &r, Rm, &n, &DZERO, s->T1, &r);
    for (int k = 0; k < r; k++) memcpy(s->Zc + (size_t)k * n, s->Zi + (size_t)U[k] * n, sizeof(double) * n);
    pdgemm("N", "N", &n, &n, &r, &DONE, s->Zc, &n, s->T1, &r, &DZERO, out, &n);
}

/* Zi F R for any F on the pattern of Z: by the sparse factor of Z when the block has one (5.12) */
static void prod_zi(int n, BS *s, const double *F, const double *Rm, double *out) {
    if (s->zon) {
        #pragma omp atomic
        cnt_zfr++;
        prod_zsp(n, s, F, Rm, out);
        return;
    }
    prod3(n, s->Zi, F, Rm, s->W, out);
}

static void prod_ZFR(const Block *B, BS *s, const double *F, const double *Rm, double *out) {
    prod_pat(&s->upat, B->n, s, F, Rm, out);
}

/* out[con] += scale * <A_con, Zi R0 X>, evaluating Zi R0 X only on the
 * constraint pattern: cost |pattern| * nnz(R0) instead of n^3.            */
static void Aop_ZR0X(const Block *B, BS *s, double scale, double *out) {
    const int n = B->n;
    const Pat *u = &s->upat;
    memset(s->G, 0, sizeof(double) * (size_t)n * n);
    for (int k = 0; k < u->nf; k++) {
        const double *zp = s->Zi + (size_t)u->fr[k] * n;      /* Zi[p, r] = Zi[r, p] */
        const double *xq = s->X + (size_t)u->fc[k] * n;       /* X[c, q] */
        double acc = 0;
        for (int e = 0; e < s->r0n; e++) acc += s->r0v[e] * zp[s->r0r[e]] * xq[s->r0c[e]];
        s->G[u->fr[k] + (size_t)u->fc[k] * n] = acc;
    }
    blk_Aop(B, s, s->G, out, scale);
}

/* Build the sparse description of R0 and the combined pattern (SDP block). */
static void setup_residual_pattern(const Block *B, BS *s, const Params *par) {
    const int n = B->n;
    const size_t len = (size_t)n * n;
    double amax = 0;
    for (size_t i = 0; i < len; i++) amax = fmax(amax, fabs(s->R0[i]));
    const double thr = 1e-13 * fmax(1.0, amax);
    unsigned char *mark = ccalloc(len);
    s->r0n = 0;
    for (size_t i = 0; i < len; i++) {
        if (fabs(s->R0[i]) <= thr) s->R0[i] = 0.0;
        else { mark[i] = 1; s->r0n++; }
    }
    free(s->r0r); free(s->r0c); free(s->r0v);
    s->r0r = amalloc(sizeof(int) * (s->r0n + 1));
    s->r0c = amalloc(sizeof(int) * (s->r0n + 1));
    s->r0v = amalloc(sizeof(double) * (s->r0n + 1));
    int c = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (mark[i + (size_t)j * n]) { s->r0r[c] = i; s->r0c[c] = j; s->r0v[c] = s->R0[i + (size_t)j * n]; c++; }
    if (s->r0n < (long)(len / 8)) { free(s->R0); s->R0 = NULL; }   /* keep only the sparse list */
    for (int k = 0; k < s->upat.nf; k++) mark[s->upat.fr[k] + (size_t)s->upat.fc[k] * n] = 1;
    if (s->cpat.owned) { free(s->cpat.fr); free(s->cpat.fc); free(s->cpat.rows); }
    size_t nf = 0;
    for (size_t i = 0; i < len; i++) nf += mark[i];
    s->cpat.nf = (int)nf;
    s->cpat.fr = amalloc(sizeof(int) * (nf + 1));
    s->cpat.fc = amalloc(sizeof(int) * (nf + 1));
    char *rm = ccalloc(n);
    c = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (mark[i + (size_t)j * n]) { s->cpat.fr[c] = i; s->cpat.fc[c] = j; c++; rm[i] = 1; }
    s->cpat.nr = 0;
    for (int i = 0; i < n; i++) s->cpat.nr += rm[i];
    s->cpat.rows = amalloc(sizeof(int) * (s->cpat.nr + 1));
    for (int i = 0, q = 0; i < n; i++) if (rm[i]) s->cpat.rows[q++] = i;
    s->cpat.route = pat_route(n, (double)nf, s->cpat.nr, par->c_sparse, par->c_blas);
    s->cpat.owned = 1;
    if (s->cpat.route == 0) rowbuf_reserve(s, s->cpat.nr);
    free(rm); free(mark);
    /* scalar restricted evaluation vs one BLAS-3 product */
    double c_restr = par->c_sparse * (double)s->upat.nf * s->r0n;
    double c_full = par->c_blas * (double)len * n;
    s->h_restricted = c_restr < c_full;
}

/* 5.12, dense X and sparse Z (HKM): Z = C - A'y + theta R0 keeps the pattern of C, the A_t
 * and the start Z0 (diagonal), so for max-cut, box-QP and structural-design blocks its
 * Cholesky factor under a fill-reducing ordering is sparse while X and Z^{-1} are dense.
 * The factor then gives Z^{-1} and its products (prod_zsp), the PSD test of a step and the
 * dual step length; X keeps its dense factor. Enabled when nnz(L) <= zsparse_fill n^2.
 * Returns 1 when the block uses it (s->zf factored at the current Z). */
static int cmp_int_asc(const void *a, const void *b) { const int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }
/* 5.12: pattern entry of (r, c) in the CSC pattern zcp/zri (sorted columns), or -1 */
static int zsp_find(const BS *s, int r, int c) {
    int lo = s->zcp[c], hi = s->zcp[c + 1];
    while (lo < hi) { const int mid = (lo + hi) / 2; if (s->zri[mid] < r) lo = mid + 1; else hi = mid; }
    return lo < s->zcp[c + 1] && s->zri[lo] == r ? lo : -1;
}
/* R0 on the pattern (after setup_residual_pattern); 0 if an entry falls outside it */
static int zsp_r0(BS *s) {
    const int n = s->n;
    memset(s->zr0q, 0, sizeof(double) * s->nzp);
    if (s->R0) { for (int q = 0; q < s->nzp; q++) s->zr0q[q] = s->R0[s->zpos[q]]; return 1; }   /* (zero off the pattern: see zsp_pattern_setup) */
    for (int e = 0; e < s->r0n; e++) { const int q = zsp_find(s, s->r0r[e], s->r0c[e]); if (q < 0) return 0; s->zr0q[q] = s->r0v[e]; }
    (void)n;
    return 1;
}
/* the positions, C and R0 on the pattern, the constraints' entries mapped to it; zpat = 0 when
 * an entry falls outside the pattern or the block has no flattened entries (dense passes then) */
static void zsp_pattern_setup(const Block *B, BS *s) {
    const int n = B->n, nzp = s->zcp[n];
    s->zpat = 0; s->nzp = nzp; s->zbc = s->zrc = 0;
    if (!B->foff || g_noflat || B->nd) return;
    s->zpos = malloc(sizeof(size_t) * (nzp + 1));
    for (int c = 0; c < n; c++) for (int q = s->zcp[c]; q < s->zcp[c + 1]; q++) s->zpos[q] = (size_t)s->zri[q] + (size_t)c * n;
    s->zfq = calloc(nzp + 1, sizeof(double)); s->zcq = calloc(nzp + 1, sizeof(double)); s->zr0q = calloc(nzp + 1, sizeof(double));
    const int nf = B->foff[B->ns];
    s->zamap = malloc(sizeof(int) * (nf + 1));
    for (int k = 0; k < nf; k++) if ((s->zamap[k] = zsp_find(s, B->ffr[k], B->ffc[k])) < 0) return;
    for (int q = 0; q < B->C.ef; q++) { const int p = zsp_find(s, B->C.fr[q], B->C.fc[q]); if (p < 0) return; s->zcq[p] += B->C.fv[q]; }
    if (s->R0) {                                         /* a dense R0 must vanish off the pattern */
        char *on = calloc((size_t)n * n, 1);
        for (int q = 0; q < nzp; q++) on[s->zpos[q]] = 1;
        for (size_t i = 0; i < (size_t)n * n; i++) if (!on[i] && s->R0[i] != 0) { free(on); return; }
        free(on);
    }
    if (!zsp_r0(s)) return;
    pzero((size_t)n * n, s->dZ);                         /* dZ is written on the pattern only from here */
    s->zpat = 1;
}
/* fq = A'y on the pattern: the additions of blk_ATy's flattened loop in its order */
static void zsp_aty(const Block *B, const BS *s, const double *y, double *fq) {
    memset(fq, 0, sizeof(double) * s->nzp);
    const int *foff = B->foff, *sl = B->slist, *con = B->con;
    const double *ffv = B->ffv;
    for (int a = 0; a < B->ns; a++) {
        const double yi = y[con[sl[a]]];
        if (yi == 0) continue;
        for (int k = foff[a]; k < foff[a + 1]; k++) fq[s->zamap[k]] += yi * ffv[k];
    }
}
/* dZ = -A'dy + theta R0 on the pattern (blk_ATy, the negation and add_R0 in one) */
static void zsp_dz(const Block *B, BS *s, const double *dy, double theta) {
    zsp_aty(B, s, dy, s->zfq);
    for (int q = 0; q < s->nzp; q++) {
        double v = -s->zfq[q];
        if (theta != 0) v += theta * s->zr0q[q];
        s->dZ[s->zpos[q]] = v;
    }
}
/* sum over the pattern of A .* B (A or B zero off it) */
static double zsp_dot(const BS *s, const double *A, const double *B) {
    double t = 0;
    for (int q = 0; q < s->nzp; q++) t += A[s->zpos[q]] * B[s->zpos[q]];
    return t;
}
static void zsp_copy(const BS *s, double *dst, const double *src) {
    for (int q = 0; q < s->nzp; q++) dst[s->zpos[q]] = src[s->zpos[q]];
}

static int zsp_setup(const Block *B, BS *s, const Params *par, int kblk) {
    const int n = B->n;
    if (B->type != BLK_SDP || par->zsparse_n <= 0 || n < par->zsparse_n || B->lowrank || B->dict || B->nd) return 0;
    const double cap = par->zsparse_fill * (double)n * n;
    int *deg = calloc(n, sizeof(int)), *cnt = calloc(n, sizeof(int)), *mk = malloc(sizeof(int) * n);
    /* entries (r, c), r != c: C, every A_t, the patterns of the products, Z itself */
    size_t ne = 0;
#define ZSP_ADD(r_, c_) do { const int r__ = (r_), c__ = (c_); if (r__ != c__) { cnt[r__]++; cnt[c__]++; ne++; } } while (0)
    /* first pass: counts (with duplicates) */
    for (int q = 0; q < B->C.ef; q++) ZSP_ADD(B->C.fr[q], B->C.fc[q]);
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) ZSP_ADD(B->A[t].fr[q], B->A[t].fc[q]);
    for (int q = 0; q < s->upat.nf; q++) ZSP_ADD(s->upat.fr[q], s->upat.fc[q]);
    for (int q = 0; q < s->cpat.nf; q++) ZSP_ADD(s->cpat.fr[q], s->cpat.fc[q]);
    for (int c = 0; c < n; c++) for (int r = c + 1; r < n; r++) if (s->Z[r + (size_t)c * n] != 0) ZSP_ADD(r, c);
#undef ZSP_ADD
    /* (entries counted with their repetitions: C, the A_t, both patterns and Z overlap) */
    if ((double)ne > 8.0 * cap) { free(deg); free(cnt); free(mk); return 0; }
    int **nbr = malloc(sizeof(int *) * n);
    for (int i = 0; i < n; i++) nbr[i] = malloc(sizeof(int) * (cnt[i] + 1));
#define ZSP_PUT(r_, c_) do { const int r__ = (r_), c__ = (c_); if (r__ != c__) { nbr[r__][deg[r__]++] = c__; nbr[c__][deg[c__]++] = r__; } } while (0)
    for (int q = 0; q < B->C.ef; q++) ZSP_PUT(B->C.fr[q], B->C.fc[q]);
    for (int t = 0; t < B->ncon; t++) for (int q = 0; q < B->A[t].ef; q++) ZSP_PUT(B->A[t].fr[q], B->A[t].fc[q]);
    for (int q = 0; q < s->upat.nf; q++) ZSP_PUT(s->upat.fr[q], s->upat.fc[q]);
    for (int q = 0; q < s->cpat.nf; q++) ZSP_PUT(s->cpat.fr[q], s->cpat.fc[q]);
    for (int c = 0; c < n; c++) for (int r = c + 1; r < n; r++) if (s->Z[r + (size_t)c * n] != 0) ZSP_PUT(r, c);
#undef ZSP_PUT
    for (int i = 0; i < n; i++) mk[i] = -1;
    size_t tot = 0;
    for (int i = 0; i < n; i++) {
        int d = 0;
        for (int a = 0; a < deg[i]; a++) { const int j = nbr[i][a]; if (mk[j] != i) { mk[j] = i; nbr[i][d++] = j; } }
        deg[i] = d; tot += d;
    }
    SChol *F = (double)tot <= cap ? schol_analyze_adj(n, deg, nbr, (size_t)cap + 1000) : NULL;
    if (getenv("BRISK_ZSPDBG")) printf("   [zsparse block %d: n %d, %zu entries off the diagonal, factor %zu (cap %.0f)]\n", kblk + 1, n, tot, F ? schol_nnz(F) : 0, cap);
    if (F && (double)schol_nnz(F) > cap) { schol_free(F); F = NULL; }
    if (F) {
        schol_gather_dense(F, s->Z);
        if (schol_factor(F, 0.0) != 0) { schol_free(F); F = NULL; }
    }
    if (F) {
        s->zcp = malloc(sizeof(int) * (n + 1));
        s->zri = malloc(sizeof(int) * (tot + n + 1));
        s->zcp[0] = 0;
        for (int c = 0; c < n; c++) {
            int *col = s->zri + s->zcp[c];
            col[0] = c;
            memcpy(col + 1, nbr[c], sizeof(int) * deg[c]);
            qsort(col, deg[c] + 1, sizeof(int), cmp_int_asc);
            s->zcp[c + 1] = s->zcp[c] + deg[c] + 1;
        }
        s->zt = malloc(sizeof(double) * 2 * n);
        s->zdp = malloc(sizeof(double) * (s->zcp[n] + 1));
        /* the constraints' pattern by columns (duplicates dropped) */
        int *uc = calloc(n + 1, sizeof(int));
        for (int q = 0; q < s->upat.nf; q++) uc[s->upat.fc[q] + 1]++;
        for (int c = 0; c < n; c++) uc[c + 1] += uc[c];
        int *ur = malloc(sizeof(int) * (s->upat.nf + 1)), *pos = malloc(sizeof(int) * n);
        memcpy(pos, uc, sizeof(int) * n);
        for (int q = 0; q < s->upat.nf; q++) ur[pos[s->upat.fc[q]]++] = s->upat.fr[q];
        for (int i = 0; i < n; i++) mk[i] = -1;
        int nu = 0;
        for (int c = 0; c < n; c++) {
            const int b = uc[c], e = uc[c + 1];
            uc[c] = nu;
            for (int q = b; q < e; q++) { const int r = ur[q]; if (mk[r] != c) { mk[r] = c; ur[nu++] = r; } }
        }
        uc[n] = nu;
        free(pos);
        s->ucp = uc; s->uri = ur;
        s->zf = F; s->zon = 1;
        zsp_pattern_setup(B, s);
        if (par->verbose > 0)
            printf("   block %d (n = %d): sparse factor of Z, %zu nonzeros (%.1f%% of n^2/2); X dense\n",
                   kblk + 1, n, schol_nnz(F), 200.0 * schol_nnz(F) / ((double)n * n));
    }
    for (int i = 0; i < n; i++) free(nbr[i]);
    free(nbr); free(deg); free(cnt); free(mk);
    return s->zon;
}


/* pattern = combined pattern (constraints + initial dual residual) + pattern of C: the
 * self-dual embedding's middle matrices theta R0 - A'w + a C are supported there */
static void setup_hsd_pattern(const Block *B, BS *s, const Params *par, Pat *out) {
    const int n = B->n;
    const size_t len = (size_t)n * n;
    unsigned char *mark = calloc(len, 1);
    for (int k = 0; k < s->cpat.nf; k++) mark[s->cpat.fr[k] + (size_t)s->cpat.fc[k] * n] = 1;
    for (int q = 0; q < B->C.ef; q++) mark[B->C.fr[q] + (size_t)B->C.fc[q] * n] = 1;
    size_t nf = 0;
    for (size_t i = 0; i < len; i++) nf += mark[i];
    out->nf = (int)nf;
    out->fr = malloc(sizeof(int) * (nf + 1));
    out->fc = malloc(sizeof(int) * (nf + 1));
    char *rm = calloc(n, 1);
    int c = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (mark[i + (size_t)j * n]) { out->fr[c] = i; out->fc[c] = j; c++; rm[i] = 1; }
    out->nr = 0;
    for (int i = 0; i < n; i++) out->nr += rm[i];
    out->rows = malloc(sizeof(int) * (out->nr + 1));
    for (int i = 0, q = 0; i < n; i++) if (rm[i]) out->rows[q++] = i;
    /* dense constraints are applied through a dense n^2 x nd product anyway */
    out->route = B->nd ? 2 : pat_route(n, (double)nf, out->nr, par->c_sparse, par->c_blas);
    out->owned = 1;
    if (out->route == 0) rowbuf_reserve(s, out->nr);
    free(rm); free(mark);
}


/* lower-triangle storage: row = larger index, column = smaller index */
#define MIDX(a, b, m) ((a) > (b) ? (size_t)(a) + (size_t)(b) * (m) : (size_t)(b) + (size_t)(a) * (m))
/* ------------------------------------------------------------------------ */
/* Envelope (profile) storage for a sparse Schur complement.
 * Pattern: union of constraint cliques per SDP block and per LP variable.
 * Ordering: reverse Cuthill-McKee. Row i (permuted) stores columns fst[i]..i. */
typedef struct {
    int m;
    int *perm, *iperm;     /* perm[new] = old, iperm[old] = new */
    int *fst;
    long *rptr;
    long nnz;
    double *val, *L;
} Envelope;

static const Envelope *g_env = NULL;   /* active during Schur assembly */
static SChol *g_schol = NULL;          /* ... or the sparse factor */
/* direct scatter into the sparse factor's storage: per block, the panel offsets of its
 * (constraint, constraint) pairs (local indices, row-major ncon x ncon), and the map
 * global constraint -> local index of the block being assembled (-1 elsewhere).
 * Replaces one hash lookup per Schur entry (the hot spot of the assembly on problems
 * with many small blocks: 719 cliques on the 793-bus AC-OPF). */
static CliqueTab **g_bpos = NULL;   /* 18 October 2026: offset tables (schol_clique_table), were nl x nl squares */
static size_t **g_bsoff = NULL;     /* per block: lower-triangle offsets sorted, and their buffer index */
static int **g_bsidx = NULL, *g_bsn = NULL;
static double *g_lbuf_all = NULL;
static const Block *g_blk0 = NULL;
static int g_nbk = 0;
static __thread int g_nl = 0;            /* 4.24: per-thread block state (parallel assembly) */
static __thread int *g_loc = NULL;
static __thread const CliqueTab *g_pos = NULL;
static double *g_pm = NULL;
/* local accumulation buffer of the block being assembled (lower triangle, hi * nl + lo),
 * flushed into the factor storage in increasing offset order: the scattered writes into
 * the 5 MB panel storage were 2/3 of the assembly time on the TSSOS models */
static __thread double *g_lbuf = NULL;
static inline void sc_add(size_t r, size_t c, double v) {
    const int lr = g_loc[r], lc = g_loc[c];
    if (lr >= 0 && lc >= 0) {
        if (g_lbuf) { const int hi = lr > lc ? lr : lc, lo = lr > lc ? lc : lr; g_lbuf[(size_t)lo * g_nl + hi] += v; }
        else g_pm[clique_off(g_pos, lr, lc)] += v;
    }
    else schol_add(g_schol, r, c, v);
}

/* 17 October 2026: the scatter state above is per thread (the parallel assembly gives each thread its own block); a
 * block assembled by a team of threads hands the master's state to the team, or the workers' writes went through
 * schol_add, a search of the factor's column and a write to the shared factor per entry (broyden25: three of the
 * four threads of its 19 unbuffered blocks). The offsets are the same, so are the sums. */
typedef struct { const CliqueTab *pos; int *loc; int nl; } AsmTL;
static inline AsmTL asm_tl_set(AsmTL a) { const AsmTL o = { g_pos, g_loc, g_nl }; g_pos = a.pos; g_loc = a.loc; g_nl = a.nl; return o; }
#define ASM_TL_MASTER const AsmTL asm_tl0_ = { g_pos, g_loc, g_nl }
#define ASM_TL_TEAM   const AsmTL asm_tlw_ = asm_tl_set(asm_tl0_)
#define ASM_TL_DONE   asm_tl_set(asm_tlw_)
static inline void env_add(const Envelope *E, size_t r, size_t c, double v) {
    int pi = E->iperm[r], pj = E->iperm[c];
    if (pi < pj) { int t = pi; pi = pj; pj = t; }
    E->val[E->rptr[pi] + (pj - E->fst[pi])] += v;
}
/* the class route's core (schur_cls): its writes go to the K x K matrix handed down, set
 * for the whole team outside parallel regions and for the thread inside one */
static int g_direct_g = 0; static __thread int g_direct_t = 0;
/* ordered write (r >= c) and unordered write */
#define MADDL(r, c, v) do { size_t r_ = (r), c_ = (c); double v_ = (v); \
    if (g_direct_g | g_direct_t) M[r_ + c_ * (size_t)m] += v_; \
    else if (g_pos) sc_add(c_, r_, v_);   /* (the offsets are symmetric; the callers run r for a fixed c: a row of g_pos) */ \
    else if (g_schol) schol_add(g_schol, r_, c_, v_); \
    else if (g_env) env_add(g_env, r_, c_, v_); else M[r_ + c_ * (size_t)m] += v_; } while (0)
#define MADD(r, c, v) do { size_t r_ = (r), c_ = (c); double v_ = (v); \
    if (g_direct_g | g_direct_t) M[MIDX(r_, c_, m)] += v_; \
    else if (g_pos) sc_add(r_, c_, v_); \
    else if (g_schol) schol_add(g_schol, r_, c_, v_); \
    else if (g_env) env_add(g_env, r_, c_, v_); else M[MIDX(r_, c_, m)] += v_; } while (0)


/* 4.24: C (uplo triangle, R x R, ld ldc) = V' V (V: n x R, ld ldv) without the locked
 * OpenBLAS dsyrk path: dgemm into a thread-local buffer, then the triangle */
static void xsyrk_t(const char *uplo, int R, int n, const double *V, int ldv, double *C, int ldc) {
    if (!g_lockfree || R > 256) { BL(dsyrk_)(uplo, "T", &R, &n, &DONE, V, &ldv, &DZERO, C, &ldc); return; }
    static __thread double *buf = NULL; static __thread size_t cap = 0;
    const size_t need = (size_t)R * R;
    if (need > cap) { free(buf); cap = need; buf = malloc(sizeof(double) * cap); }
    pdgemm("T", "N", &R, &R, &n, &DONE, V, &ldv, V, &ldv, &DZERO, buf, &R);
    const int lo = uplo[0] == 'L';
    for (int c = 0; c < R; c++)
        if (lo) for (int r = c; r < R; r++) C[r + (size_t)c * ldc] = buf[r + (size_t)c * R];
        else for (int r = 0; r <= c; r++) C[r + (size_t)c * ldc] = buf[r + (size_t)c * R];
}
/* Buffers of the low-rank route. Allocated at setup: the Schur routines may be handed a
 * shallow copy of the block state (NT view), so lazily created buffers would be lost. */
static void lowrank_alloc(const Block *B, BS *s) {
    const int n = B->n, R = B->lrR, nc = B->ncon;
    if (R == 0 || s->lrV) return;
    s->lrV = amalloc(sizeof(double) * (size_t)n * R);
    s->lrPQ = amalloc(sizeof(double) * (size_t)R * R);
    s->lrPd = amalloc(sizeof(double) * R);
    s->lrQd = amalloc(sizeof(double) * R);
    s->lr_owner = malloc(sizeof(int) * R);
    if (16.0 * R * R <= s->lr_mem) s->lrQ = amalloc(sizeof(double) * (size_t)R * R);
    for (int t = 0; t < nc; t++)
        for (int k = B->lr_off[t]; k < B->lr_off[t + 1]; k++) s->lr_owner[k] = t;
}

/* Low-rank Schur route: A_t = W_t S_t W_t',
 *   M_tu = sum_{k in t, l in u} s_k s_l (w_k' X w_l)(w_k' Zi w_l).
 * P = (LX' W)'(LX' W) and Q = (LZ^{-1} W)'(LZ^{-1} W) by dsyrk, stored in one
 * R x R buffer (P upper, Q lower, diagonals separately).                      */
static void schur_lowrank(const Block *B, BS *s, int m, double *M, int identity) {
    const int n = B->n, R = B->lrR, nc = B->ncon;
    if (R == 0) return;
    const size_t nR = (size_t)n * R;
    if (!s->lrV) lowrank_alloc(B, s);
    double *V = s->lrV, *PQ = s->lrPQ;
    double tl0 = wtime();
    const int two = s->lrQ != NULL;
    double *QB = two ? s->lrQ : PQ;
    const char *qtri = two ? "L" : "U";
    if (s->ntG && !identity) {
        /* NT: X and Zi are both replaced by W = G G', so P = Q = (G'W)'(G'W) */
        pdgemm("T", "N", &n, &R, &n, &DONE, s->ntG, &n, B->lr_W, &n, &DZERO, V, &n);
        xsyrk_t("L", R, n, V, n, PQ, R);
        for (int k = 0; k < R; k++) s->lrPd[k] = s->lrQd[k] = PQ[k + (size_t)k * R];
        if (two) memcpy(QB, PQ, sizeof(double) * (size_t)R * R);
        else for (int c = 0; c < R; c++) for (int r = c + 1; r < R; r++) PQ[c + (size_t)r * R] = PQ[r + (size_t)c * R];
    } else {
    memcpy(V, B->lr_W, sizeof(double) * nR);
    if (!identity) xtrmm("L", "L", "T", "N", &n, &R, &DONE, s->LX, &n, V, &n);   /* LX' W */
    xsyrk_t("L", R, n, V, n, PQ, R);                                                      /* P lower */
    for (int k = 0; k < R; k++) s->lrPd[k] = PQ[k + (size_t)k * R];
    memcpy(V, B->lr_W, sizeof(double) * nR);
    if (!identity) xtrsm("L", "L", "N", "N", &n, &R, &DONE, s->LZ, &n, V, &n);   /* LZ^{-1} W */
    xsyrk_t(qtri, R, n, V, n, QB, R);
    for (int k = 0; k < R; k++) s->lrQd[k] = QB[k + (size_t)k * R];
    }
    double tl1 = wtime();
    const double *sig = B->lr_sig;
    const int *own = s->lr_owner;
    ASM_TL_MASTER;
    #pragma omp parallel if (nc > 32 && nthreads() > 1 && !in_par())
    {
        ASM_TL_TEAM;
        double *qtmp = two ? NULL : malloc(sizeof(double) * R);
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < nc; t++) {                       /* writes go to column con[t] */
            const size_t ct = B->con[t];
            for (int k = B->lr_off[t]; k < B->lr_off[t + 1]; k++) {
                const double sk = sig[k];
                const double *pcol = PQ + (size_t)k * R;       /* P[l,k], l > k */
                const double *qcol;
                if (two) qcol = QB + (size_t)k * R;            /* Q[l,k], l > k */
                else {
                    for (int ll = k + 1; ll < R; ll++) qtmp[ll] = PQ[k + (size_t)ll * R];
                    qcol = qtmp;
                }
                int l = k + 1;
                while (l < R) {
                    const int u = own[l];
                    const int lend = B->lr_off[u + 1];
                    double part = 0;
                    for (int ll = l; ll < lend; ll++) part += sig[ll] * pcol[ll] * qcol[ll];
                    if (u == t) part *= 2.0;                    /* (k,l) and (l,k) */
                    MADDL(B->con[u], ct, sk * part);
                    l = lend;
                }
                MADDL(ct, ct, sk * sk * s->lrPd[k] * s->lrQd[k]);
            }
        }
        free(qtmp);
        ASM_TL_DONE;
    }
    if (ENV_ON("BRISK_DEBUG")) printf("     [lowrank: blas %.3fs, aggregate %.3fs]\n", tl1 - tl0, wtime() - tl1);
}

/* ------------------------------------------------------------------------ */
/* Schur complement  M_ij = sum_blocks tr(A_i X A_j Z^{-1})  (upper triangle) */


/* Dictionary route: P = D'XD, Q = D'Zi D with D = [e_{dunit} | V]. */
static void dict_form(const Block *B, int n, const double *M, double *out, double *W) {
    const int nu = B->dnu, nv = B->dnv, q = nu + nv;
    const int *U = B->dunit;
    for (int b = 0; b < nu; b++) {
        const double *mc = M + (size_t)U[b] * n;
        double *oc = out + (size_t)b * q;
        for (int a = 0; a < nu; a++) oc[a] = mc[U[a]];
    }
    if (nv == 0) return;
    xsymm("L", "L", &n, &nv, &DONE, M, &n, B->dV, &n, &DZERO, W, &n);      /* M V */
    for (int c = 0; c < nv; c++) {
        const double *wc = W + (size_t)c * n;
        double *oc = out + (size_t)(nu + c) * q;
        for (int a = 0; a < nu; a++) {
            double v = wc[U[a]];
            oc[a] = v;                                   /* (U, V) block */
            out[(nu + c) + (size_t)a * q] = v;           /* (V, U) block */
        }
    }
    pdgemm("T", "N", &nv, &nv, &n, &DONE, B->dV, &n, W, &n, &DZERO,
               out + nu + (size_t)nu * q, &q);                                     /* V' M V */
    for (int c = 0; c < nv; c++)
        for (int r = 0; r < c; r++) {
            size_t i1 = (nu + r) + (size_t)(nu + c) * q, i2 = (nu + c) + (size_t)(nu + r) * q;
            double v = 0.5 * (out[i1] + out[i2]);
            out[i1] = out[i2] = v;
        }
}

static void schur_sdp_id(const Block *B, BS *s, int m, double *M, int identity);
static int g_asmdbg = 0; static double g_asmf[4], g_asmr[6];
static double g_split[8]; static int g_split_on = -1;
static void split_print(void) { fprintf(stderr, "   [assembly split: sparse-sparse %.2fs; row-product: setup %.2fs, dgemm %.2fs (%.3g flops, %.0f calls, %.1f GF/s), gather %.2fs (%.3g entries, %.2f ns each)]\n", g_split[0], g_split[1], g_split[2], g_split[4], g_split[5], g_split[4] / (g_split[2] + 1e-30) / 1e9, g_split[3], g_split[6], 1e9 * g_split[3] / (g_split[6] + 1e-30)); }
static __thread const Block *g_cur_blk = NULL; static int g_nofastl = 0;   /* 4.23: the block whose local buffer g_lbuf is active */
static void schur_sdp(const Block *B, BS *s, int m, double *M) {
    if (g_bpos && g_schol) {
        const long k = B - g_blk0;
        if (k >= 0 && k < g_nbk && g_bpos[k]) {
            g_pos = g_bpos[k]; g_nl = B->ncon;
            for (int t = 0; t < B->ncon; t++) g_loc[B->con[t]] = t;
            const int use_buf = g_bsoff && g_bsoff[k] && g_lbuf_all;
            if (use_buf) { g_lbuf = g_lbuf_all; g_cur_blk = B; }
            const double tf0 = g_asmdbg ? wtime() : 0;
            schur_sdp_id(B, s, m, M, 0);
            const double tf1 = g_asmdbg ? wtime() : 0;
            if (use_buf) {
                const size_t *of = g_bsoff[k];
                const int *ix = g_bsidx[k], nq = g_bsn[k];
                double *lb = g_lbuf;
                for (int q = 0; q < nq; q++) { g_pm[of[q]] += lb[ix[q]]; lb[ix[q]] = 0; }
                g_lbuf = NULL; g_cur_blk = NULL;
                if (g_asmdbg) { g_asmf[0] += wtime() - tf1; g_asmf[2] += nq; }
            }
            if (g_asmdbg) g_asmf[1] += tf1 - tf0;
            for (int t = 0; t < B->ncon; t++) g_loc[B->con[t]] = -1;
            g_pos = NULL;
            return;
        }
    }
    schur_sdp_id(B, s, m, M, 0);
}
/* 4.24: assembly of a buffered block into its own value list (panel-offset order), no flush */
static void schur_sdp_vals(const Block *B, BS *s, int m, double *M, long k, int *loc, double *lb, double *vals) {
    g_pos = g_bpos[k]; g_nl = B->ncon; g_loc = loc;
    for (int t = 0; t < B->ncon; t++) loc[B->con[t]] = t;
    g_lbuf = lb; g_cur_blk = B;
    schur_sdp_id(B, s, m, M, 0);
    const int *ix = g_bsidx[k], nq = g_bsn[k];
    for (int q = 0; q < nq; q++) { vals[q] = lb[ix[q]]; lb[ix[q]] = 0; }
    g_lbuf = NULL; g_cur_blk = NULL;
    for (int t = 0; t < B->ncon; t++) loc[B->con[t]] = -1;
    g_pos = NULL;
}
/* The entry-class route (dictroute.c): the K x K Schur matrix M_K of the class indicators by
 * the ordinary routes, then M_st = (G M_K G')_st for s >= t: column t of M_K G' (K values, from
 * the classes of constraint t), then a sparse dot of each row s >= t of G with it. Every entry
 * of M is formed by one thread in a fixed order (independent of the thread count). */
static void lower_to_full(int n, double *A);
static double g_clst[3]; static int g_clsdbg = -1;
/* the K x K cores, one per thread of the team that assembles (freed at the end of a solve) */
#define CLS_NT 256
static double *g_mkbuf[CLS_NT]; static size_t g_mkcap[CLS_NT];
static void cls_release(void) { for (int t = 0; t < CLS_NT; t++) { free(g_mkbuf[t]); g_mkbuf[t] = NULL; g_mkcap[t] = 0; } }
static void cls_print(void) { fprintf(stderr, "   [class route: core %.2fs, symmetrize %.2fs, map %.2fs]\n", g_clst[0], g_clst[1], g_clst[2]); }
static void schur_cls(const Block *B, BS *s, int m, double *M) {
    if (g_clsdbg < 0) { g_clsdbg = getenv("BRISK_CLSDBG") != NULL; if (g_clsdbg) atexit(cls_print); }
    const double tc0 = g_clsdbg ? wtime() : 0;
    const Block *V = B->cb;
    const int K = V->ncon, nl = B->ncon;
    const int *gp = B->cg_ptr, *gk = B->cg_k, *con = B->con;
    const double *gv = B->cg_v;
    const int ip = in_par();
#ifdef _OPENMP
    const int tid = ip ? omp_get_thread_num() : 0;
#else
    const int tid = 0;
#endif
    const size_t need = (size_t)K * K;
    double *MK;
    if (tid < CLS_NT) {
        if (need > g_mkcap[tid]) { free(g_mkbuf[tid]); g_mkbuf[tid] = amalloc(sizeof(double) * need); g_mkcap[tid] = need; }
        MK = g_mkbuf[tid];
    } else MK = amalloc(sizeof(double) * need);
    if (ip) memset(MK, 0, sizeof(double) * need); else pzero(need, MK);
    if (ip) g_direct_t = 1; else g_direct_g = 1;
    BS v = *s;
    schur_sdp_id(V, &v, K, MK, 0);
    if (ip) g_direct_t = 0; else g_direct_g = 0;
    const double tc1 = g_clsdbg ? wtime() : 0;
    lower_to_full(K, MK);
    const double tc2 = g_clsdbg ? wtime() : 0;
    double *const lbuf = (g_lbuf && B == g_cur_blk && !g_nofastl) ? g_lbuf : NULL;
    const size_t lnl = (size_t)g_nl;
    /* panels of W constraints t: Tc = M_K G(panel)' (K x W, by columns: axpys along K), stored by
     * rows (Tr: W values of a class side by side), then for every row u >= t0 the W sums
     * of G(u,:) Tr at once; written column by column */
    enum { W = 16 };
    const int np = (nl + W - 1) / W;
    const int par = nthreads() > 1 && !ip && np > 1;
    ASM_TL_MASTER;
    #pragma omp parallel if (par)
    {
        ASM_TL_TEAM;
        double *Tc = amalloc(sizeof(double) * ((size_t)K * W + 1));
        double *Tr = amalloc(sizeof(double) * ((size_t)K * W + 1));
        double *Mp = amalloc(sizeof(double) * ((size_t)nl * W + 1));
        #pragma omp for schedule(dynamic, 1)
        for (int p = 0; p < np; p++) {
            const int t0 = p * W, w = t0 + W <= nl ? W : nl - t0;
            for (int j = 0; j < w; j++) {
                double *tc = Tc + (size_t)j * K;
                const int t = t0 + j;
                if (gp[t] == gp[t + 1]) { memset(tc, 0, sizeof(double) * K); continue; }
                {   const double *c = MK + (size_t)gk[gp[t]] * K; const double g = gv[gp[t]];
                    for (int i = 0; i < K; i++) tc[i] = g * c[i]; }
                for (int q = gp[t] + 1; q < gp[t + 1]; q++) {
                    const double *c = MK + (size_t)gk[q] * K; const double g = gv[q];
                    for (int i = 0; i < K; i++) tc[i] += g * c[i];
                }
            }
            for (int i = 0; i < K; i++) {
                double *tr = Tr + (size_t)i * W;
                for (int j = 0; j < w; j++) tr[j] = Tc[i + (size_t)j * K];
                for (int j = w; j < W; j++) tr[j] = 0;
            }
            for (int u = t0; u < nl; u++) {
                double acc[W];
                for (int j = 0; j < W; j++) acc[j] = 0;
                for (int q = gp[u]; q < gp[u + 1]; q++) {
                    const double g = gv[q], *tr = Tr + (size_t)gk[q] * W;
                    for (int j = 0; j < W; j++) acc[j] += g * tr[j];
                }
                double *mp = Mp + (size_t)(u - t0) * W;
                for (int j = 0; j < W; j++) mp[j] = acc[j];
            }
            for (int j = 0; j < w; j++) {
                const int t = t0 + j, ct = con[t];
                for (int u = t; u < nl; u++) {
                    const double sum = Mp[(size_t)(u - t0) * W + j];
                    if (lbuf) lbuf[(size_t)t * lnl + u] += sum; else MADDL(con[u], ct, sum);
                }
            }
        }
        free(Tc); free(Tr); free(Mp);
        ASM_TL_DONE;
    }
    if (tid >= CLS_NT) free(MK);
    if (g_clsdbg && !ip) { g_clst[0] += tc1 - tc0; g_clst[1] += tc2 - tc1; g_clst[2] += wtime() - tc2; }
}
int g_asmsp = 0;   /* 5.12: the row products of this assembly in single precision (set by the embedding) */
#define GBMAX 8
static int g_gbatch = -1;   /* 5.12: constraints a batch of the symmetric gather (BRISK_GBATCH, 1..8) */
/* 16 October 2026 (ISSUES 96a): the batched route of a block whose constraints are all monomial. For a chunk of
 * constraints a0 .. a1: S_a = A_a Zi (row i of A_a holds the entry (i, k, v): row i of S_a is v times row k of
 * Zi, which is column k, Zi being symmetric), G = X [S_a0 | ... ] in one product of n x (cn) x n, then
 * M_ab = <A_b, G_a> for b >= a by a gather of the n entries of A_b (M is symmetric: <A_b, X A_a Zi> =
 * <A_a, X A_b Zi>). qst_6q_25 (n = 128, 1,017 constraints): the row products took 0.3 s a CPU per assembly. */
static void schur_mono(const Block *B, BS *s, int m, double *M) {
    const int n = B->n, ns = B->ns;
    const size_t n2 = (size_t)n * n;
    const double *X = s->X, *Zi = s->Zi;
    const int *foff = B->foff, *ffr = B->ffr, *ffc = B->ffc, *ffp = B->ffp;
    const double *ffv = B->ffv;
    /* eight constraints a chunk: the interleaved products GT are then n^2 x 8 doubles (4 MB at n = 256, in the
     * last-level cache), and the gather of an entry reads one cache line of eight values, all used; a chunk of 61
     * (32 MB) made every entry a fetch of eight lines from memory: qst_7q_15 4 s an assembly against 2.3 */
    int c = 8; if (c > ns) c = ns;
    double *S = amalloc(sizeof(double) * n2 * c), *G = amalloc(sizeof(double) * n2 * c), *GT = amalloc(sizeof(double) * n2 * c);
    const int par = nthreads() > 1 && !in_par();
    static int mdbg = -1; if (mdbg < 0) mdbg = getenv("BRISK_MONODBG") != NULL;
    double tm[4] = { 0, 0, 0, 0 }, t0 = mdbg ? wtime() : 0;
    for (int a0 = 0; a0 < ns; a0 += c) {
        const int a1 = a0 + c < ns ? a0 + c : ns, cc = a1 - a0, cn = cc * n;
        #pragma omp parallel for schedule(static) if (par && cc > 1)
        for (int a = a0; a < a1; a++) {
            double *Sa = S + (size_t)(a - a0) * n2;
            memset(Sa, 0, sizeof(double) * n2);
            const int e0 = foff[a], e1 = foff[a + 1];
            for (int j = 0; j < n; j++) {
                double *sj = Sa + (size_t)j * n; const double *zj = Zi + (size_t)j * n;
                for (int e = e0; e < e1; e++) sj[ffr[e]] = ffv[e] * zj[ffc[e]];
            }
        }
        if (mdbg) { tm[0] += wtime() - t0; t0 = wtime(); }
        pdgemm("N", "N", &n, &cn, &n, &DONE, X, &n, S, &n, &DZERO, G, &n);
        if (mdbg) { tm[1] += wtime() - t0; t0 = wtime(); }
        /* the chunk's products interleaved, the chunk index fastest (GT[a + p c] = G_a[p]): the gather for a
         * constraint b then reads, per entry, cc contiguous values and adds them to cc accumulators (SIMD), where
         * the layout G_a after G_a read one cache line per entry and chunk member (qst_6q_25: 1.98 s of
         * assembly for 13 iterations, the same as the row products, the gather being the cost, not the products) */
        #pragma omp parallel for schedule(static) if (par && cc > 1)
        for (int p0 = 0; p0 < (int)n2; p0 += 256) {
            const int p1 = p0 + 256 < (int)n2 ? p0 + 256 : (int)n2;
            for (int a = 0; a < cc; a++) { const double *Ga = G + (size_t)a * n2; double *gt = GT + a; for (int p = p0; p < p1; p++) gt[(size_t)p * cc] = Ga[p]; }
        }
        if (mdbg) { tm[2] += wtime() - t0; t0 = wtime(); }
        if (par) blas_serial_begin();
        #pragma omp parallel for schedule(dynamic, 16) if (par && ns - a0 > 32)
        for (int b = a0; b < ns; b++) {
            double acc[8];
            for (int q = 0; q < cc; q++) acc[q] = 0;
            for (int k = foff[b]; k < foff[b + 1]; k++) {
                const double v = ffv[k];
                const double *g = GT + (size_t)(ffp ? ffp[k] : ffr[k] + ffc[k] * n) * cc;
                for (int q = 0; q < cc; q++) acc[q] += v * g[q];
            }
            const int cb = B->con[B->slist[b]];
            for (int q = 0; q < cc; q++) if (b >= a0 + q) MADD(B->con[B->slist[a0 + q]], cb, acc[q]);
        }
        if (par) blas_serial_end();
        if (mdbg) { tm[3] += wtime() - t0; t0 = wtime(); }
    }
    if (mdbg) printf("   [mono route n %d ns %d chunk %d: rows %.3f, product %.3f, interleave %.3f, gather %.3f s]\n", n, ns, c, tm[0], tm[1], tm[2], tm[3]);
    free(S); free(G); free(GT);
}
/* 5.18 (ISSUES 96a): the Weyl route. Every constraint is A_b[c ^ x_b, c] = k_b (-1)^(z_b . c) (problem.c), so with
 * G_a = X A_a Zi (the products of the monomial route, one BLAS-3 call per chunk)
 *     M_ab = <A_b, G_a> = k_b sum_c (-1)^(z_b . c) G_a[c ^ x_b, c] = k_b H_a(x_b)[z_b],
 * H_a(x) the Walsh-Hadamard transform of the x-th XOR diagonal of G_a: for each a, n transforms of length n
 * (n^2 log n) give the pairs with every b, where the gather read n entries a pair at random. */
static void fwht(double *v, int n) {
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; j++) { const double a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
}
/* 23 October 2026: the Weyl route on a Hermitian block in its real form (weylpd.c): Y_a = Wc D_a Xc in complex
 * arithmetic of order d = n/2 (three real products per chunk), the pairs M_ab = 2 wa wb Re[i^(ta+tb) tr(D_b Y_a)] from
 * the Walsh-Hadamard transforms of the XOR diagonals of Y_a; the same entries as the real form's route below. */
static void schur_weyl_pd(const Block *B, BS *s, int m, double *M) {
    const int n = B->n, d = n / 2, ns = B->ns, k = B->wpd_k;
    int q = 0; while ((1 << q) < d) q++;
    const size_t D2 = (size_t)d * d;
    const int *ap = B->wpd_ap; const unsigned char *ti = B->wpd_t;
    static int wdbg = -1; if (wdbg < 0) wdbg = getenv("BRISK_MONODBG") != NULL;
    double tm[3] = { 0, 0, 0 }, t0 = wdbg ? wtime() : 0;
    double *Xr = amalloc(sizeof(double) * D2), *Xi = amalloc(sizeof(double) * D2);
    double *Wr = amalloc(sizeof(double) * D2), *Wi = amalloc(sizeof(double) * D2), *Ws = amalloc(sizeof(double) * D2);
    weylpd_split(s->X, n, k, Xr, Xi);
    weylpd_split(s->Zi, n, k, Wr, Wi);
    for (size_t i = 0; i < D2; i++) Ws[i] = Wr[i] + Wi[i];
    int c = (int)fmin(16.0, fmax(1.0, 8e6 / (6.0 * (double)D2)));   /* (six chunk arrays within 8 million entries: 64 MB) */
    if (c > ns) c = ns;
    double *Br = amalloc(sizeof(double) * D2 * c), *Bi = amalloc(sizeof(double) * D2 * c), *Bs = amalloc(sizeof(double) * D2 * c);
    double *P1 = amalloc(sizeof(double) * D2 * c), *P2 = amalloc(sizeof(double) * D2 * c), *P3 = amalloc(sizeof(double) * D2 * c);
    const int par = nthreads() > 1 && !in_par();
    for (int a0 = 0; a0 < ns; a0 += c) {
        const int a1 = a0 + c < ns ? a0 + c : ns, cc = a1 - a0, cn = cc * d;
        /* B_a = D_a Xc: (D_a Xc)[r, col] = (-1)^(az . (r ^ ax)) Xc[r ^ ax, col] */
        #pragma omp parallel for schedule(static) if (par && cc > 1)
        for (int a = a0; a < a1; a++) {
            const int ax = ap[a] >> q, az = ap[a] & (d - 1);
            double *br = Br + (size_t)(a - a0) * D2, *bi = Bi + (size_t)(a - a0) * D2, *bs = Bs + (size_t)(a - a0) * D2;
            for (int col = 0; col < d; col++)
                for (int r = 0; r < d; r++) {
                    const int kk = r ^ ax; const size_t o = r + (size_t)col * d, src = kk + (size_t)col * d;
                    const double sg = __builtin_parity(az & kk) ? -1.0 : 1.0;
                    br[o] = sg * Xr[src]; bi[o] = sg * Xi[src]; bs[o] = br[o] + bi[o];
                }
        }
        if (wdbg) { tm[0] += wtime() - t0; t0 = wtime(); }
        pdgemm("N", "N", &d, &cn, &d, &DONE, Wr, &d, Br, &d, &DZERO, P1, &d);
        pdgemm("N", "N", &d, &cn, &d, &DONE, Wi, &d, Bi, &d, &DZERO, P2, &d);
        pdgemm("N", "N", &d, &cn, &d, &DONE, Ws, &d, Bs, &d, &DZERO, P3, &d);
        if (wdbg) { tm[1] += wtime() - t0; t0 = wtime(); }
        if (par) blas_serial_begin();
        ASM_TL_MASTER;
        #pragma omp parallel if (par && cc > 1)
        {
            ASM_TL_TEAM;
            double *Hr = malloc(sizeof(double) * D2), *Hi = malloc(sizeof(double) * D2);
            char *done = malloc(d);
            #pragma omp for schedule(dynamic, 1)
            for (int a = a0; a < a1; a++) {
                const size_t ob = (size_t)(a - a0) * D2;
                const double *p1 = P1 + ob, *p2 = P2 + ob, *p3 = P3 + ob;
                memset(done, 0, d);
                const int ca = B->con[B->slist[a]];
                for (int b = a; b < ns; b++) {
                    const int bx = ap[b] >> q, bz = ap[b] & (d - 1);
                    double *hr = Hr + (size_t)bx * d, *hi = Hi + (size_t)bx * d;
                    if (!done[bx]) {
                        /* the bx-th XOR diagonal of Y_a = P1 - P2 + i (P3 - P1 - P2) */
                        for (int j = 0; j < d; j++) {
                            const size_t o = j + (size_t)(j ^ bx) * d;
                            hr[j] = p1[o] - p2[o]; hi[j] = p3[o] - p1[o] - p2[o];
                        }
                        fwht(hr, d); fwht(hi, d);
                        done[bx] = 1;
                    }
                    double v;
                    switch ((ti[a] + ti[b]) & 3) {      /* Re[i^k T] */
                        case 0: v = hr[bz]; break;
                        case 1: v = -hi[bz]; break;
                        case 2: v = -hr[bz]; break;
                        default: v = hi[bz]; break;
                    }
                    MADD(ca, B->con[B->slist[b]], 2.0 * B->wk[a] * B->wk[b] * v);
                }
            }
            free(Hr); free(Hi); free(done);
            ASM_TL_DONE;
        }
        if (par) blas_serial_end();
        if (wdbg) { tm[2] += wtime() - t0; t0 = wtime(); }
    }
    if (wdbg) printf("   [Weyl route in complex form, d %d ns %d: rows %.3f, products %.3f, transforms and pairs %.3f s]\n", d, ns, tm[0], tm[1], tm[2]);
    free(Xr); free(Xi); free(Wr); free(Wi); free(Ws); free(Br); free(Bi); free(Bs); free(P1); free(P2); free(P3);
}
static void schur_weyl(const Block *B, BS *s, int m, double *M) {
    if (B->wpd) { schur_weyl_pd(B, s, m, M); return; }
    const int n = B->n, ns = B->ns;
    const size_t n2 = (size_t)n * n;
    const double *X = s->X, *Zi = s->Zi;
    const int *foff = B->foff, *ffr = B->ffr, *ffc = B->ffc;
    const double *ffv = B->ffv;
    int c = 8; if (c > ns) c = ns;
    double *S = amalloc(sizeof(double) * n2 * c), *G = amalloc(sizeof(double) * n2 * c);
    /* the constraints by their translation x: only the diagonals that some b >= a needs are transformed */
    int *xcnt = calloc(n + 1, sizeof(int)), *xlast = malloc(sizeof(int) * n);
    for (int x = 0; x < n; x++) xlast[x] = -1;
    for (int b = 0; b < ns; b++) { xcnt[B->wx[b]]++; xlast[B->wx[b]] = b; }
    const int par = nthreads() > 1 && !in_par();
    static int wdbg = -1; if (wdbg < 0) wdbg = getenv("BRISK_MONODBG") != NULL;
    double tm[3] = { 0, 0, 0 }, t0 = wdbg ? wtime() : 0;
    for (int a0 = 0; a0 < ns; a0 += c) {
        const int a1 = a0 + c < ns ? a0 + c : ns, cc = a1 - a0, cn = cc * n;
        #pragma omp parallel for schedule(static) if (par && cc > 1)
        for (int a = a0; a < a1; a++) {
            double *Sa = S + (size_t)(a - a0) * n2;
            memset(Sa, 0, sizeof(double) * n2);
            const int e0 = foff[a], e1 = foff[a + 1];
            for (int j = 0; j < n; j++) {
                double *sj = Sa + (size_t)j * n; const double *zj = Zi + (size_t)j * n;
                for (int e = e0; e < e1; e++) sj[ffr[e]] = ffv[e] * zj[ffc[e]];
            }
        }
        if (wdbg) { tm[0] += wtime() - t0; t0 = wtime(); }
        pdgemm("N", "N", &n, &cn, &n, &DONE, X, &n, S, &n, &DZERO, G, &n);
        if (wdbg) { tm[1] += wtime() - t0; t0 = wtime(); }
        if (par) blas_serial_begin();
        ASM_TL_MASTER;
        #pragma omp parallel if (par && cc > 1)
        {
            ASM_TL_TEAM;
            double *H = malloc(sizeof(double) * n2);      /* H[x n + z] */
            char *done = malloc(n);
            #pragma omp for schedule(dynamic, 1)
            for (int a = a0; a < a1; a++) {
                const double *Ga = G + (size_t)(a - a0) * n2;
                memset(done, 0, n);
                const int ca = B->con[B->slist[a]];
                for (int b = a; b < ns; b++) {
                    const int x = B->wx[b];
                    double *h = H + (size_t)x * n;
                    if (!done[x]) {
                        for (int cidx = 0; cidx < n; cidx++) h[cidx] = Ga[(cidx ^ x) + (size_t)cidx * n];
                        fwht(h, n);
                        done[x] = 1;
                    }
                    MADD(ca, B->con[B->slist[b]], B->wk[b] * h[B->wz[b]]);
                }
            }
            free(H); free(done);
            ASM_TL_DONE;
        }
        if (par) blas_serial_end();
        if (wdbg) { tm[2] += wtime() - t0; t0 = wtime(); }
    }
    if (wdbg) printf("   [Weyl route n %d ns %d: rows %.3f, product %.3f, transforms and pairs %.3f s]\n", n, ns, tm[0], tm[1], tm[2]);
    free(S); free(G); free(xcnt); free(xlast);
}
static void schur_sdp_id(const Block *B, BS *s, int m, double *M, int identity) {
    if (B->cls) { schur_cls(B, s, m, M); return; }
    if (B->allweyl && B->ns > 0) { schur_weyl(B, s, m, M); return; }
    if (B->allmono && B->ns > 0 && B->route[B->slist[0]] == 5) { schur_mono(B, s, m, M); return; }
    if (B->dict) {
        dict_form(B, B->n, s->X, s->vbs->X, s->dW);
        dict_form(B, B->n, s->Zi, s->vbs->Zi, s->dW);
        schur_sdp_id(B->vb, s->vbs, m, M, 0);
        return;
    }
    if (B->lowrank) { schur_lowrank(B, s, m, M, identity); return; }
    const int n = B->n, ns = B->ns, nd = B->nd;
    const size_t n2 = (size_t)n * n;
    const double *X = s->X, *Zi = s->Zi;
    const int *foff = B->foff, *ffr = B->ffr, *ffc = B->ffc, *ffp = B->ffp;
    const double *ffv = B->ffv;
    const int fend = foff[ns];

    /* phase 1: sparse-sparse pairs. Parallel over rows unless each row is a big
     * BLAS call (then BLAS itself should get the threads). */
    int *scon = malloc(sizeof(int) * (ns + 1));
    for (int b = 0; b < ns; b++) scon[b] = B->con[B->slist[b]];
    if (g_split_on < 0) { g_split_on = getenv("BRISK_ASMSPLIT") != NULL; if (g_split_on) atexit(split_print); }
    const int par1 = nthreads() > 1 && ns > 32 && !in_par();
    /* 4.23: pairs of this block's sparse constraints go straight to its local buffer: the
     * local indices are slist[b] >= slist[a] (no g_loc lookups per pair) */
    double *const lbuf = (g_lbuf && B == g_cur_blk && !g_nofastl) ? g_lbuf : NULL;
    const int *const sl = B->slist;
    const size_t lnl = (size_t)g_nl;
    /* 5.5: the live region (problem.c): X and Zi in the order of the indices' death */
    double *XP = NULL, *ZP = NULL;
    float *XPf = NULL, *ZPf = NULL;
    const int sp = g_asmsp && B->live;        /* 5.12: the row products in single precision (see g_asmsp) */
    /* 5.12: NT (the view X = Zi = W): X A Zi + Zi A X = 2 W A W, the symmetric gather's product
     * with inner dimension r instead of 2r, and the symmetric gather in double precision too */
    const int xzs = X == Zi && !ENV_ON("BRISK_NOXZS");
    if (g_gbatch < 0) { const char *e = getenv("BRISK_GBATCH"); g_gbatch = e ? atoi(e) : GBMAX; if (g_gbatch < 1) g_gbatch = 1; if (g_gbatch > GBMAX) g_gbatch = GBMAX; }
    if (B->live && sp) {
        XPf = amalloc(sizeof(float) * n2); ZPf = amalloc(sizeof(float) * n2);
        const int *pm = B->lv_perm;
        #pragma omp parallel for schedule(static) if (nthreads() > 1 && n >= 256 && !in_par())
        for (int j = 0; j < n; j++) {
            const double *xj = X + (size_t)pm[j] * n, *zj = Zi + (size_t)pm[j] * n;
            float *xo = XPf + (size_t)j * n, *zo = ZPf + (size_t)j * n;
            for (int i = 0; i < n; i++) { xo[i] = (float)xj[pm[i]]; zo[i] = (float)zj[pm[i]]; }
        }
    } else if (B->live) {
        XP = amalloc(sizeof(double) * n2); ZP = amalloc(sizeof(double) * n2);
        const int *pm = B->lv_perm;
        #pragma omp parallel for schedule(static) if (nthreads() > 1 && n >= 256 && !in_par())
        for (int j = 0; j < n; j++) {
            const double *xj = X + (size_t)pm[j] * n, *zj = Zi + (size_t)pm[j] * n;
            double *xo = XP + (size_t)j * n, *zo = ZP + (size_t)j * n;
            for (int i = 0; i < n; i++) { xo[i] = xj[pm[i]]; zo[i] = zj[pm[i]]; }
        }
    }
    if (par1) blas_serial_begin();
    ASM_TL_MASTER;
    #pragma omp parallel if (par1)
    {
        ASM_TL_TEAM;
        double *Xp = NULL, *Tm = NULL, *G = NULL, *tmp = NULL, *acc = NULL, *Xs = NULL, *Ts = NULL;
        float *Xsf = NULL, *Tsf = NULL, *Gf = NULL;
        int *rmap = NULL, *cptr = NULL;
        /* 5.12: the batched gather (BRISK_GBATCH, 8; 1 off): consecutive symmetric-path constraints
         * in groups of gbat, one pass over the entries for all of them (see below) */
        double *accB = NULL, *bXs = NULL, *bTs = NULL, *bG = NULL; float *bXsf = NULL, *bTsf = NULL, *bGf = NULL;
        size_t bXcap = 0, bTcap = 0;
        const int gbat = ((B->live && B->lu_cp && (B->symg || sp || xzs)) || (!B->live && B->s2k)) ? g_gbatch : 1;
        double *sG = NULL;      /* (the batch's H on the dsyr2k route: GBMAX of n x n) */
        const int nchunk = (ns + gbat - 1) / gbat;
        #pragma omp for schedule(dynamic, 1)
        for (int cb_ = 0; cb_ < nchunk; cb_++) {
          const int a_lo = cb_ * gbat, a_hi = a_lo + gbat < ns ? a_lo + gbat : ns;
          if (gbat > 1 && a_hi - a_lo > 1) {
            int okb = 1;
            for (int a = a_lo; a < a_hi; a++) if (B->route[B->slist[a]] != 1) { okb = 0; break; }
            if (okb && !B->live) {
                /* 18 October 2026, the dsyr2k route (s2k) by batches: the H of each constraint of the batch by its
                 * dsyr2k, then one sweep over the upper entries of the later constraints for all of them, each entry
                 * read once for the batch (the sums of b < a0 + j formed and dropped). Each accumulator takes its
                 * terms in the order of the route without batches, so the results are its. */
                const int nbt = a_hi - a_lo, a0 = a_lo;
                const double ts0 = g_split_on ? wtime() : 0;
                if (!Xs) { Xs = amalloc(sizeof(double) * 2 * n2); Ts = amalloc(sizeof(double) * 2 * n2); }
                if (!rmap) rmap = malloc(sizeof(int) * n);
                if (!accB) accB = calloc(((size_t)ns + 1) * GBMAX, sizeof(double));
                if (!sG) sG = amalloc(sizeof(double) * GBMAX * n2);
                double tpr = 0;
                for (int j = 0; j < nbt; j++) {
                    const SpSym *Aj = &B->A[B->slist[a0 + j]];
                    const int r = Aj->nr;
                    for (int k = 0; k < r; k++) { rmap[Aj->rows[k]] = k; memcpy(Xs + (size_t)k * n, X + (size_t)Aj->rows[k] * n, sizeof(double) * n); }
                    memset(Ts, 0, sizeof(double) * (size_t)r * n);
                    for (int e = 0; e < Aj->ef; e++) {
                        const double v = Aj->fv[e];
                        const double *zq = Zi + (size_t)Aj->fc[e] * n;
                        double *tk = Ts + (size_t)rmap[Aj->fr[e]] * n;
                        for (int i = 0; i < n; i++) tk[i] += v * zq[i];
                    }
                    const double tp0 = g_split_on ? wtime() : 0;
                    BL(dsyr2k_)("U", "N", &n, &r, &DONE, Xs, &n, Ts, &n, &DZERO, sG + (size_t)j * n2, &n);
                    if (g_split_on) { tpr += wtime() - tp0;
                        #pragma omp atomic
                        g_split[4] += 2.0 * n * (double)n * r; }
                }
                const double ts1 = g_split_on ? wtime() : 0;
                const int *cp = B->lu_cp, *ca = B->lu_ca, *cr = B->lu_cr; const double *cv = B->lu_cv;
                double ne = 0;
                for (int q = 0; q < n; q++) {
                    int lo = cp[q], hi = cp[q + 1];
                    while (lo < hi) { const int mid = (lo + hi) >> 1; if (ca[mid] < a0) lo = mid + 1; else hi = mid; }
                    const double *g0 = sG + (size_t)q * n;
                    if (nbt == GBMAX)
                        for (int e = lo; e < cp[q + 1]; e++) {
                            const double v = cv[e]; const double *gr = g0 + cr[e];
                            double *ab = accB + (size_t)(ca[e] - a0) * GBMAX;
                            for (int j = 0; j < GBMAX; j++) ab[j] += v * gr[(size_t)j * n2];
                        }
                    else
                        for (int e = lo; e < cp[q + 1]; e++) {
                            const double v = cv[e]; const double *gr = g0 + cr[e];
                            double *ab = accB + (size_t)(ca[e] - a0) * GBMAX;
                            for (int j = 0; j < nbt; j++) ab[j] += v * gr[(size_t)j * n2];
                        }
                    ne += cp[q + 1] - lo;
                }
                for (int j = 0; j < nbt; j++) {
                    const int aj = a0 + j;
                    const size_t cij = B->con[B->slist[aj]];
                    for (int b = aj; b < ns; b++) {
                        const double sum = accB[(size_t)(b - a0) * GBMAX + j];
                        if (lbuf) lbuf[(size_t)sl[aj] * lnl + sl[b]] += sum; else MADDL(scon[b], cij, sum);
                    }
                }
                memset(accB, 0, sizeof(double) * (size_t)(ns - a0) * GBMAX);
                if (g_split_on) { const double ts2 = wtime();
                    #pragma omp critical(split)
                    { g_split[1] += ts1 - ts0 - tpr; g_split[2] += tpr; g_split[3] += ts2 - ts1; g_split[5] += nbt; g_split[6] += ne * nbt; } }
                continue;
            }
            if (okb) {
                /* the batch a0 .. a0 + nbt - 1: the staircase of a0 (it covers the later ones), one
                 * product per constraint and panel, then one pass over each column's entries that
                 * adds v G_j[row] to the interleaved accumulators accB[(b - a0) GB + j]: the entry's
                 * constraint, row and value are read once for the batch and the additions are
                 * contiguous. The sums of b < a0 + j (below the diagonal) are formed and dropped. */
                const int nbt = a_hi - a_lo, a0 = a_lo;
                const double tr0 = g_asmdbg ? wtime() : 0, ts0 = g_split_on ? wtime() : 0;
                if (!rmap) rmap = malloc(sizeof(int) * n);
                if (!cptr) cptr = malloc(sizeof(int) * (3 * (size_t)n + 3));
                if (!accB) accB = calloc(((size_t)ns + 1) * GBMAX, sizeof(double));
                int *const cq = cptr + n + 1, *const hq = cptr + 2 * (n + 1);
                const int u = B->lv_u[a0];
                const int *rank = B->lv_rank, *cp = B->lu_cp, *ca = B->lu_ca, *cr = B->lu_cr, *chh = B->lu_ch;
                const double *cv = B->lu_cv;
                int rj[GBMAX], wj[GBMAX]; size_t oX[GBMAX], oT[GBMAX], needX = 0, needT = 0;
                for (int j = 0; j < nbt; j++) {
                    rj[j] = B->A[B->slist[a0 + j]].nr; wj[j] = xzs ? rj[j] : 2 * rj[j];
                    oX[j] = needX; needX += (size_t)wj[j] * n; oT[j] = needT; needT += (size_t)wj[j] * u;
                }
                if (needX > bXcap) { free(bXs); free(bXsf); bXs = NULL; bXsf = NULL; bXcap = needX + needX / 4; }
                if (needT > bTcap) { free(bTs); free(bTsf); bTs = NULL; bTsf = NULL; bTcap = needT + needT / 4; }
                if (sp) {
                    if (!bXsf) bXsf = amalloc(sizeof(float) * bXcap);
                    if (!bTsf) bTsf = amalloc(sizeof(float) * bTcap);
                    if (!bGf) bGf = amalloc(sizeof(float) * (size_t)GBMAX * n * 64);
                } else {
                    if (!bXs) bXs = amalloc(sizeof(double) * bXcap);
                    if (!bTs) bTs = amalloc(sizeof(double) * bTcap);
                    if (!bG) bG = amalloc(sizeof(double) * (size_t)GBMAX * n * 64);
                }
                for (int j = 0; j < nbt; j++) {
                    const SpSym *Aj = &B->A[B->slist[a0 + j]];
                    const int r = rj[j];
                    if (sp) {
                        float *xs = bXsf + oX[j], *ts = bTsf + oT[j];
                        for (int k = 0; k < r; k++) {
                            rmap[Aj->rows[k]] = k;
                            memcpy(xs + (size_t)k * n, XPf + (size_t)rank[Aj->rows[k]] * n, sizeof(float) * (size_t)u);
                            if (!xzs) memcpy(xs + (size_t)(r + k) * n, ZPf + (size_t)rank[Aj->rows[k]] * n, sizeof(float) * (size_t)u);
                        }
                        memset(ts, 0, sizeof(float) * (size_t)wj[j] * u);
                        for (int e = 0; e < Aj->ef; e++) {
                            const float v = (float)Aj->fv[e];
                            const float *zq = ZPf + (size_t)rank[Aj->fc[e]] * n, *xq = XPf + (size_t)rank[Aj->fc[e]] * n;
                            float *tk = ts + (size_t)rmap[Aj->fr[e]] * u;
                            if (xzs) { for (int i = 0; i < u; i++) tk[i] += v * zq[i]; continue; }
                            float *tx = ts + (size_t)(r + rmap[Aj->fr[e]]) * u;
                            for (int i = 0; i < u; i++) { tk[i] += v * zq[i]; tx[i] += v * xq[i]; }
                        }
                    } else {
                        double *xs = bXs + oX[j], *ts = bTs + oT[j];
                        for (int k = 0; k < r; k++) {
                            rmap[Aj->rows[k]] = k;
                            memcpy(xs + (size_t)k * n, XP + (size_t)rank[Aj->rows[k]] * n, sizeof(double) * (size_t)u);
                            if (!xzs) memcpy(xs + (size_t)(r + k) * n, ZP + (size_t)rank[Aj->rows[k]] * n, sizeof(double) * (size_t)u);
                        }
                        memset(ts, 0, sizeof(double) * (size_t)wj[j] * u);
                        for (int e = 0; e < Aj->ef; e++) {
                            const double v = Aj->fv[e];
                            const double *zq = ZP + (size_t)rank[Aj->fc[e]] * n, *xq = XP + (size_t)rank[Aj->fc[e]] * n;
                            double *tk = ts + (size_t)rmap[Aj->fr[e]] * u;
                            if (xzs) { for (int i = 0; i < u; i++) tk[i] += v * zq[i]; continue; }
                            double *tx = ts + (size_t)(r + rmap[Aj->fr[e]]) * u;
                            for (int i = 0; i < u; i++) { tk[i] += v * zq[i]; tx[i] += v * xq[i]; }
                        }
                    }
                }
                const double ts1 = g_split_on ? wtime() : 0;
                double tg = 0, tq = 0;
                for (int q = 0; q < u; q++) {
                    int lo = cp[q], hi = cp[q + 1];
                    while (lo < hi) { const int mid = (lo + hi) >> 1; if (ca[mid] < a0) lo = mid + 1; else hi = mid; }
                    cq[q] = lo; hq[q] = lo < cp[q + 1] ? chh[lo] : 0;
                }
                for (int q0 = 0; q0 < u; ) {
                    if (hq[q0] == 0) { q0++; continue; }
                    int q1 = q0 + 1, h = hq[q0]; double area = h;
                    while (q1 < u && q1 - q0 < 64) {
                        const int hn = hq[q1] > h ? hq[q1] : h;
                        if (q1 - q0 >= 16 && (double)hn * (q1 - q0 + 1) > 1.3333 * (area + hq[q1])) break;
                        h = hn; area += hq[q1]; q1++;
                    }
                    const int ww = q1 - q0;
                    const size_t pst = (size_t)h * ww;
                    const double tp0 = g_split_on ? wtime() : 0;
                    for (int j = 0; j < nbt; j++) {
                        if (sp) {
                            const float one_ = 1.0f, zero_ = 0.0f;
                            BL(sgemm_)("N", "T", &h, &ww, &wj[j], &one_, bXsf + oX[j], &n, bTsf + oT[j] + q0, &u, &zero_, bGf + j * pst, &h);
                        } else pdgemm("N", "T", &h, &ww, &wj[j], &DONE, bXs + oX[j], &n, bTs + oT[j] + q0, &u, &DZERO, bG + j * pst, &h);
                    }
                    const double tp1 = g_split_on ? wtime() : 0;
                    for (int q = q0; q < q1; q++) {
                        const size_t co = (size_t)(q - q0) * h;
                        if (sp) {
                            const float *g0 = bGf + co;
                            for (int e = cq[q]; e < cp[q + 1]; e++) {
                                const double v = cv[e]; const float *gr = g0 + cr[e];
                                double *ab = accB + (size_t)(ca[e] - a0) * GBMAX;
                                for (int j = 0; j < nbt; j++) ab[j] += v * (double)gr[j * pst];
                            }
                        } else {
                            const double *g0 = bG + co;
                            for (int e = cq[q]; e < cp[q + 1]; e++) {
                                const double v = cv[e]; const double *gr = g0 + cr[e];
                                double *ab = accB + (size_t)(ca[e] - a0) * GBMAX;
                                for (int j = 0; j < nbt; j++) ab[j] += v * gr[j * pst];
                            }
                        }
                    }
                    if (g_split_on) { tg += tp1 - tp0; tq += wtime() - tp1;
                        double ne = 0; for (int q = q0; q < q1; q++) ne += cp[q + 1] - cq[q];
                        double fl = 0; for (int j = 0; j < nbt; j++) fl += 2.0 * ww * (double)h * wj[j];
                        #pragma omp critical(split)
                        { g_split[4] += fl; g_split[5] += nbt; g_split[6] += ne * nbt; } }
                    q0 = q1;
                }
                const double hsc = xzs ? 2.0 : 1.0;
                for (int j = 0; j < nbt; j++) {
                    const int aj = a0 + j;
                    const size_t cij = B->con[B->slist[aj]];
                    for (int b = aj; b < ns; b++) {
                        const double sum = hsc * accB[(size_t)(b - a0) * GBMAX + j];
                        if (lbuf) lbuf[(size_t)sl[aj] * lnl + sl[b]] += sum; else MADDL(scon[b], cij, sum);
                    }
                }
                memset(accB, 0, sizeof(double) * (size_t)(ns - a0) * GBMAX);
                if (g_split_on) {
                    #pragma omp atomic
                    g_split[1] += ts1 - ts0;
                    #pragma omp atomic
                    g_split[2] += tg;
                    #pragma omp atomic
                    g_split[3] += tq; }
                if (g_asmdbg) { g_asmr[1] += wtime() - tr0; g_asmr[4] += (double)nbt * (ns - a0); }
                continue;
            }
          }
          for (int a = a_lo; a < a_hi; a++) {
            const int t = B->slist[a];
            const SpSym *At = &B->A[t];
            const size_t ci = B->con[t];
            const double tr0 = g_asmdbg ? wtime() : 0;
            if (B->route[t] == 0) {
                const double tq0 = g_split_on ? wtime() : 0;
                if (!tmp) tmp = amalloc(sizeof(double) * (fend + 1));
                const int k0 = foff[a];
                double *restrict tp = tmp;
                if (At->ef == 1) {                     /* fused single pass: diagonal unit-type */
                    const double *xq = X + (size_t)At->fc[0] * n, *zp = Zi + (size_t)At->fr[0] * n;
                    const double v = At->fv[0];
                    for (int k = k0; k < fend; k++) tp[k] = v * ffv[k] * xq[ffr[k]] * zp[ffc[k]];
                } else if (At->ef == 2) {              /* fused single pass: edge-type */
                    const double *xq0 = X + (size_t)At->fc[0] * n, *zp0 = Zi + (size_t)At->fr[0] * n;
                    const double *xq1 = X + (size_t)At->fc[1] * n, *zp1 = Zi + (size_t)At->fr[1] * n;
                    const double v0 = At->fv[0], v1 = At->fv[1];
                    for (int k = k0; k < fend; k++) {
                        const int r = ffr[k], c = ffc[k];
                        tp[k] = ffv[k] * (v0 * xq0[r] * zp0[c] + v1 * xq1[r] * zp1[c]);
                    }
                } else {
                    memset(tp + k0, 0, sizeof(double) * (fend - k0));
                    for (int e = 0; e < At->ef; e++) {
                        const double *xq = X + (size_t)At->fc[e] * n;
                        const double *zp = Zi + (size_t)At->fr[e] * n;
                        const double v = At->fv[e];
                        for (int k = k0; k < fend; k++) tp[k] += v * ffv[k] * xq[ffr[k]] * zp[ffc[k]];
                    }
                }
                for (int b = a; b < ns; b++) {
                    const int kb = foff[b], ke = foff[b + 1];
                    double sum = tp[kb];
                    for (int k = kb + 1; k < ke; k++) sum += tp[k];
                    if (lbuf) lbuf[(size_t)sl[a] * lnl + sl[b]] += sum; else MADDL(scon[b], ci, sum);
                }
                if (g_split_on) {
                    #pragma omp atomic
                    g_split[0] += wtime() - tq0; }
            } else {
                if (!Xp) {
                    Xp = amalloc(sizeof(double) * n2);
                    Tm = amalloc(sizeof(double) * n2);
                    G  = amalloc(sizeof(double) * n2);
                    if (!rmap) rmap = malloc(sizeof(int) * n);   /* (the batched route may have it: leaked before 19 October 2026) */
                }
                const double ts0 = g_split_on ? wtime() : 0;
                int r = At->nr;
                if (B->live && B->route[t] == 1 && B->lu_cp && (B->symg || sp || xzs)) {
                    /* 5.12, the symmetric gather: H = X A_a Zi + Zi A_a X = [X Zi] [A_a Zi; A_a X] by one
                     * product of inner dimension 2r, read at the upper triangle of the permuted order
                     * only (<A_b, X A_a Zi> = <A_b, H>/2; the diagonal values are halved): half the
                     * entries of the gather for the same flops (the panels stop at the diagonal) */
                    if (!acc) { acc = calloc((size_t)ns + 1, sizeof(double)); if (!cptr) cptr = malloc(sizeof(int) * (3 * (size_t)n + 3)); }
                    int *const cq = cptr + n + 1, *const hq = cptr + 2 * (n + 1);
                    const int u = B->lv_u[a];
                    const int *rank = B->lv_rank, *cp = B->lu_cp, *ca = B->lu_ca, *cr = B->lu_cr, *chh = B->lu_ch;
                    const double *cv = B->lu_cv;
                    if (sp) {
                        if (!Xsf) { Xsf = amalloc(sizeof(float) * 2 * n2); Tsf = amalloc(sizeof(float) * 2 * n2); Gf = amalloc(sizeof(float) * n2); }
                        for (int k = 0; k < r; k++) {
                            rmap[At->rows[k]] = k;
                            memcpy(Xsf + (size_t)k * n, XPf + (size_t)rank[At->rows[k]] * n, sizeof(float) * (size_t)u);
                            if (!xzs) memcpy(Xsf + (size_t)(r + k) * n, ZPf + (size_t)rank[At->rows[k]] * n, sizeof(float) * (size_t)u);
                        }
                        memset(Tsf, 0, sizeof(float) * (size_t)(xzs ? 1 : 2) * r * u);
                        for (int e = 0; e < At->ef; e++) {
                            const float v = (float)At->fv[e];
                            const float *zq = ZPf + (size_t)rank[At->fc[e]] * n, *xq = XPf + (size_t)rank[At->fc[e]] * n;
                            float *tk = Tsf + (size_t)rmap[At->fr[e]] * u;
                            if (xzs) { for (int i = 0; i < u; i++) tk[i] += v * zq[i]; continue; }
                            float *tx = Tsf + (size_t)(r + rmap[At->fr[e]]) * u;
                            for (int i = 0; i < u; i++) { tk[i] += v * zq[i]; tx[i] += v * xq[i]; }
                        }
                    } else {
                    if (!Xs) { Xs = amalloc(sizeof(double) * 2 * n2); Ts = amalloc(sizeof(double) * 2 * n2); }
                    for (int k = 0; k < r; k++) {
                        rmap[At->rows[k]] = k;
                        memcpy(Xs + (size_t)k * n, XP + (size_t)rank[At->rows[k]] * n, sizeof(double) * (size_t)u);
                        if (!xzs) memcpy(Xs + (size_t)(r + k) * n, ZP + (size_t)rank[At->rows[k]] * n, sizeof(double) * (size_t)u);
                    }
                    memset(Ts, 0, sizeof(double) * (size_t)(xzs ? 1 : 2) * r * u);
                    for (int e = 0; e < At->ef; e++) {
                        const double v = At->fv[e];
                        const double *zq = ZP + (size_t)rank[At->fc[e]] * n, *xq = XP + (size_t)rank[At->fc[e]] * n;
                        double *tk = Ts + (size_t)rmap[At->fr[e]] * u;
                        if (xzs) { for (int i = 0; i < u; i++) tk[i] += v * zq[i]; continue; }
                        double *tx = Ts + (size_t)(r + rmap[At->fr[e]]) * u;
                        for (int i = 0; i < u; i++) { tk[i] += v * zq[i]; tx[i] += v * xq[i]; }
                    }
                    }
                    const double ts1 = g_split_on ? wtime() : 0;
                    double tg = 0, tq = 0;
                    for (int q = 0; q < u; q++) {
                        int lo = cp[q], hi = cp[q + 1];
                        while (lo < hi) { const int mid = (lo + hi) >> 1; if (ca[mid] < a) lo = mid + 1; else hi = mid; }
                        cq[q] = lo; hq[q] = lo < cp[q + 1] ? chh[lo] : 0;
                    }
                    const int r2 = xzs ? r : 2 * r;
                    for (int q0 = 0; q0 < u; ) {
                        if (hq[q0] == 0) { q0++; continue; }
                        /* a panel: up to 64 columns while the rectangle stays within 4/3 of what is read */
                        int q1 = q0 + 1, h = hq[q0]; double area = h;
                        while (q1 < u && q1 - q0 < 64) {
                            const int hn = hq[q1] > h ? hq[q1] : h;
                            if (q1 - q0 >= 16 && (double)hn * (q1 - q0 + 1) > 1.3333 * (area + hq[q1])) break;
                            h = hn; area += hq[q1]; q1++;
                        }
                        const int ww = q1 - q0;
                        const double tp0 = g_split_on ? wtime() : 0;
                        if (sp) {
                            const float one_ = 1.0f, zero_ = 0.0f;
                            BL(sgemm_)("N", "T", &h, &ww, &r2, &one_, Xsf, &n, Tsf + q0, &u, &zero_, Gf, &h);
                        } else
                        pdgemm("N", "T", &h, &ww, &r2, &DONE, Xs, &n, Ts + q0, &u, &DZERO, G, &h);
                        const double tp1 = g_split_on ? wtime() : 0;
                        if (sp)
                            for (int q = q0; q < q1; q++) {
                                const float *gq = Gf + (size_t)(q - q0) * h;
                                for (int e = cq[q]; e < cp[q + 1]; e++) acc[ca[e] - a] += cv[e] * (double)gq[cr[e]];
                            }
                        else
                        for (int q = q0; q < q1; q++) {
                            const double *gq = G + (size_t)(q - q0) * h;
                            for (int e = cq[q]; e < cp[q + 1]; e++) acc[ca[e] - a] += cv[e] * gq[cr[e]];
                        }
                        if (g_split_on) { tg += tp1 - tp0; tq += wtime() - tp1;
                            double ne = 0; for (int q = q0; q < q1; q++) ne += cp[q + 1] - cq[q];
                            #pragma omp critical(split)
                            { g_split[4] += 2.0 * ww * (double)h * r2; g_split[5] += 1; g_split[6] += ne; } }
                        q0 = q1;
                    }
                    const double hsc = xzs ? 2.0 : 1.0;          /* (W A W is half of H) */
                    for (int b = a; b < ns; b++) {
                        const double sum = hsc * acc[b - a]; acc[b - a] = 0;
                        if (lbuf) lbuf[(size_t)sl[a] * lnl + sl[b]] += sum; else MADDL(scon[b], ci, sum);
                    }
                    if (g_split_on) {
                        #pragma omp atomic
                        g_split[1] += ts1 - ts0;
                        #pragma omp atomic
                        g_split[2] += tg;
                        #pragma omp atomic
                        g_split[3] += tq; }
                    if (g_asmdbg) { g_asmr[1] += wtime() - tr0; g_asmr[4] += ns - a; }
                    continue;
                }
                if (B->live && B->route[t] == 1) {
                    /* the product on the staircase of the entries that the constraints b >= a read:
                     * live indices 0..u-1 (permuted), column panels, the rows above the panel's
                     * highest entry only; the panel is read while it is in the cache */
                    if (!acc) { acc = calloc((size_t)ns + 1, sizeof(double)); if (!cptr) cptr = malloc(sizeof(int) * (3 * (size_t)n + 3)); }
                    int *const cq = cptr + n + 1, *const hq = cptr + 2 * (n + 1);
                    const int u = B->lv_u[a];
                    const int *rank = B->lv_rank, *cp = B->lv_cp, *ca = B->lv_ca, *cr = B->lv_cr, *chh = B->lv_ch;
                    const double *cv = B->lv_cv;
                    /* Xp (r x u): column p holds X[p, rows] - by the symmetry of X a gather from one column of XP */
                    /* 5.7: Xp (u x r, leading dimension n): column k is the column rank[rows[k]] of XP
                     * (X is symmetric), copied whole instead of gathered entry by entry */
                    if (sp) {
                        if (!Xsf) { Xsf = amalloc(sizeof(float) * 2 * n2); Tsf = amalloc(sizeof(float) * 2 * n2); Gf = amalloc(sizeof(float) * n2); }
                        for (int k = 0; k < r; k++) { rmap[At->rows[k]] = k; memcpy(Xsf + (size_t)k * n, XPf + (size_t)rank[At->rows[k]] * n, sizeof(float) * (size_t)u); }
                        memset(Tsf, 0, sizeof(float) * (size_t)r * u);
                        for (int e = 0; e < At->ef; e++) {
                            const float v = (float)At->fv[e];
                            const float *zq = ZPf + (size_t)rank[At->fc[e]] * n;
                            float *tk = Tsf + (size_t)rmap[At->fr[e]] * u;
                            for (int i = 0; i < u; i++) tk[i] += v * zq[i];
                        }
                    } else {
                    for (int k = 0; k < r; k++) { rmap[At->rows[k]] = k; memcpy(Xp + (size_t)k * n, XP + (size_t)rank[At->rows[k]] * n, sizeof(double) * (size_t)u); }
                    memset(Tm, 0, sizeof(double) * (size_t)r * u);
                    for (int e = 0; e < At->ef; e++) {
                        const double v = At->fv[e];
                        const double *zq = ZP + (size_t)rank[At->fc[e]] * n;
                        double *tk = Tm + (size_t)rmap[At->fr[e]] * u;
                        for (int i = 0; i < u; i++) tk[i] += v * zq[i];
                    }
                    }
                    double tg = 0, tq = 0;
                    const double ts1 = g_split_on ? wtime() : 0;
                    /* the entries still read in each column (from cq on) and the staircase: hq[q] rows are
                     * needed in column q or in a later one. Panels end where the staircase has fallen to
                     * 3/4 of its height at the panel's start (at least 24 columns): few products, each on
                     * what is needed */
                    for (int q = 0; q < u; q++) {
                        int lo = cp[q], hi = cp[q + 1];
                        while (lo < hi) { const int mid = (lo + hi) >> 1; if (ca[mid] < a) lo = mid + 1; else hi = mid; }
                        cq[q] = lo; hq[q] = lo < cp[q + 1] ? chh[lo] : 0;
                    }
                    for (int q = u - 2; q >= 0; q--) if (hq[q + 1] > hq[q]) hq[q] = hq[q + 1];
                    static double pfrac = -1; if (pfrac < 0) pfrac = getenv("BRISK_LIVEFRAC") ? atof(getenv("BRISK_LIVEFRAC")) : 0.75;
                    for (int q0 = 0; q0 < u; ) {
                        const int h = hq[q0];
                        if (h == 0) break;                             /* (nothing is read from here on) */
                        int q1 = q0 + 1;
                        while (q1 < u && (q1 - q0 < 24 || hq[q1] >= pfrac * h)) q1++;
                        const int ww = q1 - q0;
                        const double tp0 = g_split_on ? wtime() : 0;
                        /* G[p, q - q0] = sum_k Xp[p, k] Tm[q, k]: the entry (p, q) of X A Zi */
                        /* 5.7: the panel as h x ww (column q contiguous): the gather reads one column of
                         * h doubles at a time (in the first-level cache) instead of striding across the panel */
                        if (sp) {
                            const float one_ = 1.0f, zero_ = 0.0f;
                            BL(sgemm_)("N", "T", &h, &ww, &r, &one_, Xsf, &n, Tsf + q0, &u, &zero_, Gf, &h);
                        } else
                        pdgemm("N", "T", &h, &ww, &r, &DONE, Xp, &n, Tm + q0, &u, &DZERO, G, &h);
                        const double tp1 = g_split_on ? wtime() : 0;
                        if (sp)
                            for (int q = q0; q < q1; q++) {
                                const float *gq = Gf + (size_t)(q - q0) * h;
                                for (int e = cq[q]; e < cp[q + 1]; e++) acc[ca[e] - a] += cv[e] * (double)gq[cr[e]];
                            }
                        else
                        for (int q = q0; q < q1; q++) {
                            const double *gq = G + (size_t)(q - q0) * h;
                            for (int e = cq[q]; e < cp[q + 1]; e++) acc[ca[e] - a] += cv[e] * gq[cr[e]];
                        }
                        if (g_split_on) { tg += tp1 - tp0; tq += wtime() - tp1;
                            double ne = 0; for (int q = q0; q < q1; q++) ne += cp[q + 1] - cq[q];
                            #pragma omp critical(split)
                            { g_split[4] += 2.0 * ww * (double)h * r; g_split[5] += 1; g_split[6] += ne; } }
                        q0 = q1;
                    }
                    if (getenv("BRISK_LIVECHECK")) {       /* (test: the full product as before, and the largest difference) */
                        double *Xq = malloc(sizeof(double) * n2), *Tq = calloc(n2, sizeof(double)), *Gq = malloc(sizeof(double) * n2);
                        for (int k = 0; k < r; k++) memcpy(Xq + (size_t)k * n, X + (size_t)At->rows[k] * n, sizeof(double) * n);
                        for (int e = 0; e < At->ef; e++) { const int k = rmap[At->fr[e]]; const double v = At->fv[e]; const double *zq = Zi + (size_t)At->fc[e] * n; for (int j = 0; j < n; j++) Tq[k + (size_t)j * r] += v * zq[j]; }
                        BL(dgemm_)("N", "N", &n, &n, &r, &DONE, Xq, &n, Tq, &r, &DZERO, Gq, &n);
                        double dmax = 0, smax = 0; int bw = -1;
                        for (int b = a; b < ns; b++) {
                            double sum = 0;
                            for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * Gq[ffr[k] + (size_t)ffc[k] * n];
                            if (fabs(sum - acc[b - a]) > dmax) { dmax = fabs(sum - acc[b - a]); bw = b; }
                            if (fabs(sum) > smax) smax = fabs(sum);
                        }
                        if (dmax > 1e-9 * (smax + 1e-300)) fprintf(stderr, "LIVECHECK a %d (u %d, r %d): max difference %.3e at b %d (largest value %.3e)\n", a, u, r, dmax, bw, smax);
                        free(Xq); free(Tq); free(Gq);
                    }
                    for (int b = a; b < ns; b++) {
                        const double sum = acc[b - a]; acc[b - a] = 0;
                        if (lbuf) lbuf[(size_t)sl[a] * lnl + sl[b]] += sum; else MADDL(scon[b], ci, sum);
                    }
                    if (g_split_on) {
                        #pragma omp atomic
                        g_split[1] += ts1 - ts0;
                        #pragma omp atomic
                        g_split[2] += tg;
                        #pragma omp atomic
                        g_split[3] += tq; }
                    if (g_asmdbg) { g_asmr[1] += wtime() - tr0; g_asmr[4] += ns - a; }
                    continue;
                }
                if (!B->live && B->s2k && B->route[t] == 1) {
                    /* 18 October 2026 (see s2k, problem.c): H = X A_a Zi + Zi A_a X = Xs Ts' + Ts Xs' with Xs = X(:, rows)
                     * and Ts = Zi(:, cols) A's values gathered by row (n x r each): its upper triangle by one dsyr2k,
                     * then <A_b, X A_a Zi> = <A_b, H> / 2 at the upper entries of b >= a (diagonal halved), column by
                     * column (the column of H in the first-level cache, the entries in memory order) */
                    if (!Xs) { Xs = amalloc(sizeof(double) * 2 * n2); Ts = amalloc(sizeof(double) * 2 * n2); }
                    if (!acc) { acc = calloc((size_t)ns + 1, sizeof(double)); if (!cptr) cptr = malloc(sizeof(int) * (3 * (size_t)n + 3)); }
                    for (int k = 0; k < r; k++) { rmap[At->rows[k]] = k; memcpy(Xs + (size_t)k * n, X + (size_t)At->rows[k] * n, sizeof(double) * n); }
                    memset(Ts, 0, sizeof(double) * (size_t)r * n);
                    for (int e = 0; e < At->ef; e++) {
                        const double v = At->fv[e];
                        const double *zq = Zi + (size_t)At->fc[e] * n;
                        double *tk = Ts + (size_t)rmap[At->fr[e]] * n;
                        for (int i = 0; i < n; i++) tk[i] += v * zq[i];
                    }
                    const double ts1 = g_split_on ? wtime() : 0;
                    BL(dsyr2k_)("U", "N", &n, &r, &DONE, Xs, &n, Ts, &n, &DZERO, G, &n);
                    const double ts2 = g_split_on ? wtime() : 0;
                    const int *cp = B->lu_cp, *ca = B->lu_ca, *cr = B->lu_cr; const double *cv = B->lu_cv;
                    double ne = 0;
                    for (int q = 0; q < n; q++) {
                        int lo = cp[q], hi = cp[q + 1];
                        while (lo < hi) { const int mid = (lo + hi) >> 1; if (ca[mid] < a) lo = mid + 1; else hi = mid; }
                        const double *gq = G + (size_t)q * n;
                        for (int e = lo; e < cp[q + 1]; e++) acc[ca[e] - a] += cv[e] * gq[cr[e]];
                        ne += cp[q + 1] - lo;
                    }
                    for (int b = a; b < ns; b++) {
                        const double sum = acc[b - a]; acc[b - a] = 0;
                        if (lbuf) lbuf[(size_t)sl[a] * lnl + sl[b]] += sum; else MADDL(scon[b], ci, sum);
                    }
                    if (g_split_on) { const double ts3 = wtime();
                        #pragma omp critical(split)
                        { g_split[1] += ts1 - ts0; g_split[2] += ts2 - ts1; g_split[3] += ts3 - ts2; g_split[4] += 2.0 * n * (double)n * r; g_split[5] += 1; g_split[6] += ne; } }
                    if (g_asmdbg) { g_asmr[1] += wtime() - tr0; g_asmr[4] += ns - a; }
                    continue;
                }
                for (int k = 0; k < r; k++) rmap[At->rows[k]] = k;
                memset(Tm, 0, sizeof(double) * (size_t)r * n);
                for (int e = 0; e < At->ef; e++) {
                    const int k = rmap[At->fr[e]];
                    const double v = At->fv[e];
                    const double *zq = Zi + (size_t)At->fc[e] * n;
                    for (int j = 0; j < n; j++) Tm[k + (size_t)j * r] += v * zq[j];
                }
                if (B->route[t] == 3) {
                    /* G = X A_t Zi on the union pattern only: G[p,q] = sum_k X[p,rows_k] Tm[k,q] */
                    for (int p = 0; p < n; p++)
                        for (int k = 0; k < r; k++) Xp[k + (size_t)p * r] = X[p + (size_t)At->rows[k] * n];
                    const int *ufr = B->ufr, *ufc = B->ufc;
                    for (int u = 0; u < B->unf; u++) {
                        const double *xp = Xp + (size_t)ufr[u] * r, *tq = Tm + (size_t)ufc[u] * r;
                        double g = 0;
                        for (int k = 0; k < r; k++) g += xp[k] * tq[k];
                        G[ufr[u] + (size_t)ufc[u] * n] = g;
                    }
                } else {
                    for (int k = 0; k < r; k++)
                        memcpy(Xp + (size_t)k * n, X + (size_t)At->rows[k] * n, sizeof(double) * n);
                    const double ts1 = g_split_on ? wtime() : 0;
                    pdgemm("N", "N", &n, &n, &r, &DONE, Xp, &n, Tm, &r, &DZERO, G, &n);
                    if (g_split_on) { const double ts2 = wtime();
                        #pragma omp atomic
                        g_split[1] += ts1 - ts0;
                        #pragma omp atomic
                        g_split[2] += ts2 - ts1; }
                }
                const double ts3 = g_split_on ? wtime() : 0;
                for (int b = a; b < ns; b++) {
                    double sum = 0;
                    if (ffp) for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * G[ffp[k]];
                    else for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * G[ffr[k] + (size_t)ffc[k] * n];
                    if (lbuf) lbuf[(size_t)sl[a] * lnl + sl[b]] += sum; else MADDL(scon[b], ci, sum);
                }
                if (g_split_on) {
                    #pragma omp atomic
                    g_split[3] += wtime() - ts3; }
            }
            if (g_asmdbg) { const int rt = B->route[t] == 0 ? 0 : B->route[t] == 3 ? 2 : 1; g_asmr[rt] += wtime() - tr0; g_asmr[3 + rt] += ns - a; }
          }
        }
        free(Xp); free(Tm); free(G); free(tmp); free(Xs); free(Ts); free(Xsf); free(Tsf); free(Gf);
        free(accB); free(bXs); free(bTs); free(bG); free(bXsf); free(bTsf); free(bGf); free(sG);
        free(rmap); free(acc); free(cptr);
        ASM_TL_DONE;
    }
    if (par1) blas_serial_end();
    free(XP); free(ZP); free(XPf); free(ZPf);
    free(scon);
    if (nd == 0) return;

    /* phase 2: dense rows. G_a = X A_a Zi (BLAS-3); dense-dense via one dgemm per chunk */
    const int ch = s->chunk;
    for (int c0 = 0; c0 < nd; c0 += ch) {
        const int c1 = (c0 + ch < nd) ? c0 + ch : nd;
        const int par2 = nthreads() > 1 && c1 - c0 > 1 && !in_par();
        if (par2) blas_serial_begin();
        ASM_TL_MASTER;
        #pragma omp parallel if (par2)
        {
            ASM_TL_TEAM;
            double *W = amalloc(sizeof(double) * n2);
            #pragma omp for schedule(dynamic, 1)
            for (int a = c0; a < c1; a++) {
                const double *D = B->Ad + (size_t)a * n2;
                double *G = s->Gd + (size_t)(a - c0) * n2;
                if (B->drk && B->drk[a] > 0) {
                    /* 15 October 2026 (ISSUES 96): a dense row of low rank r: G = sum_k sig_k (X w_k)(Zi w_k)', three
                     * products of n^2 r instead of two of n^3 (G48_mb: the row <J, X> = 0, n = 3000) */
                    const int r = B->drk[a];
                    double *U = W, *V = W + (size_t)n * r;      /* W holds n^2 >= 2 n r doubles */
                    xsymm("L", "L", &n, &r, &DONE, X, &n, B->drW[a], &n, &DZERO, U, &n);
                    xsymm("L", "L", &n, &r, &DONE, Zi, &n, B->drW[a], &n, &DZERO, V, &n);
                    for (int k = 0; k < r; k++) { const double sg = B->drS[a][k]; double *vk = V + (size_t)k * n; for (int i = 0; i < n; i++) vk[i] *= sg; }
                    BL(dgemm_)("N", "T", &n, &n, &r, &DONE, U, &n, V, &n, &DZERO, G, &n);
                } else {
                xsymm("R", "L", &n, &n, &DONE, Zi, &n, D, &n, &DZERO, W, &n);
                xsymm("L", "L", &n, &n, &DONE, X, &n, W, &n, &DZERO, G, &n);
                }
                const int ci = B->con[B->dlist[a]];
                for (int b = 0; b < ns; b++) {
                    double sum = 0;
                    if (ffp) for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * G[ffp[k]];
                    else for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * G[ffr[k] + (size_t)ffc[k] * n];
                    MADD(ci, B->con[B->slist[b]], sum);
                }
            }
            free(W);
            ASM_TL_DONE;
        }
        if (par2) blas_serial_end();
        int rows = nd - c0, cols = c1 - c0, nn = (int)n2;
        pdgemm("T", "N", &rows, &cols, &nn, &DONE, B->Ad + (size_t)c0 * n2, &nn,
                   s->Gd, &nn, &DZERO, s->Mdd, &rows);
        for (int a = c0; a < c1; a++) {
            const int ci = B->con[B->dlist[a]];
            for (int b = a; b < nd; b++)
                MADD(ci, B->con[B->dlist[b]], s->Mdd[(b - c0) + (size_t)(a - c0) * rows]);
        }
    }
}

/* LP block into the sparse factor: offsets of every (constraint, constraint) product
 * precomputed, and columns equal up to sign (split free pairs) merged into one group
 * whose weight is the sum of theirs (the hash lookup per entry was 7% of the run on the
 * TSSOS AC-OPF models, where the pairs are half of the LP columns). */
typedef struct { const Block *B; long uid; int ng; int *gof, *grep; size_t *optr, *off; double *gd; } LPCache;
static LPCache g_lpc[8];
static const int *g_bpf = NULL;      /* bordered pairs: LP column -> pair index (-1), block */
static const Block *g_bpB = NULL;
static double *g_bD = NULL;          /* accumulated pair weights D+ + D- */
static int g_nlpc = 0;
static int lp_col_eq(const Block *B, int p, int q) {
    const int np = B->lp_ptr[p + 1] - B->lp_ptr[p];
    if (np != B->lp_ptr[q + 1] - B->lp_ptr[q] || np == 0) return 0;
    const double *vp = B->lp_val + B->lp_ptr[p], *vq = B->lp_val + B->lp_ptr[q];
    const int *cp = B->lp_con + B->lp_ptr[p], *cq = B->lp_con + B->lp_ptr[q];
    const double sg = (vp[0] > 0) == (vq[0] > 0) ? 1.0 : -1.0;
    for (int e = 0; e < np; e++) if (cp[e] != cq[e] || vp[e] != sg * vq[e]) return 0;
    return 1;
}
static LPCache *lp_cache(const Block *B, const SChol *S) {
    for (int i = 0; i < g_nlpc; i++) if (g_lpc[i].B == B && g_lpc[i].uid == schol_uid(S)) return &g_lpc[i];
    LPCache *c = NULL;
    #pragma omp critical(lp_cache_build)
    {
        for (int i = 0; i < g_nlpc; i++) if (g_lpc[i].B == B && g_lpc[i].uid == schol_uid(S)) c = &g_lpc[i];
        if (!c) {
            const int n = B->n;
            /* hash columns (canonical sign), open addressing over representatives */
            int hs = 1; while (hs < 2 * n + 2) hs <<= 1;
            int *tab = malloc(sizeof(int) * hs);
            for (int i = 0; i < hs; i++) tab[i] = -1;
            int *gof = malloc(sizeof(int) * (n + 1)), *grep = malloc(sizeof(int) * (n + 1)), ng = 0;
            for (int p = 0; p < n; p++) {
                const int a0 = B->lp_ptr[p], a1 = B->lp_ptr[p + 1];
                if (a1 == a0) { gof[p] = -1; continue; }
                const double sg = B->lp_val[a0] > 0 ? 1.0 : -1.0;
                uint64_t h = 1469598103934665603ULL;
                for (int a = a0; a < a1; a++) {
                    double v = sg * B->lp_val[a] + 0.0; uint64_t u; memcpy(&u, &v, 8);
                    h = (h ^ (uint64_t)B->lp_con[a]) * 1099511628211ULL; h = (h ^ u) * 1099511628211ULL;
                }
                size_t t = (size_t)(h & (uint64_t)(hs - 1));
                int g = -1;
                while (tab[t] >= 0) {
                    if (lp_col_eq(B, grep[tab[t]], p)) { g = tab[t]; break; }
                    t = (t + 1) & (size_t)(hs - 1);
                }
                if (g < 0) { g = ng++; grep[g] = p; tab[t] = g; }
                gof[p] = g;
            }
            free(tab);
            size_t *optr = malloc(sizeof(size_t) * (ng + 1)), tot = 0;
            for (int g = 0; g < ng; g++) { const int np = B->lp_ptr[grep[g] + 1] - B->lp_ptr[grep[g]]; optr[g] = tot; tot += (size_t)np * (np + 1) / 2; }
            optr[ng] = tot;
            size_t *off = malloc(sizeof(size_t) * (tot + 1));
            for (int g = 0; g < ng; g++) {
                const int p = grep[g];
                size_t k = optr[g];
                for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++)
                    for (int b = a; b < B->lp_ptr[p + 1]; b++) off[k++] = schol_offset(S, B->lp_con[b], B->lp_con[a]);
            }
            static int rr = 0;
            if (g_nlpc < 8) c = &g_lpc[g_nlpc++];
            else { c = &g_lpc[rr]; rr = (rr + 1) % 8; free(c->gof); free(c->grep); free(c->optr); free(c->off); free(c->gd); }
            c->B = B; c->uid = schol_uid(S); c->ng = ng; c->gof = gof; c->grep = grep; c->optr = optr; c->off = off;
            c->gd = malloc(sizeof(double) * (ng + 1));
        }
    }
    return c;
}
/* M += A_L D A_L' for an LP block with dense columns, as one matrix product.
 *
 * The pair loops below cost sum_p d_p^2 / 2 scattered updates of M (d_p: the constraints that
 * variable p enters) and run on one thread. Dense columns arise from free variables and inequalities
 * that touch many constraints: the JuMP wrapper passes every linear equality constraint of a model as
 * a free variable split into two LP variables, whose column holds that constraint's coefficients, so
 * an equality over many variables gives two columns as long as the number of variables; SeDuMi data
 * (MATLAB, CVXPY, solve_sedumi) and SDPA files can have them too. A model with an equality over
 * 6,000 variables (602 such LP variables, m = 6,210): schur_lp held one thread for 82 % of a 100 s
 * solve while the log said eight threads.
 *
 * Here the block's term is formed as L = A_L' (D A_L) by pdgemm (nc x nc, nc = the block's
 * constraints; A_L as n x nc with the data of constraint t in column t), which runs on the solver's
 * threads in tiles fixed by the sizes, and L's lower triangle is added to M column by column, every
 * entry of M by one thread, once: the bits do not depend on the thread count. They differ from the
 * pair loops' (another order of the sums) on the blocks that take this route only.
 *
 * Taken when M is the dense matrix itself (MADDL's first and last cases: the direct route of the
 * class assembly, or a dense Schur complement), outside parallel regions (inside one, a block's
 * caller has buffers of its own; the product would also run on one thread there), when the data are
 * dense enough that the product's 2 nc^2 n flops beat the pair loops (sum_p d_p^2 > 0.1 nc^2 n, the
 * rule of schur_soc), when the product is worth it at all (nc^2 n >= 1e6: below that the pair loops
 * cost microseconds, and the route's allocations made small occupation-measure relaxations 20-60 %
 * slower; their bits stay those of the pair loops), and when A_L, D A_L and L fit in a sixteenth of
 * the memory limit. The sparse factor's storage (g_schol, the cached offsets below) and the
 * envelope keep the pair loops.
 * BRISK_LPDENSE=0: never; =1: whenever M is dense (tests). */
static int schur_lp_dense(const Block *B, const BS *s, int m, double *M) {
    const int n = B->n, nc = B->ncon;
    if (n == 0 || nc == 0 || in_par()) return 0;
    if (!((g_direct_g | g_direct_t) || (!g_pos && !g_schol && !g_env))) return 0;
    static int env = -2; if (env == -2) { const char *e = getenv("BRISK_LPDENSE"); env = e ? atoi(e) : -1; }
    if (env == 0) return 0;
    double s2 = 0;
    for (int p = 0; p < n; p++) { const double d = B->lp_ptr[p + 1] - B->lp_ptr[p]; s2 += d * d; }
    if (env != 1 && !(s2 > 0.1 * (double)nc * nc * n && (double)nc * nc * n >= 1e6)) return 0;
    if (16.0 * (double)nc * n + 8.0 * (double)nc * nc >= brisk_mem_limit() / 16) return 0;
    double *Ad = calloc((size_t)n * nc + 1, sizeof(double)), *Bd = malloc(sizeof(double) * ((size_t)n * nc + 1));
    double *L = malloc(sizeof(double) * ((size_t)nc * nc + 1));
    if (!Ad || !Bd || !L) { free(Ad); free(Bd); free(L); return 0; }
    for (int t = 0; t < nc; t++) { const SpSym *A = &B->A[t]; for (int q = 0; q < A->nnz; q++) Ad[(size_t)A->row[q] + (size_t)t * n] += A->val[q]; }
    for (int t = 0; t < nc; t++) {
        const double *a = Ad + (size_t)t * n; double *b = Bd + (size_t)t * n;
        for (int p = 0; p < n; p++) b[p] = a[p] * (s->X[p] * s->Zi[p]);     /* D A_L: the pair loops' va = a_p d_p */
    }
    pdgemm("T", "N", &nc, &nc, &n, &DONE, Ad, &n, Bd, &n, &DZERO, L, &nc);
    const int *con = B->con;
    #pragma omp parallel for schedule(static) if (nthreads() > 1 && (double)nc * nc >= 65536.0)
    for (int u = 0; u < nc; u++) {
        const double *lu = L + (size_t)u * nc;
        for (int t = u; t < nc; t++) {
            const size_t r = (size_t)(con[t] > con[u] ? con[t] : con[u]), c = (size_t)(con[t] > con[u] ? con[u] : con[t]);
            M[r + c * (size_t)m] += lu[t];                                  /* MADDL's dense write: lower triangle */
        }
    }
    free(Ad); free(Bd); free(L);
    return 1;
}
static void schur_lp(const Block *B, const BS *s, int m, double *M) {
    if (schur_lp_dense(B, s, m, M)) return;
    LPCache *lc = (g_schol && !g_pos) ? lp_cache(B, g_schol) : NULL;
    if (lc) {
        double *pm = schol_values(g_schol), *gd = lc->gd;
        memset(gd, 0, sizeof(double) * lc->ng);
        if (g_bpf && B == g_bpB) {
            /* bordered split pairs: their weights go to the border, not into M */
            for (int p = 0; p < B->n; p++) {
                if (g_bpf[p] >= 0) { g_bD[g_bpf[p]] += s->X[p] * s->Zi[p]; continue; }
                if (lc->gof[p] >= 0) gd[lc->gof[p]] += s->X[p] * s->Zi[p];
            }
        } else
        for (int p = 0; p < B->n; p++) if (lc->gof[p] >= 0) gd[lc->gof[p]] += s->X[p] * s->Zi[p];
        for (int g = 0; g < lc->ng; g++) {
            const int p = lc->grep[g];
            const double d = gd[g];
            const size_t *of = lc->off + lc->optr[g];
            const int a0 = B->lp_ptr[p], a1 = B->lp_ptr[p + 1];
            const double *v = B->lp_val;
            for (int a = a0; a < a1; a++) {
                const double va = v[a] * d;
                for (int b = a; b < a1; b++) { const size_t o = *of++; if (o != SIZE_MAX) pm[o] += va * v[b]; }
            }
        }
        return;
    }
    for (int p = 0; p < B->n; p++) {
        double dp = s->X[p] * s->Zi[p];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) {
            size_t ca = B->lp_con[a];
            double va = B->lp_val[a] * dp;
            for (int b = a; b < B->lp_ptr[p + 1]; b++)
                MADDL(B->lp_con[b], ca, va * B->lp_val[b]);
        }
    }
}


/* 5.16: a second-order cone block: M += A_c W^2 A_c' with W^2 = beta^2 (I + 4 |v|^2 v v' - 2 (v (Jv)' +
 * (Jv) v')): the identity part variable by variable (an LP block of weight beta^2), the rank-two part
 * over the block's constraints with p = A_c v, q = A_c J v */
/* 5.16: the cone's term of the Schur complement, beta^2 (A_c A_c' + 4|v|^2 p p' - 2 (p q' + q p')) on
 * its nc constraints (A_c: the cone's data, nc x n; p = A_c v, q = A_c J v). Column u of this nc x nc
 * matrix is formed in a buffer of its own (A_c A_c' from the coordinates of constraint u and their
 * constraint lists, or read from one product A_c A_c' when the data are dense; then the rank-two
 * term) and added to M once. The columns run on the solver's threads; every entry of M is written by
 * one thread, in an order fixed by the data, so the bits do not depend on the thread count. Until
 * then the term ran on one thread, row by row across M's columns (ncm_100: 1.2 s of 4.4 s at four
 * threads; filter48_socp, a cone of 49 with dense data: 1.0 s of 2.0 s). */
static void schur_soc(const Block *B, const BS *s, int m, double *M) {
    const int n = B->n, nc = B->ncon;
    if (nc == 0) return;
    const int *con = B->con;
    const double b2 = s->sbeta * s->sbeta, *v = s->sv;
    double vv = 0;
    for (int i = 0; i < n; i++) vv += v[i] * v[i];
    const double c4 = 4.0 * vv * b2, c2 = -2.0 * b2;
    /* where the values go: 0 M itself (dense), 1 the sparse factor's storage at offsets known for
     * the cone's constraint pairs, 2 the envelope, 3 through MADDL on this thread (the sparse factor
     * by search, or the local buffer of a caller that set one up) */
    const CliqueTab *pos = NULL;
    if (g_bpos && g_schol) { const long k = B - g_blk0; if (k >= 0 && k < g_nbk) pos = g_bpos[k]; }
    const int mode = (g_direct_g | g_direct_t) ? 0 : g_pos ? 3 : !g_schol && !g_env ? 0 : g_schol ? (pos ? 1 : 3) : 2;
    double *pm = mode == 1 ? g_pm : NULL;
    const Envelope *env = g_env;
    const int par = mode != 3 && !in_par() && nthreads() > 1 && (double)nc * nc >= 65536.0;
    double *pq = malloc(sizeof(double) * 2 * ((size_t)nc + 1));
    #pragma omp parallel for schedule(static) if (par)
    for (int t = 0; t < nc; t++) {
        const SpSym *A = &B->A[t];
        double pt = 0, qt = 0;
        for (int q = 0; q < A->nnz; q++) { const int i = A->row[q]; pt += A->val[q] * v[i]; qt += A->val[q] * (i ? -v[i] : v[i]); }
        pq[2 * (size_t)t] = pt; pq[2 * (size_t)t + 1] = qt;
    }
    /* A_c A_c' as one product when the data are dense: the pair loops cost sum_p nnz_p^2 scattered
     * updates (about 1 ns each), the product 2 nc^2 n flops in fixed panels (about 0.05 ns each) */
    double s2 = 0;
    for (int p = 0; p < n; p++) { const double d = B->lp_ptr[p + 1] - B->lp_ptr[p]; s2 += d * d; }
    double *L = NULL;
    static int dense_env = -2; if (dense_env == -2) { const char *e = getenv("BRISK_SOCDENSE"); dense_env = e ? atoi(e) : -1; }
    if (dense_env != 0 && (dense_env == 1 || s2 > 0.1 * (double)nc * nc * n)
        && 8.0 * ((double)nc * nc + (double)nc * n) < brisk_mem_limit() / 16) {
        double *Ad = calloc((size_t)n * nc + 1, sizeof(double));
        L = malloc(sizeof(double) * ((size_t)nc * nc + 1));
        if (Ad && L) {
            for (int t = 0; t < nc; t++) { const SpSym *A = &B->A[t]; for (int q = 0; q < A->nnz; q++) Ad[(size_t)A->row[q] + (size_t)t * n] += A->val[q]; }
            pdgemm("T", "N", &nc, &nc, &n, &DONE, Ad, &n, Ad, &n, &DZERO, L, &nc);
        } else { free(L); L = NULL; }
        free(Ad);
    }
    int *loc = NULL;
    if (!L) { loc = malloc(sizeof(int) * ((size_t)m + 1)); for (int t = 0; t < nc; t++) loc[con[t]] = t; }
    #pragma omp parallel if (par)
    {
        double *col = L ? NULL : calloc((size_t)nc + 1, sizeof(double));
        #pragma omp for schedule(dynamic, 16)
        for (int u = 0; u < nc; u++) {
            const double *cu = L ? L + (size_t)u * nc : col;
            if (!L) {                                     /* (A_c A_c')_tu, t >= u, over the coordinates of constraint u */
                const SpSym *A = &B->A[u];
                for (int q = 0; q < A->nnz; q++) {
                    const int p = A->row[q], e1 = B->lp_ptr[p + 1];
                    const double a = A->val[q];
                    int e = B->lp_ptr[p], hi = e1;            /* the constraint list of p is sorted: start at con[u] */
                    while (e < hi) { const int mid = (e + hi) >> 1; if (B->lp_con[mid] < con[u]) e = mid + 1; else hi = mid; }
                    for (; e < e1; e++) col[loc[B->lp_con[e]]] += a * B->lp_val[e];
                }
            }
            const double pu = pq[2 * (size_t)u], qu = pq[2 * (size_t)u + 1];
            const size_t cuc = (size_t)con[u];
            for (int t = u; t < nc; t++) {
                const double pt = pq[2 * (size_t)t], qt = pq[2 * (size_t)t + 1];
                const double val = b2 * cu[t] + c4 * pt * pu + c2 * (pt * qu + qt * pu);
                const size_t ct = (size_t)con[t];
                if (mode == 0) M[ct + cuc * (size_t)m] += val;
                else if (mode == 1) pm[clique_off(pos, t, u)] += val;
                else if (mode == 2) env_add(env, ct, cuc, val);
                else MADDL(ct, cuc, val);
                if (!L) col[t] = 0;
            }
        }
        free(col);
    }
    free(loc); free(L); free(pq);
}


static int *deg_ctx;
static int cmp_by_deg(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return deg_ctx[x] - deg_ctx[y];
}

/* Build the envelope structure; returns NULL if the Schur complement is not
 * sparse enough for the envelope factorization to beat dense BLAS.          */
/* Pattern of the Schur complement as adjacency lists (same cliques as envelope_build):
 * every SDP block makes a clique of the constraints touching it, every LP variable makes
 * a clique of its constraints. */
static int schur_pattern(const Problem *P, long **aptr_out, int **adj_out) {
    const int m = P->m;
    int ncl = 0;
    long tot = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_LP) { ncl++; tot += B->ncon; }      /* (a second-order cone too: W^2 couples all its constraints) */
        else for (int p = 0; p < B->n; p++) { ncl++; tot += B->lp_ptr[p + 1] - B->lp_ptr[p]; }
    }
    int *cptr = malloc(sizeof(int) * (ncl + 1));
    int *cidx = malloc(sizeof(int) * (tot ? tot : 1));
    int c = 0; long pos = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_LP) {
            cptr[c++] = (int)pos;
            for (int t = 0; t < B->ncon; t++) cidx[pos++] = B->con[t];
        } else for (int p = 0; p < B->n; p++) {
            cptr[c++] = (int)pos;
            for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) cidx[pos++] = B->lp_con[a];
        }
    }
    cptr[ncl] = (int)pos;
    int *nptr = calloc(m + 1, sizeof(int));
    for (long q = 0; q < tot; q++) nptr[cidx[q] + 1]++;
    for (int i = 0; i < m; i++) nptr[i + 1] += nptr[i];
    int *ncl_of = malloc(sizeof(int) * (tot ? tot : 1));
    int *fill = malloc(sizeof(int) * m);
    memcpy(fill, nptr, sizeof(int) * m);
    for (int cc = 0; cc < ncl; cc++)
        for (int q = cptr[cc]; q < cptr[cc + 1]; q++) ncl_of[fill[cidx[q]]++] = cc;
    free(fill);
    int *mark = malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) mark[i] = -1;
    long *aptr = calloc(m + 1, sizeof(long));
    long adj_tot = 0;
    for (int i = 0; i < m; i++) {
        long d = 0;
        for (int q = nptr[i]; q < nptr[i + 1]; q++)
            for (int r = cptr[ncl_of[q]]; r < cptr[ncl_of[q] + 1]; r++) {
                int j = cidx[r];
                if (j != i && mark[j] != i) { mark[j] = i; d++; }
            }
        aptr[i + 1] = aptr[i] + d;
        adj_tot += d;
        if (adj_tot > 200000000L) {
            free(cptr); free(cidx); free(nptr); free(ncl_of); free(mark); free(aptr);
            return 1;
        }
    }
    int *adj = malloc(sizeof(int) * (adj_tot ? adj_tot : 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    for (int i = 0; i < m; i++) {
        long w = aptr[i];
        for (int q = nptr[i]; q < nptr[i + 1]; q++)
            for (int r = cptr[ncl_of[q]]; r < cptr[ncl_of[q] + 1]; r++) {
                int j = cidx[r];
                if (j != i && mark[j] != i) { mark[j] = i; adj[w++] = j; }
            }
    }
    free(cptr); free(cidx); free(nptr); free(ncl_of); free(mark);
    *aptr_out = aptr;
    *adj_out = adj;
    return 0;
}

/* Fill-reducing sparse Cholesky of the Schur complement, chosen when its exact
 * factorization cost (from the symbolic phase) beats both the dense path and the
 * envelope. */
static SChol *g_sb_cache = NULL; static const Problem *g_sb_cache_P = NULL; static int g_sb_cache_m = 0;
static Envelope *envelope_build(const Problem *P, const Params *par, int verbose);
static void envelope_free(Envelope *E);
static SChol *sparse_build(const Problem *P, const Params *par, int verbose);
/* 5.8 (a user's report): does the standard method need a dense Schur complement? The early
 * routing (main.c) tests the pattern of the problem as read; the sparse factorization and the
 * envelope can still be rejected inside brisk_solve, which then exited with advice when the
 * dense matrix does not fit (pglib case89 at minimal order, m = 34,898, after 4 s). Called by
 * run_pipeline for large m; the sparse analysis it makes is kept for the solve. */
int schur_dense_needed(const Problem *P, const Params *par) {
    SChol *SC = sparse_build(P, par, 0);
    if (SC) { schol_free(g_sb_cache); g_sb_cache = SC; g_sb_cache_P = P; g_sb_cache_m = P->m; return 0; }
    Envelope *E = envelope_build(P, par, 0);
    if (E) { envelope_free(E); return 0; }
    return 1;
}
static SChol *sparse_build(const Problem *P, const Params *par, int verbose) {
    const int m = P->m;
    if (g_sb_cache && g_sb_cache_P == P && g_sb_cache_m == m) {
        SChol *SC = g_sb_cache; g_sb_cache = NULL; g_sb_cache_P = NULL;
        if (verbose) printf("Schur pattern: sparse Cholesky (analysis from the routing check), %zu nonzeros in L, %.1e flops, %d supernodes\n", schol_nnz(SC), schol_flops(SC), schol_nsuper(SC));
        return SC;
    }
    if (m < 200 || par->sparse_schur == 0) return NULL;
    /* fill cap: a factor with more than ~half of the lower triangle cannot win; below
     * m = 8000 the analysis may look that far (moment-SOS image forms: 30% dense patterns
     * made of a few overlapping cliques factor in a tenth of the dense flops) */
    static double fillfrac = -1;
    if (fillfrac < 0) { const char *e = getenv("BRISK_FILLFRAC"); fillfrac = e ? atof(e) : 0.30; }   /* the fraction above m = 8000 (0.25 until 18 October 2026) */
    size_t cap = (size_t)((m <= 8000 ? 0.45 : fillfrac) * (double)m * m) + 1000;
    /* memory: L costs ~35-40 bytes per entry since 18 October 2026 (the factor and the assembled M with the panels'
     * diagonal blocks, the offset tables; 62 before), so 1e8 per 8 GB keeps it at about half the memory (6e7 before;
     * with 0.30 m^2 above m = 8000, was 0.25: mbb14weight_nmt__016 655 -> 660 s, 4414 -> 4012 MB); 2e7 entries is about 0.75 GB. L holds at least the lower half of the pattern, so a pattern
     * beyond the cap is rejected before the analysis (whose per-row lists would stay in
     * the heap: roa_vdp_inner_d12, m = 15327 with a 45% dense pattern, then exceeded the
     * address-space limit when the dense Schur matrix was allocated) */
    { extern double g_fillmax; if (!getenv("BRISK_FILLMAX")) g_fillmax = 1e8 * brisk_mem_scale(); if (cap > (size_t)g_fillmax) cap = (size_t)g_fillmax; }
    {   /* 5.12: the constraints of one block form a clique of the pattern: when the largest clique
         * alone passes the cap, the pattern is not built (a moment block touches every constraint:
         * 0.55 s at m = 11627) */
        double cl = 0;   /* SDP blocks only: an LP block couples two constraints only through a shared
                          * variable (pglib case1888_rte: its LP block touches nearly every constraint,
                          * and counting it as a clique sent a 1.5e6-entry factor to the dense path) */
        for (int k = 0; k < P->nblk; k++) { if (P->blk[k].type == BLK_LP) continue; const double c = P->blk[k].ncon; if (0.5 * c * (c + 1) > cl) cl = 0.5 * c * (c + 1); }
        if (cl > (double)cap && !getenv("BRISK_DUMP_SCHUR")) {
            if (verbose > 1 || getenv("BRISK_ROUTEDBG")) printf("Schur pattern: a block's constraints alone give %.3g entries in the lower triangle, above the cap of %.3g for a sparse factorization: not analysed\n", cl, (double)cap);
            return NULL;
        }
    }
    long *aptr = NULL;
    int *adj = NULL;
    if (schur_pattern(P, &aptr, &adj)) return NULL;
    if (getenv("BRISK_DUMP_SCHUR")) {
        FILE *fd = fopen(getenv("BRISK_DUMP_SCHUR"), "wb");
        if (fd) { fwrite(&m, sizeof(int), 1, fd); fwrite(aptr, sizeof(long), m + 1, fd); fwrite(adj, sizeof(int), aptr[m], fd); fclose(fd); }
        if (getenv("BRISK_DUMP_EXIT")) exit(0);
    }
    {   /* the chordal conversion may have analysed this very pattern already */
        SChol *SC0 = chordal_take_analysis(m, (size_t)aptr[m] / 2);
        if (SC0) {
            free(aptr); free(adj);
            if (verbose)
                printf("Schur pattern: sparse Cholesky (analysis from the conversion), %zu nonzeros in L, %.1e flops, %d supernodes\n",
                       schol_nnz(SC0), schol_flops(SC0), schol_nsuper(SC0));
            return SC0;
        }
    }
    int *deg = malloc(sizeof(int) * m);
    int **nbr = malloc(sizeof(int *) * m);
    for (int i = 0; i < m; i++) { deg[i] = (int)(aptr[i + 1] - aptr[i]); nbr[i] = adj + aptr[i]; }
    if ((double)aptr[m] * 0.5 > (double)cap) {
        if (verbose > 1 || getenv("BRISK_ROUTEDBG")) printf("Schur pattern: %.3g entries in the lower triangle (%.0f %% of it), above the cap of %.3g for a sparse factorization: not analysed\n", 0.5 * (double)aptr[m], 100.0 * (double)aptr[m] / ((double)m * m), (double)cap);
        free(deg); free(nbr); free(aptr); free(adj);
        return NULL;
    }
    /* the factor cannot win once its flops exceed dense / (2 x the best per-flop rate) */
    /* 4.20: the wide supernodes (dense cliques) now factor at about the dense rate, also in
     * tiny-pivot mode (blocked), so they are charged wide x the dense rate instead of 3x */
    static double wide = -1, spfrac = -1;
    if (wide < 0) { const char *e = getenv("BRISK_SPWIDE"); wide = e ? atof(e) : 1.3; e = getenv("BRISK_SPFRAC"); spfrac = e ? atof(e) : 0.75; }
    if (par->sparse_schur < 0) schol_set_flopcap((double)m * m * m / 3.0 * spfrac / wide);
    const double pat_lower = 0.5 * (double)aptr[m];
    /* 17 October 2026: the presolve's estimate may have analysed this very pattern (the symmetry search's budget,
     * the chordal decision); its analysis is taken when it is the one these caps give (BRISK_ICTAKE=0 off) */
    static int ictake = -1; if (ictake < 0) { const char *e = getenv("BRISK_ICTAKE"); ictake = !(e && atoi(e) == 0); }
    /* 18 October 2026: a chordal Schur pattern can take the perfect elimination ordering (schol_set_peo; BRISK_PEO=1; off by default since 19 October) */
    static int peo_on = -1; if (peo_on < 0) { const char *e = getenv("BRISK_PEO"); peo_on = e && atoi(e) != 0; }   /* 19 October 2026: off by default (BRISK_PEO=1 on): two statuses lost on ill-conditioned chordal files for 3 % on broyden25 */
    const int peo_prev = schol_set_peo(peo_on);
    SChol *SC = ictake ? chordal_take_estimate(m, deg, nbr, cap, par->sparse_schur < 0 ? (double)m * m * m / 3.0 * spfrac / wide : 0) : NULL;
    if (SC && (verbose > 1 || getenv("BRISK_ROUTEDBG"))) printf("Schur pattern: the analysis of the presolve's estimate taken\n");
    if (!SC) SC = schol_analyze_adj(m, deg, nbr, cap);
    schol_set_peo(peo_prev);
    if (SC && schol_peo_used() && (verbose > 1 || getenv("BRISK_ROUTEDBG"))) printf("Schur pattern: chordal, perfect elimination ordering (no fill)\n");
    schol_set_flopcap(0);
    free(deg); free(nbr); free(aptr); free(adj);
    if (!SC) {
        if (verbose > 1 || getenv("BRISK_ROUTEDBG")) printf("Schur pattern: %.3g entries in the lower triangle (%.0f %%): the analysis of a sparse factorization stopped at its fill or flop cap\n", pat_lower, 200.0 * pat_lower / ((double)m * m));
        return NULL;
    }
    double dense = par->c_blas * (double)m * m * m / 3.0;
    /* wide supernodes (the dense cliques of SOS/moment constraints) run at BLAS-3 speed:
     * charge them 1.3x the dense rate for the scatter and the panel overheads */
    double sp = schol_cost(SC, par->c_sparse, wide * par->c_blas);
    if (par->sparse_schur < 0 && sp > spfrac * dense) {
        if (verbose)
            printf("Schur pattern: sparse Cholesky not used (%.1e vs dense %.1e cost units)\n", sp, dense);
        schol_free(SC);
        return NULL;
    }
    if (verbose)
        printf("Schur pattern: sparse Cholesky, %zu nonzeros in L, %.1e flops (dense would be %.1e), %d supernodes\n",
               schol_nnz(SC), schol_flops(SC), (double)m * m * m / 3.0, schol_nsuper(SC));
    return SC;
}

static Envelope *envelope_build(const Problem *P, const Params *par, int verbose) {
    const int m = P->m;
    if (m < 300 || par->sparse_schur == 0) return NULL;
    /* cliques */
    int ncl = 0;
    long tot = 0, maxcl = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_LP) { ncl++; tot += B->ncon; if (B->ncon > maxcl) maxcl = B->ncon; }
        else for (int p = 0; p < B->n; p++) {
            int c = B->lp_ptr[p + 1] - B->lp_ptr[p];
            ncl++; tot += c; if (c > maxcl) maxcl = c;
        }
    }
    if (par->sparse_schur < 0 && maxcl > 0.3 * m) return NULL;
    int *cptr = malloc(sizeof(int) * (ncl + 1));
    int *cidx = malloc(sizeof(int) * (tot ? tot : 1));
    {
        int c = 0; long pos = 0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            if (B->type != BLK_LP) {
                cptr[c++] = (int)pos;
                for (int t = 0; t < B->ncon; t++) cidx[pos++] = B->con[t];
            } else for (int p = 0; p < B->n; p++) {
                cptr[c++] = (int)pos;
                for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) cidx[pos++] = B->lp_con[a];
            }
        }
        cptr[ncl] = (int)pos;
    }
    /* node -> cliques */
    int *nptr = calloc(m + 1, sizeof(int));
    for (long q = 0; q < tot; q++) nptr[cidx[q] + 1]++;
    for (int i = 0; i < m; i++) nptr[i + 1] += nptr[i];
    int *ncl_of = malloc(sizeof(int) * (tot ? tot : 1));
    {
        int *fill = malloc(sizeof(int) * m);
        memcpy(fill, nptr, sizeof(int) * m);
        for (int c = 0; c < ncl; c++)
            for (int q = cptr[c]; q < cptr[c + 1]; q++) ncl_of[fill[cidx[q]]++] = c;
        free(fill);
    }
    /* adjacency (deduplicated with a marker) */
    int *mark = malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) mark[i] = -1;
    long *aptr = calloc(m + 1, sizeof(long));
    long adj_tot = 0;
    for (int i = 0; i < m; i++) {
        long d = 0;
        for (int q = nptr[i]; q < nptr[i + 1]; q++) {
            int c = ncl_of[q];
            for (int r = cptr[c]; r < cptr[c + 1]; r++) {
                int j = cidx[r];
                if (j != i && mark[j] != i) { mark[j] = i; d++; }
            }
        }
        aptr[i + 1] = aptr[i] + d;
        adj_tot += d;
        if (adj_tot > 400000000L) { free(cptr); free(cidx); free(nptr); free(ncl_of); free(mark); free(aptr); return NULL; }
    }
    int *adj = malloc(sizeof(int) * (adj_tot ? adj_tot : 1));
    for (int i = 0; i < m; i++) mark[i] = -1;
    for (int i = 0; i < m; i++) {
        long w = aptr[i];
        for (int q = nptr[i]; q < nptr[i + 1]; q++) {
            int c = ncl_of[q];
            for (int r = cptr[c]; r < cptr[c + 1]; r++) {
                int j = cidx[r];
                if (j != i && mark[j] != i) { mark[j] = i; adj[w++] = j; }
            }
        }
    }
    free(cptr); free(cidx); free(nptr); free(ncl_of);
    if (par->sparse_schur < 0 && (double)adj_tot > 0.3 * (double)m * m) { free(mark); free(aptr); free(adj); return NULL; }
    /* reverse Cuthill-McKee */
    int *deg = malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) deg[i] = (int)(aptr[i + 1] - aptr[i]);
    int *order = malloc(sizeof(int) * m);
    int *level = malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) mark[i] = 0;
    int done = 0;
    deg_ctx = deg;
    /* nodes by increasing degree: next unvisited start node in amortized O(1) */
    int *bydeg = malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) bydeg[i] = i;
    qsort(bydeg, m, sizeof(int), cmp_by_deg);
    for (int i = 0; i < m; i++) level[i] = -1;
    int cursor = 0;
    while (done < m) {
        while (mark[bydeg[cursor]]) cursor++;
        int st = bydeg[cursor];
        /* pseudo-peripheral refinement: two BFS sweeps, only touching the component */
        for (int pass = 0; pass < 2; pass++) {
            int qh = 0, qt = 0, last = st;
            level[st] = 0; order[done + qt++] = st;
            while (qh < qt) {
                int v = order[done + qh++];
                for (long e = aptr[v]; e < aptr[v + 1]; e++) {
                    int w = adj[e];
                    if (!mark[w] && level[w] < 0) { level[w] = level[v] + 1; order[done + qt++] = w; }
                }
            }
            int lmax = level[order[done + qt - 1]];
            int best = -1;
            for (int q = 0; q < qt; q++) {
                int v = order[done + q];
                if (level[v] == lmax && (best < 0 || deg[v] < deg[best])) best = v;
                level[v] = -1;                          /* reset only what was touched */
            }
            last = best;
            if (last == st) break;
            st = last;
        }
        /* Cuthill-McKee BFS from st */
        int qh = done, qt = done;
        order[qt++] = st; mark[st] = 1;
        while (qh < qt) {
            int v = order[qh++];
            int first = qt;
            for (long e = aptr[v]; e < aptr[v + 1]; e++) {
                int w = adj[e];
                if (!mark[w]) { mark[w] = 1; order[qt++] = w; }
            }
            if (qt - first > 1) qsort(order + first, qt - first, sizeof(int), cmp_by_deg);
        }
        done = qt;
    }
    free(bydeg);
    Envelope *E = calloc(1, sizeof(Envelope));
    E->m = m;
    E->perm = malloc(sizeof(int) * m);
    E->iperm = malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) { E->perm[i] = order[m - 1 - i]; E->iperm[E->perm[i]] = i; }
    E->fst = malloc(sizeof(int) * m);
    E->rptr = malloc(sizeof(long) * (m + 1));
    double cost = 0;
    E->rptr[0] = 0;
    for (int i = 0; i < m; i++) {
        int v = E->perm[i], f = i;
        for (long e = aptr[v]; e < aptr[v + 1]; e++) { int pj = E->iperm[adj[e]]; if (pj < f) f = pj; }
        E->fst[i] = f;
        E->rptr[i + 1] = E->rptr[i] + (i - f + 1);
        cost += 0.5 * (double)(i - f) * (i - f);
    }
    E->nnz = E->rptr[m];
    free(order); free(level); free(deg); free(mark); free(aptr); free(adj);
    double dense_cost = par->c_blas * (double)m * m * m / 3.0;
    int use = par->sparse_schur > 0 || cost * par->c_sparse < 0.5 * dense_cost;
    if (verbose)
        printf("Schur pattern: %.1f%% dense, RCM profile %.1f%%, envelope Cholesky %s (%.2g vs %.2g)\n",
               100.0 * (double)adj_tot / ((double)m * m), 200.0 * (double)E->nnz / ((double)m * (m + 1)),
               use ? "selected" : "rejected", cost * par->c_sparse, dense_cost);
    if (!use) { free(E->perm); free(E->iperm); free(E->fst); free(E->rptr); free(E); return NULL; }
    E->val = amalloc(sizeof(double) * E->nnz);
    E->L = amalloc(sizeof(double) * E->nnz);
    return E;
}

static void envelope_free(Envelope *E) {
    if (!E) return;
    free(E->perm); free(E->iperm); free(E->fst); free(E->rptr); free(E->val); free(E->L); free(E);
}

/* in-place Cholesky of the (scaled) envelope matrix in E->L; returns 0 or failing row+1 */
static int env_cholesky(Envelope *E) {
    const int m = E->m;
    const double *L = E->L;
    for (int i = 0; i < m; i++) {
        const int fi = E->fst[i];
        double *Li = E->L + E->rptr[i] - fi;
        for (int j = fi; j < i; j++) {
            const int fj = E->fst[j];
            const double *Lj = L + E->rptr[j] - fj;
            const int k0 = fi > fj ? fi : fj;
            double sum = Li[j];
            for (int k = k0; k < j; k++) sum -= Li[k] * Lj[k];
            Li[j] = sum / Lj[j];
        }
        double d = Li[i];
        for (int k = fi; k < i; k++) d -= Li[k] * Li[k];
        if (!(d > 0)) return i + 1;
        Li[i] = sqrt(d);
    }
    return 0;
}

/* solve L L' x = b in permuted order, in place */
static void env_solve(const Envelope *E, double *x) {
    const int m = E->m;
    for (int i = 0; i < m; i++) {
        const int fi = E->fst[i];
        const double *Li = E->L + E->rptr[i] - fi;
        double s = x[i];
        for (int k = fi; k < i; k++) s -= Li[k] * x[k];
        x[i] = s / Li[i];
    }
    for (int i = m - 1; i >= 0; i--) {
        const int fi = E->fst[i];
        const double *Li = E->L + E->rptr[i] - fi;
        double xi = x[i] / Li[i];
        x[i] = xi;
        for (int k = fi; k < i; k++) x[k] -= Li[k] * xi;
    }
}

/* y = alpha * A x + beta * y with A the assembled envelope matrix (original order) */
static void env_mv(const Envelope *E, double alpha, const double *x, double beta, double *y, double *tx, double *ty) {
    const int m = E->m;
    for (int i = 0; i < m; i++) { tx[i] = x[E->perm[i]]; ty[i] = 0; }
    for (int i = 0; i < m; i++) {
        const int fi = E->fst[i];
        const double *Ai = E->val + E->rptr[i] - fi;
        const double xi = tx[i];
        double s = Ai[i] * xi;
        for (int j = fi; j < i; j++) { s += Ai[j] * tx[j]; ty[j] += Ai[j] * xi; }
        ty[i] += s;
    }
    for (int i = 0; i < m; i++) {
        size_t o = E->perm[i];
        y[o] = alpha * ty[i] + (beta == 0 ? 0.0 : beta * y[o]);
    }
}

/* ------------------------------------------------------------------------ */
/* Schur factorization: equilibrate, Cholesky, regularize on failure,
 * iterative refinement against the unregularized matrix.                     */
typedef struct {
    int m;
    double *M, *Mc, *D, *r, *t, *p, *q, *xb;
    double reg;
    int nreg;
    int pcg_iters, nsolve;
    int tp_sticky;        /* > 0: start the next sparse factorization in tiny-pivot mode */
    double ok_rel;        /* 4.22: best first-check residual seen on this factor (loose solves skip their check) */
    int **loc_t; double **lbuf_t; int nthr_asm; size_t lbuf_n;   /* 4.24: per-thread assembly state */
    double *asm_vals; size_t asm_cap;
    int trust;            /* the current exact factor passed its first residual check: later
                           * full-accuracy solves with it skip the check (sparse path) */
    double worst_rel;     /* worst final relative residual seen */
    int allow_float;      /* mixed precision permitted */
    int is_float;         /* current factor is single precision (stored in Mf) */
    /* 4.26: one m x m array for M and its factor: the factor overwrites the lower triangle,
     * M is kept as its strict upper triangle (transposed) plus the diagonal Md (was: two
     * m x m arrays M and Mc; the single-precision factor Mf is allocated on first use) */
    double *Md; float *Mf;
    int mT_ok;            /* the upper copy and Md hold the assembled M */
    int low_M;            /* the lower triangle still holds M (not yet overwritten) */
    int want_double;      /* single precision no longer accurate enough */
    int nfloat;           /* factorizations done in single precision */
    float *fw;
    double t_fact, t_solve;
    CliqueTab **bpos;     /* per-block scatter offsets into SC (see g_bpos), or NULL */
    size_t **bsoff; int **bsidx, *bsn; double *lbuf;   /* sorted flush lists, buffer */
    int *loc;
    const Block *blk0;
    int nbk;
    Envelope *E;          /* sparse (envelope) Schur complement, or NULL */
    SChol *SC;            /* fill-reducing sparse Cholesky, or NULL */
    double *Masm;         /* dense assembly buffer for SC on dense-ish patterns, or NULL */
    double pivtol;        /* smallest acceptable squared pivot of the equilibrated factor */
    double reg_ill;       /* regularization used when the factor is ill-conditioned */
    double min_pivot;
    double *e1, *e2, *e3;
    int use_hint, reg_hint;   /* start at the regularization the previous factorization needed */
    /* bordered split pairs (see bord_factor): M = M_K + Af D Af' is solved through the
     * quasi-definite K = [M_K Af; Af' -D^{-1}], never formed (bF: the FreeVars, bP) */
    void *bF;
    const void *bP;
    double *bE;               /* D^{-1} of each pair */
    double *ya;               /* 4.27: |M||x| of the residual check (sparse path) */
    double *e3m;              /* the two products of schur_mv_multi's paired pass (sparse path) */
    int mv_ok;                /* 5.12: products with M from the operator snapshot mvf (dense path), not from the stored M */
    void *mvf;
    int qr_fact;              /* 5.16 (experiment): the dense factor is R' from a QR of the Gram vectors (hsd_qr_factor): solves return without the residual check */
    double *qrG; size_t qrN;  /* ... and the Gram vectors (N x m, equilibrated) kept for the refinement with double-double residuals (BRISK_HSDQRREF passes) */
} Schur;

long g_sc_cnt[8]; long g_pcgh[12];   /* debug (BRISK_SOLDBG): single trusted/ok/pcg, multi trusted/ok/pcg, bord, mv */
static void bord_mv(Schur *S, const double *x, double *y);
static double bord_solve(Schur *S, const double *rhs, double *x, double tol_rel);
static int g_mv2off = -1;   /* BRISK_MV2=0: schur_mv_multi without the paired pass */
static int g_looseskip = 0; static long g_multiskip = 0;   /* 4.23: loose solves on a factor already verified far below their tolerance skip the residual check */
/* wr += R xc and wc += R' xr in one pass over R (nr x nc, column-major, leading dimension ld):
 * the two products of an off-diagonal rectangle of a symmetric matrix, R read once. Eight
 * columns at a time (four for the rest); the order of every sum is fixed by the code, not by the
 * threads. On one core as fast as dsymv on the same entries when the rectangles are tall (the
 * dots run down whole columns), which is why the parts below are column panels. */
static void rect_mv2_4(int nr, int nc, const double *R, int ld, const double *restrict xr, const double *restrict xc,
                       double *restrict wr, double *restrict wc) {
    int j = 0;
    for (; j + 4 <= nc; j += 4) {
        const double *c0 = R + (size_t)j * ld, *c1 = c0 + ld, *c2 = c1 + ld, *c3 = c2 + ld;
        const double x0 = xc[j], x1 = xc[j + 1], x2 = xc[j + 2], x3 = xc[j + 3];
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        #pragma omp simd reduction(+:t0,t1,t2,t3)
        for (int i = 0; i < nr; i++) {
            const double a0 = c0[i], a1 = c1[i], a2 = c2[i], a3 = c3[i], xi = xr[i];
            wr[i] += (a0 * x0 + a1 * x1) + (a2 * x2 + a3 * x3);
            t0 += a0 * xi; t1 += a1 * xi; t2 += a2 * xi; t3 += a3 * xi;
        }
        wc[j] += t0; wc[j + 1] += t1; wc[j + 2] += t2; wc[j + 3] += t3;
    }
    for (; j < nc; j++) {
        const double *c0 = R + (size_t)j * ld, x0 = xc[j];
        double t0 = 0;
        #pragma omp simd reduction(+:t0)
        for (int i = 0; i < nr; i++) { wr[i] += c0[i] * x0; t0 += c0[i] * xr[i]; }
        wc[j] += t0;
    }
}
static void rect_mv2(int nr, int nc, const double *R, int ld, const double *restrict xr, const double *restrict xc,
                     double *restrict wr, double *restrict wc) {
    int j = 0;
    for (; j + 8 <= nc; j += 8) {
        const double *c0 = R + (size_t)j * ld;
        const size_t l = (size_t)ld;
        const double x0 = xc[j], x1 = xc[j + 1], x2 = xc[j + 2], x3 = xc[j + 3], x4 = xc[j + 4], x5 = xc[j + 5], x6 = xc[j + 6], x7 = xc[j + 7];
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        #pragma omp simd reduction(+:t0,t1,t2,t3,t4,t5,t6,t7)
        for (int i = 0; i < nr; i++) {
            const double a0 = c0[i], a1 = c0[i + l], a2 = c0[i + 2 * l], a3 = c0[i + 3 * l];
            const double a4 = c0[i + 4 * l], a5 = c0[i + 5 * l], a6 = c0[i + 6 * l], a7 = c0[i + 7 * l], xi = xr[i];
            wr[i] += ((a0 * x0 + a1 * x1) + (a2 * x2 + a3 * x3)) + ((a4 * x4 + a5 * x5) + (a6 * x6 + a7 * x7));
            t0 += a0 * xi; t1 += a1 * xi; t2 += a2 * xi; t3 += a3 * xi; t4 += a4 * xi; t5 += a5 * xi; t6 += a6 * xi; t7 += a7 * xi;
        }
        wc[j] += t0; wc[j + 1] += t1; wc[j + 2] += t2; wc[j + 3] += t3; wc[j + 4] += t4; wc[j + 5] += t5; wc[j + 6] += t6; wc[j + 7] += t7;
    }
    if (j < nc) rect_mv2_4(nr, nc - j, R + (size_t)j * ld, ld, xr, xc + j, wr, wc + j);
}
/* 5.7: y = alpha M x + beta y for a symmetric M stored in one triangle, on the OpenMP threads: the BLAS
 * is single-threaded here and dsymv was the larger part of the solves with a dense Schur complement
 * (m = 3,739: 325 products, 1.2 of 2.4 s).
 * 5.11: the same bits for any thread count, and no slower than dsymv on one thread. The columns
 * are cut into P panels of equal area, P from m alone (about m/128, 4 to 64; in 5.7 the parts
 * were rows and P the thread count). Panel b = columns [c0, c1) owns its diagonal block (dsymv)
 * and the rectangle that joins it to the other indices - below it for the lower triangle, above
 * it for the upper one - whose two products it adds in one pass (rect_mv2) into a vector of its
 * own; the vectors are summed in panel order. One thread runs the panels in turn through the
 * same vectors: the same additions in the same order, at the same addresses (5.15: through one
 * vector until then; Accelerate rounds a dsymv differently by where its vectors lie, modulo up
 * to 256 bytes, so on macOS one thread and several gave other bits). Measured against one dsymv (m = 1,024 ... 12,000,
 * bench/threads_5.11/symv_bench.c): 0.91-1.03 times its time on one thread; 2.0-2.3 times faster
 * than the 5.7 version at four threads. Below m = 1,024 one dsymv. BRISK_SYMVSER=1: one dsymv
 * always (the result of 5.6 and earlier). */
#define SYMV_PMAX 64
static double *g_symv_w = NULL; static size_t g_symv_wn = 0;
static void par_symv(char uplo, int m, double alpha, const double *M, const double *x, double beta, double *y) {
    if (m < 1024 || ENV_ON("BRISK_SYMVSER")) { BL(dsymv_)(uplo == 'L' ? "L" : "U", &m, &alpha, M, &m, x, &IONE, &beta, y, &IONE); return; }
    const int low = uplo == 'L';
    int P = m / 128; if (P < 4) P = 4; if (P > SYMV_PMAX) P = SYMV_PMAX;
    int cut[SYMV_PMAX + 1];             /* panel b: columns [cut[b], cut[b + 1]); equal areas of the stored triangle */
    for (int b = 0; b <= P; b++) cut[b] = (int)(low ? m - m * sqrt(1.0 - (double)b / P) + 0.5 : m * sqrt((double)b / P) + 0.5);
    cut[0] = 0; cut[P] = m;
    const int inp = in_par(), T0 = inp ? 1 : nthreads(), T = T0 < P ? T0 : P;
    const size_t need = sizeof(double) * (size_t)m * (P + 1);
    double *buf;                                    /* (inside a parallel region: a buffer of its own) */
    if (inp) buf = amalloc(need);
    else { if (need > g_symv_wn) { free(g_symv_w); g_symv_w = amalloc(need); g_symv_wn = need; } buf = g_symv_w; }
    double *acc = buf, *W = buf + m;                /* acc: the sum over the panels; W: the panels' vectors */
    const double one = 1.0;
    /* panel b touches [lo, hi): lower [c0, m), upper [0, c1) */
    #define SYMV_LO(b) (low ? cut[b] : 0)
    #define SYMV_HI(b) (low ? m : cut[(b) + 1])
    #define SYMV_PART(b, w) do {                                                                  \
        const int c0 = cut[b], c1 = cut[(b) + 1], nc = c1 - c0;                                    \
        memset((w) + SYMV_LO(b), 0, sizeof(double) * (size_t)(SYMV_HI(b) - SYMV_LO(b)));          \
        if (nc > 0) {                                                                              \
            BL(dsymv_)(low ? "L" : "U", &nc, &one, M + c0 + (size_t)c0 * m, &m, x + c0, &IONE, &one, (w) + c0, &IONE); \
            if (low && c1 < m) rect_mv2(m - c1, nc, M + c1 + (size_t)c0 * m, m, x + c1, x + c0, (w) + c1, (w) + c0); \
            if (!low && c0 > 0) rect_mv2(c0, nc, M + (size_t)c0 * m, m, x, x + c0, (w), (w) + c0); \
        } } while (0)
    if (!inp) blas_serial_begin();
    if (T <= 1) {
        for (int b = 0; b < P; b++) { double *w = W + (size_t)b * m; SYMV_PART(b, w); }
        for (int i = 0; i < m; i++) {
            double t = 0;
            for (int b = 0; b < P; b++) if (i >= SYMV_LO(b) && i < SYMV_HI(b)) t += W[i + (size_t)b * m];
            acc[i] = t;
        }
    } else {
        #pragma omp parallel num_threads(T)
        {
            #pragma omp for schedule(dynamic, 1)
            for (int b = 0; b < P; b++) { double *w = W + (size_t)b * m; SYMV_PART(b, w); }
            /* the sum in panel order, parallel over the indices */
            #pragma omp for schedule(static)
            for (int i = 0; i < m; i++) {
                double t = 0;
                for (int b = 0; b < P; b++) if (i >= SYMV_LO(b) && i < SYMV_HI(b)) t += W[i + (size_t)b * m];
                acc[i] = t;
            }
        }
    }
    if (!inp) blas_serial_end();
    #undef SYMV_PART
    #undef SYMV_LO
    #undef SYMV_HI
    for (int i = 0; i < m; i++) y[i] = alpha * acc[i] + (beta == 0 ? 0.0 : beta * y[i]);
    if (inp) free(buf);
}
static double g_solprof_p[4];
/* 5.12: the products with the dense Schur matrix without reading it. M x = sum_b A_b(X_b A_b*(x) Zi_b)
 * (+ A diag(x z^{-1}) A' for linear blocks) costs about 4 sum n_b^3 flops; reading the stored M costs
 * 4 m^2 bytes. With many constraints and small blocks (moment relaxations with m = 15,000 and blocks
 * of 276: 2 ms against 90 ms) the PCG steps of the Schur solves then cost the triangular solves
 * alone. The snapshot holds the scaling matrices the assembly used (X = Zi = W for NT), taken right
 * after the assembly; any new assembly (schur_zero) turns it off. */
typedef struct {
    const Problem *P; BS *S;
    int nb, m, nmax;
    double **Xs, **Zs;        /* per block: X and Zi of the assembly (Zs == Xs for NT); LP: Xs = x z^{-1} */
    double *F, *T, *w;
} MvFree;
static void mvfree_apply(MvFree *c, const double *x, double *y) {
    const Problem *P = c->P;
    memset(y, 0, sizeof(double) * c->m);
    for (int k = 0; k < c->nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &c->S[k];
        const int n = B->n;
        if (B->type == BLK_LP) {
            blk_ATy(B, s, x, c->F);
            for (int p = 0; p < n; p++) c->F[p] *= c->Xs[k][p];
            blk_Aop_serial(B, s, c->F, y, 1.0, B->con);
            continue;
        }
        blk_ATy(B, s, x, c->F);
        pdgemm("N", "N", &n, &n, &n, &DONE, c->Xs[k], &n, c->F, &n, &DZERO, c->T, &n);
        pdgemm("N", "N", &n, &n, &n, &DONE, c->T, &n, c->Zs[k], &n, &DZERO, c->F, &n);
        blk_Aop_serial(B, s, c->F, y, 1.0, B->con);     /* <A_i, D> for a nonsymmetric D */
    }
}
static void mvfree_free(Schur *S) {
    MvFree *c = S->mvf;
    if (!c) return;
    for (int k = 0; k < c->nb; k++) { if (c->Zs[k] != c->Xs[k]) free(c->Zs[k]); free(c->Xs[k]); }
    free(c->Xs); free(c->Zs); free(c->F); free(c->T); free(c->w); free(c);
    S->mvf = NULL; S->mv_ok = 0;
}
/* the snapshot after an assembly; nt[k] != 0: block k was assembled in the NT view (X = Zi) */
static void mvfree_take(Schur *S, const Problem *P, BS *BSv, double **ntv) {
    MvFree *c = S->mvf;
    if (!c) {
        c = calloc(1, sizeof(MvFree));
        c->P = P; c->S = BSv; c->nb = P->nblk; c->m = S->m;
        c->Xs = calloc(P->nblk, sizeof(double *)); c->Zs = calloc(P->nblk, sizeof(double *));
        size_t need = 1;           /* n^2 for a semidefinite block, n for a linear one (butcher: 22,512 LP variables) */
        for (int k = 0; k < P->nblk; k++) {
            const size_t nk = (size_t)P->blk[k].n, len = P->blk[k].type == BLK_LP ? nk : nk * nk;
            if (len > need) need = len;
            if (P->blk[k].type != BLK_LP && P->blk[k].n > c->nmax) c->nmax = P->blk[k].n;
        }
        c->F = amalloc(sizeof(double) * need); c->T = amalloc(sizeof(double) * need);
        c->w = malloc(sizeof(double) * S->m);
        S->mvf = c;
    }
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        const BS *s = &BSv[k];
        const size_t len = B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
        const int nt = ntv && ntv[k];
        if (!c->Xs[k]) c->Xs[k] = malloc(sizeof(double) * len);
        if (B->type == BLK_LP) { for (size_t p = 0; p < len; p++) c->Xs[k][p] = s->X[p] * s->Zi[p]; c->Zs[k] = c->Xs[k]; continue; }
        if (nt) { if (c->Zs[k] && c->Zs[k] != c->Xs[k]) { free(c->Zs[k]); } c->Zs[k] = c->Xs[k]; memcpy(c->Xs[k], s->Zi, sizeof(double) * len); }
        else {
            if (!c->Zs[k] || c->Zs[k] == c->Xs[k]) c->Zs[k] = malloc(sizeof(double) * len);
            memcpy(c->Xs[k], s->X, sizeof(double) * len); memcpy(c->Zs[k], s->Zi, sizeof(double) * len);
        }
    }
    S->mv_ok = 1;
}
static void schur_mv_(Schur *S, double alpha, const double *x, double beta, double *y);
static void schur_mv(Schur *S, double alpha, const double *x, double beta, double *y) { const double t0 = wtime(); schur_mv_(S, alpha, x, beta, y); g_solprof_p[2] += wtime() - t0; g_solprof_p[3] += 1; }
static void schur_mv_(Schur *S, double alpha, const double *x, double beta, double *y) {
    const int m = S->m;
    if (S->bF) {
        bord_mv(S, x, S->e2);
        for (int i = 0; i < m; i++) y[i] = alpha * S->e2[i] + (beta == 0 ? 0.0 : beta * y[i]);
    } else if (S->SC) {
        g_sc_cnt[7]++;
        schol_mv(S->SC, x, S->e2);
        for (int i = 0; i < m; i++) y[i] = alpha * S->e2[i] + (beta == 0 ? 0.0 : beta * y[i]);
    } else if (S->E) env_mv(S->E, alpha, x, beta, y, S->e2, S->e3);
    else if (S->mv_ok && S->is_float) {
        /* only with the single-precision factor: a double factor of a nearly singular M (the
         * last iterations of degenerate problems) preconditions the M it was made from, not the
         * exact operator, and PCG stalled (control7_s: 8 -> 24 iterations) */
        double *w = ((MvFree *)S->mvf)->w;
        mvfree_apply(S->mvf, x, w);
        for (int i = 0; i < m; i++) y[i] = alpha * w[i] + (beta == 0 ? 0.0 : beta * y[i]);
    }
    else if (!S->mT_ok || S->low_M) par_symv('L', m, alpha, S->M, x, beta, y);
    else {
        /* M from the strict upper copy; the array diagonal is the factor's, so correct it */
        par_symv('U', m, alpha, S->M, x, beta, y);
        for (int i = 0; i < m; i++) y[i] += alpha * (S->Md[i] - S->M[i + (size_t)i * m]) * x[i];
    }
}
static void schur_zero(Schur *S) {
    S->mv_ok = 0;
    if (S->SC && S->Masm) memset(S->Masm, 0, sizeof(double) * (size_t)S->m * S->m);
    else if (S->SC) schol_zero(S->SC);
    else if (S->E) memset(S->E->val, 0, sizeof(double) * S->E->nnz);
    else { pzero((size_t)S->m * S->m, S->M); S->mT_ok = 0; S->low_M = 1; }     /* 5.12: on the threads */
}

/* Schur assembly: the target matrix and the global write routing */
static double *schur_asm_begin(Schur *S) {
    schur_zero(S);
    g_env = S->E;
    g_schol = S->Masm ? NULL : S->SC;
    if (g_schol && S->bpos) { g_bpos = S->bpos; g_blk0 = S->blk0; g_nbk = S->nbk; g_loc = S->loc; g_pm = schol_values(S->SC);
                              g_bsoff = S->bsoff; g_bsidx = S->bsidx; g_bsn = S->bsn; g_lbuf_all = S->lbuf; }
    return S->Masm ? S->Masm : S->M;
}
static void schur_asm_end(Schur *S) {
    g_env = NULL; g_schol = NULL; g_bpos = NULL; g_pos = NULL; g_bsoff = NULL; g_lbuf_all = NULL;
    if (S->SC && S->Masm) schol_gather_dense(S->SC, S->Masm);
}
/* 4.24: Schur assembly over all blocks, parallel and bitwise equal to the serial loop.
 * Runs of buffered SDP blocks (their pairs have sorted panel offsets) are assembled in
 * parallel into value lists, then added to the panel storage in parallel by disjoint
 * offset ranges, each entry receiving its contributions in block order. Other blocks
 * (LP, unbuffered) run serially at their place in the order. */
static BS *nt_view(BS *s, BS *tmp, int nt);
static double g_asm_budget = -1; static double g_asmp[4];
static void schur_assemble_all(const Problem *P, BS *S, Schur *Sc, double **Zv, int m, double *Ma) {
    const int nb = P->nblk;
    int nt = nthreads();
    const int bufok = g_schol && g_bpos && g_bsoff && g_lbuf_all && !in_par();
    if (g_asm_budget < 0) { const char *e = getenv("BRISK_ASMBUDGET"); g_asm_budget = e ? atof(e) : 4e6; }
    if (nt <= 1 || !bufok || getenv("BRISK_ASMSERIAL")) {
        static int blkt = -1; static double *tb = NULL; static int ntb = 0;
        if (blkt < 0) blkt = getenv("BRISK_BLKT") != NULL;
        for (int k = 0; k < nb; k++) {
            BS vtmp;
            const double tk0 = blkt ? wtime() : 0;
            if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], &S[k], m, Ma);
            else if (P->blk[k].type == BLK_SOC) schur_soc(&P->blk[k], &S[k], m, Ma);
            else schur_sdp(&P->blk[k], nt_view(&S[k], &vtmp, Zv[k] != NULL), m, Ma);
            if (blkt && k < 8) {
                if (!tb) { tb = calloc(8, sizeof(double)); ntb = nb < 8 ? nb : 8; }
                tb[k] += wtime() - tk0;
            }
        }
        if (blkt && tb) { fprintf(stderr, "   [assembly by block (first %d), so far:", ntb); for (int k = 0; k < ntb; k++) fprintf(stderr, " n%d %.3fs", P->blk[k].n, tb[k]); fprintf(stderr, "]\n"); }
        return;
    }
    if (Sc->nthr_asm < nt) {
        for (int t = 0; t < Sc->nthr_asm; t++) { free(Sc->loc_t[t]); free(Sc->lbuf_t[t]); }
        free(Sc->loc_t); free(Sc->lbuf_t);
        Sc->loc_t = malloc(sizeof(int *) * nt); Sc->lbuf_t = malloc(sizeof(double *) * nt);
        for (int t = 0; t < nt; t++) {
            Sc->loc_t[t] = malloc(sizeof(int) * (m + 1)); for (int i = 0; i < m; i++) Sc->loc_t[t][i] = -1;
            Sc->lbuf_t[t] = calloc(Sc->lbuf_n, sizeof(double));
        }
        Sc->nthr_asm = nt;
    }
    size_t *voff = malloc(sizeof(size_t) * (nb + 1));
    double *pm = g_pm;
    const int dbg = g_asmdbg; g_asmdbg = 0;
    int k = 0;
    while (k < nb) {
        const Block *B = &P->blk[k];
        const int buf = B->type == BLK_SDP && g_bpos[k] && g_bsoff[k];
        if (!buf) {
            BS vtmp;
            const double tq2 = wtime();
            if (B->type == BLK_LP) schur_lp(B, &S[k], m, Ma);
            else if (B->type == BLK_SOC) schur_soc(B, &S[k], m, Ma);
            else schur_sdp(B, nt_view(&S[k], &vtmp, Zv[k] != NULL), m, Ma);
            g_asmp[2] += wtime() - tq2;
            k++; continue;
        }
        int k1 = k; size_t tot = 0;
        while (k1 < nb && P->blk[k1].type == BLK_SDP && g_bpos[k1] && g_bsoff[k1] && (k1 == k || tot + (size_t)g_bsn[k1] <= g_asm_budget)) {
            voff[k1] = tot; tot += (size_t)g_bsn[k1]; k1++;
        }
        if (tot > Sc->asm_cap) { free(Sc->asm_vals); Sc->asm_cap = tot + tot / 4; Sc->asm_vals = malloc(sizeof(double) * Sc->asm_cap); }
        double *vals = Sc->asm_vals;
        const int ka = k, kb = k1;
        double tq0 = wtime();
        #pragma omp parallel for schedule(dynamic, 4)
        for (int kk = ka; kk < kb; kk++) {
            BS vtmp;
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            schur_sdp_vals(&P->blk[kk], nt_view(&S[kk], &vtmp, Zv[kk] != NULL), m, Ma, kk, Sc->loc_t[tid], Sc->lbuf_t[tid], vals + voff[kk]);
        }
        double tq1 = wtime(); g_asmp[0] += tq1 - tq0; g_asmp[3] += 1;
        /* flush by offset ranges */
        const size_t lo0 = g_bsoff[ka][0];
        size_t hi0 = 0;
        for (int kk = ka; kk < kb; kk++) if (g_bsn[kk] > 0 && g_bsoff[kk][g_bsn[kk] - 1] + 1 > hi0) hi0 = g_bsoff[kk][g_bsn[kk] - 1] + 1;
        size_t lomin = lo0; for (int kk = ka; kk < kb; kk++) if (g_bsn[kk] > 0 && g_bsoff[kk][0] < lomin) lomin = g_bsoff[kk][0];
        const int nr = 8 * nt;
        const size_t span = hi0 > lomin ? hi0 - lomin : 0, step = span / nr + 1;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int r = 0; r < nr; r++) {
            const size_t lo = lomin + (size_t)r * step, hi = lo + step;
            for (int kk = ka; kk < kb; kk++) {
                const size_t *of = g_bsoff[kk]; const int nq = g_bsn[kk];
                if (nq == 0 || of[nq - 1] < lo || of[0] >= hi) continue;
                int a = 0, b = nq;
                while (a < b) { const int mid = (a + b) >> 1; if (of[mid] < lo) a = mid + 1; else b = mid; }
                const double *v = vals + voff[kk];
                for (int q = a; q < nq && of[q] < hi; q++) pm[of[q]] += v[q];
            }
        }
        g_asmp[1] += wtime() - tq1;
        k = k1;
    }
    g_asmdbg = dbg;
    free(voff);
}
/* the dense assembly buffer pays when the factor holds a good share of the entries
 * (with the direct scatter maps only when it is nearly dense) */
static double g_masm_frac = -1;
static void schur_asm_setup(Schur *S) {
    S->Masm = NULL;
    if (g_masm_frac < 0) { const char *e = getenv("BRISK_MASM"); g_masm_frac = e ? atof(e) : 0.5; }
    if (S->SC && S->m <= 16000 && (double)schol_nnz(S->SC) > g_masm_frac * 0.5 * (double)S->m * S->m)
        S->Masm = amalloc(sizeof(double) * (size_t)S->m * S->m);
}

static int __attribute__((unused)) cmp_offidx(const void *x, const void *y) {
    const size_t a = *(const size_t *)x, b = *(const size_t *)y;
    return a < b ? -1 : a > b ? 1 : 0;
}
/* scatter offsets of every SDP block's constraint pairs (memory capped); 5.16: a second-order cone's
 * too (schur_soc writes them on the threads), without the flush lists */
#ifdef __GLIBC__
#include <malloc.h>
#endif
static void schur_pos_setup(Schur *S, const Problem *P) {
#ifdef __GLIBC__
    /* 18 October 2026: the analysis's many small pattern lists are freed by now, but glibc keeps their pages in the
     * heap: returned to the system before the factor's storage is used (broyden25: peak 1049 -> 789 MB). */
    if (!getenv("BRISK_NOTRIM")) malloc_trim(0);
#endif
    S->bpos = NULL; S->loc = NULL; S->blk0 = P->blk; S->nbk = P->nblk;
    S->bsoff = NULL; S->bsidx = NULL; S->bsn = NULL; S->lbuf = NULL;
    if (!S->SC || S->Masm) return;
    double tot = 0;
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type != BLK_LP) tot += (double)P->blk[k].ncon * P->blk[k].ncon;
    if (tot * sizeof(size_t) > 1.5e9 * brisk_mem_scale()) return;
    /* 18 October 2026: an offset table per block (schol_clique_table: nl x the supernodes its columns fall in, never
     * more than nl^2; were nl x nl squares of 8-byte offsets, 22 bytes per entry of L on broyden25, the largest item
     * of its memory). The same offsets. */
    S->bpos = calloc(P->nblk + 1, sizeof(CliqueTab *));
    S->loc = malloc(sizeof(int) * (S->m + 1));
    for (int i = 0; i < S->m; i++) S->loc[i] = -1;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_LP || B->ncon == 0) continue;
        S->bpos[k] = schol_clique_table(S->SC, B->ncon, B->con);
    }
    /* sorted flush lists (columns and rows in factor order: offsets come out increasing) */
    S->bsoff = calloc(P->nblk + 1, sizeof(size_t *)); S->bsidx = calloc(P->nblk + 1, sizeof(int *)); S->bsn = calloc(P->nblk + 1, sizeof(int));
    int nlmax = 0;
    for (int k = 0; k < P->nblk; k++) {
        if (!S->bpos[k] || P->blk[k].type != BLK_SDP) continue;
        const int nl = P->blk[k].ncon;
        { static int lbmax = -1; if (lbmax < 0) { const char *e = getenv("BRISK_LBUFMAX"); lbmax = e ? atoi(e) : 1500; }
          if (nl > lbmax) continue; }            /* buffer nl^2: keep it small */
        if (nl > nlmax) nlmax = nl;
        const int nq = nl * (nl + 1) / 2;
        S->bsoff[k] = malloc(sizeof(size_t) * (nq + 1)); S->bsidx[k] = malloc(sizeof(int) * (nq + 1)); S->bsn[k] = nq;
        size_t *tmp = malloc(sizeof(size_t) * (size_t)nl * nl);
        schol_clique_offsets(S->SC, nl, P->blk[k].con, tmp, S->bsoff[k], S->bsidx[k]);
        free(tmp);
    }
    S->lbuf = calloc((size_t)nlmax * nlmax + 1, sizeof(double));
    S->lbuf_n = (size_t)nlmax * nlmax + 1;
}
static void schur_pos_free(Schur *S) {
    for (int t = 0; t < S->nthr_asm; t++) { free(S->loc_t[t]); free(S->lbuf_t[t]); }
    free(S->loc_t); free(S->lbuf_t); S->loc_t = NULL; S->lbuf_t = NULL; S->nthr_asm = 0;
    free(S->asm_vals); S->asm_vals = NULL; S->asm_cap = 0;
    if (S->bsoff) for (int k = 0; k < S->nbk; k++) { free(S->bsoff[k]); free(S->bsidx[k]); }
    free(S->bsoff); free(S->bsidx); free(S->bsn); free(S->lbuf);
    S->bsoff = NULL; S->bsidx = NULL; S->bsn = NULL; S->lbuf = NULL;
    if (S->bpos) for (int k = 0; k < S->nbk; k++) clique_tab_free(S->bpos[k]);
    free(S->bpos); free(S->loc);
    S->bpos = NULL; S->loc = NULL;
}

static int schur_factor_(Schur *S);
static int g_blas_T = 0;          /* BLAS threads while block-parallel loops run serial BLAS */
static int schur_factor(Schur *S) {
    S->qr_fact = 0;
    double t = wtime();
    S->trust = 0;
    S->ok_rel = INFINITY;
    if (g_blas_T > 1) BL(openblas_set_num_threads)(g_blas_T);
    int r = schur_factor_(S);
    if (g_blas_T > 1) BL(openblas_set_num_threads)(1);
    S->t_fact += wtime() - t;
    return r;
}
static int schur_factor_env(Schur *S) {
    Envelope *E = S->E;
    const int m = S->m;
    for (int i = 0; i < m; i++) {
        double d = E->val[E->rptr[i] + (i - E->fst[i])];
        S->D[E->perm[i]] = d > 0 ? 1.0 / sqrt(d) : 1.0;
    }
    double reg = 0;
    for (int attempt = 0; attempt < 10; attempt++) {
        for (int i = 0; i < m; i++) {
            const int fi = E->fst[i];
            const double di = S->D[E->perm[i]];
            const double *src = E->val + E->rptr[i] - fi;
            double *dst = E->L + E->rptr[i] - fi;
            for (int j = fi; j <= i; j++) dst[j] = src[j] * di * S->D[E->perm[j]];
            dst[i] += reg;
        }
        int info = env_cholesky(E);
        if (info == 0) {
            double pmin = 1.0;
            for (int i = 0; i < m; i++) { double p = E->L[E->rptr[i] + (i - E->fst[i])]; if (p * p < pmin) pmin = p * p; }
            if (pmin >= S->pivtol || reg >= S->reg_ill) {
                S->reg = reg; if (reg > 0) S->nreg++; S->is_float = 0;
                S->min_pivot = pmin;
                return 0;
            }
            reg = fmax(S->reg_ill, reg * 100);
            continue;
        }
        reg = (reg == 0) ? 1e-13 : reg * 100;
    }
    return 1;
}

/* 4.26: dense Schur storage. M is assembled into the lower triangle of the m x m array;
 * before the first factorization its strict lower triangle is copied (transposed, in 64 x 64
 * tiles) into the strict upper one and its diagonal into Md, so the factor can overwrite the
 * lower triangle in place: one array instead of M and Mc (m = 18k: 5.2 -> 2.6 GB). */
static void schur_dense_keep(Schur *S) {
    if (S->mT_ok) return;
    const int m = S->m;
    double *A = S->M;
    for (int i = 0; i < m; i++) S->Md[i] = A[i + (size_t)i * m];
    const int nbt = (m + 63) / 64;
    #pragma omp parallel for schedule(dynamic, 1) if (m >= 512 && !omp_in_parallel())     /* 5.12: from 512 (was > 2000; element-wise) */
    for (int jt = 0; jt < nbt; jt++) {
        const int jb = jt * 64, je = jb + 64 < m ? jb + 64 : m;
        for (int ib = jb; ib < m; ib += 64) {
            const int ie = ib + 64 < m ? ib + 64 : m;
            for (int j = jb; j < je; j++)
                for (int i = (ib > j + 1 ? ib : j + 1); i < ie; i++) A[j + (size_t)i * m] = A[i + (size_t)j * m];
        }
    }
    S->mT_ok = 1; S->low_M = 1;
}
/* the equilibrated lower triangle D M D + reg I into the array (the factor input) */
static void schur_dense_load(Schur *S, double reg) {
    const int m = S->m;
    double *A = S->M;
    const int low = S->low_M;
    #pragma omp parallel for schedule(static) if (m >= 512 && !omp_in_parallel())
    for (int j = 0; j < m; j++) {
        const double dj = S->D[j];
        double *col = A + (size_t)j * m;
        if (low) for (int i = j + 1; i < m; i++) col[i] = col[i] * S->D[i] * dj;
        else for (int i = j + 1; i < m; i++) col[i] = A[j + (size_t)i * m] * S->D[i] * dj;
        col[j] = S->Md[j] * dj * dj + reg;
    }
    S->low_M = 0;
}
/* M_ij of the dense Schur complement, whatever the lower triangle holds */
static inline double schur_dense_get(const Schur *S, int i, int j) {
    if (i < j) { int t = i; i = j; j = t; }
    if (!S->mT_ok || S->low_M) return S->M[i + (size_t)j * S->m];
    return i == j ? S->Md[i] : S->M[j + (size_t)i * S->m];
}
static int schur_factor_(Schur *S) {
    if (S->SC) {
        double reg = 0;
        static double tp = -1;
        if (tp < 0) { const char *e = getenv("BRISK_TINYPIV"); tp = e ? atof(e) : 1e-14; }
        for (int attempt = 0; attempt < 10; attempt++) {
            /* after a factorization that needed tiny pivots, the next one goes to that mode
             * directly (the failed attempt costs a whole factorization) */
            if (attempt == 0 && tp > 0 && S->tp_sticky > 0) {
                schol_set_tinypiv(S->SC, tp);
                int bad = schol_factor(S->SC, 0.0);
                const int nt = schol_ntiny(S->SC);
                schol_set_tinypiv(S->SC, 0.0);
                if (!bad) {
                    S->tp_sticky = nt > 0 ? 3 : S->tp_sticky - 1;
                    S->reg = nt > 0 ? 1e-300 : 0.0; if (nt > 0) S->nreg++; S->is_float = 0;
                    if (getenv("BRISK_TPDBG")) printf("     [tiny pivots (sticky): %d]\n", nt);
                    return 0;
                }
            }
            if (schol_factor(S->SC, reg) == 0) { S->reg = reg; if (reg > 0) S->nreg++; S->is_float = 0; return 0; }
            if (attempt == 0 && tp > 0) {
                /* degenerate pivots: drop them instead of shifting the whole matrix */
                schol_set_tinypiv(S->SC, tp);
                int bad = schol_factor(S->SC, 0.0);
                const int nt = schol_ntiny(S->SC);
                schol_set_tinypiv(S->SC, 0.0);
                if (!bad) { S->reg = 1e-300; S->nreg++; S->is_float = 0; S->tp_sticky = 3; if (getenv("BRISK_TPDBG")) printf("     [tiny pivots: %d]\n", nt); return 0; }
            }
            reg = (reg == 0) ? 1e-13 : reg * 100;
        }
        return 1;
    }
    if (S->E) return schur_factor_env(S);
    int m = S->m, info = 0;
    const double tk0 = wtime();
    /* 5.12: the single-precision factor has its own array and leaves M in the lower triangle: the
     * upper copy (0.19 s at m = 14949) only when a double-precision factorization will overwrite it */
    if (S->allow_float && !S->want_double && !S->mT_ok) { for (int i = 0; i < m; i++) S->Md[i] = S->M[i + (size_t)i * m]; }
    else schur_dense_keep(S);
    const double tk1 = wtime();
    double maxd = 0;
    for (int i = 0; i < m; i++) {
        double d = S->Md[i];
        S->D[i] = d > 0 ? 1.0 / sqrt(d) : 1.0;
        if (d > maxd) maxd = d;
    }
    if (S->allow_float && !S->want_double) {
        if (!S->Mf) S->Mf = amalloc(sizeof(float) * (size_t)m * m);
        float *F = S->Mf;
        const int low = S->low_M;
        #pragma omp parallel for schedule(dynamic, 32) if (m >= 512 && !omp_in_parallel())     /* 5.12: from 512 (was > 2000) */
        for (int j = 0; j < m; j++) {
            const double dj = S->D[j];
            const double *src = S->M + (size_t)j * m;
            float *dst = F + (size_t)j * m;
            { const double v = S->Md[j] * dj * dj; dst[j] = (fabs(v) < 1e-30) ? 0.0f : (float)v; }
            for (int i = j + 1; i < m; i++) {
                double v = (low ? src[i] : S->M[j + (size_t)i * m]) * S->D[i] * dj;
                dst[i] = (fabs(v) < 1e-30) ? 0.0f : (float)v;   /* no subnormals in the float factor */
            }
        }
        {   /* testing: BRISK_FLNOISE=seed perturbs the single-precision copy by one ulp at hashed
             * places (what a BLAS whose threaded rounding varies from run to run does to the factor) */
            static int fln = -1; static unsigned fcall = 0;
            if (fln < 0) { const char *e = getenv("BRISK_FLNOISE"); fln = e ? atoi(e) : 0; }
            if (fln) {
                fcall++;
                for (int j = 0; j < m; j++) for (int i = j; i < m; i++) {
                    unsigned h = ((unsigned)i * 2654435761u) ^ ((unsigned)j * 40503u) ^ ((unsigned)fln * 2246822519u) ^ (fcall * 3266489917u);
                    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
                    if ((h & 3) == 0) { static float mag = -1; if (mag < 0) { const char *e2 = getenv("BRISK_FLNOISE_MAG"); mag = e2 ? (float)atof(e2) : 1.0f; }
                        if (mag == 1.0f) F[i + (size_t)j * m] = nextafterf(F[i + (size_t)j * m], (h & 4) ? 1e30f : -1e30f);
                        else F[i + (size_t)j * m] *= 1.0f + ((h & 4) ? 1.0f : -1.0f) * mag * 6e-8f; }
                }
            }
        }
        const double tk2 = wtime();
        info = brisk_spotrf(m, F, m);
        const double tk3 = wtime();
        if (info == 0) {
            double pmin = 1.0;
            for (int i = 0; i < m; i++) { double p = F[i + (size_t)i * m]; if (p * p < pmin) pmin = p * p; }
            static double fl_pmin = -1; if (fl_pmin < 0) { const char *e = getenv("BRISK_FLPMIN"); fl_pmin = e ? atof(e) : 1e-6; }   /* (test) */
            if (getenv("BRISK_FACTDBG")) printf("       [spotrf m %d pmin %.2e: keep %.3fs, single copy %.3fs, factor %.3fs]\n", m, pmin, tk1 - tk0, tk2 - tk1, tk3 - tk2);
            if (pmin >= fl_pmin) { S->is_float = 1; S->reg = 0; S->nfloat++; return 0; }
        }
        S->want_double = 1;
    }
    S->is_float = 0;
    schur_dense_keep(S);
    /* 4.20 (BRISK_DTINY): tiny-pivot replacement on the dense path as on the sparse one,
     * instead of shifting the whole matrix (then PCG restores the dropped directions) */
    static int dtiny = -1;
    if (dtiny < 0) { const char *e = getenv("BRISK_DTINY"); dtiny = e ? atoi(e) : 0; }
    for (int pass = 0; pass < 2; pass++) {
    if (dtiny && (S->tp_sticky > 0 || pass == 1)) {
        schur_dense_load(S, 0.0);
        const double one = 1.0;
        const int nt = chol_tiny_blocked(m, m, S->Mc, m, &one, 0, 1e-14);
        S->tp_sticky = nt > 0 ? 3 : S->tp_sticky - 1;
        S->reg = nt > 0 ? 1e-300 : 0.0; if (nt > 0) S->nreg++;
        S->min_pivot = 1e-300;
        if (getenv("BRISK_TPDBG")) printf("     [dense tiny pivots: %d]\n", nt);
        if (nt == 0) S->reg_hint = 0;
        return 0;
    }
    if (pass == 1) break;
    if (dtiny) {
        /* plain attempt; on a tiny or failed pivot, the tiny-pivot mode (pass 1) */
        schur_dense_load(S, 0.0);
        info = brisk_dpotrf(m, S->Mc, m);
        if (info == 0) {
            double pmin = 1.0;
            for (int i = 0; i < m; i++) { double p = S->Mc[i + (size_t)i * m]; if (p * p < pmin) pmin = p * p; }
            if (pmin >= S->pivtol) { S->reg = 0; S->min_pivot = pmin; return 0; }
        }
        continue;
    }
    break;
    }
    /* once a factorization has needed the regularization, the next ones (same phase of the
     * run, M only gets worse) start with it instead of paying for a failed attempt first */
    double reg = S->reg_hint ? S->reg_ill : 0;
    for (int attempt = 0; attempt < 10; attempt++) {
        schur_dense_load(S, reg);
        double tpf = wtime();
        info = brisk_dpotrf(m, S->Mc, m);
        if (getenv("BRISK_FACTDBG")) printf("       [dpotrf m %d reg %.0e info %d %.3fs]\n", m, reg, info, wtime() - tpf);
        if (info == 0) {
            /* Equilibrated M has unit diagonal, so squared pivots lie in (0,1]. A tiny
             * pivot means the factor is numerically meaningless even though Cholesky
             * "succeeded": regularize instead (PCG then refines against the true M). */
            double pmin = 1.0;
            for (int i = 0; i < m; i++) { double p = S->Mc[i + (size_t)i * m]; if (p * p < pmin) pmin = p * p; }
            if (pmin >= S->pivtol || reg >= S->reg_ill) {
                S->reg = reg; if (reg > 0) S->nreg++;
                S->min_pivot = pmin;
                if (S->use_hint) S->reg_hint = reg > 0;
                return 0;
            }
            reg = fmax(S->reg_ill, reg * 100);
            continue;
        }
        reg = (reg == 0) ? 1e-13 : reg * 100;
    }
    return info;
}

static inline void schur_prec_(Schur *S, const double *r, double *z);
static inline void schur_prec(Schur *S, const double *r, double *z) { const double t0 = wtime(); schur_prec_(S, r, z); g_solprof_p[0] += wtime() - t0; g_solprof_p[1] += 1; }
static inline void schur_prec_(Schur *S, const double *r, double *z) {
    int m = S->m, info; (void)info;
    if (S->SC) {
        memcpy(z, r, sizeof(double) * m);
        schol_solve(S->SC, z, S->e1);
        return;
    }
    if (S->E) {
        const Envelope *E = S->E;
        for (int i = 0; i < m; i++) { int o = E->perm[i]; S->e1[i] = r[o] * S->D[o]; }
        env_solve(E, S->e1);
        for (int i = 0; i < m; i++) { int o = E->perm[i]; z[o] = S->e1[i] * S->D[o]; }
        return;
    }
    if (S->is_float) {
        for (int i = 0; i < m; i++) S->fw[i] = (float)(r[i] * S->D[i]);
        /* 5.7: two triangular solves of level 2 for one right-hand side: (s/d)potrs goes through the
         * level-3 solve, which for a single column is 6 times slower in single precision and 1.5 times
         * in double (OpenBLAS, m = 1,000 ... 8,000). 5.11: from m = 1,024 the threaded panel solve
         * of sparsechol.c (brisk_spotrs1 / brisk_dpotrs1), the same bits for any thread count */
        if (ENV_ON("BRISK_POTRS")) BL(spotrs_)("L", &m, &IONE, S->Mf, &m, S->fw, &m, &info); else {
        brisk_spotrs1(m, S->Mf, m, S->fw); }
        for (int i = 0; i < m; i++) z[i] = S->fw[i] * S->D[i];
        return;
    }
    for (int i = 0; i < m; i++) z[i] = r[i] * S->D[i];
    if (ENV_ON("BRISK_POTRS")) BL(dpotrs_)("L", &m, &IONE, S->Mc, &m, z, &m, &info); else {
    brisk_dpotrs1(m, S->Mc, m, z); }
    for (int i = 0; i < m; i++) z[i] *= S->D[i];
}

/* Preconditioned CG on M x = rhs, preconditioned by the (possibly regularized)
 * Cholesky factor. In the well-conditioned case the first residual check
 * already passes; near a degenerate optimum CG recovers the accuracy that
 * plain iterative refinement loses. Returns the relative residual.            */
static double g_berr = -1;
static inline double berr_c(void) { if (g_berr < 0) { const char *e = getenv("BRISK_BERR"); g_berr = e ? atof(e) : 0.0; } return g_berr; }
static int g_pcgstag = -1;
static inline int pcg_stag_on(void) { if (g_pcgstag < 0) { const char *e = getenv("BRISK_PCGSTAG"); g_pcgstag = e ? atoi(e) : 0; } return g_pcgstag; }
/* debug (BRISK_RESDBG): largest diagonal entry of M (a lower bound on ||M||_2) */
static double schur_dmax(Schur *S) {
    const int m = S->m; double d = 0;
    if (S->SC) { for (int i = 0; i < m; i++) { double v = schol_get(S->SC, i, i); if (v > d) d = v; } return d; }
    if (S->E) return 0;
    for (int i = 0; i < m; i++) { double v = schur_dense_get(S, i, i); if (v > d) d = v; }
    return d;
}
static int g_resdbg = -1;
static double schur_solve_(Schur *S, const double *rhs, double *x, double tol_rel);
/* tol_rel = 0: full accuracy (1e-13; 1e-10 with a single-precision factor) */
static void solprof_print(void) { fprintf(stderr, "   [solve profile: prec %.3f s in %.0f calls, mv %.3f s in %.0f calls]\n", g_solprof_p[0], g_solprof_p[1], g_solprof_p[2], g_solprof_p[3]); }
static double schur_solve(Schur *S, const double *rhs, double *x, double tol_rel) {
    static int pr = -1; if (pr < 0) { pr = getenv("BRISK_SOLPROF") != NULL; if (pr) atexit(solprof_print); }
    double t = wtime();
    double r = schur_solve_(S, rhs, x, tol_rel);
    S->t_solve += wtime() - t;
    return r;
}
static double schur_solve_(Schur *S, const double *rhs, double *x, double tol_rel) {
    const int m = S->m;
    if (S->bF) { g_sc_cnt[6]++; return bord_solve(S, rhs, x, tol_rel); }
    const double mone = -1.0;
    double nb = BL(dnrm2_)(&m, rhs, &IONE);
    if (nb == 0) { memset(x, 0, sizeof(double) * m); return 0; }
    schur_prec(S, rhs, x);
    if (S->qr_fact) {   /* 5.16: the square-root factor is trusted as it is (a check against the double M would undo its accuracy) */
        /* refinement with the residual computed through the Gram vectors in double-double:
         * r = D rhs - G'(G z), z the equilibrated solution; the correction solved with R */
        static int nref = -1; if (nref < 0) { const char *e = getenv("BRISK_HSDQRREF"); nref = e ? atoi(e) : 0; }
        if (nref > 0 && S->qrG) {
            const size_t N = S->qrN;
            double *z = malloc(sizeof(double) * m), *th = malloc(sizeof(double) * N), *tl = malloc(sizeof(double) * N), *c = malloc(sizeof(double) * m);
            for (int pass = 0; pass < nref; pass++) {
                for (int i = 0; i < m; i++) z[i] = x[i] / S->D[i];
                for (size_t p = 0; p < N; p++) { th[p] = 0; tl[p] = 0; }
                for (int j = 0; j < m; j++) {               /* t = G z in dd */
                    const double *g = S->qrG + (size_t)j * N, zj = z[j];
                    for (size_t p = 0; p < N; p++) {
                        const double pr = g[p] * zj, pe = fma(g[p], zj, -pr);
                        double s = th[p] + pr; const double bb = s - th[p]; tl[p] += (th[p] - (s - bb)) + (pr - bb) + pe; th[p] = s;
                    }
                }
                double rn = 0, bn = 0;
                for (int i = 0; i < m; i++) {               /* r_i = D_i rhs_i - g_i' t */
                    const double *g = S->qrG + (size_t)i * N;
                    double sh = 0, sl = 0;
                    for (size_t p = 0; p < N; p++) {
                        const double pr = g[p] * th[p], pe = fma(g[p], th[p], -pr) + g[p] * tl[p];
                        double s = sh + pr; const double bb = s - sh; sl += (sh - (s - bb)) + (pr - bb) + pe; sh = s;
                    }
                    c[i] = (rhs[i] * S->D[i] - sh) - sl;
                    rn += c[i] * c[i]; bn += rhs[i] * S->D[i] * rhs[i] * S->D[i];
                }
                if (ENV_ON("BRISK_HSDQRDBG")) printf("     [HSDQR refine pass %d: |r| / |D rhs| = %.2e]\n", pass, sqrt(rn / bn));
                brisk_dpotrs1(m, S->Mc, m, c);
                for (int i = 0; i < m; i++) x[i] = (z[i] + c[i]) * S->D[i];
            }
            free(th); free(tl); free(z); free(c);
        }
        S->nsolve++; g_sc_cnt[0]++; return 0;
    }
    static int notrust = -1;
    if (notrust < 0) notrust = getenv("BRISK_NOTRUST") != NULL;
    if (S->trust && !notrust && tol_rel <= 0) { S->nsolve++; g_sc_cnt[0]++; return 0; }
    /* a loose solve (correctors: 1e-7) on a factor that already gave 1e-3 of that accuracy
     * at a first check: the residual product (as costly as the solve) is skipped */
    if (g_looseskip && tol_rel >= 1e-8 && S->SC && S->ok_rel <= 1e-3 * tol_rel) { S->nsolve++; g_sc_cnt[0]++; return 0; }
    /* r = rhs - M x */
    memcpy(S->r, rhs, sizeof(double) * m);
    double nabs = -1;                                 /* || |M||x| ||, when computed */
    if (S->SC && berr_c() > 0) {
        /* 4.27: with |M||x| from the same pass (see below) */
        if (!S->ya) S->ya = malloc(sizeof(double) * (m + 1));
        g_sc_cnt[7]++;
        schol_mv_abs(S->SC, x, S->e2, S->ya);
        for (int i = 0; i < m; i++) S->r[i] = mone * S->e2[i] + S->r[i];
        nabs = BL(dnrm2_)(&m, S->ya, &IONE);
    } else schur_mv(S, mone, x, DONE, S->r);
    double nr = BL(dnrm2_)(&m, S->r, &IONE);
    /* single precision is only used far from the optimum, where an inexact
     * Newton step is enough; A(dX) = rp is enforced separately afterwards */
    const double tol = fmax(tol_rel, S->is_float ? 1e-10 : 1e-13) * nb;
    /* 4.27: the rounding floor of the residual itself, c u (|M||x| + |b|) (LAPACK's backward
     * error test, normwise). An exact double factor whose residual is there is as accurate
     * as the residual can tell; PCG cannot lower it (the moment image forms: |x| = |dy| the
     * moments >> |b| / |M|, so the 1e-13 |b| test failed on every solve and PCG took 1-10
     * steps at a median gain of 1.0-1.6: roa_vdp_d10_I ~340 steps, 23% of the run). */
    const int exact_f = S->reg == 0 && !S->is_float;
    const double floor_r = (nabs >= 0 && exact_f) ? berr_c() * 1.1102230246251565e-16 * (nabs + nb) : 0.0;
    S->nsolve++;
    if (nb > 0 && nr / nb < S->ok_rel) S->ok_rel = nr / nb;
    if (nr <= tol || nr <= floor_r) {
        g_sc_cnt[1]++;
        if (S->SC && S->reg == 0 && !S->is_float && tol_rel <= 0) S->trust = 1;
        return nr / nb;
    }
    g_sc_cnt[2]++;
    double best = nr;
    const double nr0_dbg = nr;
    double hist_dbg[50];
    memcpy(S->xb, x, sizeof(double) * m);
    schur_prec(S, S->r, S->t);                        /* z */
    memcpy(S->p, S->t, sizeof(double) * m);
    double rz = BL(ddot_)(&m, S->r, &IONE, S->t, &IONE);
    int maxit = (S->reg > 0 || S->is_float) ? 50 : 10, stall = 0, used = 0;
    /* 4.27: an exact double factor is backward stable, so a first residual above the target
     * is (almost always) the rounding floor u |M||x|, which PCG cannot lower: the moment
     * image forms (|x| = |dy| the moments, >> |b| / |M|) failed the 1e-13 |b| check on
     * every solve and PCG took 1-10 steps at a median gain of 1.0-1.6 (roa_vdp_d10_I: ~340
     * steps, 23% of the run). There PCG stops at the first step that does not halve the
     * residual (was: four steps below a 10% gain). BRISK_PCGSTAG=0 restores the old rule. */
    const int stag = pcg_stag_on() && S->reg == 0 && !S->is_float;
    const double sgain = stag ? 0.5 : 0.9;
    const int smax = stag ? 1 : 4;
    for (int k = 0; k < maxit; k++) {
        schur_mv(S, DONE, S->p, DZERO, S->q);
        double pq = BL(ddot_)(&m, S->p, &IONE, S->q, &IONE);
        if (!(pq > 0)) break;
        double a = rz / pq, ma = -a;
        BL(daxpy_)(&m, &a, S->p, &IONE, x, &IONE);
        BL(daxpy_)(&m, &ma, S->q, &IONE, S->r, &IONE);
        S->pcg_iters++;
        used++;
        nr = BL(dnrm2_)(&m, S->r, &IONE);
        if (used <= 50) hist_dbg[used - 1] = nr / nr0_dbg;
        if (nr < best) {
            stall = (nr > sgain * best) ? stall + 1 : 0;
            best = nr;
            memcpy(S->xb, x, sizeof(double) * m);
        } else stall++;
        if (nr <= tol || nr <= floor_r || stall >= smax) break;
        schur_prec(S, S->r, S->t);
        double rz2 = BL(ddot_)(&m, S->r, &IONE, S->t, &IONE);
        double beta = rz2 / rz;
        rz = rz2;
        for (int i = 0; i < m; i++) S->p[i] = S->t[i] + beta * S->p[i];
    }
    { extern long g_pcgh[12]; g_pcgh[used < 10 ? used : 10]++; if (best > tol) g_pcgh[11]++; }
    /* the recursively updated residual drifts; confirm with a true residual */
    memcpy(x, S->xb, sizeof(double) * m);
    memcpy(S->r, rhs, sizeof(double) * m);
    schur_mv(S, mone, x, DONE, S->r);
    double rel = BL(dnrm2_)(&m, S->r, &IONE) / nb;
    if (g_resdbg < 0) g_resdbg = getenv("BRISK_RESDBG") != NULL;
    if (g_resdbg) {
        const double nx = BL(dnrm2_)(&m, x, &IONE), dm = schur_dmax(S);
        printf("     RES1 tol %.0e reg %.0e fl %d nb %.2e nx %.2e dmax %.2e | r0/nb %.2e r0/(u dmax nx) %.2e | used %d final/nb %.2e final/(u dmax nx) %.2e\n",
               tol_rel, S->reg, S->is_float, nb, nx, dm, nr0_dbg / nb, nr0_dbg / (1.1e-16 * dm * nx), used, rel, rel * nb / (1.1e-16 * dm * nx));
        printf("     HIST1 tolmet %d floor %.2e r0/floor1 %.2e fin/floor1 %.2e :", best <= tol, floor_r, nabs >= 0 ? nr0_dbg / (1.1e-16 * (nabs + nb)) : -1.0, nabs >= 0 ? rel * nb / (1.1e-16 * (nabs + nb)) : -1.0);
        for (int q = 0; q < used && q < 50; q++) printf(" %.2e", hist_dbg[q]);
        printf("\n");
    }
    if (rel > S->worst_rel) S->worst_rel = rel;
    /* single precision is losing: go back to double from the next factorization */
    if (S->is_float && (used > g_fl_maxpcg || best > tol || rel > fmax(1e-9, 10 * tol_rel))) S->want_double = 1;
    return rel;
}

/* Several right-hand sides at once: the preconditioner (triangular solves) and the
 * products with M become BLAS-3 on the dense paths, so k solves cost about one in memory
 * traffic (on dense Schur complements of 5000+ the solves are memory-bound). PCG runs in
 * lockstep; each column stops on its own criterion.                                      */
static void schur_prec_multi(Schur *S, int k, const double *R, double *Zo) {
    const int m = S->m;
    int info;
    if (S->SC && k >= 2 && k <= 8 && ENV_ON("BRISK_SOLVE2")) {
        /* pairs of columns in one pass over L (4.22, opt-in: 2.5% of the solve time on
         * TSSOS case2869, and the changed rounding moved attr_henon_d8_I from 32 to 41 its) */
        double *wk = malloc(sizeof(double) * 2 * (size_t)m);
        memcpy(Zo, R, sizeof(double) * (size_t)m * k);
        int c = 0;
        for (; c + 1 < k; c += 2) schol_solve2(S->SC, Zo + (size_t)c * m, Zo + (size_t)(c + 1) * m, wk);
        if (c < k) schol_solve(S->SC, Zo + (size_t)c * m, S->e1);
        free(wk);
        return;
    }
    if (S->SC && k > 8) {       /* one pass over L for many columns (for 2 it was slower: case162 2.05 -> 2.24 s) */
        memcpy(Zo, R, sizeof(double) * (size_t)m * k);
        double *wk = malloc(sizeof(double) * (size_t)m * k);
        schol_solve_many(S->SC, Zo, k, wk);
        free(wk);
        return;
    }
    if (S->SC || S->E) { for (int c = 0; c < k; c++) schur_prec(S, R + (size_t)c * m, Zo + (size_t)c * m); return; }
    /* 5.12: the k columns by one panel solve on the threads (brisk_*potrsk: L read once; the
     * level-2 solves per column ran on one thread, dpotrs too). BRISK_POTRSK=0: as before. */
    static int potrsk_on = -1; if (potrsk_on < 0) { const char *e = getenv("BRISK_POTRSK"); potrsk_on = e ? atoi(e) : 1; }
    const double tq0 = wtime();
    if (S->is_float) {
        float *fw = malloc(sizeof(float) * (size_t)m * k);
        for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) fw[i + (size_t)c * m] = (float)(R[i + (size_t)c * m] * S->D[i]);
        if (ENV_ON("BRISK_POTRS")) BL(spotrs_)("L", &m, &k, S->Mf, &m, fw, &m, &info);
        else if (potrsk_on) brisk_spotrsk(m, k, S->Mf, m, fw, m);
        else
        for (int c = 0; c < k; c++) {                 /* (per column: see schur_prec) */
            BL(strsv_)("L", "N", "N", &m, S->Mf, &m, fw + (size_t)c * m, &IONE);
            BL(strsv_)("L", "T", "N", &m, S->Mf, &m, fw + (size_t)c * m, &IONE);
        }
        for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) Zo[i + (size_t)c * m] = fw[i + (size_t)c * m] * S->D[i];
        free(fw);
        g_solprof_p[0] += wtime() - tq0; g_solprof_p[1] += k;
        return;
    }
    for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) Zo[i + (size_t)c * m] = R[i + (size_t)c * m] * S->D[i];
    if (potrsk_on && !ENV_ON("BRISK_POTRS")) brisk_dpotrsk(m, k, S->Mc, m, Zo, m);
    else if (k <= 2 && !ENV_ON("BRISK_POTRS")) for (int c = 0; c < k; c++) {
        BL(dtrsv_)("L", "N", "N", &m, S->Mc, &m, Zo + (size_t)c * m, &IONE);
        BL(dtrsv_)("L", "T", "N", &m, S->Mc, &m, Zo + (size_t)c * m, &IONE);
    } else BL(dpotrs_)("L", &m, &k, S->Mc, &m, Zo, &m, &info);
    for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) Zo[i + (size_t)c * m] *= S->D[i];
    g_solprof_p[0] += wtime() - tq0; g_solprof_p[1] += k;
}
static void schur_mv_multi(Schur *S, int k, double alpha, const double *X, double beta, double *Y) {
    const int m = S->m;
    /* few columns: dsymm packs all of M on every call (a copy of m^2 values, 14% of the time
     * on roa_dint_d6_I); dsymv per column only reads it */
    if (g_mv2off < 0) { const char *e = getenv("BRISK_MV2"); g_mv2off = e && atoi(e) == 0; }
    if (S->SC && !S->bF && k >= 2 && !g_mv2off) {
        /* 18 October 2026: the sparse factor's M in one pass for two columns (schol_mv2; the product is bound by the
         * memory traffic of M). BRISK_MV2=0: a pass per column */
        const double t0 = wtime();
        int c = 0;
        for (; c + 1 < k; c += 2) {
            double *y1 = Y + (size_t)c * m, *y2 = Y + (size_t)(c + 1) * m;
            if (!S->e3m) S->e3m = malloc(sizeof(double) * 2 * (size_t)m);
            double *t1 = S->e3m, *t2 = S->e3m + m;
            g_sc_cnt[7] += 2;
            schol_mv2(S->SC, X + (size_t)c * m, X + (size_t)(c + 1) * m, t1, t2);
            for (int i = 0; i < m; i++) { y1[i] = alpha * t1[i] + (beta == 0 ? 0.0 : beta * y1[i]); y2[i] = alpha * t2[i] + (beta == 0 ? 0.0 : beta * y2[i]); }
        }
        g_solprof_p[2] += wtime() - t0; g_solprof_p[3] += c;
        if (c < k) schur_mv(S, alpha, X + (size_t)c * m, beta, Y + (size_t)c * m);
        return;
    }
    if (S->SC || S->E || (S->mv_ok && S->is_float) || k <= 4) { for (int c = 0; c < k; c++) schur_mv(S, alpha, X + (size_t)c * m, beta, Y + (size_t)c * m); return; }
    if (!S->mT_ok || S->low_M) { xsymm("L", "L", &m, &k, &alpha, S->M, &m, X, &m, &beta, Y, &m); return; }
    xsymm("L", "U", &m, &k, &alpha, S->M, &m, X, &m, &beta, Y, &m);    /* 4.26: M in the upper copy */
    for (int c = 0; c < k; c++)
        for (int i = 0; i < m; i++) Y[i + (size_t)c * m] += alpha * (S->Md[i] - S->M[i + (size_t)i * m]) * X[i + (size_t)c * m];
}
static void schur_solve_multi(Schur *S, int k, const double *const *rhs, double *const *x, const double *tol_rel) {
    if (S->bF || S->qr_fact) { for (int c = 0; c < k; c++) schur_solve(S, rhs[c], x[c], tol_rel[c]); return; }
    const double t0 = wtime();
    const int m = S->m;
    const size_t mk = (size_t)m * k;
    double *B = malloc(sizeof(double) * mk * 7);
    double *X = B + mk, *Rr = X + mk, *Zz = Rr + mk, *Pp = Zz + mk, *Qq = Pp + mk, *XB = Qq + mk;
    double nb[8], tol[8], best[8], rz[8], nr[8];
    int done[8], stall[8], used[8];
    for (int c = 0; c < k; c++) {
        memcpy(B + (size_t)c * m, rhs[c], sizeof(double) * m);
        nb[c] = BL(dnrm2_)(&m, rhs[c], &IONE);
        tol[c] = fmax(tol_rel[c], S->is_float ? 1e-10 : 1e-13) * nb[c];
        done[c] = !(nb[c] > 0); stall[c] = 0; used[c] = 0;
    }
    schur_prec_multi(S, k, B, X);
    /* trusted exact factor (its first residual check passed): as in schur_solve (4.22) */
    static int notrust_m = -1;
    if (notrust_m < 0) notrust_m = getenv("BRISK_NOTRUST") != NULL;
    int all0 = 1;
    for (int c = 0; c < k; c++) all0 &= tol_rel[c] <= 0;
    if (S->trust && !notrust_m && all0) {
        for (int c = 0; c < k; c++) { memcpy(x[c], X + (size_t)c * m, sizeof(double) * m); S->nsolve++; }
        g_sc_cnt[3]++;
        free(B);
        S->t_solve += wtime() - t0;
        return;
    }
    memcpy(Rr, B, sizeof(double) * mk);
    int first_done = 0;
    if (g_looseskip && S->SC && k >= 2 && !getenv("BRISK_NOMULTISKIP")) {
        /* 4.23 (class default with the loose skip): the first column is checked alone; when it
         * meets 1e-3 of the looser tolerances of the others (predictor 1e-7 against the v
         * solve's 1e-10), their checks are skipped (one product with M per iteration) */
        double tmin = INFINITY;
        for (int c = 1; c < k; c++) tmin = fmin(tmin, tol_rel[c]);
        schur_mv(S, -1.0, X, 1.0, Rr);
        first_done = 1;
        const double r0 = BL(dnrm2_)(&m, Rr, &IONE), rel0 = nb[0] > 0 ? r0 / nb[0] : 0.0;
        if (tmin >= 1e-8 && r0 <= tol[0] && rel0 <= 1e-3 * tmin) {
            if (nb[0] > 0 && rel0 < S->ok_rel) S->ok_rel = rel0;
            for (int c = 0; c < k; c++) {
                memcpy(x[c], X + (size_t)c * m, sizeof(double) * m);
                if (!(nb[c] > 0)) memset(x[c], 0, sizeof(double) * m);
                S->nsolve++;
            }
            if (rel0 > S->worst_rel) S->worst_rel = rel0;
            g_sc_cnt[4]++; g_multiskip++;
            free(B);
            S->t_solve += wtime() - t0;
            return;
        }
    }
    double flr[8], fl1[8];                       /* 4.27: rounding floors (see schur_solve_) */
    for (int c = 0; c < k; c++) { flr[c] = 0.0; fl1[c] = -1.0; }
    const int exact_f = S->reg == 0 && !S->is_float;
    if (S->SC && berr_c() > 0 && !first_done) {
        if (!S->ya) S->ya = malloc(sizeof(double) * (m + 1));
        for (int c = 0; c < k; c++) {
            double *rc = Rr + (size_t)c * m;
            g_sc_cnt[7]++;
            schol_mv_abs(S->SC, X + (size_t)c * m, S->e2, S->ya);
            for (int i = 0; i < m; i++) rc[i] = -1.0 * S->e2[i] + rc[i];
            fl1[c] = 1.1102230246251565e-16 * (BL(dnrm2_)(&m, S->ya, &IONE) + nb[c]);
            if (exact_f) flr[c] = berr_c() * fl1[c];
        }
    }
    else if (first_done) { for (int c = 1; c < k; c++) schur_mv(S, -1.0, X + (size_t)c * m, 1.0, Rr + (size_t)c * m); }
    else schur_mv_multi(S, k, -1.0, X, 1.0, Rr);
    memcpy(XB, X, sizeof(double) * mk);
    double nr0m[8], histm[8][50];
    for (int c = 0; c < k; c++) {
        nr[c] = BL(dnrm2_)(&m, Rr + (size_t)c * m, &IONE);
        nr0m[c] = nr[c];
        if (nb[c] > 0 && nr[c] / nb[c] < S->ok_rel) S->ok_rel = nr[c] / nb[c];
        best[c] = nr[c];
        if (nr[c] <= tol[c] || nr[c] <= flr[c]) done[c] = 1;
        S->nsolve++;
    }
    int alldone = 1;
    for (int c = 0; c < k; c++) alldone &= done[c];
    if (alldone && !getenv("BRISK_MULTI_RECHECK")) {
        /* every column passed the first check: X is returned and its residual is known */
        int exact = S->SC && S->reg == 0 && !S->is_float && all0;
        for (int c = 0; c < k; c++) {
            memcpy(x[c], X + (size_t)c * m, sizeof(double) * m);
            double rel = nb[c] > 0 ? nr[c] / nb[c] : 0.0;
            if (rel > S->worst_rel) S->worst_rel = rel;
            if (!(nb[c] > 0)) memset(x[c], 0, sizeof(double) * m);
        }
        if (exact) S->trust = 1;
        g_sc_cnt[4]++;
        free(B);
        S->t_solve += wtime() - t0;
        return;
    }
    if (!alldone) {
        g_sc_cnt[5]++;
        schur_prec_multi(S, k, Rr, Zz);
        memcpy(Pp, Zz, sizeof(double) * mk);
        for (int c = 0; c < k; c++) rz[c] = BL(ddot_)(&m, Rr + (size_t)c * m, &IONE, Zz + (size_t)c * m, &IONE);
        const int maxit = (S->reg > 0 || S->is_float) ? 50 : 10;
        const int stag = pcg_stag_on() && S->reg == 0 && !S->is_float;   /* 4.27: as in schur_solve_ */
        const double sgain = stag ? 0.5 : 0.9;
        const int smax = stag ? 1 : 4;
        for (int it = 0; it < maxit; it++) {
            schur_mv_multi(S, k, 1.0, Pp, 0.0, Qq);
            alldone = 1;
            for (int c = 0; c < k; c++) {
                if (done[c]) continue;
                double *xc = X + (size_t)c * m, *rc = Rr + (size_t)c * m, *pc = Pp + (size_t)c * m, *qc = Qq + (size_t)c * m;
                double pq = BL(ddot_)(&m, pc, &IONE, qc, &IONE);
                if (!(pq > 0)) { done[c] = 1; continue; }
                double a = rz[c] / pq, ma = -a;
                BL(daxpy_)(&m, &a, pc, &IONE, xc, &IONE);
                BL(daxpy_)(&m, &ma, qc, &IONE, rc, &IONE);
                S->pcg_iters++; used[c]++;
                nr[c] = BL(dnrm2_)(&m, rc, &IONE);
                if (used[c] <= 50) histm[c][used[c] - 1] = nr[c] / fmax(nr0m[c], 1e-300);
                if (nr[c] < best[c]) {
                    stall[c] = (nr[c] > sgain * best[c]) ? stall[c] + 1 : 0;
                    best[c] = nr[c];
                    memcpy(XB + (size_t)c * m, xc, sizeof(double) * m);
                } else stall[c]++;
                if (nr[c] <= tol[c] || nr[c] <= flr[c] || stall[c] >= smax) done[c] = 1;
                alldone &= done[c];
            }
            if (alldone) break;
            schur_prec_multi(S, k, Rr, Zz);
            for (int c = 0; c < k; c++) {
                if (done[c]) continue;
                double *rc = Rr + (size_t)c * m, *zc = Zz + (size_t)c * m, *pc = Pp + (size_t)c * m;
                double rz2 = BL(ddot_)(&m, rc, &IONE, zc, &IONE), beta = rz2 / rz[c];
                rz[c] = rz2;
                for (int i = 0; i < m; i++) pc[i] = zc[i] + beta * pc[i];
            }
        }
    }
    for (int c = 0; c < k; c++) memcpy(x[c], XB + (size_t)c * m, sizeof(double) * m);
    /* true residuals of the returned solutions */
    memcpy(Rr, B, sizeof(double) * mk);
    schur_mv_multi(S, k, -1.0, XB, 1.0, Rr);
    for (int c = 0; c < k; c++) {
        double rel = nb[c] > 0 ? BL(dnrm2_)(&m, Rr + (size_t)c * m, &IONE) / nb[c] : 0.0;
        if (g_resdbg < 0) g_resdbg = getenv("BRISK_RESDBG") != NULL;
        if (g_resdbg && nb[c] > 0) {
            const double nx = BL(dnrm2_)(&m, x[c], &IONE), dm = schur_dmax(S);
            printf("     RESM%d/%d tol %.0e reg %.0e fl %d nb %.2e nx %.2e dmax %.2e | r0/nb %.2e r0/(u dmax nx) %.2e | used %d final/nb %.2e final/(u dmax nx) %.2e\n",
                   c, k, tol_rel[c], S->reg, S->is_float, nb[c], nx, dm, nr0m[c] / nb[c], nr0m[c] / (1.1e-16 * dm * nx), used[c], rel, rel * nb[c] / (1.1e-16 * dm * nx));
            printf("     HISTM tolmet %d floor %.2e r0/floor1 %.2e fin/floor1 %.2e :", best[c] <= tol[c], flr[c], fl1[c] > 0 ? nr0m[c] / fl1[c] : -1.0, fl1[c] > 0 ? rel * nb[c] / fl1[c] : -1.0);
            for (int q = 0; q < used[c] && q < 50; q++) printf(" %.2e", histm[c][q]);
            printf("\n");
        }
        if (rel > S->worst_rel) S->worst_rel = rel;
        if (S->is_float && (used[c] > g_fl_maxpcg || best[c] > tol[c] || rel > fmax(1e-9, 10 * tol_rel[c]))) S->want_double = 1;
    }
    free(B);
    S->t_solve += wtime() - t0;
}

/* ------------------------------------------------------------------------ */

/* ------------------------------------------------------------------------ */
/* Factor-based operators with Lhat Lhat' = M (M = D^{-1} Lc Lc' D^{-1}, Lc = Cholesky of
 * the equilibrated matrix; envelope: with the RCM permutation). "z-space" vectors live in
 * factor coordinates. Double-precision factors only (returns 0 otherwise).             */
static void env_forward(const Envelope *E, double *x) {
    for (int i = 0; i < E->m; i++) {
        const int fi = E->fst[i];
        const double *Li = E->L + E->rptr[i] - fi;
        double s = x[i];
        for (int k = fi; k < i; k++) s -= Li[k] * x[k];
        x[i] = s / Li[i];
    }
}
static void env_backward(const Envelope *E, double *x) {
    for (int i = E->m - 1; i >= 0; i--) {
        const int fi = E->fst[i];
        const double *Li = E->L + E->rptr[i] - fi;
        double xi = x[i] / Li[i];
        x[i] = xi;
        for (int k = fi; k < i; k++) x[k] -= Li[k] * xi;
    }
}
static void env_mulT(const Envelope *E, const double *t, double *z) {       /* z = L' t */
    for (int i = 0; i < E->m; i++) z[i] = 0;
    for (int i = 0; i < E->m; i++) {
        const int fi = E->fst[i];
        const double *Li = E->L + E->rptr[i] - fi;
        for (int k = fi; k <= i; k++) z[k] += Li[k] * t[i];
    }
}
static int lhat_ok(const Schur *S) { return !S->is_float; }
/* w = Lhat^{-1} v */
static void lhat_solveN(Schur *S, const double *v, double *w) {
    const int m = S->m;
    if (S->E) {
        for (int i = 0; i < m; i++) { int o = S->E->perm[i]; w[i] = v[o] * S->D[o]; }
        env_forward(S->E, w);
        return;
    }
    for (int i = 0; i < m; i++) w[i] = v[i] * S->D[i];
    BL(dtrsv_)("L", "N", "N", &m, S->Mc, &m, w, &IONE);
}
/* y = Lhat^{-T} z */
static void lhat_solveT(Schur *S, const double *z, double *y) {
    const int m = S->m;
    memcpy(S->e3 ? S->e3 : S->t, z, sizeof(double) * m);
    double *t = S->e3 ? S->e3 : S->t;
    if (S->E) {
        env_backward(S->E, t);
        for (int i = 0; i < m; i++) { int o = S->E->perm[i]; y[o] = t[i] * S->D[o]; }
        return;
    }
    BL(dtrsv_)("L", "T", "N", &m, S->Mc, &m, t, &IONE);
    for (int i = 0; i < m; i++) y[i] = t[i] * S->D[i];
}
/* z = Lhat' y */
__attribute__((unused)) static void lhat_mulT(Schur *S, const double *y, double *z) {
    const int m = S->m;
    if (S->E) {
        double *t = S->e2;
        for (int i = 0; i < m; i++) { int o = S->E->perm[i]; t[i] = y[o] / S->D[o]; }
        env_mulT(S->E, t, z);
        return;
    }
    for (int i = 0; i < m; i++) z[i] = y[i] / S->D[i];
    BL(dtrmv_)("L", "T", "N", &m, S->Mc, &m, z, &IONE);
}

/* Max step for L L' + a D >= 0 via Lanczos on L^{-1} D L^{-T}.               */

/* Exact lambda_min of L^{-1} D L^{-T} (or of D if L == NULL) for tiny blocks:
 * explicit formation, tridiagonalization and Sturm bisection, no BLAS call. */
#define SMALL_BLOCK 32
static double g_lam_rtol = 1e-10;
static int g_prek = -1;
static double small_lammin(int n, const double *L, const double *D) {
    { static int init = 0; if (!init) { const char *e = getenv("BRISK_LAMTOL"); if (e) g_lam_rtol = atof(e); init = 1; } }
    double B[SMALL_BLOCK * SMALL_BLOCK], Y[SMALL_BLOCK * SMALL_BLOCK];
    if (L) {
        /* Y = L^{-1} D (column by column), then B = L^{-1} Y' */
        for (int c = 0; c < n; c++)
            for (int i = 0; i < n; i++) {
                double v = D[i + c * n];
                for (int k = 0; k < i; k++) v -= L[i + k * n] * Y[k + c * n];
                Y[i + c * n] = v / L[i + i * n];
            }
        for (int c = 0; c < n; c++)
            for (int i = 0; i < n; i++) {
                double v = Y[c + i * n];
                for (int k = 0; k < i; k++) v -= L[i + k * n] * B[k + c * n];
                B[i + c * n] = v / L[i + i * n];
            }
        for (int c = 0; c < n; c++)
            for (int i = 0; i < c; i++) { double v = 0.5 * (B[i + c * n] + B[c + i * n]); B[i + c * n] = B[c + i * n] = v; }
    } else memcpy(B, D, sizeof(double) * n * n);
    /* Householder tridiagonalization (4n^3/3) and Sturm bisection for the smallest
     * eigenvalue; the lower end of the bracket is returned, so steps stay safe. (The
     * Jacobi sweeps used before cost ~5x more: 17% of the run on 1000 small cliques.) */
    double d[SMALL_BLOCK] = {0}, e[SMALL_BLOCK] = {0}, vv[SMALL_BLOCK], pw[SMALL_BLOCK];
    for (int k = 0; k + 2 < n; k++) {
        const int r = n - k - 1;                       /* length of the column below */
        double *x = &B[(k + 1) + k * n];
        double sig = 0;
        for (int i = 1; i < r; i++) sig += x[i] * x[i];
        if (sig == 0) { e[k] = x[0]; continue; }
        const double a = sqrt(x[0] * x[0] + sig), alpha = x[0] > 0 ? -a : a;
        vv[0] = x[0] - alpha;
        for (int i = 1; i < r; i++) vv[i] = x[i];
        const double vn2 = vv[0] * vv[0] + sig, beta = 2.0 / vn2;
        e[k] = alpha;
        /* trailing block A = B[k+1.., k+1..]: p = beta A v, w = p - (beta/2)(p'v) v */
        double pv = 0;
        for (int i = 0; i < r; i++) {
            double t = 0;
            for (int jj = 0; jj < r; jj++) t += B[(k + 1 + i) + (k + 1 + jj) * n] * vv[jj];
            pw[i] = beta * t; pv += pw[i] * vv[i];
        }
        const double hb = 0.5 * beta * pv;
        for (int i = 0; i < r; i++) pw[i] -= hb * vv[i];
        for (int jj = 0; jj < r; jj++)
            for (int i = jj; i < r; i++) {
                double t = B[(k + 1 + i) + (k + 1 + jj) * n] - vv[i] * pw[jj] - pw[i] * vv[jj];
                B[(k + 1 + i) + (k + 1 + jj) * n] = t; B[(k + 1 + jj) + (k + 1 + i) * n] = t;
            }
    }
    for (int k = 0; k < n; k++) d[k] = B[k + k * n];
    if (n >= 2) e[n - 2] = B[(n - 1) + (n - 2) * n];
    /* Gershgorin bracket */
    double lo = INFINITY, hi = -INFINITY, sc = 0;
    for (int k = 0; k < n; k++) {
        const double rad = (k > 0 ? fabs(e[k - 1]) : 0) + (k + 1 < n ? fabs(e[k]) : 0);
        lo = fmin(lo, d[k] - rad); hi = fmax(hi, d[k] + rad);
        sc = fmax(sc, fabs(d[k]) + rad);
    }
    if (n == 1) return d[0];
    hi = fmin(hi, d[0]);                               /* lambda_min <= any diagonal entry */
    for (int k = 1; k < n; k++) hi = fmin(hi, d[k]);
    const double tiny = 1e-300;
    for (int it = 0; it < 80; it++) {
        const double w = hi - lo;
        if (w <= g_lam_rtol * fmax(fabs(lo), fabs(hi)) || w <= 1e-14 * sc) break;
        const double x = 0.5 * (lo + hi);
        /* number of eigenvalues < x */
        int cnt = 0;
        double q = d[0] - x;
        if (q < 0) cnt++;
        for (int k = 1; k < n; k++) {
            if (fabs(q) < tiny) q = q < 0 ? -tiny : tiny;
            q = d[k] - x - e[k - 1] * e[k - 1] / q;
            if (q < 0) cnt++;
        }
        if (cnt >= 1) hi = x; else lo = x;
    }
    /* safety margin for the rounding of the reduction */
    return lo - 1e-13 * sc;
}

static long lz_calls = 0, lz_iters = 0;
static double ritz_bound(int kk, const double *alpha, const double *beta, double *out_theta) {
    double d[LANCZOS_K], e[LANCZOS_K], zv[LANCZOS_K * LANCZOS_K], work[2 * LANCZOS_K];
    int info;
    memcpy(d, alpha, sizeof(double) * kk);
    for (int i = 0; i + 1 < kk; i++) e[i] = beta[i];
    BL(dstev_)("V", &kk, d, e, zv, &kk, work, &info);
    if (info != 0) { *out_theta = 0; return 1e300; }
    *out_theta = d[0];
    return fabs(beta[kk - 1] * zv[kk - 1]);
}

static double lz_tol = 0.02;     /* relative Ritz accuracy for early termination */

/* Parallelism over blocks (many small blocks): per-block work only, every reduction
 * over blocks is either order-independent (min, and) or done serially afterwards,
 * so results do not depend on the number of threads.                               */
static int g_par_blk = 0;
#define PAR_BLOCKS _Pragma("omp parallel for schedule(dynamic, 8) if (g_par_blk)")
/* 5.12: a loop over the blocks that opens a parallel region only when it runs in parallel. With
 * `if (g_par_blk)` false the region is still there (inactive), every parallel region inside it
 * is a nested team: libgomp creates its threads for it and, with more threads alive than cores,
 * its barriers barely spin (a reduction of order 799 with ~2700 barriers: 96 against 33 ms).
 * One block (the large dense case) took that path everywhere. */
#define PBLK_FOR(k, n_, ...) do { if (g_par_blk) { _Pragma("omp parallel for schedule(dynamic, 8)") for (int k = 0; k < (n_); k++) { __VA_ARGS__ } } \
                                  else { for (int k = 0; k < (n_); k++) { __VA_ARGS__ } } } while (0)
static double sdp_maxstep_lam(int n, const double *L, const double *D, BS *s, int Kmax, double *lam_out);
/* 5.12: L = ZSP_L in the step-length calls: the sparse factor of Z (s->zf), Z = P' L L' P. The
 * operator L^{-1} P D P' L^{-T} (same eigenvalues as Z^{-1} D) costs two sparse triangular
 * solves and a product with D on the pattern of Z, instead of the n^2 of the dense factor. */
static const double ZSP_L[1] = { 0 };
static void zsp_op(int n, BS *s, const double *D, const double *v, double *w) {
    const int *perm = schol_perm(s->zf);
    double *u = s->zt, *y = s->zt + n;
    memcpy(w, v, sizeof(double) * n);
    schol_ltsolve(s->zf, w);
    for (int i = 0; i < n; i++) u[perm[i]] = w[i];
    memset(y, 0, sizeof(double) * n);
    for (int c = 0; c < n; c++) {
        const double uc = u[c];
        if (uc == 0) continue;
        const double *dc = D + (size_t)c * n;
        for (int q = s->zcp[c]; q < s->zcp[c + 1]; q++) y[s->zri[q]] += dc[s->zri[q]] * uc;
    }
    for (int i = 0; i < n; i++) w[i] = y[perm[i]];
    schol_lsolve(s->zf, w);
}
static double sdp_maxstep(int n, const double *L, const double *D, BS *s, int Kmax) {
    return sdp_maxstep_lam(n, L, D, s, Kmax, NULL);
}
/* Lanczos on L^{-1} D L^{-T} (or on D itself when L == NULL). Returns the max step
 * for L L' + a D >= 0 and, optionally, a conservative estimate of lambda_min.     */
static double sdp_maxstep_lam(int n, const double *L, const double *D, BS *s, int Kmax, double *lam_out) {
    if (n <= SMALL_BLOCK) {
        double lam = small_lammin(n, L, D);
        if (lam_out) *lam_out = lam;
        return lam >= 0 ? 1e30 : -1.0 / lam;
    }
    /* the fixed vector enters a warm start with weight 0.3 (BRISK_LZWMIX): at 1e-2 (5.12 stage 1)
     * a new extreme eigenvector that is nearly orthogonal to the last one did not enter the
     * Krylov space before the convergence test passed on the old one, and the step was too long
     * (control7_s: 8 -> 15-17 iterations, REDUCED; control10: 24 -> 28). 0.3 keeps most of the
     * saving (maxG32 11.0 -> 9.0 steps a call, 8.8 at 1e-2; vibra4 10.2 -> 7.8, 7.4) and the
     * control family is all OPTIMAL (22 files, 315 iterations; cold 319, 1e-2 328). Even so
     * the warm start changed the path of ill-posed moment relaxations with small blocks
     * (nn_n20_m100_K, n <= 211: OPTIMAL -> REDUCED; roa_dint_d8_K, roa_dint_d10_K worse): only
     * blocks of order >= 400 start warm (BRISK_LZWN), where the steps cost memory traffic;
     * below, a step reads at most 1.3 MB and the fixed start costs little. And on maxG51 every
     * warm start cost iterations under every perturbation (cold 14, weight 0.01 and 0.1 15,
     * 0.3 17): fewer Lanczos steps leave a larger residual bound, the estimate theta - res is
     * more conservative and the steps shorter, and an interior-point iteration costs far more
     * than the steps saved. So the warm start is off (BRISK_LZWARM=1: on, for experiments). */
    static int warm_on = -1, par_on = -1, lzw_k = 3, lzw_n = 400; static double lzw_mix = 0.3;
    if (warm_on < 0) {
        warm_on = ENV_ON("BRISK_LZWARM"); par_on = !ENV_ON("BRISK_LZSER");     /* warm start off: see below */
        const char *e;
        if ((e = getenv("BRISK_LZWK"))) lzw_k = atoi(e);        /* experiment: first convergence test of a warm start */
        if ((e = getenv("BRISK_LZWN"))) lzw_n = atoi(e);        /* smallest block with a warm start */
        if ((e = getenv("BRISK_LZWMIX"))) lzw_mix = atof(e);    /* experiment: weight of the fixed vector */
    }
    int K = n < Kmax ? n : Kmax;
    if (K > LANCZOS_K) K = LANCZOS_K;
    double alpha[LANCZOS_K], beta[LANCZOS_K];
    double *Q = s->lq, *w = s->v2;
    /* 5.12: the start vector. The extreme eigenvector of L^{-1} D L^{-T} changes little from one
     * call to the next on the same side of a block (the predictor's and the corrector's
     * directions, the next iteration), so the last Ritz vector of that side starts the next
     * call, with a fixed vector added at weight 0.3 so that no direction is missing; the
     * convergence test then runs from the third step instead of the sixth. BRISK_LZCOLD=1: the
     * fixed start. */
    const int side = L && L == s->LX ? 0 : L && (L == s->LZ || L == ZSP_L) ? 1 : -1;
    double *qw = side >= 0 ? s->lq + (size_t)n * (LANCZOS_K + 1 + side) : NULL;
    const int warm = warm_on && side >= 0 && s->lzw_ok[side] && n >= lzw_n;
    double nrm = 0;
    for (int i = 0; i < n; i++) { Q[i] = 1.0 + 0.5 * sin(1.7 * i + 0.3); nrm += Q[i] * Q[i]; }
    nrm = 1.0 / sqrt(nrm);
    for (int i = 0; i < n; i++) Q[i] *= nrm;
    if (warm) {
        nrm = 0;
        for (int i = 0; i < n; i++) { Q[i] = qw[i] + lzw_mix * Q[i]; nrm += Q[i] * Q[i]; }
        nrm = 1.0 / sqrt(nrm);
        for (int i = 0; i < n; i++) Q[i] *= nrm;
    }
    /* 5.12: the operator on the solver's threads from n = 1024: the triangular solves in panels
     * (brisk_dtrsv1) and the product in column panels (par_symv), fixed by n, so the same bits for
     * any thread count, inside a parallel region too. Measured slower at first (maxG32's step
     * lengths 2.75 -> 3.91 s): maxstep ran it inside an inactive parallel region (one chunk of
     * blocks), where every region of the operator was a nested team; without that region the
     * primal step lengths of maxG32 take 0.50 s instead of 0.91 (BRISK_LZSER=1: one thread). */
    const int par = par_on && n >= 1024;
    int kk = 0;
    double scale = 0, theta = 0, res = 0;
    for (int j = 0; j < K; j++) {
        double *qj = Q + (size_t)j * n;
        memcpy(s->v1, qj, sizeof(double) * n);
        if (L == ZSP_L) zsp_op(n, s, D, qj, w);
        else if (par) {
            if (L) brisk_dtrsv1(1, n, L, n, s->v1);
            par_symv('L', n, 1.0, D, s->v1, 0.0, w);
            if (L) brisk_dtrsv1(0, n, L, n, w);
        } else
        if (g_lockfree && n <= 512) {
            /* 4.24: lock-free level-2 (dsymv takes the OpenBLAS buffer lock) */
            double *x = s->v1;
            /* (4.25: simd reductions - the scalar dependency chains made the step 2x slower
             * than BLAS on mcp500 at n = 500; the vector order is fixed at compile time) */
            if (L) for (int i = n - 1; i >= 0; i--) {
                double t = 0; const double *li = L + (size_t)i * n;
                #pragma omp simd reduction(+:t)
                for (int r = i + 1; r < n; r++) t += li[r] * x[r];
                x[i] = (x[i] - t) / li[i];
            }
            for (int i = 0; i < n; i++) w[i] = 0;
            for (int c = 0; c < n; c++) {
                const double *dc = D + (size_t)c * n; const double xc = x[c];
                double acc = 0;
                #pragma omp simd reduction(+:acc)
                for (int r = c + 1; r < n; r++) { acc += dc[r] * x[r]; w[r] += dc[r] * xc; }
                w[c] += acc + dc[c] * xc;
            }
            if (L) for (int c = 0; c < n; c++) { const double *lc = L + (size_t)c * n; const double t = w[c] / lc[c]; w[c] = t; for (int r = c + 1; r < n; r++) w[r] -= lc[r] * t; }
        } else {
        if (L) BL(dtrsv_)("L", "T", "N", &n, L, &n, s->v1, &IONE);
        BL(dsymv_)("L", &n, &DONE, D, &n, s->v1, &IONE, &DZERO, w, &IONE);
        if (L) BL(dtrsv_)("L", "N", "N", &n, L, &n, w, &IONE);
        }
        double a = BL(ddot_)(&n, qj, &IONE, w, &IONE);
        alpha[j] = a;
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i <= j; i++) {
                double *qi = Q + (size_t)i * n;
                double c = -BL(ddot_)(&n, qi, &IONE, w, &IONE);
                BL(daxpy_)(&n, &c, qi, &IONE, w, &IONE);
            }
        double b = BL(dnrm2_)(&n, w, &IONE);
        beta[j] = b;
        kk = j + 1;
        if (fabs(a) > scale) scale = fabs(a);
        if (b <= 1e-12 * (scale + 1e-300)) { beta[j] = 0; break; }
        /* early termination once the extreme Ritz value is resolved */
        if (kk >= (warm ? lzw_k : 6) && (kk % 3 == 0)) {
            res = ritz_bound(kk, alpha, beta, &theta);
            if (res <= lz_tol * fabs(theta) + 1e-14 * scale) break;
        }
        if (j + 1 < K) {
            double inv = 1.0 / b;
            double *qn = Q + (size_t)(j + 1) * n;
            for (int i = 0; i < n; i++) qn[i] = w[i] * inv;
        }
    }
    res = ritz_bound(kk, alpha, beta, &theta);
    if (qw && warm_on && res < 1e299) {          /* (only with the warm start) */
        /* the Ritz vector of the smallest Ritz value, for the next call on this side */
        double d[LANCZOS_K], e[LANCZOS_K], zv[LANCZOS_K * LANCZOS_K], work[2 * LANCZOS_K];
        int info;
        memcpy(d, alpha, sizeof(double) * kk);
        for (int i = 0; i + 1 < kk; i++) e[i] = beta[i];
        BL(dstev_)("V", &kk, d, e, zv, &kk, work, &info);
        if (info == 0) {
            memset(qw, 0, sizeof(double) * n);
            for (int i = 0; i < kk; i++) { const double c = zv[i]; BL(daxpy_)(&n, &c, Q + (size_t)i * n, &IONE, qw, &IONE); }
            const double q2 = BL(dnrm2_)(&n, qw, &IONE);
            if (q2 > 0 && isfinite(q2)) { const double iq = 1.0 / q2; for (int i = 0; i < n; i++) qw[i] *= iq; s->lzw_ok[side] = 1; }
        }
    }
    #pragma omp atomic
    lz_calls++;
    #pragma omp atomic
    lz_iters += kk;
    if (res > 1e299) { if (lam_out) *lam_out = -1e300; return 0.0; }
    double lam = theta - res;
    if (lam_out) *lam_out = lam;
    if (lam >= 0) return 1e30;
    return -1.0 / lam;
}

static double lp_maxstep(int n, const double *x, const double *dx, const char *fm) {
    double a = 1e30;
    for (int i = 0; i < n; i++) {
        if (!isfinite(dx[i])) return 0.0;
        if (fm && fm[i]) continue;                 /* free pair (embedding) */
        if (dx[i] < 0) { double r = -x[i] / dx[i]; if (r < a) a = r; }
    }
    return a;
}

/* V + a dV positive definite? (plain Cholesky on a copy; n <= SMALL_BLOCK) */
static int small_pd_at(int n, const double *V, const double *dV, double a) {
    double T[SMALL_BLOCK * SMALL_BLOCK];
    for (int j = 0; j < n; j++)
        for (int i = j; i < n; i++) T[i + j * n] = V[i + j * n] + a * dV[i + j * n];
    for (int j = 0; j < n; j++) {
        double d = T[j + j * n];
        for (int k = 0; k < j; k++) d -= T[j + k * n] * T[j + k * n];
        if (!(d > 0)) return 0;
        d = sqrt(d); T[j + j * n] = d;
        const double id = 1.0 / d;
        for (int i = j + 1; i < n; i++) {
            double t = T[i + j * n];
            for (int k = 0; k < j; k++) t -= T[i + k * n] * T[j + k * n];
            T[i + j * n] = t * id;
        }
    }
    return 1;
}

/* sgn (E - c I) positive definite? (small blocks, Cholesky on the stack) */
static int small_pd_shift(int n, const double *E, double c, double sgn) {
    double T[SMALL_BLOCK * SMALL_BLOCK];
    for (int j = 0; j < n; j++)
        for (int i = j; i < n; i++) T[i + j * n] = sgn * (E[i + j * n] - (i == j ? c : 0.0));
    for (int j = 0; j < n; j++) {
        double d = T[j + j * n];
        for (int k = 0; k < j; k++) d -= T[j + k * n] * T[j + k * n];
        if (!(d > 0)) return 0;
        d = sqrt(d); T[j + j * n] = d;
        const double id = 1.0 / d;
        for (int i = j + 1; i < n; i++) {
            double t = T[i + j * n];
            for (int k = 0; k < j; k++) t -= T[i + k * n] * T[j + k * n];
            T[i + j * n] = t * id;
        }
    }
    return 1;
}
#define MS_CH 64
static int g_detpar = 1;   /* 4.24: thread-count independent arithmetic in the parallel loops (BRISK_DETPAR=0: 4.23 serial forms at T = 1) */
/* 5.16: the step to the boundary of a vector block (LP: ratio test; second-order cone: the quadratic) */
static double vec_maxstep(const Block *B, const BS *s, int primal) {
    const double *x = primal ? s->X : s->Z, *dx = primal ? s->dX : s->dZ;
    return B->type == BLK_SOC ? soc_maxstep(B->n, x, dx) : lp_maxstep(B->n, x, dx, s->fm);
}
static double maxstep(const Problem *P, BS *S, int primal, int K) {
    double a = 1e30;
    if (!g_par_blk) {
        /* serial: LP blocks first, then a small block whose V + a dV is still definite at
         * the running minimum a cannot lower it (the feasible steps form an interval), and a
         * Cholesky (n^3/6) replaces the eigenvalue computation (~2.3 n^3) */
        static int who = -1;
        if (who < 0) who = getenv("BRISK_STEPWHO") != NULL;
        int kmin = -1, imin = -1;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            if (B->type == BLK_SDP) continue;
            BS *s = &S[k];
            double ak = vec_maxstep(B, s, primal);
            if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
            if (ak < a) {
                a = ak; kmin = k;
                if (who && B->type == BLK_LP) { const double *x = primal ? s->X : s->Z, *dx = primal ? s->dX : s->dZ; for (int i = 0; i < B->n; i++) if (dx[i] < 0 && -x[i] / dx[i] <= ak * (1 + 1e-12)) { imin = i; break; } }
            }
        }
        if (who) {
            double a0 = a;
            for (int k = 0; k < P->nblk; k++) {
                const Block *B = &P->blk[k];
                if (B->type != BLK_SDP) continue;
                BS *s = &S[k];
                const double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
                if (a < 1e29 && B->n <= SMALL_BLOCK && small_pd_at(B->n, V, dV, a)) continue;
                double ak = sdp_maxstep(B->n, primal ? s->LX : s->zon ? ZSP_L : s->LZ, dV, s, K);
                if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
                if (ak < a) { a = ak; kmin = k; imin = -1; }
            }
            const Block *Bm = kmin >= 0 ? &P->blk[kmin] : NULL;
            printf("       [step %s %.3g: %s blk %d n %d idx %d (LP bound %.3g)]\n", primal ? "X" : "Z", a,
                   Bm ? (Bm->type == BLK_LP ? "LP" : "SDP") : "-", kmin, Bm ? Bm->n : 0, imin, a0);
            return a;
        }
        if (g_detpar) goto chunked;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            if (B->type != BLK_SDP) continue;
            BS *s = &S[k];
            const double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
            if (a < 1e29 && B->n <= SMALL_BLOCK && small_pd_at(B->n, V, dV, a)) continue;
            double ak = sdp_maxstep(B->n, primal ? s->LX : s->zon ? ZSP_L : s->LZ, dV, s, K);
            if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
            if (ak < a) a = ak;
        }
        return a;
    }
    a = 1e30;
chunked:
    /* 4.24: parallel form with the result of the serial one for any thread count: the LP
     * minimum first, then fixed chunks of MS_CH SDP blocks, each with the serial running-
     * minimum shortcut started from the LP minimum (the chunking, not the threads, decides
     * which blocks are skipped). Before, every block computed its eigenvalue step (dynamic
     * schedule, 5x the serial time on the 2,568 blocks of TSSOS case1354). */
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_SDP) continue;
        BS *s = &S[k];
        double ak = vec_maxstep(B, s, primal);
        if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
        if (ak < a) a = ak;
    }
    const int nb = P->nblk, nch = (nb + MS_CH - 1) / MS_CH;
    const double a0 = a;
    /* one chunk: no parallel region (same arithmetic). The step lengths are computed about
     * twelve times per iteration; with a single chunk each call forked a team to run one
     * iteration on one thread, and when the other threads' cores were not free at that moment
     * (a spinning BLAS worker of the host, another process) every such fork/join waited for a
     * scheduler slice: step 0.006 -> 0.07..0.4 s on a 56 x 56 block with 462 constraints */
    /* (5.12: no region at all for one chunk: an inactive one made the parallel parts of a
     * large block's step length nested teams; see PBLK_FOR) */
#define MS_CHUNK(c) do {                                                                          \
        double ac = a0;                                                                           \
        const int k1 = ((c) + 1) * MS_CH < nb ? ((c) + 1) * MS_CH : nb;                           \
        for (int k = (c) * MS_CH; k < k1; k++) {                                                  \
            const Block *B = &P->blk[k];                                                          \
            if (B->type != BLK_SDP) continue;                                                     \
            BS *s = &S[k];                                                                        \
            const double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;                \
            if (ac < 1e29 && B->n <= SMALL_BLOCK && small_pd_at(B->n, V, dV, ac)) continue;       \
            double ak = sdp_maxstep(B->n, primal ? s->LX : s->zon ? ZSP_L : s->LZ, dV, s, K);     \
            if (!isfinite(ak) && !(ak > 0)) ak = 0.0;                                             \
            if (ak < ac) ac = ak;                                                                 \
        }                                                                                         \
        if (ac < a) a = ac;                                                                       \
    } while (0)
    if (nch <= 1) { if (nch == 1) MS_CHUNK(0); return a; }
    #pragma omp parallel for schedule(dynamic, 1) reduction(min: a)
    for (int c = 0; c < nch; c++) MS_CHUNK(c);
#undef MS_CHUNK
    return a;
}

/* Take step alpha in place (shrinking it if a Cholesky test fails).
 * X <- X + alpha dX and LX <- chol(X); the previous factor is no longer needed
 * because step lengths have already been computed from it.                    */
static int try_step(const Problem *P, BS *S, int primal, double *alpha) {
    double a = *alpha;
    for (int k = 0; k < P->nblk; k++) S[k].stepped[primal] = 0;
    for (int tries = 0; tries < 40; tries++) {
        int ok = 1;
        PBLK_FOR(k, P->nblk, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n, okk = 1;
            size_t len = bsz(B);
            double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
            double *L = primal ? s->LX : s->LZ;
            double inc = a - s->stepped[primal];
            s->stepped[primal] = a;          /* all blocks stay at the same step */
            if (B->type == BLK_SDP && !(!primal && s->zon)) {
                /* 5.12: the step and the copy for the factor in one pass, on the threads */
                PAR_LEN_FOR(len, const double v = V[i] + inc * dV[i]; V[i] = v; L[i] = v;);
                if (chol_lower(n, L) != 0) okk = 0;
                if (!okk) {
                    _Pragma("omp atomic write")
                    ok = 0;
                }
                continue;
            }
            if (!primal && s->zon && s->zpat) { for (int q = 0; q < s->nzp; q++) { const size_t p_ = s->zpos[q]; V[p_] += inc * dV[p_]; } }
            else {
            PAR_LEN_FOR(len, V[i] += inc * dV[i];);
            }
            if (B->type == BLK_LP) {
                for (int i = 0; i < n; i++) if (!(V[i] > 0) && !(s->fm && s->fm[i])) { okk = 0; break; }
            } else if (B->type == BLK_SOC) {
                if (!soc_interior(n, V)) okk = 0;
                memcpy(L, V, sizeof(double) * len);
            } else if (!primal && s->zon) {                  /* 5.12: the sparse factor of Z */
                schol_gather_dense(s->zf, V);
                if (schol_factor(s->zf, 0.0) != 0) okk = 0;
                if (okk && s->zlz) { pcopy(len, L, V); if (chol_lower(n, L) != 0) okk = 0; }
            } else {
                memcpy(L, V, sizeof(double) * len);
                if (chol_lower(n, L) != 0) okk = 0;
            }
            if (!okk) {
                _Pragma("omp atomic write")
                ok = 0;
            }
        });
        if (ok) {
            *alpha = a;
            return 1;
        }
        a *= 0.8;
    }
    return 0;
}

/* undo the step try_step applied in place (the factor is recomputed by the next try) */
static void step_undo(const Problem *P, BS *S, int primal) {
    for (int k = 0; k < P->nblk; k++) {
        BS *s = &S[k];
        double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
        const size_t len = bsz(&P->blk[k]);
        if (!primal && s->zon && s->zpat) for (int q = 0; q < s->nzp; q++) V[s->zpos[q]] -= s->stepped[primal] * dV[s->zpos[q]];
        else
        for (size_t i = 0; i < len; i++) V[i] -= s->stepped[primal] * dV[i];
        s->stepped[primal] = 0;
    }
}

/* ------------------------------------------------------------------------ */
/* Gondzio-type centrality corrector adapted to SDP (HKM form).
 * Trial point at enlarged steps; eigenvalues of sym(X~ Z~) outside
 * [bmin*mu_t, bmax*mu_t] are pushed back; the correction solves
 *   M dyc = -A(Zi T),  dZc = -A'dyc,  dXc = sym(Zi T) + sym(Zi A'dyc X).
 * Returns 1 if the corrected direction was accepted (dX, dZ, dy updated). */
/* The neighbourhood test of the long step: 1 when lambda_min((X + a dX)(Z + a dZ)) >= thr in every
 * SDP block (three Cholesky-type operations of order n per block: L L' = X + a dX, then
 * L'(Z + a dZ)L - thr I positive definite), and x z >= thr for the LP variables. */
static int sdp_nb_ok(const Problem *P, BS *S, double al, double thr) {
    int ok = 1;
    for (int k = 0; k < P->nblk && ok; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int n = B->n, info = 0;
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) { if (s->fm && s->fm[i]) continue; if (!((s->X[i] + al * s->dX[i]) * (s->Z[i] + al * s->dZ[i]) >= thr)) { ok = 0; break; } }
            continue;
        }
        if (B->type == BLK_SOC) {
            double *t2 = malloc(sizeof(double) * 2 * ((size_t)n + 1));
            if (!(soc_lmin_at(n, s->X, s->dX, s->Z, s->dZ, al, t2) >= thr)) ok = 0;
            free(t2);
            continue;
        }
        const size_t n2 = (size_t)n * n;
        double *L = amalloc(sizeof(double) * n2), *T = amalloc(sizeof(double) * n2);
        for (size_t i = 0; i < n2; i++) { L[i] = s->X[i] + al * s->dX[i]; T[i] = s->Z[i] + al * s->dZ[i]; }
        BL(dpotrf_)("L", &n, L, &n, &info);
        if (!info) {
            BL(dtrmm_)("L", "L", "T", "N", &n, &n, &DONE, L, &n, T, &n);
            BL(dtrmm_)("R", "L", "N", "N", &n, &n, &DONE, L, &n, T, &n);
            for (int i = 0; i < n; i++) T[i + (size_t)i * n] -= thr;
            BL(dpotrf_)("L", &n, T, &n, &info);
        }
        if (info) ok = 0;
        free(L); free(T);
    }
    return ok;
}
static int centrality_corrector(const Problem *P, BS *S, Schur *Sc, const Params *par,
                                double mu_t, double *dy, double *ap, double *ad,
                                double *rhs, double *dyc, Result *R) {
    const int m = P->m, nb = P->nblk;
    const double bmin = 0.1, bmax = 10.0;
    double atp = fmin(1.0, 1.5 * *ap + 0.1), atd = fmin(1.0, 1.5 * *ad + 0.1);
    double td = wtime();
    memset(rhs, 0, sizeof(double) * m);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int n = B->n;
        size_t len = bsz(B);
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) {
                double v = (s->X[i] + atp * s->dX[i]) * (s->Z[i] + atd * s->dZ[i]);
                double t = 0;
                if (v < bmin * mu_t) t = bmin * mu_t - v;
                else if (v > bmax * mu_t) t = fmax(bmax * mu_t - v, -bmax * mu_t);
                s->G[i] = t;
                s->Q[i] = s->Zi[i] * t;
            }
        } else {
            /* the frame of this HKM form (Z dX + dZ X linearized, dX = K - sym(Zi dZ X)):
             * Z = L L', Xh = L' X~ L, Zh = L^{-1} Z~ L^{-T} = I + atd L^{-1} dZ L^{-T};
             * a correction T of sym(Xh Zh) is K = L^{-T} T L^{-1} (4.15 used the X factor
             * here, which does not match the linearization and made the correctors useless) */
            const double *L = s->LZ;
            for (size_t i = 0; i < len; i++) s->Xn[i] = s->X[i] + atp * s->dX[i];
            BL(dtrmm_)("L", "L", "T", "N", &n, &n, &DONE, L, &n, s->Xn, &n);
            BL(dtrmm_)("R", "L", "N", "N", &n, &n, &DONE, L, &n, s->Xn, &n);
            memcpy(s->Zn, s->dZ, sizeof(double) * len);
            xtrsm("L", "L", "N", "N", &n, &n, &atd, L, &n, s->Zn, &n);
            xtrsm("R", "L", "T", "N", &n, &n, &DONE, L, &n, s->Zn, &n);
            for (int i = 0; i < n; i++) s->Zn[i + (size_t)i * n] += 1.0;
            pdgemm("N", "N", &n, &n, &n, &DONE, s->Xn, &n, s->Zn, &n, &DZERO, s->Ct, &n);
            symmetrize(n, s->Ct);
            int lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n, info;
            double *work = malloc(sizeof(double) * lwork);
            int *iwork = malloc(sizeof(int) * liwork);
            BL(dsyevd_)("V", "L", &n, s->Ct, &n, s->ev, work, &lwork, iwork, &liwork, &info);
            free(work); free(iwork);
            if (info != 0) return 0;
            /* T = V diag(t) V' over eigenpairs that need correction */
            memset(s->G, 0, sizeof(double) * len);
            int nt = 0;
            for (int j = 0; j < n; j++) {
                double v = s->ev[j], t = 0;
                if (v < bmin * mu_t) t = bmin * mu_t - v;
                else if (v > bmax * mu_t) t = fmax(bmax * mu_t - v, -bmax * mu_t);
                if (t == 0) continue;
                nt++;
                const double *vj = s->Ct + (size_t)j * n;
                for (int c = 0; c < n; c++) {
                    double f = t * vj[c];
                    double *gc = s->G + (size_t)c * n;
                    for (int r = 0; r < n; r++) gc[r] += f * vj[r];
                }
            }
            if (nt == 0) { memset(s->Q, 0, sizeof(double) * len); }
            else {
                /* Q = K = L^{-T} T L^{-1} */
                memcpy(s->Q, s->G, sizeof(double) * len);
                xtrsm("L", "L", "T", "N", &n, &n, &DONE, L, &n, s->Q, &n);
                xtrsm("R", "L", "N", "N", &n, &n, &DONE, L, &n, s->Q, &n);
            }
        }
        blk_Aop(B, s, s->Q, rhs, -1.0);
    }
    R->t_dense += wtime() - td;
    double tc = wtime();
    schur_solve(Sc, rhs, dyc, 1e-7);
    R->t_chol += wtime() - tc;
    td = wtime();
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        size_t len = bsz(B);
        blk_ATy(B, s, dyc, s->F);
        if (B->type == BLK_LP) {
            for (int i = 0; i < B->n; i++) s->Dxc[i] = s->Q[i] + s->Zi[i] * s->F[i] * s->X[i];
        } else {
            prod_ZFR(B, s, s->F, s->X, s->Dxc);
            for (size_t i = 0; i < len; i++) s->Dxc[i] += s->Q[i];
            symmetrize(B->n, s->Dxc);
        }
        /* keep the uncorrected direction in Xn/Zn, form the candidate in place */
        memcpy(s->Xn, s->dX, sizeof(double) * len);
        memcpy(s->Zn, s->dZ, sizeof(double) * len);
        for (size_t i = 0; i < len; i++) { s->dX[i] += s->Dxc[i]; s->dZ[i] -= s->F[i]; }
    }
    R->t_dense += wtime() - td;
    double ts = wtime();
    double ap1 = fmin(1.0, maxstep(P, S, 1, par->lanczos_k));
    double ad1 = fmin(1.0, maxstep(P, S, 0, par->lanczos_k));
    R->t_step += wtime() - ts;
    if (ENV_ON("BRISK_DEBUG")) {
        double a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        for (int k = 0; k < nb; k++) {
            size_t len = bsz(&P->blk[k]);
            a1 += ddot_n(len, S[k].Xn, S[k].Xn); a2 += ddot_n(len, S[k].Dxc, S[k].Dxc);
            b1 += ddot_n(len, S[k].Zn, S[k].Zn); b2 += ddot_n(len, S[k].F, S[k].F);
        }
        printf("     corrector: (%.3f,%.3f) -> (%.3f,%.3f)  |dX| %.2e |dXc| %.2e |dZ| %.2e |dZc| %.2e mu_t %.2e\n",
               *ap, *ad, ap1, ad1, sqrt(a1), sqrt(a2), sqrt(b1), sqrt(b2), mu_t);
    }
    if (ap1 + ad1 >= *ap + *ad + 0.02 && fmin(ap1, ad1) >= 0.9 * fmin(*ap, *ad)) {
        for (int i = 0; i < m; i++) dy[i] += dyc[i];
        *ap = ap1; *ad = ad1;
        return 1;
    }
    for (int k = 0; k < nb; k++) {
        BS *s = &S[k];
        size_t len = bsz(&P->blk[k]);
        memcpy(s->dX, s->Xn, sizeof(double) * len);
        memcpy(s->dZ, s->Zn, sizeof(double) * len);
    }
    return 0;
}

static void start_sdpt3(const Problem *P, BS *S, double *y) {
    const int nb = P->nblk;
    memset(y, 0, sizeof(double) * P->m);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int n = B->n;
        double nC = 0, maxA = 0, ratio = 0;
        for (int q = 0; q < B->C.ef; q++) nC += B->C.fv[q] * B->C.fv[q];
        nC = sqrt(nC);
        for (int t = 0; t < B->ncon; t++) {
            double a = 0;
            for (int q = 0; q < B->A[t].ef; q++) a += B->A[t].fv[q] * B->A[t].fv[q];
            a = sqrt(a);
            if (a > maxA) maxA = a;
            double rr = (1 + fabs(P->b[B->con[t]])) / (1 + a);
            if (rr > ratio) ratio = rr;
        }
        double xi = fmax(10.0, fmax(sqrt((double)n), n * ratio));
        double eta = fmax(10.0, fmax(sqrt((double)n), fmax(maxA, nC)));
        size_t len = bsz(B);
        memset(s->X, 0, sizeof(double) * len);
        memset(s->Z, 0, sizeof(double) * len);
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) { s->X[i] = xi; s->Z[i] = eta; }
        } else if (B->type == BLK_SOC) {
            s->X[0] = xi; s->Z[0] = eta;               /* xi e, eta e (SDPT3's start for a cone) */
            memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len);
        } else {
            for (int i = 0; i < n; i++) { s->X[i + (size_t)i * n] = xi; s->Z[i + (size_t)i * n] = eta; }
            /* 5.12: the factors of the diagonal starts, without a dense factorization (the same
             * entries: chol_lower of order 5000 took 0.25 s each) */
            pzero(len, s->LX); pzero(len, s->LZ);
            const double sx_ = sqrt(xi), sz_ = sqrt(eta);
            for (int i = 0; i < n; i++) { s->LX[i + (size_t)i * n] = sx_; s->LZ[i + (size_t)i * n] = sz_; }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* out = A(A'(v)) without forming A A'. Sparse SDP blocks touch only their
 * pattern (F is kept zero outside it between calls).                         */
static void gram_mv_blk(const Block *B, BS *s, const double *v, double *out) {
    if (B->type != BLK_SDP || B->nd > 0) {
        blk_ATy(B, s, v, s->F);
        blk_Aop(B, s, s->F, out, 1.0);
        memset(s->F, 0, sizeof(double) * bsz(B));
        return;
    }
    const int n = B->n;
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *A = &B->A[t];
        double vi = v[B->con[t]];
        for (int q = 0; q < A->ef; q++) s->F[A->fr[q] + (size_t)A->fc[q] * n] += vi * A->fv[q];
    }
    blk_Aop(B, s, s->F, out, 1.0);
    for (int k2 = 0; k2 < s->upat.nf; k2++) s->F[s->upat.fr[k2] + (size_t)s->upat.fc[k2] * n] = 0.0;
}
/* 18 October 2026: the blocks on the threads when the order-preserving map of A(.) is the problem's (the
 * embedding's polishing; aop_begin/aop_end: each block's values in its own slots, then added per constraint in
 * block order, the additions of the serial loop): mater-5's 2440 blocks, the Gram CG of the polish was a third of
 * the run on one thread. BRISK_GRAMPAR=0: serial. */
static void gram_mv(const Problem *P, BS *S, const double *v, double *out) {
    memset(out, 0, sizeof(double) * P->m);
    static int env = -1; if (env < 0) { const char *e = getenv("BRISK_GRAMPAR"); env = e ? atoi(e) : 1; }
    const int nb = P->nblk;
    if (env && nb >= 16 && nthreads() > 1 && !in_par() && g_am.blk0 == P->blk && g_am.nb == nb && g_am.m == P->m && g_am.boff) {
        g_aop_out[0] = out; g_aop_out[1] = g_aop_out[2] = g_aop_out[3] = NULL; g_aop_on = 1;
        #pragma omp parallel for schedule(dynamic, 16)
        for (int k = 0; k < nb; k++) gram_mv_blk(&P->blk[k], &S[k], v, out);
        aop_end();
        return;
    }
    for (int k = 0; k < nb; k++) gram_mv_blk(&P->blk[k], &S[k], v, out);
}

/* ||A A' x - rhs|| / ||rhs|| (q: m scratch; each block's F is the products' scratch) */
static double gram_resid(const Problem *P, BS *S, const double *x, const double *rhs, double *q) {
    const int m = P->m;
    for (int k = 0; k < P->nblk; k++) memset(S[k].F, 0, sizeof(double) * bsz(&P->blk[k]));
    gram_mv(P, S, x, q);
    double rr = 0, bb = 0;
    for (int i = 0; i < m; i++) { const double d = q[i] - rhs[i]; rr += d * d; bb += rhs[i] * rhs[i]; }
    return bb > 0 ? sqrt(rr / bb) : sqrt(rr);
}
/* CG on (A A') x = rhs; returns the final relative residual. */
static double gram_cg(const Problem *P, BS *S, const double *rhs, double *x, int maxit, double tol,
                      double *r, double *p, double *q) {
    const int m = P->m;
    for (int k = 0; k < P->nblk; k++) memset(S[k].F, 0, sizeof(double) * bsz(&P->blk[k]));
    double nb = sqrt(ddot_n(m, rhs, rhs));
    memset(x, 0, sizeof(double) * m);
    if (nb == 0) return 0;
    memcpy(r, rhs, sizeof(double) * m);
    memcpy(p, r, sizeof(double) * m);
    double rr = ddot_n(m, r, r);
    for (int it = 0; it < maxit && sqrt(rr) > tol * nb; it++) {
        gram_mv(P, S, p, q);
        double pq = ddot_n(m, p, q);
        if (!(pq > 0)) break;
        double a = rr / pq;
        for (int i = 0; i < m; i++) { x[i] += a * p[i]; r[i] -= a * q[i]; }
        double rr2 = ddot_n(m, r, r);
        for (int i = 0; i < m; i++) p[i] = r[i] + (rr2 / rr) * p[i];
        rr = rr2;
    }
    return sqrt(rr) / nb;
}


/* 4.27: gram_cg for the polish and its probe, up to gram_maxit() steps (was 500: roa_vdp_d10_I
 * needs ~800 and stopped at 5e-7), stopped once 200 steps have not halved the residual (an
 * ill-conditioned A A' that CG will not solve in the budget either) */
static double gram_cg_polish(const Problem *P, BS *S, const double *rhs, double *x, double tol,
                             double abs_tol, double *r, double *p, double *q);
static int gram_maxit(void) { static int v = -1; if (v < 0) { const char *e = getenv("BRISK_GRAMIT"); v = e ? atoi(e) : 2000; } return v; }
/* 23 October 2026: the CG steps a polish probe may take inside the iterations, worth about one iteration's work (the
 * factorization and the assembly at the BLAS-3 rate against a CG step's entries at the scattered rate, as for the start,
 * set with it; 0: no cap). The probes, up to four a run, had 500 to 2000 steps whatever an iteration costs: metric_50,
 * the owner's benchmark, SDP, m = 4,000: about 14 s of probes in a 35 s run that converged without them.
 * BRISK_PROBECAP=0: no cap */
static int g_probe_cgcap = 0, g_gram_cap = 0;
static double gram_cg_polish(const Problem *P, BS *S, const double *rhs, double *x, double tol,
                             double abs_tol, double *r, double *p, double *q) {
    const int m = P->m, maxit = g_gram_cap > 0 && g_gram_cap < gram_maxit() ? g_gram_cap : gram_maxit();
    for (int k = 0; k < P->nblk; k++) memset(S[k].F, 0, sizeof(double) * bsz(&P->blk[k]));
    double nb = sqrt(ddot_n(m, rhs, rhs));
    memset(x, 0, sizeof(double) * m);
    if (nb == 0) return 0;
    memcpy(r, rhs, sizeof(double) * m);
    memcpy(p, r, sizeof(double) * m);
    double rr = ddot_n(m, r, r), h[3] = { INFINITY, INFINITY, INFINITY };
    const double stop = fmax(tol * nb, abs_tol);
    for (int it = 0; it < maxit && sqrt(rr) > stop; it++) {
        if (it % 100 == 0) {
            h[0] = h[1]; h[1] = h[2]; h[2] = sqrt(rr);
            /* past the old cap of 500 steps only when nearly there and still converging */
            if (it >= 500 && (!(h[2] < 0.5 * h[0]) || h[2] > 1e-6 * nb)) break;
        }
        gram_mv(P, S, p, q);
        double pq = ddot_n(m, p, q);
        if (!(pq > 0)) break;
        double a = rr / pq;
        for (int i = 0; i < m; i++) { x[i] += a * p[i]; r[i] -= a * q[i]; }
        double rr2 = ddot_n(m, r, r);
        for (int i = 0; i < m; i++) p[i] = r[i] + (rr2 / rr) * p[i];
        rr = rr2;
    }
    return sqrt(rr) / nb;
}

/* lambda_min of a semidefinite block of a candidate whose Lanczos estimate est is negative, measured as the
 * postsolve measures the point on the problem as read (ps_lammin_tol: 0 when the Cholesky succeeds, -delta when
 * that of M + delta I does, else the value itself, by a Cholesky ladder with bisection or the eigenvalue).
 * 24 October 2026: the estimate alone stood here when the Cholesky failed, and it is neither converged nor a bound
 * (the postsolve's lesson of 4.42): 30 Lanczos steps on an ill-conditioned block leave a residual up to 1e-14 times
 * its largest eigenvalue, and the estimate subtracts it. On sat_roa_n96 (control LMIs, SDP, no presolve applied) a
 * polished point was rejected at 2.3e-6 that is 1.2e-8 on the problem as read (X >= 0 1.2e-8), and only the
 * judgment on the problem as read (ISSUES 266) returned it. delta is a thousandth of the default tolerance (1e-11 in
 * the DIMACS units of errors 2 and 4), so a violation below it changes no comparison of candidates. Blocks above order
 * 4000 test only M + delta I and keep the estimate otherwise (the ladder costs up to twelve factorizations of the
 * block per candidate there). */
static double cand_lammin(int n, const double *M, double *work, double est, double delta) {
    if (n <= 4000) return ps_lammin_tol(n, M, work, delta);
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    if (brisk_dpotrf(n, work, n) == 0) return 0.0;
    memcpy(work, M, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) work[i + (size_t)i * n] += delta;
    if (brisk_dpotrf(n, work, n) == 0) return -delta;
    return est;
}

/* All six DIMACS measures of a candidate (X, y, Z) in scaled units (X, Z given as
 * per-block arrays). Unlike the iterates, a polished candidate need not be
 * positive definite, so errors 2 and 4 are computed (cand_lammin where the
 * Cholesky fails).  Scratch: s->W, s->F.  Returns the overall score.          */
static double measure_candidate(const Problem *P, BS *S, double **Xs, double **Zs, const double *y,
                                Result *R, double *rp) {
    const int m = P->m, nb = P->nblk;
    memcpy(rp, P->b, sizeof(double) * m);
    double pobj = 0, xz = 0, rd2 = 0, lamx = 1e300, lamz = 1e300;
    double save_tol = lz_tol;
    lz_tol = 1e-4;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        size_t len = bsz(B);
        blk_Aop(B, s, Xs[k], rp, -1.0);
        blk_ATy(B, s, y, s->W);
        for (size_t i = 0; i < len; i++) s->W[i] = -s->W[i] - Zs[k][i];
        sp_add(B, &B->C, 1.0, s->W);                      /* Rd = C - A'y - Z */
        rd2 += ddot_n(len, s->W, s->W);
        pobj += sp_inner(B, &B->C, Xs[k]);
        xz += ddot_n(len, Xs[k], Zs[k]);
        if (B->type == BLK_LP) {
            for (int i = 0; i < B->n; i++) { lamx = fmin(lamx, Xs[k][i]); lamz = fmin(lamz, Zs[k][i]); }
        } else if (B->type == BLK_SOC) {        /* 23 October 2026: the smaller spectral value x_0 - ||xb|| */
            lamx = fmin(lamx, Xs[k][0] - soc_nb(B->n, Xs[k])); lamz = fmin(lamz, Zs[k][0] - soc_nb(B->n, Zs[k]));
        } else {
            /* The Lanczos estimate (Ritz value minus residual) can turn slightly negative for an
             * ill-conditioned but positive definite matrix, so a successful Cholesky overrides it
             * (it proves definiteness in floating point). When the Cholesky fails, the value is
             * measured as on the problem as read (cand_lammin) */
            double lx, lz;
            const int n = B->n;
            sdp_maxstep_lam(n, NULL, Xs[k], s, LANCZOS_K, &lx);
            sdp_maxstep_lam(n, NULL, Zs[k], s, LANCZOS_K, &lz);
            if (lx < 0) lx = cand_lammin(n, Xs[k], s->W, lx, 1e-11 * (1 + P->normb1) / P->bs);
            if (lz < 0) lz = cand_lammin(n, Zs[k], s->W, lz, 1e-11 * (1 + P->normC1) / P->cs);
            lamx = fmin(lamx, lx); lamz = fmin(lamz, lz);
        }
    }
    lz_tol = save_tol;
    double dobj = ddot_n(m, P->b, y);
    double rpo = 0;
    for (int i = 0; i < m; i++) { double v = rp[i] * P->du[i]; rpo += v * v; }
    rpo = sqrt(rpo);
    double rdo = P->cs * sqrt(rd2), sc = P->bs * P->cs;
    double po = sc * pobj, dob = sc * dobj, den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
    R->pobj = po; R->dobj = dob;
    R->pinf = rpo / (1 + P->normb2);
    R->dinf = rdo / (1 + P->normC2);
    R->relgap = fabs(po - dob) / den;
    R->relcomp = fabs(sc * xz) / den;
    R->err[1] = rpo / (1 + P->normb1);
    R->err[2] = fmax(0.0, -lamx * P->bs) / (1 + P->normb1);
    R->err[3] = rdo / (1 + P->normC1);
    R->err[4] = fmax(0.0, -lamz * P->cs) / (1 + P->normC1);
    R->err[5] = (po - dob) / den;
    R->err[6] = sc * xz / den;
    return fmax(fmax(fmax(R->relgap, R->relcomp), fmax(R->pinf, R->dinf)), fmax(R->err[2], R->err[4]));
}

/* Polishing: project X onto {A(X) = b} (Gram-matrix CG, well-conditioned after row
 * scaling) and recompute Z = C - A'y exactly. Accepted only if the full measure
 * improves. Candidates are written to s->G (X) and s->Q (Z).                    */
/* 5.16: the double-precision measure handed to the dd endgame (ddend.c): it keeps the iterate that is
 * best by this measure, which is what the solver reports, not by its own dd score */
typedef struct { const Problem *P; BS *S; Result R; double *rp; } DdMeasCtx;
static double dd_meas_cb(void *c, double **X, double **Z, const double *y, double *rest) {
    DdMeasCtx *ctx = (DdMeasCtx *)c; Result Rt = ctx->R;
    const double v = measure_candidate(ctx->P, ctx->S, X, Z, y, &Rt, ctx->rp);
    if (rest) *rest = fmax(fmax(fmax(Rt.relgap, Rt.relcomp), Rt.dinf), fmax(Rt.err[2], Rt.err[4]));
    return v;
}
static void rej_offer(const Problem *P, double *const *Xk, const double *y, const Result *R, double score);   /* (ISSUES 266) */
static int polish_solution(const Problem *P, BS *S, double **Xb, double **Zb, const double *yb,
                           Result *R, double score_now, Schur *Sc, double *w, int verbose) {
    const int m = P->m, nb = P->nblk;
    memcpy(Sc->t, P->b, sizeof(double) * m);
    for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], Xb[k], Sc->t, -1.0);
    const double tg0 = wtime();
    double rres = gram_cg_polish(P, S, Sc->t, w, 1e-15, 0.0, Sc->r, Sc->p, Sc->q);
    if (getenv("BRISK_GRAMDBG")) printf("   [gram_cg: residual %.2e in %.3fs]\n", rres, wtime() - tg0);
    double **Xs = malloc(sizeof(double *) * nb), **Zs = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        size_t len = bsz(B);
        blk_ATy(B, s, w, s->G);
        for (size_t i = 0; i < len; i++) s->G[i] += Xb[k][i];
        blk_ATy(B, s, yb, s->Q);
        for (size_t i = 0; i < len; i++) s->Q[i] = -s->Q[i];
        sp_add(B, &B->C, 1.0, s->Q);
        Xs[k] = s->G; Zs[k] = s->Q;
    }
    Result Rp = *R;
    double sp = measure_candidate(P, S, Xs, Zs, yb, &Rp, Sc->t);
    if (!(sp < score_now)) rej_offer(P, Xs, yb, &Rp, sp);
    free(Xs); free(Zs);
    if (verbose > 0)
        printf("   polishing: score %.2e -> %.2e (Gram CG residual %.1e) %s\n",
               score_now, sp, rres, sp < score_now ? "accepted" : "rejected");
    if (sp < score_now) {
        double tt[6] = { R->t_setup, R->t_schur, R->t_chol, R->t_dense, R->t_step, R->t_total };
        int it = R->iters;
        *R = Rp;
        R->t_setup = tt[0]; R->t_schur = tt[1]; R->t_chol = tt[2]; R->t_dense = tt[3]; R->t_step = tt[4];
        R->t_total = tt[5]; R->iters = it;
        return 1;
    }
    return 0;
}


/* 4.27: the Euclidean polish of the current embedding iterate (X/tau, y/tau), measured but
 * not committed: the stopping probe of hsd_run. Scratch: Sc->t, r, p, q; s->F (zeroed by
 * gram_cg: the caller restores it), s->G, s->Q (the candidate), s->W; yt, w (m each).  */
/* 4.28: endgame stop rules shared by both methods (BRISK_ENDRULES=0 turns them off).
 * Replayed on the per-iteration score traces of the 186-instance regression set before they
 * were adopted (21 run segments stopped earlier, 4.8% of the time, no later iterate beating
 * the stopping point's best or changing its class at 1e-6 or tol); 190-instance regression:
 * the 19 changed runs 488 -> 378 s, no class change. Unlike the older stall rules they do not depend on the score level:
 *  - collapse: three steps in a row shorter than 0.05, none of them giving a new best
 *    (roa_vdp_inner_d14_K: 14 steps of alpha ~0 after its best, kept alive by the
 *    "mu still falling" exception of the embedding's stall rule);
 *  - creep (best score < 1e-3): four steps shorter than 0.25 in which the best score did not
 *    halve (neu2c's standard-method re-solve crawling at 1e-3 when the first attempt had
 *    reached 1.7e-6; sensor_500's embedding at 1.6e-8). Suspended within 10x above a
 *    re-solve threshold and where the double-double endgame follows (endrules_init). */
typedef struct { int on, col, creep_on; double retry_at, floor_at; double *bh, *ah; } EndRules;
/* dd_small: the double-double endgame may follow (m <= dd_maxm, small blocks). There the
 * creep rule is off: a crawl still improves the endgame's starting point (hinf15: creep
 * stopped at 3.5e-4 instead of 2.8e-4 at iteration 32, and the endgame reached 1.2e-6
 * instead of 3e-7); a collapse (alpha < 0.05) does not */
/* retry_at: the score above which this run's result is re-solved with the other method (0:
 * no re-solve follows). Creeping within 10x above it is not stopped: continuing may still
 * cross it and save the whole re-solve (neu1 under BRISK_PERTURB=1: the embedding crept to
 * 7.9e-7 in 25 iterations, 17 s; stopped at 2e-6, the re-solve took the run to 33 s). */
static void endrules_init(EndRules *E, int dd_small, double retry_at) {
    const char *e = getenv("BRISK_ENDRULES");
    E->on = e ? atoi(e) : 1; E->col = 0; E->creep_on = !dd_small; E->retry_at = retry_at; E->floor_at = 0;
    E->bh = E->on ? malloc(sizeof(double) * 4096) : NULL; E->ah = E->on ? malloc(sizeof(double) * 4096) : NULL;
}
/* 15 October 2026 (the owner's benchmark, ISSUES 99): the endgame's size cap on the sum of the squared block orders
 * follows from the memory (eleven double-double arrays of n^2 per block in ddend.c, four double copies: about 210
 * bytes per entry, allowed a quarter of the limit), the work budget decides the rest; -ddmaxn2 sets it by hand. The
 * old cap of 2e5 (a block of order about 450) skipped trto4, vibra3 and shmup3 without a word, which -ddfactor 4
 * brings to OPTIMAL (the owner's measurements with the cap lifted). */
static double dd_n2_cap(const Params *par) {
    if (par->dd_maxn2 > 0) return par->dd_maxn2;
    extern double brisk_mem_limit(void);
    const double lim = brisk_mem_limit();
    return fmax(2.0e5, (lim > 0 ? lim : 8e9) / 4.0 / 210.0);
}
/* (the creep rule's suspension, endrules_init: where the endgame follows at the default budget, which a
 * sum of squared orders of 2e5 bounds in practice: a dd iteration of a block of order 450 costs 5e10
 * flop units, three of them the whole floor of the budget; the endgame's own gate is dd_n2_cap) */
static int dd_small_problem(const Problem *P, const Params *par) {
    double n2 = 0;
    for (int k = 0; k < P->nblk; k++) n2 += (double)bsz(&P->blk[k]);
    return par->dd_end && !par->dd_done && P->m <= par->dd_maxm && n2 <= fmin(2.0e5, dd_n2_cap(par));
}
static void endrules_free(EndRules *E) { free(E->bh); free(E->ah); E->bh = E->ah = NULL; }
/* it: this iterate; a: the step that produced it (0 at it = 0); improved: it set a new best;
 * best: the best score including this iterate. Returns 1 (collapse) or 2 (creep) to stop. */
static int endrules_check(EndRules *E, int it, double a, int improved, double best) {
    if (!E->on || it >= 4096) return 0;
    E->bh[it] = best; E->ah[it] = a;
    E->col = (it > 0 && a < 0.05 && !improved) ? E->col + 1 : 0;
    if (E->col >= 3) return 1;
    /* 4.29 floor: below the default tolerance (only when a tighter one was asked for), four
     * iterations that do not halve the best score end the run whatever the step length */
    if (E->floor_at > 0 && it >= 4 && best <= E->floor_at && best > 0.5 * E->bh[it - 4]) return 3;
    if (E->creep_on && it >= 4 && best < 1e-3 && !(E->retry_at > 0 && best > E->retry_at && best <= 10.0 * E->retry_at)) {
        for (int q = it - 3; q <= it; q++) if (!(E->ah[q] < 0.25)) return 0;
        if (best > 0.5 * E->bh[it - 4]) return 2;
    }
    return 0;
}

/* 5.16 (experiment, BRISK_HSDDUMP=<file>): at the first refinement solve at the floor (rel_e > 0.5),
 * the Schur system of the refinement is written for the square-root-solve microbenchmark
 * (tools in the session's scratchpad): the blocks' NT factors G (W = G G'), their constraint
 * matrices as the solver holds them, the right-hand side e and the double solution */
static void hsd_dump_floor(const char *fn, const Problem *P, BS *S, int m, const double *e, const double *u, int it) {
    FILE *f = fopen(fn, "wb"); if (!f) return;
    int hdr[3] = { m, P->nblk, it }; fwrite(hdr, sizeof(int), 3, f);
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k]; BS *s = &S[k];
        const size_t len = B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
        int h[4] = { B->type, B->n, B->ncon, s->ntGbuf != NULL }; fwrite(h, sizeof(int), 4, f);
        fwrite(s->X, sizeof(double), len, f); fwrite(s->Z, sizeof(double), len, f); fwrite(s->Zi, sizeof(double), len, f);
        fwrite(B->con, sizeof(int), B->ncon, f);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t];
            fwrite(&A->nnz, sizeof(int), 1, f);
            fwrite(A->row, sizeof(int), A->nnz, f); fwrite(A->col, sizeof(int), A->nnz, f); fwrite(A->val, sizeof(double), A->nnz, f);
        }
    }
    fwrite(e, sizeof(double), m, f); fwrite(u, sizeof(double), m, f);
    fclose(f);
    printf("   [HSDDUMP: refinement system of iteration %d written to %s]\n", it, fn);
}
/* 5.16 (experiment, BRISK_HSDQR=1: after the solve floor has been met, 2: from the first iteration):
 * the square-root solve. The Schur matrix is a Gram matrix, M_ij = <g_i, g_j> with
 * g_i = vec(L_X' A_i R_Z) (HKM: X = L_X L_X', Z^-1 = R_Z R_Z') or vec(G' A_i G) (NT: W = G G').
 * The Cholesky factor of D M D (+ reg I) is the triangular factor of a QR of the N x m matrix
 * [G D; sqrt(reg) I]: its error is that of a perturbation of the vectors, kappa(G) u =
 * sqrt(kappa(M)) u, where the Cholesky of the assembled M carries kappa(M) u (a solve that has
 * no digits at kappa(M) ~ 1e18 keeps 1e-9 here; the microbenchmark of REPORT_5.16). The QR
 * runs over chunks of rows stacked under the current R (dgeqrf on [R; chunk]). Dense Schur
 * path only. Returns 0 when done, 1 when not applicable. */
static int g_qr_n = 0; static double g_qr_t = 0;
extern void BL(dgeqrf_)(const int *, const int *, double *, const int *, double *, double *, const int *, int *);
static int hsd_qr_factor(const Problem *P, BS *S, Schur *Sc, double **Zv) {
    const int m = Sc->m, nb = P->nblk;
    if (Sc->SC || Sc->E || Sc->bF || Sc->is_float || !Sc->Mc || !Sc->D) return 1;
    size_t N = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_SOC) return 1;
        N += B->type == BLK_LP ? (size_t)B->n : (size_t)B->n * B->n;
    }
    const double t0 = wtime();
    double *G = calloc(N * (size_t)m, sizeof(double));
    if (!G) return 1;
    size_t off = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; BS *s = &S[k];
        const int n = B->n;
        if (B->type == BLK_LP) {
            for (int t = 0; t < B->ncon; t++) {
                const SpSym *A = &B->A[t]; const int i = B->con[t];
                double *g = G + off + (size_t)i * N;
                for (int q = 0; q < A->nnz; q++) { const int r = A->row[q]; g[r] = A->val[q] * sqrt(s->X[r] / s->Z[r]) * Sc->D[i]; }
            }
            off += (size_t)n; continue;
        }
        const int nt = Zv && Zv[k] && s->ntGbuf;
        double *LX = NULL, *RZ = NULL;
        if (!nt) {
            int info;
            LX = malloc(sizeof(double) * (size_t)n * n); RZ = malloc(sizeof(double) * (size_t)n * n);
            memcpy(LX, s->X, sizeof(double) * (size_t)n * n); memcpy(RZ, s->Zi, sizeof(double) * (size_t)n * n);
            BL(dpotrf_)("L", &n, LX, &n, &info); if (info) { free(LX); free(RZ); free(G); return 1; }
            BL(dpotrf_)("L", &n, RZ, &n, &info); if (info) { free(LX); free(RZ); free(G); return 1; }
        }
        double *Ad = malloc(sizeof(double) * (size_t)n * n), *T1 = malloc(sizeof(double) * (size_t)n * n);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *A = &B->A[t]; const int i = B->con[t];
            memset(Ad, 0, sizeof(double) * (size_t)n * n);
            for (int q = 0; q < A->nnz; q++) { const int r = A->row[q], c = A->col[q]; Ad[r + (size_t)c * n] = A->val[q]; Ad[c + (size_t)r * n] = A->val[q]; }
            double *T;
            if (nt) {   /* T = G' A G */
                BL(dsymm_)("L", "L", &n, &n, &DONE, Ad, &n, s->ntGbuf, &n, &DZERO, T1, &n);
                BL(dgemm_)("T", "N", &n, &n, &n, &DONE, s->ntGbuf, &n, T1, &n, &DZERO, Ad, &n);
                T = Ad;
            } else {    /* T = L_X' A R_Z */
                BL(dtrmm_)("L", "L", "T", "N", &n, &n, &DONE, LX, &n, Ad, &n);
                BL(dtrmm_)("R", "L", "N", "N", &n, &n, &DONE, RZ, &n, Ad, &n);
                T = Ad;
            }
            double *g = G + off + (size_t)i * N; const double d = Sc->D[i];
            for (size_t p = 0; p < (size_t)n * n; p++) g[p] = T[p] * d;
        }
        free(Ad); free(T1); free(LX); free(RZ);
        off += (size_t)n * n;
    }
    if (ENV_ON("BRISK_HSDQRDBG")) {
        for (int i = 0; i < (m < 3 ? m : 3); i++) {
            double gg = 0; for (size_t p = 0; p < N; p++) gg += G[p + (size_t)i * N] * G[p + (size_t)i * N];
            printf("     [HSDQR diag %d: M_ii D_i^2 = %.6e (Md %.3e, D %.3e), |g_i|^2 = %.6e, Mc_ii %.3e]\n", i, Sc->Md[i] * Sc->D[i] * Sc->D[i], Sc->Md[i], Sc->D[i], gg, Sc->Mc[i + (size_t)i * m]);
        }
    }
    /* the QR by row chunks: [R; chunk] -> R */
    const int c = (int)fmin((double)N, fmax((double)m, 512.0));
    const int ld = m + c;
    double *St = calloc((size_t)ld * m, sizeof(double)), *tau = malloc(sizeof(double) * m);
    int lwork = -1, info; double wq;
    BL(dgeqrf_)(&ld, &m, St, &ld, tau, &wq, &lwork, &info);
    lwork = (int)wq + 1; double *work = malloc(sizeof(double) * lwork);
    const double sreg = Sc->reg > 0 ? sqrt(Sc->reg) : 0.0;
    const size_t Nreg = N + (sreg > 0 ? (size_t)m : 0);
    for (size_t r0 = 0; r0 < Nreg; r0 += c) {
        const int rows = (int)fmin((double)c, (double)(Nreg - r0));
        const int tot = m + rows;
        /* keep R (upper m x m) in place, zero below the diagonal, load the chunk */
        for (int j = 0; j < m; j++) {
            double *col = St + (size_t)j * ld;
            for (int i = j + 1; i < m; i++) col[i] = 0.0;
            for (int i = 0; i < rows; i++) {
                const size_t r = r0 + i;
                col[m + i] = r < N ? G[r + (size_t)j * N] : (r - N == (size_t)j ? sreg : 0.0);
            }
        }
        BL(dgeqrf_)(&tot, &m, St, &ld, tau, work, &lwork, &info);
        if (info) { free(St); free(tau); free(work); free(G); return 1; }
    }
    /* the factor: Mc (lower) = R' */
    for (int j = 0; j < m; j++)   /* the lower triangle only: the upper one holds M */
        for (int i = j; i < m; i++) Sc->Mc[i + (size_t)j * m] = St[j + (size_t)i * ld];
    if (ENV_ON("BRISK_HSDQRDBG")) {
        /* check: M x against D^-1 (R'R - reg I) D^-1 x on a random x */
        double *x = malloc(sizeof(double) * m), *y1 = malloc(sizeof(double) * m), *y2 = malloc(sizeof(double) * m);
        unsigned sd = 12345u;
        for (int i = 0; i < m; i++) { sd = sd * 1103515245u + 12345u; x[i] = ((sd >> 8) & 0xffff) / 65536.0 - 0.5; }
        schur_mv(Sc, 1.0, x, 0.0, y1);
        for (int i = 0; i < m; i++) y2[i] = x[i] / Sc->D[i];          /* R'R = D M D + reg I: M x = D^-1 (R'R - reg I) D^-1 x */
        BL(dtrmv_)("L", "T", "N", &m, Sc->Mc, &m, y2, &IONE);   /* R z */
        BL(dtrmv_)("L", "N", "N", &m, Sc->Mc, &m, y2, &IONE);   /* R' (R z) */
        double nd = 0, ny = 0;
        for (int i = 0; i < m; i++) { y2[i] = (y2[i] - Sc->reg * x[i] / Sc->D[i]) / Sc->D[i]; nd += (y2[i] - y1[i]) * (y2[i] - y1[i]); ny += y1[i] * y1[i]; }
        double rf = 0, nx = 0; for (int j = 0; j < m; j++) for (int i = j; i < m; i++) rf += Sc->Mc[i + (size_t)j * m] * Sc->Mc[i + (size_t)j * m];
        for (int i = 0; i < m; i++) nx += x[i] * x[i];
        printf("     [HSDQR: m %d, N %zu, chunk %d, reg %.0e, |R'R - M x| / |M x| = %.2e (|R|_F %.3e, |x| %.3e, |Mx| %.3e, |R'Rx| %.3e, mv_ok %d, mT_ok %d, low_M %d), %.3f s]\n", m, N, c, Sc->reg, sqrt(nd / ny), sqrt(rf), sqrt(nx), sqrt(ny), sqrt(nd), Sc->mv_ok, Sc->mT_ok, Sc->low_M, wtime() - t0);
        free(x); free(y1); free(y2);
    }
    free(St); free(tau); free(work);
    { static int nref = -1; if (nref < 0) { const char *e = getenv("BRISK_HSDQRREF"); nref = e ? atoi(e) : 0; }
      free(Sc->qrG); Sc->qrG = NULL; Sc->qrN = 0;
      if (nref > 0) { Sc->qrG = G; Sc->qrN = N; } else free(G); }
    Sc->qr_fact = 1; Sc->min_pivot = 1.0; Sc->trust = 0;
    g_qr_n++; g_qr_t += wtime() - t0;
    return 0;
}
static int g_epol_stop = 0;   /* hsd_run stopped on a passing polish probe */
static int g_objcand = 0;     /* hsd_run returned its objective candidate (BRISK_OBJBEST) */
static int g_oc_fits = 0;     /* the memory cap of the best iterate leaves room for the candidate's copies (set per solve) */
static int g_has_soc = 0;     /* 5.16: the problem of this solve has second-order cone blocks */
static int g_epol_fb = 0;     /* ... or returned a fallback probe candidate (tol below 1e-8) */
static int g_drift_held = 0;  /* 20 October 2026: the drift guard held a tolerance stop back in the last embedding run */
static Result g_epol_R;       /* ... and the measures of its candidate */
static double probe_polish(const Problem *P, BS *S, double tau, const double *y, Schur *Sc,
                           double *yt, double *w, Result *Rp, double *rres, double tol) {
    const int m = P->m, nb = P->nblk;
    const double it = 1.0 / tau;
    for (int i = 0; i < m; i++) yt[i] = y[i] * it;
    memcpy(Sc->t, P->b, sizeof(double) * m);
    for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].X, Sc->t, -it);
    /* enough for the probe: a leftover primal residual 1e-3 tol in the original units
     * (r_orig_i = r_i bs / d_i, pinf = |r_orig| / (1 + |b|)) */
    {   /* 5.12: first the correction in the metric of the last NT scaling, X <- X + W A'(u) W with
         * A (W (x) W) A' u = b - A(X): the matrix is the Schur complement just factored (one solve,
         * no assembly). Like the X-metric correction it moves X along its range and leaves its
         * near-null directions alone, where the Euclidean correction below breaks X >= 0 on moment
         * relaxations (|y| large: the complementarity <X, C - A'y> lags by y'r_p). BRISK_WPOL=0 off. */
        static int wpol = -1; if (wpol < 0) { const char *e = getenv("BRISK_WPOL"); wpol = e ? atoi(e) : 1; }
        int wok = wpol;
        for (int k = 0; k < nb && wok; k++) if (P->blk[k].type != BLK_SDP || !S[k].ntGold) wok = 0;
        if (wok) {
            double *rhs = malloc(sizeof(double) * (m + 1));
            memcpy(rhs, Sc->t, sizeof(double) * m);
            schur_solve(Sc, rhs, w, 1e-14);
            double **Xs = malloc(sizeof(double *) * nb), **Zs = malloc(sizeof(double *) * nb);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                const int n = B->n;
                const size_t len = (size_t)n * n;
                double *T = amalloc(sizeof(double) * (len + 1));
                blk_ATy(B, s, w, s->Q);                                                                    /* A'(u) */
                xsymm("L", "L", &n, &n, &DONE, s->Q, &n, s->ntGold, &n, &DZERO, T, &n);                 /* A'(u) G */
                pdgemm("T", "N", &n, &n, &n, &DONE, s->ntGold, &n, T, &n, &DZERO, s->G, &n);            /* G' A'(u) G */
                pdgemm("N", "N", &n, &n, &n, &DONE, s->ntGold, &n, s->G, &n, &DZERO, T, &n);
                pdgemm("N", "T", &n, &n, &n, &DONE, T, &n, s->ntGold, &n, &DZERO, s->G, &n);            /* W A'(u) W */
                symmetrize(n, s->G);
                for (size_t i = 0; i < len; i++) s->G[i] += s->X[i] * it;
                blk_ATy(B, s, yt, s->Q);
                for (size_t i = 0; i < len; i++) s->Q[i] = -s->Q[i];
                sp_add(B, &B->C, 1.0, s->Q);
                Xs[k] = s->G; Zs[k] = s->Q;
                free(T);
            }
            const double sw = measure_candidate(P, S, Xs, Zs, yt, Rp, Sc->t);
            free(Xs); free(Zs); free(rhs);
            if (getenv("BRISK_WPOLDBG")) printf("   [W-metric correction: score %.2e (pinf %.1e gap %.1e comp %.1e err2 %.1e err4 %.1e)]\n", sw, Rp->pinf, Rp->relgap, Rp->relcomp, Rp->err[2], Rp->err[4]);
            if (sw <= tol) { *rres = 0; return sw; }
            /* (Sc->t again for the Euclidean probe) */
            memcpy(Sc->t, P->b, sizeof(double) * m);
            for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].X, Sc->t, -it);
        }
    }
    double dmin = INFINITY;
    for (int i = 0; i < m; i++) dmin = fmin(dmin, P->d[i]);
    const double abs_tol = 1e-3 * tol * (1.0 + P->normb2) * dmin / P->bs;
    g_gram_cap = g_probe_cgcap;
    *rres = gram_cg_polish(P, S, Sc->t, w, 1e-15, abs_tol, Sc->r, Sc->p, Sc->q);
    g_gram_cap = 0;
    double **Xs = malloc(sizeof(double *) * nb), **Zs = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const size_t len = bsz(B);
        blk_ATy(B, s, w, s->G);
        for (size_t i = 0; i < len; i++) s->G[i] += s->X[i] * it;
        blk_ATy(B, s, yt, s->Q);
        for (size_t i = 0; i < len; i++) s->Q[i] = -s->Q[i];
        sp_add(B, &B->C, 1.0, s->Q);
        Xs[k] = s->G; Zs[k] = s->Q;
    }
    const double sp = measure_candidate(P, S, Xs, Zs, yt, Rp, Sc->t);
    free(Xs); free(Zs);
    return sp;
}

/* Restoration of the primal feasibility in the square-root metric of X:
 *   X <- X + D W D,  D = X^{1/2},  W = A'(w),  A (D (x) D) A' w = bt - A(X).
 * The correction is multiplicative, X^{1/2} (I + W) X^{1/2}: X stays positive semidefinite
 * while I + W is, its small eigenvalues stay small and <X, Z> changes by <W, D Z D>, of the
 * order of the complementarity. The matrix is the Schur complement with X^{1/2} in the place
 * of both X and Z^{-1} (assembled and factored by the usual routes); its condition number is
 * the square root of that of the X-metric matrix A (X (x) X) A', to which the Schur complement
 * of an interior-point iteration is close: where that one is numerically singular (the last
 * iterations on problems without an interior: its solve leaves a residual of order one) this
 * one is solved to 1e-8. A step that would leave the cone is halved; further passes take
 * what a shortened step or the solve left. Xc is corrected in place;
 * returns the smallest step of the passes, 0 on failure (Xc is then partly corrected: callers
 * work on a copy). */
#define RST_MARGIN 0.01
static double sqrt_restore(const Problem *P, BS *S, double **Xc, const double *bt, Schur *Sc, int maxpass, double *rres_out, const char **why) {
    const int m = P->m, nb = P->nblk;
    double **D = malloc(sizeof(double *) * nb), **Wk = malloc(sizeof(double *) * nb), **Fk = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) {
        const size_t len = bsz(&P->blk[k]);
        D[k] = amalloc(sizeof(double) * (len + 1)); Wk[k] = amalloc(sizeof(double) * (len + 1));
        Fk[k] = P->blk[k].type == BLK_LP ? NULL : amalloc(sizeof(double) * (len + 1));
    }
    double *rp = malloc(sizeof(double) * (m + 1)), *w = malloc(sizeof(double) * (m + 1));
    double rres = 0, alpha_min = 1, nr0 = 0;
    int ok = 1;
    *why = "";
    for (int pass = 0; pass < maxpass && ok; pass++) {
        memcpy(rp, bt, sizeof(double) * m);
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], Xc[k], rp, -1.0);
        const double nr = sqrt(ddot_n(m, rp, rp));
        if (pass == 0) nr0 = nr; else if (nr <= 1e-6 * nr0) break;
        /* D = X^{1/2} by the eigenvalue decomposition (negative eigenvalues of the rounding set to 0) */
        for (int k = 0; k < nb && ok; k++) {
            const Block *B = &P->blk[k];
            const int n = B->n;
            if (B->type == BLK_LP) { for (int i = 0; i < n; i++) D[k][i] = 1.0; continue; }
            double *Q = Wk[k], *ev = malloc(sizeof(double) * (n + 1)), wq; int lwork = -1, liwork = -1, iwq, info = 0;
            memcpy(Q, Xc[k], sizeof(double) * (size_t)n * n);
            BL(dsyevd_)("V", "L", &n, Q, &n, ev, &wq, &lwork, &iwq, &liwork, &info);
            lwork = (int)wq + 1; liwork = iwq;
            double *work = malloc(sizeof(double) * lwork); int *iwork = malloc(sizeof(int) * (liwork + 1));
            BL(dsyevd_)("V", "L", &n, Q, &n, ev, work, &lwork, iwork, &liwork, &info);
            free(work); free(iwork);
            if (info) { ok = 0; *why = " (eigenvalue decomposition)"; free(ev); break; }
            for (int j = 0; j < n; j++) { const double f = ev[j] > 0 ? sqrt(sqrt(ev[j])) : 0.0; for (int i = 0; i < n; i++) Q[i + (size_t)j * n] *= f; }
            BL(dsyrk_)("L", "N", &n, &n, &DONE, Q, &n, &DZERO, D[k], &n);
            lower_to_full(n, D[k]);
            free(ev);
        }
        if (!ok) break;
        double *Ma = schur_asm_begin(Sc);
        for (int k = 0; k < nb; k++) {
            BS v = S[k];
            if (P->blk[k].type == BLK_LP) { v.X = Xc[k]; v.Zi = D[k]; schur_lp(&P->blk[k], &v, m, Ma); }
            else { v.X = D[k]; v.Zi = D[k]; schur_sdp(&P->blk[k], &v, m, Ma); }
        }
        schur_asm_end(Sc);
        if (schur_factor(Sc)) { ok = 0; *why = " (factorization)"; break; }
        rres = schur_solve(Sc, rp, w, 1e-14);
        for (int k = 0; k < nb && ok; k++) {
            const Block *B = &P->blk[k];
            const int n = B->n;
            blk_ATy(B, &S[k], w, Wk[k]);
            /* the step: X^{1/2} (I + a W) X^{1/2} stays in the cone while I + a W does. The test is
             * made on I + a W, with a margin (its eigenvalues stay above RST_MARGIN, so no
             * eigenvalue of X shrinks by more than that factor in a pass), not on the new X: a
             * Cholesky factorization of X itself breaks down on the numerically singular X of
             * the last iterations whatever the step. */
            double a = 1;
            if (B->type == BLK_LP) {
                double wmin = 0;
                for (int i = 0; i < n; i++) if (Wk[k][i] < wmin) wmin = Wk[k][i];
                if (a * wmin < -(1.0 - RST_MARGIN)) a = -(1.0 - RST_MARGIN) / wmin;
                for (int i = 0; i < n; i++) Xc[k][i] += a * Xc[k][i] * Wk[k][i];
                alpha_min = fmin(alpha_min, a);
                continue;
            }
            const size_t len = (size_t)n * n;
            int info = 1;
            for (int h = 0; h < 8; h++, a *= 0.5) {
                for (int j = 0; j < n; j++) {
                    for (int i = j; i < n; i++) Fk[k][i + (size_t)j * n] = a * 0.5 * (Wk[k][i + (size_t)j * n] + Wk[k][j + (size_t)i * n]);
                    Fk[k][j + (size_t)j * n] += 1.0 - RST_MARGIN;
                }
                BL(dpotrf_)("L", &n, Fk[k], &n, &info);
                if (info == 0) break;
            }
            if (info != 0) { ok = 0; *why = " (no step keeps X positive semidefinite)"; break; }
            /* dX = D W D */
            xsymm("R", "L", &n, &n, &DONE, D[k], &n, Wk[k], &n, &DZERO, Fk[k], &n);
            xsymm("L", "L", &n, &n, &DONE, D[k], &n, Fk[k], &n, &DZERO, Wk[k], &n);
            for (size_t i = 0; i < len; i++) Fk[k][i] = Xc[k][i] + a * 0.5 * (Wk[k][i] + Wk[k][(i % n) * n + i / n]);
            memcpy(Xc[k], Fk[k], sizeof(double) * len);
            alpha_min = fmin(alpha_min, a);
        }
    }
    for (int k = 0; k < nb; k++) { free(D[k]); free(Wk[k]); free(Fk[k]); }
    free(D); free(Wk); free(Fk); free(rp); free(w);
    if (rres_out) *rres_out = rres;
    return ok ? alpha_min : 0.0;
}

/* Polishing in the metric of the iterate: X <- X + sym(X (A'w) Z^-1) with
 * A (X (x) Z^-1) A' w = b - A(X) (the primal half of an HKM step; at a central point
 * Z^-1 = X / mu, so this is the correction of minimal norm in ||X^-1/2 dX X^-1/2||). Unlike the Euclidean
 * projection it moves near-null directions of X only a little, so it keeps X positive
 * definite (checked by Cholesky, the step is halved otherwise). The matrix is the Schur
 * complement A (X (x) Z^-1) A' with the inverse of the candidate's own Z (half = 0; half = 1 is
 * sqrt_restore), assembled and factored by the usual routes. It matters
 * after a chordal conversion: the Euclidean projection leaves the clique blocks indefinite
 * at the 1e-8 level, which the positive semidefinite completion then has to clip.
 * Two passes (the residual after one is second order). Candidates as in polish_solution. */
static int polish_xmetric(const Problem *P, BS *S, double **Xb, double **Zb, const double *yb,
                          Result *R, double score_now, Schur *Sc, double *w, int verbose, int half) {
    const int m = P->m, nb = P->nblk;
    double **Xc = malloc(sizeof(double *) * nb), **Zi = malloc(sizeof(double *) * nb);
    int zok = 1;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        size_t len = bsz(B);
        Xc[k] = malloc(sizeof(double) * (len + 1));
        memcpy(Xc[k], Xb[k], sizeof(double) * len);
        Zi[k] = NULL;
        if (half) continue;                       /* (sqrt_restore has its own weights) */
        Zi[k] = malloc(sizeof(double) * (len + 1));
        memcpy(Zi[k], Zb[k], sizeof(double) * len);
        if (B->type == BLK_LP) { for (int i = 0; i < B->n; i++) { if (Zi[k][i] > 0) Zi[k][i] = 1.0 / Zi[k][i]; else zok = 0; } continue; }
        int n = B->n, info = 0;
        BL(dpotrf_)("L", &n, Zi[k], &n, &info);
        if (info == 0) BL(dpotri_)("L", &n, Zi[k], &n, &info);
        if (info != 0) { zok = 0; continue; }
        for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) Zi[k][j + (size_t)i * n] = Zi[k][i + (size_t)j * n];
    }
    double rres = 0, alpha_min = 1;
    int ok = zok || half;
    const char *why = (zok || half) ? "" : " (Z is not positive definite)";
    double *rp = malloc(sizeof(double) * (m + 1));      /* Sc->t is PCG scratch */
    if (half) {
        alpha_min = sqrt_restore(P, S, Xc, P->b, Sc, 3, &rres, &why);
        ok = alpha_min > 0;
    }
    static int xpol_np = -1; if (xpol_np < 0) { const char *e = getenv("BRISK_XPOLPASS"); xpol_np = e ? atoi(e) : 2; }   /* (test) */
    for (int pass = 0; pass < xpol_np && ok && !half; pass++) {
        memcpy(rp, P->b, sizeof(double) * m);
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], Xc[k], rp, -1.0);
        /* 5.12: the second pass only when the first one left a primal residual that matters: on a
         * user's moment relaxations the first pass reached the final score every time (ball (8,4):
         * 1.3e-8 -> 6.5e-11 either way, the second pass 6.9 of 50 s: an assembly and a
         * double-precision factorization of order 12869): not when the primal residual is below
         * 1e-3 times the score being polished. BRISK_XPOLPASS=2 always two. */
        if (pass > 0 && !getenv("BRISK_XPOLPASS")) {
            double r2 = 0;
            for (int i = 0; i < m; i++) { const double w = rp[i] * P->du[i]; r2 += w * w; }
            if (sqrt(r2) / (1 + P->normb2) <= 1e-3 * score_now) break;
        }
        double *Ma = schur_asm_begin(Sc);
        for (int k = 0; k < nb; k++) {
            BS v = S[k];
            v.X = Xc[k]; v.Zi = Zi[k];
            if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], &v, m, Ma);
            else schur_sdp(&P->blk[k], &v, m, Ma);
        }
        schur_asm_end(Sc);
        if (schur_factor(Sc)) { ok = 0; why = " (factorization)"; break; }
        rres = schur_solve(Sc, rp, w, 1e-14);
        for (int k = 0; k < nb && ok; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            const int n = B->n;
            blk_ATy(B, s, w, s->W);
            if (B->type == BLK_LP) {
                double a = 1;
                for (int i = 0; i < n; i++) {
                    double d = Xc[k][i] * Zi[k][i] * s->W[i];
                    if (Xc[k][i] + a * d <= 0) a = fmin(a, -0.5 * Xc[k][i] / d);
                }
                for (int i = 0; i < n; i++) Xc[k][i] += a * Xc[k][i] * Zi[k][i] * s->W[i];
                alpha_min = fmin(alpha_min, a);
                continue;
            }
            /* dX = sym(X F Zi) (s->W holds F, then F Zi, then X F Zi) */
            xsymm("R", "L", &n, &n, &DONE, Zi[k], &n, s->W, &n, &DZERO, s->F, &n);
            xsymm("L", "L", &n, &n, &DONE, Xc[k], &n, s->F, &n, &DZERO, s->W, &n);
            double a = 1;
            int info = 1;
            for (int h = 0; h < 6 && info != 0; h++, a *= 0.5) {
                const size_t len = (size_t)n * n;
                for (size_t i = 0; i < len; i++) s->F[i] = Xc[k][i] + a * 0.5 * (s->W[i] + s->W[(i % n) * n + i / n]);
                double *T = malloc(sizeof(double) * (len + 1));
                memcpy(T, s->F, sizeof(double) * len);
                BL(dpotrf_)("L", &n, T, &n, &info);
                free(T);
                if (info == 0) memcpy(Xc[k], s->F, sizeof(double) * len);
            }
            if (info != 0) { ok = 0; why = " (no step keeps X positive definite)"; break; }
            alpha_min = fmin(alpha_min, 2 * a);
        }
    }
    int acc = 0;
    if (ok) {
        double **Xs = malloc(sizeof(double *) * nb), **Zs = malloc(sizeof(double *) * nb);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            memcpy(s->G, Xc[k], sizeof(double) * len);
            blk_ATy(B, s, yb, s->Q);
            for (size_t i = 0; i < len; i++) s->Q[i] = -s->Q[i];
            sp_add(B, &B->C, 1.0, s->Q);
            Xs[k] = s->G; Zs[k] = s->Q;
        }
        Result Rp = *R;
        double sp = measure_candidate(P, S, Xs, Zs, yb, &Rp, Sc->t);
        free(Xs); free(Zs);
        /* 4.23: a correction whose Newton system was not solved (relative residual ~0.9) is
         * not the projection it stands for: on dense case6468 every such acceptance (7 of 7
         * runs, rres 0.88-0.99) lowered the converted score but ended at 2-3e-6 on the
         * original problem (second solve, +90 s), every rejection at 6e-8 - 7e-7 */
        static double xres_max = -1;
        if (xres_max < 0) { const char *e = getenv("BRISK_XPOLRES"); xres_max = e ? atof(e) : 0.1; }
        /* only on the moment form of a chordal conversion, whose converted score is what
         * misleads (attr_henon_d8_K, unconverted: rres 0.94, 5.2e-7 -> 1.8e-7, then optimal) */
        /* 4.24: also with split free pairs (the embedding chosen for them): their slots are
         * re-balanced after the solve, so the score of the candidate misleads the same way
         * (TSSOS case6468: rres 0.94 accepted, 6.6e-6 -> 2.5e-6 on the original data with a
         * gap; the plain projection gives 4e-8 there) */
        const int conv = (P->ps && P->ps->mom) || g_free_hsd;
        const int take = sp < score_now && (rres <= xres_max || !conv);
        if (!take) { double **Xg = malloc(sizeof(double *) * (nb + 1)); for (int k = 0; k < nb; k++) Xg[k] = S[k].G; rej_offer(P, Xg, yb, &Rp, sp); free(Xg); }
        if (verbose > 0)
            printf("   %s polishing: score %.2e -> %.2e (solve residual %.1e, step %.2f) %s\n", half ? "square-root-metric" : "X-metric",
                   score_now, sp, rres, alpha_min, take ? "accepted" : sp < score_now ? "rejected (solve residual)" : "rejected");
        if (half && ENV_ON("BRISK_POLDBG")) printf("     [candidate: pinf %.2e dinf %.2e gap %.2e comp %.2e err %.1e %.1e]\n", Rp.pinf, Rp.dinf, Rp.relgap, Rp.relcomp, Rp.err[2], Rp.err[4]);
        if (take) {
            double tt[6] = { R->t_setup, R->t_schur, R->t_chol, R->t_dense, R->t_step, R->t_total };
            int it = R->iters;
            *R = Rp;
            R->t_setup = tt[0]; R->t_schur = tt[1]; R->t_chol = tt[2]; R->t_dense = tt[3]; R->t_step = tt[4];
            R->t_total = tt[5]; R->iters = it;
            acc = 1;
        }
    } else if (verbose > 0) printf("   %s polishing: failed%s\n", half ? "square-root-metric" : "X-metric", why);
    free(rp);
    for (int k = 0; k < nb; k++) { free(Xc[k]); free(Zi[k]); }
    free(Xc); free(Zi);
    return acc;
}

/* ======================================================================== */
/* NT (Nesterov-Todd) direction in the scaled space.
 *   X = L_X L_X',  Z = L_Z L_Z',  L_Z' L_X = U diag(lam) V'
 *   G = L_X V diag(lam)^{-1/2},  W = G G'  (W Z W = X),  G' Z G = G^{-1} X G^{-T} = diag(lam)
 * Scaled directions dXt = G^{-1} dX G^{-T}, dZt = G' dZ G satisfy dXt + dZt = Rt with
 * Rt = L_lam^{-1}(sigma mu I - lam^2 - sym(dXt_a dZt_a)), where
 * L_lam^{-1}(M)_ij = 2 M_ij / (lam_i + lam_j). All scaled quantities are of size ~sqrt(mu).
 * LP blocks: G = sqrt(x/z) (diagonal), lam = sqrt(x z); stored W = sqrt(x/z).       */

/* view of a block state in which X and Zi both mean W (for Schur assembly and products) */
static BS *nt_view(BS *s, BS *tmp, int nt) {
    if (!nt) return s;
    *tmp = *s;
    tmp->X = s->Zi;
    tmp->ntG = s->ntGbuf;
    return tmp;
}

static int g_nt_ginv = -1;   /* (test, BRISK_NTGINV=1) nt_toX by the explicit G^{-1} */
/* the NT scaling: the products (threaded), the eigen-decomposition (one thread), the rest (threaded) */
static int g_nteig_env = -1;
static void nt_prep(const Block *B, BS *s) {
    const int n = B->n;
    s->ntf_ok = 0;
    double *Mb = s->ntT;
    size_t len = (size_t)n * n;
    memset(Mb, 0, sizeof(double) * len);
    for (int j = 0; j < n; j++)
        for (int i = j; i < n; i++) Mb[i + (size_t)j * n] = s->LX[i + (size_t)j * n];
    xtrmm("L", "L", "T", "N", &n, &n, &DONE, s->LZ, &n, Mb, &n);               /* L_Z' L_X */
    /* 5.12: the singular vectors of L_Z' L_X are the eigenvectors of H = (L_Z' L_X)'(L_Z' L_X)
     * = L_X' Z L_X, whose eigenvalues are the squares lam^2 (those of X Z, within the
     * neighbourhood's range of mu): the product on the threads and dsyevd instead of dgesdd,
     * half the time of dgesdd on one thread at n = 330-792 (bench: the singular values agree to
     * 1.6e-15 of the largest). The squares lose accuracy only far from the central path
     * (relative error u (lam_max / lam_i)^2); the eigenvalues must come out positive, else
     * dgesdd as before. BRISK_NTEIG=0: dgesdd always. */
    if (g_nteig_env < 0) { const char *e = getenv("BRISK_NTEIG"); g_nteig_env = e ? atoi(e) : 1; }
    if (g_nteig_env) pdgemm("T", "N", &n, &n, &n, &DONE, Mb, &n, Mb, &n, &DZERO, s->ntVt, &n);
}
static void nt_eig(const Block *B, BS *s) {
    const int n = B->n;
    s->nt_einfo = 1;
    if (!g_nteig_env) return;
    double *H = s->ntVt;
    int lwork = -1, liwork = -1, iwq = 0, info = 0;
    double wq = 0;
    BL(dsyevd_)("V", "L", &n, H, &n, s->lam, &wq, &lwork, &iwq, &liwork, &info);
    lwork = (int)wq + 1; liwork = iwq + 1;
    double *work = malloc(sizeof(double) * lwork);
    int *iwork = malloc(sizeof(int) * liwork);
    BL(dsyevd_)("V", "L", &n, H, &n, s->lam, work, &lwork, iwork, &liwork, &info);
    free(work); free(iwork);
    s->nt_einfo = info;
}
static int nt_post(const Block *B, BS *s) {
    const int n = B->n;
    double *Mb = s->ntT, *G = s->ntGbuf;
    int done = 0, info = 0;
    if (g_nteig_env && s->nt_einfo == 0 && s->lam[0] > 0) {
        double *H = s->ntVt;
        /* descending, as dgesdd's: lam_j = sqrt(w_{n-1-j}), V(:, j) = Q(:, n-1-j) */
        for (int j = 0; j < n; j++) memcpy(G + (size_t)j * n, H + (size_t)(n - 1 - j) * n, sizeof(double) * n);
        for (int j = 0; j < n / 2; j++) { const double t = s->lam[j]; s->lam[j] = s->lam[n - 1 - j]; s->lam[n - 1 - j] = t; }
        for (int j = 0; j < n; j++) s->lam[j] = sqrt(s->lam[j]);
        done = 1;
    }
    if (!done) {
        int lwork = -1, one = 1;
        double wq, dummy;
        int *iwork = malloc(sizeof(int) * 8 * (size_t)n);
        BL(dgesdd_)("O", &n, &n, Mb, &n, s->lam, &dummy, &one, s->ntVt, &n, &wq, &lwork, iwork, &info);
        lwork = (int)wq + 1;
        double *work = malloc(sizeof(double) * lwork);
        BL(dgesdd_)("O", &n, &n, Mb, &n, s->lam, &dummy, &one, s->ntVt, &n, work, &lwork, iwork, &info);
        free(work); free(iwork);
        if (info != 0) return 1;
        for (int j = 0; j < n; j++)                                               /* V = Vt' */
            for (int i = 0; i < n; i++) G[i + (size_t)j * n] = s->ntVt[j + (size_t)i * n];
    }
    for (int j = 0; j < n; j++) if (!(s->lam[j] > 0)) return 1;
    if (done || g_nt_ginv) {
        double *Gi = s->ntVt;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) Gi[j + (size_t)i * n] = G[i + (size_t)j * n];   /* V' */
        if (g_nt_ginv) {
            /* (test) G^{-1} = diag(lam)^{1/2} V' L_X^{-1} kept in ntVt: nt_toX two products */
            xtrsm("R", "L", "N", "N", &n, &n, &DONE, s->LX, &n, Gi, &n);              /* V' L_X^{-1} */
            for (int i = 0; i < n; i++) {
                double *col = Gi + (size_t)i * n;
                for (int j = 0; j < n; j++) col[j] *= sqrt(s->lam[j]);
            }
        }
    }
    xtrmm("L", "L", "N", "N", &n, &n, &DONE, s->LX, &n, G, &n);                   /* L_X V */
    for (int j = 0; j < n; j++) {
        double f = 1.0 / sqrt(s->lam[j]);
        for (int i = 0; i < n; i++) G[i + (size_t)j * n] *= f;
    }
    pdgemm("N", "T", &n, &n, &n, &DONE, G, &n, G, &n, &DZERO, s->Zi, &n);        /* W = G G' */
    lower_to_full(n, s->Zi);
    return 0;
}
static int nt_scaling(const Block *B, BS *s) {
    if (g_nt_ginv < 0) g_nt_ginv = ENV_ON("BRISK_NTGINV");
    const int n = B->n;
    if (B->type == BLK_LP) {
        for (int i = 0; i < n; i++) {
            double w = sqrt(s->X[i] / s->Z[i]);       /* W = G^2 */
            s->Zi[i] = w;
            s->ntGbuf[i] = sqrt(w);
            s->lam[i] = sqrt(s->X[i] * s->Z[i]);
        }
        return 0;
    }
    nt_prep(B, s);
    nt_eig(B, s);
    return nt_post(B, s);
}

/* 5.12: the frame changes of the embedding's second-order term and correctors in single precision
 * while the row products are (g_ntsp: the factor single precision, the residuals above 1e-3):
 * they enter only the corrections of the direction. Products and triangular solves in fixed
 * panels of 64 columns (rows for a right-side solve), each one single-threaded BLAS call: the
 * same bits for any thread count. BRISK_NTSP=0 off. */
int g_ntsp = 0;
extern void BL(strsm_)(const char *, const char *, const char *, const char *, const int *, const int *, const float *, const float *, const int *, float *, const int *);
static void ps_gemm(const char *ta, const char *tb, int m, int n, int k, const float *A, int lda, const float *Bm, int ldb, float *C, int ldc) {
    const float one = 1.0f, zero = 0.0f;
    const int PW = 64, np = (n + PW - 1) / PW;
    const int nb = tb[0] == 'N' || tb[0] == 'n';
    #pragma omp parallel for schedule(static) if (np > 1 && !omp_in_parallel())
    for (int p = 0; p < np; p++) {
        const int c0 = p * PW, w = n - c0 < PW ? n - c0 : PW;
        const float *Bp = nb ? Bm + (size_t)c0 * ldb : Bm + c0;
        BL(sgemm_)(ta, tb, &m, &w, &k, &one, A, &lda, Bp, &ldb, &zero, C + (size_t)c0 * ldc, &ldc);
    }
}
/* 5.12: a panel's solve by blocks of 64 (strsm on the diagonal blocks, sgemm updates; L lower,
 * non-unit), as pt_blocked in sparsechol.c: strsm on these shapes ran at a fraction of sgemm */
static void ps_trsm_blk(int left, int tr, int m, int n, const float *L, int ldl, float *B, int ldb) {
    const float one = 1.0f, mone = -1.0f;
    const int NB = 64, k = left ? m : n, nbk = (k + NB - 1) / NB;
    for (int s2 = 0; s2 < nbk; s2++) {
        const int q = (left ? !tr : tr) ? s2 : nbk - 1 - s2;
        const int j = q * NB, b = k - j < NB ? k - j : NB, r = k - j - b;
        const float *Ljj = L + j + (size_t)j * ldl;
        if (left) {
            float *Bj = B + j;
            BL(strsm_)("L", "L", tr ? "T" : "N", "N", &b, &n, &one, Ljj, &ldl, Bj, &ldb);
            if (!tr && r > 0) BL(sgemm_)("N", "N", &r, &n, &b, &mone, Ljj + b, &ldl, Bj, &ldb, &one, Bj + b, &ldb);
            if (tr && j > 0) BL(sgemm_)("T", "N", &j, &n, &b, &mone, L + j, &ldl, Bj, &ldb, &one, B, &ldb);
        } else {
            float *Bj = B + (size_t)j * ldb;
            BL(strsm_)("R", "L", tr ? "T" : "N", "N", &m, &b, &one, Ljj, &ldl, Bj, &ldb);
            if (!tr && j > 0) BL(sgemm_)("N", "N", &m, &j, &b, &mone, Bj, &ldb, L + j, &ldl, &one, B, &ldb);
            if (tr && r > 0) BL(sgemm_)("N", "T", &m, &r, &b, &mone, Bj, &ldb, Ljj + b, &ldl, &one, Bj + (size_t)b * ldb, &ldb);
        }
    }
}
static void ps_trsm(const char *side, const char *ta, int m, int n, const float *L, int ldl, float *Bm, int ldb) {
    const float one = 1.0f;
    const int left = side[0] == 'L', PW = 64, tot = left ? n : m, np = (tot + PW - 1) / PW;
    static int blk_env = -1; if (blk_env < 0) blk_env = !ENV_ON("BRISK_PTBLK0");
    const int blk = blk_env && (left ? m : n) > 128;
    #pragma omp parallel for schedule(static) if (np > 1 && !omp_in_parallel())
    for (int p = 0; p < np; p++) {
        const int r0 = p * PW, w = tot - r0 < PW ? tot - r0 : PW;
        float *Bp = left ? Bm + (size_t)r0 * ldb : Bm + r0;
        const int mm = left ? m : w, nn = left ? w : n;
        if (blk) ps_trsm_blk(left, ta[0] == 'T', mm, nn, L, ldl, Bp, ldb);
        else BL(strsm_)(side, "L", ta, "N", &mm, &nn, &one, L, &ldl, Bp, &ldb);
    }
}
static void nt_float_prep(const Block *B, BS *s) {
    const int n = B->n;
    const size_t len = (size_t)n * n;
    if (!s->ntGf) {
        s->ntGf = amalloc(sizeof(float) * len); s->ntLXf = amalloc(sizeof(float) * len); s->ntVtf = amalloc(sizeof(float) * len);
        s->ntTf = amalloc(sizeof(float) * len); s->ntT2f = amalloc(sizeof(float) * len);
    }
    if (s->ntf_ok) return;
    for (size_t i = 0; i < len; i++) { s->ntGf[i] = (float)s->ntGbuf[i]; s->ntLXf[i] = (float)s->LX[i]; s->ntVtf[i] = (float)s->ntVt[i]; }
    s->ntf_ok = 1;
}
static inline int nt_use_float(const Block *B) {
    static int env = -1; if (env < 0) { const char *e = getenv("BRISK_NTSP"); env = e ? atoi(e) : 1; }
    /* 5.16 (ISSUES 83): also inside a loop over blocks (the panels then run on one thread, the
     * same bits). Until 5.15 the float path ran only outside a parallel region: a block of order
     * >= 128 among several got the single-precision frames at one thread and the double ones at
     * two or more, other bits for another thread count. */
    return env && g_ntsp && B->type == BLK_SDP && B->n >= 128;
}

/* out = G' D G  (D symmetric) */
static void nt_toZ(const Block *B, BS *s, const double *D, double *out) {
    const int n = B->n;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) out[i] = s->ntGbuf[i] * s->ntGbuf[i] * D[i]; return; }
    if (nt_use_float(B)) {
        nt_float_prep(B, s);
        const size_t len = (size_t)n * n;
        float *Df = s->ntT2f, *T = s->ntTf;
        for (size_t i = 0; i < len; i++) Df[i] = (float)D[i];
        ps_gemm("N", "N", n, n, n, Df, n, s->ntGf, n, T, n);               /* D G */
        ps_gemm("T", "N", n, n, n, s->ntGf, n, T, n, Df, n);               /* G' (D G) */
        for (size_t i = 0; i < len; i++) out[i] = (double)Df[i];
        symmetrize(n, out);
        return;
    }
    xsymm("L", "L", &n, &n, &DONE, D, &n, s->ntGbuf, &n, &DZERO, s->ntT, &n);
    pdgemm("T", "N", &n, &n, &n, &DONE, s->ntGbuf, &n, s->ntT, &n, &DZERO, out, &n);
    symmetrize(n, out);
}

/* out = G^{-1} D G^{-T} = diag(lam)^{1/2} V' L_X^{-1} D L_X^{-T} V diag(lam)^{1/2} */
static void nt_toX(const Block *B, BS *s, const double *D, double *out) {
    const int n = B->n;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) out[i] = D[i] / (s->ntGbuf[i] * s->ntGbuf[i]); return; }
    if (nt_use_float(B) && !g_nt_ginv) {
        nt_float_prep(B, s);
        const size_t len = (size_t)n * n;
        float *T = s->ntTf, *U = s->ntT2f;
        for (size_t i = 0; i < len; i++) T[i] = (float)D[i];
        ps_trsm("L", "N", n, n, s->ntLXf, n, T, n);                         /* L_X^{-1} D */
        ps_trsm("R", "T", n, n, s->ntLXf, n, T, n);                         /* ... L_X^{-T} */
        ps_gemm("N", "N", n, n, n, s->ntVtf, n, T, n, U, n);                /* V' T */
        ps_gemm("N", "T", n, n, n, U, n, s->ntVtf, n, T, n);                /* ... V */
        for (int j = 0; j < n; j++) {
            const double sj = sqrt(s->lam[j]);
            for (int i = 0; i < n; i++) out[i + (size_t)j * n] = (double)T[i + (size_t)j * n] * sj * sqrt(s->lam[i]);
        }
        symmetrize(n, out);
        return;
    }
    if (g_nt_ginv) {
        /* (test) G^{-1} (in ntVt, nt_scaling) times D times its transpose */
        xsymm("R", "L", &n, &n, &DONE, D, &n, s->ntVt, &n, &DZERO, s->ntT, &n);
        pdgemm("N", "T", &n, &n, &n, &DONE, s->ntT, &n, s->ntVt, &n, &DZERO, out, &n);
        symmetrize(n, out);
        return;
    }
    size_t len = (size_t)n * n;
    double *T = s->ntT;
    memcpy(T, D, sizeof(double) * len);
    xtrsm("L", "L", "N", "N", &n, &n, &DONE, s->LX, &n, T, &n);
    xtrsm("R", "L", "T", "N", &n, &n, &DONE, s->LX, &n, T, &n);
    pdgemm("N", "N", &n, &n, &n, &DONE, s->ntVt, &n, T, &n, &DZERO, out, &n);    /* V' T */
    pdgemm("N", "T", &n, &n, &n, &DONE, out, &n, s->ntVt, &n, &DZERO, T, &n);    /* ... V */
    for (int j = 0; j < n; j++) {
        double sj = sqrt(s->lam[j]);
        for (int i = 0; i < n; i++) out[i + (size_t)j * n] = T[i + (size_t)j * n] * sj * sqrt(s->lam[i]);
    }
    symmetrize(n, out);
}

/* out = G Dt G' */
static void nt_fromS(const Block *B, BS *s, const double *Dt, double *out) {
    const int n = B->n;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) out[i] = s->ntGbuf[i] * s->ntGbuf[i] * Dt[i]; return; }
    if (nt_use_float(B)) {
        nt_float_prep(B, s);
        const size_t len = (size_t)n * n;
        float *Df = s->ntT2f, *T = s->ntTf;
        for (size_t i = 0; i < len; i++) Df[i] = (float)Dt[i];
        ps_gemm("N", "N", n, n, n, s->ntGf, n, Df, n, T, n);               /* G Dt */
        ps_gemm("N", "T", n, n, n, T, n, s->ntGf, n, Df, n);               /* (G Dt) G' */
        for (size_t i = 0; i < len; i++) out[i] = (double)Df[i];
        symmetrize(n, out);
        return;
    }
    xsymm("R", "L", &n, &n, &DONE, Dt, &n, s->ntGbuf, &n, &DZERO, s->ntT, &n);
    pdgemm("N", "T", &n, &n, &n, &DONE, s->ntT, &n, s->ntGbuf, &n, &DZERO, out, &n);
    symmetrize(n, out);
}

/* Rt = L_lam^{-1}(smu I - lam^2 - H) with H symmetric (NULL = 0); corrector/predictor rhs */
static void nt_rhs_scaled(const Block *B, BS *s, double smu, const double *H, double *Rt) {
    const int n = B->n;
    const double *l = s->lam;
    if (B->type == BLK_LP) {
        for (int i = 0; i < n; i++) Rt[i] = (smu - l[i] * l[i] - (H ? H[i] : 0.0)) / l[i];
        return;
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double hij = H ? H[i + (size_t)j * n] : 0.0;
            double v = -2.0 * hij / (l[i] + l[j]);
            if (i == j) v += smu / l[i] - l[i];
            Rt[i + (size_t)j * n] = v;
        }
}

/* accuracy measure used for termination and for choosing the best iterate */
static inline double nan_inf(double v) { return isfinite(v) ? v : INFINITY; }
static double res_score(const Result *R) {
    /* non-finite measures must never look good (fmax drops NaN) */
    /* 23 October 2026: with the violations of X >= 0 and Z >= 0 (errors 2 and 4), the measure measure_candidate returns,
     * the polish accepts by and the status is set from: a polished point outside the cone scored only by the other four
     * was kept against better candidates (hinf11, SDPLIB: the iterate's polish at 2.2e-7 here, 2.5e-6 in full, chosen
     * over the endgame's point, and the run ended at 2.5e-6 with a re-solve). An iterate's errors 2 and 4 are 0. */
    static int old = -1; if (old < 0) old = ENV_ON("BRISK_RESSCORE_OLD");
    const double s4 = fmax(fmax(nan_inf(R->relgap), nan_inf(R->relcomp)), fmax(nan_inf(R->pinf), nan_inf(R->dinf)));
    return old ? s4 : fmax(s4, fmax(nan_inf(R->err[2]), nan_inf(R->err[4])));
}
/* 5.12: the same point in the DIMACS measures the status is set from (||b||_1 and ||C||_1 in the
 * denominators against ||b||_2 and ||C||_F in res_score; errors 2 and 4 are 0 inside the cone) */
static double dimacs_score(const Result *R) {
    return fmax(fmax(fabs(nan_inf(R->err[1])), fabs(nan_inf(R->err[3]))), fmax(fabs(nan_inf(R->err[5])), fabs(nan_inf(R->err[6]))));
}
/* the loop stopped on dimacs_score (the best iterate is returned without endgame or polishing) */
static int g_dstop = 0;
static int dstop_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("BRISK_DSTOP"); on = e ? atoi(e) : 1; } return on; }

static void nt_toZ(const Block *B, struct BS *s, const double *D, double *out);
static void nt_fromS(const Block *B, struct BS *s, const double *Dt, double *out);
/* (A A')^{-1} b: factor when available (PCG against the exact Gram matrix), else CG */
static double gram_solve(const Problem *P, BS *S, Schur *Gs, int have, const double *b, double *x, Schur *Sc) {
    if (have) return schur_solve(Gs, b, x, 1e-14);
    return gram_cg(P, S, b, x, 1000, 1e-14, Sc->r, Sc->p, Sc->q);
}

/* ls_kstar: out = K* r = A(G r G'), block space -> R^m (uses s->G) */
static void ls_kstar(const Problem *P, BS *S, double **r, double *out) {
    memset(out, 0, sizeof(double) * P->m);
    for (int k = 0; k < P->nblk; k++) {
        nt_fromS(&P->blk[k], &S[k], r[k], S[k].G);
        blk_Aop(&P->blk[k], &S[k], S[k].G, out, 1.0);
    }
}
/* ls_k: q = K y = G' A*(y) G (uses s->F) */
static void ls_k(const Problem *P, BS *S, const double *y, double **q) {
    for (int k = 0; k < P->nblk; k++) {
        blk_ATy(&P->blk[k], &S[k], y, S[k].F);
        nt_toZ(&P->blk[k], &S[k], S[k].F, q[k]);
    }
}

static void set_fast_fp(void) {
#if defined(__x86_64__) || defined(__i386__)
    /* flush-to-zero / denormals-are-zero: subnormal arithmetic is ~100x slower in
       microcode and such magnitudes are irrelevant to an interior-point method */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}


/* ======================================================================== */
/* Homogeneous self-dual embedding (Goldman-Tucker / Ye-Todd-Mizuno style).
 *
 * The infeasible-start iteration enforces feasibility only implicitly through
 * A(dX) = r_p. Once that equation stops holding accurately (near-singular Schur
 * complement) nothing prevents the method from driving mu down by shrinking X
 * instead of converging: on the 793-bus AC-OPF relaxation X collapsed to zero and
 * the primal residual jumped from 1.6e-10 to 0.5 while the dual objective sat at
 * the optimum. The embedding removes that freedom: the iterate carries its own
 * scale tau, a collapse shows up as tau -> 0 (an infeasibility certificate), and
 * one step length is shared by X, Z, tau and kappa.
 *
 *   A(X) - b tau = 0,   A'(y) + Z - C tau = 0,   b'y - <C,X> - kappa = 0
 *
 * Newton step (HKM symmetrization), with eta = 1 - sigma:
 *   dZ = eta Rd + C dtau - A'(dy)
 *   dX = sym(Zi K) - X - sym(Zi dZ X),   K = sigma mu I - sym(dZa dXa)
 *   M dy = eta r_p + A(X) - A(sym(Zi K)) + eta h_d + dtau (b + h_c)
 * so dy = u + dtau v with two solves sharing one factorization
 * (h_d = A(sym(Zi Rd X)), h_c = A(sym(Zi C X)); v does not depend on sigma).
 * dtau comes from the third row together with tau dkappa + kappa dtau = sigma mu - tau kappa.
 */
/* ------------------------------------------------------------------------ */
/* Free variables inside the self-dual embedding.
 * A free variable written as a split pair x = x+ - x- of LP variables (columns a, -a and
 * costs c, -c) leaves the dual side without an interior point: z+ + z- = 0 at every
 * feasible y, so both slacks go to zero while x+ and x- grow without bound. The embedding
 * works with the pair natively instead: the '+' slot holds the free value (of any sign),
 * the '-' slot and both slacks are zero, the pair drops out of the complementarity, and
 * its dual row a'y = c tau becomes an equality of the Newton system:
 *   [M  Af] [dy ]   [r1]
 *   [Af' 0] [dxf] = [r2],  solved through Y = M^{-1} Af and Sf = Af' Y (nf x nf).       */
typedef struct {
    int nf;
    int *kb, *ip, *im;      /* LP block, '+' and '-' index of each pair */
    double *c;              /* c_f (scaled data) */
    double *Y, *Sf;         /* m x nf, nf x nf (Cholesky) */
    double *w, *t;          /* work: m, nf */
    double *xf;             /* work: nf */
    double *sd;             /* equilibration of Sf */
    double reg;
    int aug;                /* dense augmented factorization in use */
    int al;                 /* augmented-Lagrangian mode: M + Af W Af' factored, refinement */
    double *wf;             /* its weights W */
    double *K, *ks;         /* (m+nf)^2 Bunch-Kaufman factor of the scaled saddle matrix, scaling */
    int *ipiv;
    double *ke; int kconv;  /* 18 October 2026: the factor in dsyconv's form (brisk_dsyconv), its D's off-diagonal */
    /* sparse quasi-definite mode: K = [M Af; Af' -delta] (equilibrated) factored as L S L'
     * with the sparse Cholesky in signed mode */
    SChol *KS;
    size_t nkm, *km_sc, *km_k;       /* M entries: offset in the Schur factor, in K */
    int *km_r, *km_c;
    size_t naf, *kaf_off;            /* Af entries: offset in K, value, constraint, free index */
    double *kaf_val;
    int *kaf_r, *kaf_f;
    size_t *kdg;                     /* (m+f, m+f) */
    double *kd, kdelta;              /* equilibration (m + nf), regularization */
    double *kz, *kr;                 /* work (m + nf) */
} FreeVars;

static uint64_t fv_hash(const Block *B, int p, double cp, int sgn) {
    uint64_t h = 1469598103934665603ULL;
    for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) {
        double v = sgn * B->lp_val[a] + 0.0;
        uint64_t u; memcpy(&u, &v, 8);
        h = (h ^ (uint64_t)B->lp_con[a]) * 1099511628211ULL;
        h = (h ^ u) * 1099511628211ULL;
    }
    double v = sgn * cp + 0.0; uint64_t u; memcpy(&u, &v, 8);    /* + 0.0: no negative zero */
    h = (h ^ u) * 1099511628211ULL;
    return h;
}
typedef struct { uint64_t h; int p, sgn; } FvKey;
static int fv_cmp(const void *a, const void *b) {
    const FvKey *x = a, *y = b;
    if (x->h != y->h) return x->h < y->h ? -1 : 1;
    if (x->sgn != y->sgn) return x->sgn < y->sgn ? -1 : 1;
    return x->p - y->p;
}

/* detect split pairs; sets the masks S[k].fm (1 '+', 2 '-'); returns the number of pairs */
static int free_detect(const Problem *P, BS *S, FreeVars *F) {
    memset(F, 0, sizeof(*F));
    int cap = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_LP || B->n < 2) continue;
        const int n = B->n;
        double *c = calloc(n, sizeof(double));
        for (int q = 0; q < B->C.nnz; q++) c[B->C.row[q]] += B->C.val[q];
        /* canonical sign: the first nonzero (value, else cost) positive */
        FvKey *K = malloc(sizeof(FvKey) * n);
        int nk = 0;
        for (int p = 0; p < n; p++) {
            if (B->lp_ptr[p + 1] == B->lp_ptr[p]) continue;
            int sgn = B->lp_val[B->lp_ptr[p]] > 0 ? 1 : -1;
            K[nk].h = fv_hash(B, p, c[p], sgn); K[nk].p = p; K[nk].sgn = sgn; nk++;
        }
        qsort(K, nk, sizeof(FvKey), fv_cmp);
        char *used = calloc(n, 1);
        for (int a = 0; a < nk; ) {
            int b = a;
            while (b < nk && K[b].h == K[a].h) b++;
            /* within a hash group: sgn -1 entries first, then +1 */
            int neg = a, pos = a;
            while (pos < b && K[pos].sgn < 0) pos++;
            for (int i = neg, j = pos; i < pos && j < b; ) {
                int p = K[j].p, q = K[i].p;                 /* column p = -column q */
                int ok = !used[p] && !used[q] && (B->lp_ptr[p + 1] - B->lp_ptr[p]) == (B->lp_ptr[q + 1] - B->lp_ptr[q])
                         && c[p] == -c[q];
                for (int e = 0; ok && e < B->lp_ptr[p + 1] - B->lp_ptr[p]; e++)
                    ok = B->lp_con[B->lp_ptr[p] + e] == B->lp_con[B->lp_ptr[q] + e]
                         && B->lp_val[B->lp_ptr[p] + e] == -B->lp_val[B->lp_ptr[q] + e];
                if (ok) {
                    if (F->nf == cap) {
                        cap = cap ? 2 * cap : 64;
                        F->kb = realloc(F->kb, sizeof(int) * cap); F->ip = realloc(F->ip, sizeof(int) * cap);
                        F->im = realloc(F->im, sizeof(int) * cap); F->c = realloc(F->c, sizeof(double) * cap);
                    }
                    if (!S[k].fm) S[k].fm = calloc(n, 1);
                    F->kb[F->nf] = k; F->ip[F->nf] = p; F->im[F->nf] = q; F->c[F->nf] = c[p];
                    S[k].fm[p] = 1; S[k].fm[q] = 2;
                    used[p] = used[q] = 1;
                    F->nf++;
                    i++; j++;
                } else if (used[q]) i++;
                else j++;
            }
            a = b;
        }
        if (getenv("BRISK_FVDBG")) {
            int ng = 0;
            for (int a = 0; a < nk; ) { int b = a; while (b < nk && K[b].h == K[a].h) b++; ng++; a = b; }
            printf("   free_detect: block %d n %d nonempty %d groups %d pairs so far %d\n", k, n, nk, ng, F->nf);
            for (int t = 0; t < 6 && t < nk; t++) {
                int p = K[t].p;
                printf("     key %016llx p %d sgn %d c %.3g nnz %d first (%d %.3g)\n", (unsigned long long)K[t].h, p, K[t].sgn, c[p],
                       B->lp_ptr[p + 1] - B->lp_ptr[p], B->lp_con[B->lp_ptr[p]], B->lp_val[B->lp_ptr[p]]);
            }
        }
        free(used); free(K); free(c);
    }
    if (F->nf) {
        const int m = P->m, nf = F->nf;
        F->w = malloc(sizeof(double) * m);
        F->t = malloc(sizeof(double) * nf);
        F->xf = malloc(sizeof(double) * nf);
    }
    return F->nf;
}

static void free_release(const Problem *P, BS *S, FreeVars *F) {
    free(F->kb); free(F->ip); free(F->im); free(F->c);
    free(F->Y); free(F->Sf); free(F->w); free(F->t); free(F->xf); free(F->sd);
    free(F->K); free(F->ks); free(F->ipiv); free(F->wf); free(F->ke);
    schol_free(F->KS); free(F->km_sc); free(F->km_k); free(F->km_r); free(F->km_c);
    free(F->kaf_off); free(F->kaf_val); free(F->kaf_r); free(F->kaf_f); free(F->kdg); free(F->kd); free(F->kz); free(F->kr);
    for (int k = 0; k < P->nblk; k++) { free(S[k].fm); S[k].fm = NULL; }
    memset(F, 0, sizeof(*F));
}

/* out[f] = a_f' v */
static void free_AT(const Problem *P, const FreeVars *F, const double *v, double *out) {
    for (int f = 0; f < F->nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        double s = 0;
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) s += B->lp_val[a] * v[B->lp_con[a]];
        out[f] = s;
    }
}
/* out += alpha * Af xf */
static void free_A(const Problem *P, const FreeVars *F, const double *xf, double alpha, double *out) {
    for (int f = 0; f < F->nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) out[B->lp_con[a]] += alpha * xf[f] * B->lp_val[a];
    }
}

/* Augmented-Lagrangian form of the bordered system (default): M_W = M + Af W Af' is
 * assembled with the rest of the Schur complement (each free variable adds the clique of
 * its constraints, like an LP variable with a large weight, so the dense and the sparse
 * Schur paths both apply and M_W stays positive definite where M alone is singular) and
 * factored as usual. The regularized system  M dy + Af dxf = r1, Af'dy - W^{-1} dxf = r2
 * gives  M_W dy = r1 + Af W r2,  dxf = W (Af'dy - r2);  iterative refinement against the
 * true bordered operator removes the W^{-1} perturbation (contraction ~ 1/(1 + w s),
 * s the curvature A_f' M^{-1} A_f along the free direction).                            */
static double asm_diag(const Schur *S, const double *Ma, int i) {
    if (g_schol) return schol_get(g_schol, i, i);
    if (g_env) { const Envelope *E = g_env; const int pi = E->iperm[i]; return E->val[E->rptr[pi] + (pi - E->fst[pi])]; }
    return Ma[i + (size_t)i * S->m];
}
static void free_al_assemble(const Problem *P, const Schur *Sc, FreeVars *F, double *M, double omega) {
    const int m = P->m;
    if (!F->wf) F->wf = malloc(sizeof(double) * F->nf);
    double dsum = 0; int dn = 0;
    for (int i = 0; i < m; i++) { double d = asm_diag(Sc, M, i); if (d > 0) { dsum += d; dn++; } }
    const double dmean = dn ? dsum / dn : 1.0;
    for (int f = 0; f < F->nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        double a2 = 0, dd = 0; int nd = 0;
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) {
            a2 += B->lp_val[a] * B->lp_val[a];
            double d = asm_diag(Sc, M, B->lp_con[a]);
            if (d > 0) { dd += d; nd++; }
        }
        const double dbar = nd ? dd / nd : dmean;
        F->wf[f] = a2 > 0 ? omega * dbar / a2 : 0.0;
    }
    for (int f = 0; f < F->nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        const double w = F->wf[f];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) {
            const size_t ca = B->lp_con[a];
            const double va = B->lp_val[a] * w;
            for (int b = a; b < B->lp_ptr[p + 1]; b++) MADDL(B->lp_con[b], ca, va * B->lp_val[b]);
        }
    }
    F->al = 1;
}

/* Dense Schur complement: factor the saddle matrix K = [M Af; Af' 0] itself (symmetric
 * indefinite, Bunch-Kaufman, after a symmetric diagonal scaling). M alone is often nearly
 * singular here (constraints held mostly by free variables) while K is not, so this is
 * much more accurate than going through M^{-1}; it also replaces the factorization of M. */
static double g_tfa = 0, g_tfs = 0; static int g_nfs = 0, g_nfp = 0;
/* the equilibrated saddle matrix [M Af; Af' 0] (lower triangle) into F->K; the columns of M on the threads */
static void free_aug_build(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf, N = m + nf;
    double *K = F->K, *ks = F->ks;
    for (int i = 0; i < m; i++) { double d = schur_dense_get(Sc, i, i); ks[i] = d > 0 ? 1.0 / sqrt(d) : 1.0; }
    #pragma omp parallel for schedule(dynamic, 16) if (N >= 512 && !omp_in_parallel())
    for (int j = 0; j < N; j++) {
        double *kj = K + (size_t)j * N;
        memset(kj, 0, sizeof(double) * (size_t)N);
        if (j < m) for (int i = j; i < m; i++) kj[i] = schur_dense_get(Sc, i, j) * ks[i] * ks[j];
    }
    for (int f = 0; f < nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        double nrm = 0;
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) { double v = B->lp_val[a] * ks[B->lp_con[a]]; nrm += v * v; }
        ks[m + f] = nrm > 0 ? 1.0 / sqrt(nrm) : 1.0;
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++)       /* row m+f, lower part: K[m+f, con] */
            K[(m + f) + (size_t)B->lp_con[a] * N] += B->lp_val[a] * ks[B->lp_con[a]] * ks[m + f];
    }
}
static int free_factor_aug(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf, N = m + nf;
    double t0 = wtime();
    if (!F->K) {
        F->K = amalloc(sizeof(double) * (size_t)N * N);
        F->ks = malloc(sizeof(double) * N);
        F->ipiv = malloc(sizeof(int) * N);
    }
    double *K = F->K;
    /* 18 October 2026: dsytrf's algorithm with its trailing updates on the threads (brisk_dsytrf, the same bits at
     * any thread count; a sphere SOS's saddle factorizations 13.4 s on one core). BRISK_SYTRF=0: LAPACK's */
    static int sy_env = -1; if (sy_env < 0) { const char *e = getenv("BRISK_SYTRF"); sy_env = !(e && atoi(e) == 0); }
    /* 19 October 2026: from order 512 (below it the threads' start costs more than a solve: an occupation-measure
     * file with a saddle system of order 199 was 12 % slower); below, LAPACK's dsytrf and dsytrs as before */
    const int sy = sy_env && N >= 512;
    int info = 0, lwork = 1;
    double *work = NULL;
    if (!sy || getenv("BRISK_SYTRFCHK")) {
        double wq; lwork = -1;
        BL(dsytrf_)("L", &N, K, &N, F->ipiv, &wq, &lwork, &info);
        lwork = (int)wq + 1;
        work = malloc(sizeof(double) * lwork);
    }
    /* singular saddle matrix (dependent free columns, empty rows): quasi-definite shift
     * +d on the y block, -d on the free block; the refinement works with the true one.
     * Start from the shift that worked last time (singularity is structural); the matrix is built
     * again for a retry (was: a copy of it kept at every factorization) */
    for (double d = F->reg; d <= 1e-6; d = d ? d * 100 : 1e-12) {
        free_aug_build(P, Sc, F);
        if (d > 0) {
            for (int i = 0; i < m; i++) K[i + (size_t)i * N] += d;
            for (int f = 0; f < nf; f++) K[(m + f) + (size_t)(m + f) * N] -= d;
        }
        if (getenv("BRISK_SYTRFCHK")) {
            double *Kc = malloc(sizeof(double) * (size_t)N * N); int *pc = malloc(sizeof(int) * N), ic = 0;
            memcpy(Kc, K, sizeof(double) * (size_t)N * N);
            BL(dsytrf_)("L", &N, Kc, &N, pc, work, &lwork, &ic);
            info = brisk_dsytrf(N, K, N, F->ipiv);
            size_t nd = 0; for (int j = 0; j < N; j++) for (int i = j; i < N; i++) nd += memcmp(&K[i + (size_t)j * N], &Kc[i + (size_t)j * N], 8) != 0;
            printf("     [sytrf check N %d: info %d/%d, pivots %s, entries differing %zu]\n", N, info, ic, memcmp(pc, F->ipiv, sizeof(int) * N) ? "differ" : "equal", nd);
            free(Kc); free(pc);
        } else if (sy) info = brisk_dsytrf(N, K, N, F->ipiv);
        else BL(dsytrf_)("L", &N, K, &N, F->ipiv, work, &lwork, &info);
        F->reg = d;
        if (info == 0) break;
    }
    free(work);
    /* (the solves with the threads: the factor converted once, brisk_dsytrs1; BRISK_SYTRF=0: dsytrs) */
    F->kconv = 0;
    if (sy && info == 0) {
        if (!F->ke) F->ke = malloc(sizeof(double) * N);
        if (getenv("BRISK_SYTRSCHK")) {
            double *Kc = malloc(sizeof(double) * (size_t)N * N), *z1 = malloc(sizeof(double) * N), *z2 = malloc(sizeof(double) * N);
            memcpy(Kc, K, sizeof(double) * (size_t)N * N);
            for (int i = 0; i < N; i++) z1[i] = z2[i] = sin(1.0 + i);
            int one_ = 1, inf2 = 0;
            BL(dsytrs_)("L", &N, &one_, Kc, &N, F->ipiv, z1, &N, &inf2);
            brisk_dsyconv(N, K, N, F->ipiv, F->ke);
            brisk_dsytrs1(N, K, N, F->ipiv, F->ke, z2);
            double dm = 0, xm = 0; for (int i = 0; i < N; i++) { dm = fmax(dm, fabs(z1[i] - z2[i])); xm = fmax(xm, fabs(z1[i])); }
            printf("     [sytrs check N %d: largest difference %.2e of %.2e]\n", N, dm, xm);
            free(Kc); free(z1); free(z2);
        } else brisk_dsyconv(N, K, N, F->ipiv, F->ke);
        F->kconv = 1;
    }
    F->aug = 1;
    g_tfa += wtime() - t0;
    return info;
}

/* after the factorization of M: Y = M^{-1} Af, Sf = chol(Af' Y + reg) */
static int free_factor(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf;
    if (!F->Y) F->Y = amalloc(sizeof(double) * (size_t)m * nf);
    if (!F->Sf) F->Sf = amalloc(sizeof(double) * (size_t)nf * nf);
    for (int f = 0; f < nf; f++) {
        double *col = F->w;
        memset(col, 0, sizeof(double) * m);
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) col[B->lp_con[a]] += B->lp_val[a];
        schur_solve(Sc, col, F->Y + (size_t)f * m, 0);
    }
    for (int f = 0; f < nf; f++) free_AT(P, F, F->Y + (size_t)f * m, F->Sf + (size_t)f * nf);
    symmetrize(nf, F->Sf);
    /* equilibrated factor: Sf = D Se D, D = diag(Sf)^{1/2}; regularized only if needed */
    double *T = malloc(sizeof(double) * (size_t)nf * nf);
    if (!F->sd) F->sd = malloc(sizeof(double) * nf);
    for (int f = 0; f < nf; f++) {
        double d = F->Sf[f + (size_t)f * nf];
        F->sd[f] = d > 0 ? 1.0 / sqrt(d) : 1.0;
    }
    for (int j = 0; j < nf; j++)
        for (int i = 0; i < nf; i++) F->Sf[i + (size_t)j * nf] *= F->sd[i] * F->sd[j];
    int info = 1;
    for (double reg = 0; reg < 1e-2 && info; reg = reg ? reg * 100 : 1e-14) {
        memcpy(T, F->Sf, sizeof(double) * (size_t)nf * nf);
        for (int f = 0; f < nf; f++) T[f + (size_t)f * nf] += reg;
        BL(dpotrf_)("L", &nf, T, &nf, &info);
        F->reg = reg;
    }
    memcpy(F->Sf, T, sizeof(double) * (size_t)nf * nf);
    free(T);
    return info;
}

/* Sparse quasi-definite form of the bordered system: K = [M Af; Af' -delta I] on the
 * pattern of the Schur complement plus one node per free variable, symmetrically
 * equilibrated, factored as L S L' (S = +1 on the M part, -1 on the free part; any
 * ordering of a quasi-definite matrix factors without pivoting). The free nodes are
 * eliminated with their constraints, so the factor costs about what M alone does. This is
 * what the moment form of the chordal conversion needs: its free variables are the whole
 * original y (7019 on the 793-bus AC-OPF), M alone is singular, and the augmented-
 * Lagrangian passes of the other mode did not converge there (the residual of the free
 * rows stalled at 4e-6).                                                               */
static int free_ks_setup(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf, N = m + nf;
    long *aptr = NULL; int *adj = NULL;
    if (!Sc->SC || schur_pattern(P, &aptr, &adj)) return 1;
    int *cnt = calloc(N + 1, sizeof(int));
    for (int i = 0; i < m; i++) cnt[i] = (int)(aptr[i + 1] - aptr[i]);
    for (int f = 0; f < nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) { cnt[B->lp_con[a]]++; cnt[m + f]++; }
    }
    int **nbr = malloc(sizeof(int *) * (N + 1)), *deg = calloc(N + 1, sizeof(int));
    for (int i = 0; i < N; i++) nbr[i] = malloc(sizeof(int) * (cnt[i] + 1));
    for (int i = 0; i < m; i++) for (long q = aptr[i]; q < aptr[i + 1]; q++) nbr[i][deg[i]++] = adj[q];
    for (int f = 0; f < nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) {
            const int c = B->lp_con[a];
            nbr[c][deg[c]++] = m + f;
            nbr[m + f][deg[m + f]++] = c;
        }
    }
    const size_t cap = (size_t)fmin(3e7 * brisk_mem_scale(), 0.45 * (double)N * N) + 1000;
    { extern int g_nd_block; g_nd_block = !getenv("BRISK_NDKS"); }
    F->KS = schol_analyze_adj(N, deg, nbr, cap);
    { extern int g_nd_block; g_nd_block = 0; }
    for (int i = 0; i < N; i++) free(nbr[i]);
    free(nbr); free(deg); free(cnt);
    if (!F->KS) { free(aptr); free(adj); return 1; }
    schol_set_inplace(F->KS, 1);       /* its values are refilled before every factorization */
    signed char *sg = malloc(N + 1);
    for (int i = 0; i < N; i++) sg[i] = i < m ? 1 : -1;
    schol_set_signs(F->KS, sg);
    free(sg);
    /* maps: M entries (lower incl. diagonal), Af entries, free diagonals */
    size_t nk = m;
    for (int i = 0; i < m; i++) for (long q = aptr[i]; q < aptr[i + 1]; q++) if (adj[q] < i) nk++;
    F->km_sc = malloc(sizeof(size_t) * nk); F->km_k = malloc(sizeof(size_t) * nk);
    F->km_r = malloc(sizeof(int) * nk); F->km_c = malloc(sizeof(int) * nk);
    size_t w = 0;
    for (int i = 0; i < m; i++) {
        for (long q = aptr[i]; q <= aptr[i + 1]; q++) {
            const int j = q < aptr[i + 1] ? adj[q] : i;
            if (j > i) continue;
            size_t o1 = schol_offset(Sc->SC, i, j), o2 = schol_offset(F->KS, i, j);
            if (o1 == SIZE_MAX || o2 == SIZE_MAX) continue;
            F->km_sc[w] = o1; F->km_k[w] = o2; F->km_r[w] = i; F->km_c[w] = j; w++;
        }
    }
    F->nkm = w;
    free(aptr); free(adj);
    size_t na = 0;
    for (int f = 0; f < nf; f++) { const Block *B = &P->blk[F->kb[f]]; na += B->lp_ptr[F->ip[f] + 1] - B->lp_ptr[F->ip[f]]; }
    F->kaf_off = malloc(sizeof(size_t) * (na + 1)); F->kaf_val = malloc(sizeof(double) * (na + 1));
    F->kaf_r = malloc(sizeof(int) * (na + 1)); F->kaf_f = malloc(sizeof(int) * (na + 1));
    F->kdg = malloc(sizeof(size_t) * (nf + 1));
    w = 0;
    for (int f = 0; f < nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) {
            F->kaf_off[w] = schol_offset(F->KS, m + f, B->lp_con[a]);
            F->kaf_val[w] = B->lp_val[a]; F->kaf_r[w] = B->lp_con[a]; F->kaf_f[w] = f; w++;
        }
        F->kdg[f] = schol_offset(F->KS, m + f, m + f);
    }
    F->naf = w;
    F->kd = malloc(sizeof(double) * (N + 1));
    F->kz = malloc(sizeof(double) * (N + 1));
    F->kr = malloc(sizeof(double) * (N + 1));
    { const char *e = getenv("BRISK_KSDELTA"); F->kdelta = e ? atof(e) : 1e-9; }
    return 0;
}

static int free_ks_factor(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf;
    double t0 = wtime();
    const double *mv = schol_values(Sc->SC);
    double *kv = schol_values(F->KS);
    double *d = F->kd;
    for (int i = 0; i < m + nf; i++) d[i] = 0;
    for (size_t q = 0; q < F->nkm; q++) if (F->km_r[q] == F->km_c[q]) { double x = mv[F->km_sc[q]]; d[F->km_r[q]] = x > 0 ? 1.0 / sqrt(x) : 1.0; }
    for (int i = 0; i < m; i++) if (d[i] == 0) d[i] = 1.0;
    for (size_t q = 0; q < F->naf; q++) { double v = F->kaf_val[q] * d[F->kaf_r[q]]; d[m + F->kaf_f[q]] += v * v; }
    for (int f = 0; f < nf; f++) d[m + f] = d[m + f] > 0 ? 1.0 / sqrt(d[m + f]) : 1.0;
    int ok = 0;
    for (int attempt = 0; attempt < 6 && !ok; attempt++) {
        schol_zero(F->KS);
        for (size_t q = 0; q < F->nkm; q++) kv[F->km_k[q]] = mv[F->km_sc[q]] * d[F->km_r[q]] * d[F->km_c[q]];
        for (size_t q = 0; q < F->naf; q++) kv[F->kaf_off[q]] += F->kaf_val[q] * d[F->kaf_r[q]] * d[m + F->kaf_f[q]];
        for (int f = 0; f < nf; f++) kv[F->kdg[f]] = -F->kdelta;
        /* primal regularization of the M part too: M alone is singular near the end (its
         * null directions are held by Af), and a quasi-definite matrix needs M + dp I > 0 */
        for (size_t q = 0; q < F->nkm; q++) if (F->km_r[q] == F->km_c[q]) kv[F->km_k[q]] += F->kdelta;
        if (schol_factor(F->KS, 0.0) == 0) ok = 1;
        else F->kdelta *= 10;
    }
    g_tfa += wtime() - t0;
    if (getenv("BRISK_FVDBG")) printf("       [ks factor: delta %.1e ok %d]\n", F->kdelta, ok);
    return ok ? 0 : 1;
}

/* Bordered split pairs. The pairs stay in the method (x+, x-, their complementarity), only
 * the linear algebra changes: their weights D = x+/z+ + x-/z- grow like 1/mu, and in M =
 * M_K + Af D Af' they swamp the other entries of the rows they touch, whose rounding
 * (eps * D) then caps the accuracy of A(dX) at the end (primal residual stuck at 1e-6 on
 * the TSSOS models, steps collapsing). K = [M_K Af; Af' -D^{-1}] is quasi-definite and
 * holds no large entries; eliminating the border gives M exactly. Factored with the
 * signed sparse Cholesky on the pattern of free_ks_setup, solved with refinement on K. */
double g_bf_fact = 0, g_bf_all = 0; long g_bf_n = 0;
double g_ph[10], g_ph_last;
double g_pt[8];   /* (BRISK_SOLDBG) predictor parts: second-order term, endpoints, sigma search of the passes, -, -, passes */
static double g_hsdg0 = 0.9;   /* (test, BRISK_HSDG0) floor of the step factor in the embedding */
int g_xtri = 1;   /* 4.24: on by default (lock-free small triangular kernels) */
/* 4.22: small recursive dtrsm (BRISK_XTRI=1): 2.5-5x faster per call, no measurable gain on the runs */
int g_xsymm_loops = 0;   /* 4.23: BRISK_XSYMMLOOPS restores the inline loops of xsymm */
int g_bord_minit = -1;
static int bord_factor(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf;
    double t0 = wtime();
    Sc->ok_rel = INFINITY;
    const double *mv = schol_values(Sc->SC);
    double *kv = schol_values(F->KS);
    double *d = F->kd;
    for (int i = 0; i < m + nf; i++) d[i] = 0;
    for (size_t q = 0; q < F->nkm; q++) if (F->km_r[q] == F->km_c[q]) { double x = mv[F->km_sc[q]]; d[F->km_r[q]] = x > 0 ? 1.0 / sqrt(x) : 1.0; }
    for (int i = 0; i < m; i++) if (d[i] == 0) d[i] = 1.0;
    for (size_t q = 0; q < F->naf; q++) { double v = F->kaf_val[q] * d[F->kaf_r[q]]; d[m + F->kaf_f[q]] += v * v; }
    for (int f = 0; f < nf; f++) {
        /* border row scale: the larger of |a_f|_D and D^{-1/2}, so the -D^{-1} pivot is O(1) or less */
        const double a2 = d[m + f], e = Sc->bE[f];
        d[m + f] = 1.0 / sqrt(fmax(a2, e) > 0 ? fmax(a2, e) : 1.0);
    }
    int ok = 0;
    double delta = 0;
    for (int attempt = 0; attempt < 8 && !ok; attempt++) {
        schol_zero(F->KS);
        for (size_t q = 0; q < F->nkm; q++) kv[F->km_k[q]] = mv[F->km_sc[q]] * d[F->km_r[q]] * d[F->km_c[q]];
        for (size_t q = 0; q < F->naf; q++) kv[F->kaf_off[q]] += F->kaf_val[q] * d[F->kaf_r[q]] * d[m + F->kaf_f[q]];
        for (int f = 0; f < nf; f++) kv[F->kdg[f]] = -Sc->bE[f] * d[m + f] * d[m + f] - delta;
        if (delta > 0) for (size_t q = 0; q < F->nkm; q++) if (F->km_r[q] == F->km_c[q]) kv[F->km_k[q]] += delta;
        double tf0 = wtime();
        if (schol_factor(F->KS, 0.0) == 0) ok = 1;
        else delta = delta ? delta * 100 : 1e-14;
        g_bf_fact += wtime() - tf0;
    }
    Sc->reg = delta;
    if (delta > 0) Sc->nreg++;
    Sc->is_float = 0;
    Sc->t_fact += wtime() - t0;
    g_bf_all += wtime() - t0; g_bf_n++;
    return ok ? 0 : 1;
}
/* y = M x = M_K x + Af D Af' x (diagnostics and callers that need M itself) */
static void bord_mv(Schur *S, const double *x, double *y) {
    const Problem *P = S->bP;
    FreeVars *F = S->bF;
    schol_mv(S->SC, x, y);
    free_AT(P, F, x, F->t);
    for (int f = 0; f < F->nf; f++) F->t[f] /= S->bE[f];
    free_A(P, F, F->t, 1.0, y);
}
/* x = M^{-1} rhs through K [x; t] = [rhs; 0], refined on K itself (no large entries) */
static int g_bdbg = -1, g_bskip = -1, g_btol = -1; static double g_btolcap = -1; static long g_bh0[5], g_bhp[6], g_b1ok, g_b1bad; static double g_bprev0, g_b1worst;
static double bord_solve(Schur *S, const double *rhs, double *x, double tol_rel) {
    if (g_bdbg < 0) g_bdbg = getenv("BRISK_SOLDBG") != NULL;
    if (g_bskip < 0) g_bskip = getenv("BRISK_BSKIP0") == NULL;
    if (g_btol < 0) g_btol = getenv("BRISK_BORDTOL") != NULL;   /* opt-in: 2-20% of the solve time, iteration counts +-1 */
    if (g_btolcap < 0) { const char *e = getenv("BRISK_BORDTOLCAP"); g_btolcap = e ? atof(e) : 1e-10; }
    const Problem *P = S->bP;
    FreeVars *F = S->bF;
    const int m = S->m, nf = F->nf, N = m + nf;
    double *z = F->kz, *res = F->kr, *tt = F->xf;
    double *wk = malloc(sizeof(double) * (N + 1)), *sol = calloc(N + 1, sizeof(double));
    double nb = sqrt(ddot_n(m, rhs, rhs));
    S->nsolve++;
    if (!(nb > 0)) { memset(x, 0, sizeof(double) * m); free(wk); free(sol); return 0; }
    memcpy(res, rhs, sizeof(double) * m);
    for (int f = 0; f < nf; f++) res[m + f] = 0;
    double rel = 1, prev = INFINITY;
    if (g_looseskip && tol_rel >= 1e-8 && S->ok_rel <= 1e-3 * tol_rel) {
        for (int i = 0; i < N; i++) z[i] = res[i] * F->kd[i];
        schol_solve(F->KS, z, wk);
        for (int i = 0; i < m; i++) x[i] = z[i] * F->kd[i];
        g_sc_cnt[0]++; free(wk); free(sol); return 0;
    }
    for (int pass = 0; pass < 6; pass++) {
        for (int i = 0; i < N; i++) z[i] = res[i] * F->kd[i];
        schol_solve(F->KS, z, wk);
        for (int i = 0; i < N; i++) sol[i] += z[i] * F->kd[i];
        /* residual of K: [rhs - M_K x - Af t; -(Af'x - E t)] */
        memcpy(res, rhs, sizeof(double) * m);
        schol_mv(S->SC, sol, S->e3);
        for (int i = 0; i < m; i++) res[i] -= S->e3[i];
        free_A(P, F, sol + m, -1.0, res);
        free_AT(P, F, sol, tt);
        for (int f = 0; f < nf; f++) res[m + f] = -(tt[f] - S->bE[f] * sol[m + f]);
        const double nr = sqrt(ddot_n(N, res, res));
        rel = nr / nb;
        if (pass == 0 && rel < S->ok_rel) S->ok_rel = rel;
        if (pass) S->pcg_iters++;
        if (g_bdbg) { if (pass == 0) { int b = rel <= 1e-13 ? 0 : rel <= 1e-11 ? 1 : rel <= 1e-9 ? 2 : rel <= 1e-7 ? 3 : 4; g_bh0[b]++; g_bprev0 = rel; }
                      else if (pass == 1 && g_bprev0 > 1e-13 && g_bprev0 <= 1e-11) { if (rel <= 1e-13) g_b1ok++; else { g_b1bad++; if (rel > g_b1worst) g_b1worst = rel; } } }
        /* 4.22: the caller's tolerance (correctors ask for 1e-7) */
        if (rel <= fmax(1e-13, g_btol ? fmin(tol_rel, g_btolcap) : 0.0) || !(nr < 0.5 * prev)) { if (g_bdbg) g_bhp[pass < 5 ? pass : 5]++; break; }
        /* 4.22: from <= 1e-11 one more pass is applied without its residual check (a
         * product with M and the border saved). Refinement stagnates at that level anyway
         * (TSSOS case2869: from (1e-13, 1e-11] one pass reached 1e-13 in 27 of 109 solves,
         * the others stayed at 1e-12..1e-11 and stopped); the pre-pass value is reported */
        if (pass == 0 && rel <= 1e-11 && g_bskip) {
            for (int i = 0; i < N; i++) z[i] = res[i] * F->kd[i];
            schol_solve(F->KS, z, wk);
            for (int i = 0; i < N; i++) sol[i] += z[i] * F->kd[i];
            S->pcg_iters++;
            if (g_bdbg) g_bhp[1]++;
            break;
        }
        prev = nr;
    }
    memcpy(x, sol, sizeof(double) * m);
    free(wk); free(sol);
    if (rel > S->worst_rel) S->worst_rel = rel;
    return rel;
}

/* [M Af; Af' 0] [dy; dxf] = [r1; r2] (r2 NULL = 0), with one refinement pass */
static void free_solve(const Problem *P, Schur *Sc, FreeVars *F, const double *r1, const double *r2,
                       double *dy, double *dxf) {
    const int m = P->m, nf = F->nf, one = 1;
    int info;
    if (F->KS) {
        /* right-preconditioned GMRES on the true bordered operator K0 = [M Af; Af' 0], the
         * preconditioner being the regularized quasi-definite factor (plain refinement
         * diverged once M turned singular: the few outlying eigenvalues of K~^-1 K0 are
         * what GMRES removes in a few steps) */
        double t0 = wtime(); g_nfs++;
        const int N = m + nf, KM = 30;
        double *V = malloc(sizeof(double) * (size_t)N * (KM + 1)), *Zp = malloc(sizeof(double) * (size_t)N * KM);
        double *H = calloc((size_t)(KM + 1) * KM, sizeof(double)), *cs = malloc(sizeof(double) * KM), *sn = malloc(sizeof(double) * KM);
        double *g = calloc(KM + 1, sizeof(double)), *wk = malloc(sizeof(double) * (N + 1)), *xx = calloc(N + 1, sizeof(double));
        double *res = F->kr;
        double nb = 0;
        for (int i = 0; i < m; i++) nb += r1[i] * r1[i];
        for (int f = 0; f < nf; f++) nb += r2 ? r2[f] * r2[f] : 0.0;
        nb = sqrt(nb);
#define K0MV(xin, yout) do { \
            memset(yout, 0, sizeof(double) * N); \
            schur_mv(Sc, 1.0, xin, 0.0, yout); \
            free_A(P, F, (xin) + m, 1.0, yout); \
            free_AT(P, F, xin, (yout) + m); } while (0)
        for (int cycle = 0; cycle < 3 && nb > 0; cycle++) {
            /* residual */
            K0MV(xx, wk);
            for (int i = 0; i < m; i++) res[i] = r1[i] - wk[i];
            for (int f = 0; f < nf; f++) res[m + f] = (r2 ? r2[f] : 0.0) - wk[m + f];
            double beta = sqrt(ddot_n(N, res, res));
            if (getenv("BRISK_FVDBG")) printf("       [free_solve gmres cycle %d: |r| %.2e |res| %.2e]\n", cycle, nb, beta);
            if (!(beta > 1e-13 * nb)) break;
            for (int i = 0; i < N; i++) V[i] = res[i] / beta;
            memset(g, 0, sizeof(double) * (KM + 1));
            g[0] = beta;
            int k = 0;
            for (; k < KM; k++) {
                double *z = Zp + (size_t)k * N, *v = V + (size_t)k * N, *vn = V + (size_t)(k + 1) * N;
                for (int i = 0; i < N; i++) z[i] = v[i] * F->kd[i];
                schol_solve(F->KS, z, wk);
                for (int i = 0; i < N; i++) z[i] *= F->kd[i];
                K0MV(z, vn);
                for (int j = 0; j <= k; j++) {              /* modified Gram-Schmidt */
                    const double *vj = V + (size_t)j * N;
                    double h = ddot_n(N, vn, vj);
                    H[j + (size_t)k * (KM + 1)] = h;
                    for (int i = 0; i < N; i++) vn[i] -= h * vj[i];
                }
                double hn = sqrt(ddot_n(N, vn, vn));
                H[k + 1 + (size_t)k * (KM + 1)] = hn;
                if (hn > 0) for (int i = 0; i < N; i++) vn[i] /= hn;
                for (int j = 0; j < k; j++) {               /* previous rotations */
                    double *hc = H + (size_t)k * (KM + 1);
                    double t = cs[j] * hc[j] + sn[j] * hc[j + 1];
                    hc[j + 1] = -sn[j] * hc[j] + cs[j] * hc[j + 1];
                    hc[j] = t;
                }
                double *hc = H + (size_t)k * (KM + 1);
                double den = hypot(hc[k], hc[k + 1]);
                cs[k] = den > 0 ? hc[k] / den : 1.0; sn[k] = den > 0 ? hc[k + 1] / den : 0.0;
                hc[k] = den; hc[k + 1] = 0;
                g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
                g_nfp++;
                if (fabs(g[k + 1]) <= 1e-14 * nb || hn == 0) { k++; break; }
            }
            /* solve the triangle, update x */
            double *yv = calloc(k + 1, sizeof(double));
            for (int j = k - 1; j >= 0; j--) {
                double t = g[j];
                for (int q = j + 1; q < k; q++) t -= H[j + (size_t)q * (KM + 1)] * yv[q];
                yv[j] = H[j + (size_t)j * (KM + 1)] != 0 ? t / H[j + (size_t)j * (KM + 1)] : 0.0;
            }
            for (int j = 0; j < k; j++) { const double *z = Zp + (size_t)j * N; for (int i = 0; i < N; i++) xx[i] += yv[j] * z[i]; }
            free(yv);
            if (fabs(g[k]) <= 1e-13 * nb) break;
        }
#undef K0MV
        memcpy(dy, xx, sizeof(double) * m);
        memcpy(dxf, xx + m, sizeof(double) * nf);
        free(V); free(Zp); free(H); free(cs); free(sn); free(g); free(wk); free(xx);
        g_tfs += wtime() - t0;
        return;
    }
    if (F->al) {
        double t0 = wtime(); g_nfs++;
        double *q1 = malloc(sizeof(double) * m), *q2 = malloc(sizeof(double) * nf);
        double *w = malloc(sizeof(double) * m), *t = malloc(sizeof(double) * nf);
        double n1 = 0, prev = INFINITY;
        for (int i = 0; i < m; i++) { q1[i] = r1[i]; n1 += r1[i] * r1[i]; }
        for (int f = 0; f < nf; f++) { q2[f] = r2 ? r2[f] : 0.0; n1 += q2[f] * q2[f]; }
        memset(dy, 0, sizeof(double) * m);
        memset(dxf, 0, sizeof(double) * nf);
        static int alp = -1;
        if (alp < 0) { const char *e = getenv("BRISK_ALPASS"); alp = e ? atoi(e) : 8; }
        for (int pass = 0; pass < alp; pass++) {
            for (int f = 0; f < nf; f++) t[f] = F->wf[f] * q2[f];
            memcpy(w, q1, sizeof(double) * m);
            free_A(P, F, t, 1.0, w);                               /* q1 + A W q2 */
            double *z = malloc(sizeof(double) * m);
            schur_solve(Sc, w, z, 0);
            free_AT(P, F, z, t);
            for (int f = 0; f < nf; f++) dxf[f] += F->wf[f] * (t[f] - q2[f]);
            for (int i = 0; i < m; i++) dy[i] += z[i];
            free(z);
            /* residual of the true bordered system: M dy = M_W dy - A W A' dy */
            memcpy(q1, r1, sizeof(double) * m);
            schur_mv(Sc, -1.0, dy, 1.0, q1);
            free_AT(P, F, dy, q2);                                 /* A' dy */
            for (int f = 0; f < nf; f++) t[f] = F->wf[f] * q2[f];
            free_A(P, F, t, 1.0, q1);
            free_A(P, F, dxf, -1.0, q1);
            for (int f = 0; f < nf; f++) q2[f] = (r2 ? r2[f] : 0.0) - q2[f];
            double nr = ddot_n(m, q1, q1) + ddot_n(nf, q2, q2);
            g_nfp++;
            if (getenv("BRISK_FVDBG")) printf("       [free_solve al pass %d: |r| %.2e |res| %.2e]\n", pass, sqrt(n1), sqrt(nr));
            if (!(nr > 1e-24 * n1) || !(nr < 0.25 * prev)) break;
            prev = nr;
        }
        free(q1); free(q2); free(w); free(t);
        g_tfs += wtime() - t0;
        return;
    }
    if (F->aug) {
        const int N = m + nf;
        double t0 = wtime(); g_nfs++;
        double *z = malloc(sizeof(double) * N), *res = malloc(sizeof(double) * N);
        double n1 = 0, prev = INFINITY;
        for (int i = 0; i < m; i++) { res[i] = r1[i]; n1 += r1[i] * r1[i]; }
        for (int f = 0; f < nf; f++) { res[m + f] = r2 ? r2[f] : 0.0; n1 += res[m + f] * res[m + f]; }
        memset(dy, 0, sizeof(double) * m);
        memset(dxf, 0, sizeof(double) * nf);
        for (int pass = 0; pass < 4; pass++) {
            for (int i = 0; i < N; i++) z[i] = res[i] * F->ks[i];
            if (F->kconv) brisk_dsytrs1(N, F->K, N, F->ipiv, F->ke, z); else BL(dsytrs_)("L", &N, &one, F->K, &N, F->ipiv, z, &N, &info);
            for (int i = 0; i < m; i++) dy[i] += z[i] * F->ks[i];
            for (int f = 0; f < nf; f++) dxf[f] += z[m + f] * F->ks[m + f];
            /* residual with the true operator */
            memcpy(res, r1, sizeof(double) * m);
            schur_mv(Sc, -1.0, dy, 1.0, res);
            free_A(P, F, dxf, -1.0, res);
            free_AT(P, F, dy, res + m);
            for (int f = 0; f < nf; f++) res[m + f] = (r2 ? r2[f] : 0.0) - res[m + f];
            double nr = ddot_n(N, res, res);
            if (getenv("BRISK_FVDBG")) printf("       [free_solve aug pass %d: |r| %.2e |res| %.2e]\n", pass, sqrt(n1), sqrt(nr));
            g_nfp++;
            if (!(nr > 1e-24 * n1) || !(nr < 0.25 * prev)) break;     /* 1e-12 relative is enough */
            prev = nr;
        }
        free(z); free(res);
        g_tfs += wtime() - t0;
        return;
    }
    double *res1 = malloc(sizeof(double) * m), *res2 = malloc(sizeof(double) * nf);
    double *ey = malloc(sizeof(double) * m), *ex = malloc(sizeof(double) * nf);
    double prev = INFINITY, n1 = 0;
    for (int i = 0; i < m; i++) n1 += r1[i] * r1[i];
    for (int f = 0; f < nf; f++) n1 += r2 ? r2[f] * r2[f] : 0.0;
    for (int pass = 0; pass < 4; pass++) {
        const double *q1 = pass ? res1 : r1;
        const double *q2 = pass ? res2 : r2;
        double *oy = pass ? ey : dy, *ox = pass ? ex : dxf;
        schur_solve(Sc, q1, F->w, 0);
        free_AT(P, F, F->w, ox);
        if (q2) for (int f = 0; f < nf; f++) ox[f] -= q2[f];
        for (int f = 0; f < nf; f++) ox[f] *= F->sd[f];
        BL(dpotrs_)("L", &nf, &one, F->Sf, &nf, ox, &nf, &info);
        for (int f = 0; f < nf; f++) ox[f] *= F->sd[f];
        memcpy(oy, F->w, sizeof(double) * m);
        const double mone = -1.0;
        BL(dgemv_)("N", &m, &nf, &mone, F->Y, &m, ox, &IONE, &DONE, oy, &IONE);
        if (pass) {
            for (int i = 0; i < m; i++) dy[i] += ey[i];
            for (int f = 0; f < nf; f++) dxf[f] += ex[f];
        }
        /* residual of the bordered system */
        memcpy(res1, r1, sizeof(double) * m);
        schur_mv(Sc, -1.0, dy, 1.0, res1);
        free_A(P, F, dxf, -1.0, res1);
        free_AT(P, F, dy, res2);
        for (int f = 0; f < nf; f++) res2[f] = (r2 ? r2[f] : 0.0) - res2[f];
        double nr = 0;
        for (int i = 0; i < m; i++) nr += res1[i] * res1[i];
        for (int f = 0; f < nf; f++) nr += res2[f] * res2[f];
        if (getenv("BRISK_FVDBG")) printf("       [free_solve pass %d: |r| %.2e |res| %.2e]\n", pass, sqrt(n1), sqrt(nr));
        if (pass && !(nr < 0.25 * prev)) {         /* no progress: undo this correction */
            if (nr > prev) {
                for (int i = 0; i < m; i++) dy[i] -= ey[i];
                for (int f = 0; f < nf; f++) dxf[f] -= ex[f];
            }
            break;
        }
        prev = nr;
        if (!(nr > 1e-28 * n1)) break;
    }
    free(res1); free(res2); free(ey); free(ex);
}

/* masked slots of a direction: dX+ = dxf, dX- = 0, dZ+- = 0 */
static void free_set_dir(const FreeVars *F, BS *S, double **DX, double **DZ, const double *dxf, int add) {
    for (int f = 0; f < F->nf; f++) {
        const int k = F->kb[f];
        double *dx = DX ? DX[k] : S[k].dX, *dz = DZ ? DZ[k] : S[k].dZ;
        if (add) dx[F->ip[f]] += dxf[f]; else dx[F->ip[f]] = dxf[f];
        dx[F->im[f]] = 0; dz[F->ip[f]] = 0; dz[F->im[f]] = 0;
    }
}

/* Eigenpairs of the symmetric matrix E (lower triangle used, destroyed) whose eigenvalues
 * lie outside [lo, hi]: Householder tridiagonalization, bisection for the eigenvalues,
 * inverse iteration for the selected vectors and back-transformation of those only, so
 * the cost is 4/3 n^3 + O(n^2 k) instead of the ~9 n^3 of a full decomposition.
 * Returns k (vectors in V, n x k, values in w), or -1 on failure. */
static int eig_outside_full(int n, double *E, double lo, double hi, double *w, double *V) {
    int lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n, info;
    double *work = malloc(sizeof(double) * lwork), *ev = malloc(sizeof(double) * n);
    int *iwork = malloc(sizeof(int) * liwork);
    BL(dsyevd_)("V", "L", &n, E, &n, ev, work, &lwork, iwork, &liwork, &info);
    int k = 0;
    if (info == 0) {
        for (int j = 0; j < n; j++)
            if (ev[j] < lo || ev[j] > hi) { w[k] = ev[j]; memcpy(V + (size_t)k * n, E + (size_t)j * n, sizeof(double) * n); k++; }
    } else k = -1;
    free(work); free(iwork); free(ev);
    return k;
}
static double g_ko[6]; static int g_kostat = -1;
static double g_cp[8];      /* debug (BRISK_KOSTAT): corrector pieces: frame, product, T sum, K solves */      /* debug (BRISK_KOSTAT): calls, ko == 0, 1-10, 11-100, more; time */
static void kostat_print(void) { fprintf(stderr, "   [eig_outside: %.0f calls, outside 0: %.0f, 1-10: %.0f, 11-100: %.0f, more: %.0f; %.3f s; corrector frame %.3f product %.3f T sum %.3f K solves %.3f; tridiagonalization %.3f]\n", g_ko[0], g_ko[1], g_ko[2], g_ko[3], g_ko[4], g_ko[5], g_cp[0], g_cp[1], g_cp[2], g_cp[3], g_cp[4]); }
static int eig_outside_(int n, double *E, double lo, double hi, double *w, double *V);
static int eig_outside(int n, double *E, double lo, double hi, double *w, double *V) {
    if (g_kostat < 0) { g_kostat = getenv("BRISK_KOSTAT") != NULL; if (g_kostat) atexit(kostat_print); }
    const double t0 = g_kostat ? omp_get_wtime() : 0;
    const int k = eig_outside_(n, E, lo, hi, w, V);
    if (g_kostat) {
        #pragma omp critical
        { g_ko[0]++; g_ko[k <= 0 ? 1 : k <= 10 ? 2 : k <= 100 ? 3 : 4]++; g_ko[5] += omp_get_wtime() - t0; }
    }
    return k;
}
static int eig_outside_(int n, double *E, double lo, double hi, double *w, double *V) {
    static int eigfull = -1, eigdbg = -1;
    if (eigfull < 0) { eigfull = getenv("BRISK_EIGFULL") != NULL; eigdbg = getenv("BRISK_EIGDBG") != NULL; }
    if (eigfull) return eig_outside_full(n, E, lo, hi, w, V);
    int info, lwork = -1;
    double wq;
    double *d = malloc(sizeof(double) * 4 * (size_t)n), *e = d + n, *tau = e + n, *wt = tau + n;
    BL(dsytrd_)("L", &n, E, &n, d, e, tau, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    if (lwork < 64 * n) lwork = 64 * n;
    double *work = malloc(sizeof(double) * (size_t)lwork);
    int *iw = malloc(sizeof(int) * 8 * (size_t)n);
    { const double tt0 = omp_get_wtime();
    info = brisk_dsytrd_low(n, E, n, d, e, tau);           /* 5.12: on the threads, fixed panels (dsytrd below 256) */
    g_cp[4] += omp_get_wtime() - tt0; }
    int k = 0;
    if (info != 0) k = -1;
    /* the two tails separately (bisection costs in proportion to the eigenvalues found) */
    double big = 0;
    for (int i = 0; i < n; i++) big = fmax(big, fabs(d[i]) + (i ? fabs(e[i - 1]) : 0) + (i < n - 1 ? fabs(e[i]) : 0));
    big = 2.0 * big + 1.0;
    if (!isfinite(big)) k = -1;
    if (eigdbg && !isfinite(big)) printf("     [eig_outside: non-finite matrix, n %d]\n", n);
    for (int side = 0; side < 2 && k >= 0; side++) {
        double vl = side ? hi : -big, vu = side ? big : lo;
        if (!(vu > vl)) continue;
        int mf = 0, nsplit = 0, zero = 0;
        double abstol = 0;
        int *iblock = iw, *isplit = iw + n, *iwk = iw + 2 * n, *ifail = iw + 5 * n;
        BL(dstebz_)("V", "B", &n, &vl, &vu, &zero, &zero, &abstol, d, e, &mf, &nsplit, wt,
                    iblock, isplit, work, iwk, &info);
        if (info != 0 || mf < 0 || mf > n - k) { k = -1; break; }
        if (mf == 0) continue;
        memcpy(w + k, wt, sizeof(double) * mf);
        BL(dstein_)(&n, d, e, &mf, wt, iblock, isplit, V + (size_t)k * n, &n, work, iwk, ifail, &info);
        if (info < 0) { k = -1; break; }
        k += mf;
    }
    if (k > 0) {
        int lw2 = lwork;
        BL(dormtr_)("L", "L", "N", &n, &k, E, &n, tau, V, &n, work, &lw2, &info);
        if (info != 0) k = -1;
    }
    free(d); free(work); free(iw);
    return k;
}

/* ------------------------------------------------------------------------ */
/* Gondzio centrality corrector inside the self-dual embedding.
 * Trial point at the enlarged step at = min(1, 1.5 am + 0.1) (am the raw step of the
 * current direction); the eigenvalues of the trial complementarity products (in the
 * frame of the direction: MZ with X = L L' for HKM, the NT frame for NT) and the trial
 * tau kappa product that leave [bmin mu_t, bmax mu_t] are pushed back. The shift is made
 * trace-free, so the complementarity decrease mu_new = mu (1 - alpha (1 - sigma)) of the
 * embedding (and the matching residual decrease) is kept. The correction solves the
 * embedded system with zero residuals and complementarity rhs K (dX + Zi dZ X = K, or
 * dX + W dZ W = K), t_c (kappa dtau + tau dkappa = t_c), reusing the factor of M:
 *   M u = -A(K),  dtau = (<C,K> - (b - hc)'u + t_c / tau) / denom,  dy = u + dtau v,
 *   dZ = C dtau - A'dy,  dX = K - sym(Zi dZ X),  dkappa = (t_c - kappa dtau) / tau.
 * The corrected direction is kept if its step grows by the factor acc.
 * Returns 1 if accepted (dX, dZ, dy, dt, dk, am updated).                          */
static int g_cbox = 1;
static int g_pfloor = 1;   /* HSD: stop at the primal residual floor (BRISK_PFLOOR=0 off) */
static int g_brkdn = 1;    /* HSD: stop at a primal breakdown at the accuracy floor (BRISK_BREAKDOWN=0 off) */   /* corrector: skip blocks inside the box (BRISK_NOCBOX) */
static int g_hpar = 0;   /* 4.24: parallel block loops in the embedding (BRISK_HPAR=0 off) */
#define HPAR_FOR _Pragma("omp parallel for schedule(dynamic, 16) if (g_hpar)")
/* Blocks too large to balance in a loop over blocks (g_hbig[k]: the block's dense work n^3 exceeds
 * the total over all blocks divided by the thread count) run after the parallel loop, one at a
 * time, so that the dense products inside them get all threads (pd_run is serial inside a parallel
 * region). Every body is per-block work only, so the order of the blocks does not change the results. */
static unsigned char *g_hbig = NULL; static int g_hbig_n = 0;
#define HBLK_FOR(k, n_, ...) do { if (g_hpar) { _Pragma("omp parallel for schedule(dynamic, 16)") for (int k = 0; k < (n_); k++) { if (g_hbig_n && g_hbig[k]) continue; __VA_ARGS__ } \
                                               if (g_hbig_n) for (int k = 0; k < (n_); k++) { if (!g_hbig[k]) continue; __VA_ARGS__ } } \
                                  else { for (int k = 0; k < (n_); k++) { __VA_ARGS__ } } } while (0)
static double *g_hred = NULL; static int g_hred_n = 0;
static double *hred(int nb, int w) {      /* per-block partials, w per block, zeroed */
    if (nb * w > g_hred_n) { free(g_hred); g_hred_n = nb * w; g_hred = malloc(sizeof(double) * g_hred_n); }
    if (nb * w > 0) memset(g_hred, 0, sizeof(double) * nb * w);
    return g_hred;
}
static double g_ct[12], g_asm[4];   /* 4.23 SOLDBG: corrector parts (pass1, pass2, Aop, solve, back, maxstep) */

/* research hook (4.35): BRISK_DUMPXZ=prefix writes the iterates X/tau, Z/tau of every
 * iteration (scaled problem coordinates) with the row scaling d, bs, cs and y/tau, for offline experiments */
static void dump_xz(const Problem *P, const BS *S, const double *y, int it, double tau) {
    const char *pre = getenv("BRISK_DUMPXZ"); if (!pre) return;
    char fn[1024]; snprintf(fn, sizeof fn, "%s_it%02d.bin", pre, it);
    FILE *f = fopen(fn, "wb"); if (!f) return;
    const int m = P->m, nb = P->nblk;
    fwrite(&m, sizeof(int), 1, f); fwrite(&nb, sizeof(int), 1, f);
    for (int k = 0; k < nb; k++) { fwrite(&P->blk[k].type, sizeof(int), 1, f); fwrite(&P->blk[k].n, sizeof(int), 1, f); }
    for (int k = 0; k < nb; k++) {
        const size_t len = bsz(&P->blk[k]); double *t = malloc(sizeof(double) * len);
        for (size_t i = 0; i < len; i++) t[i] = S[k].X[i] / tau;
        fwrite(t, sizeof(double), len, f);
        for (size_t i = 0; i < len; i++) t[i] = S[k].Z[i] / tau;
        fwrite(t, sizeof(double), len, f);
        free(t);
    }
    fwrite(P->d, sizeof(double), m, f); fwrite(&P->bs, sizeof(double), 1, f); fwrite(&P->cs, sizeof(double), 1, f);
    /* (23 October 2026) and y / tau, scaled, for the Newton system at the iterate (ISSUES 241 prototype) */
    { double *t = malloc(sizeof(double) * (m + 1)); for (int i = 0; i < m; i++) t[i] = y[i] / tau; fwrite(t, sizeof(double), m, f); free(t); }
    fclose(f);
}

/* 5.16 (13 October): the tau column of the embedding, in the original or in shifted variables. The Newton system of the embedding has the
 * column (-b; C) for dtau. Eliminating it through the Schur complement forms <C, H(C)>, A(H(C)) and
 * <C, H(Rd)> (H the scaling, X (x) Z^-1 for HKM, W (x) W for NT), and the scalar of the tau equation is
 * their difference: when X grows without bound (no strictly feasible dual point) these terms reach 1e14
 * while the scalar is 1e-5, and dtau loses every digit. With dy = dy' + (dtau / tau) y the column becomes
 * C^ = C - A'y / tau = (Z + Rd) / tau (Rd = tau C - A'y - Z), and H(C^) = (X + H(Rd)) / tau exactly, since
 * H(Z) = X for HKM, NT and the cones. Every scalar is then formed from X, Z and Rd without the large terms
 * (<C^, Z^-1> uses <Z, Z^-1> = nu, <C^, Q> uses <Z, Q> = <dX, dZ>), and the gap row's right-hand side
 * becomes kap + <C^, X>. Exact in exact arithmetic. The shifted form has its own error: its gap row holds
 * in dy', and in dy up to y'e / tau (e the primal row's miss), large when y grows without bound (no strictly
 * feasible primal point: qap, |y| ~ 4e6). So each iteration takes the form whose error in the gap row was
 * smaller at the previous one, eps <C, H C> |dtau| against |y'e| / tau (ISSUES 84). BRISK_HSDSHIFT=0: always
 * the original column, 2: always shifted. Not with free variables (their column is c_f; the original form). */
typedef struct { int on; double tau, theta; double **F; const char *pk; const double *bmh; double tauf; } HsdShift;
/* with the shift A(H C^) = b + (h_d - rp) / tau is close to b: b - A(H C^) is formed as the vector
 * bmh = (rp - h_d) / tau (the products b'u and hc'u would cancel where M^-1 is large) */
/* <Rd_k, D>: Rd is the copy F[k], or theta R0 on the pattern route */
static double shift_dotF(const HsdShift *sh, const BS *s, int k, size_t len, const double *D) {
    return sh->pk[k] ? sh->theta * dot_R0(s, len, D) : ddot_n(len, sh->F[k], D);
}
/* D += c C^_k = (c / tau) (Z + Rd) */
static void shift_addC(const HsdShift *sh, const BS *s, int k, size_t len, double c, double *D) {
    const double f = c / sh->tau;
    const double *Z = s->Z;
    if (sh->pk[k]) { PAR_LEN_FOR(len, D[i] += f * Z[i];); add_R0(s, len, f * sh->theta, D); }
    else { const double *F = sh->F[k]; PAR_LEN_FOR(len, D[i] += f * (Z[i] + F[i]);); }
}
/* The shifted column applied consistently: the elimination uses H(C^) = (X + H(Rd)) / tau with H(Z) = X taken
 * exactly, so the direction takes it exactly as well. dZ gets c Rd / tau before the scaling is applied and c Z / tau
 * after it, and dX gets -c X / tau in place of the computed -H(c Z / tau): Z^-1 Z X (or W Z W) formed in floating
 * point is X only to eps kappa(Z), and that error went into A(dX) unbalanced by the right-hand side (an option
 * pricing file: the primal residual 5.7e-11 -> 4.2e-7 at one shifted step). Experiment, off (BRISK_SHIFTCONS=1): no
 * effect there, the jump comes from the Schur solve at the end (both forms; ISSUES 84). */
static int shift_cons(void) { static int v = -1; if (v < 0) { const char *e = getenv("BRISK_SHIFTCONS"); v = e ? atoi(e) : 0; } return v; }
/* D += (c / tau) Rd (the part of c C^ that goes through the scaling) */
static void shift_addRd(const HsdShift *sh, const BS *s, int k, size_t len, double c, double *D) {
    const double f = c / sh->tau;
    if (sh->pk[k]) add_R0(s, len, f * sh->theta, D);
    else { const double *F = sh->F[k]; PAR_LEN_FOR(len, D[i] += f * F[i];); }
}
/* before the scaling: c C^ (old) or c Rd / tau (consistent) */
static void shift_pre(const HsdShift *sh, const BS *s, int k, size_t len, double c, double *D) {
    if (shift_cons()) shift_addRd(sh, s, k, len, c, D); else shift_addC(sh, s, k, len, c, D);
}
/* after the scaling (consistent form only): DX += sx (c / tau) X and DZ += (c / tau) Z */
static void shift_post(const HsdShift *sh, const BS *s, size_t len, double c, double sx, double *DX, double *DZ) {
    if (!shift_cons()) return;
    const double f = c / sh->tau, *X = s->X, *Z = s->Z;
    if (DX) PAR_LEN_FOR(len, DX[i] += sx * f * X[i];);
    if (DZ) PAR_LEN_FOR(len, DZ[i] += f * Z[i];);
}
/* nu_k = <Z, Z^-1>: the order of an SDP or LP block, 1 for a cone (x'z = mu on the path) */
static inline double shift_nu(const Block *B) { return B->type == BLK_SOC ? 1.0 : (double)B->n; }

static int hsd_corrector(Problem *P, const Params *par, BS *S, Schur *Sc, double **Zv, const Pat *hp,
                         const char *pk, double tau, double kap, double *dt, double *dk, double *dyd,
                         const double *v, const double *hc, double denom, double *am, double mu_t,
                         double **bX, double **bZ, double *rhs, double *uc, double ndim1, Result *R,
                         FreeVars *FV, const double *vf, double accf, const HsdShift *shc) {
    const int m = P->m, nb = P->nblk;
    const double bmin = par->hsd_cbmin, bmax = par->hsd_cbmax;
    const double ucap = par->hsd_cucap > 0 ? par->hsd_cucap : bmax;   /* largest downward push, in mu_t */
    const double at = fmin(1.0, 1.5 * *am + 0.1);
    double td = wtime(), trT = 0, trial = 0;
    double *pr1 = hred(nb, 2);
    /* pass 1: trial products (LP: in Xn; SDP: eigenpairs in Ct, ev) */
    HBLK_FOR(k, nb, {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        double trial = 0;
        if (B->type == BLK_SOC) {          /* 5.16: <x~, z~> enters the trial mean; the cone is not pushed back */
            for (int i = 0; i < n; i++) trial += (s->X[i] + at * s->dX[i]) * (s->Z[i] + at * s->dZ[i]);
            pr1[2 * (size_t)k] = trial;
            continue;
        }
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) {
                s->Xn[i] = (s->fm && s->fm[i]) ? 0.0 : (s->X[i] + at * s->dX[i]) * (s->Z[i] + at * s->dZ[i]);
                trial += s->Xn[i];
            }
            pr1[2 * (size_t)k] = trial;
            continue;
        }
        double *Xt = s->Xn, *Zt = s->Zn, *E = s->Ct;
        if (Zv[k]) {                      /* NT frame: G^{-1} X~ G^{-T} = lam + at dXt, ... */
            nt_toX(B, s, s->dX, Xt);
            nt_toZ(B, s, s->dZ, Zt);
            PAR_LEN_FOR(len, Xt[i] *= at; Zt[i] *= at;);
            for (int i = 0; i < n; i++) { Xt[i + (size_t)i * n] += s->lam[i]; Zt[i + (size_t)i * n] += s->lam[i]; }
        } else {
            /* the frame of this HKM form (Z dX + dZ X linearized): Z = L L',
             * L' X~ L and L^{-1} Z~ L^{-T} = I + at L^{-1} dZ L^{-T}                  */
            const double *L = s->LZ;
            const double tq0 = omp_get_wtime();
            PAR_LEN_FOR(len, Xt[i] = s->X[i] + at * s->dX[i];);
            xtrmm("L", "L", "T", "N", &n, &n, &DONE, L, &n, Xt, &n);
            xtrmm("R", "L", "N", "N", &n, &n, &DONE, L, &n, Xt, &n);
            memcpy(Zt, s->dZ, sizeof(double) * len);
            xtrsm("L", "L", "N", "N", &n, &n, &at, L, &n, Zt, &n);
            xtrsm("R", "L", "T", "N", &n, &n, &DONE, L, &n, Zt, &n);
            for (int i = 0; i < n; i++) Zt[i + (size_t)i * n] += 1.0;
            if (g_kostat > 0) g_cp[0] += omp_get_wtime() - tq0;
        }
        const double tq1 = omp_get_wtime();
        pdgemm("N", "N", &n, &n, &n, &DONE, Xt, &n, Zt, &n, &DZERO, E, &n);
        if (g_kostat > 0) g_cp[1] += omp_get_wtime() - tq1;
        symmetrize(n, E);
        for (int j = 0; j < n; j++) trial += E[j + (size_t)j * n];
        pr1[2 * (size_t)k] = trial;
    });
    for (int k = 0; k < nb; k++) trial += pr1[2 * (size_t)k];
    { double _t = wtime(); g_ct[0] += _t - td; }
    double tk = (tau + at * *dt) * (kap + at * *dk);
    trial += tk;
    if (mu_t <= 0) mu_t = trial / ndim1;  /* target: the trial mean */
    if (!(mu_t > 0)) { R->t_dense += wtime() - td; return 0; }
    /* pass 2: pushed-back parts and K */
    int eigfail = 0;
    HBLK_FOR(k, nb, {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        double *K = s->Q;
        double trT = 0;
        if (B->type == BLK_SOC) { memset(K, 0, sizeof(double) * len); pr1[2 * (size_t)k + 1] = 0; continue; }
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) {
                double w = s->Xn[i], t = 0;
                if (s->fm && s->fm[i]) { K[i] = 0; continue; }
                if (w < bmin * mu_t) t = bmin * mu_t - w;
                else if (w > bmax * mu_t) t = fmax(bmax * mu_t - w, -ucap * mu_t);
                trT += t;
                K[i] = t / s->Z[i];
            }
            pr1[2 * (size_t)k + 1] = trT;
            continue;
        }
        double *T = s->Xn, *Vv = s->Zn;   /* T = V diag(t) V' over the eigenpairs outside the box */
        /* 4.20: a small block whose trial product lies inside the box (two Cholesky tests,
         * n^3/3) needs no eigenvalues and contributes K = 0 (case1354: most of 2,500 blocks) */
        if (n <= SMALL_BLOCK && g_cbox && small_pd_shift(n, s->Ct, bmin * mu_t, 1.0) && small_pd_shift(n, s->Ct, bmax * mu_t, -1.0)) {
            memset(K, 0, sizeof(double) * len);
            continue;
        }
        const int ko = eig_outside(n, s->Ct, bmin * mu_t, bmax * mu_t, s->ev, Vv);
        if (ko < 0) {
            _Pragma("omp atomic write")
            eigfail = 1;
            continue;
        }
        const double tq2 = omp_get_wtime();
        memset(T, 0, sizeof(double) * len);
        if (ko >= 8 && n >= 128 && !ENV_ON("BRISK_TSUM0")) {
            /* 5.12: T = (V diag(t)) V' by one product (was ko rank-one updates in a scalar loop: up to
             * 100 of order 792 per call); the pushed-back eigenpairs first moved to the front of V */
            int q = 0;
            for (int j = 0; j < ko; j++) {
                double w = s->ev[j], t = 0;
                if (w < bmin * mu_t) t = bmin * mu_t - w;
                else if (w > bmax * mu_t) t = fmax(bmax * mu_t - w, -ucap * mu_t);
                if (t == 0) continue;
                trT += t;
                if (q != j) memcpy(Vv + (size_t)q * n, Vv + (size_t)j * n, sizeof(double) * n);
                s->ev[q++] = t;
            }
            if (q > 0) {
                double *Wt = K;                                     /* (K is free until the end of this block) */
                for (int j = 0; j < q; j++) { const double t = s->ev[j]; const double *vj = Vv + (size_t)j * n; double *wj = Wt + (size_t)j * n; for (int r = 0; r < n; r++) wj[r] = t * vj[r]; }
                pdgemm("N", "T", &n, &n, &q, &DONE, Wt, &n, Vv, &n, &DZERO, T, &n);
                symmetrize(n, T);
            }
        } else
        for (int j = 0; j < ko; j++) {
            double w = s->ev[j], t = 0;
            if (w < bmin * mu_t) t = bmin * mu_t - w;
            else if (w > bmax * mu_t) t = fmax(bmax * mu_t - w, -ucap * mu_t);
            if (t == 0) continue;
            trT += t;
            const double *vj = Vv + (size_t)j * n;
            for (int c = 0; c < n; c++) {
                double f = t * vj[c];
                double *tc = T + (size_t)c * n;
                for (int r = 0; r < n; r++) tc[r] += f * vj[r];
            }
        }
        if (Zv[k]) {                      /* K = G L_lam^{-1}(T) G' */
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) T[i + (size_t)j * n] *= 2.0 / (s->lam[i] + s->lam[j]);
            nt_fromS(B, s, T, K);
        } else {                          /* K = L^{-T} T L^{-1} */
            const double tq3 = omp_get_wtime();
            if (g_kostat > 0) g_cp[2] += tq3 - tq2;
            const double *L = s->LZ;
            memcpy(K, T, sizeof(double) * len);
            xtrsm("L", "L", "T", "N", &n, &n, &DONE, L, &n, K, &n);
            xtrsm("R", "L", "N", "N", &n, &n, &DONE, L, &n, K, &n);
            symmetrize(n, K);
            if (g_kostat > 0) g_cp[3] += omp_get_wtime() - tq3;
        }
        pr1[2 * (size_t)k + 1] = trT;
    });
    if (eigfail) { R->t_dense += wtime() - td; return 0; }
    for (int k = 0; k < nb; k++) trT += pr1[2 * (size_t)k + 1];
    double tp2 = wtime(); g_ct[1] += tp2 - td;
    double tcc = 0;
    if (tk < bmin * mu_t) tcc = bmin * mu_t - tk;
    else if (tk > bmax * mu_t) tcc = fmax(bmax * mu_t - tk, -ucap * mu_t);
    trT += tcc;
    if (trT == 0 && tcc == 0) { R->t_dense += wtime() - td; return 0; }
    /* trace-free: T -= s I in every frame, i.e. K -= s Z^{-1}, t_c -= s */
    const double sh = par->hsd_ctrace ? trT / ndim1 : 0.0;
    tcc -= sh;
    memset(rhs, 0, sizeof(double) * m);
    double CK = 0;
    aop_begin(rhs, NULL, NULL, NULL);
    HBLK_FOR(k, nb, {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const size_t len = bsz(B);
        const double *ZI = Zv[k] ? Zv[k] : s->Zi;
        if (sh != 0) for (size_t i = 0; i < len; i++) s->Q[i] -= sh * ZI[i];
        blk_Aop(B, s, s->Q, rhs, -1.0);
        /* <C^, K - sh Z^-1> = (<Z, K> - sh nu + <Rd, .>) / tau, <Z, K> = tr T (the frame of T is Z's) */
        if (shc->on) pr1[2 * (size_t)k] = (pr1[2 * (size_t)k + 1] - sh * shift_nu(B) + shift_dotF(shc, s, k, len, s->Q)) / tau;
        else pr1[2 * (size_t)k] = sp_inner(B, &B->C, s->Q);
    });
    aop_end();
    for (int k = 0; k < nb; k++) CK += pr1[2 * (size_t)k];
    R->t_dense += wtime() - td;
    double tc0 = wtime(); g_ct[2] += tc0 - tp2;
    double *ufc = FV->nf ? malloc(sizeof(double) * FV->nf) : NULL;
    if (FV->nf) free_solve(P, Sc, FV, rhs, NULL, uc, ufc);
    else schur_solve(Sc, rhs, uc, 1e-7);
    R->t_chol += wtime() - tc0; g_ct[3] += wtime() - tc0;
    const double dtc = shc->tauf * (shc->on ? CK - ddot_n(m, shc->bmh, uc) + tcc / tau : CK - ddot_n(m, P->b, uc) + ddot_n(m, hc, uc) + tcc / tau
                        + (FV->nf ? ddot_n(FV->nf, FV->c, ufc) : 0.0)) / denom;
    /* (experiment BRISK_TAUFREEZE: tau kept in the endgame) */
    const double dkc = (tcc - kap * dtc) / tau;
    for (int i = 0; i < m; i++) uc[i] += dtc * v[i];
    td = wtime();
    HBLK_FOR(k, nb, {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        blk_ATy(B, s, uc, s->F);
        PAR_LEN_FOR(len, s->F[i] = -s->F[i];);
        if (shc->on) shift_pre(shc, s, k, len, dtc, s->F);          /* dZc */
        else sp_add(B, &B->C, dtc, s->F);
        const double *RX = Zv[k] ? s->Zi : s->X;
        if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * s->F[i] * s->X[i];
        else if (B->type == BLK_SOC) soc_H(s, n, s->F, s->G, s->stmp);
        else if (pk[k]) { prod_pat(&hp[k], n, s, s->F, RX, s->G); symmetrize(n, s->G); }
        else { prod_zi(n, s, s->F, RX, s->G); symmetrize(n, s->G); }
        if (shc->on) shift_post(shc, s, len, dtc, 1.0, s->G, s->F);   /* G = H(dZc) with H(Z) = X */
        memcpy(bX[k], s->dX, sizeof(double) * len);
        memcpy(bZ[k], s->dZ, sizeof(double) * len);
        PAR_LEN_FOR(len, s->dX[i] += s->Q[i] - s->G[i]; s->dZ[i] += s->F[i];);
    });
    if (FV->nf) {
        for (int f = 0; f < FV->nf; f++) ufc[f] += dtc * vf[f];
        free_set_dir(FV, S, NULL, NULL, ufc, 1);
        free(ufc);
    }
    R->t_dense += wtime() - td; g_ct[4] += wtime() - td;
    double ts = wtime();
    double a1 = fmin(maxstep(P, S, 1, par->lanczos_k), maxstep(P, S, 0, par->lanczos_k));
    R->t_step += wtime() - ts; g_ct[5] += wtime() - ts;
    const double ndt = *dt + dtc, ndk = *dk + dkc;
    if (ndt < 0) a1 = fmin(a1, -tau / ndt);
    if (ndk < 0) a1 = fmin(a1, -kap / ndk);
    if (ENV_ON("BRISK_HSDDBG"))
        printf("     [corrector: am %.3f -> %.3f (at %.3f, trT %.2e mu_t %.2e)]\n", *am, a1, at, trT, mu_t);
    if (a1 >= fmax(par->hsd_cacc, accf) * *am) {
        for (int i = 0; i < m; i++) dyd[i] += uc[i];
        *dt = ndt; *dk = ndk; *am = a1;
        return 1;
    }
    HBLK_FOR(k, nb, {
        const size_t len = bsz(&P->blk[k]);
        memcpy(S[k].dX, bX[k], sizeof(double) * len);
        memcpy(S[k].dZ, bZ[k], sizeof(double) * len);
    });
    return 0;
}

static double g_kfix0 = 0.3;   /* below this factor/corrector cost ratio: no correctors */
/* 4.25: the time limit, tested after iteration `it`: stop when the next iteration (at the
 * average length so far) would end more than half of it past the limit (thetaG51 at 7 s per
 * iteration otherwise overshot a 1 s limit by one full iteration) */
static int time_up(const Params *par, int it, double t0) {
    if (brisk_stop_flag) return 1;
    if (par->timelimit <= 0) return 0;
    const double now = wtime(), dt = (now - t0) / (it + 1);
    return now - par->t_start + 0.5 * dt > par->timelimit;
}
/* The iterate from which the primal feasibility is restored at the end (polish_xmetric with
 * half = 1). In the last iterations the Schur complement is numerically singular and the Newton
 * direction misses the primal equations by more than the residual it should remove: the
 * complementarity and the dual residual go on falling (the dual direction is exact by
 * construction) while the primal residual rises to 1e-6 and the best iterate by the largest
 * error is an early one. The restoration removes the primal residual and leaves the
 * complementarity where it is, so the iterate worth restoring is the one with the smallest
 * max(dual residual, complementarity) among those whose primal residual is still moderate
 * (RC_PMAX: the correction X^1/2 W X^1/2 keeps X positive definite only while |W| < 1). */
#define RC_PMAX 1e-3
typedef struct { double *y; double score; int it, have; } RCand;
static void rcand_offer(RCand *rc, const Problem *P, BS *S, const double *y, double tau, const Result *R, int it) {
    if (!rc || !rc->y) return;
    const double rs = fmax(nan_inf(R->dinf), nan_inf(fabs(R->relcomp)));
    if (!(R->pinf <= RC_PMAX) || !(rs < rc->score)) return;
    rc->score = rs; rc->it = it; rc->have = 1;
    const double f = 1.0 / tau;
    for (int i = 0; i < P->m; i++) rc->y[i] = y[i] * f;
    for (int k = 0; k < P->nblk; k++) {
        const size_t len = bsz(&P->blk[k]);
        const double *X = S[k].X, *Z = S[k].Z; double *Xr = S[k].Xr, *Zr = S[k].Zr;
        if (S[k].zon && S[k].zpat && S[k].zrc) {      /* 5.12: Z on its pattern (Zr zero elsewhere) */
            PAR_LEN_FOR(len, Xr[i] = X[i] * f;);
            for (int q = 0; q < S[k].nzp; q++) Zr[S[k].zpos[q]] = Z[S[k].zpos[q]] * f;
            continue;
        }
        PAR_LEN_FOR(len, Xr[i] = X[i] * f; Zr[i] = Z[i] * f;);
        if (S[k].zon && S[k].zpat) S[k].zrc = 1;
    }
}
static const char *g_hsd_why = NULL;      /* why the embedding stopped short (printed with the retry: user reports must be diagnosable) */
/* 23 October 2026 (nishioka, the MVU problems; ISSUES 253): the Frobenius norm of X at the embedding's start and of X/tau
 * at its end; their ratio decides the re-solve of an embedding that failed with its primal start at the size the first one
 * revealed */
static double g_hsd_fx0 = 0, g_hsd_fxe = 0;
/* 5.16 (ISSUES 83): BRISK_BITSDBG=1 prints a fingerprint (FNV-1a) of the bits of X, Z, y, tau and
 * kappa at every iteration, to find the first iteration at which two runs differ. Not inlined: code
 * added inside hsd_run changes how the compiler contracts and vectorizes its arithmetic, and so the
 * bits of every run. */
static __attribute__((noinline)) void hsd_bits(const Problem *P, const BS *S, const double *y, double tau, double kap, int it) {
    uint64_t h[3] = { 1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull };
    for (int k = 0; k < P->nblk; k++) {
        const size_t len = bsz(&P->blk[k]);
        for (size_t i = 0; i < len; i++) { uint64_t u, v; memcpy(&u, &S[k].X[i], 8); memcpy(&v, &S[k].Z[i], 8); h[0] = (h[0] ^ u) * 1099511628211ull; h[1] = (h[1] ^ v) * 1099511628211ull; }
    }
    for (int i = 0; i < P->m; i++) { uint64_t u; memcpy(&u, &y[i], 8); h[2] = (h[2] ^ u) * 1099511628211ull; }
    uint64_t ut, uk; memcpy(&ut, &tau, 8); memcpy(&uk, &kap, 8);
    printf("     [bits it %d X %016llx Z %016llx y %016llx tau %016llx kap %016llx]\n", it, (unsigned long long)h[0], (unsigned long long)h[1],
           (unsigned long long)h[2], (unsigned long long)ut, (unsigned long long)uk);
}
static int hsd_run(Problem *P, const Params *par, Result *R, BS *S, Schur *Sc, double *y,
                   double *ybest, double *best_score, int *it_out, int *it_best, Result *best,
                   int keep_best, RCand *rc) {
    const int m = P->m, nb = P->nblk;
    g_hsd_why = NULL;
    for (int k = 0; k < nb; k++) S[k].zpat = 0;        /* 5.12: the pattern passes are the standard loop's (dense passes here) */
    { const char *e = getenv("BRISK_KFIX0"); if (e) g_kfix0 = atof(e); if (getenv("BRISK_NOCBOX")) g_cbox = 0; }
    double tau = 1.0, kap = 1.0, ndim = 0;
    for (int k = 0; k < nb; k++) ndim += P->blk[k].type == BLK_SOC ? 1 : P->blk[k].n;   /* 5.16: a cone counts once (x o z = mu e) */
    if (par->hsd_corr > 0)            /* 4.25: the corrector buffers, on first use */
        for (int k = 0; k < nb; k++) {
            BS *s = &S[k];
            const size_t len = bsz(&P->blk[k]) * sizeof(double);
            if (!s->Xn) s->Xn = amalloc(len);
            if (!s->Zn) s->Zn = amalloc(len);
            if (P->blk[k].type == BLK_SDP) {
                if (!s->Ct) s->Ct = amalloc(len);
                if (!s->ev) s->ev = amalloc(sizeof(double) * (P->blk[k].n + 1));
            }
        }
    { const char *e = getenv("BRISK_XTRI"); if (e) g_xtri = atoi(e); }
    { const char *e = getenv("BRISK_LOCKFREE"); if (e) { g_lockfree = atoi(e); if (!g_lockfree && !getenv("BRISK_XTRI")) g_xtri = 0; } }
    if (getenv("BRISK_XSYMMLOOPS")) g_xsymm_loops = 1;
    if (getenv("BRISK_NOFLAT")) g_noflat = 1;
    if (getenv("BRISK_NOFASTL")) g_nofastl = 1;
    { const char *e = getenv("BRISK_HPAR"); g_hpar = (e ? atoi(e) : 1) && nthreads_cap() > 1 && nb >= 16; }
    {   /* the automatic split of the block loops above (BRISK_HPAR=1 keeps every block in the loop) */
        const char *e = getenv("BRISK_HPAR");
        g_hbig_n = 0;
        if (g_hpar && !e) {
            const int T = nthreads_cap();
            /* (23 October 2026: an LP block's work is its entries, which its products read every pass: a D-optimal design's
             * LP block of 9.9 million entries ran on one thread inside the loop while the 63 small blocks shared the others) */
            double tot = 0;
            for (int k = 0; k < nb; k++) { const double n = P->blk[k].n; tot += P->blk[k].type == BLK_SDP ? n * n * n : (double)blk_nnz(&P->blk[k]); }
            free(g_hbig); g_hbig = calloc((size_t)nb + 1, 1);
            int nbig = 0;
            for (int k = 0; g_hbig && k < nb; k++) {
                const double n = P->blk[k].n;
                g_hbig[k] = P->blk[k].type == BLK_SDP ? (n >= 128 && n * n * n > tot / T)
                                                      : (P->blk[k].ltr_off != NULL && (double)blk_nnz(&P->blk[k]) > tot / T);
                nbig += g_hbig[k];
            }
            g_hbig_n = nbig ? nb : 0;
            if (nbig && par->verbose > 1) printf("   (HSD: %d large block(s) outside the parallel block loops)\n", nbig);
        }
    }
    aop_map_build(P);
    if (getenv("BRISK_TREEDBG") && Sc->SC) { extern void schol_tree_stats(const SChol *); schol_tree_stats(Sc->SC); }
    if (getenv("BRISK_FLATCHK")) {
        double worst = 0; long nbad = 0, ntot = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            if (B->type == BLK_LP || !B->foff) continue;
            for (int a = 0; a < B->ns; a++) {
                const SpSym *S2 = &B->A[B->slist[a]];
                const int k0 = B->foff[a], k1 = B->foff[a + 1];
                if (k1 - k0 != S2->ef) { nbad++; continue; }
                for (int q = 0; q < S2->ef; q++) {
                    ntot++;
                    if (B->ffr[k0 + q] != S2->fr[q] || B->ffc[k0 + q] != S2->fc[q]) { nbad++; continue; }
                    double d = fabs(B->ffv[k0 + q] - S2->fv[q]) / (fabs(S2->fv[q]) + 1e-300);
                    if (d > worst) worst = d;
                }
            }
        }
        printf("   [flat check: %ld entries, %ld mismatched patterns, worst relative value difference %.2e]\n", ntot, nbad, worst);
    }
    /* 4.22: no switch to the bordered factorization on the first iteration: its miss there
     * comes from the starting point (TSSOS case2869 with X0 x 10: 1.7e5 of 1.3e3, then 1e-12
     * until iteration 40), and the switch made all later factorizations the bordered one
     * (2.28e10 against 1.57e10 flops on TSSOS case6468) */
    { const char *e = getenv("BRISK_BORDMINIT"); g_bord_minit = e ? atoi(e) : 1; }
    int acopf_cls = 0;   /* 4.22: AC-OPF-type relaxation (moment form or TSSOS-like), set below */
    {   /* 4.22: primal starting point scaled up on AC-OPF-type relaxations. Their primal
         * optimum (the multipliers of the moment problem) is far larger than the least-norm
         * start while the dual start is too large: TSSOS case2869 trX 6.7e4 -> 2-3e5, trZ
         * 1.0e5 -> 3e4, and the embedding spends ~25 iterations moving the objective (71 its,
         * MOSEK 71; 45 with X0 x 10). Measured on the moment-SOS image/kernel forms and
         * SDPLIB it hurts (control10, attr_vdp_d12_I, roa_acrobot_d4_I fail at 1e-6), so it
         * applies to the class only: the moment form of a chordal conversion (m >= 1000), or
         * many small SDP blocks with split free pairs (TSSOS models). BRISK_HSDX0 overrides. */
        const char *ex = getenv("BRISK_HSDX0"), *ez = getenv("BRISK_HSDZ0");
        double fx = ex ? atof(ex) : par->hsd_x0 > 0 ? par->hsd_x0 : 1.0;
        const double fz = ez ? atof(ez) : 1.0;
        if (!ex && !(par->hsd_x0 > 0)) {
            int nsdp = 0, maxn = 0;
            for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; }
            /* the moment form of a dense conversion too (dense case9241: 63 its, 375 s
             * against 100 its, 615 s; case6468 101 s against 141 s once the collapsed endgame
             * stops early and both polishings are tried) */
            int cls = P->ps && P->ps->mom && m >= 1000;
            if (!cls && nsdp >= 100 && maxn <= 64 && m >= 1000) {
                BS *tmp = calloc(nb + 1, sizeof(BS));
                FreeVars F0;
                const int np = free_detect(P, tmp, &F0);
                free_release(P, tmp, &F0);
                free(tmp);
                cls = np >= 100;
            }
            acopf_cls = cls;
            if (cls) {
                fx = 10.0;
                if (par->verbose > 1) printf("   HSD: AC-OPF-type start (X0 x %g)\n", fx);
            }
        }
        if (fx != 1.0 || fz != 1.0)
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]);
                PAR_LEN_FOR(len, S[k].X[i] *= fx; S[k].Z[i] *= fz;);
            }
        g_hsd_fx0 = 0;
        for (int k = 0; k < nb; k++) g_hsd_fx0 += ddot_n(bsz(&P->blk[k]), S[k].X, S[k].X);
        g_hsd_fx0 = sqrt(g_hsd_fx0);
    }
    if (ENV_ON("BRISK_SCALEDBG")) {
        double tx = 0, tz = 0, nd = 0;
        for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) { tx += S[k].X[i]; tz += S[k].Z[i]; }
            else if (B->type == BLK_SOC) { tx += S[k].X[0]; tz += S[k].Z[0]; }
            else for (int i = 0; i < n; i++) { tx += S[k].X[i + (size_t)i * n]; tz += S[k].Z[i + (size_t)i * n]; }
            nd += n; }
        printf("   SCALEDBG start trX %.3e trZ %.3e ndim %.0f normb %.3e normC %.3e bs %.3e cs %.3e\n", tx, tz, nd, P->normb2, P->normC2, P->bs, P->cs);
    }
    /* 4.22: the predictor-phase Schur solves of AC-OPF-type problems to 1e-10 instead of full
     * accuracy: on M (split-pair weights) the 1e-13 check failed in ~85% of the iterations and
     * PCG took ~19 products with M per iteration (18% of the time on TSSOS case6468); the
     * refinement passes enforce A(dX) = r_p afterwards. TSSOS case6468: 98.7 s against 107.6 s
     * side by side, same 50 iterations, 1.7e-8 against 6.8e-8. Everywhere (all HSD runs) it
     * cost qap8 its 1e-8 and hinf11 39 -> 68 iterations, so it is limited to the class. */
    double soltol = par->hsd_soltol;
    if (soltol == 0 && acopf_cls && !getenv("BRISK_SOLTOL0")) soltol = 1e-10;
    const double soltol_cls = soltol;
    /* 4.23 option: strict solves again near the end (relative gap below BRISK_STRICTGAP or
     * the bordered factorization on) */
    double strict_gap = 0;
    { const char *e = getenv("BRISK_STRICTGAP"); if (e) strict_gap = atof(e); }
    /* 4.23: loose solves (correctors, 1e-7) skip their residual check once the factor has
     * met 1e-3 of that at a first check. TSSOS case6468: 102.4 -> 91.8 s side by side (50 ->
     * 48 its, reduced -> optimal). Everywhere it cost attr_vdp_d12_I 30 -> 41 iterations, so
     * the class only; BRISK_LOOSESKIP=0/1 overrides. */
    { const char *e = getenv("BRISK_LOOSESKIP"); g_looseskip = e ? atoi(e) : acopf_cls; }
    const int looseskip_cls = g_looseskip;
    int looseskip_safe = 0;
    /* free variables given as split LP pairs: handled natively (see FreeVars) */
    FreeVars FV, BV;
    memset(&BV, 0, sizeof(BV));
    int bord = 0, nbord = 0, *bpf = NULL, bord_on = 0, bord_next = 0;
    double *bD = NULL;
    int nf = (par->hsd_free || par->hsd_pshrink > 0) ? free_detect(P, S, &FV) : 0;
    /* split pairs kept as LP variables: remember them for the shrink step */
    int npk = 0, *pk_b = NULL, *pk_p = NULL, *pk_q = NULL;
    if (nf && par->hsd_pshrink > 0) {
        npk = nf; pk_b = malloc(sizeof(int) * nf); pk_p = malloc(sizeof(int) * nf); pk_q = malloc(sizeof(int) * nf);
        memcpy(pk_b, FV.kb, sizeof(int) * nf); memcpy(pk_p, FV.ip, sizeof(int) * nf); memcpy(pk_q, FV.im, sizeof(int) * nf);
    }
    if (nf && !par->hsd_free) { free_release(P, S, &FV); nf = 0; }
    if (!par->hsd_free) memset(&FV, 0, sizeof(FV));
    int fmode = par->hsd_free;
    if (nf && fmode < 0) {
        /* automatic: the saddle factorization (m + nf)^3/3 when M is dense and that costs at
         * most ~2e10 flops more than M itself; otherwise the split pairs are kept (they only
         * hurt when the moment side has no interior, and there nf is small: kernel forms)  */
        /* 4.30: the saddle factorization is a Bunch-Kaufman LDL' (dsytrf), 2.5x the cost of
         * the Cholesky of M at equal size (29 s against 12 s at n = 11 346, one thread). It
         * stays the choice where it was (its robustness matters: the split pairs kept in the
         * standard treatment stall on roa_acrobot_d6_K as read), except on large dense
         * problems with few free variables (nf <= 2% of m and a saddle costing over 1e11
         * flops), where the Schur complement of M is used: Cholesky of M, Y = M^{-1} A_f, an
         * nf x nf factor, refinement passes - dpotrf + 2 m^2 nf. The same when the saddle is
         * over the budget and the free variables are few (they were kept as pairs before). */
        const double N = (double)m + nf, extra = (N * N * N - (double)m * m * m) / 3.0, saddle = 2.5 * N * N * N / 3.0;
        static int sc_env = -1;
        if (sc_env < 0) { const char *e = getenv("BRISK_FREESCHUR"); sc_env = e ? atoi(e) : 1; }
        const int few = sc_env && nf <= 0.02 * m;
        if (!Sc->SC && !Sc->E && few && saddle > 1e11) fmode = 2;
        else if (!Sc->SC && !Sc->E && extra <= 2e10) fmode = 3;
        else if (!Sc->SC && !Sc->E && few) fmode = 2;
        else {
            /* split pairs kept; on the sparse path, bordered (see bord_factor) when all
             * pairs are in one LP block */
            int one = 1;
            for (int f = 1; f < nf; f++) if (FV.kb[f] != FV.kb[0]) one = 0;
            static int bord_env = -1;
            if (bord_env < 0) { const char *e = getenv("BRISK_BORD"); bord_env = e ? atoi(e) : 1; }
            if (Sc->SC && one && bord_env && par->hsd_free < 0) {
                for (int k = 0; k < P->nblk; k++) { free(S[k].fm); S[k].fm = NULL; }
                {   /* the quasi-definite analysis is made when the switch happens (bord_setup) */
                    bord = 1;
                    const Block *LB = &P->blk[FV.kb[0]];
                    bpf = malloc(sizeof(int) * (LB->n + 1));
                    for (int p = 0; p < LB->n; p++) bpf[p] = -1;
                    for (int f = 0; f < nf; f++) { bpf[FV.ip[f]] = f; bpf[FV.im[f]] = f; }
                    bD = calloc(nf + 1, sizeof(double));
                    Sc->bE = calloc(nf + 1, sizeof(double));
                    nbord = nf;
                    BV = FV; memset(&FV, 0, sizeof(FV));
                    if (par->verbose > 1) printf("   HSD: %d split free pairs, bordered Schur factorization available\n", nbord);
                }
            }
            if (!bord) free_release(P, S, &FV);
            nf = 0; fmode = 0;
        }
        if (par->verbose > 1 && !nf) printf("   HSD: split free pairs kept (saddle system too costly)\n");
    }
    double *uf0 = NULL, *uf1 = NULL, *vf = NULL, *ufa = NULL, *rf = NULL, *dxfd = NULL, *qf = NULL;
    if (nf) {
        ndim -= 2.0 * nf;
        uf0 = calloc(nf, sizeof(double)); uf1 = calloc(nf, sizeof(double)); vf = calloc(nf, sizeof(double));
        ufa = calloc(nf, sizeof(double)); rf = calloc(nf, sizeof(double)); dxfd = calloc(nf, sizeof(double));
        qf = calloc(nf, sizeof(double));
        /* the '+' slot takes the free value, the '-' slot and the slacks are zero; the
         * initial dual residual changes accordingly (R0 = C - A'y - Z) */
        for (int f = 0; f < nf; f++) {
            BS *s = &S[FV.kb[f]];
            const int p = FV.ip[f], q = FV.im[f];
            s->X[p] -= s->X[q]; s->X[q] = 0;
            s->R0[p] += s->Z[p]; s->R0[q] += s->Z[q];
            s->Z[p] = 0; s->Z[q] = 0;
        }
        if (fmode == 4 && free_ks_setup(P, Sc, &FV)) fmode = 1;      /* fall back: augmented Lagrangian */
        if (par->verbose) printf("   HSD: %d free variables (split pairs) handled natively%s\n", nf,
                                 fmode == 4 ? " (sparse quasi-definite bordered factorization)" : fmode == 1 ? " (augmented Lagrangian)" :
                                 fmode == 2 ? " (Schur complement of M)" : fmode == 3 ? " (saddle factorization)" : "");
    }
    const double ndim1 = ndim + 1.0, sc = P->bs * P->cs;
    double *rp = malloc(sizeof(double) * m), *rhs = malloc(sizeof(double) * m);
    double *u0 = malloc(sizeof(double) * m), *u1 = malloc(sizeof(double) * m);
    double *v = malloc(sizeof(double) * m), *dy = malloc(sizeof(double) * m);
    double *a0 = malloc(sizeof(double) * m), *hc = malloc(sizeof(double) * m);
    double *hd = malloc(sizeof(double) * m), *qv = malloc(sizeof(double) * m);
    double *dyd = malloc(sizeof(double) * m);
    int ncorr = 0, n_tstep = 0, nrefine = 0, ntry_c = 0, hoc_nfail = 0, hoc_nkept = 0;
    double tc_avg = 0, tc_sum = 0, t_iter_prev = 0, t_iter_start = wtime(), mu_at_best = INFINITY, mu_best_it = INFINITY;
    double score_prog = INFINITY;
    double score_last = INFINITY;     /* (test BRISK_EPOLFAST) the previous iteration's score */
    static int epolfast = -1; if (epolfast < 0) { const char *e = getenv("BRISK_EPOLFAST"); epolfast = e ? atoi(e) : 0; }
    double *shist = malloc(sizeof(double) * 4096), *phist = calloc(4096, sizeof(double));
    double *ghist = calloc(4096, sizeof(double));   /* relative gap per iteration (breakdown rule) */
    double *ihist = calloc(4096, sizeof(double)), *mhist = calloc(4096, sizeof(double));
    int it_sprog = 0;   /* last iteration of score progress (caps the steady-progress extension) */
    const int steady_on = !getenv("BRISK_NOSTEADY");
    /* 5.16: the path of the embedding. With the residual target of this method the homogeneous
     * residuals are theta R0 in exact arithmetic (theta = prod (1 - alpha (1 - sigma))). Once a
     * residual exceeds twice its path value, the part a step got wrong is larger than the part
     * the method controls, and later steps only scale it with theta: progress of the current
     * iterate no longer keeps the run going (a better iterate still does). BRISK_PATHRULE=0: off. */
    static int path_on = -1;
    if (path_on < 0) { const char *e = getenv("BRISK_PATHRULE"); path_on = e ? atoi(e) != 0 : 1; }
    double pth = 1.0, pth_best = 1.0, prp0 = 0, prd0 = 0;
    int offpath = 0;
    static int shift_env = -1;
    if (shift_env < 0) { const char *e = getenv("BRISK_HSDSHIFT"); shift_env = e ? atoi(e) : 1; }
    int shift_next = 0;                 /* the formulation chosen for the next iteration (1: shifted) */
    int shift_off = 0, shift_prev = 0;  /* the shifted form spoiled the primal rows once: not again in this run */
    static double inex_beta = -1; if (inex_beta < 0) { const char *e = getenv("BRISK_HSDINEX"); inex_beta = e ? atof(e) : 0.0; }
    int inex_have = 0, n_inex = 0, n_proj = 0, n_miss = 0, n_ray = 0, n_gal = 0; double mu0_ray = 0;
    static int ray_on = -1; if (ray_on < 0) { const char *re = getenv("BRISK_HSDRAY"); ray_on = re ? atoi(re) : 0; }
    /* experiment (BRISK_HSDYTM=1): dtau from the normalization row of the Ye-Todd-Mizuno embedding instead of the gap
     * row. YTM's extra row -b~'y + <c~, X> - z~ tau = -(n+1) (b~, c~, z~ the residual data at the start) pins the scale
     * of the iterate; its derivative along the direction, <Rd0, dX> - rp0'dy - rg0 dtau = 0, gives dtau as the step's
     * component along the initial residuals. Applied here as a side condition on the present direction: dy, dX, dZ
     * are affine in dtau through v and the tau column, so one scalar correction delta moves the direction to the row */
    static int ytm_on = -1; if (ytm_on < 0) { const char *ye = getenv("BRISK_HSDYTM"); ytm_on = ye ? atoi(ye) : 0; }
    double *rp0 = ytm_on ? malloc(sizeof(double) * (m + 1)) : NULL; double rg0 = 0; int n_ytm = 0; double inex_R2 = 0, inex_RE = 0, inex_E2 = 0, inex_sig = 0;
    if ((getenv("BRISK_HSDPROJ") || getenv("BRISK_HSDMISSSTOP") || getenv("BRISK_HSDGAL")) && inex_beta <= 0) inex_beta = 1e300;   /* the projection needs the miss measured (the cap never binds at beta = 1e300) */
    double rpo_it = 0, rpo_pred = 0;    /* this iteration's primal residual, and the next one's predicted value */
    double hcC0 = 0;                    /* <C, H C> of this iteration (the size of the cancelling terms) */
    HsdShift shc = { 0, 1.0, 0.0, NULL, NULL, NULL, 1.0 };
    shc.F = calloc(nb, sizeof(double *));
    double *bmh = malloc(sizeof(double) * (2 * (size_t)m + 1));
    shc.bmh = bmh;
    { const char *e = getenv("BRISK_PFLOOR"); g_pfloor = e ? atoi(e) : 1; }
    { const char *e = getenv("BRISK_BREAKDOWN"); g_brkdn = e ? atoi(e) : 1; }
    int kfix = 0, kf_n = 0, tiny_steps = 0, small_steps = 0;
    static int epolish_on = -1; if (epolish_on < 0) { const char *e = getenv("BRISK_EPOLISH"); epolish_on = e ? atoi(e) : 1; }
    int n_epol = 0;
    int n_xprobe = 0, n_rprobe = 0; double rprobe_last = INFINITY, pinf_lo = INFINITY;
    static int rprobe_on = -1; if (rprobe_on < 0) rprobe_on = getenv("BRISK_NORPROBE") == NULL;
    static int xprobe_on = -1; if (xprobe_on < 0) xprobe_on = getenv("BRISK_NOXPROBE") == NULL;
    static double xprobe_gap = -1; if (xprobe_gap < 0) xprobe_gap = getenv("BRISK_XPROBEGAP") ? atof(getenv("BRISK_XPROBEGAP")) : 0.5;
    EndRules ER; endrules_init(&ER, dd_small_problem(P, par), (par->hsd_first_on && !par->no_retry) ? par->retry_hsd1 : 0.0);
    if (par->tol < 1e-8 && !getenv("BRISK_NOFLOOR")) ER.floor_at = fmin(1e-8, 100.0 * par->tol);
    double epol_last = INFINITY;      /* max(pinf, gap, dinf) at the last failed probe: the next needs half of it */
    /* 4.29: with a tolerance below the default, the probe keeps its gates at 1e-8 and a
     * candidate that passes 1e-8 but not the tolerance is kept as a fallback: the run goes
     * on, and returns the candidate if no later point beats it (-acc high must never end
     * worse than the default run would have)                                             */
    /* ptol: the tolerance for the rules that stop a run short of its target (the probe gates,
     * the crawl and floor-stall rules): at most the default 1e-8, so that above 1e-8 a
     * high-accuracy run stops exactly where the default run does (inc_1200 crawled from
     * 2e-7 into the 600 s limit, swissroll 64 -> 130 s, inc_600 37 -> 72 s, no gain) */
    const double ptol = fmax(par->tol, 1e-8);
    double fb_score = INFINITY; Result fb_R; memset(&fb_R, 0, sizeof(fb_R));
    int fb_drift = 0;   /* 20 October 2026: the fallback is the probe point the drift guard held back (returned only when the run then ends without a tolerance stop) */
    /* Objective drift (on by default since 21 October 2026; BRISK_DRIFT=0 off). On a degenerate problem whose optimal face is
     * thin (moment relaxations with free off-diagonal entries: ni35 order 2 with the full
     * objective cut), the iterates reach residuals of 1e-11 and a duality gap of 1e-9 while
     * <C,X> and b'y still fall together by 2e-5 relative per iteration: every tolerance stop
     * passed at iteration 31 with the value 4.6 (of a 23 gap) short of the optimum the
     * solve reached ten iterations later. A tolerance stop is therefore held back while
     * the objectives move by more than max(100 tol, 10 x the previous gap) in one
     * iteration: ordinary convergence moves them by about the previous gap. */
    int drift_on; { const char *e = getenv("BRISK_DRIFT"); drift_on = e ? atoi(e) : 1; }   /* (read per solve, as BRISK_BREAKDOWN; off in
     * 1.5.4 (its cost untested); on since 21 October 2026, the owner's decision, in the form without the double-double endgame
     * after a held-back stop: on SDPLIB, a user's frames and the occupation measures the values reported OPTIMAL 4.7e-7 to
     * 1.1e-5 off without it are within 1e-8 with it on the 13 files checked against -prec dd, at +1-3 % of the time;
     * AC-OPF, the control LMIs and option pricing unchanged (REPORT_5.21 §4, §6). BRISK_DRIFT=0 off) */
    double po_prev = NAN, dob_prev = NAN, gap_prev = INFINITY;
    int drift_seen = 0, drift_first = -1;
    g_drift_held = 0;
    /* (19 October 2026, the review) the hold-back lasts at most drift_max iterations from its first trigger (the cases it
     * was made for needed 8 and 10 more): a drift that persists (an unattained optimum, tau -> 0) must not keep the run
     * from its stops until the iteration limit or the infeasibility test; then the guard is off for the run */
    const int drift_max = 12;
    /* The objective candidate (BRISK_OBJBEST=0 off): the iterate with the best primal
     * objective among those with primal and dual residuals within 10 tol. The best
     * iterate is the one with the smallest DIMACS score, and on such a problem that can be
     * an early one: ni35 order 2 with the completed objective cut ended at a primal
     * breakdown after iteration 40 (residuals 2e-8) and returned iteration 20, whose gap
     * was smaller and whose objective 7 worse. A run that does not end at the tolerance
     * returns the candidate instead when it is better by more than max(100 tol, 10 x the
     * best iterate's gap). */
    int objbest_on; { const char *e = getenv("BRISK_OBJBEST"); objbest_on = e ? atoi(e) : 1; }
    double **ocX = NULL, **ocZ = NULL, *ocy = NULL, oc_po = INFINITY;
    int oc_it = -1, oc_used = 0;
    g_objcand = 0;
    Result oc_R; memset(&oc_R, 0, sizeof(oc_R));
    double **fbX = NULL, **fbZ = NULL, *fby = NULL;
    static double epol_pgate = -1; if (epol_pgate < 0) { const char *e = getenv("BRISK_EPOLPGATE"); epol_pgate = e ? atof(e) : 100.0; }
    static int refskip_on = -1; if (refskip_on < 0) { const char *e = getenv("BRISK_REFSKIP"); refskip_on = e ? atoi(e) : 1; }
    double ref_floor = 0;             /* absolute A(dX) miss the last correction could not resolve */
    int nref_skip = 0, nref_fail = 0;
    static int crawl_on = -1; if (crawl_on < 0) { const char *e = getenv("BRISK_CRAWL"); crawl_on = e ? atoi(e) : 1; }
    double kap_best = 1.0, theta_best = 1.0;
    int rescued = 0;
    if (g_prek < 0) { const char *e = getenv("BRISK_PREK"); g_prek = e ? atoi(e) : 2; }
    double kf_fact = 0;
    int it_prog = 0;
    /* NT inside the embedding (-hsddir 1): s->Zi holds the scaling W (W Z W = X), and the
     * products Z^{-1} F X become W F W; the true inverse Z^{-1} (needed for A(Z^{-1}) and
     * the sigma mu Z^{-1} term) is kept in Zv. LP blocks are identical under HKM and NT and
     * keep Zi = 1/z. */
    const int nt = par->direction == 1;
    double **Zv = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++)
        Zv[k] = (nt && P->blk[k].type == BLK_SDP) ? amalloc(bsz(&P->blk[k]) * sizeof(double)) : NULL;
    /* endpoint directions of the sigma family (the Newton direction is affine in sigma) */
    double **dX0 = malloc(sizeof(double *) * nb), **dZ0 = malloc(sizeof(double *) * nb);
    double **dX1 = malloc(sizeof(double *) * nb), **dZ1 = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) {
        size_t len = bsz(&P->blk[k]) * sizeof(double);
        dX0[k] = amalloc(len); dZ0[k] = amalloc(len); dX1[k] = amalloc(len); dZ1[k] = amalloc(len);
    }
    /* Pattern-restricted products. The dual residual of the embedding contracts exactly,
     * Rd = theta R0 (R0 the residual at entry, sparse for the standard starting point),
     * so every middle matrix (Rd, C, the dZ's) lies on the pattern of C, R0 and the
     * constraints, and the six products per iteration use the standard method's
     * pattern kernel instead of two dense n^3 products each. The true residual is
     * checked every iteration; if it drifts from theta R0, R0 is re-based.           */
    Pat *hp = calloc(nb, sizeof(Pat));
    char *pk = calloc(nb, 1);
    double theta = 1.0, r0norm = 0;
    int nrebase = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_SDP || B->n <= PROD_SMALL || !par->hsd_pat) continue;
        setup_hsd_pattern(B, &S[k], par, &hp[k]);
        pk[k] = hp[k].route != 2;
    }
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        size_t len = bsz(B);
        memset(S[k].dX, 0, sizeof(double) * len);
        r0norm += norm2_plus_R0(&S[k], len, S[k].dX, 1.0);
    }
    r0norm = sqrt(r0norm);
    int use_pat = 0;
    for (int k = 0; k < nb; k++) use_pat |= pk[k];
    if (par->verbose > 1 && use_pat) printf("   HSD: pattern-restricted products on %s blocks\n", nb > 1 ? "some" : "the");
    /* balanced start: the (tau, kappa) pair on the same central path as the cone part */
    {
        double xz0 = 0;
        for (int k = 0; k < nb; k++) xz0 += ddot_n(bsz(&P->blk[k]), S[k].X, S[k].Z);
        kap = xz0 / ndim;
        if (!(kap > 0)) kap = 1.0;
    }
    int status = 3, it, stall = 0, nsafe = 0;
    double sigma = 0, alpha = 0;
    Sc->use_hint = par->hsd_reghint;
    const double pivtol_saved = Sc->pivtol;
    if (par->hsd_pivtol > 0) Sc->pivtol = par->hsd_pivtol;

    const double t_loop0 = wtime();
    /* 4.30: the trace-bound row (sum tr X + s = R tau) and its slack; R grows online when
     * the bound binds (its dual w far from zero once the gap is small): s += dR tau keeps
     * the row satisfied, so nothing restarts. A bound that never binds leaves X at a
     * moderate scale, which is the point: the Gram matrices of kernel-form SOS problems
     * otherwise run off along a recession direction (trX/tau 1e5-1e7, tau -> 0). */
    int tb_row = -1, tb_blk = -1, tb_idx = -1, tb_grow = 0;
    if (P->tbR != 0)
        for (int i = 0; i < m; i++) if (P->b0[i] == P->tbR) { tb_row = i; break; }
    if (tb_row >= 0) {
        for (int k = 0; k < nb && tb_blk < 0; k++) {
            const Block *B = &P->blk[k];
            if (B->type != BLK_LP) continue;
            for (int t = 0; t < B->ncon; t++) if (B->con[t] == tb_row && B->A[t].nnz == 1) { tb_blk = k; tb_idx = B->A[t].row[0]; break; }
        }
        if (tb_blk < 0) tb_row = -1;
    }
    for (it = 0; it <= par->maxit; it++) {
        if (it > 0) brisk_threads_adapt(par->verbose);          /* 5.11: the threads of this iteration = the free cores */
        if (tb_row >= 0 && it > 0 && tb_grow < 8 && R->relgap < 1e-3 && getenv("BRISK_TBGROW")) {
            /* the bound's price in original units: w = cs d y; binding if w R is a
             * noticeable fraction of the objective */
            const double w = fabs(P->cs * P->d[tb_row] * y[tb_row]) * P->tbR;
            if (w > 1e-4 * (1.0 + fabs(R->pobj) + fabs(R->dobj))) {
                const double f = 100.0, db = (f - 1.0) * P->b[tb_row];
                S[tb_blk].X[tb_idx] += db * tau / P->d[tb_row];
                P->b[tb_row] *= f; P->b0[tb_row] *= f; P->tbR *= f;
                tb_grow++;
                if (par->verbose > 1) printf("   HSD: trace bound binding (w R = %.1e), R -> %.3g\n", w, P->tbR);
            }
        }
        double td = wtime();
        if (it > 0) t_iter_prev = td - t_iter_start;
        t_iter_start = td;
        int zerr = 0;
        HBLK_FOR(k, nb, { /* Z^{-1} */
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n, info;
            if (B->type == BLK_LP) {
                for (int i = 0; i < n; i++) s->Zi[i] = 1.0 / s->Z[i];
                if (s->fm) for (int i = 0; i < n; i++) if (s->fm[i]) s->Zi[i] = 0;
            }
            else if (B->type == BLK_SOC) {                         /* 5.16: z^{-1} and the NT scaling */
                if (soc_nt(s, n, s->X, s->Z)) {
                    _Pragma("omp atomic write")
                    zerr = 1;
                } else soc_inv(n, s->Z, s->Zi);
            }
            else if (s->zon && !Zv[k]) {                           /* 5.12: Z^{-1} from the sparse factor */
                if (getenv("BRISK_ZINVOLD")) {
                    pzero(bsz(B), s->Zi);
                    for (int i = 0; i < n; i++) s->Zi[i + (size_t)i * n] = 1.0;
                    schol_solve_rows(s->zf, s->Zi, n, n);
                    symmetrize(n, s->Zi);
                } else schol_inverse(s->zf, s->Zi, n);
            }
            else {
                double *Zinv = Zv[k] ? Zv[k] : s->Zi;
                memcpy(Zinv, s->LZ, sizeof(double) * bsz(B));
                if (n <= (g_lockfree ? LOCKFREE_N : PROD_SMALL)) { inv_from_chol_small(n, Zinv); lower_to_full(n, Zinv); }
                else {
                    info = brisk_dpotri_full(n, Zinv, n);       /* 5.12: in panels on the threads from n = 768 */
                    if (info) {
                        _Pragma("omp atomic write")
                        zerr = 1;
                        continue;
                    }
                }
                if (Zv[k] && B->type == BLK_SDP && s->ntGbuf && it > 0) {
                    /* the scaling of the Schur matrix still factored (the end's correction, probe_polish) */
                    if (!s->ntGold) s->ntGold = amalloc(sizeof(double) * bsz(B));
                    memcpy(s->ntGold, s->ntGbuf, sizeof(double) * bsz(B));
                }
                if (Zv[k] && nt_scaling(B, s)) {
                    _Pragma("omp atomic write")
                    zerr = 1;
                }
            }
        });
        if (zerr) { status = 4; g_hsd_why = "Z of the start is not positive definite"; goto done; }
        g_ph_last = wtime();
        /* residuals of the embedding (tau = 1 by normalization below) */
        for (int i = 0; i < m; i++) rp[i] = tau * P->b[i];
        double cX = 0, xz = 0, rd2 = 0, nAty = 0, dev2 = 0;
        {
            /* 4.24: per-block partials summed in block order (any thread count) */
            double *pr = hred(nb, 5);
            aop_begin(rp, NULL, NULL, NULL);
            HBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                double *r5 = pr + 5 * (size_t)k;
                blk_Aop(B, s, s->X, rp, -1.0);
                blk_ATy(B, s, y, s->F);
                r5[0] += add_norm2(len, s->F, s->Z);                      /* 5.12: on the threads for large blocks */
                PAR_LEN_FOR(len, s->F[i] = -s->F[i];);
                sp_add(B, &B->C, tau, s->F);                               /* F = Rd */
                r5[1] = ddot_n(len, s->F, s->F);
                if (use_pat) r5[2] = norm2_plus_R0(s, len, s->F, -theta);
                r5[3] = sp_inner(B, &B->C, s->X);
                r5[4] = ddot_n(len, s->X, s->Z);
            });
            aop_end();
            for (int k = 0; k < nb; k++) { const double *r5 = pr + 5 * (size_t)k; nAty += r5[0]; rd2 += r5[1]; dev2 += r5[2]; cX += r5[3]; xz += r5[4]; }
        }
        double bty = ddot_n(m, P->b, y), rg = kap + cX - bty;
        if (rp0 && it == 0) { memcpy(rp0, rp, sizeof(double) * m); rg0 = rg; }
        double cfx = 0;                            /* c_f' x_f: cX without it is the cone part */
        for (int f = 0; f < nf; f++) {
            rf[f] = S[FV.kb[f]].F[FV.ip[f]];       /* tau c_f - Af' y */
            FV.xf[f] = S[FV.kb[f]].X[FV.ip[f]];
            cfx += FV.c[f] * FV.xf[f];
        }
        double mu = (xz + tau * kap) / ndim1;
        if (it == 0) mu0_ray = mu;
        if (ENV_ON("BRISK_SCALEDBG")) {
            double tx = 0, tz = 0;
            for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n;
                if (B->type == BLK_LP) for (int i = 0; i < n; i++) { tx += S[k].X[i]; tz += S[k].Z[i]; }
                else if (B->type == BLK_SOC) { tx += S[k].X[0]; tz += S[k].Z[0]; }
                else for (int i = 0; i < n; i++) { tx += S[k].X[i + (size_t)i * n]; tz += S[k].Z[i + (size_t)i * n]; } }
            printf("   SCALEDBG it %d trX/tau %.3e trZ/tau %.3e ratio %.3e tau %.3e\n", it, tx / tau, tz / tau, tz / fmax(tx, 1e-300), tau);
        }
        double rpo = 0;
        for (int i = 0; i < m; i++) { double w = rp[i] * P->du[i]; rpo += w * w; }
        rpo = sqrt(rpo);
        double rdo = P->cs * sqrt(rd2);
        double po = sc * cX / tau, dob = sc * bty / tau, den = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        R->pobj = po; R->dobj = dob;
        R->pinf = rpo / (tau * (1 + P->normb2));
        R->dinf = rdo / (tau * (1 + P->normC2));
        R->relgap = fabs(po - dob) / den;
        R->relcomp = fabs(sc * xz / (tau * tau)) / den;
        R->err[1] = rpo / (tau * (1 + P->normb1));
        R->err[3] = rdo / (tau * (1 + P->normC1));
        R->err[5] = (po - dob) / den;
        R->err[6] = sc * xz / (tau * tau) / den;
        R->err[2] = R->err[4] = 0;        /* (an iterate is inside the cone; a candidate's values must not stay behind) */
        /* a shifted step whose primal residual came out above twice its prediction (1 - alpha (1 - sigma)) rp: the shifted
         * form's own miss in the primal rows (A(dX) through C^ = (Z + Rd) / tau) dominates there; the original form from now
         * on (option pricing: rp 5.7e-11 -> 3.6e-6 at one shifted step) */
        static int shift_kill = -1; if (shift_kill < 0) { const char *e = getenv("BRISK_SHIFTKILL"); shift_kill = e ? atoi(e) : 1; }
        if (shift_kill && it > 0 && shift_prev && !shift_off && rpo > 2.0 * rpo_pred) {
            shift_off = 1;
            if (ENV_ON("BRISK_SHIFTDBG")) printf("     [tau column: the shifted step left rp at %.2e, %.1f times its prediction: original form from here]\n", rpo, rpo / fmax(rpo_pred, 1e-300));
        }
        rpo_it = rpo;
        if (it == 0) { prp0 = rpo; prd0 = rdo; }
        else if (!offpath && (rpo > 2.0 * pth * prp0 || rdo > 2.0 * pth * prd0)) {
            offpath = 1;
            if (par->verbose > 1) printf("   HSD: the residuals left the path at iteration %d (primal %.1f, dual %.1f times theta R0)\n",
                                         it, rpo / fmax(pth * prp0, 1e-300), rdo / fmax(pth * prd0, 1e-300));
        }
        if (ENV_ON("BRISK_BITSDBG")) hsd_bits(P, S, y, tau, kap, it);
        if (ENV_ON("BRISK_PATHDBG")) printf("     [path it %d theta %.2e  rp/(theta rp0) %.2e  rd/(theta rd0) %.2e  tau %.2e]\n", it, pth,
                                          rpo / fmax(pth * prp0, 1e-300), rdo / fmax(pth * prd0, 1e-300), tau);
        const int onpath = !(path_on && offpath);
        R->t_dense += wtime() - td;
        if (strict_gap > 0 && !looseskip_safe) {
            const int strict = bord_on || R->relgap < strict_gap;
            soltol = strict ? par->hsd_soltol : soltol_cls;
            g_looseskip = strict ? 0 : looseskip_cls;
        }
        double score = res_score(R);
        double drift = INFINITY;
        if (it > 0 && isfinite(po_prev))
            drift = fmax(fabs(po - po_prev), fabs(dob - dob_prev)) / den;
        if (drift_on && drift_first >= 0 && it - drift_first > drift_max) {
            if (par->verbose > 1) printf("   HSD: the objective still moves %d iterations after the tolerance was first met: the drift guard is off for the rest of the run\n", it - drift_first);
            drift_on = 0; drift_seen = 0;
        }
        const int moving = drift_on && it > 0 && drift > fmax(100.0 * par->tol, 10.0 * gap_prev);
        po_prev = po; dob_prev = dob; gap_prev = R->relgap;
        if (par->verbose)
            printf("%3d %+.8e %+.8e %.2e %.2e %.2e %.2e %.3f %.3f %.3f %7.2f  kap/mu %.1e%s\n",
                   it, po, dob, R->pinf, R->dinf, R->relgap, mu, alpha, alpha, sigma,
                   wtime() - par->t_start, tau * kap / mu, "");
        dump_xz(P, S, y, it, tau);
        if (ENV_ON("BRISK_SCOREDBG")) printf("   SCORE it %d score %.3e comp %.3e best %.3e itbest %d t %.4f\n", it, score, R->relcomp, fmin(score, *best_score), score < *best_score ? it : *it_best, wtime() - par->t_start);
        if (ENV_ON("BRISK_HSDDBG")) {
            double trX = 0, trZ = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) { trX += S[k].X[i]; trZ += S[k].Z[i]; }
                else if (B->type == BLK_SOC) { trX += S[k].X[0]; trZ += S[k].Z[0]; }
                else for (int i = 0; i < B->n; i++) { trX += S[k].X[i + (size_t)i * B->n]; trZ += S[k].Z[i + (size_t)i * B->n]; }
            }
            printf("     [tau %.3e kap %.3e  trX/tau %.3e trZ/tau %.3e |y|/tau %.3e relcomp %.2e score %.2e]\n", tau, kap, trX / tau, trZ / tau,
                   sqrt(ddot_n(m, y, y)) / tau, R->relcomp, score);
        }
        btrack_offer(par, P, S, y, tau, mu, R, it);
        rcand_offer(rc, P, S, y, tau, R, it);
        const int improved_er = score < *best_score;
        /* (19 October 2026, the review: the candidate's gap within 10 tol as well, so that it cannot be better in the
         * objective through a gap; its copies only when the memory cap leaves room for them, allocated with a check) */
        if (objbest_on && keep_best && g_oc_fits && R->pinf <= 10.0 * ptol && R->dinf <= 10.0 * ptol && fabs(R->relgap) <= 10.0 * ptol && po < oc_po) {
            if (!ocX) {
                int okm = (ocX = calloc(nb, sizeof(double *))) && (ocZ = calloc(nb, sizeof(double *))) && (ocy = malloc(sizeof(double) * (m + 1)));
                for (int k = 0; k < nb && okm; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    okm = (ocX[k] = malloc(sizeof(double) * (len + 1))) && (ocZ[k] = malloc(sizeof(double) * (len + 1)));
                }
                if (!okm) {
                    if (ocX) for (int k = 0; k < nb; k++) free(ocX[k]);
                    if (ocZ) for (int k = 0; k < nb; k++) free(ocZ[k]);
                    free(ocX); free(ocZ); free(ocy); ocX = ocZ = NULL; ocy = NULL; objbest_on = 0;
                }
            }
        }
        if (ocX && objbest_on && keep_best && R->pinf <= 10.0 * ptol && R->dinf <= 10.0 * ptol && fabs(R->relgap) <= 10.0 * ptol && po < oc_po) {
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]);
                double *ox = ocX[k], *oz = ocZ[k]; const double *xk = S[k].X, *zk = S[k].Z;
                PAR_LEN_FOR(len, ox[i] = xk[i] / tau; oz[i] = zk[i] / tau;);
            }
            for (int i = 0; i < m; i++) ocy[i] = y[i] / tau;
            oc_po = po; oc_it = it; oc_R = *R;
        }
        if (score < *best_score) {
            *best_score = score; *it_best = it; *best = *R;
            mu_best_it = mu;
            kap_best = kap / tau; theta_best = theta / tau; pth_best = pth / tau;
            for (int i = 0; i < m; i++) ybest[i] = y[i] / tau;
            if (keep_best)
                for (int k = 0; k < nb; k++) {
                    size_t len = bsz(&P->blk[k]);
                    PAR_LEN_FOR(len, S[k].Xb[i] = S[k].X[i] / tau; S[k].Zb[i] = S[k].Z[i] / tau;);
                }
        }
        if (score < par->tol && moving) {
            if (!drift_seen && par->verbose > 1) printf("   HSD: the tolerance is met but the objective still moves (%.1e per iteration against a gap of %.1e), continuing\n", drift, R->relgap);
            drift_seen = 1; if (drift_first < 0) drift_first = it;
        }
        if (score < par->tol && !moving) { status = 0; break; }
        /* 5.12: the best iterate already meets the tolerance in the DIMACS measures, with a
         * factor 2 to spare, and this iteration did not improve on it: stop and return it.
         * res_score divides by 1 + ||b||_2 and 1 + ||C||_F, the status by 1 + ||b||_1 and
         * 1 + ||C||_1; at a floor of the scaled residual the iterations that follow only
         * try to close that difference (thetaG51, b of 6910 ones: primal residual 2.8e-8
         * here, 3.4e-10 as read; six iterations and two failed polishes, 30 of 68 s).
         * BRISK_DSTOP=0 off. */
        if (dstop_on() && !moving && !drift_seen && keep_best && it > *it_best && dimacs_score(best) <= 0.5 * par->tol) {
            if (par->verbose > 1) printf("   HSD: the best iterate (%d) meets the tolerance in the DIMACS measures (%.1e), stopping\n", *it_best, dimacs_score(best));
            g_dstop = 1; status = 4; g_hsd_why = "the best iterate meets the tolerance in the DIMACS measures"; break;
        }
        /* 4.27: the complementarity lags alone. <X,Z>/tau^2 = gap + (y'r_p - <X,R_d>)/tau^2,
         * and the dual residual is exact by construction (dZ from dy) while the primal one
         * has the floor of the Schur solves, so where |y| is large (the moments of the image
         * forms, |y|/tau ~ 1e3) relcomp ~ |y| pinf stalls 100-1000x above the other measures
         * and the run went on for 1-7 iterations until a stall rule stopped it and the final
         * Gram polish (which makes err6 = err5) fixed it. The polish is tried as soon as gap,
         * dinf and pinf allow it, and the run stops only if its verified score meets tol.
         * BRISK_EPOLISH=0 turns it off. */
        /* (test) fast local convergence: the score fell tenfold in this iteration and is within ten
         * times the tolerance, so the next iteration is expected to meet it: no probes now */
        const int fast_conv = epolfast && score_last >= 10.0 * score && score <= 10.0 * par->tol;
        score_last = score;
        if (epolish_on && !fast_conv && keep_best && !nf && !bord && !g_has_soc && n_epol < 4 && score > par->tol && R->relcomp > par->tol && !drift_seen &&
            fmax(R->relgap, R->dinf) <= ptol && R->pinf <= epol_pgate * ptol &&
            fmax(R->pinf, fmax(R->relgap, R->dinf)) <= 0.5 * epol_last) {
            Result Rp = *R;
            double rres = 0;
            const double sp = probe_polish(P, S, tau, y, Sc, a0, dy, &Rp, &rres, par->tol);
            n_epol++;
            epol_last = fmax(R->pinf, fmax(R->relgap, R->dinf));
            if (sp <= par->tol && moving) {
                if (par->verbose > 1 && !drift_seen) printf("   HSD: the probe meets the tolerance but the objective still moves (%.1e per iteration against a gap of %.1e), continuing\n", drift, R->relgap);
                drift_seen = 1; if (drift_first < 0) drift_first = it;
            }
            if (par->verbose > 1) printf("   HSD: polish probe at iteration %d: score %.2e -> %.2e (Gram CG residual %.1e; pinf %.1e dinf %.1e gap %.1e comp %.1e err2 %.1e err4 %.1e)%s\n",
                                         it, score, sp, rres, Rp.pinf, Rp.dinf, Rp.relgap, Rp.relcomp, Rp.err[2], Rp.err[4], sp <= par->tol && !moving && !drift_seen ? ", stopping" : "");
            if (sp <= par->tol && !moving && !drift_seen) {
                /* the current iterate becomes the best one; brisk_solve returns the probe's
                 * candidate (s->G, s->Q, y/tau) with its measures */
                g_epol_R = Rp;
                *best_score = score; *it_best = it; *best = *R;
                memcpy(ybest, a0, sizeof(double) * m);                     /* y/tau of the candidate */
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    PAR_LEN_FOR(len, S[k].Xb[i] = S[k].X[i] / tau; S[k].Zb[i] = S[k].Z[i] / tau;);
                }
                g_epol_stop = 1;
                status = 0; break;
            }
            /* 5.4: a verified candidate within red_acc is kept too (a cantilever relaxation sent by a user:
             * the probe reached 3.7e-8 at iteration 19 and was dropped for being above 1e-8; the run
             * ended at 1.4e-6 after a re-solve, 137 s instead of 40 s) */
            /* 20 October 2026: the point the drift guard holds back (it meets the tolerance while the objective moves)
             * is kept too: when the continuation ends without meeting the tolerance (the residuals leave the path, a
             * breakdown at the floor: qap9, control7, control8), it is returned instead of the endgame on a worse
             * iterate; when the continuation stops at the tolerance, its point is returned. Off (BRISK_DRIFTFB=1 on): the
             * continuation's value is the more accurate one on those files (qap9 4.9e-6 -> 4.2e-8 from the double-double
             * reference), so the fallback trades that accuracy for the time; the owner's decision */
            static int drift_fb = -1; if (drift_fb < 0) { const char *e = getenv("BRISK_DRIFTFB"); drift_fb = e ? atoi(e) != 0 : 0; }   /* (off: the owner decides; BRISK_DRIFTFB=1 on) */
            const int held = drift_fb && sp <= par->tol && moving && !fb_drift;
            if (held || (sp <= fmax(ptol, par->red_acc) && sp < fb_score && sp < *best_score && !moving && !drift_seen)) {   /* (not a point of the drift, but the held one) */
                if (!fbX) {
                    fbX = calloc(nb, sizeof(double *)); fbZ = calloc(nb, sizeof(double *));
                    fby = malloc(sizeof(double) * (m + 1));
                    for (int k = 0; k < nb; k++) {
                        const size_t len = bsz(&P->blk[k]);
                        fbX[k] = malloc(sizeof(double) * (len + 1)); fbZ[k] = malloc(sizeof(double) * (len + 1));
                    }
                }
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    memcpy(fbX[k], S[k].G, sizeof(double) * len); memcpy(fbZ[k], S[k].Q, sizeof(double) * len);
                }
                memcpy(fby, a0, sizeof(double) * m);
                fb_score = sp; fb_R = Rp; fb_drift = held;
                if (par->verbose > 1) printf("   HSD: probe candidate %.2e kept as the fallback%s\n", sp, held ? " (the point the drift guard holds back)" : "");
            }
            /* 5.5: the Euclidean probe failed although the residuals and the gap are within the
             * tolerance and the score is close: the X-metric correction (the one the end of the solve
             * applies to the best iterate) is tried now, once. On degenerate moment relaxations
             * (Motzkin's form on the sphere, order 20) the run otherwise went on for four iterations
             * to the primal residual floor and then returned this iterate, polished, anyway. */
            if (xprobe_on && !n_xprobe && sp > par->tol && improved_er && par->polish_x && score <= 10.0 * par->tol
                && fmax(R->pinf, fmax(R->relgap, R->dinf)) <= par->tol
                && R->relgap <= xprobe_gap * par->tol        /* (the correction brings the complementarity down to the gap, not below it) */
                && fabs(Rp.err[4]) <= par->tol && fabs(Rp.err[3]) <= par->tol) {   /* (Z = C - A'y at this y passes: only X needs the correction, and the X-metric one does not touch y) */
                n_xprobe++;
                double **Xb2 = malloc(sizeof(double *) * nb), **Zb2 = malloc(sizeof(double *) * nb), **Wsave = malloc(sizeof(double *) * nb);
                for (int k = 0; k < nb; k++) { Xb2[k] = S[k].Xb; Zb2[k] = S[k].Zb; const size_t len = bsz(&P->blk[k]); Wsave[k] = malloc(sizeof(double) * (len + 1)); memcpy(Wsave[k], S[k].W, sizeof(double) * len); }
                Result Rx = *R;
                const int okx = polish_xmetric(P, S, Xb2, Zb2, ybest, &Rx, score, Sc, dy, par->verbose, 0);
                for (int k = 0; k < nb; k++) { memcpy(S[k].W, Wsave[k], sizeof(double) * bsz(&P->blk[k])); free(Wsave[k]); }
                free(Xb2); free(Zb2); free(Wsave);
                if (okx && !moving && !drift_seen && fmax(res_score(&Rx), fmax(Rx.err[2], Rx.err[4])) <= par->tol) {
                    if (par->verbose > 1) printf("   HSD: the X-metric correction of iteration %d meets the tolerance, stopping\n", it);
                    g_epol_R = Rx;
                    *best_score = score; *it_best = it; *best = *R;
                    g_epol_stop = 1;
                    status = 0; break;
                }
            }
            /* F = tau C - A'y - Z again (gram_cg used it) */
            HBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                const size_t len = bsz(B);
                blk_ATy(B, s, y, s->F);
                PAR_LEN_FOR(len, s->F[i] += s->Z[i];);
                PAR_LEN_FOR(len, s->F[i] = -s->F[i];);
                sp_add(B, &B->C, tau, s->F);
            });
        }
        /* 5.10: the restoration probe. The dual residual and the complementarity are within
         * half the tolerance and only the primal residual (and the gap with it) is not: the
         * primal residual is above four times its lowest value so far - the sign that the
         * direction no longer satisfies the primal equations, which the iterations that follow
         * cannot repair and the restoration of the end of the solve can. (Without that sign
         * the run is converging and the next iteration ends it: thetaG51, where X is of low
         * rank and the restoration fails, paid 7 of 64 s for a probe at a primal residual
         * that was still falling.) The restoration is applied to
         * this iterate now (it is the one just stored for it) and the run stops if the
         * restored point passes; at most twice, the second time only after the two measures
         * have fallen by four more. A user's relaxation with a block of order 473: stop at
         * iteration 33 with 1e-10 instead of six more iterations at steps below 0.3.
         * BRISK_NORPROBE turns it off. */
        if (rprobe_on && par->polish_r && rc && rc->have && rc->it == it && keep_best && !nf && !bord && !g_has_soc && n_rprobe < 2 && score > par->tol && !drift_seen) {
            const double rs = fmax(R->dinf, fabs(R->relcomp));
            if (rs <= 0.5 * par->tol && rs <= 0.25 * rprobe_last && R->pinf >= 4.0 * pinf_lo) {
                n_rprobe++; rprobe_last = rs;
                double **Xr2 = malloc(sizeof(double *) * nb), **Zr2 = malloc(sizeof(double *) * nb), **Wsave = malloc(sizeof(double *) * nb);
                for (int k = 0; k < nb; k++) { Xr2[k] = S[k].Xr; Zr2[k] = S[k].Zr; const size_t len = bsz(&P->blk[k]); Wsave[k] = malloc(sizeof(double) * (len + 1)); memcpy(Wsave[k], S[k].W, sizeof(double) * len); }
                Result Rx = *R;
                const int okr = polish_xmetric(P, S, Xr2, Zr2, rc->y, &Rx, score, Sc, dy, par->verbose > 1, 1);
                for (int k = 0; k < nb; k++) { memcpy(S[k].W, Wsave[k], sizeof(double) * bsz(&P->blk[k])); free(Wsave[k]); }
                free(Xr2); free(Zr2); free(Wsave);
                if (okr && !moving && !drift_seen && fmax(res_score(&Rx), fmax(Rx.err[2], Rx.err[4])) <= par->tol) {
                    if (par->verbose > 1) printf("   HSD: the restored iterate %d meets the tolerance, stopping\n", it);
                    g_epol_R = Rx;
                    *best_score = score; *it_best = it; *best = *R;
                    memcpy(ybest, rc->y, sizeof(double) * m);
                    for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); memcpy(S[k].Xb, S[k].Xr, len); memcpy(S[k].Zb, S[k].Zr, len); }
                    g_epol_stop = 1;
                    status = 0; break;
                }
            }
        }
        if (R->pinf < pinf_lo) pinf_lo = R->pinf;
        if (time_up(par, it, t_loop0)) { status = ST_TIME; break; }
        double nAX = 0;
        for (int i = 0; i < m; i++) { double w = tau * P->b[i] - rp[i]; nAX += w * w; }
        nAX = sqrt(nAX);
        nAty = sqrt(nAty);
        /* from the fourth iteration. 18-19 October 2026: from the first one was tried (the adversarial set's random
         * infeasible SDPs 4 -> 1-3 iterations), but on Mittelmann's e_moment_quadknap_17_100 the test fired at
         * iteration 1 on the presolved problem with an X that is no ray of the data as read: a false "dual
         * infeasible". BRISK_CERTIT=k: from iteration k + 1 (test). */
        static int certit = -1; if (certit < 0) { const char *e = getenv("BRISK_CERTIT"); certit = e ? atoi(e) : 3; }
        if (it > certit && !par->known_feasible) {
            /* certificates need tau small against kappa: near a normal optimum (tau ~ 1)
             * the ratio tests alone can fire on badly scaled data */
            if (bty > 0 && nAty / bty < 1e-8 && tau < kap) { status = 1; break; }
            if (cX < 0 && nAX / (-cX) < 1e-8 && tau < kap) { status = 2; break; }
            if (tau < 1e-10 * kap) { status = (bty > 0) ? 1 : 2; break; }
        }
        if (it == par->maxit) { status = 3; break; }
        if (stall >= 3) { status = 4; g_hsd_why = "three steps below 1e-6"; break; }
        /* collapsed steps near the accuracy floor: the directions have lost their primal
         * accuracy (case162 TSSOS: alpha 0.009, 0.000, 0.000 after reaching 1.2e-7); the
         * best iterate is kept and polished, the remaining iterations were wasted */
        if (tiny_steps >= 2 && *best_score < 1e-6) { status = 4; g_hsd_why = "collapsed steps near the accuracy floor"; break; }
        /* 4.26: a crawl near the target. Once within 100 tol, three steps shorter than 0.25
         * in a row, without the current point beating the best one, mean the direction has
         * lost its accuracy: the best iterate goes to polishing. The stall rules below
         * would keep the crawl going on a creeping score (control11_s: best at iteration
         * 12, then 20 iterations at alpha 0.01-0.2; whether it ended OPTIMAL or at 1e-8
         * depended on the rounding of the creep). BRISK_CRAWL=0 turns it off. */
        {
            const int er = endrules_check(&ER, it, it > 0 ? alpha : 0.0, improved_er, *best_score);
            if (er) {
                if (par->verbose > 1) printf("   HSD: %s (best %.1e at iteration %d), stopping\n",
                                             er == 1 ? "three collapsed steps" : er == 2 ? "creeping at short steps" : "no halving below 1e-8", *best_score, *it_best);
                status = 4; g_hsd_why = "collapsed or creeping steps"; break;
            }
        }
        if (crawl_on && small_steps >= 3 && *best_score < 100.0 * ptol && *best_score > ptol &&
            it > *it_best && score >= 0.7 * *best_score) {
            if (par->verbose > 1) printf("   HSD: crawl near the target (best %.1e at iteration %d), stopping\n", *best_score, *it_best);
            status = 4; g_hsd_why = "crawl near the target"; break;
        }
        /* 4.22: a collapsed step whose (tiny) move multiplied the primal residual by 1000:
         * the direction was garbage and the following ones are too (TSSOS case2869/6468:
         * alpha 0.000 and pinf 1e-4 -> 78, three more iterations without progress) */
        if (tiny_steps >= 1 && it - *it_best >= 1 && *best_score < 1e-2 && R->pinf > 1e3 * fmax(best->pinf, 1e-300)) {
            /* 4.24: once, back to the best iterate and on with safe settings (full-accuracy
             * solves, no loose checks, no correctors): the collapse at mu ~ 1e-8 is a lost
             * direction, and which iteration it hits depends on rounding (TSSOS case6468:
             * collapsed at iteration 43 with pinf 6e-6 in one build, fine to 1e-8 in another) */
            if (!rescued && keep_best && *best_score > par->tol && getenv("BRISK_RESCUE")) {   /* opt-in: did not help on TSSOS case6468 (collapsed again from the restored point) */
                rescued = 1;
                if (par->verbose > 1) printf("   HSD: collapsed step (primal residual %.1e against %.1e): back to iteration %d with safe settings\n", R->pinf, best->pinf, *it_best);
                int bad = 0;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    const size_t len = bsz(B);
                    memcpy(s->X, s->Xb, sizeof(double) * len); memcpy(s->Z, s->Zb, sizeof(double) * len);
                    if (B->type != BLK_SDP) { memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len); }
                    else {
                        memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len);
                        if (chol_lower(B->n, s->LX) || chol_lower(B->n, s->LZ)) bad = 1;
                    }
                    s->stepped[0] = s->stepped[1] = 0;
                }
                if (!bad) {
                    for (int k = 0; k < nb; k++) S[k].zon = 0;      /* the dense factors were just made */
                    for (int i = 0; i < m; i++) y[i] = ybest[i];
                    tau = 1.0; kap = kap_best; theta = theta_best; pth = pth_best; offpath = 0;
                    g_looseskip = 0; looseskip_safe = 1; soltol = par->hsd_soltol;
                    kfix = -1; tiny_steps = 0; stall = 0;
                    continue;
                }
            }
            if (par->verbose > 1) printf("   HSD: collapsed step (primal residual %.1e against %.1e), stopping\n", R->pinf, best->pinf);
            status = 4; g_hsd_why = "collapsed step"; break;
        }
        /* progress for the stall rules: the best score must drop by 30%, not by a hair
         * (at the accuracy floor it creeps down by 1% per iteration for dozens of them) */
        if (*best_score < 0.7 * score_prog) { score_prog = *best_score; it_prog = *it_best; mu_at_best = mu_best_it; }
        /* steady progress of the current iterate also counts: the score can sit above an
         * early best while it falls (split free pairs: their complementarity dominates
         * relcomp until they settle, and the gap measure rose from a lucky early value) */
        if (it < 4096) { shist[it] = score; phist[it] = R->pinf; ihist[it] = fmax(R->pinf, R->dinf); mhist[it] = mu; ghist[it] = R->relgap; }
        if (onpath && it >= 4 && it < 4096 && score < 0.7 * shist[it - 4] && score < 10.0 * *best_score && it > it_prog) it_prog = it;
        /* 4.38 bound mode (A3): an improving tracked bound is progress: the stall rules do
         * not stop the solve while it improves (the other side may be blowing up) */
        if (btrack_it_imp(par) == it) it_prog = it;
        if (it_prog > it_sprog) it_sprog = it_prog;
        /* slow but steady convergence (4.22): over 8 iterations the infeasibility fell by
         * 10% and mu by half, while the score (relcomp, as tau falls) does not improve.
         * Large AC-OPF (TSSOS case6468: the objective moves 42 -> 149 over 60 iterations at
         * alpha 0.1-0.4, as MOSEK's does) stopped at iteration 27 with 8e-5. Bounded by
         * 6 hsd_stall iterations after the last score progress. */
        if (steady_on && onpath && it >= 8 && it < 4096 && ihist[it - 8] > 0 && ihist[it] < 0.9 * ihist[it - 8] &&
            mhist[it] < 0.5 * mhist[it - 8] && it - it_sprog < 6 * par->hsd_stall && it > it_prog) it_prog = it;
        if (it - it_prog >= par->stall_win && *best_score < 1e-6) { status = 4; g_hsd_why = "no progress near the accuracy floor"; break; }
        /* no better iterate for hsd_stall iterations: give up, unless mu keeps falling fast
         * (the score can rise for a while when tau goes to zero, e.g. kernel forms, whose
         * optimum is approached only as the Gram matrices grow) */
        if (it - it_prog >= par->hsd_stall && !(*best_score > 1e-5 && mu < 1e-1 * mu_at_best && it - it_prog < 5 * par->hsd_stall)) { status = 4; g_hsd_why = "no better iterate for the stall window"; break; }
        /* at the accuracy floor: a best iterate within 10 tol that has not improved for three
         * iterations while the current one is much worse (the Newton directions have lost
         * their primal accuracy) will not be improved on */
        /* 5.4: not while the iteration takes full steps and mu still falls by 100x since the best
         * iterate: the best iterate then sits on a plateau of the central path (an ill-posed
         * problem whose objective is decided by a tiny coefficient: a peak-power problem
         * sent by a user, value 0.0072, reached only at mu 1e-18 after a plateau at 0 with all
         * measures below 4e-8). Bounded by 12 iterations past the best. */
        const int full_steps = onpath && small_steps == 0 && mu < 1e-2 * mu_best_it && it - *it_best < 12;
        if (*best_score < 10.0 * ptol && it - *it_best >= 3 && score > 10.0 * *best_score && !full_steps) { status = 4; g_hsd_why = "the errors rise past the best iterate"; break; }
        /* primal floor (4.20): the primal residual dominates the score by 10x, has not fallen
         * in 4 iterations and the best iterate is 4 old: the directions no longer reduce it
         * (roa_univar_d16_I: pinf 3.5e-6 .. 4.4e-6 for 14 iterations while mu fell 1e4x) */
        if (g_pfloor && it >= 8 && it < 4096 && *best_score < 1e-4 && it - *it_best >= 4 && phist[it - 4] > 0 &&
            R->pinf >= 0.9 * phist[it - 4] && fmax(fmax(R->relgap, R->relcomp), R->dinf) < 0.1 * R->pinf) {
            if (par->verbose > 1) printf("   HSD: primal residual floor (%.1e), stopping\n", R->pinf);
            status = 4; g_hsd_why = "primal residual floor"; break;
        }
        /* primal breakdown at the accuracy floor: mu has reached the rounding level and the
         * primal residual, after falling below 1e-6, jumps 100x above its lowest value (or 10x
         * in two consecutive iterations). The directions have lost their primal accuracy and
         * do not regain it: on degenerate moment relaxations (singular optima, no strict
         * complementarity) the residual then stays 1e3-1e6 above its minimum while the run
         * goes on for 6-11 more iterations (ni35 order 3: 2.3e-8 at iteration 43, 2.8e-5 at
         * 45, 1e-2 until 56; ni10 order 3: 2.2e-10, then 1.5e-6 .. 8e-4). The score rules
         * above do not see it, since the relative complementarity keeps the best score at
         * 1e-6..1e-4 there. The best iterate is returned as by the other rules.
         * Only once the duality gap had converged (below 1e-7 at the lowest residual): while
         * the gap is still closing, the iterations after such a jump keep improving it and
         * the objective (multifreq8: residual 6e-9 -> 4e-5 at gap 1e-6, then the gap fell to
         * 3e-8 and the objective moved 1e-3 relative; stopping there changed the result).
         * And only once the best iterate is two iterations old: right after the jump the
         * objective can still improve, and polishing repairs the residual of such a point
         * (ni35 order 3: the best iterate 46, after the jump at 45, gave the certified bound;
         * stopping at 45 would have lost 0.3-0.6 of it). And only once the residual itself is
         * above the reduced-accuracy level 1e-6: a jump from 1e-15 to 2.6e-8 is no breakdown
         * (beamSelfWeight_3_msos_ad: stopping there returned 9.1e-9 instead of 2.9e-9). */
        if (g_brkdn && it >= 8 && it < 4096 && mu < 1e-13) {
            double pmin = INFINITY;
            int imin = -1;
            for (int j = 0; j < it; j++) if (phist[j] > 0 && phist[j] < pmin) { pmin = phist[j]; imin = j; }
            if (pmin < 1e-6 && imin >= 0 && ghist[imin] < 1e-7 && it - *it_best >= 2 && R->pinf > 1e-6 && (R->pinf > 100.0 * pmin || (R->pinf > 10.0 * pmin && phist[it - 1] > 10.0 * pmin))) {
                if (par->verbose > 1) printf("   HSD: primal breakdown at the accuracy floor (residual %.1e against %.1e at iteration %d, mu %.1e), stopping\n",
                                             R->pinf, pmin, imin, mu);
                status = 4; g_hsd_why = "primal breakdown at the accuracy floor"; break;
            }
        }

        if (use_pat) {
            const double dev = sqrt(dev2);
            if (dev > 1e-10 * (1.0 + theta * r0norm) && dev > 1e-12) {
                /* re-base: R0 <- true residual (dense: it contains Z), products go dense */
                if (rp0) { memcpy(rp0, rp, sizeof(double) * m); rg0 = rg; }
                theta = 1.0; r0norm = sqrt(rd2); nrebase++; use_pat = 0;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    size_t len = bsz(B);
                    if (!s->R0) s->R0 = amalloc(sizeof(double) * len);
                    memcpy(s->R0, s->F, sizeof(double) * len);
                    if (B->type == BLK_SDP) setup_residual_pattern(B, s, par);
                    if (pk[k]) {
                        free(hp[k].fr); free(hp[k].fc); free(hp[k].rows);
                        setup_hsd_pattern(B, s, par, &hp[k]);
                        pk[k] = hp[k].route != 2;
                    }
                    use_pat |= pk[k];
                }
            }
        }
        { double _t = wtime(); g_ph[0] += _t - g_ph_last; g_ph_last = _t; }
        /* ---- Schur complement */
        double tsch = wtime(), t_fact_it = 0;
        {
            /* 5.12: the row products of the Schur assembly in single precision (sgemm, the gather summed
             * in double) while the factor is single precision and the residuals and the gap are above
             * 1e-3: there M is factored in single precision anyway and the direction needs a few digits
             * only (from 1e-3 on, a user's ball (5,5) took 9 iterations instead of 5). The products run
             * at 62-64 GF/s a thread against 37 (box (4,8): assembly 16.2 -> 10.8 s with the symmetric
             * gather, which the single-precision products always take). BRISK_ASMF=0 off,
             * BRISK_ASMFTHR the threshold. */
            /* 5.12: the factor in double precision once the residuals and the gap are all below 1e-6
             * (the last iteration or two): the single-precision factor with PCG to the requested
             * tolerance gave a direction that raised the primal residual from 3e-8 to 7e-6 where the
             * PCG statistics had not yet switched to double (a generated ball (5,7): 5 -> 7 iterations
             * after the row products of the first iterations changed in the last bits). BRISK_FLLATE
             * the threshold (0: off). */
            { static double fl_late = -1; if (fl_late < 0) { const char *e = getenv("BRISK_FLLATE"); fl_late = e ? atof(e) : 1e-6; }
              if (Sc->allow_float && !Sc->want_double && fmax(R->pinf, fmax(R->dinf, R->relgap)) < fl_late) Sc->want_double = 1; }
            { extern int g_asmsp; static int asmf_env = -1; static double asmf_thr = 1e-3;
              if (asmf_env < 0) { const char *e0 = getenv("BRISK_ASMF"); asmf_env = e0 ? atoi(e0) : 1; const char *e = getenv("BRISK_ASMFTHR"); if (e) asmf_thr = atof(e); }
              g_asmsp = asmf_env && Sc->allow_float && !Sc->want_double && !Sc->SC && !Sc->E
                        && fmax(R->pinf, fmax(R->dinf, R->relgap)) > asmf_thr;
              extern int g_ntsp; g_ntsp = g_asmsp; }
            double *Ma = schur_asm_begin(Sc);
            if (bord && bord_next && !bord_on) {
                if (!BV.KS && free_ks_setup(P, Sc, &BV) != 0) { bord = 0; bord_next = 0; }
                else {
                    bord_on = 1;
                    if (par->verbose > 1) printf("   HSD: bordered factorization %.2e flops (M alone %.2e)\n", schol_flops(BV.KS), schol_flops(Sc->SC));
                    /* M is only assembled from now on (bord_factor reads its values): its factor
                     * storage goes (0.8 GB on dense case9241) */
                    schol_release_factor(Sc->SC);
                }
            }
            if (bord_on) { memset(bD, 0, sizeof(double) * nbord); g_bpf = bpf; g_bpB = &P->blk[BV.kb[0]]; g_bD = bD; }
            static int asmdbg = -1; if (asmdbg < 0) asmdbg = getenv("BRISK_SOLDBG") != NULL;
            g_asmdbg = asmdbg;
            if (asmdbg && nthreads() <= 1)
            for (int k = 0; k < nb; k++) {
                BS vtmp;
                const double ta0 = asmdbg ? wtime() : 0;
                if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], &S[k], m, Ma);
                else if (P->blk[k].type == BLK_SOC) schur_soc(&P->blk[k], &S[k], m, Ma);
                else schur_sdp(&P->blk[k], nt_view(&S[k], &vtmp, Zv[k] != NULL), m, Ma);
                if (asmdbg) { const Block *Bk = &P->blk[k]; g_asm[Bk->type == BLK_LP ? 0 : Bk->lowrank ? 1 : Bk->dict ? 2 : 3] += wtime() - ta0; }
            }
            else schur_assemble_all(P, S, Sc, Zv, m, Ma);
            if (bord_on) {
                g_bpf = NULL; g_bpB = NULL; g_bD = NULL;
                for (int f = 0; f < nbord; f++) Sc->bE[f] = bD[f] > 1e-300 ? 1.0 / bD[f] : 1e300;
            }
            if (nf && fmode == 1) { double *M = Ma; free_al_assemble(P, Sc, &FV, M, par->hsd_fomega); }
            schur_asm_end(Sc);
            /* 5.12: the Schur solves' products with M from the blocks when that is cheaper than
             * reading M: sum n_b^3 < 1.5 m^2 (4 sum n^3 flops against 4 m^2 bytes: the products of order
             * 792 ran at 70 GF/s, M was read at 41 GB/s; BRISK_MVFREE=0 off, 1 always, BRISK_MVFRATIO) */
            { static int mvf_env = -2; if (mvf_env == -2) { const char *e = getenv("BRISK_MVFREE"); mvf_env = e ? atoi(e) : -1; }
              static double mvf_ratio = -1; if (mvf_ratio < 0) { const char *e = getenv("BRISK_MVFRATIO"); mvf_ratio = e ? atof(e) : 1.5; }
              if (mvf_env != 0 && !Sc->SC && !Sc->E && !bord_on && !(nf && fmode == 1) && !g_has_soc) {
                  double cub = 0, dat = 0;
                  for (int k = 0; k < nb; k++) {
                      const Block *Bk = &P->blk[k]; const double nk = Bk->n;
                      if (Bk->type != BLK_LP) cub += nk * nk * nk;
                      /* 5.18: ... and the constraint data the product reads twice (blk_ATy, blk_Aop): the dense
                       * copies Ad (n^2 doubles each), the stored entries of the others (about 16 bytes each); in
                       * cub units at the rates above, 0.43 per byte. A-optimal design (one LMI of order 180, 2,000
                       * dense rank-one constraints) read 1 GB a product: 63 ms against 10 ms for M itself */
                      if (Bk->type == BLK_SDP && Bk->nd) dat += 16.0 * nk * nk * Bk->nd;
                      for (int t = 0; t < Bk->ncon; t++) {
                          if (Bk->type == BLK_SDP && Bk->dpos && Bk->dpos[t] >= 0) continue;
                          dat += 32.0 * (Bk->type == BLK_SDP ? Bk->A[t].ef : Bk->A[t].nnz);
                      }
                  }
                  static int mvf_dat = -1; if (mvf_dat < 0) { const char *e = getenv("BRISK_MVFDATA"); mvf_dat = e ? atoi(e) : 1; }
                  if (mvf_env == 1 || cub + (mvf_dat ? 0.43 * dat : 0.0) < mvf_ratio * (double)m * m) mvfree_take(Sc, P, S, Zv);
              } }
            {   /* (test) BRISK_DUMPW=k: the scaling matrices of the assemblies from iteration k on, to BRISK_DUMPDIR */
                static int dw = -2; if (dw == -2) { const char *e = getenv("BRISK_DUMPW"); dw = e ? atoi(e) : -1; }
                if (dw >= 0 && it >= dw) {
                    const char *dir = getenv("BRISK_DUMPDIR"); char fn[512];
                    for (int k = 0; k < nb; k++) {
                        const Block *Bk = &P->blk[k];
                        const size_t len = Bk->type != BLK_SDP ? (size_t)Bk->n : (size_t)Bk->n * Bk->n;
                        snprintf(fn, sizeof fn, "%s/W_%d_%d.bin", dir ? dir : ".", it, k);
                        FILE *fw = fopen(fn, "wb");
                        if (fw) { fwrite(Zv[k] ? S[k].Zi : S[k].X, sizeof(double), len, fw); if (!Zv[k]) fwrite(S[k].Zi, sizeof(double), len, fw); fclose(fw); }
                    }
                    printf("   [dumped the scaling matrices of iteration %d]\n", it);
                }
            }
        }
        R->t_schur += wtime() - tsch;
        double tc = wtime();
        t_fact_it = wtime() - tsch;
        if (nf && fmode == 3 && !Sc->SC && !Sc->E) {
            if (free_factor_aug(P, Sc, &FV)) { status = 4; g_hsd_why = "factorization of the free-variable system failed"; break; }
        } else if (nf && fmode == 4) {
            if (free_ks_factor(P, Sc, &FV)) { status = 4; g_hsd_why = "factorization of the free-variable system failed"; break; }
        } else if (bord_on) {
            Sc->bF = NULL;
            if (bord_factor(P, Sc, &BV)) { status = 4; g_hsd_why = "factorization of the bordered system failed"; break; }
            Sc->bF = &BV; Sc->bP = P;
        } else {
            if (schur_factor(Sc)) { status = 4; g_hsd_why = "factorization of the Schur complement failed"; break; }
            if (nf && fmode != 1 && free_factor(P, Sc, &FV)) { status = 4; g_hsd_why = "factorization of the free-variable system failed"; break; }
            { static int qrm = -1; if (qrm < 0) { const char *e = getenv("BRISK_HSDQR"); qrm = e ? atoi(e) : 0; }
              if (qrm && !nf && (qrm >= 2 || nref_fail > 0)) hsd_qr_factor(P, S, Sc, Zv); }
        }
        R->t_chol += wtime() - tc;
        t_fact_it = wtime() - tsch;
        if (getenv("BRISK_SOLVEBENCH") && it == 1) {
            SChol *F = bord_on ? BV.KS : Sc->SC;
            if (F) {
                const int n = bord_on ? m + BV.nf : m;
                double *b = malloc(sizeof(double) * (n + 1)), *wk = malloc(sizeof(double) * (n + 1)), *y2 = malloc(sizeof(double) * (n + 1));
                for (int i = 0; i < n; i++) b[i] = 1.0 / (1 + i % 7);
                double t0 = wtime();
                for (int r = 0; r < 10; r++) schol_solve(F, b, wk);
                double ts = (wtime() - t0) / 10;
                t0 = wtime();
                for (int r = 0; r < 10; r++) schol_mv(Sc->SC, b, y2);
                double tm = (wtime() - t0) / 10;
                double *B3 = malloc(sizeof(double) * 3 * (n + 1)), *W3 = malloc(sizeof(double) * 3 * (n + 1));
                for (int i = 0; i < 3 * n; i++) B3[i] = 1.0 / (1 + i % 5);
                t0 = wtime();
                for (int r = 0; r < 10; r++) schol_solve_many(F, B3, 3, W3);
                double t3 = (wtime() - t0) / 10;
                t0 = wtime();
                for (int r = 0; r < 10; r++) { schol_solve2(F, B3, B3 + n, W3); schol_solve(F, B3 + 2 * n, wk); }
                double t21 = (wtime() - t0) / 10;
                printf("   [solve bench: 3 rhs: solve_many %.1f ms, solve2+solve %.1f ms, 3 x solve %.1f ms]\n", 1e3 * t3, 1e3 * t21, 3e3 * ts);
                free(B3); free(W3);
                const double gb = 2.0 * 8.0 * (double)schol_pansz(F) / 1e9;
                printf("   [solve bench: n %d, panels %.3g entries, nnz(L) %.3g: solve %.1f ms (%.2f GB/s over the panels), M*x %.1f ms, %d supernodes]\n",
                       n, (double)schol_pansz(F), (double)schol_nnz(F), 1e3 * ts, gb / ts, 1e3 * tm, schol_nsuper(F));
                free(b); free(wk); free(y2);
            }
        }

        { double _t = wtime(); g_ph[1] += _t - g_ph_last; g_ph_last = _t; }
        /* ---- pieces: a0 = A(Zi), h_d = A(sym(Zi Rd X)), h_c = A(sym(Zi C X)) */
        td = wtime();
        shc.on = nf ? 0 : shift_env == 2 ? 1 : (shift_env == 1 && shift_next); shc.tau = tau; shc.theta = theta; shc.pk = pk;
        {   /* experiment (BRISK_TAUFREEZE=g): tau kept (dtau = 0) once the relative gap is below g. Near the end the
             * embedding's solutions form a ray, tau is a normalization only, and the Newton direction's tau component
             * (scaling y, Z and tau one way and X the other) reopens the gap (ISSUES 84) */
            static double tfz = -1; if (tfz < 0) { const char *e = getenv("BRISK_TAUFREEZE"); tfz = e ? atof(e) : 0.0; }
            shc.tauf = (tfz > 0 && it > 0 && R->relgap < tfz) ? 0.0 : 1.0;
        }
        memset(a0, 0, sizeof(double) * m);
        memset(hd, 0, sizeof(double) * m);
        memset(hc, 0, sizeof(double) * m);
        double hcRd = 0, hcC = 0, cZi = 0;
        double *prp3 = hred(nb, 3);
        aop_begin(a0, hd, hc, NULL);
        HBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            int n = B->n;
            double hcRd = 0, hcC = 0, cZi = 0;
            const double *ZI = Zv[k] ? Zv[k] : s->Zi;              /* true Z^{-1} */
            const double *RX = Zv[k] ? s->Zi : s->X;              /* right factor: X or W */
            blk_Aop(B, s, ZI, a0, 1.0);
            cZi += sp_inner(B, &B->C, ZI);
            /* full products: in the embedding these matrices contain Z and C, so the
             * pattern-restricted route of the standard method does not apply          */
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * s->F[i] * s->X[i];
            else if (B->type == BLK_SOC) soc_H(s, n, s->F, s->G, s->stmp);
            else if (pk[k]) {                                  /* Rd = theta R0 on the pattern */
                memset(s->G, 0, sizeof(double) * len);
                add_R0(s, len, theta, s->G);
                prod_pat(&hp[k], n, s, s->G, RX, s->G);
                symmetrize(n, s->G);
            }
            else { prod_zi(n, s, s->F, RX, s->G); symmetrize(n, s->G); }
            blk_Aop(B, s, s->G, hd, 1.0);
            hcRd += sp_inner(B, &B->C, s->G);
            memset(s->dX, 0, sizeof(double) * len);
            sp_add(B, &B->C, 1.0, s->dX);                              /* dense copy of C */
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->Q[i] = s->Zi[i] * s->dX[i] * s->X[i];
            else if (B->type == BLK_SOC) soc_H(s, n, s->dX, s->Q, s->stmp);
            else if (pk[k]) { prod_pat(&hp[k], n, s, s->dX, RX, s->Q); symmetrize(n, s->Q); }
            else { prod_zi(n, s, s->dX, RX, s->Q); symmetrize(n, s->Q); }
            blk_Aop(B, s, s->Q, hc, 1.0);
            hcC += sp_inner(B, &B->C, s->Q);
            prp3[3 * (size_t)k] = cZi; prp3[3 * (size_t)k + 1] = hcRd; prp3[3 * (size_t)k + 2] = hcC;
        });
        aop_end();
        for (int k = 0; k < nb; k++) { cZi += prp3[3 * (size_t)k]; hcRd += prp3[3 * (size_t)k + 1]; hcC += prp3[3 * (size_t)k + 2]; }
        double rgs = rg, cXs = cX;      /* the gap row's right-hand side and <C, X> of the tau column */
        hcC0 = hcC;
        const double hcRd_o = hcRd, hcC_o = hcC;   /* (diagnostic BRISK_TAUBOTH: the original form's scalars) */
        double *hc_o = NULL;
        if (shc.on && ENV_ON("BRISK_TAUBOTH")) { hc_o = malloc(sizeof(double) * (m + 1)); memcpy(hc_o, hc, sizeof(double) * m); }
        if (shc.on) {
            /* G = H(Rd) from the pieces; with Rd' = theta R0 on the pattern route (the model the
             * directions use) and a copy of Rd otherwise (the correctors reuse F as a work array) */
            double *p4 = hred(nb, 4);
            HBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                const size_t len = bsz(B);
                if (!pk[k]) {
                    if (!shc.F[k]) shc.F[k] = amalloc(sizeof(double) * len);
                    memcpy(shc.F[k], s->F, sizeof(double) * len);
                }
                const double *ZI = Zv[k] ? Zv[k] : s->Zi;
                p4[4 * (size_t)k] = shift_dotF(&shc, s, k, len, s->X);
                p4[4 * (size_t)k + 1] = shift_dotF(&shc, s, k, len, s->G);
                p4[4 * (size_t)k + 2] = shift_dotF(&shc, s, k, len, ZI);
                p4[4 * (size_t)k + 3] = shift_nu(B);
            });
            double fx = 0, fg = 0, fzi = 0, nu = 0;
            for (int k = 0; k < nb; k++) { fx += p4[4 * (size_t)k]; fg += p4[4 * (size_t)k + 1]; fzi += p4[4 * (size_t)k + 2]; nu += p4[4 * (size_t)k + 3]; }
            /* A(H(C^)) = (A(X) + h_d) / tau = b + (h_d - rp) / tau (rp = tau b - A(X));
             * <C^, H C^> = (<X,Z> + 2 <Rd,X> + <Rd,H Rd>) / tau^2; <C^, H Rd> = (<Rd,X> + <Rd,H Rd>) / tau;
             * <C^, Z^-1> = (nu + <Rd,Z^-1>) / tau; <C^, X> = (<X,Z> + <Rd,X>) / tau */
            for (int i = 0; i < m; i++) { bmh[i] = (rp[i] - hd[i]) / tau; hc[i] = P->b[i] - bmh[i]; }
            hcC = (xz + 2.0 * fx + fg) / (tau * tau);
            hcRd = (fx + fg) / tau;
            cZi = (nu + fzi) / tau;
            cXs = (xz + fx) / tau;
            rgs = kap + cXs;
        }
        R->t_dense += wtime() - td;
        g_ct[8] += wtime() - td;
        tc = wtime();
        for (int i = 0; i < m; i++) rhs[i] = P->b[i] + hc[i];
        if (nf) free_solve(P, Sc, &FV, rhs, FV.c, v, vf);
        else if (par->hsd_batch) {
            /* v and the predictor solve together (both right-hand sides are known now) */
            for (int i = 0; i < m; i++) qv[i] = tau * P->b[i] + hd[i];
            const double *rr[2] = { rhs, qv };
            double *xx[2] = { v, u0 };
            const double tt[2] = { soltol, 1e-7 };
            schur_solve_multi(Sc, 2, rr, xx, tt);
        }
        else schur_solve(Sc, rhs, v, soltol);                               /* independent of sigma */
        R->t_chol += wtime() - tc;
        double bv = ddot_n(m, P->b, v), hcv = ddot_n(m, hc, v);
        double denom = shc.on ? ddot_n(m, bmh, v) + hcC + kap / tau
                              : bv + hcC - hcv + kap / tau - (nf ? ddot_n(nf, FV.c, vf) : 0.0);
        if (!(fabs(denom) > 0)) { status = 4; g_hsd_why = "the embedding's scalar equation is singular or not finite"; break; }

        { double _t = wtime(); g_ph[2] += _t - g_ph_last; g_ph_last = _t; }
        /* ---- predictor (sigma = 0, eta = 1, K = 0) */
        td = wtime();
        for (int i = 0; i < m; i++) rhs[i] = tau * P->b[i] + hd[i];
        if (nf) free_A(P, &FV, FV.xf, -1.0, rhs);                  /* -A(K) = A(X) of the cone only */
        R->t_dense += wtime() - td;
        tc = wtime();
        if (nf) free_solve(P, Sc, &FV, rhs, rf, u0, ufa);
        else if (!par->hsd_batch) schur_solve(Sc, rhs, u0, 1e-7);
        R->t_chol += wtime() - tc;
        double dtau_a = shc.tauf * ((shc.on ? rgs - cXs - hcRd - ddot_n(m, bmh, u0) : rg - (cX - cfx) - hcRd - ddot_n(m, P->b, u0) + ddot_n(m, hc, u0)) - kap
                         + (nf ? ddot_n(nf, FV.c, ufa) : 0.0)) / denom;
        if (hc_o) {   /* diagnostic: the predictor's dtau in the original form at this iterate */
            double *rh = malloc(sizeof(double) * (m + 1)), *vo = malloc(sizeof(double) * (m + 1));
            for (int i = 0; i < m; i++) rh[i] = P->b[i] + hc_o[i];
            schur_solve(Sc, rh, vo, soltol);
            const double den_o = ddot_n(m, P->b, vo) + hcC_o - ddot_n(m, hc_o, vo) + kap / tau;
            const double num_o = rg - cX - hcRd_o - ddot_n(m, P->b, u0) + ddot_n(m, hc_o, u0) - kap;
            const double num_s = rgs - cXs - hcRd - ddot_n(m, bmh, u0) - kap;
            printf("     [tau both it %d: shifted dtau %.6e (num %.3e den %.3e; bmh'u0 %.3e bmh'v %.3e hcC %.3e)  original dtau %.6e (num %.3e den %.3e; b'v %.3e hcC %.3e hc'v %.3e)]\n",
                   it, dtau_a, num_s, denom, ddot_n(m, bmh, u0), ddot_n(m, bmh, v), hcC, num_o / den_o, num_o, den_o, ddot_n(m, P->b, vo), hcC_o, ddot_n(m, hc_o, vo));
            free(rh); free(vo); free(hc_o); hc_o = NULL;
        }
        if (shc.on && !nf && ENV_ON("BRISK_TAUDWR")) {
            /* the scalars of the shifted tau equation with the dual-weighted residual correction: the error of w'x for a
             * computed solve x of M x = r is z'(r - M x), z = M^-1 w, to first order (w = bmh here) */
            double *z = malloc(sizeof(double) * (m + 1)), *rr = malloc(sizeof(double) * (m + 1));
            schur_solve(Sc, bmh, z, soltol);
            for (int i = 0; i < m; i++) rr[i] = P->b[i] + hc[i];
            schur_mv(Sc, -1.0, v, 1.0, rr);
            const double cv = ddot_n(m, z, rr);
            for (int i = 0; i < m; i++) rr[i] = tau * P->b[i] + hd[i];
            schur_mv(Sc, -1.0, u0, 1.0, rr);
            const double cu = ddot_n(m, z, rr);
            const double bv = ddot_n(m, bmh, v), bu = ddot_n(m, bmh, u0);
            const double den_c = bv + cv + hcC + kap / tau, num_c = rgs - cXs - hcRd - (bu + cu) - kap;
            printf("     [tau dwr it %d: dtau %.6e -> %.6e (den %.3e -> %.3e, bmh'v %.3e + %.3e; bmh'u0 %.3e + %.3e)]\n",
                   it, dtau_a, num_c / den_c, denom, den_c, bv, cv, bu, cu);
            free(z); free(rr);
        }
        double dkap_a = -kap - (kap / tau) * dtau_a;
        td = wtime();
        for (int i = 0; i < m; i++) dy[i] = u0[i] + dtau_a * v[i];
        HBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            int n = B->n;
            blk_ATy(B, s, dy, s->dZ);
            if (pk[k]) { for (size_t i = 0; i < len; i++) s->dZ[i] = -s->dZ[i]; add_R0(s, len, theta, s->dZ); }
            else for (size_t i = 0; i < len; i++) s->dZ[i] = s->F[i] - s->dZ[i];
            if (shc.on) shift_pre(&shc, s, k, len, dtau_a, s->dZ);
            else sp_add(B, &B->C, dtau_a, s->dZ);
            const double *RX = Zv[k] ? s->Zi : s->X;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * s->dZ[i] * s->X[i];
            else if (B->type == BLK_SOC) soc_H(s, n, s->dZ, s->G, s->stmp);
            else if (pk[k]) { prod_pat(&hp[k], n, s, s->dZ, RX, s->G); symmetrize(n, s->G); }
            else { prod_zi(n, s, s->dZ, RX, s->G); symmetrize(n, s->G); }
            if (shc.on) shift_post(&shc, s, len, dtau_a, 1.0, s->G, s->dZ);
            PAR_LEN_FOR(len, s->dX[i] = -s->G[i] - s->X[i];);
        });
        if (nf) {
            for (int f2 = 0; f2 < nf; f2++) dxfd[f2] = ufa[f2] + dtau_a * vf[f2];
            free_set_dir(&FV, S, NULL, NULL, dxfd, 0);
        }
        R->t_dense += wtime() - td;
        g_ct[9] += wtime() - td;
        double tst = wtime();
        const double aaX = maxstep(P, S, 1, par->lanczos_k), aaZ = maxstep(P, S, 0, par->lanczos_k);
        double aa = fmin(aaX, aaZ);
        if (dtau_a < 0) aa = fmin(aa, -tau / dtau_a);
        if (dkap_a < 0) aa = fmin(aa, -kap / dkap_a);
        aa = fmin(1.0, aa);
        R->t_step += wtime() - tst;
        double xza = 0;
        {
            double *px = hred(nb, 1);
            HBLK_FOR(k, nb, {
                size_t len = bsz(&P->blk[k]);
                px[k] = dot_step(len, S[k].X, S[k].dX, S[k].Z, S[k].dZ, aa);
            });
            for (int k = 0; k < nb; k++) xza += px[k];
        }
        double mua = (xza + (tau + aa * dtau_a) * (kap + aa * dkap_a)) / ndim1;
        if (ENV_ON("BRISK_HSDDBG")) {
            /* centrality: spectrum of L_X' Z L_X / mu over the SDP blocks */
            double lmn = 1e300, lmx = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                int n = B->n;
                if (B->type == BLK_LP) { for (int i = 0; i < n; i++) { if (s->fm && s->fm[i]) continue; double w = s->X[i] * s->Z[i] / mu; lmn = fmin(lmn, w); lmx = fmax(lmx, w); } continue; }
                if (B->type == BLK_SOC) { const double l0 = s->lam[0], lb = soc_nb(n, s->lam); lmn = fmin(lmn, (l0 - lb) * (l0 - lb) / mu); lmx = fmax(lmx, (l0 + lb) * (l0 + lb) / mu); continue; }
                double *T = malloc(sizeof(double) * n * n), *ev = malloc(sizeof(double) * n);
                memcpy(T, s->Z, sizeof(double) * n * n);
                BL(dtrmm_)("L", "L", "T", "N", &n, &n, &DONE, s->LX, &n, T, &n);
                BL(dtrmm_)("R", "L", "N", "N", &n, &n, &DONE, s->LX, &n, T, &n);
                int lw = 3 * n + 1, info; double *wk = malloc(sizeof(double) * lw);
                BL(dsyev_)("N", "L", &n, T, &n, ev, wk, &lw, &info);
                for (int i = 0; i < n; i++) { lmn = fmin(lmn, ev[i] / mu); lmx = fmax(lmx, ev[i] / mu); }
                free(T); free(ev); free(wk);
            }
            printf("     [centrality: lambda(XZ)/mu in [%.2e, %.2e], tau kap/mu %.2f; affine aX %.3f aZ %.3f -> %.3f, sigma_m %.3f]\n",
                   lmn, lmx, tau * kap / mu, aaX, aaZ, aa, pow(fmax(mua / mu, 0.0), 3.0));
        }
        double expon = R->relgap > 1e-6 ? fmax(1.0, 3.0 * aa * aa) : 3.0;
        { static double hexp = -2; if (hexp < -1) { const char *e = getenv("BRISK_HSDEXP"); hexp = e ? atof(e) : -1; } if (hexp > 0) expon = hexp; }   /* (test) */
        { static int g0i = 0; if (!g0i) { const char *e = getenv("BRISK_HSDG0"); if (e) g_hsdg0 = atof(e); g0i = 1; } }
        double sig_m = fmin(1.0, pow(fmax(mua / mu, 0.0), expon));

        /* the sigma search on the family d0 + sigma (d1 - d0) at hand: the candidate with the largest predicted
         * reduction alpha (1 - sigma); leaves the last candidate's direction in S[k].dX, S[k].dZ */
#define HSD_SIGSEARCH(BM, BSIG, BAL, BAM) do { \
        double cand_[6]; int nc_ = 0; \
        cand_[nc_++] = sig_m; \
        if (par->hsd_sig > 0) { \
            cand_[nc_++] = fmin(0.95, fmax(1e-4, sig_m * 0.2)); \
            cand_[nc_++] = fmin(0.95, fmax(1e-3, sig_m * 3.0)); \
            if (par->hsd_sig > 3) { cand_[nc_++] = 0.05; cand_[nc_++] = 0.4; } \
        } \
        for (int c_ = 0; c_ < nc_; c_++) { \
            double sg_ = cand_[c_], dt_ = dt0 + sg_ * (dt1 - dt0), dk_ = dk0 + sg_ * (dk1 - dk0); \
            HBLK_FOR(k, nb, \
                size_t len = bsz(&P->blk[k]); \
                PAR_LEN_FOR(len, \
                    S[k].dX[i] = dX0[k][i] + sg_ * (dX1[k][i] - dX0[k][i]); \
                    S[k].dZ[i] = dZ0[k][i] + sg_ * (dZ1[k][i] - dZ0[k][i]);); \
            ); \
            tst = wtime(); \
            const double amX_ = maxstep(P, S, 1, par->lanczos_k), amZ_ = maxstep(P, S, 0, par->lanczos_k); \
            double am_ = fmin(amX_, amZ_); \
            R->t_step += wtime() - tst; \
            if (dt_ < 0) am_ = fmin(am_, -tau / dt_); \
            if (dk_ < 0) am_ = fmin(am_, -kap / dk_); \
            if (ENV_ON("BRISK_HSDDBG")) \
                printf("     [sigma %.3g: aX %.3f aZ %.3f atau %.3g akap %.3g -> %.3f]\n", sg_, amX_, amZ_, \
                       dt_ < 0 ? -tau / dt_ : 1e30, dk_ < 0 ? -kap / dk_ : 1e30, am_); \
            double gam_ = g_hsdg0 + (par->gamma_max - g_hsdg0) * fmin(1.0, am_); \
            double al_ = fmin(1.0, gam_ * am_); \
            double merit_ = al_ * (1.0 - sg_);                     /* predicted mu reduction */ \
            if (merit_ > BM) { BM = merit_; BSIG = sg_; BAL = al_; BAM = am_; } \
        } } while (0)
        /* experimental (BRISK_HSDHOC = k): the second-order term re-evaluated k times at the combined
         * direction of the Mehrotra sigma (a fixed-point iteration on the complementarity equation
         * with the factorization at hand) */
        int hoc_on = 0, hoc_kept = 0;   /* passes allowed in this iteration; passes kept */
        double cQ = 0, tk2 = 0, n0 = 0, n1 = 0, dt0 = 0, dt1 = 0, dk0 = 0, dk1 = 0;
        static int nhoc_env = -2; static double hoc_theta = 0.9, hoc_theta1 = 0.9, hoc_amin = 0;
        if (nhoc_env < -1) { const char *e1 = getenv("BRISK_HSDHOCT1"); if (e1) hoc_theta1 = atof(e1); }
        if (nhoc_env < -1) { const char *eh = getenv("BRISK_HSDHOC"); nhoc_env = eh ? atoi(eh) : -1; const char *et = getenv("BRISK_HSDHOCT"); if (et) hoc_theta = atof(et); const char *ek = getenv("BRISK_HSDHOCA"); if (ek) hoc_amin = atof(ek); }
        static int hoc_fmax = -1; if (hoc_fmax < 0) { const char *ef = getenv("BRISK_HSDHOCF"); hoc_fmax = ef ? atoi(ef) : 3; }
        /* the passes (and the long step) are given up for the rest of the solve after hoc_fmax iterations in which
         * they predicted less than the first direction: the fixed point is not the better direction on this problem */
        const int nhoc_max = hoc_nfail >= hoc_fmax ? 0 : nhoc_env >= 0 ? nhoc_env : par->hsd_hoc;
        /* the passes and the long step only after a predictor that went at least hoc_amin of the way to the boundary */
        static double hoc_end = -1; if (hoc_end < 0) { const char *ee = getenv("BRISK_HSDHOCEND"); hoc_end = ee ? atof(ee) : 100.0; }
        /* not within a factor 100 of the tolerance: a problem that arrives there step by step (not in one
         * superlinear step) is ill-conditioned, and the standard step settles better at its accuracy floor
         * (12 fragile problems x 12 perturbations: 82 OPTIMAL with this rule, 75 without, 87 before 5.7) */
        const int hoc_late = hoc_end > 0 && fmax(R->pinf, fmax(R->relgap, R->dinf)) <= hoc_end * par->tol;
        const int nhoc = (aa >= hoc_amin && !hoc_late) ? nhoc_max : 0;
        hoc_on = nhoc > 0;
        /* Qp, qvp, ...: the state of the last accepted pass; Q0s, ...: of pass 0 (Mehrotra's direction), to
         * which the iteration returns unless the passes have converged (total contraction <= kappa) */
        double **Qp = NULL, *qvp = NULL, cQp = 0, tk2p = 0, hoc_d = 0, hoc_d0 = 0;
        double **Q0s = NULL, *qv0s = NULL, *u0s = NULL, *uf0s = NULL, cQ0s = 0, tk20s = 0;
        int hoc_redo = 0, hoc_have = 0; double hoc_m0 = 0, hoc_s0 = 0, hoc_al0 = 0, hoc_am0 = 0, hoc_sig = 0, hoc_ms = 0, hoc_ss = 0, hoc_als = 0, hoc_ams = 0;
        if (nhoc > 0) {
            Qp = calloc(nb + 1, sizeof(double *)); Q0s = calloc(nb + 1, sizeof(double *));
            for (int k = 0; k < nb; k++) { Qp[k] = amalloc(sizeof(double) * bsz(&P->blk[k])); Q0s[k] = amalloc(sizeof(double) * bsz(&P->blk[k])); }
            qvp = amalloc(sizeof(double) * (m + 1)); qv0s = amalloc(sizeof(double) * (m + 1)); u0s = amalloc(sizeof(double) * (m + 1));
            if (nf) uf0s = amalloc(sizeof(double) * (nf + 1));
        }
        for (int hoc = 0; ; hoc++) {
        /* ---- second-order terms (cone and the tau-kappa pair) */
        td = wtime();
        const double tpt0 = td;
        memset(qv, 0, sizeof(double) * m);
        /* second-order term, optionally damped by the affine step (the correction for a step
         * alpha is alpha^2 dX dZ; Mehrotra's alpha = 1 over-corrects after short predictors) */
        const double qsc = par->hsd_qscale == 1 ? aa : par->hsd_qscale == 2 ? aa * aa : 1.0;
        cQ = 0; tk2 = qsc * dtau_a * dkap_a;
        double *pq = hred(nb, 1);
        aop_begin(qv, NULL, NULL, NULL);
        HBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->Q[i] = s->Zi[i] * s->dZ[i] * s->dX[i];
            else if (B->type == BLK_SOC) {
                /* 5.16: Q = W (lam \ ((W^{-1} dX) o (W dZ))) */
                double *t1 = s->stmp, *t2 = s->stmp + n, *t3 = s->stmp + 2 * (size_t)n;
                soc_Winv(s, n, s->dX, t1); soc_W(s, n, s->dZ, t2);
                soc_jprod(n, t1, t2, t3); soc_jdiv(n, s->lam, t3, t3);
                soc_W(s, n, t3, s->Q);
            }
            else if (Zv[k]) {
                /* NT: Q = G L_lam^{-1}(sym(dXt dZt)) G', dXt = G^{-1} dX G^{-T}, dZt = G' dZ G */
                nt_toX(B, s, s->dX, s->dXt);
                nt_toZ(B, s, s->dZ, s->dZt);
                pdgemm("N", "N", &n, &n, &n, &DONE, s->dXt, &n, s->dZt, &n, &DZERO, s->Rt, &n);
                symmetrize(n, s->Rt);
                for (int jj = 0; jj < n; jj++)
                    for (int ii = 0; ii < n; ii++) s->Rt[ii + (size_t)jj * n] *= 2.0 / (s->lam[ii] + s->lam[jj]);
                nt_fromS(B, s, s->Rt, s->Q);
            }
            else if (pk[k]) { prod_pat(&hp[k], n, s, s->dZ, s->dX, s->Q); symmetrize(n, s->Q); }
            else { prod_zi(n, s, s->dZ, s->dX, s->Q); symmetrize(n, s->Q); }
            if (qsc != 1.0) { const size_t len = bsz(B); for (size_t i = 0; i < len; i++) s->Q[i] *= qsc; }
            blk_Aop(B, s, s->Q, qv, 1.0);
            /* <C^, Q> = (<Z, Q> + <Rd, Q>) / tau, <Z, Q> = qsc <dX, dZ> (the frames of Q are Z's) */
            if (shc.on) { const size_t len = bsz(B); pq[k] = (qsc * ddot_n(len, s->dX, s->dZ) + shift_dotF(&shc, s, k, len, s->Q)) / tau; }
            else pq[k] = sp_inner(B, &B->C, s->Q);
        });
        aop_end();
        g_pt[0] += wtime() - tpt0; g_pt[5] += 1;
        for (int k = 0; k < nb; k++) cQ += pq[k];
        if (nhoc > 0) {
            /* the fixed point is followed while it contracts: |Q_{k+1} - Q_k| <= theta |Q_k - Q_{k-1}| */
            double d = 0;
            for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]); const double *q = S[k].Q, *qp = Qp[k]; double t = 0;
                if (hoc == 0) t = len < 2 * (size_t)PN_CH ? ({ double u = 0; for (size_t i = 0; i < len; i++) u += q[i] * q[i]; u; }) : ddot_n(len, q, q);
                else t = dist2(len, q, qp);
                d += t; }
            d = sqrt(d + (hoc == 0 ? tk2 * tk2 : (tk2 - tk2p) * (tk2 - tk2p)));
            const int fail = hoc > 0 && !(d <= (hoc == 1 ? hoc_theta1 : hoc_theta) * hoc_d);
            if (ENV_ON("BRISK_HSDDBG") && hoc > 0) printf("     [second-order pass %d: change %.3e (previous %.3e, first %.3e)%s]\n", hoc, d, hoc_d, hoc_d0, fail ? " stop" : "");
            if (fail) {
                if (1) {
                    /* converged enough (or nothing was accepted): the last accepted state stands */
                    for (int k = 0; k < nb; k++) memcpy(S[k].Q, Qp[k], sizeof(double) * bsz(&P->blk[k]));
                    memcpy(qv, qvp, sizeof(double) * m); cQ = cQp; tk2 = tk2p;
                    R->t_dense += wtime() - td;
                    break;
                }
                for (int k = 0; k < nb; k++) memcpy(S[k].Q, Q0s[k], sizeof(double) * bsz(&P->blk[k]));
                memcpy(qv, qv0s, sizeof(double) * m); cQ = cQ0s; tk2 = tk20s;
                memcpy(u0, u0s, sizeof(double) * m); if (nf) memcpy(uf0, uf0s, sizeof(double) * nf);
                hoc_redo = 1;
            } else {
                if (hoc == 0) { hoc_d0 = d; for (int k = 0; k < nb; k++) memcpy(Q0s[k], S[k].Q, sizeof(double) * bsz(&P->blk[k])); memcpy(qv0s, qv, sizeof(double) * m); cQ0s = cQ; tk20s = tk2; }
                hoc_d = d;
                for (int k = 0; k < nb; k++) memcpy(Qp[k], S[k].Q, sizeof(double) * bsz(&P->blk[k]));
                memcpy(qvp, qv, sizeof(double) * m); cQp = cQ; tk2p = tk2;
            }
        }
        /* rhs(sigma) = c0 + sigma c1, so the whole corrector family is affine in sigma */
        for (int i = 0; i < m; i++) rhs[i] = rp[i] + hd[i] + qv[i];
        aop_begin(rhs, NULL, NULL, NULL);
        HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], S[k].X, rhs, 1.0););
        aop_end();
        if (nf) free_A(P, &FV, FV.xf, -1.0, rhs);
        R->t_dense += wtime() - td;
        tc = wtime();
        if (hoc_redo) { /* (u0 of the first pass is back in place) */ }
        else if (nf) free_solve(P, Sc, &FV, rhs, rf, u0, uf0);
        else if (!par->hsd_batch || hoc > 0) schur_solve(Sc, rhs, u0, soltol);
        if (!nf && par->hsd_batch && hoc == 0) {
            for (int i = 0; i < m; i++) dyd[i] = -(rp[i] + hd[i]) - mu * a0[i];
            const double *rr[2] = { rhs, dyd };
            double *xx[2] = { u0, u1 };
            const double tt[2] = { soltol, soltol };
            schur_solve_multi(Sc, 2, rr, xx, tt);
        }
        for (int i = 0; i < m; i++) rhs[i] = -(rp[i] + hd[i]) - mu * a0[i];
        if (hoc == 0) {                 /* (u1 does not depend on the second-order term) */
        if (nf) {
            for (int f2 = 0; f2 < nf; f2++) qf[f2] = -rf[f2];
            free_solve(P, Sc, &FV, rhs, qf, u1, uf1);
        }
        else if (!par->hsd_batch) schur_solve(Sc, rhs, u1, soltol);
        }
        R->t_chol += wtime() - tc;
        if (nhoc > 0 && hoc == 0) { memcpy(u0s, u0, sizeof(double) * m); if (nf) memcpy(uf0s, uf0, sizeof(double) * nf); }
        n0 = (shc.on ? rgs - cQ - cXs - hcRd - ddot_n(m, bmh, u0) : rg - cQ - (cX - cfx) - hcRd - ddot_n(m, P->b, u0) + ddot_n(m, hc, u0))
                    + (-tau * kap - tk2) / tau + (nf ? ddot_n(nf, FV.c, uf0) : 0.0);
        n1 = (shc.on ? -rgs + mu * cZi + hcRd - ddot_n(m, bmh, u1) : -rg + mu * cZi + hcRd - ddot_n(m, P->b, u1) + ddot_n(m, hc, u1)) + mu / tau
                    + (nf ? ddot_n(nf, FV.c, uf1) : 0.0);
        dt0 = shc.tauf * n0 / denom; dt1 = shc.tauf * (n0 + n1) / denom;
        dk0 = (-tau * kap - tk2) / tau - (kap / tau) * dt0;
        dk1 = (mu - tau * kap - tk2) / tau - (kap / tau) * dt1;

        /* endpoint directions at sigma = 0 and sigma = 1 */
        td = wtime();
        const double tpt1 = td;
        for (int e = 0; e < 2; e++) {
            double sg = e ? 1.0 : 0.0, dt = e ? dt1 : dt0;
            for (int i = 0; i < m; i++) dy[i] = u0[i] + sg * u1[i] + dt * v[i];
            HBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                int n = B->n;
                double *DZ = e ? dZ1[k] : dZ0[k], *DX = e ? dX1[k] : dX0[k];
                blk_ATy(B, s, dy, DZ);
                if (pk[k]) { for (size_t i = 0; i < len; i++) DZ[i] = -DZ[i]; add_R0(s, len, (1.0 - sg) * theta, DZ); }
                else for (size_t i = 0; i < len; i++) DZ[i] = (1.0 - sg) * s->F[i] - DZ[i];
                if (shc.on) shift_pre(&shc, s, k, len, dt, DZ);
                else sp_add(B, &B->C, dt, DZ);
                const double *RX = Zv[k] ? s->Zi : s->X, *ZI = Zv[k] ? Zv[k] : s->Zi;
                if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * DZ[i] * s->X[i];
                else if (B->type == BLK_SOC) soc_H(s, n, DZ, s->G, s->stmp);
                else if (pk[k]) { prod_pat(&hp[k], n, s, DZ, RX, s->G); symmetrize(n, s->G); }
                else { prod_zi(n, s, DZ, RX, s->G); symmetrize(n, s->G); }
                if (shc.on) shift_post(&shc, s, len, dt, 1.0, s->G, DZ);
                PAR_LEN_FOR(len, DX[i] = -s->G[i] - s->X[i] - s->Q[i] + sg * mu * ZI[i];);
            });
            if (nf) {
                for (int f2 = 0; f2 < nf; f2++) dxfd[f2] = uf0[f2] + sg * uf1[f2] + dt * vf[f2];
                free_set_dir(&FV, S, e ? dX1 : dX0, e ? dZ1 : dZ0, dxfd, 0);
            }
        }
        R->t_dense += wtime() - td;
        g_pt[1] += wtime() - tpt1;
        if (hoc_redo || nhoc == 0) break;
        {
            /* the predicted reduction of this pass's family (the sigma search): the passes are kept only if it
             * is at least the one of the first family; the fixed point is taken at the sigma the first search chose */
            double mer = -1, sgb = sig_m, alb = 0, amb = 0;
            const double tpt2 = wtime();
            HSD_SIGSEARCH(mer, sgb, alb, amb);
            g_pt[2] += wtime() - tpt2;
            if (ENV_ON("BRISK_HSDDBG")) printf("     [second-order pass %d: predicted reduction %.4f (sigma %.3g, step %.4f)]\n", hoc, mer, sgb, alb);
            hoc_kept = hoc;
            if (hoc == 0) { hoc_m0 = mer; hoc_s0 = sgb; hoc_al0 = alb; hoc_am0 = amb; hoc_sig = getenv("BRISK_HSDHOCSIGM") ? sig_m : sgb; }
            if (hoc == 0 || mer >= hoc_m0) { hoc_ms = mer; hoc_ss = sgb; hoc_als = alb; hoc_ams = amb; hoc_have = 1; }
            else {
                /* worse than the first family: back to it */
                for (int k = 0; k < nb; k++) memcpy(S[k].Q, Q0s[k], sizeof(double) * bsz(&P->blk[k]));
                memcpy(qv, qv0s, sizeof(double) * m); cQ = cQ0s; tk2 = tk20s;
                memcpy(u0, u0s, sizeof(double) * m); if (nf) memcpy(uf0, uf0s, sizeof(double) * nf);
                hoc_redo = 2; hoc_kept = 0; hoc_nfail++;
                hoc_ms = hoc_m0; hoc_ss = hoc_s0; hoc_als = hoc_al0; hoc_ams = hoc_am0;
            }
            if (hoc_redo != 2) {
                if (hoc >= nhoc) break;
                const double sg = hoc_sig;
                HBLK_FOR(k, nb, {
                    const size_t len = bsz(&P->blk[k]);
                    PAR_LEN_FOR(len, S[k].dX[i] = dX0[k][i] + sg * (dX1[k][i] - dX0[k][i]);
                        S[k].dZ[i] = dZ0[k][i] + sg * (dZ1[k][i] - dZ0[k][i]););
                });
                dtau_a = dt0 + sg * (dt1 - dt0); dkap_a = dk0 + sg * (dk1 - dk0);
            }
        }
        if (hoc_redo == 2) {
            /* recompute the scalars and the endpoints from the restored u0 */
            hoc_redo = 1;
            n0 = (shc.on ? rgs - cQ - cXs - hcRd - ddot_n(m, bmh, u0) : rg - cQ - (cX - cfx) - hcRd - ddot_n(m, P->b, u0) + ddot_n(m, hc, u0)) + (-tau * kap - tk2) / tau + (nf ? ddot_n(nf, FV.c, uf0) : 0.0);
            dt0 = shc.tauf * n0 / denom; dt1 = shc.tauf * (n0 + n1) / denom;
            dk0 = (-tau * kap - tk2) / tau - (kap / tau) * dt0;
            dk1 = (mu - tau * kap - tk2) / tau - (kap / tau) * dt1;
            for (int e = 0; e < 2; e++) {
                double sg = e ? 1.0 : 0.0, dt = e ? dt1 : dt0;
                for (int i = 0; i < m; i++) dy[i] = u0[i] + sg * u1[i] + dt * v[i];
                HBLK_FOR(k, nb, {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    size_t len = bsz(B);
                    int n = B->n;
                    double *DZ = e ? dZ1[k] : dZ0[k], *DX = e ? dX1[k] : dX0[k];
                    blk_ATy(B, s, dy, DZ);
                    if (pk[k]) { for (size_t i = 0; i < len; i++) DZ[i] = -DZ[i]; add_R0(s, len, (1.0 - sg) * theta, DZ); }
                    else for (size_t i = 0; i < len; i++) DZ[i] = (1.0 - sg) * s->F[i] - DZ[i];
                    if (shc.on) shift_pre(&shc, s, k, len, dt, DZ);
                    else sp_add(B, &B->C, dt, DZ);
                    const double *RX = Zv[k] ? s->Zi : s->X, *ZI = Zv[k] ? Zv[k] : s->Zi;
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * DZ[i] * s->X[i];
                    else if (B->type == BLK_SOC) soc_H(s, n, DZ, s->G, s->stmp);
                    else if (pk[k]) { prod_pat(&hp[k], n, s, DZ, RX, s->G); symmetrize(n, s->G); }
                    else { prod_zi(n, s, DZ, RX, s->G); symmetrize(n, s->G); }
                    if (shc.on) shift_post(&shc, s, len, dt, 1.0, s->G, DZ);
                    PAR_LEN_FOR(len, DX[i] = -s->G[i] - s->X[i] - s->Q[i] + sg * mu * ZI[i];);
                });
                if (nf) {
                    for (int f2 = 0; f2 < nf; f2++) dxfd[f2] = uf0[f2] + sg * uf1[f2] + dt * vf[f2];
                    free_set_dir(&FV, S, e ? dX1 : dX0, e ? dZ1 : dZ0, dxfd, 0);
                }
            }
            break;
        }
        }
        if (Qp) { for (int k = 0; k < nb; k++) { free(Qp[k]); free(Q0s[k]); } free(Qp); free(Q0s); free(qvp); free(qv0s); free(u0s); free(uf0s); }

        { double _t = wtime(); g_ph[3] += _t - g_ph_last; g_ph_last = _t; }
        /* ---- choose sigma: the embedded system gives mu_new = mu (1 - alpha (1 - sigma))
         * exactly, so maximize alpha(sigma) (1 - sigma) over a few candidates.        */
        double best_merit = -1, best_sig = sig_m, best_alpha = 0, best_am = 0;
        const double tsig0 = wtime();
        if (hoc_kept > 0) hoc_nkept++;
        if (hoc_have) { best_merit = hoc_ms; best_sig = hoc_ss; best_alpha = hoc_als; best_am = hoc_ams; }   /* (the search of the family that stands was made in the passes) */
        else HSD_SIGSEARCH(best_merit, best_sig, best_alpha, best_am);
        g_ph[8] += wtime() - tsig0;
        sigma = best_sig;
        alpha = best_alpha;
        {
            double dt = dt0 + sigma * (dt1 - dt0), dk = dk0 + sigma * (dk1 - dk0);
            HBLK_FOR(k, nb, {
                size_t len = bsz(&P->blk[k]);
                PAR_LEN_FOR(len, S[k].dX[i] = dX0[k][i] + sigma * (dX1[k][i] - dX0[k][i]);
                    S[k].dZ[i] = dZ0[k][i] + sigma * (dZ1[k][i] - dZ0[k][i]););
            });
            for (int i = 0; i < m; i++) dyd[i] = u0[i] + sigma * u1[i] + dt * v[i];
            /* Gondzio centrality correctors (the endpoint buffers are free now) */
            if (par->hsd_corr > 0 && best_am < 1.0) {
                int nacc = 0;
                double tcorr = 0;
                for (int c = 0; c < par->hsd_corr && best_am < 1.0; c++) {
                    /* cost control: correctors pay off when they are cheap against the
                     * assembly and factorization of M; after the first one, continue only
                     * while their total stays within hsd_cfrac of that cost             */
                    if (kfix != 0) { if (c >= (kfix > 0 ? kfix : 0)) break; }
                    else if (c >= g_prek) break;     /* until the count is fixed: a fixed limit, not timings */
                    (void)t_fact_it;
                    /* ... and a corrector must lengthen the step by at least hsd_cgain times
                     * its share of an iteration (progress per iteration ~ alpha)         */
                    const double tc_est = c > 0 ? tcorr / c : tc_avg;
                    const double accf = (par->hsd_cgain > 0 && t_iter_prev > 0 && tc_est > 0)
                                        ? 1.0 + par->hsd_cgain * tc_est / t_iter_prev : 0.0;
                    double mut = par->hsd_ctarget ? sigma * mu : 0.0;
                    const double tc0 = wtime();
                    const int acc = hsd_corrector(P, par, S, Sc, Zv, hp, pk, tau, kap, &dt, &dk, dyd, v, hc, denom,
                                                  &best_am, mut, dX0, dZ0, rhs, u1, ndim1, R, &FV, vf, accf, &shc);
                    tcorr += wtime() - tc0;
                    g_ph[9] += wtime() - tc0;
                    ntry_c++;
                    tc_sum += wtime() - tc0;
                    tc_avg = tc_sum / ntry_c;
                    if (!acc) break;
                    nacc++;
                }
                if (ENV_ON("BRISK_HSDDBG")) printf("     [correctors: %d accepted, %.4fs (factor+assembly %.4fs)]\n", nacc, tcorr, t_fact_it);
                /* freeze the corrector count after three iterations from the measured cost
                 * ratio (decisions on timings every iteration made the run irreproducible:
                 * 28 to 33 iterations on the same input) */
                if (kfix == 0 && ntry_c > 0) {
                    kf_fact += t_fact_it; kf_n++;
                    if (kf_n >= 3) {
                        /* 4.25: the ratio from a deterministic work model, not from timings: the
                         * measured ratio made the corrector count - and the whole run - depend on
                         * the machine load and the thread count (attr_henon_d6_I diverged at T = 2
                         * next to a compile). Model: per-iteration factorization + assembly against
                         * one corrector, least-squares fit of the measured times on 135 instances
                         * (median error 40%); the thresholds are set where the corrector counts
                         * tested best (buck3, trto4: none; mater-4: 3; sep30000: 2). */
                        double f_chol = 0, f_nnz = 0, f_asm = 0, f_dense = 0, f_small = 0, f_aop = 0;
                        if (Sc->SC) { f_chol = schol_flops(Sc->SC); f_nnz = (double)schol_nnz(Sc->SC); }
                        else if (Sc->E) { for (int i = 0; i < m; i++) { double r = i - Sc->E->fst[i] + 1; f_chol += r * r; f_nnz += r; } }
                        else { f_chol = (double)m * m * m / 3.0; f_nnz = 0.5 * (double)m * m; }
                        for (int k = 0; k < nb; k++) {
                            const Block *B = &P->blk[k];
                            const double nk = B->n;
                            if (B->type == BLK_SDP) { if (nk > 16) f_dense += nk * nk * nk; else f_small += nk * nk * nk + 64.0; }
                            for (int t = 0; t < B->ncon; t++) { f_aop += B->A[t].ef; f_asm += 2.0 * B->A[t].ef * (B->type == BLK_SDP ? nk : 1.0); }
                        }
                        const double w_fact = (Sc->allow_float ? 7.0e-12 : 2.0e-11) * f_chol + 9.2e-11 * f_asm + 3.9e-8 * f_nnz + 1.6e-6 * nb + 6.0e-5;
                        const double w_corr = 9.9e-10 * f_dense + 7.8e-9 * f_small + 3.6e-9 * f_nnz + 2.7e-9 * f_aop + 1.05e-6 * nb + 3.4e-7 * m + 6.9e-5;
                        const double rho_t = par->hsd_cfrac * (kf_fact / kf_n) / fmax(tc_avg, 1e-12);
                        const double rho_m = par->hsd_cfrac * w_fact / w_corr;
                        const int timed = ENV_ON("BRISK_KFTIME");      /* the 4.20-4.24 rule */
                        const double rho = timed ? rho_t : rho_m;
                        if (timed) kfix = (int)fmin((double)par->hsd_corr, rho >= 1.2 ? 3.0 : rho >= 0.6 ? 2.0 : rho >= g_kfix0 ? 1.0 : -1.0);
                        else kfix = (int)fmin((double)par->hsd_corr, rho >= 0.9 ? 3.0 : rho >= 0.45 ? 2.0 : rho >= 0.2 ? 1.0 : -1.0);
                        /* tiny problems: correctors are cheap in absolute terms (and their timings noise) */
                        if (timed ? kf_fact / kf_n < 2e-3 : w_fact < 2e-3) kfix = par->hsd_corr;
                        if (ENV_ON("BRISK_KFDBG"))
                            printf("   KFDBG rho_time %.4f fact %.4e corr %.4e model %.4f (%.3e / %.3e) chol %.4e nnz %.4e asm %.4e dense %.4e small %.4e aop %.4e m %d nb %d float %d kfix %d\n",
                                   rho_t / par->hsd_cfrac, kf_fact / kf_n, tc_avg, rho_m / par->hsd_cfrac, w_fact, w_corr, f_chol, f_nnz, f_asm, f_dense, f_small, f_aop, m, nb, Sc->allow_float, kfix);
                        { const char *e = getenv("BRISK_KFIX"); if (e) kfix = atoi(e); }   /* rounded: the AC-OPF ratios sit near 2 */
                        if (par->verbose > 1) printf("   HSD: corrector count fixed at %d (factor/corrector cost %.2f)\n", kfix, rho / par->hsd_cfrac);
                    }
                }
                if (nacc) {
                    double gam = g_hsdg0 + (par->gamma_max - g_hsdg0) * fmin(1.0, best_am);
                    alpha = fmin(1.0, gam * best_am);
                    ncorr += nacc;
                }
            }
            /* refinement: enforce A(dX) - b dtau = (1 - sigma) rp. Near the end the rhs is a
             * sum of O(1/mu) terms that cancel, and A(dX) can miss by far more than the
             * reduction asked for. The correction is the embedded Newton step for that
             * primal residual alone (zero complementarity and dual rhs):
             *   M w = e, dtau_e = -(b - hc)'w / denom, dy_e = w + dtau_e v,
             *   dZ_e = C dtau_e - A'dy_e, dX_e = -sym(Zi dZ_e X), dkap_e = -kap dtau_e / tau */
            { double _t = wtime(); g_ph[5] += _t - g_ph_last; g_ph_last = _t; }
            for (int pass = 0; pass < par->hsd_refine; pass++) {
                td = wtime();
                double *e = rhs;
                for (int i = 0; i < m; i++) e[i] = (1.0 - sigma) * rp[i] + P->b[i] * dt;
                aop_begin(e, NULL, NULL, NULL);
                HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], S[k].dX, e, -1.0););
                aop_end();
                double ne = sqrt(ddot_n(m, e, e)), nrp = sqrt(ddot_n(m, rp, rp));
                R->t_dense += wtime() - td;
                /* the direction misses the primal residual reduction by more than it asks:
                 * M has lost the accuracy (split-pair weights), bordered from the next
                 * factorization on (the M path is faster while it is accurate) */
                if (getenv("BRISK_BORDDBG") && pass == 0 && bord && !bord_on) printf("     [bord check it %d: miss %.2e of %.2e]\n", it, ne, (1.0 - sigma) * nrp);
                if (pass == 0 && bord && !bord_on && it >= g_bord_minit && ne > 0.3 * (1.0 - sigma) * nrp && ne > 0) {
                    bord_next = 1;
                    if (par->verbose > 1) printf("   HSD: bordered Schur factorization from iteration %d (direction misses %.1e of %.1e)\n", it + 1, ne, (1.0 - sigma) * nrp);
                }
                if (!(ne > 1e-2 * (1.0 - sigma) * nrp && ne > 0)) break;
                /* 4.27: the miss is at the level where the last correction solve could not
                 * resolve it (M w = e met less than half of e): a new solve would not either */
                if (refskip_on == 2 && !nf && ref_floor > 0 && ne <= 2.0 * ref_floor) { nref_skip++; break; }
                tc = wtime();
                if (nf) {
                    /* free rows too: (1 - sigma) r_f + c_f dtau - Af' dy */
                    free_AT(P, &FV, dyd, qf);
                    for (int f2 = 0; f2 < nf; f2++) qf[f2] = (1.0 - sigma) * rf[f2] + FV.c[f2] * dt - qf[f2];
                    free_solve(P, Sc, &FV, e, qf, u1, dxfd);
                } else {
                    const double rel_e = schur_solve(Sc, e, u1, 0);
                    { static int dumped = 0; const char *df = getenv("BRISK_HSDDUMP"); if (df && !dumped && rel_e > 0.5) { dumped = 1; hsd_dump_floor(df, P, S, m, e, u1, it); } }
                    /* 4.27: the correction resolves at most (1 - rel_e) of the miss, and the pass
                     * is kept only if it halves it: past rel_e 0.5 it would be rejected anyway
                     * (the image-form endgames: |e| ~1e-9 below the absolute accuracy of the
                     * Schur solve, rel_e 0.1-1, a solve and two block products per iteration) */
                    if (refskip_on && rel_e > 0.5) { ref_floor = rel_e * ne; R->t_chol += wtime() - tc; nref_fail++; break; }
                    if (refskip_on) ref_floor = 0;
                }
                R->t_chol += wtime() - tc;
                td = wtime();
                const double dte = -shc.tauf * ((shc.on ? ddot_n(m, bmh, u1) : ddot_n(m, P->b, u1) - ddot_n(m, hc, u1))
                                     - (nf ? ddot_n(nf, FV.c, dxfd) : 0.0)) / denom;
                for (int i = 0; i < m; i++) u1[i] += dte * v[i];
                for (int i = 0; i < m; i++) qv[i] = (1.0 - sigma) * rp[i] + P->b[i] * (dt + dte);
                HBLK_FOR(k, nb, {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    const int n = B->n;
                    const size_t len = bsz(B);
                    blk_ATy(B, s, u1, s->F);
                    PAR_LEN_FOR(len, s->F[i] = -s->F[i];);
                    if (shc.on) shift_pre(&shc, s, k, len, dte, s->F);        /* dZ_e */
                    else sp_add(B, &B->C, dte, s->F);
                    const double *RX = Zv[k] ? s->Zi : s->X;
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = -s->Zi[i] * s->F[i] * s->X[i];
                    else if (B->type == BLK_SOC) { soc_H(s, n, s->F, s->G, s->stmp); PAR_LEN_FOR(len, s->G[i] = -s->G[i];); }
                    else {
                        if (pk[k]) prod_pat(&hp[k], n, s, s->F, RX, s->G);
                        else prod_zi(n, s, s->F, RX, s->G);
                        symmetrize(n, s->G);
                        PAR_LEN_FOR(len, s->G[i] = -s->G[i];);
                    }
                    if (shc.on) shift_post(&shc, s, len, dte, -1.0, s->G, s->F);   /* G = -H(dZ_e) */
                    PAR_LEN_FOR(len, dX0[k][i] = s->dX[i] + s->G[i];);
                });
                if (nf) {
                    for (int f2 = 0; f2 < nf; f2++) dxfd[f2] += dte * vf[f2];
                    free_set_dir(&FV, S, dX0, NULL, dxfd, 1);    /* dX0 = dX + dX_e (also zeroes the masked dZ slots) */
                    for (int f2 = 0; f2 < nf; f2++) { S[FV.kb[f2]].F[FV.ip[f2]] = 0; S[FV.kb[f2]].F[FV.im[f2]] = 0; }
                }
                aop_begin(qv, NULL, NULL, NULL);
                HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], dX0[k], qv, -1.0););
                aop_end();
                double ne2 = sqrt(ddot_n(m, qv, qv));
                R->t_dense += wtime() - td;
                if (ENV_ON("BRISK_HSDDBG"))
                    printf("     [refine %d: |e| %.2e -> %.2e (|rp| %.2e)]\n", pass, ne, ne2, nrp);
                if (!(ne2 < 0.5 * ne)) { if (refskip_on && !nf) { ref_floor = ne2; nref_fail++; } break; }
                HBLK_FOR(k, nb, {
                    const size_t len = bsz(&P->blk[k]);
                    memcpy(S[k].dX, dX0[k], sizeof(double) * len);
                    PAR_LEN_FOR(len, S[k].dZ[i] += S[k].F[i];);
                });
                for (int i = 0; i < m; i++) dyd[i] += u1[i];
                dt += dte;
                dk -= kap * dte / tau;
                nrefine++;
            }
            if (rp0 && !nf && rg0 != 0) {
                /* the normalization row's value at the direction and its slope in dtau (dZ_v = C^ - A'v, dX_v = -H(dZ_v)) */
                double f0 = -rg0 * dt, fs = -rg0;
                for (int i = 0; i < m; i++) { const double dyi = dyd[i] + (shc.on ? dt * y[i] / tau : 0.0), vi = v[i] + (shc.on ? y[i] / tau : 0.0); f0 -= rp0[i] * dyi; fs -= rp0[i] * vi; }
                double *p2 = hred(nb, 2);
                HBLK_FOR(k, nb, {
                    const Block *B = &P->blk[k]; BS *s = &S[k]; const int n = B->n; const size_t len = bsz(B);
                    double *dZv = malloc(sizeof(double) * len), *dXv = malloc(sizeof(double) * len);
                    blk_ATy(B, s, v, dZv); PAR_LEN_FOR(len, dZv[i] = -dZv[i];);
                    if (shc.on) shift_addC(&shc, s, k, len, 1.0, dZv); else sp_add(B, &B->C, 1.0, dZv);
                    const double *RX = Zv[k] ? s->Zi : s->X;
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) dXv[i] = -s->Zi[i] * dZv[i] * s->X[i];
                    else if (B->type == BLK_SOC) { soc_H(s, n, dZv, dXv, s->stmp); PAR_LEN_FOR(len, dXv[i] = -dXv[i];); }
                    else if (pk[k]) { prod_pat(&hp[k], n, s, dZv, RX, dXv); symmetrize(n, dXv); PAR_LEN_FOR(len, dXv[i] = -dXv[i];); }
                    else { prod_zi(n, s, dZv, RX, dXv); symmetrize(n, dXv); PAR_LEN_FOR(len, dXv[i] = -dXv[i];); }
                    p2[2 * (size_t)k] = dot_R0(s, len, s->dX); p2[2 * (size_t)k + 1] = dot_R0(s, len, dXv);
                    memcpy(s->F, dZv, sizeof(double) * len); memcpy(s->G, dXv, sizeof(double) * len);   /* kept for the update */
                    free(dZv); free(dXv);
                });
                for (int k = 0; k < nb; k++) { f0 += p2[2 * (size_t)k]; fs += p2[2 * (size_t)k + 1]; }
                if (fs != 0 && isfinite(f0 / fs)) {
                    const double delta = -f0 / fs;
                    if (ENV_ON("BRISK_HSDDBG")) printf("     [ytm it %d: dtau %.4e -> %.4e (row value %.2e, slope %.2e)]\n", it, dt, dt + delta, f0, fs);
                    for (int i = 0; i < m; i++) dyd[i] += delta * v[i];
                    HBLK_FOR(k, nb, { const size_t len = bsz(&P->blk[k]); PAR_LEN_FOR(len, S[k].dZ[i] += delta * S[k].F[i]; S[k].dX[i] += delta * S[k].G[i];); });
                    dt += delta; dk -= kap * delta / tau; n_ytm++;
                }
            }
            /* back from the shifted variable: dy = dy' + (dtau / tau) y */
            if (shc.on) for (int i = 0; i < m; i++) dyd[i] += dt * y[i] / tau;
            if (shift_env == 1 && !nf) {
                /* which form of the tau column for the next iteration: the error each puts into the gap row.
                 * Original: its scalars cancel terms of size <C, H C>, an error of eps <C, H C> |dtau|.
                 * Shifted: the gap row holds in dy', and in dy up to y'e / tau, e = A(dX) - b dtau - eta rp the
                 * primal row's miss (the shifted row uses A(dX) through C^ = C - A'y / tau). */
                double *e = bmh + m;            /* (the second half of the bmh buffer) */
                for (int i = 0; i < m; i++) e[i] = (1.0 - sigma) * rp[i] + P->b[i] * dt;
                aop_begin(e, NULL, NULL, NULL);
                HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], S[k].dX, e, -1.0););
                aop_end();
                const double err1 = fabs(ddot_n(m, y, e)) / tau, err0 = DBL_EPSILON * hcC0 * fabs(dt);
                if (ENV_ON("BRISK_HSDDBG")) { double ne_ = sqrt(ddot_n(m, e, e)), nt_ = 0; for (int i = 0; i < m; i++) nt_ += rp[i] * rp[i];
                    printf("     [final direction it %d: primal miss %.2e of the target %.2e (sigma %.3f)]\n", it, ne_, (1.0 - sigma) * sqrt(nt_), sigma); }
                shift_next = !shift_off && err1 < err0;
                if (ENV_ON("BRISK_SHIFTDBG")) printf("     [tau column it %d: %s; next %s (original %.2e, shifted %.2e)]\n", it,
                                                    shc.on ? "shifted" : "original", shift_next ? "shifted" : "original", err0, err1);
            }
            inex_have = 0;
            if (inex_beta > 0 && !nf) {
                /* the final direction's primal miss e = (1 - sigma) rp + b dtau - A(dX): after a step alpha the primal residual
                 * is (1 - alpha (1 - sigma)) rp + alpha e, whatever form or solve produced the direction */
                double *e = bmh + m;            /* (the second half of the bmh buffer: free after the form's choice) */
                for (int i = 0; i < m; i++) e[i] = (1.0 - sigma) * rp[i] + P->b[i] * dt;
                aop_begin(e, NULL, NULL, NULL);
                HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], S[k].dX, e, -1.0););
                aop_end();
                static int proj_on = -1; if (proj_on < 0) { const char *pe = getenv("BRISK_HSDPROJ"); proj_on = pe ? atoi(pe) : 0; }
                if (proj_on) {
                    /* experiment (BRISK_HSDPROJ=1): the primal rows restored exactly by the least-norm correction
                     * dX += A'(w), (A A') w = e. The Gram matrix is constant and well conditioned, so the correction does
                     * not depend on the Schur solve's accuracy; the complementarity row absorbs the change (that is
                     * where an inexact direction's error belongs, Bellavia 1998) */
                    double ne0 = sqrt(ddot_n(m, e, e));
                    if (ne0 > 0) {
                        double *w = malloc(sizeof(double) * (m + 1));
                        gram_cg(P, S, e, w, 1000, 1e-14, Sc->r, Sc->p, Sc->q);
                        HBLK_FOR(k, nb, { const Block *B = &P->blk[k]; BS *s = &S[k]; const size_t len = bsz(B);
                            blk_ATy(B, s, w, s->F); PAR_LEN_FOR(len, s->dX[i] += s->F[i];); });
                        for (int i = 0; i < m; i++) e[i] = (1.0 - sigma) * rp[i] + P->b[i] * dt;
                        aop_begin(e, NULL, NULL, NULL);
                        HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], S[k].dX, e, -1.0););
                        aop_end();
                        if (ENV_ON("BRISK_HSDDBG")) printf("     [projection it %d: primal miss %.2e -> %.2e]\n", it, ne0, sqrt(ddot_n(m, e, e)));
                        free(w); n_proj++;
                    }
                }
                static int gal_on = -1; if (gal_on < 0) { const char *ge = getenv("BRISK_HSDGAL"); gal_on = ge ? atoi(ge) : 0; }
                if (gal_on) {
                    /* experiment (BRISK_HSDGAL=1): the primal miss e enters the gap as y'e (b'y - <C,X> = y'r_p - <X,Z> + <Rd,X>),
                     * and y'e is what the ill-posed files cannot close (|y| up to 1e7 against e at the solve floor). One
                     * Galerkin step in the direction the multipliers weight: dX += t H(A'y) with t = y'e / y'(M y), so that
                     * y'A(dX) meets its target exactly; the step lies in the scaling's metric like the direction itself,
                     * and y'My is the dominant quadratic form of M, no small pivot is touched. One block product. */
                    double ye = ddot_n(m, y, e);
                    if (ye != 0) {
                        double *g = malloc(sizeof(double) * (m + 1)); memset(g, 0, sizeof(double) * m);
                        aop_begin(g, NULL, NULL, NULL);
                        const double ne_before = sqrt(ddot_n(m, e, e));
                        HBLK_FOR(k, nb, { const Block *B = &P->blk[k]; BS *s = &S[k]; const int n = B->n;
                            blk_ATy(B, s, y, s->F);
                            const double *RX = Zv[k] ? s->Zi : s->X;
                            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * s->F[i] * s->X[i];
                            else if (B->type == BLK_SOC) soc_H(s, n, s->F, s->G, s->stmp);
                            else if (pk[k]) { prod_pat(&hp[k], n, s, s->F, RX, s->G); symmetrize(n, s->G); }
                            else { prod_zi(n, s, s->F, RX, s->G); symmetrize(n, s->G); }
                            blk_Aop(B, s, s->G, g, 1.0); });
                        aop_end();
                        const double yg = ddot_n(m, y, g), t = yg != 0 ? ye / yg : 0.0;
                        if (t != 0 && isfinite(t)) {
                            HBLK_FOR(k, nb, { const size_t len = bsz(&P->blk[k]); PAR_LEN_FOR(len, S[k].dX[i] += t * S[k].G[i];); });
                            for (int i = 0; i < m; i++) e[i] -= t * g[i];
                            if (ENV_ON("BRISK_HSDDBG")) printf("     [galerkin it %d: y'e %.2e -> %.2e, t %.2e, |e| %.2e -> %.2e]\n", it, ye, ddot_n(m, y, e), t, ne_before, sqrt(ddot_n(m, e, e)));
                            n_gal++;
                        }
                        free(g);
                    }
                }
                inex_R2 = inex_RE = inex_E2 = 0;
                for (int i = 0; i < m; i++) { const double r_ = rp[i] * P->du[i], e_ = e[i] * P->du[i]; inex_R2 += r_ * r_; inex_RE += r_ * e_; inex_E2 += e_ * e_; }
                inex_sig = sigma; inex_have = 1;
                /* experiment (BRISK_HSDMISSSTOP=f): the direction misses its primal target by more than f times in two
                 * consecutive iterations near the end: the Schur solve has reached its floor and no form of the direction
                 * is a Newton direction any more; stop at the best iterate (the double-double endgame follows) */
                static double mstop = -1; if (mstop < 0) { const char *me = getenv("BRISK_HSDMISSSTOP"); mstop = me ? atof(me) : 0.0; }
                if (mstop > 0) {
                    const double tgt = (1.0 - sigma) * sqrt(inex_R2), ms = sqrt(inex_E2);
                    n_miss = (ms > mstop * tgt && ms > 0.1 * par->tol * tau * (1 + P->normb2)) ? n_miss + 1 : 0;
                    static int mcount = -1; if (mcount < 0) { const char *mc = getenv("BRISK_HSDMISSN"); mcount = mc ? atoi(mc) : 1; }
                    if (n_miss >= mcount && *best_score < 1e-5) { status = 4; g_hsd_why = "the directions miss the primal rows at the solve floor"; if (par->verbose > 1) printf("   HSD: the direction misses its primal target by %.1e (%.1e of %.1e), stopping\n", ms / fmax(tgt, 1e-300), ms, tgt); break; }
                }
            }
            {   /* experimental: a longer step when the new point stays in the neighbourhood */
                static double nb_beta = -2, nb_g = 0; static int nb_always = 0;
                static int nb_mode = 1;      /* 1 (default): only in iterations whose passes were kept; 0: whenever the passes are allowed */
                if (nb_beta < -1) { nb_always = getenv("BRISK_HSDNBALL") != NULL; const char *em = getenv("BRISK_HSDNBMODE"); if (em) nb_mode = atoi(em); }
                static double nb_env = -2, nb_amin = 0.9;
                if (nb_env < -1.5) { const char *e3 = getenv("BRISK_HSDNBAMIN"); if (e3) nb_amin = atof(e3); }
                if (nb_env < -1.5) { const char *e1 = getenv("BRISK_HSDNB"); nb_env = e1 ? atof(e1) : -1; const char *e2 = getenv("BRISK_HSDNBG"); nb_g = e2 ? atof(e2) : 0.9995; }
                nb_beta = nb_env >= 0 ? nb_env : par->hsd_nb;
                if (nb_beta > 0 && best_am >= nb_amin && ((nb_mode == 0 && hoc_on) || (nb_mode == 1 && hoc_kept > 0) || nb_always)) {
                    const double gs[4] = { nb_g, 1.0 - 4.0 * (1.0 - nb_g), 1.0 - 16.0 * (1.0 - nb_g), 0 };
                    for (int c = 0; c < 3; c++) {
                        const double al = fmin(1.0, gs[c] * best_am);
                        if (al <= alpha) break;
                        if ((tau + al * dt) <= 0 || (kap + al * dk) <= 0) continue;
                        const double mun = mu * (1.0 - al * (1.0 - sigma));
                        const int okn = (tau + al * dt) * (kap + al * dk) >= nb_beta * mun && sdp_nb_ok(P, S, al, nb_beta * mun);
                        if (ENV_ON("BRISK_HSDDBG")) printf("     [neighbourhood: alpha %.5f (default %.5f): %s]\n", al, alpha, okn ? "inside" : "outside");
                        if (okn) { alpha = al; break; }
                    }
                }
            }
            /* neighbourhood safeguard: keep tau kappa and the LP products near mu */
            for (int t2 = 0; t2 < 10; t2++) {
                double munew = 0, tk = (tau + alpha * dt) * (kap + alpha * dk), lo = 1e300;
                {
                    double *pm2 = hred(nb, 1);
                    HBLK_FOR(k, nb, {
                        size_t len = bsz(&P->blk[k]);
                        pm2[k] = dot_step(len, S[k].X, S[k].dX, S[k].Z, S[k].dZ, alpha);
                    });
                    for (int k = 0; k < nb; k++) munew += pm2[k];
                }
                munew = (munew + tk) / ndim1;
                lo = tk;
                for (int k = 0; k < nb; k++) {
                    if (P->blk[k].type == BLK_SOC) {
                        const double w = soc_lmin_at(P->blk[k].n, S[k].X, S[k].dX, S[k].Z, S[k].dZ, alpha, S[k].stmp);
                        if (w < lo) lo = w;
                        continue;
                    }
                    if (P->blk[k].type != BLK_LP) continue;
                    for (int i = 0; i < P->blk[k].n; i++) {
                        if (S[k].fm && S[k].fm[i]) continue;
                        double w = (S[k].X[i] + alpha * S[k].dX[i]) * (S[k].Z[i] + alpha * S[k].dZ[i]);
                        if (w < lo) lo = w;
                    }
                }
                if (!(munew > 0) || lo >= par->hsd_beta * munew || alpha < 1e-3) break;
                alpha *= 0.7;
                nsafe++;
            }
            if (getenv("BRISK_HSDCHK")) {
                /* residuals of the embedded Newton system at the chosen sigma */
                double *t1 = malloc(sizeof(double) * m);
                for (int i = 0; i < m; i++) t1[i] = -(1.0 - sigma) * rp[i] - P->b[i] * dt;
                double cdX = 0;
                for (int k = 0; k < nb; k++) {
                    blk_Aop(&P->blk[k], &S[k], S[k].dX, t1, 1.0);
                    cdX += sp_inner(&P->blk[k], &P->blk[k].C, S[k].dX);
                }
                double r1 = sqrt(ddot_n(m, t1, t1)), nrp = sqrt(ddot_n(m, rp, rp));
                double ddy = 0;
                for (int i = 0; i < m; i++) ddy += P->b[i] * dyd[i];
                double r3 = ddy - cdX - dk - (1.0 - sigma) * rg;
                printf("     [hsd check: |A(dX) - b dtau - eta rp| %.2e (|rp| %.2e), row3 %.2e (rg %.2e)]\n",
                       r1, nrp, r3, rg);
                free(t1);
            }
            /* 5.12: the terminal step. Near the end the default step (gamma times the step to the
             * boundary) leaves the complementarity at about (1 - alpha) mu, which can stay just
             * above the tolerance; the run then paid for an X-metric polish (two Schur assemblies
             * and double-precision factorizations) or for one more iteration. The measures after
             * a step a are known in advance: the primal residual rp + a q with q = b dtau - A(dX)
             * (exact), the dual residual (1 - a (1 - sigma)) times the current one, the objectives
             * and <X,Z> (linear and quadratic in a). When a longer step, up to tstep_g of the way
             * to the boundary, is predicted to meet tstep_t times the tolerance (0.9: the predicted
             * measures matched the computed ones to two digits), the shortest such step of
             * a few is taken; the residual check of the next iteration then stops the run (the
             * point is less centred, which no later iteration sees). BRISK_TSTEP=0 off. */
            {
                static int tstep_on = -1; static double tstep_g = 0.9999, tstep_t = 0.9;
                if (tstep_on < 0) { const char *e = getenv("BRISK_TSTEP"); tstep_on = e ? atoi(e) : 1; const char *e2 = getenv("BRISK_TSTEPG"); if (e2) tstep_g = atof(e2);
                                    const char *e3 = getenv("BRISK_TSTEPT"); if (e3) tstep_t = atof(e3); }
                const double amax = fmin(1.0, tstep_g * best_am);
                if (tstep_on && !nf && !npk && !bord && amax > alpha && score <= 1e3 * par->tol) {
                    td = wtime();
                    double *q = rhs;                                    /* q = b dtau - A(dX) */
                    for (int i = 0; i < m; i++) q[i] = P->b[i] * dt;
                    aop_begin(q, NULL, NULL, NULL);
                    HBLK_FOR(k, nb, blk_Aop(&P->blk[k], &S[k], S[k].dX, q, -1.0););
                    aop_end();
                    double r00 = 0, r01 = 0, r11 = 0;
                    for (int i = 0; i < m; i++) { const double w0 = rp[i] * P->du[i], w1 = q[i] * P->du[i]; r00 += w0 * w0; r01 += w0 * w1; r11 += w1 * w1; }
                    double cdX = 0, x01 = 0, x11 = 0;
                    const double bdy = ddot_n(m, P->b, dyd);
                    double *p3 = hred(nb, 3);
                    HBLK_FOR(k, nb, {
                        const Block *B = &P->blk[k];
                        const size_t len = bsz(B);
                        p3[3 * (size_t)k] = sp_inner(B, &B->C, S[k].dX);
                        p3[3 * (size_t)k + 1] = ddot_n(len, S[k].X, S[k].dZ) + ddot_n(len, S[k].dX, S[k].Z);
                        p3[3 * (size_t)k + 2] = ddot_n(len, S[k].dX, S[k].dZ);
                    });
                    for (int k = 0; k < nb; k++) { cdX += p3[3 * (size_t)k]; x01 += p3[3 * (size_t)k + 1]; x11 += p3[3 * (size_t)k + 2]; }
                    double pm[4] = { 0, 0, 0, 0 };
#define TSTEP_PRED(A_) ({ const double a_ = (A_), ta_ = tau + a_ * dt;                                              \
                    pm[0] = sqrt(fmax(0.0, r00 + 2.0 * a_ * r01 + a_ * a_ * r11)) / (ta_ * (1 + P->normb2));          \
                    pm[1] = fabs(1.0 - a_ * (1.0 - sigma)) * rdo / (ta_ * (1 + P->normC2));                             \
                    const double po_ = sc * (cX + a_ * cdX) / ta_, do_ = sc * (bty + a_ * bdy) / ta_;                   \
                    const double den_ = 1 + fabs(po_ + P->obj_off) + fabs(do_ + P->obj_off);                            \
                    pm[2] = fabs(po_ - do_) / den_;                                                                     \
                    pm[3] = fabs(sc * (xz + a_ * x01 + a_ * a_ * x11)) / (ta_ * ta_) / den_;                           \
                    fmax(fmax(pm[0], pm[1]), fmax(pm[2], pm[3])); })
                    const double s_def = (tau + alpha * dt > 0) ? TSTEP_PRED(alpha) : INFINITY;
                    if (ENV_ON("BRISK_TSTEPDBG")) printf("     [terminal step: default %.5f predicted %.2e (pinf %.1e dinf %.1e gap %.1e comp %.1e), boundary %.5f]\n", alpha, s_def, pm[0], pm[1], pm[2], pm[3], best_am);
                    if (s_def > tstep_t * par->tol) {
                        const double fr[4] = { 0.25, 0.5, 0.75, 1.0 };
                        for (int c = 0; c < 4; c++) {
                            const double a = alpha + fr[c] * (amax - alpha);
                            if (!(tau + a * dt > 0) || !(kap + a * dk > 0)) continue;
                            const double s_a = TSTEP_PRED(a);
                            if (ENV_ON("BRISK_TSTEPDBG")) printf("     [terminal step: %.5f predicted %.2e (pinf %.1e dinf %.1e gap %.1e comp %.1e)]\n", a, s_a, pm[0], pm[1], pm[2], pm[3]);
                            if (s_a <= tstep_t * par->tol) { alpha = a; n_tstep++; break; }
                        }
                    }
#undef TSTEP_PRED
                    R->t_dense += wtime() - td;
                }
            }
            { double _t = wtime(); g_ph[6] += _t - g_ph_last; g_ph_last = _t; }
            /* experiment (BRISK_HSDINEX=beta): the inexact-Newton condition on the step. The primal residual after the step must
             * stay within beta times its Newton prediction (1 - alpha (1 - sigma)) |rp| (plus a tenth of the tolerance); a
             * direction that misses its primal target is followed only as far as that holds. With an exact direction the cap never
             * binds; where the Schur solves have reached their accuracy the steps shrink and the floor stops end the iteration at
             * the best iterate, in either form of the tau column */
            if (inex_have && alpha > 0) {
                const double nr = sqrt(inex_R2), fl = 0.1 * par->tol * tau * (1 + P->normb2);
                #define INEX_G(A_) ({ const double a_ = (A_), q_ = 1.0 - a_ * (1.0 - inex_sig); \
                    sqrt(fmax(0.0, q_ * q_ * inex_R2 + 2.0 * q_ * a_ * inex_RE + a_ * a_ * inex_E2)) - (inex_beta * q_ * nr + fl); })
                if (INEX_G(alpha) > 0) {
                    double lo = 0, hi = alpha;
                    for (int t = 0; t < 50; t++) { const double mid = 0.5 * (lo + hi); if (INEX_G(mid) > 0) hi = mid; else lo = mid; }
                    if (ENV_ON("BRISK_HSDDBG")) printf("     [inexact Newton: step %.4f -> %.4f (miss %.2e, |rp| %.2e)]\n", alpha, lo, sqrt(inex_E2), nr);
                    alpha = lo; n_inex++;
                }
                #undef INEX_G
            }
            /* ---- step: one alpha for X, Z, tau, kappa */
            tst = wtime();
            /* one step length for X, Z, tau and kappa (try_step works in place, so a
             * side that was stepped too far is undone before it is stepped again) */
            double ap = alpha, ad = alpha;
            double *dbgA0 = NULL, *dbgAd = NULL;
            if (getenv("BRISK_STEPDBG")) {
                dbgA0 = calloc(m, sizeof(double)); dbgAd = calloc(m, sizeof(double));
                for (int k = 0; k < nb; k++) { blk_Aop(&P->blk[k], &S[k], S[k].X, dbgA0, 1.0); blk_Aop(&P->blk[k], &S[k], S[k].dX, dbgAd, 1.0); }
            }
            int okp = try_step(P, S, 1, &ap);
            int okd = try_step(P, S, 0, &ad);
            for (int rep = 0; rep < 8 && okp && okd && ap != ad; rep++) {
                if (ad < ap) { step_undo(P, S, 1); ap = ad; okp = try_step(P, S, 1, &ap); }
                else { step_undo(P, S, 0); ad = ap; okd = try_step(P, S, 0, &ad); }
            }
            if (ap != ad) okp = 0;
            double a = ap;
            if (dbgA0) {
                double *A1 = calloc(m, sizeof(double)), e = 0, nd = 0;
                for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].X, A1, 1.0);
                for (int i = 0; i < m; i++) { double t = A1[i] - dbgA0[i] - a * dbgAd[i]; e += t * t; nd += dbgAd[i] * dbgAd[i]; }
                printf("     [step check: a %.3f |A(X1)-A(X0)-a A(dX)| %.2e |A(dX)| %.2e]\n", a, sqrt(e), sqrt(nd));
                free(A1); free(dbgA0); free(dbgAd);
            }
            R->t_step += wtime() - tst;
            if (!okp || !okd) { status = 4; g_hsd_why = "no step keeps X and Z positive definite"; break; }
            for (int i = 0; i < m; i++) y[i] += a * dyd[i];
            tau += a * dt;
            kap += a * dk;
            theta *= 1.0 - a * (1.0 - sigma);
            pth *= 1.0 - a * (1.0 - sigma);
            /* experiment (BRISK_HSDRAY=1): the scale of the iterate restored along the ray. Scaling (X, y, Z, tau, kappa) by c
             * multiplies the residuals by c and mu by c^2, so the identity mu = theta mu0 of the ray form (Xu-Hung-Ye: the
             * residuals are (mu/mu0) r0 along the path; the content of YTM's normalization row) fixes c = theta mu0 / mu.
             * The direction's tau component drifts along the ray at the solve floor; the rescale undoes the drift
             * without touching the direction (one n^2 pass per block) */
            if (ray_on && it > 0 && mu0_ray > 0) {
                double xzn = 0; for (int k = 0; k < nb; k++) xzn += ddot_n(bsz(&P->blk[k]), S[k].X, S[k].Z);
                const double mun = (xzn + tau * kap) / ndim1, c = pth * mu0_ray / mun;
                if (c > 0.5 && c < 2.0 && fabs(c - 1.0) > 1e-14) {
                    for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]); PAR_LEN_FOR(len, S[k].X[i] *= c; S[k].Z[i] *= c;); }
                    for (int i = 0; i < m; i++) y[i] *= c;
                    tau *= c; kap *= c; n_ray++;
                    if (ENV_ON("BRISK_HSDDBG")) printf("     [ray it %d: scale %.6f (mu %.2e, theta mu0 %.2e)]\n", it, c, mun, pth * mu0_ray);
                }
            }
            rpo_pred = (1.0 - a * (1.0 - sigma)) * rpo_it; shift_prev = shc.on;
            alpha = a;
            if (npk && !nf) {
                /* x+ - x- is all that matters: shrink both by a fraction of the smaller one
                 * (A and C see opposite columns: residuals and objective are unchanged) */
                const double th = par->hsd_pshrink;
                for (int f = 0; f < npk; f++) {
                    BS *s = &S[pk_b[f]];
                    double *xp = &s->X[pk_p[f]], *xq = &s->X[pk_q[f]];
                    const double sm = fmin(*xp, *xq) * th;
                    if (sm > 0) { *xp -= sm; *xq -= sm; }
                }
            }
            stall = (a < 1e-6) ? stall + 1 : 0;
            tiny_steps = (a < 0.02) ? tiny_steps + 1 : 0;
            small_steps = (a < 0.25) ? small_steps + 1 : 0;
            if (!(tau > 0) || !(kap > 0)) { status = 4; g_hsd_why = "tau or kappa left the positive range"; break; }
        }
        { double _t = wtime(); g_ph[7] += _t - g_ph_last; g_ph_last = _t; }
        /* ---- normalize tau to 1 (the embedding is homogeneous) */
        /* only shrink: tau -> 0 is the infeasibility signal and must be preserved */
        if (tau > 10.0) {
            double f = 1.0 / tau, sq = sqrt(f);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                size_t len = bsz(B);
                PAR_LEN_FOR(len, S[k].X[i] *= f; S[k].Z[i] *= f;);
                if (B->type != BLK_SDP) for (int i = 0; i < B->n; i++) { S[k].LX[i] *= f; S[k].LZ[i] *= f; }
                else for (size_t i = 0; i < len; i++) { S[k].LX[i] *= sq; S[k].LZ[i] *= sq; }
            }
            for (int i = 0; i < m; i++) y[i] *= f;
            kap *= f;
            theta *= f;
            pth *= f;
            tau = 1.0;
        }
    }
done:
    { extern int g_asmsp, g_ntsp; g_asmsp = 0; g_ntsp = 0; }   /* (single precision belongs to the embedding's iterations only) */
    mvfree_free(Sc);
    if (ocX) {
        /* (19 October 2026, the review) only when the run missed the tolerance: not after the DIMACS stop, a probe
         * stop or an infeasibility verdict, and only when the best iterate itself misses the tolerance; and only when
         * the candidate scores no worse than the final iterate, which brisk_solve_impl would otherwise return in its
         * place (the best-scored iterate then lost) */
        if (status != 0 && status != 1 && status != 2 && !g_dstop && !g_epol_stop && *best_score > par->tol
            && oc_it >= 0 && oc_it != *it_best && res_score(&oc_R) <= res_score(R)) {
            const double denb = 1 + fabs(best->pobj + P->obj_off) + fabs(best->dobj + P->obj_off);
            const double thr = fmax(100.0 * par->tol, 10.0 * best->relgap) * denb;
            if (oc_po < best->pobj - thr) {
                if (par->verbose > 0) printf("   HSD: the iterate %d is better in the objective than the best-scored iterate %d by %.1e relative (residuals %.1e, %.1e): returning it\n",
                                             oc_it, *it_best, (best->pobj - oc_po) / denb, oc_R.pinf, oc_R.dinf);
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    memcpy(S[k].Xb, ocX[k], sizeof(double) * len); memcpy(S[k].Zb, ocZ[k], sizeof(double) * len);
                }
                memcpy(ybest, ocy, sizeof(double) * m);
                *best = oc_R; *it_best = oc_it; *best_score = res_score(&oc_R);
                oc_used = 1; g_objcand = 1;
            }
        }
        for (int k = 0; k < nb; k++) { free(ocX[k]); free(ocZ[k]); }
        free(ocX); free(ocZ); free(ocy);
    }
    g_drift_held = drift_first >= 0;
    if (fbX) {
        /* a probe candidate of an earlier iteration does not replace the objective candidate */
        if (fb_drift ? (status != 0 && !g_epol_stop && !g_dstop && !oc_used) : (fb_score < *best_score && !g_epol_stop && !oc_used)) {
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]);
                memcpy(S[k].G, fbX[k], sizeof(double) * len); memcpy(S[k].Q, fbZ[k], sizeof(double) * len);
            }
            memcpy(ybest, fby, sizeof(double) * m);
            g_epol_R = fb_R;
            g_epol_stop = 1; g_epol_fb = 1;
            if (par->verbose > 0) printf("   HSD: returning the probe candidate (%.2e; best iterate %.2e)\n", fb_score, *best_score);
        }
        for (int k = 0; k < nb; k++) { free(fbX[k]); free(fbZ[k]); }
        free(fbX); free(fbZ); free(fby);
    }
    if (tau > 0)
        for (int k = 0; k < nb; k++) {
            size_t len = bsz(&P->blk[k]);
            PAR_LEN_FOR(len, S[k].X[i] /= tau; S[k].Z[i] /= tau;);
        }
    g_hsd_fxe = 0;
    for (int k = 0; k < nb; k++) g_hsd_fxe += ddot_n(bsz(&P->blk[k]), S[k].X, S[k].X);
    g_hsd_fxe = sqrt(g_hsd_fxe);
    if (ENV_ON("BRISK_SCALEDBG")) {
        double tx = 0, tz = 0;
        for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) { tx += S[k].X[i]; tz += S[k].Z[i]; }
            else if (B->type == BLK_SOC) { tx += S[k].X[0]; tz += S[k].Z[0]; }
            else for (int i = 0; i < n; i++) { tx += S[k].X[i + (size_t)i * n]; tz += S[k].Z[i + (size_t)i * n]; } }
        printf("   SCALEDBG end trX %.3e trZ %.3e tau %.3e its %d\n", tx, tz, tau, it);
    }
    if (nf) {
        /* back to split pairs: x+ = max(x,0) + s0, x- = max(-x,0) + s0 (the shift cancels in
         * A(X) and <C,X>), slacks at the complementarity level of the cone part */
        double muf = 0;
        {
            double xz = 0;
            for (int k = 0; k < nb; k++) xz += ddot_n(bsz(&P->blk[k]), S[k].X, S[k].Z);
            muf = fmax(xz / fmax(ndim, 1.0), 1e-300);
        }
        for (int f = 0; f < nf; f++) {
            for (int w = 0; w < 3; w++) {
                if ((w == 1 && !keep_best) || (w == 2 && !(rc && rc->have))) continue;
                double *Xk = w == 2 ? S[FV.kb[f]].Xr : w ? S[FV.kb[f]].Xb : S[FV.kb[f]].X, *Zk = w == 2 ? S[FV.kb[f]].Zr : w ? S[FV.kb[f]].Zb : S[FV.kb[f]].Z;
                const double x = Xk[FV.ip[f]], s0 = 1e-8 * (1.0 + fabs(x));
                Xk[FV.ip[f]] = fmax(x, 0.0) + s0;
                Xk[FV.im[f]] = fmax(-x, 0.0) + s0;
                Zk[FV.ip[f]] = muf / Xk[FV.ip[f]];
                Zk[FV.im[f]] = muf / Xk[FV.im[f]];
            }
        }
        if (par->verbose > 1) printf("   (free variables: factor %.3fs, %d solves / %d passes %.3fs)\n", g_tfa, g_nfs, g_nfp, g_tfs);
        free_release(P, S, &FV);
        free(uf0); free(uf1); free(vf); free(ufa); free(rf); free(dxfd); free(qf);
    }
    for (int i = 0; i < m; i++) y[i] /= (tau > 0 ? tau : 1.0);
    if (par->verbose && nsafe) printf("   (HSD neighbourhood safeguard shortened %d steps)\n", nsafe);
    Sc->pivtol = pivtol_saved; Sc->use_hint = 0; Sc->reg_hint = 0;
    if (par->verbose && ncorr) printf("   (HSD: %d centrality correctors accepted)\n", ncorr);
    if (par->verbose > 1 && n_tstep) printf("   (HSD: terminal step taken in %d iteration%s)\n", n_tstep, n_tstep > 1 ? "s" : "");
    if (par->verbose && nrefine) printf("   (HSD: %d refinement passes)\n", nrefine);
    if (par->verbose && (hoc_nkept || hoc_nfail)) printf("   (HSD: second-order passes kept in %d iterations, worse than the first direction in %d)\n", hoc_nkept, hoc_nfail);
    if (par->verbose > 1 && (nref_skip || nref_fail)) printf("   (HSD: refinement at the solve floor: %d stopped after the solve, %d skipped)\n", nref_fail, nref_skip);
    if (par->verbose > 1 && n_inex) printf("   (HSD: %d step%s shortened by the inexact-Newton condition)\n", n_inex, n_inex > 1 ? "s" : "");
    if (par->verbose > 1 && n_proj) printf("   (HSD: the primal rows restored by a least-norm correction in %d iterations)\n", n_proj);
    if (par->verbose > 1 && n_ytm) printf("   (HSD: dtau from the normalization row in %d iterations)\n", n_ytm);
    if (par->verbose > 1 && n_ray) printf("   (HSD: the scale restored along the ray in %d iterations)\n", n_ray);
    if (par->verbose > 1 && n_gal) printf("   (HSD: the multipliers' component of the primal miss removed in %d iterations)\n", n_gal);
    free(rp0);
    free(dyd);
    *it_out = it;
    for (int k = 0; k < nb; k++) { free(dX0[k]); free(dZ0[k]); free(dX1[k]); free(dZ1[k]); }
    free(dX0); free(dZ0); free(dX1); free(dZ1);
    for (int k = 0; k < nb; k++) { free(hp[k].fr); free(hp[k].fc); free(hp[k].rows); }
    free(hp); free(pk);
    for (int k = 0; k < nb; k++) free(shc.F[k]);
    free(shc.F); free(bmh);
    for (int k = 0; k < nb; k++) free(Zv[k]);
    free(Zv);
    if (par->verbose > 1 && nrebase) printf("   (HSD: residual re-based %d times)\n", nrebase);
    free(pk_b); free(pk_p); free(pk_q); free(shist); free(phist); free(ihist); free(mhist); free(ghist);
    if (bord) {
        Sc->bF = NULL; Sc->bP = NULL; free(Sc->bE); Sc->bE = NULL;
        free(bpf); free(bD);
        free_release(P, S, &BV);
    }
    free(rp); free(rhs); free(u0); free(u1); free(v); free(dy); free(a0); free(hc); free(hd); free(qv);
    endrules_free(&ER);
    return status;
}

static int brisk_solve_impl(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout);
/* 4.42: threads of a solve.
 * - Tiny problems run on one thread. An iteration of under 5e6 flops gains nothing from
 *   threads, and with libgomp's spinning idle threads a busy machine made it 400 times slower
 *   (truss1 with another process on the second core: 0.85 s instead of 0.002 s).
 *   BRISK_TINYMT=1 keeps the threads.
 * - A busy machine: when other threads are running right now, the solve starts with the cores
 *   that are free (until 5.10 only with the default count, and the count stayed; 5.11: any
 *   count, only when every core is allowed, and the count follows the load, below). With
 *   spinning barriers, one thread sharing a core with another process stalls all the others
 *   (arch0 on two cores with one taken: 1.34 s against 0.37 s on one thread). The count of
 *   runnable threads is the fourth field of /proc/loadavg (Linux; elsewhere no change): the
 *   minimum of three samples a millisecond apart, so that a passing wake-up does not count.
 *   BRISK_NOBUSY=1 turns the rule off.
 * The thread count is restored on return. */
int g_threads_user = 0;      /* -threads or OMP_NUM_THREADS given (main.c) */
/* the runnable threads on the machine, this one included: the fourth field of /proc/loadavg
 * (Linux), else -1 */
static int runnable_now(void) {
#ifdef __linux__
    char b[96];
    const int fd = open("/proc/loadavg", O_RDONLY);
    if (fd < 0) return -1;
    const ssize_t n = read(fd, b, sizeof b - 1);
    close(fd);
    if (n <= 0) return -1;
    b[n] = 0;
    const char *p = b;
    for (int sp = 0; *p && sp < 3; p++) if (*p == ' ') sp++;
    return *p >= '0' && *p <= '9' ? atoi(p) : -1;
#else
    return -1;
#endif
}
/* the runnable threads of this process (state R in /proc/self/task/<tid>/stat), the caller included, or -1 */
static int runnable_own(void) {
#ifdef __linux__
    DIR *d = opendir("/proc/self/task");
    if (!d) return -1;
    int n = 0; struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char path[300], b[512];   /* (a d_name holds up to 255 bytes) */
        snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
        const int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        const ssize_t k = read(fd, b, sizeof b - 1);
        close(fd);
        if (k <= 0) continue;
        b[k] = 0;
        const char *q = strrchr(b, ')');          /* the state follows the command name, which may hold spaces */
        if (q && q[1] == ' ' && q[2] == 'R') n++;
    }
    closedir(d);
    return n;
#else
    return -1;
#endif
}
#ifdef __linux__
static int allowed_cores(cpu_set_t *ms);
#endif
/* the cores that are free, from /proc/loadavg, or -1. It counts the runnable threads of the
 * whole machine, so it is used only when every core is allowed (5.11). The threads of this
 * process are not other threads: 23 October 2026, libgomp's idle threads spin for a while after
 * a parallel region and are runnable, so the solve after the matrix-free method counted its own
 * three workers and started on 1-2 threads of 4 (tru11, POEMA, SDP, the machine otherwise idle:
 * the first iteration 2.4 s against about 1 s for the others). */
static int free_cores(void) {
#ifdef __linux__
    const long on = sysconf(_SC_NPROCESSORS_ONLN);
    { cpu_set_t ms; if (allowed_cores(&ms) < on) return -1; }
    int other = 1 << 30;
    for (int s = 0; s < 3; s++) {
        const int r = runnable_now(), own = runnable_own();
        if (r < 0) return -1;
        const int o = r - (own >= 1 ? own : 1);
        if (o < other) other = o;
        if (other <= 0) break;
        struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL);
    }
    if (other < 0) other = 0;
    const int fr = (int)on - other;
    return fr < 1 ? 1 : fr;
#else
    return -1;
#endif
}
/* 5.11: the thread count follows the free cores during the solve, and a count asked for
 * (-threads k, OMP_NUM_THREADS) is an upper limit like the default one (a user's patch; the
 * measures are this version's).
 * A team with more threads than free cores is slower than one thread: its barriers spin
 * while a thread waits for a core, and what a thread without a core holds up, all wait for.
 * With another process on one of two cores, 2 threads took 1.2 to 3.4 times the time of
 * 1 thread on 14 problems (theta3: 0.89 s against 0.43 s), and the count chosen when the
 * solve started stayed for the whole solve.
 * Two measures, both of the cores this process may use (its affinity mask: a taskset, a
 * cpuset of a container or of a cluster job), at the start of an iteration once 50 ms have
 * passed since the last decision:
 * - down: the time the solver's threads spent waiting for a core (run-queue wait, second
 *   field of /proc/<pid>/task/<tid>/schedstat, in ns), divided by the time: c threads on
 *   average. c >= 0.3 in two windows in a row lowers the count by max(1, round(c)) (the
 *   smaller c; one window is not enough: on an idle machine a burst of another process gave
 *   0.58 once in 23 windows of control10). It is the harm itself: other
 *   processes on cores this process may not use cost nothing and do not count. Measured
 *   with 4 threads spinning on 4 cores: idle 0.00-0.02, one other busy process 0.75-0.93,
 *   three 1.8; 2 threads confined to 2 cores with three busy processes on the other two
 *   0.00 (windows of 10 ms and more; under 10 ms a waiting thread is often not yet seen);
 * - up (when under the limit): the busy time of the allowed cores (/proc/stat, ticks of
 *   10 ms) less this process's CPU time is the share the other processes use; with f the
 *   allowed cores less that share, the count rises to floor(f + 0.4) if that is higher.
 *   Over 100 ms, f was 3.86-4.16 with 4 cores idle and 2.74-3.09 with one taken.
 * /proc/loadavg, used before, counts the runnable threads of the whole machine: in a cpuset
 * busy cores outside it took the count down for nothing (2 -> 1 threads, 38 % slower on
 * the largest of 29 problems; on a shared cluster node it would be 8 -> 1). It now decides
 * only the count of the start (free_cores), and only when every core is allowed.
 * The per-thread workspaces grow on demand (fscr_ensure, tps_ensure, nthr_asm), and what is
 * fixed once per solve (g_par_blk, g_hpar) is decided from the limit (nthreads_cap); every
 * kernel gives the same bits for any count, so the changes do not change the result.
 * BRISK_NOBUSY=1 turns all of it off: the count is then exactly the one asked for.
 * BRISK_THREADTEST=<seed> draws the count of every iteration at random (a test of the code
 * paths); BRISK_NTDBG=1 prints the measures. Linux; elsewhere nothing changes. */
static int g_nt_adapt = 0;                              /* the rule is on for the running solve */
static unsigned g_nt_test = 0;                          /* BRISK_THREADTEST */
static int g_nt_lo = 0, g_nt_hi = 0, g_nt_lim = 0, g_nt_chg = 0;      /* for the log and the summary */
static void nt_used(int nt) { if (g_nt_lo == 0 || nt < g_nt_lo) g_nt_lo = nt; if (nt > g_nt_hi) g_nt_hi = nt; }
#ifdef __linux__
static cpu_set_t g_nt_mask;                             /* the cores the solve may use */
static int g_nt_ncore = 0;
/* the cores this process may use: the affinity mask, or with thread binding (which narrows
 * the initial thread's mask to its place) the union of the OpenMP places */
static int allowed_cores(cpu_set_t *ms) {
    CPU_ZERO(ms);
#if defined(_OPENMP) && _OPENMP >= 201511 && !defined(BRISK_OMP_H)
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        const int np = omp_get_num_places();
        int ids[CPU_SETSIZE];
        for (int i = 0; i < np; i++) {
            const int n = omp_get_place_num_procs(i);
            if (n <= 0 || n > CPU_SETSIZE) continue;
            omp_get_place_proc_ids(i, ids);
            for (int j = 0; j < n; j++) if (ids[j] >= 0 && ids[j] < CPU_SETSIZE) CPU_SET(ids[j], ms);
        }
        if (CPU_COUNT(ms) > 0) return CPU_COUNT(ms);
    }
#endif
    if (sched_getaffinity(0, sizeof *ms, ms) != 0) return -1;
    return CPU_COUNT(ms);
}
/* the run-queue wait of all threads of this process so far (ns), or -1 */
static long long proc_wait_ns(void) {
    DIR *d = opendir("/proc/self/task");
    if (!d) return -1;
    long long s = 0; int nt = 0;
    struct dirent *e;
    char path[300], b[128];
    while ((e = readdir(d))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        snprintf(path, sizeof path, "/proc/self/task/%s/schedstat", e->d_name);
        const int fd = open(path, O_RDONLY);
        if (fd < 0) continue;                       /* a thread that has just ended */
        const ssize_t n = read(fd, b, sizeof b - 1);
        close(fd);
        if (n <= 0) continue;
        b[n] = 0;
        const char *p = strchr(b, ' ');
        if (p) { s += atoll(p + 1); nt++; }
    }
    closedir(d);
    return nt > 0 ? s : -1;
}
/* the busy time of the allowed cores so far (s; /proc/stat), or -1 */
static double mask_busy_s(void) {
    static char *buf = NULL; static size_t cap = 0;
    const int fd = open("/proc/stat", O_RDONLY);
    if (fd < 0) return -1;
    size_t n = 0;
    for (;;) {
        if (cap - n < 4096) { char *nb = realloc(buf, cap ? 2 * cap : 1 << 16); if (!nb) { close(fd); return -1; } buf = nb; cap = cap ? 2 * cap : 1 << 16; }
        const ssize_t r = read(fd, buf + n, cap - n - 1);
        if (r <= 0) break;
        n += (size_t)r;
    }
    close(fd);
    buf[n] = 0;
    long long t = 0;
    for (const char *p = strstr(buf, "\ncpu"); p; p = strstr(p + 1, "\ncpu")) {
        int c; long long u, ni, sy, id, io, irq, si, st;
        if (sscanf(p + 4, "%d %lld %lld %lld %lld %lld %lld %lld %lld", &c, &u, &ni, &sy, &id, &io, &irq, &si, &st) != 9) continue;
        if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &g_nt_mask)) t += u + ni + sy + irq + si + st;
    }
    return (double)t / (double)sysconf(_SC_CLK_TCK);
}
static double proc_cpu_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return -1;
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
/* the windows of the two measures: since the last decision (down) and since the last change (up) */
static double g_dn_t = -1, g_up_t = -1, g_up_busy = 0, g_up_cpu = 0;
static long long g_dn_wait = 0;
static int g_dn_pend = 0; static double g_dn_c = 0;      /* the last window was over the threshold, its c */
static void nt_window_up(double now) { g_up_t = now; g_up_busy = mask_busy_s(); g_up_cpu = proc_cpu_s(); }
/* the start of a solve: the allowed cores, and whether the run-queue wait can be read */
static int nt_measure_start(void) {
    g_nt_ncore = allowed_cores(&g_nt_mask);
    const long long w = proc_wait_ns();
    if (g_nt_ncore < 1 || w < 0) return 0;
    const double now = wtime();
    g_dn_t = now; g_dn_wait = w; g_dn_pend = 0;
    nt_window_up(now);
    return 1;
}
#endif
void brisk_threads_adapt(int verbose) {
#ifdef _OPENMP
    if (!g_nt_adapt || g_nt_cap <= 1 || omp_in_parallel()) return;
    const int cur = omp_get_max_threads(), cap = g_nt_cap;
    int want = cur;
    if (g_nt_test) {
        g_nt_test = g_nt_test * 1103515245u + 12345u;
        want = 1 + (int)((g_nt_test >> 16) % (unsigned)cap);
    } else {
#ifdef __linux__
        const double now = wtime();
        if (now - g_dn_t < 0.05) return;
        const long long w = proc_wait_ns();
        if (w < 0) return;
        const double c = 1e-9 * (double)(w - g_dn_wait) / (now - g_dn_t);
        g_dn_t = now; g_dn_wait = w;
        double f = -1;
        if (c >= 0.3 && cur > 1) {
            if (g_dn_pend) {                        /* the second window in a row: down */
                const int k = (int)((c < g_dn_c ? c : g_dn_c) + 0.5);
                want = cur - (k > 1 ? k : 1);
                if (want < 1) want = 1;
                g_dn_pend = 0;
            } else { g_dn_pend = 1; g_dn_c = c; }
        } else if (cur < cap && now - g_up_t >= 0.1) {
            const double bz = mask_busy_s(), cp = proc_cpu_s();
            if (bz >= 0 && cp >= 0 && g_up_busy >= 0 && g_up_cpu >= 0) {
                f = g_nt_ncore - ((bz - g_up_busy) - (cp - g_up_cpu)) / (now - g_up_t);
                int up = (int)(f + 0.4);
                if (up > cap) up = cap;
                if (up > cur) want = up;
            }
            g_up_t = now; g_up_busy = bz; g_up_cpu = cp;
        } else if (cur == cap) g_up_t = now;
        if (c < 0.3) g_dn_pend = 0;
        if (getenv("BRISK_NTDBG")) printf("   [threads %d: waiting %.2f, free %.2f -> %d]\n", cur, c, f, want);
        if (want != cur) nt_window_up(now);
#endif
    }
    if (want == cur) return;
    omp_set_num_threads(want);
    nt_used(want);
    g_nt_chg++;
    if (verbose > 0 && g_nt_chg <= 6)
        printf("   threads: %d of %d (%s)%s\n", want, cap, want < cur ? "the threads were waiting for cores" : "cores are free again",
               g_nt_chg == 6 ? "; further changes are not shown" : "");
#else
    (void)verbose;
#endif
}
/* the summary's line: the count, or the range the solve used */
const char *brisk_threads_summary(void) {
    static char b[160];
    if (g_nt_hi <= 0) return NULL;
    if (g_nt_lo < g_nt_hi) snprintf(b, sizeof b, "%d to %d (the count followed the free cores; limit %d)", g_nt_lo, g_nt_hi, g_nt_lim);
    else if (g_nt_hi < g_nt_lim) snprintf(b, sizeof b, "%d of %d (the cores that were free)", g_nt_hi, g_nt_lim);
    else return NULL;
    return b;
}
/* the busy-machine rule for the other engines (main.c): lowers the thread count to the free
 * cores and returns the count to restore, or -1 when nothing was changed */
/* the log line "Number of threads: k": the OpenMP threads the engine that starts now will use.
 * Printed once per run, and again only if a later solve of the same run uses another count
 * (brisk_log_threads_reset at the start of a run). */
static int g_nt_logged = -1;
void brisk_log_threads_reset(void) { g_nt_logged = -1; g_nt_lo = g_nt_hi = g_nt_lim = g_nt_chg = 0; }
int brisk_threads_logged(void) { return g_nt_logged; }   /* 5.8: the count last logged, for the summary (-1: none) */
void brisk_log_threads(int verbose) {
    const int nt = nthreads();
    if (verbose > 0 && nt != g_nt_logged) { printf("Number of threads: %d\n", nt); g_nt_logged = nt; }
}
int brisk_threads_busy(int verbose) {
#ifdef _OPENMP
    const int nt0 = omp_get_max_threads();
    if (nt0 > 1 && !getenv("BRISK_NOBUSY")) {
        const int fr = free_cores();
        if (fr >= 1 && fr < nt0) {
            if (verbose > 0) printf("   threads: %d of %d (other threads are running on this machine; BRISK_NOBUSY=1 keeps the count)\n", fr, nt0);
            omp_set_num_threads(fr);
            return nt0;
        }
    }
#endif
    (void)verbose;
    return -1;
}
/* ---- 5.16: semidefinite blocks with arrow data solved as second-order cones (REPORT_5.16 §9).
 * A block of order d whose C and A_i are all arrow matrices Arw(a) = [[a_0, ab'], [ab, a_0 I]]
 * (in some ordering: one hub index carries every off-diagonal entry, the diagonal is constant)
 * is a second-order cone: Z = Arw(z) >= 0 iff z in Q, and <Arw(a), X> = a'x with x_0 = tr X,
 * x_t = 2 X_{hub, t}, where x ranges over Q as X ranges over the semidefinite cone. This is how a
 * cone is written into the semidefinite form (sedumi.c) and how users write norm constraints as
 * LMIs; as a d x d block it costs O(d^3) an iteration, as a cone O(d). The embedding is run on the
 * problem with these blocks as cones; X of the problem as given is the rank-two completion
 * X = 1/2 [[x_0, xb'], [xb, x_0 u u']] (u = xb / ||xb||), positive semidefinite iff x in Q. */
void spsym_finish(SpSym *S, int n, int is_lp);
/* the hub of a block with arrow data, or -1 */
static int arrow_hub(const Block *B) {
    const int n = B->n;
    static int dbg = -1; if (dbg < 0) dbg = getenv("BRISK_ARROWDBG") != NULL;
    if (B->type != BLK_SDP || n < 2) return -1;      /* (whatever its Schur route: the cone block is rebuilt from the data) */
    int c1 = -1, c2 = -1, offd = 0;            /* the hub candidates: in every off-diagonal entry */
    for (int t = -1; t < B->ncon; t++) {
        const SpSym *S = t < 0 ? &B->C : &B->A[t];
        int nd = 0; double dv = 0;
        for (int q = 0; q < S->nnz; q++) {
            const int i = S->row[q], j = S->col[q];
            const double v = S->val[q];
            if (i == j) {
                if (nd == 0) dv = v;
                else if (!(fabs(v - dv) <= 1e-14 * fabs(dv))) { if (dbg) printf("   [arrow n %d: diagonal %.17g vs %.17g in matrix %d]\n", n, v, dv, t); return -1; }
                nd++;
                continue;
            }
            offd++;
            if (c1 < 0 && c2 < 0) { c1 = i; c2 = j; continue; }
            if (c1 >= 0 && c1 != i && c1 != j) c1 = -1;
            if (c2 >= 0 && c2 != i && c2 != j) c2 = -1;
            if (c1 < 0 && c2 < 0) { if (dbg) printf("   [arrow n %d: no hub, entry (%d,%d) %.3g in matrix %d]\n", n, i, j, v, t); return -1; }
        }
        if (nd != 0 && nd != n) { if (dbg) printf("   [arrow n %d: %d diagonal entries in matrix %d]\n", n, nd, t); return -1; }
    }
    if (!offd) return -1;
    return c1 >= 0 ? c1 : c2;
}
int brisk_arrow_hub(const Block *B) { return arrow_hub(B); }
/* 17 October 2026: whether brisk_solve will solve this block as a second-order cone (the gate of brisk_solve below);
 * problem_prepare then skips its low-rank and dictionary analyses, which the cone never uses (filter48_socp: 0.28 s
 * of factorizations of the 966 constraints of its arrow block of order 49, in a solve of 0.7 s) */
int brisk_block_as_cone(const Params *par, const Block *B) {
    return par->soc && !par->dual && !par->mfipm && !par->btrack && !par->bound_side && !par->warm_X && par->hsd != 0
        && B->type == BLK_SDP && B->n >= par->soc_minn && arrow_hub(B) >= 0;
}
/* order: ord[0] = hub, then the other indices; pos[i] = its place in the cone vector */
static void arrow_order(int n, int hub, int *ord, int *pos) {
    ord[0] = hub; pos[hub] = 0;
    int c = 1;
    for (int i = 0; i < n; i++) if (i != hub) { ord[c] = i; pos[i] = c++; }
}
/* the cone vector of an arrow matrix (stored upper entries) as an LP-type SpSym */
static void arrow_vec(const SpSym *S, const int *pos, int hub, int n, SpSym *V) {
    double *a = calloc((size_t)n + 1, sizeof(double));
    char *in = calloc((size_t)n + 1, 1);
    for (int q = 0; q < S->nnz; q++) {
        const int i = S->row[q], j = S->col[q];
        if (i == j) { a[0] = S->val[q]; in[0] = 1; }
        else { const int o = i == hub ? j : i; a[pos[o]] = S->val[q]; in[pos[o]] = 1; }
    }
    int nz = 0;
    for (int i = 0; i < n; i++) nz += in[i];
    memset(V, 0, sizeof(*V));
    V->nnz = nz;
    V->row = malloc(sizeof(int) * (nz + 1)); V->col = malloc(sizeof(int) * (nz + 1)); V->val = malloc(sizeof(double) * (nz + 1));
    int w = 0;
    for (int i = 0; i < n; i++) if (in[i]) { V->row[w] = V->col[w] = i; V->val[w] = a[i]; w++; }
    spsym_finish(V, n, 1);
    free(a); free(in);
}
/* Q: P with the arrow blocks as cone blocks (the other blocks shared); hub[k] >= 0 marks them */
static void soc_problem(const Problem *P, Problem *Q, const int *hub, int **ordk) {
    *Q = *P;
    Q->blk = calloc((size_t)P->nblk + 1, sizeof(Block));
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (hub[k] < 0) { Q->blk[k] = *B; continue; }
        Block *C = &Q->blk[k];
        const int n = B->n;
        int *pos = malloc(sizeof(int) * (n + 1));
        ordk[k] = malloc(sizeof(int) * (n + 1));
        arrow_order(n, hub[k], ordk[k], pos);
        C->type = BLK_SOC; C->n = n; C->ncon = B->ncon;
        C->con = malloc(sizeof(int) * (B->ncon + 1));
        memcpy(C->con, B->con, sizeof(int) * B->ncon);
        C->A = calloc((size_t)B->ncon + 1, sizeof(SpSym));
        for (int t = 0; t < B->ncon; t++) arrow_vec(&B->A[t], pos, hub[k], n, &C->A[t]);
        arrow_vec(&B->C, pos, hub[k], n, &C->C);
        /* coefficient lists by variable, as problem_prepare makes them for an LP block */
        int *cnt = calloc((size_t)n + 2, sizeof(int)), tot = 0;
        for (int t = 0; t < C->ncon; t++) for (int q = 0; q < C->A[t].nnz; q++) { cnt[C->A[t].row[q] + 1]++; tot++; }
        for (int p = 0; p < n; p++) cnt[p + 1] += cnt[p];
        C->lp_ptr = malloc(sizeof(int) * (n + 1)); memcpy(C->lp_ptr, cnt, sizeof(int) * (n + 1));
        C->lp_con = malloc(sizeof(int) * (tot + 1)); C->lp_val = malloc(sizeof(double) * (tot + 1));
        for (int t = 0; t < C->ncon; t++)
            for (int q = 0; q < C->A[t].nnz; q++) { const int p = C->A[t].row[q]; C->lp_con[cnt[p]] = C->con[t]; C->lp_val[cnt[p]] = C->A[t].val[q]; cnt[p]++; }
        free(cnt); free(pos);
    }
}
/* X of the arrow block (n x n, full) from the cone vector x: the rank-two completion (the cone vector's order is the
 * hub first, then the other indices in increasing order: arrow_order; postsolve.c's xc_complete) */
static int brisk_solve_thr(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout);
int g_soc_used = 0;          /* 5.16: a solve of this run had cone blocks (main.c's fallback) */
static int brisk_solve_cones(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout);
/* 23 October 2026: par->xcone asks for some blocks' X as their cone vectors (brisk.h; main.c decides). The solve as a
 * cone writes them directly; a solve that does not take that route (the gate below, or a block it keeps as a semidefinite
 * block) writes the dense X into a scratch array, and its cone vector (tr X, 2 X_{hub, i}) is returned: that is the
 * projection that keeps <A_i, X> for arrow data, and it lies in the cone when X is positive semidefinite */
/* 23 October 2026 (ISSUES 241; IDEAS 22 October afternoon (c), bench/aug_5.22): the infeasible-start Newton system with
 * the NT direction solved through the augmented system, a few iterations from the best iterate, in double precision. In
 * the NT-scaled variables (W = G G', dX~ = G^-1 dX G^-T, A~_i = G' A_i G) the system is [-I, A~'; A~, 0] [dX~; dy] =
 * [Rd~ - Rt; rp], whose Schur complement A~ A~' is the solver's M: preconditioned MINRES on it, the preconditioner
 * diag(I, M^-1) with M assembled and factored at the iterate as in the loop, started from the Schur solution. The
 * Schur complement formed in double squares the condition of the scaling, and at a degenerate optimum its solution
 * misses the primal equations by far more than rounding; the augmented system's residual is computed from the
 * operator and keeps them at rounding (option_prices_example1x1_inf__r1: from BRISK's iteration 18, 2.2e-8 in seven
 * steps in double, where the loop ended at 4.1e-6). Semidefinite and LP blocks only. On by default since 23 October 2026 (BRISK_AUGEND=0: off; IMPLEMENTATION 14.3). */
/* 24 October 2026 (ISSUES 283): the phase's budget, its deadlines and its projected stops in work units (the flop
 * estimates of the double-double endgame and of the solve's work_it), not in seconds, so that its result depends on the
 * problem alone: wused the work spent so far, wbudget the budget, wstep the estimated work of one MINRES step */
typedef struct { const Problem *P; BS *S; Schur *Sc; size_t *off; size_t L; int m; double *tmp; double wused, wbudget, wstep; } AugOp;
static void aug_apply(const AugOp *o, const double *x, double *out) {      /* out = [-u + A~'v; A~ u] */
    const Problem *P = o->P; const int m = o->m, nb = P->nblk;
    double *ov = out + o->L; const double *xv = x + o->L;
    memset(ov, 0, sizeof(double) * m);
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; BS *s = &o->S[k]; const size_t len = bsz(B);
        const double *xu = x + o->off[k]; double *ou = out + o->off[k];
        nt_fromS(B, s, xu, s->Q);                           /* G u G' */
        blk_Aop(B, s, s->Q, ov, 1.0);
        blk_ATy(B, s, xv, s->F);
        nt_toZ(B, s, s->F, ou);                             /* G' (A'v) G */
        for (size_t i = 0; i < len; i++) ou[i] -= xu[i];
    }
}
static void aug_prec(const AugOp *o, const double *r, double *z) {          /* z = [r_u; M^-1 r_v] */
    memcpy(z, r, sizeof(double) * o->L);
    schur_prec(o->Sc, r + o->L, z + o->L);
}
/* preconditioned MINRES (Elman, Silvester and Wathen, Algorithm 2.4) on K x = b from x; returns the iterations. It runs to
 * rtol unless its rate so far cannot get there within maxit or the end phase's work budget; then it stops once the
 * residual is below rtarget, or at once when rtarget is out of reach as well */
static int aug_minres(const AugOp *o, const double *b, double *x, int maxit, double rtol, double rtarget, double ref,
                      double *eta0_out, double *rel_out) {   /* ref > 0: rtol and rtarget relative to it, else to the start */
    const size_t N = o->L + o->m;
    double *v0 = calloc(N, sizeof(double)), *v1 = malloc(sizeof(double) * N), *z1 = malloc(sizeof(double) * N);
    double *w0 = calloc(N, sizeof(double)), *w1 = calloc(N, sizeof(double)), *Kz = malloc(sizeof(double) * N);
    double *wn = malloc(sizeof(double) * N);
    aug_apply(o, x, Kz);
    for (size_t i = 0; i < N; i++) v1[i] = b[i] - Kz[i];
    aug_prec(o, v1, z1);
    double g1 = sqrt(fmax(0.0, ddot_n(N, z1, v1))), g0 = 1.0, eta = g1, eta0 = g1;
    const double den = ref > 0 ? ref : eta0;
    if (eta0_out) *eta0_out = eta0;
    double c0 = 1, c1 = 1, s0 = 0, s1 = 0;
    int j = 0;
    double *hist = malloc(sizeof(double) * (maxit + 2)); hist[0] = eta0;   /* |eta| by step, for the rate of the last ten */
    if (!(g1 > 0)) goto out;
    for (j = 1; j <= maxit; j++) {
        for (size_t i = 0; i < N; i++) z1[i] /= g1;
        aug_apply(o, z1, Kz);
        const double dl = ddot_n(N, Kz, z1);
        for (size_t i = 0; i < N; i++) { const double vn = Kz[i] - (dl / g1) * v1[i] - (g1 / g0) * v0[i]; v0[i] = v1[i]; v1[i] = vn; }
        double *zn = Kz;                                     /* (reused) */
        aug_prec(o, v1, zn);
        const double g2 = sqrt(fmax(0.0, ddot_n(N, zn, v1)));
        const double a0 = c1 * dl - c0 * s1 * g1, a1 = sqrt(a0 * a0 + g2 * g2), a2 = s1 * dl + c0 * c1 * g1, a3 = s0 * g1;
        if (!(a1 > 0)) break;
        const double c2 = a0 / a1, s2 = g2 / a1;
        for (size_t i = 0; i < N; i++) { wn[i] = (z1[i] - a3 * w0[i] - a2 * w1[i]) / a1; w0[i] = w1[i]; w1[i] = wn[i]; x[i] += c2 * eta * wn[i]; }
        eta = -s2 * eta;
        memcpy(z1, zn, sizeof(double) * N);
        g0 = g1; g1 = g2; c0 = c1; c1 = c2; s0 = s1; s1 = s2;
        hist[j] = fabs(eta);
        if (fabs(eta) <= rtol * den || !(g1 > 0)) break;
        /* (roa_univar_d24_I as read, m = 6077: 2e-2 after 200 steps of 25 ms each, the preconditioner a factor in single
         * precision; trto5: 4.9e-4 after 200 steps of 0.75 s, and that direction took the score from 1.2e-6 to 3.5e-8) */
        if (j >= 10) {
            /* the rate of the last ten steps: MINRES slows as it goes (trto5: 63 steps, 21 s, projected from the rate since
             * the start, before its stop) */
            const double rel = fabs(eta) / den, lr = log(fmax(fabs(eta) / fmax(hist[j - 10], 1e-300), 1e-300)) / 10;
            const double used = o->wused + j * o->wstep;                       /* (the work units of the phase so far) */
            if (used > o->wbudget) break;
            if (lr < 0) {
                const double need_t = log(rtol / rel) / lr;                    /* steps still needed at this rate */
                if (need_t > maxit - j || used + need_t * o->wstep > o->wbudget) {
                    if (rel <= rtarget) break;
                    const double need_g = log(rtarget / rel) / lr;
                    if (need_g > maxit - j || used + need_g * o->wstep > o->wbudget) break;
                }
            } else break;
        }
    }
out:
    if (rel_out) *rel_out = den > 0 ? fabs(eta) / den : 0;
    free(hist);
    free(v0); free(v1); free(z1); free(w0); free(w1); free(Kz); free(wn);
    return j;
}
/* the largest step a <= 1 with lam + a D positive definite in the scaled frame (lam diagonal): one eigenvalue problem */
static double aug_maxstep(const Block *B, const BS *s, const double *D, double *wk) {
    const int n = B->n; double a = 1e300;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) if (D[i] < 0) a = fmin(a, -s->lam[i] / D[i]); return a; }
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) wk[i + (size_t)j * n] = D[i + (size_t)j * n] / sqrt(s->lam[i] * s->lam[j]);
    double *ev = malloc(sizeof(double) * (n + 1)), wq; int lw = -1, info = 0;
    BL(dsyev_)("N", "L", &n, wk, &n, ev, &wq, &lw, &info);
    lw = (int)wq + 1; double *work = malloc(sizeof(double) * lw);
    BL(dsyev_)("N", "L", &n, wk, &n, ev, work, &lw, &info);
    if (info == 0 && ev[0] < 0) a = -1.0 / ev[0];
    free(work); free(ev);
    return info == 0 ? a : 0.0;
}
static int aug_endgame(const Problem *P, BS *S, Schur *Sc, double **Xio, double **Zio, double *yio, const Params *par,
                       double Ntot, int maxit, double wbudget, double work_it, int verbose, Result *Rio) {
    const int m = P->m, nb = P->nblk;
    for (int k = 0; k < nb; k++) if (P->blk[k].type != BLK_SDP && P->blk[k].type != BLK_LP) return 0;
    AugOp o = { P, S, Sc, malloc(sizeof(size_t) * (nb + 1)), 0, m, NULL, 0.0, wbudget, 0.0 };
    size_t maxlen = 1;
    for (int k = 0; k < nb; k++) { o.off[k] = o.L; o.L += bsz(&P->blk[k]); if (bsz(&P->blk[k]) > maxlen) maxlen = bsz(&P->blk[k]); }
    /* one MINRES step in work units: per semidefinite block the two scaled products G u G' and G'(A'v)G (four n x n
     * products, 8 n^3), the products with A and A' (2 x 2 entries of A each), and the preconditioner's two triangular
     * solves (2 x 2 entries of the Schur factor); an iteration of the phase as one iteration of the solve (work_it:
     * the Schur complement's assembly and factorization and the block algebra) */
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k]; const double nk = B->n;
        if (B->type == BLK_SDP) o.wstep += 8.0 * nk * nk * nk; else o.wstep += 2.0 * nk;
        for (int t = 0; t < B->ncon; t++) o.wstep += 4.0 * B->A[t].ef;
    }
    o.wstep += Sc->SC ? 4.0 * (double)schol_nnz(Sc->SC) : 2.0 * (double)m * m;
    /* the iteration's own block algebra beyond the solve's (the Cholesky factors of X and Z, the scaling, the step lengths
     * from eigenvalues, the candidate's measure), priced as the double-double endgame prices its own: 30 n^3 a block, 50
     * n^3 from order 512. Without it the phase took 0.7 to 3.4 times the seconds per unit of the solve on 12 files */
    double wit_blk = 0;
    for (int k = 0; k < nb; k++) { const double nk = P->blk[k].n; if (P->blk[k].type == BLK_SDP) wit_blk += (nk >= 512 ? 50.0 : 30.0) * nk * nk * nk; }
    const size_t N = o.L + m;
    /* the NT buffers of a run with the HKM direction are allocated here and released at the end */
    char *own = calloc((size_t)(nb > 0 ? nb : 1), 1);
    double **saveX = malloc(sizeof(double *) * nb), **saveZ = malloc(sizeof(double *) * nb);
    double **Xc = malloc(sizeof(double *) * nb), **Zc = malloc(sizeof(double *) * nb);
    for (int k = 0; k < nb; k++) {
        BS *s = &S[k]; const size_t len = bsz(&P->blk[k]); const int n = P->blk[k].n;
        if (!s->ntGbuf) { s->ntGbuf = amalloc(sizeof(double) * len); s->ntVt = amalloc(sizeof(double) * len); s->ntT = amalloc(sizeof(double) * len); own[k] |= 1; }
        if (!s->lam) { s->lam = amalloc(sizeof(double) * (n + 1)); own[k] |= 2; }
        saveX[k] = s->X; saveZ[k] = s->Z;
        Xc[k] = malloc(sizeof(double) * len); Zc[k] = malloc(sizeof(double) * len);
        memcpy(Xc[k], Xio[k], sizeof(double) * len); memcpy(Zc[k], Zio[k], sizeof(double) * len);
    }
    double *yc = malloc(sizeof(double) * (m + 1)), *rp = malloc(sizeof(double) * (m + 1)), *rhs = malloc(sizeof(double) * (m + 1));
    double *u = malloc(sizeof(double) * N), *bvec = malloc(sizeof(double) * N), *g = malloc(sizeof(double) * o.L);
    double *Rd = malloc(sizeof(double) * o.L), *Rt = malloc(sizeof(double) * o.L), *dX = malloc(sizeof(double) * o.L), *dZ = malloc(sizeof(double) * o.L);
    double *wk = malloc(sizeof(double) * maxlen), *dXa = malloc(sizeof(double) * o.L), *dZa = malloc(sizeof(double) * o.L);
    double *up = malloc(sizeof(double) * N), *gp = malloc(sizeof(double) * o.L);     /* the predictor's solution and g */
    /* the corrector's second-order term sym(dX~ dZ~) of the predictor's scaled steps, as in the loop (BRISK_AUGSOC=0: none) */
    static int soc2 = -1; if (soc2 < 0) { const char *e = getenv("BRISK_AUGSOC"); soc2 = e ? atoi(e) != 0 : 1; }
    memcpy(yc, yio, sizeof(double) * m);
    Result Rb = *Rio; double best = res_score(Rio); int improved = 0, bad = 0;
    const double t0 = wtime(), s0 = best;
    for (int it = 0; it < maxit; it++) {
        if ((par->timelimit > 0 && wtime() - par->t_start > par->timelimit) || o.wused > o.wbudget) break;
        const double ti0 = wtime(); double ti_m = 0, ti_k = 0;     /* the iteration's time: scaling and Schur, MINRES */
        int fail = 0;
        for (int k = 0; k < nb && !fail; k++) {
            const Block *B = &P->blk[k]; BS *s = &S[k]; const size_t len = bsz(B); const int n = B->n;
            s->X = Xc[k]; s->Z = Zc[k];
            if (B->type == BLK_SDP) {
                memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len);
                if (brisk_dpotrf(n, s->LX, n) || brisk_dpotrf(n, s->LZ, n)) { fail = 1; break; }
            } else for (int i = 0; i < n; i++) if (!(s->X[i] > 0 && s->Z[i] > 0)) { fail = 1; break; }
            if (!fail && nt_scaling(B, s)) fail = 1;
        }
        if (fail) break;
        /* residuals: rp = b - A(X), Rd = C - A'y - Z, mu */
        memcpy(rp, P->b, sizeof(double) * m);
        double xz = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k]; BS *s = &S[k]; const size_t len = bsz(B);
            blk_Aop(B, s, s->X, rp, -1.0);
            blk_ATy(B, s, yc, s->F);
            double *rd = Rd + o.off[k];
            for (size_t i = 0; i < len; i++) rd[i] = -(s->F[i] + s->Z[i]);
            sp_add(B, &B->C, 1.0, rd);
            xz += ddot_n(len, s->X, s->Z);
        }
        const double mu = xz / Ntot;
        /* the Schur complement of the NT scaling, factored: the preconditioner */
        {
            double *Ma = schur_asm_begin(Sc);
            for (int k = 0; k < nb; k++) {
                BS vtmp; BS *v = nt_view(&S[k], &vtmp, 1);
                if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], v, m, Ma); else schur_sdp(&P->blk[k], v, m, Ma);
            }
            schur_asm_end(Sc);
            if (schur_factor(Sc) != 0) break;
        }
        o.wused += work_it + wit_blk;
        ti_k = wtime() - ti0;
        double ap = 0, ad = 0, sigma = 0, rel = 0, eta_p = 0; int nit = 0, mfail = 0;
        /* the direction has to be about tol / score better than the Schur solution it starts from */
        const double rtarget = fmin(1e-2, fmax(1e-12, 0.1 * par->tol / fmax(best, 1e-300)));
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; BS *s = &S[k]; const size_t len = bsz(B); int n = B->n;
                const double *H = NULL;
                if (pass && soc2) {
                    const double *xa = dXa + o.off[k], *za = dZa + o.off[k];
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) wk[i] = xa[i] * za[i];
                    else { pdgemm("N", "N", &n, &n, &n, &DONE, xa, &n, za, &n, &DZERO, wk, &n); symmetrize(n, wk); }
                    H = wk;
                }
                nt_rhs_scaled(B, s, pass ? sigma * mu : 0.0, H, Rt + o.off[k]);
                nt_toZ(B, s, Rd + o.off[k], g + o.off[k]);                  /* Rd~ */
                for (size_t i = 0; i < len; i++) g[o.off[k] + i] -= Rt[o.off[k] + i];
            }
            /* the start: the Schur solution, dy = M^-1 (rp + A~ g), u = A~'dy - g; the corrector's from the predictor's
             * solution plus the Schur solution of the change of the right-hand side (the same matrix, a small change) */
            if (pass) for (size_t i = 0; i < o.L; i++) gp[i] = g[i] - gp[i];
            const double *gs = pass ? gp : g;
            if (pass) memset(rhs, 0, sizeof(double) * m); else memcpy(rhs, rp, sizeof(double) * m);
            for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; BS *s = &S[k]; nt_fromS(B, s, gs + o.off[k], s->Q); blk_Aop(B, s, s->Q, rhs, 1.0); }
            schur_solve(Sc, rhs, u + o.L, 1e-12);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; BS *s = &S[k]; const size_t len = bsz(B);
                blk_ATy(B, s, u + o.L, s->F); nt_toZ(B, s, s->F, u + o.off[k]);
                for (size_t i = 0; i < len; i++) u[o.off[k] + i] -= gs[o.off[k] + i];
            }
            if (pass) for (size_t i = 0; i < N; i++) u[i] += up[i];
            memcpy(bvec, g, sizeof(double) * o.L); memcpy(bvec + o.L, rp, sizeof(double) * m);
            { static int mx = -1; static double rt = -1; if (mx < 0) { const char *e = getenv("BRISK_AUGMAXIT"); mx = e ? atoi(e) : 200; e = getenv("BRISK_AUGRTOL"); rt = e ? atof(e) : 1e-12; }
              /* the corrector's targets relative to the predictor's start, so that its smaller residual counts */
              const double tm = wtime();
              nit = aug_minres(&o, bvec, u, mx, rt, rtarget, pass ? eta_p : 0, pass ? NULL : &eta_p, &rel); ti_m += wtime() - tm;
              o.wused += nit * o.wstep; }
            if (!pass) { memcpy(up, u, sizeof(double) * N); memcpy(gp, g, sizeof(double) * o.L); }
            if (!(rel <= rtarget)) { mfail = pass ? 1 : 2; if (pass == 0) break; }   /* the last iteration: the preconditioner does not hold */
            /* dX = G u G', dZ = Rd - A'dy; the steps in the scaled frame: lam + a u and lam + a (Rt - u) */
            ap = 1e300; ad = 1e300;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; BS *s = &S[k]; const size_t len = bsz(B);
                const double *uk = u + o.off[k];
                ap = fmin(ap, aug_maxstep(B, s, uk, wk));
                for (size_t i = 0; i < len; i++) wk[i] = Rt[o.off[k] + i] - uk[i];
                if (B->type == BLK_SDP) { double *t2 = malloc(sizeof(double) * len); memcpy(t2, wk, sizeof(double) * len); ad = fmin(ad, aug_maxstep(B, s, t2, wk)); free(t2); }
                else ad = fmin(ad, aug_maxstep(B, s, wk, wk + 0));
                nt_fromS(B, s, uk, dX + o.off[k]);
                blk_ATy(B, s, u + o.L, s->F);
                for (size_t i = 0; i < len; i++) dZ[o.off[k] + i] = Rd[o.off[k] + i] - s->F[i];
            }
            if (pass == 0) {
                for (size_t i = 0; i < o.L; i++) { dXa[i] = u[i]; dZa[i] = Rt[i] - u[i]; }    /* the scaled steps */
                const double a1 = fmin(1.0, ap), a2 = fmin(1.0, ad);
                double xz2 = 0;
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    for (size_t i = 0; i < len; i++) xz2 += (Xc[k][i] + a1 * dX[o.off[k] + i]) * (Zc[k][i] + a2 * dZ[o.off[k] + i]);
                }
                const double r_ = fmax(0.0, xz2 / Ntot) / mu;
                sigma = fmax(1e-3, fmin(1.0, r_ * r_ * r_));
            }
        }
        if (mfail == 2) {                                    /* no direction from the predictor */
            for (int k = 0; k < nb; k++) { S[k].X = saveX[k]; S[k].Z = saveZ[k]; }
            if (verbose > 1) printf("   augmented-system iteration %d: mu %.2e, MINRES %d steps (%.1e, %.1e wanted): stopped\n", it, mu, nit, rel, rtarget);
            break;
        }
        ap = fmin(1.0, 0.98 * ap); ad = fmin(1.0, 0.98 * ad);
        for (int k = 0; k < nb; k++) {
            const size_t len = bsz(&P->blk[k]);
            for (size_t i = 0; i < len; i++) { Xc[k][i] += ap * dX[o.off[k] + i]; Zc[k][i] += ad * dZ[o.off[k] + i]; }
        }
        for (int i = 0; i < m; i++) yc[i] += ad * u[o.L + i];
        Result Rc = *Rio;
        for (int k = 0; k < nb; k++) { S[k].X = saveX[k]; S[k].Z = saveZ[k]; }
        const double sc = measure_candidate(P, S, Xc, Zc, yc, &Rc, Sc->t);
        if (verbose > 1) printf("   augmented-system iteration %d: mu %.2e, MINRES %d steps (%.1e), steps %.3f %.3f, score %.2e (%.2f s: scaling and Schur %.2f, MINRES %.2f)\n", it, mu, nit, rel, ap, ad, sc, wtime() - ti0, ti_k, ti_m);
        if (it == 0 && sc > 10.0 * best) break;           /* the first step worsens the score tenfold (vibra4: 6.1e-8 -> 8.9e-7) */
        if (sc < best) {
            const double prev = best;
            best = sc; Rb = Rc; improved = 1; bad = 0;
            for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]); memcpy(Xio[k], Xc[k], sizeof(double) * len); memcpy(Zio[k], Zc[k], sizeof(double) * len); }
            memcpy(yio, yc, sizeof(double) * m);
            if (sc <= par->tol) break;
            /* 24 October 2026: the phase is for the tolerance; it stops when the steps still needed for it, at the faster
             * of its average rate and its last step's rate and at its time per step so far, do not fit in the budget
             * left (tiger_texture, Mittelmann, SDP: from 2.6e-3 to 4.5e-4 in 23 s, and the re-solve followed;
             * ni10fv_r4__d3, a user's frame relaxation, SDP: 5.4e-7 -> 1.3e-7 for 50 s of a 76 s solve).
             * BRISK_AUGPROJ=0: no such stop */
            static int proj = -1; if (proj < 0) { const char *e = getenv("BRISK_AUGPROJ"); proj = !(e && atoi(e) == 0); }
            const double wit = o.wused / (it + 1), left = o.wbudget - o.wused;     /* (work units: ISSUES 283) */
            const double rate = fmin(pow(best / s0, 1.0 / (it + 1)), best / prev);
            const double need = rate < 1 ? log(par->tol / best) / log(rate) : INFINITY;
            if (proj && !(need * wit <= left)) {
                if (verbose > 1) printf("   augmented-system end phase: %.1f more steps of %.2e work units needed for the tolerance at the rate so far, %.2e left: stopped\n", need, wit, left);
                break;
            }
        } else if (++bad >= 3) break;
        if (!(ap > 1e-6 || ad > 1e-6) || mfail) break;
    }
    for (int k = 0; k < nb; k++) {
        BS *s = &S[k];
        s->X = saveX[k]; s->Z = saveZ[k];
        if (own[k] & 1) { free(s->ntGbuf); free(s->ntVt); free(s->ntT); s->ntGbuf = s->ntVt = s->ntT = NULL; }
        if (own[k] & 2) { free(s->lam); s->lam = NULL; }
        free(Xc[k]); free(Zc[k]);
    }
    if (improved) { const double tt[6] = { Rio->t_setup, Rio->t_schur, Rio->t_chol, Rio->t_dense, Rio->t_step, Rio->t_total }; *Rio = Rb;
        Rio->t_setup = tt[0]; Rio->t_schur = tt[1]; Rio->t_chol = tt[2]; Rio->t_dense = tt[3]; Rio->t_step = tt[4]; Rio->t_total = tt[5]; }
    if (verbose > 1) printf("   augmented-system end phase: %s, score %.2e (%.2f s; %.3e of %.3e work units, a MINRES step %.3e, an iteration %.3e)\n", improved ? "improved" : "no improvement", best, wtime() - t0, o.wused, o.wbudget, o.wstep, work_it + wit_blk);
    free(own); free(saveX); free(saveZ); free(Xc); free(Zc); free(yc); free(rp); free(rhs); free(u); free(bvec); free(g);
    free(Rd); free(Rt); free(dX); free(dZ); free(wk); free(dXa); free(dZa); free(up); free(gp); free(o.off);
    return improved;
}

/* block k of a point in the layout of the solve's output: scaled back by P->bs, a semidefinite block symmetrized */
static void blk_out(const Problem *P, int k, const double *Xk, double *dst) {
    const size_t len = bsz(&P->blk[k]);
    if (P->blk[k].type != BLK_SDP) { for (size_t i = 0; i < len; i++) dst[i] = Xk[i] * P->bs; return; }
    const int n = P->blk[k].n;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            dst[i + (size_t)j * n] = 0.5 * (Xk[i + (size_t)j * n] + Xk[j + (size_t)i * n]) * P->bs;
}
/* an arrow block's dense X as its cone vector (the trace, then twice the hub's column without the hub) */
static void dense_to_cone(int n, int h, const double *D, double *v) {
    double tr = 0, trc = 0;
    for (int i = 0; i < n; i++) { const double d = D[(size_t)i * n + i], t = tr + d; trc += fabs(tr) >= fabs(d) ? (tr - t) + d : (d - t) + tr; tr = t; }
    v[0] = tr + trc;
    int c = 1;
    for (int i = 0; i < n; i++) if (i != h) v[c++] = 2.0 * D[(size_t)i + (size_t)h * n];
}
/* 24 October 2026 (ISSUES 266): the point the solve had before the end phase on the augmented Newton system replaced it,
 * with its Result, in the layout of the solve's output. The end phase is judged on the problem solved; the run is judged
 * on the problem as read, and the two need not agree: the duals the facial reduction removed are rebuilt from the reduced
 * dual slack, which needs a margin that a more accurate reduced point has less of. main.c takes the point
 * (brisk_alt_take) and returns it when it is the better of the two on the problem as read. brisk_solve clears the slot;
 * the solver's own retry keeps the first solve's point while it runs, and the wrappers map it as they map Xout. */
typedef struct { int have, nb; Result R; double *y; double **X; const char *what; } AltSlot;
static AltSlot g_alt;
static void alt_free(AltSlot *a) {
    if (a->X) { for (int k = 0; k < a->nb; k++) free(a->X[k]); free(a->X); }
    free(a->y);
    memset(a, 0, sizeof(*a));
}
void brisk_alt_clear(void) { alt_free(&g_alt); }
int brisk_alt_take(Result *R, double **y, double ***X, const char **what) {
    if (!g_alt.have) { alt_free(&g_alt); return 0; }
    *R = g_alt.R; *y = g_alt.y; *X = g_alt.X; *what = g_alt.what;
    memset(&g_alt, 0, sizeof(g_alt));
    return 1;
}
/* a point (blocks Xk, y in the solver's scaling) into a slot, in the layout of the output (not when the copy would not
 * fit in the memory: then the slot stays empty) */
static void slot_store(AltSlot *a, const Problem *P, double *const *Xk, const double *y, const Result *R, const char *what) {
    alt_free(a);
    const int nb = P->nblk, m = P->m;
    double need = 8.0 * m;
    for (int k = 0; k < nb; k++) need += 8.0 * bsz(&P->blk[k]);
    if (!mem_fits(2.0 * need, NULL)) return;
    a->X = calloc((size_t)nb + 1, sizeof(double *)); a->y = malloc(sizeof(double) * (m + 1)); a->nb = nb;
    if (!a->X || !a->y) { alt_free(a); return; }
    for (int k = 0; k < nb; k++) {
        a->X[k] = malloc(sizeof(double) * (bsz(&P->blk[k]) + 1));
        if (!a->X[k]) { alt_free(a); return; }
        blk_out(P, k, Xk[k], a->X[k]);
    }
    for (int i = 0; i < m; i++) a->y[i] = P->cs * P->d[i] * y[i];
    a->R = *R; a->what = what; a->have = 1;
}
static void alt_store(const Problem *P, double *const *Xk, const double *y, const Result *R, const char *what) { slot_store(&g_alt, P, Xk, y, R, what); }
/* the polishes' rejected candidates, while the polishes of the returned point run (rej_begin .. rej_end): the best of
 * them by the solver's own errors is kept; at the end the runner-up of the polishes goes into the slot above, the best
 * rejected candidate or, when a polish replaced the method's point, the better of that point and the best rejected one */
static AltSlot g_rej;
static int g_rej_on;
static double g_rej_score;
static void rej_offer(const Problem *P, double *const *Xk, const double *y, const Result *R, double score) {
    if (!g_rej_on || !(score < g_rej_score)) return;
    slot_store(&g_rej, P, Xk, y, R, "a polished point the solve had rejected");
    if (g_rej.have) g_rej_score = score;
}
static void rej_begin(void) { alt_free(&g_rej); g_rej_on = 1; g_rej_score = INFINITY; }
static void rej_end(const Problem *P, double *const *Xm, const double *ym, const Result *Rm, int polished) {
    g_rej_on = 0;
    if (polished && !(g_rej.have && g_rej_score < res_score(Rm))) {
        alt_free(&g_rej);
        alt_store(P, Xm, ym, Rm, "the method's point before the polishes");
    } else if (g_rej.have) { alt_free(&g_alt); g_alt = g_rej; memset(&g_rej, 0, sizeof(g_rej)); }
}

int brisk_solve(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout) {
    brisk_alt_clear();
    if (!par_user->xcone || !Xout) return brisk_solve_cones(P, par_user, R, yout, Xout);
    double **Xs = malloc(sizeof(double *) * (P->nblk + 1));
    for (int k = 0; k < P->nblk; k++) Xs[k] = Xout[k];
    const int cone_route = par_user->soc && !par_user->dual && !par_user->mfipm && !par_user->btrack && !par_user->bound_side && !par_user->warm_X && par_user->hsd != 0;
    int dense = 0;
    for (int k = 0; k < P->nblk; k++) {
        if (par_user->xcone[k] < 0) continue;
        if (cone_route && P->blk[k].n >= par_user->soc_minn && arrow_hub(&P->blk[k]) == par_user->xcone[k]) continue;
        const size_t n = P->blk[k].n;
        Xs[k] = calloc(n * n + 1, sizeof(double));
        if (!Xs[k]) { for (int q = 0; q < k; q++) if (Xs[q] != Xout[q]) free(Xs[q]); free(Xs); return -1; }
        dense = 1;
    }
    /* the cone route below reads par->xcone for the blocks it writes as vectors */
    const int rc = brisk_solve_cones(P, par_user, R, yout, Xs);
    if (dense) for (int k = 0; k < P->nblk; k++) if (Xs[k] != Xout[k]) {
        const int n = P->blk[k].n, h = par_user->xcone[k];
        dense_to_cone(n, h, Xs[k], Xout[k]);
        free(Xs[k]);
        if (g_alt.have) {                                    /* the earlier point, mapped the same way */
            double *v = malloc(sizeof(double) * ((size_t)n + 1));
            if (!v) { alt_free(&g_alt); continue; }
            dense_to_cone(n, h, g_alt.X[k], v);
            free(g_alt.X[k]); g_alt.X[k] = v;
        }
    }
    free(Xs);
    return rc;
}
static int brisk_solve_cones(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout) {
    if (!par_user->soc || par_user->dual || par_user->mfipm || par_user->btrack || par_user->bound_side || par_user->warm_X || par_user->hsd == 0)
        return brisk_solve_thr(P, par_user, R, yout, Xout);
    int *hub = malloc(sizeof(int) * (P->nblk + 1)), nconv = 0;
    long dsum = 0;
    for (int k = 0; k < P->nblk; k++) {
        hub[k] = P->blk[k].n >= par_user->soc_minn ? arrow_hub(&P->blk[k]) : -1;
        if (hub[k] >= 0) { nconv++; dsum += P->blk[k].n; }
    }
    if (!nconv) { free(hub); return brisk_solve_thr(P, par_user, R, yout, Xout); }
    if (par_user->verbose > 0) printf("   %d semidefinite block(s) with arrow data (orders %ld in all) solved as second-order cones\n", nconv, dsum);
    g_soc_used = 1;
    Problem Q;
    int **ord = calloc((size_t)P->nblk + 1, sizeof(int *));
    soc_problem(P, &Q, hub, ord);
    /* (a block asked for as its cone vector, par->xcone, gets it directly: Xout[k] has n entries) */
    const int *xcv = par_user->xcone;
    double **Xq = Xout ? calloc((size_t)P->nblk + 1, sizeof(double *)) : NULL;
    if (Xq) for (int k = 0; k < P->nblk; k++) Xq[k] = hub[k] >= 0 && !(xcv && xcv[k] == hub[k]) ? calloc((size_t)P->blk[k].n + 1, sizeof(double)) : Xout[k];
    const int rc = brisk_solve_thr(&Q, par_user, R, yout, Xq);
    if (Xq) {
        for (int k = 0; k < P->nblk; k++) if (hub[k] >= 0 && Xq[k] != Xout[k]) {
            xc_complete(P->blk[k].n, hub[k], Xq[k], Xout[k]); free(Xq[k]);
            if (g_alt.have) {                                /* the earlier point, mapped the same way */
                const size_t n = P->blk[k].n;
                double *D = malloc(sizeof(double) * (n * n + 1));
                if (!D) { alt_free(&g_alt); continue; }
                xc_complete((int)n, hub[k], g_alt.X[k], D);
                free(g_alt.X[k]); g_alt.X[k] = D;
            }
        }
        free(Xq);
    }
    for (int k = 0; k < P->nblk; k++) if (hub[k] >= 0) { block_free_contents(&Q.blk[k]); free(ord[k]); }
    free(Q.blk); free(ord); free(hub);
    g_am.blk0 = NULL;     /* the slot map is keyed by the block array's address: a later array may get it */
    g_has_soc = 0;
    return rc;
}
static int brisk_solve_thr(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout) {
#ifdef _OPENMP
    const int nt0 = omp_get_max_threads();
    if (nt0 > 1) {
        int nt = nt0, cap = nt0, adapt = 0;
        double w = (double)P->m * P->m * P->m / 3.0;
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            const double n = B->n;
            if (B->type == BLK_SDP) w += 12.0 * n * n * n + (double)B->ncon * n * n;
            else if (B->type == BLK_SOC && B->lp_ptr) {
                /* a second-order cone: its Schur term (schur_soc), A_c A_c' at sum_p d_p^2 (d_p the
                 * constraints on coordinate p) and the rank-two part at 3 nc^2. Counted as an LP block
                 * (n (1 + nc)), a cone of 1501 coordinates on 151 constraints read as 2.3e5 flops
                 * instead of 3.4e7, and the solve ran on one thread where 1.4.3, with the cone as a
                 * semidefinite block, used them all ("Number of threads: 1" from JuMP models with a
                 * norm constraint next to a semidefinite block). */
                double s2 = 0;
                for (int p = 0; p < B->n; p++) { const double d = B->lp_ptr[p + 1] - B->lp_ptr[p]; s2 += d * d; }
                w += s2 + 3.0 * (double)B->ncon * B->ncon;
            } else w += n * (1.0 + B->ncon);
        }
        if (w < 5e6 && !getenv("BRISK_TINYMT")) nt = cap = 1;
        else if (!getenv("BRISK_NOBUSY")) {
#ifdef __linux__
            adapt = nt_measure_start();            /* the run-queue wait can be read */
            const int fr = free_cores();           /* the count of the start; a wrong one is put right during the solve */
            if (adapt && nt > g_nt_ncore) {
                nt = g_nt_ncore;
                if (par_user->verbose > 0) printf("   threads: %d of %d (the solve may use %d cores)\n", nt, nt0, nt);
            }
            if (fr >= 1 && fr < nt) {
                nt = fr;
                if (par_user->verbose > 0) printf("   threads: %d of %d (other threads are running on this machine; the count follows the free cores)\n", nt, nt0);
            }
#endif
        }
        const int cap_prev = g_nt_cap, adapt_prev = g_nt_adapt;
        const unsigned test_prev = g_nt_test;
        g_nt_cap = cap; g_nt_adapt = adapt;
        if (getenv("BRISK_NTDBG")) printf("   [threads: start %d, limit %d, rule %s]\n", nt, cap, adapt ? "on" : "off");
        { const char *e = getenv("BRISK_THREADTEST"); g_nt_test = e && cap > 1 ? (unsigned)atoi(e) * 2654435761u + 1u : 0; if (g_nt_test) g_nt_adapt = 1; }
        if (cap > g_nt_lim) g_nt_lim = cap;
        nt_used(nt);
        if (nt < nt0) omp_set_num_threads(nt);
        brisk_log_threads(par_user->verbose);
        const int rc = brisk_solve_impl(P, par_user, R, yout, Xout);
        omp_set_num_threads(nt0);
        g_nt_cap = cap_prev; g_nt_adapt = adapt_prev; g_nt_test = test_prev;
        return rc;
    }
#endif
    brisk_log_threads(par_user->verbose);
    return brisk_solve_impl(P, par_user, R, yout, Xout);
}
static int brisk_solve_impl(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout) {
    double t0 = wtime();
    g_drift_held = 0;             /* (set by this solve's embedding run only) */
    /* direction: -1 = automatic. NT costs about 15 n^3 more dense work per SDP block per
     * iteration than HKM; use it when that is small next to the Schur factorization
     * (m^3/3) plus HKM's own dense work, i.e. on Schur-dominated problems,            */
    Params par_local = *par_user;
    const Params *par = &par_local;
    g_free_hsd = 0;
    for (int k = 0; k < P->nblk; k++) lp_rows_build(&P->blk[k]);   /* (23 October 2026: large LP blocks' products on the threads) */
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SOC) { par_local.hsd = 1; break; }   /* 5.16: cones in the embedding */
    if (par_local.hsd < 0) {
        /* auto: standard first, retry below; but free variables given as split pairs (SOS
         * models with equality multipliers, e.g. TSSOS) go to the embedding directly: the
         * infeasible method drives both halves of a pair to infinity and stalls */
        par_local.hsd = 0;
        BS *tmp = calloc(P->nblk + 1, sizeof(BS));
        FreeVars F0;
        const int np = free_detect(P, tmp, &F0);
        free_release(P, tmp, &F0);
        free(tmp);
        if (np >= 10 && np >= 0.02 * P->m) {
            par_local.hsd = 1;
            g_free_hsd = 1;
            if (par->verbose > 0) printf("   %d split free-variable pairs: using the self-dual embedding\n", np);
        } else if (par_local.hsd_first && !par_local.no_retry) {
            /* 4.20: the embedding first, unless its extra dense work per iteration (about
             * hsd_first_c n^3 per SDP block: the C and residual products, the second
             * endpoint, the correctors) is large next to the Schur factorization. The
             * standard method fails on a quarter of the general problems and its retry
             * starts from scratch; the embedding rarely needs one.                     */
            double n3 = 0, mm = P->m;
            for (int k = 0; k < P->nblk; k++)
                if (P->blk[k].type == BLK_SDP) { double nk = P->blk[k].n; n3 += nk * nk * nk; }
            const double fac = mm * mm * mm / 3.0;
            /* 5.10: ... or next to the assembly. Where the assembly of the Schur complement, not
             * its factorization, is most of an iteration (few constraints with wide rows on
             * blocks of order 250-800: a user's relaxations of structural optimization), the
             * embedding's dense work is a small overhead although it is several times m^3/3:
             * the embedding is first when that work is at most hsd_first_asm (0.5) of
             * factorization plus assembly (the route cost model of analyze_sdp_block, in
             * BLAS-3 flops; 0 for low-rank blocks). On the regression sets the ratio is
             * below 0.2 or above 0.9 (arch: 0.92-0.97, max-cut and graph partitioning: 35). */
            double asm_fl = 0;
            for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) asm_fl += P->blk[k].asm_cost;
            const double dense = par_local.hsd_first_c * n3;
            if (ENV_ON("BRISK_HSDASMDBG")) printf("   [embedding-first rule: dense %.2e, factor %.2e, assembly %.2e]\n", dense, fac, asm_fl);
            /* ... and on tiny problems (below 1e7 flops per iteration either way), where the
             * robustness of the embedding costs nothing (hinf11) */
            if (dense <= fac || dense <= par_local.hsd_first_asm * (fac + asm_fl) || fac + dense < 1e7) {
                par_local.hsd = 1;
                par_local.hsd_first_on = 1;
                if (par->verbose > 0) {
                    if (dense <= fac) printf("   self-dual embedding first (dense work %.1e <= Schur factor %.1e)\n", dense, fac);
                    else if (!(dense <= par_local.hsd_first_asm * (fac + asm_fl))) printf("   self-dual embedding first (a small problem: %.1e flops an iteration)\n", fac + dense);
                    else printf("   self-dual embedding first (dense work %.1e <= %.2g x Schur factor and assembly %.1e)\n", dense, par_local.hsd_first_asm, fac + asm_fl);
                }
            }
        }
    }
    if (par_local.hsd == 1) par_local.direction = par_local.hsd_dir == 1 ? 1 : 0;   /* HKM or NT in the embedding */
    if (par_local.direction < 0) {
        double n3 = 0, mm = P->m;
        for (int k = 0; k < P->nblk; k++)
            if (P->blk[k].type == BLK_SDP) { double nk = P->blk[k].n; n3 += nk * nk * nk; }
        /* ... or when that extra work is negligible in absolute terms (<= 1e8 flops,
         * a few ms per iteration): small problems profit from NT's better behaviour */
        par_local.direction = (15.0 * n3 < 0.25 * (mm * mm * mm / 3.0 + 5.0 * n3) || 15.0 * n3 <= 1e8) ? 1 : 0;
    }
    set_fast_fp();
    #pragma omp parallel
    set_fast_fp();
    memset(R, 0, sizeof(*R));   /* err[2], err[4] stay 0 unless a polished point is returned */
    const int m = P->m, nb = P->nblk;
    /* 5.16: second-order cone blocks: the embedding only; the polishing, the endgame and the
     * re-solve with the standard method work on semidefinite and LP blocks */
    BS *S = calloc(nb, sizeof(BS));
    int has_soc = 0;
    for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SOC) has_soc = 1;
    g_has_soc = has_soc;
    double Ntot = 0, polish_bytes = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int n = B->n;
        size_t len = bsz(B) * sizeof(double);
        s->n = n; s->type = B->type;
        double **bufs[] = { &s->X, &s->Z, &s->LX, &s->LZ, &s->Zi, &s->dX, &s->dZ,
                            &s->F, &s->G, &s->Q, &s->W };
        for (size_t q = 0; q < sizeof(bufs) / sizeof(bufs[0]); q++) *bufs[q] = amalloc(len);
        /* 5.12: the first touch of large blocks' arrays on the threads (the page faults of
         * maxG55's 11 arrays of 200 MB: 1.2-2.4 s through the serial first writes) */
        if (len >= ((size_t)8 << 20)) for (size_t q = 0; q < sizeof(bufs) / sizeof(bufs[0]); q++) pzero(len / sizeof(double), *bufs[q]);
        /* 4.25: Xn, Zn (and Ct, ev) serve the correctors only: the embedding's are allocated
         * when it runs (hsd_run), 4 n^2 fewer in the standard method (maxG60: 1.6 GB) */
        if (par->max_correctors > 0) { s->Xn = amalloc(len); s->Zn = amalloc(len); }
        if (par->direction == 1) {
            double **nb_[] = { &s->ntGbuf, &s->ntVt, &s->ntT, &s->dXt, &s->dZt, &s->Rt };
            for (size_t q = 0; q < sizeof(nb_) / sizeof(nb_[0]); q++) *nb_[q] = amalloc(len);
            if (par->nt_split) {        /* buffers of the experimental split only */
                double **sb_[] = { &s->dX1, &s->dS1, &s->dX1t, &s->dS1t };
                for (size_t q = 0; q < sizeof(sb_) / sizeof(sb_[0]); q++) *sb_[q] = amalloc(len);
            }
            s->lam = amalloc(sizeof(double) * (n + 1));
        }
        if (B->type == BLK_LP && par->max_correctors > 0) s->Dxc = amalloc(len);
        if (B->type == BLK_SOC) {                 /* 5.16: the cone's NT scaling (any direction of the SDP blocks) */
            if (!s->lam) s->lam = amalloc(sizeof(double) * (n + 1));
            s->sv = amalloc(sizeof(double) * (n + 1)); s->stmp = amalloc(sizeof(double) * (3 * (size_t)n + 1));
            soc_nt_identity(s, n);
        }
        s->R0 = amalloc(len);
        if (len >= ((size_t)8 << 20)) pzero(len / sizeof(double), s->R0);
        s->lr_mem = par->dense_mem;
        if (B->type == BLK_SDP && B->lowrank) lowrank_alloc(B, s);
        polish_bytes += 2.0 * len;
        if (B->type == BLK_SDP) {
            s->upat.nf = B->unf; s->upat.fr = B->ufr; s->upat.fc = B->ufc;
            s->upat.nr = B->unr; s->upat.rows = B->urows; s->upat.route = B->prod_route; s->upat.owned = 0;
        }
        if (B->type == BLK_SDP && B->dict) {
            const Block *vb = B->vb;
            size_t ql = (size_t)vb->n * vb->n * sizeof(double);
            BS *v = calloc(1, sizeof(BS));
            v->n = vb->n; v->type = BLK_SDP;
            v->X = amalloc(ql); v->Zi = amalloc(ql);
            if (vb->nd) {
                double cap = fmax(1.0, 256.0 * 1024 * 1024 / (8.0 * ql / sizeof(double)));
                v->chunk = (int)fmin((double)vb->nd, cap);
                v->dtmp = amalloc(sizeof(double) * vb->nd);
                v->Gd = amalloc(ql * v->chunk);
                v->Mdd = amalloc(sizeof(double) * (size_t)vb->nd * v->chunk);
            }
            s->vbs = v;
            s->dW = amalloc(sizeof(double) * (size_t)n * (B->dnv ? B->dnv : 1));
        }
        if (B->type == BLK_SDP) {
            if (B->prod_route == 0) rowbuf_reserve(s, B->unr);
            if (B->nd) {
                double cap = fmax(1.0, 256.0 * 1024 * 1024 / (8.0 * len / sizeof(double)));
                s->chunk = (int)fmin((double)B->nd, cap);
                s->dtmp = amalloc(sizeof(double) * B->nd);
                s->Gd = amalloc(len * s->chunk);
                s->Mdd = amalloc(sizeof(double) * (size_t)B->nd * s->chunk);
            }
            s->lq = amalloc(sizeof(double) * (size_t)n * (LANCZOS_K + 3));
            if (par->max_correctors > 0) { s->Ct = amalloc(len); s->Dxc = amalloc(len); s->ev = amalloc(sizeof(double) * n); }
            s->v1 = amalloc(sizeof(double) * n);
            s->v2 = amalloc(sizeof(double) * n);
        }
        Ntot += n;
    }
    /* The best iterate's X and Z are stored within polish_mem (512 MB). Above it, the cap grows with
     * the memory that is left: half of what is available after a reserve for what the solve still
     * allocates (the Schur matrix, about 1.5 m^2 doubles, and the method's dense block work arrays,
     * the retry's estimate below). Without the copies a failed solve returns the best iterate's y
     * with the final X and Z (ni35 order 3, m = 30,305: y of iteration 46 with X of iteration 56
     * gave a gap of 4.4e-3; iteration 45 had 1.3e-8). At or below 512 MB nothing changes. */
    double polish_cap = par->polish_mem;
    const char *bce = getenv("BRISK_BESTCAP");                         /* (BRISK_BESTCAP=0: the fixed cap, as until 1.5.3) */
    if (polish_bytes > polish_cap && !(bce && atoi(bce) == 0)) {
        double avail = 0, reserve = 1.5 * 8.0 * (double)m * m;
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) reserve += 12.0 * 8.0 * (double)bsz(&P->blk[k]);
        mem_fits(0, &avail);
        polish_cap = fmax(polish_cap, 0.5 * (avail - reserve));
        if (par->verbose > 0)
            printf("   best iterate: X and Z %s (%.2f GB; %.2f GB available after a reserve of %.2f GB for the solve)\n",
                   polish_bytes <= polish_cap ? "stored" : "not stored, only y", polish_bytes / 1e9, avail - reserve > 0 ? (avail - reserve) / 1e9 : 0.0, reserve / 1e9);
    }
    const int keep_best = par->polish && polish_bytes <= polish_cap;
    /* the objective candidate's copies: one more X and Z beside the best iterate's and the restoration's (review S5) */
    g_oc_fits = keep_best && (par->polish_r ? 3.0 : 2.0) * polish_bytes <= polish_cap;
    /* parallel over blocks when there are several and all are small (larger blocks get
     * more from threaded BLAS inside each block); BLAS runs single-threaded meanwhile,
     * except in the Schur factorization                                               */
    const int par_blk_prev = g_par_blk, blas_T_prev = g_blas_T;
    {
        int maxn_sdp = 0, nsdp = 0;
        for (int k = 0; k < nb; k++)
            if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > maxn_sdp) maxn_sdp = P->blk[k].n; }
        g_par_blk = par->par_blocks > 0 || (par->par_blocks < 0 && nthreads_cap() > 1 && nsdp >= 2 && maxn_sdp <= 256);
        if (nthreads_cap() <= 1) g_par_blk = 0;
        if (g_par_blk && BL(openblas_get_num_threads) && BL(openblas_set_num_threads) && g_blas_T == 0) {
            g_blas_T = BL(openblas_get_num_threads)();
            if (g_blas_T > 1) BL(openblas_set_num_threads)(1);
        }
        if (par->verbose > 1 && g_par_blk) printf("   parallel over %d blocks\n", nb);
    }
    if (getenv("BRISK_SETUPT")) printf("   [setup: blocks allocated at %.3fs]\n", wtime() - t0);
    if (keep_best)
        for (int k = 0; k < nb; k++) {
            size_t len = bsz(&P->blk[k]) * sizeof(double);
            S[k].Xb = amalloc(len); S[k].Zb = amalloc(len);
            if (len >= ((size_t)8 << 20)) { pzero(len / sizeof(double), S[k].Xb); pzero(len / sizeof(double), S[k].Zb); }
        }
    double *y = calloc(m, sizeof(double)), *dy = malloc(sizeof(double) * m);
    double *rhs = malloc(sizeof(double) * m), *rp = malloc(sizeof(double) * m);
    double *a0 = malloc(sizeof(double) * m), *h = malloc(sizeof(double) * m);
    double *qv = malloc(sizeof(double) * m);
    double *ev = malloc(sizeof(double) * m), *wv = malloc(sizeof(double) * m);
    double nbs = sqrt(ddot_n(m, P->b, P->b));
    int nrefine = 0, ncorr_total = 0, cnt_feas = 0;
    const int nt = par->direction == 1;
    double *w1 = calloc(m + 1, sizeof(double)), *y1 = calloc(m + 1, sizeof(double));
    int nsplit = 0, nsplit_fail = 0, nls_try = 0, nls_acc = 0;
    /* Deterministic choice (no timing dependence): use the cheaper predictor Lanczos when
     * step lengths are a noticeable share of the work. A level-2 flop costs about 0.7
     * cost units (5 GFlops measured) against c_blas = 0.05 for a BLAS-3 flop (70 GFlops);
     * about 120 n^2 level-2 flops per SDP block per iteration vs 4 n^3 dense work plus
     * the m^3/3 factorization.                                                           */
    int loose_pred = 0;
    {
        double l2 = 0, l3 = (double)m * m * m / 3.0;
        for (int k = 0; k < nb; k++)
            if (P->blk[k].type == BLK_SDP && P->blk[k].n > 12) {
                double nk = P->blk[k].n;
                l2 += 120.0 * nk * nk;
                l3 += 4.0 * nk * nk * nk;
            }
        /* ... and only when that work is substantial in absolute terms (n in the
         * hundreds): on small degenerate problems the cruder sigma cost accuracy */
        loose_pred = 0.7 * par->c_sparse * l2 > 0.15 * par->c_blas * l3 && l2 >= 1e7;
    }
    Schur Sc = { 0 };
    Sc.m = m;
    double tsb0 = wtime();
    Sc.SC = sparse_build(P, par, par->verbose);
    if (getenv("BRISK_SETUPT")) printf("   [setup: %.3fs before sparse_build, sparse_build %.3fs]\n", tsb0 - t0, wtime() - tsb0);
    double tps0 = wtime();
    schur_asm_setup(&Sc);
    schur_pos_setup(&Sc, P);
    if (getenv("BRISK_SETUPT")) printf("   [setup: schur_pos_setup %.3fs]\n", wtime() - tps0);
    Sc.E = Sc.SC ? NULL : envelope_build(P, par, par->verbose);
    if (getenv("BRISK_SETUPT")) printf("   [setup: envelope analysed at %.3fs]\n", wtime() - t0);
    if (Sc.E || Sc.SC) {
        Sc.e1 = malloc(sizeof(double) * m); Sc.e2 = malloc(sizeof(double) * m); Sc.e3 = malloc(sizeof(double) * m);
    } else {
        mem_check(8.0 * (double)m * (double)m, "the dense Schur complement", m);
        Sc.M = amalloc(sizeof(double) * (size_t)m * m);
        Sc.Mc = Sc.M;                                   /* 4.26: the factor in place */
        Sc.Md = malloc(sizeof(double) * (m + 1)); Sc.low_M = 1;
    }
    Sc.D = malloc(sizeof(double) * m);
    Sc.r = malloc(sizeof(double) * m);
    Sc.t = malloc(sizeof(double) * m);
    Sc.p = malloc(sizeof(double) * m);
    Sc.q = malloc(sizeof(double) * m);
    Sc.xb = malloc(sizeof(double) * m);
    Sc.fw = malloc(sizeof(float) * (m ? m : 1));
    /* 4.20: in the embedding, single precision only from m = 4000 (below, its PCG steps and
     * double-precision refactorizations cost more than it saves: roa_vdp_d8_I 5.9 -> 3.3 s) */
    Sc.allow_float = !Sc.E && (par->mixed > 0 || (par->mixed < 0 && m >= (par->hsd == 1 ? 4000 : 1000)));
    /* 4.31: between m = 1000 and 4000 in the embedding, single precision when the dense
     * factorization is the larger part of the iteration by the work model of the corrector
     * rule (a float factor is ~2.5x cheaper; where the block algebra or the assembly
     * dominates it only adds PCG steps: swissroll 66 -> 78 s, qap10 0.8 -> 1.7 s forced;
     * neu1 20 -> 17 s, rose15 29 -> 26 s, Example 8.1.3 reduced 6.0 -> 5.5 s where it pays) */
    if (!Sc.E && !Sc.SC && par->mixed < 0 && par->hsd == 1 && m >= 1000 && m < 4000) {
        double f_asm = 0, f_dense = 0, f_small = 0, f_aop = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            const double nk = B->n;
            if (B->type == BLK_SDP) { if (nk > 16) f_dense += nk * nk * nk; else f_small += nk * nk * nk + 64.0; }
            for (int t = 0; t < B->ncon; t++) { f_aop += B->A[t].ef; f_asm += 2.0 * B->A[t].ef * (B->type == BLK_SDP ? nk : 1.0); }
        }
        const double w_chol = 2.0e-11 * (double)m * m * m / 3.0;
        const double w_rest = 9.2e-11 * f_asm + 3.9e-8 * 0.5 * (double)m * m + 3.0 * (9.9e-10 * f_dense + 7.8e-9 * f_small + 2.7e-9 * f_aop);
        Sc.allow_float = w_chol >= par->mixed_frac * (w_chol + w_rest);
        if (ENV_ON("BRISK_KFDBG")) printf("   MIXDBG m %d chol %.3e rest %.3e -> float %d\n", m, w_chol, w_rest, Sc.allow_float);
    }
    { const char *e = getenv("BRISK_FLMAXPCG"); if (e) g_fl_maxpcg = atoi(e); }
    /* deterministic work estimate of one iteration (flop-equivalents): Schur
     * factorization, Schur assembly, dense block algebra. Used to budget the
     * double-double endgame without looking at the clock.                        */
    double work_it = 0;
    {
        if (Sc.SC) work_it += schol_flops(Sc.SC);
        else if (Sc.E) { for (int i = 0; i < m; i++) { double r = i - Sc.E->fst[i] + 1; work_it += r * r; } }
        else work_it += (double)m * m * m / 3.0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            double nk = B->n, asm_k = 0;
            if (B->type == BLK_SDP) work_it += 15.0 * nk * nk * nk;
            for (int t = 0; t < B->ncon; t++) asm_k += 2.0 * B->A[t].ef * (B->type == BLK_SDP ? nk : 1.0);
            /* 5.16: the assembly counted by its routes' estimated cost (problem.c), not by the cheapest
             * route: a user's frame24weight file spends 9 of 15 s in the assembly and was counted at
             * 3.6e10 flops, so the endgame's budget (its own estimate counts the row-product route)
             * allowed 0 of the 2 dd iterations that make it OPTIMAL (ISSUES 86) */
            work_it += B->type == BLK_SDP && B->asm_cost > asm_k ? B->asm_cost : asm_k;
        }
    }
    Sc.pivtol = par->pivtol;
    Sc.reg_ill = par->reg_ill;
    const int ntmode = par->direction == 1;
    R->direction = par->direction;
    /* NT split: factor the (constant) Gram matrix A A' once */
    Schur Gs = { 0 };
    int have_gram = 0;
    if (ntmode && par->nt_split && m > 0) {
        Gs.m = m;
        Gs.E = envelope_build(P, par, 0);
        if (Gs.E || 16.0 * m * m <= 2.0 * par->dense_mem) {
            if (Gs.E) { Gs.e1 = malloc(sizeof(double) * m); Gs.e2 = malloc(sizeof(double) * m); Gs.e3 = malloc(sizeof(double) * m); }
            else { Gs.M = amalloc(sizeof(double) * (size_t)m * m); Gs.Mc = Gs.M; Gs.Md = malloc(sizeof(double) * (m + 1)); Gs.low_M = 1; }
            Gs.D = malloc(sizeof(double) * m); Gs.r = malloc(sizeof(double) * m); Gs.t = malloc(sizeof(double) * m);
            Gs.p = malloc(sizeof(double) * m); Gs.q = malloc(sizeof(double) * m); Gs.xb = malloc(sizeof(double) * m);
            Gs.fw = malloc(sizeof(float) * m);
            Gs.pivtol = 0; Gs.reg_ill = par->reg_ill;
            schur_zero(&Gs);
            g_env = Gs.E;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                double *sx = s->X, *sz = s->Zi;
                double *I = s->ntT;                                   /* identity (scratch) */
                memset(I, 0, sizeof(double) * len);
                if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) I[i] = 1.0;
                else for (int i = 0; i < B->n; i++) I[i + (size_t)i * B->n] = 1.0;
                s->X = I; s->Zi = I;
                if (B->type == BLK_LP) schur_lp(B, s, m, Gs.M);
                else schur_sdp_id(B, s, m, Gs.M, 1);
                s->X = sx; s->Zi = sz;
            }
            g_env = NULL; g_schol = NULL;
            have_gram = schur_factor(&Gs) == 0;
        }
        if (par->verbose > 1) printf("   NT split: Gram matrix %s\n", have_gram ? (Gs.E ? "factored (envelope)" : "factored (dense)") : "not factored, CG");
    }
    double *lsz = NULL, *lsp = NULL, *lss = NULL, *lsy = NULL, *lst = NULL, *lsn = NULL;
    double **lsr = NULL, **lsq = NULL;
    if (ntmode && par->nt_split && par->nt_ls) {
        lsz = malloc(sizeof(double) * (m + 1)); lsp = malloc(sizeof(double) * (m + 1));
        lss = malloc(sizeof(double) * (m + 1)); lsy = malloc(sizeof(double) * (m + 1));
        lst = malloc(sizeof(double) * (m + 1)); lsn = malloc(sizeof(double) * (m + 1));
        lsr = malloc(sizeof(double *) * nb); lsq = malloc(sizeof(double *) * nb);
    }

    /* ---- starting point ---- */
    /* 4.30: the trace-bound row takes no part in the least-norm / least-squares start (its
     * tiny coefficients make the Gram matrix nearly singular and its rhs would put R/n on
     * every diagonal): its entries are zeroed here and restored after the start */
    int tb_row0 = -1;
    double **tb_sav = NULL;
    if (P->tbR != 0) {
        for (int i = 0; i < m; i++) if (P->b0[i] == P->tbR) { tb_row0 = i; break; }
        if (tb_row0 >= 0) {
            tb_sav = calloc(2 * nb + 2, sizeof(double *));
            for (int k = 0; k < nb; k++) {
                Block *B = &P->blk[k];
                for (int t = 0; t < B->ncon; t++) if (B->con[t] == tb_row0) {
                    SpSym *A = &B->A[t];
                    tb_sav[2 * k] = malloc(sizeof(double) * (A->nnz + 1)); memcpy(tb_sav[2 * k], A->val, sizeof(double) * A->nnz);
                    tb_sav[2 * k + 1] = malloc(sizeof(double) * (A->ef + 1)); memcpy(tb_sav[2 * k + 1], A->fv, sizeof(double) * A->ef);
                    /* during the start the row reads "s = 1": unit coefficient on the slack, nothing
                     * else (a well-conditioned Gram matrix; X0 and y0 are those of the problem alone) */
                    const double v = B->type == BLK_LP ? 1.0 : 0.0;
                    for (int q = 0; q < A->nnz; q++) A->val[q] = v;
                    for (int q = 0; q < A->ef; q++) A->fv[q] = v;
                }
            }
        }
    }
    if (getenv("BRISK_SETUPT")) printf("   [setup: before start at %.3fs]\n", wtime() - t0);
    start_sdpt3(P, S, y);
    if (getenv("BRISK_SETUPT")) printf("   [setup: after start at %.3fs]\n", wtime() - t0);
    /* 5.12: not for the centred start of moment relaxations (init 3), which replaces the whole least-
     * squares point below (its two Gram CG solves took 1-2.8 s on the user's (5,7) relaxations) */
    g_probe_cgcap = 0;     /* 23 October 2026: the polish probes' CG cap, set with the least-squares start below (none without it) */
    if (par->init > 0 && m > 0 && (par->init != 3 || ENV_ON("BRISK_INITLS"))) {
        /* Least-norm primal / least-squares dual on the Gram matrix A A'
         * (= Schur complement at X = Zi = I), then shift into the cone. */
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            memset(s->X, 0, sizeof(double) * len);
            memset(s->Zi, 0, sizeof(double) * len);
            if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) s->X[i] = s->Zi[i] = 1.0;
            else if (B->type == BLK_SOC) { s->X[0] = s->Zi[0] = 1.0; soc_nt_identity(s, B->n); }
            else for (int i = 0; i < B->n; i++) s->X[i + (size_t)i * B->n] = s->Zi[i + (size_t)i * B->n] = 1.0;
        }
        /* right-hand side of the dual least-squares problem: A(C) */
        memset(rhs, 0, sizeof(double) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            memset(s->W, 0, sizeof(double) * bsz(B));
            sp_add(B, &B->C, 1.0, s->W);
            blk_Aop(B, s, s->W, rhs, 1.0);
        }
        /* matrix-free CG first; the starting point needs only modest accuracy */
    if (getenv("BRISK_SETUPT")) printf("   [setup: before init CG at %.3fs]\n", wtime() - t0);
        double tcg = wtime();
        /* 17 October 2026: the CG solves A A* dy = b, the system the direct start factors (the Schur complement at
         * X = Zi = I), so its work is capped at that of the direct start: the assembly (the routes' cost model) and
         * one factorization (the sparse factor's flops, or m^3/3), against two passes over the constraint data a CG
         * step; CG is skipped when that buys fewer than 20 steps. Counted work, not time: the same start on every
         * machine. mater-5 (2,440 blocks, a factor of 1.6e8 flops): 2 x 200 steps that missed 1e-3 (1.1 s) before
         * the direct start ran anyway. BRISK_INITCGFULL=1: 200 steps as before. */
        int cg_it = 200;
        if (!getenv("BRISK_INITCGFULL")) {
            double wdir = Sc.SC ? schol_flops(Sc.SC) : (double)m * m * m / 3.0, wcg = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                if (B->type == BLK_SDP) { wdir += B->asm_cost; for (int t = 0; t < B->ncon; t++) wcg += B->A[t].ef; wcg += (double)B->nd * B->n * B->n + (double)B->n * B->n; }
                else { for (int t = 0; t < B->ncon; t++) wcg += B->A[t].nnz; wcg += B->n; }
            }
            wcg *= 2.0;
            /* 22 October 2026: in the cost model's units (problem.c): a flop of the factorization and of the assembly's
             * routes at the BLAS-3 rate (c_blas), an entry of a CG step at the scattered rate (c_sparse); counted one to
             * one, a CG step was taken as 20 times cheaper than it is (eopt_60_2000, SDP: 185 steps, 3.5 s, that missed
             * the tolerance before the direct start ran anyway; polylyap_n50_K20: 1.7 s where the direct start takes 0.1) */
            static int units = -1; if (units < 0) { const char *e = getenv("BRISK_INITUNITS"); units = e ? atoi(e) : 1; }   /* 0: counted one to one, as before */
            const double steps = wcg > 0 ? (units ? par->c_blas * wdir / (par->c_sparse * wcg) : wdir / wcg) : 1e9;
            { static int pc = -1; if (pc < 0) { const char *e = getenv("BRISK_PROBECAP"); pc = e ? atoi(e) : 1; }
              const double ps = wcg > 0 ? par->c_blas * wdir / (par->c_sparse * wcg) : 1e9;
              g_probe_cgcap = pc ? (int)fmin(1e9, fmax(20.0, ps)) : 0; }
            /* (only where the CG's full budget is work that counts, 1e9 flops or more: below, the two starts cost
             * little either way and the CG's, the more accurate, is kept; on small ill-posed problems the start
             * steers the path: hinf4 13 -> 18 iterations, gpp124-1 OPTIMAL -> reduced accuracy with the direct one) */
            if (2.0 * 200.0 * wcg >= 1e9) cg_it = steps < 20 ? 0 : steps < 200 ? (int)steps : 200;
            if (par->verbose > 1) printf("   init: CG budget %d steps (direct start %.2e flops, a CG step %.2e)\n", cg_it, wdir, wcg);
        }
        double r1 = cg_it > 0 ? gram_cg(P, S, P->b, dy, cg_it, 1e-10, Sc.r, Sc.p, Sc.q) : 1.0;
        /* (the cap is for a CG that does not converge; one whose first solve converged within it gets the full
         * budget for the second: the factored start is not the least-norm one where A A* is singular -
         * nn_n20_m100_K, 100 split free pairs: tr X0 2.9e8 against 352 and a false infeasibility verdict) */
        double r2 = (r1 < 1e-3) ? gram_cg(P, S, rhs, y, 200, 1e-10, Sc.r, Sc.p, Sc.q) : 1.0;
        int init_ok = (r1 < 1e-3 && r2 < 1e-3);
        if (par->verbose > 1) printf("   init: CG residuals %.1e %.1e (%.2fs)\n", r1, r2, wtime() - tcg);
        if (!init_ok) {
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                memset(s->X, 0, sizeof(double) * len);
                memset(s->Zi, 0, sizeof(double) * len);
                if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) s->X[i] = s->Zi[i] = 1.0;
                else if (B->type == BLK_SOC) { s->X[0] = s->Zi[0] = 1.0; soc_nt_identity(s, B->n); }
                else for (int i = 0; i < B->n; i++) s->X[i + (size_t)i * B->n] = s->Zi[i + (size_t)i * B->n] = 1.0;
            }
            {
                double *Ma = schur_asm_begin(&Sc);
                for (int k = 0; k < nb; k++) {
                    if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], &S[k], m, Ma);
                    else if (P->blk[k].type == BLK_SOC) schur_soc(&P->blk[k], &S[k], m, Ma);
                    else schur_sdp_id(&P->blk[k], &S[k], m, Ma, 1);
                }
                schur_asm_end(&Sc);
            }
            if (schur_factor(&Sc) == 0) {
                schur_solve(&Sc, P->b, dy, 1e-8);
                schur_solve(&Sc, rhs, y, 1e-8);
                init_ok = 1;
                /* 24 October 2026: the factored start is the least-norm start only if it solves A A' dy = b (X0 = A' dy is
                 * then the same for every solution dy): where A A' is singular the regularized factor can miss it by far
                 * (nn_n10_m100_K, an occupation measure, SDP, with the CG skipped by its budget: tr X0 1.5e13 against 2,
                 * an embedding that took the problem for dual infeasible and a re-solve of 21 s). Its residuals are
                 * measured, and where one misses, CG, which solves a consistent singular system to its least-norm
                 * solution, takes over with the full budget; the better of the two is kept */
                double *rv[2] = { dy, y }; const double *rr_[2] = { P->b, rhs };
                for (int t = 0; t < 2; t++) {
                    const double rf = gram_resid(P, S, rv[t], rr_[t], Sc.q);
                    if (!(rf > 1e-6)) continue;
                    double *keep = malloc(sizeof(double) * (m + 1)); memcpy(keep, rv[t], sizeof(double) * m);
                    const double rc_ = gram_cg(P, S, rr_[t], rv[t], 200, 1e-10, Sc.r, Sc.p, Sc.q);
                    if (par->verbose > 1) printf("   init: the factored start's %s residual %.1e; CG %.1e, %s\n", t ? "dual" : "primal", rf, rc_, rc_ < rf ? "kept" : "not kept");
                    if (!(rc_ < rf)) memcpy(rv[t], keep, sizeof(double) * m);
                    free(keep);
                }
            }
        }
        if (init_ok) {
            /* 4.30: the trace-bound row takes no part in the least-norm start (its rhs R would
             * put R/n on every diagonal); its slack is set to R - tr X0 in hsd_run */
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                blk_ATy(B, s, dy, s->X);
                memset(s->W, 0, sizeof(double) * bsz(B));
                sp_add(B, &B->C, 1.0, s->W);
            }
            double ax = -1e300, az = -1e300, trX = 0, trZ = 0, xz = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                int n = B->n;
                size_t len = bsz(B);
                blk_ATy(B, s, y, s->F);
                for (size_t i = 0; i < len; i++) s->Z[i] = s->W[i] - s->F[i];
                if (B->type == BLK_LP) {
                    for (int i = 0; i < n; i++) { ax = fmax(ax, -s->X[i]); az = fmax(az, -s->Z[i]); }
                } else if (B->type == BLK_SOC) {     /* the smaller spectral value x_0 - ||xb|| */
                    ax = fmax(ax, soc_nb(n, s->X) - s->X[0]); az = fmax(az, soc_nb(n, s->Z) - s->Z[0]);
                } else {
                    double lx, lz;
                    sdp_maxstep_lam(n, NULL, s->X, s, par->lanczos_k, &lx);
                    sdp_maxstep_lam(n, NULL, s->Z, s, par->lanczos_k, &lz);
                    ax = fmax(ax, -lx);
                    az = fmax(az, -lz);
                }
            }
            static double x0sx = -1, x0sz = -1;
            if (x0sx < 0) { const char *e1 = getenv("BRISK_X0SX"), *e2 = getenv("BRISK_X0SZ"); x0sx = e1 ? atof(e1) : 1.0; x0sz = e2 ? atof(e2) : 1.0; }   /* (test) the constants of the shifts */
            double shx = ax >= -1e-8 ? x0sx + ax : 0.0;
            double shz = az >= -1e-8 ? x0sz + az : 0.0;
            /* 5.12: on the dual side the shift is continuous in lambda_min: Z0 gets lambda_min >= 1
             * also when the least-squares Z is already positive definite. Before, such a Z was taken
             * unshifted, however close to singular: a user's moment relaxations with ball and box
             * constraints have lambda_min 2e-3 against a mean eigenvalue of 2e-2 there, and spent their
             * first iterations recentring (44 of them: 363 -> 326 iterations); inc_600 has 2e-6.
             * No file of the 188 of the regression list has a positive definite least-squares Z (the
             * same results). The primal side keeps its rule: a positive definite least-norm X is a
             * multiple of the identity on the max-cut and partitioning problems, and the shift cost
             * them iterations (mcp 300 -> 328, gpp 346 -> 360). BRISK_X0CONT (bits: 1 primal, 2 dual;
             * default 2, 0 the rule before) */
            { static int x0c = -1; if (x0c < 0) x0c = getenv("BRISK_X0CONT") ? atoi(getenv("BRISK_X0CONT")) : 2;
              if (x0c & 1) shx = fmax(0.0, x0sx + ax);
              if (x0c & 2) shz = fmax(0.0, x0sz + az); }
            static int x0df = -1; if (x0df < 0) x0df = getenv("BRISK_X0DFEAS") ? atoi(getenv("BRISK_X0DFEAS")) : 0;
            const int dfeas = x0df && az < -1e-8;     /* (test) the least-squares Z is strictly dual feasible: kept, X0 = mu Z0^-1 */
            if (dfeas) shz = 0.0;
            static int x0rule = -1; if (x0rule < 0) { const char *e = getenv("BRISK_X0RULE"); x0rule = e ? atoi(e) : 0; }
            if (ENV_ON("BRISK_SCALEDBG") || x0rule) {
                double tx0 = 0, tz0 = 0, nx0 = 0, nz0 = 0, nd0 = 0;
                for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n; const size_t len = bsz(B);
                    for (int i = 0; i < (B->type == BLK_SOC ? 1 : n); i++) { const size_t d = B->type != BLK_SDP ? (size_t)i : (size_t)i * n + i; tx0 += S[k].X[d]; tz0 += S[k].Z[d]; }
                    nx0 += ddot_n(len, S[k].X, S[k].X); nz0 += ddot_n(len, S[k].Z, S[k].Z); nd0 += B->type == BLK_SOC ? 1 : n; }
                if (ENV_ON("BRISK_SCALEDBG"))
                    printf("   SCALEDBG least-norm X: tr %.3e |X|F %.3e lmin %.3e; least-squares Z: tr %.3e |Z|F %.3e lmin %.3e; ndim %.0f -> shifts %.3e %.3e\n",
                           tx0, sqrt(nx0), -ax, tz0, sqrt(nz0), -az, nd0, shx, shz);
                /* (test) relative shifts: |lambda_min| plus the root mean square size of the least-norm
                 * point per dimension (factor BRISK_X0C, default 1) instead of 1 */
                if (x0rule == 2) {
                    const char *e = getenv("BRISK_X0C"); const double c = e ? atof(e) : 1.0;
                    const double rx = c * sqrt(nx0 / fmax(nd0, 1.0)), rz = c * sqrt(nz0 / fmax(nd0, 1.0));
                    shx = ax >= -1e-8 ? ax + fmax(rx, 1e-8) : 0.0;
                    shz = az >= -1e-8 ? az + fmax(rz, 1e-8) : 0.0;
                }
                /* (test) Mehrotra's rule without absolute constants: 1.5 |lambda_min| where negative,
                 * then the balancing of the second pass (both sides raised by 1/2 <X,Z>/tr) */
                if (x0rule == 3) { shx = ax > 0 ? 1.5 * ax : 0.0; shz = az > 0 ? 1.5 * az : 0.0; }
            }
            for (int pass = 0; pass < 2; pass++) {
                trX = trZ = xz = 0;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    int n = B->n;
                    for (int i = 0; i < (B->type == BLK_SOC ? 1 : n); i++) {
                        size_t d = (B->type != BLK_SDP) ? (size_t)i : (size_t)i * n + i;
                        s->X[d] += shx; s->Z[d] += shz;
                        trX += s->X[d]; trZ += s->Z[d];
                    }
                    xz += ddot_n(bsz(B), s->X, s->Z);
                }
                if ((par->init < 2 && x0rule != 3) || pass == 1) break;
                /* Mehrotra balancing: raise both sides until complementarity is centred */
                shx = 0.5 * xz / trZ;
                shz = 0.5 * xz / trX;
            }
            if (dfeas) {
                /* X0 = mu Z0^-1 with the trace of the shifted least-norm X: X0 Z0 = mu I exactly */
                double tri = 0;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k]; BS *s = &S[k]; const int n = B->n; const size_t len = bsz(B);
                    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) { s->W[i] = 1.0 / s->Z[i]; tri += s->W[i]; } continue; }
                    if (B->type == BLK_SOC) { soc_inv(n, s->Z, s->W); tri += s->W[0]; continue; }
                    memcpy(s->W, s->Z, sizeof(double) * len);
                    int info = chol_lower(n, s->W);
                    if (info == 0) { BL(dpotri_)("L", &n, s->W, &n, &info); }
                    for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) s->W[i + (size_t)j * n] = s->W[j + (size_t)i * n];
                    for (int i = 0; i < n; i++) tri += s->W[i + (size_t)i * n];
                }
                const double mu0 = trX / fmax(tri, 1e-300);
                for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]); for (size_t i = 0; i < len; i++) S[k].X[i] = mu0 * S[k].W[i]; }
                if (ENV_ON("BRISK_SCALEDBG")) printf("   SCALEDBG dual-feasible start: X0 = %.3e Z0^-1 (tr Z0^-1 %.3e)\n", mu0, tri);
            }
            if (x0rule == 1 && trX > 0 && trZ > 0) {   /* (test) equal traces, the product kept */
                const double fx = sqrt(trZ / trX), fz = 1.0 / fx;
                for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]); for (size_t i = 0; i < len; i++) { S[k].X[i] *= fx; S[k].Z[i] *= fz; } }
            }
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                if (B->type == BLK_LP) continue;
                int n = B->n;
                size_t len = bsz(B);
                if (B->type == BLK_SOC) {             /* (the shift above made both interior) */
                    for (double extra = 1e-8; !soc_interior(n, s->X); extra *= 10) s->X[0] += extra * (1.0 + s->X[0]);
                    for (double extra = 1e-8; !soc_interior(n, s->Z); extra *= 10) s->Z[0] += extra * (1.0 + s->Z[0]);
                    memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len);
                    continue;
                }
                for (double extra = 1e-8; ; extra *= 10) {
                    memcpy(s->LX, s->X, sizeof(double) * len);
                    if (chol_lower(n, s->LX) == 0) break;
                    for (int i = 0; i < n; i++) s->X[i + (size_t)i * n] += extra;
                }
                for (double extra = 1e-8; ; extra *= 10) {
                    memcpy(s->LZ, s->Z, sizeof(double) * len);
                    if (chol_lower(n, s->LZ) == 0) break;
                    for (int i = 0; i < n; i++) s->Z[i + (size_t)i * n] += extra;
                }
            }
        }
        else start_sdpt3(P, S, y);          /* Gram matrix unusable: fall back */
        Sc.nreg = 0; Sc.pcg_iters = 0; Sc.want_double = 0; Sc.nfloat = 0;
    }
    {   /* (test) BRISK_X0RULE=4: the centred start X0 = xi I, Z0 = eta I, y0 = 0 of the scaled
         * problem (BRISK_X0XI, BRISK_X0ETA, default 1: MOSEK's start) */
        const char *e = getenv("BRISK_X0RULE");
        if ((e && atoi(e) == 4) || par->init == 3) {
            const char *ex = getenv("BRISK_X0XI"), *ez = getenv("BRISK_X0ETA");
            const double xi = ex ? atof(ex) : par->init == 3 ? 0.3 : 1.0, eta = ez ? atof(ez) : 1.0;
            memset(y, 0, sizeof(double) * m);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; BS *s = &S[k]; const int n = B->n; const size_t len = bsz(B);
                memset(s->X, 0, sizeof(double) * len); memset(s->Z, 0, sizeof(double) * len);
                memset(s->Zi, 0, sizeof(double) * len);                 /* (as the least-squares start leaves it) */
                if (B->type == BLK_LP) { for (int i = 0; i < n; i++) { s->X[i] = xi; s->Z[i] = eta; s->Zi[i] = 1.0; } continue; }
                if (B->type == BLK_SOC) { s->X[0] = xi; s->Z[0] = eta; s->Zi[0] = 1.0; memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len); continue; }
                for (int i = 0; i < n; i++) s->Zi[i + (size_t)i * n] = 1.0;
                memset(s->LX, 0, sizeof(double) * len); memset(s->LZ, 0, sizeof(double) * len);
                for (int i = 0; i < n; i++) { s->X[i + (size_t)i * n] = xi; s->Z[i + (size_t)i * n] = eta; s->LX[i + (size_t)i * n] = sqrt(xi); s->LZ[i + (size_t)i * n] = sqrt(eta); }
            }
        }
    }
    if (getenv("BRISK_SETUPT")) printf("   [setup: init point done at %.3fs]\n", wtime() - t0);
    if (par->warm_X && par->warm_lam > 0) {
        /* 4.29 warm start: X0 = lam Xw + (1 - lam) X0 (and Z, y); positive definite as long
         * as Xw is positive semidefinite; checked, the cold start is kept otherwise     */
        const double lam = par->warm_lam;
        int ok = 1;
        if (ENV_ON("BRISK_WARMSCALE")) {
            /* scale-only variant: the cold start rescaled to the traces of the first point */
            double tw = 0, ts = 0, uw = 0, us = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k]; const int n = B->n;
                for (int i = 0; i < n; i++) {
                    size_t d = B->type == BLK_LP ? (size_t)i : (size_t)i * n + i;
                    tw += par->warm_X[k][d]; ts += S[k].X[d]; uw += par->warm_Z[k][d]; us += S[k].Z[d];
                }
            }
            const double fx = pow(fmax(tw, 1e-300) / ts, lam), fz = pow(fmax(uw, 1e-300) / us, lam);
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]);
                for (size_t i = 0; i < len; i++) { S[k].X[i] *= fx; S[k].Z[i] *= fz; }
            }
            if (par->verbose) printf("   scaled start: X x %.3g, Z x %.3g\n", fx, fz);
            ok = -1;
        }
        for (int k = 0; k < nb && ok == 1; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            const size_t len = bsz(B);
            for (size_t i = 0; i < len; i++) {
                s->W[i] = lam * par->warm_X[k][i] + (1 - lam) * s->X[i];
                s->F[i] = lam * par->warm_Z[k][i] + (1 - lam) * s->Z[i];
            }
            if (B->type == BLK_LP) { for (int i = 0; i < B->n; i++) if (!(s->W[i] > 0) || !(s->F[i] > 0)) ok = 0; }
            else {
                const int n = B->n;
                for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) {
                    double a = 0.5 * (s->W[i + (size_t)j * n] + s->W[j + (size_t)i * n]);
                    s->W[i + (size_t)j * n] = s->W[j + (size_t)i * n] = a;
                    a = 0.5 * (s->F[i + (size_t)j * n] + s->F[j + (size_t)i * n]);
                    s->F[i + (size_t)j * n] = s->F[j + (size_t)i * n] = a;
                }
                memcpy(s->LX, s->W, sizeof(double) * len);
                memcpy(s->LZ, s->F, sizeof(double) * len);
                if (chol_lower(n, s->LX) != 0 || chol_lower(n, s->LZ) != 0) ok = 0;
            }
        }
        if (ok == 1) {
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]);
                memcpy(S[k].X, S[k].W, sizeof(double) * len);
                memcpy(S[k].Z, S[k].F, sizeof(double) * len);
            }
            for (int i = 0; i < m; i++) y[i] = lam * par->warm_y[i] + (1 - lam) * y[i];
        }
        if (par->verbose && ok >= 0) printf("   warm start (lambda %.3g)%s\n", lam, ok ? "" : ": not positive definite, cold start");
    }
    if (tb_row0 >= 0) {
        for (int k = 0; k < nb; k++) {
            Block *B = &P->blk[k];
            for (int t = 0; t < B->ncon; t++) if (B->con[t] == tb_row0) {
                SpSym *A = &B->A[t];
                memcpy(A->val, tb_sav[2 * k], sizeof(double) * A->nnz);
                memcpy(A->fv, tb_sav[2 * k + 1], sizeof(double) * A->ef);
            }
            free(tb_sav[2 * k]); free(tb_sav[2 * k + 1]);
        }
        free(tb_sav);
        /* the bound's price starts at 0, its slack at R - tr X0 with z = mu0 / s0 */
        y[tb_row0] = 0;
        double trX0 = 0, xz0 = 0, nd0 = 0;
        int tbk = -1, tbi = -1;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            xz0 += ddot_n(bsz(B), S[k].X, S[k].Z); nd0 += B->type == BLK_SOC ? 1 : B->n;
            if (B->type == BLK_LP) {
                for (int t = 0; t < B->ncon; t++) if (B->con[t] == tb_row0 && B->A[t].nnz == 1) { tbk = k; tbi = B->A[t].row[0]; }
                continue;
            }
            if (B->type == BLK_SOC) { trX0 += S[k].X[0]; continue; }     /* (tr of the arrow block = x_0) */
            for (int i = 0; i < B->n; i++) trX0 += S[k].X[i + (size_t)i * B->n];
        }
        if (tbk >= 0) {
            const double Rs = P->b[tb_row0] / P->d[tb_row0];
            double s0 = Rs - trX0;
            if (!(s0 > 0.1 * Rs)) s0 = 0.1 * Rs;
            S[tbk].X[tbi] = s0;
            S[tbk].Z[tbi] = (xz0 / fmax(nd0, 1.0)) / s0;
            if (par->verbose > 1) printf("   trace bound row %d: slack %.3g of %.3g (tr X0 %.3g)\n", tb_row0, s0, Rs, trX0);
        }
    }
    if (getenv("BRISK_SETUPT")) printf("   [setup: before R0 at %.3fs]\n", wtime() - t0);
    /* initial dual residual: its pattern is invariant, Rd_k = theta_k * R0 */
    double theta = 1.0, r0norm = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        size_t len = bsz(B);
        blk_ATy(B, s, y, s->F);
        for (size_t i = 0; i < len; i++) s->R0[i] = -s->Z[i] - s->F[i];
        sp_add(B, &B->C, 1.0, s->R0);
        r0norm += ddot_n(len, s->R0, s->R0);
        if (B->type == BLK_SDP) setup_residual_pattern(B, s, par);   /* may drop the dense copy */
    }
    r0norm = sqrt(r0norm);
    int nrebase = 0;
    if (getenv("BRISK_SETUPT")) printf("   [setup: before zsp_setup at %.3fs]\n", wtime() - t0);
    /* 5.12: dense X, sparse Z (the standard HKM loop; the correctors use the dense factor) */
    int nzsp = 0;
    if (!nt && par->max_correctors <= 0)
        for (int k = 0; k < nb; k++)
            if (zsp_setup(&P->blk[k], &S[k], par, k)) { nzsp++; S[k].zlz = par->hsd == 1 && par->hsd_corr > 0; }
    /* 25 October 2026: a block with the sparse factor of Z does the dense algebra of X only (its Cholesky factor and the
     * step length's triangular products, about 3 n^3) and Z's through the factor (Z^-1 times an n x n matrix, 4 n nnz(L),
     * about twice an iteration), not the 15 n^3 work_it counted for it above. The 1.5.5 runs of the full benchmark, the
     * dense and step-length time of an iteration per n^3: 3.3e-11 to 3.7e-11 s on vibra5, trto5 and diamond_patch
     * (Mittelmann, SDP; factors of 2.6 % of n^2/2 on vibra5's blocks of order 1,761) against 1.4e-10 to 2.1e-10 s on
     * one-block problems of order 1,000 with a dense Z (boxqp_n1000, equalG51_s, 1zc.1024, mvu_1000): 4 to 6 times less.
     * Counted at 15 n^3, the budgets of the double-double endgame and of the end phase on the augmented Newton system
     * (dd_factor times the solve's count) came to two to four times the solve's time (vibra5: one step of the end phase
     * 68 s against a budget meant as the solve's 40 s) */
    if (nzsp) for (int k = 0; k < nb; k++) if (S[k].zon) { const double nk = P->blk[k].n; work_it += 3.0 * nk * nk * nk + 8.0 * nk * (double)schol_nnz(S[k].zf) - 15.0 * nk * nk * nk; }
    R->t_setup = wtime() - t0;

    if (par->verbose) {
        printf(" it   pobj            dobj            pinf     dinf     gap      mu       alp   ald   sigma  time\n");
    }
    int status = 3, stall = 0, it;
    double alp = 0, ald = 0, sigma = 0;
    EndRules ER_std; endrules_init(&ER_std, dd_small_problem(P, par), (par_user->hsd < 0 && !par->no_retry) ? par->retry_acc : 0.0);
    if (par->tol < 1e-8 && !getenv("BRISK_NOFLOOR")) ER_std.floor_at = fmin(1e-8, 100.0 * par->tol);
    double best_score = 1e300;
    int it_best = 0;
    Result best = *R;
    double *ybest = calloc(m, sizeof(double));
    double *slowh = (par->std_slow > 0 && par_user->hsd < 0 && !par->no_retry && par->hsd != 1)
                    ? calloc(par->maxit + 2, sizeof(double)) : NULL;
    R->methods |= par->hsd == 1 ? 2 : 1;
    if (g_free_hsd) R->free_hsd = 1;
    const double t_loop0 = wtime();
    /* the restoration's iterate: stored when the best one is and the memory allows two more copies */
    RCand rc = { NULL, 1e300, 0, 0 };
    if (keep_best && par->polish_r && 2.0 * polish_bytes <= polish_cap) {
        rc.y = malloc(sizeof(double) * (m + 1));
        for (int k = 0; k < nb; k++) {
            const size_t len = bsz(&P->blk[k]) * sizeof(double); S[k].Xr = amalloc(len); S[k].Zr = amalloc(len);
            if (len >= ((size_t)8 << 20)) { pzero(len / sizeof(double), S[k].Xr); pzero(len / sizeof(double), S[k].Zr); }
        }
    }
    g_dstop = 0;
    if (par->hsd == 1) status = hsd_run(P, par, R, S, &Sc, y, ybest, &best_score, &it, &it_best, &best, keep_best, &rc);
    else
    for (it = 0; it <= par->maxit; it++) {
        if (it > 0) brisk_threads_adapt(par->verbose);          /* 5.11: the threads of this iteration = the free cores */
        double td = wtime();
        /* inverse of Z (HKM) or the NT scaling point W (stored in Zi) */
        int nt_fail = 0;
        PBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n;
            if (nt) {
                if (nt_scaling(B, s)) {
                    _Pragma("omp atomic write")
                    nt_fail = 1;
                }
            }
            else if (B->type == BLK_LP) { for (int i = 0; i < n; i++) s->Zi[i] = 1.0 / s->Z[i]; }
            else if (s->zon) {                                  /* 5.12: Z^{-1} from the sparse factor */
                if (getenv("BRISK_ZINVOLD")) {
                    memset(s->Zi, 0, sizeof(double) * bsz(B));
                    for (int i = 0; i < n; i++) s->Zi[i + (size_t)i * n] = 1.0;
                    schol_solve_rows(s->zf, s->Zi, n, n);
                    symmetrize(n, s->Zi);
                } else schol_inverse(s->zf, s->Zi, n);
            } else {
                memcpy(s->Zi, s->LZ, sizeof(double) * bsz(B));
                (void)brisk_dpotri_full(n, s->Zi, n);           /* 5.12: in panels on the threads from n = 768 */
            }
        });
        if (nt_fail) { status = 4; break; }
        /* residuals and objectives */
        double zph = wtime(), zsc0 = R->t_schur + R->t_chol;
        if (ZT_ON()) g_zt[0] += zph - td;
        memcpy(rp, P->b, sizeof(double) * m);
        double pobj = 0, xz = 0, rdn2 = 0, nCR2 = 0, dev2 = 0;
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].X, rp, -1.0);
        PBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            if (s->zon && s->zpat) {
                /* 5.12: on the pattern of Z (the dense F is not formed; the re-base below makes it) */
                zsp_aty(B, s, y, s->zfq);
                double ncr = 0, rd = 0, dv = 0, xz_ = 0;
                for (int q = 0; q < s->nzp; q++) {
                    const size_t p_ = s->zpos[q];
                    double v = s->zfq[q] + s->Z[p_];
                    ncr += v * v;
                    v -= s->zcq[q];
                    rd += v * v;
                    const double w_ = v + theta * s->zr0q[q];
                    dv += w_ * w_;
                    xz_ += s->X[p_] * s->Z[p_];
                }
                s->part[0] = ncr; s->part[1] = sp_inner(B, &B->C, s->X); s->part[2] = xz_; s->part[3] = rd; s->part[4] = dv;
                continue;
            }
            blk_ATy(B, s, y, s->F);
            /* F <- A'y + Z - C = -Rd (true dual residual, not stored separately) */
            const double ncr = add_norm2(len, s->F, s->Z);     /* ||A'y + Z||^2 */
            sp_add(B, &B->C, -1.0, s->F);
            s->part[0] = ncr;
            s->part[1] = sp_inner(B, &B->C, s->X);
            s->part[2] = ddot_n(len, s->X, s->Z);
            s->part[3] = ddot_n(len, s->F, s->F);
            s->part[4] = norm2_plus_R0(s, len, s->F, theta);   /* ||Rd - theta R0||^2 */
        });
        for (int k = 0; k < nb; k++) {          /* sums in block order: deterministic */
            nCR2 += S[k].part[0]; pobj += S[k].part[1]; xz += S[k].part[2];
            rdn2 += S[k].part[3]; dev2 += S[k].part[4];
        }
        double dobj = ddot_n(m, P->b, y);
        double mu = xz / Ntot;
        double rpo = 0;
        for (int i = 0; i < m; i++) { double v = rp[i] * P->du[i]; rpo += v * v; }
        rpo = sqrt(rpo);
        double rdo = P->cs * sqrt(rdn2);
        double sc = P->bs * P->cs;
        double po = sc * pobj, dob = sc * dobj;
        R->pobj = po; R->dobj = dob;
        R->pinf = rpo / (1 + P->normb2);
        R->dinf = rdo / (1 + P->normC2);
        const double oden = 1 + fabs(po + P->obj_off) + fabs(dob + P->obj_off);
        R->relgap = fabs(po - dob) / oden;
        /* complementarity <X,Z> relative to the objectives: equals relgap for feasible
         * iterates, but differs when tiny infeasibilities meet huge norms            */
        R->relcomp = sc * xz / oden;
        R->err[1] = rpo / (1 + P->normb1);
        R->err[3] = rdo / (1 + P->normC1);
        R->err[5] = (po - dob) / oden;
        R->err[6] = sc * xz / oden;
        R->err[2] = R->err[4] = 0;        /* (an iterate is inside the cone; a candidate's values must not stay behind) */
        R->t_dense += wtime() - td;
        R->iters = it;
        if (par->verbose)
            printf("%3d %+.8e %+.8e %.2e %.2e %.2e %.2e %.3f %.3f %.3f %6.2f\n", it, -po, -dob,
                   R->pinf, R->dinf, R->relgap, sc * mu, alp, ald, sigma, wtime() - t0);
        dump_xz(P, S, y, it, 1.0);
        if (!isfinite(po) || !isfinite(dob)) { status = 4; break; }
        double score = res_score(R);
        if (ENV_ON("BRISK_SCOREDBG")) printf("   SCORE it %d score %.3e comp %.3e best %.3e itbest %d t %.4f\n", it, score, R->relcomp, fmin(score, best_score), score < best_score ? it : it_best, wtime() - par->t_start);
        btrack_offer(par, P, S, y, 1.0, sc * mu, R, it);
        rcand_offer(&rc, P, S, y, 1.0, R, it);
        const int improved_er = score < best_score;
        if (score < best_score) {
            best_score = score; it_best = it; best = *R;
            memcpy(ybest, y, sizeof(double) * m);
            if (keep_best)
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    pcopy(len, S[k].Xb, S[k].X);
                    if (S[k].zon && S[k].zpat && S[k].zbc) zsp_copy(&S[k], S[k].Zb, S[k].Z);   /* (zero off the pattern) */
                    else { pcopy(len, S[k].Zb, S[k].Z); S[k].zbc = 1; }
                }
        }
        if (res_score(R) < par->tol) { status = 0; break; }
        /* 5.12: as in the embedding (hsd_run): a best iterate that meets the tolerance in the
         * DIMACS measures, not improved on by this iteration, is returned */
        if (dstop_on() && keep_best && it > it_best && dimacs_score(&best) <= 0.5 * par->tol) {
            if (par->verbose > 1) printf("   the best iterate (%d) meets the tolerance in the DIMACS measures (%.1e), stopping\n", it_best, dimacs_score(&best));
            g_dstop = 1; status = 4; break;
        }
        if (time_up(par, it, t_loop0)) { status = ST_TIME; break; }
        /* infeasibility certificates (scaled quantities) */
        double nCR = sqrt(nCR2);   /* || C - Rd || = || A'y + Z || */
        if (!par->known_feasible && dobj > 0 && nCR / dobj < 1e-8 && it > 5) { status = 1; break; }
        double nAX = 0;
        for (int i = 0; i < m; i++) { double v = P->b[i] - rp[i]; nAX += v * v; }
        nAX = sqrt(nAX);
        if (!par->known_feasible && pobj < 0 && nAX / (-pobj) < 1e-8 && it > 5) { status = 2; break; }
        if (it == par->maxit) { status = 3; break; }
        if (stall >= 3) { status = 4; break; }
        if (ER_std.on) {
            const int er = endrules_check(&ER_std, it, it > 0 ? fmin(alp, ald) : 0.0, improved_er, best_score);
            if (er) {
                if (par->verbose > 1) printf("   %s (best %.1e at iteration %d), stopping\n",
                                             er == 1 ? "three collapsed steps" : er == 2 ? "creeping at short steps" : "no halving below 1e-8", best_score, it_best);
                status = 4; break;
            }
        }
        /* progress has stopped: near-optimal but not improving */
        /* stop when progress has ended (a shorter window is untested: its first
         * evaluation was confounded, see CHANGES.md) */
        /* 4.38 bound mode (A3): an improving tracked bound counts as progress */
        const int it_best_eff = btrack_it_imp(par) > it_best ? btrack_it_imp(par) : it_best;
        if (it - it_best_eff >= par->stall_win && best_score < 1e-6) { status = 4; break; }
        if (it - it_best_eff >= 8 && best_score < 1e-4) { status = 4; break; }
        if (it - it_best_eff >= 12) { status = 4; break; }          /* no progress at any level */
        /* slow progress with the embedding still to come: the best score has not halved in
         * std_slow iterations near the end; the retry (from scratch) is cheaper than the
         * tail (buck3: 60 iterations at 1e-6 .. 5e-7)                                   */
        if (slowh) {
            slowh[it] = best_score;
            if (it >= 2 * par->std_slow && best_score < par->slow_gate && best_score > par->retry_acc &&
                best_score > 0.5 * slowh[it - par->std_slow]) {
                if (par->verbose) printf("   slow progress (best %.1e, %.1e %d iterations ago): stopping for the retry\n",
                                         best_score, slowh[it - par->std_slow], par->std_slow);
                status = 4; break;
            }
        }

        /* ---- structured dual residual: use Rd = theta*R0 (exact recurrence);
         *      the true residual differs only by roundoff, else re-base ---- */
        {
            double dev = sqrt(dev2);
            if (dev > 1e-10 * (1.0 + theta * r0norm) && dev > 1e-12) {
                theta = 1.0; r0norm = sqrt(rdn2); nrebase++;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    size_t len = bsz(B);
                    if (!s->R0) s->R0 = amalloc(sizeof(double) * len);
                    if (s->zon && s->zpat) {                    /* (F was not formed: the residual pass ran on the pattern) */
                        blk_ATy(B, s, y, s->F);
                        for (size_t i = 0; i < len; i++) s->F[i] += s->Z[i];
                        sp_add(B, &B->C, -1.0, s->F);
                    }
                    for (size_t i = 0; i < len; i++) s->R0[i] = -s->F[i];
                    if (B->type == BLK_SDP) setup_residual_pattern(B, s, par);
                    if (s->zon && s->zpat && !zsp_r0(s)) s->zpat = 0;
                }
            }
            if (theta * r0norm <= 1e-13) theta = 0.0;
        }

        if (getenv("BRISK_CENTRALITY")) {
            /* centrality of the LP part: spread of x_i z_i / mu (1 = perfectly centred) */
            double lo = 1e300, hi = 0;
            for (int k = 0; k < nb; k++) {
                if (P->blk[k].type != BLK_LP) continue;
                for (int i = 0; i < P->blk[k].n; i++) {
                    double v = S[k].X[i] * S[k].Z[i] / mu;
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            if (hi > 0) printf("     [centrality: min x_i z_i / mu = %.2e, max = %.2e]\n", lo, hi);
        }

        /* ---- Schur complement ---- */
        double fact_before = R->t_schur + R->t_chol;
        double dense_before = R->t_dense + R->t_step;
        double ts = wtime();
        {
            double *Ma = schur_asm_begin(&Sc);
            for (int k = 0; k < nb; k++) {
                BS vtmp;
                BS *v = nt_view(&S[k], &vtmp, nt);
                if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], v, m, Ma);
                else schur_sdp(&P->blk[k], v, m, Ma);
            }
            schur_asm_end(&Sc);
        }
        double tc = wtime();
        R->t_schur += tc - ts;
        if (schur_factor(&Sc) != 0) { status = 4; R->t_chol += wtime() - tc; break; }
        R->t_chol += wtime() - tc;

        /* ---- right-hand-side pieces ---- */
        td = wtime();
        memset(a0, 0, sizeof(double) * m);
        memset(h, 0, sizeof(double) * m);
        const int hasH = theta > 0;
        int split = 0;
        if (nt && par->nt_split) {
            /* Jarre-Hergenroeder split: remove the primal and dual residuals with the
             * well-conditioned Gram matrix A A' first; the remaining Newton system has
             * zero feasibility rhs and only the scaled complementarity residual.     */
            double nrp = sqrt(ddot_n(m, rp, rp)), r1 = 0, r2 = 0;
            if (nrp > 0) r1 = gram_solve(P, S, &Gs, have_gram, rp, w1, &Sc);
            else memset(w1, 0, sizeof(double) * m);
            memset(qv, 0, sizeof(double) * m);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                memset(s->dS1, 0, sizeof(double) * len);
                add_R0(s, len, theta, s->dS1);
                if (hasH) blk_Aop(B, s, s->dS1, qv, 1.0);
            }
            if (hasH) r2 = gram_solve(P, S, &Gs, have_gram, qv, y1, &Sc);
            else memset(y1, 0, sizeof(double) * m);
            if (r1 <= 1e-10 && r2 <= 1e-10) {
                split = 1; nsplit++;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    size_t len = bsz(B);
                    blk_ATy(B, s, w1, s->dX1);                     /* least-norm primal correction */
                    blk_ATy(B, s, y1, s->F);
                    for (size_t i = 0; i < len; i++) s->dS1[i] -= s->F[i];   /* Rd projected on null(A) */
                    nt_toX(B, s, s->dX1, s->dX1t);
                    nt_toZ(B, s, s->dS1, s->dS1t);
                }
            } else nsplit_fail++;
        }
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS vtmp;
            BS *s = nt_view(&S[k], &vtmp, nt);
            if (!nt) blk_Aop(B, s, s->Zi, a0, 1.0);
            if (!hasH || split) continue;
            if (B->type == BLK_LP) {
                for (int i = 0; i < B->n; i++) s->G[i] = s->Zi[i] * theta * s->R0[i] * s->X[i];
                blk_Aop(B, s, s->G, h, 1.0);
            } else if (s->h_restricted) {
                Aop_ZR0X(B, s, theta, h);
            } else {
                size_t len = bsz(B);
                memset(s->dZ, 0, sizeof(double) * len);        /* dZ is free here */
                add_R0(s, len, theta, s->dZ);
                prod_pat(&s->cpat, B->n, s, s->dZ, s->X, s->G);
                blk_Aop(B, s, s->G, h, 1.0);
            }
        }
        R->t_dense += wtime() - td;

        /* ---- predictor ---- */
        if (ZT_ON()) { const double t_ = wtime(); g_zt[1] += t_ - zph - (R->t_schur + R->t_chol - zsc0); zph = t_; zsc0 = R->t_schur + R->t_chol; }
        if (nt) {
            td = wtime();
            memset(rhs, 0, sizeof(double) * m);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                nt_rhs_scaled(B, s, 0.0, NULL, s->Rt);                 /* -lam */
                if (split) {
                    for (size_t i = 0; i < len; i++) s->Rt[i] -= s->dX1t[i] + s->dS1t[i];
                    nt_fromS(B, s, s->Rt, s->G);
                    blk_Aop(B, s, s->G, rhs, -1.0);
                }
            }
            if (!split) for (int i = 0; i < m; i++) rhs[i] = P->b[i] + h[i];
            R->t_dense += wtime() - td;
            tc = wtime();
            schur_solve(&Sc, rhs, dy, 1e-7);
            R->t_chol += wtime() - tc;
            td = wtime();
            PBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                blk_ATy(B, s, dy, s->F);
                if (split) {
                    nt_toZ(B, s, s->F, s->Q);
                    for (size_t i = 0; i < len; i++) {
                        s->Q[i] = -s->Q[i];                            /* dS2t */
                        s->dZ[i] = s->dS1[i] - s->F[i];
                        s->dZt[i] = s->dS1t[i] + s->Q[i];
                        s->dXt[i] = s->Rt[i] - s->Q[i];                /* dX2t */
                    }
                    nt_fromS(B, s, s->dXt, s->G);
                    for (size_t i = 0; i < len; i++) { s->dX[i] = s->dX1[i] + s->G[i]; s->dXt[i] += s->dX1t[i]; }
                } else {
                    for (size_t i = 0; i < len; i++) s->dZ[i] = -s->F[i];
                    add_R0(s, len, theta, s->dZ);
                    nt_toZ(B, s, s->dZ, s->dZt);
                    for (size_t i = 0; i < len; i++) s->dXt[i] = s->Rt[i] - s->dZt[i];
                    if (par->nt_fast && B->type == BLK_SDP) {
                        /* G Rt G' = -G lam G' = -X, so dX = -X - W dZ W with W dZ W on the
                         * pattern of dZ (the HKM kernel with Zi = X = W) */
                        prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->Zi, s->G);
                        symmetrize(B->n, s->G);
                        for (size_t i = 0; i < len; i++) s->dX[i] = -s->G[i] - s->X[i];
                    } else nt_fromS(B, s, s->dXt, s->dX);
                }
            });
            if (split) for (int i = 0; i < m; i++) dy[i] += y1[i];
            R->t_dense += wtime() - td;
        } else {
        for (int i = 0; i < m; i++) rhs[i] = P->b[i] + h[i];
        tc = wtime();
        schur_solve(&Sc, rhs, dy, 1e-7);
        R->t_chol += wtime() - tc;
        td = wtime();
        PBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            if (s->zon && s->zpat) zsp_dz(B, s, dy, theta);      /* 5.12: on the pattern */
            else {
            blk_ATy(B, s, dy, s->F);
            PAR_LEN_FOR(len, s->dZ[i] = -s->F[i];);
            add_R0(s, len, theta, s->dZ);
            }
            if (B->type == BLK_LP) {
                for (int i = 0; i < B->n; i++) s->dX[i] = -s->Zi[i] * s->dZ[i] * s->X[i] - s->X[i];
            } else {
                /* Zi (A'dy - Rd) X in one product over the (combined) pattern */
                prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->X, s->G);
                symmetrize(B->n, s->G);
                PAR_LEN_FOR(len, s->dX[i] = -s->G[i] - s->X[i];);
            }
        });
        R->t_dense += wtime() - td;
        }
        double tst = wtime();
        if (ZT_ON()) g_zt[2] += tst - zph - (R->t_schur + R->t_chol - zsc0);
        /* Predictor step lengths only feed the centering heuristic. A looser Lanczos
         * pays off only when step lengths are a noticeable part of the work (large
         * dense blocks); otherwise the more accurate sigma saves iterations. */
        int loose = loose_pred;
        lz_tol = loose ? 0.1 : 0.02;
        int kp = loose ? (par->lanczos_k < 12 ? par->lanczos_k : 12) : par->lanczos_k;
        double zt0 = wtime();
        double ap = fmin(1.0, maxstep(P, S, 1, kp));
        double zt1 = wtime();
        double ad = fmin(1.0, maxstep(P, S, 0, kp));
        if (ZT_ON()) { g_zt[3] += zt1 - zt0; g_zt[4] += wtime() - zt1; }
        lz_tol = 0.02;
        R->t_step += wtime() - tst;
        double xdz = 0, dxz = 0, dxdz = 0;
        for (int k = 0; k < nb; k++) {
            size_t len = bsz(&P->blk[k]);
            BS *s = &S[k];
            if (s->zon && s->zpat) { xdz += zsp_dot(s, s->X, s->dZ); dxz += zsp_dot(s, s->dX, s->Z); dxdz += zsp_dot(s, s->dX, s->dZ); continue; }
            xdz += ddot_n(len, s->X, s->dZ);
            dxz += ddot_n(len, s->dX, s->Z);
            dxdz += ddot_n(len, s->dX, s->dZ);
        }
        double mu_a = (xz + ap * dxz + ad * xdz + ap * ad * dxdz) / Ntot;
        double frac = fmax(0.0, mu_a / mu);
        double mina = fmin(ap, ad);
        double expon;
        int rule = par->sigma_rule;
        if (rule < 0) {
            rule = 1;
            for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) rule = 0;
        }
        if (rule == 1) expon = 3.0;
        else if (rule == 2) expon = (mina < 0.1) ? 1.0 : 3.0;   /* cubic unless predictor stalls */
        else expon = (R->relgap > 1e-6) ? fmax(1.0, 3 * mina * mina) : 3.0;
        sigma = fmin(1.0, pow(frac, expon));
        if (sigma < par->sigma_min) sigma = par->sigma_min;
        /* Balance: do not let complementarity collapse ahead of feasibility.
         * Very small mu makes X, Z ill-conditioned and the Newton direction
         * inaccurate, so feasibility can then no longer be restored. When the
         * (relative) infeasibility exceeds the relative complementarity by more
         * than a factor `bal`, raise sigma so mu decreases no faster than the
         * infeasibility (infeasible-IPM neighbourhood condition).                */
        {
            double infeas = fmax(R->pinf, R->dinf);
            double bal = par->balance;
            if (bal > 0 && infeas > bal * R->relcomp && R->relcomp > 0) {
                double s_min = fmin(1.0, infeas / (bal * R->relcomp) * 0.1);
                if (sigma < s_min) sigma = s_min;
            }
        }
        double gamma = 0.9 + (par->gamma_max - 0.9) * mina;

        /* ---- corrector ---- */
        double relc = 0, smu = sigma * mu;
        if (nt) {
            td = wtime();
            memset(rhs, 0, sizeof(double) * m);
            PBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                int n = B->n;
                if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->W[i] = s->dXt[i] * s->dZt[i];
                else {
                    pdgemm("N", "N", &n, &n, &n, &DONE, s->dXt, &n, s->dZt, &n, &DZERO, s->W, &n);
                    symmetrize(n, s->W);                              /* sym(dXt dZt) */
                }
                nt_rhs_scaled(B, s, smu, s->W, s->Rt);
                if (split) {
                    for (size_t i = 0; i < len; i++) s->Rt[i] -= s->dX1t[i] + s->dS1t[i];
                    nt_fromS(B, s, s->Rt, s->G);
                } else {
                    /* b - A(G (Rt + lam) G') + h  (no explicit cancellation against A(X)) */
                    memcpy(s->Q, s->Rt, sizeof(double) * len);
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->Q[i] += s->lam[i];
                    else for (int i = 0; i < n; i++) s->Q[i + (size_t)i * n] += s->lam[i];
                    nt_fromS(B, s, s->Q, s->G);
                }
            });
            for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].G, rhs, -1.0);
            if (!split) for (int i = 0; i < m; i++) rhs[i] += P->b[i] + h[i];
            R->t_dense += wtime() - td;
            tc = wtime();
            relc = schur_solve(&Sc, rhs, dy, 0);
            R->t_chol += wtime() - tc;
            td = wtime();
            if (split && lsr && lhat_ok(&Sc) && R->relgap < par->nt_ls_gap) {
                /* Least-squares refinement (Jarre-Hergenroeder, eq. 33): min_y ||K y + Rt2||,
                 * K y = G' A*(y) G, by CGLS right-preconditioned with Lhat (Lhat Lhat' = K*K).
                 * Accuracy follows cond(K) instead of cond(K*K) = cond(M).                  */
                for (int k = 0; k < nb; k++) { lsr[k] = S[k].dXt; lsq[k] = S[k].dZt; }
                /* correction form: y = dy + Lhat^{-T} z, z0 = 0 (no round trip of dy
                 * through the ill-conditioned factor) */
                memset(lsz, 0, sizeof(double) * m);
                ls_k(P, S, dy, lsq);
                for (int k = 0; k < nb; k++) {
                    size_t len = bsz(&P->blk[k]);
                    for (size_t i = 0; i < len; i++) lsr[k][i] = -(S[k].Rt[i] + lsq[k][i]);
                }
                ls_kstar(P, S, lsr, lst);
                double nb0 = sqrt(ddot_n(m, lst, lst));
                double rn = 0, mdmax = 0;
                for (int k = 0; k < nb; k++) rn += ddot_n(bsz(&P->blk[k]), lsr[k], lsr[k]);
                for (int i = 0; i < m; i++) { double d = 1.0 / (Sc.D[i] * Sc.D[i]); if (d > mdmax) mdmax = d; }
                if (ENV_ON("BRISK_DEBUG"))
                    printf("     [NT LS: |K*r| %.2e vs eps*|K|*|r| %.2e (|r| %.2e, |K| ~ %.2e)]\n",
                           nb0, 2.2e-16 * sqrt(mdmax * m) * sqrt(rn), sqrt(rn), sqrt(mdmax * m));
                lhat_solveN(&Sc, lst, lss);
                memcpy(lsp, lss, sizeof(double) * m);
                double gam = ddot_n(m, lss, lss);
                for (int itl = 0; itl < par->nt_ls && gam > 0; itl++) {
                    lhat_solveT(&Sc, lsp, lsy);
                    ls_k(P, S, lsy, lsq);
                    double qq = 0;
                    for (int k = 0; k < nb; k++) qq += ddot_n(bsz(&P->blk[k]), lsq[k], lsq[k]);
                    if (!(qq > 0)) break;
                    double al = gam / qq;
                    for (int i = 0; i < m; i++) lsz[i] += al * lsp[i];
                    for (int k = 0; k < nb; k++) {
                        size_t len = bsz(&P->blk[k]);
                        for (size_t i = 0; i < len; i++) lsr[k][i] -= al * lsq[k][i];
                    }
                    ls_kstar(P, S, lsr, lst);
                    lhat_solveN(&Sc, lst, lss);
                    double gn = ddot_n(m, lss, lss);
                    for (int i = 0; i < m; i++) lsp[i] = lss[i] + (gn / gam) * lsp[i];
                    gam = gn;
                }
                lhat_solveT(&Sc, lsz, lsn);
                for (int i = 0; i < m; i++) lsn[i] += dy[i];
                ls_k(P, S, lsn, lsq);                                   /* true residual check */
                for (int k = 0; k < nb; k++) {
                    size_t len = bsz(&P->blk[k]);
                    for (size_t i = 0; i < len; i++) lsr[k][i] = -(S[k].Rt[i] + lsq[k][i]);
                }
                ls_kstar(P, S, lsr, lst);
                double nb1 = sqrt(ddot_n(m, lst, lst));
                if (ENV_ON("BRISK_DEBUG"))
                    printf("     [NT least squares: |K* r| %.2e -> %.2e %s]\n", nb0, nb1, nb1 < nb0 ? "accepted" : "rejected");
                if (nb1 < nb0) { memcpy(dy, lsn, sizeof(double) * m); nls_acc++; }
                nls_try++;
            }
            PBLK_FOR(k, nb, {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                blk_ATy(B, s, dy, s->F);
                if (split) {
                    nt_toZ(B, s, s->F, s->Q);
                    for (size_t i = 0; i < len; i++) {
                        s->dZ[i] = s->dS1[i] - s->F[i];
                        s->W[i] = s->Rt[i] + s->Q[i];                  /* dX2t = Rt2 - dS2t */
                    }
                    nt_fromS(B, s, s->W, s->G);
                    for (size_t i = 0; i < len; i++) s->dX[i] = s->dX1[i] + s->G[i];
                } else {
                    for (size_t i = 0; i < len; i++) s->dZ[i] = -s->F[i];
                    add_R0(s, len, theta, s->dZ);
                    if (par->nt_fast && B->type == BLK_SDP) {
                        /* s->G still holds G (Rt + lam) G' from the rhs, and G lam G' = X:
                         * dX = G Rt G' - W dZ W = s->G - X - W dZ W                        */
                        prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->Zi, s->Q);
                        symmetrize(B->n, s->Q);
                        for (size_t i = 0; i < len; i++) s->dX[i] = s->G[i] - s->X[i] - s->Q[i];
                    } else {
                        nt_toZ(B, s, s->dZ, s->Q);
                        for (size_t i = 0; i < len; i++) s->W[i] = s->Rt[i] - s->Q[i];
                        nt_fromS(B, s, s->W, s->dX);
                    }
                }
            });
            if (split) for (int i = 0; i < m; i++) dy[i] += y1[i];
            R->t_dense += wtime() - td;
        } else {
        td = wtime();
        zph = td;
        memset(qv, 0, sizeof(double) * m);
        PBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) s->Q[i] = s->Zi[i] * s->dZ[i] * s->dX[i];
            else if (s->zon && zmerge_on()) {
                /* 5.12: A(Zi dZp dXp) from the dense Zi; the product goes into the corrector's */
                zsp_compact(B->n, s, s->dZ, s->zdp);
                aop_zifr(B->n, s, s->zdp, s->dX, s->Q);
            }
            else prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->dX, s->Q);
        });
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].Q, qv, 1.0);
        for (int i = 0; i < m; i++) rhs[i] = P->b[i] - smu * a0[i] + qv[i] + h[i];
        R->t_dense += wtime() - td;
        tc = wtime();
        if (ZT_ON()) g_zt[5] += tc - zph;
        relc = schur_solve(&Sc, rhs, dy, 0);
        if (ENV_ON("BRISK_DEBUG"))
            printf("     [corrector solve rel res %.1e, reg %.1e, min pivot %.1e, pcg total %d, |rhs| %.2e, |dy| %.2e]\n",
                   relc, Sc.reg, Sc.min_pivot, Sc.pcg_iters, sqrt(ddot_n(m, rhs, rhs)), sqrt(ddot_n(m, dy, dy)));
        R->t_chol += wtime() - tc;
        td = wtime();
        PBLK_FOR(k, nb, {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            if (s->zon && s->zpat) zsp_dz(B, s, dy, theta);      /* 5.12: on the pattern */
            else {
            blk_ATy(B, s, dy, s->F);
            PAR_LEN_FOR(len, s->dZ[i] = -s->F[i];);
            add_R0(s, len, theta, s->dZ);
            }
            if (B->type == BLK_LP) {
                for (int i = 0; i < B->n; i++)
                    s->dX[i] = smu * s->Zi[i] - s->Zi[i] * s->dZ[i] * s->X[i] - s->Q[i] - s->X[i];
            } else if (s->zon && zmerge_on()) {                     /* Zi (dZc X + dZp dXp) */
                _Pragma("omp atomic")
                cnt_zfr++;
                prod_zsp2(B->n, s, s->dZ, s->X, s->zdp, s->dX, s->G);
                PAR_LEN_FOR(len, s->dX[i] = -s->G[i];);
                symmetrize(B->n, s->dX);
                PAR_LEN_FOR(len, s->dX[i] += smu * s->Zi[i] - s->X[i];);
            } else {
                prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->X, s->G);
                PAR_LEN_FOR(len, s->dX[i] = -s->G[i] - s->Q[i];);
                symmetrize(B->n, s->dX);
                PAR_LEN_FOR(len, s->dX[i] += smu * s->Zi[i] - s->X[i];);
            }
        });
        R->t_dense += wtime() - td;
        }

        if (ZT_ON()) { const double t_ = wtime(); g_zt[6] += t_ - td; zph = t_; zsc0 = R->t_schur + R->t_chol; }
        /* ---- refinement of the full Newton system: enforce A(dX) = rp ----
         * Near a degenerate optimum the rhs is a sum of O(1/mu) terms that
         * cancel, so A(dX) can miss rp by far more than the Schur residual.  */
        for (int pass = 0; pass < 3; pass++) {
            td = wtime();
            memcpy(ev, rp, sizeof(double) * m);
            for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].dX, ev, -1.0);
            double ne = sqrt(ddot_n(m, ev, ev)), nrp = sqrt(ddot_n(m, rp, rp));
            R->t_dense += wtime() - td;
            if (!(ne > 1e-2 * nrp && ne > 1e-3 * par->tol * (1 + nbs))) break;
            tc = wtime();
            cnt_refine_try++;
            schur_solve(&Sc, ev, wv, 0);
            R->t_chol += wtime() - tc;
            td = wtime();
            /* tentative correction: G = sym(Zi A'w X); accept only if it cuts ||e|| */
            memcpy(qv, ev, sizeof(double) * m);
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS vtmp;
                BS *s = nt_view(&S[k], &vtmp, nt);
                blk_ATy(B, s, wv, s->Q);
                if (B->type == BLK_LP) {
                    for (int i = 0; i < B->n; i++) s->G[i] = s->Zi[i] * s->Q[i] * s->X[i];
                } else {
                    prod_ZFR(B, s, s->Q, s->X, s->G);
                    symmetrize(B->n, s->G);
                }
                blk_Aop(B, s, s->G, qv, -1.0);
            }
            double ne2 = sqrt(ddot_n(m, qv, qv));
            R->t_dense += wtime() - td;
            if (ENV_ON("BRISK_DEBUG"))
                printf("     [refine pass %d: |rp| %.2e |e| %.2e -> %.2e (solve rel %.1e, reg %.0e) %s]\n",
                       pass, nrp, ne, ne2, Sc.worst_rel, Sc.reg, ne2 < 0.5 * ne ? "accepted" : "rejected");
            if (!(ne2 < 0.5 * ne)) break;
            for (int i = 0; i < m; i++) dy[i] += wv[i];
            for (int k = 0; k < nb; k++) {
                BS *s = &S[k];
                size_t len = bsz(&P->blk[k]);
                for (size_t i = 0; i < len; i++) { s->dZ[i] -= s->Q[i]; s->dX[i] += s->G[i]; }
            }
            nrefine++;
        }
        if (ZT_ON()) g_zt[7] += wtime() - zph - (R->t_schur + R->t_chol - zsc0);
        /* ---- multiple centrality correctors (only when factorization dominates) ---- */
        tst = wtime();
        zt0 = wtime();
        double apr = fmin(1.0, maxstep(P, S, 1, par->lanczos_k));
        zt1 = wtime();
        double adr = fmin(1.0, maxstep(P, S, 0, par->lanczos_k));
        if (ZT_ON()) { g_zt[8] += zt1 - zt0; g_zt[9] += wtime() - zt1; }
        R->t_step += wtime() - tst;
        double it_fact = (R->t_schur + R->t_chol) - fact_before;
        double it_dense = (R->t_dense + R->t_step) - dense_before;
        int ncorr = 0;
        if (!nt && par->max_correctors > 0 && it_fact > 8.0 * it_dense && fmin(apr, adr) < 0.9 && R->relgap > 1e-4) {
            for (int c = 0; c < par->max_correctors; c++) {
                if (!centrality_corrector(P, S, &Sc, par, smu, dy, &apr, &adr, ev, wv, R)) break;
                ncorr++;
                if (fmin(apr, adr) >= 0.95) break;
            }
            ncorr_total += ncorr;
        }

        /* ---- step ---- */
        tst = wtime();
        ap = fmin(1.0, gamma * apr);
        ad = fmin(1.0, gamma * adr);
        /* ---- feasibility guard ----
         * Late in a run on degenerate problems the Schur complement becomes nearly
         * singular, |dy| explodes, and the direction's own error (rp - A(dX)) grows
         * beyond rp itself. Taking a full primal step then destroys feasibility: on the
         * 793-bus AC-OPF relaxation pinf jumped from 1.6e-10 to 5e-1, wasting every
         * later iteration. Cap the primal step so the predicted residual cannot grow by
         * more than a factor `feas_grow` once feasibility is already good.            */
        /* Catastrophe guard (always on): the direction's own error e = rp - A(dX) is
         * normally below |rp|; when it is 1000 times larger the Schur solve has failed,
         * and a full step throws away the feasibility reached so far (the 793-bus AC-OPF:
         * pinf 1.4e-9 -> 0.49 at iteration 40, then 12 wasted iterations). Cap the primal
         * step so the residual grows at most 1000-fold and stays below 1e-6 relative;
         * the run then ends with its best iterate instead (or recovers).                 */
        if (par->feas_grow <= 0 && R->pinf < 1e-6) {
            td = wtime();
            memcpy(ev, rp, sizeof(double) * m);
            for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].dX, ev, -1.0);
            double nrp = sqrt(ddot_n(m, rp, rp)), nev = sqrt(ddot_n(m, ev, ev));
            R->t_dense += wtime() - td;
            /* allowed growth: 1000-fold, and never below a relative residual of 1e-6 */
            const double g = fmax(1e3, 1e-6 / fmax(R->pinf, 1e-300));
            const double lim = g * nrp;
            if (nev > lim && nrp > 0) {
                double cap = lim / nev;
                if (cap < ap) { ap = cap; cnt_feas++; }
                if (Sc.allow_float) Sc.want_double = 1;
            }
        }
        if (par->feas_grow > 0 && R->pinf < 1e-6) {
            td = wtime();
            memcpy(ev, rp, sizeof(double) * m);
            for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].dX, ev, -1.0);
            double nrp = sqrt(ddot_n(m, rp, rp));
            R->t_dense += wtime() - td;
            for (int tries = 0; tries < 12; tries++) {
                double pred = 0;
                for (int i = 0; i < m; i++) { double v = (1 - ap) * rp[i] + ap * ev[i]; pred += v * v; }
                if (sqrt(pred) <= par->feas_grow * nrp || ap < 0.05) break;
                ap *= 0.5;
                cnt_feas++;
            }
        }

        zt0 = wtime();
        int okp = try_step(P, S, 1, &ap);
        zt1 = wtime();
        int okd = try_step(P, S, 0, &ad);
        if (ZT_ON()) { g_zt[10] += zt1 - zt0; g_zt[11] += wtime() - zt1; }
        R->t_step += wtime() - tst;
        if (!okp || !okd) { status = 4; break; }
        for (int i = 0; i < m; i++) y[i] += ad * dy[i];
        theta *= (1.0 - ad);
        alp = ap; ald = ad;
        stall = (ap < 1e-6 && ad < 1e-6) ? stall + 1 : 0;
    }
    const double work_run = work_it * (it + 1);
    for (int k = 0; k < nb; k++) S[k].zon = 0;     /* polishing and endgame: the dense Zi and factors */
    int xsrc = 0;                 /* returned X: 0 S.X, 1 S.Xb, 2 S.G (polished) */
    const int objcand = g_objcand; g_objcand = 0;
    int ran_dd = 0;
    if (g_epol_stop) {
        /* 4.27: the embedding stopped on a passing polish probe: its candidate (the
         * Euclidean polish of the final iterate, in s->G, s->Q; y = y/tau) is returned */
        g_epol_stop = 0;
        const double tt[6] = { R->t_setup, R->t_schur, R->t_chol, R->t_dense, R->t_step, R->t_total };
        *R = g_epol_R;
        R->t_setup = tt[0]; R->t_schur = tt[1]; R->t_chol = tt[2]; R->t_dense = tt[3]; R->t_step = tt[4]; R->t_total = tt[5];
        R->iters = it;
        xsrc = 2;
        memcpy(y, ybest, sizeof(double) * m);
        const double acc = fmax(res_score(R), fmax(R->err[2], R->err[4]));
        status = acc < par->tol ? 0 : 4;
        if (par->verbose > 0) printf("   polishing: score %.2e (the probe's candidate)\n", acc);
        if (g_epol_fb && status == 4 && keep_best) {
            /* a fallback candidate above the tolerance: it becomes the best iterate, so the
             * endgame and the polishing start from it and it is returned if they fail
             * (they use s->G, s->Q as scratch)                                          */
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]) * sizeof(double);
                memcpy(S[k].Xb, S[k].G, len); memcpy(S[k].Zb, S[k].Q, len);
            }
            best = *R; best_score = res_score(R);
            memcpy(ybest, y, sizeof(double) * m);
            xsrc = 1;
        }
    }
    g_epol_fb = 0;
    if (status != 0 && status != 1 && status != 2) {
        /* fall back to the best iterate seen */
        double cur = res_score(R);
        int mixed = 0;
        if (best_score < cur) {
            double tt[5] = { R->t_setup, R->t_schur, R->t_chol, R->t_dense, R->t_step };
            *R = best;
            R->t_setup = tt[0]; R->t_schur = tt[1]; R->t_chol = tt[2]; R->t_dense = tt[3]; R->t_step = tt[4];
            memcpy(y, ybest, sizeof(double) * m);
            if (keep_best) xsrc = 1;
            else mixed = 1;       /* X and Z of the best iterate were not stored */
            if (par->verbose) printf("   returning best iterate (iteration %d)%s\n", it_best,
                                     mixed ? " (y only: X, Z of the final iterate)" : "");
        }
        /* ---- high-precision endgame: small problems that missed the tolerance.
         * Budgeted by estimated work, not by the clock: it may cost at most dd_factor
         * times what the double-precision solve cost so far, or dd_minwork. ---- */
        /* 24 October 2026 (ISSUES 279): by the clock too, not by the status alone: a loop that ended for another reason
         * after the limit (kmeans_n240_k12, adversarial, SDP: reduced accuracy at the iteration that reached 901 s of a
         * 900 s limit) ran the polishes after it, 90 s of Gram solves on 28,921 rows. Every use below skips work done
         * after the loop (the double-double endgame, the end phase on the augmented system, the polishes); without a limit
         * and an interrupt nothing changes */
        const int timeout = status == ST_TIME || brisk_time_up(par);
        /* 20 October 2026: not after the drift guard held a stop back (BRISK_DRIFT=1): the continuation's iterate has the
         * value the guard waited for, and the endgame on it cost the most of the guard's time (qap9 0.75 -> 3.1 s, control8
         * 1.3 -> 2.3 s; the values the same or near). BRISK_DRIFTDD=1: the endgame as before */
        static int drift_dd = -1; if (drift_dd < 0) { const char *e = getenv("BRISK_DRIFTDD"); drift_dd = e ? atoi(e) != 0 : 0; }
        const int skip_dd_drift = g_drift_held && !drift_dd;
        if (skip_dd_drift && par->verbose > 1 && res_score(R) >= par->tol) printf("   double-double endgame skipped: the drift guard held the stop back\n");
        int dd_gated = 0;             /* the double-double endgame skipped by its gates (constraints, memory, budget) */
        if (par->dd_end && !par->dd_done && !mixed && !timeout && status != 3 && res_score(R) >= par->tol && !g_dstop && !has_soc && !skip_dd_drift) {
            double n2 = 0;
            for (int k = 0; k < nb; k++) n2 += (double)bsz(&P->blk[k]);
            /* one dd iteration: Schur assembly through G_t = X A_t Zi (n ef_t + n^2 r_t per
             * constraint plus the inner products), the dd Cholesky, and ~30 n^3 of block
             * algebra and step search; a dd operation counts as 45 flops                   */
            double assem = 0, n3 = 0, sum_ef = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                double nk = B->n;
                for (int t = 0; t < B->ncon; t++) {
                    sum_ef += B->A[t].ef;
                    if (B->type == BLK_SDP) assem += nk * B->A[t].ef + nk * nk * B->A[t].nr;
                }
                /* (21 October 2026, ISSUES 228: a block of order >= 512 has its steps from eigenvalues (ddend.c, DD_EIG_N) and
                 * its products on the threads, but they cost more per n^3 than the small blocks' measured rate: 50 n^3 from
                 * shmup5, four blocks of order 1,861, 37 s a block and an iteration at the solve's 1.7e11 units per second) */
                if (B->type == BLK_SDP) n3 += (nk >= 512 ? 50.0 : 30.0) * nk * nk * nk;
            }
            assem += 0.5 * (double)m * sum_ef;
            /* 5.9: the endgame factors the Schur complement as a sparse matrix (ddend.c): the work
             * of the double solver's sparse factorization when there is one, 4 m^2 for the dense
             * passes over M */
            const double chol_fl = Sc.SC ? schol_flops(Sc.SC) + 4.0 * (double)m * m : (double)m * m * m / 3.0;
            /* 5.16: a dd operation counted as 20 flops (was 45 until the assembly on the live entries, the SIMD
             * column products and the transposed gather; measured on nine files, 14 October: the assembly runs
             * at 1.4-5 s per 1e9 of its units, the block algebra at 2 s, against a solve counted at ~1e10 flops/s) */
            const double dd_it = 20.0 * (assem + chol_fl + n3);
            if (getenv("BRISK_DDTIME")) printf("   [endgame cost terms: assembly %.3e, factor %.3e, blocks %.3e]\n", assem, chol_fl, n3);
            const double budget = fmax(par->dd_minwork, par->dd_factor * (par->work_done + work_run));
            int dd_maxit = (int)fmin((double)par->dd_iters, floor(budget / dd_it));
            if (par->verbose > 1)
                printf("   endgame: %.1e flops per dd iteration, budget %.1e -> %d iterations allowed\n",
                       dd_it, budget, dd_maxit);
            /* (a gate of two iterations was measured on 14 October: a user's frame24weight becomes OPTIMAL after
             * two, but the sets pay 5 % for two-iteration endgames that bring nothing: three stays) */
            const double n2cap = dd_n2_cap(par);
            if (par->verbose > 0 && !(m <= par->dd_maxm && n2 <= n2cap && dd_maxit >= 3))
                printf("   double-double endgame skipped: %s (m %d, sum of the squared block orders %.1e, %d iterations within the budget; -ddmaxm %d, -ddmaxn2 %.0e, -ddfactor %g)\n",
                       m > par->dd_maxm ? "too many constraints" : n2 > n2cap ? "blocks too large for the memory" : "fewer than three iterations within the work budget",
                       m, n2, dd_maxit, par->dd_maxm, n2cap, par->dd_factor);
            if (!(m <= par->dd_maxm && n2 <= n2cap && dd_maxit >= 3)) dd_gated = 1;
            if (m <= par->dd_maxm && n2 <= n2cap && dd_maxit >= 3) {
                int use_best = keep_best && best_score <= res_score(R) * (1 + 1e-12);
                double **Xs = malloc(sizeof(double *) * nb), **Zs = malloc(sizeof(double *) * nb);
                double *yc = malloc(sizeof(double) * (m + 1));
                memcpy(yc, use_best ? ybest : y, sizeof(double) * m);
                /* 5.16: the endgame's own buffers (S.G / S.Q are the polish's output: the candidates
                 * are compared after a polish below) */
                double **Xpre = malloc(sizeof(double *) * nb), **Zpre = malloc(sizeof(double *) * nb);
                for (int k = 0; k < nb; k++) {
                    size_t len = bsz(&P->blk[k]) * sizeof(double);
                    Xpre[k] = use_best ? S[k].Xb : S[k].X; Zpre[k] = use_best ? S[k].Zb : S[k].Z;
                    Xs[k] = malloc(len + 8); Zs[k] = malloc(len + 8);
                    memcpy(Xs[k], Xpre[k], len); memcpy(Zs[k], Zpre[k], len);
                }
                const double *ypre = use_best ? ybest : y;
                double cur2 = use_best ? best_score : res_score(R);
                /* 5.16: the endgame starts from the better of the iterate and its Euclidean polish (ISSUES 86:
                 * on roa_vdp_inner_d12_K the iterate's complementarity is 5e-5 and its polish scores 7e-8; the
                 * endgame from the iterate crept from 5e-5 to 2.6e-5 in its seven iterations and was rejected
                 * against the polish). A polished point can sit outside the cone by rounding: the endgame's
                 * shift takes care of that. */
                /* (off: the polished point is a worse start for the dd iteration — the projection takes its
                 * centrality — hinf15 ended at 2.7e-6 from it against 5.4e-7 from the iterate; BRISK_DDPOLSTART=1) */
                static int polstart = -1; if (polstart < 0) polstart = ENV_ON("BRISK_DDPOLSTART");
                int from_polish = 0; double pol_score = 1e300;
                if (polstart && par->polish) {
                    Result Rb0 = *R;
                    if (polish_solution(P, S, Xpre, Zpre, ypre, &Rb0, cur2, &Sc, dy, 0) && res_score(&Rb0) < cur2) {
                        for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); memcpy(Xs[k], S[k].G, len); memcpy(Zs[k], S[k].Q, len); }
                        if (par->verbose > 0) printf("   endgame from the polished iterate (%.2e against %.2e)\n", res_score(&Rb0), cur2);
                        from_polish = 1; pol_score = res_score(&Rb0);
                    }
                }
                (void)from_polish; (void)pol_score;
                ran_dd = 1;
                DdMeasCtx mctx = { P, S, *R, Sc.t };
                /* 23 October 2026: the score the endgame's result must beat, the polished iterate's (the comparison below),
                 * computed first so that the endgame can stop when its progress cannot reach it (ddend.c); the polish is a
                 * function of the point alone (its solve starts from zero), so the comparison below is unchanged */
                static int noproj = -1; if (noproj < 0) noproj = ENV_ON("BRISK_DDNOPROJ");
                double ddbar = 0;
                if (!noproj && par->polish) {
                    Result Rq = *R;
                    ddbar = polish_solution(P, S, Xpre, Zpre, ypre, &Rq, cur2, &Sc, dy, 0) ? fmin(cur2, res_score(&Rq)) : cur2;
                }
                /* 24 October 2026: not run when that polish alone meets the tolerance, as for the augmented-system end phase
                 * (control6, SDPLIB, SDP: the polish probes inside the iterations, capped at one iteration's work since 23
                 * October, handed over a point at 1.87e-8 whose full polish is at 8.76e-9; the endgame ran 9 iterations,
                 * 0.9 s, and its result lost to that polish). The polishes after it make the same point (the polish is a
                 * function of the point) */
                const double tdd0 = wtime(); int dd_ok = 0;
                if (!noproj && par->polish && ddbar <= par->tol) { if (par->verbose > 0) printf("   double-double endgame not needed: the polished point is at %.2e\n", ddbar); }
                else dd_ok = dd_endgame(P, Xs, Zs, yc, par, dd_maxit, par->tol, par->verbose, dd_meas_cb, &mctx, ddbar);
                if (getenv("BRISK_DDTIME")) printf("   [endgame: %.3f s, estimate %.2e flop units per iteration, %d allowed; the solve counted %.2e in %.2f s]\n", wtime() - tdd0, dd_it, dd_maxit, par->work_done + work_run, wtime() - par->t_start);
                if (dd_ok > 0) {
                    Result Rp = *R;
                    double sp = measure_candidate(P, S, Xs, Zs, yc, &Rp, Sc.t);
                    /* 5.16: the candidates compared after the Euclidean polish that follows anyway (ISSUES 86:
                     * on roa_vdp_inner_d12_K the endgame's point scored 2.3e-5 against the iterate's 5.0e-5
                     * and, polished, 2.2e-6 against 2.1e-6, and the run ended worse on the problem as read):
                     * the endgame's result is accepted only if its polished score is the lower one */
                    static int polcmp = -1; if (polcmp < 0) polcmp = !ENV_ON("BRISK_DDNOPOLCMP");
                    if (polcmp && sp < cur2 && par->polish) {
                        Result Re = Rp, Rb = *R;
                        const int ae = polish_solution(P, S, Xs, Zs, yc, &Re, sp, &Sc, dy, par->verbose > 1 ? par->verbose : 0);
                        const double pe = ae ? res_score(&Re) : sp;
                        const int ab = polish_solution(P, S, Xpre, Zpre, ypre, &Rb, cur2, &Sc, dy, par->verbose > 1 ? par->verbose : 0);
                        const double pb = ab ? res_score(&Rb) : cur2;
                        if (par->verbose > 0) printf("   endgame candidates after the Euclidean polish: %.2e (endgame) against %.2e (the iterate)\n", pe, pb);
                        if (!(pe < pb)) {
                            sp = cur2;      /* rejected: the iterate polishes better */
                            /* 15 October 2026 (the owner's benchmark, ISSUES 99): the iterate's polish, computed here and the
                             * better candidate, is kept as the point in hand (S.G, S.Q, the same y); the polish sequence below
                             * had re-polished the raw iterate after corrections that change the Schur state and ended worse
                             * (his hinf5: 1.5e-8 here, 7.1e-7 at the end) */
                            if (ab && pb < cur2) {
                                *R = Rb;
                                best_score = pb;
                                if (!use_best) memcpy(ybest, y, sizeof(double) * m);
                                for (int k = 0; k < nb; k++) {
                                    const size_t len = bsz(&P->blk[k]) * sizeof(double);
                                    if (keep_best) { memcpy(S[k].Xb, S[k].G, len); memcpy(S[k].Zb, S[k].Q, len); }
                                    memcpy(S[k].X, S[k].G, len); memcpy(S[k].Z, S[k].Q, len);
                                }
                                if (par->verbose > 0) printf("   the polished iterate is kept as the point in hand (%.2e)\n", pb);
                            }
                        }
                    }
                    if (par->verbose > 0)
                        printf("   high-precision endgame: score %.2e -> %.2e %s\n",
                               cur2, sp, sp < cur2 ? "accepted" : "rejected");
                    { extern void brisk_note(const char *, ...); brisk_note("double-double endgame of the embedding: score %.1e -> %.1e, %s", cur2, sp, sp < cur2 ? "accepted" : "rejected"); }
                    if (sp < cur2) {
                        double tt[6] = { R->t_setup, R->t_schur, R->t_chol, R->t_dense, R->t_step, R->t_total };
                        int itn = R->iters;
                        *R = Rp;
                        R->t_setup = tt[0]; R->t_schur = tt[1]; R->t_chol = tt[2]; R->t_dense = tt[3];
                        R->t_step = tt[4]; R->t_total = tt[5]; R->iters = itn;
                        best_score = sp;
                        memcpy(ybest, yc, sizeof(double) * m);
                        memcpy(y, yc, sizeof(double) * m);
                        for (int k = 0; k < nb; k++) {
                            size_t len = bsz(&P->blk[k]) * sizeof(double);
                            if (keep_best) { memcpy(S[k].Xb, Xs[k], len); memcpy(S[k].Zb, Zs[k], len); }
                            memcpy(S[k].X, Xs[k], len); memcpy(S[k].Z, Zs[k], len);
                        }
                        xsrc = 0;
                    }
                }
                for (int k = 0; k < nb; k++) { free(Xs[k]); free(Zs[k]); }
                free(Xs); free(Zs); free(yc); free(Xpre); free(Zpre);
            }
        }
        /* 23 October 2026 (on by default; BRISK_AUGEND=0: off): where the double-double endgame is skipped by its gates, the
         * augmented-system iterations (aug_endgame, §14.3) from the point it would have started from, on copies. Every gain
         * measured was on such files (the material-design files buck3, buck4, vibra3, shmup3, trto4, trto5; equalG11 and
         * equalG51 of SDPLIB), while run before an endgame that runs they only added time (the occupation measures
         * attr_vdp_d14_I 1.1 -> 1.8 s: the endgame did the work). The point is saved here and the phase runs after the
         * polishes below, judged against their result (24 October 2026: run before them and judged against the Euclidean
         * projection alone, its point replaced the one from which the square-root-metric restoration reached 1.1e-6, and
         * the chain then ended at 6.5e-5: beamSelfWeight_10_nmt_ad__d4, a user's frame relaxation, SDP, 1.1e-6 in 270 s
         * in 1.5.5, 4.3e-5 in 446 s after two re-solves) */
        static int augend = -1; if (augend < 0) { const char *e = getenv("BRISK_AUGEND"); augend = e ? atoi(e) : 1; }   /* (BRISK_AUGEND=0: off) */
        double **augX = NULL, **augZ = NULL, *augy = NULL; Result augR0; memset(&augR0, 0, sizeof(augR0));
        if (augend > 0 && dd_gated && !mixed && !timeout && status != 3 && res_score(R) >= par->tol && !g_dstop && !has_soc && !g_free_hsd) {
            const int use_best = keep_best && best_score <= res_score(R) * (1 + 1e-12);
            augX = malloc(sizeof(double *) * nb); augZ = malloc(sizeof(double *) * nb); augy = malloc(sizeof(double) * (m + 1));
            for (int k = 0; k < nb; k++) {
                const size_t len = bsz(&P->blk[k]) * sizeof(double);
                augX[k] = malloc(len + 8); augZ[k] = malloc(len + 8);
                memcpy(augX[k], use_best ? S[k].Xb : S[k].X, len); memcpy(augZ[k], use_best ? S[k].Zb : S[k].Z, len);
            }
            memcpy(augy, use_best ? ybest : y, sizeof(double) * m);
            augR0 = use_best ? best : *R;
        }

        /* 23 October 2026 (ISSUES 257): with second-order cone blocks the polishes below are skipped (they are written
         * for semidefinite and linear blocks); the Euclidean projection works on any block (the Gram products through
         * A and A', the cones' check by their smaller spectral value): a solve that ends one step short of the tolerance
         * on its primal residual can finish (the D-optimal designs, SDP with SOCP: the best iterate at a primal error
         * of 2e-8 with the gap at 1e-13). On 101 files with cones: three better (dopt_n8_p400 and dopt_n32_p8000 REDUCED -> OPTIMAL,
         * filter48_socp 1.5e-7 -> 1.0e-8), none worse, the same time; under 16 perturbations each the two D-optimal files 24 -> 32
         * of 32 runs OPTIMAL. BRISK_SOCPOLISH=0: none, as before */
        if (par->polish && status != 3 && !mixed && !timeout && !g_dstop && has_soc && res_score(R) > par->tol && !(getenv("BRISK_SOCPOLISH") && atoi(getenv("BRISK_SOCPOLISH")) == 0)) {
            const int use_best = keep_best && best_score <= res_score(R) * (1 + 1e-12);
            double **Xb = malloc(sizeof(double *) * nb), **Zb = malloc(sizeof(double *) * nb);
            for (int k = 0; k < nb; k++) { Xb[k] = use_best ? S[k].Xb : S[k].X; Zb[k] = use_best ? S[k].Zb : S[k].Z; }
            R->iters = it;
            if (use_best || !keep_best || res_score(R) <= best_score) {
                if (polish_solution(P, S, Xb, Zb, use_best ? ybest : y, R, res_score(R), &Sc, dy, par->verbose)) {
                    xsrc = 2;
                    if (use_best) memcpy(y, ybest, sizeof(double) * m);
                }
            }
            free(Xb); free(Zb);
        }
        /* polishing of the returned point (best iterate if stored, else final) */
        /* 4.25: not past the time limit (thetaG51 -timelimit 1: 5 s of polishing after the stop) */
        if (par->polish && status != 3 && !mixed && !timeout && !g_dstop && !has_soc) {
            double **Xb = malloc(sizeof(double *) * nb), **Zb = malloc(sizeof(double *) * nb);
            int use_best = keep_best && best_score <= res_score(R) * (1 + 1e-12);
            for (int k = 0; k < nb; k++) {
                Xb[k] = use_best ? S[k].Xb : S[k].X;
                Zb[k] = use_best ? S[k].Zb : S[k].Z;
            }
            R->iters = it;
            if (use_best || !keep_best || res_score(R) <= best_score) {
                /* 24 October 2026 (ISSUES 266): the runner-up of the polishes is kept for the judgment on the problem as read */
                const int xsrc0 = xsrc;
                const Result Rm = use_best ? best : *R;
                double *ym = malloc(sizeof(double) * (m + 1));
                memcpy(ym, use_best ? ybest : y, sizeof(double) * m);
                if (Xout) rej_begin();
                /* the X-metric correction first (it keeps X positive definite); the
                 * Euclidean projection only if that one was not accepted */
                /* the restoration of the primal feasibility in the square-root metric first: from
                 * the iterate kept for it when that promises more than the best one, else from the
                 * best one; the older corrections only if neither is accepted */
                int ph = 0;
                if (par->polish_r) {
                    /* first from the iterate kept for it (the smallest dual residual and
                     * complementarity), then, if the tolerance is still missed, from the best
                     * iterate; the better candidate stays in s->G, s->Q, with its y in y_cand */
                    const int it_ret = use_best ? it_best : it;        /* the iteration of the point in hand */
                    const double *y_ret = use_best ? ybest : y, *y_cand = NULL;
                    /* the best iterate is restored only if its primal residual, or what that
                     * residual puts into the gap, is a quarter of its largest error: a restoration
                     * moves X in proportion to the residual, so it cannot help an error that is
                     * the complementarity of a feasible point. The
                     * complementarity before the restoration is no guide to the one after it:
                     * with a large residual X has a component that Z sees and the restoration
                     * removes (attr_henon_d10_K: 2.5e-5 before, 1.2e-8 after). */
                    const double sc0 = res_score(R), pr0 = fmax(nan_inf(R->pinf), fabs(R->relgap - fabs(R->relcomp)));
                    double **Gs = malloc(sizeof(double *) * 2 * nb);
                    for (int k = 0; k < 2 * nb; k++) Gs[k] = NULL;
                    #define CAND_SAVE() for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); \
                        if (!Gs[k]) { Gs[k] = malloc(len + 8); Gs[nb + k] = malloc(len + 8); } memcpy(Gs[k], S[k].G, len); memcpy(Gs[nb + k], S[k].Q, len); }
                    #define CAND_BACK() for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); memcpy(S[k].G, Gs[k], len); memcpy(S[k].Q, Gs[nb + k], len); }
                    /* not when the returned iterate is the objective candidate: the restoration's iterate
                     * is chosen by its dual residual and complementarity, and on the problems where the
                     * candidate is taken that is an early iterate with the worse objective (ni35 order 2
                     * with the completed cut: iteration 20 restored, <C,X> 1949.26, instead of iteration
                     * 40, 1942.44) */
                    if (objcand && rc.have && rc.it != it_ret && rc.score < sc0 && par->verbose > 1)
                        printf("   not restoring iteration %d: the returned iterate %d was chosen by its objective\n", rc.it, it_ret);
                    if (rc.have && rc.it != it_ret && rc.score < sc0 && !objcand) {
                        double **Xr = malloc(sizeof(double *) * nb), **Zr = malloc(sizeof(double *) * nb);
                        for (int k = 0; k < nb; k++) { Xr[k] = S[k].Xr; Zr[k] = S[k].Zr; }
                        if (par->verbose > 1) printf("   restoring the primal feasibility of iteration %d (dual residual and complementarity %.1e)\n", rc.it, rc.score);
                        if (polish_xmetric(P, S, Xr, Zr, rc.y, R, res_score(R), &Sc, dy, par->verbose, 1)) { ph = 1; y_cand = rc.y; }
                        free(Xr); free(Zr);
                    }
                    if ((!ph || res_score(R) > par->tol) && pr0 >= 0.25 * sc0) {
                        if (ph) CAND_SAVE();
                        if (polish_xmetric(P, S, Xb, Zb, y_ret, R, res_score(R), &Sc, dy, par->verbose, 1)) { ph = 1; y_cand = y_ret; }
                        else if (ph) CAND_BACK();
                    }
                    /* a restoration that was cut short (halved steps) leaves a primal residual: the
                     * Euclidean projection of its result, kept if better */
                    if (ph && res_score(R) > par->tol && R->pinf > 0.5 * res_score(R)) {
                        CAND_SAVE();
                        if (!polish_solution(P, S, Gs, Gs + nb, y_cand, R, res_score(R), &Sc, dy, par->verbose)) CAND_BACK();
                    }
                    /* still above the tolerance: the Euclidean projection of the unpolished point as
                     * well, the better one kept (it may trade a small indefiniteness of X for the gap) */
                    if (ph && res_score(R) > par->tol) {
                        CAND_SAVE();
                        if (polish_solution(P, S, Xb, Zb, y_ret, R, res_score(R), &Sc, dy, par->verbose)) y_cand = y_ret;
                        else CAND_BACK();
                    }
                    #undef CAND_SAVE
                    #undef CAND_BACK
                    for (int k = 0; k < 2 * nb; k++) free(Gs[k]);
                    free(Gs);
                    if (ph) { xsrc = 2; if (y_cand != y) memcpy(y, y_cand, sizeof(double) * m); }
                }
                int px = !ph && par->polish_x && polish_xmetric(P, S, Xb, Zb, use_best ? ybest : y, R, res_score(R), &Sc, dy, par->verbose, 0);
                if (ph) { }
                else if (px) {
                    xsrc = 2;
                    if (use_best) memcpy(y, ybest, sizeof(double) * m);
                    /* 4.22: still above the tolerance: the Euclidean projection of the
                     * X-metric result as well (dense case6468: X-metric 1.1e-5 -> 7.4e-6) */
                    if (res_score(R) > par->tol && !getenv("BRISK_NOPOLISH2")) {
                        double **Xc = malloc(sizeof(double *) * nb), **Zc = malloc(sizeof(double *) * nb);
                        for (int k = 0; k < nb; k++) {
                            const size_t len = bsz(&P->blk[k]);
                            Xc[k] = malloc(sizeof(double) * (len + 1)); memcpy(Xc[k], S[k].G, sizeof(double) * len);
                            Zc[k] = malloc(sizeof(double) * (len + 1)); memcpy(Zc[k], S[k].Q, sizeof(double) * len);
                        }
                        if (!polish_solution(P, S, Xc, Zc, y, R, res_score(R), &Sc, dy, par->verbose))
                            for (int k = 0; k < nb; k++) {       /* keep the X-metric candidate */
                                const size_t len = bsz(&P->blk[k]);
                                memcpy(S[k].G, Xc[k], sizeof(double) * len); memcpy(S[k].Q, Zc[k], sizeof(double) * len);
                            }
                        /* 4.24: still above the tolerance: the Euclidean projection of the unpolished
                         * point as well, the better one kept (TSSOS case6468: the X-metric chain
                         * 6.6e-6 -> 2.5e-6 and a second solve, the plain projection 4e-8) */
                        if (res_score(R) > par->tol && !getenv("BRISK_NOPOLISH3")) {
                            for (int k = 0; k < nb; k++) {
                                const size_t len = bsz(&P->blk[k]);
                                memcpy(Xc[k], S[k].G, sizeof(double) * len); memcpy(Zc[k], S[k].Q, sizeof(double) * len);
                            }
                            if (!polish_solution(P, S, Xb, Zb, use_best ? ybest : y, R, res_score(R), &Sc, dy, par->verbose))
                                for (int k = 0; k < nb; k++) {
                                    const size_t len = bsz(&P->blk[k]);
                                    memcpy(S[k].G, Xc[k], sizeof(double) * len); memcpy(S[k].Q, Zc[k], sizeof(double) * len);
                                }
                        }
                        for (int k = 0; k < nb; k++) { free(Xc[k]); free(Zc[k]); }
                        free(Xc); free(Zc);
                    }
                } else if (polish_solution(P, S, Xb, Zb, use_best ? ybest : y, R, res_score(R), &Sc, dy, par->verbose)) {
                    xsrc = 2;
                    if (use_best) memcpy(y, ybest, sizeof(double) * m);
                } else if (use_best) {
                    xsrc = 1;
                    memcpy(y, ybest, sizeof(double) * m);
                }
                if (Xout) rej_end(P, Xb, ym, &Rm, xsrc == 2 && xsrc0 != 2);
                free(ym);
            }
            free(Xb); free(Zb);
        }
        /* the augmented-system end phase (saved above), when the polishes left the point above the tolerance: its result,
         * polished by the Euclidean projection, is kept only if it beats theirs. It uses each block's Q as scratch and its
         * polish writes G and Q, where the point in hand lies when the polishes were accepted (xsrc 2): saved and restored */
        if (augX) {
            const double bar = res_score(R);
            const int run = bar > par->tol && !(par->timelimit > 0 && wtime() - par->t_start >= par->timelimit) && !brisk_stop_flag;
            int take = 0; double pe_out = bar;
            if (!run && par->verbose > 1) printf("   augmented-system end phase not needed: the polished point is at %.2e\n", bar);
            if (run) {
                double **sG = NULL, **sQ = NULL;
                if (xsrc == 2) {
                    sG = malloc(sizeof(double *) * nb); sQ = malloc(sizeof(double *) * nb);
                    for (int k = 0; k < nb; k++) {
                        const size_t len = bsz(&P->blk[k]) * sizeof(double);
                        sG[k] = malloc(len + 8); sQ[k] = malloc(len + 8); memcpy(sG[k], S[k].G, len); memcpy(sQ[k], S[k].Q, len);
                    }
                }
                /* the double-double endgame's work budget, in its units (ISSUES 283: until then the solve's elapsed seconds) */
                const double wbud = fmax(par->dd_minwork, par->dd_factor * (par->work_done + work_run));
                if (par->verbose > 1) printf("   augmented-system end phase: budget %.3e work units (the solve counted %.3e in %.2f s)\n", wbud, par->work_done + work_run, wtime() - t0);
                Result augR = augR0;
                int pol = 0;                                            /* 1: the polished result (in G, Q) is the better one */
                if (aug_endgame(P, S, &Sc, augX, augZ, augy, par, Ntot, augend > 1 ? augend : 15, wbud, work_it, par->verbose, &augR)) {
                    double pe = res_score(&augR);
                    Result Re = augR;
                    /* its point gets the chain's main tools as the method's point did: the restoration in the square-root
                     * metric (rose15, Mittelmann, SDP: 4.05e-8 -> 3.57e-9; with the Euclidean projection alone it stayed at
                     * 4.05e-8), the Euclidean projection of that result, and that of the unpolished point; each candidate
                     * kept in G, Q only if it beats the one before (polish_xmetric and polish_solution accept only an
                     * improvement on the score they are given) */
                    if (par->polish && pe > par->tol) {
                        if (par->polish_r && polish_xmetric(P, S, augX, augZ, augy, &Re, pe, &Sc, dy, par->verbose > 1, 1)) { pe = res_score(&Re); pol = 1; }
                        double **tG = malloc(sizeof(double *) * nb), **tQ = malloc(sizeof(double *) * nb);
                        for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); tG[k] = malloc(len + 8); tQ[k] = malloc(len + 8); }
                        #define AUG_CAND_SAVE() for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); memcpy(tG[k], S[k].G, len); memcpy(tQ[k], S[k].Q, len); }
                        #define AUG_CAND_BACK() for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); memcpy(S[k].G, tG[k], len); memcpy(S[k].Q, tQ[k], len); }
                        if (pol && pe > par->tol) {          /* the restoration's result projected (its input copied: the output is G, Q) */
                            AUG_CAND_SAVE();
                            if (polish_solution(P, S, tG, tQ, augy, &Re, pe, &Sc, dy, 0)) pe = res_score(&Re); else AUG_CAND_BACK();
                        }
                        if (pe > par->tol) {                 /* the unpolished point projected, kept if better */
                            if (pol) AUG_CAND_SAVE();
                            if (polish_solution(P, S, augX, augZ, augy, &Re, pe, &Sc, dy, 0)) { pe = res_score(&Re); pol = 1; }
                            else if (pol) AUG_CAND_BACK();
                        }
                        #undef AUG_CAND_SAVE
                        #undef AUG_CAND_BACK
                        for (int k = 0; k < nb; k++) { free(tG[k]); free(tQ[k]); }
                        free(tG); free(tQ);
                    }
                    take = pe < bar; pe_out = pe;
                    if (par->verbose > 0) printf("   augmented-system end phase: polished %.2e against %.2e after the polishes, %s\n", pe, bar, take ? "kept" : "not kept");
                    if (take) {
                        if (Xout) {                          /* the chain's point, for the judgment on the problem as read */
                            double **Xk = malloc(sizeof(double *) * (nb + 1));
                            for (int k = 0; k < nb; k++) Xk[k] = xsrc == 2 ? sG[k] : xsrc == 1 ? S[k].Xb : S[k].X;
                            alt_store(P, Xk, y, R, "the point before the end phase on the augmented Newton system");
                            free(Xk);
                        }
                        const int itn = R->iters;
                        *R = pol ? Re : augR; R->iters = itn; best = *R; best_score = res_score(R);
                        memcpy(y, augy, sizeof(double) * m); memcpy(ybest, augy, sizeof(double) * m);
                        if (pol) xsrc = 2;
                        else {
                            for (int k = 0; k < nb; k++) {
                                const size_t len = bsz(&P->blk[k]) * sizeof(double);
                                if (keep_best) { memcpy(S[k].Xb, augX[k], len); memcpy(S[k].Zb, augZ[k], len); }
                                memcpy(S[k].X, augX[k], len); memcpy(S[k].Z, augZ[k], len);
                            }
                            xsrc = 0;
                        }
                    }
                }
                if (sG) {
                    if (!take) for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); memcpy(S[k].G, sG[k], len); memcpy(S[k].Q, sQ[k], len); }
                    for (int k = 0; k < nb; k++) { free(sG[k]); free(sQ[k]); }
                    free(sG); free(sQ);
                }
                /* the Summary's line (the scores after the polishes) */
                { extern void brisk_note(const char *, ...); brisk_note("end phase on the augmented Newton system: score %.1e -> %.1e, %s", bar, take ? pe_out : bar, take ? "kept" : "nothing gained"); }
            }
            for (int k = 0; k < nb; k++) { free(augX[k]); free(augZ[k]); }
            free(augX); free(augZ); free(augy);
        }
        double acc = fmax(res_score(R), fmax(R->err[2], R->err[4]));
        if (status != ST_TIME) {
            if (acc < par->tol) status = 0;
            else if (acc < par->red_acc) status = 5;
        }
    }
    R->iters = it;
    R->status = status;
    if (yout) for (int i = 0; i < m; i++) yout[i] = P->cs * P->d[i] * y[i];
    if (Xout) for (int k = 0; k < nb; k++) blk_out(P, k, xsrc == 2 ? S[k].G : xsrc == 1 ? S[k].Xb : S[k].X, Xout[k]);
    R->t_total = wtime() - t0;
    /* Automatic method choice: a non-optimal exit that is not an infeasibility
     * certificate is retried with the embedding, and the better result is kept.
     * Results already within retry_acc are returned as they are: going from 1e-7 to
     * 1e-10 is not worth a second solve. The retry runs after this run's work arrays
     * have been freed.                                                              */
    double s1 = fmax(res_score(R), fmax(R->err[2], R->err[4]));
    const int time_left = !(brisk_time_up(par));
    /* retry: standard -> embedding; with the embedding first, embedding -> standard */
    /* 23 October 2026: after the embedding, the infeasible-start method only when the embedding failed, above ten times
     * retry_hsd1, not when it ended at reduced accuracy (the band the creep rule also treats as near): of the full
     * benchmark's retries, all 10 from (1e-6, 1e-5] ended where the embedding had (hinf_sf_n96 41 s, sat_roa_n96 157 s,
     * p3_w25 17 s), the 4 kept came from above 1e-5. BRISK_RETRYBAND=0: from retry_hsd1 as before */
    static int rband = -1; if (rband < 0) { const char *e = getenv("BRISK_RETRYBAND"); rband = !(e && atoi(e) == 0); }
    const double rgate = par_local.hsd_first_on ? (rband ? 10.0 : 1.0) * par->retry_hsd1 : par->retry_acc;
    const int do_retry = !has_soc && par_user->hsd < 0 && (!par_local.hsd || par_local.hsd_first_on) && status != 0 && status != 1 && status != 2 &&
                         !par_local.no_retry && time_left && !(s1 <= rgate) &&
                         !g_dstop;      /* 5.12: the returned best iterate meets the tolerance in the DIMACS measures */
    if (par->verbose && (Sc.nreg || nrefine || Sc.pcg_iters))
        printf("   (Schur regularized in %d its; %d Newton refinements; %d PCG steps)\n",
               Sc.nreg, nrefine, Sc.pcg_iters);
    if (ENV_ON("BRISK_DEBUG")) printf("   [Lanczos: %ld calls, %.1f iterations each]\n", lz_calls, lz_calls ? (double)lz_iters / lz_calls : 0.0);
    if (ENV_ON("BRISK_DEBUG")) printf("   [prod3 %ld, prodZFR %ld, refinement attempts %ld, residual re-bases %d]\n",
                                      cnt_prod3, cnt_zfr, cnt_refine_try, nrebase);
    if (par->verbose && Sc.nfloat) printf("   (%d Schur factorizations in single precision)\n", Sc.nfloat);
    if (par->verbose) printf("   (Schur factor %.3fs, solves %.3fs in %d calls)\n", Sc.t_fact, Sc.t_solve, Sc.nsolve);
    if (g_p3chk > 0) printf("   [prod3 symmetry check: max relative asymmetry %.2e]\n", g_p3asym);
    if (getenv("BRISK_SOLDBG")) printf("   [parallel assembly: blocks %.2f flush %.2f serial blocks %.2f (%.0f chunks)]\n", g_asmp[0], g_asmp[1], g_asmp[2], g_asmp[3]);
    if (getenv("BRISK_SOLDBG")) printf("   [assembly: LP %.2f lowrank %.2f dict %.2f other SDP %.2f (id %.2f, flush %.2f of %.3g entries; routes 0/1/3: %.2f %.2f %.2f s, %.3g %.3g %.3g pairs)]\n", g_asm[0], g_asm[1], g_asm[2], g_asm[3], g_asmf[1], g_asmf[0], g_asmf[2], g_asmr[0], g_asmr[1], g_asmr[2], g_asmr[3], g_asmr[4], g_asmr[5]);
    if (getenv("BRISK_SOLDBG")) printf("   [corrector parts: pass1 %.2f pass2 %.2f Aop %.2f solve %.2f back %.2f maxstep %.2f; eig_outside %.2f s in %.0f calls | pieces loop %.2f, affine recovery loop %.2f]\n", g_ct[0], g_ct[1], g_ct[2], g_ct[3], g_ct[4], g_ct[5], g_ct[6], g_ct[7], g_ct[8], g_ct[9]);
    if (getenv("BRISK_SOLDBG")) printf("   [phases: residuals %.2f | schur+factor %.2f | pieces %.2f | predictor %.2f | sigma+correctors %.2f | refine %.2f | step %.2f | (unused %.2f) sigma-search %.2f correctors %.2f]\n",
        g_ph[0], g_ph[1], g_ph[2], g_ph[3], g_ph[5], g_ph[6], g_ph[7], g_ph[4], g_ph[8], g_ph[9]);
    if (getenv("BRISK_SOLDBG")) printf("   [predictor parts: second-order term %.2f in %.0f passes, endpoint directions %.2f, sigma search of the passes %.2f]\n", g_pt[0], g_pt[5], g_pt[1], g_pt[2]);
    if (getenv("BRISK_SOLDBG")) printf("   [bord factor: %ld calls, %.3fs total, %.3fs in schol_factor]\n", g_bf_n, g_bf_all, g_bf_fact);
    if (getenv("BRISK_SOLDBG")) { extern long g_pcgh[12]; printf("   [single PCG runs by steps 0..10+:"); for (int q = 0; q <= 10; q++) printf(" %ld", g_pcgh[q]); printf("; ended above tol %ld]\n", g_pcgh[11]); }
    if (getenv("BRISK_SOLDBG")) { extern double g_tmv[2], g_tmvg; extern long g_nmv[2], g_nmvg; printf("   [schol_mv %ld calls %.3fs (the value copies %ld, %.3fs); sparse solves %ld rhs %.3fs]\n", g_nmv[0], g_tmv[0], g_nmvg, g_tmvg, g_nmv[1], g_tmv[1]); }
    if (getenv("BRISK_SOLDBG")) printf("   [bord: from (1e-13, 1e-11] one pass reached 1e-13 %ld times, not %ld (worst %.1e)]\n", g_b1ok, g_b1bad, g_b1worst);
    if (getenv("BRISK_SOLDBG")) printf("   [bord: first-pass rel <=1e-13 %ld, <=1e-11 %ld, <=1e-9 %ld, <=1e-7 %ld, worse %ld; stop after pass 0:%ld 1:%ld 2:%ld 3:%ld 4:%ld 5+:%ld]\n",
        g_bh0[0], g_bh0[1], g_bh0[2], g_bh0[3], g_bh0[4], g_bhp[0], g_bhp[1], g_bhp[2], g_bhp[3], g_bhp[4], g_bhp[5]);
    if (getenv("BRISK_SOLDBG")) printf("   [multi column-check skips %ld] [solves: single trusted %ld ok %ld pcg %ld | multi trusted %ld ok %ld pcg %ld | bord %ld | mv %ld]\n",
        g_multiskip, g_sc_cnt[0], g_sc_cnt[1], g_sc_cnt[2], g_sc_cnt[3], g_sc_cnt[4], g_sc_cnt[5], g_sc_cnt[6], g_sc_cnt[7]);
    { extern void schol_fstat_print(void); schol_fstat_print(); }
    { extern void schol_ftime_print(void); schol_ftime_print(); }
    if (par->verbose && cnt_feas) printf("   (primal step capped by the feasibility guard %d times)\n", cnt_feas);
    if (par->verbose && nt) printf("   (NT direction: Gram split used in %d iterations, unavailable in %d; least-squares refinement accepted %d/%d)\n",
                                   nsplit, nsplit_fail, nls_acc, nls_try);
    if (par->verbose && ncorr_total) printf("   (%d centrality correctors accepted)\n", ncorr_total);

    /* 4.29: the re-solve starts from a blend of this run's returned point and its own
     * standard start (warm_lam; the copy is in this problem's scaling)            */
    double **wX = NULL, **wZ = NULL, *wy = NULL;
    if (do_retry && par->warm_lam > 0 && !(xsrc == 1 && !keep_best)) {
        wX = malloc(sizeof(double *) * nb); wZ = malloc(sizeof(double *) * nb);
        wy = malloc(sizeof(double) * (m + 1));
        memcpy(wy, y, sizeof(double) * m);
        for (int k = 0; k < nb; k++) {
            const size_t len = bsz(&P->blk[k]);
            const double *Xk = xsrc == 2 ? S[k].G : xsrc == 1 ? S[k].Xb : S[k].X;
            const double *Zk = xsrc == 2 ? S[k].Q : xsrc == 1 ? S[k].Zb : S[k].Z;
            wX[k] = malloc(sizeof(double) * (len + 1)); wZ[k] = malloc(sizeof(double) * (len + 1));
            memcpy(wX[k], Xk, sizeof(double) * len); memcpy(wZ[k], Zk, sizeof(double) * len);
        }
    }
    for (int k = 0; k < nb; k++) {
        BS *s = &S[k];
        double *bufs[] = { s->X, s->Z, s->LX, s->LZ, s->Zi, s->dX, s->dZ, s->F, s->G,
                           s->Q, s->W, s->Xn, s->Zn, s->Fr, s->T1, s->Zc,
                           s->lq, s->v1, s->v2, s->dtmp, s->Gd, s->Mdd, s->Ct, s->Dxc, s->ev, s->R0, s->r0v,
                           s->lrV, s->lrPQ, s->lrPd, s->lrQd, s->lrQ, s->Xb, s->Zb, s->Xr, s->Zr,
                           s->ntGbuf, s->ntVt, s->ntT, s->lam, s->dX1, s->dS1, s->dX1t, s->dS1t,
                           s->dXt, s->dZt, s->Rt, s->sv, s->stmp };   /* (sv, stmp: the second-order cone's; leaked since 1.4.4) */
        for (size_t q = 0; q < sizeof(bufs) / sizeof(bufs[0]); q++) free(bufs[q]);
        free(s->ntGf); free(s->ntLXf); free(s->ntVtf); free(s->ntTf); free(s->ntT2f); free(s->ntGold); s->ntGold = NULL;
        free(s->r0r); free(s->r0c); free(s->lr_owner);
        if (s->vbs) {
            free(s->vbs->X); free(s->vbs->Zi); free(s->vbs->dtmp); free(s->vbs->Gd); free(s->vbs->Mdd);
            free(s->vbs);
        }
        free(s->dW);
        if (s->cpat.owned) { free(s->cpat.fr); free(s->cpat.fc); free(s->cpat.rows); }
        schol_free(s->zf); free(s->zcp); free(s->zri); free(s->zt); free(s->ucp); free(s->uri); free(s->zdp); free(s->zpos); free(s->zamap); free(s->zfq); free(s->zcq); free(s->zr0q);
    }
    cls_release();
    free(ev); free(wv); free(w1); free(y1); free(ybest); free(slowh); free(rc.y);
    endrules_free(&ER_std);
    free(S); free(y); free(dy); free(rhs); free(rp); free(a0); free(h); free(qv);
    free(Gs.M); free(Gs.Md); free(Gs.Mf); free(Gs.ya); free(Gs.e3m); envelope_free(Gs.E); free(Gs.e1); free(Gs.e2); free(Gs.e3);
    free(Gs.D); free(Gs.r); free(Gs.t); free(Gs.p); free(Gs.q); free(Gs.xb); free(Gs.fw);
    free(lsz); free(lsp); free(lss); free(lsy); free(lst); free(lsn); free(lsr); free(lsq);
    if (g_blas_T > 1 && blas_T_prev == 0) BL(openblas_set_num_threads)(g_blas_T);
    g_blas_T = blas_T_prev;
    g_par_blk = par_blk_prev;
    R->nreg = Sc.nreg; R->npcg = Sc.pcg_iters;
    schur_pos_free(&Sc);
    free(Sc.M); free(Sc.Md); free(Sc.Mf); free(Sc.ya); free(Sc.e3m); free(Sc.Masm); envelope_free(Sc.E); schol_free(Sc.SC); free(Sc.e1); free(Sc.e2); free(Sc.e3); free(Sc.D); free(Sc.r); free(Sc.t); free(Sc.p); free(Sc.q); free(Sc.xb); free(Sc.fw);
    int do_retry2 = do_retry;
    if (do_retry2) {
        /* the retry's memory (ISSUES 98): about twelve dense arrays of n^2 per semidefinite block and one and a
         * half Schur matrices of m^2, on top of what is held now; when that does not fit, the point in hand is
         * returned with a note instead of a process killed by the system */
        double need = 1.5 * 8.0 * (double)P->m * P->m, avail = 0;
        for (int k = 0; k < nb; k++) if (P->blk[k].type == BLK_SDP) need += 12.0 * 8.0 * (double)P->blk[k].n * P->blk[k].n;
        if (!mem_fits(need, &avail)) {
            if (par_user->verbose >= 0) printf("   no retry with the %s: it needs about %.1f GB and %.1f GB are available; the result in hand is returned\n",
                                               par_local.hsd_first_on ? "infeasible-start method" : "self-dual embedding", need / 1e9, avail / 1e9);
            { extern void brisk_note(const char *, ...); brisk_note("the retry with the %s was not made: it needs about %.1f GB, %.1f GB were available", par_local.hsd_first_on ? "infeasible-start method" : "self-dual embedding", need / 1e9, avail / 1e9); }
            do_retry2 = 0;
        }
    }
    if (do_retry2) {
        Params p2 = *par_user;
        p2.hsd = par_local.hsd_first_on ? 0 : 1;
        p2.no_retry = 1;
        p2.dd_done = par_user->dd_done || ran_dd;
        p2.work_done = par_user->work_done + work_run;
        p2.warm_X = wX; p2.warm_Z = wZ; p2.warm_y = wy;
        p2.xcone = NULL;     /* (X2 has this problem's own layout: its cone blocks are already cones) */
        /* 23 October 2026 (ISSUES 253): after an embedding whose primal iterate ended far beyond its start's size, the
         * embedding again, cold, with its primal start scaled to that size, instead of the infeasible-start method. The
         * embedding's accuracy depends on how far its start is from the solution: the final tau falls with the distance and
         * the residuals lose accuracy with it. The size is the Frobenius norm, X/tau's at the end against X's at the start,
         * the re-solve from a ratio of 10: a low-rank solution concentrates in a few eigenvalues, which the trace
         * understates (mvu_400, SDP: the trace ratio 6.7, the Frobenius ratio 130). On the 16 benchmark files where 1.5.5's
         * embedding ends above 1e-5 it fires on seven: nishioka 2.6e-5 -> 2.5e-7, mvu_400, mvu_800 and mvu_1200 1e-4 ->
         * 2.2e-7 to 2.8e-7 (MOSEK fails all three), hinfsyn_48 6.5e-4 -> 6.1e-5, hinfsyn_64/80 and p3_w11 unchanged (the
         * trace ratio from 1e3 caught only nishioka). BRISK_HSDRESCALE=0: the infeasible-start method as before */
        static int rescale_on = -1; if (rescale_on < 0) { const char *e = getenv("BRISK_HSDRESCALE"); rescale_on = !(e && atoi(e) == 0); }
        const double fmis = g_hsd_fx0 > 0 ? g_hsd_fxe / g_hsd_fx0 : 0;
        const int rescale = par_local.hsd_first_on && rescale_on && !(par_user->hsd_x0 > 0) && fmis >= 10;
        const double xfac = fmis;
        if (rescale) { p2.hsd = 1; p2.hsd_x0 = xfac; p2.warm_X = NULL; p2.warm_Z = NULL; p2.warm_y = NULL; }
        if (par_user->verbose && rescale) printf("   retrying with the self-dual embedding, its primal start scaled by %.1e (the first one's X/tau ended at that many times its start's Frobenius norm)\n", xfac);
        else if (par_user->verbose) printf("   retrying with the %s (status %d, accuracy %.1e%s%s)\n", p2.hsd ? "self-dual embedding" : "infeasible-start method", status, s1,
                                      (!p2.hsd && status == 4 && g_hsd_why) ? "; the embedding stopped: " : "", (!p2.hsd && status == 4 && g_hsd_why) ? g_hsd_why : "");
        Result R2;
        memset(&R2, 0, sizeof(R2));
        double *y2 = calloc(P->m + 1, sizeof(double));
        double **X2 = NULL;
        if (Xout) {
            X2 = malloc(sizeof(double *) * nb);
            for (int k = 0; k < nb; k++) X2[k] = calloc(bsz(&P->blk[k]) + 1, sizeof(double));
        }
        const double t_r0 = wtime();
        AltSlot alt1 = g_alt; memset(&g_alt, 0, sizeof(g_alt));   /* this solve's earlier point (the call clears the slot) */
        int st2 = brisk_solve(P, &p2, &R2, y2, X2);
        const double t_r = wtime() - t_r0;
        double s2 = fmax(res_score(&R2), fmax(R2.err[2], R2.err[4]));
        /* an infeasibility claim replaces a first result only if that was far from optimal */
        int take = ((st2 == 1 || st2 == 2) && s1 > 1e-4) || (st2 != 1 && st2 != 2 && s2 < s1);
        R2.methods |= R->methods;
        if (take) {
            R2.t_total += R->t_total;
            R2.iters += R->iters;
            R2.retried = 1;
            *R = R2;
            status = st2;
            if (yout) memcpy(yout, y2, sizeof(double) * P->m);
            if (Xout) for (int k = 0; k < nb; k++) memcpy(Xout[k], X2[k], sizeof(double) * bsz(&P->blk[k]));
            alt_free(&alt1);                                 /* the retry's own earlier point, if any, goes with its result */
        } else {
            alt_free(&g_alt); g_alt = alt1;
            R->methods |= R2.methods;
            R->t_total += R2.t_total;          /* the rejected retry still cost time */
            R->iters += R2.iters;
        }
        R->nretry = 1; R->retry_kept = take; R->retry_hsd = rescale ? 2 : p2.hsd; R->retry_x0 = rescale ? xfac : 0; R->t_retry = t_r; R->acc_first = s1;
        if (X2) { for (int k = 0; k < nb; k++) free(X2[k]); free(X2); }
        free(y2);
    }
    if (wX) { for (int k = 0; k < nb; k++) { free(wX[k]); free(wZ[k]); } free(wX); free(wZ); free(wy); }
    return status;
}

/* ======================================================================== */
/* Dual-scaling method (-dual), DSDP style, on the primal-dual infrastructure.
 *
 * Only y is iterated: Z(y) = C - A'(y) > 0. Each iteration forms
 *     M_ij = <A_i, Zi A_j Zi>   (the HKM Schur complement with X = Zi, so every route of
 *                                the primal-dual method applies: sparse/row products,
 *                                low rank, dictionary, sparse or envelope Schur factor,
 *                                mixed precision with refinement)
 * and the two solves d1 = M^{-1} b, d2 = M^{-1} A(Zi). The direction for barrier
 * parameter mu is dy(mu) = d1/mu - d2; the implied primal point
 *     X(mu) = mu Zi (Z + A'(dy(mu))) Zi
 * satisfies A(X) = b exactly and is PSD iff C - A'(y - dy(mu)) is, so an upper bound
 * b'y + mu n + A(Zi)'dy(mu) is certified by one Cholesky test per candidate mu.
 * Steps: potential reduction toward mu = gap / (rho n), step length from Lanczos on
 * L^{-1} dZ L^{-T} (as in the primal-dual method), one Cholesky to accept it.
 *
 * Start: y = 0 if Z(0) > 0; else y = t y_I when A'(y_I) = -I is solvable (the identity
 * is in the range of A': trace constraints, max-cut), so Z = C + t I; otherwise a
 * phase 1 on Z = C - A'(y) + r I that maximizes -r and stops as soon as r < 0, with the
 * step limited to just past the crossing so y does not overshoot.                 */

static void ds_Z(const Block *B, BS *s, const double *y, double r, double *out) {
    const int n = B->n;
    const size_t len = bsz(B);
    blk_ATy(B, s, y, s->F);
    for (size_t i = 0; i < len; i++) out[i] = -s->F[i];
    sp_add(B, &B->C, 1.0, out);
    if (r != 0) {
        if (B->type == BLK_LP) for (int i = 0; i < n; i++) out[i] += r;
        else for (int i = 0; i < n; i++) out[i + (size_t)i * n] += r;
    }
}
/* Cholesky of Z in place (lower); 1 if positive definite, with log det */
static int ds_chol(const Block *B, double *L, double *logdet) {
    const int n = B->n;
    double ld = 0;
    if (B->type == BLK_LP) {
        for (int i = 0; i < n; i++) { if (!(L[i] > 0) || !isfinite(L[i])) return 0; ld += log(L[i]); }
    } else {
        if (chol_lower(n, L) != 0) return 0;
        for (int i = 0; i < n; i++) { double d = L[i + (size_t)i * n]; if (!(d > 0) || !isfinite(d)) return 0; ld += 2.0 * log(d); }
    }
    *logdet = ld;
    return 1;
}
/* factor Z(y) + r I of every block into LZ (and Z); 1 if all positive definite */
/* Z(y) + r I assembled straight into a sparse factor (triangle entries, O(nnz)); 1 if
 * positive definite */
static int ds_schol_y(const Block *B, SChol *F, const double *y, double r, double *logdet) {
    schol_zero(F);
    for (int q = 0; q < B->C.nnz; q++) schol_add(F, B->C.row[q], B->C.col[q], B->C.val[q]);
    for (int t = 0; t < B->ncon; t++) {
        const SpSym *A = &B->A[t];
        const double yi = y[B->con[t]];
        if (yi == 0) continue;
        for (int q = 0; q < A->nnz; q++) schol_add(F, A->row[q], A->col[q], -yi * A->val[q]);
    }
    if (r != 0) for (int i = 0; i < B->n; i++) schol_add(F, i, i, r);
    if (schol_factor(F, 0.0) != 0) return 0;
    double ld = schol_logdet(F);
    if (!isfinite(ld)) return 0;
    *logdet = ld;
    return 1;
}
/* sparse factor of a dense symmetric Z (full storage): 1 if positive definite */
static int ds_schol(SChol *F, const double *Zd, double *logdet) {
    schol_gather_dense(F, Zd);
    if (schol_factor(F, 0.0) != 0) return 0;
    double ld = schol_logdet(F);
    if (!isfinite(ld)) return 0;
    *logdet = ld;
    return 1;
}
static int ds_factor_all(const Problem *P, BS *S, const double *y, double r, double *logdet) {
    int ok = 1;
    PBLK_FOR(k, P->nblk, {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        ds_Z(B, s, y, r, s->Z);
        if (s->dzs) {
            if (!ds_schol(s->dzs, s->Z, &s->part[0])) {
                _Pragma("omp atomic write")
                ok = 0;
            }
            continue;
        }
        memcpy(s->LZ, s->Z, sizeof(double) * bsz(B));
        if (!ds_chol(B, s->LZ, &s->part[0])) {
            _Pragma("omp atomic write")
            ok = 0;
        }
    });
    if (!ok) return 0;
    double ld = 0;
    for (int k = 0; k < P->nblk; k++) ld += S[k].part[0];
    if (logdet) *logdet = ld;
    return 1;
}
/* PSD test of C - A'(yt) + r I using the scratch buffer G (does not touch LZ) */
static int ds_test(const Problem *P, BS *S, const double *yt, double r, double *logdet) {
    int ok = 1;
    PBLK_FOR(k, P->nblk, {
        if (!ok) continue;
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int okb;
        if (s->dzt) okb = ds_schol_y(B, s->dzt, yt, r, &s->part[1]);
        else { ds_Z(B, s, yt, r, s->G); okb = ds_chol(B, s->G, &s->part[1]); }
        if (!okb) {
            _Pragma("omp atomic write")
            ok = 0;
        }
    });
    if (!ok) return 0;
    if (logdet) { double ld = 0; for (int k = 0; k < P->nblk; k++) ld += S[k].part[1]; *logdet = ld; }
    return 1;
}
static void ds_inverse(const Problem *P, BS *S, int need_lx) {
    PBLK_FOR(k, P->nblk, {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        if (B->type == BLK_LP) { for (int i = 0; i < n; i++) s->Zi[i] = 1.0 / s->LZ[i]; continue; }
        if (s->dzs) {                                 /* Z^{-1} by sparse solves with I */
            memset(s->Zi, 0, sizeof(double) * bsz(B));
            for (int i = 0; i < n; i++) s->Zi[i + (size_t)i * n] = 1.0;
            schol_solve_many(s->dzs, s->Zi, n, s->W);
            symmetrize(n, s->Zi);
            continue;
        }
        memcpy(s->Zi, s->LZ, sizeof(double) * bsz(B));
        if (n <= PROD_SMALL) { inv_from_chol_small(n, s->Zi); lower_to_full(n, s->Zi); }
        else (void)brisk_dpotri_full(n, s->Zi, n);
        if (need_lx && B->lowrank) {                  /* low-rank route: LX = chol(X), X = Zi */
            memcpy(s->LX, s->Zi, sizeof(double) * bsz(B));
            if (chol_lower(n, s->LX)) {               /* rounding: fall back to a shifted factor */
                memcpy(s->LX, s->Zi, sizeof(double) * bsz(B));
                double dmax = 0;
                for (int i = 0; i < n; i++) dmax = fmax(dmax, s->Zi[i + (size_t)i * n]);
                for (int i = 0; i < n; i++) s->LX[i + (size_t)i * n] += 1e-14 * dmax;
                chol_lower(n, s->LX);
            }
        }
    });
}

/* largest a with Z + a dZ > 0 (capped at 2): Lanczos with the dense factor, or, for a
 * block with a sparse factor of Z, bisection with sparse Cholesky tests (in G / dzt) */
static double ds_blk_maxstep(const Block *B, BS *s, int lanczos_k, int m, const double *y, const double *dy,
                             double r, double dr) {
    if (B->type == BLK_LP) return lp_maxstep(B->n, s->Z, s->dZ, NULL);
    if (!s->dzt) return sdp_maxstep(B->n, s->LZ, s->dZ, s, lanczos_k);
    double ld, lo = 0.0, hi = 2.0;
    double *yt = malloc(sizeof(double) * (m + 1));
#define DSTRY(A) (({ for (int i_ = 0; i_ < m; i_++) yt[i_] = y[i_] + (A) * dy[i_]; ds_schol_y(B, s->dzt, yt, r + (A) * dr, &ld); }))
    if (DSTRY(hi)) { free(yt); return 1e30; }
    for (int t = 0; t < 12; t++) {
        double mid = 0.5 * (lo + hi);
        if (DSTRY(mid)) lo = mid; else hi = mid;
    }
#undef DSTRY
    free(yt);
    return lo;
}

/* bordered solve for the phase-1 variable r: [M u; u' c] [x; xr] = [f; g] with w = M^{-1} u */
static void ds_bordered(Schur *Sc, int m, const double *u, const double *w, double c,
                        const double *f, double g, double *x, double *xr) {
    schur_solve(Sc, f, x, 0);
    double uf = 0, uw = 0;
    for (int i = 0; i < m; i++) { uf += u[i] * x[i]; uw += u[i] * w[i]; }
    double sch = c - uw;
    if (!(sch > 1e-14 * fabs(c))) sch = 1e-14 * fabs(c) + 1e-300;
    *xr = (g - uf) / sch;
    for (int i = 0; i < m; i++) x[i] -= w[i] * (*xr);
}

/* certificate X(mu) = mu Zi (Z + A'(dy)) Zi: PSD, and A(X) = b to working accuracy?
 * X is written to s->X (scaled units). */
/* certificate X(mu) = mu Zi (Z + A'(dy)) Zi, made to satisfy A(X) = b to working accuracy
 * by one least-norm correction X += A'(w), (A A') w = b - A(X) (the Schur solves leave a
 * residual of cond(M) eps). X is written to s->X (scaled units). Returns the measured
 * maximum error (residual, PSD violation at the tolerance level, and the relative gap of
 * <C, X> against b'y) and sets *pobj_out = <C, X>. */
static double ds_verify(const Problem *P, BS *S, Schur *Sc, const double *dy, double mu, const double *y,
                        double tol, int cheap, double *res_out, double *ax, double *pobj_out,
                        double *r1_, double *r2_, double *r3_, double *wv_) {
    const int m = P->m;
    (void)r1_; (void)r2_; (void)r3_; (void)wv_;
    /* own work vectors: schur_solve uses the Schur scratch vectors */
    double *wbuf = calloc(4 * (size_t)m + 4, sizeof(double));
    double *r1 = wbuf, *r2 = wbuf + m + 1, *r3 = wbuf + 2 * (m + 1), *wv = wbuf + 3 * (m + 1);
    const double sc = fabs(P->bs * P->cs);     /* gap measured on the original objective scale */
    memset(ax, 0, sizeof(double) * m);
    double nb1 = 0, nbb = 0;
    for (int i = 0; i < m; i++) { nb1 += fabs(P->b[i]); nbb += P->b[i] * P->b[i]; }
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        blk_ATy(B, s, dy, s->F);                      /* A'(dy) */
        if (B->type == BLK_LP) {
            for (int i = 0; i < n; i++) s->X[i] = mu * (s->Zi[i] + s->Zi[i] * s->Zi[i] * s->F[i]);
        } else {
            xsymm("L", "L", &n, &n, &DONE, s->Zi, &n, s->F, &n, &DZERO, s->W, &n);
            xsymm("R", "L", &n, &n, &DONE, s->Zi, &n, s->W, &n, &DZERO, s->X, &n);
            for (size_t q = 0; q < len; q++) s->X[q] = mu * (s->Zi[q] + s->X[q]);
            symmetrize(n, s->X);
        }
        blk_Aop(B, s, s->X, ax, 1.0);
    }
    /* iterative refinement in the Zi metric: r = b - A(X), M w = r, X += Zi A'(w) Zi, i.e.
     * dy += w / mu with the true operator. It keeps the form mu Zi (Z + A'dy) Zi, so it does
     * not destroy positive semidefiniteness the way the least-norm correction does when X
     * is nearly singular (hinf) */
    double nbq = sqrt(nbb), res0 = 1e300;
    /* large blocks (each pass costs two n^3 products): stop once the residual is
     * negligible against the tolerance; otherwise refine to rounding level */
    const double rstop = cheap ? 1e-5 * tol * (1 + nbq) : 1e-15 * (1 + nbq);
    for (int pass = 0; pass < 3; pass++) {
        double rr = 0;
        for (int i = 0; i < m; i++) { r3[i] = P->b[i] - ax[i]; rr += r3[i] * r3[i]; }
        rr = sqrt(rr);
        if (!(rr < 0.5 * res0) || rr <= rstop) break;
        res0 = rr;
        schur_solve(Sc, r3, wv, 0);
        memset(ax, 0, sizeof(double) * m);
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            const int n = B->n;
            const size_t len = bsz(B);
            blk_ATy(B, s, wv, s->F);
            if (B->type == BLK_LP) for (int q = 0; q < n; q++) s->X[q] += s->Zi[q] * s->Zi[q] * s->F[q];
            else {
                xsymm("L", "L", &n, &n, &DONE, s->Zi, &n, s->F, &n, &DZERO, s->W, &n);
                xsymm("R", "L", &n, &n, &DONE, s->Zi, &n, s->W, &n, &DONE, s->X, &n);
                symmetrize(n, s->X);
            }
            memset(s->F, 0, sizeof(double) * len);
            blk_Aop(B, s, s->X, ax, 1.0);
        }
    }
    /* least-norm correction of what is left, kept only if X stays PSD */
    for (int i = 0; i < m; i++) r3[i] = P->b[i] - ax[i];
    int use_ln = 0;
    {
        double rr = 0; for (int i = 0; i < m; i++) rr += r3[i] * r3[i];
        use_ln = sqrt(rr) > rstop;
    }
    for (int k = 0; k < P->nblk; k++) memset(S[k].F, 0, sizeof(double) * bsz(&P->blk[k]));
    if (use_ln) {
        gram_cg(P, S, r3, wv, 300, 1e-13, r1, r2, ax);
        /* keep the refined X in Xtmp = G: the correction is undone if it breaks PSD */
        for (int k = 0; k < P->nblk; k++) memcpy(S[k].G, S[k].X, sizeof(double) * bsz(&P->blk[k]));
        memset(ax, 0, sizeof(double) * m);
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            const size_t len = bsz(B);
            blk_ATy(B, s, wv, s->F);
            for (size_t q = 0; q < len; q++) s->X[q] += s->F[q];
            memset(s->F, 0, sizeof(double) * len);
            blk_Aop(B, s, s->X, ax, 1.0);
        }
    }
    double res = 0;
    for (int i = 0; i < m; i++) { double r = ax[i] - P->b[i]; res += r * r; }
    res = sqrt(res);
    if (res_out) *res_out = res;
    /* PSD to the tolerance (DIMACS err2): singular at the optimum, so test X + delta I */
    const double xshift = 0.1 * tol * (1 + nb1);
    int psd = 1;
    for (int k = 0; k < P->nblk && psd; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        if (B->type == BLK_LP) { for (int i = 0; i < n; i++) if (s->X[i] < -xshift) psd = 0; continue; }
        memcpy(s->W, s->X, sizeof(double) * bsz(B));
        for (int i = 0; i < n; i++) s->W[i + (size_t)i * n] += xshift;
        if (chol_lower(n, s->W)) psd = 0;
    }
    if (!psd && use_ln) {                               /* undo the least-norm correction */
        memset(ax, 0, sizeof(double) * m);
        for (int k = 0; k < P->nblk; k++) {
            memcpy(S[k].X, S[k].G, sizeof(double) * bsz(&P->blk[k]));
            blk_Aop(&P->blk[k], &S[k], S[k].X, ax, 1.0);
        }
        res = 0;
        for (int i = 0; i < m; i++) { double r = ax[i] - P->b[i]; res += r * r; }
        res = sqrt(res);
        if (res_out) *res_out = res;
        psd = 1;
        for (int k = 0; k < P->nblk && psd; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            const int n = B->n;
            if (B->type == BLK_LP) { for (int i = 0; i < n; i++) if (s->X[i] < -xshift) psd = 0; continue; }
            memcpy(s->W, s->X, sizeof(double) * bsz(B));
            for (int i = 0; i < n; i++) s->W[i + (size_t)i * n] += xshift;
            if (chol_lower(n, s->W)) psd = 0;
        }
    }
    const double bty = ddot_n(m, P->b, y);
    double cx = 0;
    for (int k = 0; k < P->nblk; k++) cx += sp_inner(&P->blk[k], &P->blk[k].C, S[k].X);
    if (pobj_out) *pobj_out = cx;
    double err = fmax(res / (1 + sqrt(nbb)), sc * fabs(cx - bty) / (1 + sc * (fabs(cx) + fabs(bty))));
    free(wbuf);
    if (!psd) err = fmax(err, 1.0);
    return err;
}

/* Newton decrement of the barrier problem max b'y/mu + log det Z along dy(s) = d1 s - d2
 * (s = 1/mu): delta^2(s) = beta s^2 - 2 gamma s + kappa, beta = b'd1, gamma = a0'd1,
 * kappa = a0'd2 (= ||Z^{-1/2} dZ Z^{-1/2}||_F^2, so delta < 1 keeps Z + dZ > 0). Returns
 * the largest s with delta(s) <= tau, or the minimizer gamma/beta when there is none. */
static double ds_s_for(double beta, double gamma, double kappa, double tau) {
    if (!(beta > 0)) return 0;
    double disc = gamma * gamma - beta * (kappa - tau * tau);
    if (disc < 0) return gamma / beta;
    return (gamma + sqrt(disc)) / beta;
}

int dsdp_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout) {
    double t0 = wtime();
    set_fast_fp();
    #pragma omp parallel
    set_fast_fp();
    memset(R, 0, sizeof(*R));
    const int m = P->m, nb = P->nblk;
    const double sc = P->bs * P->cs;
    BS *S = calloc(nb, sizeof(BS));
    double ntot = 0;
    int any_lowrank = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int n = B->n;
        size_t len = bsz(B) * sizeof(double);
        s->n = n; s->type = B->type;
        double **bufs[] = { &s->X, &s->Z, &s->LX, &s->LZ, &s->Zi, &s->dX, &s->dZ, &s->F, &s->G, &s->Q, &s->W };
        for (size_t q = 0; q < sizeof(bufs) / sizeof(bufs[0]); q++) *bufs[q] = amalloc(len);
        memset(s->F, 0, len);
        s->lr_mem = par->dense_mem;
        if (B->type == BLK_SDP && B->lowrank) { lowrank_alloc(B, s); any_lowrank = 1; }
        if (B->type == BLK_SDP) {
            s->upat.nf = B->unf; s->upat.fr = B->ufr; s->upat.fc = B->ufc;
            s->upat.nr = B->unr; s->upat.rows = B->urows; s->upat.route = B->prod_route; s->upat.owned = 0;
        }
        if (B->type == BLK_SDP && B->dict) {
            const Block *vb = B->vb;
            size_t ql = (size_t)vb->n * vb->n * sizeof(double);
            BS *v = calloc(1, sizeof(BS));
            v->n = vb->n; v->type = BLK_SDP;
            v->X = amalloc(ql); v->Zi = amalloc(ql);
            if (vb->nd) {
                double cap = fmax(1.0, 256.0 * 1024 * 1024 / (8.0 * ql / sizeof(double)));
                v->chunk = (int)fmin((double)vb->nd, cap);
                v->dtmp = amalloc(sizeof(double) * vb->nd);
                v->Gd = amalloc(ql * v->chunk);
                v->Mdd = amalloc(sizeof(double) * (size_t)vb->nd * v->chunk);
            }
            s->vbs = v;
            s->dW = amalloc(sizeof(double) * (size_t)n * (B->dnv ? B->dnv : 1));
        }
        if (B->type == BLK_SDP) {
            if (B->prod_route == 0) rowbuf_reserve(s, B->unr);
            if (B->nd) {
                double cap = fmax(1.0, 256.0 * 1024 * 1024 / (8.0 * len / sizeof(double)));
                s->chunk = (int)fmin((double)B->nd, cap);
                s->dtmp = amalloc(sizeof(double) * B->nd);
                s->Gd = amalloc(len * s->chunk);
                s->Mdd = amalloc(sizeof(double) * (size_t)B->nd * s->chunk);
            }
            s->lq = amalloc(sizeof(double) * (size_t)n * (LANCZOS_K + 3));
            s->v1 = amalloc(sizeof(double) * n);
            s->v2 = amalloc(sizeof(double) * n);
        }
        ntot += n;
    }
    const int par_blk_prev = g_par_blk, blas_T_prev = g_blas_T;
    {
        int maxn_sdp = 0, nsdp = 0;
        for (int k = 0; k < nb; k++)
            if (P->blk[k].type == BLK_SDP) { nsdp++; if (P->blk[k].n > maxn_sdp) maxn_sdp = P->blk[k].n; }
        g_par_blk = par->par_blocks > 0 || (par->par_blocks < 0 && nthreads() > 1 && nsdp >= 2 && maxn_sdp <= 256);
        if (nthreads() <= 1) g_par_blk = 0;
        if (g_par_blk && BL(openblas_get_num_threads) && BL(openblas_set_num_threads) && g_blas_T == 0) {
            g_blas_T = BL(openblas_get_num_threads)();
            if (g_blas_T > 1) BL(openblas_set_num_threads)(1);
        }
    }
    /* sparse Z: blocks whose aggregate pattern (C and every A_t) is sparse and factors
     * with little fill (max-cut, box QP: Z = C - A'y keeps the graph's pattern) */
    int nzsparse = 0, maxn_all = 0;
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        if (B->type == BLK_SDP && B->n > maxn_all) maxn_all = B->n;
        if (B->type != BLK_SDP || B->n < par->dual_sparse_n || B->lowrank || B->dict || B->nd) continue;
        const int n = B->n;
        int *cnt = calloc(n, sizeof(int)), *mk = malloc(sizeof(int) * n);
        long tot = 0;
        const SpSym *lists[1];
        (void)lists;
        /* collect entries: C and all A_t (full storage fr/fc) */
        int ntot_e = B->C.ef;
        for (int t2 = 0; t2 < B->ncon; t2++) ntot_e += B->A[t2].ef;
        int *er = malloc(sizeof(int) * (ntot_e + 1)), *ec = malloc(sizeof(int) * (ntot_e + 1));
        int ne = 0;
        for (int q = 0; q < B->C.ef; q++) { er[ne] = B->C.fr[q]; ec[ne] = B->C.fc[q]; ne++; }
        for (int t2 = 0; t2 < B->ncon; t2++)
            for (int q = 0; q < B->A[t2].ef; q++) { er[ne] = B->A[t2].fr[q]; ec[ne] = B->A[t2].fc[q]; ne++; }
        for (int q = 0; q < ne; q++) if (er[q] != ec[q]) cnt[er[q]]++;
        int **nbr = malloc(sizeof(int *) * n), *deg = calloc(n, sizeof(int));
        for (int i = 0; i < n; i++) nbr[i] = malloc(sizeof(int) * (cnt[i] + 1));
        for (int i = 0; i < n; i++) mk[i] = -1;
        /* dedupe per row (rows visited in entry order; mark by row id via a second pass) */
        for (int q = 0; q < ne; q++) if (er[q] != ec[q]) nbr[er[q]][deg[er[q]]++] = ec[q];
        for (int i = 0; i < n; i++) {
            int d = 0;
            for (int a = 0; a < deg[i]; a++) { int j = nbr[i][a]; if (mk[j] != i) { mk[j] = i; nbr[i][d++] = j; } }
            deg[i] = d; tot += d;
        }
        /* make symmetric (entries may be stored one-sided) */
        int *deg2 = calloc(n, sizeof(int));
        for (int i = 0; i < n; i++) for (int a = 0; a < deg[i]; a++) deg2[nbr[i][a]]++;
        int **nb2 = malloc(sizeof(int *) * n);
        for (int i = 0; i < n; i++) { nb2[i] = malloc(sizeof(int) * (deg[i] + deg2[i] + 1)); deg2[i] = 0; }
        for (int i = 0; i < n; i++) for (int a = 0; a < deg[i]; a++) { int j = nbr[i][a]; nb2[i][deg2[i]++] = j; nb2[j][deg2[j]++] = i; }
        tot = 0;
        for (int i = 0; i < n; i++) mk[i] = -1;
        for (int i = 0; i < n; i++) {
            int d = 0;
            for (int a = 0; a < deg2[i]; a++) { int j = nb2[i][a]; if (mk[j] != i) { mk[j] = i; nb2[i][d++] = j; } }
            deg2[i] = d; tot += d;
        }
        if ((double)tot <= 0.05 * (double)n * n) {
            size_t cap = (size_t)(0.1 * (double)n * n) + 1000;
            s->dzs = schol_analyze_adj(n, deg2, nb2, cap);
            if (s->dzs && (double)schol_nnz(s->dzs) > 0.1 * (double)n * n) { schol_free(s->dzs); s->dzs = NULL; }
            if (s->dzs) s->dzt = schol_analyze_adj(n, deg2, nb2, cap);
            if (s->dzs && !s->dzt) { schol_free(s->dzs); s->dzs = NULL; }
            if (s->dzs) {
                nzsparse++;
                if (par->verbose > 0)
                    printf("dual method: block %d (n = %d): sparse Cholesky of Z, %zu nonzeros in L (%.1f%% of n^2/2)\n",
                           k + 1, n, schol_nnz(s->dzs), 200.0 * schol_nnz(s->dzs) / ((double)n * n));
            }
        }
        for (int i = 0; i < n; i++) { free(nbr[i]); free(nb2[i]); }
        free(nbr); free(nb2); free(deg); free(deg2); free(cnt); free(mk); free(er); free(ec);
    }
    /* large blocks: every verified certificate costs O(n^3) (X, its PSD test), so it is
     * verified only when its analytic gap is 1% of the best measured error, is near the
     * tolerance, or every 10 iterations; the analytic bound drives mu in between */
    const int big_verify = maxn_all >= par->dual_verify_n;
    int it_last_verify = -100;
    double t_verify = 0, t_inv = 0;
    /* correctors pay when the Schur complement costs much more than Z^{-1} (theta, gpp,
     * control); for max-cut M = Z^{-1} o Z^{-1} costs less than Z^{-1} itself, and a
     * corrector is as expensive as an iteration (deterministic flop estimates): there they
     * stop at decrement 1.5 instead of 0.9 (none at all stalls maxG51 uncentred) */
    int ncorr_max = par->dual_corr;
    double corr_tau = par->dual_tau;     /* corrector target decrement */
    {
        double inv = 0, sch = (double)m * m * m / 3.0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            if (B->type != BLK_SDP) continue;
            const double n = B->n;
            /* sparse: the multi-RHS solve is bound by its scattered updates (~75 flop-
             * equivalents each, measured on maxG11) */
            inv += S[k].dzs ? 150.0 * n * (double)schol_nnz(S[k].dzs) : n * n * n;
            sch += 2.0 * n * n * (double)B->ncon;      /* assembly, roughly */
        }
        /* only where Z^{-1} is expensive in absolute terms (small problems keep the
         * correctors: arch0, ss30, hinf3/7 need them) */
        if (par->dual_corr_auto && inv > 5e7 && sch < 8.0 * inv) corr_tau = fmax(corr_tau, 1.5);
        if (par->verbose > 1) printf("dual method: Schur %.1e vs Z^-1 %.1e flops: correctors down to decrement %.2g\n", sch, inv, corr_tau);
    }
    Schur Sc = { 0 };
    Sc.m = m;
    Sc.SC = sparse_build(P, par, par->verbose > 1);
    schur_asm_setup(&Sc);
    schur_pos_setup(&Sc, P);
    Sc.E = Sc.SC ? NULL : envelope_build(P, par, par->verbose > 1);
    if (Sc.E || Sc.SC) {
        Sc.e1 = malloc(sizeof(double) * m); Sc.e2 = malloc(sizeof(double) * m); Sc.e3 = malloc(sizeof(double) * m);
    } else {
        mem_check(8.0 * (double)m * (double)m, "the dense Schur complement", m);
        Sc.M = amalloc(sizeof(double) * (size_t)m * m);
        Sc.Mc = Sc.M;                                   /* 4.26: the factor in place */
        Sc.Md = malloc(sizeof(double) * (m + 1)); Sc.low_M = 1;
    }
    Sc.D = malloc(sizeof(double) * m); Sc.r = malloc(sizeof(double) * m); Sc.t = malloc(sizeof(double) * m);
    Sc.p = malloc(sizeof(double) * m); Sc.q = malloc(sizeof(double) * m); Sc.xb = malloc(sizeof(double) * m);
    Sc.fw = malloc(sizeof(float) * (m ? m : 1));
    Sc.allow_float = !Sc.E && (par->mixed > 0 || (par->mixed < 0 && m >= 1000));
    Sc.pivtol = par->pivtol;
    Sc.reg_ill = par->reg_ill;

    double *y = calloc(m + 1, sizeof(double)), *yt = calloc(m + 1, sizeof(double));
    double *a0 = calloc(m + 1, sizeof(double)), *d1 = calloc(m + 1, sizeof(double)), *d2 = calloc(m + 1, sizeof(double));
    double *dy = calloc(m + 1, sizeof(double)), *dycert = calloc(m + 1, sizeof(double));
    double *u = calloc(m + 1, sizeof(double)), *w = calloc(m + 1, sizeof(double)), *zero = calloc(m + 1, sizeof(double));
    double *ax = calloc(m + 1, sizeof(double)), *rq = calloc(m + 1, sizeof(double));

    /* ---- Schur assembly with X = Zi */
#define DS_ASSEMBLE() do { \
        double *Ma_ = schur_asm_begin(&Sc); \
        for (int k_ = 0; k_ < nb; k_++) { BS *s_ = &S[k_]; double *xs_ = s_->X; s_->X = s_->Zi; \
            if (P->blk[k_].type == BLK_LP) schur_lp(&P->blk[k_], s_, m, Ma_); \
            else schur_sdp(&P->blk[k_], s_, m, Ma_); \
            s_->X = xs_; } \
        schur_asm_end(&Sc); } while (0)

    /* ---- start */
    int status = 3, it = 0, phase1_its = 0, ncert = 0, ncorr = 0, nverify = 0;
    double r = 0, logdet = 0;
    const char *start = "y = 0";
    if (!ds_test(P, S, y, 0.0, NULL) && par->dual_gstart) {
        /* Gershgorin start: Z = C + D with D_ii = (off-diagonal row sum - C_ii)_+ plus a 10%
         * margin, which is diagonally dominant, when D is in the range of A' (every diagonal
         * is, for max-cut and gpp). Unlike C + tI it follows the row sums: on maxG51 the
         * uniform shift t ~ 220 started at b'y = -2.2e5 against an optimum near -4e3 with
         * very uneven degrees, and the method stalled uncentred. */
        memset(rq, 0, sizeof(double) * m);
        double dn2 = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            const int n = B->n;
            memset(s->W, 0, sizeof(double) * bsz(B));
            double *Rs = calloc(n, sizeof(double)), *Cd = calloc(n, sizeof(double)), cmax = 0;
            for (int q = 0; q < B->C.nnz; q++) {
                const int a = B->C.row[q], c = B->C.col[q];
                const double v = B->C.val[q];
                cmax = fmax(cmax, fabs(v));
                if (B->type == BLK_LP || a == c) Cd[a] += v;
                else { Rs[a] += fabs(v); Rs[c] += fabs(v); }
            }
            for (int i = 0; i < n; i++) {
                double d = fmax(0.0, Rs[i] - Cd[i]) + 0.1 * (Rs[i] + fabs(Cd[i])) + 1e-3 * (1 + cmax);
                if (B->type == BLK_LP) s->W[i] = -d; else s->W[i + (size_t)i * n] = -d;
                dn2 += d * d;
            }
            free(Rs); free(Cd);
            blk_Aop(B, s, s->W, rq, 1.0);
        }
        double rres = gram_cg(P, S, rq, w, 500, 1e-12, Sc.r, Sc.p, Sc.q);
        for (int k = 0; k < nb; k++) memset(S[k].F, 0, sizeof(double) * bsz(&P->blk[k]));
        /* residual ||D + A'(w)|| / ||D||, with W still holding -D */
        double e2 = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            blk_ATy(B, s, w, s->F);
            for (size_t q = 0; q < bsz(B); q++) { double v = s->F[q] - s->W[q]; e2 += v * v; }
            memset(s->F, 0, sizeof(double) * bsz(B));
        }
        if (rres < 1e-8 && sqrt(e2 / fmax(dn2, 1e-300)) < 1e-8 && ds_test(P, S, w, 0.0, NULL)) {
            memcpy(y, w, sizeof(double) * m);
            start = "Gershgorin diagonal in the range of A'";
        }
    }
    if (!ds_test(P, S, y, 0.0, NULL)) {
        /* least squares A'(y_I) ~ -I: rhs = A(-I) */
        memset(rq, 0, sizeof(double) * m);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            memset(s->W, 0, sizeof(double) * bsz(B));
            if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) s->W[i] = -1.0;
            else for (int i = 0; i < B->n; i++) s->W[i + (size_t)i * B->n] = -1.0;
            blk_Aop(B, s, s->W, rq, 1.0);
        }
        double rres = gram_cg(P, S, rq, w, 500, 1e-12, Sc.r, Sc.p, Sc.q);
        for (int k = 0; k < nb; k++) memset(S[k].F, 0, sizeof(double) * bsz(&P->blk[k]));
        /* residual ||I + A'(y_I)||_F relative to ||I||_F */
        double e2 = 0;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            blk_ATy(B, s, w, s->W);
            const int n = B->n;
            for (int i = 0; i < n; i++) {
                if (B->type == BLK_LP) { double v = 1 + s->W[i]; e2 += v * v; continue; }
                for (int j = 0; j < n; j++) { double v = (i == j) + s->W[i + (size_t)j * n]; e2 += v * v; }
            }
        }
        const double idres = sqrt(e2 / ntot);
        if (rres < 1e-8 && idres < 1e-8) {
            /* Z(t y_I) = C + t I: t above -lambda_min(C), doubled until positive definite */
            double t = 1.0, cn = 0;
            for (int k = 0; k < nb; k++) for (int q = 0; q < P->blk[k].C.ef; q++) cn = fmax(cn, fabs(P->blk[k].C.fv[q]));
            t = 1.0 + cn;
            int ok = 0;
            for (int tries = 0; tries < 60 && !ok; tries++) {
                for (int i = 0; i < m; i++) yt[i] = t * w[i];
                ok = ds_test(P, S, yt, 0.0, NULL);
                if (!ok) t *= 2;
            }
            if (ok) {
                /* back off toward the smallest t that keeps Z > 0 (a smaller |y| start) */
                double lo = 0, hi = t;
                for (int s2 = 0; s2 < 12; s2++) {
                    double mid = 0.5 * (lo + hi);
                    for (int i = 0; i < m; i++) yt[i] = mid * w[i];
                    if (ds_test(P, S, yt, 0.0, NULL)) hi = mid; else lo = mid;
                }
                t = 2.0 * hi + 1.0;                  /* comfortably inside */
                for (int i = 0; i < m; i++) y[i] = t * w[i];
                if (!ds_test(P, S, y, 0.0, NULL)) for (int i = 0; i < m; i++) y[i] = hi * 4.0 * w[i];
                start = "identity in the range of A': y = t y_I";
            }
        }
        if (!ds_test(P, S, y, 0.0, NULL)) {
            /* phase 1: Z = C - A'(y) + r I, maximize -r until r < 0 */
            memset(y, 0, sizeof(double) * m);
            double gl = 1e300;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                const int n = B->n;
                ds_Z(B, s, y, 0.0, s->Z);
                for (int i = 0; i < n; i++) {
                    double v;
                    if (B->type == BLK_LP) v = s->Z[i];
                    else { double rs = 0; for (int j = 0; j < n; j++) if (j != i) rs += fabs(s->Z[i + (size_t)j * n]); v = s->Z[i + (size_t)i * n] - rs; }
                    if (v < gl) gl = v;
                }
            }
            r = fmax(0.0, -gl) + 1.0;
            start = "phase 1";
            double s1 = 0;
            for (phase1_its = 0; phase1_its < 60; phase1_its++) {
                if (!ds_factor_all(P, S, y, r, &logdet)) { status = 4; break; }
                if (r < 0) break;
                ds_inverse(P, S, any_lowrank);
                memset(a0, 0, sizeof(double) * m);
                double a0r = 0, c = 0;
                memset(u, 0, sizeof(double) * m);
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    const int n = B->n;
                    blk_Aop(B, s, s->Zi, a0, 1.0);
                    if (B->type == BLK_LP) {
                        for (int i = 0; i < n; i++) { s->W[i] = s->Zi[i] * s->Zi[i]; a0r -= s->Zi[i]; c += s->W[i]; }
                    } else {
                        BL(dsyrk_)("L", "N", &n, &n, &DONE, s->Zi, &n, &DZERO, s->W, &n);
                        lower_to_full(n, s->W);
                        for (int i = 0; i < n; i++) { a0r -= s->Zi[i + (size_t)i * n]; c += s->W[i + (size_t)i * n]; }
                    }
                    blk_Aop(B, s, s->W, u, -1.0);             /* u_i = <A_i, Zi A_r Zi>, A_r = -I */
                }
                DS_ASSEMBLE();
                if (schur_factor(&Sc) != 0) { status = 4; break; }
                schur_solve(&Sc, u, w, 0);
                /* Newton direction of max -r/mu1 + log det Z (b1 = (0, -1)) with mu1 reduced
                 * geometrically according to the step taken (no recentring: the phase-1
                 * barrier has no centre when the dual feasible set is unbounded) */
                if (s1 <= 0) {
                    double tz = 0;
                    for (int k = 0; k < nb; k++) {
                        const Block *B = &P->blk[k];
                        for (int i = 0; i < B->n; i++) tz += B->type == BLK_LP ? S[k].Z[i] : S[k].Z[i + (size_t)i * B->n];
                    }
                    s1 = ntot / fmax(tz, 1e-300);
                }
                double d1r, d2r;
                /* objective eps b'y - r (Gamma = 1/eps): the phase-1 path then ends near
                 * the phase-2 central path at s = eps s1 instead of at an arbitrary point */
                const double eps1 = par->dual_gamma > 0 ? 1.0 / par->dual_gamma : 0.0;
                for (int i = 0; i < m; i++) rq[i] = eps1 * P->b[i];
                ds_bordered(&Sc, m, u, w, c, rq, -1.0, d1, &d1r);
                ds_bordered(&Sc, m, u, w, c, a0, a0r, d2, &d2r);
                const double del2 = 0;
                for (int i = 0; i < m; i++) dy[i] = d1[i] * s1 - d2[i];
                double dr = d1r * s1 - d2r;
                /* step length */
                double amax = 1e30;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    const int n = B->n;
                    blk_ATy(B, s, dy, s->dZ);
                    for (size_t q = 0; q < bsz(B); q++) s->dZ[q] = -s->dZ[q];
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->dZ[i] += dr;
                    else for (int i = 0; i < n; i++) s->dZ[i + (size_t)i * n] += dr;
                    double ak = ds_blk_maxstep(B, s, par->lanczos_k, m, y, dy, r, dr);
                    if (!(ak >= 0)) ak = 0;
                    if (ak < amax) amax = ak;
                }
                double a = fmin(1.0, 0.9 * amax);
                if (dr < 0 && r + a * dr < 0) a = fmin(a, 1.5 * r / (-dr));   /* just past r = 0 */
                for (int tries = 0; tries < 30; tries++) {
                    for (int i = 0; i < m; i++) yt[i] = y[i] + a * dy[i];
                    if (ds_test(P, S, yt, r + a * dr, NULL)) break;
                    a *= 0.7;
                }
                s1 /= (a > 0.8 ? 0.2 : a > 0.4 ? 0.5 : 0.9);
                memcpy(y, yt, sizeof(double) * m);
                r += a * dr;
                if (par->verbose > 1) printf("   dual phase 1 %2d: r %.3e step %.2f (decrement %.2e, 1/mu %.2e)\n", phase1_its, r, a, sqrt(del2), s1);
            }
            if (r >= 0 || status == 4) {
                if (par->verbose >= 0) printf("dual method: no strictly feasible dual point found (phase 1, %d iterations)\n", phase1_its);
                status = 4;
            }
            r = 0;
        }
    }
    const double rho_n = par->dual_rho * ntot;
    double mu = 0, ub = 0, mucert = 0, bty = 0, gap = 0, ubnow = 0, last_res = 0;
    double best_err = 1e300, best_gap = 0, best_res = 0;
    int it_best_err = 0;
    double *ybest = calloc(m + 1, sizeof(double));
    int have_ub = 0, nskip = 0;
    if (status != 4) {
        ds_factor_all(P, S, y, 0.0, &logdet);
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            for (int i = 0; i < B->n; i++) mu += B->type == BLK_LP ? S[k].Z[i] : S[k].Z[i + (size_t)i * B->n];
        }
        mu /= ntot;
        if (par->verbose >= 0) printf("dual method: start %s%s\n", start, phase1_its ? "" : "");
    }

    for (it = 0; status != 4 && it <= par->maxit; it++) {
        if (brisk_time_up(par)) { status = ST_TIME; break; }
        /* LZ, Z of the current y are valid (from the accepted step) */
        { double ti = wtime(); ds_inverse(P, S, any_lowrank); t_inv += wtime() - ti; }
        bty = ddot_n(m, P->b, y);
        memset(a0, 0, sizeof(double) * m);
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].Zi, a0, 1.0);
        double ts = wtime();
        DS_ASSEMBLE();
        double tc = wtime();
        R->t_schur += tc - ts;
        if (schur_factor(&Sc) != 0) { status = 4; R->t_chol += wtime() - tc; break; }
        schur_solve(&Sc, P->b, d1, 0);
        schur_solve(&Sc, a0, d2, 0);
        R->t_chol += wtime() - tc;
        const double a_d1 = ddot_n(m, a0, d1), a_d2 = ddot_n(m, a0, d2);
        const double slope = ntot - a_d2;

        /* ---- certificate: gap(mu) = mu slope + a0'd1, certified when C - A'(y - dy(mu)) > 0 */
        int cert_now = 0;
        const double est = mu * ntot;
        const int want = !have_ub || it < 3 || est < 0.8 * (ub - bty) || sc * est / (1 + fabs(sc * bty + P->obj_off)) < 10 * par->tol;
        double tst = wtime();
        if (want) {
            for (int attempt = 0; attempt < 2 && !cert_now; attempt++) {
                double lo = 1e-16 * (1 + fabs(bty)), hi = fmax(mu * 100.0, 1.0), mtest = mu;
                int steps = par->dual_cert;
                if (attempt == 0) {
                    if (!(mucert > 0 && it > 0)) continue;
                    lo = fmax(lo, mucert / 16.0);
                    mtest = fmax(mtest, mucert / 4.0);
                    steps = par->dual_cert / 2 + 2;
                }
                double gbest = -1, mubest = 0;
#define DSFEAS(MU) (({ for (int i2 = 0; i2 < m; i2++) yt[i2] = y[i2] - (d1[i2] / (MU) - d2[i2]); ds_test(P, S, yt, 0.0, NULL); }))
                int f0 = DSFEAS(mtest);
                if (f0) {
                    double g0 = mtest * slope + a_d1;
                    if (g0 > 0) { gbest = g0; mubest = mtest; }
                    if (slope > 0) {
                        double fl = lo, fh = mtest;
                        for (int s2 = 0; s2 < steps; s2++) {
                            double mid = sqrt(fl * fh);
                            if (DSFEAS(mid)) { fh = mid; double g2 = mid * slope + a_d1; if (g2 > 0 && g2 < gbest) { gbest = g2; mubest = mid; } }
                            else fl = mid;
                        }
                    } else {
                        double fl = mtest, fh = hi;
                        for (int s2 = 0; s2 < steps; s2++) {
                            double mid = sqrt(fl * fh);
                            if (DSFEAS(mid)) { fl = mid; double g2 = mid * slope + a_d1; if (g2 > 0 && g2 < gbest) { gbest = g2; mubest = mid; } }
                            else fh = mid;
                        }
                    }
                } else {
                    double mt = mtest;
                    for (int s2 = 0; s2 < steps && !f0; s2++) { mt *= 3.0; f0 = DSFEAS(mt); }
                    if (f0) { double g0 = mt * slope + a_d1; if (g0 > 0) { gbest = g0; mubest = mt; } }
                }
#undef DSFEAS
                if (gbest > 0 && slope > 0) {                 /* back off the cone boundary */
                    double mm2 = mubest * par->dual_margin;
                    if (mm2 <= mtest) { mubest = mm2; gbest = mm2 * slope + a_d1; }
                }
                ncert++;
                const double grel = sc * gbest / (1 + fabs(sc * bty + P->obj_off));
                if (gbest > 0 && big_verify && best_err < 1e300 && !(grel <= 0.01 * best_err || grel < 10 * par->tol || it - it_last_verify >= 10)) {
                    /* analytic bound, used for mu only (termination needs a measured one) */
                    cert_now = 1;
                    mucert = mubest;
                    ubnow = bty + gbest;
                    if (!have_ub || ubnow < ub) { ub = ubnow; have_ub = 1; }
                    if (par->verbose > 1) printf("     [certificate: mu %.2e, analytic gap %.2e (not verified)]\n", mubest, gbest);
                } else if (gbest > 0) {
                    it_last_verify = it;
                    /* the bound is taken from the measured primal point: X(mu) after the
                     * least-norm correction must be PSD, and its objective <C, X> is the
                     * bound (the analytic gap assumes exact solves, and ill-conditioned
                     * Schur systems made it wrong by orders of magnitude on truss6) */
                    for (int i = 0; i < m; i++) dycert[i] = d1[i] / mubest - d2[i];
                    double res = 0, cx = 0;
                    double tver = wtime();
                    const double err = ds_verify(P, S, &Sc, dycert, mubest, y, par->tol, big_verify, &res, ax, &cx, Sc.r, Sc.p, Sc.q, rq);
                    t_verify += wtime() - tver;
                    nverify++;
                    if (par->verbose > 1) printf("     [certificate: mu %.2e, measured error %.2e (residual %.1e, <C,X> - b'y %.2e vs %.2e)]\n",
                                                 mubest, err, res, cx - bty, gbest);
                    if (err < 1.0 && cx > bty - 1e-12 * (1 + fabs(bty))) {
                        cert_now = 1;
                        ubnow = fmax(cx, bty);
                        mucert = mubest;
                        if (!have_ub || ubnow < ub) { ub = ubnow; have_ub = 1; }
                        if (err < best_err) {
                            best_err = err; it_best_err = it; best_gap = cx - bty; best_res = res;
                            memcpy(ybest, y, sizeof(double) * m);
                            for (int k = 0; k < nb; k++) {
                                const size_t len = bsz(&P->blk[k]);
                                if (!S[k].Xb) S[k].Xb = amalloc(len * sizeof(double));
                                memcpy(S[k].Xb, S[k].X, len * sizeof(double));
                            }
                        }
                    }
                }
            }
        } else nskip++;
        R->t_step += wtime() - tst;
        gap = have_ub ? ub - bty : mu * ntot;
        const double rel = sc * gap / (1 + fabs(sc * bty + P->obj_off));
        R->iters = it;
        if (par->verbose)
            printf("%3d  dual obj %+.10e   gap %.2e   mu %.2e%s   %6.2f\n", it, sc * bty, rel, mu,
                   have_ub ? "" : " (no certificate)", wtime() - t0);
        if (have_ub && ub - bty < -1e-10 * (1 + fabs(bty))) {
            /* b'y above the certified bound: the certificate was inexact (ill-conditioned
             * solves); drop it */
            have_ub = 0;
            if (par->verbose > 1) printf("     [upper bound below b'y: discarded]\n");
        }
        if (best_err <= 0.5 * par->tol) { status = 0; break; }   /* margin for the unscaled measures */
        /* measured accuracy no longer improving: return the best certificate */
        if (best_err <= par->red_acc && it - it_best_err >= 6) { status = best_err <= 0.5 * par->tol ? 0 : 5; break; }
        if (it == par->maxit) { status = 3; break; }
        if (!have_ub && it >= par->dual_giveup) { status = 4; break; }

        /* ---- step: long-step barrier method on f_s(y) = -s b'y - log det Z(y), s = 1/mu.
         * When the iterate is centred for the current s (Newton decrement <= tau), s is
         * raised to the largest value whose decrement is tau_long; otherwise s is kept
         * and the damped Newton step recentres. Every step decreases f_s (backtracking
         * from min(1, 0.95 of the way to the boundary)). The certified bound is used for
         * termination; with dual_murule 0 mu follows gap / (rho n) (potential reduction). */
        const double beta = ddot_n(m, P->b, d1);
        double sk = 1.0 / mu;
        double dl2 = fmax(beta * sk * sk - 2 * a_d1 * sk + a_d2, 0.0);
        if (par->dual_murule == 1) {
            if (dl2 <= par->dual_tau * par->dual_tau) {
                const double s_long = ds_s_for(beta, a_d1, a_d2, par->dual_tau_long);
                if (s_long > sk) sk = s_long;
            }
            if (have_ub) sk = fmin(sk, rho_n / fmax(1e-3 * gap, 1e-300));   /* not far below gap/(rho n) */
        } else if (have_ub) {
            sk = 1.0 / fmax(gap / rho_n, 1e-14 * (1 + fabs(bty)));
            /* ... but, while the iterate is not centred for the previous mu, at most 10x
             * below it: the first certificate can be far
             * better than the iterate's centrality (maxG51: found at mu 4e-16, it cut mu
             * 15x and then 6x, and the uncentred iterate never produced another one) */
            if (par->dual_mudrop > 0 && mu > 0 && sk > 1.0 / (par->dual_mudrop * mu)) {
                const double s0 = 1.0 / mu;             /* decrement at the previous mu */
                const double d0 = beta * s0 * s0 - 2 * a_d1 * s0 + a_d2;
                if (d0 > par->dual_tau * par->dual_tau) sk = 1.0 / (par->dual_mudrop * mu);   /* only when not centred */
            }
            /* when the iterate is well centred for gap / (rho n), go further: the largest
             * s whose decrement is dual_tau_ub (the bound can stall when the certificates
             * stop improving, hinf2 sat at mu = gap / (rho n) with decrement 1e-3) */
            if (par->dual_tau_ub > 0) {
                const double sl = ds_s_for(beta, a_d1, a_d2, par->dual_tau_ub);
                if (sl > sk && sl < 1e14 / (1 + fabs(bty))) sk = sl;
            }
        } else {
            /* no bound yet: the smallest mu whose Newton decrement is tau_nb, or the most
             * central mu (smallest decrement) if there is none */
            const double snb = ds_s_for(beta, a_d1, a_d2, par->dual_tau_nb);
            if (snb > 0) sk = snb;
        }
        dl2 = fmax(beta * sk * sk - 2 * a_d1 * sk + a_d2, 0.0);
        mu = 1.0 / sk;
        for (int i = 0; i < m; i++) dy[i] = d1[i] * sk - d2[i];
        tst = wtime();
        double amax = 1e30;
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            blk_ATy(B, s, dy, s->dZ);
            for (size_t q = 0; q < bsz(B); q++) s->dZ[q] = -s->dZ[q];
            double ak = ds_blk_maxstep(B, s, par->lanczos_k, m, y, dy, 0.0, 0.0);
            if (!(ak >= 0)) ak = 0;
            if (ak < amax) amax = ak;
        }
        const int pot = par->dual_murule != 1 && have_ub;
        const double phi0 = (pot ? rho_n * log(fmax(ub - bty, 1e-300)) : -sk * bty) - logdet;
        /* directional derivative of f_s along dy is -delta^2 (Newton direction) */
        const double slope_f = pot ? -1.0 : -dl2;
        double a = fmin(1.0, 0.95 * amax), ldn = 0;
        int accepted = 0;
        if (pot && par->dual_ls > 0) {
            /* potential line search over a few step lengths up to 0.95 of the way to the
             * boundary (steps longer than the Newton step are allowed) */
            const double fr[6] = { 0.95, 0.8, 0.6, 0.4, 0.25, 0.1 };
            double best = phi0, abest = 0, ldb = 0;
            for (int c = -1; c < 6; c++) {
                double at = c < 0 ? 1.0 : fr[c] * amax;
                if (at > 0.95 * amax || at > 1.0 || !(at > 0)) continue;
                for (int i = 0; i < m; i++) yt[i] = y[i] + at * dy[i];
                double bt = ddot_n(m, P->b, yt), ldt;
                if (ub - bt <= 0 || !ds_test(P, S, yt, 0.0, &ldt)) continue;
                double phi = rho_n * log(ub - bt) - ldt;
                if (phi < best) { best = phi; abest = at; ldb = ldt; }
            }
            (void)ldb;
            if (abest > 0) {
                a = abest;
                for (int i = 0; i < m; i++) yt[i] = y[i] + a * dy[i];
                accepted = ds_factor_all(P, S, yt, 0.0, &ldn);
            }
        }
        for (int tries = 0; tries < 30 && !accepted; tries++) {
            for (int i = 0; i < m; i++) yt[i] = y[i] + a * dy[i];
            double bt = ddot_n(m, P->b, yt);
            if (!(pot && ub - bt <= 0) && ds_factor_all(P, S, yt, 0.0, &ldn)) {
                double phi = (pot ? rho_n * log(ub - bt) : -sk * bt) - ldn;
                const double armijo = pot ? 1e-12 * fabs(phi0) : 1e-4 * a * slope_f;
                if (phi <= phi0 + armijo || tries >= 12) { accepted = 1; break; }
            }
            a *= 0.6;
        }
        R->t_step += wtime() - tst;
        if (!accepted) {
            if (!ds_factor_all(P, S, y, 0.0, &logdet)) { status = 4; break; }
            mu *= 2.0;
            continue;
        }
        memcpy(y, yt, sizeof(double) * m);
        logdet = ldn;
        if (par->verbose > 1) printf("     [step %.3f (max %.3f), mu %.2e, decrement %.2e, |y| %.2e, logdet %.3e]\n", a, amax, mu, sqrt(dl2), sqrt(ddot_n(m, y, y)), logdet);
        /* ---- corrector steps (DSDP): recentre for the same mu with the factor of M from
         * this iteration, which is not recomputed. Each costs Z^{-1}, A(Z^{-1}), one solve
         * and a step; they stop when the (approximate) decrement is small or the barrier
         * no longer decreases. */
        double dprev = dl2;
        for (int cst = 0; cst < ncorr_max; cst++) {
            { double ti = wtime(); ds_inverse(P, S, 0); t_inv += wtime() - ti; }
            memset(a0, 0, sizeof(double) * m);
            for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].Zi, a0, 1.0);
            double tcs = wtime();
            schur_solve(&Sc, a0, d2, 0);
            R->t_chol += wtime() - tcs;
            for (int i = 0; i < m; i++) dy[i] = d1[i] * sk - d2[i];
            double dc2 = 0;
            for (int i = 0; i < m; i++) dc2 += dy[i] * (P->b[i] * sk - a0[i]);
            if (!(dc2 > corr_tau * corr_tau)) break;
            if (cst > 0 && dc2 > dprev) break;            /* the factor of M is stale */
            dprev = dc2;
            double am = 1e30;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                blk_ATy(B, s, dy, s->dZ);
                for (size_t q = 0; q < bsz(B); q++) s->dZ[q] = -s->dZ[q];
                double ak = ds_blk_maxstep(B, s, par->lanczos_k, m, y, dy, 0.0, 0.0);
                if (!(ak >= 0)) ak = 0;
                if (ak < am) am = ak;
            }
            const double f0c = -sk * ddot_n(m, P->b, y) - logdet;
            double ac = fmin(1.0, 0.95 * am);
            if (dc2 > 1) ac = fmin(ac, 1.0 / (1.0 + sqrt(dc2)) * 2.0);
            int okc = 0;
            for (int tries = 0; tries < 8; tries++) {
                for (int i = 0; i < m; i++) yt[i] = y[i] + ac * dy[i];
                double ldt;
                if (ds_test(P, S, yt, 0.0, &ldt) && -sk * ddot_n(m, P->b, yt) - ldt < f0c) { okc = 1; break; }
                ac *= 0.5;
            }
            if (!okc) { break; }
            memcpy(y, yt, sizeof(double) * m);
            if (!ds_factor_all(P, S, y, 0.0, &logdet)) { status = 4; break; }
            ncorr++;
            if (par->verbose > 1) printf("       [corrector %d: step %.3f, decrement %.2e]\n", cst, ac, sqrt(dc2));
        }
        if (status == 4) break;
        /* no bound yet: keep mu and recentre (the next Newton decrement below 1 gives a
         * certificate at about this mu); reducing mu blindly kept the iterates off the
         * central path for good (control, hinf). mu_red > 0 restores the old heuristic. */
        if (!have_ub && par->dual_murule != 1) {
            if (par->dual_mured > 0) mu *= (a > 0.9 ? 0.3 : a > 0.5 ? 0.6 : 0.9);
            else if (dl2 < 0.25) mu *= 0.5;
        }
    }
#undef DS_ASSEMBLE

    /* ---- report (original problem): y, and the best certificate X */
    if (best_err < 1e300 && (status == 0 || status == 5 || status == 3)) {
        memcpy(y, ybest, sizeof(double) * m);
        gap = best_gap; last_res = best_res;
        for (int k = 0; k < nb; k++) memcpy(S[k].X, S[k].Xb, bsz(&P->blk[k]) * sizeof(double));
        if (status == 3 && best_err <= par->red_acc) status = 5;
        if (best_err <= 0.5 * par->tol) status = 0;
    }
    double bto = ddot_n(m, P->b, y);
    R->iters = it;
    R->pobj = sc * (bto + gap);
    R->dobj = sc * bto;
    R->relgap = sc * gap / (1 + fabs(sc * bto + P->obj_off));
    R->relcomp = R->relgap;
    R->pinf = last_res * P->bs / (1 + P->normb2);
    R->dinf = 0;
    R->err[1] = R->pinf; R->err[5] = R->relgap; R->err[6] = R->relgap;
    if (yout) for (int i = 0; i < m; i++) yout[i] = P->cs * P->d[i] * y[i];
    if (Xout && (status == 0 || status == 5))
        for (int k = 0; k < nb; k++) {
            const size_t len = bsz(&P->blk[k]);
            for (size_t q = 0; q < len; q++) Xout[k][q] = P->bs * S[k].X[q];
        }
    R->status = status;
    R->t_total = wtime() - t0;
    if (par->verbose)
        printf("   (dual method: %d iterations, %d correctors, %d certificate searches, %d skipped, %d verified; Schur factor %.2fs, Z^-1 %.2fs, verification %.2fs)\n",
               it, ncorr, ncert, nskip, nverify, Sc.t_fact, t_inv, t_verify);

    if (g_blas_T > 1 && blas_T_prev == 0) BL(openblas_set_num_threads)(g_blas_T);
    g_blas_T = blas_T_prev;
    g_par_blk = par_blk_prev;
    for (int k = 0; k < nb; k++) {
        BS *s = &S[k];
        double *bufs[] = { s->X, s->Z, s->LX, s->LZ, s->Zi, s->dX, s->dZ, s->F, s->G, s->Q, s->W,
                           s->Fr, s->T1, s->Zc, s->lq, s->v1, s->v2, s->dtmp, s->Gd, s->Mdd,
                           s->lrV, s->lrPQ, s->lrPd, s->lrQd, s->lrQ, s->Xb };
        for (size_t q = 0; q < sizeof(bufs) / sizeof(bufs[0]); q++) free(bufs[q]);
        free(s->lr_owner);
        if (s->vbs) { free(s->vbs->X); free(s->vbs->Zi); free(s->vbs->dtmp); free(s->vbs->Gd); free(s->vbs->Mdd); free(s->vbs); }
        free(s->dW);
        schol_free(s->dzs); schol_free(s->dzt);
        schol_free(s->zf); free(s->zcp); free(s->zri); free(s->zt); free(s->ucp); free(s->uri); free(s->zdp); free(s->zpos); free(s->zamap); free(s->zfq); free(s->zcq); free(s->zr0q);
    }
    cls_release();
    free(S);
    schur_pos_free(&Sc);
    free(Sc.M); free(Sc.Md); free(Sc.Mf); free(Sc.ya); free(Sc.e3m); free(Sc.Masm); envelope_free(Sc.E); schol_free(Sc.SC); free(Sc.e1); free(Sc.e2); free(Sc.e3);
    free(Sc.D); free(Sc.r); free(Sc.t); free(Sc.p); free(Sc.q); free(Sc.xb); free(Sc.fw);
    free(y); free(ybest); free(yt); free(a0); free(d1); free(d2); free(dy); free(dycert); free(u); free(w); free(zero); free(ax); free(rq);
    return status;
}
