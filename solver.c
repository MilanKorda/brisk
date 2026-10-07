#include "brisk.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
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
    p->hsd_sig = 3;          /* candidates in the sigma search (0 = Mehrotra only) */
    p->hsd_beta = 1e-3;
    p->hsd_batch = 1;        /* embedding: paired Schur solves as one multi-rhs PCG */
    p->hsd_cgain = 0;        /* embedding: a corrector must gain cgain x its share of an iteration in step length */
    p->hsd_cfrac = 1.0;      /* embedding: corrector time budget as a fraction of assembly + factorization */
    p->hsd_soltol = 0;       /* embedding: relative accuracy of the main Schur solves (0 = full) */
    p->hsd_reghint = 1;      /* embedding: reuse the Schur regularization of the previous iteration */
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
    p->dd_maxn2 = 2.0e5;
    p->dd_budget = 1.0e9;
    p->dd_time = 60.0;
    p->polish = 1;
    p->polish_x = 1;
    p->polish_r = 1;         /* the restoration of the primal feasibility in the square-root metric before the other corrections */
    p->polish_mem = 512.0 * 1024 * 1024;
    p->reg_ill = 1e-10;
    p->sigma_rule = -1;   /* auto: cubic for pure LP, adaptive otherwise */
    p->timelimit = 0;
    p->bound_side = 0; p->bound_tol = 1e-10; p->bound_margin = 1e-13; p->bound_track_tol = -1; p->btrack = NULL;
    p->t_start = 0;
    p->retry_acc = 1e-7;     /* results within 1e-7 are returned as they are (no second solve) */
    p->dd_factor = 1.0;      /* endgame may cost at most as much as the solve itself ... */
    p->dd_minwork = 2e9;     /* ... or about a second, whichever is more */
    p->dd_done = 0;
    p->work_done = 0;
    p->dd_time = 0;          /* no wall-clock cap by default: the work budget is deterministic */
    p->acc_level = 1; p->red_acc = 1e-6; p->retry_hsd1 = 1e-6; p->slow_gate = 1e-5;
    p->tracebound = -1;
    p->dualize = -1;
    p->symfile = "auto"; p->symtime = 20.0; p->symnodes = 500; p->symbd = 1; p->symmin = 1.5; p->symsign = 1; p->symsigned = 1; p->returnx = -1; p->fom_race = 0.2; p->fom_race_budget = 0; p->hp_kind = 0; p->hp_digits = 0; p->hp_tol = 0; p->hp_ext = 2; p->symalg = -1; p->symalgmax = 1000;
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
    double *lq, *v1, *v2;
    double *Ct, *Dxc;    /* centrality correctors: trial product/eigvecs, correction */
    double *ev;          /* eigenvalues */
    double *dtmp;        /* nd */
    Pat upat;            /* union pattern of the constraint matrices */
    struct BS *vbs;      /* dictionary route: state of the virtual q x q block (X = P, Zi = Q) */
    double *dW;          /* n x dnv workspace (X * V) */
    double *Xb, *Zb;     /* best iterate (for polishing), if memory allows */
    double *Xr, *Zr;     /* the iterate the primal feasibility is restored from (RCand), if memory allows */
    SChol *dzs, *dzt;    /* dual method: sparse Cholesky of Z (iterate, tests), or NULL */
    /* NT direction: W = G G' is stored in Zi; lam = diag of the scaled point */
    double *ntG;         /* set only in the NT "view" (NULL in the block's own state) */
    double *ntGbuf, *ntVt, *ntT, *lam;
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
} BS;

static inline size_t bsz(const Block *B) { return B->type == BLK_SDP ? (size_t)B->n * B->n : (size_t)B->n; }

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
        if (P->blk[k].type == BLK_LP) { for (int i = 0; i < P->blk[k].n; i++) X[k][i] = f * Xk[i]; }
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
static void btrack_offer(const Params *par, const Problem *P, struct BS *S, const double *y, double tau,
                         double mu, const Result *R, int it) {
    BoundTrack *T = par->btrack;
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

/* 4.23: the parallel region only for large n. An "omp parallel if (false)" still enters the
 * runtime (0.22 us per call against 0.006 us for the loop at n = 3; TSSOS case6468 calls
 * these ~10^5 times per iteration on its 10,421 blocks) */
static void symmetrize(int n, double *A) {
    if (n > 1500) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < j; i++) {
                double s = 0.5 * (A[i + j * n] + A[j + i * n]);
                A[i + j * n] = s; A[j + i * n] = s;
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
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < j; i++) A[i + j * n] = A[j + i * n];
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
static double ddot_n(size_t len, const double *a, const double *b) {
    double s = 0;
    for (size_t i = 0; i < len; i++) s += a[i] * b[i];
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
static void blk_Aop_serial(const Block *B, BS *s, const double *D, double *out, double scale, const int *cpos) {
    int n = B->n;
    if (B->type == BLK_LP) {
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
    if (B->nd) {
        int n2 = n * n, nd = B->nd;
        BL(dgemv_)("T", &n2, &nd, &scale, B->Ad, &n2, D, &IONE, &DZERO, s->dtmp, &IONE);
        for (int a = 0; a < nd; a++) AOUT(B->dlist[a], s->dtmp[a]);
    }
}

/* F = sum_i y_i A_i (dense full / vector) */
static void blk_ATy(const Block *B, BS *s, const double *y, double *F) {
    int n = B->n;
    if (B->type == BLK_LP) {
        memset(F, 0, sizeof(double) * n);
        for (int t = 0; t < B->ncon; t++) {
            const SpSym *S = &B->A[t];
            double yi = y[B->con[t]];
            for (int q = 0; q < S->nnz; q++) F[S->row[q]] += yi * S->val[q];
        }
        return;
    }
    if (B->nd) {
        int n2 = n * n, nd = B->nd;
        for (int a = 0; a < nd; a++) s->dtmp[a] = y[B->con[B->dlist[a]]];
        BL(dgemv_)("N", &n2, &nd, &DONE, B->Ad, &n2, s->dtmp, &IONE, &DZERO, F, &IONE);
    } else memset(F, 0, sizeof(double) * (size_t)n * n);
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
    if (B->type == BLK_LP)
        for (int q = 0; q < S->nnz; q++) D[S->row[q]] += a * S->val[q];
    else
        for (int q = 0; q < S->ef; q++) D[S->fr[q] + (size_t)S->fc[q] * n] += a * S->fv[q];
}

static double sp_inner(const Block *B, const SpSym *S, const double *D) {
    int n = B->n;
    double s = 0;
    if (B->type == BLK_LP)
        for (int q = 0; q < S->nnz; q++) s += S->val[q] * D[S->row[q]];
    else
        for (int q = 0; q < S->ef; q++) s += S->fv[q] * D[S->fr[q] + (size_t)S->fc[q] * n];
    return s;
}

static long cnt_prod3 = 0, cnt_zfr = 0, cnt_refine_try = 0;
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

/* sum_i (D_i + theta R0_i)^2 */
static double norm2_plus_R0(const BS *s, size_t len, const double *D, double theta) {
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
    if (s->type == BLK_LP) {
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

/* out = Zi * F * R, where F is nonzero only on pattern pt.
 * Route 0: F has few nonzero rows -> two reduced dgemm (2 r n^2).
 * Route 1: F sparse -> Zi*F by column axpys (nnz*n), then one dsymm.
 * Route 2: dense -> two dsymm (2 n^3).                                   */
static void prod_pat(const Pat *pt, int n, BS *s, const double *F, const double *Rm, double *out) {
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
    unsigned char *mark = calloc(len, 1);
    s->r0n = 0;
    for (size_t i = 0; i < len; i++) {
        if (fabs(s->R0[i]) <= thr) s->R0[i] = 0.0;
        else { mark[i] = 1; s->r0n++; }
    }
    free(s->r0r); free(s->r0c); free(s->r0v);
    s->r0r = malloc(sizeof(int) * (s->r0n + 1));
    s->r0c = malloc(sizeof(int) * (s->r0n + 1));
    s->r0v = malloc(sizeof(double) * (s->r0n + 1));
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
    s->cpat.fr = malloc(sizeof(int) * (nf + 1));
    s->cpat.fc = malloc(sizeof(int) * (nf + 1));
    char *rm = calloc(n, 1);
    c = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (mark[i + (size_t)j * n]) { s->cpat.fr[c] = i; s->cpat.fc[c] = j; c++; rm[i] = 1; }
    s->cpat.nr = 0;
    for (int i = 0; i < n; i++) s->cpat.nr += rm[i];
    s->cpat.rows = malloc(sizeof(int) * (s->cpat.nr + 1));
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
static size_t **g_bpos = NULL;
static size_t **g_bsoff = NULL;     /* per block: lower-triangle offsets sorted, and their buffer index */
static int **g_bsidx = NULL, *g_bsn = NULL;
static double *g_lbuf_all = NULL;
static const Block *g_blk0 = NULL;
static int g_nbk = 0;
static __thread int g_nl = 0;            /* 4.24: per-thread block state (parallel assembly) */
static __thread int *g_loc = NULL;
static __thread const size_t *g_pos = NULL;
static double *g_pm = NULL;
/* local accumulation buffer of the block being assembled (lower triangle, hi * nl + lo),
 * flushed into the factor storage in increasing offset order: the scattered writes into
 * the 5 MB panel storage were 2/3 of the assembly time on the TSSOS models */
static __thread double *g_lbuf = NULL;
static inline void sc_add(size_t r, size_t c, double v) {
    const int lr = g_loc[r], lc = g_loc[c];
    if (lr >= 0 && lc >= 0) {
        if (g_lbuf) { const int hi = lr > lc ? lr : lc, lo = lr > lc ? lc : lr; g_lbuf[(size_t)lo * g_nl + hi] += v; }
        else g_pm[g_pos[(size_t)lr * g_nl + lc]] += v;
    }
    else schol_add(g_schol, r, c, v);
}

static inline void env_add(const Envelope *E, size_t r, size_t c, double v) {
    int pi = E->iperm[r], pj = E->iperm[c];
    if (pi < pj) { int t = pi; pi = pj; pj = t; }
    E->val[E->rptr[pi] + (pj - E->fst[pi])] += v;
}
/* ordered write (r >= c) and unordered write */
#define MADDL(r, c, v) do { size_t r_ = (r), c_ = (c); double v_ = (v); \
    if (g_pos) sc_add(r_, c_, v_); \
    else if (g_schol) schol_add(g_schol, r_, c_, v_); \
    else if (g_env) env_add(g_env, r_, c_, v_); else M[r_ + c_ * (size_t)m] += v_; } while (0)
#define MADD(r, c, v) do { size_t r_ = (r), c_ = (c); double v_ = (v); \
    if (g_pos) sc_add(r_, c_, v_); \
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
    #pragma omp parallel if (nc > 32 && nthreads() > 1 && !in_par())
    {
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
static void schur_sdp_id(const Block *B, BS *s, int m, double *M, int identity) {
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
    if (B->live) {
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
    #pragma omp parallel if (par1)
    {
        double *Xp = NULL, *Tm = NULL, *G = NULL, *tmp = NULL, *acc = NULL;
        int *rmap = NULL, *cptr = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int a = 0; a < ns; a++) {
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
                    rmap = malloc(sizeof(int) * n);
                }
                const double ts0 = g_split_on ? wtime() : 0;
                int r = At->nr;
                if (B->live && B->route[t] == 1) {
                    /* the product on the staircase of the entries that the constraints b >= a read:
                     * live indices 0..u-1 (permuted), column panels, the rows above the panel's
                     * highest entry only; the panel is read while it is in the cache */
                    if (!acc) { acc = calloc((size_t)ns + 1, sizeof(double)); cptr = malloc(sizeof(int) * (3 * (size_t)n + 3)); }
                    int *const cq = cptr + n + 1, *const hq = cptr + 2 * (n + 1);
                    const int u = B->lv_u[a];
                    const int *rank = B->lv_rank, *cp = B->lv_cp, *ca = B->lv_ca, *cr = B->lv_cr, *chh = B->lv_ch;
                    const double *cv = B->lv_cv;
                    /* Xp (r x u): column p holds X[p, rows] - by the symmetry of X a gather from one column of XP */
                    /* 5.7: Xp (u x r, leading dimension n): column k is the column rank[rows[k]] of XP
                     * (X is symmetric), copied whole instead of gathered entry by entry */
                    for (int k = 0; k < r; k++) { rmap[At->rows[k]] = k; memcpy(Xp + (size_t)k * n, XP + (size_t)rank[At->rows[k]] * n, sizeof(double) * (size_t)u); }
                    memset(Tm, 0, sizeof(double) * (size_t)r * u);
                    for (int e = 0; e < At->ef; e++) {
                        const double v = At->fv[e];
                        const double *zq = ZP + (size_t)rank[At->fc[e]] * n;
                        double *tk = Tm + (size_t)rmap[At->fr[e]] * u;
                        for (int i = 0; i < u; i++) tk[i] += v * zq[i];
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
                        pdgemm("N", "T", &h, &ww, &r, &DONE, Xp, &n, Tm + q0, &u, &DZERO, G, &h);
                        const double tp1 = g_split_on ? wtime() : 0;
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
        free(Xp); free(Tm); free(G); free(tmp);
        free(rmap); free(acc); free(cptr);
    }
    if (par1) blas_serial_end();
    free(XP); free(ZP);
    free(scon);
    if (nd == 0) return;

    /* phase 2: dense rows. G_a = X A_a Zi (BLAS-3); dense-dense via one dgemm per chunk */
    const int ch = s->chunk;
    for (int c0 = 0; c0 < nd; c0 += ch) {
        const int c1 = (c0 + ch < nd) ? c0 + ch : nd;
        const int par2 = nthreads() > 1 && c1 - c0 > 1 && !in_par();
        if (par2) blas_serial_begin();
        #pragma omp parallel if (par2)
        {
            double *W = amalloc(sizeof(double) * n2);
            #pragma omp for schedule(dynamic, 1)
            for (int a = c0; a < c1; a++) {
                const double *D = B->Ad + (size_t)a * n2;
                double *G = s->Gd + (size_t)(a - c0) * n2;
                xsymm("R", "L", &n, &n, &DONE, Zi, &n, D, &n, &DZERO, W, &n);
                xsymm("L", "L", &n, &n, &DONE, X, &n, W, &n, &DZERO, G, &n);
                const int ci = B->con[B->dlist[a]];
                for (int b = 0; b < ns; b++) {
                    double sum = 0;
                    if (ffp) for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * G[ffp[k]];
                    else for (int k = foff[b]; k < foff[b + 1]; k++) sum += ffv[k] * G[ffr[k] + (size_t)ffc[k] * n];
                    MADD(ci, B->con[B->slist[b]], sum);
                }
            }
            free(W);
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
static void schur_lp(const Block *B, const BS *s, int m, double *M) {
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
        if (B->type == BLK_SDP) { ncl++; tot += B->ncon; }
        else for (int p = 0; p < B->n; p++) { ncl++; tot += B->lp_ptr[p + 1] - B->lp_ptr[p]; }
    }
    int *cptr = malloc(sizeof(int) * (ncl + 1));
    int *cidx = malloc(sizeof(int) * (tot ? tot : 1));
    int c = 0; long pos = 0;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type == BLK_SDP) {
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
    /* fill cap: a factor with more than ~half of the lower triangle cannot win; below
     * m = 8000 the analysis may look that far (moment-SOS image forms: 30% dense patterns
     * made of a few overlapping cliques factor in a tenth of the dense flops) */
    static double fillfrac = -1;
    if (fillfrac < 0) { const char *e = getenv("BRISK_FILLFRAC"); fillfrac = e ? atof(e) : 0.25; }   /* test: the fraction above m = 8000 */
    size_t cap = (size_t)((m <= 8000 ? 0.45 : fillfrac) * (double)m * m) + 1000;
    /* memory: L costs ~48 bytes per entry (factor, assembled copy, assembly hash); 2e7
     * entries is about 1 GB. L holds at least the lower half of the pattern, so a pattern
     * beyond the cap is rejected before the analysis (whose per-row lists would stay in
     * the heap: roa_vdp_inner_d12, m = 15327 with a 45% dense pattern, then exceeded the
     * address-space limit when the dense Schur matrix was allocated) */
    { extern double g_fillmax; if (!getenv("BRISK_FILLMAX")) g_fillmax = 6e7 * brisk_mem_scale(); if (cap > (size_t)g_fillmax) cap = (size_t)g_fillmax; }
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
    SChol *SC = schol_analyze_adj(m, deg, nbr, cap);
    schol_set_flopcap(0);
    free(deg); free(nbr); free(aptr); free(adj);
    if (!SC) {
        if (verbose > 1 || getenv("BRISK_ROUTEDBG")) printf("Schur pattern: %.3g entries in the lower triangle (%.0f %%): the analysis of a sparse factorization stopped at its fill or flop cap\n", pat_lower, 200.0 * pat_lower / ((double)m * m));
        return NULL;
    }
    double dense = par->c_blas * (double)m * m * m / 3.0;
    /* wide supernodes (the dense cliques of SOS/moment constraints) run at BLAS-3 speed:
     * charge them 3x the dense rate for the scatter and the panel overheads */
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
        if (B->type == BLK_SDP) { ncl++; tot += B->ncon; if (B->ncon > maxcl) maxcl = B->ncon; }
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
            if (B->type == BLK_SDP) {
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
    size_t **bpos;        /* per-block scatter offsets into SC (see g_bpos), or NULL */
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
} Schur;

long g_sc_cnt[8]; long g_pcgh[12];   /* debug (BRISK_SOLDBG): single trusted/ok/pcg, multi trusted/ok/pcg, bord, mv */
static void bord_mv(Schur *S, const double *x, double *y);
static double bord_solve(Schur *S, const double *rhs, double *x, double tol_rel);
static int g_looseskip = 0; static long g_multiskip = 0;   /* 4.23: loose solves on a factor already verified far below their tolerance skip the residual check */
/* 5.7: y = alpha M x + beta y for a symmetric M stored in one triangle, on the OpenMP threads: the BLAS
 * is single-threaded here and dsymv was the larger part of the solves with a dense Schur complement
 * (m = 3,739: 325 products, 1.2 of 2.4 s). The index range is cut into T parts of equal area; part b
 * owns its diagonal block and the rectangle that joins it to the earlier indices, and adds both of
 * the rectangle's products into a vector of its own. The result does not depend on the thread
 * count only up to rounding (the parts are summed in order); T comes from the thread count. */
static void par_symv(char uplo, int m, double alpha, const double *M, const double *x, double beta, double *y) {
    const int T0 = nthreads();
    const int T = (T0 > 1 && m >= 1024 && !in_par() && !ENV_ON("BRISK_SYMVSER")) ? (T0 > 16 ? 16 : T0) : 1;
    if (T == 1) { BL(dsymv_)(uplo == 'L' ? "L" : "U", &m, &alpha, M, &m, x, &IONE, &beta, y, &IONE); return; }
    int cut[18];
    for (int b = 0; b <= T; b++) cut[b] = (int)(m * sqrt((double)b / T) + 0.5);
    cut[0] = 0; cut[T] = m;
    double *W = amalloc(sizeof(double) * (size_t)m * T);
    blas_serial_begin();
    #pragma omp parallel for schedule(static, 1) num_threads(T)
    for (int b = 0; b < T; b++) {
        const int r0 = cut[b], r1 = cut[b + 1], nr = r1 - r0;
        double *w = W + (size_t)b * m;
        memset(w, 0, sizeof(double) * (size_t)m);
        if (nr <= 0) continue;
        const double one = 1.0;
        /* diagonal block */
        BL(dsymv_)(uplo == 'L' ? "L" : "U", &nr, &one, M + r0 + (size_t)r0 * m, &m, x + r0, &IONE, &one, w + r0, &IONE);
        if (r0 > 0) {
            if (uplo == 'L') {
                /* R = M[r0:r1, 0:r0] */
                const double *Rm = M + r0;
                BL(dgemv_)("N", &nr, &r0, &one, Rm, &m, x, &IONE, &one, w + r0, &IONE);
                BL(dgemv_)("T", &nr, &r0, &one, Rm, &m, x + r0, &IONE, &one, w, &IONE);
            } else {
                /* R = M[0:r0, r0:r1] */
                const double *Rm = M + (size_t)r0 * m;
                BL(dgemv_)("N", &r0, &nr, &one, Rm, &m, x + r0, &IONE, &one, w, &IONE);
                BL(dgemv_)("T", &r0, &nr, &one, Rm, &m, x, &IONE, &one, w + r0, &IONE);
            }
        }
    }
    blas_serial_end();
    for (int i = 0; i < m; i++) {
        double t = 0;
        for (int b = 0; b < T; b++) t += W[i + (size_t)b * m];
        y[i] = alpha * t + (beta == 0 ? 0.0 : beta * y[i]);
    }
    free(W);
}
static double g_solprof_p[4];
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
    else if (!S->mT_ok || S->low_M) par_symv('L', m, alpha, S->M, x, beta, y);
    else {
        /* M from the strict upper copy; the array diagonal is the factor's, so correct it */
        par_symv('U', m, alpha, S->M, x, beta, y);
        for (int i = 0; i < m; i++) y[i] += alpha * (S->Md[i] - S->M[i + (size_t)i * m]) * x[i];
    }
}
static void schur_zero(Schur *S) {
    if (S->SC && S->Masm) memset(S->Masm, 0, sizeof(double) * (size_t)S->m * S->m);
    else if (S->SC) schol_zero(S->SC);
    else if (S->E) memset(S->E->val, 0, sizeof(double) * S->E->nnz);
    else { memset(S->M, 0, sizeof(double) * (size_t)S->m * S->m); S->mT_ok = 0; S->low_M = 1; }
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
        for (int k = 0; k < nb; k++) {
            BS vtmp;
            if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], &S[k], m, Ma);
            else schur_sdp(&P->blk[k], nt_view(&S[k], &vtmp, Zv[k] != NULL), m, Ma);
        }
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
/* scatter offsets of every SDP block's constraint pairs (memory capped) */
static void schur_pos_setup(Schur *S, const Problem *P) {
    S->bpos = NULL; S->loc = NULL; S->blk0 = P->blk; S->nbk = P->nblk;
    S->bsoff = NULL; S->bsidx = NULL; S->bsn = NULL; S->lbuf = NULL;
    if (!S->SC || S->Masm) return;
    double tot = 0;
    for (int k = 0; k < P->nblk; k++) if (P->blk[k].type == BLK_SDP) tot += (double)P->blk[k].ncon * P->blk[k].ncon;
    if (tot * sizeof(size_t) > 1.5e9 * brisk_mem_scale()) return;
    S->bpos = calloc(P->nblk + 1, sizeof(size_t *));
    S->loc = malloc(sizeof(int) * (S->m + 1));
    for (int i = 0; i < S->m; i++) S->loc[i] = -1;
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        if (B->type != BLK_SDP || B->ncon == 0) continue;
        const int nl = B->ncon;
        size_t *pos = malloc(sizeof(size_t) * (size_t)nl * nl);
        if (schol_clique_offsets(S->SC, nl, B->con, pos, NULL, NULL) == 0) S->bpos[k] = pos; else free(pos);
    }
    /* sorted flush lists (columns and rows in factor order: offsets come out increasing) */
    S->bsoff = calloc(P->nblk + 1, sizeof(size_t *)); S->bsidx = calloc(P->nblk + 1, sizeof(int *)); S->bsn = calloc(P->nblk + 1, sizeof(int));
    int nlmax = 0;
    for (int k = 0; k < P->nblk; k++) {
        if (!S->bpos[k]) continue;
        const int nl = P->blk[k].ncon;
        if (nl > 1500) continue;                 /* buffer nl^2: keep it small */
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
    if (S->bpos) for (int k = 0; k < S->nbk; k++) free(S->bpos[k]);
    free(S->bpos); free(S->loc);
    S->bpos = NULL; S->loc = NULL;
}

static int schur_factor_(Schur *S);
static int g_blas_T = 0;          /* BLAS threads while block-parallel loops run serial BLAS */
static int schur_factor(Schur *S) {
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
    #pragma omp parallel for schedule(dynamic, 1) if (m > 2000 && !omp_in_parallel())
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
    #pragma omp parallel for schedule(static) if (m > 2000 && !omp_in_parallel())
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
    schur_dense_keep(S);
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
        #pragma omp parallel for schedule(static) if (m > 2000)
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
        info = brisk_spotrf(m, F, m);
        if (info == 0) {
            double pmin = 1.0;
            for (int i = 0; i < m; i++) { double p = F[i + (size_t)i * m]; if (p * p < pmin) pmin = p * p; }
            if (pmin >= 1e-6) { S->is_float = 1; S->reg = 0; S->nfloat++; return 0; }
        }
        S->want_double = 1;
    }
    S->is_float = 0;
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
         * in double (OpenBLAS, m = 1,000 ... 8,000) */
        if (ENV_ON("BRISK_POTRS")) BL(spotrs_)("L", &m, &IONE, S->Mf, &m, S->fw, &m, &info); else {
        BL(strsv_)("L", "N", "N", &m, S->Mf, &m, S->fw, &IONE);
        BL(strsv_)("L", "T", "N", &m, S->Mf, &m, S->fw, &IONE); }
        for (int i = 0; i < m; i++) z[i] = S->fw[i] * S->D[i];
        return;
    }
    for (int i = 0; i < m; i++) z[i] = r[i] * S->D[i];
    if (ENV_ON("BRISK_POTRS")) BL(dpotrs_)("L", &m, &IONE, S->Mc, &m, z, &m, &info); else {
    BL(dtrsv_)("L", "N", "N", &m, S->Mc, &m, z, &IONE);
    BL(dtrsv_)("L", "T", "N", &m, S->Mc, &m, z, &IONE); }
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
    if (S->is_float) {
        float *fw = malloc(sizeof(float) * (size_t)m * k);
        for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) fw[i + (size_t)c * m] = (float)(R[i + (size_t)c * m] * S->D[i]);
        if (ENV_ON("BRISK_POTRS")) BL(spotrs_)("L", &m, &k, S->Mf, &m, fw, &m, &info); else
        for (int c = 0; c < k; c++) {                 /* (per column: see schur_prec) */
            BL(strsv_)("L", "N", "N", &m, S->Mf, &m, fw + (size_t)c * m, &IONE);
            BL(strsv_)("L", "T", "N", &m, S->Mf, &m, fw + (size_t)c * m, &IONE);
        }
        for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) Zo[i + (size_t)c * m] = fw[i + (size_t)c * m] * S->D[i];
        free(fw);
        return;
    }
    for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) Zo[i + (size_t)c * m] = R[i + (size_t)c * m] * S->D[i];
    if (k <= 2 && !ENV_ON("BRISK_POTRS")) for (int c = 0; c < k; c++) {
        BL(dtrsv_)("L", "N", "N", &m, S->Mc, &m, Zo + (size_t)c * m, &IONE);
        BL(dtrsv_)("L", "T", "N", &m, S->Mc, &m, Zo + (size_t)c * m, &IONE);
    } else BL(dpotrs_)("L", &m, &k, S->Mc, &m, Zo, &m, &info);
    for (int c = 0; c < k; c++) for (int i = 0; i < m; i++) Zo[i + (size_t)c * m] *= S->D[i];
}
static void schur_mv_multi(Schur *S, int k, double alpha, const double *X, double beta, double *Y) {
    const int m = S->m;
    /* few columns: dsymm packs all of M on every call (a copy of m^2 values, 14% of the time
     * on roa_dint_d6_I); dsymv per column only reads it */
    if (S->SC || S->E || k <= 4) { for (int c = 0; c < k; c++) schur_mv(S, alpha, X + (size_t)c * m, beta, Y + (size_t)c * m); return; }
    if (!S->mT_ok || S->low_M) { xsymm("L", "L", &m, &k, &alpha, S->M, &m, X, &m, &beta, Y, &m); return; }
    xsymm("L", "U", &m, &k, &alpha, S->M, &m, X, &m, &beta, Y, &m);    /* 4.26: M in the upper copy */
    for (int c = 0; c < k; c++)
        for (int i = 0; i < m; i++) Y[i + (size_t)c * m] += alpha * (S->Md[i] - S->M[i + (size_t)i * m]) * X[i + (size_t)c * m];
}
static void schur_solve_multi(Schur *S, int k, const double *const *rhs, double *const *x, const double *tol_rel) {
    if (S->bF) { for (int c = 0; c < k; c++) schur_solve(S, rhs[c], x[c], tol_rel[c]); return; }
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
 * explicit formation plus cyclic Jacobi, no BLAS call overhead.             */
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
static double sdp_maxstep_lam(int n, const double *L, const double *D, BS *s, int Kmax, double *lam_out);
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
    int K = n < Kmax ? n : Kmax;
    if (K > LANCZOS_K) K = LANCZOS_K;
    double alpha[LANCZOS_K], beta[LANCZOS_K];
    double *Q = s->lq, *w = s->v2;
    double nrm = 0;
    for (int i = 0; i < n; i++) { Q[i] = 1.0 + 0.5 * sin(1.7 * i + 0.3); nrm += Q[i] * Q[i]; }
    nrm = 1.0 / sqrt(nrm);
    for (int i = 0; i < n; i++) Q[i] *= nrm;
    int kk = 0;
    double scale = 0, theta = 0, res = 0;
    for (int j = 0; j < K; j++) {
        double *qj = Q + (size_t)j * n;
        memcpy(s->v1, qj, sizeof(double) * n);
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
        if (kk >= 6 && (kk % 3 == 0)) {
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
            if (B->type != BLK_LP) continue;
            BS *s = &S[k];
            double ak = lp_maxstep(B->n, primal ? s->X : s->Z, primal ? s->dX : s->dZ, s->fm);
            if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
            if (ak < a) {
                a = ak; kmin = k;
                if (who) { const double *x = primal ? s->X : s->Z, *dx = primal ? s->dX : s->dZ; for (int i = 0; i < B->n; i++) if (dx[i] < 0 && -x[i] / dx[i] <= ak * (1 + 1e-12)) { imin = i; break; } }
            }
        }
        if (who) {
            double a0 = a;
            for (int k = 0; k < P->nblk; k++) {
                const Block *B = &P->blk[k];
                if (B->type == BLK_LP) continue;
                BS *s = &S[k];
                const double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
                if (a < 1e29 && B->n <= SMALL_BLOCK && small_pd_at(B->n, V, dV, a)) continue;
                double ak = sdp_maxstep(B->n, primal ? s->LX : s->LZ, dV, s, K);
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
            if (B->type == BLK_LP) continue;
            BS *s = &S[k];
            const double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
            if (a < 1e29 && B->n <= SMALL_BLOCK && small_pd_at(B->n, V, dV, a)) continue;
            double ak = sdp_maxstep(B->n, primal ? s->LX : s->LZ, dV, s, K);
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
        if (B->type != BLK_LP) continue;
        BS *s = &S[k];
        double ak = lp_maxstep(B->n, primal ? s->X : s->Z, primal ? s->dX : s->dZ, s->fm);
        if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
        if (ak < a) a = ak;
    }
    const int nb = P->nblk, nch = (nb + MS_CH - 1) / MS_CH;
    const double a0 = a;
    #pragma omp parallel for schedule(dynamic, 1) reduction(min: a)
    for (int c = 0; c < nch; c++) {
        double ac = a0;
        const int k1 = (c + 1) * MS_CH < nb ? (c + 1) * MS_CH : nb;
        for (int k = c * MS_CH; k < k1; k++) {
            const Block *B = &P->blk[k];
            if (B->type == BLK_LP) continue;
            BS *s = &S[k];
            const double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
            if (ac < 1e29 && B->n <= SMALL_BLOCK && small_pd_at(B->n, V, dV, ac)) continue;
            double ak = sdp_maxstep(B->n, primal ? s->LX : s->LZ, dV, s, K);
            if (!isfinite(ak) && !(ak > 0)) ak = 0.0;
            if (ak < ac) ac = ak;
        }
        if (ac < a) a = ac;
    }
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
        PAR_BLOCKS
        for (int k = 0; k < P->nblk; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n, okk = 1;
            size_t len = bsz(B);
            double *V = primal ? s->X : s->Z, *dV = primal ? s->dX : s->dZ;
            double *L = primal ? s->LX : s->LZ;
            double inc = a - s->stepped[primal];
            for (size_t i = 0; i < len; i++) V[i] += inc * dV[i];
            s->stepped[primal] = a;          /* all blocks stay at the same step */
            if (B->type == BLK_LP) {
                for (int i = 0; i < n; i++) if (!(V[i] > 0) && !(s->fm && s->fm[i])) { okk = 0; break; }
            } else {
                memcpy(L, V, sizeof(double) * len);
                if (chol_lower(n, L) != 0) okk = 0;
            }
            if (!okk) {
                #pragma omp atomic write
                ok = 0;
            }
        }
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
        } else {
            for (int i = 0; i < n; i++) { s->X[i + (size_t)i * n] = xi; s->Z[i + (size_t)i * n] = eta; }
            memcpy(s->LX, s->X, sizeof(double) * len); chol_lower(n, s->LX);
            memcpy(s->LZ, s->Z, sizeof(double) * len); chol_lower(n, s->LZ);
        }
    }
}

/* ------------------------------------------------------------------------ */
/* out = A(A'(v)) without forming A A'. Sparse SDP blocks touch only their
 * pattern (F is kept zero outside it between calls).                         */
static void gram_mv(const Problem *P, BS *S, const double *v, double *out) {
    memset(out, 0, sizeof(double) * P->m);
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        if (B->type == BLK_LP || B->nd > 0) {
            blk_ATy(B, s, v, s->F);
            blk_Aop(B, s, s->F, out, 1.0);
            memset(s->F, 0, sizeof(double) * bsz(B));
            continue;
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
static double gram_cg_polish(const Problem *P, BS *S, const double *rhs, double *x, double tol,
                             double abs_tol, double *r, double *p, double *q) {
    const int m = P->m, maxit = gram_maxit();
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

/* All six DIMACS measures of a candidate (X, y, Z) in scaled units (X, Z given as
 * per-block arrays). Unlike the iterates, a polished candidate need not be
 * positive definite, so errors 2 and 4 are computed (Lanczos lambda_min,
 * conservative).  Scratch: s->W, s->F.  Returns the overall score.            */
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
        } else {
            /* The Lanczos bound is conservative and can turn slightly negative for an
             * ill-conditioned but positive definite matrix, so a successful Cholesky
             * overrides it (it proves definiteness in floating point).                */
            double lx, lz;
            int n = B->n, info;
            sdp_maxstep_lam(n, NULL, Xs[k], s, LANCZOS_K, &lx);
            sdp_maxstep_lam(n, NULL, Zs[k], s, LANCZOS_K, &lz);
            if (lx < 0) {
                memcpy(s->W, Xs[k], sizeof(double) * len);
                BL(dpotrf_)("L", &n, s->W, &n, &info);
                if (info == 0) lx = 0.0;
            }
            if (lz < 0) {
                memcpy(s->W, Zs[k], sizeof(double) * len);
                BL(dpotrf_)("L", &n, s->W, &n, &info);
                if (info == 0) lz = 0.0;
            }
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
static int dd_small_problem(const Problem *P, const Params *par) {
    double n2 = 0;
    for (int k = 0; k < P->nblk; k++) n2 += (double)bsz(&P->blk[k]);
    return par->dd_end && !par->dd_done && P->m <= par->dd_maxm && n2 <= par->dd_maxn2;
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

static int g_epol_stop = 0;   /* hsd_run stopped on a passing polish probe */
static int g_epol_fb = 0;     /* ... or returned a fallback probe candidate (tol below 1e-8) */
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
    double dmin = INFINITY;
    for (int i = 0; i < m; i++) dmin = fmin(dmin, P->d[i]);
    const double abs_tol = 1e-3 * tol * (1.0 + P->normb2) * dmin / P->bs;
    *rres = gram_cg_polish(P, S, Sc->t, w, 1e-15, abs_tol, Sc->r, Sc->p, Sc->q);
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
 * complement with Zi replaced by X, assembled and factored by the usual routes. It matters
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
    for (int pass = 0; pass < 2 && ok && !half; pass++) {
        memcpy(rp, P->b, sizeof(double) * m);
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], Xc[k], rp, -1.0);
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

static int nt_scaling(const Block *B, BS *s) {
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
    double *Mb = s->ntT;
    size_t len = (size_t)n * n;
    memset(Mb, 0, sizeof(double) * len);
    for (int j = 0; j < n; j++)
        for (int i = j; i < n; i++) Mb[i + (size_t)j * n] = s->LX[i + (size_t)j * n];
    BL(dtrmm_)("L", "L", "T", "N", &n, &n, &DONE, s->LZ, &n, Mb, &n);           /* L_Z' L_X */
    int lwork = -1, info, one = 1;
    double wq, dummy;
    int *iwork = malloc(sizeof(int) * 8 * (size_t)n);
    BL(dgesdd_)("O", &n, &n, Mb, &n, s->lam, &dummy, &one, s->ntVt, &n, &wq, &lwork, iwork, &info);
    lwork = (int)wq + 1;
    double *work = malloc(sizeof(double) * lwork);
    BL(dgesdd_)("O", &n, &n, Mb, &n, s->lam, &dummy, &one, s->ntVt, &n, work, &lwork, iwork, &info);
    free(work); free(iwork);
    if (info != 0) return 1;
    double *G = s->ntGbuf;
    for (int j = 0; j < n; j++)                                                   /* V = Vt' */
        for (int i = 0; i < n; i++) G[i + (size_t)j * n] = s->ntVt[j + (size_t)i * n];
    BL(dtrmm_)("L", "L", "N", "N", &n, &n, &DONE, s->LX, &n, G, &n);             /* L_X V */
    for (int j = 0; j < n; j++) {
        if (!(s->lam[j] > 0)) return 1;
        double f = 1.0 / sqrt(s->lam[j]);
        for (int i = 0; i < n; i++) G[i + (size_t)j * n] *= f;
    }
    BL(dsyrk_)("L", "N", &n, &n, &DONE, G, &n, &DZERO, s->Zi, &n);               /* W = G G' */
    lower_to_full(n, s->Zi);
    return 0;
}

/* out = G' D G  (D symmetric) */
static void nt_toZ(const Block *B, BS *s, const double *D, double *out) {
    const int n = B->n;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) out[i] = s->ntGbuf[i] * s->ntGbuf[i] * D[i]; return; }
    xsymm("L", "L", &n, &n, &DONE, D, &n, s->ntGbuf, &n, &DZERO, s->ntT, &n);
    pdgemm("T", "N", &n, &n, &n, &DONE, s->ntGbuf, &n, s->ntT, &n, &DZERO, out, &n);
    symmetrize(n, out);
}

/* out = G^{-1} D G^{-T} = diag(lam)^{1/2} V' L_X^{-1} D L_X^{-T} V diag(lam)^{1/2} */
static void nt_toX(const Block *B, BS *s, const double *D, double *out) {
    const int n = B->n;
    if (B->type == BLK_LP) { for (int i = 0; i < n; i++) out[i] = D[i] / (s->ntGbuf[i] * s->ntGbuf[i]); return; }
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
    return fmax(fmax(nan_inf(R->relgap), nan_inf(R->relcomp)), fmax(nan_inf(R->pinf), nan_inf(R->dinf)));
}

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
    free(F->K); free(F->ks); free(F->ipiv); free(F->wf);
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
static int free_factor_aug(const Problem *P, Schur *Sc, FreeVars *F) {
    const int m = P->m, nf = F->nf, N = m + nf;
    double t0 = wtime();
    if (!F->K) {
        F->K = amalloc(sizeof(double) * (size_t)N * N);
        F->ks = malloc(sizeof(double) * N);
        F->ipiv = malloc(sizeof(int) * N);
    }
    double *K = F->K, *ks = F->ks;
    for (int i = 0; i < m; i++) { double d = schur_dense_get(Sc, i, i); ks[i] = d > 0 ? 1.0 / sqrt(d) : 1.0; }
    memset(K, 0, sizeof(double) * (size_t)N * N);
    for (int j = 0; j < m; j++)
        for (int i = j; i < m; i++) K[i + (size_t)j * N] = schur_dense_get(Sc, i, j) * ks[i] * ks[j];
    for (int f = 0; f < nf; f++) {
        const Block *B = &P->blk[F->kb[f]];
        const int p = F->ip[f];
        double nrm = 0;
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++) { double v = B->lp_val[a] * ks[B->lp_con[a]]; nrm += v * v; }
        ks[m + f] = nrm > 0 ? 1.0 / sqrt(nrm) : 1.0;
        for (int a = B->lp_ptr[p]; a < B->lp_ptr[p + 1]; a++)       /* row m+f, lower part: K[m+f, con] */
            K[(m + f) + (size_t)B->lp_con[a] * N] += B->lp_val[a] * ks[B->lp_con[a]] * ks[m + f];
    }
    int info, lwork = -1;
    double wq;
    BL(dsytrf_)("L", &N, K, &N, F->ipiv, &wq, &lwork, &info);
    lwork = (int)wq + 1;
    double *work = malloc(sizeof(double) * lwork);
    /* singular saddle matrix (dependent free columns, empty rows): quasi-definite shift
     * +d on the y block, -d on the free block; the refinement works with the true one */
    double *K0 = NULL;
    /* start from the shift that worked last time (singularity is structural) */
    for (double d = F->reg; d <= 1e-6; d = d ? d * 100 : 1e-12) {
        if (d > 0) {
            if (!K0) {
                K0 = malloc(sizeof(double) * (size_t)N * N);
                if (!K0) break;
                memcpy(K0, K, sizeof(double) * (size_t)N * N);
            }
            memcpy(K, K0, sizeof(double) * (size_t)N * N);
            for (int i = 0; i < m; i++) K[i + (size_t)i * N] += d;
            for (int f = 0; f < nf; f++) K[(m + f) + (size_t)(m + f) * N] -= d;
        } else {
            K0 = malloc(sizeof(double) * (size_t)N * N);
            if (K0) memcpy(K0, K, sizeof(double) * (size_t)N * N);
        }
        BL(dsytrf_)("L", &N, K, &N, F->ipiv, work, &lwork, &info);
        F->reg = d;
        if (info == 0) break;
    }
    free(K0);
    free(work);
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
            BL(dsytrs_)("L", &N, &one, F->K, &N, F->ipiv, z, &N, &info);
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
static int eig_outside(int n, double *E, double lo, double hi, double *w, double *V) {
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
    BL(dsytrd_)("L", &n, E, &n, d, e, tau, work, &lwork, &info);
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
static int g_pfloor = 1;   /* HSD: stop at the primal residual floor (BRISK_PFLOOR=0 off) */   /* corrector: skip blocks inside the box (BRISK_NOCBOX) */
static int g_hpar = 0;   /* 4.24: parallel block loops in the embedding (BRISK_HPAR=0 off) */
#define HPAR_FOR _Pragma("omp parallel for schedule(dynamic, 16) if (g_hpar)")
static double *g_hred = NULL; static int g_hred_n = 0;
static double *hred(int nb, int w) {      /* per-block partials, w per block, zeroed */
    if (nb * w > g_hred_n) { free(g_hred); g_hred_n = nb * w; g_hred = malloc(sizeof(double) * g_hred_n); }
    if (nb * w > 0) memset(g_hred, 0, sizeof(double) * nb * w);
    return g_hred;
}
static double g_ct[12], g_asm[4];   /* 4.23 SOLDBG: corrector parts (pass1, pass2, Aop, solve, back, maxstep) */

/* research hook (4.35): BRISK_DUMPXZ=prefix writes the iterates X/tau, Z/tau of every
 * iteration (scaled problem coordinates) with the row scaling d, for offline experiments */
static void dump_xz(const Problem *P, const BS *S, int it, double tau) {
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
    fclose(f);
}

static int hsd_corrector(Problem *P, const Params *par, BS *S, Schur *Sc, double **Zv, const Pat *hp,
                         const char *pk, double tau, double kap, double *dt, double *dk, double *dyd,
                         const double *v, const double *hc, double denom, double *am, double mu_t,
                         double **bX, double **bZ, double *rhs, double *uc, double ndim1, Result *R,
                         FreeVars *FV, const double *vf, double accf) {
    const int m = P->m, nb = P->nblk;
    const double bmin = par->hsd_cbmin, bmax = par->hsd_cbmax;
    const double ucap = par->hsd_cucap > 0 ? par->hsd_cucap : bmax;   /* largest downward push, in mu_t */
    const double at = fmin(1.0, 1.5 * *am + 0.1);
    double td = wtime(), trT = 0, trial = 0;
    double *pr1 = hred(nb, 2);
    /* pass 1: trial products (LP: in Xn; SDP: eigenpairs in Ct, ev) */
    HPAR_FOR
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        double trial = 0;
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
            for (size_t i = 0; i < len; i++) { Xt[i] *= at; Zt[i] *= at; }
            for (int i = 0; i < n; i++) { Xt[i + (size_t)i * n] += s->lam[i]; Zt[i + (size_t)i * n] += s->lam[i]; }
        } else {
            /* the frame of this HKM form (Z dX + dZ X linearized): Z = L L',
             * L' X~ L and L^{-1} Z~ L^{-T} = I + at L^{-1} dZ L^{-T}                  */
            const double *L = s->LZ;
            for (size_t i = 0; i < len; i++) Xt[i] = s->X[i] + at * s->dX[i];
            xtrmm("L", "L", "T", "N", &n, &n, &DONE, L, &n, Xt, &n);
            xtrmm("R", "L", "N", "N", &n, &n, &DONE, L, &n, Xt, &n);
            memcpy(Zt, s->dZ, sizeof(double) * len);
            xtrsm("L", "L", "N", "N", &n, &n, &at, L, &n, Zt, &n);
            xtrsm("R", "L", "T", "N", &n, &n, &DONE, L, &n, Zt, &n);
            for (int i = 0; i < n; i++) Zt[i + (size_t)i * n] += 1.0;
        }
        pdgemm("N", "N", &n, &n, &n, &DONE, Xt, &n, Zt, &n, &DZERO, E, &n);
        symmetrize(n, E);
        for (int j = 0; j < n; j++) trial += E[j + (size_t)j * n];
        pr1[2 * (size_t)k] = trial;
    }
    for (int k = 0; k < nb; k++) trial += pr1[2 * (size_t)k];
    { double _t = wtime(); g_ct[0] += _t - td; }
    double tk = (tau + at * *dt) * (kap + at * *dk);
    trial += tk;
    if (mu_t <= 0) mu_t = trial / ndim1;  /* target: the trial mean */
    if (!(mu_t > 0)) { R->t_dense += wtime() - td; return 0; }
    /* pass 2: pushed-back parts and K */
    int eigfail = 0;
    HPAR_FOR
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        double *K = s->Q;
        double trT = 0;
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
            #pragma omp atomic write
            eigfail = 1;
            continue;
        }
        memset(T, 0, sizeof(double) * len);
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
            const double *L = s->LZ;
            memcpy(K, T, sizeof(double) * len);
            xtrsm("L", "L", "T", "N", &n, &n, &DONE, L, &n, K, &n);
            xtrsm("R", "L", "N", "N", &n, &n, &DONE, L, &n, K, &n);
            symmetrize(n, K);
        }
        pr1[2 * (size_t)k + 1] = trT;
    }
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
    HPAR_FOR
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const size_t len = bsz(B);
        const double *ZI = Zv[k] ? Zv[k] : s->Zi;
        if (sh != 0) for (size_t i = 0; i < len; i++) s->Q[i] -= sh * ZI[i];
        blk_Aop(B, s, s->Q, rhs, -1.0);
        pr1[2 * (size_t)k] = sp_inner(B, &B->C, s->Q);
    }
    aop_end();
    for (int k = 0; k < nb; k++) CK += pr1[2 * (size_t)k];
    R->t_dense += wtime() - td;
    double tc0 = wtime(); g_ct[2] += tc0 - tp2;
    double *ufc = FV->nf ? malloc(sizeof(double) * FV->nf) : NULL;
    if (FV->nf) free_solve(P, Sc, FV, rhs, NULL, uc, ufc);
    else schur_solve(Sc, rhs, uc, 1e-7);
    R->t_chol += wtime() - tc0; g_ct[3] += wtime() - tc0;
    const double dtc = (CK - ddot_n(m, P->b, uc) + ddot_n(m, hc, uc) + tcc / tau
                        + (FV->nf ? ddot_n(FV->nf, FV->c, ufc) : 0.0)) / denom;
    const double dkc = (tcc - kap * dtc) / tau;
    for (int i = 0; i < m; i++) uc[i] += dtc * v[i];
    td = wtime();
    HPAR_FOR
    for (int k = 0; k < nb; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        const int n = B->n;
        const size_t len = bsz(B);
        blk_ATy(B, s, uc, s->F);
        for (size_t i = 0; i < len; i++) s->F[i] = -s->F[i];
        sp_add(B, &B->C, dtc, s->F);                               /* dZc */
        const double *RX = Zv[k] ? s->Zi : s->X;
        if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * s->F[i] * s->X[i];
        else if (pk[k]) { prod_pat(&hp[k], n, s, s->F, RX, s->G); symmetrize(n, s->G); }
        else { prod3(n, s->Zi, s->F, RX, s->W, s->G); symmetrize(n, s->G); }
        memcpy(bX[k], s->dX, sizeof(double) * len);
        memcpy(bZ[k], s->dZ, sizeof(double) * len);
        for (size_t i = 0; i < len; i++) { s->dX[i] += s->Q[i] - s->G[i]; s->dZ[i] += s->F[i]; }
    }
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
    HPAR_FOR
    for (int k = 0; k < nb; k++) {
        const size_t len = bsz(&P->blk[k]);
        memcpy(S[k].dX, bX[k], sizeof(double) * len);
        memcpy(S[k].dZ, bZ[k], sizeof(double) * len);
    }
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
        for (size_t i = 0; i < len; i++) { Xr[i] = X[i] * f; Zr[i] = Z[i] * f; }
    }
}
static const char *g_hsd_why = NULL;      /* why the embedding stopped short (printed with the retry: user reports must be diagnosable) */
static int hsd_run(Problem *P, const Params *par, Result *R, BS *S, Schur *Sc, double *y,
                   double *ybest, double *best_score, int *it_out, int *it_best, Result *best,
                   int keep_best, RCand *rc) {
    const int m = P->m, nb = P->nblk;
    g_hsd_why = NULL;
    { const char *e = getenv("BRISK_KFIX0"); if (e) g_kfix0 = atof(e); if (getenv("BRISK_NOCBOX")) g_cbox = 0; }
    double tau = 1.0, kap = 1.0, ndim = 0;
    for (int k = 0; k < nb; k++) ndim += P->blk[k].n;
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
    { const char *e = getenv("BRISK_HPAR"); g_hpar = (e ? atoi(e) : 1) && nthreads() > 1 && nb >= 16; }
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
        double fx = ex ? atof(ex) : 1.0;
        const double fz = ez ? atof(ez) : 1.0;
        if (!ex) {
            int nsdp = 0, maxn = 0;
            for (int k = 0; k < nb; k++) if (P->blk[k].type != BLK_LP) { nsdp++; if (P->blk[k].n > maxn) maxn = P->blk[k].n; }
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
                for (size_t i = 0; i < len; i++) { S[k].X[i] *= fx; S[k].Z[i] *= fz; }
            }
    }
    if (ENV_ON("BRISK_SCALEDBG")) {
        double tx = 0, tz = 0, nd = 0;
        for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) { tx += S[k].X[i]; tz += S[k].Z[i]; }
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
    int ncorr = 0, nrefine = 0, ntry_c = 0, hoc_nfail = 0, hoc_nkept = 0;
    double tc_avg = 0, tc_sum = 0, t_iter_prev = 0, t_iter_start = wtime(), mu_at_best = INFINITY, mu_best_it = INFINITY;
    double score_prog = INFINITY;
    double *shist = malloc(sizeof(double) * 4096), *phist = calloc(4096, sizeof(double));
    double *ihist = calloc(4096, sizeof(double)), *mhist = calloc(4096, sizeof(double));
    int it_sprog = 0;   /* last iteration of score progress (caps the steady-progress extension) */
    const int steady_on = !getenv("BRISK_NOSTEADY");
    { const char *e = getenv("BRISK_PFLOOR"); g_pfloor = e ? atoi(e) : 1; }
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
        HPAR_FOR
        for (int k = 0; k < nb; k++) {                                 /* Z^{-1} */
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n, info;
            if (B->type == BLK_LP) {
                for (int i = 0; i < n; i++) s->Zi[i] = 1.0 / s->Z[i];
                if (s->fm) for (int i = 0; i < n; i++) if (s->fm[i]) s->Zi[i] = 0;
            }
            else {
                double *Zinv = Zv[k] ? Zv[k] : s->Zi;
                memcpy(Zinv, s->LZ, sizeof(double) * bsz(B));
                if (n <= (g_lockfree ? LOCKFREE_N : PROD_SMALL)) inv_from_chol_small(n, Zinv);
                else {
                    BL(dpotri_)("L", &n, Zinv, &n, &info);
                    if (info) {
                        #pragma omp atomic write
                        zerr = 1;
                        continue;
                    }
                }
                lower_to_full(n, Zinv);
                if (Zv[k] && nt_scaling(B, s)) {
                    #pragma omp atomic write
                    zerr = 1;
                }
            }
        }
        if (zerr) { status = 4; g_hsd_why = "Z of the start is not positive definite"; goto done; }
        g_ph_last = wtime();
        /* residuals of the embedding (tau = 1 by normalization below) */
        for (int i = 0; i < m; i++) rp[i] = tau * P->b[i];
        double cX = 0, xz = 0, rd2 = 0, nAty = 0, dev2 = 0;
        {
            /* 4.24: per-block partials summed in block order (any thread count) */
            double *pr = hred(nb, 5);
            aop_begin(rp, NULL, NULL, NULL);
            HPAR_FOR
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                double *r5 = pr + 5 * (size_t)k;
                blk_Aop(B, s, s->X, rp, -1.0);
                blk_ATy(B, s, y, s->F);
                for (size_t i = 0; i < len; i++) { s->F[i] += s->Z[i]; r5[0] += s->F[i] * s->F[i]; }
                for (size_t i = 0; i < len; i++) s->F[i] = -s->F[i];
                sp_add(B, &B->C, tau, s->F);                               /* F = Rd */
                r5[1] = ddot_n(len, s->F, s->F);
                if (use_pat) r5[2] = norm2_plus_R0(s, len, s->F, -theta);
                r5[3] = sp_inner(B, &B->C, s->X);
                r5[4] = ddot_n(len, s->X, s->Z);
            }
            aop_end();
            for (int k = 0; k < nb; k++) { const double *r5 = pr + 5 * (size_t)k; nAty += r5[0]; rd2 += r5[1]; dev2 += r5[2]; cX += r5[3]; xz += r5[4]; }
        }
        double bty = ddot_n(m, P->b, y), rg = kap + cX - bty;
        double cfx = 0;                            /* c_f' x_f: cX without it is the cone part */
        for (int f = 0; f < nf; f++) {
            rf[f] = S[FV.kb[f]].F[FV.ip[f]];       /* tau c_f - Af' y */
            FV.xf[f] = S[FV.kb[f]].X[FV.ip[f]];
            cfx += FV.c[f] * FV.xf[f];
        }
        double mu = (xz + tau * kap) / ndim1;
        if (ENV_ON("BRISK_SCALEDBG")) {
            double tx = 0, tz = 0;
            for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n;
                if (B->type == BLK_LP) for (int i = 0; i < n; i++) { tx += S[k].X[i]; tz += S[k].Z[i]; }
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
        R->t_dense += wtime() - td;
        if (strict_gap > 0 && !looseskip_safe) {
            const int strict = bord_on || R->relgap < strict_gap;
            soltol = strict ? par->hsd_soltol : soltol_cls;
            g_looseskip = strict ? 0 : looseskip_cls;
        }
        double score = res_score(R);
        if (par->verbose)
            printf("%3d %+.8e %+.8e %.2e %.2e %.2e %.2e %.3f %.3f %.3f %7.2f  kap/mu %.1e%s\n",
                   it, po, dob, R->pinf, R->dinf, R->relgap, mu, alpha, alpha, sigma,
                   wtime() - par->t_start, tau * kap / mu, "");
        dump_xz(P, S, it, tau);
        if (ENV_ON("BRISK_SCOREDBG")) printf("   SCORE it %d score %.3e comp %.3e best %.3e itbest %d t %.4f\n", it, score, R->relcomp, fmin(score, *best_score), score < *best_score ? it : *it_best, wtime() - par->t_start);
        if (ENV_ON("BRISK_HSDDBG")) {
            double trX = 0, trZ = 0;
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) { trX += S[k].X[i]; trZ += S[k].Z[i]; }
                else for (int i = 0; i < B->n; i++) { trX += S[k].X[i + (size_t)i * B->n]; trZ += S[k].Z[i + (size_t)i * B->n]; }
            }
            printf("     [tau %.3e kap %.3e  trX/tau %.3e trZ/tau %.3e |y|/tau %.3e relcomp %.2e score %.2e]\n", tau, kap, trX / tau, trZ / tau,
                   sqrt(ddot_n(m, y, y)) / tau, R->relcomp, score);
        }
        btrack_offer(par, P, S, y, tau, mu, R, it);
        rcand_offer(rc, P, S, y, tau, R, it);
        const int improved_er = score < *best_score;
        if (score < *best_score) {
            *best_score = score; *it_best = it; *best = *R;
            mu_best_it = mu;
            kap_best = kap / tau; theta_best = theta / tau;
            for (int i = 0; i < m; i++) ybest[i] = y[i] / tau;
            if (keep_best)
                for (int k = 0; k < nb; k++) {
                    size_t len = bsz(&P->blk[k]);
                    for (size_t i = 0; i < len; i++) { S[k].Xb[i] = S[k].X[i] / tau; S[k].Zb[i] = S[k].Z[i] / tau; }
                }
        }
        if (score < par->tol) { status = 0; break; }
        /* 4.27: the complementarity lags alone. <X,Z>/tau^2 = gap + (y'r_p - <X,R_d>)/tau^2,
         * and the dual residual is exact by construction (dZ from dy) while the primal one
         * has the floor of the Schur solves, so where |y| is large (the moments of the image
         * forms, |y|/tau ~ 1e3) relcomp ~ |y| pinf stalls 100-1000x above the other measures
         * and the run went on for 1-7 iterations until a stall rule stopped it and the final
         * Gram polish (which makes err6 = err5) fixed it. The polish is tried as soon as gap,
         * dinf and pinf allow it, and the run stops only if its verified score meets tol.
         * BRISK_EPOLISH=0 turns it off. */
        if (epolish_on && keep_best && !nf && !bord && n_epol < 4 && score > par->tol && R->relcomp > par->tol &&
            fmax(R->relgap, R->dinf) <= ptol && R->pinf <= epol_pgate * ptol &&
            fmax(R->pinf, fmax(R->relgap, R->dinf)) <= 0.5 * epol_last) {
            Result Rp = *R;
            double rres = 0;
            const double sp = probe_polish(P, S, tau, y, Sc, a0, dy, &Rp, &rres, par->tol);
            n_epol++;
            epol_last = fmax(R->pinf, fmax(R->relgap, R->dinf));
            if (par->verbose > 1) printf("   HSD: polish probe at iteration %d: score %.2e -> %.2e (Gram CG residual %.1e; pinf %.1e dinf %.1e gap %.1e comp %.1e err2 %.1e err4 %.1e)%s\n",
                                         it, score, sp, rres, Rp.pinf, Rp.dinf, Rp.relgap, Rp.relcomp, Rp.err[2], Rp.err[4], sp <= par->tol ? ", stopping" : "");
            if (sp <= par->tol) {
                /* the current iterate becomes the best one; brisk_solve returns the probe's
                 * candidate (s->G, s->Q, y/tau) with its measures */
                g_epol_R = Rp;
                *best_score = score; *it_best = it; *best = *R;
                memcpy(ybest, a0, sizeof(double) * m);                     /* y/tau of the candidate */
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    for (size_t i = 0; i < len; i++) { S[k].Xb[i] = S[k].X[i] / tau; S[k].Zb[i] = S[k].Z[i] / tau; }
                }
                g_epol_stop = 1;
                status = 0; break;
            }
            /* 5.4: a verified candidate within red_acc is kept too (a cantilever relaxation sent by a user:
             * the probe reached 3.7e-8 at iteration 19 and was dropped for being above 1e-8; the run
             * ended at 1.4e-6 after a re-solve, 137 s instead of 40 s) */
            if (sp <= fmax(ptol, par->red_acc) && sp < fb_score && sp < *best_score) {
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
                fb_score = sp; fb_R = Rp;
                if (par->verbose > 1) printf("   HSD: probe candidate %.2e kept as the fallback\n", sp);
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
                if (okx && fmax(res_score(&Rx), fmax(Rx.err[2], Rx.err[4])) <= par->tol) {
                    if (par->verbose > 1) printf("   HSD: the X-metric correction of iteration %d meets the tolerance, stopping\n", it);
                    g_epol_R = Rx;
                    *best_score = score; *it_best = it; *best = *R;
                    g_epol_stop = 1;
                    status = 0; break;
                }
            }
            /* F = tau C - A'y - Z again (gram_cg used it) */
            HPAR_FOR
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                const size_t len = bsz(B);
                blk_ATy(B, s, y, s->F);
                for (size_t i = 0; i < len; i++) s->F[i] += s->Z[i];
                for (size_t i = 0; i < len; i++) s->F[i] = -s->F[i];
                sp_add(B, &B->C, tau, s->F);
            }
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
        if (rprobe_on && par->polish_r && rc && rc->have && rc->it == it && keep_best && !nf && !bord && n_rprobe < 2 && score > par->tol) {
            const double rs = fmax(R->dinf, fabs(R->relcomp));
            if (rs <= 0.5 * par->tol && rs <= 0.25 * rprobe_last && R->pinf >= 4.0 * pinf_lo) {
                n_rprobe++; rprobe_last = rs;
                double **Xr2 = malloc(sizeof(double *) * nb), **Zr2 = malloc(sizeof(double *) * nb), **Wsave = malloc(sizeof(double *) * nb);
                for (int k = 0; k < nb; k++) { Xr2[k] = S[k].Xr; Zr2[k] = S[k].Zr; const size_t len = bsz(&P->blk[k]); Wsave[k] = malloc(sizeof(double) * (len + 1)); memcpy(Wsave[k], S[k].W, sizeof(double) * len); }
                Result Rx = *R;
                const int okr = polish_xmetric(P, S, Xr2, Zr2, rc->y, &Rx, score, Sc, dy, par->verbose > 1, 1);
                for (int k = 0; k < nb; k++) { memcpy(S[k].W, Wsave[k], sizeof(double) * bsz(&P->blk[k])); free(Wsave[k]); }
                free(Xr2); free(Zr2); free(Wsave);
                if (okr && fmax(res_score(&Rx), fmax(Rx.err[2], Rx.err[4])) <= par->tol) {
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
        if (it > 3 && !par->known_feasible) {
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
                    if (B->type == BLK_LP) { memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len); }
                    else {
                        memcpy(s->LX, s->X, sizeof(double) * len); memcpy(s->LZ, s->Z, sizeof(double) * len);
                        if (chol_lower(B->n, s->LX) || chol_lower(B->n, s->LZ)) bad = 1;
                    }
                    s->stepped[0] = s->stepped[1] = 0;
                }
                if (!bad) {
                    for (int i = 0; i < m; i++) y[i] = ybest[i];
                    tau = 1.0; kap = kap_best; theta = theta_best;
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
        if (it < 4096) { shist[it] = score; phist[it] = R->pinf; ihist[it] = fmax(R->pinf, R->dinf); mhist[it] = mu; }
        if (it >= 4 && it < 4096 && score < 0.7 * shist[it - 4] && score < 10.0 * *best_score && it > it_prog) it_prog = it;
        /* 4.38 bound mode (A3): an improving tracked bound is progress: the stall rules do
         * not stop the solve while it improves (the other side may be blowing up) */
        if (par->btrack && par->btrack->have && par->btrack->it_imp == it) it_prog = it;
        if (it_prog > it_sprog) it_sprog = it_prog;
        /* slow but steady convergence (4.22): over 8 iterations the infeasibility fell by
         * 10% and mu by half, while the score (relcomp, as tau falls) does not improve.
         * Large AC-OPF (TSSOS case6468: the objective moves 42 -> 149 over 60 iterations at
         * alpha 0.1-0.4, as MOSEK's does) stopped at iteration 27 with 8e-5. Bounded by
         * 6 hsd_stall iterations after the last score progress. */
        if (steady_on && it >= 8 && it < 4096 && ihist[it - 8] > 0 && ihist[it] < 0.9 * ihist[it - 8] &&
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
        const int full_steps = small_steps == 0 && mu < 1e-2 * mu_best_it && it - *it_best < 12;
        if (*best_score < 10.0 * ptol && it - *it_best >= 3 && score > 10.0 * *best_score && !full_steps) { status = 4; g_hsd_why = "the errors rise past the best iterate"; break; }
        /* primal floor (4.20): the primal residual dominates the score by 10x, has not fallen
         * in 4 iterations and the best iterate is 4 old: the directions no longer reduce it
         * (roa_univar_d16_I: pinf 3.5e-6 .. 4.4e-6 for 14 iterations while mu fell 1e4x) */
        if (g_pfloor && it >= 8 && it < 4096 && *best_score < 1e-4 && it - *it_best >= 4 && phist[it - 4] > 0 &&
            R->pinf >= 0.9 * phist[it - 4] && fmax(fmax(R->relgap, R->relcomp), R->dinf) < 0.1 * R->pinf) {
            if (par->verbose > 1) printf("   HSD: primal residual floor (%.1e), stopping\n", R->pinf);
            status = 4; g_hsd_why = "primal residual floor"; break;
        }

        if (use_pat) {
            const double dev = sqrt(dev2);
            if (dev > 1e-10 * (1.0 + theta * r0norm) && dev > 1e-12) {
                /* re-base: R0 <- true residual (dense: it contains Z), products go dense */
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
        memset(a0, 0, sizeof(double) * m);
        memset(hd, 0, sizeof(double) * m);
        memset(hc, 0, sizeof(double) * m);
        double hcRd = 0, hcC = 0, cZi = 0;
        double *prp3 = hred(nb, 3);
        aop_begin(a0, hd, hc, NULL);
        HPAR_FOR
        for (int k = 0; k < nb; k++) {
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
            else if (pk[k]) {                                  /* Rd = theta R0 on the pattern */
                memset(s->G, 0, sizeof(double) * len);
                add_R0(s, len, theta, s->G);
                prod_pat(&hp[k], n, s, s->G, RX, s->G);
                symmetrize(n, s->G);
            }
            else { prod3(n, s->Zi, s->F, RX, s->W, s->G); symmetrize(n, s->G); }
            blk_Aop(B, s, s->G, hd, 1.0);
            hcRd += sp_inner(B, &B->C, s->G);
            memset(s->dX, 0, sizeof(double) * len);
            sp_add(B, &B->C, 1.0, s->dX);                              /* dense copy of C */
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->Q[i] = s->Zi[i] * s->dX[i] * s->X[i];
            else if (pk[k]) { prod_pat(&hp[k], n, s, s->dX, RX, s->Q); symmetrize(n, s->Q); }
            else { prod3(n, s->Zi, s->dX, RX, s->W, s->Q); symmetrize(n, s->Q); }
            blk_Aop(B, s, s->Q, hc, 1.0);
            hcC += sp_inner(B, &B->C, s->Q);
            prp3[3 * (size_t)k] = cZi; prp3[3 * (size_t)k + 1] = hcRd; prp3[3 * (size_t)k + 2] = hcC;
        }
        aop_end();
        for (int k = 0; k < nb; k++) { cZi += prp3[3 * (size_t)k]; hcRd += prp3[3 * (size_t)k + 1]; hcC += prp3[3 * (size_t)k + 2]; }
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
        double denom = bv + hcC - hcv + kap / tau - (nf ? ddot_n(nf, FV.c, vf) : 0.0);
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
        double dtau_a = (rg - (cX - cfx) - hcRd - ddot_n(m, P->b, u0) + ddot_n(m, hc, u0) - kap
                         + (nf ? ddot_n(nf, FV.c, ufa) : 0.0)) / denom;
        double dkap_a = -kap - (kap / tau) * dtau_a;
        td = wtime();
        for (int i = 0; i < m; i++) dy[i] = u0[i] + dtau_a * v[i];
        HPAR_FOR
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            int n = B->n;
            blk_ATy(B, s, dy, s->dZ);
            if (pk[k]) { for (size_t i = 0; i < len; i++) s->dZ[i] = -s->dZ[i]; add_R0(s, len, theta, s->dZ); }
            else for (size_t i = 0; i < len; i++) s->dZ[i] = s->F[i] - s->dZ[i];
            sp_add(B, &B->C, dtau_a, s->dZ);
            const double *RX = Zv[k] ? s->Zi : s->X;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * s->dZ[i] * s->X[i];
            else if (pk[k]) { prod_pat(&hp[k], n, s, s->dZ, RX, s->G); symmetrize(n, s->G); }
            else { prod3(n, s->Zi, s->dZ, RX, s->W, s->G); symmetrize(n, s->G); }
            for (size_t i = 0; i < len; i++) s->dX[i] = -s->G[i] - s->X[i];
        }
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
            HPAR_FOR
            for (int k = 0; k < nb; k++) {
                size_t len = bsz(&P->blk[k]);
                double t = 0;
                for (size_t i = 0; i < len; i++)
                    t += (S[k].X[i] + aa * S[k].dX[i]) * (S[k].Z[i] + aa * S[k].dZ[i]);
                px[k] = t;
            }
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
            HPAR_FOR \
            for (int k = 0; k < nb; k++) { \
                size_t len = bsz(&P->blk[k]); \
                for (size_t i = 0; i < len; i++) { \
                    S[k].dX[i] = dX0[k][i] + sg_ * (dX1[k][i] - dX0[k][i]); \
                    S[k].dZ[i] = dZ0[k][i] + sg_ * (dZ1[k][i] - dZ0[k][i]); \
                } \
            } \
            tst = wtime(); \
            const double amX_ = maxstep(P, S, 1, par->lanczos_k), amZ_ = maxstep(P, S, 0, par->lanczos_k); \
            double am_ = fmin(amX_, amZ_); \
            R->t_step += wtime() - tst; \
            if (dt_ < 0) am_ = fmin(am_, -tau / dt_); \
            if (dk_ < 0) am_ = fmin(am_, -kap / dk_); \
            if (ENV_ON("BRISK_HSDDBG")) \
                printf("     [sigma %.3g: aX %.3f aZ %.3f atau %.3g akap %.3g -> %.3f]\n", sg_, amX_, amZ_, \
                       dt_ < 0 ? -tau / dt_ : 1e30, dk_ < 0 ? -kap / dk_ : 1e30, am_); \
            double gam_ = 0.9 + (par->gamma_max - 0.9) * fmin(1.0, am_); \
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
        memset(qv, 0, sizeof(double) * m);
        /* second-order term, optionally damped by the affine step (the correction for a step
         * alpha is alpha^2 dX dZ; Mehrotra's alpha = 1 over-corrects after short predictors) */
        const double qsc = par->hsd_qscale == 1 ? aa : par->hsd_qscale == 2 ? aa * aa : 1.0;
        cQ = 0; tk2 = qsc * dtau_a * dkap_a;
        double *pq = hred(nb, 1);
        aop_begin(qv, NULL, NULL, NULL);
        HPAR_FOR
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->Q[i] = s->Zi[i] * s->dZ[i] * s->dX[i];
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
            else { prod3(n, s->Zi, s->dZ, s->dX, s->W, s->Q); symmetrize(n, s->Q); }
            if (qsc != 1.0) { const size_t len = bsz(B); for (size_t i = 0; i < len; i++) s->Q[i] *= qsc; }
            blk_Aop(B, s, s->Q, qv, 1.0);
            pq[k] = sp_inner(B, &B->C, s->Q);
        }
        aop_end();
        for (int k = 0; k < nb; k++) cQ += pq[k];
        if (nhoc > 0) {
            /* the fixed point is followed while it contracts: |Q_{k+1} - Q_k| <= theta |Q_k - Q_{k-1}| */
            double d = 0;
            for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]); const double *q = S[k].Q, *qp = Qp[k]; double t = 0;
                if (hoc == 0) for (size_t i = 0; i < len; i++) t += q[i] * q[i]; else for (size_t i = 0; i < len; i++) t += (q[i] - qp[i]) * (q[i] - qp[i]);
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
        HPAR_FOR
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].X, rhs, 1.0);
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
        n0 = rg - cQ - (cX - cfx) - hcRd - ddot_n(m, P->b, u0) + ddot_n(m, hc, u0)
                    + (-tau * kap - tk2) / tau + (nf ? ddot_n(nf, FV.c, uf0) : 0.0);
        n1 = -rg + mu * cZi + hcRd - ddot_n(m, P->b, u1) + ddot_n(m, hc, u1) + mu / tau
                    + (nf ? ddot_n(nf, FV.c, uf1) : 0.0);
        dt0 = n0 / denom; dt1 = (n0 + n1) / denom;
        dk0 = (-tau * kap - tk2) / tau - (kap / tau) * dt0;
        dk1 = (mu - tau * kap - tk2) / tau - (kap / tau) * dt1;

        /* endpoint directions at sigma = 0 and sigma = 1 */
        td = wtime();
        for (int e = 0; e < 2; e++) {
            double sg = e ? 1.0 : 0.0, dt = e ? dt1 : dt0;
            for (int i = 0; i < m; i++) dy[i] = u0[i] + sg * u1[i] + dt * v[i];
            HPAR_FOR
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                size_t len = bsz(B);
                int n = B->n;
                double *DZ = e ? dZ1[k] : dZ0[k], *DX = e ? dX1[k] : dX0[k];
                blk_ATy(B, s, dy, DZ);
                if (pk[k]) { for (size_t i = 0; i < len; i++) DZ[i] = -DZ[i]; add_R0(s, len, (1.0 - sg) * theta, DZ); }
                else for (size_t i = 0; i < len; i++) DZ[i] = (1.0 - sg) * s->F[i] - DZ[i];
                sp_add(B, &B->C, dt, DZ);
                const double *RX = Zv[k] ? s->Zi : s->X, *ZI = Zv[k] ? Zv[k] : s->Zi;
                if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * DZ[i] * s->X[i];
                else if (pk[k]) { prod_pat(&hp[k], n, s, DZ, RX, s->G); symmetrize(n, s->G); }
                else { prod3(n, s->Zi, DZ, RX, s->W, s->G); symmetrize(n, s->G); }
                for (size_t i = 0; i < len; i++) DX[i] = -s->G[i] - s->X[i] - s->Q[i] + sg * mu * ZI[i];
            }
            if (nf) {
                for (int f2 = 0; f2 < nf; f2++) dxfd[f2] = uf0[f2] + sg * uf1[f2] + dt * vf[f2];
                free_set_dir(&FV, S, e ? dX1 : dX0, e ? dZ1 : dZ0, dxfd, 0);
            }
        }
        R->t_dense += wtime() - td;
        if (hoc_redo || nhoc == 0) break;
        {
            /* the predicted reduction of this pass's family (the sigma search): the passes are kept only if it
             * is at least the one of the first family; the fixed point is taken at the sigma the first search chose */
            double mer = -1, sgb = sig_m, alb = 0, amb = 0;
            HSD_SIGSEARCH(mer, sgb, alb, amb);
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
                HPAR_FOR
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    for (size_t i = 0; i < len; i++) {
                        S[k].dX[i] = dX0[k][i] + sg * (dX1[k][i] - dX0[k][i]);
                        S[k].dZ[i] = dZ0[k][i] + sg * (dZ1[k][i] - dZ0[k][i]);
                    }
                }
                dtau_a = dt0 + sg * (dt1 - dt0); dkap_a = dk0 + sg * (dk1 - dk0);
            }
        }
        if (hoc_redo == 2) {
            /* recompute the scalars and the endpoints from the restored u0 */
            hoc_redo = 1;
            n0 = rg - cQ - (cX - cfx) - hcRd - ddot_n(m, P->b, u0) + ddot_n(m, hc, u0) + (-tau * kap - tk2) / tau + (nf ? ddot_n(nf, FV.c, uf0) : 0.0);
            dt0 = n0 / denom; dt1 = (n0 + n1) / denom;
            dk0 = (-tau * kap - tk2) / tau - (kap / tau) * dt0;
            dk1 = (mu - tau * kap - tk2) / tau - (kap / tau) * dt1;
            for (int e = 0; e < 2; e++) {
                double sg = e ? 1.0 : 0.0, dt = e ? dt1 : dt0;
                for (int i = 0; i < m; i++) dy[i] = u0[i] + sg * u1[i] + dt * v[i];
                HPAR_FOR
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    size_t len = bsz(B);
                    int n = B->n;
                    double *DZ = e ? dZ1[k] : dZ0[k], *DX = e ? dX1[k] : dX0[k];
                    blk_ATy(B, s, dy, DZ);
                    if (pk[k]) { for (size_t i = 0; i < len; i++) DZ[i] = -DZ[i]; add_R0(s, len, (1.0 - sg) * theta, DZ); }
                    else for (size_t i = 0; i < len; i++) DZ[i] = (1.0 - sg) * s->F[i] - DZ[i];
                    sp_add(B, &B->C, dt, DZ);
                    const double *RX = Zv[k] ? s->Zi : s->X, *ZI = Zv[k] ? Zv[k] : s->Zi;
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = s->Zi[i] * DZ[i] * s->X[i];
                    else if (pk[k]) { prod_pat(&hp[k], n, s, DZ, RX, s->G); symmetrize(n, s->G); }
                    else { prod3(n, s->Zi, DZ, RX, s->W, s->G); symmetrize(n, s->G); }
                    for (size_t i = 0; i < len; i++) DX[i] = -s->G[i] - s->X[i] - s->Q[i] + sg * mu * ZI[i];
                }
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
            HPAR_FOR
            for (int k = 0; k < nb; k++) {
                size_t len = bsz(&P->blk[k]);
                for (size_t i = 0; i < len; i++) {
                    S[k].dX[i] = dX0[k][i] + sigma * (dX1[k][i] - dX0[k][i]);
                    S[k].dZ[i] = dZ0[k][i] + sigma * (dZ1[k][i] - dZ0[k][i]);
                }
            }
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
                                                  &best_am, mut, dX0, dZ0, rhs, u1, ndim1, R, &FV, vf, accf);
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
                    double gam = 0.9 + (par->gamma_max - 0.9) * fmin(1.0, best_am);
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
                HPAR_FOR
                for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].dX, e, -1.0);
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
                    /* 4.27: the correction resolves at most (1 - rel_e) of the miss, and the pass
                     * is kept only if it halves it: past rel_e 0.5 it would be rejected anyway
                     * (the image-form endgames: |e| ~1e-9 below the absolute accuracy of the
                     * Schur solve, rel_e 0.1-1, a solve and two block products per iteration) */
                    if (refskip_on && rel_e > 0.5) { ref_floor = rel_e * ne; R->t_chol += wtime() - tc; nref_fail++; break; }
                    if (refskip_on) ref_floor = 0;
                }
                R->t_chol += wtime() - tc;
                td = wtime();
                const double dte = -(ddot_n(m, P->b, u1) - ddot_n(m, hc, u1)
                                     - (nf ? ddot_n(nf, FV.c, dxfd) : 0.0)) / denom;
                for (int i = 0; i < m; i++) u1[i] += dte * v[i];
                for (int i = 0; i < m; i++) qv[i] = (1.0 - sigma) * rp[i] + P->b[i] * (dt + dte);
                HPAR_FOR
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    const int n = B->n;
                    const size_t len = bsz(B);
                    blk_ATy(B, s, u1, s->F);
                    for (size_t i = 0; i < len; i++) s->F[i] = -s->F[i];
                    sp_add(B, &B->C, dte, s->F);                           /* dZ_e */
                    const double *RX = Zv[k] ? s->Zi : s->X;
                    if (B->type == BLK_LP) for (int i = 0; i < n; i++) s->G[i] = -s->Zi[i] * s->F[i] * s->X[i];
                    else {
                        if (pk[k]) prod_pat(&hp[k], n, s, s->F, RX, s->G);
                        else prod3(n, s->Zi, s->F, RX, s->W, s->G);
                        symmetrize(n, s->G);
                        for (size_t i = 0; i < len; i++) s->G[i] = -s->G[i];
                    }
                    for (size_t i = 0; i < len; i++) dX0[k][i] = s->dX[i] + s->G[i];
                }
                if (nf) {
                    for (int f2 = 0; f2 < nf; f2++) dxfd[f2] += dte * vf[f2];
                    free_set_dir(&FV, S, dX0, NULL, dxfd, 1);    /* dX0 = dX + dX_e (also zeroes the masked dZ slots) */
                    for (int f2 = 0; f2 < nf; f2++) { S[FV.kb[f2]].F[FV.ip[f2]] = 0; S[FV.kb[f2]].F[FV.im[f2]] = 0; }
                }
                aop_begin(qv, NULL, NULL, NULL);
                HPAR_FOR
                for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], dX0[k], qv, -1.0);
                aop_end();
                double ne2 = sqrt(ddot_n(m, qv, qv));
                R->t_dense += wtime() - td;
                if (ENV_ON("BRISK_HSDDBG"))
                    printf("     [refine %d: |e| %.2e -> %.2e (|rp| %.2e)]\n", pass, ne, ne2, nrp);
                if (!(ne2 < 0.5 * ne)) { if (refskip_on && !nf) { ref_floor = ne2; nref_fail++; } break; }
                HPAR_FOR
                for (int k = 0; k < nb; k++) {
                    const size_t len = bsz(&P->blk[k]);
                    memcpy(S[k].dX, dX0[k], sizeof(double) * len);
                    for (size_t i = 0; i < len; i++) S[k].dZ[i] += S[k].F[i];
                }
                for (int i = 0; i < m; i++) dyd[i] += u1[i];
                dt += dte;
                dk -= kap * dte / tau;
                nrefine++;
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
                    HPAR_FOR
                    for (int k = 0; k < nb; k++) {
                        size_t len = bsz(&P->blk[k]);
                        double t = 0;
                        for (size_t i = 0; i < len; i++)
                            t += (S[k].X[i] + alpha * S[k].dX[i]) * (S[k].Z[i] + alpha * S[k].dZ[i]);
                        pm2[k] = t;
                    }
                    for (int k = 0; k < nb; k++) munew += pm2[k];
                }
                munew = (munew + tk) / ndim1;
                lo = tk;
                for (int k = 0; k < nb; k++) {
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
            { double _t = wtime(); g_ph[6] += _t - g_ph_last; g_ph_last = _t; }
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
                for (size_t i = 0; i < len; i++) { S[k].X[i] *= f; S[k].Z[i] *= f; }
                if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) { S[k].LX[i] *= f; S[k].LZ[i] *= f; }
                else for (size_t i = 0; i < len; i++) { S[k].LX[i] *= sq; S[k].LZ[i] *= sq; }
            }
            for (int i = 0; i < m; i++) y[i] *= f;
            kap *= f;
            theta *= f;
            tau = 1.0;
        }
    }
done:
    if (fbX) {
        if (fb_score < *best_score && !g_epol_stop) {
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
            for (size_t i = 0; i < len; i++) { S[k].X[i] /= tau; S[k].Z[i] /= tau; }
        }
    if (ENV_ON("BRISK_SCALEDBG")) {
        double tx = 0, tz = 0;
        for (int k = 0; k < nb; k++) { const Block *B = &P->blk[k]; const int n = B->n;
            if (B->type == BLK_LP) for (int i = 0; i < n; i++) { tx += S[k].X[i]; tz += S[k].Z[i]; }
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
    if (par->verbose && nrefine) printf("   (HSD: %d refinement passes)\n", nrefine);
    if (par->verbose && (hoc_nkept || hoc_nfail)) printf("   (HSD: second-order passes kept in %d iterations, worse than the first direction in %d)\n", hoc_nkept, hoc_nfail);
    if (par->verbose > 1 && (nref_skip || nref_fail)) printf("   (HSD: refinement at the solve floor: %d stopped after the solve, %d skipped)\n", nref_fail, nref_skip);
    free(dyd);
    *it_out = it;
    for (int k = 0; k < nb; k++) { free(dX0[k]); free(dZ0[k]); free(dX1[k]); free(dZ1[k]); }
    free(dX0); free(dZ0); free(dX1); free(dZ1);
    for (int k = 0; k < nb; k++) { free(hp[k].fr); free(hp[k].fc); free(hp[k].rows); }
    free(hp); free(pk);
    for (int k = 0; k < nb; k++) free(Zv[k]);
    free(Zv);
    if (par->verbose > 1 && nrebase) printf("   (HSD: residual re-based %d times)\n", nrebase);
    free(pk_b); free(pk_p); free(pk_q); free(shist); free(phist); free(ihist); free(mhist);
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
 * - A busy machine: when the thread count is the default (no -threads, no OMP_NUM_THREADS)
 *   and other threads are running right now, the solve uses the cores that are free. With
 *   spinning barriers, one thread sharing a core with another process stalls all the others
 *   (arch0 on two cores with one taken: 1.34 s against 0.37 s on one thread). The count of
 *   runnable threads is the fourth field of /proc/loadavg (Linux; elsewhere no change): the
 *   minimum of three samples a millisecond apart, so that a passing wake-up does not count.
 *   BRISK_NOBUSY=1 turns the rule off.
 * The thread count is restored on return. */
int g_threads_user = 0;      /* -threads or OMP_NUM_THREADS given (main.c) */
static int free_cores(void) {
#ifdef __linux__
    int other = 1 << 30;
    for (int s = 0; s < 3; s++) {
        FILE *f = fopen("/proc/loadavg", "r");
        if (!f) return -1;
        double a, b, c; int r = 0, t = 0;
        const int ok = fscanf(f, "%lf %lf %lf %d/%d", &a, &b, &c, &r, &t) == 5;
        fclose(f);
        if (!ok) return -1;
        if (r - 1 < other) other = r - 1;
        if (other <= 0) break;
        struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL);
    }
    if (other < 0) other = 0;
    const long on = sysconf(_SC_NPROCESSORS_ONLN);
    const int fr = (int)on - other;
    return fr < 1 ? 1 : fr;
#else
    return -1;
#endif
}
/* the busy-machine rule for the other engines (main.c): lowers the thread count to the free
 * cores and returns the count to restore, or -1 when nothing was changed */
/* the log line "Number of threads: k": the OpenMP threads the engine that starts now will use.
 * Printed once per run, and again only if a later solve of the same run uses another count
 * (brisk_log_threads_reset at the start of a run). */
static int g_nt_logged = -1;
void brisk_log_threads_reset(void) { g_nt_logged = -1; }
int brisk_threads_logged(void) { return g_nt_logged; }   /* 5.8: the count last logged, for the summary (-1: none) */
void brisk_log_threads(int verbose) {
    const int nt = nthreads();
    if (verbose > 0 && nt != g_nt_logged) { printf("Number of threads: %d\n", nt); g_nt_logged = nt; }
}
int brisk_threads_busy(int verbose) {
#ifdef _OPENMP
    const int nt0 = omp_get_max_threads();
    if (nt0 > 1 && !g_threads_user && !getenv("BRISK_NOBUSY")) {
        const int fr = free_cores();
        if (fr >= 1 && fr < nt0) {
            if (verbose > 0) printf("   threads: %d of %d (other threads are running on this machine; -threads k fixes the count)\n", fr, nt0);
            omp_set_num_threads(fr);
            return nt0;
        }
    }
#endif
    (void)verbose;
    return -1;
}
int brisk_solve(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout) {
#ifdef _OPENMP
    const int nt0 = omp_get_max_threads();
    if (nt0 > 1) {
        int nt = nt0;
        double w = (double)P->m * P->m * P->m / 3.0;
        for (int k = 0; k < P->nblk; k++) {
            const double n = P->blk[k].n;
            w += P->blk[k].type == BLK_SDP ? 12.0 * n * n * n + (double)P->blk[k].ncon * n * n : n * (1.0 + P->blk[k].ncon);
        }
        if (w < 5e6 && !getenv("BRISK_TINYMT")) nt = 1;
        else if (!g_threads_user && !getenv("BRISK_NOBUSY")) {
            const int fr = free_cores();
            if (fr >= 1 && fr < nt) {
                nt = fr;
                if (par_user->verbose > 0) printf("   threads: %d of %d (other threads are running on this machine; -threads k fixes the count)\n", nt, nt0);
            }
        }
        if (nt < nt0) {
            omp_set_num_threads(nt);
            brisk_log_threads(par_user->verbose);
            const int rc = brisk_solve_impl(P, par_user, R, yout, Xout);
            omp_set_num_threads(nt0);
            return rc;
        }
    }
#endif
    brisk_log_threads(par_user->verbose);
    return brisk_solve_impl(P, par_user, R, yout, Xout);
}
static int brisk_solve_impl(Problem *P, const Params *par_user, Result *R, double *yout, double **Xout) {
    double t0 = wtime();
    /* direction: -1 = automatic. NT costs about 15 n^3 more dense work per SDP block per
     * iteration than HKM; use it when that is small next to the Schur factorization
     * (m^3/3) plus HKM's own dense work, i.e. on Schur-dominated problems,            */
    Params par_local = *par_user;
    const Params *par = &par_local;
    g_free_hsd = 0;
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
    BS *S = calloc(nb, sizeof(BS));
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
        s->R0 = amalloc(len);
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
            s->lq = amalloc(sizeof(double) * (size_t)n * (LANCZOS_K + 1));
            if (par->max_correctors > 0) { s->Ct = amalloc(len); s->Dxc = amalloc(len); s->ev = amalloc(sizeof(double) * n); }
            s->v1 = amalloc(sizeof(double) * n);
            s->v2 = amalloc(sizeof(double) * n);
        }
        Ntot += n;
    }
    const int keep_best = par->polish && polish_bytes <= par->polish_mem;
    /* parallel over blocks when there are several and all are small (larger blocks get
     * more from threaded BLAS inside each block); BLAS runs single-threaded meanwhile,
     * except in the Schur factorization                                               */
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
        if (par->verbose > 1 && g_par_blk) printf("   parallel over %d blocks\n", nb);
    }
    if (keep_best)
        for (int k = 0; k < nb; k++) {
            size_t len = bsz(&P->blk[k]) * sizeof(double);
            S[k].Xb = amalloc(len); S[k].Zb = amalloc(len);
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
            double nk = B->n;
            if (B->type == BLK_SDP) work_it += 15.0 * nk * nk * nk;
            for (int t = 0; t < B->ncon; t++) work_it += 2.0 * B->A[t].ef * (B->type == BLK_SDP ? nk : 1.0);
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
    start_sdpt3(P, S, y);
    if (par->init > 0 && m > 0) {
        /* Least-norm primal / least-squares dual on the Gram matrix A A'
         * (= Schur complement at X = Zi = I), then shift into the cone. */
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            memset(s->X, 0, sizeof(double) * len);
            memset(s->Zi, 0, sizeof(double) * len);
            if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) s->X[i] = s->Zi[i] = 1.0;
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
        double tcg = wtime();
        double r1 = gram_cg(P, S, P->b, dy, 200, 1e-10, Sc.r, Sc.p, Sc.q);
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
                else for (int i = 0; i < B->n; i++) s->X[i + (size_t)i * B->n] = s->Zi[i + (size_t)i * B->n] = 1.0;
            }
            {
                double *Ma = schur_asm_begin(&Sc);
                for (int k = 0; k < nb; k++) {
                    if (P->blk[k].type == BLK_LP) schur_lp(&P->blk[k], &S[k], m, Ma);
                    else schur_sdp_id(&P->blk[k], &S[k], m, Ma, 1);
                }
                schur_asm_end(&Sc);
            }
            if (schur_factor(&Sc) == 0) {
                schur_solve(&Sc, P->b, dy, 1e-8);
                schur_solve(&Sc, rhs, y, 1e-8);
                init_ok = 1;
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
                } else {
                    double lx, lz;
                    sdp_maxstep_lam(n, NULL, s->X, s, par->lanczos_k, &lx);
                    sdp_maxstep_lam(n, NULL, s->Z, s, par->lanczos_k, &lz);
                    ax = fmax(ax, -lx);
                    az = fmax(az, -lz);
                }
            }
            double shx = ax >= -1e-8 ? 1.0 + ax : 0.0;
            double shz = az >= -1e-8 ? 1.0 + az : 0.0;
            for (int pass = 0; pass < 2; pass++) {
                trX = trZ = xz = 0;
                for (int k = 0; k < nb; k++) {
                    const Block *B = &P->blk[k];
                    BS *s = &S[k];
                    int n = B->n;
                    for (int i = 0; i < n; i++) {
                        size_t d = (B->type == BLK_LP) ? (size_t)i : (size_t)i * n + i;
                        s->X[d] += shx; s->Z[d] += shz;
                        trX += s->X[d]; trZ += s->Z[d];
                    }
                    xz += ddot_n(bsz(B), s->X, s->Z);
                }
                if (par->init < 2 || pass == 1) break;
                /* Mehrotra balancing: raise both sides until complementarity is centred */
                shx = 0.5 * xz / trZ;
                shz = 0.5 * xz / trX;
            }
            for (int k = 0; k < nb; k++) {
                const Block *B = &P->blk[k];
                BS *s = &S[k];
                if (B->type == BLK_LP) continue;
                int n = B->n;
                size_t len = bsz(B);
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
            xz0 += ddot_n(bsz(B), S[k].X, S[k].Z); nd0 += B->n;
            if (B->type == BLK_LP) {
                for (int t = 0; t < B->ncon; t++) if (B->con[t] == tb_row0 && B->A[t].nnz == 1) { tbk = k; tbi = B->A[t].row[0]; }
                continue;
            }
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
    if (keep_best && par->polish_r && 2.0 * polish_bytes <= par->polish_mem) {
        rc.y = malloc(sizeof(double) * (m + 1));
        for (int k = 0; k < nb; k++) { const size_t len = bsz(&P->blk[k]) * sizeof(double); S[k].Xr = amalloc(len); S[k].Zr = amalloc(len); }
    }
    if (par->hsd == 1) status = hsd_run(P, par, R, S, &Sc, y, ybest, &best_score, &it, &it_best, &best, keep_best, &rc);
    else
    for (it = 0; it <= par->maxit; it++) {
        double td = wtime();
        /* inverse of Z (HKM) or the NT scaling point W (stored in Zi) */
        int nt_fail = 0;
        PAR_BLOCKS
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            int n = B->n, info;
            if (nt) {
                if (nt_scaling(B, s)) {
                    #pragma omp atomic write
                    nt_fail = 1;
                }
            }
            else if (B->type == BLK_LP) { for (int i = 0; i < n; i++) s->Zi[i] = 1.0 / s->Z[i]; }
            else {
                memcpy(s->Zi, s->LZ, sizeof(double) * bsz(B));
                BL(dpotri_)("L", &n, s->Zi, &n, &info);
                lower_to_full(n, s->Zi);
            }
        }
        if (nt_fail) { status = 4; break; }
        /* residuals and objectives */
        memcpy(rp, P->b, sizeof(double) * m);
        double pobj = 0, xz = 0, rdn2 = 0, nCR2 = 0, dev2 = 0;
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].X, rp, -1.0);
        PAR_BLOCKS
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            blk_ATy(B, s, y, s->F);
            /* F <- A'y + Z - C = -Rd (true dual residual, not stored separately) */
            double ncr = 0;
            for (size_t i = 0; i < len; i++) {
                double v = s->F[i] + s->Z[i];
                ncr += v * v;                      /* ||A'y + Z||^2 */
                s->F[i] = v;
            }
            sp_add(B, &B->C, -1.0, s->F);
            s->part[0] = ncr;
            s->part[1] = sp_inner(B, &B->C, s->X);
            s->part[2] = ddot_n(len, s->X, s->Z);
            s->part[3] = ddot_n(len, s->F, s->F);
            s->part[4] = norm2_plus_R0(s, len, s->F, theta);   /* ||Rd - theta R0||^2 */
        }
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
        R->t_dense += wtime() - td;
        R->iters = it;
        if (par->verbose)
            printf("%3d %+.8e %+.8e %.2e %.2e %.2e %.2e %.3f %.3f %.3f %6.2f\n", it, -po, -dob,
                   R->pinf, R->dinf, R->relgap, sc * mu, alp, ald, sigma, wtime() - t0);
        dump_xz(P, S, it, 1.0);
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
                    size_t len = bsz(&P->blk[k]) * sizeof(double);
                    memcpy(S[k].Xb, S[k].X, len); memcpy(S[k].Zb, S[k].Z, len);
                }
        }
        if (res_score(R) < par->tol) { status = 0; break; }
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
        const int it_best_eff = par->btrack && par->btrack->have && par->btrack->it_imp > it_best ? par->btrack->it_imp : it_best;
        if (it - it_best_eff >= par->stall_win && best_score < 1e-6) { status = 4; break; }
        if (it - it_best_eff >= 8 && best_score < 1e-4) { status = 4; break; }
        if (it - it_best_eff >= 12) { status = 4; break; }          /* no progress at any level */
        if (it - it_best_eff >= 25) { status = 4; break; }
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
                    for (size_t i = 0; i < len; i++) s->R0[i] = -s->F[i];
                    if (B->type == BLK_SDP) setup_residual_pattern(B, s, par);
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
            PAR_BLOCKS
            for (int k = 0; k < nb; k++) {
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
            }
            if (split) for (int i = 0; i < m; i++) dy[i] += y1[i];
            R->t_dense += wtime() - td;
        } else {
        for (int i = 0; i < m; i++) rhs[i] = P->b[i] + h[i];
        tc = wtime();
        schur_solve(&Sc, rhs, dy, 1e-7);
        R->t_chol += wtime() - tc;
        td = wtime();
        PAR_BLOCKS
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            blk_ATy(B, s, dy, s->F);
            for (size_t i = 0; i < len; i++) s->dZ[i] = -s->F[i];
            add_R0(s, len, theta, s->dZ);
            if (B->type == BLK_LP) {
                for (int i = 0; i < B->n; i++) s->dX[i] = -s->Zi[i] * s->dZ[i] * s->X[i] - s->X[i];
            } else {
                /* Zi (A'dy - Rd) X in one product over the (combined) pattern */
                prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->X, s->G);
                symmetrize(B->n, s->G);
                for (size_t i = 0; i < len; i++) s->dX[i] = -s->G[i] - s->X[i];
            }
        }
        R->t_dense += wtime() - td;
        }
        double tst = wtime();
        /* Predictor step lengths only feed the centering heuristic. A looser Lanczos
         * pays off only when step lengths are a noticeable part of the work (large
         * dense blocks); otherwise the more accurate sigma saves iterations. */
        int loose = loose_pred;
        lz_tol = loose ? 0.1 : 0.02;
        int kp = loose ? (par->lanczos_k < 12 ? par->lanczos_k : 12) : par->lanczos_k;
        double ap = fmin(1.0, maxstep(P, S, 1, kp));
        double ad = fmin(1.0, maxstep(P, S, 0, kp));
        lz_tol = 0.02;
        R->t_step += wtime() - tst;
        double xdz = 0, dxz = 0, dxdz = 0;
        for (int k = 0; k < nb; k++) {
            size_t len = bsz(&P->blk[k]);
            BS *s = &S[k];
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
            PAR_BLOCKS
            for (int k = 0; k < nb; k++) {
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
            }
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
            PAR_BLOCKS
            for (int k = 0; k < nb; k++) {
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
            }
            if (split) for (int i = 0; i < m; i++) dy[i] += y1[i];
            R->t_dense += wtime() - td;
        } else {
        td = wtime();
        memset(qv, 0, sizeof(double) * m);
        PAR_BLOCKS
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            if (B->type == BLK_LP) for (int i = 0; i < B->n; i++) s->Q[i] = s->Zi[i] * s->dZ[i] * s->dX[i];
            else prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->dX, s->Q);
        }
        for (int k = 0; k < nb; k++) blk_Aop(&P->blk[k], &S[k], S[k].Q, qv, 1.0);
        for (int i = 0; i < m; i++) rhs[i] = P->b[i] - smu * a0[i] + qv[i] + h[i];
        R->t_dense += wtime() - td;
        tc = wtime();
        relc = schur_solve(&Sc, rhs, dy, 0);
        if (ENV_ON("BRISK_DEBUG"))
            printf("     [corrector solve rel res %.1e, reg %.1e, min pivot %.1e, pcg total %d, |rhs| %.2e, |dy| %.2e]\n",
                   relc, Sc.reg, Sc.min_pivot, Sc.pcg_iters, sqrt(ddot_n(m, rhs, rhs)), sqrt(ddot_n(m, dy, dy)));
        R->t_chol += wtime() - tc;
        td = wtime();
        PAR_BLOCKS
        for (int k = 0; k < nb; k++) {
            const Block *B = &P->blk[k];
            BS *s = &S[k];
            size_t len = bsz(B);
            blk_ATy(B, s, dy, s->F);
            for (size_t i = 0; i < len; i++) s->dZ[i] = -s->F[i];
            add_R0(s, len, theta, s->dZ);
            if (B->type == BLK_LP) {
                for (int i = 0; i < B->n; i++)
                    s->dX[i] = smu * s->Zi[i] - s->Zi[i] * s->dZ[i] * s->X[i] - s->Q[i] - s->X[i];
            } else {
                prod_pat(hasH ? &s->cpat : &s->upat, B->n, s, s->dZ, s->X, s->G);
                for (size_t i = 0; i < len; i++) s->dX[i] = -s->G[i] - s->Q[i];
                symmetrize(B->n, s->dX);
                for (size_t i = 0; i < len; i++) s->dX[i] += smu * s->Zi[i] - s->X[i];
            }
        }
        R->t_dense += wtime() - td;
        }

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

        /* ---- multiple centrality correctors (only when factorization dominates) ---- */
        tst = wtime();
        double apr = fmin(1.0, maxstep(P, S, 1, par->lanczos_k));
        double adr = fmin(1.0, maxstep(P, S, 0, par->lanczos_k));
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

        int okp = try_step(P, S, 1, &ap);
        int okd = try_step(P, S, 0, &ad);
        R->t_step += wtime() - tst;
        if (!okp || !okd) { status = 4; break; }
        for (int i = 0; i < m; i++) y[i] += ad * dy[i];
        theta *= (1.0 - ad);
        alp = ap; ald = ad;
        stall = (ap < 1e-6 && ad < 1e-6) ? stall + 1 : 0;
    }
    const double work_run = work_it * (it + 1);
    int xsrc = 0;                 /* returned X: 0 S.X, 1 S.Xb, 2 S.G (polished) */
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
        const int timeout = status == ST_TIME;
        if (par->dd_end && !par->dd_done && !mixed && !timeout && status != 3 && res_score(R) >= par->tol) {
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
                if (B->type == BLK_SDP) n3 += 30.0 * nk * nk * nk;
            }
            assem += 0.5 * (double)m * sum_ef;
            /* 5.9: the endgame factors the Schur complement as a sparse matrix (ddend.c): the work
             * of the double solver's sparse factorization when there is one, 4 m^2 for the dense
             * passes over M */
            const double chol_fl = Sc.SC ? schol_flops(Sc.SC) + 4.0 * (double)m * m : (double)m * m * m / 3.0;
            const double dd_it = 45.0 * (assem + chol_fl + n3);   /* measured: a dd iteration costs ~45 double ones (m = 936) */
            const double budget = fmax(par->dd_minwork, par->dd_factor * (par->work_done + work_run));
            int dd_maxit = (int)fmin((double)par->dd_iters, floor(budget / dd_it));
            if (par->verbose > 1)
                printf("   endgame: %.1e flops per dd iteration, budget %.1e -> %d iterations allowed\n",
                       dd_it, budget, dd_maxit);
            if (m <= par->dd_maxm && n2 <= par->dd_maxn2 && dd_maxit >= 3) {
                int use_best = keep_best && best_score <= res_score(R) * (1 + 1e-12);
                double **Xs = malloc(sizeof(double *) * nb), **Zs = malloc(sizeof(double *) * nb);
                double *yc = malloc(sizeof(double) * (m + 1));
                memcpy(yc, use_best ? ybest : y, sizeof(double) * m);
                for (int k = 0; k < nb; k++) {
                    size_t len = bsz(&P->blk[k]) * sizeof(double);
                    memcpy(S[k].G, use_best ? S[k].Xb : S[k].X, len);
                    memcpy(S[k].Q, use_best ? S[k].Zb : S[k].Z, len);
                    Xs[k] = S[k].G; Zs[k] = S[k].Q;
                }
                double cur2 = use_best ? best_score : res_score(R);
                ran_dd = 1;
                const double tdd0 = wtime(); int dd_ok = dd_endgame(P, Xs, Zs, yc, par, dd_maxit, par->tol, par->verbose);
                if (getenv("BRISK_DDTIME")) printf("   [endgame: %.3f s, estimate %.2e flop units per iteration, %d allowed]\n", wtime() - tdd0, dd_it, dd_maxit);
                if (dd_ok > 0) {
                    Result Rp = *R;
                    double sp = measure_candidate(P, S, Xs, Zs, yc, &Rp, Sc.t);
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
                free(Xs); free(Zs); free(yc);
            }
        }

        /* polishing of the returned point (best iterate if stored, else final) */
        /* 4.25: not past the time limit (thetaG51 -timelimit 1: 5 s of polishing after the stop) */
        if (par->polish && status != 3 && !mixed && !timeout) {
            double **Xb = malloc(sizeof(double *) * nb), **Zb = malloc(sizeof(double *) * nb);
            int use_best = keep_best && best_score <= res_score(R) * (1 + 1e-12);
            for (int k = 0; k < nb; k++) {
                Xb[k] = use_best ? S[k].Xb : S[k].X;
                Zb[k] = use_best ? S[k].Zb : S[k].Z;
            }
            R->iters = it;
            if (use_best || !keep_best || res_score(R) <= best_score) {
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
                    if (rc.have && rc.it != it_ret && rc.score < sc0) {
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
            }
            free(Xb); free(Zb);
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
    if (Xout)
        for (int k = 0; k < nb; k++) {
            const double *Xk = xsrc == 2 ? S[k].G : xsrc == 1 ? S[k].Xb : S[k].X;
            size_t len = bsz(&P->blk[k]);
            if (P->blk[k].type == BLK_LP) for (size_t i = 0; i < len; i++) Xout[k][i] = Xk[i] * P->bs;
            else {
                const int n = P->blk[k].n;
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++)
                        Xout[k][i + (size_t)j * n] = 0.5 * (Xk[i + (size_t)j * n] + Xk[j + (size_t)i * n]) * P->bs;
            }
        }
    R->t_total = wtime() - t0;
    /* Automatic method choice: a non-optimal exit that is not an infeasibility
     * certificate is retried with the embedding, and the better result is kept.
     * Results already within retry_acc are returned as they are: going from 1e-7 to
     * 1e-10 is not worth a second solve. The retry runs after this run's work arrays
     * have been freed.                                                              */
    double s1 = fmax(res_score(R), fmax(R->err[2], R->err[4]));
    const int time_left = !(brisk_time_up(par));
    /* retry: standard -> embedding; with the embedding first, embedding -> standard */
    const int do_retry = par_user->hsd < 0 && (!par_local.hsd || par_local.hsd_first_on) && status != 0 && status != 1 && status != 2 &&
                         !par_local.no_retry && time_left && !(s1 <= (par_local.hsd_first_on ? par->retry_hsd1 : par->retry_acc));
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
    if (getenv("BRISK_SOLDBG")) printf("   [bord factor: %ld calls, %.3fs total, %.3fs in schol_factor]\n", g_bf_n, g_bf_all, g_bf_fact);
    if (getenv("BRISK_SOLDBG")) { extern long g_pcgh[12]; printf("   [single PCG runs by steps 0..10+:"); for (int q = 0; q <= 10; q++) printf(" %ld", g_pcgh[q]); printf("; ended above tol %ld]\n", g_pcgh[11]); }
    if (getenv("BRISK_SOLDBG")) { extern double g_tmv[2]; extern long g_nmv[2]; printf("   [schol_mv %ld calls %.3fs; sparse solves %ld rhs %.3fs]\n", g_nmv[0], g_tmv[0], g_nmv[1], g_tmv[1]); }
    if (getenv("BRISK_SOLDBG")) printf("   [bord: from (1e-13, 1e-11] one pass reached 1e-13 %ld times, not %ld (worst %.1e)]\n", g_b1ok, g_b1bad, g_b1worst);
    if (getenv("BRISK_SOLDBG")) printf("   [bord: first-pass rel <=1e-13 %ld, <=1e-11 %ld, <=1e-9 %ld, <=1e-7 %ld, worse %ld; stop after pass 0:%ld 1:%ld 2:%ld 3:%ld 4:%ld 5+:%ld]\n",
        g_bh0[0], g_bh0[1], g_bh0[2], g_bh0[3], g_bh0[4], g_bhp[0], g_bhp[1], g_bhp[2], g_bhp[3], g_bhp[4], g_bhp[5]);
    if (getenv("BRISK_SOLDBG")) printf("   [multi column-check skips %ld] [solves: single trusted %ld ok %ld pcg %ld | multi trusted %ld ok %ld pcg %ld | bord %ld | mv %ld]\n",
        g_multiskip, g_sc_cnt[0], g_sc_cnt[1], g_sc_cnt[2], g_sc_cnt[3], g_sc_cnt[4], g_sc_cnt[5], g_sc_cnt[6], g_sc_cnt[7]);
    { extern void schol_fstat_print(void); schol_fstat_print(); }
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
                           s->dXt, s->dZt, s->Rt };
        for (size_t q = 0; q < sizeof(bufs) / sizeof(bufs[0]); q++) free(bufs[q]);
        free(s->r0r); free(s->r0c); free(s->lr_owner);
        if (s->vbs) {
            free(s->vbs->X); free(s->vbs->Zi); free(s->vbs->dtmp); free(s->vbs->Gd); free(s->vbs->Mdd);
            free(s->vbs);
        }
        free(s->dW);
        if (s->cpat.owned) { free(s->cpat.fr); free(s->cpat.fc); free(s->cpat.rows); }
    }
    free(ev); free(wv); free(w1); free(y1); free(ybest); free(slowh); free(rc.y);
    endrules_free(&ER_std);
    free(S); free(y); free(dy); free(rhs); free(rp); free(a0); free(h); free(qv);
    free(Gs.M); free(Gs.Md); free(Gs.Mf); free(Gs.ya); envelope_free(Gs.E); free(Gs.e1); free(Gs.e2); free(Gs.e3);
    free(Gs.D); free(Gs.r); free(Gs.t); free(Gs.p); free(Gs.q); free(Gs.xb); free(Gs.fw);
    free(lsz); free(lsp); free(lss); free(lsy); free(lst); free(lsn); free(lsr); free(lsq);
    if (g_blas_T > 1 && blas_T_prev == 0) BL(openblas_set_num_threads)(g_blas_T);
    g_blas_T = blas_T_prev;
    g_par_blk = par_blk_prev;
    R->nreg = Sc.nreg; R->npcg = Sc.pcg_iters;
    schur_pos_free(&Sc);
    free(Sc.M); free(Sc.Md); free(Sc.Mf); free(Sc.ya); free(Sc.Masm); envelope_free(Sc.E); schol_free(Sc.SC); free(Sc.e1); free(Sc.e2); free(Sc.e3); free(Sc.D); free(Sc.r); free(Sc.t); free(Sc.p); free(Sc.q); free(Sc.xb); free(Sc.fw);
    if (do_retry) {
        Params p2 = *par_user;
        p2.hsd = par_local.hsd_first_on ? 0 : 1;
        p2.no_retry = 1;
        p2.dd_done = par_user->dd_done || ran_dd;
        p2.work_done = par_user->work_done + work_run;
        p2.warm_X = wX; p2.warm_Z = wZ; p2.warm_y = wy;
        if (par_user->verbose) printf("   retrying with the %s (status %d, accuracy %.1e%s%s)\n", p2.hsd ? "self-dual embedding" : "standard method", status, s1,
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
        } else {
            R->methods |= R2.methods;
            R->t_total += R2.t_total;          /* the rejected retry still cost time */
            R->iters += R2.iters;
        }
        R->nretry = 1; R->retry_kept = take; R->retry_hsd = p2.hsd; R->t_retry = t_r; R->acc_first = s1;
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
    PAR_BLOCKS
    for (int k = 0; k < P->nblk; k++) {
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        ds_Z(B, s, y, r, s->Z);
        if (s->dzs) {
            if (!ds_schol(s->dzs, s->Z, &s->part[0])) {
                #pragma omp atomic write
                ok = 0;
            }
            continue;
        }
        memcpy(s->LZ, s->Z, sizeof(double) * bsz(B));
        if (!ds_chol(B, s->LZ, &s->part[0])) {
            #pragma omp atomic write
            ok = 0;
        }
    }
    if (!ok) return 0;
    double ld = 0;
    for (int k = 0; k < P->nblk; k++) ld += S[k].part[0];
    if (logdet) *logdet = ld;
    return 1;
}
/* PSD test of C - A'(yt) + r I using the scratch buffer G (does not touch LZ) */
static int ds_test(const Problem *P, BS *S, const double *yt, double r, double *logdet) {
    int ok = 1;
    PAR_BLOCKS
    for (int k = 0; k < P->nblk; k++) {
        if (!ok) continue;
        const Block *B = &P->blk[k];
        BS *s = &S[k];
        int okb;
        if (s->dzt) okb = ds_schol_y(B, s->dzt, yt, r, &s->part[1]);
        else { ds_Z(B, s, yt, r, s->G); okb = ds_chol(B, s->G, &s->part[1]); }
        if (!okb) {
            #pragma omp atomic write
            ok = 0;
        }
    }
    if (!ok) return 0;
    if (logdet) { double ld = 0; for (int k = 0; k < P->nblk; k++) ld += S[k].part[1]; *logdet = ld; }
    return 1;
}
static void ds_inverse(const Problem *P, BS *S, int need_lx) {
    PAR_BLOCKS
    for (int k = 0; k < P->nblk; k++) {
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
        if (n <= PROD_SMALL) inv_from_chol_small(n, s->Zi);
        else { int info; BL(dpotri_)("L", &n, s->Zi, &n, &info); }
        lower_to_full(n, s->Zi);
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
    }
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
            s->lq = amalloc(sizeof(double) * (size_t)n * (LANCZOS_K + 1));
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
    }
    free(S);
    schur_pos_free(&Sc);
    free(Sc.M); free(Sc.Md); free(Sc.Mf); free(Sc.ya); free(Sc.Masm); envelope_free(Sc.E); schol_free(Sc.SC); free(Sc.e1); free(Sc.e2); free(Sc.e3);
    free(Sc.D); free(Sc.r); free(Sc.t); free(Sc.p); free(Sc.q); free(Sc.xb); free(Sc.fw);
    free(y); free(ybest); free(yt); free(a0); free(d1); free(d2); free(dy); free(dycert); free(u); free(w); free(zero); free(ax); free(rq);
    return status;
}
