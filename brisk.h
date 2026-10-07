/*
 * BRISK - a primal-dual interior-point solver for block-diagonal SDP/LP.
 *
 *   (P)  min <C,X>  s.t. <A_i,X> = b_i, i=1..m,  X in K
 *   (D)  max b'y    s.t. sum_i y_i A_i + Z = C,   Z in K
 *
 * K is a product of PSD cones and nonnegative orthants (LP blocks).
 * Input is SDPA sparse format; C = -F0, A_i = F_i, b = c.
 */
#ifndef BRISK_H
#define BRISK_H

#define BRISK_VERSION "1.3.2"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- 4.30 library interface (MATLAB/Octave MEX, see matlab/): brisk_main is the command
 * line; with a BriskResult attached (brisk_run) the solution of the problem as read is
 * returned in memory. Built with -DBRISK_LIBRARY, exit() returns to the caller instead of
 * ending the process and printing goes through brisk_print_hook. */
typedef struct {
    int status;             /* 0 optimal, 1 primal infeasible, 2 dual infeasible, 3 iteration limit,
                               4 numerical difficulties, 5 reduced accuracy, 6 time limit; -1 not solved */
    char status_str[40];
    int exit_code, iters, have_x;
    int m, nblk, *bs;       /* the problem as read: rows, blocks, block sizes (LP negative) */
    double pobj, dobj;      /* <C,X>, b'y (the SDPA value is -pobj) */
    double err[7];          /* DIMACS errors 1..6 on the data as read */
    double time;            /* total solve time, s */
    double *y, **X, **Z;    /* y (m); X, Z per block: n x n column-major (SDP) or n (LP) */
    /* 4.37: the bound mode (-bound p|d): side (0 none, 1 p, 2 d), <C,X> or b'y of the
     * certificate, its relative residual and lambda_min, valid (2 with the margin, 1 PSD, 0
     * not), certified (set by tools/certify, 0 here) */
    int bound_side, bound_valid, bound_certified;
    double bound_value, bound_resid, bound_lammin;
    double bound_rigorous;  /* 4.39 -certify: the rigorous bound (NaN when not run) */
    int n_resolves;         /* 4.37: re-solves run (method retry, fallbacks) and their time */
    double t_resolves;
    char cause[320];        /* 4.37: the likely cause when the tolerance is missed ("" otherwise) */
    int sn;                 /* a problem in SeDuMi format: its n, x (n) and z = c - A'y (n); y above */
    double *sx, *sz;
} BriskResult;
/* 4.37: a problem given in memory instead of a file: the numbers of an SDPA sparse file.
 * c has m entries; the nnz entries (mat[q], blk[q], i[q], j[q], v[q]) are the lines of the
 * file's body, 1-based as there (mat 0 = F0, 1..m = F_i; i > j is swapped; LP blocks
 * diagonal only; zero values skipped). Nothing is copied: the arrays must stay valid for the
 * whole solve (retries rebuild the problem from them). */
typedef struct {
    int m, nblk;
    const int *bs;          /* block sizes, LP blocks negative */
    const double *c;
    size_t nnz;
    const int *mat, *blk, *i, *j;
    const double *v;
} BriskData;
int  brisk_main(int argc, char **argv);
int  brisk_run(int argc, char **argv, BriskResult *res);   /* BRISK_LIBRARY: returns the exit code; < 0: aborted */
/* 4.37 BRISK_LIBRARY: solve the problem in d with the command-line options argv[1..argc-1]
 * (argv[0] is ignored; no file name): same pipeline, same result as brisk_run on the file */
int  brisk_run_data(const BriskData *d, int argc, char **argv, BriskResult *res);
void brisk_result_free(BriskResult *res);
extern int (*brisk_print_hook)(const char *s, int is_err); /* BRISK_LIBRARY: where printed output goes (is_err: stderr) */
#if defined(BRISK_LIBRARY) && !defined(BRISK_NO_IO_MACROS)
void brisk_exit(int code) __attribute__((noreturn));
int  brisk_printf(const char *fmt, ...);
int  brisk_fprintf(FILE *f, const char *fmt, ...);
#define exit(c) brisk_exit(c)
#define printf(...) brisk_printf(__VA_ARGS__)
#define fprintf(...) brisk_fprintf(__VA_ARGS__)
#endif

#define BLK_SDP 0
#define BLK_LP  1

/* Sparse symmetric matrix. Stored entries are upper-triangular (row <= col).
 * The "full" arrays hold the symmetric expansion (both triangles).
 * For LP blocks row == col == variable index. */
typedef struct {
    int nnz;
    int *row, *col;
    double *val;
    int ef;              /* entries in full expansion */
    int *fr, *fc;
    double *fv;
    int nr;              /* distinct rows of the full pattern */
    int *rows;
} SpSym;

typedef struct Block {
    int type, n;
    int ncon;            /* constraints with a nonzero in this block */
    int *con;            /* their ids, ascending */
    SpSym *A;            /* matrices, parallel to con */
    SpSym C;
    unsigned char *route;/* Schur route per constraint: 0 sparse-sparse, 1 row-product, 2 dense BLAS3 */
    double asm_cost;     /* estimated work of the block's Schur assembly by the chosen routes, in BLAS-3 flops (0 for the low-rank route) */
    int nd, ns;          /* number of dense / sparse constraints */
    int *dlist, *slist;  /* positions (into con/A) of dense and sparse constraints */
    int *dpos;           /* dense index of position t, or -1 */
    double *Ad;          /* dense copies: nd columns of length n*n */
    int *foff;           /* flattened full entries of sparse constraints (ns+1 offsets) */
    int *ffr, *ffc;
    int *ffp;            /* 5.7 (a user's patch): their flat positions ffr + ffc*n for the gather of the row products (NULL when n*n >= 2^31) */
    double *ffv;
    int *lfoff, *lfr, *lfc;  /* 4.23: flattened stored (upper) entries of sparse constraints, same order */
    double *lfv;
    int unr;             /* distinct rows in union pattern of all A in block */
    int *urows;
    int unf;             /* union pattern, full symmetric expansion */
    int *ufr, *ufc;
    /* 5.5: the live region of the row products (problem.c): indices in the order of their last
     * sparse constraint (lv_perm[new] = old, lv_rank its inverse), the number of live indices at
     * each sparse constraint, and the full entries of the sparse constraints by permuted column,
     * sorted by constraint (lv_ca: position in slist, lv_cr: permuted row, lv_ch: one more than
     * the largest row from this entry to the end of the column) */
    int live; int *lv_perm, *lv_rank, *lv_u, *lv_cp, *lv_ca, *lv_cr, *lv_ch; double *lv_cv;
    int prod_route;      /* Zi*(A'y)*R: 0 row-subset, 1 sparse-left, 2 dense */
    /* low-rank Schur route (whole block): A_t = W_t diag(sig_t) W_t' */
    int lowrank;
    int lrR;             /* total rank */
    int *lr_off;         /* ncon+1 offsets into the columns of lr_W */
    double *lr_W;        /* n x lrR */
    double *lr_sig;      /* lrR */
    /* dictionary route (whole block): A_t = D S_t D',  D = [unit columns | dense columns] */
    int dict;            /* route selected */
    int dnu, dnv;        /* number of unit / dense dictionary columns */
    int *dunit;          /* row index of each unit column */
    double *dV;          /* n x dnv dense columns */
    struct Block *vb;    /* virtual q x q block holding the S_t (q = dnu + dnv) */
    int dstruct;         /* constraints represented through dense columns */
    /* LP: coefficient lists by variable (CSC) */
    int *lp_ptr, *lp_con;
    double *lp_val;
} Block;

typedef struct {
    int m, nblk;
    double *b;           /* scaled rhs */
    double *b0;          /* original rhs */
    Block *blk;
    /* scaling: A~_i = d_i A_i, b~_i = d_i b_i / bs, C~ = C / cs */
    double *d;
    double *du;          /* residual unscaling: bs/d_i; the trace-bound row is measured relative to R (4.30) */
    double bs, cs;
    double normb1, normC1, normb2, normC2;  /* original-data norms */
    int scaled;
    /* presolve bookkeeping */
    int morig;           /* constraints in the file */
    int *orig;           /* orig[i] = file index of constraint i */
    int fr_removed;      /* constraints removed by facial reduction */
    int fr_depth;        /* longest chain of dependent reductions (4.20) */
    double tbR;          /* 4.30: rhs of the trace-bound row (excluded from the rhs norms and scaling), 0 = none */
    double obj_off;      /* 4.30: objective constant dropped by the free elimination (file objective = this problem's + obj_off);
                            the relative gap and complementarity are measured against the file's objective magnitude */
    struct Postsolve *ps;/* reduction log for mapping solutions back (postsolve.c), or NULL */
} Problem;

typedef struct {
    double tol;          /* gap and feasibility tolerance */
    int maxit;
    int verbose;
    double c_sparse;     /* cost model constants for Schur routing */
    double c_blas;
    int max_correctors;  /* Gondzio-type extra centrality correctors */
    double dense_mem;    /* bytes allowed for dense constraint copies (Schur BLAS3 route) */
    int lanczos_k;       /* max Lanczos steps for step length */
    int sigma_rule;      /* 0 SDPT3 adaptive exponent, 1 Mehrotra cubic, 2 cubic with safeguard */
    int lowrank;         /* low-rank Schur route: -1 auto, 0 off, 1 force */
    int direction;       /* 0 HKM, 1 NT (Nesterov-Todd, computed in the scaled space) */
    int hsd_pat;         /* self-dual embedding: pattern-restricted products */
    int known_feasible;  /* a solution is known to exist (fallback solves): no infeasibility exits */
    int par_blocks;      /* OpenMP over blocks: -1 auto (several small blocks), 0 off, 1 on */
    int nt_fast;         /* NT: dX = G Rt G' - W dZ W on the dZ pattern (skips two dense transforms) */
    int nt_split;        /* NT: remove primal/dual residuals by Gram projections first (Jarre-Hergenroeder) */
    int nt_ls;           /* NT+split: preconditioned CGLS steps on the scaled least-squares problem (late phase) */
    double nt_ls_gap;    /* ... used once the relative gap is below this value */
    double gamma_max;    /* step-length factor: gamma = 0.9 + (gamma_max - 0.9) * min predictor step */
    int stall_win;       /* stop after this many non-improving iterations once below 1e-6 */
    int dict;            /* dictionary (shared-vector) Schur route: -1 auto, 0 off, 1 force */
    int sparse_schur;    /* envelope (sparse) Schur complement: -1 auto, 0 off, 1 force */
    int hsd_sig;         /* sigma-search candidates (the corrector family is affine in sigma) */
    double hsd_pshrink;
    int hsd_refine, hsd_free, hsd_stall, hsd_reghint, hsd_qscale, hsd_batch;
    int hsd_corr, hsd_ctrace, hsd_ctarget;   /* embedding: correctors, trace-free shift, target (0 trial mean, 1 sigma mu) */
    double hsd_cbmin, hsd_cbmax, hsd_cacc, hsd_soltol, hsd_cfrac, hsd_fomega, hsd_pivtol, hsd_cucap, hsd_cgain;   /* corrector box and acceptance factor */
    int hsd_dir;          /* self-dual embedding: 0 HKM, 1 NT */
    int hp_kind, hp_digits;   /* 5.0: high precision: 0 off, 1 double-double, 2 quad-double, 3 variable (hp_digits) */
    double hp_tol;            /* its tolerance (0: from the precision) */
    int hp_ext;               /* interior-point method: levels of higher precision added when the precision asked for is exhausted short of the tolerance (2) */
    double hsd_beta;     /* neighbourhood floor for tau kappa and LP products */
    int chordal;         /* chordal decomposition presolve: -1 auto, 0 off, 1 force */
    int chordal_minn;    /* only blocks at least this large */
    double chordal_density;  /* ... and at most this dense */
    int chordal_maxclique;   /* give up above this clique size; merging cap */
    int chordal_sup;         /* constraint supports up to this size join the pattern as cliques (0 off) */
    int free_elim;           /* eliminate free variables (split pairs) by pivoting before the solve */
    double free_elim_fill;   /* ... unless the Markowitz count of the best pivot exceeds this */
    int chordal_form;        /* 1 moment (range-space, shared entries; default), 0 overlap equalities */
    double chordal_merge;    /* merge a child into its parent when separator >= this fraction */
    double chordal_gain;     /* accept only if sum c^3 <= gain * n^3 */
    double chordal_gain2;    /* ... and the estimated per-iteration cost drops by this factor */
    double chordal_maxextra; /* ... and overlap constraints <= maxextra * m */
    double dual_rho;     /* potential-reduction parameter: rho * n */
    int dual_giveup;
    double dual_tau, dual_tau_long;
    double dual_gamma;
    int dual_murule;
    int dual_ls;
    int dual_corr;
    int dual_mured;
    double dual_mudrop;  /* dual method: smallest factor by which mu may drop per iteration (0: none) */
    int dual_gstart;     /* dual method: Gershgorin diagonal start when the diagonal is in range */
    int dual_corr_auto;  /* dual method: fewer correctors when Z^{-1} costs as much as the Schur matrix */
    int dual_sparse_n;   /* dual method: sparse Cholesky of Z for SDP blocks at least this large (0: off) */
    int dual_verify_n;   /* dual method: blocks this large verify certificates only now and then */
    double dual_tau_ub;  /* dual method: with a bound, mu may go below gap/(rho n) down to this decrement */
    double dual_tau_nb;  /* dual method: decrement target while no bound is certified */      /* dual method: reduce mu without a certificate (old heuristic) */       /* dual method: corrector steps per Schur factorization */         /* dual method: potential line search over several step lengths */     /* dual method: 1 adaptive long step (decrement), 0 potential reduction */   /* dual method phase 1: weight of r in b'y - Gamma r */   /* dual method: Newton-decrement targets (short / long step) */     /* iterations without a certificate before falling back */
    double dual_margin;  /* back-off factor from the certificate boundary */
    int dual_cert;       /* certificate search (bisection on mu) steps */
    int dual;            /* sparse dual-scaling method (DSDP style): 0 off, 1 on */
    int hsd;             /* self-dual embedding: -1 automatic (retry on a stalled run), 0 off, 1 on */
    int no_retry;        /* internal: set on the retry to avoid recursion */
    int hsd_first;       /* auto method: the embedding first when its extra dense work is small (4.20) */
    double hsd_first_c;  /* ... i.e. when hsd_first_c * sum n^3 <= m^3/3 */
    double hsd_first_asm;   /* ... or at most this fraction of Schur factorization plus assembly (0.5; 0: the factorization alone) */
    int hsd_hoc;         /* 5.7: embedding: passes that re-evaluate the second-order term at the chosen direction (0 = off) */
    double hsd_nb;       /* 5.7: embedding: neighbourhood parameter of the long step (lambda_min(XZ) >= hsd_nb mu after the step; 0 = off) */
    int hsd_first_on;    /* internal: this run is the embedding-first attempt */
    int crossover;       /* 0 off, 1 on, -1 auto (when cheap) */
    double fr_gain;      /* facial reduction of depth >= 2 kept only if it cuts the cost per iteration this much (4) */
    int std_slow;        /* standard method: stop on slow progress when a retry follows (iterations, 0 off) */
    double sigma_min;    /* floor for the centering parameter (a zero sigma makes the corrector affine) */
    double feas_grow;    /* cap the primal step so the predicted residual grows at most this much (0 = off) */
    int dd_end;          /* high-precision (double-double) endgame on small stalled problems */
    int dd_iters;        /* its iteration limit */
    int dd_maxm;         /* only if m <= dd_maxm ... */
    double dd_maxn2;     /* ... and sum of block sizes squared <= dd_maxn2 */
    double dd_budget;    /* ... and estimated dd operations per iteration <= dd_budget */
    double dd_time;      /* wall-clock limit for the endgame (seconds) */
    int polish;          /* project the returned point onto A(X) = b, recompute Z (if it improves) */
    int polish_x;        /* polishing in the metric of X (keeps X positive definite) */
    int polish_r;        /* the restoration of the primal feasibility in the square-root metric (first) */
    double polish_mem;   /* bytes allowed for storing the best iterate */
    double pivtol;       /* squared-pivot threshold of the equilibrated Schur factor (1e-14) */
    double reg_ill;      /* regularization applied when that threshold is violated (1e-10) */
    double balance;      /* keep relative infeasibility <= balance * relative complementarity (0 = off) */
    int mixed;           /* single-precision Schur factor as PCG preconditioner: -1 auto, 0 off, 1 on */
    double mixed_frac;   /* 4.31: auto, m in [1000, 4000): float when the factorization is at least this fraction of the iteration's work (0.6) */
    int init;            /* starting point: 0 SDPT3-style, 1 least-squares + shift, 2 ... + balancing */
    double timelimit;    /* wall-clock limit in seconds for the whole solve (0 = none) */
    double t_start;      /* internal: wtime() when the solve started (for timelimit) */
    double retry_acc;    /* automatic self-dual retry only if the result is worse than this */
    double dd_factor;    /* endgame work budget: <= dd_factor x work of the double-precision solve ... */
    double dd_minwork;   /* ... or this many flop-equivalents, whichever is larger */
    int dd_done;         /* internal: the endgame already ran (not repeated in the retry) */
    double work_done;    /* internal: estimated flops spent so far (all runs) */
    int acc_level;       /* -acc: 0 low (tol 1e-6), 1 default (1e-8), 2 high (1e-10) */
    double red_acc;      /* "solved to reduced accuracy" below this (1e-6; 1e-4 at -acc low) */
    double retry_hsd1;   /* embedding-first attempt re-solved if worse than this (1e-6 = 100 tol) */
    double slow_gate;    /* standard method: slow-progress stop only below this (1e-5 = 100 retry_acc) */
    double tracebound;   /* 4.30: -1 auto (with free elimination), 0 off, > 0 the bound R */
    int fom;             /* 4.31: first-order engine: 0 off, 1 force, -1 auto (when the IPM does not fit) */
    int fom_halpern;     /* 4.31: Halpern-restarted Peaceman-Rachford (1) or plain ADMM with step 1.618 (0) */
    int fom_maxit;       /* 4.31: its iteration limit (100000) */
    double fom_tol;      /* 4.31: its stopping tolerance on the DIMACS-type residuals (max(tol, 1e-6)) */
    double fom_sigma;    /* 4.31: fixed penalty (0 = adaptive) */
    int fom_aa;          /* 4.31: Anderson acceleration memory (0 off) */
    int fom_bm;          /* 4.31: kernel B (low-rank Burer-Monteiro ALM) after the splitting: 1 on, 0 off */
    int fom_bm_rank0;    /* 4.31: kernel B starting rank cap (10) */
    int fom_bm_outer, fom_bm_inner;   /* 4.31: its limits (100 outer, 300 L-BFGS steps per outer) */
    double fom_bm_rho, fom_bm_gtol, fom_bm_negtol;   /* 4.31: initial penalty (1), inner gradient tolerance (1e-7), eigenvalue threshold for a rank increase (1e-6) */
    int fom_ssn;         /* 4.31: phase II (ALM + semismooth Newton-CG): 1 on, 0 off */
    int fom_ssn_after;   /* 4.31: ... after this many splitting iterations (300) ... */
    double fom_ssn_res;  /* 4.31: ... or once the residual is below this (1e-3) */
    double fom_ssn_rho;  /* 4.31: phase II penalty growth factor (3) */
    int fom_ssn_prec;    /* 4.31: phase II CG preconditioner: 1 Gram matrix, 0 none */
    double fom_ssn_eta;  /* 4.32: phase II CG relative tolerance cap (0.1) */
    int fom_ssn_warm;    /* 4.32: phase II CG warm start along the previous Newton direction (1) */
    double fom_ssn_sig0; /* 4.31: phase II starts with the splitting's penalty times this (10) */
    int fom_ssn_outer, fom_ssn_newton, fom_ssn_cg;   /* 4.31: phase II limits: outer (200), Newton per outer (30), CG per Newton (200) */
    double fom_aasafe;   /* 4.31: ... safeguard: reject when the residual grows by more than this factor (1) */
    int fom_sigint;      /* 4.31: penalty checked every this many iterations (20) */
    int fom_sigrule;     /* 4.31: penalty rule: 0 dead zone (ratio > 5), 1 continuous sqrt balancing */
    double fom_sigmax;   /* 4.31: largest factor of one penalty change (2) */
    double symtime;      /* 4.32: -sym auto: time cap of the search (20 s) */
    int symnodes;        /* 4.32: ... node cap (500) */
    const char *symfile; /* 4.32: -sym: "auto" (default, 4.33), NULL (none) or a file of generators (symred.c) */
    int symsign;         /* 4.33: sign symmetries first (1) */
    double fom_race, fom_race_budget;   /* 4.42: the first-order engine first at tol >= 1e-6 on few large dense blocks: share of the expected IPM time (0.2; 0 off); the budget in seconds (set by run_pipeline) */
    int returnx;         /* 4.42 (library): X of a chordal-decomposed block: -1 returned unless its dense form exceeds 30% of the memory, 0 never (measured on the cliques, fastest), 1 always */
    int symalg;          /* 4.41: *-algebra block diagonalisation (symalg.c): -1 auto (blocks up to symalgmax), 0 off, 1 every block */
    int symalgmax;       /* 4.41: ... -symalg -1: largest block size tried (1000) */
    int symsigned;       /* 4.37: -sym auto: signed permutations (a search on |data|, signs lifted over GF(2)) (1) */
    int mfipm;           /* 4.34: matrix-free interior-point method (mfipm.c): 1 on, 0 off; 5.4: 2 the hybrid (hand-off to the standard method) */
    int mf_proj; double mf_eta;      /* 5.5: 1 = exact primal projection of dX and CG to the P^-1-norm tolerance (the 4.34 scheme); 0 = residual-controlled CG, no projection; eta: the share of |Rp| CG may leave (0.1) */
    double mf_try_tl;                /* internal: the run's own time limit during that attempt (its timelimit is the attempt's) */
    int mf_try_fits;                 /* internal (5.8): the dense Schur complement fits, i.e. the standard method can follow the attempt (else the first-order engine does) */
    double mf_try_budget;            /* seconds of that first attempt (set by run_pipeline) */
    long mf_trial_total;             /* ... and the products M v it may use in all */
    int mf_try, mf_trial;            /* 5.7: the matrix-free method first on the problems it is made for (1; -mftry 0 off); inside that first attempt: the CG steps a solve may take away from the optimum */
    double mf_hand; int mf_handcg;   /* 5.4: hand-off rule of the hybrid: merit (1e-3; 1e-5 before 5.7), also the level of the iterate kept by the automatic attempt; CG steps of an iteration (600) */
    int lralm;           /* 4.40: low-rank augmented Lagrangian (lralm.c): 1 on, 0 off (default) */
    int chordal_need;    /* 4.40 (internal): a chordal re-solve; skip it when the form does not convert and the unconverted problem does not fit */
    int lr_rank, lr_rmax, lr_outer, lr_inner, lr_escape, lr_prec, lr_newton;   /* 4.40: ... starting rank (1), largest rank (32), outer (500) and inner (2000) iteration caps, rank escape (1) */
    double lr_sigma, lr_tol, lr_trace;                               /* 4.40: ... initial penalty (10), target accuracy (1e-6) */
    double mf_rho;       /* 4.34: ... bulk spread of the Jacobi-scaled W the preconditioner models by one scalar (10) */
    int mf_rmax;         /* 4.34: ... outlier eigenvectors per block (4) */
    double mf_drop;      /* 4.34: ... tangent pairs within this relative distance of the bulk scalar are dropped (0.5) */
    int mf_kmax;         /* 4.34: ... largest capacitance matrix (8000) */
    int mf_cgmax;        /* 4.34: ... CG iterations per Newton system (3000) */
    double mf_cgtol_min, mf_cgtol_max;   /* 4.34: ... CG tolerance clamp(1e-2 mu/mu0, min, max) (1e-10, 1e-3) */
    int mf_diag;         /* 4.35: base of the preconditioner: 0 the handoff's tau^2 diag(A(D x D)A*), 1 the exact diagonal of M */
    int mf_warm;         /* 4.35: corrector CG started from the predictor's solution (1) */
    int fom_ssn_stall;   /* 4.35: phase II gives up after this many outer steps without a 20% improvement (3; 0 never) */
    int fom_aadr;        /* 4.35: Anderson acceleration on the Douglas-Rachford variable (half the state; step 1) */
    double fom_single;   /* 4.37: projections in single precision while the residual is above this (1e-4; 0: never) */
    double fom_sigma0;   /* 4.35: starting penalty of the splitting phase when adaptive (1) */
    const char *fom_start_x, *fom_start_y;   /* 4.39: -fomstart-x / -fomstart-y: a starting point (the -x / -y formats) */
    const char *bound_anchor;   /* -boundanchor: a y with Z = C - A'y positive definite, for -bound d */
    double **fom_X0, *fom_y0;                /* 4.39: ... mapped to the engine's problem by run_pipeline (NULL: none) */
    double mf_cgtime;    /* 4.35: time cap of one CG solve in seconds (0: none) */
    int mf_stall;        /* 4.35: stop after this many iterations without a 20% improvement once CG hit its budget (5) */
    int mf_recycle;      /* 4.35: Ritz vectors recycled from the previous CG run for deflation (0 off) */
    int symbd;           /* 4.33: symmetry-adapted basis (block diagonalisation) after the orbit aggregation (1) */
    double symmin;       /* 4.33: -sym auto applies the reduction only when the constraints or the block algebra shrink by this factor (1.5) */
    int dualize;         /* 4.30: solve the dual form: -1 auto (smaller Schur complement), 0 off, 1 force */
    double warm_lam;     /* re-solve warm start: blend weight of the first attempt's point (0 = off) */
    double **warm_X, **warm_Z, *warm_y;   /* internal: that point (solver scaling), NULL if none */
    /* 4.37 bound mode (-bound p|d, bound.c): the side of the problem as read (0 off, 1 p, 2 d),
     * the certificate tolerance and margin; the tracker of the problem being solved */
    int bound_side;
    double bound_tol, bound_margin, bound_track_tol;
    struct BoundTrack *btrack;
} Params;

#define BOUND_NANCHOR 4
/* 4.37 bound mode: the best candidate of one side during a solve (solver.c), in the output
 * form of the problem handed to brisk_solve (unscaled; X/tau, y/tau in the embedding) */
typedef struct BoundTrack {
    int side;                /* of that problem: 1 X of (P), 2 y of (D) */
    double tol, tol_a;       /* admissible when pinf (side 1) / dinf (side 2) <= tol; anchors: <= tol_a */
    int m, nblk, disabled;
    size_t *len;
    int have, it; double score, mu, res;
    double **X; double *y;
    /* anchors: earlier best candidates, each at least 100x more central than the next
     * (a[0] the oldest); the certificate's margin comes from the most recent one that has it */
    int na; double mu_a[BOUND_NANCHOR];
    double **Xa[BOUND_NANCHOR]; double *ya[BOUND_NANCHOR];
    int offers;
    int it_imp;              /* last iteration whose candidate improved the bound by > 1e-9 relative (A3) */
} BoundTrack;
/* a certificate on the problem as read (bound.c) */
typedef struct {
    int side;                /* 1 p (X, upper bound <C,X>), 2 d (y, lower bound b'y) */
    int have, valid;         /* valid: 2 PSD with the margin, 1 PSD, 0 not */
    double value, resid, lammin, theta;
    int attempt;
    char src[48];
    int nblk, m;
    double **X; double *y;
} BoundCert;

#define ST_OPTIMAL 0
#define ST_PINFEAS 1
#define ST_DINFEAS 2
#define ST_MAXIT   3
#define ST_NUMERIC 4
#define ST_REDUCED 5
#define ST_TIME    6
typedef struct {
    int status;          /* 0 optimal, 1 primal infeas, 2 dual infeas, 3 maxit, 4 numerical, 5 inaccurate, 6 time limit */
    int iters;
    double pobj, dobj;   /* our convention: min <C,X>, max b'y (original scale) */
    double pinf, dinf, relgap, relcomp;
    int direction;       /* direction actually used (0 HKM, 1 NT) */
    int retried;         /* the self-dual embedding produced this result */
    int methods;         /* methods that ran: 1 standard, 2 self-dual embedding */
    int free_hsd;        /* 4.24: the embedding was chosen for split free pairs (the standard method stalls there) */
    double err[7];       /* DIMACS errors 1..6 */
    double t_total, t_setup, t_schur, t_chol, t_dense, t_step;
    /* 4.37 reporting: the in-solver method retry (standard <-> embedding) */
    int nretry, retry_kept, retry_hsd;   /* retries run (0/1), whether its result was kept, it was the embedding */
    double t_retry, acc_first;           /* its time; the first method's accuracy (internal) */
    int nreg, npcg;                      /* regularized Schur factorizations, PCG steps (last method) */
} Result;

int  read_sdpa(const char *fname, Problem *P);
int  problem_from_sdpa_data(const BriskData *d, Problem *P);   /* 4.37: read_sdpa from memory */
void problem_prepare(Problem *P, const Params *par);
void problem_free(Problem *P);
int  facial_reduction(Problem *P, int verbose);
int  chordal_convert(Problem *P, const Params *par, int verbose);
void problem_from_trips(Problem *P, int m, int nblk, const int *bsz, const double *b,
                        size_t nt, const int *con, const int *blk, const int *ii, const int *jj, const double *v);
void brisk_note(const char *fmt, ...);
int  brisk_threads_logged(void);         /* 5.8: the thread count the log stated last (-1: none) */   /* 5.8: a line for the summary at the end of the run (events of the solve) */
int  schur_dense_needed(const Problem *P, const Params *par);   /* 5.8: the standard method would need a dense Schur complement (solver.c) */
int  dsdp_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout);
int  dual_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout);
int  fom_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout);   /* 4.31 fom.c */
void brisk_log_threads(int verbose);   /* (solver.c) the log line "Number of threads: k", once per run */
void brisk_log_threads_reset(void);
int  brisk_threads_busy(int verbose);   /* 4.42 (solver.c): the busy-machine thread rule; returns the count to restore or -1 */
/* sparse Cholesky of the Schur complement (sparsechol.c) */
typedef struct SChol SChol;
int chol_tiny_blocked(int nr, int w, double *P, int ld, const double *d0, int d0stride, double tinypiv);
int schol_clique_offsets(const SChol *S, int nl, const int *con, size_t *pos, size_t *soff, int *sidx);
long schol_uid(const SChol *S);
void schol_set_tinypiv(SChol *S, double t);
void schol_set_dynpiv(SChol *S, double p);       /* signed factorization: wrong-sign pivots replaced by sign * p */
int schol_ndyn(const SChol *S);
int schol_ntiny(const SChol *S);
typedef struct SAdj SAdj;
SChol *schol_analyze_adj(int m, int *deg, int **nbr, size_t fillcap);
void   schol_free(SChol *S);
void   schol_zero(SChol *S);
void   schol_gather_dense(SChol *S, const double *M);
double schol_get(const SChol *S, size_t r, size_t c);
void   schol_add(SChol *S, size_t r, size_t c, double v);
size_t schol_offset(const SChol *S, size_t r, size_t c);
double *schol_values(SChol *S);
void   schol_set_signs(SChol *S, const signed char *sgn_orig);
void   schol_mv(const SChol *S, const double *x, double *y);
void   schol_mv_abs(const SChol *S, const double *x, double *y, double *ya);
int    schol_factor(SChol *S, double shift);
void   schol_release_factor(SChol *S);
size_t schol_pansz(const SChol *S);
void   schol_set_inplace(SChol *S, int on);
void   schol_solve(const SChol *S, double *b, double *work);
void   schol_solve2(const SChol *S, double *b1, double *b2, double *work);
void   schol_solve_many(const SChol *S, double *Bm, int nrhs, double *work);
double schol_cost(const SChol *S, double c_narrow, double c_wide);
void   schol_set_flopcap(double cap);
double schol_flops(const SChol *S);
double schol_logdet(const SChol *S);
size_t schol_nnz(const SChol *S);
int    schol_nsuper(const SChol *S);
size_t schol_mpat_nnz(const SChol *S);
SChol *chordal_take_analysis(int m, size_t patnnz);
/* internal (problem.c / dictroute.c) */
void   spsym_finish(SpSym *S, int n, int is_lp);
double analyze_sdp_block(Block *B, const Params *par);
void   block_free_contents(Block *B);
void   problem_clone(const Problem *P, Problem *Q);   /* 4.30: exact copy of an unprepared problem */
double dict_build(Block *B, const Params *par);
int  dd_endgame(const Problem *P, double **Xio, double **Zio, double *y, const Params *par,
                int maxit, double tol, int verbose);
void   dict_free(Block *B);
void params_default(Params *par);
void params_accuracy(Params *par, int level);
void tracebound_add(Problem *P, double R);   /* 4.30: sum tr(X_k) + s = R as the last row / block */   /* -acc: 0 low, 1 default, 2 high */
int  brisk_solve(Problem *P, const Params *par, Result *res, double *yout, double **Xout);
double wtime(void);
#include <signal.h>
extern volatile sig_atomic_t brisk_stop_flag;   /* 4.37: set by brisk_interrupt (capi.c) */
int brisk_time_up(const Params *par);           /* the time limit is reached or an interrupt came */

/* ---------------- postsolve (postsolve.c) ---------------------------------
 * Facial reduction replaces a block by X = T W T' and removes constraints whose dual
 * multipliers are then free. The log below lets the solution be mapped back to the
 * problem as read: X_orig = T X_red T', and a dual-feasible y for removed constraints
 * is recovered in reverse order (Schur-complement bound per reduction).            */
typedef struct {            /* X_orig = T X_cur T', T stored by sparse columns */
    int n_orig, n_cur;
    int identity;           /* T = I (no storage) */
    int *nz;                /* nonzeros per column */
    int **idx;
    double **val;
} PSBasis;
typedef struct {
    int kind;               /* PS_RANK1, PS_DIAG, PS_LP, PS_FACE */
    int blk;                /* original block id */
    int n_prev;             /* block size before the reduction */
    PSBasis T;              /* basis before the reduction */
    int *K, nk, *J, nj;     /* deletions: kept / deleted indices (previous coordinates) */
    int p; double *w;       /* rank one: pivot and w (length n_prev) */
    int *S, s, kn;          /* face: support, its size, null-space dimension */
    double *N, *U;          /* face: null space (s x kn) and range (s x (s - kn)) */
} PSPart;
enum { PS_RANK1 = 0, PS_DIAG = 1, PS_LP = 2, PS_FACE = 3 };
typedef struct {
    int con;                /* index of the removed constraint in the file */
    double sgn;             /* sgn * A_con is positive semidefinite */
    int np;
    PSPart *parts;
} PSRecord;
/* one chordal conversion: the cliques (in the order of the new blocks), their nodes in the
 * converted block and their clique-tree parents; pos is the index the block had */
typedef struct { int pos, n, ncl; int *cptr, *cv, *par; } PSChordal;
/* moment (range-space) form of the chordal conversion: the problem solved is the dual
 * written as a primal, with one row per moment (entry of the chordal pattern, entry of a
 * small SDP block, LP variable), the clique blocks Z_k of the dual slack, the LP slacks,
 * and the original y as free split pairs. Postsolve reads X from the multipliers of the
 * rows and y from the pairs. */
typedef struct {
    int m0, nblk0;          /* the problem before the conversion */
    int *orig0, *bs0;       /* its orig[] and block sizes (LP negative) */
    int nd;                 /* 4.34: decomposed blocks (any number), largest first: */
    int *dpos;              /*   their indices in the problem before the conversion, */
    PSChordal *dch;         /*   cliques, */
    int **dcrow, **dcoff;   /*   per clique the rows of its local entries (a <= b), packed, with offsets */
    int chpos;              /* = dpos[0] (the largest; the sparse postsolve measure takes it), -1 */
    PSChordal ch;           /* = dch[0] (a shallow copy) */
    int *crow;              /* = dcrow[0] */
    int *coff;              /* = dcoff[0] */
    int *brow0;             /* per original block: first row (SDP non-decomposed: n(n+1)/2 rows, LP: n rows), -1 */
    int tlp, off;           /* LP block of the new problem, position of y+ (y- at off + m0) */
    int nrow;
    double cost0, cost1;    /* estimated cost per iteration: unconverted, converted */
    /* inequality multipliers eliminated through their slack rows (z_s + a y_i = 0 with the
     * slack in no other constraint: y_i = -z_s / a, the row and the pair are dropped) */
    int *rowmap;            /* rows as built -> rows of the problem solved (-1: dropped), or NULL */
    double *ysc;            /* column scale of y_i (kept: y = ysc (y+ - y-); eliminated: z_s = |a| ysc zhat) */
    int *ymap, np;          /* y_i -> kept pair index (y+ at off + ymap, y- at off + np + ymap), -1 */
    int ne;                 /* eliminated multipliers: i, slack s, pivot a, dropped row, and the */
    int *e_i, *e_s, *e_row; /* column of y_i in the other rows (solved numbering) for the slack's */
    double *e_a;            /* value X_s = (b_i + sum_r A_ir w_r) / a                          */
    int *e_ptr, *e_r; double *e_v, *e_b;
} PSMoment;
/* free-variable elimination (freeelim.c): one step per eliminated split pair */
typedef struct {
    int blk, ip, im;        /* LP block and the pair's slots */
    int row;                /* pivot row (file numbering) */
    double a, b, c;         /* pivot, rhs of the pivot row, cost of the free variable (at elimination) */
    int nr; uint64_t *rk; double *rv;   /* the pivot row without f: entry keys and values */
    int nc; int *ce; double *cv;        /* the column of f without the pivot: rows (file numbering), values */
} PSFreeStep;
typedef struct {
    int mfile, mnew, ns;
    int *row2file;          /* rows of the reduced problem -> rows of the file */
    PSFreeStep *st;
    int np, *pb, *pp, *pm;  /* all detected pairs (eliminated or kept) */
    double offset;          /* objective constant of the reduced problem */
    int cheap;              /* 4.30: only fill-free eliminations were made (outside the SOS class): no trace bound, no fallback */
    int nblk, *lpoff, *lpmap;   /* 4.30: LP slots dropped from the reduced problem: file slot lpoff[k] + l -> reduced slot or -1 (lpmap NULL: none dropped) */
} PSFreeElim;
typedef struct Postsolve {
    int nblk_orig;
    int *type, *norig;      /* original block types and sizes */
    PSBasis *basis;         /* current basis per original block */
    int *cur2orig;          /* current block -> original block */
    void *blk0;             /* P->blk during presolve (to identify blocks) */
    int nrec, rcap;
    PSRecord *rec;
    double snap_bytes, snap_cap;
    int dual_ok;            /* 0 if snapshots were dropped (memory cap) */
    int nblk_after;         /* blocks when facial reduction ended */
    int chordal;            /* chordal conversion changed the blocks afterwards */
    int nch;                /* chordal conversions, in the order they were made */
    PSMoment *mom;          /* moment-form conversion, or NULL */
    PSChordal *ch;
} Postsolve;
/* compact copy of the data as read, for measuring the returned solution */
typedef struct {
    int m, nblk;
    int *bs;                /* block sizes, LP negative */
    double *b;
    size_t nnz;
    int *con, *blk, *ii, *jj;   /* con = -1 for C; ii <= jj */
    double *v;
    double nb1, nC1;
    double tbR;          /* 4.30: rhs of the trace-bound row (0 = none) */
    double obj_off;      /* 4.30: objective constant of the problem this describes (see Problem) */
} PSOrig;
double *chordal_complete(const PSChordal *r, double *const *Xcl);
void ps_moment_free(PSMoment *M);
typedef struct { int blk, ip, im; } FreePair;    /* 4.30: a split pair: LP block, + slot, - slot */
int free_pairs_detect(const Problem *P, FreePair **out);
PSFreeElim *free_eliminate(Problem *P, int verbose, double maxfill, int automatic);
/* 4.30 dual form (dualize.c): the problem is replaced by its dual written as a primal,
 * Z + sum_i y_i A_i = C entrywise with y as split free pairs, when that has the smaller Schur
 * complement after free elimination (the image form of an SOS program given in kernel form
 * and the other way round). The record maps the solution back to the problem as read. */
typedef struct {
    int m, nblk, mp;        /* rows and blocks as read; rows of the dual form */
    int *bs;                /* block sizes as read */
    int *sdpmap;            /* block as read -> block of the dual form (SDP), -1 for LP */
    int *rowbase;           /* SDP block as read -> first row of its entries (i <= j, column-major upper) */
    int *lpbase, *lpslot;   /* LP block as read -> base into lpslot; lpslot[base + l] = row of slot l (>= 0), or -1 for a pair slot */
    int zblk, nz;           /* dual-form LP block of the nonneg variables (or -1), its size */
    int yblk;               /* dual-form LP block of the y pairs: y_i = x[i] - x[m + i] */
    int np; FreePair *pr;   /* pairs as read; pair p has row prow[p] */
    int *prow;
} PSDual;
PSDual *dualize_apply(Problem *P, int automatic, int verbose);   /* NULL: not applied */
void dualize_unmap(const PSDual *D, double **Xp, const double *yp, double ***Xo, double *yo);
void dualize_free(PSDual *D);
void free_elim_refine(const PSFreeElim *F, const PSOrig *O, double **Xo, int verbose);   /* 4.30 */
void free_elim_free(PSFreeElim *F);
void free_elim_post(const PSFreeElim *F, const PSOrig *Ofile, double **Xo, const double *yred, double *yfile);
PSOrig *ps_orig_build(const Problem *P);
/* 4.32 symmetry reduction (symred.c): generators of the automorphism group -> constraint orbits aggregated */
typedef struct { int k, t, d, n, lp, p, rblk; double *W; } SymBlk;   /* rblk: block of the reduced problem; lp: a 1 x 1 piece kept as LP variable p of block lpblk */   /* 4.33: reduced block of type t (-1: LP) of file block k: d copies (the dimension of the irreducible), size n; W: d matrices n_k x n */
typedef struct { int m, mr, order, NI, nblk; int *cls; int **G; int *off, *gblk; double *b0; int ngen; int **gen;
                 int bd, nbr, nsb, lpblk; int *bsr; SymBlk *sb; int *brep, **bmap;
                 signed char **gsg;   /* 4.37: signed generators: index signs (NULL: a permutation) */
                 signed char *csg;    /* 4.37: sign of each file constraint relative to its orbit's aggregate (cls = -1: dropped) */
                 signed char **bsgn;  /* 4.37: signs of the exchanged blocks' maps (NULL: none) */ } PSSym;
/* the Gram matrix A A* of the (scaled) problem, factored: sparse (supernodal Cholesky) or
 * dense (fom.c; also used by the matrix-free IPM for its exact primal projection) */
typedef struct { int m, dense; double *M; SChol *S; double *work; double reg; } Gram;
Gram *fom_gram_build(const Problem *P, int verbose, double *t_gram);
double brisk_mem_limit(void);
double brisk_mem_scale(void);   /* max(1, memory / 8 GB): the factor of the caps sized on the 8 GB machine */
/* 5.0 (hpsolve.c): high-precision solve of an SDPA file; kind 1 dd, 2 qd, 3 variable precision with `digits` */
void brisk_hp_clear(void);
int brisk_hp_certify_given(const char *fname, const BriskData *data, BriskResult *res, const char *cx, const char *cy, int kind, int digits, int verbose);   /* 5.2 */
int brisk_hp_run(const char *fname, const BriskData *data, BriskResult *res, const BriskResult *warm, const Params *par, int kind, int digits, double tol, const char *yfile, const char *xfile, const char *zfile);
int bound_certify(const PSOrig *O, int side, double **X, const double *y, int na, double ***Xa, double **ya,
                  double margin, double tol, const char *src, int verbose, BoundCert *out);
int bound_better(const BoundCert *a, const BoundCert *b);
void bound_cert_free(BoundCert *c);
void bound_gram_drop(const PSOrig *O);   /* 4.42: drops the cached Gram factor (of O, or any with NULL) */
int bound_find_pairs(const PSOrig *O, int **P1, int **P2, int **PK);
/* 4.39 (boundcert.c): rigorous certification on the problem as read; side 1 X, 2 y; exact: the
 * data are exact doubles (in memory) rather than decimals read from a file. Returns 1 when
 * certified; *bound is the rigorous upper (p) / lower (d) bound (also when not certified) */
int brisk_certify_orig(const PSOrig *O, int side, double **X, const double *y, int exact,
                       double *bound, char *reason, size_t rlen);
double brisk_gram_lammin_lower(const PSOrig *O, int exact);                                   /* 5.2: for hpsolve.c */
double brisk_pairs_lammin_lower(const PSOrig *O, int exact, int np, const int *PK, const int *P1);
void bound_gram_apply(const PSOrig *O, const double *r, double *w);
double **bound_alloc_blocks(const PSOrig *O);
void bound_free_blocks(double **X, int nblk);    /* 4.37 (main.c): RAM or the cgroup limit when lower */
void  fom_gram_solve(const Gram *G, double *x);
void  fom_gram_free(Gram *G);
/* 4.34 matrix-free interior-point method (mfipm.c) */
int mfipm_solve(Problem *P, const Params *par, Result *R, double *yout, double **Xout);
/* 5.4: the hybrid (-mfipm 2): the matrix-free iteration to a merit of mf_hand or until a solve
 * needs more than mf_handcg CG steps, then the standard method from its iterate (warm start).
 * The internal (scaled) iterate is returned in Xint, Zint (per block, allocated by the caller)
 * and yint; mfipm_solve stops on the hand-off rule when they are given. Returns 1 at a hand-off,
 * 2 when CG gave up with the merit above 1e-2 (the point is no use as a start), 0 when the
 * matrix-free method ended by itself, < 0 on an error. */
int mfipm_solve_snap(Problem *P, const Params *par, Result *R, double *yout, double **Xout, double **Xint, double **Zint, double *yint, double merit, int *taken);
int mfipm_solve_hand(Problem *P, const Params *par, Result *R, double **Xint, double **Zint, double *yint, double hand_merit, int hand_cg);
/* 4.40 low-rank augmented Lagrangian (lralm.c): reads, solves, reports; returns the exit code */
int lralm_run(const char *fname, Params *par, const char *yfile, const BriskData *data, BriskResult *res);
extern int g_read_maxn;   /* 4.40: largest SDP block the reader accepts (46340; lralm lifts it) */
PSSym *sym_reduce(Problem *P, const char *fname, int verbose, double symtime, long symnodes, int symbd, double symmin, int chordal_minn, double chordal_density, int symsigned);
/* 4.33 sign symmetries (symred.c): diagonal ±1 automorphisms; blocks split by sign pattern, odd constraints dropped */
typedef struct { int m, mk, nblk, ncls, ngen, NI, nbr; char *keep; int *cls, *cpos, *csz, *off, *bs, *qblk, *qpos0; } PSSign;   /* qblk: class -> reduced block; qpos0: offset of a size-1 class in the LP block */
PSSign *sign_reduce(Problem *P, int verbose, int auto_mode, double symmin, int chordal_minn, double chordal_density);
void sign_unmap(const PSSign *S, double **Xred, const double *yred, double **Xo, double *yo);
void sign_free(PSSign *S);
/* 4.41 (symalg.c): numerical block diagonalisation of the *-algebra of each SDP block (any
 * symmetry group, also non-permutation); component g of file block ck[g]: size cd[g],
 * multiplicity cm[g], basis cQ[g] (n x cd cm), reduced block crb[g] (-1 dropped; lpblk: an LP
 * variable at cpos[g]); bmap[k]: reduced block of a block kept as it is (-1: decomposed) */
typedef struct { int m, nblk, nbr, ncomp, lpblk; int *bs, *bmap, *ck, *cd, *cm, *crb, *cpos; double **cQ; } PSAlg;
PSAlg *alg_reduce(Problem *P, int mode, int maxn, double symmin, int verbose);
void alg_unmap(const PSAlg *S, double **Xred, const double *yred, double **Xo, double *yo);
void alg_free(PSAlg *S);
void sym_unmap(const PSSym *S, double **Xred, const double *yred, double **Xo, double *yo, const int *bs);
void sym_free(PSSym *S);
/* crossover.c: Newton on the rank-revealed KKT system; returns 1 if the result was kept */
int sdp_crossover(const PSOrig *O, double **Xo, double *yo, int verbose, double max_seconds,
                  double max_unknowns, double *err_before, double *err_after, double *pd);
void    ps_orig_free(PSOrig *O);
Postsolve *ps_new(const Problem *P);
void    ps_free(Postsolve *ps);
/* hooks called by presolve.c */
void    ps_hook_delete(Problem *P, const Block *B, const char *del);
void    ps_hook_rank1(Problem *P, const Block *B, const double *w, int p);
void    ps_hook_face(Problem *P, const Block *B, const int *S, int s, const double *N, int k, const char *inS);
void    ps_hook_vanish(Problem *P, const Block *B);
PSRecord *ps_record_begin(Problem *P, int con_orig, double sgn);
void    ps_record_part(Problem *P, PSRecord *r, const Block *B, int kind, const char *del,
                       const double *w, int p, const int *S, int s, const double *N, int kn, const double *U);
typedef struct {
    double err[7];          /* DIMACS errors 1..6 in the original problem (Z = C - A'y) */
    double pobj, dobj;      /* <C,X>, b'y */
    int have_x;             /* X could be mapped back (no chordal conversion) */
    int recovered;          /* removed constraints whose y was recovered */
    int unrecovered;        /* ... and those where recovery was not possible */
    double t;               /* seconds */
} PSResult;
int  ps_measure(const PSOrig *O, double **Xo, const double *yo, PSResult *res);   /* 4.30: errors of (Xo, yo) on O */
int  ps_infeas_check(const PSOrig *O, double **Xo, const double *yo, int status, double *obj, double *viol);   /* 5.8: is the point a certificate of the infeasibility verdict (1), not (0), none (-1) */
/* Map (X_red, y_red) back, recover y of removed constraints, measure everything on O.
 * Xo[k]: n_k x n_k (SDP) or n_k (LP) arrays for the original blocks, allocated here
 * (NULL if unavailable); yo: length O->m.                                          */
int     postsolve(const Problem *P, const PSOrig *O, double **Xred, const double *yred,
                  double ***Xo_out, double *yo, PSResult *res, int verbose);
double  ps_lammin(int n, const double *M, double *work);
double  ps_lammin_tol(int n, const double *M, double *work, double delta);

/* ---------------- BLAS / LAPACK (Fortran interface, 32-bit ints) ----------- */
#ifndef BLAS_PREFIX
#define BLAS_PREFIX
#endif
#define BR_CAT2(a, b) a##b
#define BR_CAT(a, b)  BR_CAT2(a, b)
#define BL(name) BR_CAT(BLAS_PREFIX, name)
/* 4.25: a debug flag read from the environment once per call site (getenv was called
 * per block per iteration: 210,000 times on 30,000 blocks) */
#define ENV_ON(name) ({ static int env_c_ = -1; if (env_c_ < 0) env_c_ = getenv(name) != NULL; env_c_; })

void   BL(dgemm_)(const char *, const char *, const int *, const int *, const int *,
                  const double *, const double *, const int *, const double *, const int *,
                  const double *, double *, const int *);
void   BL(dsymm_)(const char *, const char *, const int *, const int *, const double *,
                  const double *, const int *, const double *, const int *,
                  const double *, double *, const int *);
void   BL(dgemv_)(const char *, const int *, const int *, const double *, const double *,
                  const int *, const double *, const int *, const double *, double *, const int *);
void   BL(ssyevr_)(const char *, const char *, const char *, const int *, float *, const int *, const float *, const float *,
                   const int *, const int *, const float *, int *, float *, float *, const int *, int *, float *, const int *, int *, const int *, int *);
void   BL(dsterf_)(const int *, double *, double *, int *);
void   BL(dsymv_)(const char *, const int *, const double *, const double *, const int *,
                  const double *, const int *, const double *, double *, const int *);
void   BL(dtrsv_)(const char *, const char *, const char *, const int *, const double *,
                  const int *, double *, const int *);
void   BL(strsv_)(const char *, const char *, const char *, const int *, const float *,
                  const int *, float *, const int *);
void   BL(dtrmv_)(const char *, const char *, const char *, const int *, const double *,
                  const int *, double *, const int *);
double BL(ddot_)(const int *, const double *, const int *, const double *, const int *);
void   BL(daxpy_)(const int *, const double *, const double *, const int *, double *, const int *);
double BL(dnrm2_)(const int *, const double *, const int *);
void   BL(dpotrf_)(const char *, const int *, double *, const int *, int *);
void   BL(dpotri_)(const char *, const int *, double *, const int *, int *);
void   BL(dpotrs_)(const char *, const int *, const int *, const double *, const int *,
                   double *, const int *, int *);
void   BL(dsyev_)(const char *, const char *, const int *, double *, const int *, double *,
                  double *, const int *, int *);
void   BL(dsyevd_)(const char *, const char *, const int *, double *, const int *, double *,
                   double *, const int *, int *, const int *, int *);
void   BL(dtrsm_)(const char *, const char *, const char *, const char *, const int *, const int *,
                  const double *, const double *, const int *, double *, const int *);
void   BL(dtrmm_)(const char *, const char *, const char *, const char *, const int *, const int *,
                  const double *, const double *, const int *, double *, const int *);
void   BL(spotrf_)(const char *, const int *, float *, const int *, int *);
void   BL(ssyevd_)(const char *, const char *, const int *, float *, const int *, float *, float *, const int *, int *, const int *, int *);   /* 4.37: the first-order engine's single-precision projections */
void   BL(ssytrd_)(const char *, const int *, float *, const int *, float *, float *, float *, float *, const int *, int *);   /* 4.39: split symmetric eigensolver */
void   BL(sstedc_)(const char *, const int *, float *, float *, float *, const int *, float *, const int *, int *, const int *, int *);
void   BL(sormtr_)(const char *, const char *, const char *, const int *, const int *, const float *, const int *, const float *, float *, const int *, float *, const int *, int *);
void   BL(dstedc_)(const char *, const int *, double *, double *, double *, const int *, double *, const int *, int *, const int *, int *);
void   BL(spotrs_)(const char *, const int *, const int *, const float *, const int *,
                   float *, const int *, int *);
void   BL(dsyrk_)(const char *, const char *, const int *, const int *, const double *,
                  const double *, const int *, const double *, double *, const int *);
void   BL(dgesdd_)(const char *, const int *, const int *, double *, const int *, double *,
                   double *, const int *, double *, const int *, double *, const int *, int *, int *);
void   BL(dsyevr_)(const char *, const char *, const char *, const int *, double *, const int *,
                   const double *, const double *, const int *, const int *, const double *, int *,
                   double *, double *, const int *, int *, double *, const int *, int *, const int *, int *);
void   BL(dsytrd_)(const char *, const int *, double *, const int *, double *, double *, double *,
                   double *, const int *, int *);
void   BL(dstebz_)(const char *, const char *, const int *, const double *, const double *, const int *,
                   const int *, const double *, const double *, const double *, int *, int *, double *,
                   int *, int *, double *, int *, int *);
void   BL(dstein_)(const int *, const double *, const double *, const int *, const double *, const int *,
                   const int *, double *, const int *, double *, int *, int *, int *);
void   BL(dormtr_)(const char *, const char *, const char *, const int *, const int *, const double *,
                   const int *, const double *, double *, const int *, double *, const int *, int *);
void   BL(dsytrf_)(const char *, const int *, double *, const int *, int *, double *, const int *, int *);
void   BL(dsytrs_)(const char *, const int *, const int *, const double *, const int *, const int *,
                   double *, const int *, int *);
void   BL(dstev_)(const char *, const int *, double *, double *, double *, const int *,
                  double *, int *);


/* dsymm for the small blocks (uplo "L" only): the symmetric factor expanded to full
 * storage and a plain column-axpy product the compiler vectorizes. OpenBLAS dsymm packs
 * both operands and allocates a buffer per call, which dominated on 1000+ cliques. */
/* 4.22: triangular solves / products with small lower-triangular factors (the per-block
 * work of the embedding on thousands of blocks of size <= 64). OpenBLAS's dtrsm costs
 * 13 us at n = 32 against 1.1 us for a dgemm of the same size; the recursive split below
 * puts the flops into dgemm and leaves 8x8 triangles to plain loops. L lower, non-unit.
 * g_xtri = 0 falls back to the BLAS calls (BRISK_NOXTRI). */
extern int g_xtri;
int brisk_dpotrf(int n, double *A, int lda);
void pdgemm(const char *ta, const char *tb, const int *m, const int *n, const int *k, const double *alpha,
            const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
void pdsymm(const char *side, const char *uplo, const int *m, const int *n, const double *alpha,
            const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
int brisk_spotrf(int n, float *A, int lda);
void brisk_threads_init(void);
#define XTRI_BASE 8
/* B (m x nb, ld ldb) := L^{-1} B, L m x m */
static void xtri_lln(int m, int nb, const double *L, int ldl, double *B, int ldb) {
    if (m <= XTRI_BASE) {
        for (int j = 0; j < nb; j++) {
            double *b = B + (size_t)j * ldb;
            for (int k = 0; k < m; k++) {
                const double bk = b[k] / L[k + (size_t)k * ldl];
                b[k] = bk;
                const double *l = L + (size_t)k * ldl;
                for (int i = k + 1; i < m; i++) b[i] -= bk * l[i];
            }
        }
        return;
    }
    const int m1 = m / 2, m2 = m - m1;
    const double mone = -1.0, one = 1.0;
    xtri_lln(m1, nb, L, ldl, B, ldb);
    BL(dgemm_)("N", "N", &m2, &nb, &m1, &mone, L + m1, &ldl, B, &ldb, &one, B + m1, &ldb);
    xtri_lln(m2, nb, L + m1 + (size_t)m1 * ldl, ldl, B + m1, ldb);
}
/* B := L^{-T} B */
static void xtri_llt(int m, int nb, const double *L, int ldl, double *B, int ldb) {
    if (m <= XTRI_BASE) {
        for (int j = 0; j < nb; j++) {
            double *b = B + (size_t)j * ldb;
            for (int k = m - 1; k >= 0; k--) {
                const double *l = L + (size_t)k * ldl;
                double v = b[k];
                for (int i = k + 1; i < m; i++) v -= l[i] * b[i];
                b[k] = v / l[k];
            }
        }
        return;
    }
    const int m1 = m / 2, m2 = m - m1;
    const double mone = -1.0, one = 1.0;
    xtri_llt(m2, nb, L + m1 + (size_t)m1 * ldl, ldl, B + m1, ldb);
    BL(dgemm_)("T", "N", &m1, &nb, &m2, &mone, L + m1, &ldl, B + m1, &ldb, &one, B, &ldb);
    xtri_llt(m1, nb, L, ldl, B, ldb);
}
/* B (mb x n) := B L^{-T} */
static void xtri_rlt(int mb, int n, const double *L, int ldl, double *B, int ldb) {
    if (n <= XTRI_BASE) {
        for (int k = 0; k < n; k++) {
            double *bk = B + (size_t)k * ldb;
            const double d = 1.0 / L[k + (size_t)k * ldl];
            for (int q = 0; q < k; q++) {
                const double lkq = L[k + (size_t)q * ldl];
                if (lkq == 0) continue;
                const double *bq = B + (size_t)q * ldb;
                for (int i = 0; i < mb; i++) bk[i] -= lkq * bq[i];
            }
            for (int i = 0; i < mb; i++) bk[i] *= d;
        }
        return;
    }
    const int n1 = n / 2, n2 = n - n1;
    const double mone = -1.0, one = 1.0;
    xtri_rlt(mb, n1, L, ldl, B, ldb);
    BL(dgemm_)("N", "T", &mb, &n2, &n1, &mone, B, &ldb, L + n1, &ldl, &one, B + (size_t)n1 * ldb, &ldb);
    xtri_rlt(mb, n2, L + n1 + (size_t)n1 * ldl, ldl, B + (size_t)n1 * ldb, ldb);
}
/* B := B L^{-1} */
static void xtri_rln(int mb, int n, const double *L, int ldl, double *B, int ldb) {
    if (n <= XTRI_BASE) {
        for (int k = n - 1; k >= 0; k--) {
            double *bk = B + (size_t)k * ldb;
            for (int q = k + 1; q < n; q++) {
                const double lqk = L[q + (size_t)k * ldl];
                if (lqk == 0) continue;
                const double *bq = B + (size_t)q * ldb;
                for (int i = 0; i < mb; i++) bk[i] -= lqk * bq[i];
            }
            const double d = 1.0 / L[k + (size_t)k * ldl];
            for (int i = 0; i < mb; i++) bk[i] *= d;
        }
        return;
    }
    const int n1 = n / 2, n2 = n - n1;
    const double mone = -1.0, one = 1.0;
    xtri_rln(mb, n2, L + n1 + (size_t)n1 * ldl, ldl, B + (size_t)n1 * ldb, ldb);
    BL(dgemm_)("N", "N", &mb, &n1, &n2, &mone, B + (size_t)n1 * ldb, &ldb, L + n1, &ldl, &one, B, &ldb);
    xtri_rln(mb, n1, L, ldl, B, ldb);
}
/* products: B := L B, B := L' B, B := B L */
static void xtrm_lln(int m, int nb, const double *L, int ldl, double *B, int ldb) {
    if (m <= XTRI_BASE) {
        for (int j = 0; j < nb; j++) {
            double *b = B + (size_t)j * ldb;
            for (int i = m - 1; i >= 0; i--) {
                double v = 0;
                for (int k = 0; k <= i; k++) v += L[i + (size_t)k * ldl] * b[k];
                b[i] = v;
            }
        }
        return;
    }
    const int m1 = m / 2, m2 = m - m1;
    const double one = 1.0;
    xtrm_lln(m2, nb, L + m1 + (size_t)m1 * ldl, ldl, B + m1, ldb);
    BL(dgemm_)("N", "N", &m2, &nb, &m1, &one, L + m1, &ldl, B, &ldb, &one, B + m1, &ldb);
    xtrm_lln(m1, nb, L, ldl, B, ldb);
}
static void xtrm_llt(int m, int nb, const double *L, int ldl, double *B, int ldb) {
    if (m <= XTRI_BASE) {
        for (int j = 0; j < nb; j++) {
            double *b = B + (size_t)j * ldb;
            for (int i = 0; i < m; i++) {
                const double *l = L + (size_t)i * ldl;
                double v = 0;
                for (int k = i; k < m; k++) v += l[k] * b[k];
                b[i] = v;
            }
        }
        return;
    }
    const int m1 = m / 2, m2 = m - m1;
    const double one = 1.0;
    xtrm_llt(m1, nb, L, ldl, B, ldb);
    BL(dgemm_)("T", "N", &m1, &nb, &m2, &one, L + m1, &ldl, B + m1, &ldb, &one, B, &ldb);
    xtrm_llt(m2, nb, L + m1 + (size_t)m1 * ldl, ldl, B + m1, ldb);
}
static void xtrm_rln(int mb, int n, const double *L, int ldl, double *B, int ldb) {
    if (n <= XTRI_BASE) {
        for (int k = 0; k < n; k++) {
            double *bk = B + (size_t)k * ldb;
            const double d = L[k + (size_t)k * ldl];
            for (int i = 0; i < mb; i++) bk[i] *= d;
            for (int q = k + 1; q < n; q++) {
                const double lqk = L[q + (size_t)k * ldl];
                if (lqk == 0) continue;
                const double *bq = B + (size_t)q * ldb;
                for (int i = 0; i < mb; i++) bk[i] += lqk * bq[i];
            }
        }
        return;
    }
    const int n1 = n / 2, n2 = n - n1;
    const double one = 1.0;
    xtrm_rln(mb, n1, L, ldl, B, ldb);
    BL(dgemm_)("N", "N", &mb, &n1, &n2, &one, B + (size_t)n1 * ldb, &ldb, L + n1, &ldl, &one, B, &ldb);
    xtrm_rln(mb, n2, L + n1 + (size_t)n1 * ldl, ldl, B + (size_t)n1 * ldb, ldb);
}
static inline void xscale(int m, int n, double a, double *B, int ldb) {
    if (a == 1.0) return;
    for (int j = 0; j < n; j++) { double *b = B + (size_t)j * ldb; for (int i = 0; i < m; i++) b[i] *= a; }
}
static inline void xtrsm(const char *side, const char *uplo, const char *ta, const char *diag, const int *m_, const int *n_,
                         const double *alpha, const double *L, const int *ldl_, double *B, const int *ldb_) {
    const int m = *m_, n = *n_, left = side[0] == 'L', tr = ta[0] == 'T';
    const int k = left ? m : n;
    if (!g_xtri || k > 64 || uplo[0] != 'L' || diag[0] != 'N' || k <= 0 || (left ? n : m) <= 0) {
        BL(dtrsm_)(side, uplo, ta, diag, m_, n_, alpha, L, ldl_, B, ldb_);
        return;
    }
    xscale(m, n, *alpha, B, *ldb_);
    if (left) { if (tr) xtri_llt(m, n, L, *ldl_, B, *ldb_); else xtri_lln(m, n, L, *ldl_, B, *ldb_); }
    else { if (tr) xtri_rlt(m, n, L, *ldl_, B, *ldb_); else xtri_rln(m, n, L, *ldl_, B, *ldb_); }
}
static inline void xtrmm(const char *side, const char *uplo, const char *ta, const char *diag, const int *m_, const int *n_,
                         const double *alpha, const double *L, const int *ldl_, double *B, const int *ldb_) {
    const int m = *m_, n = *n_, left = side[0] == 'L', tr = ta[0] == 'T';
    const int k = left ? m : n;
    if (!g_xtri || k > 64 || uplo[0] != 'L' || diag[0] != 'N' || k <= 0 || (left ? n : m) <= 0 || (!left && tr)) {
        BL(dtrmm_)(side, uplo, ta, diag, m_, n_, alpha, L, ldl_, B, ldb_);
        return;
    }
    if (left) { if (tr) xtrm_llt(m, n, L, *ldl_, B, *ldb_); else xtrm_lln(m, n, L, *ldl_, B, *ldb_); }
    else xtrm_rln(m, n, L, *ldl_, B, *ldb_);
    xscale(m, n, *alpha, B, *ldb_);
}
extern int g_xsymm_loops;
static inline void xsymm(const char *side, const char *uplo, const int *m_, const int *n_, const double *alpha_,
                         const double *A, const int *lda_, const double *B, const int *ldb_, const double *beta_,
                         double *C, const int *ldc_) {
    const int m = *m_, n = *n_, lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const int left = (side[0] == 'L' || side[0] == 'l'), k = left ? m : n;
    if (k > 32 || (uplo[0] != 'L' && uplo[0] != 'l') || (size_t)m * n > 4096) {
        pdsymm(side, uplo, m_, n_, alpha_, A, lda_, B, ldb_, beta_, C, ldc_);
        return;
    }
    const double alpha = *alpha_, beta = *beta_;
    double F[32 * 32];
    for (int j = 0; j < k; j++) {
        for (int i = j; i < k; i++) { const double v = A[i + (size_t)j * lda]; F[i + j * k] = v; F[j + i * k] = v; }
    }
    /* 4.23: the full copy through dgemm (OpenBLAS small-matrix kernels): 3-6x faster than
     * the loops below at n = 2..32 (prod3 at n = 3: 0.05 against 0.14 us) */
    if (!g_xsymm_loops) {
        if (left) BL(dgemm_)("N", "N", m_, n_, m_, alpha_, F, &k, B, ldb_, beta_, C, ldc_);
        else      BL(dgemm_)("N", "N", m_, n_, n_, alpha_, B, ldb_, F, &k, beta_, C, ldc_);
        return;
    }
    for (int j = 0; j < n; j++) {
        double *c = C + (size_t)j * ldc;
        if (beta == 0) for (int i = 0; i < m; i++) c[i] = 0;
        else if (beta != 1) for (int i = 0; i < m; i++) c[i] *= beta;
        if (left) {
            const double *b = B + (size_t)j * ldb;
            for (int q = 0; q < m; q++) {
                const double t = alpha * b[q];
                const double *f = F + q * k;
                for (int i = 0; i < m; i++) c[i] += f[i] * t;
            }
        } else {
            const double *f = F + j * k;
            for (int q = 0; q < n; q++) {
                const double t = alpha * f[q];
                const double *b = B + (size_t)q * ldb;
                for (int i = 0; i < m; i++) c[i] += b[i] * t;
            }
        }
    }
}

#endif
