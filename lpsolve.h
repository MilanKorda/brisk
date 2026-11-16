/* lpsolve.h - the interior-point method for linear programs in bounded standard form (lpsolve.c):
 *     min c'x,   A x = b,   x_j free (j < nfree),   0 <= x_j <= u_j (the others; u_j >= 1e30: none) */
#ifndef BRISK_LPSOLVE_H
#define BRISK_LPSOLVE_H
#ifndef BRISK_FLAG_FNS
#define BRISK_FLAG_FNS
/* (10 November 2026, the outside LP benchmark's audit) the flags one thread sets and another reads (the LP race's stop flags,
 * a job's end and result): atomic loads (acquire) and stores (release) on the plain int the flag is, so that no access races
 * and a job's result, written before its flag, is seen by the thread that reads the flag (the volatile qualifier alone gave
 * neither). The flags keep their type (volatile int), so that the modules' interfaces do not change */
static inline int brisk_flag_get(const volatile int *p) { return p ? __atomic_load_n(p, __ATOMIC_ACQUIRE) : 0; }
static inline void brisk_flag_set(volatile int *p, int v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#endif
typedef struct {
    double tol;                 /* relative accuracy (residuals and gap); <= 0: 1e-8 */
    int maxit;                  /* <= 0: 200 */
    int verbose;                /* 0 silent, 1 header, 2 iterations */
    double timelimit;           /* seconds; <= 0: none */
    const volatile int *stop;   /* set to nonzero from outside to stop (may be NULL) */
    int kcorr;                  /* centrality correctors per iteration; < 0: chosen from the costs */
    double objoff;              /* (22 October 2026) the problem's objective is c'x + objoff (the constant the standard form
                                   dropped, from columns shifted to their bounds and the presolve): the gap is measured
                                   against |c'x + objoff|, the objective as given, as Gurobi and MOSEK measure it */
    double dthr;                /* (22 October 2026) the dense columns: 0 chosen by cost, > 0 the columns longer than
                                   this, < 0 the default rule (the dual form passes the choice its cost model made) */
    int nopdreg;                /* (22 October 2026) 1: no primal-dual regularization at a breakdown (the crossover's
                                   solve: a regularized end point cost dfl001's crossover 17,075 iterations against 3) */
    int (*analyzed)(void *ctx, double flops);   /* (23 October 2026) called once after the analysis of the normal
                                   equations with an iteration's cost in flops (lpipm_iter_flops' measure); a nonzero
                                   return stops the method at once (status 6, why "declined") */
    void *analyzed_ctx;
    void (*iter_cb)(void *ctx, int it, double pobj, double err);   /* (4 November 2026) if set: called at each iteration with
                                   its number, the primal objective c'x + objoff and the largest relative error (the race's log) */
    void *iter_ctx;
    volatile int *fact_ms;      /* (10 November 2026) if set: the wall time of the method's last factorization in milliseconds (0: none yet; the race's idle
                                   priority for the dual simplex job, main.c: not before the job has run that long) */
    volatile int *fact_heavy;   /* (10 November 2026) if set: nonzero while the normal equations' factorization costs more than
                                   2,000 products with A Theta A' (the rule of the early iterations without a factorization), 2 when it
                                   also costs at least 1e10 flops, else 0 (the race's retirement of the dual simplex job and its idle
                                   priority, main.c: an extra thread speeds such an iteration) */
    int foldscale;              /* (7 November 2026, BRISK_LPFOLDSCALE) 1: when the data do not fold, the folding on the scaling that
                                   does not depend on the input's is tried (the caller found the problem as read's pattern coarse) */
} LpIpmOpts;
typedef struct {
    int status;                 /* 0 optimal, 2 dual infeasible (x: a ray), 3 iteration limit, 4 stalled or diverging (see why), 6 stopped */
    int iters, ncorr, ndense, ntiny;
    long lnz, nfact, nsolve, ncg;
    double pobj, dobj, pinf, dinf, gap, time, t_fact, t_solve;
    char why[192];
    int dense_lost;             /* (31 October 2026) stopped because the dense columns' correction lost its accuracy and
                                   joining them was refused: the SOCP solver, whose augmented system keeps them apart, follows */
    int free_stalled;           /* (2 November 2026) stopped for no progress with the free columns' weight at its largest (1);
                                   2: and the split of the free columns stalled as well */
    int geo_used;               /* (4 November 2026) the attempt's scaling took the geometric-mean passes (lp_scale_choose) */
} LpIpmInfo;
/* (23 October 2026) the cost in flops of one iteration of the method on this form (its own analysis: the dense columns'
 * choice and the nested dissection ordering, not an estimate; 1e300 when the factor does not fit), for a choice between
 * forms of a problem; the form as given (the method's choice of the dual form not made) */
double lpipm_iter_flops(int m, int n, const int *Ap, const int *Ai, const double *Ax, double budget);   /* (budget > 0: the analysis stops above it, 1e300) */
/* x (n), y (m), z = c - A'y (n; may be NULL) of the problem as given; returns the status */
int lpipm_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info);
/* 20 October 2026 (LP/SOCP): folding by the coarsest equitable partition (IMPLEMENTATION 26.8), for the cone solver's
 * linear case too (sedumi.c). lp_fold_partition: rcl (m), ccl (n) the classes, *nrc / *ncc their counts; returns 1 when
 * the partition is not the trivial one. lp_fold_quotient: the quotient (by columns, the free classes first; u_h 1e300
 * where unbounded) and the class sizes rcnt (nr), ccnt (nc); returns 0 when the partition is not equitable. */
int lp_fold_partition(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                      int *rcl, int *ccl, int *nrc, int *ncc);
/* (22 October 2026) the same partition by incremental refinement (the work of a pass is that of the elements that
 * changed class); the classes numbered differently. The LP method's own (lpipm_solve, the command line's check) */
int lp_fold_partition_inc(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                          int *rcl, int *ccl, int *nrc, int *ncc);
/* the same, stopping with "no fold" (0, every row and column its own class) as soon as the classes exceed stop (m + n):
 * the refinement only splits classes, so their number then stays above (the command line's check, stop 0.5) */
int lp_fold_partition_stop(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                           int *rcl, int *ccl, int *nrc, int *ncc, double stop);
int lp_fold_quotient(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                     int *rcl, int *ccl, int nr, int nc, int *nfree_h, int **Ap_h, int **Ai_h, double **Ax_h, double **b_h, double **c_h, double **u_h,
                     int *rcnt, int *ccnt);
int lp_fold_partition_cones(int m, int n, int nfree, int nlin, int nq, const int *q, int nr, const int *r,
                            const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c,
                            int *rcl, int *ccl, int *nrc, int *ncc, int *nkc);   /* (a measurement: the cones kept together) */
/* (7 November 2026) the folding on a scaling that does not depend on the input's (lpsolve.c, before the sifting): the
 * least-squares scaling of the matrix bordered by the sides r1, r2 (rows), the costs cv and the bounds c1, c2 (columns; any
 * may be NULL), as factors rs (m) and cs (n), returns 1 when converged; values equal up to tol made equal; the scaled and
 * grouped standard form (BRISK_LPFOLDSCALE) */
int lp_canon_scale(int m, int n, const int *Ap, const int *Ai, const double *Ax,
                   const double *r1, const double *r2, const double *cv, const double *c1, const double *c2,
                   double *rs, double *cs, int *iters, double *relres);
int lp_canon_scale_tol(int m, int n, const int *Ap, const int *Ai, const double *Ax,
                       const double *r1, const double *r2, const double *cv, const double *c1, const double *c2,
                       double *rs, double *cs, int *iters, double *relres, double rtol, int itmax);   /* (rtol 1e-14, itmax 1000 above) */
void lp_snap_values(double *v, int k, double tol);
int lp_fold_canon(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                  double **Axs, double **bs, double **cs, double **us, double **rsc, double **csc, int verbose);
/* (7 November 2026, the owner's decision; ISSUES 843) BRISK_LPGRBTOL=1: the LP methods stop by Gurobi's criteria (its
 * documented defaults: FeasibilityTol 1e-6 and OptimalityTol 1e-6, absolute, on the presolved and scaled problem the method
 * works on; BarConvTol 1e-8 for the interior-point method: the relative gap and the relative primal and dual infeasibility).
 * Off by default (an experiment); read at every call. A bit mask of its parts (7: all): 1 the simplex methods' tolerances
 * 1e-6; 2 the simplex methods' final rounds checked on the scaled problem, the postsolved solution taken within 1e-6, and
 * the check on the problem as read that keeps a plainly wrong result from being optimal; 4 the interior-point method's
 * infeasibilities measured on its scaled problem and its point taken up to 1e-3 of a row's activity (the default route).
 * 8 (8 November 2026) the simplex methods' scaling leaving out of its geometric passes the entries below 1e-9 of their row's or
 * column's largest (15: all). The Errors line gives absolute violations with any part on */
int brisk_lpgrbtol(void);
#define LPGRB_SXTOL 1
#define LPGRB_SXROUNDS 2
#define LPGRB_IPM 4
#define LPGRB_SXSCALE 8   /* (8 November 2026) the simplex methods' scaling: the geometric passes leave out entries below 1e-9 of their row's or column's largest (simplex.c, scale) */
int lp_ne_cost_nd(int m, int n, const int *Ap, const int *Ai, double budget, double tmax, double *flops, double *lnz);   /* (ISSUES 880) */
#endif
