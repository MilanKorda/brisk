/* lpsolve.h - the interior-point method for linear programs in bounded standard form (lpsolve.c):
 *     min c'x,   A x = b,   x_j free (j < nfree),   0 <= x_j <= u_j (the others; u_j >= 1e30: none) */
#ifndef BRISK_LPSOLVE_H
#define BRISK_LPSOLVE_H
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
} LpIpmOpts;
typedef struct {
    int status;                 /* 0 optimal, 3 iteration limit, 4 stalled or diverging (see why), 6 stopped */
    int iters, ncorr, ndense, ntiny;
    long lnz, nfact, nsolve, ncg;
    double pobj, dobj, pinf, dinf, gap, time, t_fact, t_solve;
    char why[96];
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
#endif
