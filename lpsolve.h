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
} LpIpmOpts;
typedef struct {
    int status;                 /* 0 optimal, 3 iteration limit, 4 stalled or diverging (see why), 6 stopped */
    int iters, ncorr, ndense, ntiny;
    long lnz, nfact, nsolve, ncg;
    double pobj, dobj, pinf, dinf, gap, time, t_fact, t_solve;
    char why[96];
} LpIpmInfo;
/* x (n), y (m), z = c - A'y (n; may be NULL) of the problem as given; returns the status */
int lpipm_solve(int m, int n, int nfree, const int *Ap, const int *Ai, const double *Ax, const double *b, const double *c, const double *u,
                const LpIpmOpts *opt, double *x, double *y, double *z, LpIpmInfo *info);
#endif
