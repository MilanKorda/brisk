/* sxmodel.h - the simplex method's state kept between solves (simplex.c), for the re-solve interface (lpmodel.c).
 * Internal: lpmodel.h is the interface. The problem is the one given (no presolve), scaled once; the basis, its
 * factorization and the values stay from one solve to the next. Variables: the n columns, then the m row logicals. */
#ifndef BRISK_SXMODEL_H
#define BRISK_SXMODEL_H
#include "lpio.h"

typedef struct SxModel SxModel;
typedef struct SxmSaved SxmSaved;

/* the problem as given, scaled by the simplex method's own scaling (cscale NULL) or by the factors given (cscale n,
 * rscale for the first mscaled rows, costscale > 0; the other rows' factors computed); the basis of the logicals */
SxModel *sxm_new(const LpProb *L, const double *cscale, const double *rscale, int mscaled, double costscale);
void sxm_free(SxModel *M);
void sxm_scaling(const SxModel *M, double *cscale, double *rscale, double *costscale);

/* changes (values as given; infinite: +-LP_INF); they take effect at the next solve */
void sxm_col_bounds(SxModel *M, int j, double l, double u);
void sxm_row_bounds(SxModel *M, int i, double rl, double ru);
void sxm_cost(SxModel *M, int j, double c);

/* the basis (statuses of simplex.h); set: repaired by the factorization, returns the number of columns replaced */
void sxm_get_basis(const SxModel *M, int *cstat, int *rstat);
int  sxm_set_basis(SxModel *M, const int *cstat, const int *rstat);
/* (28 October 2026) the same with the pricing weights of the basis chosen (wmode): 0 the slack basis's 1 (sxm_set_basis;
 * BRISK_SXMLAZYW=1 or BRISK_SXMEXACTW=1 for experiments), 1 each computed the first time its row is a candidate to leave, 2 all
 * computed at once (a backward solve a row) */
int  sxm_set_basis_w(SxModel *M, const int *cstat, const int *rstat, int wmode);

typedef struct {
    long maxit; double timelimit; const volatile int *stop;
    double tol_primal, tol_dual;    /* <= 0: 1e-7 */
    int cutoff_on; double cutoff;   /* the objective as given without its constant */
} SxmOpts;
typedef struct { int status; long iters, iters_dual, iters_primal; int nfactor; double work; } SxmRes;
/* from the current basis: the dual method when it is dual feasible (boxed columns moved to the bound their reduced cost
 * asks for), the primal method when it is primal feasible and the costs changed or it is not dual feasible, else the
 * dual method with its phase 1. Status: 0 optimal, 1 infeasible, 2 unbounded, 3 iteration limit, 4 numerical
 * difficulties, 6 time limit or stopped, 7 the cutoff passed */
int  sxm_solve(SxModel *M, const SxmOpts *o, SxmRes *r);

/* the values on the problem as given: x (n), the row duals y (m) and the reduced costs z = c - A'y (n), as
 * simplex_solve; NULL skips one */
void sxm_values(const SxModel *M, double *x, double *y, double *z);
double sxm_objective(const SxModel *M);    /* the objective as given without its constant */
/* after status 1: y (m) with max over l <= x <= u, rl <= r <= ru of y'(A x - r) < 0; returns 0 when there is none */
int  sxm_farkas(const SxModel *M, double *y);
/* after status 2: d (n) with c'd < 0 (minimizing; > 0 maximizing), d within the directions the bounds of the columns
 * and of the rows (A d) allow; returns 0 when there is none */
int  sxm_ray(const SxModel *M, double *d);

/* the tableau, with the row activities r = A x as the logicals ([A, -I] (x, r) = 0): head[i] the basic variable of
 * position i (j < n a column, n + k the activity of row k); row i of B^{-1} (m) and of B^{-1} A (n) */
void sxm_head(const SxModel *M, int *head);
void sxm_binv_row(SxModel *M, int i, double *out);
void sxm_binva_row(SxModel *M, int i, double *out);

/* the state (basis, factorization with its updates, values, steepest-edge weights, bounds), saved and restored */
SxmSaved *sxm_save(const SxModel *M);
void sxm_restore(SxModel *M, const SxmSaved *V);
void sxm_saved_free(SxmSaved *V);
#endif
