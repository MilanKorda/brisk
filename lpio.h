/* lpio.h - linear programs in general form and their files (lpio.c):
 *     min / max  c'x + c0,   rl <= A x <= ru,   l <= x <= u        (infinite bounds: +-LP_INF)
 * read from an MPS file (fixed or free format), presolved, and brought to the standard form
 * A x = b, x in (free) x (nonnegative) that the cone solver takes (sedumi.h), with the map back. */
#ifndef BRISK_LPIO_H
#define BRISK_LPIO_H
#include <stddef.h>
#include "sedumi.h"
#define LP_INF 1e30
typedef struct {
    int m, n;
    int *Ap, *Ai; double *Ax;       /* A by columns */
    double *c, *rl, *ru, *l, *u;
    double c0; int maxim;           /* objective constant; 1 = maximize */
    char **rname, **cname;          /* names (may be NULL) */
    char name[64];
} LpProb;
/* the map from the solution of the standard form back to the general form */
typedef struct {
    int n0, m0;                     /* columns and rows of the problem as read */
    int *ckind;                     /* per original column: 0 x = lo + x', 1 x = up - x', 2 free, 3 fixed (value in cval) */
    int *cpos;                      /* its column in the standard form (-1: none) */
    double *cval;                   /* the shift lo or up, or the fixed value */
    double c0;                      /* objective constant of the standard form (c'x_std + c0 = the objective, before the sign) */
    int maxim;
    int status;                     /* 0 a problem to solve, 1 primal infeasible found by the presolve, 2 unbounded (dual infeasible) */
    int nrow_removed, ncol_removed; long nnz_removed;
    int *rpos;                      /* per original row: its row in the standard form (-1: removed by the presolve) */
    int *rsgn;                      /* its sign there: y_orig = rsgn * y_std */
} LpMap;
int  lp_read_mps(const char *fname, LpProb *L, char *msg, size_t nmsg);
void lp_free(LpProb *L);
/* presolve (level 0: none) and standard form; S and M are allocated */
int  lp_to_sedumi(LpProb *L, int presolve, int verbose, SedumiProb *S, LpMap *M);
/* the same in the bounded form (ub != NULL): upper bounds returned in *ub (per column, LP_INF: none), no bound rows */
int  lp_to_std(LpProb *L, int presolve, int verbose, SedumiProb *S, LpMap *M, double **ub);
/* x of the general form from the standard form's solution; returns the objective (file convention) */
double lp_recover(const LpProb *L, const LpMap *M, const double *xstd, double *x);
/* errors of x on the problem as read: max bound and row violations, relative */
void lp_errors(const LpProb *L, const double *x, double *ebound, double *erow);
void lp_map_free(LpMap *M);
int  lp_is_mps(const char *fname);
/* CBF files (CBLIB): continuous problems with linear and second-order cones, in SeDuMi form;
 * the objective of the file is +-(c'x) + c0 (maxim: the file maximizes, c is negated) */
int  cbf_is_file(const char *fname);
int  cbf_read(const char *fname, SedumiProb *S, double *c0, int *maxim, int *nint, char *msg, size_t nmsg);
#endif
