/* simplex_pre.h - the presolve and postsolve of the simplex method (simplex_pre.c) */
#ifndef BRISK_SIMPLEX_PRE_H
#define BRISK_SIMPLEX_PRE_H
#include <stddef.h>
#include "lpio.h"
typedef struct SxPre SxPre;
/* reduces L into R (a minimization; R's arrays allocated, free with lp_free); returns 0, 1 if L is primal infeasible,
 * 2 if dual infeasible (why: the reason). *P holds what the postsolve needs (free with sxpre_free) */
int  sxpre_presolve(const LpProb *L, LpProb *R, SxPre **P, int verbose, char *why, size_t nwhy);
int  sxpre_presolve_ex(const LpProb *L, LpProb *R, SxPre **P, int verbose, int aggbmode, char *why, size_t nwhy);
/* x, y (row duals as minimization) and the basis statuses of L from those of the reduced problem */
void sxpre_postsolve(const SxPre *P, const double *xr, const double *yr, const int *cstr, const int *rstr, double *x, double *y, int *cst, int *rst);
void sxpre_free(SxPre *P);
/* the reduced problem's x and row duals (as minimization) from those of L: the values of the columns and rows it kept,
 * the presolve's merges replayed (y may be NULL: zeros). Not exact in general: an optimal point of L need not be optimal
 * for the reduced problem (a diagnostic: tools/sxpre_restrict_check.c) */
void sxpre_restrict(const SxPre *P, const double *x, const double *y, double *xr, double *yr);
/* the dual of a problem as a problem of the same form (a minimization; one row per column that is not fixed, one column
 * per row with a bound, two for a ranged row, and one per boxed column); *M maps its basis back (sxpre_dual_basis) */
typedef struct SxDual SxDual;
int  sxpre_dualize(const LpProb *A0, LpProb *D, SxDual **M);
void sxpre_dual_basis(const SxDual *M, const LpProb *A0, const int *cstd, const int *rstd, int *cst, int *rst);
void sxpre_dual_free(SxDual *M);
/* the problem on the classes of its symmetry (the coarsest equitable partition of rows and columns): 1 and Q (and *M)
 * when it is at most 0.7 of the size, else 0; sxpre_sym_lift gives the problem's point from Q's */
typedef struct SxSym SxSym;
int  sxpre_symmetry(const LpProb *A0, LpProb *Q, SxSym **M, int verbose, double budget);   /* budget > 0: at most that many entries touched by the refinement's rounds, else 0 */
void sxpre_sym_lift(const SxSym *M, const double *xq, const double *yq, double *x, double *y);
void sxpre_sym_free(SxSym *M);
#endif
