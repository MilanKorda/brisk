/* socpmodel.h - an SOCP kept for re-solves after bound changes, each solve warm-started from the last optimal one
 * (branch and bound on mixed-integer SOCPs; the SOCP developer, 23 October 2026). See socpmodel.c. */
#ifndef SOCPMODEL_H
#define SOCPMODEL_H
#include "socp.h"
typedef struct SocpModel SocpModel;
/* the problem copied; opts: tolerance, iteration and time limits, verbosity, wlam (the warm start's weight, 0.99) */
SocpModel *socpm_create(const SocpProb *P, const SocpOpts *opts);
/* lo <= x_j <= hi for any column j (-inf / +inf: no bound on that side); replaces the column's earlier bounds */
int socpm_set_var_bounds(SocpModel *M, int j, double lo, double hi);
int socpm_set_cost(SocpModel *M, const double *c);
/* stop a solve (SOCP_CUTOFF) once an estimate of its lower bound exceeds the cutoff (a branch and bound's incumbent); +inf: off.
 * The estimate counts the iterate's size for that of a feasible point: a test, not a proof (SocpOpts.cutoff, socp.h) */
void socpm_set_cutoff(SocpModel *M, double cutoff);
/* solves the current problem; R's x, z over the original columns, y over the original rows */
int socpm_solve(SocpModel *M, SocpRes *R);
void socpm_destroy(SocpModel *M);
/* the warm-start point kept by the model: saved (NULL when none) and put back, for example the parent's before each child
 * of a branch-and-bound node; the saved copy is freed by socpm_point_free */
typedef struct SocpmPoint SocpmPoint;
SocpmPoint *socpm_save_point(const SocpModel *M);
/* 0: restored; 1: refused (a point of another model or shape), the model's own point unchanged; -1: out of memory */
int socpm_restore_point(SocpModel *M, const SocpmPoint *p);
void socpm_point_free(SocpmPoint *p);
#endif
