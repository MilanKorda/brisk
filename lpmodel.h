/* lpmodel.h - a linear program kept between solves, for a mixed-integer solver (branch and bound, cutting planes)
 * that solves many closely related linear programs (lpmodel.c; internal to the tree, not the public C API).
 *
 *     min / max  c'x + c0,   rl <= A x <= ru,   l <= x <= u        (infinite bounds: +-LP_INF = 1e30)
 *
 * The first solve may use everything BRISK has (presolve, symmetry, the automatic choice of the method, the
 * interior-point method with the crossover to a basis). Every later solve works on the problem as given, with its
 * scaling fixed at the first solve, from the last basis and its factorization: after changed bounds or added rows the
 * dual simplex method, after changed costs the primal simplex method, with no setup. All values going in and out are
 * those of the problem as given (unscaled, in its sense).
 *
 * The variables of the basis: the n columns x and the m row activities r = A x (bounds rl <= r <= ru), so that
 * [A, -I] (x, r) = 0 and B is formed from the columns of [A, -I] of the basic variables. Basis statuses: the SX_*
 * values of simplex.h (SX_BASIC, SX_LOWER, SX_UPPER, SX_ZERO for a free nonbasic variable at zero, SX_FIXED); a row's
 * status names the bound of its activity (SX_LOWER: r_i = rl_i).
 *
 * Threads: an object holds no global state; independent objects may be used at the same time from different threads,
 * one object by one thread at a time. A solve uses one thread (the first solve as many as the option threads asks).
 * Determinism: the same calls give the same iterates and results on every run and at any number of threads.
 *
 * Return values: functions returning int give 0 on success and -1 on a wrong argument or a call out of order, unless
 * stated otherwise. */
#ifndef BRISK_LPMODEL_H
#define BRISK_LPMODEL_H
#include "lpio.h"
#include "simplex.h"

typedef struct LpModel LpModel;

/* the status of a solve */
enum {
    LM_NOTSOLVED = -1,     /* no solve since the object was created or changed */
    LM_OPTIMAL = 0,
    LM_INFEASIBLE = 1,     /* a Farkas certificate is available (lm_get_farkas) when the dual simplex method found it */
    LM_UNBOUNDED = 2,      /* a ray is available (lm_get_ray) when the primal simplex method found it */
    LM_ITERLIMIT = 3,
    LM_NUMERICAL = 4,      /* numerical difficulties */
    LM_TIMELIMIT = 5,
    LM_STOPPED = 6,        /* the stop flag was set */
    LM_CUTOFF = 7          /* the dual simplex method's objective passed the cutoff: the optimum lies beyond it */
};

/* ---- creation: the problem is copied. A by columns (Ap[n+1], Ai, Ax; entries of a column in any order, no duplicates);
 * c (n); l, u (n); rl, ru (m); maximize: 0 or 1. Returns NULL on a wrong argument. */
LpModel *lm_create(int m, int n, const int *Ap, const int *Ai, const double *Ax, const double *c, double c0,
                   const double *l, const double *u, const double *rl, const double *ru, int maximize);
LpModel *lm_create_lp(const LpProb *L);
void lm_free(LpModel *M);
int lm_rows(const LpModel *M);
int lm_cols(const LpModel *M);

/* ---- options: lm_default_opts fills the defaults; lm_set_opts copies them into the object (used by the next solves) */
typedef struct {
    long maxit;                 /* simplex iterations per solve; <= 0: none (default) */
    double timelimit;           /* seconds per solve; <= 0: none (default) */
    const volatile int *stop;   /* set to nonzero from outside to stop a solve (LM_STOPPED); NULL: none (default) */
    double tol_primal;          /* feasibility tolerance (of the scaled problem); default 1e-7 */
    double tol_dual;            /* optimality tolerance (of the scaled problem); default 1e-7 */
    int cutoff_on;              /* 1: the objective cutoff below is used (default 0) */
    double cutoff;              /* the dual simplex method stops with LM_CUTOFF as soon as the bound on the optimum that
                                   its current duals give passes this value (above it when minimizing, below it when
                                   maximizing); a solve that ends optimal first is LM_OPTIMAL, its objective then
                                   possibly beyond the cutoff (the caller compares it) */
    int threads;                /* threads of the first solve (default 1); re-solves use one thread */
    int first_presolve;         /* 1 (default): the first solve by simplex_solve (presolve, symmetry, the automatic choice
                                   of the method), its basis then taken by the object; 0: the dual simplex method on the
                                   problem as given from the basis of the row activities */
    int verbose;                /* 0 silent (default) */
} LmOpts;
void lm_default_opts(LmOpts *o);
int lm_set_opts(LpModel *M, const LmOpts *o);

/* ---- changes between solves (values as given). They keep the basis; the results of the last solve are no longer
 * available after any change. Bounds may be infinite (+-LP_INF); a lower bound above the upper bound makes the next
 * solve LM_INFEASIBLE without a certificate. */
int lm_set_col_bounds(LpModel *M, int j, double l, double u);
int lm_set_col_bounds_list(LpModel *M, int k, const int *cols, const double *l, const double *u);
int lm_set_row_bounds(LpModel *M, int i, double rl, double ru);
int lm_set_obj(LpModel *M, int j, double c);
/* k rows added at the end (cuts), by rows: row t has the entries cols[start[t] .. start[t+1]-1] with values val[...],
 * and the bounds rl[t], ru[t]. Their row activities enter the basis (a basis stays a basis and dual feasible). */
int lm_add_rows(LpModel *M, int k, const int *start, const int *cols, const double *val, const double *rl, const double *ru);
/* k rows deleted (indices, any order, no duplicates); the other rows keep their order and are numbered again. A row whose
 * activity is basic (a slack cut) leaves the basis as it was; deleting a row whose activity is nonbasic leaves one basic
 * variable too many, which becomes nonbasic (the basis may then lose its dual feasibility). */
int lm_del_rows(LpModel *M, int k, const int *rows);

/* ---- solve. Returns the status (LM_*), or -1 on an error; info (may be NULL) gets the details. The time is that of the
 * whole call. */
typedef struct {
    int status;
    double obj;                 /* the objective as given (sense and constant included); at LM_CUTOFF the bound reached */
    long iters, iters_dual, iters_primal;
    int nfactor;                /* factorizations of the basis in this solve */
    int first;                  /* 1: this was the first solve (with the presolve when first_presolve) */
    double work;                /* the deterministic work of the simplex method in this solve: the entries its operations
                                   touched, the factorizations' included (the same on every run; in the first solve
                                   the simplex method's work, its presolve and interior-point method not counted) */
    double time;                /* seconds */
} LmInfo;
int lm_solve(LpModel *M, LmInfo *info);

/* ---- results of the last solve on the problem as given; available until the next change (else -1). At LM_OPTIMAL an
 * optimal basic solution; at a limit, a stop or the cutoff the current basic solution of the method. */
int lm_status(const LpModel *M);
double lm_obj(const LpModel *M);
int lm_get_x(const LpModel *M, double *x);             /* n */
int lm_get_activity(const LpModel *M, double *r);      /* m: r = A x */
int lm_get_duals(const LpModel *M, double *y);         /* m: the row duals (for a maximization: of the problem as written) */
int lm_get_redcosts(const LpModel *M, double *z);      /* n: z = c - A'y */
/* after LM_INFEASIBLE: y (m) with max { y'(A x - r) : l <= x <= u, rl <= r <= ru } < 0, which no x can satisfy;
 * returns 1 when given, 0 when there is none (the infeasibility was found otherwise) */
int lm_get_farkas(const LpModel *M, double *y);
/* after LM_UNBOUNDED: d (n) along which the objective improves without end: c'd < 0 when minimizing (> 0 maximizing),
 * d_j >= 0 where l_j is finite, d_j <= 0 where u_j is finite, (A d)_i >= 0 where rl_i is finite and <= 0 where ru_i is
 * finite (within the tolerances); returns 1 when given, 0 when there is none */
int lm_get_ray(const LpModel *M, double *d);

/* ---- the basis. get: the statuses of the columns (n) and rows (m) as of the last solve or set (either may be NULL).
 * set: a basis given by statuses; one that cannot be factorized is repaired (row activities in place of dependent
 * columns, which become nonbasic at a bound); more basic variables than rows: the last become nonbasic; fewer: row
 * activities are added. Returns the number of variables the repair changed (>= 0), or -1 on an error. */
int lm_get_basis(LpModel *M, int *cstat, int *rstat);
int lm_set_basis(LpModel *M, const int *cstat, const int *rstat);

/* ---- the tableau of the current basis (for cutting planes), values as given:
 * head (m): the basic variable of each position i of the basis (j < n: column j; n + k: the activity of row k);
 * row i of B^{-1} (m) and row i of B^{-1} A (n). Row i of B^{-1} [A, -I] has 1 at the basic variable of position i, so
 * x_head(i) (or r_k) = - sum over the nonbasic variables of that row's entries times their values. */
int lm_get_head(LpModel *M, int *head);
int lm_binv_row(LpModel *M, int i, double *out);
int lm_binva_row(LpModel *M, int i, double *out);

/* ---- a trial of bound changes (strong branching): lm_trial_begin saves the state (basis, its factorization with the
 * updates, values, weights, bounds); then bounds and costs may change and solves run (with an iteration limit, say);
 * lm_trial_end restores the saved state without a new factorization (a copy of the factorization) and makes the
 * results of the solve before lm_trial_begin available again. Rows may not be added or deleted during a trial; trials
 * do not nest. */
int lm_trial_begin(LpModel *M);
int lm_trial_end(LpModel *M);
#endif
