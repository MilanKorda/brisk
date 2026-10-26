/* simplex.h - the simplex method for linear programs in general form (simplex.c):
 *     min / max  c'x + c0,   rl <= A x <= ru,   l <= x <= u        (infinite bounds: +-LP_INF)
 * by the dual or the primal simplex method; the result is an optimal basic solution with its basis. */
#ifndef BRISK_SIMPLEX_H
#define BRISK_SIMPLEX_H
#include "lpio.h"

typedef struct {
    int method;                 /* 0 automatic (the dual simplex method; with two free threads or more the interior-point
                                   method and the crossover beside it once a first budget of work is spent, the first to end
                                   optimal wins; on one thread, or with deterministic, the interior-point method when the
                                   simplex method passes its predicted work), 1 dual simplex,
                                   2 primal simplex, 3 the interior-point method, then the crossover to an optimal basis */
    double tol_primal;          /* feasibility tolerance on the scaled problem; <= 0: 1e-7 */
    double tol_dual;            /* optimality (dual feasibility) tolerance; <= 0: 1e-7 */
    double timelimit;           /* seconds; <= 0: none */
    long maxit;                 /* <= 0: none */
    int verbose;                /* 0 silent, 1 summary lines, 2 iteration log */
    int noscale, noperturb;     /* switches for experiments */
    int nopresolve;             /* 1: no presolve */
    double *ray;                /* if not NULL (m): on a primal infeasible result, the row multipliers of the proof
                                   (Farkas): the row r of B^{-1} that no column can make feasible, by the rows as read */
    const volatile int *stop;   /* set to nonzero from outside to stop (may be NULL) */
    const volatile int *threads;  /* (the race) if not NULL and positive: the threads the method may use now, read at every
                                     iteration (the interior-point method beside it takes the others while it runs) */
    const int *cstat0, *rstat0; /* a starting basis (statuses as below; NULL: the basis of the logicals) */
    const double *xstart;       /* an optimal point that is not basic (n): a basis by the crossover (pushes), then the
                                   primal method; NULL: none */
    const double *ystart;       /* with xstart: optimal row duals (m, as returned in y), for the dual pushes; may be NULL */
    int nosymmetry;             /* 1: no reduction by the problem's symmetry */
    double workbudget;          /* > 0: stop (status 6) once the work (entries touched, SimplexInfo.work) exceeds it */
    double (*budget_cb)(void *ctx, double work);   /* if set: called then; a value above the work is the new budget */
    void *budget_ctx;
    int deterministic;          /* method 0 with two free threads or more: 1 = the choice of the method by counted work
                                   (the same result on every run and at every thread count); 0 = the race by the clock:
                                   the interior-point method beside the simplex method, the first to end optimal wins */
    const struct SxOrig *orig;  /* (internal, set by simplex_solve for its routes: the problem as read and the presolve's
                                   maps, for the interior-point method's second attempt; callers leave it NULL) */
    int given_perturb;          /* (internal, the sifting on rows: a given basis that is dual feasible is solved with the
                                   costs perturbed, as from the slack basis, instead of the dual method's phase 2 alone) */
    int dual_race;              /* (internal: dual_route as one of the race's two simplex routes, the direct one beside it) */
    int in_dual;                /* (internal: the interior-point route on the dual of a tall problem; dual_route then runs
                                   the interior-point method and the crossover on the dual, which is not dualized again) */
    int ipm_kcorr;              /* the interior-point method's centrality correctors per iteration, plus one (0: chosen
                                   from the cost of a factorization against a solve; k + 1: k, the command line's -lpcorr k) */
} SimplexOpts;

/* basis status of a column or row */
enum { SX_BASIC = 0, SX_LOWER = 1, SX_UPPER = 2, SX_ZERO = 3, SX_FIXED = 4 };

typedef struct {
    int status;                 /* 0 optimal, 1 primal infeasible, 2 dual infeasible (unbounded), 3 iteration limit,
                                   4 numerical difficulties, 6 time limit or stopped */
    long iters, iters_dual, iters_primal, iters_phase1;
    int nfactor, nflips;
    double obj;                 /* the objective as in the file (sense and constant included) */
    double pinf, dinf;          /* largest primal and dual violation on the problem as read (absolute) */
    double time, t_factor, t_ftran, t_btran, t_price, t_presolve;
    long iters_post;            /* iterations on the problem as read after the postsolve */
    int have_ray;               /* the ray is in o->ray: a column j can help only if ray_sign * (ray'a_j) * dir_j > 0,
                                   dir_j = +1 for a column that may increase, -1 for one that may decrease */
    int ray_sign;
    int m_reduced, n_reduced;   /* the presolved problem's size */
    int dualized;               /* 1: solved through its dual problem (a tall problem), then from the dual's basis */
    int sym_m, sym_n;           /* > 0: solved on the classes of its symmetry (rows x columns), then the crossover */
    long pushes;                /* the crossover's pushes */
    int ipm_iters; double t_ipm;  /* method 3: the interior-point method's iterations and time */
    int auto_ipm;               /* method 0: the interior-point method was chosen, after iters_first iterations of the
                                   simplex method in t_first seconds (counted in iters and time) */
    long iters_first; double t_first;
    int raced;                  /* method 0: the interior-point method ran beside the simplex method (the race by the clock) */
    double work;                /* entries touched by the solves, the pricing and the factorizations (the same on every
                                   run and at every thread count: a clock for decisions that must not depend on time) */
    char why[96];
} SimplexInfo;

/* x (n), the row activities ax (m, may be NULL), the row duals y (m) and the reduced costs z (n) with z = c - A'y
 * (for a maximization: of the problem as written, so that z >= 0 at a lower bound means optimal for the minimum of -c);
 * the basis status of the columns cstat (n) and of the rows rstat (m) (may be NULL). Returns the status. */
int simplex_solve(const LpProb *L, const SimplexOpts *o, double *x, double *ax, double *y, double *z, int *cstat, int *rstat,
                  SimplexInfo *info);
/* a linear program in the cone form min c'x, A x = b, x = (free part K.f, nonnegative part K.l), as the interfaces give it
 * (P->nq = P->nr = P->ns = 0), by simplex_solve (method 0 when o is NULL: the automatic choice and its race): x (n), y (m)
 * and z = c - A'y (n), *pobj = c'x, *dobj = b'y, with the cone solver's conventions for a primal infeasible problem (y
 * with b'y = 1, z = -A'y, x = 0, pobj 0, dobj 1). Returns the status as simplex_solve, or -1 when the problem is not a
 * linear program in that form or is infeasible or unbounded without a certificate at hand (the caller then uses the cone
 * solver, which gives one). */
int simplex_solve_sedumi(const SedumiProb *P, const SimplexOpts *o, double *x, double *y, double *z, double *pobj, double *dobj,
                         SimplexInfo *info);
/* (24 October 2026) the LP interior-point method on the problem the simplex method's presolve left (main.c's route of
 * -lpmethod 1) declines when the problem as read is clearly cheaper, as the interior-point route of the simplex method
 * does: a context for LpIpmOpts.analyzed (sx_asread_cb), whether it declined, and its release */
void *sx_asread_new(const LpProb *asread, const LpProb *presolved, int verbose);
int sx_asread_cb(void *ctx, double flops);
int sx_asread_declined(const void *ctx);
void sx_asread_free(void *ctx);
#endif
