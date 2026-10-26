/* qp.h - BRISK's convex quadratic programming solver (the SOCP developer, 22 October 2026).
 *
 *     min  1/2 x'Hx + c'x + c0    s.t.   rl <= A x <= ru,    xl <= x <= xu
 *
 * H symmetric positive semidefinite, given by its lower triangle (diagonal included) by columns; the factor 1/2 is the
 * QPS convention: the problem min x'Qx + h'x is given with H = 2Q. A (m x n) by columns. Infinite bounds: +-QP_INF
 * (any value of magnitude QP_INF or more counts as infinite). An equality row has rl = ru; Ax <= b is rl = -QP_INF,
 * ru = b; bounds on single variables (boxes, nonnegativity) are given by xl, xu, not as rows.
 *
 * Two methods: the interior-point method (primal-dual, Mehrotra's predictor-corrector, the quasi-definite KKT system
 * factored by BRISK's sparse signed Cholesky) and the active-set method (the dual active-set method of Goldfarb and
 * Idnani, warm-started from a working set). A solver object keeps the problem, its factorizations and the last
 * solution, so that a changed problem (bounds, row bounds, the cost) is re-solved from where the last solve ended:
 * the interface for branch and bound in mixed-integer QP. */
#ifndef BRISK_QP_H
#define BRISK_QP_H
#include <stddef.h>
#define QP_INF 1e20

typedef struct {
    int n, m;
    int *Hp, *Hi; double *Hx;      /* lower triangle of H by columns (Hp: n+1); NULL Hp: H = 0 (an LP) */
    int *Ap, *Ai; double *Ax;      /* A by columns (Ap: n+1) */
    double *c, c0;
    double *rl, *ru;               /* row bounds (m) */
    double *xl, *xu;               /* variable bounds (n) */
    int maxim;                     /* the file maximizes: the objective was negated on reading */
    char name[64];
    char **rname, **cname;         /* names (may be NULL) */
    /* quadratic constraints (a QCQP; nqc = 0 for a QP): row qrow[k] is rl <= a'x + 1/2 x'Q_k x <= ru, with Q_k's lower
     * triangle (row qi >= column qj) in entries qs[k] .. qs[k+1]-1. Read from QPS's QCMATRIX sections (which list
     * M = Q_k / 2 in full, Gurobi's and CPLEX's convention) */
    int nqc, *qrow, *qs, *qi, *qj; double *qv;
} QpProb;

enum { QP_OPTIMAL = 0, QP_PRIMAL_INFEASIBLE = 1, QP_DUAL_INFEASIBLE = 2, QP_MAXIT = 3, QP_NUMERICAL = 4,
       QP_TIME = 5, QP_REDUCED = 6, QP_INPUT = 7 };
enum { QP_METHOD_AUTO = 0, QP_METHOD_IPM = 1, QP_METHOD_ACTIVESET = 2 };

typedef struct {
    double tol;          /* relative primal, dual and gap tolerance (1e-8) */
    int maxit;           /* iterations (interior point: 200; active set: 100 n + 1000) */
    int verbose;         /* 0 silent, 1 summary, 2 iterations, 3 details */
    double timelimit;    /* seconds, 0: none */
    int method;          /* QP_METHOD_* */
    int threads;         /* for the factorization's large blocks (0: as set) */
} QpOpts;

typedef struct {
    int status;
    double obj;          /* 1/2 x'Hx + c'x + c0 at x (the file's sense) */
    double *x;           /* n */
    double *y;           /* m: the rows' multipliers, y > 0 at the lower bound, y < 0 at the upper (y'(Ax)) */
    double *z;           /* n: the bounds' multipliers, z > 0 at the lower bound, z < 0 at the upper */
    int iter;
    double pres, dres, gap;   /* relative primal and dual residuals and duality gap at the end */
    double time;
    int resolves;        /* (23 October 2026) 1: solved a second time with other dependent rows dropped (qpipm.c), t_first the first solve's time */
    double t_first;
    int method;          /* (23 October 2026) the method that gave the result (QP_METHOD_IPM or QP_METHOD_ACTIVESET; 0: as asked) */
    double t_as;         /* > 0: the default choice ran the active-set method first (this long), it did not end optimal, and
                          * the interior-point method gave the result */
    double tol2;         /* (24 October 2026) resolves == 2: a QCQP's cones solved a second time at this tolerance (qcqp.c), */
    int kept2;           /* its result kept (1) or the first one's (0) */
} QpRes;

void qp_default_opts(QpOpts *o);
void qp_prob_free(QpProb *P);
void qp_res_free(QpRes *R);
/* where the QP and QCQP code prints (default printf; the command line and the library set BRISK's own) */
extern int (*qp_printf)(const char *fmt, ...);
const char *qp_status_name(int st);
/* the objective and the errors of (x, y, z) on P: max relative bound and row violation, the relative dual residual
 * Hx + c - A'y - z, the relative gap between the primal and the dual objective */
double qp_objective(const QpProb *P, const double *x);
void qp_errors(const QpProb *P, const double *x, const double *y, const double *z, double *epri, double *edual, double *egap);

/* QPS files (MPS with QUADOBJ, QMATRIX or QSECTION): 0 on success, message in msg */
int qp_read_qps(const char *fname, QpProb *P, char *msg, size_t nmsg);
int qp_is_qps(const char *fname);
int qp_write_qps(const QpProb *P, const char *fname);
int qp_write_text(const QpProb *P, const char *fname);    /* plain triplets, for the comparison harness */
int qcqp_solve(const QpProb *P, const QpOpts *o, QpRes *R);
double qp_scale_of(const QpProb *P);
double qp_cost_scale(const QpProb *P);
int qp_farkas_proof(const QpProb *P, const double *y, const double *z);   /* 1: (y, z) proves that no x is feasible (24 October 2026) */
void qp_infeas_measures(const QpProb *P, const double *x, const double *y, const double *z,
                        double *pinf_val, double *pinf_res, double *dinf_val, double *dinf_res);  /* quadratic constraints: by the SOCP solver (qcqp.c) */     /* free format, plain names */

/* (24 October 2026) the exact test that H and the quadratic rows' Q_k are positive semidefinite where it needs no
 * tolerance (no negative diagonal entry; a zero diagonal entry only in a zero row): 0, or QP_INPUT with the message on
 * stderr; qp_solve and the solver object's first solve refuse a problem that fails it */
int qp_convex_exact(const QpProb *P);
int qp_psd_exact(int n, int ne, const int *qi, const int *qj, const double *qv, double sg, int *j1, int *j2);
/* one solve */
int qp_solve(const QpProb *P, const QpOpts *o, QpRes *R);
int qp_solve_ipm(const QpProb *P, const QpOpts *o, QpRes *R);
int qp_solve_activeset(const QpProb *P, const QpOpts *o, QpRes *R);
int qp_solve_activeset_pd(const QpProb *P, const QpOpts *o, QpRes *R);   /* only for H positive definite; else -1, nothing solved */

/* the solver object for warm starts */
typedef struct QpSolver QpSolver;
QpSolver *qp_create(const QpProb *P, const QpOpts *o);       /* P is copied */
int  qp_resolve(QpSolver *S, QpRes *R);                       /* from the last solution (the first call: cold) */
int  qp_set_var_bounds(QpSolver *S, int j, double lo, double hi);
int  qp_set_row_bounds(QpSolver *S, int i, double lo, double hi);
int  qp_set_cost(QpSolver *S, const double *c);
const QpProb *qp_problem(const QpSolver *S);
void qp_destroy(QpSolver *S);
#endif
