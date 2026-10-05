/* socp.h - BRISK's solver for linear and second-order cone programs (socp.c) */
#ifndef BRISK_SOCP_H
#define BRISK_SOCP_H
enum { SOCP_OPTIMAL = 0, SOCP_PINF = 1, SOCP_DINF = 2, SOCP_MAXIT = 3, SOCP_NUMERR = 4, SOCP_REDUCED = 5, SOCP_TIME = 6, SOCP_INPUT = 7 };
typedef struct {
    int m, n;                       /* rows and columns of A */
    const int *Ap, *Ai; const double *Ax;   /* A by columns (compressed sparse columns, 0-based) */
    const double *b, *c;
    int nf, nl;                     /* free variables, nonnegative variables (in this order, first) */
    int nq; const int *q;           /* second-order cones x0 >= |x(1:)| and their dimensions */
    int nr; const int *r;           /* rotated cones 2 x0 x1 >= |x(2:)|^2, x0, x1 >= 0 */
} SocpProb;
typedef struct { double tol; int maxit; double timelimit; int verbose; int equil;
                 double step;        /* fraction of the way to the boundary (0: the default 0.99) */
                 int attempt;        /* internal: 0 a single solve, 1 the first attempt, 2 the second */
} SocpOpts;
typedef struct {
    int status, iters;
    double *x, *y, *z;              /* x (n), y (m), z (n): the dual slack of the iteration, in K* and equal to c - A'y up to
                                       the dual residual; a certificate when infeasible */
    double pobj, dobj, err[4], maxerr, time;
    long lnz;
} SocpRes;
void socp_default_opts(SocpOpts *o);
int socp_solve(const SocpProb *P, const SocpOpts *opt, SocpRes *R);
void socp_result_free(SocpRes *R);
const char *socp_status_str(int s);
extern int (*socp_printf)(const char *fmt, ...);   /* where the solver prints (default: printf) */
extern volatile int *socp_stop;                    /* a flag that stops the solve when it becomes nonzero */
#endif
