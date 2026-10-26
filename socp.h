/* socp.h - BRISK's solver for linear and second-order cone programs (socp.c) */
#ifndef BRISK_SOCP_H
#define BRISK_SOCP_H
enum { SOCP_OPTIMAL = 0, SOCP_PINF = 1, SOCP_DINF = 2, SOCP_MAXIT = 3, SOCP_NUMERR = 4, SOCP_REDUCED = 5, SOCP_TIME = 6, SOCP_INPUT = 7, SOCP_CUTOFF = 8 };
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
                 /* (22 October 2026) a warm start: a point (x, y, z) of this problem (x and z in its cones, for example a
                  * related problem's solution with the entries of new variables set inside their cones), started from as
                  * wlam (x, y, z) + (1 - wlam) (e, 0, e), e the cones' identity (Skajaa, Andersen and Ye 2013); the
                  * problem is then solved as given (no presolve, no dual form). NULL wx: the usual start */
                 const double *wx, *wy, *wz; double wlam;
                 double cutoff;      /* (22 October 2026) stop (SOCP_CUTOFF) once the lower bound b'y - |c - A'y - z| max(1, |x|) of an
                                        iterate exceeds it, z in the dual cone and x the iterate (a branch and bound's incumbent:
                                        the node cannot improve on it); the bound counts the iterate's size for that of a
                                        feasible point, so it is a test, not a proof. Default: +inf (off) */
                 double wsave;       /* > 0: keep the first iterate with a relative gap at most wsave and residuals below 1e-2 (a
                                        centred point to warm-start related problems from) in SocpRes cx, cy, cz; such a solve
                                        is also done as given (no presolve, no dual form) */
                 int noretry;        /* (25 October 2026) 1: no second attempt with shorter steps after a first attempt that ends at
                                        reduced accuracy (errors within 1e-6; the QCQP route, whose polishing and second pass
                                        recover such a point); one that ends in numerical difficulties is repeated; default 0 */
} SocpOpts;
typedef struct {
    int status, iters;
    double *x, *y, *z;              /* x (n), y (m), z (n): the dual slack of the iteration, in K* and equal to c - A'y up to
                                       the dual residual; a certificate when infeasible */
    double pobj, dobj, err[4], maxerr, time;
    long lnz;
    double *cx, *cy, *cz;           /* the centred iterate kept with SocpOpts.wsave (NULL when none) */
} SocpRes;
void socp_default_opts(SocpOpts *o);
int socp_solve(const SocpProb *P, const SocpOpts *opt, SocpRes *R);
void socp_result_free(SocpRes *R);
const char *socp_status_str(int s);
extern int (*socp_printf)(const char *fmt, ...);   /* where the solver prints (default: printf) */
extern void (*socp_note)(const char *fmt, ...);    /* 5.8: lines for the caller's summary (NULL: none) */
extern volatile int *socp_stop;                    /* a flag that stops the solve when it becomes nonzero */
#endif
