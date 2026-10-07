/* sedumi.h - problems in SeDuMi format (sedumi.c): min c'x, A x = b, x in K, with
 * K = free (K.f) x nonnegative (K.l) x second-order cones (K.q) x rotated second-order cones
 * (K.r) x semidefinite blocks (K.s, each as its d*d entries by columns), and its dual
 * max b'y, c - A'y = z in K*. */
#ifndef BRISK_SEDUMI_H
#define BRISK_SEDUMI_H
#include <stddef.h>
#include "brisk.h"
typedef struct {
    int m, n;                       /* rows and columns of A */
    int *Ap, *Ai; double *Ax;       /* A by columns (compressed sparse columns, 0-based) */
    double *b, *c;
    int nf, nl;                     /* K.f, K.l */
    int nq; int *q;                 /* K.q: x0 >= |x(1:)| */
    int nr; int *r;                 /* K.r: 2 x0 x1 >= |x(2:)|^2, x0, x1 >= 0 */
    int ns; int *s;                 /* K.s */
} SedumiProb;
typedef struct {
    int status;                     /* as BriskResult: 0 optimal, 1 primal infeasible, 2 dual infeasible, 3 iteration
                                       limit, 4 numerical difficulties, 5 reduced accuracy, 6 time limit; -1 not solved */
    char status_str[40];
    int exit_code, iters, have_x, m, n;
    double *x, *y, *z;              /* x (n), y (m), z = c - A'y (n) */
    double pobj, dobj;              /* c'x, b'y */
    double err[6];                  /* DIMACS errors 1..6 */
    double time;
    int cone;                       /* 1: solved by the cone solver (0: by the semidefinite solver, which reports itself) */
} SedumiRes;
int  sedumi_is_mat(const char *fname);
int  sedumi_read_mat(const char *fname, SedumiProb *P, char *msg, size_t nmsg);
void sedumi_free(SedumiProb *P);
int  sedumi_solve(const SedumiProb *P, int argc, char **argv, SedumiRes *R,
                  int (*run_sdp)(const BriskData *d, int argc, char **argv, BriskResult *res));
void sedumi_result_free(SedumiRes *R);
/* main.c: solve with the options argv[0..argc-1] (no program name); in the library build the
 * call is protected as brisk_run */
int  brisk_run_sedumi(const SedumiProb *P, int argc, char **argv, SedumiRes *R);
#endif
