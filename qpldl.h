/* qpldl.h (the QP developer, 15 November 2026): the QP interior-point method's column-by-column LDL' factorization for
 * very sparse factors (qpldl.c). */
#ifndef BRISK_QPLDL_H
#define BRISK_QPLDL_H
#include <stddef.h>

typedef struct QpLdl QpLdl;

/* The analysis of a symmetric matrix of order N in the order perm (perm[k]: the unknown at position k). Its entries off
 * the diagonal are given once each as pairs (er[e], ec[e]), e < ne, in the caller's numbering; every diagonal entry is
 * present. pos[e] and dpos[i] receive the places of entry e and of unknown i's diagonal in the value array (both may be
 * NULL: the places are then looked up by qp_ldl_offset). sgn[i]: the sign of unknown i's pivot (+1 or -1; copied).
 * NULL when memory is short. */
QpLdl  *qp_ldl_build(int N, const int *perm, size_t ne, const int *er, const int *ec, const signed char *sgn, size_t *pos, size_t *dpos);
size_t  qp_ldl_offset(const QpLdl *F, int r, int c);  /* the place of the entry (r, c) in the value array; (size_t)-1: no such entry */
double *qp_ldl_values(QpLdl *F);                    /* the matrix's values, filled by the caller before each factorization (kept by it) */
/* P K P' = L D L'. tinypiv = 0, a quasi-definite matrix: a pivot without its sign is replaced by sign * dynpiv when
 * dynpiv > 0 (the count in *nrep), otherwise 1 is returned. tinypiv > 0, a positive definite matrix: a pivot that is not
 * above tinypiv times its diagonal entry of the matrix is replaced by 1e128, which drops its direction (the count in
 * *nrep), the shared factorization's rule for the normal equations. */
int     qp_ldl_factor(QpLdl *F, double dynpiv, int *nrep, double tinypiv);
void    qp_ldl_solve(const QpLdl *F, double *b, double *work);   /* b := K^-1 b; work: N numbers */
/* the plan before the build: perm composed with the elimination tree's postorder (in place), the factor's flops (the sum
 * of the squared column lengths, the diagonal counted) and an upper bound on the updates between fundamental supernodes;
 * 1 when memory is short */
int     qp_ldl_plan(int N, int *perm, size_t ne, const int *er, const int *ec, double *flops, double *ubound, double *entries);      /* (entries: of the factor, the diagonal counted) */
double  qp_ldl_nupdates(const QpLdl *F);            /* the updates of one fundamental supernode by another */
double  qp_ldl_flops(const QpLdl *F);               /* the sum of the squared column lengths, the diagonal counted */
size_t  qp_ldl_nnz(const QpLdl *F);                 /* entries of L below the diagonal */
void    qp_ldl_free(QpLdl *F);
#endif
