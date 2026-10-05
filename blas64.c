/* blas64.c - BRISK on a BLAS/LAPACK with 64-bit integers (5.4).
 *
 * BRISK calls BLAS and LAPACK through the Fortran interface with 32-bit integers (brisk.h:
 * `const int *`). MATLAB's MKL (libmwblas, libmwlapack) and other ILP64 libraries take 64-bit
 * integers; inside MATLAB a MEX file that calls dgemm_ with 32-bit integers crashes on the first
 * call (the symbol resolves to MATLAB's library whatever the MEX file was linked with).
 *
 * This file is the boundary: compiled with -DBLAS_PREFIX=bk_ the solver calls bk_dgemm_ etc.,
 * defined here with the 32-bit interface; each widens its integer arguments, allocates 64-bit
 * copies of the integer arrays (pivots, work, ...) and calls the 64-bit routine, then copies
 * the integer results back. Nothing else in BRISK changes.
 *
 *   BLAS64_INT     the library's integer type (default ptrdiff_t: MATLAB, MKL ILP64, OpenBLAS64)
 *   BLAS64_PRE     a prefix of the library's symbols (default none; scipy_ for scipy_openblas64)
 *   BLAS64_SUF     a suffix after the trailing underscore (default none; 64_ for scipy_openblas64)
 *
 * MATLAB on Linux: build_brisk_mex('blas', 'matlab') compiles the solver with -DBLAS_PREFIX=bk_
 * and this file, and links -lmwblas -lmwlapack. (macOS uses Accelerate, 32-bit integers.)
 * The test of the C library against scipy_openblas64 is tools/blas64_test.sh. */
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#ifndef BLAS64_INT
#define BLAS64_INT ptrdiff_t
#endif
typedef BLAS64_INT bi;
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#ifndef BLAS64_PRE
#define BLAS64_PRE
#endif
#ifndef BLAS64_SUF
#define BLAS64_SUF
#endif
#define B64(name) CAT(CAT(BLAS64_PRE, name), BLAS64_SUF)
#ifndef BLAS_PREFIX
#define BLAS_PREFIX bk_
#endif
#define BW(name) CAT(BLAS_PREFIX, name)

/* ---- the 64-bit routines ---- */
void   B64(dgemm_)(const char *, const char *, const bi *, const bi *, const bi *, const double *, const double *, const bi *, const double *, const bi *, const double *, double *, const bi *);
void   B64(dsymm_)(const char *, const char *, const bi *, const bi *, const double *, const double *, const bi *, const double *, const bi *, const double *, double *, const bi *);
void   B64(dsyrk_)(const char *, const char *, const bi *, const bi *, const double *, const double *, const bi *, const double *, double *, const bi *);
void   B64(dgemv_)(const char *, const bi *, const bi *, const double *, const double *, const bi *, const double *, const bi *, const double *, double *, const bi *);
void   B64(dsymv_)(const char *, const bi *, const double *, const double *, const bi *, const double *, const bi *, const double *, double *, const bi *);
void   B64(dtrsv_)(const char *, const char *, const char *, const bi *, const double *, const bi *, double *, const bi *);
void   B64(dtrmv_)(const char *, const char *, const char *, const bi *, const double *, const bi *, double *, const bi *);
double B64(ddot_)(const bi *, const double *, const bi *, const double *, const bi *);
void   B64(daxpy_)(const bi *, const double *, const double *, const bi *, double *, const bi *);
double B64(dnrm2_)(const bi *, const double *, const bi *);
void   B64(dtrsm_)(const char *, const char *, const char *, const char *, const bi *, const bi *, const double *, const double *, const bi *, double *, const bi *);
void   B64(dtrmm_)(const char *, const char *, const char *, const char *, const bi *, const bi *, const double *, const double *, const bi *, double *, const bi *);
void   B64(dpotrf_)(const char *, const bi *, double *, const bi *, bi *);
void   B64(dpotri_)(const char *, const bi *, double *, const bi *, bi *);
void   B64(dpotrs_)(const char *, const bi *, const bi *, const double *, const bi *, double *, const bi *, bi *);
void   B64(dsyev_)(const char *, const char *, const bi *, double *, const bi *, double *, double *, const bi *, bi *);
void   B64(dsyevd_)(const char *, const char *, const bi *, double *, const bi *, double *, double *, const bi *, bi *, const bi *, bi *);
void   B64(dsyevr_)(const char *, const char *, const char *, const bi *, double *, const bi *, const double *, const double *, const bi *, const bi *, const double *, bi *, double *, double *, const bi *, bi *, double *, const bi *, bi *, const bi *, bi *);
void   B64(dsterf_)(const bi *, double *, double *, bi *);
void   B64(dstev_)(const char *, const bi *, double *, double *, double *, const bi *, double *, bi *);
void   B64(dstedc_)(const char *, const bi *, double *, double *, double *, const bi *, double *, const bi *, bi *, const bi *, bi *);
void   B64(dstebz_)(const char *, const char *, const bi *, const double *, const double *, const bi *, const bi *, const double *, const double *, const double *, bi *, bi *, double *, bi *, bi *, double *, bi *, bi *);
void   B64(dstein_)(const bi *, const double *, const double *, const bi *, const double *, const bi *, const bi *, double *, const bi *, double *, bi *, bi *, bi *);
void   B64(dsytrd_)(const char *, const bi *, double *, const bi *, double *, double *, double *, double *, const bi *, bi *);
void   B64(dormtr_)(const char *, const char *, const char *, const bi *, const bi *, const double *, const bi *, const double *, double *, const bi *, double *, const bi *, bi *);
void   B64(dsytrf_)(const char *, const bi *, double *, const bi *, bi *, double *, const bi *, bi *);
void   B64(dsytrs_)(const char *, const bi *, const bi *, const double *, const bi *, const bi *, double *, const bi *, bi *);
void   B64(dsysv_)(const char *, const bi *, const bi *, double *, const bi *, bi *, double *, const bi *, double *, const bi *, bi *);
void   B64(dgelsy_)(const bi *, const bi *, const bi *, double *, const bi *, double *, const bi *, bi *, const double *, bi *, double *, const bi *, bi *);
void   B64(dgesdd_)(const char *, const bi *, const bi *, double *, const bi *, double *, double *, const bi *, double *, const bi *, double *, const bi *, bi *, bi *);
void   B64(sgemm_)(const char *, const char *, const bi *, const bi *, const bi *, const float *, const float *, const bi *, const float *, const bi *, const float *, float *, const bi *);
void   B64(ssyrk_)(const char *, const char *, const bi *, const bi *, const float *, const float *, const bi *, const float *, float *, const bi *);
void   B64(strsm_)(const char *, const char *, const char *, const char *, const bi *, const bi *, const float *, const float *, const bi *, float *, const bi *);
void   B64(spotrf_)(const char *, const bi *, float *, const bi *, bi *);
void   B64(spotrs_)(const char *, const bi *, const bi *, const float *, const bi *, float *, const bi *, bi *);
void   B64(ssyevd_)(const char *, const char *, const bi *, float *, const bi *, float *, float *, const bi *, bi *, const bi *, bi *);
void   B64(ssyevr_)(const char *, const char *, const char *, const bi *, float *, const bi *, const float *, const float *, const bi *, const bi *, const float *, bi *, float *, float *, const bi *, bi *, float *, const bi *, bi *, const bi *, bi *);
void   B64(ssytrd_)(const char *, const bi *, float *, const bi *, float *, float *, float *, float *, const bi *, bi *);
void   B64(sormtr_)(const char *, const char *, const char *, const bi *, const bi *, const float *, const bi *, const float *, float *, const bi *, float *, const bi *, bi *);
void   B64(sstedc_)(const char *, const bi *, float *, float *, float *, const bi *, float *, const bi *, bi *, const bi *, bi *);

/* ---- integer arrays: 64-bit copies on the stack for small ones, the heap otherwise ---- */
#define IBUF 64
static bi *iget(bi *buf, size_t n) { return n <= IBUF ? buf : (bi *)malloc(sizeof(bi) * (n ? n : 1)); }
static void iput(bi *buf, bi *p) { if (p != buf) free(p); }
static void i2b(bi *dst, const int *src, size_t n) { for (size_t i = 0; i < n; i++) dst[i] = src[i]; }
static void b2i(int *dst, const bi *src, size_t n) { for (size_t i = 0; i < n; i++) dst[i] = (int)src[i]; }
#define I(x) ((bi)*(x))
#define OUT(ptr, v) (*(ptr) = (int)(v))

/* ---- BLAS ---- */
void BW(dgemm_)(const char *ta, const char *tb, const int *m, const int *n, const int *k, const double *al, const double *a, const int *lda, const double *b, const int *ldb, const double *be, double *c, const int *ldc) {
    const bi M = I(m), N = I(n), K = I(k), LA = I(lda), LB = I(ldb), LC = I(ldc);
    B64(dgemm_)(ta, tb, &M, &N, &K, al, a, &LA, b, &LB, be, c, &LC);
}
void BW(dsymm_)(const char *s, const char *u, const int *m, const int *n, const double *al, const double *a, const int *lda, const double *b, const int *ldb, const double *be, double *c, const int *ldc) {
    const bi M = I(m), N = I(n), LA = I(lda), LB = I(ldb), LC = I(ldc);
    B64(dsymm_)(s, u, &M, &N, al, a, &LA, b, &LB, be, c, &LC);
}
void BW(dsyrk_)(const char *u, const char *t, const int *n, const int *k, const double *al, const double *a, const int *lda, const double *be, double *c, const int *ldc) {
    const bi N = I(n), K = I(k), LA = I(lda), LC = I(ldc);
    B64(dsyrk_)(u, t, &N, &K, al, a, &LA, be, c, &LC);
}
void BW(dgemv_)(const char *t, const int *m, const int *n, const double *al, const double *a, const int *lda, const double *x, const int *incx, const double *be, double *y, const int *incy) {
    const bi M = I(m), N = I(n), LA = I(lda), IX = I(incx), IY = I(incy);
    B64(dgemv_)(t, &M, &N, al, a, &LA, x, &IX, be, y, &IY);
}
void BW(dsymv_)(const char *u, const int *n, const double *al, const double *a, const int *lda, const double *x, const int *incx, const double *be, double *y, const int *incy) {
    const bi N = I(n), LA = I(lda), IX = I(incx), IY = I(incy);
    B64(dsymv_)(u, &N, al, a, &LA, x, &IX, be, y, &IY);
}
void BW(dtrsv_)(const char *u, const char *t, const char *d, const int *n, const double *a, const int *lda, double *x, const int *incx) {
    const bi N = I(n), LA = I(lda), IX = I(incx);
    B64(dtrsv_)(u, t, d, &N, a, &LA, x, &IX);
}
void BW(dtrmv_)(const char *u, const char *t, const char *d, const int *n, const double *a, const int *lda, double *x, const int *incx) {
    const bi N = I(n), LA = I(lda), IX = I(incx);
    B64(dtrmv_)(u, t, d, &N, a, &LA, x, &IX);
}
double BW(ddot_)(const int *n, const double *x, const int *incx, const double *y, const int *incy) {
    const bi N = I(n), IX = I(incx), IY = I(incy);
    return B64(ddot_)(&N, x, &IX, y, &IY);
}
void BW(daxpy_)(const int *n, const double *al, const double *x, const int *incx, double *y, const int *incy) {
    const bi N = I(n), IX = I(incx), IY = I(incy);
    B64(daxpy_)(&N, al, x, &IX, y, &IY);
}
double BW(dnrm2_)(const int *n, const double *x, const int *incx) {
    const bi N = I(n), IX = I(incx);
    return B64(dnrm2_)(&N, x, &IX);
}
void BW(dtrsm_)(const char *s, const char *u, const char *t, const char *d, const int *m, const int *n, const double *al, const double *a, const int *lda, double *b, const int *ldb) {
    const bi M = I(m), N = I(n), LA = I(lda), LB = I(ldb);
    B64(dtrsm_)(s, u, t, d, &M, &N, al, a, &LA, b, &LB);
}
void BW(dtrmm_)(const char *s, const char *u, const char *t, const char *d, const int *m, const int *n, const double *al, const double *a, const int *lda, double *b, const int *ldb) {
    const bi M = I(m), N = I(n), LA = I(lda), LB = I(ldb);
    B64(dtrmm_)(s, u, t, d, &M, &N, al, a, &LA, b, &LB);
}
void BW(sgemm_)(const char *ta, const char *tb, const int *m, const int *n, const int *k, const float *al, const float *a, const int *lda, const float *b, const int *ldb, const float *be, float *c, const int *ldc) {
    const bi M = I(m), N = I(n), K = I(k), LA = I(lda), LB = I(ldb), LC = I(ldc);
    B64(sgemm_)(ta, tb, &M, &N, &K, al, a, &LA, b, &LB, be, c, &LC);
}
void BW(ssyrk_)(const char *u, const char *t, const int *n, const int *k, const float *al, const float *a, const int *lda, const float *be, float *c, const int *ldc) {
    const bi N = I(n), K = I(k), LA = I(lda), LC = I(ldc);
    B64(ssyrk_)(u, t, &N, &K, al, a, &LA, be, c, &LC);
}
void BW(strsm_)(const char *s, const char *u, const char *t, const char *d, const int *m, const int *n, const float *al, const float *a, const int *lda, float *b, const int *ldb) {
    const bi M = I(m), N = I(n), LA = I(lda), LB = I(ldb);
    B64(strsm_)(s, u, t, d, &M, &N, al, a, &LA, b, &LB);
}

/* ---- LAPACK ---- */
void BW(dpotrf_)(const char *u, const int *n, double *a, const int *lda, int *info) {
    const bi N = I(n), LA = I(lda); bi inf = 0;
    B64(dpotrf_)(u, &N, a, &LA, &inf); OUT(info, inf);
}
void BW(dpotri_)(const char *u, const int *n, double *a, const int *lda, int *info) {
    const bi N = I(n), LA = I(lda); bi inf = 0;
    B64(dpotri_)(u, &N, a, &LA, &inf); OUT(info, inf);
}
void BW(dpotrs_)(const char *u, const int *n, const int *nrhs, const double *a, const int *lda, double *b, const int *ldb, int *info) {
    const bi N = I(n), R = I(nrhs), LA = I(lda), LB = I(ldb); bi inf = 0;
    B64(dpotrs_)(u, &N, &R, a, &LA, b, &LB, &inf); OUT(info, inf);
}
void BW(spotrf_)(const char *u, const int *n, float *a, const int *lda, int *info) {
    const bi N = I(n), LA = I(lda); bi inf = 0;
    B64(spotrf_)(u, &N, a, &LA, &inf); OUT(info, inf);
}
void BW(spotrs_)(const char *u, const int *n, const int *nrhs, const float *a, const int *lda, float *b, const int *ldb, int *info) {
    const bi N = I(n), R = I(nrhs), LA = I(lda), LB = I(ldb); bi inf = 0;
    B64(spotrs_)(u, &N, &R, a, &LA, b, &LB, &inf); OUT(info, inf);
}
void BW(dsyev_)(const char *jz, const char *u, const int *n, double *a, const int *lda, double *w, double *work, const int *lwork, int *info) {
    const bi N = I(n), LA = I(lda), LW = I(lwork); bi inf = 0;
    B64(dsyev_)(jz, u, &N, a, &LA, w, work, &LW, &inf); OUT(info, inf);
}
/* iwork of size liwork (a query, liwork = -1, returns the size in iwork[0]) */
void BW(dsyevd_)(const char *jz, const char *u, const int *n, double *a, const int *lda, double *w, double *work, const int *lwork, int *iwork, const int *liwork, int *info) {
    const bi N = I(n), LA = I(lda), LW = I(lwork), LI = I(liwork); bi inf = 0, ib[IBUF];
    const size_t ni = LI > 0 ? (size_t)LI : 1;
    bi *iw = iget(ib, ni);
    B64(dsyevd_)(jz, u, &N, a, &LA, w, work, &LW, iw, &LI, &inf);
    if (LI == -1) OUT(iwork, iw[0]);
    iput(ib, iw); OUT(info, inf);
}
void BW(ssyevd_)(const char *jz, const char *u, const int *n, float *a, const int *lda, float *w, float *work, const int *lwork, int *iwork, const int *liwork, int *info) {
    const bi N = I(n), LA = I(lda), LW = I(lwork), LI = I(liwork); bi inf = 0, ib[IBUF];
    const size_t ni = LI > 0 ? (size_t)LI : 1;
    bi *iw = iget(ib, ni);
    B64(ssyevd_)(jz, u, &N, a, &LA, w, work, &LW, iw, &LI, &inf);
    if (LI == -1) OUT(iwork, iw[0]);
    iput(ib, iw); OUT(info, inf);
}
/* m out, isuppz (2 max(1,m), allocated 2n by the callers) out, iwork (liwork; query returns the size) */
void BW(dsyevr_)(const char *jz, const char *rg, const char *u, const int *n, double *a, const int *lda, const double *vl, const double *vu, const int *il, const int *iu, const double *abstol, int *m, double *w, double *z, const int *ldz, int *isuppz, double *work, const int *lwork, int *iwork, const int *liwork, int *info) {
    const bi N = I(n), LA = I(lda), IL = I(il), IU = I(iu), LZ = I(ldz), LW = I(lwork), LI = I(liwork); bi M = 0, inf = 0, ib[IBUF], ib2[IBUF];
    const size_t ni = LI > 0 ? (size_t)LI : 1, ns = 2 * (size_t)(N > 0 ? N : 1);
    bi *iw = iget(ib, ni), *is = iget(ib2, ns);
    B64(dsyevr_)(jz, rg, u, &N, a, &LA, vl, vu, &IL, &IU, abstol, &M, w, z, &LZ, is, work, &LW, iw, &LI, &inf);
    OUT(m, M);
    if (LI == -1) OUT(iwork, iw[0]);
    else if (isuppz && *jz == 'V' && M > 0) b2i(isuppz, is, 2 * (size_t)M);
    iput(ib, iw); iput(ib2, is); OUT(info, inf);
}
void BW(ssyevr_)(const char *jz, const char *rg, const char *u, const int *n, float *a, const int *lda, const float *vl, const float *vu, const int *il, const int *iu, const float *abstol, int *m, float *w, float *z, const int *ldz, int *isuppz, float *work, const int *lwork, int *iwork, const int *liwork, int *info) {
    const bi N = I(n), LA = I(lda), IL = I(il), IU = I(iu), LZ = I(ldz), LW = I(lwork), LI = I(liwork); bi M = 0, inf = 0, ib[IBUF], ib2[IBUF];
    const size_t ni = LI > 0 ? (size_t)LI : 1, ns = 2 * (size_t)(N > 0 ? N : 1);
    bi *iw = iget(ib, ni), *is = iget(ib2, ns);
    B64(ssyevr_)(jz, rg, u, &N, a, &LA, vl, vu, &IL, &IU, abstol, &M, w, z, &LZ, is, work, &LW, iw, &LI, &inf);
    OUT(m, M);
    if (LI == -1) OUT(iwork, iw[0]);
    else if (isuppz && *jz == 'V' && M > 0) b2i(isuppz, is, 2 * (size_t)M);
    iput(ib, iw); iput(ib2, is); OUT(info, inf);
}
void BW(dsterf_)(const int *n, double *d, double *e, int *info) {
    const bi N = I(n); bi inf = 0;
    B64(dsterf_)(&N, d, e, &inf); OUT(info, inf);
}
void BW(dstev_)(const char *jz, const int *n, double *d, double *e, double *z, const int *ldz, double *work, int *info) {
    const bi N = I(n), LZ = I(ldz); bi inf = 0;
    B64(dstev_)(jz, &N, d, e, z, &LZ, work, &inf); OUT(info, inf);
}
void BW(dstedc_)(const char *c, const int *n, double *d, double *e, double *z, const int *ldz, double *work, const int *lwork, int *iwork, const int *liwork, int *info) {
    const bi N = I(n), LZ = I(ldz), LW = I(lwork), LI = I(liwork); bi inf = 0, ib[IBUF];
    const size_t ni = LI > 0 ? (size_t)LI : 1;
    bi *iw = iget(ib, ni);
    B64(dstedc_)(c, &N, d, e, z, &LZ, work, &LW, iw, &LI, &inf);
    if (LI == -1) OUT(iwork, iw[0]);
    iput(ib, iw); OUT(info, inf);
}
void BW(sstedc_)(const char *c, const int *n, float *d, float *e, float *z, const int *ldz, float *work, const int *lwork, int *iwork, const int *liwork, int *info) {
    const bi N = I(n), LZ = I(ldz), LW = I(lwork), LI = I(liwork); bi inf = 0, ib[IBUF];
    const size_t ni = LI > 0 ? (size_t)LI : 1;
    bi *iw = iget(ib, ni);
    B64(sstedc_)(c, &N, d, e, z, &LZ, work, &LW, iw, &LI, &inf);
    if (LI == -1) OUT(iwork, iw[0]);
    iput(ib, iw); OUT(info, inf);
}
/* m, nsplit out; iblock, isplit (n) out; iwork 3n */
void BW(dstebz_)(const char *rg, const char *order, const int *n, const double *vl, const double *vu, const int *il, const int *iu, const double *abstol, const double *d, const double *e, int *m, int *nsplit, double *w, int *iblock, int *isplit, double *work, int *iwork, int *info) {
    const bi N = I(n), IL = I(il), IU = I(iu); bi M = 0, NS = 0, inf = 0;
    const size_t nn = (size_t)(N > 0 ? N : 1);
    bi *ibl = (bi *)malloc(sizeof(bi) * nn), *isp = (bi *)malloc(sizeof(bi) * nn), *iw = (bi *)malloc(sizeof(bi) * 3 * nn);
    B64(dstebz_)(rg, order, &N, vl, vu, &IL, &IU, abstol, d, e, &M, &NS, w, ibl, isp, work, iw, &inf);
    OUT(m, M); OUT(nsplit, NS);
    b2i(iblock, ibl, (size_t)(M > 0 ? M : 0)); b2i(isplit, isp, (size_t)(NS > 0 ? NS : 0));
    free(ibl); free(isp); free(iw); OUT(info, inf);
    (void)iwork;
}
/* iblock, isplit (n) in; iwork n; ifail (m) out */
void BW(dstein_)(const int *n, const double *d, const double *e, const int *m, const double *w, const int *iblock, const int *isplit, double *z, const int *ldz, double *work, int *iwork, int *ifail, int *info) {
    const bi N = I(n), M = I(m), LZ = I(ldz); bi inf = 0;
    const size_t nn = (size_t)(N > 0 ? N : 1), mm = (size_t)(M > 0 ? M : 1);
    bi *ibl = (bi *)malloc(sizeof(bi) * nn), *isp = (bi *)malloc(sizeof(bi) * nn), *iw = (bi *)malloc(sizeof(bi) * nn), *ifl = (bi *)malloc(sizeof(bi) * mm);
    i2b(ibl, iblock, (size_t)N); i2b(isp, isplit, (size_t)N);
    B64(dstein_)(&N, d, e, &M, w, ibl, isp, z, &LZ, work, iw, ifl, &inf);
    b2i(ifail, ifl, (size_t)M);
    free(ibl); free(isp); free(iw); free(ifl); OUT(info, inf);
    (void)iwork;
}
void BW(dsytrd_)(const char *u, const int *n, double *a, const int *lda, double *d, double *e, double *tau, double *work, const int *lwork, int *info) {
    const bi N = I(n), LA = I(lda), LW = I(lwork); bi inf = 0;
    B64(dsytrd_)(u, &N, a, &LA, d, e, tau, work, &LW, &inf); OUT(info, inf);
}
void BW(ssytrd_)(const char *u, const int *n, float *a, const int *lda, float *d, float *e, float *tau, float *work, const int *lwork, int *info) {
    const bi N = I(n), LA = I(lda), LW = I(lwork); bi inf = 0;
    B64(ssytrd_)(u, &N, a, &LA, d, e, tau, work, &LW, &inf); OUT(info, inf);
}
void BW(dormtr_)(const char *s, const char *u, const char *t, const int *m, const int *n, const double *a, const int *lda, const double *tau, double *c, const int *ldc, double *work, const int *lwork, int *info) {
    const bi M = I(m), N = I(n), LA = I(lda), LC = I(ldc), LW = I(lwork); bi inf = 0;
    B64(dormtr_)(s, u, t, &M, &N, a, &LA, tau, c, &LC, work, &LW, &inf); OUT(info, inf);
}
void BW(sormtr_)(const char *s, const char *u, const char *t, const int *m, const int *n, const float *a, const int *lda, const float *tau, float *c, const int *ldc, float *work, const int *lwork, int *info) {
    const bi M = I(m), N = I(n), LA = I(lda), LC = I(ldc), LW = I(lwork); bi inf = 0;
    B64(sormtr_)(s, u, t, &M, &N, a, &LA, tau, c, &LC, work, &LW, &inf); OUT(info, inf);
}
/* ipiv (n) out (a workspace query leaves it) */
void BW(dsytrf_)(const char *u, const int *n, double *a, const int *lda, int *ipiv, double *work, const int *lwork, int *info) {
    const bi N = I(n), LA = I(lda), LW = I(lwork); bi inf = 0, ib[IBUF];
    const size_t nn = (size_t)(N > 0 ? N : 1);
    bi *ip = iget(ib, nn);
    B64(dsytrf_)(u, &N, a, &LA, ip, work, &LW, &inf);
    if (LW != -1) b2i(ipiv, ip, (size_t)N);
    iput(ib, ip); OUT(info, inf);
}
void BW(dsytrs_)(const char *u, const int *n, const int *nrhs, const double *a, const int *lda, const int *ipiv, double *b, const int *ldb, int *info) {
    const bi N = I(n), R = I(nrhs), LA = I(lda), LB = I(ldb); bi inf = 0, ib[IBUF];
    const size_t nn = (size_t)(N > 0 ? N : 1);
    bi *ip = iget(ib, nn); i2b(ip, ipiv, (size_t)N);
    B64(dsytrs_)(u, &N, &R, a, &LA, ip, b, &LB, &inf);
    iput(ib, ip); OUT(info, inf);
}
void BW(dsysv_)(const char *u, const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, double *b, const int *ldb, double *work, const int *lwork, int *info) {
    const bi N = I(n), R = I(nrhs), LA = I(lda), LB = I(ldb), LW = I(lwork); bi inf = 0, ib[IBUF];
    const size_t nn = (size_t)(N > 0 ? N : 1);
    bi *ip = iget(ib, nn);
    B64(dsysv_)(u, &N, &R, a, &LA, ip, b, &LB, work, &LW, &inf);
    if (LW != -1) b2i(ipiv, ip, (size_t)N);
    iput(ib, ip); OUT(info, inf);
}
/* jpvt (n) in/out, rank out */
void BW(dgelsy_)(const int *m, const int *n, const int *nrhs, double *a, const int *lda, double *b, const int *ldb, int *jpvt, const double *rcond, int *rank, double *work, const int *lwork, int *info) {
    const bi M = I(m), N = I(n), R = I(nrhs), LA = I(lda), LB = I(ldb), LW = I(lwork); bi RK = 0, inf = 0, ib[IBUF];
    const size_t nn = (size_t)(N > 0 ? N : 1);
    bi *jp = iget(ib, nn); i2b(jp, jpvt, (size_t)N);
    B64(dgelsy_)(&M, &N, &R, a, &LA, b, &LB, jp, rcond, &RK, work, &LW, &inf);
    if (LW != -1) b2i(jpvt, jp, (size_t)N);
    OUT(rank, RK); iput(ib, jp); OUT(info, inf);
}
/* iwork 8 min(m,n) */
void BW(dgesdd_)(const char *jz, const int *m, const int *n, double *a, const int *lda, double *s, double *u, const int *ldu, double *vt, const int *ldvt, double *work, const int *lwork, int *iwork, int *info) {
    const bi M = I(m), N = I(n), LA = I(lda), LU = I(ldu), LV = I(ldvt), LW = I(lwork); bi inf = 0;
    const bi mn = M < N ? M : N;
    bi *iw = (bi *)malloc(sizeof(bi) * (size_t)(8 * (mn > 0 ? mn : 1)));
    B64(dgesdd_)(jz, &M, &N, a, &LA, s, u, &LU, vt, &LV, work, &LW, iw, &inf);
    free(iw); OUT(info, inf);
    (void)iwork;
}
